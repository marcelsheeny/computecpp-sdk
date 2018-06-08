/***************************************************************************
 *
 *  Copyright (C) 2018 Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Codeplay's ComputeCpp SDK
 *
 *  sim.hpp
 *
 *  Description:
 *    SYCL kernel for Game of Life demo.
 *
 **************************************************************************/

#pragma once

#include <iostream>

#include <CL/sycl.hpp>

#include <sycl_utils.hpp>

namespace sycl = cl::sycl;

#include <double_buf.hpp>

enum class CellState : cl::sycl::cl_uint {
  LIVE = 1,
  DEAD = 0,
};

struct GameGrid {
  /// The states of cells
  sycl::buffer<CellState, 2> cells;
  /// The "velocities" of cells
  sycl::buffer<sycl::float2, 2> vels;

  /// Image representing our game state
  sycl::buffer<sycl::cl_uchar4, 2> img;

  GameGrid(size_t width, size_t height)
      : cells(sycl::range<2>(width, height)),
        vels(sycl::range<2>(width, height)),

        // image is flipped since OpenGL expects column-major order
        img(sycl::range<2>(height, width)) {}
};

class GameOfLifeSim {
  /// Grid dimensions
  size_t m_width;
  size_t m_height;

  /// Double-buffers the game grid so that we can read and write in parallel
  DoubleBuf<GameGrid> m_game;

  /// Mouse clicks on the grid recorded since last frame
  std::vector<std::tuple<size_t, size_t, CellState>> m_clicks;

  sycl::queue m_q;

 public:
  GameOfLifeSim(size_t width, size_t height)
      : m_width(width),
        m_height(height),
        m_game(width, height),
        m_q(sycl::default_selector{}, sycl_exception_handler) {}

  /// Add a button press (cell spawn) to be processed
  void add_click(size_t x, size_t y, CellState state) {
    // Click processing is deferred until update
    m_clicks.emplace_back(x, y, state);
  }

  void step();

  /// Calls the provided function with image data
  template <typename Func>
  void with_img(Func&& func) {
    auto acc = m_game.read().img.get_access<sycl::access::mode::read>();
    func(acc.get_pointer());
  }

 private:
  /// Executes an update frame
  void internal_step() {
    using sycl::access::mode;
    using sycl::access::target;

    // Read mouse clicks since last frame and apply them to the game.
    {
      // Have to write into read-buffer rather than write-buffer, since it is
      // the read-buffer
      // that will be read by the kernel.
      auto acc = this->m_game.read().cells.get_access<mode::write>();

      while (!m_clicks.empty()) {
        auto press = m_clicks.back();
        m_clicks.pop_back();
        acc[sycl::id<2>(std::get<0>(press), std::get<1>(press))] =
            std::get<2>(press);
      }
    }

    this->m_q.submit([&](sycl::handler& cgh) {
      auto r = this->m_game.read().cells.get_access<mode::read>(cgh);
      auto rv = this->m_game.read().vels.get_access<mode::read>(cgh);
      auto w = this->m_game.write().cells.get_access<mode::discard_write>(cgh);
      auto wv = this->m_game.write().vels.get_access<mode::discard_write>(cgh);
      auto img = this->m_game.write().img.get_access<mode::discard_write>(cgh);

      // These dummy variables have to be made to avoid capturing 'this' in the
      // kernel
      auto width = this->m_width;
      auto height = this->m_height;

      cgh.parallel_for<class gameoflifesimkernel>(
          // Work on each cell in parallel
          cl::sycl::range<2>(width, height), [=](cl::sycl::item<2> item) {
            size_t x = item.get(0);
            size_t y = item.get(1);

            // lambda that process index
            auto process_index = [](int ind, int offset, int max_size) {
              return (ind + offset) % max_size;
            };

            bool live[8];
            int count = 0;
            for (int offset_j = 1; offset_j >= -1; offset_j--) {
              for (int offset_i = -1; offset_i <= 1; offset_i++) {
                if (offset_j != 0 || offset_i != 0) {
                  int x_ind = process_index(x, offset_i, width);
                  int y_ind = process_index(y, offset_j, height);
                  live[count++] =
                      r[sycl::id<2>(x_ind, y_ind)] == CellState::LIVE;
                }
              }
            }

            // Sets the "velocity" of a cell depending on which neighbour
            // cells are alive
            float vel_consts[8][2] = {
                {-0.7f, 0.7f}, {0.0f, 1.7f},   {0.7f, 0.7f},  {-1.0f, 0.0f},
                {1.0f, 0.0f},  {-0.7f, -0.7f}, {0.0f, -1.0f}, {0.7f, 0.7f}};

            auto vel = sycl::float2(0.0f, 0.0f);
            for (size_t i = 0; i < 8; i++) {
              vel += sycl::float2(vel_consts[i][0], vel_consts[i][1]) * live[i];
            }
            vel /= 8.0f;

            // Counts the alive neighbours
            size_t live_neighbours = 0;
            for (size_t i = 0; i < 8; i++) {
              live_neighbours += live[i];
            }

            // Advances the cell state according to Conway's rules
            CellState new_state;
            if (r[item] == CellState::LIVE) {
              if (live_neighbours < 2) {
                new_state = CellState::DEAD;
              } else if (live_neighbours < 4) {
                new_state = CellState::LIVE;
              } else {
                new_state = CellState::DEAD;
              }
            } else if (live_neighbours == 3) {
              new_state = CellState::LIVE;
            } else {
              new_state = CellState::DEAD;
            }
            w[item] = new_state;

            auto new_vel = (rv[item] + vel) / 2.0f;
            wv[item] = new_vel;

            // Increase values to have brighter colours
            new_vel = sycl::fabs(new_vel) * 5.0f + sycl::float2(0.2f, 0.2f);

            // Set image pixel to new colour decided by state and "velocity"
            img[sycl::id<2>(y, x)] = sycl::cl_uchar4(
                (float)((int)(new_state)) * new_vel.x() * 255.0f, 0,
                (float)((int)(new_state)) * new_vel.y() * 255.0f, 255);
          });
    });

    // Swap read-buffer with write-buffer
    this->m_game.swap();
  }
};
