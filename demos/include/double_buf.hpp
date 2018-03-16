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
 *  double_buf.hpp
 *
 *  Description:
 *    Provides a double-buffer class.
 *
 **************************************************************************/

#pragma once

#include <utility>

enum class Buffer { USE_A, USE_B };

// Double-buffers any kind of value.
template <typename T>
class DoubleBuf {
 public:
  template <typename... U>
  DoubleBuf(U&&... vals)
      : buffer(Buffer::USE_A),
        m_a(std::forward<U>(vals)...),
        m_b(std::forward<U>(vals)...) {}

  void swap() {
    if (buffer == Buffer::USE_A) {
      buffer = Buffer::USE_B;
    } else {
      buffer = Buffer::USE_A;
    }
  }

  T& read() {
    if (buffer == Buffer::USE_A) {
      return m_a;
    } else {
      return m_b;
    }
  }

  T& write() {
    if (buffer == Buffer::USE_A) {
      return m_b;
    } else {
      return m_a;
    }
  }

 private:
  DoubleBuf() {}

  // Buffer::USE_A -> read a;  Buffer::USE_B -> read b;
  // Buffer::USE_A -> write b; Buffer::USE_B -> write a;
  Buffer buffer;
  T m_a;
  T m_b;
};
