/* Auto-generated from main.cpp
   Segment: struct_SharedMemorySlice
   Original lines: 1628-1649
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   079 Project is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with 079 Project.  If not, see <http://www.gnu.org/licenses/>. */

struct SharedMemorySlice {
  std::shared_ptr<std::vector<uint8_t>> owner;
  const uint8_t *ptr{nullptr};
  size_t bytes{0};

  static SharedMemorySlice
  fromVector(std::shared_ptr<std::vector<uint8_t>> data) {
    SharedMemorySlice s;
    s.owner = std::move(data);
    if (s.owner && !s.owner->empty()) {
      s.ptr = s.owner->data();
      s.bytes = s.owner->size();
    }
    return s;
  }

  // empty 判断当前切片是否为空。
  // 调用方式：在消费共享数据前先做空值短路。
  // 实现思路：检查指针为空或字节数为零。
  // 注意事项：返回 true 不代表 owner 一定为空。
  bool empty() const { return ptr == nullptr || bytes == 0; }
};
