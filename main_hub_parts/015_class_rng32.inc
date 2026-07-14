/* Auto-generated from main.cpp
   Segment: class_Rng32
   Original lines: 2753-2765
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

class Rng32 {
public:
  explicit Rng32(uint32_t seed) : x_(seed ? seed : 0x9e3779b9u) {}
  double next() {
    x_ ^= x_ << 13;
    x_ ^= x_ >> 17;
    x_ ^= x_ << 5;
    return (double)(x_ & 0xffffffffu) / (double)0xffffffffu;
  }

private:
  uint32_t x_{0x9e3779b9u};
};
