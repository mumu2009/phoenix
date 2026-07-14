/* Auto-generated from main.cpp
   Segment: section_before_NamespacedStore
   Original lines: 3354-3360
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

/* NamespacedStore adds namespace prefix to underlying storage */
// 调用方式：将同一底座按不同业务域隔离键空间。
// 实现思路：统一在 key 前拼接固定前缀后转发调用。
// 注意事项：命名空间为空会抛异常，调用方需预校验。
// 注意事项：前缀策略变更会影响历史数据可见性。
// 注意事项：该封装不改变底层一致性语义。
