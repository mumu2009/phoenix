/* Auto-generated from main.cpp
   Segment: section_before_RuntimeBase
   Original lines: 3403-3414
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

/* ------------------ Core model ------------------ */

class RuntimeState;
struct RobotsDocument;

/* RuntimeBase defines runtime controller base capability set */
// 调用方式：具体运行时实现继承该基类并重写核心接口。
// 实现思路：抽象公共依赖、配置与通用工具函数。
// 注意事项：析构需保证派生类资源正确释放。
// 注意事项：并发成员访问需遵守线程模型约束。
// 注意事项：该类本身不直接承载完整业务流程。
