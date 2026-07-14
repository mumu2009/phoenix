/* Auto-generated from main.cpp
   Segment: section_before_ModelDefaults
   Original lines: 3458-3464
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

/* ModelDefaults records default hyperparameter set for model runtime */
// 调用方式：当请求未提供参数时作为回退配置。
// 实现思路：聚合采样、长度和权重等常用参数。
// 注意事项：默认值需与当前模型能力匹配。
// 注意事项：外部显式参数通常优先于该结构。
// 注意事项：调整需评估兼容性与稳定性影响。
