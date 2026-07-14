/* Auto-generated from main.cpp
   Segment: section_before_OnlineResearcher
   Original lines: 3488-3496
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

inline const ModelDefaults& getModelDefaults() {
  static ModelDefaults instance = loadModelDefaultsFromConfig();
  return instance;
}

/* OnlineResearcher provides online material retrieval and summary assistance */
// 调用方式：在复杂问题处理中由运行时按需调用。
// 实现思路：封装抓取、过滤、聚合和提示增强流程。
// 注意事项：外部网络失败时应支持降级路径。
// 注意事项：结果需接受安全和可信度约束。
// 注意事项：该组件不应阻塞核心请求主链路。
