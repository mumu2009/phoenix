/* Auto-generated from main.cpp
   Segment: section_before_SearchConfig
   Original lines: 3441-3447
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

/* SearchConfig describes online retrieval strategy and threshold parameters */
// 调用方式：由 API 或配置中心下发后驱动检索行为。
// 实现思路：集中管理超时、数量限制与过滤开关。
// 注意事项：参数组合会影响时延、召回和成本。
// 注意事项：变更后建议结合监控指标回归验证。
// 注意事项：默认值用于兼容历史请求行为。
