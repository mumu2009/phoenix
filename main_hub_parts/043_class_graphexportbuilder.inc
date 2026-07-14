/* Auto-generated from main.cpp
   Segment: class_GraphExportBuilder
   Original lines: 4727-4772
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

class GraphExportBuilder
/* GraphExportBuilder converts graph data to export format */
// 调用方式：提供文件路径或流对象后构建导出内容。
// 实现思路：按节点与边组织结构并附带属性信息。
// 注意事项：大规模导出建议分段写入降低峰值内存。
// 注意事项：缺失字段需有稳健默认值。
// 注意事项：该类不修改源图数据。
{
public:
  static json fromWindow(const MemeGraph::WindowInfo &windowInfo,
                         const std::vector<float> *activation = nullptr) {
    const auto &ids = windowInfo.ids;
    int n = (int)ids.size();
    json nodes = json::array();
    json edges = json::array();
    for (int i = 0; i < n; i++) {
      double ax = (activation && i < (int)activation->size() &&
                   std::isfinite((*activation)[i]))
                      ? (*activation)[i]
                      : 0.0;
      double ay = ((i * 9973) % 101) / 100.0;
      nodes.push_back({{"id", ids[i]},
                       {"x", ax},
                       {"y", ay},
                       {"z", 0.0},
                       {"value", ax},
                       {"attrs", json::object()}});
    }
    const auto &csr = windowInfo.csr;
    for (uint32_t r = 0; r < csr.nRows; r++) {
      uint32_t start = csr.rowPtr[r];
      uint32_t end = csr.rowPtr[r + 1];
      for (uint32_t k = start; k < end; k++) {
        uint32_t c = csr.colIdx[k];
        double w = (k < csr.values.size()) ? csr.values[k] : 0.0;
        if (r < ids.size() && c < ids.size()) {
          edges.push_back({{"from", ids[r]},
                           {"to", ids[c]},
                           {"weight", std::isfinite(w) ? w : 0.0},
                           {"dir", 1}});
        }
      }
    }
    return json{{"Nodes", nodes}, {"Edges", edges}};
  }
};
