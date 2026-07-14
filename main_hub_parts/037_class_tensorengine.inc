/* Auto-generated from main.cpp
   Segment: class_TensorEngine
   Original lines: 3996-4081
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

class TensorEngine
/* TensorEngine provides tensor operations and feature processing capabilities */
// 调用方式：由上层学习模块调用其矩阵/向量计算接口。
// 实现思路：封装常用线代、归一化与统计计算流程。
// 注意事项：输入维度不一致时应尽早回退或报错。
// 注意事项：性能路径应减少临时对象分配。
// 注意事项：并发使用需遵守实例线程安全约束。
{
public:
  std::vector<float> spmm(const CSRMatrix &csr, const std::vector<float> &x) {
    auto &logger = LoggerCXX::instance();
    if (logger.enabled())
      logger.log(LoggerCXX::Type::COMPUTE,
                 std::string("graph spmm rows=") + std::to_string(csr.nRows) +
                     " cols=" + std::to_string(csr.nCols) +
                     " nnz=" + std::to_string(csr.values.size()));
    std::vector<float> out(csr.nRows, 0.0f);
    for (uint32_t row = 0; row < csr.nRows; row++) {
      float acc = 0;
      for (uint32_t idx = csr.rowPtr[row]; idx < csr.rowPtr[row + 1]; idx++) {
        uint32_t col = csr.colIdx[idx];
        float w = csr.values[idx];
        float xv = (col < x.size()) ? x[col] : 0;
        if (!std::isfinite(w) || !std::isfinite(xv))
          continue;
        acc += w * xv;
      }
      out[row] = std::isfinite(acc) ? acc : 0;
    }
    return out;
  }

  std::vector<float>
  iteratePropagation(const CSRMatrix &csr, const std::vector<float> &seeds,
                     int steps, const std::function<double(double)> &actFn,
                     double decayK, double damp = 0.02) {
    auto &logger = LoggerCXX::instance();
    int64_t startMs = nowEpochMs();
    std::vector<float> state = seeds;
    std::vector<float> next(seeds.size());
    for (int s = 0; s < steps; s++) {
      next = spmm(csr, state);
      for (size_t i = 0; i < next.size(); i++) {
        double si = std::isfinite(state[i]) ? state[i] : 0;
        double ni = std::isfinite(next[i]) ? next[i] : 0;
        double raw = si + ni - (decayK * si * damp);
        double y = 0;
        try {
          y = actFn(raw);
        } catch (...) {
          y = 0;
        }
        state[i] = std::isfinite(y) ? (float)y : 0.0f;
      }
    }
    for (auto &v : state)
      if (!std::isfinite(v))
        v = 0;
    if (logger.enabled())
      logger.log(LoggerCXX::Type::COMPUTE,
                 std::string("graph iteratePropagation steps=") +
                     std::to_string(steps) +
                     " nodes=" + std::to_string(csr.nRows) +
                     " costMs=" + std::to_string(nowEpochMs() - startMs));
    return state;
  }

  void l2NormalizeRows(std::vector<float> &data, size_t nRows, size_t nCols) {
    if (nRows == 0 || nCols == 0)
      return;
    for (size_t row = 0; row < nRows; row++) {
      double acc = 0.0;
      size_t base = row * nCols;
      for (size_t col = 0; col < nCols; col++) {
        double v = data[base + col];
        acc += v * v;
      }
      double norm = std::sqrt(acc);
      if (norm <= 0)
        continue;
      for (size_t col = 0; col < nCols; col++) {
        data[base + col] = (float)(data[base + col] / norm);
      }
    }
  }
};
