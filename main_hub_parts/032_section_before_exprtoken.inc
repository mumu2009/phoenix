/* Auto-generated from main.cpp
   Segment: section_before_ExprToken
   Original lines: 3566-3602
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

/* relu calculates ReLU activation function */
// 调用方式：在图传播或 MLP 步骤中作为激活。
// 实现思路：负值归零，正值保持原样。
// 注意事项：在 0 点不可导，数值优化需留意。
static double relu(double x) { return x > 0 ? x : 0; }
// leaky_relu 计算 LeakyReLU 激活函数。
// 调用方式：用于避免 ReLU 死亡神经元问题。
// 实现思路：负半轴保留小斜率 0.01。
// 注意事项：斜率固定，若需可配置需在上层扩展。
static double leaky_relu(double x) { return x > 0 ? x : 0.01 * x; }
// sigmoid 计算 Sigmoid 映射。
// 调用方式：用于概率化输出或门控计算。
// 实现思路：按标准 1/(1+e^-x) 公式实现。
// 注意事项：大绝对值输入可能导致梯度饱和。
static double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
static constexpr double kPi = 3.14159265358979323846;
// gelu 计算 GELU 激活近似。
// 调用方式：在更平滑激活需求下替代 ReLU。
// 实现思路：采用 tanh 近似表达式。
// 注意事项：相比 ReLU 计算开销更高。
static double gelu(double x) {
  return 0.5 * x *
         (1 +
          std::tanh(std::sqrt(2.0 / kPi) * (x + 0.044715 * std::pow(x, 3))));
}
// identity_fn 返回输入值本身。
// 调用方式：在需要关闭激活时作为透传函数。
// 实现思路：直接返回参数不做变换。
// 注意事项：上游需自行保证数值稳定性。
static double identity_fn(double x) { return x; }

// ExprToken 表示表达式解析阶段的最小语法单元。
// 调用方式：tokenizeExpr/toRpn/evalRpn 在词法与语法阶段共享。
// 实现思路：统一描述类型、文本值、优先级和函数参数信息。
// 注意事项：同一 token 的字段有效性由 type 决定。
// 注意事项：Op/Func 类型会依赖 precedence/arity 字段。
// 注意事项：该结构仅承载解析信息，不直接执行计算。
