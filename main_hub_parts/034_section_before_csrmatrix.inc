/* Auto-generated from main.cpp
   Segment: section_before_CSRMatrix
   Original lines: 3619-3981
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

/* isIdentChar checks if character can be identifier component */
// 调用方式：表达式词法分析时识别变量名与函数名。
// 实现思路：允许字母数字、下划线与点号。
// 注意事项：仅覆盖 ASCII 规则，不做完整 Unicode 标识符支持。
static bool isIdentChar(char c) {
  return std::isalnum((unsigned char)c) || c == '_' || c == '.';
}

// 返回内置函数的参数个数。
// 调用方式：RPN 构建与求值阶段校验函数调用时使用。
// 实现思路：对特殊函数（min/max/pow）返回 2，其余默认 1。
// 注意事项：新增函数时需要同步更新该映射。
static int funcArity(const std::string &name) {
  if (name == "min" || name == "max" || name == "pow")
    return 2;
  return 1;
}

// 判断标识符是否为支持的函数名。
// 调用方式：词法分析遇到标识符时区分变量与函数。
// 实现思路：在静态哈希集合中做 O(1) 查询。
// 注意事项：名称比较区分大小写，输入前需自行规范化。
static bool isFuncName(const std::string &name) {
  static const std::unordered_set<std::string> funcs = {
      "sin",  "cos", "tan", "tanh", "exp", "log",
      "sqrt", "abs", "min", "max",  "pow"};
  return funcs.count(name) > 0;
}

// 将表达式字符串切分为词法 token 序列。
// 调用方式：compileExpression 的第一阶段。
// 实现思路：按数字、标识符、运算符和括号逐字符扫描。
// 注意事项：解析失败会设置 ok=false 并返回空/部分结果。
static std::vector<ExprToken> tokenizeExpr(const std::string &expr, bool &ok) {
  ok = true;
  std::vector<ExprToken> out;
  for (size_t i = 0; i < expr.size();) {
    char c = expr[i];
    if (std::isspace((unsigned char)c)) {
      i++;
      continue;
    }
    if (std::isdigit((unsigned char)c) || c == '.') {
      size_t j = i + 1;
      while (j < expr.size() &&
             (std::isdigit((unsigned char)expr[j]) || expr[j] == '.' ||
              expr[j] == 'e' || expr[j] == 'E' || expr[j] == '+' ||
              expr[j] == '-')) {
        if ((expr[j] == '+' || expr[j] == '-') &&
            !(expr[j - 1] == 'e' || expr[j - 1] == 'E'))
          break;
        j++;
      }
      double v = 0.0;
      try {
        v = std::stod(expr.substr(i, j - i));
      } catch (...) {
        ok = false;
        return {};
      }
      out.push_back(ExprToken{ExprToken::Type::Number, "", v});
      i = j;
      continue;
    }
    if (std::isalpha((unsigned char)c) || c == '_') {
      size_t j = i + 1;
      while (j < expr.size() && isIdentChar(expr[j]))
        j++;
      std::string name = expr.substr(i, j - i);
      if (name == "x") {
        out.push_back(ExprToken{ExprToken::Type::Var, name, 0.0});
      } else if (name == "pi") {
        out.push_back(ExprToken{ExprToken::Type::Number, "", kPi});
      } else if (name == "e") {
        out.push_back(ExprToken{ExprToken::Type::Number, "", std::exp(1.0)});
      } else if (isFuncName(name)) {
        out.push_back(ExprToken{ExprToken::Type::Func, name, 0.0, 0, false,
                                funcArity(name)});
      } else {
        ok = false;
        return {};
      }
      i = j;
      continue;
    }
    if (c == '(') {
      out.push_back(ExprToken{ExprToken::Type::LParen});
      i++;
      continue;
    }
    if (c == ')') {
      out.push_back(ExprToken{ExprToken::Type::RParen});
      i++;
      continue;
    }
    if (c == ',') {
      out.push_back(ExprToken{ExprToken::Type::Comma});
      i++;
      continue;
    }
    if (c == '+' || c == '-' || c == '*' || c == '/' || c == '^') {
      ExprToken tok;
      tok.type = ExprToken::Type::Op;
      tok.text = std::string(1, c);
      if (c == '^') {
        tok.precedence = 3;
        tok.rightAssoc = true;
      } else if (c == '*' || c == '/') {
        tok.precedence = 2;
      } else {
        tok.precedence = 1;
      }
      out.push_back(tok);
      i++;
      continue;
    }
    ok = false;
    return {};
  }
  return out;
}

static std::vector<ExprToken> toRpn(const std::vector<ExprToken> &tokens,
                                    bool &ok) {
  ok = true;
  std::vector<ExprToken> output;
  std::vector<ExprToken> ops;
  ExprToken prev;
  bool hasPrev = false;
  for (size_t i = 0; i < tokens.size(); i++) {
    auto tok = tokens[i];
    if (tok.type == ExprToken::Type::Op && tok.text == "-") {
      if (!hasPrev || prev.type == ExprToken::Type::Op ||
          prev.type == ExprToken::Type::LParen ||
          prev.type == ExprToken::Type::Comma) {
        tok.text = "neg";
        tok.precedence = 4;
        tok.rightAssoc = true;
        tok.arity = 1;
      }
    }
    if (tok.type == ExprToken::Type::Number ||
        tok.type == ExprToken::Type::Var) {
      output.push_back(tok);
    } else if (tok.type == ExprToken::Type::Func) {
      ops.push_back(tok);
    } else if (tok.type == ExprToken::Type::Comma) {
      while (!ops.empty() && ops.back().type != ExprToken::Type::LParen) {
        output.push_back(ops.back());
        ops.pop_back();
      }
      if (ops.empty()) {
        ok = false;
        return {};
      }
    } else if (tok.type == ExprToken::Type::Op) {
      while (!ops.empty()) {
        auto top = ops.back();
        if (top.type == ExprToken::Type::Op) {
          if ((tok.rightAssoc && tok.precedence < top.precedence) ||
              (!tok.rightAssoc && tok.precedence <= top.precedence)) {
            output.push_back(top);
            ops.pop_back();
            continue;
          }
        } else if (top.type == ExprToken::Type::Func) {
          output.push_back(top);
          ops.pop_back();
          continue;
        }
        break;
      }
      ops.push_back(tok);
    } else if (tok.type == ExprToken::Type::LParen) {
      ops.push_back(tok);
    } else if (tok.type == ExprToken::Type::RParen) {
      bool matched = false;
      while (!ops.empty()) {
        auto top = ops.back();
        ops.pop_back();
        if (top.type == ExprToken::Type::LParen) {
          matched = true;
          break;
        }
        output.push_back(top);
      }
      if (!matched) {
        ok = false;
        return {};
      }
      if (!ops.empty() && ops.back().type == ExprToken::Type::Func) {
        output.push_back(ops.back());
        ops.pop_back();
      }
    }
    prev = tok;
    hasPrev = true;
  }
  while (!ops.empty()) {
    if (ops.back().type == ExprToken::Type::LParen ||
        ops.back().type == ExprToken::Type::RParen) {
      ok = false;
      return {};
    }
    output.push_back(ops.back());
    ops.pop_back();
  }
  return output;
}

static double evalRpn(const std::vector<ExprToken> &rpn, double x, bool &ok) {
  ok = true;
  std::vector<double> st;
  st.reserve(rpn.size());
  for (const auto &tok : rpn) {
    if (tok.type == ExprToken::Type::Number) {
      st.push_back(tok.value);
    } else if (tok.type == ExprToken::Type::Var) {
      st.push_back(x);
    } else if (tok.type == ExprToken::Type::Op) {
      if (tok.text == "neg") {
        if (st.empty()) {
          ok = false;
          return 0.0;
        }
        double v = st.back();
        st.pop_back();
        st.push_back(-v);
      } else {
        if (st.size() < 2) {
          ok = false;
          return 0.0;
        }
        double b = st.back();
        st.pop_back();
        double a = st.back();
        st.pop_back();
        if (tok.text == "+")
          st.push_back(a + b);
        else if (tok.text == "-")
          st.push_back(a - b);
        else if (tok.text == "*")
          st.push_back(a * b);
        else if (tok.text == "/")
          st.push_back(b == 0.0 ? 0.0 : a / b);
        else if (tok.text == "^")
          st.push_back(std::pow(a, b));
        else {
          ok = false;
          return 0.0;
        }
      }
    } else if (tok.type == ExprToken::Type::Func) {
      int arity = tok.arity ? tok.arity : funcArity(tok.text);
      if ((int)st.size() < arity) {
        ok = false;
        return 0.0;
      }
      double b = 0.0, a = 0.0;
      if (arity >= 2) {
        b = st.back();
        st.pop_back();
        a = st.back();
        st.pop_back();
      } else {
        a = st.back();
        st.pop_back();
      }
      double v = 0.0;
      if (tok.text == "sin")
        v = std::sin(a);
      else if (tok.text == "cos")
        v = std::cos(a);
      else if (tok.text == "tan")
        v = std::tan(a);
      else if (tok.text == "tanh")
        v = std::tanh(a);
      else if (tok.text == "exp")
        v = std::exp(a);
      else if (tok.text == "log")
        v = a <= 0 ? 0.0 : std::log(a);
      else if (tok.text == "sqrt")
        v = a < 0 ? 0.0 : std::sqrt(a);
      else if (tok.text == "abs")
        v = std::abs(a);
      else if (tok.text == "min")
        v = std::min(a, b);
      else if (tok.text == "max")
        v = std::max(a, b);
      else if (tok.text == "pow")
        v = std::pow(a, b);
      else {
        ok = false;
        return 0.0;
      }
      st.push_back(v);
    }
  }
  if (st.size() != 1 || !std::isfinite(st.back())) {
    ok = false;
    return 0.0;
  }
  return st.back();
}

static std::shared_ptr<std::vector<ExprToken>>
compileExpression(const std::string &expr) {
  static std::mutex mu;
  static std::unordered_map<std::string,
                            std::shared_ptr<std::vector<ExprToken>>>
      cache;
  if (expr.empty())
    return nullptr;
  {
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(expr);
    if (it != cache.end())
      return it->second;
  }
  bool ok = true;
  auto tokens = tokenizeExpr(expr, ok);
  if (!ok)
    return nullptr;
  auto rpn = toRpn(tokens, ok);
  if (!ok)
    return nullptr;
  auto ptr = std::make_shared<std::vector<ExprToken>>(std::move(rpn));
  {
    std::lock_guard<std::mutex> lock(mu);
    cache[expr] = ptr;
  }
  return ptr;
}

static std::function<double(double)> activationFn(const ModelDefaults &params) {
  if (params.activationType == "identity" || params.activationType == "linear")
    return identity_fn;
  if (params.activationType == "tanh")
    return [](double x) { return std::tanh(x); };
  if (params.activationType == "sigmoid")
    return sigmoid;
  if (params.activationType == "leaky_relu")
    return leaky_relu;
  if (params.activationType == "gelu")
    return gelu;
  if (params.activationType == "elu")
    return [](double x) { return x >= 0 ? x : (std::exp(x) - 1); };
  if (params.activationType == "softplus")
    return [](double x) { return std::log(1 + std::exp(x)); };
  if (params.activationType == "custom" && !params.activationCustom.empty()) {
    auto rpn = compileExpression(params.activationCustom);
    if (rpn) {
      return [rpn](double x) {
        bool ok = true;
        double v = evalRpn(*rpn, x, ok);
        return ok ? v : 0.0;
      };
    }
  }
  return relu;
}

