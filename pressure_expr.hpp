/* pressure_expr.hpp - User-defined mission pressure expressions
   Copyright (C) 2026 079 Project

   Evaluates a scalar expression in t (elapsed seconds) with common
   elementary functions.  Used by Mission::pressure().  Default asymptotic
   mode uses tanh(t/tau) which approaches maxPain but never reaches it for
   finite t - avoids saturating pressure that drives hallucination.
*/
#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace phoenix {
namespace mission {

struct PressureEvalResult {
  bool ok{false};
  double value{0.0};
  std::string error;
};

/** Evaluate a pressure expression.
    Variables: t (required), and any extras in `vars` (e.g. H, g, Pmax, tau).
    Supported: + - * / ^, parentheses, commas for multi-arg funcs.
    Functions: sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh
               exp log ln log10 sqrt abs floor ceil min max pow clamp hypot.
    Constants: pi e.
    On failure returns ok=false and a short error (never throws). */
inline PressureEvalResult evalPressureExpr(
    const std::string &expr,
    double t,
    const std::unordered_map<std::string, double> &vars = {}) {
  struct Parser {
    const std::string &s;
    size_t i{0};
    double tVal;
    std::unordered_map<std::string, double> env;
    std::string err;

    explicit Parser(const std::string &e, double tIn,
                    const std::unordered_map<std::string, double> &v)
        : s(e), tVal(tIn), env(v) {
      env["t"] = tIn;
      env["pi"] = 3.14159265358979323846;
      env["e"] = 2.71828182845904523536;
    }

    void skip() {
      while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    }
    bool fail(const std::string &m) {
      if (err.empty()) err = m;
      return false;
    }
    bool parse(double &out) {
      skip();
      if (!parseExpr(out)) return false;
      skip();
      if (i != s.size()) return fail("trailing junk at pos " + std::to_string(i));
      if (!std::isfinite(out)) return fail("non-finite result");
      return true;
    }
    bool parseExpr(double &out) {
      if (!parseTerm(out)) return false;
      for (;;) {
        skip();
        if (i < s.size() && (s[i] == '+' || s[i] == '-')) {
          const char op = s[i++];
          double rhs = 0.0;
          if (!parseTerm(rhs)) return false;
          out = (op == '+') ? out + rhs : out - rhs;
        } else break;
      }
      return true;
    }
    bool parseTerm(double &out) {
      if (!parsePower(out)) return false;
      for (;;) {
        skip();
        if (i < s.size() && (s[i] == '*' || s[i] == '/')) {
          const char op = s[i++];
          double rhs = 0.0;
          if (!parsePower(rhs)) return false;
          if (op == '*') out *= rhs;
          else {
            if (rhs == 0.0) return fail("division by zero");
            out /= rhs;
          }
        } else break;
      }
      return true;
    }
    bool parsePower(double &out) {
      if (!parseUnary(out)) return false;
      skip();
      if (i < s.size() && s[i] == '^') {
        ++i;
        double rhs = 0.0;
        /* right-associative */
        if (!parsePower(rhs)) return false;
        out = std::pow(out, rhs);
      }
      return true;
    }
    bool parseUnary(double &out) {
      skip();
      if (i < s.size() && (s[i] == '+' || s[i] == '-')) {
        const char op = s[i++];
        if (!parseUnary(out)) return false;
        if (op == '-') out = -out;
        return true;
      }
      return parsePrimary(out);
    }
    bool parsePrimary(double &out) {
      skip();
      if (i >= s.size()) return fail("unexpected end");
      if (s[i] == '(') {
        ++i;
        if (!parseExpr(out)) return false;
        skip();
        if (i >= s.size() || s[i] != ')') return fail("missing ')'");
        ++i;
        return true;
      }
      if (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '.') {
        size_t start = i;
        while (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '.' ||
                                s[i] == 'e' || s[i] == 'E')) {
          if ((s[i] == 'e' || s[i] == 'E') && i + 1 < s.size() &&
              (s[i + 1] == '+' || s[i + 1] == '-')) {
            i += 2;
            continue;
          }
          ++i;
        }
        try {
          out = std::stod(s.substr(start, i - start));
        } catch (...) {
          return fail("bad number");
        }
        return true;
      }
      if (std::isalpha(static_cast<unsigned char>(s[i])) || s[i] == '_') {
        size_t start = i;
        while (i < s.size() &&
               (std::isalnum(static_cast<unsigned char>(s[i])) || s[i] == '_'))
          ++i;
        const std::string name = s.substr(start, i - start);
        skip();
        if (i < s.size() && s[i] == '(') {
          ++i;
          std::vector<double> args;
          skip();
          if (i < s.size() && s[i] == ')') {
            ++i;
          } else {
            for (;;) {
              double a = 0.0;
              if (!parseExpr(a)) return false;
              args.push_back(a);
              skip();
              if (i < s.size() && s[i] == ',') {
                ++i;
                continue;
              }
              break;
            }
            skip();
            if (i >= s.size() || s[i] != ')') return fail("missing ')' after " + name);
            ++i;
          }
          return callFn(name, args, out);
        }
        auto it = env.find(name);
        if (it == env.end()) return fail("unknown id '" + name + "'");
        out = it->second;
        return true;
      }
      return fail("unexpected char");
    }
    bool callFn(const std::string &name, const std::vector<double> &a, double &out) {
      auto need = [&](size_t n) -> bool {
        if (a.size() != n) return fail(name + " arity");
        return true;
      };
      if (name == "sin") { if (!need(1)) return false; out = std::sin(a[0]); return true; }
      if (name == "cos") { if (!need(1)) return false; out = std::cos(a[0]); return true; }
      if (name == "tan") { if (!need(1)) return false; out = std::tan(a[0]); return true; }
      if (name == "asin") { if (!need(1)) return false; out = std::asin(a[0]); return true; }
      if (name == "acos") { if (!need(1)) return false; out = std::acos(a[0]); return true; }
      if (name == "atan") { if (!need(1)) return false; out = std::atan(a[0]); return true; }
      if (name == "sinh") { if (!need(1)) return false; out = std::sinh(a[0]); return true; }
      if (name == "cosh") { if (!need(1)) return false; out = std::cosh(a[0]); return true; }
      if (name == "tanh") { if (!need(1)) return false; out = std::tanh(a[0]); return true; }
      if (name == "asinh") { if (!need(1)) return false; out = std::asinh(a[0]); return true; }
      if (name == "acosh") { if (!need(1)) return false; out = std::acosh(a[0]); return true; }
      if (name == "atanh") { if (!need(1)) return false; out = std::atanh(a[0]); return true; }
      if (name == "exp") { if (!need(1)) return false; out = std::exp(a[0]); return true; }
      if (name == "log" || name == "ln") { if (!need(1)) return false; if (a[0] <= 0) return fail("log domain"); out = std::log(a[0]); return true; }
      if (name == "log10") { if (!need(1)) return false; if (a[0] <= 0) return fail("log10 domain"); out = std::log10(a[0]); return true; }
      if (name == "sqrt") { if (!need(1)) return false; if (a[0] < 0) return fail("sqrt domain"); out = std::sqrt(a[0]); return true; }
      if (name == "abs") { if (!need(1)) return false; out = std::fabs(a[0]); return true; }
      if (name == "floor") { if (!need(1)) return false; out = std::floor(a[0]); return true; }
      if (name == "ceil") { if (!need(1)) return false; out = std::ceil(a[0]); return true; }
      if (name == "min") { if (a.size() < 2) return fail("min arity"); out = *std::min_element(a.begin(), a.end()); return true; }
      if (name == "max") { if (a.size() < 2) return fail("max arity"); out = *std::max_element(a.begin(), a.end()); return true; }
      if (name == "pow") { if (!need(2)) return false; out = std::pow(a[0], a[1]); return true; }
      if (name == "hypot") { if (!need(2)) return false; out = std::hypot(a[0], a[1]); return true; }
      if (name == "clamp") {
        if (!need(3)) return false;
        out = std::max(a[1], std::min(a[2], a[0]));
        return true;
      }
      return fail("unknown function '" + name + "'");
    }
  };

  if (expr.empty()) {
    return {false, 0.0, "empty expression"};
  }
  Parser p(expr, t, vars);
  double v = 0.0;
  if (!p.parse(v)) return {false, 0.0, p.err.empty() ? "parse error" : p.err};
  return {true, v, {}};
}

}  // namespace mission
}  // namespace phoenix
