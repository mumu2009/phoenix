/* math_addon.cpp - Math addon: exact arithmetic + scientific evaluator.
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
   along with 079 Project.  If not, see <http://www.gnu.org/licenses/>.

   Evaluator design (two modes, no silent rounding):
    - EXACT mode: expressions built from integers and the exact operators
      (+ - * / % ^ ! gcd lcm abs min max floor ceil round trunc) evaluate
      with arbitrary-precision integers/rationals (math_exact.hpp).  Rational
      division keeps p/q reduced; display prefers exact terminating decimals,
      falls back to "p/q" form.  This is the "the model can always trust
      arithmetic" guarantee: 0.1+0.2 == 0.3, 1/3*3 == 1, 100! is exact.
    - FLOAT mode: any transcendental function, irrational constant (pi, e) or
      decimal literal switches the subtree to IEEE-754 doubles; the result is
      printed with up to 15 significant digits.
    - Statements: 'a=2; b=3; a^b' assigns variables and returns the last
      value.  Assignment never escapes the request (fresh scope per call).
    - Errors carry a character position for agent-friendly messages.
*/
#include "math_addon.hpp"

#include "math_exact.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace addon::builtins {

namespace {

using mathx::BigInt;
using mathx::Rational;

/* ------------------------------ evaluator ------------------------------ */

struct EvalError {
  std::string msg;
  size_t pos{0};
};

enum class ValKind { Exact, Float };

struct NumVal {
  ValKind kind{ValKind::Exact};
  Rational r;      /* exact value (kind == Exact) */
  double d{0.0};   /* float value (kind == Float) */

  static NumVal fromRational(const Rational &v) {
    NumVal out;
    out.kind = ValKind::Exact;
    out.r = v;
    return out;
  }
  static NumVal fromDouble(double v) {
    NumVal out;
    out.kind = ValKind::Float;
    out.d = v;
    return out;
  }
  double toDouble() const {
    return kind == ValKind::Exact ? r.toDouble() : d;
  }
};

struct Token {
  enum class T { Number, Ident, Op, LParen, RParen, Comma, Assign, Semi, End } type{T::End};
  std::string text;
  bool integer{false};    /* Number: plain integer literal -> exact */
  bool plainDecimal{false}; /* Number: "1.25" style (no exponent) -> exact rational */
  size_t pos{0};
};

struct Lexer {
  const std::string &src;
  size_t i{0};

  explicit Lexer(const std::string &s) : src(s) {}

  char peek(size_t off = 0) const {
    return i + off < src.size() ? src[i + off] : '\0';
  }

  Token next() {
    while (i < src.size() && std::isspace(static_cast<unsigned char>(src[i]))) ++i;
    Token t;
    t.pos = i;
    if (i >= src.size()) { t.type = Token::T::End; return t; }
    const char c = src[i];
    if (c == '(') { ++i; t.type = Token::T::LParen; return t; }
    if (c == ')') { ++i; t.type = Token::T::RParen; return t; }
    if (c == ',') { ++i; t.type = Token::T::Comma; return t; }
    if (c == '=') { ++i; t.type = Token::T::Assign; return t; }
    if (c == ';') { ++i; t.type = Token::T::Semi; return t; }
    if (c == '+' || c == '-' || c == '*' || c == '/' || c == '^' || c == '%' || c == '!') {
      ++i;
      t.type = Token::T::Op;
      t.text = std::string(1, c);
      return t;
    }
    if (std::isdigit(static_cast<unsigned char>(c)) || (c == '.' && std::isdigit(static_cast<unsigned char>(peek(1))))) {
      const size_t start = i;
      bool isInt = true;
      while (std::isdigit(static_cast<unsigned char>(peek()))) ++i;
      if (peek() == '.') { isInt = false; ++i; while (std::isdigit(static_cast<unsigned char>(peek()))) ++i; }
      const bool hasExp = (peek() == 'e' || peek() == 'E');
      if (hasExp) {
        ++i;
        if (peek() == '+' || peek() == '-') ++i;
        while (std::isdigit(static_cast<unsigned char>(peek()))) ++i;
      }
      t.type = Token::T::Number;
      t.text = src.substr(start, i - start);
      t.integer = isInt;
      t.plainDecimal = !isInt && !hasExp;
      return t;
    }
    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
      const size_t start = i;
      while (std::isalnum(static_cast<unsigned char>(peek())) || peek() == '_') ++i;
      t.type = Token::T::Ident;
      t.text = src.substr(start, i - start);
      return t;
    }
    throw EvalError{"unexpected character", i};
  }
};

struct Parser {
  Lexer lex;
  Token cur;
  std::unordered_map<std::string, NumVal> vars;

  explicit Parser(const std::string &src) : lex(src) { advance(); }

  void advance() { cur = lex.next(); }

  [[noreturn]] void fail(const std::string &msg) const { throw EvalError{msg, cur.pos}; }

  /* input := stmt (';' stmt)*  -> value of last statement */
  NumVal parseInput() {
    NumVal v;
    bool any = false;
    while (cur.type != Token::T::End) {
      v = parseStmt();
      any = true;
      if (cur.type == Token::T::Semi) advance();
      else if (cur.type != Token::T::End) fail("expected ';'");
    }
    if (!any) fail("empty expression");
    return v;
  }

  NumVal parseStmt() {
    /* assignment: ident '=' expr */
    if (cur.type == Token::T::Ident && lex.peek() == '=') {
      const std::string name = cur.text;
      advance(); /* ident */
      advance(); /* '=' */
      NumVal v = parseExpr();
      if (!isFuncName(name)) vars[name] = v;
      return v;
    }
    return parseExpr();
  }

  /* expr := term (('+'|'-') term)* */
  NumVal parseExpr() {
    NumVal lhs = parseTerm();
    while (cur.type == Token::T::Op && (cur.text == "+" || cur.text == "-")) {
      const std::string op = cur.text;
      advance();
      NumVal rhs = parseTerm();
      lhs = applyBinary(op, lhs, rhs);
    }
    return lhs;
  }

  /* term := unary (('*'|'/'|'%'|infix "mod") unary)* */
  NumVal parseTerm() {
    NumVal lhs = parseUnary();
    while (true) {
      std::string op;
      if (cur.type == Token::T::Op &&
          (cur.text == "*" || cur.text == "/" || cur.text == "%")) {
        op = cur.text;
      } else if (cur.type == Token::T::Ident && lowerCopy(cur.text) == "mod") {
        op = "%";
      } else {
        break;
      }
      advance();
      NumVal rhs = parseUnary();
      lhs = applyBinary(op, lhs, rhs);
    }
    return lhs;
  }

  /* unary := ('-'|'+') unary | pow */
  NumVal parseUnary() {
    if (cur.type == Token::T::Op && cur.text == "-") {
      advance();
      NumVal v = parseUnary();
      if (v.kind == ValKind::Exact) {
        v.r.p.neg = !v.r.p.neg;
        return v;
      }
      v.d = -v.d;
      return v;
    }
    if (cur.type == Token::T::Op && cur.text == "+") {
      advance();
      return parseUnary();
    }
    return parsePow();
  }

  /* pow := postfix ('^' unary)?   (right-associative) */
  NumVal parsePow() {
    NumVal base = parsePostfix();
    if (cur.type == Token::T::Op && cur.text == "^") {
      advance();
      NumVal exp = parseUnary();
      return applyPow(base, exp);
    }
    return base;
  }

  /* postfix := primary ('!')* */
  NumVal parsePostfix() {
    NumVal v = parsePrimary();
    while (cur.type == Token::T::Op && cur.text == "!") {
      advance();
      v = applyFactorial(v);
    }
    return v;
  }

  NumVal parsePrimary() {
    if (cur.type == Token::T::Number) {
      const Token t = cur;
      advance();
      if (t.integer) {
        return NumVal::fromRational(Rational::fromInt(BigInt::fromString(t.text)));
      }
      if (t.plainDecimal) {
        /* "1.25" -> exact 125/100 so 0.1+0.2 == 0.3 exactly */
        const size_t dot = t.text.find('.');
        const std::string ip = t.text.substr(0, dot);
        const std::string fp = t.text.substr(dot + 1);
        BigInt num = BigInt::fromString(ip + fp);
        BigInt den = BigInt::fromString("1" + std::string(fp.size(), '0'));
        return NumVal::fromRational(Rational::make(num, den));
      }
      try {
        double d = std::stod(t.text);
        if (!std::isfinite(d)) fail("number out of range");
        return NumVal::fromDouble(d);
      } catch (const EvalError &) { throw; }
      catch (...) { fail("invalid number"); }
    }
    if (cur.type == Token::T::Ident) {
      const std::string name = cur.text;
      advance();
      if (cur.type == Token::T::LParen) {
        return parseCall(name);
      }
      const std::string low = lowerCopy(name);
      if (low == "pi") return NumVal::fromDouble(3.14159265358979323846);
      if (low == "e") return NumVal::fromDouble(2.71828182845904523536);
      if (low == "tau") return NumVal::fromDouble(6.28318530717958647692);
      if (low == "phi") return NumVal::fromDouble(1.61803398874989484820);
      auto it = vars.find(name);
      if (it != vars.end()) return it->second;
      fail("unknown name '" + name + "'");
    }
    if (cur.type == Token::T::LParen) {
      advance();
      NumVal v = parseExpr();
      if (cur.type != Token::T::RParen) fail("expected ')'");
      advance();
      return v;
    }
    fail("expected number, name or '('");
  }

  NumVal parseCall(const std::string &name) {
    advance(); /* '(' */
    std::vector<NumVal> args;
    if (cur.type != Token::T::RParen) {
      args.push_back(parseExpr());
      while (cur.type == Token::T::Comma) {
        advance();
        args.push_back(parseExpr());
      }
    }
    if (cur.type != Token::T::RParen) fail("expected ')'");
    advance();
    return applyFunc(name, args);
  }

  /* --------------------------- operations --------------------------- */

  static std::string lowerCopy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return s;
  }

  static bool isFuncName(const std::string &n) {
    static const std::unordered_map<std::string, int> funcs = {
      {"sin",1},{"cos",1},{"tan",1},{"asin",1},{"acos",1},{"atan",1},
      {"sinh",1},{"cosh",1},{"tanh",1},{"asinh",1},{"acosh",1},{"atanh",1},
      {"exp",1},{"ln",1},{"log",1},{"log2",1},{"log10",1},{"sqrt",1},{"cbrt",1},
      {"abs",1},{"floor",1},{"ceil",1},{"round",1},{"trunc",1},{"sign",1},
      {"expm1",1},{"log1p",1},{"erf",1},{"gamma",1},
      {"pow",2},{"atan2",2},{"hypot",2},{"mod",2},{"fmod",2},{"gcd",2},{"lcm",2},
      {"min",-1},{"max",-1},{"sum",-1},{"avg",-1}
    };
    return funcs.count(lowerCopy(n)) > 0;
  }

  NumVal applyBinary(const std::string &op, const NumVal &a, const NumVal &b) {
    if (op == "%") return applyMod(a, b);
    if (a.kind == ValKind::Exact && b.kind == ValKind::Exact) {
      try {
        if (op == "+") return NumVal::fromRational(Rational::add(a.r, b.r));
        if (op == "-") return NumVal::fromRational(Rational::sub(a.r, b.r));
        if (op == "*") return NumVal::fromRational(Rational::mul(a.r, b.r));
        if (op == "/") return NumVal::fromRational(Rational::div(a.r, b.r));
      } catch (const std::runtime_error &e) { fail(e.what()); }
    }
    const double x = a.toDouble(), y = b.toDouble();
    if (op == "+") return NumVal::fromDouble(x + y);
    if (op == "-") return NumVal::fromDouble(x - y);
    if (op == "*") return NumVal::fromDouble(x * y);
    if (op == "/") {
      if (y == 0.0) fail("division by zero");
      return NumVal::fromDouble(x / y);
    }
    fail("unknown operator");
  }

  NumVal applyMod(const NumVal &a, const NumVal &b) {
    if (a.kind == ValKind::Exact && b.kind == ValKind::Exact) {
      if (b.r.isZero()) fail("modulo by zero");
      /* Python-style: result sign follows divisor; integers only (exact). */
      BigInt x, y;
      if (!a.r.isInteger() || !b.r.isInteger()) {
        /* fall through to float fmod below */
      } else {
        x = a.r.p; y = b.r.p;
        BigInt m = BigInt::mod(x, y);
        return NumVal::fromRational(Rational::fromInt(m));
      }
    }
    const double x = a.toDouble(), y = b.toDouble();
    if (y == 0.0) fail("modulo by zero");
    return NumVal::fromDouble(std::fmod(x, y));
  }

  NumVal applyPow(const NumVal &base, const NumVal &exp) {
    /* exact: rational base with integer exponent (|exp| <= 10000) */
    if (base.kind == ValKind::Exact && exp.kind == ValKind::Exact &&
        exp.r.isInteger()) {
      int64_t n = 0;
      if (exp.r.p.fitsInt64(n) && (n >= -10000 && n <= 10000)) {
        if (n == 0) return NumVal::fromRational(Rational::fromInt(BigInt::one()));
        if (base.r.isZero() && n < 0) fail("division by zero (0^negative)");
        const uint64_t m = static_cast<uint64_t>(n < 0 ? -n : n);
        if (n > 0) {
          return NumVal::fromRational(
              Rational::make(BigInt::pow(base.r.p, BigInt::fromInt64(m)),
                             BigInt::pow(base.r.q, BigInt::fromInt64(m))));
        }
        return NumVal::fromRational(
            Rational::make(BigInt::pow(base.r.q, BigInt::fromInt64(m)),
                           BigInt::pow(base.r.p, BigInt::fromInt64(m))));
      }
      fail("exponent too large for exact pow (cap |exp| <= 10000)");
    }
    const double x = base.toDouble(), y = exp.toDouble();
    const double v = std::pow(x, y);
    if (!std::isfinite(v)) fail("pow result is not finite");
    return NumVal::fromDouble(v);
  }

  NumVal applyFactorial(const NumVal &a) {
    if (a.kind == ValKind::Exact && a.r.isInteger() && !a.r.p.neg) {
      int64_t n = 0;
      if (a.r.p.fitsInt64(n)) {
        if (n > 5000) fail("factorial too large (cap 5000!)");
        return NumVal::fromRational(Rational::fromInt(BigInt::factorial(static_cast<uint64_t>(n))));
      }
      fail("factorial argument too large");
    }
    const double x = a.toDouble();
    if (x < 0.0 || x != std::floor(x)) fail("factorial requires a non-negative integer");
    if (x > 5000.0) fail("factorial too large (cap 5000!)");
    return NumVal::fromDouble(std::tgamma(x + 1.0));
  }

  NumVal applyFunc(const std::string &nameIn, const std::vector<NumVal> &args) {
    const std::string f = lowerCopy(nameIn);
    auto need = [&](size_t n) {
      if (args.size() != n) fail(f + " expects " + std::to_string(n) + " argument(s)");
    };
    auto d1 = [&](size_t idx = 0) { return args[idx].toDouble(); };

    /* exact-preserving 1-arg */
    if (f == "abs" && args.size() == 1 && args[0].kind == ValKind::Exact) {
      Rational r = args[0].r;
      if (r.p.neg) r.p.neg = false;
      return NumVal::fromRational(r);
    }
    if ((f == "floor" || f == "ceil" || f == "round" || f == "trunc") &&
        args.size() == 1 && args[0].kind == ValKind::Exact && args[0].r.isInteger()) {
      return args[0]; /* integral already */
    }
    if ((f == "floor" || f == "ceil") && args.size() == 1 && args[0].kind == ValKind::Exact) {
      BigInt q, r;
      BigInt::divmod(args[0].r.p, args[0].r.q, q, r);
      if (f == "floor" && r.neg) q = BigInt::sub(q, BigInt::one());
      if (f == "ceil" && !r.isZero() && !r.neg) q = BigInt::add(q, BigInt::one());
      return NumVal::fromRational(Rational::fromInt(q));
    }
    if (f == "gcd" && args.size() == 2 && args[0].kind == ValKind::Exact &&
        args[1].kind == ValKind::Exact && args[0].r.isInteger() && args[1].r.isInteger()) {
      return NumVal::fromRational(Rational::fromInt(BigInt::gcd(args[0].r.p, args[1].r.p)));
    }
    if (f == "lcm" && args.size() == 2 && args[0].kind == ValKind::Exact &&
        args[1].kind == ValKind::Exact && args[0].r.isInteger() && args[1].r.isInteger()) {
      BigInt g = BigInt::gcd(args[0].r.p, args[1].r.p);
      if (g.isZero()) return NumVal::fromRational(Rational::fromInt(BigInt::zero()));
      BigInt prod = BigInt::mul(args[0].r.p, args[1].r.p);
      BigInt q, r;
      BigInt::divmod(prod, g, q, r);
      return NumVal::fromRational(Rational::fromInt(q));
    }
    if ((f == "min" || f == "max") && !args.empty() && args.size() <= 64) {
      bool allExact = true;
      for (const auto &a : args) allExact &= (a.kind == ValKind::Exact);
      if (allExact) {
        NumVal best = args[0];
        for (size_t k = 1; k < args.size(); ++k) {
          const int c = BigInt::cmp(args[k].r.p, best.r.p) >= 0 ? 0 : -1;
          if ((f == "min" && BigInt::cmp(args[k].r.p, best.r.p) < 0) ||
              (f == "max" && BigInt::cmp(args[k].r.p, best.r.p) > 0)) best = args[k];
        }
        return best;
      }
    }

    /* n-ary exact sum/avg over exact values */
    if ((f == "sum" || f == "avg") && !args.empty() && args.size() <= 256) {
      bool allExact = true;
      for (const auto &a : args) allExact &= (a.kind == ValKind::Exact);
      if (allExact) {
        Rational acc = Rational::fromInt(BigInt::zero());
        for (const auto &a : args) acc = Rational::add(acc, a.r);
        if (f == "sum") return NumVal::fromRational(acc);
        return NumVal::fromRational(Rational::div(acc,
            Rational::fromInt(BigInt::fromInt64(static_cast<int64_t>(args.size())))));
      }
    }

    /* float path */
    if (f == "min" || f == "max") {
      need(2);
      const double a = d1(0), b = d1(1);
      return NumVal::fromDouble(f == "min" ? std::min(a, b) : std::max(a, b));
    }
    if (f == "sum" || f == "avg") {
      if (args.empty() || args.size() > 256) fail(f + " takes 1..256 arguments");
      double acc = 0.0;
      for (const auto &a : args) acc += a.toDouble();
      return NumVal::fromDouble(f == "avg" ? acc / static_cast<double>(args.size()) : acc);
    }

    /* 1-arg transcendental / rounding */
    if (args.size() == 1) {
      const double x = d1();
      double v = 0.0;
      bool ok = true;
      if (f == "sin") v = std::sin(x);
      else if (f == "cos") v = std::cos(x);
      else if (f == "tan") v = std::tan(x);
      else if (f == "asin") v = std::asin(x);
      else if (f == "acos") v = std::acos(x);
      else if (f == "atan") v = std::atan(x);
      else if (f == "sinh") v = std::sinh(x);
      else if (f == "cosh") v = std::cosh(x);
      else if (f == "tanh") v = std::tanh(x);
      else if (f == "asinh") v = std::asinh(x);
      else if (f == "acosh") v = std::acosh(x);
      else if (f == "atanh") v = std::atanh(x);
      else if (f == "exp") v = std::exp(x);
      else if (f == "ln" || f == "log") v = std::log(x);
      else if (f == "log2") v = std::log2(x);
      else if (f == "log10") v = std::log10(x);
      else if (f == "sqrt") v = std::sqrt(x);
      else if (f == "cbrt") v = std::cbrt(x);
      else if (f == "abs") v = std::fabs(x);
      else if (f == "floor") v = std::floor(x);
      else if (f == "ceil") v = std::ceil(x);
      else if (f == "round") v = std::round(x);
      else if (f == "trunc") v = std::trunc(x);
      else if (f == "sign") v = (x > 0.0) - (x < 0.0);
      else if (f == "expm1") v = std::expm1(x);
      else if (f == "log1p") v = std::log1p(x);
      else if (f == "erf") v = std::erf(x);
      else if (f == "gamma") v = std::tgamma(x);
      else ok = false;
      if (!ok) fail("unknown function '" + f + "'");
      if (!std::isfinite(v)) fail(f + " result is not finite (out of domain?)");
      return NumVal::fromDouble(v);
    }

    /* 2-arg float */
    if (args.size() == 2) {
      const double a = d1(0), b = d1(1);
      double v = 0.0;
      bool ok = true;
      if (f == "pow") v = std::pow(a, b);
      else if (f == "atan2") v = std::atan2(a, b);
      else if (f == "hypot") v = std::hypot(a, b);
      else if (f == "fmod") { if (b == 0.0) fail("fmod by zero"); v = std::fmod(a, b); }
      else if (f == "mod") { if (b == 0.0) fail("mod by zero"); v = std::fmod(std::fmod(a, b) + b, b); }
      else if (f == "gcd" || f == "lcm") {
        if (a != std::floor(a) || b != std::floor(b)) fail(f + " requires integers");
        const long long ia = static_cast<long long>(a), ib = static_cast<long long>(b);
        const auto ig = std::gcd(ia, ib);
        v = f == "gcd" ? static_cast<double>(ig)
                       : (ig == 0 ? 0.0 : static_cast<double>((ia / ig) * ib));
      }
      else ok = false;
      if (!ok) fail("unknown function '" + f + "'");
      if (!std::isfinite(v)) fail(f + " result is not finite");
      return NumVal::fromDouble(v);
    }

    fail("wrong argument count for '" + f + "'");
  }
};

/* --------------------------- formatting --------------------------- */

std::string floatToString(double v) {
  if (v == 0.0) return "0";
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.15g", v);
  return buf;
}

} /* namespace */

/* --------------------------- public API --------------------------- */

json evaluateMathExpression(const std::string &expr) {
  try {
    Parser p(expr);
    const NumVal v = p.parseInput();
    if (p.cur.type != Token::T::End) throw EvalError{"unexpected trailing input", p.cur.pos};
    json out;
    if (v.kind == ValKind::Exact) {
      std::string disp;
      v.r.toDisplay(disp, 4096);
      out = json{{"ok", true}, {"value", disp}, {"exact", true},
                 {"mode", "exact"}, {"expression", expr}};
      if (v.r.isInteger()) {
        int64_t iv = 0;
        if (v.r.p.fitsInt64(iv)) out["numeric"] = iv;
        else out["numeric"] = v.r.toDouble();
      } else {
        out["numeric"] = v.r.toDouble();
      }
      return out;
    }
    return json{{"ok", true}, {"value", floatToString(v.d)}, {"exact", false},
                {"mode", "float"}, {"expression", expr}, {"numeric", v.d}};
  } catch (const EvalError &e) {
    return json{{"ok", false}, {"error", e.msg}, {"position", e.pos}, {"expression", expr}};
  }
}

class MathAddon : public Addon {
public:
  explicit MathAddon(std::string name) : name_(std::move(name)) {}
  std::string name() const override { return name_; }
  std::string type() const override { return "math"; }

  AddonResult handle(const std::string &text, const json &payload) override {
    AddonResult res;
    std::string addonType = payload.value("__addonType", std::string());
    std::transform(addonType.begin(), addonType.end(), addonType.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    addonType.erase(std::remove_if(addonType.begin(), addonType.end(), ::isspace), addonType.end());
    if (!addonType.empty() && addonType != "math" && addonType != "calculator" &&
        addonType != "calc") return res;

    std::string expr = text;
    /* strip human prefixes */
    const std::vector<std::string> prefixes = {"math:", "calc:", "math=", "calc=", "计算:", "计算"};
    std::string low = expr;
    std::transform(low.begin(), low.end(), low.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    for (const auto &pfx : prefixes) {
      std::string plow = pfx;
      std::transform(plow.begin(), plow.end(), plow.begin(),
                     [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
      if (low.rfind(plow, 0) == 0) { expr = expr.substr(plow.size()); break; }
    }
    /* strip surrounding whitespace */
    auto st = expr.find_first_not_of(" \t\r\n");
    if (st == std::string::npos) return res;
    auto en = expr.find_last_not_of(" \t\r\n");
    expr = expr.substr(st, en - st + 1);
    if (expr.empty()) return res;

    const json out = evaluateMathExpression(expr);
    res.handled = true;
    res.meta = json{{"addon", "math"}, {"name", name_}, {"result", out}};
    if (out.value("ok", false)) {
      res.reply = out.value("value", std::string());
      if (res.reply.size() > 2000) res.reply.resize(2000);
    } else {
      res.reply = "[math error] " + out.value("error", std::string("?"));
      if (out.contains("position")) {
        res.reply += " (at char " + std::to_string(out["position"].get<size_t>()) + ")";
      }
    }
    return res;
  }

private:
  std::string name_;
};

std::shared_ptr<Addon> createMathAddon(const std::string &name) {
  const std::string addonName = name.empty() ? std::string("math") : name;
  return std::make_shared<MathAddon>(addonName);
}

} /* namespace addon::builtins */
