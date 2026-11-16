/* test_math_addon_exact.cpp - Exact arithmetic guarantees for the math addon.
   Ship gate: the model must be able to TRUST arithmetic (0.1+0.2 == 0.3
   exactly, 1/3*3 == 1, big integers exact); floats only for transcendental
   expressions; clean errors with positions. */

#include <gtest/gtest.h>
#include "addons/math_addon.hpp"

using addon::builtins::evaluateMathExpression;
using addon::json;

namespace {
void expectValue(const std::string &expr, const std::string &value, bool exact = true) {
  json r = evaluateMathExpression(expr);
  ASSERT_TRUE(r.value("ok", false)) << expr << ": " << r.value("error", std::string());
  EXPECT_EQ(r.value("value", std::string()), value) << expr;
  if (exact) EXPECT_TRUE(r.value("exact", false)) << expr << " should be exact";
}
void expectFloat(const std::string &expr, double want, double tol = 1e-9) {
  json r = evaluateMathExpression(expr);
  ASSERT_TRUE(r.value("ok", false)) << expr << ": " << r.value("error", std::string());
  const double got = r.value("numeric", 0.0);
  const double scale = want > 1.0 ? want : 1.0;
  EXPECT_NEAR(got, want, tol * scale) << expr;
}
}  // namespace

TEST(MathAddonExact, TrustworthyArithmetic) {
  expectValue("2+2", "4");
  expectValue("0.1+0.2", "0.3");
  expectValue("1/3*3", "1");
  expectValue("1/3", "1/3");
  expectValue("2/6", "1/3");
  expectValue("7/2", "3.5");
  expectValue("-7/2", "-3.5");
  expectValue("1.25*4", "5");
  expectValue("10/4", "2.5");
  expectValue("0.25^2", "0.0625");
  expectValue("2^-2", "0.25");
}

TEST(MathAddonExact, BigIntegers) {
  expectValue("2^100", "1267650600228229401496703205376");
  expectValue("25!", "15511210043330985984000000");
  expectValue("99999999999999999999+1", "100000000000000000000");
  expectValue("123456789012345678901234567890*2", "246913578024691357802469135780");
  expectValue("1000000000000000000000/5", "200000000000000000000");
  expectValue("gcd(48,18)", "6");
  expectValue("lcm(4,6)", "12");
}

TEST(MathAddonExact, ModuloAndFloor) {
  expectValue("17 mod 5", "2");
  expectValue("-17 mod 5", "3");
  expectValue("17 mod -5", "-3");
  expectValue("10 mod 3", "1");
  expectValue("floor(7/2)", "3");
  expectValue("ceil(7/2)", "4");
  expectValue("floor(-7/2)", "-4");
  expectValue("ceil(-7/2)", "-3");
}

TEST(MathAddonExact, PrecedenceAndStatements) {
  expectValue("2+3*4", "14");
  expectValue("(2+3)*4", "20");
  expectValue("3^3^2", "19683");
  expectValue("a=2; b=3; a^b", "8");
  expectValue("--5", "5");
  expectValue("2*(-3)", "-6");
  expectValue("0^0", "1");
  expectValue("abs(-5)", "5");
  expectValue("min(3,1,2)", "1");
  expectValue("max(3,1,2)", "3");
  expectValue("sum(1,2,3,4)", "10");
  expectValue("avg(1,2,3,4)", "2.5");
}

TEST(MathAddonFloat, Transcendentals) {
  expectFloat("sin(pi/6)", 0.5);
  expectFloat("cos(pi)", -1.0);
  expectFloat("sqrt(2)", 1.4142135623730951);
  expectFloat("ln(e)", 1.0);
  expectFloat("log10(1000)", 3.0);
  expectFloat("2^0.5", 1.4142135623730951);
  expectFloat("atan2(1,1)", 0.7853981633974483);
  expectFloat("hypot(3,4)", 5.0);
  expectFloat("gamma(5)", 24.0);
  expectFloat("round(2.5)", 3.0);
  expectFloat("sign(-3)", -1.0);
}

TEST(MathAddonErrors, FailCleanly) {
  EXPECT_FALSE(evaluateMathExpression("1/0").value("ok", false));
  EXPECT_FALSE(evaluateMathExpression("sqrt(-1)").value("ok", false));
  EXPECT_FALSE(evaluateMathExpression("foo(1)").value("ok", false));
  EXPECT_FALSE(evaluateMathExpression("1+").value("ok", false));
  json r = evaluateMathExpression("1/0");
  EXPECT_TRUE(r.contains("position")) << "errors should carry a position";
}
