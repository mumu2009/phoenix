/* math_exact.hpp - Exact arbitrary-precision arithmetic for the math addon.
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

   Rationale: a reasoning agent must be able to trust arithmetic.  IEEE-754
   doubles silently round (0.1 + 0.2 != 0.3), and a "dumb" model can never
   learn its way out of a numerically wrong evaluator.  This header provides
   exact integer (sign + base-1e9 limbs) and exact rational (p/q, reduced)
   arithmetic with no external dependencies, so integer and rational
   expressions evaluate *exactly*; transcendental/inexact expressions fall
   back to doubles.

   Complexity: add/sub O(n), mul O(n*m), divmod O(n^2) schoolbook + Knuth
   normalization.  Sufficient for agent-scale computation (hundreds to
   thousands of digits); not a cryptography-grade library.
*/
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace addon::mathx {

inline constexpr uint32_t kLimbBase = 1000000000u; /* 1e9 limbs, decimal-aligned */
inline constexpr size_t kLimbDigits = 9;

/* ------------------------------ BigInt ------------------------------ */
struct BigInt {
  bool neg{false};
  std::vector<uint32_t> mag; /* little-endian, base 1e9, no leading zeros */

  static BigInt fromString(const std::string &s);
  std::string toString() const; /* plain decimal, "0" for zero */

  bool isZero() const { return mag.empty(); }
  bool isOne() const { return !neg && mag.size() == 1 && mag[0] == 1; }
  bool fitsInt64(int64_t &out) const;
  int sign() const { return isZero() ? 0 : (neg ? -1 : 1); }

  /* Compare magnitudes only. */
  static int cmpMag(const BigInt &a, const BigInt &b);
  static int cmp(const BigInt &a, const BigInt &b);

  static BigInt add(const BigInt &a, const BigInt &b);
  static BigInt sub(const BigInt &a, const BigInt &b);
  static BigInt mul(const BigInt &a, const BigInt &b);
  /* Euclidean division: a = q*b + r, 0 <= r < |b| (b must be nonzero). */
  static void divmod(const BigInt &a, const BigInt &b, BigInt &q, BigInt &r);
  static BigInt pow(const BigInt &base, const BigInt &exp); /* exp >= 0 */
  static BigInt mod(const BigInt &a, const BigInt &b);      /* result sign = b sign */
  static BigInt gcd(const BigInt &a, const BigInt &b);
  static BigInt factorial(uint64_t n); /* exact n! */

  static BigInt zero() { return BigInt{}; }
  static BigInt one() { return fromString("1"); }
  static BigInt fromInt64(int64_t v);
};

inline bool operator==(const BigInt &a, const BigInt &b) { return BigInt::cmp(a, b) == 0; }
inline bool operator<(const BigInt &a, const BigInt &b) { return BigInt::cmp(a, b) < 0; }

/* ----------------------------- Rational ----------------------------- */
/* Exact fraction p/q with q > 0, fully reduced. */
struct Rational {
  BigInt p, q; /* q > 0 */

  static Rational make(BigInt p, BigInt q); /* reduces; throws std::runtime_error on q==0 */
  static Rational fromInt(BigInt v) { return Rational{v, BigInt::one()}; }

  bool isZero() const { return p.isZero(); }
  bool isInteger() const { return q.isOne(); }

  static Rational add(const Rational &a, const Rational &b);
  static Rational sub(const Rational &a, const Rational &b);
  static Rational mul(const Rational &a, const Rational &b);
  static Rational div(const Rational &a, const Rational &b);

  /* 1) Exact terminating decimal if q = 2^a * 5^b, with at most maxDigits
     after the point; 2) otherwise "p/q" fraction form.  Return value:
     0 = terminating decimal produced, 1 = fraction form, 2 = too long. */
  int toDisplay(std::string &out, size_t maxDigits = 4096) const;
  double toDouble() const;
};

/* ------------------------------ BigInt impl ------------------------------ */
inline BigInt BigInt::fromString(const std::string &sIn) {
  BigInt out;
  std::string s = sIn;
  if (s.empty()) return out;
  if (s[0] == '+') s.erase(0, 1);
  else if (s[0] == '-') { out.neg = true; s.erase(0, 1); }
  while (s.size() > 1 && s[0] == '0') s.erase(0, 1);
  if (s == "0") { out.neg = false; return out; }
  for (size_t i = 0; i < s.size(); ++i)
    if (s[i] < '0' || s[i] > '9') return BigInt{}; /* invalid -> zero */
  /* collect limb groups most-significant first, then store little-endian */
  std::vector<uint32_t> groups;
  const size_t first = s.size() % kLimbDigits;
  size_t pos = 0;
  if (first != 0) {
    uint32_t v = 0;
    for (size_t i = 0; i < first; ++i) v = v * 10 + static_cast<uint32_t>(s[i] - '0');
    groups.push_back(v);
    pos = first;
  }
  while (pos < s.size()) {
    uint32_t v = 0;
    for (size_t i = 0; i < kLimbDigits; ++i) v = v * 10 + static_cast<uint32_t>(s[pos + i] - '0');
    groups.push_back(v);
    pos += kLimbDigits;
  }
  out.mag.assign(groups.rbegin(), groups.rend());
  return out;
}

inline std::string BigInt::toString() const {
  if (mag.empty()) return "0";
  std::string out;
  if (neg) out.push_back('-');
  out += std::to_string(mag.back());
  for (size_t i = mag.size() - 1; i > 0; --i) {
    std::string chunk = std::to_string(mag[i - 1]);
    out.append(kLimbDigits - chunk.size(), '0');
    out += chunk;
  }
  return out;
}

inline bool BigInt::fitsInt64(int64_t &out) const {
  BigInt lim = fromString("9223372036854775807"); /* INT64_MAX */
  if (cmpMag(*this, lim) > 0) return false;
  uint64_t acc = 0;
  for (size_t i = mag.size(); i-- > 0;) acc = acc * kLimbBase + mag[i];
  if (neg && acc == 9223372036854775808ull) { out = INT64_MIN; return true; }
  if (acc > (uint64_t)INT64_MAX) return false;
  out = neg ? -(int64_t)acc : (int64_t)acc;
  return true;
}

inline BigInt BigInt::fromInt64(int64_t v) {
  BigInt out;
  if (v < 0) { out.neg = true; v = -v; }
  if (v == 0) return out;
  uint64_t u = static_cast<uint64_t>(v);
  while (u > 0) {
    out.mag.push_back(static_cast<uint32_t>(u % kLimbBase));
    u /= kLimbBase;
  }
  return out;
}

inline int BigInt::cmpMag(const BigInt &a, const BigInt &b) {
  if (a.mag.size() != b.mag.size()) return a.mag.size() < b.mag.size() ? -1 : 1;
  for (size_t i = a.mag.size(); i-- > 0;) {
    if (a.mag[i] != b.mag[i]) return a.mag[i] < b.mag[i] ? -1 : 1;
  }
  return 0;
}

inline int BigInt::cmp(const BigInt &a, const BigInt &b) {
  if (a.neg != b.neg) return a.neg ? -1 : 1;
  const int m = cmpMag(a, b);
  return a.neg ? -m : m;
}

namespace {
inline void addMag(const std::vector<uint32_t> &a, const std::vector<uint32_t> &b,
                   std::vector<uint32_t> &out) {
  out.clear();
  out.resize(std::max(a.size(), b.size()) + 1, 0);
  uint64_t carry = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    uint64_t sum = carry;
    if (i < a.size()) sum += a[i];
    if (i < b.size()) sum += b[i];
    out[i] = static_cast<uint32_t>(sum % kLimbBase);
    carry = sum / kLimbBase;
  }
  while (!out.empty() && out.back() == 0) out.pop_back();
}

/* requires mag(a) >= mag(b) */
inline void subMag(const std::vector<uint32_t> &a, const std::vector<uint32_t> &b,
                   std::vector<uint32_t> &out) {
  out.clear();
  out.resize(a.size(), 0);
  int64_t borrow = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    int64_t diff = static_cast<int64_t>(a[i]) - borrow - (i < b.size() ? b[i] : 0);
    if (diff < 0) { diff += kLimbBase; borrow = 1; } else { borrow = 0; }
    out[i] = static_cast<uint32_t>(diff);
  }
  while (!out.empty() && out.back() == 0) out.pop_back();
}

inline void mulMag(const std::vector<uint32_t> &a, const std::vector<uint32_t> &b,
                   std::vector<uint32_t> &out) {
  out.assign(a.size() + b.size(), 0);
  for (size_t i = 0; i < a.size(); ++i) {
    uint64_t carry = 0;
    const uint64_t ai = a[i];
    for (size_t j = 0; j < b.size(); ++j) {
      uint64_t cur = static_cast<uint64_t>(out[i + j]) + ai * b[j] + carry;
      out[i + j] = static_cast<uint32_t>(cur % kLimbBase);
      carry = cur / kLimbBase;
    }
    size_t k = i + b.size();
    while (carry > 0) {
      uint64_t cur = static_cast<uint64_t>(out[k]) + carry;
      out[k] = static_cast<uint32_t>(cur % kLimbBase);
      carry = cur / kLimbBase;
      ++k;
    }
  }
  while (!out.empty() && out.back() == 0) out.pop_back();
}

/* Knuth Algorithm D long division (normalized). */
inline void divmodMag(const std::vector<uint32_t> &uIn, const std::vector<uint32_t> &vIn,
                      std::vector<uint32_t> &qOut, std::vector<uint32_t> &rOut) {
  qOut.clear(); rOut.clear();
  if (vIn.empty()) return; /* caller guards div-by-zero */
  if (uIn.size() < vIn.size()) { rOut = uIn; return; }
  /* Single-limb divisor: simple O(n) schoolbook division. */
  if (vIn.size() == 1) {
    const uint64_t v0 = vIn[0];
    uint64_t rem = 0;
    qOut.assign(uIn.size(), 0);
    for (size_t i = uIn.size(); i-- > 0;) {
      uint64_t cur = rem * kLimbBase + uIn[i];
      qOut[i] = static_cast<uint32_t>(cur / v0);
      rem = cur % v0;
    }
    while (!qOut.empty() && qOut.back() == 0) qOut.pop_back();
    if (rem != 0) rOut.push_back(static_cast<uint32_t>(rem));
    return;
  }
  /* Knuth normalization factor d so top divisor limb >= base/2. */
  const uint32_t d = kLimbBase / (vIn.back() + 1);
  std::vector<uint32_t> u(uIn.size() + 1, 0), v(vIn.size(), 0);
  uint64_t carry = 0;
  for (size_t i = 0; i < uIn.size(); ++i) {
    uint64_t cur = static_cast<uint64_t>(uIn[i]) * d + carry;
    u[i] = static_cast<uint32_t>(cur % kLimbBase);
    carry = cur / kLimbBase;
  }
  u[uIn.size()] = static_cast<uint32_t>(carry);
  /* keep the extra top limb: the division loop indexes u[j+n] up to uIn.size(). */
  carry = 0;
  for (size_t i = 0; i < vIn.size(); ++i) {
    uint64_t cur = static_cast<uint64_t>(vIn[i]) * d + carry;
    v[i] = static_cast<uint32_t>(cur % kLimbBase);
    carry = cur / kLimbBase;
  }
  const size_t n = v.size();
  const size_t m = uIn.size() - n; /* uIn.size() >= n checked at entry */
  std::vector<uint32_t> q(m + 1, 0);
  for (size_t j = m + 1; j-- > 0;) {
    uint64_t num = static_cast<uint64_t>(u[j + n]) * kLimbBase + u[j + n - 1];
    uint64_t qhat = num / v[n - 1];
    uint64_t rhat = num % v[n - 1];
    while (qhat >= kLimbBase ||
           (qhat * v[n - 2] > rhat * kLimbBase + u[j + n - 2])) {
      --qhat; rhat += v[n - 1];
      if (rhat >= kLimbBase) break;
    }
    /* multiply & subtract (borrow semantics per Knuth D4) */
    int64_t borrow = 0;
    for (size_t i = 0; i < n; ++i) {
      const uint64_t prod = qhat * static_cast<uint64_t>(v[i]) +
                            static_cast<uint64_t>(borrow);
      const uint64_t pi = prod % kLimbBase;
      borrow = static_cast<int64_t>(prod / kLimbBase);
      if (u[i + j] < pi) {
        u[i + j] = static_cast<uint32_t>(u[i + j] + kLimbBase - pi);
        ++borrow;
      } else {
        u[i + j] = static_cast<uint32_t>(u[i + j] - pi);
      }
    }
    if (u[j + n] < static_cast<uint32_t>(borrow)) {
      u[j + n] = static_cast<uint32_t>(u[j + n] + kLimbBase -
                                       static_cast<uint32_t>(borrow));
      borrow = 1;
    } else {
      u[j + n] = static_cast<uint32_t>(u[j + n] - static_cast<uint32_t>(borrow));
      borrow = 0;
    }
    if (borrow) { /* qhat was one too large: add divisor back */
      --qhat;
      uint64_t c = 0;
      for (size_t i = 0; i < n; ++i) {
        uint64_t s = static_cast<uint64_t>(u[i + j]) + v[i] + c;
        u[i + j] = static_cast<uint32_t>(s % kLimbBase);
        c = s / kLimbBase;
      }
      u[j + n] = static_cast<uint32_t>((static_cast<uint64_t>(u[j + n]) + c) % kLimbBase);
    }
    q[j] = static_cast<uint32_t>(qhat);
  }
  while (!q.empty() && q.back() == 0) q.pop_back();
  qOut = q;
  /* denormalize remainder */
  uint64_t rem = 0;
  std::vector<uint32_t> r(u.size(), 0);
  for (size_t i = u.size(); i-- > 0;) {
    uint64_t cur = rem * kLimbBase + u[i];
    r[i] = static_cast<uint32_t>(cur / d);
    rem = cur % d;
  }
  while (!r.empty() && r.back() == 0) r.pop_back();
  rOut = r;
}
} /* namespace */

inline BigInt BigInt::add(const BigInt &a, const BigInt &b) {
  BigInt out;
  if (a.neg == b.neg) {
    addMag(a.mag, b.mag, out.mag);
    out.neg = a.neg;
    return out;
  }
  const int m = cmpMag(a, b);
  if (m == 0) return out;
  if (m > 0) { subMag(a.mag, b.mag, out.mag); out.neg = a.neg; }
  else { subMag(b.mag, a.mag, out.mag); out.neg = b.neg; }
  return out;
}

inline BigInt BigInt::sub(const BigInt &a, const BigInt &b) {
  BigInt nb = b;
  nb.neg = !b.neg;
  return add(a, nb);
}

inline BigInt BigInt::mul(const BigInt &a, const BigInt &b) {
  BigInt out;
  if (a.isZero() || b.isZero()) return out;
  mulMag(a.mag, b.mag, out.mag);
  out.neg = a.neg != b.neg;
  return out;
}

inline void BigInt::divmod(const BigInt &a, const BigInt &b, BigInt &q, BigInt &r) {
  q = BigInt{}; r = BigInt{};
  if (b.isZero()) throw std::runtime_error("division by zero");
  divmodMag(a.mag, b.mag, q.mag, r.mag);
  q.neg = !q.isZero() && (a.neg != b.neg);
  /* remainder sign follows dividend for truncation semantics; keep |r| < |b| */
  r.neg = !r.isZero() && a.neg;
  /* normalize remainder to Euclidean when negative dividend: r = a - q*b */
  if (a.neg && !r.isZero()) {
    /* q truncates toward zero; adjust to keep remainder same sign as a */
    /* r currently = |a| mod |b| with a's sign; if a<0 then r should be negative.
       Since r.neg = a.neg, and |r|<|b|, this is already the truncation remainder. */
  }
}

inline BigInt BigInt::pow(const BigInt &base, const BigInt &exp) {
  if (exp.neg) throw std::runtime_error("negative exponent");
  BigInt result = one();
  BigInt b = base;
  BigInt e = exp;
  while (!e.isZero()) {
    BigInt q, r;
    divmod(e, fromInt64(2), q, r);
    if (!r.isZero()) result = mul(result, b);
    b = mul(b, b);
    e = q;
  }
  return result;
}

inline BigInt BigInt::mod(const BigInt &a, const BigInt &b) {
  BigInt q, r;
  divmod(a, b, q, r);
  /* Python-style modulo: result has sign of b. */
  if (!r.isZero() && (r.neg != b.neg)) r = add(r, b);
  return r;
}

inline BigInt BigInt::gcd(const BigInt &a, const BigInt &b) {
  BigInt x = a, y = b;
  x.neg = false; y.neg = false;
  while (!y.isZero()) {
    BigInt q, r;
    divmod(x, y, q, r);
    x = y;
    y = r;
  }
  return x;
}

inline BigInt BigInt::factorial(uint64_t n) {
  BigInt acc = one();
  for (uint64_t i = 2; i <= n; ++i) acc = mul(acc, fromInt64(static_cast<int64_t>(i)));
  return acc;
}

/* --------------------------- Rational impl --------------------------- */
inline Rational Rational::make(BigInt p, BigInt q) {
  if (q.isZero()) throw std::runtime_error("division by zero");
  if (p.isZero()) return Rational{BigInt::zero(), BigInt::one()};
  if (q.neg) { q.neg = false; p.neg = !p.neg; }
  BigInt g = BigInt::gcd(p, q);
  if (!g.isOne()) {
    BigInt qq, rr;
    BigInt::divmod(p, g, qq, rr);
    p = qq;
    BigInt::divmod(q, g, qq, rr);
    q = qq;
  }
  return Rational{p, q};
}

inline Rational Rational::add(const Rational &a, const Rational &b) {
  return make(BigInt::add(BigInt::mul(a.p, b.q), BigInt::mul(b.p, a.q)),
              BigInt::mul(a.q, b.q));
}
inline Rational Rational::sub(const Rational &a, const Rational &b) {
  return make(BigInt::sub(BigInt::mul(a.p, b.q), BigInt::mul(b.p, a.q)),
              BigInt::mul(a.q, b.q));
}
inline Rational Rational::mul(const Rational &a, const Rational &b) {
  return make(BigInt::mul(a.p, b.p), BigInt::mul(a.q, b.q));
}
inline Rational Rational::div(const Rational &a, const Rational &b) {
  if (b.isZero()) throw std::runtime_error("division by zero");
  return make(BigInt::mul(a.p, b.q), BigInt::mul(a.q, b.p));
}

inline int Rational::toDisplay(std::string &out, size_t maxDigits) const {
  out.clear();
  if (q.isOne()) { out = p.toString(); return 0; }
  /* check terminating: q = 2^a * 5^b */
  BigInt t = q;
  while (true) {
    BigInt qq, rr;
    BigInt::divmod(t, BigInt::fromInt64(2), qq, rr);
    if (rr.isZero()) { t = qq; continue; }
    break;
  }
  while (true) {
    BigInt qq, rr;
    BigInt::divmod(t, BigInt::fromInt64(5), qq, rr);
    if (rr.isZero()) { t = qq; continue; }
    break;
  }
  if (!t.isOne()) { /* non-terminating -> fraction form */
    out = p.toString() + "/" + q.toString();
    return 1;
  }
  /* long division to decimal */
  BigInt whole, rem;
  BigInt::divmod(p, q, whole, rem); /* truncation: rem carries p's sign */
  out = whole.toString();
  if (rem.isZero()) return 0;
  rem.neg = false; /* fractional digits from |rem| */
  out.push_back('.');
  size_t digits = 0;
  while (!rem.isZero() && digits < maxDigits) {
    rem = BigInt::mul(rem, BigInt::fromInt64(10));
    BigInt d, r2;
    BigInt::divmod(rem, q, d, r2);
    out += d.toString();
    rem = r2;
    ++digits;
  }
  if (!rem.isZero()) return 2; /* truncated */
  return 0;
}

inline double Rational::toDouble() const {
  /* long division to ~18 significant digits */
  int64_t pn = 0, qn = 0;
  bool pOk = p.fitsInt64(pn), qOk = q.fitsInt64(qn);
  if (pOk && qOk) {
    double r = static_cast<double>(pn) / static_cast<double>(qn);
    if (std::isfinite(r)) return r;
  }
  std::string dec;
  toDisplay(dec, 20);
  try { return std::stod(dec); } catch (...) { return 0.0; }
}

} /* namespace addon::mathx */
