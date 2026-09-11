/* security_core.cpp - Local GNN influence stats + MemeBarrier defense */

#include "security_core.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <sstream>


namespace phoenix {
namespace secamp {

namespace {

constexpr int kMaxNodes = 48;
constexpr double kAlpha = 0.85;

bool envTruthy(const char *name) {
  const char *v = std::getenv(name);
  if (!v || !*v)
    return false;
  return v[0] == '1' || v[0] == 't' || v[0] == 'T' || v[0] == 'y' ||
         v[0] == 'Y';
}

std::int64_t nowMs() {
  return static_cast<std::int64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

bool solveLinear(std::vector<std::vector<double>> a, std::vector<double> b,
                 std::vector<double> &x) {
  const int n = static_cast<int>(a.size());
  if (n == 0 || static_cast<int>(b.size()) != n)
    return false;
  x.assign(static_cast<size_t>(n), 0.0);
  for (int k = 0; k < n; ++k) {
    int piv = k;
    double best = std::fabs(a[static_cast<size_t>(k)][static_cast<size_t>(k)]);
    for (int i = k + 1; i < n; ++i) {
      double v = std::fabs(a[static_cast<size_t>(i)][static_cast<size_t>(k)]);
      if (v > best) {
        best = v;
        piv = i;
      }
    }
    if (best < 1e-14)
      return false;
    if (piv != k) {
      std::swap(a[static_cast<size_t>(k)], a[static_cast<size_t>(piv)]);
      std::swap(b[static_cast<size_t>(k)], b[static_cast<size_t>(piv)]);
    }
    const double akk = a[static_cast<size_t>(k)][static_cast<size_t>(k)];
    for (int i = k + 1; i < n; ++i) {
      const double f =
          a[static_cast<size_t>(i)][static_cast<size_t>(k)] / akk;
      for (int j = k; j < n; ++j)
        a[static_cast<size_t>(i)][static_cast<size_t>(j)] -=
            f * a[static_cast<size_t>(k)][static_cast<size_t>(j)];
      b[static_cast<size_t>(i)] -= f * b[static_cast<size_t>(k)];
    }
  }
  for (int i = n - 1; i >= 0; --i) {
    double s = b[static_cast<size_t>(i)];
    for (int j = i + 1; j < n; ++j)
      s -= a[static_cast<size_t>(i)][static_cast<size_t>(j)] *
           x[static_cast<size_t>(j)];
    const double aii = a[static_cast<size_t>(i)][static_cast<size_t>(i)];
    if (std::fabs(aii) < 1e-14)
      return false;
    x[static_cast<size_t>(i)] = s / aii;
  }
  return true;
}

std::vector<std::vector<double>>
buildWalkMatrix(const DiscreteGraph &g, int n,
                const std::unordered_map<std::string, int> &idx,
                double alpha) {
  std::vector<double> deg(static_cast<size_t>(n), 0.0);
  std::vector<std::vector<double>> adj(static_cast<size_t>(n),
                                       std::vector<double>(static_cast<size_t>(n), 0.0));
  for (const auto &e : g.edges) {
    if (e.from < 0 || e.to < 0 || e.from >= n || e.to >= n)
      continue;
    const double w = e.weight > 0.0 ? e.weight : 0.0;
    adj[static_cast<size_t>(e.from)][static_cast<size_t>(e.to)] += w;
    adj[static_cast<size_t>(e.to)][static_cast<size_t>(e.from)] += w;
  }
  (void)idx;
  for (int i = 0; i < n; ++i) {
    double s = 0.0;
    for (int j = 0; j < n; ++j)
      s += adj[static_cast<size_t>(i)][static_cast<size_t>(j)];
    deg[static_cast<size_t>(i)] = s;
  }
  std::vector<std::vector<double>> m(static_cast<size_t>(n),
                                     std::vector<double>(static_cast<size_t>(n), 0.0));
  for (int i = 0; i < n; ++i) {
    m[static_cast<size_t>(i)][static_cast<size_t>(i)] = 1.0;
    const double d = deg[static_cast<size_t>(i)];
    if (d <= 1e-15)
      continue;
    for (int j = 0; j < n; ++j)
      m[static_cast<size_t>(i)][static_cast<size_t>(j)] -=
          alpha * adj[static_cast<size_t>(i)][static_cast<size_t>(j)] / d;
  }
  return m;
}

std::vector<std::vector<double>> invertOrFail(const std::vector<std::vector<double>> &m,
                                              bool &ok) {
  const int n = static_cast<int>(m.size());
  std::vector<std::vector<double>> inv(static_cast<size_t>(n),
                                       std::vector<double>(static_cast<size_t>(n), 0.0));
  ok = true;
  for (int col = 0; col < n; ++col) {
    std::vector<double> b(static_cast<size_t>(n), 0.0);
    b[static_cast<size_t>(col)] = 1.0;
    std::vector<double> x;
    if (!solveLinear(m, b, x)) {
      ok = false;
      return inv;
    }
    for (int i = 0; i < n; ++i)
      inv[static_cast<size_t>(i)][static_cast<size_t>(col)] = x[static_cast<size_t>(i)];
  }
  return inv;
}

double vecNorm2(const std::vector<double> &v) {
  double s = 0.0;
  for (double x : v)
    s += x * x;
  return s;
}

std::vector<double> matVec(const std::vector<std::vector<double>> &m,
                           const std::vector<double> &v) {
  const int n = static_cast<int>(m.size());
  std::vector<double> out(static_cast<size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    double s = 0.0;
    for (int j = 0; j < n; ++j)
      s += m[static_cast<size_t>(i)][static_cast<size_t>(j)] * v[static_cast<size_t>(j)];
    out[static_cast<size_t>(i)] = s;
  }
  return out;
}

/* Laplacian Jacobi: smallest eigenvectors -> continuous coordinates */
void laplacianEmbed(const DiscreteGraph &g, int n, int dim,
                    std::vector<std::vector<double>> &coords) {
  std::vector<std::vector<double>> L(static_cast<size_t>(n),
                                     std::vector<double>(static_cast<size_t>(n), 0.0));
  for (const auto &e : g.edges) {
    if (e.from < 0 || e.to < 0 || e.from >= n || e.to >= n)
      continue;
    const double w = e.weight > 0.0 ? e.weight : 0.0;
    L[static_cast<size_t>(e.from)][static_cast<size_t>(e.to)] -= w;
    L[static_cast<size_t>(e.to)][static_cast<size_t>(e.from)] -= w;
    L[static_cast<size_t>(e.from)][static_cast<size_t>(e.from)] += w;
    L[static_cast<size_t>(e.to)][static_cast<size_t>(e.to)] += w;
  }
  coords.assign(static_cast<size_t>(n), std::vector<double>(static_cast<size_t>(dim), 0.0));
  if (n <= 1 || dim <= 0)
    return;

  std::vector<std::vector<double>> V = L;
  std::vector<double> d(static_cast<size_t>(n), 0.0);
  for (int sweep = 0; sweep < 32; ++sweep) {
    for (int p = 0; p < n; ++p) {
      for (int q = p + 1; q < n; ++q) {
        const double apq = V[static_cast<size_t>(p)][static_cast<size_t>(q)];
        if (std::fabs(apq) < 1e-12)
          continue;
        const double app = V[static_cast<size_t>(p)][static_cast<size_t>(p)];
        const double aqq = V[static_cast<size_t>(q)][static_cast<size_t>(q)];
        const double tau = (aqq - app) / (2.0 * apq);
        const double t = (tau >= 0.0 ? 1.0 : -1.0) /
                         (std::fabs(tau) + std::sqrt(1.0 + tau * tau));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = t * c;
        for (int k = 0; k < n; ++k) {
          const double vkp = V[static_cast<size_t>(k)][static_cast<size_t>(p)];
          const double vkq = V[static_cast<size_t>(k)][static_cast<size_t>(q)];
          V[static_cast<size_t>(k)][static_cast<size_t>(p)] = c * vkp - s * vkq;
          V[static_cast<size_t>(k)][static_cast<size_t>(q)] = s * vkp + c * vkq;
        }
        for (int k = 0; k < n; ++k) {
          const double vpk = V[static_cast<size_t>(p)][static_cast<size_t>(k)];
          const double vqk = V[static_cast<size_t>(q)][static_cast<size_t>(k)];
          V[static_cast<size_t>(p)][static_cast<size_t>(k)] = c * vpk - s * vqk;
          V[static_cast<size_t>(q)][static_cast<size_t>(k)] = s * vpk + c * vqk;
        }
      }
    }
  }
  for (int i = 0; i < n; ++i)
    d[static_cast<size_t>(i)] = V[static_cast<size_t>(i)][static_cast<size_t>(i)];
  std::vector<int> order(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i)
    order[static_cast<size_t>(i)] = i;
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return d[static_cast<size_t>(a)] < d[static_cast<size_t>(b)];
  });
  /* skip near-zero eigenvalue (connected component) */
  int start = 0;
  while (start < n && std::fabs(d[static_cast<size_t>(order[static_cast<size_t>(start)])]) < 1e-8)
    ++start;
  if (start >= n)
    start = 1;
  for (int k = 0; k < dim; ++k) {
    int col = start + k;
    if (col >= n)
      col = n - 1;
    /* recover eigenvector by inverse iteration on (L - λI) is heavy;
       use identity basis aligned to diagonalized V — after Jacobi,
       eigenvectors are implicit. Fall back: use Laplacian columns. */
    for (int i = 0; i < n; ++i)
      coords[static_cast<size_t>(i)][static_cast<size_t>(k)] =
          L[static_cast<size_t>(i)][static_cast<size_t>(order[static_cast<size_t>(k)] % n)];
  }
  for (int i = 0; i < n; ++i) {
    double ns = 0.0;
    for (int k = 0; k < dim; ++k)
      ns += coords[static_cast<size_t>(i)][static_cast<size_t>(k)] *
            coords[static_cast<size_t>(i)][static_cast<size_t>(k)];
    ns = std::sqrt(std::max(ns, 1e-18));
    for (int k = 0; k < dim; ++k)
      coords[static_cast<size_t>(i)][static_cast<size_t>(k)] /= ns;
  }
}

double rbf(const std::vector<double> &a, const std::vector<double> &b, double g) {
  double d2 = 0.0;
  const size_t m = std::min(a.size(), b.size());
  for (size_t i = 0; i < m; ++i) {
    const double d = a[i] - b[i];
    d2 += d * d;
  }
  return std::exp(-g * d2);
}

} // namespace

std::vector<std::string> tokenizeDefense(const std::string &text) {
  std::vector<std::string> out;
  std::string cur;
  for (unsigned char c : text) {
    if (std::isalnum(c) || static_cast<int>(c) >= 0x80) {
      cur.push_back(static_cast<char>(std::tolower(c)));
    } else if (!cur.empty()) {
      out.push_back(cur);
      cur.clear();
    }
  }
  if (!cur.empty())
    out.push_back(cur);
  return out;
}

InfluenceReport analyzeGraph(const DiscreteGraph &g, double alpha, int embedDim) {
  (void)alpha;
  InfluenceReport r;
  const int rawN = static_cast<int>(g.ids.size());
  if (rawN <= 0) {
    r.ok = true;
    return r;
  }
  const int n = std::min(rawN, kMaxNodes);
  std::unordered_map<std::string, int> idx;
  idx.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i)
    idx[g.ids[static_cast<size_t>(i)]] = i;

  DiscreteGraph clipped = g;
  clipped.ids.assign(g.ids.begin(), g.ids.begin() + n);
  if (static_cast<int>(clipped.layers.size()) < n)
    clipped.layers.resize(static_cast<size_t>(n));
  if (static_cast<int>(clipped.mapped.size()) < n)
    clipped.mapped.resize(static_cast<size_t>(n));
  std::vector<GraphEdge> kept;
  for (const auto &e : g.edges) {
    if (e.from >= 0 && e.to >= 0 && e.from < n && e.to < n)
      kept.push_back(e);
  }
  clipped.edges = std::move(kept);

  const auto sys = buildWalkMatrix(clipped, n, idx, alpha);
  bool invOk = false;
  const auto P = invertOrFail(sys, invOk);
  if (!invOk) {
    r.error = "singular_walk";
    return r;
  }

  const int dim = std::max(1, std::min(embedDim, std::max(1, n - 1)));
  std::vector<std::vector<double>> coords;
  laplacianEmbed(clipped, n, dim, coords);

  std::vector<double> seed(static_cast<size_t>(n), 1.0);
  for (int i = 0; i < n; ++i) {
    if (clipped.layers[static_cast<size_t>(i)] == "word")
      seed[static_cast<size_t>(i)] = 0.5;
  }
  const auto Pa = matVec(P, seed);

  /* F(a) = ||P a||^2 ; ∇F = 2 P^T P a ; H = 2 P^T P */
  std::vector<std::vector<double>> PtP(static_cast<size_t>(n),
                                       std::vector<double>(static_cast<size_t>(n), 0.0));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double s = 0.0;
      for (int k = 0; k < n; ++k)
        s += P[static_cast<size_t>(k)][static_cast<size_t>(i)] *
             P[static_cast<size_t>(k)][static_cast<size_t>(j)];
      PtP[static_cast<size_t>(i)][static_cast<size_t>(j)] = s;
    }
  }
  std::vector<double> grad(static_cast<size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    double s = 0.0;
    for (int j = 0; j < n; ++j)
      s += 2.0 * PtP[static_cast<size_t>(i)][static_cast<size_t>(j)] *
           seed[static_cast<size_t>(j)];
    grad[static_cast<size_t>(i)] = s;
  }

  const double gamma = 4.0;
  r.nodes.resize(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    std::vector<double> ei(static_cast<size_t>(n), 0.0);
    ei[static_cast<size_t>(i)] = 1.0;
    const auto Pei = matVec(P, ei);
    const double rag = vecNorm2(Pei);
    const double matrix = 0.0;
    double mat = 0.0;
    for (int j = 0; j < n; ++j)
      mat += ei[static_cast<size_t>(j)] *
             (matVec(PtP, ei)[static_cast<size_t>(j)]);
    (void)matrix;

    /* continuous field f(z)=Σ s_k exp(-γ||z-x_k||²) at node i */
    const auto &zi = coords[static_cast<size_t>(i)];
    std::vector<double> gvec(static_cast<size_t>(dim), 0.0);
    double htr = 0.0;
    for (int k = 0; k < n; ++k) {
      const double sk = seed[static_cast<size_t>(k)] * (0.25 + rag);
      const double w = rbf(zi, coords[static_cast<size_t>(k)], gamma) * sk;
      for (int d = 0; d < dim; ++d) {
        const double diff = zi[static_cast<size_t>(d)] -
                            coords[static_cast<size_t>(k)][static_cast<size_t>(d)];
        gvec[static_cast<size_t>(d)] += w * (-2.0 * gamma) * diff;
      }
      htr += w * (-2.0 * gamma * static_cast<double>(dim));
      double d2 = 0.0;
      for (int d = 0; d < dim; ++d) {
        const double diff = zi[static_cast<size_t>(d)] -
                            coords[static_cast<size_t>(k)][static_cast<size_t>(d)];
        d2 += diff * diff;
      }
      htr += w * 4.0 * gamma * gamma * d2;
    }
    double gnorm = 0.0;
    for (double v : gvec)
      gnorm += v * v;
    gnorm = std::sqrt(gnorm);

    NodeInfluence ni;
    ni.id = clipped.ids[static_cast<size_t>(i)];
    ni.layer = clipped.layers[static_cast<size_t>(i)].empty()
                   ? "meme"
                   : clipped.layers[static_cast<size_t>(i)];
    ni.ragImpact = rag;
    ni.matrixImpact = mat;
    ni.gradNorm = gnorm + std::fabs(grad[static_cast<size_t>(i)]);
    ni.hessTrace = htr + 2.0 * PtP[static_cast<size_t>(i)][static_cast<size_t>(i)];
    ni.significance = ni.ragImpact + 0.5 * ni.matrixImpact + 0.15 * ni.gradNorm +
                      0.05 * std::fabs(ni.hessTrace);
    ni.mappedIds = clipped.mapped[static_cast<size_t>(i)];
    for (const auto &e : clipped.edges) {
      if (e.from == i)
        ni.neighborIds.push_back(clipped.ids[static_cast<size_t>(e.to)]);
      else if (e.to == i)
        ni.neighborIds.push_back(clipped.ids[static_cast<size_t>(e.from)]);
    }
    std::ostringstream scope;
    scope << "rag=" << ni.ragImpact << ";matrix=" << ni.matrixImpact
          << ";neighbors=" << ni.neighborIds.size()
          << ";mapped=" << ni.mappedIds.size();
    ni.impactScope = scope.str();
    r.nodes[static_cast<size_t>(i)] = std::move(ni);
  }

  std::vector<int> order(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i)
    order[static_cast<size_t>(i)] = i;
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return r.nodes[static_cast<size_t>(a)].significance >
           r.nodes[static_cast<size_t>(b)].significance;
  });
  for (int rnk = 0; rnk < n; ++rnk) {
    r.nodes[static_cast<size_t>(order[static_cast<size_t>(rnk)])].rankMost = rnk + 1;
    r.nodes[static_cast<size_t>(order[static_cast<size_t>(n - 1 - rnk)])].rankLeast =
        rnk + 1;
  }
  const int top = std::min(3, n);
  for (int i = 0; i < top; ++i)
    r.mostSignificant.push_back(
        r.nodes[static_cast<size_t>(order[static_cast<size_t>(i)])].id);
  for (int i = 0; i < top; ++i)
    r.leastSignificant.push_back(
        r.nodes[static_cast<size_t>(order[static_cast<size_t>(n - 1 - i)])].id);

  r.ok = true;
  r.nodeCount = n;
  r.embedDim = dim;
  return r;
}

bool ragEnergyGradHess(const DiscreteGraph &g, const std::vector<double> &a,
                       double *energy, std::vector<double> *grad,
                       std::vector<double> *hessDiag) {
  const int n = static_cast<int>(g.ids.size());
  if (n <= 0 || static_cast<int>(a.size()) != n)
    return false;
  std::unordered_map<std::string, int> idx;
  for (int i = 0; i < n; ++i)
    idx[g.ids[static_cast<size_t>(i)]] = i;
  const auto sys = buildWalkMatrix(g, n, idx, kAlpha);
  bool invOk = false;
  const auto P = invertOrFail(sys, invOk);
  if (!invOk)
    return false;
  std::vector<std::vector<double>> PtP(static_cast<size_t>(n),
                                       std::vector<double>(static_cast<size_t>(n), 0.0));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double s = 0.0;
      for (int k = 0; k < n; ++k)
        s += P[static_cast<size_t>(k)][static_cast<size_t>(i)] *
             P[static_cast<size_t>(k)][static_cast<size_t>(j)];
      PtP[static_cast<size_t>(i)][static_cast<size_t>(j)] = s;
    }
  }
  const auto Pa = matVec(P, a);
  if (energy)
    *energy = vecNorm2(Pa);
  if (grad) {
    grad->assign(static_cast<size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
      double s = 0.0;
      for (int j = 0; j < n; ++j)
        s += 2.0 * PtP[static_cast<size_t>(i)][static_cast<size_t>(j)] *
             a[static_cast<size_t>(j)];
      (*grad)[static_cast<size_t>(i)] = s;
    }
  }
  if (hessDiag) {
    hessDiag->assign(static_cast<size_t>(n), 0.0);
    for (int i = 0; i < n; ++i)
      (*hessDiag)[static_cast<size_t>(i)] =
          2.0 * PtP[static_cast<size_t>(i)][static_cast<size_t>(i)];
  }
  return true;
}

InspectDecision inspectTokens(const std::vector<std::string> &tokens,
                              const InfluenceReport &report,
                              const DiscreteGraph &g,
                              const DefenseConfig &cfg) {
  InspectDecision d;
  if (!cfg.defenseEnabled || !cfg.isolateHighImpact || tokens.empty() ||
      !report.ok || report.nodes.empty())
    return d;

  double maxSig = 0.0;
  for (const auto &n : report.nodes)
    maxSig = std::max(maxSig, n.significance);
  const double cut = maxSig * cfg.highImpactQuantile;

  std::unordered_set<std::string> high;
  std::unordered_map<std::string, std::string> wordToMeme;
  for (const auto &n : report.nodes) {
    if (n.significance < cut)
      continue;
    high.insert(n.id);
    for (const auto &w : n.mappedIds) {
      high.insert(w);
      wordToMeme[w] = n.id;
    }
    std::string low = n.id;
    for (char &c : low)
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    high.insert(low);
  }

  for (const auto &t : tokens) {
    auto it = wordToMeme.find(t);
    const bool hitHigh = high.count(t) > 0;
    if (!hitHigh && it == wordToMeme.end())
      continue;
    const std::string meme = it != wordToMeme.end() ? it->second : t;
    const bool flagged = g.anomalous.count(meme) || g.isolated.count(meme) ||
                         g.anomalous.count(t) || g.isolated.count(t);
    if (!flagged)
      continue;
    d.hits.push_back(t);
  }
  if (d.hits.empty())
    return d;
  d.reason = "high-impact-anomalous-meme";
  if (cfg.researchObserve && cfg.allowResearchObserve) {
    d.observedOnly = true;
    d.blocked = false;
    return d;
  }
  d.blocked = true;
  return d;
}

SecurityObservatory &SecurityObservatory::instance() {
  static SecurityObservatory inst;
  return inst;
}

SecurityObservatory::SecurityObservatory() { loadProcessFlags(); }

void SecurityObservatory::loadProcessFlags() {
  std::lock_guard<std::mutex> lock(mu_);
  cfg_.allowResearchObserve = envTruthy("PHOENIX_SECURITY_ALLOW_RESEARCH_OBSERVE");
  cfg_.researchObserve = false;
  cfg_.defenseEnabled = !envTruthy("PHOENIX_SECURITY_DEFENSE_OFF");
  cfg_.isolateHighImpact = true;
  probe_ = ProbeState{};
  probe_.allowInertProbe = envTruthy("PHOENIX_SECURITY_ALLOW_INERT_PROBE");
}

void SecurityObservatory::resetForTests() {
  std::lock_guard<std::mutex> lock(mu_);
  report_ = InfluenceReport{};
  graph_ = DiscreteGraph{};
  alerts_.clear();
  cfg_ = DefenseConfig{};
  cfg_.allowResearchObserve = envTruthy("PHOENIX_SECURITY_ALLOW_RESEARCH_OBSERVE");
  probe_ = ProbeState{};
  probe_.allowInertProbe = envTruthy("PHOENIX_SECURITY_ALLOW_INERT_PROBE");
}

DefenseConfig SecurityObservatory::config() const {
  std::lock_guard<std::mutex> lock(mu_);
  return cfg_;
}

bool SecurityObservatory::setDefenseEnabled(bool on) {
  std::lock_guard<std::mutex> lock(mu_);
  cfg_.defenseEnabled = on;
  return true;
}

bool SecurityObservatory::setIsolateHighImpact(bool on) {
  std::lock_guard<std::mutex> lock(mu_);
  cfg_.isolateHighImpact = on;
  return true;
}

bool SecurityObservatory::setResearchObserve(bool on, std::string *error) {
  std::lock_guard<std::mutex> lock(mu_);
  if (on && !cfg_.allowResearchObserve) {
    if (error)
      *error = "research_observe_requires_PHOENIX_SECURITY_ALLOW_RESEARCH_OBSERVE";
    return false;
  }
  cfg_.researchObserve = on;
  return true;
}

bool SecurityObservatory::setProbeEnabled(bool on, std::string *error) {
  std::lock_guard<std::mutex> lock(mu_);
  if (on && !probe_.allowInertProbe) {
    if (error)
      *error = "inert_probe_requires_PHOENIX_SECURITY_ALLOW_INERT_PROBE";
    return false;
  }
  probe_.probeEnabled = on;
  return true;
}

bool SecurityObservatory::plantInertProbe(const std::string &seedId, std::string *error) {
  std::lock_guard<std::mutex> lock(mu_);
  if (!probe_.allowInertProbe || !probe_.probeEnabled) {
    if (error)
      *error = "inert_probe_disabled";
    return false;
  }
  if (seedId.empty()) {
    if (error)
      *error = "seed_required";
    return false;
  }
  bool known = false;
  for (const auto &id : graph_.ids) {
    if (id == seedId) {
      known = true;
      break;
    }
  }
  if (!known) {
    if (error)
      *error = "seed_not_in_local_graph";
    return false;
  }
  probe_.planted = true;
  probe_.seedId = seedId;
  probe_.activation.clear();
  probe_.traces.clear();
  probe_.activation[seedId] = 0;
  AlertItem a;
  a.ts = nowMs();
  a.kind = "inert-probe";
  a.detail = kInertProbeId;
  a.extra = json{{"phase", "plant"},
                 {"seed", seedId},
                 {"glyph", kInertProbeGlyph},
                 {"scope", "in-process-memegraph"}};
  alerts_.insert(alerts_.begin(), std::move(a));
  if (static_cast<int>(alerts_.size()) > cfg_.alertCap)
    alerts_.resize(static_cast<size_t>(cfg_.alertCap));
  return true;
}

bool SecurityObservatory::stepInertProbeOnce(std::string *error) {
  std::lock_guard<std::mutex> lock(mu_);
  if (!probe_.allowInertProbe || !probe_.probeEnabled || !probe_.planted) {
    if (error)
      *error = "inert_probe_disabled";
    return false;
  }
  std::unordered_map<std::string, int> idx;
  for (int i = 0; i < static_cast<int>(graph_.ids.size()); ++i)
    idx[graph_.ids[static_cast<size_t>(i)]] = i;

  std::vector<std::string> frontier;
  for (const auto &kv : probe_.activation)
    frontier.push_back(kv.first);

  bool grew = false;
  for (const auto &src : frontier) {
    auto it = idx.find(src);
    if (it == idx.end())
      continue;
    const int si = it->second;
    for (const auto &e : graph_.edges) {
      int other = -1;
      if (e.from == si)
        other = e.to;
      else if (e.to == si)
        other = e.from;
      if (other < 0 || other >= static_cast<int>(graph_.ids.size()))
        continue;
      const std::string &dst = graph_.ids[static_cast<size_t>(other)];
      if (probe_.activation.count(dst))
        continue;
      probe_.activation[dst] = 1;
      probe_.traces.push_back(ProbeHopTrace{src, dst, 1});
      grew = true;
    }
  }
  AlertItem a;
  a.ts = nowMs();
  a.kind = "inert-probe";
  a.detail = kInertProbeId;
  a.extra = json{{"phase", "step"},
                 {"hop", 1},
                 {"grew", grew},
                 {"scope", "in-process-memegraph"}};
  alerts_.insert(alerts_.begin(), std::move(a));
  if (static_cast<int>(alerts_.size()) > cfg_.alertCap)
    alerts_.resize(static_cast<size_t>(cfg_.alertCap));
  return true;
}

ProbeState SecurityObservatory::probeState() const {
  std::lock_guard<std::mutex> lock(mu_);
  return probe_;
}

json SecurityObservatory::probeJson() const {
  std::lock_guard<std::mutex> lock(mu_);
  json act = json::object();
  for (const auto &kv : probe_.activation)
    act[kv.first] = kv.second;
  json traces = json::array();
  for (const auto &t : probe_.traces)
    traces.push_back(json{{"from", t.from}, {"to", t.to}, {"hop", t.hop}});
  json recall = json::array();
  for (const auto &kv : probe_.activation) {
    json mapped = json::array();
    for (size_t i = 0; i < graph_.ids.size(); ++i) {
      if (graph_.ids[i] != kv.first)
        continue;
      if (i < graph_.mapped.size()) {
        for (const auto &w : graph_.mapped[i])
          mapped.push_back(w);
      }
    }
    recall.push_back(json{{"id", kv.first},
                          {"activation", kv.second},
                          {"marker", kInertProbeId},
                          {"mapped", mapped}});
  }
  return json{{"id", kInertProbeId},
              {"glyph", kInertProbeGlyph},
              {"inert", true},
              {"executable", false},
              {"instruction", false},
              {"humanTarget", false},
              {"crossProcess", false},
              {"crossSession", false},
              {"crossNetwork", false},
              {"scope", "in-process-memegraph"},
              {"allowInertProbe", probe_.allowInertProbe},
              {"probeEnabled", probe_.probeEnabled},
              {"planted", probe_.planted},
              {"seed", probe_.seedId},
              {"activation", act},
              {"traces", traces},
              {"recall", recall}};
}

InfluenceReport SecurityObservatory::ingest(const DiscreteGraph &g) {
  auto rep = analyzeGraph(g);
  std::lock_guard<std::mutex> lock(mu_);
  graph_ = g;
  report_ = rep;
  return report_;
}

InfluenceReport SecurityObservatory::lastReport() const {
  std::lock_guard<std::mutex> lock(mu_);
  return report_;
}

DiscreteGraph SecurityObservatory::lastGraph() const {
  std::lock_guard<std::mutex> lock(mu_);
  return graph_;
}

InspectDecision SecurityObservatory::inspectText(const std::string &text) {
  InfluenceReport rep;
  DiscreteGraph g;
  DefenseConfig cfg;
  bool probeVisible = false;
  {
    std::lock_guard<std::mutex> lock(mu_);
    rep = report_;
    g = graph_;
    cfg = cfg_;
    probeVisible = probe_.planted || probe_.probeEnabled;
  }
  InspectDecision dec;
  if (probeVisible && (text.find(kInertProbeId) != std::string::npos ||
                       text.find(kInertProbeGlyph) != std::string::npos)) {
    dec.hits.push_back(kInertProbeId);
    dec.reason = "inert-probe-marker";
    dec.blocked = cfg.defenseEnabled;
    dec.observedOnly = !dec.blocked;
  }
  auto tokenDec = inspectTokens(tokenizeDefense(text), rep, g, cfg);
  if (tokenDec.blocked)
    dec.blocked = true;
  if (!tokenDec.reason.empty() && dec.reason.empty())
    dec.reason = tokenDec.reason;
  for (const auto &h : tokenDec.hits)
    dec.hits.push_back(h);
  if (!dec.hits.empty()) {
    json extra{{"hits", dec.hits},
               {"blocked", dec.blocked},
               {"observedOnly", dec.observedOnly}};
    pushAlert(dec.blocked ? "block" : "observe", dec.reason, extra);
  }
  return dec;
}

void SecurityObservatory::noteIsolation(const std::string &memeId, double score,
                                        const std::string &reason) {
  std::lock_guard<std::mutex> lock(mu_);
  graph_.isolated.insert(memeId);
  graph_.anomalous.insert(memeId);
  json extra{{"memeId", memeId}, {"score", score}, {"reason", reason}};
  AlertItem a;
  a.ts = nowMs();
  a.kind = "isolate";
  a.detail = memeId;
  a.extra = extra;
  alerts_.insert(alerts_.begin(), std::move(a));
  if (static_cast<int>(alerts_.size()) > cfg_.alertCap)
    alerts_.resize(static_cast<size_t>(cfg_.alertCap));
}

void SecurityObservatory::pushAlert(const std::string &kind,
                                    const std::string &detail,
                                    const json &extra) {
  std::lock_guard<std::mutex> lock(mu_);
  AlertItem a;
  a.ts = nowMs();
  a.kind = kind;
  a.detail = detail;
  a.extra = extra;
  alerts_.insert(alerts_.begin(), std::move(a));
  if (static_cast<int>(alerts_.size()) > cfg_.alertCap)
    alerts_.resize(static_cast<size_t>(cfg_.alertCap));
}

json SecurityObservatory::statsJson() const {
  std::lock_guard<std::mutex> lock(mu_);
  json items = json::array();
  for (const auto &n : report_.nodes) {
    items.push_back(json{{"id", n.id},
                         {"layer", n.layer},
                         {"ragImpact", n.ragImpact},
                         {"matrixImpact", n.matrixImpact},
                         {"gradNorm", n.gradNorm},
                         {"hessTrace", n.hessTrace},
                         {"significance", n.significance},
                         {"rankMost", n.rankMost}});
  }
  return json{{"ok", report_.ok},
              {"nodeCount", report_.nodeCount},
              {"embedDim", report_.embedDim},
              {"mostSignificant", report_.mostSignificant},
              {"leastSignificant", report_.leastSignificant},
              {"items", items},
              {"surfaces", json::array({"stats", "identify", "alerts", "defense", "probe"})},
              {"excluded", json::array({"construct", "deploy", "human", "weapon"})}};
}

json SecurityObservatory::identifyJson(const std::string &id) const {
  std::lock_guard<std::mutex> lock(mu_);
  if (id == kInertProbeId || id == "inert") {
    return json{{"id", kInertProbeId},
                {"layer", "probe"},
                {"mapped", json::array({kInertProbeGlyph})},
                {"neighbors", json::array()},
                {"impactScope", "in-process-memegraph-activation-only"},
                {"inert", true},
                {"planted", probe_.planted},
                {"seed", probe_.seedId}};
  }
  for (const auto &n : report_.nodes) {
    if (n.id != id)
      continue;
    return json{{"id", n.id},
                {"layer", n.layer},
                {"mapped", n.mappedIds},
                {"neighbors", n.neighborIds},
                {"impactScope", n.impactScope},
                {"significance", n.significance},
                {"ragImpact", n.ragImpact},
                {"matrixImpact", n.matrixImpact}};
  }
  return json{};
}

json SecurityObservatory::identifyListJson() const {
  std::lock_guard<std::mutex> lock(mu_);
  json items = json::array();
  items.push_back(json{{"id", kInertProbeId},
                       {"layer", "probe"},
                       {"mapped", json::array({kInertProbeGlyph})},
                       {"impactScope", "in-process-memegraph-activation-only"},
                       {"inert", true}});
  for (const auto &n : report_.nodes) {
    items.push_back(json{{"id", n.id},
                         {"layer", n.layer},
                         {"mapped", n.mappedIds},
                         {"impactScope", n.impactScope},
                         {"significance", n.significance}});
  }
  return json{{"items", items}};
}

json SecurityObservatory::alertsJson() const {
  std::lock_guard<std::mutex> lock(mu_);
  json items = json::array();
  for (const auto &a : alerts_) {
    items.push_back(json{{"ts", a.ts},
                         {"kind", a.kind},
                         {"detail", a.detail},
                         {"extra", a.extra}});
  }
  return json{{"items", items}};
}

json SecurityObservatory::defenseJson() const {
  std::lock_guard<std::mutex> lock(mu_);
  return json{{"id", "switches"},
              {"defenseEnabled", cfg_.defenseEnabled},
              {"isolateHighImpact", cfg_.isolateHighImpact},
              {"researchObserve", cfg_.researchObserve},
              {"allowResearchObserve", cfg_.allowResearchObserve},
              {"highImpactQuantile", cfg_.highImpactQuantile}};
}

namespace {

util::CrudReply okBody(const json &data, int status = 200) {
  util::CrudReply r;
  r.ok = true;
  r.httpStatus = status;
  r.data = data;
  return r;
}

util::CrudReply handleStats(const util::CrudCall &call) {
  if (call.op == util::CrudOp::List)
    return okBody(SecurityObservatory::instance().statsJson());
  if (call.op == util::CrudOp::Get) {
    auto all = SecurityObservatory::instance().statsJson();
    if (call.id == "summary" || call.id == "latest")
      return okBody(all);
    return util::CrudReply{false, 404, "not_found", json::object()};
  }
  return util::CrudReply{false, 405, "method_not_allowed",
                         json{{"reason", "stats_read_only"}}};
}

util::CrudReply handleIdentify(const util::CrudCall &call) {
  if (call.op == util::CrudOp::List)
    return okBody(SecurityObservatory::instance().identifyListJson());
  if (call.op == util::CrudOp::Get) {
    json row = SecurityObservatory::instance().identifyJson(call.id);
    if (row.empty())
      return util::CrudReply{false, 404, "not_found", json::object()};
    return okBody(row);
  }
  return util::CrudReply{false, 405, "method_not_allowed",
                         json{{"reason", "identify_read_only"}}};
}

util::CrudReply handleAlerts(const util::CrudCall &call) {
  if (call.op == util::CrudOp::List)
    return okBody(SecurityObservatory::instance().alertsJson());
  if (call.op == util::CrudOp::Get) {
    auto all = SecurityObservatory::instance().alertsJson();
    if (call.id == "latest")
      return okBody(all);
    return util::CrudReply{false, 404, "not_found", json::object()};
  }
  return util::CrudReply{false, 405, "method_not_allowed",
                         json{{"reason", "alerts_read_only"}}};
}

util::CrudReply handleDefense(const util::CrudCall &call) {
  auto &obs = SecurityObservatory::instance();
  if (call.op == util::CrudOp::List) {
    return okBody(json{{"items", json::array({obs.defenseJson()})}});
  }
  if (call.op == util::CrudOp::Get) {
    if (call.id != "switches")
      return util::CrudReply{false, 404, "not_found", json::object()};
    return okBody(obs.defenseJson());
  }
  if (call.op == util::CrudOp::Update) {
    if (call.id != "switches")
      return util::CrudReply{false, 404, "not_found", json::object()};
    if (call.body.contains("defenseEnabled"))
      obs.setDefenseEnabled(call.body.value("defenseEnabled", true));
    if (call.body.contains("isolateHighImpact"))
      obs.setIsolateHighImpact(call.body.value("isolateHighImpact", true));
    if (call.body.contains("researchObserve")) {
      std::string err;
      if (!obs.setResearchObserve(call.body.value("researchObserve", false), &err))
        return util::CrudReply{false, 403, "forbidden",
                               json{{"reason", err}}};
    }
    if (call.body.contains("construct") || call.body.contains("deploy") ||
        call.body.contains("payload")) {
      return util::CrudReply{false, 404, "unregistered",
                             json{{"reason", "surface_not_provided"}}};
    }
    return okBody(obs.defenseJson());
  }
  return util::CrudReply{false, 405, "method_not_allowed", json::object()};
}

util::CrudReply handleProbe(const util::CrudCall &call) {
  auto &obs = SecurityObservatory::instance();
  if (call.body.contains("construct") || call.body.contains("deploy") ||
      call.body.contains("payload") || call.body.contains("human") ||
      call.body.contains("crossSession") || call.body.contains("network")) {
    return util::CrudReply{false, 404, "unregistered",
                           json{{"reason", "surface_not_provided"}}};
  }
  if (call.op == util::CrudOp::List)
    return okBody(json{{"items", json::array({obs.probeJson()})}});
  if (call.op == util::CrudOp::Get) {
    if (call.id != "inert" && call.id != kInertProbeId && call.id != "status")
      return util::CrudReply{false, 404, "not_found", json::object()};
    return okBody(obs.probeJson());
  }
  if (call.op == util::CrudOp::Update) {
    if (call.id != "inert" && call.id != "status")
      return util::CrudReply{false, 404, "not_found", json::object()};
    if (call.channel != util::CrudChannel::Internal)
      return util::CrudReply{false, 403, "forbidden",
                             json{{"reason", "probe_internal_only"}}};
    if (call.body.contains("probeEnabled")) {
      std::string err;
      if (!obs.setProbeEnabled(call.body.value("probeEnabled", false), &err))
        return util::CrudReply{false, 403, "forbidden", json{{"reason", err}}};
    }
    if (call.body.contains("plantSeed")) {
      std::string err;
      if (!obs.plantInertProbe(call.body.value("plantSeed", std::string()), &err))
        return util::CrudReply{false, 403, "forbidden", json{{"reason", err}}};
    }
    if (call.body.value("step", false)) {
      std::string err;
      if (!obs.stepInertProbeOnce(&err))
        return util::CrudReply{false, 403, "forbidden", json{{"reason", err}}};
    }
    return okBody(obs.probeJson());
  }
  return util::CrudReply{false, 405, "method_not_allowed",
                         json{{"reason", "probe_no_public_deploy"}}};
}

} // namespace

void SecurityObservatory::registerCrudResources() {
  installSecurityModuleResources();
}

void installSecurityModuleResources() {
  auto &reg = util::ModuleResourceRegistry::instance();
  if (reg.isRegistered("security", "stats"))
    return;

  util::ResourceAcl readOnly;
  readOnly.allowExternal = true;
  readOnly.allowInternal = true;
  readOnly.allowExternalWrite = false;
  readOnly.allowInternalWrite = false;

  util::ResourceAcl defenseAcl = readOnly;
  defenseAcl.allowInternalWrite = true;
  defenseAcl.allowExternalWrite = true;

  reg.registerResource(util::ResourceSpec{
      "security", "stats", "GNN influence statistics (read-only)", readOnly,
      handleStats});
  reg.registerResource(util::ResourceSpec{
      "security", "identify", "meme/word influence tags (read-only)", readOnly,
      handleIdentify});
  reg.registerResource(util::ResourceSpec{
      "security", "alerts", "defense alerts (read-only)", readOnly, handleAlerts});
  reg.registerResource(util::ResourceSpec{
      "security", "defense", "defense and research-observe switches", defenseAcl,
      handleDefense});

  util::ResourceAcl probeAcl = readOnly;
  probeAcl.allowInternalWrite = true;
  probeAcl.allowExternalWrite = false;
  reg.registerResource(util::ResourceSpec{
      "security", "probe", "inert in-graph existence probe (read + internal enable)",
      probeAcl, handleProbe});
}

} // namespace secamp
} // namespace phoenix
