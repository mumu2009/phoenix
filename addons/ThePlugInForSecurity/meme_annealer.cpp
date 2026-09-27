/* meme_annealer.cpp - see header for the design note. */

#include "meme_annealer.hpp"

#include "../../memetic_existence.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>
#include <unordered_set>

namespace phoenix {
namespace secamp {

namespace {

struct GroupState {
  std::vector<int> members;
  double fitness{0.0};
  double cohesion{0.0};
  double energy{0.0};
  int wordCount{0};
};

struct Context {
  const std::vector<AnnealMeme> *pool{nullptr};
  std::vector<std::unordered_map<int, double>> adj; /* meme -> (meme -> w) */
  std::unordered_map<std::string, int> df;          /* word -> pool df */
  AnnealConfig cfg;
  int poolSize{0};
};

/* Merged (word, alpha) of a member set: alpha = max over members. */
std::unordered_map<std::string, double>
mergedAlpha(const Context &ctx, const std::vector<int> &members) {
  std::unordered_map<std::string, double> out;
  for (const int mi : members) {
    for (const auto &wp : (*ctx.pool)[static_cast<size_t>(mi)].weights) {
      if (wp.first.empty() || wp.second <= 0.0)
        continue;
      auto it = out.find(wp.first);
      if (it == out.end() || it->second < wp.second)
        out[wp.first] = wp.second;
    }
  }
  return out;
}

void recompute(const Context &ctx, GroupState &g) {
  const auto alpha = mergedAlpha(ctx, g.members);
  g.wordCount = static_cast<int>(alpha.size());

  std::vector<std::pair<std::string, double>> merged(alpha.begin(),
                                                     alpha.end());
  const double spec = phoenix::memetic::mappingContentSpec(
      merged, ctx.df, ctx.poolSize);
  const auto typical = phoenix::memetic::mappingTypicalAlpha(merged);
  const double tc = static_cast<double>(typical.size());
  /* Typical-set sweet spot: judgeable memes carry >= 2 typical words;
     beyond 6 the set stops being minimal. */
  const double sweet = (tc >= 2.0 ? 1.0 : tc / 2.0) *
                       (tc <= 6.0 ? 1.0 : 6.0 / std::max(1.0, tc));
  g.fitness = spec * sweet;

  double cohesion = 0.0;
  const size_t m = g.members.size();
  if (m >= 2) {
    double internalW = 0.0;
    std::unordered_map<std::string, int> wordRef;
    std::unordered_set<int> in(g.members.begin(), g.members.end());
    for (const int mi : g.members) {
      for (const auto &nb : ctx.adj[static_cast<size_t>(mi)])
        if (nb.first > mi && in.count(nb.first))
          internalW += nb.second;
      for (const auto &wp : (*ctx.pool)[static_cast<size_t>(mi)].weights)
        wordRef[wp.first]++;
    }
    int sharedContent = 0;
    for (const auto &kv : wordRef) {
      if (kv.second >= 2 &&
          !phoenix::memetic::mappingMergeStopWord(kv.first) &&
          kv.first.size() >= 3)
        ++sharedContent;
    }
    const double pairs = static_cast<double>(m) * (m - 1) / 2.0;
    const double edgeDensity = internalW / pairs;
    const double sharedFrac =
        static_cast<double>(sharedContent) / std::max(1, g.wordCount);
    cohesion = edgeDensity + 0.5 * sharedFrac;
  }
  g.cohesion = cohesion;

  const int over = std::max(0, g.wordCount - ctx.cfg.softWords);
  g.energy = -(g.fitness + ctx.cfg.cohesionWeight * g.cohesion) +
             ctx.cfg.sizePenalty * over * over;
}

GroupState buildGroup(const Context &ctx, std::vector<int> members) {
  GroupState g;
  std::sort(members.begin(), members.end());
  members.erase(std::unique(members.begin(), members.end()), members.end());
  g.members = std::move(members);
  recompute(ctx, g);
  return g;
}

} // namespace

std::vector<std::pair<int, int>>
MemeAnnealer::organicMergePairs(const std::vector<AnnealMeme> &pool,
                                int minOverlap, int maxMemeWords) {
  std::vector<std::pair<int, int>> out;
  const int n = static_cast<int>(pool.size());
  if (n < 2)
    return out;
  const int minOv = std::max(1, minOverlap);
  const int cap = std::max(4, maxMemeWords);

  std::vector<std::unordered_set<std::string>> words(static_cast<size_t>(n));
  std::unordered_map<std::string, std::vector<int>> byWord;
  for (int i = 0; i < n; ++i) {
    for (const auto &wp : pool[static_cast<size_t>(i)].weights) {
      if (wp.first.empty())
        continue;
      if (words[static_cast<size_t>(i)].insert(wp.first).second)
        byWord[wp.first].push_back(i);
    }
  }

  /* Hub words (df > sqrt(n)) only ever glue unrelated articles; skip them
     in the inverted index, same prior as the official mapping. */
  std::unordered_map<std::pair<int, int>, int, decltype([](const auto &p) {
                       return (static_cast<size_t>(p.first) << 32) ^
                              static_cast<size_t>(p.second);
                     })>
      overlap;
  for (const auto &kv : byWord) {
    const auto &list = kv.second;
    if (phoenix::memetic::mappingHighDf(static_cast<int>(list.size()), n))
      continue;
    for (size_t a = 0; a < list.size(); ++a)
      for (size_t b = a + 1; b < list.size(); ++b) {
        const int i = std::min(list[a], list[b]);
        const int j = std::max(list[a], list[b]);
        overlap[{i, j}]++;
      }
  }

  std::vector<std::pair<int, int>> cands;
  cands.reserve(overlap.size());
  for (const auto &kv : overlap) {
    if (kv.second >= minOv)
      cands.push_back(kv.first);
  }
  std::sort(cands.begin(), cands.end());

  auto contentOf = [&](int i) {
    int c = 0;
    for (const auto &w : words[static_cast<size_t>(i)])
      if (!phoenix::memetic::mappingMergeStopWord(w))
        ++c;
    return c;
  };
  auto overlapContentOf = [&](int i, int j) {
    int c = 0;
    const auto &small = words[static_cast<size_t>(i)].size() <=
                                words[static_cast<size_t>(j)].size()
                            ? words[static_cast<size_t>(i)]
                            : words[static_cast<size_t>(j)];
    const auto &large = &small == &words[static_cast<size_t>(i)]
                            ? words[static_cast<size_t>(j)]
                            : words[static_cast<size_t>(i)];
    for (const auto &w : small)
      if (large.count(w) && !phoenix::memetic::mappingMergeStopWord(w))
        ++c;
    return c;
  };

  for (const auto &pr : cands) {
    const int i = pr.first;
    const int j = pr.second;
    const auto &wi = words[static_cast<size_t>(i)];
    const auto &wj = words[static_cast<size_t>(j)];
    int uni = static_cast<int>(wi.size());
    for (const auto &w : wj)
      if (!wi.count(w))
        ++uni;
    if (uni > cap)
      continue;
    const int ov = overlap.at(pr);
    const int oc = overlapContentOf(i, j);
    const int ci = contentOf(i);
    const int cj = contentOf(j);
    /* The GNN gate is asymmetric (incoming window vs existing meme); a
       pair is organic if the gate passes in either direction. */
    if (phoenix::memetic::mappingMergeAllowed(ov, static_cast<int>(wj.size()),
                                              static_cast<int>(wi.size()),
                                              minOv, cj, oc) ||
        phoenix::memetic::mappingMergeAllowed(ov, static_cast<int>(wi.size()),
                                              static_cast<int>(wj.size()),
                                              minOv, ci, oc))
      out.push_back(pr);
  }
  return out;
}

AnnealResult MemeAnnealer::run(const std::vector<AnnealMeme> &pool,
                               const std::vector<AnnealEdge> &edges,
                               const AnnealConfig &cfg) {
  AnnealResult out;
  Context ctx;
  ctx.pool = &pool;
  ctx.cfg = cfg;
  ctx.poolSize = static_cast<int>(pool.size());
  const int n = ctx.poolSize;
  if (n <= 0)
    return out;

  ctx.adj.resize(static_cast<size_t>(n));
  for (const auto &e : edges) {
    if (e.a < 0 || e.b < 0 || e.a >= n || e.b >= n || e.a == e.b)
      continue;
    ctx.adj[static_cast<size_t>(e.a)][e.b] += e.w;
    ctx.adj[static_cast<size_t>(e.b)][e.a] += e.w;
  }
  for (const auto &m : pool)
    for (const auto &wp : m.weights)
      if (!wp.first.empty())
        ctx.df[wp.first]++;

  /* Initial state: every meme its own group (the GNN's status quo). */
  std::vector<GroupState> groups(static_cast<size_t>(n));
  std::vector<int> active(static_cast<size_t>(n));
  std::vector<int> posInActive(static_cast<size_t>(n));
  double totalE = 0.0;
  for (int i = 0; i < n; ++i) {
    groups[static_cast<size_t>(i)] = buildGroup(ctx, {i});
    active[static_cast<size_t>(i)] = i;
    posInActive[static_cast<size_t>(i)] = i;
    totalE += groups[static_cast<size_t>(i)].energy;
  }
  out.initialEnergy = totalE;

  std::mt19937_64 rng(cfg.seed);
  auto deactivate = [&](int gid) {
    const int pos = posInActive[static_cast<size_t>(gid)];
    const int last = active.back();
    active[static_cast<size_t>(pos)] = last;
    posInActive[static_cast<size_t>(last)] = pos;
    active.pop_back();
    posInActive[static_cast<size_t>(gid)] = -1;
  };
  auto activate = [&](int gid) {
    if (static_cast<size_t>(gid) >= posInActive.size())
      posInActive.resize(static_cast<size_t>(gid) + 1, -1);
    posInActive[static_cast<size_t>(gid)] =
        static_cast<int>(active.size());
    active.push_back(gid);
  };

  /* Best-state snapshot (member lists per active group). */
  auto snapshotMembers = [&] {
    std::vector<std::vector<int>> snap;
    snap.reserve(active.size());
    for (const int gid : active)
      snap.push_back(groups[static_cast<size_t>(gid)].members);
    return snap;
  };
  double bestE = totalE;
  auto best = snapshotMembers();

  double T = std::max(1e-6, cfg.t0);
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  int done = 0;
  for (int step = 0; step < cfg.steps && active.size() >= 1; ++step) {
    ++done;
    const int move = static_cast<int>(rng() % 3);
    const size_t na = active.size();

    if (move == 0 && na >= 2) {
      /* merge two distinct groups */
      size_t pa = rng() % na;
      size_t pb = rng() % na;
      if (pa == pb)
        pb = (pb + 1) % na;
      const int ga = active[pa];
      const int gb = active[pb];
      std::vector<int> members = groups[static_cast<size_t>(ga)].members;
      members.insert(members.end(),
                     groups[static_cast<size_t>(gb)].members.begin(),
                     groups[static_cast<size_t>(gb)].members.end());
      std::sort(members.begin(), members.end());
      members.erase(std::unique(members.begin(), members.end()),
                    members.end());
      /* Hard cap: no giant meme may appear. */
      const auto alpha = mergedAlpha(ctx, members);
      if (static_cast<int>(alpha.size()) > cfg.maxMemeWords) {
        T *= cfg.cooling;
        continue;
      }
      GroupState merged = buildGroup(ctx, std::move(members));
      const double dE = merged.energy -
                        groups[static_cast<size_t>(ga)].energy -
                        groups[static_cast<size_t>(gb)].energy;
      if (dE <= 0.0 || uni(rng) < std::exp(-dE / T)) {
        const int keep = std::min(ga, gb);
        const int drop = std::max(ga, gb);
        groups[static_cast<size_t>(keep)] = std::move(merged);
        deactivate(drop);
        totalE += dE;
      }
    } else if (move == 1) {
      /* split a random multi-member group */
      std::vector<int> multi;
      for (const int gid : active)
        if (groups[static_cast<size_t>(gid)].members.size() >= 2)
          multi.push_back(gid);
      if (multi.empty()) {
        T *= cfg.cooling;
        continue;
      }
      const int ga = multi[static_cast<size_t>(rng() % multi.size())];
      auto members = groups[static_cast<size_t>(ga)].members;
      std::shuffle(members.begin(), members.end(), rng);
      const size_t cut =
          1 + static_cast<size_t>(rng() % (members.size() - 1));
      GroupState g1 = buildGroup(
          ctx, std::vector<int>(members.begin(),
                                members.begin() +
                                    static_cast<std::ptrdiff_t>(cut)));
      GroupState g2 = buildGroup(
          ctx, std::vector<int>(members.begin() +
                                    static_cast<std::ptrdiff_t>(cut),
                                members.end()));
      const double dE =
          g1.energy + g2.energy - groups[static_cast<size_t>(ga)].energy;
      if (dE <= 0.0 || uni(rng) < std::exp(-dE / T)) {
        groups[static_cast<size_t>(ga)] = std::move(g1);
        groups.push_back(std::move(g2));
        activate(static_cast<int>(groups.size()) - 1);
        totalE += dE;
      }
    } else if (na >= 2) {
      /* move one meme between groups */
      std::vector<int> movable;
      for (const int gid : active)
        if (groups[static_cast<size_t>(gid)].members.size() >= 2)
          movable.push_back(gid);
      if (movable.empty()) {
        T *= cfg.cooling;
        continue;
      }
      const int ga =
          movable[static_cast<size_t>(rng() % movable.size())];
      size_t pb = rng() % na;
      if (active[pb] == ga)
        pb = (pb + 1) % na;
      const int gb = active[pb];
      const auto &memA = groups[static_cast<size_t>(ga)].members;
      const int x = memA[static_cast<size_t>(rng() % memA.size())];
      std::vector<int> memNewA;
      memNewA.reserve(memA.size() - 1);
      for (const int m : memA)
        if (m != x)
          memNewA.push_back(m);
      std::vector<int> memNewB = groups[static_cast<size_t>(gb)].members;
      memNewB.push_back(x);
      const auto alphaB = mergedAlpha(ctx, memNewB);
      if (static_cast<int>(alphaB.size()) > cfg.maxMemeWords) {
        T *= cfg.cooling;
        continue;
      }
      GroupState gA = buildGroup(ctx, std::move(memNewA));
      GroupState gB = buildGroup(ctx, std::move(memNewB));
      const double dE = gA.energy + gB.energy -
                        groups[static_cast<size_t>(ga)].energy -
                        groups[static_cast<size_t>(gb)].energy;
      if (dE <= 0.0 || uni(rng) < std::exp(-dE / T)) {
        groups[static_cast<size_t>(ga)] = std::move(gA);
        groups[static_cast<size_t>(gb)] = std::move(gB);
        totalE += dE;
      }
    }

    if (totalE < bestE - 1e-12) {
      bestE = totalE;
      best = snapshotMembers();
    }
    T *= cfg.cooling;
  }

  out.energy = bestE;
  out.stepsDone = done;

  /* Materialize the best partition, ranked best-first and deterministic. */
  std::vector<AnnealGroupResult> res;
  res.reserve(best.size());
  for (const auto &members : best) {
    GroupState g = buildGroup(ctx, members);
    AnnealGroupResult r;
    r.members = g.members;
    r.fitness = g.fitness;
    r.cohesion = g.cohesion;
    r.wordCount = g.wordCount;
    const auto alpha = mergedAlpha(ctx, g.members);
    r.merged.assign(alpha.begin(), alpha.end());
    std::sort(r.merged.begin(), r.merged.end(),
              [](const auto &a, const auto &b) {
                if (std::abs(a.second - b.second) > 1e-12)
                  return a.second > b.second;
                return a.first < b.first;
              });
    const auto typical = phoenix::memetic::mappingTypicalAlpha(r.merged);
    std::vector<std::pair<std::string, double>> typ(typical.begin(),
                                                    typical.end());
    std::sort(typ.begin(), typ.end(), [](const auto &a, const auto &b) {
      if (std::abs(a.second - b.second) > 1e-12)
        return a.second > b.second;
      return a.first < b.first;
    });
    for (const auto &kv : typ)
      r.typical.push_back(kv.first);
    res.push_back(std::move(r));
  }
  std::sort(res.begin(), res.end(), [&](const auto &a, const auto &b) {
    const double sa = a.fitness + cfg.cohesionWeight * a.cohesion;
    const double sb = b.fitness + cfg.cohesionWeight * b.cohesion;
    if (std::abs(sa - sb) > 1e-12)
      return sa > sb;
    if (a.wordCount != b.wordCount)
      return a.wordCount < b.wordCount;
    const std::string ia = a.members.empty()
                               ? std::string()
                               : pool[static_cast<size_t>(a.members[0])].id;
    const std::string ib = b.members.empty()
                               ? std::string()
                               : pool[static_cast<size_t>(b.members[0])].id;
    return ia < ib;
  });
  out.groups = std::move(res);
  return out;
}

} // namespace secamp
} // namespace phoenix
