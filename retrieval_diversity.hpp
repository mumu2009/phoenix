/* retrieval_diversity.hpp - width-preserving top-k selection (MMR).

   WHY
   ---
   Mission recall (`experienceTop`, `appendRecalledMemory`) sorted candidates by
   relevance and took the first k.  With k=3 that means three near-identical
   recipes can fill every slot: the model receives one idea repeated three
   times instead of three different ideas.  That is the same exploration
   collapse the summoner and the explore-path picker had - a greedy argmax
   over a plateau - applied to retrieval.

   This header implements Maximal Marginal Relevance (Carbonell & Goldstein,
   1998), which is the standard fix:

       mmr(i) = lambda * relevance(i) - (1 - lambda) * max_{j in S} sim(i, j)

   so a candidate is scored on how relevant it is AND how different it is from
   everything already selected.  lambda = 1 reproduces the old greedy order
   exactly (no behaviour change unless asked for), lambda = 0 is pure
   diversity.

   The selector is pure: it takes a relevance vector and a similarity callback,
   returns indices in selection order.  It is unit tested directly
   (tests/gtest/unit/autonomy/test_retrieval_diversity.cpp).
*/
#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <vector>

namespace phoenix {
namespace retrieval {

/* Select up to `k` indices from `n` candidates.

   relevance[i]  : higher = more relevant (any scale; only its order and the
                   relative gaps matter because the two terms are combined
                   with the same lambda weight).
   sim(i, j)     : similarity in [0,1] between two candidates.
   lambda_       : 1.0 = pure relevance (legacy greedy), 0.0 = pure diversity.

   Returns indices of the selected candidates in selection order.  Ties are
   broken by the lower index, so the result is deterministic. */
inline std::vector<int> mmrSelect(
    size_t n, const std::function<double(size_t)> &relevance,
    const std::function<double(size_t, size_t)> &sim, size_t k,
    double lambda_) {
    std::vector<int> out;
    if (n == 0 || k == 0) return out;
    if (!(lambda_ >= 0.0)) lambda_ = 0.0;
    if (lambda_ > 1.0) lambda_ = 1.0;

    /* Scale relevance by its maximum instead of min-max.
       Min-max maps the WORST candidate to 0, which over-penalises it: with
       relevance 0.9/0.88/0.7 the distinct-but-less-relevant third item could
       never win a slot no matter how different it was.  Dividing by the max
       keeps the relative gaps (a 0.88 is still "almost as relevant as 0.9")
       and leaves similarity able to decide. */
    double hi = 0.0;
    bool anyPositive = false;
    for (size_t i = 0; i < n; ++i) {
        const double r = relevance(i);
        if (r > hi) hi = r;
        if (r > 0.0) anyPositive = true;
    }
    const bool flat = !(hi > 0.0);
    auto rel01 = [&](size_t i) -> double {
        const double r = relevance(i);
        if (flat) return 1.0; /* all zero/equal: relevance cannot separate */
        if (!anyPositive) return 0.0;
        return std::max(0.0, std::min(1.0, r / hi));
    };

    std::vector<char> taken(n, 0);
    for (size_t pick = 0; pick < k && pick < n; ++pick) {
        int best = -1;
        double bestScore = 0.0;
        for (size_t i = 0; i < n; ++i) {
            if (taken[i]) continue;
            double maxSim = 0.0;
            for (int j : out) {
                double sv = 0.0;
                try {
                    sv = sim(i, static_cast<size_t>(j));
                } catch (...) {
                    sv = 0.0;
                }
                if (!(sv >= 0.0) || sv > 1.0) sv = 0.0; /* NaN / out of range */
                maxSim = std::max(maxSim, sv);
            }
            const double sc =
                lambda_ * rel01(i) - (1.0 - lambda_) * maxSim;
            if (best < 0 || sc > bestScore) {
                best = static_cast<int>(i);
                bestScore = sc;
            }
        }
        if (best < 0) break;
        taken[static_cast<size_t>(best)] = 1;
        out.push_back(best);
    }
    return out;
}

}  // namespace retrieval
}  // namespace phoenix
