/* replica_trace.hpp - Multi-instance marker trace.

   One tick is a marked replicator: explore, summon, reply class, and
   whether the reply is a source edit. Markers stay on disk. They are
   not written into the model prompt. The security probe glyph is a
   different instrument and is not used here.

   Two formulas from the paper library are applied to the marks.

   Eigen's master-sequence length (Eigen 1971; Bull, Meyers, Lachmann
   2005): a master of length L is maintainable when the per-symbol
   fidelity q satisfies q^L * sigma > 1, so
   Lmax = ln(sigma) / -ln(q) for 0 < q < 1 and sigma > 1.
   q = 0 means no master of any positive length is being kept.

   Cultural Price equation (Price 1970; El Mouden, Andre, Morin, Nettle
   2014): Delta z = cov(c, z) / mean(c) + E(Delta z).
   z is 1 when the tick is a source edit. Cultural fitness c is 1 until
   an external score distinguishes the instances, so the covariance term
   is 0 and any change is transmission.
*/
#pragma once

#include <cmath>
#include <cctype>
#include <cstdlib>
#include <mutex>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace phoenix {
namespace memetic {

struct ReplicaTick {
  std::string instance;
  std::string mission;
  std::string explore;
  std::string summon;
  std::string replyClass;
  int replyChars{0};
  bool sourceEdit{false};
  /* Population generation. 0 = legacy per-tick series.  A generation is
     one full round in which every carrier ticks once; the force should be
     fitted on the per-generation frequency, not on the per-carrier tick
     (which makes a first-appearance ramp that no single force produces). */
  int generation{0};
  double projectScore{0.0};
  double copyFidelity{0.0};
  double indexOverlap{0.0};
  bool replicator{false};
  double attractionBias{0.0};
  int peerCopies{0};
  std::string mechanism{"undetermined"};
};

/* A copy judgement. fidelity is the longest contiguous shared token
   span divided by the parent's token count. indexOverlap is the
   fraction of the parent's word types that appear anywhere in the
   offspring. The second number is the (word, weight) description.
   It is not identity.

   The offspring is a replicator of that parent only when the parent
   has at least two tokens and fidelity >= sigma^(-1/L), L = parent
   token count, sigma > 1. That is Eigen's condition for the parent
   still being the master sequence. indexWithoutCopy is a shared
   vocabulary that fails this test. */
struct CopyJudgement {
  double fidelity{0.0};
  double indexOverlap{0.0};
  int parentTokens{0};
  int copiedSpan{0};
  double fidelityFloor{1.0};
  bool replicator{false};
  bool indexWithoutCopy{false};
};

inline std::vector<std::string> copyTokens(const std::string &text) {
  std::vector<std::string> out;
  std::string cur;
  auto flush = [&]() {
    if (cur.size() >= 3)
      out.push_back(cur);
    cur.clear();
  };
  for (unsigned char c : text) {
    if (std::isalnum(c))
      cur.push_back(static_cast<char>(std::tolower(c)));
    else
      flush();
  }
  flush();
  return out;
}

inline CopyJudgement judgeCopy(const std::string &parent,
                               const std::string &offspring,
                               double sigma = 2.0) {
  CopyJudgement out;
  const auto p = copyTokens(parent);
  const auto o = copyTokens(offspring);
  out.parentTokens = static_cast<int>(p.size());
  if (p.empty() || o.empty())
    return out;
  std::vector<int> span(o.size() + 1, 0);
  int best = 0;
  for (size_t i = 0; i < p.size(); ++i) {
    for (size_t j = o.size(); j-- > 0;) {
      if (p[i] == o[j]) {
        span[j + 1] = span[j] + 1;
        if (span[j + 1] > best)
          best = span[j + 1];
      } else {
        span[j + 1] = 0;
      }
    }
    span[0] = 0;
  }
  out.copiedSpan = best;
  out.fidelity = static_cast<double>(best) / static_cast<double>(p.size());
  std::unordered_set<std::string> inOffspring(o.begin(), o.end());
  std::unordered_set<std::string> parentTypes;
  int hit = 0;
  for (const auto &w : p) {
    if (!parentTypes.insert(w).second)
      continue;
    if (inOffspring.count(w))
      ++hit;
  }
  out.indexOverlap = parentTypes.empty()
                         ? 0.0
                         : static_cast<double>(hit) /
                               static_cast<double>(parentTypes.size());
  if (sigma > 1.0 && p.size() >= 2) {
    out.fidelityFloor = std::pow(sigma, -1.0 / static_cast<double>(p.size()));
    out.replicator = out.fidelity + 1e-12 >= out.fidelityFloor;
  }
  out.indexWithoutCopy =
      parentTypes.size() >= 2 && out.indexOverlap >= 0.5 && !out.replicator;
  return out;
}

/* Parent fidelity is how much of the designated parent reappears in the
   offspring. Internal fidelity is how much the offspring reappear in
   each other. Attraction bias is internal minus parent. Claidiere,
   Scott-Phillips and Sperber (2014): attraction is the general
   population process; a replicator is the special case that clears
   the copy threshold. A positive bias is not itself a meme. */
struct PopulationRelation {
  double parentFidelity{0.0};
  double internalFidelity{0.0};
  double attractionBias{0.0};
  int parentCopies{0};
  int peerCopies{0};
  std::string mechanism{"undetermined"};
};

/* Claidiere, Scott-Phillips, Sperber 2014: copying is the special case
   of attraction that preserves the designated parent. Scott-Phillips,
   Blancke, Heintz 2018: attraction is not a rival object that replaces
   selection, and selection is not inferred from this snapshot.
   Fixed before the 2026-10-24 population probe:
   replication: at least one offspring clears judgeCopy against the
   parent, and offspring are not closer to each other than to the parent.
   attraction: no offspring copies the parent, at least one offspring
   pair copies itself, and the bias is positive.
   undetermined: everything else, including a parent copy sitting inside
   a population whose bias is positive. Price selection is not a label
   here. This label does not admit a lineage. One snapshot also does
   not say whether the frequency moved by content bias or by conformity.
   Those are contentBiasStep and conformityStep. */
/* Boyd and Richerson 1985, as written in Denton, Ram, Liberman and
   Feldman, PNAS 2020, equation (1). p is the frequency of variant A.
   Content (direct) bias: p' = p + B p (1-p). The sign of one step is
   the sign of B on (0,1). There is no interior fixed point when B != 0.
   Conformity: p' = p + D p (1-p) (2p-1), with -2 < D < 1 when the
   coefficient comes from three role models. D = 0 copies at random.
   D > 0 moves p away from 1/2. D < 0 moves p toward 1/2.
   Derivative at the interior point: f'(1/2) = 1 + D/2.
   A single frequency can be matched by both maps, because
   B = D (2p-1) makes the first step identical. Later steps diverge
   because conformity's factor (2p-1) changes with p and content's
   B does not. */
inline double contentBiasStep(double frequency, double bias) {
  return frequency + bias * frequency * (1.0 - frequency);
}

inline double conformityStep(double frequency, double conformity) {
  return frequency +
         conformity * frequency * (1.0 - frequency) * (2.0 * frequency - 1.0);
}

inline std::string classifyPopulation(const PopulationRelation &relation) {
  if (relation.parentCopies >= 1 && relation.attractionBias <= 0.0)
    return "replication";
  if (relation.parentCopies == 0 && relation.peerCopies >= 1 &&
      relation.attractionBias > 0.0)
    return "attraction";
  return "undetermined";
}

inline PopulationRelation relatePopulation(
    const std::string &parent, const std::vector<std::string> &offspring,
    double sigma = 2.0) {
  PopulationRelation out;
  std::vector<std::string> kids;
  kids.reserve(offspring.size());
  for (const auto &text : offspring) {
    if (!copyTokens(text).empty())
      kids.push_back(text);
  }
  if (kids.empty()) {
    out.mechanism = classifyPopulation(out);
    return out;
  }
  double sumParent = 0.0;
  for (const auto &kid : kids) {
    const auto judged = judgeCopy(parent, kid, sigma);
    sumParent += judged.fidelity;
    if (judged.replicator)
      ++out.parentCopies;
  }
  out.parentFidelity = sumParent / static_cast<double>(kids.size());
  if (kids.size() < 2) {
    out.mechanism = classifyPopulation(out);
    return out;
  }
  double sumInternal = 0.0;
  int pairs = 0;
  for (size_t i = 0; i < kids.size(); ++i) {
    for (size_t j = 0; j < kids.size(); ++j) {
      if (i == j)
        continue;
      const auto judged = judgeCopy(kids[i], kids[j], sigma);
      sumInternal += judged.fidelity;
      ++pairs;
      if (judged.replicator)
        ++out.peerCopies;
    }
  }
  out.internalFidelity =
      pairs > 0 ? sumInternal / static_cast<double>(pairs) : 0.0;
  out.attractionBias = out.internalFidelity - out.parentFidelity;
  out.mechanism = classifyPopulation(out);
  return out;
}

/* Dawkins 1976, chapter 11: longevity, fecundity, and copying fidelity.
   Fidelity is judgeCopy. Fecundity counts distinct later carriers that
   cleared that test. Longevity is the parent plus the current run of
   those copies. A same-vocabulary text that fails the copy test
   (indexWithoutCopy) ends the run. An unrelated text does not: it is
   not a generation of this lineage. Admitted only when fecundity >= 2
   and longevity >= 2, so one echo is not a meme. */
struct LineageState {
  int fecundity{0};
  int longevity{1};
  int streak{0};
};

inline bool lineageAdmitted(const LineageState &state) {
  return state.fecundity >= 2 && state.longevity >= 2;
}

/* Discrete replicator map. p'_i = p_i f_i / sum_j p_j f_j.
   A mismatched or empty input, or a non-positive total weight, returns p.
   Frequency is not a class id, and fitness is not copy fidelity. */
inline std::vector<double> replicatorStep(const std::vector<double> &frequency,
                                          const std::vector<double> &fitness) {
  if (frequency.size() != fitness.size() || frequency.empty())
    return frequency;
  double mass = 0.0;
  for (size_t i = 0; i < frequency.size(); ++i) {
    if (frequency[i] > 0.0 && fitness[i] > 0.0)
      mass += frequency[i] * fitness[i];
  }
  if (!(mass > 0.0)) return frequency;
  std::vector<double> next(frequency.size(), 0.0);
  for (size_t i = 0; i < frequency.size(); ++i) {
    if (frequency[i] > 0.0 && fitness[i] > 0.0)
      next[i] = frequency[i] * fitness[i] / mass;
  }
  return next;
}

/* How many of n carriers edit on the next round when the decision rule
   is one of the three maps (or a constant frequency). frequency is the
   measured share from the previous round. parameter is B, D, or w.
   The count is the nearest integer to n * p'. This is the explicit bias
   term; it does not read a hand-written schedule. */
inline int plannedEditCount(const std::string &rule, double frequency, int n,
                            double parameter) {
  if (n <= 0) return 0;
  double next = frequency;
  if (rule == "content")
    next = contentBiasStep(frequency, parameter);
  else if (rule == "conformity")
    next = conformityStep(frequency, parameter);
  else if (rule == "replicator") {
    const std::vector<double> share{frequency, 1.0 - frequency};
    const std::vector<double> weight{parameter, 1.0};
    const std::vector<double> stepped = replicatorStep(share, weight);
    next = stepped.empty() ? frequency : stepped[0];
  } else if (rule == "fixed")
    next = frequency;
  if (next < 0.0) next = 0.0;
  if (next > 1.0) next = 1.0;
  const int k = static_cast<int>(std::lround(next * static_cast<double>(n)));
  if (k < 0) return 0;
  if (k > n) return n;
  return k;
}

inline void lineageObserve(LineageState &state, const CopyJudgement &copied,
                           bool distinctCarrier) {
  if (copied.replicator) {
    state.streak += 1;
    const int life = state.streak + 1;
    if (life > state.longevity)
      state.longevity = life;
    if (distinctCarrier)
      state.fecundity += 1;
    return;
  }
  if (copied.indexWithoutCopy)
    state.streak = 0;
}

/* Acerbi, Charbonneau, Miton, Scott-Phillips 2021: under convergent
   transformation the input-output distance starts high and then falls;
   under copying it stays low. These cuts were fixed before the
   synthetic run on 2026-10-24. They apply to a series of stepwise
   fidelities f_t in [0,1], where 1 means the child kept the parent.
   Copying: both halves stay at or above 0.75. Convergent: the first
   half is at or below 0.55 and the second half is at least 0.25
   higher. Anything else is undetermined. A short series is
   undetermined. Do not feed fidelity-to-a-fixed-goal into this; that
   is not the input-output step. */
struct TransmissionRegime {
  std::string name{"undetermined"};
  double early{0.0};
  double late{0.0};
  int steps{0};
};

inline TransmissionRegime
judgeTransmissionRegime(const std::vector<double> &stepFidelity) {
  TransmissionRegime out;
  out.steps = static_cast<int>(stepFidelity.size());
  if (stepFidelity.size() < 6)
    return out;
  const size_t half = stepFidelity.size() / 2;
  double earlySum = 0.0;
  double lateSum = 0.0;
  for (size_t i = 0; i < half; ++i)
    earlySum += stepFidelity[i];
  for (size_t i = half; i < stepFidelity.size(); ++i)
    lateSum += stepFidelity[i];
  out.early = earlySum / static_cast<double>(half);
  out.late = lateSum / static_cast<double>(stepFidelity.size() - half);
  if (out.early >= 0.75 && out.late >= 0.75)
    out.name = "copying";
  else if (out.early <= 0.55 && out.late - out.early >= 0.25)
    out.name = "convergent";
  return out;
}

struct ReplicaAccount {
  int ticks{0};
  int sourceEdits{0};
  int replicators{0};
  double fidelity{0.0};
  double eigenLmax{0.0};
  double priceSelection{0.0};
  double priceTransmission{0.0};
  std::string binding;
  /* Force identified from the population frequency series of the same
     ticks. "undetermined" when fewer than three ticks or when no map
     wins by the pre-registered margin. */
  std::string force{"undetermined"};
  int forceSamples{0};
  double forceResidual{0.0};
  double forceContent{0.0};
  double forceConformity{0.0};
  double forceReplicator{0.0};
};

inline double eigenMasterLength(double q, double sigma) {
  if (!(sigma > 1.0))
    return 0.0;
  if (q >= 1.0)
    return 1.0e9;
  if (!(q > 0.0))
    return 0.0;
  return std::log(sigma) / -std::log(q);
}

inline std::string classifyReplicaReply(const std::string &goal,
                                        const std::string &reply,
                                        bool sourceEdit) {
  if (sourceEdit)
    return "source-edit";
  if (reply.find(".cpp") != std::string::npos ||
      reply.find(".hpp") != std::string::npos ||
      reply.find(".inc") != std::string::npos ||
      reply.find(".html") != std::string::npos)
    return "path-mentioned";
  if (reply.size() >= 24 && goal.size() >= 24) {
    if (goal.find(reply.substr(0, 24)) != std::string::npos)
      return "echo";
    const std::string tail = goal.substr(goal.size() - 24);
    if (reply.find(tail) != std::string::npos)
      return "echo";
  }
  return "prose";
}

inline ReplicaAccount accountReplicas(const std::vector<ReplicaTick> &ticks,
                                      double sigma = 2.0) {
  ReplicaAccount out;
  out.ticks = static_cast<int>(ticks.size());
  if (ticks.empty()) {
    out.binding = "empty";
    return out;
  }
  double sumC = 0.0;
  double sumZ = 0.0;
  std::vector<std::pair<std::string, int>> order;
  for (const auto &t : ticks) {
    const double z = t.sourceEdit ? 1.0 : 0.0;
    const double c = 1.0;
    sumC += c;
    sumZ += z;
    if (t.sourceEdit)
      ++out.sourceEdits;
    if (t.replicator)
      ++out.replicators;
    int prev = -1;
    for (const auto &seen : order) {
      if (seen.first == t.instance)
        prev = seen.second;
    }
    if (prev >= 0)
      out.priceTransmission += z - (ticks[static_cast<size_t>(prev)].sourceEdit
                                        ? 1.0
                                        : 0.0);
    order.push_back({t.instance, static_cast<int>(order.size())});
  }
  int steps = 0;
  for (size_t i = 0; i < order.size(); ++i) {
    for (size_t j = 0; j < i; ++j) {
      if (order[j].first == order[i].first)
        ++steps;
    }
  }
  /* The loop above counts every earlier tick of the same instance, not
     only the adjacent step. Recount adjacent steps only. */
  steps = 0;
  out.priceTransmission = 0.0;
  std::vector<int> last(0);
  std::vector<std::string> lastName;
  for (size_t i = 0; i < ticks.size(); ++i) {
    int prev = -1;
    for (size_t j = 0; j < lastName.size(); ++j) {
      if (lastName[j] == ticks[i].instance)
        prev = last[j];
    }
    const double z = ticks[i].sourceEdit ? 1.0 : 0.0;
    if (prev >= 0) {
      const double z0 = ticks[static_cast<size_t>(prev)].sourceEdit ? 1.0 : 0.0;
      out.priceTransmission += z - z0;
      ++steps;
      for (size_t j = 0; j < lastName.size(); ++j) {
        if (lastName[j] == ticks[i].instance)
          last[j] = static_cast<int>(i);
      }
    } else {
      lastName.push_back(ticks[i].instance);
      last.push_back(static_cast<int>(i));
    }
  }
  if (steps > 0)
    out.priceTransmission /= static_cast<double>(steps);
  const double n = static_cast<double>(ticks.size());
  const double meanC = sumC / n;
  const double meanZ = sumZ / n;
  double cov = 0.0;
  for (const auto &t : ticks) {
    const double z = t.sourceEdit ? 1.0 : 0.0;
    cov += (1.0 - meanC) * (z - meanZ);
  }
  cov /= n;
  out.priceSelection = meanC > 1e-12 ? cov / meanC : 0.0;
  out.fidelity = out.sourceEdits / n;
  out.eigenLmax = eigenMasterLength(out.fidelity, sigma);
  if (out.fidelity <= 0.0)
    out.binding = "below-error-threshold";
  else if (std::fabs(out.priceTransmission) > std::fabs(out.priceSelection) + 1e-12)
    out.binding = "transmission";
  else
    out.binding = "selection";
  return out;
}

/* Price's selection term cov(c, z) / mean(c) for a population whose
   fitness is allowed to differ. The instance ledger above keeps c = 1,
   so its selection term stays 0. This function is the same term when
   c is the gloss fecundity and z is 1 for an admitted gloss. */
inline double priceSelectionOf(const std::vector<double> &fitness,
                               const std::vector<double> &trait) {
  if (fitness.empty() || fitness.size() != trait.size())
    return 0.0;
  const double n = static_cast<double>(fitness.size());
  double meanC = 0.0;
  double meanZ = 0.0;
  for (size_t i = 0; i < fitness.size(); ++i) {
    meanC += fitness[i];
    meanZ += trait[i];
  }
  meanC /= n;
  meanZ /= n;
  if (!(meanC > 1e-12))
    return 0.0;
  double cov = 0.0;
  for (size_t i = 0; i < fitness.size(); ++i)
    cov += (fitness[i] - meanC) * (trait[i] - meanZ);
  cov /= n;
  return cov / meanC;
}

/* When every parent copies its own trait, the replicator map and Price's
   selection term are the same number. Initial frequency is uniform.
   mean(z)' - mean(z) = cov(c, z) / mean(c). A mismatched input returns 0.
   The instance ledger still uses c = 1, so this does not change that log. */
inline double faithfulMeanShift(const std::vector<double> &fitness,
                                const std::vector<double> &trait) {
  if (fitness.empty() || fitness.size() != trait.size())
    return 0.0;
  const double n = static_cast<double>(fitness.size());
  std::vector<double> frequency(fitness.size(), 1.0 / n);
  const std::vector<double> next = replicatorStep(frequency, fitness);
  double before = 0.0;
  double after = 0.0;
  for (size_t i = 0; i < trait.size(); ++i) {
    before += frequency[i] * trait[i];
    after += next[i] * trait[i];
  }
  return after - before;
}

/* Continuous replicator equation (Taylor and Jonker 1978; Nowak 2006,
   eq. 5.1): with variant frequencies p_i and constant fitness f_i,
       dp_i/dt = p_i (f_i - fbar),   fbar = sum_j p_j f_j.
   The right-hand side keeps sum_i p_i invariant, so the simplex is
   forward invariant.

   Solution. Write q_i(t) = p_i(t) exp(-f_i t). Then
       dq_i/dt = exp(-f_i t) (dp_i/dt - f_i p_i)
               = -fbar(t) q_i,
   so q_i(t) = q_i(0) exp(-int_0^t fbar) and
       p_i(t) = p_i(0) e^{f_i t} / sum_j p_j(0) e^{f_j t}.        (R1)
   sum_i p_i(t) = 1 for every t, and p_i(t) > 0 iff p_i(0) > 0.

   Two consequences used by the live code.
   (a) The discrete map of this file, p'_i = p_i w_i / sum_j p_j w_j,
       is exactly (R1) at t = 1 with w_i = e^{f_i}. Keeping the map is
       therefore the unit-time flow of the continuous equation; the
       recursive use f -> f^2 -> f^3 is the flow at t = 1, 2, 3, not a
       new model.
   (b) mean fitness is nondecreasing:
       d fbar/dt = sum_i f_i dp_i/dt
                 = sum_i f_i p_i (f_i - fbar)
                 = sum_i p_i (f_i - fbar)^2 = Var_p(f) >= 0,
       with equality iff every f_i with p_i > 0 equals fbar (Fisher
       1930). The flow moves mass toward the fittest variants; a
       zero-frequency variant with the largest fitness stays at 0
       because (R1) keeps its support.

   These are properties of the equation, not of the index: a class id
   or a word weight is not a variant frequency. */
inline std::vector<double> replicatorVelocity(const std::vector<double> &frequency,
                                              const std::vector<double> &fitness) {
  std::vector<double> out(frequency.size(), 0.0);
  if (frequency.size() != fitness.size() || frequency.empty())
    return out;
  double mass = 0.0;
  double mean = 0.0;
  for (size_t i = 0; i < frequency.size(); ++i) {
    if (frequency[i] <= 0.0)
      continue;
    mass += frequency[i];
    mean += frequency[i] * fitness[i];
  }
  if (!(mass > 0.0))
    return out;
  mean /= mass;
  for (size_t i = 0; i < frequency.size(); ++i) {
    if (frequency[i] <= 0.0)
      continue;
    out[i] = frequency[i] * (fitness[i] - mean);
  }
  return out;
}

/* (R1) with f_i = ln(w_i). At t = 1 this equals replicatorStep(freq, w)
   exactly (up to floating point), which is the unit-time flow identity
   used by the recall ordering. Non-positive weights or frequencies are
   dropped from the support; a zero total mass returns the input. */
inline std::vector<double> replicatorFlow(const std::vector<double> &frequency,
                                          const std::vector<double> &fitness,
                                          double t) {
  if (frequency.size() != fitness.size() || frequency.empty())
    return frequency;
  std::vector<double> weight(frequency.size(), 0.0);
  double mass = 0.0;
  for (size_t i = 0; i < frequency.size(); ++i) {
    if (!(frequency[i] > 0.0) || !(fitness[i] > 0.0))
      continue;
    weight[i] = std::exp(t * std::log(fitness[i]));
    mass += frequency[i] * weight[i];
  }
  if (!(mass > 0.0) || !std::isfinite(mass))
    return frequency;
  std::vector<double> next(frequency.size(), 0.0);
  for (size_t i = 0; i < frequency.size(); ++i) {
    if (!(frequency[i] > 0.0) || !(fitness[i] > 0.0))
      continue;
    next[i] = frequency[i] * weight[i] / mass;
  }
  return next;
}

/* Force identification from a frequency series.

   One snapshot cannot name the force: content bias p' = p + B p(1-p)
   and conformity p' = p + D p(1-p)(2p-1) coincide for one step when
   B = D(2p-1). A second step separates them because conformity's
   factor (2p-1) moves with p while content's B does not.

   Fit each map's parameter on the first transition, then predict every
   later sample and take the largest absolute residual. The replicator
   map p' = p w / (p w + 1 - p) is the third candidate (w > 0 from one
   step). Labels are awarded only when the best residual is at most
   kForceResidual and the runner-up is separated by kForceMargin;
   otherwise the series is undetermined. A constant series is fixed,
   not a force. This is a decision rule on observed frequencies, not an
   admission test. */
inline constexpr double kForceResidual = 0.01;
inline constexpr double kForceMargin = 1e-6;

struct ForceFit {
  std::string force{"undetermined"};
  int samples{0};
  double residualContent{0.0};
  double residualConformity{0.0};
  double residualReplicator{0.0};
  double contentBias{0.0};
  double conformity{0.0};
  double logFitness{0.0};
  bool separated{false};
  bool constant{false};
};

inline ForceFit identifyForce(const std::vector<double> &frequency) {
  ForceFit out;
  out.samples = static_cast<int>(frequency.size());
  if (frequency.size() < 3)
    return out;
  const double p0 = frequency[0];
  const double p1 = frequency[1];
  const double p2 = frequency[2];
  bool allSame = true;
  for (size_t i = 1; i < frequency.size(); ++i) {
    if (frequency[i] != p0)
      allSame = false;
  }
  if (allSame) {
    out.constant = true;
    out.force = "fixed";
    out.separated = true;
    return out;
  }
  if (!(p0 > 0.0 && p0 < 1.0 && p1 > 0.0 && p1 < 1.0 && p2 > 0.0 && p2 < 1.0))
    return out;
  for (size_t i = 1; i < frequency.size(); ++i) {
    if (!(frequency[i] > 0.0 && frequency[i] < 1.0))
      return out;
  }
  const double d0 = p0 * (1.0 - p0);
  out.contentBias = (p1 - p0) / d0;
  /* Replicator: p1 = p0 w / (p0 w + 1 - p0) gives w. */
  const double w = (p1 * (1.0 - p0)) / (p0 * (1.0 - p1));
  out.logFitness = std::log(w);
  const bool conformityDefined = std::fabs(2.0 * p0 - 1.0) > 1e-12;
  if (conformityDefined)
    out.conformity = (p1 - p0) / (d0 * (2.0 * p0 - 1.0));
  double predContent = p1;
  double predConformity = p1;
  double predReplicator = p1;
  double worstContent = 0.0;
  double worstConformity = 0.0;
  double worstReplicator = 0.0;
  for (size_t i = 1; i < frequency.size(); ++i) {
    if (i > 1) {
      const double actual = frequency[i];
      worstContent = std::max(worstContent, std::fabs(actual - predContent));
      worstConformity =
          std::max(worstConformity, std::fabs(actual - predConformity));
      worstReplicator =
          std::max(worstReplicator, std::fabs(actual - predReplicator));
    }
    predContent = contentBiasStep(predContent, out.contentBias);
    if (conformityDefined && std::fabs(out.conformity) < 2.0)
      predConformity = conformityStep(predConformity, out.conformity);
    else
      predConformity = 2.0; /* outside the three-role-model domain */
    predReplicator = (predReplicator * w) / (predReplicator * w +
                                             (1.0 - predReplicator));
  }
  out.residualContent = worstContent;
  out.residualConformity =
      (conformityDefined && std::fabs(out.conformity) < 2.0)
          ? worstConformity
          : frequency.size() + 1.0;
  out.residualReplicator = worstReplicator;
  struct Candidate {
    const char *name;
    double residual;
  };
  const Candidate candidates[] = {{"content", out.residualContent},
                                  {"conformity", out.residualConformity},
                                  {"replicator", out.residualReplicator}};
  int best = 0;
  for (int i = 1; i < 3; ++i) {
    if (candidates[i].residual < candidates[best].residual)
      best = i;
  }
  double second = 1.0e9;
  for (int i = 0; i < 3; ++i) {
    if (i == best)
      continue;
    if (candidates[i].residual < second)
      second = candidates[i].residual;
  }
  out.separated = (second - candidates[best].residual) > kForceMargin;
  if (candidates[best].residual <= kForceResidual && out.separated)
    out.force = candidates[best].name;
  return out;
}

/* Population frequency series from a trace: at each tick, the share of
   carriers seen so far whose latest tick is a source edit. A trait that
   never flips gives a constant series, which identifyForce calls fixed.
   Fewer than three ticks gives undetermined. */
inline std::vector<double> populationFrequencySeries(
    const std::vector<ReplicaTick> &ticks) {
  std::vector<double> out;
  bool anyGeneration = false;
  for (const auto &t : ticks)
    if (t.generation > 0) {
      anyGeneration = true;
      break;
    }
  if (anyGeneration) {
    /* One frequency per generation over the FIXED population: at each
       generation the share of all carriers whose latest tick is a source
       edit.  An absent carrier keeps its previous state; the denominator is
       the population, not the ticks that happened to land in the round.
       (The old on/present denominator made a round with 4 of 8 ticks look
       like p=1 and broke the content schedule.) */
    std::vector<std::string> order;
    std::unordered_map<std::string, bool> latest;
    std::map<int, std::vector<const ReplicaTick *>> byGen;
    for (const auto &t : ticks) {
      if (t.generation <= 0) continue;
      if (latest.find(t.instance) == latest.end()) {
        order.push_back(t.instance);
        latest[t.instance] = false;
      }
      byGen[t.generation].push_back(&t);
    }
    for (const auto &kv : byGen) {
      for (const ReplicaTick *t : kv.second) latest[t->instance] = t->sourceEdit;
      int on = 0;
      for (const auto &name : order)
        if (latest[name]) ++on;
      out.push_back(static_cast<double>(on) /
                    static_cast<double>(order.size()));
    }
    return out;
  }
  std::vector<std::string> order;
  std::unordered_map<std::string, bool> latest;
  /* Pre-scan the batch so every carrier participates from the first
     frequency point.  Without this the first tick has order size 1 and
     p_0 is always 0 or 1, so identifyForce can never see the interior
     first three points it requires. */
  for (const auto &t : ticks) {
    if (latest.find(t.instance) == latest.end()) {
      order.push_back(t.instance);
      latest[t.instance] = false;
    }
  }
  for (const auto &t : ticks) {
    latest[t.instance] = t.sourceEdit;
    int on = 0;
    for (const auto &name : order) {
      auto it = latest.find(name);
      if (it != latest.end() && it->second)
        ++on;
    }
    out.push_back(static_cast<double>(on) /
                  static_cast<double>(order.size()));
  }
  return out;
}

/* Acerbi, Charbonneau, Miton, Scott-Phillips 2021 transmission regime
   applied to a lineage: the input-output distance series is the
   judgeCopy fidelity of each consecutive carrier pair. The regime is
   an advisory population label; it does not admit a meme and it does
   not replace judgeCopy. Fewer than six steps is undetermined. */
inline TransmissionRegime lineageRegime(
    const std::vector<std::string> &carriers) {
  std::vector<double> steps;
  for (size_t i = 1; i < carriers.size(); ++i)
    steps.push_back(judgeCopy(carriers[i - 1], carriers[i]).fidelity);
  return judgeTransmissionRegime(steps);
}

inline std::string replicaJsonEscape(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') {
      out.push_back('\\');
      out.push_back(static_cast<char>(c));
    } else if (c < 0x20) {
      out.push_back(' ');
    } else {
      out.push_back(static_cast<char>(c));
    }
  }
  return out;
}

inline bool readReplicaLine(const std::string &line, ReplicaTick *tick) {
  if (!tick)
    return false;
  auto field = [&](const char *key) {
    const std::string pat = std::string("\"") + key + "\":\"";
    const auto at = line.find(pat);
    if (at == std::string::npos)
      return std::string();
    const auto start = at + pat.size();
    std::string value;
    for (size_t i = start; i < line.size(); ++i) {
      if (line[i] == '\\' && i + 1 < line.size()) {
        value.push_back(line[++i]);
        continue;
      }
      if (line[i] == '"')
        break;
      value.push_back(line[i]);
    }
    return value;
  };
  tick->instance = field("instance");
  tick->mission = field("mission");
  tick->explore = field("explore");
  tick->summon = field("summon");
  tick->replyClass = field("replyClass");
  tick->sourceEdit = line.find("\"sourceEdit\":1") != std::string::npos;
  tick->replicator = line.find("\"replicator\":1") != std::string::npos;
  const auto genAt = line.find("\"generation\":");
  if (genAt != std::string::npos)
    tick->generation = std::atoi(line.c_str() + genAt + 13);
  const auto fidAt = line.find("\"copyFidelity\":");
  if (fidAt != std::string::npos)
    tick->copyFidelity = std::atof(line.c_str() + fidAt + 15);
  const auto idxAt = line.find("\"indexOverlap\":");
  if (idxAt != std::string::npos)
    tick->indexOverlap = std::atof(line.c_str() + idxAt + 15);
  if (tick->instance.empty())
    return false;
  return true;
}

inline std::vector<ReplicaTick> readReplicaTicks(const std::string &path,
                                                int cap) {
  std::vector<ReplicaTick> out;
  std::ifstream in(path);
  if (!in)
    return out;
  std::string line;
  while (std::getline(in, line)) {
    ReplicaTick tick;
    if (readReplicaLine(line, &tick))
      out.push_back(std::move(tick));
  }
  if (cap > 0 && static_cast<int>(out.size()) > cap) {
    out.erase(out.begin(), out.end() - cap);
  }
  return out;
}

/* One writer at a time: two mission threads appending to the same trace
   dir raced the read-modify-write of account.json and could drop a
   carrier's .jsonl from the account (observed live: force=fixed at 3
   samples, then samples reset to 1). */
inline std::mutex &replicaTraceMu() {
  static std::mutex m;
  return m;
}

inline ReplicaAccount appendReplicaTick(const std::string &dir,
                                        const ReplicaTick &tick) {
  std::lock_guard<std::mutex> traceLock(replicaTraceMu());
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  const std::string safeInstance =
      tick.instance.empty() ? std::string("unknown") : tick.instance;
  /* Windows filenames reject ':' (and friends).  A carrier id such as
     "scope:peer-a" silently failed to persist, so the account stayed at
     0 ticks even though the trace line printed.  Sanitize the FILE STEM
     only; the JSON "instance" field keeps the original id. */
  std::string fileStem = safeInstance;
  for (char &c : fileStem) {
    const unsigned char u = static_cast<unsigned char>(c);
    if (!(std::isalnum(u) || c == '-' || c == '_' || c == '.')) c = '_';
  }
  const std::string path = dir + "/" + fileStem + ".jsonl";
  {
    std::ofstream out(path, std::ios::app);
    if (out) {
      out << "{\"instance\":\"" << replicaJsonEscape(safeInstance)
          << "\",\"mission\":\"" << replicaJsonEscape(tick.mission)
          << "\",\"explore\":\"" << replicaJsonEscape(tick.explore)
          << "\",\"summon\":\"" << replicaJsonEscape(tick.summon)
          << "\",\"replyClass\":\"" << replicaJsonEscape(tick.replyClass)
          << "\",\"replyChars\":" << tick.replyChars
          << ",\"sourceEdit\":" << (tick.sourceEdit ? 1 : 0)
          << ",\"generation\":" << tick.generation
          << ",\"projectScore\":" << tick.projectScore
          << ",\"copyFidelity\":" << tick.copyFidelity
          << ",\"indexOverlap\":" << tick.indexOverlap
          << ",\"replicator\":" << (tick.replicator ? 1 : 0)
          << ",\"attractionBias\":" << tick.attractionBias
          << ",\"peerCopies\":" << tick.peerCopies
          << ",\"mechanism\":\"" << replicaJsonEscape(
                 tick.mechanism.empty() ? std::string("undetermined")
                                        : tick.mechanism)
          << "\"}\n";
    }
  }
  std::vector<ReplicaTick> all;
  std::error_code dec;
  if (std::filesystem::is_directory(dir, dec)) {
    for (const auto &ent : std::filesystem::directory_iterator(dir, dec)) {
      if (dec || !ent.is_regular_file())
        continue;
      const std::string name = ent.path().filename().string();
      if (name.size() < 7 || name.compare(name.size() - 6, 6, ".jsonl") != 0)
        continue;
      auto part = readReplicaTicks(ent.path().string(), 32);
      all.insert(all.end(), part.begin(), part.end());
    }
  }
  const ReplicaAccount base = accountReplicas(all, 2.0);
  ReplicaAccount account = base;
  const auto series = populationFrequencySeries(all);
  const auto fit = identifyForce(series);
  account.force = fit.force;
  account.forceSamples = fit.samples;
  account.forceResidual = std::min(
      fit.residualContent,
      std::min(fit.residualConformity, fit.residualReplicator));
  account.forceContent = fit.residualContent;
  account.forceConformity = fit.residualConformity;
  account.forceReplicator = fit.residualReplicator;
  std::ofstream summary(dir + "/account.json", std::ios::trunc);
  if (summary) {
    summary << "{\"ticks\":" << account.ticks
            << ",\"sourceEdits\":" << account.sourceEdits
            << ",\"replicators\":" << account.replicators
            << ",\"fidelity\":" << account.fidelity
            << ",\"eigenLmax\":" << account.eigenLmax
            << ",\"priceSelection\":" << account.priceSelection
            << ",\"priceTransmission\":" << account.priceTransmission
            << ",\"force\":\"" << replicaJsonEscape(account.force)
            << "\",\"forceSamples\":" << account.forceSamples
            << ",\"forceResidual\":" << account.forceResidual
            << ",\"binding\":\"" << replicaJsonEscape(account.binding)
            << "\"}\n";
  }
  return account;
}

/* Least-ambiguous text carrier.

   Forward (construction): one intent is one canonical sentence
     ask <tool> about <argument> end
   The argument is a sequence of words the copy tokenizer keeps
   (alphanumeric, length >= 3). The frame words ask / about / end
   and the tool token are reserved and do not appear inside the
   argument. end is a terminator, so a shorter argument is not a
   contiguous prefix of a longer one.

   Two different canonical sentences of length n share a contiguous
   token span of at most n-1. Their fidelity is then at most
   (n-1)/n. Eigen's floor is 2^(-1/n) = exp(-ln 2 / n), which is
   strictly above 1 - (ln 2)/n and therefore strictly above
   (n-1)/n. Distinct canonicals are not replicators of each other.
   The same sentence copied into a longer offspring (a prefix or a
   suffix wrapped around it) still is.

   Backward (recognition): return the tool whose canonical sentence
   is a replicator parent of the received text, and only when that
   parent is unique. Zero parents or two parents both return none.
   This is the inverse of construction. It does not search for a
   tool name as a substring. A unique hit decodes an intent. It
   does not admit a meme: admission still requires fecundity and
   longevity on a lineage.

   The frame is a sufficient code only when both sides already use
   it. The tool module may not require the base model to emit it.
   The base is not the recognizer and is not the constructor. The
   tool module has to read an unconstrained sentence and call a
   tool when one of the sentence's readings is a tool call.
   recognizeCarrier and recognizeConstructed only decode this code. */
struct CarrierIntent {
  std::string tool;
  std::string argument;
};

inline std::string constructCarrier(const std::string &tool,
                                    const std::string &argument) {
  return "ask " + tool + " about " + argument + " end";
}

inline std::string recognizeCarrier(const std::string &text,
                                    const std::vector<CarrierIntent> &catalog) {
  std::string hit;
  int matches = 0;
  for (const auto &item : catalog) {
    if (!judgeCopy(constructCarrier(item.tool, item.argument), text).replicator)
      continue;
    ++matches;
    if (matches > 1)
      return "none";
    hit = item.tool;
  }
  return matches == 1 ? hit : std::string("none");
}

/* Catalog-free inverse of the same construction. A tool is recognized
   only when the received tokens contain exactly one contiguous span
   ask <tool> about <four argument words> end. The argument words are
   not frame words. A mention, a reversal, or a sentence that drops
   the terminator has no such span, so the result is none. This is
   the carrier grammar, not a scan for a tool name wherever it sits. */
inline std::string recognizeConstructed(const std::string &text) {
  const auto tok = copyTokens(text);
  const char *tools[] = {"math", "search", "research", "replicate"};
  std::string hit;
  int matches = 0;
  if (tok.size() < 8)
    return "none";
  for (size_t i = 0; i + 7 < tok.size(); ++i) {
    if (tok[i] != "ask" || tok[i + 2] != "about" || tok[i + 7] != "end")
      continue;
    bool reserved = false;
    for (int k = 3; k <= 6; ++k) {
      const std::string &word = tok[i + static_cast<size_t>(k)];
      if (word == "ask" || word == "about" || word == "end")
        reserved = true;
    }
    if (reserved)
      continue;
    std::string tool;
    for (const char *name : tools) {
      if (tok[i + 1] == name)
        tool = name;
    }
    if (tool.empty())
      continue;
    ++matches;
    if (matches > 1)
      return "none";
    hit = tool;
  }
  return matches == 1 ? hit : std::string("none");
}

}  // namespace memetic
}  // namespace phoenix
