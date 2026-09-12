#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace phoenix {

/* aiCount is the hard total. groupCount/groupSize only partition that budget.
   AI_COUNT=1 must not become 3 or 7 just because json defaults say so. */
struct ControllerLayout {
  int groupCount{0};
  int totalControllers{0};
  std::vector<int> groupSizes;
};

inline int clampPositiveCount(int raw, int fallback = 1) {
  if (raw < 1)
    return fallback < 1 ? 1 : fallback;
  return raw;
}

inline ControllerLayout planControllerLayout(int aiCount, int groupCount,
                                             int groupSize) {
  ControllerLayout out;
  const int n = clampPositiveCount(aiCount, 1);
  const int gc = clampPositiveCount(groupCount, 1);
  const int gs = clampPositiveCount(groupSize, 1);
  int remaining = n;
  for (int i = 0; i < gc && remaining > 0; ++i) {
    const int take = std::min(gs, remaining);
    out.groupSizes.push_back(take);
    remaining -= take;
  }
  out.groupCount = static_cast<int>(out.groupSizes.size());
  out.totalControllers = n - remaining;
  return out;
}

/* Unbounded per-key hit counters on the chat/graph path. */
class BoundedHitMap {
public:
  explicit BoundedHitMap(std::size_t maxKeys = 4096) : maxKeys_(maxKeys) {}

  int increment(const std::string &key) {
    auto it = hits_.find(key);
    if (it != hits_.end()) {
      ++it->second;
      return it->second;
    }
    if (hits_.size() >= maxKeys_)
      hits_.clear();
    hits_[key] = 1;
    return 1;
  }

  std::size_t size() const { return hits_.size(); }
  void clear() { hits_.clear(); }

private:
  std::size_t maxKeys_;
  std::unordered_map<std::string, int> hits_;
};

/* llamaEmbDim=4096 + JL default nz=target/3 => millions of triplets (~GiB).
   Keep one projection under kMaxProjectionTriplets. */
inline constexpr std::size_t kMaxProjectionDim = 4096;
inline constexpr std::size_t kMaxProjectionTriplets = 65536;
inline constexpr std::size_t kDefaultSparseNonZeros = 3;

inline std::size_t boundedProjectionNonZeros(std::size_t sourceDim,
                                             std::size_t targetDim,
                                             std::size_t requested = 0) {
  const std::size_t src =
      sourceDim < 1 ? 1 : std::min(sourceDim, kMaxProjectionDim);
  const std::size_t dst =
      targetDim < 1 ? 1 : std::min(targetDim, kMaxProjectionDim);
  std::size_t nz = requested == 0 ? std::max(kDefaultSparseNonZeros, dst / 3)
                                  : requested;
  if (nz > dst)
    nz = dst;
  const std::size_t maxNz = kMaxProjectionTriplets / src;
  if (nz > maxNz)
    nz = maxNz < 1 ? 1 : maxNz;
  return nz;
}

} // namespace phoenix
