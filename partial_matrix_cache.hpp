/* partial_matrix_cache.hpp - Optional partial result cache for matrix / graph sub-blocks
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

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <list>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace phoenix::cache {

/* Configuration for a partial result cache.
   enableCorrection=true keeps a representative fingerprint with each entry so
   callers can apply a first-order correction when the current input is
   quantised to the same bin but not exactly identical. */
struct PartialMatrixCacheConfig {
    bool enabled{true};
    std::size_t maxEntries{1024};
    std::size_t ttlMs{0};
    /* tolerance > 0 quantises continuous values before hashing.  Inputs that
       differ by less than tolerance share the same cache key and therefore
       reuse the same cached result (with optional correction). */
    double tolerance{0.0};
    bool enableCorrection{true};
    double correctionScale{1.0};
    /* When fingerprinting large matrices, sample at most this many rows/cols
       to keep keys compact.  Set to 0 to fingerprint the whole block. */
    std::size_t maxBlockSamples{8};
};

/* Lightweight FNV-1a helper. */
inline std::uint64_t fnv1a64(const void *data, std::size_t len,
                             std::uint64_t h = 1469598103934665603ull) {
    const auto *bytes = static_cast<const unsigned char *>(data);
    for (std::size_t i = 0; i < len; ++i) {
        h ^= static_cast<std::uint64_t>(bytes[i]);
        h *= 1099511628211ull;
    }
    return h;
}

inline std::uint64_t fnv1a64String(std::string_view s,
                                    std::uint64_t h = 1469598103934665603ull) {
    return fnv1a64(s.data(), s.size(), h);
}

/* Generic partial result cache.  T is usually double, std::vector<double>,
   or std::vector<std::vector<double>>.  Keys are opaque strings built by the
   caller; this class only handles storage, eviction, expiry and optional
   correction. */
template <typename T>
class PartialMatrixCache {
public:
    using Config = PartialMatrixCacheConfig;

    explicit PartialMatrixCache(const Config &cfg = Config{}) : cfg_(cfg) {}

    bool enabled() const { return cfg_.enabled; }
    void setEnabled(bool on) { cfg_.enabled = on; }

    const Config &config() const { return cfg_; }
    void setConfig(const Config &cfg) { std::lock_guard<std::mutex> lock(mu_); cfg_ = cfg; }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mu_);
        return data_.size();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mu_);
        data_.clear();
        queue_.clear();
    }

    /* Try to retrieve a value.  If correction is enabled and a fingerprint
       is supplied, the cached value is adjusted by the stored fingerprint. */
    bool get(const std::string &key, T &out,
             const std::vector<double> *currentFingerprint = nullptr) const {
        if (!cfg_.enabled)
            return false;
        std::lock_guard<std::mutex> lock(mu_);
        evictExpiredLocked();
        auto it = data_.find(key);
        if (it == data_.end())
            return false;
        if (cfg_.ttlMs > 0 && nowMs() >= it->second.expires) {
            data_.erase(it);
            return false;
        }
        it->second.lastAccess = ++clock_;
        out = it->second.value;
        if (cfg_.enableCorrection && currentFingerprint &&
            !it->second.fingerprint.empty()) {
            applyCorrection(out, *currentFingerprint, it->second.fingerprint);
        }
        return true;
    }

    void set(const std::string &key, const T &value,
             const std::vector<double> &fingerprint = {}) {
        if (!cfg_.enabled)
            return;
        std::lock_guard<std::mutex> lock(mu_);
        evictLocked();
        auto it = data_.find(key);
        if (it == data_.end()) {
            queue_.push_back(key);
            it = data_.emplace(key, Entry{}).first;
        }
        it->second.value = value;
        it->second.fingerprint = fingerprint;
        it->second.lastAccess = ++clock_;
        it->second.expires =
            (cfg_.ttlMs > 0) ? (nowMs() + static_cast<int64_t>(cfg_.ttlMs))
                             : std::numeric_limits<int64_t>::max();
    }

    /* Fingerprint a 1-D vector.  With tolerance > 0 values are quantised to
       the nearest tolerance grid before hashing, so nearly-identical vectors
       map to the same key. */
    static std::string fingerprintVector(const std::vector<double> &v,
                                         double tolerance = 0.0,
                                         std::size_t maxSamples = 0) {
        std::size_t step = 1;
        if (maxSamples > 0 && v.size() > maxSamples) {
            step = std::max<std::size_t>(1, v.size() / maxSamples);
        }
        std::uint64_t h = 1469598103934665603ull;
        for (std::size_t i = 0; i < v.size(); i += step) {
            double q = v[i];
            if (tolerance > 0.0) {
                q = std::llround(q / tolerance) * tolerance;
            }
            h = fnv1a64(&q, sizeof(q), h);
        }
        // Mix-in length so [a] and [a,0] do not collide.
        std::size_t n = v.size();
        h = fnv1a64(&n, sizeof(n), h);
        std::ostringstream oss;
        oss << h;
        return oss.str();
    }

    /* Float overload for Transformer embeddings. */
    static std::string fingerprintVector(const std::vector<float> &v,
                                         double tolerance = 0.0,
                                         std::size_t maxSamples = 0) {
        std::size_t step = 1;
        if (maxSamples > 0 && v.size() > maxSamples) {
            step = std::max<std::size_t>(1, v.size() / maxSamples);
        }
        std::uint64_t h = 1469598103934665603ull;
        for (std::size_t i = 0; i < v.size(); i += step) {
            double q = static_cast<double>(v[i]);
            if (tolerance > 0.0) {
                q = std::llround(q / tolerance) * tolerance;
            }
            h = fnv1a64(&q, sizeof(q), h);
        }
        // Mix-in length so [a] and [a,0] do not collide.
        std::size_t n = v.size();
        h = fnv1a64(&n, sizeof(n), h);
        std::ostringstream oss;
        oss << h;
        return oss.str();
    }

    /* Fingerprint a rectangular sub-block of a 2-D matrix.  Coordinates are
       clamped to the matrix bounds. */
    static std::string fingerprintMatrixBlock(
        const std::vector<std::vector<double>> &m,
        std::size_t r0,
        std::size_t r1,
        std::size_t c0,
        std::size_t c1,
        double tolerance = 0.0,
        std::size_t maxSamples = 0) {
        if (m.empty())
            return "empty";
        r1 = std::min(r1, m.size());
        if (r0 >= r1)
            r0 = 0;
        std::size_t rowStep = 1;
        if (maxSamples > 0 && (r1 - r0) > maxSamples) {
            rowStep = std::max<std::size_t>(1, (r1 - r0) / maxSamples);
        }
        std::uint64_t h = 1469598103934665603ull;
        for (std::size_t r = r0; r < r1; r += rowStep) {
            if (m[r].empty())
                continue;
            std::size_t endCol = std::min(c1, m[r].size());
            std::size_t startCol = std::min(c0, endCol);
            std::size_t colStep = 1;
            if (maxSamples > 0 && (endCol - startCol) > maxSamples) {
                colStep = std::max<std::size_t>(1, (endCol - startCol) / maxSamples);
            }
            for (std::size_t c = startCol; c < endCol; c += colStep) {
                double q = m[r][c];
                if (tolerance > 0.0) {
                    q = std::llround(q / tolerance) * tolerance;
                }
                h = fnv1a64(&q, sizeof(q), h);
            }
        }
        // Mix dimensions.
        std::size_t dims[4] = {r0, r1, c0, c1};
        h = fnv1a64(dims, sizeof(dims), h);
        std::ostringstream oss;
        oss << h;
        return oss.str();
    }

    /* Float overload for Transformer hidden-state blocks. */
    static std::string fingerprintMatrixBlock(
        const std::vector<std::vector<float>> &m,
        std::size_t r0,
        std::size_t r1,
        std::size_t c0,
        std::size_t c1,
        double tolerance = 0.0,
        std::size_t maxSamples = 0) {
        if (m.empty())
            return "empty";
        r1 = std::min(r1, m.size());
        if (r0 >= r1)
            r0 = 0;
        std::size_t rowStep = 1;
        if (maxSamples > 0 && (r1 - r0) > maxSamples) {
            rowStep = std::max<std::size_t>(1, (r1 - r0) / maxSamples);
        }
        std::uint64_t h = 1469598103934665603ull;
        for (std::size_t r = r0; r < r1; r += rowStep) {
            if (m[r].empty())
                continue;
            std::size_t endCol = std::min(c1, m[r].size());
            std::size_t startCol = std::min(c0, endCol);
            std::size_t colStep = 1;
            if (maxSamples > 0 && (endCol - startCol) > maxSamples) {
                colStep = std::max<std::size_t>(1, (endCol - startCol) / maxSamples);
            }
            for (std::size_t c = startCol; c < endCol; c += colStep) {
                double q = static_cast<double>(m[r][c]);
                if (tolerance > 0.0) {
                    q = std::llround(q / tolerance) * tolerance;
                }
                h = fnv1a64(&q, sizeof(q), h);
            }
        }
        // Mix dimensions.
        std::size_t dims[4] = {r0, r1, c0, c1};
        h = fnv1a64(dims, sizeof(dims), h);
        std::ostringstream oss;
        oss << h;
        return oss.str();
    }

    /* Build a cache key from an operation tag, block coordinates and the
       input fingerprint string. */
    static std::string makeKey(const std::string &op,
                               int blockRow,
                               int blockCol,
                               const std::string &inputFingerprint,
                               const std::string &extra = "") {
        std::ostringstream oss;
        oss << op << "|" << blockRow << "," << blockCol << "|" << inputFingerprint;
        if (!extra.empty())
            oss << "|" << extra;
        return oss.str();
    }

private:
    struct Entry {
        T value{};
        std::vector<double> fingerprint;
        int64_t lastAccess{0};
        int64_t expires{std::numeric_limits<int64_t>::max()};
    };

    static int64_t nowMs() {
        using namespace std::chrono;
        return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).
            count();
    }

    mutable std::mutex mu_;
    Config cfg_;
    mutable std::unordered_map<std::string, Entry> data_;
    mutable std::deque<std::string> queue_;
    mutable int64_t clock_{0};

    void evictExpiredLocked() const {
        if (cfg_.ttlMs == 0)
            return;
        int64_t now = nowMs();
        for (auto qit = queue_.begin(); qit != queue_.end();) {
            auto it = data_.find(*qit);
            if (it == data_.end() || now >= it->second.expires) {
                if (it != data_.end())
                    data_.erase(it);
                qit = queue_.erase(qit);
            } else {
                ++qit;
            }
        }
    }

    void evictLocked() {
        evictExpiredLocked();
        while (data_.size() >= cfg_.maxEntries && !queue_.empty()) {
            auto key = queue_.front();
            auto it = data_.find(key);
            if (it != data_.end())
                data_.erase(it);
            queue_.pop_front();
        }
    }

    /* First-order correction: shift the cached value by the per-element
       difference between the current fingerprint and the stored fingerprint,
       scaled by correctionScale.  This compensates for the small quantisation
       error introduced by tolerance-based cache keys. */
    void applyCorrection(T &value, const std::vector<double> &current,
                         const std::vector<double> &stored) const {
        if constexpr (std::is_same_v<T, double>) {
            if (!current.empty() && !stored.empty()) {
                value += (current[0] - stored[0]) * cfg_.correctionScale;
            }
        } else if constexpr (std::is_same_v<T, float>) {
            if (!current.empty() && !stored.empty()) {
                value += static_cast<float>((current[0] - stored[0]) * cfg_.correctionScale);
            }
        } else if constexpr (std::is_same_v<T, std::vector<double>>) {
            const std::size_t n = std::min({current.size(), stored.size(), value.size()});
            for (std::size_t i = 0; i < n; ++i) {
                value[i] += (current[i] - stored[i]) * cfg_.correctionScale;
            }
        } else if constexpr (std::is_same_v<T, std::vector<float>>) {
            const std::size_t n = std::min({current.size(), stored.size(), value.size()});
            for (std::size_t i = 0; i < n; ++i) {
                value[i] += static_cast<float>((current[i] - stored[i]) * cfg_.correctionScale);
            }
        } else if constexpr (std::is_same_v<T, std::vector<std::vector<double>>>) {
            std::size_t idx = 0;
            const std::size_t maxIdx = std::min(current.size(), stored.size());
            for (std::size_t i = 0; i < value.size() && idx < maxIdx; ++i) {
                for (std::size_t j = 0; j < value[i].size() && idx < maxIdx; ++j, ++idx) {
                    value[i][j] += (current[idx] - stored[idx]) * cfg_.correctionScale;
                }
            }
        } else if constexpr (std::is_same_v<T, std::vector<std::vector<float>>>) {
            std::size_t idx = 0;
            const std::size_t maxIdx = std::min(current.size(), stored.size());
            for (std::size_t i = 0; i < value.size() && idx < maxIdx; ++i) {
                for (std::size_t j = 0; j < value[i].size() && idx < maxIdx; ++j, ++idx) {
                    value[i][j] += static_cast<float>((current[idx] - stored[idx]) * cfg_.correctionScale);
                }
            }
        }
        (void)current;
        (void)stored;
    }
};

} // namespace phoenix::cache

