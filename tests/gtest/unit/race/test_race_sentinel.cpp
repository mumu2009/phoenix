/* test_race_sentinel.cpp
 *
 * RaceSentinel-MinGW (doc/v10.1/methods_backlog.md M1) -- Phase 1 control
 * experiment.
 *
 * WHAT THIS FILE IS
 * -----------------
 * A TEST-ONLY, standalone reproduction of the four race *patterns* that were
 * already fixed in product code (crash_forensics_2026-11-15.md / round_2026-11-15.md
 * section 10.1).  It does NOT touch any product source.  Each pattern is written
 * twice with standalone test code:
 *
 *   positive controls (must be detected):
 *     A) lazy std::vector::push_back while another thread iterates
 *     B) process-global std::string written from two threads
 *     C) bare std::map::operator[] insert racing a reader
 *     D) std::unordered_map::erase racing find
 *
 *   negative controls (must NOT be flagged):
 *     A'..D') the same four patterns, every container access under a
 *              race_sentinel::TrackedMutex.
 *
 * plus the "no instrumentation" reference workload used for the overhead
 * multiplier and the matrix experiment that tools/race_sentinel_run.ps1 drives.
 *
 * SAFETY OF THE HARNESS ITSELF
 * ----------------------------
 * Reproducing real races is UB.  The controls are deliberately shaped so the
 * *harness* cannot corrupt the heap, otherwise the experiment would crash
 * instead of measure:
 *   - vector: capacity is reserved up-front, the writer never reallocates,
 *     so the race is on size()/elements (memory stays allocated);
 *   - string: only short SSO assignments (< 16 chars) -> no heap buffer race;
 *   - map: insertion is bounded to 256 keys, after that only value writes;
 *   - unordered_map: a never-freeing arena allocator keeps erased nodes mapped,
 *     so the erase-vs-find race cannot turn into a use-after-free AV.  The
 *     data race (structural mutation vs traversal) is unchanged.
 *
 * HOW TO RUN
 * ----------
 *   powershell -NoProfile -ExecutionPolicy Bypass -File tools/race_sentinel_run.ps1
 * (the script builds a sidecar gtest binary, runs the correctness tests and the
 *  20-seed matrix, and writes JSON+MD under build/tmp/race_sentinel/).
 *
 * The full gtest suite (compile_gtest.bat) also picks this file up because the
 * build globs every .cpp under the tests/gtest tree; the experiment test is
 * GTEST_SKIP() unless RACE_SENTINEL_MODE is set, so a normal regression run is
 * unaffected.
 */

#include "support/race_track.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <map>
#include <mutex>
#include <new>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace rs = race_sentinel;

namespace {

using Clock = std::chrono::steady_clock;

double msBetween(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// Deterministic per-run jitter source (xorshift64).
struct XorShift {
    std::uint64_t s;
    explicit XorShift(std::uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ull) {}
    std::uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
    std::uint32_t below(std::uint32_t n) {
        return static_cast<std::uint32_t>(next() % n);
    }
};

struct Outcome {
    bool detected = false;
    double first_report_ms = -1.0;
    std::uint64_t reports = 0;
    std::uint64_t sampled = 0;
    double wall_ms = 0.0;
    std::string sample_report;
};

// Runs two racing threads that start together and polls the detector for up to
// `window_ms`.  Positive controls stop at the first report (detection latency);
// negative controls must run the FULL window even if a report appears, otherwise
// a false positive would shorten the exposure used to measure the FP rate.
template <class FA, class FB>
Outcome runConcurrentPair(FA fa, FB fb, int window_ms, bool stop_on_report = true) {
    Outcome o;
    rs::resetAll();
    std::atomic<bool> stop{false};
    std::atomic<int> ready{0};
    const std::uint64_t hb = rs::beginConcurrent();
    const auto t0 = Clock::now();

    auto wrap = [&](const char* label, auto&& body) {
        return rs::spawnWith(
            [&, label]() {
                rs::setThreadLabel(label);
                ready.fetch_add(1, std::memory_order_relaxed);
                while (ready.load(std::memory_order_relaxed) < 2) {
                    std::this_thread::yield();
                }
                body(stop);
            },
            hb);
    };

    std::thread ta = wrap("A", fa);
    std::thread tb = wrap("B", fb);

    while (true) {
        if (rs::reportCount() > 0) {
            if (o.first_report_ms < 0.0) o.first_report_ms = msBetween(t0, Clock::now());
            if (stop_on_report) {
                o.detected = true;
                break;
            }
        }
        if (msBetween(t0, Clock::now()) >= static_cast<double>(window_ms)) break;
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }

    stop.store(true, std::memory_order_relaxed);
    rs::joinTracked(ta);
    rs::joinTracked(tb);

    o.reports = rs::reportCount();
    o.detected = (o.reports > 0);
    o.sampled = rs::sampledAccesses();
    o.wall_ms = msBetween(t0, Clock::now());
    auto rp = rs::reports();
    if (!rp.empty()) {
        char buf[320];
        std::snprintf(buf, sizeof(buf), "%s/%s -> %s/%s", rp[0].label_a ? rp[0].label_a : "?",
                      rp[0].op_a ? rp[0].op_a : "?", rp[0].label_b ? rp[0].label_b : "?",
                      rp[0].op_b ? rp[0].op_b : "?");
        o.sample_report = buf;
    }
    return o;
}

// ---------------------------------------------------------------------------
// Positive controls: standalone reproductions of the four fixed race patterns.
// ---------------------------------------------------------------------------

// (a) lazy vector push_back while another thread iterates.
Outcome positiveVectorPushBack(int seed, int window_ms) {
    rs::RaceTrack<std::vector<int>> v;
    v.reserve(8192);
    for (int i = 0; i < 64; ++i) v.push_back(i);
    return runConcurrentPair(
        [&v, seed](std::atomic<bool>& stop) {
            XorShift rng(static_cast<std::uint64_t>(seed) * 2 + 1);
            std::uint32_t i = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                if (v.size() < 4000) {
                    v.push_back(1);  // lazy growth pattern from the crash report
                } else {
                    v[i % 4000] = static_cast<int>(i);  // keep write pressure
                    ++i;
                }
                if ((rng.next() & 7) == 0) std::this_thread::yield();
            }
        },
        [&v, seed](std::atomic<bool>& stop) {
            XorShift rng(static_cast<std::uint64_t>(seed) * 2 + 2);
            long long sum = 0;
            std::uint32_t guard = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                for (int x : v) {
                    sum += x;
                    if (((++guard) & 31) == 0 && stop.load(std::memory_order_relaxed)) {
                        return;
                    }
                }
                if ((rng.next() & 15) == 0) std::this_thread::yield();
            }
            (void)sum;
        },
        window_ms);
}

// (b) process-global std::string written from two threads.
Outcome positiveGlobalString(int seed, int window_ms) {
    rs::RaceTrack<std::string> g;
    g.assign("initial");
    return runConcurrentPair(
        [&g, seed](std::atomic<bool>& stop) {
            XorShift rng(static_cast<std::uint64_t>(seed) * 3 + 1);
            while (!stop.load(std::memory_order_relaxed)) {
                g.assign("alpha-aa");  // SSO only: no heap buffer race
                if ((rng.next() & 7) == 0) std::this_thread::yield();
            }
        },
        [&g, seed](std::atomic<bool>& stop) {
            XorShift rng(static_cast<std::uint64_t>(seed) * 3 + 2);
            while (!stop.load(std::memory_order_relaxed)) {
                g.assign("bravo-bb");
                if ((rng.next() & 7) == 0) std::this_thread::yield();
            }
        },
        window_ms);
}

// (c) bare std::map::operator[] insert racing a reader.
Outcome positiveMapInsert(int seed, int window_ms) {
    rs::RaceTrack<std::map<int, int>> m;
    for (int i = 0; i < 64; ++i) m[i] = i * 3;
    return runConcurrentPair(
        [&m, seed](std::atomic<bool>& stop) {
            XorShift rng(static_cast<std::uint64_t>(seed) * 5 + 1);
            while (!stop.load(std::memory_order_relaxed)) {
                const int key = static_cast<int>(rng.below(256));
                m[key] = key;  // inserts (bounded, 256 keys) then overwrites
                if ((rng.next() & 15) == 0) std::this_thread::yield();
            }
        },
        [&m, seed](std::atomic<bool>& stop) {
            XorShift rng(static_cast<std::uint64_t>(seed) * 5 + 2);
            long long sum = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                auto it = m.find(7);
                if (it != m.end()) sum += it->second;
                if ((rng.next() & 15) == 0) std::this_thread::yield();
            }
            (void)sum;
        },
        window_ms);
}

// Arena allocator used ONLY so that unordered_map erase cannot free memory a
// concurrent reader is walking (keeps the control from turning into an AV).
template <class T>
class ArenaAlloc {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    template <class U>
    struct rebind {
        using other = ArenaAlloc<U>;
    };

    ArenaAlloc() noexcept = default;
    template <class U>
    ArenaAlloc(const ArenaAlloc<U>&) noexcept {}

    T* allocate(std::size_t n) {
        static constexpr std::size_t kArenaBytes = 64u << 20;
        static char* arena = static_cast<char*>(::operator new(kArenaBytes));
        static std::atomic<std::size_t> offset{0};
        const std::size_t bytes = n * sizeof(T);
        const std::size_t off = offset.fetch_add(bytes, std::memory_order_relaxed);
        if (off + bytes <= kArenaBytes) {
            return reinterpret_cast<T*>(arena + off);
        }
        return static_cast<T*>(::operator new(bytes));  // intentionally never freed
    }
    void deallocate(T*, std::size_t) noexcept {}

    template <class U>
    bool operator==(const ArenaAlloc<U>&) const noexcept {
        return true;
    }
    template <class U>
    bool operator!=(const ArenaAlloc<U>&) const noexcept {
        return false;
    }
};

using NeverFreeUMap =
    std::unordered_map<int, int, std::hash<int>, std::equal_to<int>,
                       ArenaAlloc<std::pair<const int, int>>>;

// (d) unordered_map erase racing find.
Outcome positiveUnorderedErase(int seed, int window_ms) {
    rs::RaceTrack<NeverFreeUMap> u;
    for (int i = 0; i < 128; ++i) u[i] = i;
    return runConcurrentPair(
        [&u, seed](std::atomic<bool>& stop) {
            XorShift rng(static_cast<std::uint64_t>(seed) * 7 + 1);
            while (!stop.load(std::memory_order_relaxed)) {
                const int key = static_cast<int>(rng.below(128));
                u.erase(key);
                u[key] = key;
                if ((rng.next() & 15) == 0) std::this_thread::yield();
            }
        },
        [&u, seed](std::atomic<bool>& stop) {
            XorShift rng(static_cast<std::uint64_t>(seed) * 7 + 2);
            long long sum = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                auto it = u.find(5);
                if (it != u.end()) sum += it->second;
                if ((rng.next() & 15) == 0) std::this_thread::yield();
            }
            (void)sum;
        },
        window_ms);
}

// ---------------------------------------------------------------------------
// Negative controls: the same four patterns, correctly locked.
// ---------------------------------------------------------------------------

Outcome negativeVectorPushBack(int seed, int window_ms) {
    rs::RaceTrack<std::vector<int>> v;
    v.reserve(8192);
    for (int i = 0; i < 64; ++i) v.push_back(i);
    rs::TrackedMutex mu;
    return runConcurrentPair(
        [&v, &mu, seed](std::atomic<bool>& stop) {
            XorShift rng(static_cast<std::uint64_t>(seed) * 11 + 1);
            std::uint32_t i = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                {
                    std::lock_guard<rs::TrackedMutex> g(mu);
                    if (v.size() < 4000) {
                        v.push_back(1);
                    } else {
                        v[i % 4000] = static_cast<int>(i);
                        ++i;
                    }
                }
                if ((rng.next() & 7) == 0) std::this_thread::yield();
            }
        },
        [&v, &mu, seed](std::atomic<bool>& stop) {
            XorShift rng(static_cast<std::uint64_t>(seed) * 11 + 2);
            long long sum = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                std::size_t n = 0;
                {
                    std::lock_guard<rs::TrackedMutex> g(mu);
                    n = v.size();
                }
                for (std::size_t i = 0; i < n; ++i) {
                    if (stop.load(std::memory_order_relaxed)) return;
                    int x = 0;
                    {
                        std::lock_guard<rs::TrackedMutex> g(mu);
                        x = v[i];
                    }
                    sum += x;
                }
                if ((rng.next() & 15) == 0) std::this_thread::yield();
            }
            (void)sum;
        },
        window_ms, false);  // negative control: always run the full window
}

Outcome negativeGlobalString(int seed, int window_ms) {
    rs::RaceTrack<std::string> g;
    g.assign("initial");
    rs::TrackedMutex mu;
    return runConcurrentPair(
        [&g, &mu, seed](std::atomic<bool>& stop) {
            XorShift rng(static_cast<std::uint64_t>(seed) * 13 + 1);
            while (!stop.load(std::memory_order_relaxed)) {
                {
                    std::lock_guard<rs::TrackedMutex> g2(mu);
                    g.assign("alpha-aa");
                }
                if ((rng.next() & 7) == 0) std::this_thread::yield();
            }
        },
        [&g, &mu, seed](std::atomic<bool>& stop) {
            XorShift rng(static_cast<std::uint64_t>(seed) * 13 + 2);
            while (!stop.load(std::memory_order_relaxed)) {
                {
                    std::lock_guard<rs::TrackedMutex> g2(mu);
                    g.assign("bravo-bb");
                }
                if ((rng.next() & 7) == 0) std::this_thread::yield();
            }
        },
        window_ms, false);  // negative control: always run the full window
}

Outcome negativeMapInsert(int seed, int window_ms) {
    rs::RaceTrack<std::map<int, int>> m;
    for (int i = 0; i < 64; ++i) m[i] = i * 3;
    rs::TrackedMutex mu;
    return runConcurrentPair(
        [&m, &mu, seed](std::atomic<bool>& stop) {
            XorShift rng(static_cast<std::uint64_t>(seed) * 17 + 1);
            while (!stop.load(std::memory_order_relaxed)) {
                const int key = static_cast<int>(rng.below(256));
                {
                    std::lock_guard<rs::TrackedMutex> g(mu);
                    m[key] = key;
                }
                if ((rng.next() & 15) == 0) std::this_thread::yield();
            }
        },
        [&m, &mu, seed](std::atomic<bool>& stop) {
            XorShift rng(static_cast<std::uint64_t>(seed) * 17 + 2);
            long long sum = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                {
                    std::lock_guard<rs::TrackedMutex> g(mu);
                    auto it = m.find(7);
                    if (it != m.end()) sum += it->second;
                }
                if ((rng.next() & 15) == 0) std::this_thread::yield();
            }
            (void)sum;
        },
        window_ms, false);  // negative control: always run the full window
}

Outcome negativeUnorderedErase(int seed, int window_ms) {
    rs::RaceTrack<NeverFreeUMap> u;
    for (int i = 0; i < 128; ++i) u[i] = i;
    rs::TrackedMutex mu;
    return runConcurrentPair(
        [&u, &mu, seed](std::atomic<bool>& stop) {
            XorShift rng(static_cast<std::uint64_t>(seed) * 19 + 1);
            while (!stop.load(std::memory_order_relaxed)) {
                const int key = static_cast<int>(rng.below(128));
                {
                    std::lock_guard<rs::TrackedMutex> g(mu);
                    u.erase(key);
                    u[key] = key;
                }
                if ((rng.next() & 15) == 0) std::this_thread::yield();
            }
        },
        [&u, &mu, seed](std::atomic<bool>& stop) {
            XorShift rng(static_cast<std::uint64_t>(seed) * 19 + 2);
            long long sum = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                {
                    std::lock_guard<rs::TrackedMutex> g(mu);
                    auto it = u.find(5);
                    if (it != u.end()) sum += it->second;
                }
                if ((rng.next() & 15) == 0) std::this_thread::yield();
            }
            (void)sum;
        },
        window_ms, false);  // negative control: always run the full window
}

// ---------------------------------------------------------------------------
// Experiment matrix (driven by tools/race_sentinel_run.ps1)
// ---------------------------------------------------------------------------

struct Pattern {
    const char* id;
    const char* name;
    Outcome (*positive)(int, int);
    Outcome (*negative)(int, int);
};

const Pattern kPatterns[4] = {
    {"A", "vector lazy push_back vs concurrent iteration", positiveVectorPushBack,
     negativeVectorPushBack},
    {"B", "process-global std::string written from two threads", positiveGlobalString,
     negativeGlobalString},
    {"C", "std::map::operator[] insert vs concurrent reader", positiveMapInsert,
     negativeMapInsert},
    {"D", "std::unordered_map erase vs concurrent find", positiveUnorderedErase,
     negativeUnorderedErase},
};

int envInt(const char* name, int fallback) {
    const char* v = std::getenv(name);
    if (!v || !*v) return fallback;
    return static_cast<int>(std::strtol(v, nullptr, 10));
}

std::string jsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c; break;
        }
    }
    return out;
}

std::string isoNow() {
    const std::time_t t = std::time(nullptr);
    std::tm tmv{};
#if defined(_WIN32)
    localtime_s(&tmv, &t);
#else
    localtime_r(&t, &tmv);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tmv);
    return buf;
}

std::string runJson(const Pattern& p, int seed, const Outcome& o, bool positive) {
    std::ostringstream os;
    os << "{";
    os << "\"pattern\":\"" << p.id << "\",";
    os << "\"pattern_name\":\"" << jsonEscape(p.name) << "\",";
    os << "\"kind\":\"" << (positive ? "positive" : "negative") << "\",";
    os << "\"seed\":" << seed << ",";
    os << "\"detected\":" << (o.detected ? "true" : "false") << ",";
    os << "\"reports\":" << o.reports << ",";
    os << "\"first_report_ms\":" << (o.first_report_ms < 0 ? -1.0 : o.first_report_ms) << ",";
    os << "\"sampled\":" << o.sampled << ",";
    os << "\"wall_ms\":" << o.wall_ms;
    if (!o.sample_report.empty()) {
        os << ",\"sample_report\":\"" << jsonEscape(o.sample_report) << "\"";
    }
    os << "}";
    return os.str();
}

std::string runPositiveMatrix() {
    const int seedFirst = envInt("RACE_SENTINEL_SEED_FIRST", 0);
    const int seedCount = envInt("RACE_SENTINEL_SEED_COUNT", 20);
    const int windowMs = envInt("RACE_SENTINEL_WINDOW_MS", 500);
    std::ostringstream os;
    os << "{\"mode\":\"positive\",\"generated\":\"" << isoNow() << "\",";
    os << "\"detector\":{\"active\":" << (rs::active() ? "true" : "false")
       << ",\"sampling\":" << rs::samplingRate() << "},";
    os << "\"config\":{\"seed_first\":" << seedFirst << ",\"seed_count\":" << seedCount
       << ",\"window_ms\":" << windowMs << "},";
    os << "\"runs\":[";
    bool first = true;
    for (const auto& p : kPatterns) {
        for (int s = seedFirst; s < seedFirst + seedCount; ++s) {
            Outcome o = p.positive(s, windowMs);
            if (!first) os << ",";
            first = false;
            os << runJson(p, s, o, true);
        }
    }
    os << "]}";
    return os.str();
}

std::string runNegativeMatrix() {
    const int seedFirst = envInt("RACE_SENTINEL_SEED_FIRST", 0);
    const int seedCount = envInt("RACE_SENTINEL_SEED_COUNT", 20);
    const int windowMs = envInt("RACE_SENTINEL_WINDOW_MS", 2500);
    std::ostringstream os;
    os << "{\"mode\":\"negative\",\"generated\":\"" << isoNow() << "\",";
    os << "\"detector\":{\"active\":" << (rs::active() ? "true" : "false")
       << ",\"sampling\":" << rs::samplingRate() << "},";
    os << "\"config\":{\"seed_first\":" << seedFirst << ",\"seed_count\":" << seedCount
       << ",\"window_ms\":" << windowMs << "},";
    os << "\"runs\":[";
    bool first = true;
    for (const auto& p : kPatterns) {
        for (int s = seedFirst; s < seedFirst + seedCount; ++s) {
            Outcome o = p.negative(s, windowMs);
            if (!first) os << ",";
            first = false;
            os << runJson(p, s, o, false);
        }
    }
    os << "]}";
    return os.str();
}

// Fixed-op-count locked workloads: used for the overhead multiplier.  The same
// code runs with the detector suspended (reference) and armed (instrumented).
double runLockedWorkload(int which, std::size_t ops) {
    switch (which) {
        case 0: {
            rs::RaceTrack<std::vector<int>> v;
            v.reserve(ops + 128);
            for (int i = 0; i < 64; ++i) v.push_back(i);
            rs::TrackedMutex mu;
            const auto t0 = Clock::now();
            std::thread w([&] {
                for (std::size_t i = 0; i < ops; ++i) {
                    std::lock_guard<rs::TrackedMutex> g(mu);
                    v.push_back(static_cast<int>(i & 0xffff));
                }
            });
            std::thread r([&] {
                long long sum = 0;
                for (std::size_t i = 0; i < ops; ++i) {
                    std::lock_guard<rs::TrackedMutex> g(mu);
                    sum += v[i % 64];
                }
                (void)sum;
            });
            w.join();
            r.join();
            return msBetween(t0, Clock::now());
        }
        case 1: {
            rs::RaceTrack<std::string> s;
            s.assign("init");
            rs::TrackedMutex mu;
            const auto t0 = Clock::now();
            std::thread w([&] {
                for (std::size_t i = 0; i < ops; ++i) {
                    std::lock_guard<rs::TrackedMutex> g(mu);
                    s.assign((i & 1) ? "alpha-aa" : "bravo-bb");
                }
            });
            std::thread r([&] {
                long long sum = 0;
                for (std::size_t i = 0; i < ops; ++i) {
                    std::lock_guard<rs::TrackedMutex> g(mu);
                    sum += static_cast<long long>(s.size()) + static_cast<long long>(s.data()[0]);
                }
                (void)sum;
            });
            w.join();
            r.join();
            return msBetween(t0, Clock::now());
        }
        case 2: {
            rs::RaceTrack<std::map<int, int>> m;
            for (int i = 0; i < 64; ++i) m[i] = i;
            rs::TrackedMutex mu;
            const auto t0 = Clock::now();
            std::thread w([&] {
                for (std::size_t i = 0; i < ops; ++i) {
                    const int key = static_cast<int>(i & 255);
                    std::lock_guard<rs::TrackedMutex> g(mu);
                    m[key] = key;
                }
            });
            std::thread r([&] {
                long long sum = 0;
                for (std::size_t i = 0; i < ops; ++i) {
                    std::lock_guard<rs::TrackedMutex> g(mu);
                    auto it = m.find(7);
                    if (it != m.end()) sum += it->second;
                }
                (void)sum;
            });
            w.join();
            r.join();
            return msBetween(t0, Clock::now());
        }
        default: {
            rs::RaceTrack<NeverFreeUMap> u;
            for (int i = 0; i < 128; ++i) u[i] = i;
            rs::TrackedMutex mu;
            const auto t0 = Clock::now();
            std::thread w([&] {
                for (std::size_t i = 0; i < ops; ++i) {
                    const int key = static_cast<int>(i & 127);
                    std::lock_guard<rs::TrackedMutex> g(mu);
                    u.erase(key);
                    u[key] = key;
                }
            });
            std::thread r([&] {
                long long sum = 0;
                for (std::size_t i = 0; i < ops; ++i) {
                    std::lock_guard<rs::TrackedMutex> g(mu);
                    auto it = u.find(5);
                    if (it != u.end()) sum += it->second;
                }
                (void)sum;
            });
            w.join();
            r.join();
            return msBetween(t0, Clock::now());
        }
    }
}

std::string runOverheadMatrix() {
    const int reps = envInt("RACE_SENTINEL_REPS", 3);
    const int ops = envInt("RACE_SENTINEL_OPS", 200000);
    std::ostringstream os;
    os << "{\"mode\":\"overhead\",\"generated\":\"" << isoNow() << "\",";
    os << "\"detector\":{\"sampling\":" << rs::samplingRate() << "},";
    os << "\"config\":{\"reps\":" << reps << ",\"ops\":" << ops << "},";
    os << "\"patterns\":[";
    bool firstPattern = true;
    for (int which = 0; which < 4; ++which) {
        std::vector<double> off, on;
        for (int rep = 0; rep < reps; ++rep) {
            rs::setActiveForTesting(false);
            double tOff = runLockedWorkload(which, static_cast<std::size_t>(ops));
            rs::setActiveForTesting(true);
            double tOn = runLockedWorkload(which, static_cast<std::size_t>(ops));
            off.push_back(tOff);
            on.push_back(tOn);
        }
        std::sort(off.begin(), off.end());
        std::sort(on.begin(), on.end());
        const double medOff = off[off.size() / 2];
        const double medOn = on[on.size() / 2];
        if (!firstPattern) os << ",";
        firstPattern = false;
        os << "{\"pattern\":\"" << kPatterns[which].id << "\",";
        os << "\"pattern_name\":\"" << jsonEscape(kPatterns[which].name) << "\",";
        os << "\"off_ms\":" << medOff << ",\"on_ms\":" << medOn << ",";
        os << "\"ratio\":" << (medOff > 0 ? medOn / medOff : 0.0) << ",";
        os << "\"reps\":[";
        for (int rep = 0; rep < reps; ++rep) {
            if (rep) os << ",";
            os << "{\"off_ms\":" << off[rep] << ",\"on_ms\":" << on[rep] << "}";
        }
        os << "]}";
    }
    os << "]}";
    rs::setActiveForTesting(true);
    return os.str();
}

}  // namespace


namespace {

// The full gtest suite runs without RACE_SENTINEL=1, so these tests must arm the
// detector themselves (and restore the previous state) to be meaningful in both
// the sidecar experiment and a normal regression run.
struct ArmedDetector {
    bool prev;
    ArmedDetector() : prev(rs::active()) { rs::setActiveForTesting(true); }
    ~ArmedDetector() { rs::setActiveForTesting(prev); }
    ArmedDetector(const ArmedDetector&) = delete;
    ArmedDetector& operator=(const ArmedDetector&) = delete;
};

}  // namespace
// ===========================================================================
// Correctness tests: the detector must flag the positives and stay silent on
// the negatives.  These run in the sidecar build and in the full gtest suite.
// ===========================================================================

TEST(RaceSentinelControls, PositiveAVectorPushBackDetected) {
    ArmedDetector arm;  // works in the full suite with RACE_SENTINEL unset
    const Outcome o = positiveVectorPushBack(1, 500);
    EXPECT_TRUE(o.detected) << "sampled=" << o.sampled;
    EXPECT_GT(o.reports, 0u);
    EXPECT_GE(o.first_report_ms, 0.0);
}

TEST(RaceSentinelControls, PositiveBGlobalStringDetected) {
    ArmedDetector arm;  // works in the full suite with RACE_SENTINEL unset
    const Outcome o = positiveGlobalString(1, 500);
    EXPECT_TRUE(o.detected) << "sampled=" << o.sampled;
    EXPECT_GT(o.reports, 0u);
}

TEST(RaceSentinelControls, PositiveCMapInsertDetected) {
    ArmedDetector arm;  // works in the full suite with RACE_SENTINEL unset
    const Outcome o = positiveMapInsert(1, 500);
    EXPECT_TRUE(o.detected) << "sampled=" << o.sampled;
    EXPECT_GT(o.reports, 0u);
}

TEST(RaceSentinelControls, PositiveDUnorderedEraseDetected) {
    ArmedDetector arm;  // works in the full suite with RACE_SENTINEL unset
    const Outcome o = positiveUnorderedErase(1, 500);
    EXPECT_TRUE(o.detected) << "sampled=" << o.sampled;
    EXPECT_GT(o.reports, 0u);
}

TEST(RaceSentinelControls, NegativeAVectorPushBackSilent) {
    ArmedDetector arm;  // works in the full suite with RACE_SENTINEL unset
    const Outcome o = negativeVectorPushBack(1, 800);
    EXPECT_FALSE(o.detected);
    EXPECT_EQ(o.reports, 0u);
    EXPECT_GT(o.sampled, 0u);
}

TEST(RaceSentinelControls, NegativeBGlobalStringSilent) {
    ArmedDetector arm;  // works in the full suite with RACE_SENTINEL unset
    const Outcome o = negativeGlobalString(1, 800);
    EXPECT_FALSE(o.detected);
    EXPECT_EQ(o.reports, 0u);
}

TEST(RaceSentinelControls, NegativeCMapInsertSilent) {
    ArmedDetector arm;  // works in the full suite with RACE_SENTINEL unset
    const Outcome o = negativeMapInsert(1, 800);
    EXPECT_FALSE(o.detected);
    EXPECT_EQ(o.reports, 0u);
}

TEST(RaceSentinelControls, NegativeDUnorderedEraseSilent) {
    ArmedDetector arm;  // works in the full suite with RACE_SENTINEL unset
    const Outcome o = negativeUnorderedErase(1, 800);
    EXPECT_FALSE(o.detected);
    EXPECT_EQ(o.reports, 0u);
}

// ===========================================================================
// Unit tests for the detector mechanics.
// ===========================================================================

TEST(RaceSentinelUnit, ArmedByEnvironmentOrTestHook) {
    // The launcher sets RACE_SENTINEL=1; either way active() must be readable
    // and settable for tests.
    const bool wasActive = rs::active();
    rs::setActiveForTesting(true);
    EXPECT_TRUE(rs::active());
    rs::setActiveForTesting(false);
    EXPECT_FALSE(rs::active());
    rs::setActiveForTesting(wasActive);
}

TEST(RaceSentinelUnit, SamplingRateIsPositive) { EXPECT_GE(rs::samplingRate(), 1u); }
    ArmedDetector arm;  // works in the full suite with RACE_SENTINEL unset

TEST(RaceSentinelUnit, InactiveDetectorIsPassThroughAndSilent) {
    rs::setActiveForTesting(false);
    const Outcome o = positiveVectorPushBack(3, 200);
    rs::setActiveForTesting(true);
    EXPECT_FALSE(o.detected);
    EXPECT_EQ(o.reports, 0u);
    EXPECT_EQ(o.sampled, 0u);
}

TEST(RaceSentinelUnit, SpawnEpochSuppressesPreSpawnWrites) {
    ArmedDetector arm;  // works in the full suite with RACE_SENTINEL unset
    // Access made before the concurrent group is spawned must not be flagged
    // when the spawned thread reads it (spawn = happens-before edge).
    rs::resetAll();
    rs::RaceTrack<std::vector<int>> v;
    v.push_back(42);
    const std::uint64_t hb = rs::beginConcurrent();
    std::thread t = rs::spawnWith(
        [&v] {
            rs::setThreadLabel("child");
            long long sum = 0;
            for (int x : v) sum += x;
            (void)sum;
        },
        hb);
    rs::joinTracked(t);
    EXPECT_EQ(rs::reportCount(), 0u);
}

TEST(RaceSentinelUnit, JoinEdgeSuppressesPostJoinRead) {
    ArmedDetector arm;  // works in the full suite with RACE_SENTINEL unset
    rs::resetAll();
    rs::RaceTrack<std::vector<int>> v;
    const std::uint64_t hb = rs::beginConcurrent();
    std::thread t = rs::spawnWith(
        [&v] {
            rs::setThreadLabel("child");
            v.push_back(1);
            v.push_back(2);
        },
        hb);
    rs::joinTracked(t);  // advances our epoch: child's writes happen-before us
    long long sum = 0;
    for (int x : v) sum += x;
    (void)sum;
    EXPECT_EQ(rs::reportCount(), 0u);
}

TEST(RaceSentinelUnit, DifferentLockSetsAreReported) {
    ArmedDetector arm;  // works in the full suite with RACE_SENTINEL unset
    // Two threads writing the same element under *different* mutexes is a race;
    // the lockset intersection is empty so it must be reported.
    rs::resetAll();
    rs::RaceTrack<std::vector<int>> v;
    v.push_back(0);
    rs::TrackedMutex m1, m2;
    std::atomic<bool> go{false};
    const std::uint64_t hb = rs::beginConcurrent();
    std::thread a = rs::spawnWith(
        [&] {
            rs::setThreadLabel("A");
            while (!go.load(std::memory_order_relaxed)) std::this_thread::yield();
            for (int i = 0; i < 200000; ++i) {
                std::lock_guard<rs::TrackedMutex> g(m1);
                v[0] = i;
            }
        },
        hb);
    std::thread b = rs::spawnWith(
        [&] {
            rs::setThreadLabel("B");
            while (!go.load(std::memory_order_relaxed)) std::this_thread::yield();
            for (int i = 0; i < 200000; ++i) {
                std::lock_guard<rs::TrackedMutex> g(m2);
                v[0] = -i;
            }
        },
        hb);
    go.store(true, std::memory_order_relaxed);
    rs::joinTracked(a);
    rs::joinTracked(b);
    EXPECT_GT(rs::reportCount(), 0u);
    auto rp = rs::reports();
    ASSERT_FALSE(rp.empty());
}

TEST(RaceSentinelUnit, SameThreadIsNeverFlagged) {
    ArmedDetector arm;  // works in the full suite with RACE_SENTINEL unset
    rs::resetAll();
    rs::RaceTrack<std::vector<int>> v;
    v.push_back(0);
    rs::TrackedMutex m1;
    for (int i = 0; i < 1000; ++i) {
        if (i & 1) {
            std::lock_guard<rs::TrackedMutex> g(m1);
            v[0] = i;
        } else {
            v[0] = -i;  // unlocked, but same thread -> no report
        }
    }
    EXPECT_EQ(rs::reportCount(), 0u);
}

TEST(RaceSentinelUnit, ReportIsDeduplicatedPerElement) {
    ArmedDetector arm;  // works in the full suite with RACE_SENTINEL unset
    // A repeated race on the same element must produce at least one report but
    // the detector caps it at one per element (bounded report memory).
    rs::resetAll();
    rs::RaceTrack<std::vector<int>> v;
    v.push_back(0);
    std::atomic<bool> stop{false};
    const std::uint64_t hb = rs::beginConcurrent();
    auto a = rs::spawnWith(
        [&] {
            while (!stop.load(std::memory_order_relaxed)) v[0] = 1;
        },
        hb);
    auto b = rs::spawnWith(
        [&] {
            while (!stop.load(std::memory_order_relaxed)) v[0] = 2;
        },
        hb);
    while (rs::reportCount() == 0 && rs::sampledAccesses() < 1000) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    stop.store(true, std::memory_order_relaxed);
    rs::joinTracked(a);
    rs::joinTracked(b);
    EXPECT_GT(rs::reportCount(), 0u);
    EXPECT_LE(rs::reportCount(), 1u);
}

TEST(RaceSentinelUnit, DecoratorPreservesContainerSemantics) {
    ArmedDetector arm;  // works in the full suite with RACE_SENTINEL unset
    rs::RaceTrack<std::vector<int>> v;
    v.push_back(1);
    v.push_back(2);
    EXPECT_EQ(v.size(), 2u);
    EXPECT_FALSE(v.empty());
    int sum = 0;
    for (int x : v) sum += x;
    EXPECT_EQ(sum, 3);
    v.clear();
    EXPECT_TRUE(v.empty());

    rs::RaceTrack<std::map<int, int>> m;
    m[1] = 10;
    m[2] = 20;
    EXPECT_EQ(m.size(), 2u);
    auto it = m.find(2);
    ASSERT_NE(it, m.end());
    EXPECT_EQ(it->second, 20);
    EXPECT_EQ(m.erase(1), 1u);
    EXPECT_EQ(m.count(1), 0u);

    rs::RaceTrack<NeverFreeUMap> u;
    u[5] = 50;
    EXPECT_EQ(u.find(5)->second, 50);
    EXPECT_EQ(u.erase(5), 1u);

    rs::RaceTrack<std::string> s;
    s.assign("hello");
    EXPECT_EQ(s.size(), 5u);
    EXPECT_STREQ(s.c_str(), "hello");
    s.assign("yo");
    EXPECT_EQ(s.size(), 2u);
}

TEST(RaceSentinelUnit, LockIdsAreRecycledAcrossManyMutexInstances) {
    ArmedDetector arm;  // works in the full suite with RACE_SENTINEL unset
    // Regression for a defect found by the -SoakMinutes path: the lockset is a
    // 64-bit mask, and each control run creates a fresh TrackedMutex. Without
    // id recycling, the 64th instance became "untracked" and the following
    // negative-control runs produced thousands of false positives.
    for (int i = 0; i < 500; ++i) {
        rs::TrackedMutex mu;
        {
            std::lock_guard<rs::TrackedMutex> g(mu);
            ASSERT_EQ(rs::detail::tls().cur.count(), 1u) << "mutex instance " << i;
        }
        ASSERT_EQ(rs::detail::tls().cur.count(), 0u);
    }
}

TEST(RaceSentinelUnit, ManyShortNegativeRunsStaySilent) {
    ArmedDetector arm;  // works in the full suite with RACE_SENTINEL unset
    // Same shape as the soak path: many sequential negative controls, each with
    // its own TrackedMutex instance. Must stay at zero false positives.
    std::uint64_t totalReports = 0;
    for (int i = 0; i < 100; ++i) {
        const Outcome o = negativeVectorPushBack(i, 20);
        totalReports += o.reports;
    }
    EXPECT_EQ(totalReports, 0u);
    EXPECT_GT(rs::sampledAccesses(), 0u);
}

TEST(RaceSentinelUnit, LocksetIsTrackedThroughTrackedMutex) {
    ArmedDetector arm;  // works in the full suite with RACE_SENTINEL unset
    rs::TrackedMutex mu, mu2;
    EXPECT_EQ(rs::detail::tls().cur.count(), 0u);
    {
        std::lock_guard<rs::TrackedMutex> g(mu);
        EXPECT_EQ(rs::detail::tls().cur.count(), 1u);
        {
            std::lock_guard<rs::TrackedMutex> g2(mu2);
            EXPECT_EQ(rs::detail::tls().cur.count(), 2u);
        }
        EXPECT_EQ(rs::detail::tls().cur.count(), 1u);
    }
    EXPECT_EQ(rs::detail::tls().cur.count(), 0u);
}

// ===========================================================================
// Experiment matrix entry point.  GTEST_SKIP() unless the launcher asks for it,
// so the full regression suite is unaffected.
// ===========================================================================

TEST(RaceSentinelExperiment, Matrix) {
    const char* mode = std::getenv("RACE_SENTINEL_MODE");
    const char* out = std::getenv("RACE_SENTINEL_OUT");
    if (!mode) {
        GTEST_SKIP() << "RACE_SENTINEL_MODE not set (experiment driver only)";
    }
    ASSERT_NE(out, nullptr) << "RACE_SENTINEL_OUT must be set with RACE_SENTINEL_MODE";
    ASSERT_TRUE(rs::active()) << "RACE_SENTINEL=1 must arm the detector for the experiment";

    std::string json;
    const std::string m(mode);
    if (m == "positive") {
        json = runPositiveMatrix();
    } else if (m == "negative") {
        json = runNegativeMatrix();
    } else if (m == "overhead") {
        json = runOverheadMatrix();
    } else {
        FAIL() << "unknown RACE_SENTINEL_MODE: " << m;
    }

    std::ofstream f(out, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(f.good()) << "cannot open " << out;
    f << json;
    f.close();
    EXPECT_FALSE(json.empty());
}
