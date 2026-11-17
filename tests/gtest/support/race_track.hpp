// tests/gtest/support/race_track.hpp
//
// RaceSentinel-MinGW (M1, Phase 1) -- TEST-ONLY lockset + sampling race detector.
//
// Why this file exists
// --------------------
// doc/v10.1/methods_backlog.md M1: this machine has no ThreadSanitizer/ASan
// (MinGW gcc 16.1.0 ships neither libtsan nor libasan, clang is not installed,
// and Clang TSan does not support Windows), yet the residual 0xC0000374 heap
// corruption is suspected to come from remaining data races.  This header is a
// self-built detector that lives ONLY under tests/ and is never included from
// product sources (HARD RULE: no product source edits).
//
// Design (lockset algorithm from Eraser, 1997 + sampling from LiteRace, 2009)
// ---------------------------------------------------------------------------
//  * `RaceTrack<Container>` is a decorator around a standard container
//    (std::vector / std::string / std::map / std::unordered_map).  Every
//    container operation records, for the touched *element* (and for the
//    container's structural "meta" slot):
//        - the accessing thread id,
//        - a coarse happens-before ("lock-set epoch") stamp,
//        - the set of test-visible locks the thread held (lockset).
//  * A conflicting unsynchronised access is:
//        different thread, neither access happens-before the other,
//        and the intersection of the two locksets is empty.
//    Same-thread accesses are never flagged (sequential consistency inside a
//    thread), and accesses guarded by a common lock are never flagged.
//  * Locks become visible only through `race_sentinel::TrackedMutex`
//    (test-only).  This is deliberate: Phase 1 has to be standalone and must
//    not touch product code, so it cannot instrument phoenix's real mutexes.
//    Phase 2 (documented in doc/v10.1/race_sentinel.md) points the same
//    `touch()`/lockset hooks at the product's lock wrappers.
//  * Sampling: only every N-th container operation is instrumented
//    (1/N, default N=16, env RACE_SENTINEL_SAMPLE).  Hot repeated races are
//    still seen because slot state persists between sampled events, but the
//    steady-state overhead drops well below 5x.
//  * Activation: the detector is a pure pass-through unless the environment
//    variable `RACE_SENTINEL=1` is set at process start
//    (`race_sentinel::setActiveForTesting()` exists for unit tests only).
//
// Happens-before model (and its honest limits)
// --------------------------------------------
// Real detectors use vector clocks (FastTrack).  Phase 1 uses one global,
// monotonically increasing epoch:
//    - before spawning a group of threads, call `beginConcurrent()` once; all
//      threads of the group inherit that same epoch, so accesses between them
//      are considered concurrent (a race is possible);
//    - `joinTracked(t)` joins and then advances the calling thread's epoch, so
//      everything the joined thread did happens-before the caller's later
//      accesses;
//    - a recorded access with a strictly lower epoch than the current thread's
//      epoch is treated as happens-before the current access.
// This is sound for thread-spawn / thread-join edges (which is what the
// control patterns need) and is blind to condition variables, atomics used as
// hand-offs and one-way "flag" publication.  It can therefore produce false
// negatives, and -- for a thread that accesses without observing any of these
// edges -- it cannot manufacture a happens-before edge, so it does not produce
// false positives on correctly synchronised code.  Phase 2 replaces the epoch
// with per-thread vector clocks.
//
// What is NOT covered (see doc/v10.1/race_sentinel.md "limits"):
//    - locks the detector cannot see (raw std::mutex, std::shared_mutex,
//      condition variables, atomics, one-shot publication, async futures);
//    - container types other than the four decorated here;
//    - interleaving-level non-atomicity bugs that are not data races.
//
// Build: header-only, C++20 (MSYS2 gcc 16.1.0 / MinGW).  Included by
// tests/gtest/unit/race/test_race_sentinel.cpp; also picked up by the full
// gtest suite (compile_gtest.bat globs the tests/gtest tree).

#pragma once

#include <array>
#include <atomic>
#include <bitset>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iterator>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace race_sentinel {

// ---------------------------------------------------------------------------
// Reports
// ---------------------------------------------------------------------------

/// One detected unsynchronised conflicting access (a race candidate).
struct Report {
    std::uint64_t element_id = 0;   ///< hashed container/element identity
    std::uint64_t tid_a = 0;        ///< previous accessor thread id
    std::uint64_t tid_b = 0;        ///< current accessor thread id
    std::uint64_t hb_a = 0;         ///< epoch stamped on previous access
    std::uint64_t hb_b = 0;         ///< epoch stamped on current access
    std::uint64_t locks_a = 0;      ///< lockset bitmask, previous accessor
    std::uint64_t locks_b = 0;      ///< lockset bitmask, current accessor
    const char* label_a = "";       ///< thread label, previous accessor
    const char* label_b = "";       ///< thread label, current accessor
    const char* op_a = "";          ///< operation, previous accessor
    const char* op_b = "";          ///< operation, current accessor
    double since_reset_ms = 0.0;    ///< ms since last resetAll()
};

namespace detail {

inline std::uint64_t mix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

using Lockset = std::bitset<64>;

struct AccessRec {
    bool valid = false;
    std::uint64_t tid = 0;
    std::uint64_t hb = 1;
    Lockset locks{};
    const char* label = "";
    const char* op = "";
};

struct Slot {
    AccessRec writer{};
    AccessRec reader{};
    bool reported = false;
};

struct Shard {
    std::mutex mu;
    std::unordered_map<std::uint64_t, Slot> slots;
};

// Per-thread state.  Kept in a function-local `thread_local` so no lifecycle
// hook is needed for threads created by the tests.
struct ThreadState {
    std::uint64_t tid = 0;
    std::uint64_t hb = 1;               // observed happens-before epoch
    std::uint64_t counter = 0;          // access counter used for 1/N sampling
    const char* label = "";
    Lockset cur{};                      // currently held locks
    std::array<unsigned char, 64> depth{};  // re-entrant lock depth per lock id
};

struct GlobalState {
    std::atomic<bool> active{true};
    std::size_t sampling = 16;
    std::atomic<std::uint64_t> epoch{1};
    std::atomic<std::uint64_t> next_container_id{1};
    std::atomic<std::uint64_t> next_thread_id{1};
    std::atomic<std::uint64_t> next_lock_id{1};
    // Lockset bits are limited to ids 1..63.  Long experiments create many
    // short-lived TrackedMutex instances (one per control run), so ids must be
    // recycled on destruction -- otherwise every mutex created after the 63rd
    // is "untracked" and the negative controls start reporting false positives.
    std::mutex lock_id_mu;
    std::vector<std::uint64_t> free_lock_ids;
    std::atomic<std::uint64_t> sampled_accesses{0};
    std::atomic<std::uint64_t> total_reports{0};
    std::mutex report_mu;
    std::vector<Report> reports;
    std::array<Shard, 16> shards;
    std::chrono::steady_clock::time_point reset_time = std::chrono::steady_clock::now();
};

inline std::uint64_t allocLockId(GlobalState& g) {
    std::lock_guard<std::mutex> lk(g.lock_id_mu);
    if (!g.free_lock_ids.empty()) {
        const std::uint64_t id = g.free_lock_ids.back();
        g.free_lock_ids.pop_back();
        return id;
    }
    const std::uint64_t id = g.next_lock_id.fetch_add(1, std::memory_order_relaxed);
    return id < 64 ? id : 0;  // 0 = exhausted, i.e. "not tracked"
}

inline void releaseLockId(GlobalState& g, std::uint64_t id) {
    if (id == 0) return;
    std::lock_guard<std::mutex> lk(g.lock_id_mu);
    g.free_lock_ids.push_back(id);
}

// Intentionally leaked: detector state may be touched from destructors of
// threads/objects that outlive static destruction order.
inline GlobalState& state() {
    static GlobalState* s = [] {
        auto* g = new GlobalState();
        const char* en = std::getenv("RACE_SENTINEL");
        g->active.store(en != nullptr && std::strcmp(en, "1") == 0,
                        std::memory_order_relaxed);
        const char* n = std::getenv("RACE_SENTINEL_SAMPLE");
        long v = n ? std::strtol(n, nullptr, 10) : 16;
        if (v < 1) v = 1;
        g->sampling = static_cast<std::size_t>(v);
        g->reports.reserve(4096);
        return g;
    }();
    return *s;
}

inline ThreadState& tls() {
    thread_local ThreadState ts;
    if (ts.tid == 0) {
        ts.tid = state().next_thread_id.fetch_add(1, std::memory_order_relaxed);
        ts.hb = state().epoch.load(std::memory_order_relaxed);
    }
    return ts;
}

inline void record_report(std::uint64_t elem, const AccessRec& prev,
                          const AccessRec& cur) {
    GlobalState& g = state();
    if (g.total_reports.load(std::memory_order_relaxed) >= 4096) {
        g.total_reports.fetch_add(1, std::memory_order_relaxed);
        return;  // still count, but stop growing memory
    }
    std::lock_guard<std::mutex> lk(g.report_mu);
    Report r;
    r.element_id = elem;
    r.tid_a = prev.tid;
    r.tid_b = cur.tid;
    r.hb_a = prev.hb;
    r.hb_b = cur.hb;
    r.locks_a = prev.locks.to_ullong();
    r.locks_b = cur.locks.to_ullong();
    r.label_a = prev.label;
    r.label_b = cur.label;
    r.op_a = prev.op;
    r.op_b = cur.op;
    r.since_reset_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                  g.reset_time)
            .count();
    g.reports.push_back(r);
    g.total_reports.fetch_add(1, std::memory_order_relaxed);
}

/// Core instrumentation hook: called for every (sampled) element/meta access.
inline void touch(std::uint64_t elem, bool is_write, const char* op) {
    GlobalState& g = state();
    ThreadState& ts = tls();
    if ((++ts.counter % g.sampling) != 0) return;
    g.sampled_accesses.fetch_add(1, std::memory_order_relaxed);

    Shard& sh = g.shards[elem & 15u];
    AccessRec prev;
    bool conflict = false;
    {
        // The shard lock is held only for the slot update; report recording is
        // done after releasing it so the detector never nests shard -> report
        // locks (resetAll() takes them in the opposite order).
        std::lock_guard<std::mutex> lk(sh.mu);
        auto it = sh.slots.find(elem);
        if (it == sh.slots.end()) {
            if (sh.slots.size() > 100000) sh.slots.clear();
            it = sh.slots.emplace(elem, Slot{}).first;
        }
        Slot& s = it->second;

        // Any pair of accesses to the same element where at least one is a
        // write, the threads differ, neither happened-before the other and no
        // lock is shared is a race candidate.  A read only has to be compared
        // with the last writer; a write has to be compared with both the last
        // reader and the last writer (write-write race).
        auto conflict_with = [&](const AccessRec& a) {
            return a.valid && a.tid != ts.tid && a.hb >= ts.hb &&
                   (a.locks & ts.cur).none();
        };
        const AccessRec* other = nullptr;
        if (!is_write) {
            if (conflict_with(s.writer)) other = &s.writer;
        } else {
            if (conflict_with(s.reader)) {
                other = &s.reader;
            } else if (conflict_with(s.writer)) {
                other = &s.writer;
            }
        }
        if (other != nullptr && !s.reported) {
            s.reported = true;
            conflict = true;
            prev = *other;
        }

        AccessRec& self = is_write ? s.writer : s.reader;
        self.valid = true;
        self.tid = ts.tid;
        self.hb = ts.hb;
        self.locks = ts.cur;
        self.label = ts.label;
        self.op = op;
    }

    if (conflict) {
        AccessRec cur;
        cur.valid = true;
        cur.tid = ts.tid;
        cur.hb = ts.hb;
        cur.locks = ts.cur;
        cur.label = ts.label;
        cur.op = op;
        record_report(elem, prev, cur);
    }
}

// ---- container traits -----------------------------------------------------

template <class C, class = void>
struct has_key_type : std::false_type {};
template <class C>
struct has_key_type<C, std::void_t<typename C::key_type>> : std::true_type {};

template <class C, class = void>
struct key_type_of {
    using type = int;  // never instantiated for keyless containers
};
template <class C>
struct key_type_of<C, std::void_t<typename C::key_type>> {
    using type = typename C::key_type;
};

}  // namespace detail

// ---------------------------------------------------------------------------
// Public control API
// ---------------------------------------------------------------------------

/// True when the detector was armed at process start (RACE_SENTINEL=1).
inline bool active() {
    return detail::state().active.load(std::memory_order_relaxed);
}

/// Test-only override; production runs are armed by the environment variable.
inline void setActiveForTesting(bool on) {
    detail::state().active.store(on, std::memory_order_relaxed);
}

/// Effective 1/N sampling rate.
inline std::size_t samplingRate() { return detail::state().sampling; }

/// Number of instrumented (sampled) accesses since reset.
inline std::uint64_t sampledAccesses() {
    return detail::state().sampled_accesses.load(std::memory_order_relaxed);
}

/// Number of reported races since reset (cheap poll target).
inline std::uint64_t reportCount() {
    return detail::state().total_reports.load(std::memory_order_relaxed);
}

/// Snapshot of the reports collected since reset.
inline std::vector<Report> reports() {
    detail::GlobalState& g = detail::state();
    std::lock_guard<std::mutex> lk(g.report_mu);
    return g.reports;
}

/// Clear the detector's slot table / reports and restart the epoch.
inline void resetAll() {
    detail::GlobalState& g = detail::state();
    {
        std::lock_guard<std::mutex> lk(g.report_mu);
        g.reports.clear();
    }
    for (auto& sh : g.shards) {
        std::lock_guard<std::mutex> lk(sh.mu);
        sh.slots.clear();
    }
    g.total_reports.store(0, std::memory_order_relaxed);
    g.sampled_accesses.store(0, std::memory_order_relaxed);
    g.epoch.store(1, std::memory_order_relaxed);
    g.reset_time = std::chrono::steady_clock::now();
    detail::ThreadState& ts = detail::tls();
    ts.hb = 1;
    ts.counter = 0;
    ts.label = "";
}

/// Label the current thread (used in reports).
inline void setThreadLabel(const char* label) { detail::tls().label = label; }

/// RAII label scope.
class LabelScope {
public:
    explicit LabelScope(const char* label) : prev_(detail::tls().label) {
        detail::tls().label = label;
    }
    ~LabelScope() { detail::tls().label = prev_; }
    LabelScope(const LabelScope&) = delete;
    LabelScope& operator=(const LabelScope&) = delete;

private:
    const char* prev_;
};

// ---------------------------------------------------------------------------
// Synchronisation primitives the detector can see
// ---------------------------------------------------------------------------

/// std::mutex-identical interface; membership is tracked into the caller's
/// lockset so the detector can prove "guarded by a common lock".
class TrackedMutex {
public:
    TrackedMutex() : id_(detail::allocLockId(detail::state())) {}
    ~TrackedMutex() { detail::releaseLockId(detail::state(), id_); }
    TrackedMutex(const TrackedMutex&) = delete;
    TrackedMutex& operator=(const TrackedMutex&) = delete;

    void lock() {
        mu_.lock();
        if (!active() || id_ == 0) return;
        detail::ThreadState& ts = detail::tls();
        if (ts.depth[id_]++ == 0) ts.cur.set(id_);
    }
    void unlock() {
        if (active() && id_ != 0) {
            detail::ThreadState& ts = detail::tls();
            if (ts.depth[id_] != 0 && --ts.depth[id_] == 0) ts.cur.reset(id_);
        }
        mu_.unlock();
    }
    bool try_lock() {
        if (!mu_.try_lock()) return false;
        if (!active() || id_ == 0) return true;
        detail::ThreadState& ts = detail::tls();
        if (ts.depth[id_]++ == 0) ts.cur.set(id_);
        return true;
    }

private:
    std::mutex mu_;
    std::uint64_t id_;
};

/// Start a group of concurrent threads: returns the shared happens-before
/// epoch that every member of the group must inherit via `spawnWith`.
inline std::uint64_t beginConcurrent() {
    return detail::state().epoch.fetch_add(1, std::memory_order_relaxed) + 1;
}

/// Spawn a thread whose observed epoch is `hb` (from beginConcurrent()).
template <class F>
std::thread spawnWith(F&& fn, std::uint64_t hb) {
    return std::thread([f = std::forward<F>(fn), hb]() mutable {
        detail::ThreadState& ts = detail::tls();  // assigns the tid
        ts.hb = hb;
        f();
    });
}

/// Join, then publish a happens-before edge from the joined thread to us.
inline void joinTracked(std::thread& t) {
    if (t.joinable()) t.join();
    detail::tls().hb = detail::state().epoch.fetch_add(1, std::memory_order_relaxed) + 1;
}

// ---------------------------------------------------------------------------
// RaceTrack<Container> -- the container decorator
// ---------------------------------------------------------------------------

/// Decorates a standard container and records per-element access metadata.
/// When `RACE_SENTINEL=1` is absent every method is a plain forward, so the
/// same workload can be used as the uninstrumented reference run.
template <class Container>
class RaceTrack {
public:
    using container_type = Container;
    using value_type = typename Container::value_type;
    using size_type = typename Container::size_type;
    using key_type = typename detail::key_type_of<Container>::type;
    using reference = typename Container::reference;
    using const_reference = typename Container::const_reference;

    RaceTrack()
        : cid_(detail::state().next_container_id.fetch_add(1,
                                                          std::memory_order_relaxed)) {}
    explicit RaceTrack(Container&& c)
        : c_(std::move(c)),
          cid_(detail::state().next_container_id.fetch_add(1,
                                                          std::memory_order_relaxed)) {}
    explicit RaceTrack(const Container& c)
        : c_(c),
          cid_(detail::state().next_container_id.fetch_add(1,
                                                          std::memory_order_relaxed)) {}

    RaceTrack(RaceTrack&&) = default;
    RaceTrack& operator=(RaceTrack&&) = default;
    RaceTrack(const RaceTrack&) = delete;
    RaceTrack& operator=(const RaceTrack&) = delete;

    Container& get() { return c_; }
    const Container& get() const { return c_; }

    // ---- structural (meta slot) ------------------------------------------
    size_type size() const {
        readMeta("size");
        return c_.size();
    }
    bool empty() const {
        readMeta("empty");
        return c_.empty();
    }
    void reserve(size_type n) {
        writeMeta("reserve");
        c_.reserve(n);
    }
    size_type capacity() const {
        readMeta("capacity");
        return c_.capacity();
    }
    void clear() {
        writeMeta("clear");
        c_.clear();
    }

    // ---- sequence containers --------------------------------------------
    void push_back(const value_type& v) {
        if (active()) {
            writeMeta("push_back.structure");
            writeElem(elemLocal(c_.size() + 1), "push_back.element");
        }
        c_.push_back(v);
    }
    void push_back(value_type&& v) {
        if (active()) {
            writeMeta("push_back.structure");
            writeElem(elemLocal(c_.size() + 1), "push_back.element");
        }
        c_.push_back(std::move(v));
    }
    reference operator[](size_type i) {
        if (active()) writeElem(elemLocal(i + 1), "operator[]");
        return c_[i];
    }
    const_reference operator[](size_type i) const {
        readElem(elemLocal(i + 1), "operator[]");
        return c_[i];
    }
    reference at(size_type i) {
        if (active()) writeElem(elemLocal(i + 1), "at");
        return c_.at(i);
    }
    const_reference at(size_type i) const {
        readElem(elemLocal(i + 1), "at");
        return c_.at(i);
    }
    reference front() {
        if (active()) writeElem(elemLocal(1), "front");
        return c_.front();
    }
    const_reference front() const {
        readElem(elemLocal(1), "front");
        return c_.front();
    }

    // ---- associative containers -----------------------------------------
    // All of these only instantiate when called on a map-like container.
    template <class K>
    auto operator[](const K& k) -> decltype(std::declval<Container&>()[k]) {
        if (active()) {
            writeMeta("operator[].structure");
            writeElem(elemKeyLocal(k), "operator[]");
        }
        return c_[k];
    }
    template <class K>
    size_type erase(const K& k) {
        if (active()) {
            writeMeta("erase.structure");
            writeElem(elemKeyLocal(k), "erase");
        }
        return c_.erase(k);
    }
    template <class K>
    size_type count(const K& k) const {
        if (active()) {
            readMeta("count.structure");
            readElem(elemKeyLocal(k), "count");
        }
        return c_.count(k);
    }

    // ---- string ----------------------------------------------------------
    void assign(const char* s) {
        if (active()) writeElem(elemLocal(1), "assign");
        c_.assign(s);
    }
    void assign(const std::string& s) {
        if (active()) writeElem(elemLocal(1), "assign");
        c_.assign(s);
    }
    void append(const char* s) {
        if (active()) writeElem(elemLocal(1), "append");
        c_.append(s);
    }
    const char* c_str() const {
        readElem(elemLocal(1), "c_str");
        return c_.c_str();
    }
    const char* data() const {
        readElem(elemLocal(1), "data");
        return c_.data();
    }

    // ---- iteration -------------------------------------------------------
    template <bool Const>
    class basic_iterator {
    public:
        using base_iter = std::conditional_t<Const, typename Container::const_iterator,
                                             typename Container::iterator>;
        using iterator_category = std::forward_iterator_tag;
        using value_type = typename Container::value_type;

        basic_iterator() = default;
        basic_iterator(base_iter it, const RaceTrack* owner)
            : it_(it), owner_(owner) {}

        decltype(auto) operator*() const {
            owner_->touchIterDeref(it_, idx_);
            return *it_;
        }
        auto operator->() const { return it_.operator->(); }
        basic_iterator& operator++() {
            ++it_;
            ++idx_;
            return *this;
        }
        bool operator==(const basic_iterator& o) const { return it_ == o.it_; }
        bool operator!=(const basic_iterator& o) const { return it_ != o.it_; }

    private:
        base_iter it_{};
        const RaceTrack* owner_ = nullptr;
        std::size_t idx_ = 0;
    };

    using iterator = basic_iterator<false>;
    using const_iterator = basic_iterator<true>;

    iterator begin() {
        readMeta("begin");
        return iterator(c_.begin(), this);
    }
    iterator end() {
        readMeta("end");
        return iterator(c_.end(), this);
    }
    const_iterator begin() const {
        readMeta("begin");
        return const_iterator(c_.begin(), this);
    }
    const_iterator end() const {
        readMeta("end");
        return const_iterator(c_.end(), this);
    }

    // ---- associative lookup (declared after the iterator aliases) ---------
    template <class K>
    iterator find(const K& k) {
        if (active()) {
            readMeta("find.structure");
            readElem(elemKeyLocal(k), "find");
        }
        return iterator(c_.find(k), this);
    }
    template <class K>
    const_iterator find(const K& k) const {
        if (active()) {
            readMeta("find.structure");
            readElem(elemKeyLocal(k), "find");
        }
        return const_iterator(c_.find(k), this);
    }

private:
    std::uint64_t metaId() const {
        return detail::mix64(cid_ ^ 0xD1B54A32D192ED03ull);
    }
    std::uint64_t elemLocal(std::uint64_t local) const {
        return detail::mix64(cid_ * 0x9E3779B97F4A7C15ull + local);
    }
    std::uint64_t elemKeyLocal(std::uint64_t local) const {
        return detail::mix64(cid_ * 0xC2B2AE3D27D4EB4Full + local + 1);
    }
    template <class K>
    std::uint64_t elemKeyLocal(const K& k) const {
        return elemKeyLocal(static_cast<std::uint64_t>(std::hash<K>{}(k)));
    }

    void readMeta(const char* op) const {
        if (!active()) return;
        detail::touch(metaId(), false, op);
    }
    void writeMeta(const char* op) const {
        if (!active()) return;
        detail::touch(metaId(), true, op);
    }
    void readElem(std::uint64_t id, const char* op) const {
        if (!active()) return;
        detail::touch(id, false, op);
    }
    void writeElem(std::uint64_t id, const char* op) const {
        if (!active()) return;
        detail::touch(id, true, op);
    }

    template <class BaseIter>
    void touchIterDeref(const BaseIter& it, std::size_t idx) const {
        if (!active()) return;
        if constexpr (detail::has_key_type<Container>::value) {
            readElem(elemKeyLocal(static_cast<std::uint64_t>(std::hash<typename Container::key_type>{}(it->first))),
                     "iter-deref");
        } else if constexpr (std::is_same_v<Container, std::string>) {
            readElem(elemLocal(1), "iter-deref");
        } else {
            readElem(elemLocal(idx + 1), "iter-deref");
        }
    }

    Container c_{};
    std::uint64_t cid_ = 0;
};

}  // namespace race_sentinel
