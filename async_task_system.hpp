/* async_task_system.hpp - Work-stealing async task system for Phoenix v7.0
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version. */

#pragma once

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace phoenix {
namespace v7 {

/**
 * @brief Asynchronous task system.
 *
 * v7.0 deliberately does NOT rely on a single global event loop.  Instead it
 * uses a pool of worker threads plus per-module priority queues.  Workers
 * prefer their own module queue, then steal from other modules.  This gives:
 *   - latency isolation (a slow encoder cannot block the backend),
 *   - back-pressure (each queue has a max depth),
 *   - priority awareness (user-facing inference > learning > persistence),
 *   - better CPU utilization on heterogeneous hardware.
 */

enum class TaskPriority : int {
    Critical = 0,   // user-facing inference
    High = 1,       // encoders, GNN lookup
    Normal = 2,     // emotion, memory summary
    Low = 3,        // learning, adaptation
    Background = 4  // persistence, snapshot, metrics
};

enum class TaskModule : int {
    Unknown = 0,
    Ahead,      // pre-GNN processing
    Memory,     // memory summary
    Emotion,    // emotion observation / influence
    Encoder,    // audio/video/text enc
    Decoder,    // audio/video/text dec
    Gnn,        // meme graph
    Backend,    // llama inference
    Learning,   // RL / adversarial / online updates
    Persistence // save to disk / DB
};

std::string taskModuleToString(TaskModule m);
TaskModule taskModuleFromString(const std::string &s);

/** A task is a movable callable with a priority and affinity. */
class AsyncTask {
public:
    using Payload = std::function<void()>;

    AsyncTask(TaskModule module, TaskPriority priority, Payload payload,
              const std::string &tag = "");

    TaskModule module() const { return module_; }
    TaskPriority priority() const { return priority_; }
    const std::string &tag() const { return tag_; }
    void operator()();

    /** For priority ordering (lower numeric = higher priority). */
    bool operator<(const AsyncTask &other) const;

private:
    TaskModule module_;
    TaskPriority priority_;
    Payload payload_;
    std::string tag_;
    uint64_t seq_;
    static std::atomic<uint64_t> s_seq_;
};

/**
 * @brief Priority queue for a single module.
 */
class ModuleQueue {
public:
    void push(AsyncTask task);
    bool tryPop(AsyncTask &task);
    bool empty() const;
    size_t size() const;
    nlohmann::json status() const;

private:
    mutable std::mutex mutex_;
    std::priority_queue<AsyncTask> queue_;
    size_t maxDepth_{0};
};

struct AsyncTaskSystemConfig {
    /** Number of worker threads. 0 = hardware_concurrency. */
    size_t workerCount{0};

    /** Max per-module queue depth. */
    size_t maxQueueDepth{256};

    /** How long a worker waits for new work before checking steal. */
    std::chrono::milliseconds waitTimeout{20};

    /** Whether background tasks run only when CPU is idle. */
    bool backgroundBackPressure{true};

    static AsyncTaskSystemConfig fromJson(const nlohmann::json &j);
    nlohmann::json toJson() const;
};

/**
 * @brief Central scheduler.
 *
 * One AsyncTaskSystem per Phoenix process is enough.  It owns the worker
 * threads and the module queues.  Sub-systems get a TaskModule handle and
 * submit lambdas.
 */
class AsyncTaskSystem {
public:
    explicit AsyncTaskSystem(const AsyncTaskSystemConfig &cfg = AsyncTaskSystemConfig{});
    ~AsyncTaskSystem();

    /** Start worker threads. */
    bool start();

    /** Submit a fire-and-forget task. */
    bool submit(TaskModule module, TaskPriority priority,
                AsyncTask::Payload payload, const std::string &tag = "");

    /** Submit a task and get a future. */
    std::future<void> submitWithFuture(TaskModule module, TaskPriority priority,
                                       AsyncTask::Payload payload,
                                       const std::string &tag = "");

    /** Wait until all queues are empty (for testing / shutdown). */
    void drain();

    /** Wait for one specific module to drain. */
    void drain(TaskModule module);

    /** Stop all workers. */
    void stop();

    /** Status per module. */
    nlohmann::json status() const;

    /** Global singleton accessor. */
    static AsyncTaskSystem &global();

private:
    AsyncTaskSystemConfig cfg_;
    std::vector<std::thread> workers_;
    std::unordered_map<TaskModule, std::unique_ptr<ModuleQueue>> queues_;
    std::atomic<bool> running_{false};
    std::atomic<size_t> activeTasks_{0};
    std::condition_variable_any drainCv_;
    mutable std::mutex statusMutex_;

    void workerLoop_();
    bool tryLocal_(AsyncTask &task, TaskModule preferred);
    bool trySteal_(AsyncTask &task, TaskModule except);
};

} // namespace v7
} // namespace phoenix
