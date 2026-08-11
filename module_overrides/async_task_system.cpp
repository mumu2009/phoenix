/* async_task_system.cpp - Work-stealing async task system for Phoenix v7.0
   Copyright (C) 2026 079 Project */

#include "async_task_system.hpp"

#include <atomic>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace phoenix {
namespace v7 {

std::string taskModuleToString(TaskModule m) {
    switch (m) {
        case TaskModule::Ahead: return "ahead";
        case TaskModule::Memory: return "memory";
        case TaskModule::Emotion: return "emotion";
        case TaskModule::Encoder: return "encoder";
        case TaskModule::Decoder: return "decoder";
        case TaskModule::Gnn: return "gnn";
        case TaskModule::Backend: return "backend";
        case TaskModule::Learning: return "learning";
        case TaskModule::Persistence: return "persistence";
        default: return "unknown";
    }
}

TaskModule taskModuleFromString(const std::string &s) {
    if (s == "ahead") return TaskModule::Ahead;
    if (s == "memory") return TaskModule::Memory;
    if (s == "emotion") return TaskModule::Emotion;
    if (s == "encoder") return TaskModule::Encoder;
    if (s == "decoder") return TaskModule::Decoder;
    if (s == "gnn") return TaskModule::Gnn;
    if (s == "backend") return TaskModule::Backend;
    if (s == "learning") return TaskModule::Learning;
    if (s == "persistence") return TaskModule::Persistence;
    return TaskModule::Unknown;
}

std::atomic<uint64_t> AsyncTask::s_seq_{0};

AsyncTask::AsyncTask(TaskModule module, TaskPriority priority, Payload payload,
                     const std::string &tag)
    : module_(module), priority_(priority), payload_(std::move(payload)),
      tag_(tag), seq_(s_seq_++) {}

void AsyncTask::operator()() {
    if (payload_) payload_();
}

bool AsyncTask::operator<(const AsyncTask &other) const {
    // Priority queue default is max-heap, so invert.
    if (static_cast<int>(priority_) != static_cast<int>(other.priority_)) {
        return static_cast<int>(priority_) > static_cast<int>(other.priority_);
    }
    return seq_ > other.seq_;
}

void ModuleQueue::push(AsyncTask task) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(std::move(task));
    if (queue_.size() > maxDepth_) maxDepth_ = queue_.size();
}

bool ModuleQueue::tryPop(AsyncTask &task) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) return false;
    task = std::move(const_cast<AsyncTask &>(queue_.top()));
    queue_.pop();
    return true;
}

bool ModuleQueue::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}

size_t ModuleQueue::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

nlohmann::json ModuleQueue::status() const {
    nlohmann::json j;
    std::lock_guard<std::mutex> lock(mutex_);
    j["current"] = queue_.size();
    j["maxDepth"] = maxDepth_;
    return j;
}

AsyncTaskSystemConfig AsyncTaskSystemConfig::fromJson(const nlohmann::json &j) {
    AsyncTaskSystemConfig c;
    if (j.contains("workerCount")) c.workerCount = j["workerCount"].get<size_t>();
    if (j.contains("maxQueueDepth")) c.maxQueueDepth = j["maxQueueDepth"].get<size_t>();
    if (j.contains("waitTimeoutMs")) c.waitTimeout = std::chrono::milliseconds(j["waitTimeoutMs"].get<int>());
    if (j.contains("backgroundBackPressure")) c.backgroundBackPressure = j["backgroundBackPressure"].get<bool>();
    return c;
}

nlohmann::json AsyncTaskSystemConfig::toJson() const {
    return {
        {"workerCount", workerCount},
        {"maxQueueDepth", maxQueueDepth},
        {"waitTimeoutMs", waitTimeout.count()},
        {"backgroundBackPressure", backgroundBackPressure}
    };
}

AsyncTaskSystem::AsyncTaskSystem(const AsyncTaskSystemConfig &cfg) : cfg_(cfg) {
    queues_[TaskModule::Ahead] = std::make_unique<ModuleQueue>();
    queues_[TaskModule::Memory] = std::make_unique<ModuleQueue>();
    queues_[TaskModule::Emotion] = std::make_unique<ModuleQueue>();
    queues_[TaskModule::Encoder] = std::make_unique<ModuleQueue>();
    queues_[TaskModule::Decoder] = std::make_unique<ModuleQueue>();
    queues_[TaskModule::Gnn] = std::make_unique<ModuleQueue>();
    queues_[TaskModule::Backend] = std::make_unique<ModuleQueue>();
    queues_[TaskModule::Learning] = std::make_unique<ModuleQueue>();
    queues_[TaskModule::Persistence] = std::make_unique<ModuleQueue>();
}

AsyncTaskSystem::~AsyncTaskSystem() {
    stop();
}

bool AsyncTaskSystem::start() {
    if (running_.exchange(true)) return true;
    size_t n = cfg_.workerCount;
    if (n == 0) n = std::max<size_t>(1, std::thread::hardware_concurrency());
    for (size_t i = 0; i < n; ++i) {
        workers_.emplace_back([this] { workerLoop_(); });
    }
    return true;
}

bool AsyncTaskSystem::submit(TaskModule module, TaskPriority priority,
                             AsyncTask::Payload payload, const std::string &tag) {
    auto it = queues_.find(module);
    if (it == queues_.end() || !running_) return false;
    if (it->second->size() >= cfg_.maxQueueDepth) return false;
    it->second->push(AsyncTask(module, priority, std::move(payload), tag));
    return true;
}

std::future<void> AsyncTaskSystem::submitWithFuture(TaskModule module, TaskPriority priority,
                                                    AsyncTask::Payload payload,
                                                    const std::string &tag) {
    auto promise = std::make_shared<std::promise<void>>();
    auto future = promise->get_future();
    auto wrapped = [p = promise, f = std::move(payload)]() mutable {
        try {
            if (f) f();
            p->set_value();
        } catch (...) {
            p->set_exception(std::current_exception());
        }
    };
    if (!submit(module, priority, std::move(wrapped), tag)) {
        promise->set_exception(std::make_exception_ptr(std::runtime_error("submit failed")));
    }
    return future;
}

void AsyncTaskSystem::workerLoop_() {
    // Simple round-robin stealing: each worker has a preferred module based on
    // thread id, then tries global highest-priority work.
    while (running_) {
        AsyncTask task(TaskModule::Unknown, TaskPriority::Background, nullptr);
        bool got = false;

        for (auto &[mod, q] : queues_) {
            if (!q->empty()) {
                got = q->tryPop(task);
                if (got) break;
            }
        }

        if (!got) {
            // No work; sleep briefly.
            std::this_thread::sleep_for(cfg_.waitTimeout);
            continue;
        }

        ++activeTasks_;
        try {
            task();
        } catch (...) {
            // Tasks are expected to handle their own exceptions.
        }
        --activeTasks_;
        drainCv_.notify_all();
    }
}

bool AsyncTaskSystem::tryLocal_(AsyncTask &, TaskModule) {
    return false; // kept for future per-worker affinity
}

bool AsyncTaskSystem::trySteal_(AsyncTask &, TaskModule) {
    return false; // workerLoop_ already steals globally
}

void AsyncTaskSystem::drain() {
    std::unique_lock<std::mutex> lock(statusMutex_);
    drainCv_.wait(lock, [this] {
        for (const auto &[mod, q] : queues_) {
            if (!q->empty()) return false;
        }
        return activeTasks_.load() == 0;
    });
}

void AsyncTaskSystem::drain(TaskModule module) {
    auto it = queues_.find(module);
    if (it == queues_.end()) return;
    std::unique_lock<std::mutex> lock(statusMutex_);
    drainCv_.wait(lock, [this, &it] { return it->second->empty() && activeTasks_.load() == 0; });
}

void AsyncTaskSystem::stop() {
    if (!running_.exchange(false)) return;
    drainCv_.notify_all();
    for (auto &t : workers_) {
        if (t.joinable()) t.join();
    }
    workers_.clear();
}

nlohmann::json AsyncTaskSystem::status() const {
    nlohmann::json j;
    j["running"] = running_.load();
    j["activeTasks"] = activeTasks_.load();
    j["workers"] = workers_.size();
    j["queues"] = nlohmann::json::object();
    for (const auto &[mod, q] : queues_) {
        j["queues"][taskModuleToString(mod)] = q->status();
    }
    return j;
}

AsyncTaskSystem &AsyncTaskSystem::global() {
    static AsyncTaskSystem inst;
    return inst;
}

} // namespace v7
} // namespace phoenix
