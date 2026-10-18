/* tiny_expert.cpp - Train, save, load, and summon a tiny goal expert. */

#include "tiny_expert.hpp"

#include "study_unit_learn.hpp"
#include "summon_registry.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <unordered_map>

namespace phoenix {
namespace study {

namespace {

float clampf(float v, float lo, float hi) { return std::max(lo, std::min(hi, v)); }

std::string readCap(const std::string &path, size_t cap) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::string data;
    data.resize(cap);
    in.read(data.data(), static_cast<std::streamsize>(cap));
    data.resize(static_cast<size_t>(in.gcount()));
    return data;
}

}  // namespace

nlohmann::json TinyExpert::configure(const nlohmann::json &payload) {
    std::lock_guard<std::mutex> lock(mu_);
    return configureLocked(payload);
}

nlohmann::json TinyExpert::configureLocked(const nlohmann::json &payload) {
    const nlohmann::json &p = payload.is_object() ? payload : nlohmann::json::object();
    const int vocab = std::max(2, std::min(64, p.value("vocab", vocab_)));
    const int hidden = std::max(2, std::min(32, p.value("hidden", hidden_)));
    const int classes = std::max(2, std::min(64, p.value("classes", classes_)));
    const bool shapeChanged = vocab != vocab_ || hidden != hidden_ || classes != classes_;
    vocab_ = vocab;
    hidden_ = hidden;
    classes_ = classes;
    if (p.contains("enabled")) enabled_ = p.value("enabled", false);
    if (p.contains("lr")) lr_ = clampf(p.value("lr", lr_), 0.f, 2.f);
    if (p.contains("epochs")) epochs_ = std::max(1, std::min(32, p.value("epochs", epochs_)));
    if (p.contains("corpusPath") && p["corpusPath"].is_string())
        corpusPath_ = p["corpusPath"].get<std::string>();
    if (p.contains("weightsPath") && p["weightsPath"].is_string())
        weightsPath_ = p["weightsPath"].get<std::string>();
    if (p.contains("goal") && p["goal"].is_string()) goal_ = p["goal"].get<std::string>();
    if (shapeChanged || p.value("reset", false) || W1_.empty()) initWeightsLocked(p.value("seed", 1));
    auto st = nlohmann::json{{"ok", true},
                             {"enabled", enabled_},
                             {"vocab", vocab_},
                             {"hidden", hidden_},
                             {"classes", classes_},
                             {"paramCount", paramCount()},
                             {"ready", ready_}};
    return st;
}

void TinyExpert::initWeightsLocked(unsigned seed) {
    std::mt19937 rng(seed ? seed : 1u);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    W1_.assign(static_cast<size_t>(hidden_ * vocab_), 0.f);
    b1_.assign(static_cast<size_t>(hidden_), 0.f);
    W2_.assign(static_cast<size_t>(classes_ * hidden_), 0.f);
    b2_.assign(static_cast<size_t>(classes_), 0.f);
    for (float &v : W1_) v = dist(rng);
    for (float &v : W2_) v = dist(rng);
    ready_ = true;
}

int TinyExpert::paramCount() const {
    return hidden_ * vocab_ + hidden_ + classes_ * hidden_ + classes_;
}

bool TinyExpert::enabled() const {
    std::lock_guard<std::mutex> lock(mu_);
    return enabled_;
}

std::vector<float> TinyExpert::forwardLocked(const std::vector<float> &x, std::vector<float> *hidden) const {
    std::vector<float> z(static_cast<size_t>(classes_), 0.f);
    if (static_cast<int>(x.size()) != vocab_ || static_cast<int>(W1_.size()) != hidden_ * vocab_) return z;
    std::vector<float> h(static_cast<size_t>(hidden_), 0.f);
    for (int j = 0; j < hidden_; ++j) {
        float s = b1_[static_cast<size_t>(j)];
        for (int i = 0; i < vocab_; ++i)
            s += W1_[static_cast<size_t>(j * vocab_ + i)] * x[static_cast<size_t>(i)];
        h[static_cast<size_t>(j)] = std::tanh(s);
    }
    for (int k = 0; k < classes_; ++k) {
        float s = b2_[static_cast<size_t>(k)];
        for (int j = 0; j < hidden_; ++j)
            s += W2_[static_cast<size_t>(k * hidden_ + j)] * h[static_cast<size_t>(j)];
        z[static_cast<size_t>(k)] = s;
    }
    if (hidden) *hidden = std::move(h);
    return z;
}

double TinyExpert::epochLocked(std::vector<Sample> &data, float lr, unsigned seed) {
    if (data.empty()) return 0.0;
    std::mt19937 rng(seed ? seed : 1u);
    std::shuffle(data.begin(), data.end(), rng);
    double loss = 0.0;
    for (const auto &sample : data) {
        if (static_cast<int>(sample.x.size()) != vocab_ || sample.y < 0 || sample.y >= classes_) continue;
        std::vector<float> h;
        const auto z = forwardLocked(sample.x, &h);
        float m = z.empty() ? 0.f : z[0];
        for (float v : z) m = std::max(m, v);
        std::vector<float> p(z.size(), 0.f);
        float sum = 0.f;
        for (size_t k = 0; k < z.size(); ++k) {
            p[k] = std::exp(z[k] - m);
            sum += p[k];
        }
        const float inv = sum > 0.f ? 1.f / sum : 0.f;
        for (float &v : p) v *= inv;
        const float py = p[static_cast<size_t>(sample.y)];
        loss += -std::log(std::max(py, 1.0e-8f));
        if (lr <= 0.f) continue;
        std::vector<float> dZ = p;
        dZ[static_cast<size_t>(sample.y)] -= 1.f;
        std::vector<float> dh(static_cast<size_t>(hidden_), 0.f);
        for (int j = 0; j < hidden_; ++j) {
            for (int k = 0; k < classes_; ++k)
                dh[static_cast<size_t>(j)] += dZ[static_cast<size_t>(k)] * W2_[static_cast<size_t>(k * hidden_ + j)];
        }
        for (int k = 0; k < classes_; ++k) {
            for (int j = 0; j < hidden_; ++j) {
                float &w = W2_[static_cast<size_t>(k * hidden_ + j)];
                w = clampf(w - lr * dZ[static_cast<size_t>(k)] * h[static_cast<size_t>(j)], -3.f, 3.f);
            }
            b2_[static_cast<size_t>(k)] = clampf(b2_[static_cast<size_t>(k)] - lr * dZ[static_cast<size_t>(k)], -3.f, 3.f);
        }
        for (int j = 0; j < hidden_; ++j) {
            const float hj = h[static_cast<size_t>(j)];
            const float dpre = dh[static_cast<size_t>(j)] * (1.f - hj * hj);
            for (int i = 0; i < vocab_; ++i) {
                float &w = W1_[static_cast<size_t>(j * vocab_ + i)];
                w = clampf(w - lr * dpre * sample.x[static_cast<size_t>(i)], -3.f, 3.f);
            }
            b1_[static_cast<size_t>(j)] = clampf(b1_[static_cast<size_t>(j)] - lr * dpre, -3.f, 3.f);
        }
    }
    return loss / static_cast<double>(data.size());
}

nlohmann::json TinyExpert::trainForGoal(const nlohmann::json &payload) {
    std::unique_lock<std::mutex> lock(mu_);
    if (!enabled_) {
        return nlohmann::json{{"ok", false}, {"ran", false}, {"reason", "disabled"}};
    }
    const nlohmann::json &p = payload.is_object() ? payload : nlohmann::json::object();
    if (p.contains("goal") && p["goal"].is_string()) goal_ = p["goal"].get<std::string>();
    const int epochs = std::max(1, std::min(32, p.value("epochs", epochs_)));
    const float lr = p.contains("lr") ? clampf(p.value("lr", lr_), 0.f, 2.f) : lr_;
    const unsigned seed = static_cast<unsigned>(std::max(1, p.value("seed", 1)));

    std::vector<Sample> data;
    bool filtered = false;
    std::string mode = "samples";

    if (p.contains("samples") && p["samples"].is_array()) {
        for (const auto &s : p["samples"]) {
            if (!s.is_object() || !s.contains("x") || !s["x"].is_array()) continue;
            Sample sample;
            sample.y = s.value("y", 0);
            for (const auto &v : s["x"])
                if (v.is_number()) sample.x.push_back(v.get<float>());
            if (static_cast<int>(sample.x.size()) == vocab_ && sample.y >= 0 && sample.y < classes_)
                data.push_back(std::move(sample));
        }
    } else {
        mode = "next-unit";
        std::string path = p.value("corpusPath", corpusPath_);
        if (path.empty()) return nlohmann::json{{"ok", false}, {"ran", false}, {"reason", "no-corpus"}};
        const std::string text = readCap(path, 120000);
        if (text.size() < 64) return nlohmann::json{{"ok", false}, {"ran", false}, {"reason", "corpus-short"}};
        const std::filesystem::path provPath = std::filesystem::path(path).parent_path() / "PROVENANCE.json";
        std::ifstream provIn(provPath);
        if (provIn) {
            try {
                const auto prov = nlohmann::json::parse(provIn);
                source_ = prov.value("sourceUrl", std::string());
                license_ = prov.value("license", std::string());
            } catch (...) {
            }
        }
        BytePairEncoder encoder;
        const std::string fitSlice = text.size() > 8000 ? text.substr(0, 8000) : text;
        encoder.fit(fitSlice, 24);
        std::string body = text.size() > 60000 ? text.substr(0, 60000) : text;
        auto ids = encoder.encode(body);
        if (ids.size() > 20000) ids.resize(20000);
        const std::string goal = goal_.empty() ? std::string("article") : goal_;
        const auto goalIdsVec = encoder.encode(goal);
        std::unordered_map<int, int> goalSet;
        for (int id : goalIdsVec) goalSet[id] = 1;
        std::vector<int> positions;
        positions.reserve(ids.size());
        for (size_t i = 0; i + 1 < ids.size(); ++i) {
            if (goalSet.count(ids[i]) || goalSet.count(ids[i + 1])) positions.push_back(static_cast<int>(i));
        }
        filtered = static_cast<int>(positions.size()) >= 64;
        if (!filtered) {
            positions.clear();
            for (size_t i = 0; i + 1 < ids.size(); ++i) positions.push_back(static_cast<int>(i));
        }
        std::unordered_map<int, int> freq;
        for (int pos : positions) {
            freq[ids[static_cast<size_t>(pos)]] += 1;
            freq[ids[static_cast<size_t>(pos) + 1]] += 1;
        }
        std::vector<std::pair<int, int>> ranked;
        ranked.reserve(freq.size());
        for (const auto &kv : freq) ranked.push_back({kv.second, kv.first});
        std::sort(ranked.begin(), ranked.end(), [](const auto &a, const auto &b) {
            if (a.first != b.first) return a.first > b.first;
            return a.second < b.second;
        });
        if (static_cast<int>(ranked.size()) < 2)
            return nlohmann::json{{"ok", false}, {"ran", false}, {"reason", "vocab-short"}};
        const int useV = std::min(vocab_, static_cast<int>(ranked.size()));
        vocab_ = useV;
        classes_ = useV;
        std::unordered_map<int, int> cls;
        for (int i = 0; i < useV; ++i) cls[ranked[static_cast<size_t>(i)].second] = i;
        for (int pos : positions) {
            const int a = ids[static_cast<size_t>(pos)];
            const int b = ids[static_cast<size_t>(pos) + 1];
            auto ia = cls.find(a);
            auto ib = cls.find(b);
            if (ia == cls.end() || ib == cls.end()) continue;
            Sample sample;
            sample.x.assign(static_cast<size_t>(vocab_), 0.f);
            sample.x[static_cast<size_t>(ia->second)] = 1.f;
            sample.y = ib->second;
            data.push_back(std::move(sample));
            if (data.size() >= 1500) break;
        }
        initWeightsLocked(seed);
    }

    if (data.size() < 16) {
        return nlohmann::json{{"ok", false},
                              {"ran", false},
                              {"reason", "too-few-samples"},
                              {"samples", static_cast<int>(data.size())}};
    }
    if (W1_.empty()) initWeightsLocked(seed);
    const double initial = epochLocked(data, 0.f, seed);
    double last = initial;
    for (int ep = 0; ep < epochs; ++ep) last = epochLocked(data, lr, seed + 17u + static_cast<unsigned>(ep));
    lastLoss_ = last;
    ready_ = true;
    int correct = 0;
    for (const auto &sample : data) {
        const auto got = predictLocked(sample.x);
        if (got.value("class", -1) == sample.y) ++correct;
    }
    const double acc = static_cast<double>(correct) / static_cast<double>(data.size());
    nlohmann::json out{{"ok", true},
                       {"ran", true},
                       {"mode", mode},
                       {"goal", goal_},
                       {"samples", static_cast<int>(data.size())},
                       {"epochs", epochs},
                       {"initialLoss", initial},
                       {"finalLoss", last},
                       {"accuracy", acc},
                       {"paramCount", paramCount()},
                       {"filtered", filtered},
                       {"source", source_},
                       {"license", license_},
                       {"vocab", vocab_},
                       {"hidden", hidden_},
                       {"classes", classes_}};
    std::string outPath = p.value("weightsPath", weightsPath_);
    if (!outPath.empty()) weightsPath_ = outPath;
    lock.unlock();
    if (!outPath.empty()) {
        out["saved"] = save(outPath).value("ok", false);
        out["weightsPath"] = outPath;
    }
    return out;
}

nlohmann::json TinyExpert::predict(const nlohmann::json &payload) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (!payload.is_object() || !payload.contains("x") || !payload["x"].is_array()) {
        return nlohmann::json{{"ok", false}, {"reason", "x"}};
    }
    std::vector<float> x;
    for (const auto &v : payload["x"])
        if (v.is_number()) x.push_back(v.get<float>());
    return predictLocked(x);
}

nlohmann::json TinyExpert::predictLocked(const std::vector<float> &x) const {
    if (!ready_) return nlohmann::json{{"ok", false}, {"reason", "not-ready"}};
    const auto z = forwardLocked(x, nullptr);
    if (z.empty() || static_cast<int>(x.size()) != vocab_)
        return nlohmann::json{{"ok", false}, {"reason", "shape"}};
    float m = z[0];
    for (float v : z) m = std::max(m, v);
    std::vector<float> p(z.size(), 0.f);
    float sum = 0.f;
    for (size_t k = 0; k < z.size(); ++k) {
        p[k] = std::exp(z[k] - m);
        sum += p[k];
    }
    int best = 0;
    nlohmann::json probs = nlohmann::json::array();
    for (size_t k = 0; k < p.size(); ++k) {
        p[k] = sum > 0.f ? p[k] / sum : 0.f;
        probs.push_back(p[k]);
        if (p[k] > p[static_cast<size_t>(best)]) best = static_cast<int>(k);
    }
    return nlohmann::json{{"ok", true}, {"class", best}, {"probs", probs}};
}

nlohmann::json TinyExpert::save(const std::string &path) const {
    if (path.empty() || !ready_) return nlohmann::json{{"ok", false}, {"reason", "not-ready"}};
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
    std::ofstream out(path, std::ios::trunc);
    if (!out) return nlohmann::json{{"ok", false}, {"reason", "io"}};
    out << toJson().dump();
    return nlohmann::json{{"ok", true}, {"path", path}, {"paramCount", paramCount()}};
}

nlohmann::json TinyExpert::load(const std::string &path) {
    std::lock_guard<std::mutex> lock(mu_);
    std::ifstream in(path);
    if (!in) return nlohmann::json{{"ok", false}, {"reason", "io"}};
    try {
        const auto j = nlohmann::json::parse(in);
        fromJson(j);
    } catch (const std::exception &e) {
        return nlohmann::json{{"ok", false}, {"reason", e.what()}};
    }
    weightsPath_ = path;
    return nlohmann::json{{"ok", ready_}, {"paramCount", paramCount()}, {"path", path}};
}

bool TinyExpert::registerSummon(const std::string &name) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (!enabled_ || !ready_ || weightsPath_.empty()) return false;
    phoenix::summon::ModelEndpoint ep;
    ep.name = name.empty() ? std::string("tiny-expert") : name;
    ep.kind = "tiny-expert";
    ep.weights = weightsPath_;
    ep.description = goal_.empty() ? std::string("local tiny expert") : goal_;
    ep.maxTokens = 1;
    ep.timeoutMs = 1000;
    const auto rep = phoenix::summon::SummonRegistry::instance().upsert(ep);
    return rep.value("ok", false);
}

nlohmann::json TinyExpert::status() const {
    std::lock_guard<std::mutex> lock(mu_);
    return nlohmann::json{{"enabled", enabled_},
                          {"ready", ready_},
                          {"vocab", vocab_},
                          {"hidden", hidden_},
                          {"classes", classes_},
                          {"paramCount", paramCount()},
                          {"goal", goal_},
                          {"source", source_},
                          {"license", license_},
                          {"weightsPath", weightsPath_},
                          {"lastLoss", lastLoss_}};
}

nlohmann::json TinyExpert::toJson() const {
    std::lock_guard<std::mutex> lock(mu_);
    return nlohmann::json{{"enabled", enabled_},
                          {"ready", ready_},
                          {"vocab", vocab_},
                          {"hidden", hidden_},
                          {"classes", classes_},
                          {"lr", lr_},
                          {"epochs", epochs_},
                          {"goal", goal_},
                          {"source", source_},
                          {"license", license_},
                          {"weightsPath", weightsPath_},
                          {"corpusPath", corpusPath_},
                          {"lastLoss", lastLoss_},
                          {"paramCount", paramCount()},
                          {"W1", W1_},
                          {"b1", b1_},
                          {"W2", W2_},
                          {"b2", b2_}};
}

void TinyExpert::fromJson(const nlohmann::json &j) {
    if (!j.is_object()) return;
    enabled_ = j.value("enabled", enabled_);
    vocab_ = j.value("vocab", vocab_);
    hidden_ = j.value("hidden", hidden_);
    classes_ = j.value("classes", classes_);
    lr_ = j.value("lr", lr_);
    epochs_ = j.value("epochs", epochs_);
    goal_ = j.value("goal", std::string());
    source_ = j.value("source", std::string());
    license_ = j.value("license", std::string());
    weightsPath_ = j.value("weightsPath", std::string());
    corpusPath_ = j.value("corpusPath", std::string());
    lastLoss_ = j.value("lastLoss", -1.0);
    auto take = [](const nlohmann::json &arr, std::vector<float> &dst) {
        dst.clear();
        if (!arr.is_array()) return;
        for (const auto &v : arr)
            if (v.is_number()) dst.push_back(v.get<float>());
    };
    take(j.value("W1", nlohmann::json::array()), W1_);
    take(j.value("b1", nlohmann::json::array()), b1_);
    take(j.value("W2", nlohmann::json::array()), W2_);
    take(j.value("b2", nlohmann::json::array()), b2_);
    ready_ = static_cast<int>(W1_.size()) == hidden_ * vocab_ &&
             static_cast<int>(b1_.size()) == hidden_ &&
             static_cast<int>(W2_.size()) == classes_ * hidden_ &&
             static_cast<int>(b2_.size()) == classes_;
}

}  // namespace study
}  // namespace phoenix
