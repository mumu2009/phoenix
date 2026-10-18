/* study_unit_learn.cpp - BPE units, meme graph, deepened propagation. */

#include "study_unit_learn.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace phoenix {
namespace study {

namespace {

std::uint64_t edgeKey(int from, int to) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(from)) << 32) |
           static_cast<std::uint32_t>(to);
}

}  // namespace

void BytePairEncoder::fit(const std::string &text, int maxMerges) {
    maxMerges = std::max(0, std::min(maxMerges, 64));
    merges_.clear();
    surfaces_.assign(256, std::string());
    for (int i = 0; i < 256; ++i) surfaces_[static_cast<size_t>(i)] = std::string(1, static_cast<char>(i));
    std::vector<int> seq;
    seq.reserve(text.size());
    for (unsigned char c : text) seq.push_back(static_cast<int>(c));
    int nextId = 256;
    for (int m = 0; m < maxMerges && seq.size() >= 2; ++m) {
        std::unordered_map<std::uint64_t, int> counts;
        for (size_t i = 0; i + 1 < seq.size(); ++i) {
            const std::uint64_t k = edgeKey(seq[i], seq[i + 1]);
            counts[k] += 1;
        }
        int best = 1;
        std::uint64_t bestK = 0;
        bool found = false;
        for (const auto &kv : counts) {
            if (kv.second > best) {
                best = kv.second;
                bestK = kv.first;
                found = true;
            }
        }
        if (!found) break;
        const int left = static_cast<int>(bestK >> 32);
        const int right = static_cast<int>(bestK & 0xffffffffu);
        std::vector<int> next;
        next.reserve(seq.size());
        for (size_t i = 0; i < seq.size();) {
            if (i + 1 < seq.size() && seq[i] == left && seq[i + 1] == right) {
                next.push_back(nextId);
                i += 2;
            } else {
                next.push_back(seq[i]);
                ++i;
            }
        }
        Merge merge;
        merge.left = left;
        merge.right = right;
        merge.id = nextId;
        merges_.push_back(merge);
        std::string surf;
        if (left >= 0 && left < static_cast<int>(surfaces_.size())) surf += surfaces_[static_cast<size_t>(left)];
        if (right >= 0 && right < static_cast<int>(surfaces_.size())) surf += surfaces_[static_cast<size_t>(right)];
        surfaces_.push_back(std::move(surf));
        ++nextId;
        seq.swap(next);
    }
}

std::vector<int> BytePairEncoder::encode(const std::string &text) const {
    std::vector<int> seq;
    seq.reserve(text.size());
    for (unsigned char c : text) seq.push_back(static_cast<int>(c));
    for (const auto &m : merges_) {
        std::vector<int> next;
        next.reserve(seq.size());
        for (size_t i = 0; i < seq.size();) {
            if (i + 1 < seq.size() && seq[i] == m.left && seq[i + 1] == m.right) {
                next.push_back(m.id);
                i += 2;
            } else {
                next.push_back(seq[i]);
                ++i;
            }
        }
        seq.swap(next);
    }
    return seq;
}

const std::string &BytePairEncoder::surface(int id) const {
    static const std::string kEmpty;
    if (id < 0 || id >= static_cast<int>(surfaces_.size())) return kEmpty;
    return surfaces_[static_cast<size_t>(id)];
}

nlohmann::json BytePairEncoder::toJson() const {
    nlohmann::json merges = nlohmann::json::array();
    for (const auto &m : merges_) {
        merges.push_back(nlohmann::json{{"left", m.left}, {"right", m.right}, {"id", m.id}});
    }
    return nlohmann::json{{"merges", merges}, {"surfaces", surfaces_}};
}

void BytePairEncoder::fromJson(const nlohmann::json &j) {
    merges_.clear();
    surfaces_.clear();
    if (!j.is_object()) return;
    if (j.contains("surfaces") && j["surfaces"].is_array()) {
        for (const auto &s : j["surfaces"]) {
            surfaces_.push_back(s.is_string() ? s.get<std::string>() : std::string());
        }
    }
    if (j.contains("merges") && j["merges"].is_array()) {
        for (const auto &m : j["merges"]) {
            if (!m.is_object()) continue;
            Merge merge;
            merge.left = m.value("left", 0);
            merge.right = m.value("right", 0);
            merge.id = m.value("id", 0);
            merges_.push_back(merge);
        }
    }
}

nlohmann::json UnitLearner::configure(const nlohmann::json &payload) {
    std::lock_guard<std::mutex> lock(mu_);
    return configureLocked(payload);
}

nlohmann::json UnitLearner::configureLocked(const nlohmann::json &payload) {
    const nlohmann::json &p = payload.is_object() ? payload : nlohmann::json::object();
    const int newDim = std::max(4, std::min(64, p.value("dim", dim_)));
    if (newDim != dim_ || p.value("reset", false)) {
        dim_ = newDim;
        nodes_.clear();
        edges_.clear();
        W_.assign(static_cast<size_t>(dim_ * dim_), 0.f);
        lastLoss_ = -1.0;
        episodes_ = 0;
        encoder_ = BytePairEncoder();
        encoderReady_ = false;
    }
    if (W_.size() != static_cast<size_t>(dim_ * dim_)) W_.assign(static_cast<size_t>(dim_ * dim_), 0.f);
    enabled_ = p.value("enabled", enabled_);
    steps_ = std::max(1, std::min(8, p.value("steps", steps_)));
    maxMerges_ = std::max(0, std::min(64, p.value("maxMerges", maxMerges_)));
    lr_ = std::max(0.f, std::min(2.f, p.value("lr", lr_)));
    alpha_ = std::max(0.f, std::min(1.f, p.value("alpha", alpha_)));
    return statusLocked();
}

nlohmann::json UnitLearner::fitEncoder(const std::string &text) {
    std::lock_guard<std::mutex> lock(mu_);
    std::string slice = text.size() > 8192 ? text.substr(0, 8192) : text;
    encoder_.fit(slice, maxMerges_);
    encoderReady_ = true;
    return nlohmann::json{{"ok", true},
                          {"merges", encoder_.mergeCount()},
                          {"writesCausalString", false}};
}

std::vector<int> UnitLearner::encode(const std::string &text) const {
    std::lock_guard<std::mutex> lock(mu_);
    return encoder_.encode(text);
}

bool UnitLearner::enabled() const {
    std::lock_guard<std::mutex> lock(mu_);
    return enabled_;
}

nlohmann::json UnitLearner::learnEpisode(const std::string &text) {
    std::lock_guard<std::mutex> lock(mu_);
    return learnLocked(text);
}

nlohmann::json UnitLearner::learnLocked(const std::string &text) {
    if (!enabled_) {
        return nlohmann::json{{"ok", false},
                              {"reason", "disabled"},
                              {"writesCausalString", false},
                              {"unitCount", 0}};
    }
    std::string slice = text.size() > 512 ? text.substr(0, 512) : text;
    if (!encoderReady_) {
        encoder_.fit(slice, maxMerges_);
        encoderReady_ = true;
    }
    const std::vector<int> ids = encoder_.encode(slice);
    nlohmann::json stages = nlohmann::json::array(
        {"encode-units", "meme-import", "window-edges", "propagate", "predict-update"});
    for (int id : ids) ensureNode(id);
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i + 1 < ids.size()) addEdge(ids[i], ids[i + 1], 0.2f);
        if (i + 2 < ids.size()) addEdge(ids[i], ids[i + 2], 0.05f);
    }
    propagate();

    double loss = 0.0;
    int pairs = 0;
    for (size_t i = 0; i + 1 < ids.size(); ++i) {
        auto itA = nodes_.find(ids[i]);
        auto itB = nodes_.find(ids[i + 1]);
        if (itA == nodes_.end() || itB == nodes_.end()) continue;
        const auto pred = applyW(itA->second.base);
        std::vector<float> err(static_cast<size_t>(dim_));
        double mse = 0.0;
        for (int k = 0; k < dim_; ++k) {
            err[static_cast<size_t>(k)] = itB->second.base[static_cast<size_t>(k)] - pred[static_cast<size_t>(k)];
            mse += static_cast<double>(err[static_cast<size_t>(k)]) * err[static_cast<size_t>(k)];
        }
        loss += mse;
        ++pairs;
        if (lr_ > 0.f) {
            for (int j = 0; j < dim_; ++j) {
                for (int k = 0; k < dim_; ++k) {
                    float &w = W_[static_cast<size_t>(j * dim_ + k)];
                    w = std::clamp(w + lr_ * err[static_cast<size_t>(j)] * itA->second.base[static_cast<size_t>(k)],
                                   -2.f, 2.f);
                }
            }
        }
    }
    if (pairs > 0) loss /= static_cast<double>(pairs);
    lastLoss_ = pairs > 0 ? loss : lastLoss_;
    ++episodes_;
    double stateDelta = 0.0;
    for (int id : ids) {
        auto it = nodes_.find(id);
        if (it == nodes_.end()) continue;
        const auto &base = it->second.base;
        const auto &state = it->second.state;
        const int n = std::min(static_cast<int>(base.size()), static_cast<int>(state.size()));
        for (int k = 0; k < n; ++k) {
            stateDelta = std::max(stateDelta, static_cast<double>(std::fabs(state[static_cast<size_t>(k)] - base[static_cast<size_t>(k)])));
        }
    }
    nlohmann::json out{{"ok", true},
                       {"writesCausalString", false},
                       {"import", "unit"},
                       {"tokenEqualsUnit", true},
                       {"unitCount", static_cast<int>(ids.size())},
                       {"edgeCount", static_cast<int>(edges_.size())},
                       {"nodeCount", static_cast<int>(nodes_.size())},
                       {"steps", steps_},
                       {"merges", encoder_.mergeCount()},
                       {"episodes", episodes_},
                       {"stateDelta", stateDelta},
                       {"stages", stages}};
    if (pairs > 0) out["loss"] = loss;
    return out;
}

nlohmann::json UnitLearner::importUnits(const nlohmann::json &payload) {
    std::lock_guard<std::mutex> lock(mu_);
    return importLocked(payload);
}

nlohmann::json UnitLearner::importLocked(const nlohmann::json &payload) {
    if (!enabled_) {
        return nlohmann::json{{"ok", false}, {"reason", "disabled"}, {"writesCausalString", false}};
    }
    std::vector<int> order;
    auto takeRow = [&](int id, const nlohmann::json &row) {
        ensureNode(id);
        if (row.is_array()) {
            std::vector<float> raw;
            raw.reserve(row.size());
            for (const auto &v : row) {
                if (v.is_number()) raw.push_back(v.get<float>());
            }
            writeBase(id, fitRow(raw, dim_));
        }
        order.push_back(id);
    };
    if (payload.contains("units") && payload["units"].is_array()) {
        for (const auto &u : payload["units"]) {
            if (!u.is_object() || !u.contains("id") || !u["id"].is_number()) continue;
            takeRow(u["id"].get<int>(), u.value("row", nlohmann::json::array()));
        }
    }
    if (payload.contains("memes") && payload["memes"].is_array()) {
        for (const auto &m : payload["memes"]) {
            if (!m.is_object()) continue;
            const std::string surface = m.value("surface", std::string());
            if (surface.empty()) continue;
            if (!encoderReady_) {
                encoder_.fit(surface, maxMerges_);
                encoderReady_ = true;
            }
            const auto ids = encoder_.encode(surface);
            const nlohmann::json row = m.value("row", nlohmann::json::array());
            for (int id : ids) takeRow(id, row);
        }
    }
    for (size_t i = 0; i + 1 < order.size(); ++i) addEdge(order[i], order[i + 1], 0.25f);
    propagate();
    return nlohmann::json{{"ok", true},
                          {"writesCausalString", false},
                          {"import", "unit"},
                          {"unitCount", static_cast<int>(order.size())},
                          {"edgeCount", static_cast<int>(edges_.size())},
                          {"steps", steps_}};
}

nlohmann::json UnitLearner::recall(const nlohmann::json &payload) const {
    std::lock_guard<std::mutex> lock(mu_);
    return recallLocked(payload);
}

nlohmann::json UnitLearner::recallLocked(const nlohmann::json &payload) const {
    std::unordered_map<int, float> score;
    if (payload.contains("units") && payload["units"].is_array()) {
        for (const auto &u : payload["units"]) {
            if (u.is_number()) score[u.get<int>()] += 1.f;
        }
    }
    if (payload.contains("text") && payload["text"].is_string()) {
        for (int id : encoder_.encode(payload["text"].get<std::string>())) score[id] += 1.f;
    }
    for (int step = 0; step < steps_; ++step) {
        std::unordered_map<int, float> nxt = score;
        for (const auto &e : edges_) {
            const int from = static_cast<int>(e.first >> 32);
            const int to = static_cast<int>(e.first & 0xffffffffu);
            auto it = score.find(from);
            if (it == score.end()) continue;
            nxt[to] += it->second * e.second;
        }
        score.swap(nxt);
    }
    std::vector<std::pair<float, int>> ranked;
    ranked.reserve(score.size());
    for (const auto &kv : score) ranked.push_back({kv.second, kv.first});
    std::sort(ranked.begin(), ranked.end(), [](const auto &a, const auto &b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    });
    if (ranked.size() > 8) ranked.resize(8);
    nlohmann::json units = nlohmann::json::array();
    for (const auto &rk : ranked) {
        nlohmann::json row = nlohmann::json::array();
        auto it = nodes_.find(rk.second);
        if (it != nodes_.end()) {
            for (float v : it->second.base) row.push_back(v);
        }
        units.push_back(nlohmann::json{{"id", rk.second}, {"score", rk.first}, {"row", row}});
    }
    return nlohmann::json{{"ok", true},
                          {"writesCausalString", false},
                          {"units", units},
                          {"steps", steps_}};
}

nlohmann::json UnitLearner::status() const {
    std::lock_guard<std::mutex> lock(mu_);
    return statusLocked();
}

nlohmann::json UnitLearner::statusLocked() const {
    return nlohmann::json{{"enabled", enabled_},
                          {"writesCausalString", false},
                          {"dim", dim_},
                          {"steps", steps_},
                          {"maxMerges", maxMerges_},
                          {"merges", encoder_.mergeCount()},
                          {"nodeCount", static_cast<int>(nodes_.size())},
                          {"edgeCount", static_cast<int>(edges_.size())},
                          {"episodes", episodes_},
                          {"lastLoss", lastLoss_},
                          {"tokenEqualsUnit", true}};
}

nlohmann::json UnitLearner::sideChannel(const nlohmann::json &learned) const {
    nlohmann::json j{{"enabled", enabled()},
                     {"writesCausalString", false},
                     {"ok", learned.value("ok", false)},
                     {"unitCount", learned.value("unitCount", 0)},
                     {"edgeCount", learned.value("edgeCount", 0)},
                     {"steps", learned.value("steps", 0)},
                     {"import", learned.value("import", std::string("unit"))}};
    if (learned.contains("loss") && learned["loss"].is_number()) j["loss"] = learned["loss"];
    if (learned.contains("stages")) j["stages"] = learned["stages"];
    return j;
}

std::vector<float> UnitLearner::unitRow(int id) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = nodes_.find(id);
    if (it == nodes_.end()) return {};
    return it->second.base;
}

nlohmann::json UnitLearner::toJson() const {
    std::lock_guard<std::mutex> lock(mu_);
    return toJsonLocked();
}

nlohmann::json UnitLearner::toJsonLocked() const {
    nlohmann::json nodes = nlohmann::json::array();
    for (const auto &kv : nodes_) {
        nlohmann::json base = nlohmann::json::array();
        for (float v : kv.second.base) base.push_back(v);
        nodes.push_back(nlohmann::json{{"id", kv.first}, {"count", kv.second.count}, {"base", base}});
    }
    nlohmann::json edges = nlohmann::json::array();
    for (const auto &e : edges_) {
        edges.push_back(nlohmann::json{{"from", static_cast<int>(e.first >> 32)},
                                       {"to", static_cast<int>(e.first & 0xffffffffu)},
                                       {"w", e.second}});
    }
    return nlohmann::json{{"enabled", enabled_},
                          {"dim", dim_},
                          {"steps", steps_},
                          {"maxMerges", maxMerges_},
                          {"lr", lr_},
                          {"alpha", alpha_},
                          {"episodes", episodes_},
                          {"lastLoss", lastLoss_},
                          {"encoderReady", encoderReady_},
                          {"encoder", encoder_.toJson()},
                          {"W", W_},
                          {"nodes", nodes},
                          {"edges", edges}};
}

void UnitLearner::fromJson(const nlohmann::json &j) {
    std::lock_guard<std::mutex> lock(mu_);
    fromJsonLocked(j);
}

void UnitLearner::fromJsonLocked(const nlohmann::json &j) {
    if (!j.is_object()) return;
    dim_ = std::max(4, j.value("dim", dim_));
    steps_ = j.value("steps", steps_);
    maxMerges_ = j.value("maxMerges", maxMerges_);
    lr_ = j.value("lr", lr_);
    alpha_ = j.value("alpha", alpha_);
    enabled_ = j.value("enabled", enabled_);
    episodes_ = j.value("episodes", 0);
    lastLoss_ = j.value("lastLoss", -1.0);
    encoderReady_ = j.value("encoderReady", false);
    if (j.contains("encoder")) encoder_.fromJson(j["encoder"]);
    W_.clear();
    if (j.contains("W") && j["W"].is_array()) {
        for (const auto &v : j["W"])
            if (v.is_number()) W_.push_back(v.get<float>());
    }
    if (W_.size() != static_cast<size_t>(dim_ * dim_)) W_.assign(static_cast<size_t>(dim_ * dim_), 0.f);
    nodes_.clear();
    if (j.contains("nodes") && j["nodes"].is_array()) {
        for (const auto &n : j["nodes"]) {
            if (!n.is_object()) continue;
            Node node;
            node.count = n.value("count", 0);
            if (n.contains("base") && n["base"].is_array()) {
                for (const auto &v : n["base"])
                    if (v.is_number()) node.base.push_back(v.get<float>());
            }
            node.state = node.base;
            nodes_[n.value("id", 0)] = std::move(node);
        }
    }
    edges_.clear();
    if (j.contains("edges") && j["edges"].is_array()) {
        for (const auto &e : j["edges"]) {
            if (!e.is_object()) continue;
            const int from = e.value("from", 0);
            const int to = e.value("to", 0);
            edges_[edgeKey(from, to)] = e.value("w", 0.f);
        }
    }
}

void UnitLearner::ensureNode(int id) {
    if (nodes_.find(id) != nodes_.end()) {
        nodes_[id].count += 1;
        return;
    }
    if (static_cast<int>(nodes_.size()) >= maxNodes_) return;
    Node node;
    node.count = 1;
    initBase(node.base, id);
    node.state = node.base;
    nodes_.emplace(id, std::move(node));
}

void UnitLearner::initBase(std::vector<float> &dst, int id) const {
    dst.assign(static_cast<size_t>(dim_), 0.f);
    std::uint32_t x = static_cast<std::uint32_t>(id) * 2654435761u + 0x9e3779b9u;
    double ss = 0.0;
    for (int i = 0; i < dim_; ++i) {
        x = x * 1664525u + 1013904223u;
        const float v = static_cast<float>((x >> 8) & 255) / 127.5f - 1.f;
        dst[static_cast<size_t>(i)] = v;
        ss += static_cast<double>(v) * v;
    }
    const float inv = ss > 1e-12 ? static_cast<float>(1.0 / std::sqrt(ss)) : 0.f;
    for (float &v : dst) v *= inv;
}

void UnitLearner::writeBase(int id, const std::vector<float> &row) {
    auto it = nodes_.find(id);
    if (it == nodes_.end()) return;
    it->second.base = row.size() == static_cast<size_t>(dim_) ? row : fitRow(row, dim_);
    it->second.state = it->second.base;
}

void UnitLearner::addEdge(int from, int to, float w) {
    if (from == to) return;
    if (!nodes_.count(from) || !nodes_.count(to)) return;
    const std::uint64_t key = edgeKey(from, to);
    auto it = edges_.find(key);
    if (it == edges_.end()) {
        if (static_cast<int>(edges_.size()) >= maxEdges_) return;
        edges_.emplace(key, std::min(8.f, w));
        return;
    }
    it->second = std::min(8.f, it->second + w);
}

void UnitLearner::propagate() {
    if (nodes_.empty() || steps_ <= 0) return;
    std::unordered_map<int, std::vector<float>> cur;
    for (auto &kv : nodes_) cur[kv.first] = kv.second.base;
    for (int step = 0; step < steps_; ++step) {
        std::unordered_map<int, std::vector<float>> acc;
        std::unordered_map<int, float> wsum;
        for (const auto &e : edges_) {
            const int from = static_cast<int>(e.first >> 32);
            const int to = static_cast<int>(e.first & 0xffffffffu);
            auto it = cur.find(from);
            if (it == cur.end() || it->second.size() != static_cast<size_t>(dim_)) continue;
            auto &dst = acc[to];
            if (dst.empty()) dst.assign(static_cast<size_t>(dim_), 0.f);
            for (int k = 0; k < dim_; ++k) dst[static_cast<size_t>(k)] += it->second[static_cast<size_t>(k)] * e.second;
            wsum[to] += e.second;
        }
        for (auto &kv : cur) {
            auto it = acc.find(kv.first);
            if (it == acc.end() || wsum[kv.first] <= 1e-8f) continue;
            const float inv = 1.f / wsum[kv.first];
            std::vector<float> mixed(static_cast<size_t>(dim_));
            const auto &base = nodes_[kv.first].base;
            for (int k = 0; k < dim_; ++k) {
                const float msg = it->second[static_cast<size_t>(k)] * inv;
                mixed[static_cast<size_t>(k)] =
                    (1.f - alpha_) * base[static_cast<size_t>(k)] + alpha_ * msg;
            }
            kv.second = normalize(std::move(mixed));
        }
    }
    for (auto &kv : nodes_) {
        auto it = cur.find(kv.first);
        if (it != cur.end()) kv.second.state = it->second;
    }
}

std::vector<float> UnitLearner::applyW(const std::vector<float> &h) const {
    std::vector<float> y(static_cast<size_t>(dim_), 0.f);
    if (static_cast<int>(h.size()) != dim_ || static_cast<int>(W_.size()) != dim_ * dim_) return y;
    for (int j = 0; j < dim_; ++j) {
        float s = 0.f;
        for (int i = 0; i < dim_; ++i) s += W_[static_cast<size_t>(j * dim_ + i)] * h[static_cast<size_t>(i)];
        y[static_cast<size_t>(j)] = s;
    }
    return y;
}

std::vector<float> UnitLearner::normalize(std::vector<float> v) {
    double ss = 0.0;
    for (float x : v) ss += static_cast<double>(x) * x;
    if (ss < 1e-12) return v;
    const float inv = static_cast<float>(1.0 / std::sqrt(ss));
    for (float &x : v) x *= inv;
    return v;
}

std::vector<float> UnitLearner::fitRow(const std::vector<float> &row, int dim) {
    std::vector<float> out(static_cast<size_t>(dim), 0.f);
    if (row.empty() || dim <= 0) return out;
    for (int i = 0; i < dim; ++i) out[static_cast<size_t>(i)] = row[static_cast<size_t>(i % row.size())];
    return normalize(std::move(out));
}

}  // namespace study
}  // namespace phoenix
