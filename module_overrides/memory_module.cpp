/* memory_module.cpp - Explicit memory branch with dual-summary design
   Copyright (C) 2026 079 Project */

#include "memory_module.hpp"
#include "async_task_system.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace phoenix {
namespace v7 {

MemoryConfig MemoryConfig::fromJson(const nlohmann::json &j) {
    MemoryConfig c;
    if (j.contains("cellType")) {
        auto s = j["cellType"].get<std::string>();
        if (s == "RNN") c.cellType = MemoryConfig::CellType::RNN;
        else if (s == "LSTM") c.cellType = MemoryConfig::CellType::LSTM;
        else c.cellType = MemoryConfig::CellType::Transformer;
    }
    if (j.contains("inputDim")) c.inputDim = j["inputDim"].get<int>();
    if (j.contains("hiddenDim")) c.hiddenDim = j["hiddenDim"].get<int>();
    if (j.contains("outputDim")) c.outputDim = j["outputDim"].get<int>();
    if (j.contains("maxSummaryTokens")) c.maxSummaryTokens = j["maxSummaryTokens"].get<int>();
    if (j.contains("useTinyLlamaFallback")) c.useTinyLlamaFallback = j["useTinyLlamaFallback"].get<bool>();
    if (j.contains("tinyLlamaUrl")) c.tinyLlamaUrl = j["tinyLlamaUrl"].get<std::string>();
    if (j.contains("tinyLlamaTimeoutMs")) c.tinyLlamaTimeoutMs = j["tinyLlamaTimeoutMs"].get<int>();
    return c;
}

nlohmann::json MemoryConfig::toJson() const {
    std::string ct = "Transformer";
    if (cellType == CellType::RNN) ct = "RNN";
    else if (cellType == CellType::LSTM) ct = "LSTM";
    return {
        {"cellType", ct},
        {"inputDim", inputDim},
        {"hiddenDim", hiddenDim},
        {"outputDim", outputDim},
        {"maxSummaryTokens", maxSummaryTokens},
        {"useTinyLlamaFallback", useTinyLlamaFallback},
        {"tinyLlamaUrl", tinyLlamaUrl},
        {"tinyLlamaTimeoutMs", tinyLlamaTimeoutMs}
    };
}

namespace {

static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

class RNNCell {
public:
    int inputDim, hiddenDim;
    std::vector<float> Wx, Wh, b;

    RNNCell(int inD, int hidD)
        : inputDim(inD), hiddenDim(hidD),
          Wx(static_cast<size_t>(hidD * inD), 0.0f),
          Wh(static_cast<size_t>(hidD * hidD), 0.0f),
          b(static_cast<size_t>(hidD), 0.0f) {
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> dist(-0.08f, 0.08f);
        for (auto &v : Wx) v = dist(rng);
        for (auto &v : Wh) v = dist(rng);
        for (auto &v : b) v = dist(rng);
    }

    std::vector<float> step(const std::vector<float> &x, const std::vector<float> &h) const {
        std::vector<float> out(static_cast<size_t>(hiddenDim), 0.0f);
        for (int r = 0; r < hiddenDim; ++r) {
            float sum = b[static_cast<size_t>(r)];
            size_t rowX = static_cast<size_t>(r) * inputDim;
            for (int c = 0; c < inputDim; ++c) sum += Wx[rowX + c] * x[c];
            size_t rowH = static_cast<size_t>(r) * hiddenDim;
            for (int c = 0; c < hiddenDim; ++c) sum += Wh[rowH + c] * h[c];
            out[static_cast<size_t>(r)] = std::tanh(sum);
        }
        return out;
    }
};

class LSTMCell {
public:
    int inputDim, hiddenDim;
    std::vector<float> Wf, Wi, Wo, Wg, Uf, Ui, Uo, Ug, bf, bi, bo, bg;

    LSTMCell(int inD, int hidD)
        : inputDim(inD), hiddenDim(hidD) {
        auto initM = [&](size_t rows, size_t cols) {
            std::vector<float> m(rows * cols, 0.0f);
            std::mt19937 rng(7);
            std::uniform_real_distribution<float> dist(-0.08f, 0.08f);
            for (auto &v : m) v = dist(rng);
            return m;
        };
        Wf = initM(hiddenDim, inputDim);
        Wi = initM(hiddenDim, inputDim);
        Wo = initM(hiddenDim, inputDim);
        Wg = initM(hiddenDim, inputDim);
        Uf = initM(hiddenDim, hiddenDim);
        Ui = initM(hiddenDim, hiddenDim);
        Uo = initM(hiddenDim, hiddenDim);
        Ug = initM(hiddenDim, hiddenDim);
        bf = initM(hiddenDim, 1);
        bi = initM(hiddenDim, 1);
        bo = initM(hiddenDim, 1);
        bg = initM(hiddenDim, 1);
    }

    std::pair<std::vector<float>, std::vector<float>>
    step(const std::vector<float> &x,
         const std::vector<float> &h,
         const std::vector<float> &c) const {
        std::vector<float> f(hiddenDim, 0.0f), i(hiddenDim, 0.0f),
                           o(hiddenDim, 0.0f), g(hiddenDim, 0.0f),
                           cNext(hiddenDim, 0.0f), hNext(hiddenDim, 0.0f);
        for (int r = 0; r < hiddenDim; ++r) {
            float vf = bf[r], vi = bi[r], vo = bo[r], vg = bg[r];
            for (int c2 = 0; c2 < inputDim; ++c2) {
                vf += Wf[r * inputDim + c2] * x[c2];
                vi += Wi[r * inputDim + c2] * x[c2];
                vo += Wo[r * inputDim + c2] * x[c2];
                vg += Wg[r * inputDim + c2] * x[c2];
            }
            for (int c2 = 0; c2 < hiddenDim; ++c2) {
                vf += Uf[r * hiddenDim + c2] * h[c2];
                vi += Ui[r * hiddenDim + c2] * h[c2];
                vo += Uo[r * hiddenDim + c2] * h[c2];
                vg += Ug[r * hiddenDim + c2] * h[c2];
            }
            f[r] = sigmoid(vf);
            i[r] = sigmoid(vi);
            o[r] = sigmoid(vo);
            g[r] = std::tanh(vg);
            cNext[r] = f[r] * c[r] + i[r] * g[r];
            hNext[r] = o[r] * std::tanh(cNext[r]);
        }
        return {hNext, cNext};
    }
};

} // namespace

struct MemoryModule::Impl {
    MemoryConfig cfg;
    std::unique_ptr<RNNCell> rnn;
    std::unique_ptr<LSTMCell> lstm;
    std::vector<float> zeroState;
    std::unordered_map<std::string, std::vector<phoenix::multimodal::SemanticUnit>> sessionContext;
    mutable std::mutex mutex;

    explicit Impl(const MemoryConfig &c) : cfg(c) {
        if (cfg.cellType == MemoryConfig::CellType::RNN) {
            rnn = std::make_unique<RNNCell>(cfg.inputDim, cfg.hiddenDim);
        } else if (cfg.cellType == MemoryConfig::CellType::LSTM) {
            lstm = std::make_unique<LSTMCell>(cfg.inputDim, cfg.hiddenDim);
        }
        zeroState.assign(static_cast<size_t>(cfg.hiddenDim), 0.0f);
    }

    std::vector<float> encodeVector(const phoenix::multimodal::SemanticUnit &u) const {
        auto v = u.semanticVector;
        if (v.empty()) {
            // textual fallback: deterministic hash of content
            v.resize(16, 0.0f);
            uint64_t h = 0;
            for (char ch : u.content) h = h * 1315423911u + static_cast<unsigned char>(ch);
            std::mt19937_64 rng(h);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (auto &x : v) x = dist(rng);
        }
        if (static_cast<int>(v.size()) != cfg.inputDim) {
            v = phoenix::multimodal::projectToDimension(v, cfg.inputDim, 0x4d656d6fU);
        }
        return v;
    }

    std::vector<float> runSequence(const std::vector<phoenix::multimodal::SemanticUnit> &context) const {
        if (cfg.cellType == MemoryConfig::CellType::RNN && rnn) {
            std::vector<float> h = zeroState;
            for (const auto &u : context) {
                auto x = encodeVector(u);
                h = rnn->step(x, h);
            }
            return h;
        }
        if (cfg.cellType == MemoryConfig::CellType::LSTM && lstm) {
            std::vector<float> h = zeroState, c = zeroState;
            for (const auto &u : context) {
                auto x = encodeVector(u);
                auto [hNext, cNext] = lstm->step(x, h, c);
                h = hNext;
                c = cNext;
            }
            return h;
        }
        // Transformer / fallback: attention-weighted fusion
        if (!context.empty()) {
            phoenix::multimodal::SemanticUnit query;
            query.semanticVector = context.back().semanticVector;
            auto fused = phoenix::multimodal::fuseAttention(query, context, cfg.outputDim);
            return fused.semanticVector;
        }
        return zeroState;
    }

    std::string compressText(const std::string &s) const {
        if (s.empty()) return "(empty)";
        if (s.size() <= static_cast<size_t>(cfg.maxSummaryTokens)) return s;
        return s.substr(0, cfg.maxSummaryTokens - 3) + "...";
    }

    std::string buildSummaryText(const std::vector<phoenix::multimodal::SemanticUnit> &context,
                                 const std::string &userPrompt) const {
        if (!userPrompt.empty()) return compressText(userPrompt);
        std::ostringstream oss;
        for (const auto &u : context) {
            if (!u.content.empty()) {
                if (oss.tellp() > 0) oss << "; ";
                oss << "[" << phoenix::multimodal::modalityToString(u.modality) << "] "
                    << u.content;
            }
        }
        std::string s = oss.str();
        if (s.empty()) s = "(no text context)";
        return compressText(s);
    }
};

MemoryModule::MemoryModule(const MemoryConfig &cfg)
    : impl_(std::make_unique<Impl>(cfg)) {}

MemoryModule::~MemoryModule() = default;

bool MemoryModule::initialize() {
    return true;
}

MemorySummary MemoryModule::summarize(const std::string &sessionId,
                                      const std::vector<phoenix::multimodal::SemanticUnit> &context,
                                      const std::string &userPrompt) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    MemorySummary ms;
    if (context.empty()) {
        ms.empty = true;
        return ms;
    }

    auto &history = impl_->sessionContext[sessionId];
    history.insert(history.end(), context.begin(), context.end());
    if (history.size() > 64) {
        history.erase(history.begin(), history.begin() + (static_cast<int>(history.size()) - 64));
    }

    auto hidden = impl_->runSequence(history);
    if (hidden.size() != static_cast<size_t>(impl_->cfg.outputDim)) {
        hidden = phoenix::multimodal::projectToDimension(hidden, impl_->cfg.outputDim, 0x4d656d6fU);
    }

    std::string summaryText = impl_->buildSummaryText(history, userPrompt);

    ms.forBackend.modality = phoenix::multimodal::Modality::Text;
    ms.forBackend.content = summaryText;
    ms.forBackend.semanticVector = hidden;
    ms.forBackend.confidence = 1.0f;
    ms.forBackend.id = phoenix::multimodal::generateSemanticId(summaryText, 0x4d656d6fU);

    ms.forGnn = ms.forBackend;
    ms.forGnn.metadata["branch"] = "gnn";

    ms.forUser = ms.forBackend;
    ms.forUser.metadata["branch"] = "user";

    return ms;
}

SentenceMemoryFork MemoryModule::forkSentence(const std::string &sentence) const {
    SentenceMemoryFork fork;
    std::vector<float> v(16, 0.0f);
    uint64_t h = 0;
    for (char ch : sentence) h = h * 1315423911u + static_cast<unsigned char>(ch);
    std::mt19937_64 rng(h);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : v) x = dist(rng);
    v = phoenix::multimodal::projectToDimension(v, impl_->cfg.outputDim, 0x4d656d6fU);

    fork.text = sentence;
    fork.forBackend.modality = phoenix::multimodal::Modality::Text;
    fork.forBackend.content = sentence;
    fork.forBackend.semanticVector = v;
    fork.forBackend.id = phoenix::multimodal::generateSemanticId(sentence, 0x4d656d6fU);
    fork.forBackend.metadata["branch"] = "backend";

    fork.forMemeBarrier = fork.forBackend;
    fork.forMemeBarrier.metadata["branch"] = "memebarrier";
    return fork;
}

bool MemoryModule::saveSession(const std::string &sessionId, const std::string &path) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto it = impl_->sessionContext.find(sessionId);
    if (it == impl_->sessionContext.end()) return false;
    nlohmann::json j = nlohmann::json::array();
    for (const auto &u : it->second) j.push_back(u.toJson());
    try {
        std::ofstream ofs(path);
        ofs << j.dump(2);
        return true;
    } catch (...) {
        return false;
    }
}

bool MemoryModule::loadSession(const std::string &sessionId, const std::string &path) {
    try {
        std::ifstream ifs(path);
        nlohmann::json j;
        ifs >> j;
        std::vector<phoenix::multimodal::SemanticUnit> units;
        for (const auto &item : j) units.push_back(phoenix::multimodal::SemanticUnit::fromJson(item));
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->sessionContext[sessionId] = std::move(units);
        return true;
    } catch (...) {
        return false;
    }
}

nlohmann::json MemoryModule::status() const {
    nlohmann::json j;
    j["config"] = impl_->cfg.toJson();
    j["sessionCount"] = impl_->sessionContext.size();
    return j;
}

void MemoryModule::resetSession(const std::string &sessionId) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->sessionContext.erase(sessionId);
}

} // namespace v7
} // namespace phoenix
