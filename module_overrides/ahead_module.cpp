/* ahead_module.cpp - Pre-GNN "ahead" processing module for Phoenix v7.0 "Arthur"
   Copyright (C) 2026 079 Project */

#include "ahead_module.hpp"
#include "async_task_system.hpp"

#include <algorithm>
#include <sstream>

namespace phoenix {
namespace v7 {

AheadConfig AheadConfig::fromJson(const nlohmann::json &j) {
    AheadConfig c;
    if (j.contains("conceptDim")) c.conceptDim = j["conceptDim"].get<size_t>();
    if (j.contains("memoryEnabled")) c.memoryEnabled = j["memoryEnabled"].get<bool>();
    if (j.contains("emotionEnabled")) c.emotionEnabled = j["emotionEnabled"].get<bool>();
    if (j.contains("audioVideoEncDecEnabled")) c.audioVideoEncDecEnabled = j["audioVideoEncDecEnabled"].get<bool>();
    if (j.contains("schedulerName")) c.schedulerName = j["schedulerName"].get<std::string>();
    return c;
}

nlohmann::json AheadConfig::toJson() const {
    return {
        {"conceptDim", conceptDim},
        {"memoryEnabled", memoryEnabled},
        {"emotionEnabled", emotionEnabled},
        {"audioVideoEncDecEnabled", audioVideoEncDecEnabled},
        {"schedulerName", schedulerName}
    };
}

AheadModule::AheadModule(const AheadConfig &cfg)
    : cfg_(cfg),
      memory_(std::make_unique<MemoryModule>()),
      emotion_(std::make_unique<phoenix::emotion::EmotionSystem>(phoenix::emotion::EmotionSystem::Config{})),
      emotionInfluence_(std::make_unique<phoenix::emotion_influence::EmotionInfluence>()) {}

AheadModule::~AheadModule() = default;

bool AheadModule::initialize() {
    bool ok = true;
    if (cfg_.memoryEnabled) ok = memory_->initialize() && ok;
    return ok;
}

std::vector<phoenix::multimodal::SemanticUnit>
AheadModule::encodePackets_(const std::vector<phoenix::io::MixedModalPacket> &packets) {
    std::vector<phoenix::multimodal::SemanticUnit> out;
    out.reserve(packets.size());
    for (const auto &p : packets) {
        out.push_back(phoenix::io::MixedModalConceptBridge::encode(p, cfg_.conceptDim, ""));
    }
    return out;
}

phoenix::emotion::EmotionTensor
AheadModule::observeEmotion_(const std::string &sessionId,
                             const std::vector<phoenix::multimodal::SemanticUnit> &units) {
    std::string text;
    for (const auto &u : units) {
        if (u.modality == phoenix::multimodal::Modality::Text) {
            text += u.content + " ";
        }
    }
    if (!text.empty()) {
        text.pop_back();
        return emotion_->processMessage(sessionId, text, 0).current;
    }
    phoenix::emotion::EmotionTensor e;
    return e;
}

AheadResult AheadModule::processBatch(
    const std::vector<phoenix::io::MixedModalPacket> &packets,
    const std::string &sessionId) {
    AheadResult r;

    r.units = encodePackets_(packets);

    if (cfg_.memoryEnabled && memory_) {
        std::string userPrompt;
        for (const auto &u : r.units) {
            if (u.modality == phoenix::multimodal::Modality::Text && !u.content.empty()) {
                userPrompt = u.content;
                break;
            }
        }
        auto summary = memory_->summarize(sessionId, r.units, userPrompt);
        r.memoryForBackend = summary.forBackend;
        r.memoryForGnn = summary.forGnn;
        r.memoryReady = !summary.empty;
    }

    if (cfg_.emotionEnabled && emotion_ && emotionInfluence_) {
        r.emotion = observeEmotion_(sessionId, r.units);
        r.emotionInfluence = emotionInfluence_->compute(r.emotion, "", 0);
    }

    return r;
}

AheadModule::SentenceResult AheadModule::routeSentence(const std::string &sentence,
                                                       const std::string &sessionId) {
    AheadModule::SentenceResult sr;

    if (memory_) {
        auto fork = memory_->forkSentence(sentence);
        sr.textForBackend = fork.forBackend.content.empty() ? sentence : fork.forBackend.content;
        sr.textForMemeBarrier = fork.forMemeBarrier.content.empty() ? sentence : fork.forMemeBarrier.content;
        sr.unitForBackend = fork.forBackend;
        sr.unitForMemeBarrier = fork.forMemeBarrier;
    } else {
        sr.textForBackend = sentence;
        sr.textForMemeBarrier = sentence;
        phoenix::multimodal::SemanticUnit u;
        u.content = sentence;
        u.modality = phoenix::multimodal::Modality::Text;
        sr.unitForBackend = u;
        sr.unitForMemeBarrier = u;
    }

    return sr;
}

nlohmann::json AheadModule::status() const {
    nlohmann::json j;
    j["config"] = cfg_.toJson();
    j["memory"] = memory_ ? memory_->status() : nlohmann::json{};
    return j;
}

void AheadModule::resetSession(const std::string &sessionId) {
    if (memory_) memory_->resetSession(sessionId);
    if (emotionInfluence_) emotionInfluence_->resetSession(sessionId);
}

} // namespace v7
} // namespace phoenix
