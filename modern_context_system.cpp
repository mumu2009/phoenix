/* modern_context_system.cpp - Modern context system implementation
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

#include "modern_context_system.hpp"
#include "phoenix_config.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <regex>
#include <iostream>
#include <set>

namespace phoenix {
namespace context {

/* Modern context manager implementation */
struct ModernContextManager::Impl {
    ContextWindowConfig config;
    std::deque<ContextEntry> contextEntries;
    std::unique_ptr<AttentionSinkManager> sinkManager;
    std::unique_ptr<HierarchicalContextManager> hierarchicalManager;
    std::unique_ptr<SemanticChunker> chunker;
    std::unique_ptr<ContextCompressor> compressor;
    std::unique_ptr<ContextCache> cache;
    std::mutex mutex;
    
    Impl(const ContextWindowConfig& cfg) : config(cfg) {
        if (config.enableAttentionSink) {
            sinkManager = std::make_unique<AttentionSinkManager>(
                AttentionSinkManager::SinkConfig{}
            );
        }
        if (config.enableHierarchical) {
            hierarchicalManager = std::make_unique<HierarchicalContextManager>();
        }
        chunker = std::make_unique<SemanticChunker>(SemanticChunker::ChunkConfig{});
        compressor = std::make_unique<ContextCompressor>(ContextCompressor::CompressionConfig{});
        cache = std::make_unique<ContextCache>(100);
    }
};

ModernContextManager::ModernContextManager(const ContextWindowConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

ModernContextManager::~ModernContextManager() = default;

bool ModernContextManager::addEntry(const ContextEntry& entry) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    
    // Estimate tokens if not provided
    ContextEntry entryWithTokens = entry;
    if (entryWithTokens.estimatedTokens == 0) {
        entryWithTokens.estimatedTokens = TokenCounter::estimateTokens(entry.content);
    }
    
    impl_->contextEntries.push_back(entryWithTokens);
    
    // Prune if exceeding max tokens (already under impl_->mutex, use locked helper)
    if (impl_->config.maxTokens > 0) {
        size_t totalTokens = TokenCounter::estimateTokens(
            std::vector<ContextEntry>(impl_->contextEntries.begin(), impl_->contextEntries.end())
        );
        
        if (totalTokens > impl_->config.maxTokens) {
            pruneContextLocked();
        }
    }
    
    return true;
}

bool ModernContextManager::addSemanticUnit(const phoenix::multimodal::SemanticUnit& unit, const std::string& role) {
    ContextEntry entry;
    entry.content = unit.content;
    entry.role = role;
    entry.importance = unit.confidence;
    entry.semanticUnits.push_back(unit);
    if (entry.semanticUnits.front().timestampMs == 0) {
        entry.semanticUnits.front().timestampMs = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
    }
    return addEntry(entry);
}

std::vector<ContextEntry> ModernContextManager::getContext(size_t maxTokens) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    
    size_t targetTokens = maxTokens > 0 ? maxTokens : impl_->config.maxTokens;
    
    // Apply strategy-specific processing
    std::vector<ContextEntry> context(
        impl_->contextEntries.begin(), impl_->contextEntries.end()
    );
    
    switch (impl_->config.strategy) {
        case ContextStrategy::SLIDING_WINDOW:
            // Standard sliding window - already handled by pruneContext
            break;
            
        case ContextStrategy::ATTENTION_SINK:
            if (impl_->sinkManager) {
                context = impl_->sinkManager->optimizeWithContextSink(context);
            }
            break;
            
        case ContextStrategy::HIERARCHICAL:
            if (impl_->hierarchicalManager) {
                // Rebuild hierarchical context from flat entries
                for (const auto& entry : impl_->contextEntries) {
                    impl_->hierarchicalManager->addContext("default", entry);
                }
                context = impl_->hierarchicalManager->getFlattenedContext(targetTokens);
            }
            break;
            
        case ContextStrategy::SEMANTIC_CHUNKING:
            // Semantic chunking is applied during addEntry
            break;
            
        case ContextStrategy::HYBRID:
            // Combine multiple strategies
            if (impl_->sinkManager) {
                context = impl_->sinkManager->optimizeWithContextSink(context);
            }
            if (impl_->compressor) {
                context = impl_->compressor->compressContext(context, targetTokens);
            }
            break;
    }
    
    // Ensure token limit
    if (targetTokens > 0) {
        size_t currentTokens = TokenCounter::estimateTokens(context);
        if (currentTokens > targetTokens) {
            // Trim from beginning
            while (currentTokens > targetTokens && !context.empty()) {
                currentTokens -= context.front().estimatedTokens;
                context.erase(context.begin());
            }
        }
    }
    
    return context;
}

std::vector<SemanticSearchResult> ModernContextManager::semanticSearch(
    const std::string& query, size_t topK) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    
    if (!impl_->config.enableSemanticSearch) {
        return {};
    }
    
    std::vector<SemanticSearchResult> results;
    
    // Simple similarity search (in real implementation, would use embeddings)
    for (size_t i = 0; i < impl_->contextEntries.size(); ++i) {
        const auto& entry = impl_->contextEntries[i];
        
        // Calculate simple text similarity
        float similarity = calculateTextSimilarity(query, entry.content);
        
        if (similarity >= impl_->config.similarityThreshold) {
            SemanticSearchResult result;
            result.entry = entry;
            result.similarity = similarity;
            result.position = i;
            results.push_back(result);
        }
    }
    
    // Sort by similarity and return top K
    std::sort(results.begin(), results.end(),
        [](const SemanticSearchResult& a, const SemanticSearchResult& b) {
            return a.similarity > b.similarity;
        });
    
    if (results.size() > topK) {
        results.resize(topK);
    }
    
    return results;
}

std::vector<SemanticSearchResult> ModernContextManager::semanticSearch(
    const phoenix::multimodal::SemanticUnit& query, size_t topK) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    
    if (!impl_->config.enableSemanticSearch || !impl_->config.enableMultimodal) {
        return {};
    }
    
    std::vector<SemanticSearchResult> results;
    const auto& qvec = query.semanticVector;
    
    for (size_t i = 0; i < impl_->contextEntries.size(); ++i) {
        const auto& entry = impl_->contextEntries[i];
        if (entry.semanticUnits.empty() && entry.embedding.empty()) {
            continue;
        }
        
        float bestSim = 0.0f;
        for (const auto& unit : entry.semanticUnits) {
            float sim = phoenix::multimodal::cosineSimilarity(qvec, unit.semanticVector);
            if (sim > bestSim) bestSim = sim;
        }
        if (!entry.embedding.empty()) {
            float sim = phoenix::multimodal::cosineSimilarity(qvec, entry.embedding);
            if (sim > bestSim) bestSim = sim;
        }
        
        if (bestSim >= impl_->config.similarityThreshold) {
            SemanticSearchResult result;
            result.entry = entry;
            result.similarity = bestSim;
            result.position = i;
            results.push_back(result);
        }
    }
    
    std::sort(results.begin(), results.end(),
        [](const SemanticSearchResult& a, const SemanticSearchResult& b) {
            return a.similarity > b.similarity;
        });
    
    if (results.size() > topK) {
        results.resize(topK);
    }
    
    return results;
}

bool ModernContextManager::updateImportance(const std::string& contentId, float newImportance) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    
    for (auto& entry : impl_->contextEntries) {
        if (entry.metadata.count("id") && entry.metadata["id"] == contentId) {
            entry.importance = newImportance;
            return true;
        }
    }
    
    return false;
}

void ModernContextManager::pruneContextLocked() {
    if (impl_->contextEntries.empty()) {
        return;
    }

    // Sanitize NaN / invalid importance values to avoid std::sort UB.
    for (auto& entry : impl_->contextEntries) {
        if (std::isnan(entry.importance) || std::isinf(entry.importance)) {
            entry.importance = 0.5f;
        }
    }
    
    // Sort by importance (lower importance first)
    std::vector<size_t> indices;
    for (size_t i = 0; i < impl_->contextEntries.size(); ++i) {
        indices.push_back(i);
    }
    
    std::sort(indices.begin(), indices.end(),
        [this](size_t a, size_t b) {
            return impl_->contextEntries[a].importance < impl_->contextEntries[b].importance;
        });
    
    // Remove low importance entries until under token limit
    size_t totalTokens = TokenCounter::estimateTokens(
        std::vector<ContextEntry>(impl_->contextEntries.begin(), impl_->contextEntries.end())
    );
    
    while (totalTokens > impl_->config.maxTokens && !indices.empty()) {
        size_t idxToRemove = indices.back();
        indices.pop_back();
        
        totalTokens -= impl_->contextEntries[idxToRemove].estimatedTokens;
        impl_->contextEntries.erase(impl_->contextEntries.begin() + idxToRemove);
    }
}

bool ModernContextManager::pruneContext() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    pruneContextLocked();
    return true;
}

nlohmann::json ModernContextManager::getStatistics() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    
    size_t totalTokens = TokenCounter::estimateTokens(
        std::vector<ContextEntry>(impl_->contextEntries.begin(), impl_->contextEntries.end())
    );
    
    float avgImportance = 0.0f;
    if (!impl_->contextEntries.empty()) {
        for (const auto& entry : impl_->contextEntries) {
            avgImportance += entry.importance;
        }
        avgImportance /= impl_->contextEntries.size();
    }
    
    size_t totalSemanticUnits = 0;
    for (const auto& entry : impl_->contextEntries) {
        totalSemanticUnits += entry.semanticUnits.size();
    }
    
    return {
        {"totalEntries", impl_->contextEntries.size()},
        {"totalTokens", totalTokens},
        {"totalSemanticUnits", totalSemanticUnits},
        {"maxTokens", impl_->config.maxTokens},
        {"strategy", static_cast<int>(impl_->config.strategy)},
        {"averageImportance", avgImportance},
        {"semanticSearchEnabled", impl_->config.enableSemanticSearch},
        {"attentionSinkEnabled", impl_->config.enableAttentionSink},
        {"hierarchicalEnabled", impl_->config.enableHierarchical},
        {"multimodalEnabled", impl_->config.enableMultimodal},
        {"multimodalEmbeddingDim", impl_->config.multimodalEmbeddingDim}
    };
}

void ModernContextManager::clear() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->contextEntries.clear();
}

std::string ModernContextManager::exportContext() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    
    nlohmann::json exportData;
    exportData["config"] = {
        {"maxTokens", impl_->config.maxTokens},
        {"strategy", static_cast<int>(impl_->config.strategy)},
        {"semanticSearchEnabled", impl_->config.enableSemanticSearch},
        {"multimodalEnabled", impl_->config.enableMultimodal},
        {"multimodalEmbeddingDim", impl_->config.multimodalEmbeddingDim}
    };
    
    nlohmann::json entriesJson = nlohmann::json::array();
    for (const auto& entry : impl_->contextEntries) {
        entriesJson.push_back(entry.toJson());
    }
    exportData["entries"] = entriesJson;
    
    return exportData.dump(2);
}

bool ModernContextManager::importContext(const std::string& jsonStr) {
    try {
        nlohmann::json importData = nlohmann::json::parse(jsonStr);
        
        std::lock_guard<std::mutex> lock(impl_->mutex);
        
        impl_->contextEntries.clear();
        
        if (importData.contains("entries")) {
            for (const auto& entryJson : importData["entries"]) {
                impl_->contextEntries.push_back(ContextEntry::fromJson(entryJson));
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[ModernContextManager] Import failed: " << e.what() << std::endl;
        return false;
    }
}

void ModernContextManager::setConfig(const ContextWindowConfig& config) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->config = config;
}

ContextWindowConfig ModernContextManager::getConfig() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->config;
}

float ModernContextManager::calculateTextSimilarity(const std::string& text1, const std::string& text2) {
    // Simple Jaccard similarity for words
    auto getWords = [](const std::string& text) {
        std::vector<std::string> words;
        std::istringstream iss(text);
        std::string word;
        while (iss >> word) {
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);
            words.push_back(word);
        }
        return words;
    };
    
    auto words1 = getWords(text1);
    auto words2 = getWords(text2);
    
    if (words1.empty() || words2.empty()) {
        return 0.0f;
    }
    
    std::set<std::string> set1(words1.begin(), words1.end());
    std::set<std::string> set2(words2.begin(), words2.end());
    
    std::set<std::string> intersection;
    std::set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(),
                        std::inserter(intersection, intersection.begin()));
    
    std::set<std::string> unionSet;
    std::set_union(set1.begin(), set1.end(), set2.begin(), set2.end(),
                  std::inserter(unionSet, unionSet.begin()));
    
    if (unionSet.empty()) {
        return 0.0f;
    }
    
    return static_cast<float>(intersection.size()) / static_cast<float>(unionSet.size());
}

// Attention sink manager implementation
AttentionSinkManager::AttentionSinkManager(const SinkConfig& config)
    : config_(config) {}

std::string AttentionSinkManager::generateSink(const std::vector<ContextEntry>& context) {
    if (!config_.dynamicSink) {
        return config_.sinkContent;
    }
    
    // Generate dynamic sink based on context
    std::ostringstream sink;
    sink << "[Context Summary: " << context.size() << " messages]";
    
    if (config_.sinkTokens > 50) {
        sink << " Previous conversation covers various topics including ";
        // In real implementation, would extract key themes
        sink << "general discussion and information exchange.";
    }
    
    return sink.str();
}

std::vector<ContextEntry> AttentionSinkManager::optimizeWithContextSink(
    const std::vector<ContextEntry>& context) {
    if (context.empty()) {
        return context;
    }
    
    std::vector<ContextEntry> optimized;
    
    // Add attention sink at the beginning
    ContextEntry sinkEntry;
    sinkEntry.content = generateSink(context);
    sinkEntry.role = "system";
    sinkEntry.importance = config_.sinkImportance;
    sinkEntry.estimatedTokens = config_.sinkTokens;
    optimized.push_back(sinkEntry);
    
    // Add original context
    optimized.insert(optimized.end(), context.begin(), context.end());
    
    return optimized;
}

void AttentionSinkManager::setConfig(const SinkConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
}

// Hierarchical context manager implementation
HierarchicalContextManager::HierarchicalContextManager() {
    // Load level capacities from the single authoritative config.
    // Defaults allow the manager to work when config/phoenix.json is not loaded (unit tests).
    updateLevel("global", 1, phoenix::cfgOr<size_t>("context.hierarchicalLevels.global", 1024));
    updateLevel("workspace", 2, phoenix::cfgOr<size_t>("context.hierarchicalLevels.workspace", 512));
    updateLevel("file", 3, phoenix::cfgOr<size_t>("context.hierarchicalLevels.file", 256));
    updateLevel("function", 4, phoenix::cfgOr<size_t>("context.hierarchicalLevels.function", 128));
}

bool HierarchicalContextManager::addContext(const std::string& level, const ContextEntry& entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = levels_.find(level);
    if (it == levels_.end()) {
        return false;
    }
    
    it->second.entries.push_back(entry);
    return true;
}

std::vector<ContextEntry> HierarchicalContextManager::getFlattenedContext(size_t maxTokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Sort levels by priority (higher priority first)
    std::vector<std::pair<int, std::string>> sortedLevels;
    for (const auto& pair : levels_) {
        sortedLevels.push_back({pair.second.priority, pair.first});
    }
    std::sort(sortedLevels.begin(), sortedLevels.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    
    std::vector<ContextEntry> flattened;
    size_t currentTokens = 0;
    
    // Add entries from highest priority levels first
    for (const auto& pair : sortedLevels) {
        const auto& level = levels_[pair.second];
        
        for (const auto& entry : level.entries) {
            if (maxTokens > 0 && currentTokens + entry.estimatedTokens > maxTokens) {
                break;
            }
            
            flattened.push_back(entry);
            currentTokens += entry.estimatedTokens;
        }
        
        if (maxTokens > 0 && currentTokens >= maxTokens) {
            break;
        }
    }
    
    return flattened;
}

bool HierarchicalContextManager::updateLevel(const std::string& level, int priority, size_t maxTokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    ContextLevel& ctxLevel = levels_[level];
    ctxLevel.name = level;
    ctxLevel.priority = priority;
    ctxLevel.maxTokens = maxTokens;
    
    return true;
}

bool HierarchicalContextManager::removeLevel(const std::string& level) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = levels_.find(level);
    if (it != levels_.end()) {
        levels_.erase(it);
        return true;
    }
    
    return false;
}

std::map<std::string, nlohmann::json> HierarchicalContextManager::getLevelStatistics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::map<std::string, nlohmann::json> stats;
    for (const auto& pair : levels_) {
        const auto& level = pair.second;
        
        size_t totalTokens = 0;
        for (const auto& entry : level.entries) {
            totalTokens += entry.estimatedTokens;
        }
        
        stats[pair.first] = {
            {"priority", level.priority},
            {"maxTokens", level.maxTokens},
            {"entryCount", level.entries.size()},
            {"totalTokens", totalTokens}
        };
    }
    
    return stats;
}

// Semantic chunker implementation
SemanticChunker::SemanticChunker(const ChunkConfig& config)
    : config_(config) {}

std::vector<ContextEntry> SemanticChunker::chunkText(const std::string& text, const std::string& role) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<ContextEntry> chunks;
    
    if (config_.respectParagraphBoundaries) {
        auto paragraphs = splitByParagraphs(text);
        
        for (const auto& paragraph : paragraphs) {
            if (paragraph.empty()) continue;
            
            float importance = calculateChunkImportance(paragraph);
            if (importance < config_.minChunkImportance) {
                continue;
            }
            
            ContextEntry entry;
            entry.content = paragraph;
            entry.role = role;
            entry.importance = importance;
            entry.estimatedTokens = TokenCounter::estimateTokens(paragraph);
            
            chunks.push_back(entry);
        }
    } else if (config_.respectSentenceBoundaries) {
        auto sentences = splitBySentences(text);
        
        std::string currentChunk;
        size_t currentTokens = 0;
        
        for (const auto& sentence : sentences) {
            size_t sentenceTokens = TokenCounter::estimateTokens(sentence);
            
            if (currentTokens + sentenceTokens > config_.chunkSize && !currentChunk.empty()) {
                ContextEntry entry;
                entry.content = currentChunk;
                entry.role = role;
                entry.importance = calculateChunkImportance(currentChunk);
                entry.estimatedTokens = currentTokens;
                chunks.push_back(entry);
                
                currentChunk.clear();
                currentTokens = 0;
            }
            
            currentChunk += sentence + " ";
            currentTokens += sentenceTokens;
        }
        
        if (!currentChunk.empty()) {
            ContextEntry entry;
            entry.content = currentChunk;
            entry.role = role;
            entry.importance = calculateChunkImportance(currentChunk);
            entry.estimatedTokens = currentTokens;
            chunks.push_back(entry);
        }
    } else {
        // Simple fixed-size chunking
        for (size_t i = 0; i < text.length(); i += config_.chunkSize) {
            std::string chunk = text.substr(i, config_.chunkSize);
            
            ContextEntry entry;
            entry.content = chunk;
            entry.role = role;
            entry.importance = calculateChunkImportance(chunk);
            entry.estimatedTokens = TokenCounter::estimateTokens(chunk);
            chunks.push_back(entry);
        }
    }
    
    return chunks;
}

std::string SemanticChunker::mergeChunks(const std::vector<ContextEntry>& chunks) {
    std::ostringstream merged;
    for (const auto& chunk : chunks) {
        merged << chunk.content << " ";
    }
    return merged.str();
}

nlohmann::json SemanticChunker::getChunkStatistics(const std::vector<ContextEntry>& chunks) const {
    size_t totalTokens = 0;
    float avgImportance = 0.0f;
    
    for (const auto& chunk : chunks) {
        totalTokens += chunk.estimatedTokens;
        avgImportance += chunk.importance;
    }
    
    if (!chunks.empty()) {
        avgImportance /= chunks.size();
    }
    
    return {
        {"chunkCount", chunks.size()},
        {"totalTokens", totalTokens},
        {"averageImportance", avgImportance},
        {"averageChunkSize", chunks.empty() ? 0.0 : static_cast<double>(totalTokens) / chunks.size()}
    };
}

std::vector<std::string> SemanticChunker::splitBySentences(const std::string& text) {
    std::vector<std::string> sentences;
    std::regex sentenceRegex(R"([.!?]+[\s\n]+)");
    std::sregex_token_iterator it(text.begin(), text.end(), sentenceRegex, -1);
    std::sregex_token_iterator end;
    
    while (it != end) {
        std::string sentence = *it;
        if (!sentence.empty()) {
            sentences.push_back(sentence);
        }
        ++it;
    }
    
    return sentences;
}

std::vector<std::string> SemanticChunker::splitByParagraphs(const std::string& text) {
    std::vector<std::string> paragraphs;
    std::istringstream iss(text);
    std::string paragraph;
    
    while (std::getline(iss, paragraph)) {
        if (!paragraph.empty()) {
            paragraphs.push_back(paragraph);
        }
    }
    
    return paragraphs;
}

float SemanticChunker::calculateChunkImportance(const std::string& chunk) {
    // Simple heuristic: longer chunks with more unique words are more important
    auto words = splitBySentences(chunk);
    std::set<std::string> uniqueWords;
    
    for (const auto& word : words) {
        std::string lowerWord = word;
        std::transform(lowerWord.begin(), lowerWord.end(), lowerWord.begin(), ::tolower);
        uniqueWords.insert(lowerWord);
    }
    
    float lengthScore = std::min(1.0f, static_cast<float>(chunk.length()) / 500.0f);
    float diversityScore = std::min(1.0f, static_cast<float>(uniqueWords.size()) / 50.0f);
    
    return (lengthScore + diversityScore) / 2.0f;
}

// Context compressor implementation
ContextCompressor::ContextCompressor(const CompressionConfig& config)
    : config_(config) {}

std::vector<ContextEntry> ContextCompressor::compressContext(
    const std::vector<ContextEntry>& context, size_t targetTokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (context.empty()) {
        return context;
    }
    
    size_t currentTokens = TokenCounter::estimateTokens(context);
    if (currentTokens <= targetTokens) {
        return context;
    }
    
    std::vector<ContextEntry> compressed;
    
    if (config_.useSemanticPruning) {
        compressed = semanticPrune(context, targetTokens);
    } else if (config_.useSummarization) {
        // Simple summarization approach
        for (const auto& entry : context) {
            if (entry.importance >= 0.5f || config_.preserveKeyInformation) {
                compressed.push_back(entry);
            } else {
                // Summarize less important entries
                ContextEntry summarized = entry;
                summarized.content = summarizeText(entry.content, entry.estimatedTokens / 2);
                summarized.estimatedTokens = TokenCounter::estimateTokens(summarized.content);
                compressed.push_back(summarized);
            }
        }
    } else {
        // Simple truncation
        size_t tokens = 0;
        for (const auto& entry : context) {
            if (tokens + entry.estimatedTokens <= targetTokens) {
                compressed.push_back(entry);
                tokens += entry.estimatedTokens;
            }
        }
    }
    
    return compressed;
}

std::vector<ContextEntry> ContextCompressor::decompressContext(
    const std::vector<ContextEntry>& compressed) {
    // In a real implementation, this would attempt to restore original context
    // For now, return as-is
    return compressed;
}

nlohmann::json ContextCompressor::getCompressionStatistics(
    const std::vector<ContextEntry>& original,
    const std::vector<ContextEntry>& compressed) const {
    size_t originalTokens = TokenCounter::estimateTokens(original);
    size_t compressedTokens = TokenCounter::estimateTokens(compressed);
    
    float ratio = originalTokens > 0 ? 
        static_cast<float>(compressedTokens) / static_cast<float>(originalTokens) : 1.0f;
    
    return {
        {"originalTokens", originalTokens},
        {"compressedTokens", compressedTokens},
        {"compressionRatio", ratio},
        {"originalEntries", original.size()},
        {"compressedEntries", compressed.size()}
    };
}

std::string ContextCompressor::summarizeText(const std::string& text, size_t targetLength) {
    // Simple summarization: take first and last sentences
    std::vector<std::string> sentences;
    std::regex sentenceRegex(R"([.!?]+[\s\n]+)");
    std::sregex_token_iterator it(text.begin(), text.end(), sentenceRegex, -1);
    std::sregex_token_iterator end;
    
    while (it != end) {
        std::string sentence = *it;
        if (!sentence.empty()) {
            sentences.push_back(sentence);
        }
        ++it;
    }
    
    if (sentences.empty()) {
        return text.substr(0, targetLength);
    }
    
    std::ostringstream summary;
    if (sentences.size() <= 2) {
        summary << text;
    } else {
        summary << sentences.front() << " ... " << sentences.back();
    }
    
    std::string result = summary.str();
    if (result.length() > targetLength && targetLength > 0) {
        result = result.substr(0, targetLength);
    }
    
    return result;
}

std::vector<ContextEntry> ContextCompressor::semanticPrune(
    const std::vector<ContextEntry>& context, size_t targetTokens) {
    // Sort by importance and keep highest importance entries
    std::vector<ContextEntry> sorted = context;
    std::sort(sorted.begin(), sorted.end(),
        [](const ContextEntry& a, const ContextEntry& b) {
            return a.importance > b.importance;
        });
    
    std::vector<ContextEntry> pruned;
    size_t tokens = 0;
    
    for (const auto& entry : sorted) {
        if (tokens + entry.estimatedTokens <= targetTokens) {
            pruned.push_back(entry);
            tokens += entry.estimatedTokens;
        }
    }
    
    // Restore original order
    std::sort(pruned.begin(), pruned.end(),
        [](const ContextEntry& a, const ContextEntry& b) {
            return a.turnNumber < b.turnNumber;
        });
    
    return pruned;
}

// Context cache implementation
ContextCache::ContextCache(size_t maxEntries) : maxEntries_(maxEntries) {}

std::optional<std::vector<ContextEntry>> ContextCache::get(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = cache_.find(key);
    if (it == cache_.end()) {
        return std::nullopt;
    }
    
    it->second.lastAccess = std::chrono::system_clock::now();
    it->second.accessCount++;
    
    return it->second.context;
}

void ContextCache::put(const std::string& key, const std::vector<ContextEntry>& context) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    CacheEntry entry;
    entry.context = context;
    entry.lastAccess = std::chrono::system_clock::now();
    entry.accessCount = 1;
    entry.cacheKey = key;
    
    cache_[key] = entry;
    
    evictIfNeeded();
}

void ContextCache::invalidate(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.erase(key);
}

void ContextCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
}

nlohmann::json ContextCache::getStatistics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t totalAccessCount = 0;
    for (const auto& pair : cache_) {
        totalAccessCount += pair.second.accessCount;
    }
    
    return {
        {"entryCount", cache_.size()},
        {"maxEntries", maxEntries_},
        {"totalAccessCount", totalAccessCount},
        {"hitRate", cache_.empty() ? 0.0 : static_cast<double>(totalAccessCount) / cache_.size()}
    };
}

void ContextCache::evictIfNeeded() {
    if (cache_.size() <= maxEntries_) {
        return;
    }
    
    // Evict least recently used entry
    auto lruIt = std::min_element(cache_.begin(), cache_.end(),
        [](const auto& a, const auto& b) {
            return a.second.lastAccess < b.second.lastAccess;
        });
    
    if (lruIt != cache_.end()) {
        cache_.erase(lruIt);
    }
}

// Token counter implementation
size_t TokenCounter::estimateTokens(const std::string& text) {
    if (text.empty()) {
        return 0;
    }
    
    float ratio = getCharToTokenRatio(text);
    size_t charCount = countCharacters(text);
    
    return static_cast<size_t>(std::ceil(charCount / ratio));
}

size_t TokenCounter::estimateTokens(const std::vector<ContextEntry>& entries) {
    size_t total = 0;
    for (const auto& entry : entries) {
        total += entry.estimatedTokens > 0 ? entry.estimatedTokens : estimateTokens(entry.content);
    }
    return total;
}

float TokenCounter::getCharToTokenRatio(const std::string& text) {
    if (isCJK(text)) {
        return 2.5f;  // CJK characters: ~2.5 chars per token
    }
    return 4.0f;  // English: ~4 chars per token
}

size_t TokenCounter::countWords(const std::string& text) {
    std::istringstream iss(text);
    return std::distance(std::istream_iterator<std::string>(iss),
                        std::istream_iterator<std::string>());
}

size_t TokenCounter::countCharacters(const std::string& text) {
    return text.length();
}

bool TokenCounter::isCJK(const std::string& text) {
    // Simple CJK detection
    for (char c : text) {
        if (static_cast<unsigned char>(c) > 127) {
            return true;
        }
    }
    return false;
}

// Context builder implementation
ContextBuilder::ContextBuilder(const ContextWindowConfig& config)
    : config_(config), currentTokens_(0) {}

ContextBuilder& ContextBuilder::addSystemPrompt(const std::string& prompt) {
    ContextEntry entry;
    entry.content = prompt;
    entry.role = "system";
    entry.importance = 1.0f;  // System prompts are always important
    entry.estimatedTokens = TokenCounter::estimateTokens(prompt);
    
    entries_.push_back(entry);
    currentTokens_ += entry.estimatedTokens;
    
    return *this;
}

ContextBuilder& ContextBuilder::addUserMessage(const std::string& message, float importance) {
    ContextEntry entry;
    entry.content = message;
    entry.role = "user";
    entry.importance = importance;
    entry.estimatedTokens = TokenCounter::estimateTokens(message);
    
    entries_.push_back(entry);
    currentTokens_ += entry.estimatedTokens;
    
    return *this;
}

ContextBuilder& ContextBuilder::addAssistantMessage(const std::string& message, float importance) {
    ContextEntry entry;
    entry.content = message;
    entry.role = "assistant";
    entry.importance = importance;
    entry.estimatedTokens = TokenCounter::estimateTokens(message);
    
    entries_.push_back(entry);
    currentTokens_ += entry.estimatedTokens;
    
    return *this;
}

ContextBuilder& ContextBuilder::addToolResult(const std::string& toolName, const std::string& result) {
    ContextEntry entry;
    entry.content = result;
    entry.role = "tool";
    entry.importance = 0.8f;
    entry.metadata["toolName"] = toolName;
    entry.estimatedTokens = TokenCounter::estimateTokens(result);
    
    entries_.push_back(entry);
    currentTokens_ += entry.estimatedTokens;
    
    return *this;
}

ContextBuilder& ContextBuilder::addEntry(const ContextEntry& entry) {
    ContextEntry entryWithTokens = entry;
    if (entryWithTokens.estimatedTokens == 0) {
        entryWithTokens.estimatedTokens = TokenCounter::estimateTokens(entry.content);
    }
    
    entries_.push_back(entryWithTokens);
    currentTokens_ += entryWithTokens.estimatedTokens;
    
    return *this;
}

ContextBuilder& ContextBuilder::addSemanticUnit(const phoenix::multimodal::SemanticUnit& unit, const std::string& role) {
    ContextEntry entry;
    entry.content = unit.content;
    entry.role = role;
    entry.importance = unit.confidence;
    entry.semanticUnits.push_back(unit);
    return addEntry(entry);
}

std::vector<ContextEntry> ContextBuilder::build() {
    return build(config_.maxTokens);
}

std::vector<ContextEntry> ContextBuilder::build(size_t maxTokens) {
    if (maxTokens == 0 || currentTokens_ <= maxTokens) {
        return entries_;
    }
    
    // Trim from beginning to respect token limit
    std::vector<ContextEntry> result;
    size_t tokens = 0;
    
    // Keep system prompts and recent entries
    for (auto it = entries_.rbegin(); it != entries_.rend(); ++it) {
        if (it->role == "system" || tokens + it->estimatedTokens <= maxTokens) {
            result.insert(result.begin(), *it);
            tokens += it->estimatedTokens;
        }
    }
    
    return result;
}

void ContextBuilder::reset() {
    entries_.clear();
    currentTokens_ = 0;
}

size_t ContextBuilder::getCurrentTokenCount() const {
    return currentTokens_;
}

// Context hooks implementation
std::function<std::vector<ContextEntry>(const std::vector<ContextEntry>&)> 
    ContextHooks::onPreBuild = nullptr;
std::function<std::vector<ContextEntry>(const std::vector<ContextEntry>&)> 
    ContextHooks::onPostBuild = nullptr;
std::function<void(const std::vector<ContextEntry>&, const std::vector<ContextEntry>&)> 
    ContextHooks::onPrune = nullptr;

void ContextHooks::setPreBuildHook(
    std::function<std::vector<ContextEntry>(const std::vector<ContextEntry>&)> hook) {
    onPreBuild = hook;
}

void ContextHooks::setPostBuildHook(
    std::function<std::vector<ContextEntry>(const std::vector<ContextEntry>&)> hook) {
    onPostBuild = hook;
}

void ContextHooks::setPruneHook(
    std::function<void(const std::vector<ContextEntry>&, const std::vector<ContextEntry>&)> hook) {
    onPrune = hook;
}

} // namespace context
} // namespace phoenix
