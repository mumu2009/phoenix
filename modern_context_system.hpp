/* modern_context_system.hpp - Modern context window management system
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

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <deque>
#include <chrono>
#include <mutex>
#include <functional>
#include <nlohmann/json.hpp>

namespace phoenix {
namespace context {

/* Context window management strategies (inspired by Claude, Copilot, Ollama) */
enum class ContextStrategy {
    SLIDING_WINDOW,      /* Traditional sliding window */
    ATTENTION_SINK,      /* Attention sink for long documents */
    HIERARCHICAL,        /* Hierarchical context organization */
    SEMANTIC_CHUNKING,   /* Semantic-based chunking */
    HYBRID               /* Combination of strategies */
};

/* Context entry with metadata */
struct ContextEntry {
    std::string content;                                    /* Entry content */
    std::string role;                                       /* Role: user, assistant, system, tool */
    std::chrono::system_clock::time_point timestamp;       /* Timestamp */
    int64_t turnNumber;                                     /* Turn number */
    float importance;                                       /* Importance score for retention */
    std::vector<float> embedding;                           /* Semantic embedding */
    std::map<std::string, std::string> metadata;           /* Additional metadata */

    /* Token count estimation */
    size_t estimatedTokens;

    ContextEntry() : turnNumber(0), importance(0.5f), estimatedTokens(0) {
        timestamp = std::chrono::system_clock::now();
    }

    nlohmann::json toJson() const {
        return {
            {"content", content},
            {"role", role},
            {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                timestamp.time_since_epoch()).count()},
            {"turnNumber", turnNumber},
            {"importance", importance},
            {"estimatedTokens", estimatedTokens},
            {"metadata", metadata}
        };
    }

    static ContextEntry fromJson(const nlohmann::json& j) {
        ContextEntry entry;
        if (!j.is_object()) return entry;
        if (j.contains("content") && j["content"].is_string()) entry.content = j["content"].get<std::string>();
        if (j.contains("role") && j["role"].is_string()) entry.role = j["role"].get<std::string>();
        if (j.contains("turnNumber") && j["turnNumber"].is_number_integer()) entry.turnNumber = j["turnNumber"].get<int64_t>();
        if (j.contains("importance") && j["importance"].is_number()) entry.importance = j["importance"].get<float>();
        if (j.contains("estimatedTokens") && j["estimatedTokens"].is_number_integer()) entry.estimatedTokens = j["estimatedTokens"].get<size_t>();
        if (j.contains("metadata") && j["metadata"].is_object()) entry.metadata = j["metadata"].get<std::map<std::string, std::string>>();

        if (j.contains("timestamp") && j["timestamp"].is_number_integer()) {
            int64_t ts = j["timestamp"].get<int64_t>();
            entry.timestamp = std::chrono::system_clock::time_point(std::chrono::milliseconds(ts));
        }

        return entry;
    }
};

/* Context window configuration */
struct ContextWindowConfig {
    size_t maxTokens{4096};              /* Maximum context window size */
    size_t reservedSystemTokens{256};    /* Reserved for system prompts */
    float importanceThreshold{0.3f};     /* Minimum importance to retain */
    ContextStrategy strategy{ContextStrategy::HYBRID}; /* Context strategy */
    bool enableSemanticSearch{true};    /* Enable semantic similarity search */
    bool enableAttentionSink{true};      /* Enable attention sink mechanism */
    bool enableHierarchical{true};       /* Enable hierarchical organization */
    size_t semanticChunkSize{512};       /* Size of semantic chunks */
    float similarityThreshold{0.7f};     /* Minimum similarity for retrieval */
};

/* Semantic search result */
struct SemanticSearchResult {
    ContextEntry entry;   /* Matching context entry */
    float similarity;     /* Similarity score */
    size_t position;      /* Position in context */

    SemanticSearchResult() : similarity(0.0f), position(0) {}
};

/* Context manager for advanced context handling */
class ModernContextManager {
public:
    explicit ModernContextManager(const ContextWindowConfig& config);
    ~ModernContextManager();

    /* Add context entry */
    bool addEntry(const ContextEntry& entry);

    /* Get context for LLM (optimized for current strategy) */
    std::vector<ContextEntry> getContext(size_t maxTokens = 0);

    /* Semantic search in context */
    std::vector<SemanticSearchResult> semanticSearch(
        const std::string& query,
        size_t topK = 5);

    /* Update importance scores */
    bool updateImportance(const std::string& contentId, float newImportance);

    /* Prune context based on strategy */
    bool pruneContext();

    /* Get context statistics */
    nlohmann::json getStatistics() const;

    /* Clear all context */
    void clear();

    /* Export/Import context */
    std::string exportContext() const;
    bool importContext(const std::string& jsonStr);

    /* Set configuration */
    void setConfig(const ContextWindowConfig& config);
    ContextWindowConfig getConfig() const;

private:
    void pruneContextLocked();
    struct Impl;
    std::unique_ptr<Impl> impl_;

    /* Helper function for text similarity calculation */
    static float calculateTextSimilarity(const std::string& text1, const std::string& text2);
};

/* Attention sink mechanism (inspired by Claude's long context handling) */
class AttentionSinkManager {
public:
    struct SinkConfig {
        size_t sinkTokens{128};           /* Size of attention sink */
        std::string sinkContent{"..."};   /* Default sink content */
        bool dynamicSink{true};           /* Dynamically adjust sink size */
        float sinkImportance{0.1f};       /* Low importance for sink */
    };

    explicit AttentionSinkManager(const SinkConfig& config);

    /* Generate attention sink content */
    std::string generateSink(const std::vector<ContextEntry>& context);

    /* Optimize context with attention sink */
    std::vector<ContextEntry> optimizeWithContextSink(
        const std::vector<ContextEntry>& context);

    /* Update sink configuration */
    void setConfig(const SinkConfig& config);

private:
    SinkConfig config_;
    std::mutex mutex_;
};

/* Hierarchical context organization (inspired by Copilot's workspace context) */
class HierarchicalContextManager {
public:
    struct ContextLevel {
        std::string name;           /* Level name: global, workspace, file, function */
        int priority;              /* Higher priority = more important */
        size_t maxTokens;          /* Max tokens for this level */
        std::vector<ContextEntry> entries; /* Entries at this level */
    };

    explicit HierarchicalContextManager();

    /* Add context at specific level */
    bool addContext(const std::string& level, const ContextEntry& entry);

    /* Get flattened context respecting priorities */
    std::vector<ContextEntry> getFlattenedContext(size_t maxTokens);

    /* Add or update context level */
    bool updateLevel(const std::string& level, int priority, size_t maxTokens);

    /* Remove context level */
    bool removeLevel(const std::string& level);

    /* Get context statistics by level */
    std::map<std::string, nlohmann::json> getLevelStatistics() const;

private:
    std::map<std::string, ContextLevel> levels_;
    mutable std::mutex mutex_;
};

/* Semantic chunking (inspired by modern RAG systems) */
class SemanticChunker {
public:
    struct ChunkConfig {
        size_t chunkSize{512};                    /* Chunk size in tokens */
        size_t chunkOverlap{50};                  /* Chunk overlap in tokens */
        bool respectSentenceBoundaries{true};     /* Respect sentence boundaries */
        bool respectParagraphBoundaries{true};    /* Respect paragraph boundaries */
        float minChunkImportance{0.2f};           /* Minimum chunk importance */
    };

    explicit SemanticChunker(const ChunkConfig& config);

    /* Chunk text semantically */
    std::vector<ContextEntry> chunkText(const std::string& text, const std::string& role = "user");

    /* Merge chunks back into original text */
    std::string mergeChunks(const std::vector<ContextEntry>& chunks);

    /* Get chunk statistics */
    nlohmann::json getChunkStatistics(const std::vector<ContextEntry>& chunks) const;

private:
    ChunkConfig config_;
    std::mutex mutex_;

    std::vector<std::string> splitBySentences(const std::string& text);
    std::vector<std::string> splitByParagraphs(const std::string& text);
    float calculateChunkImportance(const std::string& chunk);
};

/* Context compression (inspired by Ollama's context compression) */
class ContextCompressor {
public:
    struct CompressionConfig {
        float compressionRatio{0.5f};      /* Target compression ratio */
        bool preserveKeyInformation{true}; /* Preserve important information */
        bool useSummarization{true};       /* Use summarization for compression */
        bool useSemanticPruning{true};     /* Use semantic similarity for pruning */
    };

    explicit ContextCompressor(const CompressionConfig& config);

    /* Compress context entries */
    std::vector<ContextEntry> compressContext(
        const std::vector<ContextEntry>& context,
        size_t targetTokens);

    /* Decompress context (if possible) */
    std::vector<ContextEntry> decompressContext(
        const std::vector<ContextEntry>& compressed);

    /* Get compression statistics */
    nlohmann::json getCompressionStatistics(
        const std::vector<ContextEntry>& original,
        const std::vector<ContextEntry>& compressed) const;

private:
    CompressionConfig config_;
    std::mutex mutex_;

    std::string summarizeText(const std::string& text, size_t targetLength);
    std::vector<ContextEntry> semanticPrune(
        const std::vector<ContextEntry>& context,
        size_t targetTokens);
};

/* Context cache for frequently used contexts */
class ContextCache {
public:
    struct CacheEntry {
        std::vector<ContextEntry> context;                  /* Cached context */
        std::chrono::system_clock::time_point lastAccess;   /* Last access time */
        size_t accessCount;                                 /* Access count */
        std::string cacheKey;                               /* Cache key */
    };

    explicit ContextCache(size_t maxEntries = 100);

    /* Get context from cache */
    std::optional<std::vector<ContextEntry>> get(const std::string& key);

    /* Put context in cache */
    void put(const std::string& key, const std::vector<ContextEntry>& context);

    /* Invalidate cache entry */
    void invalidate(const std::string& key);

    /* Clear all cache */
    void clear();

    /* Get cache statistics */
    nlohmann::json getStatistics() const;

private:
    std::map<std::string, CacheEntry> cache_;
    size_t maxEntries_;
    mutable std::mutex mutex_;

    void evictIfNeeded();
};

/* Token counter (language-agnostic estimation) */
class TokenCounter {
public:
    /* Estimate token count for text */
    static size_t estimateTokens(const std::string& text);

    /* Estimate token count for context entries */
    static size_t estimateTokens(const std::vector<ContextEntry>& entries);

    /* Get character-to-token ratio for different languages */
    static float getCharToTokenRatio(const std::string& text);

private:
    static size_t countWords(const std::string& text);
    static size_t countCharacters(const std::string& text);
    static bool isCJK(const std::string& text);
};

/* Context builder for constructing optimized context */
class ContextBuilder {
public:
    explicit ContextBuilder(const ContextWindowConfig& config);

    /* Add system prompt */
    ContextBuilder& addSystemPrompt(const std::string& prompt);

    /* Add user message */
    ContextBuilder& addUserMessage(const std::string& message, float importance = 0.5f);

    /* Add assistant message */
    ContextBuilder& addAssistantMessage(const std::string& message, float importance = 0.5f);

    /* Add tool result */
    ContextBuilder& addToolResult(const std::string& toolName, const std::string& result);

    /* Add custom context entry */
    ContextBuilder& addEntry(const ContextEntry& entry);

    /* Build final context */
    std::vector<ContextEntry> build();

    /* Build with token limit */
    std::vector<ContextEntry> build(size_t maxTokens);

    /* Reset builder */
    void reset();

    /* Get current token count */
    size_t getCurrentTokenCount() const;

private:
    ContextWindowConfig config_;
    std::vector<ContextEntry> entries_;
    size_t currentTokens_;
};

/* Context hooks for integration with existing system */
class ContextHooks {
public:
    /* Called before context is built */
    static std::function<std::vector<ContextEntry>(const std::vector<ContextEntry>&)>
        onPreBuild;

    /* Called after context is built */
    static std::function<std::vector<ContextEntry>(const std::vector<ContextEntry>&)>
        onPostBuild;

    /* Called when context is pruned */
    static std::function<void(const std::vector<ContextEntry>&, const std::vector<ContextEntry>&)>
        onPrune;

    /* Set hooks */
    static void setPreBuildHook(
        std::function<std::vector<ContextEntry>(const std::vector<ContextEntry>&)> hook);
    static void setPostBuildHook(
        std::function<std::vector<ContextEntry>(const std::vector<ContextEntry>&)> hook);
    static void setPruneHook(
        std::function<void(const std::vector<ContextEntry>&, const std::vector<ContextEntry>&)> hook);
};

} // namespace context
} // namespace phoenix
