/* memebarrier_phrase_blocklist.hpp - Phrase blocklist for content filtering */

#pragma once

#include "memebarrier_phrase_feedback.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace memebarrier_phrase_blocklist
{

/* Match result for phrase blocking */
struct Match
{
    bool blocked{false};                    /* Whether content is blocked */
    std::vector<std::string> matches;     /* Matching phrases */
};

/* Import result for phrase blocklist */
struct ImportResult
{
    bool ok{false};                        /* Import success */
    std::vector<std::string> imported;    /* Imported phrases */
    std::vector<std::string> duplicates;  /* Duplicate phrases */
    std::vector<std::string> rejected;   /* Rejected phrases */
};

/* Store for blocked phrases */
class Store
{
public:
    using json = nlohmann::json;

    explicit Store(std::string persistentPath = "runtime_store/memebarrier_phrase_blocklist.json")
        : persistentPath_(std::move(persistentPath))
    {
    }

    /* Get store summary */
    json summary() const
    {
        std::lock_guard<std::mutex> lock(mu_);
        ensureLoadedLocked();
        return json{{"persistentPath", persistentPath_},
                    {"persistentBlockedCount", blockedPhrases_.size()}};
    }

    /* Import phrases into blocklist */
    ImportResult importPhrases(const std::vector<std::string> &phrases)
    {
        std::lock_guard<std::mutex> lock(mu_);
        ensureLoadedLocked();

        ImportResult result;
        for (const auto &phraseText : phrases)
        {
            const std::string normalized = memebarrier_phrase_feedback::Store::normalizePhrase(phraseText);
            if (normalized.empty())
            {
                result.rejected.push_back(phraseText);
                continue;
            }
            auto inserted = blockedPhrases_.insert(normalized);
            if (inserted.second)
                result.imported.push_back(normalized);
            else
                result.duplicates.push_back(normalized);
        }
        if (!result.imported.empty())
            persistLocked();
        result.ok = !result.imported.empty() || !result.duplicates.empty();
        return result;
    }

    /* Match tokens against blocklist */
    Match matchTokens(const std::vector<std::string> &tokens) const
    {
        std::lock_guard<std::mutex> lock(mu_);
        ensureLoadedLocked();

        Match match;
        const auto normalizedTokens = memebarrier_phrase_feedback::Store::normalizeTokens(tokens);
        if (normalizedTokens.empty())
            return match;

        for (const auto &phrase : blockedPhrases_)
        {
            const auto phraseTokens = memebarrier_phrase_feedback::Store::splitNormalizedPhrase(phrase);
            if (memebarrier_phrase_feedback::Store::containsContiguousPhrase(normalizedTokens, phraseTokens))
                match.matches.push_back(phrase);
        }
        match.blocked = !match.matches.empty();
        return match;
    }

private:
    /* Ensure blocklist is loaded from disk */
    void ensureLoadedLocked() const
    {
        if (loaded_)
            return;
        loaded_ = true;
        try
        {
            const std::filesystem::path path(persistentPath_);
            if (path.empty())
                return;
            std::error_code ec;
            if (!std::filesystem::exists(path, ec) || !std::filesystem::is_regular_file(path, ec))
                return;
            std::ifstream input(path, std::ios::binary);
            if (!input)
                return;
            json doc = json::parse(input, nullptr, true, true);
            if (!doc.is_object())
                return;
            const json blocked = doc.value("blockedPhrases", json::array());
            if (!blocked.is_array())
                return;
            for (const auto &entry : blocked)
            {
                if (!entry.is_string())
                    continue;
                const std::string normalized = memebarrier_phrase_feedback::Store::normalizePhrase(entry.get<std::string>());
                if (!normalized.empty())
                    blockedPhrases_.insert(normalized);
            }
        }
        catch (...)
        {
        }
    }

    /* Persist blocklist to disk */
    void persistLocked() const
    {
        try
        {
            const std::filesystem::path path(persistentPath_);
            if (!path.parent_path().empty())
                std::filesystem::create_directories(path.parent_path());
            json doc;
            doc["blockedPhrases"] = json::array();
            for (const auto &phrase : blockedPhrases_)
                doc["blockedPhrases"].push_back(phrase);
            std::ofstream output(path, std::ios::binary | std::ios::trunc);
            if (!output)
                return;
            output << doc.dump(2);
        }
        catch (...)
        {
        }
    }

    std::string persistentPath_; /* Persistent storage path */
    mutable std::mutex mu_;      /* Mutex for thread safety */
    mutable bool loaded_{false}; /* Load flag */
    mutable std::set<std::string> blockedPhrases_; /* Blocked phrases */
};

} // namespace memebarrier_phrase_blocklist