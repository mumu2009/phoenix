/* memebarrier_phrase_feedback.hpp - Phrase feedback system for threshold adjustment
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

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace memebarrier_phrase_feedback
{

/* Threshold adjustment result */
struct Adjustment
{
    double offset{0.0};                         /* Total offset adjustment */
    std::vector<std::string> positiveMatches;  /* Positive matching phrases */
    std::vector<std::string> negativeMatches;  /* Negative matching phrases */
};

/* Feedback submission result */
struct FeedbackResult
{
    bool ok{false};                    /* Operation success */
    std::string error;                 /* Error message if any */
    std::string phrase;                /* Normalized phrase */
    bool positive{false};              /* Positive feedback flag */
    bool persistent{false};            /* Persistent flag */
    double persistentOffset{0.0};      /* Persistent offset */
    double transientOffset{0.0};       /* Transient offset */
    double combinedOffset{0.0};        /* Combined offset */
};

/* Store for phrase feedback and threshold adjustments */
class Store
{
public:
    using json = nlohmann::json;

    explicit Store(std::string persistentPath = "runtime_store/memebarrier_phrase_feedback.json",
                   double step = 0.05,
                   double maxOffset = 0.30)
        : persistentPath_(std::move(persistentPath)),
          step_(clampStepValue(step)),
          maxOffset_(clampMaxOffsetValue(maxOffset))
    {
    }

    /* Set adjustment step size */
    void setStep(double step)
    {
        std::lock_guard<std::mutex> lock(mu_);
        step_ = clampStepValue(step);
    }

    /* Get adjustment step size */
    double step() const
    {
        std::lock_guard<std::mutex> lock(mu_);
        return step_;
    }

    /* Set maximum offset */
    void setMaxOffset(double maxOffset)
    {
        std::lock_guard<std::mutex> lock(mu_);
        ensureLoadedLocked();
        maxOffset_ = clampMaxOffsetValue(maxOffset);
        clampOffsetsLocked();
        persistPositiveLocked();
    }

    /* Get maximum offset */
    double maxOffset() const
    {
        std::lock_guard<std::mutex> lock(mu_);
        return maxOffset_;
    }

    /* Set configuration */
    void setConfig(double step, double maxOffset)
    {
        std::lock_guard<std::mutex> lock(mu_);
        ensureLoadedLocked();
        step_ = clampStepValue(step);
        maxOffset_ = clampMaxOffsetValue(maxOffset);
        clampOffsetsLocked();
        persistPositiveLocked();
    }

    /* Get store summary */
    json summary() const
    {
        std::lock_guard<std::mutex> lock(mu_);
        ensureLoadedLocked();
        return json{{"persistentPath", persistentPath_},
                    {"step", step_},
                    {"maxOffset", maxOffset_},
                    {"persistentPositiveCount", persistentPositiveOffsets_.size()},
                    {"transientNegativeCount", transientNegativeOffsets_.size()}};
    }

    /* Submit feedback for a phrase */
    FeedbackResult submitFeedback(const std::string &phraseText, bool positive)
    {
        std::lock_guard<std::mutex> lock(mu_);
        ensureLoadedLocked();

        FeedbackResult result;
        result.positive = positive;
        result.persistent = positive;
        result.phrase = normalizePhrase(phraseText);
        if (result.phrase.empty())
        {
            result.error = "phrase required";
            return result;
        }

        if (positive)
        {
            double next = step_;
            auto it = persistentPositiveOffsets_.find(result.phrase);
            if (it != persistentPositiveOffsets_.end())
                next = it->second + step_;
            next = std::max(0.0, std::min(maxOffset_, next));
            if (next <= 1e-9)
                persistentPositiveOffsets_.erase(result.phrase);
            else
                persistentPositiveOffsets_[result.phrase] = next;
            persistPositiveLocked();
        }
        else
        {
            double next = -step_;
            auto it = transientNegativeOffsets_.find(result.phrase);
            if (it != transientNegativeOffsets_.end())
                next = it->second - step_;
            next = std::max(-maxOffset_, std::min(0.0, next));
            if (next >= -1e-9)
                transientNegativeOffsets_.erase(result.phrase);
            else
                transientNegativeOffsets_[result.phrase] = next;
        }

        auto posIt = persistentPositiveOffsets_.find(result.phrase);
        if (posIt != persistentPositiveOffsets_.end())
            result.persistentOffset = posIt->second;
        auto negIt = transientNegativeOffsets_.find(result.phrase);
        if (negIt != transientNegativeOffsets_.end())
            result.transientOffset = negIt->second;
        result.combinedOffset = clampOffsetValue(result.persistentOffset + result.transientOffset, maxOffset_);
        result.ok = true;
        return result;
    }

    /* Compute adjustment for tokens */
    Adjustment computeAdjustment(const std::vector<std::string> &tokens) const
    {
        std::lock_guard<std::mutex> lock(mu_);
        ensureLoadedLocked();

        Adjustment adjustment;
        const auto normalizedTokens = normalizeTokens(tokens);
        if (normalizedTokens.empty())
            return adjustment;

        for (const auto &entry : persistentPositiveOffsets_)
        {
            const auto phraseTokens = splitNormalizedPhrase(entry.first);
            if (containsContiguousPhrase(normalizedTokens, phraseTokens))
            {
                adjustment.offset += entry.second;
                adjustment.positiveMatches.push_back(entry.first);
            }
        }
        for (const auto &entry : transientNegativeOffsets_)
        {
            const auto phraseTokens = splitNormalizedPhrase(entry.first);
            if (containsContiguousPhrase(normalizedTokens, phraseTokens))
            {
                adjustment.offset += entry.second;
                adjustment.negativeMatches.push_back(entry.first);
            }
        }
        adjustment.offset = clampOffsetValue(adjustment.offset, maxOffset_);
        return adjustment;
    }

    /* Normalize tokens */
    static std::vector<std::string> normalizeTokens(const std::vector<std::string> &tokens)
    {
        std::vector<std::string> out;
        out.reserve(tokens.size());
        for (const auto &token : tokens)
        {
            const std::string normalized = lowerToken(trimCopy(token));
            if (!normalized.empty())
                out.push_back(normalized);
        }
        return out;
    }

    /* Normalize phrase */
    static std::string normalizePhrase(const std::string &phrase)
    {
        const auto tokens = tokenizeText(phrase);
        std::ostringstream oss;
        for (size_t i = 0; i < tokens.size(); ++i)
        {
            if (i > 0)
                oss << ' ';
            oss << tokens[i];
        }
        return oss.str();
    }

    /* Split normalized phrase into tokens */
    static std::vector<std::string> splitNormalizedPhrase(const std::string &phrase)
    {
        std::vector<std::string> out;
        std::istringstream iss(phrase);
        std::string token;
        while (iss >> token)
            out.push_back(token);
        return out;
    }

    /* Check if tokens contain contiguous phrase */
    static bool containsContiguousPhrase(const std::vector<std::string> &tokens,
                                         const std::vector<std::string> &phraseTokens)
    {
        if (tokens.empty() || phraseTokens.empty() || phraseTokens.size() > tokens.size())
            return false;
        for (size_t start = 0; start + phraseTokens.size() <= tokens.size(); ++start)
        {
            bool matched = true;
            for (size_t i = 0; i < phraseTokens.size(); ++i)
            {
                if (tokens[start + i] != phraseTokens[i])
                {
                    matched = false;
                    break;
                }
            }
            if (matched)
                return true;
        }
        return false;
    }

private:
    /* Clamp step value to valid range */
    static double clampStepValue(double value)
    {
        if (!std::isfinite(value))
            return 0.05;
        return std::max(0.001, std::min(0.30, value));
    }

    /* Clamp max offset value to valid range */
    static double clampMaxOffsetValue(double value)
    {
        if (!std::isfinite(value))
            return 0.30;
        return std::max(0.01, std::min(0.60, value));
    }

    /* Clamp offset value to valid range */
    static double clampOffsetValue(double value, double maxOffset)
    {
        return std::max(-maxOffset, std::min(maxOffset, value));
    }

    /* Convert token to lowercase */
    static std::string lowerToken(std::string value)
    {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch)
                       { return static_cast<char>(std::tolower(ch)); });
        return value;
    }

    /* Trim whitespace from string */
    static std::string trimCopy(const std::string &value)
    {
        size_t start = 0;
        while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start])))
            ++start;
        size_t end = value.size();
        while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1])))
            --end;
        return value.substr(start, end - start);
    }

    /* Tokenize text */
    static std::vector<std::string> tokenizeText(const std::string &text)
    {
        std::vector<std::string> out;
        std::string current;
        for (unsigned char ch : text)
        {
            if (std::isalnum(ch))
            {
                current.push_back(static_cast<char>(std::tolower(ch)));
                continue;
            }
            if (!current.empty())
            {
                out.push_back(current);
                current.clear();
            }
        }
        if (!current.empty())
            out.push_back(current);
        return out;
    }

    /* Ensure store is loaded from disk */
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
            json positive = doc.value("positiveThresholdOffsets", json::object());
            if (!positive.is_object())
                return;
            for (auto it = positive.begin(); it != positive.end(); ++it)
            {
                if (!it.value().is_number())
                    continue;
                const std::string normalizedPhrase = normalizePhrase(it.key());
                if (normalizedPhrase.empty())
                    continue;
                const double offset = std::max(0.0, std::min(maxOffset_, it.value().get<double>()));
                if (offset > 1e-9)
                    persistentPositiveOffsets_[normalizedPhrase] = offset;
            }
        }
        catch (...)
        {
        }
    }

    /* Persist positive offsets to disk */
    void persistPositiveLocked() const
    {
        try
        {
            const std::filesystem::path path(persistentPath_);
            if (!path.parent_path().empty())
                std::filesystem::create_directories(path.parent_path());
            json doc;
            doc["positiveThresholdOffsets"] = json::object();
            for (const auto &entry : persistentPositiveOffsets_)
                doc["positiveThresholdOffsets"][entry.first] = entry.second;
            std::ofstream output(path, std::ios::binary | std::ios::trunc);
            if (!output)
                return;
            output << doc.dump(2);
        }
        catch (...)
        {
        }
    }

    /* Clamp all offsets to valid range */
    void clampOffsetsLocked()
    {
        for (auto it = persistentPositiveOffsets_.begin(); it != persistentPositiveOffsets_.end();)
        {
            it->second = std::max(0.0, std::min(maxOffset_, it->second));
            if (it->second <= 1e-9)
                it = persistentPositiveOffsets_.erase(it);
            else
                ++it;
        }
        for (auto it = transientNegativeOffsets_.begin(); it != transientNegativeOffsets_.end();)
        {
            it->second = std::max(-maxOffset_, std::min(0.0, it->second));
            if (it->second >= -1e-9)
                it = transientNegativeOffsets_.erase(it);
            else
                ++it;
        }
    }

    std::string persistentPath_; /* Persistent storage path */
    mutable std::mutex mu_;      /* Mutex for thread safety */
    mutable bool loaded_{false}; /* Load flag */
    double step_{0.05};          /* Adjustment step */
    double maxOffset_{0.30};    /* Maximum offset */
    mutable std::unordered_map<std::string, double> persistentPositiveOffsets_; /* Persistent positive offsets */
    mutable std::unordered_map<std::string, double> transientNegativeOffsets_; /* Transient negative offsets */
};

} // namespace memebarrier_phrase_feedback