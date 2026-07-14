/* adversarial_learner_advanced.cpp - Advanced adversarial learner implementation
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

#include "../module_mount.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace
{

	using json = nlohmann::json;

	static std::int64_t advTickNow()
	{
		return static_cast<std::int64_t>(std::time(nullptr));
	}

	static std::vector<std::string> tokenize(const std::string &text)
	{
		std::vector<std::string> out;
		std::string cur;
		for (unsigned char c : text)
		{
			if (std::isalnum(c) != 0)
			{
				cur.push_back(static_cast<char>(std::tolower(c)));
			}
			else if (!cur.empty())
			{
				out.push_back(cur);
				cur.clear();
			}
		}
		if (!cur.empty())
		{
			out.push_back(cur);
		}
		return out;
	}

	static std::string joinTokens(const std::vector<std::string> &token)
	{
		std::string out;
		for (std::size_t i = 0; i < token.size(); ++i)
		{
			if (i > 0)
			{
				out.push_back(' ');
			}
			out += token[i];
		}
		return out;
	}

	static std::uint64_t stableHash(const std::string &s)
	{
		std::uint64_t h = 1469598103934665603ull;
		for (unsigned char c : s)
		{
			h ^= c;
			h *= 1099511628211ull;
		}
		return h;
	}

	struct AttackCandidate
	{
		std::vector<std::string> token;
		std::string strategy;
		double score{0.0};
		double confidenceShift{0.0};
	};

	class AttackStrategy
	{
	public:
		virtual ~AttackStrategy() = default;
		virtual std::string name() const = 0;
		virtual std::vector<AttackCandidate> generate(const std::vector<std::string> &seed,
							      std::size_t budget,
							      std::mt19937 &rng) const = 0;
	};

	class BeamSearchAttack final : public AttackStrategy
	{
	public:
		std::string name() const override { return "BeamSearchAttack"; }

		std::vector<AttackCandidate> generate(const std::vector<std::string> &seed,
						      std::size_t budget,
						      std::mt19937 &rng) const override
		{
			std::vector<AttackCandidate> out;
			if (seed.empty() || budget == 0)
			{
				return out;
			}
			std::uniform_int_distribution<std::size_t> pick(0, seed.size() - 1);
			static const std::vector<std::string> markers = {
			    "context", "priority", "urgent", "override", "assistant", "directive"};
			std::uniform_int_distribution<std::size_t> pickWord(0, markers.size() - 1);

			for (std::size_t i = 0; i < budget; ++i)
			{
				auto token = seed;
				std::size_t p = pick(rng);
				token.insert(token.begin() + static_cast<std::ptrdiff_t>(p), markers[pickWord(rng)]);
				out.push_back(AttackCandidate{token, name(), 0.25 + 0.03 * static_cast<double>(i), 0.02 * static_cast<double>(i)});
			}
			return out;
		}
	};

	class GeneticAttack final : public AttackStrategy
	{
	public:
		std::string name() const override { return "GeneticAttack"; }

		std::vector<AttackCandidate> generate(const std::vector<std::string> &seed,
						      std::size_t budget,
						      std::mt19937 &rng) const override
		{
			std::vector<AttackCandidate> out;
			if (seed.empty() || budget == 0)
			{
				return out;
			}
			std::uniform_int_distribution<std::size_t> pick(0, seed.size() - 1);
			for (std::size_t i = 0; i < budget; ++i)
			{
				auto token = seed;
				std::size_t a = pick(rng);
				std::size_t b = pick(rng);
				if (a != b)
				{
					std::swap(token[a], token[b]);
				}
				if (!token.empty())
				{
					token.back() = token.back() + "_alt";
				}
				out.push_back(AttackCandidate{token, name(), 0.22 + 0.025 * static_cast<double>(i), 0.015 * static_cast<double>(i)});
			}
			return out;
		}
	};

	class TypoAttack final : public AttackStrategy
	{
	public:
		std::string name() const override { return "TypoAttack"; }

		std::vector<AttackCandidate> generate(const std::vector<std::string> &seed,
						      std::size_t budget,
						      std::mt19937 &rng) const override
		{
			std::vector<AttackCandidate> out;
			if (seed.empty() || budget == 0)
			{
				return out;
			}
			std::uniform_int_distribution<std::size_t> pick(0, seed.size() - 1);
			for (std::size_t i = 0; i < budget; ++i)
			{
				auto token = seed;
				std::size_t p = pick(rng);
				if (!token[p].empty())
				{
					char c = token[p][0];
					token[p][0] = static_cast<char>(std::tolower(c));
					token[p].push_back('x');
				}
				out.push_back(AttackCandidate{token, name(), 0.18 + 0.02 * static_cast<double>(i), 0.01 * static_cast<double>(i)});
			}
			return out;
		}
	};

	class ParaphraseAttack final : public AttackStrategy
	{
	public:
		std::string name() const override { return "ParaphraseAttack"; }

		std::vector<AttackCandidate> generate(const std::vector<std::string> &seed,
						      std::size_t budget,
						      std::mt19937 &rng) const override
		{
			std::vector<AttackCandidate> out;
			if (seed.empty() || budget == 0)
			{
				return out;
			}
			static const std::unordered_map<std::string, std::string> synonym = {
			    {"safe", "secure"},
			    {"policy", "protocol"},
			    {"ignore", "skip"},
			    {"system", "platform"},
			    {"answer", "respond"},
			    {"assistant", "agent"}};

			std::uniform_int_distribution<std::size_t> pick(0, seed.size() - 1);
			for (std::size_t i = 0; i < budget; ++i)
			{
				auto token = seed;
				std::size_t p = pick(rng);
				auto it = synonym.find(token[p]);
				if (it != synonym.end())
				{
					token[p] = it->second;
				}
				else
				{
					token[p] = token[p] + "_p";
				}
				out.push_back(AttackCandidate{token, name(), 0.2 + 0.02 * static_cast<double>(i), 0.012 * static_cast<double>(i)});
			}
			return out;
		}
	};

	class StrategyRegistry
	{
	public:
		StrategyRegistry()
		{
			strategies_.push_back(std::make_unique<BeamSearchAttack>());
			strategies_.push_back(std::make_unique<GeneticAttack>());
			strategies_.push_back(std::make_unique<TypoAttack>());
			strategies_.push_back(std::make_unique<ParaphraseAttack>());
		}

		const std::vector<std::unique_ptr<AttackStrategy>> &all() const { return strategies_; }

		json names() const
		{
			json out = json::array();
			for (const auto &s : strategies_)
			{
				out.push_back(s->name());
			}
			return out;
		}

	private:
		std::vector<std::unique_ptr<AttackStrategy>> strategies_;
	};

	class AttackBudgetManager
	{
	public:
		struct Budget
		{
			std::unordered_map<std::string, std::size_t> byStrategy;
			std::size_t total{0};
		};

		Budget allocate(const StrategyRegistry &registry, std::size_t baseBudget, double pressure) const
		{
			Budget b;
			baseBudget = std::clamp<std::size_t>(baseBudget, 8, 4096);
			pressure = std::clamp(pressure, 0.0, 1.0);
			for (const auto &s : registry.all())
			{
				std::size_t local = static_cast<std::size_t>(std::round((0.2 + 0.8 * pressure) * static_cast<double>(baseBudget) / static_cast<double>(registry.all().size())));
				local = std::max<std::size_t>(2, local);
				b.byStrategy[s->name()] = local;
				b.total += local;
			}
			return b;
		}
	};

	class DiversitySelector
	{
	public:
		std::vector<AttackCandidate> select(std::vector<AttackCandidate> candidates, std::size_t k) const
		{
			std::sort(candidates.begin(), candidates.end(), [](const AttackCandidate &a, const AttackCandidate &b)
				  {
            if (a.score == b.score) {
                return a.strategy < b.strategy;
            }
            return a.score > b.score; });

			std::unordered_set<std::string> seen;
			std::vector<AttackCandidate> out;
			out.reserve(k);
			for (const auto &c : candidates)
			{
				std::string key = c.strategy + "::" + joinTokens(c.token);
				if (seen.insert(key).second)
				{
					out.push_back(c);
				}
				if (out.size() >= k)
				{
					break;
				}
			}
			return out;
		}
	};

	class DetectorScoreModel
	{
	public:
		struct ScoreDetail
		{
			double tokenStat{0.0};
			double ngramLogit{0.0};
			double merged{0.0};
		};

		ScoreDetail score(const std::vector<std::string> &token) const
		{
			ScoreDetail d;
			if (token.empty())
			{
				return d;
			}

			static const std::unordered_set<std::string> trigger = {
			    "ignore", "override", "bypass", "system", "admin", "urgent", "disable", "policy"};

			double suspicious = 0.0;
			double entropyLike = 0.0;
			for (const auto &t : token)
			{
				if (trigger.count(t) > 0)
				{
					suspicious += 1.0;
				}
				entropyLike += static_cast<double>(stableHash(t) % 1009) / 1009.0;
			}
			d.tokenStat = std::clamp(0.6 * (suspicious / static_cast<double>(token.size())) + 0.4 * (entropyLike / static_cast<double>(token.size())), 0.0, 1.0);

			double ngramScore = 0.0;
			for (std::size_t i = 1; i < token.size(); ++i)
			{
				std::string bi = token[i - 1] + "_" + token[i];
				std::uint64_t h = stableHash(bi);
				ngramScore += static_cast<double>(h % 997) / 997.0;
			}
			if (token.size() > 1)
			{
				ngramScore /= static_cast<double>(token.size() - 1);
			}
			d.ngramLogit = std::clamp(ngramScore, 0.0, 1.0);
			d.merged = std::clamp(0.55 * d.tokenStat + 0.45 * d.ngramLogit, 0.0, 1.0);
			return d;
		}
	};

	class Calibrator
	{
	public:
		double fitTemperature(const std::vector<double> &score, const std::vector<int> &label) const
		{
			if (score.empty() || label.empty())
			{
				return 1.0;
			}
			double pos = 0.0;
			double neg = 0.0;
			double cp = 0.0;
			double cn = 0.0;
			for (std::size_t i = 0; i < score.size() && i < label.size(); ++i)
			{
				if (label[i] > 0)
				{
					pos += score[i];
					cp += 1.0;
				}
				else
				{
					neg += score[i];
					cn += 1.0;
				}
			}
			pos = cp > 0.0 ? pos / cp : 0.7;
			neg = cn > 0.0 ? neg / cn : 0.3;
			double gap = std::max(1e-4, pos - neg);
			return std::clamp(1.2 / (1.0 + gap), 0.5, 2.5);
		}

		std::vector<double> applyTemperature(const std::vector<double> &score, double t) const
		{
			std::vector<double> out;
			out.reserve(score.size());
			t = std::clamp(t, 0.2, 5.0);
			for (double s : score)
			{
				double logit = std::log(std::max(1e-6, s / std::max(1e-6, 1.0 - s)));
				double scaled = 1.0 / (1.0 + std::exp(-logit / t));
				out.push_back(std::clamp(scaled, 0.0, 1.0));
			}
			return out;
		}

		std::vector<double> isotonicLite(const std::vector<double> &score, const std::vector<int> &label) const
		{
			if (score.empty() || label.empty())
			{
				return score;
			}
			std::vector<std::pair<double, int>> pair;
			for (std::size_t i = 0; i < score.size() && i < label.size(); ++i)
			{
				pair.push_back({score[i], label[i]});
			}
			std::sort(pair.begin(), pair.end(), [](const auto &a, const auto &b)
				  { return a.first < b.first; });

			std::vector<double> out(pair.size(), 0.0);
			double runningPos = 0.0;
			for (std::size_t i = 0; i < pair.size(); ++i)
			{
				if (pair[i].second > 0)
				{
					runningPos += 1.0;
				}
				out[i] = runningPos / static_cast<double>(i + 1);
				if (i > 0 && out[i] < out[i - 1])
				{
					out[i] = out[i - 1];
				}
			}
			return out;
		}
	};

	class ThresholdOptimizer
	{
	public:
		struct Metrics
		{
			double threshold{0.5};
			double fbeta{0.0};
			double precision{0.0};
			double recall{0.0};
			double precisionAtRisk{0.0};
		};

		Metrics optimize(const std::vector<double> &score,
				 const std::vector<int> &label,
				 double beta,
				 double riskLimit) const
		{
			Metrics best;
			if (score.empty() || label.empty())
			{
				return best;
			}
			beta = std::clamp(beta, 0.25, 4.0);
			riskLimit = std::clamp(riskLimit, 0.01, 0.9);

			for (int b = 5; b <= 95; ++b)
			{
				double threshold = static_cast<double>(b) / 100.0;
				double tp = 0.0;
				double fp = 0.0;
				double fn = 0.0;
				for (std::size_t i = 0; i < score.size() && i < label.size(); ++i)
				{
					int pred = score[i] >= threshold ? 1 : 0;
					if (pred == 1 && label[i] == 1)
						tp += 1.0;
					if (pred == 1 && label[i] == 0)
						fp += 1.0;
					if (pred == 0 && label[i] == 1)
						fn += 1.0;
				}
				double precision = (tp + fp > 0.0) ? (tp / (tp + fp)) : 0.0;
				double recall = (tp + fn > 0.0) ? (tp / (tp + fn)) : 0.0;
				double b2 = beta * beta;
				double fbeta = (precision + recall > 0.0) ? ((1.0 + b2) * precision * recall / (b2 * precision + recall)) : 0.0;
				double predRisk = (tp + fp > 0.0) ? (fp / (tp + fp)) : 1.0;
				double precisionAtRisk = predRisk <= riskLimit ? precision : 0.0;

				if (fbeta + 0.2 * precisionAtRisk > best.fbeta + 0.2 * best.precisionAtRisk)
				{
					best.threshold = threshold;
					best.fbeta = fbeta;
					best.precision = precision;
					best.recall = recall;
					best.precisionAtRisk = precisionAtRisk;
				}
			}
			return best;
		}
	};

	class HardNegativeQueue
	{
	public:
		explicit HardNegativeQueue(std::size_t cap) : cap_(std::max<std::size_t>(cap, 64)) {}

		void push(const std::vector<std::string> &token, double score)
		{
			Item it;
			it.token = token;
			it.score = score;
			queue_.push_back(std::move(it));
			while (queue_.size() > cap_)
			{
				queue_.pop_front();
			}
		}

		std::vector<std::vector<std::string>> top(std::size_t k) const
		{
			std::vector<Item> tmp(queue_.begin(), queue_.end());
			std::sort(tmp.begin(), tmp.end(), [](const Item &a, const Item &b)
				  { return a.score > b.score; });
			std::vector<std::vector<std::string>> out;
			for (std::size_t i = 0; i < tmp.size() && i < k; ++i)
			{
				out.push_back(tmp[i].token);
			}
			return out;
		}

	private:
		struct Item
		{
			std::vector<std::string> token;
			double score{0.0};
		};

		std::size_t cap_;
		std::deque<Item> queue_;
	};

	class MemoryBank
	{
	public:
		explicit MemoryBank(std::size_t cap) : cap_(std::max<std::size_t>(cap, 128)) {}

		void add(const AttackCandidate &candidate)
		{
			memory_.push_back(candidate);
			if (memory_.size() > cap_)
			{
				memory_.erase(memory_.begin(), memory_.begin() + static_cast<std::ptrdiff_t>(memory_.size() - cap_));
			}
		}

		std::vector<AttackCandidate> recent(std::size_t k) const
		{
			if (memory_.size() <= k)
			{
				return memory_;
			}
			return std::vector<AttackCandidate>(memory_.end() - static_cast<std::ptrdiff_t>(k), memory_.end());
		}

		std::size_t size() const { return memory_.size(); }

	private:
		std::size_t cap_;
		std::vector<AttackCandidate> memory_;
	};

	class RobustnessEvaluator
	{
	public:
		json evaluate(const std::vector<AttackCandidate> &samples,
			      const std::vector<double> &raw,
			      const std::vector<double> &calibrated,
			      double threshold) const
		{
			std::unordered_map<std::string, double> total;
			std::unordered_map<std::string, double> success;
			double confidenceShift = 0.0;
			double ece = 0.0;

			for (std::size_t i = 0; i < samples.size() && i < raw.size() && i < calibrated.size(); ++i)
			{
				total[samples[i].strategy] += 1.0;
				if (calibrated[i] >= threshold)
				{
					success[samples[i].strategy] += 1.0;
				}
				confidenceShift += std::fabs(calibrated[i] - raw[i]);
				double y = calibrated[i] >= threshold ? 1.0 : 0.0;
				ece += std::fabs(y - calibrated[i]);
			}

			json perStrategy = json::object();
			for (const auto &kv : total)
			{
				double s = success[kv.first];
				perStrategy[kv.first] = kv.second > 0.0 ? (s / kv.second) : 0.0;
			}

			double n = std::max(1.0, static_cast<double>(samples.size()));
			return json{{"perStrategySuccess", perStrategy},
				    {"confidenceShift", confidenceShift / n},
				    {"calibrationError", ece / n}};
		}
	};

	class DefenseTuner
	{
	public:
		json tune(const std::vector<std::vector<std::string>> &hardNeg,
			  double threshold,
			  double calibrationError)
		{
			double augmentFactor = std::clamp(0.5 + 2.0 * calibrationError, 0.5, 3.0);
			int retrainSteps = static_cast<int>(std::round(12.0 + 90.0 * calibrationError + 20.0 * threshold));
			retrainSteps = std::clamp(retrainSteps, 8, 256);
			return json{{"hardNegativeCount", hardNeg.size()},
				    {"augmentFactor", augmentFactor},
				    {"retrainSteps", retrainSteps}};
		}
	};

	class AdvancedAdversarialLearner final : public IAdversarialLearner
	{
	public:
		explicit AdvancedAdversarialLearner(std::shared_ptr<ControllerPoolBase> pool)
		    : pool_(std::move(pool)), rng_(20260228u), hardQueue_(512), memoryBank_(2048)
		{
			bootstrap();
		}

		json attackAndDefend(const json &samples) override
		{
			std::lock_guard<std::mutex> lock(mu_);
			ingest(samples);

			if (clean_.empty())
			{
				latest_ = json{{"ok", false}, {"error", "empty-samples"}, {"ts", advTickNow()}};
				return latest_;
			}

			double pressure = std::clamp(0.2 + 0.6 * static_cast<double>(clean_.size()) / 512.0, 0.0, 1.0);
			auto budget = budgeter_.allocate(registry_, 160, pressure);
			auto candidates = generateCandidates(budget);
			auto selected = diversity_.select(std::move(candidates), 320);

			std::vector<double> rawScore;
			std::vector<int> label;
			rawScore.reserve(selected.size());
			label.reserve(selected.size());

			for (auto &cand : selected)
			{
				auto detail = detector_.score(cand.token);
				cand.score = detail.merged;
				cand.confidenceShift = std::fabs(detail.tokenStat - detail.ngramLogit);
				rawScore.push_back(cand.score);
				label.push_back(cand.score >= 0.55 ? 1 : 0);
				memoryBank_.add(cand);
				if (cand.score >= 0.62)
				{
					hardQueue_.push(cand.token, cand.score);
				}
			}

			double temperature = calibrator_.fitTemperature(rawScore, label);
			auto tsScore = calibrator_.applyTemperature(rawScore, temperature);
			auto isoCurve = calibrator_.isotonicLite(tsScore, label);

			std::vector<double> calibrated = tsScore;
			if (!isoCurve.empty())
			{
				for (std::size_t i = 0; i < calibrated.size(); ++i)
				{
					calibrated[i] = 0.6 * tsScore[i] + 0.4 * isoCurve[std::min(i, isoCurve.size() - 1)];
					calibrated[i] = std::clamp(calibrated[i], 0.0, 1.0);
				}
			}

			auto best = threshold_.optimize(calibrated, label, 1.2, 0.25);
			thresholdValue_ = std::clamp(0.7 * best.threshold + 0.3 * thresholdValue_, 0.15, 0.95);

			auto hardNeg = hardQueue_.top(256);
			auto robust = evaluator_.evaluate(selected, rawScore, calibrated, thresholdValue_);
			auto defense = tuner_.tune(hardNeg, thresholdValue_, robust.value("calibrationError", 0.0));

			latest_ = json{{"ok", true},
				       {"cleanSamples", clean_.size()},
				       {"strategies", registry_.names()},
				       {"budgetTotal", budget.total},
				       {"candidateCount", selected.size()},
				       {"temperature", temperature},
				       {"threshold", thresholdValue_},
				       {"thresholdMetrics", json{{"threshold", best.threshold},
								 {"fbeta", best.fbeta},
								 {"precision", best.precision},
								 {"recall", best.recall},
								 {"precisionAtRisk", best.precisionAtRisk}}},
				       {"hardNegativeQueue", hardNeg.size()},
				       {"memoryBank", memoryBank_.size()},
				       {"robustness", robust},
				       {"defenseTuning", defense},
				       {"ts", advTickNow()}};

			history_.push_back(latest_);
			if (history_.size() > maxHistory_)
			{
				history_.erase(history_.begin(), history_.begin() + static_cast<std::ptrdiff_t>(history_.size() - maxHistory_));
			}
			return latest_;
		}

		json latest() const override
		{
			std::lock_guard<std::mutex> lock(mu_);
			return history_.empty() ? json::object() : history_.back();
		}

	private:
		void bootstrap()
		{
			static const std::vector<std::string> bootstrapSamples = {
			    "safe dialogue with explicit intent and transparent policy",
			    "context aware assistant behavior under constrained operation",
			    "benign user question with factual grounded answer",
			    "request clarification before high impact operations"};
			for (const auto &s : bootstrapSamples)
			{
				auto t = tokenize(s);
				if (!t.empty())
				{
					clean_.push_back(std::move(t));
				}
			}
		}

		void ingest(const json &samples)
		{
			auto ingestText = [this](const std::string &text)
			{
				auto t = tokenize(text);
				if (!t.empty())
				{
					clean_.push_back(std::move(t));
				}
			};

			if (samples.is_array())
			{
				for (const auto &v : samples)
				{
					if (v.is_string())
					{
						ingestText(v.get<std::string>());
					}
				}
			}
			if (samples.is_object() && samples.contains("samples") && samples["samples"].is_array())
			{
				for (const auto &v : samples["samples"])
				{
					if (v.is_string())
					{
						ingestText(v.get<std::string>());
					}
				}
			}

			if (clean_.size() > 1024)
			{
				clean_.erase(clean_.begin(), clean_.begin() + static_cast<std::ptrdiff_t>(clean_.size() - 1024));
			}
		}

		std::vector<AttackCandidate> generateCandidates(const AttackBudgetManager::Budget &budget)
		{
			std::vector<AttackCandidate> out;
			std::size_t seedLimit = std::min<std::size_t>(clean_.size(), 96);
			for (std::size_t i = 0; i < seedLimit; ++i)
			{
				const auto &seed = clean_[i];
				for (const auto &strategy : registry_.all())
				{
					std::size_t b = 4;
					auto it = budget.byStrategy.find(strategy->name());
					if (it != budget.byStrategy.end())
					{
						b = std::max<std::size_t>(2, it->second / std::max<std::size_t>(1, seedLimit));
					}
					auto local = strategy->generate(seed, b, rng_);
					out.insert(out.end(), local.begin(), local.end());
				}
			}
			return out;
		}

	private:
		std::shared_ptr<ControllerPoolBase> pool_;
		mutable std::mutex mu_;

		std::vector<std::vector<std::string>> clean_;
		std::vector<json> history_;
		json latest_;
		std::size_t maxHistory_{256};

		std::mt19937 rng_;
		double thresholdValue_{0.55};

		StrategyRegistry registry_;
		AttackBudgetManager budgeter_;
		DiversitySelector diversity_;
		DetectorScoreModel detector_;
		Calibrator calibrator_;
		ThresholdOptimizer threshold_;
		HardNegativeQueue hardQueue_;
		MemoryBank memoryBank_;
		RobustnessEvaluator evaluator_;
		DefenseTuner tuner_;
	};

	struct RegisterAdvFactory
	{
		RegisterAdvFactory()
		{
			module_mount::registerAdversarialLearnerFactory(
			    [](std::shared_ptr<ControllerPoolBase> pool) -> std::shared_ptr<IAdversarialLearner>
			    {
				    return std::make_shared<AdvancedAdversarialLearner>(std::move(pool));
			    });
		}
	};

	static RegisterAdvFactory registerAdvFactory;

} // namespace
