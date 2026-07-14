/* reinforcement_learner_advanced.cpp - Advanced reinforcement learner implementation
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
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{

	using json = nlohmann::json;
	namespace fs = std::filesystem;

	static std::int64_t rlTickNow()
	{
		return static_cast<std::int64_t>(std::time(nullptr));
	}

	struct DenseVector
	{
		std::vector<double> v;
		DenseVector() = default;
		explicit DenseVector(std::size_t n) : v(n, 0.0) {}
		std::size_t size() const { return v.size(); }
		double &operator[](std::size_t i) { return v[i]; }
		const double &operator[](std::size_t i) const { return v[i]; }
	};

	static double vectorDot(const DenseVector &a, const DenseVector &b)
	{
		std::size_t n = std::min(a.size(), b.size());
		double acc = 0.0;
		for (std::size_t i = 0; i < n; ++i)
		{
			acc += a[i] * b[i];
		}
		return acc;
	}

	static std::vector<std::string> tokenSplit(const std::string &text)
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

	static DenseVector embedDocument(const std::string &text, std::size_t dim)
	{
		DenseVector result(dim);
		auto tokens = tokenSplit(text);
		if (tokens.empty() || dim == 0)
		{
			return result;
		}
		for (const auto &token : tokens)
		{
			std::uint64_t h = 1469598103934665603ull;
			for (unsigned char c : token)
			{
				h ^= c;
				h *= 1099511628211ull;
			}
			std::size_t idx = static_cast<std::size_t>(h % dim);
			result[idx] += 1.0;
		}
		double n = std::sqrt(vectorDot(result, result));
		if (n > 0.0)
		{
			for (double &x : result.v)
			{
				x /= n;
			}
		}
		return result;
	}

	static DenseVector softmaxVector(const DenseVector &logits)
	{
		DenseVector out(logits.size());
		if (logits.size() == 0)
		{
			return out;
		}
		double m = -std::numeric_limits<double>::infinity();
		for (double x : logits.v)
		{
			m = std::max(m, x);
		}
		double z = 0.0;
		for (std::size_t i = 0; i < logits.size(); ++i)
		{
			out[i] = std::exp(logits[i] - m);
			z += out[i];
		}
		if (z > 0.0)
		{
			for (double &x : out.v)
			{
				x /= z;
			}
		}
		return out;
	}

	struct Transition
	{
		DenseVector state;
		int action{0};
		double reward{0.0};
		double done{0.0};
		double behaviorProb{1e-6};
		double valueEstimate{0.0};
		double risk{0.0};
		double domainWeight{1.0};
		std::int64_t stepId{0};
	};

	class PolicyHead
	{
	public:
		PolicyHead(std::size_t stateDim, std::size_t actionDim, std::uint32_t seed)
		    : stateDim_(stateDim), actionDim_(actionDim), rng_(seed), weight_(actionDim, DenseVector(stateDim))
		{
			std::normal_distribution<double> init(0.0, 0.04);
			for (auto &row : weight_)
			{
				for (double &x : row.v)
				{
					x = init(rng_);
				}
			}
		}

		DenseVector probability(const DenseVector &state) const
		{
			DenseVector logits(actionDim_);
			for (std::size_t a = 0; a < actionDim_; ++a)
			{
				logits[a] = vectorDot(weight_[a], state);
			}
			return softmaxVector(logits);
		}

		std::size_t actionDim() const { return actionDim_; }

		void updateByGradient(const DenseVector &state, const DenseVector &grad, double lr)
		{
			for (std::size_t a = 0; a < actionDim_; ++a)
			{
				double g = a < grad.size() ? grad[a] : 0.0;
				for (std::size_t i = 0; i < stateDim_; ++i)
				{
					weight_[a][i] += lr * g * state[i];
				}
			}
		}

	private:
		std::size_t stateDim_;
		std::size_t actionDim_;
		mutable std::mt19937 rng_;
		std::vector<DenseVector> weight_;
	};

	class ValueHead
	{
	public:
		ValueHead(std::size_t stateDim, std::uint32_t seed)
		    : rng_(seed), weight_(stateDim)
		{
			std::normal_distribution<double> init(0.0, 0.03);
			for (double &x : weight_.v)
			{
				x = init(rng_);
			}
		}

		double estimate(const DenseVector &state) const
		{
			return vectorDot(weight_, state);
		}

		void trainStep(const DenseVector &state, double target, double lr)
		{
			double err = target - estimate(state);
			for (std::size_t i = 0; i < state.size(); ++i)
			{
				weight_[i] += lr * err * state[i];
			}
		}

	private:
		mutable std::mt19937 rng_;
		DenseVector weight_;
	};

	class ReplayDataset
	{
	public:
		explicit ReplayDataset(std::size_t cap) : cap_(std::max<std::size_t>(cap, 64)) {}

		std::size_t push(const Transition &transition)
		{
			if (data_.size() < cap_)
			{
				data_.push_back(transition);
				return data_.size() - 1;
			}
			data_[head_] = transition;
			std::size_t idx = head_;
			head_ = (head_ + 1) % cap_;
			return idx;
		}

		const Transition &at(std::size_t index) const { return data_[index % data_.size()]; }
		std::size_t size() const { return data_.size(); }
		bool empty() const { return data_.empty(); }
		std::size_t capacity() const { return cap_; }

	private:
		std::size_t cap_;
		std::size_t head_{0};
		std::vector<Transition> data_;
	};

	class PrioritizedReplayTree
	{
	public:
		explicit PrioritizedReplayTree(std::size_t capacity)
		    : cap_(std::max<std::size_t>(capacity, 64)),
		      tree_(2 * cap_, 0.0),
		      priority_(cap_, 1.0) {}

		void updatePriority(std::size_t index, double p)
		{
			std::size_t i = index % cap_;
			double clipped = std::clamp(p, 1e-6, 1000.0);
			double delta = clipped - priority_[i];
			priority_[i] = clipped;
			std::size_t node = i + cap_;
			while (node > 0)
			{
				tree_[node] += delta;
				node /= 2;
			}
		}

		std::size_t sampleIndex(double mass) const
		{
			if (tree_[1] <= 0.0)
			{
				return 0;
			}
			double x = std::fmod(std::fabs(mass), tree_[1]);
			std::size_t node = 1;
			while (node < cap_)
			{
				std::size_t left = node * 2;
				if (x <= tree_[left])
				{
					node = left;
				}
				else
				{
					x -= tree_[left];
					node = left + 1;
				}
			}
			return node - cap_;
		}

		double total() const { return tree_[1]; }

	private:
		std::size_t cap_;
		std::vector<double> tree_;
		std::vector<double> priority_;
	};

	class TrajectoryStore
	{
	public:
		void addTrajectory(std::vector<Transition> trajectory)
		{
			if (!trajectory.empty())
			{
				trajectories_.push_back(std::move(trajectory));
			}
			if (trajectories_.size() > maxTraj_)
			{
				trajectories_.erase(trajectories_.begin(), trajectories_.begin() + static_cast<std::ptrdiff_t>(trajectories_.size() - maxTraj_));
			}
		}

		std::vector<Transition> flatten() const
		{
			std::vector<Transition> all;
			for (const auto &traj : trajectories_)
			{
				all.insert(all.end(), traj.begin(), traj.end());
			}
			return all;
		}

		std::size_t trajectoryCount() const { return trajectories_.size(); }
		std::size_t transitionCount() const
		{
			std::size_t n = 0;
			for (const auto &traj : trajectories_)
			{
				n += traj.size();
			}
			return n;
		}

	private:
		std::size_t maxTraj_{128};
		std::vector<std::vector<Transition>> trajectories_;
	};

	class GaeEstimator
	{
	public:
		std::vector<double> compute(const std::vector<Transition> &trace, double gamma, double lambda) const
		{
			std::vector<double> adv(trace.size(), 0.0);
			double running = 0.0;
			for (std::size_t r = 0; r < trace.size(); ++r)
			{
				std::size_t i = trace.size() - 1 - r;
				double nextV = (i + 1 < trace.size()) ? trace[i + 1].valueEstimate : 0.0;
				double td = trace[i].reward + gamma * (1.0 - trace[i].done) * nextV - trace[i].valueEstimate;
				running = td + gamma * lambda * (1.0 - trace[i].done) * running;
				adv[i] = running;
			}
			normalize(adv);
			return adv;
		}

	private:
		static void normalize(std::vector<double> &x)
		{
			if (x.empty())
			{
				return;
			}
			double m = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
			double var = 0.0;
			for (double v : x)
			{
				var += (v - m) * (v - m);
			}
			var /= static_cast<double>(x.size());
			double s = std::sqrt(std::max(1e-8, var));
			for (double &v : x)
			{
				v = (v - m) / s;
			}
		}
	};

	class NStepReturn
	{
	public:
		std::vector<double> compute(const std::vector<Transition> &trace, int nStep, double gamma) const
		{
			std::vector<double> out(trace.size(), 0.0);
			nStep = std::clamp(nStep, 1, 16);
			for (std::size_t i = 0; i < trace.size(); ++i)
			{
				double g = 0.0;
				double discount = 1.0;
				for (int n = 0; n < nStep; ++n)
				{
					std::size_t j = i + static_cast<std::size_t>(n);
					if (j >= trace.size())
					{
						break;
					}
					g += discount * trace[j].reward;
					discount *= gamma;
					if (trace[j].done > 0.5)
					{
						break;
					}
				}
				out[i] = g;
			}
			return out;
		}
	};

	class RewardShapingPipeline
	{
	public:
		struct Metrics
		{
			double stability{0.0};
			double sparsity{0.0};
			double coverage{0.0};
			double shaped{0.0};
		};

		Metrics shape(const DenseVector &state, double rawReward, int action, std::size_t actionDim) const
		{
			Metrics m;
			m.stability = computeStability(state);
			m.sparsity = computeSparsity(state);
			m.coverage = computeCoverage(action, actionDim);
			m.shaped = rawReward + 0.32 * m.stability + 0.18 * m.coverage - 0.15 * (1.0 - m.sparsity);
			return m;
		}

	private:
		static double computeStability(const DenseVector &state)
		{
			if (state.size() == 0)
			{
				return 0.0;
			}
			double l2 = std::sqrt(vectorDot(state, state));
			return std::clamp(std::tanh(l2), 0.0, 1.0);
		}

		static double computeSparsity(const DenseVector &state)
		{
			if (state.size() == 0)
			{
				return 0.0;
			}
			double active = 0.0;
			for (double x : state.v)
			{
				if (std::fabs(x) > 1e-6)
				{
					active += 1.0;
				}
			}
			double density = active / static_cast<double>(state.size());
			return std::clamp(1.0 - density, 0.0, 1.0);
		}

		static double computeCoverage(int action, std::size_t actionDim)
		{
			if (actionDim == 0)
			{
				return 0.0;
			}
			double centered = std::fabs(static_cast<double>(action) - 0.5 * static_cast<double>(actionDim - 1));
			double normalized = centered / std::max(1.0, 0.5 * static_cast<double>(actionDim - 1));
			return std::clamp(1.0 - normalized, 0.0, 1.0);
		}
	};

	class ConstraintMonitor
	{
	public:
		struct State
		{
			double meanRisk{0.0};
			double riskConstraint{0.45};
			double violation{0.0};
		};

		State assess(const std::vector<Transition> &trace, double riskConstraint) const
		{
			State s;
			s.riskConstraint = std::clamp(riskConstraint, 0.05, 2.0);
			if (trace.empty())
			{
				return s;
			}
			for (const auto &t : trace)
			{
				s.meanRisk += t.risk;
			}
			s.meanRisk /= static_cast<double>(trace.size());
			s.violation = std::max(0.0, s.meanRisk - s.riskConstraint);
			return s;
		}
	};

	class LagrangeUpdater
	{
	public:
		double update(double lambda, double violation, double lr) const
		{
			double next = lambda + lr * violation;
			return std::clamp(next, 0.0, 3.0);
		}
	};

	class OffPolicySuite
	{
	public:
		json evaluate(const std::vector<Transition> &batch, PolicyHead &policy) const
		{
			if (batch.empty())
			{
				return json{{"weightedIS", 0.0}, {"doublyRobust", 0.0}, {"fqeLite", 0.0}, {"count", 0}};
			}

			double wisNum = 0.0;
			double wisDen = 0.0;
			double dr = 0.0;
			double fqe = 0.0;
			for (const auto &t : batch)
			{
				DenseVector p = policy.probability(t.state);
				double pi = (t.action >= 0 && static_cast<std::size_t>(t.action) < p.size()) ? p[static_cast<std::size_t>(t.action)] : 1e-6;
				double w = std::clamp(pi / std::max(1e-6, t.behaviorProb), 0.05, 15.0);
				wisNum += w * t.reward;
				wisDen += w;
				dr += t.valueEstimate + w * (t.reward - t.valueEstimate);
				fqe += 0.8 * t.valueEstimate + 0.2 * t.reward;
			}

			return json{{"weightedIS", wisDen > 0.0 ? wisNum / wisDen : 0.0},
				    {"doublyRobust", dr / static_cast<double>(batch.size())},
				    {"fqeLite", fqe / static_cast<double>(batch.size())},
				    {"count", batch.size()}};
		}
	};

	class CurriculumPlanner
	{
	public:
		struct Phase
		{
			double rolloutMultiplier{1.0};
			double replayFocus{0.5};
			double domainMix{0.5};
		};

		Phase plan(int cycle, double stability, double risk) const
		{
			double progress = std::clamp(static_cast<double>(cycle) / 128.0, 0.0, 1.0);
			Phase p;
			p.rolloutMultiplier = 0.8 + 0.9 * progress + 0.3 * stability;
			p.replayFocus = std::clamp(0.35 + 0.45 * progress + 0.2 * (1.0 - risk), 0.1, 0.95);
			p.domainMix = std::clamp(0.2 + 0.6 * progress + 0.2 * stability, 0.0, 1.0);
			return p;
		}
	};

	class DomainSampler
	{
	public:
		explicit DomainSampler(std::uint32_t seed) : rng_(seed) {}

		std::size_t sample(const std::vector<std::string> &docs, double domainMix)
		{
			if (docs.empty())
			{
				return 0;
			}
			domainMix = std::clamp(domainMix, 0.0, 1.0);
			std::uniform_real_distribution<double> p01(0.0, 1.0);
			if (p01(rng_) < domainMix)
			{
				std::uniform_int_distribution<std::size_t> pick(0, docs.size() - 1);
				return pick(rng_);
			}
			return static_cast<std::size_t>(cursor_++ % static_cast<std::int64_t>(docs.size()));
		}

	private:
		std::mt19937 rng_;
		std::int64_t cursor_{0};
	};

	class CheckpointSerializer
	{
	public:
		explicit CheckpointSerializer(fs::path path) : path_(std::move(path)) {}

		void save(const json &snapshot) const
		{
			std::error_code ec;
			fs::create_directories(path_.parent_path(), ec);
			std::ofstream out(path_, std::ios::binary);
			if (!out)
			{
				return;
			}
			out << snapshot.dump(2);
		}

		json load() const
		{
			std::ifstream in(path_, std::ios::binary);
			if (!in)
			{
				return json::object();
			}
			try
			{
				json data = json::parse(in, nullptr, false);
				return data.is_discarded() ? json::object() : data;
			}
			catch (...)
			{
				return json::object();
			}
		}

	private:
		fs::path path_;
	};

	class PPOBatchOptimizer
	{
	public:
		struct Summary
		{
			double surrogate{0.0};
			double entropy{0.0};
			double kl{0.0};
			double valueLoss{0.0};
		};

		PPOBatchOptimizer(double clip, double policyLr, double valueLr)
		    : clip_(clip), policyLr_(policyLr), valueLr_(valueLr) {}

		Summary train(PolicyHead &policy,
			      ValueHead &value,
			      const std::vector<Transition> &batch,
			      const std::vector<double> &returns,
			      const std::vector<double> &advantages,
			      double entropyCoef,
			      double klCoef) const
		{
			Summary s;
			if (batch.empty())
			{
				return s;
			}

			for (std::size_t i = 0; i < batch.size(); ++i)
			{
				const auto &t = batch[i];
				DenseVector p = policy.probability(t.state);
				DenseVector grad(policy.actionDim());
				for (std::size_t a = 0; a < grad.size(); ++a)
				{
					grad[a] = -p[a];
				}

				if (t.action >= 0 && static_cast<std::size_t>(t.action) < p.size())
				{
					std::size_t a = static_cast<std::size_t>(t.action);
					double pi = std::max(1e-6, p[a]);
					double ratio = pi / std::max(1e-6, t.behaviorProb);
					double unclipped = ratio * advantages[i];
					double clipped = std::clamp(ratio, 1.0 - clip_, 1.0 + clip_) * advantages[i];
					double objective = std::min(unclipped, clipped);
					grad[a] += objective;
					s.surrogate += objective;
					s.kl += std::max(0.0, std::log(std::max(1e-6, t.behaviorProb)) - std::log(pi));
				}

				double entropy = 0.0;
				for (double x : p.v)
				{
					entropy += -x * std::log(std::max(1e-8, x));
				}
				s.entropy += entropy;

				for (double &g : grad.v)
				{
					g += entropyCoef * entropy;
				}
				policy.updateByGradient(t.state, grad, policyLr_);

				value.trainStep(t.state, returns[i], valueLr_);
				double err = returns[i] - value.estimate(t.state);
				s.valueLoss += err * err;
			}

			double n = static_cast<double>(batch.size());
			s.surrogate /= n;
			s.entropy /= n;
			s.kl = (s.kl / n) * klCoef;
			s.valueLoss /= n;
			return s;
		}

	private:
		double clip_;
		double policyLr_;
		double valueLr_;
	};

	class AdvancedReinforcementLearner final : public IReinforcementLearner
	{
	public:
		AdvancedReinforcementLearner(std::shared_ptr<ControllerPoolBase> pool, fs::path testsDir)
		    : pool_(std::move(pool)),
		      testsDir_(std::move(testsDir)),
		      policy_(stateDim_, actionDim_, 20260228u),
		      value_(stateDim_, 20260229u),
		      replay_(4096),
		      replayTree_(4096),
		      sampler_(20260301u),
		      checkpoint_(fs::path("runtime_store") / "rl_advanced_snapshot.json"),
		      ppo_(0.2, 0.0025, 0.004)
		{
			loadTests();
			loadCheckpoint();
		}

		json learn(int cycles) override
		{
			std::lock_guard<std::mutex> lock(mu_);
			cycles = std::clamp(cycles, 1, 128);
			if (docs_.empty())
			{
				loadTests();
			}
			if (docs_.empty())
			{
				latest_ = json{{"ok", false}, {"error", "empty-tests"}, {"ts", rlTickNow()}};
				return latest_;
			}

			json cycleArray = json::array();

			for (int cycle = 0; cycle < cycles; ++cycle)
			{
				auto phase = curriculum_.plan(globalCycle_, movingStability_, movingRisk_);
				auto trajectory = rolloutOneCycle(phase, cycle);
				if (trajectory.empty())
				{
					continue;
				}

				trajectoryStore_.addTrajectory(trajectory);
				auto flat = trajectoryStore_.flatten();
				auto advantages = gae_.compute(flat, gamma_, lambda_);
				auto nstep = nstep_.compute(flat, nStep_, gamma_);

				std::vector<double> returns(flat.size(), 0.0);
				for (std::size_t i = 0; i < flat.size(); ++i)
				{
					returns[i] = 0.7 * nstep[i] + 0.3 * (flat[i].reward + flat[i].valueEstimate);
				}

				auto sampled = sampleFromReplay(phase.replayFocus, 192);
				if (!sampled.empty())
				{
					flat.insert(flat.end(), sampled.begin(), sampled.end());
					auto advReplay = gae_.compute(sampled, gamma_, lambda_);
					auto retReplay = nstep_.compute(sampled, nStep_, gamma_);
					advantages.insert(advantages.end(), advReplay.begin(), advReplay.end());
					returns.insert(returns.end(), retReplay.begin(), retReplay.end());
				}

				auto ppoSummary = ppo_.train(policy_, value_, flat, returns, advantages, entropyCoef_, klCoef_);
				auto constraintState = constraint_.assess(flat, riskLimit_);
				lagrangeLambda_ = lagrange_.update(lagrangeLambda_, constraintState.violation, lagrangeLr_);
				auto offPolicy = offPolicy_.evaluate(flat, policy_);

				movingRisk_ = 0.9 * movingRisk_ + 0.1 * constraintState.meanRisk;
				movingStability_ = 0.9 * movingStability_ + 0.1 * std::clamp(ppoSummary.entropy, 0.0, 1.0);

				cycleArray.push_back(json{{"cycle", globalCycle_},
							  {"rolloutMultiplier", phase.rolloutMultiplier},
							  {"replayFocus", phase.replayFocus},
							  {"domainMix", phase.domainMix},
							  {"surrogate", ppoSummary.surrogate},
							  {"entropy", ppoSummary.entropy},
							  {"kl", ppoSummary.kl},
							  {"valueLoss", ppoSummary.valueLoss},
							  {"meanRisk", constraintState.meanRisk},
							  {"riskViolation", constraintState.violation},
							  {"lagrange", lagrangeLambda_},
							  {"offPolicy", offPolicy}});
				++globalCycle_;
			}

			latest_ = json{{"ok", true},
				       {"cycles", cycles},
				       {"globalCycle", globalCycle_},
				       {"docs", docs_.size()},
				       {"stateDim", stateDim_},
				       {"actionDim", actionDim_},
				       {"trajectoryCount", trajectoryStore_.trajectoryCount()},
				       {"transitionCount", trajectoryStore_.transitionCount()},
				       {"replaySize", replay_.size()},
				       {"movingRisk", movingRisk_},
				       {"movingStability", movingStability_},
				       {"lagrange", lagrangeLambda_},
				       {"cycleInfo", cycleArray},
				       {"ts", rlTickNow()}};

			history_.push_back(latest_);
			if (history_.size() > maxHistory_)
			{
				history_.erase(history_.begin(), history_.begin() + static_cast<std::ptrdiff_t>(history_.size() - maxHistory_));
			}
			saveCheckpoint();
			return latest_;
		}

		json latest() const override
		{
			std::lock_guard<std::mutex> lock(mu_);
			return history_.empty() ? json::object() : history_.back();
		}

		json refreshTests(const fs::path &testsDir) override
		{
			std::lock_guard<std::mutex> lock(mu_);
			testsDir_ = testsDir;
			loadTests();
			return json{{"ok", true}, {"testsDir", testsDir_.string()}, {"docs", docs_.size()}, {"ts", rlTickNow()}};
		}

		json setTestsDir(const fs::path &testsDir) override
		{
			return refreshTests(testsDir);
		}

	private:
		std::vector<Transition> rolloutOneCycle(const CurriculumPlanner::Phase &phase, int cycle)
		{
			std::vector<Transition> out;
			int rolloutLen = static_cast<int>(16.0 * phase.rolloutMultiplier) + (cycle % 24);
			rolloutLen = std::clamp(rolloutLen, 8, 96);
			std::uniform_real_distribution<double> p01(0.0, 1.0);

			out.reserve(static_cast<std::size_t>(rolloutLen));
			for (int i = 0; i < rolloutLen; ++i)
			{
				std::size_t docIdx = sampler_.sample(docs_, phase.domainMix);
				DenseVector state = embedDocument(docs_[docIdx], stateDim_);
				DenseVector prob = policy_.probability(state);
				int action = sampleAction(prob, p01(rng_));
				double semantic = std::clamp(vectorDot(state, state), 0.0, 1.0);
				double risk = std::clamp(0.5 * std::fabs(static_cast<double>(action) - 1.5) + 0.5 * (1.0 - semantic), 0.0, 1.0);

				double baseReward = 0.68 * semantic + 0.32 * (1.0 - 0.25 * std::fabs(static_cast<double>(action) - 1.0));
				auto shaped = shaper_.shape(state, baseReward, action, actionDim_);

				Transition t;
				t.state = std::move(state);
				t.action = action;
				t.risk = risk;
				t.reward = shaped.shaped - lagrangeLambda_ * risk;
				t.done = (i + 1 == rolloutLen) ? 1.0 : 0.0;
				t.behaviorProb = (action >= 0 && static_cast<std::size_t>(action) < prob.size())
						     ? std::max(1e-6, prob[static_cast<std::size_t>(action)])
						     : 1e-6;
				t.valueEstimate = value_.estimate(t.state);
				t.domainWeight = 1.0 + 0.5 * phase.domainMix;
				t.stepId = static_cast<std::int64_t>(globalCycle_) * 1000 + i;
				out.push_back(t);

				std::size_t replayIndex = replay_.push(t);
				replayTree_.updatePriority(replayIndex, std::fabs(t.reward) + 0.5 * t.risk + 1e-3);
			}
			return out;
		}

		std::vector<Transition> sampleFromReplay(double replayFocus, std::size_t count)
		{
			std::vector<Transition> out;
			if (replay_.empty() || count == 0)
			{
				return out;
			}

			replayFocus = std::clamp(replayFocus, 0.0, 1.0);
			std::uniform_real_distribution<double> p01(0.0, 1.0);
			std::uniform_int_distribution<std::size_t> uni(0, replay_.size() - 1);
			out.reserve(count);
			for (std::size_t i = 0; i < count; ++i)
			{
				std::size_t idx = 0;
				if (p01(rng_) < replayFocus && replayTree_.total() > 0.0)
				{
					double mass = p01(rng_) * replayTree_.total();
					idx = replayTree_.sampleIndex(mass);
				}
				else
				{
					idx = uni(rng_);
				}
				out.push_back(replay_.at(idx));
			}
			return out;
		}

		int sampleAction(const DenseVector &prob, double r) const
		{
			if (prob.size() == 0)
			{
				return 0;
			}
			double cum = 0.0;
			for (std::size_t i = 0; i < prob.size(); ++i)
			{
				cum += prob[i];
				if (r <= cum)
				{
					return static_cast<int>(i);
				}
			}
			return static_cast<int>(prob.size() - 1);
		}

		void loadTests()
		{
			docs_.clear();
			if (!fs::exists(testsDir_))
			{
				return;
			}
			std::vector<fs::path> files;
			for (const auto &entry : fs::directory_iterator(testsDir_))
			{
				if (!entry.is_regular_file())
				{
					continue;
				}
				auto ext = entry.path().extension().string();
				std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c)
					       { return static_cast<char>(std::tolower(c)); });
				if (ext == ".txt" || ext == ".md" || ext == ".json" || ext == ".jsonl")
				{
					files.push_back(entry.path());
				}
			}
			std::sort(files.begin(), files.end());
			if (files.size() > 4096)
			{
				files.resize(4096);
			}

			for (const auto &f : files)
			{
				std::ifstream in(f, std::ios::binary);
				if (!in)
				{
					continue;
				}
				std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
				if (!text.empty())
				{
					docs_.push_back(std::move(text));
				}
			}
		}

		void loadCheckpoint()
		{
			json snap = checkpoint_.load();
			if (!snap.is_object())
			{
				return;
			}
			if (snap.contains("globalCycle") && snap["globalCycle"].is_number_integer())
			{
				globalCycle_ = snap["globalCycle"].get<int>();
			}
			if (snap.contains("movingRisk") && snap["movingRisk"].is_number())
			{
				movingRisk_ = snap["movingRisk"].get<double>();
			}
			if (snap.contains("movingStability") && snap["movingStability"].is_number())
			{
				movingStability_ = snap["movingStability"].get<double>();
			}
			if (snap.contains("lagrange") && snap["lagrange"].is_number())
			{
				lagrangeLambda_ = snap["lagrange"].get<double>();
			}
		}

		void saveCheckpoint() const
		{
			checkpoint_.save(json{{"globalCycle", globalCycle_},
					      {"movingRisk", movingRisk_},
					      {"movingStability", movingStability_},
					      {"lagrange", lagrangeLambda_},
					      {"ts", rlTickNow()}});
		}

	private:
		std::shared_ptr<ControllerPoolBase> pool_;
		fs::path testsDir_;
		mutable std::mutex mu_;

		std::vector<std::string> docs_;
		std::vector<json> history_;
		json latest_;

		std::size_t maxHistory_{256};
		std::size_t stateDim_{192};
		std::size_t actionDim_{5};

		double gamma_{0.985};
		double lambda_{0.95};
		int nStep_{5};
		double riskLimit_{0.42};
		double lagrangeLambda_{0.12};
		double lagrangeLr_{0.03};
		double entropyCoef_{0.0015};
		double klCoef_{0.06};

		int globalCycle_{0};
		double movingRisk_{0.25};
		double movingStability_{0.4};

		std::mt19937 rng_{20260303u};

		PolicyHead policy_;
		ValueHead value_;
		ReplayDataset replay_;
		PrioritizedReplayTree replayTree_;
		TrajectoryStore trajectoryStore_;
		GaeEstimator gae_;
		NStepReturn nstep_;
		RewardShapingPipeline shaper_;
		ConstraintMonitor constraint_;
		LagrangeUpdater lagrange_;
		OffPolicySuite offPolicy_;
		CurriculumPlanner curriculum_;
		DomainSampler sampler_;
		CheckpointSerializer checkpoint_;
		PPOBatchOptimizer ppo_;
	};

	struct RegisterRLFactory
	{
		RegisterRLFactory()
		{
			module_mount::registerReinforcementLearnerFactory(
			    [](std::shared_ptr<ControllerPoolBase> pool, const std::filesystem::path &testsDir) -> std::shared_ptr<IReinforcementLearner>
			    {
				    return std::make_shared<AdvancedReinforcementLearner>(std::move(pool), testsDir);
			    });
		}
	};

	static RegisterRLFactory registerRlFactory;

} // namespace
