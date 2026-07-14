/* gnn_ga_learner_advanced.cpp - Advanced GNN GA learner implementation
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
#include <array>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
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
	namespace fs = std::filesystem;

	static std::int64_t gnnTick() { return static_cast<std::int64_t>(std::time(nullptr)); }

	struct Matrix
	{
		std::size_t row{0};
		std::size_t col{0};
		std::vector<double> val;

		Matrix() = default;
		Matrix(std::size_t r, std::size_t c) : row(r), col(c), val(r * c, 0.0) {}

		double &at(std::size_t r, std::size_t c) { return val[r * col + c]; }
		const double &at(std::size_t r, std::size_t c) const { return val[r * col + c]; }
	};

	struct Graph
	{
		std::vector<std::string> node;
		std::vector<std::pair<std::size_t, std::size_t>> edge;
		std::vector<std::vector<std::size_t>> adj;
	};

	static std::vector<std::string> token(const std::string &text)
	{
		std::vector<std::string> out;
		std::string cur;
		cur.reserve(16);
		for (unsigned char c : text)
		{
			if (std::isalnum(c))
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
			out.push_back(cur);
		return out;
	}

	class GraphBuilder
	{
	public:
		Graph fromDocuments(const std::vector<std::string> &docs, std::size_t nodeCap, std::size_t edgeCap) const
		{
			Graph g;
			std::unordered_map<std::string, std::size_t> id;
			id.reserve(8192);
			for (const auto &d : docs)
			{
				auto t = token(d);
				if (t.empty())
					continue;
				for (const auto &w : t)
				{
					if (id.find(w) != id.end())
						continue;
					std::size_t idx = g.node.size();
					id.emplace(w, idx);
					g.node.push_back(w);
					if (g.node.size() >= nodeCap)
						break;
				}
				if (g.node.size() >= nodeCap)
					break;
			}

			for (const auto &d : docs)
			{
				auto t = token(d);
				if (t.size() < 2)
					continue;
				for (std::size_t i = 1; i < t.size(); ++i)
				{
					auto itU = id.find(t[i - 1]);
					auto itV = id.find(t[i]);
					if (itU == id.end() || itV == id.end())
						continue;
					g.edge.push_back({itU->second, itV->second});
					if (g.edge.size() >= edgeCap)
						break;
				}
				if (g.edge.size() >= edgeCap)
					break;
			}

			g.adj.assign(g.node.size(), {});
			for (const auto &e : g.edge)
			{
				if (e.first < g.adj.size())
					g.adj[e.first].push_back(e.second);
			}
			return g;
		}
	};

	class SpectralFeatureExtractor
	{
	public:
		Matrix extract(const Graph &g, std::size_t dim) const
		{
			Matrix out(g.node.size(), dim);
			if (g.node.empty() || dim == 0)
				return out;

			std::vector<double> degree(g.node.size(), 0.0);
			for (const auto &e : g.edge)
			{
				if (e.first < degree.size())
					degree[e.first] += 1.0;
				if (e.second < degree.size())
					degree[e.second] += 1.0;
			}

			for (std::size_t i = 0; i < g.node.size(); ++i)
			{
				std::uint64_t h = hashNode(g.node[i]);
				double deg = std::log1p(degree[i]);
				for (std::size_t j = 0; j < dim; ++j)
				{
					double base = static_cast<double>((h + 1315423911ull * (j + 1)) % 100003) / 100003.0;
					double trig = std::sin((base + deg * 0.017) * 6.28318530718);
					double poly = std::pow(base, 2.0) - 0.5 * base + 0.1 * deg;
					out.at(i, j) = std::tanh(0.7 * trig + 0.3 * poly);
				}
			}

			smoothByNeighbors(out, g);
			return out;
		}

	private:
		static std::uint64_t hashNode(const std::string &s)
		{
			std::uint64_t h = 1469598103934665603ull;
			for (unsigned char c : s)
			{
				h ^= c;
				h *= 1099511628211ull;
			}
			return h;
		}

		static void smoothByNeighbors(Matrix &m, const Graph &g)
		{
			if (m.row == 0 || m.col == 0 || g.adj.empty())
				return;
			Matrix tmp = m;
			for (std::size_t i = 0; i < m.row; ++i)
			{
				if (i >= g.adj.size() || g.adj[i].empty())
					continue;
				for (std::size_t c = 0; c < m.col; ++c)
				{
					double acc = 0.0;
					int cnt = 0;
					for (auto v : g.adj[i])
					{
						if (v >= m.row)
							continue;
						acc += m.at(v, c);
						cnt++;
					}
					if (cnt > 0)
						tmp.at(i, c) = 0.8 * m.at(i, c) + 0.2 * (acc / cnt);
				}
			}
			m = std::move(tmp);
		}
	};

	class SemanticGraphAnalyzer
	{
	public:
		struct Summary
		{
			double lexicalDiversity{0.0};
			double edgeEntropy{0.0};
			double reciprocity{0.0};
			double avgClustering{0.0};
			double randomWalkCoverage{0.0};
			double giantComponentRatio{0.0};
			double pagerankSkew{0.0};
			std::vector<std::pair<std::string, double>> topKeywords;
		};

		Summary analyze(const Graph &g) const
		{
			Summary s;
			if (g.node.empty())
				return s;

			s.lexicalDiversity = lexicalDiversity(g);
			s.edgeEntropy = edgeEntropy(g);
			s.reciprocity = reciprocity(g);
			s.avgClustering = avgClustering(g);
			s.randomWalkCoverage = randomWalkCoverage(g, 8, 64);
			s.giantComponentRatio = giantComponentRatio(g);

			auto pr = pagerank(g, 24, 0.85);
			s.pagerankSkew = gini(pr);
			s.topKeywords = topByScore(g, pr, 16);
			return s;
		}

	private:
		static double lexicalDiversity(const Graph &g)
		{
			if (g.node.empty())
				return 0.0;
			std::unordered_set<std::string> unique;
			unique.reserve(g.node.size());
			std::size_t chars = 0;
			for (const auto &n : g.node)
			{
				unique.insert(n);
				chars += n.size();
			}
			double uniqRatio = static_cast<double>(unique.size()) / std::max<std::size_t>(1, g.node.size());
			double avgLen = static_cast<double>(chars) / std::max<std::size_t>(1, g.node.size());
			return std::tanh(0.7 * uniqRatio + 0.3 * (avgLen / 12.0));
		}

		static double edgeEntropy(const Graph &g)
		{
			if (g.edge.empty() || g.node.empty())
				return 0.0;
			std::vector<double> out(g.node.size(), 0.0);
			for (const auto &e : g.edge)
			{
				if (e.first < out.size())
					out[e.first] += 1.0;
			}
			double total = static_cast<double>(g.edge.size());
			double h = 0.0;
			for (double x : out)
			{
				if (x <= 0.0)
					continue;
				double p = x / total;
				h -= p * std::log(std::max(1e-12, p));
			}
			return h / std::log(std::max<double>(2.0, static_cast<double>(g.node.size())));
		}

		static double reciprocity(const Graph &g)
		{
			if (g.edge.empty())
				return 0.0;
			std::unordered_set<std::uint64_t> e;
			e.reserve(g.edge.size() * 2 + 1);
			auto key = [](std::size_t u, std::size_t v)
			{
				return (static_cast<std::uint64_t>(u) << 32) ^ static_cast<std::uint64_t>(v);
			};
			for (const auto &ed : g.edge)
				e.insert(key(ed.first, ed.second));

			std::size_t mutual = 0;
			for (const auto &ed : g.edge)
				if (e.count(key(ed.second, ed.first)))
					mutual++;
			return static_cast<double>(mutual) / std::max<std::size_t>(1, g.edge.size());
		}

		static double avgClustering(const Graph &g)
		{
			if (g.node.empty())
				return 0.0;
			std::vector<std::unordered_set<std::size_t>> nbr(g.node.size());
			for (const auto &ed : g.edge)
			{
				if (ed.first < nbr.size() && ed.second < nbr.size())
				{
					nbr[ed.first].insert(ed.second);
					nbr[ed.second].insert(ed.first);
				}
			}

			double total = 0.0;
			std::size_t valid = 0;
			for (std::size_t i = 0; i < nbr.size(); ++i)
			{
				const auto &s = nbr[i];
				if (s.size() < 2)
					continue;
				std::vector<std::size_t> list(s.begin(), s.end());
				std::size_t tri = 0;
				std::size_t poss = list.size() * (list.size() - 1) / 2;
				for (std::size_t a = 0; a < list.size(); ++a)
				{
					for (std::size_t b = a + 1; b < list.size(); ++b)
					{
						if (nbr[list[a]].count(list[b]))
							tri++;
					}
				}
				total += static_cast<double>(tri) / std::max<std::size_t>(1, poss);
				valid++;
			}
			return valid ? (total / valid) : 0.0;
		}

		static double randomWalkCoverage(const Graph &g, int walks, int steps)
		{
			if (g.node.empty())
				return 0.0;
			std::mt19937 rng(20260307u);
			std::uniform_int_distribution<std::size_t> pick(0, g.node.size() - 1);
			std::unordered_set<std::size_t> seen;
			seen.reserve(g.node.size());

			for (int w = 0; w < walks; ++w)
			{
				std::size_t cur = pick(rng);
				seen.insert(cur);
				for (int s = 0; s < steps; ++s)
				{
					if (cur >= g.adj.size() || g.adj[cur].empty())
						break;
					std::uniform_int_distribution<std::size_t> nxt(0, g.adj[cur].size() - 1);
					cur = g.adj[cur][nxt(rng)];
					if (cur < g.node.size())
						seen.insert(cur);
				}
			}
			return static_cast<double>(seen.size()) / std::max<std::size_t>(1, g.node.size());
		}

		static double giantComponentRatio(const Graph &g)
		{
			if (g.node.empty())
				return 0.0;
			std::vector<std::vector<std::size_t>> und(g.node.size());
			for (const auto &e : g.edge)
			{
				if (e.first >= g.node.size() || e.second >= g.node.size())
					continue;
				und[e.first].push_back(e.second);
				und[e.second].push_back(e.first);
			}
			std::vector<char> vis(g.node.size(), 0);
			std::size_t best = 0;
			for (std::size_t i = 0; i < g.node.size(); ++i)
			{
				if (vis[i])
					continue;
				std::queue<std::size_t> q;
				q.push(i);
				vis[i] = 1;
				std::size_t cnt = 0;
				while (!q.empty())
				{
					auto u = q.front();
					q.pop();
					cnt++;
					for (auto v : und[u])
					{
						if (!vis[v])
						{
							vis[v] = 1;
							q.push(v);
						}
					}
				}
				best = std::max(best, cnt);
			}
			return static_cast<double>(best) / std::max<std::size_t>(1, g.node.size());
		}

		static std::vector<double> pagerank(const Graph &g, int iters, double d)
		{
			std::vector<double> rank(g.node.size(), g.node.empty() ? 0.0 : 1.0 / g.node.size());
			if (g.node.empty())
				return rank;
			std::vector<double> outDeg(g.node.size(), 0.0);
			for (const auto &ed : g.edge)
				if (ed.first < outDeg.size())
					outDeg[ed.first] += 1.0;

			for (int t = 0; t < iters; ++t)
			{
				std::vector<double> next(g.node.size(), (1.0 - d) / g.node.size());
				for (const auto &ed : g.edge)
				{
					if (ed.first >= rank.size() || ed.second >= rank.size())
						continue;
					double denom = std::max(1.0, outDeg[ed.first]);
					next[ed.second] += d * rank[ed.first] / denom;
				}
				rank.swap(next);
			}
			return rank;
		}

		static double gini(std::vector<double> x)
		{
			if (x.empty())
				return 0.0;
			std::sort(x.begin(), x.end());
			double sum = std::accumulate(x.begin(), x.end(), 0.0);
			if (sum <= 1e-12)
				return 0.0;
			double acc = 0.0;
			for (std::size_t i = 0; i < x.size(); ++i)
				acc += (i + 1) * x[i];
			return (2.0 * acc) / (x.size() * sum) - (x.size() + 1.0) / x.size();
		}

		static std::vector<std::pair<std::string, double>> topByScore(const Graph &g, const std::vector<double> &score, std::size_t k)
		{
			std::vector<std::pair<std::string, double>> out;
			for (std::size_t i = 0; i < g.node.size() && i < score.size(); ++i)
				out.push_back({g.node[i], score[i]});
			std::sort(out.begin(), out.end(), [](const auto &a, const auto &b)
				  { return a.second > b.second; });
			if (out.size() > k)
				out.resize(k);
			return out;
		}
	};

	struct ObjectivePoint
	{
		double fitness{0.0};
		double robustness{0.0};
		double latency{0.0};
	};

	class SubgraphSampler
	{
	public:
		explicit SubgraphSampler(std::uint32_t seed) : rng_(seed) {}

		std::vector<std::vector<std::size_t>> sample(const Graph &g, std::size_t round, std::size_t width)
		{
			std::vector<std::vector<std::size_t>> out;
			if (g.node.empty())
				return out;
			out.reserve(round);
			std::uniform_int_distribution<std::size_t> pick(0, g.node.size() - 1);

			for (std::size_t r = 0; r < round; ++r)
			{
				std::vector<std::size_t> idx;
				idx.reserve(width);
				std::size_t start = pick(rng_);
				idx.push_back(start);
				std::size_t cur = start;
				for (std::size_t w = 1; w < width; ++w)
				{
					if (cur < g.adj.size() && !g.adj[cur].empty())
					{
						std::uniform_int_distribution<std::size_t> nextPick(0, g.adj[cur].size() - 1);
						cur = g.adj[cur][nextPick(rng_)];
					}
					else
					{
						cur = pick(rng_);
					}
					idx.push_back(cur);
				}
				out.push_back(std::move(idx));
			}
			return out;
		}

	private:
		std::mt19937 rng_;
	};

	class MultiObjectiveScorer
	{
	public:
		ObjectivePoint score(const Matrix &feature,
				     const std::vector<std::size_t> &subgraph,
				     double residualWeight,
				     double mutationRate,
				     double mutationScale) const
		{
			if (feature.row == 0 || feature.col == 0 || subgraph.empty())
				return {};

			double energy = subgraphEnergy(feature, subgraph);
			double smooth = neighborhoodSmoothness(feature, subgraph);
			double variance = featureVariance(feature, subgraph);

			double fit = std::tanh(0.6 * energy + 0.4 * smooth + 0.1 * residualWeight);
			double robust = std::tanh(0.7 * smooth + 0.3 * (1.0 - mutationRate) + 0.1 * (1.0 - variance));
			double latency = 1.0 / (1.0 + mutationScale + 0.25 * subgraph.size() + 0.1 * variance);
			return {fit, robust, latency};
		}

	private:
		static double subgraphEnergy(const Matrix &feature, const std::vector<std::size_t> &subgraph)
		{
			double acc = 0.0;
			std::size_t cnt = 0;
			for (auto idx : subgraph)
			{
				if (idx >= feature.row)
					continue;
				for (std::size_t c = 0; c < feature.col; ++c)
				{
					acc += std::abs(feature.at(idx, c));
					cnt++;
				}
			}
			return cnt ? (acc / cnt) : 0.0;
		}

		static double neighborhoodSmoothness(const Matrix &feature, const std::vector<std::size_t> &subgraph)
		{
			if (subgraph.size() < 2)
				return 0.0;
			double acc = 0.0;
			std::size_t cnt = 0;
			for (std::size_t i = 1; i < subgraph.size(); ++i)
			{
				auto a = subgraph[i - 1];
				auto b = subgraph[i];
				if (a >= feature.row || b >= feature.row)
					continue;
				double d = 0.0;
				for (std::size_t c = 0; c < feature.col; ++c)
				{
					double diff = feature.at(a, c) - feature.at(b, c);
					d += diff * diff;
				}
				acc += 1.0 / (1.0 + std::sqrt(d));
				cnt++;
			}
			return cnt ? (acc / cnt) : 0.0;
		}

		static double featureVariance(const Matrix &feature, const std::vector<std::size_t> &subgraph)
		{
			if (subgraph.empty())
				return 0.0;
			std::vector<double> mean(feature.col, 0.0);
			std::size_t valid = 0;
			for (auto idx : subgraph)
			{
				if (idx >= feature.row)
					continue;
				valid++;
				for (std::size_t c = 0; c < feature.col; ++c)
					mean[c] += feature.at(idx, c);
			}
			if (!valid)
				return 0.0;
			for (double &m : mean)
				m /= valid;

			double var = 0.0;
			std::size_t cnt = 0;
			for (auto idx : subgraph)
			{
				if (idx >= feature.row)
					continue;
				for (std::size_t c = 0; c < feature.col; ++c)
				{
					double d = feature.at(idx, c) - mean[c];
					var += d * d;
					cnt++;
				}
			}
			return cnt ? (var / cnt) : 0.0;
		}
	};

	class ParetoArchive
	{
	public:
		void push(const ObjectivePoint &p)
		{
			points_.push_back(p);
			prune();
		}

		const std::vector<ObjectivePoint> &points() const { return points_; }

	private:
		static bool dominate(const ObjectivePoint &a, const ObjectivePoint &b)
		{
			bool noWorse = a.fitness >= b.fitness && a.robustness >= b.robustness && a.latency >= b.latency;
			bool strict = a.fitness > b.fitness || a.robustness > b.robustness || a.latency > b.latency;
			return noWorse && strict;
		}

		void prune()
		{
			std::vector<ObjectivePoint> keep;
			keep.reserve(points_.size());
			for (std::size_t i = 0; i < points_.size(); ++i)
			{
				bool dominated = false;
				for (std::size_t j = 0; j < points_.size(); ++j)
				{
					if (i == j)
						continue;
					if (dominate(points_[j], points_[i]))
					{
						dominated = true;
						break;
					}
				}
				if (!dominated)
					keep.push_back(points_[i]);
			}
			points_.swap(keep);
			if (points_.size() > 1024)
				points_.resize(1024);
		}

		std::vector<ObjectivePoint> points_;
	};

	class CrowdingDistance
	{
	public:
		std::vector<double> compute(const std::vector<ObjectivePoint> &pts) const
		{
			if (pts.empty())
				return {};
			std::vector<double> d(pts.size(), 0.0);
			accumulateAxis(pts, d, [](const ObjectivePoint &p)
				       { return p.fitness; });
			accumulateAxis(pts, d, [](const ObjectivePoint &p)
				       { return p.robustness; });
			accumulateAxis(pts, d, [](const ObjectivePoint &p)
				       { return p.latency; });
			return d;
		}

	private:
		template <typename Getter>
		static void accumulateAxis(const std::vector<ObjectivePoint> &pts, std::vector<double> &d, Getter getter)
		{
			std::vector<std::size_t> idx(pts.size());
			std::iota(idx.begin(), idx.end(), 0);
			std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b)
				  { return getter(pts[a]) < getter(pts[b]); });

			d[idx.front()] = std::numeric_limits<double>::infinity();
			d[idx.back()] = std::numeric_limits<double>::infinity();

			double minV = getter(pts[idx.front()]);
			double maxV = getter(pts[idx.back()]);
			double span = std::max(1e-9, maxV - minV);
			for (std::size_t i = 1; i + 1 < idx.size(); ++i)
			{
				double prev = getter(pts[idx[i - 1]]);
				double next = getter(pts[idx[i + 1]]);
				if (std::isfinite(d[idx[i]]))
					d[idx[i]] += (next - prev) / span;
			}
		}
	};

	class ConstraintHandler
	{
	public:
		bool feasible(const ObjectivePoint &point, double residualWeight) const
		{
			return point.fitness >= 0.0 && point.robustness >= 0.0 && point.latency + residualWeight > 0.2;
		}

		double penalty(const ObjectivePoint &point, double residualWeight) const
		{
			double p = 0.0;
			if (point.fitness < 0.0)
				p += -point.fitness;
			if (point.robustness < 0.0)
				p += -point.robustness;
			if (point.latency + residualWeight <= 0.2)
				p += 0.2 - (point.latency + residualWeight);
			return p;
		}
	};

	class EvolutionController
	{
	public:
		std::vector<ObjectivePoint> run(const Matrix &feature,
						int generations,
						int population,
						double residualWeight,
						double mutationRate,
						double mutationScale)
		{
			std::vector<ObjectivePoint> out;
			int gens = std::clamp(generations, 1, 128);
			int pop = std::clamp(population, 8, 256);

			for (int g = 0; g < gens; ++g)
			{
				auto sub = sampler_.sample(graphStub_, static_cast<std::size_t>(pop), 8);
				for (const auto &sg : sub)
				{
					auto pt = scorer_.score(feature, sg, residualWeight, mutationRate, mutationScale);
					if (!constraint_.feasible(pt, residualWeight))
					{
						double p = constraint_.penalty(pt, residualWeight);
						pt.fitness -= p;
						pt.robustness -= 0.5 * p;
						pt.latency = std::max(0.0, pt.latency - 0.2 * p);
					}
					archive_.push(pt);
					out.push_back(pt);
				}
				adaptiveMutation(g, gens, mutationRate, mutationScale);
			}
			return out;
		}

		void setGraphStub(const Graph &g) { graphStub_ = g; }
		const ParetoArchive &archive() const { return archive_; }

	private:
		void adaptiveMutation(int gen, int total, double baseRate, double baseScale)
		{
			double progress = static_cast<double>(gen + 1) / std::max(1, total);
			mutationRate_ = std::clamp(baseRate * (1.0 - 0.4 * progress), 0.0, 1.0);
			mutationScale_ = std::clamp(baseScale * (1.0 - 0.3 * progress), 0.0, 2.0);
		}

		SubgraphSampler sampler_{20260307u};
		MultiObjectiveScorer scorer_;
		ConstraintHandler constraint_;
		ParetoArchive archive_;
		Graph graphStub_;
		double mutationRate_{0.1};
		double mutationScale_{0.1};
	};

	struct NormalizedAdjacency
	{
		std::size_t nodeCount{0};
		std::vector<std::vector<std::pair<std::size_t, double>>> out;
		std::vector<std::vector<std::pair<std::size_t, double>>> in;
		std::vector<double> outDegree;
		std::vector<double> inDegree;
	};

	class AdjacencyNormalizer
	{
	public:
		NormalizedAdjacency build(const Graph &g, double selfLoop, double hubPenalty) const
		{
			NormalizedAdjacency norm;
			norm.nodeCount = g.node.size();
			norm.out.assign(norm.nodeCount, {});
			norm.in.assign(norm.nodeCount, {});
			norm.outDegree.assign(norm.nodeCount, 0.0);
			norm.inDegree.assign(norm.nodeCount, 0.0);
			if (norm.nodeCount == 0)
				return norm;

			auto rawOut = buildRawOut(g);
			auto degree = computeOutDegree(rawOut);
			auto inDegree = computeInDegree(rawOut, norm.nodeCount);
			auto damp = makeHubDamping(degree, hubPenalty);

			for (std::size_t u = 0; u < norm.nodeCount; ++u)
			{
				double sum = 0.0;
				for (const auto &e : rawOut[u])
				{
					if (e.first >= norm.nodeCount)
						continue;
					double w = e.second * damp[u];
					sum += w;
				}
				if (selfLoop > 0.0)
					sum += selfLoop;
				if (sum <= 0.0)
					sum = 1.0;

				for (const auto &e : rawOut[u])
				{
					if (e.first >= norm.nodeCount)
						continue;
					double w = (e.second * damp[u]) / sum;
					norm.out[u].push_back({e.first, clipProbability(w)});
					norm.in[e.first].push_back({u, clipProbability(w)});
					norm.outDegree[u] += w;
					norm.inDegree[e.first] += w;
				}

				if (selfLoop > 0.0)
				{
					double w = selfLoop / sum;
					norm.out[u].push_back({u, clipProbability(w)});
					norm.in[u].push_back({u, clipProbability(w)});
					norm.outDegree[u] += w;
					norm.inDegree[u] += w;
				}

				normalizeRow(norm.out[u]);
				normalizeRow(norm.in[u]);
			}

			rebalanceDegrees(norm.outDegree, norm.inDegree);
			return norm;
		}

	private:
		static std::vector<std::vector<std::pair<std::size_t, double>>> buildRawOut(const Graph &g)
		{
			std::vector<std::vector<std::pair<std::size_t, double>>> raw(g.node.size());
			for (const auto &ed : g.edge)
			{
				if (ed.first >= g.node.size() || ed.second >= g.node.size())
					continue;
				raw[ed.first].push_back({ed.second, 1.0});
			}
			mergeParallel(raw);
			return raw;
		}

		static std::vector<double> computeOutDegree(const std::vector<std::vector<std::pair<std::size_t, double>>> &raw)
		{
			std::vector<double> degree(raw.size(), 0.0);
			for (std::size_t u = 0; u < raw.size(); ++u)
			{
				double acc = 0.0;
				for (const auto &e : raw[u])
					acc += e.second;
				degree[u] = acc;
			}
			return degree;
		}

		static std::vector<double> computeInDegree(const std::vector<std::vector<std::pair<std::size_t, double>>> &raw,
							   std::size_t n)
		{
			std::vector<double> degree(n, 0.0);
			for (std::size_t u = 0; u < raw.size(); ++u)
			{
				for (const auto &e : raw[u])
				{
					if (e.first < n)
						degree[e.first] += e.second;
				}
			}
			return degree;
		}

		static std::vector<double> makeHubDamping(const std::vector<double> &degree, double hubPenalty)
		{
			std::vector<double> damp(degree.size(), 1.0);
			if (degree.empty())
				return damp;
			std::vector<double> copy = degree;
			std::sort(copy.begin(), copy.end());
			double pivot = copy[copy.size() / 2];
			pivot = std::max(1.0, pivot);
			double penalty = std::clamp(hubPenalty, 0.0, 0.95);
			for (std::size_t i = 0; i < degree.size(); ++i)
			{
				double ratio = degree[i] / pivot;
				double value = 1.0 / (1.0 + penalty * std::max(0.0, ratio - 1.0));
				damp[i] = std::clamp(value, 0.05, 1.0);
			}
			return damp;
		}

		static void mergeParallel(std::vector<std::vector<std::pair<std::size_t, double>>> &raw)
		{
			for (auto &row : raw)
			{
				if (row.empty())
					continue;
				std::sort(row.begin(), row.end(), [](const auto &a, const auto &b)
					  { return a.first < b.first; });
				std::vector<std::pair<std::size_t, double>> merged;
				merged.reserve(row.size());
				std::size_t cur = row.front().first;
				double acc = row.front().second;
				for (std::size_t i = 1; i < row.size(); ++i)
				{
					if (row[i].first == cur)
					{
						acc += row[i].second;
					}
					else
					{
						merged.push_back({cur, acc});
						cur = row[i].first;
						acc = row[i].second;
					}
				}
				merged.push_back({cur, acc});
				row.swap(merged);
			}
		}

		static void normalizeRow(std::vector<std::pair<std::size_t, double>> &row)
		{
			double sum = 0.0;
			for (const auto &e : row)
				sum += e.second;
			if (sum <= 0.0)
				return;
			for (auto &e : row)
				e.second = clipProbability(e.second / sum);
		}

		static double clipProbability(double value)
		{
			return std::clamp(value, 1e-9, 1.0);
		}

		static void rebalanceDegrees(std::vector<double> &outDegree, std::vector<double> &inDegree)
		{
			double outTotal = std::accumulate(outDegree.begin(), outDegree.end(), 0.0);
			double inTotal = std::accumulate(inDegree.begin(), inDegree.end(), 0.0);
			if (outTotal <= 0.0 || inTotal <= 0.0)
				return;
			double ratio = outTotal / inTotal;
			for (double &v : inDegree)
				v *= ratio;
		}
	};

	class LaplacianBuilder
	{
	public:
		Matrix buildSymmetricNormalized(const Graph &g, const NormalizedAdjacency &norm) const
		{
			std::size_t n = g.node.size();
			Matrix lap(n, n);
			if (n == 0)
				return lap;

			std::vector<double> degree = effectiveDegree(norm);
			for (std::size_t i = 0; i < n; ++i)
			{
				lap.at(i, i) = 1.0;
			}

			for (std::size_t u = 0; u < norm.out.size(); ++u)
			{
				if (u >= n)
					break;
				for (const auto &edge : norm.out[u])
				{
					std::size_t v = edge.first;
					if (v >= n)
						continue;
					double inv = 1.0 / std::sqrt(std::max(1e-9, degree[u] * degree[v]));
					double value = edge.second * inv;
					lap.at(u, v) -= value;
					if (u != v)
						lap.at(v, u) -= value;
				}
			}

			stabilizeDiagonal(lap);
			sparsifySmall(lap, 1e-8);
			return lap;
		}

	private:
		static std::vector<double> effectiveDegree(const NormalizedAdjacency &norm)
		{
			std::vector<double> degree(norm.nodeCount, 0.0);
			for (std::size_t i = 0; i < norm.nodeCount; ++i)
			{
				double out = i < norm.outDegree.size() ? norm.outDegree[i] : 0.0;
				double in = i < norm.inDegree.size() ? norm.inDegree[i] : 0.0;
				degree[i] = std::max(1e-9, 0.5 * (out + in));
			}
			return degree;
		}

		static void stabilizeDiagonal(Matrix &lap)
		{
			for (std::size_t i = 0; i < lap.row; ++i)
			{
				double off = 0.0;
				for (std::size_t j = 0; j < lap.col; ++j)
				{
					if (i == j)
						continue;
					off += std::abs(lap.at(i, j));
				}
				if (lap.at(i, i) < off * 0.5)
				{
					lap.at(i, i) = off * 0.5 + 1e-6;
				}
			}
		}

		static void sparsifySmall(Matrix &lap, double eps)
		{
			for (std::size_t i = 0; i < lap.row; ++i)
			{
				for (std::size_t j = 0; j < lap.col; ++j)
				{
					if (std::abs(lap.at(i, j)) < eps)
						lap.at(i, j) = 0.0;
				}
			}
		}
	};

	class PowerIterationSolver
	{
	public:
		std::pair<double, std::vector<double>> principalEigen(const Matrix &m,
								      int maxIter,
								      double tol,
								      std::uint32_t seed) const
		{
			if (m.row == 0 || m.col == 0 || m.row != m.col)
				return {0.0, {}};

			std::vector<double> vec = randomVector(m.row, seed);
			normalize(vec);

			double lambda = 0.0;
			for (int i = 0; i < std::max(1, maxIter); ++i)
			{
				auto nxt = multiply(m, vec);
				normalize(nxt);
				double nextLambda = rayleigh(m, nxt);
				if (std::abs(nextLambda - lambda) < tol)
				{
					lambda = nextLambda;
					vec = std::move(nxt);
					break;
				}
				lambda = nextLambda;
				vec = std::move(nxt);
			}
			return {lambda, vec};
		}

		std::vector<std::pair<double, std::vector<double>>> topKEigenApprox(const Matrix &m,
										    std::size_t k,
										    int maxIter,
										    double tol,
										    std::uint32_t seed) const
		{
			std::vector<std::pair<double, std::vector<double>>> out;
			if (m.row == 0 || m.col == 0 || m.row != m.col || k == 0)
				return out;
			Matrix residual = m;
			for (std::size_t i = 0; i < k; ++i)
			{
				auto eig = principalEigen(residual, maxIter, tol, seed + static_cast<std::uint32_t>(i * 17));
				if (eig.second.empty())
					break;
				out.push_back(eig);
				deflate(residual, eig.first, eig.second);
			}
			return out;
		}

		double spectralRadiusBound(const Matrix &m) const
		{
			if (m.row == 0 || m.col == 0)
				return 0.0;
			double bound = 0.0;
			for (std::size_t i = 0; i < m.row; ++i)
			{
				double rowSum = 0.0;
				for (std::size_t j = 0; j < m.col; ++j)
					rowSum += std::abs(m.at(i, j));
				bound = std::max(bound, rowSum);
			}
			return bound;
		}

	private:
		static std::vector<double> randomVector(std::size_t n, std::uint32_t seed)
		{
			std::vector<double> vec(n, 0.0);
			std::mt19937 rng(seed);
			std::uniform_real_distribution<double> dist(-1.0, 1.0);
			for (double &v : vec)
				v = dist(rng);
			return vec;
		}

		static std::vector<double> multiply(const Matrix &m, const std::vector<double> &v)
		{
			std::vector<double> out(m.row, 0.0);
			for (std::size_t i = 0; i < m.row; ++i)
			{
				double acc = 0.0;
				for (std::size_t j = 0; j < m.col; ++j)
				{
					acc += m.at(i, j) * v[j];
				}
				out[i] = acc;
			}
			return out;
		}

		static void normalize(std::vector<double> &vec)
		{
			double sq = 0.0;
			for (double v : vec)
				sq += v * v;
			double norm = std::sqrt(std::max(1e-18, sq));
			for (double &v : vec)
				v /= norm;
		}

		static double rayleigh(const Matrix &m, const std::vector<double> &v)
		{
			auto mv = multiply(m, v);
			double num = 0.0;
			double den = 0.0;
			for (std::size_t i = 0; i < v.size(); ++i)
			{
				num += v[i] * mv[i];
				den += v[i] * v[i];
			}
			if (den <= 1e-18)
				return 0.0;
			return num / den;
		}

		static void deflate(Matrix &m, double lambda, const std::vector<double> &vec)
		{
			for (std::size_t i = 0; i < m.row; ++i)
			{
				for (std::size_t j = 0; j < m.col; ++j)
				{
					m.at(i, j) -= lambda * vec[i] * vec[j];
				}
			}
		}
	};

	class RandomWalkEmbedding
	{
	public:
		Matrix embed(const NormalizedAdjacency &norm,
			     std::size_t dim,
			     std::size_t walkLen,
			     std::size_t walksPerNode,
			     std::uint32_t seed) const
		{
			Matrix emb(norm.nodeCount, dim);
			if (norm.nodeCount == 0 || dim == 0 || norm.out.empty())
				return emb;

			std::mt19937 rng(seed);
			for (std::size_t node = 0; node < norm.nodeCount; ++node)
			{
				for (std::size_t w = 0; w < walksPerNode; ++w)
				{
					std::vector<std::size_t> path;
					path.reserve(walkLen + 1);
					sampleWalk(norm, node, walkLen, rng, path);
					accumulatePath(path, emb, node);
				}
			}

			smoothEmbedding(emb, norm);
			normalizeRows(emb);
			return emb;
		}

	private:
		static void sampleWalk(const NormalizedAdjacency &norm,
				       std::size_t start,
				       std::size_t walkLen,
				       std::mt19937 &rng,
				       std::vector<std::size_t> &path)
		{
			std::size_t cur = start;
			path.push_back(cur);
			for (std::size_t step = 0; step < walkLen; ++step)
			{
				if (cur >= norm.out.size() || norm.out[cur].empty())
					break;
				cur = drawNext(norm.out[cur], rng);
				path.push_back(cur);
			}
		}

		static std::size_t drawNext(const std::vector<std::pair<std::size_t, double>> &row, std::mt19937 &rng)
		{
			std::uniform_real_distribution<double> pick(0.0, 1.0);
			double p = pick(rng);
			double acc = 0.0;
			for (const auto &e : row)
			{
				acc += e.second;
				if (p <= acc)
					return e.first;
			}
			return row.back().first;
		}

		static void accumulatePath(const std::vector<std::size_t> &path, Matrix &emb, std::size_t anchor)
		{
			if (path.empty() || anchor >= emb.row)
				return;
			for (std::size_t i = 0; i < path.size(); ++i)
			{
				std::size_t node = path[i];
				if (node >= emb.row)
					continue;
				for (std::size_t c = 0; c < emb.col; ++c)
				{
					double freq = harmonicWeight(i + 1);
					double phase = static_cast<double>((anchor + 1) * (c + 3) + (node + 5)) * 0.003;
					emb.at(anchor, c) += freq * std::sin(phase);
					emb.at(node, c) += 0.5 * freq * std::cos(phase * 0.8);
				}
			}
		}

		static double harmonicWeight(std::size_t rank)
		{
			return 1.0 / std::max(1.0, static_cast<double>(rank));
		}

		static void smoothEmbedding(Matrix &emb, const NormalizedAdjacency &norm)
		{
			if (emb.row == 0 || emb.col == 0)
				return;
			Matrix tmp = emb;
			for (std::size_t u = 0; u < emb.row; ++u)
			{
				if (u >= norm.out.size() || norm.out[u].empty())
					continue;
				for (std::size_t c = 0; c < emb.col; ++c)
				{
					double acc = 0.0;
					for (const auto &e : norm.out[u])
					{
						if (e.first >= emb.row)
							continue;
						acc += e.second * emb.at(e.first, c);
					}
					tmp.at(u, c) = 0.7 * emb.at(u, c) + 0.3 * acc;
				}
			}
			emb = std::move(tmp);
		}

		static void normalizeRows(Matrix &emb)
		{
			for (std::size_t r = 0; r < emb.row; ++r)
			{
				double sq = 0.0;
				for (std::size_t c = 0; c < emb.col; ++c)
					sq += emb.at(r, c) * emb.at(r, c);
				double norm = std::sqrt(std::max(1e-18, sq));
				for (std::size_t c = 0; c < emb.col; ++c)
					emb.at(r, c) /= norm;
			}
		}
	};

	class CommunityDetector
	{
	public:
		struct Result
		{
			std::vector<int> label;
			int groups{0};
			double modularity{0.0};
		};

		Result detect(const Graph &g, const NormalizedAdjacency &norm, int maxRound) const
		{
			Result result;
			result.label.assign(g.node.size(), -1);
			if (g.node.empty())
				return result;

			initLabels(result.label);
			propagate(g, norm, result.label, std::max(2, maxRound));
			compressLabels(result.label, result.groups);
			result.modularity = estimateModularity(norm, result.label, result.groups);
			return result;
		}

	private:
		static void initLabels(std::vector<int> &label)
		{
			for (std::size_t i = 0; i < label.size(); ++i)
				label[i] = static_cast<int>(i);
		}

		static void propagate(const Graph &g,
				      const NormalizedAdjacency &norm,
				      std::vector<int> &label,
				      int round)
		{
			std::vector<int> order(label.size());
			std::iota(order.begin(), order.end(), 0);
			std::mt19937 rng(20260228u);
			for (int r = 0; r < round; ++r)
			{
				std::shuffle(order.begin(), order.end(), rng);
				bool changed = false;
				for (int idx : order)
				{
					std::size_t u = static_cast<std::size_t>(idx);
					auto best = bestNeighborLabel(g, norm, u, label);
					if (best >= 0 && best != label[u])
					{
						label[u] = best;
						changed = true;
					}
				}
				if (!changed)
					break;
			}
		}

		static int bestNeighborLabel(const Graph &g,
					     const NormalizedAdjacency &norm,
					     std::size_t u,
					     const std::vector<int> &label)
		{
			std::unordered_map<int, double> score;
			score.reserve(16);

			if (u < norm.out.size())
			{
				for (const auto &e : norm.out[u])
				{
					if (e.first >= label.size())
						continue;
					score[label[e.first]] += e.second;
				}
			}
			if (u < norm.in.size())
			{
				for (const auto &e : norm.in[u])
				{
					if (e.first >= label.size())
						continue;
					score[label[e.first]] += 0.7 * e.second;
				}
			}
			if (u < g.adj.size())
			{
				for (auto v : g.adj[u])
				{
					if (v >= label.size())
						continue;
					score[label[v]] += 0.2;
				}
			}

			int bestLabel = -1;
			double bestScore = -1.0;
			for (const auto &kv : score)
			{
				if (kv.second > bestScore)
				{
					bestScore = kv.second;
					bestLabel = kv.first;
				}
			}
			return bestLabel;
		}

		static void compressLabels(std::vector<int> &label, int &groups)
		{
			std::unordered_map<int, int> id;
			id.reserve(label.size());
			int next = 0;
			for (int &v : label)
			{
				auto it = id.find(v);
				if (it == id.end())
				{
					id.emplace(v, next);
					v = next;
					next++;
				}
				else
				{
					v = it->second;
				}
			}
			groups = next;
		}

		static double estimateModularity(const NormalizedAdjacency &norm,
						 const std::vector<int> &label,
						 int groups)
		{
			if (groups <= 0 || norm.nodeCount == 0)
				return 0.0;
			std::vector<double> in(groups, 0.0);
			std::vector<double> out(groups, 0.0);

			for (std::size_t u = 0; u < norm.out.size(); ++u)
			{
				int lu = label[u];
				for (const auto &e : norm.out[u])
				{
					if (e.first >= label.size())
						continue;
					int lv = label[e.first];
					if (lu == lv)
						in[lu] += e.second;
					out[lu] += e.second;
				}
			}

			double m = std::accumulate(out.begin(), out.end(), 0.0);
			if (m <= 1e-18)
				return 0.0;

			double q = 0.0;
			for (int c = 0; c < groups; ++c)
			{
				double ec = in[c] / m;
				double ac = out[c] / m;
				q += ec - ac * ac;
			}
			return q;
		}
	};

	class MotifCounter
	{
	public:
		struct MotifStats
		{
			std::size_t triangle{0};
			std::size_t wedge{0};
			std::size_t reciprocal{0};
			std::size_t feedForward{0};
			std::size_t cycle{0};
			double density{0.0};
			double transitivity{0.0};
		};

		MotifStats count(const Graph &g, const NormalizedAdjacency &norm) const
		{
			MotifStats stats;
			if (g.node.empty())
				return stats;

			auto undirected = makeUndirectedSet(g);
			stats.triangle = countTriangles(undirected);
			stats.wedge = countWedges(undirected);
			stats.reciprocal = countReciprocal(g);
			stats.feedForward = countFeedForward(g);
			stats.cycle = countThreeCycle(g);
			stats.density = edgeDensity(g);
			stats.transitivity = computeTransitivity(stats.triangle, stats.wedge);
			adjustByNormalization(stats, norm);
			return stats;
		}

	private:
		static std::vector<std::unordered_set<std::size_t>> makeUndirectedSet(const Graph &g)
		{
			std::vector<std::unordered_set<std::size_t>> nbr(g.node.size());
			for (const auto &e : g.edge)
			{
				if (e.first >= g.node.size() || e.second >= g.node.size())
					continue;
				nbr[e.first].insert(e.second);
				nbr[e.second].insert(e.first);
			}
			return nbr;
		}

		static std::size_t countTriangles(const std::vector<std::unordered_set<std::size_t>> &nbr)
		{
			std::size_t tri = 0;
			for (std::size_t u = 0; u < nbr.size(); ++u)
			{
				for (auto v : nbr[u])
				{
					if (v <= u)
						continue;
					for (auto w : nbr[v])
					{
						if (w <= v)
							continue;
						if (nbr[u].count(w))
							tri++;
					}
				}
			}
			return tri;
		}

		static std::size_t countWedges(const std::vector<std::unordered_set<std::size_t>> &nbr)
		{
			std::size_t wedge = 0;
			for (const auto &s : nbr)
			{
				if (s.size() < 2)
					continue;
				wedge += s.size() * (s.size() - 1) / 2;
			}
			return wedge;
		}

		static std::size_t countReciprocal(const Graph &g)
		{
			std::unordered_set<std::uint64_t> set;
			auto key = [](std::size_t u, std::size_t v)
			{
				return (static_cast<std::uint64_t>(u) << 32) ^ static_cast<std::uint64_t>(v);
			};
			set.reserve(g.edge.size() * 2 + 1);
			for (const auto &e : g.edge)
				set.insert(key(e.first, e.second));

			std::size_t count = 0;
			for (const auto &e : g.edge)
			{
				if (set.count(key(e.second, e.first)))
					count++;
			}
			return count / 2;
		}

		static std::size_t countFeedForward(const Graph &g)
		{
			std::vector<std::unordered_set<std::size_t>> out(g.node.size());
			for (const auto &e : g.edge)
			{
				if (e.first < out.size() && e.second < out.size())
					out[e.first].insert(e.second);
			}
			std::size_t total = 0;
			for (std::size_t a = 0; a < out.size(); ++a)
			{
				for (auto b : out[a])
				{
					for (auto c : out[b])
					{
						if (out[a].count(c))
							total++;
					}
				}
			}
			return total;
		}

		static std::size_t countThreeCycle(const Graph &g)
		{
			std::vector<std::unordered_set<std::size_t>> out(g.node.size());
			for (const auto &e : g.edge)
			{
				if (e.first < out.size() && e.second < out.size())
					out[e.first].insert(e.second);
			}
			std::size_t cyc = 0;
			for (std::size_t a = 0; a < out.size(); ++a)
			{
				for (auto b : out[a])
				{
					if (b == a)
						continue;
					for (auto c : out[b])
					{
						if (c == a || c == b)
							continue;
						if (out[c].count(a))
							cyc++;
					}
				}
			}
			return cyc / 3;
		}

		static double edgeDensity(const Graph &g)
		{
			if (g.node.size() < 2)
				return 0.0;
			double possible = static_cast<double>(g.node.size()) * static_cast<double>(g.node.size() - 1);
			return static_cast<double>(g.edge.size()) / std::max(1.0, possible);
		}

		static double computeTransitivity(std::size_t tri, std::size_t wedge)
		{
			if (wedge == 0)
				return 0.0;
			return 3.0 * static_cast<double>(tri) / static_cast<double>(wedge);
		}

		static void adjustByNormalization(MotifStats &stats, const NormalizedAdjacency &norm)
		{
			double meanOut = norm.nodeCount == 0 ? 0.0 : std::accumulate(norm.outDegree.begin(), norm.outDegree.end(), 0.0) / static_cast<double>(norm.nodeCount);
			double meanIn = norm.nodeCount == 0 ? 0.0 : std::accumulate(norm.inDegree.begin(), norm.inDegree.end(), 0.0) / static_cast<double>(norm.nodeCount);
			double correction = 0.5 * (meanOut + meanIn);
			if (correction > 1e-12)
			{
				stats.transitivity = std::tanh(stats.transitivity * correction);
				stats.density = std::tanh(stats.density * (0.5 + correction));
			}
		}
	};

	class NSGA2Ranker
	{
	public:
		struct Ranking
		{
			std::vector<int> rank;
			std::vector<double> crowding;
			std::vector<std::vector<std::size_t>> fronts;
		};

		Ranking rank(const std::vector<ObjectivePoint> &points) const
		{
			Ranking out;
			out.rank.assign(points.size(), 0);
			out.crowding.assign(points.size(), 0.0);
			if (points.empty())
				return out;

			nonDominatedSort(points, out.rank, out.fronts);
			for (const auto &front : out.fronts)
			{
				assignCrowding(points, front, out.crowding);
			}
			return out;
		}

	private:
		static bool dominates(const ObjectivePoint &a, const ObjectivePoint &b)
		{
			bool noWorse = a.fitness >= b.fitness && a.robustness >= b.robustness && a.latency >= b.latency;
			bool strict = a.fitness > b.fitness || a.robustness > b.robustness || a.latency > b.latency;
			return noWorse && strict;
		}

		static void nonDominatedSort(const std::vector<ObjectivePoint> &points,
					     std::vector<int> &rank,
					     std::vector<std::vector<std::size_t>> &fronts)
		{
			std::size_t n = points.size();
			std::vector<std::vector<std::size_t>> dom(n);
			std::vector<int> dominatedBy(n, 0);
			std::vector<std::size_t> first;

			for (std::size_t i = 0; i < n; ++i)
			{
				for (std::size_t j = 0; j < n; ++j)
				{
					if (i == j)
						continue;
					if (dominates(points[i], points[j]))
						dom[i].push_back(j);
					else if (dominates(points[j], points[i]))
						dominatedBy[i]++;
				}
				if (dominatedBy[i] == 0)
				{
					rank[i] = 0;
					first.push_back(i);
				}
			}

			fronts.push_back(first);
			int current = 0;
			while (current < static_cast<int>(fronts.size()) && !fronts[current].empty())
			{
				std::vector<std::size_t> next;
				for (auto p : fronts[current])
				{
					for (auto q : dom[p])
					{
						dominatedBy[q]--;
						if (dominatedBy[q] == 0)
						{
							rank[q] = current + 1;
							next.push_back(q);
						}
					}
				}
				if (!next.empty())
					fronts.push_back(std::move(next));
				current++;
			}
		}

		static void assignCrowding(const std::vector<ObjectivePoint> &points,
					   const std::vector<std::size_t> &front,
					   std::vector<double> &crowding)
		{
			if (front.empty())
				return;
			if (front.size() <= 2)
			{
				for (auto i : front)
					crowding[i] = std::numeric_limits<double>::infinity();
				return;
			}
			assignAxis(points, front, crowding, [](const ObjectivePoint &p)
				   { return p.fitness; });
			assignAxis(points, front, crowding, [](const ObjectivePoint &p)
				   { return p.robustness; });
			assignAxis(points, front, crowding, [](const ObjectivePoint &p)
				   { return p.latency; });
		}

		template <typename Getter>
		static void assignAxis(const std::vector<ObjectivePoint> &points,
				       const std::vector<std::size_t> &front,
				       std::vector<double> &crowding,
				       Getter getter)
		{
			std::vector<std::size_t> idx = front;
			std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b)
				  { return getter(points[a]) < getter(points[b]); });

			crowding[idx.front()] = std::numeric_limits<double>::infinity();
			crowding[idx.back()] = std::numeric_limits<double>::infinity();

			double minV = getter(points[idx.front()]);
			double maxV = getter(points[idx.back()]);
			double span = std::max(1e-12, maxV - minV);
			for (std::size_t i = 1; i + 1 < idx.size(); ++i)
			{
				if (!std::isfinite(crowding[idx[i]]))
					continue;
				double prev = getter(points[idx[i - 1]]);
				double next = getter(points[idx[i + 1]]);
				crowding[idx[i]] += (next - prev) / span;
			}
		}
	};

	class ParetoFrontier
	{
	public:
		struct FrontierSummary
		{
			std::vector<ObjectivePoint> frontier;
			double hyperVolume{0.0};
			double spread{0.0};
			double meanFitness{0.0};
		};

		FrontierSummary summarize(const std::vector<ObjectivePoint> &points, const std::vector<int> &rank) const
		{
			FrontierSummary s;
			if (points.empty() || rank.size() != points.size())
				return s;

			for (std::size_t i = 0; i < points.size(); ++i)
			{
				if (rank[i] == 0)
					s.frontier.push_back(points[i]);
			}
			if (s.frontier.empty())
				return s;

			s.hyperVolume = estimateHyperVolume(s.frontier);
			s.spread = diversitySpread(s.frontier);
			s.meanFitness = averageFitness(s.frontier);
			return s;
		}

	private:
		static double estimateHyperVolume(std::vector<ObjectivePoint> frontier)
		{
			if (frontier.empty())
				return 0.0;
			std::sort(frontier.begin(), frontier.end(), [](const ObjectivePoint &a, const ObjectivePoint &b)
				  { return a.fitness > b.fitness; });

			double refFit = -1.0;
			double refRob = -1.0;
			double refLat = -1.0;
			double volume = 0.0;
			for (const auto &p : frontier)
			{
				double dx = std::max(0.0, p.fitness - refFit);
				double dy = std::max(0.0, p.robustness - refRob);
				double dz = std::max(0.0, p.latency - refLat);
				volume += dx * dy * dz;
				refFit = std::max(refFit, p.fitness);
				refRob = std::max(refRob, p.robustness);
				refLat = std::max(refLat, p.latency);
			}
			return volume;
		}

		static double diversitySpread(const std::vector<ObjectivePoint> &frontier)
		{
			if (frontier.size() < 2)
				return 0.0;
			std::vector<double> gaps;
			gaps.reserve(frontier.size() - 1);

			auto sorted = frontier;
			std::sort(sorted.begin(), sorted.end(), [](const ObjectivePoint &a, const ObjectivePoint &b)
				  { return a.fitness < b.fitness; });
			for (std::size_t i = 1; i < sorted.size(); ++i)
			{
				double dFit = sorted[i].fitness - sorted[i - 1].fitness;
				double dRob = sorted[i].robustness - sorted[i - 1].robustness;
				double dLat = sorted[i].latency - sorted[i - 1].latency;
				gaps.push_back(std::sqrt(dFit * dFit + dRob * dRob + dLat * dLat));
			}
			double mean = std::accumulate(gaps.begin(), gaps.end(), 0.0) / gaps.size();
			double var = 0.0;
			for (double g : gaps)
			{
				double d = g - mean;
				var += d * d;
			}
			var /= gaps.size();
			return std::sqrt(var);
		}

		static double averageFitness(const std::vector<ObjectivePoint> &frontier)
		{
			double sum = 0.0;
			for (const auto &p : frontier)
				sum += p.fitness;
			return sum / std::max<std::size_t>(1, frontier.size());
		}
	};

	class SurrogateEvaluator
	{
	public:
		struct Model
		{
			std::vector<double> weight;
			double bias{0.0};
			double mse{0.0};
		};

		Model fit(const Matrix &feature, const std::vector<ObjectivePoint> &points) const
		{
			Model model;
			if (feature.row == 0 || feature.col == 0 || points.empty())
				return model;

			std::size_t n = std::min(feature.row, points.size());
			model.weight.assign(feature.col, 0.0);
			model.bias = 0.0;

			std::vector<double> target(n, 0.0);
			for (std::size_t i = 0; i < n; ++i)
			{
				target[i] = compositeTarget(points[i]);
			}

			trainRidge(feature, target, n, model.weight, model.bias);
			model.mse = evaluateMse(feature, target, n, model.weight, model.bias);
			return model;
		}

		std::vector<double> predict(const Matrix &feature, const Model &model) const
		{
			std::vector<double> out(feature.row, 0.0);
			if (feature.row == 0 || feature.col == 0 || model.weight.empty())
				return out;
			for (std::size_t i = 0; i < feature.row; ++i)
			{
				out[i] = model.bias;
				for (std::size_t c = 0; c < feature.col && c < model.weight.size(); ++c)
				{
					out[i] += model.weight[c] * feature.at(i, c);
				}
			}
			return out;
		}

	private:
		static double compositeTarget(const ObjectivePoint &p)
		{
			return 0.5 * p.fitness + 0.35 * p.robustness + 0.15 * p.latency;
		}

		static void trainRidge(const Matrix &feature,
				       const std::vector<double> &target,
				       std::size_t n,
				       std::vector<double> &weight,
				       double &bias)
		{
			double lr = 0.01;
			double reg = 1e-3;
			int epoch = 80;

			for (int e = 0; e < epoch; ++e)
			{
				std::vector<double> grad(weight.size(), 0.0);
				double gradBias = 0.0;
				for (std::size_t i = 0; i < n; ++i)
				{
					double pred = bias;
					for (std::size_t c = 0; c < weight.size(); ++c)
					{
						pred += weight[c] * feature.at(i, c);
					}
					double err = pred - target[i];
					gradBias += err;
					for (std::size_t c = 0; c < weight.size(); ++c)
					{
						grad[c] += err * feature.at(i, c) + reg * weight[c];
					}
				}
				double scale = 1.0 / std::max<std::size_t>(1, n);
				bias -= lr * gradBias * scale;
				for (std::size_t c = 0; c < weight.size(); ++c)
				{
					weight[c] -= lr * grad[c] * scale;
				}
				lr *= 0.985;
			}
		}

		static double evaluateMse(const Matrix &feature,
					  const std::vector<double> &target,
					  std::size_t n,
					  const std::vector<double> &weight,
					  double bias)
		{
			double mse = 0.0;
			for (std::size_t i = 0; i < n; ++i)
			{
				double pred = bias;
				for (std::size_t c = 0; c < weight.size(); ++c)
					pred += weight[c] * feature.at(i, c);
				double err = pred - target[i];
				mse += err * err;
			}
			return mse / std::max<std::size_t>(1, n);
		}
	};

	class ConstraintProjector
	{
	public:
		std::vector<ObjectivePoint> project(const std::vector<ObjectivePoint> &points,
						    double minFitness,
						    double minRobustness,
						    double minLatency,
						    double limitPenalty) const
		{
			std::vector<ObjectivePoint> out;
			out.reserve(points.size());
			for (const auto &p : points)
			{
				out.push_back(projectPoint(p, minFitness, minRobustness, minLatency, limitPenalty));
			}
			return out;
		}

	private:
		static ObjectivePoint projectPoint(const ObjectivePoint &p,
						   double minFitness,
						   double minRobustness,
						   double minLatency,
						   double limitPenalty)
		{
			ObjectivePoint q = p;
			double penalty = 0.0;
			if (q.fitness < minFitness)
			{
				penalty += (minFitness - q.fitness);
				q.fitness = minFitness;
			}
			if (q.robustness < minRobustness)
			{
				penalty += (minRobustness - q.robustness);
				q.robustness = minRobustness;
			}
			if (q.latency < minLatency)
			{
				penalty += (minLatency - q.latency);
				q.latency = minLatency;
			}

			double scale = std::clamp(limitPenalty, 0.0, 1.0);
			q.fitness -= scale * 0.25 * penalty;
			q.robustness -= scale * 0.25 * penalty;
			q.latency -= scale * 0.15 * penalty;
			q.fitness = std::clamp(q.fitness, -1.0, 1.0);
			q.robustness = std::clamp(q.robustness, -1.0, 1.0);
			q.latency = std::clamp(q.latency, 0.0, 1.0);
			return q;
		}
	};

	class AnnealingScheduler
	{
	public:
		struct Step
		{
			double temperature{1.0};
			double mutationRate{0.1};
			double mutationScale{0.1};
			double residualWeight{0.5};
		};

		std::vector<Step> buildSchedule(int generations,
						double mutationRate,
						double mutationScale,
						double residualWeight) const
		{
			int count = std::clamp(generations, 1, 256);
			std::vector<Step> schedule;
			schedule.reserve(count);

			double temp = 1.0;
			for (int g = 0; g < count; ++g)
			{
				double ratio = static_cast<double>(g) / std::max(1, count - 1);
				Step step;
				step.temperature = std::max(0.02, temp);
				step.mutationRate = std::clamp(mutationRate * (1.0 - 0.35 * ratio), 0.0, 1.0);
				step.mutationScale = std::clamp(mutationScale * (1.0 - 0.5 * ratio) + 0.02, 0.0, 2.0);
				step.residualWeight = std::clamp(residualWeight * (0.85 + 0.15 * (1.0 - ratio)), 0.0, 2.0);
				schedule.push_back(step);
				temp *= 0.97;
			}
			return schedule;
		}

		Step aggregate(const std::vector<Step> &schedule) const
		{
			Step agg;
			if (schedule.empty())
				return agg;
			for (const auto &s : schedule)
			{
				agg.temperature += s.temperature;
				agg.mutationRate += s.mutationRate;
				agg.mutationScale += s.mutationScale;
				agg.residualWeight += s.residualWeight;
			}
			double n = static_cast<double>(schedule.size()) + 1.0;
			agg.temperature /= n;
			agg.mutationRate /= n;
			agg.mutationScale /= n;
			agg.residualWeight /= n;
			return agg;
		}
	};

	class EvolutionPopulationManager
	{
	public:
		struct Result
		{
			std::vector<ObjectivePoint> refined;
			std::vector<int> rank;
			std::vector<double> crowding;
			ParetoFrontier::FrontierSummary frontier;
		};

		Result refine(const std::vector<ObjectivePoint> &raw,
			      const NSGA2Ranker::Ranking &ranking,
			      const std::vector<double> &surrogate,
			      const ConstraintProjector &projector,
			      double residualWeight) const
		{
			Result out;
			if (raw.empty())
				return out;

			out.refined = raw;
			injectSurrogateSignal(out.refined, surrogate);
			out.refined = projector.project(out.refined, 0.0, 0.0, 0.02, 0.5 + 0.2 * residualWeight);

			out.rank = ranking.rank;
			out.crowding = ranking.crowding;
			rebalanceByRankAndCrowding(out.refined, out.rank, out.crowding);
			out.frontier = frontier_.summarize(out.refined, out.rank);
			return out;
		}

		Matrix fuseFeature(const Matrix &base,
				   const Matrix &walkEmbedding,
				   const std::vector<std::pair<double, std::vector<double>>> &spectrum) const
		{
			std::size_t rows = base.row;
			std::size_t extra = walkEmbedding.col + spectrum.size();
			Matrix fused(rows, base.col + extra);
			if (rows == 0)
				return fused;

			for (std::size_t r = 0; r < rows; ++r)
			{
				for (std::size_t c = 0; c < base.col; ++c)
					fused.at(r, c) = base.at(r, c);
				for (std::size_t c = 0; c < walkEmbedding.col; ++c)
				{
					if (r < walkEmbedding.row)
						fused.at(r, base.col + c) = walkEmbedding.at(r, c);
				}
				for (std::size_t k = 0; k < spectrum.size(); ++k)
				{
					double proj = 0.0;
					if (!spectrum[k].second.empty() && r < spectrum[k].second.size())
					{
						proj = spectrum[k].second[r] * spectrum[k].first;
					}
					fused.at(r, base.col + walkEmbedding.col + k) = proj;
				}
			}
			standardize(fused);
			return fused;
		}

	private:
		static void injectSurrogateSignal(std::vector<ObjectivePoint> &points, const std::vector<double> &surrogate)
		{
			if (surrogate.empty())
				return;
			for (std::size_t i = 0; i < points.size(); ++i)
			{
				double s = surrogate[i % surrogate.size()];
				points[i].fitness = std::tanh(points[i].fitness + 0.1 * s);
				points[i].robustness = std::tanh(points[i].robustness + 0.05 * s);
				points[i].latency = std::clamp(points[i].latency + 0.02 * s, 0.0, 1.0);
			}
		}

		static void rebalanceByRankAndCrowding(std::vector<ObjectivePoint> &points,
						       const std::vector<int> &rank,
						       const std::vector<double> &crowding)
		{
			std::size_t n = std::min(points.size(), std::min(rank.size(), crowding.size()));
			for (std::size_t i = 0; i < n; ++i)
			{
				double rankPenalty = 1.0 / (1.0 + static_cast<double>(rank[i]));
				double crowdBoost = std::isfinite(crowding[i]) ? std::tanh(0.2 * crowding[i]) : 1.0;
				points[i].fitness = std::tanh(points[i].fitness * rankPenalty + 0.15 * crowdBoost);
				points[i].robustness = std::tanh(points[i].robustness * (0.7 + 0.3 * rankPenalty));
				points[i].latency = std::clamp(points[i].latency * (0.85 + 0.15 * crowdBoost), 0.0, 1.0);
			}
		}

		static void standardize(Matrix &m)
		{
			if (m.row == 0 || m.col == 0)
				return;
			std::vector<double> mean(m.col, 0.0);
			for (std::size_t r = 0; r < m.row; ++r)
			{
				for (std::size_t c = 0; c < m.col; ++c)
					mean[c] += m.at(r, c);
			}
			for (double &v : mean)
				v /= std::max<std::size_t>(1, m.row);

			std::vector<double> var(m.col, 0.0);
			for (std::size_t r = 0; r < m.row; ++r)
			{
				for (std::size_t c = 0; c < m.col; ++c)
				{
					double d = m.at(r, c) - mean[c];
					var[c] += d * d;
				}
			}
			for (double &v : var)
				v = std::sqrt(v / std::max<std::size_t>(1, m.row) + 1e-8);

			for (std::size_t r = 0; r < m.row; ++r)
			{
				for (std::size_t c = 0; c < m.col; ++c)
				{
					m.at(r, c) = (m.at(r, c) - mean[c]) / var[c];
				}
			}
		}

		ParetoFrontier frontier_;
	};

	class DiagnosticsReporter
	{
	public:
		json build(const Graph &graph,
			   const NormalizedAdjacency &norm,
			   const CommunityDetector::Result &community,
			   const MotifCounter::MotifStats &motif,
			   const SurrogateEvaluator::Model &surrogate,
			   const AnnealingScheduler::Step &anneal,
			   const ParetoFrontier::FrontierSummary &frontier,
			   double spectralRadius,
			   const SemanticGraphAnalyzer::Summary &semantic) const
		{
			json j;
			j["graph"] = graphBlock(graph, norm);
			j["community"] = communityBlock(community);
			j["motif"] = motifBlock(motif);
			j["surrogate"] = surrogateBlock(surrogate);
			j["annealing"] = annealingBlock(anneal);
			j["frontier"] = frontierBlock(frontier);
			j["spectral"] = json{{"radiusBound", spectralRadius}};
			j["semantic"] = semanticBlock(semantic);
			j["quality"] = qualityBlock(community, motif, frontier, surrogate, semantic);
			return j;
		}

	private:
		static json graphBlock(const Graph &graph, const NormalizedAdjacency &norm)
		{
			double avgOut = norm.nodeCount == 0 ? 0.0 : std::accumulate(norm.outDegree.begin(), norm.outDegree.end(), 0.0) / std::max<std::size_t>(1, norm.nodeCount);
			double avgIn = norm.nodeCount == 0 ? 0.0 : std::accumulate(norm.inDegree.begin(), norm.inDegree.end(), 0.0) / std::max<std::size_t>(1, norm.nodeCount);
			return json{{"nodes", graph.node.size()},
				    {"edges", graph.edge.size()},
				    {"avgOutDegree", avgOut},
				    {"avgInDegree", avgIn}};
		}

		static json communityBlock(const CommunityDetector::Result &community)
		{
			std::unordered_map<int, int> size;
			for (int label : community.label)
				size[label]++;
			std::vector<std::pair<int, int>> groups(size.begin(), size.end());
			std::sort(groups.begin(), groups.end(), [](const auto &a, const auto &b)
				  { return a.second > b.second; });
			json top = json::array();
			for (std::size_t i = 0; i < groups.size() && i < 12; ++i)
			{
				top.push_back(json{{"id", groups[i].first}, {"size", groups[i].second}});
			}
			return json{{"groups", community.groups}, {"modularity", community.modularity}, {"topGroups", top}};
		}

		static json motifBlock(const MotifCounter::MotifStats &motif)
		{
			return json{{"triangle", motif.triangle},
				    {"wedge", motif.wedge},
				    {"reciprocal", motif.reciprocal},
				    {"feedForward", motif.feedForward},
				    {"cycle", motif.cycle},
				    {"density", motif.density},
				    {"transitivity", motif.transitivity}};
		}

		static json surrogateBlock(const SurrogateEvaluator::Model &surrogate)
		{
			json weightStat = json::object();
			if (surrogate.weight.empty())
			{
				weightStat["l1"] = 0.0;
				weightStat["l2"] = 0.0;
			}
			else
			{
				double l1 = 0.0;
				double l2 = 0.0;
				for (double w : surrogate.weight)
				{
					l1 += std::abs(w);
					l2 += w * w;
				}
				weightStat["l1"] = l1;
				weightStat["l2"] = std::sqrt(l2);
			}
			return json{{"bias", surrogate.bias}, {"mse", surrogate.mse}, {"weight", weightStat}};
		}

		static json annealingBlock(const AnnealingScheduler::Step &anneal)
		{
			return json{{"temperature", anneal.temperature},
				    {"mutationRate", anneal.mutationRate},
				    {"mutationScale", anneal.mutationScale},
				    {"residualWeight", anneal.residualWeight}};
		}

		static json frontierBlock(const ParetoFrontier::FrontierSummary &frontier)
		{
			return json{{"size", frontier.frontier.size()},
				    {"hyperVolume", frontier.hyperVolume},
				    {"spread", frontier.spread},
				    {"meanFitness", frontier.meanFitness}};
		}

		static json semanticBlock(const SemanticGraphAnalyzer::Summary &semantic)
		{
			return json{{"lexicalDiversity", semantic.lexicalDiversity},
				    {"edgeEntropy", semantic.edgeEntropy},
				    {"reciprocity", semantic.reciprocity},
				    {"avgClustering", semantic.avgClustering},
				    {"randomWalkCoverage", semantic.randomWalkCoverage},
				    {"giantComponentRatio", semantic.giantComponentRatio},
				    {"pagerankSkew", semantic.pagerankSkew}};
		}

		static json qualityBlock(const CommunityDetector::Result &community,
					 const MotifCounter::MotifStats &motif,
					 const ParetoFrontier::FrontierSummary &frontier,
					 const SurrogateEvaluator::Model &surrogate,
					 const SemanticGraphAnalyzer::Summary &semantic)
		{
			double diversity = std::tanh(0.3 * static_cast<double>(community.groups) + 0.7 * semantic.lexicalDiversity);
			double structure = std::tanh(0.5 * motif.transitivity + 0.5 * std::max(0.0, community.modularity));
			double optimization = std::tanh(frontier.hyperVolume + 0.2 * frontier.meanFitness);
			double reliability = std::exp(-std::max(0.0, surrogate.mse));
			double overall = 0.28 * diversity + 0.26 * structure + 0.26 * optimization + 0.20 * reliability;
			return json{{"diversity", diversity},
				    {"structure", structure},
				    {"optimization", optimization},
				    {"reliability", reliability},
				    {"overall", overall}};
		}
	};

	class AdvancedGnnGaLearner final : public IGnnGaLearner
	{
	public:
		AdvancedGnnGaLearner(std::shared_ptr<ControllerPoolBase> pool, fs::path testsDir)
		    : pool_(std::move(pool)), testsDir_(std::move(testsDir))
		{
			loadCorpus();
		}

		json evolve(int generations,
			    int population,
			    const std::vector<std::string> &samples,
			    double residualWeight,
			    double mutationRate,
			    double mutationScale) override
		{
			std::lock_guard<std::mutex> lock(mu_);
			mergeSamples(samples);
			if (docs_.empty())
				loadCorpus();
			if (docs_.empty())
			{
				latest_ = json{{"ok", false}, {"error", "no-corpus"}, {"ts", gnnTick()}};
				return latest_;
			}

			Graph graph = builder_.fromDocuments(docs_, 4000, 50000);
			if (graph.node.empty())
			{
				latest_ = json{{"ok", false}, {"error", "empty-graph"}, {"ts", gnnTick()}};
				return latest_;
			}

			auto feature = extractor_.extract(graph, featureDim_);
			auto semantic = analyzer_.analyze(graph);

			controller_.setGraphStub(graph);
			auto points = controller_.run(feature, generations, population, residualWeight, mutationRate, mutationScale);
			auto crowd = crowding_.compute(controller_.archive().points());

			json genInfo = json::array();
			for (std::size_t i = 0; i < points.size() && i < 256; ++i)
			{
				genInfo.push_back(json{{"fitness", points[i].fitness}, {"robustness", points[i].robustness}, {"latency", points[i].latency}});
			}

			json topKeywords = json::array();
			for (const auto &kv : semantic.topKeywords)
			{
				topKeywords.push_back(json{{"token", kv.first}, {"score", kv.second}});
			}

			latest_ = json{{"ok", true},
				       {"nodes", graph.node.size()},
				       {"edges", graph.edge.size()},
				       {"generations", std::clamp(generations, 1, 128)},
				       {"population", std::clamp(population, 8, 256)},
				       {"archive", controller_.archive().points().size()},
				       {"crowding", crowd},
				       {"genInfo", genInfo},
				       {"semantic", json{{"lexicalDiversity", semantic.lexicalDiversity},
							 {"edgeEntropy", semantic.edgeEntropy},
							 {"reciprocity", semantic.reciprocity},
							 {"avgClustering", semantic.avgClustering},
							 {"randomWalkCoverage", semantic.randomWalkCoverage},
							 {"giantComponentRatio", semantic.giantComponentRatio},
							 {"pagerankSkew", semantic.pagerankSkew},
							 {"topKeywords", topKeywords}}},
				       {"ts", gnnTick()}};

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
		void loadCorpus()
		{
			docs_.clear();
			if (!fs::exists(testsDir_))
				return;
			std::vector<fs::path> files;
			for (const auto &entry : fs::directory_iterator(testsDir_))
			{
				if (!entry.is_regular_file())
					continue;
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
				files.resize(4096);

			for (const auto &p : files)
			{
				std::ifstream in(p, std::ios::binary);
				if (!in)
					continue;
				std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
				if (!text.empty())
					docs_.push_back(std::move(text));
			}
		}

		void mergeSamples(const std::vector<std::string> &samples)
		{
			for (const auto &s : samples)
			{
				if (!s.empty())
					docs_.push_back(s);
			}
			if (docs_.size() > 5000)
			{
				docs_.erase(docs_.begin(), docs_.begin() + static_cast<std::ptrdiff_t>(docs_.size() - 5000));
			}
		}

		std::shared_ptr<ControllerPoolBase> pool_;
		fs::path testsDir_;
		mutable std::mutex mu_;

		std::vector<std::string> docs_;
		std::vector<json> history_;
		json latest_;

		std::size_t maxHistory_{256};
		std::size_t featureDim_{96};

		GraphBuilder builder_;
		SpectralFeatureExtractor extractor_;
		SemanticGraphAnalyzer analyzer_;
		CrowdingDistance crowding_;
		EvolutionController controller_;
	};

	struct RegisterGnnGaFactory
	{
		RegisterGnnGaFactory()
		{
			module_mount::registerGnnGaLearnerFactory(
			    [](std::shared_ptr<ControllerPoolBase> pool, const std::filesystem::path &testsDir) -> std::shared_ptr<IGnnGaLearner>
			    {
				    return std::make_shared<AdvancedGnnGaLearner>(std::move(pool), testsDir);
			    });
		}
	};

	static RegisterGnnGaFactory registerGnnGaFactory;

} // namespace
