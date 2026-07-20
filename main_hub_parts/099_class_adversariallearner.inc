// Auto-generated from main.cpp
// Segment: class_AdversarialLearner
// Original lines: 12130-12294

class AdversarialLearner : public IAdversarialLearner {
public:
  explicit AdversarialLearner(std::shared_ptr<ControllerPoolBase> pool)
      : pool_(std::move(pool)), rng_(0x12345678u) {
    loadLearningConfig();
  }

  void loadLearningConfig() {
    maxAdversaries_ =
        phoenix::cfgOr<int>("learning.advMaxAdversaries", maxAdversaries_);
    noiseLevel_ =
        phoenix::cfgOr<double>("learning.advNoiseLevel", noiseLevel_);
    attackRounds_ =
        phoenix::cfgOr<int>("learning.advAttackRounds", attackRounds_);
    defenseRounds_ =
        phoenix::cfgOr<int>("learning.advDefenseRounds", defenseRounds_);
    benchLimit_ =
        phoenix::cfgOr<int>("learning.advBenchLimit", benchLimit_);
  }

  json attackAndDefend(const json &samples) override {
    auto bench = samples.is_array() ? samples : json::array();
    if ((int)bench.size() > benchLimit_)
      bench.erase(bench.begin() + benchLimit_, bench.end());
    json rounds = json::array();
    for (int r = 0; r < attackRounds_; r++) {
      json results = json::array();
      for (auto &s : bench) {
        auto advs = generateAdversaries(s.get<std::string>());
        for (auto &a : advs) {
          results.push_back(probe(a));
        }
      }
      rounds.push_back(json{{"phase", "attack"}, {"results", results}});
    }
    json fixes = json::array();
    for (int d = 0; d < defenseRounds_; d++) {
      for (auto &s : bench) {
        auto res = probe(s.get<std::string>());
        if (res.value("quality", 0.0) < 0.25) {
          auto words = tokenize(res.value("reply", ""));
          auto seeds = pool_->getActive()->runtime()->mapWordsToMemes(words);
          std::vector<std::string> ids;
          for (auto &kv : seeds)
            ids.push_back(kv.first);
          for (size_t i = 0; i + 1 < ids.size(); i++) {
            pool_->getActive()->runtime()->graphLink(ids[i], ids[i + 1], 0.5,
                                                     0);
          }
          fixes.push_back(
              json{{"sample", s},
                   {"addedEdges", (int)std::max<int>(0, (int)ids.size() - 1)}});
        }
      }
      rounds.push_back(json{{"phase", "defense"}, {"fixes", fixes}});
    }
    auto summary = summarize(rounds);
    latest_ = json{{"ok", true}, {"summary", summary}, {"rounds", rounds}};
    history_.push_back(latest_);
    return latest_;
  }

  json latest() const override {
    return history_.empty() ? json() : history_.back();
  }

private:
  std::shared_ptr<ControllerPoolBase> pool_;
  int maxAdversaries_{64};
  double noiseLevel_{0.2};
  int attackRounds_{3};
  int defenseRounds_{3};
  int benchLimit_{50};
  Rng32 rng_;
  std::vector<json> history_;
  json latest_ = json::object();

  std::vector<std::string>
  perturbTokens(const std::vector<std::string> &tokens) {
    std::vector<std::string> out;
    for (auto &t : tokens) {
      double r = rng_.next();
      if (r < noiseLevel_ / 2)
        continue;
      out.push_back(t);
    }
    if (rng_.next() < noiseLevel_)
      out.push_back("noise_token_" + std::to_string((int)(rng_.next() * 1000)));
    return out;
  }

  std::vector<std::string> generateAdversaries(const std::string &text) {
    auto base = tokenize(text);
    std::vector<std::string> adversaries;
    int times = std::min(8, std::max(2, (int)(base.size() / 3)));
    for (int i = 0; i < times; i++) {
      auto mut = perturbTokens(base);
      std::ostringstream oss;
      for (size_t j = 0; j < mut.size(); j++) {
        if (j)
          oss << " ";
        oss << mut[j];
      }
      adversaries.push_back(oss.str());
    }
    return adversaries;
  }

  json probe(const std::string &text) {
    auto ctrl = pool_->getActive();
    auto result = ctrl->respond(json{{"text", text}});
    std::unordered_map<std::string, double> seeds;
    for (auto &s : result.value("seeds", json::array())) {
      if (s.is_array() && s.size() >= 2)
        seeds[s[0].get<std::string>()] = s[1].get<double>();
    }
    double quality = qualityMetric(result.value("reply", ""), seeds);
    return json{{"text", text},
                {"reply", result.value("reply", "")},
                {"quality", quality},
                {"latency", result.value("latency", 0)},
                {"seeds", result.value("seeds", json::array())}};
  }

  double qualityMetric(const std::string &reply,
                       const std::unordered_map<std::string, double> &seeds) {
    auto base = tokenize(reply);
    if (base.empty())
      return 0.0;
    double linkedHit = linkedCoverage(base, seeds);
    double uniq =
        (double)std::unordered_set<std::string>(base.begin(), base.end())
            .size() /
        std::max(1.0, (double)base.size());
    double lenPenalty = base.size() > 50 ? 0.9 : 1.0;
    return (0.6 * linkedHit + 0.4 * uniq) * lenPenalty;
  }

  double linkedCoverage(const std::vector<std::string> &words,
                        const std::unordered_map<std::string, double> &seeds) {
    double coverage = 0.0;
    int count = 0;
    for (auto &kv : seeds) {
      auto linked = pool_->getActive()->runtime()->getMemeWords(kv.first);
      if (linked.empty())
        continue;
      std::unordered_set<std::string> linkedSet(linked.begin(), linked.end());
      int hit = 0;
      for (auto &w : words)
        if (linkedSet.count(w))
          hit++;
      coverage += (double)hit / std::max(1.0, (double)linked.size());
      count++;
    }
    return count ? coverage / count : 0.0;
  }

  json summarize(const json &rounds) {
    json stats{{"attack", json{{"n", 0}, {"avgQuality", 0.0}}},
               {"defense", json{{"fixes", 0}}}};
    double qsum = 0.0;
    int qcnt = 0;
    for (auto &r : rounds) {
      if (r.value("phase", "") == "attack") {
        auto results = r.value("results", json::array());
        stats["attack"]["n"] =
            stats["attack"]["n"].get<int>() + (int)results.size();
        for (auto &item : results) {
          qsum += item.value("quality", 0.0);
          qcnt++;
        }
      } else if (r.value("phase", "") == "defense") {
        stats["defense"]["fixes"] = stats["defense"]["fixes"].get<int>() +
                                    (int)r.value("fixes", json::array()).size();
      }
    }
    stats["attack"]["avgQuality"] = qcnt ? (qsum / qcnt) : 0.0;
    return stats;
  }
};
