// Auto-generated from main.cpp
// Segment: class_GnnGaLearner
// Original lines: 12296-12626

class GnnGaLearner : public IGnnGaLearner {
public:
  GnnGaLearner(std::shared_ptr<ControllerPoolBase> pool, fs::path testsDir)
      : pool_(std::move(pool)), testsDir_(std::move(testsDir)) {
    loadLearningConfig();
  }

  void loadLearningConfig() {
    maxDocs_ = phoenix::cfgOr<int>("learning.gnnGaMaxDocs", maxDocs_);
    population_ = phoenix::cfgOr<int>("learning.gnnGaPopulation", population_);
    generations_ =
        phoenix::cfgOr<int>("learning.gnnGaGenerations", generations_);
  }

  json evolve(int generations, int population,
              const std::vector<std::string> &samples, double residualWeight,
              double mutationRate, double mutationScale) override {
    generations = std::max(1, generations);
    population = std::max(4, population);
    residualWeight = std::max(0.0, std::min(1.0, residualWeight));
    mutationRate = std::max(0.0, std::min(1.0, mutationRate));
    mutationScale = std::max(0.05, std::min(0.75, mutationScale));

    auto texts = loadSamples(samples);
    if (texts.empty()) {
      latest_ = json{{"ok", false}, {"error", "no-samples"}};
      history_.push_back(latest_);
      return latest_;
    }

    auto ctrl = pool_->getActive();
    if (!ctrl) {
      latest_ = json{{"ok", false}, {"error", "controller-not-available"}};
      history_.push_back(latest_);
      return latest_;
    }
    auto baseParams = ctrl->runtime()->cloneParams();
    Candidate base = fromParams(baseParams);

    std::vector<Candidate> pop;
    pop.reserve(population);
    pop.push_back(base);
    while ((int)pop.size() < population) {
      pop.push_back(mutate(base, mutationRate, mutationScale));
    }

    json genStats = json::array();
    Candidate best = base;
    for (int g = 0; g < generations; g++) {
      for (auto &c : pop) {
        c = evaluate(c, texts, residualWeight);
      }
      std::sort(pop.begin(), pop.end(),
                [](const Candidate &a, const Candidate &b) {
                  return a.fitness > b.fitness;
                });
      if (g == 0 || pop.front().fitness > best.fitness)
        best = pop.front();

      double avgFit = 0.0;
      for (auto &c : pop)
        avgFit += c.fitness;
      avgFit = pop.empty() ? 0.0 : avgFit / pop.size();
      genStats.push_back(json{
          {"gen", g}, {"best", pop.front().toJson()}, {"avgFitness", avgFit}});

      int elite = std::max(2, population / 4);
      std::vector<Candidate> next;
      next.reserve(population);
      for (int i = 0; i < elite && i < (int)pop.size(); i++)
        next.push_back(pop[i]);
      while ((int)next.size() < population) {
        const Candidate &a = pop[randIndex(elite)];
        const Candidate &b = pop[randIndex(elite)];
        Candidate child = crossover(a, b);
        child = mutate(child, mutationRate, mutationScale);
        next.push_back(child);
      }
      pop.swap(next);
    }

    ctrl->runtime()->setParams(patchForCandidate(best));
    latest_ = json{
        {"ok", true},
        {"best", best.toJson()},
        {"generations", generations},
        {"population", population},
        {"residualWeight", residualWeight},
        {"mutationRate", mutationRate},
        {"mutationScale", mutationScale},
        {"samples", (int)texts.size()},
        {"genStats", genStats},
        {"ts",
         (int64_t)std::chrono::system_clock::now().time_since_epoch().count()}};
    history_.push_back(latest_);
    return latest_;
  }

  json latest() const override {
    return history_.empty() ? json() : history_.back();
  }

  int population() const override { return population_; }
  int generations() const override { return generations_; }

  json refreshTests(const fs::path &testsDir) {
    testsDir_ = testsDir;
    return json{{"ok", true}, {"testsDir", testsDir_.string()}};
  }

private:
  struct Candidate {
    int iteration{5};
    int maxLen{16};
    double decayK{1.0};
    double decayFactor{0.5};
    double threshold{3.0};
    double decay{1.0};
    double edgeWeight{1.0};
    double fitness{-1e9};
    double avgQuality{0.0};
    double avgResidual{0.0};

    json toJson() const {
      return json{{"iteration", iteration},   {"maxLen", maxLen},
                  {"decayK", decayK},         {"decayFactor", decayFactor},
                  {"threshold", threshold},   {"decay", decay},
                  {"edgeWeight", edgeWeight}, {"fitness", fitness},
                  {"avgQuality", avgQuality}, {"avgResidual", avgResidual}};
    }
  };

  std::shared_ptr<ControllerPoolBase> pool_;
  fs::path testsDir_;
  int maxDocs_{32};
  int population_{6};
  int generations_{1};
  std::mt19937 rng_{0xC0FFEEu};
  std::vector<json> history_;
  json latest_ = json::object();

  int randIndex(int upper) {
    if (upper <= 1)
      return 0;
    std::uniform_int_distribution<int> dist(0, upper - 1);
    return dist(rng_);
  }

  static int clampIntRange(int v, int lo, int hi) {
    if (v < lo)
      return lo;
    if (v > hi)
      return hi;
    return v;
  }

  static double clampDoubleRange(double v, double lo, double hi) {
    if (!std::isfinite(v))
      return (lo + hi) * 0.5;
    if (v < lo)
      return lo;
    if (v > hi)
      return hi;
    return v;
  }

  Candidate fromParams(const json &params) const {
    Candidate c;
    c.iteration = clampIntRange(params.value("iteration", 5), 3, 12);
    c.maxLen = clampIntRange(params.value("maxLen", 16), 8, 32);
    c.decayK = clampDoubleRange(params.value("decayK", 1.0), 0.5, 2.0);
    c.decayFactor =
        clampDoubleRange(params.value("decayFactor", 0.5), 0.2, 0.9);
    c.threshold = clampDoubleRange(params.value("threshold", 3.0), 1.0, 6.0);
    c.decay = clampDoubleRange(params.value("decay", 1.0), 0.5, 2.0);
    c.edgeWeight = clampDoubleRange(params.value("edgeWeight", 1.0), 0.5, 2.0);
    return c;
  }

  json patchForCandidate(const Candidate &c) const {
    return json{{"iteration", c.iteration},  {"maxLen", c.maxLen},
                {"decayK", c.decayK},        {"decayFactor", c.decayFactor},
                {"threshold", c.threshold},  {"decay", c.decay},
                {"edgeWeight", c.edgeWeight}};
  }

  Candidate crossover(const Candidate &a, const Candidate &b) {
    std::uniform_real_distribution<double> pick(0.0, 1.0);
    Candidate out;
    out.iteration = (pick(rng_) < 0.5) ? a.iteration : b.iteration;
    out.maxLen = (pick(rng_) < 0.5) ? a.maxLen : b.maxLen;
    out.decayK = (pick(rng_) < 0.5) ? a.decayK : b.decayK;
    out.decayFactor = (pick(rng_) < 0.5) ? a.decayFactor : b.decayFactor;
    out.threshold = (pick(rng_) < 0.5) ? a.threshold : b.threshold;
    out.decay = (pick(rng_) < 0.5) ? a.decay : b.decay;
    out.edgeWeight = (pick(rng_) < 0.5) ? a.edgeWeight : b.edgeWeight;
    return out;
  }

  Candidate mutate(Candidate c, double rate, double scale) {
    std::uniform_real_distribution<double> chance(0.0, 1.0);
    auto mutateDouble = [&](double v, double lo, double hi) {
      if (chance(rng_) >= rate)
        return v;
      double delta = (chance(rng_) * 2.0 - 1.0) * scale * (hi - lo);
      return clampDoubleRange(v + delta, lo, hi);
    };
    auto mutateInt = [&](int v, int lo, int hi) {
      if (chance(rng_) >= rate)
        return v;
      int span = std::max(1, hi - lo);
      int delta = (int)std::round(((chance(rng_) * 2.0 - 1.0) * scale * span));
      return clampIntRange(v + delta, lo, hi);
    };

    c.iteration = mutateInt(c.iteration, 3, 12);
    c.maxLen = mutateInt(c.maxLen, 8, 32);
    c.decayK = mutateDouble(c.decayK, 0.5, 2.0);
    c.decayFactor = mutateDouble(c.decayFactor, 0.2, 0.9);
    c.threshold = mutateDouble(c.threshold, 1.0, 6.0);
    c.decay = mutateDouble(c.decay, 0.5, 2.0);
    c.edgeWeight = mutateDouble(c.edgeWeight, 0.5, 2.0);
    return c;
  }

  Candidate evaluate(const Candidate &candidate,
                     const std::vector<std::string> &texts,
                     double residualWeight) {
    Candidate out = candidate;
    auto ctrl = pool_->getActive();
    if (!ctrl) {
      out.fitness = -1e9;
      return out;
    }
    auto runtime = ctrl->runtime();
    if (!runtime) {
      out.fitness = -1e9;
      return out;
    }

    json baseParams = runtime->cloneParams();
    runtime->setParams(patchForCandidate(candidate));
    double qsum = 0.0;
    double rsum = 0.0;
    int count = 0;
    try {
      for (const auto &text : texts) {
        auto resp = ctrl->respond(json{{"text", text}});
        std::unordered_map<std::string, double> seeds;
        for (auto &s : resp.value("seeds", json::array())) {
          if (s.is_array() && s.size() >= 2 && s[0].is_string())
            seeds[s[0].get<std::string>()] = s[1].get<double>();
        }
        std::string reply = resp.value("reply", "");
        qsum += scoreReplyQuality(reply, seeds);
        rsum += residualScore(text, reply);
        count++;
      }
    } catch (...) {
    }
    runtime->setParams(baseParams);

    out.avgQuality = count ? (qsum / count) : 0.0;
    out.avgResidual = count ? (rsum / count) : 1.0;
    out.fitness = out.avgQuality - residualWeight * out.avgResidual;
    return out;
  }

  double scoreReplyQuality(
      const std::string &reply,
      const std::unordered_map<std::string, double> &seeds) const {
    auto words = tokenize(reply);
    if (words.empty())
      return 0.0;
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
      coverage += (double)hit / std::max(1.0, (double)linkedSet.size());
      count++;
    }
    coverage = count ? coverage / count : 0.0;
    double uniq =
        (double)std::unordered_set<std::string>(words.begin(), words.end())
            .size() /
        std::max(1.0, (double)words.size());
    return 0.7 * coverage + 0.3 * uniq;
  }

  double residualScore(const std::string &question,
                       const std::string &reply) const {
    if (question.empty() || reply.empty())
      return 1.0;
    auto qEmb = textToMiniEmbedding(question, 128);
    auto rEmb = textToMiniEmbedding(reply, 128);
    double cos = cosineSim(qEmb, rEmb);
    double sim = clamp01((cos + 1.0) * 0.5);
    return 1.0 - sim;
  }

  std::vector<std::string>
  loadSamples(const std::vector<std::string> &samples) const {
    if (!samples.empty())
      return samples;
    auto files = listTestFiles();
    std::vector<std::string> texts;
    for (auto &file : files) {
      auto text = readText(file);
      if (!text.empty())
        texts.push_back(text);
    }
    return texts;
  }

  std::vector<std::string> listTestFiles() const {
    std::vector<std::string> files;
    if (!fs::exists(testsDir_))
      return files;
    for (auto &e : fs::directory_iterator(testsDir_)) {
      if (!e.is_regular_file())
        continue;
      auto ext = e.path().extension().string();
      std::string lower;
      for (char c : ext)
        lower.push_back((char)std::tolower((unsigned char)c));
      if (lower == ".txt")
        files.push_back(e.path().filename().string());
    }
    std::sort(files.begin(), files.end());
    if ((int)files.size() > maxDocs_)
      files.resize(maxDocs_);
    return files;
  }

  std::string readText(const std::string &file) const {
    return readTextFile(testsDir_ / file);
  }
};
