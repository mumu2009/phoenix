// Auto-generated from main.cpp
// Segment: class_ReinforcementLearner
// Original lines: 11775-12128

class ReinforcementLearner : public IReinforcementLearner {
public:
  ReinforcementLearner(std::shared_ptr<ControllerPoolBase> pool,
                       fs::path testsDir)
      : pool_(std::move(pool)), testsDir_(std::move(testsDir)) {}

  json learn(int cycles) override {
    (void)cycles;
    auto files = listTestFiles();
    std::vector<json> evals;
    for (auto &file : files) {
      evals.push_back(evaluateOnce(file));
    }
    std::vector<std::vector<float>> features;
    for (auto &e : evals) {
      features.push_back(buildFeatureForDoc(e.value("reply", "")));
    }
    auto coords = reduceFeatures(features);
    auto clusters = cluster(
        coords, std::min(4, std::max(2, (int)std::sqrt((double)files.size()))));
    std::unordered_map<int, std::vector<json>> grouped;
    for (size_t i = 0; i < evals.size(); i++)
      grouped[clusters[i]].push_back(evals[i]);
    json clusterStats = json::array();
    for (auto &kv : grouped) {
      double sum = 0;
      for (auto &e : kv.second)
        sum += e.value("score", 0.0);
      clusterStats.push_back(
          json{{"gid", kv.first},
               {"avg", kv.second.empty() ? 0.0 : sum / kv.second.size()},
               {"n", (int)kv.second.size()}});
    }
    double baseAvg = 0;
    for (auto &e : evals)
      baseAvg += e.value("score", 0.0);
    baseAvg = evals.empty() ? 0.0 : baseAvg / evals.size();
    double learnedAvg = 0;
    for (auto &c : clusterStats)
      learnedAvg += c.value("avg", 0.0);
    learnedAvg = clusterStats.empty() ? 0.0 : learnedAvg / clusterStats.size();
    json evalStats{{"baseAvg", baseAvg},
                   {"learnedAvg", learnedAvg},
                   {"clusters", clusterStats}};
    auto params = adjustParams(evalStats);
    latest_ = json{
        {"ok", true},
        {"evalStats", evalStats},
        {"params", params},
        {"evals", evals},
        {"ts",
         (int64_t)std::chrono::system_clock::now().time_since_epoch().count()}};
    history_.push_back(latest_);
    return latest_;
  }

  json latest() const override {
    return history_.empty() ? json() : history_.back();
  }

  json refreshTests(const fs::path &testsDir) override {
    testsDir_ = testsDir;
    return json{{"ok", true}, {"testsDir", testsDir_.string()}};
  }

  json setTestsDir(const fs::path &testsDir) override {
    return refreshTests(testsDir);
  }

private:
  std::shared_ptr<ControllerPoolBase> pool_;
  fs::path testsDir_;
  int maxDocs_{64};
  int topKWords_{30};
  double improvementThreshold_{0.01};
  int iterations_{5};
  bool useUmap_{true};
  std::vector<json> history_;
  json latest_ = json::object();

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

  std::vector<float> buildFeatureForDoc(const std::string &text) const {
    return textToMiniEmbedding(text, 128);
  }

  std::vector<std::array<double, 2>>
  reduceFeatures(const std::vector<std::vector<float>> &features) const {
    if (features.empty())
      return {};
    int nRows = (int)features.size();
    int nCols = (int)features[0].size();
    std::vector<double> mean(nCols, 0.0);
    for (auto &row : features)
      for (int j = 0; j < nCols; j++)
        mean[j] += row[j];
    for (int j = 0; j < nCols; j++)
      mean[j] /= std::max(1, nRows);
    std::vector<double> centered(nRows * nCols, 0.0);
    for (int i = 0; i < nRows; i++)
      for (int j = 0; j < nCols; j++)
        centered[i * nCols + j] = features[i][j] - mean[j];
    std::vector<std::vector<double>> cov(nCols,
                                         std::vector<double>(nCols, 0.0));
    for (int i = 0; i < nRows; i++) {
      for (int a = 0; a < nCols; a++) {
        for (int b = a; b < nCols; b++) {
          double v = centered[i * nCols + a] * centered[i * nCols + b];
          cov[a][b] += v;
          if (a != b)
            cov[b][a] += v;
        }
      }
    }
    auto powerIter = [&](std::vector<double> vec) {
      std::vector<double> next(nCols, 0.0);
      for (int i = 0; i < nCols; i++) {
        double acc = 0.0;
        for (int j = 0; j < nCols; j++)
          acc += cov[i][j] * vec[j];
        next[i] = acc;
      }
      double norm = 0.0;
      for (double v : next)
        norm += v * v;
      norm = std::sqrt(norm) ? std::sqrt(norm) : 1.0;
      for (double &v : next)
        v /= norm;
      return next;
    };
    std::vector<double> v1(nCols, 0.0), v2(nCols, 0.0);
    for (int j = 0; j < nCols; j++)
      v1[j] = ((double)std::rand() / RAND_MAX) - 0.5;
    for (int i = 0; i < 8; i++)
      v1 = powerIter(v1);
    for (int j = 0; j < nCols; j++)
      v2[j] = ((double)std::rand() / RAND_MAX) - 0.5;
    for (int i = 0; i < 8; i++) {
      double proj = 0;
      for (int j = 0; j < nCols; j++)
        proj += v2[j] * v1[j];
      for (int j = 0; j < nCols; j++)
        v2[j] -= proj * v1[j];
      v2 = powerIter(v2);
    }
    std::vector<std::array<double, 2>> coords;
    coords.reserve(nRows);
    for (int i = 0; i < nRows; i++) {
      double x = 0, y = 0;
      for (int j = 0; j < nCols; j++) {
        double v = centered[i * nCols + j];
        x += v * v1[j];
        y += v * v2[j];
      }
      coords.push_back({x, y});
    }
    return coords;
  }

  std::vector<int> cluster(const std::vector<std::array<double, 2>> &coords,
                           int k) const {
    auto out = clusterKMeans(coords, k);
    if (!out.empty())
      return out;
    if (coords.empty())
      return {};
    double minX = coords[0][0], maxX = coords[0][0];
    for (auto &c : coords) {
      minX = std::min(minX, c[0]);
      maxX = std::max(maxX, c[0]);
    }
    double step = (maxX - minX) / std::max(1, k);
    std::vector<int> fallback;
    fallback.reserve(coords.size());
    for (auto &c : coords) {
      int idx = (step <= 1e-6) ? 0 : (int)std::floor((c[0] - minX) / step);
      if (idx < 0)
        idx = 0;
      if (idx >= k)
        idx = k - 1;
      fallback.push_back(idx);
    }
    return fallback;
  }

  std::vector<int>
  clusterKMeans(const std::vector<std::array<double, 2>> &coords, int k) const {
    if (coords.empty() || k <= 0)
      return {};
    int n = (int)coords.size();
    k = std::min(k, n);
    std::mt19937 rng(0xC0FFEEu);
    std::uniform_int_distribution<int> pick(0, n - 1);

    std::vector<std::array<double, 2>> centroids;
    centroids.reserve(k);
    centroids.push_back(coords[pick(rng)]);
    for (int c = 1; c < k; c++) {
      std::vector<double> dist(n, 0.0);
      double total = 0.0;
      for (int i = 0; i < n; i++) {
        double best = std::numeric_limits<double>::max();
        for (auto &ct : centroids) {
          double dx = coords[i][0] - ct[0];
          double dy = coords[i][1] - ct[1];
          double d = dx * dx + dy * dy;
          if (d < best)
            best = d;
        }
        dist[i] = best;
        total += best;
      }
      if (total <= 1e-9) {
        centroids.push_back(coords[pick(rng)]);
        continue;
      }
      std::uniform_real_distribution<double> ur(0.0, total);
      double r = ur(rng);
      double acc = 0.0;
      int idx = 0;
      for (int i = 0; i < n; i++) {
        acc += dist[i];
        if (acc >= r) {
          idx = i;
          break;
        }
      }
      centroids.push_back(coords[idx]);
    }

    std::vector<int> assign(n, 0);
    for (int iter = 0; iter < 25; iter++) {
      bool changed = false;
      for (int i = 0; i < n; i++) {
        int bestK = 0;
        double best = std::numeric_limits<double>::max();
        for (int c = 0; c < k; c++) {
          double dx = coords[i][0] - centroids[c][0];
          double dy = coords[i][1] - centroids[c][1];
          double d = dx * dx + dy * dy;
          if (d < best) {
            best = d;
            bestK = c;
          }
        }
        if (assign[i] != bestK) {
          assign[i] = bestK;
          changed = true;
        }
      }
      std::vector<std::array<double, 2>> next(k, {0.0, 0.0});
      std::vector<int> counts(k, 0);
      for (int i = 0; i < n; i++) {
        next[assign[i]][0] += coords[i][0];
        next[assign[i]][1] += coords[i][1];
        counts[assign[i]]++;
      }
      for (int c = 0; c < k; c++) {
        if (counts[c] == 0) {
          next[c] = coords[pick(rng)];
        } else {
          next[c][0] /= counts[c];
          next[c][1] /= counts[c];
        }
      }
      centroids.swap(next);
      if (!changed)
        break;
    }
    return assign;
  }

  json adjustParams(const json &evalStats) {
    auto active = pool_->getActive();
    auto params = active->runtime()->cloneParams();
    double baseScore = evalStats.value("baseAvg", 0.0);
    double learnedScore = evalStats.value("learnedAvg", 0.0);
    if (learnedScore - baseScore >= improvementThreshold_) {
      params["iteration"] = std::min(12, params.value("iteration", 5) + 1);
      params["decayK"] = std::max(0.5, params.value("decayK", 1.0) - 0.05);
    } else {
      params["iteration"] = std::max(3, params.value("iteration", 5) - 1);
      params["decayK"] = std::min(2.0, params.value("decayK", 1.0) + 0.05);
    }
    active->applyParams(params);
    return params;
  }

  json evaluateOnce(const std::string &file) {
    std::string text = readText(file);
    auto ctrl = pool_->getActive();
    auto response = ctrl->respond(json{{"text", text}});
    std::unordered_map<std::string, double> seeds;
    for (auto &s : response.value("seeds", json::array())) {
      if (s.is_array() && s.size() >= 2)
        seeds[s[0].get<std::string>()] = s[1].get<double>();
    }
    double score = scoreReplyQuality(response.value("reply", ""), seeds);
    return json{{"file", file},
                {"score", score},
                {"words", tokenize(text)},
                {"reply", response.value("reply", "")}};
  }
};
