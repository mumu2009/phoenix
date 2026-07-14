// Auto-generated from main.cpp
// Segment: class_MemeBarrier
// Original lines: 8004-8831

class MemeBarrier {
public:
  struct Stats {
    int scans{0};
    int isolated{0};
    int64_t lastScanTime{0};
    std::vector<json> lastIsolated;
    int evaluated{0};
    int candidates{0};
    double avgScore{0.0};
    double maxScore{0.0};
    double p90Score{0.0};
    double lastThreshold{0.0};
    double avgDegree{0.0};
    double stdDegree{0.0};
    double avgOutDegree{0.0};
    double stdOutDegree{0.0};
    std::vector<int> scoreHistogram;
  };

  explicit MemeBarrier(std::shared_ptr<RuntimeState> runtime)
      : runtime_(std::move(runtime)) {}

  void start() {
    if (running_)
      return;
    running_ = true;
    worker_ = std::thread([this]() {
      while (running_) {
        try {
          scanNetwork();
        } catch (...) {
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(scanIntervalMs_));
      }
    });
  }

  void stop() {
    running_ = false;
    if (worker_.joinable())
      worker_.join();
  }

  bool running() const { return running_; }

  void setThreshold(double t) {
    baseThreshold_ = t;
    maliciousThreshold_ = t;
  }
  double threshold() const { return maliciousThreshold_; }

  void setDisableRateTarget(double rate) {
    if (!std::isfinite(rate))
      return;
    disableRateTarget_ = std::max(0.001, std::min(0.2, rate));
  }
  double disableRateTarget() const { return disableRateTarget_; }

  void setUseTextCNN(bool on) { useTextCNN_ = on; }
  void setUseRecommender(bool on) { useRecommender_ = on; }
  void setUseTorchModels(bool on) { useTorchModels_ = on; }
  bool useTextCNN() const { return useTextCNN_; }
  bool useRecommender() const { return useRecommender_; }
  bool useTorchModels() const { return useTorchModels_; }

  Stats getStats() const { return stats_; }

private:
  struct NodeStats {
    int degree{0};
    int outDegree{0};
    int selfLoops{0};
  };

  std::shared_ptr<RuntimeState> runtime_;
  int scanIntervalMs_{phoenix::cfg<int>("memebarrier.scanIntervalMs")};
  double maliciousThreshold_{phoenix::cfg<double>("memebarrier.maliciousThreshold")};
  double baseThreshold_{phoenix::cfg<double>("memebarrier.maliciousThreshold")};
  double minThreshold_{phoenix::cfg<double>("memebarrier.minThreshold")};
  double maxThreshold_{phoenix::cfg<double>("memebarrier.maxThreshold")};
  double thresholdMomentum_{phoenix::cfg<double>("memebarrier.thresholdMomentum")};
  double minIsolationMargin_{phoenix::cfg<double>("memebarrier.minIsolationMargin")};
  int minConsecutiveHits_{phoenix::cfg<int>("memebarrier.minConsecutiveHits")};
  bool adaptiveEnabled_{phoenix::cfg<bool>("memebarrier.adaptiveEnabled")};
  bool anomalyEnabled_{phoenix::cfg<bool>("memebarrier.anomalyEnabled")};
  int maxIsolatePerScan_{phoenix::cfg<int>("memebarrier.maxIsolatePerScan")};
  std::atomic<bool> running_{false};
  std::thread worker_;
  Stats stats_;
  std::deque<float> scoreWindow_;
  size_t scoreWindowSize_{phoenix::cfg<size_t>("memebarrier.scoreWindowSize")};
  std::unordered_map<std::string, int> hitCounts_;
  bool useTextCNN_{phoenix::cfg<bool>("memebarrier.useTextCNN")};
  bool useRecommender_{phoenix::cfg<bool>("memebarrier.useRecommender")};
  bool useTorchModels_{phoenix::cfg<bool>("memebarrier.useTorchModels")};
  double disableRateTarget_{phoenix::cfg<double>("memebarrier.disableRateTarget")};
  int recommenderTopics_{phoenix::cfg<int>("memebarrier.recommenderTopics")};
  int recommenderDim_{phoenix::cfg<int>("memebarrier.recommenderDim")};
  std::once_flag recommenderInit_;
  std::vector<std::vector<float>> topicVectors_;
  std::unordered_map<std::string, int> corpusFreq_;
  int corpusTotal_{0};
  int corpusMaxFreq_{1};

#ifdef HAVE_TORCH
  struct TorchVocab {
    int64_t padId{0};
    int64_t unkId{1};
    int64_t maxVocab{5000};
    std::unordered_map<std::string, int64_t> stoi;
    std::vector<std::string> itos;

    void build(const std::unordered_map<std::string, int> &freq) {
      std::vector<std::pair<std::string, int>> items(freq.begin(), freq.end());
      std::sort(items.begin(), items.end(),
                [](auto &a, auto &b) { return a.second > b.second; });
      itos.clear();
      stoi.clear();
      itos.push_back("<pad>");
      itos.push_back("<unk>");
      for (size_t i = 0; i < items.size() && (int64_t)itos.size() < maxVocab;
           i++) {
        stoi[items[i].first] = (int64_t)itos.size();
        itos.push_back(items[i].first);
      }
    }

    std::vector<int64_t> encode(const std::vector<std::string> &tokens,
                                int64_t maxLen) const {
      std::vector<int64_t> ids(maxLen, padId);
      int64_t idx = 0;
      for (const auto &t : tokens) {
        if (idx >= maxLen)
          break;
        auto it = stoi.find(t);
        ids[idx++] = (it == stoi.end()) ? unkId : it->second;
      }
      return ids;
    }
  };

  struct TextCNNImpl : torch::nn::Module {
    torch::nn::Embedding emb{nullptr};
    torch::nn::Conv1d conv3{nullptr};
    torch::nn::Conv1d conv4{nullptr};
    torch::nn::Conv1d conv5{nullptr};
    torch::nn::Linear fc{nullptr};

    TextCNNImpl(int64_t vocab, int64_t embDim, int64_t outDim)
        : emb(register_module("emb", torch::nn::Embedding(vocab, embDim))),
          conv3(register_module(
              "conv3",
              torch::nn::Conv1d(
                  torch::nn::Conv1dOptions(embDim, 64, 3).padding(1)))),
          conv4(register_module(
              "conv4",
              torch::nn::Conv1d(
                  torch::nn::Conv1dOptions(embDim, 64, 4).padding(2)))),
          conv5(register_module(
              "conv5",
              torch::nn::Conv1d(
                  torch::nn::Conv1dOptions(embDim, 64, 5).padding(2)))),
          fc(register_module("fc", torch::nn::Linear(64 * 3, outDim))) {}

    torch::Tensor forward(torch::Tensor x) {
      auto e = emb(x).transpose(1, 2);
      auto c3 = torch::relu(conv3(e));
      auto c4 = torch::relu(conv4(e));
      auto c5 = torch::relu(conv5(e));
      auto p3 = torch::adaptive_max_pool1d(c3, 1).squeeze(-1);
      auto p4 = torch::adaptive_max_pool1d(c4, 1).squeeze(-1);
      auto p5 = torch::adaptive_max_pool1d(c5, 1).squeeze(-1);
      auto cat = torch::cat({p3, p4, p5}, 1);
      auto out = fc(cat);
      return torch::sigmoid(out);
    }
  };
  TORCH_MODULE(TextCNN);

  struct TextRNNImpl : torch::nn::Module {
    torch::nn::Embedding emb{nullptr};
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear fc{nullptr};

    TextRNNImpl(int64_t vocab, int64_t embDim, int64_t hidDim, int64_t outDim)
        : emb(register_module("emb", torch::nn::Embedding(vocab, embDim))),
          lstm(register_module(
              "lstm",
              torch::nn::LSTM(
                  torch::nn::LSTMOptions(embDim, hidDim).batch_first(true)))),
          fc(register_module("fc", torch::nn::Linear(hidDim, outDim))) {}

    torch::Tensor forward(torch::Tensor x) {
      auto e = emb(x);
      auto out = std::get<0>(lstm(e));
      auto last = out.select(1, out.size(1) - 1);
      auto logits = fc(last);
      return torch::sigmoid(logits);
    }
  };
  TORCH_MODULE(TextRNN);

  struct TorchBundle {
    bool ready{false};
    TorchVocab vocab;
    int64_t maxLen{48};
    int64_t embDim{64};
    int64_t hidDim{64};
    TextCNN cnn{nullptr};
    TextRNN rnn{nullptr};
  };

  std::once_flag torchInit_;
  TorchBundle torch_;
#endif

  double evaluateMaliciousness(const std::string &memeId, const NodeStats &ns,
                               double avgConn, double avgOut, double stdDeg,
                               double stdOut) {
    if (!runtime_)
      return 0.0;
    double outSkew =
        avgConn > 0 ? std::min(1.0, ns.outDegree / (avgConn * 3.0)) : 0;
    double selfSkew =
        ns.degree > 0 ? std::min(1.0, (double)ns.selfLoops / ns.degree) : 0;
    double zDeg = (stdDeg > 1e-6) ? (ns.degree - avgConn) / stdDeg : 0.0;
    double zOut = (stdOut > 1e-6) ? (ns.outDegree - avgOut) / stdOut : 0.0;
    double anomaly =
        anomalyEnabled_
            ? clamp01((std::max(0.0, zDeg) + std::max(0.0, zOut)) / 6.0)
            : 0.0;
    double growth = 0; // 暂无 access log，对齐 JS 的可选逻辑
    double wGrowth =
        phoenix::cfg<double>("memebarrier.weights.growth");
    double wOutSkew =
        phoenix::cfg<double>("memebarrier.weights.outSkew");
    double wSelfSkew =
        phoenix::cfg<double>("memebarrier.weights.selfSkew");
    double wAnomaly =
        phoenix::cfg<double>("memebarrier.weights.anomaly");
    double score =
        wGrowth * growth + wOutSkew * outSkew + wSelfSkew * selfSkew + wAnomaly * anomaly;
    return clamp01(score);
  }

  static float hashToUnit(const std::string &s) {
    uint32_t h = 2166136261u;
    for (unsigned char c : s) {
      h ^= c;
      h *= 16777619u;
    }
    return (float)(h % 10000) / 10000.0f;
  }

  std::vector<float> wordVector(const std::string &word, int dim) const {
    std::vector<float> v(dim, 0.0f);
    std::seed_seq seed{
        static_cast<unsigned int>(std::hash<std::string>{}(word))};
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (int i = 0; i < dim; i++)
      v[i] = dist(rng);
    return v;
  }

  std::vector<float> embedTokens(const std::vector<std::string> &tokens,
                                 int dim) const {
    std::vector<float> out(dim, 0.0f);
    int count = 0;
    for (const auto &t : tokens) {
      if (t.empty())
        continue;
      auto v = wordVector(t, dim);
      for (int i = 0; i < dim; i++)
        out[i] += v[i];
      count++;
    }
    if (count > 0) {
      for (int i = 0; i < dim; i++)
        out[i] /= (float)count;
    }
    return out;
  }

  void ensureRecommenderReady() {
    if (!runtime_ || !useRecommender_)
      return;
    std::call_once(recommenderInit_, [&]() {
      try {
        auto docs = runtime_->collectRobotsDocuments(
            json{{"limit", 200}, {"shuffle", true}});
        for (const auto &doc : docs) {
          for (const auto &t : doc.tokens) {
            if (t.empty())
              continue;
            corpusFreq_[t] += 1;
            corpusTotal_ += 1;
            corpusMaxFreq_ = std::max(corpusMaxFreq_, corpusFreq_[t]);
          }
        }
        for (const auto &doc : docs) {
          if (doc.tokens.empty())
            continue;
          topicVectors_.push_back(embedTokens(doc.tokens, recommenderDim_));
          if ((int)topicVectors_.size() >= recommenderTopics_)
            break;
        }
      } catch (...) {
      }
    });
  }

#ifdef HAVE_TORCH
  void ensureTorchReady() {
    if (!runtime_ || !useTorchModels_)
      return;
    std::call_once(torchInit_, [&]() {
      try {
        ensureRecommenderReady();
        torch_.maxLen =
            static_cast<int64_t>(phoenix::cfg<int>("memebarrier.torch.maxLen"));
        torch_.embDim =
            static_cast<int64_t>(phoenix::cfg<int>("memebarrier.torch.embDim"));
        torch_.hidDim =
            static_cast<int64_t>(phoenix::cfg<int>("memebarrier.torch.hidDim"));
        auto docs = runtime_->collectRobotsDocuments(
            json{{"limit", 400}, {"shuffle", true}});
        torch_.vocab.build(corpusFreq_);
        int64_t vocabSize =
            std::max<int64_t>(2, (int64_t)torch_.vocab.itos.size());
        torch_.cnn = TextCNN(vocabSize, torch_.embDim, 1);
        torch_.rnn = TextRNN(vocabSize, torch_.embDim, torch_.hidDim, 1);
        torch_.cnn->train();
        torch_.rnn->train();

        std::vector<std::vector<int64_t>> xs;
        std::vector<float> ys;
        xs.reserve(docs.size());
        ys.reserve(docs.size());
        for (const auto &doc : docs) {
          if (doc.tokens.empty())
            continue;
          xs.push_back(torch_.vocab.encode(doc.tokens, torch_.maxLen));
          ys.push_back((float)computeCopyFeatures(doc.tokens));
          if (xs.size() >= 512)
            break;
        }
        if (xs.empty()) {
          torch_.ready = true;
          return;
        }

        auto xTensor =
            torch::zeros({(int64_t)xs.size(), torch_.maxLen}, torch::kInt64);
        for (size_t i = 0; i < xs.size(); i++) {
          for (int64_t j = 0; j < torch_.maxLen; j++) {
            xTensor.index_put_({(int64_t)i, j}, xs[i][(size_t)j]);
          }
        }
        auto yTensor = torch::from_blob(ys.data(), {(int64_t)ys.size(), 1},
                                        torch::kFloat32)
                           .clone();

        float adamLR =
            phoenix::cfg<float>("memebarrier.torch.adamLR");
        int epochs =
            phoenix::cfg<int>("memebarrier.torch.epochs");
        int batch =
            phoenix::cfg<int>("memebarrier.torch.batch");
        float lrDecay =
            phoenix::cfg<float>("memebarrier.torch.lrDecay");
        int patience =
            std::max(1, phoenix::cfg<int>("memebarrier.torch.patience"));
        float currentLR = adamLR;
        float bestLoss = std::numeric_limits<float>::infinity();
        int staleEpochs = 0;
        for (int e = 0; e < epochs; e++) {
          // Recreate optimizers with the current adaptive learning rate.
          torch::optim::Adam optCnn(torch_.cnn->parameters(),
                                    torch::optim::AdamOptions(currentLR));
          torch::optim::Adam optRnn(torch_.rnn->parameters(),
                                    torch::optim::AdamOptions(currentLR));
          float epochLoss = 0.0f;
          int batches = 0;
          for (int64_t i = 0; i < xTensor.size(0); i += batch) {
            int64_t end = std::min<int64_t>(xTensor.size(0), i + batch);
            auto xb = xTensor.slice(0, i, end);
            auto yb = yTensor.slice(0, i, end);

            optCnn.zero_grad();
            auto predC = torch_.cnn->forward(xb);
            auto lossC = torch::mse_loss(predC, yb);
            lossC.backward();
            optCnn.step();

            optRnn.zero_grad();
            auto predR = torch_.rnn->forward(xb);
            auto lossR = torch::mse_loss(predR, yb);
            lossR.backward();
            optRnn.step();

            epochLoss += lossC.item<float>() + lossR.item<float>();
            batches++;
          }
          if (batches > 0)
            epochLoss /= static_cast<float>(batches);
          if (epochLoss < bestLoss * 0.99f) {
            bestLoss = epochLoss;
            staleEpochs = 0;
          } else {
            staleEpochs++;
            if (staleEpochs >= patience) {
              currentLR *= lrDecay;
              staleEpochs = 0;
            }
          }
        }
        torch_.cnn->eval();
        torch_.rnn->eval();
        torch_.ready = true;
      } catch (...) {
      }
    });
  }

  double torchCnnScore(const std::vector<std::string> &tokens) {
    if (!useTorchModels_)
      return 0.0;
    ensureTorchReady();
    if (!torch_.ready || !torch_.cnn)
      return 0.0;
    auto ids = torch_.vocab.encode(tokens, torch_.maxLen);
    auto x = torch::zeros({1, torch_.maxLen}, torch::kInt64);
    for (int64_t j = 0; j < torch_.maxLen; j++)
      x.index_put_({0, j}, ids[(size_t)j]);
    auto y = torch_.cnn->forward(x);
    return clamp01(y.item<double>());
  }

  double torchRnnScore(const std::vector<std::string> &tokens) {
    if (!useTorchModels_)
      return 0.0;
    ensureTorchReady();
    if (!torch_.ready || !torch_.rnn)
      return 0.0;
    auto ids = torch_.vocab.encode(tokens, torch_.maxLen);
    auto x = torch::zeros({1, torch_.maxLen}, torch::kInt64);
    for (int64_t j = 0; j < torch_.maxLen; j++)
      x.index_put_({0, j}, ids[(size_t)j]);
    auto y = torch_.rnn->forward(x);
    return clamp01(y.item<double>());
  }
#endif

  double cosineSimVec(const std::vector<float> &a,
                      const std::vector<float> &b) const {
    if (a.empty() || b.empty() || a.size() != b.size())
      return 0.0;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    if (na <= 0.0 || nb <= 0.0)
      return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
  }

  double computeRecommenderScore(const std::vector<std::string> &tokens) {
    if (!useRecommender_)
      return 0.0;
    ensureRecommenderReady();
    if (topicVectors_.empty())
      return 0.0;
    auto emb = embedTokens(tokens, recommenderDim_);
    double best = 0.0;
    for (const auto &topic : topicVectors_) {
      best = std::max(best, cosineSimVec(emb, topic));
    }
    double risk = 1.0 - clamp01(best);
    return clamp01(risk);
  }

  double computeCnnScore(const std::vector<std::string> &tokens) const {
    if (!useTextCNN_)
      return 0.0;
    if (tokens.empty())
      return 0.0;

    const int side = 24;
    std::vector<float> grid((size_t)side * side, 0.0f);
    std::vector<float> grid2((size_t)side * side, 0.0f);

    for (size_t i = 0; i < tokens.size(); i++) {
      const std::string &token = tokens[i];
      if (token.empty())
        continue;
      uint32_t h1 = hashStrSimple(token, 2166136261u);
      uint32_t h2 = hashStrSimple(token, 16777619u);
      int r = (int)(h1 % side);
      int c = (int)((h1 / (uint32_t)side + h2) % side);
      double posDecay = 1.0 / (1.0 + 0.02 * (double)i);
      double lenGain = 1.0 + 0.1 * std::log1p((double)token.size());
      float v = (float)((0.4 + 0.6 * hashToUnit(token)) * posDecay * lenGain);
      grid[(size_t)r * side + c] += v;

      int r2 = (int)((h2 + i) % (uint32_t)side);
      int c2 = (int)((h2 / (uint32_t)side + h1 + 3 * i) % (uint32_t)side);
      grid2[(size_t)r2 * side + c2] += (float)(0.5 * v);
    }

    auto normalizeGrid = [&](std::vector<float> &g) {
      if (g.empty())
        return;
      double mean = 0.0;
      for (float x : g)
        mean += x;
      mean /= (double)g.size();
      double var = 0.0;
      for (float x : g) {
        double d = x - mean;
        var += d * d;
      }
      var /= (double)g.size();
      double stdv = std::sqrt(std::max(1e-9, var));
      for (auto &x : g)
        x = (float)((x - mean) / stdv);
    };
    normalizeGrid(grid);
    normalizeGrid(grid2);

    auto convStats = [&](const std::vector<float> &g, const float *k,
                         int stride, double &avg, double &mx, double &energy) {
      avg = 0.0;
      mx = -1e9;
      energy = 0.0;
      int cnt = 0;
      for (int r = 0; r < side; r += std::max(1, stride)) {
        for (int c = 0; c < side; c += std::max(1, stride)) {
          double sum = 0.0;
          for (int dr = -1; dr <= 1; dr++) {
            for (int dc = -1; dc <= 1; dc++) {
              int rr = std::max(0, std::min(side - 1, r + dr));
              int cc = std::max(0, std::min(side - 1, c + dc));
              sum += g[(size_t)rr * side + cc] * k[(dr + 1) * 3 + (dc + 1)];
            }
          }
          sum = std::max(0.0, sum);
          avg += sum;
          mx = std::max(mx, sum);
          energy += sum * sum;
          cnt++;
        }
      }
      if (cnt > 0) {
        avg /= cnt;
        energy = std::sqrt(energy / cnt);
      } else {
        avg = 0.0;
        mx = 0.0;
        energy = 0.0;
      }
    };

    const float kEdge[9] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
    const float kBlob[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
    const float kDiag[9] = {2, -1, 0, -1, 2, -1, 0, -1, 2};

    double avg1 = 0, max1 = 0, e1 = 0;
    double avg2 = 0, max2 = 0, e2 = 0;
    double avg3 = 0, max3 = 0, e3 = 0;
    convStats(grid, kEdge, 1, avg1, max1, e1);
    convStats(grid, kBlob, 2, avg2, max2, e2);
    convStats(grid2, kDiag, 1, avg3, max3, e3);

    std::unordered_set<std::string> uniq(tokens.begin(), tokens.end());
    double repeatRisk =
        1.0 - (double)uniq.size() / std::max(1.0, (double)tokens.size());
    double convRisk = 0.20 * clamp01(avg1 / 3.0) + 0.18 * clamp01(max1 / 5.0) +
                      0.18 * clamp01(e1 / 4.0) + 0.14 * clamp01(avg2 / 3.0) +
                      0.12 * clamp01(max2 / 4.5) + 0.10 * clamp01(e2 / 3.5) +
                      0.08 * clamp01((avg3 + max3 + e3) / 10.0);

    double mixed = 0.78 * convRisk + 0.22 * repeatRisk;
    return clamp01(mixed);
  }

  double computeCopyFeatures(const std::vector<std::string> &tokens) {
    if (tokens.empty())
      return 0.0;
    std::unordered_set<std::string> uniq(tokens.begin(), tokens.end());
    double variability =
        tokens.empty()
            ? 0.0
            : (double)uniq.size() / std::max<size_t>(1, tokens.size());
    double rep = 0.0;
    double sel = 0.0;
    double normIdf = std::log(1.0 + std::max(1, corpusTotal_));
    for (const auto &t : uniq) {
      int f = 0;
      auto it = corpusFreq_.find(t);
      if (it != corpusFreq_.end())
        f = it->second;
      rep += (double)f / std::max(1, corpusMaxFreq_);
      double idf =
          std::log(1.0 + (double)std::max(1, corpusTotal_) / (1.0 + f));
      sel += (normIdf > 0.0 ? idf / normIdf : 0.0);
    }
    rep = rep / std::max<size_t>(1, uniq.size());
    sel = sel / std::max<size_t>(1, uniq.size());
    double score = 0.5 * rep + 0.3 * (1.0 - variability) + 0.2 * (1.0 - sel);
    return clamp01(score);
  }

  double forestAggregate(double baseScore, double fScore, double cnnScore,
                         double recScore, double rnnScore) const {
    baseScore = clamp01(baseScore);
    fScore = clamp01(fScore);
    cnnScore = clamp01(cnnScore);
    recScore = clamp01(recScore);
    rnnScore = clamp01(rnnScore);

    double cnnRnnAgreement = 1.0 - std::abs(cnnScore - rnnScore);
    double textModelScore = 0.65 * cnnScore + 0.35 * rnnScore;
    textModelScore = clamp01(textModelScore);

    std::array<double, 4> aux = {fScore, textModelScore, recScore, baseScore};
    double mean = 0.0;
    for (double v : aux)
      mean += v;
    mean /= (double)aux.size();
    double var = 0.0;
    for (double v : aux) {
      double d = v - mean;
      var += d * d;
    }
    var /= (double)aux.size();
    double disagreement = clamp01(std::sqrt(std::max(0.0, var)) * 2.0);

    double wBase = 0.35;
    double wCopy = 0.20;
    double wText = 0.25 * (0.5 + 0.5 * cnnRnnAgreement);
    double wRec = 0.20;
    double wSum = wBase + wCopy + wText + wRec;

    double linear = (wBase * baseScore + wCopy * fScore +
                     wText * textModelScore + wRec * recScore) /
                    std::max(1e-9, wSum);
    double interaction =
        0.25 * baseScore * recScore + 0.20 * baseScore * textModelScore +
        0.15 * fScore * textModelScore + 0.10 * recScore * fScore;
    interaction = clamp01(interaction);

    double conservative =
        std::max({baseScore, 0.85 * textModelScore, 0.9 * recScore});
    double uncertaintyPenalty = 0.28 * disagreement;
    double score = 0.52 * linear + 0.30 * interaction + 0.18 * conservative -
                   uncertaintyPenalty;
    return clamp01(score);
  }

  double evaluateCompositeScore(const std::string &memeId, const NodeStats &ns,
                                double avgConn, double avgOut, double stdDeg,
                                double stdOut) {
    double baseScore =
        evaluateMaliciousness(memeId, ns, avgConn, avgOut, stdDeg, stdOut);
    std::vector<std::string> tokens =
        runtime_ ? runtime_->getMemeWords(memeId) : std::vector<std::string>();
    if (tokens.empty())
      tokens = tokenize(memeId);
    if (tokens.empty())
      return baseScore;
    ensureRecommenderReady();
    double fScore = computeCopyFeatures(tokens);
    double cnnScore = computeCnnScore(tokens);
    double rnnScore = 0.0;
#ifdef HAVE_TORCH
    if (useTorchModels_) {
      cnnScore = torchCnnScore(tokens);
      rnnScore = torchRnnScore(tokens);
    }
#endif
    double recScore = computeRecommenderScore(tokens);
    double forest =
        forestAggregate(baseScore, fScore, cnnScore, recScore, rnnScore);
    return clamp01(0.6 * forest + 0.4 * baseScore);
  }

  double getAverageConnections() {
    if (!runtime_)
      return 0.0;
    auto points = runtime_->graph().getAllPoints();
    if (points.empty())
      return 0;
    double sum = 0;
    for (auto &id : points)
      sum += runtime_->graph().degreeOf(id);
    return sum / points.size();
  }

  std::string generateReason(const std::string &memeId, double score) {
    if (!runtime_)
      return "score=0.000 conn=0";
    int degree = runtime_->graph().degreeOf(memeId);
    std::ostringstream oss;
    oss << "score=" << std::fixed << std::setprecision(3) << score
        << " conn=" << degree;
    if (useTextCNN_)
      oss << " cnn=on";
    else
      oss << " cnn=off";
    if (useRecommender_)
      oss << " rec=on";
    else
      oss << " rec=off";
    return oss.str();
  }

  bool isolateMeme(const std::string &memeId, double score,
                   const std::string &reason) {
    if (!runtime_)
      return false;
    int cut = runtime_->graph().removeOutgoingEdges(memeId);
    if (cut > 0) {
      stats_.isolated++;
      json item{{"memeID", memeId},
                {"score", score},
                {"reason", reason},
                {"cut", cut}};
      stats_.lastIsolated.insert(stats_.lastIsolated.begin(), item);
      if (stats_.lastIsolated.size() > 20)
        stats_.lastIsolated.resize(20);
      auto &logger = LoggerCXX::instance();
      if (logger.enabled())
        logger.log(LoggerCXX::Type::MONITORING,
                   std::string("memebarrier blocked meme=") + memeId +
                       " score=" + std::to_string(score) +
                       " cut=" + std::to_string(cut) + " reason=" + reason);
      return true;
    }
    return false;
  }

  void scanNetwork() {
    if (!runtime_)
      return;
    auto points = runtime_->graph().getAllPoints();
    stats_.scans++;
    stats_.lastScanTime = nowEpochMs();
    if (points.empty())
      return;
    struct Scored {
      std::string id;
      double score;
      std::string reason;
    };
    std::vector<Scored> scored;
    scored.reserve(points.size());
    std::unordered_map<std::string, NodeStats> nodeStats;
    nodeStats.reserve(points.size());
    double sumDeg = 0.0;
    double sumOut = 0.0;
    double sumDeg2 = 0.0;
    double sumOut2 = 0.0;
    for (auto &id : points) {
      auto neighbors = runtime_->graph().neighborsOf(id);
      NodeStats ns;
      ns.degree = (int)neighbors.size();
      for (auto &n : neighbors) {
        if (n.first == id)
          ns.selfLoops++;
        if (n.second.second == 2)
          ns.outDegree++;
      }
      nodeStats.emplace(id, ns);
      sumDeg += ns.degree;
      sumOut += ns.outDegree;
      sumDeg2 += (double)ns.degree * ns.degree;
      sumOut2 += (double)ns.outDegree * ns.outDegree;
    }
    double avgDeg = sumDeg / points.size();
    double avgOut = sumOut / points.size();
    double varDeg = sumDeg2 / points.size() - avgDeg * avgDeg;
    double varOut = sumOut2 / points.size() - avgOut * avgOut;
    double stdDeg = std::sqrt(std::max(0.0, varDeg));
    double stdOut = std::sqrt(std::max(0.0, varOut));
    stats_.avgDegree = avgDeg;
    stats_.stdDegree = stdDeg;
    stats_.avgOutDegree = avgOut;
    stats_.stdOutDegree = stdOut;

    std::vector<double> scoreValues;
    scoreValues.reserve(points.size());
    for (auto &id : points) {
      const auto &ns = nodeStats[id];
      double s = evaluateCompositeScore(id, ns, avgDeg, avgOut, stdDeg, stdOut);
      scored.push_back(Scored{id, s, generateReason(id, s)});
      scoreValues.push_back(s);
    }

    std::sort(scoreValues.begin(), scoreValues.end());
    stats_.evaluated = (int)scoreValues.size();
    stats_.avgScore = 0.0;
    stats_.maxScore = scoreValues.empty() ? 0.0 : scoreValues.back();
    for (double v : scoreValues)
      stats_.avgScore += v;
    if (!scoreValues.empty())
      stats_.avgScore /= scoreValues.size();
    if (!scoreValues.empty()) {
      size_t p90 = (size_t)std::floor(0.9 * (scoreValues.size() - 1));
      stats_.p90Score = scoreValues[p90];
    } else {
      stats_.p90Score = 0.0;
    }

    stats_.scoreHistogram.assign(10, 0);
    for (double v : scoreValues) {
      int bucket = (int)std::floor(std::min(0.999, std::max(0.0, v)) * 10.0);
      stats_.scoreHistogram[bucket]++;
    }

    double dynamicTarget =
        clamp01(0.7 * baseThreshold_ + 0.3 * stats_.p90Score);
    std::sort(
        scored.begin(), scored.end(),
        [](const Scored &a, const Scored &b) { return a.score > b.score; });
    int isolated = 0;
    stats_.candidates = 0;
    for (auto &item : scored) {
      if (item.score >= maliciousThreshold_) {
        stats_.candidates++;
        auto &hits = hitCounts_[item.id];
        if (item.score >= maliciousThreshold_ + minIsolationMargin_)
          hits = minConsecutiveHits_;
        else
          hits = std::min(minConsecutiveHits_, hits + 1);
        if (hits >= minConsecutiveHits_) {
          if (isolateMeme(item.id, item.score, item.reason)) {
            isolated++;
            if (isolated >= maxIsolatePerScan_)
              break;
          }
        }
      } else {
        hitCounts_[item.id] = 0;
        break;
      }
    }
    if (adaptiveEnabled_ && !scoreValues.empty()) {
      double isolatedRate =
          stats_.evaluated > 0 ? (double)isolated / stats_.evaluated : 0.0;
      double high = std::max(0.002, disableRateTarget_ * 1.5);
      double low = std::max(0.0005, disableRateTarget_ * 0.5);
      if (isolatedRate > high)
        dynamicTarget = clamp01(dynamicTarget + 0.05);
      if (isolatedRate < low)
        dynamicTarget = clamp01(dynamicTarget - 0.02);
      double blended = maliciousThreshold_ * (1.0 - thresholdMomentum_) +
                       dynamicTarget * thresholdMomentum_;
      maliciousThreshold_ =
          std::min(maxThreshold_, std::max(minThreshold_, blended));
    }
    stats_.lastThreshold = maliciousThreshold_;
  }
};
