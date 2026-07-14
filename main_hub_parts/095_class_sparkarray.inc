// Auto-generated from main.cpp
// Segment: class_SparkArray
// Original lines: 11390-11765

class SparkArray : public ISparkArray {
public:
  SparkArray(std::shared_ptr<ControllerPoolBase> pool,
             std::shared_ptr<ShardManager> shards,
             const json &options = json::object())
      : pool_(std::move(pool)), shards_(std::move(shards)) {
    auto personaOptions = options.value("personaForest", json::object());
    auto personaFactory =
        module_mount::ModuleRegistry::instance().personaForestAveragerFactory();
    if (personaFactory)
      personaForest_ = personaFactory(personaOptions);
    else
      personaForest_ = std::make_shared<PersonaForestAverager>(personaOptions);

    groupId_ = options.value("groupId", std::string("G1"));
    rfInstances_ = std::max(
        1,
        std::min(32, options.value("rfInstances", options.value("numAI", 7))));
    std::vector<std::string> available;
    if (options.contains("groupIds") && options["groupIds"].is_array() &&
        !options["groupIds"].empty()) {
      for (auto &gid : options["groupIds"]) {
        if (!gid.is_string())
          continue;
        auto vec = pool_->listControllersInGroup(gid.get<std::string>());
        available.insert(available.end(), vec.begin(), vec.end());
      }
    }
    if (available.empty())
      available = pool_->listControllersInGroup(groupId_);
    auto wantedRaw =
        options.contains("numAI")
            ? options["numAI"].get<int>()
            : (options.contains("groupSize") ? options["groupSize"].get<int>()
                                             : 7);
    int wanted = std::max(1, wantedRaw);
    int numAI = std::max(
        1, std::min((int)available.size() ? (int)available.size() : wanted,
                    wanted));
    layers_ = json::array();
    for (int i = 0; i < numAI; i++) {
      if (i >= (int)available.size())
        break;
      layers_.push_back(json{{"name", groupId_ + ":a" + std::to_string(i + 1)},
                             {"controllers", json::array({available[i]})},
                             {"strategy", "max"}});
    }
    if (options.contains("layers") && options["layers"].is_array() &&
        !options["layers"].empty()) {
      layers_ = json::array();
      for (auto &layer : options["layers"]) {
        json next = layer;
        if (!next.contains("strategy"))
          next["strategy"] = "max";
        layers_.push_back(next);
      }
    }
    defaultBudget_ = normalizeBudget(options.value("budget", json()));
  }

  json getLayers() const override { return layers_; }

  void updateLayers(const json &layers) override {
    if (!layers.is_array() || layers.empty())
      throw std::runtime_error("layers must be a non-empty array");
    for (auto &layer : layers) {
      if (!layer.contains("name") || !layer.contains("controllers") ||
          !layer["controllers"].is_array()) {
        throw std::runtime_error("invalid layer specification");
      }
    }
    layers_ = json::array();
    for (auto &layer : layers) {
      json next = layer;
      if (!next.contains("strategy"))
        next["strategy"] = "max";
      layers_.push_back(next);
    }
  }

  json dispatch(const json &payload,
                const json &options = json::object()) override {
    std::string text = payload.value("text", "");
    auto requestEmbedding = textToMiniEmbedding(text, 64);
    auto variants = buildVariants(text, options.value("perturbations", 0));
    json layers =
        options.value("multiLayer", true)
            ? layers_
            : json::array({layers_.empty() ? json::object() : layers_[0]});
    if (options.contains("numAI") && options["numAI"].is_number()) {
      int cap = std::max(1, options["numAI"].get<int>());
      json trimmed = json::array();
      for (int i = 0; i < (int)layers.size() && i < cap; i++)
        trimmed.push_back(layers[i]);
      layers = trimmed;
    }
    auto budget = mergeBudgets(defaultBudget_,
                               normalizeBudget(options.value(
                                   "budget", payload.value("budget", json()))));

    json layerResults = json::array();

    auto callRespond = [&](const std::string &controllerName,
                           const json &nextPayload) {
      return pool_->respondByName(controllerName, nextPayload);
    };

    for (auto &layer : layers) {
      json controllers = json::array();
      for (auto &controllerSpec : layer.value("controllers", json::array())) {
        std::string controllerName =
            controllerSpec.is_string()
                ? controllerSpec.get<std::string>()
                : controllerSpec.value("name",
                                       controllerSpec.value("controller", ""));
        if (controllerName.empty()) {
          controllers.push_back(
              json{{"controller", ""}, {"error", "invalid-controller"}});
          continue;
        }
        int weight = 1;
        if (controllerSpec.is_object()) {
          auto w = controllerSpec.value("weight", controllerSpec.value("w", 1));
          weight = std::max(1, std::min(8, w));
        }
        std::string weightedText = weight <= 1 ? text : std::string();
        if (weight > 1) {
          for (int i = 0; i < weight; i++) {
            if (i)
              weightedText += " ";
            weightedText += text;
          }
        }

        json nextPayload = payload;
        nextPayload["text"] = weightedText;
        if (budget)
          nextPayload["budget"] = *budget;

        json baseResult;
        try {
          baseResult = callRespond(controllerName, nextPayload);
        } catch (const std::exception &e) {
          shards_->recordFailure(controllerName);
          controllers.push_back(
              json{{"controller", controllerName}, {"error", e.what()}});
          continue;
        }
        double affinity =
            cosineSim(requestEmbedding,
                      textToMiniEmbedding(baseResult.value("reply", ""), 64));
        shards_->record(controllerName, requestEmbedding,
                        baseResult.value("reply", ""),
                        baseResult.value("latency", 0));
        json variantResults = json::array();
        for (auto &variant : variants) {
          json vrPayload = payload;
          vrPayload["text"] = variant;
          if (budget)
            vrPayload["budget"] = *budget;
          try {
            auto vr = callRespond(controllerName, vrPayload);
            double vAffinity =
                cosineSim(textToMiniEmbedding(variant, 64),
                          textToMiniEmbedding(vr.value("reply", ""), 64));
            variantResults.push_back(json{
                {"text", variant}, {"response", vr}, {"affinity", vAffinity}});
          } catch (const std::exception &e) {
            variantResults.push_back(
                json{{"text", variant}, {"error", e.what()}});
          }
        }
        controllers.push_back(json{{"controller", controllerName},
                                   {"base", baseResult},
                                   {"affinity", affinity},
                                   {"variants", variantResults}});
      }
      layerResults.push_back(json{{"layer", layer.value("name", "")},
                                  {"controllers", controllers}});
    }

    auto aggregate = aggregateResults(layerResults, payload, requestEmbedding);
    recordHistory(payload, aggregate, layerResults);
    return json{{"aggregate", aggregate}, {"layers", layerResults}};
  }

  json dispatchBig(const json &payload, const json &options) override {
    int rounds = std::max(
        1, std::min(10, options.value("bigRounds",
                                      options.value("bigSparkRounds", 1))));
    json history = json::array();
    json current = payload;
    std::string originalText = payload.value("text", "");
    for (int i = 0; i < rounds; i++) {
      auto r = dispatch(current, options);
      history.push_back(r);
      std::string reply =
          r.value("aggregate", json::object()).value("reply", "");
      if (reply.empty())
        break;
      std::string combined = originalText + "\n" + reply;
      if (combined.size() > 4000)
        combined = combined.substr(combined.size() - 4000);
      current["text"] = combined;
    }
    json last = history.empty()
                    ? json{{"aggregate", nullptr}, {"layers", json::array()}}
                    : history.back();
    last["rounds"] = history;
    return last;
  }

  const std::vector<json> &history() const override { return history_; }

private:
  struct RouteMetric {
    uint64_t calls{0};
    uint64_t errors{0};
    uint64_t totalMs{0};
    uint64_t maxMs{0};
    uint64_t lastMs{0};
  };

  std::shared_ptr<ControllerPoolBase> pool_;
  std::shared_ptr<ShardManager> shards_;
  std::string groupId_{"G1"};
  int rfInstances_{7};
  json layers_ = json::array();
  std::shared_ptr<IPersonaForestAverager> personaForest_;
  std::optional<json> defaultBudget_;
  std::vector<json> history_;

  json aggregateResults(const json &layerResults, const json &payload,
                        const std::vector<float> &requestEmbedding) {
    std::unordered_map<std::string, double> tokenVotes;
    std::vector<double> rfScores;
    rfScores.reserve((size_t)rfInstances_);
    std::unordered_set<std::string> stop{
        "the", "a",  "an", "is",   "are", "was",  "were", "and",  "or",
        "to",  "of", "in", "on",   "for", "with", "that", "this", "it",
        "as",  "be", "by", "from", "at",  "we",   "you",  "i"};
    auto requestText = payload.value("text", "");
    auto inputTok = tokenize(requestText);
    json transformerCandidates =
        payload.value("sparkTransformerCandidates", json::array());
    int voteBudget = std::max(
        1, rfInstances_ + std::min(8, (int)transformerCandidates.size()));
    int used = 0;
    int gnnUsed = 0;
    int transformerUsed = 0;

    auto scoreAndVoteReply = [&](const std::string &reply, double affinity,
                                 double latency, double affinityWeight,
                                 double latencyWeight, double overlapWeight,
                                 double scoreBoost) {
      auto replyTok = tokenize(reply);
      std::unordered_set<std::string> setReply(replyTok.begin(),
                                               replyTok.end());
      int hit = 0;
      for (auto &w : inputTok) {
        if (setReply.count(w))
          hit++;
      }
      double overlap =
          inputTok.empty()
              ? 0.0
              : (double)hit / std::max(1.0, (double)inputTok.size());
      double latencyNorm = clamp01(1.0 - std::max(0.0, latency) / 2000.0);
      double score = affinityWeight * clamp01((affinity + 1.0) * 0.5) +
                     latencyWeight * latencyNorm + overlapWeight * overlap;
      score = clamp01(score * std::max(0.1, scoreBoost));
      rfScores.push_back(score);
      if (used < voteBudget) {
        used++;
        double weight = std::max(0.05, score);
        for (auto &t : replyTok) {
          if (t.size() < 2 || t.size() > 24)
            continue;
          if (stop.count(t))
            continue;
          tokenVotes[t] += weight;
        }
      }
    };

    for (auto &layer : layerResults) {
      for (auto &controller : layer.value("controllers", json::array())) {
        if (!controller.contains("base"))
          continue;
        auto base = controller["base"];
        if (!base.contains("reply"))
          continue;
        double affinity = controller.value("affinity", 0.0);
        std::string reply = base.value("reply", "");
        if (reply.empty())
          continue;
        double latency = std::max(0.0, base.value("latency", 0.0));
        scoreAndVoteReply(reply, affinity, latency, 0.55, 0.25, 0.20, 1.0);
        gnnUsed++;
      }
    }

    if (transformerCandidates.is_array()) {
      for (auto &candidate : transformerCandidates) {
        if (!candidate.is_object())
          continue;
        std::string reply = candidate.value("reply", "");
        if (reply.empty())
          continue;
        double latency = std::max(0.0, candidate.value("latency", 0.0));
        double affinity =
            candidate.contains("affinity") && candidate["affinity"].is_number()
                ? candidate["affinity"].get<double>()
                : cosineSim(requestEmbedding, textToMiniEmbedding(reply, 64));
        scoreAndVoteReply(reply, affinity, latency, 0.60, 0.20, 0.20, 1.10);
        transformerUsed++;
      }
    }

    std::vector<std::pair<std::string, double>> ranked;
    ranked.reserve(tokenVotes.size());
    for (auto &kv : tokenVotes)
      ranked.push_back(kv);
    std::sort(ranked.begin(), ranked.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });
    if (ranked.size() > 24)
      ranked.resize(24);

    std::ostringstream ann;
    ann << "gnn_transformer_rf_stage3|rf=" << rfInstances_
        << "|budget=" << voteBudget << "|n=" << used << "|gnn=" << gnnUsed
        << "|transformer=" << transformerUsed << ":";
    for (size_t i = 0; i < ranked.size(); i++) {
      if (i)
        ann << ' ';
      ann << ranked[i].first;
    }
    std::string annotation = ann.str();
    if (annotation.size() > 512)
      annotation.resize(512);

    double conf = 0.0;
    if (!rfScores.empty()) {
      for (double s : rfScores)
        conf += s;
      conf /= (double)rfScores.size();
    }

    json topTokens = json::array();
    for (auto &p : ranked)
      topTokens.push_back(json{{"token", p.first}, {"vote", p.second}});

    return json{{"annotation", annotation},
                {"confidence", conf},
                {"reply", ""},
                {"method", "gnn-transformer-rf-ensemble"},
                {"rfInstances", rfInstances_},
                {"voteBudget", voteBudget},
                {"gnnSignals", gnnUsed},
                {"transformerSignals", transformerUsed},
                {"topTokens", topTokens}};
  }

  void recordHistory(const json &payload, const json &aggregate,
                     const json &layerResults) {
    history_.push_back(json{
        {"ts",
         (int64_t)std::chrono::system_clock::now().time_since_epoch().count()},
        {"request", json{{"text", payload.value("text", "")},
                         {"sessionId", payload.value("sessionId", "")}}},
        {"aggregate", aggregate},
        {"layers", layerResults}});
    if (history_.size() > 100)
      history_.erase(history_.begin());
  }
};
