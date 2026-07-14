// Auto-generated from main.cpp
// Segment: class_PersonaForestAverager
// Original lines: 11052-11388

class PersonaForestAverager : public IPersonaForestAverager {
public:
  explicit PersonaForestAverager(const json &options = json::object()) {
    enabled_ = options.value("enabled", true);
    trees_ = std::max(8, options.value("trees", 32));
    featureSubspace_ = std::max(2, options.value("featureSubspace", 4));
    sampleRate_ = clamp01(options.value("sampleRate", 0.85));
    personaMomentum_ = clamp01(options.value("personaMomentum", 0.92));
    targetReplyLen_ = std::max(20, options.value("targetReplyLen", 220));
    maxHistory_ = std::max(10, options.value("maxHistory", 60));
  }

  int maxHistory() const override { return maxHistory_; }

  std::optional<json> pick(const json &payload, const json &layerResults,
                           const std::vector<float> &requestEmbedding,
                           const std::vector<json> &history) override {
    if (!enabled_)
      return std::nullopt;
    int dim = requestEmbedding.empty() ? 64 : (int)requestEmbedding.size();
    std::string inputText = payload.value("text", "");
    auto inputTokens = tokenize(inputText);

    std::vector<Candidate> candidates;
    if (layerResults.is_array()) {
      for (auto &layer : layerResults) {
        auto controllers = layer.value("controllers", json::array());
        for (auto &controller : controllers) {
          auto base = controller.value("base", json::object());
          if (!base.contains("reply"))
            continue;
          std::string reply = base.value("reply", "");
          if (reply.empty())
            continue;
          Candidate c;
          c.layer = layer.value("layer", "");
          c.controller = controller.value("controller", "");
          c.reply = reply;
          c.latency = base.value("latency", 0);
          c.sessionId = base.value("sessionId", "");
          c.emb = textToMiniEmbedding(reply, dim);
          if (controller.contains("affinity") &&
              controller["affinity"].is_number()) {
            c.affinity = clamp11(controller["affinity"].get<double>());
          } else {
            c.affinity = clamp11(cosineSim(requestEmbedding, c.emb));
          }
          c.controllerRef = controller;
          candidates.push_back(std::move(c));
        }
      }
    }

    if (candidates.empty())
      return std::nullopt;
    if (candidates.size() == 1) {
      updatePersonas(candidates[0]);
      return json{{"layer", candidates[0].layer},
                  {"controller", candidates[0].controller},
                  {"affinity", candidates[0].affinity},
                  {"reply", candidates[0].reply},
                  {"latency", candidates[0].latency},
                  {"sessionId", candidates[0].sessionId},
                  {"score", 0},
                  {"method", "forest-single"}};
    }

    auto centroid = centroidEmbedding(candidates, dim);
    json lastAgg = history.empty()
                       ? json::object()
                       : history.back().value("aggregate", json::object());
    std::vector<float> lastReplyEmb;
    if (lastAgg.contains("reply"))
      lastReplyEmb = textToMiniEmbedding(lastAgg.value("reply", ""), dim);
    int lastReplyLen =
        lastAgg.contains("reply") ? (int)lastAgg.value("reply", "").size() : 0;
    int targetLen = (lastReplyLen > 20) ? lastReplyLen : targetReplyLen_;

    for (auto &c : candidates) {
      auto replyTokens = tokenize(c.reply);
      double overlap = tokenOverlapRatio(inputTokens, replyTokens);
      double consensus =
          centroid.empty() ? 0.0 : clamp11(cosineSim(c.emb, centroid));
      double persona = globalPersona_.empty()
                           ? 0.0
                           : clamp11(cosineSim(c.emb, globalPersona_));
      auto it = controllerPersona_.find(c.controller);
      double controllerPersona =
          (it == controllerPersona_.end() || it->second.empty())
              ? 0.0
              : clamp11(cosineSim(c.emb, it->second));
      double novelty =
          lastReplyEmb.empty()
              ? 0.5
              : clamp01(1 -
                        ((clamp11(cosineSim(c.emb, lastReplyEmb)) + 1) / 2));
      double lenPenalty = -clamp01(std::abs((int)c.reply.size() - targetLen) /
                                   (double)std::max(40, targetLen));
      double stability = variantStability(c.controllerRef);
      double latencyNorm =
          c.latency > 0 ? clamp01(1 - (c.latency / 2000.0)) : 0.5;
      c.features =
          json{{"affinity", clamp01((c.affinity + 1) / 2)},
               {"overlap", overlap},
               {"consensus", clamp01((consensus + 1) / 2)},
               {"persona", clamp01((persona + 1) / 2)},
               {"controllerPersona", clamp01((controllerPersona + 1) / 2)},
               {"novelty", novelty},
               {"lenPenalty", clamp01(1 + lenPenalty)},
               {"stability", stability},
               {"latency", latencyNorm}};
    }

    auto featureNames = json::array({"affinity", "overlap", "consensus",
                                     "persona", "controllerPersona", "novelty",
                                     "lenPenalty", "stability", "latency"});
    std::string seedText = payload.value("sessionId", "") + "|" +
                           std::to_string(hashStrSimple(inputText)) + "|" +
                           std::to_string(candidates.size());
    Rng32 rng(hashStrSimple(seedText));

    std::unordered_map<int, double> scores;
    std::unordered_map<int, int> votes;
    for (size_t i = 0; i < candidates.size(); i++) {
      scores[(int)i] = 0;
      votes[(int)i] = 0;
    }

    auto pickSubspace = [&]() {
      std::unordered_set<int> chosen;
      int maxPick = std::min(featureSubspace_, (int)featureNames.size());
      while ((int)chosen.size() < maxPick) {
        int idx = (int)std::floor(rng.next() * featureNames.size());
        chosen.insert(idx);
      }
      std::vector<int> out(chosen.begin(), chosen.end());
      return out;
    };

    for (int t = 0; t < trees_; t++) {
      auto subspace = pickSubspace();
      std::unordered_map<std::string, double> weights;
      for (auto idx : subspace) {
        std::string f = featureNames[idx].get<std::string>();
        double w = (rng.next() * 2 - 1);
        if (f == "affinity" || f == "consensus" || f == "persona")
          w *= 1.4;
        if (f == "controllerPersona" || f == "stability")
          w *= 1.1;
        if (f == "novelty")
          w *= 0.6;
        weights[f] = w;
      }

      std::vector<int> bag;
      for (size_t i = 0; i < candidates.size(); i++) {
        if (rng.next() <= sampleRate_)
          bag.push_back((int)i);
      }
      if (bag.size() < 2) {
        bag.clear();
        for (size_t i = 0; i < candidates.size(); i++)
          bag.push_back((int)i);
      }

      int bestIdx = -1;
      double bestScore = -1e9;
      for (auto idx : bag) {
        double s = 0.0;
        for (auto &kv : weights) {
          double v = candidates[idx].features.value(kv.first, 0.0);
          s += kv.second * v;
        }
        if (s > bestScore) {
          bestScore = s;
          bestIdx = idx;
        }
      }
      if (bestIdx >= 0)
        votes[bestIdx] += 1;
      for (size_t i = 0; i < candidates.size(); i++) {
        double s = 0.0;
        for (auto &kv : weights) {
          double v = candidates[i].features.value(kv.first, 0.0);
          s += kv.second * v;
        }
        scores[(int)i] += s;
      }
    }

    int winner = -1;
    for (size_t i = 0; i < candidates.size(); i++) {
      if (winner == -1) {
        winner = (int)i;
        continue;
      }
      int vi = votes[(int)i];
      int vw = votes[winner];
      double si = scores[(int)i];
      double sw = scores[winner];
      double ci = candidates[i].features.value("consensus", 0.0);
      double cw = candidates[winner].features.value("consensus", 0.0);
      double ai = candidates[i].features.value("affinity", 0.0);
      double aw = candidates[winner].features.value("affinity", 0.0);
      if (vi > vw ||
          (vi == vw &&
           (si > sw || (si == sw && (ci > cw || (ci == cw && ai > aw)))))) {
        winner = (int)i;
      }
    }

    if (winner < 0)
      return std::nullopt;
    updatePersonas(candidates[winner]);
    return json{{"layer", candidates[winner].layer},
                {"controller", candidates[winner].controller},
                {"affinity", candidates[winner].affinity},
                {"reply", candidates[winner].reply},
                {"latency", candidates[winner].latency},
                {"sessionId", candidates[winner].sessionId},
                {"score", scores[winner] / std::max(1, trees_)},
                {"votes", votes[winner]},
                {"method", "persona-forest"}};
  }

private:
  struct Candidate {
    std::string layer;
    std::string controller;
    std::string reply;
    int latency{0};
    std::string sessionId;
    double affinity{0};
    std::vector<float> emb;
    json controllerRef;
    json features;
  };
  bool enabled_{true};
  int trees_{32};
  int featureSubspace_{4};
  double sampleRate_{0.85};
  double personaMomentum_{0.92};
  int targetReplyLen_{220};
  int maxHistory_{60};
  std::vector<float> globalPersona_;
  std::unordered_map<std::string, std::vector<float>> controllerPersona_;

  static std::vector<float>
  centroidEmbedding(const std::vector<Candidate> &cands, int dim) {
    if (cands.empty())
      return {};
    std::vector<float> out(dim, 0.0f);
    int count = 0;
    for (auto &c : cands) {
      if ((int)c.emb.size() != dim)
        continue;
      for (int i = 0; i < dim; i++)
        out[i] += c.emb[i];
      count++;
    }
    if (!count)
      return {};
    for (int i = 0; i < dim; i++)
      out[i] /= count;
    return out;
  }

  static double tokenOverlapRatio(const std::vector<std::string> &a,
                                  const std::vector<std::string> &b) {
    if (a.empty() || b.empty())
      return 0.0;
    std::unordered_set<std::string> setB(b.begin(), b.end());
    double hit = 0.0;
    for (auto &t : a)
      if (setB.count(t))
        hit += 1.0;
    return hit / std::max(1.0, (double)a.size());
  }

  double variantStability(const json &controllerRef) const {
    auto variants = controllerRef.value("variants", json::array());
    if (!variants.is_array() || variants.empty())
      return 0.5;
    double sum = 0.0;
    int n = 0;
    for (auto &v : variants) {
      if (!v.is_object() || !v.contains("affinity"))
        continue;
      if (!v["affinity"].is_number())
        continue;
      double a = v["affinity"].get<double>();
      if (!std::isfinite(a))
        continue;
      sum += clamp11(a);
      n++;
    }
    if (!n)
      return 0.5;
    return clamp01((sum / n + 1) / 2);
  }

  void updatePersonas(const Candidate &winner) {
    if (winner.emb.empty())
      return;
    auto blend = [&](const std::vector<float> &prev,
                     const std::vector<float> &next) {
      if (next.empty())
        return prev;
      if (prev.empty())
        return next;
      std::vector<float> out(prev.size(), 0.0f);
      for (size_t i = 0; i < prev.size(); i++) {
        double v =
            personaMomentum_ * prev[i] + (1 - personaMomentum_) * next[i];
        out[i] = std::isfinite(v) ? (float)v : 0.0f;
      }
      return out;
    };
    globalPersona_ = blend(globalPersona_, winner.emb);
    auto it = controllerPersona_.find(winner.controller);
    if (it == controllerPersona_.end())
      controllerPersona_[winner.controller] = blend({}, winner.emb);
    else
      it->second = blend(it->second, winner.emb);

    if (controllerPersona_.size() > 64) {
      std::vector<std::string> keys;
      keys.reserve(controllerPersona_.size());
      for (auto &kv : controllerPersona_)
        keys.push_back(kv.first);
      while (controllerPersona_.size() > 64) {
        size_t idx = (size_t)(std::rand() % keys.size());
        controllerPersona_.erase(keys[idx]);
      }
    }
  }
};
