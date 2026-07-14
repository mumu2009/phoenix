// Auto-generated from main.cpp
// Segment: class_RuntimeState
// Original lines: 6268-7994

class RuntimeState : public RuntimeBase {
public:
  RuntimeState(std::shared_ptr<KeyValueStore> kvmStore,
               std::shared_ptr<KeyValueStore> memeStore,
               std::shared_ptr<KeyValueStore> sessionStore,
               const Config &config)
      : config_(config), kvm_(kvmStore, config.kvmCacheMaxEntries),
        graph_(memeStore), sessions_(sessionStore), pattern_(&kvm_, &tensor_) {
    surfaceStore_ = std::make_shared<NamespacedStore>(sessionStore, "surface");
    dialogStore_ = std::make_shared<NamespacedStore>(sessionStore, "dialog");
    surfaceLexicon_ = std::make_shared<MemeSurfaceLexicon>(surfaceStore_);
    dialogMemory_ = std::make_shared<DialogMemory>(dialogStore_);
    // Increase maxSessions to 10000 to handle long-running benchmarks with many unique sessions
    sessionManager_ = std::make_shared<SessionManager>(sessionStore, 10 * 60 * 1000, 10000);
    params_ = getModelDefaults();
    searchConfig_.enabled = true;
    if (!config_.searchEndpoint.empty()) {
      searchConfig_.active = config_.searchEndpoint;
      searchConfig_.endpoints.push_back(config_.searchEndpoint);
    }
    if (!searchConfig_.active.empty() &&
        std::find(searchConfig_.endpoints.begin(),
                  searchConfig_.endpoints.end(),
                  searchConfig_.active) == searchConfig_.endpoints.end()) {
      searchConfig_.endpoints.push_back(searchConfig_.active);
    }
    researcher_ =
        std::make_shared<OnlineResearcher>(this, searchConfig_.active);
    researcher_->setEnabled(searchConfig_.enabled);
    addons_ = addon::createDefaultAddons();
  }

  json cloneParams() const override {
    std::lock_guard<std::mutex> lock(stateMu_);
    return cloneParamsLocked();
  }

  json listAddons() const {
    return addons_ ? addons_->listAddons() : json::array();
  }

  bool addAddon(const std::string &type, const std::string &name,
                std::string &error) {
    if (!addons_) {
      error = "addon manager unavailable";
      return false;
    }
    return addons_->addBuiltin(type, name, &error);
  }

  bool addAddon(const std::string &path, std::string &error) {
    if (!addons_) {
      error = "addon manager unavailable";
      return false;
    }
    return addons_->loadLibrary(path, &error);
  }

  bool removeAddon(const std::string &name, std::string &error) {
    if (!addons_) {
      error = "addon manager unavailable";
      return false;
    }
    return addons_->removeAddon(name, &error);
  }

  addon::AddonResult runAddonTool(const std::string &text,
                                  const json &payload) {
    if (!addons_)
      return addon::AddonResult{};
    addon::setAddonOnlineLookupHandler(
        [this](const json &input, const json &options) {
          return this->onlineLookup(input, options);
        });
    struct LookupGuard {
      ~LookupGuard() { addon::clearAddonOnlineLookupHandler(); }
    } guard;
    auto result = addons_->run(text, payload);
    return result;
  }

  void setParamsLocked(const json &patch) {
    if (patch.contains("decayFactor"))
      params_.decayFactor = patch["decayFactor"].get<double>();
    if (patch.contains("maxMemeWords"))
      params_.maxMemeWords = patch["maxMemeWords"].get<int>();
    if (patch.contains("minOverlapThreshold"))
      params_.minOverlapThreshold = patch["minOverlapThreshold"].get<int>();
    if (patch.contains("memeNgramMin"))
      params_.memeNgramMin = patch["memeNgramMin"].get<int>();
    if (patch.contains("memeNgramMax"))
      params_.memeNgramMax = patch["memeNgramMax"].get<int>();
    if (patch.contains("maliciousThreshold"))
      params_.maliciousThreshold = patch["maliciousThreshold"].get<double>();
    if (patch.contains("learningIterations"))
      params_.learningIterations = patch["learningIterations"].get<int>();
    if (patch.contains("iteration"))
      params_.iteration = patch["iteration"].get<int>();
    if (patch.contains("threshold"))
      params_.threshold = patch["threshold"].get<double>();
    if (patch.contains("decay"))
      params_.decay = patch["decay"].get<double>();
    if (patch.contains("decayK"))
      params_.decayK = patch["decayK"].get<double>();
    if (patch.contains("maxLen"))
      params_.maxLen = patch["maxLen"].get<int>();
    if (patch.contains("edgeWeight"))
      params_.edgeWeight = patch["edgeWeight"].get<double>();
    if (patch.contains("activationType"))
      params_.activationType = patch["activationType"].get<std::string>();
    if (patch.contains("transferType"))
      params_.transferType = patch["transferType"].get<std::string>();
    if (patch.contains("activationCustom"))
      params_.activationCustom = patch["activationCustom"].get<std::string>();
    if (patch.contains("transferCustom"))
      params_.transferCustom = patch["transferCustom"].get<std::string>();
    if (patch.contains("mappingDepth"))
      params_.mappingDepth = patch["mappingDepth"].get<int>();
    if (patch.contains("reflectionTopMemes"))
      params_.reflectionTopMemes = patch["reflectionTopMemes"].get<int>();
    if (patch.contains("reflectionTopWords"))
      params_.reflectionTopWords = patch["reflectionTopWords"].get<int>();
    if (patch.contains("reflectionMinScore"))
      params_.reflectionMinScore = patch["reflectionMinScore"].get<double>();
  }
  void setParams(const json &patch) override {
    std::lock_guard<std::mutex> lock(stateMu_);
    setParamsLocked(patch);
  }

  struct SeedMapping {
    std::unordered_map<std::string, double> seeds;
    std::vector<std::string> order;
  };

  SeedMapping mapWordsToMemesOrdered(const std::vector<std::string> &words) {
    SeedMapping out;
    auto normalizeToken = [](std::string s) {
      s = trimCopy(s);
      if (s.empty())
        return s;
      std::transform(s.begin(), s.end(), s.begin(),
                     [](unsigned char ch) { return (char)std::tolower(ch); });
      while (!s.empty()) {
        unsigned char c = (unsigned char)s.front();
        if (std::isalnum(c) || c == '_' || c == '-' || c >= 0x80)
          break;
        s.erase(s.begin());
      }
      while (!s.empty()) {
        unsigned char c = (unsigned char)s.back();
        if (std::isalnum(c) || c == '_' || c == '-' || c >= 0x80)
          break;
        s.pop_back();
      }
      return s;
    };

    std::vector<std::string> rawTokens;
    rawTokens.reserve(words.size());
    for (auto &w : words) {
      std::string t = normalizeToken(w);
      if (!t.empty())
        rawTokens.push_back(t);
    }

    const auto &stop = getStopWords();
    std::vector<std::string> tokens;
    tokens.reserve(rawTokens.size());
    for (auto &t : rawTokens) {
      bool hasNonAscii = false;
      for (unsigned char c : t) {
        if (c >= 0x80) {
          hasNonAscii = true;
          break;
        }
      }
      if (!hasNonAscii) {
        if (t.size() <= 1)
          continue;
        if (stop.count(t))
          continue;
      }
      tokens.push_back(t);
    }
    if (tokens.empty())
      tokens = rawTokens;

    std::unordered_map<std::string, int> tokenFreq;
    for (const auto &t : tokens)
      tokenFreq[t]++;

    auto tokenWeight = [&](const std::string &token, size_t idx) {
      double posDecay = 1.0 / (1.0 + 0.03 * (double)idx);
      double freqBoost = 1.0 + 0.15 * std::log1p((double)tokenFreq[token]);
      return posDecay * freqBoost;
    };

    auto addSeed = [&](const std::string &memeId, double delta) {
      if (memeId.empty())
        return;
      auto it = out.seeds.find(memeId);
      if (it == out.seeds.end()) {
        out.seeds[memeId] = delta;
        out.order.push_back(memeId);
      } else {
        it->second += delta;
      }
    };

    for (size_t wi = 0; wi < tokens.size(); ++wi) {
      const auto &word = tokens[wi];
      auto memes = kvm_.getWordMemeSet(word);
      if (memes.empty()) {
        std::string memeId = "meme_" + sha1Hex(word);
        graph_.ensureNode(memeId);
        kvm_.link(word, memeId);
        addSeed(memeId, tokenWeight(word, wi));
      } else {
        for (auto &mid : memes)
          addSeed(mid, tokenWeight(word, wi));
      }
    }

    int nMin = std::max(2, params_.memeNgramMin);
    int nMax = std::max(nMin, params_.memeNgramMax);
    int minOverlap = std::max(1, params_.minOverlapThreshold);
    int maxWordSet = std::max(4, params_.maxMemeWords);
    int units = 0;
    int maxUnits = 128;
    std::unordered_set<std::string> seenNgram;

    auto resolveOrCreate =
        [&](const std::vector<std::string> &tokenSet) -> std::string {
      std::vector<std::string> uniq;
      std::unordered_set<std::string> seen;
      for (auto &t : tokenSet) {
        if (t.empty())
          continue;
        if (seen.insert(t).second)
          uniq.push_back(t);
      }
      if ((int)uniq.size() > maxWordSet)
        uniq.resize(maxWordSet);
      if (uniq.size() <= 1)
        return uniq.empty() ? "" : ("meme_" + sha1Hex(uniq[0]));

      std::unordered_map<std::string, int> counts;
      for (auto &w : uniq) {
        for (auto &mid : kvm_.getWordMemeSet(w))
          counts[mid]++;
      }
      std::string best;
      int bestOverlap = 0;
      for (auto &kv : counts) {
        if (kv.second > bestOverlap) {
          bestOverlap = kv.second;
          best = kv.first;
        }
      }
      if (!best.empty() && bestOverlap >= minOverlap) {
        for (auto &w : uniq)
          kvm_.link(w, best);
        graph_.ensureNode(best);
        return best;
      }
      std::vector<std::string> sorted = uniq;
      std::sort(sorted.begin(), sorted.end());
      std::string memeId = "meme_p_" + sha1Hex([&]() {
                             std::string join;
                             for (size_t i = 0; i < sorted.size(); i++) {
                               if (i)
                                 join += "|";
                               join += sorted[i];
                             }
                             return join;
                           }());
      graph_.ensureNode(memeId);
      for (auto &w : sorted)
        kvm_.link(w, memeId);
      return memeId;
    };

    for (size_t i = 0; i < tokens.size() && units < maxUnits; i++) {
      for (int n = nMin; n <= nMax && units < maxUnits; n++) {
        if (i + n > tokens.size())
          break;
        std::vector<std::string> gram(tokens.begin() + i,
                                      tokens.begin() + i + n);
        std::string gramKey;
        gramKey.reserve(n * 8);
        for (int gi = 0; gi < n; gi++) {
          if (gi)
            gramKey.push_back('\x1f');
          gramKey += gram[(size_t)gi];
        }
        if (!seenNgram.insert(gramKey).second)
          continue;
        std::string memeId = resolveOrCreate(gram);
        if (memeId.empty())
          continue;
        std::unordered_set<std::string> uniq(gram.begin(), gram.end());
        double uniqRatio = (double)uniq.size() / std::max(1, n);
        double w = (1.0 + 0.45 * (n - 1)) * uniqRatio;
        w *= 1.0 / (1.0 + 0.02 * (double)i);
        addSeed(memeId, w);
        units++;
      }
    }

    return out;
  }

  std::unordered_map<std::string, double>
  mapWordsToMemes(const std::vector<std::string> &words) override {
    std::lock_guard<std::mutex> lock(stateMu_);
    return mapWordsToMemesOrdered(words).seeds;
  }

  json processInput(const json &payload) override {
    std::lock_guard<std::mutex> lock(stateMu_);
    auto started = std::chrono::steady_clock::now();
    std::function<std::string(const json &)> jsStringForJoin;
    std::function<std::string(const json &)> jsStringValue;
    jsStringForJoin = [&](const json &v) -> std::string {
      if (v.is_null())
        return "";
      if (v.is_string())
        return v.get<std::string>();
      if (v.is_boolean())
        return v.get<bool>() ? "true" : "false";
      if (v.is_number())
        return v.dump();
      if (v.is_array()) {
        std::string out;
        bool first = true;
        for (auto &item : v) {
          if (!first)
            out += ",";
          first = false;
          out += jsStringForJoin(item);
        }
        return out;
      }
      return "[object Object]";
    };
    jsStringValue = [&](const json &v) -> std::string {
      if (v.is_null())
        return "null";
      if (v.is_string())
        return v.get<std::string>();
      if (v.is_boolean())
        return v.get<bool>() ? "true" : "false";
      if (v.is_number())
        return v.dump();
      if (v.is_array())
        return jsStringForJoin(v);
      return "[object Object]";
    };
    auto jsStringOrEmpty = [&](const json &v) -> std::string {
      if (v.is_null())
        return "";
      if (v.is_boolean())
        return v.get<bool>() ? "true" : "";
      if (v.is_number()) {
        double n = v.get<double>();
        if (n == 0.0)
          return "";
        return v.dump();
      }
      if (v.is_string())
        return v.get<std::string>();
      if (v.is_array()) {
        std::string out;
        bool first = true;
        for (auto &item : v) {
          if (!first)
            out += ",";
          first = false;
          out += jsStringForJoin(item);
        }
        return out;
      }
      return "[object Object]";
    };
    json textVal = json::object();
    if (payload.contains("text") && !payload["text"].is_null())
      textVal = payload["text"];
    else if (payload.contains("message") && !payload["message"].is_null())
      textVal = payload["message"];
    std::string text = jsStringOrEmpty(textVal);
    auto tokensRaw = payload.value(
        "tokens",
        payload.value("words", payload.value("vocab", json::array())));
    std::vector<std::string> words;
    if (tokensRaw.is_array() && !tokensRaw.empty()) {
      for (auto &w : tokensRaw) {
        words.push_back(jsStringValue(w));
      }
    } else {
      words = tokenize(text);
    }
    auto jsTruthy = [&](const json &v) -> bool {
      if (v.is_null())
        return false;
      if (v.is_boolean())
        return v.get<bool>();
      if (v.is_number()) {
        double n = v.get<double>();
        return std::isfinite(n) && n != 0.0;
      }
      if (v.is_string())
        return !v.get<std::string>().empty();
      return true;
    };
    std::string sessionId;
    if (payload.contains("sessionId") && jsTruthy(payload["sessionId"])) {
      sessionId = jsStringValue(payload["sessionId"]);
    }
    if (sessionManager_)
      sessionId = sessionManager_->ensure(sessionId);
    if (sessionId.empty())
      sessionId = randomSessionId();

    json budget = payload.contains("budget") && payload["budget"].is_object()
                      ? payload["budget"]
                      : json::object();
    auto jsToNumber = [&](const json &v, bool &ok) -> double {
      ok = true;
      if (v.is_null()) {
        ok = false;
        return std::numeric_limits<double>::quiet_NaN();
      }
      if (v.is_number())
        return v.get<double>();
      if (v.is_boolean())
        return v.get<bool>() ? 1.0 : 0.0;
      if (v.is_string()) {
        std::string s = v.get<std::string>();
        auto start = s.find_first_not_of(" \n\r\t");
        if (start == std::string::npos)
          return 0.0;
        auto end = s.find_last_not_of(" \n\r\t");
        s = s.substr(start, end - start + 1);
        if (s.empty())
          return 0.0;
        try {
          return std::stod(s);
        } catch (...) {
          ok = false;
          return std::numeric_limits<double>::quiet_NaN();
        }
      }
      ok = false;
      return std::numeric_limits<double>::quiet_NaN();
    };
    auto pickValue = [&](const json &obj,
                         const std::vector<std::string> &keys) -> const json * {
      for (auto &k : keys) {
        if (obj.contains(k) && !obj[k].is_null())
          return &obj[k];
      }
      return nullptr;
    };
    auto numberOrHard = [&](const json *v, double paramFallback,
                            double hardFallback) -> double {
      double n = paramFallback;
      if (v) {
        bool ok = false;
        n = jsToNumber(*v, ok);
      }
      if (!std::isfinite(n) || n == 0.0)
        return hardFallback;
      return n;
    };
    auto numberFinite = [&](const json *v, double fallback) -> double {
      if (!v)
        return fallback;
      bool ok = false;
      double n = jsToNumber(*v, ok);
      return std::isfinite(n) ? n : fallback;
    };

    int depth = std::max(
        1,
        (int)numberOrHard(pickValue(budget, {"mappingDepth", "depth"}),
                          params_.mappingDepth ? params_.mappingDepth : 1, 1));
    int topMemesK = std::max(
        3,
        (int)numberOrHard(
            pickValue(budget, {"reflectionTopMemes", "topMemes"}),
            params_.reflectionTopMemes ? params_.reflectionTopMemes : 18, 18));
    int topWordsK = std::max(
        3,
        (int)numberOrHard(
            pickValue(budget, {"reflectionTopWords", "topWords"}),
            params_.reflectionTopWords ? params_.reflectionTopWords : 24, 24));
    double minScoreRaw =
        numberFinite(pickValue(budget, {"reflectionMinScore", "minScore"}),
                     params_.reflectionMinScore);
    double minScore = std::isfinite(minScoreRaw) ? minScoreRaw : 1e-6;
    int iteration = std::max(
        1, (int)numberOrHard(pickValue(budget, {"iteration"}),
                             params_.iteration ? params_.iteration : 5, 5));
    int radius = std::max(
        1,
        std::min(6, (int)numberOrHard(
                        pickValue(budget, {"radius", "windowRadius"}), 2, 2)));

    auto mapping = mapWordsToMemesOrdered(words);
    auto seeds = mapping.seeds;
    auto seedOrder = mapping.order;
    PropagationResult result;
    if (depth > 1) {
      for (int hop = 1; hop < depth; hop++) {
        result = runPropagation(seeds, json{{"iteration", iteration},
                                            {"radius", radius},
                                            {"seedOrder", seedOrder}});
        auto topMemes =
            pickTopActivatedMemes(result, seeds, topMemesK, minScore);
        std::unordered_map<std::string, double> wordScore;
        for (auto &m : topMemes) {
          auto linked = kvm_.getMemeWords(m.memeId);
          for (auto &w : linked) {
            if (w.empty())
              continue;
            auto prev = wordScore.count(w) ? wordScore[w] : 0.0;
            wordScore[w] = std::max(prev, m.score);
          }
        }
        std::vector<std::pair<std::string, double>> sorted(wordScore.begin(),
                                                           wordScore.end());
        std::sort(sorted.begin(), sorted.end(),
                  [](auto &a, auto &b) { return a.second > b.second; });
        std::vector<std::string> expanded;
        for (size_t i = 0;
             i < sorted.size() && (int)expanded.size() < topWordsK; i++)
          expanded.push_back(sorted[i].first);
        std::vector<std::string> merged;
        merged.reserve(std::min<size_t>(64, words.size()) + expanded.size());
        for (size_t i = 0; i < words.size() && i < 64; i++)
          merged.push_back(words[i]);
        for (auto &w : expanded)
          merged.push_back(w);
        std::unordered_set<std::string> dedup;
        std::vector<std::string> final;
        for (auto &w : merged)
          if (!w.empty() && !dedup.count(w)) {
            dedup.insert(w);
            final.push_back(w);
          }
        auto nextMapping = mapWordsToMemesOrdered(final);
        seeds = nextMapping.seeds;
        seedOrder = nextMapping.order;
      }
    }
    result = runPropagation(seeds, json{{"iteration", iteration},
                                        {"radius", radius},
                                        {"seedOrder", seedOrder}});
    std::string reply = composeReply(sessionId, result, words, seeds);
    auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - started)
                       .count();
    metrics_["requests"] = metrics_.value("requests", 0) + 1;
    metrics_["lastLatency"] = latency;
    metrics_["updatedAt"] =
        (int64_t)std::chrono::system_clock::now().time_since_epoch().count();

    json out;
    out["reply"] = reply;
    out["seeds"] = json::array();
    for (auto &id : seedOrder) {
      auto it = seeds.find(id);
      if (it != seeds.end())
        out["seeds"].push_back({id, it->second});
    }
    out["activation"] = result.activation;
    out["memes"] = result.windowInfo.ids;
    out["sessionId"] = sessionId;
    out["latency"] = latency;
    out["params"] = cloneParamsLocked();
    return out;
  }

  json metrics() const {
    std::lock_guard<std::mutex> lock(stateMu_);
    return metrics_;
  }

  json toSnapshot() override {
    std::lock_guard<std::mutex> lock(stateMu_);
    json out;
    out["params"] = cloneParamsLocked();
    out["graph"] = graph_.exportSnapshot();
    if (sessionManager_)
      out["sessions"] = sessionManager_->exportSessions();
    out["kvm"] = kvm_.exportEntries();
    out["surface"] = surfaceLexicon_->exportSnapshot(512);
    out["dialog"] = dialogMemory_->exportSnapshot(512);
    return out;
  }

  void fromSnapshot(const json &snapshot) override {
    std::lock_guard<std::mutex> lock(stateMu_);
    if (snapshot.is_null())
      return;
    if (snapshot.contains("params"))
      setParamsLocked(snapshot["params"]);
    graph_.importSnapshot(snapshot.value("graph", json::object()));
    if (snapshot.contains("sessions") && sessionManager_)
      sessionManager_->importSessions(snapshot["sessions"]);
    if (snapshot.contains("kvm")) {
      for (auto &kv : snapshot["kvm"]["words"].items()) {
        for (auto &m : kv.value())
          kvm_.link(kv.key(), m.get<std::string>());
      }
    }
    if (snapshot.contains("surface"))
      surfaceLexicon_->importSnapshot(snapshot["surface"]);
    if (snapshot.contains("dialog"))
      dialogMemory_->importSnapshot(snapshot["dialog"]);
  }

  MemeGraph &graph() { return graph_; }
  KVMStore &kvm() { return kvm_; }
  MemeSurfaceLexicon &surfaceLexicon() { return *surfaceLexicon_; }
  DialogMemory &dialogMemory() { return *dialogMemory_; }
  const Config &config() const override { return config_; }

  std::vector<std::string> getMemeWords(const std::string &memeId) override {
    return kvm_.getMemeWords(memeId);
  }

  void graphLink(const std::string &fromId, const std::string &toId,
                 double weight, int direction) override {
    graph_.link(fromId, toId, weight, direction);
  }

private:
  json cloneParamsLocked() const {
    json j;
    j["decayFactor"] = params_.decayFactor;
    j["maxMemeWords"] = params_.maxMemeWords;
    j["minOverlapThreshold"] = params_.minOverlapThreshold;
    j["memeNgramMin"] = params_.memeNgramMin;
    j["memeNgramMax"] = params_.memeNgramMax;
    j["maliciousThreshold"] = params_.maliciousThreshold;
    j["learningIterations"] = params_.learningIterations;
    j["iteration"] = params_.iteration;
    j["threshold"] = params_.threshold;
    j["decay"] = params_.decay;
    j["decayK"] = params_.decayK;
    j["maxLen"] = params_.maxLen;
    j["edgeWeight"] = params_.edgeWeight;
    j["activationType"] = params_.activationType;
    j["transferType"] = params_.transferType;
    j["activationCustom"] = params_.activationCustom;
    j["transferCustom"] = params_.transferCustom;
    j["mappingDepth"] = params_.mappingDepth;
    j["reflectionTopMemes"] = params_.reflectionTopMemes;
    j["reflectionTopWords"] = params_.reflectionTopWords;
    j["reflectionMinScore"] = params_.reflectionMinScore;
    return j;
  }

  std::shared_ptr<RobotsCorpus> ensureRobotsCorpusLocked() const {
    if (!robotsCorpus_) {
      robotsCorpus_ = std::make_shared<RobotsCorpus>(
          config_.robotsDir, config_.lemmaCsv, config_.lemmaAutoload,
          config_.lemmaMaxBytes, config_.lemmaForce,
          config_.robotsChunkMinWords, config_.robotsChunkMaxWords);
    }
    return robotsCorpus_;
  }

public:

  json getSearchConfig() const override {
    std::lock_guard<std::mutex> lock(stateMu_);
    return searchConfig_.toJson();
  }

  json setSearchConfig(const json &patch) override {
    std::lock_guard<std::mutex> lock(stateMu_);
    auto trim = [](std::string s) {
      auto start = s.find_first_not_of(" \n\r\t");
      if (start == std::string::npos)
        return std::string();
      auto end = s.find_last_not_of(" \n\r\t");
      return s.substr(start, end - start + 1);
    };

    if (patch.contains("enabled") && patch["enabled"].is_boolean()) {
      searchConfig_.enabled = patch["enabled"].get<bool>();
    }
    if (patch.contains("endpoints") && patch["endpoints"].is_array()) {
      searchConfig_.endpoints.clear();
      std::unordered_set<std::string> seen;
      for (auto &v : patch["endpoints"]) {
        std::string s;
        if (v.is_string())
          s = v.get<std::string>();
        else if (v.is_number() || v.is_boolean())
          s = v.dump();
        else if (v.is_null())
          s.clear();
        else
          s = v.dump();
        s = trim(s);
        if (!s.empty() && !seen.count(s)) {
          seen.insert(s);
          searchConfig_.endpoints.push_back(s);
        }
      }
    }
    if (patch.contains("active") && patch["active"].is_string()) {
      searchConfig_.active = trim(patch["active"].get<std::string>());
      if (!searchConfig_.active.empty() &&
          std::find(searchConfig_.endpoints.begin(),
                    searchConfig_.endpoints.end(),
                    searchConfig_.active) == searchConfig_.endpoints.end()) {
        searchConfig_.endpoints.push_back(searchConfig_.active);
      }
    }

    config_.searchEndpoint = searchConfig_.active;
    if (!researcher_)
      researcher_ =
          std::make_shared<OnlineResearcher>(this, searchConfig_.active);
    researcher_->setEnabled(searchConfig_.enabled);
    researcher_->setEndpoint(searchConfig_.active);
    return searchConfig_.toJson();
  }

  std::vector<std::string> listRobotsFiles() const override {
    std::lock_guard<std::mutex> lock(stateMu_);
    return ensureRobotsCorpusLocked()->listFiles();
  }

  std::vector<RobotsDocument>
  collectRobotsDocuments(const json &options) const override {
    std::lock_guard<std::mutex> lock(stateMu_);
    return ensureRobotsCorpusLocked()->collect(options);
  }

  std::vector<std::string>
  lemmatizeTokens(const std::vector<std::string> &tokens) {
    std::lock_guard<std::mutex> lock(stateMu_);
    return ensureRobotsCorpusLocked()->normalizeWords(tokens);
  }

  json ingestDocument(const json &doc) override {
    std::lock_guard<std::mutex> lock(stateMu_);
    std::string text = doc.value("text", "");
    std::vector<std::string> tokens;
    if (doc.contains("tokens") && doc["tokens"].is_array()) {
      std::function<std::string(const json &)> jsStringForJoin;
      jsStringForJoin = [&](const json &v) -> std::string {
        if (v.is_null())
          return "";
        if (v.is_string())
          return v.get<std::string>();
        if (v.is_boolean())
          return v.get<bool>() ? "true" : "false";
        if (v.is_number())
          return v.dump();
        if (v.is_array()) {
          std::string out;
          bool first = true;
          for (auto &item : v) {
            if (!first)
              out += ",";
            first = false;
            out += jsStringForJoin(item);
          }
          return out;
        }
        return "[object Object]";
      };
      auto jsStringOrEmpty = [&](const json &v) -> std::string {
        if (v.is_null())
          return "";
        if (v.is_boolean())
          return v.get<bool>() ? "true" : "";
        if (v.is_number()) {
          double n = v.get<double>();
          if (n == 0.0)
            return "";
          return v.dump();
        }
        if (v.is_string())
          return v.get<std::string>();
        if (v.is_array()) {
          std::string out;
          bool first = true;
          for (auto &item : v) {
            if (!first)
              out += ",";
            first = false;
            out += jsStringForJoin(item);
          }
          return out;
        }
        return "[object Object]";
      };
      for (auto &t : doc["tokens"]) {
        std::string s = jsStringOrEmpty(t);
        if (!s.empty())
          tokens.push_back(s);
      }
    }
    if (tokens.empty() && !text.empty())
      tokens = tokenize(text);
    if (text.empty() && tokens.empty())
      return json{{"ok", false}, {"reason", "empty-text"}};
    if (tokens.empty())
      return json{{"ok", false}, {"reason", "no-tokens"}};

    auto memeIds = buildMemeSequenceFromTokens(tokens);
    for (size_t i = 0; i + 1 < memeIds.size(); i++) {
      graph_.link(memeIds[i], memeIds[i + 1], 1.0, 0);
    }
    std::string source = doc.value("source", "");
    for (auto &memeId : memeIds) {
      json meta = graph_.getMeta(memeId);
      meta["lastTouched"] = nowMs();
      meta["degree"] = graph_.degreeOf(memeId);
      if (!source.empty()) {
        if (!meta.contains("sources") || !meta["sources"].is_object())
          meta["sources"] = json::object();
        meta["sources"][source] = meta["sources"].value(source, 0) + 1;
      }
      graph_.setMeta(memeId, meta);
    }
    corpusStats_["ingested"] = corpusStats_.value("ingested", 0) + 1;
    corpusStats_["lastIngest"] = nowMs();
    return json{{"ok", true},
                {"memes", (int)memeIds.size()},
                {"edges", (int)std::max<int>(0, (int)memeIds.size() - 1)},
                {"source", source.empty() ? json(nullptr) : json(source)}};
  }

  json forgetMemes(const json &payload) override {
    std::lock_guard<std::mutex> lock(stateMu_);
    int64_t threshold = payload.contains("olderThan")
                            ? payload.value("olderThan", 0LL)
                            : (nowMs() - 7LL * 24LL * 60LL * 60LL * 1000LL);
    int limit = payload.value("limit", 64);
    json removed = json::array();
    for (auto &kv : graph_.metaEntries()) {
      auto &meta = kv.second;
      if (meta.value("lastTouched", 0LL) >= threshold)
        continue;
      auto words = kvm_.getMemeWords(kv.first);
      for (auto &w : words)
        kvm_.unlink(w, kv.first);
      if (graph_.removeNode(kv.first)) {
        json ws = json::array();
        for (auto &w : words)
          ws.push_back(w);
        removed.push_back(json{{"memeId", kv.first}, {"words", ws}});
      }
      if (limit > 0 && (int)removed.size() >= limit)
        break;
    }
    return json{{"ok", true}, {"removed", removed}};
  }

  json onlineLookup(const std::string &query, const json &options) override {
    return onlineLookup(json(query), options);
  }

  json onlineLookup(const json &input, const json &options) override {
    std::shared_ptr<OnlineResearcher> r;
    {
      std::lock_guard<std::mutex> lock(stateMu_);
      if (!researcher_) {
        researcher_ =
            std::make_shared<OnlineResearcher>(this, searchConfig_.active);
        researcher_->setEnabled(searchConfig_.enabled);
      }
      r = researcher_;
    }
    return r->lookup(input, options);
  }

  fs::path exportGraphToFile(const json &options) override {
    ensureDir(config_.exportDir);
    auto jsToNumber = [&](const json &v) -> double {
      if (v.is_null())
        return std::numeric_limits<double>::quiet_NaN();
      if (v.is_number())
        return v.get<double>();
      if (v.is_boolean())
        return v.get<bool>() ? 1.0 : 0.0;
      if (v.is_string()) {
        std::string s = v.get<std::string>();
        auto start = s.find_first_not_of(" \n\r\t");
        if (start == std::string::npos)
          return 0.0;
        auto end = s.find_last_not_of(" \n\r\t");
        s = s.substr(start, end - start + 1);
        if (s.empty())
          return 0.0;
        try {
          return std::stod(s);
        } catch (...) {
          return std::numeric_limits<double>::quiet_NaN();
        }
      }
      return std::numeric_limits<double>::quiet_NaN();
    };
    auto asBool = [](const json &v, bool fallback) {
      if (v.is_boolean())
        return v.get<bool>();
      if (v.is_number())
        return v.get<double>() != 0.0;
      if (v.is_string()) {
        std::string s = v.get<std::string>();
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char ch) { return (char)std::tolower(ch); });
        if (s == "1" || s == "true" || s == "yes" || s == "on")
          return true;
        if (s == "0" || s == "false" || s == "no" || s == "off")
          return false;
      }
      return fallback;
    };

    int radius = 2;
    if (options.contains("radius")) {
      double r = jsToNumber(options["radius"]);
      if (!std::isfinite(r))
        radius = 0;
      else
        radius = (int)r;
    }
    radius = std::max(0, std::min(8, radius));

    int topKNodes = 0;
    if (options.contains("topKNodes")) {
      double v = jsToNumber(options["topKNodes"]);
      if (std::isfinite(v) && v > 0)
        topKNodes = std::max(1, std::min(100000, (int)v));
    }
    double minActivation = -std::numeric_limits<double>::infinity();
    if (options.contains("minActivation")) {
      double v = jsToNumber(options["minActivation"]);
      if (std::isfinite(v))
        minActivation = v;
    }
    std::string format = options.value("format", "json");
    std::transform(format.begin(), format.end(), format.begin(),
                   [](unsigned char ch) { return (char)std::tolower(ch); });
    if (format != "json" && format != "jsonl" && format != "csv")
      format = "json";
    bool includeMeta = asBool(options.value("includeMeta", json()), true);
    bool allowAbsolute =
        asBool(options.value("allowAbsolutePath", json()), false);

    std::unordered_map<std::string, double> usedSeeds;
    std::vector<std::string> seedOrder;
    if (options.contains("seeds") && options["seeds"].is_array()) {
      std::vector<std::string> seedWords;
      for (auto &s : options["seeds"]) {
        if (s.is_string())
          seedWords.push_back(s.get<std::string>());
        else if (s.is_number())
          seedWords.push_back(s.dump());
        else if (s.is_boolean())
          seedWords.push_back(s.get<bool>() ? "true" : "false");
      }
      auto mapping = mapWordsToMemesOrdered(seedWords);
      usedSeeds = mapping.seeds;
      seedOrder = mapping.order;
    } else {
      auto mapping = mapWordsToMemesOrdered(std::vector<std::string>{});
      usedSeeds = mapping.seeds;
      seedOrder = mapping.order;
    }
    std::vector<std::string> seedIds;
    if (!seedOrder.empty()) {
      seedIds = seedOrder;
    } else {
      for (auto &kv : usedSeeds)
        seedIds.push_back(kv.first);
    }
    auto win = graph_.buildWindow(seedIds, radius);
    std::vector<float> seedVec(win.ids.size(), 0.0f);
    for (auto &kv : usedSeeds) {
      auto it = win.index.find(kv.first);
      if (it != win.index.end())
        seedVec[it->second] = (float)kv.second;
    }
    auto actFn = activationFn(params_);
    int iteration = params_.iteration ? params_.iteration : 5;
    if (iteration == 0)
      iteration = 5;
    auto activation = tensor_.iteratePropagation(win.csr, seedVec, iteration,
                                                 actFn, params_.decayK, 0.02);
    auto graphObj = GraphExportBuilder::fromWindow(win, &activation);

    if (graphObj.contains("Nodes") && graphObj["Nodes"].is_array()) {
      std::vector<json> nodes;
      nodes.reserve(graphObj["Nodes"].size());
      for (auto &n : graphObj["Nodes"]) {
        if (!n.is_object())
          continue;
        double value = n.value("value", 0.0);
        if (value < minActivation)
          continue;
        nodes.push_back(n);
      }
      std::sort(nodes.begin(), nodes.end(), [](const json &a, const json &b) {
        return a.value("value", 0.0) > b.value("value", 0.0);
      });
      if (topKNodes > 0 && (int)nodes.size() > topKNodes)
        nodes.resize(topKNodes);

      std::unordered_set<std::string> keepIds;
      for (auto &n : nodes) {
        std::string id = n.value("id", "");
        if (!id.empty())
          keepIds.insert(id);
      }

      json outNodes = json::array();
      for (auto &n : nodes)
        outNodes.push_back(n);
      graphObj["Nodes"] = outNodes;

      if (graphObj.contains("Edges") && graphObj["Edges"].is_array()) {
        json outEdges = json::array();
        for (auto &e : graphObj["Edges"]) {
          if (!e.is_object())
            continue;
          std::string from = e.value("from", "");
          std::string to = e.value("to", "");
          if (keepIds.count(from) && keepIds.count(to))
            outEdges.push_back(e);
        }
        graphObj["Edges"] = outEdges;
      }
    }

    if (includeMeta) {
      json meta = json::object();
      meta["createdAt"] = nowIso();
      meta["radius"] = radius;
      meta["iteration"] = iteration;
      meta["seedCount"] = (int)usedSeeds.size();
      meta["seeds"] = json::array();
      for (auto &sid : seedIds) {
        meta["seeds"].push_back(
            json{{"id", sid},
                 {"value", usedSeeds.count(sid) ? usedSeeds[sid] : 0.0}});
      }
      graphObj["Meta"] = meta;
    }

    std::string fileArg = options.value("file", "");
    fs::path outFile;
    if (!fileArg.empty()) {
      fs::path candidate = fs::path(fileArg);
      if (candidate.is_absolute()) {
        if (allowAbsolute)
          outFile = candidate;
        else
          outFile = config_.exportDir / candidate.filename();
      } else {
        outFile = config_.exportDir / candidate;
      }
    } else {
      std::string ts = nowIso();
      std::replace(ts.begin(), ts.end(), ':', '-');
      std::replace(ts.begin(), ts.end(), '.', '-');
      std::string ext =
          (format == "json" ? ".json"
                            : (format == "jsonl" ? ".jsonl" : ".csv"));
      outFile = config_.exportDir / ("graph_export_" + ts + ext);
    }

    if (outFile.extension().empty()) {
      std::string ext =
          (format == "json" ? ".json"
                            : (format == "jsonl" ? ".jsonl" : ".csv"));
      outFile += ext;
    }

    ensureDir(outFile.parent_path());
    fs::path tmp = outFile;
    tmp += ".tmp";

    if (format == "json") {
      std::ofstream o(tmp, std::ios::trunc);
      o << graphObj.dump(2);
    } else if (format == "jsonl") {
      std::ofstream o(tmp, std::ios::trunc);
      if (graphObj.contains("Nodes") && graphObj["Nodes"].is_array()) {
        for (auto &n : graphObj["Nodes"]) {
          o << json{{"type", "node"}, {"data", n}}.dump() << "\n";
        }
      }
      if (graphObj.contains("Edges") && graphObj["Edges"].is_array()) {
        for (auto &e : graphObj["Edges"]) {
          o << json{{"type", "edge"}, {"data", e}}.dump() << "\n";
        }
      }
      if (graphObj.contains("Meta"))
        o << json{{"type", "meta"}, {"data", graphObj["Meta"]}}.dump() << "\n";
    } else {
      std::ofstream nodesFile(tmp, std::ios::trunc);
      nodesFile << "id,x,y,z,value\n";
      if (graphObj.contains("Nodes") && graphObj["Nodes"].is_array()) {
        for (auto &n : graphObj["Nodes"]) {
          if (!n.is_object())
            continue;
          auto esc = [](std::string s) {
            std::replace(s.begin(), s.end(), '"', '\'');
            return s;
          };
          nodesFile << '"' << esc(n.value("id", "")) << "\","
                    << n.value("x", 0.0) << "," << n.value("y", 0.0) << ","
                    << n.value("z", 0.0) << "," << n.value("value", 0.0)
                    << "\n";
        }
      }
      fs::path edgeFile = outFile;
      edgeFile.replace_extension(".edges.csv");
      std::ofstream edges(edgeFile, std::ios::trunc);
      edges << "from,to,weight,dir\n";
      if (graphObj.contains("Edges") && graphObj["Edges"].is_array()) {
        for (auto &e : graphObj["Edges"]) {
          if (!e.is_object())
            continue;
          auto esc = [](std::string s) {
            std::replace(s.begin(), s.end(), '"', '\'');
            return s;
          };
          edges << '"' << esc(e.value("from", "")) << "\",\""
                << esc(e.value("to", "")) << "\"," << e.value("weight", 0.0)
                << "," << e.value("dir", 0) << "\n";
        }
      }
    }

    if (fs::exists(outFile))
      fs::remove(outFile);
    fs::rename(tmp, outFile);
    return outFile;
  }

  json learnFromDialog(const json &payload, const json &result) override {
    std::lock_guard<std::mutex> lock(stateMu_);
    try {
      json actualPayload = payload;
      json actualResult = result;
      if (payload.is_object() && payload.contains("payload") &&
          payload.contains("result")) {
        if (result.is_null() || (result.is_object() && result.empty())) {
          actualPayload = payload.value("payload", json::object());
          actualResult = payload.value("result", json::object());
        }
      }
      auto trim = [](const std::string &s) {
        auto start = s.find_first_not_of(" \n\r\t");
        if (start == std::string::npos)
          return std::string();
        auto end = s.find_last_not_of(" \n\r\t");
        return s.substr(start, end - start + 1);
      };
      std::function<std::string(const json &)> jsStringForJoin;
      jsStringForJoin = [&](const json &v) -> std::string {
        if (v.is_null())
          return "";
        if (v.is_string())
          return v.get<std::string>();
        if (v.is_boolean())
          return v.get<bool>() ? "true" : "false";
        if (v.is_number())
          return v.dump();
        if (v.is_array()) {
          std::string out;
          bool first = true;
          for (auto &item : v) {
            if (!first)
              out += ",";
            first = false;
            out += jsStringForJoin(item);
          }
          return out;
        }
        return "[object Object]";
      };

      std::string question;
      if (actualPayload.contains("text") && actualPayload["text"].is_string()) {
        question = actualPayload["text"].get<std::string>();
      } else if (actualPayload.contains("tokens") &&
                 actualPayload["tokens"].is_array()) {
        std::ostringstream oss;
        bool first = true;
        for (auto &t : actualPayload["tokens"]) {
          std::string token = jsStringForJoin(t);
          if (!first)
            oss << " ";
          oss << token;
          first = false;
        }
        question = oss.str();
      }

      std::string reply =
          actualResult.contains("reply") && actualResult["reply"].is_string()
              ? actualResult["reply"].get<std::string>()
              : std::string();

      if (trim(question).empty() || trim(reply).empty()) {
        return json{{"ok", false}, {"reason", "missing-text"}};
      }
      if (isLikelyGibberishReply(reply)) {
        return json{{"ok", false}, {"reason", "gibberish-reply-skipped"}};
      }

      double residual = 0.0;
      double residualGate = 1.0;
      try {
        auto qEmb = textToMiniEmbedding(question, 128);
        auto rEmb = textToMiniEmbedding(reply, 128);
        double cos = cosineSim(qEmb, rEmb);
        double sim = clamp01((cos + 1.0) * 0.5);
        residual = 1.0 - sim;
        residualGate = std::max(0.15, 1.0 - residual);
        residualStats_.count += 1;
        residualStats_.last = residual;
        if (residualStats_.count == 1) {
          residualStats_.ema = residual;
        } else {
          residualStats_.ema =
              residualStats_.ema * (1.0 - residualStats_.alpha) +
              residual * residualStats_.alpha;
        }
        if (residualStats_.count % std::max(1, residualStats_.adjustEvery) ==
            0) {
          if (residualStats_.ema > 0.55) {
            params_.iteration =
                std::min(12, std::max(3, params_.iteration) + 1);
            params_.decayK = std::max(0.5, params_.decayK - 0.05);
          } else if (residualStats_.ema < 0.30) {
            params_.iteration = std::max(3, std::max(3, params_.iteration) - 1);
            params_.decayK = std::min(2.0, params_.decayK + 0.05);
          }
        }
      } catch (...) {
        residual = 0.0;
        residualGate = 1.0;
      }

      std::unordered_map<std::string, double> seeds;
      std::vector<std::string> seedOrder;
      auto jsToNumber = [&](const json &v) -> double {
        if (v.is_null())
          return 0.0;
        if (v.is_number())
          return v.get<double>();
        if (v.is_boolean())
          return v.get<bool>() ? 1.0 : 0.0;
        if (v.is_string()) {
          std::string s = v.get<std::string>();
          auto start = s.find_first_not_of(" \n\r\t");
          if (start == std::string::npos)
            return 0.0;
          auto end = s.find_last_not_of(" \n\r\t");
          s = s.substr(start, end - start + 1);
          if (s.empty())
            return 0.0;
          try {
            return std::stod(s);
          } catch (...) {
            return std::numeric_limits<double>::quiet_NaN();
          }
        }
        return std::numeric_limits<double>::quiet_NaN();
      };

      if (actualResult.contains("seeds") && actualResult["seeds"].is_array()) {
        for (auto &m : actualResult["seeds"]) {
          if (m.is_array() && m.size() >= 2 && m[0].is_string()) {
            std::string id = m[0].get<std::string>();
            seeds[id] = jsToNumber(m[1]);
            seedOrder.push_back(id);
          }
        }
      } else {
        auto mapping = mapWordsToMemesOrdered(tokenize(question));
        seeds = mapping.seeds;
        seedOrder = mapping.order;
      }

      PropagationResult resObj;
      if (actualResult.contains("memes") && actualResult["memes"].is_array() &&
          actualResult.contains("activation") &&
          actualResult["activation"].is_array()) {
        for (auto &m : actualResult["memes"])
          if (m.is_string())
            resObj.windowInfo.ids.push_back(m.get<std::string>());
        for (size_t i = 0; i < resObj.windowInfo.ids.size(); i++)
          resObj.windowInfo.index[resObj.windowInfo.ids[i]] = (uint32_t)i;
        for (auto &a : actualResult["activation"])
          resObj.activation.push_back((float)jsToNumber(a));
      } else {
        resObj = runPropagation(seeds, seedOrder.empty()
                                           ? json::object()
                                           : json{{"seedOrder", seedOrder}});
      }

      auto topMemes = pickTopActivatedMemes(resObj, seeds, 18, 1e-6);
      auto signature = makeSignatureFromTopMemes(topMemes, 12);
      std::vector<std::string> memeIds;
      for (auto &m : topMemes)
        memeIds.push_back(m.memeId);

      for (size_t i = 0; i < topMemes.size() && i < 10; i++) {
        double w = std::max(0.5, std::min(3.0, topMemes[i].score));
        w = std::max(0.2, w * residualGate);
        surfaceLexicon_->learn(topMemes[i].memeId, reply, w);
      }
      std::string sessionId;
      if (actualPayload.contains("sessionId") &&
          actualPayload["sessionId"].is_string())
        sessionId = actualPayload["sessionId"].get<std::string>();
      if (sessionId.empty())
        return json{{"ok", false}, {"reason", "missing-session"}};

      double hint = topMemes.empty() ? 0.0 : topMemes.front().score;
      dialogMemory_->remember(sessionId, signature, memeIds, question, reply,
                              hint * residualGate);
      int sigLen = 0;
      if (!signature.empty()) {
        sigLen = 1;
        for (char c : signature)
          if (c == '|')
            sigLen++;
      }
      return json{{"ok", true},
                  {"memes", (int)memeIds.size()},
                  {"signatureLen", sigLen},
                  {"residual", residual},
                  {"residualEma", residualStats_.ema},
                  {"residualGate", residualGate}};
    } catch (const std::exception &e) {
      return json{{"ok", false}, {"error", e.what()}};
    }
  }

private:
  mutable std::mutex stateMu_;
  friend class OnlineResearcher;
  Config config_;
  KVMStore kvm_;
  MemeGraph graph_;
  TensorEngine tensor_;
  std::shared_ptr<KeyValueStore> sessions_;
  std::shared_ptr<SessionManager> sessionManager_;
  std::shared_ptr<NamespacedStore> surfaceStore_;
  std::shared_ptr<NamespacedStore> dialogStore_;
  std::shared_ptr<MemeSurfaceLexicon> surfaceLexicon_;
  std::shared_ptr<DialogMemory> dialogMemory_;
  mutable std::shared_ptr<RobotsCorpus> robotsCorpus_;
  ModelDefaults params_;
  struct ResidualStats {
    double ema{0.0};
    double last{0.0};
    double alpha{0.05};
    int64_t count{0};
    int adjustEvery{20};
  } residualStats_;
  json metrics_ = json{{"requests", 0}, {"lastLatency", 0}, {"updatedAt", 0}};
  json corpusStats_ = json{{"ingested", 0}, {"lastIngest", nullptr}};
  SearchConfig searchConfig_;
  std::shared_ptr<OnlineResearcher> researcher_;
  PatternMatrix pattern_;
  std::shared_ptr<addon::AddonManager> addons_;

  struct PropagationResult {
    MemeGraph::WindowInfo windowInfo;
    std::vector<float> seedVector;
    std::vector<float> activation;
  };

  struct TopMeme {
    std::string memeId;
    double score{0.0};
  };

  static int64_t nowMs() {
    return (int64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
  }

  PropagationResult
  runPropagation(const std::unordered_map<std::string, double> &seeds,
                 const json &options) {
    PropagationResult result;
    auto jsToNumber = [&](const json &v) -> double {
      if (v.is_null())
        return std::numeric_limits<double>::quiet_NaN();
      if (v.is_number())
        return v.get<double>();
      if (v.is_boolean())
        return v.get<bool>() ? 1.0 : 0.0;
      if (v.is_string()) {
        std::string s = v.get<std::string>();
        auto start = s.find_first_not_of(" \n\r\t");
        if (start == std::string::npos)
          return 0.0;
        auto end = s.find_last_not_of(" \n\r\t");
        s = s.substr(start, end - start + 1);
        if (s.empty())
          return 0.0;
        try {
          return std::stod(s);
        } catch (...) {
          return std::numeric_limits<double>::quiet_NaN();
        }
      }
      return std::numeric_limits<double>::quiet_NaN();
    };
    auto numberOrHard = [&](const json *v, double paramFallback,
                            double hardFallback) -> double {
      double n = paramFallback;
      if (v)
        n = jsToNumber(*v);
      if (!std::isfinite(n) || n == 0.0)
        return hardFallback;
      return n;
    };
    const json *radiusVal =
        options.contains("radius") && !options["radius"].is_null()
            ? &options["radius"]
            : (options.contains("windowRadius") &&
                       !options["windowRadius"].is_null()
                   ? &options["windowRadius"]
                   : nullptr);
    int radius = std::max(1, std::min(6, (int)numberOrHard(radiusVal, 2, 2)));
    int iteration = std::max(
        1, (int)numberOrHard(options.contains("iteration") &&
                                     !options["iteration"].is_null()
                                 ? &options["iteration"]
                                 : nullptr,
                             params_.iteration ? params_.iteration : 5, 5));
    std::vector<std::string> seedIds;
    if (options.contains("seedOrder") && options["seedOrder"].is_array()) {
      for (auto &v : options["seedOrder"])
        if (v.is_string())
          seedIds.push_back(v.get<std::string>());
    }
    if (seedIds.empty()) {
      seedIds.reserve(seeds.size());
      for (auto &kv : seeds)
        seedIds.push_back(kv.first);
    }
    result.windowInfo = graph_.buildWindow(seedIds, radius);
    result.seedVector.assign(result.windowInfo.ids.size(), 0.0f);
    for (auto &kv : seeds) {
      auto it = result.windowInfo.index.find(kv.first);
      if (it != result.windowInfo.index.end())
        result.seedVector[it->second] = (float)kv.second;
    }
    auto actFn = activationFn(params_);
    result.activation =
        tensor_.iteratePropagation(result.windowInfo.csr, result.seedVector,
                                   iteration, actFn, params_.decayK, 0.02);
    pattern_.rebuild(result.windowInfo);
    return result;
  }

  std::vector<TopMeme>
  pickTopActivatedMemes(const PropagationResult &result,
                        const std::unordered_map<std::string, double> &seeds,
                        int limit, double minScore) {
    std::vector<TopMeme> scored;
    std::unordered_set<std::string> seedIds;
    for (auto &kv : seeds)
      seedIds.insert(kv.first);

    auto isConnectedToSeeds = [&](const std::string &memeId) {
      if (seedIds.empty() || seedIds.count(memeId))
        return true;
      for (auto &n : graph_.neighborsOf(memeId)) {
        if (seedIds.count(n.first))
          return true;
      }
      for (auto &sid : seedIds) {
        for (auto &n : graph_.neighborsOf(sid)) {
          if (n.first == memeId)
            return true;
        }
      }
      return false;
    };

    for (size_t i = 0; i < result.windowInfo.ids.size(); i++) {
      if (i >= result.activation.size())
        break;
      double score = result.activation[i];
      if (!std::isfinite(score) || score <= minScore)
        continue;
      const auto &memeId = result.windowInfo.ids[i];
      if (!isConnectedToSeeds(memeId))
        continue;
      scored.push_back({memeId, score});
    }
    std::sort(scored.begin(), scored.end(),
              [](auto &a, auto &b) { return a.score > b.score; });
    if ((int)scored.size() > limit)
      scored.resize(limit);
    return scored;
  }

  std::string makeSignatureFromTopMemes(const std::vector<TopMeme> &topMemes,
                                        int limit) {
    std::vector<std::string> ids;
    int cap = std::max(3, limit);
    for (auto &m : topMemes) {
      if (!m.memeId.empty())
        ids.push_back(m.memeId);
      if ((int)ids.size() >= cap)
        break;
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    std::ostringstream oss;
    for (size_t i = 0; i < ids.size(); i++) {
      if (i)
        oss << "|";
      oss << ids[i];
    }
    return oss.str();
  }

  std::vector<std::string>
  buildMemeSequenceFromTokens(const std::vector<std::string> &tokens) {
    std::vector<std::string> list;
    auto trim = [](const std::string &s) {
      auto start = s.find_first_not_of(" \n\r\t");
      if (start == std::string::npos)
        return std::string();
      auto end = s.find_last_not_of(" \n\r\t");
      return s.substr(start, end - start + 1);
    };
    for (auto &t : tokens) {
      std::string s = trim(t);
      if (!s.empty())
        list.push_back(s);
    }
    int nMin = std::max(2, params_.memeNgramMin);
    int nMax = std::max(nMin, params_.memeNgramMax);
    int minOverlap = std::max(1, params_.minOverlapThreshold);
    int maxWordSet = std::max(4, params_.maxMemeWords);

    auto resolveOrCreate =
        [&](const std::vector<std::string> &tokenSet) -> std::string {
      std::vector<std::string> uniq;
      std::unordered_set<std::string> seen;
      for (auto &t : tokenSet) {
        if (t.empty())
          continue;
        if (seen.insert(t).second)
          uniq.push_back(t);
      }
      if ((int)uniq.size() > maxWordSet)
        uniq.resize(maxWordSet);
      if (uniq.size() <= 1)
        return uniq.empty() ? "" : ("meme_" + sha1Hex(uniq[0]));

      std::unordered_map<std::string, int> counts;
      for (auto &w : uniq) {
        for (auto &mid : kvm_.getWordMemeSet(w))
          counts[mid]++;
      }
      std::string best;
      int bestOverlap = 0;
      for (auto &kv : counts) {
        if (kv.second > bestOverlap) {
          bestOverlap = kv.second;
          best = kv.first;
        }
      }
      if (!best.empty() && bestOverlap >= minOverlap) {
        for (auto &w : uniq)
          kvm_.link(w, best);
        graph_.ensureNode(best);
        return best;
      }
      std::vector<std::string> sorted = uniq;
      std::sort(sorted.begin(), sorted.end());
      std::string memeId = "meme_p_" + sha1Hex([&]() {
                             std::string join;
                             for (size_t i = 0; i < sorted.size(); i++) {
                               if (i)
                                 join += "|";
                               join += sorted[i];
                             }
                             return join;
                           }());
      graph_.ensureNode(memeId);
      for (auto &w : sorted)
        kvm_.link(w, memeId);
      return memeId;
    };

    std::vector<std::string> seq;
    for (size_t i = 0; i < list.size(); i++) {
      std::string picked;
      for (int n = nMax; n >= nMin; n--) {
        if (i + n > list.size())
          continue;
        picked = resolveOrCreate(
            std::vector<std::string>(list.begin() + i, list.begin() + i + n));
        if (!picked.empty())
          break;
      }
      if (picked.empty()) {
        auto &w = list[i];
        if (!w.empty()) {
          picked = "meme_" + sha1Hex(w);
          graph_.ensureNode(picked);
          kvm_.link(w, picked);
        }
      }
      if (!picked.empty() && (seq.empty() || seq.back() != picked))
        seq.push_back(picked);
    }
    return seq;
  }

  std::string
  composeReply(const std::string &sessionId, const PropagationResult &result,
               const std::vector<std::string> &words,
               const std::unordered_map<std::string, double> &seeds) {
    auto topMemes = pickTopActivatedMemes(result, seeds, 18, 1e-6);
    std::string signature = makeSignatureFromTopMemes(topMemes, 12);

    std::vector<std::string> topIds;
    for (auto &m : topMemes)
      topIds.push_back(m.memeId);
    json memoryHit = dialogMemory_
                         ? dialogMemory_->retrieve(sessionId, topIds, signature)
                         : json();
    if (!memoryHit.is_null() && memoryHit.contains("reply") &&
        memoryHit["reply"].is_string()) {
      auto r = memoryHit["reply"].get<std::string>();
      auto start = r.find_first_not_of(" \n\r\t");
      if (start != std::string::npos) {
        auto end = r.find_last_not_of(" \n\r\t");
        r = r.substr(start, end - start + 1);
      } else {
        r.clear();
      }
      if (!r.empty())
        return r;
    }

    std::unordered_map<std::string, double> phraseScores;
    for (auto &m : topMemes) {
      for (auto &p : surfaceLexicon_->getTop(m.memeId, 4)) {
        std::string key = p.phrase;
        auto start = key.find_first_not_of(" \n\r\t");
        if (start == std::string::npos)
          continue;
        auto end = key.find_last_not_of(" \n\r\t");
        key = key.substr(start, end - start + 1);
        if (key.empty())
          continue;
        phraseScores[key] +=
            std::max(0.0, m.score) * (0.5 + std::max(0.0, p.weight));
      }
    }
    if (!phraseScores.empty()) {
      std::vector<std::pair<std::string, double>> ordered(phraseScores.begin(),
                                                          phraseScores.end());
      std::sort(ordered.begin(), ordered.end(),
                [](auto &a, auto &b) { return a.second > b.second; });
      if (ordered.size() > 2)
        ordered.resize(2);
      std::string out;
      for (size_t i = 0; i < ordered.size(); i++) {
        if (i)
          out += "。";
        out += ordered[i].first;
      }
      return out;
    }

    std::unordered_set<std::string> seedIds;
    for (auto &kv : seeds)
      seedIds.insert(kv.first);
    auto isConnectedToSeeds = [&](const std::string &memeId) {
      if (seedIds.empty() || seedIds.count(memeId))
        return true;
      for (auto &n : graph_.neighborsOf(memeId)) {
        if (seedIds.count(n.first))
          return true;
      }
      for (auto &sid : seedIds) {
        for (auto &n : graph_.neighborsOf(sid)) {
          if (n.first == memeId)
            return true;
        }
      }
      return false;
    };

    std::unordered_map<std::string, double> candidateScores;
    for (size_t i = 0; i < result.windowInfo.ids.size(); i++) {
      if (i >= result.activation.size())
        break;
      double score = result.activation[i];
      if (!std::isfinite(score) || score <= 0)
        continue;
      const auto &memeId = result.windowInfo.ids[i];
      if (!isConnectedToSeeds(memeId))
        continue;
      for (auto &w : kvm_.getMemeWords(memeId)) {
        if (!w.empty())
          candidateScores[w] = std::max(candidateScores[w], score);
      }
    }
    std::vector<std::string> ordered;
    for (auto &kv : candidateScores)
      ordered.push_back(kv.first);
    std::sort(ordered.begin(), ordered.end(), [&](auto &a, auto &b) {
      return candidateScores[a] > candidateScores[b];
    });

    std::vector<std::string> finalWords;
    std::unordered_set<std::string> seen;
    auto pushWord = [&](const std::string &w) {
      if (!w.empty() && !seen.count(w)) {
        seen.insert(w);
        finalWords.push_back(w);
      }
    };
    for (auto &w : words)
      pushWord(w);
    for (auto &w : ordered) {
      pushWord(w);
      if (finalWords.size() >= 30)
        break;
    }
    if (finalWords.empty())
      return "I need more context to respond.";
    std::ostringstream oss;
    for (size_t i = 0; i < finalWords.size() && i < 30; i++) {
      if (i)
        oss << " ";
      oss << finalWords[i];
    }
    return oss.str();
  }
};
