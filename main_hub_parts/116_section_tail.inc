// Auto-generated from main.cpp
// Segment: section_tail
// Original lines: 14142-21164


  static double tokenOverlapJaccard(const std::vector<std::string> &a,
                                    const std::vector<std::string> &b) {
    if (a.empty() || b.empty())
      return 0.0;
    std::unordered_set<std::string> sa(a.begin(), a.end());
    std::unordered_set<std::string> sb(b.begin(), b.end());
    double inter = 0.0;
    for (auto &t : sa)
      if (sb.count(t))
        inter++;
    double uni = (double)sa.size() + (double)sb.size() - inter;
    return uni <= 0.0 ? 0.0 : (inter / uni);
  }

  GnnHintSummary buildGnnHintSummary(const json &graphResult,
                                     const json &payload,
                                     const ContextHint &hint) {
    GnnHintSummary out;
    auto runtime = activeRuntime();
    if (!runtime)
      return out;

    std::vector<std::pair<std::string, double>> memes;
    if (graphResult.contains("memes") && graphResult["memes"].is_array()) {
      auto mem = graphResult["memes"];
      auto act = graphResult.value("activation", json::array());
      for (size_t i = 0; i < mem.size(); i++) {
        if (!mem[i].is_string())
          continue;
        double score = 0.0;
        if (act.is_array() && i < act.size() && act[i].is_number())
          score = act[i].get<double>();
        memes.push_back({mem[i].get<std::string>(), score});
      }
    }
    std::sort(memes.begin(), memes.end(),
              [](auto &a, auto &b) { return a.second > b.second; });
    if (memes.size() > 8)
      memes.resize(8);

    std::unordered_set<std::string> seen;
    for (auto &m : memes) {
      auto ws = runtime->getMemeWords(m.first);
      for (auto &w : ws) {
        if (w.empty())
          continue;
        for (auto &t : tokenize(w)) {
          if (t.empty())
            continue;
          if (seen.insert(t).second)
            out.keywords.push_back(t);
          if (out.keywords.size() >= 24)
            break;
        }
        if (out.keywords.size() >= 24)
          break;
      }
      if (out.keywords.size() >= 24)
        break;
    }

    std::vector<std::string> hintTokens;
    if (!hint.text.empty()) {
      hintTokens = tokenize(hint.text);
    } else if (payload.contains("tokens") && payload["tokens"].is_array()) {
      for (auto &t : payload["tokens"])
        if (t.is_string())
          hintTokens.push_back(t.get<std::string>());
    } else if (payload.contains("text") && payload["text"].is_string()) {
      hintTokens = tokenize(payload["text"].get<std::string>());
    }
    out.align = tokenOverlapJaccard(out.keywords, hintTokens);

    if (!out.keywords.empty()) {
      double w = std::max(0.15, std::min(1.0, 0.2 + 0.8 * out.align));
      std::ostringstream oss;
      oss << "gnn_stage2|align=" << std::fixed << std::setprecision(2)
          << out.align << "|w=" << std::fixed << std::setprecision(2) << w
          << ":";
      for (size_t i = 0; i < out.keywords.size(); i++) {
        if (i)
          oss << " ";
        oss << out.keywords[i];
      }
      out.annotation = oss.str();
      if (out.annotation.size() > 512)
        out.annotation.resize(512);
    }
    return out;
  }

  ContextHint harmonizeHintWithGnn(const ContextHint &hint,
                                   double align) const {
    ContextHint out = hint;
    if (out.text.empty())
      return out;
    // When align=0 (no GNN alignment info), preserve the original weight
    // to avoid reducing user-provided contextWeight (e.g., 0.95 -> 0.38)
    if (align <= 0.0) {
      return out;
    }
    double gate = std::max(0.2, std::min(1.0, 0.4 + 0.6 * align));
    out.weight = clamp01(out.weight * gate);
    return out;
  }

  ContextHint extractContextHint(const json &payload) const {
    ContextHint hint;
    if (payload.contains("contextHint") && payload["contextHint"].is_string()) {
      hint.text = payload["contextHint"].get<std::string>();
    } else if (payload.contains("context") && payload["context"].is_string()) {
      hint.text = payload["context"].get<std::string>();
    }
    if (payload.contains("contextMode") && payload["contextMode"].is_string()) {
      hint.mode = payload["contextMode"].get<std::string>();
    }
    if (payload.contains("contextWeight") &&
        payload["contextWeight"].is_number()) {
      hint.weight = payload["contextWeight"].get<double>();
    }
    if (!std::isfinite(hint.weight))
      hint.weight = 0.35;
    hint.weight = std::max(0.0, std::min(1.0, hint.weight));
    auto trim = [](std::string &s) {
      auto start = s.find_first_not_of(" \t\r\n");
      if (start == std::string::npos) {
        s.clear();
        return;
      }
      auto end = s.find_last_not_of(" \t\r\n");
      s = s.substr(start, end - start + 1);
    };
    trim(hint.text);
    trim(hint.mode);
    return hint;
  }

  std::string applyContextHintToText(const std::string &text,
                                     const ContextHint &hint) const {
    if (hint.text.empty() || hint.weight <= 0.0)
      return text;
    std::string cleanedHint = hint.text;
    // Strip the trailing user line if it is the current message, so the
    // concatenated text does not duplicate the user input.
    // The hint text ends with a closing ']' after the last user line.
    if (!text.empty()) {
      std::string userLine = "  User: " + text + "\n";
      if (cleanedHint.size() >= userLine.size() + 1 &&
          cleanedHint.back() == ']' &&
          cleanedHint.compare(cleanedHint.size() - userLine.size() - 1,
                              userLine.size(), userLine) == 0) {
        // Remove the user line and keep the closing ']'.
        cleanedHint.erase(cleanedHint.size() - userLine.size() - 1,
                          userLine.size());
      }
    }
    std::ostringstream oss;
    oss << "[Context hint w=" << std::fixed << std::setprecision(2)
        << hint.weight;
    if (!hint.mode.empty())
      oss << " mode=" << hint.mode;
    oss << "]" << cleanedHint;
    if (text.empty())
      return oss.str();
    std::string result;
    if (hint.weight >= 0.5) {
      result = oss.str() + "\n" + text;
    } else {
      result = text + "\n" + oss.str();
    }
    return result;
  }

  std::string applyContextHintToGraph(const std::string &graphContext,
                                      const ContextHint &hint) const {
    if (hint.text.empty() || hint.weight <= 0.0)
      return graphContext;
    std::ostringstream oss;
    oss << "ctx_stage1";
    if (!hint.mode.empty())
      oss << "|" << hint.mode;
    oss << "|w=" << std::fixed << std::setprecision(2) << hint.weight << ":"
        << hint.text;
    if (graphContext.empty())
      return oss.str();
    if (hint.weight >= 0.5) {
      return oss.str() + "\n" + graphContext;
    }
    return graphContext + "\n" + oss.str();
  }

  static std::vector<float> readFloatArray(const json &arr,
                                           size_t maxItems = 8192) {
    std::vector<float> out;
    if (!arr.is_array())
      return out;
    out.reserve(std::min(arr.size(), maxItems));
    for (size_t i = 0; i < arr.size() && out.size() < maxItems; i++) {
      if (arr[i].is_number()) {
        out.push_back(arr[i].get<float>());
      }
    }
    return out;
  }

  static std::vector<std::vector<float>>
  chunkEmbedding(const std::vector<float> &vec, int dim, int maxChunks = 8) {
    std::vector<std::vector<float>> out;
    if (vec.empty() || dim <= 0)
      return out;
    int total = (int)vec.size();
    int chunks = std::min(maxChunks, (total + dim - 1) / dim);
    out.reserve((size_t)chunks);
    for (int c = 0; c < chunks; c++) {
      std::vector<float> v((size_t)dim, 0.0f);
      int start = c * dim;
      int end = std::min(total, start + dim);
      for (int i = start; i < end; i++) {
        v[(size_t)(i - start)] = vec[(size_t)i];
      }
      float norm = 0.0f;
      for (float x : v)
        norm += x * x;
      norm = std::sqrt(std::max(1e-9f, norm));
      for (auto &x : v)
        x /= norm;
      out.push_back(std::move(v));
    }
    return out;
  }

  static std::string buildImageContextFromJson(const json &img) {
    if (!img.is_object())
      return "";
    if (img.contains("graphContext") && img["graphContext"].is_string()) {
      return img["graphContext"].get<std::string>();
    }
    std::ostringstream oss;
    bool has = false;
    if (img.contains("detections") && img["detections"].is_array()) {
      oss << "vision|labels:";
      size_t count = 0;
      for (auto &d : img["detections"]) {
        if (!d.is_object())
          continue;
        std::string label = d.value("label", "");
        double score = d.value("score", 0.0);
        if (label.empty())
          continue;
        if (count++)
          oss << ",";
        oss << label << "(" << std::fixed << std::setprecision(2) << score
            << ")";
        if (count >= 16)
          break;
      }
      has = count > 0;
    }
    if (img.contains("details") && img["details"].is_array()) {
      if (has)
        oss << "|";
      oss << "details:";
      size_t count = 0;
      for (auto &d : img["details"]) {
        if (!d.is_object())
          continue;
        std::string label = d.value("label", "");
        double score = d.value("score", 0.0);
        if (label.empty())
          continue;
        if (count++)
          oss << ",";
        oss << label << "(" << std::fixed << std::setprecision(2) << score
            << ")";
        if (count >= 16)
          break;
      }
      has = has || (count > 0);
    }
    if (!has)
      return "";
    return oss.str();
  }

  static std::string extractImageContext(const json &payload) {
    if (payload.contains("imageContext") &&
        payload["imageContext"].is_string()) {
      return payload["imageContext"].get<std::string>();
    }
    if (payload.contains("image") && payload["image"].is_object()) {
      return buildImageContextFromJson(payload["image"]);
    }
    if (payload.contains("vision") && payload["vision"].is_object()) {
      return buildImageContextFromJson(payload["vision"]);
    }
    if (payload.contains("imageEmbedding") ||
        payload.contains("imageEmbeddings")) {
      return "vision|embedding";
    }
    return "";
  }

  static std::vector<std::vector<float>>
  extractImageEmbeddings(const json &payload, int dim) {
    std::vector<std::vector<float>> out;
    if (payload.contains("imageEmbeddings") &&
        payload["imageEmbeddings"].is_array()) {
      for (auto &e : payload["imageEmbeddings"]) {
        auto vec = readFloatArray(e);
        auto chunks = chunkEmbedding(vec, dim, 1);
        if (!chunks.empty())
          out.push_back(std::move(chunks.front()));
      }
    }
    if (payload.contains("imageEmbedding")) {
      auto vec = readFloatArray(payload["imageEmbedding"]);
      auto chunks = chunkEmbedding(vec, dim, 8);
      out.insert(out.end(), chunks.begin(), chunks.end());
    }
    if (payload.contains("image") && payload["image"].is_object()) {
      auto &img = payload["image"];
      if (img.contains("embeddings") && img["embeddings"].is_array()) {
        for (auto &e : img["embeddings"]) {
          auto vec = readFloatArray(e);
          auto chunks = chunkEmbedding(vec, dim, 1);
          if (!chunks.empty())
            out.push_back(std::move(chunks.front()));
        }
      }
      if (img.contains("embedding")) {
        auto vec = readFloatArray(img["embedding"]);
        auto chunks = chunkEmbedding(vec, dim, 8);
        out.insert(out.end(), chunks.begin(), chunks.end());
      }
    }
    if (payload.contains("vision") && payload["vision"].is_object()) {
      auto &img = payload["vision"];
      if (img.contains("embeddings") && img["embeddings"].is_array()) {
        for (auto &e : img["embeddings"]) {
          auto vec = readFloatArray(e);
          auto chunks = chunkEmbedding(vec, dim, 1);
          if (!chunks.empty())
            out.push_back(std::move(chunks.front()));
        }
      }
      if (img.contains("embedding")) {
        auto vec = readFloatArray(img["embedding"]);
        auto chunks = chunkEmbedding(vec, dim, 8);
        out.insert(out.end(), chunks.begin(), chunks.end());
      }
    }
    return out;
  }

  static void injectImageContext(json &payload,
                                 const std::string &imageContext) {
    if (imageContext.empty())
      return;
    payload["imageContext"] = imageContext;
    if (payload.contains("text") && payload["text"].is_string()) {
      payload["text"] =
          payload["text"].get<std::string>() + "\n" + imageContext;
    } else if (payload.contains("message") && payload["message"].is_string()) {
      payload["message"] =
          payload["message"].get<std::string>() + "\n" + imageContext;
    }
  }

  std::string buildInferenceCacheKey(const std::string &sessionId,
                                     const std::string &text,
                                     const std::string &graphContext,
                                     int maxTokens) const {
    std::ostringstream oss;
    oss << "v3|s=" << sessionId << "|t=" << text << "|g=" << graphContext
        << "|m=" << maxTokens;
    return sha1Hex(oss.str());
  }

  std::vector<std::vector<float>>
  buildGraphEmbeddings(const json &graphResult, int dim, int rounds = 1) {
    auto runtime = activeRuntime();
    if (!runtime)
      return {};
    std::vector<std::string> ids;
    if (graphResult.contains("memes") && graphResult["memes"].is_array()) {
      for (auto &m : graphResult["memes"]) {
        if (!m.is_string())
          continue;
        ids.push_back(m.get<std::string>());
        if (ids.size() >= 64)
          break;
      }
    }
    if (ids.empty())
      return {};

    auto buildLexicalEmbeddings = [&]() -> std::vector<std::vector<float>> {
      std::vector<std::vector<float>> out;
      out.reserve(ids.size());
      for (auto &id : ids) {
        auto words = runtime->getMemeWords(id);
        std::ostringstream oss;
        oss << id;
        for (size_t i = 0; i < words.size() && i < 6; i++) {
          oss << ' ' << words[i];
        }
        out.push_back(textToMiniEmbedding(oss.str(), dim));
      }
      return out;
    };

    if (matrixCache_ && matrixCache_->enabled()) {
      std::ostringstream oss;
      oss << "graphEmb|d=" << dim << "|r=" << rounds << "|";
      for (size_t i = 0; i < ids.size(); i++) {
        if (i)
          oss << ",";
        oss << ids[i];
      }
      std::string key = sha1Hex(oss.str());
      std::vector<std::vector<float>> cached;
      if (matrixCache_->getMatrix(key, (int)ids.size(), dim, cached)) {
        return cached;
      }
      std::vector<std::vector<float>> out;
      if (auto rt = std::dynamic_pointer_cast<RuntimeState>(runtime)) {
        GraphTensorBridge bridge(&rt->kvm(), nullptr);
        auto base = bridge.computeNodeEmbeddings(ids, dim);
        out = rt->graph().messagePassingEmbeddings(ids, base, rounds, 0.65f,
                                                   0.35f);
      } else {
        out = buildLexicalEmbeddings();
      }
      matrixCache_->recordAndMaybeSet(key, out, 900, 3);
      return out;
    }
    if (auto rt = std::dynamic_pointer_cast<RuntimeState>(runtime)) {
      GraphTensorBridge bridge(&rt->kvm(), nullptr);
      auto base = bridge.computeNodeEmbeddings(ids, dim);
      return rt->graph().messagePassingEmbeddings(ids, base, rounds, 0.65f,
                                                  0.35f);
    }
    return buildLexicalEmbeddings();
  }

  std::vector<transformer::TrainSample> parseTrainSamples(const json &payload) {
    std::vector<transformer::TrainSample> samples;
    if (!payload.contains("samples") || !payload["samples"].is_array())
      return samples;
    for (auto &s : payload["samples"]) {
      if (!s.is_object())
        continue;
      transformer::TrainSample sample;
      bool cleanedInput = false;
      bool cleanedTarget = false;
      bool cleanedGraph = false;
      sample.input = sanitizeInputText(s.value("input", ""), cleanedInput);
      sample.target = sanitizeInputText(s.value("target", ""), cleanedTarget);
      sample.graph = sanitizeInputText(s.value("graph", ""), cleanedGraph);
      if (cleanedInput || cleanedTarget || cleanedGraph)
        cleanedSampleCount_.fetch_add(1, std::memory_order_relaxed);
      if (sample.input.empty() || sample.target.empty())
        continue;
      samples.push_back(std::move(sample));
    }
    return samples;
  }

  std::vector<std::string>
  deriveGnnStyleKeywords(const std::string &text,
                         size_t maxKeywords = 8) const {
    std::vector<std::string> out;
    auto runtime = activeRuntime();
    if (!runtime)
      return out;

    auto words = tokenize(text);
    if (words.empty())
      return out;

    auto seeds = runtime->mapWordsToMemes(words);
    std::vector<std::pair<std::string, double>> rankedSeeds(seeds.begin(),
                                                            seeds.end());
    std::sort(rankedSeeds.begin(), rankedSeeds.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });

    std::unordered_set<std::string> seen;
    for (const auto &seed : rankedSeeds) {
      auto linked = runtime->getMemeWords(seed.first);
      for (const auto &w : linked) {
        std::string kw = trimCopy(w);
        if (kw.size() < 2 || kw.size() > 24)
          continue;
        if (seen.insert(kw).second)
          out.push_back(kw);
        if (out.size() >= maxKeywords)
          return out;
      }
    }

    for (const auto &w : words) {
      std::string kw = trimCopy(w);
      if (kw.size() < 2 || kw.size() > 24)
        continue;
      if (seen.insert(kw).second)
        out.push_back(kw);
      if (out.size() >= maxKeywords)
        break;
    }
    return out;
  }

  json
  buildExternalStyleFeatures(const std::string &text,
                             const std::vector<std::string> &keywords) const {
    static const std::unordered_set<std::string> positiveLex = {
        "good",    "great",  "excellent", "clear",        "smart",
        "helpful", "fast",   "friendly",  "professional", "confident",
        "precise", "stable", "strong",    "clean"};
    static const std::unordered_set<std::string> negativeLex = {
        "bad",  "poor",   "slow",  "confused", "messy",   "unclear", "angry",
        "rude", "broken", "error", "fail",     "failure", "chaotic", "weak"};

    std::unordered_set<std::string> normalizedKeywords;
    normalizedKeywords.reserve(keywords.size());
    for (const auto &kwRaw : keywords) {
      std::string kw = trimCopy(kwRaw);
      if (kw.empty())
        continue;
      std::transform(kw.begin(), kw.end(), kw.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
      normalizedKeywords.insert(kw);
    }

    auto words = tokenize(text);
    size_t keywordHits = 0;
    size_t posHits = 0;
    size_t negHits = 0;
    for (auto w : words) {
      std::transform(w.begin(), w.end(), w.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
      if (normalizedKeywords.find(w) != normalizedKeywords.end())
        keywordHits++;
      if (positiveLex.find(w) != positiveLex.end())
        posHits++;
      if (negativeLex.find(w) != negativeLex.end())
        negHits++;
    }

    size_t punctuationCount = 0;
    for (unsigned char ch : text) {
      if (std::ispunct(ch))
        punctuationCount++;
    }

    const double tokenCount = static_cast<double>(words.size());
    const double keywordDensity =
        tokenCount > 0.0 ? static_cast<double>(keywordHits) / tokenCount : 0.0;

    const double sentiDen = static_cast<double>(posHits + negHits);
    double sentiment = 0.0;
    if (sentiDen > 0.0) {
      sentiment =
          (static_cast<double>(posHits) - static_cast<double>(negHits)) /
          sentiDen;
    }

    const double textLen = static_cast<double>(text.size());
    const double punctuationDensity =
        textLen > 0.0 ? static_cast<double>(punctuationCount) / textLen : 0.0;

    return json{
        {"keywordDensity", std::clamp(keywordDensity, 0.0, 1.0)},
        {"sentiment", std::clamp(sentiment, -1.0, 1.0)},
        {"punctuationDensity", std::clamp(punctuationDensity, 0.0, 1.0)}};
  }

  json buildExternalStyleTrainPayload(
      bool usingBitNet, const json &payload, const std::string &text,
      double observedAlign, const std::vector<std::string> &keywords) const {
    json features = buildExternalStyleFeatures(text, keywords);
    double graphEdgeWeight = payload.value(
        "graphEdgeWeight",
        payload.value("edgeWeight", payload.value("avgEdgeWeight", 0.0)));
    double graphResidual = payload.value(
        "graphResidual",
        payload.value("residual", payload.value("avgResidual", 0.0)));
    graphEdgeWeight = std::clamp(graphEdgeWeight, 0.0, 1.0);
    graphResidual = std::clamp(graphResidual, 0.0, 1.0);
    return json{
        {"provider", usingBitNet ? "bitnet" : "llamacpp"},
        {"model", usingBitNet ? config_.bitnetModel : config_.llamaCppModel},
        {"text", text},
        {"style",
         trimCopy(payload.value(
             "style",
             payload.value("persona", payload.value("styleProfile", ""))))},
        {"targetAlign",
         payload.value("targetAlign", payload.value("styleTargetAlign", 0.85))},
        {"observedAlign", payload.value("observedAlign", observedAlign)},
        {"lossScale", payload.value("lossScale", 1.0)},
        {"styleKeywords", keywords},
        {"features", features},
        {"keywordDensity", features.value("keywordDensity", 0.0)},
        {"sentiment", features.value("sentiment", 0.0)},
        {"punctuationDensity", features.value("punctuationDensity", 0.0)},
        {"graphEdgeWeight", graphEdgeWeight},
        {"graphResidual", graphResidual}};
  }

  std::vector<std::string> parseStyleKeywords(const json &payload,
                                              const std::string &text) const {
    std::vector<std::string> keywords;
    if (payload.contains("styleKeywords") &&
        payload["styleKeywords"].is_array()) {
      for (const auto &it : payload["styleKeywords"]) {
        if (it.is_string()) {
          std::string kw = trimCopy(it.get<std::string>());
          if (!kw.empty())
            keywords.push_back(kw);
        }
        if (keywords.size() >= 8)
          break;
      }
    }
    if (keywords.empty())
      keywords = deriveGnnStyleKeywords(text, 8);
    return keywords;
  }

  json postExternalStyleTrainStep(const std::string &provider,
                                  const std::string &baseUrl,
                                  const std::string &model, int timeoutMs,
                                  const json &payload) const {
    json out = json::object();
    out["provider"] = provider;
    out["ok"] = false;
    try {
      std::string endpoint = trimCopy(baseUrl);
      if (endpoint.empty())
        endpoint = provider == "bitnet" ? "http://127.0.0.1:8090"
                                        : "http://127.0.0.1:8080";
      auto client = drogon::HttpClient::newHttpClient(endpoint);
      if (!client) {
        out["error"] = provider + " client init failed";
        return out;
      }

      auto callPath = [&](const std::string &path) -> json {
        json ret = json::object();
        auto req = drogon::HttpRequest::newHttpRequest();
        req->setMethod(drogon::Post);
        req->setPath(path);
        req->setContentTypeCode(drogon::CT_APPLICATION_JSON);
        req->setBody(payload.dump());

        std::promise<std::pair<drogon::ReqResult, drogon::HttpResponsePtr>>
            promise;
        auto future = promise.get_future();
        client->sendRequest(req,
                            [&promise](drogon::ReqResult result,
                                       const drogon::HttpResponsePtr &resp) {
                              try {
                                promise.set_value({result, resp});
                              } catch (...) {
                              }
                            });

        if (future.wait_for(std::chrono::milliseconds(
                std::max(1000, timeoutMs))) != std::future_status::ready) {
          ret["ok"] = false;
          ret["timeout"] = true;
          ret["error"] = provider + " style train timeout";
          return ret;
        }

        auto pair = future.get();
        auto reqResult = pair.first;
        auto resp = pair.second;
        if (reqResult != drogon::ReqResult::Ok || !resp) {
          ret["ok"] = false;
          ret["error"] = provider + " style train request failed";
          return ret;
        }
        ret["status"] = (int)resp->statusCode();
        if (resp->statusCode() < 200 || resp->statusCode() >= 300) {
          ret["ok"] = false;
          ret["error"] = provider + " style train bad status";
          return ret;
        }

        json doc = json::parse(resp->getBody(), nullptr, false);
        if (doc.is_discarded()) {
          ret["ok"] = false;
          ret["error"] = provider + " style train invalid json";
          return ret;
        }
        ret["ok"] = true;
        ret["result"] = doc;
        return ret;
      };

      json first = callPath("/api/style/adapter/train_step");
      if (first.value("ok", false))
        return first;

      int firstStatus = first.value("status", 0);
      if (firstStatus == 404 || firstStatus == 405 || firstStatus == 0) {
        json second = callPath("/v1/style/adapter/train_step");
        if (second.value("ok", false))
          return second;
        second["fallback"] = "tried";
        return second;
      }
      return first;
    } catch (const std::exception &e) {
      out["error"] = e.what();
      return out;
    } catch (...) {
      out["error"] = provider + " style train unknown error";
      return out;
    }
  }

  void dispatchExternalStyleTrainStep(const std::string &backendMode,
                                      const json &payload,
                                      const std::string &text,
                                      double gnnAlign) {
    if (!config_.externalBackendStyleSim)
      return;
    std::string mode = trimCopy(backendMode);
    std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    bool usingLlamaCpp =
        (mode == "llamacpp" || mode == "llama.cpp" || mode == "llama_cpp");
    bool usingBitNet = (mode == "bitnet");
    if (!usingLlamaCpp && !usingBitNet)
      return;

    std::vector<std::string> keywords = parseStyleKeywords(payload, text);
    json trainPayload = buildExternalStyleTrainPayload(
        usingBitNet, payload, text, gnnAlign, keywords);

    json trainResp =
        usingBitNet
            ? postExternalStyleTrainStep(
                  "bitnet", config_.bitnetBaseUrl, config_.bitnetModel,
                  config_.bitnetRequestTimeoutMs, trainPayload)
            : postExternalStyleTrainStep(
                  "llamacpp", config_.llamaCppBaseUrl, config_.llamaCppModel,
                  config_.llamaCppRequestTimeoutMs, trainPayload);
    if (!trainResp.value("ok", false) && logger_.enabled()) {
      logger_.log(
          LoggerCXX::Type::WARNING,
          std::string("external style train dispatch failed provider=") +
              (usingBitNet ? "bitnet" : "llamacpp") +
              " error=" + trainResp.value("error", std::string("unknown")));
    }
  }

  bool isExternalBackendMode(std::string mode) const {
    mode = trimCopy(mode);
    std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    return mode == "llamacpp" || mode == "llama.cpp" || mode == "llama_cpp" ||
           mode == "bitnet";
  }

  json bridgeLearnerResultToExternalBackend(const std::string &learner,
                                            const std::string &backendMode,
                                            const json &requestBody,
                                            const json &learnerResult) {
    std::string mode = trimCopy(backendMode);
    std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    bool usingLlamaCpp =
        (mode == "llamacpp" || mode == "llama.cpp" || mode == "llama_cpp");
    bool usingBitNet = (mode == "bitnet");
    if (!usingLlamaCpp && !usingBitNet) {
      return json{{"applied", false}, {"reason", "non-external-backend"}};
    }

    std::vector<std::pair<std::string, double>> trainItems;
    if (learnerResult.contains("evals") && learnerResult["evals"].is_array()) {
      for (const auto &it : learnerResult["evals"]) {
        std::string text = trimCopy(it.value("reply", ""));
        if (text.empty())
          continue;
        trainItems.push_back(
            {text, std::clamp(it.value("score", 0.0), 0.0, 1.0)});
        if (trainItems.size() >= 24)
          break;
      }
    }
    if (trainItems.empty() && learnerResult.contains("rounds") &&
        learnerResult["rounds"].is_array()) {
      for (const auto &round : learnerResult["rounds"]) {
        auto results = round.value("results", json::array());
        if (!results.is_array())
          continue;
        for (const auto &item : results) {
          std::string text = trimCopy(item.value("text", ""));
          if (text.empty())
            continue;
          trainItems.push_back(
              {text, std::clamp(item.value("quality", 0.0), 0.0, 1.0)});
          if (trainItems.size() >= 24)
            break;
        }
        if (trainItems.size() >= 24)
          break;
      }
    }
    if (trainItems.empty() && requestBody.contains("samples") &&
        requestBody["samples"].is_array()) {
      double align = 0.5;
      if (learnerResult.contains("best") && learnerResult["best"].is_object())
        align = std::clamp(learnerResult["best"].value("avgQuality", 0.5), 0.0,
                           1.0);
      for (const auto &s : requestBody["samples"]) {
        if (!s.is_string())
          continue;
        std::string text = trimCopy(s.get<std::string>());
        if (text.empty())
          continue;
        trainItems.push_back({text, align});
        if (trainItems.size() >= 24)
          break;
      }
    }

    if (trainItems.empty()) {
      return json{{"applied", false}, {"reason", "no-train-items"}};
    }

    json stats = json{{"applied", true},
                      {"learner", learner},
                      {"provider", usingBitNet ? "bitnet" : "llamacpp"},
                      {"steps", 0},
                      {"ok", 0},
                      {"failed", 0},
                      {"errors", json::array()}};

    const int maxThreads = std::max(1, config_.inferWorkers);
    const int threadCount =
        std::max(1, std::min<int>(maxThreads, (int)trainItems.size()));
    stats["threads"] = threadCount;

    std::atomic<size_t> index{0};
    std::atomic<int> stepCount{0};
    std::atomic<int> okCount{0};
    std::atomic<int> failCount{0};
    std::mutex errMu;
    std::vector<std::string> errors;

    auto worker = [&]() {
      while (true) {
        size_t i = index.fetch_add(1);
        if (i >= trainItems.size())
          return;

        const std::string &text = trainItems[i].first;
        const double observed = trainItems[i].second;
        json merged = requestBody;
        merged["style"] = learner;
        merged["lossScale"] = requestBody.value("lossScale", 1.0);
        merged["targetAlign"] = requestBody.value("targetAlign", 0.85);
        merged["observedAlign"] = observed;
        merged["learner"] = learner;
        merged["source"] = "learning-bridge";
        if (learnerResult.contains("best") &&
            learnerResult["best"].is_object()) {
          merged["graphEdgeWeight"] =
              learnerResult["best"].value("edgeWeight", 0.0);
          merged["graphResidual"] =
              learnerResult["best"].value("avgResidual", 0.0);
        } else {
          merged["graphEdgeWeight"] = learnerResult.value("avgEdgeWeight", 0.0);
          merged["graphResidual"] = learnerResult.value(
              "avgResidual", learnerResult.value("residual", 0.0));
        }

        auto keywords = parseStyleKeywords(merged, text);
        auto trainPayload = buildExternalStyleTrainPayload(
            usingBitNet, merged, text, observed, keywords);
        trainPayload["learningBridge"] = json{{"learner", learner},
                                              {"backend", mode},
                                              {"groupProc", isGroupProc()}};

        json step =
            usingBitNet
                ? postExternalStyleTrainStep(
                      "bitnet", config_.bitnetBaseUrl, config_.bitnetModel,
                      config_.bitnetRequestTimeoutMs, trainPayload)
                : postExternalStyleTrainStep(
                      "llamacpp", config_.llamaCppBaseUrl,
                      config_.llamaCppModel, config_.llamaCppRequestTimeoutMs,
                      trainPayload);

        stepCount.fetch_add(1);
        if (step.value("ok", false)) {
          okCount.fetch_add(1);
        } else {
          failCount.fetch_add(1);
          std::lock_guard<std::mutex> lock(errMu);
          errors.push_back(step.value("error", std::string("unknown")));
        }
      }
    };

    std::vector<std::thread> threads;
    threads.reserve((size_t)threadCount);
    for (int i = 0; i < threadCount; i++) {
      threads.emplace_back(worker);
    }
    for (auto &t : threads) {
      if (t.joinable())
        t.join();
    }

    stats["steps"] = stepCount.load();
    stats["ok"] = okCount.load();
    stats["failed"] = failCount.load();
    for (const auto &e : errors)
      stats["errors"].push_back(e);

    return stats;
  }

  bool parseBoundedInt64(const json &payload, const char *key, int64_t minValue,
                         int64_t maxValue, int64_t &out, std::string &err,
                         bool required = false) const {
    if (!payload.contains(key)) {
      if (required) {
        err = std::string(key) + " required";
        return false;
      }
      return true;
    }
    const auto &v = payload[key];
    if (!(v.is_number_integer() || v.is_number_unsigned())) {
      err = std::string(key) + " must be integer";
      return false;
    }
    int64_t parsed = 0;
    try {
      parsed = v.get<int64_t>();
    } catch (...) {
      err = std::string(key) + " out of int64 range";
      return false;
    }
    if (parsed < minValue || parsed > maxValue) {
      std::ostringstream oss;
      oss << key << " out of range [" << minValue << ", " << maxValue << "]";
      err = oss.str();
      return false;
    }
    out = parsed;
    return true;
  }

  uint64_t currentRssBytes() const {
    auto mem = getMemoryUsage();
    if (mem.contains("rss") && mem["rss"].is_number_integer()) {
      return (uint64_t)std::max<int64_t>(0, mem["rss"].get<int64_t>());
    }
    return 0;
  }

  json trainMonitorToJson(const TrainMonitorState &s) const {
    auto toMB = [](uint64_t bytes) { return (double)bytes / 1024.0 / 1024.0; };
    int64_t elapsedMs = 0;
    if (s.startedAtMs > 0) {
      int64_t endMs =
          s.running ? nowEpochMs()
                    : (s.finishedAtMs > 0 ? s.finishedAtMs : s.updatedAtMs);
      elapsedMs = std::max<int64_t>(0, endMs - s.startedAtMs);
    }
    return json{{"id", s.id},
                {"running", s.running},
                {"ok", s.ok},
                {"route", s.route},
                {"stage", s.stage},
                {"error", s.error},
                {"startedAtMs", s.startedAtMs},
                {"updatedAtMs", s.updatedAtMs},
                {"finishedAtMs", s.finishedAtMs},
                {"elapsedMs", elapsedMs},
                {"rssBefore", s.rssBefore},
                {"rssPeak", s.rssPeak},
                {"rssAfter", s.rssAfter},
                {"rssBeforeMB", toMB(s.rssBefore)},
                {"rssPeakMB", toMB(s.rssPeak)},
                {"rssAfterMB", toMB(s.rssAfter)},
                {"docs", s.docs},
                {"processedDocs", s.processedDocs},
                {"samples", s.samples},
                {"chars", s.chars},
                {"request", s.requestSummary},
                {"result", s.resultSummary}};
  }

  uint64_t beginTrainMonitor(const std::string &route,
                             const json &requestSummary) {
    std::lock_guard<std::mutex> lock(trainMonitorMu_);
    trainMonitorSeq_ += 1;
    trainMonitorActive_ = TrainMonitorState{};
    trainMonitorActive_.id = trainMonitorSeq_;
    trainMonitorActive_.running = true;
    trainMonitorActive_.route = route;
    trainMonitorActive_.stage = "started";
    trainMonitorActive_.startedAtMs = nowEpochMs();
    trainMonitorActive_.updatedAtMs = trainMonitorActive_.startedAtMs;
    trainMonitorActive_.requestSummary = requestSummary;
    trainMonitorActive_.rssBefore = currentRssBytes();
    trainMonitorActive_.rssPeak = trainMonitorActive_.rssBefore;
    return trainMonitorActive_.id;
  }

  void touchTrainMonitor(const std::string &stage, uint64_t docs,
                         uint64_t processedDocs, uint64_t samples,
                         uint64_t chars, const std::string &error = "") {
    std::lock_guard<std::mutex> lock(trainMonitorMu_);
    if (!trainMonitorActive_.running)
      return;
    trainMonitorActive_.stage = stage;
    trainMonitorActive_.updatedAtMs = nowEpochMs();
    trainMonitorActive_.docs = docs;
    trainMonitorActive_.processedDocs = processedDocs;
    trainMonitorActive_.samples = samples;
    trainMonitorActive_.chars = chars;
    if (!error.empty())
      trainMonitorActive_.error = error;
    uint64_t rss = currentRssBytes();
    trainMonitorActive_.rssPeak = std::max(trainMonitorActive_.rssPeak, rss);
  }

  void finishTrainMonitor(bool ok, const json &resultSummary,
                          const std::string &error = "") {
    std::lock_guard<std::mutex> lock(trainMonitorMu_);
    if (!trainMonitorActive_.running)
      return;
    trainMonitorActive_.running = false;
    trainMonitorActive_.ok = ok;
    trainMonitorActive_.finishedAtMs = nowEpochMs();
    trainMonitorActive_.updatedAtMs = trainMonitorActive_.finishedAtMs;
    trainMonitorActive_.resultSummary = resultSummary;
    if (!error.empty())
      trainMonitorActive_.error = error;
    trainMonitorActive_.rssAfter = currentRssBytes();
    trainMonitorActive_.rssPeak =
        std::max(trainMonitorActive_.rssPeak, trainMonitorActive_.rssAfter);
    trainMonitorHistory_.push_front(trainMonitorToJson(trainMonitorActive_));
    while (trainMonitorHistory_.size() > 24)
      trainMonitorHistory_.pop_back();
  }

  json getTrainMonitorStats() {
    std::lock_guard<std::mutex> lock(trainMonitorMu_);
    json history = json::array();
    for (const auto &item : trainMonitorHistory_)
      history.push_back(item);
    return json{{"active", trainMonitorToJson(trainMonitorActive_)},
                {"history", history}};
  }

  void resetTrainMonitor() {
    std::lock_guard<std::mutex> lock(trainMonitorMu_);
    trainMonitorActive_ = TrainMonitorState{};
    trainMonitorHistory_.clear();
  }

  void recordRouteMetric(const std::string &route, long long elapsedMs,
                         bool ok) {
    std::lock_guard<std::mutex> lock(routeMetricMu_);
    auto &m = routeMetrics_[route];
    m.calls += 1;
    uint64_t ms = (uint64_t)std::max<long long>(0, elapsedMs);
    m.totalMs += ms;
    m.lastMs = ms;
    m.maxMs = std::max(m.maxMs, ms);
    if (!ok)
      m.errors += 1;
  }

  json getMonitoringStats() {
    json out;
    out["uptimeSec"] = std::chrono::duration_cast<std::chrono::seconds>(
                           std::chrono::steady_clock::now() - startedAt_)
                           .count();
    out["performance"] = json{{"memory", getMemoryUsage()},
                              {"processCpuPercent", getProcessCpuPercent()}};
    out["cleaning"] = {
        {"enabled", dataCleaningEnabled_},
        {"maxChars", dataCleanMaxChars_},
        {"cleanedInputs", cleanedInputCount_.load(std::memory_order_relaxed)},
        {"cleanedSamples",
         cleanedSampleCount_.load(std::memory_order_relaxed)}};
    json routes = json::object();
    {
      std::lock_guard<std::mutex> lock(routeMetricMu_);
      for (const auto &kv : routeMetrics_) {
        const auto &m = kv.second;
        uint64_t okCalls = m.calls >= m.errors ? (m.calls - m.errors) : 0;
        routes[kv.first] = {{"calls", m.calls},
                            {"errors", m.errors},
                            {"lastMs", m.lastMs},
                            {"maxMs", m.maxMs},
                            {"avgMs", okCalls ? (m.totalMs / okCalls) : 0}};
      }
    }
    out["routes"] = routes;
    out["training"] = getTrainMonitorStats();
    return out;
  }

  void resetMonitoringStats() {
    std::lock_guard<std::mutex> lock(routeMetricMu_);
    routeMetrics_.clear();
    cleanedInputCount_.store(0, std::memory_order_relaxed);
    cleanedSampleCount_.store(0, std::memory_order_relaxed);
    resetTrainMonitor();
  }

  json getRuntimeFeatureState() {
    json out;
    out["memebarrier"] = {
        {"enabled", barrier_ ? barrier_->running() : false},
        {"available", !config_.disableBarrier},
        {"threshold",
         barrier_ ? barrier_->threshold() : getModelDefaults().maliciousThreshold},
        {"disableRate", barrier_ ? barrier_->disableRateTarget() : 0.01},
        {"useTextCNN", barrier_ ? barrier_->useTextCNN() : true},
        {"useRecommender", barrier_ ? barrier_->useRecommender() : true},
        {"useTorchModels", barrier_ ? barrier_->useTorchModels() : false}};
    out["learning"] = {
        {"enabled", !(rlDisabled_ && advDisabled_ && gnnGaDisabled_)},
        {"cliDisabled", config_.disableLearning}};
    out["rl"] = {
        {"enabled", !rlDisabled_},
        {"cliDisabled", (config_.disableLearning || config_.disableRL)}};
    out["adv"] = {
        {"enabled", !advDisabled_},
        {"cliDisabled", (config_.disableLearning || config_.disableADV)}};
    out["gnnGa"] = {{"enabled", !gnnGaDisabled_},
                    {"cliDisabled", config_.disableLearning}};
    out["dialogLearning"] = {
        {"enabled", dialogLearningEnabled_},
        {"transformerEnabled", dialogTransformerLearning_}};
    out["dialogThresholds"] = {{"rlEvery", dialogThresholds_.rlEvery},
                               {"advEvery", dialogThresholds_.advEvery},
                               {"gnnEvery", dialogThresholds_.gnnEvery}};
    out["dialogCounters"] = {{"total", dialogCounters_.total},
                             {"lastRL", dialogCounters_.lastRL},
                             {"lastADV", dialogCounters_.lastADV},
                             {"lastGNN", dialogCounters_.lastGNN}};
    out["dataCleaning"] = {{"enabled", dataCleaningEnabled_},
                           {"maxChars", dataCleanMaxChars_}};
    std::string transformerMode = "off";
    if (config_.inferenceEnabled) {
      if (config_.transformerBackend.empty()) {
        // Do not silently fall back to the slow native Python transformer.
        config_.transformerBackend =
            config_.usingOllama
                ? (config_.ollamaFineTuning ? "ollama-fine-tuning" : "ollama")
                : "llamacpp";
      }
      transformerMode = trimCopy(config_.transformerBackend);
    }
    out["pipeline"] = {{"gnnModuleDisabled", config_.disableGnnModule},
                       {"contextModuleDisabled", config_.disableContextModule},
                       {"inferenceEnabled", config_.inferenceEnabled},
                       {"usingOllama", config_.usingOllama},
                       {"ollamaFineTuning", config_.ollamaFineTuning},
                       {"transformerMode", transformerMode},
                       {"ollamaModel", config_.ollamaModel},
                       {"llamacppModel", config_.llamaCppModel},
                       {"bitnetModel", config_.bitnetModel},
                       {"shmProtocolVersion", kSharedMemoryProtocolVersion},
                       {"outsidesLinkMode", currentOutsidesLinkMode()},
                       {"version", phoenix::PHOENIX_VERSION_STRING},
                       {"codename", phoenix::PHOENIX_CODENAME}};
    return out;
  }

  json applyRuntimeFeaturePatch(const json &patch) {
    json out{{"applied", json::object()}, {"warnings", json::array()}};
    if (patch.contains("memebarrierEnabled") &&
        patch["memebarrierEnabled"].is_boolean()) {
      bool on = patch["memebarrierEnabled"].get<bool>();
      if (on) {
        auto rt = std::dynamic_pointer_cast<RuntimeState>(
            pool_->getActive()->runtime());
        if (!barrier_ && rt)
          barrier_ = std::make_shared<MemeBarrier>(rt);
        if (barrier_)
          barrier_->start();
        else
          out["warnings"].push_back("memebarrier requires RuntimeState; "
                                    "current backend does not expose it");
      } else if (barrier_) {
        barrier_->stop();
      }
      out["applied"]["memebarrierEnabled"] = on;
    }
    if (patch.contains("maliciousThreshold") &&
        patch["maliciousThreshold"].is_number()) {
      double v = patch["maliciousThreshold"].get<double>();
      auto rt = std::dynamic_pointer_cast<RuntimeState>(
          pool_->getActive()->runtime());
      if (!barrier_ && rt)
        barrier_ = std::make_shared<MemeBarrier>(rt);
      if (barrier_)
        barrier_->setThreshold(v);
      out["applied"]["maliciousThreshold"] = v;
    }
    if (patch.contains("memebarrierDisableRate") &&
        patch["memebarrierDisableRate"].is_number()) {
      double v = patch["memebarrierDisableRate"].get<double>();
      auto rt = std::dynamic_pointer_cast<RuntimeState>(
          pool_->getActive()->runtime());
      if (!barrier_ && rt)
        barrier_ = std::make_shared<MemeBarrier>(rt);
      if (barrier_)
        barrier_->setDisableRateTarget(v);
      out["applied"]["memebarrierDisableRate"] = v;
    }
    if (patch.contains("memebarrierUseCNN") &&
        patch["memebarrierUseCNN"].is_boolean()) {
      bool on = patch["memebarrierUseCNN"].get<bool>();
      auto rt = std::dynamic_pointer_cast<RuntimeState>(
          pool_->getActive()->runtime());
      if (!barrier_ && rt)
        barrier_ = std::make_shared<MemeBarrier>(rt);
      if (barrier_)
        barrier_->setUseTextCNN(on);
      out["applied"]["memebarrierUseCNN"] = on;
    }
    if (patch.contains("memebarrierUseRecommender") &&
        patch["memebarrierUseRecommender"].is_boolean()) {
      bool on = patch["memebarrierUseRecommender"].get<bool>();
      auto rt = std::dynamic_pointer_cast<RuntimeState>(
          pool_->getActive()->runtime());
      if (!barrier_ && rt)
        barrier_ = std::make_shared<MemeBarrier>(rt);
      if (barrier_)
        barrier_->setUseRecommender(on);
      out["applied"]["memebarrierUseRecommender"] = on;
    }
    if (patch.contains("memebarrierUseTorch") &&
        patch["memebarrierUseTorch"].is_boolean()) {
      bool on = patch["memebarrierUseTorch"].get<bool>();
      auto rt = std::dynamic_pointer_cast<RuntimeState>(
          pool_->getActive()->runtime());
      if (!barrier_ && rt)
        barrier_ = std::make_shared<MemeBarrier>(rt);
      if (barrier_)
        barrier_->setUseTorchModels(on);
      out["applied"]["memebarrierUseTorch"] = on;
    }
    if (patch.contains("learningEnabled") &&
        patch["learningEnabled"].is_boolean()) {
      bool on = patch["learningEnabled"].get<bool>();
      if (!on) {
        rlDisabled_ = true;
        advDisabled_ = true;
        gnnGaDisabled_ = true;
        dialogLearningEnabled_ = false;
      } else {
        if (config_.disableLearning) {
          out["warnings"].push_back(
              "learning was CLI-disabled; runtime override enabled");
        } else {
          gnnGaDisabled_ = false;
        }
      }
      out["applied"]["learningEnabled"] = on;
    }
    if (patch.contains("rlEnabled") && patch["rlEnabled"].is_boolean()) {
      bool on = patch["rlEnabled"].get<bool>();
      if (on && (config_.disableLearning || config_.disableRL)) {
        out["warnings"].push_back(
            "rl was CLI-disabled; runtime override enabled");
      }
      rlDisabled_ = !on;
      out["applied"]["rlEnabled"] = on;
    }
    if (patch.contains("advEnabled") && patch["advEnabled"].is_boolean()) {
      bool on = patch["advEnabled"].get<bool>();
      if (on && (config_.disableLearning || config_.disableADV)) {
        out["warnings"].push_back(
            "adv was CLI-disabled; runtime override enabled");
      }
      advDisabled_ = !on;
      out["applied"]["advEnabled"] = on;
    }
    if (patch.contains("gnnGaEnabled") && patch["gnnGaEnabled"].is_boolean()) {
      bool on = patch["gnnGaEnabled"].get<bool>();
      if (on && config_.disableLearning) {
        out["warnings"].push_back(
            "gnnGa was CLI-disabled; runtime override enabled");
      }
      gnnGaDisabled_ = !on;
      out["applied"]["gnnGaEnabled"] = on;
    }
    if (patch.contains("dialogLearningEnabled") &&
        patch["dialogLearningEnabled"].is_boolean()) {
      dialogLearningEnabled_ = patch["dialogLearningEnabled"].get<bool>();
      out["applied"]["dialogLearningEnabled"] = dialogLearningEnabled_;
    }
    if (patch.contains("dialogTransformerLearningEnabled") &&
        patch["dialogTransformerLearningEnabled"].is_boolean()) {
      dialogTransformerLearning_ =
          patch["dialogTransformerLearningEnabled"].get<bool>();
      out["applied"]["dialogTransformerLearningEnabled"] =
          dialogTransformerLearning_;
    }
    if (patch.contains("rlEvery") && patch["rlEvery"].is_number()) {
      double v = patch["rlEvery"].get<double>();
      if (std::isfinite(v) && v > 0) {
        dialogThresholds_.rlEvery = v;
        out["applied"]["rlEvery"] = dialogThresholds_.rlEvery;
      }
    }
    if (patch.contains("advEvery") && patch["advEvery"].is_number()) {
      double v = patch["advEvery"].get<double>();
      if (std::isfinite(v) && v > 0) {
        dialogThresholds_.advEvery = v;
        out["applied"]["advEvery"] = dialogThresholds_.advEvery;
      }
    }
    if (patch.contains("gnnEvery") && patch["gnnEvery"].is_number()) {
      double v = patch["gnnEvery"].get<double>();
      if (std::isfinite(v) && v > 0) {
        dialogThresholds_.gnnEvery = v;
        out["applied"]["gnnEvery"] = dialogThresholds_.gnnEvery;
      }
    }
    if (patch.contains("dataCleaningEnabled") &&
        patch["dataCleaningEnabled"].is_boolean()) {
      dataCleaningEnabled_ = patch["dataCleaningEnabled"].get<bool>();
      out["applied"]["dataCleaningEnabled"] = dataCleaningEnabled_;
    }
    if (patch.contains("dataCleanMaxChars") &&
        patch["dataCleanMaxChars"].is_number_integer()) {
      int v = patch["dataCleanMaxChars"].get<int>();
      if (v >= 128 && v <= 65536) {
        dataCleanMaxChars_ = (size_t)v;
        out["applied"]["dataCleanMaxChars"] = dataCleanMaxChars_;
      }
    }
    if (patch.contains("contextModuleDisabled") &&
        patch["contextModuleDisabled"].is_boolean()) {
      config_.disableContextModule = patch["contextModuleDisabled"].get<bool>();
      out["applied"]["contextModuleDisabled"] = config_.disableContextModule;
    }
    if (patch.contains("contextLayerEnabled") &&
        patch["contextLayerEnabled"].is_boolean()) {
      bool on = patch["contextLayerEnabled"].get<bool>();
      config_.disableContextModule = !on;
      out["applied"]["contextLayerEnabled"] = on;
      out["applied"]["contextModuleDisabled"] = config_.disableContextModule;
    }
    if (patch.contains("gnnModuleDisabled") &&
        patch["gnnModuleDisabled"].is_boolean()) {
      config_.disableGnnModule = patch["gnnModuleDisabled"].get<bool>();
      out["applied"]["gnnModuleDisabled"] = config_.disableGnnModule;
    }
    if (patch.contains("usingOllama") && patch["usingOllama"].is_boolean()) {
      config_.usingOllama = patch["usingOllama"].get<bool>();
      config_.inferenceEnabled = true;
      config_.transformerBackend =
          config_.usingOllama
              ? (config_.ollamaFineTuning ? "ollama-fine-tuning" : "ollama")
              : "llamacpp";
      out["applied"]["usingOllama"] = config_.usingOllama;
      out["applied"]["inferenceEnabled"] = config_.inferenceEnabled;
      out["applied"]["transformerMode"] = config_.transformerBackend;
    }
    if (patch.contains("ollamaFineTuning") &&
        patch["ollamaFineTuning"].is_boolean()) {
      config_.ollamaFineTuning = patch["ollamaFineTuning"].get<bool>();
      if (config_.usingOllama)
        config_.transformerBackend =
            config_.ollamaFineTuning ? "ollama-fine-tuning" : "ollama";
      out["applied"]["ollamaFineTuning"] = config_.ollamaFineTuning;
      out["applied"]["transformerMode"] = config_.transformerBackend;
    }
    if (patch.contains("inferenceEnabled") &&
        patch["inferenceEnabled"].is_boolean()) {
      config_.inferenceEnabled = patch["inferenceEnabled"].get<bool>();
      out["applied"]["inferenceEnabled"] = config_.inferenceEnabled;
    }
    if (patch.contains("transformerMode") &&
        patch["transformerMode"].is_string()) {
      std::string mode = trimCopy(patch["transformerMode"].get<std::string>());
      std::transform(
          mode.begin(), mode.end(), mode.begin(),
          [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
      if (mode == "native") {
        config_.inferenceEnabled = true;
        config_.usingOllama = false;
        config_.ollamaFineTuning = false;
        config_.transformerBackend = "native";
        out["applied"]["transformerMode"] = "native";
      } else if (mode == "ollama") {
        config_.inferenceEnabled = true;
        config_.usingOllama = true;
        config_.ollamaFineTuning = false;
        config_.transformerBackend = "ollama";
        out["applied"]["transformerMode"] = "ollama";
      } else if (mode == "ollama-fine-tuning" || mode == "ollama_fine_tuning") {
        config_.inferenceEnabled = true;
        config_.usingOllama = true;
        config_.ollamaFineTuning = true;
        config_.transformerBackend = "ollama-fine-tuning";
        out["applied"]["transformerMode"] = "ollama-fine-tuning";
      } else if (mode == "llamacpp" || mode == "llama.cpp" ||
                 mode == "llama_cpp") {
        config_.inferenceEnabled = true;
        config_.usingOllama = false;
        config_.ollamaFineTuning = false;
        config_.transformerBackend = "llamacpp";
        out["applied"]["transformerMode"] = "llamacpp";
      } else if (mode == "bitnet") {
        config_.inferenceEnabled = true;
        config_.usingOllama = false;
        config_.ollamaFineTuning = false;
        config_.transformerBackend = "bitnet";
        out["applied"]["transformerMode"] = "bitnet";
      } else if (mode == "off" || mode == "false" || mode == "disabled") {
        config_.inferenceEnabled = false;
        config_.transformerBackend = "off";
        out["applied"]["transformerMode"] = "off";
      } else {
        out["warnings"].push_back(std::string("unknown transformerMode: ") +
                                  mode);
      }
      out["applied"]["inferenceEnabled"] = config_.inferenceEnabled;
      out["applied"]["usingOllama"] = config_.usingOllama;
      out["applied"]["ollamaFineTuning"] = config_.ollamaFineTuning;
      out["applied"]["transformerBackend"] = config_.transformerBackend;
    }
    return out;
  }

  std::shared_ptr<IReinforcementLearner> ensureRL() {
    if (!rl_) {
      auto factory = module_mount::ModuleRegistry::instance()
                         .reinforcementLearnerFactory();
      if (factory)
        rl_ = factory(pool_, getProjectRoot() / "tests");
      else
        rl_ = std::make_shared<ReinforcementLearner>(pool_, getProjectRoot() /
                                                                "tests");
    }
    return rl_;
  }

  std::shared_ptr<IAdversarialLearner> ensureADV() {
    if (!adv_) {
      auto factory =
          module_mount::ModuleRegistry::instance().adversarialLearnerFactory();
      if (factory)
        adv_ = factory(pool_);
      else
        adv_ = std::make_shared<AdversarialLearner>(pool_);
    }
    return adv_;
  }

  std::shared_ptr<IGnnGaLearner> ensureGnnGa() {
    if (!gnnGa_) {
      auto factory =
          module_mount::ModuleRegistry::instance().gnnGaLearnerFactory();
      if (factory)
        gnnGa_ = factory(pool_, getProjectRoot() / "tests");
      else
        gnnGa_ =
            std::make_shared<GnnGaLearner>(pool_, getProjectRoot() / "tests");
    }
    return gnnGa_;
  }

  std::string buildDialogContextLog(const json &result,
                                    const json &payload) const {
    std::string text = payload.value("text", payload.value("message", ""));
    std::string reply = result.value("reply", "");
    std::string sessionId =
        result.value("sessionId", payload.value("sessionId", ""));
    double align = result.value("gnnAlign", 0.0);
    double ctxW = result.value("contextWeight", 0.0);
    std::ostringstream oss;
    oss << "session=" << sessionId << " | align=" << std::fixed
        << std::setprecision(2) << align << " | ctxW=" << std::fixed
        << std::setprecision(2) << ctxW << " | text=" << text
        << " | reply=" << reply;
    return oss.str();
  }

  std::string buildComputeRouteLog(const json &result) const {
    std::ostringstream oss;
    oss << "seeds=";
    if (result.contains("seeds") && result["seeds"].is_array()) {
      int count = 0;
      for (auto &s : result["seeds"]) {
        if (s.is_array() && s.size() >= 2 && s[0].is_string()) {
          if (count++)
            oss << ",";
          oss << s[0].get<std::string>();
        }
        if (count >= 10)
          break;
      }
    }
    oss << " | memes=";
    if (result.contains("memes") && result["memes"].is_array()) {
      int count = 0;
      for (auto &m : result["memes"]) {
        if (!m.is_string())
          continue;
        if (count++)
          oss << ",";
        oss << m.get<std::string>();
        if (count >= 10)
          break;
      }
    }
    if (result.contains("gnnStage2") && result["gnnStage2"].is_string()) {
      oss << " | stage2=" << result["gnnStage2"].get<std::string>();
    }
    return oss.str();
  }

  std::string buildMemorySnapshotLog() const {
    auto ctrl = pool_->getActive();
    if (!ctrl)
      return "snapshot=unavailable";
    auto snap = ctrl->snapshot();
    return std::string("snapshot=") + snap.dump();
  }

  std::string buildMonitoringLog() const {
    if (!barrier_)
      return "memebarrier=disabled";
    auto st = barrier_->getStats();
    std::ostringstream oss;
    oss << "memebarrier scans=" << st.scans << " isolated=" << st.isolated
        << " avgScore=" << std::fixed << std::setprecision(3) << st.avgScore
        << " maxScore=" << std::fixed << std::setprecision(3) << st.maxScore;
    return oss.str();
  }

  void logSecurityEvent(const std::string &message) const {
    if (logger_.enabled())
      logger_.log(LoggerCXX::Type::SECURITY, message);
  }

  void logRouteError(const std::string &route,
                     const std::string &message) const {
    if (logger_.enabled())
      logger_.log(LoggerCXX::Type::ERROR,
                  std::string("route=") + route + " error=" + message);
  }

  void logRouteCompute(const std::string &route,
                       const std::string &message) const {
    if (logger_.enabled())
      logger_.log(LoggerCXX::Type::COMPUTE,
                  std::string("route=") + route + " " + message);
  }

  void setupRoutes() {
    if (config_.httpLog) {
      drogon::app().registerPostHandlingAdvice(
          [](const drogon::HttpRequestPtr &req,
             const drogon::HttpResponsePtr &resp) {
            try {
              int status = resp ? (int)resp->statusCode() : 0;
              std::string path = req ? req->path() : std::string();
              std::string method =
                  req ? req->methodString() : std::string("UNKNOWN");
              std::cout << "[gateway] " << method << " " << path << " -> "
                        << status << std::endl;
            } catch (...) {
            }
          });
    }

    drogon::app().registerHandler(
        "/", [](const drogon::HttpRequestPtr &,
                std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          auto resp = drogon::HttpResponse::newHttpResponse();
          resp->setStatusCode(drogon::k404NotFound);
          resp->setBody("UI moved to auth_frontend_server.cjs (default :5081)");
          cb(resp);
        });

    drogon::app().registerHandler(
        "/api/health",
        [this](const drogon::HttpRequestPtr &,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
          respondJson(cb, json{{"ok", true},
                               {"service", "phoenix-gateway"},
                               {"timestamp", ts}});
        },
        {drogon::Get, drogon::Post});

    drogon::app().registerHandler(
        "/api/graph/chat",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          try {
            auto jsonStringOrEmpty = [](const json &obj,
                                        const char *key) -> std::string {
              if (!obj.is_object() || !obj.contains(key) ||
                  !obj[key].is_string()) {
                return std::string();
              }
              return obj[key].get<std::string>();
            };
            bool ok = true;
            json body = parseRequestBody(req, ok);
            if (!ok)
              return badRequest(cb, "invalid json");
            auto controller = pool_->getActive();
            json graphBody = body;
            graphBody["disableAddon"] = true;
            json result = controller->respond(graphBody);
            logRouteCompute("/api/graph/chat", "ok");
            respondJson(cb, json{{"ok", true}, {"result", result}});
            onDialogCompleted(result, body);
          } catch (const std::exception &e) {
            logRouteError("/api/graph/chat", e.what());
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            logRouteError("/api/graph/chat", "internal error");
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/chat",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          auto routeStarted = std::chrono::steady_clock::now();
          if (config_.httpLog)
            std::cout << "[gateway] ENTER /api/chat" << std::endl;
          auto finalizeMetric = [&](bool ok) {
            auto elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - routeStarted)
                    .count();
            recordRouteMetric("/api/chat", elapsed, ok);
          };
          if (!authOK(req)) {
            finalizeMetric(false);
            return unauthorized(cb);
          }
          try {
            auto jsonStringOrEmpty = [](const json &obj,
                                        const char *key) -> std::string {
              if (!obj.is_object() || !obj.contains(key) ||
                  !obj[key].is_string()) {
                return std::string();
              }
              return obj[key].get<std::string>();
            };
            bool ok = true;
            json body = parseRequestBody(req, ok);
            if (!ok) {
              finalizeMetric(false);
              return badRequest(cb, "invalid json");
            }
            std::string text = body.value("text", body.value("message", ""));
            std::string prelimSessionId = body.value("sessionId", "");
            ingestWorldEvidence(prelimSessionId, text, "");
            bool cleanedText = false;
            text = sanitizeInputText(text, cleanedText);
            if (cleanedText)
              cleanedInputCount_.fetch_add(1, std::memory_order_relaxed);
            if (text.empty()) {
              finalizeMetric(false);
              return badRequest(cb, "text required");
            }

            // Concept matrix encoder: words -> unified unit positions.
            std::string matrixContext;
            if (config_.conceptMatrixEnabled) {
              auto cmStart = std::chrono::steady_clock::now();
              if (config_.httpLog)
                std::cout << "[gateway] /api/chat stage=concept_matrix_start" << std::endl;
              if (config_.conceptMatrixClearPerRequest)
                conceptMatrix_.clear();
              conceptMatrix_.encodeText(text);
              conceptMatrix_.propagate(0);
              matrixContext = conceptMatrix_.toContextString(32, 0.15f);
              if (config_.httpLog) {
                auto cmMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - cmStart).count();
                std::cout << "[gateway] /api/chat stage=concept_matrix_done active="
                          << conceptMatrix_.activeCount()
                          << " elapsedMs=" << cmMs << std::endl;
              }
            }

            bool simpleQuery = false;
            int tokenCount = roughTokenCount(text);
            bool longQuerySafeMode = false;  // Temporarily disabled for debugging
            if (!config_.inferenceEnabled) {
              finalizeMetric(false);
              return respondJson(
                  cb, json{{"ok", false}, {"error", "inference disabled"}},
                  drogon::k503ServiceUnavailable);
            }
            std::string backendMode = trimCopy(config_.transformerBackend);
            std::transform(backendMode.begin(), backendMode.end(),
                           backendMode.begin(), [](unsigned char ch) {
                             return static_cast<char>(std::tolower(ch));
                           });
            if (backendMode.empty())
              backendMode = config_.usingOllama ? (config_.ollamaFineTuning
                                                       ? "ollama-fine-tuning"
                                                       : "ollama")
                                                : "native";
            bool usingOllama = (backendMode == "ollama" ||
                                backendMode == "ollama-fine-tuning");
            bool usingLlamaCpp =
                (backendMode == "llamacpp" || backendMode == "llama.cpp" ||
                 backendMode == "llama_cpp");
            bool usingBitNet = (backendMode == "bitnet");
            bool usingNativeTransformer =
                !(usingOllama || usingLlamaCpp || usingBitNet);
            bool gnnModuleDisabled = config_.disableGnnModule;
            bool contextModuleDisabled = config_.disableContextModule;
            bool useGraphSelector = body.value("enableGraphSelector", false);
            bool hardDisableGraphSelector =
                phoenix::resolveConfig<bool>("main.disableGraphSelector", true, "AI_DISABLE_GRAPH_SELECTOR");
            if (hardDisableGraphSelector)
              useGraphSelector = false;
            if (gnnModuleDisabled)
              useGraphSelector = false;
            if (usingNativeTransformer) {
              std::lock_guard<std::mutex> lock(transformerMu_);
              auto tp = transformer_.params();
              if (tp.value("enableDynamicReasoning", true)) {
                int th = std::max(2, tp.value("simpleTokenThreshold", 10));
                simpleQuery = tokenCount > 0 && tokenCount <= th;
              }
            }
            std::string imageContext = extractImageContext(body);
            if (contextModuleDisabled)
              imageContext.clear();
            json graphResult = json::object();
            if (useGraphSelector && !longQuerySafeMode) {
              auto graphStart = std::chrono::steady_clock::now();
              if (config_.httpLog)
                std::cout << "[gateway] /api/chat stage=graph_selector_start"
                          << std::endl;
              json bodyForGraph = body;
              injectImageContext(bodyForGraph, imageContext);
              bodyForGraph["disableAddon"] = true;
              auto controller = pool_->getActive();
              graphResult = controller->respond(bodyForGraph);
              if (config_.httpLog) {
                auto graphMs =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - graphStart)
                        .count();
                std::cout << "[gateway] /api/chat stage=graph_selector_done "
                             "elapsedMs="
                          << graphMs << std::endl;
              }
            }
            std::string graphContext;
            if (!imageContext.empty()) {
              if (graphContext.empty())
                graphContext = imageContext;
              else
                graphContext = graphContext + "\n" + imageContext;
            }
            std::string graphReplyText = graphResult.is_object()
                                             ? graphResult.value("reply", "")
                                             : std::string();
            if (!graphReplyText.empty()) {
              if (graphContext.empty())
                graphContext = graphReplyText;
              else
                graphContext = graphContext + "\n" + graphReplyText;
            }
            if (!matrixContext.empty()) {
              if (graphContext.empty())
                graphContext = matrixContext;
              else
                graphContext = graphContext + "\n" + matrixContext;
            }
            int maxTokens = graphResult.value(
                "maxTokens", phoenix::cfgOr<int>("context.maxTokens", 8192));
            if (body.contains("maxTokens") &&
                body["maxTokens"].is_number_integer())
              maxTokens = body["maxTokens"].get<int>();
            if (longQuerySafeMode)
              maxTokens = std::min(maxTokens, 32);
            if (maxTokens <= 0) maxTokens = 16;
            auto hint = extractContextHint(body);

            // The frontend proxy already runs prepareChatContext and injects
            // contextHint/contextWeight into the body.  Avoid a re-entrant
            // HTTP call from the gateway back to the frontend (same process,
            // same Drogon event loop) which deadlocks the /api/chat handler.
            std::cout << "[gateway] /api/chat contextHint=" << (hint.text.empty() ? "empty" : "present")
                      << ", sessionId=" << body.value("sessionId", "") << std::endl;

            GnnHintSummary gnnHint;
            if (!gnnModuleDisabled) {
              gnnHint = buildGnnHintSummary(graphResult, body, hint);
              hint = harmonizeHintWithGnn(hint, gnnHint.align);
            }
            std::string sparkAnn;
            bool sparkGnnScheduler = false;
            try {
              sparkGnnScheduler = phoenix::cfgOr<bool>("spark.gnnScheduler.enabled", false);
            } catch (...) {}
            if (sparkGnnScheduler && spark_ && !gnnModuleDisabled) {
              try {
                json sparkPayload;
                sparkPayload["sessionId"] = body.value("sessionId", "");
                sparkPayload["text"] = text;
                sparkPayload["graphResult"] = graphResult;
                sparkPayload["gnnAlign"] = gnnHint.align;
                json sparkOptions;
                sparkOptions["perturbations"] = phoenix::cfgOr<int>("spark.gnnScheduler.perturbations", 2);
                sparkOptions["multiLayer"] = phoenix::cfgOr<bool>("spark.gnnScheduler.useHistory", false);
                auto sparkResult = spark_->dispatch(sparkPayload, sparkOptions);
                sparkAnn = sparkResult.value("aggregate", json::object()).value("annotation", "");
              } catch (...) {}
            }
            if (!sparkAnn.empty()) {
              if (graphContext.empty())
                graphContext = sparkAnn;
              else
                graphContext = graphContext + "\n" + sparkAnn;
            }

            // TinyLlama post-GNN summarization step: convert GNN tokens/outputs
            // into a coherent sentence before the main LLM decodes the final reply.
            std::string tinyllamaSummary;
            if (!gnnModuleDisabled && config_.tinyllamaEnabled) {
              auto tinyStart = std::chrono::steady_clock::now();
              if (config_.httpLog)
                std::cout << "[gateway] /api/chat stage=tinyllama_start" << std::endl;
              try {
                std::string tinyInput = graphReplyText;
                if (tinyInput.empty())
                  tinyInput = text;
                json tinyOptions = json{{"num_predict", 128}, {"temperature", 0.3}};
                json tinyReply = chatWithTinyllama(tinyInput, "", 128, tinyOptions);
                if (tinyReply.value("ok", false)) {
                  tinyllamaSummary = tinyReply.value("reply", "");
                }
              } catch (const std::exception &e) {
                if (config_.httpLog)
                  std::cout << "[gateway] /api/chat stage=tinyllama_exception error="
                            << e.what() << std::endl;
              }
              if (!tinyllamaSummary.empty()) {
                if (graphContext.empty())
                  graphContext = tinyllamaSummary;
                else
                  graphContext = graphContext + "\n" + tinyllamaSummary;
              }
              if (config_.httpLog) {
                auto tinyMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - tinyStart).count();
                std::cout << "[gateway] /api/chat stage=tinyllama_done summary_length="
                          << tinyllamaSummary.size() << " elapsedMs=" << tinyMs << std::endl;
              }
            }

            // Preserve the original user message for the emotion system.
            std::string userText = text;

            // Always apply session-level context hint; only mix graph context when context module is enabled.
            if (!contextModuleDisabled) {
              text = applyContextHintToText(text, hint);
              graphContext = applyContextHintToGraph(graphContext, hint);
            } else {
              graphContext.clear();
              text = applyContextHintToText(text, hint);
            }
            bool cleanedGraph = false;
            graphContext = sanitizeInputText(graphContext, cleanedGraph);
            if (cleanedGraph)
              cleanedInputCount_.fetch_add(1, std::memory_order_relaxed);
            bool addonContractEnabled =
                body.value("enableAddonToolContract",
                           body.value("enableAddonTools", false));
            if (longQuerySafeMode)
              addonContractEnabled = false;
            if (!simpleQuery && addonContractEnabled) {
              graphContext = attachAddonToolContract(graphContext);
            }
            std::string sessionId = body.value("sessionId", "");

            ingestWorldEvidence(sessionId, userText, graphContext);

            // v7.0 ahead-of-GNN branch: memory + emotion on raw user input.
            std::string emotionModulation;
            nlohmann::json emotionLogitBias;
            std::string emotionContextStr;
            nlohmann::json aheadInferenceOptions;
            nlohmann::json inferenceOptions;
            nlohmann::json cognitionEmotionTensor;
            nlohmann::json cognitionWorldEvidence;
            std::vector<nlohmann::json> cognitionMixedModalOutputs;
            std::string cognitionContext;
            if (aheadModule_ && !sessionId.empty()) {
              try {
                phoenix::io::MixedModalPacket pkt;
                pkt.modality = phoenix::io::MixedModalModality::Text;
                pkt.payload.assign(userText.begin(), userText.end());
                pkt.source = "gateway-chat";
                auto ahead = aheadModule_->processBatch({pkt}, sessionId);
                if (ahead.memoryReady) {
                  if (!graphContext.empty())
                    graphContext += "\n";
                  graphContext += "[Ahead memory]\n" + ahead.memoryForBackend.content;
                }
                emotionModulation = ahead.emotionInfluence.promptModulation;
                aheadInferenceOptions = ahead.emotionInfluence.inferenceOptions;
                if (!ahead.emotionInfluence.logitBias.empty())
                  emotionLogitBias = ahead.emotionInfluence.logitBias;
                cognitionEmotionTensor = ahead.emotion.toJson();
              } catch (...) {
              }
            }

            // The emotion system is reachable through the frontend, but the
            // gateway must not make re-entrant HTTP calls to the same process.
            // Keep emotion fields empty here; runCognitionAutonomy may later
            // fill cognitionEmotionTensor from the autonomy stack directly.

            // v7.0: run the autonomy cognition stack to observe world state,
            // iterate the primal/instinct/prompt layers and obtain affect-driven
            // context + inference options.
            try {
              auto pack = runCognitionAutonomy(body, userText, graphContext);
              graphContext = pack.context;
              if (pack.inferenceOptions.is_object()) {
                for (auto &el : pack.inferenceOptions.items())
                  inferenceOptions[el.key()] = el.value();
              }
              if (pack.emotionTensor.is_object())
                cognitionEmotionTensor = pack.emotionTensor;
              cognitionWorldEvidence = pack.worldEvidence;
              cognitionMixedModalOutputs = pack.mixedModalOutputs;
            } catch (...) {
            }

            if (inferenceOptions.is_object() && aheadInferenceOptions.is_object()) {
              for (auto &el : aheadInferenceOptions.items())
                inferenceOptions[el.key()] = el.value();
            } else if (inferenceOptions.is_null() && aheadInferenceOptions.is_object()) {
              inferenceOptions = aheadInferenceOptions;
            }

            std::string providerTag = std::string("provider:") + backendMode;
            std::string cacheKey = buildInferenceCacheKey(
                sessionId, text, providerTag + "\n" + graphContext, maxTokens);
            json tReply;
            json verify;
            bool cacheHit = false;
            long long modelElapsedMs = 0;
            long long verifyElapsedMs = 0;
            int imageEmbCount = 0;
            if (db_) {
              json cached;
              if (db_->getInferenceCache(cacheKey, cached)) {
                cacheHit = true;
                tReply = cached.contains("tReply") ? cached["tReply"] : cached;
                if (cached.contains("verify"))
                  verify = cached["verify"];
              }
            }
            if (cacheHit && verify.is_null()) {
              verify = json{{"ok", true}, {"cached", true}};
            }
            if (!cacheHit) {
              // Prevent running the heavy LLM when the system is already
              // critically low on memory.  This is a guard rail, not a
              // business-logic change.
              std::string memErr;
              if (!preInferenceMemoryCheck(memErr)) {
                appendFatalLog(std::string("oom_guard /api/chat: ") + memErr +
                               " maxTokens=" + std::to_string(maxTokens));
                logRouteError("/api/chat",
                              std::string("oom_guard: ") + memErr);
                respondJson(cb,
                            json{{"ok", false},
                                 {"error", memErr},
                                 {"stage", "oom_guard"}},
                            drogon::k503ServiceUnavailable);
                finalizeMetric(false);
                return;
              }

              if (usingLlamaCpp || usingBitNet)
                dispatchExternalStyleTrainStep(backendMode, body, text,
                                               gnnHint.align);
              std::string modelGraphContext = graphContext;
              if (usingOllama) {
                auto t0 = std::chrono::steady_clock::now();
                if (config_.httpLog)
                  std::cout << "[gateway] /api/chat stage=model_start "
                               "provider=ollama maxTokens="
                            << maxTokens << std::endl;
                tReply = chatWithOllama(text, modelGraphContext, maxTokens,
                        inferenceOptions);
                modelElapsedMs =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
                if (config_.httpLog)
                  std::cout << "[gateway] /api/chat stage=model_done "
                               "provider=ollama elapsedMs="
                            << modelElapsedMs << std::endl;
              } else if (usingLlamaCpp) {
                auto t0 = std::chrono::steady_clock::now();
                if (config_.httpLog)
                  std::cout << "[gateway] /api/chat stage=model_start "
                               "provider=llamacpp maxTokens="
                            << maxTokens << std::endl;
                tReply = chatWithLlamaCpp(text, modelGraphContext, maxTokens,
                          inferenceOptions);
                modelElapsedMs =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
                if (config_.httpLog)
                  std::cout << "[gateway] /api/chat stage=model_done "
                               "provider=llamacpp elapsedMs="
                            << modelElapsedMs << std::endl;
              } else if (usingBitNet) {
                auto t0 = std::chrono::steady_clock::now();
                if (config_.httpLog)
                  std::cout << "[gateway] /api/chat stage=model_start "
                               "provider=bitnet maxTokens="
                            << maxTokens << std::endl;
                tReply = chatWithBitNet(text, modelGraphContext, maxTokens,
                        inferenceOptions);
                modelElapsedMs =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
                if (config_.httpLog)
                  std::cout << "[gateway] /api/chat stage=model_done "
                               "provider=bitnet elapsedMs="
                            << modelElapsedMs << std::endl;
              } else {
                auto t0 = std::chrono::steady_clock::now();
                if (config_.httpLog)
                  std::cout << "[gateway] /api/chat stage=model_start "
                               "provider=transformer maxTokens="
                            << maxTokens << std::endl;
                std::lock_guard<std::mutex> lock(transformerMu_);
                // Pre-inference memory check to avoid OOM kills
                std::string memErr;
                if (!preInferenceMemoryCheck(memErr)) {
                  tReply = json{{"ok", false}, {"error", memErr}};
                  modelElapsedMs = 0;
                  if (config_.httpLog)
                    std::cout << "[gateway] /api/chat stage=model_oom_prevented "
                              << memErr << std::endl;
                } else {
                  int dim = transformer_.params().value("dModel", 128);
                  int rounds = gnnHint.align >= 0.45 ? 2 : 1;
                  std::vector<std::vector<float>> graphEmb;
                  bool enableGraphEmbeddings =
                      body.value("enableGraphEmbeddings", false);
                  if (enableGraphEmbeddings) {
                    graphEmb = buildGraphEmbeddings(graphResult, dim, rounds);
                  }
                  auto imageEmb = extractImageEmbeddings(body, dim);
                  if (!imageEmb.empty()) {
                    imageEmbCount = (int)imageEmb.size();
                    graphEmb.insert(graphEmb.end(), imageEmb.begin(),
                                    imageEmb.end());
                  }
                  tReply = transformer_.chat(text, modelGraphContext, graphEmb,
                                             maxTokens);
                  postInferenceMemoryTrim();
                  modelElapsedMs =
                      std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
                  if (config_.httpLog)
                    std::cout << "[gateway] /api/chat stage=model_done "
                                 "provider=transformer elapsedMs="
                              << modelElapsedMs << std::endl;
                }
              }
              std::string replyCandidate = tReply.value("reply", "");
              if (longQuerySafeMode) {
                verify = json{{"ok", false},
                              {"skipped", true},
                              {"reason", "long-query-safe-mode"}};
              } else if (!usingNativeTransformer) {
                verify = json{{"ok", false},
                              {"skipped", true},
                              {"reason", std::string("using-") + backendMode}};
              } else {
                auto v0 = std::chrono::steady_clock::now();
                if (config_.httpLog)
                  std::cout << "[gateway] /api/chat stage=verify_start"
                            << std::endl;
                std::lock_guard<std::mutex> lock(transformerMu_);
                verify =
                    transformer_.verify(text, graphContext, replyCandidate);
                verifyElapsedMs =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - v0)
                        .count();
                if (config_.httpLog)
                  std::cout
                      << "[gateway] /api/chat stage=verify_done elapsedMs="
                      << verifyElapsedMs << std::endl;
              }
              if (db_) {
                db_->setInferenceCache(
                    cacheKey, json{{"tReply", tReply}, {"verify", verify}},
                    86400);
              }
            }
            std::string transformerReplyRaw = tReply.value("reply", "");
            std::string reply = stripAddonToolDirective(transformerReplyRaw);
            addon::AddonResult addonExec;
            AddonToolCall addonCall;
            bool addonUsed = false;
            if (addonContractEnabled) {
              addonCall = parseAddonToolCall(transformerReplyRaw, text);
              addonUsed = invokeAddonTool(addonCall, body, addonExec);
            }
            if (addonUsed) {
              std::string followupGraphContext =
                  buildAddonResultContext(graphContext, addonCall, addonExec);
              if (usingLlamaCpp || usingBitNet)
                dispatchExternalStyleTrainStep(backendMode, body, text,
                                               gnnHint.align);
              json tReplyAfterTool;
              if (usingOllama) {
                tReplyAfterTool =
                    chatWithOllama(text, followupGraphContext, maxTokens,
                   inferenceOptions);
              } else if (usingLlamaCpp) {
                tReplyAfterTool =
                    chatWithLlamaCpp(text, followupGraphContext, maxTokens,
                     inferenceOptions);
              } else if (usingBitNet) {
                tReplyAfterTool =
                    chatWithBitNet(text, followupGraphContext, maxTokens,
                   inferenceOptions);
              } else {
                std::lock_guard<std::mutex> lock(transformerMu_);
                tReplyAfterTool =
                    transformer_.chat(text, followupGraphContext, maxTokens);
                postInferenceMemoryTrim();
              }
              std::string rewritten =
                  stripAddonToolDirective(tReplyAfterTool.value("reply", ""));
              if (!rewritten.empty()) {
                reply = rewritten;
                transformerReplyRaw =
                    tReplyAfterTool.value("reply", transformerReplyRaw);
              } else {
                reply = addonExec.reply;
              }
              if (usingNativeTransformer) {
                std::string source =
                    "addon:" +
                    addonExec.meta.value("addon", addonCall.type.empty()
                                                      ? std::string("auto")
                                                      : addonCall.type);
                std::lock_guard<std::mutex> lock(transformerMu_);
                transformer_.addPreferenceFeedback(
                    text, graphContext, reply,
                    stripAddonToolDirective(transformerReplyRaw), 1.0f, source);
              }
            }
            if (aheadModule_ && !sessionId.empty() && !reply.empty()) {
              try {
                auto routed = aheadModule_->routeSentence(reply, sessionId);
                if (!routed.blockedByMemeBarrier)
                  reply = routed.textForBackend;
              } catch (...) {
              }
            }
            json result = graphResult;
            result["graph"] = graphResult;
            result["graphReply"] = graphResult.value("reply", "");
            result["reply"] = reply;
            result["transformerReply"] = transformerReplyRaw;
            result["graphContext"] = graphContext;
            result["gnnAlign"] = gnnHint.align;
            if (!gnnHint.annotation.empty())
              result["gnnStage2"] = gnnHint.annotation;
            if (!sparkAnn.empty())
              result["sparkGnnStage2"] = sparkAnn;
            if (!hint.text.empty()) {
              result["contextHint"] = hint.text;
              result["contextWeight"] = hint.weight;
              if (!hint.mode.empty())
                result["contextMode"] = hint.mode;
            }
            if (!imageContext.empty())
              result["imageContext"] = imageContext;
            if (imageEmbCount > 0)
              result["imageEmbeddingCount"] = imageEmbCount;
            if (addonUsed) {
              result["addon"] = addonExec.meta;
              result["toolCall"] =
                  json{{"type", addonCall.type}, {"input", addonCall.input}};
            }
            result["verify"] = verify;
            result["cached"] = cacheHit;
            result["longQuerySafeMode"] = longQuerySafeMode;
            result["gnnModuleDisabled"] = gnnModuleDisabled;
            result["contextModuleDisabled"] = contextModuleDisabled;
            result["usingOllama"] = usingOllama;
            result["transformerMode"] = backendMode;
            result["modelElapsedMs"] = (Json::Int64)modelElapsedMs;
            result["verifyElapsedMs"] = (Json::Int64)verifyElapsedMs;
            if (!emotionModulation.empty())
              result["emotionModulation"] = emotionModulation;
            if (!emotionLogitBias.empty())
              result["emotionLogitBias"] = emotionLogitBias;
            if (!inferenceOptions.is_null())
              result["cognitionInferenceOptions"] = inferenceOptions;
            if (!cognitionEmotionTensor.is_null())
              result["cognitionEmotionTensor"] = cognitionEmotionTensor;
            if (!cognitionWorldEvidence.is_null() && !cognitionWorldEvidence.empty())
              result["cognitionWorldEvidence"] = cognitionWorldEvidence;
            if (!cognitionMixedModalOutputs.empty())
              result["cognitionMixedModalOutputs"] = cognitionMixedModalOutputs;
            if (usingOllama)
              result["provider"] =
                  json{{"id", (config_.ollamaFineTuning ? "ollama+fine-tuning"
                                                        : "ollama")},
                       {"model", config_.ollamaModel}};
            else if (usingLlamaCpp)
              result["provider"] =
                  json{{"id", "llamacpp"}, {"model", config_.llamaCppModel}};
            else if (usingBitNet)
              result["provider"] =
                  json{{"id", "bitnet"}, {"model", config_.bitnetModel}};
            else
              result["provider"] =
                  json{{"id", "core"}, {"model", "native-transformer"}};
            double verifyScore = verify.value("score", 0.0);
            (void)verifyScore;
            auto totalElapsedMs =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - routeStarted)
                    .count();
            logRouteCompute("/api/chat",
                            std::string("ok cached=") +
                                (cacheHit ? "true" : "false") +
                                " addon=" + (addonUsed ? "true" : "false") +
                                " modelMs=" + std::to_string(modelElapsedMs) +
                                " verifyMs=" + std::to_string(verifyElapsedMs) +
                                " totalMs=" + std::to_string(totalElapsedMs) +
                                " provider=" + backendMode);
            respondJson(cb, json{{"ok", true}, {"result", result}});
            onDialogCompleted(result, body);
            // Record assistant reply back to frontend context service for next turn compression.
            // If the frontend already provided a contextHint, it will record the assistant reply itself.
            if (!body.contains("contextHint") && !sessionId.empty() && result.contains("reply") && result["reply"].is_string()) {
              std::string assistantReply = result["reply"].get<std::string>();
              if (!assistantReply.empty()) {
                try {
                  auto ctxClient = drogon::HttpClient::newHttpClient("http://127.0.0.1:5081");
                  if (ctxClient) {
                    json recordJson;
                    recordJson["sessionId"] = sessionId;
                    recordJson["reply"] = assistantReply;
                    auto recordReq = drogon::HttpRequest::newHttpRequest();
                    recordReq->setMethod(drogon::Post);
                    recordReq->setPath("/context/record");
                    recordReq->addHeader("Authorization", "Bearer local-dev");
                    recordReq->setContentTypeCode(drogon::CT_APPLICATION_JSON);
                    recordReq->setBody(recordJson.dump());
                    ctxClient->sendRequest(recordReq, [](drogon::ReqResult, const drogon::HttpResponsePtr &) {});
                  }
                } catch (...) {
                }
              }
            }
            finalizeMetric(true);
          } catch (const std::exception &e) {
            logRouteError("/api/chat", e.what());
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
            finalizeMetric(false);
          } catch (...) {
            logRouteError("/api/chat", "internal error");
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
            finalizeMetric(false);
          }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/chat",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          logRouteError("/api/chat", std::string("method-not-allowed:") +
                                         req->methodString());
          respondJson(cb,
                      json{{"ok", false},
                           {"error", "method not allowed"},
                           {"allowed", "POST"}},
                      drogon::k405MethodNotAllowed);
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/api/transformer/chat",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          auto routeStarted = std::chrono::steady_clock::now();
          if (config_.httpLog)
            std::cout << "[gateway] ENTER /api/transformer/chat" << std::endl;
          auto finalizeMetric = [&](bool ok) {
            auto elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - routeStarted)
                    .count();
            recordRouteMetric("/api/transformer/chat", elapsed, ok);
          };
          if (!authOK(req)) {
            finalizeMetric(false);
            return unauthorized(cb);
          }
          try {
            auto jsonStringOrEmpty = [](const json &obj,
                                        const char *key) -> std::string {
              if (!obj.is_object() || !obj.contains(key) ||
                  !obj[key].is_string()) {
                return std::string();
              }
              return obj[key].get<std::string>();
            };
            bool ok = true;
            json body = parseRequestBody(req, ok);
            if (!ok) {
              finalizeMetric(false);
              return badRequest(cb, "invalid json");
            }
            std::string text = body.value("text", body.value("message", ""));
            std::string prelimSessionId = body.value("sessionId", "");
            ingestWorldEvidence(prelimSessionId, text, "");
            bool cleanedText = false;
            text = sanitizeInputText(text, cleanedText);
            if (cleanedText)
              cleanedInputCount_.fetch_add(1, std::memory_order_relaxed);
            if (text.empty()) {
              finalizeMetric(false);
              return badRequest(cb, "text required");
            }

            // Concept matrix encoder: words -> unified unit positions.
            std::string matrixContext;
            if (config_.conceptMatrixEnabled) {
              auto cmStart = std::chrono::steady_clock::now();
              if (config_.httpLog)
                std::cout << "[gateway] /api/chat stage=concept_matrix_start" << std::endl;
              if (config_.conceptMatrixClearPerRequest)
                conceptMatrix_.clear();
              conceptMatrix_.encodeText(text);
              conceptMatrix_.propagate(0);
              matrixContext = conceptMatrix_.toContextString(32, 0.15f);
              if (config_.httpLog) {
                auto cmMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - cmStart).count();
                std::cout << "[gateway] /api/chat stage=concept_matrix_done active="
                          << conceptMatrix_.activeCount()
                          << " elapsedMs=" << cmMs << std::endl;
              }
            }

            bool simpleQuery = false;
            int tokenCount = roughTokenCount(text);
            bool longQuerySafeMode = false;  // Temporarily disabled for debugging
            if (!config_.inferenceEnabled) {
              finalizeMetric(false);
              return respondJson(
                  cb, json{{"ok", false}, {"error", "inference disabled"}},
                  drogon::k503ServiceUnavailable);
            }
            std::string backendMode = trimCopy(config_.transformerBackend);
            std::transform(backendMode.begin(), backendMode.end(),
                           backendMode.begin(), [](unsigned char ch) {
                             return static_cast<char>(std::tolower(ch));
                           });
            if (backendMode.empty())
              backendMode = config_.usingOllama ? (config_.ollamaFineTuning
                                                       ? "ollama-fine-tuning"
                                                       : "ollama")
                                                : "native";
            bool usingOllama = (backendMode == "ollama" ||
                                backendMode == "ollama-fine-tuning");
            bool usingLlamaCpp =
                (backendMode == "llamacpp" || backendMode == "llama.cpp" ||
                 backendMode == "llama_cpp");
            bool usingBitNet = (backendMode == "bitnet");
            bool usingNativeTransformer =
                !(usingOllama || usingLlamaCpp || usingBitNet);
            bool gnnModuleDisabled = config_.disableGnnModule;
            bool contextModuleDisabled = config_.disableContextModule;
            bool useGraphSelector = body.value("enableGraphSelector", false);
            bool hardDisableGraphSelector =
                phoenix::resolveConfig<bool>("main.disableGraphSelector", true, "AI_DISABLE_GRAPH_SELECTOR");
            if (hardDisableGraphSelector)
              useGraphSelector = false;
            if (gnnModuleDisabled)
              useGraphSelector = false;
            if (usingNativeTransformer) {
              std::lock_guard<std::mutex> lock(transformerMu_);
              auto tp = transformer_.params();
              if (tp.value("enableDynamicReasoning", true)) {
                int th = std::max(2, tp.value("simpleTokenThreshold", 10));
                simpleQuery = tokenCount > 0 && tokenCount <= th;
              }
            }
            std::string graphContext = body.value("graphContext", "");
            auto hint = extractContextHint(body);
            json graphResult;
            std::string imageContext = extractImageContext(body);
            if (contextModuleDisabled) {
              graphContext.clear();
              imageContext.clear();
            }
            if (useGraphSelector && graphContext.empty() &&
                !longQuerySafeMode) {
              json bodyForGraph = body;
              injectImageContext(bodyForGraph, imageContext);
              bodyForGraph["disableAddon"] = true;
              auto controller = pool_->getActive();
              graphResult = controller->respond(bodyForGraph);
            }
            if (!imageContext.empty()) {
              if (graphContext.empty())
                graphContext = imageContext;
              else
                graphContext = graphContext + "\n" + imageContext;
            }
            std::string graphReplyText = graphResult.is_object()
                                             ? graphResult.value("reply", "")
                                             : std::string();
            if (!graphReplyText.empty()) {
              if (graphContext.empty())
                graphContext = graphReplyText;
              else
                graphContext = graphContext + "\n" + graphReplyText;
            }
            if (!matrixContext.empty()) {
              if (graphContext.empty())
                graphContext = matrixContext;
              else
                graphContext = graphContext + "\n" + matrixContext;
            }
            GnnHintSummary gnnHint;
            if (!gnnModuleDisabled) {
              gnnHint = buildGnnHintSummary(graphResult, body, hint);
              hint = harmonizeHintWithGnn(hint, gnnHint.align);
            }
            std::string sparkAnn;
            bool sparkGnnScheduler = false;
            try {
              sparkGnnScheduler = phoenix::cfgOr<bool>("spark.gnnScheduler.enabled", false);
            } catch (...) {}
            if (sparkGnnScheduler && spark_ && !gnnModuleDisabled) {
              try {
                json sparkPayload;
                sparkPayload["sessionId"] = body.value("sessionId", "");
                sparkPayload["text"] = text;
                sparkPayload["graphResult"] = graphResult;
                sparkPayload["gnnAlign"] = gnnHint.align;
                json sparkOptions;
                sparkOptions["perturbations"] = phoenix::cfgOr<int>("spark.gnnScheduler.perturbations", 2);
                sparkOptions["multiLayer"] = phoenix::cfgOr<bool>("spark.gnnScheduler.useHistory", false);
                auto sparkResult = spark_->dispatch(sparkPayload, sparkOptions);
                sparkAnn = sparkResult.value("aggregate", json::object()).value("annotation", "");
              } catch (...) {}
            }
            if (!sparkAnn.empty()) {
              if (graphContext.empty())
                graphContext = sparkAnn;
              else
                graphContext = graphContext + "\n" + sparkAnn;
            }
            if (!contextModuleDisabled) {
              text = applyContextHintToText(text, hint);
              graphContext = applyContextHintToGraph(graphContext, hint);
            } else {
              graphContext.clear();
              // User-provided contextHint is still applied when context module is disabled
              if (!hint.text.empty()) {
                text = applyContextHintToText(text, hint);
              }
            }
            bool cleanedGraph = false;
            graphContext = sanitizeInputText(graphContext, cleanedGraph);
            if (cleanedGraph)
              cleanedInputCount_.fetch_add(1, std::memory_order_relaxed);
            bool addonContractEnabled =
                body.value("enableAddonToolContract",
                           body.value("enableAddonTools", false));
            if (longQuerySafeMode)
              addonContractEnabled = false;
            if (!simpleQuery && addonContractEnabled) {
              graphContext = attachAddonToolContract(graphContext);
            }
            int maxTokens = body.value(
                "maxTokens", phoenix::cfgOr<int>("context.maxTokens", 8192));
            if (longQuerySafeMode)
              maxTokens = std::min(maxTokens, 32);
            if (maxTokens <= 0) maxTokens = 16;
            std::string sessionId = body.value("sessionId", "");

            ingestWorldEvidence(sessionId, text, graphContext);

            // v7.0 ahead-of-GNN branch: memory + emotion on raw user input.
            nlohmann::json inferenceOptions;
            nlohmann::json cognitionEmotionTensor;
            nlohmann::json aheadInferenceOptions;
            if (aheadModule_ && !sessionId.empty()) {
              try {
                phoenix::io::MixedModalPacket pkt;
                pkt.modality = phoenix::io::MixedModalModality::Text;
                pkt.payload.assign(text.begin(), text.end());
                pkt.source = "gateway-chat";
                auto ahead = aheadModule_->processBatch({pkt}, sessionId);
                if (ahead.memoryReady) {
                  if (!graphContext.empty())
                    graphContext += "\n";
                  graphContext += "[Ahead memory]\n" + ahead.memoryForBackend.content;
                }
                aheadInferenceOptions = ahead.emotionInfluence.inferenceOptions;
                cognitionEmotionTensor = ahead.emotion.toJson();
              } catch (...) {
              }
            }

            nlohmann::json cognitionWorldEvidence;
            std::vector<nlohmann::json> cognitionMixedModalOutputs;
            try {
              auto pack = runCognitionAutonomy(body, text, graphContext);
              graphContext = pack.context;
              if (pack.inferenceOptions.is_object()) {
                for (auto &el : pack.inferenceOptions.items())
                  inferenceOptions[el.key()] = el.value();
              }
              if (pack.emotionTensor.is_object())
                cognitionEmotionTensor = pack.emotionTensor;
              cognitionWorldEvidence = pack.worldEvidence;
              cognitionMixedModalOutputs = pack.mixedModalOutputs;
            } catch (...) {
            }

            if (inferenceOptions.is_object() && aheadInferenceOptions.is_object()) {
              for (auto &el : aheadInferenceOptions.items())
                inferenceOptions[el.key()] = el.value();
            } else if (inferenceOptions.is_null() && aheadInferenceOptions.is_object()) {
              inferenceOptions = aheadInferenceOptions;
            }
            std::string providerTag = std::string("provider:") + backendMode;
            std::string cacheKey = buildInferenceCacheKey(
                sessionId, text, providerTag + "\n" + graphContext, maxTokens);
            json tReply;
            json verify;
            bool cacheHit = false;
            int imageEmbCount = 0;
            if (db_) {
              json cached;
              if (db_->getInferenceCache(cacheKey, cached)) {
                cacheHit = true;
                tReply = cached.contains("tReply") ? cached["tReply"] : cached;
                if (cached.contains("verify"))
                  verify = cached["verify"];
              }
            }
            if (cacheHit && verify.is_null()) {
              verify = json{{"ok", true}, {"cached", true}};
            }
            if (!cacheHit) {
              // Prevent running the heavy LLM when the system is already
              // critically low on memory.  This is a guard rail, not a
              // business-logic change.
              std::string memErr;
              if (!preInferenceMemoryCheck(memErr)) {
                appendFatalLog(std::string("oom_guard /api/transformer/chat: ") + memErr +
                               " maxTokens=" + std::to_string(maxTokens));
                logRouteError("/api/transformer/chat",
                              std::string("oom_guard: ") + memErr);
                respondJson(cb,
                            json{{"ok", false},
                                 {"error", memErr},
                                 {"stage", "oom_guard"}},
                            drogon::k503ServiceUnavailable);
                finalizeMetric(false);
                return;
              }

              if (usingLlamaCpp || usingBitNet)
                dispatchExternalStyleTrainStep(backendMode, body, text,
                                               gnnHint.align);
              std::string modelGraphContext = graphContext;
              if (usingOllama) {
                tReply = chatWithOllama(text, modelGraphContext, maxTokens,
                        inferenceOptions);
              } else if (usingLlamaCpp) {
                tReply = chatWithLlamaCpp(text, modelGraphContext, maxTokens,
                          inferenceOptions);
              } else if (usingBitNet) {
                tReply = chatWithBitNet(text, modelGraphContext, maxTokens,
                        inferenceOptions);
              } else {
                std::lock_guard<std::mutex> lock(transformerMu_);
                int dim = transformer_.params().value("dModel", 128);
                int rounds = gnnHint.align >= 0.45 ? 2 : 1;
                std::vector<std::vector<float>> graphEmb;
                bool enableGraphEmbeddings =
                    body.value("enableGraphEmbeddings", false);
                if (enableGraphEmbeddings) {
                  graphEmb = buildGraphEmbeddings(graphResult, dim, rounds);
                }
                auto imageEmb = extractImageEmbeddings(body, dim);
                if (!imageEmb.empty()) {
                  imageEmbCount = (int)imageEmb.size();
                  graphEmb.insert(graphEmb.end(), imageEmb.begin(),
                                  imageEmb.end());
                }
                tReply = transformer_.chat(text, modelGraphContext, graphEmb,
                                           maxTokens);
                postInferenceMemoryTrim();
              }
              if (!tReply.is_object()) {
                finalizeMetric(false);
                return respondJson(cb, json{{"ok", false},
                                            {"connected", false},
                                            {"error", "disconnected"}});
              }
              if (longQuerySafeMode) {
                verify = json{{"ok", false},
                              {"skipped", true},
                              {"reason", "long-query-safe-mode"}};
              } else if (!usingNativeTransformer) {
                verify = json{{"ok", false},
                              {"skipped", true},
                              {"reason", std::string("using-") + backendMode}};
              } else {
                std::lock_guard<std::mutex> lock(transformerMu_);
                verify = transformer_.verify(
                    text, graphContext, jsonStringOrEmpty(tReply, "reply"));
              }
              if (db_) {
                db_->setInferenceCache(
                    cacheKey, json{{"tReply", tReply}, {"verify", verify}},
                    86400);
              }
            }
            std::string transformerReplyRaw =
                jsonStringOrEmpty(tReply, "reply");
            std::string finalReply =
                stripAddonToolDirective(transformerReplyRaw);
            addon::AddonResult addonExec;
            AddonToolCall addonCall;
            bool addonUsed = false;
            if (addonContractEnabled) {
              addonCall = parseAddonToolCall(transformerReplyRaw, text);
              addonUsed = invokeAddonTool(addonCall, body, addonExec);
            }
            if (addonUsed) {
              std::string followupGraphContext =
                  buildAddonResultContext(graphContext, addonCall, addonExec);
              if (usingLlamaCpp || usingBitNet)
                dispatchExternalStyleTrainStep(backendMode, body, text,
                                               gnnHint.align);
              json tReplyAfterTool;
              if (usingOllama) {
                tReplyAfterTool =
                    chatWithOllama(text, followupGraphContext, maxTokens,
                   inferenceOptions);
              } else if (usingLlamaCpp) {
                tReplyAfterTool =
                    chatWithLlamaCpp(text, followupGraphContext, maxTokens,
                     inferenceOptions);
              } else if (usingBitNet) {
                tReplyAfterTool =
                    chatWithBitNet(text, followupGraphContext, maxTokens,
                   inferenceOptions);
              } else {
                std::lock_guard<std::mutex> lock(transformerMu_);
                tReplyAfterTool =
                    transformer_.chat(text, followupGraphContext, maxTokens);
                postInferenceMemoryTrim();
              }
              std::string rewritten = stripAddonToolDirective(
                  jsonStringOrEmpty(tReplyAfterTool, "reply"));
              if (!rewritten.empty()) {
                finalReply = rewritten;
                std::string replyAfterToolRaw =
                    jsonStringOrEmpty(tReplyAfterTool, "reply");
                if (!replyAfterToolRaw.empty()) {
                  transformerReplyRaw = replyAfterToolRaw;
                }
              } else {
                finalReply = addonExec.reply;
              }
              if (usingNativeTransformer) {
                std::string source =
                    "addon:" +
                    addonExec.meta.value("addon", addonCall.type.empty()
                                                      ? std::string("auto")
                                                      : addonCall.type);
                std::lock_guard<std::mutex> lock(transformerMu_);
                transformer_.addPreferenceFeedback(
                    text, graphContext, finalReply,
                    stripAddonToolDirective(transformerReplyRaw), 1.0f, source);
              }
            }
            if (aheadModule_ && !sessionId.empty() && !finalReply.empty()) {
              try {
                auto routed = aheadModule_->routeSentence(finalReply, sessionId);
                if (!routed.blockedByMemeBarrier)
                  finalReply = routed.textForBackend;
              } catch (...) {
              }
            }
            if (!tReply.is_object()) {
              finalizeMetric(false);
              return respondJson(cb, json{{"ok", false},
                                          {"connected", false},
                                          {"error", "disconnected"}});
            }
            if (!verify.is_object()) {
              verify = json{{"ok", false}, {"error", "verify unavailable"}};
            }
            json result = tReply;
            result["transformerReply"] = transformerReplyRaw;
            result["reply"] = finalReply;
            result["graphContext"] = graphContext;
            result["gnnAlign"] = gnnHint.align;
            if (!gnnHint.annotation.empty())
              result["gnnStage2"] = gnnHint.annotation;
            if (!sparkAnn.empty())
              result["sparkGnnStage2"] = sparkAnn;
            if (!hint.text.empty()) {
              result["contextHint"] = hint.text;
              result["contextWeight"] = hint.weight;
              if (!hint.mode.empty())
                result["contextMode"] = hint.mode;
            }
            if (!imageContext.empty())
              result["imageContext"] = imageContext;
            if (imageEmbCount > 0)
              result["imageEmbeddingCount"] = imageEmbCount;
            if (addonUsed) {
              result["addon"] = addonExec.meta;
              result["toolCall"] =
                  json{{"type", addonCall.type}, {"input", addonCall.input}};
            }
            if (!inferenceOptions.is_null())
              result["cognitionInferenceOptions"] = inferenceOptions;
            if (!cognitionEmotionTensor.is_null())
              result["cognitionEmotionTensor"] = cognitionEmotionTensor;
            if (!cognitionWorldEvidence.is_null() && !cognitionWorldEvidence.empty())
              result["cognitionWorldEvidence"] = cognitionWorldEvidence;
            if (!cognitionMixedModalOutputs.empty())
              result["cognitionMixedModalOutputs"] = cognitionMixedModalOutputs;
            result["verify"] = verify;
            result["cached"] = cacheHit;
            if (!graphResult.is_null())
              result["graph"] = graphResult;
            result["longQuerySafeMode"] = longQuerySafeMode;
            result["gnnModuleDisabled"] = gnnModuleDisabled;
            result["contextModuleDisabled"] = contextModuleDisabled;
            result["usingOllama"] = usingOllama;
            result["transformerMode"] = backendMode;
            if (usingOllama)
              result["provider"] =
                  json{{"id", "ollama"}, {"model", config_.ollamaModel}};
            else if (usingLlamaCpp)
              result["provider"] =
                  json{{"id", "llamacpp"}, {"model", config_.llamaCppModel}};
            else if (usingBitNet)
              result["provider"] =
                  json{{"id", "bitnet"}, {"model", config_.bitnetModel}};
            else
              result["provider"] =
                  json{{"id", "core"}, {"model", "native-transformer"}};
            double verifyScore = verify.value("score", 0.0);
            (void)verifyScore;
            logRouteCompute("/api/transformer/chat",
                            std::string("ok cached=") +
                                (cacheHit ? "true" : "false") +
                                " addon=" + (addonUsed ? "true" : "false"));
            respondJson(cb, json{{"ok", true}, {"result", result}});
            finalizeMetric(true);
          } catch (const std::exception &e) {
            logRouteError("/api/transformer/chat", e.what());
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
            finalizeMetric(false);
          } catch (...) {
            logRouteError("/api/transformer/chat", "internal error");
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
            finalizeMetric(false);
          }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/transformer/chat",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          logRouteError("/api/transformer/chat",
                        std::string("method-not-allowed:") +
                            req->methodString());
          respondJson(cb,
                      json{{"ok", false},
                           {"error", "method not allowed"},
                           {"allowed", "POST"}},
                      drogon::k405MethodNotAllowed);
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/api/transformer/pretrain",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          try {
            if (!authOK(req))
              return unauthorized(cb);
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok)
              return badRequest(cb, "invalid json");
            int64_t epochs64 = 1;
            std::string parseErr;
            if (!parseBoundedInt64(payload, "epochs", 1, 1000, epochs64,
                                   parseErr))
              return badRequest(cb, parseErr);
            int epochs = (int)epochs64;
            float lr = payload.value("lr", 0.001f);
            if (!(lr > 0.0f) || lr > 1.0f)
              return badRequest(cb, "lr out of range");
            auto samples = parseTrainSamples(payload);
            if (samples.empty())
              return badRequest(cb, "samples required");
            if (samples.size() > 50000)
              return badRequest(
                  cb, "samples too large, please batch train (<=50000)");
            uint64_t charCount = 0;
            for (const auto &s : samples)
              charCount += (uint64_t)s.input.size() +
                           (uint64_t)s.target.size() + (uint64_t)s.graph.size();
            uint64_t monitorId = beginTrainMonitor(
                "/api/transformer/pretrain", json{{"epochs", epochs},
                                                  {"lr", lr},
                                                  {"samples", samples.size()}});
            touchTrainMonitor("pretrain-running", 0, 0,
                              (uint64_t)samples.size(), charCount);
            json out;
            {
              std::lock_guard<std::mutex> lock(transformerMu_);
              out = transformer_.pretrain(samples, epochs, lr);
            }
            finishTrainMonitor(out.value("ok", false), out);
            if (logger_.enabled() &&
                logger_.mode() == LoggerCXX::Mode::Release &&
                !releaseSnapshotLogged_) {
              logger_.log(LoggerCXX::Type::MEMORY, buildMemorySnapshotLog());
              releaseSnapshotLogged_ = true;
            }
            respondJson(
                cb,
                json{{"ok", true}, {"result", out}, {"monitorId", monitorId}});
          } catch (const std::bad_alloc &e) {
            finishTrainMonitor(false, json::object(),
                               std::string("bad_alloc: ") + e.what());
            logRouteError("/api/transformer/pretrain",
                          std::string("bad_alloc: ") + e.what());
            respondJson(cb,
                        json{{"ok", false},
                             {"error", "insufficient memory during pretrain"}},
                        drogon::k500InternalServerError);
          } catch (const std::exception &e) {
            finishTrainMonitor(false, json::object(), e.what());
            logRouteError("/api/transformer/pretrain", e.what());
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            finishTrainMonitor(false, json::object(), "internal error");
            logRouteError("/api/transformer/pretrain", "internal error");
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/external_style/train_step",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          try {
            if (!authOK(req))
              return unauthorized(cb);
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok)
              return badRequest(cb, "invalid json");

            std::string mode = trimCopy(payload.value(
                "provider",
                payload.value("backend", config_.transformerBackend)));
            std::transform(mode.begin(), mode.end(), mode.begin(),
                           [](unsigned char c) {
                             return static_cast<char>(std::tolower(c));
                           });
            bool usingLlamaCpp = (mode == "llamacpp" || mode == "llama.cpp" ||
                                  mode == "llama_cpp");
            bool usingBitNet = (mode == "bitnet");
            if (!usingLlamaCpp && !usingBitNet)
              return badRequest(cb, "provider must be llamacpp or bitnet");

            std::string text =
                trimCopy(payload.value("text", payload.value("message", "")));
            if (text.empty())
              return badRequest(cb, "text required");

            std::vector<std::string> keywords =
                parseStyleKeywords(payload, text);
            json trainPayload = buildExternalStyleTrainPayload(
                usingBitNet, payload, text,
                payload.value("observedAlign", payload.value("gnnAlign", 0.0)),
                keywords);

            json result =
                usingBitNet
                    ? postExternalStyleTrainStep(
                          "bitnet", config_.bitnetBaseUrl, config_.bitnetModel,
                          config_.bitnetRequestTimeoutMs, trainPayload)
                    : postExternalStyleTrainStep(
                          "llamacpp", config_.llamaCppBaseUrl,
                          config_.llamaCppModel,
                          config_.llamaCppRequestTimeoutMs, trainPayload);

            respondJson(cb,
                        json{{"ok", result.value("ok", false)},
                             {"provider", usingBitNet ? "bitnet" : "llamacpp"},
                             {"request", trainPayload},
                             {"result", result}});
          } catch (const std::exception &e) {
            logRouteError("/api/external_style/train_step", e.what());
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            logRouteError("/api/external_style/train_step", "internal error");
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/fine_tuning/corpus/add",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          try {
            if (!authOK(req))
              return unauthorized(cb);
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok)
              return badRequest(cb, "invalid json");

            fs::path defaultCorpus =
                config_.exportDir / "fine_tuning" / "append_corpus.jsonl";
            fs::path corpusPath = defaultCorpus;
            if (payload.contains("appendCorpusPath") &&
                payload["appendCorpusPath"].is_string()) {
              auto raw =
                  trimCopy(payload["appendCorpusPath"].get<std::string>());
              if (!raw.empty()) {
                fs::path p(raw);
                corpusPath = p.is_relative() ? fs::absolute(config_.baseDir / p)
                                             : fs::absolute(p);
              }
            }
            std::error_code ec;
            fs::create_directories(corpusPath.parent_path(), ec);

            json items = json::array();
            if (payload.contains("items") && payload["items"].is_array()) {
              items = payload["items"];
            } else {
              std::string q = payload.value("question", "");
              std::string a = payload.value("answer", "");
              std::string s = payload.value("style", "default");
              if (!trimCopy(q).empty() || !trimCopy(a).empty()) {
                items.push_back(
                    json{{"question", q}, {"answer", a}, {"style", s}});
              }
            }

            if (!items.is_array() || items.empty()) {
              return badRequest(cb, "items required");
            }

            uint64_t added = 0;
            uint64_t charCount = 0;
            std::ofstream out(corpusPath,
                              std::ios::out | std::ios::app | std::ios::binary);
            if (!out.is_open()) {
              return respondJson(
                  cb, json{{"ok", false}, {"error", "cannot open corpus file"}},
                  drogon::k500InternalServerError);
            }
            for (const auto &it : items) {
              if (!it.is_object())
                continue;
              std::string q = trimCopy(it.value("question", ""));
              std::string a = trimCopy(it.value("answer", ""));
              std::string s = trimCopy(it.value("style", "default"));
              if (q.empty() || a.empty())
                continue;
              auto gnnKeywords = deriveGnnStyleKeywords(q + " " + a, 8);
              std::string questionAug = q;
              if (!gnnKeywords.empty()) {
                std::ostringstream kw;
                kw << "[GNN关键词:";
                for (size_t i = 0; i < gnnKeywords.size(); i++) {
                  if (i)
                    kw << ",";
                  kw << gnnKeywords[i];
                }
                kw << "] ";
                questionAug = kw.str() + q;
              }
              json row{{"question", questionAug},
                       {"questionRaw", q},
                       {"answer", a},
                       {"style", s.empty() ? "default" : s},
                       {"gnnKeywords", gnnKeywords}};
              std::string line = row.dump();
              out << line << "\n";
              added += 1;
              charCount += (uint64_t)q.size() + (uint64_t)a.size();
            }
            out.flush();

            if (added == 0) {
              return badRequest(cb, "no valid qa pairs");
            }

            uint64_t totalLines = 0;
            {
              std::ifstream in(corpusPath, std::ios::in | std::ios::binary);
              std::string line;
              while (std::getline(in, line)) {
                if (!trimCopy(line).empty())
                  totalLines += 1;
              }
            }

            respondJson(cb, json{{"ok", true},
                                 {"result", json{{"path", corpusPath.string()},
                                                 {"added", added},
                                                 {"total", totalLines},
                                                 {"chars", charCount}}}});
          } catch (const std::exception &e) {
            logRouteError("/api/fine_tuning/corpus/add", e.what());
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            logRouteError("/api/fine_tuning/corpus/add", "internal error");
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/fine_tuning/run",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          try {
            if (!authOK(req))
              return unauthorized(cb);
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok)
              return badRequest(cb, "invalid json");

            {
              std::lock_guard<std::mutex> lock(fineTuningMu_);
              if (fineTuningRunning_) {
                return respondJson(
                    cb,
                    json{{"ok", false},
                         {"error", "fine-tuning already running"}},
                    drogon::k409Conflict);
              }
              fineTuningRunning_ = true;
            }

            auto guardRunning = std::shared_ptr<void>(nullptr, [this](void *) {
              std::lock_guard<std::mutex> lock(this->fineTuningMu_);
              this->fineTuningRunning_ = false;
            });

            fs::path defaultCorpus =
                config_.exportDir / "fine_tuning" / "append_corpus.jsonl";
            fs::path defaultOutDir =
                config_.exportDir / "fine_tuning" / "checkpoints";

            fs::path corpusPath = defaultCorpus;
            fs::path outputDir = defaultOutDir;
            std::string ollamaModel =
                payload.value("ollamaModel", config_.ollamaModel.empty()
                                                 ? std::string("gpt-oss:20b")
                                                 : config_.ollamaModel);
            std::string hfModel =
                payload.value("hfModel", std::string("gpt-oss:20b"));

            if (payload.contains("appendCorpusPath") &&
                payload["appendCorpusPath"].is_string()) {
              auto raw =
                  trimCopy(payload["appendCorpusPath"].get<std::string>());
              if (!raw.empty()) {
                fs::path p(raw);
                corpusPath = p.is_relative() ? fs::absolute(config_.baseDir / p)
                                             : fs::absolute(p);
              }
            }
            if (payload.contains("outputDir") &&
                payload["outputDir"].is_string()) {
              auto raw = trimCopy(payload["outputDir"].get<std::string>());
              if (!raw.empty()) {
                fs::path p(raw);
                outputDir = p.is_relative() ? fs::absolute(config_.baseDir / p)
                                            : fs::absolute(p);
              }
            }

            std::error_code ec;
            fs::create_directories(corpusPath.parent_path(), ec);
            fs::create_directories(outputDir, ec);

            uint64_t appendedNow = 0;
            {
              json incomingItems = json::array();
              if (payload.contains("items") && payload["items"].is_array()) {
                incomingItems = payload["items"];
              }
              if (payload.contains("question") || payload.contains("answer")) {
                incomingItems.push_back(
                    json{{"question", payload.value("question", "")},
                         {"answer", payload.value("answer", "")},
                         {"style", payload.value("style", "default")}});
              }

              if (incomingItems.is_array() && !incomingItems.empty()) {
                std::ofstream out(corpusPath, std::ios::out | std::ios::app |
                                                  std::ios::binary);
                if (!out.is_open()) {
                  return respondJson(
                      cb,
                      json{{"ok", false}, {"error", "cannot open corpus file"}},
                      drogon::k500InternalServerError);
                }
                for (const auto &it : incomingItems) {
                  if (!it.is_object())
                    continue;
                  std::string q = trimCopy(it.value("question", ""));
                  std::string a = trimCopy(it.value("answer", ""));
                  std::string s = trimCopy(it.value("style", "default"));
                  if (q.empty() || a.empty())
                    continue;
                  auto gnnKeywords = deriveGnnStyleKeywords(q + " " + a, 8);
                  std::string questionAug = q;
                  if (!gnnKeywords.empty()) {
                    std::ostringstream kw;
                    kw << "[GNN关键词:";
                    for (size_t i = 0; i < gnnKeywords.size(); i++) {
                      if (i)
                        kw << ",";
                      kw << gnnKeywords[i];
                    }
                    kw << "] ";
                    questionAug = kw.str() + q;
                  }
                  out << json{{"question", questionAug},
                              {"questionRaw", q},
                              {"answer", a},
                              {"style", s.empty() ? "default" : s},
                              {"gnnKeywords", gnnKeywords}}
                             .dump()
                      << "\n";
                  appendedNow += 1;
                }
                out.flush();
              }
            }

            if (!fs::exists(corpusPath)) {
              return badRequest(cb, std::string("append corpus not found: ") +
                                        corpusPath.string());
            }

            int64_t selfPlayPairs64 = 0;
            int64_t epochs64 = 1;
            std::string parseErr;
            if (!parseBoundedInt64(payload, "selfPlayPairs", 0, 100000,
                                   selfPlayPairs64, parseErr)) {
              return badRequest(cb, parseErr);
            }
            if (!parseBoundedInt64(payload, "epochs", 1, 1000, epochs64,
                                   parseErr)) {
              return badRequest(cb, parseErr);
            }
            double lr = payload.value("lr", 2e-5);
            if (!(lr > 0.0) || lr > 1.0) {
              return badRequest(cb, "lr out of range");
            }

            uint64_t monitorId = beginTrainMonitor(
                "/api/fine_tuning/run",
                json{{"appendCorpusPath", corpusPath.string()},
                     {"outputDir", outputDir.string()},
                     {"appendedNow", appendedNow},
                     {"ollamaModel", ollamaModel},
                     {"hfModel", hfModel},
                     {"selfPlayPairs", selfPlayPairs64},
                     {"epochs", epochs64},
                     {"lr", lr}});
            touchTrainMonitor("fine-tuning-init", 0, 0, 0, 0);

            char errBuf[4096] = {0};
            if (!transformer_ollama_ft_bridge_set_module(
                    "transformer_ollama_fine_tuning", errBuf, sizeof(errBuf))) {
              std::string err = errBuf[0]
                                    ? std::string(errBuf)
                                    : std::string("bridge_set_module failed");
              finishTrainMonitor(false, json::object(), err);
              return respondJson(cb, json{{"ok", false}, {"error", err}},
                                 drogon::k500InternalServerError);
            }
            if (!transformer_ollama_ft_bridge_init(errBuf, sizeof(errBuf))) {
              std::string err = errBuf[0] ? std::string(errBuf)
                                          : std::string("bridge_init failed");
              finishTrainMonitor(false, json::object(), err);
              return respondJson(cb, json{{"ok", false}, {"error", err}},
                                 drogon::k500InternalServerError);
            }

            touchTrainMonitor("fine-tuning-running", 0, 0,
                              (uint64_t)selfPlayPairs64, 0);
            int rc = transformer_ollama_ft_bridge_train(
                corpusPath.string().c_str(), outputDir.string().c_str(),
                ollamaModel.c_str(), hfModel.c_str(), (int)selfPlayPairs64,
                (int)epochs64, lr, errBuf, sizeof(errBuf));

            if (!rc) {
              std::string err = errBuf[0] ? std::string(errBuf)
                                          : std::string("fine-tuning failed");
              finishTrainMonitor(false, json::object(), err);
              return respondJson(
                  cb,
                  json{{"ok", false}, {"error", err}, {"monitorId", monitorId}},
                  drogon::k500InternalServerError);
            }

            json result{{"ok", true},
                        {"appendCorpusPath", corpusPath.string()},
                        {"outputDir", outputDir.string()},
                        {"appendedNow", appendedNow},
                        {"ollamaModel", ollamaModel},
                        {"hfModel", hfModel},
                        {"epochs", epochs64},
                        {"selfPlayPairs", selfPlayPairs64},
                        {"lr", lr}};
            finishTrainMonitor(true, result);
            respondJson(cb, json{{"ok", true},
                                 {"result", result},
                                 {"monitorId", monitorId}});
          } catch (const std::exception &e) {
            finishTrainMonitor(false, json::object(), e.what());
            logRouteError("/api/fine_tuning/run", e.what());
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            finishTrainMonitor(false, json::object(), "internal error");
            logRouteError("/api/fine_tuning/run", "internal error");
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/transformer/joint_train",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          try {
            if (!authOK(req))
              return unauthorized(cb);
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok)
              return badRequest(cb, "invalid json");
            int64_t epochs64 = 1;
            std::string parseErr;
            if (!parseBoundedInt64(payload, "epochs", 1, 1000, epochs64,
                                   parseErr))
              return badRequest(cb, parseErr);
            int epochs = (int)epochs64;
            float lr = payload.value("lr", 0.001f);
            float gw = payload.value("graphWeight", 0.35f);
            if (!(lr > 0.0f) || lr > 1.0f)
              return badRequest(cb, "lr out of range");
            if (gw < 0.0f || gw > 1.0f)
              return badRequest(cb, "graphWeight out of range");
            auto samples = parseTrainSamples(payload);
            if (samples.empty())
              return badRequest(cb, "samples required");
            if (samples.size() > 50000)
              return badRequest(
                  cb, "samples too large, please batch train (<=50000)");
            uint64_t charCount = 0;
            for (const auto &s : samples)
              charCount += (uint64_t)s.input.size() +
                           (uint64_t)s.target.size() + (uint64_t)s.graph.size();
            uint64_t monitorId =
                beginTrainMonitor("/api/transformer/joint_train",
                                  json{{"epochs", epochs},
                                       {"lr", lr},
                                       {"graphWeight", gw},
                                       {"samples", samples.size()}});
            touchTrainMonitor("joint-train-running", 0, 0,
                              (uint64_t)samples.size(), charCount);
            json out;
            {
              std::lock_guard<std::mutex> lock(transformerMu_);
              out = transformer_.jointTrain(samples, epochs, lr, gw);
            }
            finishTrainMonitor(out.value("ok", false), out);
            if (logger_.enabled() &&
                logger_.mode() == LoggerCXX::Mode::Release &&
                !releaseSnapshotLogged_) {
              logger_.log(LoggerCXX::Type::MEMORY, buildMemorySnapshotLog());
              releaseSnapshotLogged_ = true;
            }
            respondJson(
                cb,
                json{{"ok", true}, {"result", out}, {"monitorId", monitorId}});
          } catch (const std::bad_alloc &e) {
            finishTrainMonitor(false, json::object(),
                               std::string("bad_alloc: ") + e.what());
            logRouteError("/api/transformer/joint_train",
                          std::string("bad_alloc: ") + e.what());
            respondJson(
                cb,
                json{{"ok", false},
                     {"error", "insufficient memory during joint_train"}},
                drogon::k500InternalServerError);
          } catch (const std::exception &e) {
            finishTrainMonitor(false, json::object(), e.what());
            logRouteError("/api/transformer/joint_train", e.what());
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            finishTrainMonitor(false, json::object(), "internal error");
            logRouteError("/api/transformer/joint_train", "internal error");
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/transformer/ga_optimize",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          try {
            if (!authOK(req))
              return unauthorized(cb);
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok)
              return badRequest(cb, "invalid json");
            int64_t generations64 = 3;
            int64_t population64 = 6;
            std::string parseErr;
            if (!parseBoundedInt64(payload, "generations", 1, 1000,
                                   generations64, parseErr))
              return badRequest(cb, parseErr);
            if (!parseBoundedInt64(payload, "population", 2, 2048, population64,
                                   parseErr))
              return badRequest(cb, parseErr);
            int generations = (int)generations64;
            int population = (int)population64;
            auto samples = parseTrainSamples(payload);
            if (samples.empty())
              return badRequest(cb, "samples required");
            uint64_t monitorId =
                beginTrainMonitor("/api/transformer/ga_optimize",
                                  json{{"generations", generations},
                                       {"population", population},
                                       {"samples", samples.size()}});
            touchTrainMonitor("ga-optimize-running", 0, 0,
                              (uint64_t)samples.size(), 0);
            json out;
            {
              std::lock_guard<std::mutex> lock(transformerMu_);
              out = transformer_.optimizeGA(samples, generations, population);
            }
            finishTrainMonitor(out.value("ok", false), out);
            if (logger_.enabled() &&
                logger_.mode() == LoggerCXX::Mode::Release &&
                !releaseSnapshotLogged_) {
              logger_.log(LoggerCXX::Type::MEMORY, buildMemorySnapshotLog());
              releaseSnapshotLogged_ = true;
            }
            respondJson(
                cb,
                json{{"ok", true}, {"result", out}, {"monitorId", monitorId}});
          } catch (const std::bad_alloc &e) {
            finishTrainMonitor(false, json::object(),
                               std::string("bad_alloc: ") + e.what());
            logRouteError("/api/transformer/ga_optimize",
                          std::string("bad_alloc: ") + e.what());
            respondJson(
                cb,
                json{{"ok", false},
                     {"error", "insufficient memory during ga_optimize"}},
                drogon::k500InternalServerError);
          } catch (const std::exception &e) {
            finishTrainMonitor(false, json::object(), e.what());
            logRouteError("/api/transformer/ga_optimize", e.what());
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            finishTrainMonitor(false, json::object(), "internal error");
            logRouteError("/api/transformer/ga_optimize", "internal error");
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/transformer/verify",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          bool ok = true;
          json payload = parseRequestBody(req, ok);
          if (!ok)
            return badRequest(cb, "invalid json");
          std::string text = payload.value("text", "");
          std::string graphContext = payload.value("graphContext", "");
          std::string reply = payload.value("reply", "");
          json out;
          {
            std::lock_guard<std::mutex> lock(transformerMu_);
            out = transformer_.verify(text, graphContext, reply);
          }
          respondJson(cb, json{{"ok", true}, {"result", out}});
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/transformer/feedback",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          bool ok = true;
          json payload = parseRequestBody(req, ok);
          if (!ok)
            return badRequest(cb, "invalid json");
          std::string text = payload.value("text", payload.value("prompt", ""));
          std::string graphContext = payload.value("graphContext", "");
          std::string reply =
              payload.value("reply", payload.value("output", ""));
          std::string chosen = payload.value("chosen", reply);
          std::string rejected = payload.value("rejected", "");
          std::string source = payload.value("source", "api");
          float score = payload.value("score", 0.0f);
          if (payload.contains("accepted")) {
            score = payload.value("accepted", false) ? 1.0f : -1.0f;
          }
          if (payload.contains("addon") && payload["addon"].is_object()) {
            float addonScale = 0.35f;
            {
              std::lock_guard<std::mutex> lock(transformerMu_);
              addonScale =
                  transformer_.params().value("addonRewardScale", 0.35f);
            }
            bool addonOk = true;
            const auto &a = payload["addon"];
            if (a.contains("ok") && a["ok"].is_boolean())
              addonOk = a["ok"].get<bool>();
            if (source == "api") {
              std::string n = a.value("name", a.value("addon", ""));
              std::string t = a.value("type", "");
              if (!n.empty())
                source = "addon:" + n;
              else if (!t.empty())
                source = "addonType:" + t;
            }
            score += addonOk ? addonScale : -addonScale;
          }
          bool trainNow = payload.value("trainNow", false);
          int steps = payload.value("steps", 1);
          if (chosen.empty())
            return badRequest(cb, "chosen/reply required");
          json stats = json::object();
          json train = json::object();
          {
            std::lock_guard<std::mutex> lock(transformerMu_);
            transformer_.addFeedback(chosen, score);
            transformer_.addPreferenceFeedback(text, graphContext, chosen,
                                               rejected, score, source);
            if (trainNow) {
              train = transformer_.trainFromFeedback(std::max(1, steps));
            }
            stats = transformer_.rlhfStats();
          }
          json out{{"ok", true}, {"result", stats}};
          if (trainNow)
            out["train"] = train;
          respondJson(cb, out);
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/transformer/feedback/train",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          bool ok = true;
          json payload = parseRequestBody(req, ok);
          if (!ok)
            return badRequest(cb, "invalid json");
          int64_t steps64 = 1;
          std::string parseErr;
          if (!parseBoundedInt64(payload, "steps", 1, 1000000, steps64,
                                 parseErr))
            return badRequest(cb, parseErr);
          int steps = (int)steps64;
          uint64_t monitorId = beginTrainMonitor(
              "/api/transformer/feedback/train", json{{"steps", steps}});
          touchTrainMonitor("feedback-train-running", 0, 0, 0, 0);
          json out;
          {
            std::lock_guard<std::mutex> lock(transformerMu_);
            out = transformer_.trainFromFeedback(steps);
          }
          finishTrainMonitor(out.value("ok", false), out);
          respondJson(
              cb,
              json{{"ok", true}, {"result", out}, {"monitorId", monitorId}});
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/transformer/feedback/status",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          json out;
          {
            std::lock_guard<std::mutex> lock(transformerMu_);
            out = transformer_.rlhfStats();
          }
          respondJson(cb, json{{"ok", true}, {"result", out}});
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/api/transformer/params",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          if (req->method() == drogon::Get) {
            json out;
            {
              std::lock_guard<std::mutex> lock(transformerMu_);
              out = transformer_.params();
            }
            respondJson(cb, json{{"ok", true}, {"params", out}});
            return;
          }
          bool ok = true;
          json payload = parseRequestBody(req, ok);
          if (!ok)
            return badRequest(cb, "invalid json");
          {
            std::lock_guard<std::mutex> lock(transformerMu_);
            transformer_.applyParams(payload);
          }
          json out;
          {
            std::lock_guard<std::mutex> lock(transformerMu_);
            out = transformer_.params();
          }
          respondJson(cb, json{{"ok", true}, {"params", out}});
        },
        {drogon::Get, drogon::Post});

    drogon::app().registerHandler(
        "/api/transformer/checkpoint/save",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          try {
            if (!authOK(req))
              return unauthorized(cb);
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok)
              payload = json::object();
            std::string path = payload.value(
                "path",
                std::string("runtime_store/transformer_checkpoint.json"));
            json out;
            {
              std::lock_guard<std::mutex> lock(transformerMu_);
              out = transformer_.saveCheckpoint(path);
            }
            auto status = out.value("ok", false)
                              ? drogon::k200OK
                              : drogon::k500InternalServerError;
            respondJson(cb, out, status);
          } catch (const std::exception &e) {
            logRouteError("/api/transformer/checkpoint/save", e.what());
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            logRouteError("/api/transformer/checkpoint/save", "internal error");
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/transformer/checkpoint/load",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          try {
            if (!authOK(req))
              return unauthorized(cb);
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok)
              payload = json::object();
            std::string path = payload.value(
                "path",
                std::string("runtime_store/transformer_checkpoint.json"));
            json out;
            {
              std::lock_guard<std::mutex> lock(transformerMu_);
              out = transformer_.loadCheckpoint(path);
            }
            auto status = out.value("ok", false)
                              ? drogon::k200OK
                              : drogon::k500InternalServerError;
            respondJson(cb, out, status);
          } catch (const std::exception &e) {
            logRouteError("/api/transformer/checkpoint/load", e.what());
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            logRouteError("/api/transformer/checkpoint/load", "internal error");
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/model/lifecycle",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          respondJson(cb, modelLifecycle_.status());
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/api/model/compress",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          bool ok = true;
          json payload = parseRequestBody(req, ok);
          if (!ok)
            return badRequest(cb, "invalid json");
          json out = modelLifecycle_.compressPlan(payload);
          if (payload.value("applyTransformerParams", false)) {
            json patch = json::object();
            std::string quant = payload.value("quant", std::string(""));
            double pruneRatio = payload.value("pruneRatio", 0.0);
            if (quant == "int8")
              patch["dModel"] = 96;
            if (quant == "int4")
              patch["dModel"] = 64;
            if (pruneRatio >= 0.2)
              patch["nLayers"] = 2;
            if (!patch.empty()) {
              std::lock_guard<std::mutex> lock(transformerMu_);
              transformer_.applyParams(patch);
              out["transformerPatch"] = patch;
            }
          }
          respondJson(cb, out);
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/model/explain",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          bool ok = true;
          json payload = parseRequestBody(req, ok);
          if (!ok)
            return badRequest(cb, "invalid json");
          std::string text = payload.value("text", "");
          std::string graphContext = payload.value("graphContext", "");
          std::string reply = payload.value("reply", "");
          if (!payload.contains("reply") && reply.empty() && !text.empty()) {
            int maxTokens = payload.value("maxTokens", 120);
            json chatResult;
            if (config_.transformerBackend == "native") {
              std::lock_guard<std::mutex> lock(transformerMu_);
              chatResult = transformer_.chat(text, graphContext, maxTokens);
              postInferenceMemoryTrim();
            } else if (config_.tinyllamaEnabled &&
                       !config_.tinyllamaBaseUrl.empty()) {
              chatResult = chatWithTinyllama(text, graphContext, maxTokens);
            } else if (config_.inferenceEnabled &&
                       !config_.llamaCppBaseUrl.empty()) {
              chatResult = chatWithLlamaCpp(text, graphContext, maxTokens);
            }
            if (chatResult.is_object()) {
              reply = chatResult.value("reply", "");
              payload["reply"] = reply;
              payload["generatedByTransformer"] =
                  (config_.transformerBackend == "native");
            }
          }
          json out = modelLifecycle_.explainOutput(payload);
          if (!reply.empty() && config_.transformerBackend == "native") {
            json verify;
            {
              std::lock_guard<std::mutex> lock(transformerMu_);
              verify = transformer_.verify(text, graphContext, reply);
            }
            out["result"]["verify"] = verify;
          }
          respondJson(cb, out);
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/model/deploy",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          bool ok = true;
          json payload = parseRequestBody(req, ok);
          if (!ok)
            return badRequest(cb, "invalid json");
          respondJson(cb, modelLifecycle_.deployTarget(payload));
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/model/update",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          bool ok = true;
          json payload = parseRequestBody(req, ok);
          if (!ok)
            return badRequest(cb, "invalid json");
          respondJson(cb, modelLifecycle_.applyOnlineUpdate(payload));
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/cluster/status",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          respondJson(cb, clusterManager_.status());
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/api/cluster/nodes",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          bool ok = true;
          json payload = parseRequestBody(req, ok);
          if (!ok)
            return badRequest(cb, "invalid json");
          respondJson(cb, clusterManager_.updateNodes(payload));
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/cluster/route",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          bool ok = true;
          json payload = parseRequestBody(req, ok);
          if (!ok)
            return badRequest(cb, "invalid json");
          respondJson(cb, clusterManager_.pickNode(payload));
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/cluster/feedback",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          bool ok = true;
          json payload = parseRequestBody(req, ok);
          if (!ok)
            return badRequest(cb, "invalid json");
          respondJson(cb, clusterManager_.feedback(payload));
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/dataset/catalog",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          respondJson(cb, datasetCatalog_.list());
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/api/data/governance",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          respondJson(cb, datasetCatalog_.governance());
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/api/dataset/register",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          bool ok = true;
          json payload = parseRequestBody(req, ok);
          if (!ok)
            return badRequest(cb, "invalid json");
          respondJson(cb, datasetCatalog_.registerDataset(payload));
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/dataset/activate",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          bool ok = true;
          json payload = parseRequestBody(req, ok);
          if (!ok)
            return badRequest(cb, "invalid json");
          respondJson(cb, datasetCatalog_.activate(payload));
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/data/collect",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          auto routeStarted = std::chrono::steady_clock::now();
          auto finalizeMetric = [&](bool okFlag) {
            auto elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - routeStarted)
                    .count();
            recordRouteMetric("/api/data/collect", elapsed, okFlag);
          };
          if (!authOK(req)) {
            finalizeMetric(false);
            return unauthorized(cb);
          }
          try {
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok) {
              finalizeMetric(false);
              return badRequest(cb, "invalid json");
            }
            auto out = datasetCatalog_.collectData(payload);
            finalizeMetric(out.value("ok", false));
            respondJson(cb, out);
          } catch (const std::exception &e) {
            logRouteError("/api/data/collect", e.what());
            finalizeMetric(false);
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            logRouteError("/api/data/collect", "internal error");
            finalizeMetric(false);
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/data/cleaning/profile",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          auto routeStarted = std::chrono::steady_clock::now();
          auto finalizeMetric = [&](bool okFlag) {
            auto elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - routeStarted)
                    .count();
            recordRouteMetric("/api/data/cleaning/profile", elapsed, okFlag);
          };
          if (!authOK(req)) {
            finalizeMetric(false);
            return unauthorized(cb);
          }
          try {
            if (req->method() == drogon::Get) {
              auto out = datasetCatalog_.list();
              finalizeMetric(out.value("ok", false));
              respondJson(cb, out);
              return;
            }
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok) {
              finalizeMetric(false);
              return badRequest(cb, "invalid json");
            }
            auto out = datasetCatalog_.updateCleaningProfile(payload);
            finalizeMetric(out.value("ok", false));
            respondJson(cb, out);
          } catch (const std::exception &e) {
            logRouteError("/api/data/cleaning/profile", e.what());
            finalizeMetric(false);
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            logRouteError("/api/data/cleaning/profile", "internal error");
            finalizeMetric(false);
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Get, drogon::Post});

    drogon::app().registerHandler(
        "/api/spider/autonomy/status",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          respondJson(cb, spiderAutonomy_.status());
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/api/spider/autonomy/adapt",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          bool ok = true;
          json payload = parseRequestBody(req, ok);
          if (!ok)
            return badRequest(cb, "invalid json");
          respondJson(cb, spiderAutonomy_.adapt(payload, getMonitoringStats()));
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/optimizer/autonomy/status",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          respondJson(cb, optimizerAutonomy_.status());
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/api/optimizer/autonomy/iterate",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          bool ok = true;
          json payload = parseRequestBody(req, ok);
          if (!ok)
            return badRequest(cb, "invalid json");
          json currentParams;
          {
            std::lock_guard<std::mutex> lock(transformerMu_);
            currentParams = transformer_.params();
          }
          json out = optimizerAutonomy_.iterate(payload, getMonitoringStats(),
                                                currentParams);
          if (payload.value("apply", true) && out.value("ok", false) &&
              out.contains("result") &&
              out["result"].contains("transformerPatch")) {
            json patch = out["result"]["transformerPatch"];
            if (patch.is_object() && !patch.empty()) {
              std::lock_guard<std::mutex> lock(transformerMu_);
              transformer_.applyParams(patch);
              out["result"]["applied"] = patch;
            }
          }
          respondJson(cb, out);
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/perf/profile",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          bool ok = true;
          json payload = parseRequestBody(req, ok);
          if (!ok)
            return badRequest(cb, "invalid json");
          json currentParams;
          {
            std::lock_guard<std::mutex> lock(transformerMu_);
            currentParams = transformer_.params();
          }
          json out =
              optimizerAutonomy_.applyPerfProfile(payload, currentParams);
          if (out.value("ok", false) && out.contains("result") &&
              out["result"].contains("transformerPatch")) {
            json patch = out["result"]["transformerPatch"];
            if (patch.is_object() && !patch.empty()) {
              std::lock_guard<std::mutex> lock(transformerMu_);
              transformer_.applyParams(patch);
              out["result"]["applied"] = patch;
            }
          }
          respondJson(cb, out);
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/gnn/upgrade",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          bool ok = true;
          json payload = parseRequestBody(req, ok);
          if (!ok)
            return badRequest(cb, "invalid json");
          json currentParams;
          {
            std::lock_guard<std::mutex> lock(transformerMu_);
            currentParams = transformer_.params();
          }
          json out =
              optimizerAutonomy_.proposeGnnUpgrade(payload, currentParams);
          if (out.value("ok", false) && out.contains("result") &&
              out["result"].contains("transformerPatch")) {
            json patch = out["result"]["transformerPatch"];
            if (patch.is_object() && !patch.empty() &&
                config_.transformerBackend == "native") {
              std::lock_guard<std::mutex> lock(transformerMu_);
              transformer_.applyParams(patch);
              out["result"]["applied"] = patch;
            }
          }
          respondJson(cb, out);
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/transformer/upgrade",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          bool ok = true;
          json payload = parseRequestBody(req, ok);
          if (!ok)
            return badRequest(cb, "invalid json");
          json currentParams;
          {
            std::lock_guard<std::mutex> lock(transformerMu_);
            currentParams = transformer_.params();
          }
          json out = optimizerAutonomy_.proposeTransformerUpgrade(
              payload, currentParams);
          if (out.value("ok", false) && out.contains("result") &&
              out["result"].contains("transformerPatch")) {
            json patch = out["result"]["transformerPatch"];
            if (patch.is_object() && !patch.empty() &&
                config_.transformerBackend == "native") {
              std::lock_guard<std::mutex> lock(transformerMu_);
              transformer_.applyParams(patch);
              out["result"]["applied"] = patch;
            }
          }
          respondJson(cb, out);
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/transformer/modernize",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          bool ok = true;
          json payload = parseRequestBody(req, ok);
          if (!ok)
            return badRequest(cb, "invalid json");
          json currentParams =
              payload.value("currentParams", json::object());
          if (currentParams.empty()) {
            currentParams["nLayers"] =
                phoenix::cfgOr<int>("transformer.nLayers", 4);
            currentParams["dModel"] =
                phoenix::cfgOr<int>("transformer.dModel", 128);
            currentParams["nHeads"] =
                phoenix::cfgOr<int>("transformer.nHeads", 4);
          }
          json out =
              optimizerAutonomy_.modernizeTransformer(payload, currentParams);
          if (out.value("ok", false) && out.contains("result") &&
              out["result"].contains("transformerPatch")) {
            json patch = out["result"]["transformerPatch"];
            if (patch.is_object() && !patch.empty() &&
                config_.transformerBackend == "native") {
              std::lock_guard<std::mutex> lock(transformerMu_);
              transformer_.applyParams(patch);
              out["result"]["applied"] = patch;
            }
          }
          respondJson(cb, out);
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/runtime/features",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          respondJson(
              cb, json{{"ok", true}, {"features", getRuntimeFeatureState()}});
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/api/provider/capabilities",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          std::string backendMode = trimCopy(config_.transformerBackend);
          std::transform(backendMode.begin(), backendMode.end(),
                         backendMode.begin(), [](unsigned char ch) {
                           return static_cast<char>(std::tolower(ch));
                         });
          if (backendMode.empty())
            backendMode = config_.usingOllama
                              ? (config_.ollamaFineTuning ? "ollama-fine-tuning"
                                                          : "ollama")
                              : "native";
          std::string defaultProvider = "off";
          if (config_.inferenceEnabled) {
            if (backendMode == "ollama" || backendMode == "ollama-fine-tuning")
              defaultProvider = "ollama";
            else if (backendMode == "llamacpp" || backendMode == "llama.cpp" ||
                     backendMode == "llama_cpp")
              defaultProvider = "llamacpp";
            else if (backendMode == "bitnet")
              defaultProvider = "bitnet";
            else
              defaultProvider = "core";
          }
          json out = {
              {"ok", true},
              {"result",
               {{"providers",
                 json::array(
                     {json{{"id", "core"},
                           {"mode", "backend-mainchain"},
                           {"supports", json::array({"chat", "transformer-chat",
                                                     "verify"})}},
                      json{{"id", "ollama"},
                           {"mode", "backend-adapter"},
                           {"supports",
                            json::array({"chat", "transformer-chat"})},
                           {"model", config_.ollamaModel}},
                      json{{"id", "llamacpp"},
                           {"mode", "backend-adapter"},
                           {"supports",
                            json::array({"chat", "transformer-chat"})},
                           {"model", config_.llamaCppModel}},
                      json{{"id", "bitnet"},
                           {"mode", "backend-adapter"},
                           {"supports",
                            json::array({"chat", "transformer-chat"})},
                           {"model", config_.bitnetModel}}})},
                {"defaultProvider", defaultProvider},
                {"compat",
                 json{{"shmProtocolVersion", kSharedMemoryProtocolVersion},
                      {"outsidesLinkMode", currentOutsidesLinkMode()}}}}}};
          respondJson(cb, out);
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/api/runtime/features",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          try {
            bool ok = true;
            json body = parseRequestBody(req, ok);
            if (!ok)
              return badRequest(cb, "invalid json");
            auto result = applyRuntimeFeaturePatch(body);
            respondJson(cb, json{{"ok", true},
                                 {"result", result},
                                 {"features", getRuntimeFeatureState()}});
          } catch (const std::exception &e) {
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Patch});

    drogon::app().registerHandler(
        "/api/addons",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          auto controller = pool_->getActive();
          if (!controller)
            return badRequest(cb, "controller not available");
          auto runtime = controller->runtime();
          auto local = std::dynamic_pointer_cast<RuntimeState>(runtime);
          if (!local)
            return badRequest(cb, "runtime not supported");
          respondJson(cb, json{{"ok", true}, {"addons", local->listAddons()}});
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/api/addons/add",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          bool ok = true;
          json body = parseRequestBody(req, ok);
          if (!ok)
            return badRequest(cb, "invalid json");
          auto controller = pool_->getActive();
          if (!controller)
            return badRequest(cb, "controller not available");
          auto runtime = controller->runtime();
          auto local = std::dynamic_pointer_cast<RuntimeState>(runtime);
          if (!local)
            return badRequest(cb, "runtime not supported");
          std::string path = body.value("path", body.value("libraryPath", ""));
          std::string type = body.value("type", "");
          std::string name = body.value("name", "");
          std::string error;
          bool okAdd = false;
          if (!path.empty()) {
            okAdd = local->addAddon(path, error);
          } else {
            okAdd = local->addAddon(type, name, error);
          }
          if (!okAdd) {
            respondJson(
                cb,
                json{{"ok", false},
                     {"error", error.empty() ? "addon add failed" : error}},
                drogon::k400BadRequest);
            return;
          }
          respondJson(cb, json{{"ok", true}, {"addons", local->listAddons()}});
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/addons/remove",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          bool ok = true;
          json body = parseRequestBody(req, ok);
          if (!ok)
            return badRequest(cb, "invalid json");
          auto controller = pool_->getActive();
          if (!controller)
            return badRequest(cb, "controller not available");
          auto runtime = controller->runtime();
          auto local = std::dynamic_pointer_cast<RuntimeState>(runtime);
          if (!local)
            return badRequest(cb, "runtime not supported");
          std::string name = body.value("name", "");
          std::string error;
          if (!local->removeAddon(name, error)) {
            respondJson(
                cb,
                json{{"ok", false},
                     {"error", error.empty() ? "addon remove failed" : error}},
                drogon::k400BadRequest);
            return;
          }
          respondJson(cb, json{{"ok", true}, {"addons", local->listAddons()}});
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/study/status",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          if (!study_) {
            respondJson(cb, json{{"ok", true},
                                 {"running", false},
                                 {"queue", 0},
                                 {"metrics", nullptr}});
            return;
          }
          auto status = study_->status();
          json metrics =
              status.contains("metrics") ? status["metrics"] : json();
          respondJson(cb, json{{"ok", true},
                               {"running", status.value("running", false)},
                               {"queue", status.value("queue", 0)},
                               {"metrics", metrics}});
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/api/monitoring/stats",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          json out = getMonitoringStats();
          out["ok"] = true;
          respondJson(cb, out);
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/api/monitoring/training",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          respondJson(cb,
                      json{{"ok", true}, {"result", getTrainMonitorStats()}});
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/api/monitoring/reset",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          resetMonitoringStats();
          respondJson(cb, json{{"ok", true}, {"result", getMonitoringStats()}});
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/monitoring/training/reset",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          resetTrainMonitor();
          respondJson(cb,
                      json{{"ok", true}, {"result", getTrainMonitorStats()}});
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/learn/dialog/reset",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          dialogCounters_ = {};
          respondJson(cb, json{{"ok", true},
                               {"dialogCounters",
                                json{{"total", dialogCounters_.total},
                                     {"lastRL", dialogCounters_.lastRL},
                                     {"lastADV", dialogCounters_.lastADV},
                                     {"lastGNN", dialogCounters_.lastGNN}}}});
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/array/chat",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          try {
            bool ok = true;
            json body = parseRequestBody(req, ok);
            if (!ok)
              return badRequest(cb, "invalid json");
            auto payload = std::make_shared<json>(body);
            if (!payload || !payload->contains("text"))
              return badRequest(cb, "text required");
            auto &textVal = (*payload)["text"];
            bool truthy = true;
            if (textVal.is_null())
              truthy = false;
            else if (textVal.is_boolean())
              truthy = textVal.get<bool>();
            else if (textVal.is_number())
              truthy = (textVal.get<double>() != 0.0);
            else if (textVal.is_string())
              truthy = !textVal.get<std::string>().empty();
            if (!truthy)
              return badRequest(cb, "text required");
            json opts = payload->value("options", json::object());
            auto jsNumber = [&](const json &v, double fallback) -> double {
              if (v.is_number())
                return v.get<double>();
              if (v.is_boolean())
                return v.get<bool>() ? 1.0 : 0.0;
              if (v.is_string()) {
                double out = 0.0;
                if (parseNumberLikeJs(v.get<std::string>(), out))
                  return out;
                return fallback;
              }
              return fallback;
            };
            auto jsBool = [&](const json &v, bool fallback) -> bool {
              if (v.is_boolean())
                return v.get<bool>();
              if (v.is_number_integer())
                return v.get<int>() != 0;
              if (v.is_number_float())
                return std::fabs(v.get<double>()) > 1e-12;
              if (v.is_string()) {
                std::string s = trimCopy(v.get<std::string>());
                std::transform(s.begin(), s.end(), s.begin(),
                               [](unsigned char ch) {
                                 return static_cast<char>(std::tolower(ch));
                               });
                if (s == "1" || s == "true" || s == "yes" || s == "on")
                  return true;
                if (s == "0" || s == "false" || s == "no" || s == "off")
                  return false;
              }
              return fallback;
            };
            std::string text = textVal.is_string() ? textVal.get<std::string>()
                                                   : textVal.dump();
            (*payload)["text"] = text;
            int bigRounds = 1;
            if (opts.contains("bigRounds")) {
              double v = jsNumber(opts["bigRounds"], 1.0);
              if (std::isfinite(v)) {
                bigRounds = (v == 0.0) ? 1 : (int)v;
              }
            } else if (opts.contains("bigSparkRounds")) {
              double v = jsNumber(opts["bigSparkRounds"], 1.0);
              if (std::isfinite(v)) {
                bigRounds = (v == 0.0) ? 1 : (int)v;
              }
            }

            nlohmann::json inferenceOptions;
            nlohmann::json cognitionEmotionTensor;
            nlohmann::json cognitionWorldEvidence;
            std::vector<nlohmann::json> cognitionMixedModalOutputs;

            bool includeTransformerNlp = true;
            if (opts.contains("includeTransformerNlp")) {
              includeTransformerNlp =
                  jsBool(opts["includeTransformerNlp"], true);
            }
            if (includeTransformerNlp && config_.inferenceEnabled) {
              std::string backendMode = trimCopy(config_.transformerBackend);
              std::transform(backendMode.begin(), backendMode.end(),
                             backendMode.begin(), [](unsigned char ch) {
                               return static_cast<char>(std::tolower(ch));
                             });
              if (backendMode.empty())
                backendMode = config_.usingOllama ? (config_.ollamaFineTuning
                                                         ? "ollama-fine-tuning"
                                                         : "ollama")
                                                  : "native";

              std::string graphContext = payload->value("graphContext", "");

              bool modeOff = (backendMode == "off" || backendMode == "false" ||
                              backendMode == "disabled");
              if (!modeOff) {
                int maxTokens = 256;
                if (payload->contains("maxTokens")) {
                  double v = jsNumber((*payload)["maxTokens"], 256.0);
                  if (std::isfinite(v) && v > 0)
                    maxTokens = std::max(16, (int)v);
                } else if (payload->contains("max_tokens")) {
                  double v = jsNumber((*payload)["max_tokens"], 256.0);
                  if (std::isfinite(v) && v > 0)
                    maxTokens = std::max(16, (int)v);
                } else if (opts.contains("maxTokens")) {
                  double v = jsNumber(opts["maxTokens"], 256.0);
                  if (std::isfinite(v) && v > 0)
                    maxTokens = std::max(16, (int)v);
                }

                try {
                  auto pack = runCognitionAutonomy(*payload, text, graphContext);
                  graphContext = pack.context;
                  inferenceOptions = pack.inferenceOptions;
                  cognitionEmotionTensor = pack.emotionTensor;
                  cognitionWorldEvidence = pack.worldEvidence;
                  cognitionMixedModalOutputs = pack.mixedModalOutputs;
                } catch (...) {
                  inferenceOptions = json::object();
                }
                json transformerReply = json::object();
                bool usingOllama = (backendMode == "ollama" ||
                                    backendMode == "ollama-fine-tuning");
                bool usingLlamaCpp =
                    (backendMode == "llamacpp" || backendMode == "llama.cpp" ||
                     backendMode == "llama_cpp");
                bool usingBitNet = (backendMode == "bitnet");

                try {
                  if (usingOllama)
                    transformerReply =
                        chatWithOllama(text, graphContext, maxTokens,
                   inferenceOptions);
                  else if (usingLlamaCpp)
                    transformerReply =
                        chatWithLlamaCpp(text, graphContext, maxTokens,
                     inferenceOptions);
                  else if (usingBitNet)
                    transformerReply =
                        chatWithBitNet(text, graphContext, maxTokens,
                   inferenceOptions);
                  else {
                    std::lock_guard<std::mutex> lock(transformerMu_);
                    std::vector<std::vector<float>> graphEmb;
                    transformerReply = transformer_.chat(text, graphContext,
                                                         graphEmb, maxTokens);
                    postInferenceMemoryTrim();
                  }
                } catch (...) {
                  transformerReply = json::object();
                }

                std::string reply = transformerReply.value("reply", "");
                if (!reply.empty()) {
                  auto reqEmb = textToMiniEmbedding(text, 64);
                  auto repEmb = textToMiniEmbedding(reply, 64);
                  json candidate =
                      json{{"source", backendMode},
                           {"reply", reply},
                           {"latency", transformerReply.value("latency", 0.0)},
                           {"affinity", cosineSim(reqEmb, repEmb)}};
                  (*payload)["sparkTransformerCandidates"] =
                      json::array({candidate});
                }
              }
            }

            json result = bigRounds > 1 ? spark_->dispatchBig(*payload, opts)
                                        : spark_->dispatch(*payload, opts);
            if (!inferenceOptions.is_null())
              result["cognitionInferenceOptions"] = inferenceOptions;
            if (!cognitionEmotionTensor.is_null())
              result["cognitionEmotionTensor"] = cognitionEmotionTensor;
            if (!cognitionWorldEvidence.is_null() && !cognitionWorldEvidence.empty())
              result["cognitionWorldEvidence"] = cognitionWorldEvidence;
            if (!cognitionMixedModalOutputs.empty())
              result["cognitionMixedModalOutputs"] = cognitionMixedModalOutputs;
            respondJson(cb, json{{"ok", true}, {"result", result}});
          } catch (const std::exception &e) {
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/learn/reinforce",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          if (rlDisabled_)
            return respondJson(cb,
                               json{{"ok", false}, {"error", "rl-disabled"}},
                               drogon::k503ServiceUnavailable);
          try {
            bool ok = true;
            json body = parseRequestBody(req, ok);
            if (!ok)
              return badRequest(cb, "invalid json");
            auto payload = std::make_shared<json>(body);
            int cycles = 3;
            if (payload && payload->contains("cycles")) {
              auto jsNumber = [&](const json &v, double fallback) -> double {
                if (v.is_number())
                  return v.get<double>();
                if (v.is_boolean())
                  return v.get<bool>() ? 1.0 : 0.0;
                if (v.is_string()) {
                  double out = 0.0;
                  if (parseNumberLikeJs(v.get<std::string>(), out))
                    return out;
                  return fallback;
                }
                return fallback;
              };
              double v = jsNumber((*payload)["cycles"], 3.0);
              if (std::isfinite(v))
                cycles = (v == 0.0) ? 3 : (int)v;
            }
            auto out = ensureRL()->learn(cycles);
            json bridge = bridgeLearnerResultToExternalBackend(
                "reinforcement", config_.transformerBackend, body, out);
            respondJson(cb, json{{"ok", true},
                                 {"result", out},
                                 {"externalBridge", bridge}});
          } catch (const std::exception &e) {
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/learn/reinforce/latest",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          respondJson(
              cb, json{{"ok", true}, {"latest", rl_ ? rl_->latest() : json()}});
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/api/learn/gnn_ga",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          if (gnnGaDisabled_)
            return respondJson(
                cb, json{{"ok", false}, {"error", "gnn-ga-disabled"}},
                drogon::k503ServiceUnavailable);
          try {
            bool ok = true;
            json body = parseRequestBody(req, ok);
            if (!ok)
              return badRequest(cb, "invalid json");
            int generations = body.value("generations", 3);
            int population = body.value("population", 8);
            double residualWeight = body.value("residualWeight", 0.35);
            double mutationRate = body.value("mutationRate", 0.25);
            double mutationScale = body.value("mutationScale", 0.35);
            std::vector<std::string> samples;
            if (body.contains("samples") && body["samples"].is_array()) {
              for (auto &s : body["samples"])
                if (s.is_string())
                  samples.push_back(s.get<std::string>());
            }
            auto out = ensureGnnGa()->evolve(generations, population, samples,
                                             residualWeight, mutationRate,
                                             mutationScale);
            json bridge = bridgeLearnerResultToExternalBackend(
                "gnn-ga", config_.transformerBackend, body, out);
            respondJson(cb, json{{"ok", true},
                                 {"result", out},
                                 {"externalBridge", bridge}});
          } catch (const std::exception &e) {
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/learn/gnn_ga/latest",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          respondJson(cb, json{{"ok", true},
                               {"latest", gnnGa_ ? gnnGa_->latest() : json()}});
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/api/learn/adversarial",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          if (advDisabled_)
            return respondJson(cb,
                               json{{"ok", false}, {"error", "adv-disabled"}},
                               drogon::k503ServiceUnavailable);
          try {
            bool ok = true;
            json body = parseRequestBody(req, ok);
            if (!ok)
              return badRequest(cb, "invalid json");
            auto payload = std::make_shared<json>(body);
            if (!payload || !payload->contains("samples") ||
                !(*payload)["samples"].is_array())
              return badRequest(cb, "samples required");
            if ((*payload)["samples"].empty())
              return badRequest(cb, "samples required");
            auto out = ensureADV()->attackAndDefend((*payload)["samples"]);
            json bridge = bridgeLearnerResultToExternalBackend(
                "adversarial", config_.transformerBackend, body, out);
            respondJson(cb, json{{"ok", true},
                                 {"result", out},
                                 {"externalBridge", bridge}});
          } catch (const std::exception &e) {
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/learn/adversarial/latest",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          respondJson(cb, json{{"ok", true},
                               {"latest", adv_ ? adv_->latest() : json()}});
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/api/learn/thresholds",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          bool ok = true;
          json body = parseRequestBody(req, ok);
          if (!ok)
            return badRequest(cb, "invalid json");
          if (body.is_object()) {
            if (body.contains("rlEvery") && body["rlEvery"].is_number()) {
              double v = body["rlEvery"].get<double>();
              if (std::isfinite(v) && v > 0)
                dialogThresholds_.rlEvery = v;
            }
            if (body.contains("advEvery") && body["advEvery"].is_number()) {
              double v = body["advEvery"].get<double>();
              if (std::isfinite(v) && v > 0)
                dialogThresholds_.advEvery = v;
            }
            if (body.contains("gnnEvery") && body["gnnEvery"].is_number()) {
              double v = body["gnnEvery"].get<double>();
              if (std::isfinite(v) && v > 0)
                dialogThresholds_.gnnEvery = v;
            }
          }
          respondJson(cb,
                      json{{"ok", true},
                           {"thresholds",
                            json{{"rlEvery", dialogThresholds_.rlEvery},
                                 {"advEvery", dialogThresholds_.advEvery},
                                 {"gnnEvery", dialogThresholds_.gnnEvery}}}});
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/memebarrier/start",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          try {
            bool ok = true;
            json body = parseRequestBody(req, ok);
            if (!ok)
              return badRequest(cb, "invalid json");
            double th = getModelDefaults().maliciousThreshold;
            if (body.contains("maliciousThreshold") &&
                body["maliciousThreshold"].is_number()) {
              th = body["maliciousThreshold"].get<double>();
            }
            auto rt = std::dynamic_pointer_cast<RuntimeState>(
                pool_->getActive()->runtime());
            if (!rt)
              return respondJson(
                  cb,
                  json{{"ok", false},
                       {"error",
                        "memebarrier not supported on current backend"}},
                  drogon::k400BadRequest);
            if (!barrier_)
              barrier_ = std::make_shared<MemeBarrier>(rt);
            barrier_->setThreshold(th);
            barrier_->start();
            respondJson(cb, json{{"ok", true},
                                 {"running", true},
                                 {"threshold", barrier_->threshold()}});
          } catch (const std::exception &e) {
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/memebarrier/stop",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          try {
            if (barrier_)
              barrier_->stop();
            respondJson(cb, json{{"ok", true}, {"running", false}});
          } catch (const std::exception &e) {
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/memebarrier/stats",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          try {
            json stats;
            if (barrier_) {
              auto s = barrier_->getStats();
              stats = json{{"scans", s.scans},
                           {"isolated", s.isolated},
                           {"lastScanTime", s.lastScanTime},
                           {"lastIsolated", s.lastIsolated}};
            }
            respondJson(cb, json{{"ok", true}, {"stats", stats}});
          } catch (const std::exception &e) {
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Get});
    drogon::app().registerHandler(
        "/api/model/params",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          if (req->method() == drogon::Get) {
            respondJson(
                cb,
                json{{"ok", true},
                     {"params", pool_->getActive()->runtime()->cloneParams()}});
          } else if (req->method() == drogon::Post) {
            bool ok = true;
            json body = parseRequestBody(req, ok);
            if (!ok)
              return badRequest(cb, "invalid json");
            pool_->getActive()->applyParams(body);
            respondJson(
                cb,
                json{{"ok", true},
                     {"params", pool_->getActive()->runtime()->cloneParams()}});
          }
        },
        {drogon::Get, drogon::Post});

    drogon::app().registerHandler(
        "/api/model/params/reset",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          json defaults =
              json{{"decayFactor", getModelDefaults().decayFactor},
                   {"maxMemeWords", getModelDefaults().maxMemeWords},
                   {"minOverlapThreshold", getModelDefaults().minOverlapThreshold},
                   {"memeNgramMin", getModelDefaults().memeNgramMin},
                   {"memeNgramMax", getModelDefaults().memeNgramMax},
                   {"maliciousThreshold", getModelDefaults().maliciousThreshold},
                   {"learningIterations", getModelDefaults().learningIterations},
                   {"iteration", getModelDefaults().iteration},
                   {"threshold", getModelDefaults().threshold},
                   {"decay", getModelDefaults().decay},
                   {"decayK", getModelDefaults().decayK},
                   {"maxLen", getModelDefaults().maxLen},
                   {"edgeWeight", getModelDefaults().edgeWeight},
                   {"activationType", getModelDefaults().activationType},
                   {"transferType", getModelDefaults().transferType},
                   {"activationCustom", getModelDefaults().activationCustom},
                   {"transferCustom", getModelDefaults().transferCustom},
                   {"mappingDepth", getModelDefaults().mappingDepth},
                   {"reflectionTopMemes", getModelDefaults().reflectionTopMemes},
                   {"reflectionTopWords", getModelDefaults().reflectionTopWords},
                   {"reflectionMinScore", getModelDefaults().reflectionMinScore}};
          pool_->getActive()->applyParams(defaults);
          respondJson(cb, json{{"ok", true},
                               {"params",
                                pool_->getActive()->runtime()->cloneParams()}});
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/array/layers",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          json history = json::array();
          auto &h = spark_->history();
          int start = (int)h.size() > 20 ? (int)h.size() - 20 : 0;
          for (int i = start; i < (int)h.size(); i++)
            history.push_back(h[i]);
          respondJson(cb, json{{"ok", true},
                               {"layers", spark_->getLayers()},
                               {"history", history}});
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/api/array/layers",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          bool ok = true;
          json body = parseRequestBody(req, ok);
          if (!ok)
            return badRequest(cb, "invalid json");
          try {
            spark_->updateLayers(body.value("layers", json::array()));
            respondJson(cb,
                        json{{"ok", true}, {"layers", spark_->getLayers()}});
          } catch (const std::exception &e) {
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k400BadRequest);
          }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/array/history",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          json history = json::array();
          auto &h = spark_->history();
          int start = (int)h.size() > 20 ? (int)h.size() - 20 : 0;
          for (int i = start; i < (int)h.size(); i++)
            history.push_back(h[i]);
          respondJson(cb, json{{"ok", true}, {"history", history}});
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/api/system/status",
        [this](const drogon::HttpRequestPtr &,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          json out;
          const auto now = std::chrono::steady_clock::now();
          const auto uptime =
              std::chrono::duration_cast<std::chrono::duration<double>>(
                  now - startedAt_)
                  .count();
          out["ok"] = true;
          out["version"] = phoenix::PHOENIX_VERSION_STRING;
          out["codename"] = phoenix::PHOENIX_CODENAME;
          out["uptime"] = uptime;
          out["load"] = getLoadAvg();
          out["processCpuPercent"] = getProcessCpuPercent();
          out["memory"] = getMemoryUsage();
          out["controllers"] = pool_->listMetrics();
          out["groups"] = json::array();
          for (auto &gid : pool_->listGroupIds()) {
            out["groups"].push_back(
                {{"gid", gid},
                 {"controllers", pool_->listControllersInGroup(gid)}});
          }
          out["rotation"] =
              json{{"running", rotation_ ? rotation_->running() : false},
                   {"cycleMs", rotation_ ? rotation_->cycleMs() : 0}};
          out["redis"] =
              json{{"channel", redis_ ? redis_->channel() : std::string("")}};
          respondJson(cb, out);
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/api/system/config",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          json cfg;
          cfg["groupCount"] = config_.groupCount;
          cfg["groupSize"] = config_.groupSize;
          cfg["sparkNumAI"] = config_.sparkNumAI;
          cfg["sparkBudget"] = config_.sparkBudget;
          cfg["groupIds"] = pool_->listGroupIds();
          cfg["gatewayHost"] = config_.gatewayHost;
          cfg["portGateway"] = config_.portGateway;
          cfg["portStudy"] = config_.portStudy;
          respondJson(cb, json{{"ok", true}, {"config", cfg}});
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/api/world/status",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          if (!worldModel_)
            return respondJson(
                cb, json{{"ok", false}, {"error", "world model unavailable"}});
          respondJson(cb,
                      json{{"ok", true}, {"status", worldModel_->status()}});
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/api/world/config",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          json out;
          out["ok"] = true;
          out["worldModelEnabled"] = (worldModel_ != nullptr);
          out["cognitionAutonomyEnabled"] = cognitionAutonomyEnabled_;
          out["ingestEnabled"] = (worldModel_ != nullptr);
          respondJson(cb, out);
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/api/groups",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          json out;
          out["ok"] = true;
          out["groupSize"] = config_.groupSize;
          out["groupCount"] = config_.groupCount;
          out["groups"] = json::array();
          for (auto &gid : pool_->listGroupIds()) {
            out["groups"].push_back(
                {{"gid", gid},
                 {"controllers", pool_->listControllersInGroup(gid)}});
          }
          respondJson(cb, out);
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/api/groups/{1}/metrics",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb,
               const std::string &gid) {
          if (!authOK(req))
            return unauthorized(cb);
          auto list = pool_->listControllersInGroup(gid);
          if (list.empty())
            return respondJson(
                cb, json{{"ok", false}, {"error", "group-not-found"}},
                drogon::k404NotFound);
          json metrics = json::array();
          if (isGroupProc()) {
            for (auto &name : list) {
              (void)name;
              metrics.push_back(json::object());
            }
          } else {
            for (auto &name : list) {
              auto ctrl = pool_->getByName(name);
              if (ctrl)
                metrics.push_back(ctrl->metrics());
            }
          }
          respondJson(
              cb, json{{"ok", true}, {"gid", gid}, {"controllers", metrics}});
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/api/system/ai/{1}",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb,
               const std::string &name) {
          if (!authOK(req))
            return unauthorized(cb);
          auto ctrl = pool_->getByName(name);
          if (!ctrl)
            return respondJson(
                cb, json{{"ok", false}, {"error", "Controller not found"}},
                drogon::k404NotFound);
          if (isGroupProc()) {
            respondJson(cb, json{{"ok", true}, {"controller", json::object()}});
          } else {
            respondJson(cb,
                        json{{"ok", true}, {"controller", ctrl->metrics()}});
          }
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/api/shards",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          respondJson(cb, json{{"ok", true}, {"shards", shards_->metrics()}});
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/api/study/enqueue",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          bool ok = true;
          json body = parseRequestBody(req, ok);
          if (!ok)
            return badRequest(cb, "invalid json");
          if (!body.contains("text") || !jsTruthy(body["text"]))
            return badRequest(cb, "text required");
          json doc{{"text", body["text"]}, {"queuedAt", nowEpochMs()}};
          if (study_)
            study_->enqueueDocument(doc);
          respondJson(cb, json{{"ok", true}});
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/corpus/ingest",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          bool ok = true;
          json payload = parseRequestBody(req, ok);
          if (!ok)
            return badRequest(cb, "invalid json");
          if (!payload.contains("text") || !jsTruthy(payload["text"]))
            return badRequest(cb, "text required");
          try {
            auto results = pool_->ingestDocument(payload);
            if (payload.value("enqueueStudy", false) && study_) {
              study_->enqueueDocument(payload);
            }
            respondJson(cb, json{{"ok", true}, {"results", results}});
          } catch (const std::exception &e) {
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/corpus/forget",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          try {
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok)
              return badRequest(cb, "invalid json");
            auto results = pool_->forgetMemes(payload);
            respondJson(cb, json{{"ok", true}, {"results", results}});
          } catch (const std::exception &e) {
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/corpus/online",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          try {
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok)
              return badRequest(cb, "invalid json");
            std::string query = payload.value("query", "");
            json tokens = payload.value("tokens", json::array());
            if (query.empty() && (!tokens.is_array() || tokens.empty()))
              return badRequest(cb, "query or tokens required");
            auto result =
                pool_->onlineResearch(!query.empty() ? json(query) : tokens,
                                      payload.value("options", json::object()));
            respondJson(cb, json{{"ok", true}, {"result", result}});
          } catch (const std::exception &e) {
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/corpus/crawl",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          bool ok = true;
          json payload = parseRequestBody(req, ok);
          if (!ok)
            return badRequest(cb, "invalid json");
          std::string url = trimString(payload.value("startUrl", ""));
          if (url.empty())
            return badRequest(cb, "startUrl required");
          json options = payload.value("options", json::object());
          json crawl = json{{"startUrl", url}};
          if (options.is_object()) {
            for (auto it = options.begin(); it != options.end(); ++it) {
              crawl[it.key()] = it.value();
            }
          }
          json result = pool_->getActive()->runtime()->onlineLookup(
              url, json{{"mode", "crawl"}, {"crawl", crawl}});
          json ingested = nullptr;
          if (payload.value("ingest", false)) {
            json details = json::array();
            int docsCount = 0;
            for (auto &p : result.value("pages", json::array())) {
              std::string text = p.value("text", "");
              if (text.empty())
                continue;
              std::string source =
                  payload.value("source", p.value("url", "crawl"));
              json doc{{"text", text}, {"source", source}};
              std::string gid = payload.value("groupId", "");
              json r = gid.empty() ? pool_->ingestDocument(doc)
                                   : pool_->ingestDocumentToGroup(gid, doc);
              details.push_back(json{{"source", source}, {"result", r}});
              docsCount++;
              if (docsCount >= 20)
                break;
            }
            ingested = json{{"docs", docsCount}, {"details", details}};
          }
          respondJson(
              cb,
              json{{"ok", true}, {"result", result}, {"ingested", ingested}});
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/search/config",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          try {
            auto runtime = pool_->getActive()->runtime();
            respondJson(
                cb, json{{"ok", true}, {"config", runtime->getSearchConfig()}});
          } catch (const std::exception &e) {
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/api/search/config",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          try {
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok)
              return badRequest(cb, "invalid json");
            auto runtime = pool_->getActive()->runtime();
            auto next = runtime->setSearchConfig(payload);
            respondJson(cb, json{{"ok", true}, {"config", next}});
          } catch (const std::exception &e) {
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Put});

    drogon::app().registerHandler(
        "/api/search/endpoints/add",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          try {
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok)
              return badRequest(cb, "invalid json");
            std::string url = trimString(payload.value("url", ""));
            if (url.empty())
              return badRequest(cb, "url required");
            auto runtime = pool_->getActive()->runtime();
            auto cfg = runtime->getSearchConfig();
            std::unordered_set<std::string> uniq;
            json endpoints = json::array();
            for (auto &e : cfg.value("endpoints", json::array())) {
              if (e.is_string() && uniq.insert(e.get<std::string>()).second)
                endpoints.push_back(e);
            }
            if (uniq.insert(url).second)
              endpoints.push_back(url);
            auto next =
                runtime->setSearchConfig(json{{"endpoints", endpoints}});
            respondJson(cb, json{{"ok", true}, {"config", next}});
          } catch (const std::exception &e) {
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/search/endpoints/remove",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          try {
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok)
              return badRequest(cb, "invalid json");
            std::string url = trimString(payload.value("url", ""));
            if (url.empty())
              return badRequest(cb, "url required");
            auto runtime = pool_->getActive()->runtime();
            auto cfg = runtime->getSearchConfig();
            json endpoints = json::array();
            for (auto &e : cfg.value("endpoints", json::array())) {
              if (e.is_string() && e.get<std::string>() != url)
                endpoints.push_back(e);
            }
            json patch{{"endpoints", endpoints}};
            if (cfg.value("active", "") == url)
              patch["active"] = endpoints.empty() ? "" : endpoints[0];
            auto next = runtime->setSearchConfig(patch);
            respondJson(cb, json{{"ok", true}, {"config", next}});
          } catch (const std::exception &e) {
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/export/graph",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          try {
            bool ok = true;
            json body = parseRequestBody(req, ok);
            if (!ok)
              return badRequest(cb, "invalid json");
            json opts{{"seeds", body.value("seeds", json())},
                      {"radius", body.value("radius", json())},
                      {"file", body.value("file", json())}};
            auto runtime = pool_->getActive()->runtime();
            auto outFile = runtime->exportGraphToFile(opts);
            auto content = readTextFile(outFile);
            respondJson(cb, json{{"ok", true},
                                 {"file", outFile.filename().string()},
                                 {"path", outFile.string()},
                                 {"content", content}});
          } catch (const std::exception &e) {
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/export/graph/group",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          try {
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok)
              return badRequest(cb, "invalid json");
            std::string gid = trimString(payload.value("groupId", ""));
            if (gid.empty())
              return badRequest(cb, "groupId required");
            auto list = pool_->listControllersInGroup(gid);
            if (list.empty())
              return respondJson(
                  cb, json{{"ok", false}, {"error", "group-not-found"}},
                  drogon::k404NotFound);
            auto ctrl = pool_->getByName(list.front());
            if (!ctrl)
              return respondJson(
                  cb, json{{"ok", false}, {"error", "controller-not-found"}},
                  drogon::k404NotFound);
            json opts{{"seeds", payload.value("seeds", json())},
                      {"radius", payload.value("radius", json())},
                      {"file", payload.value("file", json())}};
            auto outFile = ctrl->runtime()->exportGraphToFile(opts);
            auto content = readTextFile(outFile);
            respondJson(cb, json{{"ok", true},
                                 {"groupId", gid},
                                 {"file", outFile.filename().string()},
                                 {"path", outFile.string()},
                                 {"content", content}});
          } catch (const std::exception &e) {
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/robots/retrain",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          auto routeStarted = std::chrono::steady_clock::now();
          auto finalizeMetric = [&](bool okFlag) {
            auto elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - routeStarted)
                    .count();
            recordRouteMetric("/api/robots/retrain", elapsed, okFlag);
          };
          if (!authOK(req)) {
            finalizeMetric(false);
            return unauthorized(cb);
          }
          try {
            bool ok = true;
            json options = parseRequestBody(req, ok);
            if (!ok) {
              finalizeMetric(false);
              return badRequest(cb, "invalid json");
            }
            json files = json();
            if (options.contains("files")) {
              if (options["files"].is_array()) {
                files = options["files"];
              } else if (options["files"].is_string()) {
                std::string f = trimString(options["files"].get<std::string>());
                if (!f.empty())
                  files = json::array({f});
              }
            }
            bool shuffle = options.contains("shuffle")
                               ? jsTruthy(options["shuffle"])
                               : false;
            int64_t limit = 2000;
            int64_t offset = 0;
            int64_t limitCap = getRobotsRetrainLimitCap();
            int64_t batchSize = getRobotsRetrainBatchDefault();
            int64_t batchCharsBudget = getRobotsRetrainBatchCharsBudget();
            int64_t docMaxChars = getRobotsRetrainDocMaxChars();
            int64_t docMaxTokens = getRobotsRetrainDocMaxTokens();
            std::string parseErr;
            if (!parseBoundedInt64(options, "limit", 1, limitCap, limit,
                                   parseErr, false)) {
              finalizeMetric(false);
              return badRequest(cb, parseErr);
            }
            if (!parseBoundedInt64(options, "offset", 0, 1000000000LL, offset,
                                   parseErr, false)) {
              finalizeMetric(false);
              return badRequest(cb, parseErr);
            }
            if (!parseBoundedInt64(options, "batchSize", 64, 5000, batchSize,
                                   parseErr, false)) {
              finalizeMetric(false);
              return badRequest(cb, parseErr);
            }
            batchSize = std::min(batchSize, limit);
            if (retrainTrace_) {
              appendFatalLog(std::string("retrain-start limit=") +
                             std::to_string(limit) +
                             " offset=" + std::to_string(offset) +
                             " batch=" + std::to_string(batchSize));
            }
            uint64_t monitorId = beginTrainMonitor(
                "/api/robots/retrain",
                json{{"limit", limit},
                     {"offset", offset},
                     {"shuffle", shuffle},
                     {"enqueueStudy", options.value("enqueueStudy", false)}});
            touchTrainMonitor("collect-documents", 0, 0, 0, 0);
            auto runtime = pool_->getActive()->runtime();
            int64_t remaining = limit;
            int64_t cursorOffset = offset;
            int64_t collected = 0;
            int64_t batchIndex = 0;
            int64_t ingested = 0;
            int64_t memes = 0;
            int64_t edges = 0;
            int64_t failed = 0;
            uint64_t totalChars = 0;
            double avgDocChars = 1200.0;
            bool anyDocs = false;
            auto groups = pool_->listGroupIds();
            while (remaining > 0) {
              batchIndex++;
              int64_t take = std::min(batchSize, remaining);
              if (batchCharsBudget > 0) {
                double safeAvg = std::max(64.0, avgDocChars);
                int64_t byChars =
                    (int64_t)std::floor((double)batchCharsBudget / safeAvg);
                byChars = std::max<int64_t>(32, byChars);
                take = std::max<int64_t>(1, std::min<int64_t>(take, byChars));
              }
              json collect = json::object();
              collect["limit"] = take;
              collect["offset"] = cursorOffset;
              collect["shuffle"] = shuffle;
              if (files.is_array() && !files.empty())
                collect["files"] = files;

              auto docs = runtime->collectRobotsDocuments(collect);
              if (docs.empty()) {
                break;
              }
              uint64_t batchChars = 0;
              for (const auto &doc : docs)
                batchChars += (uint64_t)doc.text.size();
              double batchAvg =
                  docs.empty() ? avgDocChars
                               : ((double)batchChars / (double)docs.size());
              if (std::isfinite(batchAvg) && batchAvg > 0.0)
                avgDocChars = avgDocChars * 0.75 + batchAvg * 0.25;
              anyDocs = true;
              collected += (int64_t)docs.size();
              touchTrainMonitor("collect-documents", (uint64_t)limit,
                                (uint64_t)collected, 0, totalChars);

              for (auto &doc : docs) {
                std::string docText = doc.text;
                if ((int64_t)docText.size() > docMaxChars)
                  docText.resize((size_t)docMaxChars);
                std::vector<std::string> docTokens = doc.tokens;
                if ((int64_t)docTokens.size() > docMaxTokens)
                  docTokens.resize((size_t)docMaxTokens);
                if (docText.empty() && docTokens.empty())
                  continue;

                totalChars += (uint64_t)docText.size();
                json j{{"text", docText},
                       {"tokens", docTokens},
                       {"source", doc.source},
                       {"file", doc.file},
                       {"id", doc.id}};
                std::string sourceKey =
                    !doc.source.empty()
                        ? doc.source
                        : (!doc.file.empty() ? doc.file : doc.id);
                std::string key = sourceKey + "|" + docText.substr(0, 256);
                int gi = groups.empty()
                             ? 0
                             : (int)(hashStrSimple(key) % groups.size());
                std::string target = groups.empty() ? "" : groups[gi];
                json result = target.empty()
                                  ? pool_->ingestDocument(j)
                                  : pool_->ingestDocumentToGroup(target, j);
                ingested++;
                if (result.is_array()) {
                  for (auto &item : result) {
                    bool okItem = item.is_object() ? item.value("ok", true)
                                                   : jsTruthy(item);
                    if (okItem && item.is_object()) {
                      memes += item.value("memes", 0);
                      edges += item.value("edges", 0);
                    } else if (!okItem) {
                      failed++;
                    }
                  }
                } else {
                  bool okItem = result.is_object() ? result.value("ok", true)
                                                   : jsTruthy(result);
                  if (okItem && result.is_object()) {
                    memes += result.value("memes", 0);
                    edges += result.value("edges", 0);
                  } else if (!okItem) {
                    failed++;
                  }
                }
                if (options.value("enqueueStudy", false) && study_)
                  study_->enqueueDocument(j);
                if ((ingested % 64) == 0) {
                  touchTrainMonitor("ingest-progress", (uint64_t)limit,
                                    (uint64_t)ingested, 0, totalChars);
                }
              }

              cursorOffset += (int64_t)docs.size();
              remaining -= (int64_t)docs.size();
              if (retrainTrace_ && (batchIndex % 10 == 0)) {
                appendFatalLog(std::string("retrain-progress batch=") +
                               std::to_string(batchIndex) + " take=" +
                               std::to_string(take) + " avgDocChars=" +
                               std::to_string((int64_t)avgDocChars) +
                               " collected=" + std::to_string(collected) +
                               " ingested=" + std::to_string(ingested) +
                               " remaining=" + std::to_string(remaining));
              }
              if ((int64_t)docs.size() < take) {
                break;
              }
            }

            if (!anyDocs) {
              finishTrainMonitor(false,
                                 json{{"ok", false}, {"error", "no-documents"}},
                                 "no-documents");
              finalizeMetric(false);
              return respondJson(cb,
                                 json{{"ok", false},
                                      {"error", "no-documents"},
                                      {"monitorId", monitorId}},
                                 drogon::k404NotFound);
            }

            json out{{"ok", true},
                     {"ingested", ingested},
                     {"memes", memes},
                     {"edges", edges},
                     {"failedControllers", failed},
                     {"requestedLimit", limit},
                     {"limitCap", limitCap},
                     {"batchSize", batchSize},
                     {"monitorId", monitorId}};
            if (ingested < limit) {
              out["warning"] = "source exhausted before requested limit";
            } else {
              out["warning"] =
                  "requested limit reached; use offset for next batch";
            }
            finishTrainMonitor(true, out);
            if (retrainTrace_) {
              appendFatalLog(std::string("retrain-done ingested=") +
                             std::to_string(ingested) +
                             " failed=" + std::to_string(failed));
            }
            finalizeMetric(true);
            respondJson(cb, out);
          } catch (const std::bad_alloc &e) {
            finishTrainMonitor(false, json::object(),
                               std::string("bad_alloc: ") + e.what());
            if (retrainTrace_) {
              appendFatalLog(std::string("retrain-bad-alloc: ") + e.what());
            }
            logRouteError("/api/robots/retrain",
                          std::string("bad_alloc: ") + e.what());
            finalizeMetric(false);
            respondJson(cb,
                        json{{"ok", false},
                             {"error", "insufficient memory during retrain"}},
                        drogon::k500InternalServerError);
          } catch (const std::exception &e) {
            finishTrainMonitor(false, json::object(), e.what());
            if (retrainTrace_) {
              appendFatalLog(std::string("retrain-exception: ") + e.what());
            }
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
            finalizeMetric(false);
          } catch (...) {
            finishTrainMonitor(false, json::object(), "internal error");
            if (retrainTrace_) {
              appendFatalLog("retrain-exception: internal error");
            }
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
            finalizeMetric(false);
          }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/robots/list",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          try {
            auto runtime = pool_->getActive()->runtime();
            respondJson(
                cb, json{{"ok", true},
                         {"directory", runtime->config().robotsDir.string()},
                         {"files", runtime->listRobotsFiles()}});
          } catch (const std::exception &e) {
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/robots/ingest",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          try {
            bool ok = true;
            json options = parseRequestBody(req, ok);
            if (!ok)
              return badRequest(cb, "invalid json");
            json files = json();
            if (options.contains("files")) {
              if (options["files"].is_array()) {
                files = options["files"];
              } else if (options["files"].is_string()) {
                std::string f = trimString(options["files"].get<std::string>());
                if (!f.empty())
                  files = json::array({f});
              }
            }
            bool shuffle = options.contains("shuffle")
                               ? jsTruthy(options["shuffle"])
                               : false;
            json collect = json::object();
            if (options.contains("limit"))
              collect["limit"] = options["limit"];
            if (options.contains("offset"))
              collect["offset"] = options["offset"];
            collect["shuffle"] = shuffle;
            if (files.is_array() && !files.empty())
              collect["files"] = files;
            auto runtime = pool_->getActive()->runtime();
            auto docs = runtime->collectRobotsDocuments(collect);
            if (docs.empty())
              return respondJson(cb,
                                 json{{"ok", false}, {"error", "no-documents"}},
                                 drogon::k404NotFound);
            json summary = json::array();
            auto groups = pool_->listGroupIds();
            for (auto &doc : docs) {
              json j{{"text", doc.text},
                     {"tokens", doc.tokens},
                     {"source", doc.source},
                     {"file", doc.file},
                     {"id", doc.id}};
              std::string sourceKey =
                  !doc.source.empty() ? doc.source
                                      : (!doc.file.empty() ? doc.file : doc.id);
              std::string key = sourceKey + "|" + doc.text.substr(0, 256);
              int gi = groups.empty()
                           ? 0
                           : (int)(hashStrSimple(key) % groups.size());
              std::string target = groups.empty() ? "" : groups[gi];
              json result = target.empty()
                                ? pool_->ingestDocument(j)
                                : pool_->ingestDocumentToGroup(target, j);
              int docMemes = 0;
              int docEdges = 0;
              int docFailed = 0;
              if (result.is_array()) {
                for (auto &item : result) {
                  bool okItem = item.is_object() ? item.value("ok", true)
                                                 : jsTruthy(item);
                  if (okItem && item.is_object()) {
                    docMemes += item.value("memes", 0);
                    docEdges += item.value("edges", 0);
                  } else if (!okItem) {
                    docFailed++;
                  }
                }
              } else {
                bool okItem = result.is_object() ? result.value("ok", true)
                                                 : jsTruthy(result);
                if (okItem && result.is_object()) {
                  docMemes += result.value("memes", 0);
                  docEdges += result.value("edges", 0);
                } else if (!okItem) {
                  docFailed++;
                }
              }
              json item{{"id", doc.id},
                        {"source", doc.source},
                        {"file", doc.file},
                        {"tokens", (int)doc.tokens.size()},
                        {"memes", docMemes},
                        {"edges", docEdges},
                        {"failedControllers", docFailed}};
              summary.push_back(item);
              if (options.value("enqueueStudy", false) && study_)
                study_->enqueueDocument(j);
            }
            std::unordered_set<std::string> seen;
            json filesOut = json::array();
            bool nullAdded = false;
            for (auto &item : summary) {
              if (item.contains("file") && item["file"].is_string()) {
                std::string f = item["file"].get<std::string>();
                if (seen.insert(f).second)
                  filesOut.push_back(f);
              } else if (!nullAdded) {
                filesOut.push_back(nullptr);
                nullAdded = true;
              }
            }
            respondJson(cb, json{{"ok", true},
                                 {"ingested", (int)summary.size()},
                                 {"files", filesOut},
                                 {"summary", summary}});
          } catch (const std::exception &e) {
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/tests/list",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          try {
            fs::path testsDir =
                pool_->getActive()->runtime()->config().testsDir;
            std::vector<std::string> names;
            if (fs::exists(testsDir)) {
              for (auto &entry : fs::directory_iterator(testsDir)) {
                if (!entry.is_regular_file())
                  continue;
                auto ext = entry.path().extension().string();
                std::string lowerExt;
                lowerExt.reserve(ext.size());
                for (char c : ext)
                  lowerExt.push_back(static_cast<char>(
                      std::tolower(static_cast<unsigned char>(c))));
                if (lowerExt == ".txt")
                  names.push_back(entry.path().filename().string());
              }
            }
            std::sort(names.begin(), names.end());
            json files = json::array();
            for (auto &n : names)
              files.push_back(n);
            respondJson(cb, json{{"ok", true},
                                 {"directory", testsDir.string()},
                                 {"files", files}});
          } catch (const std::exception &e) {
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/api/tests/case",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          try {
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok)
              return badRequest(cb, "invalid json");
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
            json nameVal = payload.contains("name") ? payload["name"] : json();
            json contentVal =
                payload.contains("content") ? payload["content"] : json();
            std::string nameRaw =
                trimString(jsTruthy(nameVal) ? jsStringValue(nameVal) : "");
            std::string content =
                jsTruthy(contentVal) ? jsStringValue(contentVal) : "";
            if (nameRaw.empty())
              return badRequest(cb, "name required");
            std::string safe;
            safe.reserve(nameRaw.size());
            for (char c : nameRaw) {
              if (std::isalnum(static_cast<unsigned char>(c)) || c == '.' ||
                  c == '_' || c == '-')
                safe.push_back(c);
              else
                safe.push_back('_');
            }
            auto lower = safe;
            std::transform(lower.begin(), lower.end(), lower.begin(),
                           [](unsigned char c) {
                             return static_cast<char>(std::tolower(c));
                           });
            if (lower.size() < 4 || lower.substr(lower.size() - 4) != ".txt")
              safe += ".txt";
            fs::path testsDir =
                pool_->getActive()->runtime()->config().testsDir;
            ensureDir(testsDir);
            fs::path filePath = testsDir / safe;
            std::ofstream ofs(filePath, std::ios::binary);
            ofs << content;
            respondJson(cb, json{{"ok", true},
                                 {"file", safe},
                                 {"path", filePath.string()}});
          } catch (const std::exception &e) {
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/tests/refresh",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          try {
            fs::path testsDir =
                pool_->getActive()->runtime()->config().testsDir;
            bool refreshed = false;
            if (rl_) {
              try {
                rl_->refreshTests(testsDir);
                refreshed = true;
              } catch (...) {
                try {
                  rl_->setTestsDir(testsDir);
                  refreshed = true;
                } catch (...) {
                }
              }
            } else {
              auto factory = module_mount::ModuleRegistry::instance()
                                 .reinforcementLearnerFactory();
              if (factory)
                rl_ = factory(pool_, testsDir);
              else
                rl_ = std::make_shared<ReinforcementLearner>(pool_, testsDir);
              refreshed = true;
            }
            respondJson(cb, json{{"ok", true},
                                 {"refreshed", refreshed},
                                 {"testsDir", testsDir.string()}});
          } catch (const std::exception &e) {
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          } catch (...) {
            respondJson(cb, json{{"ok", false}, {"error", "internal error"}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/snapshots",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          respondJson(cb, json{{"ok", true}, {"list", snapshots_->list()}});
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/api/snapshots/create",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (!authOK(req))
            return unauthorized(cb);
          bool ok = true;
          json payload = parseRequestBody(req, ok);
          if (!ok)
            return badRequest(cb, "invalid json");
          std::string name = payload.value("name", "manual");
          auto file = snapshots_->create(name);
          respondJson(cb,
                      json{{"ok", true}, {"file", file.filename().string()}});
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/snapshots/restore/{1}",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb,
               const std::string &id) {
          if (!authOK(req))
            return unauthorized(cb);
          try {
            snapshots_->restore(id);
            respondJson(cb, json{{"ok", true}});
          } catch (const std::exception &e) {
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k404NotFound);
          }
        },
        {drogon::Post});

    drogon::app().registerHandler(
        "/api/snapshots/{1}",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb,
               const std::string &id) {
          if (!authOK(req))
            return unauthorized(cb);
          snapshots_->remove(id);
          respondJson(cb, json{{"ok", true}});
        },
        {drogon::Delete});

    // /context/reset endpoint for cross-session learning
    std::cout << "[gateway] Registering /context/reset handler" << std::endl;
    drogon::app().registerHandler(
        "/context/reset",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (config_.httpLog)
            std::cout << "[gateway] ENTER /context/reset" << std::endl;
          try {
            auto jsonObj = req->getJsonObject();
            if (!jsonObj || !jsonObj->isMember("sessionId")) {
              respondJson(cb,
                          nlohmann::json{{"ok", false}, {"error", "Missing sessionId"}},
                          drogon::k400BadRequest);
              return;
            }
            const std::string sessionId = (*jsonObj)["sessionId"].asString();
            bool removed = contextService_.reset(sessionId);
            respondJson(cb,
                        nlohmann::json{{"ok", true},
                             {"sessionId", sessionId},
                             {"removed", removed}},
                        drogon::k200OK);
          } catch (const std::exception &e) {
            logRouteError("/context/reset", e.what());
            respondJson(cb, nlohmann::json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Post});

    // /context/knowledge endpoint for cross-session learning status
    std::cout << "[gateway] Registering /context/knowledge handler"
              << std::endl;
    drogon::app().registerHandler(
        "/context/knowledge",
        [this](const drogon::HttpRequestPtr &req,
               std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
          if (config_.httpLog)
            std::cout << "[gateway] ENTER /context/knowledge" << std::endl;
          try {
            auto status = contextService_.knowledgeStatus();
            respondJson(cb, status, drogon::k200OK);
          } catch (const std::exception &e) {
            logRouteError("/context/knowledge", e.what());
            respondJson(cb, json{{"ok", false}, {"error", e.what()}},
                        drogon::k500InternalServerError);
          }
        },
        {drogon::Get});
  }

  void onDialogCompleted(const json &result, const json &payload) {
    try {
      dialogCounters_.total += 1;
      if (logger_.enabled()) {
        logger_.log(LoggerCXX::Type::LOG,
                    buildDialogContextLog(result, payload));
        if (logger_.mode() == LoggerCXX::Mode::Debug) {
          logger_.log(LoggerCXX::Type::MEMORY, buildMemorySnapshotLog());
          logger_.log(LoggerCXX::Type::COMPUTE, buildComputeRouteLog(result));
          logger_.log(LoggerCXX::Type::MONITORING, buildMonitoringLog());
        } else if (logger_.mode() == LoggerCXX::Mode::Release) {
          if (!releaseSnapshotLogged_) {
            logger_.log(LoggerCXX::Type::MEMORY, buildMemorySnapshotLog());
            releaseSnapshotLogged_ = true;
          }
        }
      }
      if (!dialogLearningEnabled_)
        return;

      if (dialogAsyncInFlight_.load(std::memory_order_relaxed) >=
          std::max(1, dialogAsyncLimit_))
        return;

      std::string guardText =
          payload.value("text", payload.value("message", ""));
      int guardTokens = roughTokenCount(guardText);
      std::string guardReply = result.value("reply", "");
      bool guardLongQuery = result.value("longQuerySafeMode", false);
      std::transform(guardReply.begin(), guardReply.end(), guardReply.begin(),
                     [](unsigned char c) { return (char)std::tolower(c); });
      if (guardLongQuery || guardReply == "disconnected" ||
          guardText.size() >= 180 || guardTokens >= 16)
        return;

      try {
        json wrapped = json{{"payload", payload}, {"result", result}};
        pool_->getActive()->runtime()->learnFromDialog(wrapped, json());
      } catch (...) {
      }

      try {
        std::string text = payload.value("text", payload.value("message", ""));
        std::string reply = result.value("reply", "");
        std::string graphContext = result.value("graphContext", "");
        if (!text.empty() && !reply.empty()) {
          bool longQuerySafeMode = result.value("longQuerySafeMode", false);
          int textTokens = roughTokenCount(text);
          std::string replyLower = reply;
          std::transform(replyLower.begin(), replyLower.end(),
                         replyLower.begin(),
                         [](unsigned char c) { return (char)std::tolower(c); });
          if (longQuerySafeMode || replyLower == "disconnected" ||
              text.size() >= 180 || textTokens >= 16)
            return;
          if (isLikelyGibberishReply(reply))
            return;
          if (containsInternalContextMarkers(reply) ||
              hasExcessUnknownTokens(reply))
            return;

          if (!dialogLearnGraphContext_) {
            graphContext.clear();
          } else {
            bool cleanedGraph = false;
            graphContext = sanitizeInputText(graphContext, cleanedGraph);
            if (containsInternalContextMarkers(graphContext))
              graphContext.clear();
            if (graphContext.size() > 512)
              graphContext.resize(512);
          }

          float verifyScore = 0.0f;
          if (result.contains("verify") && result["verify"].is_object()) {
            verifyScore = result["verify"].value("score", 0.0f);
          }
          float addonSignal = 0.0f;
          std::string feedbackSource = "implicit_dialog";
          if (result.contains("graph") && result["graph"].is_object()) {
            const auto &g = result["graph"];
            if (g.contains("addon") && g["addon"].is_object()) {
              addonSignal = 1.0f;
              std::string n =
                  g["addon"].value("name", g["addon"].value("addon", ""));
              std::string t = g["addon"].value("type", "");
              if (!n.empty())
                feedbackSource = "addon:" + n;
              else if (!t.empty())
                feedbackSource = "addonType:" + t;
            }
          }
          if (result.contains("sparkGnnStage2") &&
              result["sparkGnnStage2"].is_string()) {
            std::string s = result["sparkGnnStage2"].get<std::string>();
            if (!s.empty())
              addonSignal += 0.2f;
          }
          float addonScale = 0.35f;
          {
            std::lock_guard<std::mutex> lock(transformerMu_);
            addonScale = transformer_.params().value("addonRewardScale", 0.35f);
          }
          float reward = std::max(
              -1.0f, std::min(1.0f, verifyScore + addonScale * addonSignal));
          bool trainTick = dialogTransformerLearning_ && !longQuerySafeMode &&
                           (dialogCounters_.total % 6 == 0);
          transformer::TrainSample sample{text, reply, graphContext};
          if (dialogTransformerLearning_ && !longQuerySafeMode) {
            launchDialogTask([this, sample]() {
              try {
                std::unique_lock<std::mutex> lock(transformerMu_,
                                                  std::try_to_lock);
                if (!lock.owns_lock())
                  return;
                transformer_.jointTrain(
                    std::vector<transformer::TrainSample>{sample}, 1, 0.001f,
                    0.35f);
              } catch (...) {
              }
            });
          }
          if (dialogTransformerLearning_) {
            launchDialogTask([this, text, graphContext, reply, reward,
                              trainTick, feedbackSource]() {
              try {
                std::unique_lock<std::mutex> lock(transformerMu_,
                                                  std::try_to_lock);
                if (!lock.owns_lock())
                  return;
                transformer_.addPreferenceFeedback(text, graphContext, reply,
                                                   std::string(), reward,
                                                   feedbackSource);
                if (trainTick) {
                  transformer_.trainFromFeedback(1);
                }
              } catch (...) {
              }
            });
          }
        }
      } catch (...) {
      }

      const auto total = dialogCounters_.total;
      if (!rlDisabled_ &&
          (total - dialogCounters_.lastRL >= dialogThresholds_.rlEvery)) {
        dialogCounters_.lastRL = total;
        launchDialogTask([this]() {
          try {
            ensureRL()->learn(1);
          } catch (...) {
          }
        });
      }
      if (!advDisabled_ &&
          (total - dialogCounters_.lastADV >= dialogThresholds_.advEvery)) {
        dialogCounters_.lastADV = total;
        std::string text;
        if (payload.contains("text") && payload["text"].is_string()) {
          text = payload["text"].get<std::string>();
        } else if (payload.contains("tokens") && payload["tokens"].is_array()) {
          std::ostringstream oss;
          bool first = true;
          for (auto &t : payload["tokens"]) {
            if (!t.is_string())
              continue;
            if (!first)
              oss << " ";
            oss << t.get<std::string>();
            first = false;
          }
          text = oss.str();
        }
        auto trim = [](std::string s) {
          auto start = s.find_first_not_of(" \n\r\t");
          if (start == std::string::npos)
            return std::string();
          auto end = s.find_last_not_of(" \n\r\t");
          return s.substr(start, end - start + 1);
        };
        text = trim(text);
        std::vector<std::string> samples;
        if (!text.empty())
          samples.push_back(text);
        if (result.contains("reply") && result["reply"].is_string()) {
          samples.push_back(result["reply"].get<std::string>());
        }
        if (!samples.empty()) {
          launchDialogTask([this, samples]() {
            try {
              ensureADV()->attackAndDefend(samples);
            } catch (...) {
            }
          });
        }
      }

      if (!gnnGaDisabled_ &&
          (total - dialogCounters_.lastGNN >= dialogThresholds_.gnnEvery)) {
        dialogCounters_.lastGNN = total;
        std::vector<std::string> samples;
        std::string text = payload.value("text", payload.value("message", ""));
        if (!text.empty())
          samples.push_back(text);
        if (result.contains("reply") && result["reply"].is_string()) {
          std::string reply = result["reply"].get<std::string>();
          if (!reply.empty())
            samples.push_back(reply);
        }
        if (result.contains("graphContext") &&
            result["graphContext"].is_string()) {
          std::string gc = result["graphContext"].get<std::string>();
          if (dialogLearnGraphContext_ && !gc.empty() &&
              !containsInternalContextMarkers(gc))
            samples.push_back(gc);
        }
        launchDialogTask([this, samples]() {
          try {
            ensureGnnGa()->evolve(gnnGa_->generations(), gnnGa_->population(),
                                 samples, 0.35, 0.25, 0.30);
          } catch (...) {
          }
        });
      }
    } catch (...) {
    }
  }

  void
  respondJson(const std::function<void(const drogon::HttpResponsePtr &)> &cb,
              const json &j, drogon::HttpStatusCode code = drogon::k200OK) {
    try {
      if (logger_.enabled()) {
        int sc = (int)code;
        std::string payload;
        try {
          payload = j.dump();
        } catch (const std::exception &e) {
          payload = std::string("<json-dump-failed: ") + e.what() + ">";
        } catch (...) {
          payload = "<json-dump-failed: unknown>";
        }
        if (payload.size() > 2000)
          payload = payload.substr(0, 2000) + "...";
        if (sc >= 500) {
          logger_.log(LoggerCXX::Type::ERROR, std::string("http status=") +
                                                  std::to_string(sc) +
                                                  " payload=" + payload);
        } else if (sc >= 400) {
          logger_.log(LoggerCXX::Type::WARNING, std::string("http status=") +
                                                    std::to_string(sc) +
                                                    " payload=" + payload);
        }
      }

      auto resp = drogon::HttpResponse::newHttpJsonResponse(toJsoncpp(j));
      resp->setStatusCode(code);
      cb(resp);
      return;
    } catch (const std::exception &e) {
      if (logger_.enabled())
        logger_.log(LoggerCXX::Type::ERROR,
                    std::string("respondJson failed: ") + e.what());
      appendFatalLog(std::string("respondJson exception: ") + e.what());
    } catch (...) {
      if (logger_.enabled())
        logger_.log(LoggerCXX::Type::ERROR,
                    "respondJson failed: unknown exception");
      appendFatalLog("respondJson unknown exception");
    }

    try {
      Json::Value fallback;
      fallback["ok"] = false;
      fallback["error"] = "response-encode-failed";
      auto resp = drogon::HttpResponse::newHttpJsonResponse(fallback);
      resp->setStatusCode(drogon::k500InternalServerError);
      cb(resp);
      return;
    } catch (...) {
    }

    auto resp = drogon::HttpResponse::newHttpResponse();
    resp->setStatusCode(drogon::k500InternalServerError);
    resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
    resp->setBody("{\"ok\":false,\"error\":\"response-failed\"}");
    cb(resp);
  }

  void
  unauthorized(const std::function<void(const drogon::HttpResponsePtr &)> &cb) {
    if (hasAuthError_) {
      json payload{{"ok", false}, {"error", lastAuthError_.error}};
      if (!lastAuthError_.message.empty())
        payload["message"] = lastAuthError_.message;
      logSecurityEvent(std::string("unauthorized response error=") +
                       lastAuthError_.error);
      respondJson(cb, payload, lastAuthError_.code);
      return;
    }
    logSecurityEvent("unauthorized response error=unauthorized");
    respondJson(cb, json{{"ok", false}, {"error", "unauthorized"}},
                drogon::k401Unauthorized);
  }

  void
  badRequest(const std::function<void(const drogon::HttpResponsePtr &)> &cb,
             const std::string &msg) {
    if (logger_.enabled())
      logger_.log(LoggerCXX::Type::WARNING, std::string("bad request: ") + msg);
    respondJson(cb, json{{"ok", false}, {"error", msg}},
                drogon::k400BadRequest);
  }
};

thread_local GatewayServer::AuthError GatewayServer::lastAuthError_{};
thread_local bool GatewayServer::hasAuthError_ = false;

// runGroupWorker 启动组进程工作模式下的本地 RPC 服务。
// 调用方式：通过 group-worker 启动参数进入该函数。
// 实现思路：监听本地端口并按命令分发到组内运行时。
// 注意事项：端口与组配置缺失会直接返回错误码。
static int runGroupWorker(const Config &config) {
  initProcessMemoryReservation();
  std::string groupId = phoenix::resolveConfig<std::string>("main.groupId", "G1", "AI_GROUP_ID");
  if (groupId.empty())
    groupId = "G1";
  int groupSize = std::max(1, phoenix::resolveConfig<int>("main.groupSize", 7, "AI_GROUP_SIZE"));
  int port = 0;
  try {
    port = phoenix::resolveConfig<int>("main.groupWorkerPort", 0, "AI_GROUP_WORKER_PORT");
  } catch (...) {
    port = 0;
  }
  if (port <= 0)
    return 2;

  fs::path lmdbRoot = getProjectRoot() / "lmdb";
  std::string lmdbRootEnv = phoenix::resolveConfig<std::string>("main.lmdbRoot", "", "AI_LMDB_ROOT");
  if (!lmdbRootEnv.empty())
    lmdbRoot = fs::path(lmdbRootEnv);

  std::size_t lmdbMapBytes = phoenix::resolveConfig<std::size_t>("main.lmdbMapBytes", 512ull * 1024ull * 1024ull, "AI_LMDB_MAP_BYTES");
  lmdbMapBytes = std::max<std::size_t>(64ull * 1024ull * 1024ull, lmdbMapBytes);

  ensureDir(config.baseDir);
  if (!config.dbPath.empty()) {
    ensureDir(config.dbPath.parent_path());
  }

#ifdef _WIN32
  WSADATA wsaData{};
  WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif

  SocketHandle serverSock = socket(AF_INET, SOCK_STREAM, 0);
  if (serverSock == kInvalidSocket)
    return 3;
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons((uint16_t)port);
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  int opt = 1;
#ifdef _WIN32
  setsockopt(serverSock, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt,
             sizeof(opt));
#else
  setsockopt(serverSock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif
  if (bind(serverSock, (sockaddr *)&addr, sizeof(addr)) != 0)
    return 4;
  if (listen(serverSock, 1) != 0)
    return 5;

  SocketHandle client = accept(serverSock, nullptr, nullptr);
  if (client == kInvalidSocket)
    return 6;

  std::shared_ptr<KeyValueStore> kvmStore;
  std::shared_ptr<KeyValueStore> memeStore;
  std::shared_ptr<KeyValueStore> sessionStore;
#ifdef HAVE_SQLITE
  auto db = std::make_shared<Database079>(config.dbPath, lmdbRoot,
                                          config.redisUrl, config.redisCacheDb,
                                          config.redisCachePrefix);
  if (db->open()) {
    kvmStore = db->createStore("kvm");
    memeStore = db->createStore("meme_graph");
    sessionStore = db->createStore("session");
  }
#endif
#ifdef HAVE_LMDB
  if (!Database079::isSqliteStore(kvmStore)) {
    auto lmdbK = std::make_shared<LmdbStore>("kvm", lmdbRoot, lmdbMapBytes);
    auto lmdbM =
        std::make_shared<LmdbStore>("meme_graph", lmdbRoot, lmdbMapBytes);
    auto lmdbS = std::make_shared<LmdbStore>("session", lmdbRoot, lmdbMapBytes);
    if (lmdbK->ok())
      kvmStore = lmdbK;
    if (lmdbM->ok())
      memeStore = lmdbM;
    if (lmdbS->ok())
      sessionStore = lmdbS;
  }
#endif
  if (!kvmStore) {
    kvmStore = std::make_shared<JsonFileStore>("kvm", lmdbRoot);
    memeStore = std::make_shared<JsonFileStore>("meme_graph", lmdbRoot);
    sessionStore = std::make_shared<JsonFileStore>("session", lmdbRoot);
  }

  auto kvmNs = std::make_shared<NamespacedStore>(kvmStore, groupId + ":kvm");
  auto memeNs =
      std::make_shared<NamespacedStore>(memeStore, groupId + ":graph");
  auto sessNs =
      std::make_shared<NamespacedStore>(sessionStore, groupId + ":session");

  std::unordered_map<std::string, std::shared_ptr<RuntimeState>> runtimes;
  std::vector<std::string> controllerNames;
  for (int i = 0; i < groupSize; i++) {
    std::string name = groupId + "_AI" + std::to_string(i + 1);
    controllerNames.push_back(name);
    runtimes[name] =
        std::make_shared<RuntimeState>(kvmNs, memeNs, sessNs, config);
  }

  std::shared_ptr<MemeBarrier> memeBarrier;
  auto getRuntime =
      [&](const std::string &name) -> std::shared_ptr<RuntimeState> {
    auto it = runtimes.find(name);
    if (it == runtimes.end())
      throw std::runtime_error("controller-not-found:" + name);
    return it->second;
  };

  auto safeCall = [&](const std::function<json()> &fn) -> json {
    try {
      return json{{"ok", true}, {"result", fn()}};
    } catch (const std::exception &e) {
      return json{{"ok", false}, {"error", std::string(e.what())}};
    }
  };

  std::vector<uint8_t> requestBytes;
  requestBytes.reserve(8ull * 1024ull * 1024ull);
  while (recvPacket(client, requestBytes, 600000)) {
    json msg;
    try {
      msg = json::from_msgpack(requestBytes);
    } catch (...) {
      continue;
    }
    std::string id = msg.value("id", "");
    std::string cmd = msg.value("cmd", "");
    if (id.empty() || cmd.empty())
      continue;

    int shmProtocol = msg.value("shmProtocolVersion", 1);
    if (shmProtocol > kSharedMemoryProtocolVersion) {
      json unsupported{{"ok", false},
                       {"error", "unsupported-shm-protocol"},
                       {"id", id},
                       {"shmProtocolVersion", kSharedMemoryProtocolVersion},
                       {"requestedProtocol", shmProtocol}};
      auto responseBytes = json::to_msgpack(unsupported);
      sendPacket(client, responseBytes);
      continue;
    }

    json reply = safeCall([&]() -> json {
      if (cmd == "respond") {
        auto rt = getRuntime(msg.value("controllerName", ""));
        return rt->processInput(msg.value("payload", json::object()));
      }
      if (cmd == "metrics") {
        auto rt = getRuntime(msg.value("controllerName", ""));
        auto name = msg.value("controllerName", "");
        return json{
            {"name", name},
            {"online", true},
            {"health",
             json{{"status", "ok"}, {"since", nowEpochMs()}, {"failures", 0}}},
            {"metrics", rt->metrics()},
            {"params", rt->cloneParams()}};
      }
      if (cmd == "cloneParams") {
        auto rt = getRuntime(msg.value("controllerName", ""));
        return rt->cloneParams();
      }
      if (cmd == "applyParams") {
        auto rt = getRuntime(msg.value("controllerName", ""));
        rt->setParams(msg.value("params", json::object()));
        return true;
      }
      if (cmd == "snapshot") {
        auto rt = getRuntime(msg.value("controllerName", ""));
        return rt->toSnapshot();
      }
      if (cmd == "applySnapshot") {
        auto rt = getRuntime(msg.value("controllerName", ""));
        rt->fromSnapshot(msg.value("snapshot", json::object()));
        return true;
      }
      if (cmd == "applySnapshotAll") {
        auto snap = msg.value("snapshot", json::object());
        for (auto &name : controllerNames) {
          getRuntime(name)->fromSnapshot(snap);
        }
        return true;
      }
      if (cmd == "listMetrics") {
        json out = json::array();
        for (auto &name : controllerNames) {
          auto rt = getRuntime(name);
          out.push_back(json{{"name", name},
                             {"online", true},
                             {"health", json{{"status", "ok"},
                                             {"since", nowEpochMs()},
                                             {"failures", 0}}},
                             {"metrics", rt->metrics()},
                             {"params", rt->cloneParams()}});
        }
        return out;
      }
      if (cmd == "ingestDocument") {
        json results = json::array();
        auto doc = msg.value("doc", json::object());
        for (auto &name : controllerNames) {
          results.push_back(getRuntime(name)->ingestDocument(doc));
        }
        return results;
      }
      if (cmd == "ingestDocumentTo") {
        auto rt = getRuntime(msg.value("controllerName", ""));
        return rt->ingestDocument(msg.value("doc", json::object()));
      }
      if (cmd == "ingestDocumentToGroup") {
        json results = json::array();
        auto doc = msg.value("doc", json::object());
        for (auto &name : controllerNames) {
          results.push_back(getRuntime(name)->ingestDocument(doc));
        }
        return results;
      }
      if (cmd == "forgetMemes") {
        json results = json::array();
        auto criteria = msg.value("criteria", json::object());
        for (auto &name : controllerNames) {
          results.push_back(getRuntime(name)->forgetMemes(criteria));
        }
        return results;
      }
      if (cmd == "runtimeCall") {
        auto rt = getRuntime(msg.value("controllerName", ""));
        std::string method = msg.value("method", "");
        auto args = msg.value("args", json::array());
        if (method == "onlineLookup") {
          auto a0 =
              (args.is_array() && args.size() > 0) ? args[0] : json::object();
          auto a1 =
              (args.is_array() && args.size() > 1) ? args[1] : json::object();
          return rt->onlineLookup(a0, a1);
        }
        if (method == "collectRobotsDocuments") {
          json out = json::array();
          auto a0 =
              (args.is_array() && args.size() > 0) ? args[0] : json::object();
          for (auto &d : rt->collectRobotsDocuments(a0)) {
            out.push_back(json{{"id", d.id},
                               {"text", d.text},
                               {"tokens", d.tokens},
                               {"source", d.source},
                               {"file", d.file}});
          }
          return out;
        }
        if (method == "listRobotsFiles")
          return rt->listRobotsFiles();
        if (method == "exportGraphToFile") {
          auto a0 =
              (args.is_array() && args.size() > 0) ? args[0] : json::object();
          auto p = rt->exportGraphToFile(a0);
          return json{{"path", p.string()}};
        }
        if (method == "getSearchConfig")
          return rt->getSearchConfig();
        if (method == "setSearchConfig") {
          auto a0 =
              (args.is_array() && args.size() > 0) ? args[0] : json::object();
          return rt->setSearchConfig(a0);
        }
        if (method == "learnFromDialog") {
          if (args.is_array() && args.size() == 1 && args[0].is_object()) {
            auto obj = args[0];
            return rt->learnFromDialog(obj.value("payload", json::object()),
                                       obj.value("result", json::object()));
          }
          auto a0 =
              (args.is_array() && args.size() > 0) ? args[0] : json::object();
          auto a1 =
              (args.is_array() && args.size() > 1) ? args[1] : json::object();
          return rt->learnFromDialog(a0, a1);
        }
        throw std::runtime_error("runtimeCall not allowed: " + method);
      }
      if (cmd == "memebarrierStart") {
        double th =
            msg.value("maliciousThreshold",
                      phoenix::cfg<double>("memebarrier.maliciousThreshold"));
        if (!memeBarrier) {
          memeBarrier = std::make_shared<MemeBarrier>(
              getRuntime(controllerNames.front()));
        }
        memeBarrier->setThreshold(th);
        memeBarrier->start();
        auto stats = memeBarrier->getStats();
        return json{{"running", true},
                    {"threshold", memeBarrier->threshold()},
                    {"stats", json{{"scans", stats.scans},
                                   {"isolated", stats.isolated},
                                   {"lastScanTime", stats.lastScanTime},
                                   {"lastIsolated", stats.lastIsolated},
                                   {"evaluated", stats.evaluated},
                                   {"candidates", stats.candidates},
                                   {"avgScore", stats.avgScore},
                                   {"maxScore", stats.maxScore},
                                   {"p90Score", stats.p90Score},
                                   {"lastThreshold", stats.lastThreshold},
                                   {"avgDegree", stats.avgDegree},
                                   {"stdDegree", stats.stdDegree},
                                   {"avgOutDegree", stats.avgOutDegree},
                                   {"stdOutDegree", stats.stdOutDegree},
                                   {"scoreHistogram", stats.scoreHistogram}}}};
      }
      if (cmd == "memebarrierStop") {
        if (memeBarrier)
          memeBarrier->stop();
        return json{{"running", false}};
      }
      if (cmd == "memebarrierStats") {
        if (!memeBarrier)
          return json{{"running", false},
                      {"threshold", getModelDefaults().maliciousThreshold},
                      {"stats", nullptr}};
        auto stats = memeBarrier->getStats();
        return json{{"running", memeBarrier->running()},
                    {"threshold", memeBarrier->threshold()},
                    {"stats", json{{"scans", stats.scans},
                                   {"isolated", stats.isolated},
                                   {"lastScanTime", stats.lastScanTime},
                                   {"lastIsolated", stats.lastIsolated},
                                   {"evaluated", stats.evaluated},
                                   {"candidates", stats.candidates},
                                   {"avgScore", stats.avgScore},
                                   {"maxScore", stats.maxScore},
                                   {"p90Score", stats.p90Score},
                                   {"lastThreshold", stats.lastThreshold},
                                   {"avgDegree", stats.avgDegree},
                                   {"stdDegree", stats.stdDegree},
                                   {"avgOutDegree", stats.avgOutDegree},
                                   {"stdOutDegree", stats.stdOutDegree},
                                   {"scoreHistogram", stats.scoreHistogram}}}};
      }
      if (cmd == "memebarrierSetThreshold") {
        double th =
            msg.value("maliciousThreshold",
                      phoenix::cfg<double>("memebarrier.maliciousThreshold"));
        if (!memeBarrier) {
          memeBarrier = std::make_shared<MemeBarrier>(
              getRuntime(controllerNames.front()));
        }
        memeBarrier->setThreshold(th);
        return json{{"threshold", memeBarrier->threshold()}};
      }
      throw std::runtime_error("unknown-cmd:" + cmd);
    });

    reply["id"] = id;
    reply["shmProtocolVersion"] = kSharedMemoryProtocolVersion;
    reply["transport"] = "socket-msgpack-shm2";
    reply["groupId"] = groupId;
    auto responseBytes = json::to_msgpack(reply);
    if (!sendPacket(client, responseBytes))
      break;
  }

  closeSocket(client);
  closeSocket(serverSock);
  return 0;
}

static bool spawnFrontendProcess() {
  int spawnDepth =
      std::max(0, phoenix::resolveConfig<int>("main.frontendSpawnDepth", 0, "AI_FRONTEND_SPAWN_DEPTH"));
  if (spawnDepth >= 2)
    return false;
  std::string nextDepth = std::to_string(spawnDepth + 1);
  std::string projectRoot = getProjectRoot().string();
#ifdef _WIN32
  char oldSubBuf[64] = {0};
  char oldDepthBuf[64] = {0};
  char oldFrontendHttpLogBuf[64] = {0};
  DWORD oldSubLen = GetEnvironmentVariableA("AI_FRONTEND_SUBPROCESS", oldSubBuf,
                                            (DWORD)sizeof(oldSubBuf));
  DWORD oldDepthLen = GetEnvironmentVariableA(
      "AI_FRONTEND_SPAWN_DEPTH", oldDepthBuf, (DWORD)sizeof(oldDepthBuf));
  DWORD oldFrontendHttpLogLen =
      GetEnvironmentVariableA("FRONTEND_HTTP_LOG", oldFrontendHttpLogBuf,
                              (DWORD)sizeof(oldFrontendHttpLogBuf));
  bool hadSub = oldSubLen > 0 && oldSubLen < sizeof(oldSubBuf);
  bool hadDepth = oldDepthLen > 0 && oldDepthLen < sizeof(oldDepthBuf);
  bool hadFrontendHttpLog =
      oldFrontendHttpLogLen > 0 &&
      oldFrontendHttpLogLen < sizeof(oldFrontendHttpLogBuf);

  SetEnvironmentVariableA("AI_FRONTEND_SUBPROCESS", "1");
  SetEnvironmentVariableA("AI_FRONTEND_SPAWN_DEPTH", nextDepth.c_str());
  STARTUPINFOA si{};
  PROCESS_INFORMATION pi{};
  si.cb = sizeof(si);
  std::string cmdLine = std::string("\"") + g_selfPath + "\" --frontend-only=1";
  BOOL ok = CreateProcessA(nullptr, cmdLine.data(), nullptr, nullptr, FALSE,
                           CREATE_NEW_PROCESS_GROUP, nullptr,
                           projectRoot.empty() ? nullptr : projectRoot.c_str(),
                           &si, &pi);

  if (hadSub)
    SetEnvironmentVariableA("AI_FRONTEND_SUBPROCESS", oldSubBuf);
  else
    SetEnvironmentVariableA("AI_FRONTEND_SUBPROCESS", nullptr);
  if (hadDepth)
    SetEnvironmentVariableA("AI_FRONTEND_SPAWN_DEPTH", oldDepthBuf);
  else
    SetEnvironmentVariableA("AI_FRONTEND_SPAWN_DEPTH", nullptr);
  if (hadFrontendHttpLog)
    SetEnvironmentVariableA("FRONTEND_HTTP_LOG", oldFrontendHttpLogBuf);
  else
    SetEnvironmentVariableA("FRONTEND_HTTP_LOG", nullptr);

  if (!ok)
    return false;
  {
    std::lock_guard<std::mutex> lock(gFrontendProcMu);
    if (gFrontendProcessHandle) {
      DWORD waitCode = WaitForSingleObject(gFrontendProcessHandle, 0);
      if (waitCode == WAIT_TIMEOUT) {
        TerminateProcess(gFrontendProcessHandle, 0);
        WaitForSingleObject(gFrontendProcessHandle, 1500);
      }
      CloseHandle(gFrontendProcessHandle);
      gFrontendProcessHandle = nullptr;
      gFrontendProcessId = 0;
    }
    gFrontendProcessHandle = pi.hProcess;
    gFrontendProcessId = pi.dwProcessId;
    ensureFrontendJobObjectLocked();
    if (gFrontendJobHandle)
      AssignProcessToJobObject(gFrontendJobHandle, gFrontendProcessHandle);
  }
  CloseHandle(pi.hThread);
  return true;
#else
  setenv("AI_FRONTEND_SUBPROCESS", "1", 1);
  setenv("AI_FRONTEND_SPAWN_DEPTH", nextDepth.c_str(), 1);
  pid_t pid = fork();
  if (pid == 0) {
    if (!projectRoot.empty()) {
      std::error_code ec;
      fs::current_path(projectRoot, ec);
    }
    execl(g_selfPath.c_str(), g_selfPath.c_str(), "--frontend-only=1", nullptr);
    _exit(1);
  }
  if (pid > 0) {
    std::lock_guard<std::mutex> lock(gFrontendProcMu);
    gFrontendProcessPid = pid;
    return true;
  }
  return false;
#endif
}

// main 为程序主入口，负责初始化与模式分派。
// 调用方式：进程启动后由操作系统直接调用。
// 实现思路：加载配置、构建存储与池化组件后启动网关。
// 注意事项：命令行参数可改变进程运行模式。
int main(int argc, char **argv) {
  std::cout << "AI Controller starting..." << std::endl;
  g_selfPath = fs::absolute(fs::path(argv[0])).string();
  std::atexit([]() {
    appendFatalLog("atexit handler invoked (process exiting via normal path)");
  });
  std::atexit(cleanupFrontendProcess);
  std::atexit(cleanupPidFile);
#ifdef _WIN32
  SetConsoleCtrlHandler(onConsoleControl, TRUE);
  SetUnhandledExceptionFilter(onUnhandledException);
#endif
  installFatalHooks(fs::path("runtime_store"));
  Config config;
  std::map<std::string, std::string> args;
  try {
    args = parseArgs(argc, argv);

    int forceExitMs = [&] {
      auto it = args.find("sigint-force-exit-ms");
      if (it != args.end() && !it->second.empty()) {
        try {
          return std::stoi(it->second);
        } catch (...) {
          return 12000;
        }
      }
      return phoenix::resolveConfig<int>("main.sigintForceExitMs", 12000, "AI_SIGINT_FORCE_EXIT_MS");
    }();
    forceExitMs = std::max(1000, std::min(forceExitMs, 600000));
#ifdef _WIN32
    gCtrlForceExitMs.store(static_cast<DWORD>(forceExitMs),
                           std::memory_order_relaxed);
    // Disable watchdog if running as externally managed subprocess (e.g., benchmark)
    bool disableWatchdog = phoenix::resolveConfig<bool>("main.disableWatchdog", false, "AI_DISABLE_WATCHDOG");
    if (args.count("disable-watchdog") != 0 || disableWatchdog) {
      gDisableWatchdog = true;
      std::cout << "[startup] watchdog disabled (externally managed mode)" << std::endl;
    }
#endif

    applyFrontendEnvFromArgs(args);

    bool frontendOnlyRequested =
        args.count("frontend-only") != 0 ||
        phoenix::resolveConfig<bool>("main.frontendSubprocess", false, "AI_FRONTEND_SUBPROCESS");
    if (frontendOnlyRequested) {
      try {
        setupFrontendServer();
        drogon::app().run();
        return 0;
      } catch (const std::exception &e) {
        std::cerr << "frontend-only exception: " << e.what() << std::endl;
        return 1;
      } catch (...) {
        std::cerr << "frontend-only unknown exception" << std::endl;
        return 1;
      }
    }
    fs::path pidPath = fs::path(
        phoenix::resolveConfig<std::string>(
            "main.pidFile",
            (fs::path("runtime_store") / "phoenix_main.pid").string(),
            "AI_PID_FILE"));
    if (args.count("pid-file"))
      pidPath = fs::path(args["pid-file"]);
    std::string pidError;
    if (!acquirePidFile(pidPath, pidError)) {
      std::cerr << "startup blocked: " << pidError << std::endl;
      return 1;
    }
    std::cout << "pid file locked: " << pidPath.string() << std::endl;
    initProcessMemoryReservation();
    config = loadConfig(argc, argv);
    applyFrontendEnvFromConfig(config);
    installFatalHooks(config.baseDir);
#ifdef _WIN32
    startMemoryTelemetry(config.baseDir);
#endif
  } catch (const std::exception &e) {
    std::cerr << "startup exception: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "startup unknown exception" << std::endl;
    return 1;
  }
  std::cout << "Configuration loaded. Starting in "
            << (args.count("group-worker") ? "group worker" : "gateway")
            << " mode." << std::endl;
  std::cout << "startup-flags usingOllama="
            << (config.usingOllama ? "true" : "false")
            << " httpLog=" << (config.httpLog ? "true" : "false")
            << " port=" << config.portGateway
            << " studyPort=" << config.portStudy << std::endl;
#ifdef _WIN32
  if (config.httpLog)
    SetEnvironmentVariableA("AI_HTTP_LOG", "true");
#else
  if (config.httpLog)
    setenv("AI_HTTP_LOG", "true", 1);
#endif
  try {
    auto cwd = fs::current_path();
    auto root = getProjectRoot();
    std::string diag = std::string("startup-paths cwd=") + cwd.string() +
                       " projectRoot=" + root.string() +
                       " robotsDir=" + config.robotsDir.string();
    std::cout << diag << std::endl;
    appendFatalLog(diag);
  } catch (...) {
  }
  if (args.count("group-worker")) {
    return runGroupWorker(config);
  }
  ensureDir(config.baseDir);
  ensureDir(config.snapshotDir);
  ensureDir(config.lmdbRoot);
  ensureDir(config.robotsDir);
  if (!config.dbPath.empty()) {
    ensureDir(config.dbPath.parent_path());
  }
  std::cout << "Directories ensured. Initializing storage..." << std::endl;
  std::shared_ptr<KeyValueStore> kvmStore;
  std::shared_ptr<KeyValueStore> memeStore;
  std::shared_ptr<KeyValueStore> sessionStore;

  std::shared_ptr<Database079> db079;
#ifdef HAVE_SQLITE
  db079 = std::make_shared<Database079>(config.dbPath, config.lmdbRoot,
                                        config.redisUrl, config.redisCacheDb,
                                        config.redisCachePrefix);
  if (db079->open()) {
    kvmStore = db079->createStore("kvm");
    memeStore = db079->createStore("meme_graph");
    sessionStore = db079->createStore("session");
  }
#endif
  std::cout << "Storage initialized. Starting gateway server..." << std::endl;
#ifdef HAVE_LMDB
  if (!Database079::isSqliteStore(kvmStore)) {
    auto lmdbK = std::make_shared<LmdbStore>("kvm", config.lmdbRoot,
                                             config.lmdbMapSizeBytes);
    auto lmdbM = std::make_shared<LmdbStore>("meme_graph", config.lmdbRoot,
                                             config.lmdbMapSizeBytes);
    auto lmdbS = std::make_shared<LmdbStore>("session", config.lmdbRoot,
                                             config.lmdbMapSizeBytes);
    if (lmdbK->ok())
      kvmStore = lmdbK;
    if (lmdbM->ok())
      memeStore = lmdbM;
    if (lmdbS->ok())
      sessionStore = lmdbS;
  }
#endif
  std::cout << "Gateway server storage setup complete. Starting application..."
            << std::endl;
  if (!kvmStore) {
    kvmStore = std::make_shared<JsonFileStore>("kvm", config.lmdbRoot);
    memeStore = std::make_shared<JsonFileStore>("meme_graph", config.lmdbRoot);
    sessionStore = std::make_shared<JsonFileStore>("session", config.lmdbRoot);
  }

  // Dedicated world-model storage: shared with frontend_server by default.
  std::shared_ptr<Database079> worldModelDb;
#ifdef HAVE_SQLITE
  {
    fs::path worldModelDbPath = fs::path(phoenix::resolveConfig<std::string>(
        "world_model.storage.db",
        std::string("./runtime_store/frontend_world_model.sqlite"),
        "FRONTEND_WORLDMODEL_DB"));
    fs::path worldModelLegacyDir = fs::path(phoenix::resolveConfig<std::string>(
        "world_model.storage.legacyDir",
        std::string("./lmdb/frontend_world_model"),
        "FRONTEND_WORLDMODEL_LEGACY_DIR"));
    std::string worldModelRedisUrl = phoenix::resolveConfig<std::string>(
        "world_model.storage.redisUrl",
        std::string("redis://127.0.0.1:6379"),
        "FRONTEND_WORLDMODEL_REDIS_URL", "REDIS_URL");
    int worldModelRedisDb = phoenix::resolveConfig<int>(
        "world_model.storage.redisDb", 7, "FRONTEND_WORLDMODEL_REDIS_DB");
    std::string worldModelRedisPrefix = phoenix::resolveConfig<std::string>(
        "world_model.storage.redisPrefix",
        std::string("phoenix:frontend:world"),
        "FRONTEND_WORLDMODEL_REDIS_PREFIX");
    if (!worldModelDbPath.parent_path().empty())
      fs::create_directories(worldModelDbPath.parent_path());
    if (!worldModelLegacyDir.empty())
      fs::create_directories(worldModelLegacyDir);
    worldModelDb = std::make_shared<Database079>(
        worldModelDbPath, worldModelLegacyDir, worldModelRedisUrl,
        worldModelRedisDb, worldModelRedisPrefix);
    worldModelDb->open();
  }
#endif

  // Initialize advanced systems
  std::cout << "Initializing emotion system..." << std::endl;
  phoenix::emotion::EmotionSystem::Config emotionConfig;
  emotionConfig.storageBackend = "sqlite";
  emotionConfig.storagePath = (config.lmdbRoot / "emotion_states.db").string();
  emotionConfig.enabled = true;
  emotionConfig.emotionDecayRate = 0.95f;
  emotionConfig.emotionInfluence = 0.3f;
  emotionConfig.enableRuntimeFineTuning = true;
  auto emotionSystem = std::make_shared<phoenix::emotion::EmotionSystem>(emotionConfig);

  std::cout << "Initializing LLM emotion weight adjuster..." << std::endl;
  phoenix::emotion::LlamaCppEmotionWeightAdjuster::Config adjusterConfig;
  adjusterConfig.maxAdjustment = 0.1f;
  auto emotionWeightAdjuster = std::make_shared<phoenix::emotion::LlamaCppEmotionWeightAdjuster>(adjusterConfig);
  phoenix::emotion::LlamaCppEmotionHooks::setWeightAdjuster(emotionWeightAdjuster);

  std::cout << "Initializing plugin system..." << std::endl;
  phoenix::plugin::PluginManager::Config pluginConfig;
  pluginConfig.pluginDirectory = (config.baseDir / "plugins").string();
  pluginConfig.autoLoad = true;
  pluginConfig.hotReload = false;
  ensureDir(fs::path(pluginConfig.pluginDirectory));
  auto pluginManager = std::make_shared<phoenix::plugin::PluginManager>(pluginConfig);

  std::cout << "Initializing modern context system..." << std::endl;
  phoenix::context::ContextWindowConfig contextConfig;
  contextConfig.maxTokens =
      phoenix::cfg<size_t>("context.maxTokens");
  contextConfig.reservedSystemTokens =
      phoenix::cfg<size_t>("context.reservedSystemTokens");
  contextConfig.importanceThreshold =
      phoenix::cfg<float>("context.importanceThreshold");
  contextConfig.similarityThreshold =
      phoenix::cfg<float>("context.similarityThreshold");
  contextConfig.semanticChunkSize =
      phoenix::cfg<size_t>("context.semanticChunkSize");
  contextConfig.strategy = phoenix::context::ContextStrategy::HYBRID;
  contextConfig.enableSemanticSearch =
      phoenix::cfg<bool>("context.enableSemanticSearch");
  contextConfig.enableAttentionSink =
      phoenix::cfg<size_t>("context.attentionSink.sinkTokens") > 0;
  contextConfig.enableHierarchical =
      phoenix::cfg<bool>("context.enableHierarchical");
  auto modernContextManager = std::make_shared<phoenix::context::ModernContextManager>(contextConfig);

  std::cout << "Advanced systems initialized successfully." << std::endl;

  std::shared_ptr<ControllerPoolBase> pool;
  std::shared_ptr<IRedisSynchronizer> redisSync;
  std::shared_ptr<RotationManager> rotation;
  std::shared_ptr<IStudyEngine> study;
  std::cout << "Storage setup complete. Initializing controller pool..."
            << std::endl;
  if (config.groupProc) {
    pool = std::make_shared<GroupProcessPool>(kvmStore, memeStore, sessionStore,
                                              config);
    auto redisFactory =
        module_mount::ModuleRegistry::instance().redisSynchronizerFactory();
    if (redisFactory)
      redisSync = redisFactory(pool, config.redisUrl, config.redisChannel);
    else
      redisSync = std::make_shared<RedisSynchronizer>(pool, config.redisUrl,
                                                      config.redisChannel);
    std::cout << "[Bootstrap] group-proc enabled: one process per group"
              << std::endl;
  } else {
    pool = std::make_shared<ControllerPool>(kvmStore, memeStore, sessionStore,
                                            config);
    rotation = std::make_shared<RotationManager>(pool);
    rotation->start();
    auto redisFactory =
        module_mount::ModuleRegistry::instance().redisSynchronizerFactory();
    if (redisFactory)
      redisSync = redisFactory(pool, config.redisUrl, config.redisChannel);
    else
      redisSync = std::make_shared<RedisSynchronizer>(pool, config.redisUrl,
                                                      config.redisChannel);
    try {
      redisSync->start();
    } catch (...) {
    }
    auto studyFactory =
        module_mount::ModuleRegistry::instance().studyEngineFactory();
    if (studyFactory)
      study = studyFactory(pool, redisSync);
    else
      study = std::make_shared<StudyEngine>(pool, redisSync);
    study->start();
  }

  std::shared_ptr<ISnapshotManager> snapshots;
  auto snapshotFactory =
      module_mount::ModuleRegistry::instance().snapshotManagerFactory();
  if (snapshotFactory)
    snapshots =
        snapshotFactory(pool->getActive()->runtime(), config.snapshotDir);
  else
    snapshots = std::make_shared<SnapshotManager>(pool->getActive()->runtime(),
                                                  config.snapshotDir);

  bool restoredFromSnapshot = false;
  std::cout
      << "Controller pool initialized. Attempting to restore from snapshot..."
      << std::endl;
  try {
    auto list = snapshots->list();
    std::sort(list.begin(), list.end(), std::greater<>());
    if (!list.empty()) {
      auto snap = snapshots->restore(list[0]);
      restoredFromSnapshot = true;
      if (config.groupProc) {
        if (auto gp = dynamic_cast<GroupProcessPool *>(pool.get())) {
          try {
            gp->applySnapshotAll(snap);
          } catch (...) {
          }
        }
      }
      if (config.syncStandbyOnBoot) {
        try {
          pool->getStandby()->applySnapshot(snap);
        } catch (...) {
        }
        try {
          pool->getValidation()->applySnapshot(snap);
        } catch (...) {
        }
      }
    }
  } catch (...) {
  }

  if (config.robotsAutoload && config.robotsWarmupLimit > 0 &&
      !restoredFromSnapshot) {
    std::cout << "Robots autoload enabled. Warming up robots..." << std::endl;
    try {
      auto docs = pool->getActive()->runtime()->collectRobotsDocuments(
          json{{"limit", config.robotsWarmupLimit},
               {"shuffle", config.robotsWarmupShuffle}});
      auto groups = pool->listGroupIds();
      for (auto &doc : docs) {
        std::string key =
            (doc.source.empty() ? (doc.file.empty() ? doc.id : doc.file)
                                : doc.source);
        key += "|" + doc.text.substr(0, 256);
        int idx =
            groups.empty() ? 0 : (int)(hashStrSimple(key) % groups.size());
        std::string target =
            groups.empty()
                ? "G1"
                : (groups[idx].empty() ? (groups.empty() ? "G1" : groups[0])
                                       : groups[idx]);
        std::string source =
            doc.source.empty()
                ? (doc.file.empty() ? "robots:unknown" : ("robots:" + doc.file))
                : doc.source;
        pool->ingestDocumentToGroup(target, json{{"text", doc.text},
                                                 {"tokens", doc.tokens},
                                                 {"source", source},
                                                 {"file", doc.file},
                                                 {"id", doc.id}});
      }
    } catch (...) {
    }
  }
  std::cout << "Snapshot restoration complete. Starting gateway server..."
            << std::endl;
  if (config.testsAutoload) {
    std::thread([pool, testsDir = config.testsDir]() {
      try {
        if (!fs::exists(testsDir))
          return;

        std::vector<std::string> files;
        for (auto &e : fs::directory_iterator(testsDir)) {
          if (!e.is_regular_file())
            continue;
          auto ext = e.path().extension().string();
          std::string lowerExt;
          lowerExt.reserve(ext.size());
          for (char c : ext)
            lowerExt.push_back(
                static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
          if (lowerExt == ".txt")
            files.push_back(e.path().filename().string());
        }

        std::sort(files.begin(), files.end());
        const size_t maxFiles = 200;
        const size_t maxFileBytes = 2ull * 1024ull * 1024ull;
        if (files.size() > maxFiles)
          files.resize(maxFiles);

        auto groups = pool->listGroupIds();
        for (auto &f : files) {
          fs::path p = testsDir / f;
          std::error_code ec;
          auto sz = fs::file_size(p, ec);
          if (!ec && sz > maxFileBytes)
            continue;

          std::string text = readTextFile(p);
          if (text.empty())
            continue;
          std::string key = "tests:" + f;

          if (groups.empty()) {
            pool->ingestDocument(json{{"text", text}, {"source", key}});
            continue;
          }

          int idx = (int)(hashStrSimple(key) % groups.size());
          std::string target = groups[idx].empty() ? groups[0] : groups[idx];
          pool->ingestDocumentToGroup(target,
                                      json{{"text", text}, {"source", key}});
        }
      } catch (...) {
      }
    }).detach();
  }
  std::cout << "Gateway server starting..." << std::endl;
  std::shared_ptr<IGatewayServer> gateway;
  auto gatewayFactory =
      module_mount::ModuleRegistry::instance().gatewayServerFactory();
  if (gatewayFactory)
    gateway = gatewayFactory(pool, snapshots, redisSync, rotation, study, db079,
                             config);
  else
    gateway = std::make_shared<GatewayServer>(pool, snapshots, redisSync,
                                              rotation, study, db079, config);
  if (worldModelDb) {
    auto concreteGateway = std::dynamic_pointer_cast<GatewayServer>(gateway);
    if (concreteGateway)
      concreteGateway->setWorldModelDatabase(worldModelDb);
  }
  {
    bool frontendEnabled =
        phoenix::resolveConfig<bool>("main.frontendEnabled", true, "AI_FRONTEND_ENABLED");
    bool isFrontendSubprocess =
        phoenix::resolveConfig<bool>("main.frontendSubprocess", false, "AI_FRONTEND_SUBPROCESS");
    if (frontendEnabled && !isFrontendSubprocess) {
      if (!spawnFrontendProcess()) {
        std::cerr << "frontend subprocess spawn failed" << std::endl;
      }
    }
  }
  if (config.learningWarmup) {
    std::thread([gateway]() {
      try {
        gateway->warmupLearning();
      } catch (...) {
      }
    }).detach();
  }
  if (config.transformerBootstrap) {
    int bootstrapDocs = std::max(16, config.transformerBootstrapDocs);
    std::thread([gateway, bootstrapDocs]() {
      try {
        gateway->bootstrapTransformerFromCorpus(bootstrapDocs);
      } catch (...) {
      }
    }).detach();
  }
  std::cout << "Initialization complete. Listening for requests..."
            << std::endl;
  try {
    gateway->listen();
    appendFatalLog("gateway listen returned normally (unexpected clean exit)");
    std::cerr << "gateway listen returned normally (unexpected clean exit)" << std::endl;
    return 0;
  } catch (const std::exception &e) {
    appendFatalLog(std::string("gateway listen exception: ") + e.what());
    std::cerr << "gateway listen exception: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    appendFatalLog("gateway listen unknown exception");
    std::cerr << "gateway listen unknown exception" << std::endl;
    return 1;
  }
}
