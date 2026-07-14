// Auto-generated from main.cpp
// Segment: section_before_GnnHintSummary
// Original lines: 14008-14136


  std::shared_ptr<RuntimeBase> activeRuntime() const {
    auto ctrl = pool_ ? pool_->getActive() : nullptr;
    return ctrl ? ctrl->runtime() : nullptr;
  }

  std::string buildGraphContext(const json &graphResult, const json &payload) {
    auto runtime = activeRuntime();
    if (!runtime)
      return "";
    std::vector<std::string> words;
    if (payload.contains("tokens") && payload["tokens"].is_array()) {
      for (auto &t : payload["tokens"]) {
        if (t.is_string())
          words.push_back(t.get<std::string>());
      }
    } else if (payload.contains("text") && payload["text"].is_string()) {
      words = tokenize(payload["text"].get<std::string>());
    }

    std::vector<std::pair<std::string, double>> memes;
    std::unordered_set<std::string> seedIds;
    if (graphResult.contains("seeds") && graphResult["seeds"].is_array()) {
      for (auto &s : graphResult["seeds"]) {
        if (s.is_array() && s.size() >= 1 && s[0].is_string())
          seedIds.insert(s[0].get<std::string>());
      }
    }
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
    if (memes.size() > 16)
      memes.resize(16);
    std::unordered_set<std::string> memeSet;
    for (auto &m : memes)
      memeSet.insert(m.first);

    std::ostringstream oss;
    oss << "ctx_v2";
    if (!words.empty()) {
      oss << "\ninput:";
      for (size_t i = 0; i < words.size() && i < 24; i++) {
        if (i)
          oss << " ";
        oss << words[i];
      }
    }
    if (!seedIds.empty()) {
      oss << "\nseeds:";
      int cnt = 0;
      for (auto &sid : seedIds) {
        if (cnt++)
          oss << ",";
        oss << sid;
        if (cnt >= 16)
          break;
      }
    }

    oss << "\nnodes:";
    int nodeCount = 0;
    auto rtState = std::dynamic_pointer_cast<RuntimeState>(runtime);
    for (auto &m : memes) {
      if (nodeCount++)
        oss << " ; ";
      auto ws = runtime->getMemeWords(m.first);
      std::string kws;
      int kwCount = 0;
      for (auto &w : ws) {
        if (w.empty())
          continue;
        if (kwCount++)
          kws += ",";
        kws += w;
        if (kwCount >= 4)
          break;
      }
      oss << m.first << "|act=" << std::fixed << std::setprecision(3)
          << m.second << "|deg="
          << (rtState ? rtState->graph().degreeOf(m.first) : (int)ws.size())
          << "|seed=" << (seedIds.count(m.first) ? 1 : 0) << "|kw=" << kws;
      if (nodeCount >= 12)
        break;
    }

    oss << "\nedges:";
    int edgeCount = 0;
    if (rtState) {
      for (auto &m : memes) {
        for (auto &n : rtState->graph().neighborsOf(m.first)) {
          if (!memeSet.count(n.first))
            continue;
          if (edgeCount++)
            oss << " ; ";
          oss << m.first << "->" << n.first << "|w=" << std::fixed
              << std::setprecision(2) << n.second.first
              << "|d=" << n.second.second;
          if (edgeCount >= 20)
            break;
        }
        if (edgeCount >= 20)
          break;
      }
    } else {
      for (size_t i = 0; i + 1 < memes.size() && edgeCount < 20; i++) {
        if (edgeCount++)
          oss << " ; ";
        oss << memes[i].first << "->" << memes[i + 1].first << "|w=0.35|d=0";
      }
    }

    oss << "\nsummary:n=" << memes.size() << " e=" << edgeCount;
    std::string out = oss.str();
    if (out.size() > 4096)
      out.resize(4096);
    return out;
  }

