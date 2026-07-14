// Auto-generated from main.cpp
// Segment: section_before_PersonaForestAverager
// Original lines: 10876-11051


json ProxyRuntime::processInput(const json &payload) {
  return pool_->respondByName(controllerName_, payload);
}

json ProxyRuntime::cloneParams() const {
  return pool_->rpc(
      pool_->groupForController(controllerName_),
      json{{"cmd", "cloneParams"}, {"controllerName", controllerName_}});
}

void ProxyRuntime::setParams(const json &patch) {
  pool_->rpc(pool_->groupForController(controllerName_),
             json{{"cmd", "applyParams"},
                  {"controllerName", controllerName_},
                  {"params", patch}});
}

json ProxyRuntime::toSnapshot() {
  return pool_->rpc(
      pool_->groupForController(controllerName_),
      json{{"cmd", "snapshot"}, {"controllerName", controllerName_}});
}

void ProxyRuntime::fromSnapshot(const json &snapshot) {
  pool_->rpc(pool_->groupForController(controllerName_),
             json{{"cmd", "applySnapshot"},
                  {"controllerName", controllerName_},
                  {"snapshot", snapshot}});
}

json ProxyRuntime::ingestDocument(const json &doc) {
  return pool_->rpc(pool_->groupForController(controllerName_),
                    json{{"cmd", "ingestDocumentTo"},
                         {"controllerName", controllerName_},
                         {"doc", doc}});
}

json ProxyRuntime::forgetMemes(const json &payload) {
  return pool_->forgetMemes(payload);
}

json ProxyRuntime::onlineLookup(const std::string &query, const json &options) {
  return pool_->runtimeCall(controllerName_, "onlineLookup",
                            json::array({query, options}));
}

json ProxyRuntime::onlineLookup(const json &input, const json &options) {
  return pool_->runtimeCall(controllerName_, "onlineLookup",
                            json::array({input, options}));
}

json ProxyRuntime::getSearchConfig() const {
  return pool_->runtimeCall(controllerName_, "getSearchConfig", json::array());
}

json ProxyRuntime::setSearchConfig(const json &patch) {
  return pool_->runtimeCall(controllerName_, "setSearchConfig",
                            json::array({patch}));
}

std::vector<RobotsDocument>
ProxyRuntime::collectRobotsDocuments(const json &options) const {
  auto out = pool_->runtimeCall(controllerName_, "collectRobotsDocuments",
                                json::array({options}));
  std::vector<RobotsDocument> docs;
  if (out.is_array()) {
    for (auto &item : out) {
      RobotsDocument d;
      d.id = item.value("id", "");
      d.text = item.value("text", "");
      d.tokens = item.value("tokens", std::vector<std::string>{});
      d.source = item.value("source", "");
      d.file = item.value("file", "");
      docs.push_back(std::move(d));
    }
  }
  return docs;
}

std::vector<std::string> ProxyRuntime::listRobotsFiles() const {
  auto out =
      pool_->runtimeCall(controllerName_, "listRobotsFiles", json::array());
  std::vector<std::string> files;
  if (out.is_array()) {
    for (auto &v : out)
      if (v.is_string())
        files.push_back(v.get<std::string>());
  }
  return files;
}

fs::path ProxyRuntime::exportGraphToFile(const json &options) {
  auto out = pool_->runtimeCall(controllerName_, "exportGraphToFile",
                                json::array({options}));
  if (out.is_string())
    return fs::path(out.get<std::string>());
  if (out.is_object() && out.contains("path"))
    return fs::path(out["path"].get<std::string>());
  return {};
}

json ProxyRuntime::learnFromDialog(const json &payload, const json &result) {
  return pool_->runtimeCall(
      controllerName_, "learnFromDialog",
      json::array({json{{"payload", payload}, {"result", result}}}));
}

std::unordered_map<std::string, double>
ProxyRuntime::mapWordsToMemes(const std::vector<std::string> &words) {
  auto out = pool_->runtimeCall(controllerName_, "mapWordsToMemes",
                                json::array({words}));
  std::unordered_map<std::string, double> res;
  if (out.is_object()) {
    for (auto it = out.begin(); it != out.end(); ++it) {
      if (it.value().is_number())
        res[it.key()] = it.value().get<double>();
    }
  }
  return res;
}

std::vector<std::string> ProxyRuntime::getMemeWords(const std::string &memeId) {
  auto out = pool_->runtimeCall(controllerName_, "getMemeWords",
                                json::array({memeId}));
  std::vector<std::string> res;
  if (out.is_array()) {
    for (auto &v : out)
      if (v.is_string())
        res.push_back(v.get<std::string>());
  }
  return res;
}

void ProxyRuntime::graphLink(const std::string &fromId, const std::string &toId,
                             double weight, int direction) {
  pool_->runtimeCall(controllerName_, "graphLink",
                     json::array({fromId, toId, weight, direction}));
}

const Config &ProxyRuntime::config() const { return pool_->config(); }

json ProxyController::respond(const json &payload) {
  return pool_->respondByName(name_, payload);
}

json ProxyController::metrics() const {
  return pool_->rpc(pool_->groupForController(name_),
                    json{{"cmd", "metrics"}, {"controllerName", name_}});
}

void ProxyController::applyParams(const json &p) {
  pool_->rpc(
      pool_->groupForController(name_),
      json{{"cmd", "applyParams"}, {"controllerName", name_}, {"params", p}});
}

json ProxyController::snapshot() const {
  return pool_->rpc(pool_->groupForController(name_),
                    json{{"cmd", "snapshot"}, {"controllerName", name_}});
}

void ProxyController::applySnapshot(const json &snap) {
  pool_->rpc(pool_->groupForController(name_), json{{"cmd", "applySnapshot"},
                                                    {"controllerName", name_},
                                                    {"snapshot", snap}});
}

// ------------------ Spark Arrays ------------------

// PersonaForestAverager 负责多候选回复的森林式聚合决策。
// 调用方式：接收层级候选后返回最终回复与解释信息。
// 实现思路：子空间随机采样 + 投票评分融合。
// 注意事项：特征权重与样本率会影响稳定性与多样性。
// 注意事项：候选过少时会自动降级为直接返回。
// 注意事项：该模块依赖输入候选质量。
