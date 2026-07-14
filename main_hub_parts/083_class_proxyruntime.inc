// Auto-generated from main.cpp
// Segment: class_ProxyRuntime
// Original lines: 10101-10132

class ProxyRuntime : public RuntimeBase {
public:
  ProxyRuntime(GroupProcessPool *pool, std::string controllerName)
      : pool_(pool), controllerName_(std::move(controllerName)) {}

  json processInput(const json &payload) override;
  json cloneParams() const override;
  void setParams(const json &patch) override;
  json toSnapshot() override;
  void fromSnapshot(const json &snapshot) override;
  json ingestDocument(const json &doc) override;
  json forgetMemes(const json &payload) override;
  json onlineLookup(const std::string &query, const json &options) override;
  json onlineLookup(const json &input, const json &options) override;
  json getSearchConfig() const override;
  json setSearchConfig(const json &patch) override;
  std::vector<RobotsDocument>
  collectRobotsDocuments(const json &options) const override;
  std::vector<std::string> listRobotsFiles() const override;
  fs::path exportGraphToFile(const json &options) override;
  json learnFromDialog(const json &payload, const json &result) override;
  std::unordered_map<std::string, double>
  mapWordsToMemes(const std::vector<std::string> &words) override;
  std::vector<std::string> getMemeWords(const std::string &memeId) override;
  void graphLink(const std::string &fromId, const std::string &toId,
                 double weight, int direction) override;
  const Config &config() const override;

private:
  GroupProcessPool *pool_{nullptr};
  std::string controllerName_;
};
