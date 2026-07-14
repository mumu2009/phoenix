// Auto-generated from main.cpp
// Segment: class_ControllerPoolBase
// Original lines: 9882-9904

class Controller;

class ControllerPoolBase {
public:
  virtual ~ControllerPoolBase() = default;
  virtual std::shared_ptr<Controller> getActive() = 0;
  virtual std::shared_ptr<Controller> getStandby() = 0;
  virtual std::shared_ptr<Controller> getValidation() = 0;
  virtual std::shared_ptr<Controller> getByName(const std::string &name) = 0;
  virtual std::vector<std::string> listControllerNames() const = 0;
  virtual std::vector<std::string> listGroupIds() const = 0;
  virtual std::vector<std::string>
  listControllersInGroup(const std::string &gid) = 0;
  virtual std::vector<json> listMetrics() const = 0;
  virtual json ingestDocument(const json &doc) = 0;
  virtual json ingestDocumentToGroup(const std::string &gid,
                                     const json &doc) = 0;
  virtual json forgetMemes(const json &payload) = 0;
  virtual json onlineResearch(const json &queryOrTokens,
                              const json &options) = 0;
  virtual void hotSwap(const json &snapshot) = 0;
  virtual const Config &config() const = 0;
  virtual json respondByName(const std::string &controllerName,
                             const json &payload) = 0;
};
