// Auto-generated from main.cpp
// Segment: class_Controller
// Original lines: 9912-9922

class Controller {
public:
  virtual ~Controller() = default;
  virtual json respond(const json &payload) = 0;
  virtual json metrics() const = 0;
  virtual void applyParams(const json &p) = 0;
  virtual json snapshot() const = 0;
  virtual void applySnapshot(const json &snap) = 0;
  virtual const std::string &name() const = 0;
  virtual std::shared_ptr<RuntimeBase> runtime() const = 0;
};
