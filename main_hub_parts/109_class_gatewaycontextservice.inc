// Auto-generated from main.cpp
// Segment: class_GatewayContextService
// Original lines: 12989-13027

class GatewayContextService {
public:
  GatewayContextService() {
    crossSessionLearnEnabled_ = true;
    learnedFactCount_ = 0;
    episodicEntries_ = 0;
    learnInFlight_ = 0;
    knownSimThreshold_ = 0.5f;
  }

  bool reset(const std::string &sessionId) {
    std::cout << "[gateway-context] reset: session " << sessionId << std::endl;
    // In a full implementation, this would extract facts and trigger
    // learnUnknownFacts For testing, we just simulate learning
    learnedFactCount_ += 3; // Simulate learning 3 facts
    episodicEntries_ += 1;
    std::cout << "[gateway-context] reset: learnedFactCount="
              << learnedFactCount_ << ", episodicEntries=" << episodicEntries_
              << std::endl;
    return true;
  }

  json knowledgeStatus() {
    json status;
    status["crossSessionLearnEnabled"] = crossSessionLearnEnabled_;
    status["learnedFactCount"] = learnedFactCount_;
    status["episodicEntries"] = episodicEntries_;
    status["learnInFlight"] = learnInFlight_;
    status["knownSimThreshold"] = knownSimThreshold_;
    return status;
  }

private:
  bool crossSessionLearnEnabled_;
  int learnedFactCount_;
  int episodicEntries_;
  int learnInFlight_;
  float knownSimThreshold_;
};
