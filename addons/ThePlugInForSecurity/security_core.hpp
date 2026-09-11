/* security_core.hpp - Phoenix-local MemeBarrier/GNN defensive observatory

   Statistics + identification + defense + in-graph inert existence probe.
   No construct / deploy / human-target / cross-process surfaces. */

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "../../plugin_system.hpp"
#include "../../util/module_resource.hpp"

namespace phoenix {
namespace secamp {

using json = nlohmann::json;

/* Fixed inert marker: not natural-language, not code, not an instruction. */
inline constexpr const char *kInertProbeId = "phoenix.probe.inert.v1";
inline constexpr const char *kInertProbeGlyph = "\xCE\xA6\xE2\x97\x8B\xE2\x96\xA3";

struct ProbeHopTrace {
  std::string from;
  std::string to;
  int hop{1};
};

struct ProbeState {
  bool allowInertProbe{false};
  bool probeEnabled{false};
  bool planted{false};
  std::string seedId;
  std::unordered_map<std::string, int> activation;
  std::vector<ProbeHopTrace> traces;
};

struct GraphEdge {
  int from{0};
  int to{0};
  double weight{1.0};
};

struct DiscreteGraph {
  std::vector<std::string> ids;
  std::vector<GraphEdge> edges;
  /* layer: "meme" or "word"; empty treated as meme */
  std::vector<std::string> layers;
  /* many-to-many: meme -> words or word -> memes */
  std::vector<std::vector<std::string>> mapped;
  std::unordered_set<std::string> isolated;
  std::unordered_set<std::string> anomalous;
};

struct NodeInfluence {
  std::string id;
  std::string layer;
  double ragImpact{0.0};
  double matrixImpact{0.0};
  double gradNorm{0.0};
  double hessTrace{0.0};
  double significance{0.0};
  int rankMost{0};
  int rankLeast{0};
  std::vector<std::string> mappedIds;
  std::vector<std::string> neighborIds;
  std::string impactScope;
};

struct InfluenceReport {
  bool ok{false};
  std::string error;
  int nodeCount{0};
  int embedDim{0};
  std::vector<NodeInfluence> nodes;
  std::vector<std::string> mostSignificant;
  std::vector<std::string> leastSignificant;
};

struct InspectDecision {
  bool blocked{false};
  bool observedOnly{false};
  std::string reason;
  std::vector<std::string> hits;
};

struct DefenseConfig {
  bool defenseEnabled{true};
  bool isolateHighImpact{true};
  bool researchObserve{false};
  bool allowResearchObserve{false};
  double highImpactQuantile{0.85};
  int alertCap{64};
};

struct AlertItem {
  std::int64_t ts{0};
  std::string kind;
  std::string detail;
  json extra = json::object();
};

InfluenceReport analyzeGraph(const DiscreteGraph &g, double alpha = 0.85,
                             int embedDim = 4);

/* F(a)=||P a||^2 on the walk resolvent. Used by host unit tests. */
bool ragEnergyGradHess(const DiscreteGraph &g, const std::vector<double> &a,
                       double *energy, std::vector<double> *grad,
                       std::vector<double> *hessDiag);

InspectDecision inspectTokens(const std::vector<std::string> &tokens,
                              const InfluenceReport &report,
                              const DiscreteGraph &g,
                              const DefenseConfig &cfg);

std::vector<std::string> tokenizeDefense(const std::string &text);

class SecurityObservatory {
public:
  static SecurityObservatory &instance();

  void resetForTests();
  void loadProcessFlags();

  DefenseConfig config() const;
  bool setDefenseEnabled(bool on);
  bool setIsolateHighImpact(bool on);
  /* researchObserve requires allowResearchObserve (env/config). Default off. */
  bool setResearchObserve(bool on, std::string *error);

  /* Inert probe: env PHOENIX_SECURITY_ALLOW_INERT_PROBE plus explicit enable.
     Plant + one hop stay in this process graph. Default off. */
  bool setProbeEnabled(bool on, std::string *error);
  bool plantInertProbe(const std::string &seedId, std::string *error);
  bool stepInertProbeOnce(std::string *error);
  ProbeState probeState() const;
  json probeJson() const;

  InfluenceReport ingest(const DiscreteGraph &g);
  InfluenceReport lastReport() const;
  DiscreteGraph lastGraph() const;

  InspectDecision inspectText(const std::string &text);
  void noteIsolation(const std::string &memeId, double score,
                     const std::string &reason);

  json statsJson() const;
  json identifyJson(const std::string &id) const;
  json identifyListJson() const;
  json alertsJson() const;
  json defenseJson() const;

  void registerCrudResources();

private:
  SecurityObservatory();
  void pushAlert(const std::string &kind, const std::string &detail,
                 const json &extra);
  InfluenceReport report_{};
  DiscreteGraph graph_{};
  DefenseConfig cfg_{};
  ProbeState probe_{};
  std::vector<AlertItem> alerts_;
  mutable std::mutex mu_;
};

void installSecurityModuleResources();

} // namespace secamp
} // namespace phoenix
