/* v10.2 study-layer config/state split.

   Root cause pinned by these tests: importState() restored every layer's
   `enabled` flag from the persisted state, so a state file written while the
   expert was disabled kept `study.expert.enabled: true` (config/phoenix.json)
   from ever taking effect.  The runtime therefore reported
   study.expert.enabled=false, every field-train opportunity logged
   `expert=skipped`, and weights.json stayed frozen for days.

   Contract now enforced:
     - config (or an explicit configureStudy payload) owns switches/paths;
     - the state file owns learned weights and counters only;
     - importState() must not resurrect an older enable switch.

   NOTE ON STYLE: every read below goes through `.value(key, fallback)` or
   `.contains(key)`.  The const `json::operator[]` asserts (nlohmann line 2135)
   when a key is absent, which turns a missing key into a process abort instead
   of a test failure - that is what made the first version of this file abort
   instead of reporting. */
#include <gtest/gtest.h>

#include "autonomy_stack.hpp"

using autonomy::CognitionAutonomyManager;
using nlohmann::json;

namespace {
/* status() wraps everything in {ok, result}; accept either shape so the
   helper does not depend on that detail. */
json statusSub(const json &st, const char *key) {
  if (st.contains(key)) return st.value(key, json::object());
  const json r = st.value("result", json::object());
  return r.value(key, json::object());
}
bool expertEnabled(const json &st) {
  const json study = statusSub(st, "study");
  const json expert = study.value("expert", json::object());
  return expert.value("enabled", false);
}
int expertParamCount(const json &st) {
  const json study = statusSub(st, "study");
  const json expert = study.value("expert", json::object());
  return expert.value("paramCount", -1);
}
}  // namespace

TEST(StudyConfigStateSplit, Sample1_ConfigureStudyEnablesExpert) {
  CognitionAutonomyManager mgr;
  mgr.configureStudy(json{{"expert", json{{"enabled", true}}}});
  const json st = mgr.status();
  EXPECT_TRUE(st.contains("result") || st.contains("study"));
  EXPECT_TRUE(expertEnabled(st));
}

TEST(StudyConfigStateSplit, Sample2_ImportStateCannotDisableConfiguredExpert) {
  CognitionAutonomyManager mgr;
  mgr.configureStudy(json{{"expert", json{{"enabled", true}}}});
  json state = mgr.exportState();
  ASSERT_TRUE(state.is_object());
  ASSERT_TRUE(state.contains("studyExpert"));
  /* A state file from the era when the layer was disabled. */
  state["studyExpert"]["enabled"] = false;
  if (state.contains("studyUnitLearn") && state["studyUnitLearn"].is_object())
    state["studyUnitLearn"]["enabled"] = false;
  mgr.importState(state);
  const json st = mgr.status();
  EXPECT_TRUE(expertEnabled(st)) << st.dump().substr(0, 400);
}

TEST(StudyConfigStateSplit, Sample3_StatusKeepsTheExpertShape) {
  /* status() must stay a pure read of the layer objects: the async trainer's
     counters are logged (`[field-train] expert=queued`) and verified by the
     long run (weights.json advancing while the loop ticks), not by adding a
     new sub-object here. */
  CognitionAutonomyManager mgr;
  const json st = mgr.status();
  const json study = statusSub(st, "study");
  ASSERT_TRUE(study.contains("expert")) << st.dump().substr(0, 400);
  const json expert = study.value("expert", json::object());
  EXPECT_TRUE(expert.contains("enabled"));
  EXPECT_TRUE(expert.contains("paramCount"));
}

TEST(StudyConfigStateSplit, Sample4_ExportStateRoundTripsWeights) {
  CognitionAutonomyManager mgr;
  mgr.configureStudy(json{{"expert", json{{"enabled", true}}}});
  const json before = mgr.status();
  mgr.importState(mgr.exportState());
  const json after = mgr.status();
  /* The restore must keep the weights (paramCount) and the config switch. */
  EXPECT_EQ(expertParamCount(before), expertParamCount(after));
  EXPECT_TRUE(expertEnabled(after));
}
