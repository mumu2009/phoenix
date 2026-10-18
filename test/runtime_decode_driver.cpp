/* Header-only checks: pack budget stays 1x, chat causal text is the
   user line, and a self-kernel thought in graph context stays off it. */

#include "inference_unit_pipeline.hpp"

#include <iostream>
#include <string>

namespace {

int gFails = 0;

void check(bool cond, const char *name) {
  if (cond) {
    std::cout << "PASS " << name << "\n";
  } else {
    std::cerr << "FAIL " << name << "\n";
    ++gFails;
  }
}

}  // namespace

int main() {
  using phoenix::inference::causalBudgetNgram;
  using phoenix::inference::selectCausalPrefix;
  using json = nlohmann::json;

  check(causalBudgetNgram(false, 2) == 1, "pack-off-ngram2-budget-1");
  check(causalBudgetNgram(false, 3) == 1, "pack-off-ngram3-budget-1");
  check(causalBudgetNgram(false, 0) == 1, "pack-off-ngram0-budget-1");
  check(causalBudgetNgram(true, 2) == 2, "pack-on-ngram2-budget-2");
  check(causalBudgetNgram(true, 0) == 1, "pack-on-ngram0-is-1");
  check(causalBudgetNgram(true, 9) == 3, "pack-on-ngram-clamped-3");

  const int ctx = 4096;
  const int genReserve = 192;
  const int fac = causalBudgetNgram(false, 2);
  const int budget = std::max(256, (ctx - genReserve - 64) * fac);
  const int doubled = std::max(256, (ctx - genReserve - 64) * 2);
  check(fac == 1, "long-prompt-factor-1");
  check(budget == 3840, "long-prompt-budget-not-doubled");
  check(budget * 2 == doubled, "doubled-would-be-7680");

  const std::string user = "What is 2+2?";
  const std::string graph =
      std::string("[Memory] Benefit=1\n[self-kernel]\nI am self.\n") +
      "GNN outline node\n" + user;
  json chatOpts = {{"skipGraphContext", true},
                   {"pack_prompt", false},
                   {"ngram_merge", 2},
                   {"loop_mode", "unit"},
                   {"lowPriority", false}};
  const std::string prefix = selectCausalPrefix(chatOpts, graph);
  check(prefix.empty(), "chat-prefix-empty");
  const std::string causal = prefix.empty() ? user : prefix + "\n" + user;
  check(causal == user, "chat-causal-is-user");
  check(causal.find("[self-kernel]") == std::string::npos, "chat-no-self-kernel");
  check(causal.find("[Memory]") == std::string::npos, "chat-no-memory");
  check(causal.find("GNN outline") == std::string::npos, "chat-no-gnn");

  json missionOpts = {{"skipGraphContext", true},
                      {"lowPriority", true},
                      {"system_content", std::string("working brief")}};
  check(selectCausalPrefix(missionOpts, graph) == "working brief",
        "mission-keeps-document-head");

  json bareChat = json::object();
  check(selectCausalPrefix(bareChat, graph).empty(),
        "default-chat-drops-graph");

  if (gFails) {
    std::cerr << gFails << " failed\n";
    return 1;
  }
  std::cout << "ALL PASS\n";
  return 0;
}
