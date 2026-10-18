/* Same assertion as SelfMindTest.IterateThoughtStaysOutOfChatPrompt. */

#include "autonomy_stack.hpp"

#include <iostream>
#include <string>

int main() {
  const char *kCorpus = "doc/v8.3/self_mind_sample.json";
  autonomy::CognitionAutonomyManager mgr;
  mgr.configureSelfMind(nlohmann::json{{"enabled", true},
                                       {"reset", true},
                                       {"learningRate", 0.35},
                                       {"name", "self"},
                                       {"corpusPath", kCorpus}});
  mgr.observe(autonomy::buildCognitionAutonomySeedPayload("self-sess", "inner", 0.4),
              nlohmann::json::object());
  const auto iter = mgr.iterate(nlohmann::json{{"sessionId", "self-sess"},
                                               {"userPrompt", "ZZZ-QUESTION-TOKEN?"}},
                                nlohmann::json::object());
  if (!iter.value("ok", false)) {
    std::cerr << "FAIL iterate " << iter.dump() << "\n";
    return 1;
  }
  const auto &sm = iter["result"]["selfMind"];
  const std::string thought = sm.value("text", std::string());
  const std::string composed = iter["result"].value("composedPrompt", std::string());
  int fails = 0;
  auto check = [&](bool cond, const char *name) {
    if (cond) {
      std::cout << "PASS " << name << "\n";
    } else {
      std::cerr << "FAIL " << name << "\n";
      ++fails;
    }
  };
  check(sm.value("spontaneous", false), "spontaneous");
  check(!thought.empty(), "thought-nonempty");
  check(thought.find("ZZZ-QUESTION-TOKEN") == std::string::npos, "thought-ignores-question");
  check(composed == "ZZZ-QUESTION-TOKEN?\n", "composed-is-user-only");
  check(composed.find("[self-kernel]") == std::string::npos, "composed-no-kernel");
  check(composed.find("[Memory]") == std::string::npos, "composed-no-memory");
  check(composed.find(thought) == std::string::npos, "composed-no-thought");
  check(iter["result"].value("memoryScope", std::string()) == "chat:self-sess",
        "chat-scope");
  if (fails) {
    std::cerr << fails << " failed\ncomposed=" << composed << "\nthought=" << thought << "\n";
    return 1;
  }
  std::cout << "ALL PASS\n";
  return 0;
}
