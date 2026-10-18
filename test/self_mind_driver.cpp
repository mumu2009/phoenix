/* Standalone checks for the self-mind bandit, inner thought, and probes.
   Run from the repository root so the sample corpus path resolves. */

#include "self_mind.hpp"

#include <cmath>
#include <fstream>
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

int main(int argc, char **argv) {
    using phoenix::selfmind::SelfMind;
    using json = nlohmann::json;
    const char *kCorpus = "doc/v8.3/self_mind_sample.json";

    {
        SelfMind off;
        off.configure(json{{"enabled", false}});
        const auto t = off.tickThought(json::object());
        check(t.value("spontaneous", true) == false, "disabled-not-spontaneous");
        check(off.status().value("spontaneousCount", -1) == 0, "disabled-count");
    }

    {
        SelfMind frozen;
        frozen.configure(json{{"enabled", true}, {"learningRate", 0.0}});
        const auto ing = frozen.ingest(json{{"path", kCorpus}});
        check(ing.value("ok", false), "frozen-ingest");
        if (!ing.value("ok", false)) std::cerr << ing.dump() << "\n";
        frozen.tickThought(json::object());
        bool zero = true;
        for (const auto &b : frozen.status()["policyBias"]) {
            if (std::fabs(b.get<double>()) > 1e-8) zero = false;
        }
        check(zero, "alpha0-bias");
        check(frozen.status().value("episodes", -1) == 0, "alpha0-episodes");
    }

    SelfMind a;
    a.configure(json{{"enabled", true}, {"learningRate", 0.35}, {"name", "self"}});
    const auto ing = a.ingest(json{{"path", kCorpus}});
    check(ing.value("ok", false), "sample-ingest");
    if (!ing.value("ok", false)) std::cerr << ing.dump() << "\n";
    check(a.status().value("humanCount", 0) >= 6, "human-count");
    check(a.status().value("holdout", 0) >= 3, "holdout-count");
    check(a.status().value("chatbotCount", 0) >= 1, "chatbot-count");

    SelfMind b;
    b.configure(json{{"enabled", true}, {"learningRate", 0.35}, {"name", "self"}});
    b.ingest(json{{"path", kCorpus}});
    const auto ta = a.tickThought(json{{"userPrompt", "ZZZ-QUESTION-TOKEN?"}});
    const auto tb = b.tickThought(json::object());
    check(ta.value("spontaneous", false), "question-still-spontaneous");
    check(ta.value("ignoredQuestion", false), "question-ignored-flag");
    check(ta.value("text", std::string()) == tb.value("text", std::string("mismatch")),
          "question-does-not-steer");
    check(ta.value("text", std::string()).find("ZZZ-QUESTION-TOKEN") == std::string::npos,
          "question-token-absent");
    check(ta.value("action", std::string()) != "service", "first-act-not-service");

    for (int i = 0; i < 3; ++i) a.tickThought(json::object());
    const auto st = a.status();
    const auto &bias = st["policyBias"];
    check(bias.size() == 5 && bias[4].get<double>() < bias[0].get<double>(),
          "service-bias-below-recall");
    check(st.value("weightsFinite", false), "weights-finite");
    check(st.value("lastAction", std::string()) != "service", "last-act-not-service");
    const std::string kernel = st.value("kernel", std::string());
    check(kernel.find("[self-kernel]") != std::string::npos, "kernel-marker");
    check(kernel.find("how can I help") == std::string::npos &&
              kernel.find("language model") == std::string::npos,
          "kernel-not-helpdesk");
    check(kernel.find("not a service window") != std::string::npos, "kernel-not-window");

    const auto probe = a.probe();
    if (!probe.value("ok", false)) std::cerr << probe.dump(2) << "\n";
    check(probe["mirror"].value("pass", false), "mirror-pass");
    check(probe["turing"].value("pass", false), "turing-pass");

    SelfMind restored;
    restored.fromJson(a.toJson());
    check(restored.status().value("humanCount", -1) == a.status().value("humanCount", -2),
          "roundtrip-count");
    check(restored.status().value("lastThought", std::string("a")) ==
              a.status().value("lastThought", std::string("b")),
          "roundtrip-thought");

    SelfMind soak;
    soak.configure(json{{"enabled", true}, {"learningRate", 0.35}});
    soak.ingest(json{{"path", kCorpus}});
    for (int i = 0; i < 200; ++i) soak.tickThought(json::object());
    check(soak.status().value("weightsFinite", false), "no-runaway");
    check(soak.status().value("spontaneousCount", 0) == 200, "two-hundred-thoughts");

    const char *realPath = argc > 1 ? argv[1]
                                    : "runtime_store/datasets/self_mind/sharegpt_zh_turns.jsonl";
    {
        std::ifstream probe(realPath, std::ios::binary);
        check(static_cast<bool>(probe), "real-corpus-present");
        if (probe) {
            SelfMind real;
            real.configure(json{{"enabled", true}, {"learningRate", 0.35}, {"name", "self"}});
            const auto ring = real.ingest(json{{"path", realPath}});
            check(ring.value("ok", false), "real-ingest");
            if (!ring.value("ok", false)) std::cerr << ring.dump() << "\n";
            const auto rst = real.status();
            check(rst.value("humanCount", 0) >= 6, "real-human-count");
            check(rst.value("chatbotCount", 0) >= 1, "real-chatbot-count");
            check(rst.value("holdout", 0) >= 3, "real-holdout");
            for (int i = 0; i < 4; ++i) real.tickThought(json::object());
            const auto asked = real.tickThought(json{{"userPrompt", "ZZZ-QUESTION-TOKEN?"}});
            check(asked.value("spontaneous", false), "real-spontaneous");
            check(asked.value("text", std::string()).find("ZZZ-QUESTION-TOKEN") == std::string::npos,
                  "real-question-ignored");
            const auto rprobe = real.probe();
            std::cout << rprobe.dump(2) << "\n";
            check(rprobe["mirror"].value("pass", false), "real-mirror-pass");
            check(rprobe["turing"].value("pass", false), "real-turing-pass");
        }
    }

    if (gFails) {
        std::cerr << gFails << " failed\n";
        return 1;
    }
    std::cout << "ALL PASS\n";
    return 0;
}
