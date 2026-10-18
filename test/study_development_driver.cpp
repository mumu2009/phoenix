/* Standalone checks for unit learning, the population layer, and the tiny expert.
   Run from the repository root. Does not link phoenix_main and does not
   start or stop llama-server. */

#include "population_layer.hpp"
#include "study_unit_learn.hpp"
#include "summon_registry.hpp"
#include "tiny_expert.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

int gFails = 0;
int gPass = 0;

void check(bool cond, const char *name) {
    if (cond) {
        std::cout << "PASS " << name << "\n";
        ++gPass;
    } else {
        std::cerr << "FAIL " << name << "\n";
        ++gFails;
    }
}

double cosine(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size() || a.empty()) return 0.0;
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    if (na < 1e-12 || nb < 1e-12) return 0.0;
    return dot / std::sqrt(na * nb);
}

bool paramsEqual(const nlohmann::json &a, const nlohmann::json &b) {
    if (!a.is_array() || !b.is_array() || a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i].get<double>() - b[i].get<double>()) > 1e-6) return false;
    }
    return true;
}

nlohmann::json findIndividual(const nlohmann::json &st, const std::string &id) {
    if (!st.contains("individuals")) return nlohmann::json::object();
    for (const auto &ind : st["individuals"]) {
        if (ind.value("id", std::string()) == id) return ind;
    }
    return nlohmann::json::object();
}

void testUnits() {
    using phoenix::study::UnitLearner;
    const std::string marker = "ZEBRA-CAUSAL-MARKER-9981";

    {
        UnitLearner off;
        off.configure(nlohmann::json{{"enabled", false}, {"reset", true}});
        const int before = off.status().value("nodeCount", -1);
        const auto rep = off.learnEpisode(marker);
        check(rep.value("ok", true) == false, "unit-disabled");
        check(rep.value("writesCausalString", true) == false, "unit-disabled-no-causal");
        check(off.status().value("nodeCount", -1) == before, "unit-disabled-no-nodes");
    }

    {
        UnitLearner byte;
        const std::string text = "unit-byte-check";
        byte.configure(nlohmann::json{{"enabled", true}, {"reset", true}, {"maxMerges", 0}, {"dim", 16}});
        byte.fitEncoder(text);
        const auto ids = byte.encode(text);
        const auto rep = byte.learnEpisode(text);
        check(ids.size() == text.size(), "bpe-bytes-when-no-merges");
        check(rep.value("unitCount", -1) == static_cast<int>(ids.size()), "one-token-one-unit");
        check(rep.value("tokenEqualsUnit", false), "token-equals-unit-flag");
    }

    {
        UnitLearner merged;
        std::string pat;
        for (int i = 0; i < 24; ++i) pat += "abcd";
        merged.configure(nlohmann::json{{"enabled", true}, {"reset", true}, {"maxMerges", 2},
                                        {"dim", 16}, {"lr", 0.8}, {"steps", 3}});
        merged.fitEncoder(pat);
        const auto ids = merged.encode(pat);
        check(!ids.empty() && ids.size() < pat.size(), "bpe-merges-shrink-units");
        const auto first = merged.learnEpisode(pat);
        check(first.value("unitCount", -1) == static_cast<int>(ids.size()), "learn-count-matches-encode");
        check(first.value("steps", 0) >= 2, "propagate-steps");
        check(first["stages"].size() == 5, "full-process-stages");
        check(first.value("stateDelta", 0.0) > 1e-4, "propagation-moves-state");
        nlohmann::json last = first;
        for (int i = 0; i < 40; ++i) last = merged.learnEpisode(pat);
        const double l0 = first.value("loss", 1.0e9);
        const double l1 = last.value("loss", 1.0e9);
        if (!(l1 < l0 * 0.5)) {
            std::cerr << "loss " << l0 << " -> " << l1 << "\n";
        }
        check(l1 < l0 * 0.5, "predict-loss-drops");
        const std::string dump = merged.status().dump() + first.dump() + merged.sideChannel(last).dump();
        check(dump.find(pat) == std::string::npos, "episode-text-not-in-status");
        check(merged.sideChannel(last).value("writesCausalString", true) == false, "side-channel-no-causal");
        check(!merged.sideChannel(last).contains("composedPrompt"), "side-channel-no-prompt");
        check(!merged.sideChannel(last).contains("text"), "side-channel-no-text");

        UnitLearner restored;
        restored.fromJson(merged.toJson());
        check(restored.status().value("edgeCount", -1) == merged.status().value("edgeCount", -2),
              "unit-json-roundtrip");
    }

    {
        UnitLearner imp;
        imp.configure(nlohmann::json{{"enabled", true}, {"reset", true}, {"dim", 16}, {"steps", 3}});
        std::vector<float> row(16, 0.f);
        row[0] = 1.f;
        nlohmann::json units = nlohmann::json::array();
        units.push_back(nlohmann::json{{"id", 7}, {"row", row}});
        units.push_back(nlohmann::json{{"id", 9}, {"row", row}});
        const auto rep = imp.importUnits(nlohmann::json{{"units", units}});
        check(rep.value("ok", false) && rep.value("writesCausalString", true) == false, "import-units");
        check(cosine(imp.unitRow(7), row) > 0.99, "imported-row-kept");
        const auto recall = imp.recall(nlohmann::json{{"units", nlohmann::json::array({7})}});
        check(recall.value("writesCausalString", true) == false, "recall-no-causal");
        check(recall.contains("units") && recall["units"].is_array() && !recall["units"].empty(), "recall-rows");
        check(!recall.contains("composedPrompt"), "recall-no-prompt");
        bool hasRow = false;
        if (recall.contains("units") && !recall["units"].empty())
            hasRow = recall["units"][0].contains("row") && recall["units"][0]["row"].is_array();
        check(hasRow, "recall-row-present");
        const std::string markerDump = imp.status().dump() + rep.dump();
        check(markerDump.find(marker) == std::string::npos, "import-has-no-marker");
    }
}

void testPopulation() {
    using phoenix::study::PopulationLayer;
    using phoenix::study::populationPerfectParams;
    PopulationLayer bare;
    check(bare.enabled() == false, "population-default-off");
    const auto skipped = bare.runGoal(nlohmann::json{{"input", {0, 0, 0, 0}}, {"target", {0, 0}}});
    check(skipped.value("ran", true) == false && skipped.value("mutated", 1) == 0, "population-disabled-noop");

    PopulationLayer cap;
    const auto capped = cap.configure(nlohmann::json{{"enabled", true},
                                                     {"maxPopulation", 1000},
                                                     {"n", 100},
                                                     {"seed", 4}});
    check(capped.value("ok", false), "population-large-n-accepted-inside-cap");
    check(cap.status().value("maxPopulation", 0) == PopulationLayer::kHardCap, "population-hard-cap");
    check(cap.status().value("n", 0) == PopulationLayer::kHardCap, "population-n-clamped");
    const auto tooSmall = cap.configure(nlohmann::json{{"n", 1}});
    check(tooSmall.value("ok", true) == false, "population-rejects-n-below-2");
    check(cap.status().value("n", 0) == PopulationLayer::kHardCap, "population-n-unchanged-after-reject");

    PopulationLayer pop;
    pop.configure(nlohmann::json{{"enabled", true}, {"n", 4}, {"maxPopulation", 8}, {"seed", 2},
                                 {"mutationScale", 0.05}, {"tolerance", 0.2}});
    const nlohmann::json input = nlohmann::json::array({0.2, -0.1, 0.5, 0.0});
    const nlohmann::json target = nlohmann::json::array({0.3, -0.4});
    const auto perfect = populationPerfectParams(4, 4, 2, {0.3f, -0.4f});
    nlohmann::json perfectJson = nlohmann::json::array();
    for (float v : perfect) perfectJson.push_back(v);
    check(pop.setIndividual(nlohmann::json{{"index", 0}, {"params", perfectJson}, {"lastError", 0.0}})
              .value("ok", false),
          "population-set-anchor");
    const auto y = pop.forwardIndex(0, {0.2f, -0.1f, 0.5f, 0.0f});
    float err = 0.f;
    if (y.size() == 2) err = (std::fabs(y[0] - 0.3f) + std::fabs(y[1] + 0.4f)) * 0.5f;
    check(err < 1e-5f, "population-forward-matches-target");
    const auto before = findIndividual(pop.status(), "ind-0");
    const auto goal = pop.runGoal(nlohmann::json{{"input", input}, {"target", target}});
    check(goal.value("ran", false), "population-goal-ran");
    check(goal.value("mutated", 0) == 3, "population-mutates-n-minus-one");
    check(goal.value("anchorUnchanged", false), "population-anchor-not-mutated");
    check(goal.value("kept", 0) == 2, "population-keeps-half");
    check(goal.value("n", 0) == 4, "population-size-fixed");
    check(goal.value("spawnedInstances", 1) == 0, "population-no-instance-spawn");
    check(goal.value("correctCount", 0) >= 1, "population-correct-output");
    check(goal.value("anchorKept", false), "population-anchor-kept");
    const auto after = findIndividual(pop.status(), "ind-0");
    check(paramsEqual(before.value("params", nlohmann::json::array()),
                      after.value("params", nlohmann::json::array())),
          "population-anchor-params-kept");

    PopulationLayer frozen;
    frozen.configure(nlohmann::json{{"enabled", true}, {"n", 4}, {"seed", 9}});
    const auto snap = frozen.status()["individuals"];
    frozen.configure(nlohmann::json{{"enabled", false}});
    frozen.runGoal(nlohmann::json{{"input", input}, {"target", target}});
    check(frozen.status()["individuals"].dump() == snap.dump(), "population-off-keeps-params");

    PopulationLayer evo;
    evo.configure(nlohmann::json{{"enabled", true}, {"n", 4}, {"maxPopulation", 8}, {"seed", 11},
                                 {"mutationScale", 0.08}, {"tolerance", 0.2}});
    double prev = 1.0e9;
    bool monotone = true;
    bool bounded = true;
    for (int g = 0; g < 20; ++g) {
        const auto rep = evo.runGoal(nlohmann::json{{"input", input}, {"target", target}});
        const double best = rep.value("bestError", 1.0e9);
        if (best > prev + 1e-5) monotone = false;
        if (rep.value("n", 0) != 4 || rep.value("spawnedInstances", 1) != 0) bounded = false;
        if (rep.value("maxPopulation", 0) > PopulationLayer::kHardCap) bounded = false;
        prev = best;
    }
    check(monotone, "population-best-error-monotone");
    check(bounded, "population-stays-inside-cap");

    PopulationLayer copy;
    copy.fromJson(evo.toJson());
    check(copy.status().value("n", 0) == evo.status().value("n", -1), "population-json-roundtrip");
}

void testExpert() {
    using phoenix::study::TinyExpert;
    TinyExpert off;
    const auto denied = off.trainForGoal(nlohmann::json{{"goal", "x"}});
    check(off.enabled() == false && denied.value("ran", true) == false, "expert-default-off");

    TinyExpert toy;
    toy.configure(nlohmann::json{{"enabled", true},
                                 {"vocab", 8},
                                 {"hidden", 8},
                                 {"classes", 2},
                                 {"epochs", 24},
                                 {"lr", 0.35},
                                 {"seed", 5},
                                 {"reset", true}});
    check(toy.paramCount() < 5000, "expert-param-count-small");
    nlohmann::json samples = nlohmann::json::array();
    for (int rep = 0; rep < 24; ++rep) {
        for (int i = 0; i < 8; ++i) {
            nlohmann::json x = nlohmann::json::array();
            for (int k = 0; k < 8; ++k) x.push_back(k == i ? 1.0 : 0.0);
            samples.push_back(nlohmann::json{{"x", x}, {"y", i < 4 ? 0 : 1}});
        }
    }
    const std::string toyPath = "runtime_store/experts/study_tiny/synthetic.json";
    const auto trained = toy.trainForGoal(nlohmann::json{{"samples", samples},
                                                         {"epochs", 24},
                                                         {"lr", 0.35},
                                                         {"seed", 5},
                                                         {"weightsPath", toyPath}});
    if (!trained.value("ok", false)) std::cerr << trained.dump() << "\n";
    const double l0 = trained.value("initialLoss", 0.0);
    const double l1 = trained.value("finalLoss", 1.0e9);
    if (!(l1 < l0)) std::cerr << "synthetic loss " << l0 << " -> " << l1 << "\n";
    check(trained.value("ok", false) && l1 < l0, "expert-synthetic-loss-drops");
    check(trained.value("accuracy", 0.0) >= 0.9, "expert-synthetic-accuracy");

    nlohmann::json probeX = nlohmann::json::array();
    for (int k = 0; k < 8; ++k) probeX.push_back(k == 0 ? 1.0 : 0.0);
    const auto before = toy.predict(nlohmann::json{{"x", probeX}});
    TinyExpert loaded;
    const auto ld = loaded.load(toyPath);
    const auto after = loaded.predict(nlohmann::json{{"x", probeX}});
    bool close = ld.value("ok", false) && before.value("ok", false) && after.value("ok", false);
    if (close) {
        const auto &pa = before["probs"];
        const auto &pb = after["probs"];
        if (!pa.is_array() || pa.size() != pb.size()) close = false;
        for (size_t i = 0; close && i < pa.size(); ++i) {
            if (std::fabs(pa[i].get<double>() - pb[i].get<double>()) > 1e-5) close = false;
        }
    }
    check(close, "expert-save-load-forward");

    const std::string corpus = "runtime_store/datasets/study_expert/corpus.txt";
    const std::string prov = "runtime_store/datasets/study_expert/PROVENANCE.json";
    std::ifstream pin(prov);
    nlohmann::json provenance;
    bool provOk = false;
    if (pin) {
        try {
            provenance = nlohmann::json::parse(pin);
            provOk = provenance.contains("sourceUrl") && provenance.contains("license") &&
                     provenance["sourceUrl"].is_string() && !provenance["sourceUrl"].get<std::string>().empty() &&
                     provenance["license"].is_string() && !provenance["license"].get<std::string>().empty();
        } catch (...) {
            provOk = false;
        }
    }
    check(provOk, "expert-provenance");

    TinyExpert real;
    real.configure(nlohmann::json{{"enabled", true},
                                  {"vocab", 32},
                                  {"hidden", 16},
                                  {"classes", 32},
                                  {"epochs", 6},
                                  {"lr", 0.4},
                                  {"seed", 3},
                                  {"reset", true},
                                  {"goal", " the "},
                                  {"corpusPath", corpus},
                                  {"weightsPath", "runtime_store/experts/study_tiny/weights.json"}});
    const auto got = real.trainForGoal(nlohmann::json{{"goal", " the "},
                                                      {"corpusPath", corpus},
                                                      {"epochs", 6},
                                                      {"lr", 0.4},
                                                      {"seed", 3}});
    if (!got.value("ok", false)) std::cerr << got.dump() << "\n";
    const double r0 = got.value("initialLoss", 0.0);
    const double r1 = got.value("finalLoss", 1.0e9);
    if (got.value("ok", false) && !(r1 < r0)) std::cerr << "real loss " << r0 << " -> " << r1 << "\n";
    check(got.value("ok", false), "expert-real-trained");
    check(got.value("ran", false) && r1 < r0, "expert-real-loss-drops");
    check(got.value("paramCount", 100000) < 5000, "expert-real-stays-tiny");
    check(!got.value("source", std::string()).empty() && !got.value("license", std::string()).empty(),
          "expert-records-source-license");

    phoenix::summon::SummonRegistry::instance().loadFromConfig(true, nlohmann::json::array());
    check(real.registerSummon("goal-expert"), "expert-summon-register");
    const auto *ep = phoenix::summon::SummonRegistry::instance().find("goal-expert");
    check(ep != nullptr && ep->kind == "tiny-expert" && ep->url.empty() && !ep->weights.empty(),
          "expert-summon-endpoint");
    phoenix::summon::SummonRegistry::instance().loadFromConfig(false, nlohmann::json::array());
    check(real.registerSummon("goal-expert") == false, "expert-summon-respects-disable");

    nlohmann::json models = nlohmann::json::array({
        nlohmann::json{{"name", "llama-local"},
                       {"kind", "llamacpp"},
                       {"url", "http://127.0.0.1:8082"}},
        nlohmann::json{{"name", "bad-ollama"}, {"kind", "ollama"}},
        nlohmann::json{{"name", "goal-expert"},
                       {"kind", "tiny-expert"},
                       {"weights", "runtime_store/experts/study_tiny/weights.json"}},
    });
    const auto loadedReg = phoenix::summon::SummonRegistry::instance().loadFromConfig(true, models);
    check(loadedReg["loaded"].size() == 2 && loadedReg["skipped"].size() == 1, "summon-keeps-url-rule");
    const auto *tiny = phoenix::summon::SummonRegistry::instance().find("goal-expert");
    const auto *base = phoenix::summon::SummonRegistry::instance().find("llama-local");
    check(tiny != nullptr && base != nullptr &&
              phoenix::summon::SummonRegistry::instance().find("bad-ollama") == nullptr,
          "summon-lists-tiny-expert");

    nlohmann::json moeSpec = {
        {"name", "single-moe"},
        {"kind", "moe"},
        {"path", "GGUF_models/granite.gguf"},
        {"port", 8100},
        {"K", 2},
        {"forceExpert", 0},
    };
    const auto moeUp = phoenix::summon::SummonRegistry::instance().upsertMoe(moeSpec);
    check(moeUp.value("ok", false) && moeUp.value("started", true) == false &&
              moeUp.value("updated", true) == false,
          "moe-upsert-no-process");
    const auto *moeEp = phoenix::summon::SummonRegistry::instance().find("single-moe");
    check(moeEp != nullptr && moeEp->kind == "moe" &&
              moeEp->modelPath == "GGUF_models/granite.gguf" && moeEp->port == 8100 &&
              moeEp->url == "http://127.0.0.1:8100" && moeEp->k == 2 &&
              moeEp->pinnedExpert == 0,
          "moe-spec-fields");
    const auto listed = phoenix::summon::SummonRegistry::instance().list();
    bool keptTiny = false, keptLlama = false;
    for (const auto &row : listed) {
        if (row.name == "goal-expert" && row.kind == "tiny-expert") keptTiny = true;
        if (row.name == "llama-local" && row.kind == "llamacpp") keptLlama = true;
    }
    check(keptTiny && keptLlama && listed.size() == 3, "moe-keeps-llama-and-tiny");

    nlohmann::json moeDefault = {
        {"name", "single-moe"},
        {"modelPath", "GGUF_models/granite.gguf"},
        {"url", "http://127.0.0.1:8100"},
    };
    const auto moeDef = phoenix::summon::SummonRegistry::instance().upsertMoe(moeDefault);
    const auto *moeEp2 = phoenix::summon::SummonRegistry::instance().find("single-moe");
    check(moeDef.value("ok", false) && moeDef.value("updated", false) &&
              moeDef.value("started", true) == false && moeEp2 != nullptr &&
              moeEp2->kind == "moe" && moeEp2->k == 1 && moeEp2->pinnedExpert == -1 &&
              moeEp2->url == "http://127.0.0.1:8100" &&
              phoenix::summon::SummonRegistry::instance().list().size() == 3,
          "moe-defaults");
    phoenix::summon::SummonRegistry::instance().loadFromConfig(false, nlohmann::json::array());
}

}  // namespace

int main() {
    testUnits();
    testPopulation();
    testExpert();
    std::cout << "STUDY_DRIVER passed=" << gPass << " failed=" << gFails << "\n";
    return gFails == 0 ? 0 : 1;
}
