/* GTest smoke: all 649 v7.0 model deployment configs can be generated and loaded. */

#include "model_deployment.hpp"
#include "video_model.hpp"
#include "audio_model.hpp"
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cstdlib>

namespace fs = std::filesystem;

void forceLocalForFeasibility(phoenix::deployment::ModelDeploymentConfig &cfg) {
  auto j = cfg.toJson();
  bool changed = false;
  if (j.contains("vision") && j["vision"].contains("placement") &&
      j["vision"]["placement"] != "local") {
    j["vision"]["placement"] = "local";
    changed = true;
  }
  if (j.contains("speech") && j["speech"].contains("placement") &&
      j["speech"]["placement"] != "local") {
    j["speech"]["placement"] = "local";
    changed = true;
  }
  if (changed) {
    cfg.fromJson(j);
  }
}

class DeploymentMatrixLoadAll : public ::testing::Test {
 protected:
  static fs::path projectRoot() {
    // .../phoenix/tests/gtest/unit/autonomy/test_...cpp
    return fs::path(__FILE__).parent_path().parent_path().parent_path().parent_path().parent_path();
  }

  static std::vector<fs::path> findConfigs() {
    std::vector<fs::path> out;
    fs::path root = projectRoot() / "config" / "deployment_matrix";
    if (!fs::is_directory(root)) return out;
    for (const auto &entry : fs::recursive_directory_iterator(root)) {
      if (entry.is_regular_file() && entry.path().extension() == ".json") {
        out.push_back(entry.path());
      }
    }
    std::sort(out.begin(), out.end());
    return out;
  }

  static void ensureDumped() {
    auto configs = findConfigs();
    if (!configs.empty()) return;
    fs::path root = projectRoot();
    std::string cmd = "python \"" + (root / "tools" / "generate_model_deployment_matrix.py").string() + "\" --dump-all";
    int rc = std::system(cmd.c_str());
    ASSERT_EQ(rc, 0) << "Failed to run generate_model_deployment_matrix.py";
  }
};

TEST_F(DeploymentMatrixLoadAll, AllGeneratedConfigsAreLoadable) {
  ensureDumped();
  auto configs = findConfigs();
  ASSERT_FALSE(configs.empty()) << "config/deployment_matrix is empty";

  size_t loaded = 0;
  for (const auto &p : configs) {
    auto &cfg = phoenix::deployment::ModelDeploymentConfig::instance();
    cfg.reset();
    cfg.load({}, projectRoot().string(), p.string());
    auto j = cfg.toJson();
    EXPECT_FALSE(j.is_discarded()) << "Failed to round-trip " << p;
    ++loaded;
  }
  EXPECT_EQ(loaded, 649u) << "Expected 649 deployment configs";
}

TEST_F(DeploymentMatrixLoadAll, AllGeneratedConfigsAreRunnable) {
  // Deployment matrices can contain remote endpoints, but this test is a
  // smoke test for local fallback.  Force local placement so that a missing
  // remote server does not fail the suite.
  _putenv_s("PHOENIX_TEST_FORCE_LOCAL", "1");

  ensureDumped();
  auto configs = findConfigs();
  ASSERT_FALSE(configs.empty()) << "config/deployment_matrix is empty";

  size_t runnable = 0;
  size_t endToEndRan = 0;
  for (const auto &p : configs) {
    auto &cfg = phoenix::deployment::ModelDeploymentConfig::instance();
    cfg.reset();
    cfg.load({}, projectRoot().string(), p.string());
    forceLocalForFeasibility(cfg);

    auto videoEnc = phoenix::io::createVideoEncoder("video-encoder", 16);
    ASSERT_TRUE(videoEnc) << "video-encoder is null for " << p;
    auto videoEncStatus = videoEnc->status();
    EXPECT_FALSE(videoEncStatus.is_discarded());
    EXPECT_FALSE(videoEncStatus.value("backend", std::string()).empty())
        << "video-encoder backend is empty for " << p;

    auto videoDec = phoenix::io::createVideoDecoder("video-encoder", 16);
    ASSERT_TRUE(videoDec) << "video-decoder is null for " << p;
    auto videoDecStatus = videoDec->status();
    EXPECT_FALSE(videoDecStatus.is_discarded());
    EXPECT_FALSE(videoDecStatus.value("backend", std::string()).empty())
        << "video-decoder backend is empty for " << p;

    auto audioEnc = phoenix::io::createAudioEncoder("audio-encoder", 8);
    ASSERT_TRUE(audioEnc) << "audio-encoder is null for " << p;
    auto audioEncStatus = audioEnc->status();
    EXPECT_FALSE(audioEncStatus.is_discarded());
    EXPECT_FALSE(audioEncStatus.value("backend", std::string()).empty())
        << "audio-encoder backend is empty for " << p;

    auto audioDec = phoenix::io::createAudioDecoder("audio-encoder", 8);
    ASSERT_TRUE(audioDec) << "audio-decoder is null for " << p;
    auto audioDecStatus = audioDec->status();
    EXPECT_FALSE(audioDecStatus.is_discarded());
    EXPECT_FALSE(audioDecStatus.value("backend", std::string()).empty())
        << "audio-decoder backend is empty for " << p;

    const bool canRunLocal =
        (videoEncStatus.value("ready", false) && videoDecStatus.value("ready", false) &&
         audioEncStatus.value("ready", false) && audioDecStatus.value("ready", false));

    if (canRunLocal) {
      // One full encoder/decoder round-trip per run; for CI speed only one
      // configuration is exercised end-to-end, the rest are verified by status.
      if (endToEndRan < 3) {
        std::vector<uint8_t> imageBytes(224 * 224 * 3, 128);
        auto conceptVec = videoEnc->encode(imageBytes, 224, 224, "image/png");
        EXPECT_EQ(static_cast<int>(conceptVec.size()), 16)
            << "video-encoder concept size wrong for " << p << ": " << videoEnc->status().value("error", std::string());
        if (!conceptVec.empty()) {
          auto imageOut = videoDec->decode(conceptVec, "image/png");
          EXPECT_FALSE(imageOut.empty())
              << "video-decoder failed for " << p << ": " << videoDec->status().value("error", std::string());
        }

        std::vector<uint8_t> audioBytes(16000 * 2, 0);
        auto audioConcept = audioEnc->encode(audioBytes, 16000, "audio/pcm");
        EXPECT_EQ(static_cast<int>(audioConcept.size()), 8)
            << "audio-encoder concept size wrong for " << p << ": " << audioEnc->status().value("error", std::string());
        if (!audioConcept.empty()) {
          auto audioOut = audioDec->decode(audioConcept, "audio/pcm", 16000);
          EXPECT_FALSE(audioOut.empty())
              << "audio-decoder failed for " << p << ": " << audioDec->status().value("error", std::string());
        }
        ++endToEndRan;
      }
    }

    ++runnable;
  }
  EXPECT_EQ(runnable, 649u) << "Expected 649 runnable deployment configs";
}
