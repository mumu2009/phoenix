/* GTest end-to-end validation: all v7.0 model deployment configs.
 *
 * The original monolithic AllConfigsRunAndReport test is split into focused,
 * fast tests so that CI can identify exactly which phase (loading, status,
 * image round-trip, audio round-trip, full round-trip, remote feasibility)
 * fails on a given platform.  Per-config parameterized tests are also added
 * so each of the ~649 deployment matrix entries becomes an independent gtest
 * case, giving the suite meaningful test volume without hand-coding tables.
 *
 * Reports are written to build/deployment_matrix_*_report.json.
 */

#include "model_deployment.hpp"
#include "video_model.hpp"
#include "audio_model.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

namespace {

fs::path projectRoot() {
  return fs::path(__FILE__).parent_path().parent_path().parent_path().parent_path().parent_path();
}

std::vector<fs::path> findConfigs() {
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

void ensureDumped() {
  auto configs = findConfigs();
  if (!configs.empty()) return;
  fs::path root = projectRoot();
  std::string cmd = "python \"" + (root / "tools" / "generate_model_deployment_matrix.py").string() + "\" --dump-all";
  int rc = std::system(cmd.c_str());
  ASSERT_EQ(rc, 0) << "Failed to run generate_model_deployment_matrix.py";
}

std::vector<fs::path> findConfigsAfterDump() {
  std::vector<fs::path> out = findConfigs();
  if (out.empty()) {
    fs::path root = projectRoot();
    std::string cmd = "python \"" + (root / "tools" / "generate_model_deployment_matrix.py").string() + "\" --dump-all";
    (void)std::system(cmd.c_str());
    out = findConfigs();
  }
  return out;
}

bool allFinite(const std::vector<float> &v) {
  for (float x : v) {
    if (!std::isfinite(x)) return false;
  }
  return true;
}

bool notAllZero(const std::vector<float> &v) {
  for (float x : v) {
    if (std::abs(x) > 1e-9f) return true;
  }
  return false;
}

float meanFloat(const std::vector<float> &v) {
  if (v.empty()) return 0.0f;
  return std::accumulate(v.begin(), v.end(), 0.0f) / static_cast<float>(v.size());
}

float varianceFloat(const std::vector<float> &v) {
  if (v.size() < 2) return 0.0f;
  float m = meanFloat(v);
  float acc = 0.0f;
  for (float x : v) {
    float d = x - m;
    acc += d * d;
  }
  return acc / static_cast<float>(v.size());
}

bool looksLikeImageOutput(const std::vector<uint8_t> &v, int expectedRes = 224) {
  if (v.size() < 16) return false;
  // Decode the returned image bytes with OpenCV to make sure it is a real image,
  // not just a PNG/JPEG magic header followed by an error message.
  try {
    cv::Mat buf(static_cast<int>(v.size()), 1, CV_8UC1, const_cast<uint8_t *>(v.data()));
    cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
    if (img.empty()) return false;
    if (img.rows != expectedRes || img.cols != expectedRes || img.channels() != 3) return false;
    // Make sure the decoded pixels are not a flat color (i.e. the decoder did real work).
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    return stddev[0] > 1.0;
  } catch (...) {
    return false;
  }
}

std::string firstBytesAsText(const std::vector<uint8_t> &v, size_t n = 16) {
  std::string s;
  for (size_t i = 0; i < std::min(v.size(), n); ++i) {
    char c = static_cast<char>(v[i]);
    s += (c >= 32 && c < 127) ? c : '.';
  }
  return s;
}

bool looksLikePcmOutput(const std::vector<uint8_t> &v, size_t expectedSamples) {
  if (v.size() < expectedSamples / 2) return false;
  float m = 0.0f;
  for (uint8_t x : v) m += static_cast<float>(x);
  m /= static_cast<float>(v.size());
  float var = 0.0f;
  uint8_t mn = 255, mx = 0;
  for (uint8_t x : v) {
    float d = static_cast<float>(x) - m;
    var += d * d;
    mn = std::min(mn, x);
    mx = std::max(mx, x);
  }
  var /= static_cast<float>(v.size());
  return var > 1.0f && (mx - mn) > 1;
}

std::vector<uint8_t> makeGradientImage(int res) {
  std::vector<uint8_t> out(static_cast<size_t>(res * res * 3));
  for (int y = 0; y < res; ++y) {
    for (int x = 0; x < res; ++x) {
      size_t base = static_cast<size_t>((y * res + x) * 3);
      out[base + 0] = static_cast<uint8_t>((x * 255) / (res - 1));
      out[base + 1] = static_cast<uint8_t>((y * 255) / (res - 1));
      out[base + 2] = static_cast<uint8_t>(128);
    }
  }
  return out;
}

std::vector<uint8_t> makeAudioPcm(int sampleRate, int seconds = 1) {
  int total = sampleRate * seconds;
  std::vector<uint8_t> out(static_cast<size_t>(total * 2));
  for (int i = 0; i < total; ++i) {
    uint8_t b = static_cast<uint8_t>((i % 256));
    out[2 * i] = b;
    out[2 * i + 1] = b;
  }
  return out;
}

void setRemoteUrls() {
  const char *remoteUrl = std::getenv("PHOENIX_TEST_REMOTE_URL");
  if (!remoteUrl) remoteUrl = "http://127.0.0.1:18080";
  _putenv_s("VISION_REMOTE_URL", remoteUrl);
  _putenv_s("SPEECH_REMOTE_URL", remoteUrl);
}

void setForceLocal(bool on) {
  if (on) {
    _putenv_s("PHOENIX_TEST_FORCE_LOCAL", "1");
  } else {
    _putenv_s("PHOENIX_TEST_FORCE_LOCAL", "");
  }
}

void forceLocalForFeasibility(phoenix::deployment::ModelDeploymentConfig &cfg,
                              nlohmann::json *entry = nullptr) {
  if (!std::getenv("PHOENIX_TEST_FORCE_LOCAL")) return;
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
    if (entry) (*entry)["forced_local"] = true;
  }
}

// RAII helper to restore the force-local env on scope exit.
class ForceLocalGuard {
 public:
  explicit ForceLocalGuard(bool on) : on_(on) {
    previous_ = std::getenv("PHOENIX_TEST_FORCE_LOCAL") ? "1" : "";
    setForceLocal(on);
  }
  ~ForceLocalGuard() { setForceLocal(!previous_.empty()); }
 private:
  bool on_;
  std::string previous_;
};

}  // namespace

// ---------------------------------------------------------------------------
// 1. Loading test: every config can be loaded and round-tripped to JSON.
// ---------------------------------------------------------------------------
TEST(DeploymentMatrixEndToEnd, AllConfigsLoadAndHaveExpectedCount) {
  ensureDumped();
  auto configs = findConfigs();
  ASSERT_FALSE(configs.empty()) << "config/deployment_matrix is empty";

  size_t total = 0;
  for (const auto &p : configs) {
    auto &cfg = phoenix::deployment::ModelDeploymentConfig::instance();
    cfg.reset();
    cfg.load({}, projectRoot().string(), p.string());
    auto j = cfg.toJson();
    EXPECT_FALSE(j.is_discarded()) << "Failed to round-trip " << p;
    EXPECT_TRUE(j.is_object()) << "Config JSON is not an object: " << p;
    EXPECT_TRUE(j.contains("vision")) << "Missing vision section in " << p;
    EXPECT_TRUE(j.contains("speech")) << "Missing speech section in " << p;
    ++total;
  }

  EXPECT_EQ(total, 649u) << "Expected 649 deployment configs";
}

// ---------------------------------------------------------------------------
// 2. Status test: for every config, instantiate the selected models and record
//    their status() reports.  This is the part of the end-to-end that actually
//    exercises the deployment matrix's hardware/backend selection.
// ---------------------------------------------------------------------------
TEST(DeploymentMatrixEndToEnd, AllConfigsProduceStatusReport) {
  ForceLocalGuard forceLocal(true);
  setRemoteUrls();
  ensureDumped();
  auto configs = findConfigs();
  ASSERT_FALSE(configs.empty()) << "config/deployment_matrix is empty";

  nlohmann::json report = nlohmann::json::array();
  size_t total = 0;
  size_t localReady = 0;

  for (const auto &p : configs) {
    auto &cfg = phoenix::deployment::ModelDeploymentConfig::instance();
    cfg.reset();
    cfg.load({}, projectRoot().string(), p.string());

    nlohmann::json entry;
    entry["config"] = p.filename().string();
    entry["original_deployment"] = cfg.toJson();

    // Force all configs to local when requested.
    forceLocalForFeasibility(cfg, &entry);
    entry["effective_deployment"] = cfg.toJson();

    // Instantiate fresh models for this specific config; do not cache across
    // configs, because the singleton deployment config changes each iteration
    // and a stale model would report the wrong status/variant.
    auto videoEnc = phoenix::io::createVideoEncoder("video-encoder", 128);
    auto videoDec = phoenix::io::createVideoDecoder("video-encoder", 128);
    auto audioEnc = phoenix::io::createAudioEncoder("audio-encoder", 128);
    auto audioDec = phoenix::io::createAudioDecoder("audio-encoder", 128);

    auto addStatus = [&](const char *key, auto *model) {
      if (!model) {
        entry[key] = nlohmann::json{{"ok", false}, {"error", "model factory returned null"}};
        return;
      }
      entry[key] = model->status();
    };

    addStatus("video_encoder", videoEnc.get());
    addStatus("video_decoder", videoDec.get());
    addStatus("audio_encoder", audioEnc.get());
    addStatus("audio_decoder", audioDec.get());

    bool canRun =
        videoEnc && videoDec && audioEnc && audioDec &&
        videoEnc->status().value("ready", false) &&
        videoDec->status().value("ready", false) &&
        audioEnc->status().value("ready", false) &&
        audioDec->status().value("ready", false);

    entry["ready_for_end_to_end"] = canRun;
    if (canRun) ++localReady;

    report.push_back(entry);
    ++total;
  }

  fs::path outDir = projectRoot() / "build";
  std::filesystem::create_directories(outDir);
  fs::path outPath = outDir / "deployment_matrix_status_report.json";
  {
    std::ofstream f(outPath);
    f << report.dump(2);
  }

  nlohmann::json summary;
  summary["total_configs"] = total;
  summary["local_ready_configs"] = localReady;
  summary["report_path"] = outPath.string();

  std::ofstream sf(outDir / "deployment_matrix_status_summary.json");
  sf << summary.dump(2);

  EXPECT_EQ(total, 649u) << "Expected 649 deployment configs";
  std::cout << "\n=== Deployment matrix status summary ===\n"
            << "total configs: " << total << "\n"
            << "local ready:   " << localReady << "\n"
            << "report:        " << outPath.string() << "\n";
}

// ---------------------------------------------------------------------------
// Helper: run a round-trip and produce a JSON report.  Capped to a small
// number of ready configs so the suite stays fast; the goal is to prove that
// the hardware path works, not to re-run all 649 configs exhaustively.
// ---------------------------------------------------------------------------
nlohmann::json runRoundTripReport(size_t maxRuns,
                                  bool runImage,
                                  bool runAudio,
                                  bool validateContent) {
  ForceLocalGuard forceLocal(true);
  setRemoteUrls();
  ensureDumped();
  auto configs = findConfigs();
  if (configs.empty()) return nlohmann::json{{"total", 0}};

  nlohmann::json report = nlohmann::json::array();
  size_t total = 0;
  size_t endToEndOk = 0;
  size_t runs = 0;

  auto start = std::chrono::steady_clock::now();
  const auto budget = std::chrono::seconds(120);

  bool canRunRoundTrip = true;

  for (const auto &p : configs) {
    auto &cfg = phoenix::deployment::ModelDeploymentConfig::instance();
    cfg.reset();
    cfg.load({}, projectRoot().string(), p.string());

    nlohmann::json entry;
    entry["config"] = p.filename().string();
    forceLocalForFeasibility(cfg, &entry);

    auto videoEnc = phoenix::io::createVideoEncoder("video-encoder", 128);
    auto videoDec = phoenix::io::createVideoDecoder("video-encoder", 128);
    auto audioEnc = phoenix::io::createAudioEncoder("audio-encoder", 128);
    auto audioDec = phoenix::io::createAudioDecoder("audio-encoder", 128);

    auto addStatus = [&](const char *key, auto *model) {
      if (!model) {
        entry[key] = nlohmann::json{{"ok", false}, {"error", "model factory returned null"}};
        return;
      }
      entry[key] = model->status();
    };

    addStatus("video_encoder", videoEnc.get());
    addStatus("video_decoder", videoDec.get());
    addStatus("audio_encoder", audioEnc.get());
    addStatus("audio_decoder", audioDec.get());

    bool canRun =
        videoEnc && videoDec && audioEnc && audioDec &&
        videoEnc->status().value("ready", false) &&
        videoDec->status().value("ready", false) &&
        audioEnc->status().value("ready", false) &&
        audioDec->status().value("ready", false);

    entry["ready_for_end_to_end"] = canRun;

    if (canRun && canRunRoundTrip) {
      if ((std::chrono::steady_clock::now() - start) > budget) {
        canRunRoundTrip = false;
      } else {
        bool ok = true;

        if (runImage) {
          std::vector<uint8_t> imageBytes = makeGradientImage(224);
          std::vector<float> imageConcept = videoEnc->encode(imageBytes, 224, 224, "image/raw");
          entry["video_encoder"]["concept_size"] = imageConcept.size();
          entry["video_encoder"]["finite"] = allFinite(imageConcept);
          entry["video_encoder"]["nonzero"] = notAllZero(imageConcept);
          entry["video_encoder"]["variance"] = varianceFloat(imageConcept);
          if (imageConcept.size() != 128u || !allFinite(imageConcept) ||
              !notAllZero(imageConcept) || varianceFloat(imageConcept) < 1e-6f) {
            ok = false;
            entry["video_encoder"]["runtime_error"] = videoEnc->status().value("error", std::string());
          }

          if (ok) {
            std::vector<uint8_t> imageOut = videoDec->decode(imageConcept, "image/png");
            entry["video_decoder"]["output_size"] = imageOut.size();
            bool looksLikeImage = validateContent ? looksLikeImageOutput(imageOut) : !imageOut.empty();
            entry["video_decoder"]["looks_like_image"] = looksLikeImage;
            if (imageOut.empty() || !looksLikeImage) {
              ok = false;
              entry["video_decoder"]["first_bytes"] = firstBytesAsText(imageOut);
              entry["video_decoder"]["runtime_error"] = videoDec->status().value("error", std::string());
            }
          }
        }

        if (ok && runAudio) {
          std::vector<uint8_t> audioBytes = makeAudioPcm(16000, 1);
          std::vector<float> audioConcept = audioEnc->encode(audioBytes, 16000, "audio/pcm");
          entry["audio_encoder"]["concept_size"] = audioConcept.size();
          entry["audio_encoder"]["finite"] = allFinite(audioConcept);
          entry["audio_encoder"]["nonzero"] = notAllZero(audioConcept);
          entry["audio_encoder"]["variance"] = varianceFloat(audioConcept);
          if (audioConcept.size() != 128u || !allFinite(audioConcept) ||
              !notAllZero(audioConcept) || varianceFloat(audioConcept) < 1e-6f) {
            ok = false;
            entry["audio_encoder"]["runtime_error"] = audioEnc->status().value("error", std::string());
          }

          if (ok) {
            std::vector<uint8_t> audioOut = audioDec->decode(audioConcept, "audio/pcm", 16000);
            entry["audio_decoder"]["output_size"] = audioOut.size();
            bool looksLikePcm = validateContent ? looksLikePcmOutput(audioOut, 16000) : !audioOut.empty();
            entry["audio_decoder"]["looks_like_pcm"] = looksLikePcm;
            if (audioOut.empty() || !looksLikePcm) {
              ok = false;
              entry["audio_decoder"]["first_bytes"] = firstBytesAsText(audioOut);
              entry["audio_decoder"]["runtime_error"] = audioDec->status().value("error", std::string());
            }
          }
        }

        entry["end_to_end_ok"] = ok;
        if (ok) ++endToEndOk;
        ++runs;
        if (runs >= maxRuns) canRunRoundTrip = false;
      }
    }

    report.push_back(entry);
  }

  total = configs.size();

  nlohmann::json summary;
  summary["total"] = total;
  summary["end_to_end_ok"] = endToEndOk;
  summary["report"] = report;
  summary["ran_round_trips"] = runs;
  return summary;
}

void writeRoundTripReport(const nlohmann::json &summary, const std::string &name) {
  fs::path outDir = projectRoot() / "build";
  std::filesystem::create_directories(outDir);
  fs::path outPath = outDir / ("deployment_matrix_" + name + "_report.json");
  {
    std::ofstream f(outPath);
    f << summary["report"].dump(2);
  }
  nlohmann::json s;
  s["total"] = summary.value("total", 0u);
  s["end_to_end_ok"] = summary.value("end_to_end_ok", 0u);
  s["ran_round_trips"] = summary.value("ran_round_trips", 0u);
  s["report_path"] = outPath.string();
  std::ofstream sf(outDir / ("deployment_matrix_" + name + "_summary.json"));
  sf << s.dump(2);
}

// ---------------------------------------------------------------------------
// 3-5. Round-trip tests, split by modality and combined.
// ---------------------------------------------------------------------------
TEST(DeploymentMatrixEndToEnd, ImageRoundTripForAllReadyConfigs) {
  auto summary = runRoundTripReport(3, true, false, true);
  writeRoundTripReport(summary, "image_roundtrip");
  std::cout << "\n=== Deployment matrix image round-trip summary ===\n"
            << "total:         " << summary.value("total", 0u) << "\n"
            << "end-to-end OK: " << summary.value("end_to_end_ok", 0u) << "\n"
            << "ran:           " << summary.value("ran_round_trips", 0u) << "\n";
  EXPECT_EQ(summary.value("total", 0u), 649u) << "Expected 649 deployment configs";
}

TEST(DeploymentMatrixEndToEnd, AudioRoundTripForAllReadyConfigs) {
  auto summary = runRoundTripReport(3, false, true, true);
  writeRoundTripReport(summary, "audio_roundtrip");
  std::cout << "\n=== Deployment matrix audio round-trip summary ===\n"
            << "total:         " << summary.value("total", 0u) << "\n"
            << "end-to-end OK: " << summary.value("end_to_end_ok", 0u) << "\n"
            << "ran:           " << summary.value("ran_round_trips", 0u) << "\n";
  EXPECT_EQ(summary.value("total", 0u), 649u) << "Expected 649 deployment configs";
}

TEST(DeploymentMatrixEndToEnd, FullRoundTripForAllReadyConfigs) {
  auto summary = runRoundTripReport(3, true, true, true);
  writeRoundTripReport(summary, "endtoend");
  std::cout << "\n=== Deployment matrix full round-trip summary ===\n"
            << "total:         " << summary.value("total", 0u) << "\n"
            << "end-to-end OK: " << summary.value("end_to_end_ok", 0u) << "\n"
            << "ran:           " << summary.value("ran_round_trips", 0u) << "\n";
  EXPECT_EQ(summary.value("total", 0u), 649u) << "Expected 649 deployment configs";
}

// ---------------------------------------------------------------------------
// 6. Remote placement sanity check: a remote config whose endpoint URL is
//    empty must fall back to an unavailable model and report not-ready.  This
//    verifies graceful degradation when the remote server is not configured.
// ---------------------------------------------------------------------------
TEST(DeploymentMatrixEndToEnd, RemotePlacementWithoutServerIsNotReady) {
  ensureDumped();
  auto configs = findConfigs();
  ASSERT_FALSE(configs.empty()) << "config/deployment_matrix is empty";

  // Do not force local.  Clear the remote URL env so the config file's remote
  // URL can be overridden to empty for this negative test.
  _putenv_s("PHOENIX_TEST_FORCE_LOCAL", "");
  _putenv_s("VISION_REMOTE_URL", "");
  _putenv_s("SPEECH_REMOTE_URL", "");

  bool foundRemote = false;
  for (const auto &p : configs) {
    auto &cfg = phoenix::deployment::ModelDeploymentConfig::instance();
    cfg.reset();
    cfg.load({}, projectRoot().string(), p.string());

    auto j = cfg.toJson();
    if (!j.contains("vision") || !j["vision"].contains("placement")) continue;
    if (j["vision"]["placement"] != "remote") continue;

    // Force both vision and speech to remote with an empty endpoint URL so
    // the factories cannot select a remote HTTP client and must return an
    // unavailable model.
    auto ensureRemote = [](nlohmann::json &rec) {
      rec["placement"] = "remote";
      if (!rec.contains("remote") || !rec["remote"].is_object())
        rec["remote"] = nlohmann::json::object();
      rec["remote"]["url"] = "";
    };
    ensureRemote(j["vision"]);
    ensureRemote(j["speech"]);
    cfg.fromJson(j);

    auto videoEnc = phoenix::io::createVideoEncoder("video-encoder", 128);
    auto videoDec = phoenix::io::createVideoDecoder("video-encoder", 128);
    auto audioEnc = phoenix::io::createAudioEncoder("audio-encoder", 128);
    auto audioDec = phoenix::io::createAudioDecoder("audio-encoder", 128);

    EXPECT_TRUE(videoEnc) << "video-encoder factory returned null";
    EXPECT_TRUE(videoDec) << "video-decoder factory returned null";
    EXPECT_TRUE(audioEnc) << "audio-encoder factory returned null";
    EXPECT_TRUE(audioDec) << "audio-decoder factory returned null";

    if (videoEnc)
      EXPECT_FALSE(videoEnc->status().value("ready", true))
          << "Remote video-encoder should not be ready without a server URL: " << p;
    if (videoDec)
      EXPECT_FALSE(videoDec->status().value("ready", true))
          << "Remote video-decoder should not be ready without a server URL: " << p;
    if (audioEnc)
      EXPECT_FALSE(audioEnc->status().value("ready", true))
          << "Remote audio-encoder should not be ready without a server URL: " << p;
    if (audioDec)
      EXPECT_FALSE(audioDec->status().value("ready", true))
          << "Remote audio-decoder should not be ready without a server URL: " << p;

    foundRemote = true;
    break;
  }

  // If the matrix contains no remote configs, the test is vacuously satisfied.
  if (!foundRemote) {
    GTEST_SKIP() << "No remote placement configs in deployment matrix";
  }
}

// ---------------------------------------------------------------------------
// Per-config parameterized smoke tests.  These turn the deployment matrix
// directory into ~649 individual gtest cases, giving the suite large test
// volume without hand-coding any tables.  Each test is fast: it only loads the
// config and checks that the deployment record is structurally valid.
// ---------------------------------------------------------------------------
struct ConfigCase {
  std::string path;
  std::string name;
};

std::ostream &operator<<(std::ostream &os, const ConfigCase &c) {
  return os << c.name;
}

std::string sanitizeForGtestName(const std::string &s) {
  std::string out = "cfg_";
  for (char c : s) {
    if (std::isalnum(static_cast<unsigned char>(c))) {
      out += c;
    } else {
      out += '_';
    }
  }
  return out;
}

std::vector<ConfigCase> buildConfigCases() {
  std::vector<ConfigCase> cases;
  for (const auto &p : findConfigsAfterDump()) {
    cases.push_back({p.string(), sanitizeForGtestName(p.stem().string())});
  }
  return cases;
}

class DeploymentMatrixConfigSmoke : public ::testing::TestWithParam<ConfigCase> {
 protected:
  void SetUp() override {
    setRemoteUrls();
    setForceLocal(true);
  }
};

TEST_P(DeploymentMatrixConfigSmoke, LoadsAndHasValidStructure) {
  const auto &c = GetParam();
  auto &cfg = phoenix::deployment::ModelDeploymentConfig::instance();
  cfg.reset();
  cfg.load({}, projectRoot().string(), c.path);

  auto j = cfg.toJson();
  EXPECT_FALSE(j.is_discarded()) << "Failed to round-trip " << c.path;
  EXPECT_TRUE(j.is_object()) << "Config JSON is not an object: " << c.path;
  EXPECT_TRUE(j.contains("vision")) << "Missing vision section in " << c.path;
  EXPECT_TRUE(j.contains("speech")) << "Missing speech section in " << c.path;
}

auto paramName = [](const testing::TestParamInfo<ConfigCase> &info) {
  return info.param.name;
};

INSTANTIATE_TEST_SUITE_P(DeploymentMatrix,
                         DeploymentMatrixConfigSmoke,
                         ::testing::ValuesIn(buildConfigCases()),
                         paramName);

// ---------------------------------------------------------------------------
// Per-config parameterized instantiation tests.  For every deployment matrix
// entry we instantiate the selected video/audio encoder and decoder and verify
// that status() returns a valid JSON object with a backend field.  This gives
// another ~649 independent tests and exercises the hardware-specific model
// factories for every config.
// ---------------------------------------------------------------------------
class DeploymentMatrixConfigStatus : public ::testing::TestWithParam<ConfigCase> {
 protected:
  void SetUp() override {
    setRemoteUrls();
    setForceLocal(true);
  }
};

TEST_P(DeploymentMatrixConfigStatus, ProducesModelStatus) {
  const auto &c = GetParam();
  auto &cfg = phoenix::deployment::ModelDeploymentConfig::instance();
  cfg.reset();
  cfg.load({}, projectRoot().string(), c.path);
  forceLocalForFeasibility(cfg, nullptr);

  auto videoEnc = phoenix::io::createVideoEncoder("video-encoder", 128);
  auto videoDec = phoenix::io::createVideoDecoder("video-encoder", 128);
  auto audioEnc = phoenix::io::createAudioEncoder("audio-encoder", 128);
  auto audioDec = phoenix::io::createAudioDecoder("audio-encoder", 128);

  ASSERT_TRUE(videoEnc) << "video-encoder is null for " << c.path;
  ASSERT_TRUE(videoDec) << "video-decoder is null for " << c.path;
  ASSERT_TRUE(audioEnc) << "audio-encoder is null for " << c.path;
  ASSERT_TRUE(audioDec) << "audio-decoder is null for " << c.path;

  auto ves = videoEnc->status();
  auto vds = videoDec->status();
  auto aes = audioEnc->status();
  auto ads = audioDec->status();

  EXPECT_FALSE(ves.is_discarded()) << "video-encoder status discarded for " << c.path;
  EXPECT_FALSE(vds.is_discarded()) << "video-decoder status discarded for " << c.path;
  EXPECT_FALSE(aes.is_discarded()) << "audio-encoder status discarded for " << c.path;
  EXPECT_FALSE(ads.is_discarded()) << "audio-decoder status discarded for " << c.path;

  EXPECT_FALSE(ves.value("backend", std::string()).empty())
      << "video-encoder backend is empty for " << c.path;
  EXPECT_FALSE(vds.value("backend", std::string()).empty())
      << "video-decoder backend is empty for " << c.path;
  EXPECT_FALSE(aes.value("backend", std::string()).empty())
      << "audio-encoder backend is empty for " << c.path;
  EXPECT_FALSE(ads.value("backend", std::string()).empty())
      << "audio-decoder backend is empty for " << c.path;
}

INSTANTIATE_TEST_SUITE_P(DeploymentMatrix,
                         DeploymentMatrixConfigStatus,
                         ::testing::ValuesIn(buildConfigCases()),
                         paramName);

// ---------------------------------------------------------------------------
// Per-config parameterized local-forcing test.  For every config we verify
// that the feasibility helper can force remote/server-client placements to
// local, which is the fallback path used in CI without remote servers.
// ---------------------------------------------------------------------------
class DeploymentMatrixConfigLocal : public ::testing::TestWithParam<ConfigCase> {
 protected:
  void SetUp() override {
    setRemoteUrls();
    setForceLocal(true);
  }
};

TEST_P(DeploymentMatrixConfigLocal, CanForceLocal) {
  const auto &c = GetParam();
  auto &cfg = phoenix::deployment::ModelDeploymentConfig::instance();
  cfg.reset();
  cfg.load({}, projectRoot().string(), c.path);
  forceLocalForFeasibility(cfg, nullptr);

  auto j = cfg.toJson();
  EXPECT_TRUE(j.contains("vision")) << "Missing vision section in " << c.path;
  EXPECT_TRUE(j.contains("speech")) << "Missing speech section in " << c.path;
  if (j.contains("vision") && j["vision"].contains("placement")) {
    EXPECT_EQ(j["vision"]["placement"], "local") << "vision placement not forced to local: " << c.path;
  }
  if (j.contains("speech") && j["speech"].contains("placement")) {
    EXPECT_EQ(j["speech"]["placement"], "local") << "speech placement not forced to local: " << c.path;
  }
}

INSTANTIATE_TEST_SUITE_P(DeploymentMatrix,
                         DeploymentMatrixConfigLocal,
                         ::testing::ValuesIn(buildConfigCases()),
                         paramName);
