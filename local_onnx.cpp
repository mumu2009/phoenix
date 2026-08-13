/* local_onnx.cpp - Shared C++->Python one-shot ONNX runner helpers
   Copyright (C) 2026 079 Project */

#include "local_onnx.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>

namespace phoenix {
namespace io {

std::filesystem::path temporaryOnnxPath() {
  static std::atomic<uint64_t> sequence{0};
  std::error_code ec;
  std::filesystem::path base;
  const char *env = std::getenv("PHOENIX_ONNX_TMP");
  if (env && *env) {
    base = std::filesystem::path(env);
  } else {
    base = std::filesystem::path("build") / "tmp";
  }
  std::filesystem::create_directories(base, ec);
  if (!std::filesystem::is_directory(base, ec)) {
    base = std::filesystem::temp_directory_path(ec);
    if (ec) base = ".";
  }
  return base / ("phoenix-onnx-" + std::to_string(sequence.fetch_add(1)));
}

std::string pythonExecutable() {
  std::error_code ec;
  const char *env = std::getenv("PHOENIX_PYTHON");
  if (env && *env) return env;
  const std::vector<std::string> candidates = {
      "Python314/python", "Python314/python.exe", "Python314/pythonw.exe",
      "python3", "python", "py"};
  for (const auto &c : candidates) {
    if (std::filesystem::is_regular_file(c, ec))
      return std::filesystem::absolute(c, ec).string();
  }
  const char *envPath = std::getenv("PATH");
  if (!envPath) return "python3";
  std::string path(envPath);
  std::istringstream iss(path);
  std::string dir;
  while (std::getline(iss, dir, ';')) {
    for (const auto &name : {"python.exe", "python3.exe", "python"}) {
      std::filesystem::path p(dir);
      p /= name;
      if (std::filesystem::is_regular_file(p, ec)) return p.string();
    }
  }
  return "python3";
}

std::string toShapeString(const std::vector<int> &shape) {
  std::ostringstream oss;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) oss << "x";
    oss << shape[i];
  }
  return oss.str();
}

nlohmann::json runLocalOnnx(
    const std::string &modelPath,
    const std::string &inputName,
    const std::vector<int> &inputShape,
    const std::vector<float> &inputFloats,
    const std::string &outputName,
    const std::vector<int> &outputShape,
    bool gpu) {
  std::error_code ec;
  const auto inPath = temporaryOnnxPath().replace_extension(".in");
  const auto outPath = temporaryOnnxPath().replace_extension(".out");

  {
    std::ofstream out(inPath, std::ios::binary | std::ios::trunc);
    if (!out) {
      return {{"ok", false}, {"error", "failed to write ONNX input binary"}};
    }
    out.write(reinterpret_cast<const char *>(inputFloats.data()),
              static_cast<std::streamsize>(inputFloats.size() * sizeof(float)));
  }

  const std::string py = pythonExecutable();
  if (py.find_first_of(" \t") != std::string::npos) {
    std::filesystem::remove(inPath, ec);
    return {{"ok", false}, {"error", "python executable path contains spaces; set PHOENIX_PYTHON"}};
  }
  auto quoteIfNeeded = [](const std::string &s) -> std::string {
    if (s.find_first_of(" \t\"&|<>^%;") == std::string::npos) return s;
    std::string out;
    out.push_back('\"');
    for (char c : s) {
      if (c == '\"') out.push_back('\"');
      out.push_back(c);
    }
    out.push_back('\"');
    return out;
  };

  std::ostringstream cmd;
  cmd << py << " "
      << "tools/local_onnx_runner.py "
      << "--model " << quoteIfNeeded(modelPath) << " "
      << "--input " << quoteIfNeeded(inPath.string()) << " "
      << "--input-name " << quoteIfNeeded(inputName) << " "
      << "--input-shape " << toShapeString(inputShape) << " "
      << "--output " << quoteIfNeeded(outPath.string()) << " "
      << "--output-name " << quoteIfNeeded(outputName) << " "
      << "--output-shape " << toShapeString(outputShape);
  if (gpu) cmd << " --gpu";

  std::string outputJson;
  FILE *pipe = _popen(cmd.str().c_str(), "r");
  if (!pipe) {
    std::filesystem::remove(inPath, ec);
    return {{"ok", false}, {"error", "failed to start local ONNX runner"}};
  }
  char buffer[4096];
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    outputJson += buffer;
  }
  const int rc = _pclose(pipe);
  std::filesystem::remove(inPath, ec);

  if (outputJson.empty()) {
    return {{"ok", false}, {"error", "local ONNX runner produced no output"}};
  }
  auto result = nlohmann::json::parse(outputJson, nullptr, false);
  if (result.is_discarded()) {
    return {{"ok", false}, {"error", "local ONNX runner returned invalid JSON"}, {"raw", outputJson}};
  }
  if (!result.value("ok", false) || rc != 0) {
    return result;
  }

  std::ifstream in(outPath, std::ios::binary);
  if (!in) {
    return {{"ok", false}, {"error", "local ONNX runner did not write output file"}};
  }
  const size_t expected = static_cast<size_t>(std::accumulate(outputShape.begin(), outputShape.end(), 1, std::multiplies<int>()));
  std::vector<float> outputFloats(expected);
  in.read(reinterpret_cast<char *>(outputFloats.data()),
          static_cast<std::streamsize>(expected * sizeof(float)));
  if (!in) {
    return {{"ok", false}, {"error", "failed to read ONNX output binary"}};
  }
  std::filesystem::remove(outPath, ec);
  result["floats"] = std::move(outputFloats);
  return result;
}

}  // namespace io
}  // namespace phoenix
