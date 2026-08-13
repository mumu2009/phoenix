/* local_onnx.hpp - Shared C++->Python one-shot ONNX runner helpers
   Copyright (C) 2026 079 Project */

#ifndef PHOENIX_LOCAL_ONNX_HPP
#define PHOENIX_LOCAL_ONNX_HPP

#include <nlohmann/json.hpp>
#include <filesystem>
#include <string>
#include <vector>

namespace phoenix {
namespace io {

std::filesystem::path temporaryOnnxPath();
std::string pythonExecutable();
std::string toShapeString(const std::vector<int> &shape);
nlohmann::json runLocalOnnx(
    const std::string &modelPath,
    const std::string &inputName,
    const std::vector<int> &inputShape,
    const std::vector<float> &inputFloats,
    const std::string &outputName,
    const std::vector<int> &outputShape,
    bool gpu);

}  // namespace io
}  // namespace phoenix

#endif  // PHOENIX_LOCAL_ONNX_HPP
