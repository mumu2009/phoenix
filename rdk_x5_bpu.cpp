#include "rdk_x5_bpu.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <cstring>
#include <limits>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#ifndef PHOENIX_EDGE_IMAGE_ENABLED
#define PHOENIX_EDGE_IMAGE_ENABLED 1
#endif
#ifndef PHOENIX_EDGE_SPEECH_ENABLED
#define PHOENIX_EDGE_SPEECH_ENABLED 1
#endif

#if __has_include(<dnn/hb_dnn.h>)
#if PHOENIX_EDGE_IMAGE_ENABLED || PHOENIX_EDGE_SPEECH_ENABLED
#include <dnn/hb_dnn.h>
#define PHOENIX_RDK_X5_HAVE_HBDNN 1

// Older hb_dnn.h headers expose the quanti type enum values directly; newer
// versions might not define the legacy constants.  Provide fallbacks so the
// same source builds against both.
#ifndef HB_DNN_QUANTI_TYPE_SCALE
#define HB_DNN_QUANTI_TYPE_SCALE SCALE
#endif
#ifndef HB_DNN_QUANTI_TYPE_SHIFT
#define HB_DNN_QUANTI_TYPE_SHIFT SHIFT
#endif

#endif
#endif

namespace rdk_x5_bpu {

namespace {

namespace fs = std::filesystem;

std::string modelPath(const json &payload) {
    if (!payload.is_object()) {
        return {};
    }
    return payload.value("bpuModelPath", payload.value("horizonModelPath", std::string()));
}

std::string inputPath(const json &payload) {
    if (!payload.is_object()) {
        return {};
    }
    return payload.value("bpuInputPath", payload.value("inputPath", std::string()));
}

std::string inputFloatsPath(const json &payload) {
    if (!payload.is_object()) {
        return {};
    }
    return payload.value("bpuInputFloatsPath", payload.value("inputFloatsPath", std::string()));
}

bool devicePresent() {
    std::error_code ec;
    return fs::exists("/dev/bpu", ec) || fs::exists("/dev/bpu_core0", ec);
}

#ifdef PHOENIX_RDK_X5_HAVE_HBDNN
json outputSummary(hbDNNHandle_t modelHandle, hbDNNTensor *outputTensors, int maxValues) {
    int32_t outputCount = 0;
    int32_t status = hbDNNGetOutputCount(&outputCount, modelHandle);
    if (status != 0 || outputCount < 1) {
        return json{{"error", "hbDNNGetOutputCount failed: " + std::to_string(status)}};
    }

    json outputs = json::array();
    for (int32_t index = 0; index < outputCount; ++index) {
        hbDNNTensorProperties properties{};
        status = hbDNNGetOutputTensorProperties(&properties, modelHandle, index);
        if (status != 0) {
            return json{{"error", "hbDNNGetOutputTensorProperties failed: " + std::to_string(status)}};
        }
        hbSysFlushMem(&outputTensors[index].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
        json output{{"index", index},
                    {"bytes", properties.alignedByteSize},
                    {"tensorType", properties.tensorType},
                    {"shape", json::array()}};
        for (int dimension = 0; dimension < properties.validShape.numDimensions; ++dimension) {
            output["shape"].push_back(properties.validShape.dimensionSize[dimension]);
        }
        int elementSize = static_cast<int32_t>(sizeof(float));
        switch (properties.tensorType) {
            case HB_DNN_TENSOR_TYPE_S8:
            case HB_DNN_TENSOR_TYPE_U8:
                elementSize = 1;
                break;
            case HB_DNN_TENSOR_TYPE_F16:
                elementSize = static_cast<int32_t>(sizeof(uint16_t));
                break;
            default:
                break;
        }
        const int totalValues = properties.alignedByteSize / elementSize;
        const int count = std::min(maxValues, totalValues > 0 ? totalValues : maxValues);
        const void *raw = outputTensors[index].sysMem[0].virAddr;
        if (raw == nullptr || count <= 0) {
            outputs.push_back(std::move(output));
            continue;
        }
        output["values"] = json::array();
        output["quantiType"] = properties.quantiType;
        output["quantizeAxis"] = properties.quantizeAxis;

        // Precompute row-major strides for the valid shape to map a linear index
        // to the per-axis scale/shift coordinate.
        std::vector<int> strides(properties.validShape.numDimensions, 1);
        for (int d = properties.validShape.numDimensions - 2; d >= 0; --d) {
            strides[d] = strides[d + 1] * properties.validShape.dimensionSize[d + 1];
        }

        auto quantIndex = [&](int linearIndex) -> int {
            const int axis = properties.quantizeAxis;
            if (axis < 0 || axis >= properties.validShape.numDimensions) return 0;
            return (linearIndex / strides[axis]) % properties.validShape.dimensionSize[axis];
        };

        auto dequant = [&](float raw, int linearIndex) -> float {
            switch (properties.quantiType) {
                case HB_DNN_QUANTI_TYPE_SCALE: {
                    if (properties.scale.scaleData == nullptr || properties.scale.scaleLen <= 0) return raw;
                    const int si = properties.scale.scaleLen > 1 ? quantIndex(linearIndex) : 0;
                    const float scale = properties.scale.scaleData[si];
                    int zeroPoint = 0;
                    if (properties.scale.zeroPointData != nullptr) {
                        if (properties.tensorType == HB_DNN_TENSOR_TYPE_U8) {
                            zeroPoint = static_cast<int>(reinterpret_cast<const uint8_t *>(properties.scale.zeroPointData)[si]);
                        } else {
                            zeroPoint = static_cast<int>(properties.scale.zeroPointData[si]);
                        }
                    }
                    return (raw - static_cast<float>(zeroPoint)) * scale;
                }
                case HB_DNN_QUANTI_TYPE_SHIFT: {
                    if (properties.shift.shiftData == nullptr || properties.shift.shiftLen <= 0) return raw;
                    const int si = properties.shift.shiftLen > 1 ? quantIndex(linearIndex) : 0;
                    const int shift = static_cast<int>(properties.shift.shiftData[si]);
                    return raw / std::ldexp(1.0f, shift);
                }
                default:
                    return raw;
            }
        };

        auto append = [&](float v) { output["values"].push_back(v); };
        switch (properties.tensorType) {
            case HB_DNN_TENSOR_TYPE_F32: {
                const float *values = static_cast<const float *>(raw);
                for (int i = 0; i < count; ++i) append(values[i]);
                break;
            }
            case HB_DNN_TENSOR_TYPE_S8: {
                const int8_t *values = static_cast<const int8_t *>(raw);
                for (int i = 0; i < count; ++i) append(dequant(static_cast<float>(values[i]), i));
                break;
            }
            case HB_DNN_TENSOR_TYPE_U8: {
                const uint8_t *values = static_cast<const uint8_t *>(raw);
                for (int i = 0; i < count; ++i) append(dequant(static_cast<float>(values[i]), i));
                break;
            }
            case HB_DNN_TENSOR_TYPE_F16: {
                const uint16_t *values = static_cast<const uint16_t *>(raw);
                for (int i = 0; i < count; ++i) {
                    // Simple F16 -> F32 conversion (not NaN/Inf aware).
                    const uint16_t h = values[i];
                    const int sign = (h >> 15) & 0x1;
                    const int exp = (h >> 10) & 0x1f;
                    const int mant = h & 0x3ff;
                    float val = 0.0f;
                    if (exp == 0) {
                        val = std::ldexp(static_cast<float>(mant), -24);
                    } else if (exp == 31) {
                        val = (mant == 0) ? std::numeric_limits<float>::infinity() : std::numeric_limits<float>::quiet_NaN();
                    } else {
                        val = std::ldexp(static_cast<float>(mant + 1024), exp - 25);
                    }
                    append(sign ? -val : val);
                }
                break;
            }
            default:
                output["error"] = "unsupported BPU output tensor type: " + std::to_string(properties.tensorType);
                break;
        }
        outputs.push_back(std::move(output));
    }
    return outputs;
}
#endif

} // namespace

bool available() {
#ifdef PHOENIX_RDK_X5_HAVE_HBDNN
    return devicePresent();
#else
    return false;
#endif
}

bool requested(const json &payload) {
    return !modelPath(payload).empty();
}

json inspect(const json &payload) {
    const std::string path = modelPath(payload);
    std::error_code ec;
    return json{{"requested", !path.empty()},
                {"available", available()},
#ifdef PHOENIX_RDK_X5_HAVE_HBDNN
                {"runtime", "hbDNN 1.24"},
#else
                {"runtime", "hbDNN headers unavailable"},
#endif
                {"modelPath", path},
                {"modelExists", !path.empty() && fs::is_regular_file(path, ec)}};
}

#ifdef PHOENIX_RDK_X5_HAVE_HBDNN
static std::string writeBpuInputFromFloats(const hbDNNTensorProperties &props,
                                           const std::vector<float> &floats,
                                           void *dst) {
    int elementSize = 0;
    switch (props.tensorType) {
        case HB_DNN_TENSOR_TYPE_F32:
            elementSize = static_cast<int>(sizeof(float));
            break;
        case HB_DNN_TENSOR_TYPE_S8:
        case HB_DNN_TENSOR_TYPE_U8:
            elementSize = 1;
            break;
        case HB_DNN_TENSOR_TYPE_F16:
            elementSize = static_cast<int>(sizeof(uint16_t));
            break;
        default:
            return "unsupported BPU input tensor type: " + std::to_string(props.tensorType);
    }

    int validTotal = 1;
    for (int d = 0; d < props.validShape.numDimensions; ++d) {
        validTotal *= props.validShape.dimensionSize[d];
    }
    if (validTotal <= 0) {
        return "invalid BPU input shape";
    }
    if (static_cast<int>(floats.size()) < validTotal) {
        return "input float count too small (expected " + std::to_string(validTotal) + ", got " + std::to_string(floats.size()) + ")";
    }
    if (validTotal * elementSize > props.alignedByteSize) {
        return "BPU input valid size exceeds aligned buffer";
    }

    std::memset(dst, 0, static_cast<std::size_t>(props.alignedByteSize));

    std::vector<int> strides(props.validShape.numDimensions, 1);
    for (int d = props.validShape.numDimensions - 2; d >= 0; --d) {
        strides[d] = strides[d + 1] * props.validShape.dimensionSize[d + 1];
    }

    auto quantIndex = [&](int linearIndex) -> int {
        const int axis = props.quantizeAxis;
        if (axis < 0 || axis >= props.validShape.numDimensions) return 0;
        return (linearIndex / strides[axis]) % props.validShape.dimensionSize[axis];
    };

    auto quantize = [&](float f, int linearIndex) -> float {
        if (props.tensorType != HB_DNN_TENSOR_TYPE_S8 &&
            props.tensorType != HB_DNN_TENSOR_TYPE_U8) {
            return f;
        }
        switch (props.quantiType) {
            case HB_DNN_QUANTI_TYPE_SCALE: {
                if (props.scale.scaleData == nullptr || props.scale.scaleLen <= 0) return f;
                const int si = props.scale.scaleLen > 1 ? quantIndex(linearIndex) : 0;
                const float scale = props.scale.scaleData[si];
                int zeroPoint = 0;
                if (props.scale.zeroPointData != nullptr) {
                    if (props.tensorType == HB_DNN_TENSOR_TYPE_U8) {
                        zeroPoint = static_cast<int>(reinterpret_cast<const uint8_t *>(props.scale.zeroPointData)[si]);
                    } else {
                        zeroPoint = static_cast<int>(props.scale.zeroPointData[si]);
                    }
                }
                return f / scale + static_cast<float>(zeroPoint);
            }
            case HB_DNN_QUANTI_TYPE_SHIFT: {
                if (props.shift.shiftData == nullptr || props.shift.shiftLen <= 0) return f;
                const int si = props.shift.shiftLen > 1 ? quantIndex(linearIndex) : 0;
                const int shift = static_cast<int>(props.shift.shiftData[si]);
                return f * std::ldexp(1.0f, shift);
            }
            default:
                return f;
        }
    };

    auto floatToHalf = [](float v) -> uint16_t {
        const uint32_t b = *reinterpret_cast<const uint32_t *>(&v);
        const uint32_t sign = (b >> 31) & 0x1;
        const int32_t exp = static_cast<int32_t>((b >> 23) & 0xff) - 127 + 15;
        const uint32_t mant = b & 0x7fffff;
        uint16_t h;
        if ((b & 0x7fffffff) == 0) {
            h = static_cast<uint16_t>(sign << 15);
        } else if (exp >= 31) {
            h = static_cast<uint16_t>((sign << 15) | (0x1f << 10));
        } else if (exp <= 0) {
            if (exp < -10) {
                h = static_cast<uint16_t>(sign << 15);
            } else {
                const uint32_t m = (mant | 0x800000) >> (1 - exp);
                h = static_cast<uint16_t>((sign << 15) | (m >> 13));
            }
        } else {
            const uint32_t m = (mant + 0x00001000u + ((mant >> 12) & 1u)) >> 13;
            if (m >= 1024u) {
                h = static_cast<uint16_t>((sign << 15) | ((exp + 1) << 10));
            } else {
                h = static_cast<uint16_t>((sign << 15) | (exp << 10) | m);
            }
        }
        return h;
    };

    switch (props.tensorType) {
        case HB_DNN_TENSOR_TYPE_F32: {
            float *out = static_cast<float *>(dst);
            for (int i = 0; i < validTotal; ++i) out[i] = quantize(floats[i], i);
            break;
        }
        case HB_DNN_TENSOR_TYPE_S8: {
            int8_t *out = static_cast<int8_t *>(dst);
            for (int i = 0; i < validTotal; ++i) {
                const float q = quantize(floats[i], i);
                const int rounded = static_cast<int>(std::round(q));
                out[i] = static_cast<int8_t>(std::max(-128, std::min(127, rounded)));
            }
            break;
        }
        case HB_DNN_TENSOR_TYPE_U8: {
            uint8_t *out = static_cast<uint8_t *>(dst);
            for (int i = 0; i < validTotal; ++i) {
                const float q = quantize(floats[i], i);
                const int rounded = static_cast<int>(std::round(q));
                out[i] = static_cast<uint8_t>(std::max(0, std::min(255, rounded)));
            }
            break;
        }
        case HB_DNN_TENSOR_TYPE_F16: {
            uint16_t *out = static_cast<uint16_t *>(dst);
            for (int i = 0; i < validTotal; ++i) out[i] = floatToHalf(quantize(floats[i], i));
            break;
        }
        default:
            return "unsupported BPU input tensor type: " + std::to_string(props.tensorType);
    }
    return {};
}
#endif

json execute(const json &payload) {
#ifndef PHOENIX_RDK_X5_HAVE_HBDNN
    return json{{"executed", false}, {"driver", "horizon-hbdnn"}, {"error", "hbDNN headers are unavailable at build time"}};
#else
    const std::string model = modelPath(payload);
    const std::string rawInput = inputPath(payload);
    const std::string floatsInput = inputFloatsPath(payload);
    if (model.empty()) {
        return json{{"executed", false}, {"driver", "horizon-hbdnn"}, {"error", "bpuModelPath is required"}};
    }
    if (rawInput.empty() && floatsInput.empty()) {
        return json{{"executed", false}, {"driver", "horizon-hbdnn"}, {"error", "bpuInputPath or bpuInputFloatsPath is required"}};
    }
    if (!available()) {
        return json{{"executed", false}, {"driver", "horizon-hbdnn"}, {"error", "RDK X5 BPU device nodes are unavailable"}};
    }

    std::vector<char> rawBytes;
    std::vector<float> floatValues;
    if (!rawInput.empty()) {
        std::ifstream inputFile(rawInput, std::ios::binary);
        if (!inputFile) {
            return json{{"executed", false}, {"driver", "horizon-hbdnn"}, {"error", "failed to open input: " + rawInput}};
        }
        rawBytes = std::vector<char>((std::istreambuf_iterator<char>(inputFile)), std::istreambuf_iterator<char>());
        if (rawBytes.empty()) {
            return json{{"executed", false}, {"driver", "horizon-hbdnn"}, {"error", "input is empty"}};
        }
    } else {
        std::ifstream inputFile(floatsInput, std::ios::binary);
        if (!inputFile) {
            return json{{"executed", false}, {"driver", "horizon-hbdnn"}, {"error", "failed to open input floats: " + floatsInput}};
        }
        const std::vector<char> fileBytes((std::istreambuf_iterator<char>(inputFile)), std::istreambuf_iterator<char>());
        if (fileBytes.empty()) {
            return json{{"executed", false}, {"driver", "horizon-hbdnn"}, {"error", "input floats are empty"}};
        }
        if (fileBytes.size() % sizeof(float) != 0) {
            return json{{"executed", false}, {"driver", "horizon-hbdnn"}, {"error", "input floats file size is not a multiple of sizeof(float)"}};
        }
        const std::size_t count = fileBytes.size() / sizeof(float);
        floatValues.resize(count);
        std::memcpy(floatValues.data(), fileBytes.data(), fileBytes.size());
    }

    hbPackedDNNHandle_t packed = nullptr;
    const char *modelFiles[] = {model.c_str()};
    int32_t status = hbDNNInitializeFromFiles(&packed, modelFiles, 1);
    if (status != 0 || packed == nullptr) {
        return json{{"executed", false}, {"driver", "horizon-hbdnn"}, {"error", "hbDNNInitializeFromFiles failed: " + std::to_string(status)}};
    }
    struct PackedGuard { hbPackedDNNHandle_t handle; ~PackedGuard() { hbDNNRelease(handle); } } packedGuard{packed};

    char const **names = nullptr;
    int32_t modelCount = 0;
    status = hbDNNGetModelNameList(&names, &modelCount, packed);
    if (status != 0 || modelCount < 1 || names == nullptr) {
        return json{{"executed", false}, {"driver", "horizon-hbdnn"}, {"error", "hbDNNGetModelNameList failed: " + std::to_string(status)}};
    }
    const std::string requestedName = payload.value("bpuModelName", std::string());
    const char *name = names[0];
    for (int32_t index = 0; index < modelCount; ++index) {
        if (requestedName == names[index]) {
            name = names[index];
            break;
        }
    }

    hbDNNHandle_t modelHandle = nullptr;
    status = hbDNNGetModelHandle(&modelHandle, packed, name);
    int32_t inputCount = 0;
    if (status != 0 || hbDNNGetInputCount(&inputCount, modelHandle) != 0 || inputCount != 1) {
        return json{{"executed", false}, {"driver", "horizon-hbdnn"}, {"error", "only valid single-input BPU models are supported"}};
    }
    hbDNNTensorProperties inputProperties{};
    status = hbDNNGetInputTensorProperties(&inputProperties, modelHandle, 0);
    if (status != 0 || inputProperties.alignedByteSize <= 0) {
        return json{{"executed", false}, {"driver", "horizon-hbdnn"}, {"error", "failed to read BPU input properties: " + std::to_string(status)}};
    }

    hbDNNTensor inputTensor{};
    inputTensor.properties = inputProperties;
    status = hbSysAllocCachedMem(&inputTensor.sysMem[0], static_cast<uint32_t>(inputProperties.alignedByteSize));
    if (status != 0 || inputTensor.sysMem[0].virAddr == nullptr) {
        return json{{"executed", false}, {"driver", "horizon-hbdnn"}, {"error", "hbSysAllocCachedMem failed: " + std::to_string(status)}};
    }
    struct MemoryGuard { hbSysMem *memory; ~MemoryGuard() { hbSysFreeMem(memory); } } memoryGuard{&inputTensor.sysMem[0]};
    if (!floatsInput.empty()) {
        std::string inputError = writeBpuInputFromFloats(inputProperties, floatValues, inputTensor.sysMem[0].virAddr);
        if (!inputError.empty()) {
            return json{{"executed", false}, {"driver", "horizon-hbdnn"}, {"error", inputError}, {"inputTensorType", inputProperties.tensorType}};
        }
    } else {
        if (rawBytes.size() != static_cast<std::size_t>(inputProperties.alignedByteSize)) {
            return json{{"executed", false}, {"driver", "horizon-hbdnn"}, {"error", "input byte count must equal model aligned input size"}, {"expectedBytes", inputProperties.alignedByteSize}, {"actualBytes", rawBytes.size()}};
        }
        std::memcpy(inputTensor.sysMem[0].virAddr, rawBytes.data(), rawBytes.size());
    }
    status = hbSysFlushMem(&inputTensor.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
    if (status != 0) {
        return json{{"executed", false}, {"driver", "horizon-hbdnn"}, {"error", "hbSysFlushMem failed: " + std::to_string(status)}};
    }

    int32_t outputCount = 0;
    status = hbDNNGetOutputCount(&outputCount, modelHandle);
    if (status != 0 || outputCount < 1) {
        return json{{"executed", false}, {"driver", "horizon-hbdnn"}, {"error", "hbDNNGetOutputCount failed: " + std::to_string(status)}};
    }
    std::vector<hbDNNTensor> outputStorage(static_cast<std::size_t>(outputCount));
    for (int32_t index = 0; index < outputCount; ++index) {
        status = hbDNNGetOutputTensorProperties(&outputStorage[index].properties, modelHandle, index);
        if (status != 0 || outputStorage[index].properties.alignedByteSize <= 0) {
            return json{{"executed", false}, {"driver", "horizon-hbdnn"}, {"error", "hbDNNGetOutputTensorProperties failed: " + std::to_string(status)}};
        }
        status = hbSysAllocCachedMem(&outputStorage[index].sysMem[0], static_cast<uint32_t>(outputStorage[index].properties.alignedByteSize));
        if (status != 0 || outputStorage[index].sysMem[0].virAddr == nullptr) {
            for (int32_t allocated = 0; allocated < index; ++allocated) {
                hbSysFreeMem(&outputStorage[allocated].sysMem[0]);
            }
            return json{{"executed", false}, {"driver", "horizon-hbdnn"}, {"error", "hbSysAllocCachedMem for output failed: " + std::to_string(status)}};
        }
    }
    struct OutputsGuard {
        std::vector<hbDNNTensor> &tensors;
        ~OutputsGuard() { for (auto &tensor : tensors) { if (tensor.sysMem[0].virAddr != nullptr) hbSysFreeMem(&tensor.sysMem[0]); } }
    } outputsGuard{outputStorage};

    hbDNNInferCtrlParam control{};
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&control);
    control.bpuCoreId = payload.value("bpuCoreId", static_cast<int>(HB_BPU_CORE_ANY));
    control.priority = payload.value("bpuPriority", static_cast<int>(HB_DNN_PRIORITY_LOWEST));
    hbDNNTaskHandle_t task = nullptr;
    hbDNNTensor *outputs = outputStorage.data();
    const auto started = std::chrono::steady_clock::now();
    status = hbDNNInfer(&task, &outputs, &inputTensor, modelHandle, &control);
    if (status != 0 || task == nullptr || outputs == nullptr) {
        return json{{"executed", false}, {"driver", "horizon-hbdnn"}, {"error", "hbDNNInfer failed: " + std::to_string(status)}};
    }
    struct TaskGuard { hbDNNTaskHandle_t task; ~TaskGuard() { hbDNNReleaseTask(task); } } taskGuard{task};
    status = hbDNNWaitTaskDone(task, std::max(1, payload.value("bpuTimeoutMs", 5000)));
    const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    if (status != 0) {
        return json{{"executed", false}, {"driver", "horizon-hbdnn"}, {"error", "hbDNNWaitTaskDone failed: " + std::to_string(status)}, {"totalElapsedMs", elapsed}};
    }
    json outputsJson = outputSummary(modelHandle, outputs, std::clamp(payload.value("maxBpuOutputValues", 128), 0, 4096));
    if (outputsJson.is_object() && outputsJson.contains("error")) {
        return json{{"executed", false}, {"driver", "horizon-hbdnn"}, {"error", outputsJson["error"]}};
    }
    return json{{"executed", true}, {"driver", "horizon-hbdnn"}, {"model", name}, {"bpuCoreId", control.bpuCoreId}, {"totalElapsedMs", elapsed}, {"outputs", outputsJson}};
#endif
}

} // namespace rdk_x5_bpu
