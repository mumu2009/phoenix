/* x5_bpu_smoke.cpp - Minimal RDK X5 BPU smoke test for a compiled .bin model
 *
 * Usage:
 *   ./x5_bpu_smoke <model.bin> [--run]
 *
 * The program loads the Horizon BPU model, prints its input/output tensor
 * properties, and (with --run) allocates input buffers, fills them with a
 * neutral value, and runs one forward pass, printing a few output values.
 */

#include <dnn/hb_dnn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int64_t tensorElements(const hbDNNTensorProperties &props) {
  int64_t n = 1;
  for (int i = 0; i < props.validShape.numDimensions; ++i) {
    n *= static_cast<int64_t>(props.validShape.dimensionSize[i]);
  }
  return n;
}

static void printShape(const hbDNNTensorProperties &props) {
  printf("[");
  for (int i = 0; i < props.validShape.numDimensions; ++i) {
    if (i) printf("x");
    printf("%d", props.validShape.dimensionSize[i]);
  }
  printf("]");
}

static void printTensorType(int32_t type) {
  switch (type) {
    case HB_DNN_TENSOR_TYPE_F32: printf("F32"); break;
    case HB_DNN_TENSOR_TYPE_S8:  printf("S8");  break;
    case HB_DNN_TENSOR_TYPE_U8:  printf("U8");  break;
    case HB_DNN_TENSOR_TYPE_F16: printf("F16"); break;
    default: printf("unknown(%d)", type); break;
  }
}

static uint16_t floatToHalf(float v) {
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
}

static float halfToFloat(uint16_t h) {
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
  return sign ? -val : val;
}

static float dequantizeValue(float raw, const hbDNNTensorProperties &props, int linearIndex) {
  if (props.tensorType == HB_DNN_TENSOR_TYPE_F32 || props.tensorType == HB_DNN_TENSOR_TYPE_F16) {
    return raw;
  }
  std::vector<int> strides(props.validShape.numDimensions, 1);
  for (int d = props.validShape.numDimensions - 2; d >= 0; --d) {
    strides[d] = strides[d + 1] * props.validShape.dimensionSize[d + 1];
  }
  int axis = props.quantizeAxis;
  if (axis < 0 || axis >= props.validShape.numDimensions) axis = 0;
  const int si = (props.scale.scaleLen > 1)
                     ? ((linearIndex / strides[axis]) % props.validShape.dimensionSize[axis])
                     : 0;
  const float scale = (props.scale.scaleData != nullptr && props.scale.scaleLen > 0)
                          ? props.scale.scaleData[si] : 1.0f;
  int zeroPoint = 0;
  if (props.scale.zeroPointData != nullptr) {
    if (props.tensorType == HB_DNN_TENSOR_TYPE_U8) {
      zeroPoint = static_cast<int>(reinterpret_cast<const uint8_t *>(props.scale.zeroPointData)[si]);
    } else {
      zeroPoint = static_cast<int>(props.scale.zeroPointData[si]);
    }
  }
  return (raw - static_cast<float>(zeroPoint)) * scale;
}

static void fillNeutralInput(hbDNNTensor &tensor) {
  const hbDNNTensorProperties &props = tensor.properties;
  const int64_t total = tensorElements(props);
  void *dst = tensor.sysMem[0].virAddr;
  if (dst == nullptr) return;

  int zeroPoint = 0;
  if (props.scale.zeroPointData != nullptr) {
    if (props.tensorType == HB_DNN_TENSOR_TYPE_U8) {
      zeroPoint = static_cast<int>(reinterpret_cast<const uint8_t *>(props.scale.zeroPointData)[0]);
    } else if (props.tensorType == HB_DNN_TENSOR_TYPE_S8) {
      zeroPoint = static_cast<int>(props.scale.zeroPointData[0]);
    }
  }

  switch (props.tensorType) {
    case HB_DNN_TENSOR_TYPE_F32: {
      float *out = static_cast<float *>(dst);
      for (int64_t i = 0; i < total; ++i) out[i] = 0.0f;
      break;
    }
    case HB_DNN_TENSOR_TYPE_U8: {
      uint8_t *out = static_cast<uint8_t *>(dst);
      for (int64_t i = 0; i < total; ++i) out[i] = static_cast<uint8_t>(std::max(0, std::min(255, zeroPoint)));
      break;
    }
    case HB_DNN_TENSOR_TYPE_S8: {
      int8_t *out = static_cast<int8_t *>(dst);
      for (int64_t i = 0; i < total; ++i) out[i] = static_cast<int8_t>(std::max(-128, std::min(127, zeroPoint)));
      break;
    }
    case HB_DNN_TENSOR_TYPE_F16: {
      uint16_t *out = static_cast<uint16_t *>(dst);
      const uint16_t half = floatToHalf(0.0f);
      for (int64_t i = 0; i < total; ++i) out[i] = half;
      break;
    }
  }
}

static void printOutputSummary(const hbDNNTensor &tensor, int maxValues = 8) {
  const hbDNNTensorProperties &props = tensor.properties;
  const int64_t total = tensorElements(props);
  const void *raw = tensor.sysMem[0].virAddr;
  printf("  output shape="); printShape(props); printf(" type="); printTensorType(props.tensorType);
  printf(" total=%lld alignedBytes=%u\n", (long long)total, props.alignedByteSize);
  if (raw == nullptr || total <= 0) return;

  const int count = static_cast<int>(std::min<int64_t>(maxValues, total));
  printf("  first %d values: [", count);
  for (int i = 0; i < count; ++i) {
    float v = 0.0f;
    switch (props.tensorType) {
      case HB_DNN_TENSOR_TYPE_F32: v = static_cast<const float *>(raw)[i]; break;
      case HB_DNN_TENSOR_TYPE_U8:  v = dequantizeValue(static_cast<float>(static_cast<const uint8_t *>(raw)[i]), props, i); break;
      case HB_DNN_TENSOR_TYPE_S8:  v = dequantizeValue(static_cast<float>(static_cast<const int8_t *>(raw)[i]), props, i); break;
      case HB_DNN_TENSOR_TYPE_F16: v = halfToFloat(static_cast<const uint16_t *>(raw)[i]); break;
    }
    if (i) printf(", ");
    printf("%.4f", v);
  }
  if (count < total) printf(", ...");
  printf("]\n");
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <model.bin> [--run]\n", argv[0]);
    return 1;
  }
  const char *modelPath = argv[1];
  bool doRun = (argc >= 3 && std::string(argv[2]) == "--run");

  hbPackedDNNHandle_t packedHandle = nullptr;
  const char *files[] = {modelPath};
  int32_t status = hbDNNInitializeFromFiles(&packedHandle, files, 1);
  if (status != 0 || packedHandle == nullptr) {
    fprintf(stderr, "hbDNNInitializeFromFiles failed for %s (status=%d)\n", modelPath, status);
    return 1;
  }

  char const **names = nullptr;
  int32_t modelCount = 0;
  status = hbDNNGetModelNameList(&names, &modelCount, packedHandle);
  if (status != 0 || modelCount < 1 || names == nullptr) {
    fprintf(stderr, "hbDNNGetModelNameList failed (status=%d)\n", status);
    return 1;
  }
  printf("modelCount=%d firstName=%s\n", modelCount, names[0]);

  hbDNNHandle_t modelHandle = nullptr;
  status = hbDNNGetModelHandle(&modelHandle, packedHandle, names[0]);
  if (status != 0 || modelHandle == nullptr) {
    fprintf(stderr, "hbDNNGetModelHandle failed (status=%d)\n", status);
    return 1;
  }

  int32_t inputCount = 0, outputCount = 0;
  hbDNNGetInputCount(&inputCount, modelHandle);
  hbDNNGetOutputCount(&outputCount, modelHandle);
  printf("inputCount=%d outputCount=%d\n", inputCount, outputCount);

  std::vector<hbDNNTensorProperties> inputProps(inputCount), outputProps(outputCount);
  printf("Inputs:\n");
  for (int i = 0; i < inputCount; ++i) {
    hbDNNGetInputTensorProperties(&inputProps[i], modelHandle, i);
    printf("  [%d] ", i); printShape(inputProps[i]); printf(" type="); printTensorType(inputProps[i].tensorType);
    printf(" total=%lld alignedBytes=%u\n", (long long)tensorElements(inputProps[i]), inputProps[i].alignedByteSize);
  }

  printf("Outputs:\n");
  for (int i = 0; i < outputCount; ++i) {
    hbDNNGetOutputTensorProperties(&outputProps[i], modelHandle, i);
    printf("  [%d] ", i); printShape(outputProps[i]); printf(" type="); printTensorType(outputProps[i].tensorType);
    printf(" total=%lld alignedBytes=%u\n", (long long)tensorElements(outputProps[i]), outputProps[i].alignedByteSize);
  }

  if (!doRun) {
    printf("Load OK. Use --run to execute a forward pass.\n");
    return 0;
  }

  std::vector<hbDNNTensor> inputTensors(inputCount), outputTensors(outputCount);
  for (int i = 0; i < inputCount; ++i) {
    inputTensors[i].properties = inputProps[i];
    status = hbSysAllocCachedMem(&inputTensors[i].sysMem[0], inputProps[i].alignedByteSize);
    if (status != 0 || inputTensors[i].sysMem[0].virAddr == nullptr) {
      fprintf(stderr, "hbSysAllocCachedMem for input %d failed (status=%d)\n", i, status);
      return 1;
    }
    fillNeutralInput(inputTensors[i]);
    hbSysFlushMem(&inputTensors[i].sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
  }

  for (int i = 0; i < outputCount; ++i) {
    outputTensors[i].properties = outputProps[i];
    status = hbSysAllocCachedMem(&outputTensors[i].sysMem[0], outputProps[i].alignedByteSize);
    if (status != 0 || outputTensors[i].sysMem[0].virAddr == nullptr) {
      fprintf(stderr, "hbSysAllocCachedMem for output %d failed (status=%d)\n", i, status);
      return 1;
    }
  }

  hbDNNInferCtrlParam control{};
  HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&control);
  hbDNNTaskHandle_t task = nullptr;
  hbDNNTensor *outputPtr = outputTensors.data();
  status = hbDNNInfer(&task, &outputPtr, &inputTensors[0], modelHandle, &control);
  if (status != 0 || task == nullptr) {
    fprintf(stderr, "hbDNNInfer failed (status=%d)\n", status);
    return 1;
  }
  status = hbDNNWaitTaskDone(task, 5000);
  if (status != 0) {
    fprintf(stderr, "hbDNNWaitTaskDone failed (status=%d)\n", status);
    return 1;
  }

  printf("Run OK. Outputs:\n");
  for (int i = 0; i < outputCount; ++i) {
    hbSysFlushMem(&outputTensors[i].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    printOutputSummary(outputTensors[i], 8);
  }

  for (auto &t : inputTensors) hbSysFreeMem(&t.sysMem[0]);
  for (auto &t : outputTensors) hbSysFreeMem(&t.sysMem[0]);
  if (task != nullptr) hbDNNReleaseTask(task);
  hbDNNRelease(packedHandle);
  return 0;
}
