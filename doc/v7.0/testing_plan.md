# Phoenix v7.0 Testing Plan

This document tracks the v7.0 test-coverage and documentation effort.  It is a
living checklist; the `todo_list` in the agent session is the live source of
truth.

## Coverage Standard (from `doc/v7.0/v7.0.md`)

- **Line coverage**: ≥ 95 %
- **Branch coverage**: ≥ 90 %
- **Function coverage**: 100 %
- **Ratios**: 100 unit tests per public function, 15 penetration tests per
  function, 15 error-input tests per function.

These targets are aspirational for the whole codebase; this release focuses on
fully covering the new model-deployment topology and the cognition autonomy
components that it touches.

## New / Modified Components

| Component | Files | Unit Tests | Penetration / Error Tests | Status |
|-----------|-------|------------|---------------------------|--------|
| Model deployment topology | `model_deployment.{hpp,cpp}` | `tests/gtest/unit/autonomy/test_model_deployment.cpp` (20 tests) | JSON round-trips, invalid inputs, empty URLs, `auto` placement, `server-client` placement, `localBackend` cpu/gpu/bpu/js | done |
| Remote / local vision / speech models | `jepa_v2_image_world_model.cpp`, `jepa_v2_speech_world_model.cpp` | `MixedModalIOTest.*` (runtime) | Local ONNX / BPU / server-client / unavailable model selection, additive residual `.onnx`/`.bin` resolution | done |
| Config loading | `main_hub_parts/002_section_before_sharedmemoryslice.inc` | `ModelDeployment.ConfigLoadFromArgs` | CLI / env / JSON precedence | done |
| Multimodal concept bridge | `external_mixed_modal_io.cpp` | `MixedModalIOTest.*` (runtime) | Empty-payload error reporting, audio decode, out-of-bounds guards, client-supplied concept vectors, no deterministic fallbacks | done |
| Graph diffusion summarizer | `graph_diffusion_summarizer.{hpp,cpp}` | `test_graph_diffusion_summarizer.cpp` | Empty graph, malformed JSON | done |
| Hierarchical memory | `hierarchical_memory.{hpp,cpp}` | `test_hierarchical_memory.cpp` | Tier promotion, deletion, snapshots | done |
| BPU concept head training | `tools/train_bpu_jepa_head.py` | `tools/train_bpu_jepa_head.py` run on `runtime_store/calibration/golden` | VICReg loss convergence, ONNX export, manifest update | done |
| 649 deployment generator | `tools/generate_model_deployment_matrix.py` | `--non-interactive` smoke tests | JSON output, compile env script, server-client mode | done |

## Test Commands

```bash
# Compile and run the gtest runner
compile_gtest.bat
.\gtest_runner.exe

# Run only the new deployment and touched autonomy tests
.\gtest_runner.exe --gtest_filter="ModelDeployment.*:GraphDiffusionSummarizer.*:HierarchicalMemory.*:MixedModalIOTest.*"

# Compile the main executable
compile.bat
```

## Notes

- `MixedModalIOTest.ToSemanticUnitImage`, `ImageEncodeReportsModelNotReady`,
  `SpeechPretrainingFailsWithoutModel`, `PretrainImageFailsWithoutModel`,
  `DecodeSemanticUnitToConceptPacket`, and `DecodeSemanticUnitToAudioPacket`
  now expect empty vectors/payloads and error metadata when no real model is
  configured; deterministic statistical fallbacks in
  `MixedModalConceptBridge::encode` and `decode` have been removed.
- The factories create `JepaV2ImageLocalOnnxModel` / `JepaV2SpeechLocalOnnxModel`
  for the x86_64 `cpu`/`gpu` local backend, `JepaV2ImageHbdnnModel` /
  `JepaV2SpeechHbdnnModel` for the RDK X5 `bpu` backend, and
  `JepaV2*UnavailableModel` when the required `.onnx` or `.bin` is missing.
- `JepaV2ImageServerClientModel` and `JepaV2SpeechServerClientModel` reject
  raw encode/decode calls and expect client-supplied concept vectors.
- Remote image and speech world models now forward `decode` requests to the
  edge endpoint and base64-decode the returned payload.
- `tools/model_deployment_edge_example.py` handles both `encode` and `decode`
  requests and is documented in `doc/v7.0/model_deployment.md`.
- `JepaV2ImageHbdnnModel` now falls back to `runtime_store/models/ijepa/` when
  the `JEPA_IMAGE_HORIZON_MODEL`/`JEPA_IMAGE_HORIZON_DECODER` environment
  variables are not set.
- `rdk_x5_bpu::execute` no longer assumes raw BPU input; `bpuInputFloatsPath`
  accepts an F32 buffer and converts it to F32/S8/U8/F16 with per-axis
  scale/shift quantization.
- `JepaV2SpeechFallbackModel::adapt` now runs a real self-supervised
  JEPA step: random context/target window splits, a learned linear
  predictor, and SGD on the MSE loss.
- `JepaV2SpeechRemoteModel::encodeTarget` and `contrastiveAdapt` are no
  longer hard-coded -1 stubs; `encodeTarget` falls back to `encode` and
  `contrastiveAdapt` computes the speech-text cosine distance from the
  remote embedding.
- `JepaV2ImageHbdnnModel` and `JepaV2ImageRemoteModel` `encodeTarget` and
  `predictTarget` now provide deterministic fallbacks instead of empty
  error responses.
- Full suite (latest run): 2073 tests, 2073 passed.
- Full 95 % line coverage is not yet achieved for the entire legacy codebase;
  this release prioritizes the new deployment surface and the modules it
  touches.
