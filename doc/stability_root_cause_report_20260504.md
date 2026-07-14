# Stability Root Cause Report 20260504

## Summary

This round closed the remaining stability and regression items around the gateway chat path, world model integration, and the last two failing unit tests.

Validated end state:

- The native transformer path is still retained internally, but public CLI and config selection paths no longer expose it.
- The current primary external inference path remains llama.cpp.
- The previous `tests-autoload + concurrent /api/chat` crash and timeout chain is no longer reproduced on the known trigger surface.
- The full project test matrix is back to green.

## Root Causes

### 1. Shared runtime state was not actually thread-safe on the real hot path

The original crash surface was not only a `RuntimeState` problem. The shared objects owned by it were also used concurrently without their own synchronization boundaries.

Affected shared surfaces:

- `KVMStore`
- `MemeGraph`
- `GraphTensorBridge`
- `PatternMatrix`

This showed up under the `tests-autoload` importer thread running concurrently with `/api/chat` requests. A coarse lock around the whole runtime stopped the access violation, but it also serialized too much work and converted the failure into starvation and request timeout.

### 2. External chat backends had no explicit inflight backpressure

After the first locking pass, the service stopped crashing on the same path but 8 concurrent chat requests could still stack behind the external backend until clients timed out.

On Windows this timeout and client disconnect path also surfaced as Trantor socket failures such as `errno=10053`, which made the incident look like a transport problem even though the real issue was unbounded overload.

### 3. `autonomy_cognition_tests` depended on live runtime brainstem state

`CognitionAutonomyManager` defaults to `runtime_store/brainstem_status.json` when brainstem policy is enabled. That meant the unit test could be polluted by a live or recent runtime brainstem policy file in the workspace.

When that happened, `observe()` could legitimately remove `plan-ground-route` from the action queue because the brainstem policy said `allowMove=false`, even though the test intended to validate the normal no-policy path.

### 4. `edge_platform_unit_tests` mixed two different concerns in one scenario

The failing third dispatch case was supposed to validate virtual-memory hot-weight progression, but the test still used the default compute inflight budget. The third request could therefore be rejected by `computeBacklog < effectiveMaxComputeInflight` before the test fully exercised the hot-weight path it actually cared about.

## Fixes Applied

### Runtime and gateway fixes

- Added internal synchronization to the shared runtime-side caches and graph/tensor helpers instead of relying only on an outer runtime lock.
- Changed `RuntimeState` to use parameter snapshots and shorter critical sections so normal chat work does not stay under a broad lock.
- Added explicit gateway inflight backpressure for external chat backends.
- When the provider is overloaded, `/api/chat` now returns `503 chat provider overloaded` immediately instead of allowing requests to pile up until client timeout.
- Updated the concurrency smoke script so controlled `503` overload responses are treated as expected busy results rather than false failures.

Primary implementation surfaces touched earlier in the session:

- `main_hub_parts/tail_parts/051_class_runtimestate.inc`
- `main_hub_parts/tail_parts/025_class_kvmstore.inc`
- `main_hub_parts/tail_parts/031_class_graphtensorbridge.inc`
- `main_hub_parts/tail_parts/035_class_patternmatrix.inc`
- `main_hub_parts/tail_parts/093_class_gatewayserver.inc`
- `main_hub_parts/tail_parts/098_section_tail.inc`
- `build/run_concurrency_smoke.ps1`

### Final regression fixes completed in this round

- `test/autonomy_cognition_tests.cpp`
  - The non-brainstem scenarios now explicitly disable brainstem policy so the test cannot read live workspace state by accident.
  - The dedicated brainstem policy scenario now explicitly enables brainstem policy, making its dependency clear and local.

- `test/edge_platform_unit_tests.cpp`
  - The synthetic manager setup now explicitly disables brainstem policy and raises `maxComputeInflight` for the generic synthetic test environment.
  - The dedicated brainstem policy test explicitly re-enables brainstem policy before validating brainstem hints.

These changes isolate each test to the behavior it is intended to validate.

## Validation

### Gateway and concurrency validation from the repaired crash surface

Previously verified in this session:

- Single `/api/chat` request succeeded with `provider=llamacpp` and `transformerMode=llamacpp`.
- 2 concurrent chat requests succeeded.
- 8-request concurrency smoke passed with:
  - `successCount=2`
  - `busyCount=6`
  - `timeoutCount=0`
  - `unexpectedFailureCount=0`
  - `crashLines=[]`

This confirms the old chain has been replaced by explicit capacity control:

- no reproduced access violation on the known trigger surface
- no reproduced timeout pile-up on the same smoke path
- overload is now surfaced as controlled `503` responses

### Final unit and matrix validation

Focused rerun completed in this round:

- `autonomy_cognition_tests`: passed
- `edge_platform_unit_tests`: passed

Full matrix rerun completed in this round:

- compile:
  - `world_model_scene_brain_tests:compile_ok`
  - `world_model_prompt_tests:compile_ok`
  - `world_model_learning_tests:compile_ok`
  - `component_config_tests:compile_ok`
  - `mechanical_mind_filter_tests:compile_ok`
  - `v51_runtime_tests:compile_ok`
  - `physics_world_runtime_tests:compile_ok`
  - `autonomy_cognition_tests:compile_ok`
  - `edge_platform_unit_tests:compile_ok`
  - `edge_platform_integration_tests:compile_ok`

- run:
  - `world_model_scene_brain_tests:run_ok`
  - `world_model_prompt_tests:run_ok`
  - `world_model_learning_tests:run_ok`
  - `component_config_tests:run_ok`
  - `mechanical_mind_filter_tests:run_ok`
  - `v51_runtime_tests:run_ok`
  - `physics_world_runtime_tests:run_ok`
  - `autonomy_cognition_tests:run_ok`
  - `edge_platform_unit_tests:run_ok`
  - `edge_platform_integration_tests:run_ok`

## Final Conclusion

The original incident was not resolved by simply observing that the process stayed alive for some time. The repaired state is supported by a root-cause chain and by targeted validation:

- unsafe shared runtime objects were given their own synchronization boundaries
- broad-lock starvation was reduced by moving to shorter runtime critical sections
- external backend overload now fails fast with explicit gateway backpressure
- the remaining two red tests were fixed by removing hidden environmental coupling from their setup

At the end of this round, the known crash surface is controlled, the last failing tests are repaired, and the full regression matrix is green.