# Deployment and Development Report

## Completed

- The exFAT volume is persistently mounted at `/media/KINGSTON` with `sunrise` ownership, directory mode `755`, and file mode `644` through the UUID `2CB9-7531` entry in `/etc/fstab`.
- `/vision/analyze` now emits a JEPA image-world-model concept vector and persists it into the world model. It no longer invokes the legacy `VisionPipeline` YOLO/CNN route.
- The advanced GNN-GA learner caches a stable graph execution plan keyed by a structural fingerprint. The cached plan contains normalized adjacency, Laplacian, random-walk embedding, spectral features, semantic analysis, community detection, and motifs.
- GNN-GA now resets its Pareto archive for each evolution and reports normalization, constraint projection, and spectral-radius invariant correction.
- The RDK X5 BPU devices and hbDNN development headers are present and accessible to `sunrise`.
- The temporary X5-native llama-server validation deployment and its dedicated swapfile were removed after confirming that the 8B CPU model is functional but unsuitable for interactive X5 use.
- The agreed runtime split is host llama-server for LLM inference and X5 for hbDNN/BPU vision and encoder models plus CPU GNN control work.
- Board configuration keeps `external_auto_launch` disabled so it cannot attempt to start the incompatible Windows llama-server executable.
- The host deployment procedure and board handoff contract are maintained in `doc/host_llama_server_handoff.md`.

## Validated

- exFAT mount configuration validates with `findmnt --verify --tab-file=/etc/fstab`.
- The project runtime JSON files parse successfully.
- `video_model.cpp`, `rdk_x5_bpu.cpp`, and `module_overrides/gnn_ga_learner_advanced.cpp` pass local C++ syntax validation.
- No `/vision/analyze` call site remains that invokes `visionPipeline.analyze`.
- A temporary native X5 validation completed `/health` and `/completion` successfully. The 8B Q4_K model reached approximately 4.45 GiB RSS and used approximately 155 MiB swap without an OOM event; this validation was then removed in favor of host inference.

## Blocking Deployment Issues

- No Horizon `.bin` or `.hbm` I-JEPA/JEPA model is present. The current JEPA implementation reports its fallback backend, so it must not be represented as BPU execution. A compatible quantized Horizon model and its exact input layout are required before wiring this route to `rdk_x5_bpu::execute`.
- The configured Llama 3.1 GGUF exists and is 4,920,738,944 bytes. The 4,685.30 MiB historical value is the CPU-mapped file buffer, not the process RSS. On the X5 the completed local request reached about 4.45 GiB RSS because mmap pages are resident on demand.
- The bundled `llama-server.exe` and `phoenix_main.exe` remain x86-64 Windows PE binaries. The board must not start them; its runtime must use the reachable host llama-server URL.
- The X5 validation measured prompt evaluation at about 0.058 tokens/second, confirming that 8B CPU inference is functional but not interactive on Cortex-A55.

## Required Host Actions

1. Provide the host LAN IP address or resolvable DNS name, then set `llamacpp_base_url` to `http://<HOST_LAN_ADDRESS>:8082` as documented in `doc/host_llama_server_handoff.md`.
2. Start host llama-server, verify `/health` locally and from X5, and retain host-side RSS and latency monitoring.
3. Convert and deploy a supported JEPA/I-JEPA model to Horizon format, then provide its model path, tensor layout, quantization, and preprocessing contract.
4. Build the full Phoenix gateway for aarch64 Linux if the gateway itself is intended to run on the X5.

## Safety Note

Do not use `chmod` to manage ownership on exFAT. Ownership and permission presentation are mount options; the persistent `/etc/fstab` entry is the applicable fix.
