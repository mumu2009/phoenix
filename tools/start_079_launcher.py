from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def bool_str(value: bool) -> str:
    return "true" if value else "false"


def split_list(raw: str) -> list[str]:
    normalized = raw.replace(";", ",").replace("\n", ",")
    items = [item.strip() for item in normalized.split(",")]
    return [item for item in items if item]


def parse_extra_args(raw: str) -> list[str]:
    text = raw.strip()
    if not text:
        return []
    return shlex.split(text, posix=False)


def detect_repo_root(start: Path | None = None) -> Path:
    probe = (start or Path(sys.executable if getattr(sys, "frozen", False) else __file__)).resolve()
    current = probe.parent if probe.is_file() else probe
    for candidate in [current, *current.parents]:
        if (candidate / "phoenix_main.exe").exists() or (candidate / "start_079_oneclick.bat").exists():
            return candidate
    return current


@dataclass(frozen=True)
class OptionField:
    key: str
    label: str
    kind: str
    default: Any
    tab: str
    section: str
    cli: str | None = None
    choices: tuple[str, ...] = ()
    width: int = 52
    rows: int = 4
    emit_default: bool = False
    invert: bool = False


TAB_ORDER = ["General", "Inference", "Autonomy", "World", "Safety", "Addons", "Expert"]


OPTION_FIELDS: list[OptionField] = [
    OptionField("internal_name", "Internal Name", "string", "079", "General", "Launcher"),
    OptionField("phoenix_executable", "Phoenix Executable", "path", "phoenix_main.exe", "General", "Launcher"),
    OptionField("open_frontend_after_launch", "Open Frontend After Launch", "bool", True, "General", "Launcher"),
    OptionField("gateway_host", "Gateway Host", "string", "127.0.0.1", "General", "Gateway", "gateway-host"),
    OptionField("gateway_port", "Gateway Port", "int", 5080, "General", "Gateway", "port", emit_default=True),
    OptionField("study_port", "Study Port", "int", 5081, "General", "Gateway", "study-port", emit_default=True),
    OptionField("frontend_enabled", "Enable Frontend", "bool", True, "General", "Gateway", "frontend-enabled", emit_default=True),
    OptionField("http_log", "Enable HTTP Log", "bool", False, "General", "Gateway", "http-log", emit_default=True),
    OptionField("frontend_http_log", "Enable Frontend HTTP Log", "bool", False, "General", "Gateway", "frontend-http-log", emit_default=True),
    OptionField("log_mode", "Log Mode", "string", "release", "General", "Gateway", "log-mode"),
    OptionField("base_dir", "Base Dir", "directory", "runtime_store", "General", "Workspace", "base-dir"),
    OptionField("export_dir", "Export Dir", "directory", "runtime_store", "General", "Workspace", "export-dir"),
    OptionField("snapshot_dir", "Snapshot Dir", "directory", "snapshots", "General", "Workspace", "snapshot-dir"),
    OptionField("db_path", "SQLite DB Path", "path", "runtime_store/ai_store.sqlite", "General", "Workspace", "db-path"),
    OptionField("transformer_mode", "Transformer Mode", "choice", "llamacpp", "Inference", "Core", "transformer-mode", ("llamacpp", "bitnet", "ollama", "ollama-fine-tuning", "off"), emit_default=True),
    OptionField("gguf_models_dir", "GGUF Models Dir", "directory", "GGUF_models", "Inference", "Core", "gguf-models-dir", emit_default=True),
    OptionField("external_auto_launch", "Auto Launch External Backend", "bool", True, "Inference", "Core", "external-auto-launch", emit_default=True),
    OptionField("bug_shooter_enabled", "Enable Bug Shooter", "bool", True, "Inference", "Core", "bug-shooter", emit_default=True),
    OptionField("inference_enabled", "Enable Inference", "bool", True, "Inference", "Core", "inference-enabled"),
    OptionField("using_ollama", "Use Ollama Legacy Switch", "bool", False, "Inference", "Core", "using-ollama"),
    OptionField("ollama_fine_tuning", "Enable Ollama Fine Tuning", "bool", False, "Inference", "Core", "ollama-fine-tuning"),
    OptionField("ollama_model", "Ollama Model", "string", "", "Inference", "Ollama", "ollama-model"),
    OptionField("ollama_base_url", "Ollama Base URL", "string", "http://127.0.0.1:11434", "Inference", "Ollama", "ollama-base-url"),
    OptionField("ollama_keep_alive", "Ollama Keep Alive", "string", "", "Inference", "Ollama", "ollama-keep-alive"),
    OptionField("ollama_timeout_ms", "Ollama Timeout (ms)", "int", 120000, "Inference", "Ollama", "ollama-timeout-ms"),
    OptionField("llamacpp_model", "llama.cpp Model", "path", "", "Inference", "llama.cpp", "llamacpp-model"),
    OptionField("llamacpp_base_url", "llama.cpp Base URL", "string", "http://127.0.0.1:8082", "Inference", "llama.cpp", "llamacpp-base-url", emit_default=True),
    OptionField("llamacpp_timeout_ms", "llama.cpp Timeout (ms)", "int", 120000, "Inference", "llama.cpp", "llamacpp-timeout-ms"),
    OptionField("llamacpp_ctx_size", "llama.cpp Context Size", "int", 10000000, "Inference", "llama.cpp", "llamacpp-ctx-size", emit_default=True),
    OptionField("llamacpp_batch_size", "llama.cpp Batch Size", "int", 512, "Inference", "llama.cpp", "llamacpp-batch-size", emit_default=True),
    OptionField("llamacpp_ubatch_size", "llama.cpp UBatch Size", "int", 128, "Inference", "llama.cpp", "llamacpp-ubatch-size", emit_default=True),
    OptionField("llamacpp_lora_files", "llama.cpp LoRA Files", "string", "", "Inference", "llama.cpp", "llamacpp-lora-files"),
    OptionField("llamacpp_lora_init_without_apply", "llama.cpp LoRA Init Without Apply", "bool", False, "Inference", "llama.cpp", "llamacpp-lora-init-without-apply"),
    OptionField("llamacpp_fine_tuning", "llama.cpp Fine Tuning", "bool", False, "Inference", "llama.cpp", "llamacpp-fine-tuning"),
    OptionField("bitnet_model", "BitNet Model", "path", "", "Inference", "BitNet", "bitnet-model"),
    OptionField("bitnet_base_url", "BitNet Base URL", "string", "http://127.0.0.1:8090", "Inference", "BitNet", "bitnet-base-url"),
    OptionField("bitnet_timeout_ms", "BitNet Timeout (ms)", "int", 120000, "Inference", "BitNet", "bitnet-timeout-ms"),
    OptionField("learning_enabled", "Enable Learning", "bool", True, "Autonomy", "Learning", "disable-learning", invert=True),
    OptionField("rl_enabled", "Enable RL", "bool", True, "Autonomy", "Learning", "disable-rl", invert=True),
    OptionField("adv_enabled", "Enable ADV", "bool", True, "Autonomy", "Learning", "disable-adv", invert=True),
    OptionField("gnnga_enabled", "Enable GNN-GA", "bool", True, "Autonomy", "Learning", "disable-gnnga", invert=True),
    OptionField("gnn_enabled", "Enable GNN Module", "bool", True, "Autonomy", "Learning", "disable-gnn-module", emit_default=True, invert=True),
    OptionField("context_enabled", "Enable Context Module", "bool", True, "Autonomy", "Learning", "disable-context-module", emit_default=True, invert=True),
    OptionField("dialog_learning_enabled", "Enable Dialog Learning", "bool", True, "Autonomy", "Learning", "disable-dialog-learning", invert=True),
    OptionField("dialog_transformer_learning", "Enable Dialog Transformer Learning", "bool", False, "Autonomy", "Learning", "dialog-transformer-learning", emit_default=True),
    OptionField("dialog_graph_context", "Enable Dialog Graph Context", "bool", False, "Autonomy", "Learning", "dialog-learn-graph-context", emit_default=True),
    OptionField("learning_warmup", "Enable Learning Warmup", "bool", False, "Autonomy", "Learning", "learning-warmup"),
    OptionField("sync_standby_on_boot", "Sync Standby On Boot", "bool", False, "Autonomy", "Learning", "sync-standby"),
    OptionField("transformer_bootstrap", "Enable Transformer Bootstrap", "bool", False, "Autonomy", "Learning", "transformer-bootstrap"),
    OptionField("transformer_bootstrap_docs", "Transformer Bootstrap Docs", "int", 256, "Autonomy", "Learning", "transformer-bootstrap-docs"),
    OptionField("persistent_session_memory_enabled", "Persist Session Memory", "bool", True, "Autonomy", "Reasoning & Brain", "persistent-session-memory", emit_default=True),
    OptionField("brain_enabled", "Enable Brain Context", "bool", True, "Autonomy", "Reasoning & Brain", "brain-enabled", emit_default=True),
    OptionField("brain_profile", "Brain Profile", "choice", "functional", "Autonomy", "Reasoning & Brain", "brain-profile", ("functional", "structural", "research")),
    OptionField("reasoning_agenda_enabled", "Enable Reasoning Agenda", "bool", True, "Autonomy", "Reasoning & Brain", "reasoning-agenda-enabled"),
    OptionField("reasoning_planner_enabled", "Enable Reasoning Planner", "bool", True, "Autonomy", "Reasoning & Brain", "reasoning-planner-enabled"),
    OptionField("reasoning_critic_enabled", "Enable Reasoning Critic", "bool", True, "Autonomy", "Reasoning & Brain", "reasoning-critic-enabled"),
    OptionField("reasoning_critic_uncertainty_threshold", "Critic Uncertainty Threshold", "float", 0.45, "Autonomy", "Reasoning & Brain", "reasoning-critic-uncertainty-threshold"),
    OptionField("reasoning_critic_verify_threshold", "Critic Verify Threshold", "float", 0.60, "Autonomy", "Reasoning & Brain", "reasoning-critic-verify-threshold"),
    OptionField("cognition_autonomy_enabled", "Enable Cognition Autonomy", "bool", True, "Autonomy", "Cognition Autonomy", "cognition-autonomy-enabled"),
    OptionField("cognition_autonomy_background_enabled", "Enable Background Cognition Autonomy", "bool", True, "Autonomy", "Cognition Autonomy", "cognition-autonomy-background-enabled"),
    OptionField("cognition_autonomy_every", "Cognition Cadence", "int", 4, "Autonomy", "Cognition Autonomy", "cognition-autonomy-every"),
    OptionField("cognition_autonomy_seed_mission", "Seed Mission", "text", "", "Autonomy", "Cognition Autonomy", "cognition-autonomy-seed-mission", rows=3),
    OptionField("cognition_autonomy_seed_session_id", "Seed Session ID", "string", "autonomy-seed", "Autonomy", "Cognition Autonomy", "cognition-autonomy-seed-session-id"),
    OptionField("cognition_autonomy_tick_ms", "Background Tick (ms)", "int", 15000, "Autonomy", "Cognition Autonomy", "cognition-autonomy-tick-ms"),
    OptionField("cognition_autonomy_max_iterations", "Max Background Iterations", "int", 0, "Autonomy", "Cognition Autonomy", "cognition-autonomy-max-iterations"),
    OptionField("collective_routing_storage_enabled", "Enable Collective Routing Storage", "bool", True, "Autonomy", "Cognition Autonomy", "collective-routing-storage"),
    OptionField("collective_unmatched_compute", "Ignore Matching For Compute Nodes", "bool", True, "Autonomy", "Cognition Autonomy", "collective-unmatched-compute"),
    OptionField("memebarrier_enabled", "Enable MemeBarrier", "bool", True, "Safety", "MemeBarrier", "disable-memebarrier", emit_default=True, invert=True),
    OptionField("memebarrier_threshold", "Barrier Threshold", "float", 0.70, "Safety", "MemeBarrier", "memebarrier-threshold", emit_default=True),
    OptionField("memebarrier_disable_rate", "Barrier Disable Rate", "float", 0.01, "Safety", "MemeBarrier", "memebarrier-disable-rate", emit_default=True),
    OptionField("memebarrier_phrase_feedback_step", "Phrase Feedback Step", "float", 0.05, "Safety", "MemeBarrier", "memebarrier-phrase-feedback-step"),
    OptionField("memebarrier_phrase_feedback_max_offset", "Phrase Feedback Max Offset", "float", 0.30, "Safety", "MemeBarrier", "memebarrier-phrase-feedback-max-offset"),
    OptionField("memebarrier_use_cnn", "Use Text CNN", "bool", True, "Safety", "MemeBarrier", "memebarrier-use-cnn", emit_default=True),
    OptionField("memebarrier_use_recommender", "Use Recommender", "bool", True, "Safety", "MemeBarrier", "memebarrier-use-recommender", emit_default=True),
    OptionField("memebarrier_use_torch", "Use Torch Models", "bool", True, "Safety", "MemeBarrier", "memebarrier-use-torch", emit_default=True),
    OptionField("mechanical_mind_enabled", "Enable Mechanical Mind", "bool", False, "Safety", "Mechanical Mind", "mechanical-mind-enabled", emit_default=True),
    OptionField("mechanical_mind_threshold", "Mechanical Mind Threshold", "float", 0.58, "Safety", "Mechanical Mind", "mechanical-mind-threshold", emit_default=True),
    OptionField("mechanical_mind_token_threshold", "Mechanical Mind Token Threshold", "float", 0.60, "Safety", "Mechanical Mind", "mechanical-mind-token-threshold", emit_default=True),
    OptionField("ai_count", "AI Count", "int", 7, "World", "Scale", "ai-count"),
    OptionField("group_count", "Group Count", "int", 3, "World", "Scale", "group-count"),
    OptionField("group_size", "Group Size", "int", 7, "World", "Scale", "group-size"),
    OptionField("spark_num_ai", "Spark Num AI", "int", 7, "World", "Scale", "spark-num-ai"),
    OptionField("spark_budget", "Spark Budget", "string", "default", "World", "Scale", "spark-budget"),
    OptionField("world_agent_count", "World Agent Count", "int", 4, "World", "World Model", "world-agent-count"),
    OptionField("world_map_width", "Map Width", "int", 6, "World", "World Model", "world-map-width"),
    OptionField("world_map_height", "Map Height", "int", 6, "World", "World Model", "world-map-height"),
    OptionField("world_map_depth", "Map Depth", "int", 3, "World", "World Model", "world-map-depth"),
    OptionField("world_dialogue_turns", "Dialogue Turns", "int", 2, "World", "World Model", "world-dialogue-turns"),
    OptionField("world_ecology_clusters", "Ecology Clusters", "int", 2, "World", "World Model", "world-ecology-clusters"),
    OptionField("world_3d_map_enabled", "Enable 3D Map", "bool", True, "World", "World Model", "world-3d-map-enabled"),
    OptionField("world_embodied_agents_enabled", "Enable Embodied Agents", "bool", True, "World", "World Model", "world-embodied-agents-enabled"),
    OptionField("world_ecology_video_enabled", "Enable Ecology Video", "bool", True, "World", "World Model", "world-ecology-video-enabled"),
    OptionField("world_physics_enabled", "Enable Physics", "bool", True, "World", "World Model", "world-physics-enabled"),
    OptionField("world_physics_substeps", "Physics Substeps", "int", 4, "World", "World Model", "world-physics-substeps"),
    OptionField("world_physics_backend", "Physics Backend", "string", "bullet3", "World", "World Model", "world-physics-backend"),
    OptionField("world_earth_map_enabled", "Enable Earth Map", "bool", True, "World", "World Model", "world-earth-map-enabled"),
    OptionField("world_earth_map_uri", "Earth Map URI", "path", "static/earth_maps/china_relief_heightfield.json", "World", "World Model", "world-earth-map-uri"),
    OptionField("world_earth_map_format", "Earth Map Format", "string", "heightfield", "World", "World Model", "world-earth-map-format"),
    OptionField("edge_platform_enabled", "Enable Edge Platform", "bool", True, "World", "Platform", "edge-platform"),
    OptionField("edge_platform_npu_enabled", "Enable Edge NPU", "bool", True, "World", "Platform", "edge-platform-npu"),
    OptionField("edge_platform_peripherals_enabled", "Enable Peripheral Scheduling", "bool", True, "World", "Platform", "edge-platform-peripherals"),
    OptionField("edge_preferred_backend", "Preferred Compute Backend", "choice", "auto", "World", "Platform", "edge-preferred-backend", ("auto", "cpu", "npu", "hybrid")),
    OptionField("edge_compute_inflight", "Compute Inflight", "int", 2, "World", "Platform", "edge-compute-inflight"),
    OptionField("edge_peripheral_inflight", "Peripheral Inflight", "int", 4, "World", "Platform", "edge-peripheral-inflight"),
    OptionField("edge_netlist_root", "Netlist Root", "directory", "catastrophe", "World", "Platform", "edge-netlist-root"),
    OptionField("component_config_path", "Component Config Path", "path", "", "Addons", "Components", "component-config"),
    OptionField("component_config_format", "Component Config Format", "string", "", "Addons", "Components", "component-config-format"),
    OptionField("component_overrides", "Component Overrides", "text", "", "Addons", "Components", "components", rows=5),
    OptionField("computer_shell_enabled", "Enable Computer Shell Bridge", "bool", False, "Addons", "Computer Shell", "computer-shell-enabled", emit_default=True),
    OptionField("computer_shell_readonly", "Computer Shell Readonly", "bool", True, "Addons", "Computer Shell", "computer-shell-readonly", emit_default=True),
    OptionField("builtin_addons", "Builtin Addons (comma separated)", "string", "math,search", "Addons", "Addons", "builtin-addons", emit_default=True),
    OptionField("addon_libraries", "Addon Libraries (one per line)", "text", "", "Addons", "Addons", "addon-libraries", rows=6),
    OptionField("computer_shell_bridge_script", "Computer Shell Bridge Script", "path", "tools/computer_shell_bridge.py", "Expert", "Computer Shell", "computer-shell-bridge-script"),
    OptionField("computer_shell_working_dir", "Computer Shell Working Dir", "directory", ".", "Expert", "Computer Shell", "computer-shell-working-dir"),
    OptionField("computer_shell_timeout_ms", "Computer Shell Timeout (ms)", "int", 15000, "Expert", "Computer Shell", "computer-shell-timeout-ms"),
    OptionField("computer_shell_max_output_bytes", "Computer Shell Max Output Bytes", "int", 8192, "Expert", "Computer Shell", "computer-shell-max-output-bytes"),
    OptionField("single_proc", "Single Process Mode", "bool", False, "Expert", "Process Model", "single-proc"),
    OptionField("group_proc", "Group Process Mode", "bool", False, "Expert", "Process Model", "group-proc"),
    OptionField("group_proc_timeout_ms", "Group Process Timeout (ms)", "int", 12000, "Expert", "Process Model", "group-proc-timeout-ms"),
    OptionField("infer_mp", "Enable Infer MP", "bool", False, "Expert", "Process Model", "infer-mp"),
    OptionField("infer_workers", "Infer Workers", "int", 0, "Expert", "Process Model", "infer-workers"),
    OptionField("redis_timeout_ms", "Redis Timeout (ms)", "int", 1500, "Expert", "Storage & Data", "redis-timeout-ms"),
    OptionField("redis_url", "Redis URL", "string", "redis://127.0.0.1:6379", "Expert", "Storage & Data", "redis-url"),
    OptionField("redis_channel", "Redis Channel", "string", "AI-model-workspace", "Expert", "Storage & Data", "channel"),
    OptionField("redis_cache_db", "Redis Cache DB", "int", 1, "Expert", "Storage & Data", "redis-cache-db"),
    OptionField("redis_cache_prefix", "Redis Cache Prefix", "string", "AI079", "Expert", "Storage & Data", "redis-cache-prefix"),
    OptionField("lmdb_dir", "LMDB Dir", "directory", "lmdb", "Expert", "Storage & Data", "lmdb-dir"),
    OptionField("lmdb_map_mb", "LMDB Map (MB)", "int", 512, "Expert", "Storage & Data", "lmdb-map-mb"),
    OptionField("search_endpoint", "Search Endpoint", "string", "", "Expert", "Storage & Data", "search-endpoint"),
    OptionField("robots_dir", "Robots Dir", "directory", "robots", "Expert", "Storage & Data", "robots-dir"),
    OptionField("lemma_csv", "Lemma CSV", "path", "lemma.csv", "Expert", "Storage & Data", "lemma-csv"),
    OptionField("lemma_autoload", "Autoload Lemma CSV", "bool", False, "Expert", "Storage & Data", "lemma-autoload"),
    OptionField("lemma_max_mb", "Lemma Max (MB)", "int", 64, "Expert", "Storage & Data", "lemma-max-mb"),
    OptionField("lemma_force", "Force Lemma Reload", "bool", False, "Expert", "Storage & Data", "lemma-force"),
    OptionField("robots_limit", "Robots Warmup Limit", "int", 200, "Expert", "Storage & Data", "robots-limit"),
    OptionField("robots_autoload", "Autoload Robots", "bool", True, "Expert", "Storage & Data", "robots-autoload"),
    OptionField("robots_warmup_shuffle", "Shuffle Robots Warmup", "bool", False, "Expert", "Storage & Data", "robots-warmup-shuffle"),
    OptionField("robots_chunk_min", "Robot Chunk Min Words", "int", 3, "Expert", "Storage & Data", "robots-chunk-min"),
    OptionField("robots_chunk_max", "Robot Chunk Max Words", "int", 20, "Expert", "Storage & Data", "robots-chunk-max"),
    OptionField("kvm_cache_max", "KVM Cache Max Entries", "int", 50000, "Expert", "Storage & Data", "kvm-cache-max"),
    OptionField("tests_autoload", "Autoload Tests", "bool", True, "Expert", "Storage & Data", "tests-autoload"),
    OptionField("outsides_root", "Outsides Root", "directory", "outsides", "Expert", "Backend Paths & Templates", "outsides-root"),
    OptionField("llamacpp_root", "llama.cpp Root", "directory", "outsides/llamacpp", "Expert", "Backend Paths & Templates", "llamacpp-root"),
    OptionField("bitnet_root", "BitNet Root", "directory", "outsides/BitNet", "Expert", "Backend Paths & Templates", "bitnet-root"),
    OptionField("calculator_root", "Calculator Root", "directory", "outsides/_calculator", "Expert", "Backend Paths & Templates", "calculator-root"),
    OptionField("diving_root", "Diving Agreement Root", "directory", "outsides/_diving_agreement", "Expert", "Backend Paths & Templates", "diving-root"),
    OptionField("python_exe", "Python Executable", "path", "Python314/python.exe", "Expert", "Backend Paths & Templates", "python-exe"),
    OptionField("external_adapter_script", "External Adapter Script", "path", "tools/external_model_adapter.py", "Expert", "Backend Paths & Templates", "external-adapter-script"),
    OptionField("external_ready_timeout_ms", "External Ready Timeout (ms)", "int", 30000, "Expert", "Backend Paths & Templates", "external-ready-timeout-ms"),
    OptionField("external_health_poll_ms", "External Health Poll (ms)", "int", 500, "Expert", "Backend Paths & Templates", "external-health-poll-ms"),
    OptionField("external_style_sim", "Enable External Style Simulation", "bool", True, "Expert", "Backend Paths & Templates", "external-style-sim"),
    OptionField("llamacpp_rope_scaling", "llama.cpp Rope Scaling", "string", "yarn", "Expert", "Backend Paths & Templates", "llamacpp-rope-scaling"),
    OptionField("llamacpp_rope_freq_base", "llama.cpp Rope Freq Base", "float", 0.0, "Expert", "Backend Paths & Templates", "llamacpp-rope-freq-base"),
    OptionField("llamacpp_rope_freq_scale", "llama.cpp Rope Freq Scale", "float", 0.0, "Expert", "Backend Paths & Templates", "llamacpp-rope-freq-scale"),
    OptionField("llamacpp_yarn_orig_ctx", "llama.cpp YARN Orig Ctx", "int", 4096, "Expert", "Backend Paths & Templates", "llamacpp-yarn-orig-ctx"),
    OptionField("llamacpp_yarn_ext_factor", "llama.cpp YARN Ext Factor", "float", 1.0, "Expert", "Backend Paths & Templates", "llamacpp-yarn-ext-factor"),
    OptionField("llamacpp_yarn_attn_factor", "llama.cpp YARN Attn Factor", "float", 1.0, "Expert", "Backend Paths & Templates", "llamacpp-yarn-attn-factor"),
    OptionField("llamacpp_yarn_beta_fast", "llama.cpp YARN Beta Fast", "float", 32.0, "Expert", "Backend Paths & Templates", "llamacpp-yarn-beta-fast"),
    OptionField("llamacpp_yarn_beta_slow", "llama.cpp YARN Beta Slow", "float", 1.0, "Expert", "Backend Paths & Templates", "llamacpp-yarn-beta-slow"),
    OptionField("llamacpp_launch_args", "llama.cpp Launch Args Template", "text", "-m \"{model}\" --host {host} --port {port} --ctx-size {ctx_size} --batch-size {batch_size} --ubatch-size {ubatch_size} -ngl 0 --rope-scaling {rope_scaling} --rope-freq-base {rope_freq_base} --rope-freq-scale {rope_freq_scale} --yarn-orig-ctx {yarn_orig_ctx} --yarn-ext-factor {yarn_ext_factor} --yarn-attn-factor {yarn_attn_factor} --yarn-beta-fast {yarn_beta_fast} --yarn-beta-slow {yarn_beta_slow}", "Expert", "Backend Paths & Templates", "llamacpp-launch-args", rows=4),
    OptionField("bitnet_launch_args", "BitNet Launch Args Template", "text", "-m \"{model}\" --host {host} --port {port} --brain-map \"{brain_map}\"", "Expert", "Backend Paths & Templates", "bitnet-launch-args", rows=3),
    OptionField("bug_shooter_exe", "Bug Shooter Executable", "path", "bug_shooter.exe", "Expert", "Monitoring & Hot Cache", "bug-shooter-exe"),
    OptionField("bug_shooter_poll_ms", "Bug Shooter Poll (ms)", "int", 1500, "Expert", "Monitoring & Hot Cache", "bug-shooter-poll-ms"),
    OptionField("bug_shooter_soft_limit_mb", "Bug Shooter Soft Limit (MB)", "int", 3072, "Expert", "Monitoring & Hot Cache", "bug-shooter-soft-limit-mb"),
    OptionField("bug_shooter_hard_limit_mb", "Bug Shooter Hard Limit (MB)", "int", 4096, "Expert", "Monitoring & Hot Cache", "bug-shooter-hard-limit-mb"),
    OptionField("inference_hot_limit", "Inference Hot Limit", "int", 512, "Expert", "Monitoring & Hot Cache", "inference-hot-limit"),
    OptionField("inference_promote_hits", "Inference Promote Hits", "int", 2, "Expert", "Monitoring & Hot Cache", "inference-promote-hits"),
    OptionField("inference_hot_ttl_seconds", "Inference Hot TTL Seconds", "int", 3600, "Expert", "Monitoring & Hot Cache", "inference-hot-ttl-seconds"),
    OptionField("inference_rebalance_every", "Inference Rebalance Every", "int", 64, "Expert", "Monitoring & Hot Cache", "inference-rebalance-every"),
    OptionField("extra_args", "Raw Extra Args", "text", "", "Expert", "Raw Overrides", rows=7),
]


FIELD_BY_KEY = {field.key: field for field in OPTION_FIELDS}


SECTION_NOTES: dict[tuple[str, str], str] = {
    ("Addons", "Components"): "Component Overrides supports one key per line or a comma-separated list.",
    ("Addons", "Addons"): "Builtin Addons accepts a comma-separated list such as math,search.",
    ("Addons", "Computer Shell"): "Enable Computer Shell Bridge to attach a local desktop-control bridge. Keep readonly on unless you explicitly want command execution or open actions.",
    ("Expert", "Raw Overrides"): "Raw Extra Args is appended last. Use it for any future or highly specialized flags not yet surfaced above.",
}


@dataclass
class LaunchOptions:
    values: dict[str, Any] = field(default_factory=dict)


def default_option_values() -> dict[str, Any]:
    return {field.key: field.default for field in OPTION_FIELDS}


def serialize_builtin_addons(raw: str) -> str:
    values = split_list(raw)
    return ",".join(values) if values else "none"


def normalize_command_text(raw: Any) -> str:
    lines = [line.strip() for line in str(raw).replace("\r", "\n").split("\n")]
    return " ".join(line for line in lines if line)


def normalize_csv_text(raw: Any) -> str:
    return ",".join(split_list(str(raw)))


def normalize_line_block(raw: Any) -> str:
    lines = [line.strip() for line in str(raw).replace("\r", "\n").split("\n")]
    return "\n".join(line for line in lines if line)


def as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


def migrate_legacy_profile(data: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(data)
    if "builtin_addons" in migrated and isinstance(migrated["builtin_addons"], dict):
        enabled = [name for name, on in migrated["builtin_addons"].items() if on]
        migrated["builtin_addons"] = ",".join(enabled) if enabled else "none"
    elif "builtin_addons" not in migrated and ("addon_math" in migrated or "addon_search" in migrated):
        enabled = []
        if as_bool(migrated.get("addon_math"), True):
            enabled.append("math")
        if as_bool(migrated.get("addon_search"), True):
            enabled.append("search")
        migrated["builtin_addons"] = ",".join(enabled) if enabled else "none"
    if "addon_libraries" in migrated and isinstance(migrated["addon_libraries"], list):
        migrated["addon_libraries"] = "\n".join(str(item).strip() for item in migrated["addon_libraries"] if str(item).strip())
    return migrated


def config_path(root: Path) -> Path:
    return root / "runtime_store" / "start_079_launcher.json"


def load_profile(root: Path) -> LaunchOptions:
    path = config_path(root)
    values = default_option_values()
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and isinstance(raw.get("values"), dict):
            raw_values = migrate_legacy_profile(raw["values"])
        elif isinstance(raw, dict):
            raw_values = migrate_legacy_profile(raw)
        else:
            raw_values = {}
        for option in OPTION_FIELDS:
            if option.key not in raw_values:
                continue
            if option.kind == "bool":
                values[option.key] = as_bool(raw_values[option.key], bool(option.default))
            else:
                values[option.key] = str(raw_values[option.key])

    env_extra_args = "\n".join(
        text.strip()
        for text in [
            os.getenv("PHOENIX_ARGS", ""),
            os.getenv("ODIN_PHOENIX_ARGS", ""),
            os.getenv("PHOENIX_AUTONOMY_ARGS", ""),
        ]
        if text.strip()
    )
    if env_extra_args:
        current = normalize_line_block(values.get("extra_args", ""))
        values["extra_args"] = env_extra_args if not current else current + "\n" + env_extra_args

    current_mode = str(values.get("transformer_mode", "llamacpp")).strip().lower()
    valid_modes = set(FIELD_BY_KEY["transformer_mode"].choices)
    if current_mode not in valid_modes:
        values["transformer_mode"] = FIELD_BY_KEY["transformer_mode"].default

    return LaunchOptions(values)


def save_profile(root: Path, options: LaunchOptions) -> None:
    path = config_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 2, "values": options.values}, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_path(root: Path, raw: str) -> str:
    text = raw.strip()
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        path = root / path
    return str(path.resolve())


def normalized_compare_value(option: OptionField, raw: Any) -> Any:
    if option.kind == "bool":
        return as_bool(raw, bool(option.default))
    if option.kind == "int":
        text = str(raw).strip()
        if not text:
            text = str(option.default).strip()
        return int(text)
    if option.kind == "float":
        text = str(raw).strip()
        if not text:
            text = str(option.default).strip()
        return float(text)
    text = str(raw)
    if option.key == "builtin_addons":
        return serialize_builtin_addons(text)
    if option.key == "addon_libraries":
        return normalize_line_block(text)
    if option.key == "component_overrides":
        return normalize_csv_text(text)
    if option.key in {"llamacpp_launch_args", "bitnet_launch_args", "extra_args", "cognition_autonomy_seed_mission"}:
        return normalize_command_text(text)
    return text.strip()


def should_emit_option(option: OptionField, value: Any) -> bool:
    if option.cli is None:
        return False
    current = normalized_compare_value(option, value)
    default = normalized_compare_value(option, option.default)
    if option.emit_default:
        return True
    if option.kind == "bool":
        return current != default
    return bool(current) and current != default


def build_argument(option: OptionField, value: Any, root: Path) -> str | None:
    if option.cli is None:
        return None
    if option.kind == "bool":
        flag_value = as_bool(value, bool(option.default))
        if option.invert:
            flag_value = not flag_value
        return f"--{option.cli}={bool_str(flag_value)}"
    if option.kind == "int":
        text = str(value).strip()
        if not text:
            text = str(option.default)
        return f"--{option.cli}={int(text)}"
    if option.kind == "float":
        text = str(value).strip()
        if not text:
            text = str(option.default)
        return f"--{option.cli}={format(float(text), 'g')}"

    text = str(value)
    if option.key == "builtin_addons":
        text = serialize_builtin_addons(text)
    elif option.key == "addon_libraries":
        lines = [line.strip() for line in normalize_line_block(text).split("\n") if line.strip()]
        resolved = [resolve_path(root, line) for line in lines]
        text = ";".join(item for item in resolved if item)
    elif option.key == "component_overrides":
        text = normalize_csv_text(text)
    elif option.key in {"llamacpp_launch_args", "bitnet_launch_args", "cognition_autonomy_seed_mission"}:
        text = normalize_command_text(text)
    else:
        text = text.strip()

    if option.kind in {"path", "directory"} and text:
        text = resolve_path(root, text)
    if not text:
        return None
    return f"--{option.cli}={text}"


def build_launch_command(options: LaunchOptions, root: Path) -> list[str]:
    executable = resolve_path(root, str(options.values.get("phoenix_executable", FIELD_BY_KEY["phoenix_executable"].default)))
    command = [executable]
    for option in OPTION_FIELDS:
        if option.key == "extra_args":
            continue
        if not should_emit_option(option, options.values.get(option.key, option.default)):
            continue
        argument = build_argument(option, options.values.get(option.key, option.default), root)
        if argument:
            command.append(argument)
    command.extend(parse_extra_args(normalize_line_block(options.values.get("extra_args", ""))))
    return command


def launch_process(root: Path, options: LaunchOptions) -> tuple[subprocess.Popen[str], list[str]]:
    internal_name = str(options.values.get("internal_name", "")).strip()
    if internal_name != "079":
        raise ValueError("internal name must be 079")
    command = build_launch_command(options, root)
    executable = Path(command[0])
    if not executable.exists():
        raise FileNotFoundError(f"phoenix executable not found: {executable}")
    transformer_mode = str(options.values.get("transformer_mode", "")).strip().lower()
    gguf_dir = Path(resolve_path(root, str(options.values.get("gguf_models_dir", ""))))
    if transformer_mode in {"llamacpp", "llama.cpp", "llama_cpp"} and not gguf_dir.exists():
        raise FileNotFoundError(f"GGUF directory not found: {gguf_dir}")
    # Kill all phoenix_main.exe processes (including frontend-only processes)
    subprocess.run(["taskkill", "/IM", "phoenix_main.exe", "/F"], cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    subprocess.run(["taskkill", "/IM", "bug_shooter.exe", "/F"], cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    # Kill external adapter processes
    subprocess.run(["taskkill", "/IM", "python.exe", "/FI", "WINDOWTITLE eq external_model_adapter*", "/F"], cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    # Wait a moment to ensure processes are fully terminated
    import time
    time.sleep(2)
    
    # Launch external adapter if external_auto_launch is true and transformer_mode is llamacpp
    external_auto_launch = as_bool(options.values.get("external_auto_launch", True), True)
    if external_auto_launch and transformer_mode in {"llamacpp", "llama.cpp", "llama_cpp"}:
        adapter_script = Path(resolve_path(root, str(options.values.get("external_adapter_script", "tools/external_model_adapter.py"))))
        if adapter_script.exists():
            python_exe = Path(resolve_path(root, str(options.values.get("python_executable", "python"))))
            adapter_args = [
                str(python_exe),
                str(adapter_script),
                "--provider", "llamacpp",
                "--model", str(options.values.get("llamacpp_model", "")),
                "--base-url", str(options.values.get("llamacpp_base_url", "")),
                "--timeout-ms", str(options.values.get("llamacpp_timeout_ms", "30000")),
                "--ctx-size", str(options.values.get("llamacpp_ctx_size", "4096")),
                "--batch-size", str(options.values.get("llamacpp_batch_size", "512")),
            ]
            # Add LoRA parameters if specified
            lora_files = str(options.values.get("llamacpp_lora_files", "")).strip()
            if lora_files:
                adapter_args.extend(["--lora-files", lora_files])
            lora_init_without_apply = as_bool(options.values.get("llamacpp_lora_init_without_apply", False), False)
            if lora_init_without_apply:
                adapter_args.append("--lora-init-without-apply")
            fine_tuning = as_bool(options.values.get("llamacpp_fine_tuning", False), False)
            if fine_tuning:
                adapter_args.append("--fine-tuning")
            subprocess.Popen(adapter_args, cwd=root, creationflags=subprocess.CREATE_NEW_CONSOLE)
    
    process = subprocess.Popen(command, cwd=root)
    return process, command


def build_exe(root: Path) -> subprocess.CompletedProcess[str]:
    script = root / "build_start_079_oneclick_exe.bat"
    if not script.exists():
        raise FileNotFoundError(f"build script not found: {script}")
    return subprocess.run([str(script)], cwd=root, capture_output=True, text=True, check=False)


def launch_gui() -> None:
    import tkinter as tk
    from tkinter import messagebox, ttk

    root_path = detect_repo_root()
    options = load_profile(root_path)

    app = tk.Tk()
    app.title("079 OneClick Launcher")
    app.geometry("1200x900")
    app.minsize(1024, 760)

    style = ttk.Style(app)
    style.configure("PrimaryLaunch.TButton", padding=(18, 10))

    state_vars: dict[str, Any] = {"status": tk.StringVar(value=f"Root: {root_path}")}
    text_widgets: dict[str, tk.Text] = {}

    for option in OPTION_FIELDS:
        current = options.values.get(option.key, option.default)
        if option.kind == "bool":
            state_vars[option.key] = tk.BooleanVar(value=as_bool(current, bool(option.default)))
        else:
            state_vars[option.key] = tk.StringVar(value=str(current))

    def make_scrollable_tab(name: str) -> ttk.Frame:
        container = ttk.Frame(notebook)
        canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=inner, anchor="nw")

        def on_inner_configure(_event: Any) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_canvas_configure(event: Any) -> None:
            canvas.itemconfigure(canvas_window, width=event.width)

        inner.bind("<Configure>", on_inner_configure)
        canvas.bind("<Configure>", on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        notebook.add(container, text=name)
        return inner

    notebook = ttk.Notebook(app)
    notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
    tab_frames = {tab: make_scrollable_tab(tab) for tab in TAB_ORDER}

    launch_bar = ttk.Frame(app)
    launch_bar.pack(fill=tk.X, padx=12, pady=(12, 8), before=notebook)
    ttk.Label(launch_bar, text="Direct launch:").pack(side=tk.LEFT)
    ttk.Label(launch_bar, text="Common options stay on the main tabs. Low-level tuning and raw overrides live on Expert.").pack(side=tk.LEFT, padx=(8, 0))

    def collect_options() -> LaunchOptions:
        values = default_option_values()
        for option in OPTION_FIELDS:
            if option.kind == "bool":
                values[option.key] = bool(state_vars[option.key].get())
            elif option.kind == "text":
                values[option.key] = text_widgets[option.key].get("1.0", "end").strip()
            else:
                values[option.key] = str(state_vars[option.key].get()).strip()
        return LaunchOptions(values)

    def add_option_widget(parent: ttk.LabelFrame, row: int, option: OptionField) -> int:
        if option.kind == "bool":
            ttk.Checkbutton(parent, text=option.label, variable=state_vars[option.key]).grid(row=row, column=0, columnspan=2, sticky="w", padx=8, pady=4)
            return row + 1

        ttk.Label(parent, text=option.label).grid(row=row, column=0, sticky="nw", padx=8, pady=6)
        if option.kind == "choice":
            current = str(state_vars[option.key].get()).strip()
            values = option.choices if not current or current in option.choices else tuple(list(option.choices) + [current])
            widget = ttk.Combobox(parent, textvariable=state_vars[option.key], values=values, width=min(option.width, 40), state="readonly")
            widget.grid(row=row, column=1, sticky="ew", padx=8, pady=6)
            return row + 1
        if option.kind == "text":
            widget = tk.Text(parent, width=option.width, height=option.rows, wrap="word")
            widget.grid(row=row, column=1, sticky="ew", padx=8, pady=6)
            widget.insert("1.0", str(options.values.get(option.key, option.default)))
            text_widgets[option.key] = widget
            return row + 1

        widget = ttk.Entry(parent, textvariable=state_vars[option.key], width=option.width)
        widget.grid(row=row, column=1, sticky="ew", padx=8, pady=6)
        return row + 1

    tab_sections: dict[str, list[str]] = {tab: [] for tab in TAB_ORDER}
    for option in OPTION_FIELDS:
        if option.section not in tab_sections[option.tab]:
            tab_sections[option.tab].append(option.section)

    fields_by_section: dict[tuple[str, str], list[OptionField]] = {}
    for option in OPTION_FIELDS:
        fields_by_section.setdefault((option.tab, option.section), []).append(option)

    for tab_name in TAB_ORDER:
        tab_frame = tab_frames[tab_name]
        tab_frame.columnconfigure(0, weight=1)
        current_row = 0
        if tab_name == "Expert":
            ttk.Label(
                tab_frame,
                text="Expert contains process model, storage, backend launch templates, cache limits and raw CLI overrides. Leave values untouched to keep defaults.",
                wraplength=980,
                justify=tk.LEFT,
            ).grid(row=current_row, column=0, sticky="ew", padx=12, pady=(10, 4))
            current_row += 1
        for section_name in tab_sections[tab_name]:
            frame = ttk.LabelFrame(tab_frame, text=section_name)
            frame.grid(row=current_row, column=0, sticky="ew", padx=12, pady=(6, 8))
            frame.columnconfigure(1, weight=1)
            note = SECTION_NOTES.get((tab_name, section_name))
            row = 0
            if note:
                ttk.Label(frame, text=note, wraplength=930, justify=tk.LEFT).grid(row=row, column=0, columnspan=2, sticky="ew", padx=8, pady=(6, 2))
                row += 1
            for option in fields_by_section[(tab_name, section_name)]:
                row = add_option_widget(frame, row, option)
            current_row += 1

    preview = tk.Text(app, width=140, height=12, wrap="word")
    preview.pack(fill=tk.BOTH, expand=False, padx=12, pady=(0, 12))

    def show_preview() -> None:
        try:
            current = collect_options()
            command = build_launch_command(current, root_path)
            preview.delete("1.0", "end")
            preview.insert("1.0", " ".join(shlex.quote(part) for part in command))
            state_vars["status"].set("Command preview updated")
        except Exception as exc:
            messagebox.showerror("Preview Failed", str(exc), parent=app)

    def do_launch() -> None:
        try:
            current = collect_options()
            process, command = launch_process(root_path, current)
            save_profile(root_path, current)
            preview.delete("1.0", "end")
            preview.insert("1.0", " ".join(shlex.quote(part) for part in command))
            state_vars["status"].set(f"Launched phoenix_main.exe pid={process.pid}")
            if as_bool(current.values.get("open_frontend_after_launch", True), True) and as_bool(current.values.get("frontend_enabled", True), True):
                app.after(1200, lambda: webbrowser.open(f"http://127.0.0.1:{int(str(current.values.get('study_port', 5081)).strip() or '5081')}"))
        except Exception as exc:
            messagebox.showerror("Launch Failed", str(exc), parent=app)

    def do_save() -> None:
        try:
            current = collect_options()
            save_profile(root_path, current)
            state_vars["status"].set(f"Profile saved: {config_path(root_path)}")
        except Exception as exc:
            messagebox.showerror("Save Failed", str(exc), parent=app)

    def do_build() -> None:
        def worker() -> None:
            try:
                result = build_exe(root_path)
                message = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
                if result.returncode == 0:
                    state_vars["status"].set("EXE build completed: dist/start_079_oneclick.exe")
                    messagebox.showinfo("Build Completed", message or "dist/start_079_oneclick.exe", parent=app)
                else:
                    state_vars["status"].set("EXE build failed")
                    messagebox.showerror("Build Failed", message or "PyInstaller build failed", parent=app)
            except Exception as exc:
                state_vars["status"].set("EXE build failed")
                messagebox.showerror("Build Failed", str(exc), parent=app)

        threading.Thread(target=worker, daemon=True).start()

    ttk.Button(launch_bar, text="Start phoenix_main.exe", style="PrimaryLaunch.TButton", command=do_launch).pack(side=tk.RIGHT)

    button_bar = ttk.Frame(app)
    button_bar.pack(fill=tk.X, padx=12, pady=(0, 8))
    ttk.Button(button_bar, text="Preview Command", command=show_preview).pack(side=tk.LEFT, padx=6)
    ttk.Button(button_bar, text="Save Profile", command=do_save).pack(side=tk.LEFT, padx=6)
    ttk.Button(button_bar, text="Build EXE", command=do_build).pack(side=tk.LEFT, padx=6)
    ttk.Label(app, textvariable=state_vars["status"]).pack(fill=tk.X, padx=12, pady=(0, 12))

    app.bind("<Return>", lambda _event: do_launch())
    show_preview()
    app.mainloop()


if __name__ == "__main__":
    launch_gui()