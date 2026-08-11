#!/usr/bin/env python3
"""Download / resolve multimodal encoder model paths with China-friendly mirrors.

Phoenix can run in environments where huggingface.co is slow or unreachable.
This helper lets both the preparation script and the runtime server resolve a
model identifier to a local directory by trying, in order:

1. An explicit local path.
2. Hugging Face (with an optional mirror endpoint, e.g. https://hf-mirror.com).
3. ModelScope (with a configurable list of aliases, e.g. qwen/Qwen2-Audio-7B-Instruct).

The returned path can be passed to ``transformers.from_pretrained(...,
local_files_only=True)`` or loaded as an ONNX / saved checkpoint directory.
"""

import os
import traceback
from pathlib import Path
from typing import List, Optional


def _is_existing_dir(path: str) -> bool:
    p = Path(path)
    if not p.is_dir():
        return False
    # Require at least one file/folder so an empty placeholder is not accepted.
    try:
        return any(p.iterdir())
    except Exception:
        return False


def _is_connection_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        kw in msg
        for kw in [
            "connection",
            "connect",
            "timeout",
            "network",
            "couldn't connect",
            "offline",
            "no route",
            "unreachable",
            "failed to establish",
        ]
    )


def _known_modelscope_aliases(repo_id: str) -> List[str]:
    """Return a best-effort list of ModelScope aliases for well-known HF ids."""
    aliases = []
    lowered = repo_id.lower()

    # Qwen2-Audio family
    if "qwen2-audio" in lowered:
        base = repo_id.split("/")[-1]
        for org in ("qwen", "Qwen"):
            aliases.append(f"{org}/{base}")
        # Common variants
        if "instruct" not in lowered:
            for org in ("qwen", "Qwen"):
                aliases.append(f"{org}/Qwen2-Audio-7B")
        else:
            for org in ("qwen", "Qwen"):
                aliases.append(f"{org}/Qwen2-Audio-7B-Instruct")

    # LLaVA 1.5 7B family
    if "llava" in lowered and "1.5" in lowered and "7b" in lowered:
        for org in ("liuhaotian", "llava-hf", "AI-ModelScope"):
            aliases.append(f"{org}/llava-v1.5-7b")
            aliases.append(f"{org}/llava-1.5-7b-hf")
            aliases.append(f"{org}/llava-1.5-7b")

    return aliases


def _try_hf_download(repo_id: str, hf_endpoint: Optional[str]) -> Optional[str]:
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print(f"[model_loader] huggingface_hub not available: {e}")
        return None

    old_endpoint = os.environ.get("HF_ENDPOINT")
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
    try:
        local_path = snapshot_download(
            repo_id,
            local_files_only=False,
            endpoint=hf_endpoint or None,
        )
        return local_path
    except Exception as e:
        if _is_connection_error(e):
            print(f"[model_loader] HF endpoint {hf_endpoint or 'default'} unreachable for {repo_id}: {e}")
            return None
        # Re-raise unexpected errors (bad repo name, missing file, etc.)
        raise
    finally:
        if hf_endpoint:
            if old_endpoint is None:
                os.environ.pop("HF_ENDPOINT", None)
            else:
                os.environ["HF_ENDPOINT"] = old_endpoint


def _try_hf_local_cache(repo_id: str, hf_endpoint: Optional[str]) -> Optional[str]:
    """Return a cached path without any network access."""
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return None
    try:
        return snapshot_download(repo_id, local_files_only=True, endpoint=hf_endpoint or None)
    except Exception:
        return None


def _try_modelscope_download(
    repo_id: str,
    endpoint: Optional[str] = None,
    aliases: Optional[List[str]] = None,
) -> Optional[str]:
    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
    except Exception as e:
        print(f"[model_loader] modelscope not available: {e}")
        return None

    candidates = [repo_id] + list(aliases or [])
    for mid in candidates:
        try:
            return ms_snapshot_download(
                mid,
                local_files_only=False,
                endpoint=endpoint or None,
            )
        except Exception as e:
            if _is_connection_error(e):
                print(f"[model_loader] ModelScope endpoint {endpoint or 'default'} unreachable for {mid}: {e}")
                continue
            # Alias may simply not exist; try the next one.
            print(f"[model_loader] ModelScope {mid} not available: {e}")
            continue
    return None


def resolve_model_path(
    model_id_or_dir: str,
    allow_download: bool = True,
    hf_endpoint: Optional[str] = None,
    use_modelscope: bool = True,
    modelscope_aliases: Optional[List[str]] = None,
    modelscope_endpoint: Optional[str] = None,
) -> str:
    """Resolve a model identifier or local path to a local directory.

    Args:
        model_id_or_dir: Either a local directory or a Hugging Face model id.
        allow_download: Whether network downloads are permitted.
        hf_endpoint: Optional Hugging Face mirror (e.g. https://hf-mirror.com).
        use_modelscope: Try ModelScope if HF is unavailable or disabled.
        modelscope_aliases: Extra ModelScope ids to try (in addition to the HF id).
        modelscope_endpoint: Optional ModelScope mirror endpoint.

    Returns:
        Absolute path to the model directory.

    Raises:
        RuntimeError: If the model cannot be found or downloaded.
    """
    if _is_existing_dir(model_id_or_dir):
        return str(Path(model_id_or_dir).resolve())

    # Try the local Hugging Face cache first (no network).
    local_path = _try_hf_local_cache(model_id_or_dir, hf_endpoint)
    if local_path:
        return local_path

    if not allow_download:
        raise RuntimeError(
            f"Model not found locally and download disabled: {model_id_or_dir}. "
            "Set --image-model-dir / --audio-model-dir to a local copy, or enable download."
        )

    # Try Hugging Face (possibly via a mirror).
    try:
        local_path = _try_hf_download(model_id_or_dir, hf_endpoint)
        if local_path:
            return local_path
    except Exception as e:
        print(f"[model_loader] HF download failed for {model_id_or_dir}: {e}")
        traceback.print_exc()

    # Try ModelScope.
    if use_modelscope:
        aliases = list(modelscope_aliases or [])
        aliases.extend(_known_modelscope_aliases(model_id_or_dir))
        # Deduplicate while preserving order.
        seen = set()
        unique_aliases = []
        for a in aliases:
            if a not in seen and a != model_id_or_dir:
                seen.add(a)
                unique_aliases.append(a)
        local_path = _try_modelscope_download(
            model_id_or_dir,
            endpoint=modelscope_endpoint,
            aliases=unique_aliases,
        )
        if local_path:
            return local_path

    raise RuntimeError(
        f"Could not resolve model {model_id_or_dir}. "
        f"Tried HF (endpoint={hf_endpoint or 'default'}), "
        f"HF cache, and ModelScope (endpoint={modelscope_endpoint or 'default'})."
    )
