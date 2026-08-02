#!/usr/bin/env python3
"""
x5_onnx_append_moe.py

Prototype of two ideas the user asked to explore on RDK X5:

1. "Append" to an ONNX model by expanding the shape of a weight matrix.
   This is a structural append, not training.  You can copy the old weights
   into the top-left block of a larger matrix, leave the new rows/cols as
   zeros (or random init), then fine-tune the new block on a remote/GPU box
   and re-export.

2. "MoE" by running several ONNX experts in Python and combining their
   outputs with a small gating network.  This lets you scale capacity without
   recompiling the model into a single ONNX (which is not possible on X5
   without the Horizon host-side `hbdk' toolchain).

Requirements on X5:
  pip3 install onnx onnxruntime numpy

The script is deliberately tiny so it runs on 2 GB ARM devices.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# ONNX helpers (no PyTorch / torch dependency on X5)
# ---------------------------------------------------------------------------

def _numpy_to_onnx_type(np_type: np.dtype) -> int:
    mapping = {
        np.float32: 1,  # onnx.TensorProto.FLOAT
        np.uint8: 2,
        np.int8: 3,
        np.uint16: 4,
        np.int16: 5,
        np.int32: 6,
        np.int64: 7,
        np.float64: 11,
        np.uint32: 12,
        np.uint64: 13,
    }
    return mapping.get(np_type, 1)


def make_mlp_onnx(input_dim: int, hidden_dim: int, output_dim: int, seed: int = 0) -> bytes:
    """Build a tiny 2-layer MLP as an ONNX model (byte string)."""
    np.random.seed(seed)
    W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.1
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    W2 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * 0.1
    b2 = np.zeros(output_dim, dtype=np.float32)

    try:
        import onnx
        from onnx import helper, TensorProto, numpy_helper

        w1_init = numpy_helper.from_array(W1, name="W1")
        b1_init = numpy_helper.from_array(b1, name="b1")
        w2_init = numpy_helper.from_array(W2, name="W2")
        b2_init = numpy_helper.from_array(b2, name="b2")

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, input_dim])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, output_dim])

        nodes = [
            helper.make_node("MatMul", inputs=["X", "W1"], outputs=["H1"], name="matmul1"),
            helper.make_node("Add", inputs=["H1", "b1"], outputs=["A1"], name="add1"),
            helper.make_node("Relu", inputs=["A1"], outputs=["R1"], name="relu1"),
            helper.make_node("MatMul", inputs=["R1", "W2"], outputs=["H2"], name="matmul2"),
            helper.make_node("Add", inputs=["H2", "b2"], outputs=["Y"], name="add2"),
        ]

        graph = helper.make_graph(
            nodes, "mlp", [X], [Y], [w1_init, b1_init, w2_init, b2_init]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8
        return model.SerializeToString()
    except ImportError:
        raise RuntimeError("Please run: pip3 install onnx")


# ---------------------------------------------------------------------------
# Append a matrix in-place in an ONNX model
# ---------------------------------------------------------------------------

def append_onnx_matrix(
    src_path: Path,
    dst_path: Path,
    initializer_name: str,
    new_rows: int = 0,
    new_cols: int = 0,
    fill: str = "zeros",
) -> dict[str, Any]:
    """Create a new ONNX model where one weight matrix has extra rows/cols.

    Old weights are copied into the top-left block of the expanded matrix.
    The newly added rows/cols are filled according to `fill` (zeros / random).

    Returns a report with the original and new shapes.
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(src_path))
    target = None
    for init in model.graph.initializer:
        if init.name == initializer_name:
            target = init
            break
    if target is None:
        raise ValueError(f"Initializer '{initializer_name}' not found in {src_path}")

    arr = numpy_helper.to_array(target)
    old_shape = arr.shape
    new_shape = list(old_shape)
    new_shape[0] += new_rows
    if len(new_shape) > 1:
        new_shape[1] += new_cols

    new_arr = np.zeros(new_shape, dtype=arr.dtype)
    if fill == "random":
        scale = float(np.std(arr)) if arr.size else 0.01
        new_arr = np.random.randn(*new_shape).astype(arr.dtype) * max(scale, 0.01)
        new_arr[: old_shape[0], : old_shape[1]] = arr
    else:
        # zeros fill: copy old weights, new part stays zero
        new_arr[: old_shape[0], : old_shape[1]] = arr

    new_init = numpy_helper.from_array(new_arr, name=initializer_name)
    target.CopyFrom(new_init)

    onnx.save(model, str(dst_path))
    return {
        "initializer": initializer_name,
        "old_shape": list(old_shape),
        "new_shape": list(new_shape),
        "src": str(src_path),
        "dst": str(dst_path),
    }


# ---------------------------------------------------------------------------
# MoE: multiple ONNX experts + small gating MLP in NumPy
# ---------------------------------------------------------------------------

class NumpyGatingNetwork:
    """Tiny softmax gating network.  Weights are fixed here for demo."""

    def __init__(self, input_dim: int, num_experts: int, seed: int = 1):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(input_dim, num_experts).astype(np.float32) * 0.05
        self.b = np.zeros(num_experts, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        logits = x @ self.W + self.b
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)


class OnnxMoE:
    """Run `expert_paths` ONNX models and combine outputs with gating."""

    def __init__(self, expert_paths: list[Path], input_dim: int, seed: int = 1):
        import onnxruntime as ort

        self.experts = []
        for p in expert_paths:
            if not p.exists():
                raise FileNotFoundError(p)
            # `onnxruntime` on X5 is built for CPUExecutionProvider
            sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
            self.experts.append(sess)

        self.gate = NumpyGatingNetwork(input_dim, len(expert_paths), seed=seed)
        self.input_name = self.experts[0].get_inputs()[0].name
        self.output_name = self.experts[0].get_outputs()[0].name

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        start = time.time()
        weights = self.gate(x)
        expert_outputs: list[np.ndarray] = []
        expert_times: list[float] = []
        for sess in self.experts:
            t0 = time.time()
            out = sess.run([self.output_name], {self.input_name: x})[0]
            expert_times.append((time.time() - t0) * 1000.0)
            expert_outputs.append(out)

        # Weighted sum: shape (batch, output_dim)
        stack = np.stack(expert_outputs, axis=0)  # (E, B, D)
        # weights shape (B, E) -> (E, B, 1) to broadcast over output dim
        w = np.transpose(weights, (1, 0))[:, :, np.newaxis]
        combined = np.sum(stack * w, axis=0)
        total_ms = (time.time() - start) * 1000.0
        return combined, {
            "expert_weights": weights.tolist(),
            "expert_latencies_ms": expert_times,
            "total_latency_ms": total_ms,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_experts(root: Path, input_dim: int, hidden_dim: int, output_dim: int) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    experts: list[Path] = []
    for i in range(2):
        p = root / f"expert_{i}.onnx"
        data = make_mlp_onnx(input_dim, hidden_dim, output_dim, seed=i)
        p.write_bytes(data)
        experts.append(p)
    return experts


def main() -> int:
    parser = argparse.ArgumentParser(description="ONNX append / MoE prototype for RDK X5")
    parser.add_argument("--workdir", default="runtime_store/x5_onnx_proto")
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=16)
    parser.add_argument("--append-rows", type=int, default=8)
    parser.add_argument("--append-cols", type=int, default=8)
    parser.add_argument("--fill", default="zeros", choices=["zeros", "random"])
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"workdir": str(workdir), "args": vars(args)}

    print("[x5-onnx] building two tiny expert ONNX models...", flush=True)
    experts = build_experts(workdir, args.input_dim, args.hidden_dim, args.output_dim)
    print(f"[x5-onnx] experts: {experts}", flush=True)

    print("[x5-onnx] appending rows/cols to W1 of expert_0...", flush=True)
    base = experts[0]
    expanded = workdir / "expert_0_expanded.onnx"
    append_report = append_onnx_matrix(
        base, expanded, "W1",
        new_rows=args.append_rows, new_cols=args.append_cols, fill=args.fill,
    )
    print(f"[x5-onnx] append report: {json.dumps(append_report)}", flush=True)
    report["append"] = append_report

    # Because we expanded W1 but not the matching b1/W2, the model no longer
    # runs.  This is fine for a pure "matrix-range expansion" demo, but we also
    # create a second expert that is already the larger shape.
    print("[x5-onnx] building a pre-sized larger expert for MoE...", flush=True)
    big = workdir / "expert_big.onnx"
    big_data = make_mlp_onnx(args.input_dim, args.hidden_dim + args.append_rows, args.output_dim, seed=99)
    big.write_bytes(big_data)

    print("[x5-onnx] running MoE with two experts...", flush=True)
    moe = OnnxMoE([experts[1], big], args.input_dim)
    x = np.random.randn(args.batch, args.input_dim).astype(np.float32)
    out, moe_report = moe.predict(x)
    moe_report["output_shape"] = list(out.shape)
    print(f"[x5-onnx] MoE report: {json.dumps(moe_report, indent=2)}", flush=True)
    report["moe"] = moe_report

    report_path = workdir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[x5-onnx] full report written to {report_path}", flush=True)

    # Suggest the next real step
    print("\n[x5-onnx] Next real step:")
    print("  1. Fine-tune the expanded block on a remote/GPU machine using the")
    print("     original training data + the new slice, then re-export to ONNX.")
    print("  2. If you need multiple experts on BPU, convert each expert to a")
    print("     Horizon .bin on a host PC and run them sequentially on X5 with a")
    print("     small gating network in C++/Python (this prototype uses")
    print("     onnxruntime on the X5 CPU).")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
