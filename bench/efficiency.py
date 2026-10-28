"""Params, GFLOPs and batch-1 CPU latency for an ONNX pose model.

Params and FLOPs are counted from the ONNX graph itself (no PyTorch checkpoint needed), so every
exported model - including top-down ones - is measured the same way. GFLOPs follows the Ultralytics
convention of 2 x multiply-accumulates, over Conv / MatMul / Gemm at a fixed input
size. Count it on the fp32 export: a quantized graph's integer ops aren't counted.
"""

from __future__ import annotations

import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference


def cpu_name() -> str:
    try:  # wmic is gone on Windows 11; ask CIM via PowerShell
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
            text=True, stderr=subprocess.DEVNULL,
        )
        return out.strip() or platform.processor()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor()


def count_params(model_path: str | Path) -> int:
    """Number of weight elements in the graph's float initializers (fp32 export)."""
    model = onnx.load(str(model_path))
    total = 0
    for init in model.graph.initializer:
        if init.data_type in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16):
            total += int(np.prod(init.dims)) if init.dims else 1
    return total


def count_gflops(model_path: str | Path, input_hw: tuple[int, int] = (640, 640)) -> float:
    """2 x MACs of Conv/MatMul/Gemm nodes at a static batch-1 input of input_hw."""
    model = onnx.load(str(model_path))
    inp = model.graph.input[0]
    dims = inp.type.tensor_type.shape.dim
    static = [1, int(dims[1].dim_value) or 3, *input_hw]
    for d, v in zip(dims, static):
        d.ClearField("dim_param")
        d.dim_value = v
    model = shape_inference.infer_shapes(model)

    shapes: dict[str, list[int]] = {}
    for vi in list(model.graph.value_info) + list(model.graph.output) + list(model.graph.input):
        shapes[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    weights = {i.name: list(i.dims) for i in model.graph.initializer}
    shapes.update(weights)

    macs = 0
    for node in model.graph.node:
        if node.op_type == "Conv" and len(node.input) > 1:
            w = shapes.get(node.input[1])
            out = shapes.get(node.output[0])
            if not w or not out or 0 in out:
                continue
            # weight (Cout, Cin/group, kh, kw), output (N, Cout, H, W): each output element costs
            # Cin/group * kh * kw MACs, so grouped/depthwise convs are handled by the weight shape.
            macs += int(np.prod(out)) * int(np.prod(w[1:]))
        elif node.op_type in ("MatMul", "Gemm"):
            a = shapes.get(node.input[0])
            out = shapes.get(node.output[0])
            if not a or not out or 0 in out or 0 in a:
                continue
            trans_a = any(x.name == "transA" and x.i for x in node.attribute)
            macs += int(np.prod(out)) * (a[-2] if trans_a else a[-1])
    return 2.0 * macs / 1e9


def latency_ms(fn, warmup: int, n: int) -> dict[str, float]:
    """Median / p90 / mean wall time of fn() in milliseconds after `warmup` untimed calls."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return {
        "median": float(np.median(ts)),
        "p90": float(ts[min(len(ts) - 1, int(round(0.9 * (len(ts) - 1))))]),
        "mean": float(np.mean(ts)),
        "n": n,
    }
