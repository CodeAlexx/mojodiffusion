#!/usr/bin/env python3
"""Benchmark the real ComfyUI H3 Sage patch against Comfy Kitchen INT8.

Run this with ComfyUI's Python environment.  Both approximate backends receive
the same deterministic BF16 NHD tensors at the real MiniMax-H3 geometries.
The KJNodes Sage path consumes and mutates Q/K/V, so fresh clones are prepared
and synchronized *before* its CUDA timing event; the copy is not attention
work and a real H3 block produces fresh Q/K/V for every invocation.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path


def _gpu_snapshot() -> dict[str, str]:
    fields = (
        "name,compute_cap,temperature.gpu,power.draw,clocks.sm,clocks.mem,"
        "memory.used,memory.free,utilization.gpu"
    )
    command = [
        "nvidia-smi",
        f"--query-gpu={fields}",
        "--format=csv,noheader,nounits",
    ]
    names = fields.split(",")
    try:
        values = subprocess.check_output(command, text=True).strip().split(", ")
    except (OSError, subprocess.CalledProcessError):
        return {}
    return dict(zip(names, values, strict=False))


def _load_comfyui_h3_sage():
    # KJNodes registers preview helpers at import time.  A standalone benchmark
    # has no PromptServer, so provide only the inert surface those imports read.
    import server

    class _Router:
        frozen = True

    class _App:
        router = _Router()

    class _PromptServer:
        last_node_id = ""
        client_id = ""
        app = _App()

        def send_sync(self, *_args, **_kwargs):
            return None

    server.PromptServer.instance = _PromptServer()
    module = importlib.import_module("custom_nodes.ComfyUI-KJNodes.nodes.ltxv_nodes")
    if module._cuda_archs != ["sm86"]:
        raise RuntimeError(
            "this measured reference gate is pinned to the local SM86 path; "
            f"KJNodes reported {module._cuda_archs!r}"
        )
    return module


def _stats(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p90_index = min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)
    return {
        "count": len(samples),
        "min_ms": min(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "p90_ms": ordered[p90_index],
        "stdev_ms": statistics.pstdev(samples),
    }


def _error_metrics(candidate, reference) -> dict[str, float | int]:
    import torch

    a = candidate.float().flatten()
    b = reference.float().flatten()
    delta = a - b
    result = {
        "cosine": torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
        "relative_l2": (
            torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(b)
        ).item(),
        "max_abs": delta.abs().max().item(),
        "mean_abs": delta.abs().mean().item(),
        "nonfinite": int((~torch.isfinite(a)).sum().item()),
    }
    del a, b, delta
    return result


def _time_cuda(call) -> tuple[float, object]:
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = call()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)), output


def _run_geometry(sequence: int, heads: int, dim: int, warmups: int, rounds: int):
    import comfy_kitchen
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    kj = _load_comfyui_h3_sage()
    scale = dim**-0.5
    generator = torch.Generator(device="cuda")
    generator.manual_seed(24680 + sequence)
    q = torch.randn(
        (1, sequence, heads, dim),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        (1, sequence, heads, dim),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.randn(
        (1, sequence, heads, dim),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()

    def prepare_sage():
        prepared = [q.clone(), k.clone(), v.clone()]
        torch.cuda.synchronize()
        return lambda: kj._sageattn_int8_fp8_nhd(prepared, torch.bfloat16)

    def call_ck():
        # Comfy Kitchen accepts HND.  These transposed views retain H3's actual
        # contiguous NHD bytes and strides, matching the Mojo bridge geometry.
        return comfy_kitchen.int8_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            scale=scale,
        ).transpose(1, 2)

    def call_cudnn():
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            return torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                scale=scale,
            ).transpose(1, 2)

    for _ in range(warmups):
        for backend in ("ck", "sage", "cudnn"):
            call = call_ck if backend == "ck" else call_cudnn
            if backend == "sage":
                call = prepare_sage()
            _, output = _time_cuda(call)
            del output

    samples: dict[str, list[float]] = {"ck": [], "sage": [], "cudnn": []}
    orders = (
        ("ck", "sage", "cudnn"),
        ("sage", "cudnn", "ck"),
        ("cudnn", "ck", "sage"),
    )
    for round_index in range(rounds):
        for backend in orders[round_index % len(orders)]:
            call = call_ck if backend == "ck" else call_cudnn
            if backend == "sage":
                call = prepare_sage()
            elapsed, output = _time_cuda(call)
            samples[backend].append(elapsed)
            del output

    _, exact = _time_cuda(call_cudnn)
    _, ck_output = _time_cuda(call_ck)
    _, sage_output = _time_cuda(prepare_sage())
    metrics = {
        "ck_vs_cudnn": _error_metrics(ck_output, exact),
        "sage_vs_cudnn": _error_metrics(sage_output, exact),
        "ck_vs_sage": _error_metrics(ck_output, sage_output),
    }
    # Keep Q/K/V bound until the closures above leave scope.  Explicitly
    # deleting closure cells makes static analysis (and future callback reuse)
    # ambiguous; function return releases them immediately after this report.
    del exact, ck_output, sage_output
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    timing = {backend: _stats(values) for backend, values in samples.items()}
    timing["ck_speedup_vs_sage_median"] = (
        timing["sage"]["median_ms"] / timing["ck"]["median_ms"]
    )
    timing["ck_speedup_vs_cudnn_median"] = (
        timing["cudnn"]["median_ms"] / timing["ck"]["median_ms"]
    )
    timing["sage_speedup_vs_cudnn_median"] = (
        timing["cudnn"]["median_ms"] / timing["sage"]["median_ms"]
    )
    return {
        "sequence": sequence,
        "heads": heads,
        "head_dim": dim,
        "timing": timing,
        "numeric": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", default="19029,21291")
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import comfy_kitchen
    import torch
    from sageattention.core import get_cuda_arch_versions

    sequences = [int(value) for value in args.sequences.split(",")]
    started = time.time()
    report = {
        "schema": "serenity.h3.attention-reference-ab.v1",
        "honesty_contract": {
            "same_bf16_inputs": True,
            "same_gpu_process": True,
            "alternating_order": True,
            "cuda_event_synchronized": True,
            "sage_clone_outside_timing": True,
            "sage_path": "ComfyUI KJNodes MiniMax-H3 SM86 memory-efficient Sage patch",
            "ck_path": "Comfy Kitchen int8_attention v0.2.31",
            "exact_path": "PyTorch forced cuDNN SDPA",
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "sageattention": importlib.metadata.version("sageattention"),
            "comfy_kitchen": importlib.metadata.version("comfy-kitchen"),
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "sage_archs": get_cuda_arch_versions(),
            "gpu_before": _gpu_snapshot(),
        },
        "warmups": args.warmups,
        "rounds": args.rounds,
        "geometries": [],
    }
    if not comfy_kitchen.int8_attention_is_available():
        raise RuntimeError("Comfy Kitchen INT8 attention is unavailable")
    if report["environment"]["sage_archs"] != ["sm86"]:
        raise RuntimeError("official SageAttention did not load the SM86 build")

    for sequence in sequences:
        report["geometries"].append(
            _run_geometry(
                sequence, args.heads, args.head_dim, args.warmups, args.rounds
            )
        )
    report["environment"]["gpu_after"] = _gpu_snapshot()
    report["elapsed_seconds"] = time.time() - started

    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
