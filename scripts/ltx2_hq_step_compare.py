#!/usr/bin/env python3
"""Per-step parity compare: Mojo HQ pipeline vs the ltx_core reference run.

Inputs are the dumps from scripts/ltx2_hq_ref_run.py (reference) and the Mojo
HQ pipeline's step dumps (same keys). Prints per-step cos + rel-L2 + std ratio
for video and audio, per stage — the accumulation-aware gate (tenet: gate loops
per-step, check magnitude, not just final cosine).

Usage:
  python scripts/ltx2_hq_step_compare.py <ref_dir> <mojo_dir> [--bar 0.999]
Each dir holds stage1_steps.safetensors / stage2_steps.safetensors /
final_latents.safetensors (+ upsampler.safetensors in ref).
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    # These product tensors contain millions of values. F32 dot/norm reductions
    # lose enough precision at that size to move the reported cosine across the
    # 0.999 release gate even when relative-L2 is unchanged. The tensors remain
    # in their storage dtype on disk; only oracle-side metric reduction is F64.
    a = a.double().flatten()
    b = b.double().flatten()
    value = float((a @ b) / (a.norm() * b.norm() + 1e-12))
    return max(-1.0, min(1.0, value))


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().flatten()
    b = b.double().flatten()
    return float((a - b).norm() / (b.norm() + 1e-12))


CREATOR_REVISION = "780984275fd47128b02bef9b5c085404276866ee"
REPO = Path(__file__).resolve().parents[1]
MOJO_RUNNER = REPO / "output/bin/ltx2_video_smoke_runner"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_file(ref_path: str, mojo_path: str, bar: float) -> tuple[bool, dict[str, dict[str, float | list[int] | bool]]]:
    ref = load_file(ref_path)
    mojo = load_file(mojo_path)
    ok = True
    metrics: dict[str, dict[str, float | list[int] | bool]] = {}
    for k in sorted(ref.keys()):
        if k not in mojo:
            print(f"  {k:24s} MISSING in mojo dump")
            ok = False
            metrics[k] = {"passed": False}
            continue
        r, m = ref[k], mojo[k]
        if tuple(r.shape) != tuple(m.shape):
            print(f"  {k:24s} SHAPE ref{tuple(r.shape)} != mojo{tuple(m.shape)}")
            ok = False
            metrics[k] = {
                "passed": False,
                "reference_shape": list(r.shape),
                "mojo_shape": list(m.shape),
            }
            continue
        c = cos(m, r)
        rl = rel_l2(m, r)
        sr = float(m.double().std() / (r.double().std() + 1e-12))
        flag = "PASS" if c >= bar else "FAIL"
        if c < bar:
            ok = False
        print(f"  {k:24s} cos={c:.7f} relL2={rl:.5f} std_ratio={sr:.4f} {flag}")
        metrics[k] = {
            "cosine": c,
            "relative_l2": rl,
            "std_ratio": sr,
            "shape": list(r.shape),
            "passed": c >= bar,
        }
    return ok, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_dir")
    ap.add_argument("mojo_dir")
    ap.add_argument("--bar", type=float, default=0.999)
    ap.add_argument("--json-out")
    ap.add_argument("--mojo-runner", default=str(MOJO_RUNNER))
    args = ap.parse_args()
    mojo_runner = Path(args.mojo_runner).resolve()
    if not mojo_runner.is_file():
        ap.error(f"Mojo runner is absent: {mojo_runner}")

    all_ok = True
    comparisons: dict[str, object] = {}
    pairs = [
        ("stage1_steps", "stage1_steps.safetensors", "stage1_steps.safetensors", True),
        ("stage2_steps", "stage2_steps.safetensors", "stage2_steps.safetensors", True),
        ("upsampler", "upsampler.safetensors", "upsampler.safetensors", True),
        ("contexts", "contexts.safetensors", "contexts.safetensors", False),
    ]
    if os.path.exists(os.path.join(args.ref_dir, "stage1_final.safetensors")):
        final_label = "stage1_final"
        pairs.append(("stage1_final", "stage1_final.safetensors", "final_latents.safetensors", True))
    else:
        final_label = "final_latents"
        pairs.append(("final_latents", "final_latents.safetensors", "final_latents.safetensors", True))

    for label, ref_name, mojo_name, required_pair in pairs:
        rp = os.path.join(args.ref_dir, ref_name)
        mp = os.path.join(args.mojo_dir, mojo_name)
        if not os.path.exists(rp):
            continue
        if not os.path.exists(mp):
            if required_pair:
                print(f"[{label}] required mojo-side dump is missing: {mp}")
                comparisons[label] = {"passed": False, "error": "missing_mojo_artifact"}
                all_ok = False
            else:
                print(f"[{label}] optional mojo-side dump is absent (not gated)")
                comparisons[label] = {"skipped": True, "reason": "optional_mojo_artifact_absent"}
            continue
        print(f"[{label}] {ref_name} -> {mojo_name}")
        passed, metrics = compare_file(rp, mp, args.bar)
        comparisons[label] = {
            "passed": passed,
            "reference": os.path.abspath(rp),
            "mojo": os.path.abspath(mp),
            "tensors": metrics,
        }
        if not passed:
            all_ok = False

    required = ("stage1_steps", final_label)
    for label in required:
        if label not in comparisons:
            print(f"[{label}] required fresh stage-1 evidence is missing")
            comparisons[label] = {"passed": False, "error": "missing_reference_artifact"}
            all_ok = False

    gated_scores: list[tuple[str, float]] = []
    for comparison_name, comparison in comparisons.items():
        if not isinstance(comparison, dict) or comparison.get("skipped"):
            continue
        tensors = comparison.get("tensors")
        if not isinstance(tensors, dict):
            continue
        for tensor_name, tensor_metrics in tensors.items():
            if isinstance(tensor_metrics, dict) and isinstance(tensor_metrics.get("cosine"), float):
                gated_scores.append(
                    (f"{comparison_name}:{tensor_name}", tensor_metrics["cosine"])
                )
    first_failure = next(
        (name for name, score in gated_scores if score < args.bar),
        None,
    )

    report = {
        "schema": "serenity.ltx2.sampler_parity.v1",
        "creator_revision": CREATOR_REVISION,
        "metric_reduction_dtype": "float64",
        "mojo_runner": str(mojo_runner),
        "mojo_runner_sha256": sha256_file(mojo_runner),
        "bar": args.bar,
        "passed": all_ok,
        "reference_dir": os.path.abspath(args.ref_dir),
        "mojo_dir": os.path.abspath(args.mojo_dir),
        "minimum_cosine": min((score for _, score in gated_scores), default=None),
        "first_failing_tensor": first_failure,
        "comparisons": comparisons,
    }
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")

    print("VERDICT:", "PASS — per-step parity holds" if all_ok
          else "DIVERGES — first failing step above is the isolation point")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
