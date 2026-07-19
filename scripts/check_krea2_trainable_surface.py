#!/usr/bin/env python3
"""Check Krea2 ai-toolkit vs Mojo trainable LoRA surface coverage.

This is a blocker/readiness gate, not parity evidence. The current expected
state is a known mismatch: ai-toolkit saves txtfusion LoRA tensors while the
Mojo Krea2 smoke saves only main block LoRA tensors. The common main-block
LoRA tensor names, shapes, and dtypes are expected to match.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open


DEFAULT_AI_TOOLKIT = Path(
    "/home/alex/ai-toolkit/output/my_first_lora_v1/my_first_lora_v1_000002994.safetensors"
)
DEFAULT_MOJO = Path(
    "/tmp/krea2_devicegrad_realcache_smoke/krea2_devicegrad_realcache_smoke_2.safetensors"
)


@dataclass(frozen=True)
class SurfaceSummary:
    path: Path
    keys: set[str]
    dtypes: tuple[str, ...]
    tensor_dtypes: dict[str, str]
    shapes: dict[str, tuple[int, ...]]

    @property
    def block_keys(self) -> set[str]:
        return {k for k in self.keys if k.startswith("diffusion_model.blocks.")}

    @property
    def txtfusion_keys(self) -> set[str]:
        return {k for k in self.keys if "txtfusion" in k}

    @property
    def non_block_keys(self) -> set[str]:
        return self.keys - self.block_keys

    @property
    def target_prefix_count(self) -> int:
        return len({_target_prefix(k) for k in self.keys})


def _load_surface(path: Path) -> SurfaceSummary:
    if not path.exists():
        raise FileNotFoundError(path)
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        tensor_dtypes: dict[str, str] = {}
        shapes: dict[str, tuple[int, ...]] = {}
        for key in keys:
            tensor = handle.get_tensor(key)
            tensor_dtypes[key] = str(tensor.dtype)
            shapes[key] = tuple(int(d) for d in tensor.shape)
        dtypes = sorted(set(tensor_dtypes.values()))
    return SurfaceSummary(
        path=path,
        keys=keys,
        dtypes=tuple(dtypes),
        tensor_dtypes=tensor_dtypes,
        shapes=shapes,
    )


def _sample(items: set[str], limit: int = 8) -> list[str]:
    return sorted(items)[:limit]


def _target_prefix(key: str) -> str:
    for suffix in (".lora_A.weight", ".lora_B.weight"):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def _meta_mismatch_sample(
    keys: set[str], ai: SurfaceSummary, mojo: SurfaceSummary, limit: int = 4
) -> list[str]:
    out: list[str] = []
    for key in sorted(keys)[:limit]:
        out.append(
            f"{key}: ai_shape={ai.shapes[key]} mojo_shape={mojo.shapes[key]} "
            f"ai_dtype={ai.tensor_dtypes[key]} mojo_dtype={mojo.tensor_dtypes[key]}"
        )
    return out


def _print_summary(label: str, summary: SurfaceSummary) -> None:
    print(
        f"[krea2-surface] {label}: path={summary.path} total={len(summary.keys)} "
        f"blocks={len(summary.block_keys)} txtfusion={len(summary.txtfusion_keys)} "
        f"non_block={len(summary.non_block_keys)} target_prefixes={summary.target_prefix_count} "
        f"dtypes={list(summary.dtypes)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai-toolkit", default=str(DEFAULT_AI_TOOLKIT))
    parser.add_argument("--mojo", default=str(DEFAULT_MOJO))
    parser.add_argument(
        "--expect-known-mismatch",
        action="store_true",
        help="pass only when the current txtfusion surface mismatch is detected",
    )
    parser.add_argument(
        "--expect-match",
        action="store_true",
        help="pass only when Mojo covers the ai-toolkit key surface exactly",
    )
    args = parser.parse_args()

    if args.expect_known_mismatch == args.expect_match:
        print("[krea2-surface] FAIL choose exactly one expectation")
        return 2

    try:
        ai = _load_surface(Path(args.ai_toolkit))
        mojo = _load_surface(Path(args.mojo))
    except FileNotFoundError as exc:
        print(f"[krea2-surface] FAIL missing safetensors: {exc}")
        return 1

    _print_summary("ai_toolkit", ai)
    _print_summary("mojo", mojo)

    missing_from_mojo = ai.keys - mojo.keys
    extra_in_mojo = mojo.keys - ai.keys
    missing_txtfusion = missing_from_mojo & ai.txtfusion_keys
    missing_non_txtfusion = {k for k in missing_from_mojo if "txtfusion" not in k}
    block_delta = ai.block_keys ^ mojo.block_keys
    common_keys = ai.keys & mojo.keys
    shape_mismatch = {k for k in common_keys if ai.shapes[k] != mojo.shapes[k]}
    dtype_mismatch = {
        k for k in common_keys if ai.tensor_dtypes[k] != mojo.tensor_dtypes[k]
    }

    print(
        "[krea2-surface] delta: "
        f"common_keys={len(common_keys)} "
        f"missing_from_mojo={len(missing_from_mojo)} "
        f"missing_txtfusion={len(missing_txtfusion)} "
        f"missing_non_txtfusion={len(missing_non_txtfusion)} "
        f"extra_in_mojo={len(extra_in_mojo)} "
        f"block_key_delta={len(block_delta)} "
        f"shape_mismatch={len(shape_mismatch)} "
        f"dtype_mismatch={len(dtype_mismatch)}"
    )
    if missing_from_mojo:
        print(f"[krea2-surface] missing_sample={_sample(missing_from_mojo)}")
    if extra_in_mojo:
        print(f"[krea2-surface] extra_sample={_sample(extra_in_mojo)}")
    if shape_mismatch:
        print(
            "[krea2-surface] shape_mismatch_sample="
            f"{_meta_mismatch_sample(shape_mismatch, ai, mojo)}"
        )
    if dtype_mismatch:
        print(
            "[krea2-surface] dtype_mismatch_sample="
            f"{_meta_mismatch_sample(dtype_mismatch, ai, mojo)}"
        )

    known_mismatch = (
        len(ai.keys) == 512
        and len(ai.block_keys) == 448
        and len(ai.txtfusion_keys) == 64
        and ai.target_prefix_count == 256
        and len(mojo.keys) == 448
        and len(mojo.block_keys) == 448
        and len(mojo.txtfusion_keys) == 0
        and mojo.target_prefix_count == 224
        and len(common_keys) == 448
        and len(missing_from_mojo) == 64
        and len(missing_txtfusion) == 64
        and len(missing_non_txtfusion) == 0
        and len(extra_in_mojo) == 0
        and len(block_delta) == 0
        and len(shape_mismatch) == 0
        and len(dtype_mismatch) == 0
        and ai.dtypes == ("torch.bfloat16",)
        and mojo.dtypes == ("torch.bfloat16",)
    )
    exact_match = (
        ai.keys == mojo.keys
        and ai.tensor_dtypes == mojo.tensor_dtypes
        and ai.shapes == mojo.shapes
        and len(ai.keys) > 0
    )

    if args.expect_known_mismatch:
        if not known_mismatch:
            print("[krea2-surface] FAIL expected known txtfusion mismatch was not reproduced")
            return 1
        print(
            "[krea2-surface] PASS known_mismatch "
            "ai_toolkit_total=512 mojo_total=448 common_keys=448 "
            "missing_txtfusion=64 block_key_delta=0 shape_mismatch=0 dtype_mismatch=0 "
            "ai_target_prefixes=256 mojo_target_prefixes=224"
        )
        print(
            "[krea2-surface] scope=trainable-surface blocker only; "
            "not gradient, optimizer, loss, save/resume, speed, or convergence parity"
        )
        return 0

    if not exact_match:
        print("[krea2-surface] FAIL expected exact surface match")
        return 1
    print("[krea2-surface] PASS exact_match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
