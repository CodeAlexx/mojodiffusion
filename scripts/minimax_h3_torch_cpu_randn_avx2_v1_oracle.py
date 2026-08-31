#!/usr/bin/env python3
"""Pinned exact-byte oracle for ``torch_cpu_randn_f32_avx2_v1``.

This is deliberately a local-build compatibility profile, not a promise about
arbitrary PyTorch releases or CPU dispatch targets.  It executes the installed
PyTorch CPU kernel with ``ATEN_CPU_CAPABILITY=avx2`` and records contiguous F32
C-order output bytes plus the complete CPU generator state before and after
each draw.  Canonical fixture bytes contain no timestamps or host paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path


PROFILE = "torch_cpu_randn_f32_avx2_v1"
TORCH_VERSION = "2.12.0+cu130"
TORCH_GIT_VERSION = "7661cd9c6b841b62b7f411aa52ec51f05457263b"
TORCH_BUILD_CONFIG_SHA256 = (
    "3e4fc6c11e746aa3905f3b7f7ba4a4f8b32019c6bad21301361a6df63b94b6fb"
)
HEADER_SHA256 = {
    "ATen/core/MT19937RNGEngine.h": (
        "8df329422d29c965f1356b511caa26f0f72f74db97b1987509e7bafa94f57467"
    ),
    "ATen/core/TransformationHelper.h": (
        "8eff8a994ada7b28c0f0ed8c6564d232160de78b211ba49e49145bc586259260"
    ),
    "ATen/native/cpu/DistributionTemplates.h": (
        "dce75f8036a0dbeed823f7b282245673b74782e7af182755fd6babee7708331b"
    ),
    "ATen/native/cpu/avx_mathfun.h": (
        "e713d6fc64a0e034b13f2cf4b3b9d481fcb4ab0ea2a925154ee1286fe526d07a"
    ),
}
DERIVED_SEED = 6149187155249314651
DERIVED_SEED_LOW32 = DERIVED_SEED & 0xFFFFFFFF


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _load_torch():
    if os.environ.get("ATEN_CPU_CAPABILITY") != "avx2":
        raise RuntimeError("oracle requires ATEN_CPU_CAPABILITY=avx2 before import")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import torch

    if torch.__version__ != TORCH_VERSION:
        raise RuntimeError(f"torch version mismatch: {torch.__version__}")
    if torch.version.git_version != TORCH_GIT_VERSION:
        raise RuntimeError(f"torch git version mismatch: {torch.version.git_version}")
    if torch.backends.cpu.get_cpu_capability() != "AVX2":
        raise RuntimeError(
            "torch CPU dispatch mismatch: "
            + str(torch.backends.cpu.get_cpu_capability())
        )
    if _sha256(torch.__config__.show().encode("utf-8")) != TORCH_BUILD_CONFIG_SHA256:
        raise RuntimeError("torch build-config digest mismatch")
    return torch


def _verify_platform(torch) -> dict[str, object]:
    if sys.byteorder != "little":
        raise RuntimeError("profile is restricted to little-endian hosts")
    machine = platform.machine().lower()
    if machine not in {"x86_64", "amd64"}:
        raise RuntimeError(f"profile is restricted to x86-64, got {machine}")
    include_root = Path(torch.__file__).resolve().parent / "include"
    for relative, expected in HEADER_SHA256.items():
        path = include_root / relative
        if not path.is_file():
            raise RuntimeError(f"missing pinned torch header: {relative}")
        actual = _sha256(path.read_bytes())
        if actual != expected:
            raise RuntimeError(
                f"pinned torch header digest mismatch for {relative}: {actual}"
            )
    return {
        "arch": "x86_64",
        "byteorder": "little",
        "cpu_dispatch": "AVX2",
        "dtype": "float32",
        "layout": "contiguous_c_order",
    }


def _case(torch, label: str, seed: int, shape: list[int], reason: str) -> dict[str, object]:
    if not shape or any(dim <= 0 for dim in shape):
        raise RuntimeError(f"invalid oracle shape for {label}: {shape}")
    numel = 1
    for dim in shape:
        numel *= dim
    if numel < 16:
        raise RuntimeError(f"profile requires N >= 16, got {numel}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    state_before = bytes(generator.get_state().numpy())
    value = torch.randn(
        tuple(shape), generator=generator, dtype=torch.float32, device="cpu"
    )
    if not value.is_contiguous():
        raise RuntimeError(f"oracle output is unexpectedly non-contiguous: {label}")
    output = value.numpy().tobytes(order="C")
    state_after = bytes(generator.get_state().numpy())
    if len(state_before) != 5056 or len(state_after) != 5056:
        raise RuntimeError("pinned CPU generator state is not 5056 bytes")
    return {
        "label": label,
        "reason": reason,
        "seed_u64": seed,
        "seed_low32": seed & 0xFFFFFFFF,
        "shape": shape,
        "numel": numel,
        "uniform_words_consumed": numel if numel % 16 == 0 else numel + 16,
        "output_raw_le_hex": output.hex(),
        "output_sha256": _sha256(output),
        "state_before_raw_hex": state_before.hex(),
        "state_before_sha256": _sha256(state_before),
        "state_after_raw_hex": state_after.hex(),
        "state_after_sha256": _sha256(state_after),
    }


def _payload(torch) -> dict[str, object]:
    platform_contract = _verify_platform(torch)
    cases = [
        _case(torch, "threshold_n16", 42, [16], "minimum admitted N; no tail redraw"),
        _case(torch, "tail_n17", 42, [17], "N+16 tail-redraw consumption"),
        _case(torch, "tail_n31", 7, [31], "overlapping last-16 tail replacement"),
        _case(torch, "aligned_n32", 7, [32], "two complete AVX2 normal blocks"),
        _case(
            torch,
            "sha_derived_seed_high64",
            DERIVED_SEED,
            [16],
            "H3 sha256-derived seed with nonzero high 32 bits",
        ),
        _case(
            torch,
            "sha_derived_seed_low32_alias",
            DERIVED_SEED_LOW32,
            [16],
            "documents MT19937 output alias while full generator state retains seed",
        ),
        _case(
            torch,
            "one_frame_30x52",
            DERIVED_SEED,
            [24, 1, 30, 52],
            "existing H3 one-frame latent bucket",
        ),
        _case(
            torch,
            "one_frame_32x32",
            42,
            [24, 1, 32, 32],
            "1024-square H3 one-frame latent bucket",
        ),
    ]
    by_label = {str(case["label"]): case for case in cases}
    high = bytes.fromhex(str(by_label["sha_derived_seed_high64"]["output_raw_le_hex"]))
    alias = bytes.fromhex(
        str(by_label["sha_derived_seed_low32_alias"]["output_raw_le_hex"])
    )
    if high != alias:
        raise RuntimeError("pinned MT19937 low32 seed alias no longer holds")
    high_state = bytes.fromhex(
        str(by_label["sha_derived_seed_high64"]["state_after_raw_hex"])
    )
    alias_state = bytes.fromhex(
        str(by_label["sha_derived_seed_low32_alias"]["state_after_raw_hex"])
    )
    if high_state[:8] == alias_state[:8] or high_state[8:] != alias_state[8:]:
        raise RuntimeError("full-seed state / low32 engine alias contract changed")
    return {
        "schema": "serenity.minimax_h3.torch_cpu_randn_f32_avx2_v1.fixture.v1",
        "profile": PROFILE,
        "oracle": {
            "torch_version": TORCH_VERSION,
            "torch_git_version": TORCH_GIT_VERSION,
            "torch_build_config_sha256": TORCH_BUILD_CONFIG_SHA256,
            "installed_header_sha256": HEADER_SHA256,
        },
        "platform_contract": platform_contract,
        "algorithm_contract": {
            "engine": "at::mt19937_engine; initialization aliases seed modulo 2^32",
            "uniform": "(random_u32 & 0x00ffffff) * 2^-24 in F32",
            "normal": "normal_fill_AVX2 + log256_ps + sincos256_ps",
            "rounding": "installed GCC/FMA AVX2 statement contraction profile",
            "tail": "N uniforms plus 16 replacement uniforms iff N mod 16 != 0",
            "minimum_numel": 16,
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    if (args.write is None) == (args.check is None):
        parser.error("choose exactly one of --write or --check")
    torch = _load_torch()
    rendered = _canonical(_payload(torch))
    if args.check is not None:
        current = args.check.read_bytes()
        if current != rendered:
            raise RuntimeError("fixture bytes differ from deterministic regeneration")
    else:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_bytes(rendered)
    print(_sha256(rendered))


if __name__ == "__main__":
    main()
