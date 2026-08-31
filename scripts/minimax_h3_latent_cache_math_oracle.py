#!/usr/bin/env python3
"""Generate the pinned MiniMax-H3 one-frame latent-cache math fixture.

Python/Torch are development-oracle dependencies only.  The canonical payload
contains pinned source identity and numerical tensors, never runtime package or
machine metadata.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import torch


ORACLE_COMMIT = "b8717864713c9e4e7ef3d56eba1fc695a9b626a5"
VIDEO_VAE_SOURCE = "src/musubi_tuner/minimax_h3/video_vae.py"
CACHE_LATENTS_SOURCE = "src/musubi_tuner/minimax_h3_cache_latents.py"
VIDEO_VAE_SHA256 = "96e6698e5072adc258b6610881749d3748173d78c01c9b833e4cc42253165671"
CACHE_LATENTS_SHA256 = "a27d4541add4b256719de530a2daa5a3746d99a32ba168f5579a7e6cb69cb69b"
SCHEMA = "serenity.minimax_h3.latent_cache_math_oracle.v1"


def _git_source(repo: Path, source: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{ORACLE_COMMIT}:{source}"]
    )


def _require_digest(source: bytes, expected: str, label: str) -> None:
    actual = hashlib.sha256(source).hexdigest()
    if actual != expected:
        raise RuntimeError(f"pinned {label} digest mismatch: {actual}")


def _extract_function(source: bytes, filename: str, name: str, namespace: dict):
    tree = ast.parse(source.decode("utf-8"), filename=filename)
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, filename, "exec"), namespace)
    return namespace[name], [node.lineno, node.end_lineno]


def _extract_assignments(
    source: bytes, filename: str, names: tuple[str, ...]
) -> tuple[dict[str, object], dict[str, list[int]]]:
    tree = ast.parse(source.decode("utf-8"), filename=filename)
    nodes = []
    spans: dict[str, list[int]] = {}
    for item in tree.body:
        if not isinstance(item, (ast.Assign, ast.AnnAssign)):
            continue
        targets = item.targets if isinstance(item, ast.Assign) else [item.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id in names:
                nodes.append(item)
                spans[target.id] = [item.lineno, item.end_lineno]
    missing = sorted(set(names) - set(spans))
    if missing:
        raise RuntimeError(f"missing pinned assignments: {missing}")
    namespace: dict[str, object] = {}
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, filename, "exec"), namespace)
    return {name: namespace[name] for name in names}, spans


def _seed(cache_seed: int, canonical_item_key: str) -> int:
    digest = hashlib.sha256(f"{cache_seed}\0{canonical_item_key}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63)


def _payload(video_source: bytes, cache_source: bytes) -> dict[str, object]:
    _require_digest(video_source, VIDEO_VAE_SHA256, VIDEO_VAE_SOURCE)
    _require_digest(cache_source, CACHE_LATENTS_SHA256, CACHE_LATENTS_SOURCE)

    prepare_ns = {"torch": torch, "np": np}
    prepare_pixels, prepare_span = _extract_function(
        cache_source, CACHE_LATENTS_SOURCE, "_prepare_pixels", prepare_ns
    )
    posterior_ns = {"torch": torch}
    posterior_sample, posterior_span = _extract_function(
        video_source, VIDEO_VAE_SOURCE, "_video_posterior_sample", posterior_ns
    )
    constants, constant_spans = _extract_assignments(
        video_source,
        VIDEO_VAE_SOURCE,
        ("IMAGENET_MEAN", "IMAGENET_STD", "LATENTS_MEAN", "LATENTS_STD"),
    )

    height, width = 2, 3
    rgb = np.asarray(
        [
            [[0, 64, 128], [255, 32, 16], [7, 200, 99]],
            [[250, 5, 180], [31, 127, 223], [91, 17, 239]],
        ],
        dtype=np.uint8,
    )
    prepared_ncfhw = prepare_pixels(rgb[None, ...])
    if tuple(prepared_ncfhw.shape) != (1, 3, 1, height, width):
        raise RuntimeError(f"unexpected pinned _prepare_pixels shape: {prepared_ncfhw.shape}")
    prepared_ndhwc = prepared_ncfhw.permute(0, 2, 3, 4, 1).contiguous()

    pixel_mean = torch.tensor(constants["IMAGENET_MEAN"], dtype=torch.float32).view(
        1, 3, 1, 1, 1
    )
    pixel_std = torch.tensor(constants["IMAGENET_STD"], dtype=torch.float32).view(
        1, 3, 1, 1, 1
    )
    encoder_ncfhw = (prepared_ncfhw + 1.0) * 0.5
    encoder_ncfhw = (encoder_ncfhw - pixel_mean) / pixel_std
    encoder_ndhwc = encoder_ncfhw.permute(0, 2, 3, 4, 1).contiguous()

    # Non-degenerate [B,48,T,H,W] moments.  Logvar deliberately crosses both
    # clamp boundaries and includes ordinary interior values.
    mean = torch.empty((1, 24, 1, height, width), dtype=torch.float32)
    logvar = torch.empty_like(mean)
    noise = torch.empty_like(mean)
    logvar_pattern = (-35.0, -30.0, -4.0, -0.75, 4.0, 25.0)
    for channel in range(24):
        for y in range(height):
            for x in range(width):
                linear = (channel * height + y) * width + x
                mean[0, channel, 0, y, x] = (
                    ((linear * 37 + channel * 11) % 211) - 105
                ) / 29.0
                logvar[0, channel, 0, y, x] = logvar_pattern[linear % 6]
                noise[0, channel, 0, y, x] = (
                    ((linear * 19 + channel * 7) % 97) - 48
                ) / 31.0
    moments = torch.cat((mean, logvar), dim=1)

    if not torch.cuda.is_available():
        raise RuntimeError("the pinned posterior fixture requires a CUDA Torch oracle")
    device = torch.device("cuda")

    class FakeVAE:
        def __init__(self) -> None:
            self.latents_mean = torch.tensor(
                constants["LATENTS_MEAN"], dtype=torch.float32, device=device
            )
            self.latents_std = torch.tensor(
                constants["LATENTS_STD"], dtype=torch.float32, device=device
            )

        def encode_moments(self, _pixels: torch.Tensor) -> torch.Tensor:
            return moments.to(device)

    original_randn = torch.randn

    def injected_randn(shape, *, generator, dtype, device):
        del generator
        if tuple(shape) != tuple(noise.shape) or dtype != torch.float32 or device != "cpu":
            raise RuntimeError("pinned posterior requested an unexpected noise contract")
        return noise.clone()

    torch.randn = injected_randn
    try:
        sampled = posterior_sample(
            FakeVAE(),
            torch.zeros((1, 3, 1, height, width), dtype=torch.float32, device=device),
            torch.Generator(device="cpu").manual_seed(0),
            False,
        )
    finally:
        torch.randn = original_randn

    expected_cache = sampled[0].contiguous().cpu()
    moments_ndhwc = moments.permute(0, 2, 3, 4, 1).contiguous()
    mean_ndhwc = mean.permute(0, 2, 3, 4, 1).contiguous()
    logvar_ndhwc = logvar.permute(0, 2, 3, 4, 1).contiguous()
    noise_ndhwc = noise.permute(0, 2, 3, 4, 1).contiguous()

    seed_cases = [
        {"cache_seed": 0, "canonical_item_key": "", "derived_seed": _seed(0, "")},
        {
            "cache_seed": 123,
            "canonical_item_key": "eri_with_trigger/000001.png",
            "derived_seed": _seed(123, "eri_with_trigger/000001.png"),
        },
        {
            "cache_seed": 9223372036854775807,
            "canonical_item_key": "relative/path with spaces.webp",
            "derived_seed": _seed(9223372036854775807, "relative/path with spaces.webp"),
        },
    ]

    return {
        "schema": SCHEMA,
        "oracle_commit": ORACLE_COMMIT,
        "source_contracts": {
            CACHE_LATENTS_SOURCE: {
                "sha256": CACHE_LATENTS_SHA256,
                "qualnames": {"_prepare_pixels": prepare_span},
            },
            VIDEO_VAE_SOURCE: {
                "sha256": VIDEO_VAE_SHA256,
                "qualnames": {
                    "_video_posterior_sample": posterior_span,
                    **constant_spans,
                },
            },
        },
        "pixel_case": {
            "height": height,
            "width": width,
            "rgb8_hwc": rgb.reshape(-1).tolist(),
            "prepared_ndhwc_f32": prepared_ndhwc.reshape(-1).tolist(),
            "encoder_input_ndhwc_f32": encoder_ndhwc.reshape(-1).tolist(),
        },
        "posterior_case": {
            "moments_shape_ndhwc": list(moments_ndhwc.shape),
            "noise_shape_ndhwc": list(noise_ndhwc.shape),
            "cache_shape_cthw": list(expected_cache.shape),
            "moments_ndhwc_f32": moments_ndhwc.reshape(-1).tolist(),
            "mean_ndhwc_f32": mean_ndhwc.reshape(-1).tolist(),
            "logvar_ndhwc_f32": logvar_ndhwc.reshape(-1).tolist(),
            "noise_ndhwc_f32": noise_ndhwc.reshape(-1).tolist(),
            "expected_cache_cthw_f32": expected_cache.reshape(-1).tolist(),
        },
        "seed_cases": seed_cases,
        "execution_receipt": {
            "prepare_pixels": "pinned source AST executed on CPU",
            "posterior": "pinned source AST executed on CUDA with injected F32 noise",
            "torch_cpu_rng": "not generated or claimed",
        },
    }


def _render(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--musubi-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    if (args.output is None) == (args.check is None):
        parser.error("exactly one of --output or --check is required")

    video_source = _git_source(args.musubi_repo, VIDEO_VAE_SOURCE)
    cache_source = _git_source(args.musubi_repo, CACHE_LATENTS_SOURCE)
    rendered = _render(_payload(video_source, cache_source))
    if args.check is not None:
        if args.check.read_bytes() != rendered:
            raise RuntimeError(f"fixture differs from deterministic regeneration: {args.check}")
        print(hashlib.sha256(rendered).hexdigest())
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(rendered)
    print(hashlib.sha256(rendered).hexdigest())


if __name__ == "__main__":
    main()
