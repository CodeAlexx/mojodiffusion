#!/usr/bin/env python3
"""Creator-stack oracle for Ideogram-4 product geometry.

Uses torchref's production Ideogram pipeline and MRoPE implementation. The
compact binary is consumed by ideogram4_product_geometry_gate.mojo.
"""

from __future__ import annotations

import math
import importlib.util
import struct
import sys
import types
from pathlib import Path

import torch


ROOT = Path("/home/alex/torchref-image/extensions_built_in/diffusion_models/ideogram4/src")
PIPELINE_SRC = ROOT / "pipeline.py"
TRANSFORMER_SRC = ROOT / "transformer.py"
REPO = Path(__file__).resolve().parents[4]
BACKEND = REPO / "serenitymojo/serve/ideogram4_backend.mojo"
OUT = Path("/tmp/serenity_ideogram4_product_geometry_ref.bin")
SHAPES = (
    (1024, 1024),
    (1152, 896),
    (896, 1152),
    (1344, 768),
    (768, 1344),
    (1280, 832),
    (832, 1280),
)
RECORD_FLOATS = 537


def require_contracts() -> None:
    pipeline = PIPELINE_SRC.read_text(encoding="utf-8")
    transformer = TRANSFORMER_SRC.read_text(encoding="utf-8")
    backend = BACKEND.read_text(encoding="utf-8")
    creator_needles = (
        "mean = mu + 0.5 * math.log((width * height) / (512 * 512))",
        "gh = height // (ae_scale * patch)",
        "gw = width // (ae_scale * patch)",
        "h_idx = torch.arange(gh, device=device).view(-1, 1).expand(gh, gw).reshape(-1)",
        "w_idx = torch.arange(gw, device=device).view(1, -1).expand(gh, gw).reshape(-1)",
        "image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET",
        "freqs_t[..., idx] = freqs[axis][..., idx]",
    )
    missing = [needle for needle in creator_needles if needle not in pipeline + transformer]
    if missing:
        raise RuntimeError(f"local Ideogram-4 creator contract drift: {missing}")

    for width, height in SHAPES:
        if f"params.width == {width} and params.height == {height}" not in backend:
            raise RuntimeError(f"Ideogram-4 backend admission missing {width}x{height}")
    required_specializations = (
        ("GH_1152X896", "GW_1152X896"),
        ("GH_896X1152", "GW_896X1152"),
        ("GH_1344X768", "GW_1344X768"),
        ("GH_768X1344", "GW_768X1344"),
        ("GH_1280X832", "GW_1280X832"),
        ("GH_832X1280", "GW_832X1280"),
    )
    for gh_name, gw_name in required_specializations:
        for helper in ("_ensure_static_b", "_prepare_job_b", "_denoise_one_b", "_decode_and_save_b"):
            if f"{helper}[{gh_name}, {gw_name}]" not in backend:
                raise RuntimeError(f"Ideogram-4 backend missing {helper} arm for {gh_name}/{gw_name}")


def record(width: int, height: int, rope_cls, schedule_fn) -> list[float]:
    gh, gw = height // 16, width // 16
    nimg = gh * gw
    last_pos = torch.tensor([[[65536, 65536 + gh - 1, 65536 + gw - 1]]])
    rope = rope_cls(head_dim=256, base=5_000_000, mrope_section=(24, 20, 20))
    cos, sin = rope(last_pos)
    sigmas = schedule_fn(4, width, height).tolist()
    vae_h, vae_w = 2 * gh, 2 * gw
    tile_h, tile_w = vae_h // 2, vae_w // 2
    return [
        width, height, gh, gw, nimg, 1024 + nimg, vae_h, vae_w,
        65536, 65536 + gh - 1, 65536 + gw - 1,
        0.5 * math.log((width * height) / (512 * 512)),
        *sigmas, *cos.flatten().tolist(), *sin.flatten().tolist(),
        tile_h, 0, tile_h // 2, tile_h,
        tile_w, 0, tile_w // 2, tile_w,
    ]


def main() -> None:
    require_contracts()
    package = types.ModuleType("ideogram4_creator_src")
    package.__path__ = [str(ROOT)]
    sys.modules[package.__name__] = package
    for module_name, source in (
        ("transformer", TRANSFORMER_SRC),
        ("pipeline", PIPELINE_SRC),
    ):
        qualified = f"{package.__name__}.{module_name}"
        spec = importlib.util.spec_from_file_location(qualified, source)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load creator module {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
    transformer = sys.modules[f"{package.__name__}.transformer"]
    pipeline = sys.modules[f"{package.__name__}.pipeline"]

    records = [
        record(w, h, transformer.Ideogram4MRoPE, pipeline.get_ideogram4_sigmas)
        for w, h in SHAPES
    ]
    assert all(len(row) == RECORD_FLOATS for row in records)
    values = [value for row in records for value in row]
    OUT.write_bytes(struct.pack(f"<{len(values)}f", *values))
    print(f"[oracle] torchref Ideogram-4 records={len(records)} floats={len(values)}")
    for row in records:
        print(
            f"  {int(row[0])}x{int(row[1])}: grid={int(row[2])}x{int(row[3])} "
            f"N_IMG={int(row[4])} VAE={int(row[6])}x{int(row[7])}"
        )
    print(f"[oracle] wrote {OUT}")


if __name__ == "__main__":
    main()
