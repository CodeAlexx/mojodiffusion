#!/usr/bin/env python3
"""Static guard for Qwen/Wan 1024 VAE tiled decode wiring.

This does not prove image parity. It prevents the production 1024 Qwen-Image
and Anima paths from drifting back to monolithic QwenImageVaeDecoder[128,128]
decode, which is the high-VRAM failure mode this wrapper avoids.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TILED = ROOT / "serenitymojo/models/vae/qwenimage_tiled_decode.mojo"

QWEN_SITES = [
    ROOT / "serenitymojo/serve/qwenimage_backend.mojo",
    ROOT / "serenitymojo/pipeline/qwenimage_sample_cli.mojo",
    ROOT / "serenitymojo/pipeline/qwenimage_pipeline_1024_multistep.mojo",
]

WAN21_SITES = [
    ROOT / "serenitymojo/pipeline/anima_decode_cli.mojo",
    ROOT / "serenitymojo/pipeline/anima_pipeline_1024_multistep.mojo",
    ROOT / "serenitymojo/pipeline/anima_vae_latent_smoke.mojo",
]


def read(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"[qwen-wan-vae-tiled] missing file: {path}")
    return path.read_text(encoding="utf-8")


def require(path: Path, label: str, needles: list[str]) -> None:
    text = read(path)
    missing = [needle for needle in needles if needle not in text]
    if missing:
        print(f"[qwen-wan-vae-tiled] FAIL {label}: {path}")
        for needle in missing:
            print(f"  missing: {needle}")
        raise SystemExit(1)
    print(f"[qwen-wan-vae-tiled] PASS {label}")


def forbid(path: Path, label: str, needles: list[str]) -> None:
    text = read(path)
    found = [needle for needle in needles if needle in text]
    if found:
        print(f"[qwen-wan-vae-tiled] FAIL {label}: {path}")
        for needle in found:
            print(f"  forbidden: {needle}")
        raise SystemExit(1)
    print(f"[qwen-wan-vae-tiled] PASS {label}")


def main() -> int:
    require(
        TILED,
        "shared tiled decode wrapper",
        [
            "def qwenimage_tiled_decode_with_decoder",
            "def qwenimage_tiled_decode[",
            "def wan21_image_tiled_decode_with_decoder",
            "def wan21_image_tiled_decode[",
            "comptime assert TILE_H == LATENT_H // 2",
            "comptime assert TILE_W == LATENT_W // 2",
            "var half = TILE_H // 2",
            "slice(latent, 2, 0, TILE_H, ctx)",
            "slice(r, 3, half, TILE_W, ctx)",
            "QwenImageVaeDecoder[TILE_H, TILE_W].load(",
            "QwenImageVaeDecoder[TILE_H, TILE_W].load_wan21_keys(",
            "dec.decode(",
            "dec.decode_wan21_keys(",
            "STDtype.F32",
            "return _blend3(row0, row1, row2, 2, ctx)",
        ],
    )

    for site in QWEN_SITES:
        require(
            site,
            f"Qwen 1024 site uses tiled decode: {site.name}",
            [
                "from serenitymojo.models.vae.qwenimage_tiled_decode import qwenimage_tiled_decode",
                "qwenimage_tiled_decode[LH, LW]",
            ],
        )
        forbid(
            site,
            f"Qwen 1024 site avoids monolithic decoder: {site.name}",
            [
                "from serenitymojo.models.vae.qwenimage_decoder import QwenImageVaeDecoder",
                "QwenImageVaeDecoder[LH, LW].load(",
                "var vae = QwenImageVaeDecoder",
                "vae.decode(latent",
            ],
        )

    for site in WAN21_SITES:
        require(
            site,
            f"Wan21/Anima 1024 site uses tiled decode: {site.name}",
            [
                "from serenitymojo.models.vae.qwenimage_tiled_decode import wan21_image_tiled_decode",
                "wan21_image_tiled_decode[",
            ],
        )
        forbid(
            site,
            f"Wan21/Anima 1024 site avoids monolithic decoder: {site.name}",
            [
                "from serenitymojo.models.vae.qwenimage_decoder import QwenImageVaeDecoder",
                "QwenImageVaeDecoder[LH, LW].load_wan21_keys(",
                "QwenImageVaeDecoder[ANIMA_LATENT_H, ANIMA_LATENT_W].load_wan21_keys(",
                "decode_wan21_keys(lat",
                "decode_wan21_keys(vae_input",
            ],
        )

    print("[qwen-wan-vae-tiled] PASS all checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
