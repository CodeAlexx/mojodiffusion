#!/usr/bin/env python3
"""No-CUDA SDXL contract: seven compiled cores, five runtime-admitted shapes."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "serenitymojo/serve/sdxl_backend.mojo"
TILED_VAE = ROOT / "serenitymojo/models/vae/sdxl_tiled_decode.mojo"
ADM_ORACLE = ROOT / "serenitymojo/models/sdxl/parity/conditioning_oracle.py"
CAPABILITIES = ROOT / "serenity-server/crates/server/src/capabilities.rs"

SHAPES = (
    (1024, 1024, "LH_SQUARE", "LW_SQUARE", "model_square", 128, 128),
    (1152, 896, "LH_1152X896", "LW_1152X896", "model_1152x896", 112, 144),
    (896, 1152, "LH_896X1152", "LW_896X1152", "model_896x1152", 144, 112),
    (1344, 768, "LH_LANDSCAPE", "LW_LANDSCAPE", "model_landscape", 96, 168),
    (768, 1344, "LH_PORTRAIT", "LW_PORTRAIT", "model_portrait", 168, 96),
    (1280, 832, "LH_1280X832", "LW_1280X832", "model_1280x832", 104, 160),
    (832, 1280, "LH_832X1280", "LW_832X1280", "model_832x1280", 160, 104),
)


def require(source: str, needle: str, failures: list[str]) -> None:
    if needle not in source:
        failures.append(needle)


def main() -> None:
    backend = BACKEND.read_text(encoding="utf-8")
    tiled = TILED_VAE.read_text(encoding="utf-8")
    oracle = ADM_ORACLE.read_text(encoding="utf-8")
    capabilities = CAPABILITIES.read_text(encoding="utf-8")
    failures: list[str] = []

    for width, height, lh_name, lw_name, model, lh, lw in SHAPES:
        require(backend, f"(params.width == {width} and params.height == {height})", failures)
        require(backend, f"comptime {lh_name} = {lh}", failures)
        require(backend, f"comptime {lw_name} = {lw}", failures)
        require(backend, f"var {model}: List[ArcPointer[SDXLUNet[{lh_name}, {lw_name}]]]", failures)
        require(backend, f"SDXLUNet[{lh_name}, {lw_name}].load", failures)
        require(backend, f"self.{model}[0][].forward", failures)
        require(backend, f"self._decode_shape[{lh_name}, {lw_name}]", failures)
        key = "square" if width == height else (
            "landscape" if (width, height) == (1344, 768) else
            "portrait" if (width, height) == (768, 1344) else
            f"{width}x{height}"
        )
        require(oracle, f'"expected_{key}": adm(pooled, {height}, {width})', failures)

    # The existing generic VAE path must keep row/column crop strides separate.
    for needle in (
        "var half_h = TILE_H // 2",
        "var half_w = TILE_W // 2",
        "slice(r, 3, half_w, TILE_W, ctx)",
        "slice(latent, 2, half_h, TILE_H, ctx)",
    ):
        require(tiled, needle, failures)

    admitted = """const SDXL_SIZES: &[(i64, i64)] = &[
    (1024, 1024),
    (1152, 896),
    (896, 1152),
    (1344, 768),
    (768, 1344),
];"""
    require(capabilities, admitted, failures)

    if failures:
        raise SystemExit("FAIL SDXL aspect ladder contract missing:\n  " + "\n  ".join(failures))
    print("PASS SDXL seven compiled/oracle-gated cores + five runtime-admitted shapes")
    for width, height, _, _, _, lh, lw in SHAPES:
        print(f"  {width}x{height}: latent={lh}x{lw} pixels={width * height}")


if __name__ == "__main__":
    main()
