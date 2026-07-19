#!/usr/bin/env python3
"""No-CUDA gate for FLUX product shapes and compiled Chroma geometry arms."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FLUX = ROOT / "serenitymojo/serve/flux_backend.mojo"
CHROMA = ROOT / "serenitymojo/serve/chroma_backend.mojo"
CHROMA_DECODE = ROOT / "serenitymojo/serve/chroma_decode_subprocess.mojo"

SHAPES = (
    "1024x1024",
    "1152x896",
    "896x1152",
    "1344x768",
    "768x1344",
    "1280x832",
    "832x1280",
)


def require(source: str, label: str, needles: tuple[str, ...]) -> None:
    missing = [needle for needle in needles if needle not in source]
    if missing:
        raise SystemExit(f"FAIL {label} rectangular product contract missing: {missing}")


def main() -> None:
    flux = FLUX.read_text(encoding="utf-8")
    chroma = CHROMA.read_text(encoding="utf-8")
    chroma_decode = CHROMA_DECODE.read_text(encoding="utf-8")

    require(
        flux,
        "Flux",
        (
            "def _flux_shape_supported(width: Int, height: Int) -> Bool:",
            "def _load_model_shape[",
            "def _prepare_job_shape[",
            "def _denoise_one_shape[N_IMG_: Int]",
            "def _decode_and_save_shape[LATENT_H_: Int, LATENT_W_: Int]",
            "_pack_latent_shape[LATENT_H_, LATENT_W_]",
            "_unpack_flux_packed_latent_shape[LATENT_H_, LATENT_W_]",
            "_flux_forward_turbo[N_IMG_, N_TXT, S_]",
            "load_flux1_ldm_decoder[LATENT_H_, LATENT_W_]",
            "flux_tiled_decode[LATENT_H_, LATENT_W_]",
            "self.loaded_width != self.params.width",
            '"resolution":{"width":',
        ),
    )
    require(
        chroma,
        "Chroma",
        (
            "def _chroma_shape_supported(width: Int, height: Int) -> Bool:",
            "def _load_model_shape[LH_: Int, LW_: Int, N_IMG_: Int]",
            "def _prepare_job_shape[LH_: Int, LW_: Int, N_IMG_: Int]",
            "def _denoise_one_shape[N_IMG_: Int]",
            "def _decode_and_save_shape[LH_: Int, LW_: Int]",
            "_pack_latent_shape[LH_, LW_]",
            "_unpack_latent_shape[LH_, LW_]",
            "_chroma_forward_turbo[N_IMG_]",
            "decode_whole_subprocess(packed, LH_, LW_, self.ctx)",
            "flux_tiled_decode[LH_, LW_]",
            '"resolution":{"width":',
        ),
    )
    require(
        chroma_decode,
        "Chroma child decode",
        (
            "comptime for bi in range(DEFAULT_ASPECT_LADDER_LEN):",
            "_decode_child_shape[LH_BI, LW_BI]",
        ),
    )

    for shape in SHAPES:
        if shape not in flux:
            raise SystemExit(f"FAIL Flux compiled size {shape} is absent")
    if "return width == 1024 and height == 1024" not in chroma:
        raise SystemExit("FAIL Chroma must remain 1024-only until rectangular runtime clears")
    if "measured ~24x denoise slowdown" not in chroma:
        raise SystemExit("FAIL Chroma rectangular runtime blocker is undocumented")
    if "1024x1024 only: the FLUX" in flux:
        raise SystemExit("FAIL Flux retains the stale 1024-only admission gate")

    print("PASS Flux product dispatch and gated Chroma rectangular core contract")


if __name__ == "__main__":
    main()
