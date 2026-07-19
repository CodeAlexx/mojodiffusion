# Anima metadata contract smoke.
#
# Header-only: no DeviceContext, no Tensor allocations, no model math, and no
# decode. This keeps the local Anima sidecar facts compile-checked while the
# full Mojo model body remains unported.

from serenitymojo.models.dit.anima_contract import (
    ANIMA_ADAPTER_BLOCKS,
    ANIMA_ADAPTER_DIM,
    ANIMA_CFG_SCALE_X10,
    ANIMA_DEPTH,
    ANIMA_HEAD_DIM,
    ANIMA_HIDDEN,
    ANIMA_IMAGE_TOKENS,
    ANIMA_LATENT_H,
    ANIMA_LATENT_W,
    ANIMA_NUM_HEADS,
    ANIMA_NUM_STEPS,
    ANIMA_PATCH_GRID_H,
    ANIMA_PATCH_GRID_W,
    ANIMA_PATCH_IN_DIM,
    ANIMA_PATCH_OUT_DIM,
    ANIMA_QWEN3_PAD_ID,
    ANIMA_T5_PAD_ID,
    anima_default_conditioning_path,
    anima_default_rust_latent_path,
    validate_anima_conditioning_header,
    validate_anima_local_paths,
    validate_anima_metadata_contract,
    validate_anima_rust_latent_header,
)
from serenitymojo.registry.checkpoints import path_exists


def main() raises:
    var checked = validate_anima_local_paths()
    print("[anima-contract] local paths checked:", checked)
    var plan = validate_anima_metadata_contract()
    print(
        "[anima-contract] shape constants hidden/depth/heads=",
        ANIMA_HIDDEN,
        ANIMA_DEPTH,
        ANIMA_NUM_HEADS,
        "head_dim=",
        ANIMA_HEAD_DIM,
    )
    print(
        "[anima-contract] adapter dim/blocks=",
        ANIMA_ADAPTER_DIM,
        ANIMA_ADAPTER_BLOCKS,
        "pads=",
        ANIMA_QWEN3_PAD_ID,
        ANIMA_T5_PAD_ID,
    )
    print(
        "[anima-contract] steps/cfg_x10=",
        ANIMA_NUM_STEPS,
        ANIMA_CFG_SCALE_X10,
    )
    print(
        "[anima-contract] token geometry latent/patch/tokens=",
        ANIMA_LATENT_H,
        "x",
        ANIMA_LATENT_W,
        ANIMA_PATCH_GRID_H,
        "x",
        ANIMA_PATCH_GRID_W,
        ANIMA_IMAGE_TOKENS,
    )
    if plan.patch_in_dim != ANIMA_PATCH_IN_DIM:
        raise Error("Anima patch input dim mismatch")
    if plan.patch_out_dim != ANIMA_PATCH_OUT_DIM:
        raise Error("Anima patch output dim mismatch")

    var embeddings_path = anima_default_conditioning_path()
    if path_exists(embeddings_path):
        var cond = validate_anima_conditioning_header(embeddings_path)
        print(
            "[anima-contract] conditioning context tokens/hidden=",
            cond.text_tokens,
            cond.hidden,
        )
    else:
        print("[anima-contract] conditioning sidecar missing:", embeddings_path)
        print(
            "[anima-contract] expected keys: context_cond/context_uncond [1, 256, 1024]"
        )

    var latent_path = anima_default_rust_latent_path()
    if path_exists(latent_path):
        var latent = validate_anima_rust_latent_header(latent_path)
        print(
            "[anima-contract] rust latent shape=",
            latent.batch,
            latent.channels,
            latent.frames,
            latent.latent_h,
            latent.latent_w,
        )
    else:
        print("[anima-contract] rust latent oracle missing:", latent_path)
    print("Anima metadata contract PASS")
