# SD3.5 Large pipeline contract smoke.
#
# Metadata/header-only: no DeviceContext, no MMDiT/VAE imports, and no tensor
# loads. It keeps the first SD3 entry contract compile-checked while the real
# MMDiT, triple-encoder, and embedded-VAE path is still unported.

from serenitymojo.models.dit.sd3_contract import (
    SD3_LARGE_DEPTH,
    SD3_LARGE_HEAD_DIM,
    SD3_LARGE_HIDDEN,
    SD3_LARGE_IMAGE_TOKENS,
    SD3_LARGE_LATENT_ELEMENTS,
    SD3_LARGE_NUM_HEADS,
    SD3_LARGE_NUM_STEPS,
    SD3_LARGE_PATCH_GRID_H,
    SD3_LARGE_PATCH_GRID_W,
    SD3_LARGE_PATCH_VECTOR_DIM,
    SD3_LARGE_TOTAL_SEQUENCE,
    build_sd3_large_token_plan,
    sd3_large_cfg_scale,
    sd3_large_schedule_shift,
    sd3_large_vae_scale,
    sd3_large_vae_shift,
    validate_sd3_large_checkpoint_header,
    validate_sd3_large_manifest_contract,
)
from serenitymojo.registry.checkpoints import (
    default_manifest_by_id,
    validate_manifest_paths,
)


def main() raises:
    var manifest = default_manifest_by_id(String("sd3_5_large"))
    validate_sd3_large_manifest_contract(manifest)
    var status = validate_manifest_paths(manifest)
    print(
        "[sd3-contract] manifest paths checked/missing:",
        status.checked,
        status.missing,
    )
    var plan = build_sd3_large_token_plan(
        manifest.default_width, manifest.default_height, manifest.text_tokens
    )
    plan.validate_large_1024_contract()
    if plan.image_tokens != SD3_LARGE_IMAGE_TOKENS:
        raise Error("SD3.5 Large image token contract mismatch")
    if plan.patch_vector_dim != SD3_LARGE_PATCH_VECTOR_DIM:
        raise Error("SD3.5 Large patch vector contract mismatch")
    if plan.latent_elements != SD3_LARGE_LATENT_ELEMENTS:
        raise Error("SD3.5 Large latent element contract mismatch")
    if plan.total_sequence != SD3_LARGE_TOTAL_SEQUENCE:
        raise Error("SD3.5 Large sequence contract mismatch")
    validate_sd3_large_checkpoint_header(manifest)
    print(
        "[sd3-contract] shape constants hidden/depth/heads=",
        SD3_LARGE_HIDDEN,
        SD3_LARGE_DEPTH,
        SD3_LARGE_NUM_HEADS,
        "head_dim=",
        SD3_LARGE_HEAD_DIM,
    )
    print(
        "[sd3-contract] steps/cfg/shift/vae=",
        SD3_LARGE_NUM_STEPS,
        sd3_large_cfg_scale(),
        sd3_large_schedule_shift(),
        sd3_large_vae_scale(),
        sd3_large_vae_shift(),
    )
    print(
        "[sd3-contract] token geometry latent/patch/tokens=",
        plan.latent_h,
        "x",
        plan.latent_w,
        SD3_LARGE_PATCH_GRID_H,
        "x",
        SD3_LARGE_PATCH_GRID_W,
        plan.image_tokens,
    )
    print("SD3.5 Large pipeline contract PASS")
