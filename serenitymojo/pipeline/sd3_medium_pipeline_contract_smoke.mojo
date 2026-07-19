# SD3.5 Medium pipeline contract smoke.
#
# Metadata/header-only: no DeviceContext, no MMDiT/VAE imports, and no tensor
# loads. This pins the local "small" SD3 lane to the Medium checkpoint used by
# the Rust reference.

from serenitymojo.models.dit.sd3_contract import (
    SD3_MEDIUM_DEPTH,
    SD3_MEDIUM_DUAL_ATTENTION_BLOCKS,
    SD3_MEDIUM_HEAD_DIM,
    SD3_MEDIUM_HIDDEN,
    SD3_MEDIUM_IMAGE_TOKENS,
    SD3_MEDIUM_LATENT_ELEMENTS,
    SD3_MEDIUM_NUM_HEADS,
    SD3_MEDIUM_NUM_STEPS,
    SD3_MEDIUM_PATCH_GRID_H,
    SD3_MEDIUM_PATCH_GRID_W,
    SD3_MEDIUM_PATCH_VECTOR_DIM,
    SD3_MEDIUM_TOTAL_SEQUENCE,
    build_sd3_medium_token_plan,
    sd3_medium_cfg_scale,
    sd3_medium_schedule_shift,
    sd3_medium_vae_scale,
    sd3_medium_vae_shift,
    validate_sd3_medium_checkpoint_header,
    validate_sd3_medium_manifest_contract,
)
from serenitymojo.registry.checkpoints import (
    default_manifest_by_id,
    validate_manifest_paths,
)


def main() raises:
    var manifest = default_manifest_by_id(String("sd3_5_medium"))
    validate_sd3_medium_manifest_contract(manifest)
    var status = validate_manifest_paths(manifest)
    print(
        "[sd3-medium-contract] manifest paths checked/missing:",
        status.checked,
        status.missing,
    )
    var plan = build_sd3_medium_token_plan(
        manifest.default_width, manifest.default_height, manifest.text_tokens
    )
    plan.validate_medium_1024_contract()
    if plan.image_tokens != SD3_MEDIUM_IMAGE_TOKENS:
        raise Error("SD3.5 Medium image token contract mismatch")
    if plan.patch_vector_dim != SD3_MEDIUM_PATCH_VECTOR_DIM:
        raise Error("SD3.5 Medium patch vector contract mismatch")
    if plan.latent_elements != SD3_MEDIUM_LATENT_ELEMENTS:
        raise Error("SD3.5 Medium latent element contract mismatch")
    if plan.total_sequence != SD3_MEDIUM_TOTAL_SEQUENCE:
        raise Error("SD3.5 Medium sequence contract mismatch")
    validate_sd3_medium_checkpoint_header(manifest)
    print(
        "[sd3-medium-contract] shape constants hidden/depth/heads=",
        SD3_MEDIUM_HIDDEN,
        SD3_MEDIUM_DEPTH,
        SD3_MEDIUM_NUM_HEADS,
        "head_dim=",
        SD3_MEDIUM_HEAD_DIM,
    )
    print(
        "[sd3-medium-contract] dual attention blocks/steps/cfg/shift/vae=",
        SD3_MEDIUM_DUAL_ATTENTION_BLOCKS,
        SD3_MEDIUM_NUM_STEPS,
        sd3_medium_cfg_scale(),
        sd3_medium_schedule_shift(),
        sd3_medium_vae_scale(),
        sd3_medium_vae_shift(),
    )
    print(
        "[sd3-medium-contract] token geometry latent/patch/tokens=",
        plan.latent_h,
        "x",
        plan.latent_w,
        SD3_MEDIUM_PATCH_GRID_H,
        "x",
        SD3_MEDIUM_PATCH_GRID_W,
        plan.image_tokens,
    )
    print("SD3.5 Medium pipeline contract PASS")
