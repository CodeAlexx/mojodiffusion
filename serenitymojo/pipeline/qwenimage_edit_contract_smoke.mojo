# Qwen-Image-Edit metadata contract smoke.
#
# Header-only: no DeviceContext, no tensor H2D load, no Qwen2.5-VL/DiT math,
# and no VAE encode/decode.

from serenitymojo.models.dit.qwenimage_edit_contract import (
    QWENIMAGE_EDIT_DEFAULT_CFG_X10,
    QWENIMAGE_EDIT_DEFAULT_STEPS,
    QWENIMAGE_EDIT_DIT_HEAD_DIM,
    QWENIMAGE_EDIT_DIT_HEADS,
    QWENIMAGE_EDIT_DIT_HIDDEN,
    QWENIMAGE_EDIT_DIT_LAYERS,
    QWENIMAGE_EDIT_DROP_IDX,
    QWENIMAGE_EDIT_IMAGE_TOKENS,
    QWENIMAGE_EDIT_LATENT_CHANNELS,
    QWENIMAGE_EDIT_LATENT_H,
    QWENIMAGE_EDIT_LATENT_W,
    QWENIMAGE_EDIT_PAD_ID,
    QWENIMAGE_EDIT_REFERENCE_TOKENS,
    QWENIMAGE_EDIT_TARGET_TOKENS,
    QWENIMAGE_EDIT_TEXT_ENCODER_TENSORS,
    QWENIMAGE_EDIT_TEXT_HIDDEN,
    QWENIMAGE_EDIT_TEXT_KV_HEADS,
    QWENIMAGE_EDIT_TEXT_LAYERS,
    QWENIMAGE_EDIT_TEXT_MAX_TOKENS,
    QWENIMAGE_EDIT_TOTAL_SEQUENCE,
    QWENIMAGE_EDIT_TRANSFORMER_TENSORS,
    QWENIMAGE_EDIT_VAE_TENSORS,
    QWENIMAGE_EDIT_ZERO_COND_T,
    build_qwenimage_edit_token_plan,
    qwenimage_edit_default_cfg_scale,
    validate_qwenimage_edit_local_paths,
    validate_qwenimage_edit_metadata_contract,
)
from serenitymojo.runtime.model_manifest import qwen_image_edit_default_manifest
from serenitymojo.sampling.flow_match import build_qwen_sigma_schedule, qwen_mu


def main() raises:
    var manifest = qwen_image_edit_default_manifest()
    var checked_paths = validate_qwenimage_edit_local_paths()
    var plan = validate_qwenimage_edit_metadata_contract(manifest)
    var explicit_plan = build_qwenimage_edit_token_plan(1024, 1024, 1, QWENIMAGE_EDIT_TEXT_MAX_TOKENS)
    if explicit_plan.total_sequence != plan.total_sequence:
        raise Error("Qwen-Image-Edit explicit token plan mismatch")

    var sigmas = build_qwen_sigma_schedule(
        QWENIMAGE_EDIT_DEFAULT_STEPS, Float32(QWENIMAGE_EDIT_TARGET_TOKENS)
    )
    print("[qwen-image-edit-contract] paths checked/missing:", checked_paths, 0)
    print(
        "[qwen-image-edit-contract] headers transformer/text/vae tensors:",
        QWENIMAGE_EDIT_TRANSFORMER_TENSORS,
        QWENIMAGE_EDIT_TEXT_ENCODER_TENSORS,
        QWENIMAGE_EDIT_VAE_TENSORS,
    )
    print(
        "[qwen-image-edit-contract] DiT hidden/layers/heads/head_dim:",
        QWENIMAGE_EDIT_DIT_HIDDEN,
        QWENIMAGE_EDIT_DIT_LAYERS,
        QWENIMAGE_EDIT_DIT_HEADS,
        QWENIMAGE_EDIT_DIT_HEAD_DIM,
    )
    print(
        "[qwen-image-edit-contract] text hidden/layers/kv:",
        QWENIMAGE_EDIT_TEXT_HIDDEN,
        QWENIMAGE_EDIT_TEXT_LAYERS,
        QWENIMAGE_EDIT_TEXT_KV_HEADS,
    )
    print(
        "[qwen-image-edit-contract] target/reference/text/sequence:",
        QWENIMAGE_EDIT_TARGET_TOKENS,
        QWENIMAGE_EDIT_REFERENCE_TOKENS,
        QWENIMAGE_EDIT_IMAGE_TOKENS,
        QWENIMAGE_EDIT_TEXT_MAX_TOKENS,
        QWENIMAGE_EDIT_TOTAL_SEQUENCE,
    )
    print(
        "[qwen-image-edit-contract] latent/drop/pad/zero_cond_t:",
        QWENIMAGE_EDIT_LATENT_CHANNELS,
        QWENIMAGE_EDIT_LATENT_H,
        "x",
        QWENIMAGE_EDIT_LATENT_W,
        QWENIMAGE_EDIT_DROP_IDX,
        QWENIMAGE_EDIT_PAD_ID,
        QWENIMAGE_EDIT_ZERO_COND_T,
    )
    print(
        "[qwen-image-edit-contract] steps/cfg/mu/sigma0/preterminal/end:",
        QWENIMAGE_EDIT_DEFAULT_STEPS,
        qwenimage_edit_default_cfg_scale(),
        qwen_mu(Float32(QWENIMAGE_EDIT_TARGET_TOKENS)),
        sigmas[0],
        sigmas[QWENIMAGE_EDIT_DEFAULT_STEPS - 1],
        sigmas[QWENIMAGE_EDIT_DEFAULT_STEPS],
    )
    print("Qwen-Image-Edit metadata contract PASS")
