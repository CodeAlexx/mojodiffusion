# Qwen-Image metadata contract smoke.
#
# Header-only: no DeviceContext, no tensor H2D load, no Qwen2.5-VL/DiT math,
# and no VAE decode.

from serenitymojo.models.dit.qwenimage_contract import (
    QWENIMAGE_DEFAULT_CFG_X10,
    QWENIMAGE_DEFAULT_STEPS,
    QWENIMAGE_DIT_HEAD_DIM,
    QWENIMAGE_DIT_HEADS,
    QWENIMAGE_DIT_HIDDEN,
    QWENIMAGE_DIT_LAYERS,
    QWENIMAGE_DROP_IDX,
    QWENIMAGE_IMAGE_TOKENS,
    QWENIMAGE_LATENT_CHANNELS,
    QWENIMAGE_LATENT_H,
    QWENIMAGE_LATENT_W,
    QWENIMAGE_PAD_ID,
    QWENIMAGE_PATCH_GRID_H,
    QWENIMAGE_PATCH_GRID_W,
    QWENIMAGE_TEXT_ENCODER_TENSORS,
    QWENIMAGE_TEXT_HIDDEN,
    QWENIMAGE_TEXT_KV_HEADS,
    QWENIMAGE_TEXT_LAYERS,
    QWENIMAGE_TEXT_MAX_TOKENS,
    QWENIMAGE_TOTAL_SEQUENCE,
    QWENIMAGE_TRANSFORMER_TENSORS,
    QWENIMAGE_VAE_TENSORS,
    build_qwenimage_token_plan,
    qwenimage_model_timestep,
    validate_qwenimage_local_paths,
    validate_qwenimage_metadata_contract,
)
from serenitymojo.runtime.model_manifest import qwen_image_default_manifest
from serenitymojo.sampling.flow_match import build_qwen_sigma_schedule, qwen_mu


def main() raises:
    var manifest = qwen_image_default_manifest()
    var checked_paths = validate_qwenimage_local_paths()
    var plan = validate_qwenimage_metadata_contract(manifest)
    var explicit_plan = build_qwenimage_token_plan(1024, 1024, 1, QWENIMAGE_TEXT_MAX_TOKENS)
    if explicit_plan.total_sequence != plan.total_sequence:
        raise Error("Qwen-Image explicit token plan mismatch")

    var sigmas = build_qwen_sigma_schedule(
        QWENIMAGE_DEFAULT_STEPS, Float32(QWENIMAGE_IMAGE_TOKENS)
    )
    print("[qwen-image-contract] paths checked/missing:", checked_paths, 0)
    print(
        "[qwen-image-contract] headers transformer/text/vae tensors:",
        QWENIMAGE_TRANSFORMER_TENSORS,
        QWENIMAGE_TEXT_ENCODER_TENSORS,
        QWENIMAGE_VAE_TENSORS,
    )
    print(
        "[qwen-image-contract] DiT hidden/layers/heads/head_dim:",
        QWENIMAGE_DIT_HIDDEN,
        QWENIMAGE_DIT_LAYERS,
        QWENIMAGE_DIT_HEADS,
        QWENIMAGE_DIT_HEAD_DIM,
    )
    print(
        "[qwen-image-contract] text hidden/layers/kv:",
        QWENIMAGE_TEXT_HIDDEN,
        QWENIMAGE_TEXT_LAYERS,
        QWENIMAGE_TEXT_KV_HEADS,
    )
    print(
        "[qwen-image-contract] latent/patch/tokens/sequence:",
        QWENIMAGE_LATENT_CHANNELS,
        QWENIMAGE_LATENT_H,
        "x",
        QWENIMAGE_LATENT_W,
        QWENIMAGE_PATCH_GRID_H,
        "x",
        QWENIMAGE_PATCH_GRID_W,
        QWENIMAGE_IMAGE_TOKENS,
        QWENIMAGE_TEXT_MAX_TOKENS,
        QWENIMAGE_TOTAL_SEQUENCE,
    )
    print(
        "[qwen-image-contract] tokenizer drop/pad:",
        QWENIMAGE_DROP_IDX,
        QWENIMAGE_PAD_ID,
    )
    print(
        "[qwen-image-contract] steps/cfg_x10/mu:",
        QWENIMAGE_DEFAULT_STEPS,
        QWENIMAGE_DEFAULT_CFG_X10,
        qwen_mu(Float32(QWENIMAGE_IMAGE_TOKENS)),
    )
    print(
        "[qwen-image-contract] sigma0/1/preterminal/end/t0:",
        sigmas[0],
        sigmas[1],
        sigmas[QWENIMAGE_DEFAULT_STEPS - 1],
        sigmas[QWENIMAGE_DEFAULT_STEPS],
        qwenimage_model_timestep(1.0),
    )
    print("Qwen-Image metadata contract PASS")
