# ERNIE-Image metadata contract smoke.
#
# Header-only: no DeviceContext, no tensor H2D load, no ERNIE/Mistral math, and
# no Klein VAE decode.

from serenitymojo.models.dit.ernie_contract import (
    ERNIE_DEFAULT_CFG_X10,
    ERNIE_DEFAULT_STEPS,
    ERNIE_DIT_HEAD_DIM,
    ERNIE_DIT_HEADS,
    ERNIE_DIT_HIDDEN,
    ERNIE_DIT_LAYERS,
    ERNIE_IMAGE_TOKENS,
    ERNIE_LATENT_CHANNELS,
    ERNIE_LATENT_H,
    ERNIE_LATENT_W,
    ERNIE_MISTRAL_EXTRACT_LAYER,
    ERNIE_MISTRAL_HEADS,
    ERNIE_MISTRAL_HIDDEN,
    ERNIE_MISTRAL_KV_HEADS,
    ERNIE_MISTRAL_LAYERS,
    ERNIE_SCHEDULER_SHIFT_X10,
    ERNIE_TEXT_MAX_TOKENS,
    ERNIE_TOTAL_SEQUENCE,
    ERNIE_TRANSFORMER_TENSORS,
    ERNIE_TEXT_ENCODER_TENSORS,
    ERNIE_VAE_TENSORS,
    build_ernie_token_plan,
    ernie_default_shift,
    ernie_euler_delta,
    ernie_model_timestep,
    ernie_sigma,
    validate_ernie_local_paths,
    validate_ernie_metadata_contract,
)
from serenitymojo.runtime.model_manifest import ernie_image_default_manifest


def main() raises:
    var manifest = ernie_image_default_manifest()
    var checked_paths = validate_ernie_local_paths()
    var plan = validate_ernie_metadata_contract(manifest)
    var explicit_plan = build_ernie_token_plan(1024, 1024, 1, ERNIE_TEXT_MAX_TOKENS)
    if explicit_plan.total_sequence != plan.total_sequence:
        raise Error("ERNIE explicit token plan mismatch")

    var shift = ernie_default_shift()
    print("[ernie-contract] paths checked/missing:", checked_paths, 0)
    print(
        "[ernie-contract] headers transformer/text/vae tensors:",
        ERNIE_TRANSFORMER_TENSORS,
        ERNIE_TEXT_ENCODER_TENSORS,
        ERNIE_VAE_TENSORS,
    )
    print(
        "[ernie-contract] DiT hidden/layers/heads/head_dim:",
        ERNIE_DIT_HIDDEN,
        ERNIE_DIT_LAYERS,
        ERNIE_DIT_HEADS,
        ERNIE_DIT_HEAD_DIM,
    )
    print(
        "[ernie-contract] Mistral hidden/layers/extract/head/kv:",
        ERNIE_MISTRAL_HIDDEN,
        ERNIE_MISTRAL_LAYERS,
        ERNIE_MISTRAL_EXTRACT_LAYER,
        ERNIE_MISTRAL_HEADS,
        ERNIE_MISTRAL_KV_HEADS,
    )
    print(
        "[ernie-contract] latent/tokens/sequence:",
        ERNIE_LATENT_CHANNELS,
        ERNIE_LATENT_H,
        "x",
        ERNIE_LATENT_W,
        ERNIE_IMAGE_TOKENS,
        ERNIE_TEXT_MAX_TOKENS,
        ERNIE_TOTAL_SEQUENCE,
    )
    print(
        "[ernie-contract] steps/cfg_x10/shift_x10:",
        ERNIE_DEFAULT_STEPS,
        ERNIE_DEFAULT_CFG_X10,
        ERNIE_SCHEDULER_SHIFT_X10,
    )
    print(
        "[ernie-contract] sigma0/mid/end/dt0/t0:",
        ernie_sigma(0, ERNIE_DEFAULT_STEPS, shift),
        ernie_sigma(ERNIE_DEFAULT_STEPS // 2, ERNIE_DEFAULT_STEPS, shift),
        ernie_sigma(ERNIE_DEFAULT_STEPS, ERNIE_DEFAULT_STEPS, shift),
        ernie_euler_delta(0, ERNIE_DEFAULT_STEPS, shift),
        ernie_model_timestep(1.0),
    )
    print("ERNIE-Image metadata contract PASS")
