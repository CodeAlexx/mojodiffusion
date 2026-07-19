# Chroma image-model contract smoke.
#
# Metadata/header-only: no DeviceContext, no Tensor allocations, no H2D loads,
# and no denoise or VAE math. It checks the local Chroma1-HD checkpoint and the
# local HF snapshot sidecars without registering a shared manifest.

from serenitymojo.models.dit.chroma_contract import (
    CHROMA_DEFAULT_STEPS,
    CHROMA_DIT_DOUBLE_BLOCKS,
    CHROMA_DIT_HEAD_DIM,
    CHROMA_DIT_HEADS,
    CHROMA_DIT_HIDDEN,
    CHROMA_DIT_MOD_INDEX,
    CHROMA_DIT_SINGLE_BLOCKS,
    CHROMA_DIT_TENSORS,
    CHROMA_IMAGE_TOKENS,
    CHROMA_LATENT_CHANNELS,
    CHROMA_PATCH_GRID_H,
    CHROMA_PATCH_GRID_W,
    CHROMA_PATCH_VECTOR_DIM,
    CHROMA_T5_HIDDEN,
    CHROMA_T5_LAYERS,
    CHROMA_T5_SEQ_LEN,
    CHROMA_TEXT_ENCODER_TENSORS,
    CHROMA_TOTAL_SEQUENCE,
    CHROMA_VAE_TENSORS,
    build_chroma_token_plan,
    chroma_default_cfg_scale,
    chroma_default_checkpoint_path,
    chroma_schedule_delta,
    chroma_schedule_shift,
    chroma_shifted_sigma,
    chroma_text_encoder_dir,
    chroma_transformer_dir,
    chroma_vae_path,
    chroma_vae_scale,
    chroma_vae_shift,
    validate_chroma_default_checkpoint_contract,
    validate_chroma_local_paths,
)


def main() raises:
    var checked_paths = validate_chroma_local_paths()
    var plan = validate_chroma_default_checkpoint_contract()
    var explicit_plan = build_chroma_token_plan(1024, 1024, 1, CHROMA_T5_SEQ_LEN)
    if explicit_plan.total_sequence != plan.total_sequence:
        raise Error("Chroma explicit token plan mismatch")

    var shift = chroma_schedule_shift()
    print("[chroma-contract] paths checked/missing:", checked_paths, 0)
    print("[chroma-contract] DiT checkpoint:", chroma_default_checkpoint_path())
    print("[chroma-contract] transformer dir:", chroma_transformer_dir())
    print("[chroma-contract] text encoder dir:", chroma_text_encoder_dir())
    print("[chroma-contract] VAE:", chroma_vae_path())
    print(
        "[chroma-contract] headers dit/text/vae tensors:",
        CHROMA_DIT_TENSORS,
        CHROMA_TEXT_ENCODER_TENSORS,
        CHROMA_VAE_TENSORS,
    )
    print(
        "[chroma-contract] DiT hidden/double/single/heads/head_dim/mod_index:",
        CHROMA_DIT_HIDDEN,
        CHROMA_DIT_DOUBLE_BLOCKS,
        CHROMA_DIT_SINGLE_BLOCKS,
        CHROMA_DIT_HEADS,
        CHROMA_DIT_HEAD_DIM,
        CHROMA_DIT_MOD_INDEX,
    )
    print(
        "[chroma-contract] T5 layers/seq/hidden:",
        CHROMA_T5_LAYERS,
        CHROMA_T5_SEQ_LEN,
        CHROMA_T5_HIDDEN,
    )
    print(
        "[chroma-contract] latent/patch/tokens/sequence:",
        CHROMA_LATENT_CHANNELS,
        plan.latent_h,
        "x",
        plan.latent_w,
        CHROMA_PATCH_GRID_H,
        "x",
        CHROMA_PATCH_GRID_W,
        CHROMA_PATCH_VECTOR_DIM,
        CHROMA_IMAGE_TOKENS,
        CHROMA_TOTAL_SEQUENCE,
    )
    print(
        "[chroma-contract] steps/cfg/shift/vae:",
        CHROMA_DEFAULT_STEPS,
        chroma_default_cfg_scale(),
        shift,
        chroma_vae_scale(),
        chroma_vae_shift(),
    )
    print(
        "[chroma-contract] sigma0/mid/end/dt0:",
        chroma_shifted_sigma(0, CHROMA_DEFAULT_STEPS, shift),
        chroma_shifted_sigma(CHROMA_DEFAULT_STEPS // 2, CHROMA_DEFAULT_STEPS, shift),
        chroma_shifted_sigma(CHROMA_DEFAULT_STEPS, CHROMA_DEFAULT_STEPS, shift),
        chroma_schedule_delta(0, CHROMA_DEFAULT_STEPS, shift),
    )
    print("Chroma image-model contract PASS")
