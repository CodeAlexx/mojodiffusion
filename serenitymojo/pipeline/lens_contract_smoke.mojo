# Microsoft Lens sidecar contract smoke.
#
# Metadata/header-only: no DeviceContext, no tensor H2D load, no MXFP4 dequant,
# and no Lens DiT/GPT-OSS/VAE math.

from serenitymojo.models.lens.lens_contract import (
    LENS_CFG_SCALE,
    LENS_DIT_HEADS,
    LENS_DIT_INNER_DIM,
    LENS_GPT_OSS_EXPERTS,
    LENS_GPT_OSS_EXPERTS_PER_TOKEN,
    LENS_GPT_OSS_HIDDEN,
    LENS_GPT_OSS_LAYERS,
    LENS_GPT_OSS_SLIDING_WINDOW,
    LENS_IMAGE_TOKENS,
    LENS_NUM_STEPS,
    LENS_POST_OFFSET_TEXT_TOKENS,
    LENS_SCHEDULER_SHIFT_X10,
    LENS_SCHEDULER_TRAIN_STEPS,
    LENS_TEXT_ENCODER_TENSORS,
    LENS_TRANSFORMER_TENSORS,
    LENS_VAE_TENSORS,
    LENS_ZERO_FEATURE_TEXT_TOKENS,
    build_lens_token_plan,
    validate_lens_sidecar_contract,
)


def main() raises:
    var checked_paths = validate_lens_sidecar_contract()
    var real_plan = build_lens_token_plan(1024, 1024, LENS_POST_OFFSET_TEXT_TOKENS)
    var zero_plan = build_lens_token_plan(1024, 1024, LENS_ZERO_FEATURE_TEXT_TOKENS)
    print("[lens-contract] paths checked/missing:", checked_paths, 0)
    print(
        "[lens-contract] headers transformer/text/vae tensors:",
        LENS_TRANSFORMER_TENSORS,
        LENS_TEXT_ENCODER_TENSORS,
        LENS_VAE_TENSORS,
    )
    print(
        "[lens-contract] DiT hidden/heads/image_tokens:",
        LENS_DIT_INNER_DIM,
        LENS_DIT_HEADS,
        LENS_IMAGE_TOKENS,
    )
    print(
        "[lens-contract] GPT-OSS hidden/layers/experts/topk/window:",
        LENS_GPT_OSS_HIDDEN,
        LENS_GPT_OSS_LAYERS,
        LENS_GPT_OSS_EXPERTS,
        LENS_GPT_OSS_EXPERTS_PER_TOKEN,
        LENS_GPT_OSS_SLIDING_WINDOW,
    )
    print(
        "[lens-contract] real text/sequence:",
        real_plan.text_tokens,
        real_plan.total_sequence,
    )
    print(
        "[lens-contract] zero-feature text/sequence:",
        zero_plan.text_tokens,
        zero_plan.total_sequence,
    )
    print("[lens-contract] steps/cfg:", LENS_NUM_STEPS, LENS_CFG_SCALE)
    print(
        "[lens-contract] scheduler train_steps/shift_x10:",
        LENS_SCHEDULER_TRAIN_STEPS,
        LENS_SCHEDULER_SHIFT_X10,
    )
    print("Microsoft Lens sidecar contract PASS")
