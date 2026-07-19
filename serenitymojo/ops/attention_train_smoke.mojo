# attention_train_smoke.mojo — training SDPA wrapper plan smoke.
#
# This smoke validates backend labels without running a flash kernel.
# Run:
#   pixi run mojo run -I . serenitymojo/ops/attention_train_smoke.mojo

from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.attention_train import (
    TRAIN_ATTN_BACKEND_CUDNN_FLASH,
    TRAIN_ATTN_BACKEND_CUDNN_FLASH_PADMASK,
    TRAIN_ATTN_BACKEND_STRICT_MATH,
    TRAIN_ATTN_BACKEND_TILED_MATH,
    TRAIN_ATTN_BACKEND_UNSUPPORTED,
    TRAIN_ATTN_MASK_ADDITIVE,
    TRAIN_ATTN_MASK_NONE,
    TRAIN_ATTN_MASK_PAD_TAIL,
    TRAIN_ATTN_MASK_QWEN_TEXT_KEY,
    select_training_attention_plan,
    training_attention_backend_name,
    training_attention_flash_head_dim_supported,
)


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("attention_train_smoke FAILED: ") + msg)


def _expect_plan(
    batch: Int,
    query_len: Int,
    key_value_len: Int,
    heads: Int,
    head_dim: Int,
    mask_kind: Int,
    dtype: STDtype,
    prefer_flash: Bool,
    expected_backend: Int,
    expected_reason: String,
) raises:
    var plan = select_training_attention_plan(
        batch,
        query_len,
        key_value_len,
        heads,
        head_dim,
        mask_kind,
        dtype,
        prefer_flash,
    )
    _check(plan.backend == expected_backend, String("backend for ") + expected_reason)
    _check(plan.fallback_reason == expected_reason, String("reason for ") + expected_reason)


def main() raises:
    var strict = select_training_attention_plan(
        1, 1248, 1248, 30, 128, TRAIN_ATTN_MASK_NONE, STDtype.BF16, False,
    )
    _check(strict.backend == TRAIN_ATTN_BACKEND_STRICT_MATH, "strict default")
    _check(strict.supports_backward, "strict supports backward")

    var flash = select_training_attention_plan(
        1, 1248, 1248, 30, 128, TRAIN_ATTN_MASK_NONE, STDtype.BF16, True,
    )
    _check(flash.backend == TRAIN_ATTN_BACKEND_CUDNN_FLASH, "flash selected")
    _check(flash.saved_forward_required, "flash must require saved forward")

    var flash_dims: List[Int] = [64, 96, 128, 256]
    for i in range(len(flash_dims)):
        var dim = flash_dims[i]
        _check(
            training_attention_flash_head_dim_supported(dim),
            String("flash head dim registered ") + String(dim),
        )
        var by_dim = select_training_attention_plan(
            1, 1024, 1024, 16, dim, TRAIN_ATTN_MASK_NONE, STDtype.BF16, True,
        )
        _check(by_dim.backend == TRAIN_ATTN_BACKEND_CUDNN_FLASH, "flash head-dim plan")

    _expect_plan(
        1,
        1024,
        1024,
        16,
        80,
        TRAIN_ATTN_MASK_NONE,
        STDtype.BF16,
        True,
        TRAIN_ATTN_BACKEND_STRICT_MATH,
        String("flash supports head dims 64/96/128/256"),
    )
    _expect_plan(
        1,
        1024,
        1024,
        16,
        128,
        TRAIN_ATTN_MASK_NONE,
        STDtype.F32,
        True,
        TRAIN_ATTN_BACKEND_STRICT_MATH,
        String("flash requires BF16 dtype"),
    )

    var rect_flash = select_training_attention_plan(
        1, 4096, 77, 16, 64, TRAIN_ATTN_MASK_NONE, STDtype.BF16, True,
    )
    _check(rect_flash.backend == TRAIN_ATTN_BACKEND_CUDNN_FLASH, "rectangular flash selected")
    _check(rect_flash.saved_forward_required, "rectangular flash must save forward")

    var pad = select_training_attention_plan(
        1, 4864, 4864, 48, 128, TRAIN_ATTN_MASK_PAD_TAIL, STDtype.BF16, True,
    )
    _check(pad.backend == TRAIN_ATTN_BACKEND_CUDNN_FLASH_PADMASK, "padmask flash selected")
    _expect_plan(
        1,
        4865,
        4865,
        48,
        128,
        TRAIN_ATTN_MASK_PAD_TAIL,
        STDtype.BF16,
        True,
        TRAIN_ATTN_BACKEND_TILED_MATH,
        String("pad-tail flash requires 128-aligned buffer"),
    )

    var masked = select_training_attention_plan(
        1, 4096, 77, 16, 64, TRAIN_ATTN_MASK_ADDITIVE, STDtype.BF16, False,
    )
    _check(masked.backend == TRAIN_ATTN_BACKEND_TILED_MATH, "masked fallback selected")
    _expect_plan(
        1,
        4096,
        77,
        16,
        64,
        TRAIN_ATTN_MASK_QWEN_TEXT_KEY,
        STDtype.BF16,
        False,
        TRAIN_ATTN_BACKEND_TILED_MATH,
        String("mask/shape uses explicit math fallback"),
    )
    _expect_plan(
        0,
        4096,
        77,
        16,
        64,
        TRAIN_ATTN_MASK_NONE,
        STDtype.BF16,
        True,
        TRAIN_ATTN_BACKEND_UNSUPPORTED,
        String("nonpositive dimension"),
    )
    _check(
        training_attention_backend_name(TRAIN_ATTN_BACKEND_CUDNN_FLASH).byte_length() > 10,
        "backend label",
    )
    print("PASS: training SDPA wrapper backend labels")
