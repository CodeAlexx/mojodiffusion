# ops/attention_train.mojo — model-agnostic training SDPA wrapper.
#
# Existing model code calls low-level SDPA variants directly. This module gives
# trainers one place to select and label the backend: strict math/tiled paths for
# parity by default, explicit cuDNN flash paths when a caller has accepted the
# flash numerics and saved-forward contract.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.ops.attention import (
    sdpa,
    sdpa_cross_masked,
    sdpa_cross_nomask,
    sdpa_nomask,
    sdpa_nomask_tiled,
    sdpa_tiled,
)
from serenitymojo.ops.attention_backward import (
    SdpaGrads,
    sdpa_backward,
    sdpa_backward_masked,
    sdpa_backward_masked_batched,
    sdpa_backward_rect,
)
from serenitymojo.ops.attention_flash import (
    SdpaFlashFwd,
    SdpaFlashGrads,
    sdpa_flash_backward,
    sdpa_flash_fwd_padmask,
    sdpa_flash_train_fwd,
    sdpa_flash_train_fwd_rect,
)


comptime TRAIN_ATTN_MASK_NONE = 0
comptime TRAIN_ATTN_MASK_ADDITIVE = 1
comptime TRAIN_ATTN_MASK_PAD_TAIL = 2
comptime TRAIN_ATTN_MASK_QWEN_TEXT_KEY = 3

comptime TRAIN_ATTN_BACKEND_STRICT_MATH = 0
comptime TRAIN_ATTN_BACKEND_TILED_MATH = 1
comptime TRAIN_ATTN_BACKEND_CUDNN_FLASH = 2
comptime TRAIN_ATTN_BACKEND_CUDNN_FLASH_PADMASK = 3
comptime TRAIN_ATTN_BACKEND_UNSUPPORTED = 4


@fieldwise_init
struct TrainingAttentionPlan(Copyable, Movable, Writable):
    var batch: Int
    var query_len: Int
    var key_value_len: Int
    var heads: Int
    var head_dim: Int
    var mask_kind: Int
    var dtype: String
    var backend: Int
    var supports_backward: Bool
    var saved_forward_required: Bool
    var fallback_reason: String

    def backend_name(self) -> String:
        return training_attention_backend_name(self.backend)

    def mask_name(self) -> String:
        return training_attention_mask_name(self.mask_kind)

    def is_flash(self) -> Bool:
        return (
            self.backend == TRAIN_ATTN_BACKEND_CUDNN_FLASH
            or self.backend == TRAIN_ATTN_BACKEND_CUDNN_FLASH_PADMASK
        )

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "TrainingAttentionPlan(B=",
            self.batch,
            ", Sq=",
            self.query_len,
            ", Skv=",
            self.key_value_len,
            ", H=",
            self.heads,
            ", Dh=",
            self.head_dim,
            ", mask=",
            self.mask_name(),
            ", dtype=",
            self.dtype,
            ", backend=",
            self.backend_name(),
            ", reason=",
            self.fallback_reason,
            ")",
        )


struct TrainingSdpaTensorResult(Movable, Writable):
    var out: Tensor
    var plan: TrainingAttentionPlan

    def __init__(out self, var out: Tensor, plan: TrainingAttentionPlan):
        self.out = out^
        self.plan = plan.copy()

    def write_to(self, mut writer: Some[Writer]):
        writer.write("TrainingSdpaTensorResult(", self.plan, ")")


def training_attention_mask_name(mask_kind: Int) -> String:
    if mask_kind == TRAIN_ATTN_MASK_NONE:
        return String("none")
    if mask_kind == TRAIN_ATTN_MASK_ADDITIVE:
        return String("additive")
    if mask_kind == TRAIN_ATTN_MASK_PAD_TAIL:
        return String("pad-tail")
    if mask_kind == TRAIN_ATTN_MASK_QWEN_TEXT_KEY:
        return String("qwen-text-key")
    return String("unknown")


def training_attention_backend_name(backend: Int) -> String:
    if backend == TRAIN_ATTN_BACKEND_STRICT_MATH:
        return String("training-sdpa-strict-math")
    if backend == TRAIN_ATTN_BACKEND_TILED_MATH:
        return String("training-sdpa-tiled-math")
    if backend == TRAIN_ATTN_BACKEND_CUDNN_FLASH:
        return String("training-sdpa-cudnn-flash")
    if backend == TRAIN_ATTN_BACKEND_CUDNN_FLASH_PADMASK:
        return String("training-sdpa-cudnn-flash-padmask")
    return String("training-sdpa-unsupported")


def _flash_supported_dtype(dtype: STDtype) -> Bool:
    return dtype == STDtype.BF16


def training_attention_flash_head_dim_supported(head_dim: Int) -> Bool:
    return (
        head_dim == 64
        or head_dim == 96
        or head_dim == 128
        or head_dim == 256
    )


def _math_backend_for_mask(mask_kind: Int) -> Int:
    if mask_kind == TRAIN_ATTN_MASK_NONE:
        return TRAIN_ATTN_BACKEND_STRICT_MATH
    return TRAIN_ATTN_BACKEND_TILED_MATH


def _math_fallback_plan(
    batch: Int,
    query_len: Int,
    key_value_len: Int,
    heads: Int,
    head_dim: Int,
    mask_kind: Int,
    dtype_name: String,
    reason: String,
) -> TrainingAttentionPlan:
    return TrainingAttentionPlan(
        batch,
        query_len,
        key_value_len,
        heads,
        head_dim,
        mask_kind,
        dtype_name.copy(),
        _math_backend_for_mask(mask_kind),
        True,
        False,
        reason.copy(),
    )


def select_training_attention_plan(
    batch: Int,
    query_len: Int,
    key_value_len: Int,
    heads: Int,
    head_dim: Int,
    mask_kind: Int,
    dtype: STDtype,
    prefer_flash: Bool,
) -> TrainingAttentionPlan:
    var dtype_name = dtype.name()
    if batch <= 0 or query_len <= 0 or key_value_len <= 0 or heads <= 0 or head_dim <= 0:
        return TrainingAttentionPlan(
            batch,
            query_len,
            key_value_len,
            heads,
            head_dim,
            mask_kind,
            dtype_name,
            TRAIN_ATTN_BACKEND_UNSUPPORTED,
            False,
            False,
            String("nonpositive dimension"),
        )
    if prefer_flash and not _flash_supported_dtype(dtype):
        return _math_fallback_plan(
            batch,
            query_len,
            key_value_len,
            heads,
            head_dim,
            mask_kind,
            dtype_name,
            String("flash requires BF16 dtype"),
        )
    if prefer_flash and not training_attention_flash_head_dim_supported(head_dim):
        return _math_fallback_plan(
            batch,
            query_len,
            key_value_len,
            heads,
            head_dim,
            mask_kind,
            dtype_name,
            String("flash supports head dims 64/96/128/256"),
        )
    if prefer_flash:
        if mask_kind == TRAIN_ATTN_MASK_NONE:
            return TrainingAttentionPlan(
                batch,
                query_len,
                key_value_len,
                heads,
                head_dim,
                mask_kind,
                dtype_name,
                TRAIN_ATTN_BACKEND_CUDNN_FLASH,
                True,
                True,
                String("explicit flash request"),
            )
        if mask_kind == TRAIN_ATTN_MASK_PAD_TAIL and query_len == key_value_len:
            if (query_len % 128) == 0:
                return TrainingAttentionPlan(
                    batch,
                    query_len,
                    key_value_len,
                    heads,
                    head_dim,
                    mask_kind,
                    dtype_name,
                    TRAIN_ATTN_BACKEND_CUDNN_FLASH_PADMASK,
                    True,
                    True,
                    String("explicit flash pad-tail request"),
                )
            return TrainingAttentionPlan(
                batch,
                query_len,
                key_value_len,
                heads,
                head_dim,
                mask_kind,
                dtype_name,
                TRAIN_ATTN_BACKEND_TILED_MATH,
                True,
                False,
                String("pad-tail flash requires 128-aligned buffer"),
            )
    if mask_kind == TRAIN_ATTN_MASK_NONE:
        return TrainingAttentionPlan(
            batch,
            query_len,
            key_value_len,
            heads,
            head_dim,
            mask_kind,
            dtype_name,
            TRAIN_ATTN_BACKEND_STRICT_MATH,
            True,
            False,
            String("strict parity default"),
        )
    return TrainingAttentionPlan(
        batch,
        query_len,
        key_value_len,
        heads,
        head_dim,
        mask_kind,
        dtype_name,
        TRAIN_ATTN_BACKEND_TILED_MATH,
        True,
        False,
        String("mask/shape uses explicit math fallback"),
    )


def training_sdpa_nomask_strict[
    B: Int, S: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> TrainingSdpaTensorResult:
    var plan = select_training_attention_plan(
        B, S, S, H, Dh, TRAIN_ATTN_MASK_NONE, q.dtype(), False,
    )
    var out = sdpa_nomask[B, S, H, Dh](q, k, v, scale, ctx)
    return TrainingSdpaTensorResult(out^, plan)


def training_sdpa_nomask_tiled[
    B: Int, S: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> TrainingSdpaTensorResult:
    var plan = TrainingAttentionPlan(
        B,
        S,
        S,
        H,
        Dh,
        TRAIN_ATTN_MASK_NONE,
        q.dtype().name(),
        TRAIN_ATTN_BACKEND_TILED_MATH,
        True,
        False,
        String("explicit tiled request"),
    )
    var out = sdpa_nomask_tiled[B, S, H, Dh](q, k, v, scale, ctx)
    return TrainingSdpaTensorResult(out^, plan)


def training_sdpa_masked_strict[
    B: Int, S: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> TrainingSdpaTensorResult:
    var plan = TrainingAttentionPlan(
        B,
        S,
        S,
        H,
        Dh,
        TRAIN_ATTN_MASK_ADDITIVE,
        q.dtype().name(),
        TRAIN_ATTN_BACKEND_STRICT_MATH,
        True,
        False,
        String("strict additive-mask parity"),
    )
    var out = sdpa[B, S, H, Dh](q, k, v, mask, scale, ctx)
    return TrainingSdpaTensorResult(out^, plan)


def training_sdpa_masked_tiled[
    B: Int, S: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> TrainingSdpaTensorResult:
    var plan = TrainingAttentionPlan(
        B,
        S,
        S,
        H,
        Dh,
        TRAIN_ATTN_MASK_ADDITIVE,
        q.dtype().name(),
        TRAIN_ATTN_BACKEND_TILED_MATH,
        True,
        False,
        String("explicit tiled additive-mask request"),
    )
    var out = sdpa_tiled[B, S, H, Dh](q, k, v, mask, scale, ctx)
    return TrainingSdpaTensorResult(out^, plan)


def training_sdpa_cross_nomask_strict[
    B: Int, Sq: Int, Skv: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> TrainingSdpaTensorResult:
    var plan = select_training_attention_plan(
        B, Sq, Skv, H, Dh, TRAIN_ATTN_MASK_NONE, q.dtype(), False,
    )
    var out = sdpa_cross_nomask[B, Sq, Skv, H, Dh](q, k, v, scale, ctx)
    return TrainingSdpaTensorResult(out^, plan)


def training_sdpa_cross_masked_strict[
    B: Int, Sq: Int, Skv: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> TrainingSdpaTensorResult:
    var plan = TrainingAttentionPlan(
        B,
        Sq,
        Skv,
        H,
        Dh,
        TRAIN_ATTN_MASK_ADDITIVE,
        q.dtype().name(),
        TRAIN_ATTN_BACKEND_TILED_MATH,
        True,
        False,
        String("rectangular additive-mask fallback"),
    )
    var out = sdpa_cross_masked[B, Sq, Skv, H, Dh](q, k, v, mask, scale, ctx)
    return TrainingSdpaTensorResult(out^, plan)


def training_sdpa_backward_nomask_strict[
    B: Int, S: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    d_out: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> SdpaGrads:
    return sdpa_backward[B, S, H, Dh](q, k, v, d_out, scale, ctx)


def training_sdpa_backward_masked_strict[
    B: Int, S: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask_f32: Tensor,
    d_out: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> SdpaGrads:
    return sdpa_backward_masked[B, S, H, Dh](
        q, k, v, mask_f32, d_out, scale, ctx,
    )


def training_sdpa_backward_masked_batched_strict[
    B: Int, S: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask_f32: Tensor,
    d_out: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> SdpaGrads:
    return sdpa_backward_masked_batched[B, S, H, Dh](
        q, k, v, mask_f32, d_out, scale, ctx,
    )


def training_sdpa_backward_rect_strict[
    B: Int, Sq: Int, Skv: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    d_out: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> SdpaGrads:
    return sdpa_backward_rect[B, Sq, Skv, H, Dh](q, k, v, d_out, scale, ctx)


def training_sdpa_flash_saved_fwd[
    B: Int, S: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> SdpaFlashFwd:
    if q.dtype() != STDtype.BF16:
        raise Error("training_sdpa_flash_saved_fwd: q/k/v must be BF16")
    return sdpa_flash_train_fwd[B, S, H, Dh](q, k, v, scale, ctx)


def training_sdpa_flash_saved_fwd_rect[
    B: Int, Sq: Int, Skv: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> SdpaFlashFwd:
    if q.dtype() != STDtype.BF16:
        raise Error("training_sdpa_flash_saved_fwd_rect: q/k/v must be BF16")
    return sdpa_flash_train_fwd_rect[B, Sq, Skv, H, Dh](q, k, v, scale, ctx)


def training_sdpa_flash_padmask_saved_fwd[
    B: Int, S: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    real_len: Int,
    scale: Float32,
    ctx: DeviceContext,
) raises -> SdpaFlashFwd:
    if q.dtype() != STDtype.BF16:
        raise Error("training_sdpa_flash_padmask_saved_fwd: q/k/v must be BF16")
    return sdpa_flash_fwd_padmask[B, S, H, Dh](q, k, v, real_len, scale, ctx)


def training_sdpa_flash_backward[
    B: Int, S: Int, H: Int, Dh: Int
](
    fwd: SdpaFlashFwd,
    d_out: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> SdpaFlashGrads:
    return sdpa_flash_backward[B, S, H, Dh](fwd, d_out, scale, ctx)
