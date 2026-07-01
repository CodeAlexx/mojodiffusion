# models/krea2/krea2_text_fusion_lora.mojo — Krea2 TextFusion LoRA training unit.
#
# This is the txtfusion-side sibling of krea2_block.mojo. It keeps the same
# adapter slot order (wq wk wv gate wo mlp_gate mlp_up mlp_down) so indices
# 224..255 can be appended after the 28 main blocks without changing optimizer
# semantics. Storage/activation boundaries follow the product contract: BF16 in
# and out when the cache/checkpoint path is BF16; F32 is limited to internals of
# existing kernels, reductions, and optimizer-bound grad buffers.

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import sigmoid, swiglu
from serenitymojo.ops.attention import sdpa, sdpa_nomask
from serenitymojo.ops.tensor_algebra import (
    add, mul, reshape, reshape_owned, transpose,
)
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linalg_backward import linear_backward_dx
from serenitymojo.ops.attention_backward import (
    sdpa_backward, sdpa_backward_masked,
)
from serenitymojo.ops.attention_flash import (
    SdpaFlashFwd,
    sdpa_flash_backward_native,
    sdpa_flash_train_fwd_native,
)
from serenitymojo.ops.loss_swiglu_backward import swiglu_backward
from serenitymojo.ops.activation_backward import sigmoid_backward_from_output
from serenitymojo.models.dit.krea2_dit import (
    Krea2TextFusionWeights,
    krea2_rmsnorm,
    krea2_rmsnorm_backward_dx,
)
from serenitymojo.models.klein.lora_block import (
    LoraAdapterDevice,
    klein_lora_fwd_device_resident_unfused,
    klein_lora_bwd_device_resident_tensors_unfused,
)
from serenitymojo.models.krea2.krea2_block import (
    Krea2BlockLora, Krea2LoraGradT, Krea2BlockGradsT,
)
from serenitymojo.training.lora_adamw_plain_fused import (
    LoraAdamWPlainDeviceState,
    lora_adamw_plain_device_state_copy_device_grad_pair,
)

comptime TArc = ArcPointer[Tensor]


def _no_bias() -> Optional[Tensor]:
    return Optional[Tensor](None)


def _text_lora_delta(
    x: Tensor, lo: Optional[LoraAdapterDevice], M: Int, ctx: DeviceContext
) raises -> Optional[Tensor]:
    if lo:
        return Optional[Tensor](
            klein_lora_fwd_device_resident_unfused(x, lo.value(), M, ctx)
        )
    return Optional[Tensor](None)


def _text_linear_lora(
    x: Tensor, w: Tensor, lo: Optional[LoraAdapterDevice], M: Int, ctx: DeviceContext
) raises -> Tensor:
    var nb = _no_bias()
    var y = linear(x, w, nb^, ctx)
    var d = _text_lora_delta(x, lo, M, ctx)
    if d:
        y = add(y, d.value(), ctx)
    return y^


@fieldwise_init
struct Krea2TextFusionLora(Copyable, Movable):
    var layerwise0: Krea2BlockLora
    var layerwise1: Krea2BlockLora
    var refiner0: Krea2BlockLora
    var refiner1: Krea2BlockLora


struct Krea2TextFusionBlockSaved(Copyable, Movable):
    var x: TArc
    var xn: TArc
    var q_pre: TArc
    var k_pre: TArc
    var v: TArc
    var q_norm: TArc
    var k_norm: TArc
    var attn_flat: TArc
    var gate_pre: TArc
    var sg: TArc
    var gated: TArc
    var a: TArc
    var x1: TArc
    var xn2: TArc
    var mlp_gate: TArc
    var mlp_up: TArc
    var sw: TArc
    var m: TArc
    var flash_q: Optional[TArc]
    var flash_k: Optional[TArc]
    var flash_v: Optional[TArc]
    var flash_o: Optional[TArc]
    var flash_stats: Optional[TArc]

    def __init__(
        out self,
        var x: TArc, var xn: TArc,
        var q_pre: TArc, var k_pre: TArc, var v: TArc,
        var q_norm: TArc, var k_norm: TArc,
        var attn_flat: TArc, var gate_pre: TArc, var sg: TArc,
        var gated: TArc, var a: TArc, var x1: TArc, var xn2: TArc,
        var mlp_gate: TArc, var mlp_up: TArc, var sw: TArc, var m: TArc,
        var flash_q: Optional[TArc] = Optional[TArc](None),
        var flash_k: Optional[TArc] = Optional[TArc](None),
        var flash_v: Optional[TArc] = Optional[TArc](None),
        var flash_o: Optional[TArc] = Optional[TArc](None),
        var flash_stats: Optional[TArc] = Optional[TArc](None),
    ):
        self.x = x^
        self.xn = xn^
        self.q_pre = q_pre^
        self.k_pre = k_pre^
        self.v = v^
        self.q_norm = q_norm^
        self.k_norm = k_norm^
        self.attn_flat = attn_flat^
        self.gate_pre = gate_pre^
        self.sg = sg^
        self.gated = gated^
        self.a = a^
        self.x1 = x1^
        self.xn2 = xn2^
        self.mlp_gate = mlp_gate^
        self.mlp_up = mlp_up^
        self.sw = sw^
        self.m = m^
        self.flash_q = flash_q^
        self.flash_k = flash_k^
        self.flash_v = flash_v^
        self.flash_o = flash_o^
        self.flash_stats = flash_stats^


struct Krea2TextFusionBlockForward(Movable):
    var out: TArc
    var saved: Krea2TextFusionBlockSaved

    def __init__(out self, var out: TArc, var saved: Krea2TextFusionBlockSaved):
        self.out = out^
        self.saved = saved^


struct Krea2TextFusionForward(Movable):
    var out: TArc
    var layerwise0: Krea2TextFusionBlockForward
    var layerwise1: Krea2TextFusionBlockForward
    var projector_in: TArc
    var projector_transposed: TArc
    var projector_out: TArc
    var refiner0: Krea2TextFusionBlockForward
    var refiner1: Krea2TextFusionBlockForward

    def __init__(
        out self,
        var out: TArc,
        var layerwise0: Krea2TextFusionBlockForward,
        var layerwise1: Krea2TextFusionBlockForward,
        var projector_in: TArc,
        var projector_transposed: TArc,
        var projector_out: TArc,
        var refiner0: Krea2TextFusionBlockForward,
        var refiner1: Krea2TextFusionBlockForward,
    ):
        self.out = out^
        self.layerwise0 = layerwise0^
        self.layerwise1 = layerwise1^
        self.projector_in = projector_in^
        self.projector_transposed = projector_transposed^
        self.projector_out = projector_out^
        self.refiner0 = refiner0^
        self.refiner1 = refiner1^


struct _TextLinBwdT(Movable):
    var d_x: Tensor
    var lora: Krea2LoraGradT

    def __init__(out self, var d_x: Tensor, var lora: Krea2LoraGradT):
        self.d_x = d_x^
        self.lora = lora^


def _text_linear_bwd_dx_dev(
    d_y_in: Tensor,
    x: Tensor,
    w: Tensor,
    lo: Optional[LoraAdapterDevice],
    M: Int,
    in_f: Int,
    out_f: Int,
    ctx: DeviceContext,
) raises -> _TextLinBwdT:
    var d_y: Tensor
    if d_y_in.dtype() == x.dtype():
        d_y = Tensor(d_y_in.buf.copy(), d_y_in.shape(), d_y_in.dtype())
    else:
        d_y = cast_tensor(d_y_in, x.dtype(), ctx)
    var d_x_flat = linear_backward_dx(d_y, w, M, in_f, out_f, ctx)
    var d_x = reshape(d_x_flat, x.shape(), ctx)
    if lo:
        var g = klein_lora_bwd_device_resident_tensors_unfused(
            d_y, x, lo.value(), M, ctx
        )
        var d_x_lo = reshape(g.d_x[], x.shape(), ctx)
        d_x = add(d_x, d_x_lo, ctx)
        var pair = Krea2LoraGradT(
            Optional[TArc](g.d_a.copy()),
            Optional[TArc](g.d_b.copy()),
        )
        return _TextLinBwdT(d_x^, pair^)
    return _TextLinBwdT(d_x^, Krea2LoraGradT(None, None))


struct Krea2TextFusionBackwardDeviceGrads(Movable):
    var d_context: TArc
    var layerwise0: Krea2BlockGradsT
    var layerwise1: Krea2BlockGradsT
    var refiner0: Krea2BlockGradsT
    var refiner1: Krea2BlockGradsT

    def __init__(
        out self,
        var d_context: TArc,
        var layerwise0: Krea2BlockGradsT,
        var layerwise1: Krea2BlockGradsT,
        var refiner0: Krea2BlockGradsT,
        var refiner1: Krea2BlockGradsT,
    ):
        self.d_context = d_context^
        self.layerwise0 = layerwise0^
        self.layerwise1 = layerwise1^
        self.refiner0 = refiner0^
        self.refiner1 = refiner1^


struct Krea2TextFusionGradCopyKeepalive(Movable):
    var grads: List[TArc]
    var grad_count: Int

    def __init__(out self, var grads: List[TArc], grad_count: Int):
        self.grads = grads^
        self.grad_count = grad_count


def _text_device_grad_for_adamw_state(
    state: LoraAdamWPlainDeviceState,
    t: TArc,
    mut keepalive: List[TArc],
    ctx: DeviceContext,
) raises -> TArc:
    if t[].dtype() == state.grad_dtype:
        return t.copy()
    # The optimizer buffer owns the grad storage dtype. Krea2/ai-toolkit uses
    # BF16 grads; legacy callers can still request F32 through the state.
    var tg = TArc(cast_tensor(t[], state.grad_dtype, ctx))
    keepalive.append(tg.copy())
    return tg^


def _copy_text_grad_to_adamw_state(
    mut state: LoraAdamWPlainDeviceState,
    adapter_idx: Int,
    g: Krea2LoraGradT,
    mut keepalive: List[TArc],
    ctx: DeviceContext,
) raises:
    if not g.d_a:
        raise Error(
            "krea2_text_fusion_grads_to_adamw_state: missing dA at adapter "
            + String(adapter_idx)
        )
    if not g.d_b:
        raise Error(
            "krea2_text_fusion_grads_to_adamw_state: missing dB at adapter "
            + String(adapter_idx)
        )
    var d_a = _text_device_grad_for_adamw_state(state, g.d_a.value(), keepalive, ctx)
    var d_b = _text_device_grad_for_adamw_state(state, g.d_b.value(), keepalive, ctx)
    lora_adamw_plain_device_state_copy_device_grad_pair(state, adapter_idx, d_a, d_b, ctx)


def _copy_text_block_grads_to_adamw_state(
    bg: Krea2BlockGradsT,
    base: Int,
    mut state: LoraAdamWPlainDeviceState,
    mut keepalive: List[TArc],
    ctx: DeviceContext,
) raises:
    _copy_text_grad_to_adamw_state(state, base + 0, bg.wq, keepalive, ctx)
    _copy_text_grad_to_adamw_state(state, base + 1, bg.wk, keepalive, ctx)
    _copy_text_grad_to_adamw_state(state, base + 2, bg.wv, keepalive, ctx)
    _copy_text_grad_to_adamw_state(state, base + 3, bg.gate_w, keepalive, ctx)
    _copy_text_grad_to_adamw_state(state, base + 4, bg.wo, keepalive, ctx)
    _copy_text_grad_to_adamw_state(state, base + 5, bg.mlp_gate_w, keepalive, ctx)
    _copy_text_grad_to_adamw_state(state, base + 6, bg.mlp_up_w, keepalive, ctx)
    _copy_text_grad_to_adamw_state(state, base + 7, bg.mlp_down_w, keepalive, ctx)


def krea2_text_fusion_grads_to_adamw_state(
    grads: Krea2TextFusionBackwardDeviceGrads,
    base: Int,
    mut state: LoraAdamWPlainDeviceState,
    ctx: DeviceContext,
) raises -> Krea2TextFusionGradCopyKeepalive:
    var keepalive = List[TArc]()
    _copy_text_block_grads_to_adamw_state(grads.layerwise0, base + 0, state, keepalive, ctx)
    _copy_text_block_grads_to_adamw_state(grads.layerwise1, base + 8, state, keepalive, ctx)
    _copy_text_block_grads_to_adamw_state(grads.refiner0, base + 16, state, keepalive, ctx)
    _copy_text_block_grads_to_adamw_state(grads.refiner1, base + 24, state, keepalive, ctx)
    return Krea2TextFusionGradCopyKeepalive(keepalive^, 32)


def krea2_text_fusion_block_lora[
    B: Int, S: Int, HEADS: Int, HEADDIM: Int
](
    x: Tensor,
    w: Krea2TextFusionWeights,
    lora: Krea2BlockLora,
    mask: Optional[Tensor],
    ctx: DeviceContext,
) raises -> Krea2TextFusionBlockForward:
    comptime features = HEADS * HEADDIM
    var M = B * S
    var scale = Float32(1.0) / sqrt(Float32(HEADDIM))

    var xn = krea2_rmsnorm(x, w.prenorm[], Float32(1.0e-5), ctx)
    var q = _text_linear_lora(xn, w.wq[], lora.wq, M, ctx)
    var k = _text_linear_lora(xn, w.wk[], lora.wk, M, ctx)
    var v_lin = _text_linear_lora(xn, w.wv[], lora.wv, M, ctx)
    var gate_pre = _text_linear_lora(xn, w.gate_w[], lora.gate_w, M, ctx)

    var q_pre = reshape_owned(q^, [B, S, HEADS, HEADDIM])
    var k_pre = reshape_owned(k^, [B, S, HEADS, HEADDIM])
    var v = reshape_owned(v_lin^, [B, S, HEADS, HEADDIM])

    var q_norm = krea2_rmsnorm(q_pre, w.qnorm[], Float32(1.0e-5), ctx)
    var k_norm = krea2_rmsnorm(k_pre, w.knorm[], Float32(1.0e-5), ctx)

    var att: Tensor
    var flash_q = Optional[TArc](None)
    var flash_k = Optional[TArc](None)
    var flash_v = Optional[TArc](None)
    var flash_o = Optional[TArc](None)
    var flash_stats = Optional[TArc](None)
    var use_flash = q_norm.dtype() == STDtype.BF16
    # cuDNN flash is the production BF16 training path for supported head sizes.
    # Tiny parity smokes with synthetic head dims stay on the math fallback.
    comptime if not (
        HEADDIM == 64 or HEADDIM == 96 or HEADDIM == 128 or HEADDIM == 256
    ):
        use_flash = False
    if mask:
        use_flash = False
    if use_flash:
        var ff = sdpa_flash_train_fwd_native[B, S, HEADS, HEADDIM](
            q_norm, k_norm, v, scale, ctx
        )
        att = ff.o.clone(ctx)
        flash_q = Optional[TArc](
            TArc(Tensor(ff.q_pad.buf.copy(), ff.q_pad.shape(), ff.q_pad.dtype()))
        )
        flash_k = Optional[TArc](
            TArc(Tensor(ff.k_pad.buf.copy(), ff.k_pad.shape(), ff.k_pad.dtype()))
        )
        flash_v = Optional[TArc](
            TArc(Tensor(ff.v_pad.buf.copy(), ff.v_pad.shape(), ff.v_pad.dtype()))
        )
        flash_o = Optional[TArc](
            TArc(Tensor(ff.o_pad.buf.copy(), ff.o_pad.shape(), ff.o_pad.dtype()))
        )
        flash_stats = Optional[TArc](
            TArc(Tensor(ff.stats.buf.copy(), ff.stats.shape(), ff.stats.dtype()))
        )
    elif mask:
        att = sdpa[B, S, HEADS, HEADDIM](
            q_norm, k_norm, v, mask.value(), scale, ctx
        )
    else:
        att = sdpa_nomask[B, S, HEADS, HEADDIM](q_norm, k_norm, v, scale, ctx)
    var attn_flat = reshape_owned(att^, [B, S, features])
    var sg = sigmoid(gate_pre, ctx)
    var gated = mul(attn_flat, sg, ctx)
    var a = _text_linear_lora(gated, w.wo[], lora.wo, M, ctx)
    var x1 = add(x, a, ctx)

    var xn2 = krea2_rmsnorm(x1, w.postnorm[], Float32(1.0e-5), ctx)
    var mlp_gate = _text_linear_lora(
        xn2, w.mlp_gate[], lora.mlp_gate_w, M, ctx
    )
    var mlp_up = _text_linear_lora(xn2, w.mlp_up[], lora.mlp_up_w, M, ctx)
    var sw = swiglu(mlp_gate, mlp_up, ctx)
    var m = _text_linear_lora(sw, w.mlp_down[], lora.mlp_down_w, M, ctx)
    var out = add(x1, m, ctx)

    var saved = Krea2TextFusionBlockSaved(
        TArc(x.clone(ctx)), TArc(xn^),
        TArc(q_pre^), TArc(k_pre^), TArc(v^),
        TArc(q_norm^), TArc(k_norm^),
        TArc(attn_flat^), TArc(gate_pre^), TArc(sg^),
        TArc(gated^), TArc(a^), TArc(x1^), TArc(xn2^),
        TArc(mlp_gate^), TArc(mlp_up^), TArc(sw^), TArc(m^),
        flash_q^, flash_k^, flash_v^, flash_o^, flash_stats^,
    )
    return Krea2TextFusionBlockForward(TArc(out^), saved^)


def krea2_text_fusion_block_lora_backward_dev[
    B: Int, S: Int, HEADS: Int, HEADDIM: Int
](
    d_out_in: Tensor,
    w: Krea2TextFusionWeights,
    lora: Krea2BlockLora,
    saved: Krea2TextFusionBlockSaved,
    mask: Optional[Tensor],
    ctx: DeviceContext,
) raises -> Krea2BlockGradsT:
    comptime features = HEADS * HEADDIM
    var M = B * S
    var mlpdim = saved.mlp_gate[].shape()[2]
    var scale = Float32(1.0) / sqrt(Float32(HEADDIM))
    var d_out: Tensor
    if d_out_in.dtype() == saved.x[].dtype():
        d_out = Tensor(d_out_in.buf.copy(), d_out_in.shape(), d_out_in.dtype())
    else:
        d_out = cast_tensor(d_out_in, saved.x[].dtype(), ctx)

    # out = x1 + m
    var bw_down = _text_linear_bwd_dx_dev(
        d_out, saved.sw[], w.mlp_down[], lora.mlp_down_w,
        M, mlpdim, features, ctx,
    )
    var d_sw = bw_down.d_x.clone(ctx)
    var g_down = bw_down.lora.copy()

    var sg = swiglu_backward(d_sw, saved.mlp_gate[], saved.mlp_up[], ctx)
    var bw_mg = _text_linear_bwd_dx_dev(
        sg.d_gate, saved.xn2[], w.mlp_gate[], lora.mlp_gate_w,
        M, features, mlpdim, ctx,
    )
    var bw_mu = _text_linear_bwd_dx_dev(
        sg.d_up, saved.xn2[], w.mlp_up[], lora.mlp_up_w,
        M, features, mlpdim, ctx,
    )
    var g_mg = bw_mg.lora.copy()
    var g_mu = bw_mu.lora.copy()
    var d_xn2 = add(bw_mg.d_x, bw_mu.d_x, ctx)
    var d_x1_mlp = krea2_rmsnorm_backward_dx(
        d_xn2, saved.x1[],
        w.postnorm[],
        Float32(1.0e-5), ctx,
    )
    var d_x1 = add(d_out, d_x1_mlp, ctx)

    # x1 = x + a
    var bw_wo = _text_linear_bwd_dx_dev(
        d_x1, saved.gated[], w.wo[], lora.wo, M, features, features, ctx
    )
    var d_gated = bw_wo.d_x.clone(ctx)
    var g_wo = bw_wo.lora.copy()
    var d_attn_flat = mul(d_gated, saved.sg[], ctx)
    var d_sg = mul(d_gated, saved.attn_flat[], ctx)
    # ai-toolkit/PyTorch differentiates from the saved BF16 sigmoid output.
    var d_gate_pre = sigmoid_backward_from_output(d_sg, saved.sg[], ctx)

    var d_att = reshape(d_attn_flat, [B, S, HEADS, HEADDIM], ctx)
    var sb_dq: Tensor
    var sb_dk: Tensor
    var sb_dv: Tensor
    var bwd_use_flash = saved.flash_stats
    if bwd_use_flash:
        if (
            not saved.flash_q
            or not saved.flash_k
            or not saved.flash_v
            or not saved.flash_o
        ):
            raise Error("krea2 txtfusion bwd: incomplete saved flash tape")
        var ff = SdpaFlashFwd(
            Tensor(
                saved.flash_o.value()[].buf.copy(),
                saved.flash_o.value()[].shape(),
                saved.flash_o.value()[].dtype(),
            ),
            Tensor(
                saved.flash_o.value()[].buf.copy(),
                saved.flash_o.value()[].shape(),
                saved.flash_o.value()[].dtype(),
            ),
            Tensor(
                saved.flash_q.value()[].buf.copy(),
                saved.flash_q.value()[].shape(),
                saved.flash_q.value()[].dtype(),
            ),
            Tensor(
                saved.flash_k.value()[].buf.copy(),
                saved.flash_k.value()[].shape(),
                saved.flash_k.value()[].dtype(),
            ),
            Tensor(
                saved.flash_v.value()[].buf.copy(),
                saved.flash_v.value()[].shape(),
                saved.flash_v.value()[].dtype(),
            ),
            Tensor(
                saved.flash_stats.value()[].buf.copy(),
                saved.flash_stats.value()[].shape(),
                saved.flash_stats.value()[].dtype(),
            ),
        )
        var fb = sdpa_flash_backward_native[B, S, HEADS, HEADDIM](ff, d_att, scale, ctx)
        sb_dq = Tensor(fb.d_q.buf.copy(), fb.d_q.shape(), fb.d_q.dtype())
        sb_dk = Tensor(fb.d_k.buf.copy(), fb.d_k.shape(), fb.d_k.dtype())
        sb_dv = Tensor(fb.d_v.buf.copy(), fb.d_v.shape(), fb.d_v.dtype())
    elif mask:
        var m32: Tensor
        if mask.value().dtype() == STDtype.F32:
            m32 = Tensor(mask.value().buf.copy(), mask.value().shape(), mask.value().dtype())
        else:
            m32 = cast_tensor(mask.value(), STDtype.F32, ctx)
        var sb = sdpa_backward_masked[B, S, HEADS, HEADDIM](
            saved.q_norm[], saved.k_norm[], saved.v[], m32, d_att, scale, ctx
        )
        sb_dq = Tensor(sb.d_q.buf.copy(), sb.d_q.shape(), sb.d_q.dtype())
        sb_dk = Tensor(sb.d_k.buf.copy(), sb.d_k.shape(), sb.d_k.dtype())
        sb_dv = Tensor(sb.d_v.buf.copy(), sb.d_v.shape(), sb.d_v.dtype())
    else:
        var sb = sdpa_backward[B, S, HEADS, HEADDIM](
            saved.q_norm[], saved.k_norm[], saved.v[], d_att, scale, ctx
        )
        sb_dq = Tensor(sb.d_q.buf.copy(), sb.d_q.shape(), sb.d_q.dtype())
        sb_dk = Tensor(sb.d_k.buf.copy(), sb.d_k.shape(), sb.d_k.dtype())
        sb_dv = Tensor(sb.d_v.buf.copy(), sb.d_v.shape(), sb.d_v.dtype())

    var d_q_pre = krea2_rmsnorm_backward_dx(
        sb_dq, saved.q_pre[],
        w.qnorm[],
        Float32(1.0e-5), ctx,
    )
    var d_k_pre = krea2_rmsnorm_backward_dx(
        sb_dk, saved.k_pre[],
        w.knorm[],
        Float32(1.0e-5), ctx,
    )
    var d_q = reshape(d_q_pre, [B, S, features], ctx)
    var d_k = reshape(d_k_pre, [B, S, features], ctx)
    var d_v = reshape(sb_dv, [B, S, features], ctx)

    var bw_q = _text_linear_bwd_dx_dev(
        d_q, saved.xn[], w.wq[], lora.wq, M, features, features, ctx
    )
    var bw_k = _text_linear_bwd_dx_dev(
        d_k, saved.xn[], w.wk[], lora.wk, M, features, features, ctx
    )
    var bw_v = _text_linear_bwd_dx_dev(
        d_v, saved.xn[], w.wv[], lora.wv, M, features, features, ctx
    )
    var bw_g = _text_linear_bwd_dx_dev(
        d_gate_pre, saved.xn[], w.gate_w[], lora.gate_w, M, features, features, ctx
    )
    var g_wq = bw_q.lora.copy()
    var g_wk = bw_k.lora.copy()
    var g_wv = bw_v.lora.copy()
    var g_gate = bw_g.lora.copy()
    var d_xn = add(add(bw_q.d_x, bw_k.d_x, ctx), add(bw_v.d_x, bw_g.d_x, ctx), ctx)
    var d_x_attn = krea2_rmsnorm_backward_dx(
        d_xn, saved.x[],
        w.prenorm[],
        Float32(1.0e-5), ctx,
    )
    var d_x = add(d_x1, d_x_attn, ctx)

    return Krea2BlockGradsT(
        TArc(d_x^),
        g_wq^, g_wk^, g_wv^, g_gate^, g_wo^, g_mg^, g_mu^, g_down^,
    )


def krea2_text_fusion_lora_forward[
    LT: Int, NLAYERS: Int, HEADS: Int, HEADDIM: Int
](
    context: Tensor,
    layerwise0: Krea2TextFusionWeights,
    layerwise1: Krea2TextFusionWeights,
    projector_w: Tensor,
    refiner0: Krea2TextFusionWeights,
    refiner1: Krea2TextFusionWeights,
    lora: Krea2TextFusionLora,
    refiner_mask: Optional[Tensor],
    ctx: DeviceContext,
) raises -> Krea2TextFusionForward:
    var cshape = context.shape()
    var txtdim = cshape[len(cshape) - 1]

    var x = reshape(context, [LT, NLAYERS, txtdim], ctx)
    var lw0 = krea2_text_fusion_block_lora[LT, NLAYERS, HEADS, HEADDIM](
        x, layerwise0, lora.layerwise0, None, ctx
    )
    var lw1 = krea2_text_fusion_block_lora[LT, NLAYERS, HEADS, HEADDIM](
        lw0.out[], layerwise1, lora.layerwise1, None, ctx
    )

    var xt = transpose(lw1.out[], 1, 2, ctx)
    var nb = _no_bias()
    var proj = linear(xt, projector_w, nb^, ctx)
    var xr = reshape(proj, [1, LT, txtdim], ctx)

    var rf0 = krea2_text_fusion_block_lora[1, LT, HEADS, HEADDIM](
        xr, refiner0, lora.refiner0, refiner_mask, ctx
    )
    var rf1 = krea2_text_fusion_block_lora[1, LT, HEADS, HEADDIM](
        rf0.out[], refiner1, lora.refiner1, refiner_mask, ctx
    )

    return Krea2TextFusionForward(
        rf1.out.copy(),
        lw0^, lw1^,
        TArc(lw1.out[].clone(ctx)), TArc(xt^), TArc(proj^),
        rf0^, rf1^,
    )


def krea2_text_fusion_lora_backward_dev[
    LT: Int, NLAYERS: Int, HEADS: Int, HEADDIM: Int
](
    d_out: Tensor,
    fwd: Krea2TextFusionForward,
    layerwise0: Krea2TextFusionWeights,
    layerwise1: Krea2TextFusionWeights,
    projector_w: Tensor,
    refiner0: Krea2TextFusionWeights,
    refiner1: Krea2TextFusionWeights,
    lora: Krea2TextFusionLora,
    refiner_mask: Optional[Tensor],
    ctx: DeviceContext,
) raises -> Krea2TextFusionBackwardDeviceGrads:
    comptime txtdim = HEADS * HEADDIM
    var rf1 = krea2_text_fusion_block_lora_backward_dev[1, LT, HEADS, HEADDIM](
        d_out, refiner1, lora.refiner1, fwd.refiner1.saved, refiner_mask, ctx
    )
    var rf0 = krea2_text_fusion_block_lora_backward_dev[1, LT, HEADS, HEADDIM](
        rf1.d_x[], refiner0, lora.refiner0, fwd.refiner0.saved, refiner_mask, ctx
    )

    var d_proj = reshape(rf0.d_x[], [LT, txtdim, 1], ctx)
    var d_xt = linear_backward_dx(
        d_proj, projector_w, LT * txtdim, NLAYERS, 1, ctx
    )
    var d_xt3 = reshape(d_xt, [LT, txtdim, NLAYERS], ctx)
    var d_lw1_out = transpose(d_xt3, 1, 2, ctx)

    var lw1 = krea2_text_fusion_block_lora_backward_dev[
        LT, NLAYERS, HEADS, HEADDIM
    ](
        d_lw1_out, layerwise1, lora.layerwise1, fwd.layerwise1.saved, None, ctx
    )
    var lw0 = krea2_text_fusion_block_lora_backward_dev[
        LT, NLAYERS, HEADS, HEADDIM
    ](
        lw1.d_x[], layerwise0, lora.layerwise0, fwd.layerwise0.saved, None, ctx
    )
    var d_context = reshape(lw0.d_x[], [1, LT, NLAYERS, txtdim], ctx)
    return Krea2TextFusionBackwardDeviceGrads(
        TArc(d_context^), lw0^, lw1^, rf0^, rf1^
    )
