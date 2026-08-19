# MiniMax-H3 token-refiner TRAINING twin.
#
# AI Toolkit targets qkv/out/fc1/fc2 in both token_refiner blocks in addition
# to the same four projections in all 50 main blocks (208 adapters total).
# The product frontend stores qkv as contiguous q|k|v and FC1 as
# [value|gate]. Canonical PEFT FC1 B remains [gate|value], so the trainer
# transforms the BF16 compute copy on entry and transforms d_B back before
# the F32 master/optimizer boundary.
from std.collections import Dict, List
from std.math import sqrt
from std.memory import ArcPointer
from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.ops.activations import swiglu
from serenitymojo.ops.attention_flash import (
    SdpaFlashFwd, sdpa_flash_backward_dynamic,
    sdpa_flash_train_fwd_dynamic,
)
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linalg_backward import linear_backward_dx
from serenitymojo.ops.linear import linear
from serenitymojo.ops.loss_swiglu_backward import swiglu_backward
from serenitymojo.ops.norm_backward import rms_norm_backward_dx
from serenitymojo.ops.tensor_algebra import add, concat, reshape, slice
from serenitymojo.ops.vec_rms_norm import vec_rms_norm
from serenitymojo.models.klein.lora_block import (
    KleinLoraDeviceGradTensors, LoraAdapterDevice,
    klein_lora_bwd_device_resident_tensors,
    klein_lora_fwd_device_resident,
)
from serenitymojo.models.minimax_h3.h3_block_train import (
    H3BlockLoraDevice, H3BlockLoraGrads, H3BlockTrainWeights,
)

comptime TArc = ArcPointer[Tensor]
comptime H3_TOKEN_REFINER_BLOCKS = 2


struct H3TokenRefinerTrainWeights(Copyable, Movable):
    var blocks: List[H3BlockTrainWeights]
    var final_norm: TArc

    def __init__(
        out self, var blocks: List[H3BlockTrainWeights], var final_norm: TArc
    ):
        self.blocks = blocks^
        self.final_norm = final_norm^


def h3_token_refiner_train_weights(
    w: Dict[String, ArcPointer[Tensor]], layers: Int
) raises -> H3TokenRefinerTrainWeights:
    if layers != H3_TOKEN_REFINER_BLOCKS:
        raise Error("H3 token-refiner trainer requires exactly two blocks")
    var blocks = List[H3BlockTrainWeights]()
    for layer in range(layers):
        var p = String("token_refiner.blocks.") + String(layer)
        blocks.append(H3BlockTrainWeights(
            w[p + ".attn.qkv_proj.weight"].copy(),
            w[p + ".attn.out_proj.weight"].copy(),
            w[p + ".mlp.fc1.weight"].copy(),
            w[p + ".mlp.fc2.weight"].copy(),
            w[p + ".attn.q_norm.weight"].copy(),
            w[p + ".attn.k_norm.weight"].copy(),
            w[p + ".norm1.weight"].copy(),
            w[p + ".norm2.weight"].copy(),
        ))
    return H3TokenRefinerTrainWeights(
        blocks^, w[String("token_refiner.final_norm.weight")].copy()
    )


def h3_token_refiner_swap_fc1_rows(
    t: Tensor, ffn: Int, ctx: DeviceContext
) raises -> Tensor:
    """Swap [gate|value] <-> [value|gate]; the operation is self-inverse."""
    var sh = t.shape()
    if len(sh) != 2 or sh[0] != 2 * ffn:
        raise Error("H3 token-refiner FC1 LoRA B has the wrong shape")
    return concat(
        0, ctx,
        slice(t, 0, ffn, ffn, ctx),
        slice(t, 0, 0, ffn, ctx),
    )


def _projection_lora(
    x: Tensor, w: Tensor, lo: Optional[LoraAdapterDevice], rows: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var base = linear(x, w, None, ctx)
    if not lo:
        return base^
    return add(
        base, klein_lora_fwd_device_resident(x, lo.value(), rows, ctx), ctx
    )


def _dx_bf16(
    grad: Tensor, w: Tensor, rows: Int, in_f: Int, out_f: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var dx = linear_backward_dx(grad, w, rows, in_f, out_f, ctx)
    if dx.dtype() == STDtype.BF16:
        return dx^
    return cast_tensor(dx, STDtype.BF16, ctx)


struct H3TokenRefinerBlockSaved(Copyable, Movable):
    var x: TArc
    var n1: TArc
    var q_pre: TArc
    var k_pre: TArc
    var v: TArc
    var q: TArc
    var k: TArc
    var att_flat: TArc
    var h_mid: TArc
    var n2: TArc
    var value: TArc
    var gate: TArc
    var swi: TArc
    var flash_q_pad: TArc
    var flash_k_pad: TArc
    var flash_v_pad: TArc
    var flash_o_pad: TArc
    var flash_stats: TArc

    def __init__(
        out self,
        var x: TArc, var n1: TArc,
        var q_pre: TArc, var k_pre: TArc, var v: TArc,
        var q: TArc, var k: TArc, var att_flat: TArc,
        var h_mid: TArc, var n2: TArc,
        var value: TArc, var gate: TArc, var swi: TArc,
        var flash_q_pad: TArc, var flash_k_pad: TArc,
        var flash_v_pad: TArc, var flash_o_pad: TArc,
        var flash_stats: TArc,
    ):
        self.x = x^
        self.n1 = n1^
        self.q_pre = q_pre^
        self.k_pre = k_pre^
        self.v = v^
        self.q = q^
        self.k = k^
        self.att_flat = att_flat^
        self.h_mid = h_mid^
        self.n2 = n2^
        self.value = value^
        self.gate = gate^
        self.swi = swi^
        self.flash_q_pad = flash_q_pad^
        self.flash_k_pad = flash_k_pad^
        self.flash_v_pad = flash_v_pad^
        self.flash_o_pad = flash_o_pad^
        self.flash_stats = flash_stats^


struct H3TokenRefinerBlockForward(Copyable, Movable):
    var out: TArc
    var saved: H3TokenRefinerBlockSaved

    def __init__(
        out self, var out: TArc, var saved: H3TokenRefinerBlockSaved
    ):
        self.out = out^
        self.saved = saved^


def h3_token_refiner_block_forward_lora[H: Int, Dh: Int](
    x_in: Tensor, w: H3BlockTrainWeights, lora: H3BlockLoraDevice,
    hidden: Int, ffn: Int, eps: Float32, qk_eps: Float32,
    ctx: DeviceContext,
) raises -> H3TokenRefinerBlockForward:
    var rows = x_in.shape()[0]
    comptime inner = H * Dh
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var x = x_in.clone(ctx)
    var n1 = vec_rms_norm(x, w.norm1_w[], eps, ctx)
    var qkv = _projection_lora(n1, w.qkv_w[], lora.qkv, rows, ctx)
    var q_pre = reshape(slice(qkv, 1, 0, inner, ctx), [1, rows, H, Dh], ctx)
    var k_pre = reshape(slice(qkv, 1, inner, inner, ctx), [1, rows, H, Dh], ctx)
    var v = reshape(slice(qkv, 1, 2 * inner, inner, ctx), [1, rows, H, Dh], ctx)
    var q = vec_rms_norm(q_pre, w.q_norm[], qk_eps, ctx)
    var k = vec_rms_norm(k_pre, w.k_norm[], qk_eps, ctx)
    var af = sdpa_flash_train_fwd_dynamic(q, k, v, scale, ctx)
    var att_flat = reshape(af.o, [rows, inner], ctx)
    var attn_y = _projection_lora(att_flat, w.out_w[], lora.out, rows, ctx)
    var h_mid = add(x, attn_y, ctx)
    var n2 = vec_rms_norm(h_mid, w.norm2_w[], eps, ctx)
    var fc1 = _projection_lora(n2, w.fc1_w[], lora.fc1, rows, ctx)
    var value = slice(fc1, 1, 0, ffn, ctx)
    var gate = slice(fc1, 1, ffn, ffn, ctx)
    var swi = swiglu(gate, value, ctx)
    var ff_y = _projection_lora(swi, w.fc2_w[], lora.fc2, rows, ctx)
    var out = add(h_mid, ff_y, ctx)
    var saved = H3TokenRefinerBlockSaved(
        TArc(x^), TArc(n1^), TArc(q_pre^), TArc(k_pre^), TArc(v^),
        TArc(q^), TArc(k^), TArc(att_flat^), TArc(h_mid^), TArc(n2^),
        TArc(value^), TArc(gate^), TArc(swi^),
        TArc(Tensor(af.q_pad.buf.copy(), af.q_pad.shape(), af.q_pad.dtype())),
        TArc(Tensor(af.k_pad.buf.copy(), af.k_pad.shape(), af.k_pad.dtype())),
        TArc(Tensor(af.v_pad.buf.copy(), af.v_pad.shape(), af.v_pad.dtype())),
        TArc(Tensor(af.o_pad.buf.copy(), af.o_pad.shape(), af.o_pad.dtype())),
        TArc(Tensor(af.stats.buf.copy(), af.stats.shape(), af.stats.dtype())),
    )
    _ = af^
    return H3TokenRefinerBlockForward(TArc(out^), saved^)


struct H3TokenRefinerBlockBackward(Copyable, Movable):
    var d_x: TArc
    var lora: H3BlockLoraGrads

    def __init__(
        out self, var d_x: TArc, var lora: H3BlockLoraGrads
    ):
        self.d_x = d_x^
        self.lora = lora^


def h3_token_refiner_block_backward_lora[H: Int, Dh: Int](
    d_out: Tensor, w: H3BlockTrainWeights, lora: H3BlockLoraDevice,
    saved: H3TokenRefinerBlockSaved,
    hidden: Int, ffn: Int, eps: Float32, qk_eps: Float32,
    ctx: DeviceContext,
) raises -> H3TokenRefinerBlockBackward:
    var rows = d_out.shape()[0]
    comptime inner = H * Dh
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var d_h_mid = d_out.clone(ctx)
    var d_swi = _dx_bf16(d_out, w.fc2_w[], rows, ffn, hidden, ctx)
    var lg_fc2 = Optional[KleinLoraDeviceGradTensors](None)
    if lora.fc2:
        var g = klein_lora_bwd_device_resident_tensors(
            d_out, saved.swi[], lora.fc2.value(), rows, ctx
        )
        d_swi = add(d_swi, g.d_x[], ctx)
        lg_fc2 = Optional[KleinLoraDeviceGradTensors](g^)

    var sgb = swiglu_backward(d_swi, saved.gate[], saved.value[], ctx)
    # Runtime FC1 is [value|gate]; canonical PEFT is [gate|value].
    var d_fc1 = concat(1, ctx, sgb.d_up, sgb.d_gate)
    var d_n2 = _dx_bf16(d_fc1, w.fc1_w[], rows, hidden, 2 * ffn, ctx)
    var lg_fc1 = Optional[KleinLoraDeviceGradTensors](None)
    if lora.fc1:
        var g = klein_lora_bwd_device_resident_tensors(
            d_fc1, saved.n2[], lora.fc1.value(), rows, ctx
        )
        d_n2 = add(d_n2, g.d_x[], ctx)
        var canonical_db = h3_token_refiner_swap_fc1_rows(g.d_b[], ffn, ctx)
        lg_fc1 = Optional[KleinLoraDeviceGradTensors](
            KleinLoraDeviceGradTensors(
                g.d_a.copy(), TArc(canonical_db^), g.d_x.copy()
            )
        )
    var d_n2x = rms_norm_backward_dx(
        d_n2, saved.h_mid[], w.norm2_w[], eps, ctx
    )
    d_h_mid = add(d_h_mid, d_n2x, ctx)

    var d_x = d_h_mid.clone(ctx)
    var d_att = _dx_bf16(
        d_h_mid, w.out_w[], rows, inner, hidden, ctx
    )
    var lg_out = Optional[KleinLoraDeviceGradTensors](None)
    if lora.out:
        var g = klein_lora_bwd_device_resident_tensors(
            d_h_mid, saved.att_flat[], lora.out.value(), rows, ctx
        )
        d_att = add(d_att, g.d_x[], ctx)
        lg_out = Optional[KleinLoraDeviceGradTensors](g^)
    var d_att4 = reshape(d_att, [1, rows, H, Dh], ctx)
    var flash = SdpaFlashFwd(
        Tensor(saved.att_flat[].buf.copy(), [1, rows, H, Dh], saved.att_flat[].dtype()),
        Tensor(saved.flash_o_pad[].buf.copy(), saved.flash_o_pad[].shape(), saved.flash_o_pad[].dtype()),
        Tensor(saved.flash_q_pad[].buf.copy(), saved.flash_q_pad[].shape(), saved.flash_q_pad[].dtype()),
        Tensor(saved.flash_k_pad[].buf.copy(), saved.flash_k_pad[].shape(), saved.flash_k_pad[].dtype()),
        Tensor(saved.flash_v_pad[].buf.copy(), saved.flash_v_pad[].shape(), saved.flash_v_pad[].dtype()),
        Tensor(saved.flash_stats[].buf.copy(), saved.flash_stats[].shape(), saved.flash_stats[].dtype()),
    )
    var sb = sdpa_flash_backward_dynamic(flash, d_att4, scale, ctx)
    _ = flash^
    var d_q_pre = rms_norm_backward_dx(
        sb.d_q, saved.q_pre[], w.q_norm[], qk_eps, ctx
    )
    var d_k_pre = rms_norm_backward_dx(
        sb.d_k, saved.k_pre[], w.k_norm[], qk_eps, ctx
    )
    var d_qkv = concat(
        1, ctx,
        reshape(d_q_pre, [rows, inner], ctx),
        reshape(d_k_pre, [rows, inner], ctx),
        reshape(sb.d_v, [rows, inner], ctx),
    )
    var d_n1 = _dx_bf16(
        d_qkv, w.qkv_w[], rows, hidden, 3 * inner, ctx
    )
    var lg_qkv = Optional[KleinLoraDeviceGradTensors](None)
    if lora.qkv:
        var g = klein_lora_bwd_device_resident_tensors(
            d_qkv, saved.n1[], lora.qkv.value(), rows, ctx
        )
        d_n1 = add(d_n1, g.d_x[], ctx)
        lg_qkv = Optional[KleinLoraDeviceGradTensors](g^)
    var d_n1x = rms_norm_backward_dx(
        d_n1, saved.x[], w.norm1_w[], eps, ctx
    )
    d_x = add(d_x, d_n1x, ctx)
    return H3TokenRefinerBlockBackward(
        TArc(d_x^),
        H3BlockLoraGrads(lg_qkv^, lg_out^, lg_fc1^, lg_fc2^),
    )


struct H3TokenRefinerTrainForward(Copyable, Movable):
    var out: TArc
    var pre_final: TArc
    var blocks: List[H3TokenRefinerBlockForward]

    def __init__(
        out self, var out: TArc, var pre_final: TArc,
        var blocks: List[H3TokenRefinerBlockForward],
    ):
        self.out = out^
        self.pre_final = pre_final^
        self.blocks = blocks^


def h3_token_refiner_train_forward[H: Int, Dh: Int](
    x: Tensor, weights: H3TokenRefinerTrainWeights,
    loras: List[H3BlockLoraDevice], hidden: Int, ffn: Int,
    eps: Float32, qk_eps: Float32, final_eps: Float32,
    ctx: DeviceContext,
) raises -> H3TokenRefinerTrainForward:
    if len(weights.blocks) != H3_TOKEN_REFINER_BLOCKS or len(loras) != H3_TOKEN_REFINER_BLOCKS:
        raise Error("H3 token-refiner forward requires two blocks/LoRA groups")
    var state = x.clone(ctx)
    var blocks = List[H3TokenRefinerBlockForward]()
    for layer in range(H3_TOKEN_REFINER_BLOCKS):
        var f = h3_token_refiner_block_forward_lora[H, Dh](
            state, weights.blocks[layer], loras[layer], hidden, ffn,
            eps, qk_eps, ctx,
        )
        state = f.out[].clone(ctx)
        blocks.append(f^)
    var pre_final = state.clone(ctx)
    var out = vec_rms_norm(state, weights.final_norm[], final_eps, ctx)
    return H3TokenRefinerTrainForward(
        TArc(out^), TArc(pre_final^), blocks^
    )


struct H3TokenRefinerTrainBackward(Copyable, Movable):
    var d_x: TArc
    var lora: List[H3BlockLoraGrads]

    def __init__(
        out self, var d_x: TArc, var lora: List[H3BlockLoraGrads]
    ):
        self.d_x = d_x^
        self.lora = lora^


def h3_token_refiner_train_backward[H: Int, Dh: Int](
    d_out: Tensor, fwd: H3TokenRefinerTrainForward,
    weights: H3TokenRefinerTrainWeights,
    loras: List[H3BlockLoraDevice], hidden: Int, ffn: Int,
    eps: Float32, qk_eps: Float32, final_eps: Float32,
    ctx: DeviceContext,
) raises -> H3TokenRefinerTrainBackward:
    var d = rms_norm_backward_dx(
        d_out, fwd.pre_final[], weights.final_norm[], final_eps, ctx
    )
    var rev = List[H3BlockLoraGrads]()
    for offset in range(H3_TOKEN_REFINER_BLOCKS):
        var layer = H3_TOKEN_REFINER_BLOCKS - 1 - offset
        var b = h3_token_refiner_block_backward_lora[H, Dh](
            d, weights.blocks[layer], loras[layer], fwd.blocks[layer].saved,
            hidden, ffn, eps, qk_eps, ctx,
        )
        d = b.d_x[].clone(ctx)
        rev.append(b.lora.copy())
    var grads = List[H3BlockLoraGrads]()
    for layer in range(H3_TOKEN_REFINER_BLOCKS):
        grads.append(rev[H3_TOKEN_REFINER_BLOCKS - 1 - layer].copy())
    return H3TokenRefinerTrainBackward(TArc(d^), grads^)
