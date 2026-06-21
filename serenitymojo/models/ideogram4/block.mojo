# Ideogram4LoRABlock.mojo — hand-chained Ideogram4 block/stack LoRA training.
#
# This is the trainable 34-layer transformer core:
#   block forward saves activations -> block backward returns LoRA dA/dB
#   stack forward checkpoints block inputs -> stack backward recomputes blocks
#   and walks deepest-to-shallowest.
#
# LoRA targets per Ideogram4 layer, in ideogram4LoraTargets order:
#   attention.qkv, attention.o, feed_forward.w1, feed_forward.w2,
#   feed_forward.w3, adaln_modulation
#
# The full model globals are still handled by the outer trainer. This file owns
# the repeated layer stack, which is the heavy part needed for real training.

from std.math import sqrt
from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors

from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.activations import swiglu
from serenitymojo.ops.unary import tanh_op
from serenitymojo.ops.activation_backward import tanh_backward
from serenitymojo.ops.tensor_algebra import (
    add,
    add_scalar,
    concat,
    mul,
    mul_scalar,
    reshape,
    slice,
    zeros_device,
)
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.fp8 import load_fp8_dequant, fp8_e4m3_dequant_perrow_to_bf16

from serenitymojo.ops.linalg_backward import linear_backward, linear_backward_dx
from serenitymojo.ops.norm_backward import rms_norm_backward_dx
from serenitymojo.ops.attention_backward import sdpa_backward
from serenitymojo.ops.attention_flash import (
    sdpa_flash_train_fwd_f32,
    sdpa_flash_backward_f32,
)

# C13 gate-don't-delete: cuDNN flash SDPA replaces the custom Mojo SDPA
# (sdpa_nomask/sdpa_backward) for the attention fwd+bwd. Approved numerics
# change (precedent: klein KLEIN_SDPA_FLASH, zimage ZIMAGE_SDPA_FLASH). Flash
# bwd dQ is nondeterministic run-to-run -> trainer gates are 4dp value-class,
# NOT bit. Requires S % 128 == 0 (giger 512 cache: S=NT256+NIMG1024=1280=10x128),
# Build needs the cshim link args + cuDNN on LD_LIBRARY_PATH
# (see docs/IDEOGRAM4_FLASH_SDPA_WIRING_2026-06-20.md).
#
# GATED OFF — MEASURED BLOCKER 2026-06-20: cuDNN flash BACKWARD supports only
# head_dim ∈ {64,96,128} (serenitymojo/ops/cshim/cudnn_sdpa_bwd.cpp:5). The flash
# forward accepts ideogram4's Dh=256 but the backward g->build() finds no plan
# (shim rc=-1, B=1 S=1280 H=18 Dh=256). klein/zimage use Dh=128 → flash works;
# ideogram4 Dh=256 does not. Wiring kept compiled-but-off (C13) for a future
# D=256-capable flash backward (cuDNN upgrade / frontend HeurMode fallback).
comptime IDEOGRAM4_SDPA_FLASH = False
from serenitymojo.ops.loss_swiglu_backward import swiglu_backward
from serenitymojo.ops.shape_backward import broadcast_backward
from serenitymojo.ops.rope_struct_backward import rope_halfsplit_full_backward
from serenitymojo.models.dit.ideogram4_dit import apply_rope_ideogram
from serenitymojo.models.dit.ideogram4_resident import Ideogram4Weights

from serenitymojo.models.ideogram4.lora_module import LoraAdapter, make_lora_adapter
from serenitymojo.models.ideogram4.config import (
    IDEOGRAM4_ADALN_DIM,
    IDEOGRAM4_HEAD_DIM,
    IDEOGRAM4_HIDDEN,
    IDEOGRAM4_INTERMEDIATE_SIZE,
    IDEOGRAM4_NUM_HEADS,
    IDEOGRAM4_NUM_LAYERS,
)


comptime TArc = ArcPointer[Tensor]
comptime LArc = ArcPointer[LoraAdapter]

comptime I4_SLOT_QKV = 0
comptime I4_SLOT_O = 1
comptime I4_SLOT_W1 = 2
comptime I4_SLOT_W2 = 3
comptime I4_SLOT_W3 = 4
comptime I4_SLOT_ADALN = 5
comptime I4_SLOTS_PER_BLOCK = 6
comptime I4_EPS = Float32(1.0e-5)


struct Ideogram4BlockWeights(Movable):
    var adaln_w: Tensor
    var adaln_b: Tensor
    var attn_norm1: Tensor
    var attn_norm2: Tensor
    var ffn_norm1: Tensor
    var ffn_norm2: Tensor
    var qkv_w: Tensor
    var o_w: Tensor
    var norm_q: Tensor
    var norm_k: Tensor
    var w1: Tensor
    var w2: Tensor
    var w3: Tensor

    def __init__(
        out self,
        var adaln_w: Tensor,
        var adaln_b: Tensor,
        var attn_norm1: Tensor,
        var attn_norm2: Tensor,
        var ffn_norm1: Tensor,
        var ffn_norm2: Tensor,
        var qkv_w: Tensor,
        var o_w: Tensor,
        var norm_q: Tensor,
        var norm_k: Tensor,
        var w1: Tensor,
        var w2: Tensor,
        var w3: Tensor,
    ):
        self.adaln_w = adaln_w^
        self.adaln_b = adaln_b^
        self.attn_norm1 = attn_norm1^
        self.attn_norm2 = attn_norm2^
        self.ffn_norm1 = ffn_norm1^
        self.ffn_norm2 = ffn_norm2^
        self.qkv_w = qkv_w^
        self.o_w = o_w^
        self.norm_q = norm_q^
        self.norm_k = norm_k^
        self.w1 = w1^
        self.w2 = w2^
        self.w3 = w3^


def load_ideogram4_block_weights(
    st: ShardedSafeTensors, layer: Int, ctx: DeviceContext
) raises -> Ideogram4BlockWeights:
    var p = String("layers.") + String(layer) + String(".")
    return Ideogram4BlockWeights(
        load_fp8_dequant(st, p + String("adaln_modulation.weight"), ctx),
        Tensor.from_view(st.tensor_view(p + String("adaln_modulation.bias")), ctx),
        Tensor.from_view(st.tensor_view(p + String("attention_norm1.weight")), ctx),
        Tensor.from_view(st.tensor_view(p + String("attention_norm2.weight")), ctx),
        Tensor.from_view(st.tensor_view(p + String("ffn_norm1.weight")), ctx),
        Tensor.from_view(st.tensor_view(p + String("ffn_norm2.weight")), ctx),
        load_fp8_dequant(st, p + String("attention.qkv.weight"), ctx),
        load_fp8_dequant(st, p + String("attention.o.weight"), ctx),
        Tensor.from_view(st.tensor_view(p + String("attention.norm_q.weight")), ctx),
        Tensor.from_view(st.tensor_view(p + String("attention.norm_k.weight")), ctx),
        load_fp8_dequant(st, p + String("feed_forward.w1.weight"), ctx),
        load_fp8_dequant(st, p + String("feed_forward.w2.weight"), ctx),
        load_fp8_dequant(st, p + String("feed_forward.w3.weight"), ctx),
    )


def load_ideogram4_block_weights_resident(
    rw: Ideogram4Weights, layer: Int, ctx: DeviceContext
) raises -> Ideogram4BlockWeights:
    var p = String("layers.") + String(layer) + String(".")
    return Ideogram4BlockWeights(
        _resident_fp8_weight(rw, p + String("adaln_modulation.weight"), ctx),
        rw.w(p + String("adaln_modulation.bias")).clone(ctx),
        rw.w(p + String("attention_norm1.weight")).clone(ctx),
        rw.w(p + String("attention_norm2.weight")).clone(ctx),
        rw.w(p + String("ffn_norm1.weight")).clone(ctx),
        rw.w(p + String("ffn_norm2.weight")).clone(ctx),
        _resident_fp8_weight(rw, p + String("attention.qkv.weight"), ctx),
        _resident_fp8_weight(rw, p + String("attention.o.weight"), ctx),
        rw.w(p + String("attention.norm_q.weight")).clone(ctx),
        rw.w(p + String("attention.norm_k.weight")).clone(ctx),
        _resident_fp8_weight(rw, p + String("feed_forward.w1.weight"), ctx),
        _resident_fp8_weight(rw, p + String("feed_forward.w2.weight"), ctx),
        _resident_fp8_weight(rw, p + String("feed_forward.w3.weight"), ctx),
    )


def _resident_fp8_weight(
    rw: Ideogram4Weights, name: String, ctx: DeviceContext
) raises -> Tensor:
    return fp8_e4m3_dequant_perrow_to_bf16(
        rw.w(name), rw.w(name + String("_scale")), ctx
    )


struct Ideogram4LoraSet(Movable):
    var ad: List[LArc]
    var n_layers: Int
    var rank: Int
    var active: Bool

    def __init__(out self, var ad: List[LArc], n_layers: Int, rank: Int):
        self.ad = ad^
        self.n_layers = n_layers
        self.rank = rank
        self.active = True

    def __init__(out self, var ad: List[LArc], n_layers: Int, rank: Int, active: Bool):
        self.ad = ad^
        self.n_layers = n_layers
        self.rank = rank
        self.active = active


def build_ideogram4_lora_set[
    Hidden: Int, FF: Int, Adaln: Int,
](
    rank: Int,
    alpha: Float32,
    ctx: DeviceContext,
    n_layers: Int = IDEOGRAM4_NUM_LAYERS,
    seed: UInt64 = UInt64(0x1D3A_4000),
) raises -> Ideogram4LoraSet:
    var ad = List[LArc]()
    var s = seed
    for _layer in range(n_layers):
        ad.append(LArc(make_lora_adapter(Hidden, 3 * Hidden, rank, alpha, s, ctx)))
        s += 1
        ad.append(LArc(make_lora_adapter(Hidden, Hidden, rank, alpha, s, ctx)))
        s += 1
        ad.append(LArc(make_lora_adapter(Hidden, FF, rank, alpha, s, ctx)))
        s += 1
        ad.append(LArc(make_lora_adapter(FF, Hidden, rank, alpha, s, ctx)))
        s += 1
        ad.append(LArc(make_lora_adapter(Hidden, FF, rank, alpha, s, ctx)))
        s += 1
        ad.append(LArc(make_lora_adapter(Adaln, 4 * Hidden, rank, alpha, s, ctx)))
        s += 1
    return Ideogram4LoraSet(ad^, n_layers, rank)


def build_ideogram4_native_lora_set(
    rank: Int, alpha: Float32, ctx: DeviceContext,
    n_layers: Int = IDEOGRAM4_NUM_LAYERS,
    seed: UInt64 = UInt64(0x1D3A_4000),
) raises -> Ideogram4LoraSet:
    return build_ideogram4_lora_set[
        IDEOGRAM4_HIDDEN, IDEOGRAM4_INTERMEDIATE_SIZE, IDEOGRAM4_ADALN_DIM
    ](rank, alpha, ctx, n_layers, seed)


def _loras_for_block(set: Ideogram4LoraSet, layer: Int) -> List[LArc]:
    var base = layer * I4_SLOTS_PER_BLOCK
    var out = List[LArc]()
    for i in range(I4_SLOTS_PER_BLOCK):
        out.append(set.ad[base + i])
    return out^


struct _LoraFwd(Movable):
    var y: Tensor
    var down: Tensor

    def __init__(out self, var y: Tensor, var down: Tensor):
        self.y = y^
        self.down = down^


def _lora_linear_fwd(
    x: Tensor, base_w: Tensor, ad: LoraAdapter, bias: Optional[Tensor],
    ctx: DeviceContext,
) raises -> _LoraFwd:
    var base = linear(x, base_w, bias, ctx)
    var down = linear(x, ad.a, None, ctx)
    var up = linear(down, ad.b, None, ctx)
    var y = add(base, mul_scalar(up, ad.scale(), ctx), ctx)
    return _LoraFwd(y^, down^)


struct _LoraBwd(Movable):
    var d_x: Tensor
    var d_a: Tensor
    var d_b: Tensor

    def __init__(out self, var d_x: Tensor, var d_a: Tensor, var d_b: Tensor):
        self.d_x = d_x^
        self.d_a = d_a^
        self.d_b = d_b^


def _lora_linear_bwd(
    d_y: Tensor,
    x: Tensor,
    down: Tensor,
    base_w: Tensor,
    ad: LoraAdapter,
    M: Int,
    in_f: Int,
    out_f: Int,
    ctx: DeviceContext,
) raises -> _LoraBwd:
    var dx_base = linear_backward_dx(d_y, base_w, M, in_f, out_f, ctx)
    var dy_s = mul_scalar(d_y, ad.scale(), ctx)
    var up_g = linear_backward(dy_s, down, ad.b, M, ad.rank, out_f, ctx)
    var down_g = linear_backward(up_g.d_x, x, ad.a, M, in_f, ad.rank, ctx)
    var d_x = add(dx_base, down_g.d_x, ctx)
    return _LoraBwd(d_x^, _clone(down_g.d_w, ctx), _clone(up_g.d_w, ctx))


struct Ideogram4BlockActs(Movable):
    var x_in: Tensor
    var adaln_input: Tensor
    var mod_scale_msa: Tensor
    var mod_gate_msa_raw: Tensor
    var mod_scale_mlp: Tensor
    var mod_gate_mlp_raw: Tensor
    var gate_msa: Tensor
    var gate_mlp: Tensor
    var an1: Tensor
    var attn_in: Tensor
    var q_raw: Tensor
    var k_raw: Tensor
    var q_norm: Tensor
    var k_norm: Tensor
    var q_rope: Tensor
    var k_rope: Tensor
    var v_bshd: Tensor
    var attn_flat: Tensor
    var attn_out: Tensor
    var attn_n2: Tensor
    var x_mid: Tensor
    var fn1: Tensor
    var mlp_in: Tensor
    var ff_g: Tensor
    var ff_u: Tensor
    var ff_act: Tensor
    var ff_out: Tensor
    var ff_n2: Tensor
    var down_qkv: Tensor
    var down_o: Tensor
    var down_w1: Tensor
    var down_w2: Tensor
    var down_w3: Tensor
    var down_adaln: Tensor
    # Flash-SDPA saved set (filled only on the IDEOGRAM4_SDPA_FLASH path; bf16
    # q/k/v/o + F32 stats — exactly what sdpa_flash_backward_f32 consumes).
    var flash_q: Optional[TArc]
    var flash_k: Optional[TArc]
    var flash_v: Optional[TArc]
    var flash_o: Optional[TArc]
    var flash_stats: Optional[TArc]

    def __init__(
        out self,
        var x_in: Tensor,
        var adaln_input: Tensor,
        var mod_scale_msa: Tensor,
        var mod_gate_msa_raw: Tensor,
        var mod_scale_mlp: Tensor,
        var mod_gate_mlp_raw: Tensor,
        var gate_msa: Tensor,
        var gate_mlp: Tensor,
        var an1: Tensor,
        var attn_in: Tensor,
        var q_raw: Tensor,
        var k_raw: Tensor,
        var q_norm: Tensor,
        var k_norm: Tensor,
        var q_rope: Tensor,
        var k_rope: Tensor,
        var v_bshd: Tensor,
        var attn_flat: Tensor,
        var attn_out: Tensor,
        var attn_n2: Tensor,
        var x_mid: Tensor,
        var fn1: Tensor,
        var mlp_in: Tensor,
        var ff_g: Tensor,
        var ff_u: Tensor,
        var ff_act: Tensor,
        var ff_out: Tensor,
        var ff_n2: Tensor,
        var down_qkv: Tensor,
        var down_o: Tensor,
        var down_w1: Tensor,
        var down_w2: Tensor,
        var down_w3: Tensor,
        var down_adaln: Tensor,
        var flash_q: Optional[TArc] = None,
        var flash_k: Optional[TArc] = None,
        var flash_v: Optional[TArc] = None,
        var flash_o: Optional[TArc] = None,
        var flash_stats: Optional[TArc] = None,
    ):
        self.x_in = x_in^
        self.adaln_input = adaln_input^
        self.mod_scale_msa = mod_scale_msa^
        self.mod_gate_msa_raw = mod_gate_msa_raw^
        self.mod_scale_mlp = mod_scale_mlp^
        self.mod_gate_mlp_raw = mod_gate_mlp_raw^
        self.gate_msa = gate_msa^
        self.gate_mlp = gate_mlp^
        self.an1 = an1^
        self.attn_in = attn_in^
        self.q_raw = q_raw^
        self.k_raw = k_raw^
        self.q_norm = q_norm^
        self.k_norm = k_norm^
        self.q_rope = q_rope^
        self.k_rope = k_rope^
        self.v_bshd = v_bshd^
        self.attn_flat = attn_flat^
        self.attn_out = attn_out^
        self.attn_n2 = attn_n2^
        self.x_mid = x_mid^
        self.fn1 = fn1^
        self.mlp_in = mlp_in^
        self.ff_g = ff_g^
        self.ff_u = ff_u^
        self.ff_act = ff_act^
        self.ff_out = ff_out^
        self.ff_n2 = ff_n2^
        self.down_qkv = down_qkv^
        self.down_o = down_o^
        self.down_w1 = down_w1^
        self.down_w2 = down_w2^
        self.down_w3 = down_w3^
        self.down_adaln = down_adaln^
        self.flash_q = flash_q^
        self.flash_k = flash_k^
        self.flash_v = flash_v^
        self.flash_o = flash_o^
        self.flash_stats = flash_stats^


struct Ideogram4BlockOut(Movable):
    var out: Tensor
    var acts: Ideogram4BlockActs

    def __init__(out self, var out: Tensor, var acts: Ideogram4BlockActs):
        self.out = out^
        self.acts = acts^

    def take_out(deinit self) -> Tensor:
        return self.out^


def ideogram4_block_lora_forward[
    S: Int, Hidden: Int, Heads: Int, Dh: Int, FF: Int, Adaln: Int,
](
    x: Tensor,
    adaln_input: Tensor,
    cosf: Tensor,
    sinf: Tensor,
    w: Ideogram4BlockWeights,
    loras: List[LArc],
    ctx: DeviceContext,
) raises -> Ideogram4BlockOut:
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var mf = _lora_linear_fwd(
        adaln_input,
        w.adaln_w,
        loras[I4_SLOT_ADALN][],
        Optional[Tensor](w.adaln_b.clone(ctx)),
        ctx,
    )
    var mod = _clone(mf.y, ctx)
    var mod_axis = len(mod.shape()) - 1
    var scale_msa = add_scalar(slice(mod, mod_axis, 0 * Hidden, Hidden, ctx), Float32(1.0), ctx)
    var gate_msa_raw = slice(mod, mod_axis, 1 * Hidden, Hidden, ctx)
    var gate_msa = tanh_op(gate_msa_raw, ctx)
    var scale_mlp = add_scalar(slice(mod, mod_axis, 2 * Hidden, Hidden, ctx), Float32(1.0), ctx)
    var gate_mlp_raw = slice(mod, mod_axis, 3 * Hidden, Hidden, ctx)
    var gate_mlp = tanh_op(gate_mlp_raw, ctx)
    var down_adaln = _clone(mf.down, ctx)

    var an1 = rms_norm(x, w.attn_norm1, I4_EPS, ctx)
    var attn_in = mul(an1, scale_msa, ctx)

    var qkvf = _lora_linear_fwd(attn_in, w.qkv_w, loras[I4_SLOT_QKV][], None, ctx)
    var qkv5 = reshape(qkvf.y, _shape5(1, S, 3, Heads, Dh), ctx)
    var q = reshape(slice(qkv5, 2, 0, 1, ctx), _shape4(1, S, Heads, Dh), ctx)
    var k = reshape(slice(qkv5, 2, 1, 1, ctx), _shape4(1, S, Heads, Dh), ctx)
    var v = reshape(slice(qkv5, 2, 2, 1, ctx), _shape4(1, S, Heads, Dh), ctx)
    var q_norm = rms_norm(q, w.norm_q, I4_EPS, ctx)
    var k_norm = rms_norm(k, w.norm_k, I4_EPS, ctx)
    var q_rope = apply_rope_ideogram(q_norm, cosf, sinf, ctx)
    var k_rope = apply_rope_ideogram(k_norm, cosf, sinf, ctx)
    var attn4: Tensor
    var _flash_q: Optional[TArc] = None
    var _flash_k: Optional[TArc] = None
    var _flash_v: Optional[TArc] = None
    var _flash_o: Optional[TArc] = None
    var _flash_stats: Optional[TArc] = None
    comptime if IDEOGRAM4_SDPA_FLASH:
        # cuDNN flash: bf16 q/k/v/o + F32 stats go to the tape for the flash
        # backward (no recompute, no re-cast). att is the F32 [1,S,H,Dh] drop-in.
        var ff = sdpa_flash_train_fwd_f32[1, S, Heads, Dh](
            q_rope, k_rope, v, scale, ctx
        )
        # flash _f32 returns F32 att; ideogram4's block is BF16 end-to-end
        # (math sdpa_nomask returns BF16) -> cast to match the downstream.
        attn4 = cast_tensor(ff.att, STDtype.BF16, ctx)
        _flash_q = Optional[TArc](ff.q_bf)
        _flash_k = Optional[TArc](ff.k_bf)
        _flash_v = Optional[TArc](ff.v_bf)
        _flash_o = Optional[TArc](ff.o_bf)
        _flash_stats = Optional[TArc](ff.stats)
    else:
        attn4 = sdpa_nomask[1, S, Heads, Dh](q_rope, k_rope, v, scale, ctx)
    var attn_flat = reshape(attn4, _shape2(S, Hidden), ctx)

    var of = _lora_linear_fwd(attn_flat, w.o_w, loras[I4_SLOT_O][], None, ctx)
    var attn_out = _clone(of.y, ctx)
    var attn_n2 = rms_norm(attn_out, w.attn_norm2, I4_EPS, ctx)
    var x_mid = add(x, mul(gate_msa, attn_n2, ctx), ctx)

    var fn1 = rms_norm(x_mid, w.ffn_norm1, I4_EPS, ctx)
    var mlp_in = mul(fn1, scale_mlp, ctx)
    var w1f = _lora_linear_fwd(mlp_in, w.w1, loras[I4_SLOT_W1][], None, ctx)
    var ff_g = _clone(w1f.y, ctx)
    var w3f = _lora_linear_fwd(mlp_in, w.w3, loras[I4_SLOT_W3][], None, ctx)
    var ff_u = _clone(w3f.y, ctx)
    var ff_act = swiglu(ff_g, ff_u, ctx)
    var w2f = _lora_linear_fwd(ff_act, w.w2, loras[I4_SLOT_W2][], None, ctx)
    var ff_out = _clone(w2f.y, ctx)
    var ff_n2 = rms_norm(ff_out, w.ffn_norm2, I4_EPS, ctx)
    var out = add(x_mid, mul(gate_mlp, ff_n2, ctx), ctx)

    var acts = Ideogram4BlockActs(
        _clone(x, ctx),
        _clone(adaln_input, ctx),
        scale_msa^,
        gate_msa_raw^,
        scale_mlp^,
        gate_mlp_raw^,
        gate_msa^,
        gate_mlp^,
        an1^,
        _clone(attn_in, ctx),
        q^,
        k^,
        q_norm^,
        k_norm^,
        q_rope^,
        k_rope^,
        _clone(v, ctx),
        attn_flat^,
        _clone(attn_out, ctx),
        attn_n2^,
        _clone(x_mid, ctx),
        fn1^,
        _clone(mlp_in, ctx),
        ff_g^,
        ff_u^,
        _clone(ff_act, ctx),
        _clone(ff_out, ctx),
        ff_n2^,
        _clone(qkvf.down, ctx),
        _clone(of.down, ctx),
        _clone(w1f.down, ctx),
        _clone(w2f.down, ctx),
        _clone(w3f.down, ctx),
        down_adaln^,
        _flash_q^,
        _flash_k^,
        _flash_v^,
        _flash_o^,
        _flash_stats^,
    )
    return Ideogram4BlockOut(out^, acts^)


struct Ideogram4BlockLoraGrads(Movable):
    var d_a: List[TArc]
    var d_b: List[TArc]

    def __init__(out self, var d_a: List[TArc], var d_b: List[TArc]):
        self.d_a = d_a^
        self.d_b = d_b^


struct Ideogram4BlockBwd(Movable):
    var d_x: Tensor
    var d_adaln_input: Tensor
    var lora_grads: Ideogram4BlockLoraGrads

    def __init__(
        out self,
        var d_x: Tensor,
        var d_adaln_input: Tensor,
        var lora_grads: Ideogram4BlockLoraGrads,
    ):
        self.d_x = d_x^
        self.d_adaln_input = d_adaln_input^
        self.lora_grads = lora_grads^


def ideogram4_block_lora_backward[
    S: Int, Hidden: Int, Heads: Int, Dh: Int, FF: Int, Adaln: Int,
](
    d_out: Tensor,
    acts: Ideogram4BlockActs,
    cosf: Tensor,
    sinf: Tensor,
    w: Ideogram4BlockWeights,
    loras: List[LArc],
    ctx: DeviceContext,
) raises -> Ideogram4BlockBwd:
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var d_a = List[TArc]()
    var d_b = List[TArc]()
    for _ in range(I4_SLOTS_PER_BLOCK):
        d_a.append(TArc(zeros_device(_shape1(1), STDtype.BF16, ctx)))
        d_b.append(TArc(zeros_device(_shape1(1), STDtype.BF16, ctx)))

    # out = x_mid + gate_mlp * ff_n2
    var d_x_mid = _clone(d_out, ctx)
    var d_ff_n2 = mul(d_out, acts.gate_mlp, ctx)
    var d_gate_mlp = broadcast_backward(
        mul(d_out, acts.ff_n2, ctx), acts.gate_mlp.shape(), ctx
    )
    var d_ff_out = rms_norm_backward_dx(d_ff_n2, acts.ff_out, w.ffn_norm2, I4_EPS, ctx)

    var w2b = _lora_linear_bwd(
        d_ff_out, acts.ff_act, acts.down_w2, w.w2, loras[I4_SLOT_W2][],
        S, FF, Hidden, ctx,
    )
    d_a[I4_SLOT_W2] = TArc(_clone(w2b.d_a, ctx))
    d_b[I4_SLOT_W2] = TArc(_clone(w2b.d_b, ctx))

    var sg = swiglu_backward(w2b.d_x, acts.ff_g, acts.ff_u, ctx)
    var w1b = _lora_linear_bwd(
        sg.d_gate, acts.mlp_in, acts.down_w1, w.w1, loras[I4_SLOT_W1][],
        S, Hidden, FF, ctx,
    )
    d_a[I4_SLOT_W1] = TArc(_clone(w1b.d_a, ctx))
    d_b[I4_SLOT_W1] = TArc(_clone(w1b.d_b, ctx))

    var w3b = _lora_linear_bwd(
        sg.d_up, acts.mlp_in, acts.down_w3, w.w3, loras[I4_SLOT_W3][],
        S, Hidden, FF, ctx,
    )
    d_a[I4_SLOT_W3] = TArc(_clone(w3b.d_a, ctx))
    d_b[I4_SLOT_W3] = TArc(_clone(w3b.d_b, ctx))

    var d_mlp_in = add(w1b.d_x, w3b.d_x, ctx)
    var d_scale_mlp = broadcast_backward(
        mul(d_mlp_in, acts.fn1, ctx), acts.mod_scale_mlp.shape(), ctx
    )
    var d_fn1 = mul(d_mlp_in, acts.mod_scale_mlp, ctx)
    d_x_mid = add(
        d_x_mid,
        rms_norm_backward_dx(d_fn1, acts.x_mid, w.ffn_norm1, I4_EPS, ctx),
        ctx,
    )

    # x_mid = x + gate_msa * attn_n2
    var d_x = _clone(d_x_mid, ctx)
    var d_attn_n2 = mul(d_x_mid, acts.gate_msa, ctx)
    var d_gate_msa = broadcast_backward(
        mul(d_x_mid, acts.attn_n2, ctx), acts.gate_msa.shape(), ctx
    )
    var d_attn_out = rms_norm_backward_dx(
        d_attn_n2, acts.attn_out, w.attn_norm2, I4_EPS, ctx
    )

    var ob = _lora_linear_bwd(
        d_attn_out, acts.attn_flat, acts.down_o, w.o_w, loras[I4_SLOT_O][],
        S, Hidden, Hidden, ctx,
    )
    d_a[I4_SLOT_O] = TArc(_clone(ob.d_a, ctx))
    d_b[I4_SLOT_O] = TArc(_clone(ob.d_b, ctx))

    var d_attn4 = reshape(ob.d_x, _shape4(1, S, Heads, Dh), ctx)
    var sd_d_q: Tensor
    var sd_d_k: Tensor
    var sd_d_v: Tensor
    comptime if IDEOGRAM4_SDPA_FLASH:
        if not acts.flash_stats:
            raise Error(
                "ideogram4 block bwd: IDEOGRAM4_SDPA_FLASH on but the tape has"
                " no flash set (fwd/bwd flag mismatch)"
            )
        var fb = sdpa_flash_backward_f32[1, S, Heads, Dh](
            acts.flash_q.value(), acts.flash_k.value(), acts.flash_v.value(),
            acts.flash_o.value(), acts.flash_stats.value(), d_attn4, scale, ctx,
        )
        # flash _f32 returns F32 grads; the math path's grads are BF16 -> cast
        # to match the downstream rope/rms backward (also avoids a partial move).
        sd_d_q = cast_tensor(fb.d_q, STDtype.BF16, ctx)
        sd_d_k = cast_tensor(fb.d_k, STDtype.BF16, ctx)
        sd_d_v = cast_tensor(fb.d_v, STDtype.BF16, ctx)
    else:
        var sd = sdpa_backward[1, S, Heads, Dh](
            acts.q_rope, acts.k_rope, acts.v_bshd, d_attn4, scale, ctx
        )
        sd_d_q = Tensor(sd.d_q.buf.copy(), sd.d_q.shape(), sd.d_q.dtype())
        sd_d_k = Tensor(sd.d_k.buf.copy(), sd.d_k.shape(), sd.d_k.dtype())
        sd_d_v = Tensor(sd.d_v.buf.copy(), sd.d_v.shape(), sd.d_v.dtype())
    var cos_full = _expand_rope_table[Heads, Dh](cosf, S, ctx)
    var sin_full = _expand_rope_table[Heads, Dh](sinf, S, ctx)
    var d_q_norm = rope_halfsplit_full_backward(sd_d_q, cos_full, sin_full, ctx)
    var d_k_norm = rope_halfsplit_full_backward(sd_d_k, cos_full, sin_full, ctx)

    var d_q_raw = rms_norm_backward_dx(d_q_norm, acts.q_raw, w.norm_q, I4_EPS, ctx)
    var d_k_raw = rms_norm_backward_dx(d_k_norm, acts.k_raw, w.norm_k, I4_EPS, ctx)
    var d_q = reshape(d_q_raw, _shape2(S, Hidden), ctx)
    var d_k = reshape(d_k_raw, _shape2(S, Hidden), ctx)
    var d_v = reshape(sd_d_v, _shape2(S, Hidden), ctx)
    var d_qkv = concat(1, ctx, d_q, d_k, d_v)

    var qkvb = _lora_linear_bwd(
        d_qkv, acts.attn_in, acts.down_qkv, w.qkv_w, loras[I4_SLOT_QKV][],
        S, Hidden, 3 * Hidden, ctx,
    )
    d_a[I4_SLOT_QKV] = TArc(_clone(qkvb.d_a, ctx))
    d_b[I4_SLOT_QKV] = TArc(_clone(qkvb.d_b, ctx))

    var d_scale_msa = broadcast_backward(
        mul(qkvb.d_x, acts.an1, ctx), acts.mod_scale_msa.shape(), ctx
    )
    var d_an1 = mul(qkvb.d_x, acts.mod_scale_msa, ctx)
    d_x = add(
        d_x,
        rms_norm_backward_dx(d_an1, acts.x_in, w.attn_norm1, I4_EPS, ctx),
        ctx,
    )

    # d_mod chunks: scale chunks are identity; gate chunks pass tanh backward.
    var d_gate_msa_raw = tanh_backward(d_gate_msa, acts.mod_gate_msa_raw, ctx)
    var d_gate_mlp_raw = tanh_backward(d_gate_mlp, acts.mod_gate_mlp_raw, ctx)
    var d_mod_axis = len(d_scale_msa.shape()) - 1
    var d_mod = concat(
        d_mod_axis, ctx, d_scale_msa, d_gate_msa_raw, d_scale_mlp, d_gate_mlp_raw
    )
    var adalnb = _lora_linear_bwd(
        d_mod, acts.adaln_input, acts.down_adaln, w.adaln_w,
        loras[I4_SLOT_ADALN][], 1, Adaln, 4 * Hidden, ctx,
    )
    d_a[I4_SLOT_ADALN] = TArc(_clone(adalnb.d_a, ctx))
    d_b[I4_SLOT_ADALN] = TArc(_clone(adalnb.d_b, ctx))

    return Ideogram4BlockBwd(
        d_x^, _clone(adalnb.d_x, ctx), Ideogram4BlockLoraGrads(d_a^, d_b^)
    )


struct Ideogram4StackForward(Movable):
    var out: Tensor
    var x_inputs: List[TArc]

    def __init__(out self, var out: Tensor, var x_inputs: List[TArc]):
        self.out = out^
        self.x_inputs = x_inputs^


def ideogram4_stack_lora_forward[
    S: Int, Hidden: Int, Heads: Int, Dh: Int, FF: Int, Adaln: Int,
](
    x_in: Tensor,
    adaln_input: Tensor,
    cosf: Tensor,
    sinf: Tensor,
    st: ShardedSafeTensors,
    loras: Ideogram4LoraSet,
    ctx: DeviceContext,
) raises -> Ideogram4StackForward:
    var x = _clone(x_in, ctx)
    var saved = List[TArc]()
    for layer in range(loras.n_layers):
        saved.append(TArc(_clone(x, ctx)))
        var w = load_ideogram4_block_weights(st, layer, ctx)
        var bl = _loras_for_block(loras, layer)
        var out = ideogram4_block_lora_forward[S, Hidden, Heads, Dh, FF, Adaln](
            x, adaln_input, cosf, sinf, w^, bl, ctx
        )
        x = out^.take_out()
    return Ideogram4StackForward(x^, saved^)


def ideogram4_stack_lora_forward_resident[
    S: Int, Hidden: Int, Heads: Int, Dh: Int, FF: Int, Adaln: Int,
](
    x_in: Tensor,
    adaln_input: Tensor,
    cosf: Tensor,
    sinf: Tensor,
    rw: Ideogram4Weights,
    loras: Ideogram4LoraSet,
    ctx: DeviceContext,
) raises -> Ideogram4StackForward:
    var x = _clone(x_in, ctx)
    var saved = List[TArc]()
    for layer in range(loras.n_layers):
        saved.append(TArc(_clone(x, ctx)))
        var w = load_ideogram4_block_weights_resident(rw, layer, ctx)
        var bl = _loras_for_block(loras, layer)
        var out = ideogram4_block_lora_forward[S, Hidden, Heads, Dh, FF, Adaln](
            x, adaln_input, cosf, sinf, w^, bl, ctx
        )
        x = out^.take_out()
    return Ideogram4StackForward(x^, saved^)


struct Ideogram4StackLoraGrads(Movable):
    var d_a: List[TArc]
    var d_b: List[TArc]
    var d_x_in: Tensor
    var d_adaln_input: Tensor

    def __init__(
        out self,
        var d_a: List[TArc],
        var d_b: List[TArc],
        var d_x_in: Tensor,
        var d_adaln_input: Tensor,
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_x_in = d_x_in^
        self.d_adaln_input = d_adaln_input^


def ideogram4_stack_lora_backward[
    S: Int, Hidden: Int, Heads: Int, Dh: Int, FF: Int, Adaln: Int,
](
    d_out: Tensor,
    adaln_input: Tensor,
    cosf: Tensor,
    sinf: Tensor,
    st: ShardedSafeTensors,
    loras: Ideogram4LoraSet,
    fwd: Ideogram4StackForward,
    ctx: DeviceContext,
) raises -> Ideogram4StackLoraGrads:
    var d_a = List[TArc]()
    var d_b = List[TArc]()
    for _ in range(loras.n_layers * I4_SLOTS_PER_BLOCK):
        d_a.append(TArc(zeros_device(_shape1(1), STDtype.BF16, ctx)))
        d_b.append(TArc(zeros_device(_shape1(1), STDtype.BF16, ctx)))

    var d_x = _clone(d_out, ctx)
    var d_adaln = zeros_device(adaln_input.shape(), adaln_input.dtype(), ctx)
    var layer = loras.n_layers - 1
    while layer >= 0:
        var w = load_ideogram4_block_weights(st, layer, ctx)
        var bl = _loras_for_block(loras, layer)
        var rb = ideogram4_block_lora_forward[S, Hidden, Heads, Dh, FF, Adaln](
            fwd.x_inputs[layer][], adaln_input, cosf, sinf, w, bl, ctx
        )
        var bb = ideogram4_block_lora_backward[S, Hidden, Heads, Dh, FF, Adaln](
            d_x, rb.acts^, cosf, sinf, w, bl, ctx
        )
        var base = layer * I4_SLOTS_PER_BLOCK
        for slot in range(I4_SLOTS_PER_BLOCK):
            d_a[base + slot] = TArc(_clone(bb.lora_grads.d_a[slot][], ctx))
            d_b[base + slot] = TArc(_clone(bb.lora_grads.d_b[slot][], ctx))
        d_adaln = add(d_adaln, bb.d_adaln_input, ctx)
        d_x = _clone(bb.d_x, ctx)
        layer -= 1

    return Ideogram4StackLoraGrads(d_a^, d_b^, d_x^, d_adaln^)


def ideogram4_stack_lora_backward_resident[
    S: Int, Hidden: Int, Heads: Int, Dh: Int, FF: Int, Adaln: Int,
](
    d_out: Tensor,
    adaln_input: Tensor,
    cosf: Tensor,
    sinf: Tensor,
    rw: Ideogram4Weights,
    loras: Ideogram4LoraSet,
    fwd: Ideogram4StackForward,
    ctx: DeviceContext,
) raises -> Ideogram4StackLoraGrads:
    var d_a = List[TArc]()
    var d_b = List[TArc]()
    for _ in range(loras.n_layers * I4_SLOTS_PER_BLOCK):
        d_a.append(TArc(zeros_device(_shape1(1), STDtype.BF16, ctx)))
        d_b.append(TArc(zeros_device(_shape1(1), STDtype.BF16, ctx)))

    var d_x = _clone(d_out, ctx)
    var d_adaln = zeros_device(adaln_input.shape(), adaln_input.dtype(), ctx)
    var layer = loras.n_layers - 1
    while layer >= 0:
        var w = load_ideogram4_block_weights_resident(rw, layer, ctx)
        var bl = _loras_for_block(loras, layer)
        var rb = ideogram4_block_lora_forward[S, Hidden, Heads, Dh, FF, Adaln](
            fwd.x_inputs[layer][], adaln_input, cosf, sinf, w, bl, ctx
        )
        var bb = ideogram4_block_lora_backward[S, Hidden, Heads, Dh, FF, Adaln](
            d_x, rb.acts^, cosf, sinf, w, bl, ctx
        )
        var base = layer * I4_SLOTS_PER_BLOCK
        for slot in range(I4_SLOTS_PER_BLOCK):
            d_a[base + slot] = TArc(_clone(bb.lora_grads.d_a[slot][], ctx))
            d_b[base + slot] = TArc(_clone(bb.lora_grads.d_b[slot][], ctx))
        d_adaln = add(d_adaln, bb.d_adaln_input, ctx)
        d_x = _clone(bb.d_x, ctx)
        layer -= 1

    return Ideogram4StackLoraGrads(d_a^, d_b^, d_x^, d_adaln^)


def _expand_rope_table[Heads: Int, Dh: Int](
    table: Tensor, seq_len: Int, ctx: DeviceContext
) raises -> Tensor:
    var t4 = reshape(table, _shape4(1, seq_len, 1, Dh), ctx)
    var ones = add_scalar(
        zeros_device(_shape4(1, seq_len, Heads, Dh), table.dtype(), ctx),
        Float32(1.0),
        ctx,
    )
    return mul(ones, t4, ctx)


def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    return cast_tensor(x, x.dtype(), ctx)


def _shape1(a: Int) -> List[Int]:
    var s = List[Int](); s.append(a); return s^


def _shape2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); s.append(d); return s^


def _shape5(a: Int, b: Int, c: Int, d: Int, e: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    s.append(d)
    s.append(e)
    return s^
