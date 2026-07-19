# ZImageDiT.mojo — the FULL Z-Image NextDiT wrapper (hand-chained fwd+bwd) that
# turns a latent [1,16,HL,WL] into the [S,dim] block stream and back, around the
# already-verified main-block stack (model/ZImageModel.mojo).
#
# BORROWED (structure + math, read line-by-line, NOT a runtime import) FROM:
#   serenitymojo/models/dit/zimage_dit.mojo  (NextDiT.forward / _forward_impl,
#   _patchify_zimage, _unpatchify_zimage, _t_embedder, _cap_embedder, _attention,
#   _feed_forward, _block (modulated + unmodulated), _final_layer, _build_rope).
# We COPY that forward into the serenity_trainer namespace and add a HAND-CHAINED
# backward that propagates d_x through the frozen embedders / refiners / final
# layer (dx-only — those base weights are FROZEN, their d_W is discarded) and
# reuses zimage_stack_backward for the 30 LoRA-trained main blocks.
#
# Convention REF (input/output/target/timestep): Serenity
#   modules/modelSetup/BaseZImageSetup.py::predict + modules/model/ZImageModel.py
#   (and the port siblings modelSetup/BaseZImageSetup.mojo,
#   modelSetup/ZImageLoRASetup.mojo). The model output is the predicted VELOCITY
#   latent [1,16,HL,WL]; predict() does predicted = -velocity, target = noise -
#   scaled_latent.
#
# WHY HAND-CHAINED (no serenitymojo autograd.backward for the model): the main
# stack fwd/bwd is hand-chained (per-op reverse, no 9-op tape). The LoRA grads are
# returned as a ZImageStackLoraGrads the driver reads directly (one d_a/d_b pair
# per of the 210 adapters, plus the load-bearing d_x_in we DON'T need above the
# stack — the embedders are frozen so nothing above consumes the model d_x). Only
# the small per-step tape ops in predict() (noise/sigma/target arithmetic) live on
# the serenitymojo tape; the model itself does not.
#
# Config (NextDiTConfig.zimage): dim=3840, n_heads=30, head_dim=128, n_layers=30,
#   n_refiner=2, cap_feat_dim=2560, norm_eps=1e-5, rope_theta=256, t_scale=1000,
#   axes_dims=[32,48,48], patch_size=2, in_channels=16, adaln_embed_dim=256,
#   final norm_final eps=1e-6.
#
# DTYPE: BF16 storage, F32-register inside the foundation kernels. Comptime-shaped
# on HL/WL/CAPLEN so the unified sequence length is a compile-time constant for the
# foundation sdpa dispatch (via the [S] params on the block/stack/attn helpers).

from std.math import cos as fcos, sin as fsin, exp as fexp, log as flog, sqrt
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

# forward ops
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm, layer_norm
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.activations import silu, swiglu
from serenitymojo.ops.tensor_algebra import (
    add, mul, mul_scalar, add_scalar, reshape, permute, concat, slice,
    zeros_device,
)
from serenitymojo.ops.unary import tanh_op
from serenitymojo.ops.cast import cast_tensor

# backward ops (hand-chained, no Tape)
from serenitymojo.ops.linalg_backward import linear_backward_dx
from serenitymojo.ops.norm_backward import rms_norm_backward_dx, layer_norm_backward_dx
from serenitymojo.ops.rope_struct_backward import rope_backward
from serenitymojo.ops.attention_backward import sdpa_backward
from serenitymojo.ops.loss_swiglu_backward import swiglu_backward
from serenitymojo.ops.shape_backward import reshape_backward, permute_backward, slice_backward

from serenity_trainer.modelLoader.ZImageModelLoader import ZImageWeights
from serenity_trainer.model.ZImageModel import (
    ZImageLoraSet, ZImageStackForward, ZImageStackLoraGrads,
    zimage_stack_forward, zimage_stack_backward, zimage_stack_forward_infer,
    ZH as ZH_, ZDh as ZDh_, ZDIM as ZDIM_, ZEPS as ZEPS_,
)


# ── Z-Image NextDiT dims (comptime; mirror NextDiTConfig.zimage) ──────────────
comptime ZH = ZH_              # 30 heads
comptime ZDh = ZDh_            # 128 head dim
comptime ZDIM = ZDIM_          # 3840 = ZH*ZDh
comptime ZEPS = ZEPS_          # 1e-5 attn/rms eps
comptime ZN_REFINER = 2        # noise/context refiner depth
comptime ZADALN = 256          # adaln_embed_dim
comptime ZPATCH = 2            # patch_size
comptime ZIN_CH = 16           # in_channels
comptime ZTHETA = Float64(256.0)
comptime ZT_SCALE = Float32(1000.0)
comptime ZAXIS0 = 32
comptime ZAXIS1 = 48
comptime ZAXIS2 = 48
comptime ZFINAL_EPS = Float32(1e-6)
comptime ZOUT_DIM = ZPATCH * ZPATCH * ZIN_CH   # 64 = pH*pW*C


# ══════════════════════════════════════════════════════════════════════════════
# shape helpers
# ══════════════════════════════════════════════════════════════════════════════
# Explicit arity helpers (variadic-free; matches ZImageModel._shapeN discipline).
def _sh(a: Int) -> List[Int]:
    var s = List[Int](); s.append(a); return s^
def _sh(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^
def _sh(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); return s^
def _sh(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); s.append(d); return s^
def _sh(a: Int, b: Int, c: Int, d: Int, e: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); s.append(d); s.append(e); return s^

def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    return cast_tensor(x, x.dtype(), ctx)   # dtype no-op → deep device copy


# ══════════════════════════════════════════════════════════════════════════════
# t_embedder: sinusoidal timestep_embedding(t*t_scale, 256) → mlp.0 → silu → mlp.2
# (borrowed from zimage_dit.mojo::_t_embedder). adaln_input [1,256]. We SAVE the
# silu activation + pre-silu h for the (frozen) backward; the only consumer of the
# adaln grad above is the frozen MLP, so we discard it — adaln receives no grad
# that reaches a trainable param (the adaLN_modulation Linears are frozen too).
# ══════════════════════════════════════════════════════════════════════════════
def _t_embedder(w: ZImageWeights, t_val: Float32, ctx: DeviceContext) raises -> Tensor:
    var half = ZADALN // 2
    var max_period = Float32(10000.0)
    var scaled = t_val * ZT_SCALE
    var emb = List[Float32]()
    var log_mp = flog(max_period)
    for i in range(half):
        var freq = fexp(-log_mp * Float32(i) / Float32(half))
        emb.append(fcos(scaled * freq))
    for i in range(half):
        var freq = fexp(-log_mp * Float32(i) / Float32(half))
        emb.append(fsin(scaled * freq))
    var dtype = w.get(String("t_embedder.mlp.0.weight")).dtype()
    var t_freq = Tensor.from_host(emb, _sh(1, ZADALN), dtype, ctx)
    ref w0 = w.get(String("t_embedder.mlp.0.weight"))
    ref b0 = w.get(String("t_embedder.mlp.0.bias"))
    var h = linear(t_freq, w0, Optional[Tensor](_clone(b0, ctx)), ctx)
    var ha = silu(h, ctx)
    ref w2 = w.get(String("t_embedder.mlp.2.weight"))
    ref b2 = w.get(String("t_embedder.mlp.2.bias"))
    return linear(ha, w2, Optional[Tensor](_clone(b2, ctx)), ctx)


# ── caption embedder: RMSNorm(cap_feat_dim) + Linear (frozen). Saves the normed
# activation for the dx-backward. (zimage_dit.mojo::_cap_embedder)
struct _CapEmbed(Movable):
    var out: Tensor    # [CAPLEN, dim]
    var normed: Tensor # [CAPLEN, cap_feat_dim]
    def __init__(out self, var out: Tensor, var normed: Tensor):
        self.out = out^
        self.normed = normed^

def _cap_embedder(w: ZImageWeights, cap_feats: Tensor, ctx: DeviceContext) raises -> _CapEmbed:
    ref nw = w.get(String("cap_embedder.0.weight"))
    var normed = rms_norm(cap_feats, nw, ZEPS, ctx)
    ref lw = w.get(String("cap_embedder.1.weight"))
    if w.has(String("cap_embedder.1.bias")):
        ref lb = w.get(String("cap_embedder.1.bias"))
        var o = linear(normed, lw, Optional[Tensor](_clone(lb, ctx)), ctx)
        return _CapEmbed(o^, normed^)
    var o = linear(normed, lw, None, ctx)
    return _CapEmbed(o^, normed^)


# ══════════════════════════════════════════════════════════════════════════════
# patchify / unpatchify (Z-Image channel-MINOR, zimage_dit.mojo::_patchify_zimage
# / _unpatchify_zimage). Pure reshape+permute → backward is permute_backward +
# reshape_backward (no weights).
# ══════════════════════════════════════════════════════════════════════════════
def _patchify[HL: Int, WL: Int](x: Tensor, ctx: DeviceContext) raises -> Tensor:
    # latent [1,C,H,W] -> [C,Ht,p,Wt,p] -> permute(Ht,Wt,pH,pW,C) -> [1,Ht*Wt,p*p*C]
    var c = ZIN_CH
    var ht = HL // ZPATCH
    var wt = WL // ZPATCH
    var xv = reshape(x, _sh(c, ht, ZPATCH, wt, ZPATCH), ctx)
    var xp = permute(xv, _sh(1, 3, 2, 4, 0), ctx)        # [Ht,Wt,pH,pW,C]
    return reshape(xp, _sh(1, ht * wt, ZPATCH * ZPATCH * c), ctx)

def _patchify_backward[HL: Int, WL: Int](d_seq: Tensor, ctx: DeviceContext) raises -> Tensor:
    var c = ZIN_CH
    var ht = HL // ZPATCH
    var wt = WL // ZPATCH
    # reverse reshape [1,Ht*Wt,64] -> [Ht,Wt,pH,pW,C]
    var d_perm = reshape_backward(d_seq, _sh(ht, wt, ZPATCH, ZPATCH, c), ctx)
    # reverse permute(1,3,2,4,0)
    var d_view = permute_backward(d_perm, _sh(1, 3, 2, 4, 0), ctx)  # [C,Ht,p,Wt,p]
    # reverse reshape [C,Ht,p,Wt,p] -> [1,C,H,W]
    return reshape_backward(d_view, _sh(1, c, HL, WL), ctx)

def _unpatchify[HL: Int, WL: Int](seq: Tensor, ctx: DeviceContext) raises -> Tensor:
    var c = ZIN_CH
    var ht = HL // ZPATCH
    var wt = WL // ZPATCH
    var sv = reshape(seq, _sh(ht, wt, ZPATCH, ZPATCH, c), ctx)   # [Ht,Wt,pH,pW,C]
    var sp = permute(sv, _sh(4, 0, 2, 1, 3), ctx)               # [C,Ht,pH,Wt,pW]
    return reshape(sp, _sh(1, c, HL, WL), ctx)

def _unpatchify_backward[HL: Int, WL: Int](d_img: Tensor, ctx: DeviceContext) raises -> Tensor:
    var c = ZIN_CH
    var ht = HL // ZPATCH
    var wt = WL // ZPATCH
    # reverse reshape [C,Ht,pH,Wt,pW] -> [1,C,H,W]
    var d_sp = reshape_backward(d_img, _sh(c, ht, ZPATCH, wt, ZPATCH), ctx)
    # reverse permute(4,0,2,1,3)
    var d_sv = permute_backward(d_sp, _sh(4, 0, 2, 1, 3), ctx)   # [Ht,Wt,pH,pW,C]
    # reverse reshape -> [1,Ht*Wt,64]
    return reshape_backward(d_sv, _sh(1, ht * wt, ZPATCH * ZPATCH * c), ctx)


# ══════════════════════════════════════════════════════════════════════════════
# RoPE table build (host trig, zimage_dit.mojo::_build_rope). cos/sin tables
# [S*H, Dh/2]. No grad (fixed tables) — backward uses rope_backward only.
# ══════════════════════════════════════════════════════════════════════════════
struct _RopePair(Movable):
    var cos: Tensor
    var sin: Tensor
    def __init__(out self, var cos: Tensor, var sin: Tensor):
        self.cos = cos^
        self.sin = sin^

def _build_rope(positions: List[List[Int]], dtype: STDtype, ctx: DeviceContext) raises -> _RopePair:
    var half = ZDh // 2  # 64
    var s = len(positions)
    var theta = Float32(ZTHETA)
    var log_theta = flog(theta)
    var axes = List[Int]()
    axes.append(ZAXIS0); axes.append(ZAXIS1); axes.append(ZAXIS2)
    var cos_vals = List[Float32]()
    var sin_vals = List[Float32]()
    for t in range(s):
        var angles = List[Float32]()
        for a in range(3):
            var da = axes[a]
            var ha = da // 2
            var pos = Float32(positions[t][a])
            for i in range(ha):
                var inv_freq = fexp(-log_theta * Float32(2 * i) / Float32(da))
                angles.append(pos * inv_freq)
        for _head in range(ZH):
            for i in range(half):
                cos_vals.append(fcos(angles[i]))
                sin_vals.append(fsin(angles[i]))
    var rows = s * ZH
    var cos_t = Tensor.from_host(cos_vals, _sh(rows, half), dtype, ctx)
    var sin_t = Tensor.from_host(sin_vals, _sh(rows, half), dtype, ctx)
    return _RopePair(cos_t^, sin_t^)


# ══════════════════════════════════════════════════════════════════════════════
# FROZEN attention (no LoRA) — used by the refiners. Operates on [S,dim] flattened
# (batch=1). Saves what the dx-backward needs. Mirrors zimage_dit.mojo::_attention
# but with frozen base linears (dx-only backward, d_W discarded).
# ══════════════════════════════════════════════════════════════════════════════
struct _AttnSaves(Movable):
    var x_in: Tensor       # [S,dim] (attn input = normed branch input)
    var q_pre_rope: Tensor # [1,S,H,Dh]
    var k_pre_rope: Tensor # [1,S,H,Dh]
    var q_rope: Tensor
    var k_rope: Tensor
    var v_bshd: Tensor
    var to_q_raw: Tensor   # pre norm_q [1,S,H,Dh]
    var to_k_raw: Tensor
    var attn_flat: Tensor  # [S,dim] pre to_out
    def __init__(
        out self, var x_in: Tensor, var q_pre_rope: Tensor, var k_pre_rope: Tensor,
        var q_rope: Tensor, var k_rope: Tensor, var v_bshd: Tensor,
        var to_q_raw: Tensor, var to_k_raw: Tensor, var attn_flat: Tensor,
    ):
        self.x_in = x_in^; self.q_pre_rope = q_pre_rope^; self.k_pre_rope = k_pre_rope^
        self.q_rope = q_rope^; self.k_rope = k_rope^; self.v_bshd = v_bshd^
        self.to_q_raw = to_q_raw^; self.to_k_raw = to_k_raw^; self.attn_flat = attn_flat^

struct _AttnOut(Movable):
    var out: Tensor
    var saves: _AttnSaves
    def __init__(out self, var out: Tensor, var saves: _AttnSaves):
        self.out = out^
        self.saves = saves^

def _frozen_attention[S: Int](
    x: Tensor, cos: Tensor, sin: Tensor, w: ZImageWeights, prefix: String, ctx: DeviceContext
) raises -> _AttnOut:
    var scale = Float32(1.0) / sqrt(Float32(ZDh))
    ref qw = w.get(prefix + String(".attention.to_q.weight"))
    ref kw = w.get(prefix + String(".attention.to_k.weight"))
    ref vw = w.get(prefix + String(".attention.to_v.weight"))
    var q = reshape(linear(x, qw, None, ctx), _sh(1, S, ZH, ZDh), ctx)
    var k = reshape(linear(x, kw, None, ctx), _sh(1, S, ZH, ZDh), ctx)
    var v = reshape(linear(x, vw, None, ctx), _sh(1, S, ZH, ZDh), ctx)
    var to_q_raw = _clone(q, ctx)
    var to_k_raw = _clone(k, ctx)
    ref qn = w.get(prefix + String(".attention.norm_q.weight"))
    ref kn = w.get(prefix + String(".attention.norm_k.weight"))
    var qn4 = rms_norm(q, qn, ZEPS, ctx)
    var kn4 = rms_norm(k, kn, ZEPS, ctx)
    var q_pre_rope = _clone(qn4, ctx)
    var k_pre_rope = _clone(kn4, ctx)
    var qr = rope_interleaved(qn4, cos, sin, ctx)
    var kr = rope_interleaved(kn4, cos, sin, ctx)
    var attn4 = sdpa_nomask[1, S, ZH, ZDh](qr, kr, v, scale, ctx)
    var attn_flat = reshape(attn4, _sh(S, ZDIM), ctx)
    ref ow = w.get(prefix + String(".attention.to_out.0.weight"))
    var out = linear(attn_flat, ow, None, ctx)
    var saves = _AttnSaves(
        _clone(x, ctx), q_pre_rope^, k_pre_rope^, _clone(qr, ctx), _clone(kr, ctx),
        _clone(v, ctx), to_q_raw^, to_k_raw^, _clone(attn_flat, ctx),
    )
    return _AttnOut(out^, saves^)

# dx-only attention backward. d_out [S,dim] → d_x [S,dim] (grad wrt attn input).
def _frozen_attention_backward[S: Int](
    d_out: Tensor, saves: _AttnSaves, cos: Tensor, sin: Tensor,
    w: ZImageWeights, prefix: String, ctx: DeviceContext,
) raises -> Tensor:
    var scale = Float32(1.0) / sqrt(Float32(ZDh))
    # to_out: [S,dim]->[S,dim]
    ref ow = w.get(prefix + String(".attention.to_out.0.weight"))
    var d_attn_flat = linear_backward_dx(d_out, ow, S, ZDIM, ZDIM, ctx)
    var d_attn4 = reshape(d_attn_flat, _sh(1, S, ZH, ZDh), ctx)
    var sd = sdpa_backward[1, S, ZH, ZDh](saves.q_rope, saves.k_rope, saves.v_bshd, d_attn4, scale, ctx)
    var d_qn4 = rope_backward(sd.d_q, cos, sin, True, ctx)
    var d_kn4 = rope_backward(sd.d_k, cos, sin, True, ctx)
    ref qn = w.get(prefix + String(".attention.norm_q.weight"))
    ref kn = w.get(prefix + String(".attention.norm_k.weight"))
    var d_q4 = rms_norm_backward_dx(d_qn4, saves.to_q_raw, qn, ZEPS, ctx)
    var d_k4 = rms_norm_backward_dx(d_kn4, saves.to_k_raw, kn, ZEPS, ctx)
    var d_q = reshape(d_q4, _sh(S, ZDIM), ctx)
    var d_k = reshape(d_k4, _sh(S, ZDIM), ctx)
    var d_v = reshape(sd.d_v, _sh(S, ZDIM), ctx)
    ref qw = w.get(prefix + String(".attention.to_q.weight"))
    ref kw = w.get(prefix + String(".attention.to_k.weight"))
    ref vw = w.get(prefix + String(".attention.to_v.weight"))
    var dxq = linear_backward_dx(d_q, qw, S, ZDIM, ZDIM, ctx)
    var dxk = linear_backward_dx(d_k, kw, S, ZDIM, ZDIM, ctx)
    var dxv = linear_backward_dx(d_v, vw, S, ZDIM, ZDIM, ctx)
    return add(add(dxq, dxk, ctx), dxv, ctx)


# ══════════════════════════════════════════════════════════════════════════════
# FROZEN feed-forward (no LoRA, refiners). w2(silu(w1(x))*w3(x)). Saves g/u for
# swiglu backward. dx-only.
# ══════════════════════════════════════════════════════════════════════════════
struct _FFSaves(Movable):
    var x_in: Tensor   # [S,dim]
    var g: Tensor      # w1(x) [S,ff]
    var u: Tensor      # w3(x) [S,ff]
    var act: Tensor    # swiglu(g,u) [S,ff]
    def __init__(out self, var x_in: Tensor, var g: Tensor, var u: Tensor, var act: Tensor):
        self.x_in = x_in^; self.g = g^; self.u = u^; self.act = act^

struct _FFOut(Movable):
    var out: Tensor
    var saves: _FFSaves
    def __init__(out self, var out: Tensor, var saves: _FFSaves):
        self.out = out^
        self.saves = saves^

def _frozen_feed_forward[S: Int](
    x: Tensor, w: ZImageWeights, prefix: String, ctx: DeviceContext
) raises -> _FFOut:
    ref w1 = w.get(prefix + String(".feed_forward.w1.weight"))
    ref w3 = w.get(prefix + String(".feed_forward.w3.weight"))
    var g = linear(x, w1, None, ctx)
    var u = linear(x, w3, None, ctx)
    var act = swiglu(g, u, ctx)
    ref w2 = w.get(prefix + String(".feed_forward.w2.weight"))
    var out = linear(act, w2, None, ctx)
    var saves = _FFSaves(_clone(x, ctx), _clone(g, ctx), _clone(u, ctx), _clone(act, ctx))
    return _FFOut(out^, saves^)

def _frozen_feed_forward_backward[S: Int](
    d_out: Tensor, saves: _FFSaves, w: ZImageWeights, prefix: String, ctx: DeviceContext
) raises -> Tensor:
    var ff = saves.g.shape()[1]
    ref w2 = w.get(prefix + String(".feed_forward.w2.weight"))
    var d_act = linear_backward_dx(d_out, w2, S, ff, ZDIM, ctx)
    var sg = swiglu_backward(d_act, saves.g, saves.u, ctx)
    ref w1 = w.get(prefix + String(".feed_forward.w1.weight"))
    ref w3 = w.get(prefix + String(".feed_forward.w3.weight"))
    var dxg = linear_backward_dx(sg.d_gate, w1, S, ZDIM, ff, ctx)
    var dxu = linear_backward_dx(sg.d_up, w3, S, ZDIM, ff, ctx)
    return add(dxg, dxu, ctx)


# ══════════════════════════════════════════════════════════════════════════════
# FROZEN refiner block. Two variants (zimage_dit.mojo::_block):
#   modulated   (noise_refiner): uses adaln modulation (scale/gate).
#   unmodulated (context_refiner): plain residual.
# All weights FROZEN → dx-only backward. We save the modulation vectors + the attn
# / ff sub-saves. (The 30 MAIN blocks are NOT here — they go through the LoRA stack.)
# ══════════════════════════════════════════════════════════════════════════════
struct _RefinerSaves(Movable):
    var x_in: Tensor
    var modulated: Bool
    var scale_msa: Tensor   # [dim] (1+scale) — modulated only (else empty)
    var gate_msa: Tensor    # [dim] tanh(gate)
    var scale_mlp: Tensor
    var gate_mlp: Tensor
    # Hold the attn/ff sub-outputs WHOLE (out + saves) — Mojo 1.0.0b1 cannot move a
    # struct-typed field out of a destructor-bearing parent, so we store the whole
    # _AttnOut/_FFOut and read .out/.saves by-ref in backward.
    var attn_full: _AttnOut  # .out = to_out output [S,dim] (pre norm2); .saves = _AttnSaves
    var x_mid: Tensor       # after attn residual
    var ff_full: _FFOut      # .out = w2 output [S,dim] (pre norm2); .saves = _FFSaves
    def __init__(
        out self, var x_in: Tensor, modulated: Bool,
        var scale_msa: Tensor, var gate_msa: Tensor, var scale_mlp: Tensor, var gate_mlp: Tensor,
        var attn_full: _AttnOut, var x_mid: Tensor, var ff_full: _FFOut,
    ):
        self.x_in = x_in^; self.modulated = modulated
        self.scale_msa = scale_msa^; self.gate_msa = gate_msa^
        self.scale_mlp = scale_mlp^; self.gate_mlp = gate_mlp^
        self.attn_full = attn_full^; self.x_mid = x_mid^
        self.ff_full = ff_full^

struct _RefinerOut(Movable):
    var out: Tensor
    var saves: _RefinerSaves
    def __init__(out self, var out: Tensor, var saves: _RefinerSaves):
        self.out = out^
        self.saves = saves^
    # Consuming accessor: return the residual .out, drop the (forward-only) .saves.
    # deinit self consumes the whole value so the struct-typed .saves field is
    # auto-destroyed (partial field moves of struct fields are illegal in 1.0.0b1).
    def take_out(deinit self) -> Tensor:
        return self.out^

# modulation chunk: mod = adaLN_modulation.0(adaln) [1,4*dim] -> 4 chunks of [dim].
# Returned as a 4-field struct so each chunk can be ^-transferred (Tensor is
# move-only → cannot be copied out of a List by index).
struct _Mod4(Movable):
    var scale_msa: Tensor
    var gate_msa: Tensor
    var scale_mlp: Tensor
    var gate_mlp: Tensor
    def __init__(out self, var scale_msa: Tensor, var gate_msa: Tensor, var scale_mlp: Tensor, var gate_mlp: Tensor):
        self.scale_msa = scale_msa^; self.gate_msa = gate_msa^
        self.scale_mlp = scale_mlp^; self.gate_mlp = gate_mlp^

def _refiner_modulation(
    w: ZImageWeights, adaln: Tensor, prefix: String, ctx: DeviceContext
) raises -> _Mod4:
    ref mw = w.get(prefix + String(".adaLN_modulation.0.weight"))
    ref mb = w.get(prefix + String(".adaLN_modulation.0.bias"))
    var mod = linear(adaln, mw, Optional[Tensor](_clone(mb, ctx)), ctx)  # [1, 4*dim]
    var flat = reshape(mod, _sh(4 * ZDIM), ctx)
    var smsa = add_scalar(slice(flat, 0, 0 * ZDIM, ZDIM, ctx), Float32(1.0), ctx)
    var gmsa = tanh_op(slice(flat, 0, 1 * ZDIM, ZDIM, ctx), ctx)
    var smlp = add_scalar(slice(flat, 0, 2 * ZDIM, ZDIM, ctx), Float32(1.0), ctx)
    var gmlp = tanh_op(slice(flat, 0, 3 * ZDIM, ZDIM, ctx), ctx)
    return _Mod4(smsa^, gmsa^, smlp^, gmlp^)

def _frozen_refiner_block[S: Int](
    x: Tensor, cos: Tensor, sin: Tensor, adaln: Optional[Tensor],
    w: ZImageWeights, prefix: String, ctx: DeviceContext,
) raises -> _RefinerOut:
    ref n1 = w.get(prefix + String(".attention_norm1.weight"))
    ref n2 = w.get(prefix + String(".attention_norm2.weight"))
    ref fn1 = w.get(prefix + String(".ffn_norm1.weight"))
    ref fn2 = w.get(prefix + String(".ffn_norm2.weight"))
    if adaln:
        # Keep m WHOLE: read its chunks by-ref (mul borrows) and clone into the saves.
        # Moving fields out of a destructor-bearing struct is illegal in Mojo 1.0.0b1.
        var m = _refiner_modulation(w, adaln.value(), prefix, ctx)
        # attention branch
        var xn1 = rms_norm(x, n1, ZEPS, ctx)
        var xn1s = mul(xn1, m.scale_msa, ctx)
        var ao = _frozen_attention[S](xn1s, cos, sin, w, prefix, ctx)
        var attn_n2 = rms_norm(ao.out, n2, ZEPS, ctx)
        var gated = mul(attn_n2, m.gate_msa, ctx)
        var x_mid = add(x, gated, ctx)
        # mlp branch
        var xfn1 = rms_norm(x_mid, fn1, ZEPS, ctx)
        var xfn1s = mul(xfn1, m.scale_mlp, ctx)
        var fo = _frozen_feed_forward[S](xfn1s, w, prefix, ctx)
        var ff_n2 = rms_norm(fo.out, fn2, ZEPS, ctx)
        var gated_ff = mul(ff_n2, m.gate_mlp, ctx)
        var out = add(x_mid, gated_ff, ctx)
        # Store ao/fo WHOLE in the saves (move the whole struct in — partial field
        # moves of struct-typed fields are illegal in Mojo 1.0.0b1).
        var x_in_c = _clone(x, ctx)
        var x_mid_c = _clone(x_mid, ctx)
        var sv = _RefinerSaves(
            x_in_c^, True, _clone(m.scale_msa, ctx), _clone(m.gate_msa, ctx),
            _clone(m.scale_mlp, ctx), _clone(m.gate_mlp, ctx),
            ao^, x_mid_c^, fo^,
        )
        return _RefinerOut(out^, sv^)
    else:
        var xn1 = rms_norm(x, n1, ZEPS, ctx)
        var ao = _frozen_attention[S](xn1, cos, sin, w, prefix, ctx)
        var attn_n2 = rms_norm(ao.out, n2, ZEPS, ctx)
        var x_mid = add(x, attn_n2, ctx)
        var xfn1 = rms_norm(x_mid, fn1, ZEPS, ctx)
        var fo = _frozen_feed_forward[S](xfn1, w, prefix, ctx)
        var ff_n2 = rms_norm(fo.out, fn2, ZEPS, ctx)
        var out = add(x_mid, ff_n2, ctx)
        var empty = zeros_device(_sh(1), STDtype.BF16, ctx)
        var e2 = zeros_device(_sh(1), STDtype.BF16, ctx)
        var e3 = zeros_device(_sh(1), STDtype.BF16, ctx)
        var e4 = zeros_device(_sh(1), STDtype.BF16, ctx)
        var x_in_c = _clone(x, ctx)
        var x_mid_c = _clone(x_mid, ctx)
        var sv = _RefinerSaves(
            x_in_c^, False, empty^, e2^, e3^, e4^,
            ao^, x_mid_c^, fo^,
        )
        return _RefinerOut(out^, sv^)

# dx-only refiner backward. d_out [S,dim] → d_x [S,dim].
def _frozen_refiner_block_backward[S: Int](
    d_out: Tensor, saves: _RefinerSaves, cos: Tensor, sin: Tensor,
    w: ZImageWeights, prefix: String, ctx: DeviceContext,
) raises -> Tensor:
    ref n1 = w.get(prefix + String(".attention_norm1.weight"))
    ref n2 = w.get(prefix + String(".attention_norm2.weight"))
    ref fn1 = w.get(prefix + String(".ffn_norm1.weight"))
    ref fn2 = w.get(prefix + String(".ffn_norm2.weight"))
    if saves.modulated:
        # out = x_mid + gate_mlp * ff_n2
        var d_x_mid = _clone(d_out, ctx)
        var d_gated_ff = _clone(d_out, ctx)
        var d_ff_n2 = mul(d_gated_ff, saves.gate_mlp, ctx)
        var d_ff_out = rms_norm_backward_dx(d_ff_n2, saves.ff_full.out, fn2, ZEPS, ctx)
        var d_xfn1s = _frozen_feed_forward_backward[S](d_ff_out, saves.ff_full.saves, w, prefix, ctx)
        var d_xfn1 = mul(d_xfn1s, saves.scale_mlp, ctx)
        var d_xmid_mlp = rms_norm_backward_dx(d_xfn1, saves.x_mid, fn1, ZEPS, ctx)
        d_x_mid = add(d_x_mid, d_xmid_mlp, ctx)
        # x_mid = x + gate_msa * attn_n2
        var d_x = _clone(d_x_mid, ctx)
        var d_attn_n2 = mul(d_x_mid, saves.gate_msa, ctx)
        var d_attn_out = rms_norm_backward_dx(d_attn_n2, saves.attn_full.out, n2, ZEPS, ctx)
        var d_xn1s = _frozen_attention_backward[S](d_attn_out, saves.attn_full.saves, cos, sin, w, prefix, ctx)
        var d_xn1 = mul(d_xn1s, saves.scale_msa, ctx)
        var d_x_attn = rms_norm_backward_dx(d_xn1, saves.x_in, n1, ZEPS, ctx)
        return add(d_x, d_x_attn, ctx)
    else:
        # out = x_mid + ff_n2
        var d_x_mid = _clone(d_out, ctx)
        var d_ff_n2 = _clone(d_out, ctx)
        var d_ff_out = rms_norm_backward_dx(d_ff_n2, saves.ff_full.out, fn2, ZEPS, ctx)
        var d_xfn1 = _frozen_feed_forward_backward[S](d_ff_out, saves.ff_full.saves, w, prefix, ctx)
        var d_xmid_mlp = rms_norm_backward_dx(d_xfn1, saves.x_mid, fn1, ZEPS, ctx)
        d_x_mid = add(d_x_mid, d_xmid_mlp, ctx)
        # x_mid = x + attn_n2
        var d_x = _clone(d_x_mid, ctx)
        var d_attn_n2 = _clone(d_x_mid, ctx)
        var d_attn_out = rms_norm_backward_dx(d_attn_n2, saves.attn_full.out, n2, ZEPS, ctx)
        var d_xn1 = _frozen_attention_backward[S](d_attn_out, saves.attn_full.saves, cos, sin, w, prefix, ctx)
        var d_x_attn = rms_norm_backward_dx(d_xn1, saves.x_in, n1, ZEPS, ctx)
        return add(d_x, d_x_attn, ctx)


# ══════════════════════════════════════════════════════════════════════════════
# FROZEN final layer (zimage_dit.mojo::_final_layer). x [1,S,dim] →
#   scale = 1 + Linear(SiLU(adaln))   [1,1,dim]
#   xn    = LayerNorm(no-affine, eps 1e-6)(x)
#   out   = Linear(xn * scale)        [1,S,64]
# Frozen base weights → dx-only backward (adaln grad discarded; it feeds only the
# frozen MLP). We save xn, scale, the silu act, and the linear input.
# ══════════════════════════════════════════════════════════════════════════════
struct _FinalSaves(Movable):
    var x_in: Tensor      # [S,dim] (LN input, flattened)
    var scale: Tensor     # [1,dim] (1+...) broadcast
    var xn: Tensor        # LayerNorm(x) [S,dim]
    var xs: Tensor        # xn*scale [S,dim] (linear input)
    var ones_w: Tensor    # LN ones weight [dim] (for backward)
    def __init__(out self, var x_in: Tensor, var scale: Tensor, var xn: Tensor, var xs: Tensor, var ones_w: Tensor):
        self.x_in = x_in^; self.scale = scale^; self.xn = xn^; self.xs = xs^; self.ones_w = ones_w^

struct _FinalOut(Movable):
    var out: Tensor      # [S,64]
    var saves: _FinalSaves
    def __init__(out self, var out: Tensor, var saves: _FinalSaves):
        self.out = out^
        self.saves = saves^

def _frozen_final_layer[S: Int](
    x: Tensor, adaln: Tensor, w: ZImageWeights, ctx: DeviceContext
) raises -> _FinalOut:
    # x arrives [S,dim] (flattened). scale = 1 + Linear(SiLU(adaln)) [1,dim].
    var c_silu = silu(adaln, ctx)
    ref mw = w.get(String("all_final_layer.2-1.adaLN_modulation.1.weight"))
    ref mb = w.get(String("all_final_layer.2-1.adaLN_modulation.1.bias"))
    var scale = add_scalar(linear(c_silu, mw, Optional[Tensor](_clone(mb, ctx)), ctx), Float32(1.0), ctx)  # [1,dim]
    # LayerNorm no-affine eps 1e-6: ones weight / zeros bias.
    var ones = List[Float32]()
    var zeros = List[Float32]()
    for _ in range(ZDIM):
        ones.append(Float32(1.0)); zeros.append(Float32(0.0))
    var dtype = x.dtype()
    var w_ones = Tensor.from_host(ones, _sh(ZDIM), dtype, ctx)
    var b_zero = Tensor.from_host(zeros, _sh(ZDIM), dtype, ctx)
    var xn = layer_norm(x, w_ones, b_zero, ZFINAL_EPS, ctx)   # [S,dim]
    var xs = mul(xn, scale, ctx)                              # broadcast [1,dim]
    ref lw = w.get(String("all_final_layer.2-1.linear.weight"))
    ref lb = w.get(String("all_final_layer.2-1.linear.bias"))
    var out = linear(xs, lw, Optional[Tensor](_clone(lb, ctx)), ctx)  # [S,64]
    var sv = _FinalSaves(_clone(x, ctx), _clone(scale, ctx), _clone(xn, ctx), _clone(xs, ctx), _clone(w_ones, ctx))
    return _FinalOut(out^, sv^)

def _frozen_final_layer_backward[S: Int](
    d_out: Tensor, saves: _FinalSaves, w: ZImageWeights, ctx: DeviceContext
) raises -> Tensor:
    # out = Linear(xs) ; xs = xn*scale ; xn = LN(x). dx-only (lin/LN frozen; scale
    # frozen → discard d_scale, it feeds only the frozen adaLN MLP).
    ref lw = w.get(String("all_final_layer.2-1.linear.weight"))
    var d_xs = linear_backward_dx(d_out, lw, S, ZDIM, ZOUT_DIM, ctx)   # [S,dim]
    var d_xn = mul(d_xs, saves.scale, ctx)                             # broadcast [1,dim]
    return layer_norm_backward_dx(d_xn, saves.x_in, saves.ones_w, ZFINAL_EPS, ctx)


# ══════════════════════════════════════════════════════════════════════════════
# The full saved-for-backward bundle. Comptime-known sequence lengths let the
# backward redispatch the same [S] block/attn helpers. Stores per-stage saves +
# the main-stack forward checkpoint.
# ══════════════════════════════════════════════════════════════════════════════
struct ZImageFullSaved(Movable):
    # MINIMAL backward state: the backward path is LoRA-only and STOPS after the
    # main-block stack (the embedders / refiners are FROZEN, so propagating d_x
    # through them trains nothing). It therefore consumes ONLY these four fields:
    #   adaln, uni_cos, uni_sin → re-feed the verified zimage_stack_backward;
    #   stack_fwd               → the 30-block recompute checkpoints;
    #   final_saves             → the frozen final-layer dx-backward.
    # The previously-stored refiner saves + image/caption rope tables + the
    # comptime host scalars (img_tokens/img_padded/cap_padded/unified_len, all
    # recomputed comptime in backward from HL/WL/CAPLEN) were DEAD — never read by
    # zimage_backward_full_lora — so they are dropped to cut forward VRAM/clone
    # cost (the 4 refiner _AttnSaves/_FFSaves bundles were the largest waste).
    var adaln: Tensor                 # [1,256]
    var uni_cos: Tensor               # unified rope cos [unified_len*H, Dh/2]
    var uni_sin: Tensor
    var stack_fwd: ZImageStackForward        # 30 main blocks (LoRA), checkpoints
    var final_full: _FinalOut                # whole final-layer out+saves (backward reads .saves)

    def __init__(
        out self, var adaln: Tensor, var uni_cos: Tensor, var uni_sin: Tensor,
        var stack_fwd: ZImageStackForward, var final_full: _FinalOut,
    ):
        self.adaln = adaln^; self.uni_cos = uni_cos^; self.uni_sin = uni_sin^
        self.stack_fwd = stack_fwd^; self.final_full = final_full^


struct ZImageForwardOut(Movable):
    var velocity: Tensor       # [1,16,HL,WL]
    var saved: ZImageFullSaved
    def __init__(out self, var velocity: Tensor, var saved: ZImageFullSaved):
        self.velocity = velocity^
        self.saved = saved^


struct ZImageInferCache(Movable):
    var x_rope: _RopePair       # image-token RoPE, [img_padded*H,Dh/2]
    var cap_seq: Tensor         # cap embedding after context_refiner, [cap_padded,D]
    var uni_rope: _RopePair     # concat([x_rope, cap_rope]) RoPE for main stack

    def __init__(
        out self,
        var x_rope: _RopePair,
        var cap_seq: Tensor,
        var uni_rope: _RopePair,
    ):
        self.x_rope = x_rope^
        self.cap_seq = cap_seq^
        self.uni_rope = uni_rope^


def prepare_zimage_infer_cache[HL: Int, WL: Int, CAPLEN: Int](
    cap_feats: Tensor,
    w: ZImageWeights,
    ctx: DeviceContext,
) raises -> ZImageInferCache:
    comptime img_tokens = (HL // ZPATCH) * (WL // ZPATCH)
    comptime img_pad = (-img_tokens) % 32
    comptime img_padded = img_tokens + img_pad
    comptime cap_pad = (-CAPLEN) % 32
    comptime cap_padded = CAPLEN + cap_pad

    var dtype = cap_feats.dtype()

    # The caption stream and RoPE tables are prompt/resolution constants in the
    # sampler. Build them once, not once per denoise step.
    var ce = _cap_embedder(w, cap_feats, ctx)
    var cap_pad_t = _pad_rows(ce.out, CAPLEN, cap_padded, ctx)
    var cap_seq = _apply_pad_token(cap_pad_t, w, String("cap_pad_token"), CAPLEN, ctx)

    var cap_pos = List[List[Int]]()
    for i in range(cap_padded):
        var pl = List[Int](); pl.append(i + 1); pl.append(0); pl.append(0)
        cap_pos.append(pl^)

    var x_pos = List[List[Int]]()
    var ht = HL // ZPATCH
    var wt = WL // ZPATCH
    var x0 = cap_padded + 1
    for ih in range(ht):
        for iw in range(wt):
            var pl = List[Int](); pl.append(x0); pl.append(ih); pl.append(iw)
            x_pos.append(pl^)
    for _ in range(img_pad):
        var pl = List[Int](); pl.append(0); pl.append(0); pl.append(0)
        x_pos.append(pl^)

    var x_rope = _build_rope(x_pos, dtype, ctx)
    var cap_rope = _build_rope(cap_pos, dtype, ctx)

    for i in range(ZN_REFINER):
        var pre = String("context_refiner.") + String(i)
        var ro = _frozen_refiner_block[cap_padded](
            cap_seq, cap_rope.cos, cap_rope.sin, None, w, pre, ctx)
        cap_seq = ro^.take_out()

    var uni_pos = List[List[Int]]()
    for i in range(len(x_pos)):
        uni_pos.append(x_pos[i].copy())
    for i in range(len(cap_pos)):
        uni_pos.append(cap_pos[i].copy())
    var uni_rope = _build_rope(uni_pos, dtype, ctx)

    return ZImageInferCache(x_rope^, cap_seq^, uni_rope^)



# ── pad helpers (repeat-last-row) borrowed from zimage_dit.mojo::_pad_rows. The
# comptime-exact path uses CAPLEN==real caption length, so padding rows are only
# the model's own mult-of-32 slack; pad rows repeat the last real row then are
# overwritten by the learned pad token. We keep the SAME numeric path. ──────────
def _pad_rows(x: Tensor, n: Int, total: Int, ctx: DeviceContext) raises -> Tensor:
    if total == n:
        return _clone(x, ctx)
    var dim = x.shape()[len(x.shape()) - 1]
    var last = slice(x, 0, n - 1, 1, ctx)   # [1,dim]
    var padblock = _clone(last, ctx)
    for _ in range(total - n - 1):
        padblock = concat(0, ctx, padblock, last)
    return concat(0, ctx, x, padblock)

# substitute the learned pad_token into rows [real_len, total) of [total,dim].
def _apply_pad_token(
    feat: Tensor, w: ZImageWeights, pad_key: String, real_len: Int, ctx: DeviceContext
) raises -> Tensor:
    var total = feat.shape()[0]
    if real_len >= total:
        return _clone(feat, ctx)
    # keep [0,real_len) of feat; concat the learned pad token repeated (total-real).
    ref pad = w.get(pad_key)                          # [1,dim] or [dim]
    var dim = feat.shape()[1]
    if pad.numel() != dim:
        raise Error(
            String("_apply_pad_token: pad token '") + pad_key
            + String("' numel ") + String(pad.numel())
            + String(" != feature dim ") + String(dim)
        )
    var pad_row = reshape(_clone(pad, ctx), _sh(1, dim), ctx)
    var keep = slice(feat, 0, 0, real_len, ctx)       # [real_len,dim]
    var block = _clone(pad_row, ctx)
    for _ in range(total - real_len - 1):
        block = concat(0, ctx, block, pad_row)
    return concat(0, ctx, keep, block)


# ══════════════════════════════════════════════════════════════════════════════
# FULL FORWARD — latent [1,16,HL,WL] → velocity [1,16,HL,WL], saving everything
# the backward needs. Comptime-exact caption path (real_caplen == CAPLEN).
#
# Borrowed from zimage_dit.mojo::_forward_impl. The 30 MAIN blocks run through the
# verified LoRA stack (zimage_stack_forward); everything else (embedders, refiners,
# final, patchify) is FROZEN base and runs here.
# ══════════════════════════════════════════════════════════════════════════════
def zimage_forward_full_lora[HL: Int, WL: Int, CAPLEN: Int](
    scaled_noisy_latent: Tensor,    # [1,16,HL,WL] BF16
    t_model: Float32,
    cap_feats: Tensor,              # [CAPLEN, cap_feat_dim] BF16
    w: ZImageWeights,
    loras: ZImageLoraSet,
    ctx: DeviceContext,
) raises -> ZImageForwardOut:
    comptime img_tokens = (HL // ZPATCH) * (WL // ZPATCH)
    comptime img_pad = (-img_tokens) % 32
    comptime img_padded = img_tokens + img_pad
    comptime cap_pad = (-CAPLEN) % 32
    comptime cap_padded = CAPLEN + cap_pad
    comptime unified_len = img_padded + cap_padded

    var dtype = scaled_noisy_latent.dtype()

    # adaln = t_embedder(t * t_scale)  [1,256]
    var adaln = _t_embedder(w, t_model, ctx)

    # patchify image → embed → pad → x_pad_token
    var xp = _patchify[HL, WL](scaled_noisy_latent, ctx)        # [1,img_tokens,64]
    ref xw = w.get(String("all_x_embedder.2-1.weight"))
    ref xb = w.get(String("all_x_embedder.2-1.bias"))
    var xe = linear(xp, xw, Optional[Tensor](_clone(xb, ctx)), ctx)   # [1,img_tokens,dim]
    var xe_flat = reshape(xe, _sh(img_tokens, ZDIM), ctx)
    var xe_pad = _pad_rows(xe_flat, img_tokens, img_padded, ctx)      # [img_padded,dim]
    var x_seq = _apply_pad_token(xe_pad, w, String("x_pad_token"), img_tokens, ctx)  # [img_padded,dim]

    # cap embed → pad → cap_pad_token
    var ce = _cap_embedder(w, cap_feats, ctx)                  # [CAPLEN,dim]
    var cap_pad_t = _pad_rows(ce.out, CAPLEN, cap_padded, ctx) # [cap_padded,dim]
    var cap_seq = _apply_pad_token(cap_pad_t, w, String("cap_pad_token"), CAPLEN, ctx)  # [cap_padded,dim]

    # ── RoPE positions ──
    var cap_pos = List[List[Int]]()
    for i in range(cap_padded):
        var pl = List[Int](); pl.append(i + 1); pl.append(0); pl.append(0)
        cap_pos.append(pl^)
    var x_pos = List[List[Int]]()
    var ht = HL // ZPATCH
    var wt = WL // ZPATCH
    var x0 = cap_padded + 1
    for ih in range(ht):
        for iw in range(wt):
            var pl = List[Int](); pl.append(x0); pl.append(ih); pl.append(iw)
            x_pos.append(pl^)
    for _ in range(img_pad):
        var pl = List[Int](); pl.append(0); pl.append(0); pl.append(0)
        x_pos.append(pl^)
    var x_rope = _build_rope(x_pos, dtype, ctx)
    var cap_rope = _build_rope(cap_pos, dtype, ctx)

    # ── noise refiner (modulated) on x_seq ──
    # FROZEN, LoRA-free → its backward is never run; we keep ONLY the residual
    # stream (ro.out) and let ro.saves drop immediately (forward-only scratch).
    for i in range(ZN_REFINER):
        var pre = String("noise_refiner.") + String(i)
        var ro = _frozen_refiner_block[img_padded](
            x_seq, x_rope.cos, x_rope.sin, Optional[Tensor](_clone(adaln, ctx)), w, pre, ctx)
        x_seq = ro^.take_out()   # keep residual; drop forward-only refiner saves

    # ── context refiner (unmodulated) on cap_seq ──
    for i in range(ZN_REFINER):
        var pre = String("context_refiner.") + String(i)
        var ro = _frozen_refiner_block[cap_padded](
            cap_seq, cap_rope.cos, cap_rope.sin, None, w, pre, ctx)
        cap_seq = ro^.take_out()   # keep residual; drop forward-only refiner saves

    # ── unified = concat([x, cap]) along the token axis ──
    var unified = concat(0, ctx, x_seq, cap_seq)              # [unified_len, dim]
    var uni_pos = List[List[Int]]()
    for i in range(len(x_pos)):
        uni_pos.append(x_pos[i].copy())
    for i in range(len(cap_pos)):
        uni_pos.append(cap_pos[i].copy())
    var uni_rope = _build_rope(uni_pos, dtype, ctx)

    # ── 30 MAIN blocks via the verified LoRA stack ──
    var stack_fwd = zimage_stack_forward[unified_len](
        unified, adaln, uni_rope.cos, uni_rope.sin, w, loras, ctx)

    # ── final layer on the FULL unified sequence ──
    var fo = _frozen_final_layer[unified_len](stack_fwd.out, adaln, w, ctx)  # [unified_len,64]

    # ── extract image tokens, unpatchify → velocity ──
    # Borrow fo.out (slice reads it) — do NOT move fo apart; fo is stored WHOLE in
    # ZImageFullSaved below (struct-typed field partial move is illegal in 1.0.0b1).
    var x_final = slice(fo.out, 0, 0, img_tokens, ctx)        # [img_tokens,64]
    var x_final_seq = reshape(x_final, _sh(1, img_tokens, ZOUT_DIM), ctx)
    var velocity = _unpatchify[HL, WL](x_final_seq, ctx)      # [1,16,HL,WL]

    # Only the four backward-consumed fields are retained (see ZImageFullSaved):
    # the refiner saves (nr_saves/cr_saves) and the image/caption rope tables
    # (x_rope/cap_rope) are forward-only scratch — they are dropped here.
    # fo is stored whole; backward reads saved.final_full.saves.
    var saved = ZImageFullSaved(
        _clone(adaln, ctx), _clone(uni_rope.cos, ctx), _clone(uni_rope.sin, ctx),
        stack_fwd^, fo^,
    )
    return ZImageForwardOut(velocity^, saved^)


# ══════════════════════════════════════════════════════════════════════════════
# NO-GRAD INFERENCE FORWARD — latent [1,16,HL,WL] → velocity [1,16,HL,WL] ONLY.
#
# 1:1 numeric port of Serenity's torch.no_grad() sampling path
# (modules/modelSampler/ZImageSampler.py:97-108: transformer(...) under inference).
# IDENTICAL block math to zimage_forward_full_lora, but:
#   • does NOT build/return ZImageFullSaved (no saved-for-backward bundle);
#   • does NOT save per-block activations or recompute-checkpoint inputs
#     (uses zimage_stack_forward_infer → activation-free 30-block walk);
#   • the refiners already drop their saves via take_out (forward-only scratch);
#   • the final layer's _FinalSaves drop when `fo` goes out of scope (one layer).
# LoRA overlay still applies (B may be 0). Returns ONLY the velocity tensor.
# ══════════════════════════════════════════════════════════════════════════════
def zimage_forward_full_infer[HL: Int, WL: Int, CAPLEN: Int](
    scaled_noisy_latent: Tensor,    # [1,16,HL,WL] BF16
    t_model: Float32,
    cap_feats: Tensor,              # [CAPLEN, cap_feat_dim] BF16
    w: ZImageWeights,
    loras: ZImageLoraSet,
    ctx: DeviceContext,
) raises -> Tensor:
    comptime img_tokens = (HL // ZPATCH) * (WL // ZPATCH)
    comptime img_pad = (-img_tokens) % 32
    comptime img_padded = img_tokens + img_pad
    comptime cap_pad = (-CAPLEN) % 32
    comptime cap_padded = CAPLEN + cap_pad
    comptime unified_len = img_padded + cap_padded

    var dtype = scaled_noisy_latent.dtype()

    # adaln = t_embedder(t * t_scale)  [1,256]
    var adaln = _t_embedder(w, t_model, ctx)

    # patchify image → embed → pad → x_pad_token
    var xp = _patchify[HL, WL](scaled_noisy_latent, ctx)
    ref xw = w.get(String("all_x_embedder.2-1.weight"))
    ref xb = w.get(String("all_x_embedder.2-1.bias"))
    var xe = linear(xp, xw, Optional[Tensor](_clone(xb, ctx)), ctx)
    var xe_flat = reshape(xe, _sh(img_tokens, ZDIM), ctx)
    var xe_pad = _pad_rows(xe_flat, img_tokens, img_padded, ctx)
    var x_seq = _apply_pad_token(xe_pad, w, String("x_pad_token"), img_tokens, ctx)

    # cap embed → pad → cap_pad_token
    var ce = _cap_embedder(w, cap_feats, ctx)
    var cap_pad_t = _pad_rows(ce.out, CAPLEN, cap_padded, ctx)
    var cap_seq = _apply_pad_token(cap_pad_t, w, String("cap_pad_token"), CAPLEN, ctx)

    # ── RoPE positions (identical to the training forward) ──
    var cap_pos = List[List[Int]]()
    for i in range(cap_padded):
        var pl = List[Int](); pl.append(i + 1); pl.append(0); pl.append(0)
        cap_pos.append(pl^)
    var x_pos = List[List[Int]]()
    var ht = HL // ZPATCH
    var wt = WL // ZPATCH
    var x0 = cap_padded + 1
    for ih in range(ht):
        for iw in range(wt):
            var pl = List[Int](); pl.append(x0); pl.append(ih); pl.append(iw)
            x_pos.append(pl^)
    for _ in range(img_pad):
        var pl = List[Int](); pl.append(0); pl.append(0); pl.append(0)
        x_pos.append(pl^)
    var x_rope = _build_rope(x_pos, dtype, ctx)
    var cap_rope = _build_rope(cap_pos, dtype, ctx)

    # ── noise refiner (modulated) on x_seq — saves dropped via take_out ──
    for i in range(ZN_REFINER):
        var pre = String("noise_refiner.") + String(i)
        var ro = _frozen_refiner_block[img_padded](
            x_seq, x_rope.cos, x_rope.sin, Optional[Tensor](_clone(adaln, ctx)), w, pre, ctx)
        x_seq = ro^.take_out()

    # ── context refiner (unmodulated) on cap_seq ──
    for i in range(ZN_REFINER):
        var pre = String("context_refiner.") + String(i)
        var ro = _frozen_refiner_block[cap_padded](
            cap_seq, cap_rope.cos, cap_rope.sin, None, w, pre, ctx)
        cap_seq = ro^.take_out()

    # ── unified = concat([x, cap]) ──
    var unified = concat(0, ctx, x_seq, cap_seq)
    var uni_pos = List[List[Int]]()
    for i in range(len(x_pos)):
        uni_pos.append(x_pos[i].copy())
    for i in range(len(cap_pos)):
        uni_pos.append(cap_pos[i].copy())
    var uni_rope = _build_rope(uni_pos, dtype, ctx)

    # ── 30 MAIN blocks via the activation-free LoRA stack (NO checkpoints) ──
    var stack_out = zimage_stack_forward_infer[unified_len](
        unified, adaln, uni_rope.cos, uni_rope.sin, w, loras, ctx)

    # ── final layer on the FULL unified sequence (saves drop with fo) ──
    var fo = _frozen_final_layer[unified_len](stack_out, adaln, w, ctx)

    # ── extract image tokens, unpatchify → velocity ──
    var x_final = slice(fo.out, 0, 0, img_tokens, ctx)
    var x_final_seq = reshape(x_final, _sh(1, img_tokens, ZOUT_DIM), ctx)
    var velocity = _unpatchify[HL, WL](x_final_seq, ctx)
    return velocity^


def zimage_forward_full_infer_cached[HL: Int, WL: Int, CAPLEN: Int](
    scaled_noisy_latent: Tensor,    # [1,16,HL,WL] BF16
    t_model: Float32,
    cache: ZImageInferCache,
    w: ZImageWeights,
    loras: ZImageLoraSet,
    ctx: DeviceContext,
) raises -> Tensor:
    comptime img_tokens = (HL // ZPATCH) * (WL // ZPATCH)
    comptime img_pad = (-img_tokens) % 32
    comptime img_padded = img_tokens + img_pad
    comptime cap_pad = (-CAPLEN) % 32
    comptime cap_padded = CAPLEN + cap_pad
    comptime unified_len = img_padded + cap_padded

    var adaln = _t_embedder(w, t_model, ctx)

    var xp = _patchify[HL, WL](scaled_noisy_latent, ctx)
    ref xw = w.get(String("all_x_embedder.2-1.weight"))
    ref xb = w.get(String("all_x_embedder.2-1.bias"))
    var xe = linear(xp, xw, Optional[Tensor](_clone(xb, ctx)), ctx)
    var xe_flat = reshape(xe, _sh(img_tokens, ZDIM), ctx)
    var xe_pad = _pad_rows(xe_flat, img_tokens, img_padded, ctx)
    var x_seq = _apply_pad_token(xe_pad, w, String("x_pad_token"), img_tokens, ctx)

    for i in range(ZN_REFINER):
        var pre = String("noise_refiner.") + String(i)
        var ro = _frozen_refiner_block[img_padded](
            x_seq, cache.x_rope.cos, cache.x_rope.sin,
            Optional[Tensor](_clone(adaln, ctx)), w, pre, ctx)
        x_seq = ro^.take_out()

    var unified = concat(0, ctx, x_seq, cache.cap_seq)
    var stack_out = zimage_stack_forward_infer[unified_len](
        unified, adaln, cache.uni_rope.cos, cache.uni_rope.sin, w, loras, ctx)

    var fo = _frozen_final_layer[unified_len](stack_out, adaln, w, ctx)
    var x_final = slice(fo.out, 0, 0, img_tokens, ctx)
    var x_final_seq = reshape(x_final, _sh(1, img_tokens, ZOUT_DIM), ctx)
    var velocity = _unpatchify[HL, WL](x_final_seq, ctx)
    return velocity^


# ══════════════════════════════════════════════════════════════════════════════
# FULL BACKWARD — d_velocity [1,16,HL,WL] → ZImageStackLoraGrads (the 210 d_a/d_b
# pairs). Hand-chained reverse of zimage_forward_full_lora. Reuses
# zimage_stack_backward for the 30 main LoRA blocks; the embedders / refiners /
# final / patchify are FROZEN so they backprop dx-only (their d_W is discarded).
#
# The returned grads carry NO useful d_x_in above the stack — the image/caption
# embedders are frozen, so the d_x reaching the latent does not train anything.
# Only the 210 LoRA d_a/d_b are load-bearing for the optimizer. We still chain the
# full reverse path so the main-stack d_out is exact (it depends on the final
# layer + unpatchify backward).
# ══════════════════════════════════════════════════════════════════════════════
def zimage_backward_full_lora[HL: Int, WL: Int, CAPLEN: Int](
    d_velocity: Tensor,             # [1,16,HL,WL]
    saved: ZImageFullSaved,
    w: ZImageWeights,
    loras: ZImageLoraSet,
    ctx: DeviceContext,
) raises -> ZImageStackLoraGrads:
    # Recompute the comptime sequence lengths from HL/WL/CAPLEN (identical to the
    # forward) so the [S]-parameterized backward helpers dispatch on comptime S.
    comptime img_tokens = (HL // ZPATCH) * (WL // ZPATCH)
    comptime img_pad = (-img_tokens) % 32
    comptime img_padded = img_tokens + img_pad
    comptime cap_pad = (-CAPLEN) % 32
    comptime cap_padded = CAPLEN + cap_pad
    comptime unified_len = img_padded + cap_padded

    # unpatchify backward: d_velocity [1,16,HL,WL] → d_x_final_seq [1,img_tokens,64]
    var d_x_final_seq = _unpatchify_backward[HL, WL](d_velocity, ctx)
    var d_x_final = reshape(d_x_final_seq, _sh(img_tokens, ZOUT_DIM), ctx)
    # slice backward: scatter [img_tokens,64] into [unified_len,64] (rows after
    # img_tokens were the caption tokens → zero grad from the velocity head).
    var d_uni_final = slice_backward(d_x_final, _sh(unified_len, ZOUT_DIM), 0, 0, ctx)

    # final layer backward → d_unified (grad wrt the main-stack output) [unified_len,dim]
    var d_stack_out = _frozen_final_layer_backward[unified_len](d_uni_final, saved.final_full.saves, w, ctx)

    # 30 MAIN LoRA blocks backward (verified stack). Returns the 210 d_a/d_b + the
    # load-bearing d_x_in (grad wrt the unified input to the stack).
    var stack_grads = zimage_stack_backward[unified_len](
        d_stack_out, saved.adaln, saved.uni_cos, saved.uni_sin, w, loras, saved.stack_fwd, ctx)

    # NB: the chain below (refiners/embedders) does NOT affect the LoRA grads — it
    # only produces the latent d_x, which trains nothing (frozen embedders). We
    # therefore STOP here and return the LoRA grads; running the refiner backward
    # would add cost with no trainable consumer. The d_x_in carried in stack_grads
    # is the unified-input grad (also non-load-bearing for LoRA). Returned as-is so
    # the driver reads ONLY .d_a / .d_b.
    return stack_grads^
