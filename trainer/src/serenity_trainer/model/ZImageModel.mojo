# block.mojo — ONE Z-Image NextDiT MAIN block: hand-chained forward (saving
# activations) + hand-chained backward (reverse), with LoRA on the 7 trained
# projections.  Comptime-shaped on [S, H=30, Dh=128, dim=3840].
#
# ARCHITECTURE REF (block math, read line-by-line, NOT copied):
#   mojodiffusion/serenitymojo/models/dit/zimage_dit.mojo::_block (:407-468),
#   _attention (:343-393), _feed_forward (:396-403).
# BACKWARD STRUCTURE PATTERN (hand-chain fwd ops + ops/*_backward in reverse,
#   no Tape): MOJO_AUTOGRAD_INTERNALS.md §8 (the klein single/double block).
#   We COPY THE STRUCTURE (residual fan-out, per-op reverse), not the code.
#
# The MAIN block is the MODULATED variant (adaln present). Forward (ref :423-458):
#   mod = adaLN_modulation.0(adaln)  -> chunk4 -> scale_msa,gate_msa,scale_mlp,gate_mlp
#   gate = tanh(gate) ; scale = 1 + scale
#   --- attention branch ---
#   xn1     = rms_norm(x, attention_norm1)                # [S,dim]
#   xn1s    = xn1 * scale_msa                              # broadcast [dim]
#   attn    = attention(xn1s)                              # to_q/k/v->normq/k->rope->sdpa->to_out
#   attn_n2 = rms_norm(attn, attention_norm2)
#   x       = x + gate_msa * attn_n2
#   --- mlp branch ---
#   xfn1    = rms_norm(x, ffn_norm1)
#   xfn1s   = xfn1 * scale_mlp
#   ff      = feed_forward(xfn1s)  = w2(silu(w1(xfn1s)) * w3(xfn1s))
#   ff_n2   = rms_norm(ff, ffn_norm2)
#   out     = x + gate_mlp * ff_n2
#
# LoRA TARGETS (lora_targets.mojo): to_q,to_k,to_v,to_out,w1,w3,w2. The adaLN
# Linear and all RMSNorms are FROZEN and NOT LoRA targets — so in the LoRA-train
# regime the modulation vectors (scale/gate) carry NO gradient (the adaLN MLP is
# frozen). We therefore backprop d_x THROUGH the `* scale` multiply but do NOT
# compute/propagate d_scale or d_gate (they would only feed the frozen adaLN).
# This is the same simplification the klein LoRA backward uses for shared frozen
# modulation (MOJO_AUTOGRAD_INTERNALS.md §8.3: "does NOT backprop them into the
# modulation MLP — deferred finetune phase").
#
# DTYPE: BF16 storage in/out; F32 only inside the foundation kernels. No
# persistent F32 tensor here.
#
# Tensors handled as [S, dim] (batch=1 flattened). SDPA needs [1,S,H,Dh].

from std.math import sqrt
from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

# forward ops
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.attention import sdpa_nomask, sdpa_nomask_tiled
from serenitymojo.ops.activations import swiglu
from serenitymojo.ops.tensor_algebra import (
    add, mul, mul_scalar, add_scalar, reshape, zeros_device, slice,
)
from serenitymojo.ops.unary import tanh_op
from serenitymojo.ops.cast import cast_tensor

# backward ops (hand-chained, no Tape)
from serenitymojo.ops.linalg_backward import linear_backward, linear_backward_dx
from serenitymojo.ops.norm_backward import rms_norm_backward_dx
from serenitymojo.ops.rope_struct_backward import rope_backward
from serenitymojo.ops.attention_backward import sdpa_backward
from serenitymojo.ops.loss_swiglu_backward import swiglu_backward

from serenitymojo.ops.random import randn

from serenity_trainer.module.LoRAModule import LoraAdapter, make_lora_adapter
from serenity_trainer.modelLoader.ZImageModelLoader import ZImageWeights, ZImageBlockKeys
# LoRA slot constants live in the LEAF target module (no setup dep) to break the
# former model⇄setup comptime import cycle. `LT` is the leaf, not the setup spec.
from serenity_trainer.modelSetup import zImageLoraTargets as LT


comptime TArc = ArcPointer[Tensor]
comptime LArc = ArcPointer[LoraAdapter]  # box: LoraAdapter is move-only → not a List elem


# ── Z-Image MAIN block dims (comptime) ────────────────────────────────────────
comptime ZH = 30      # n_heads
comptime ZDh = 128    # head_dim
comptime ZDIM = 3840  # dim = ZH * ZDh
comptime ZEPS = Float32(1e-5)
# SwiGLU FFN hidden dim. Real zimage_base checkpoint uses 10240 (= int(2/3 * 4*dim),
# Llama-style), NOT 4*dim=15360. The feed_forward.w1/w3 are [10240,3840] and
# w2 is [3840,10240]; LoRA out/in for those slots must match this.
comptime ZFF = 10240  # feed_forward hidden dim
comptime ZIMAGE_MATH_SDPA_BUDGET_MIB = 3072


# ── LoRA-linear forward (frozen base + LoRA), saving the LoRA `down` activation
# so the backward can form d_B without recompute. Returns (y, down). Mirrors
# adapters/lora.lora_linear_forward but hand-chained (no Tape) for the block.
#   y    = x @ Wᵀ + ((x @ Aᵀ) @ Bᵀ) * (alpha/rank)
#   down = x @ Aᵀ          (saved for d_B)
struct _LoraFwd(Movable):
    var y: Tensor
    var down: Tensor

    def __init__(out self, var y: Tensor, var down: Tensor):
        self.y = y^
        self.down = down^


def _lora_linear_fwd(
    x: Tensor, base_w: Tensor, ad: LoraAdapter, ctx: DeviceContext
) raises -> _LoraFwd:
    # frozen base path (no bias)
    var base = linear(x, base_w, None, ctx)              # [M, out]
    var down = linear(x, ad.a, None, ctx)                # [M, rank]
    var up = linear(down, ad.b, None, ctx)               # [M, out]
    var scaled = mul_scalar(up, ad.scale(), ctx)
    var y = add(base, scaled, ctx)
    return _LoraFwd(y^, down^)


# ── LoRA-linear backward. Given d_y [M,out], the saved x [M,in] and down [M,rank]:
#   d_x      = d_y @ W           (frozen base)  +  ((d_y * s) @ B) @ A   (lora)
#   d_B[out,rank] = (d_y * s)ᵀ @ down
#   d_A[rank,in]  = ((d_y * s) @ B)ᵀ @ x
# We accumulate d_A/d_B into the adapter-grad accumulators (caller-owned).
struct _LoraBwd(Movable):
    var d_x: Tensor
    var d_a: Tensor   # [rank, in]
    var d_b: Tensor   # [out, rank]

    def __init__(out self, var d_x: Tensor, var d_a: Tensor, var d_b: Tensor):
        self.d_x = d_x^
        self.d_a = d_a^
        self.d_b = d_b^


def _lora_linear_bwd(
    d_y: Tensor, x: Tensor, down: Tensor, base_w: Tensor, ad: LoraAdapter,
    M: Int, in_f: Int, out_f: Int, ctx: DeviceContext,
) raises -> _LoraBwd:
    # frozen base d_x (skip d_W: base is frozen)
    var dx_base = linear_backward_dx(d_y, base_w, M, in_f, out_f, ctx)  # [M,in]

    # LoRA path: scaled gradient s*d_y flows through up(B) then down(A).
    var dy_s = mul_scalar(d_y, ad.scale(), ctx)                # [M,out]
    # up = down @ Bᵀ : d_down = dy_s @ B ; d_B = dy_sᵀ @ down
    # (BORROW grad-struct fields — they own a destructor; field-move is illegal.)
    var up_g = linear_backward(dy_s, down, ad.b, M, ad.rank, out_f, ctx)  # d_x=d_down, d_w=d_B
    var d_b = _clone(up_g.d_w, ctx)                            # [out,rank]
    # down = x @ Aᵀ : d_x_lora = d_down @ A ; d_A = d_downᵀ @ x
    var down_g = linear_backward(up_g.d_x, x, ad.a, M, in_f, ad.rank, ctx)  # d_x, d_w=d_A
    var d_a = _clone(down_g.d_w, ctx)                          # [rank,in]
    var d_x = add(dx_base, down_g.d_x, ctx)                    # [M,in]
    return _LoraBwd(d_x^, d_a^, d_b^)


# ── per-block saved activations (BF16). Only what the backward reads. ──────────
struct ZImageBlockActs(Movable):
    var x_in: Tensor        # block input            [S,dim]
    var scale_msa: Tensor   # 1+scale (msa)          [dim]
    var gate_msa: Tensor    # tanh(gate) (msa)       [dim]
    var scale_mlp: Tensor   # 1+scale (mlp)          [dim]
    var gate_mlp: Tensor    # tanh(gate) (mlp)       [dim]
    var xn1: Tensor         # rms_norm1(x_in)        [S,dim]
    var xn1s: Tensor        # xn1*scale_msa          [S,dim]
    var q_pre_rope: Tensor  # norm_q(reshape(to_q))  [1,S,H,Dh]
    var k_pre_rope: Tensor  # norm_k(reshape(to_k))  [1,S,H,Dh]
    var q_rope: Tensor      # rope(q)                [1,S,H,Dh]
    var k_rope: Tensor      # rope(k)                [1,S,H,Dh]
    var v_bshd: Tensor      # reshape(to_v)          [1,S,H,Dh]
    var to_q_raw: Tensor    # to_q(xn1s) reshaped    [1,S,H,Dh]  (pre norm_q)
    var to_k_raw: Tensor    # to_k(xn1s) reshaped    [1,S,H,Dh]  (pre norm_k)
    var attn_out: Tensor    # to_out(attn flat)      [S,dim]
    var attn_flat: Tensor   # sdpa flat (pre to_out) [S,dim]
    var x_mid: Tensor       # x after attn residual  [S,dim]
    var xfn1: Tensor        # rms_norm(x_mid,ffn1)   [S,dim]
    var xfn1s: Tensor       # xfn1*scale_mlp         [S,dim]
    var ff_g: Tensor        # w1(xfn1s)              [S,ff]
    var ff_u: Tensor        # w3(xfn1s)              [S,ff]
    var ff_act: Tensor      # swiglu(g,u)            [S,ff]
    var ff_out: Tensor      # w2(act)                [S,dim]
    # LoRA `down` activations (for d_B), one per LoRA slot.
    var down_q: Tensor
    var down_k: Tensor
    var down_v: Tensor
    var down_out: Tensor
    var down_w1: Tensor
    var down_w3: Tensor
    var down_w2: Tensor

    def __init__(
        out self,
        var x_in: Tensor, var scale_msa: Tensor, var gate_msa: Tensor,
        var scale_mlp: Tensor, var gate_mlp: Tensor, var xn1: Tensor, var xn1s: Tensor,
        var q_pre_rope: Tensor, var k_pre_rope: Tensor, var q_rope: Tensor, var k_rope: Tensor,
        var v_bshd: Tensor, var to_q_raw: Tensor, var to_k_raw: Tensor,
        var attn_out: Tensor, var attn_flat: Tensor, var x_mid: Tensor,
        var xfn1: Tensor, var xfn1s: Tensor, var ff_g: Tensor, var ff_u: Tensor,
        var ff_act: Tensor, var ff_out: Tensor,
        var down_q: Tensor, var down_k: Tensor, var down_v: Tensor, var down_out: Tensor,
        var down_w1: Tensor, var down_w3: Tensor, var down_w2: Tensor,
    ):
        self.x_in = x_in^; self.scale_msa = scale_msa^; self.gate_msa = gate_msa^
        self.scale_mlp = scale_mlp^; self.gate_mlp = gate_mlp^; self.xn1 = xn1^; self.xn1s = xn1s^
        self.q_pre_rope = q_pre_rope^; self.k_pre_rope = k_pre_rope^
        self.q_rope = q_rope^; self.k_rope = k_rope^; self.v_bshd = v_bshd^
        self.to_q_raw = to_q_raw^; self.to_k_raw = to_k_raw^
        self.attn_out = attn_out^; self.attn_flat = attn_flat^; self.x_mid = x_mid^
        self.xfn1 = xfn1^; self.xfn1s = xfn1s^; self.ff_g = ff_g^; self.ff_u = ff_u^
        self.ff_act = ff_act^; self.ff_out = ff_out^
        self.down_q = down_q^; self.down_k = down_k^; self.down_v = down_v^; self.down_out = down_out^
        self.down_w1 = down_w1^; self.down_w3 = down_w3^; self.down_w2 = down_w2^


# ── per-block LoRA gradient accumulators (d_A/d_B for each of the 7 slots) ─────
struct ZImageBlockLoraGrads(Movable):
    var d_a: List[TArc]   # [7] each [rank, in]
    var d_b: List[TArc]   # [7] each [out, rank]

    def __init__(out self, var d_a: List[TArc], var d_b: List[TArc]):
        self.d_a = d_a^
        self.d_b = d_b^


# ── block forward ─────────────────────────────────────────────────────────────
# x: [S, dim] BF16. adaln: [1, adaln_embed] (256). cos/sin: rope tables
# [S*H, Dh/2]. `w`: frozen weight store. `keys`: block key bundle. `loras`: the
# 7 LoRA adapters for THIS block, in lora_targets slot order. Returns (out, acts).
struct ZImageBlockOut(Movable):
    var out: Tensor
    var acts: ZImageBlockActs

    def __init__(out self, var out: Tensor, var acts: ZImageBlockActs):
        self.out = out^
        self.acts = acts^

    # Consuming accessor: return .out, drop the (discarded) .acts. deinit self
    # consumes the whole value so the struct-typed .acts field is auto-destroyed
    # (partial moves of struct-typed fields are illegal in Mojo 1.0.0b1).
    def take_out(deinit self) -> Tensor:
        return self.out^


def zimage_block_forward[S: Int](
    x: Tensor,
    adaln: Tensor,
    cos: Tensor,
    sin: Tensor,
    w: ZImageWeights,
    keys: ZImageBlockKeys,
    loras: List[LArc],
    ctx: DeviceContext,
) raises -> ZImageBlockOut:
    var scale = Float32(1.0) / sqrt(Float32(ZDh))

    # ── modulation: mod = adaLN_modulation.0(adaln) -> [1, 4*dim] -> chunk4 ────
    # (ref :425-442). adaln_b present. scale = 1+chunk ; gate = tanh(chunk).
    ref mw = w.get(keys.adaln_w())
    ref mb = w.get(keys.adaln_b())
    var mod = linear(adaln, mw, Optional[Tensor](_clone(mb, ctx)), ctx)   # [1, 4*dim]
    var mod_flat = reshape(mod, _shape1(4 * ZDIM), ctx)                   # [4*dim]
    var scale_msa = _add_scalar_slice(mod_flat, 0 * ZDIM, ZDIM, 1.0, ctx) # 1+scale
    var gate_msa_raw = _slice1(mod_flat, 1 * ZDIM, ZDIM, ctx)
    var scale_mlp = _add_scalar_slice(mod_flat, 2 * ZDIM, ZDIM, 1.0, ctx)
    var gate_mlp_raw = _slice1(mod_flat, 3 * ZDIM, ZDIM, ctx)
    var gate_msa = tanh_op(gate_msa_raw, ctx)
    var gate_mlp = tanh_op(gate_mlp_raw, ctx)

    # ── attention branch ──────────────────────────────────────────────────────
    ref n1 = w.get(keys.attn_norm1())
    var xn1 = rms_norm(x, n1, ZEPS, ctx)                 # [S,dim]
    var xn1s = mul(xn1, scale_msa, ctx)                  # broadcast [dim]

    # to_q/k/v (+LoRA). [S,dim] -> [S,dim]
    ref qw = w.get(keys.to_q())
    ref kw = w.get(keys.to_k())
    ref vw = w.get(keys.to_v())
    # (BORROW _LoraFwd fields — struct owns a destructor; clone what we keep.)
    var qf = _lora_linear_fwd(xn1s, qw, loras[LT.LORA_TO_Q][], ctx)
    var q4 = reshape(qf.y, _shape4(1, S, ZH, ZDh), ctx)
    var down_q = _clone(qf.down, ctx)
    var kf = _lora_linear_fwd(xn1s, kw, loras[LT.LORA_TO_K][], ctx)
    var k4 = reshape(kf.y, _shape4(1, S, ZH, ZDh), ctx)
    var down_k = _clone(kf.down, ctx)
    var vf = _lora_linear_fwd(xn1s, vw, loras[LT.LORA_TO_V][], ctx)
    var v4 = reshape(vf.y, _shape4(1, S, ZH, ZDh), ctx)
    var down_v = _clone(vf.down, ctx)
    var to_q_raw = _clone(q4, ctx)
    var to_k_raw = _clone(k4, ctx)

    # per-head RMSNorm over Dh (norm_q/norm_k, eps 1e-5)  (ref :374-377)
    ref qn = w.get(keys.norm_q())
    ref kn = w.get(keys.norm_k())
    var qn4 = rms_norm(q4, qn, ZEPS, ctx)
    var kn4 = rms_norm(k4, kn, ZEPS, ctx)
    var q_pre_rope = _clone(qn4, ctx)
    var k_pre_rope = _clone(kn4, ctx)

    # RoPE interleaved on q,k  (ref :380-381)
    var qr = rope_interleaved(qn4, cos, sin, ctx)
    var kr = rope_interleaved(kn4, cos, sin, ctx)

    # SDPA full attention (no mask)  (ref :384)
    var attn4 = sdpa_nomask[1, S, ZH, ZDh](qr, kr, v4, scale, ctx)  # [1,S,H,Dh]
    var attn_flat = reshape(attn4, _shape2(S, ZDIM), ctx)          # [S,dim]

    # to_out.0 (+LoRA)  (ref :392-393)
    ref ow = w.get(keys.to_out())
    var of = _lora_linear_fwd(attn_flat, ow, loras[LT.LORA_TO_OUT][], ctx)
    var attn_out = _clone(of.y, ctx)                     # [S,dim]
    var down_out = _clone(of.down, ctx)

    # attn_n2 = rms_norm(attn_out, attention_norm2) ; x = x + gate_msa * attn_n2
    ref n2 = w.get(keys.attn_norm2())
    var attn_n2 = rms_norm(attn_out, n2, ZEPS, ctx)
    var gated_attn = mul(attn_n2, gate_msa, ctx)          # broadcast
    var x_mid = add(x, gated_attn, ctx)                   # [S,dim]

    # ── mlp branch ────────────────────────────────────────────────────────────
    ref fn1 = w.get(keys.ffn_norm1())
    var xfn1 = rms_norm(x_mid, fn1, ZEPS, ctx)
    var xfn1s = mul(xfn1, scale_mlp, ctx)

    ref w1 = w.get(keys.ff_w1())
    ref w3 = w.get(keys.ff_w3())
    var g_f = _lora_linear_fwd(xfn1s, w1, loras[LT.LORA_FF_W1][], ctx)  # [S,ff]
    var ff_g = _clone(g_f.y, ctx); var down_w1 = _clone(g_f.down, ctx)
    var u_f = _lora_linear_fwd(xfn1s, w3, loras[LT.LORA_FF_W3][], ctx)
    var ff_u = _clone(u_f.y, ctx); var down_w3 = _clone(u_f.down, ctx)
    var ff_act = swiglu(ff_g, ff_u, ctx)                  # silu(g)*u  [S,ff]

    ref w2 = w.get(keys.ff_w2())
    var w2_f = _lora_linear_fwd(ff_act, w2, loras[LT.LORA_FF_W2][], ctx)  # [S,dim]
    var ff_out = _clone(w2_f.y, ctx)
    var down_w2 = _clone(w2_f.down, ctx)

    ref fn2 = w.get(keys.ffn_norm2())
    var ff_n2 = rms_norm(ff_out, fn2, ZEPS, ctx)
    var gated_ff = mul(ff_n2, gate_mlp, ctx)
    var out = add(x_mid, gated_ff, ctx)                   # [S,dim]

    var acts = ZImageBlockActs(
        _clone(x, ctx), scale_msa^, gate_msa^, scale_mlp^, gate_mlp^, xn1^, _clone(xn1s, ctx),
        q_pre_rope^, k_pre_rope^, _clone(qr, ctx), _clone(kr, ctx), _clone(v4, ctx),
        to_q_raw^, to_k_raw^, _clone(attn_out, ctx), attn_flat^, _clone(x_mid, ctx),
        xfn1^, _clone(xfn1s, ctx), ff_g^, ff_u^, _clone(ff_act, ctx), ff_out^,
        down_q^, down_k^, down_v^, down_out^, down_w1^, down_w3^, down_w2^,
    )
    return ZImageBlockOut(out^, acts^)


# ── block backward ────────────────────────────────────────────────────────────
# d_out: [S,dim] grad wrt block output. Returns d_x [S,dim] (grad wrt block
# input) + the 7 LoRA d_A/d_B (slot order). Hand-chained reverse of forward.
# Frozen weights (norms, adaLN, base projection) receive NO grad; their backward
# arms only pass d_x through.
struct ZImageBlockBwd(Movable):
    var d_x: Tensor
    var lora_grads: ZImageBlockLoraGrads

    def __init__(out self, var d_x: Tensor, var lora_grads: ZImageBlockLoraGrads):
        self.d_x = d_x^
        self.lora_grads = lora_grads^


def zimage_block_backward[S: Int](
    d_out: Tensor,
    acts: ZImageBlockActs,
    cos: Tensor,
    sin: Tensor,
    w: ZImageWeights,
    keys: ZImageBlockKeys,
    loras: List[LArc],
    ctx: DeviceContext,
) raises -> ZImageBlockBwd:
    var scale = Float32(1.0) / sqrt(Float32(ZDh))
    var ff_hidden = acts.ff_g.shape()[1]

    var d_a = List[TArc]()
    var d_b = List[TArc]()
    for _ in range(LT.LORA_SLOTS_PER_BLOCK):
        d_a.append(ArcPointer(zeros_device(_shape1(1), STDtype.BF16, ctx)))
        d_b.append(ArcPointer(zeros_device(_shape1(1), STDtype.BF16, ctx)))

    # out = x_mid + gate_mlp * ff_n2.  Residual fans into x_mid and the gated FFN.
    var d_x_mid = _clone(d_out, ctx)                      # via residual add
    var d_gated_ff = _clone(d_out, ctx)
    # gated_ff = ff_n2 * gate_mlp (gate frozen → only d_ff_n2 = d_gated_ff*gate)
    var d_ff_n2 = mul(d_gated_ff, acts.gate_mlp, ctx)
    # ff_n2 = rms_norm(ff_out, ffn_norm2) ; weight frozen → d_x only
    ref fn2 = w.get(keys.ffn_norm2())
    var d_ff_out = rms_norm_backward_dx(d_ff_n2, acts.ff_out, fn2, ZEPS, ctx)  # [S,dim]

    # ff_out = w2(ff_act)  (+LoRA w2). in=ff_hidden, out=dim
    ref w2 = w.get(keys.ff_w2())
    var w2_b = _lora_linear_bwd(
        d_ff_out, acts.ff_act, acts.down_w2, w2, loras[LT.LORA_FF_W2][],
        S, ff_hidden, ZDIM, ctx,
    )
    var d_ff_act = _clone(w2_b.d_x, ctx)                  # [S,ff]
    d_a[LT.LORA_FF_W2] = ArcPointer(_clone(w2_b.d_a, ctx))
    d_b[LT.LORA_FF_W2] = ArcPointer(_clone(w2_b.d_b, ctx))

    # ff_act = swiglu(g, u)  -> d_g, d_u
    var sg = swiglu_backward(d_ff_act, acts.ff_g, acts.ff_u, ctx)

    # g = w1(xfn1s)+LoRA ; u = w3(xfn1s)+LoRA. Both read xfn1s → d_x fans in.
    ref w1 = w.get(keys.ff_w1())
    ref w3 = w.get(keys.ff_w3())
    var w1_b = _lora_linear_bwd(sg.d_gate, acts.xfn1s, acts.down_w1, w1, loras[LT.LORA_FF_W1][], S, ZDIM, ff_hidden, ctx)
    d_a[LT.LORA_FF_W1] = ArcPointer(_clone(w1_b.d_a, ctx)); d_b[LT.LORA_FF_W1] = ArcPointer(_clone(w1_b.d_b, ctx))
    var w3_b = _lora_linear_bwd(sg.d_up, acts.xfn1s, acts.down_w3, w3, loras[LT.LORA_FF_W3][], S, ZDIM, ff_hidden, ctx)
    d_a[LT.LORA_FF_W3] = ArcPointer(_clone(w3_b.d_a, ctx)); d_b[LT.LORA_FF_W3] = ArcPointer(_clone(w3_b.d_b, ctx))
    var d_xfn1s = add(w1_b.d_x, w3_b.d_x, ctx)            # [S,dim]

    # xfn1s = xfn1 * scale_mlp (scale frozen → d_xfn1 = d_xfn1s * scale_mlp)
    var d_xfn1 = mul(d_xfn1s, acts.scale_mlp, ctx)
    # xfn1 = rms_norm(x_mid, ffn_norm1) → d_x adds into x_mid grad
    ref fn1 = w.get(keys.ffn_norm1())
    var d_xmid_from_mlp = rms_norm_backward_dx(d_xfn1, acts.x_mid, fn1, ZEPS, ctx)
    d_x_mid = add(d_x_mid, d_xmid_from_mlp, ctx)

    # ── attention branch backward ────────────────────────────────────────────
    # x_mid = x + gate_msa * attn_n2. Fan: d_x (residual) and d_attn_n2.
    var d_x = _clone(d_x_mid, ctx)                        # residual to block input
    var d_attn_n2 = mul(d_x_mid, acts.gate_msa, ctx)     # gate frozen
    # attn_n2 = rms_norm(attn_out, attention_norm2) → d_attn_out
    ref n2 = w.get(keys.attn_norm2())
    var d_attn_out = rms_norm_backward_dx(d_attn_n2, acts.attn_out, n2, ZEPS, ctx)

    # attn_out = to_out(attn_flat)+LoRA. in=dim,out=dim
    ref ow = w.get(keys.to_out())
    var out_b = _lora_linear_bwd(d_attn_out, acts.attn_flat, acts.down_out, ow, loras[LT.LORA_TO_OUT][], S, ZDIM, ZDIM, ctx)
    d_a[LT.LORA_TO_OUT] = ArcPointer(_clone(out_b.d_a, ctx)); d_b[LT.LORA_TO_OUT] = ArcPointer(_clone(out_b.d_b, ctx))

    # attn_flat = reshape(sdpa) → [1,S,H,Dh]
    var d_attn4 = reshape(out_b.d_x, _shape4(1, S, ZH, ZDh), ctx)
    # sdpa backward (recomputes softmax from saved q_rope,k_rope,v)
    var sd = sdpa_backward[1, S, ZH, ZDh](acts.q_rope, acts.k_rope, acts.v_bshd, d_attn4, scale, ctx)

    # rope backward (cos/sin fixed tables, no grad): apply inverse rotation.
    var d_qn4 = rope_backward(sd.d_q, cos, sin, True, ctx)  # interleaved=True
    var d_kn4 = rope_backward(sd.d_k, cos, sin, True, ctx)

    # per-head rms_norm_q/k backward (weight frozen → d_x only), on pre-norm raw
    ref qn = w.get(keys.norm_q())
    ref kn = w.get(keys.norm_k())
    var d_q4 = rms_norm_backward_dx(d_qn4, acts.to_q_raw, qn, ZEPS, ctx)  # [1,S,H,Dh]
    var d_k4 = rms_norm_backward_dx(d_kn4, acts.to_k_raw, kn, ZEPS, ctx)

    # reshape grads back to [S,dim]
    var d_q = reshape(d_q4, _shape2(S, ZDIM), ctx)
    var d_k = reshape(d_k4, _shape2(S, ZDIM), ctx)
    var d_v = reshape(sd.d_v, _shape2(S, ZDIM), ctx)

    # to_q/k/v backward (+LoRA). All read xn1s → d_xn1s fans in.
    ref qw = w.get(keys.to_q())
    ref kw = w.get(keys.to_k())
    ref vw = w.get(keys.to_v())
    var q_b = _lora_linear_bwd(d_q, acts.xn1s, acts.down_q, qw, loras[LT.LORA_TO_Q][], S, ZDIM, ZDIM, ctx)
    d_a[LT.LORA_TO_Q] = ArcPointer(_clone(q_b.d_a, ctx)); d_b[LT.LORA_TO_Q] = ArcPointer(_clone(q_b.d_b, ctx))
    var k_b = _lora_linear_bwd(d_k, acts.xn1s, acts.down_k, kw, loras[LT.LORA_TO_K][], S, ZDIM, ZDIM, ctx)
    d_a[LT.LORA_TO_K] = ArcPointer(_clone(k_b.d_a, ctx)); d_b[LT.LORA_TO_K] = ArcPointer(_clone(k_b.d_b, ctx))
    var v_b = _lora_linear_bwd(d_v, acts.xn1s, acts.down_v, vw, loras[LT.LORA_TO_V][], S, ZDIM, ZDIM, ctx)
    d_a[LT.LORA_TO_V] = ArcPointer(_clone(v_b.d_a, ctx)); d_b[LT.LORA_TO_V] = ArcPointer(_clone(v_b.d_b, ctx))
    var d_xn1s = add(add(q_b.d_x, k_b.d_x, ctx), v_b.d_x, ctx)  # [S,dim]

    # xn1s = xn1 * scale_msa (scale frozen → d_xn1 = d_xn1s * scale_msa)
    var d_xn1 = mul(d_xn1s, acts.scale_msa, ctx)
    # xn1 = rms_norm(x, attention_norm1) → d_x adds into block input grad
    ref n1 = w.get(keys.attn_norm1())
    var d_x_from_attn = rms_norm_backward_dx(d_xn1, acts.x_in, n1, ZEPS, ctx)
    d_x = add(d_x, d_x_from_attn, ctx)

    return ZImageBlockBwd(d_x^, ZImageBlockLoraGrads(d_a^, d_b^))


# ── helpers ───────────────────────────────────────────────────────────────────
def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    return cast_tensor(x, x.dtype(), ctx)   # dtype no-op → deep device copy

def _shape1(a: Int) -> List[Int]:
    var s = List[Int](); s.append(a); return s^

def _shape2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^

def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); s.append(d); return s^

def _slice1(x: Tensor, start: Int, length: Int, ctx: DeviceContext) raises -> Tensor:
    return slice(x, 0, start, length, ctx)

def _add_scalar_slice(x: Tensor, start: Int, length: Int, s: Float32, ctx: DeviceContext) raises -> Tensor:
    var sl = _slice1(x, start, length, ctx)
    return add_scalar(sl, s, ctx)


def _zimage_sdpa_nomask_infer[S: Int](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    comptime score_mib = (S * S * ZH * 4) // (1024 * 1024)
    comptime if score_mib < ZIMAGE_MATH_SDPA_BUDGET_MIB:
        return sdpa_nomask[1, S, ZH, ZDh](q, k, v, scale, ctx)
    else:
        return sdpa_nomask_tiled[1, S, ZH, ZDh](q, k, v, scale, ctx)


# ══════════════════════════════════════════════════════════════════════════════
# FULL 30-BLOCK MAIN STACK — forward (checkpoint: save only block inputs) +
# backward (per-block RECOMPUTE of acts, then verified block backward).
#
# COMPOSITION + MEMORY CONTRACT BORROWED FROM (structure, not code):
#   serenitymojo/models/klein/klein_stack.mojo::klein_stack_forward (:296) /
#   klein_stack_backward (:378) — reverse walk, inter-block handoff
#   d_x(block N) = d_y(block N+1), shared rope tables uploaded ONCE, per-block
#   recompute checkpointing (MOJO_AUTOGRAD_INTERNALS.md §8.3).
#   serenitymojo/models/klein/klein_stack_lora.mojo — flat LoRA set + grad scatter.
#
# DIFFERENCE vs klein_stack.mojo: that file RETAINS every block's saved acts in a
# List across the whole forward (dbl_saved/sgl_saved). Here we take the §8.3
# memory contract literally — forward saves ONLY each block's input tensor; the
# backward RE-RUNS that block's forward (one block, cheap) to regenerate its acts
# right before running its backward. Peak activation memory ≈ one block.
#
# The MAIN stack operates on the unified [x, cap] sequence AFTER the refiners and
# the concat (zimage_dit.mojo forward_full :main layers loop). This struct owns
# the 30 main layers only; refiner/embed/final-layer live in ZImageModel's
# forward driver (refiners are FROZEN, no LoRA → not trained here).
# Ref Serenity modules/model/ZImageModel.py (transformer module) +
# BaseZImageSetup.py predict() (input=latent+timestep+text, target=velocity).
# ══════════════════════════════════════════════════════════════════════════════

comptime ZN_MAIN = LT.ZIMAGE_N_MAIN_LAYERS   # 30
comptime ZSLOTS = LT.LORA_SLOTS_PER_BLOCK    # 7


# ── flat LoRA set for the 30-block main stack (30*7 = 210 adapters) ───────────
# Indexed block-major / slot-minor: idx = block*ZSLOTS + slot (matches
# ZImageLoRASetup.zimage_lora_target_prefixes order). LoraAdapter is move-only →
# boxed as ArcPointer (LArc) so it lives in a List.
struct ZImageLoraSet(Movable):
    var ad: List[LArc]    # [ZN_MAIN * ZSLOTS]
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


# Per-slot Linear shapes (in,out) for adapter allocation. Matches the FROZEN base
# weight [out,in] in the safetensors map (ZImageLoRASetup slot comments).
def _slot_in_out(slot: Int) raises -> Tuple[Int, Int]:
    if slot == LT.LORA_FF_W1 or slot == LT.LORA_FF_W3:
        return (ZDIM, ZFF)             # w1/w3: in=dim, out=ff_hidden (10240)
    if slot == LT.LORA_FF_W2:
        return (ZFF, ZDIM)             # w2: in=ff_hidden, out=dim
    return (ZDIM, ZDIM)                # to_q/k/v/out: square


# Build the full 210-adapter set (A small randn, B=0 → PEFT identity at step 0).
# `ff_hidden` overrides the default (real checkpoint hidden = ZFF = 10240).
def build_zimage_lora_set(
    rank: Int, alpha: Float32, ctx: DeviceContext,
    n_layers: Int = ZN_MAIN, ff_hidden: Int = ZFF,
) raises -> ZImageLoraSet:
    var ad = List[LArc]()
    var seed = UInt64(0x5A1A)
    for b in range(n_layers):
        for s in range(ZSLOTS):
            var io = _slot_in_out(s)
            var in_f = io[0]
            var out_f = io[1]
            if s == LT.LORA_FF_W1 or s == LT.LORA_FF_W3:
                out_f = ff_hidden
            elif s == LT.LORA_FF_W2:
                in_f = ff_hidden
            ad.append(ArcPointer(make_lora_adapter(in_f, out_f, rank, alpha, seed, ctx)))
            seed += 1
    return ZImageLoraSet(ad^, n_layers, rank)


def build_zimage_inactive_lora_set(rank: Int, n_layers: Int = ZN_MAIN) -> ZImageLoraSet:
    var ad = List[LArc]()
    return ZImageLoraSet(ad^, n_layers, rank, False)


# Transient per-block LoRA view (the 7 LArc slots block fwd/bwd consume).
def _loras_for_block(set: ZImageLoraSet, b: Int) -> List[LArc]:
    var base = b * ZSLOTS
    var out = List[LArc]()
    for s in range(ZSLOTS):
        out.append(set.ad[base + s])    # ArcPointer copy = shared ref (cheap)
    return out^


# ── flat stack LoRA grads (parallel to ZImageLoraSet.ad) ──────────────────────
struct ZImageStackLoraGrads(Movable):
    var d_a: List[TArc]   # [ZN_MAIN * ZSLOTS] each [rank, in]
    var d_b: List[TArc]   # [ZN_MAIN * ZSLOTS] each [out, rank]
    var d_x_in: Tensor    # load-bearing grad wrt the stack INPUT [S,dim]

    def __init__(out self, var d_a: List[TArc], var d_b: List[TArc], var d_x_in: Tensor):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_x_in = d_x_in^


# ── stack forward (checkpoint = save only per-block inputs) ────────────────────
# x: [S,dim] unified sequence entering the main layers. Returns (out, x_inputs)
# where x_inputs[b] is the saved INPUT to block b (the only thing backward needs;
# acts are recomputed in backward). Shared rope tables `cos`/`sin` uploaded by
# the caller ONCE and borrowed by every block.
struct ZImageStackForward(Movable):
    var out: Tensor
    var x_inputs: List[TArc]   # [n_layers] block inputs (recompute checkpoints)

    def __init__(out self, var out: Tensor, var x_inputs: List[TArc]):
        self.out = out^
        self.x_inputs = x_inputs^


def zimage_stack_forward[S: Int](
    x_in: Tensor,
    adaln: Tensor,
    cos: Tensor,
    sin: Tensor,
    w: ZImageWeights,
    loras: ZImageLoraSet,
    ctx: DeviceContext,
) raises -> ZImageStackForward:
    var x = _clone(x_in, ctx)
    var x_inputs = List[TArc]()
    for b in range(loras.n_layers):
        x_inputs.append(TArc(_clone(x, ctx)))             # CHECKPOINT: save input only
        var keys = ZImageBlockKeys.for_block(b)
        var bl = _loras_for_block(loras, b)
        var bo = zimage_block_forward[S](x, adaln, cos, sin, w, keys, bl, ctx)
        x = bo^.take_out()                                 # keep residual; acts DISCARDED
    return ZImageStackForward(x^, x_inputs^)


# ══════════════════════════════════════════════════════════════════════════════
# NO-GRAD INFERENCE PATH — activation-free block + stack forward.
#
# Identical block math to zimage_block_forward / zimage_stack_forward, but it does
# NOT build the ZImageBlockActs bundle (no per-tensor _clone for backward), does
# NOT save per-block input checkpoints, and returns ONLY the residual stream. The
# LoRA overlay still applies (B may be 0). Mirrors Serenity's torch.no_grad()
# sampling: same numerics, no saved-for-backward state. ──────────────────────────
def _lora_linear_fwd_y(
    x: Tensor, base_w: Tensor, ad: LoraAdapter, ctx: DeviceContext
) raises -> Tensor:
    # Same as _lora_linear_fwd but discards the `down` activation (inference only).
    var base = linear(x, base_w, None, ctx)              # [M, out]
    var down = linear(x, ad.a, None, ctx)                # [M, rank]
    var up = linear(down, ad.b, None, ctx)               # [M, out]
    var scaled = mul_scalar(up, ad.scale(), ctx)
    return add(base, scaled, ctx)


def zimage_block_forward_infer[S: Int](
    x: Tensor,
    adaln: Tensor,
    cos: Tensor,
    sin: Tensor,
    w: ZImageWeights,
    keys: ZImageBlockKeys,
    loras: List[LArc],
    ctx: DeviceContext,
) raises -> Tensor:
    # SAME math as zimage_block_forward; returns only `out`, builds no acts.
    var scale = Float32(1.0) / sqrt(Float32(ZDh))

    # ── modulation ──
    ref mw = w.get(keys.adaln_w())
    ref mb = w.get(keys.adaln_b())
    var mod = linear(adaln, mw, Optional[Tensor](_clone(mb, ctx)), ctx)   # [1, 4*dim]
    var mod_flat = reshape(mod, _shape1(4 * ZDIM), ctx)                   # [4*dim]
    var scale_msa = _add_scalar_slice(mod_flat, 0 * ZDIM, ZDIM, 1.0, ctx)
    var gate_msa = tanh_op(_slice1(mod_flat, 1 * ZDIM, ZDIM, ctx), ctx)
    var scale_mlp = _add_scalar_slice(mod_flat, 2 * ZDIM, ZDIM, 1.0, ctx)
    var gate_mlp = tanh_op(_slice1(mod_flat, 3 * ZDIM, ZDIM, ctx), ctx)

    # ── attention branch ──
    ref n1 = w.get(keys.attn_norm1())
    var xn1 = rms_norm(x, n1, ZEPS, ctx)
    var xn1s = mul(xn1, scale_msa, ctx)

    ref qw = w.get(keys.to_q())
    ref kw = w.get(keys.to_k())
    ref vw = w.get(keys.to_v())
    var q4 = reshape(_lora_linear_fwd_y(xn1s, qw, loras[LT.LORA_TO_Q][], ctx), _shape4(1, S, ZH, ZDh), ctx)
    var k4 = reshape(_lora_linear_fwd_y(xn1s, kw, loras[LT.LORA_TO_K][], ctx), _shape4(1, S, ZH, ZDh), ctx)
    var v4 = reshape(_lora_linear_fwd_y(xn1s, vw, loras[LT.LORA_TO_V][], ctx), _shape4(1, S, ZH, ZDh), ctx)

    ref qn = w.get(keys.norm_q())
    ref kn = w.get(keys.norm_k())
    var qn4 = rms_norm(q4, qn, ZEPS, ctx)
    var kn4 = rms_norm(k4, kn, ZEPS, ctx)

    var qr = rope_interleaved(qn4, cos, sin, ctx)
    var kr = rope_interleaved(kn4, cos, sin, ctx)

    # Online-softmax (tiled) SDPA: never materializes the [S,S] scores buffer, so
    # it runs the large S=unified_len attention without the OOM spike. Exact to
    # machine precision vs sdpa_nomask (online softmax is mathematically exact) →
    # preserves the latent parity vs Serenity.
    var attn4 = _zimage_sdpa_nomask_infer[S](qr, kr, v4, scale, ctx)
    var attn_flat = reshape(attn4, _shape2(S, ZDIM), ctx)

    ref ow = w.get(keys.to_out())
    var attn_out = _lora_linear_fwd_y(attn_flat, ow, loras[LT.LORA_TO_OUT][], ctx)

    ref n2 = w.get(keys.attn_norm2())
    var attn_n2 = rms_norm(attn_out, n2, ZEPS, ctx)
    var gated_attn = mul(attn_n2, gate_msa, ctx)
    var x_mid = add(x, gated_attn, ctx)

    # ── mlp branch ──
    ref fn1 = w.get(keys.ffn_norm1())
    var xfn1 = rms_norm(x_mid, fn1, ZEPS, ctx)
    var xfn1s = mul(xfn1, scale_mlp, ctx)

    ref w1 = w.get(keys.ff_w1())
    ref w3 = w.get(keys.ff_w3())
    var ff_g = _lora_linear_fwd_y(xfn1s, w1, loras[LT.LORA_FF_W1][], ctx)
    var ff_u = _lora_linear_fwd_y(xfn1s, w3, loras[LT.LORA_FF_W3][], ctx)
    var ff_act = swiglu(ff_g, ff_u, ctx)

    ref w2 = w.get(keys.ff_w2())
    var ff_out = _lora_linear_fwd_y(ff_act, w2, loras[LT.LORA_FF_W2][], ctx)

    ref fn2 = w.get(keys.ffn_norm2())
    var ff_n2 = rms_norm(ff_out, fn2, ZEPS, ctx)
    var gated_ff = mul(ff_n2, gate_mlp, ctx)
    return add(x_mid, gated_ff, ctx)


def zimage_block_forward_base_infer[S: Int](
    x: Tensor,
    adaln: Tensor,
    cos: Tensor,
    sin: Tensor,
    w: ZImageWeights,
    keys: ZImageBlockKeys,
    ctx: DeviceContext,
) raises -> Tensor:
    # Same no-grad inference block, but with no LoRA overlay. This is the product
    # base-model sampler path; identity B=0 adapters are not a license to spend
    # two extra GEMMs per LoRA target.
    var scale = Float32(1.0) / sqrt(Float32(ZDh))

    ref mw = w.get(keys.adaln_w())
    ref mb = w.get(keys.adaln_b())
    var mod = linear(adaln, mw, Optional[Tensor](_clone(mb, ctx)), ctx)
    var mod_flat = reshape(mod, _shape1(4 * ZDIM), ctx)
    var scale_msa = _add_scalar_slice(mod_flat, 0 * ZDIM, ZDIM, 1.0, ctx)
    var gate_msa = tanh_op(_slice1(mod_flat, 1 * ZDIM, ZDIM, ctx), ctx)
    var scale_mlp = _add_scalar_slice(mod_flat, 2 * ZDIM, ZDIM, 1.0, ctx)
    var gate_mlp = tanh_op(_slice1(mod_flat, 3 * ZDIM, ZDIM, ctx), ctx)

    ref n1 = w.get(keys.attn_norm1())
    var xn1 = rms_norm(x, n1, ZEPS, ctx)
    var xn1s = mul(xn1, scale_msa, ctx)

    ref qw = w.get(keys.to_q())
    ref kw = w.get(keys.to_k())
    ref vw = w.get(keys.to_v())
    var q4 = reshape(linear(xn1s, qw, None, ctx), _shape4(1, S, ZH, ZDh), ctx)
    var k4 = reshape(linear(xn1s, kw, None, ctx), _shape4(1, S, ZH, ZDh), ctx)
    var v4 = reshape(linear(xn1s, vw, None, ctx), _shape4(1, S, ZH, ZDh), ctx)

    ref qn = w.get(keys.norm_q())
    ref kn = w.get(keys.norm_k())
    var qn4 = rms_norm(q4, qn, ZEPS, ctx)
    var kn4 = rms_norm(k4, kn, ZEPS, ctx)

    var qr = rope_interleaved(qn4, cos, sin, ctx)
    var kr = rope_interleaved(kn4, cos, sin, ctx)

    var attn4 = _zimage_sdpa_nomask_infer[S](qr, kr, v4, scale, ctx)
    var attn_flat = reshape(attn4, _shape2(S, ZDIM), ctx)

    ref ow = w.get(keys.to_out())
    var attn_out = linear(attn_flat, ow, None, ctx)

    ref n2 = w.get(keys.attn_norm2())
    var attn_n2 = rms_norm(attn_out, n2, ZEPS, ctx)
    var gated_attn = mul(attn_n2, gate_msa, ctx)
    var x_mid = add(x, gated_attn, ctx)

    ref fn1 = w.get(keys.ffn_norm1())
    var xfn1 = rms_norm(x_mid, fn1, ZEPS, ctx)
    var xfn1s = mul(xfn1, scale_mlp, ctx)

    ref w1 = w.get(keys.ff_w1())
    ref w3 = w.get(keys.ff_w3())
    var ff_g = linear(xfn1s, w1, None, ctx)
    var ff_u = linear(xfn1s, w3, None, ctx)
    var ff_act = swiglu(ff_g, ff_u, ctx)

    ref w2 = w.get(keys.ff_w2())
    var ff_out = linear(ff_act, w2, None, ctx)

    ref fn2 = w.get(keys.ffn_norm2())
    var ff_n2 = rms_norm(ff_out, fn2, ZEPS, ctx)
    var gated_ff = mul(ff_n2, gate_mlp, ctx)
    return add(x_mid, gated_ff, ctx)


def zimage_stack_forward_infer[S: Int](
    x_in: Tensor,
    adaln: Tensor,
    cos: Tensor,
    sin: Tensor,
    w: ZImageWeights,
    loras: ZImageLoraSet,
    ctx: DeviceContext,
) raises -> Tensor:
    # SAME walk as zimage_stack_forward, but saves no per-block input checkpoints.
    var x = _clone(x_in, ctx)
    for b in range(loras.n_layers):
        var keys = ZImageBlockKeys.for_block(b)
        if loras.active:
            var bl = _loras_for_block(loras, b)
            x = zimage_block_forward_infer[S](x, adaln, cos, sin, w, keys, bl, ctx)
        else:
            x = zimage_block_forward_base_infer[S](x, adaln, cos, sin, w, keys, ctx)
    return x^


# ── stack backward (REVERSE walk, per-block RECOMPUTE, scatter LoRA grads) ─────
# d_out: [S,dim] grad wrt stack output. For each block (deepest first) we RE-RUN
# its forward from the saved input to regenerate acts, then run the verified
# block backward, hand off d_x → d_y of the shallower block, and scatter the 7
# d_A/d_B into the flat grad lists. Mirrors klein_stack_backward (:378) reverse
# loop + inter-block handoff, with §8.3 recompute instead of retained acts.
def zimage_stack_backward[S: Int](
    d_out: Tensor,
    adaln: Tensor,
    cos: Tensor,
    sin: Tensor,
    w: ZImageWeights,
    loras: ZImageLoraSet,
    fwd: ZImageStackForward,
    ctx: DeviceContext,
) raises -> ZImageStackLoraGrads:
    var n = loras.n_layers
    # pre-size flat grad lists (placeholder; scattered in reverse, slot order).
    var d_a = List[TArc]()
    var d_b = List[TArc]()
    for _ in range(n * ZSLOTS):
        d_a.append(TArc(zeros_device(_shape1(1), STDtype.BF16, ctx)))
        d_b.append(TArc(zeros_device(_shape1(1), STDtype.BF16, ctx)))

    var d_x = _clone(d_out, ctx)
    var b = n - 1
    while b >= 0:
        var keys = ZImageBlockKeys.for_block(b)
        var bl = _loras_for_block(loras, b)
        # RECOMPUTE this block's acts from its saved input (one block, cheap).
        var rb = zimage_block_forward[S](
            fwd.x_inputs[b][], adaln, cos, sin, w, keys, bl, ctx
        )
        # verified block backward: returns d_x (→ becomes d_y of block b-1) + 7 grads.
        var bb = zimage_block_backward[S](d_x, rb.acts^, cos, sin, w, keys, bl, ctx)
        # scatter the 7 LoRA grads into the flat lists at block-major offset.
        var base = b * ZSLOTS
        for s in range(ZSLOTS):
            d_a[base + s] = TArc(_clone(bb.lora_grads.d_a[s][], ctx))
            d_b[base + s] = TArc(_clone(bb.lora_grads.d_b[s][], ctx))
        d_x = _clone(bb.d_x, ctx)                          # INTER-BLOCK HANDOFF
        b -= 1

    return ZImageStackLoraGrads(d_a^, d_b^, d_x^)   # d_x = grad wrt stack input


# ══════════════════════════════════════════════════════════════════════════════
# COMPILE SMOKE — single-block fwd+bwd on a tiny sequence (S=4). Builds toy
# frozen weights + 7 adapters in-process, runs the block forward then backward,
# and returns finite LoRA grads (sum of all d_A/d_B abs values → a single F32).
# Tiny so it compiles+runs fast; the orchestrator's real gate uses loaded weights.
# NOTE: this builds a ZImageWeights-shaped map via a helper on the loader; if the
# loader lacks an in-memory constructor the orchestrator should call the stack
# path directly with real weights. Kept minimal + self-contained.
# ══════════════════════════════════════════════════════════════════════════════

# Fill a fresh device weight tensor [out,in] with small randn (BF16) for the smoke.
def _smoke_w(out_f: Int, in_f: Int, seed: UInt64, ctx: DeviceContext) raises -> Tensor:
    var raw = randn(_shape2(out_f, in_f), seed, STDtype.BF16, ctx)
    return mul_scalar(raw, Float32(0.02), ctx)

def _smoke_vec(n: Int, ctx: DeviceContext) raises -> Tensor:
    # RMSNorm / per-head norm weights ≈ ones; small ones via add_scalar(0)+1.
    return add_scalar(zeros_device(_shape1(n), STDtype.BF16, ctx), Float32(1.0), ctx)

# Register one toy weight into the smoke weight map (avoids a capturing closure).
def _smoke_put(
    mut name_to_idx: Dict[String, Int], mut tensors: List[TArc],
    name: String, var t: Tensor,
) raises:
    name_to_idx[name] = len(tensors)
    tensors.append(TArc(t^))


# Returns the L1 sum of all 7 LoRA d_A + d_B (a finite scalar) for a 1-block
# fwd+bwd at S. The orchestrator asserts this is finite & > 0 (grads flow).
def zimage_block_smoke[S: Int](rank: Int, ctx: DeviceContext) raises -> Float32:
    var ff = ZFF   # real checkpoint feed_forward hidden dim (10240)
    # toy frozen base weights (one block worth) — built into a ZImageWeights map.
    var tensors = List[TArc]()
    var name_to_idx = Dict[String, Int]()
    var keys = ZImageBlockKeys.for_block(0)
    _smoke_put(name_to_idx, tensors, keys.adaln_w(), _smoke_w(4 * ZDIM, 256, UInt64(1), ctx))
    _smoke_put(name_to_idx, tensors, keys.adaln_b(), _smoke_vec(4 * ZDIM, ctx))
    _smoke_put(name_to_idx, tensors, keys.attn_norm1(), _smoke_vec(ZDIM, ctx))
    _smoke_put(name_to_idx, tensors, keys.attn_norm2(), _smoke_vec(ZDIM, ctx))
    _smoke_put(name_to_idx, tensors, keys.ffn_norm1(), _smoke_vec(ZDIM, ctx))
    _smoke_put(name_to_idx, tensors, keys.ffn_norm2(), _smoke_vec(ZDIM, ctx))
    _smoke_put(name_to_idx, tensors, keys.to_q(), _smoke_w(ZDIM, ZDIM, UInt64(2), ctx))
    _smoke_put(name_to_idx, tensors, keys.to_k(), _smoke_w(ZDIM, ZDIM, UInt64(3), ctx))
    _smoke_put(name_to_idx, tensors, keys.to_v(), _smoke_w(ZDIM, ZDIM, UInt64(4), ctx))
    _smoke_put(name_to_idx, tensors, keys.to_out(), _smoke_w(ZDIM, ZDIM, UInt64(5), ctx))
    _smoke_put(name_to_idx, tensors, keys.norm_q(), _smoke_vec(ZDh, ctx))
    _smoke_put(name_to_idx, tensors, keys.norm_k(), _smoke_vec(ZDh, ctx))
    _smoke_put(name_to_idx, tensors, keys.ff_w1(), _smoke_w(ff, ZDIM, UInt64(6), ctx))
    _smoke_put(name_to_idx, tensors, keys.ff_w3(), _smoke_w(ff, ZDIM, UInt64(7), ctx))
    _smoke_put(name_to_idx, tensors, keys.ff_w2(), _smoke_w(ZDIM, ff, UInt64(8), ctx))
    var w = ZImageWeights(tensors^, name_to_idx^)

    # 7 adapters (in,out per slot), A=randn*std, B=0.
    var bl = List[LArc]()
    var seed = UInt64(100)
    for s in range(ZSLOTS):
        var io = _slot_in_out(s)
        var in_f = io[0]
        var out_f = io[1]
        if s == LT.LORA_FF_W1 or s == LT.LORA_FF_W3:
            out_f = ff
        elif s == LT.LORA_FF_W2:
            in_f = ff
        bl.append(ArcPointer(make_lora_adapter(in_f, out_f, rank, Float32(rank), seed, ctx)))
        seed += 1

    # tiny inputs: x [S,dim], adaln [1,256], rope tables [S*H, Dh/2].
    var x = mul_scalar(randn(_shape2(S, ZDIM), UInt64(9), STDtype.BF16, ctx), Float32(0.1), ctx)
    var adaln = mul_scalar(randn(_shape2(1, 256), UInt64(10), STDtype.BF16, ctx), Float32(0.1), ctx)
    var cos = _smoke_vec(S * ZH * (ZDh // 2), ctx)
    var cos_t = reshape(cos, _shape2(S * ZH, ZDh // 2), ctx)
    var sin = mul_scalar(_smoke_vec(S * ZH * (ZDh // 2), ctx), Float32(0.0), ctx)
    var sin_t = reshape(sin, _shape2(S * ZH, ZDh // 2), ctx)

    var fo = zimage_block_forward[S](x, adaln, cos_t, sin_t, w, keys, bl, ctx)
    var d_out = mul_scalar(zeros_device(_shape2(S, ZDIM), STDtype.BF16, ctx), Float32(0.0), ctx)
    var d_out1 = add_scalar(d_out, Float32(1.0), ctx)   # all-ones upstream grad
    var bb = zimage_block_backward[S](d_out1, fo.acts^, cos_t, sin_t, w, keys, bl, ctx)

    # L1-sum every d_A/d_B → finite scalar (host reduce).
    var total = Float32(0.0)
    for s in range(ZSLOTS):
        var ah = bb.lora_grads.d_a[s][].to_host(ctx)
        for i in range(len(ah)): total += abs(ah[i])
        var bh = bb.lora_grads.d_b[s][].to_host(ctx)
        for i in range(len(bh)): total += abs(bh[i])
    return total
