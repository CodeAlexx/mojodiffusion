# model/lens/lens_backward.mojo — Lens (double-stream MM-DiT) hand-chained
# backward + the LoRA adapter set + the saved-tape struct CONTRACT.
#
# BORROWED IMPLEMENTATION MATERIAL: this MIRRORS serenitymojo Klein
# model/klein/double_block.mojo::double_block_backward (the joint-attention
# reverse chain) adapted to the Lens block structure. The forward it reverses is
# serenitymojo/pipeline/lens_pipeline_1024_multistep.mojo::lens_block_forward
# (:477-616) + final_norm_proj (:622-646) + lens_forward img_in/temb (:650-681).
# The LoRA on every wrapped Linear comes from the Slice-A training forward
# `lens_forward_full_lora`, which records the LensFullSaved bundle defined HERE.
#
# ── ACYCLIC LAYERING ──────────────────────────────────────────────────────────
# This module (Slice B) OWNS: LensLoraSet/build_lens_lora_set, the saved-tape
# structs (LensStreamSaved/LensBlockSaved/LensFullSaved), LensStackLoraGrads, and
# the backward. Slice A (lens_stack_lora.mojo) IMPORTS the set + saved structs
# from here and produces LensForwardOut{velocity, saved: LensFullSaved}. So the
# forward→backward saved-type dependency is one-directional (A→B); B imports
# nothing of A. (Same layering as Klein, where DoubleBlockSaved + the backward
# live together in double_block.mojo and the stack imports them.)
#
# ── SELF-CONTAINED BASE WEIGHTS ───────────────────────────────────────────────
# To avoid coupling the backward to Slice-C's LensWeights accessor API, the
# FORWARD stashes the FROZEN base-weight HANDLES it used (TArc = ArcPointer →
# shared, no copy) INTO the saved bundle. The backward reads base weights from
# `saved` only; it needs NO LensWeights argument.
#
# ── LoRA grads returned ───────────────────────────────────────────────────────
# 480 adapters (48*10 block), flat order == lens_lora_target_prefixes
# (block-major slot-minor). The shipped Lens preset is attn-mlp (layer_filter
# "attn,mlp"): only the per-block attn/mlp Linears are wrapped — NO img_mod/txt_mod
# and NO top-level Linears (img_in/txt_in/timestep_embedder/norm_out/proj_out).
# The frozen top-level + mod chains are still traversed with BASE-only backward
# (linear_backward, LoRA grad dropped) ONLY where needed to propagate d_h into the
# blocks (proj_out → final modulate → final layer_norm → last block). img_in/txt_in
# and the timestep-embedder backward are DEAD for this preset (their d_x reaches no
# trainable adapter) and are not computed. The base-weight grads are DROPPED (LoRA
# training: bases frozen) — only A/B factor grads flow out. The LoRA d_x correction
# IS folded into the upstream activation grad (_wrapped_linear_bwd adds da.d_x),
# matching Klein's klein_lora_bwd fold (and exact at B=0 init).
#
# DTYPE: BF16 storage; F32 inside the ops' accumulators; grads returned as host
# List[Float32] (LoraGrads). No persistent F32 tensor.
#
# Mojo move-only note: op-result grad structs (LinearGrads/GateResidualGrads/
# RmsNormBackward/ModulateBackward/CatGrads2/SdpaGrads) have SYNTHESIZED
# destructors; their Tensor fields are consumed by BORROW into the next op, and
# any value that must SURVIVE into a returned struct is CLONED (_clone) — never
# `^`-moved out of the middle of such a struct. LoRA factor grads (LoraGrads) are
# Copyable, so they copy out by value.

from std.collections import List, Optional
from std.memory import ArcPointer
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.linalg_backward import linear_backward
from serenitymojo.ops.norm_backward import rms_norm_backward_dx, layer_norm_backward_dx
from serenitymojo.ops.elementwise_backward import modulate_backward
from serenitymojo.ops.rope_struct_backward import rope_backward, gate_residual_backward
from serenitymojo.ops.attention_backward import sdpa_backward
from serenitymojo.ops.loss_swiglu_backward import swiglu_backward
from serenitymojo.ops.shape_backward import cat_backward
from serenitymojo.ops.tensor_algebra import concat, slice, reshape, add, mul_scalar, zeros_device
from std.gpu.host import DeviceContext

from serenity_trainer.module.LensLoRAModule import (
    LoraAdapter, LoraGrads, make_lora_adapter, _lora_adamw,
)
from serenity_trainer.modelSetup.lensLoraTargets import (
    LORA_IMG_QKV, LORA_TXT_QKV, LORA_TO_OUT, LORA_TO_ADD_OUT,
    LORA_IMG_MLP_W1, LORA_IMG_MLP_W2, LORA_IMG_MLP_W3,
    LORA_TXT_MLP_W1, LORA_TXT_MLP_W2, LORA_TXT_MLP_W3,
    LORA_SLOTS_PER_BLOCK, LENS_N_BLOCKS,
)

comptime TArc = ArcPointer[Tensor]

# ── Lens config constants (lens config.json / transformer.py) ─────────────────
comptime DIM       = 1536      # inner_dim
comptime NUM_HEADS = 24
comptime HEAD_DIM  = 64
comptime FF        = 4096      # int(dim/3*8)
comptime IN_CH     = 128       # patchified in_channels (proj_out out / img_in in)
comptime TXT_IN_F  = 11520     # enc_hidden_dim(2880) * 4 selected layers
comptime TEMB_PROJ = 256       # timestep_embedding dim → timestep_embedder.linear_1 in
comptime BLOCK_NORM_EPS = Float32(1.0e-6)   # img/txt_norm1/2 (rms_norm=True, eps=1e-6)
comptime QK_NORM_EPS    = Float32(1.0e-6)   # attn.norm_q/k (RMSNorm dim_head, eps=1e-6: transformer.py:295 block default eps=1e-6 -> :306 attn -> :210-213; model :424-432 no override)
comptime FINAL_LN_EPS   = Float32(1.0e-6)   # AdaLayerNormContinuous eps=1e-6
comptime ROPE_INTERLEAVED = True            # Lens complex RoPE pairs (2i,2i+1)


# ── small helpers ─────────────────────────────────────────────────────────────
def _clone(t: Tensor, ctx: DeviceContext) raises -> Tensor:
    # owned copy of a borrowed/struct-field tensor (×1.0); avoids partial-move.
    return mul_scalar(t, Float32(1.0), ctx)

def _flat2d(t: Tensor, M: Int, F: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int](); sh.append(M); sh.append(F)
    return reshape(t, sh^, ctx)

def _to3d(t: Tensor, N: Int, F: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int](); sh.append(1); sh.append(N); sh.append(F)
    return reshape(t, sh^, ctx)

def _to_bshd(t: Tensor, N: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int](); sh.append(1); sh.append(N); sh.append(NUM_HEADS); sh.append(HEAD_DIM)
    return reshape(t, sh^, ctx)

def _list_to_t(v: List[Float32], ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int](); sh.append(DIM)
    return Tensor.from_host(v.copy(), sh^, STDtype.BF16, ctx)

def _sh3(N: Int) -> List[Int]:
    var s = List[Int](); s.append(1); s.append(N); s.append(DIM); return s^

# chunk i of a [1,6*DIM] mod output → [DIM] (matches lens_pipeline _adaln_chunk).
def _chunk(mod_out: Tensor, idx: Int, ctx: DeviceContext) raises -> Tensor:
    var s = slice(mod_out, 1, idx * DIM, DIM, ctx)   # [1, DIM]
    var sh = List[Int](); sh.append(DIM)
    return reshape(s, sh^, ctx)                       # [DIM]


# ══════════════════════════════════════════════════════════════════════════════
# LoRA SET  (480 adapters, flat order = lens_lora_target_prefixes; attn-mlp preset)
# ══════════════════════════════════════════════════════════════════════════════
def _block_slot_dims(slot: Int) raises -> Tuple[Int, Int]:
    if slot == LORA_IMG_QKV or slot == LORA_TXT_QKV:   return (DIM, 3 * DIM)
    if slot == LORA_TO_OUT or slot == LORA_TO_ADD_OUT: return (DIM, DIM)
    if slot == LORA_IMG_MLP_W1 or slot == LORA_TXT_MLP_W1: return (DIM, FF)
    if slot == LORA_IMG_MLP_W3 or slot == LORA_TXT_MLP_W3: return (DIM, FF)
    if slot == LORA_IMG_MLP_W2 or slot == LORA_TXT_MLP_W2: return (FF, DIM)
    raise Error(String("_block_slot_dims: bad slot ") + String(slot))


# Flat LoRA set (LoraAdapter is Copyable → plain List storage). attn-mlp preset:
# 48*10 = 480 per-block adapters, no top-level (img_mod/txt_mod and the top-level
# Linears are NOT wrapped by layer_filter "attn,mlp").
struct LensLoraSet(Copyable, Movable):
    var block: List[LoraAdapter]   # 48*10 = 480, order = (b*10 + slot)
    var rank: Int

    def __init__(out self, var block: List[LoraAdapter], rank: Int):
        self.block = block^
        self.rank = rank


# build_lens_lora_set — A~small-uniform (kaiming-ish), B=0 (PEFT identity at init,
# LoRAModule.py:550-551). Mirrors LensLoRASetup.setup_model's LoRAModuleWrapper.
def build_lens_lora_set(rank: Int, alpha: Float32, seed: UInt64, ctx: DeviceContext) raises -> LensLoraSet:
    _ = ctx
    var block = List[LoraAdapter]()
    var s = seed
    for _b in range(LENS_N_BLOCKS):
        for slot in range(LORA_SLOTS_PER_BLOCK):
            var dims = _block_slot_dims(slot)
            s = s * UInt64(6364136223846793005) + UInt64(1)
            block.append(make_lora_adapter(rank, alpha, dims[0], dims[1], s))
    return LensLoraSet(block^, rank)


fn _bidx(b: Int, slot: Int) -> Int:
    return b * LORA_SLOTS_PER_BLOCK + slot


# ══════════════════════════════════════════════════════════════════════════════
# LoRA GRADS  (480 LoraGrads, same flat order as LensLoraSet)
# ══════════════════════════════════════════════════════════════════════════════
struct LensStackLoraGrads(Movable):
    var block: List[LoraGrads]   # 480, order = (b*10 + slot)

    def __init__(out self, var block: List[LoraGrads]):
        self.block = block^


# Drive AdamW over every adapter at optimizer step `t` (1-based), parallel order.
def lens_lora_adamw_step(
    mut loras: LensLoraSet, grads: LensStackLoraGrads, t: Int, lr: Float32, ctx: DeviceContext
) raises:
    for i in range(len(loras.block)):
        _lora_adamw(loras.block[i], grads.block[i], t, lr, ctx)


# ══════════════════════════════════════════════════════════════════════════════
# SAVED-TAPE CONTRACT  (the FORWARD `lens_forward_full_lora` must populate these)
# ══════════════════════════════════════════════════════════════════════════════
struct LensStreamSaved(Copyable, Movable):
    var x: TArc          # [1,N,DIM]  block input (lens_block_forward img_h/txt_e in)
    var rn1: TArc        # [1,N,DIM]  rms_norm(x, norm1_w)                    (:515/:519)
    var m1: TArc         # [1,N,DIM]  modulate(rn1, scale1, shift1)          (:516/:520) (qkv input)
    var q_pre: TArc      # [1,N,H,Dh] post-qkv-split q, pre qk-rmsnorm       (:532/:540)
    var k_pre: TArc      # [1,N,H,Dh]
    var v: TArc          # [1,N,H,Dh]
    var att: TArc        # [1,N,DIM]  per-stream attention slice             (:572/:573) (to_out input)
    var proj: TArc       # [1,N,DIM]  linear(att, to_out/to_add_out)         (:578/:582) (gate1 residual y)
    var attn_res: TArc   # [1,N,DIM]  residual_gate(x, gate1, proj)          (:585/:586)
    var rn2: TArc        # [1,N,DIM]  rms_norm(attn_res, norm2_w)            (:590/:603)
    var m2: TArc         # [1,N,DIM]  modulate(rn2, scale2, shift2)          (:591/:604) (w1/w3 input)
    var gate: TArc       # [1,N,FF]   linear(m2, w1)                         (:596/:609)
    var up: TArc         # [1,N,FF]   linear(m2, w3)                         (:597/:610)
    var act: TArc        # [1,N,FF]   swiglu(gate, up)                       (:598/:611) (w2 input)
    var mo: TArc         # [1,N,DIM]  linear(act, w2)                        (:599/:612) (gate2 residual y)
    # FROZEN base-weight handles (no copy):
    var n1w: TArc        # img/txt_norm1.weight  [DIM]
    var n2w: TArc        # img/txt_norm2.weight  [DIM]
    var nq: TArc         # attn.norm_q/norm_added_q.weight  [Dh]
    var nk: TArc         # attn.norm_k/norm_added_k.weight  [Dh]
    var qkv_base: TArc   # attn.img_qkv/txt_qkv.weight   [3*DIM, DIM]
    var toout_base: TArc # attn.to_out.0/to_add_out.weight [DIM, DIM]
    var w1_base: TArc    # mlp.w1.weight [FF, DIM]
    var w2_base: TArc    # mlp.w2.weight [DIM, FF]
    var w3_base: TArc    # mlp.w3.weight [FF, DIM]

    def __init__(
        out self, var x: TArc, var rn1: TArc, var m1: TArc,
        var q_pre: TArc, var k_pre: TArc, var v: TArc,
        var att: TArc, var proj: TArc, var attn_res: TArc, var rn2: TArc, var m2: TArc,
        var gate: TArc, var up: TArc, var act: TArc, var mo: TArc,
        var n1w: TArc, var n2w: TArc, var nq: TArc, var nk: TArc,
        var qkv_base: TArc, var toout_base: TArc,
        var w1_base: TArc, var w2_base: TArc, var w3_base: TArc,
    ):
        self.x = x^; self.rn1 = rn1^; self.m1 = m1^
        self.q_pre = q_pre^; self.k_pre = k_pre^; self.v = v^
        self.att = att^; self.proj = proj^; self.attn_res = attn_res^; self.rn2 = rn2^; self.m2 = m2^
        self.gate = gate^; self.up = up^; self.act = act^; self.mo = mo^
        self.n1w = n1w^; self.n2w = n2w^; self.nq = nq^; self.nk = nk^
        self.qkv_base = qkv_base^; self.toout_base = toout_base^
        self.w1_base = w1_base^; self.w2_base = w2_base^; self.w3_base = w3_base^


struct LensBlockSaved(Copyable, Movable):
    var img: LensStreamSaved
    var txt: LensStreamSaved
    var q_rope: TArc      # [1,S,H,Dh]  concat(roped img_q, roped txt_q)   (:564) IMG FIRST
    var k_rope: TArc      # [1,S,H,Dh]
    var v_joint: TArc     # [1,S,H,Dh]  concat(img_v, txt_v)               (:566)
    var img_mod_out: TArc # [1,6*DIM]  linear(temb_act, img_mod.1)         (:493)
    var txt_mod_out: TArc # [1,6*DIM]  linear(temb_act, txt_mod.1)         (:497)
    var img_mod_base: TArc# img_mod.1.weight [6*DIM, DIM]
    var txt_mod_base: TArc# txt_mod.1.weight [6*DIM, DIM]
    var img_cos: TArc     # img rope tables (cos/sin) — rope_backward handles
    var img_sin: TArc
    var txt_cos: TArc
    var txt_sin: TArc

    def __init__(
        out self, var img: LensStreamSaved, var txt: LensStreamSaved,
        var q_rope: TArc, var k_rope: TArc, var v_joint: TArc,
        var img_mod_out: TArc, var txt_mod_out: TArc,
        var img_mod_base: TArc, var txt_mod_base: TArc,
        var img_cos: TArc, var img_sin: TArc, var txt_cos: TArc, var txt_sin: TArc,
    ):
        self.img = img^; self.txt = txt^
        self.q_rope = q_rope^; self.k_rope = k_rope^; self.v_joint = v_joint^
        self.img_mod_out = img_mod_out^; self.txt_mod_out = txt_mod_out^
        self.img_mod_base = img_mod_base^; self.txt_mod_base = txt_mod_base^
        self.img_cos = img_cos^; self.img_sin = img_sin^
        self.txt_cos = txt_cos^; self.txt_sin = txt_sin^


struct LensFullSaved(Movable):
    var blocks: List[LensBlockSaved]   # 48, in forward order
    var packed_latent: TArc   # [1,N_IMG,128]  img_in input                 (:662)
    var txt_in_input: TArc    # [1,N_TXT,11520] txt_in input (normed-concat) (:516 ctx)
    var h_final: TArc         # [1,N_IMG,DIM]  img stream after last block   (final layer_norm input :641)
    var normed_final: TArc    # [1,N_IMG,DIM]  layer_norm(h_final)           (:641)
    var out_final: TArc       # [1,N_IMG,DIM]  modulate(normed,scale,shift)  (:642) (proj_out input)
    var final_mod_out: TArc   # [1,2*DIM]  linear(temb_act, norm_out.linear) (:631)
    var temb: TArc            # [1,DIM]  make_temb output (pre final silu)    (:471)
    var temb_act: TArc        # [1,DIM]  silu(temb) (mod/norm_out linear input)(:489/:628)
    var ts_h1: TArc           # [1,DIM]  linear(ts_proj, timestep_embedder.linear_1)(:469)
    var ts_h2: TArc           # [1,DIM]  silu(ts_h1)                          (:470)
    var ts_proj: TArc         # [1,256]  timestep_embedding(t)                (:462) (lin1 input)
    var img_in_base: TArc     # img_in.weight   [DIM, 128]
    var txt_in_base: TArc     # txt_in.weight   [DIM, 11520]
    var norm_out_base: TArc   # norm_out.linear.weight [2*DIM, DIM]
    var proj_out_base: TArc   # proj_out.weight [128, DIM]
    var ts_lin1_base: TArc    # timestep_embedder.linear_1.weight [DIM, 256]
    var ts_lin2_base: TArc    # timestep_embedder.linear_2.weight [DIM, DIM]
    var final_ln_ones: TArc   # [DIM] ones (final layer_norm has no affine → weight=1)

    def __init__(
        out self, var blocks: List[LensBlockSaved],
        var packed_latent: TArc, var txt_in_input: TArc,
        var h_final: TArc, var normed_final: TArc, var out_final: TArc, var final_mod_out: TArc,
        var temb: TArc, var temb_act: TArc, var ts_h1: TArc, var ts_h2: TArc, var ts_proj: TArc,
        var img_in_base: TArc, var txt_in_base: TArc, var norm_out_base: TArc, var proj_out_base: TArc,
        var ts_lin1_base: TArc, var ts_lin2_base: TArc, var final_ln_ones: TArc,
    ):
        self.blocks = blocks^
        self.packed_latent = packed_latent^; self.txt_in_input = txt_in_input^
        self.h_final = h_final^; self.normed_final = normed_final^
        self.out_final = out_final^; self.final_mod_out = final_mod_out^
        self.temb = temb^; self.temb_act = temb_act^
        self.ts_h1 = ts_h1^; self.ts_h2 = ts_h2^; self.ts_proj = ts_proj^
        self.img_in_base = img_in_base^; self.txt_in_base = txt_in_base^
        self.norm_out_base = norm_out_base^; self.proj_out_base = proj_out_base^
        self.ts_lin1_base = ts_lin1_base^; self.ts_lin2_base = ts_lin2_base^
        self.final_ln_ones = final_ln_ones^


# ══════════════════════════════════════════════════════════════════════════════
# Wrapped-linear backward (forward: y = x@base_wᵀ + (x@Aᵀ@Bᵀ)*scale).
# `x` [M,in], `d_y` [M,out], base_w [out,in]. Returns d_x [M,in] (base + LoRA
# correction) and the A/B grads. Base weight grad DROPPED (frozen).
# ══════════════════════════════════════════════════════════════════════════════
struct _WLBack(Movable):
    var d_x: Tensor
    var g: LoraGrads
    def __init__(out self, var d_x: Tensor, var g: LoraGrads):
        self.d_x = d_x^
        self.g = g^


def _wrapped_linear_bwd(
    d_y: Tensor, x: Tensor, base_w: Tensor, lo: LoraAdapter,
    M: Int, in_f: Int, out_f: Int, ctx: DeviceContext,
) raises -> _WLBack:
    var base = linear_backward(d_y, x, base_w, M, in_f, out_f, ctx)       # base.d_x; d_w dropped
    var a = Tensor.from_host_bf16(lo.a.copy(), [lo.rank, lo.in_f], ctx)
    var b = Tensor.from_host_bf16(lo.b.copy(), [lo.out_f, lo.rank], ctx)
    var nb = Optional[Tensor](None)
    var xa = linear(x, a, nb^, ctx)                                      # [M,rank]
    var d_scaled = mul_scalar(d_y, lo.scale, ctx)                        # scale backward
    var db = linear_backward(d_scaled, xa, b, M, lo.rank, lo.out_f, ctx) # d_B, d_xa
    var da = linear_backward(db.d_x^, x, a, M, lo.in_f, lo.rank, ctx)    # d_A, lora d_x
    var d_x_total = add(base.d_x, da.d_x, ctx)                           # fold LoRA d_x
    return _WLBack(d_x_total^, LoraGrads(da.d_w.to_host(ctx), db.d_w.to_host(ctx)))


# ══════════════════════════════════════════════════════════════════════════════
# PER-STREAM POST backward — reverses lens_block_forward steps 6→10 for one stream.
# ══════════════════════════════════════════════════════════════════════════════
struct _PostBack(Movable):
    var d_x: Tensor        # [1,N,DIM] residual branch
    var d_att: Tensor      # [1,N,DIM] into the attention slice
    var g_toout: LoraGrads
    var g_w1: LoraGrads
    var g_w2: LoraGrads
    var g_w3: LoraGrads
    var d_gate1: Tensor    # [DIM]
    var d_scale2: Tensor   # [DIM]
    var d_shift2: Tensor   # [DIM]
    var d_gate2: Tensor    # [DIM]
    def __init__(
        out self, var d_x: Tensor, var d_att: Tensor,
        var g_toout: LoraGrads, var g_w1: LoraGrads, var g_w2: LoraGrads, var g_w3: LoraGrads,
        var d_gate1: Tensor, var d_scale2: Tensor, var d_shift2: Tensor, var d_gate2: Tensor,
    ):
        self.d_x = d_x^; self.d_att = d_att^
        self.g_toout = g_toout^; self.g_w1 = g_w1^; self.g_w2 = g_w2^; self.g_w3 = g_w3^
        self.d_gate1 = d_gate1^; self.d_scale2 = d_scale2^; self.d_shift2 = d_shift2^; self.d_gate2 = d_gate2^


def _stream_post_backward(
    d_out: Tensor, sv: LensStreamSaved, mod_out: Tensor, N: Int,
    lo_toout: LoraAdapter, lo_w1: LoraAdapter, lo_w2: LoraAdapter, lo_w3: LoraAdapter,
    ctx: DeviceContext,
) raises -> _PostBack:
    var gate2 = _chunk(mod_out, 5, ctx)
    var scale2 = _chunk(mod_out, 4, ctx)
    var gate1 = _chunk(mod_out, 2, ctx)

    # 10) out = residual_gate(attn_res, gate2, mo)  (lens :600/:613)
    var grg2 = gate_residual_backward(d_out, sv.attn_res[], gate2, sv.mo[], ctx)
    var d_gate2 = grg2.d_g.to_host(ctx)

    # 9) mlp: mo = w2(act); act = swiglu(gate,up); gate=w1(m2); up=w3(m2)
    var bw2 = _wrapped_linear_bwd(
        _flat2d(grg2.d_y, N, DIM, ctx), _flat2d(sv.act[], N, FF, ctx),
        sv.w2_base[], lo_w2, N, FF, DIM, ctx,
    )
    var sgb = swiglu_backward(_to3d(bw2.d_x, N, FF, ctx), sv.gate[], sv.up[], ctx)
    var bw1 = _wrapped_linear_bwd(
        _flat2d(sgb.d_gate, N, FF, ctx), _flat2d(sv.m2[], N, DIM, ctx),
        sv.w1_base[], lo_w1, N, DIM, FF, ctx,
    )
    var bw3 = _wrapped_linear_bwd(
        _flat2d(sgb.d_up, N, FF, ctx), _flat2d(sv.m2[], N, DIM, ctx),
        sv.w3_base[], lo_w3, N, DIM, FF, ctx,
    )
    var d_m2 = add(bw1.d_x, bw3.d_x, ctx)            # [N,DIM]

    # 8) m2 = modulate(rn2, scale2, shift2)
    var mb2 = modulate_backward(_to3d(d_m2, N, DIM, ctx), sv.rn2[], scale2, ctx)
    var d_scale2 = mb2.d_scale.to_host(ctx)
    var d_shift2 = mb2.d_shift.to_host(ctx)
    # rn2 = rms_norm(attn_res, n2w)  (n2w FROZEN → d_x-only backward)
    var rb2_dx = rms_norm_backward_dx(mb2.d_x, sv.attn_res[], sv.n2w[], BLOCK_NORM_EPS, ctx)
    # attn_res feeds BOTH the gate2 residual branch AND rn2 → sum.
    var d_attn_res = add(grg2.d_x, rb2_dx, ctx)      # [1,N,DIM]

    # 7) attn_res = residual_gate(x, gate1, proj)
    var grg1 = gate_residual_backward(d_attn_res, sv.x[], gate1, sv.proj[], ctx)
    var d_gate1 = grg1.d_g.to_host(ctx)
    var d_x_res = _clone(grg1.d_x, ctx)              # survives to block level
    # 6) proj = linear(att, to_out/to_add_out)
    var bto = _wrapped_linear_bwd(
        _flat2d(grg1.d_y, N, DIM, ctx), _flat2d(sv.att[], N, DIM, ctx),
        sv.toout_base[], lo_toout, N, DIM, DIM, ctx,
    )

    return _PostBack(
        d_x_res^, _to3d(bto.d_x, N, DIM, ctx),
        bto.g.copy(), bw1.g.copy(), bw2.g.copy(), bw3.g.copy(),
        _list_to_t(d_gate1, ctx), _list_to_t(d_scale2, ctx),
        _list_to_t(d_shift2, ctx), _list_to_t(d_gate2, ctx),
    )


# ══════════════════════════════════════════════════════════════════════════════
# PER-STREAM PRE backward — reverses lens_block_forward steps 1→4 for one stream.
# ══════════════════════════════════════════════════════════════════════════════
struct _PreBack(Movable):
    var d_x: Tensor
    var g_qkv: LoraGrads
    var d_scale1: Tensor
    var d_shift1: Tensor
    def __init__(out self, var d_x: Tensor, var g_qkv: LoraGrads, var d_scale1: Tensor, var d_shift1: Tensor):
        self.d_x = d_x^; self.g_qkv = g_qkv^; self.d_scale1 = d_scale1^; self.d_shift1 = d_shift1^


def _stream_pre_backward(
    d_q_pre: Tensor, d_k_pre: Tensor, d_v: Tensor,   # [1,N,H,Dh] each
    sv: LensStreamSaved, mod_out: Tensor, N: Int, lo_qkv: LoraAdapter, ctx: DeviceContext,
) raises -> _PreBack:
    var scale1 = _chunk(mod_out, 1, ctx)
    # join d_q_pre|d_k_pre|d_v → d_qkv [1,N,3*DIM] ([1,N,H,Dh]->[1,N,DIM] is a byte
    # no-op since DIM = H*Dh).
    var dq = _to3d(_flat2d(d_q_pre, N, DIM, ctx), N, DIM, ctx)
    var dk = _to3d(_flat2d(d_k_pre, N, DIM, ctx), N, DIM, ctx)
    var dv = _to3d(_flat2d(d_v, N, DIM, ctx), N, DIM, ctx)
    var d_qkv = concat(2, ctx, dq, dk, dv)            # [1,N,3*DIM]
    # qkv = linear(m1, qkv_w)
    var bqkv = _wrapped_linear_bwd(
        _flat2d(d_qkv, N, 3 * DIM, ctx), _flat2d(sv.m1[], N, DIM, ctx),
        sv.qkv_base[], lo_qkv, N, DIM, 3 * DIM, ctx,
    )
    # m1 = modulate(rn1, scale1, shift1)
    var mb1 = modulate_backward(_to3d(bqkv.d_x, N, DIM, ctx), sv.rn1[], scale1, ctx)
    var d_scale1 = mb1.d_scale.to_host(ctx)
    var d_shift1 = mb1.d_shift.to_host(ctx)
    # rn1 = rms_norm(x, n1w)  (n1w FROZEN → d_x-only backward; owned Tensor, no clone)
    var d_x_norm = rms_norm_backward_dx(mb1.d_x, sv.x[], sv.n1w[], BLOCK_NORM_EPS, ctx)
    return _PreBack(d_x_norm^, bqkv.g.copy(), _list_to_t(d_scale1, ctx), _list_to_t(d_shift1, ctx))


# ══════════════════════════════════════════════════════════════════════════════
# PER-BLOCK backward (joint coupling). Reverses lens_block_forward in full.
# Modulation note (attn-mlp preset): img_mod/txt_mod are NOT LoRA targets, so their
# linear backward is NOT computed. The modulation scale/shift/gate values are still
# read FORWARD-side from blk.img_mod_out/txt_mod_out (saved tape) to differentiate
# the modulate ops, but no d_temb_act is propagated (the timestep-embedder branch is
# dead for this preset — it reaches no trainable adapter).
# ══════════════════════════════════════════════════════════════════════════════
struct LensBlockGrads(Movable):
    var slots: List[LoraGrads]   # 10, slot order = LORA_* per-block indices
    var d_img_x: Tensor          # [1,N_IMG,DIM]
    var d_txt_x: Tensor          # [1,N_TXT,DIM]
    def __init__(out self, var slots: List[LoraGrads], var d_img_x: Tensor, var d_txt_x: Tensor):
        self.slots = slots^; self.d_img_x = d_img_x^; self.d_txt_x = d_txt_x^


def lens_block_backward[
    N_IMG: Int, N_TXT: Int, S: Int
](
    d_img_out: Tensor, d_txt_out: Tensor,
    blk: LensBlockSaved, temb_act: Tensor,
    lo_img_qkv: LoraAdapter, lo_txt_qkv: LoraAdapter,
    lo_to_out: LoraAdapter, lo_to_add_out: LoraAdapter,
    lo_img_w1: LoraAdapter, lo_img_w2: LoraAdapter, lo_img_w3: LoraAdapter,
    lo_txt_w1: LoraAdapter, lo_txt_w2: LoraAdapter, lo_txt_w3: LoraAdapter,
    ctx: DeviceContext,
) raises -> LensBlockGrads:
    var scale = Float32(1.0) / sqrt(Float32(HEAD_DIM))

    # ── POST per stream (img uses to_out.0; txt uses to_add_out) ──
    var ipb = _stream_post_backward(
        d_img_out, blk.img, blk.img_mod_out[], N_IMG,
        lo_to_out, lo_img_w1, lo_img_w2, lo_img_w3, ctx,
    )
    var tpb = _stream_post_backward(
        d_txt_out, blk.txt, blk.txt_mod_out[], N_TXT,
        lo_to_add_out, lo_txt_w1, lo_txt_w2, lo_txt_w3, ctx,
    )

    # ── join attention-slice grads into joint d_att (IMG FIRST; forward sliced
    #    img=[0:N_IMG], txt=[N_IMG:]) along axis=1 → [1,S,H,Dh] ──
    var d_iatt4 = _to_bshd(ipb.d_att, N_IMG, ctx)
    var d_tatt4 = _to_bshd(tpb.d_att, N_TXT, ctx)
    var d_att_joint = concat(1, ctx, d_iatt4, d_tatt4)    # [1,S,H,Dh]

    # ── joint SDPA backward → d_q_rope, d_k_rope, d_v_joint ──
    var sb = sdpa_backward[1, S, NUM_HEADS, HEAD_DIM](
        blk.q_rope[], blk.k_rope[], blk.v_joint[], d_att_joint, scale, ctx,
    )

    # ── split joint q/k/v grads per stream (IMG FIRST) along axis=1 ──
    var cq = cat_backward(sb.d_q, N_IMG, N_TXT, 1, ctx)   # cq.d_0=img, cq.d_1=txt
    var ck = cat_backward(sb.d_k, N_IMG, N_TXT, 1, ctx)
    var cv = cat_backward(sb.d_v, N_IMG, N_TXT, 1, ctx)

    # ── per stream: rope backward (cos/sin non-learnable) then qk rms_norm ──
    var d_iq_rms = rope_backward(cq.d_0, blk.img_cos[], blk.img_sin[], ROPE_INTERLEAVED, ctx)
    var d_ik_rms = rope_backward(ck.d_0, blk.img_cos[], blk.img_sin[], ROPE_INTERLEAVED, ctx)
    var d_tq_rms = rope_backward(cq.d_1, blk.txt_cos[], blk.txt_sin[], ROPE_INTERLEAVED, ctx)
    var d_tk_rms = rope_backward(ck.d_1, blk.txt_cos[], blk.txt_sin[], ROPE_INTERLEAVED, ctx)
    # (nq/nk FROZEN → d_x-only backward)
    var d_iq = rms_norm_backward_dx(d_iq_rms, blk.img.q_pre[], blk.img.nq[], QK_NORM_EPS, ctx)
    var d_ik = rms_norm_backward_dx(d_ik_rms, blk.img.k_pre[], blk.img.nk[], QK_NORM_EPS, ctx)
    var d_tq = rms_norm_backward_dx(d_tq_rms, blk.txt.q_pre[], blk.txt.nq[], QK_NORM_EPS, ctx)
    var d_tk = rms_norm_backward_dx(d_tk_rms, blk.txt.k_pre[], blk.txt.nk[], QK_NORM_EPS, ctx)

    # ── PRE per stream → d_x (norm branch) + qkv LoRA grads + scale1/shift1 ──
    var iprb = _stream_pre_backward(
        d_iq, d_ik, cv.d_0, blk.img, blk.img_mod_out[], N_IMG, lo_img_qkv, ctx,
    )
    var tprb = _stream_pre_backward(
        d_tq, d_tk, cv.d_1, blk.txt, blk.txt_mod_out[], N_TXT, lo_txt_qkv, ctx,
    )

    # ── stream input grad = residual branch (post) + norm branch (pre) ──
    var d_img_x = add(ipb.d_x, iprb.d_x, ctx)
    var d_txt_x = add(tpb.d_x, tprb.d_x, ctx)

    # ── modulation: img_mod/txt_mod are NOT LoRA targets (attn-mlp preset), so the
    #    mod linear backward is skipped. The per-stream modulate ops were already
    #    differentiated inside _stream_pre/post_backward (their scale/shift/gate grads
    #    feed the dead temb branch only); no d_temb_act is propagated. `temb_act`,
    #    blk.img_mod_base/txt_mod_base are unused for this preset. ──
    _ = temb_act

    # ── pack 10 slot grads in LORA_* order (no mod) ──
    var slots = List[LoraGrads]()
    slots.append(iprb.g_qkv.copy())        # 0 LORA_IMG_QKV
    slots.append(tprb.g_qkv.copy())        # 1 LORA_TXT_QKV
    slots.append(ipb.g_toout.copy())       # 2 LORA_TO_OUT
    slots.append(tpb.g_toout.copy())       # 3 LORA_TO_ADD_OUT
    slots.append(ipb.g_w1.copy())          # 4 LORA_IMG_MLP_W1
    slots.append(ipb.g_w2.copy())          # 5 LORA_IMG_MLP_W2
    slots.append(ipb.g_w3.copy())          # 6 LORA_IMG_MLP_W3
    slots.append(tpb.g_w1.copy())          # 7 LORA_TXT_MLP_W1
    slots.append(tpb.g_w2.copy())          # 8 LORA_TXT_MLP_W2
    slots.append(tpb.g_w3.copy())          # 9 LORA_TXT_MLP_W3
    return LensBlockGrads(slots^, d_img_x^, d_txt_x^)


# ══════════════════════════════════════════════════════════════════════════════
# FULL Lens DiT backward — reverses final_norm_proj → 48 blocks. img_in/txt_in/
# norm_out.linear/timestep_embedder are frozen non-targets (attn-mlp preset) and
# their backward is omitted (reaches no trainable adapter). `d_velocity` is the grad
# of the loss wrt the PACKED velocity output [1,N_IMG,128] (== d_predicted; the
# driver applies the unpack/unpatchify pullback before calling). Returns the 480
# LoRA grads (48*10 block, attn-mlp preset).
# ══════════════════════════════════════════════════════════════════════════════
def lens_backward_full_lora[
    HLp: Int, WLp: Int, CAPLEN: Int
](
    d_velocity: Tensor, saved: LensFullSaved, loras: LensLoraSet, ctx: DeviceContext,
) raises -> LensStackLoraGrads:
    comptime N_IMG = HLp * WLp
    comptime N_TXT = CAPLEN
    comptime S = N_IMG + N_TXT

    # ── reverse final_norm_proj (lens_pipeline :622-646). attn-mlp preset: proj_out
    #    and norm_out.linear are NOT LoRA targets → BASE-only linear_backward (LoRA
    #    grad dropped); we only need d_x to propagate d_h into the last block. ──
    # proj = linear(out_final, proj_out)  (base-only)
    var bproj = linear_backward(
        _flat2d(d_velocity, N_IMG, IN_CH, ctx), _flat2d(saved.out_final[], N_IMG, DIM, ctx),
        saved.proj_out_base[], N_IMG, DIM, IN_CH, ctx,
    )
    # out_final = modulate(normed_final, scale, shift); scale=final_mod_out[:DIM].
    var f_scale = _chunk(saved.final_mod_out[], 0, ctx)
    var mbf = modulate_backward(_to3d(bproj.d_x, N_IMG, DIM, ctx), saved.normed_final[], f_scale, ctx)
    # normed_final = layer_norm(h_final, ones, 0)  → d_h (into last block). The final
    # modulation scale/shift grads (mbf.d_scale/d_shift) feed only norm_out.linear →
    # timestep embedder, a dead branch for this preset, so they are dropped.
    # (weight = ones, no affine → d_x-only backward; owned Tensor, no clone)
    var d_h = layer_norm_backward_dx(mbf.d_x, saved.h_final[], saved.final_ln_ones[], FINAL_LN_EPS, ctx)  # [1,N_IMG,DIM]

    # ── reverse the 48 blocks (last → first). final touches only the img stream,
    #    so the last block's d_txt_out is zero. ──
    var block_grads = List[LoraGrads]()
    for _ in range(LENS_N_BLOCKS * LORA_SLOTS_PER_BLOCK):
        block_grads.append(LoraGrads(List[Float32](), List[Float32]()))

    var d_txt = zeros_device(_sh3(N_TXT), STDtype.BF16, ctx)   # [1,N_TXT,DIM]
    for ii in range(LENS_N_BLOCKS):
        var b = LENS_N_BLOCKS - 1 - ii
        var bg = lens_block_backward[N_IMG, N_TXT, S](
            d_h, d_txt, saved.blocks[b], saved.temb_act[],
            loras.block[_bidx(b, LORA_IMG_QKV)], loras.block[_bidx(b, LORA_TXT_QKV)],
            loras.block[_bidx(b, LORA_TO_OUT)], loras.block[_bidx(b, LORA_TO_ADD_OUT)],
            loras.block[_bidx(b, LORA_IMG_MLP_W1)], loras.block[_bidx(b, LORA_IMG_MLP_W2)],
            loras.block[_bidx(b, LORA_IMG_MLP_W3)],
            loras.block[_bidx(b, LORA_TXT_MLP_W1)], loras.block[_bidx(b, LORA_TXT_MLP_W2)],
            loras.block[_bidx(b, LORA_TXT_MLP_W3)],
            ctx,
        )
        for slot in range(LORA_SLOTS_PER_BLOCK):
            block_grads[_bidx(b, slot)] = bg.slots[slot].copy()
        d_h = _clone(bg.d_img_x, ctx)
        d_txt = _clone(bg.d_txt_x, ctx)

    # img_in / txt_in / norm_out.linear / timestep-embedder are frozen and NOT LoRA
    # targets (attn-mlp preset); their backward reaches no trainable adapter (the
    # network input is not trained), so it is omitted. d_h/d_txt at the top of the
    # stack are discarded.
    return LensStackLoraGrads(block_grads^)
