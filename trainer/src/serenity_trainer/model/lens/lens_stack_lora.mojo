# model/lens/lens_stack_lora.mojo — Lens (double-stream MM-DiT) SLICE-A: the full
# 48-block hand-chained LoRA TRAINING forward. Applies the host-list LoRA deltas
# (LensLoraSet, module/LensLoRAModule adapters) on every wrapped Linear AND records
# the exact LensFullSaved activation bundle that model/lens/lens_backward.mojo
# consumes for the hand-chained backward.
#
# ── BORROWED IMPLEMENTATION MATERIAL ──────────────────────────────────────────
# The block math/order MIRRORS model/LensDiT.mojo::lens_block_forward_lora (which
# is itself the verified copy of serenitymojo's working Lens pipeline forward,
# cos>=0.999 vs the Serenity oracle). The ONLY differences here vs the infer/LoRA
# block are: (1) the LoRA delta is taken from the HOST-LIST adapter set
# (LensLoraSet.block[i], LensLoRAModule.lora_forward) instead of the Tensor-backed
# LArc set, (2) the per-Linear / per-op activations are CLONED into LensStreamSaved/
# LensBlockSaved/LensFullSaved for the backward, and (3) the MLP LoRA slots use the
# lensLoraTargets index mapping (W1=4, W2=5, W3=6 — the order the LoRA set was built
# in and the backward reads), NOT LensDiT's distinct (W1=4, W3=5, W2=6) local order.
# No numeric formula is re-derived; the op sequence is identical to the verified
# forward, only saving + the host-list delta are added.
#
# ── ACYCLIC LAYERING (A → B) ──────────────────────────────────────────────────
# This module (Slice A) IMPORTS the LoRA set + saved-tape structs from
# model/lens/lens_backward.mojo (Slice B). Slice B imports NOTHING of Slice A, so
# the forward→backward saved-type dependency stays one-directional (same layering
# as Klein: double_block.mojo owns the saved structs + backward, the stack imports
# them).
#
# DTYPE: BF16 storage in/out; F32 only inside the foundation kernels. Frozen base
# weights are shared by ArcPointer handle (no F32/BF16 duplication); the saved
# activation clones are BF16.
#
# Mojo 1.0.0b1: `def` (not `fn`); Tensor move-only (clone to both use+save); no-bias
# linear = linear(x, w, Optional[Tensor](None), ctx).

from std.collections import List, Optional
from std.memory import ArcPointer
from std.math import sqrt
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.activations import silu, swiglu
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.embeddings import timestep_embedding
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm, layer_norm
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.tensor_algebra import add, mul_scalar, reshape, slice, concat

from serenity_trainer.modelLoader.LensModelLoader import LensWeights
from serenity_trainer.module.LensLoRAModule import LoraAdapter, lora_forward
from serenity_trainer.modelSetup.lensLoraTargets import (
    LORA_IMG_QKV, LORA_TXT_QKV, LORA_TO_OUT, LORA_TO_ADD_OUT,
    LORA_IMG_MLP_W1, LORA_IMG_MLP_W2, LORA_IMG_MLP_W3,
    LORA_TXT_MLP_W1, LORA_TXT_MLP_W2, LORA_TXT_MLP_W3,
    LORA_SLOTS_PER_BLOCK, LENS_N_BLOCKS,
)
from serenity_trainer.model.lens.lens_backward import (
    LensLoraSet, LensStreamSaved, LensBlockSaved, LensFullSaved,
)
# RoPE tables: parametric builder reused from the verified Lens DiT (no re-derive).
from serenity_trainer.model.LensDiT import LensRopeTables, build_lens_rope_tables


comptime TArc = ArcPointer[Tensor]

# Lens config constants (config.json; identical to LensDiT / lens_backward).
comptime DIM        = 1536      # inner_dim
comptime NUM_HEADS  = 24
comptime HEAD_DIM   = 64
comptime FF         = 4096      # int(dim/3*8)
comptime IN_CH      = 128       # patchified in_channels (proj_out out / img_in in)
comptime ENC_HIDDEN = 2880      # GPT-OSS per-layer feature dim
comptime TXT_IN_DIM = 11520     # ENC_HIDDEN * 4 selected layers
comptime TEMB_DIM   = 256       # timestep_embedding dim
comptime NUM_LAYERS = 48
comptime BLOCK_NORM_EPS = Float32(1.0e-6)   # img/txt_norm1/2
comptime QK_NORM_EPS    = Float32(1.0e-6)   # attn.norm_q/k/added
comptime TXT_NORM_EPS   = Float32(1.0e-5)   # GPT-OSS per-layer RMSNorm
comptime FINAL_LN_EPS   = Float32(1.0e-6)   # AdaLayerNormContinuous


# ── small helpers ─────────────────────────────────────────────────────────────
def _clone(t: Tensor, ctx: DeviceContext) raises -> Tensor:
    # owned BF16-preserving copy (×1.0); used to both use AND save an activation.
    return mul_scalar(t, Float32(1.0), ctx)


def _wc(weights: LensWeights, name: String, ctx: DeviceContext) raises -> Tensor:
    # COMPUTE copy of a frozen base weight, cast to BF16 (matches the verified
    # infer/LoRA forward which casts every weight to BF16 before the GEMM).
    # LensWeights already stores BF16 (LensModelLoader casts at load), so this is a
    # BF16->BF16 copy; synchronize=False drops the redundant per-weight host stall
    # (edv2-Klein resident-bf16 pattern). Numerically identical (the copy value is
    # unchanged; only the per-op ctx.synchronize() is removed) under single-stream
    # ordering, same as the cast already used inside `linear`/`rms_norm`.
    return cast_tensor(weights.get(name), STDtype.BF16, ctx, False)


def _wh(weights: LensWeights, name: String, ctx: DeviceContext) raises -> TArc:
    # SHARED handle of a frozen base weight for the saved tape (ArcPointer refcount
    # bump — no data copy). LensWeights.load stores BF16, the same dtype both the
    # compute path (_wc) and the hand-chained backward's linear_backward consume, so
    # the shared handle is dtype-correct without any duplication. `ctx` is unused
    # (no cast needed) but kept for call-site uniformity with _wc.
    _ = ctx
    if name not in weights.name_to_idx:
        raise Error(String("lens_stack_lora: missing weight: ") + name)
    var idx = weights.name_to_idx[name]
    return weights.weights[idx].copy()


def _ones_bf16(n: Int, ctx: DeviceContext) raises -> Tensor:
    var vals = List[Float32]()
    for _ in range(n):
        vals.append(Float32(1.0))
    var sh = List[Int](); sh.append(n)
    return Tensor.from_host(vals, sh^, STDtype.BF16, ctx)


def _zeros_bf16(n: Int, ctx: DeviceContext) raises -> Tensor:
    var vals = List[Float32]()
    for _ in range(n):
        vals.append(Float32(0.0))
    var sh = List[Int](); sh.append(n)
    return Tensor.from_host(vals, sh^, STDtype.BF16, ctx)


def _adaln_chunk(mod_out: Tensor, idx: Int, ctx: DeviceContext) raises -> Tensor:
    var part = slice(mod_out, 1, idx * DIM, DIM, ctx)   # [1, DIM]
    var sh = List[Int](); sh.append(DIM)
    return reshape(part, sh^, ctx)                        # [DIM]


def _to_bshd[N: Int](x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(1); sh.append(N); sh.append(NUM_HEADS); sh.append(HEAD_DIM)
    return reshape(x, sh^, ctx)


def _from_bshd[N: Int](x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(1); sh.append(N); sh.append(DIM)
    return reshape(x, sh^, ctx)


def _apply_rope[N: Int](
    x: Tensor, cos_tiled: Tensor, sin_tiled: Tensor, ctx: DeviceContext
) raises -> Tensor:
    var flat_sh = List[Int](); flat_sh.append(N * NUM_HEADS); flat_sh.append(HEAD_DIM)
    var x_flat = reshape(x, flat_sh^, ctx)
    var roped = rope_interleaved(x_flat, cos_tiled, sin_tiled, ctx)
    var bshd = List[Int]()
    bshd.append(1); bshd.append(N); bshd.append(NUM_HEADS); bshd.append(HEAD_DIM)
    return reshape(roped, bshd^, ctx)


def _lora_delta(x: Tensor, lo: LoraAdapter, M: Int, ctx: DeviceContext) raises -> Tensor:
    # host-list LoRA delta: scale*((x@Aᵀ)@Bᵀ). At B=0 this is exactly 0 (identity
    # overlay at init), so the train forward equals the verified base forward.
    return lora_forward(x, lo, M, ctx)


# ── forward output (velocity + the saved-for-backward bundle) ──────────────────
struct LensForwardOut(Movable):
    var velocity: Tensor       # [1, N_IMG, 128] BF16 (proj_out)
    var saved: LensFullSaved   # activation bundle for lens_backward_full_lora

    def __init__(out self, var velocity: Tensor, var saved: LensFullSaved):
        self.velocity = velocity^
        self.saved = saved^


# ══════════════════════════════════════════════════════════════════════════════
# PER-BLOCK forward with activation saving (mirrors LensDiT.lens_block_forward_lora).
# Updates img_h/txt_e in place and returns the LensBlockSaved tape entry.
# `b` is the block index; `temb_act` = silu(temb) (shared modulation source).
# ══════════════════════════════════════════════════════════════════════════════
def _lens_block_forward_lora_save[N_IMG_: Int, N_TXT_: Int](
    mut img_h: Tensor,
    mut txt_e: Tensor,
    temb_act: Tensor,
    weights: LensWeights,
    b: Int,
    rope: LensRopeTables,
    loras: LensLoraSet,
    ctx: DeviceContext,
) raises -> LensBlockSaved:
    comptime S_ = N_IMG_ + N_TXT_
    var p = String("transformer_blocks.") + String(b) + String(".")
    var base = b * LORA_SLOTS_PER_BLOCK

    # ── save the block stream inputs (LensStreamSaved.x) ──
    var img_x_save = TArc(_clone(img_h, ctx))
    var txt_x_save = TArc(_clone(txt_e, ctx))

    # ── modulation (img_mod/txt_mod are NOT LoRA targets in attn-mlp preset) ──
    var img_mod_w = _wc(weights, p + String("img_mod.1.weight"), ctx)
    var img_mod_b = _wc(weights, p + String("img_mod.1.bias"), ctx)
    var img_mod = linear(temb_act, img_mod_w, Optional[Tensor](img_mod_b^), ctx)   # [1,6*DIM]
    var txt_mod_w = _wc(weights, p + String("txt_mod.1.weight"), ctx)
    var txt_mod_b = _wc(weights, p + String("txt_mod.1.bias"), ctx)
    var txt_mod = linear(temb_act, txt_mod_w, Optional[Tensor](txt_mod_b^), ctx)   # [1,6*DIM]

    var img_shift1 = _adaln_chunk(img_mod, 0, ctx)
    var img_scale1 = _adaln_chunk(img_mod, 1, ctx)
    var img_gate1  = _adaln_chunk(img_mod, 2, ctx)
    var img_shift2 = _adaln_chunk(img_mod, 3, ctx)
    var img_scale2 = _adaln_chunk(img_mod, 4, ctx)
    var img_gate2  = _adaln_chunk(img_mod, 5, ctx)
    var txt_shift1 = _adaln_chunk(txt_mod, 0, ctx)
    var txt_scale1 = _adaln_chunk(txt_mod, 1, ctx)
    var txt_gate1  = _adaln_chunk(txt_mod, 2, ctx)
    var txt_shift2 = _adaln_chunk(txt_mod, 3, ctx)
    var txt_scale2 = _adaln_chunk(txt_mod, 4, ctx)
    var txt_gate2  = _adaln_chunk(txt_mod, 5, ctx)

    # ── PRE per stream: rms_norm1 → modulate1 → qkv(+LoRA) → split → qk-norm → rope ──
    var img_n1w = _wc(weights, p + String("img_norm1.weight"), ctx)
    var img_rn1 = rms_norm(img_h, img_n1w, BLOCK_NORM_EPS, ctx)
    var img_m1  = modulate(img_rn1, img_scale1, img_shift1, ctx)
    var txt_n1w = _wc(weights, p + String("txt_norm1.weight"), ctx)
    var txt_rn1 = rms_norm(txt_e, txt_n1w, BLOCK_NORM_EPS, ctx)
    var txt_m1  = modulate(txt_rn1, txt_scale1, txt_shift1, ctx)

    var iqkv_w = _wc(weights, p + String("attn.img_qkv.weight"), ctx)
    var iqkv_b = _wc(weights, p + String("attn.img_qkv.bias"), ctx)
    var img_qkv = linear(img_m1, iqkv_w, Optional[Tensor](iqkv_b^), ctx)
    img_qkv = add(img_qkv, _lora_delta(img_m1, loras.block[base + LORA_IMG_QKV], N_IMG_, ctx), ctx)
    var tqkv_w = _wc(weights, p + String("attn.txt_qkv.weight"), ctx)
    var tqkv_b = _wc(weights, p + String("attn.txt_qkv.bias"), ctx)
    var txt_qkv = linear(txt_m1, tqkv_w, Optional[Tensor](tqkv_b^), ctx)
    txt_qkv = add(txt_qkv, _lora_delta(txt_m1, loras.block[base + LORA_TXT_QKV], N_TXT_, ctx), ctx)

    var img_q_flat = slice(img_qkv, 2, 0,       DIM, ctx)
    var img_k_flat = slice(img_qkv, 2, DIM,     DIM, ctx)
    var img_v_flat = slice(img_qkv, 2, 2 * DIM, DIM, ctx)
    var txt_q_flat = slice(txt_qkv, 2, 0,       DIM, ctx)
    var txt_k_flat = slice(txt_qkv, 2, DIM,     DIM, ctx)
    var txt_v_flat = slice(txt_qkv, 2, 2 * DIM, DIM, ctx)

    var img_q = _to_bshd[N_IMG_](img_q_flat, ctx)
    var img_k = _to_bshd[N_IMG_](img_k_flat, ctx)
    var img_v = _to_bshd[N_IMG_](img_v_flat, ctx)
    var txt_q = _to_bshd[N_TXT_](txt_q_flat, ctx)
    var txt_k = _to_bshd[N_TXT_](txt_k_flat, ctx)
    var txt_v = _to_bshd[N_TXT_](txt_v_flat, ctx)

    # save q/k PRE qk-rmsnorm + v (LensStreamSaved.q_pre/k_pre/v)
    var img_q_pre_save = TArc(_clone(img_q, ctx))
    var img_k_pre_save = TArc(_clone(img_k, ctx))
    var img_v_save     = TArc(_clone(img_v, ctx))
    var txt_q_pre_save = TArc(_clone(txt_q, ctx))
    var txt_k_pre_save = TArc(_clone(txt_k, ctx))
    var txt_v_save     = TArc(_clone(txt_v, ctx))

    var nq  = _wc(weights, p + String("attn.norm_q.weight"), ctx)
    var nk  = _wc(weights, p + String("attn.norm_k.weight"), ctx)
    var naq = _wc(weights, p + String("attn.norm_added_q.weight"), ctx)
    var nak = _wc(weights, p + String("attn.norm_added_k.weight"), ctx)
    img_q = rms_norm(img_q, nq,  QK_NORM_EPS, ctx)
    img_k = rms_norm(img_k, nk,  QK_NORM_EPS, ctx)
    txt_q = rms_norm(txt_q, naq, QK_NORM_EPS, ctx)
    txt_k = rms_norm(txt_k, nak, QK_NORM_EPS, ctx)

    img_q = _apply_rope[N_IMG_](img_q, rope.img_cos, rope.img_sin, ctx)
    img_k = _apply_rope[N_IMG_](img_k, rope.img_cos, rope.img_sin, ctx)
    txt_q = _apply_rope[N_TXT_](txt_q, rope.txt_cos, rope.txt_sin, ctx)
    txt_k = _apply_rope[N_TXT_](txt_k, rope.txt_cos, rope.txt_sin, ctx)

    # ── JOINT attention (img first; train path uses unmasked SDPA, matching the
    #    backward's sdpa_backward which takes no mask — exact for unpadded text) ──
    var q_joint = concat(1, ctx, img_q, txt_q)   # [1,S_,H,Dh]
    var k_joint = concat(1, ctx, img_k, txt_k)
    var v_joint = concat(1, ctx, img_v, txt_v)
    var scale = Float32(1.0) / sqrt(Float32(HEAD_DIM))
    var attn = sdpa_nomask[1, S_, NUM_HEADS, HEAD_DIM](q_joint, k_joint, v_joint, scale, ctx)

    var q_rope_save = TArc(_clone(q_joint, ctx))
    var k_rope_save = TArc(_clone(k_joint, ctx))
    var v_joint_save = TArc(_clone(v_joint, ctx))

    var attn_flat = _from_bshd[S_](attn, ctx)
    var img_attn  = slice(attn_flat, 1, 0,      N_IMG_, ctx)
    var txt_attn  = slice(attn_flat, 1, N_IMG_, N_TXT_, ctx)
    var img_att_save = TArc(_clone(img_attn, ctx))
    var txt_att_save = TArc(_clone(txt_attn, ctx))

    # ── POST per stream: to_out(+LoRA) → gate1 residual → rms_norm2 → modulate2 →
    #    SwiGLU MLP(+LoRA on w1/w2/w3) → gate2 residual ──
    var io_w = _wc(weights, p + String("attn.to_out.0.weight"), ctx)
    var io_b = _wc(weights, p + String("attn.to_out.0.bias"), ctx)
    var img_proj = linear(img_attn, io_w, Optional[Tensor](io_b^), ctx)
    img_proj = add(img_proj, _lora_delta(img_attn, loras.block[base + LORA_TO_OUT], N_IMG_, ctx), ctx)
    var to_w = _wc(weights, p + String("attn.to_add_out.weight"), ctx)
    var to_b = _wc(weights, p + String("attn.to_add_out.bias"), ctx)
    var txt_proj = linear(txt_attn, to_w, Optional[Tensor](to_b^), ctx)
    txt_proj = add(txt_proj, _lora_delta(txt_attn, loras.block[base + LORA_TO_ADD_OUT], N_TXT_, ctx), ctx)
    var img_proj_save = TArc(_clone(img_proj, ctx))
    var txt_proj_save = TArc(_clone(txt_proj, ctx))

    var img_attn_res = residual_gate(img_h, img_gate1, img_proj, ctx)
    var txt_attn_res = residual_gate(txt_e, txt_gate1, txt_proj, ctx)
    var img_attn_res_save = TArc(_clone(img_attn_res, ctx))
    var txt_attn_res_save = TArc(_clone(txt_attn_res, ctx))

    var img_n2w = _wc(weights, p + String("img_norm2.weight"), ctx)
    var img_rn2 = rms_norm(img_attn_res, img_n2w, BLOCK_NORM_EPS, ctx)
    var img_m2  = modulate(img_rn2, img_scale2, img_shift2, ctx)
    var txt_n2w = _wc(weights, p + String("txt_norm2.weight"), ctx)
    var txt_rn2 = rms_norm(txt_attn_res, txt_n2w, BLOCK_NORM_EPS, ctx)
    var txt_m2  = modulate(txt_rn2, txt_scale2, txt_shift2, ctx)

    var iw1 = _wc(weights, p + String("img_mlp.w1.weight"), ctx)
    var iw2 = _wc(weights, p + String("img_mlp.w2.weight"), ctx)
    var iw3 = _wc(weights, p + String("img_mlp.w3.weight"), ctx)
    var img_gate = linear(img_m2, iw1, Optional[Tensor](None), ctx)
    img_gate = add(img_gate, _lora_delta(img_m2, loras.block[base + LORA_IMG_MLP_W1], N_IMG_, ctx), ctx)
    var img_up = linear(img_m2, iw3, Optional[Tensor](None), ctx)
    img_up = add(img_up, _lora_delta(img_m2, loras.block[base + LORA_IMG_MLP_W3], N_IMG_, ctx), ctx)
    var img_act = swiglu(img_gate, img_up, ctx)
    var img_mo  = linear(img_act, iw2, Optional[Tensor](None), ctx)
    img_mo = add(img_mo, _lora_delta(img_act, loras.block[base + LORA_IMG_MLP_W2], N_IMG_, ctx), ctx)
    var img_h3 = residual_gate(img_attn_res, img_gate2, img_mo, ctx)

    var tw1 = _wc(weights, p + String("txt_mlp.w1.weight"), ctx)
    var tw2 = _wc(weights, p + String("txt_mlp.w2.weight"), ctx)
    var tw3 = _wc(weights, p + String("txt_mlp.w3.weight"), ctx)
    var txt_gate = linear(txt_m2, tw1, Optional[Tensor](None), ctx)
    txt_gate = add(txt_gate, _lora_delta(txt_m2, loras.block[base + LORA_TXT_MLP_W1], N_TXT_, ctx), ctx)
    var txt_up = linear(txt_m2, tw3, Optional[Tensor](None), ctx)
    txt_up = add(txt_up, _lora_delta(txt_m2, loras.block[base + LORA_TXT_MLP_W3], N_TXT_, ctx), ctx)
    var txt_act = swiglu(txt_gate, txt_up, ctx)
    var txt_mo  = linear(txt_act, tw2, Optional[Tensor](None), ctx)
    txt_mo = add(txt_mo, _lora_delta(txt_act, loras.block[base + LORA_TXT_MLP_W2], N_TXT_, ctx), ctx)
    var txt_e3 = residual_gate(txt_attn_res, txt_gate2, txt_mo, ctx)

    # ── assemble per-stream saved tapes ──
    var img_saved = LensStreamSaved(
        img_x_save, TArc(_clone(img_rn1, ctx)), TArc(_clone(img_m1, ctx)),
        img_q_pre_save, img_k_pre_save, img_v_save,
        img_att_save, img_proj_save, img_attn_res_save,
        TArc(_clone(img_rn2, ctx)), TArc(_clone(img_m2, ctx)),
        TArc(_clone(img_gate, ctx)), TArc(_clone(img_up, ctx)),
        TArc(_clone(img_act, ctx)), TArc(_clone(img_mo, ctx)),
        _wh(weights, p + String("img_norm1.weight"), ctx),
        _wh(weights, p + String("img_norm2.weight"), ctx),
        _wh(weights, p + String("attn.norm_q.weight"), ctx),
        _wh(weights, p + String("attn.norm_k.weight"), ctx),
        _wh(weights, p + String("attn.img_qkv.weight"), ctx),
        _wh(weights, p + String("attn.to_out.0.weight"), ctx),
        _wh(weights, p + String("img_mlp.w1.weight"), ctx),
        _wh(weights, p + String("img_mlp.w2.weight"), ctx),
        _wh(weights, p + String("img_mlp.w3.weight"), ctx),
    )
    var txt_saved = LensStreamSaved(
        txt_x_save, TArc(_clone(txt_rn1, ctx)), TArc(_clone(txt_m1, ctx)),
        txt_q_pre_save, txt_k_pre_save, txt_v_save,
        txt_att_save, txt_proj_save, txt_attn_res_save,
        TArc(_clone(txt_rn2, ctx)), TArc(_clone(txt_m2, ctx)),
        TArc(_clone(txt_gate, ctx)), TArc(_clone(txt_up, ctx)),
        TArc(_clone(txt_act, ctx)), TArc(_clone(txt_mo, ctx)),
        _wh(weights, p + String("txt_norm1.weight"), ctx),
        _wh(weights, p + String("txt_norm2.weight"), ctx),
        _wh(weights, p + String("attn.norm_added_q.weight"), ctx),
        _wh(weights, p + String("attn.norm_added_k.weight"), ctx),
        _wh(weights, p + String("attn.txt_qkv.weight"), ctx),
        _wh(weights, p + String("attn.to_add_out.weight"), ctx),
        _wh(weights, p + String("txt_mlp.w1.weight"), ctx),
        _wh(weights, p + String("txt_mlp.w2.weight"), ctx),
        _wh(weights, p + String("txt_mlp.w3.weight"), ctx),
    )

    var blk_saved = LensBlockSaved(
        img_saved^, txt_saved^,
        q_rope_save, k_rope_save, v_joint_save,
        TArc(_clone(img_mod, ctx)), TArc(_clone(txt_mod, ctx)),
        _wh(weights, p + String("img_mod.1.weight"), ctx),
        _wh(weights, p + String("txt_mod.1.weight"), ctx),
        TArc(_clone(rope.img_cos, ctx)), TArc(_clone(rope.img_sin, ctx)),
        TArc(_clone(rope.txt_cos, ctx)), TArc(_clone(rope.txt_sin, ctx)),
    )

    # ── commit the stream updates for the next block ──
    img_h = img_h3^
    txt_e = txt_e3^
    return blk_saved^


# ══════════════════════════════════════════════════════════════════════════════
# FULL Lens DiT LoRA TRAINING forward. img_in → txt cond (per-layer rms_norm +
# concat + txt_in) → timestep embed → 48 blocks → AdaLayerNormContinuous + proj_out.
# Records the LensFullSaved bundle for lens_backward_full_lora. `cap_feats` is the
# raw 4-layer GPT-OSS feature concat [1, CAPLEN, 11520] (the per-layer RMSNorm is
# applied INSIDE, matching the verified forward); `t_model` is t/1000.
# ══════════════════════════════════════════════════════════════════════════════
def lens_forward_full_lora[HLp: Int, WLp: Int, CAPLEN: Int](
    packed: Tensor,          # [1, N_IMG, 128] BF16 (pack_latents output)
    t_model: Float32,        # t/1000 (BaseLensSetup.predict)
    cap_feats: Tensor,       # [1, CAPLEN, 11520] BF16 (raw 4-layer concat)
    weights: LensWeights,
    loras: LensLoraSet,
    ctx: DeviceContext,
) raises -> LensForwardOut:
    comptime N_IMG = HLp * WLp
    comptime N_TXT = CAPLEN

    # ── img_in (frozen, non-target) ──
    var img_in_w = _wc(weights, String("img_in.weight"), ctx)
    var img_in_b = _wc(weights, String("img_in.bias"), ctx)
    var h = linear(packed, img_in_w, Optional[Tensor](img_in_b^), ctx)   # [1,N_IMG,DIM]
    var packed_save = TArc(_clone(packed, ctx))

    # ── text conditioning: per-layer RMSNorm(eps 1e-5) over the 4 cap slices,
    #    concat → cat4 [1,N_TXT,11520], then txt_in linear → [1,N_TXT,DIM] ──
    var cap_sh = List[Int](); cap_sh.append(1); cap_sh.append(N_TXT); cap_sh.append(TXT_IN_DIM)
    var cap3 = reshape(cap_feats, cap_sh^, ctx)
    var f0 = slice(cap3, 2, 0 * ENC_HIDDEN, ENC_HIDDEN, ctx)
    var f1 = slice(cap3, 2, 1 * ENC_HIDDEN, ENC_HIDDEN, ctx)
    var f2 = slice(cap3, 2, 2 * ENC_HIDDEN, ENC_HIDDEN, ctx)
    var f3 = slice(cap3, 2, 3 * ENC_HIDDEN, ENC_HIDDEN, ctx)
    var tn0 = _wc(weights, String("txt_norm.0.weight"), ctx)
    var tn1 = _wc(weights, String("txt_norm.1.weight"), ctx)
    var tn2 = _wc(weights, String("txt_norm.2.weight"), ctx)
    var tn3 = _wc(weights, String("txt_norm.3.weight"), ctx)
    var n0 = rms_norm(f0, tn0, TXT_NORM_EPS, ctx)
    var n1 = rms_norm(f1, tn1, TXT_NORM_EPS, ctx)
    var n2 = rms_norm(f2, tn2, TXT_NORM_EPS, ctx)
    var n3 = rms_norm(f3, tn3, TXT_NORM_EPS, ctx)
    var cat4 = concat(2, ctx, n0, n1, n2, n3)            # [1,N_TXT,11520]
    var txt_in_input_save = TArc(_clone(cat4, ctx))
    var txt_in_w = _wc(weights, String("txt_in.weight"), ctx)
    var txt_in_b = _wc(weights, String("txt_in.bias"), ctx)
    var e = linear(cat4, txt_in_w, Optional[Tensor](txt_in_b^), ctx)     # [1,N_TXT,DIM]

    # ── timestep embedding (capture intermediates) ──
    var tvals = List[Float32](); tvals.append(t_model * Float32(1000.0))
    var tsh = List[Int](); tsh.append(1)
    var t_tensor = Tensor.from_host(tvals, tsh^, STDtype.F32, ctx)
    var ts_proj = timestep_embedding(t_tensor, TEMB_DIM, ctx, Float32(10000.0), STDtype.BF16)  # [1,256]
    var ts_l1w = _wc(weights, String("time_text_embed.timestep_embedder.linear_1.weight"), ctx)
    var ts_l1b = _wc(weights, String("time_text_embed.timestep_embedder.linear_1.bias"), ctx)
    var ts_h1 = linear(ts_proj, ts_l1w, Optional[Tensor](ts_l1b^), ctx)  # [1,DIM]
    var ts_h2 = silu(ts_h1, ctx)
    var ts_l2w = _wc(weights, String("time_text_embed.timestep_embedder.linear_2.weight"), ctx)
    var ts_l2b = _wc(weights, String("time_text_embed.timestep_embedder.linear_2.bias"), ctx)
    var temb = linear(ts_h2, ts_l2w, Optional[Tensor](ts_l2b^), ctx)     # [1,DIM]
    var temb_act = silu(temb, ctx)

    # ── RoPE tables (parametric on the patchified grid + caption length) ──
    var rope = build_lens_rope_tables[HLp, WLp, CAPLEN](ctx)

    # ── 48 blocks ──
    var blocks = List[LensBlockSaved]()
    for b in range(NUM_LAYERS):
        var bs = _lens_block_forward_lora_save[N_IMG, N_TXT](
            h, e, temb_act, weights, b, rope, loras, ctx,
        )
        blocks.append(bs^)

    # ── final AdaLayerNormContinuous + proj_out (capture h_final/normed/out_final) ──
    var h_final_save = TArc(_clone(h, ctx))
    var norm_out_w = _wc(weights, String("norm_out.linear.weight"), ctx)
    var norm_out_b = _wc(weights, String("norm_out.linear.bias"), ctx)
    var final_mod_out = linear(temb_act, norm_out_w, Optional[Tensor](norm_out_b^), ctx)  # [1,2*DIM]
    var f_scale = _adaln_chunk(final_mod_out, 0, ctx)   # idx0 = scale
    var f_shift = _adaln_chunk(final_mod_out, 1, ctx)   # idx1 = shift
    var ln_ones  = _ones_bf16(DIM, ctx)
    var ln_zeros = _zeros_bf16(DIM, ctx)
    var normed = layer_norm(h, ln_ones, ln_zeros, FINAL_LN_EPS, ctx)
    var out_final = modulate(normed, f_scale, f_shift, ctx)
    var proj_out_w = _wc(weights, String("proj_out.weight"), ctx)
    var proj_out_b = _wc(weights, String("proj_out.bias"), ctx)
    var velocity = linear(out_final, proj_out_w, Optional[Tensor](proj_out_b^), ctx)  # [1,N_IMG,128]

    var saved = LensFullSaved(
        blocks^,
        packed_save, txt_in_input_save,
        h_final_save, TArc(_clone(normed, ctx)), TArc(_clone(out_final, ctx)),
        TArc(_clone(final_mod_out, ctx)),
        TArc(_clone(temb, ctx)), TArc(_clone(temb_act, ctx)),
        TArc(_clone(ts_h1, ctx)), TArc(_clone(ts_h2, ctx)), TArc(_clone(ts_proj, ctx)),
        _wh(weights, String("img_in.weight"), ctx),
        _wh(weights, String("txt_in.weight"), ctx),
        _wh(weights, String("norm_out.linear.weight"), ctx),
        _wh(weights, String("proj_out.weight"), ctx),
        _wh(weights, String("time_text_embed.timestep_embedder.linear_1.weight"), ctx),
        _wh(weights, String("time_text_embed.timestep_embedder.linear_2.weight"), ctx),
        TArc(_ones_bf16(DIM, ctx)),
    )
    return LensForwardOut(velocity^, saved^)
