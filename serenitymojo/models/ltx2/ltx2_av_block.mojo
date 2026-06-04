# serenitymojo/models/ltx2/ltx2_av_block.mojo
#
# LTX-2.3 22B JOINT AUDIO-VIDEO TRANSFORMER BLOCK — forward only.
#
# Port of musubi-tuner BasicAVTransformerBlock._forward
#   /home/alex/musubi-tuner/src/musubi_tuner/ltx_2/model/transformer/transformer.py
#   lines 466-802 (the real math; `forward` above is offload/checkpoint boilerplate).
#
# ── MEASURED WEIGHT SHAPES (block-0, ltx-2.3-22b-dev.safetensors) ────────────
#   VIDEO (Dv=4096, H=32, Dh=128):
#     attn1.to_{q,k,v}.weight/bias      [4096,4096] / [4096]
#     attn1.to_out.0.weight/bias        [4096,4096] / [4096]
#     attn1.{q,k}_norm.weight           [4096]
#     attn1.to_gate_logits.weight/bias  [32,4096] / [32]     per-head gate
#     attn2.to_{q,k,v}.weight/bias      [4096,4096] / [4096]  (video cross-attn to text)
#     attn2.to_out.0.weight/bias        [4096,4096] / [4096]
#     attn2.{q,k}_norm.weight           [4096]
#     attn2.to_gate_logits.weight/bias  [32,4096] / [32]
#     ff.net.0.proj.weight/bias         [16384,4096] / [16384]
#     ff.net.2.weight/bias              [4096,16384] / [4096]
#     scale_shift_table                 [9,4096]   rows 0:3=MSA, 3:6=FFN, 6:9=cross-q
#     prompt_scale_shift_table          [2,4096]   KV modulation for text cross-attn
#
#   AUDIO (Da=2048, Ha=32, Dh_a=64):
#     audio_attn1.to_{q,k,v}.weight/bias  [2048,2048] / [2048]
#     audio_attn1.to_out.0.weight/bias    [2048,2048] / [2048]
#     audio_attn1.{q,k}_norm.weight       [2048]
#     audio_attn1.to_gate_logits.weight/bias [32,2048] / [32]
#     audio_attn2.to_{q,k,v}.weight/bias  [2048,2048] / [2048]
#     audio_attn2.to_out.0.weight/bias    [2048,2048] / [2048]
#     audio_attn2.{q,k}_norm.weight       [2048]
#     audio_attn2.to_gate_logits.weight/bias [32,2048] / [32]
#     audio_ff.net.0.proj.weight/bias     [8192,2048] / [8192]
#     audio_ff.net.2.weight/bias          [2048,8192] / [2048]
#     audio_scale_shift_table             [9,2048]
#     audio_prompt_scale_shift_table      [2,2048]
#
#   CROSS-MODAL (audio_to_video: Q=video, KV=audio; video_to_audio: Q=audio, KV=video):
#     audio_to_video_attn.to_q.weight/bias    [4096,4096]   Q in video space
#     audio_to_video_attn.to_k.weight/bias    [4096,2048]   K in audio space
#     audio_to_video_attn.to_v.weight/bias    [4096,2048]   V in audio space
#     audio_to_video_attn.to_out.0.weight/bias [4096,4096]  out in video space
#       NOTE: musubi names these [out,in] (PyTorch Linear weight convention).
#       The Attention module for a2v has query_dim=video.dim=4096,
#       context_dim=audio.dim=2048, heads=audio.heads=32, dim_head=audio.d_head=64.
#       So head_dim for cross-modal = Da/Ha = 2048/32 = 64.
#       to_q: [H*Dh_a, Dv] = [32*64, 4096] = [2048, 4096]  (Q projects video→audio-heads)
#       to_k: [H*Dh_a, Da] = [2048, 2048]
#       to_v: [H*Dh_a, Da] = [2048, 2048]
#       to_out: [Dv, H*Dh_a] = [4096, 2048]
#       NOTE: The task brief says to_q[2048,4096] to_k[2048,2048] to_v[2048,2048]
#       to_out[4096,2048] — these match [out,in] for PyTorch.
#     video_to_audio_attn: reverse — Q=audio, KV=video
#       to_q.weight [H*Dh_a, Da] = [2048, 2048]
#       to_k.weight [H*Dh_a, Dv] = [2048, 4096]
#       to_v.weight [H*Dh_a, Dv] = [2048, 4096]
#       to_out.weight [Da, H*Dh_a] = [2048, 2048]
#     scale_shift_table_a2v_ca_video  [5,4096]   rows: ss_a2v_v, ss_v2a_v, gate_a2v
#     scale_shift_table_a2v_ca_audio  [5,2048]   rows: ss_a2v_a, ss_v2a_a, gate_v2a
#       5-row layout (from get_av_ca_ada_values, num_scale_shift_values=4):
#         rows 0-3: scale_a2v_x, shift_a2v_x, scale_v2a_x, shift_v2a_x
#         row  4:   gate out
#
# ── SUB-LAYER ORDER (faithful to _forward lines 595-801) ─────────────────────
#   1.  VIDEO self-attn (attn1): AdaLN[0:3] -> rms_no_affine -> modulate ->
#       q/k/v linear -> QK-RMSNorm -> reshape BSHD -> RoPE -> SDPA ->
#       per-head gate -> to_out -> gated residual  (vx += gate_msa * attn_out)
#   2.  VIDEO cross-attn (attn2): AdaLN[6:9] -> rms_no_affine -> modulate_q ->
#       optional KV modulate via prompt_scale_shift_table -> SDPA with context=text
#       -> per-head gate -> to_out -> residual  (vx += gate_ca * out)
#   3.  AUDIO self-attn (audio_attn1): same shape as step 1, audio tables.
#   4.  AUDIO cross-attn (audio_attn2): same as step 2, audio tables.
#   5.  CROSS-MODAL a2v:
#       vx_norm3 = rms_no_affine(vx);  ax_norm3 = rms_no_affine(ax)
#       AdaLN tables a2v_ca_video[0:4] -> scale_a2v_v, shift_a2v_v, scale_v2a_v, shift_v2a_v
#       AdaLN tables a2v_ca_video[4]   -> gate_a2v
#       AdaLN tables a2v_ca_audio[0:4] -> scale_a2v_a, shift_a2v_a, scale_v2a_a, shift_v2a_a
#       AdaLN tables a2v_ca_audio[4]   -> gate_v2a
#       a2v: vx_scaled = modulate(vx_norm3, scale_a2v_v, shift_a2v_v)
#            ax_scaled = modulate(ax_norm3, scale_a2v_a, shift_a2v_a)
#            attn_a2v = audio_to_video_attn(Q=vx_scaled, KV=ax_scaled)
#            vx += gate_a2v * attn_a2v
#       v2a: ax_scaled2 = modulate(ax_norm3, scale_v2a_a, shift_v2a_a)
#            vx_scaled2 = modulate(vx_norm3, scale_v2a_v, shift_v2a_v)
#            attn_v2a = video_to_audio_attn(Q=ax_scaled2, KV=vx_scaled2)
#            ax += gate_v2a * attn_v2a
#   6.  VIDEO FFN: AdaLN[3:6] -> rms_no_affine -> modulate ->
#       ff0 linear -> GELU -> ff2 linear -> clamp(-60000,60000) ->
#       gated residual  (vx += gate_mlp * ff)
#   7.  AUDIO FFN: same as step 6 with audio tables.
#
# ── CRITICAL IMPLEMENTATION NOTES ─────────────────────────────────────────────
#   * AdaLN modulate is computed in F32 then cast back to bf16 (overflow guard).
#     rms_norm with ones weight (no affine) is standard rms_no_affine.
#   * FFN output is clamped to ±60000 (bf16 overflow guard, ~max bf16=65504).
#   * RoPE applied to self-attn Q/K (attn1, audio_attn1) AND cross-modal Q/K
#     (audio_to_video_attn, video_to_audio_attn). NOT applied to text cross-attn (attn2, audio_attn2).
#   * Q/K-RMSNorm (with learned scale weight) applied in ALL attention paths
#     (self-attn and cross-attn q/k, cross-modal q/k).
#   * Per-head gate: gates = 2*sigmoid(to_gate_logits(mod_in)); applied before
#     to_out (broadcast over Dh dim).
#   * Cross-attention: Q from one stream, K/V from context (text or other modal).
#     Uses sdpa_cross (new function added here) with Q-seq ≠ KV-seq.
#   * prompt_scale_shift_table[2,D] modulates the text KV in cross-attn (rows=
#     shift_kv, scale_kv): context_mod = context * (1+scale_kv) + shift_kv.
#
# ── CROSS-ATTN SDPA (NEW OP) ─────────────────────────────────────────────────
#   sdpa in ops/attention.mojo requires Sq==Skv (square attention).
#   For cross-attn (text) and cross-modal, Sq ≠ Skv in general.
#   We implement sdpa_cross_nomask here directly using the math-mode path from
#   ops/attention.mojo (gather + Q@Kᵀ + softmax + P@V + scatter) but with
#   separate Sq and Skv. This is a NEW function, additive — sdpa/sdpa_nomask
#   in ops/attention.mojo are UNTOUCHED.
#
# ── ORACLE SHAPE EXPECTATIONS ────────────────────────────────────────────────
#   Inputs to ltx2_av_block_forward[B,Sv,N_TXT,Sa]:
#     vx:        [B, Sv, 4096]  bf16   video hidden states
#     ax:        [B, Sa, 2048]  bf16   audio hidden states
#     vtemb:     [B, 9*4096]    bf16   video timestep embed (ada_values source)
#     atemb:     [B, 9*2048]    bf16   audio timestep embed
#     vcross_ss_temb: [B, 4*4096] bf16  for a2v_ca_video scale-shift rows (4 params)
#     vcross_g_temb:  [B, 1*4096] bf16  for a2v_ca_video gate row
#     across_ss_temb: [B, 4*2048] bf16  for a2v_ca_audio scale-shift rows (4 params)
#     across_g_temb:  [B, 1*2048] bf16  for a2v_ca_audio gate row
#     context:   [B, N_TXT, 4096]  bf16  text encoder hidden states
#     context_a: [B, N_TXT, 2048]  bf16  text encoder for audio cross-attn
#     vrope_cos/sin: [Sv, 4096] bf16    video RoPE (split, pre-computed)
#     arope_cos/sin: [Sa, 2048] bf16    audio RoPE
#
# ── BUILD CHECK ───────────────────────────────────────────────────────────────
#   pixi run mojo build -I . serenitymojo/models/ltx2/ltx2_av_block.mojo -o /tmp/avb_check
#
# Mojo 1.0.0b1 (def not fn, no let, `var` for owned args, `mut` for inout).

from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.math import exp, sqrt
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.activations import gelu, sigmoid
from serenitymojo.ops.tensor_algebra import (
    reshape,
    add,
    mul,
    mul_scalar,
    add_scalar,
    slice,
)
from serenitymojo.ops.rope import rope_interleaved


# ── Layout constants (reuse the math-mode kernels' convention) ───────────────
comptime _DYN1 = Layout.row_major(-1)
comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256
comptime _TPB = 256
comptime _NEG_BIG = Float32(-3.0e38)


# ── Shape helpers ─────────────────────────────────────────────────────────────
def _sh1(a: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    return s^


def _sh2(a: Int, b: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    return s^


def _sh3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    return s^


def _sh4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    s.append(d)
    return s^


# ── Device-resident clone (for bias Optionals) ─────────────────────────────────
def _clone_t(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


# ── RMSNorm with no affine (ones weight — LTX-2 norm has no learnable gamma) ──
def _rms_no_affine(
    x: Tensor, eps: Float32, ctx: DeviceContext
) raises -> Tensor:
    var d = x.shape()[len(x.shape()) - 1]
    var ones = List[Float32]()
    for _ in range(d):
        ones.append(Float32(1.0))
    var w = Tensor.from_host(ones, _sh1(d), x.dtype(), ctx)
    return rms_norm(x, w, eps, ctx)


# ── AdaLN get_ada_values: extract one modulation vector from scale_shift_table
#    and temb. table is [N_rows, D]; temb is [B, N_rows*D] (pre-expanded chunks).
#    row_idx: which row of scale_shift_table to use.
#    Returns [B, 1, D] (broadcast-ready for [B, S, D] hidden states).
#
#    Math (musubi get_ada_values standard mode):
#      ada = table[row_idx] + temb[:, row_idx*D : (row_idx+1)*D]
#    (table is broadcast over batch; temb carries the sigma-conditioned delta.)
def _ada_vec(
    table: Tensor,   # [N_rows, D]
    temb: Tensor,    # [B, N_rows*D]
    row_idx: Int,
    D: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    # slice row: table[row_idx] -> [1, D]
    var trow = slice(table, 0, row_idx, 1, ctx)           # [1, D]
    # slice temb chunk: temb[:, row_idx*D:(row_idx+1)*D] -> [B, D]
    var tchunk = slice(temb, 1, row_idx * D, D, ctx)      # [B, D]
    # add: [B, D] + [1, D] -> [B, D]  (leading broadcast)
    return add(tchunk, trow, ctx)


# ── AdaLN modulate kernel (F32 for overflow guard, cast back to bf16) ─────────
# x: [rows, D] bf16; scale: [B, D] or [D]; shift: [B, D] or [D]  (broadcast).
# For the actual call sites, scale/shift are [B, D] and x is [B, S, D].
# We flatten x to [B*S, D] and broadcast scale/shift over S tokens.
# Returns [B*S, D] bf16 (shape matches x).
#
# Implementation: wrap into existing ops to keep code compact.
#   modulated = (1 + scale) * rms_x + shift   (in F32, then cast bf16)
def _modulate_bsd(
    normed: Tensor,  # [B, S, D] bf16
    scale: Tensor,   # [B, D]    bf16
    shift: Tensor,   # [B, D]    bf16
    B: Int, S: Int, D: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    # Use the broadcast mul/add from tensor_algebra.
    # Reshape scale/shift -> [B, 1, D] to broadcast over S dim.
    var sc4 = reshape(scale, _sh3(B, 1, D), ctx)   # [B, 1, D]
    var sh4 = reshape(shift, _sh3(B, 1, D), ctx)   # [B, 1, D]
    # (1 + scale) * normed + shift using elementwise ops (all broadcast-aware).
    var one_plus = add_scalar(sc4, Float32(1.0), ctx)     # [B, 1, D]
    return add(mul(normed, one_plus, ctx), sh4, ctx)      # [B, S, D]


# ── Per-head gate (2*sigmoid(gate_logits)) ────────────────────────────────────
# gate_in: [B, S, D] (the modulated self-attn input, same as q/k/v source)
# gate_w:  [H, D]  gate_b: [H]
# Returns gated attention output: [B, S, D]
# where attn_flat [B, S, H*Dh] is broadcast-multiplied by gates [B, S, H, 1].
def _apply_head_gate[S: Int, H: Int, Dh: Int](
    attn_flat: Tensor,   # [B, S, H*Dh]
    gate_in: Tensor,     # [B, S, H*Dh] modulated input (gate logit source)
    gate_w: Tensor,      # [H, H*Dh]
    var gate_b: Tensor,  # [H] (consumed)
    B: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var gl = linear(gate_in, gate_w, Optional[Tensor](gate_b^), ctx)   # [B, S, H]
    var gates = mul_scalar(sigmoid(gl, ctx), Float32(2.0), ctx)        # [B, S, H]
    var gates4 = reshape(gates, _sh4(B, S, H, 1), ctx)                # [B, S, H, 1]
    var attn4 = reshape(attn_flat, _sh4(B, S, H, Dh), ctx)            # [B, S, H, Dh]
    var gated4 = mul(attn4, gates4, ctx)                               # broadcast -> [B,S,H,Dh]
    return reshape(gated4, _sh3(B, S, H * Dh), ctx)                   # [B, S, D]


# ── Per-head gate for cross-modal (Din ≠ H*Dh) ───────────────────────────────
# Cross-modal attention has gate_in dim = Dout (query stream dim) while the
# attention output dim = H*Dh. These differ (e.g. a2v: Dout=4096, H*Dh=2048).
# musubi applies gate_logits = to_gate_logits(x) where x = modulated Q input.
# gate_in: [B, Sq, Din]  gate_w: [H, Din]  gate_b: [H]
# attn_flat: [B, Sq, H*Dh]
# Returns [B, Sq, H*Dh] after per-head gating.
def _apply_head_gate_cross[Sq: Int, H: Int, Dh: Int, Din: Int](
    attn_flat: Tensor,   # [B, Sq, H*Dh]
    gate_in: Tensor,     # [B, Sq, Din]  modulated query stream
    gate_w: Tensor,      # [H, Din]
    var gate_b: Tensor,  # [H] (consumed)
    B: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var gl = linear(gate_in, gate_w, Optional[Tensor](gate_b^), ctx)   # [B, Sq, H]
    var gates = mul_scalar(sigmoid(gl, ctx), Float32(2.0), ctx)        # [B, Sq, H]
    var gates4 = reshape(gates, _sh4(B, Sq, H, 1), ctx)               # [B, Sq, H, 1]
    var attn4 = reshape(attn_flat, _sh4(B, Sq, H, Dh), ctx)           # [B, Sq, H, Dh]
    var gated4 = mul(attn4, gates4, ctx)                               # [B, Sq, H, Dh]
    return reshape(gated4, _sh3(B, Sq, H * Dh), ctx)                  # [B, Sq, H*Dh]


# ── Cross-attention SDPA (Sq ≠ Skv) ──────────────────────────────────────────
# ops/attention.mojo sdpa/sdpa_nomask requires Sq==Skv. For cross-attn and
# cross-modal we need a rectangular Q×KV attention. This is a NEW function;
# it does NOT touch ops/attention.mojo.
#
# Implements the math-mode path (gather BSHD->BHSD F32, Q@Kᵀ, softmax, P@V,
# scatter) with Sq ≠ Skv. No additive mask (cross-modal and cross-attn here
# are unmasked in the base model — mask=None in musubi _forward for these paths).
#
# Q: [B, Sq, H, Dh]  K,V: [B, Skv, H, Dh]  -> out [B, Sq, H, Dh]  (q's dtype)
# All in bf16; F32 internally; returns bf16.

# Gather BSHD [B,S,H,Dh] into BHSD [B*H*S, Dh], casting to F32.
def _gather_cross_bf16(
    src: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],  # [B*S*H, Dh]
    dst: LayoutTensor[DType.float32,  _DYN2, MutAnyOrigin],  # [B*H*S, Dh]
    B: Int, S: Int, H: Int, Dh: Int,
):
    var idx = Int(global_idx.x)
    var total = B * H * S * Dh
    if idx < total:
        var d = idx % Dh
        var t = idx // Dh      # (b*H+h)*S+s
        var s = t % S
        var t2 = t // S        # b*H+h
        var h = t2 % H
        var b = t2 // H
        var src_row = (b * S + s) * H + h
        var dst_row = (b * H + h) * S + s
        var v = rebind[Scalar[DType.bfloat16]](src[src_row, d]).cast[DType.float32]()
        dst[dst_row, d] = rebind[dst.element_type](v)


# Online-softmax streaming kernel for rectangular Q×KV (Sq rows, Skv cols).
# One thread per (bh, i) = query row.  q/k/v are F32 BHSD contiguous.
# q: [B*H*Sq, Dh],  k: [B*H*Skv, Dh],  v: [B*H*Skv, Dh],  o: [B*H*Sq, Dh]
comptime _DH_MAX_AV = 128

def _sdpa_cross_online_f32(
    q:  LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    k:  LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    v:  LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    o:  LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    scale: Float32,
    B: Int, Sq: Int, Skv: Int, H: Int, Dh: Int,
):
    var qrow = Int(global_idx.x)   # flat (b*H + h)*Sq + i  == query row in q
    var total = B * H * Sq
    if qrow >= total:
        return
    var i   = qrow % Sq
    var bh  = qrow // Sq            # (b*H + h)
    var kbase = bh * Skv            # first kv row of this head in k
    # Load query row into registers.
    var qreg = stack_allocation[_DH_MAX_AV, Scalar[DType.float32]]()
    for d in range(Dh):
        qreg[d] = rebind[Scalar[DType.float32]](q[qrow, d])
    # Online softmax state.
    var acc = stack_allocation[_DH_MAX_AV, Scalar[DType.float32]]()
    for d in range(Dh):
        acc[d] = 0.0
    var m: Float32 = _NEG_BIG
    var l: Float32 = 0.0
    for j in range(Skv):
        var krow = kbase + j
        var dot: Float32 = 0.0
        for d in range(Dh):
            dot += qreg[d] * rebind[Scalar[DType.float32]](k[krow, d])
        var s = dot * scale
        var m_new = m if m > s else s
        var corr = exp(m - m_new)
        var p = exp(s - m_new)
        l = l * corr + p
        for d in range(Dh):
            acc[d] = acc[d] * corr + p * rebind[Scalar[DType.float32]](v[krow, d])
        m = m_new
    var inv = 1.0 / l
    for d in range(Dh):
        o[qrow, d] = rebind[o.element_type](acc[d] * inv)


# Scatter BHSD F32 [B*H*Sq, Dh] -> BSHD bf16 [B*Sq*H, Dh].
def _scatter_cross_bf16(
    src: LayoutTensor[DType.float32,  _DYN2, MutAnyOrigin],  # [B*H*Sq, Dh]
    dst: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],  # [B*Sq*H, Dh]
    B: Int, Sq: Int, H: Int, Dh: Int,
):
    var idx = Int(global_idx.x)
    var total = B * H * Sq * Dh
    if idx < total:
        var d = idx % Dh
        var t = idx // Dh      # (b*H+h)*Sq+s
        var s = t % Sq
        var t2 = t // Sq
        var h = t2 % H
        var b = t2 // H
        var src_row = (b * H + h) * Sq + s
        var dst_row = (b * Sq + s) * H + h
        var v = rebind[Scalar[DType.float32]](src[src_row, d])
        dst[dst_row, d] = rebind[dst.element_type](v.cast[DType.bfloat16]())


# Driver: rectangular cross-attention (no mask, bf16 in/out).
# Q: [B, Sq, H, Dh]  K/V: [B, Skv, H, Dh]  -> [B, Sq, H, Dh] bf16
def sdpa_cross_nomask[
    B: Int, Sq: Int, Skv: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """Rectangular cross-attention SDPA, no additive mask.

    Q: [B, Sq, H, Dh]   K,V: [B, Skv, H, Dh]
    Returns [B, Sq, H, Dh] in q's dtype (bf16).
    Uses online-softmax streaming (O(S*Dh) memory, never [Sq*Skv]).
    Dh must be <= _DH_MAX_AV (128).

    NEW function — additive, does not touch ops/attention.mojo.
    """
    comptime if Dh > _DH_MAX_AV:
        raise Error("sdpa_cross_nomask: Dh exceeds _DH_MAX_AV (128)")

    comptime qsrc_rows = B * Sq * H
    comptime ksrc_rows = B * Skv * H
    comptime qbhsd_rows = B * H * Sq
    comptime kbhsd_rows = B * H * Skv

    var q_f32 = ctx.enqueue_create_buffer[DType.float32](qbhsd_rows * Dh)
    var k_f32 = ctx.enqueue_create_buffer[DType.float32](kbhsd_rows * Dh)
    var v_f32 = ctx.enqueue_create_buffer[DType.float32](kbhsd_rows * Dh)

    var qsrc_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](qsrc_rows, Dh))
    var ksrc_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](ksrc_rows, Dh))
    var qbhsd_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](qbhsd_rows, Dh))
    var kbhsd_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](kbhsd_rows, Dh))

    var Qd = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](q_f32.unsafe_ptr(), qbhsd_rl)
    var Kd = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](k_f32.unsafe_ptr(), kbhsd_rl)
    var Vd = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](v_f32.unsafe_ptr(), kbhsd_rl)

    # Gather Q (Sq sequence), K and V (Skv sequence) -> BHSD F32.
    var ng_q = B * H * Sq * Dh
    var ng_k = B * H * Skv * Dh
    var gg_q = (ng_q + _BLOCK - 1) // _BLOCK
    var gg_k = (ng_k + _BLOCK - 1) // _BLOCK

    var Qs = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        q.buf.unsafe_ptr().bitcast[BFloat16](), qsrc_rl)
    var Ks = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        k.buf.unsafe_ptr().bitcast[BFloat16](), ksrc_rl)
    var Vs = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        v.buf.unsafe_ptr().bitcast[BFloat16](), ksrc_rl)

    ctx.enqueue_function[_gather_cross_bf16, _gather_cross_bf16](
        Qs, Qd, B, Sq, H, Dh, grid_dim=gg_q, block_dim=_BLOCK)
    ctx.enqueue_function[_gather_cross_bf16, _gather_cross_bf16](
        Ks, Kd, B, Skv, H, Dh, grid_dim=gg_k, block_dim=_BLOCK)
    ctx.enqueue_function[_gather_cross_bf16, _gather_cross_bf16](
        Vs, Vd, B, Skv, H, Dh, grid_dim=gg_k, block_dim=_BLOCK)

    # Streaming attention: one thread per (bh, query-row).
    var out_f32 = ctx.enqueue_create_buffer[DType.float32](qbhsd_rows * Dh)
    var Od = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        out_f32.unsafe_ptr(), qbhsd_rl)
    var nq = B * H * Sq
    var qgrid = (nq + _TPB - 1) // _TPB
    ctx.enqueue_function[_sdpa_cross_online_f32, _sdpa_cross_online_f32](
        Qd, Kd, Vd, Od, scale, B, Sq, Skv, H, Dh,
        grid_dim=qgrid, block_dim=_TPB)

    # Scatter BHSD F32 -> BSHD bf16.
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](B * Sq * H * Dh * 2)
    var out_src = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        out_f32.unsafe_ptr(), qbhsd_rl)
    var nsc = B * H * Sq * Dh
    var scgrid = (nsc + _BLOCK - 1) // _BLOCK
    var Out = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), qsrc_rl)
    ctx.enqueue_function[_scatter_cross_bf16, _scatter_cross_bf16](
        out_src, Out, B, Sq, H, Dh, grid_dim=scgrid, block_dim=_BLOCK)

    return Tensor(out_buf^, _sh4(B, Sq, H, Dh), STDtype.BF16)


# ── Self-attention path (shared by video and audio) ───────────────────────────
# Implements: AdaLN -> rms -> modulate -> Q/K/V linear -> QK-RMSNorm ->
#             reshape BSHD -> RoPE -> SDPA -> per-head gate -> to_out ->
#             gated residual.
#
# All tensors are bf16. scale_shift_table rows 0:3 used (shift_msa, scale_msa, gate_msa).
# Returns updated hidden state (same shape as `x`).
def _self_attn_path[B: Int, S: Int, H: Int, Dh: Int](
    x: Tensor,          # [B, S, D]  bf16
    table: Tensor,      # [9, D]     bf16   (scale_shift_table)
    temb: Tensor,       # [B, 9*D]   bf16   (timestep embed)
    wq: Tensor, bq: Tensor,
    wk: Tensor, bk: Tensor,
    wv: Tensor, bv: Tensor,
    q_norm_w: Tensor, k_norm_w: Tensor,
    gate_w: Tensor, gate_b: Tensor,
    wo: Tensor, bo: Tensor,
    rope_cos: Tensor, rope_sin: Tensor,
    eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    var D = H * Dh
    # AdaLN: rows 0=shift_msa, 1=scale_msa, 2=gate_msa
    var shift_msa = _ada_vec(table, temb, 0, D, ctx)   # [B, D]
    var scale_msa = _ada_vec(table, temb, 1, D, ctx)
    var gate_msa  = _ada_vec(table, temb, 2, D, ctx)

    # rms_no_affine + modulate (F32 internally via _modulate_bsd).
    var norm_x = _rms_no_affine(x, eps, ctx)            # [B, S, D]
    var mod_x  = _modulate_bsd(norm_x, scale_msa, shift_msa, B, S, D, ctx)

    # Q, K, V projections.
    var q = linear(mod_x, wq, Optional[Tensor](_clone_t(bq, ctx)), ctx)   # [B,S,D]
    var k = linear(mod_x, wk, Optional[Tensor](_clone_t(bk, ctx)), ctx)
    var v = linear(mod_x, wv, Optional[Tensor](_clone_t(bv, ctx)), ctx)

    # QK-RMSNorm over full inner_dim D (with learned scale).
    q = rms_norm(q, q_norm_w, eps, ctx)
    k = rms_norm(k, k_norm_w, eps, ctx)

    # Reshape -> BSHD [B, S, H, Dh].
    var q4 = reshape(q, _sh4(B, S, H, Dh), ctx)
    var k4 = reshape(k, _sh4(B, S, H, Dh), ctx)
    var v4 = reshape(v, _sh4(B, S, H, Dh), ctx)

    # RoPE interleaved (22B AV model uses rope_type="interleaved", NOT halfsplit).
    # Applied ONLY to self-attn Q/K; NOT to cross-attn Q/K.
    q4 = rope_interleaved(q4, rope_cos, rope_sin, ctx)
    k4 = rope_interleaved(k4, rope_cos, rope_sin, ctx)

    # SDPA (self-attn: Sq==Skv==S, no mask).
    # NOTE: sdpa_nomask from ops/attention.mojo requires Sq==Skv — correct here.
    # We reuse sdpa_cross_nomask for generality since it also handles Sq==Skv.
    var attn = sdpa_cross_nomask[B, S, S, H, Dh](q4, k4, v4, Float32(1.0) / sqrt(Float32(Dh)), ctx)
    var attn_flat = reshape(attn, _sh3(B, S, D), ctx)

    # Per-head gate (input = mod_x, same as q/k/v source).
    attn_flat = _apply_head_gate[S, H, Dh](
        attn_flat, mod_x, gate_w, _clone_t(gate_b, ctx), B, ctx
    )

    # to_out projection.
    var attn_out = linear(attn_flat, wo, Optional[Tensor](_clone_t(bo, ctx)), ctx)

    # Gated residual: x + gate_msa * attn_out.
    # gate_msa [B, D] broadcasts over [B, S, D].
    var g4 = reshape(gate_msa, _sh3(B, 1, D), ctx)
    return add(x, mul(attn_out, g4, ctx), ctx)


# ── Text cross-attention path ─────────────────────────────────────────────────
# Implements: AdaLN rows 6:9 -> rms -> modulate_q -> optional KV modulate
#             (prompt_scale_shift_table + prompt_temb) -> cross-SDPA ->
#             per-head gate -> to_out -> residual.
#
# prompt_scale_shift_table [2, D]:  rows = shift_kv, scale_kv.
# prompt_temb [B, 2*D]: per-batch KV modulation delta.
# context [B, N_TXT, D]: text encoder hidden states.
#
# Returns updated hidden state [B, S, D].
def _cross_attn_path[B: Int, S: Int, N_TXT: Int, H: Int, Dh: Int](
    x: Tensor,           # [B, S, D]      bf16
    context: Tensor,     # [B, N_TXT, D]  bf16  text encoder output
    table: Tensor,       # [9, D]         bf16
    temb: Tensor,        # [B, 9*D]       bf16
    prompt_table: Tensor, # [2, D]        bf16
    prompt_temb: Tensor,  # [B, 2*D]      bf16
    wq: Tensor, bq: Tensor,
    wk: Tensor, bk: Tensor,
    wv: Tensor, bv: Tensor,
    q_norm_w: Tensor, k_norm_w: Tensor,
    gate_w: Tensor, gate_b: Tensor,
    wo: Tensor, bo: Tensor,
    eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    var D = H * Dh
    # AdaLN rows 6=shift_ca_q, 7=scale_ca_q, 8=gate_ca.
    var shift_ca = _ada_vec(table, temb, 6, D, ctx)   # [B, D]
    var scale_ca = _ada_vec(table, temb, 7, D, ctx)
    var gate_ca  = _ada_vec(table, temb, 8, D, ctx)

    # Q modulation: rms(x) -> modulate.
    var norm_x = _rms_no_affine(x, eps, ctx)
    var mod_q = _modulate_bsd(norm_x, scale_ca, shift_ca, B, S, D, ctx)

    # Q projection + QK-RMSNorm.
    var q = linear(mod_q, wq, Optional[Tensor](_clone_t(bq, ctx)), ctx)
    q = rms_norm(q, q_norm_w, eps, ctx)

    # KV modulation via prompt_scale_shift_table.
    # prompt_table[0] = shift_kv, prompt_table[1] = scale_kv.
    var shift_kv = _ada_vec(prompt_table, prompt_temb, 0, D, ctx)   # [B, D]
    var scale_kv = _ada_vec(prompt_table, prompt_temb, 1, D, ctx)
    var kv_sc4 = reshape(scale_kv, _sh3(B, 1, D), ctx)
    var kv_sh4 = reshape(shift_kv, _sh3(B, 1, D), ctx)
    var one_plus_kv = add_scalar(kv_sc4, Float32(1.0), ctx)
    var context_mod = add(mul(context, one_plus_kv, ctx), kv_sh4, ctx)  # [B, N_TXT, D]

    # K, V projections from modulated context.
    var k = linear(context_mod, wk, Optional[Tensor](_clone_t(bk, ctx)), ctx)
    var v = linear(context_mod, wv, Optional[Tensor](_clone_t(bv, ctx)), ctx)
    k = rms_norm(k, k_norm_w, eps, ctx)

    # Reshape to BSHD (note: no RoPE on cross-attn).
    var q4 = reshape(q, _sh4(B, S, H, Dh), ctx)
    var k4 = reshape(k, _sh4(B, N_TXT, H, Dh), ctx)
    var v4 = reshape(v, _sh4(B, N_TXT, H, Dh), ctx)

    # Cross-attention SDPA: Q from x (Sq=S), KV from context (Skv=N_TXT).
    var attn = sdpa_cross_nomask[B, S, N_TXT, H, Dh](
        q4, k4, v4, Float32(1.0) / sqrt(Float32(Dh)), ctx
    )
    var attn_flat = reshape(attn, _sh3(B, S, D), ctx)

    # Per-head gate (input = mod_q).
    attn_flat = _apply_head_gate[S, H, Dh](
        attn_flat, mod_q, gate_w, _clone_t(gate_b, ctx), B, ctx
    )

    # to_out projection.
    var attn_out = linear(attn_flat, wo, Optional[Tensor](_clone_t(bo, ctx)), ctx)

    # Residual: x + gate_ca * attn_out.
    var g4 = reshape(gate_ca, _sh3(B, 1, D), ctx)
    return add(x, mul(attn_out, g4, ctx), ctx)


# ── Cross-modal attention (audio_to_video or video_to_audio) ─────────────────
# Musubi _forward pre-normalizes vx and ax ONCE (vx_norm3, ax_norm3) before both
# a2v and v2a paths. We receive pre-normalized tensors here and apply the
# AdaLN modulation inside. NO re-normalization.
#
# norm_xq:  pre-normalized query stream  [B, Sq, Dout] (e.g. vx_norm3 for a2v)
# norm_xkv: pre-normalized kv stream     [B, Skv, Dkv] (e.g. ax_norm3 for a2v)
# xq_orig:  un-normalized residual       [B, Sq, Dout] (vx2 for a2v, ax2 for v2a)
# scale_q/shift_q: [B, Dout] AdaLN modulation for the query stream
# scale_kv/shift_kv: [B, Dkv] AdaLN modulation for the kv stream
# gate: [B, Dout] AdaLN gate
# to_q: [H*Dh, Dout],  to_k/v: [H*Dh, Dkv],  to_out: [Dout, H*Dh].
# rope_q_cos/sin: [Sq, H*Dh]  cross-modal RoPE for Q (interleaved).
# rope_k_cos/sin: [Skv, H*Dh] cross-modal RoPE for K.
# gate_w: [H, Dout]  gate_b: [H]  per-head gate (musubi to_gate_logits on mod_xq).
# Returns updated hidden state for the query stream [B, Sq, Dout].
def _cross_modal_attn[B: Int, Sq: Int, Skv: Int, H: Int, Dh: Int, Dout: Int, Dkv: Int](
    norm_xq:  Tensor,    # [B, Sq, Dout]  bf16  pre-normalized query stream
    norm_xkv: Tensor,    # [B, Skv, Dkv]  bf16  pre-normalized kv stream
    xq_orig:  Tensor,    # [B, Sq, Dout]  bf16  residual (un-normalized)
    scale_q: Tensor, shift_q: Tensor,      # [B, Dout] AdaLN for query stream
    scale_kv: Tensor, shift_kv: Tensor,    # [B, Dkv]  AdaLN for kv stream
    gate: Tensor,                           # [B, Dout] AdaLN gate
    wq: Tensor, bq: Tensor,                # [H*Dh, Dout] / [H*Dh]
    wk: Tensor, bk: Tensor,                # [H*Dh, Dkv]  / [H*Dh]
    wv: Tensor, bv: Tensor,                # [H*Dh, Dkv]  / [H*Dh]
    q_norm_w: Tensor, k_norm_w: Tensor,    # [H*Dh]
    wo: Tensor, bo: Tensor,                # [Dout, H*Dh] / [Dout]
    # BUG1 FIX: cross-modal RoPE tables (interleaved, matching musubi attention.py:424-426)
    rope_q_cos: Tensor, rope_q_sin: Tensor,   # [Sq, H*Dh] bf16  cross PE for Q
    rope_k_cos: Tensor, rope_k_sin: Tensor,   # [Skv, H*Dh] bf16 cross PE for K
    # BUG2 FIX: per-head gate (musubi to_gate_logits applied on mod_xq)
    gate_w: Tensor,    # [H, Dout]
    var gate_b: Tensor, # [H]
    eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    # Modulate query stream (F32 internally via broadcast add/mul).
    var sq4 = reshape(scale_q, _sh3(B, 1, Dout), ctx)
    var sh4 = reshape(shift_q, _sh3(B, 1, Dout), ctx)
    var one_plus_q = add_scalar(sq4, Float32(1.0), ctx)
    var mod_xq = add(mul(norm_xq, one_plus_q, ctx), sh4, ctx)  # [B, Sq, Dout]

    # Modulate kv stream.
    var sk4  = reshape(scale_kv, _sh3(B, 1, Dkv), ctx)
    var shk4 = reshape(shift_kv, _sh3(B, 1, Dkv), ctx)
    var one_plus_kv = add_scalar(sk4, Float32(1.0), ctx)
    var mod_xkv = add(mul(norm_xkv, one_plus_kv, ctx), shk4, ctx)  # [B, Skv, Dkv]

    # Q/K/V projections.
    var q = linear(mod_xq, wq, Optional[Tensor](_clone_t(bq, ctx)), ctx)    # [B,Sq,H*Dh]
    var k = linear(mod_xkv, wk, Optional[Tensor](_clone_t(bk, ctx)), ctx)   # [B,Skv,H*Dh]
    var v = linear(mod_xkv, wv, Optional[Tensor](_clone_t(bv, ctx)), ctx)   # [B,Skv,H*Dh]

    # QK-RMSNorm (with learned scale).
    q = rms_norm(q, q_norm_w, eps, ctx)
    k = rms_norm(k, k_norm_w, eps, ctx)

    # BSHD reshape.
    var q4 = reshape(q, _sh4(B, Sq, H, Dh), ctx)
    var k4 = reshape(k, _sh4(B, Skv, H, Dh), ctx)
    var v4 = reshape(v, _sh4(B, Skv, H, Dh), ctx)

    # BUG1 FIX: Apply interleaved RoPE to cross-modal Q and K.
    # musubi attention.py lines 424-426: q = apply_rotary_emb(q, pe); k = apply_rotary_emb(k, k_pe)
    # A2V: Q gets video.cross_positional_embeddings, K gets audio.cross_positional_embeddings.
    # V2A: Q gets audio.cross_positional_embeddings, K gets video.cross_positional_embeddings.
    # rope_q_cos/sin [Sq, H*Dh], rope_k_cos/sin [Skv, H*Dh] (passed in per-call).
    q4 = rope_interleaved(q4, rope_q_cos, rope_q_sin, ctx)
    k4 = rope_interleaved(k4, rope_k_cos, rope_k_sin, ctx)

    # Cross-SDPA.
    var attn = sdpa_cross_nomask[B, Sq, Skv, H, Dh](
        q4, k4, v4, Float32(1.0) / sqrt(Float32(Dh)), ctx
    )
    var attn_flat = reshape(attn, _sh3(B, Sq, H * Dh), ctx)  # [B,Sq,H*Dh]

    # BUG2 FIX: Per-head gate (musubi attention.py lines 479-488).
    # gate_logits = to_gate_logits(x) where x = mod_xq (the modulated Q input).
    # gate_w [H, Dout], mod_xq [B, Sq, Dout] -> gates [B, Sq, H] -> scale attn heads.
    # Dout may differ from H*Dh for cross-modal (e.g. a2v: Dout=4096, H*Dh=2048).
    attn_flat = _apply_head_gate_cross[Sq, H, Dh, Dout](
        attn_flat, mod_xq, gate_w, gate_b^, B, ctx
    )

    # to_out projection -> [B,Sq,Dout].
    var attn_out = linear(attn_flat, wo, Optional[Tensor](_clone_t(bo, ctx)), ctx)

    # Gated residual: xq_orig + gate * attn_out.
    var g4 = reshape(gate, _sh3(B, 1, Dout), ctx)
    return add(xq_orig, mul(attn_out, g4, ctx), ctx)


# ── FFN path (shared by video and audio) ─────────────────────────────────────
# Uses AdaLN rows 3:6 (shift_mlp, scale_mlp, gate_mlp).
# FFN: GELU (tanh-approx). Output clamped to ±60000 (bf16 overflow guard).
def _ffn_path[B: Int, S: Int](
    x: Tensor,      # [B, S, D]   bf16
    table: Tensor,  # [9, D]      bf16
    temb: Tensor,   # [B, 9*D]    bf16
    wff0: Tensor, bff0: Tensor,   # [FF, D] / [FF]
    wff2: Tensor, bff2: Tensor,   # [D, FF] / [D]
    D: Int, FF: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    # AdaLN rows 3=shift_mlp, 4=scale_mlp, 5=gate_mlp.
    var shift_mlp = _ada_vec(table, temb, 3, D, ctx)
    var scale_mlp = _ada_vec(table, temb, 4, D, ctx)
    var gate_mlp  = _ada_vec(table, temb, 5, D, ctx)

    # rms_no_affine + modulate.
    var norm_x = _rms_no_affine(x, eps, ctx)
    var mod_x  = _modulate_bsd(norm_x, scale_mlp, shift_mlp, B, S, D, ctx)

    # FFN: linear -> GELU -> linear.
    var h1 = linear(mod_x, wff0, Optional[Tensor](_clone_t(bff0, ctx)), ctx)   # [B,S,FF]
    var h1g = gelu(h1, ctx)
    var ff = linear(h1g, wff2, Optional[Tensor](_clone_t(bff2, ctx)), ctx)     # [B,S,D]

    # Clamp output to ±60000 (bf16 overflow guard; musubi default ffn_clamp=60000).
    # We implement clamp via add_scalar min/max. No clamp op in tensor_algebra —
    # use a simple elementwise approach: clamp via max(-60000, min(60000, ff)).
    # FLAG: ops/tensor_algebra.mojo has no clamp; implementing inline below.
    ff = _clamp_60k(ff, ctx)

    # Gated residual: x + gate_mlp * ff.
    var g4 = reshape(gate_mlp, _sh3(B, 1, D), ctx)
    return add(x, mul(ff, g4, ctx), ctx)


# ── FFN clamp kernel ±60000 ───────────────────────────────────────────────────
comptime _CLAMP_HI = Float32(60000.0)

def _clamp60k_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var idx = Int(global_idx.x)
    if idx < n:
        var v = rebind[Scalar[DType.bfloat16]](x[idx]).cast[DType.float32]()
        var c = v if v < _CLAMP_HI else _CLAMP_HI
        c = c if c > -_CLAMP_HI else -_CLAMP_HI
        o[idx] = rebind[o.element_type](c.cast[DType.bfloat16]())


def _clamp_60k(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var Xi = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[BFloat16](), rl)
    var Oi = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), rl)
    ctx.enqueue_function[_clamp60k_kernel_bf16, _clamp60k_kernel_bf16](
        Xi, Oi, n, grid_dim=grid, block_dim=_BLOCK)
    return Tensor(out_buf^, x.shape(), x.dtype())


# ── Main AV block weights (all Tensors, bf16) ─────────────────────────────────
# Weights for ONE BasicAVTransformerBlock with both video+audio+cross-modal paths.
# Ownership: moved in, held by ArcPointer or passed by ref.
struct AVBlockWeights(Movable):
    """Full weight set for one LTX-2.3 22B AV transformer block."""

    # Video self-attn (attn1): Dv=4096, H=32, Dh=128
    var v_wq: Tensor; var v_bq: Tensor
    var v_wk: Tensor; var v_bk: Tensor
    var v_wv: Tensor; var v_bv: Tensor
    var v_qn: Tensor; var v_kn: Tensor          # [4096] q/k rms norm scale
    var v_gate_w: Tensor; var v_gate_b: Tensor  # [32,4096] / [32]
    var v_wo: Tensor; var v_bo: Tensor

    # Video cross-attn (attn2, text): same Dv, N_TXT-aware shapes
    var v2_wq: Tensor; var v2_bq: Tensor
    var v2_wk: Tensor; var v2_bk: Tensor
    var v2_wv: Tensor; var v2_bv: Tensor
    var v2_qn: Tensor; var v2_kn: Tensor
    var v2_gate_w: Tensor; var v2_gate_b: Tensor
    var v2_wo: Tensor; var v2_bo: Tensor

    # Video FFN: ff [16384,4096] / ff2 [4096,16384]
    var v_wff0: Tensor; var v_bff0: Tensor
    var v_wff2: Tensor; var v_bff2: Tensor

    # Video AdaLN tables
    var v_table: Tensor          # [9, 4096]
    var v_prompt_table: Tensor   # [2, 4096]
    var v_a2v_table: Tensor      # [5, 4096]

    # Audio self-attn (audio_attn1): Da=2048, Ha=32, Dh_a=64
    var a_wq: Tensor; var a_bq: Tensor
    var a_wk: Tensor; var a_bk: Tensor
    var a_wv: Tensor; var a_bv: Tensor
    var a_qn: Tensor; var a_kn: Tensor
    var a_gate_w: Tensor; var a_gate_b: Tensor
    var a_wo: Tensor; var a_bo: Tensor

    # Audio cross-attn (audio_attn2, text)
    var a2_wq: Tensor; var a2_bq: Tensor
    var a2_wk: Tensor; var a2_bk: Tensor
    var a2_wv: Tensor; var a2_bv: Tensor
    var a2_qn: Tensor; var a2_kn: Tensor
    var a2_gate_w: Tensor; var a2_gate_b: Tensor
    var a2_wo: Tensor; var a2_bo: Tensor

    # Audio FFN: aff0 [8192,2048] / aff2 [2048,8192]
    var a_wff0: Tensor; var a_bff0: Tensor
    var a_wff2: Tensor; var a_bff2: Tensor

    # Audio AdaLN tables
    var a_table: Tensor          # [9, 2048]
    var a_prompt_table: Tensor   # [2, 2048]
    var a_a2v_table: Tensor      # [5, 2048]

    # Cross-modal: audio_to_video (Q=video, KV=audio)
    # to_q: [2048,4096], to_k: [2048,2048], to_v: [2048,2048], to_out: [4096,2048]
    var a2v_wq: Tensor; var a2v_bq: Tensor   # [H*Dh_a, Dv] = [2048,4096]
    var a2v_wk: Tensor; var a2v_bk: Tensor   # [H*Dh_a, Da] = [2048,2048]
    var a2v_wv: Tensor; var a2v_bv: Tensor
    var a2v_qn: Tensor; var a2v_kn: Tensor   # [2048]
    var a2v_wo: Tensor; var a2v_bo: Tensor   # [Dv, H*Dh_a] = [4096,2048]
    # BUG2 FIX: per-head gate for audio_to_video_attn
    # checkpoint key: audio_to_video_attn.to_gate_logits.weight [32,4096] / bias [32]
    # gate_in = mod_xq = modulated video stream [B,Sv,Dv=4096]
    var a2v_gate_w: Tensor   # [H=32, Dv=4096]
    var a2v_gate_b: Tensor   # [H=32]

    # Cross-modal: video_to_audio (Q=audio, KV=video)
    # to_q: [2048,2048], to_k: [2048,4096], to_v: [2048,4096], to_out: [2048,2048]
    var v2a_wq: Tensor; var v2a_bq: Tensor
    var v2a_wk: Tensor; var v2a_bk: Tensor
    var v2a_wv: Tensor; var v2a_bv: Tensor
    var v2a_qn: Tensor; var v2a_kn: Tensor
    var v2a_wo: Tensor; var v2a_bo: Tensor
    # BUG2 FIX: per-head gate for video_to_audio_attn
    # checkpoint key: video_to_audio_attn.to_gate_logits.weight [32,2048] / bias [32]
    # gate_in = mod_xq = modulated audio stream [B,Sa,Da=2048]
    var v2a_gate_w: Tensor   # [H=32, Da=2048]
    var v2a_gate_b: Tensor   # [H=32]

    def __init__(
        out self,
        # Video self-attn
        var v_wq: Tensor, var v_bq: Tensor,
        var v_wk: Tensor, var v_bk: Tensor,
        var v_wv: Tensor, var v_bv: Tensor,
        var v_qn: Tensor, var v_kn: Tensor,
        var v_gate_w: Tensor, var v_gate_b: Tensor,
        var v_wo: Tensor, var v_bo: Tensor,
        # Video cross-attn
        var v2_wq: Tensor, var v2_bq: Tensor,
        var v2_wk: Tensor, var v2_bk: Tensor,
        var v2_wv: Tensor, var v2_bv: Tensor,
        var v2_qn: Tensor, var v2_kn: Tensor,
        var v2_gate_w: Tensor, var v2_gate_b: Tensor,
        var v2_wo: Tensor, var v2_bo: Tensor,
        # Video FFN
        var v_wff0: Tensor, var v_bff0: Tensor,
        var v_wff2: Tensor, var v_bff2: Tensor,
        # Video AdaLN tables
        var v_table: Tensor, var v_prompt_table: Tensor, var v_a2v_table: Tensor,
        # Audio self-attn
        var a_wq: Tensor, var a_bq: Tensor,
        var a_wk: Tensor, var a_bk: Tensor,
        var a_wv: Tensor, var a_bv: Tensor,
        var a_qn: Tensor, var a_kn: Tensor,
        var a_gate_w: Tensor, var a_gate_b: Tensor,
        var a_wo: Tensor, var a_bo: Tensor,
        # Audio cross-attn
        var a2_wq: Tensor, var a2_bq: Tensor,
        var a2_wk: Tensor, var a2_bk: Tensor,
        var a2_wv: Tensor, var a2_bv: Tensor,
        var a2_qn: Tensor, var a2_kn: Tensor,
        var a2_gate_w: Tensor, var a2_gate_b: Tensor,
        var a2_wo: Tensor, var a2_bo: Tensor,
        # Audio FFN
        var a_wff0: Tensor, var a_bff0: Tensor,
        var a_wff2: Tensor, var a_bff2: Tensor,
        # Audio AdaLN tables
        var a_table: Tensor, var a_prompt_table: Tensor, var a_a2v_table: Tensor,
        # Cross-modal a2v
        var a2v_wq: Tensor, var a2v_bq: Tensor,
        var a2v_wk: Tensor, var a2v_bk: Tensor,
        var a2v_wv: Tensor, var a2v_bv: Tensor,
        var a2v_qn: Tensor, var a2v_kn: Tensor,
        var a2v_wo: Tensor, var a2v_bo: Tensor,
        # BUG2 FIX: per-head gate for audio_to_video_attn
        var a2v_gate_w: Tensor, var a2v_gate_b: Tensor,
        # Cross-modal v2a
        var v2a_wq: Tensor, var v2a_bq: Tensor,
        var v2a_wk: Tensor, var v2a_bk: Tensor,
        var v2a_wv: Tensor, var v2a_bv: Tensor,
        var v2a_qn: Tensor, var v2a_kn: Tensor,
        var v2a_wo: Tensor, var v2a_bo: Tensor,
        # BUG2 FIX: per-head gate for video_to_audio_attn
        var v2a_gate_w: Tensor, var v2a_gate_b: Tensor,
    ):
        self.v_wq = v_wq^; self.v_bq = v_bq^
        self.v_wk = v_wk^; self.v_bk = v_bk^
        self.v_wv = v_wv^; self.v_bv = v_bv^
        self.v_qn = v_qn^; self.v_kn = v_kn^
        self.v_gate_w = v_gate_w^; self.v_gate_b = v_gate_b^
        self.v_wo = v_wo^; self.v_bo = v_bo^
        self.v2_wq = v2_wq^; self.v2_bq = v2_bq^
        self.v2_wk = v2_wk^; self.v2_bk = v2_bk^
        self.v2_wv = v2_wv^; self.v2_bv = v2_bv^
        self.v2_qn = v2_qn^; self.v2_kn = v2_kn^
        self.v2_gate_w = v2_gate_w^; self.v2_gate_b = v2_gate_b^
        self.v2_wo = v2_wo^; self.v2_bo = v2_bo^
        self.v_wff0 = v_wff0^; self.v_bff0 = v_bff0^
        self.v_wff2 = v_wff2^; self.v_bff2 = v_bff2^
        self.v_table = v_table^; self.v_prompt_table = v_prompt_table^
        self.v_a2v_table = v_a2v_table^
        self.a_wq = a_wq^; self.a_bq = a_bq^
        self.a_wk = a_wk^; self.a_bk = a_bk^
        self.a_wv = a_wv^; self.a_bv = a_bv^
        self.a_qn = a_qn^; self.a_kn = a_kn^
        self.a_gate_w = a_gate_w^; self.a_gate_b = a_gate_b^
        self.a_wo = a_wo^; self.a_bo = a_bo^
        self.a2_wq = a2_wq^; self.a2_bq = a2_bq^
        self.a2_wk = a2_wk^; self.a2_bk = a2_bk^
        self.a2_wv = a2_wv^; self.a2_bv = a2_bv^
        self.a2_qn = a2_qn^; self.a2_kn = a2_kn^
        self.a2_gate_w = a2_gate_w^; self.a2_gate_b = a2_gate_b^
        self.a2_wo = a2_wo^; self.a2_bo = a2_bo^
        self.a_wff0 = a_wff0^; self.a_bff0 = a_bff0^
        self.a_wff2 = a_wff2^; self.a_bff2 = a_bff2^
        self.a_table = a_table^; self.a_prompt_table = a_prompt_table^
        self.a_a2v_table = a_a2v_table^
        self.a2v_wq = a2v_wq^; self.a2v_bq = a2v_bq^
        self.a2v_wk = a2v_wk^; self.a2v_bk = a2v_bk^
        self.a2v_wv = a2v_wv^; self.a2v_bv = a2v_bv^
        self.a2v_qn = a2v_qn^; self.a2v_kn = a2v_kn^
        self.a2v_wo = a2v_wo^; self.a2v_bo = a2v_bo^
        self.a2v_gate_w = a2v_gate_w^; self.a2v_gate_b = a2v_gate_b^
        self.v2a_wq = v2a_wq^; self.v2a_bq = v2a_bq^
        self.v2a_wk = v2a_wk^; self.v2a_bk = v2a_bk^
        self.v2a_wv = v2a_wv^; self.v2a_bv = v2a_bv^
        self.v2a_qn = v2a_qn^; self.v2a_kn = v2a_kn^
        self.v2a_wo = v2a_wo^; self.v2a_bo = v2a_bo^
        self.v2a_gate_w = v2a_gate_w^; self.v2a_gate_b = v2a_gate_b^


# ── FORWARD OUTPUT ────────────────────────────────────────────────────────────
struct AVBlockOut(Movable):
    """Output of one AV block forward: updated video + audio hidden states."""
    var vx: Tensor   # [B, Sv, 4096] bf16
    var ax: Tensor   # [B, Sa, 2048] bf16

    def __init__(out self, var vx: Tensor, var ax: Tensor):
        self.vx = vx^
        self.ax = ax^


# ── MAIN FORWARD ─────────────────────────────────────────────────────────────
# Compile-time parameters:
#   B    = batch size
#   Sv   = video token sequence length
#   N_TXT = text (encoder) token count (used for both video and audio cross-attn)
#   Sa   = audio token sequence length
#
# Video: Dv=4096, Hv=32, Dh_v=128.  Audio: Da=2048, Ha=32, Dh_a=64.
# Cross-modal head count = Ha=32, Dh = Dh_a=64 (audio head geometry).
#
# temb inputs (timestep embedding from adaln_single + sigma conditioning):
#   vtemb:         [B, 9*4096]  for video scale_shift_table (9 rows)
#   atemb:         [B, 9*2048]  for audio scale_shift_table
#   vprompt_temb:  [B, 2*4096]  for video prompt_scale_shift_table
#   aprompt_temb:  [B, 2*2048]  for audio prompt_scale_shift_table
#   vcross_ss_temb:[B, 4*4096]  for a2v_ca_video rows 0-3 (scale-shift)
#   vcross_g_temb: [B, 1*4096]  for a2v_ca_video row 4 (gate)
#   across_ss_temb:[B, 4*2048]  for a2v_ca_audio rows 0-3
#   across_g_temb: [B, 1*2048]  for a2v_ca_audio row 4
#
# The a2v AdaLN values (5-row tables) use get_av_ca_ada_values with
#   num_scale_shift_values=4: rows 0-3 are ss, row 4 is gate.
# We split them into two separate temb tensors (ss=[B,4*D], g=[B,1*D])
# so _ada_vec can use row_idx = 0..3 for ss and 0 for gate.
def ltx2_av_block_forward[B: Int, Sv: Int, N_TXT: Int, Sa: Int](
    w: AVBlockWeights,
    vx: Tensor,           # [B, Sv, 4096]  bf16
    ax: Tensor,           # [B, Sa, 2048]  bf16
    context_v: Tensor,    # [B, N_TXT, 4096]  bf16  text for video cross-attn
    context_a: Tensor,    # [B, N_TXT, 2048]  bf16  text for audio cross-attn
    vtemb: Tensor,        # [B, 9*4096]     bf16
    atemb: Tensor,        # [B, 9*2048]     bf16
    vprompt_temb: Tensor, # [B, 2*4096]     bf16
    aprompt_temb: Tensor, # [B, 2*2048]     bf16
    vcross_ss_temb: Tensor, # [B, 4*4096]   bf16  a2v_ca_video scale-shift temb
    vcross_g_temb: Tensor,  # [B, 1*4096]   bf16  a2v_ca_video gate temb
    across_ss_temb: Tensor, # [B, 4*2048]   bf16  a2v_ca_audio scale-shift temb
    across_g_temb: Tensor,  # [B, 1*2048]   bf16  a2v_ca_audio gate temb
    vrope_cos: Tensor, vrope_sin: Tensor,   # video self-attn RoPE [Sv, 4096]
    arope_cos: Tensor, arope_sin: Tensor,   # audio self-attn RoPE [Sa, 2048]
    # BUG1 FIX: cross-modal positional embeddings (interleaved RoPE for A2V/V2A).
    # oracle shapes: video_cross_pe_cos/sin [Sv, 2048], audio_cross_pe_cos/sin [Sa, 2048].
    # Both are in audio head-dim space (H*Dh_a = 32*64 = 2048).
    # A2V: Q=video uses vcross_pe_cos/sin; K=audio uses across_pe_cos/sin.
    # V2A: Q=audio uses across_pe_cos/sin; K=video uses vcross_pe_cos/sin.
    vcross_pe_cos: Tensor, vcross_pe_sin: Tensor,  # video cross PE [Sv, 2048]
    across_pe_cos: Tensor, across_pe_sin: Tensor,  # audio cross PE [Sa, 2048]
    eps: Float32,
    ctx: DeviceContext,
) raises -> AVBlockOut:
    comptime Dv = 4096
    comptime Da = 2048
    comptime Hv = 32
    comptime Ha = 32
    comptime Dh_v = 128   # Dv / Hv
    comptime Dh_a = 64    # Da / Ha
    comptime FF_v = 16384
    comptime FF_a = 8192

    # ── 1. Video self-attn ────────────────────────────────────────────────────
    var vx2 = _self_attn_path[B, Sv, Hv, Dh_v](
        vx,
        w.v_table, vtemb,
        w.v_wq, w.v_bq, w.v_wk, w.v_bk, w.v_wv, w.v_bv,
        w.v_qn, w.v_kn,
        w.v_gate_w, w.v_gate_b,
        w.v_wo, w.v_bo,
        vrope_cos, vrope_sin,
        eps, ctx,
    )

    # ── 2. Video cross-attn (text) ────────────────────────────────────────────
    vx2 = _cross_attn_path[B, Sv, N_TXT, Hv, Dh_v](
        vx2, context_v,
        w.v_table, vtemb,
        w.v_prompt_table, vprompt_temb,
        w.v2_wq, w.v2_bq, w.v2_wk, w.v2_bk, w.v2_wv, w.v2_bv,
        w.v2_qn, w.v2_kn,
        w.v2_gate_w, w.v2_gate_b,
        w.v2_wo, w.v2_bo,
        eps, ctx,
    )

    # ── 3. Audio self-attn ────────────────────────────────────────────────────
    var ax2 = _self_attn_path[B, Sa, Ha, Dh_a](
        ax,
        w.a_table, atemb,
        w.a_wq, w.a_bq, w.a_wk, w.a_bk, w.a_wv, w.a_bv,
        w.a_qn, w.a_kn,
        w.a_gate_w, w.a_gate_b,
        w.a_wo, w.a_bo,
        arope_cos, arope_sin,
        eps, ctx,
    )

    # ── 4. Audio cross-attn (text) ────────────────────────────────────────────
    ax2 = _cross_attn_path[B, Sa, N_TXT, Ha, Dh_a](
        ax2, context_a,
        w.a_table, atemb,
        w.a_prompt_table, aprompt_temb,
        w.a2_wq, w.a2_bq, w.a2_wk, w.a2_bk, w.a2_wv, w.a2_bv,
        w.a2_qn, w.a2_kn,
        w.a2_gate_w, w.a2_gate_b,
        w.a2_wo, w.a2_bo,
        eps, ctx,
    )

    # ── 5. Cross-modal attention ──────────────────────────────────────────────
    # Pre-normalize both streams (shared norms, used by both a2v and v2a).
    var vx_norm3 = _rms_no_affine(vx2, eps, ctx)   # [B, Sv, Dv]
    var ax_norm3 = _rms_no_affine(ax2, eps, ctx)   # [B, Sa, Da]

    # a2v AdaLN values from video table (5 rows: 0=sa2v_v, 1=sh2v_v, 2=sv2a_v, 3=sh2a_v, 4=gate_a2v)
    var scale_a2v_v = _ada_vec(w.v_a2v_table, vcross_ss_temb, 0, Dv, ctx)  # [B, Dv]
    var shift_a2v_v = _ada_vec(w.v_a2v_table, vcross_ss_temb, 1, Dv, ctx)
    var scale_v2a_v = _ada_vec(w.v_a2v_table, vcross_ss_temb, 2, Dv, ctx)
    var shift_v2a_v = _ada_vec(w.v_a2v_table, vcross_ss_temb, 3, Dv, ctx)
    var gate_a2v    = _ada_vec(w.v_a2v_table, vcross_g_temb,  0, Dv, ctx)  # row 0 of g-table

    # a2v AdaLN values from audio table (5 rows: 0=sa2v_a, 1=sh2v_a, 2=sv2a_a, 3=sh2a_a, 4=gate_v2a)
    var scale_a2v_a = _ada_vec(w.a_a2v_table, across_ss_temb, 0, Da, ctx)  # [B, Da]
    var shift_a2v_a = _ada_vec(w.a_a2v_table, across_ss_temb, 1, Da, ctx)
    var scale_v2a_a = _ada_vec(w.a_a2v_table, across_ss_temb, 2, Da, ctx)
    var shift_v2a_a = _ada_vec(w.a_a2v_table, across_ss_temb, 3, Da, ctx)
    var gate_v2a    = _ada_vec(w.a_a2v_table, across_g_temb,  0, Da, ctx)

    # 5a. audio_to_video_attn: Q=video, KV=audio.
    # Musubi: vx_scaled = modulate(vx_norm3, scale_a2v_v, shift_a2v_v)
    #         ax_scaled = modulate(ax_norm3, scale_a2v_a, shift_a2v_a)
    #         vx += gate_a2v * audio_to_video_attn(Q=vx_scaled, KV=ax_scaled)
    # to_q: [H*Dh_a, Dv]=[2048,4096]; to_k/v: [H*Dh_a, Da]=[2048,2048]; to_out: [Dv,H*Dh_a]=[4096,2048]
    # SDPA: Sq=Sv, Skv=Sa, H=Ha=32, Dh=Dh_a=64.
    # BUG1: Q=video gets vcross_pe, K=audio gets across_pe (musubi transformer.py:716-717).
    # BUG2: gate_w=a2v_gate_w [32,4096], gate_in=mod_xq=[B,Sv,Dv=4096].
    vx2 = _cross_modal_attn[B, Sv, Sa, Ha, Dh_a, Dv, Da](
        vx_norm3, ax_norm3, vx2,
        scale_a2v_v, shift_a2v_v,   # query (video) modulation
        scale_a2v_a, shift_a2v_a,   # kv (audio) modulation
        gate_a2v,
        w.a2v_wq, w.a2v_bq,
        w.a2v_wk, w.a2v_bk,
        w.a2v_wv, w.a2v_bv,
        w.a2v_qn, w.a2v_kn,
        w.a2v_wo, w.a2v_bo,
        vcross_pe_cos, vcross_pe_sin,    # Q=video cross PE [Sv, 2048]
        across_pe_cos, across_pe_sin,    # K=audio cross PE [Sa, 2048]
        w.a2v_gate_w, _clone_t(w.a2v_gate_b, ctx),
        eps, ctx,
    )

    # 5b. video_to_audio_attn: Q=audio, KV=video.
    # Musubi: ax_scaled = modulate(ax_norm3, scale_v2a_a, shift_v2a_a)
    #         vx_scaled = modulate(vx_norm3, scale_v2a_v, shift_v2a_v)
    #         ax += gate_v2a * video_to_audio_attn(Q=ax_scaled, KV=vx_scaled)
    # to_q: [H*Dh_a, Da]=[2048,2048]; to_k/v: [H*Dh_a, Dv]=[2048,4096]; to_out: [Da,H*Dh_a]=[2048,2048]
    # SDPA: Sq=Sa, Skv=Sv, H=Ha=32, Dh=Dh_a=64.
    # BUG1: Q=audio gets across_pe, K=video gets vcross_pe (musubi transformer.py:740-741).
    # BUG2: gate_w=v2a_gate_w [32,2048], gate_in=mod_xq=[B,Sa,Da=2048].
    ax2 = _cross_modal_attn[B, Sa, Sv, Ha, Dh_a, Da, Dv](
        ax_norm3, vx_norm3, ax2,
        scale_v2a_a, shift_v2a_a,   # query (audio) modulation
        scale_v2a_v, shift_v2a_v,   # kv (video) modulation
        gate_v2a,
        w.v2a_wq, w.v2a_bq,
        w.v2a_wk, w.v2a_bk,
        w.v2a_wv, w.v2a_bv,
        w.v2a_qn, w.v2a_kn,
        w.v2a_wo, w.v2a_bo,
        across_pe_cos, across_pe_sin,    # Q=audio cross PE [Sa, 2048]
        vcross_pe_cos, vcross_pe_sin,    # K=video cross PE [Sv, 2048]
        w.v2a_gate_w, _clone_t(w.v2a_gate_b, ctx),
        eps, ctx,
    )

    # ── 6. Video FFN ──────────────────────────────────────────────────────────
    vx2 = _ffn_path[B, Sv](
        vx2, w.v_table, vtemb,
        w.v_wff0, w.v_bff0, w.v_wff2, w.v_bff2,
        Dv, FF_v, eps, ctx,
    )

    # ── 7. Audio FFN ──────────────────────────────────────────────────────────
    ax2 = _ffn_path[B, Sa](
        ax2, w.a_table, atemb,
        w.a_wff0, w.a_bff0, w.a_wff2, w.a_bff2,
        Da, FF_a, eps, ctx,
    )

    return AVBlockOut(vx2^, ax2^)


# ── Minimal build-check main ──────────────────────────────────────────────────
# Verifies the file compiles (types check, no missing symbols). Does NOT run
# any GPU kernels (that requires a live DeviceContext + real weights).
def main():
    print("ltx2_av_block: compile OK")
