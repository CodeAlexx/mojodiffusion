# serenitymojo/models/dit/minimax_h3_frontend.mojo
#
# MiniMax-H3 DEVICE front-end + output heads — everything outside the 50
# denoiser blocks: the timestep embedding, the three per-modality input
# projections, the 2-layer TEXT-ONLY token refiner, and the final modulated
# norm + two output heads. The 50-block stack itself (and its per-block AdaLN
# modulation cache) is a SEPARATE unit and is not built here.
#
# ORACLES this mirrors (host-float32, already gated — read, not called):
#   models/minimax_h3/dit_frontend.mojo   minimax_h3_timestep_projection (15/15)
#   models/minimax_h3/block_forward.mojo  _token_refiner + the final-layer
#                                          tail of minimax_h3_forward (570-626)
#   models/minimax_h3/rearrange.mojo      minimax_h3_patchify_video (17/17) —
#                                          confirmed BY HAND identical (same
#                                          c-slowest within-patch order, same
#                                          t/h/w-major token order) to
#                                          ops/patchify3d.patchify3d, which is
#                                          what `minimax_h3_video_patchify`
#                                          reuses. NOT independently re-gated
#                                          here; flag this pairing for parity.
#
# WEIGHT-KEY CONTRACT (REVISED — see team-lead correction 2026-08-02): this
# file uses the CHECKPOINT's OWN key spelling throughout, matching
# models/minimax_h3/loader.mojo's actual device-load convention, NOT
# block_forward.mojo's diffusers-renamed oracle names. Rationale (team-lead):
# (1) it is what the checkpoint actually contains — 535 keys, verified
# 0-missing/0-extra against the released model.safetensors.index.json; (2) it
# avoids materializing separate q/k/v (or gate/value) tensors where one fused
# tensor, sliced by row-range at call time, works — real bandwidth/allocation
# savings against a 61.73 GiB streamed checkpoint. So the loader keeps the
# ORIGINAL key and rewrites VALUES in place for exactly two tensors per block:
#   attn.qkv_proj.weight  [3*inner, hidden]  de-interleaved (was per-head
#                          interleaved in the raw file) to CONTIGUOUS q|k|v
#                          blocks: q = rows [0,inner), k = [inner,2*inner),
#                          v = [2*inner,3*inner) — sliced here with ops
#                          `slice(..., dim=0, ...)`, not re-permuted.
#   mlp.fc1.weight         [2*ffn, hidden]  halves swapped from the
#                          checkpoint's [gate; value] to [value; gate] (the
#                          convention swiglu below expects) — value = rows
#                          [0,ffn), gate = rows [ffn,2*ffn).
# Every other key below (norm1/norm2, attn.q_norm/k_norm, attn.out_proj,
# mlp.fc2, video_patch_proj, audio_patch_proj, condition_proj, time_embedder,
# final_layer.*) is the ORIGINAL checkpoint spelling verbatim, no rename, no
# value transform. The 12-fp32 dtype-trap prefixes (see below) are exactly
# the ORIGINAL prefixes the task was given in, unchanged:
#
#   video_patch_proj.        FP32
#   audio_patch_proj.        FP32
#   time_embedder.            FP32  (both .proj_in. and .proj_out.)
#   final_layer.video_out.   FP32
#   final_layer.audio_out.   FP32
#   condition_proj.           bf16
#   final_layer.norm.         bf16
#   final_layer.adaln_proj.   bf16 (NOT loaded here — see "FINAL ADALN" below)
#   token_refiner.blocks.N.   bf16 (attn.qkv_proj / mlp.fc1 values rewritten
#                                   in place by the loader, key unchanged)
#
# Exactly 12 tensors in the released checkpoint are fp32, and all 12 live in
# this file's scope (video/audio patch proj weight+bias, time_embedder
# proj_in/proj_out weight+bias, the two output heads weight+bias). Every
# other tensor this file touches is bf16, INCLUDING final_layer.adaln_proj —
# which is why that one is never loaded here at all (next paragraph).
#
# IMPLIED CAST BOUNDARY (not spelled out by name in the checkpoint, but
# forced by the two facts above): the block stack is bf16-only (every
# `blocks.*` tensor is bf16; a bf16 `linear` call requires a bf16 input),
# while `video_patch_proj`/`audio_patch_proj` are fp32-only. So the fp32
# output of the video/audio patch-proj Linear MUST be cast down to bf16
# before it joins the packed sequence — `minimax_h3_frontend_embed` does this
# explicitly. UNVERIFIED against a real forward; flag for the parity pass.
#
# FINAL ADALN: `final_layer.adaln_proj.linear` ([2*hidden, time_embed_dim] =
# [10752, 2688]) is NOT loaded or computed here. Its output — call it
# `final_modulation`, shape [num_timesteps, 2*hidden_size] — is taken as an
# INPUT to `minimax_h3_final_layer`, indexed PER ROW by that row's OWN
# timestep (`timestep_indices`, the same table the packer built), not by
# (timestep, modality) like the main stack's AdaLN. `minimax_h3_modcache.mojo`
# does not exist yet at the time of writing; when it lands it is the natural
# producer of this table. Declared F32 here (not bf16), matching the sibling
# per-block modulation contract in models/dit/minimax_h3_dit.mojo's header
# ("a [rows, 6*hidden] f32 table") — modulation tables are kept F32 through
# this port even though the Linear that makes them is bf16-stored. UNVERIFIED
# against a real forward.
#
# DTYPE otherwise: bf16 storage, F32 accumulation inside every op (linear,
# vec_rms_norm, sdpa_nomask) per the house rule. `linear` requires x.dtype()
# == weight.dtype() (its one documented mixed exception — F32 x with BF16/F16
# weight — is NOT relied on here; every call site is explicit about which
# dtype it is in). Storage dtype is preserved end to end: the two output
# heads read fp32 input (post-cast, see below) against fp32 weights and
# return fp32 — matching the checkpoint's own storage for those two tensors.
#
# REUSE (do not reimplement — see MAP.md / SERENITYMOJO_MODULES.md):
#   models/dit/minimax_h3_dit.mojo   MiniMaxH3DiTConfig, minimax_h3_released_config
#   ops/linear.mojo                  linear (410)               — all Linear layers
#   ops/vec_rms_norm.mojo            vec_rms_norm (91)           — every RMSNorm
#   ops/embeddings.mojo              t_embedder (245)            — timestep MLP
#                                     (cos-first sinusoidal + Linear-SiLU-Linear;
#                                     verified identical formula to
#                                     dit_frontend.minimax_h3_timestep_projection:
#                                     both compute freq=exp(-ln(max_period)*i/half),
#                                     cos block then sin block, same max_period)
#   ops/attention.mojo               sdpa_nomask (1852)          — refiner self-attn
#                                     (math-mode for all Dh, so Dh=128 is fine on
#                                     this sm_86 GPU where the flash path is not)
#   ops/activations.mojo             swiglu (711)                — refiner FFN
#   ops/tensor_algebra.mojo          reshape (1080), add (548), mul (565),
#                                     add_scalar (766), slice (1910),
#                                     gather_rows (2091), full_device (991)
#   ops/shape_backward.mojo          index_select_backward (1015) — reused as a
#                                     plain scatter (disjoint index sets, so
#                                     scatter-ADD == scatter here)
#   ops/patchify3d.mojo              patchify3d (159)            — video unfold
#   ops/cast.mojo                    cast_tensor (65)             — dtype boundary
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.collections import Dict, List
from std.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.minimax_h3_dit import MiniMaxH3DiTConfig
from serenitymojo.ops.linear import linear
from serenitymojo.ops.vec_rms_norm import vec_rms_norm
from serenitymojo.ops.activations import swiglu
from serenitymojo.ops.embeddings import t_embedder
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.tensor_algebra import reshape, add, mul, add_scalar, slice, gather_rows
from serenitymojo.ops.shape_backward import index_select_backward
from serenitymojo.ops.patchify3d import patchify3d
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.torch_bf16 import torch_f32_to_bf16_rne


# get_timestep_embedding(flip_sin_to_cos=True, downscale_freq_shift=0,
# max_period=10000) — dit_frontend.mojo:44 MINIMAX_H3_TIME_MAX_PERIOD, matched
# here as Float32 for ops/embeddings.t_embedder's parameter type.
comptime MINIMAX_H3_TIME_MAX_PERIOD = Float32(10000.0)


def _lin(
    x: Tensor,
    w: Dict[String, ArcPointer[Tensor]],
    wname: String,
    bname: String,
    ctx: DeviceContext,
) raises -> Tensor:
    """`linear` with both weight and bias pulled from the weight dict by name."""
    return linear(x, w[wname][], Optional(w[bname][].clone(ctx)), ctx)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Timestep embedding: sinusoidal(256) -> proj_in -> silu -> proj_out -> [*,2688]
# ─────────────────────────────────────────────────────────────────────────────
def minimax_h3_timestep_embedding(
    timesteps: Tensor,  # [num_timesteps] F32, UNSCALED in [0,1] (not sigma*1000)
    w: Dict[String, ArcPointer[Tensor]],
    config: MiniMaxH3DiTConfig,
    ctx: DeviceContext,
) raises -> Tensor:
    """block_forward.mojo minimax_h3_forward step 4 (block_forward.mojo:524-542),
    device mirror via ops/embeddings.t_embedder. Returns [num_timesteps,
    time_embed_dim] F32 — `time_embedder.*` is one of the fp32 prefixes, so
    both Linear layers here run and return fp32 with no cast."""
    return t_embedder(
        timesteps,
        config.timestep_input_dim,
        w["time_embedder.proj_in.weight"][],
        Optional(w["time_embedder.proj_in.bias"][].clone(ctx)),
        w["time_embedder.proj_out.weight"][],
        Optional(w["time_embedder.proj_out.bias"][].clone(ctx)),
        ctx,
        MINIMAX_H3_TIME_MAX_PERIOD,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-modality input projections.
# ─────────────────────────────────────────────────────────────────────────────
def minimax_h3_video_patchify(
    latents: Tensor,  # [latents_dim=24, T, H, W]
    ctx: DeviceContext,
) raises -> Tensor:
    """Video patch (1,2,2) unfold -> [T*(H/2)*(W/2), latents_dim*4=96] rows.
    Reuses ops/patchify3d.patchify3d (patch_f=1, patch_h=2, patch_w=2). Its
    within-patch order (c, pf, ph, pw) c-slowest and its token order
    (t-major, then h, then w) were checked by hand against
    models/minimax_h3/rearrange.mojo's minimax_h3_patchify_video (the 17/17
    gated oracle) and found identical — but that pairing is NOT independently
    re-gated by this probe. Caller responsibility: audio has no spatial patch
    (patch is video-only), so audio rows are built elsewhere."""
    return patchify3d(latents, 1, 2, 2, ctx)


def minimax_h3_video_patch_embed(
    video_rows: Tensor,  # [Nv, latents_dim*4=96] F32, already-patchified rows
    w: Dict[String, ArcPointer[Tensor]],
    ctx: DeviceContext,
) raises -> Tensor:
    """video_patch_proj: [Nv,96] -> [Nv,5376]. FP32 in, FP32 out (dtype-trap
    prefix)."""
    return _lin(video_rows, w, "video_patch_proj.weight", "video_patch_proj.bias", ctx)


def minimax_h3_audio_patch_embed(
    audio_rows: Tensor,  # [Na, audio_latents_dim=32] F32, already-packed rows
    w: Dict[String, ArcPointer[Tensor]],
    ctx: DeviceContext,
) raises -> Tensor:
    """audio_patch_proj: [Na,32] -> [Na,5376]. FP32 in, FP32 out (dtype-trap
    prefix). Audio has no spatial unfold (patch is video-only) —
    `audio_rows` is the [2,C,T] latent reshaped to per-stereo-channel rows
    upstream (packing, out of this file's scope; see
    models/minimax_h3/rearrange.mojo's minimax_h3_unpack_audio for the
    INVERSE of that reshape)."""
    return _lin(audio_rows, w, "audio_patch_proj.weight", "audio_patch_proj.bias", ctx)


def minimax_h3_condition_embed(
    text_rows: Tensor,  # [Nt, text_dim=5120] bf16
    w: Dict[String, ArcPointer[Tensor]],
    ctx: DeviceContext,
) raises -> Tensor:
    """condition_proj: [Nt,5120] -> [Nt,5376]. bf16 throughout (NOT a
    dtype-trap prefix)."""
    return _lin(text_rows, w, "condition_proj.weight", "condition_proj.bias", ctx)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Token refiner: 2 plain pre-norm blocks over the TEXT stream only. No
#    AdaLN, no rope (block_forward.mojo _token_refiner, lines 392-444).
# ─────────────────────────────────────────────────────────────────────────────
def _minimax_h3_token_refiner_block[S: Int, H: Int, Dh: Int](
    hidden_in: Tensor,  # [S, hidden_size] bf16
    w: Dict[String, ArcPointer[Tensor]],
    prefix: String,  # e.g. "token_refiner.blocks.0" (NO trailing dot)
    config: MiniMaxH3DiTConfig,
    ctx: DeviceContext,
) raises -> Tensor:
    var inner = config.inner_dim()
    var ffn = config.ffn_hidden_size
    var eps = config.norm_eps
    var qk_eps = config.qk_norm_eps
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var state = hidden_in.clone(ctx)

    # ── self-attention (no rope, no mask, no bias — matches oracle attention()) ──
    # attn.qkv_proj.weight is [3*inner, hidden], loader-de-interleaved to
    # CONTIGUOUS q|k|v row-blocks (was per-head interleaved in the raw file):
    # slice by row-range rather than materializing 3 separate weight copies.
    var normed = vec_rms_norm(state, w[prefix + ".norm1.weight"][], eps, ctx)
    ref qkv_w = w[prefix + ".attn.qkv_proj.weight"][]
    var to_q_w = slice(qkv_w, 0, 0, inner, ctx)
    var to_k_w = slice(qkv_w, 0, inner, inner, ctx)
    var to_v_w = slice(qkv_w, 0, 2 * inner, inner, ctx)
    var q = linear(normed, to_q_w, None, ctx)
    var k = linear(normed, to_k_w, None, ctx)
    var v = linear(normed, to_v_w, None, ctx)

    var q4 = reshape(q, [1, S, H, Dh], ctx)
    var k4 = reshape(k, [1, S, H, Dh], ctx)
    var v4 = reshape(v, [1, S, H, Dh], ctx)
    q4 = vec_rms_norm(q4, w[prefix + ".attn.q_norm.weight"][], qk_eps, ctx)
    k4 = vec_rms_norm(k4, w[prefix + ".attn.k_norm.weight"][], qk_eps, ctx)

    var attn4 = sdpa_nomask[1, S, H, Dh](q4, k4, v4, scale, ctx)
    var attn = reshape(attn4, [S, inner], ctx)
    var attn_out = linear(attn, w[prefix + ".attn.out_proj.weight"][], None, ctx)
    state = add(state, attn_out, ctx)

    # ── SwiGLU FFN. mlp.fc1.weight is [2*ffn, hidden], loader-swapped from
    # the checkpoint's [gate; value] to [value; gate] row-blocks: value =
    # rows [0,ffn), gate = rows [ffn,2*ffn) — sliced the same way as qkv. ──
    var normed2 = vec_rms_norm(state, w[prefix + ".norm2.weight"][], eps, ctx)
    ref fc1_w = w[prefix + ".mlp.fc1.weight"][]
    var value_w = slice(fc1_w, 0, 0, ffn, ctx)
    var gate_w = slice(fc1_w, 0, ffn, ffn, ctx)
    var value = linear(normed2, value_w, None, ctx)
    var gate = linear(normed2, gate_w, None, ctx)
    var ff_act = swiglu(gate, value, ctx)  # silu(gate) * value
    var ff_out = linear(ff_act, w[prefix + ".mlp.fc2.weight"][], None, ctx)
    state = add(state, ff_out, ctx)

    return state^


def minimax_h3_token_refiner[S: Int, H: Int, Dh: Int](
    text_embeds: Tensor,  # [S, hidden_size] bf16, condition_proj output
    w: Dict[String, ArcPointer[Tensor]],
    config: MiniMaxH3DiTConfig,
    ctx: DeviceContext,
) raises -> Tensor:
    """`token_refiner.blocks.{0,1}` + `token_refiner.final_norm`. S/H/Dh are
    comptime — the whole port follows this convention for any sdpa-bearing
    block (see e.g. models/dit/hunyuan15_dit.mojo)."""
    if text_embeds.shape()[0] != S:
        raise Error("minimax_h3_token_refiner: text row count != comptime S")
    var state = text_embeds.clone(ctx)
    for layer in range(config.token_refiner_num_layers):
        var prefix = String("token_refiner.blocks.") + String(layer)
        state = _minimax_h3_token_refiner_block[S, H, Dh](state, w, prefix, config, ctx)
    return vec_rms_norm(state, w["token_refiner.final_norm.weight"][], config.final_norm_eps, ctx)


# ─────────────────────────────────────────────────────────────────────────────
# Scatter the three projected/refined streams into one packed sequence buffer
# (block_forward.mojo minimax_h3_forward step 3, lines 509-521).
# ─────────────────────────────────────────────────────────────────────────────
def minimax_h3_scatter_streams(
    video_embeds: Tensor,  # [Nv, hidden] bf16
    audio_embeds: Tensor,  # [Na, hidden] bf16
    text_embeds: Tensor,  # [Nt, hidden] bf16 (already refined)
    video_indices: List[Int],
    audio_indices: List[Int],
    text_indices: List[Int],
    sequence_length: Int,
    hidden_size: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """[S, hidden] with each row filled from exactly one of the three streams.
    Reuses index_select_backward (scatter-ADD) as a plain scatter: the three
    index sets are assumed DISJOINT and to cover [0, S) exactly once, so
    summing the three zero-elsewhere scatters is a disjoint union, not an
    accumulation."""
    var in_shape = List[Int]()
    in_shape.append(sequence_length)
    in_shape.append(hidden_size)
    var t = index_select_backward(text_embeds, text_indices, 0, in_shape, ctx)
    var v = index_select_backward(video_embeds, video_indices, 0, in_shape, ctx)
    var a = index_select_backward(audio_embeds, audio_indices, 0, in_shape, ctx)
    return add(add(t, v, ctx), a, ctx)


# ─────────────────────────────────────────────────────────────────────────────
# Composed pre-stack entry point: everything up to (not including) the 50
# denoiser blocks. Hands off `hidden` [S, hidden] bf16 ready for the stack,
# and `temb` [num_timesteps, time_embed_dim] F32 for whoever builds the
# per-block / final AdaLN caches.
# ─────────────────────────────────────────────────────────────────────────────
struct MiniMaxH3FrontendEmbed(Movable):
    var hidden: Tensor
    var temb: Tensor

    def __init__(out self, var hidden: Tensor, var temb: Tensor):
        self.hidden = hidden^
        self.temb = temb^


def minimax_h3_frontend_embed[TR_S: Int, TR_H: Int, TR_DH: Int](
    video_rows: Tensor,  # [Nv, 96] F32, already-patchified
    audio_rows: Tensor,  # [Na, 32] F32, already-packed
    text_rows: Tensor,  # [Nt=TR_S, 5120] bf16
    timesteps: Tensor,  # [num_timesteps] F32
    video_indices: List[Int],
    audio_indices: List[Int],
    text_indices: List[Int],
    sequence_length: Int,
    w: Dict[String, ArcPointer[Tensor]],
    config: MiniMaxH3DiTConfig,
    ctx: DeviceContext,
) raises -> MiniMaxH3FrontendEmbed:
    # video/audio patch-proj is fp32; the packed sequence the block stack
    # consumes is bf16-only (every blocks.* tensor is bf16) — cast down at
    # this boundary. See file header, "IMPLIED CAST BOUNDARY".
    var video_embeds_f32 = minimax_h3_video_patch_embed(video_rows, w, ctx)
    # torch_f32_to_bf16_rne, NOT cast_tensor. This F32->BF16 boundary runs ONCE
    # PER DENOISING STEP, and Mojo's native cast differs from PyTorch's
    # round-to-nearest-even by ~1 bf16 quantum on some values. One cast is
    # invisible (cos ~1.0); inside an N-step loop the per-element bias compounds
    # and decorrelates from torch while every single-forward gate still reads
    # cos 0.9998 and the global std still matches. That is the NAVA failure —
    # a visibly distorted decode with every component green.
    var video_embeds = torch_f32_to_bf16_rne(video_embeds_f32, ctx)
    var audio_embeds_f32 = minimax_h3_audio_patch_embed(audio_rows, w, ctx)
    var audio_embeds = torch_f32_to_bf16_rne(audio_embeds_f32, ctx)

    var text_embeds0 = minimax_h3_condition_embed(text_rows, w, ctx)
    var text_embeds = minimax_h3_token_refiner[TR_S, TR_H, TR_DH](text_embeds0, w, config, ctx)

    var hidden = minimax_h3_scatter_streams(
        video_embeds,
        audio_embeds,
        text_embeds,
        video_indices,
        audio_indices,
        text_indices,
        sequence_length,
        config.hidden_size,
        ctx,
    )
    var temb = minimax_h3_timestep_embedding(timesteps, w, config, ctx)
    return MiniMaxH3FrontendEmbed(hidden^, temb^)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Final layer: RMSNorm modulated per-row by timestep, then the two heads.
#    (block_forward.mojo minimax_h3_forward step 7, lines 570-626.)
# ─────────────────────────────────────────────────────────────────────────────
struct MiniMaxH3FrontendOutput(Movable):
    """Velocities for the rows addressed by video_indices/audio_indices, in
    that order — matches block_forward.mojo's MiniMaxH3ForwardOutput exactly.
    Patch-token level, NOT unpatchified back to a latent grid (that inverse
    fold is models/minimax_h3/rearrange.mojo's job, a separate unit)."""

    var video_out: Tensor  # [Nv, video_patch_dim=96] F32
    var audio_out: Tensor  # [Na, audio_latents_dim=32] F32

    def __init__(out self, var video_out: Tensor, var audio_out: Tensor):
        self.video_out = video_out^
        self.audio_out = audio_out^


def minimax_h3_final_layer(
    hidden: Tensor,  # [S, hidden] bf16 — the 50-block stack's output
    final_modulation: Tensor,  # [num_timesteps, 2*hidden] F32 — PRECOMPUTED (see header)
    timestep_indices: List[Int],  # length S: row -> its own timestep row
    video_indices: List[Int],
    audio_indices: List[Int],
    w: Dict[String, ArcPointer[Tensor]],
    config: MiniMaxH3DiTConfig,
    ctx: DeviceContext,
) raises -> MiniMaxH3FrontendOutput:
    """`final_modulation` is NOT computed here — see file header "FINAL
    ADALN". This function only INDEXES it, per ROW, by that row's own
    timestep (`timestep_indices`) — NOT the (timestep, modality) index the
    main stack's AdaLN uses."""
    var hidden_size = config.hidden_size
    var sequence_length = hidden.shape()[0]
    if len(timestep_indices) != sequence_length:
        raise Error(
            "minimax_h3_final_layer: timestep_indices length != sequence_length"
        )

    var final_normed = vec_rms_norm(hidden, w["final_layer.norm.weight"][], config.final_norm_eps, ctx)
    # final_layer.norm is bf16 (not a dtype-trap prefix) so final_normed is
    # bf16; cast to F32 to match the fp32 modulation table and the fp32
    # output heads.
    var final_normed_f32 = cast_tensor(final_normed, STDtype.F32, ctx)

    var row_mod = gather_rows(final_modulation, timestep_indices, ctx)  # [S, 2*hidden]
    var shift = slice(row_mod, 1, 0, hidden_size, ctx)
    var scale = slice(row_mod, 1, hidden_size, hidden_size, ctx)
    var scale_p1 = add_scalar(scale, Float32(1.0), ctx)
    var modulated = add(mul(final_normed_f32, scale_p1, ctx), shift, ctx)

    var video_all = _lin(modulated, w, "final_layer.video_out.weight", "final_layer.video_out.bias", ctx)  # [S,96] F32
    var audio_all = _lin(modulated, w, "final_layer.audio_out.weight", "final_layer.audio_out.bias", ctx)  # [S,32] F32

    var video_out = gather_rows(video_all, video_indices, ctx)
    var audio_out = gather_rows(audio_all, audio_indices, ctx)
    return MiniMaxH3FrontendOutput(video_out^, audio_out^)
