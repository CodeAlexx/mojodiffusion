# serenitymojo/models/dit/minimax_h3_dit.mojo
#
# MiniMax-H3 Omni-Transformer — DEVICE path (chunk 1: the block forward).
#
# This is the runtime. `models/minimax_h3/block_forward.mojo` is the host-float32
# ORACLE this is gated against — it says so in its own header ("the readable
# oracle for the math, not the runtime"). Nothing here may call it.
#
# CONFIG IS THE VENDOR'S OWN, not inferred: FL2VA/transformer/config.json in the
# released repo (MiniMaxAI/MiniMax-H3, verified 2026-08-03):
#
#     hidden_size 5376        num_layers 50        token_refiner_num_layers 2
#     num_attention_heads 56  attention_head_dim 128   -> inner 7168
#     ffn_hidden_size 14336   time_embed_dim 2688
#     adaln_out_features 96768   final_adaln_out_features 10752
#     rope_inv_freq_len 16    norm_eps / qk_norm_eps / final_norm_eps 1e-5
#
# NOTE inner_dim (7168) != hidden_size (5376). Every projection in the block
# crosses that boundary and a port that assumes they are equal builds cleanly
# and produces garbage: qkv is [3*7168, 5376] and out_proj is [5376, 7168].
#
# ─────────────────────────────────────────────────────────────────────────────
# BLOCK MATH  (one of 50; mirrors MiniMaxH3TransformerBlock.forward)
#
#   n1   = RMSNorm(x, norm1.weight, eps)
#   n1   = n1 * (1 + scale_msa[row]) + shift_msa[row]
#   qkv  = n1 @ qkv_proj.T                       -> q,k,v each [S, 7168]
#   q    = RMSNorm_perhead(q, q_norm.weight)     eps = qk_norm_eps
#   k    = RMSNorm_perhead(k, k_norm.weight)
#   q,k  = partial_rope(q, k, cos, sin)          only the first rotary_dim lanes
#   a    = full_self_attention(q, k, v)          NO mask, NO cross-attention
#   x    = x + gate_msa[row] * (a @ out_proj.T)
#   n2   = RMSNorm(x, norm2.weight, eps)
#   n2   = n2 * (1 + scale_mlp[row]) + shift_mlp[row]
#   g,u  = chunk2(n2 @ fc1.T)                    fc1 stores [gate; value]
#   x    = x + gate_mlp[row] * (silu(g)*u @ fc2.T)
#
# THE ADALN ROW INDEX is the part that is easy to get wrong and impossible to
# see: modulation is addressed by `timestep_index * 3 + token_tag`, so the SAME
# block applies DIFFERENT shift/scale/gate to video (tag 0), text (tag 1) and
# audio (tag 2) rows of ONE packed sequence. Indexing by timestep alone, or by
# batch as every other DiT in this repo does, still runs and still denoises —
# into mush.
#
# ─────────────────────────────────────────────────────────────────────────────
# DTYPE. Storage bf16, accumulation f32, per the house rule and the checkpoint:
# every block tensor is bf16 (the 12 fp32 tensors are all outside the stack —
# patch projections, time embedder, output heads). Attention scores, norm math
# and the modulation table are f32. There is no host-f32 boundary here; the
# oracle's `List[Float32]` surface stops at `models/minimax_h3/`.
#
# ADALN IS NOT LOADED HERE. `adaln_proj.linear` is [96768, 2688] per block and
# 13.04 B parameters over the stack — 39% of the model. Its input is the
# timestep embedding alone, so the vendor's own README says the modulation
# "can be precomputed and cached, these parameters do not need to be loaded for
# inference-only deployment". This block takes `mod` already computed:
# a [rows, 6*hidden] f32 table, rows = distinct_timesteps * 3. See
# `models/minimax_h3/fp8_policy.mojo` for the residency arithmetic.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.collections import Dict, List, Optional
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.ops.vec_rms_norm import vec_rms_norm
from serenitymojo.ops.norm import rms_norm_modulate_bf16
from serenitymojo.ops.activations import swiglu_packed_value_gate
from serenitymojo.ops.linear import linear
from serenitymojo.ops.int8_quant import int8_dequant_groupwise_to_bf16
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.rope import rope_halfsplit_full_head_broadcast
from serenitymojo.ops.attention_flash import (
    sdpa_flash_infer_fwd,
    sdpa_flash_infer_fwd_dynamic,
    sdpa_flash_infer_fwd_cross_dynamic,
)
from serenitymojo.ops.sage_attention_int8 import (
    SageInt8Scratch,
    SageUltraViCoConfig,
    sage_attention_int8_fwd,
    sage_attention_int8_fwd_dynamic,
    sage_attention_int8_fwd_scratch,
)
from serenitymojo.models.dit.minimax_h3_int8_linear import (
    MiniMaxH3Int8QKV,
    minimax_h3_bf16_linear_chunked,
    minimax_h3_bf16_mlp_residual_inplace,
    minimax_h3_bf16_qkv_linear,
    minimax_h3_groupwise_mlp_residual_inplace,
    minimax_h3_groupwise_qkv_linear,
    minimax_h3_int8_linear,
    minimax_h3_int8_mlp_residual_inplace,
    minimax_h3_int8_qkv_linear,
    minimax_h3_int8_residual_linear,
    minimax_h3_int8_residual_linear_inplace,
    minimax_h3_int8_swiglu_linear,
)
from serenitymojo.models.dit.minimax_h3_qk_inplace import (
    minimax_h3_qk_norm_partial_rope_inplace,
)
from serenitymojo.models.minimax_h3.h3_lora_overlay import H3LoraOverlay
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current
from serenitymojo.ops.tensor_algebra import (
    gather_rows,
    slice,
    reshape,
    reshape_in_place,
    reshape_owned,
    concat,
    add,
    full_device,
)


comptime MINIMAX_H3_ATTN_CUDNN = 0
comptime MINIMAX_H3_ATTN_SAGE_INT8 = 1
comptime MINIMAX_H3_ATTN_SAGE_EXACT_PREFIX_BASE = 100000


def minimax_h3_sage_exact_prefix_backend(prefix_rows: Int) raises -> Int:
    if prefix_rows <= 0 or prefix_rows >= MINIMAX_H3_ATTN_SAGE_EXACT_PREFIX_BASE:
        raise Error("MiniMax-H3 exact Sage prefix must be in [1,100000)")
    return MINIMAX_H3_ATTN_SAGE_EXACT_PREFIX_BASE + prefix_rows


def _minimax_h3_attention_dispatch(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Float32,
    ctx: DeviceContext,
    attention_backend: Int,
    sage_scratch: Optional[SageInt8Scratch] = None,
    sage_ultravico: Optional[SageUltraViCoConfig] = None,
) raises -> Tensor:
    """Exact cuDNN, raw Sage, or Sage with an exact query prefix.

    When `sage_scratch` is provided, Sage runs with zero device allocations
    through the run-lifetime scratch; without it, every call allocates and
    frees its own buffers (gate/probe surface — near the 24 GiB envelope
    that churn intermittently OOMs mid-denoise, so product runners must
    pass scratch).
    """
    if attention_backend == MINIMAX_H3_ATTN_CUDNN:
        return sdpa_flash_infer_fwd_dynamic(q, k, v, scale, ctx)
    if attention_backend == MINIMAX_H3_ATTN_SAGE_INT8:
        if sage_ultravico:
            var uv = sage_ultravico.value()
            if sage_scratch:
                return sage_attention_int8_fwd_scratch(
                    q, k, v, scale, sage_scratch.value(), ctx,
                    uv.video_start, uv.rows_per_frame, uv.window_frames,
                    uv.repeat_period_frames, uv.repeat_radius_frames,
                    uv.alpha, uv.beta,
                )
            return sage_attention_int8_fwd_dynamic(
                q, k, v, scale, ctx,
                uv.video_start, uv.rows_per_frame, uv.window_frames,
                uv.repeat_period_frames, uv.repeat_radius_frames,
                uv.alpha, uv.beta,
            )
        if sage_scratch:
            return sage_attention_int8_fwd_scratch(
                q, k, v, scale, sage_scratch.value(), ctx
            )
        return sage_attention_int8_fwd_dynamic(q, k, v, scale, ctx)
    if attention_backend >= MINIMAX_H3_ATTN_SAGE_EXACT_PREFIX_BASE:
        var prefix_rows = (
            attention_backend - MINIMAX_H3_ATTN_SAGE_EXACT_PREFIX_BASE
        )
        var shape = q.shape()
        var sequence_length = shape[1]
        if prefix_rows >= sequence_length:
            raise Error("MiniMax-H3 exact Sage prefix must be smaller than S")
        # Compute the dominant video-query slab once with Sage, then replace
        # the small text+audio prefix with exact cross-SDPA against the same
        # complete K/V document.  T2VA packs rows as text|audio|video.
        var approximate: Tensor
        if sage_ultravico:
            var uv = sage_ultravico.value()
            if sage_scratch:
                approximate = sage_attention_int8_fwd_scratch(
                    q, k, v, scale, sage_scratch.value(), ctx,
                    uv.video_start, uv.rows_per_frame, uv.window_frames,
                    uv.repeat_period_frames, uv.repeat_radius_frames,
                    uv.alpha, uv.beta,
                )
            else:
                approximate = sage_attention_int8_fwd_dynamic(
                    q, k, v, scale, ctx,
                    uv.video_start, uv.rows_per_frame, uv.window_frames,
                    uv.repeat_period_frames, uv.repeat_radius_frames,
                    uv.alpha, uv.beta,
                )
        elif sage_scratch:
            approximate = sage_attention_int8_fwd_scratch(
                q, k, v, scale, sage_scratch.value(), ctx
            )
        else:
            approximate = sage_attention_int8_fwd_dynamic(
                q, k, v, scale, ctx
            )
        var q_prefix = slice(q, 1, 0, prefix_rows, ctx)
        var exact_prefix = sdpa_flash_infer_fwd_cross_dynamic(
            q_prefix, k, v, scale, ctx
        )
        var video_tail = slice(
            approximate, 1, prefix_rows, sequence_length - prefix_rows, ctx
        )
        return concat(1, ctx, exact_prefix, video_tail)
    raise Error(
        String("minimax_h3_block_forward: unknown attention backend ")
        + String(attention_backend)
    )




@fieldwise_init
struct MiniMaxH3DiTConfig(Copyable, Movable, ImplicitlyCopyable):
    """Straight from the released `transformer/config.json`. No defaults are
    invented here: a field that is not in that file does not belong."""

    var hidden_size: Int
    var num_layers: Int
    var token_refiner_num_layers: Int
    var num_attention_heads: Int
    var attention_head_dim: Int
    var ffn_hidden_size: Int
    var latents_dim: Int
    var audio_latents_dim: Int
    var text_dim: Int
    var timestep_input_dim: Int
    var time_embed_dim: Int
    var adaln_out_features: Int
    var final_adaln_out_features: Int
    var rope_inv_freq_len: Int
    var norm_eps: Float32
    var qk_norm_eps: Float32
    var final_norm_eps: Float32

    def inner_dim(self) -> Int:
        """`heads * head_dim` = 7168, which is NOT `hidden_size` (5376).

        Kept as a method rather than a field so it cannot be set to something
        inconsistent with the head count it is derived from."""
        return self.num_attention_heads * self.attention_head_dim

    def video_patch_dim(self) -> Int:
        """Patch is (1, 2, 2), so 4 latent voxels per token."""
        return self.latents_dim * 4

    def adaln_rows_per_timestep(self) -> Int:
        """Three modality tags: video 0, text 1, audio 2."""
        return 3

    def validate(self) raises:
        """Fail loudly, before `DeviceContext()`, on a config that cannot be
        the released model. The runtime-port skill calls for preflight before
        CUDA; a wrong hidden size otherwise surfaces as an opaque shape error
        forty layers deep."""
        if self.inner_dim() * 3 % 1 != 0:
            raise Error("unreachable")
        if self.adaln_out_features != 6 * self.hidden_size * 3:
            raise Error(
                String("MiniMax-H3 config: adaln_out_features ")
                + String(self.adaln_out_features)
                + " != 6 * hidden_size * 3 = "
                + String(6 * self.hidden_size * 3)
            )
        if self.final_adaln_out_features != 2 * self.hidden_size:
            raise Error(
                String("MiniMax-H3 config: final_adaln_out_features ")
                + String(self.final_adaln_out_features)
                + " != 2 * hidden_size = "
                + String(2 * self.hidden_size)
            )
        if self.hidden_size <= 0 or self.num_layers <= 0:
            raise Error("MiniMax-H3 config: non-positive hidden_size/num_layers")


def minimax_h3_released_config() -> MiniMaxH3DiTConfig:
    """The released FL2VA/Ref2VA transformer config, verbatim.

    Both partitions ship byte-identical `transformer/config.json`, so one
    constant serves t2va, fl2va and ref2va."""
    return MiniMaxH3DiTConfig(
        5376,       # hidden_size
        50,         # num_layers
        2,          # token_refiner_num_layers
        56,         # num_attention_heads
        128,        # attention_head_dim
        14336,      # ffn_hidden_size
        24,         # latents_dim
        32,         # audio_latents_dim
        5120,       # text_dim
        256,        # timestep_input_dim
        2688,       # time_embed_dim
        96768,      # adaln_out_features
        10752,      # final_adaln_out_features
        16,         # rope_inv_freq_len
        Float32(1.0e-5),
        Float32(1.0e-5),
        Float32(1.0e-5),
    )


# ─────────────────────────────────────────────────────────────────────────────
# AdaLN row addressing.
#
# Its own function, and gated, because it is the single index in this model that
# is both load-bearing and invisible when wrong. `timestep_index` selects which
# noise level a row sits at (H3 runs video and audio on SEPARATE schedules, so
# one forward carries several), and `token_tag` selects the modality. The table
# is laid out [t0_video, t0_text, t0_audio, t1_video, ...].
#
# A row whose tag is negative is PADDING and must never index the table.
# ─────────────────────────────────────────────────────────────────────────────
def minimax_h3_adaln_row(timestep_index: Int, token_tag: Int) raises -> Int:
    if token_tag < 0:
        raise Error(
            "MiniMax-H3 adaLN: padding row (tag < 0) has no modulation; the"
            " caller must mask it out rather than index the table"
        )
    if token_tag > 2:
        raise Error(
            String("MiniMax-H3 adaLN: token_tag ") + String(token_tag)
            + " is not a modality (0 video, 1 text, 2 audio)"
        )
    if timestep_index < 0:
        raise Error("MiniMax-H3 adaLN: negative timestep index")
    return timestep_index * 3 + token_tag


def minimax_h3_adaln_rows(
    timestep_indices: List[Int], token_tags: List[Int]
) raises -> List[Int]:
    """Per-row modulation index for a whole packed sequence.

    Padding rows (tag < 0) are given row 0 and MUST be masked downstream — the
    value is never read, but a negative index would fault the gather."""
    if len(timestep_indices) != len(token_tags):
        raise Error("MiniMax-H3 adaLN: timestep_indices/token_tags length mismatch")
    var out = List[Int](capacity=len(token_tags))
    for i in range(len(token_tags)):
        if token_tags[i] < 0:
            out.append(0)
        else:
            out.append(minimax_h3_adaln_row(timestep_indices[i], token_tags[i]))
    return out^


# ─────────────────────────────────────────────────────────────────────────────
# Weight naming.
#
# The ORIGINAL checkpoint namespace, verified against the released
# `transformer/model.safetensors.index.json`: 535 tensors, 0 missing, 0 extra
# against the plan this port was built on. These builders exist so a rename
# shows up as one edit here rather than as a missing-key error mid-forward.
# ─────────────────────────────────────────────────────────────────────────────
def minimax_h3_block_prefix(layer: Int) -> String:
    return String("blocks.") + String(layer) + "."


def minimax_h3_refiner_prefix(layer: Int) -> String:
    return String("token_refiner.blocks.") + String(layer) + "."


def minimax_h3_block_tensor_names(layer: Int) raises -> List[String]:
    """Every tensor one denoiser block needs, EXCLUDING adaln_proj.

    adaln_proj is deliberately absent: it is consumed once into the modulation
    cache before sampling and never loaded again (see this file's header). A
    streaming loader that fetches it per block would move 24.2 GiB per step for
    nothing."""
    var p = minimax_h3_block_prefix(layer)
    var names = List[String]()
    names.append(p + "norm1.weight")
    names.append(p + "norm2.weight")
    names.append(p + "attn.qkv_proj.weight")
    names.append(p + "attn.q_norm.weight")
    names.append(p + "attn.k_norm.weight")
    names.append(p + "attn.out_proj.weight")
    names.append(p + "mlp.fc1.weight")
    names.append(p + "mlp.fc2.weight")
    return names^


def minimax_h3_expected_shape(
    name: String, config: MiniMaxH3DiTConfig
) raises -> List[Int]:
    """The shape a block tensor must have, so the loader can assert instead of
    discovering it through a wrong-looking image.

    The two fused ones are the reason this exists:
      qkv_proj  [3 * inner, hidden]   per-head INTERLEAVED q/k/v in the raw file
      fc1       [2 * ffn,   hidden]   stored [gate; value]
    Both are rewritten by `models/minimax_h3/loader.mojo`, which is gated at
    max_abs 0.0 against torch's own tensors."""
    var hidden = config.hidden_size
    var inner = config.inner_dim()
    var ffn = config.ffn_hidden_size
    var out = List[Int]()
    if name.endswith("attn.qkv_proj.weight"):
        out.append(3 * inner)
        out.append(hidden)
    elif name.endswith("attn.out_proj.weight"):
        out.append(hidden)
        out.append(inner)
    elif name.endswith("mlp.fc1.weight"):
        out.append(2 * ffn)
        out.append(hidden)
    elif name.endswith("mlp.fc2.weight"):
        out.append(hidden)
        out.append(ffn)
    elif name.endswith("attn.q_norm.weight") or name.endswith("attn.k_norm.weight"):
        out.append(config.attention_head_dim)
    elif name.endswith("norm1.weight") or name.endswith("norm2.weight"):
        out.append(hidden)
    else:
        raise Error(String("MiniMax-H3: no expected shape for ") + name)
    return out^


def minimax_h3_check_block_weights(
    ref weights: Dict[String, ArcPointer[Tensor]],
    layer: Int,
    config: MiniMaxH3DiTConfig,
) raises:
    """Preflight one block: every tensor present, right shape, bf16 storage.

    Called before `DeviceContext()` work per the runtime-port rule that
    unsupported dtype/shape must fail before CUDA. bf16 is asserted rather than
    coerced — silently upcasting storage to f32 is the bug that rule exists to
    prevent, and at 61.7 GiB it is also an instant OOM."""
    var names = minimax_h3_block_tensor_names(layer)
    for i in range(len(names)):
        ref n = names[i]
        if n not in weights:
            raise Error(String("MiniMax-H3: missing block tensor ") + n)
        ref t = weights[n][]
        var want = minimax_h3_expected_shape(n, config)
        var got = t.shape()
        if len(got) != len(want):
            raise Error(
                String("MiniMax-H3: rank mismatch for ") + n + ", got "
                + String(len(got)) + " dims, want " + String(len(want))
            )
        for d in range(len(want)):
            if got[d] != want[d]:
                raise Error(
                    String("MiniMax-H3: shape mismatch for ") + n + " dim "
                    + String(d) + ": got " + String(got[d]) + ", want "
                    + String(want[d])
                )
        if t.dtype() != STDtype.BF16:
            raise Error(
                String("MiniMax-H3: ") + n + " is not BF16. The stack is bf16"
                " storage by contract; upcasting it here would both break the"
                " dtype rule and blow the 24 GiB budget."
            )


# ─────────────────────────────────────────────────────────────────────────────
# DEVICE block forward.
#
# One of 50 `MiniMaxH3TransformerBlock`s, GPU-resident, mirroring
# `models/minimax_h3/block_forward.mojo::_transformer_block` (the host-float32
# ORACLE this is gated against — read that file's header for the diffusers
# line citations; this file must never call it).
#
# WEIGHT CONTRACT — this expects PRE-TRANSFORMED tensors from
# `models/dit/minimax_h3_loader_device.mojo::minimax_h3_load_block_device`,
# NOT the raw checkpoint's row order:
#   * `attn.qkv_proj.weight` already de-interleaved into contiguous
#     `[q_all; k_all; v_all]` thirds (`minimax_h3_deinterleave_qkv_bf16`).
#   * `mlp.fc1.weight` already swapped from the checkpoint's `[gate; value]`
#     to `[value; gate]` (`minimax_h3_swap_fc1_bf16`) — matching the oracle's
#     own `swiglu_ff` convention (block_forward.mojo: "the FIRST half of its
#     output is the VALUE and the second is the GATE").
# The transform is done ONCE per block-load, not once per forward call: the
# earlier version of this file did the qkv de-interleave here via a
# `gather_rows` permutation on every invocation (50 blocks x every denoising
# step), which is both redundant with, and — if both ran — a SILENT DOUBLE
# PERMUTATION on top of, the loader's own already-gated rewrite. Do not
# re-add a transform here; fix it in the loader if it's wrong.
#
# `minimax_h3_check_block_weights` (above) only checks SHAPE, not row order —
# it accepts EITHER convention silently, since de-interleaving/swapping a
# weight's rows never changes its shape. That means passing preflight is NOT
# proof this contract is satisfied; only weights that came from
# `minimax_h3_load_block_device` (or an equivalent pre-transform) are correct
# here. A caller that hands this RAW `ShardedSafeTensors`/`BlockLoader`
# tensors directly will build, run, and produce a plausible-looking wrong
# answer — the same invisible-corruption class the adaLN row index is.
#
# out_proj and fc2 need no transform in either convention: `models/minimax_h3
# /loader.mojo`'s header lists exactly three transforms (qkv, fc1,
# rope.inv_freq) and out_proj/fc2 are not among them.
#
# GUARD: `minimax_h3_check_block_weights` (above) cannot detect a caller
# passing RAW weights instead of `minimax_h3_load_block_device` output —
# de-interleaving/swapping rows never changes shape, so shape-only
# validation accepts either convention silently, and this exact bug
# (double-permuted qkv, reversed fc1 halves) nearly shipped twice. The two
# marker keys below close that hole: `minimax_h3_load_block_device` stamps
# both into the Dict it returns (one tiny presence-only tensor each, no data
# scan), and `minimax_h3_require_transformed_weights` — called unconditionally
# as the FIRST statement of `minimax_h3_block_forward`, so it cannot be
# skipped by a caller who doesn't separately call a preflight — raises a
# named error naming the likely cause if either is absent.
# ─────────────────────────────────────────────────────────────────────────────
comptime MINIMAX_H3_HEADS = 56
comptime MINIMAX_H3_HEAD_DIM = 128

comptime MINIMAX_H3_QKV_DEINTERLEAVED_MARKER = "__h3_qkv_deinterleaved__"
comptime MINIMAX_H3_FC1_SWAPPED_MARKER = "__h3_fc1_swapped__"


def minimax_h3_require_transformed_weights(
    weights: Dict[String, ArcPointer[Tensor]], layer: Int
) raises:
    """Fail loudly if `weights` looks like it came straight from
    `ShardedSafeTensors`/`BlockLoader` instead of `minimax_h3_load_block_
    device`. One dict `in` lookup per marker — O(1), no data scan — so this
    is cheap enough to run unconditionally on every block-forward call, not
    just when a caller remembers to preflight."""
    if (
        MINIMAX_H3_QKV_DEINTERLEAVED_MARKER not in weights
        or MINIMAX_H3_FC1_SWAPPED_MARKER not in weights
    ):
        raise Error(
            String("MiniMax-H3: layer ") + String(layer) + " weights are"
            " missing the qkv-deinterleave/fc1-swap marker ("
            + MINIMAX_H3_QKV_DEINTERLEAVED_MARKER + "/"
            + MINIMAX_H3_FC1_SWAPPED_MARKER + "). This means they were NOT"
            " built by minimax_h3_load_block_device — passing raw"
            " ShardedSafeTensors/BlockLoader tensors here silently scrambles"
            " q/k/v and reverses the SwiGLU gate/value roles (no shape"
            " error, no NaN; see minimax_h3_block_forward's WEIGHT CONTRACT"
            " note above). Call minimax_h3_load_block_device to load this"
            " block instead of building the weight Dict directly."
        )


def _minimax_h3_expand_rope_per_head(
    tbl: Tensor, s: Int, heads: Int, rotary_dim: Int, ctx: DeviceContext
) raises -> Tensor:
    """`[S, rotary_dim]` -> `[S*heads, rotary_dim]`, broadcasting one row to
    every head (rope is shared across heads: `block_forward.mojo::
    _apply_rope_inplace`, "cos/sin are shared across heads"). Device-side
    zero-broadcast add, mirroring `models/dit/wan22_dit.mojo::
    _expand_rope_per_head` — avoids a host-built heads-fold duplication."""
    var t3_shape: List[Int] = [s, 1, rotary_dim]
    var t3 = reshape(tbl, t3_shape^, ctx)
    var zshape: List[Int] = [s, heads, rotary_dim]
    var zeros = full_device(zshape^, Float32(0.0), tbl.dtype(), ctx)
    var bc = add(t3, zeros, ctx)
    var out_shape: List[Int] = [s * heads, rotary_dim]
    return reshape(bc, out_shape^, ctx)


def _minimax_h3_apply_partial_rope(
    x: Tensor, cos: Tensor, sin: Tensor, heads: Int,
    rotary_dim: Int, ctx: DeviceContext
) raises -> Tensor:
    """Rotate only the first `rotary_dim` channels of a `[1,S,heads,head_dim]`
    head tensor, pass the rest through untouched — the PARTIAL rope this
    model uses (rotary_dim 96 < head_dim 128 at the released geometry).
    `rope_halfsplit_full` (ops/rope.mojo) already implements the exact
    rotate-half math `_apply_rope_inplace` uses over a full-width duplicated
    cos/sin table (block0==block1 in `MiniMaxH3RopeTable`, so the general
    "two independent halves" kernel degenerates to the oracle's identical
    formula); this only slices the rotated prefix out, applies it, and
    concatenates the untouched tail back on."""
    var sh = x.shape()
    var head_dim = sh[len(sh) - 1]
    var x_rot = slice(x, 3, 0, rotary_dim, ctx)
    var x_pass = slice(x, 3, rotary_dim, head_dim - rotary_dim, ctx)
    var rotated = rope_halfsplit_full_head_broadcast(
        x_rot, cos, sin, heads, ctx
    )
    return concat(3, ctx, rotated, x_pass)


def _minimax_h3_block_linear(
    x: Tensor,
    weights: Dict[String, ArcPointer[Tensor]],
    name: String,
    ctx: DeviceContext,
) raises -> Tensor:
    """Dispatch a block projection from its explicit stored dtype.

    Ordinary streamed weights are BF16. Quantized resident stores supply an
    I8 tensor at the canonical key plus a sibling scale tensor: F32 per-row
    scales select direct W8A8, while F16 groupwise scales are dequantized only
    for the projection currently executing. This avoids keeping all four BF16
    block matrices live beside long-sequence activations.
    """
    ref weight = weights[name][]
    if weight.dtype() == STDtype.BF16:
        var k = weight.shape()[1]
        var rows = x.numel() // k
        if rows > 16384:
            return minimax_h3_bf16_linear_chunked(x, weight, ctx)
        return linear(x, weight, None, ctx)
    if weight.dtype() == STDtype.I8:
        var scale_name = name + String(".scale")
        if scale_name not in weights:
            raise Error(
                String("MiniMax-H3 INT8 block weight is missing scale: ")
                + scale_name
            )
        ref weight_scale = weights[scale_name][]
        if weight_scale.dtype() == STDtype.F32:
            return minimax_h3_int8_linear(x, weight, weight_scale, ctx)
        if weight_scale.dtype() == STDtype.F16:
            var wshape = weight.shape()
            var sshape = weight_scale.shape()
            if len(wshape) != 2 or len(sshape) != 2 \
                    or sshape[0] != wshape[0] or sshape[1] <= 0 \
                    or wshape[1] % sshape[1] != 0:
                raise Error(
                    String("MiniMax-H3 groupwise INT8 scale shape mismatch: ")
                    + name
                )
            var group_size = wshape[1] // sshape[1]
            var dequant = int8_dequant_groupwise_to_bf16(
                weight, weight_scale, group_size, ctx
            )
            var rows = x.numel() // wshape[1]
            if rows > 16384:
                return minimax_h3_bf16_linear_chunked(x, dequant, ctx)
            return linear(x, dequant, None, ctx)
        raise Error(
            String("MiniMax-H3 INT8 block scale has unsupported dtype for ")
            + name + String(": ") + weight_scale.dtype().name()
        )
    raise Error(
        String("MiniMax-H3 block projection has unsupported dtype for ")
        + name + String(": ") + weight.dtype().name()
    )


def _minimax_h3_block_linear_lora(
    x: Tensor,
    weights: Dict[String, ArcPointer[Tensor]],
    name: String,
    layer: Int,
    slot: Int,
    lora_overlay: Optional[H3LoraOverlay],
    ctx: DeviceContext,
) raises -> Tensor:
    """Base projection plus the separately-resident LoRA activation branch.

    The base may be BF16, W8A8, or groupwise INT8. The adapter is deliberately
    applied after that base projection; folding it into the weight before INT8
    encoding destroys small trained deltas and does not match training.
    """
    var base = _minimax_h3_block_linear(x, weights, name, ctx)
    if lora_overlay and lora_overlay.value().has(layer, slot):
        var delta = lora_overlay.value().activation_delta(
            layer, slot, x, ctx
        )
        return add(base, delta, ctx)
    return base^


def _minimax_h3_block_swiglu_linear_lora(
    x: Tensor,
    weights: Dict[String, ArcPointer[Tensor]],
    name: String,
    layer: Int,
    lora_overlay: Optional[H3LoraOverlay],
    ctx: DeviceContext,
) raises -> Tensor:
    if lora_overlay and lora_overlay.value().has(layer, 2):
        var packed = _minimax_h3_block_linear_lora(
            x, weights, name, layer, 2, lora_overlay, ctx
        )
        return swiglu_packed_value_gate(packed, ctx)
    return _minimax_h3_block_swiglu_linear(
        x, weights, name, False, ctx
    )


def _minimax_h3_block_residual_linear_lora(
    x: Tensor,
    residual: Tensor,
    gate: Tensor,
    weights: Dict[String, ArcPointer[Tensor]],
    name: String,
    layer: Int,
    slot: Int,
    lora_overlay: Optional[H3LoraOverlay],
    ctx: DeviceContext,
) raises -> Tensor:
    if lora_overlay and lora_overlay.value().has(layer, slot):
        var projected = _minimax_h3_block_linear_lora(
            x, weights, name, layer, slot, lora_overlay, ctx
        )
        return residual_gate(residual, gate, projected, ctx)
    return _minimax_h3_block_residual_linear(
        x, residual, gate, weights, name, False, ctx
    )


def _minimax_h3_block_swiglu_linear(
    x: Tensor,
    weights: Dict[String, ArcPointer[Tensor]],
    name: String,
    low_headroom: Bool,
    ctx: DeviceContext,
) raises -> Tensor:
    """FC1 plus SwiGLU, fusing the exact row-scale W8A8 path.

    Direct W8A8 preserves the accepted projection-BF16 and activation-BF16
    boundaries while avoiding the full `[S, 2*ffn]` allocation. Other weight
    formats retain the accepted projection followed by the accepted activation.
    """
    ref weight = weights[name][]
    if weight.dtype() == STDtype.I8:
        var scale_name = name + String(".scale")
        if scale_name in weights:
            ref weight_scale = weights[scale_name][]
            if weight_scale.dtype() == STDtype.F32:
                return minimax_h3_int8_swiglu_linear(
                    x, weight, weight_scale, ctx
                )
    var packed = _minimax_h3_block_linear(x, weights, name, ctx)
    return swiglu_packed_value_gate(packed, ctx)


def _minimax_h3_block_residual_linear(
    x: Tensor,
    residual: Tensor,
    gate: Tensor,
    weights: Dict[String, ArcPointer[Tensor]],
    name: String,
    low_headroom: Bool,
    ctx: DeviceContext,
) raises -> Tensor:
    """FC2 plus residual gate, fusing the exact row-scale W8A8 path."""
    ref weight = weights[name][]
    if weight.dtype() == STDtype.I8:
        var scale_name = name + String(".scale")
        if scale_name in weights:
            ref weight_scale = weights[scale_name][]
            if weight_scale.dtype() == STDtype.F32:
                return minimax_h3_int8_residual_linear(
                    x, weight, weight_scale, residual, gate, ctx
                )
    var projected = _minimax_h3_block_linear(x, weights, name, ctx)
    return residual_gate(residual, gate, projected, ctx)


def _minimax_h3_rms_norm_modulate(
    x: Tensor,
    weight: Tensor,
    scale: Tensor,
    shift: Tensor,
    eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """Fuse the production BF16 RMSNorm/AdaLN pair; preserve all fallbacks."""
    if x.dtype() == STDtype.BF16 and weight.dtype() == STDtype.BF16 \
            and scale.dtype() == STDtype.BF16 \
            and shift.dtype() == STDtype.BF16:
        return rms_norm_modulate_bf16(
            x, weight, scale, shift, eps, ctx
        )
    var normalized = vec_rms_norm(x, weight, eps, ctx)
    return modulate(normalized, scale, shift, ctx)


def _minimax_h3_gather_mod_chunk(
    mod: Tensor,
    adaln_indices: List[Int],
    chunk: Int,
    hidden: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Gather one `[hidden]` AdaLN chunk without first expanding all six.

    This is mathematically identical to slicing `gather_rows(mod, ids)`, but
    its largest output is `[S, hidden]` rather than `[S, 6 * hidden]`.
    """
    var table_chunk = slice(mod, 1, chunk * hidden, hidden, ctx)
    return gather_rows(table_chunk, adaln_indices, ctx)


struct _MiniMaxH3MlpPrepared(Movable):
    """Only the tensors that must survive from FC1 into fused FC2."""

    var act: Tensor
    var gate: Tensor

    def __init__(out self, var act: Tensor, var gate: Tensor):
        self.act = act^
        self.gate = gate^

    def __del__(deinit self):
        pass


def _minimax_h3_prepare_low_headroom_mlp(
    x1: Tensor,
    mod: Tensor,
    adaln_indices: List[Int],
    hidden: Int,
    weights: Dict[String, ArcPointer[Tensor]],
    prefix: String,
    norm_eps: Float32,
    ctx: DeviceContext,
) raises -> _MiniMaxH3MlpPrepared:
    """Prepare fused FC1 in a scope that dies before FC2 allocation.

    At long sequences each gathered shift/scale and each normalized input is
    close to a GiB. Returning only FC1's activation and the residual gate
    makes those preparation buffers unreachable before fused FC2 begins.
    """
    var shift = _minimax_h3_gather_mod_chunk(
        mod, adaln_indices, 3, hidden, ctx
    )
    var scale = _minimax_h3_gather_mod_chunk(
        mod, adaln_indices, 4, hidden, ctx
    )
    var gate = _minimax_h3_gather_mod_chunk(
        mod, adaln_indices, 5, hidden, ctx
    )
    var mlp_in = _minimax_h3_rms_norm_modulate(
        x1, weights[prefix + "norm2.weight"][], scale, shift,
        norm_eps, ctx,
    )
    ctx.synchronize()
    cu_mempool_trim_current(0)
    var act = _minimax_h3_block_swiglu_linear(
        mlp_in, weights, prefix + "mlp.fc1.weight", True, ctx
    )
    ctx.synchronize()
    cu_mempool_trim_current(0)
    return _MiniMaxH3MlpPrepared(act^, gate^)


def _minimax_h3_low_headroom_attention[
    Heads: Int, HeadDim: Int
](
    x: Tensor,
    weights: Dict[String, ArcPointer[Tensor]],
    prefix: String,
    mod: Tensor,
    adaln_indices: List[Int],
    hidden: Int,
    inner: Int,
    heads: Int,
    head_dim: Int,
    norm_eps: Float32,
    qk_norm_eps: Float32,
    cos: Tensor,
    sin: Tensor,
    rotary_dim: Int,
    ctx: DeviceContext,
    attention_backend: Int,
    sage_scratch: Optional[SageInt8Scratch] = None,
    sage_ultravico: Optional[SageUltraViCoConfig] = None,
) raises -> Tensor:
    """Long-sequence attention in a lifetime isolated from the MLP.

    Returning only `x + gate_msa * attention` guarantees the multi-GiB QKV,
    RoPE, SDPA, modulation, and projection temporaries are destroyed before
    FC1 begins.
    """
    var S = x.shape()[1]
    var shift = _minimax_h3_gather_mod_chunk(
        mod, adaln_indices, 0, hidden, ctx
    )
    var scale_msa = _minimax_h3_gather_mod_chunk(
        mod, adaln_indices, 1, hidden, ctx
    )
    var gate = _minimax_h3_gather_mod_chunk(
        mod, adaln_indices, 2, hidden, ctx
    )
    var attn_in = _minimax_h3_rms_norm_modulate(
        x, weights[prefix + "norm1.weight"][], scale_msa, shift,
        norm_eps, ctx,
    )
    ctx.synchronize()

    comptime if HeadDim != 128:
        if attention_backend != MINIMAX_H3_ATTN_CUDNN:
            raise Error(
                "minimax_h3_block_forward: sage-int8 requires head_dim=128"
            )
    var attn: Tensor
    var attn_scale = Float32(1.0) / (Float32(head_dim) ** Float32(0.5))
    var qkv_name = prefix + "attn.qkv_proj.weight"
    ref qkv_weight = weights[qkv_name][]
    if qkv_weight.dtype() == STDtype.BF16 or (
        qkv_weight.dtype() == STDtype.I8
        and (qkv_name + String(".scale")) in weights
    ):
        var split_qkv: MiniMaxH3Int8QKV
        if qkv_weight.dtype() == STDtype.BF16:
            split_qkv = minimax_h3_bf16_qkv_linear(
                attn_in, qkv_weight, ctx
            )
        elif weights[qkv_name + String(".scale")][].dtype() == STDtype.F32:
            ref qkv_scale = weights[qkv_name + String(".scale")][]
            split_qkv = minimax_h3_int8_qkv_linear(
                attn_in, qkv_weight, qkv_scale, ctx
            )
        elif weights[qkv_name + String(".scale")][].dtype() == STDtype.F16:
            ref qkv_scale = weights[qkv_name + String(".scale")][]
            split_qkv = minimax_h3_groupwise_qkv_linear(
                attn_in, qkv_weight, qkv_scale, ctx
            )
        else:
            raise Error("MiniMax-H3 low-headroom QKV scale dtype unsupported")
        reshape_in_place(split_qkv.q, [1, S, heads, head_dim])
        reshape_in_place(split_qkv.k, [1, S, heads, head_dim])
        reshape_in_place(split_qkv.v, [1, S, heads, head_dim])
        ctx.synchronize()
        minimax_h3_qk_norm_partial_rope_inplace(
            split_qkv.q, weights[prefix + "attn.q_norm.weight"][],
            cos, sin, heads, rotary_dim, qk_norm_eps, ctx,
        )
        minimax_h3_qk_norm_partial_rope_inplace(
            split_qkv.k, weights[prefix + "attn.k_norm.weight"][],
            cos, sin, heads, rotary_dim, qk_norm_eps, ctx,
        )
        ctx.synchronize()
        cu_mempool_trim_current(0)
        comptime if HeadDim != 128:
            if attention_backend != MINIMAX_H3_ATTN_CUDNN:
                raise Error(
                    "minimax_h3_block_forward: sage-int8 requires head_dim=128"
                )
        attn = _minimax_h3_attention_dispatch(
            split_qkv.q, split_qkv.k, split_qkv.v, attn_scale, ctx,
            attention_backend, sage_scratch, sage_ultravico,
        )
        # Consume Q/K/V before the branch-local owner is destroyed.
        ctx.synchronize()
    else:
        var qkv_out = _minimax_h3_block_linear(
            attn_in, weights, qkv_name, ctx
        )
        var q = slice(qkv_out, 2, 0 * inner, inner, ctx)
        var k = slice(qkv_out, 2, 1 * inner, inner, ctx)
        var v = slice(qkv_out, 2, 2 * inner, inner, ctx)
        var q4 = reshape_owned(q^, [1, S, heads, head_dim])
        var k4 = reshape_owned(k^, [1, S, heads, head_dim])
        var v4 = reshape_owned(v^, [1, S, heads, head_dim])
        ctx.synchronize()
        q4 = vec_rms_norm(
            q4, weights[prefix + "attn.q_norm.weight"][], qk_norm_eps, ctx
        )
        k4 = vec_rms_norm(
            k4, weights[prefix + "attn.k_norm.weight"][], qk_norm_eps, ctx
        )
        q4 = _minimax_h3_apply_partial_rope(
            q4, cos, sin, heads, rotary_dim, ctx
        )
        k4 = _minimax_h3_apply_partial_rope(
            k4, cos, sin, heads, rotary_dim, ctx
        )
        ctx.synchronize()
        cu_mempool_trim_current(0)
        comptime if HeadDim != 128:
            if attention_backend != MINIMAX_H3_ATTN_CUDNN:
                raise Error(
                    "minimax_h3_block_forward: sage-int8 requires head_dim=128"
                )
        attn = _minimax_h3_attention_dispatch(
            q4, k4, v4, attn_scale, ctx, attention_backend, sage_scratch,
            sage_ultravico,
        )
        ctx.synchronize()
    var merged = reshape_owned(attn^, [1, S, inner])
    var x1 = _minimax_h3_block_residual_linear(
        merged, x, gate, weights, prefix + "attn.out_proj.weight", True, ctx
    )
    ctx.synchronize()
    cu_mempool_trim_current(0)
    return x1^


def _minimax_h3_block_forward_low_headroom[
    Heads: Int, HeadDim: Int
](
    x: Tensor,
    weights: Dict[String, ArcPointer[Tensor]],
    layer: Int,
    config: MiniMaxH3DiTConfig,
    mod: Tensor,
    adaln_indices: List[Int],
    cos: Tensor,
    sin: Tensor,
    rotary_dim: Int,
    ctx: DeviceContext,
    attention_backend: Int,
    sage_scratch: Optional[SageInt8Scratch] = None,
    sage_ultravico: Optional[SageUltraViCoConfig] = None,
) raises -> Tensor:
    var prefix = minimax_h3_block_prefix(layer)
    var x1 = _minimax_h3_low_headroom_attention[Heads, HeadDim](
        x, weights, prefix, mod, adaln_indices, config.hidden_size,
        config.inner_dim(), config.num_attention_heads,
        config.attention_head_dim, config.norm_eps, config.qk_norm_eps,
        cos, sin, rotary_dim, ctx, attention_backend, sage_scratch,
        sage_ultravico,
    )
    # The attention helper's local QKV/rope/SDPA tensors are destroyed only
    # after it returns. Trim here—not inside that helper—so their pool pages
    # are genuinely releasable.
    ctx.synchronize()
    cu_mempool_trim_current(0)
    var fc1_name = prefix + "mlp.fc1.weight"
    var fc2_name = prefix + "mlp.fc2.weight"
    ref fc1_weight = weights[fc1_name][]
    ref fc2_weight = weights[fc2_name][]
    var fc1_scale_name = fc1_name + String(".scale")
    var fc2_scale_name = fc2_name + String(".scale")
    if fc1_weight.dtype() == STDtype.I8 and fc2_weight.dtype() == STDtype.I8 \
            and fc1_scale_name in weights and fc2_scale_name in weights \
            and weights[fc1_scale_name][].dtype() == STDtype.F32 \
            and weights[fc2_scale_name][].dtype() == STDtype.F32:
        ref fc1_scale = weights[fc1_scale_name][]
        ref fc2_scale = weights[fc2_scale_name][]
        var shift = _minimax_h3_gather_mod_chunk(
            mod, adaln_indices, 3, config.hidden_size, ctx
        )
        var scale_mlp = _minimax_h3_gather_mod_chunk(
            mod, adaln_indices, 4, config.hidden_size, ctx
        )
        var gate = _minimax_h3_gather_mod_chunk(
            mod, adaln_indices, 5, config.hidden_size, ctx
        )
        var mlp_in = _minimax_h3_rms_norm_modulate(
            x1, weights[prefix + "norm2.weight"][], scale_mlp, shift,
            config.norm_eps, ctx,
        )
        ctx.synchronize()
        cu_mempool_trim_current(0)
        return minimax_h3_int8_mlp_residual_inplace(
            mlp_in, fc1_weight, fc1_scale, fc2_weight, fc2_scale,
            x1^, gate, ctx,
        )
    if fc1_weight.dtype() == STDtype.I8 and fc2_weight.dtype() == STDtype.I8 \
            and fc1_scale_name in weights and fc2_scale_name in weights \
            and weights[fc1_scale_name][].dtype() == STDtype.F16 \
            and weights[fc2_scale_name][].dtype() == STDtype.F16:
        ref fc1_scale = weights[fc1_scale_name][]
        ref fc2_scale = weights[fc2_scale_name][]
        var shift = _minimax_h3_gather_mod_chunk(
            mod, adaln_indices, 3, config.hidden_size, ctx
        )
        var scale_mlp = _minimax_h3_gather_mod_chunk(
            mod, adaln_indices, 4, config.hidden_size, ctx
        )
        var gate = _minimax_h3_gather_mod_chunk(
            mod, adaln_indices, 5, config.hidden_size, ctx
        )
        var mlp_in = _minimax_h3_rms_norm_modulate(
            x1, weights[prefix + "norm2.weight"][], scale_mlp, shift,
            config.norm_eps, ctx,
        )
        ctx.synchronize()
        cu_mempool_trim_current(0)
        return minimax_h3_groupwise_mlp_residual_inplace(
            mlp_in, fc1_weight, fc1_scale, fc2_weight, fc2_scale,
            x1^, gate, ctx,
        )

    if fc1_weight.dtype() == STDtype.BF16 \
            and fc2_weight.dtype() == STDtype.BF16:
        var shift = _minimax_h3_gather_mod_chunk(
            mod, adaln_indices, 3, config.hidden_size, ctx
        )
        var scale_mlp = _minimax_h3_gather_mod_chunk(
            mod, adaln_indices, 4, config.hidden_size, ctx
        )
        var gate = _minimax_h3_gather_mod_chunk(
            mod, adaln_indices, 5, config.hidden_size, ctx
        )
        var mlp_in = _minimax_h3_rms_norm_modulate(
            x1, weights[prefix + "norm2.weight"][], scale_mlp, shift,
            config.norm_eps, ctx,
        )
        ctx.synchronize()
        cu_mempool_trim_current(0)
        return minimax_h3_bf16_mlp_residual_inplace(
            mlp_in, fc1_weight, fc2_weight, x1^, gate, ctx
        )

    var prepared = _minimax_h3_prepare_low_headroom_mlp(
        x1, mod, adaln_indices, config.hidden_size, weights, prefix,
        config.norm_eps, ctx,
    )
    ctx.synchronize()
    cu_mempool_trim_current(0)
    return _minimax_h3_block_residual_linear(
        prepared.act, x1, prepared.gate, weights, fc2_name, True, ctx
    )


def _minimax_h3_block_forward_impl[
    Heads: Int = MINIMAX_H3_HEADS, HeadDim: Int = MINIMAX_H3_HEAD_DIM
](
    x: Tensor,
    weights: Dict[String, ArcPointer[Tensor]],
    layer: Int,
    config: MiniMaxH3DiTConfig,
    mod: Tensor,
    adaln_indices: List[Int],
    cos: Tensor,
    sin: Tensor,
    rotary_dim: Int,
    ctx: DeviceContext,
    attention_backend: Int = MINIMAX_H3_ATTN_CUDNN,
    sage_scratch: Optional[SageInt8Scratch] = None,
    sage_ultravico: Optional[SageUltraViCoConfig] = None,
    lora_overlay: Optional[H3LoraOverlay] = Optional[H3LoraOverlay](None),
) raises -> Tensor:
    """One `MiniMaxH3TransformerBlock`. Mirrors `models/minimax_h3/
    block_forward.mojo::_transformer_block` op-for-op (see that file for the
    diffusers line citations this ports); this file's header has the block
    math summary and the adaLN row-addressing trap.

    `Heads`/`HeadDim` are comptime SDPA-instantiation parameters, defaulted to
    the released model's own geometry (56/128) so existing call sites that
    only specify `[S]` are unchanged. They exist as parameters — not hardcoded
    — because `sdpa_flash_infer_fwd` (ops/attention_flash.mojo) is itself
    comptime-generic over heads/head_dim (a cuDNN shim call, no MMA-tiling
    restriction); the earlier version of this function hardcoded them to the
    file-level `MINIMAX_H3_HEADS`/`MINIMAX_H3_HEAD_DIM` constants regardless of
    the caller, which meant the function could NEVER be instantiated at any
    geometry other than the release — including the tiny random-weight
    fixture that is the only oracle-gated reference for this block's math
    without a real checkpoint (`models/minimax_h3/parity/
    minimax_h3_block_parity.mojo`, `scripts/minimax_h3_block_oracle.py`). The
    runtime check below still enforces that `config` agrees with whatever
    `Heads`/`HeadDim` this instantiation was compiled with — the ORIGINAL
    guard's intent (no silent reinterpret of a mismatched config) is correct;
    only its rigidity (always 56/128, no matter what) was the defect.

    x:             `[1, S, hidden]` bf16 — the packed video+text+audio
                   sequence for this denoising step.
    weights:       this block's tensors, right SHAPE per
                   `minimax_h3_check_block_weights` (which must already have
                   passed for `layer` — preflight is the caller's job) AND
                   right ROW ORDER per this function's WEIGHT CONTRACT note
                   above (`minimax_h3_load_block_device` output, not raw
                   `ShardedSafeTensors`/`BlockLoader` tensors). The row-order
                   half of that contract IS enforced here, unconditionally:
                   `minimax_h3_require_transformed_weights` (this file) is
                   this function's first statement and raises if `weights`
                   is missing either of the two marker keys
                   `minimax_h3_load_block_device` stamps after transforming.
    layer:         which of the 50 blocks (keys the weight lookup via
                   `minimax_h3_block_prefix`).
    mod:           THIS LAYER's precomputed AdaLN modulation table,
                   `[distinct_timesteps*3, 6*hidden]` — this layer's own
                   `adaln_proj` applied to the timestep embedding, cached
                   ONCE per run rather than loaded here (`adaln_proj` is
                   13.04B params over the stack; see this file's header and
                   `models/minimax_h3/fp8_policy.mojo` for the residency
                   arithmetic). Row layout `[shift_msa|scale_msa|gate_msa|
                   shift_mlp|scale_mlp|gate_mlp]`, each `hidden` wide. F32 or
                   BF16 both work (`gather_rows`/`modulate`/`residual_gate`
                   are dtype-flexible and cast to `x`'s dtype internally);
                   `models/dit/minimax_h3_modcache.mojo::
                   minimax_h3_build_modulation_cache` currently produces BF16.
    adaln_indices: length-`S` row -> `mod` row map, from
                   `minimax_h3_adaln_rows`.
    cos/sin:       `[S, rotary_dim]` f32 rotary table (unit 5,
                   `dit_frontend.minimax_h3_rope_table`, or its device
                   counterpart `models/dit/minimax_h3_rope.mojo::
                   build_minimax_h3_rope_tables`), shared by every block.
    rotary_dim:    rotated channel count (96 at the released geometry);
                   `head_dim - rotary_dim` channels pass through untouched.
    attention_backend: runtime selector: cuDNN (default) or the opt-in local
                   Sage INT8-QK backend. Unknown values and unsupported Sage
                   geometry fail loudly; there is no silent fallback.
    """
    minimax_h3_require_transformed_weights(weights, layer)

    var prefix = minimax_h3_block_prefix(layer)
    var x_shape = x.shape()
    if len(x_shape) != 3 or x_shape[0] != 1:
        raise Error("minimax_h3_block_forward: x must be [1,S,hidden]")
    var S = x_shape[1]
    var hidden = config.hidden_size
    var heads = config.num_attention_heads
    var head_dim = config.attention_head_dim
    var inner = config.inner_dim()
    var ffn = config.ffn_hidden_size
    if heads != Heads or head_dim != HeadDim:
        raise Error(
            String("minimax_h3_block_forward: config heads/head_dim (")
            + String(heads) + "/" + String(head_dim)
            + ") do not match this call's comptime SDPA instantiation ("
            + String(Heads) + "/" + String(HeadDim)
            + ") — pass matching `Heads`/`HeadDim` type parameters rather"
            " than silently reinterpreting a mismatched config"
        )

    # ── AdaLN row gather: one [hidden]-wide chunk per parameter ────────────
    # The ordinary path remains byte-for-byte unchanged.  At S>=48k, the old
    # full gather alone is >4.4 GiB, then its six materialized slices add the
    # same amount again.  Gather only the attention branch here; the MLP
    # branch is gathered after attention has completed.
    var low_headroom = S >= 48000
    if low_headroom:
        if lora_overlay and lora_overlay.value().has_layer(layer):
            raise Error(
                "MiniMax-H3 activation LoRA is not admitted for S>=48000: "
                "the low-headroom fused MLP cannot expose the four projection "
                "outputs safely under the 24 GiB runtime cap"
            )
        return _minimax_h3_block_forward_low_headroom[Heads, HeadDim](
            x, weights, layer, config, mod, adaln_indices, cos, sin,
            rotary_dim, ctx, attention_backend, sage_scratch, sage_ultravico,
        )
    var shift_msa: Tensor
    var scale_msa: Tensor
    var gate_msa: Tensor
    var shift_mlp = Optional[Tensor](None)
    var scale_mlp = Optional[Tensor](None)
    var gate_mlp = Optional[Tensor](None)
    if low_headroom:
        shift_msa = _minimax_h3_gather_mod_chunk(
            mod, adaln_indices, 0, hidden, ctx
        )
        scale_msa = _minimax_h3_gather_mod_chunk(
            mod, adaln_indices, 1, hidden, ctx
        )
        gate_msa = _minimax_h3_gather_mod_chunk(
            mod, adaln_indices, 2, hidden, ctx
        )
    else:
        var gathered = gather_rows(mod, adaln_indices, ctx)
        shift_msa = slice(gathered, 1, 0 * hidden, hidden, ctx)
        scale_msa = slice(gathered, 1, 1 * hidden, hidden, ctx)
        gate_msa = slice(gathered, 1, 2 * hidden, hidden, ctx)
        var shift_mlp_t = slice(gathered, 1, 3 * hidden, hidden, ctx)
        var scale_mlp_t = slice(gathered, 1, 4 * hidden, hidden, ctx)
        var gate_mlp_t = slice(gathered, 1, 5 * hidden, hidden, ctx)
        shift_mlp = Optional[Tensor](shift_mlp_t^)
        scale_mlp = Optional[Tensor](scale_mlp_t^)
        gate_mlp = Optional[Tensor](gate_mlp_t^)

    # ── Self-attention branch ──
    var attn_in = _minimax_h3_rms_norm_modulate(
        x, weights[prefix + "norm1.weight"][], scale_msa, shift_msa,
        config.norm_eps, ctx,
    )
    if low_headroom:
        # `attn_in` is now complete; release n1 plus shift/scale before the
        # 3*inner QKV output is allocated.
        ctx.synchronize()

    var scale = Float32(1.0) / (Float32(head_dim) ** Float32(0.5))
    # Dh=128 on sm_86 fails to instantiate the SDK's `flash_attention` MMA
    # tiling (serenitymojo/MAP.md gotcha 4); this is the cuDNN v9 shim
    # instead, which carries no such restriction. No mask, no causal — H3 is
    # one packed attention document per block, no cross-attention anywhere
    # (oracle comment, block_forward.mojo `attention()`). `Heads`/`HeadDim`
    # are this function's own comptime parameters (default 56/128, the
    # released geometry) — NOT the file-level MINIMAX_H3_HEADS/
    # MINIMAX_H3_HEAD_DIM constants — so a caller instantiating at a
    # different geometry (e.g. a small test fixture) gets a matching SDPA
    # kernel instead of an unconditional guard raise. The check above
    # ensures `config` was built for exactly this instantiation.
    var attn: Tensor
    var qkv_name = prefix + "attn.qkv_proj.weight"
    ref qkv_weight = weights[qkv_name][]
    var qkv_scale_name = qkv_name + String(".scale")
    var qkv_lora = False
    if lora_overlay:
        qkv_lora = lora_overlay.value().has(layer, 0)
    var direct_w8a8_qkv = (
        qkv_weight.dtype() == STDtype.I8
        and qkv_scale_name in weights
        and weights[qkv_scale_name][].dtype() == STDtype.F32
        and not qkv_lora
    )
    if direct_w8a8_qkv:
        # Write Q/K/V directly from the single packed W8A8 accumulator. The
        # old ordinary path materialized `[S,3*inner]` and then launched three
        # full-tensor slice copies per block. Do not select the BF16/groupwise
        # split writers here: they require three separate GEMMs and are kept
        # only for the >=48k low-headroom path where peak memory is decisive.
        var split_qkv: MiniMaxH3Int8QKV
        ref qkv_scale = weights[qkv_scale_name][]
        split_qkv = minimax_h3_int8_qkv_linear(
            attn_in, qkv_weight, qkv_scale, ctx
        )
        reshape_in_place(split_qkv.q, [1, S, heads, head_dim])
        reshape_in_place(split_qkv.k, [1, S, heads, head_dim])
        reshape_in_place(split_qkv.v, [1, S, heads, head_dim])
        minimax_h3_qk_norm_partial_rope_inplace(
            split_qkv.q, weights[prefix + "attn.q_norm.weight"][],
            cos, sin, heads, rotary_dim, config.qk_norm_eps, ctx,
        )
        minimax_h3_qk_norm_partial_rope_inplace(
            split_qkv.k, weights[prefix + "attn.k_norm.weight"][],
            cos, sin, heads, rotary_dim, config.qk_norm_eps, ctx,
        )
        attn = _minimax_h3_attention_dispatch(
            split_qkv.q, split_qkv.k, split_qkv.v,
            scale, ctx, attention_backend, sage_scratch, sage_ultravico,
        )
        # The split owner is branch-local.  Finish its final consumer before
        # its three buffers are returned to the stream-ordered allocator.
        ctx.synchronize()
    else:
        # Compatibility fallback for non-production test weights.
        var qkv_out = _minimax_h3_block_linear_lora(
            attn_in, weights, qkv_name, layer, 0, lora_overlay, ctx
        )
        var q = slice(qkv_out, 2, 0 * inner, inner, ctx)
        var k = slice(qkv_out, 2, 1 * inner, inner, ctx)
        var v = slice(qkv_out, 2, 2 * inner, inner, ctx)
        var q4 = reshape_owned(q^, [1, S, heads, head_dim])
        var k4 = reshape_owned(k^, [1, S, heads, head_dim])
        var v4 = reshape_owned(v^, [1, S, heads, head_dim])
        minimax_h3_qk_norm_partial_rope_inplace(
            q4, weights[prefix + "attn.q_norm.weight"][],
            cos, sin, heads, rotary_dim, config.qk_norm_eps, ctx,
        )
        minimax_h3_qk_norm_partial_rope_inplace(
            k4, weights[prefix + "attn.k_norm.weight"][],
            cos, sin, heads, rotary_dim, config.qk_norm_eps, ctx,
        )
        attn = _minimax_h3_attention_dispatch(
            q4, k4, v4, scale, ctx, attention_backend, sage_scratch,
            sage_ultravico,
        )
        ctx.synchronize()
                                                                        # [1, S, heads, head_dim] bf16
    var merged = reshape_owned(attn^, [1, S, inner])
    var x1 = _minimax_h3_block_residual_linear_lora(
        merged, x, gate_msa, weights, prefix + "attn.out_proj.weight",
        layer, 1, lora_overlay, ctx,
    )

    # ── MLP branch ──
    var x2: Tensor
    if low_headroom:
        # Complete and release the attention branch, then prepare FC1 inside a
        # nested lifetime. Only its activation and residual gate survive into
        # fused FC2; shift/scale/norm/input buffers die at helper return.
        ctx.synchronize()
        cu_mempool_trim_current(0)
        var prepared = _minimax_h3_prepare_low_headroom_mlp(
            x1, mod, adaln_indices, hidden, weights, prefix,
            config.norm_eps, ctx,
        )
        x2 = _minimax_h3_block_residual_linear(
            prepared.act, x1, prepared.gate, weights,
            prefix + "mlp.fc2.weight", True, ctx,
        )
    else:
        var mlp_in = _minimax_h3_rms_norm_modulate(
            x1, weights[prefix + "norm2.weight"][], scale_mlp.value(),
            shift_mlp.value(), config.norm_eps, ctx,
        )
        # fc1 is swapped to [value;gate] by the transformed loader.
        var act = _minimax_h3_block_swiglu_linear_lora(
            mlp_in, weights, prefix + "mlp.fc1.weight", layer,
            lora_overlay, ctx,
        )
        x2 = _minimax_h3_block_residual_linear_lora(
            act, x1, gate_mlp.value(), weights, prefix + "mlp.fc2.weight",
            layer, 3, lora_overlay, ctx,
        )
    return x2^


def minimax_h3_block_forward[
    S: Int, Heads: Int = MINIMAX_H3_HEADS, HeadDim: Int = MINIMAX_H3_HEAD_DIM
](
    x: Tensor,
    weights: Dict[String, ArcPointer[Tensor]],
    layer: Int,
    config: MiniMaxH3DiTConfig,
    mod: Tensor,
    adaln_indices: List[Int],
    cos: Tensor,
    sin: Tensor,
    rotary_dim: Int,
    ctx: DeviceContext,
    attention_backend: Int = MINIMAX_H3_ATTN_CUDNN,
    sage_scratch: Optional[SageInt8Scratch] = None,
    sage_ultravico: Optional[SageUltraViCoConfig] = None,
    lora_overlay: Optional[H3LoraOverlay] = Optional[H3LoraOverlay](None),
) raises -> Tensor:
    """Static-S compatibility surface for parity gates and existing callers."""
    var x_shape = x.shape()
    if len(x_shape) != 3 or x_shape[1] != S:
        raise Error("minimax_h3_block_forward: runtime S != static S")
    return _minimax_h3_block_forward_impl[Heads, HeadDim](
        x, weights, layer, config, mod, adaln_indices, cos, sin, rotary_dim,
        ctx, attention_backend, sage_scratch, sage_ultravico, lora_overlay,
    )


def minimax_h3_block_forward_dynamic[
    Heads: Int = MINIMAX_H3_HEADS, HeadDim: Int = MINIMAX_H3_HEAD_DIM
](
    x: Tensor,
    weights: Dict[String, ArcPointer[Tensor]],
    layer: Int,
    config: MiniMaxH3DiTConfig,
    mod: Tensor,
    adaln_indices: List[Int],
    cos: Tensor,
    sin: Tensor,
    rotary_dim: Int,
    ctx: DeviceContext,
    attention_backend: Int = MINIMAX_H3_ATTN_CUDNN,
    sage_scratch: Optional[SageInt8Scratch] = None,
    sage_ultravico: Optional[SageUltraViCoConfig] = None,
    lora_overlay: Optional[H3LoraOverlay] = Optional[H3LoraOverlay](None),
) raises -> Tensor:
    """Runtime-S product surface used by arbitrary H3 canvas geometry."""
    return _minimax_h3_block_forward_impl[Heads, HeadDim](
        x, weights, layer, config, mod, adaln_indices, cos, sin, rotary_dim,
        ctx, attention_backend, sage_scratch, sage_ultravico, lora_overlay,
    )
