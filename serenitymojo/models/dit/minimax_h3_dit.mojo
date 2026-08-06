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

from std.collections import Dict, List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.ops.vec_rms_norm import vec_rms_norm
from serenitymojo.ops.activations import swiglu_packed_value_gate
from serenitymojo.ops.linear import linear
from serenitymojo.ops.int8_quant import int8_dequant_groupwise_to_bf16
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.rope import rope_halfsplit_full_head_broadcast
from serenitymojo.ops.attention_flash import sdpa_flash_infer_fwd
from serenitymojo.ops.sage_attention_int8 import sage_attention_int8_fwd
from serenitymojo.models.dit.minimax_h3_int8_linear import (
    minimax_h3_bf16_linear_chunked,
    minimax_h3_int8_linear,
)
from serenitymojo.ops.tensor_algebra import (
    gather_rows,
    slice,
    reshape,
    reshape_owned,
    concat,
    add,
    full_device,
)


comptime MINIMAX_H3_ATTN_CUDNN = 0
comptime MINIMAX_H3_ATTN_SAGE_INT8 = 1


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

    # ── AdaLN row gather: one [hidden]-wide chunk per (shift/scale/gate) x (msa/mlp) ──
    var gathered = gather_rows(mod, adaln_indices, ctx)              # [S, 6*hidden] f32
    var shift_msa = slice(gathered, 1, 0 * hidden, hidden, ctx)
    var scale_msa = slice(gathered, 1, 1 * hidden, hidden, ctx)
    var gate_msa = slice(gathered, 1, 2 * hidden, hidden, ctx)
    var shift_mlp = slice(gathered, 1, 3 * hidden, hidden, ctx)
    var scale_mlp = slice(gathered, 1, 4 * hidden, hidden, ctx)
    var gate_mlp = slice(gathered, 1, 5 * hidden, hidden, ctx)

    # ── Self-attention branch ──
    var n1 = vec_rms_norm(x, weights[prefix + "norm1.weight"][], config.norm_eps, ctx)
    var attn_in = modulate(n1, scale_msa, shift_msa, ctx)

    # qkv_proj is already de-interleaved into [q_all;k_all;v_all] thirds by
    # `minimax_h3_load_block_device` — no gather here (see this function's
    # weight-contract note above).
    var qkv_out = _minimax_h3_block_linear(
        attn_in, weights, prefix + "attn.qkv_proj.weight", ctx
    )                                                                # [1, S, 3*inner]

    var q = slice(qkv_out, 2, 0 * inner, inner, ctx)
    var k = slice(qkv_out, 2, 1 * inner, inner, ctx)
    var v = slice(qkv_out, 2, 2 * inner, inner, ctx)
    var q4 = reshape_owned(q^, [1, S, heads, head_dim])
    var k4 = reshape_owned(k^, [1, S, heads, head_dim])
    var v4 = reshape_owned(v^, [1, S, heads, head_dim])

    q4 = vec_rms_norm(q4, weights[prefix + "attn.q_norm.weight"][], config.qk_norm_eps, ctx)
    k4 = vec_rms_norm(k4, weights[prefix + "attn.k_norm.weight"][], config.qk_norm_eps, ctx)

    q4 = _minimax_h3_apply_partial_rope(
        q4, cos, sin, heads, rotary_dim, ctx
    )
    k4 = _minimax_h3_apply_partial_rope(
        k4, cos, sin, heads, rotary_dim, ctx
    )

    var scale = Float32(1.0) / (Float32(head_dim) ** 0.5)
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
    if attention_backend == MINIMAX_H3_ATTN_CUDNN:
        attn = sdpa_flash_infer_fwd[1, S, Heads, HeadDim](
            q4, k4, v4, scale, ctx
        )
    elif attention_backend == MINIMAX_H3_ATTN_SAGE_INT8:
        comptime if HeadDim == 128:
            attn = sage_attention_int8_fwd[1, S, Heads, HeadDim](
                q4, k4, v4, scale, ctx
            )
        else:
            raise Error(
                "minimax_h3_block_forward: sage-int8 requires head_dim=128"
            )
    else:
        raise Error(
            String("minimax_h3_block_forward: unknown attention backend ")
            + String(attention_backend)
        )
                                                                        # [1, S, heads, head_dim] bf16
    var merged = reshape_owned(attn^, [1, S, inner])
    var attn_out = _minimax_h3_block_linear(
        merged, weights, prefix + "attn.out_proj.weight", ctx
    )
    var x1 = residual_gate(x, gate_msa, attn_out, ctx)

    # ── MLP branch ──
    var n2 = vec_rms_norm(x1, weights[prefix + "norm2.weight"][], config.norm_eps, ctx)
    var mlp_in = modulate(n2, scale_mlp, shift_mlp, ctx)
    # fc1 is already swapped to [value;gate] by `minimax_h3_load_block_device`
    # (matches the oracle's own swiglu_ff convention: value first, gate second).
    var fc1_out = _minimax_h3_block_linear(
        mlp_in, weights, prefix + "mlp.fc1.weight", ctx
    )  # [1,S,2*ffn], [value;gate]
    var act = swiglu_packed_value_gate(fc1_out, ctx)                # silu(gate)*value
    var ff_out = _minimax_h3_block_linear(
        act, weights, prefix + "mlp.fc2.weight", ctx
    )
    var x2 = residual_gate(x1, gate_mlp, ff_out, ctx)
    return x2^
