# serenitymojo/models/dit/minimax_h3_modcache.mojo
#
# MiniMax-H3 AdaLN MODULATION CACHE — the single most important memory
# optimization in this model.
#
# `blocks.N.adaln_proj.linear` is [96768, 2688] per block, 13.04 B of the
# 33.12 B model — 39%, 24.288 GiB bf16 (models/minimax_h3/fp8_policy.mojo
# unit 14, ADALN class). Its INPUT is the timestep embedding alone, identical
# for every token in the packed sequence, so the whole projection can be
# consumed in ONE streamed pass before sampling starts and never read again:
#
#     mod[block] = silu(temb) @ adaln_w[block].T + adaln_b[block]
#
# `temb` is `[distinct_timesteps, time_embed_dim]` F32 — the caller's
# `time_embedder` output. THIS MODULE DOES NOT COMPUTE temb (see
# `models/minimax_h3/dit_frontend.minimax_h3_timestep_projection` /
# `time_embedder.linear_1`/`linear_2` for that piece, and
# `models/minimax_h3/fp8_policy.minimax_h3_adaln_distinct_timesteps` for how
# many rows it needs — NOT `steps`, see that HOW MANY ROWS section below).
#
# `mod[block]` comes out `[distinct_timesteps, 96768]` and is VIEWED (same
# bytes, no copy: row-major `[N, 96768]` and `[N*3, 32256]` are the identical
# flat layout since `96768 == 3 * 32256`) as
# `[distinct_timesteps * 3, 6 * hidden_size]`, addressed by
# `minimax_h3_adaln_row(timestep_index, token_tag)` — already written and
# gated in `minimax_h3_dit.mojo`, reused here rather than reimplemented. The
# vendor's own README: "Because the AdaLN modulation outputs can be
# precomputed and cached, these parameters do not need to be loaded for
# inference-only deployment."
#
# The final layer (`final_layer.adaln_proj.linear` in the ORIGINAL checkpoint,
# `[10752, 2688]` -> `[distinct_timesteps, 2 * hidden_size]`) gets the same
# treatment, but addressed by TIMESTEP ALONE — `block_forward.mojo`'s step 7
# indexes `norm_out`'s shift/scale by `timestep_indices[r]`, not by
# `(timestep, modality)`: there is no per-modality final norm, so no reshape
# is needed here — the `[distinct_timesteps, final_adaln_out_features]` shape
# `linear_bias` already produces IS the addressable table.
#
# ─────────────────────────────────────────────────────────────────────────────
# ACTIVATION PRECISION TRAP (reproduced from the reference, deliberately)
#
# `time_embedder.*` is one of the checkpoint's 12 F32 tensors
# (`models/minimax_h3/loader.mojo` `minimax_h3_is_fp32_key`); everything else
# touched here, INCLUDING the adaLN weights, is bf16 (loader.mojo: "everything
# else is bfloat16, INCLUDING the adaLN projections"). So SiLU runs at temb's
# own F32 precision and ONLY ITS RESULT is cast down to bf16 before the
# projection:
#
#     activated_f32  = silu(temb)               F32, ops/activations.silu
#     activated_bf16 = cast(activated_f32)       the ONE rounding boundary
#     mod            = activated_bf16 @ w.T + b  bf16 x bf16, F32-accumulated
#                                                 GEMM (ops/linear.linear_bias)
#
# Every block reads the SAME temb, so this rounding is applied IDENTICALLY at
# every block and every sampling step, and accumulates coherently over the
# whole trajectory — it is not noise to average away, it is what the
# checkpoint's own mixed-precision layout produces. Computing the whole
# projection in F32 instead would be a DIFFERENT, unfaithful computation, not
# a more-accurate one.
#
# ─────────────────────────────────────────────────────────────────────────────
# HOW MANY ROWS (`distinct_timesteps`) — NOT `steps`. See
# `models/minimax_h3/fp8_policy.minimax_h3_adaln_distinct_timesteps` (already
# gated, 27 checks): H3 runs video and audio on SEPARATE sigma schedules
# (shift 12.0 vs 3.0), so a run needs `2 * steps` rows plus up to 2 pinned
# conditioning rows — not `steps`. This module takes `temb` already built at
# that row count; it is the CALLER's job (whoever builds the packed-timestep
# schedule) to size it correctly. This module only checks `temb.shape()[0]`
# is positive — it cannot detect an undercount from here.
#
# ─────────────────────────────────────────────────────────────────────────────
# MEMORY DISCIPLINE. One block's adaln weight (~495.6 MiB bf16 at the
# released hidden_size=5376, time_embed_dim=2688) is streamed, projected, and
# dropped before the next block is read: `w`/`b` below are loop-local and are
# freed (Mojo end-of-scope Tensor destructor, same discipline
# `wan22_fp8_stream.mojo`/`krea2_fp8_resident` already rely on) at the end of
# each iteration. Never more than one block's adaln weight is resident at
# once. What STAYS resident is the RESULT: bf16, ~0.92 GiB at the ref2va@50
# row count (measured and cross-checked arithmetically in
# `models/dit/parity/minimax_h3_modcache_probe.mojo`).
#
# REUSE, not reimplemented:
#   models/dit/minimax_h3_dit.mojo   MiniMaxH3DiTConfig, minimax_h3_released_config,
#                                     minimax_h3_block_prefix, minimax_h3_adaln_row
#   ops/linear.mojo                  linear_bias  (F32-accumulated GEMM + bias add)
#   ops/activations.mojo             silu
#   ops/cast.mojo                    cast_tensor
#   ops/tensor_algebra.mojo          reshape_owned (metadata-only, no D2D copy), slice
#   io/sharded.mojo                  ShardedSafeTensors
#
# ORACLE: `models/minimax_h3/block_forward.mojo` `_modulation` (:285-308) is
# the readable host-F32 math description this mirrors, and
# `models/dit/minimax_h3_dit.mojo`'s ADALN ROW ADDRESSING section (:159-201)
# is the row-index contract this reuses verbatim. Per house rule (GPU code
# must be gated against a GPU-generated reference, never a CPU one), this
# module's own probe does NOT compare against `_modulation`'s host-F32 output
# bit-for-bit; it compares two GPU-computed paths (production bf16 vs a
# reference computed by the SAME ops/linear.mojo + ops/activations.mojo
# kernels at F32) — see the probe's header for the reasoning.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.collections import List
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.tensor import Tensor
from serenitymojo.ops.activations import silu
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear_bias
from serenitymojo.ops.tensor_algebra import reshape_owned, slice

from serenitymojo.models.dit.minimax_h3_dit import (
    MiniMaxH3DiTConfig,
    minimax_h3_adaln_row,
    minimax_h3_block_prefix,
)

comptime TArc = ArcPointer[Tensor]

comptime MINIMAX_H3_FINAL_ADALN_WEIGHT_KEY = String(
    "final_layer.adaln_proj.linear.weight"
)
comptime MINIMAX_H3_FINAL_ADALN_BIAS_KEY = String(
    "final_layer.adaln_proj.linear.bias"
)


def minimax_h3_block_adaln_weight_key(layer: Int) -> String:
    """ORIGINAL-checkpoint key for block `layer`'s adaLN projection weight.
    Not loaded by `minimax_h3_block_tensor_names` (minimax_h3_dit.mojo) —
    that list deliberately excludes adaln_proj; this module is the only
    reader of these keys."""
    return minimax_h3_block_prefix(layer) + "adaln_proj.linear.weight"


def minimax_h3_block_adaln_bias_key(layer: Int) -> String:
    return minimax_h3_block_prefix(layer) + "adaln_proj.linear.bias"


def _check_adaln_tensor(
    shards: ShardedSafeTensors, key: String, want_rows: Int, want_cols: Int
) raises:
    """One tensor's presence/shape/dtype. `want_cols == 0` means a 1-D bias
    vector of length `want_rows`."""
    if key not in shards.name_to_shard:
        raise Error(String("MiniMax-H3 modcache: missing adaLN tensor ") + key)
    var info = shards.tensor_info(key)
    if want_cols > 0:
        if (
            len(info.shape) != 2
            or info.shape[0] != want_rows
            or info.shape[1] != want_cols
        ):
            raise Error(
                String("MiniMax-H3 modcache: ") + key + " shape mismatch, want ["
                + String(want_rows) + "," + String(want_cols) + "]"
            )
    else:
        if len(info.shape) != 1 or info.shape[0] != want_rows:
            raise Error(
                String("MiniMax-H3 modcache: ") + key + " shape mismatch, want ["
                + String(want_rows) + "]"
            )
    if info.dtype != STDtype.BF16:
        raise Error(
            String("MiniMax-H3 modcache: ") + key + " is not BF16 — the adaLN"
            " projections are bf16 in the ORIGINAL checkpoint"
            " (loader.mojo minimax_h3_is_fp32_key); loading one upcast would"
            " both mismatch the reference math and blow the residency budget"
            " this module exists to save."
        )


def minimax_h3_check_modcache_weights(
    shards: ShardedSafeTensors, config: MiniMaxH3DiTConfig
) raises:
    """Preflight EVERY adaLN tensor — all `config.num_layers` blocks plus the
    final layer — BEFORE loading any of them. A partial download (the
    checkpoint is 61.7 GiB and streams in over time) then fails with ONE
    clear message naming the first missing/malformed tensor, rather than
    stopping mid-stream after some blocks are already cached. Never hangs:
    every check here is a single mmap'd header lookup."""
    for layer in range(config.num_layers):
        _check_adaln_tensor(
            shards,
            minimax_h3_block_adaln_weight_key(layer),
            config.adaln_out_features,
            config.time_embed_dim,
        )
        _check_adaln_tensor(
            shards,
            minimax_h3_block_adaln_bias_key(layer),
            config.adaln_out_features,
            0,
        )
    _check_adaln_tensor(
        shards,
        MINIMAX_H3_FINAL_ADALN_WEIGHT_KEY,
        config.final_adaln_out_features,
        config.time_embed_dim,
    )
    _check_adaln_tensor(
        shards,
        MINIMAX_H3_FINAL_ADALN_BIAS_KEY,
        config.final_adaln_out_features,
        0,
    )


@fieldwise_init
struct MiniMaxH3ModCache(Movable):
    """Streamed-once AdaLN modulation cache — the RESULT that replaces
    24.288 GiB of `adaln_proj` weights (fp8_policy.mojo unit 14).

    `block_mod[layer]` is `[distinct_timesteps * 3, 6 * hidden_size]` BF16,
    addressed by `minimax_h3_adaln_row(timestep_index, token_tag)` — the SAME
    index the device block stack (minimax_h3_dit.mojo) already defines.

    `final_mod` is `[distinct_timesteps, 2 * hidden_size]` BF16, addressed by
    `timestep_index` ALONE (block_forward.mojo step 7 — the final norm has no
    per-modality tag)."""

    var block_mod: List[TArc]
    var final_mod: TArc
    var distinct_timesteps: Int

    def num_layers(self) -> Int:
        return len(self.block_mod)

    def modulation_row(
        self, layer: Int, timestep_index: Int, token_tag: Int, ctx: DeviceContext
    ) raises -> Tensor:
        """One block's `[1, 6*hidden_size]` modulation row for one packed-
        sequence row (shift_msa|scale_msa|gate_msa|shift_mlp|scale_mlp|gate_mlp,
        each hidden_size wide — the layout `_transformer_block` in
        block_forward.mojo consumes). A convenience for a future block-forward
        DEVICE consumer; building that consumer is NOT this module's job."""
        if layer < 0 or layer >= len(self.block_mod):
            raise Error("MiniMax-H3 modcache: block layer out of range")
        var row = minimax_h3_adaln_row(timestep_index, token_tag)
        return slice(self.block_mod[layer][], 0, row, 1, ctx)

    def final_modulation_row(
        self, timestep_index: Int, ctx: DeviceContext
    ) raises -> Tensor:
        """The final layer's `[1, 2*hidden_size]` shift/scale row, addressed
        by timestep alone (no modality tag)."""
        if timestep_index < 0 or timestep_index >= self.distinct_timesteps:
            raise Error(
                "MiniMax-H3 modcache: final timestep_index out of range"
            )
        return slice(self.final_mod[], 0, timestep_index, 1, ctx)

    def total_bytes(self) raises -> Int:
        """Measured (not arithmetic) cache size: the sum of every resident
        tensor's actual byte count, read back off the tensors this module
        built — the same quantity `fp8_policy.mojo`'s
        `MiniMaxH3Fp8Budget.resident_bytes` predicts ahead of time."""
        var t = self.final_mod[].nbytes()
        for i in range(len(self.block_mod)):
            t += self.block_mod[i][].nbytes()
        return t


def minimax_h3_build_modulation_cache(
    shards: ShardedSafeTensors,
    temb: Tensor,
    config: MiniMaxH3DiTConfig,
    ctx: DeviceContext,
) raises -> MiniMaxH3ModCache:
    """Stream every adaLN weight ONCE, project `temb` through it, drop the
    weight; repeat for `config.num_layers` blocks, then the final layer.
    See this file's header for the math and the activation-precision
    boundary.

    `temb` must be `[distinct_timesteps, config.time_embed_dim]` F32 — the
    caller's `time_embedder` output. F32 is REQUIRED, not merely accepted:
    the activation-precision trap only reproduces the reference if SiLU runs
    at temb's own F32 precision before the ONE rounding down to bf16: a
    caller that pre-cast temb to bf16 would silently apply a SECOND rounding
    here, biasing every block identically."""
    config.validate()
    var tshape = temb.shape()
    if len(tshape) != 2:
        raise Error(
            String("MiniMax-H3 modcache: temb must be rank 2 [distinct_timesteps, ")
            + String(config.time_embed_dim) + "], got rank " + String(len(tshape))
        )
    if tshape[1] != config.time_embed_dim:
        raise Error(
            String("MiniMax-H3 modcache: temb width ") + String(tshape[1])
            + " != config.time_embed_dim " + String(config.time_embed_dim)
        )
    if temb.dtype() != STDtype.F32:
        raise Error(
            "MiniMax-H3 modcache: temb must be F32 — SiLU runs at the"
            " timestep embedder's own precision (the ACTIVATION PRECISION"
            " TRAP; see this file's header); a pre-downcast temb would apply"
            " a second, unfaithful rounding here"
        )
    var distinct_timesteps = tshape[0]
    if distinct_timesteps <= 0:
        raise Error("MiniMax-H3 modcache: temb has zero rows")

    minimax_h3_check_modcache_weights(shards, config)

    var activated_f32 = silu(temb, ctx)
    var activated = cast_tensor(activated_f32, STDtype.BF16, ctx)

    var hidden = config.hidden_size
    var tags = config.adaln_rows_per_timestep()
    var block_width = 6 * hidden

    var block_mod = List[TArc]()
    for layer in range(config.num_layers):
        var w = Tensor.from_view(
            shards.tensor_view(minimax_h3_block_adaln_weight_key(layer)), ctx
        )
        var b = Tensor.from_view(
            shards.tensor_view(minimax_h3_block_adaln_bias_key(layer)), ctx
        )
        var wide = linear_bias(activated, w, b, ctx)
        var mod = reshape_owned(wide^, [distinct_timesteps * tags, block_width])
        block_mod.append(TArc(mod^))
        # `w`/`b` (and the pre-reshape `wide`) fall out of scope HERE — one
        # block's ~495.6 MiB adaln weight is freed before the next iteration
        # allocates the next one. Never more than one block's weight resident.

    var fw = Tensor.from_view(
        shards.tensor_view(MINIMAX_H3_FINAL_ADALN_WEIGHT_KEY), ctx
    )
    var fb = Tensor.from_view(
        shards.tensor_view(MINIMAX_H3_FINAL_ADALN_BIAS_KEY), ctx
    )
    var final_mod = linear_bias(activated, fw, fb, ctx)

    return MiniMaxH3ModCache(block_mod^, TArc(final_mod^), distinct_timesteps)
