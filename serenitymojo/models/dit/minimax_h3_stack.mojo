# serenitymojo/models/dit/minimax_h3_stack.mojo
#
# MiniMax-H3 DEVICE 50-LAYER STREAMED STACK — composes the pieces that are
# each already built and verified into the actual denoise-loop core. This
# file adds NO new math: it is a loop that loads one block's weights, runs
# the gated block forward, drops the weights, and repeats.
#
# REUSED, NOT REIMPLEMENTED:
#   models/dit/minimax_h3_loader_device.mojo   minimax_h3_load_block_device
#       — per-layer streaming load. max_abs 0.0 vs the f32 oracle on real H3
#         bytes (minimax_h3_loader_device_skeptic_probe.mojo). Excludes
#         adaln_proj by construction (minimax_h3_block_tensor_names never
#         lists it) — this file does not special-case that; it inherits it.
#         Also now STAMPS the transform-contract markers, so every block this
#         stack runs is guarded by minimax_h3_require_transformed_weights
#         with no extra code here.
#   models/dit/minimax_h3_dit.mojo             minimax_h3_block_forward
#       — GATED cos 0.9999968778871595 vs the host-f32 oracle
#         (models/dit/parity/minimax_h3_block_device_gate.mojo) at the tiny
#         fixture's geometry; comptime Heads/HeadDim let this stack specialize
#         at whatever geometry `config` describes (56/128 at release).
#   models/dit/minimax_h3_modcache.mojo        MiniMaxH3ModCache
#       — per-block AdaLN modulation, precomputed ONCE before this loop runs
#         (NOT per-layer here) so adaln_proj (13.04B params / 24.2 GiB) is
#         never touched by this file, directly or indirectly.
#   models/dit/minimax_h3_rope.mojo            build_minimax_h3_rope_tables
#       — the caller's job, upstream of this file; this file just forwards
#         cos/sin/rotary_dim into every block call unchanged (rope is shared
#         across all 50 blocks, per minimax_h3_dit.mojo's own header).
#   models/dit/minimax_h3_frontend.mojo        (upstream/downstream, not
#       called here) `minimax_h3_frontend_embed` produces this file's `hidden`
#       input ([S, hidden] bf16); `minimax_h3_final_layer` consumes this
#       file's output. Shape contract matches both ends: this file accepts
#       and returns [S, hidden] bf16, reshaping to/from the [1, S, hidden]
#       `minimax_h3_block_forward` needs internally.
#
# MEMORY IS THE WHOLE POINT. The released checkpoint is 61.73 GiB bf16; the
# target card is 24 GiB. Streaming keeps exactly ONE block's weights resident
# at a time: `weights` below is a loop-local `var` `Dict[String,
# ArcPointer[Tensor]]`, and Mojo drops (frees) it at the end of each loop
# iteration BEFORE the next `minimax_h3_load_block_device` call allocates the
# next block — the same discipline `models/dit/ideogram4_dit.mojo`'s
# `ideogram4_forward_prefinal_hidden` already uses for its own per-layer loop.
# A skeptic measured the real per-block resident delta at ~1.2-1.3 GiB (NOT
# the earlier-assumed 0.77 GiB) — see this file's own probe/gate for a
# flat-vs-accumulating delta measurement, not an assumption.
#
# MISSING LAYER FAILS LOUD, NAMED, NO HANG: `minimax_h3_load_block_device`
# already fails loud on a missing/truncated shard (`ShardedSafeTensors.open`
# / `tensor_view` — verified clean-raise, no hang, by
# `minimax_h3_loader_device_skeptic_probe.mojo` against 6 corruption/missing
# shapes). This file adds ONE thing on top: it names WHICH LAYER failed,
# since a bare safetensors error two calls deep does not say that on its own.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.collections import Dict, List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.tensor_algebra import reshape_owned
from serenitymojo.models.dit.minimax_h3_dit import (
    MiniMaxH3DiTConfig,
    MINIMAX_H3_HEADS,
    MINIMAX_H3_HEAD_DIM,
    minimax_h3_block_forward,
)
from serenitymojo.models.dit.minimax_h3_loader_device import minimax_h3_load_block_device
from serenitymojo.models.dit.minimax_h3_fp8_resident import (
    MiniMaxH3ResidentFp8,
    minimax_h3_resident_block_weights,
)
from serenitymojo.models.dit.minimax_h3_modcache import MiniMaxH3ModCache


def minimax_h3_run_stack[
    S: Int, Heads: Int = MINIMAX_H3_HEADS, HeadDim: Int = MINIMAX_H3_HEAD_DIM
](
    var hidden: Tensor,
    st: ShardedSafeTensors,
    modcache: MiniMaxH3ModCache,
    adaln_indices: List[Int],
    cos: Tensor,
    sin: Tensor,
    rotary_dim: Int,
    config: MiniMaxH3DiTConfig,
    ctx: DeviceContext,
    num_layers: Int = -1,
    start_layer: Int = 0,
    resident: Optional[MiniMaxH3ResidentFp8] = Optional[MiniMaxH3ResidentFp8](None),
) raises -> Tensor:
    """Stream and run `num_layers` denoiser blocks starting at `start_layer`
    (defaults: `start_layer=0`, `num_layers=config.num_layers`, i.e. all 50
    from the top at production). Both are separate overrides — not just
    `config.num_layers` — so this SAME function runs whatever CONTIGUOUS
    range of the partial checkpoint is actually downloaded today (e.g.
    blocks 7-33) without a second copy of the loop; production callers leave
    both at their defaults.

    `start_layer` shifts the ABSOLUTE layer number used for weight lookup
    (`minimax_h3_load_block_device`/`minimax_h3_block_forward`'s `layer`
    param — this must be the real checkpoint layer, e.g. 7, so the weight
    key prefix `blocks.7.` is correct) but NOT the modcache index (`modcache
    .block_mod[i]` stays 0-based over the `num_layers`-long RANGE being run
    — a caller testing a partial range builds a modcache sized for exactly
    that range, indexed from 0, not from `start_layer`).

    hidden:        [S, hidden_size] bf16 — `minimax_h3_frontend_embed`'s
                   output. Reshaped to [1, S, hidden_size] internally (what
                   `minimax_h3_block_forward` takes) and back on return, so
                   both neighbors in the pipeline see the shape their own
                   docstrings already promise.
    st:            open `ShardedSafeTensors` over the checkpoint directory.
    modcache:      precomputed AdaLN cache — `modcache.block_mod[layer]` is
                   handed to `minimax_h3_block_forward` as `mod`, unchanged.
                   `modcache.final_mod` is NOT read here (that's the final
                   layer's input, a separate unit — see file header).
    adaln_indices: length-S row -> `mod` row map, shared by every block
                   (same table for every layer — the row LAYOUT is per-layer
                   via `modcache.block_mod[layer]`, but which physical ROW a
                   given sequence position reads is a property of the packed
                   sequence, not the layer).
    cos/sin/rotary_dim: rope table, shared by every block unchanged.
    resident:      OPTIONAL fp8-resident store (models/dit/
                   minimax_h3_fp8_resident.mojo). When present, each block's
                   weights come from an on-device E4M3 dequant instead of a
                   per-step disk stream — same Dict contract, same
                   block_forward, ~0.99+ cos vs the streamed base (the
                   documented krea2-precedent tradeoff). When absent
                   (default), this function is byte-for-byte the streaming
                   loop it always was.
    """
    var n = num_layers
    if n < 0:
        n = config.num_layers
    if start_layer < 0:
        raise Error("minimax_h3_run_stack: start_layer must be >= 0")
    if start_layer + n > config.num_layers:
        raise Error(
            String("minimax_h3_run_stack: start_layer+num_layers ")
            + String(start_layer + n) + " exceeds config.num_layers "
            + String(config.num_layers)
        )
    if n > modcache.num_layers():
        raise Error(
            String("minimax_h3_run_stack: num_layers ") + String(n)
            + " exceeds modcache.num_layers() " + String(modcache.num_layers())
            + " — the modulation cache was not built for this many blocks"
        )

    var h3_shape: List[Int] = [1, S, config.hidden_size]
    var h = reshape_owned(hidden^, h3_shape^)

    for i in range(n):
        var layer = start_layer + i
        var weights: Dict[String, ArcPointer[Tensor]]
        try:
            if resident:
                weights = minimax_h3_resident_block_weights(
                    resident.value(), layer, config, ctx
                )
            else:
                weights = minimax_h3_load_block_device(st, layer, config, ctx)
        except e:
            raise Error(
                String("minimax_h3_run_stack: layer ") + String(layer)
                + " failed to load: " + String(e)
            )
        h = minimax_h3_block_forward[S, Heads, HeadDim](
            h, weights, layer, config, modcache.block_mod[i][],
            adaln_indices, cos, sin, rotary_dim, ctx,
        )
        # `weights` (this block's ~1.2-1.3 GiB) drops HERE, at loop-scope
        # exit, before the next iteration's load — the entire memory point
        # of streaming rather than resident-loading all 50 blocks.

    var out_shape: List[Int] = [S, config.hidden_size]
    return reshape_owned(h^, out_shape^)
