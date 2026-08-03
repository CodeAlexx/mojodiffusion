# serenitymojo/models/dit/minimax_h3_loader_device.mojo
#
# MiniMax-H3 transformer — DEVICE STREAMING WEIGHT LOADER.
#
# WHY STREAMING: the released FL2VA/Ref2VA transformer is 61.73 GiB bf16 across
# 13 shards; the target GPU is a 24 GiB RTX 3090 Ti. This loader hands back one
# block's weights at a time as a `Dict[String, ArcPointer[Tensor]]`; the caller
# drops that Dict before loading the next layer — exactly the per-layer
# load/release loop `models/dit/ideogram4_dit.mojo`'s
# `ideogram4_forward_prefinal_hidden` already runs (local `var` weights inside
# the `for li in range(num_layers)` loop, freed at each iteration's scope
# exit). One block here is ~0.77 GiB bf16 (qkv 231MB + out_proj 77MB + fc1
# 308MB + fc2 154MB + negligible norms), so streaming keeps the transformer's
# resident footprint at one block instead of 50.
#
# adaln_proj IS NOT LOADED HERE — not by this file, not by anyone calling it.
# It is [96768, 2688] per block, 13.04 B params / 24.2 GiB over the whole
# stack, and its only input is the timestep embedding, so the vendor's own
# README says it "can be precomputed and cached; these parameters do not need
# to be loaded for inference-only deployment." `minimax_h3_block_tensor_names`
# (models/dit/minimax_h3_dit.mojo) already excludes it for exactly this
# reason — pulling it in here would move 24 GiB per step for nothing. See that
# file's header and `models/minimax_h3/fp8_policy.mojo` for the residency
# arithmetic.
#
# THE TWO FUSED-TENSOR REWRITES are mirrored — not reimplemented from
# scratch — from the host-f32 ORACLE `models/minimax_h3/loader.mojo`, which is
# itself gated at max_abs 0.0 against torch's own tensors
# (`minimax_h3_loader_parity.mojo`, checks [3] and [4]):
#   * `attn.qkv_proj.weight` [3*inner, hidden] is PER-HEAD INTERLEAVED
#     q/k/v in the raw shard; `minimax_h3_deinterleave_qkv_bf16` below applies
#     the IDENTICAL row-index math as the oracle's `minimax_h3_deinterleave_qkv`
#     to relocate it to `[q_all; k_all; v_all]`, same shape.
#   * `mlp.fc1.weight` [2*ffn, hidden] stores `[gate; value]`;
#     `minimax_h3_swap_fc1_bf16` applies the IDENTICAL index math as
#     `minimax_h3_swap_fc1` to produce `[value; gate]`, same shape.
# Both rewrites are pure row relocation — no arithmetic — so they are applied
# directly to bf16 elements with NO f32 detour: casting f32->bf16 commutes
# with a permutation, so operating in bf16 is exact relative to the f32
# oracle's own output cast down. `minimax_h3_loader_device_selfcheck` proves
# this equivalence on synthetic data (host-only, no GPU, no checkpoint
# needed) by comparing this file's bf16 output against the f32 oracle's
# output cast to bf16, element for element.
#
# DTYPE: every block tensor is bf16 storage (the checkpoint's 12 fp32 tensors
# are all OUTSIDE the block stack — patch projections, time embedder, output
# heads — see `models/minimax_h3/loader.mojo`'s `minimax_h3_is_fp32_key`).
# Nothing in this file upcasts to f32: the six unrewritten tensors go through
# `Tensor.from_view` (verbatim bf16 byte copy), and the two rewritten tensors
# stage through a host `List[BFloat16]` and upload via `Tensor.from_host_bf16`
# (verbatim bf16, no cast). Upcasting here would violate the dtype rule and
# instantly blow the 24 GiB budget at this size.
#
# FOUNDATION REUSED (not reimplemented):
#   `io/sharded.mojo`            ShardedSafeTensors.open/tensor_view
#   `io/tensor_view.mojo`        TensorView (dtype + shape + origin-bound Span)
#   `io/dtype.mojo`              STDtype
#   `tensor.mojo`                Tensor.from_view / Tensor.from_host_bf16
#   `models/dit/minimax_h3_dit.mojo`
#       MiniMaxH3DiTConfig, minimax_h3_released_config,
#       minimax_h3_block_prefix, minimax_h3_block_tensor_names,
#       minimax_h3_check_block_weights
#   `models/minimax_h3/loader.mojo`
#       minimax_h3_deinterleave_qkv, minimax_h3_swap_fc1  (f32 oracle, used
#       only by this file's host-only selfcheck — never on the device path)
#
# All file I/O goes through `io/ffi` (`sys_open`/`sys_pread`) via
# `ShardedSafeTensors`/`SafeTensors` — this file does no raw file I/O of its
# own; it only reads already-mmap'd `TensorView` bytes.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.collections import Dict, List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import TensorView
from serenitymojo.tensor import Tensor
from serenitymojo.models.dit.minimax_h3_dit import (
    MiniMaxH3DiTConfig,
    MINIMAX_H3_QKV_DEINTERLEAVED_MARKER,
    MINIMAX_H3_FC1_SWAPPED_MARKER,
    minimax_h3_block_prefix,
    minimax_h3_block_tensor_names,
    minimax_h3_check_block_weights,
    minimax_h3_released_config,
)
from serenitymojo.models.minimax_h3.loader import (
    minimax_h3_deinterleave_qkv,
    minimax_h3_swap_fc1,
)


comptime MINIMAX_H3_TRANSFORMER_DIR = (
    "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/transformer"
)


# ─────────────────────────────────────────────────────────────────────────────
# BF16 analogues of the two fused-tensor rewrites.
#
# Each mirrors its `models/minimax_h3/loader.mojo` counterpart line for line —
# same loop nesting, same index formulas — with `List[Float32]` replaced by
# `List[BFloat16]`. Any future change to the oracle's index math must be
# mirrored here too; `minimax_h3_loader_device_selfcheck` is the gate that
# would catch a drift between the two.
# ─────────────────────────────────────────────────────────────────────────────


def minimax_h3_deinterleave_qkv_bf16(
    fused: List[BFloat16], heads: Int, head_dim: Int, in_features: Int
) raises -> List[BFloat16]:
    """BF16 mirror of `minimax_h3_deinterleave_qkv`: per-head-interleaved fused
    QKV -> `[q_all; k_all; v_all]`, same shape, values only relocated."""
    var expected = heads * 3 * head_dim
    if len(fused) != expected * in_features:
        raise Error(
            "MiniMax-H3 device loader: fused qkv has the wrong row count for"
            " this head configuration"
        )

    var out = List[BFloat16](capacity=len(fused))
    for _ in range(len(fused)):
        out.append(BFloat16(0.0))
    for part in range(3):
        for h in range(heads):
            for d in range(head_dim):
                var source_row = h * 3 * head_dim + part * head_dim + d
                var dest_row = part * heads * head_dim + h * head_dim + d
                for i in range(in_features):
                    out[dest_row * in_features + i] = fused[source_row * in_features + i]
    return out^


def minimax_h3_swap_fc1_bf16(
    fc1: List[BFloat16], ffn_dim: Int, in_features: Int
) raises -> List[BFloat16]:
    """BF16 mirror of `minimax_h3_swap_fc1`: `[gate; value]` -> `[value; gate]`,
    same shape, values only relocated."""
    if len(fc1) != 2 * ffn_dim * in_features:
        raise Error(
            "MiniMax-H3 device loader: fc1 has the wrong shape for this ffn_dim"
        )
    var out = List[BFloat16](capacity=len(fc1))
    for row in range(ffn_dim):
        var source = (ffn_dim + row) * in_features
        for i in range(in_features):
            out.append(fc1[source + i])
    for row in range(ffn_dim):
        var source = row * in_features
        for i in range(in_features):
            out.append(fc1[source + i])
    return out^


# ─────────────────────────────────────────────────────────────────────────────
# Host-side raw bf16 readback (no device round trip). Staging step for the two
# rewrites above: they must permute rows BEFORE upload, so the raw bytes are
# read into a host list first instead of going through `Tensor.from_view`.
# ─────────────────────────────────────────────────────────────────────────────


def _tensor_view_bf16_host[
    mut: Bool, //, origin: Origin[mut=mut]
](tv: TensorView[origin]) raises -> List[BFloat16]:
    """Read a BF16 `TensorView`'s raw mmap'd bytes into a host `List[BFloat16]`,
    verbatim (no cast — the view is already bf16 on disk)."""
    if tv.dtype != STDtype.BF16:
        raise Error(
            String("MiniMax-H3 device loader: expected BF16, got ")
            + tv.dtype.name()
        )
    var n = tv.numel()
    var expect = n * STDtype.BF16.byte_size()
    if tv.nbytes() != expect:
        raise Error("MiniMax-H3 device loader: bf16 view nbytes mismatch")
    var p = tv.data.unsafe_ptr().bitcast[BFloat16]()
    var out = List[BFloat16](capacity=n)
    for i in range(n):
        out.append(p[i])
    return out^


# ─────────────────────────────────────────────────────────────────────────────
# Per-tensor device loaders for the two rewritten tensors. The six unrewritten
# block tensors need no dedicated function — `minimax_h3_load_block_device`
# calls `Tensor.from_view` on them directly, same as `ideogram4_dit.mojo`'s
# `load_w_bf16`.
# ─────────────────────────────────────────────────────────────────────────────


def minimax_h3_load_qkv_device(
    st: ShardedSafeTensors,
    name: String,
    heads: Int,
    head_dim: Int,
    in_features: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Load one block's `attn.qkv_proj.weight` and de-interleave it in BF16.
    Shape is unchanged ([3*inner, hidden]); only row order changes."""
    var tv = st.tensor_view(name)
    var shape = tv.shape.copy()
    var raw = _tensor_view_bf16_host(tv)
    var reordered = minimax_h3_deinterleave_qkv_bf16(raw, heads, head_dim, in_features)
    return Tensor.from_host_bf16(reordered^, shape^, ctx)


def minimax_h3_load_fc1_device(
    st: ShardedSafeTensors,
    name: String,
    ffn_dim: Int,
    in_features: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Load one block's `mlp.fc1.weight` and swap [gate;value]->[value;gate] in
    BF16. Shape is unchanged ([2*ffn, hidden]); only row order changes."""
    var tv = st.tensor_view(name)
    var shape = tv.shape.copy()
    var raw = _tensor_view_bf16_host(tv)
    var swapped = minimax_h3_swap_fc1_bf16(raw, ffn_dim, in_features)
    return Tensor.from_host_bf16(swapped^, shape^, ctx)


# ─────────────────────────────────────────────────────────────────────────────
# Per-block streaming loader — the deliverable. Loads exactly the 8 tensors
# `minimax_h3_block_tensor_names` lists for one layer (never adaln_proj),
# applies the two rewrites above where needed, preflights the result with
# `minimax_h3_check_block_weights` (shape + bf16-dtype), and STAMPS two
# marker keys so `minimax_h3_block_forward`'s guard
# (`minimax_h3_require_transformed_weights`, minimax_h3_dit.mojo) can tell
# these tensors apart from raw `ShardedSafeTensors`/`BlockLoader` output —
# shape alone can't, since a row permutation never changes it. The caller is
# responsible for letting the returned Dict drop before loading the next
# layer, so only one block's device memory is resident at a time.
# ─────────────────────────────────────────────────────────────────────────────


def _minimax_h3_marker_tensor(ctx: DeviceContext) raises -> Tensor:
    """Presence-only sentinel: `minimax_h3_require_transformed_weights` only
    checks the DICT KEY, never reads this tensor's bytes, so there is nothing
    to initialize — one 2-byte device allocation, no fill kernel, no host
    round trip. Negligible next to the ~0.77 GiB this stamps."""
    var buf = ctx.enqueue_create_buffer[DType.uint8](2)
    var shape: List[Int] = [1]
    return Tensor(buf^, shape^, STDtype.BF16)


def minimax_h3_load_block_device(
    st: ShardedSafeTensors,
    layer: Int,
    config: MiniMaxH3DiTConfig,
    ctx: DeviceContext,
) raises -> Dict[String, ArcPointer[Tensor]]:
    var qkv_name = minimax_h3_block_prefix(layer) + "attn.qkv_proj.weight"
    var fc1_name = minimax_h3_block_prefix(layer) + "mlp.fc1.weight"
    var heads = config.num_attention_heads
    var head_dim = config.attention_head_dim
    var hidden = config.hidden_size
    var ffn = config.ffn_hidden_size

    var weights = Dict[String, ArcPointer[Tensor]]()
    var names = minimax_h3_block_tensor_names(layer)
    for i in range(len(names)):
        ref name = names[i]
        if name == qkv_name:
            weights[name] = ArcPointer(
                minimax_h3_load_qkv_device(st, name, heads, head_dim, hidden, ctx)
            )
        elif name == fc1_name:
            weights[name] = ArcPointer(
                minimax_h3_load_fc1_device(st, name, ffn, hidden, ctx)
            )
        else:
            weights[name] = ArcPointer(Tensor.from_view(st.tensor_view(name), ctx))

    minimax_h3_check_block_weights(weights, layer, config)
    weights[MINIMAX_H3_QKV_DEINTERLEAVED_MARKER] = ArcPointer(_minimax_h3_marker_tensor(ctx))
    weights[MINIMAX_H3_FC1_SWAPPED_MARKER] = ArcPointer(_minimax_h3_marker_tensor(ctx))
    return weights^


# ─────────────────────────────────────────────────────────────────────────────
# Host-only self-check (no GPU, no checkpoint): proves the two BF16 rewrites
# above compute the IDENTICAL permutation as the gated f32 oracle. A
# permutation only relocates values, so f32->bf16 casting commutes with it:
# reordering an f32 buffer then casting must equal casting then reordering the
# already-bf16 buffer. This does not touch real MiniMax-H3 bytes (none exist
# locally yet — see `main` below); it pins this file's index math to the
# oracle's independent of any checkpoint being present.
# ─────────────────────────────────────────────────────────────────────────────


def minimax_h3_loader_device_selfcheck() raises:
    var heads = 2
    var head_dim = 3
    var hidden = 4
    var ffn = 5

    # --- qkv de-interleave ---
    var qkv_rows = heads * 3 * head_dim
    var fused_f32 = List[Float32]()
    var fused_bf16 = List[BFloat16]()
    for r in range(qkv_rows):
        for c in range(hidden):
            var v = Float32(r * hidden + c) * Float32(0.01) - Float32(3.0)
            fused_f32.append(v)
            fused_bf16.append(v.cast[DType.bfloat16]())

    var want_qkv = minimax_h3_deinterleave_qkv(fused_f32, heads, head_dim, hidden)
    var got_qkv = minimax_h3_deinterleave_qkv_bf16(fused_bf16, heads, head_dim, hidden)
    if len(want_qkv) != len(got_qkv):
        raise Error("selfcheck: qkv length mismatch")
    for i in range(len(want_qkv)):
        if got_qkv[i] != want_qkv[i].cast[DType.bfloat16]():
            raise Error(
                String("selfcheck: qkv bf16 rewrite diverges from the f32 oracle at ")
                + String(i)
            )
    print("  ok   qkv de-interleave: bf16 rewrite matches f32 oracle (", len(want_qkv), "values)")

    # --- fc1 half-swap ---
    var fc1_rows = 2 * ffn
    var fc1_f32 = List[Float32]()
    var fc1_bf16 = List[BFloat16]()
    for r in range(fc1_rows):
        for c in range(hidden):
            var v = Float32(r * hidden + c) * Float32(0.02) + Float32(1.0)
            fc1_f32.append(v)
            fc1_bf16.append(v.cast[DType.bfloat16]())

    var want_fc1 = minimax_h3_swap_fc1(fc1_f32, ffn, hidden)
    var got_fc1 = minimax_h3_swap_fc1_bf16(fc1_bf16, ffn, hidden)
    if len(want_fc1) != len(got_fc1):
        raise Error("selfcheck: fc1 length mismatch")
    for i in range(len(want_fc1)):
        if got_fc1[i] != want_fc1[i].cast[DType.bfloat16]():
            raise Error(
                String("selfcheck: fc1 bf16 rewrite diverges from the f32 oracle at ")
                + String(i)
            )
    print("  ok   fc1 half-swap: bf16 rewrite matches f32 oracle (", len(want_fc1), "values)")


# ─────────────────────────────────────────────────────────────────────────────
# Probe: self-check (always runs, no GPU/checkpoint needed) + an attempt to
# open the real shard directory and stream two real blocks. The download may
# still be in progress — an incomplete/absent shard set is reported as a clear
# message and is a VALID outcome, not a crash or a hang (every byte this file
# reads goes through `ShardedSafeTensors.open`/`SafeTensors.open`, which fail
# fast with a named error on a missing or truncated file — see
# `io/safetensors.mojo` and `io/sharded.mojo`; this probe does not add its own
# retry/poll loop that could hang).
# ─────────────────────────────────────────────────────────────────────────────


def main() raises:
    print("MiniMax-H3 device streaming loader probe")

    print("")
    print("[1] host-only self-check: bf16 rewrites vs the gated f32 oracle")
    minimax_h3_loader_device_selfcheck()

    print("")
    print("[2] released config")
    var config = minimax_h3_released_config()
    config.validate()
    print(
        "  hidden_size=", config.hidden_size,
        " num_layers=", config.num_layers,
        " inner_dim=", config.inner_dim(),
        " ffn_hidden_size=", config.ffn_hidden_size,
    )

    print("")
    print("[3] shard directory:", String(MINIMAX_H3_TRANSFORMER_DIR))
    try:
        var st = ShardedSafeTensors.open(String(MINIMAX_H3_TRANSFORMER_DIR))
        print("  ready:", st.num_shards(), "shard(s),", st.num_tensors(), "tensors")

        print("")
        print("[4] streaming blocks 0 and 1 onto the GPU (released between layers)")
        var ctx = DeviceContext()
        for layer in range(2):
            var weights = minimax_h3_load_block_device(st, layer, config, ctx)
            print(
                "  layer", layer, "loaded", len(weights),
                "tensors, preflight (shape+dtype) passed",
            )
            # `weights` drops at the end of this iteration -> device buffers
            # freed before the next layer's load, bounding residency to one
            # block (~0.77 GiB bf16) rather than the full 61.73 GiB stack.
        ctx.synchronize()
    except e:
        print("  shards not ready — this is a VALID outcome while the")
        print("  download is in progress, not a bug. Error:")
        print("   ", e)

    print("")
    print("DONE")
