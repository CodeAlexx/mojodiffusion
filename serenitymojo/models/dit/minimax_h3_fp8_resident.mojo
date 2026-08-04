# serenitymojo/models/dit/minimax_h3_fp8_resident.mojo
#
# MiniMax-H3 FP8-RESIDENT denoiser base — quantize each frozen block's big GEMM
# weights ONCE (bf16 -> E4M3 bytes + per-output-row F32 scale, on device), keep
# them resident, and rebuild a block's bf16 weight Dict per step by DEQUANT
# instead of re-streaming ~0.77 GiB/block from disk. The streaming loop re-reads
# the full ~36 GiB of block weights EVERY denoise step; this store replaces that
# with a ~18 GiB one-time residency + a cheap per-block dequant kernel.
#
# THE ARITHMETIC IS NOT THIS FILE'S: `models/minimax_h3/fp8_policy.mojo` (unit
# 14, host-gated) already classified all 638 planned keys and proved the fit —
# FP8_ROW (qkv/out_proj/fc1/fc2 of the 50 blocks, 35.90 GiB bf16 -> 17.956 GiB
# E4M3+scales) + BF16_KEEP small tensors + adaLN EVICTED to the modulation
# cache = 20.43 GiB resident worst case, 3.57 GiB spare on a 24 GiB card. This
# file is the runtime that policy describes: `minimax_h3_fp8_class` is called
# per block tensor and the store FAILS LOUD if classification ever disagrees
# with the two classes a block may contain (F32_KEEP/ADALN inside a block =
# policy drift, not something to paper over).
#
# PRECEDENT PORTED, NOT INVENTED: krea2's fp8-resident base
# (models/krea2/krea2_stack.mojo::build_krea2_resident_fp8 + models/dit/
# krea2_dit.mojo::_blk_w8) — 28 frozen blocks quantized once at launch,
# per-step dequant through the SAME parity-gated kernel pair this file uses:
#   ops/fp8_quant.mojo  fp8_e4m3_rowscale / fp8_e4m3_encode_perrow  (encode)
#   ops/fp8.mojo        fp8_e4m3_dequant_perrow_to_bf16             (decode)
# Same per-output-row symmetric-absmax E4M3 scheme (the quantized_loading.py
# convention the Ideogram-4 checkpoints already ship in). The dequant output is
# bf16 with the SAME shape/row-order the streamed path produces, so
# `minimax_h3_block_forward` is completely unchanged — it consumes the Dict this
# file returns exactly as it consumes `minimax_h3_load_block_device`'s.
#
# TRANSFORM ORDER MATTERS: quantization happens on the OUTPUT of
# `minimax_h3_load_block_device` — i.e. AFTER the qkv de-interleave and fc1
# half-swap (both gated max_abs 0.0 vs the f32 oracle). Dequant therefore
# reproduces the TRANSFORMED row order, which is why
# `minimax_h3_resident_block_weights` may honestly re-stamp the two transform
# markers `minimax_h3_require_transformed_weights` checks. Quantizing the raw
# shard bytes instead would hand block_forward untransformed rows with valid
# markers — the exact silent-scramble class the markers exist to catch.
#
# WHAT STAYS OUT, PER POLICY: adaln_proj (never loaded by the block loop at
# all — modcache owns it), the frontend / final layer / token_refiner
# (vendor-protected BF16_KEEP or F32_KEEP; loaded elsewhere, untouched here).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.collections import Dict, List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.fp8 import fp8_e4m3_dequant_perrow_to_bf16_into
from serenitymojo.ops.fp8_quant import fp8_e4m3_rowscale, fp8_e4m3_encode_perrow
from serenitymojo.models.minimax_h3.fp8_policy import (
    H3_FP8_BF16_KEEP,
    H3_FP8_ROW,
    minimax_h3_fp8_class,
)
from serenitymojo.models.dit.minimax_h3_dit import (
    MINIMAX_H3_FC1_SWAPPED_MARKER,
    MINIMAX_H3_QKV_DEINTERLEAVED_MARKER,
    MiniMaxH3DiTConfig,
    minimax_h3_block_prefix,
    minimax_h3_block_tensor_names,
    minimax_h3_check_block_weights,
    minimax_h3_expected_shape,
)
from serenitymojo.models.dit.minimax_h3_loader_device import (
    _minimax_h3_marker_tensor,
    minimax_h3_load_fc1_device,
    minimax_h3_load_qkv_device,
)


# ─────────────────────────────────────────────────────────────────────────────
# The resident store. Names are carried alongside the tensors (rather than a
# fixed comptime slot order) so the rebuild below cannot drift from whatever
# `minimax_h3_block_tensor_names` listed at build time — the classification is
# the policy's, the ordering is the loader's, and this struct hardcodes
# neither.
# ─────────────────────────────────────────────────────────────────────────────
struct MiniMaxH3BlockResidentFp8(Copyable, Movable):
    var fp8_names: List[String]          # FP8_ROW keys (qkv/out_proj/fc1/fc2)
    var fp8: List[ArcPointer[Tensor]]    # E4M3 bytes [out,in], TRANSFORMED rows
    var scale: List[ArcPointer[Tensor]]  # F32 [out] per-output-row scales
    var bf16_names: List[String]         # BF16_KEEP keys (the 1-D norm gains)
    var bf16: List[ArcPointer[Tensor]]   # bf16, kept resident verbatim

    def __init__(
        out self,
        var fp8_names: List[String],
        var fp8: List[ArcPointer[Tensor]],
        var scale: List[ArcPointer[Tensor]],
        var bf16_names: List[String],
        var bf16: List[ArcPointer[Tensor]],
    ):
        self.fp8_names = fp8_names^
        self.fp8 = fp8^
        self.scale = scale^
        self.bf16_names = bf16_names^
        self.bf16 = bf16^


struct MiniMaxH3ResidentFp8(Copyable, Movable):
    """`blocks[i]` holds ABSOLUTE layer `start_layer + i` — the same
    contiguous-range convention `minimax_h3_run_stack` already uses, so a
    store built for a partial checkpoint range indexes correctly.

    `scratch` is the dequant DESTINATION set: one preallocated bf16 tensor
    per fp8 slot (every block shares the same 4 shapes), written in place by
    `minimax_h3_resident_block_weights` and reused for all layers and all
    steps. This is NOT an optimization garnish — the first fp8-mode gate run
    OOM'd because per-layer allocating dequants raced ahead of the
    stream-ordered frees next to the 17.96 GiB store (see the gate's file
    header). With scratch, the block loop performs ZERO device allocations
    in steady state."""

    var blocks: List[MiniMaxH3BlockResidentFp8]
    var start_layer: Int
    var scratch: List[ArcPointer[Tensor]]

    def __init__(
        out self,
        var blocks: List[MiniMaxH3BlockResidentFp8],
        start_layer: Int,
        var scratch: List[ArcPointer[Tensor]],
    ):
        self.blocks = blocks^
        self.start_layer = start_layer
        self.scratch = scratch^

    def resident_bytes(self) raises -> Int:
        """Actual device bytes held by this store (E4M3 1B/param + F32 scales
        + bf16 keeps + the shared dequant scratch) — printed at build so the
        fit claim is a measurement, not fp8_policy's prediction restated."""
        var total = 0
        for bi in range(len(self.blocks)):
            ref b = self.blocks[bi]
            for k in range(len(b.fp8)):
                total += b.fp8[k][].numel()
                total += b.scale[k][].numel() * 4
            for k in range(len(b.bf16)):
                total += b.bf16[k][].numel() * 2
        for k in range(len(self.scratch)):
            total += self.scratch[k][].numel() * 2
        return total


def _check_tensor(
    t: Tensor, name: String, want: List[Int], layer: Int
) raises:
    """Per-tensor half of `minimax_h3_check_block_weights` (which needs the
    whole block Dict this build no longer materializes): exact shape against
    `minimax_h3_expected_shape` + bf16 storage, fail loud, named."""
    var got = t.shape()
    if len(got) != len(want):
        raise Error(
            String("MiniMax-H3 fp8 build: rank mismatch for ") + name
            + " (layer " + String(layer) + "), got " + String(len(got))
            + " dims, want " + String(len(want))
        )
    for d in range(len(want)):
        if got[d] != want[d]:
            raise Error(
                String("MiniMax-H3 fp8 build: shape mismatch for ") + name
                + " dim " + String(d) + ": got " + String(got[d])
                + ", want " + String(want[d])
            )
    if t.dtype() != STDtype.BF16:
        raise Error(
            String("MiniMax-H3 fp8 build: ") + name + " is not BF16 storage"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Build: stream and quantize ONE TENSOR AT A TIME, through the SAME
# per-tensor loaders the production streaming path uses (qkv de-interleave /
# fc1 half-swap included, so quantize still happens post-transform). The
# first version of this loop loaded a whole block Dict (~0.77 GiB bf16) and
# quantized from it; the pool retained those per-block transients as
# size-bucketed slack, and with the finished 18.67 GiB store on device the
# process sat at ~23 GiB reserved — the NEXT large allocation (the
# frontend weights) OOM'd (measured 2026-08-03, fp8-mode run 3). Per-tensor
# lifetimes bound the build's transient high-water to ONE tensor (~0.31 GiB
# max) instead of one block, and the per-tensor ctx.synchronize() retires
# the encode kernels before the bf16 source drops.
# ─────────────────────────────────────────────────────────────────────────────
def minimax_h3_build_resident_fp8(
    st: ShardedSafeTensors,
    config: MiniMaxH3DiTConfig,
    ctx: DeviceContext,
    num_layers: Int = -1,
    start_layer: Int = 0,
) raises -> MiniMaxH3ResidentFp8:
    var n = num_layers
    if n < 0:
        n = config.num_layers
    if start_layer < 0 or start_layer + n > config.num_layers:
        raise Error(
            String("minimax_h3_build_resident_fp8: layer range [")
            + String(start_layer) + ", " + String(start_layer + n)
            + ") exceeds config.num_layers " + String(config.num_layers)
        )

    var blocks = List[MiniMaxH3BlockResidentFp8]()
    for i in range(n):
        var layer = start_layer + i
        var qkv_name = minimax_h3_block_prefix(layer) + "attn.qkv_proj.weight"
        var fc1_name = minimax_h3_block_prefix(layer) + "mlp.fc1.weight"
        var names = minimax_h3_block_tensor_names(layer)

        var fp8_names = List[String]()
        var fp8 = List[ArcPointer[Tensor]]()
        var scale = List[ArcPointer[Tensor]]()
        var bf16_names = List[String]()
        var bf16 = List[ArcPointer[Tensor]]()
        for k in range(len(names)):
            ref name = names[k]
            var want = minimax_h3_expected_shape(name, config)
            var rows = want[0]
            var cols = want[1] if len(want) == 2 else 0
            var cls = minimax_h3_fp8_class(name, rows, cols)
            if cls == H3_FP8_ROW:
                # Load THIS tensor (transformed where the contract requires),
                # quantize, drain, drop the bf16 — one-tensor high-water.
                var t: Tensor
                if name == qkv_name:
                    t = minimax_h3_load_qkv_device(
                        st, name, config.num_attention_heads,
                        config.attention_head_dim, config.hidden_size, ctx,
                    )
                elif name == fc1_name:
                    t = minimax_h3_load_fc1_device(
                        st, name, config.ffn_hidden_size, config.hidden_size, ctx,
                    )
                else:
                    t = Tensor.from_view(st.tensor_view(name), ctx)
                _check_tensor(t, name, want, layer)
                var s = fp8_e4m3_rowscale(t, ctx)
                var q = fp8_e4m3_encode_perrow(t, s, ctx)
                # Encode must retire before `t` (the bf16 source) drops.
                ctx.synchronize()
                fp8_names.append(name.copy())
                fp8.append(ArcPointer(q^))
                scale.append(ArcPointer(s^))
            elif cls == H3_FP8_BF16_KEEP:
                var t2 = Tensor.from_view(st.tensor_view(name), ctx)
                _check_tensor(t2, name, want, layer)
                bf16_names.append(name.copy())
                bf16.append(ArcPointer(t2^))
            else:
                raise Error(
                    String("minimax_h3_build_resident_fp8: ") + name
                    + " classified as class " + String(cls)
                    + " — a block tensor must be FP8_ROW or BF16_KEEP;"
                    " F32_KEEP/ADALN here means fp8_policy and the block"
                    " tensor list have drifted apart"
                )
        blocks.append(
            MiniMaxH3BlockResidentFp8(
                fp8_names^, fp8^, scale^, bf16_names^, bf16^
            )
        )
        if (i + 1) % 10 == 0 or i + 1 == n:
            print(
                "  fp8-resident: quantized block", i + 1, "/", n,
                "(layer", layer, ")",
            )

    # Preallocate the shared dequant scratch (one bf16 tensor per fp8 slot;
    # every block has the same 4 shapes — block 0's are the template). No
    # fill needed: every byte is overwritten by the dequant kernel before any
    # consumer reads it.
    var scratch = List[ArcPointer[Tensor]]()
    if len(blocks) > 0:
        ref b0 = blocks[0]
        for k in range(len(b0.fp8)):
            var sh = b0.fp8[k][].shape()
            var buf = ctx.enqueue_create_buffer[DType.uint8](
                sh[0] * sh[1] * 2
            )
            var shape: List[Int] = [sh[0], sh[1]]
            scratch.append(ArcPointer(Tensor(buf^, shape^, STDtype.BF16)))
    return MiniMaxH3ResidentFp8(blocks^, start_layer, scratch^)


# ─────────────────────────────────────────────────────────────────────────────
# Per-step rebuild: the drop-in replacement for `minimax_h3_load_block_device`
# in the block loop. NO disk access, and in steady state NO device
# allocation either: the 4 big weights are dequanted IN PLACE into the
# store's shared scratch tensors, and the Dict carries ArcPointer references
# to them. The entry `ctx.synchronize()` is the scratch-reuse hazard fence:
# it drains everything previously enqueued (in particular the PREVIOUS
# block's forward, which reads the same scratch) before the overwrite. The
# streamed loader is implicitly synchronous per block (host-staged H2D), so
# this adds no ordering the other path didn't already have.
# ─────────────────────────────────────────────────────────────────────────────
def minimax_h3_resident_block_weights(
    resident: MiniMaxH3ResidentFp8,
    layer: Int,
    config: MiniMaxH3DiTConfig,
    ctx: DeviceContext,
) raises -> Dict[String, ArcPointer[Tensor]]:
    var i = layer - resident.start_layer
    if i < 0 or i >= len(resident.blocks):
        raise Error(
            String("minimax_h3_resident_block_weights: layer ") + String(layer)
            + " outside the resident range [" + String(resident.start_layer)
            + ", " + String(resident.start_layer + len(resident.blocks))
            + ") — build the store over the layers you run, or stream"
        )
    ref b = resident.blocks[i]
    if len(resident.scratch) != len(b.fp8):
        raise Error(
            "minimax_h3_resident_block_weights: scratch/slot count mismatch —"
            " the store was not built by minimax_h3_build_resident_fp8"
        )
    # Drain: previously enqueued consumers of `scratch` must finish before
    # this block's dequant overwrites it.
    ctx.synchronize()
    var weights = Dict[String, ArcPointer[Tensor]]()
    for k in range(len(b.fp8)):
        fp8_e4m3_dequant_perrow_to_bf16_into(
            b.fp8[k][], b.scale[k][], resident.scratch[k][], ctx
        )
        weights[b.fp8_names[k]] = resident.scratch[k].copy()
    for k in range(len(b.bf16)):
        weights[b.bf16_names[k]] = b.bf16[k].copy()
    # Same preflight the streamed loader runs (shape + bf16 dtype, fail loud).
    minimax_h3_check_block_weights(weights, layer, config)
    # Honest re-stamp: the fp8 bytes were encoded from ALREADY-TRANSFORMED
    # tensors (minimax_h3_load_block_device output), and dequant is
    # elementwise — row order is preserved through the round trip. See the
    # file header's TRANSFORM ORDER note.
    weights[MINIMAX_H3_QKV_DEINTERLEAVED_MARKER] = ArcPointer(
        _minimax_h3_marker_tensor(ctx)
    )
    weights[MINIMAX_H3_FC1_SWAPPED_MARKER] = ArcPointer(
        _minimax_h3_marker_tensor(ctx)
    )
    return weights^
