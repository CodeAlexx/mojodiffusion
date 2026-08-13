# serenitymojo/models/dit/minimax_h3_fp8_resident.mojo
#
# MiniMax-H3 quantized-RESIDENT denoiser base — quantize each frozen block's big
# GEMM weights ONCE (bf16 -> one-byte values + scales, on device), keep
# them resident, and rebuild a block's bf16 weight Dict per step by DEQUANT
# instead of re-streaming ~0.77 GiB/block from disk. The streaming loop re-reads
# the full ~36 GiB of block weights EVERY denoise step; this store replaces that
# with a ~20 GiB one-time residency + a cheap per-block dequant kernel.
#
# SCHEME (2026-08-04): the store defaults to INT8-WEIGHT-ONLY GROUPWISE
# (QKV=16, out_proj=64, fc1=32, fc2=64 input columns/group; symmetric absmax;
# persistent FP16 scales). Per-row INT8 produced block/video PASS but audio
# 0.9965372. F32 group scales improved audio to 0.9986431 at QKV32/others64,
# but QKV16 OOMed. Compact FP16 scales made QKV16 viable; the measured class
# ladder then selected FC1=32. The final fitting profile is block 0.99989784,
# video 0.99977180, audio 0.99895621: closest measured audio, but still an
# explicit FAIL vs the unchanged 0.999 bar. Finer FC1=16 and F32 QKV16 scales
# both cross allocator cliffs and OOM (24,033/23,870 MiB sampled). Groupwise
# scales preserve the bar rather than negotiating it. This follows Alex's
# earlier decision on
# this lane's measured E4M3 evidence: blockCos 0.998347 / e2eVideoCos
# 0.997255 / e2eAudioCos 0.988741 vs the 0.999 bar (the E4M3 3-bit-mantissa
# floor), while int8-weight-only per-row is the vendor's own consumer recipe
# for this model (fp8_policy header) with ~4x finer rounding at the SAME
# residency cost class. The final profile measures 19.952884 GiB of store and
# 22,026 MiB sampled peak; scale memory is material and is live-VRAM gated.
# Encode uses
# ops/int8_quant.mojo int8_groupscale/int8_encode_groupwise and rebuild uses
# int8_dequant_groupwise_to_bf16_into; their shared smoke is bit-exact against
# host decode. E4M3
# stays available behind the `scheme` argument (MINIMAX_H3_RESIDENT_E4M3)
# for A/B comparison. NAMING: "fp8" in this file's public names is HISTORICAL
# — kept to avoid churning the stack/t2va/gate call sites — read it as
# "1-byte quantized-resident"; the store's `scheme` field says which
# encoding, and the build prints it.
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
# bf16 with the SAME shape/row-order the streamed path produces, so the block's
# weight-consumption path is unchanged — it consumes the Dict this file returns
# exactly as it consumes `minimax_h3_load_block_device`'s.
#
# TRANSFORM ORDER MATTERS: quantization happens AFTER the qkv de-interleave
# and fc1 half-swap — the build stages every big tensor through the SAME
# gated bf16 rewrite functions the streaming loader uses
# (`minimax_h3_deinterleave_qkv_bf16` / `minimax_h3_swap_fc1_bf16`,
# loader_device.mojo, selfcheck-pinned to the f32 oracle). Dequant therefore
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
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.fp8 import fp8_e4m3_dequant_perrow_to_bf16_into
from serenitymojo.ops.fp8_quant import fp8_e4m3_rowscale, fp8_e4m3_encode_perrow
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.int8_quant import (
    int8_dequant_groupwise_to_bf16_into,
    int8_encode_groupwise,
    int8_encode_perrow,
    int8_groupscale,
    int8_rowscale,
)
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
    _tensor_view_bf16_host,
    minimax_h3_deinterleave_qkv_bf16,
    minimax_h3_swap_fc1_bf16,
)


# ─────────────────────────────────────────────────────────────────────────────
# Quantization scheme selectors (the store's ONE variable design point — see
# the file header's SCHEME note). Both are 1 byte/param, but INT8 uses
# class-selected group FP16 scales while E4M3 keeps its original F32 per-row
# scales.
# ─────────────────────────────────────────────────────────────────────────────
comptime MINIMAX_H3_RESIDENT_E4M3 = 0  # fp8 E4M3 per-row (the original lane)
comptime MINIMAX_H3_RESIDENT_INT8 = 1  # int8 weight-only groupwise (DEFAULT)
comptime MINIMAX_H3_RESIDENT_INT8_W8A8 = 2  # per-row store, direct INT8 GEMM
comptime MINIMAX_H3_INT8_QKV_GROUP_SIZE = 16
comptime MINIMAX_H3_INT8_OUT_GROUP_SIZE = 64
comptime MINIMAX_H3_INT8_FC1_GROUP_SIZE = 32
comptime MINIMAX_H3_INT8_FC2_GROUP_SIZE = 64


def minimax_h3_int8_group_size(name: String) raises -> Int:
    """Measured resident profile: class-specific scale-memory budget."""
    if name.endswith("attn.qkv_proj.weight"):
        return MINIMAX_H3_INT8_QKV_GROUP_SIZE
    if name.endswith("attn.out_proj.weight"):
        return MINIMAX_H3_INT8_OUT_GROUP_SIZE
    if name.endswith("mlp.fc1.weight"):
        return MINIMAX_H3_INT8_FC1_GROUP_SIZE
    if name.endswith("mlp.fc2.weight"):
        return MINIMAX_H3_INT8_FC2_GROUP_SIZE
    raise Error(
        String("MiniMax-H3 int8 group profile has no class for ") + name
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
    var fp8: List[ArcPointer[Tensor]]    # quantized bytes [out,in] (E4M3 or
    #                                      int8 per the store's `scheme`),
    #                                      TRANSFORMED rows
    var scale: List[ArcPointer[Tensor]]  # FP16 [out, in/group] for INT8;
    #                                      F32 [out] for E4M3
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
    in steady state.

    `scheme` (MINIMAX_H3_RESIDENT_INT8/_E4M3) records which encode wrote the
    quantized bytes, and the rebuild dispatches its dequant on it — the pair
    can never drift apart within one store."""

    var blocks: List[MiniMaxH3BlockResidentFp8]
    var start_layer: Int
    var scratch: List[ArcPointer[Tensor]]
    var scheme: Int
    # One open handle on the packed tail cache for the store's lifetime
    # (0/1). The streamed tail previously re-opened the 19.8 GB cache
    # (54 KB header parse + mmap/munmap) once per block per step — up to
    # 1440x per job. Filled lazily by the reload wrappers; clean file-backed
    # pages of a held mapping stay reclaimable under memory pressure.
    var tail_st: List[ArcPointer[SafeTensors]]
    var tail_st_path: String

    def __init__(
        out self,
        var blocks: List[MiniMaxH3BlockResidentFp8],
        start_layer: Int,
        var scratch: List[ArcPointer[Tensor]],
        scheme: Int,
    ):
        self.blocks = blocks^
        self.start_layer = start_layer
        self.scratch = scratch^
        self.scheme = scheme
        self.tail_st = List[ArcPointer[SafeTensors]]()
        self.tail_st_path = String("")

    def resident_bytes(self) raises -> Int:
        """Actual device bytes held by this store (quantized 1B/param + the
        scheme's scales + bf16 keeps + shared dequant scratch) —
        printed at build so the fit claim is a measurement, not fp8_policy's
        prediction restated."""
        var total = 0
        for bi in range(len(self.blocks)):
            ref b = self.blocks[bi]
            for k in range(len(b.fp8)):
                total += b.fp8[k][].numel() * b.fp8[k][].dtype().byte_size()
                total += b.scale[k][].numel() \
                    * b.scale[k][].dtype().byte_size()
            for k in range(len(b.bf16)):
                total += b.bf16[k][].numel() \
                    * b.bf16[k][].dtype().byte_size()
        for k in range(len(self.scratch)):
            total += self.scratch[k][].numel() \
                * self.scratch[k][].dtype().byte_size()
        return total


def _fill_from_host_bf16(
    data: List[BFloat16], dst: Tensor, ctx: DeviceContext
) raises:
    """Upload a host bf16 list into an EXISTING device tensor, verbatim —
    the zero-churn analogue of `Tensor.from_host_bf16`. Reusing `dst`'s
    allocation is the whole point: a fresh device allocation per staged
    tensor is exactly the pool churn that left runs 3+4 with no contiguous
    block for anything after the store."""
    var n = len(data)
    if n != dst.numel():
        raise Error(
            String("_fill_from_host_bf16: size mismatch, host ") + String(n)
            + " vs dst " + String(dst.numel())
        )
    var host = ctx.enqueue_create_host_buffer[DType.uint8](n * 2)
    var hp = host.unsafe_ptr().bitcast[BFloat16]()
    for i in range(n):
        hp[i] = data[i]
    ctx.enqueue_copy(dst_buf=dst.buf, src_buf=host)
    ctx.synchronize()


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


def minimax_h3_allocate_resident_scratch(
    config: MiniMaxH3DiTConfig,
    start_layer: Int,
    ctx: DeviceContext,
) raises -> List[ArcPointer[Tensor]]:
    """Allocate the shared BF16 dequant destinations before the store grows.

    Cache reload and fresh quantization must use the same slot order and
    allocation timing.  Keeping that derivation here prevents a disk-sidecar
    loader from silently drifting from `minimax_h3_resident_block_weights`.
    """
    var scratch = List[ArcPointer[Tensor]]()
    var template_names = minimax_h3_block_tensor_names(start_layer)
    for k in range(len(template_names)):
        ref tn = template_names[k]
        var tw = minimax_h3_expected_shape(tn, config)
        var trows = tw[0]
        var tcols = tw[1] if len(tw) == 2 else 0
        if minimax_h3_fp8_class(tn, trows, tcols) == H3_FP8_ROW:
            var buf = ctx.enqueue_create_buffer[DType.uint8](trows * tcols * 2)
            var shape: List[Int] = [trows, tcols]
            scratch.append(ArcPointer(Tensor(buf^, shape^, STDtype.BF16)))
    return scratch^


# ─────────────────────────────────────────────────────────────────────────────
# Build: ZERO-CHURN staging. Every big tensor is host-read (with the gated
# qkv/fc1 rewrite where required), FILLED INTO the preallocated scratch for
# its slot, and quantized FROM scratch — the only device allocations the
# whole build makes are the persistent fp8 bytes + scales (the store) and
# the tiny norm keeps. This is the third shape of this loop, each forced by
# a measured OOM on the 24 GiB card (2026-08-03):
#   run 2-3: whole-block Dict transients (~0.77 GiB/layer churn) left the
#            pool at ~23 GiB reserved around the 18.7 GiB store — the next
#            large allocation (frontend weights) failed;
#   run 4:   per-tensor transient lifetimes did NOT lower the end state
#            (23.2 GiB again) — churn is churn to the pool; the post-build
#            scratch allocation failed.
# Staging through the scratch eliminates the churn class entirely: reserved
# ends at approximately (store + scratch + whatever the caller loaded), and
# NOTHING large allocates after the build.
# ─────────────────────────────────────────────────────────────────────────────
def minimax_h3_build_resident_fp8(
    st: ShardedSafeTensors,
    config: MiniMaxH3DiTConfig,
    ctx: DeviceContext,
    num_layers: Int = -1,
    start_layer: Int = 0,
    scheme: Int = MINIMAX_H3_RESIDENT_INT8,
) raises -> MiniMaxH3ResidentFp8:
    if (
        scheme != MINIMAX_H3_RESIDENT_E4M3
        and scheme != MINIMAX_H3_RESIDENT_INT8
        and scheme != MINIMAX_H3_RESIDENT_INT8_W8A8
    ):
        raise Error(
            String("minimax_h3_build_resident_fp8: unknown scheme ")
            + String(scheme)
        )
    if scheme == MINIMAX_H3_RESIDENT_INT8:
        print(
            "  resident scheme: int8-weight-only groupwise fp16-scales qkv=",
            MINIMAX_H3_INT8_QKV_GROUP_SIZE,
            " out=", MINIMAX_H3_INT8_OUT_GROUP_SIZE,
            " fc1=", MINIMAX_H3_INT8_FC1_GROUP_SIZE,
            " fc2=", MINIMAX_H3_INT8_FC2_GROUP_SIZE,
        )
    elif scheme == MINIMAX_H3_RESIDENT_INT8_W8A8:
        print("  resident scheme: direct W8A8 per-row F32 scales")
    else:
        print("  resident scheme: fp8 E4M3 per-row F32 scales")
    var n = num_layers
    if n < 0:
        n = config.num_layers
    if start_layer < 0 or start_layer + n > config.num_layers:
        raise Error(
            String("minimax_h3_build_resident_fp8: layer range [")
            + String(start_layer) + ", " + String(start_layer + n)
            + ") exceeds config.num_layers " + String(config.num_layers)
        )

    # SCRATCH FIRST — before the store grows. Its shapes are a pure function
    # of `config` (all 50 blocks share them), and it is the LAST large
    # allocation this store ever needs: run 4 (2026-08-03) put it after the
    # build and the 231-308 MB allocations OOM'd at 23.2 GiB reserved even
    # though the arithmetic total fit — post-build pool fragmentation leaves
    # no large contiguous block. Law (measured across runs 3+4): every
    # large persistent allocation happens BEFORE the store build; after it,
    # only small activations.
    var scratch = List[ArcPointer[Tensor]]()
    if n > 0:
        scratch = minimax_h3_allocate_resident_scratch(
            config, start_layer, ctx
        )

    var heads = config.num_attention_heads
    var head_dim = config.attention_head_dim
    var hidden = config.hidden_size
    var ffn = config.ffn_hidden_size

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
        var slot = 0
        for k in range(len(names)):
            ref name = names[k]
            var want = minimax_h3_expected_shape(name, config)
            var rows = want[0]
            var cols = want[1] if len(want) == 2 else 0
            var cls = minimax_h3_fp8_class(name, rows, cols)
            if cls == H3_FP8_ROW:
                # Stage THROUGH THE SCRATCH tensor for this slot: host-read
                # (+ the loader's own gated bf16 rewrite where the contract
                # requires one), fill scratch in place, quantize FROM
                # scratch. ZERO fresh device allocations besides the
                # persistent fp8 bytes + scale — the per-layer transient
                # churn of the first three build shapes is what left the
                # pool with no contiguous block for anything afterwards.
                var tv = st.tensor_view(name)
                var tsh = tv.shape.copy()
                if len(tsh) != 2 or tsh[0] != rows or tsh[1] != cols:
                    raise Error(
                        String("MiniMax-H3 fp8 build: shard shape mismatch for ")
                        + name
                    )
                var raw = _tensor_view_bf16_host(tv)
                ref dst = scratch[slot][]
                if name == qkv_name:
                    _fill_from_host_bf16(
                        minimax_h3_deinterleave_qkv_bf16(raw, heads, head_dim, hidden),
                        dst, ctx,
                    )
                elif name == fc1_name:
                    _fill_from_host_bf16(
                        minimax_h3_swap_fc1_bf16(raw, ffn, hidden), dst, ctx,
                    )
                else:
                    _fill_from_host_bf16(raw, dst, ctx)
                # THE two-function encode swap the scheme flag selects (its
                # dequant twin is in minimax_h3_resident_block_weights).
                var s: Tensor
                var q: Tensor
                if scheme == MINIMAX_H3_RESIDENT_INT8:
                    var group_size = minimax_h3_int8_group_size(name)
                    if cols % group_size != 0:
                        raise Error(
                            String("MiniMax-H3 int8 groupwise: input width ")
                            + String(cols) + " for " + name
                            + " is not divisible by group size "
                            + String(group_size)
                        )
                    var s_f32 = int8_groupscale(dst, group_size, ctx)
                    q = int8_encode_groupwise(dst, s_f32, group_size, ctx)
                    # Compact persistent scales to FP16; this recovers enough
                    # residency for the measured QKV16/out64/FC1-32 profile.
                    s = cast_tensor(s_f32, STDtype.F16, ctx)
                elif scheme == MINIMAX_H3_RESIDENT_INT8_W8A8:
                    s = int8_rowscale(dst, ctx)
                    q = int8_encode_perrow(dst, s, ctx)
                else:
                    s = fp8_e4m3_rowscale(dst, ctx)
                    q = fp8_e4m3_encode_perrow(dst, s, ctx)
                # Encode must retire before scratch is refilled next slot.
                ctx.synchronize()
                fp8_names.append(name.copy())
                fp8.append(ArcPointer(q^))
                scale.append(ArcPointer(s^))
                slot += 1
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
        # Every tensor needed for this block has synchronized into persistent
        # device storage. Drop its clean mmap pages before the next block so a
        # long resident build does not leave the entire 36+ GiB transformer
        # charged as file RSS alongside the growing GPU store.
        st.release_to_os()
        if (i + 1) % 10 == 0 or i + 1 == n:
            print(
                "  fp8-resident: quantized block", i + 1, "/", n,
                "(layer", layer, ")",
            )

    # Sanity: the pre-allocated scratch must line up slot-for-slot with what
    # the per-layer loop actually quantized (both derive from
    # minimax_h3_block_tensor_names order filtered by class).
    if len(blocks) > 0 and len(scratch) != len(blocks[0].fp8):
        raise Error(
            "minimax_h3_build_resident_fp8: scratch slots do not match the"
            " quantized slot count — template/loop classification drifted"
        )
    if scheme == MINIMAX_H3_RESIDENT_INT8_W8A8:
        # Scratch is staging-only for the direct path. Releasing it leaves
        # headroom for the INT32 GEMM output while cache-hit runs allocate no
        # dequant destinations at all.
        var no_scratch = List[ArcPointer[Tensor]]()
        return MiniMaxH3ResidentFp8(
            blocks^, start_layer, no_scratch^, scheme
        )
    return MiniMaxH3ResidentFp8(blocks^, start_layer, scratch^, scheme)


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
    if resident.scheme == MINIMAX_H3_RESIDENT_INT8_W8A8:
        raise Error(
            "direct W8A8 store requires minimax_h3_resident_block_weights_w8a8"
        )
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
        # Dequant twin of the build's scheme-selected encode. Both `_into`
        # variants share the caller-owned-dst contract (fp8.mojo's header
        # records the measured OOM that forced it).
        if resident.scheme == MINIMAX_H3_RESIDENT_INT8:
            var group_size = minimax_h3_int8_group_size(b.fp8_names[k])
            int8_dequant_groupwise_to_bf16_into(
                b.fp8[k][], b.scale[k][], group_size,
                resident.scratch[k][], ctx,
            )
        else:
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


def minimax_h3_resident_block_weights_w8a8(
    resident: MiniMaxH3ResidentFp8,
    layer: Int,
    config: MiniMaxH3DiTConfig,
    ctx: DeviceContext,
) raises -> Dict[String, ArcPointer[Tensor]]:
    """Expose one per-row INT8 block without materializing BF16 weights."""
    if resident.scheme != MINIMAX_H3_RESIDENT_INT8_W8A8:
        raise Error("MiniMax-H3 direct block requested from a non-W8A8 store")
    var i = layer - resident.start_layer
    if i < 0 or i >= len(resident.blocks):
        raise Error("MiniMax-H3 direct W8A8 layer outside resident range")
    ref block = resident.blocks[i]
    var weights = Dict[String, ArcPointer[Tensor]]()
    for k in range(len(block.fp8)):
        ref weight = block.fp8[k][]
        ref scale = block.scale[k][]
        if weight.dtype() != STDtype.I8 or scale.dtype() != STDtype.F32:
            raise Error("MiniMax-H3 direct W8A8 cache dtype mismatch")
        if scale.numel() != weight.shape()[0]:
            raise Error("MiniMax-H3 direct W8A8 cache scale mismatch")
        weights[block.fp8_names[k]] = block.fp8[k].copy()
        weights[block.fp8_names[k] + String(".scale")] = block.scale[k].copy()
    for k in range(len(block.bf16)):
        weights[block.bf16_names[k]] = block.bf16[k].copy()
    weights[MINIMAX_H3_QKV_DEINTERLEAVED_MARKER] = ArcPointer(
        _minimax_h3_marker_tensor(ctx)
    )
    weights[MINIMAX_H3_FC1_SWAPPED_MARKER] = ArcPointer(
        _minimax_h3_marker_tensor(ctx)
    )
    return weights^


def minimax_h3_resident_block_weights_groupwise(
    resident: MiniMaxH3ResidentFp8,
    layer: Int,
    config: MiniMaxH3DiTConfig,
    ctx: DeviceContext,
) raises -> Dict[String, ArcPointer[Tensor]]:
    """Expose groupwise INT8 weights for projection-local BF16 dequant."""
    if resident.scheme != MINIMAX_H3_RESIDENT_INT8:
        raise Error("MiniMax-H3 groupwise block requested from wrong store")
    var i = layer - resident.start_layer
    if i < 0 or i >= len(resident.blocks):
        raise Error("MiniMax-H3 groupwise layer outside resident range")
    ref block = resident.blocks[i]
    var weights = Dict[String, ArcPointer[Tensor]]()
    for k in range(len(block.fp8)):
        ref weight = block.fp8[k][]
        ref scale = block.scale[k][]
        if weight.dtype() != STDtype.I8 or scale.dtype() != STDtype.F16:
            raise Error("MiniMax-H3 groupwise cache dtype mismatch")
        var wshape = weight.shape()
        var sshape = scale.shape()
        if len(wshape) != 2 or len(sshape) != 2 \
                or sshape[0] != wshape[0] or sshape[1] <= 0 \
                or wshape[1] % sshape[1] != 0:
            raise Error("MiniMax-H3 groupwise cache scale mismatch")
        weights[block.fp8_names[k]] = block.fp8[k].copy()
        weights[block.fp8_names[k] + String(".scale")] = block.scale[k].copy()
    for k in range(len(block.bf16)):
        weights[block.bf16_names[k]] = block.bf16[k].copy()
    weights[MINIMAX_H3_QKV_DEINTERLEAVED_MARKER] = ArcPointer(
        _minimax_h3_marker_tensor(ctx)
    )
    weights[MINIMAX_H3_FC1_SWAPPED_MARKER] = ArcPointer(
        _minimax_h3_marker_tensor(ctx)
    )
    return weights^
