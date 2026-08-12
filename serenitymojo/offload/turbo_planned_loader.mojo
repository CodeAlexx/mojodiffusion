# turbo_planned_loader.mojo — TurboPlannedLoader: plan-aware async wrapper.
#
# Phase 3: wires TurboBlockLoader (async double-buffer) to the BlockPlan /
# OffloadConfig protocol, exposing the EXACT PlannedBlockLoader call surface so
# Klein9BOffloaded can swap loaders with a single constructor change.
#
# DTYPE NOTE (confirmed 2026-05-28):
#   All Klein9B transformer-block weights (double_blocks.N.*, single_blocks.N.*)
#   are stored as BF16 in the checkpoint (flux-2-klein-base-9b.safetensors).
#   TurboBlockLoader performs a raw byte copy (H2D copy kernel, no dtype
#   conversion). Because on-disk dtype == in-model dtype (both BF16), the raw
#   copy is CORRECT and produces byte-identical blocks to the synchronous
#   PlannedBlockLoader path. No dtype-converting staging path is required.
#
#   If a future model uses F32-on-disk weights with force_bf16 policy, a
#   converting staging path would be required. This file documents that case
#   but does NOT implement it (out of scope for Phase 3; Klein passes without
#   it). A loud runtime assertion guards against inadvertent misuse.
#
# Public surface (EXACT mirror of PlannedBlockLoader):
#   TurboPlannedLoader.open(dir, plan, config, ctx) -> TurboPlannedLoader
#   loader.prefetch(i)               # plan-index bookkeeping (GPU dispatch deferred)
#   loader.prefetch_next(i)          # same pattern as PlannedBlockLoader
#   loader.prefetch_with_ctx(i, ctx) # immediate copy-stream dispatch
#   loader.prefetch_next_with_ctx(i, ctx)
#   loader.await_block(i, ctx) -> PlannedBlockHandle
#   loader.block_count() -> Int
#   loader.branch_visits() -> Int
#
# DESIGN: ctx availability at prefetch() time
# ─────────────────────────────────────────────────────────────────────────────
# PlannedBlockLoader.prefetch(i) has NO ctx parameter (it only issues MADV_WILLNEED
# via BlockLoader.prefetch_block — a CPU-only operation). TurboBlockLoader.prefetch
# DOES need ctx to dispatch the GPU copy kernel.
#
# Current implementation: prefetch(i) does CPU-side bookkeeping only (residency
# state advance, record one pending index). The actual GPU copy dispatch happens
# in await_block(i, ctx), which has ctx.
#
# Field finding, 2026-05-31: the one-pending-index design preserves correctness
# but can lose overlap in loops that call prefetch_next(i) before await_block(i).
# The next pending block may overwrite the current block's pending dispatch, so
# await_block(current) can stage the next block first and then synchronously stage
# the current block through TurboBlockLoader.await_block's fallback.
#
# Shared fix to implement next: add explicit-context prefetch methods
# (prefetch_with_ctx / prefetch_next_with_ctx) that dispatch the copy stream
# immediately, then update hot forward/backward loops to:
#   1. prefetch current with ctx before the loop
#   2. await current
#   3. prefetch next/previous with ctx
#   4. run current block math while copy stream stages the lookahead block.
#
# Residency bookkeeping is kept lightweight: we track UNLOADED → HOST_STAGED
# → PREFETCHING → GPU_READY transitions via ResidencyManager so the budget
# and eviction machinery has correct state. We do NOT perform active eviction
# in Phase 3 (the two-slot TurboBlockLoader manages its own slot rotation).
#
# The ResidencyManager is wired with a large budget (all blocks × max_bytes)
# so can_prefetch() is always True in Phase 3 — we are exercising parity, not
# memory pressure. Phase 4 can tighten the budget and enable eviction.

from std.gpu.host import DeviceBuffer, DeviceContext, DeviceEvent, HostBuffer
from std.memory import ArcPointer, alloc
from std.ffi import external_call

from serenitymojo.offload.block_loader import Block
from serenitymojo.offload.plan import BlockPlan, DTypePolicy, OffloadConfig
from serenitymojo.offload.planned_loader import PlannedBlockHandle
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import BytePtr, sys_memcpy
from serenitymojo.offload.turbo_loader import (
    TurboBlockLoader, _TensorRecord, _h2d_dma_copy,
)
from serenitymojo.offload.vmm_cuda import (
    cu_mem_get_allocation_granularity,
    cu_mem_get_info,
)
from serenitymojo.offload.vmm_manager import VmmModelHandle
from serenitymojo.offload.residency import (
    ResidencyManager,
    BudgetTracker,
    BlockState,
)
from serenitymojo.tensor import Tensor
from serenitymojo.ops.fp8_quant import (
    fp8_e4m3_rowscale, fp8_e4m3_encode_perrow,
)
from serenitymojo.ops.fp8 import (
    fp8_e4m3_dequant_perrow_to_bf16,
    fp8_e4m3_dequant_perrow_to_bf16_into,
    fp8_e4m3_dequant_to_bf16,
)
# int8-resident weight quant (Klein int8-W8A8 slice 4). Same tensorwise weight
# quant slices 1-3 use for the block-direct int8 payload, so the loader-quantized
# int8 == the block's quantize_*_int8 (parity-gated in klein_loader_int8_parity).
from serenitymojo.ops.int8_quant import (
    int8_tensorwise_scale, int8_encode_tensorwise,
)
# squareq_w4-resident (SquareQ chunk 3): packed int4+H256+low-rank sidecar,
# reconstructed to BF16 per block on await (same shape as the fp8 branch).
from serenitymojo.ops.squareq import squareq_reconstruct_weight, squareq_w8_reconstruct_weight
# squareq_nvfp4-resident (SquareQ chunk 8): packed NVFP4 (e2m1 + tiled ue4m3
# scales + low-rank + global scale) sidecar. await returns BOTH the
# reconstructed BF16 W_hat (backward + non-wired consumers) AND the packed
# payload under "::"-suffixed keys (the Klein blocks' native-fp4 forward).
from serenitymojo.ops.squareq_nvfp4 import squareq_nvfp4_reconstruct_weight
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors import read_f32_scalar_bytes
from serenitymojo.io.safetensors_writer import (
    HostBufferTensorDesc, save_safetensors_host_buffers,
)

comptime TArc = ArcPointer[Tensor]
comptime _FP8_HOST_CACHE_VERSION = 2
comptime _FP8_CACHE_SCALE_SUFFIX = ".__fp8_scale"


def _stat_size_mtime(path: String) -> List[Int]:
    """Follow symlinks and return [size, mtime_sec], or [-1, -1]."""
    var n = path.byte_length()
    var cbuf = alloc[UInt8](n + 1)
    var src = path.as_bytes()
    for i in range(n):
        cbuf[i] = src[i]
    cbuf[n] = 0
    var statbuf = alloc[UInt8](160)
    var rc = Int(external_call["stat", Int32](
        BytePtr(unsafe_from_address=Int(cbuf)),
        BytePtr(unsafe_from_address=Int(statbuf)),
    ))
    var size = -1
    var mtime = -1
    if rc == 0:
        var q = statbuf.bitcast[Int64]()
        size = Int(q[6])
        mtime = Int(q[11])
    cbuf.free()
    statbuf.free()
    var out: List[Int] = [size, mtime]
    return out^


def _shape_equal(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def _cache_i64(
    cache: ShardedSafeTensors, name: String, default: Int
) raises -> Int:
    if not cache.has_tensor(name):
        return default
    var info = cache.tensor_info(name)
    if info.dtype != STDtype.I64 or info.size < 8:
        return default
    var b = cache.tensor_bytes(name)
    var value = 0
    for i in range(8):
        value = value | (Int(b[i]) << (8 * i))
    return value


# int8-resident Block carries each big weight as an I8 [N,K] tensor plus a
# per-weight F32 scalar scale [1]; the scale is stored in the Block dict under
# `<weight_name>` + this suffix so the block-weight builder can pair them and
# assemble the StreamInt8 / SingleBlockInt8 payload (NO dequant — int8 stays int8
# to the GEMM). The suffix is chosen so it can never collide with a real tensor
# name (real Klein tensor names never contain "::").
def resident_i8_scale_key(name: String) -> String:
    return name + String("::i8scale")


# nvfp4-resident Block payload keys (SquareQ chunk 8): each quantized weight
# lands in the Block under its own name (the reconstructed BF16 W_hat) AND
# under these "::"-suffixed keys carrying the packed payload the native-FP4
# forward reads (`part` ∈ nvq/nvs/ld/lu/nvg). Same collision argument as
# resident_i8_scale_key: real Klein tensor names never contain "::".
def resident_nvfp4_key(name: String, part: String) -> String:
    return name + String("::") + part

# fp8-resident base weights (MJ-1065, 2026-07-03): the per-block permanent set,
# but the large 2-D BF16 matmul weights are quantized ONCE to E4M3 bytes + a
# per-output-row F32 scale (fp8_quant.mojo) instead of held raw BF16. Halves-plus
# the resident footprint (~17GB bf16 → ~9GB fp8 for chroma/sd35) so the whole DiT
# fits device-resident with LoRA training state. await_block dequants each block's
# weights to BF16 on the fly (fp8_e4m3_dequant_perrow_to_bf16 — cheap GPU kernel,
# NO disk), reconstructing the SAME Block the streamed path returns. Small tensors
# (1-D biases/norm scales, tiny 2-D) stay resident BF16 (exact). Lossy: E4M3 has 3
# mantissa bits → decode(encode(w)) ≈ cos 0.99 vs bf16 — a different-trajectory
# numerics class from the streamed base, gated on "loss still drops", NOT bit-exact.
comptime _FP8_MIN_ELEMS = 1 << 16   # only fp8 sizeable 2-D weights; tiny ones stay bf16


# squareq_w4-resident sibling of _ResidentFp8Tensor. If `is_quant`: `q` is the
# packed int4 qweight U8 [out,in/2], `s` the BF16 group-64 scales [in/64,out],
# `ld`/`lu` the BF16 low-rank factors [in,R]/[out,R]; await reconstructs
# W_hat = dequant@H_bd + lu@ld^T to BF16 (ops/squareq.mojo). Else `q` is the
# owned BF16 tensor held verbatim (`s`/`ld`/`lu` are unused aliases of `q`).
# Resident bytes ~0.28x bf16 — the whole point of the vertical.
struct _ResidentSquareqTensor(Copyable, Movable):
    var name: String
    var is_quant: Bool
    var is_w8: Bool   # v3.1 int8-residual (q=i8 [out,in], s=per-row bf16 [out])
    var q: TArc
    var s: TArc
    var ld: TArc
    var lu: TArc
    var in_f: Int
    var out_f: Int

    def __init__(
        out self, var name: String, is_quant: Bool,
        var q: TArc, var s: TArc, var ld: TArc, var lu: TArc,
        in_f: Int, out_f: Int, is_w8: Bool = False,
    ):
        self.name = name^
        self.is_quant = is_quant
        self.is_w8 = is_w8
        self.q = q^
        self.s = s^
        self.ld = ld^
        self.lu = lu^
        self.in_f = in_f
        self.out_f = out_f


# squareq_nvfp4-resident sibling of _ResidentSquareqTensor. If `is_quant`:
# `nvq` is the packed e2m1 U8 [out,in/2], `nvs` the U8 tiled ue4m3 weight
# scales, `ld`/`lu` the BF16 low-rank factors [in,R]/[out,R], `nvg_t` the
# pinned 1-elem F32 global-scale tensor (shared into the Block under the
# "::nvg" key) and `nvg` its HOST value read ONCE at pin time (rides cublasLt
# alpha in the fp4 forward + the reconstruct kernel). Else `nvq` is the owned
# BF16 tensor held verbatim (the other TArcs are unused aliases of `nvq`).
struct _ResidentNvfp4Tensor(Copyable, Movable):
    var name: String
    var is_quant: Bool
    var nvq: TArc
    var nvs: TArc
    var ld: TArc
    var lu: TArc
    var nvg_t: TArc
    var nvg: Float32
    var in_f: Int
    var out_f: Int

    def __init__(
        out self, var name: String, is_quant: Bool,
        var nvq: TArc, var nvs: TArc, var ld: TArc, var lu: TArc,
        var nvg_t: TArc, nvg: Float32, in_f: Int, out_f: Int,
    ):
        self.name = name^
        self.is_quant = is_quant
        self.nvq = nvq^
        self.nvs = nvs^
        self.ld = ld^
        self.lu = lu^
        self.nvg_t = nvg_t^
        self.nvg = nvg
        self.in_f = in_f
        self.out_f = out_f


struct _ResidentFp8Tensor(Copyable, Movable):
    """One resident block tensor. If `is_fp8`: `a` is E4M3 bytes [out,in], `scale`
    is the per-row F32 scale [out] (dequant to BF16 on await). Else `a` is the
    owned BF16 tensor held verbatim (`scale` is an unused alias of `a`)."""
    var name: String
    var is_fp8: Bool
    var a: TArc
    var scale: TArc

    def __init__(
        out self, var name: String, is_fp8: Bool, var a: TArc, var scale: TArc
    ):
        self.name = name^
        self.is_fp8 = is_fp8
        self.a = a^
        self.scale = scale^


# int8-resident sibling of _ResidentFp8Tensor (Klein int8-W8A8 slice 4). If
# `is_int8`: `w8` is int8 [N,K] (ONE orientation; the backward uses NN — no
# transpose) and `scale` is the F32 scalar tensorwise scale [1]; await returns
# them UNCHANGED (no dequant — unlike fp8). Else `w8` is the owned BF16 tensor
# held verbatim (small 1-D/tiny 2-D; `scale` is an unused alias of `w8`). int8 =
# 1 byte/param → ~half the raw-BF16 footprint.
struct _ResidentInt8Tensor(Copyable, Movable):
    var name: String
    var is_int8: Bool
    var w8: TArc
    var scale: TArc

    def __init__(
        out self, var name: String, is_int8: Bool, var w8: TArc, var scale: TArc
    ):
        self.name = name^
        self.is_int8 = is_int8
        self.w8 = w8^
        self.scale = scale^


comptime HArc = ArcPointer[HostBuffer[DType.uint8]]


struct _HostFp8Tensor(Copyable, Movable):
    """One HOST-pinned block tensor (fp8_e4m3_host mode — models whose fp8
    footprint does not fit device-resident with training state, e.g. qwenimage
    ~20GB fp8 on a 24GB card). If `is_fp8`: `bytes_h` holds E4M3 bytes [out,in]
    and `scale_h` the per-row F32 scale bytes; await H2Ds both and dequants.
    Else `bytes_h` holds BF16 bytes (`scale_h` is a 1-byte dummy). F32
    checkpoint tables are converted to BF16 once while this host store is
    built; denoise never revisits the checkpoint mapping."""
    var name: String
    var is_fp8: Bool
    var per_row_scale: Bool
    var scalar_scale: Float32
    var bytes_h: HArc
    var bytes_nbytes: Int
    var bytes_shape: List[Int]
    var bytes_dtype: STDtype
    var scale_h: HArc
    var scale_nbytes: Int
    var scale_shape: List[Int]

    def __init__(
        out self, var name: String, is_fp8: Bool, per_row_scale: Bool,
        scalar_scale: Float32,
        var bytes_h: HArc, bytes_nbytes: Int,
        var bytes_shape: List[Int], bytes_dtype: STDtype,
        var scale_h: HArc, scale_nbytes: Int, var scale_shape: List[Int],
    ):
        self.name = name^
        self.is_fp8 = is_fp8
        self.per_row_scale = per_row_scale
        self.scalar_scale = scalar_scale
        self.bytes_h = bytes_h^
        self.bytes_nbytes = bytes_nbytes
        self.bytes_shape = bytes_shape^
        self.bytes_dtype = bytes_dtype
        self.scale_h = scale_h^
        self.scale_nbytes = scale_nbytes
        self.scale_shape = scale_shape^


struct _HostInt8Rec(Copyable, Movable, ImplicitlyCopyable):
    """Per-tensor record into a block's CONTIGUOUS pinned int8-host buffer
    (Klein slice-6 stall fix, 2026-07-11 — the old per-tensor _HostInt8Tensor
    layout forced N little in-stream H2Ds per await; the contiguous layout
    allows ONE whole-block async H2D on the copy stream). Mirrors _TensorRecord
    but pairs each big int8 weight with its F32 scalar scale: `w_off/w_nbytes`
    locate the payload bytes (int8 [N,K] when `is_int8`, raw BF16 otherwise)
    and, when `is_int8`, `s_off/s_nbytes` locate the tensorwise-scale bytes.
    Offsets are IDENTICAL in the pinned HOST block buffer and in the persistent
    DEVICE slabs (one block-sized H2D lands every tensor at its recorded
    offset). Shapes are ArcPointer so the struct stays Copyable (the
    _TensorRecord idiom)."""
    var name: String
    var is_int8: Bool
    var w_off: Int
    var w_nbytes: Int
    var w_shape: ArcPointer[List[Int]]
    var w_dtype: STDtype
    var s_off: Int
    var s_nbytes: Int
    var s_shape: ArcPointer[List[Int]]

    def __init__(
        out self, var name: String, is_int8: Bool,
        w_off: Int, w_nbytes: Int,
        var w_shape: ArcPointer[List[Int]], w_dtype: STDtype,
        s_off: Int, s_nbytes: Int, var s_shape: ArcPointer[List[Int]],
    ):
        self.name = name^
        self.is_int8 = is_int8
        self.w_off = w_off
        self.w_nbytes = w_nbytes
        self.w_shape = w_shape^
        self.w_dtype = w_dtype
        self.s_off = s_off
        self.s_nbytes = s_nbytes
        self.s_shape = s_shape^


def _align_up(x: Int, a: Int) -> Int:
    return ((x + a - 1) // a) * a


def _copy_block_plan(plan: BlockPlan) -> BlockPlan:
    """Shallow-copy a BlockPlan (BlockRecord is Copyable; String/Int copy naturally)."""
    var out = BlockPlan(plan.name)
    for i in range(len(plan.records)):
        var r = plan.records[i].copy()
        out.append(r.prefix, r.kind, r.tensor_count_hint, r.byte_count_hint)
    return out^


struct Fp8hStage(Movable):
    """The double-buffered device staging slabs for overlapped fp8-host blocks.

    Held behind an `ArcPointer` so SEVERAL loaders can share ONE set of slabs. That
    matters for dual-expert models: only ONE expert runs per step, so a per-loader
    pair leaves the other expert's slabs (~640 MB at wan2.2 A14B block sizes) idle
    for the whole run — and on a 16 GB card that memory is worth ~2 more
    device-resident blocks, which are measurably worth more than their VRAM
    (HANDOFF_WAN_SPEED_2026-07-24D §4.3).

    ⚠ **Slots are keyed by `<tag>|<prefix>`, never by prefix alone.** Two experts are
    separate checkpoints with IDENTICAL block naming, so a bare-prefix key would
    report HIGH's `blocks.7` as a hit for LOW's `blocks.7` and silently compute with
    the wrong expert's weights. Each loader supplies its own `tag`."""

    var devs: List[DeviceBuffer[DType.uint8]]   # 2 persistent slabs
    var evs: List[DeviceEvent]                  # per-slot copy-done (copy stream)
    var cds: List[DeviceEvent]                  # per-slot compute-done (default stream)
    var cd_rec: List[Bool]                      # compute-done recorded flags
    var slot_key: List[String]                  # staged "<tag>|<prefix>" per slot
    var active: Int                             # slot compute is reading
    var capacity: Int                           # slab bytes (max fp8h block seen)

    def __init__(out self):
        self.devs = List[DeviceBuffer[DType.uint8]]()
        self.evs = List[DeviceEvent]()
        self.cds = List[DeviceEvent]()
        self.cd_rec = List[Bool]()
        self.slot_key = List[String]()
        self.active = 0
        self.capacity = 0


struct TurboPlannedLoader(Movable):
    """Plan-aware async wrapper over TurboBlockLoader.

    Exposes exactly the PlannedBlockLoader call surface (prefetch / prefetch_next
    / await_block returning PlannedBlockHandle) so Klein9BOffloaded can use
    either loader interchangeably.

    Double-buffer contract (inherited from TurboBlockLoader):
      prefetch(i) records i as pending; GPU dispatch happens in the next
      await_block or prefetch-with-ctx call.
      await_block(i) dispatches the copy kernel if not yet staged, fences
      the default stream, and returns the Block.

    Residency state machine:
      We drive ResidencyManager through the correct transitions so budget and
      eviction scoring have accurate state. In Phase 3 the budget is sized
      generously (no eviction pressure), so the machinery is wired but idle.
    """

    var _turbo: TurboBlockLoader
    var _plan: BlockPlan
    var _config: OffloadConfig
    var _residency: ResidencyManager
    var _source_path: String
    var _step: Int          # monotonic block-visit counter for mark_visit
    var _pending_idx: Int   # block index queued by prefetch() awaiting GPU dispatch
    var _has_ctx: Bool      # whether _stored_ctx is valid
    # ── resident set (Phase 4, 2026-06-11): blocks pinned PERMANENTLY on device
    # in their own buffers, loaded ONCE from the pinned block store. await_block
    # returns them with NO copy and NO slot use; prefetch is a no-op for them.
    # Default empty = behavior identical to before; opt in via pin_residents().
    var _res_prefixes: List[String]
    var _res_devs: List[DeviceBuffer[DType.uint8]]
    var _res_recs: List[List[_TensorRecord]]
    # ── fp8-resident set (MJ-1065): parallel to the raw resident set above but
    # holds each block's large weights as (E4M3 bytes, per-row F32 scale) pairs
    # (dequant on await). Opt in via pin_residents_fp8(); default empty = no-op.
    var _fp8_prefixes: List[String]
    var _fp8_blocks: List[List[_ResidentFp8Tensor]]
    # Adaptive host->device promotions use explicit CUDA VMM physical mappings,
    # not DeviceContext's process-lifetime caching allocator. Tensor records
    # hold NON-OWNING DeviceBuffer views into this handle; explicit discard can
    # therefore unmap/release the physical bytes immediately.
    var _fp8_vmm: List[ArcPointer[VmmModelHandle]]

    # ── squareq_w4-resident set: parallel to _fp8_blocks. Pinned VERBATIM from
    # the prebuilt sidecar (scripts/squareq_build_slab.py output dir — never
    # quantize-at-load); await reconstructs BF16 per block. Opt in via
    # pin_residents_squareq(); default empty = no-op.
    var _squareq_prefixes: List[String]
    var _squareq_blocks: List[List[_ResidentSquareqTensor]]
    # ── squareq_nvfp4-resident set: parallel to _squareq_blocks. Pinned
    # VERBATIM from the prebuilt nvfp4 sidecar (scripts/squareq_build_slab.py
    # --format nvfp4 output — never quantize-at-load); await reconstructs the
    # BF16 W_hat per block AND shares the packed payload under "::" keys.
    # Opt in via pin_residents_squareq_nvfp4(); default empty = no-op.
    var _nvfp4_prefixes: List[String]
    var _nvfp4_blocks: List[List[_ResidentNvfp4Tensor]]
    # P2-A: when True, nvfp4 awaits SKIP the bf16 W_hat reconstruct for
    # quantized weights (forward uses only the ::payload; a 1-elem bf16 dummy
    # rides under the weight name for shape-agnostic builders). The trainer
    # toggles this ON for forward passes and OFF before backward/recompute.
    # Default False = byte-identical to the pre-flag behavior.
    var _fwd_only_awaits: Bool
    # Shared shape-dummy backing store (largest quantized weight, allocated at
    # nvfp4 pin): fwd-only awaits return shape-correct Tensors viewing this
    # buffer (contents GARBAGE — forward reads shapes only; any value use
    # corrupts loss and the smoke catches it).
    var _nvfp4_dummy: List[DeviceBuffer[DType.uint8]]
    # ── host-pinned fp8 set (fp8_e4m3_host): E4M3+scale PINNED in host RAM,
    # H2D'd + dequanted per await (half the PCIe of bf16 streaming; NO disk).
    var _fp8h_prefixes: List[String]
    var _fp8h_blocks: List[List[_HostFp8Tensor]]
    # ── int8-resident set (Klein int8-W8A8 slice 4): parallel to _fp8_blocks but
    # holds each block's large weights as (int8 [N,K], F32 scalar scale [1]) pairs
    # and returns them UNCHANGED on await (NO dequant — int8 to the GEMM). Opt in
    # via pin_residents_int8(); default empty = no-op (bf16 + fp8 paths untouched).
    var _int8_prefixes: List[String]
    var _int8_blocks: List[List[_ResidentInt8Tensor]]
    # ── host-pinned int8 set (Klein int8-W8A8 slice 6): the NON-resident int8
    # tail — each block's big weights quantized ONCE to int8 [N,K] + F32 scalar
    # scale and laid CONTIGUOUSLY into ONE pinned host buffer per block (small
    # tensors held BF16 host bytes in the same buffer), tagged I8 on await with
    # NO dequant (HALF the bf16 stream bytes). Opt in via
    # pin_residents_int8_host(); default empty = no-op.
    #
    # OVERLAP (stall fix 2026-07-11, measured 27% GPU idle = ~0.42s/step of
    # in-stream per-tensor H2D at await time): prefetch_with_ctx H2Ds the NEXT
    # int8h block's whole contiguous buffer into one of TWO persistent device
    # slabs with a single async cuMemcpy on the turbo COPY STREAM (overlapping
    # the current block's compute); await_block only enqueues a wait on the
    # slot's copy-done event and builds the Block from sub-buffer views at the
    # recorded offsets. Slot-reuse hazard (block N+2 overwrites block N's slot
    # while N's kernels may still run) is fenced EXACTLY like TurboBlockLoader's
    # dev0/dev1: mark_active_block_done records a compute-done event on the
    # default stream, and _i8h_prefetch makes the copy stream wait on it before
    # rewriting that slot.
    var _int8h_prefixes: List[String]
    var _int8h_hosts: List[HostBuffer[DType.uint8]]  # ONE contiguous pinned buffer per block
    var _int8h_recs: List[List[_HostInt8Rec]]        # per-tensor offsets into it
    var _int8h_sizes: List[Int]                      # block buffer nbytes
    var _i8h_devs: List[DeviceBuffer[DType.uint8]]   # 2 persistent slabs (once pinned)
    var _i8h_evs: List[DeviceEvent]                  # per-slot copy-done (copy stream)
    var _i8h_cds: List[DeviceEvent]                  # per-slot compute-done (default stream)
    var _i8h_cd_rec: List[Bool]                      # compute-done recorded flags
    var _i8h_slot_prefix: List[String]               # staged prefix per slot ("" = empty)
    var _i8h_active: Int                             # slot compute is reading
    var _i8h_capacity: Int                           # slab bytes (max int8h block)
    # ── fp8-host OVERLAPPED staging (opt-in, `set_fp8h_overlap`) ──────────────
    # The same double-buffered slab machinery as int8h above, for the fp8h path.
    # WHY: fp8h stages INLINE AT AWAIT on the DEFAULT stream (see await_block),
    # so its H2D cannot overlap compute. nsys on the wan2.2 trainer measured
    # 508.6 ms/step of H2D against 859.2 ms/step of kernels with a union of
    # 1367.9 ms — i.e. **0% overlap, perfectly serialized**. Off by default so
    # every existing fp8h caller keeps the byte-identical inline path (C13).
    var _f8h: ArcPointer[Fp8hStage]                  # slabs (own by default; may be shared)
    var _f8h_tag: String                             # slot-key namespace (see Fp8hStage)
    var _f8h_overlap: Bool                           # opt-in switch

    @staticmethod
    def open(
        dir: String,
        var plan: BlockPlan,
        config: OffloadConfig,
        ctx: DeviceContext,
        fill_block_store: Bool = True,
    ) raises -> TurboPlannedLoader:
        return TurboPlannedLoader.open_with_copy_mode(
            dir, plan^, config, ctx, False, fill_block_store
        )

    @staticmethod
    def open_with_copy_mode(
        dir: String,
        var plan: BlockPlan,
        config: OffloadConfig,
        ctx: DeviceContext,
        use_default_stream_copy: Bool,
        fill_block_store: Bool = True,
    ) raises -> TurboPlannedLoader:
        """Open model directory and pre-allocate async resources.

        Constructs TurboBlockLoader (sizes slabs to the largest block),
        a ResidencyManager (one entry per plan block), and a BudgetTracker
        with a generously large limit so no eviction occurs in Phase 3.

        `use_default_stream_copy=True` is a measurement-only ablation for the
        P0 timed gate; production callers should use `open`.

        `fill_block_store=False` skips the whole-DiT PINNED host block store —
        REQUIRED when the caller pins every block device-resident via
        pin_residents_fp8 (MJ-1065 fp8_e4m3 mode): the store is ~17 GB of
        pinned RAM that would never be read again (two concurrent stores
        OOM-killed the user session, systemd-oomd 2026-07-04). Streaming
        still works without it (prefetch falls back to mmap -> slot slab).
        """
        var turbo = TurboBlockLoader.open_with_copy_mode(
            dir, ctx, use_default_stream_copy, fill_block_store
        )
        # Generous budget: 128 GB virtual limit — no eviction pressure in P3.
        var budget = BudgetTracker(
            128 * 1024 * 1024 * 1024,  # high watermark: 128 GB
            64 * 1024 * 1024 * 1024,   # low watermark: 64 GB
        )
        # ResidencyManager and TurboPlannedLoader both need to own a BlockPlan.
        # BlockPlan is Movable-only; copy it via the record-level Copyable impl.
        var plan_for_residency = _copy_block_plan(plan)
        var residency = ResidencyManager(plan_for_residency^, config, budget^)
        return TurboPlannedLoader(turbo^, plan^, config, residency^, dir.copy())

    def __init__(
        out self,
        var turbo: TurboBlockLoader,
        var plan: BlockPlan,
        config: OffloadConfig,
        var residency: ResidencyManager,
        var source_path: String,
    ):
        self._turbo = turbo^
        self._plan = plan^
        self._config = config
        self._residency = residency^
        self._source_path = source_path^
        self._step = 0
        self._pending_idx = -1
        self._has_ctx = False
        self._res_prefixes = List[String]()
        self._res_devs = List[DeviceBuffer[DType.uint8]]()
        self._res_recs = List[List[_TensorRecord]]()
        self._fp8_prefixes = List[String]()
        self._fp8_blocks = List[List[_ResidentFp8Tensor]]()
        self._fp8_vmm = List[ArcPointer[VmmModelHandle]]()
        self._squareq_prefixes = List[String]()
        self._squareq_blocks = List[List[_ResidentSquareqTensor]]()
        self._nvfp4_prefixes = List[String]()
        self._nvfp4_blocks = List[List[_ResidentNvfp4Tensor]]()
        self._fwd_only_awaits = False
        self._nvfp4_dummy = List[DeviceBuffer[DType.uint8]]()
        self._fp8h_prefixes = List[String]()
        self._fp8h_blocks = List[List[_HostFp8Tensor]]()
        self._int8_prefixes = List[String]()
        self._int8_blocks = List[List[_ResidentInt8Tensor]]()
        self._int8h_prefixes = List[String]()
        self._int8h_hosts = List[HostBuffer[DType.uint8]]()
        self._int8h_recs = List[List[_HostInt8Rec]]()
        self._int8h_sizes = List[Int]()
        self._i8h_devs = List[DeviceBuffer[DType.uint8]]()
        self._i8h_evs = List[DeviceEvent]()
        self._i8h_cds = List[DeviceEvent]()
        self._i8h_cd_rec = List[Bool]()
        self._i8h_slot_prefix = List[String]()
        self._i8h_active = 0
        self._i8h_capacity = 0
        self._f8h = ArcPointer[Fp8hStage](Fp8hStage())
        self._f8h_tag = String("")
        self._f8h_overlap = False

    def _resident_slot(self, norm_prefix: String) -> Int:
        for r in range(len(self._res_prefixes)):
            if self._res_prefixes[r] == norm_prefix:
                return r
        return -1

    def _fp8_slot(self, norm_prefix: String) -> Int:
        for r in range(len(self._fp8_prefixes)):
            if self._fp8_prefixes[r] == norm_prefix:
                return r
        return -1

    def _fp8h_slot(self, norm_prefix: String) -> Int:
        for r in range(len(self._fp8h_prefixes)):
            if self._fp8h_prefixes[r] == norm_prefix:
                return r
        return -1

    def _squareq_slot(self, norm_prefix: String) -> Int:
        for r in range(len(self._squareq_prefixes)):
            if self._squareq_prefixes[r] == norm_prefix:
                return r
        return -1

    def set_fwd_only_awaits(mut self, on: Bool):
        """P2-A speed lever: skip nvfp4 W_hat reconstructs on forward-only
        visits. MUST be reset to False before any backward/recompute pass."""
        self._fwd_only_awaits = on

    def _nvfp4_slot(self, norm_prefix: String) -> Int:
        for r in range(len(self._nvfp4_prefixes)):
            if self._nvfp4_prefixes[r] == norm_prefix:
                return r
        return -1

    def _int8_slot(self, norm_prefix: String) -> Int:
        for r in range(len(self._int8_prefixes)):
            if self._int8_prefixes[r] == norm_prefix:
                return r
        return -1

    def _int8h_slot(self, norm_prefix: String) -> Int:
        for r in range(len(self._int8h_prefixes)):
            if self._int8h_prefixes[r] == norm_prefix:
                return r
        return -1

    def pin_residents(
        mut self, budget_bytes: Int, ctx: DeviceContext, max_blocks: Int = -1
    ) raises -> Int:
        """Pin plan blocks (in plan order) permanently on device until
        `budget_bytes` is reached. Each pinned block gets its OWN device buffer,
        H2D'd ONCE from the turbo loader's pinned block store — byte-identical
        to the streaming path (same source bytes, same per-tensor records).
        Returns the number of blocks pinned. One ctx.synchronize() at the end
        fences all resident copies; afterwards await_block on a resident is a
        pure sub-buffer-view build (no copy, no slot, no fence).
        `max_blocks` (16GB refit, P6 wave 2): additional cap on the number of
        pinned blocks (-1 = unlimited, budget-only — the pre-existing behavior;
        0 = pin nothing). Residency knob only: bytes are unchanged."""
        var used = 0
        var pinned = 0
        if max_blocks == 0:
            return 0
        # Reusable PINNED staging buffer (the block_store is a 1-byte dummy
        # when TURBO_USE_PERSISTENT_BLOCK_STORE=False — measured rc=1 when
        # pointing cuMemcpy at it). mmap -> pinned staging -> device, like
        # TurboBlockLoader.prefetch's non-store path, synced per block since
        # the staging buffer is reused (one-time pinning cost).
        var staging = ctx.enqueue_create_host_buffer[DType.uint8](
            self._turbo.slab_capacity
        )
        ctx.synchronize()
        for i in range(self._plan.count()):
            self._assert_raw_copy_dtype_safe(i)
            var p = self._plan.normalized_prefix(i)
            if self._resident_slot(p) >= 0:
                continue
            var prefix_idx = -1
            for j in range(len(self._turbo.index_prefixes)):
                if self._turbo.index_prefixes[j] == p:
                    prefix_idx = j
                    break
            if prefix_idx < 0:
                raise Error(
                    String("pin_residents: no tensors for prefix: ") + p
                )
            var n_bytes = self._turbo.store_nbytes[prefix_idx]
            if used + n_bytes > budget_bytes:
                break  # plan-order contiguous pin; the rest keep streaming
            if max_blocks >= 0 and pinned >= max_blocks:
                break  # 16GB refit block-count cap; the rest keep streaming
            # per-tensor records + mmap->staging memcpy, EXACTLY like
            # TurboBlockLoader.prefetch's non-store path
            var recs = List[_TensorRecord]()
            var offset = 0
            var start = self._turbo.index_starts[prefix_idx]
            var end = start + self._turbo.index_lengths[prefix_idx]
            for ni in range(start, end):
                var nm = self._turbo.index_names[ni].copy()
                var tv = self._turbo.sharded.tensor_view(nm)
                var nb = tv.nbytes()
                var sh = ArcPointer(tv.shape.copy())
                var dst = BytePtr(
                    unsafe_from_address=Int(staging.unsafe_ptr()) + offset
                )
                var src = BytePtr(
                    unsafe_from_address=Int(tv.data.unsafe_ptr())
                )
                _ = sys_memcpy(dst, src, nb)
                recs.append(_TensorRecord(nm, offset, nb, sh, tv.dtype))
                offset += nb
            if offset != n_bytes:
                raise Error(
                    String("pin_residents: record bytes ") + String(offset)
                    + " != store bytes " + String(n_bytes) + " for " + p
                )
            var dev = ctx.enqueue_create_buffer[DType.uint8](n_bytes)
            ctx.synchronize()  # materialize alloc before raw CUDA copy
            _h2d_dma_copy(
                UInt64(Int(dev.unsafe_ptr())),
                staging.unsafe_ptr(),
                n_bytes,
                self._turbo.copy_stream,
            )
            # Fence the COPY STREAM specifically: the h2d above runs on
            # self._turbo.copy_stream, which ctx.synchronize() does NOT cover
            # (it only waits on the context's own stream — same trap documented
            # at the fp8 pin path below, "measured cos 0.09 without this
            # fence"). Without this, the staging memcpy for the NEXT block
            # races the in-flight DMA and pins nondeterministically corrupted
            # bytes (measured: mageflow run-to-run loss wobble, 2026-07-22).
            self._turbo.copy_stream.synchronize()
            ctx.synchronize()
            self._res_prefixes.append(p)
            self._res_devs.append(dev^)
            self._res_recs.append(recs^)
            used += n_bytes
            pinned += 1
        return pinned

    def pin_residents_fp8(mut self, budget_bytes: Int, ctx: DeviceContext) raises -> Int:
        """Pin plan blocks (in plan order) permanently on device as fp8-resident:
        each sizeable 2-D BF16/FP16 matmul weight is quantized ONCE to E4M3 bytes + a
        per-output-row F32 scale (fp8_quant.mojo); 1-D biases / norm scales / tiny
        2-D tensors stay resident BF16 (exact). ~half the raw-BF16 footprint, so
        the whole DiT fits resident with LoRA training state. await_block dequants
        each block's weights to BF16 on the fly (NO disk). Returns the number of
        blocks pinned; `budget_bytes` caps the fp8-resident bytes (plan-order
        contiguous — the caller must require pinned==block_count so no block is
        left per-step disk-streaming, MJ-1065). One sync per block bounds the
        transient BF16 source (a single tensor is on device at a time)."""
        var used = 0
        var pinned = 0
        for i in range(self._plan.count()):
            self._assert_raw_copy_dtype_safe(i)
            var p = self._plan.normalized_prefix(i)
            if self._fp8_slot(p) >= 0 or self._resident_slot(p) >= 0:
                continue
            var prefix_idx = -1
            for j in range(len(self._turbo.index_prefixes)):
                if self._turbo.index_prefixes[j] == p:
                    prefix_idx = j
                    break
            if prefix_idx < 0:
                raise Error(
                    String("pin_residents_fp8: no tensors for prefix: ") + p
                )
            var start = self._turbo.index_starts[prefix_idx]
            var end = start + self._turbo.index_lengths[prefix_idx]
            var tensors = List[_ResidentFp8Tensor]()
            var block_bytes = 0
            for ni in range(start, end):
                var nm = self._turbo.index_names[ni].copy()
                var tv = self._turbo.sharded.tensor_view(nm)
                var sh = tv.shape.copy()
                var big_2d = (
                    (tv.dtype == STDtype.BF16 or tv.dtype == STDtype.F16)
                    and len(sh) == 2
                    and sh[0] * sh[1] >= _FP8_MIN_ELEMS
                )
                if big_2d:
                    var w_bf = Tensor.from_view_as_bf16(tv, ctx)   # owned BF16 [out,in]
                    var scale = fp8_e4m3_rowscale(w_bf, ctx)       # F32 [out]
                    var bytes = fp8_e4m3_encode_perrow(w_bf, scale, ctx)  # E4M3 [out,in]
                    block_bytes += (
                        bytes.numel() * bytes.dtype().byte_size()
                        + scale.numel() * scale.dtype().byte_size()
                    )
                    tensors.append(
                        _ResidentFp8Tensor(nm, True, TArc(bytes^), TArc(scale^))
                    )
                    # w_bf frees at loop-iteration end (its buffer is only the
                    # transient quantize source; the ctx.synchronize below fences it).
                else:
                    var t = Tensor.from_view_as_bf16(tv, ctx)      # owned BF16 (small)
                    block_bytes += t.numel() * t.dtype().byte_size()
                    var arc = TArc(t^)
                    tensors.append(
                        _ResidentFp8Tensor(nm, False, arc.copy(), arc)
                    )
            if used + block_bytes > budget_bytes:
                break  # plan-order contiguous pin; caller enforces pinned==count
            self._fp8_prefixes.append(p)
            self._fp8_blocks.append(tensors^)
            used += block_bytes
            pinned += 1
            ctx.synchronize()  # fence this block's transient BF16 before the next
        return pinned

    def pin_residents_squareq(
        mut self, sidecar_dir: String, budget_bytes: Int, ctx: DeviceContext
    ) raises -> Int:
        """Pin plan blocks (plan order) as squareq_w4-resident from the PREBUILT
        sidecar directory (scripts/squareq_build_slab.py output: sharded
        safetensors holding `<base>.qweight/.wscales/.lora_down/.lora_up` per
        quantized linear, everything else passthrough). NEVER quantizes at load.

        For each base-checkpoint tensor name in the block:
          - `<base>.weight` with a sidecar `<base>.qweight`: pin the 4 packed
            sidecar tensors VERBATIM (qweight raw U8; scales/factors BF16).
            await_block reconstructs W_hat = dequant@H_bd + lora_up@lora_down^T
            to BF16 per visit (ops/squareq.squareq_reconstruct_weight) — the
            downstream block builders see the same names/dtypes as the streamed
            path, so no model-side changes are needed for the dequant-first tier.
          - anything else: pinned resident BF16 from the sidecar's passthrough
            copy (falls back to the base checkpoint if the sidecar lacks it).

        Resident bytes ~0.28x bf16. Same MJ-1065 contract as fp8: the caller
        must require pinned == block_count. One sync per block bounds transients.
        """
        var sc = ShardedSafeTensors.open(sidecar_dir)
        var used = 0
        var pinned = 0
        for i in range(self._plan.count()):
            self._assert_raw_copy_dtype_safe(i)
            var p = self._plan.normalized_prefix(i)
            if (
                self._squareq_slot(p) >= 0
                or self._fp8_slot(p) >= 0
                or self._resident_slot(p) >= 0
            ):
                continue
            var prefix_idx = -1
            for j in range(len(self._turbo.index_prefixes)):
                if self._turbo.index_prefixes[j] == p:
                    prefix_idx = j
                    break
            if prefix_idx < 0:
                raise Error(
                    String("pin_residents_squareq: no tensors for prefix: ") + p
                )
            var start = self._turbo.index_starts[prefix_idx]
            var end = start + self._turbo.index_lengths[prefix_idx]
            var tensors = List[_ResidentSquareqTensor]()
            var block_bytes = 0
            for ni in range(start, end):
                var nm = self._turbo.index_names[ni].copy()
                var is_weight = nm.endswith(String(".weight"))
                var base = String(nm.removesuffix(String(".weight")))
                var qname = base + String(".qweight")
                var q8name = base + String(".q8weight")
                var use_w8 = is_weight and (not sc.has_tensor(qname)) and sc.has_tensor(q8name)
                if use_w8:
                    qname = q8name
                if is_weight and (sc.has_tensor(qname)):
                    var q = Tensor.from_view_raw(sc.tensor_view(qname), ctx)
                    var s = Tensor.from_view(
                        sc.tensor_view(
                            base + (String(".w8scale") if use_w8 else String(".wscales"))
                        ), ctx
                    )
                    var ld = Tensor.from_view(
                        sc.tensor_view(base + String(".lora_down")), ctx
                    )
                    var lu = Tensor.from_view(
                        sc.tensor_view(base + String(".lora_up")), ctx
                    )
                    var out_f = q.shape()[0]
                    var in_f = q.shape()[1] * 2
                    if use_w8:
                        in_f = q.shape()[1]   # i8 is unpacked [out, in]
                    block_bytes += (
                        q.numel() * q.dtype().byte_size()
                        + s.numel() * s.dtype().byte_size()
                        + ld.numel() * ld.dtype().byte_size()
                        + lu.numel() * lu.dtype().byte_size()
                    )
                    tensors.append(
                        _ResidentSquareqTensor(
                            nm, True, TArc(q^), TArc(s^), TArc(ld^), TArc(lu^),
                            in_f, out_f, use_w8,
                        )
                    )
                else:
                    # small/passthrough: prefer the sidecar copy (bit-identical
                    # to the base by builder contract), fall back to the base.
                    var t: Tensor
                    if sc.has_tensor(nm):
                        t = Tensor.from_view_as_bf16(sc.tensor_view(nm), ctx)
                    else:
                        t = Tensor.from_view_as_bf16(
                            self._turbo.sharded.tensor_view(nm), ctx
                        )
                    block_bytes += t.numel() * t.dtype().byte_size()
                    var arc = TArc(t^)
                    tensors.append(
                        _ResidentSquareqTensor(
                            nm, False, arc.copy(), arc.copy(), arc.copy(), arc,
                            0, 0,
                        )
                    )
            if used + block_bytes > budget_bytes:
                break  # plan-order contiguous pin; caller enforces pinned==count
            self._squareq_prefixes.append(p)
            self._squareq_blocks.append(tensors^)
            used += block_bytes
            pinned += 1
            ctx.synchronize()  # fence this block's transients before the next
        print(
            "  squareq_w4-resident: pinned", pinned, "blocks,",
            used, "bytes packed (reconstruct-on-await)",
        )
        return pinned

    def pin_residents_squareq_nvfp4(
        mut self, sidecar_dir: String, budget_bytes: Int, ctx: DeviceContext
    ) raises -> Int:
        """Pin plan blocks (plan order) as squareq_nvfp4-resident from the
        PREBUILT nvfp4 sidecar (scripts/squareq_build_slab.py --format nvfp4
        output: sharded safetensors holding `<base>.nvq/.nvs/.nvg/.lora_down/
        .lora_up` per quantized linear, everything else passthrough). NEVER
        quantizes at load.

        For each base-checkpoint tensor name in the block:
          - `<base>.weight` with a sidecar `<base>.nvq`: pin the 4 packed
            device tensors VERBATIM (nvq/nvs raw U8; lora_down/lora_up BF16)
            plus the 1-elem F32 `nvg` tensor, whose HOST value is read ONCE
            here (to_host at pin time — it rides cublasLt alpha per fp4 GEMM).
            await_block then returns BOTH the reconstructed BF16 W_hat under
            the weight's own name (squareq_nvfp4_reconstruct_weight — backward
            + any non-wired consumer stays correct) AND the payload under
            resident_nvfp4_key(name, part) so the Klein block builders can
            assemble the FORWARD-ONLY native-fp4 payload.
          - anything else: pinned resident BF16 from the sidecar's passthrough
            copy (falls back to the base checkpoint if the sidecar lacks it).

        Same MJ-1065 contract as fp8/squareq_w4: the caller must require
        pinned == block_count. One sync per block bounds transients.
        """
        var sc = ShardedSafeTensors.open(sidecar_dir)
        var used = 0
        var pinned = 0
        for i in range(self._plan.count()):
            self._assert_raw_copy_dtype_safe(i)
            var p = self._plan.normalized_prefix(i)
            if (
                self._nvfp4_slot(p) >= 0
                or self._squareq_slot(p) >= 0
                or self._fp8_slot(p) >= 0
                or self._resident_slot(p) >= 0
            ):
                continue
            var prefix_idx = -1
            for j in range(len(self._turbo.index_prefixes)):
                if self._turbo.index_prefixes[j] == p:
                    prefix_idx = j
                    break
            if prefix_idx < 0:
                raise Error(
                    String("pin_residents_squareq_nvfp4: no tensors for prefix: ") + p
                )
            var start = self._turbo.index_starts[prefix_idx]
            var end = start + self._turbo.index_lengths[prefix_idx]
            var tensors = List[_ResidentNvfp4Tensor]()
            var block_bytes = 0
            for ni in range(start, end):
                var nm = self._turbo.index_names[ni].copy()
                var is_weight = nm.endswith(String(".weight"))
                var base = String(nm.removesuffix(String(".weight")))
                var qname = base + String(".nvq")
                if is_weight and sc.has_tensor(qname):
                    var nvq = Tensor.from_view_raw(sc.tensor_view(qname), ctx)
                    var nvs = Tensor.from_view_raw(
                        sc.tensor_view(base + String(".nvs")), ctx
                    )
                    var ld = Tensor.from_view(
                        sc.tensor_view(base + String(".lora_down")), ctx
                    )
                    var lu = Tensor.from_view(
                        sc.tensor_view(base + String(".lora_up")), ctx
                    )
                    var nvg_t = Tensor.from_view(
                        sc.tensor_view(base + String(".nvg")), ctx
                    )
                    # HOST global scale read ONCE at pin time (never per step).
                    var nvg_host = nvg_t.to_host(ctx)
                    if len(nvg_host) != 1:
                        raise Error(
                            String("pin_residents_squareq_nvfp4: nvg numel != 1: ")
                            + qname
                        )
                    var nvg = nvg_host[0]
                    var out_f = nvq.shape()[0]
                    var in_f = nvq.shape()[1] * 2
                    block_bytes += (
                        nvq.numel() * nvq.dtype().byte_size()
                        + nvs.numel() * nvs.dtype().byte_size()
                        + ld.numel() * ld.dtype().byte_size()
                        + lu.numel() * lu.dtype().byte_size()
                        + nvg_t.numel() * nvg_t.dtype().byte_size()
                    )
                    tensors.append(
                        _ResidentNvfp4Tensor(
                            nm, True, TArc(nvq^), TArc(nvs^), TArc(ld^),
                            TArc(lu^), TArc(nvg_t^), nvg, in_f, out_f,
                        )
                    )
                else:
                    # small/passthrough: prefer the sidecar copy (bit-identical
                    # to the base by builder contract), fall back to the base.
                    var t: Tensor
                    if sc.has_tensor(nm):
                        t = Tensor.from_view_as_bf16(sc.tensor_view(nm), ctx)
                    else:
                        t = Tensor.from_view_as_bf16(
                            self._turbo.sharded.tensor_view(nm), ctx
                        )
                    block_bytes += t.numel() * t.dtype().byte_size()
                    var arc = TArc(t^)
                    tensors.append(
                        _ResidentNvfp4Tensor(
                            nm, False, arc.copy(), arc.copy(), arc.copy(),
                            arc.copy(), arc, 0.0, 0, 0,
                        )
                    )
            if used + block_bytes > budget_bytes:
                break  # plan-order contiguous pin; caller enforces pinned==count
            self._nvfp4_prefixes.append(p)
            self._nvfp4_blocks.append(tensors^)
            used += block_bytes
            pinned += 1
            ctx.synchronize()  # fence this block's transients before the next
        # P2-A shape-dummy: size = largest quantized weight in bf16.
        var max_bytes = 2
        for bi2 in range(len(self._nvfp4_blocks)):
            for ti2 in range(len(self._nvfp4_blocks[bi2])):
                ref t2 = self._nvfp4_blocks[bi2][ti2]
                if t2.is_quant and t2.out_f * t2.in_f * 2 > max_bytes:
                    max_bytes = t2.out_f * t2.in_f * 2
        if len(self._nvfp4_dummy) == 0:
            self._nvfp4_dummy.append(
                ctx.enqueue_create_buffer[DType.uint8](max_bytes)
            )
        print(
            "  squareq_nvfp4-resident: pinned", pinned, "blocks,",
            used, "bytes packed (native-fp4 fwd payload + W_hat-on-await)",
        )
        return pinned

    def pin_residents_fp8_prequantized(
        mut self, budget_bytes: Int, ctx: DeviceContext
    ) raises -> Int:
        """Pin plan blocks (plan order) as fp8-resident from an ALREADY-FP8
        checkpoint — the Bernini-R `serenity_fp8_e4m3_*` per-block cache, whose
        big 2-D weights are stored as `F8_E4M3 [out,in]` beside a sibling
        `<key>.__fp8_scale` `F32 [out]`, with small tensors BF16.

        WHY THIS EXISTS (do not use `pin_residents_fp8` on such a checkpoint):
        that sibling QUANTIZES from BF16/FP16 source tensors.
        Against an already-fp8 cache every big weight FAILS that test and falls to
        the small-tensor branch, which calls `from_view_as_bf16` on E4M3 bytes —
        reinterpreting 1-byte fp8 as 2-byte bf16. That is silent corruption, not
        an error. Here the bytes and the scale are pinned VERBATIM
        (`from_view_raw` preserves F8_E4M3; the F32 scale rides along), so
        `await_block`'s fp8-resident branch dequants them with the SAME
        `fp8_e4m3_dequant_perrow_to_bf16` the streamed path uses via
        `_block_bf16_dev` → the resident Block is BIT-IDENTICAL to the streamed one.

        `budget_bytes` caps the resident bytes and pinning is PLAN-ORDER
        CONTIGUOUS: blocks [0, pinned) are resident and the remainder keep
        streaming, so a partial pin is legal here (unlike MJ-1065's all-or-nothing
        fp8 mode). `prefetch_with_ctx` early-returns for every pinned block, so
        each one converts a per-block staging stall into a free lookup. Returns
        the number of blocks pinned."""
        var used = 0
        var pinned = 0
        var scale_suffix = String(".__fp8_scale")
        for i in range(self._plan.count()):
            self._assert_raw_copy_dtype_safe(i)
            var p = self._plan.normalized_prefix(i)
            if self._fp8_slot(p) >= 0 or self._resident_slot(p) >= 0:
                continue
            var prefix_idx = -1
            for j in range(len(self._turbo.index_prefixes)):
                if self._turbo.index_prefixes[j] == p:
                    prefix_idx = j
                    break
            if prefix_idx < 0:
                raise Error(
                    String("pin_residents_fp8_prequantized: no tensors for prefix: ") + p
                )
            var start = self._turbo.index_starts[prefix_idx]
            var end = start + self._turbo.index_lengths[prefix_idx]
            var tensors = List[_ResidentFp8Tensor]()
            var block_bytes = 0
            for ni in range(start, end):
                var nm = self._turbo.index_names[ni].copy()
                if nm.endswith(scale_suffix):
                    continue          # consumed with its parent weight below
                var tv = self._turbo.sharded.tensor_view(nm)
                if tv.dtype == STDtype.F8_E4M3:
                    var sname = nm + scale_suffix
                    var stv = self._turbo.sharded.tensor_view(sname)
                    var bytes = Tensor.from_view_raw(tv, ctx)   # keeps F8_E4M3
                    var scale = Tensor.from_view(stv, ctx)      # F32 [out]
                    block_bytes += (
                        bytes.numel() * bytes.dtype().byte_size()
                        + scale.numel() * scale.dtype().byte_size()
                    )
                    tensors.append(
                        _ResidentFp8Tensor(nm, True, TArc(bytes^), TArc(scale^))
                    )
                else:
                    var t = Tensor.from_view_as_bf16(tv, ctx)
                    block_bytes += t.numel() * t.dtype().byte_size()
                    var arc = TArc(t^)
                    tensors.append(
                        _ResidentFp8Tensor(nm, False, arc.copy(), arc)
                    )
            if used + block_bytes > budget_bytes:
                break                 # partial pin: the rest keep streaming
            self._fp8_prefixes.append(p)
            self._fp8_blocks.append(tensors^)
            used += block_bytes
            pinned += 1
            ctx.synchronize()
        return pinned

    def pin_residents_fp8_host(
        mut self, budget_bytes: Int, ctx: DeviceContext
    ) raises -> Int:
        """fp8_e4m3_host mode: quantize each block ONCE (same E4M3 + per-row F32
        scale as pin_residents_fp8) but hold the result PINNED IN HOST RAM, not
        on device — for models whose fp8 footprint does not fit device-resident
        next to training state (qwenimage: fp8 ~20GB, card 24GB, measured OOM
        2026-07-04). await_block H2Ds the fp8 bytes + scale and dequants — HALF
        the per-step PCIe of bf16 streaming, ZERO disk (MJ-1065). `budget_bytes`
        caps HOST pinned bytes; caller must require pinned==block_count.

        BF16 and FP16 checkpoint weights take the same explicit BF16
        quantization-input path. Checkpoint mmap pages are clean, replaceable
        source data after a block's
        FP8/BF16 host buffers have been fenced.  Drop those pages at every block
        boundary instead of retaining the whole BF16 checkpoint beside its FP8
        copy until this function returns.  This keeps one-time model admission
        near destination-store size rather than source + destination size; it
        does not alter the pinned bytes or the zero-disk denoise contract."""
        var used = 0
        var pinned = 0
        for i in range(self._plan.count()):
            self._assert_raw_copy_dtype_safe(i)
            var p = self._plan.normalized_prefix(i)
            if (
                self._fp8h_slot(p) >= 0 or self._fp8_slot(p) >= 0
                or self._resident_slot(p) >= 0
            ):
                continue
            var prefix_idx = -1
            for j in range(len(self._turbo.index_prefixes)):
                if self._turbo.index_prefixes[j] == p:
                    prefix_idx = j
                    break
            if prefix_idx < 0:
                raise Error(
                    String("pin_residents_fp8_host: no tensors for prefix: ") + p
                )
            var start = self._turbo.index_starts[prefix_idx]
            var end = start + self._turbo.index_lengths[prefix_idx]
            var tensors = List[_HostFp8Tensor]()
            var block_bytes = 0
            for ni in range(start, end):
                var nm = self._turbo.index_names[ni].copy()
                var tv = self._turbo.sharded.tensor_view(nm)
                var sh = tv.shape.copy()
                var big_2d = (
                    (tv.dtype == STDtype.BF16 or tv.dtype == STDtype.F16)
                    and len(sh) == 2
                    and sh[0] * sh[1] >= _FP8_MIN_ELEMS
                )
                if big_2d:
                    var w_bf = Tensor.from_view_as_bf16(tv, ctx)
                    var scale = fp8_e4m3_rowscale(w_bf, ctx)
                    var bytes = fp8_e4m3_encode_perrow(w_bf, scale, ctx)
                    var bh = ctx.enqueue_create_host_buffer[DType.uint8](
                        bytes.nbytes()
                    )
                    var sh_h = ctx.enqueue_create_host_buffer[DType.uint8](
                        scale.nbytes()
                    )
                    ctx.enqueue_copy(bh, bytes.buf)
                    ctx.enqueue_copy(sh_h, scale.buf)
                    ctx.synchronize()  # D2H done + fences transient device tensors
                    block_bytes += bytes.nbytes() + scale.nbytes()
                    tensors.append(_HostFp8Tensor(
                        nm, True, True, Float32(1.0),
                        HArc(bh^), bytes.nbytes(), bytes.shape(), bytes.dtype(),
                        HArc(sh_h^), scale.nbytes(), scale.shape(),
                    ))
                else:
                    var t = Tensor.from_view_as_bf16(tv, ctx)
                    var bh = ctx.enqueue_create_host_buffer[DType.uint8](
                        t.nbytes()
                    )
                    var dummy = ctx.enqueue_create_host_buffer[DType.uint8](1)
                    ctx.enqueue_copy(bh, t.buf)
                    ctx.synchronize()
                    block_bytes += t.nbytes()
                    tensors.append(_HostFp8Tensor(
                        nm, False, False, Float32(1.0),
                        HArc(bh^), t.nbytes(), t.shape(), t.dtype(),
                        HArc(dummy^), 1, List[Int](),
                    ))
            if used + block_bytes > budget_bytes:
                self._turbo.sharded.release_to_os()
                break  # plan-order contiguous; caller enforces pinned==count
            self._fp8h_prefixes.append(p)
            self._fp8h_blocks.append(tensors^)
            used += block_bytes
            pinned += 1
            # Every D2H above is synchronized. No live tensor view still needs
            # this block's checkpoint pages; future blocks fault only their own
            # immutable ranges back in as needed.
            self._turbo.sharded.release_to_os()
        return pinned

    def fp8_host_cache_valid(self, cache_path: String) raises -> Bool:
        """Validate a generic E4M3-row-scale sidecar against this source/plan."""
        var source_meta = _stat_size_mtime(self._source_path)
        if source_meta[0] < 0:
            return False
        var cache: ShardedSafeTensors
        try:
            cache = ShardedSafeTensors.open(cache_path)
        except:
            return False
        if _cache_i64(
            cache, String("__serenity_fp8_host.version"), -1
        ) != _FP8_HOST_CACHE_VERSION:
            return False
        if _cache_i64(
            cache, String("__serenity_fp8_host.source_size"), -1
        ) != source_meta[0]:
            return False
        if _cache_i64(
            cache, String("__serenity_fp8_host.source_mtime"), -1
        ) != source_meta[1]:
            return False
        if _cache_i64(
            cache, String("__serenity_fp8_host.blocks"), -1
        ) != self._plan.count():
            return False

        for i in range(self._plan.count()):
            var p = self._plan.normalized_prefix(i)
            var prefix_idx = -1
            for j in range(len(self._turbo.index_prefixes)):
                if self._turbo.index_prefixes[j] == p:
                    prefix_idx = j
                    break
            if prefix_idx < 0:
                return False
            var start = self._turbo.index_starts[prefix_idx]
            var end = start + self._turbo.index_lengths[prefix_idx]
            for ni in range(start, end):
                var nm = self._turbo.index_names[ni].copy()
                if not cache.has_tensor(nm):
                    return False
                var source_info = self._turbo.sharded.tensor_info(nm)
                var cache_info = cache.tensor_info(nm)
                var big_2d = (
                    (
                        source_info.dtype == STDtype.BF16
                        or source_info.dtype == STDtype.F16
                    )
                    and len(source_info.shape) == 2
                    and source_info.shape[0] * source_info.shape[1] >= _FP8_MIN_ELEMS
                )
                if big_2d:
                    if cache_info.dtype != STDtype.F8_E4M3:
                        return False
                    if not _shape_equal(cache_info.shape, source_info.shape):
                        return False
                    var scale_name = nm + String(_FP8_CACHE_SCALE_SUFFIX)
                    if not cache.has_tensor(scale_name):
                        return False
                    var scale_info = cache.tensor_info(scale_name)
                    if (
                        scale_info.dtype != STDtype.F32
                        or len(scale_info.shape) != 1
                        or scale_info.shape[0] != source_info.shape[0]
                    ):
                        return False
                else:
                    if cache_info.dtype != STDtype.BF16:
                        return False
                    if not _shape_equal(cache_info.shape, source_info.shape):
                        return False
        return True

    def save_fp8_host_cache(
        self, cache_path: String, ctx: DeviceContext
    ) raises:
        """Atomically persist the exact pinned FP8 host store without VRAM."""
        self.require_all_blocks_memory_resident()
        if len(self._fp8h_prefixes) != self._plan.count():
            raise Error(
                "save_fp8_host_cache requires every plan block in fp8-host storage"
            )
        var source_meta = _stat_size_mtime(self._source_path)
        if source_meta[0] < 0:
            raise Error("save_fp8_host_cache cannot stat source checkpoint")

        var names = List[String]()
        var descs = List[HostBufferTensorDesc]()

        # Cache identity tensors. HostBuffer ownership rides in each descriptor.
        var meta_values: List[Int] = [
            _FP8_HOST_CACHE_VERSION,
            source_meta[0],
            source_meta[1],
            self._plan.count(),
        ]
        var meta_names: List[String] = [
            String("__serenity_fp8_host.version"),
            String("__serenity_fp8_host.source_size"),
            String("__serenity_fp8_host.source_mtime"),
            String("__serenity_fp8_host.blocks"),
        ]
        for mi in range(len(meta_values)):
            var h = ctx.enqueue_create_host_buffer[DType.uint8](8)
            h.unsafe_ptr().bitcast[Int64]()[0] = Int64(meta_values[mi])
            names.append(meta_names[mi].copy())
            descs.append(HostBufferTensorDesc(
                STDtype.I64, [1], HArc(h^), 8
            ))

        for hslot in range(len(self._fp8h_blocks)):
            for ti in range(len(self._fp8h_blocks[hslot])):
                ref rt = self._fp8h_blocks[hslot][ti]
                names.append(rt.name.copy())
                descs.append(HostBufferTensorDesc(
                    rt.bytes_dtype,
                    rt.bytes_shape.copy(),
                    rt.bytes_h.copy(),
                    rt.bytes_nbytes,
                ))
                if rt.is_fp8:
                    if not rt.per_row_scale:
                        raise Error(
                            "save_fp8_host_cache requires per-row FP8 scales"
                        )
                    names.append(rt.name + String(_FP8_CACHE_SCALE_SUFFIX))
                    descs.append(HostBufferTensorDesc(
                        STDtype.F32,
                        rt.scale_shape.copy(),
                        rt.scale_h.copy(),
                        rt.scale_nbytes,
                    ))
        save_safetensors_host_buffers(names, descs, cache_path)


    def promote_fp8_host_to_device(
        mut self,
        budget_bytes: Int,
        ctx: DeviceContext,
        phase_evictable: Bool = False,
        min_free_bytes: Int = 0,
        dequantize_on_promotion: Bool = False,
    ) raises -> Int:
        """Promote host FP8 blocks using the ownership policy the backend needs.

        Persistent workers such as Flux use DeviceContext residency, which is
        already measured and never needs same-process eviction. Phase-separated
        workers such as Chroma opt into explicit VMM so decode can physically
        unmap the prefix and later restore it from pinned host memory.

        `min_free_bytes` is a fail-closed live-VRAM floor. The loader clamps the
        caller's snapshot budget on entry and rechecks the floor before every
        physical block allocation, so concurrent VRAM use cannot make a stale
        budget consume the backend's denoise/decode headroom."""
        # Persistent backends may call this again after the resident prefix has
        # already consumed the caller's nominal free-memory budget. Preserve
        # that proven prefix; admission governs NEW physical allocations only.
        if len(self._fp8_prefixes) > 0:
            return len(self._fp8_prefixes)
        if budget_bytes <= 0:
            return 0
        var live_free = cu_mem_get_info().free_bytes
        var safe_budget = budget_bytes
        if min_free_bytes > 0:
            if live_free <= min_free_bytes:
                return 0
            var live_budget = live_free - min_free_bytes
            if safe_budget > live_budget:
                safe_budget = live_budget
        if phase_evictable:
            if dequantize_on_promotion:
                return self._promote_fp8_host_to_bf16_device_vmm(
                    safe_budget, ctx, min_free_bytes
                )
            return self._promote_fp8_host_to_device_vmm(
                safe_budget, ctx, min_free_bytes
            )
        return self._promote_fp8_host_to_device_cached(
            safe_budget, ctx, min_free_bytes
        )

    def _promote_fp8_host_to_device_cached(
        mut self, budget_bytes: Int, ctx: DeviceContext, min_free_bytes: Int
    ) raises -> Int:
        """Proven persistent DeviceContext path for backends that never evict."""
        if len(self._fp8_prefixes) > 0:
            return len(self._fp8_prefixes)
        var used = 0
        var promoted = 0
        for hslot in range(len(self._fp8h_prefixes)):
            var block_bytes = 0
            for t in range(len(self._fp8h_blocks[hslot])):
                ref rt = self._fp8h_blocks[hslot][t]
                block_bytes += rt.bytes_nbytes
                if rt.is_fp8:
                    if not rt.per_row_scale:
                        raise Error(
                            "promote_fp8_host_to_device requires per-row FP8 scales"
                        )
                    block_bytes += rt.scale_nbytes
            if used + block_bytes > budget_bytes:
                break
            # The caller's budget is only a snapshot. Recheck immediately
            # before this block's real DeviceContext allocations.
            var live_free = cu_mem_get_info().free_bytes
            if live_free < block_bytes + min_free_bytes:
                break

            var tensors = List[_ResidentFp8Tensor]()
            for t in range(len(self._fp8h_blocks[hslot])):
                ref rt = self._fp8h_blocks[hslot][t]
                var dbuf = ctx.enqueue_create_buffer[DType.uint8](rt.bytes_nbytes)
                ctx.enqueue_copy(dbuf, rt.bytes_h[])
                var a = Tensor(
                    dbuf^, rt.bytes_shape.copy(), rt.bytes_dtype
                )
                var a_arc = TArc(a^)
                if rt.is_fp8:
                    var sbuf = ctx.enqueue_create_buffer[DType.uint8](
                        rt.scale_nbytes
                    )
                    ctx.enqueue_copy(sbuf, rt.scale_h[])
                    var scale = Tensor(
                        sbuf^, rt.scale_shape.copy(), STDtype.F32
                    )
                    tensors.append(_ResidentFp8Tensor(
                        rt.name.copy(), True, a_arc, TArc(scale^)
                    ))
                else:
                    tensors.append(_ResidentFp8Tensor(
                        rt.name.copy(), False, a_arc.copy(), a_arc
                    ))
            self._fp8_prefixes.append(self._fp8h_prefixes[hslot].copy())
            self._fp8_blocks.append(tensors^)
            used += block_bytes
            promoted += 1
            ctx.synchronize()
        return promoted

    def _promote_fp8_host_to_bf16_device_vmm(
        mut self, budget_bytes: Int, ctx: DeviceContext, min_free_bytes: Int
    ) raises -> Int:
        """Decode host E4M3 weights once into evictable BF16 VMM blocks.

        This is the compute path for pre-FP8 GPUs: it preserves the exact
        E4M3-decode trajectory while removing every per-step dequantization.
        The host FP8 store remains authoritative, and phase teardown unmaps
        these larger BF16 blocks before VAE decode."""
        if len(self._fp8_prefixes) > 0:
            return len(self._fp8_prefixes)

        # Two reusable FP8 source slabs bound promotion scratch. They are
        # discarded with the other phase-local staging before decode.
        self._f8h_ensure(ctx)
        var granularity = cu_mem_get_allocation_granularity(0)
        var block_sizes = List[Int]()
        var used = 0
        for hslot in range(len(self._fp8h_prefixes)):
            var block_bytes = 0
            for t in range(len(self._fp8h_blocks[hslot])):
                ref rt = self._fp8h_blocks[hslot][t]
                if rt.is_fp8:
                    if not rt.per_row_scale:
                        raise Error(
                            "BF16 VMM promotion requires per-row FP8 scales"
                        )
                    block_bytes += rt.bytes_nbytes * STDtype.BF16.byte_size()
                else:
                    if rt.bytes_dtype != STDtype.BF16:
                        raise Error(
                            "BF16 VMM promotion requires BF16 passthrough tensors"
                        )
                    block_bytes += rt.bytes_nbytes
            var rounded = (
                (block_bytes + granularity - 1) // granularity
            ) * granularity
            if used + rounded > budget_bytes:
                break
            block_sizes.append(block_bytes)
            used += rounded

        if len(block_sizes) == 0:
            return 0

        var vmm = VmmModelHandle.create(
            String("turbo-fp8-host-bf16-promotions"),
            block_sizes.copy(),
            STDtype.U8,
            0,
        )
        self._fp8_vmm.append(ArcPointer(vmm^))

        var promoted = 0
        try:
            for hslot in range(len(block_sizes)):
                var physical_bytes = self._fp8_vmm[0][].block_reserved_bytes(
                    hslot
                )
                var live_free = cu_mem_get_info().free_bytes
                if live_free < physical_bytes + min_free_bytes:
                    break
                var block_ptr = self._fp8_vmm[0][].ensure_block_resident(hslot)

                # Reuse one bounded FP8 slab as the decode source for this
                # block; the destination lives directly in VMM.
                self._f8h_prefetch(hslot, ctx)
                var okey = self._f8h_key(self._fp8h_prefixes[hslot])
                var oslot = -1
                if self._f8h[].slot_key[0] == okey:
                    oslot = 0
                elif self._f8h[].slot_key[1] == okey:
                    oslot = 1
                if oslot < 0:
                    raise Error(
                        "BF16 VMM promotion could not stage FP8 host block"
                    )
                ctx.stream().enqueue_wait_for(self._f8h[].evs[oslot])
                self._f8h[].active = oslot
                var src_offsets = self._f8h_layout(hslot)
                var offset = 0
                var tensors = List[_ResidentFp8Tensor]()
                for t in range(len(self._fp8h_blocks[hslot])):
                    ref rt = self._fp8h_blocks[hslot][t]
                    var dst_nbytes = (
                        rt.bytes_nbytes * STDtype.BF16.byte_size()
                        if rt.is_fp8 else rt.bytes_nbytes
                    )
                    var dst_raw = BytePtr(
                        unsafe_from_address=Int(block_ptr) + offset
                    )
                    var dst_buf = DeviceBuffer[DType.uint8](
                        ctx, dst_raw, dst_nbytes, owning=False
                    )
                    var dst = Tensor(
                        dst_buf^, rt.bytes_shape.copy(), STDtype.BF16
                    )
                    if rt.is_fp8:
                        var wsub = self._f8h[].devs[oslot].create_sub_buffer[
                            DType.uint8
                        ](src_offsets[2 * t], rt.bytes_nbytes)
                        var wt = Tensor(
                            wsub^, rt.bytes_shape.copy(), rt.bytes_dtype
                        )
                        var ssub = self._f8h[].devs[oslot].create_sub_buffer[
                            DType.uint8
                        ](src_offsets[2 * t + 1], rt.scale_nbytes)
                        var st = Tensor(
                            ssub^, rt.scale_shape.copy(), STDtype.F32
                        )
                        fp8_e4m3_dequant_perrow_to_bf16_into(
                            wt, st, dst, ctx
                        )
                    else:
                        ctx.enqueue_copy(dst.buf, rt.bytes_h[])
                    var dst_arc = TArc(dst^)
                    tensors.append(_ResidentFp8Tensor(
                        rt.name.copy(), False, dst_arc.copy(), dst_arc
                    ))
                    offset += dst_nbytes
                if offset != block_sizes[hslot]:
                    raise Error(
                        String("BF16 VMM promotion packed ") + String(offset)
                        + String(" bytes != selected ")
                        + String(block_sizes[hslot])
                    )
                ctx.synchronize()
                self._fp8_prefixes.append(
                    self._fp8h_prefixes[hslot].copy()
                )
                self._fp8_blocks.append(tensors^)
                self._fp8_vmm[0][].mark_block_populated(hslot)
                promoted += 1
            if promoted == 0:
                self._discard_fp8_vmm_promotions()
        except e:
            self._turbo.copy_stream.synchronize()
            ctx.synchronize()
            self._discard_fp8_vmm_promotions()
            raise e^
        return promoted

    def _promote_fp8_host_to_device_vmm(
        mut self, budget_bytes: Int, ctx: DeviceContext, min_free_bytes: Int
    ) raises -> Int:
        """Promote a plan-order prefix of the existing per-row FP8 host store
        to device residency without rereading or requantizing the checkpoint.

        The host copy remains authoritative, so callers may discard these
        promotions before a high-headroom phase (for example VAE decode) and
        recreate them for the next denoise with only pinned-RAM H2D traffic.
        `await_block` already checks `_fp8_slot` before `_fp8h_slot`, making a
        promoted block a no-copy device hit while the unpromoted tail continues
        through the overlapped host path.  This method deliberately refuses to
        mix with checkpoint-created FP8 residents: all entries in
        `_fp8_prefixes` must be promotions from this host store."""
        if len(self._fp8_prefixes) > 0:
            return len(self._fp8_prefixes)
        # Select the prefix using PHYSICAL VMM bytes, including allocation-
        # granularity rounding. This keeps the caller's reserve honest.
        var granularity = cu_mem_get_allocation_granularity(0)
        var block_sizes = List[Int]()
        var used = 0
        for hslot in range(len(self._fp8h_prefixes)):
            var block_bytes = 0
            for t in range(len(self._fp8h_blocks[hslot])):
                ref rt = self._fp8h_blocks[hslot][t]
                block_bytes += rt.bytes_nbytes
                if rt.is_fp8:
                    if not rt.per_row_scale:
                        raise Error(
                            "promote_fp8_host_to_device requires per-row FP8 scales"
                        )
                    block_bytes += rt.scale_nbytes
            var rounded = (
                (block_bytes + granularity - 1) // granularity
            ) * granularity
            if used + rounded > budget_bytes:
                break
            block_sizes.append(block_bytes)
            used += rounded

        if len(block_sizes) == 0:
            return 0

        var vmm = VmmModelHandle.create(
            String("turbo-fp8-host-promotions"),
            block_sizes.copy(),
            STDtype.U8,
            0,
        )
        self._fp8_vmm.append(ArcPointer(vmm^))

        var promoted = 0
        try:
            for hslot in range(len(block_sizes)):
                var physical_bytes = self._fp8_vmm[0][].block_reserved_bytes(
                    hslot
                )
                # cuMemCreate is the physical allocation boundary. Recheck the
                # live floor immediately before it, not merely at selection.
                var live_free = cu_mem_get_info().free_bytes
                if live_free < physical_bytes + min_free_bytes:
                    break
                var block_ptr = self._fp8_vmm[0][].ensure_block_resident(hslot)
                var offset = 0
                var tensors = List[_ResidentFp8Tensor]()
                for t in range(len(self._fp8h_blocks[hslot])):
                    ref rt = self._fp8h_blocks[hslot][t]
                    _h2d_dma_copy(
                        block_ptr + UInt64(offset),
                        rt.bytes_h[].unsafe_ptr(),
                        rt.bytes_nbytes,
                        self._turbo.copy_stream,
                    )
                    var raw = BytePtr(
                        unsafe_from_address=Int(block_ptr) + offset
                    )
                    var dbuf = DeviceBuffer[DType.uint8](
                        ctx, raw, rt.bytes_nbytes, owning=False
                    )
                    var a = Tensor(
                        dbuf^, rt.bytes_shape.copy(), rt.bytes_dtype
                    )
                    var a_arc = TArc(a^)
                    offset += rt.bytes_nbytes
                    if rt.is_fp8:
                        var scale_raw = BytePtr(
                            unsafe_from_address=Int(block_ptr) + offset
                        )
                        _h2d_dma_copy(
                            block_ptr + UInt64(offset),
                            rt.scale_h[].unsafe_ptr(),
                            rt.scale_nbytes,
                            self._turbo.copy_stream,
                        )
                        var sbuf = DeviceBuffer[DType.uint8](
                            ctx, scale_raw, rt.scale_nbytes, owning=False
                        )
                        var scale = Tensor(
                            sbuf^, rt.scale_shape.copy(), STDtype.F32
                        )
                        tensors.append(_ResidentFp8Tensor(
                            rt.name.copy(), True, a_arc, TArc(scale^)
                        ))
                        offset += rt.scale_nbytes
                    else:
                        tensors.append(_ResidentFp8Tensor(
                            rt.name.copy(), False, a_arc.copy(), a_arc
                        ))
                if offset != block_sizes[hslot]:
                    raise Error(
                        String("promote_fp8_host_to_device: packed ")
                        + String(offset) + String(" bytes != selected ")
                        + String(block_sizes[hslot]) + String(" for block ")
                        + String(hslot)
                    )
                self._turbo.copy_stream.synchronize()
                ctx.synchronize()
                self._fp8_prefixes.append(
                    self._fp8h_prefixes[hslot].copy()
                )
                self._fp8_blocks.append(tensors^)
                self._fp8_vmm[0][].mark_block_populated(hslot)
                promoted += 1
            if promoted == 0:
                # The VA reservation is cheap but must not survive a failed
                # live-headroom check, otherwise a later retry would append a
                # second handle while `_fp8_prefixes` is still empty.
                self._discard_fp8_vmm_promotions()
        except e:
            # The caller synchronized before promotion. Any copy already issued
            # above is fenced per completed block; explicitly tear down partial
            # VMM state so a failed promotion never strands physical VRAM.
            self._turbo.copy_stream.synchronize()
            ctx.synchronize()
            self._discard_fp8_vmm_promotions()
            raise e^
        return promoted

    def fp8_device_block_count(self) -> Int:
        return len(self._fp8_prefixes)

    def _discard_fp8_vmm_promotions(mut self) raises:
        """Drop non-owning tensor views, then explicitly unmap/release VMM."""
        self._fp8_blocks = List[List[_ResidentFp8Tensor]]()
        self._fp8_prefixes = List[String]()
        if len(self._fp8_vmm) == 1:
            for i in range(self._fp8_vmm[0][].block_count()):
                if self._fp8_vmm[0][].block_refcount(i) > 0:
                    self._fp8_vmm[0][].release_block(i)
                if self._fp8_vmm[0][].is_block_resident(i):
                    self._fp8_vmm[0][].evict_block(i)
            self._fp8_vmm[0][].destroy()
        self._fp8_vmm = List[ArcPointer[VmmModelHandle]]()

    def discard_fp8_host_promotions(mut self) raises:
        """Drop device FP8 promotions while preserving the complete host store.

        The caller must synchronize first.  This is phase residency, not model
        eviction: the next denoise can call `promote_fp8_host_to_device` and
        restore the fast path without checkpoint I/O or quantization."""
        self._discard_fp8_vmm_promotions()

    def pin_residents_fp8_host_prequantized(
        mut self, budget_bytes: Int, ctx: DeviceContext
    ) raises -> Int:
        """HOST-pinned sibling of `pin_residents_fp8_prequantized` for an
        ALREADY-FP8 checkpoint (E4M3 [out,in] + `<key>.__fp8_scale` F32 [out]).

        Same corruption trap as the device path: `pin_residents_fp8_host`
        quantizes FROM BF16 and gates on `tv.dtype == STDtype.BF16`, so against an
        already-fp8 cache every big weight falls to the small-tensor branch and is
        read by `from_view_as_bf16` — 1-byte fp8 reinterpreted as 2-byte bf16.
        This variant copies the on-disk bytes VERBATIM into pinned host buffers.

        WHY: the per-block staging cost is a ~50 ms HOST memcpy of the block's
        bytes out of the mmap page cache into the loader's pinned staging slab
        (measured: `prefetch_next_with_ctx` 50.2 ms/block, ~350 MB at ~7 GB/s).
        Blocks pinned HERE skip that entirely — `await_block`'s fp8h branch H2Ds
        straight from already-pinned host memory and dequants with the SAME
        `fp8_e4m3_dequant_perrow_to_bf16` the streamed and device-resident paths
        use, so the Block is BIT-IDENTICAL to both.

        Use AFTER `pin_residents_fp8_prequantized`: this skips any prefix already
        device-resident, so the device budget takes blocks [0,N) and this absorbs
        the tail. Host cost is ~350 MB/block (~14 GB for a whole 14B expert),
        versus the ~27 GB/expert BF16 store the wan trainer deliberately declines
        (`train_wan22_real.mojo` design note) — fp8 halves it, so both experts'
        tails fit in host RAM. Returns the number of blocks pinned."""
        var used = 0
        var pinned = 0
        var scale_suffix = String(".__fp8_scale")
        for i in range(self._plan.count()):
            self._assert_raw_copy_dtype_safe(i)
            var p = self._plan.normalized_prefix(i)
            if (
                self._fp8h_slot(p) >= 0 or self._fp8_slot(p) >= 0
                or self._resident_slot(p) >= 0
            ):
                continue          # already resident somewhere — leave it alone
            var prefix_idx = -1
            for j in range(len(self._turbo.index_prefixes)):
                if self._turbo.index_prefixes[j] == p:
                    prefix_idx = j
                    break
            if prefix_idx < 0:
                raise Error(
                    String("pin_residents_fp8_host_prequantized: no tensors for prefix: ") + p
                )
            var start = self._turbo.index_starts[prefix_idx]
            var end = start + self._turbo.index_lengths[prefix_idx]
            var tensors = List[_HostFp8Tensor]()
            var block_bytes = 0
            for ni in range(start, end):
                var nm = self._turbo.index_names[ni].copy()
                if nm.endswith(scale_suffix):
                    continue      # consumed with its parent weight
                var tv = self._turbo.sharded.tensor_view(nm)
                if tv.dtype == STDtype.F8_E4M3:
                    var sname = nm + scale_suffix
                    var stv = self._turbo.sharded.tensor_view(sname)
                    var bytes = Tensor.from_view_raw(tv, ctx)   # keeps F8_E4M3
                    var scale = Tensor.from_view(stv, ctx)      # F32 [out]
                    var bh = ctx.enqueue_create_host_buffer[DType.uint8](bytes.nbytes())
                    var sh_h = ctx.enqueue_create_host_buffer[DType.uint8](scale.nbytes())
                    ctx.enqueue_copy(bh, bytes.buf)
                    ctx.enqueue_copy(sh_h, scale.buf)
                    ctx.synchronize()
                    block_bytes += bytes.nbytes() + scale.nbytes()
                    tensors.append(_HostFp8Tensor(
                        nm, True, True, Float32(1.0),
                        HArc(bh^), bytes.nbytes(), bytes.shape(), bytes.dtype(),
                        HArc(sh_h^), scale.nbytes(), scale.shape(),
                    ))
                else:
                    var t = Tensor.from_view_as_bf16(tv, ctx)
                    var bh = ctx.enqueue_create_host_buffer[DType.uint8](t.nbytes())
                    var dummy = ctx.enqueue_create_host_buffer[DType.uint8](1)
                    ctx.enqueue_copy(bh, t.buf)
                    ctx.synchronize()
                    block_bytes += t.nbytes()
                    tensors.append(_HostFp8Tensor(
                        nm, False, False, Float32(1.0),
                        HArc(bh^), t.nbytes(), t.shape(), t.dtype(),
                        HArc(dummy^), 1, List[Int](),
                    ))
            if used + block_bytes > budget_bytes:
                break             # plan-order contiguous; partial pin is legal
            self._fp8h_prefixes.append(p)
            self._fp8h_blocks.append(tensors^)
            used += block_bytes
            pinned += 1
            self._turbo.sharded.release_to_os()
        return pinned

    def pin_residents_fp8_host_raw(
        mut self, budget_bytes: Int, ctx: DeviceContext
    ) raises -> Int:
        """Pin raw-E4M3/BF16 blocks, with F32 tables converted once to BF16.

        This one path covers both Serenity formats used by product inference:
        Qwen stores unscaled E4M3 (`scale=1`), while Klein stores an F32 scalar
        sidecar per E4M3 weight. Sidecars are consumed once during the complete
        checkpoint-to-RAM copy and are never staged or reread during denoise.
        LTX stores its six small modulation tables per block as F32; match the
        normal LTX block loader by casting those tables to BF16 during this
        pre-denoise copy, then retain only their BF16 bytes in pinned RAM."""
        var used = 0
        var pinned = 0
        var scale_suffix = String("_scale")
        # Scaled raw-E4M3 exports do not share one global sentinel name. Klein
        # uses `img_in.weight_scale`, while LTX-2 namespaces every key under
        # `model.diffusion_model.*`. Detect the format from the actual header
        # suffix instead of one model-specific key, then fail closed if any
        # FP8 tensor is missing its scalar sidecar.
        var scalar_format = False
        for ref nm in self._turbo.sharded.names():
            if self._turbo.sharded.has_tensor(nm + scale_suffix):
                scalar_format = True
                break
        for i in range(self._plan.count()):
            var p = self._plan.normalized_prefix(i)
            if (
                self._fp8h_slot(p) >= 0 or self._fp8_slot(p) >= 0
                or self._resident_slot(p) >= 0
            ):
                continue
            var prefix_idx = -1
            for j in range(len(self._turbo.index_prefixes)):
                if self._turbo.index_prefixes[j] == p:
                    prefix_idx = j
                    break
            if prefix_idx < 0:
                raise Error(
                    String("pin_residents_fp8_host_raw: no tensors for prefix: ")
                    + p
                )
            var start = self._turbo.index_starts[prefix_idx]
            var end = start + self._turbo.index_lengths[prefix_idx]
            var block_bytes = 0
            for ni in range(start, end):
                var nm = self._turbo.index_names[ni]
                if nm.endswith(scale_suffix):
                    continue
                var tv = self._turbo.sharded.tensor_view(nm)
                if (
                    tv.dtype != STDtype.F8_E4M3
                    and tv.dtype != STDtype.BF16
                    and tv.dtype != STDtype.F32
                ):
                    raise Error(
                        String("pin_residents_fp8_host_raw: expected E4M3, ")
                        + String("BF16, or F32 tensor, got ") + tv.dtype.name()
                        + String(" for ") + nm
                    )
                if tv.dtype == STDtype.F32:
                    block_bytes += tv.numel() * STDtype.BF16.byte_size()
                else:
                    block_bytes += tv.nbytes()
            if used + block_bytes > budget_bytes:
                break

            var tensors = List[_HostFp8Tensor]()
            for ni in range(start, end):
                var nm = self._turbo.index_names[ni].copy()
                if nm.endswith(scale_suffix):
                    continue
                var tv = self._turbo.sharded.tensor_view(nm)
                if tv.dtype == STDtype.F32:
                    # The ordinary LTX AV loader casts these modulation tables
                    # with `Tensor.from_view_as_bf16`. Do the identical cast once
                    # here, before denoise, and copy only the BF16 result to the
                    # pinned host store. This is intentionally not a raw F32
                    # reinterpretation and cannot fall back to disk at await.
                    var t = Tensor.from_view_as_bf16(tv, ctx)
                    var bh = ctx.enqueue_create_host_buffer[DType.uint8](t.nbytes())
                    var dummy = ctx.enqueue_create_host_buffer[DType.uint8](1)
                    ctx.enqueue_copy(bh, t.buf)
                    ctx.synchronize()
                    tensors.append(_HostFp8Tensor(
                        nm, False, False, Float32(1.0),
                        HArc(bh^), t.nbytes(), t.shape(), STDtype.BF16,
                        HArc(dummy^), 0, List[Int](),
                    ))
                    continue
                var scalar_scale = Float32(1.0)
                if tv.dtype == STDtype.F8_E4M3:
                    var scale_name = nm + scale_suffix
                    if self._turbo.sharded.has_tensor(scale_name):
                        var scale_tv = self._turbo.sharded.tensor_view(scale_name)
                        if scale_tv.dtype != STDtype.F32 or len(scale_tv.shape) != 0:
                            raise Error(
                                String("FP8 scale must be an F32 scalar: ")
                                + scale_name
                            )
                        scalar_scale = read_f32_scalar_bytes(scale_tv.data)
                    elif scalar_format:
                        raise Error(
                            String("scalar-FP8 checkpoint is missing sidecar: ")
                            + scale_name
                        )
                var bh = ctx.enqueue_create_host_buffer[DType.uint8](tv.nbytes())
                var dst = BytePtr(unsafe_from_address=Int(bh.unsafe_ptr()))
                var src = BytePtr(unsafe_from_address=Int(tv.data.unsafe_ptr()))
                _ = sys_memcpy(dst, src, tv.nbytes())
                var dummy = ctx.enqueue_create_host_buffer[DType.uint8](1)
                tensors.append(_HostFp8Tensor(
                    nm,
                    tv.dtype == STDtype.F8_E4M3,
                    False,
                    scalar_scale,
                    HArc(bh^), tv.nbytes(), tv.shape.copy(), tv.dtype,
                    HArc(dummy^), 0, List[Int](),
                ))
            self._fp8h_prefixes.append(p)
            self._fp8h_blocks.append(tensors^)
            used += block_bytes
            pinned += 1
            # The complete host store can be tens of GiB. Drop checkpoint-backed
            # pages after each synchronously-copied block so pinned RAM and page
            # cache do not coexist at full-checkpoint size and OOM the machine.
            # Future blocks fault in once here, before sampling; denoise never
            # falls through to this mapping after the all-resident gate below.
            self._turbo.sharded.release_to_os()
        return pinned

    def pin_residents_int8(mut self, budget_bytes: Int, ctx: DeviceContext) raises -> Int:
        """Pin plan blocks (in plan order) permanently on device as int8-resident:
        each sizeable 2-D BF16 matmul weight is quantized ONCE to int8 [N,K] + a
        F32 scalar tensorwise scale [1] (int8_quant.mojo — the SAME weight quant
        slices 1-3 use, so the loader-quantized int8 == the block's
        quantize_*_int8); 1-D biases / norm scales / tiny 2-D tensors stay resident
        BF16 (exact). int8 = 1 byte/param → ~half the raw-BF16 footprint. await_block
        returns each block's int8 weights UNCHANGED (NO dequant — int8 to the GEMM).
        Returns the number of blocks pinned; `budget_bytes` caps the int8-resident
        bytes (plan-order contiguous — all-resident int8 does NOT fit 16GB, so the
        budget bounds how many; the rest stream). One sync per block bounds the
        transient BF16 quantize source (mirror krea2's build / pin_residents_fp8).
        ADDITIVE: the bf16 pin_residents and the fp8 paths are untouched."""
        var used = 0
        var pinned = 0
        # VRAM FIX (approach: NON-OWNING view over the turbo loader's already-
        # reserved IDLE double-buffer slabs — ZERO new device memory).
        # ROOT CAUSE: the old big_2d path materialized each tensor's bf16 source
        # via from_view_as_bf16 -> a FRESH device buffer of a DISTINCT size per
        # tensor. The size-keyed MAX alloc cache never reused those N distinct
        # shapes, so pin-phase reserved VRAM climbed toward the full ~18GB bf16
        # base and OOM'd MID-PIN (measured 15.4GB at 32 blocks).
        # FIX: stage every tensor's bf16 bytes through the turbo loader's existing
        # host0/dev0 slabs (each slab_capacity == max block bytes >= any single
        # tensor's bf16 bytes) and wrap a NON-OWNING Tensor over dev0. The slabs
        # are idle during this load-time pin, and with all blocks resident the
        # streaming double-buffer is never used again — so reusing them for the
        # transient quantize source costs no additional device memory (a separate
        # slab_capacity dev staging would linger reserved in the alloc cache).
        # mmap -> host0 -> dev0, exactly like pin_residents' reusable-staging idiom.
        ctx.synchronize()  # quiesce the turbo slab allocs before we reuse them
        for i in range(self._plan.count()):
            self._assert_raw_copy_dtype_safe(i)
            var p = self._plan.normalized_prefix(i)
            if (
                self._int8_slot(p) >= 0 or self._fp8_slot(p) >= 0
                or self._fp8h_slot(p) >= 0 or self._resident_slot(p) >= 0
            ):
                continue
            var prefix_idx = -1
            for j in range(len(self._turbo.index_prefixes)):
                if self._turbo.index_prefixes[j] == p:
                    prefix_idx = j
                    break
            if prefix_idx < 0:
                raise Error(
                    String("pin_residents_int8: no tensors for prefix: ") + p
                )
            var start = self._turbo.index_starts[prefix_idx]
            var end = start + self._turbo.index_lengths[prefix_idx]
            var tensors = List[_ResidentInt8Tensor]()
            var block_bytes = 0
            for ni in range(start, end):
                var nm = self._turbo.index_names[ni].copy()
                var tv = self._turbo.sharded.tensor_view(nm)
                var sh = tv.shape.copy()
                var big_2d = (
                    tv.dtype == STDtype.BF16
                    and len(sh) == 2
                    and sh[0] * sh[1] >= _FP8_MIN_ELEMS
                )
                if big_2d:
                    # Stage this tensor's bf16 bytes through the turbo loader's
                    # reused host0/dev0 slabs (mmap -> host0 -> dev0, the exact
                    # pin_residents idiom), then wrap a NON-OWNING Tensor over
                    # dev0 and quant it. big_2d requires tv.dtype == BF16, so the
                    # raw byte copy is byte-identical to from_view_as_bf16's BF16
                    # branch -> same source bytes -> same int8 output (loader int8
                    # parity holds).
                    var nb = tv.nbytes()  # bf16 bytes for this tensor
                    var dst = BytePtr(
                        unsafe_from_address=Int(self._turbo.host0.unsafe_ptr())
                    )
                    var src = BytePtr(
                        unsafe_from_address=Int(tv.data.unsafe_ptr())
                    )
                    _ = sys_memcpy(dst, src, nb)
                    _h2d_dma_copy(
                        UInt64(Int(self._turbo.dev0.unsafe_ptr())),
                        self._turbo.host0.unsafe_ptr(),
                        nb,
                        self._turbo.copy_stream,
                    )
                    # Fence the COPY STREAM specifically: the h2d above runs on
                    # self._turbo.copy_stream (a distinct ctx.create_stream()),
                    # which ctx.synchronize() does NOT cover — it only waits on
                    # the context's own stream. The quant kernels below run on the
                    # ctx stream and read dev0 IMMEDIATELY, so we must block until
                    # the copy_stream h2d has landed the bf16 bytes or the quant
                    # races on stale dev0 (measured cos 0.09 without this fence).
                    self._turbo.copy_stream.synchronize()
                    # NON-OWNING view over dev0: Tensor MOVES this buffer in and
                    # frees it on destruction, but owning=False makes the free a
                    # no-op, so the turbo dev0 slab survives for the next tensor
                    # (and for later streaming). No existing owning=False caller in
                    # this repo — this uses the stdlib any-origin ctor at
                    # gpu/host/device_context.mojo:1447 (ptr + owning kwarg).
                    var w_bf = Tensor(
                        DeviceBuffer[DType.uint8](
                            ctx, self._turbo.dev0.unsafe_ptr(), nb, owning=False
                        ),
                        sh.copy(),
                        STDtype.BF16,
                    )  # BF16 [N,K] over reused dev0 slab
                    var scale = int8_tensorwise_scale(w_bf, ctx)     # F32 [1]
                    var w8 = int8_encode_tensorwise(w_bf, scale, ctx)  # I8 [N,K]
                    block_bytes += (
                        w8.numel() * w8.dtype().byte_size()
                        + scale.numel() * scale.dtype().byte_size()
                    )
                    tensors.append(
                        _ResidentInt8Tensor(nm, True, TArc(w8^), TArc(scale^))
                    )
                    ctx.synchronize()  # fence quant reads before dev0 is reused
                    # w_bf's non-owning buffer frees here (no-op); dev0 persists.
                else:
                    var t = Tensor.from_view_as_bf16(tv, ctx)        # owned BF16 (small)
                    block_bytes += t.numel() * t.dtype().byte_size()
                    var arc = TArc(t^)
                    tensors.append(
                        _ResidentInt8Tensor(nm, False, arc.copy(), arc)
                    )
            if used + block_bytes > budget_bytes:
                break  # plan-order contiguous pin; the rest keep streaming
            self._int8_prefixes.append(p)
            self._int8_blocks.append(tensors^)
            used += block_bytes
            pinned += 1
            ctx.synchronize()  # fence this block's transient BF16 before the next
        return pinned

    def pin_residents_int8_host(
        mut self, budget_bytes: Int, ctx: DeviceContext
    ) raises -> Int:
        """int8-W8A8 STREAMED-tail store (Klein slice 6): quantize each remaining
        NON-resident plan block's big 2-D matmul weights ONCE to int8 [N,K] + a
        F32 scalar tensorwise scale [1] (the SAME int8_quant slices 1-5 use), then
        hold the result PINNED IN HOST RAM (small 1-D/tiny-2-D tensors held BF16
        host bytes). await_block H2Ds the raw int8 bytes + scale per step and
        returns them tagged I8 with NO dequant — HALF the per-step H2D bytes of
        the bf16 stream (the flagged speed lever), ZERO disk (MJ-1065). Call AFTER
        pin_residents_int8 pins the device-resident int8 blocks, so the tail
        streams int8-from-host instead of bf16-from-disk. `budget_bytes` caps HOST
        pinned bytes; pass a large value to host-pin the entire tail (nothing
        should stream bf16). D2H is the raw uint8 view of the device int8 buffer
        (to_host has no I8 path). ADDITIVE: bf16/fp8/int8-resident paths
        untouched."""
        var used = 0
        var pinned = 0
        for i in range(self._plan.count()):
            self._assert_raw_copy_dtype_safe(i)
            var p = self._plan.normalized_prefix(i)
            if (
                self._int8h_slot(p) >= 0 or self._int8_slot(p) >= 0
                or self._fp8_slot(p) >= 0 or self._fp8h_slot(p) >= 0
                or self._resident_slot(p) >= 0
            ):
                continue
            var prefix_idx = -1
            for j in range(len(self._turbo.index_prefixes)):
                if self._turbo.index_prefixes[j] == p:
                    prefix_idx = j
                    break
            if prefix_idx < 0:
                raise Error(
                    String("pin_residents_int8_host: no tensors for prefix: ") + p
                )
            var start = self._turbo.index_starts[prefix_idx]
            var end = start + self._turbo.index_lengths[prefix_idx]
            # ── pass 1: CONTIGUOUS block layout. Offsets are predicted from the
            # on-disk shapes (int8 = 1 byte/elem, scale = 4 bytes, small = BF16
            # 2 bytes/elem); pass 2 fail-louds if a quant output deviates. Each
            # tensor is 256B-aligned so downstream vectorized loads keep the
            # alignment a fresh per-tensor buffer used to guarantee.
            var recs = List[_HostInt8Rec]()
            var offset = 0
            for ni in range(start, end):
                var nm = self._turbo.index_names[ni].copy()
                var tv = self._turbo.sharded.tensor_view(nm)
                var sh = tv.shape.copy()
                var numel = 1
                for d in range(len(sh)):
                    numel *= sh[d]
                var big_2d = (
                    tv.dtype == STDtype.BF16
                    and len(sh) == 2
                    and sh[0] * sh[1] >= _FP8_MIN_ELEMS
                )
                if big_2d:
                    var w_off = _align_up(offset, 256)
                    var w_nb = numel                    # int8: 1 byte/elem
                    var s_off = _align_up(w_off + w_nb, 256)
                    var s_nb = 4                        # F32 scalar scale [1]
                    var s_sh: List[Int] = [1]
                    recs.append(_HostInt8Rec(
                        nm^, True, w_off, w_nb, ArcPointer(sh^), STDtype.I8,
                        s_off, s_nb, ArcPointer(s_sh^),
                    ))
                    offset = s_off + s_nb
                else:
                    var w_off = _align_up(offset, 256)
                    var w_nb = numel * 2                # held BF16 (2 bytes/elem)
                    var s_sh = List[Int]()
                    recs.append(_HostInt8Rec(
                        nm^, False, w_off, w_nb, ArcPointer(sh^), STDtype.BF16,
                        0, 0, ArcPointer(s_sh^),
                    ))
                    offset = w_off + w_nb
            var block_bytes = offset
            if used + block_bytes > budget_bytes:
                break  # plan-order contiguous; pass a large budget to pin the whole tail
            # ── pass 2: quantize each tensor (SAME kernels, SAME order as the
            # old per-tensor path → byte-identical int8 bytes + scales) and D2H
            # into the block's single pinned buffer at the recorded offsets via
            # pinned sub-buffer views.
            var hostbuf = ctx.enqueue_create_host_buffer[DType.uint8](
                block_bytes
            )
            ctx.synchronize()
            for t in range(len(recs)):
                var rec = recs[t]
                var tv = self._turbo.sharded.tensor_view(rec.name)
                if rec.is_int8:
                    var w_bf = Tensor.from_view_as_bf16(tv, ctx)     # owned BF16 [N,K]
                    var scale = int8_tensorwise_scale(w_bf, ctx)     # F32 [1]
                    var w8 = int8_encode_tensorwise(w_bf, scale, ctx)  # I8 [N,K]
                    if (
                        w8.nbytes() != rec.w_nbytes
                        or scale.nbytes() != rec.s_nbytes
                    ):
                        raise Error(
                            String("pin_residents_int8_host: layout mismatch ")
                            + rec.name + " w " + String(w8.nbytes()) + "!="
                            + String(rec.w_nbytes) + " s "
                            + String(scale.nbytes()) + "!=" + String(rec.s_nbytes)
                        )
                    # D2H the RAW int8 bytes (Tensor.buf is DeviceBuffer[uint8]
                    # regardless of logical I8 dtype — to_host has no I8 path)
                    # into the block buffer at the recorded offsets.
                    var wsub = hostbuf.create_sub_buffer[DType.uint8](
                        rec.w_off, rec.w_nbytes
                    )
                    var ssub = hostbuf.create_sub_buffer[DType.uint8](
                        rec.s_off, rec.s_nbytes
                    )
                    ctx.enqueue_copy(wsub, w8.buf)
                    ctx.enqueue_copy(ssub, scale.buf)
                    ctx.synchronize()  # D2H done + fences transient BF16/int8 device tensors
                else:
                    var tt = Tensor.from_view_as_bf16(tv, ctx)       # owned BF16 (small)
                    if tt.nbytes() != rec.w_nbytes:
                        raise Error(
                            String("pin_residents_int8_host: layout mismatch ")
                            + rec.name + " " + String(tt.nbytes()) + "!="
                            + String(rec.w_nbytes)
                        )
                    var wsub = hostbuf.create_sub_buffer[DType.uint8](
                        rec.w_off, rec.w_nbytes
                    )
                    ctx.enqueue_copy(wsub, tt.buf)
                    ctx.synchronize()
            self._int8h_prefixes.append(p)
            self._int8h_hosts.append(hostbuf^)
            self._int8h_recs.append(recs^)
            self._int8h_sizes.append(block_bytes)
            used += block_bytes
            pinned += 1
        if pinned > 0:
            self._i8h_ensure_slabs(ctx)
        return pinned

    def _i8h_ensure_slabs(mut self, ctx: DeviceContext) raises:
        """Allocate (or grow) the TWO persistent int8-host device slabs — sized
        to the largest pinned int8h block — plus the per-slot copy-done /
        compute-done event pairs that fence slot reuse (the TurboBlockLoader
        dev0/dev1 idiom). Idempotent; called at the end of
        pin_residents_int8_host."""
        var maxb = 0
        for i in range(len(self._int8h_sizes)):
            if self._int8h_sizes[i] > maxb:
                maxb = self._int8h_sizes[i]
        if maxb > self._i8h_capacity:
            self._i8h_devs = List[DeviceBuffer[DType.uint8]]()
            self._i8h_devs.append(ctx.enqueue_create_buffer[DType.uint8](maxb))
            self._i8h_devs.append(ctx.enqueue_create_buffer[DType.uint8](maxb))
            ctx.synchronize()  # materialize allocs before raw CUDA copies target them
            self._i8h_capacity = maxb
            self._i8h_slot_prefix = [String(""), String("")]
            self._i8h_active = 0
        if len(self._i8h_evs) == 0:
            self._i8h_evs.append(ctx.create_event[disable_timing=True]())
            self._i8h_evs.append(ctx.create_event[disable_timing=True]())
            self._i8h_cds.append(ctx.create_event[disable_timing=True]())
            self._i8h_cds.append(ctx.create_event[disable_timing=True]())
            self._i8h_cd_rec = [False, False]

    def _i8h_prefetch(mut self, bslot: Int) raises:
        """Stage int8-host block `bslot` into the idle persistent slab with ONE
        whole-block async H2D (cuMemcpyHtoDAsync) on the turbo COPY STREAM, so
        the copy overlaps the current block's compute. Direction-agnostic: the
        caller passes whichever plan neighbor it wants staged (fwd bi+1, bwd
        bi-1). Slot-reuse hazard (this write vs the slot's previous tenant's
        still-running kernels) is fenced by making the copy stream wait on the
        slot's compute-done event, recorded by mark_active_block_done — the
        exact TurboBlockLoader dev0/dev1 fence."""
        var p = self._int8h_prefixes[bslot].copy()
        if self._i8h_slot_prefix[0] == p or self._i8h_slot_prefix[1] == p:
            return  # already staged (idempotent — hit / CFG paired re-await)
        var slot = 1 - self._i8h_active  # idle slot (compute reads the other)
        if self._i8h_cd_rec[slot]:
            self._turbo.copy_stream.enqueue_wait_for(self._i8h_cds[slot])
            self._i8h_cd_rec[slot] = False
        _h2d_dma_copy(
            UInt64(Int(self._i8h_devs[slot].unsafe_ptr())),
            self._int8h_hosts[bslot].unsafe_ptr(),
            self._int8h_sizes[bslot],
            self._turbo.copy_stream,
        )
        self._turbo.copy_stream.record_event(self._i8h_evs[slot])
        self._i8h_slot_prefix[slot] = p^

    # ── fp8-host OVERLAPPED staging ──────────────────────────────────────────
    def set_fp8h_overlap(mut self, on: Bool):
        """Opt in to staging host-pinned fp8 blocks on the COPY STREAM during the
        previous block's compute, instead of inline on the default stream at await.

        Same bytes, same dequant kernel, same Block — only WHEN and ON WHICH STREAM
        the H2D runs changes, so results are bit-identical. Default OFF (C13): the
        inline path is what every existing fp8h caller has been running."""
        self._f8h_overlap = on

    def fp8h_stage(self) -> ArcPointer[Fp8hStage]:
        """This loader's staging slabs, for handing to a sibling via
        `share_fp8h_stage` (dual-expert: one pair of slabs for both experts)."""
        return self._f8h.copy()

    def share_fp8h_stage(
        mut self, var stage: ArcPointer[Fp8hStage], var tag: String
    ) raises:
        """Adopt `stage` as this loader's staging slabs and namespace its slots under
        `tag`. Every loader sharing a stage MUST pass a DISTINCT tag — slots are keyed
        `<tag>|<prefix>` precisely because two experts name their blocks identically
        (see Fp8hStage). Call before the first prefetch/await."""
        if len(tag) == 0:
            raise Error(
                "share_fp8h_stage: a non-empty tag is required — sharing slabs with"
                " an empty tag would let two experts' identical block prefixes"
                " collide in the same slot"
            )
        self._f8h = stage^
        self._f8h_tag = tag^

    def _f8h_key(self, prefix: String) -> String:
        return self._f8h_tag + String("|") + prefix

    def _f8h_layout(self, hslot: Int) raises -> List[Int]:
        """Byte offsets of block `hslot`'s tensors inside a staging slab, followed
        by the slab total as the LAST element. Layout order == the `_fp8h_blocks`
        iteration order, two entries per tensor (bytes, scale), each 256-aligned.

        ONE function computes this for both the prefetch (writer) and the await
        (reader) so the two can never disagree about where a tensor landed."""
        var offs = List[Int]()
        var off = 0
        for t in range(len(self._fp8h_blocks[hslot])):
            ref rt = self._fp8h_blocks[hslot][t]
            offs.append(off)
            off += ((rt.bytes_nbytes + 255) // 256) * 256
            offs.append(off)
            off += ((rt.scale_nbytes + 255) // 256) * 256
        offs.append(off)
        return offs^

    def _f8h_ensure(mut self, ctx: DeviceContext) raises:
        """Allocate the two persistent staging slabs (max fp8h block) + events.

        Grows to the largest block ANY sharer needs: a shared stage is sized by
        whichever loader calls this with the biggest block, and the reallocation
        invalidates both slots (contents are gone), so slot keys are cleared."""
        var maxb = 0
        for i in range(len(self._fp8h_blocks)):
            var offs = self._f8h_layout(i)
            var tot = offs[len(offs) - 1]
            if tot > maxb:
                maxb = tot
        if maxb > self._f8h[].capacity:
            self._f8h[].devs = List[DeviceBuffer[DType.uint8]]()
            self._f8h[].devs.append(ctx.enqueue_create_buffer[DType.uint8](maxb))
            self._f8h[].devs.append(ctx.enqueue_create_buffer[DType.uint8](maxb))
            ctx.synchronize()  # materialize allocs before raw CUDA copies target them
            self._f8h[].capacity = maxb
            self._f8h[].slot_key = [String(""), String("")]
            self._f8h[].active = 0
        if len(self._f8h[].evs) == 0:
            self._f8h[].evs.append(ctx.create_event[disable_timing=True]())
            self._f8h[].evs.append(ctx.create_event[disable_timing=True]())
            self._f8h[].cds.append(ctx.create_event[disable_timing=True]())
            self._f8h[].cds.append(ctx.create_event[disable_timing=True]())
            self._f8h[].cd_rec = [False, False]

    def _f8h_prefetch(mut self, hslot: Int, ctx: DeviceContext) raises:
        """Stage fp8-host block `hslot` into the idle slab on the turbo COPY STREAM
        so the H2D overlaps the CURRENT block's compute.

        Unlike int8h (one packed pinned buffer per block → one whole-block DMA), the
        fp8h pins keep a separate pinned buffer PER TENSOR, so this issues one async
        copy per tensor into the slab at `_f8h_layout` offsets. They are all on the
        copy stream and pipeline, so the overlap is the same; only the launch count
        differs (repacking the pins into one buffer would cut that — a follow-up).

        Slot-reuse hazard (this write vs the previous tenant's still-running kernels)
        is fenced on the slot's compute-done event, exactly as `_i8h_prefetch` does."""
        self._f8h_ensure(ctx)
        var p = self._f8h_key(self._fp8h_prefixes[hslot])
        if self._f8h[].slot_key[0] == p or self._f8h[].slot_key[1] == p:
            return  # already staged (idempotent — hit / paired re-await)
        var slot = 1 - self._f8h[].active  # idle slot (compute reads the other)
        if self._f8h[].cd_rec[slot]:
            self._turbo.copy_stream.enqueue_wait_for(self._f8h[].cds[slot])
            self._f8h[].cd_rec[slot] = False
        var offs = self._f8h_layout(hslot)
        var base = UInt64(Int(self._f8h[].devs[slot].unsafe_ptr()))
        for t in range(len(self._fp8h_blocks[hslot])):
            ref rt = self._fp8h_blocks[hslot][t]
            _h2d_dma_copy(
                base + UInt64(offs[2 * t]),
                rt.bytes_h[].unsafe_ptr(),
                rt.bytes_nbytes,
                self._turbo.copy_stream,
            )
            if rt.is_fp8 and rt.per_row_scale:
                _h2d_dma_copy(
                    base + UInt64(offs[2 * t + 1]),
                    rt.scale_h[].unsafe_ptr(),
                    rt.scale_nbytes,
                    self._turbo.copy_stream,
                )
        self._turbo.copy_stream.record_event(self._f8h[].evs[slot])
        self._f8h[].slot_key[slot] = p^

    def block_count(self) -> Int:
        return self._plan.count()

    def memory_resident_block_count(self) raises -> Int:
        """Count plan blocks whose sampling source is RAM or device memory.

        A complete TurboBlockLoader block store covers every plan block. When
        that store is disabled, every prefix must be present in one of the
        explicit resident stores. Anything else would fall through to mmap.
        """
        if self._turbo.use_block_store:
            return self._plan.count()
        var resident = 0
        for i in range(self._plan.count()):
            var p = self._plan.normalized_prefix(i)
            if (
                self._resident_slot(p) >= 0
                or self._fp8_slot(p) >= 0
                or self._fp8h_slot(p) >= 0
                or self._squareq_slot(p) >= 0
                or self._nvfp4_slot(p) >= 0
                or self._int8_slot(p) >= 0
                or self._int8h_slot(p) >= 0
            ):
                resident += 1
        return resident

    def require_all_blocks_memory_resident(self) raises:
        """Fail closed unless no plan block can fall through to checkpoint mmap."""
        var resident = self.memory_resident_block_count()
        var total = self._plan.count()
        if resident != total:
            raise Error(
                String("TurboPlannedLoader refuses sampling: only ")
                + String(resident) + String("/") + String(total)
                + String(" blocks are memory-resident; checkpoint reads during ")
                + String("denoise are forbidden")
            )

    def release_checkpoint_pages(mut self):
        """Drop checkpoint-backed page-cache residency after the RAM copy."""
        self._turbo.sharded.release_to_os()

    def discard_unused_raw_streaming_slots(
        mut self, ctx: DeviceContext
    ) raises:
        """Release duplicate raw-BF16 staging after complete explicit residency.

        Full host-FP8 and device-resident paths stage from their own stores and
        never use TurboBlockLoader's raw mmap/block-store slabs. Keeping those
        two raw slots wastes both pinned RAM and VRAM."""
        self.require_all_blocks_memory_resident()
        if self._turbo.use_block_store:
            raise Error(
                "cannot discard raw streaming slots while the raw block store "
                "is the resident sampling source"
            )
        self._turbo.discard_streaming_slots(ctx)

    def discard_fp8h_device_staging(mut self):
        """Drop transient FP8-host GPU slabs while retaining every host block.

        The caller must synchronize first. The next prefetch recreates the two
        device slabs lazily, so VAE decode can reclaim VRAM without forcing the
        next job to reread or requantize the checkpoint."""
        self._f8h = ArcPointer[Fp8hStage](Fp8hStage())

    def count(self) -> Int:
        return self._plan.count()

    def branch_visits(self) -> Int:
        return self._plan.branch_visits(self._config)

    def set_config(mut self, config: OffloadConfig):
        self._config = config

    def prefetch_index(self, index: Int) -> Int:
        return self._plan.prefetch_index(index, self._config)

    def _advance_residency_to_prefetching(mut self, index: Int) raises:
        """Drive residency state machine to PREFETCHING from any prior state."""
        var state = self._residency.get_state(index)
        if state == BlockState.prefetching() or state == BlockState.gpu_ready():
            return  # already in-flight or resident
        if state == BlockState.unloaded():
            self._residency.transition(index, BlockState.host_staged())
            self._residency.transition(index, BlockState.prefetching())
        elif state == BlockState.host_staged():
            self._residency.transition(index, BlockState.prefetching())

    def prefetch(mut self, index: Int) raises:
        """Queue block at plan index `index` for staging.

        GPU dispatch is deferred until the next await_block(ctx) call because
        PlannedBlockLoader.prefetch has no ctx parameter. The deferred dispatch
        is a no-op in TurboBlockLoader if the block is already staged (its
        idempotency guard handles this).

        Residency is advanced to PREFETCHING immediately so scheduling/budget
        logic sees the correct state.
        """
        if index < 0 or index >= self._plan.count():
            return
        self._advance_residency_to_prefetching(index)
        self._pending_idx = index

    def prefetch_next(mut self, index: Int) raises:
        """Prefetch the lookahead block for the current index."""
        self.prefetch(self.prefetch_index(index))

    def prefetch_with_ctx(mut self, index: Int, ctx: DeviceContext) raises:
        """Stage block at plan index `index` immediately on the copy stream."""
        if index < 0 or index >= self._plan.count():
            return
        self._assert_raw_copy_dtype_safe(index)
        var norm = self._plan.normalized_prefix(index)
        # fp8-host with overlap enabled: this is the ONE case where a host-pinned
        # block still wants staging work — the bytes live in pinned RAM and must
        # cross PCIe. Doing it here (during the previous block's compute) instead
        # of inline at await is the whole point. Overlap OFF keeps the early return.
        var f8h_pre = self._fp8h_slot(norm)
        if f8h_pre >= 0 and self._f8h_overlap:
            if self._pending_idx == index:
                self._pending_idx = -1
            self._f8h_prefetch(f8h_pre, ctx)
            return
        if (
            self._resident_slot(norm) >= 0
            or self._fp8_slot(norm) >= 0
            or self._squareq_slot(norm) >= 0
            or self._nvfp4_slot(norm) >= 0
            or f8h_pre >= 0
            or self._int8_slot(norm) >= 0
        ):
            return  # permanently device/host resident — no staging
        var i8h = self._int8h_slot(norm)
        if i8h >= 0:
            # int8-host block: ONE whole-block async H2D into the idle
            # persistent slab on the copy stream, overlapping the CURRENT
            # block's compute (the slice-6 stall fix). fp8h keeps its original
            # await-time path (untouched).
            if self._pending_idx == index:
                self._pending_idx = -1
            self._i8h_prefetch(i8h)
            return
        self._advance_residency_to_prefetching(index)
        if self._pending_idx == index:
            self._pending_idx = -1
        # TurboBlockLoader has only two rotating GPU slots. The plan-level
        # residency state can still say GPU_READY from a prior forward visit
        # even after that slot has been overwritten by later blocks. Always
        # delegate to TurboBlockLoader; it has the authoritative slot-prefix
        # check and will no-op when the block is truly still staged.
        var prefix = self._plan.normalized_prefix(index)
        self._turbo.prefetch(prefix, ctx)

    def prefetch_next_with_ctx(mut self, index: Int, ctx: DeviceContext) raises:
        """Stage the lookahead block for `index` immediately on the copy stream."""
        self.prefetch_with_ctx(self.prefetch_index(index), ctx)

    def mark_active_block_done(mut self, ctx: DeviceContext) raises:
        """Record compute completion for the active turbo slot.

        Hot loops call this after all kernels for the returned block have been
        queued. The low-level turbo loader gates future slot reuse with this
        event, matching the Rust BlockHandle compute_done contract.
        """
        self._turbo.mark_active_slot_compute_done(ctx)
        # int8-host double-buffer: record the compute-done event for the slab
        # slot the last int8h await returned. _i8h_prefetch makes the copy
        # stream wait on it before overwriting that slot (the block-N+2-vs-N
        # reuse hazard — same fence TurboBlockLoader applies to dev0/dev1).
        # Re-recording after a non-int8h block only delays reuse (safe).
        if len(self._i8h_cds) == 2:
            ctx.stream().record_event(self._i8h_cds[self._i8h_active])
            self._i8h_cd_rec[self._i8h_active] = True
        # Same double-buffer contract for the overlapped fp8-host slabs.
        if self._f8h_overlap and len(self._f8h[].cds) == 2:
            var a = self._f8h[].active
            ctx.stream().record_event(self._f8h[].cds[a])
            self._f8h[].cd_rec[a] = True

    def print_telemetry(self):
        """Print the underlying turbo loader counters.

        Shared by training and inference call sites. Keep telemetry on the
        offload runtime, not in per-model trainers.
        """
        self._turbo.print_telemetry()

    def _dispatch_pending(mut self, ctx: DeviceContext) raises:
        """Dispatch any pending GPU copy for the queued prefetch index."""
        if self._pending_idx < 0:
            return
        var pidx = self._pending_idx
        self._pending_idx = -1
        self.prefetch_with_ctx(pidx, ctx)

    def await_block(
        mut self, index: Int, ctx: DeviceContext
    ) raises -> PlannedBlockHandle:
        """Fence default stream, return PlannedBlockHandle (same type as sync loader).

        Steps:
          1. Dispatch any pending prefetch (queued by prefetch_next) onto the
             copy stream — this is where async overlap is achieved.
          2. Call TurboBlockLoader.await_block (fences default stream via
             enqueue_wait_for on the slot's DeviceEvent).
          3. Advance residency to GPU_READY, acquire/release refcount.
          4. Wrap Block in PlannedBlockHandle — same handle type Klein uses.

        DTYPE SAFETY: raw turbo copy does not convert bytes. If force_bf16 is
        active, the block must already be BF16 on disk; otherwise synchronous
        PlannedBlockLoader.load_block_as_bf16 or a future converting Turbo
        staging path is required.
        """
        var prefix = self._plan.prefix(index)
        var load_prefix = self._plan.normalized_prefix(index)
        self._assert_raw_copy_dtype_safe(index)

        # ── resident fast path: no copy, no slot, no fence ───────────────────
        var rslot = self._resident_slot(load_prefix)
        if rslot >= 0:
            self._dispatch_pending(ctx)  # keep lookahead for streamed blocks
            var rblock = Block()
            for t in range(len(self._res_recs[rslot])):
                var rec = self._res_recs[rslot][t].copy()
                var sub = self._res_devs[rslot].create_sub_buffer[DType.uint8](
                    rec.offset, rec.nbytes
                )
                var tt = Tensor(sub^, rec.shape[].copy(), rec.dtype)
                rblock[rec.name] = ArcPointer(tt^)
            self._step += 1
            return PlannedBlockHandle(index, prefix, rblock^)

        # ── fp8-resident fast path: dequant per block (NO disk), no slot/fence ──
        # Reconstruct the SAME Block the streamed path returns: the large weights
        # are dequantized E4M3→BF16 on the fly; small tensors are held-BF16 and
        # shared by refcount. Downstream reconstruction (`_*_from_block`) is byte-
        # for-byte agnostic to the source (it reads BF16 tensors by name).
        var fslot = self._fp8_slot(load_prefix)
        if fslot >= 0:
            self._dispatch_pending(ctx)  # keep lookahead for any streamed blocks
            var fblock = Block()
            for t in range(len(self._fp8_blocks[fslot])):
                ref rt = self._fp8_blocks[fslot][t]
                if rt.is_fp8:
                    var w = fp8_e4m3_dequant_perrow_to_bf16(rt.a[], rt.scale[], ctx)
                    fblock[rt.name] = ArcPointer(w^)
                else:
                    fblock[rt.name] = rt.a.copy()  # share the resident BF16 tensor
            self._step += 1
            return PlannedBlockHandle(index, prefix, fblock^)

        # ── squareq_w4-resident fast path: reconstruct BF16 per block (NO disk),
        # no slot/fence. Mirrors the fp8 branch: big weights are rebuilt
        # W_hat = dequant4@H_bd + lora_up@lora_down^T on the fly; small tensors
        # are held-BF16, shared by refcount. Downstream builders are source-
        # agnostic (BF16 tensors by name) — same Block the streamed path returns.
        var sqslot = self._squareq_slot(load_prefix)
        if sqslot >= 0:
            self._dispatch_pending(ctx)  # keep lookahead for any streamed blocks
            var sblock = Block()
            for t in range(len(self._squareq_blocks[sqslot])):
                ref sqt = self._squareq_blocks[sqslot][t]
                if sqt.is_quant:
                    var w: Tensor
                    if sqt.is_w8:
                        w = squareq_w8_reconstruct_weight(
                            sqt.q[], sqt.s[], sqt.ld[], sqt.lu[],
                            sqt.in_f, sqt.out_f, ctx,
                        )
                    else:
                        w = squareq_reconstruct_weight(
                            sqt.q[], sqt.s[], sqt.ld[], sqt.lu[],
                            sqt.in_f, sqt.out_f, ctx,
                        )
                    sblock[sqt.name] = ArcPointer(w^)
                else:
                    sblock[sqt.name] = sqt.q.copy()  # share the resident BF16
            self._step += 1
            return PlannedBlockHandle(index, prefix, sblock^)

        # ── squareq_nvfp4-resident fast path: native-FP4 payload + BF16 W_hat
        # (NO disk), no slot/fence. Each quantized weight lands TWICE in the
        # Block: (a) the reconstructed BF16 W_hat under its own name
        # (squareq_nvfp4_reconstruct_weight — the backward and any non-wired
        # consumer read the same names/dtypes the squareq_w4 branch returns),
        # and (b) the packed payload under resident_nvfp4_key(name, part)
        # ("::"-suffixed keys can never collide with real tensor names) so the
        # Klein block builders can assemble the FORWARD-ONLY fp4 payload.
        # Small tensors are the held-BF16 residents, shared by refcount.
        var nvslot = self._nvfp4_slot(load_prefix)
        if nvslot >= 0:
            self._dispatch_pending(ctx)  # keep lookahead for any streamed blocks
            var nblock = Block()
            for t in range(len(self._nvfp4_blocks[nvslot])):
                ref nvt = self._nvfp4_blocks[nvslot][t]
                if nvt.is_quant:
                    if self._fwd_only_awaits:
                        # fwd-only: the fp4 dispatch reads the ::payload; the
                        # name slot gets a SHAPE-correct view of the shared
                        # dummy buffer (values garbage — never read in fwd).
                        var sub = self._nvfp4_dummy[0].create_sub_buffer[
                            DType.uint8
                        ](0, nvt.out_f * nvt.in_f * 2)
                        nblock[nvt.name] = ArcPointer(
                            Tensor(sub^, [nvt.out_f, nvt.in_f], STDtype.BF16)
                        )
                    else:
                        var w = squareq_nvfp4_reconstruct_weight(
                            nvt.nvq[], nvt.nvs[], nvt.nvg, nvt.ld[], nvt.lu[],
                            nvt.in_f, nvt.out_f, ctx,
                        )
                        nblock[nvt.name] = ArcPointer(w^)
                    nblock[resident_nvfp4_key(nvt.name, String("nvq"))] = nvt.nvq.copy()
                    nblock[resident_nvfp4_key(nvt.name, String("nvs"))] = nvt.nvs.copy()
                    nblock[resident_nvfp4_key(nvt.name, String("ld"))] = nvt.ld.copy()
                    nblock[resident_nvfp4_key(nvt.name, String("lu"))] = nvt.lu.copy()
                    nblock[resident_nvfp4_key(nvt.name, String("nvg"))] = nvt.nvg_t.copy()
                else:
                    nblock[nvt.name] = nvt.nvq.copy()  # share the resident BF16
            self._step += 1
            return PlannedBlockHandle(index, prefix, nblock^)

        # ── int8-resident fast path (Klein int8-W8A8 slice 4): NO dequant, no
        # slot/fence. Return the SAME per-tensor names the streamed/bf16 path
        # returns, but each big weight is the resident int8 [N,K] tensor and its
        # F32 scalar scale is carried alongside under resident_i8_scale_key(name).
        # The block-weight builder (`_stream_weights_from_block` /
        # `_single_weights_from_block`) detects the I8 dtype, pairs each weight with
        # its scale, and assembles the StreamInt8 / SingleBlockInt8 payload — the
        # int8 stays int8 all the way to the GEMM. Small tensors (norms) are the
        # held-BF16 tensors, shared by refcount.
        var i8slot = self._int8_slot(load_prefix)
        if i8slot >= 0:
            self._dispatch_pending(ctx)  # keep lookahead for any streamed blocks
            var iblock = Block()
            for t in range(len(self._int8_blocks[i8slot])):
                ref rt = self._int8_blocks[i8slot][t]
                if rt.is_int8:
                    iblock[rt.name] = rt.w8.copy()   # resident int8 [N,K] (NO dequant)
                    iblock[resident_i8_scale_key(rt.name)] = rt.scale.copy()  # F32 [1]
                else:
                    iblock[rt.name] = rt.w8.copy()   # share the resident BF16 tensor
            self._step += 1
            return PlannedBlockHandle(index, prefix, iblock^)

        # ── host-pinned int8 path (Klein int8-W8A8 slice 6): the block's raw
        # int8 bytes + F32 scalar scales were H2D'd into a persistent device
        # slab by _i8h_prefetch DURING the previous block's compute (one async
        # whole-block cuMemcpy on the copy stream). Here the default stream
        # only enqueues a wait on the slot's copy-done EVENT (near-zero stall
        # — the old path did N in-stream per-tensor H2Ds at await, 27% GPU
        # idle) and the Block is built from NON-OWNING sub-buffer views at the
        # recorded offsets. Same names, same bytes, same I8/scale tagging
        # (weight under its name, scale under resident_i8_scale_key) — the
        # downstream block-weight builder assembles the identical
        # StreamInt8/SingleBlockInt8 payload. NO dequant, NO disk (MJ-1065).
        # The slab outlives the Block's ArcPointer[Tensor] views (persistent
        # member); slot rewrite is gated on this block's compute-done event.
        var i8hslot = self._int8h_slot(load_prefix)
        if i8hslot >= 0:
            self._dispatch_pending(ctx)  # keep lookahead for any streamed blocks
            var slot = -1
            if self._i8h_slot_prefix[0] == load_prefix:
                slot = 0
            elif self._i8h_slot_prefix[1] == load_prefix:
                slot = 1
            if slot < 0:
                # miss path (cold start / non-sequential access): stage now on
                # the copy stream; the event wait below then serializes —
                # correct, just unoverlapped (the pre-fix per-await behavior).
                self._i8h_prefetch(i8hslot)
                if self._i8h_slot_prefix[0] == load_prefix:
                    slot = 0
                elif self._i8h_slot_prefix[1] == load_prefix:
                    slot = 1
                if slot < 0:
                    raise Error(
                        String("TurboPlannedLoader.await_block: int8h block ")
                        + "not staged: " + load_prefix
                    )
            ctx.stream().enqueue_wait_for(self._i8h_evs[slot])
            self._i8h_active = slot
            var i8hblock = Block()
            for t in range(len(self._int8h_recs[i8hslot])):
                ref rec = self._int8h_recs[i8hslot][t]
                var wsub = self._i8h_devs[slot].create_sub_buffer[DType.uint8](
                    rec.w_off, rec.w_nbytes
                )
                var wt = Tensor(wsub^, rec.w_shape[].copy(), rec.w_dtype)
                if rec.is_int8:
                    var ssub = self._i8h_devs[slot].create_sub_buffer[
                        DType.uint8
                    ](rec.s_off, rec.s_nbytes)
                    var st = Tensor(ssub^, rec.s_shape[].copy(), STDtype.F32)
                    i8hblock[rec.name] = ArcPointer(wt^)  # int8 [N,K], NO dequant
                    i8hblock[resident_i8_scale_key(rec.name)] = ArcPointer(st^)  # F32 [1]
                else:
                    i8hblock[rec.name] = ArcPointer(wt^)  # BF16 small (no scale)
            self._step += 1
            return PlannedBlockHandle(index, prefix, i8hblock^)

        # ── host-pinned fp8 path (fp8_e4m3_host): H2D the E4M3 bytes + scale
        # (~half the bytes of bf16 streaming), dequant to BF16 on device. Same
        # Block shape as the streamed/fp8 paths; NO disk (MJ-1065).
        var hslot = self._fp8h_slot(load_prefix)
        if hslot >= 0 and self._f8h_overlap:
            # OVERLAPPED variant: the bytes were DMA'd into a persistent slab on the
            # copy stream during the previous block's compute (_f8h_prefetch). The
            # default stream only waits on the slot's copy-done event, then the SAME
            # dequant runs over slab sub-buffer views. Identical bytes in, identical
            # Block out — only the copy's stream and timing changed.
            self._dispatch_pending(ctx)
            self._f8h_ensure(ctx)
            var okey = self._f8h_key(load_prefix)
            var oslot = -1
            if self._f8h[].slot_key[0] == okey:
                oslot = 0
            elif self._f8h[].slot_key[1] == okey:
                oslot = 1
            if oslot < 0:
                # miss (cold start / non-sequential visit / expert switch): stage now
                # — correct, just unoverlapped, exactly like the int8h miss path.
                self._f8h_prefetch(hslot, ctx)
                if self._f8h[].slot_key[0] == okey:
                    oslot = 0
                elif self._f8h[].slot_key[1] == okey:
                    oslot = 1
                if oslot < 0:
                    raise Error(
                        String("TurboPlannedLoader.await_block: fp8h block ")
                        + "not staged: " + load_prefix
                    )
            ctx.stream().enqueue_wait_for(self._f8h[].evs[oslot])
            self._f8h[].active = oslot
            var ooffs = self._f8h_layout(hslot)
            var oblock = Block()
            for t in range(len(self._fp8h_blocks[hslot])):
                ref rt = self._fp8h_blocks[hslot][t]
                var wsub = self._f8h[].devs[oslot].create_sub_buffer[DType.uint8](
                    ooffs[2 * t], rt.bytes_nbytes
                )
                var wt = Tensor(wsub^, rt.bytes_shape.copy(), rt.bytes_dtype)
                if rt.is_fp8:
                    var w: Tensor
                    if rt.per_row_scale:
                        var ssub = self._f8h[].devs[oslot].create_sub_buffer[DType.uint8](
                            ooffs[2 * t + 1], rt.scale_nbytes
                        )
                        var st = Tensor(ssub^, rt.scale_shape.copy(), STDtype.F32)
                        w = fp8_e4m3_dequant_perrow_to_bf16(wt, st, ctx)
                    else:
                        w = fp8_e4m3_dequant_to_bf16(wt, rt.scalar_scale, ctx)
                    # dequant allocates its own output, so the slab views are read
                    # only by this kernel; the BF16 smalls below stay as views and
                    # are protected by the compute-done fence like int8h's.
                    oblock[rt.name] = ArcPointer(w^)
                else:
                    oblock[rt.name] = ArcPointer(wt^)
            self._step += 1
            return PlannedBlockHandle(index, prefix, oblock^)

        if hslot >= 0:
            self._dispatch_pending(ctx)
            var hblock = Block()
            for t in range(len(self._fp8h_blocks[hslot])):
                ref rt = self._fp8h_blocks[hslot][t]
                var dbuf = ctx.enqueue_create_buffer[DType.uint8](rt.bytes_nbytes)
                ctx.enqueue_copy(dbuf, rt.bytes_h[])
                var wt = Tensor(dbuf^, rt.bytes_shape.copy(), rt.bytes_dtype)
                if rt.is_fp8:
                    var w: Tensor
                    if rt.per_row_scale:
                        var sbuf = ctx.enqueue_create_buffer[DType.uint8](
                            rt.scale_nbytes
                        )
                        ctx.enqueue_copy(sbuf, rt.scale_h[])
                        var st = Tensor(sbuf^, rt.scale_shape.copy(), STDtype.F32)
                        w = fp8_e4m3_dequant_perrow_to_bf16(wt, st, ctx)
                    else:
                        w = fp8_e4m3_dequant_to_bf16(wt, rt.scalar_scale, ctx)
                    hblock[rt.name] = ArcPointer(w^)
                else:
                    hblock[rt.name] = ArcPointer(wt^)
            self._step += 1
            return PlannedBlockHandle(index, prefix, hblock^)

        # Dispatch pending prefetch onto copy stream BEFORE fencing default stream.
        # This is the overlap moment: copy stream starts staging the NEXT block
        # while the default stream is about to process the CURRENT block.
        self._dispatch_pending(ctx)

        # Fetch from turbo: fences default stream via enqueue_wait_for.
        var block = self._turbo.await_block(load_prefix, ctx)

        # Advance residency to GPU_READY.
        var state = self._residency.get_state(index)
        if state == BlockState.prefetching():
            self._residency.transition(index, BlockState.gpu_ready())
        elif state == BlockState.host_staged():
            self._residency.transition(index, BlockState.prefetching())
            self._residency.transition(index, BlockState.gpu_ready())
        elif state == BlockState.unloaded():
            self._residency.transition(index, BlockState.host_staged())
            self._residency.transition(index, BlockState.prefetching())
            self._residency.transition(index, BlockState.gpu_ready())
        # else: GPU_READY (re-await in CFG paired) — no transition needed.

        self._residency.acquire(index)
        self._residency.mark_visit(index, self._step)
        self._step += 1
        self._residency.release(index)

        return PlannedBlockHandle(index, prefix, block^)

    def _assert_raw_copy_dtype_safe(self, index: Int) raises:
        """Fail before raw Turbo H2D copy would reinterpret non-BF16 bytes.

        PlannedBlockLoader can honor force_bf16 by converting each tensor before
        H2D. TurboBlockLoader intentionally copies packed block bytes as-is, so
        force_bf16 is safe only when the checkpoint tensors are already BF16.
        """
        if self._config.dtype_policy != DTypePolicy.force_bf16():
            return
        if index < 0 or index >= self._plan.count():
            return
        var load_prefix = self._plan.normalized_prefix(index)
        for ref nm in self._turbo.sharded.names():
            if nm.startswith(load_prefix):
                var tv = self._turbo.sharded.tensor_view(nm)
                if tv.dtype != STDtype.BF16:
                    raise Error(
                        String("TurboPlannedLoader: force_bf16 requires BF16 ")
                        + String("on-disk tensors for raw Turbo copy; ")
                        + nm + String(" is ") + tv.dtype.name()
                        + String(" in block ") + load_prefix
                        + String(". Use PlannedBlockLoader.load_block_as_bf16 ")
                        + String("or add a converting Turbo staging path.")
                    )
