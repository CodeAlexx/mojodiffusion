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
from std.memory import ArcPointer

from serenitymojo.offload.block_loader import Block
from serenitymojo.offload.plan import BlockPlan, DTypePolicy, OffloadConfig
from serenitymojo.offload.planned_loader import PlannedBlockHandle
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import BytePtr, sys_memcpy
from serenitymojo.offload.turbo_loader import (
    TurboBlockLoader, _TensorRecord, _h2d_dma_copy,
)
from serenitymojo.offload.residency import (
    ResidencyManager,
    BudgetTracker,
    BlockState,
)
from serenitymojo.tensor import Tensor
from serenitymojo.ops.fp8_quant import (
    fp8_e4m3_rowscale, fp8_e4m3_encode_perrow,
)
from serenitymojo.ops.fp8 import fp8_e4m3_dequant_perrow_to_bf16
# int8-resident weight quant (Klein int8-W8A8 slice 4). Same tensorwise weight
# quant slices 1-3 use for the block-direct int8 payload, so the loader-quantized
# int8 == the block's quantize_*_int8 (parity-gated in klein_loader_int8_parity).
from serenitymojo.ops.int8_quant import (
    int8_tensorwise_scale, int8_encode_tensorwise,
)

comptime TArc = ArcPointer[Tensor]


# int8-resident Block carries each big weight as an I8 [N,K] tensor plus a
# per-weight F32 scalar scale [1]; the scale is stored in the Block dict under
# `<weight_name>` + this suffix so the block-weight builder can pair them and
# assemble the StreamInt8 / SingleBlockInt8 payload (NO dequant — int8 stays int8
# to the GEMM). The suffix is chosen so it can never collide with a real tensor
# name (real Klein tensor names never contain "::").
def resident_i8_scale_key(name: String) -> String:
    return name + String("::i8scale")

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
    Else `bytes_h` holds raw BF16 bytes (`scale_h` is a 1-byte dummy)."""
    var name: String
    var is_fp8: Bool
    var bytes_h: HArc
    var bytes_nbytes: Int
    var bytes_shape: List[Int]
    var bytes_dtype: STDtype
    var scale_h: HArc
    var scale_nbytes: Int
    var scale_shape: List[Int]

    def __init__(
        out self, var name: String, is_fp8: Bool,
        var bytes_h: HArc, bytes_nbytes: Int,
        var bytes_shape: List[Int], bytes_dtype: STDtype,
        var scale_h: HArc, scale_nbytes: Int, var scale_shape: List[Int],
    ):
        self.name = name^
        self.is_fp8 = is_fp8
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
        return TurboPlannedLoader(turbo^, plan^, config, residency^)

    def __init__(
        out self,
        var turbo: TurboBlockLoader,
        var plan: BlockPlan,
        config: OffloadConfig,
        var residency: ResidencyManager,
    ):
        self._turbo = turbo^
        self._plan = plan^
        self._config = config
        self._residency = residency^
        self._step = 0
        self._pending_idx = -1
        self._has_ctx = False
        self._res_prefixes = List[String]()
        self._res_devs = List[DeviceBuffer[DType.uint8]]()
        self._res_recs = List[List[_TensorRecord]]()
        self._fp8_prefixes = List[String]()
        self._fp8_blocks = List[List[_ResidentFp8Tensor]]()
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

    def pin_residents(mut self, budget_bytes: Int, ctx: DeviceContext) raises -> Int:
        """Pin plan blocks (in plan order) permanently on device until
        `budget_bytes` is reached. Each pinned block gets its OWN device buffer,
        H2D'd ONCE from the turbo loader's pinned block store — byte-identical
        to the streaming path (same source bytes, same per-tensor records).
        Returns the number of blocks pinned. One ctx.synchronize() at the end
        fences all resident copies; afterwards await_block on a resident is a
        pure sub-buffer-view build (no copy, no slot, no fence)."""
        var used = 0
        var pinned = 0
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
            ctx.synchronize()  # staging is reused next block — fence the copy
            self._res_prefixes.append(p)
            self._res_devs.append(dev^)
            self._res_recs.append(recs^)
            used += n_bytes
            pinned += 1
        return pinned

    def pin_residents_fp8(mut self, budget_bytes: Int, ctx: DeviceContext) raises -> Int:
        """Pin plan blocks (in plan order) permanently on device as fp8-resident:
        each sizeable 2-D BF16 matmul weight is quantized ONCE to E4M3 bytes + a
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
                    tv.dtype == STDtype.BF16
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

    def pin_residents_fp8_host(
        mut self, budget_bytes: Int, ctx: DeviceContext
    ) raises -> Int:
        """fp8_e4m3_host mode: quantize each block ONCE (same E4M3 + per-row F32
        scale as pin_residents_fp8) but hold the result PINNED IN HOST RAM, not
        on device — for models whose fp8 footprint does not fit device-resident
        next to training state (qwenimage: fp8 ~20GB, card 24GB, measured OOM
        2026-07-04). await_block H2Ds the fp8 bytes + scale and dequants — HALF
        the per-step PCIe of bf16 streaming, ZERO disk (MJ-1065). `budget_bytes`
        caps HOST pinned bytes; caller must require pinned==block_count."""
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
                    tv.dtype == STDtype.BF16
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
                        nm, True,
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
                        nm, False,
                        HArc(bh^), t.nbytes(), t.shape(), t.dtype(),
                        HArc(dummy^), 1, List[Int](),
                    ))
            if used + block_bytes > budget_bytes:
                break  # plan-order contiguous; caller enforces pinned==count
            self._fp8h_prefixes.append(p)
            self._fp8h_blocks.append(tensors^)
            used += block_bytes
            pinned += 1
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

    def block_count(self) -> Int:
        return self._plan.count()

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
        if (
            self._resident_slot(norm) >= 0
            or self._fp8_slot(norm) >= 0
            or self._fp8h_slot(norm) >= 0
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
        if hslot >= 0:
            self._dispatch_pending(ctx)
            var hblock = Block()
            for t in range(len(self._fp8h_blocks[hslot])):
                ref rt = self._fp8h_blocks[hslot][t]
                var dbuf = ctx.enqueue_create_buffer[DType.uint8](rt.bytes_nbytes)
                ctx.enqueue_copy(dbuf, rt.bytes_h[])
                var wt = Tensor(dbuf^, rt.bytes_shape.copy(), rt.bytes_dtype)
                if rt.is_fp8:
                    var sbuf = ctx.enqueue_create_buffer[DType.uint8](
                        rt.scale_nbytes
                    )
                    ctx.enqueue_copy(sbuf, rt.scale_h[])
                    var st = Tensor(sbuf^, rt.scale_shape.copy(), STDtype.F32)
                    var w = fp8_e4m3_dequant_perrow_to_bf16(wt, st, ctx)
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
