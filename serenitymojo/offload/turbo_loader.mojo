# turbo_loader.mojo — TurboBlockLoader: async double-buffered weight offload.
#
# Phase 1 of the async offload backend. Overlaps H2D weight staging on an
# explicit copy stream with model compute on the default GPU stream, using two
# persistent slot pairs (pinned host slab + device slab + DeviceEvent).
#
# DESIGN (Phase-0 established; Phase-1 implements):
#   - DeviceContext() is a singleton. Model compute runs on the DEFAULT stream.
#   - enqueue_copy() is DEFAULT-stream only — cannot stage weights async.
#   - H2D staging uses CUDA's copy engine via cuMemcpyHtoDAsync_v2 on the
#     explicit copy stream created via ctx.create_stream(). The old GPU
#     byte-copy kernel is kept as a fallback/probe, but the trainer path must
#     not spend SM time copying weights.
#   - Large checkpoints do not get copied into one full-model pinned host store
#     at open(). Each prefetch fills the selected pinned slot from the mmap, then
#     DMA-copies that slot to its device slab. This keeps startup memory bounded.
#   - Event handshake:
#       prefetch(): dispatch copy kernel on copy_stream → record event on copy_stream.
#       await_block(): ctx.stream().enqueue_wait_for(ev) — default stream fences.
#   - Two-slot rotation: while compute reads slot A, copy stream stages into slot B.
#     Slots sized to the largest block in the model.
#
# Public seam (same as BlockLoader):
#   TurboBlockLoader.open(dir, ctx) -> TurboBlockLoader
#   loader.prefetch(prefix, ctx)           # non-blocking: copy kernel dispatched
#   loader.await_block(prefix, ctx) -> Block  # fence default stream + build Block
#
# Mojo 1.0.0b1, Linux x86-64, NVIDIA RTX 3090 Ti, MAX 26.3.

from std.ffi import external_call
from std.memory import ArcPointer
from std.gpu.host import DeviceContext, HostBuffer, DeviceBuffer, DeviceStream, DeviceEvent
from std.gpu.host._nvidia_cuda import CUDA, CUstream
from std.gpu import global_idx
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import BytePtr, sys_memcpy
from serenitymojo.tensor import Tensor
from serenitymojo.offload.block_loader import Block
from serenitymojo.offload.telemetry import OffloadTelemetry

comptime TURBO_USE_DEFAULT_STREAM_COPY = False
comptime TURBO_USE_CUDA_DMA_COPY = True
# True since 2026-06-11 (beat-flame session 2): with False, EVERY streamed-block
# visit does a synchronous ~580 MB host memcpy (mmap → pinned slab, turbo_loader
# prefetch step 1) on the hot thread before the async DMA — ~20 GB/step of host
# memcpy at Klein-9B 512px (18 streamed blocks × 2 visits). The persistent store
# pays ONE ~10 GiB pinned alloc + populate at open() (RAM measured 62 GiB, fits)
# and prefetch becomes pure async DMA from pinned store. Bytes are identical
# (same mmap source, copied once) — gated by loss anchors + byte-identity smoke.
comptime TURBO_USE_PERSISTENT_BLOCK_STORE = True


# ─── copy kernel ──────────────────────────────────────────────────────────────
# Byte-wise H2D kernel. Reads pinned host slab (device-accessible via DMA),
# writes device slab. One thread per byte. Dispatched on the explicit copy stream.

def _h2d_copy_kernel(
    src: UnsafePointer[UInt8, MutAnyOrigin],
    dst: UnsafePointer[UInt8, MutAnyOrigin],
    n: Int,
):
    """Copy n bytes. Called on the explicit copy stream.

    The first implementation copied one byte per GPU thread. That was correct,
    but it made block swapping pay excessive launch/thread traffic on every
    prefetched transformer block. Safetensors tensors and our packed block slabs
    are naturally aligned enough for the large path here, so copy UInt64 chunks
    and let only the short tail use byte lanes.
    """
    var i = Int(global_idx.x)
    var n64 = n // 8
    if i < n64:
        var src64 = src.bitcast[UInt64]()
        var dst64 = dst.bitcast[UInt64]()
        dst64[i] = src64[i]
    else:
        var tail_i = i - n64
        var tail_n = n - n64 * 8
        if tail_i < tail_n:
            var b = n64 * 8 + tail_i
            dst[b] = src[b]


def _cu_memcpy_htod_async(
    dst_device_ptr: UInt64,
    src_host_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    nbytes: Int,
    stream: CUstream,
) -> Int32:
    return external_call["cuMemcpyHtoDAsync_v2", Int32](
        dst_device_ptr, src_host_ptr, nbytes, stream
    )


def _h2d_dma_copy(
    dst_device_ptr: UInt64,
    src_host_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    nbytes: Int,
    stream: DeviceStream,
) raises:
    var cuda_stream = CUDA(stream)
    var rc = Int(_cu_memcpy_htod_async(
        dst_device_ptr, src_host_ptr, nbytes, cuda_stream,
    ))
    if rc != 0:
        raise Error(
            String("TurboBlockLoader: cuMemcpyHtoDAsync_v2 failed rc=")
            + String(rc)
        )


# ─── per-tensor staging record ────────────────────────────────────────────────
# Byte offset + metadata for one tensor within the flat slab.
# Shape stored as ArcPointer[List[Int]] so this struct is Copyable (List[Int]
# is not ImplicitlyCopyable directly; ArcPointer is).

struct _TensorRecord(Copyable, Movable, ImplicitlyCopyable):
    var name: String
    var offset: Int       # byte offset into slab
    var nbytes: Int       # byte count
    var shape: ArcPointer[List[Int]]
    var dtype: STDtype

    def __init__(
        out self,
        name: String,
        offset: Int,
        nbytes: Int,
        shape: ArcPointer[List[Int]],
        dtype: STDtype,
    ):
        self.name = name
        self.offset = offset
        self.nbytes = nbytes
        self.shape = shape
        self.dtype = dtype


# ─── TurboBlockLoader ─────────────────────────────────────────────────────────

struct TurboBlockLoader(Movable):
    """Async double-buffered transformer-block weight loader.

    Two persistent slab pairs (pinned host + device) overlap H2D staging on the
    explicit copy stream with model compute on the default stream. DeviceEvent
    handshake ensures the default stream waits for the copy before reading.

    Double-buffer contract: never call await_block(B) after calling prefetch(A)
    but before await_block(A) finishes — the two-slot rotation assumes slot A is
    in active use while slot B is being staged. The smoke test validates this.
    """

    var sharded: ShardedSafeTensors

    # Two slot pairs: pinned host slab + device slab (Movable, not Copyable).
    var host0: HostBuffer[DType.uint8]
    var dev0: DeviceBuffer[DType.uint8]
    var host1: HostBuffer[DType.uint8]
    var dev1: DeviceBuffer[DType.uint8]
    var block_store: HostBuffer[DType.uint8]

    # Per-slot H2D-done events, recorded after the copy stream finishes staging.
    # await_block() calls ctx.stream().enqueue_wait_for(ev) to fence the default.
    var ev0: DeviceEvent
    var ev1: DeviceEvent

    # Per-slot compute-done events, recorded on the default stream after block
    # math has been queued. The copy stream waits on these before overwriting a
    # slot, matching the Rust BlockHandle drop lifecycle.
    var compute_done0: DeviceEvent
    var compute_done1: DeviceEvent
    var compute_recorded0: Bool
    var compute_recorded1: Bool

    # Explicit copy stream (one, shared across both slots).
    var copy_stream: DeviceStream

    # Per-slot staging metadata: tensor layout within the slab.
    var recs0: List[_TensorRecord]
    var recs1: List[_TensorRecord]

    # Prefix index built once at open(). Hot prefetches must not rescan every
    # safetensor name just to discover which tensors belong to one block.
    var index_prefixes: List[String]
    var index_starts: List[Int]
    var index_lengths: List[Int]
    var index_names: List[String]
    var store_offsets: List[Int]
    var store_nbytes: List[Int]

    # Slot state.
    var prefix0: String   # normalized prefix staged in slot 0 ("" = unused)
    var prefix1: String   # normalized prefix staged in slot 1 ("" = unused)
    var staged0: Bool     # copy-kernel dispatched for slot 0
    var staged1: Bool     # copy-kernel dispatched for slot 1
    var used0: Int        # bytes used in slot 0
    var used1: Int        # bytes used in slot 1

    # Which slot the model compute is currently reading.
    var active_slot: Int

    var slab_capacity: Int  # max block bytes; both slabs are this size
    var telemetry: OffloadTelemetry

    @staticmethod
    def open(dir: String, ctx: DeviceContext) raises -> TurboBlockLoader:
        """Open the model directory and pre-allocate two-slot async resources.

        Scans all tensor names to measure max block byte count, then allocates
        two pinned host + device slab pairs sized to that maximum."""
        var sharded = ShardedSafeTensors.open(dir)

        # Pass 1: compute max bytes across all blocks.
        # A block is all tensors sharing the same "X.N." prefix.
        var prefix_bytes = Dict[String, Int]()
        var prefixes = List[String]()
        for ref nm in sharded.names():
            var p = _extract_block_prefix(nm)
            var tv = sharded.tensor_view(nm)
            var nb = tv.nbytes()
            if p in prefix_bytes:
                prefix_bytes[p] += nb
            else:
                prefix_bytes[p] = nb
                prefixes.append(p)

        var max_bytes = 0
        for ref e in prefix_bytes.items():
            if e.value > max_bytes:
                max_bytes = e.value

        if max_bytes == 0:
            raise Error("TurboBlockLoader.open: no tensors found in " + dir)

        var index_starts = List[Int]()
        var index_lengths = List[Int]()
        var index_names = List[String]()
        for pi in range(len(prefixes)):
            var p = prefixes[pi].copy()
            var start = len(index_names)
            var count = 0
            for ref nm in sharded.names():
                if _extract_block_prefix(nm) == p:
                    index_names.append(nm)
                    count += 1
            index_starts.append(start)
            index_lengths.append(count)

        # Optional persistent pinned block store. The default keeps startup
        # bounded to two slot-sized pinned slabs; hot prefetches fill the active
        # host slot from mmap before dispatching H2D.
        var store_offsets = List[Int]()
        var store_nbytes = List[Int]()
        var total_store_bytes = 0
        for pi in range(len(prefixes)):
            store_offsets.append(total_store_bytes)
            var start = index_starts[pi]
            var end = start + index_lengths[pi]
            var block_bytes = 0
            for ni in range(start, end):
                var tv = sharded.tensor_view(index_names[ni])
                block_bytes += tv.nbytes()
            store_nbytes.append(block_bytes)
            total_store_bytes += block_bytes

        var block_store: HostBuffer[DType.uint8]
        comptime if TURBO_USE_PERSISTENT_BLOCK_STORE:
            block_store = ctx.enqueue_create_host_buffer[DType.uint8](total_store_bytes)
            for pi in range(len(prefixes)):
                var dst_base = store_offsets[pi]
                var offset = 0
                var start = index_starts[pi]
                var end = start + index_lengths[pi]
                for ni in range(start, end):
                    var tv = sharded.tensor_view(index_names[ni])
                    var nb = tv.nbytes()
                    var src = BytePtr(unsafe_from_address=Int(tv.data.unsafe_ptr()))
                    var dst = BytePtr(unsafe_from_address=Int(block_store.unsafe_ptr())) + dst_base + offset
                    _ = sys_memcpy(dst, src, nb)
                    offset += nb
        else:
            block_store = ctx.enqueue_create_host_buffer[DType.uint8](1)

        # Allocate two pinned host slabs and two device slabs.
        var host0 = ctx.enqueue_create_host_buffer[DType.uint8](max_bytes)
        var host1 = ctx.enqueue_create_host_buffer[DType.uint8](max_bytes)
        var dev0 = ctx.enqueue_create_buffer[DType.uint8](max_bytes)
        var dev1 = ctx.enqueue_create_buffer[DType.uint8](max_bytes)
        ctx.synchronize()

        # Allocate events and copy stream.
        var ev0 = ctx.create_event[disable_timing=True]()
        var ev1 = ctx.create_event[disable_timing=True]()
        var compute_done0 = ctx.create_event[disable_timing=True]()
        var compute_done1 = ctx.create_event[disable_timing=True]()
        var copy_stream = ctx.create_stream()

        return TurboBlockLoader(
            sharded^,
            host0^, dev0^,
            host1^, dev1^,
            block_store^,
            ev0^, ev1^,
            compute_done0^, compute_done1^,
            copy_stream^,
            prefixes^, index_starts^, index_lengths^, index_names^,
            store_offsets^, store_nbytes^,
            max_bytes,
        )

    def __init__(
        out self,
        var sharded: ShardedSafeTensors,
        var host0: HostBuffer[DType.uint8],
        var dev0: DeviceBuffer[DType.uint8],
        var host1: HostBuffer[DType.uint8],
        var dev1: DeviceBuffer[DType.uint8],
        var block_store: HostBuffer[DType.uint8],
        var ev0: DeviceEvent,
        var ev1: DeviceEvent,
        var compute_done0: DeviceEvent,
        var compute_done1: DeviceEvent,
        var copy_stream: DeviceStream,
        var index_prefixes: List[String],
        var index_starts: List[Int],
        var index_lengths: List[Int],
        var index_names: List[String],
        var store_offsets: List[Int],
        var store_nbytes: List[Int],
        slab_capacity: Int,
    ):
        self.sharded = sharded^
        self.host0 = host0^
        self.dev0 = dev0^
        self.host1 = host1^
        self.dev1 = dev1^
        self.block_store = block_store^
        self.ev0 = ev0^
        self.ev1 = ev1^
        self.compute_done0 = compute_done0^
        self.compute_done1 = compute_done1^
        self.compute_recorded0 = False
        self.compute_recorded1 = False
        self.copy_stream = copy_stream^
        self.recs0 = List[_TensorRecord]()
        self.recs1 = List[_TensorRecord]()
        self.index_prefixes = index_prefixes^
        self.index_starts = index_starts^
        self.index_lengths = index_lengths^
        self.index_names = index_names^
        self.store_offsets = store_offsets^
        self.store_nbytes = store_nbytes^
        self.prefix0 = String("")
        self.prefix1 = String("")
        self.staged0 = False
        self.staged1 = False
        self.used0 = 0
        self.used1 = 0
        self.active_slot = 0
        self.slab_capacity = slab_capacity
        self.telemetry = OffloadTelemetry(String("TurboBlockLoader"))

    def _norm(self, prefix: String) -> String:
        """Normalize block prefix to dot-terminated form (mirrors BlockLoader)."""
        return prefix if prefix.endswith(".") else prefix + "."

    def _idle_slot(self) -> Int:
        """The slot NOT currently being read by compute (the prefetch target)."""
        return 1 - self.active_slot

    def prefetch(
        mut self,
        prefix: String,
        ctx: DeviceContext,
    ) raises:
        """Stage `prefix` block into the idle slot, non-blocking.

        Steps:
          1. CPU memcpy: mmap bytes -> pinned host slab.
          2. GPU: dispatch _h2d_copy_kernel on copy_stream.
          3. copy_stream.record_event(ev)  -- fence anchor.
        await_block() later calls ctx.stream().enqueue_wait_for(ev)."""
        var p = self._norm(prefix)

        # Already staged?
        if self.prefix0 == p and self.staged0:
            self.telemetry.record_prefetch_hit(p)
            return
        if self.prefix1 == p and self.staged1:
            self.telemetry.record_prefetch_hit(p)
            return

        var slot = self._idle_slot()
        var telemetry_t0 = self.telemetry.now_ns()

        # ── Step 1: build this slot's tensor metadata.
        var new_recs = List[_TensorRecord]()
        var offset = 0

        var prefix_idx = -1
        for i in range(len(self.index_prefixes)):
            if self.index_prefixes[i] == p:
                prefix_idx = i
                break
        if prefix_idx < 0:
            raise Error(
                String("TurboBlockLoader.prefetch: no tensors for prefix: ") + p
            )
        var n_bytes = self.store_nbytes[prefix_idx]
        if n_bytes > self.slab_capacity:
            raise Error(
                String("TurboBlockLoader.prefetch: block exceeds slab: ")
                + String(n_bytes)
                + " > "
                + String(self.slab_capacity)
            )
        var start = self.index_starts[prefix_idx]
        var end = start + self.index_lengths[prefix_idx]
        for ni in range(start, end):
            var nm = self.index_names[ni].copy()
            var tv = self.sharded.tensor_view(nm)
            var nb = tv.nbytes()
            var sh = ArcPointer(tv.shape.copy())
            new_recs.append(
                _TensorRecord(nm, offset, nb, sh, tv.dtype)
            )
            offset += nb

        if offset == 0:
            raise Error(
                String("TurboBlockLoader.prefetch: no tensors for prefix: ") + p
            )

        # ── Step 2: dispatch H2D on explicit copy stream ──────────────────
        # Fast path: CUDA DMA/copy engine through cuMemcpyHtoDAsync_v2.
        # Fallback path: GPU copy kernel, kept only for portability/probes.
        var src_ptr = self.host0.unsafe_ptr()
        if slot == 1:
            src_ptr = self.host1.unsafe_ptr()
        comptime if TURBO_USE_PERSISTENT_BLOCK_STORE:
            src_ptr = self.block_store.unsafe_ptr() + self.store_offsets[prefix_idx]
        else:
            var host_offset = 0
            for ni in range(start, end):
                var tv = self.sharded.tensor_view(self.index_names[ni])
                var nb = tv.nbytes()
                var src = BytePtr(unsafe_from_address=Int(tv.data.unsafe_ptr()))
                var dst = BytePtr(unsafe_from_address=Int(src_ptr)) + host_offset
                _ = sys_memcpy(dst, src, nb)
                host_offset += nb

        # Rust parity: do not overwrite a slot until default-stream compute
        # from the previous block has passed the slot's compute_done event.
        if slot == 0 and self.compute_recorded0:
            self.copy_stream.enqueue_wait_for(self.compute_done0)
            self.compute_recorded0 = False
        elif slot == 1 and self.compute_recorded1:
            self.copy_stream.enqueue_wait_for(self.compute_done1)
            self.compute_recorded1 = False

        if slot == 0:
            comptime if TURBO_USE_DEFAULT_STREAM_COPY:
                var src_sub = self.block_store.create_sub_buffer[DType.uint8](
                    self.store_offsets[prefix_idx], n_bytes
                )
                var dst_sub = self.dev0.create_sub_buffer[DType.uint8](0, n_bytes)
                ctx.enqueue_copy(dst_buf=dst_sub, src_buf=src_sub)
                ctx.stream().record_event(self.ev0)
            else:
                comptime if TURBO_USE_CUDA_DMA_COPY:
                    _h2d_dma_copy(
                        UInt64(Int(self.dev0.unsafe_ptr())),
                        src_ptr,
                        n_bytes,
                        self.copy_stream,
                    )
                    self.copy_stream.record_event(self.ev0)
                else:
                    var compiled = ctx.compile_function[_h2d_copy_kernel, _h2d_copy_kernel]()
                    comptime COPY_BLOCK = 256
                    var n64 = n_bytes // 8
                    var copy_items = n64 + (n_bytes - n64 * 8)
                    var grid = (copy_items + COPY_BLOCK - 1) // COPY_BLOCK
                    self.copy_stream.enqueue_function(
                        compiled,
                        src_ptr,
                        self.dev0.unsafe_ptr(),
                        n_bytes,
                        grid_dim=grid,
                        block_dim=COPY_BLOCK,
                    )
                    self.copy_stream.record_event(self.ev0)
            # Update slot state.
            self.prefix0 = p
            self.staged0 = True
            self.used0 = offset
            self.recs0 = new_recs^
        else:
            comptime if TURBO_USE_DEFAULT_STREAM_COPY:
                var src_sub = self.block_store.create_sub_buffer[DType.uint8](
                    self.store_offsets[prefix_idx], n_bytes
                )
                var dst_sub = self.dev1.create_sub_buffer[DType.uint8](0, n_bytes)
                ctx.enqueue_copy(dst_buf=dst_sub, src_buf=src_sub)
                ctx.stream().record_event(self.ev1)
            else:
                comptime if TURBO_USE_CUDA_DMA_COPY:
                    _h2d_dma_copy(
                        UInt64(Int(self.dev1.unsafe_ptr())),
                        src_ptr,
                        n_bytes,
                        self.copy_stream,
                    )
                    self.copy_stream.record_event(self.ev1)
                else:
                    var compiled = ctx.compile_function[_h2d_copy_kernel, _h2d_copy_kernel]()
                    comptime COPY_BLOCK = 256
                    var n64 = n_bytes // 8
                    var copy_items = n64 + (n_bytes - n64 * 8)
                    var grid = (copy_items + COPY_BLOCK - 1) // COPY_BLOCK
                    self.copy_stream.enqueue_function(
                        compiled,
                        src_ptr,
                        self.dev1.unsafe_ptr(),
                        n_bytes,
                        grid_dim=grid,
                        block_dim=COPY_BLOCK,
                    )
                    self.copy_stream.record_event(self.ev1)
            self.prefix1 = p
            self.staged1 = True
            self.used1 = offset
            self.recs1 = new_recs^
        var telemetry_t1 = self.telemetry.now_ns()
        self.telemetry.record_prefetch(p, n_bytes, telemetry_t1 - telemetry_t0)

    def await_block(
        mut self,
        prefix: String,
        ctx: DeviceContext,
    ) raises -> Block:
        """Fence the default stream and return a Block of sub-buffer Tensor views.

        - If the block is not yet staged, calls prefetch() first.
        - Calls ctx.stream().enqueue_wait_for(ev) so the default stream waits for
          the copy kernel to finish writing the device slab.
        - Returns a Block of non-owning Tensor views backed by create_sub_buffer
          over the slot's persistent device slab."""
        var p = self._norm(prefix)

        # Determine which slot holds this block.
        var slot = -1
        if self.prefix0 == p and self.staged0:
            slot = 0
        elif self.prefix1 == p and self.staged1:
            slot = 1
        var slot_hit = slot >= 0
        var telemetry_t0 = self.telemetry.now_ns()

        if slot < 0:
            # Not yet staged: run prefetch now (will block copy stream
            # if copy stream is busy, but is non-blocking on default stream).
            self.prefetch(prefix, ctx)
            if self.prefix0 == p and self.staged0:
                slot = 0
            elif self.prefix1 == p and self.staged1:
                slot = 1
            if slot < 0:
                raise Error(
                    String("TurboBlockLoader.await_block: not staged: ") + p
                )

        # ── Fence: default stream waits for copy stream to finish this slot ──
        # This is the load-bearing ordering guarantee (Phase-0 finding):
        # without this, the default stream could read the device slab before
        # the copy kernel has finished writing it.
        if slot == 0:
            ctx.stream().enqueue_wait_for(self.ev0)
        else:
            ctx.stream().enqueue_wait_for(self.ev1)

        # Mark this slot active (idle_slot returns the other one for prefetch).
        self.active_slot = slot

        # ── Build Block from sub-buffer views ──────────────────────────────
        var block = Block()
        if slot == 0:
            for i in range(len(self.recs0)):
                var rec = self.recs0[i]
                var sub = self.dev0.create_sub_buffer[DType.uint8](
                    rec.offset, rec.nbytes
                )
                var t = Tensor(sub^, rec.shape[].copy(), rec.dtype)
                block[rec.name] = ArcPointer(t^)
        else:
            for i in range(len(self.recs1)):
                var rec = self.recs1[i]
                var sub = self.dev1.create_sub_buffer[DType.uint8](
                    rec.offset, rec.nbytes
                )
                var t = Tensor(sub^, rec.shape[].copy(), rec.dtype)
                block[rec.name] = ArcPointer(t^)

        var telemetry_t1 = self.telemetry.now_ns()
        self.telemetry.record_await(p, slot_hit, telemetry_t1 - telemetry_t0)
        return block^

    def mark_active_slot_compute_done(
        mut self,
        ctx: DeviceContext,
    ) raises:
        """Record a default-stream event for the slot returned by await_block().

        Call this after all kernels that read the current block have been
        queued. The next prefetch that reuses that slot will make the copy
        stream wait on this event before writing new weights into the slab.
        """
        if self.active_slot == 0:
            ctx.stream().record_event(self.compute_done0)
            self.compute_recorded0 = True
        else:
            ctx.stream().record_event(self.compute_done1)
            self.compute_recorded1 = True

    def async_enabled(self) -> Bool:
        return True

    def slab_bytes(self) -> Int:
        """Total bytes allocated across both slots (pinned host + device each)."""
        return self.slab_capacity * 4  # host0 + dev0 + host1 + dev1

    def print_telemetry(self):
        self.telemetry.print_summary()


# ─── helper: extract block prefix from a tensor name ─────────────────────────

def _extract_block_prefix(name: String) -> String:
    """Extract the block prefix = up to and including the first ALL-NUMERIC
    dot-segment (the block index), inclusive of its trailing dot.

    "layers.0.attn.weight"                              -> "layers.0."
    "layers.10.attn.weight"                             -> "layers.10."
    "model.diffusion_model.joint_blocks.0.x_block.w"    -> "model.diffusion_model.joint_blocks.0."
    "double_blocks.3.img_attn.norm.1.weight"            -> "double_blocks.3."
    Falls back to name + "." if no all-numeric segment is found.

    NOTE: the old rule cut after the 2nd dot ("X.N." only), which mis-grouped
    deeply-namespaced checkpoints like SD3.5 (`model.diffusion_model.joint_blocks.N.`)
    into one giant `model.diffusion_model.` super-block — breaking per-block
    prefetch ("no tensors for prefix") and oversizing the slab. The first-numeric
    rule is backward-compatible with `X.N.` naming (N is the first number)."""
    var ptr = name.unsafe_ptr()
    var n = name.byte_length()
    var seg_start = 0
    for i in range(n):
        var b = Int(ptr[i])
        if b == 46:  # ord('.')
            if i > seg_start:
                var allnum = True
                for j in range(seg_start, i):
                    var c = Int(ptr[j])
                    if c < 48 or c > 57:  # not '0'..'9'
                        allnum = False
                        break
                if allnum:
                    var result = String("")
                    for k in range(i + 1):
                        result += String(chr(Int(ptr[k])))
                    return result
            seg_start = i + 1
    # No all-numeric segment: fall back to name + "."
    var result = String("")
    for k in range(n):
        result += String(chr(Int(ptr[k])))
    if not result.endswith("."):
        result += "."
    return result
