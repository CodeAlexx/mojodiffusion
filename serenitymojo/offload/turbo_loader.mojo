# turbo_loader.mojo — TurboBlockLoader: async double-buffered weight offload.
#
# Phase 1 of the async offload backend. Overlaps H2D weight staging on an
# explicit copy stream with model compute on the default GPU stream, using two
# persistent slot pairs (pinned host slab + device slab + DeviceEvent).
#
# DESIGN (Phase-0 established; Phase-1 implements):
#   - DeviceContext() is a singleton. Model compute runs on the DEFAULT stream.
#   - enqueue_copy() is DEFAULT-stream only — cannot stage weights async.
#   - H2D staging runs as a GPU byte-copy KERNEL on an explicit copy stream
#     created via ctx.create_stream(). Pinned host buffers are device-accessible,
#     so the copy kernel reads pinned host bytes and writes the device slab.
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

from std.memory import ArcPointer
from std.gpu.host import DeviceContext, HostBuffer, DeviceBuffer, DeviceStream, DeviceEvent
from std.gpu import global_idx
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.offload.block_loader import Block


# ─── copy kernel ──────────────────────────────────────────────────────────────
# Byte-wise H2D kernel. Reads pinned host slab (device-accessible via DMA),
# writes device slab. One thread per byte. Dispatched on the explicit copy stream.

def _h2d_copy_kernel(
    src: UnsafePointer[UInt8, MutAnyOrigin],
    dst: UnsafePointer[UInt8, MutAnyOrigin],
    n: Int,
):
    """Copy n bytes: dst[i] = src[i]. Called on the explicit copy stream."""
    var i = Int(global_idx.x)
    if i < n:
        dst[i] = src[i]


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

    # Per-slot events: one event per slot, recorded after copy kernel completes.
    # await_block() calls ctx.stream().enqueue_wait_for(ev) to fence the default.
    var ev0: DeviceEvent
    var ev1: DeviceEvent

    # Explicit copy stream (one, shared across both slots).
    var copy_stream: DeviceStream

    # Per-slot staging metadata: tensor layout within the slab.
    var recs0: List[_TensorRecord]
    var recs1: List[_TensorRecord]

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

    @staticmethod
    def open(dir: String, ctx: DeviceContext) raises -> TurboBlockLoader:
        """Open the model directory and pre-allocate two-slot async resources.

        Scans all tensor names to measure max block byte count, then allocates
        two pinned host + device slab pairs sized to that maximum."""
        var sharded = ShardedSafeTensors.open(dir)

        # Pass 1: compute max bytes across all blocks.
        # A block is all tensors sharing the same "X.N." prefix.
        var prefix_bytes = Dict[String, Int]()
        for ref nm in sharded.names():
            var p = _extract_block_prefix(nm)
            var tv = sharded.tensor_view(nm)
            var nb = tv.nbytes()
            if p in prefix_bytes:
                prefix_bytes[p] += nb
            else:
                prefix_bytes[p] = nb

        var max_bytes = 0
        for ref e in prefix_bytes.items():
            if e.value > max_bytes:
                max_bytes = e.value

        if max_bytes == 0:
            raise Error("TurboBlockLoader.open: no tensors found in " + dir)

        # Allocate two pinned host slabs and two device slabs.
        var host0 = ctx.enqueue_create_host_buffer[DType.uint8](max_bytes)
        var host1 = ctx.enqueue_create_host_buffer[DType.uint8](max_bytes)
        var dev0 = ctx.enqueue_create_buffer[DType.uint8](max_bytes)
        var dev1 = ctx.enqueue_create_buffer[DType.uint8](max_bytes)
        ctx.synchronize()

        # Allocate events and copy stream.
        var ev0 = ctx.create_event[disable_timing=True]()
        var ev1 = ctx.create_event[disable_timing=True]()
        var copy_stream = ctx.create_stream()

        return TurboBlockLoader(
            sharded^,
            host0^, dev0^,
            host1^, dev1^,
            ev0^, ev1^,
            copy_stream^,
            max_bytes,
        )

    def __init__(
        out self,
        var sharded: ShardedSafeTensors,
        var host0: HostBuffer[DType.uint8],
        var dev0: DeviceBuffer[DType.uint8],
        var host1: HostBuffer[DType.uint8],
        var dev1: DeviceBuffer[DType.uint8],
        var ev0: DeviceEvent,
        var ev1: DeviceEvent,
        var copy_stream: DeviceStream,
        slab_capacity: Int,
    ):
        self.sharded = sharded^
        self.host0 = host0^
        self.dev0 = dev0^
        self.host1 = host1^
        self.dev1 = dev1^
        self.ev0 = ev0^
        self.ev1 = ev1^
        self.copy_stream = copy_stream^
        self.recs0 = List[_TensorRecord]()
        self.recs1 = List[_TensorRecord]()
        self.prefix0 = String("")
        self.prefix1 = String("")
        self.staged0 = False
        self.staged1 = False
        self.used0 = 0
        self.used1 = 0
        self.active_slot = 0
        self.slab_capacity = slab_capacity

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
            return
        if self.prefix1 == p and self.staged1:
            return

        var slot = self._idle_slot()

        # ── Step 1: CPU memcpy from mmap into pinned host slab ─────────────
        var new_recs = List[_TensorRecord]()
        var offset = 0

        for ref nm in self.sharded.names():
            if nm.startswith(p):
                var tv = self.sharded.tensor_view(nm)
                var nb = tv.nbytes()
                if offset + nb > self.slab_capacity:
                    raise Error(
                        String("TurboBlockLoader.prefetch: block exceeds slab: ")
                        + String(offset + nb)
                        + " > "
                        + String(self.slab_capacity)
                    )
                if slot == 0:
                    var hp = self.host0.unsafe_ptr()
                    for i in range(nb):
                        hp[offset + i] = tv.data[i]
                else:
                    var hp = self.host1.unsafe_ptr()
                    for i in range(nb):
                        hp[offset + i] = tv.data[i]

                var sh = ArcPointer(tv.shape.copy())
                new_recs.append(
                    _TensorRecord(nm, offset, nb, sh, tv.dtype)
                )
                offset += nb

        if offset == 0:
            raise Error(
                String("TurboBlockLoader.prefetch: no tensors for prefix: ") + p
            )

        # ── Step 2: dispatch copy kernel on explicit copy stream ────────────
        # compile_function is JIT-cached by the DeviceContext: first call ~47µs
        # (cold PTX compile), subsequent calls ~500ns (cache lookup). No re-JIT per
        # prefetch. DeviceFunction[...] cannot be stored as an unparameterized struct
        # field in MAX 26.3 without making TurboBlockLoader parametric; the cache
        # makes this a non-issue in the hot path.
        var compiled = ctx.compile_function[_h2d_copy_kernel, _h2d_copy_kernel]()
        var n_bytes = offset
        comptime COPY_BLOCK = 256
        var grid = (n_bytes + COPY_BLOCK - 1) // COPY_BLOCK

        if slot == 0:
            self.copy_stream.enqueue_function(
                compiled,
                self.host0.unsafe_ptr(),
                self.dev0.unsafe_ptr(),
                n_bytes,
                grid_dim=grid,
                block_dim=COPY_BLOCK,
            )
            # ── Step 3: record event on copy stream ─────────────────────────
            self.copy_stream.record_event(self.ev0)
            # Update slot state.
            self.prefix0 = p
            self.staged0 = True
            self.used0 = offset
            self.recs0 = new_recs^
        else:
            self.copy_stream.enqueue_function(
                compiled,
                self.host1.unsafe_ptr(),
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

        return block^

    def async_enabled(self) -> Bool:
        return True

    def slab_bytes(self) -> Int:
        """Total bytes allocated across both slots (pinned host + device each)."""
        return self.slab_capacity * 4  # host0 + dev0 + host1 + dev1


# ─── helper: extract block prefix from a tensor name ─────────────────────────

def _extract_block_prefix(name: String) -> String:
    """Extract the two-component block prefix from a tensor name.

    "layers.0.attn.weight"  -> "layers.0."
    "layers.10.attn.weight" -> "layers.10."
    Falls back to name + "." if fewer than two dots found."""
    var dots = 0
    var result = String("")
    var ptr = name.unsafe_ptr()
    for i in range(name.byte_length()):
        var b = Int(ptr[i])
        # Append char to result using chr().
        result += String(chr(b))
        if b == 46:  # ord('.')
            dots += 1
            if dots == 2:
                return result
    # Fewer than 2 dots: use name + "." as the prefix.
    if not result.endswith("."):
        result += "."
    return result
