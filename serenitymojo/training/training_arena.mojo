# training/training_arena.mojo — shared per-step training scratch arena.
#
# Product trainers should use this for forward/backward/loss/optimizer
# temporaries that do not need to survive the current step phase. It wraps the
# existing scratch ring instead of inventing a second allocator. Rewinds are
# host-side cursor moves only; callers must not rewind past tensors still needed
# by queued device work.

from std.gpu.host import DeviceBuffer, DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.scratch_ring import ScratchRingAllocator, ScratchRingMark
from serenitymojo.tensor import Tensor


comptime TRAINING_ARENA_PHASE_SETUP = 0
comptime TRAINING_ARENA_PHASE_FORWARD = 1
comptime TRAINING_ARENA_PHASE_LOSS = 2
comptime TRAINING_ARENA_PHASE_BACKWARD = 3
comptime TRAINING_ARENA_PHASE_OPTIMIZER = 4
comptime TRAINING_ARENA_PHASE_SAVE_SAMPLE = 5

comptime TRAINING_ARENA_SYNC_SCALAR_LOG = 0
comptime TRAINING_ARENA_SYNC_CHECKPOINT = 1
comptime TRAINING_ARENA_SYNC_PROFILER = 2
comptime TRAINING_ARENA_SYNC_CORRECTNESS_GATE = 3
comptime TRAINING_ARENA_SYNC_OPTIMIZER = 4


@fieldwise_init
struct TrainingArenaMark(Copyable, Movable):
    var phase: Int
    var ring_mark: ScratchRingMark
    var allocation_count: Int
    var allocated_bytes: Int


@fieldwise_init
struct TrainingArenaStats(Copyable, Movable, Writable):
    # Counters are lifetime totals since arena construction or an explicit
    # caller reset, not live allocation counts. Use mark/rewind for storage
    # lifetime; snapshot stats before/after a phase when phase deltas are needed.
    var allocation_count: Int
    var allocated_bytes: Int
    var peak_bytes: Int
    var capacity_bytes: Int
    var current_used_bytes: Int
    var rewind_count: Int
    var sync_count: Int
    var host_device_transfer_count: Int
    var scalar_sync_count: Int
    var checkpoint_sync_count: Int
    var profiler_sync_count: Int
    var correctness_sync_count: Int
    var optimizer_sync_count: Int

    def validate(self) raises:
        if self.allocation_count < 0 or self.allocated_bytes < 0:
            raise Error("TrainingArenaStats: allocation counters must be nonnegative")
        if self.peak_bytes < 0 or self.capacity_bytes < 0 or self.current_used_bytes < 0:
            raise Error("TrainingArenaStats: byte counters must be nonnegative")
        if self.peak_bytes > self.capacity_bytes:
            raise Error("TrainingArenaStats: peak exceeds capacity")
        if self.current_used_bytes > self.capacity_bytes:
            raise Error("TrainingArenaStats: current usage exceeds capacity")
        if self.rewind_count < 0 or self.sync_count < 0:
            raise Error("TrainingArenaStats: rewind/sync counters must be nonnegative")
        if self.host_device_transfer_count < 0:
            raise Error("TrainingArenaStats: transfer count must be nonnegative")
        if (
            self.scalar_sync_count
            + self.checkpoint_sync_count
            + self.profiler_sync_count
            + self.correctness_sync_count
            + self.optimizer_sync_count
            > self.sync_count
        ):
            raise Error("TrainingArenaStats: reason sync counts exceed total syncs")

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "TrainingArenaStats(allocs=",
            self.allocation_count,
            ", allocated_bytes=",
            self.allocated_bytes,
            ", peak_bytes=",
            self.peak_bytes,
            ", capacity_bytes=",
            self.capacity_bytes,
            ", used_bytes=",
            self.current_used_bytes,
            ", rewinds=",
            self.rewind_count,
            ", syncs=",
            self.sync_count,
            ", transfers=",
            self.host_device_transfer_count,
            ")",
        )


def training_arena_phase_name(phase: Int) -> String:
    if phase == TRAINING_ARENA_PHASE_SETUP:
        return String("setup")
    if phase == TRAINING_ARENA_PHASE_FORWARD:
        return String("forward")
    if phase == TRAINING_ARENA_PHASE_LOSS:
        return String("loss")
    if phase == TRAINING_ARENA_PHASE_BACKWARD:
        return String("backward")
    if phase == TRAINING_ARENA_PHASE_OPTIMIZER:
        return String("optimizer")
    if phase == TRAINING_ARENA_PHASE_SAVE_SAMPLE:
        return String("save-sample")
    return String("unknown")


def _valid_phase(phase: Int) -> Bool:
    return (
        phase == TRAINING_ARENA_PHASE_SETUP
        or phase == TRAINING_ARENA_PHASE_FORWARD
        or phase == TRAINING_ARENA_PHASE_LOSS
        or phase == TRAINING_ARENA_PHASE_BACKWARD
        or phase == TRAINING_ARENA_PHASE_OPTIMIZER
        or phase == TRAINING_ARENA_PHASE_SAVE_SAMPLE
    )


struct TrainingArena(Movable):
    var ring: ScratchRingAllocator
    var allocation_count: Int
    var allocated_bytes: Int
    var rewind_count: Int
    var sync_count: Int
    var host_device_transfer_count: Int
    var scalar_sync_count: Int
    var checkpoint_sync_count: Int
    var profiler_sync_count: Int
    var correctness_sync_count: Int
    var optimizer_sync_count: Int

    def __init__(
        out self,
        ctx: DeviceContext,
        slab_bytes: Int,
        num_slabs: Int = 1,
        alignment: Int = 256,
    ) raises:
        self.ring = ScratchRingAllocator(ctx, slab_bytes, num_slabs, alignment)
        self.allocation_count = 0
        self.allocated_bytes = 0
        self.rewind_count = 0
        self.sync_count = 0
        self.host_device_transfer_count = 0
        self.scalar_sync_count = 0
        self.checkpoint_sync_count = 0
        self.profiler_sync_count = 0
        self.correctness_sync_count = 0
        self.optimizer_sync_count = 0

    def mark(mut self, phase: Int) raises -> TrainingArenaMark:
        if not _valid_phase(phase):
            raise Error(
                String("TrainingArena.mark: invalid phase ")
                + String(phase)
            )
        return TrainingArenaMark(
            phase,
            self.ring.mark(),
            self.allocation_count,
            self.allocated_bytes,
        )

    def rewind(mut self, mark: TrainingArenaMark) raises:
        if not _valid_phase(mark.phase):
            raise Error("TrainingArena.rewind: invalid mark phase")
        self.ring.rewind(mark.ring_mark)
        self.rewind_count += 1

    def reset_step(mut self):
        self.ring.reset()

    def alloc_bytes(mut self, nbytes: Int) raises -> DeviceBuffer[DType.uint8]:
        if nbytes <= 0:
            raise Error("TrainingArena.alloc_bytes: nbytes must be positive")
        var buf = self.ring._alloc_buffer(nbytes)
        self.allocation_count += 1
        self.allocated_bytes += nbytes
        return buf^

    def alloc_tensor(
        mut self, var shape: List[Int], dtype: STDtype
    ) raises -> Tensor:
        var n = 1
        for i in range(len(shape)):
            n *= shape[i]
        if n <= 0:
            raise Error("TrainingArena.alloc_tensor: empty tensor")
        var buf = self.alloc_bytes(n * dtype.byte_size())
        return Tensor(buf^, shape^, dtype)

    def empty_like(mut self, x: Tensor) raises -> Tensor:
        return self.alloc_tensor(x.shape(), x.dtype())

    def clone_tensor(mut self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var out = self.empty_like(x)
        ctx.enqueue_copy(dst_buf=out.buf, src_buf=x.buf)
        return out^

    def record_host_device_transfer(mut self, count: Int = 1) raises:
        if count < 0:
            raise Error("TrainingArena.record_host_device_transfer: negative count")
        self.host_device_transfer_count += count

    def record_sync(mut self, reason: Int, count: Int = 1) raises:
        if count < 0:
            raise Error("TrainingArena.record_sync: negative count")
        if reason == TRAINING_ARENA_SYNC_SCALAR_LOG:
            self.scalar_sync_count += count
        elif reason == TRAINING_ARENA_SYNC_CHECKPOINT:
            self.checkpoint_sync_count += count
        elif reason == TRAINING_ARENA_SYNC_PROFILER:
            self.profiler_sync_count += count
        elif reason == TRAINING_ARENA_SYNC_CORRECTNESS_GATE:
            self.correctness_sync_count += count
        elif reason == TRAINING_ARENA_SYNC_OPTIMIZER:
            self.optimizer_sync_count += count
        else:
            raise Error(
                String("TrainingArena.record_sync: invalid reason ")
                + String(reason)
            )
        self.sync_count += count

    def synchronize_for(mut self, ctx: DeviceContext, reason: Int) raises:
        self.record_sync(reason)
        ctx.synchronize()

    def stats(self) raises -> TrainingArenaStats:
        var s = TrainingArenaStats(
            self.allocation_count,
            self.allocated_bytes,
            self.ring.peak_bytes,
            self.ring.capacity_bytes(),
            self.ring.used_bytes(),
            self.rewind_count,
            self.sync_count,
            self.host_device_transfer_count,
            self.scalar_sync_count,
            self.checkpoint_sync_count,
            self.profiler_sync_count,
            self.correctness_sync_count,
            self.optimizer_sync_count,
        )
        s.validate()
        return s^
