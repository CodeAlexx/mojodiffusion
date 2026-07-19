# memory/arena.mojo — a bump / region (arena) allocator.
#
# Grab one big block up front, hand out aligned slices by bumping an offset, and
# free everything at once with reset() (O(1)) or on drop. Ideal for "allocate a
# burst of temporaries, then throw them all away" — per-frame, per-request,
# per-token-step. Individual allocations are NOT freed; only the whole arena.
#
# The hot path (alloc_bytes) is @always_inline: an align + bounds check + pointer
# bump, no syscall, no per-object bookkeeping beyond two counters.

from std.memory import alloc, UnsafePointer
from std.sys.info import size_of, align_of

from mem.aligned import align_up, BytePtr


struct Arena(Movable):
    """A bump allocator over a fixed-capacity heap block.

    alloc_bytes / alloc_array bump an internal offset. reset() reclaims the whole
    arena in O(1). The block is freed once on drop. Not thread-safe."""

    var _base: BytePtr
    var _capacity: Int
    var _offset: Int
    var _peak: Int
    var _alloc_count: Int

    @staticmethod
    def with_capacity(capacity: Int) raises -> Arena:
        if capacity <= 0:
            raise Error("Arena: capacity must be > 0")
        var base = alloc[UInt8](capacity)
        return Arena(base, capacity)

    def __init__(out self, var base: BytePtr, capacity: Int):
        self._base = base
        self._capacity = capacity
        self._offset = 0
        self._peak = 0
        self._alloc_count = 0

    @always_inline
    def alloc_bytes(
        mut self, nbytes: Int, alignment: Int = 16
    ) raises -> BytePtr:
        """Bump-allocate `nbytes` aligned to `alignment`. Raises when full."""
        var start = align_up(self._offset, alignment)
        if start + nbytes > self._capacity:
            raise Error(
                String("Arena: out of memory (need ")
                + String(nbytes)
                + String(", have ")
                + String(self._capacity - start)
                + String(")")
            )
        var p = self._base + start
        self._offset = start + nbytes
        if self._offset > self._peak:
            self._peak = self._offset
        self._alloc_count += 1
        return p

    @always_inline
    def alloc_array[T: AnyType](mut self, count: Int) raises -> UnsafePointer[T, MutExternalOrigin]:
        """Bump-allocate `count` elements of T, correctly sized and aligned."""
        var nbytes = count * size_of[T]()
        var p = self.alloc_bytes(nbytes, align_of[T]())
        return p.bitcast[T]()

    @always_inline
    def reset(mut self):
        """Reclaim the entire arena in O(1). Existing pointers become invalid."""
        self._offset = 0
        self._alloc_count = 0

    def used(self) -> Int:
        return self._offset

    def remaining(self) -> Int:
        return self._capacity - self._offset

    def capacity(self) -> Int:
        return self._capacity

    def peak_used(self) -> Int:
        return self._peak

    def alloc_count(self) -> Int:
        return self._alloc_count

    def __del__(deinit self):
        self._base.free()
