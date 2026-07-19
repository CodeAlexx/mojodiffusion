# mem/pool.mojo — a fixed-size object pool over ONE heap allocation.
#
# A pool carves a single aligned block into `capacity` fixed-size cells and hands
# them out / takes them back in O(1) via an intrusive freelist: each FREE cell's
# first 8 bytes store the index of the next free cell (-1 = end of list). No
# per-acquire syscall, no per-object header — acquire/release are a pop/push on
# the freelist. Ideal for churny same-size objects (nodes, requests, particles).
#
# The block is over-allocated by `align` bytes so the data base can be rounded up
# to a cache-line boundary; the ORIGINAL allocation is freed once on drop
# (Movable, like graphics.Canvas / mem.Arena). Not thread-safe.

from std.memory import alloc, UnsafePointer
from mem.aligned import align_up, is_power_of_two, BytePtr


struct Pool(Movable):
    """A pool of `capacity` fixed-size blocks backed by one heap allocation.

    `block_size` is rounded up to at least max(align, 8) and to a multiple of
    `align`, so every block is cache-line aligned and large enough to hold the
    freelist Int. acquire()/release() are O(1) freelist pop/push; reset() rebuilds
    the whole freelist in O(capacity). Pointers handed out by acquire() are
    `align`-aligned; release() validates that a returned pointer belongs to this
    pool and is correctly aligned, raising on a foreign/misaligned pointer."""

    var _raw: BytePtr     # original allocation (the one we free)
    var _base: BytePtr    # aligned data base (block 0)
    var _block: Int       # bytes per block (>= 8, multiple of align)
    var _capacity: Int    # number of blocks
    var _align: Int       # alignment of each block / the base
    var _free_head: Int   # index of first free block, or -1 if exhausted
    var _free_count: Int  # blocks currently free

    @staticmethod
    def create(block_size: Int, capacity: Int, align: Int = 64) raises -> Pool:
        if capacity <= 0:
            raise Error("Pool: capacity must be > 0")
        if block_size <= 0:
            raise Error("Pool: block_size must be > 0")
        if not is_power_of_two(align):
            raise Error("Pool: align must be a power of two")
        # Block must hold the freelist Int (8B) AND be align-aligned & big enough.
        var min_block = align if align > 8 else 8
        var want = block_size if block_size > min_block else min_block
        var block = align_up(want, align)
        # Over-allocate by `align` so we can round the base up to an align boundary.
        var raw = alloc[UInt8](capacity * block + align)
        var b = Int(raw)
        var aligned = align_up(b, align)
        var base = raw + (aligned - b)
        var p = Pool(raw, base, block, capacity, align)
        p.reset()
        return p^

    def __init__(
        out self,
        var raw: BytePtr,
        var base: BytePtr,
        block: Int,
        capacity: Int,
        align: Int,
    ):
        self._raw = raw
        self._base = base
        self._block = block
        self._capacity = capacity
        self._align = align
        self._free_head = 0
        self._free_count = capacity

    @always_inline
    def _slot(self, idx: Int) -> UnsafePointer[Int, MutExternalOrigin]:
        """Pointer to the intrusive freelist Int stored in block `idx`."""
        return (self._base + idx * self._block).bitcast[Int]()

    def reset(mut self):
        """Rebuild the full freelist: every block free again, in O(capacity)."""
        for i in range(self._capacity):
            self._slot(i)[0] = i + 1 if i + 1 < self._capacity else -1
        self._free_head = 0
        self._free_count = self._capacity

    @always_inline
    def acquire(mut self) raises -> BytePtr:
        """Pop and return a free block (align-aligned). Raises when exhausted."""
        var idx = self._free_head
        if idx < 0:
            raise Error("Pool exhausted")
        self._free_head = self._slot(idx)[0]
        self._free_count -= 1
        return self._base + idx * self._block

    @always_inline
    def release(mut self, p: BytePtr) raises:
        """Return a block to the pool. Raises on a foreign/misaligned pointer."""
        var off = Int(p) - Int(self._base)
        if off < 0 or off % self._block != 0:
            raise Error("Pool.release: foreign or misaligned pointer")
        var idx = off // self._block
        if idx >= self._capacity:
            raise Error("Pool.release: pointer out of range")
        var slot = self._slot(idx)
        slot[0] = self._free_head
        self._free_head = idx
        self._free_count += 1

    def available(self) -> Int:
        return self._free_count

    def in_use(self) -> Int:
        return self._capacity - self._free_count

    def capacity(self) -> Int:
        return self._capacity

    def block_size(self) -> Int:
        return self._block

    def __del__(deinit self):
        self._raw.free()
