# mem/ring.mojo — a fixed-capacity circular byte buffer (FIFO) over ONE heap
# allocation.
#
# A ByteRing stores up to `capacity` bytes in a contiguous block, tracking a
# `head` (read index), `tail` (write index) and an explicit `count`. The count
# distinguishes full from empty cleanly (head == tail can mean either), so the
# whole capacity is usable — no sacrificed slot, no power-of-two requirement.
#
# Bulk push/pop copy in at most TWO contiguous runs (up to the end of the block,
# then wrapping to the start) via memcpy, rather than byte-by-byte, so a 256-byte
# transfer is one or two memcpys.
#
# A power-of-two capacity is detected at construction; when present, the wrap
# arithmetic uses a mask (`& (cap-1)`) instead of `% cap` on the single-byte hot
# paths. Otherwise it falls back to a branch (`if idx == cap: idx = 0`), which is
# still branch-predictor-friendly for a steady stream.
#
# The block is freed exactly once on drop (Movable, like mem.Pool / mem.Arena).
#
# NOT thread-safe and NOT atomic: a single producer/consumer on ONE thread only.
# There is no memory ordering, no lock, no atomic head/tail. Concurrent access
# from multiple threads is a data race.

from std.memory import alloc, memcpy, UnsafePointer
from mem.aligned import is_power_of_two, BytePtr


struct ByteRing(Movable):
    """A fixed-capacity circular byte FIFO backed by one heap allocation.

    Uses head/tail/count: `count` cleanly separates full vs empty so every byte
    of `capacity` is usable. push/pop of a single byte are O(1); bulk push/pop
    are at most two memcpys (the run up to the wrap, then from the start). Frees
    its allocation once on drop. Single-threaded only — NOT thread-safe."""

    var _buf: BytePtr     # the storage block (the one we free)
    var _cap: Int         # capacity in bytes
    var _head: Int        # read index  [0, cap)
    var _tail: Int        # write index [0, cap)
    var _count: Int       # bytes currently stored [0, cap]
    var _mask: Int        # cap-1 when cap is a power of two, else -1 (no mask)

    @staticmethod
    def with_capacity(capacity: Int) raises -> ByteRing:
        if capacity <= 0:
            raise Error("ByteRing: capacity must be > 0")
        var buf = alloc[UInt8](capacity)
        var mask = (capacity - 1) if is_power_of_two(capacity) else -1
        return ByteRing(buf, capacity, mask)

    def __init__(out self, var buf: BytePtr, capacity: Int, mask: Int):
        self._buf = buf
        self._cap = capacity
        self._head = 0
        self._tail = 0
        self._count = 0
        self._mask = mask

    @always_inline
    def _advance(self, idx: Int) -> Int:
        """Move an index one byte forward, wrapping at capacity."""
        if self._mask >= 0:
            return (idx + 1) & self._mask
        var n = idx + 1
        return 0 if n == self._cap else n

    # ---------------- single-byte hot paths ----------------

    @always_inline
    def push_byte(mut self, b: UInt8) raises:
        """Append one byte. Raises if the ring is full."""
        if self._count == self._cap:
            raise Error("ByteRing.push_byte: full")
        self._buf[self._tail] = b
        self._tail = self._advance(self._tail)
        self._count += 1

    @always_inline
    def try_push_byte(mut self, b: UInt8) -> Bool:
        """Append one byte; return False (no-op) if full."""
        if self._count == self._cap:
            return False
        self._buf[self._tail] = b
        self._tail = self._advance(self._tail)
        self._count += 1
        return True

    @always_inline
    def pop_byte(mut self) raises -> UInt8:
        """Remove and return the oldest byte. Raises if empty."""
        if self._count == 0:
            raise Error("ByteRing.pop_byte: empty")
        var b = self._buf[self._head]
        self._head = self._advance(self._head)
        self._count -= 1
        return b

    @always_inline
    def try_pop_byte(mut self, mut out_b: UInt8) -> Bool:
        """Pop the oldest byte into out_b; return False (no-op) if empty."""
        if self._count == 0:
            return False
        out_b = self._buf[self._head]
        self._head = self._advance(self._head)
        self._count -= 1
        return True

    # ---------------- bulk paths (at most two memcpys) ----------------

    def push(mut self, src: BytePtr, n: Int) -> Int:
        """Copy up to free_space() bytes from src; return bytes written.

        Wraps correctly: writes the run up to the end of the block, then the
        remainder from the start — at most two memcpys."""
        var free = self._cap - self._count
        var want = n if n < free else free
        if want <= 0:
            return 0
        # first contiguous run: from tail to end of block
        var first = self._cap - self._tail
        if first > want:
            first = want
        memcpy(dest=self._buf + self._tail, src=src, count=first)
        var rest = want - first
        if rest > 0:
            # wrapped run: from start of block
            memcpy(dest=self._buf, src=src + first, count=rest)
        # advance tail (mask or modulo, single op)
        self._tail = (self._tail + want) % self._cap
        self._count += want
        return want

    def pop(mut self, dst: BytePtr, n: Int) -> Int:
        """Copy up to len() bytes into dst; return bytes read.

        Wraps correctly: reads the run up to the end of the block, then the
        remainder from the start — at most two memcpys."""
        var avail = self._count
        var want = n if n < avail else avail
        if want <= 0:
            return 0
        # first contiguous run: from head to end of block
        var first = self._cap - self._head
        if first > want:
            first = want
        memcpy(dest=dst, src=self._buf + self._head, count=first)
        var rest = want - first
        if rest > 0:
            memcpy(dest=dst + first, src=self._buf, count=rest)
        self._head = (self._head + want) % self._cap
        self._count -= want
        return want

    # ---------------- queries ----------------

    @always_inline
    def len(self) -> Int:
        return self._count

    @always_inline
    def free_space(self) -> Int:
        return self._cap - self._count

    @always_inline
    def capacity(self) -> Int:
        return self._cap

    @always_inline
    def is_empty(self) -> Bool:
        return self._count == 0

    @always_inline
    def is_full(self) -> Bool:
        return self._count == self._cap

    def clear(mut self):
        """Discard all stored bytes (data is not zeroed)."""
        self._head = 0
        self._tail = 0
        self._count = 0

    def __del__(deinit self):
        self._buf.free()
