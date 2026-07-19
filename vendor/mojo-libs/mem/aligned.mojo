# memory/aligned.mojo — alignment helpers + an aligned heap buffer.
#
# Mojo's `alloc[T]` aligns to the element type only. For SIMD / cache-line work
# you sometimes need a buffer aligned to 32/64 bytes regardless of element type.
# AlignedBuffer over-allocates and returns an aligned pointer, freeing the
# original allocation exactly once on drop (Movable, like graphics.Canvas).
#
# The align math is branch-light and @always_inline so it disappears into the
# hot allocator paths that call it.

from std.memory import alloc, UnsafePointer

comptime BytePtr = UnsafePointer[UInt8, MutExternalOrigin]


@always_inline("nodebug")
def is_power_of_two(x: Int) -> Bool:
    return x > 0 and (x & (x - 1)) == 0


@always_inline("nodebug")
def align_up(n: Int, alignment: Int) -> Int:
    """Round n up to a multiple of `alignment` (any positive alignment)."""
    if alignment <= 1:
        return n
    var rem = n % alignment
    if rem == 0:
        return n
    return n + (alignment - rem)


@always_inline("nodebug")
def align_up_pow2(n: Int, alignment: Int) -> Int:
    """Round n up to a power-of-two `alignment` (branchless; alignment must be 2^k)."""
    return (n + alignment - 1) & ~(alignment - 1)


@always_inline("nodebug")
def align_down_pow2(n: Int, alignment: Int) -> Int:
    """Round n down to a power-of-two `alignment`."""
    return n & ~(alignment - 1)


struct AlignedBuffer(Movable):
    """A heap buffer whose data pointer is aligned to `alignment` bytes.

    Owns the underlying allocation and frees it once on drop. `data()` returns
    the aligned pointer; do not free it yourself."""

    var _raw: BytePtr   # original allocation (the one we free)
    var _ptr: BytePtr   # aligned pointer into _raw
    var _size: Int
    var _alignment: Int

    @staticmethod
    def alloc_aligned(size: Int, alignment: Int) raises -> AlignedBuffer:
        if size <= 0:
            raise Error("AlignedBuffer: size must be > 0")
        if not is_power_of_two(alignment):
            raise Error("AlignedBuffer: alignment must be a power of two")
        # Over-allocate so an aligned start always exists within the block.
        var raw = alloc[UInt8](size + alignment - 1)
        var base = Int(raw)
        var aligned = align_up_pow2(base, alignment)
        var ptr = raw + (aligned - base)
        return AlignedBuffer(raw, ptr, size, alignment)

    def __init__(
        out self,
        var raw: BytePtr,
        var ptr: BytePtr,
        size: Int,
        alignment: Int,
    ):
        self._raw = raw
        self._ptr = ptr
        self._size = size
        self._alignment = alignment

    @always_inline
    def data(self) -> BytePtr:
        return self._ptr

    def size(self) -> Int:
        return self._size

    def alignment(self) -> Int:
        return self._alignment

    def is_aligned(self) -> Bool:
        return (Int(self._ptr) % self._alignment) == 0

    def __del__(deinit self):
        self._raw.free()
