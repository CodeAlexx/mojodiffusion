# mem/slab.mojo — a size-class SLAB allocator (general-purpose, heavy-use).
#
# This is the workhorse allocator: O(1) allocate / free for any object that fits
# in one of a fixed set of size classes. Each class owns an intrusive singly
# linked free list of fixed-size blocks. Blocks are carved on demand from large
# backing CHUNKS (1 MiB) that we alloc lazily and remember so we can free them
# all on drop.
#
# Layout of every pooled allocation:
#
#     [ 8-byte header | class_size bytes of user data ]
#       ^block          ^user ptr (= block + 8)
#
# The header stores the size-class index. free(user_ptr) reads it back from
# (user_ptr - 8) to know which free list the block belongs to. No per-object
# search, no metadata table — the class id rides along with the block.
#
# Free list intrusiveness: a free block's first 8 bytes hold the address of the
# next free block (0 = end of list). Since the block is at least 16 bytes wide
# and is not in use while on the free list, we reuse its own storage as the link.
# We link the BLOCK (header position), so popping just rewrites the header to the
# class id again.
#
# Large allocations (> largest class) are NOT pooled: they fall back to a direct
# alloc[UInt8](8 + nbytes) with header class id = -1, and free() detects -1 and
# frees the backing allocation immediately. This path costs a real malloc/free
# per call — be honest, it is only a correctness fallback, not a fast path.
#
# Hot paths (allocate / free) are @always_inline. Not thread-safe.

from std.memory import alloc, UnsafePointer
from mem.aligned import BytePtr, align_up_pow2

# Size classes (bytes of USABLE user space per block, header excluded).
# A blend: dense at the small end where churn lives, doubling at the top.
comptime NUM_CLASSES = 15
comptime HEADER = 8          # bytes reserved before the user pointer
comptime CHUNK_SIZE = 1 << 20  # 1 MiB backing chunks
comptime LARGE_CLASS = -1    # header sentinel for the unpooled large fallback


@always_inline("nodebug")
def _class_size(i: Int) -> Int:
    """Usable bytes for size-class `i`. Kept as a function so it inlines into
    both allocate() and the carve loop without a runtime table lookup."""
    # [16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096]
    if i == 0:
        return 16
    if i == 1:
        return 32
    if i == 2:
        return 48
    if i == 3:
        return 64
    if i == 4:
        return 96
    if i == 5:
        return 128
    if i == 6:
        return 192
    if i == 7:
        return 256
    if i == 8:
        return 384
    if i == 9:
        return 512
    if i == 10:
        return 768
    if i == 11:
        return 1024
    if i == 12:
        return 1536
    if i == 13:
        return 2048
    return 4096  # i == 14


@always_inline("nodebug")
def _size_to_class(nbytes: Int) -> Int:
    """Smallest class index whose usable size >= nbytes, or LARGE_CLASS (-1) when
    it exceeds the largest class. Linear scan over 15 constants — branch-light and
    fully inlined; faster than a table indirection at this count."""
    if nbytes <= 16:
        return 0
    if nbytes <= 32:
        return 1
    if nbytes <= 48:
        return 2
    if nbytes <= 64:
        return 3
    if nbytes <= 96:
        return 4
    if nbytes <= 128:
        return 5
    if nbytes <= 192:
        return 6
    if nbytes <= 256:
        return 7
    if nbytes <= 384:
        return 8
    if nbytes <= 512:
        return 9
    if nbytes <= 768:
        return 10
    if nbytes <= 1024:
        return 11
    if nbytes <= 1536:
        return 12
    if nbytes <= 2048:
        return 13
    if nbytes <= 4096:
        return 14
    return LARGE_CLASS


struct SlabAllocator(Movable):
    """Size-class slab allocator with intrusive per-class free lists.

    allocate(nbytes) -> usable BytePtr (8-byte header behind it holds the class).
    free(user_ptr) returns the block to its class list, or frees a large
    fallback allocation directly. Backing chunks are freed on drop. Not
    thread-safe."""

    # Free-list heads, one per size class. Stored as integer addresses of the
    # first free BLOCK (header position); 0 means the list is empty.
    var _heads: List[Int]
    var _chunks: List[BytePtr]   # every backing chunk, freed on drop
    var _large: List[BytePtr]    # outstanding large fallback allocations
    var _cur: BytePtr            # current chunk we are carving from
    var _cur_off: Int            # bytes consumed in the current chunk
    var _cur_cap: Int            # capacity of the current chunk (0 = none yet)
    var _reserved: Int           # total bytes alloc'd as chunks
    var _live: Int               # live (handed-out, not freed) allocations

    @staticmethod
    def create() raises -> SlabAllocator:
        var heads = List[Int]()
        for _ in range(NUM_CLASSES):
            heads.append(0)
        return SlabAllocator(heads^)

    def __init__(out self, var heads: List[Int]):
        self._heads = heads^
        self._chunks = List[BytePtr]()
        self._large = List[BytePtr]()
        self._cur = BytePtr(unsafe_from_address=Int(0))
        self._cur_off = 0
        self._cur_cap = 0
        self._reserved = 0
        self._live = 0

    @always_inline
    def _new_chunk(mut self, need: Int) raises:
        """Allocate a fresh backing chunk big enough for at least `need` bytes
        (block stride incl. header). Chunks are CHUNK_SIZE unless a single block
        is somehow larger (never happens for pooled classes: max stride is
        4096+8)."""
        var cap = CHUNK_SIZE
        if need > cap:
            cap = need
        var c = alloc[UInt8](cap)
        self._chunks.append(c)
        self._cur = c
        self._cur_off = 0
        self._cur_cap = cap
        self._reserved += cap

    @always_inline
    def _carve(mut self, cls: Int) raises -> Int:
        """Carve ONE fresh block of class `cls` from the current chunk (allocating
        a new chunk if exhausted) and return its block address. Caller writes the
        header and returns block+HEADER."""
        var stride = HEADER + _class_size(cls)
        if self._cur_cap == 0 or self._cur_off + stride > self._cur_cap:
            self._new_chunk(stride)
        var block = Int(self._cur) + self._cur_off
        self._cur_off += stride
        return block

    @always_inline
    def allocate(mut self, nbytes: Int) raises -> BytePtr:
        """Allocate at least `nbytes` usable bytes. Returns a writable pointer.
        Pooled for nbytes <= 4096; larger requests use a direct malloc fallback."""
        if nbytes < 0:
            raise Error("SlabAllocator: negative size")
        var cls = _size_to_class(nbytes)

        if cls == LARGE_CLASS:
            # Unpooled fallback: real allocation, header marks it large.
            var raw = alloc[UInt8](HEADER + (nbytes if nbytes > 0 else 1))
            raw.bitcast[Int]()[0] = LARGE_CLASS
            self._large.append(raw)
            self._live += 1
            return raw + HEADER

        var head = self._heads[cls]
        var block: Int
        if head != 0:
            # Pop: the block's first 8 bytes hold the next-free link.
            block = head
            var bp = BytePtr(unsafe_from_address=block)
            self._heads[cls] = bp.bitcast[Int]()[0]
        else:
            block = self._carve(cls)

        var bp2 = BytePtr(unsafe_from_address=block)
        bp2.bitcast[Int]()[0] = cls   # write header
        self._live += 1
        return bp2 + HEADER

    @always_inline
    def free(mut self, user_ptr: BytePtr) raises:
        """Return an allocation obtained from allocate(). Reads the header to find
        the class, then pushes the block onto that class's free list (O(1)). The
        large fallback frees its backing block directly."""
        var block_addr = Int(user_ptr) - HEADER
        var bp = BytePtr(unsafe_from_address=block_addr)
        var cls = bp.bitcast[Int]()[0]

        if cls == LARGE_CLASS:
            # Free the real allocation and drop it from the outstanding list.
            for i in range(len(self._large)):
                if Int(self._large[i]) == block_addr:
                    self._large[i].free()
                    self._large[i] = self._large[len(self._large) - 1]
                    _ = self._large.pop()
                    break
            self._live -= 1
            return

        if cls < 0 or cls >= NUM_CLASSES:
            raise Error("SlabAllocator: corrupt header / bad free")

        # Push onto the class free list: store old head in the block, then update.
        bp.bitcast[Int]()[0] = self._heads[cls]
        self._heads[cls] = block_addr
        self._live -= 1

    # ---- stats ----

    def bytes_reserved(self) -> Int:
        """Total bytes claimed from the OS as backing chunks (excludes large
        fallback allocations, which are tracked separately)."""
        return self._reserved

    def live_allocs(self) -> Int:
        """Allocations handed out and not yet freed (pooled + large)."""
        return self._live

    def class_count(self) -> Int:
        return NUM_CLASSES

    def largest_class(self) -> Int:
        """Usable bytes of the biggest pooled class. Requests above this fall back
        to the unpooled large path."""
        return _class_size(NUM_CLASSES - 1)

    def __del__(deinit self):
        # Free every backing chunk we ever allocated...
        for i in range(len(self._chunks)):
            self._chunks[i].free()
        # ...and any large fallback allocations still outstanding.
        for i in range(len(self._large)):
            self._large[i].free()
