# mem/arena_growable.mojo — a GROWABLE (chunked) bump / region allocator.
#
# Like Arena, but with no fixed capacity. When the current chunk runs out it
# allocates a fresh chunk and keeps bumping. All chunks are owned and freed on
# drop. reset() rewinds to the very first chunk WITHOUT freeing any chunk, so
# the capacity grown in earlier rounds is retained — steady-state reuse hits
# zero new heap allocations. This is the win for "allocate a big burst, throw it
# all away, repeat" loops (per-frame, per-request, per-token-step) where the
# burst size isn't known up front.
#
# Over-large requests (bigger than the default chunk) get their own dedicated
# chunk so a single huge alloc never wastes a default-sized chunk and is always
# satisfiable.
#
# The hot path (alloc_bytes) is @always_inline: an align + bounds check + pointer
# bump for the common in-chunk case, with the grow path out of line. Not
# thread-safe.

from std.memory import alloc, UnsafePointer
from std.sys.info import size_of, align_of

from mem.aligned import align_up, BytePtr


struct GrowableArena(Movable):
    """A chunked bump allocator that grows on demand without a fixed cap.

    alloc_bytes / alloc_array bump an offset within the current chunk; when a
    request doesn't fit, a new chunk is appended and bumping continues. reset()
    rewinds to the first chunk in O(chunks) but keeps every chunk allocated, so
    repeated burst-then-reset loops reach a steady state with no new allocs.
    All chunks are freed once on drop. Not thread-safe."""

    var _chunks: List[BytePtr]      # bases of every owned chunk
    var _chunk_caps: List[Int]      # parallel: byte capacity of each chunk
    var _cur: Int                   # index of the chunk we're bumping in
    var _offset: Int                # bump offset within the current chunk
    var _default_chunk: Int         # default size for newly grown chunks
    var _total_reserved: Int        # sum of all chunk capacities ever allocated
    var _peak_used: Int             # high-water mark of live bytes
    var _alloc_count: Int           # allocations since construction / last reset

    @staticmethod
    def with_chunk_size(default_chunk: Int) raises -> GrowableArena:
        """Create a growable arena whose default chunk size is `default_chunk`.

        Allocates the first chunk immediately."""
        if default_chunk <= 0:
            raise Error("GrowableArena: default_chunk must be > 0")
        var base = alloc[UInt8](default_chunk)
        var chunks = List[BytePtr]()
        chunks.append(base)
        var caps = List[Int]()
        caps.append(default_chunk)
        return GrowableArena(chunks^, caps^, default_chunk)

    def __init__(
        out self,
        var chunks: List[BytePtr],
        var chunk_caps: List[Int],
        default_chunk: Int,
    ):
        self._chunks = chunks^
        self._chunk_caps = chunk_caps^
        self._cur = 0
        self._offset = 0
        self._default_chunk = default_chunk
        # Account for the chunks handed in at construction.
        var total = 0
        for i in range(len(self._chunk_caps)):
            total += self._chunk_caps[i]
        self._total_reserved = total
        self._peak_used = 0
        self._alloc_count = 0

    def _grow(mut self, needed: Int) raises:
        """Advance to a chunk that can hold `needed` bytes, allocating if needed.

        After a reset() earlier-grown chunks are kept; this first tries to reuse
        the already-allocated next chunk (the steady-state win — zero new allocs
        once the high-water mark is reached). Only when no existing next chunk is
        big enough do we allocate a fresh one. Over-large requests (> default
        chunk) get a dedicated right-sized chunk so they are always satisfiable
        and never strand a default chunk."""
        var next_idx = self._cur + 1
        if next_idx < len(self._chunks) and self._chunk_caps[next_idx] >= needed:
            # Reuse the existing next chunk — no allocation.
            self._cur = next_idx
            self._offset = 0
            return
        var new_cap = self._default_chunk
        if needed > new_cap:
            new_cap = needed
        var base = alloc[UInt8](new_cap)
        self._chunks.append(base)
        self._chunk_caps.append(new_cap)
        self._total_reserved += new_cap
        self._cur = len(self._chunks) - 1
        self._offset = 0

    @always_inline
    def _live_used(self) -> Int:
        """Bytes currently handed out: fully-used prior chunks + current offset."""
        var used = 0
        for i in range(self._cur):
            used += self._chunk_caps[i]
        return used + self._offset

    @always_inline
    def alloc_bytes(
        mut self, nbytes: Int, alignment: Int = 16
    ) raises -> BytePtr:
        """Bump-allocate `nbytes` aligned to `alignment`, growing if needed."""
        var start = align_up(self._offset, alignment)
        if start + nbytes > self._chunk_caps[self._cur]:
            # Doesn't fit in the current chunk — grow.
            # Reserve room for worst-case alignment so the new chunk is enough.
            self._grow(nbytes + alignment)
            start = align_up(self._offset, alignment)
        var p = self._chunks[self._cur] + start
        self._offset = start + nbytes
        var live = self._live_used()
        if live > self._peak_used:
            self._peak_used = live
        self._alloc_count += 1
        return p

    @always_inline
    def alloc_array[
        T: AnyType
    ](mut self, count: Int) raises -> UnsafePointer[T, MutExternalOrigin]:
        """Bump-allocate `count` elements of T, correctly sized and aligned."""
        var nbytes = count * size_of[T]()
        var p = self.alloc_bytes(nbytes, align_of[T]())
        return p.bitcast[T]()

    def reset(mut self):
        """Rewind to the first chunk in O(chunks). KEEPS every chunk allocated.

        Capacity grown in earlier rounds is retained, so a burst-then-reset
        loop reaches a steady state with zero new heap allocations. Existing
        pointers handed out before reset become invalid. Use shrink_to_first()
        if you want the extra memory back."""
        self._cur = 0
        self._offset = 0
        self._alloc_count = 0

    def shrink_to_first(mut self) raises:
        """Free every chunk except the first, giving memory back to the heap.

        Resets the bump state too. Use this when a burst was unusually large and
        you don't want to hold that capacity for the rest of the program."""
        # Free chunks [1 .. end), keep chunk 0.
        var n = len(self._chunks)
        for i in range(1, n):
            self._chunks[i].free()
            self._total_reserved -= self._chunk_caps[i]
        # Truncate the parallel lists back down to one entry.
        while len(self._chunks) > 1:
            _ = self._chunks.pop()
            _ = self._chunk_caps.pop()
        self._cur = 0
        self._offset = 0
        self._alloc_count = 0

    @always_inline
    def total_reserved(self) -> Int:
        """Total bytes reserved across all owned chunks."""
        return self._total_reserved

    @always_inline
    def chunk_count(self) -> Int:
        """Number of chunks currently owned."""
        return len(self._chunks)

    @always_inline
    def peak_used(self) -> Int:
        """High-water mark of live bytes handed out since construction."""
        return self._peak_used

    @always_inline
    def alloc_count(self) -> Int:
        """Allocations since construction or last reset."""
        return self._alloc_count

    def __del__(deinit self):
        for i in range(len(self._chunks)):
            self._chunks[i].free()
