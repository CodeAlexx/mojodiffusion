# mem/stats.mojo — allocation statistics + a self-accounting tracking allocator.
#
# MemStats is a pure counter record (no heap field) so it is trivially Copyable /
# Movable: snapshot it, return it by value, diff two snapshots. record_alloc /
# record_free keep live/peak/total/count tallies; leaked_* surfaces anything
# allocated but never freed.
#
# TrackingAllocator wraps raw `alloc` and prepends an 8-byte header storing the
# user byte count, so deallocate() can record the exact size on free without the
# caller remembering it. The user pointer it returns points PAST the header, so
# header bytes can never collide with user data.
#
# STATS_ENABLED demonstrates the comptime zero-cost pattern: with it False the
# `comptime if` compiles the recording call away entirely (no runtime branch).

from std.memory import alloc, UnsafePointer
from mem.aligned import BytePtr

# Flip to False to compile every maybe_record() call site to nothing.
comptime STATS_ENABLED = True

# Bytes reserved in front of each TrackingAllocator block to store the size.
comptime HEADER = 8


@fieldwise_init
struct MemStats(Copyable, Movable, Writable):
    """Pure-counter allocation statistics. No heap field → trivially copyable.

    record_alloc/record_free maintain running live/peak/total tallies plus
    counts. peak_bytes / peak_count are high-water marks that never decrease on
    free. leaked_bytes()/leaked_count() are just the current live values — the
    residue after all expected frees. Snapshot it by plain copy."""

    var live_bytes: Int       # bytes currently outstanding
    var peak_bytes: Int       # high-water mark of live_bytes
    var total_allocated: Int  # cumulative bytes ever allocated
    var total_freed: Int      # cumulative bytes ever freed
    var alloc_count: Int      # number of allocate calls
    var free_count: Int       # number of free calls
    var live_count: Int       # outstanding (allocated - freed) allocations
    var peak_count: Int       # high-water mark of live_count

    def __init__(out self):
        """Start at all-zero counters."""
        self.live_bytes = 0
        self.peak_bytes = 0
        self.total_allocated = 0
        self.total_freed = 0
        self.alloc_count = 0
        self.free_count = 0
        self.live_count = 0
        self.peak_count = 0

    @always_inline
    def record_alloc(mut self, n: Int):
        """Account for an allocation of `n` bytes, bumping peaks if exceeded."""
        self.live_bytes += n
        self.total_allocated += n
        self.alloc_count += 1
        self.live_count += 1
        if self.live_bytes > self.peak_bytes:
            self.peak_bytes = self.live_bytes
        if self.live_count > self.peak_count:
            self.peak_count = self.live_count

    @always_inline
    def record_free(mut self, n: Int):
        """Account for freeing `n` bytes. Peaks are untouched (high-water)."""
        self.live_bytes -= n
        self.total_freed += n
        self.free_count += 1
        self.live_count -= 1

    @always_inline
    def leaked_bytes(self) -> Int:
        """Bytes still outstanding (== live_bytes)."""
        return self.live_bytes

    @always_inline
    def leaked_count(self) -> Int:
        """Allocations still outstanding (== live_count)."""
        return self.live_count

    def reset(mut self):
        """Zero every counter back to a fresh state."""
        self.live_bytes = 0
        self.peak_bytes = 0
        self.total_allocated = 0
        self.total_freed = 0
        self.alloc_count = 0
        self.free_count = 0
        self.live_count = 0
        self.peak_count = 0

    def write_to(self, mut writer: Some[Writer]):
        """Human-readable one-block summary (used by print/String)."""
        writer.write("MemStats(live=", self.live_bytes, "B/", self.live_count)
        writer.write(" peak=", self.peak_bytes, "B/", self.peak_count)
        writer.write(" total_alloc=", self.total_allocated, "B")
        writer.write(" total_freed=", self.total_freed, "B")
        writer.write(" leaked=", self.leaked_bytes(), "B/", self.leaked_count(), ")")

    def summary(self) -> String:
        """Return the summary as a String."""
        return String.write(self)

    def print_summary(self):
        """Print live / peak / total / leaked to stdout."""
        print(self.summary())


@always_inline
def maybe_record_alloc(mut s: MemStats, n: Int):
    """Comptime-gated recording: compiles to nothing when STATS_ENABLED=False."""
    comptime if STATS_ENABLED:
        s.record_alloc(n)


@always_inline
def maybe_record_free(mut s: MemStats, n: Int):
    """Comptime-gated recording: compiles to nothing when STATS_ENABLED=False."""
    comptime if STATS_ENABLED:
        s.record_free(n)


struct TrackingAllocator(Movable):
    """A thin `alloc` wrapper that self-accounts into an embedded MemStats.

    Each block is over-allocated by HEADER (8) bytes; the byte count is stored in
    that header and the pointer handed back points PAST it. deallocate() reads the
    header to record the exact size, so callers never track sizes themselves.
    Not thread-safe. Free everything you allocate or leaks() will report it."""

    var _stats: MemStats

    def __init__(out self):
        """Create an allocator with zeroed stats."""
        self._stats = MemStats()

    def allocate(mut self, nbytes: Int) raises -> BytePtr:
        """Allocate `nbytes` of usable space; returns a pointer past the header."""
        if nbytes <= 0:
            raise Error("TrackingAllocator: nbytes must be > 0")
        var raw = alloc[UInt8](HEADER + nbytes)
        raw.bitcast[Int]()[0] = nbytes          # stash size in the header
        maybe_record_alloc(self._stats, nbytes)
        return raw + HEADER                      # hand out the user region

    def deallocate(mut self, user_ptr: BytePtr) raises:
        """Free a pointer previously returned by allocate(), recording its size."""
        var raw = user_ptr - HEADER
        var n = raw.bitcast[Int]()[0]            # recover the stashed size
        maybe_record_free(self._stats, n)
        raw.free()

    @always_inline
    def stats(self) -> MemStats:
        """Return a copy of the current statistics snapshot."""
        return self._stats.copy()

    @always_inline
    def live_count(self) -> Int:
        """Outstanding (un-freed) allocations."""
        return self._stats.live_count

    @always_inline
    def live_bytes(self) -> Int:
        """Bytes currently outstanding."""
        return self._stats.live_bytes

    @always_inline
    def leaks(self) -> Bool:
        """True if any allocation has not been freed."""
        return self._stats.live_count != 0
