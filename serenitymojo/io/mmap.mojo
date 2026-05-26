# mmap.mojo — MmapRegion, a pure-Mojo port of serenity-safetensors
# src/mmap.rs MmapRegion (lines 30-154). Linux x86-64 only.
#
# Maps a region of an already-open file into UNCOMMITTED virtual memory
# (MAP_PRIVATE | MAP_NORESERVE) so the OS page cache manages residency. The
# struct stores both the page-aligned mmap base/len (for cleanup) and the
# (possibly unaligned) caller-visible ptr/len.
#
# Differences from the Rust ref that are intentional and documented:
#   * Rust takes &File and reads file.metadata().len() internally. Mojo has no
#     std File abstraction here, so new() takes (fd, offset, len, file_len) and
#     the SIGBUS file-size check uses the caller-supplied file_len (computed via
#     ffi.file_size in safetensors.mojo). Same check, same guarantee.
#   * Rust returns Result<_, MmapError>; Mojo raises Error (ZeroLength, EOF,
#     mmap-failed) with equivalent messages.

from std.memory import UnsafePointer
from .ffi import (
    BytePtr,
    sys_mmap,
    sys_munmap,
    sys_madvise,
    sys_sysconf,
    map_failed,
    PROT_READ,
    MAP_PRIVATE,
    MAP_NORESERVE,
    MADV_WILLNEED,
    MADV_DONTNEED,
    _SC_PAGESIZE,
)


struct MmapRegion(Movable):
    """A memory-mapped, uncommitted (MAP_NORESERVE) view into a file.

    Dropping the region unmaps it. Not Copyable: a region uniquely owns its
    mapping (mirrors Rust's non-Clone MmapRegion + Drop)."""

    var ptr: BytePtr  # requested region start (may be unaligned)
    var _len: Int  # length of the requested region
    var mmap_base: BytePtr  # page-aligned actual mmap base
    var mmap_len: Int  # page-aligned actual mmap length

    def __init__(
        out self, ptr: BytePtr, _len: Int, mmap_base: BytePtr, mmap_len: Int
    ):
        self.ptr = ptr
        self._len = _len
        self.mmap_base = mmap_base
        self.mmap_len = mmap_len

    @staticmethod
    def new(
        fd: Int, offset: Int, length: Int, file_len: Int
    ) raises -> MmapRegion:
        """Map [offset, offset+length) of `fd`. Mirrors mmap.rs:54-107.

        file_len is the caller-known file size; we verify offset+length <=
        file_len up front (SIGBUS prevention, mmap.rs:59-71)."""
        # mmap.rs:55-57 — zero-length is an error.
        if length == 0:
            raise Error("Cannot mmap zero-length region")

        # mmap.rs:63-71 — SIGBUS prevention: file must be large enough.
        if offset + length > file_len:
            raise Error(
                String("File is ")
                + String(file_len)
                + " bytes but mapping requires "
                + String(offset + length)
                + " bytes (offset="
                + String(offset)
                + ")"
            )

        # mmap.rs:74 — page size.
        var page_size = sys_sysconf(_SC_PAGESIZE)

        # mmap.rs:76-78 — page-alignment math.
        var page_offset = offset % page_size
        var aligned_offset = offset - page_offset
        var aligned_len = length + page_offset

        # mmap.rs:86-95 — mmap(NULL, aligned_len, PROT_READ,
        #   MAP_PRIVATE | MAP_NORESERVE, fd, aligned_offset).
        var base = sys_mmap(
            0,
            aligned_len,
            PROT_READ,
            MAP_PRIVATE | MAP_NORESERVE,
            Int32(fd),
            aligned_offset,
        )

        # mmap.rs:97-99 — check MAP_FAILED.
        if Int(base) == Int(map_failed()):
            raise Error("mmap failed")

        # mmap.rs:101-106 — ptr = base + page_offset; keep base/len for cleanup.
        return MmapRegion(
            ptr=base + page_offset,
            _len=length,
            mmap_base=base,
            mmap_len=aligned_len,
        )

    def as_ptr(self) -> BytePtr:
        """Pointer to the requested region start. Mirrors mmap.rs:109-111."""
        return self.ptr

    def len(self) -> Int:
        """Length of the requested region. Mirrors mmap.rs:113-115."""
        return self._len

    def prefetch_range(self, region_offset: Int, region_len: Int):
        """Advise the OS to prefetch (MADV_WILLNEED). Mirrors mmap.rs:117-131:
        bail if out of range, then re-align the pointer down to a page boundary
        and extend the length by the alignment delta."""
        # mmap.rs:119-121 — out-of-range guard.
        if region_offset + region_len > self._len:
            return
        # mmap.rs:123 — page size.
        var page_size = sys_sysconf(_SC_PAGESIZE)
        # mmap.rs:124-126 — align the absolute pointer down to a page boundary.
        var abs_addr = Int(self.ptr) + region_offset
        var aligned_addr = abs_addr & ~(page_size - 1)
        var aligned_len = region_len + (abs_addr - aligned_addr)
        var aligned_ptr = BytePtr(unsafe_from_address=aligned_addr)
        # mmap.rs:128-130 — madvise WILLNEED.
        _ = sys_madvise(aligned_ptr, aligned_len, MADV_WILLNEED)

    def release_to_os(self):
        """Advise the OS that pages may be reclaimed (MADV_DONTNEED). Data is
        NOT lost — re-access re-reads from disk. Mirrors mmap.rs:135-144:
        madvise over the exact (mmap_base, mmap_len) range."""
        _ = sys_madvise(self.mmap_base, self.mmap_len, MADV_DONTNEED)

    def __del__(deinit self):
        """munmap the exact (mmap_base, mmap_len) range. Mirrors mmap.rs Drop
        (147-154)."""
        _ = sys_munmap(self.mmap_base, self.mmap_len)
