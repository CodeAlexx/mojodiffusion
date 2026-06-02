# safetensors.mojo — SafeTensors single-file reader. Pure-Mojo port of
# serenity-safetensors src/mmap.rs MmapFile (lines 156-252). Linux x86-64.
#
# open(path) sequence (mirrors mmap.rs:172-201):
#   1. open the file (O_RDONLY).
#   2. read first 8 bytes -> header_len = little-endian u64 (mmap.rs:175-178).
#   3. reject header_len > 100*1024*1024 (mmap.rs:180-182).
#   4. read header_len bytes of header JSON (mmap.rs:185-188).
#   5. data_offset = 8 + header_len (mmap.rs:190).
#   6. file_len = file size; data_len = file_len - data_offset
#      (mmap.rs:193-194). Reject data_len == 0 (mmap.rs:196-198).
#   7. mmap the DATA segment via MmapRegion.new (mmap.rs:201).
#   8. build name -> TensorRef index, skipping "__metadata__" (mmap.rs:204-235).
#      offset = data_offsets[0]; size = data_offsets[1] - data_offsets[0].
#
# The DATA segment is mmap'd (never read into RAM). The 8-byte length + header
# bytes ARE read via pread (small, bounded by 100MB cap) — this is the only
# eager I/O, matching the Rust reference which read_exact's the header.

from std.memory import alloc, UnsafePointer, Span
from .dtype import STDtype
from .mmap import MmapRegion
from .json_header import parse_header, HeaderEntry
from .ffi import (
    BytePtr,
    sys_open,
    sys_close,
    sys_pread,
    file_size,
    O_RDONLY,
)


comptime MAX_HEADER_LEN = 100 * 1024 * 1024  # mmap.rs:180


def _pread_exact(fd: Int, buf: BytePtr, count: Int, offset: Int) raises:
    """Read exactly `count` bytes into `buf` starting at `offset`, looping over
    short reads. Mirrors Rust's File::read_exact (mmap.rs:177/186): pread(2) may
    return fewer bytes than requested (EINTR, large header, NFS), so we loop
    until the buffer is full. A return of 0 means EOF before count -> raise; a
    negative return is an I/O error -> raise."""
    var done = 0
    while done < count:
        var n = sys_pread(fd, buf + done, count - done, offset + done)
        if n < 0:
            raise Error("pread failed (I/O error)")
        if n == 0:
            raise Error("unexpected EOF before reading requested bytes")
        done += n


@fieldwise_init
struct TensorRef(Copyable, Movable):
    """A tensor's location within the mmap'd data segment. Mirrors mmap.rs
    TensorRef (156-162) but carries a typed STDtype instead of a String."""

    var offset: Int  # byte offset into the data segment
    var size: Int  # byte length (end - start)
    var dtype: STDtype
    var shape: List[Int]


struct SafeTensors(Movable):
    """An mmap'd safetensors file with a tensor index. Mirrors mmap.rs
    MmapFile (164-252). Not Copyable: uniquely owns its MmapRegion."""

    var region: MmapRegion
    var tensors: Dict[String, TensorRef]

    def __init__(out self, var region: MmapRegion, var tensors: Dict[String, TensorRef]):
        self.region = region^
        self.tensors = tensors^

    @staticmethod
    def open(path: String) raises -> SafeTensors:
        """Open and index a single-file safetensors. Mirrors mmap.rs:172-237."""
        # mmap.rs:173 — open the file.
        var fd = sys_open(path, O_RDONLY)
        if fd < 0:
            raise Error(String("failed to open: ") + path)

        # From here on, ensure the fd is closed on every exit path. Mojo has no
        # try/finally; we close fd explicitly before each raise and at the end.

        # mmap.rs:175-178 — read 8-byte header length (little-endian u64).
        # read_exact-style loop (mmap.rs:177): absorb short reads.
        var lenbuf = alloc[UInt8](8)
        try:
            _pread_exact(fd, lenbuf, 8, 0)
        except e:
            lenbuf.free()
            _ = sys_close(fd)
            raise Error(String("failed to read 8-byte header length: ") + String(e))
        var header_len = 0
        for i in range(8):
            header_len = header_len | (Int(lenbuf[i]) << (8 * i))
        lenbuf.free()

        # mmap.rs:180-182 — reject oversized header.
        if header_len > MAX_HEADER_LEN:
            _ = sys_close(fd)
            raise Error("Header too large (>100MB)")
        if header_len <= 0:
            _ = sys_close(fd)
            raise Error("Empty or invalid header length")

        # mmap.rs:185-188 — read header bytes at offset 8.
        # read_exact-style loop (mmap.rs:186): absorb short reads.
        var hbuf = alloc[UInt8](header_len)
        try:
            _pread_exact(fd, hbuf, header_len, 8)
        except e:
            hbuf.free()
            _ = sys_close(fd)
            raise Error(String("failed to read header bytes: ") + String(e))
        var hbytes = List[UInt8]()
        for i in range(header_len):
            hbytes.append(hbuf[i])
        hbuf.free()

        # mmap.rs:190 — data segment begins right after the header.
        var data_offset = 8 + header_len

        # mmap.rs:193-194 — data_len = file_len - data_offset.
        var file_len = file_size(fd)
        var data_len = file_len - data_offset
        # mmap.rs:196-198 — empty data segment is an error.
        if data_len <= 0:
            _ = sys_close(fd)
            raise Error("Empty data segment")

        # Parse header BEFORE mmap so a parse failure closes fd cleanly.
        var entries: List[HeaderEntry]
        try:
            entries = parse_header(hbytes^)
        except e:
            _ = sys_close(fd)
            raise e^

        # mmap.rs:201 — map the data segment (MAP_NORESERVE).
        var region: MmapRegion
        try:
            region = MmapRegion.new(fd, data_offset, data_len, file_len)
        except e:
            _ = sys_close(fd)
            raise e^

        # The mapping holds its own reference to the pages; Rust keeps _file
        # alive but the mapping survives fd close on Linux. Close now.
        _ = sys_close(fd)

        # mmap.rs:204-235 — build the tensor index.
        var tensors = Dict[String, TensorRef]()
        for ref e in entries:
            # __metadata__ is already skipped by parse_header (mmap.rs:207-209).
            var start = e.off_start
            var end = e.off_end
            var size = end - start
            if size < 0:
                size = 0  # mmap.rs uses saturating_sub
            var dt = STDtype.from_name(e.dtype)
            tensors[e.name] = TensorRef(
                offset=start,
                size=size,
                dtype=dt,
                shape=e.shape.copy(),
            )

        return SafeTensors(region^, tensors^)

    def tensor_bytes(
        self, name: String
    ) raises -> Span[UInt8, origin_of(self)]:
        """Origin-bound view of a tensor's data = region.as_ptr() + ref.offset,
        length = ref.size. This is the PUBLIC accessor.

        The returned `Span` carries `origin_of(self)`, so the compiler keeps
        this `SafeTensors` (and therefore its `MmapRegion`) alive for as long as
        the Span is in use. That ties the mmap'd bytes' lifetime to the handle
        and makes the use-after-munmap footgun a *compile error* rather than a
        SIGSEGV (see parity/probe_lifetime.mojo). Mirrors mmap.rs:240-245, but
        safer: Rust gated the raw `*const u8` behind a `&self` borrow; here the
        borrow is encoded in the Span's origin so it cannot outlive the handle.

        The view is immutable: the data segment is mmap'd PROT_READ."""
        if name not in self.tensors:
            raise Error(String("Tensor '") + name + "' not found")
        var t = self.tensors[name].copy()
        # region.as_ptr() returns BytePtr (mutable, MutExternalOrigin —
        # untracked). The data segment is mmap'd PROT_READ, so drop mutability
        # (as_immutable) then re-tie the origin to self so the Span's lifetime
        # tracks this handle.
        var base = (self.region.as_ptr() + t.offset).as_immutable(
        ).unsafe_origin_cast[origin_of(self)]()
        return Span[UInt8, origin_of(self)](ptr=base, length=t.size)

    def _tensor_ptr_unsafe(self, name: String) raises -> BytePtr:
        """UNSAFE: raw, lifetime-UNTRACKED pointer to a tensor's data =
        region.as_ptr() + ref.offset. The returned `BytePtr` has
        `MutExternalOrigin`, so the compiler will NOT keep this `SafeTensors`
        alive for its users — dereferencing after the handle drops is a
        use-after-munmap (SIGSEGV / silent corruption). Prefer `tensor_bytes`.
        Internal only; nothing outside this module should obtain this pointer.
        Mirrors mmap.rs:240-245 (the bare `*const u8`)."""
        if name not in self.tensors:
            raise Error(String("Tensor '") + name + "' not found")
        var t = self.tensors[name].copy()
        return self.region.as_ptr() + t.offset

    def tensor_info(self, name: String) raises -> TensorRef:
        """(offset, size, dtype, shape) for a tensor. Mirrors mmap.rs:277-282."""
        if name not in self.tensors:
            raise Error(String("Tensor '") + name + "' not found")
        return self.tensors[name].copy()

    def names(self) -> List[String]:
        """All tensor names. Mirrors mmap.rs:285-287 tensor_names."""
        var out = List[String]()
        for ref e in self.tensors.items():
            out.append(e.key)
        return out^

    def count(self) -> Int:
        """Number of tensors."""
        return len(self.tensors)

    def prefetch_tensor(self, name: String) raises:
        """Prefetch a tensor's pages (MADV_WILLNEED). Mirrors mmap.rs:247-251."""
        if name not in self.tensors:
            return
        var t = self.tensors[name].copy()
        self.region.prefetch_range(t.offset, t.size)

    def release_to_os(self):
        """Release all pages (MADV_DONTNEED). Mirrors mmap.rs:300-302."""
        self.region.release_to_os()

    def data_size(self) -> Int:
        """Total data segment size in bytes. Mirrors mmap.rs:305-307."""
        return self.region.len()
