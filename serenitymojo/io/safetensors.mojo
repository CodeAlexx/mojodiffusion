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

from std.memory import alloc, UnsafePointer, bitcast
from .dtype import STDtype
from .mmap import MmapRegion
from .json_header import parse_header, HeaderEntry
from .ffi import (
    BytePtr,
    sys_open,
    sys_close,
    sys_posix_fadvise,
    sys_pread,
    file_size,
    O_RDONLY,
    POSIX_FADV_DONTNEED,
)


comptime MAX_HEADER_LEN = 100 * 1024 * 1024  # mmap.rs:180


@always_inline
def read_f32_scalar_bytes[
    mut: Bool, //, origin: Origin[mut=mut]
](data: Span[UInt8, origin]) raises -> Float32:
    """Decode one little-endian F32 scalar from safetensors bytes."""
    if len(data) != 4:
        raise Error(
            String("expected one F32 scalar (4 bytes), got ") + String(len(data))
        )
    var b0 = UInt32(Int(data[0]))
    var b1 = UInt32(Int(data[1]))
    var b2 = UInt32(Int(data[2]))
    var b3 = UInt32(Int(data[3]))
    var bits = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    return bitcast[DType.float32, 1](SIMD[DType.uint32, 1](bits))


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
    var storage_names: List[String]
    var path: String

    def __init__(
        out self,
        var region: MmapRegion,
        var tensors: Dict[String, TensorRef],
        var storage_names: List[String],
        path: String,
    ):
        self.region = region^
        self.tensors = tensors^
        self.storage_names = storage_names^
        self.path = path

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
        var storage_names = List[String]()
        var storage_offsets = List[Int]()
        var header_is_storage_order = True
        var last_offset = -1
        for ref e in entries:
            # __metadata__ is already skipped by parse_header (mmap.rs:207-209).
            var start = e.off_start
            var end = e.off_end
            # 2026-07-07 hardening (HF-crate parity): a corrupt/truncated file
            # used to be silently indexed and SIGSEGV on first read; validate
            # per-tensor bounds here instead so all ~337 open() sites get a
            # named error. (upstream mmap.rs is equally permissive — this
            # exceeds the port on purpose, like the sharded index does.)
            if start < 0 or end < start or end > data_len:
                raise Error(
                    String("safetensors: tensor '") + e.name
                    + String("' data_offsets [") + String(start) + String(", ")
                    + String(end) + String(") out of bounds (data_len ")
                    + String(data_len) + String(") — corrupt or truncated file: ")
                    + path
                )
            var size = end - start
            var dt = STDtype.from_name(e.dtype)
            tensors[e.name] = TensorRef(
                offset=start,
                size=size,
                dtype=dt,
                shape=e.shape.copy(),
            )
            storage_names.append(e.name.copy())
            storage_offsets.append(start)
            if start < last_offset:
                header_is_storage_order = False
            last_offset = start

        # Safetensors writers normally emit header entries in data-offset
        # order. Preserve that order for sequential mmap traversal. For a
        # valid non-canonical file, sort the small name index by byte offset
        # once at open rather than page-faulting model data in hash-map order.
        if not header_is_storage_order:
            var sorted_names = List[String]()
            var sorted_offsets = List[Int]()
            for i in range(len(storage_names)):
                var pos = len(sorted_offsets)
                for j in range(len(sorted_offsets)):
                    if storage_offsets[i] < sorted_offsets[j]:
                        pos = j
                        break
                sorted_names.insert(pos, storage_names[i])
                sorted_offsets.insert(pos, storage_offsets[i])
            storage_names = sorted_names^

        return SafeTensors(region^, tensors^, storage_names^, path.copy())

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
        return Span(unsafe_ptr=base, length=t.size)

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

    def has_tensor(self, name: String) -> Bool:
        """Non-raising tensor existence check."""
        return name in self.tensors

    def names_storage_order(self) -> List[String]:
        """Tensor names in ascending on-disk data-offset order.

        Use this for bulk loading so mmap faults and copies traverse each file
        sequentially. Name lookup remains the same dictionary-backed path.
        """
        return self.storage_names.copy()

    def count(self) -> Int:
        """Number of tensors."""
        return len(self.tensors)

    def prefetch_tensor(self, name: String) raises:
        """Prefetch a tensor's pages (MADV_WILLNEED). Mirrors mmap.rs:247-251."""
        if name not in self.tensors:
            return
        var t = self.tensors[name].copy()
        self.region.prefetch_range(t.offset, t.size)

    def release_tensor(self, name: String) raises:
        """Release one tensor's clean mmap/page-cache range after its pinned
        staging copy is complete. The immutable mapping stays valid."""
        if name not in self.tensors:
            return
        var t = self.tensors[name].copy()
        self.region.release_range(t.offset, t.size)
        var fd = sys_open(self.path, O_RDONLY)
        if fd >= 0:
            _ = sys_posix_fadvise(
                fd,
                self.region.source_offset(t.offset),
                t.size,
                POSIX_FADV_DONTNEED,
            )
            _ = sys_close(fd)

    def release_to_os(self):
        """Release mappings and clean file cache after a completed copy.

        MADV_DONTNEED drops this mapping's resident PTEs. Reopening the immutable
        checkpoint for POSIX_FADV_DONTNEED also evicts its clean page-cache
        charge, preventing source pages from crowding executable/kernel pages
        out of a memory-capped inference worker."""
        self.region.release_to_os()
        var fd = sys_open(self.path, O_RDONLY)
        if fd >= 0:
            _ = sys_posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
            _ = sys_close(fd)

    def data_size(self) -> Int:
        """Total data segment size in bytes. Mirrors mmap.rs:305-307."""
        return self.region.len()
