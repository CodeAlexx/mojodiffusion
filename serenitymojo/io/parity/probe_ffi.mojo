# probe_ffi.mojo — verify FFI sentinel round-trips and 64-bit offset handling.
from std.ffi import external_call
from serenitymojo.io.ffi import (
    BytePtr,
    map_failed,
    sys_sysconf,
    _SC_PAGESIZE,
)


def main():
    # 1. MAP_FAILED sentinel round-trip.
    var mf = map_failed()
    print("Int(map_failed()) =", Int(mf), " (expect -1)")

    # 2. sysconf page size.
    print("page_size =", sys_sysconf(_SC_PAGESIZE), " (expect 4096)")

    # 3. Big-offset pointer arithmetic: does BytePtr + (>2^32) stay 64-bit?
    var base = BytePtr(unsafe_from_address=Int(0))
    var big: Int = 9973681280  # 9.3 GB, > 2^32
    var p = base + big
    print("base + 9973681280 =", Int(p), " (expect 9973681280)")

    # 4. page-align math on a >2^32 offset (mirrors mmap.rs:76-78)
    var page = sys_sysconf(_SC_PAGESIZE)
    var off: Int = 9973681280
    var page_off = off % page
    var aligned = off - page_off
    print("aligned_offset(9973681280) =", aligned, " page_off=", page_off)

    # 5. madvise align-down mask math (mmap.rs:125): abs & ~(page-1)
    var abs_addr: Int = 9973681280 + 12345
    var aligned_addr = abs_addr & ~(page - 1)
    print("align-down(", abs_addr, ") =", aligned_addr)
