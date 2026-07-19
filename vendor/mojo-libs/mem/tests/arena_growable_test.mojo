# mem/tests/arena_growable_test.mojo
from std.memory import alloc, UnsafePointer
from std.time import perf_counter_ns
from mem.arena_growable import GrowableArena
from mem.aligned import BytePtr
from mem.bench import report


def check(mut passed: Int, mut failed: Int, cond: Bool, name: String):
    if cond:
        passed += 1
    else:
        failed += 1
        print("  FAIL:", name)


def main() raises:
    var passed = 0
    var failed = 0

    # ---- forced growth: 1024 chunk, 10x100B -> spills into a 2nd chunk ----
    var a = GrowableArena.with_chunk_size(1024)
    check(passed, failed, a.chunk_count() == 1, "starts with 1 chunk")
    check(passed, failed, a.total_reserved() == 1024, "reserved 1024 at start")

    var ptrs = List[BytePtr]()
    for i in range(10):
        var p = a.alloc_bytes(100)  # default align 16 -> 112B stride
        p[0] = UInt8(i)             # write first byte (distinct marker)
        ptrs.append(p)
    # 10 * align_up(100,16)=112 = 1120 > 1024 -> must have grown to 2 chunks.
    check(passed, failed, a.chunk_count() == 2, "10x100B forced 2nd chunk")
    check(passed, failed, a.alloc_count() == 10, "10 allocs counted")

    # each pointer writable & distinct (no two share an address)
    var all_distinct = True
    for i in range(len(ptrs)):
        for j in range(i + 1, len(ptrs)):
            if Int(ptrs[i]) == Int(ptrs[j]):
                all_distinct = False
    check(passed, failed, all_distinct, "all 10 pointers distinct")
    var markers_ok = True
    for i in range(len(ptrs)):
        if ptrs[i][0] != UInt8(i):
            markers_ok = False
    check(passed, failed, markers_ok, "all 10 markers readable")

    # ---- oversized alloc (> default chunk) gets its own chunk, fully usable ----
    var chunks_before = a.chunk_count()
    var big = a.alloc_bytes(5000)
    check(passed, failed, a.chunk_count() == chunks_before + 1, "5000B -> own new chunk")
    check(passed, failed, a.total_reserved() >= chunks_before * 1024 + 5000, "big chunk sized >= 5000")
    big[0] = UInt8(0xAB)
    big[4999] = UInt8(0xCD)
    check(passed, failed, big[0] == UInt8(0xAB) and big[4999] == UInt8(0xCD), "big chunk first+last byte rw")

    # ---- alignment honored ----
    var pa = a.alloc_bytes(1, 64)
    check(passed, failed, (Int(pa) % 64) == 0, "alloc_bytes(1,64) 64-aligned")

    # ---- reset keeps capacity; reuses chunk 0 base ----
    var chunk0_base = Int(a._chunks[0])
    var reserved_before_reset = a.total_reserved()
    var chunks_at_reset = a.chunk_count()
    a.reset()
    check(passed, failed, a.chunk_count() == chunks_at_reset, "reset keeps chunk_count (capacity retained)")
    check(passed, failed, a.total_reserved() == reserved_before_reset, "reset keeps total_reserved")
    check(passed, failed, a.alloc_count() == 0, "reset clears alloc_count")
    var first_after_reset = a.alloc_bytes(8)
    check(passed, failed, Int(first_after_reset) == chunk0_base, "first alloc after reset reuses chunk 0 base")
    a.reset()

    # ---- typed alloc_array ----
    var ints = a.alloc_array[Int32](100)
    check(passed, failed, (Int(ints) % 4) == 0, "Int32 array aligned")
    for i in range(100):
        ints[i] = Int32(i * i)
    var sum = 0
    for i in range(100):
        sum += Int(ints[i])
    check(passed, failed, sum == 328350, "Int32 array values (sum 0..99 squared)")

    # ---- 100_000 small objects across many chunks, per-index pattern ----
    var ga = GrowableArena.with_chunk_size(4096)
    var N = 100_000
    var stored = List[BytePtr]()
    for i in range(N):
        var q = ga.alloc_bytes(7)        # 7B objects, weird size to stress packing
        var pat = UInt8((i * 31 + 7) & 0xFF)
        q[0] = pat
        q[6] = UInt8((i * 17 + 3) & 0xFF)
        stored.append(q)
    check(passed, failed, ga.chunk_count() > 1, "100k objects spanned many chunks")
    check(passed, failed, ga.alloc_count() == N, "100k allocs counted")
    var corrupt = 0
    for i in range(N):
        var pat = UInt8((i * 31 + 7) & 0xFF)
        var pat2 = UInt8((i * 17 + 3) & 0xFF)
        if stored[i][0] != pat or stored[i][6] != pat2:
            corrupt += 1
    check(passed, failed, corrupt == 0, "no corruption across chunk boundaries (100k)")
    print("  (100k spanned", ga.chunk_count(), "chunks; peak_used =", ga.peak_used(), "bytes)")

    print("passed:", passed, " failed:", failed)
    if failed == 0:
        print("ALL GROWABLE ARENA TESTS PASSED")

    # ---- BENCHMARK: steady-state bump+reset vs raw malloc/free ----
    print("")
    print("--- benchmark ---")
    var sink = 0
    var R = 200
    var K = 50_000

    var barena = GrowableArena.with_chunk_size(1 << 20)  # 1 MiB chunks
    var t0 = Int(perf_counter_ns())
    for _r in range(R):
        for _i in range(K):
            var p = barena.alloc_bytes(32)
            sink ^= Int(p)
        barena.reset()
    var ns = Int(perf_counter_ns()) - t0
    report("growable arena bump+reset", R * K, ns, sink)
    print("  (steady-state chunks =", barena.chunk_count(), ")")

    var t1 = Int(perf_counter_ns())
    for _r in range(R):
        for _i in range(K):
            var rp = alloc[UInt8](32)
            sink ^= Int(rp)
            rp.free()
    var ns2 = Int(perf_counter_ns()) - t1
    report("raw alloc+free 32B", R * K, ns2, sink)
