# mem/tests/slab_test.mojo
from std.memory import alloc, UnsafePointer
from mem.aligned import BytePtr
from mem.slab import SlabAllocator
from mem.bench import report, report_speedup
from std.time import perf_counter_ns

# NOTE: we do NOT import mem.bench.Timer. Under this toolchain perf_counter_ns()
# returns UInt, and bench.Timer.__init__ takes Int, so the struct fails to
# compile (bench.mojo is owned elsewhere — not editable here). We time inline
# with explicit Int() conversion instead. report()/report_speedup() are fine.


def check(mut passed: Int, mut failed: Int, cond: Bool, name: String):
    if cond:
        passed += 1
    else:
        failed += 1
        print("  FAIL:", name)


def main() raises:
    var passed = 0
    var failed = 0

    var s = SlabAllocator.create()
    check(passed, failed, s.class_count() == 15, "class_count == 15")
    check(passed, failed, s.largest_class() == 4096, "largest_class == 4096")
    check(passed, failed, s.live_allocs() == 0, "starts with 0 live")

    # ---- sizes: each returns a usable, writable region of >= requested size ----
    # request sizes incl. a large fallback (5000 > 4096).
    var sizes = [1, 16, 17, 64, 100, 1000, 5000]
    var ptrs = List[BytePtr]()
    for k in range(len(sizes)):
        var n = sizes[k]
        var p = s.allocate(n)
        # write last byte first, then first byte, so a 1-byte region (where
        # first == last) still reads back the first-byte value distinctly.
        p[n - 1] = UInt8(0x50 + k)
        check(passed, failed, p[n - 1] == UInt8(0x50 + k), "write/read last byte sz=" + String(n))
        p[0] = UInt8(0xA0 + k)
        check(passed, failed, p[0] == UInt8(0xA0 + k), "write/read first byte sz=" + String(n))
        ptrs.append(p)

    check(passed, failed, s.live_allocs() == len(sizes), "live == 7 after allocs")

    # ---- pointers distinct / non-overlapping ----
    var distinct = True
    for a in range(len(ptrs)):
        for b in range(a + 1, len(ptrs)):
            # regions must not overlap given each has at least sizes[] bytes.
            var aa = Int(ptrs[a])
            var bb = Int(ptrs[b])
            var asz = sizes[a]
            var bsz = sizes[b]
            if aa == bb:
                distinct = False
            # overlap check (treat each as its requested length)
            if aa < bb:
                if aa + asz > bb:
                    distinct = False
            else:
                if bb + bsz > aa:
                    distinct = False
    check(passed, failed, distinct, "all allocations distinct & non-overlapping")

    # ---- free then allocate same size reuses a freed block (pooled class) ----
    # use class-64 (request 64). free it, re-alloc 64, expect the same address.
    var q1 = s.allocate(64)
    var q1_addr = Int(q1)
    s.free(q1)
    var q2 = s.allocate(64)
    check(passed, failed, Int(q2) == q1_addr, "free+realloc 64 reuses freed block")
    s.free(q2)

    # also verify reuse for a small class (16) with two outstanding then LIFO pop.
    var r1 = s.allocate(16)
    var r2 = s.allocate(16)
    var r1a = Int(r1)
    var r2a = Int(r2)
    s.free(r1)        # push r1
    s.free(r2)        # push r2 (now head)
    var r3 = s.allocate(16)   # pops r2 (LIFO)
    var r4 = s.allocate(16)   # pops r1
    check(passed, failed, Int(r3) == r2a and Int(r4) == r1a, "LIFO reuse for class 16")
    s.free(r3)
    s.free(r4)

    # free everything from the first batch.
    for k in range(len(ptrs)):
        s.free(ptrs[k])
    check(passed, failed, s.live_allocs() == 0, "all freed -> 0 live")

    # ---- 10_000 mixed-size objects with checksum (no corruption) ----
    var churn_sizes = [8, 24, 40, 80, 130, 300, 700, 2000]
    var ok_churn = True
    for i in range(10_000):
        var n = churn_sizes[i % len(churn_sizes)]
        var p = s.allocate(n)
        # write a deterministic pattern across the whole region.
        var seed = UInt8((i * 31 + 7) & 0xFF)
        for j in range(n):
            p[j] = UInt8((Int(seed) + j) & 0xFF)
        # read it back and checksum-verify before freeing.
        var good = True
        for j in range(n):
            if p[j] != UInt8((Int(seed) + j) & 0xFF):
                good = False
                break
        if not good:
            ok_churn = False
        s.free(p)
    check(passed, failed, ok_churn, "10k mixed alloc/free no corruption")
    check(passed, failed, s.live_allocs() == 0, "10k churn balanced (0 live)")

    # ---- large (>4096) alloc/free via fallback ----
    var big = s.allocate(50_000)
    big[0] = UInt8(0x11)
    big[49_999] = UInt8(0x22)
    check(passed, failed, big[0] == UInt8(0x11) and big[49_999] == UInt8(0x22), "large 50k rw")
    check(passed, failed, s.live_allocs() == 1, "large counts as 1 live")
    s.free(big)
    check(passed, failed, s.live_allocs() == 0, "large freed -> 0 live")

    check(passed, failed, s.bytes_reserved() > 0, "bytes_reserved > 0 after use")

    print("passed:", passed, " failed:", failed)
    if failed == 0:
        print("ALL SLAB TESTS PASSED")

    # =====================================================================
    # BENCHMARK: mixed-size churn, slab vs raw alloc/free.
    # =====================================================================
    var bench_sizes = [16, 48, 96, 200, 512, 1024]
    var N = 2_000_000
    var sink = 0

    # --- slab ---
    var sb = SlabAllocator.create()
    var t1 = Int(perf_counter_ns())
    for i in range(N):
        var sz = bench_sizes[i % len(bench_sizes)]
        var p = sb.allocate(sz)
        p[0] = UInt8(i & 0xFF)
        sink ^= Int(p)
        sb.free(p)
    var ns1 = Int(perf_counter_ns()) - t1
    report("slab alloc+free mixed", N, ns1, sink)

    # --- raw ---
    var t2 = Int(perf_counter_ns())
    for i in range(N):
        var sz = bench_sizes[i % len(bench_sizes)]
        var p = alloc[UInt8](sz)
        p[0] = UInt8(i & 0xFF)
        sink ^= Int(p)
        p.free()
    var ns2 = Int(perf_counter_ns()) - t2
    report("raw alloc+free mixed", N, ns2, sink)

    report_speedup("slab vs raw", ns2, ns1)
