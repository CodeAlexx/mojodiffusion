# mem/tests/pool_test.mojo
from std.memory import alloc, UnsafePointer
from std.time import perf_counter_ns
from mem.aligned import BytePtr
from mem.pool import Pool
from mem.bench import report, report_speedup


def check(mut passed: Int, mut failed: Int, cond: Bool, name: String):
    if cond:
        passed += 1
    else:
        failed += 1
        print("  FAIL:", name)


def main() raises:
    var passed = 0
    var failed = 0

    # ---- create: block rounds up, all free ----
    var pool = Pool.create(24, 8)  # default align=64
    check(passed, failed, pool.block_size() == 64, "block 24 rounds to 64")
    check(passed, failed, pool.capacity() == 8, "capacity 8")
    check(passed, failed, pool.available() == 8, "available()==8 fresh")
    check(passed, failed, pool.in_use() == 0, "in_use()==0 fresh")

    # ---- acquire all 8: distinct, in-range, 64-aligned ----
    var ptrs = List[BytePtr]()
    for _ in range(8):
        ptrs.append(pool.acquire())
    check(passed, failed, pool.available() == 0, "available()==0 after 8")
    check(passed, failed, pool.in_use() == 8, "in_use()==8 after 8")

    var all_aligned = True
    for i in range(8):
        if Int(ptrs[i]) % 64 != 0:
            all_aligned = False
    check(passed, failed, all_aligned, "all 8 ptrs 64-aligned")

    var all_distinct = True
    for i in range(8):
        for j in range(i + 1, 8):
            if Int(ptrs[i]) == Int(ptrs[j]):
                all_distinct = False
    check(passed, failed, all_distinct, "all 8 ptrs distinct")

    # all within [base, base + cap*block)
    var base0 = Int(ptrs[0])
    var lo = base0
    for i in range(8):
        if Int(ptrs[i]) < lo:
            lo = Int(ptrs[i])
    var in_range = True
    for i in range(8):
        var off = Int(ptrs[i]) - lo
        if off < 0 or off >= 8 * 64:
            in_range = False
    check(passed, failed, in_range, "all 8 ptrs in range")

    # ---- 9th acquire raises ----
    var raised9 = False
    try:
        _ = pool.acquire()
    except:
        raised9 = True
    check(passed, failed, raised9, "9th acquire raises (exhausted)")

    # ---- sentinel write/read: no aliasing ----
    for i in range(8):
        ptrs[i][0] = UInt8(100 + i)
    var no_alias = True
    for i in range(8):
        if ptrs[i][0] != UInt8(100 + i):
            no_alias = False
    check(passed, failed, no_alias, "sentinels read back (no aliasing)")

    # ---- release 3, available()==3 ----
    var released = List[Int]()
    released.append(Int(ptrs[2]))
    released.append(Int(ptrs[5]))
    released.append(Int(ptrs[7]))
    pool.release(ptrs[2])
    pool.release(ptrs[5])
    pool.release(ptrs[7])
    check(passed, failed, pool.available() == 3, "available()==3 after release 3")
    check(passed, failed, pool.in_use() == 5, "in_use()==5 after release 3")

    # ---- re-acquire 3: addresses are among the released set ----
    var reused_ok = True
    for _ in range(3):
        var q = pool.acquire()
        var found = False
        for k in range(len(released)):
            if released[k] == Int(q):
                found = True
        if not found:
            reused_ok = False
    check(passed, failed, reused_ok, "re-acquired 3 reuse freed blocks")
    check(passed, failed, pool.available() == 0, "available()==0 after re-acquire")

    # ---- foreign pointer release raises ----
    var foreign = ptrs[0] + 12345
    var raised_foreign = False
    try:
        pool.release(foreign)
    except:
        raised_foreign = True
    check(passed, failed, raised_foreign, "foreign pointer release raises")

    # ---- reset restores full capacity ----
    pool.reset()
    check(passed, failed, pool.available() == 8, "reset restores available()==8")
    check(passed, failed, pool.in_use() == 0, "reset restores in_use()==0")
    # and the pool is usable again
    var after_reset_ok = True
    for _ in range(8):
        _ = pool.acquire()
    if pool.available() != 0:
        after_reset_ok = False
    check(passed, failed, after_reset_ok, "pool usable after reset")

    print("passed:", passed, " failed:", failed)
    if failed == 0:
        print("ALL POOL TESTS PASSED")

    # ================= BENCHMARK =================
    var N = 5_000_000
    var sink = 0

    # warm pool, small capacity
    var bp = Pool.create(64, 1024)
    # warm-up touch
    var w = bp.acquire()
    bp.release(w)

    var a0 = Int(perf_counter_ns())
    for _ in range(N):
        var p = bp.acquire()
        sink ^= Int(p)
        bp.release(p)
    var ns1 = Int(perf_counter_ns()) - a0
    report("pool acquire+release", N, ns1, sink)

    var b0 = Int(perf_counter_ns())
    for _ in range(N):
        var q = alloc[UInt8](64)
        sink ^= Int(q)
        q.free()
    var ns2 = Int(perf_counter_ns()) - b0
    report("raw alloc+free 64B", N, ns2, sink)

    report_speedup("pool vs raw", ns2, ns1)
