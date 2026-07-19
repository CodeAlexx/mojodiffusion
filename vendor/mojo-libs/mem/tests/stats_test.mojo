# mem/tests/stats_test.mojo
from mem.stats import MemStats, TrackingAllocator, STATS_ENABLED, maybe_record_alloc


def check(mut passed: Int, mut failed: Int, cond: Bool, name: String):
    if cond:
        passed += 1
    else:
        failed += 1
        print("  FAIL:", name)


def main() raises:
    var passed = 0
    var failed = 0

    # ---- MemStats: basic alloc/free accounting ----
    var s = MemStats()
    s.record_alloc(100)
    s.record_alloc(50)
    check(passed, failed, s.live_bytes == 150, "live 150 after 100+50")
    check(passed, failed, s.peak_bytes == 150, "peak 150 after 100+50")
    check(passed, failed, s.alloc_count == 2, "alloc_count 2")
    check(passed, failed, s.live_count == 2, "live_count 2")

    s.record_free(100)
    check(passed, failed, s.live_bytes == 50, "live 50 after free 100")
    check(passed, failed, s.peak_bytes == 150, "peak STILL 150 after free")
    check(passed, failed, s.free_count == 1, "free_count 1")
    check(passed, failed, s.live_count == 1, "live_count 1 after free")
    check(passed, failed, s.leaked_bytes() == 50, "leaked_bytes 50")
    check(passed, failed, s.leaked_count() == 1, "leaked_count 1")

    # ---- peak tracking: alloc 10, alloc 10, free 20, alloc 5 ----
    var p = MemStats()
    p.record_alloc(10)
    p.record_alloc(10)
    p.record_free(10)
    p.record_free(10)
    p.record_alloc(5)
    check(passed, failed, p.peak_bytes == 20, "peak_bytes 20 (10+10)")
    check(passed, failed, p.live_bytes == 5, "live 5 after frees + alloc 5")
    check(passed, failed, p.peak_count == 2, "peak_count 2")

    # ---- reset ----
    var r = MemStats()
    r.record_alloc(99)
    r.reset()
    check(passed, failed, r.live_bytes == 0 and r.peak_bytes == 0, "reset zeros live/peak")
    check(passed, failed, r.alloc_count == 0 and r.total_allocated == 0, "reset zeros counts/totals")

    # ---- TrackingAllocator: usable region + header integrity ----
    var ta = TrackingAllocator()
    var b0 = ta.allocate(32)
    var b1 = ta.allocate(64)
    var b2 = ta.allocate(128)

    # Prove each returned region is usable and the header didn't corrupt data:
    # write a distinct byte at the start AND last index of each buffer.
    b0[0] = UInt8(0xA0)
    b0[31] = UInt8(0xA1)
    b1[0] = UInt8(0xB0)
    b1[63] = UInt8(0xB1)
    b2[0] = UInt8(0xC0)
    b2[127] = UInt8(0xC1)
    check(passed, failed, b0[0] == UInt8(0xA0) and b0[31] == UInt8(0xA1), "b0 (32B) rw")
    check(passed, failed, b1[0] == UInt8(0xB0) and b1[63] == UInt8(0xB1), "b1 (64B) rw")
    check(passed, failed, b2[0] == UInt8(0xC0) and b2[127] == UInt8(0xC1), "b2 (128B) rw")
    # Cross-check: writes to b1/b2 did not clobber b0's bytes (no overlap).
    check(passed, failed, b0[0] == UInt8(0xA0) and b0[31] == UInt8(0xA1), "b0 intact after b1/b2 writes")

    var st = ta.stats()
    check(passed, failed, st.live_bytes == 224, "live_bytes 224 (32+64+128)")
    check(passed, failed, st.live_count == 3, "live_count 3")
    check(passed, failed, st.total_allocated == 224, "total_allocated 224")

    # deallocate all → fully reclaimed, no leaks
    ta.deallocate(b0)
    ta.deallocate(b1)
    ta.deallocate(b2)
    var st2 = ta.stats()
    check(passed, failed, st2.live_bytes == 0, "live_bytes 0 after free all")
    check(passed, failed, st2.live_count == 0, "live_count 0 after free all")
    check(passed, failed, st2.leaked_count() == 0, "leaked_count 0 after free all")
    check(passed, failed, st2.total_allocated == 224, "total_allocated still 224")
    check(passed, failed, st2.total_freed == 224, "total_freed 224")
    check(passed, failed, not ta.leaks(), "leaks() False after balanced free")

    # ---- deliberate leak: allocate without deallocate ----
    var leaky = TrackingAllocator()
    var _orphan = leaky.allocate(16)
    check(passed, failed, leaky.live_count() != 0, "leak detected: live_count != 0")
    check(passed, failed, leaky.leaks(), "leaks() True for orphaned alloc")
    # (intentionally not freed — this is the leak the allocator is meant to flag)

    # ---- comptime-gated path compiles (STATS_ENABLED True here) ----
    var g = MemStats()
    maybe_record_alloc(g, 7)
    check(passed, failed, g.live_bytes == 7, "maybe_record_alloc gated path works (ENABLED)")
    check(passed, failed, STATS_ENABLED, "STATS_ENABLED flag visible at runtime")

    print("passed:", passed, "failed:", failed)
    if failed == 0:
        print("ALL STATS TESTS PASSED")
