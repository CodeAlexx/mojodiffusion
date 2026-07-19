# memory/tests/arena_test.mojo
from mem.aligned import align_up, align_up_pow2, is_power_of_two, AlignedBuffer
from mem.arena import Arena


def check(mut passed: Int, mut failed: Int, cond: Bool, name: String):
    if cond:
        passed += 1
    else:
        failed += 1
        print("  FAIL:", name)


def main() raises:
    var passed = 0
    var failed = 0

    # ---- align helpers ----
    check(passed, failed, align_up(0, 16) == 0, "align_up(0,16)")
    check(passed, failed, align_up(1, 16) == 16, "align_up(1,16)")
    check(passed, failed, align_up(16, 16) == 16, "align_up(16,16)")
    check(passed, failed, align_up(17, 16) == 32, "align_up(17,16)")
    check(passed, failed, align_up(10, 1) == 10, "align_up(_,1)")
    check(passed, failed, align_up(7, 3) == 9, "align_up(7,3) non-pow2")
    check(passed, failed, align_up_pow2(13, 8) == 16, "align_up_pow2(13,8)")
    check(passed, failed, is_power_of_two(64), "is_pow2(64)")
    check(passed, failed, not is_power_of_two(48), "not is_pow2(48)")
    check(passed, failed, not is_power_of_two(0), "not is_pow2(0)")

    # ---- AlignedBuffer ----
    var ab = AlignedBuffer.alloc_aligned(1000, 64)
    check(passed, failed, ab.is_aligned(), "AlignedBuffer 64-aligned")
    check(passed, failed, (Int(ab.data()) % 64) == 0, "addr %64==0")
    check(passed, failed, ab.size() == 1000, "AlignedBuffer size")
    var d = ab.data()
    d[0] = UInt8(0xAB)
    d[999] = UInt8(0xCD)
    check(passed, failed, d[0] == UInt8(0xAB) and d[999] == UInt8(0xCD), "AlignedBuffer rw")

    # ---- Arena: bump + alignment ----
    var a = Arena.with_capacity(4096)
    check(passed, failed, a.capacity() == 4096, "arena capacity")
    check(passed, failed, a.used() == 0, "arena starts empty")

    var p0 = a.alloc_bytes(10, 16)
    check(passed, failed, (Int(p0) % 16) == 0, "p0 16-aligned")
    check(passed, failed, a.used() == 10, "used after 10B")

    var p1 = a.alloc_bytes(1, 16)
    check(passed, failed, (Int(p1) % 16) == 0, "p1 re-aligned to 16")
    check(passed, failed, Int(p1) - Int(p0) == 16, "p1 is 16 past p0 (alignment gap)")
    check(passed, failed, a.used() == 17, "used after 10+align+1")

    # writes land in distinct regions
    p0[0] = UInt8(11)
    p1[0] = UInt8(22)
    check(passed, failed, p0[0] == UInt8(11) and p1[0] == UInt8(22), "distinct regions rw")

    # ---- typed alloc_array ----
    var ints = a.alloc_array[Int32](100)
    check(passed, failed, (Int(ints) % 4) == 0, "Int32 array aligned")
    for i in range(100):
        ints[i] = Int32(i * i)
    var sum = 0
    for i in range(100):
        sum += Int(ints[i])
    check(passed, failed, sum == 328350, "Int32 array values (sum 0..99 squared)")

    # ---- peak + reset ----
    var peak_before = a.peak_used()
    check(passed, failed, peak_before >= 417, "peak tracked")
    a.reset()
    check(passed, failed, a.used() == 0, "reset clears used")
    check(passed, failed, a.peak_used() == peak_before, "reset keeps peak")
    var p2 = a.alloc_bytes(8)
    check(passed, failed, Int(p2) == Int(a._base), "reuse from base after reset")

    # ---- out-of-memory raises ----
    var small = Arena.with_capacity(32)
    var raised = False
    try:
        _ = small.alloc_bytes(100)
    except:
        raised = True
    check(passed, failed, raised, "OOM raises")

    print("passed:", passed, " failed:", failed)
    if failed == 0:
        print("ALL ARENA TESTS PASSED")
