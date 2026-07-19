# mem/tests/ring_test.mojo
from std.memory import alloc, UnsafePointer
from mem.aligned import BytePtr
from mem.ring import ByteRing
from mem.bench import report
from std.time import perf_counter_ns


def check(mut passed: Int, mut failed: Int, cond: Bool, name: String):
    if cond:
        passed += 1
    else:
        failed += 1
        print("  FAIL:", name)


def main() raises:
    var passed = 0
    var failed = 0

    # ============ fill / full / drain / empty ============
    var r = ByteRing.with_capacity(4)
    check(passed, failed, r.capacity() == 4, "capacity()==4")
    check(passed, failed, r.is_empty(), "fresh is_empty()")
    check(passed, failed, r.len() == 0, "fresh len()==0")
    check(passed, failed, r.free_space() == 4, "fresh free_space()==4")

    r.push_byte(10)
    r.push_byte(20)
    r.push_byte(30)
    r.push_byte(40)
    check(passed, failed, r.is_full(), "is_full() after 4 pushes")
    check(passed, failed, r.len() == 4, "len()==4 after 4 pushes")
    check(passed, failed, r.free_space() == 0, "free_space()==0 when full")

    # 5th push raises
    var raised_full = False
    try:
        r.push_byte(50)
    except:
        raised_full = True
    check(passed, failed, raised_full, "5th push raises (full)")

    # pop all 4 in FIFO order
    var v0 = r.pop_byte()
    var v1 = r.pop_byte()
    var v2 = r.pop_byte()
    var v3 = r.pop_byte()
    check(passed, failed, v0 == 10, "FIFO pop[0]==10")
    check(passed, failed, v1 == 20, "FIFO pop[1]==20")
    check(passed, failed, v2 == 30, "FIFO pop[2]==30")
    check(passed, failed, v3 == 40, "FIFO pop[3]==40")

    # 5th pop raises
    var raised_empty = False
    try:
        _ = r.pop_byte()
    except:
        raised_empty = True
    check(passed, failed, raised_empty, "5th pop raises (empty)")
    check(passed, failed, r.is_empty(), "is_empty() after draining 4")

    # ============ wrap-around: push 3, pop 2, push 3, pop 4 ============
    var w = ByteRing.with_capacity(4)
    w.push_byte(1)
    w.push_byte(2)
    w.push_byte(3)
    var a0 = w.pop_byte()  # 1
    var a1 = w.pop_byte()  # 2
    check(passed, failed, a0 == 1 and a1 == 2, "wrap: popped 1,2")
    # ring now holds [3] with head/tail advanced; pushing 3 more wraps
    w.push_byte(4)
    w.push_byte(5)
    w.push_byte(6)
    check(passed, failed, w.len() == 4, "wrap: len()==4 (3,4,5,6)")
    check(passed, failed, w.is_full(), "wrap: is_full()")
    var b0 = w.pop_byte()  # 3
    var b1 = w.pop_byte()  # 4
    var b2 = w.pop_byte()  # 5
    var b3 = w.pop_byte()  # 6
    var wrap_ok = b0 == 3 and b1 == 4 and b2 == 5 and b3 == 6
    check(passed, failed, wrap_ok, "wrap: FIFO across wrap = 3,4,5,6")
    check(passed, failed, w.is_empty(), "wrap: empty after draining")

    # ============ bulk push/pop ============
    var br = ByteRing.with_capacity(8)

    # build a 5-byte source: 100..104
    var src1 = alloc[UInt8](5)
    for i in range(5):
        src1[i] = UInt8(100 + i)
    var n1 = br.push(src1, 5)
    check(passed, failed, n1 == 5, "bulk push 5 returns 5")
    check(passed, failed, br.len() == 5, "bulk len()==5")

    # second 5-byte source: 200..204, only 3 free
    var src2 = alloc[UInt8](5)
    for i in range(5):
        src2[i] = UInt8(200 + i)
    var n2 = br.push(src2, 5)
    check(passed, failed, n2 == 3, "bulk push 5 (3 free) returns 3")
    check(passed, failed, br.len() == 8, "bulk len()==8 (full)")
    check(passed, failed, br.is_full(), "bulk is_full()")

    # pop into a 10-byte dst: returns 8, first 8 pushed in order
    var dst = alloc[UInt8](10)
    for i in range(10):
        dst[i] = UInt8(0)
    var n3 = br.pop(dst, 10)
    check(passed, failed, n3 == 8, "bulk pop into 10-byte dst returns 8")
    # expected: 100,101,102,103,104, 200,201,202
    var expect = List[UInt8]()
    expect.append(100)
    expect.append(101)
    expect.append(102)
    expect.append(103)
    expect.append(104)
    expect.append(200)
    expect.append(201)
    expect.append(202)
    var bulk_order_ok = True
    for i in range(8):
        if dst[i] != expect[i]:
            bulk_order_ok = False
    check(passed, failed, bulk_order_ok, "bulk pop preserves FIFO order")
    check(passed, failed, br.is_empty(), "bulk empty after popping all")

    # bulk that wraps: cap 8, fill 6, pop 4, push 5 (wraps), pop 7
    var bw = ByteRing.with_capacity(8)
    var s = alloc[UInt8](6)
    for i in range(6):
        s[i] = UInt8(i + 1)  # 1..6
    _ = bw.push(s, 6)
    var d4 = alloc[UInt8](4)
    _ = bw.pop(d4, 4)  # removes 1,2,3,4 -> leaves 5,6
    var s5 = alloc[UInt8](5)
    for i in range(5):
        s5[i] = UInt8(i + 7)  # 7..11
    var nw = bw.push(s5, 5)  # 6 free now, writes 5, wraps across end
    check(passed, failed, nw == 5, "bulk wrap push returns 5")
    check(passed, failed, bw.len() == 7, "bulk wrap len()==7 (5,6,7..11)")
    var d7 = alloc[UInt8](7)
    var nr = bw.pop(d7, 7)
    check(passed, failed, nr == 7, "bulk wrap pop returns 7")
    var ewrap = List[UInt8]()
    ewrap.append(5)
    ewrap.append(6)
    ewrap.append(7)
    ewrap.append(8)
    ewrap.append(9)
    ewrap.append(10)
    ewrap.append(11)
    var wrap_bulk_ok = True
    for i in range(7):
        if d7[i] != ewrap[i]:
            wrap_bulk_ok = False
    check(passed, failed, wrap_bulk_ok, "bulk wrap pop FIFO = 5,6,7..11")

    # ============ try_* variants ============
    var t = ByteRing.with_capacity(2)
    check(passed, failed, t.try_push_byte(1), "try_push 1 ok")
    check(passed, failed, t.try_push_byte(2), "try_push 2 ok")
    check(passed, failed, not t.try_push_byte(3), "try_push 3 False (full)")
    var ob = UInt8(0)
    check(passed, failed, t.try_pop_byte(ob) and ob == 1, "try_pop 1")
    check(passed, failed, t.try_pop_byte(ob) and ob == 2, "try_pop 2")
    check(passed, failed, not t.try_pop_byte(ob), "try_pop False (empty)")

    # ============ clear ============
    var c = ByteRing.with_capacity(4)
    c.push_byte(9)
    c.push_byte(8)
    check(passed, failed, c.len() == 2, "before clear len()==2")
    c.clear()
    check(passed, failed, c.is_empty(), "clear empties it")
    check(passed, failed, c.free_space() == 4, "clear restores free_space()==4")
    # usable again after clear
    c.push_byte(42)
    check(passed, failed, c.pop_byte() == 42, "usable after clear")

    # ============ non-pow2 capacity (modulo path) ============
    var np = ByteRing.with_capacity(5)
    for i in range(5):
        np.push_byte(UInt8(i))
    check(passed, failed, np.is_full(), "non-pow2 fills to 5")
    var np_ok = True
    for i in range(5):
        if np.pop_byte() != UInt8(i):
            np_ok = False
    check(passed, failed, np_ok, "non-pow2 FIFO order")

    # free scratch buffers
    src1.free()
    src2.free()
    dst.free()
    s.free()
    d4.free()
    s5.free()
    d7.free()

    print("passed:", passed, "failed:", failed)
    if failed == 0:
        print("ALL RING TESTS PASSED")

    # ==================== BENCHMARK ====================
    var sink = 0

    # --- single-byte push+pop throughput, ring kept half-full ---
    var ring = ByteRing.with_capacity(4096)
    # prime to half full so each iter stays half-full
    for _ in range(2048):
        ring.push_byte(0)

    var N = 20_000_000
    var t0 = Int(perf_counter_ns())
    for i in range(N):
        _ = ring.try_push_byte(UInt8(i & 0xFF))
        var pv = ring.pop_byte()
        sink += Int(pv)
    var ns1 = Int(perf_counter_ns()) - t0
    report("ring push+pop byte", N, ns1, sink)

    # --- bulk throughput: push 256 + pop 256, measure MB/s ---
    var chunk = 256
    var bring = ByteRing.with_capacity(4096)
    var bsrc = alloc[UInt8](chunk)
    var bdst = alloc[UInt8](chunk)
    for i in range(chunk):
        bsrc[i] = UInt8(i & 0xFF)

    var rounds = 2_000_000
    var t1 = Int(perf_counter_ns())
    for _ in range(rounds):
        var pn = bring.push(bsrc, chunk)
        var qn = bring.pop(bdst, chunk)
        sink += Int(bdst[0]) + Int(bdst[chunk - 1]) + pn + qn
    var ns2 = Int(perf_counter_ns()) - t1
    # bytes moved = rounds * chunk (pushed) + rounds * chunk (popped)
    var total_bytes = rounds * chunk * 2
    var mbps = (Float64(total_bytes) / 1.0e6) / (Float64(ns2) / 1.0e9)
    report("ring bulk push+pop 256B", rounds, ns2, sink)
    print(
        "  bulk throughput:",
        total_bytes,
        "bytes /",
        ns2,
        "ns =",
        mbps,
        "MB/s  [sink=",
        sink & 0xFFFF,
        "]",
    )

    bsrc.free()
    bdst.free()
