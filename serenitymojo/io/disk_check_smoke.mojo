# io/disk_check_smoke.mojo — gate for the free-disk-space probe + guard.
#
# TENET 4 / parity-bitrot guard: every check ASSERTS and the file `raise`s
# (nonzero exit) on any wrong value. Bitrot demo:
#   DC_BREAK_GUARD=1 inverts the guard expectation (asserts the huge-required
#     guard did NOT raise) → that assertion fails (exit != 0), proving the gate
#     is not vacuous.
#
# GATE (a) POSITIVE: free_bytes_for("/tmp") > 0 (a real mounted fs has free space).
# GATE (b) GUARD-PASS: guard_free_space("/tmp", 1) does NOT raise (1 byte fits).
# GATE (c) GUARD-RAISE: guard_free_space("/tmp", huge) RAISES (demand > free).
#   We capture the raise via try/except and assert it fired.
# GATE (d) ANCESTOR: free_bytes_for(<nonexistent>/sub) walks up to an existing
#   ancestor and still returns a positive byte count (mirrors the rs parent walk).
#
# Run (clean PASS, exit 0):
#   rm -f serenitymojo.mojopkg && pixi run mojo run -I . \
#     serenitymojo/io/disk_check_smoke.mojo
# Run (deliberate FAIL, exit != 0):
#   DC_BREAK_GUARD=1 pixi run mojo run -I . \
#     serenitymojo/io/disk_check_smoke.mojo

from std.ffi import external_call
from std.memory import alloc, UnsafePointer
from std.builtin.type_aliases import MutExternalOrigin

from serenitymojo.io.disk_check import free_bytes_for, guard_free_space


comptime _EnvPtr = UnsafePointer[UInt8, MutExternalOrigin]


def _env_is_set(name: String) -> Bool:
    var n = name.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = name.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    var cname = _EnvPtr(unsafe_from_address=Int(buf))
    var ret = external_call["getenv", _EnvPtr](cname)
    buf.free()
    if Int(ret) == 0:
        return False
    return ret[0] == UInt8(49) and ret[1] == UInt8(0)


def main() raises:
    var break_guard = _env_is_set(String("DC_BREAK_GUARD"))
    var ok = True

    # ── (a) free_bytes_for("/tmp") > 0 ──
    var free_tmp = free_bytes_for(String("/tmp"))
    print("free_bytes_for('/tmp') =", free_tmp, "bytes")
    if free_tmp > 0:
        print("PASS (a): /tmp reports positive free bytes =", free_tmp)
    else:
        print("FAIL (a): /tmp free bytes not positive (got", free_tmp, ")"); ok = False

    # ── (b) guard passes for a 1-byte requirement ──
    var guard_passed = True
    try:
        guard_free_space(String("/tmp"), 1)
    except e:
        guard_passed = False
        print("FAIL (b): guard raised for a 1-byte requirement:", e)
    if guard_passed:
        print("PASS (b): guard_free_space('/tmp', 1) did not raise")
    else:
        ok = False

    # ── (c) guard RAISES for an unreasonable requirement (1 EB) ──
    var huge = 1000000000000000000   # 1 EB — far above any test host's free space
    var guard_raised = False
    try:
        guard_free_space(String("/tmp"), huge)
    except e:
        guard_raised = True
    # BITROT DEMO: invert the expectation so the assertion is wrong-but-checked.
    var expect_raise = True
    if break_guard:
        expect_raise = False
        print("INFO: DC_BREAK_GUARD set — inverting the guard-raise expectation to prove the gate catches it")
    if guard_raised == expect_raise:
        if guard_raised:
            print("PASS (c): guard_free_space('/tmp', 1EB) raised as required")
        else:
            print("PASS (c): guard did not raise (UNEXPECTED — only valid under break demo)")
    else:
        print("FAIL (c): guard raise =", guard_raised, " but expected", expect_raise); ok = False

    # ── (d) ancestor walk: nonexistent path still probes its existing parent ──
    var bogus = String("/tmp/.serenitymojo_nonexistent_dc_path_xyz/sub/leaf")
    var free_anc = free_bytes_for(bogus)
    print("free_bytes_for(<nonexistent under /tmp>) =", free_anc, "bytes")
    if free_anc > 0:
        print("PASS (d): ancestor walk resolved to an existing fs, free =", free_anc)
    else:
        print("FAIL (d): ancestor walk did not yield positive free bytes (got", free_anc, ")"); ok = False

    if not ok:
        raise Error("disk_check_smoke FAILED")
    print("disk_check_smoke ALL GATES PASS")
