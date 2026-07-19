# opt_adafactor_parity.mojo — verification of opt_adafactor.mojo.
#
# Reproduces the SAME deterministic fills as opt_adafactor_oracle.py, runs the
# REAL adafactor_step_factored / adafactor_step_elementwise (host-F32 mut API),
# compares the updated PARAMETER. Gate: cos >= 0.999.
# BITROT GUARD: argv "FAIL" → compare vs zeros, must EXIT 1.
#
# Run oracle first, then:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/training/parity/opt_adafactor_oracle.py
#   pixi run mojo run -I . serenitymojo/training/parity/opt_adafactor_parity.mojo

from sys import argv
from serenitymojo.parity import ParityHarness
from serenitymojo.training.opt_adafactor import (
    adafactor_step_factored, adafactor_step_factored_nd,
    adafactor_step_elementwise, adafactor_eps_param
)
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc


comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/training/parity/opt_adafactor_ref.txt"
)

comptime LR = Float32(1.0e-3)
comptime EPS = Float32(1.0e-3)
# Gate on BOTH cos and max-abs: cosine is scale-invariant, so a uniform
# magnitude error can slip past cos alone (skeptic finding). Host-F32 parity
# tracks to ~1e-6; 1e-4 is a generous but non-vacuous magnitude bound.
comptime MAX_ABS_TOL = Float64(1.0e-4)


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(0.0)
    return out^


def _read_ref(tag: String) raises -> List[Float32]:
    var fd = sys_open(String(REF_PATH), O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open ref: ") + String(REF_PATH))
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error("empty ref file")
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)

    var prefix = tag + " "
    var pl = prefix.byte_length()
    var out = List[Float32]()
    var i = 0
    while i < done:
        var le = i
        while le < done and Int(buf[le]) != 0x0A:
            le += 1
        var is_match = (le - i) > pl
        if is_match:
            for j in range(pl):
                if Int(buf[i + j]) != ord(prefix[byte=j]):
                    is_match = False
                    break
        if is_match:
            var p = i + pl
            while p < le:
                var c = Int(buf[p])
                if c == 0x20:
                    p += 1
                    continue
                var ne = p
                while ne < le and Int(buf[ne]) != 0x20:
                    ne += 1
                var chars = List[UInt8]()
                for q in range(p, ne):
                    chars.append(buf[q])
                var s = String(from_utf8=chars)
                out.append(Float32(atof(s)))
                p = ne + 1
            buf.free()
            return out^
        i = le + 1
    buf.free()
    raise Error(String("ref tag not found: ") + tag)


def _init_p16() -> List[Float32]:
    var out = List[Float32]()
    for i in range(16):
        out.append(Float32(0.1) + Float32(i) * Float32(0.01))
    return out^


def _grad16() -> List[Float32]:
    var out = List[Float32]()
    for i in range(16):
        out.append(Float32(0.05) - Float32(i) * Float32(0.003))
    return out^


# factored 4x4, `steps` steps, scale_parameter, wd
def _run_factored(steps: Int, scale_parameter: Bool, wd: Float32) raises -> List[Float32]:
    var p = _init_p16()
    var row = _zeros(4)
    var col = _zeros(4)
    var epsp = adafactor_eps_param(EPS)
    for t in range(1, steps + 1):
        var g = _grad16()
        adafactor_step_factored(p, g, row, col, 4, 4, t, LR, epsp, wd, scale_parameter)
    return p^


def _run_elementwise(steps: Int, wd: Float32) raises -> List[Float32]:
    var p = _init_p16()
    var v = _zeros(16)
    var epsp = adafactor_eps_param(EPS)
    for t in range(1, steps + 1):
        var g = _grad16()
        adafactor_step_elementwise(p, g, v, t, LR, epsp, wd, False)
    return p^


# rank-3 [L,R,C] fills MUST match opt_adafactor_oracle.af_factored_nd exactly.
def _init_p_nd(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(Float32(0.1) + Float32(i) * Float32(0.011))
    return out^


def _grad_nd(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(Float32(0.05) - Float32(i) * Float32(0.0021))
    return out^


def _run_factored_nd(steps: Int, L: Int, R: Int, C: Int, wd: Float32) raises -> List[Float32]:
    var n = L * R * C
    var p = _init_p_nd(n)
    var row = _zeros(L * R)
    var col = _zeros(L * C)
    var epsp = adafactor_eps_param(EPS)
    for t in range(1, steps + 1):
        var g = _grad_nd(n)
        adafactor_step_factored_nd(p, g, row, col, L, R, C, t, LR, epsp, wd, False)
    return p^


def main() raises:
    var sabotage = False
    var av = argv()
    for i in range(len(av)):
        if av[i] == String("FAIL"):
            sabotage = True

    var h = ParityHarness()
    var all_pass = True

    var fac = _run_factored(5, False, Float32(0.0))
    var rf = _read_ref(String("af_fac_p5"))
    if sabotage:
        rf = _zeros(16)
    var r1 = h.compare_host(fac, rf)
    print("af_fac_p5 vs oracle:", r1)
    all_pass = all_pass and r1.passed and (r1.max_abs <= MAX_ABS_TOL)

    var elem = _run_elementwise(5, Float32(0.0))
    var re = _read_ref(String("af_elem_p5"))
    if sabotage:
        re = _zeros(16)
    var r2 = h.compare_host(elem, re)
    print("af_elem_p5 vs oracle:", r2)
    all_pass = all_pass and r2.passed and (r2.max_abs <= MAX_ABS_TOL)

    var sc = _run_factored(1, True, Float32(0.0))
    var rs = _read_ref(String("af_scale_p1"))
    if sabotage:
        rs = _zeros(16)
    var r3 = h.compare_host(sc, rs)
    print("af_scale_p1 vs oracle:", r3)
    all_pass = all_pass and r3.passed and (r3.max_abs <= MAX_ABS_TOL)

    # rank-3 factored [L=2,R=3,C=4] — exercises the per-L-block row_mean fix.
    var nd = _run_factored_nd(5, 2, 3, 4, Float32(0.0))
    var rnd = _read_ref(String("af_nd_p5"))
    if sabotage:
        rnd = _zeros(24)
    var r4 = h.compare_host(nd, rnd)
    print("af_nd_p5 vs oracle:", r4)
    all_pass = all_pass and r4.passed and (r4.max_abs <= MAX_ABS_TOL)

    # wd>0 decoupled WD branch (rank-2 4x4, wd=0.05)
    var wdp = _run_factored(5, False, Float32(0.05))
    var rwd = _read_ref(String("af_wd_p5"))
    if sabotage:
        rwd = _zeros(16)
    var r5 = h.compare_host(wdp, rwd)
    print("af_wd_p5 vs oracle:", r5)
    all_pass = all_pass and r5.passed and (r5.max_abs <= MAX_ABS_TOL)

    print("")
    if all_pass:
        print("ADAFACTOR PARITY PASSED (cos >= 0.999)")
    else:
        print("ADAFACTOR PARITY FAILURE")
        raise Error("opt_adafactor_parity gate failed")
