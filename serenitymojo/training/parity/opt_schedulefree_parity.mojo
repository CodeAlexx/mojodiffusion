# opt_schedulefree_parity.mojo — verification of opt_schedulefree.mojo.
#
# Reproduces the SAME init/grad/hyperparams as opt_schedulefree_oracle.py, runs
# the REAL RAdamScheduleFree.step 5× (host-F32) + enter_eval_mode, compares the
# y-sequence param and the eval-mode x. Gate: cos >= 0.999. Also verifies the
# enter/exit_eval_mode roundtrip restores y bit-exactly.
# BITROT GUARD: argv "FAIL" → compare vs zeros, must EXIT 1.
#
# Run oracle first, then:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/training/parity/opt_schedulefree_oracle.py
#   pixi run mojo run -I . serenitymojo/training/parity/opt_schedulefree_parity.mojo

from sys import argv
from serenitymojo.parity import ParityHarness
from serenitymojo.training.opt_schedulefree import RAdamScheduleFree
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc


comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/training/parity/opt_schedulefree_ref.txt"
)

comptime LR = Float32(2.5e-3)
comptime BETA1 = Float32(0.9)
comptime BETA2 = Float32(0.999)
comptime EPS = Float32(1.0e-8)
comptime WD = Float32(0.0)
# Gate on cos AND max-abs (cosine is scale-invariant; see skeptic finding).
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


def _init4() -> List[Float32]:
    var out = List[Float32]()
    out.append(0.5); out.append(-0.2); out.append(0.7); out.append(-0.1)
    return out^


def _grad4() -> List[Float32]:
    var out = List[Float32]()
    out.append(0.1); out.append(-0.05); out.append(0.2); out.append(-0.1)
    return out^


def main() raises:
    var sabotage = False
    var av = argv()
    for i in range(len(av)):
        if av[i] == String("FAIL"):
            sabotage = True

    var h = ParityHarness()
    var all_pass = True

    var opt = RAdamScheduleFree(LR, BETA1, BETA2, EPS, WD)
    var p = _init4()
    for _ in range(5):
        var g = _grad4()
        opt.step(p, g)

    # y after 5 steps
    var y_copy = List[Float32]()
    for i in range(len(p)):
        y_copy.append(p[i])
    var ry = _read_ref(String("sf_y5"))
    if sabotage:
        ry = _zeros(4)
    var r1 = h.compare_host(p, ry)
    print("sf_y5 vs oracle:", r1)
    all_pass = all_pass and r1.passed and (r1.max_abs <= MAX_ABS_TOL)

    # enter eval → x ; compare against oracle
    opt.enter_eval_mode(p)
    var rx = _read_ref(String("sf_eval"))
    if sabotage:
        rx = _zeros(4)
    var r2 = h.compare_host(p, rx)
    print("sf_eval vs oracle:", r2)
    all_pass = all_pass and r2.passed and (r2.max_abs <= MAX_ABS_TOL)

    # exit eval → must restore y bit-exactly (roundtrip)
    opt.exit_eval_mode(p)
    var roundtrip_ok = True
    for i in range(len(p)):
        if p[i] != y_copy[i]:
            roundtrip_ok = False
    print("eval roundtrip restores y exactly:", roundtrip_ok)
    all_pass = all_pass and roundtrip_ok

    # wd>0 coupled-L2 branch (fresh optimizer, 5 steps, wd=0.05)
    var opt_wd = RAdamScheduleFree(LR, BETA1, BETA2, EPS, Float32(0.05))
    var pwd = _init4()
    for _ in range(5):
        var g = _grad4()
        opt_wd.step(pwd, g)
    var rwd = _read_ref(String("sf_wd5"))
    if sabotage:
        rwd = _zeros(4)
    var r3 = h.compare_host(pwd, rwd)
    print("sf_wd5 vs oracle:", r3)
    all_pass = all_pass and r3.passed and (r3.max_abs <= MAX_ABS_TOL)

    print("")
    if all_pass:
        print("RADAMSCHEDULEFREE PARITY PASSED (cos >= 0.999, eval roundtrip ok)")
    else:
        print("RADAMSCHEDULEFREE PARITY FAILURE")
        raise Error("opt_schedulefree_parity gate failed")
