# opt_prodigy_parity.mojo — verification of opt_prodigy.mojo.
#
# Drives the SAME strongly-convex quadratic f(x)=0.5 x^T A x, A=diag(2,3,5),
# grad=A@x as opt_prodigy_oracle.py, runs the REAL Prodigy.step (host-F32),
# compares the param trajectory at 10/50/200 steps. Gate: cos >= 0.999.
# Also asserts ||x200|| < 0.1 (convergence). BITROT GUARD: argv "FAIL" → compare
# vs zeros, must EXIT 1.
#
# Run oracle first, then:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/training/parity/opt_prodigy_oracle.py
#   pixi run mojo run -I . serenitymojo/training/parity/opt_prodigy_parity.mojo

from sys import argv
from std.math import sqrt
from serenitymojo.parity import ParityHarness
from serenitymojo.training.opt_prodigy import Prodigy
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc


comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/training/parity/opt_prodigy_ref.txt"
)
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


# run `steps` Prodigy steps on f(x)=0.5 x^T A x, grad = A@x, return x.
def _run(steps: Int, wd: Float32 = Float32(0.0)) raises -> List[Float32]:
    var a = List[Float32]()
    a.append(2.0); a.append(3.0); a.append(5.0)
    var x = List[Float32]()
    x.append(1.0); x.append(-0.7); x.append(0.4)
    var opt = Prodigy(1.0, 0.9, 0.999, 1.0e-8, wd)
    for _ in range(steps):
        var g = List[Float32]()
        for i in range(3):
            g.append(a[i] * x[i])
        opt.step(x, g)
    return x^


def main() raises:
    var sabotage = False
    var av = argv()
    for i in range(len(av)):
        if av[i] == String("FAIL"):
            sabotage = True

    var h = ParityHarness()
    var all_pass = True

    var x10 = _run(10)
    var r10ref = _read_ref(String("prodigy_x10"))
    if sabotage:
        r10ref = _zeros(3)
    var r10 = h.compare_host(x10, r10ref)
    print("prodigy_x10 vs oracle:", r10)
    all_pass = all_pass and r10.passed and (r10.max_abs <= MAX_ABS_TOL)

    var x50 = _run(50)
    var r50ref = _read_ref(String("prodigy_x50"))
    if sabotage:
        r50ref = _zeros(3)
    var r50 = h.compare_host(x50, r50ref)
    print("prodigy_x50 vs oracle:", r50)
    all_pass = all_pass and r50.passed and (r50.max_abs <= MAX_ABS_TOL)

    var x200 = _run(200)
    var r200ref = _read_ref(String("prodigy_x200"))
    if sabotage:
        r200ref = _zeros(3)
    var r200 = h.compare_host(x200, r200ref)
    print("prodigy_x200 vs oracle:", r200)
    all_pass = all_pass and r200.passed and (r200.max_abs <= MAX_ABS_TOL)

    # wd>0 decoupled-WD branch (5 steps, wd=0.05)
    var xwd = _run(5, Float32(0.05))
    var rwdref = _read_ref(String("prodigy_wd5"))
    if sabotage:
        rwdref = _zeros(3)
    var rwd = h.compare_host(xwd, rwdref)
    print("prodigy_wd5 vs oracle:", rwd)
    all_pass = all_pass and rwd.passed and (rwd.max_abs <= MAX_ABS_TOL)

    # convergence sanity: ||x200|| < 0.1
    var nrm = Float32(0.0)
    for i in range(len(x200)):
        nrm += x200[i] * x200[i]
    nrm = sqrt(nrm)
    var converged = nrm < 0.1
    print("prodigy ||x200|| =", nrm, " converged(<0.1)=", converged)
    all_pass = all_pass and converged

    print("")
    if all_pass:
        print("PRODIGY PARITY PASSED (cos >= 0.999, converged)")
    else:
        print("PRODIGY PARITY FAILURE")
        raise Error("opt_prodigy_parity gate failed")
