# opt_stableadamw_parity.mojo — GPU verification of opt_stableadamw.mojo.
#
# Reproduces the SAME deterministic fills as opt_stableadamw_oracle.py, runs the
# REAL stableadamw_step (in-place mut API), compares the updated PARAMETER.
# Gate: cos >= 0.999. BITROT GUARD: argv "FAIL" → compare vs zeros, must EXIT 1.
#
# Run oracle first, then:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/training/parity/opt_stableadamw_oracle.py
#   pixi run mojo run -I . serenitymojo/training/parity/opt_stableadamw_parity.mojo

from sys import argv
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.training.opt_stableadamw import stableadamw_step
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc


comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/training/parity/opt_stableadamw_ref.txt"
)

comptime LR = Float32(1.0e-2)
comptime BETA1 = Float32(0.9)
comptime BETA2 = Float32(0.99)
comptime EPS = Float32(1.0e-8)
comptime WD = Float32(0.01)
comptime N = 64
# Gate on cos AND max-abs (cosine is scale-invariant; see skeptic finding).
comptime MAX_ABS_TOL = Float64(1.0e-4)


def _fill(n: Int, a: Int, b: Int, c: Float32, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * a) % b) - c) * scale)
    return out^


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(0.0)
    return out^


def _shape1(n: Int) -> List[Int]:
    var s = List[Int]()
    s.append(n)
    return s^


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


def _run(steps: Int, ctx: DeviceContext) raises -> List[Float32]:
    var p = Tensor.from_host(_fill(N, 7, 13, 6.0, 0.05), _shape1(N), STDtype.F32, ctx)
    var m = Tensor.from_host(_zeros(N), _shape1(N), STDtype.F32, ctx)
    var v = Tensor.from_host(_zeros(N), _shape1(N), STDtype.F32, ctx)
    for t in range(1, steps + 1):
        var g = Tensor.from_host(_fill(N, 5, 11, 5.0, 0.05), _shape1(N), STDtype.F32, ctx)
        stableadamw_step(p, g, m, v, t, LR, BETA1, BETA2, EPS, WD, ctx)
    return p.to_host(ctx)


def main() raises:
    var sabotage = False
    var av = argv()
    for i in range(len(av)):
        if av[i] == String("FAIL"):
            sabotage = True

    var ctx = DeviceContext()
    var h = ParityHarness()
    var all_pass = True

    var p1 = _run(1, ctx)
    var ref1 = _read_ref(String("sadamw_p1"))
    if sabotage:
        ref1 = _zeros(N)
    var r1 = h.compare_host(p1, ref1)
    print("sadamw_p1 vs oracle:", r1)
    all_pass = all_pass and r1.passed and (r1.max_abs <= MAX_ABS_TOL)

    var p5 = _run(5, ctx)
    var ref5 = _read_ref(String("sadamw_p5"))
    if sabotage:
        ref5 = _zeros(N)
    var r5 = h.compare_host(p5, ref5)
    print("sadamw_p5 vs oracle:", r5)
    all_pass = all_pass and r5.passed and (r5.max_abs <= MAX_ABS_TOL)

    print("")
    if all_pass:
        print("STABLEADAMW PARITY PASSED (cos >= 0.999)")
    else:
        print("STABLEADAMW PARITY FAILURE")
        raise Error("opt_stableadamw_parity gate failed")
