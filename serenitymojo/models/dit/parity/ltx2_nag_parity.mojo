# ltx2_nag_parity.mojo — GPU unit gate for the LTX-2 NAG combine.
#
# Verifies serenitymojo/models/dit/ltx2_nag.mojo::nag_combine against the Python
# nag.py reference (_nag_combine) at cos >= 0.999. Three cases:
#   caseA  S6  x D16   small magnitude  (most rows BELOW tau -> no L1 clip)
#   caseB  S6  x D16   large magnitude  (rows ABOVE tau -> L1 clip ACTIVE)
#   caseC  S4  x D128  video-like width small magnitude
#
# Inputs are filled by the SAME deterministic formula as ltx2_nag_oracle.py; only
# the reference OUTPUT is read from ltx2_nag_ref.txt. Run the oracle first:
#   python3 serenitymojo/models/dit/parity/ltx2_nag_oracle.py
#   pixi run mojo run -I . serenitymojo/models/dit/parity/ltx2_nag_parity.mojo
#
# F32 throughout (the kernel accumulates the L1 norms in F32; nag.py is float32).

from std.gpu.host import DeviceContext
from std.memory import alloc

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.models.dit.ltx2_nag import nag_combine
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY


comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ltx2_nag_ref.txt"
)
comptime SCALE = Float32(11.0)
comptime ALPHA = Float32(0.25)
comptime TAU = Float32(2.5)


# Deterministic fills — MUST match ltx2_nag_oracle.py fill_pos / fill_neg.
def _fill_pos(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 7) % 13) - 6.0) * 0.05)
    return out^


def _fill_neg(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 5) % 11) - 5.0) * 0.05)
    return out^


def _fill_pos_big(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 7) % 13) - 6.0) * 0.5)
    return out^


def _fill_neg_big(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 5) % 11) - 5.0) * 0.5)
    return out^


def _shape3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a); s.append(b); s.append(c)
    return s^


# ── read one tagged space-separated float line from the ref file ─────────────
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

    var out = List[Float32]()
    var pl = len(tag)
    var i = 0
    while i < n:
        var le = i
        while le < n and Int(buf[le]) != 0x0A:  # newline
            le += 1
        var is_match = (le - i) > pl
        if is_match:
            for j in range(pl):
                if Int(buf[i + j]) != ord(tag[byte=j]):
                    is_match = False
                    break
        if is_match and (le - i) > pl and Int(buf[i + pl]) == 0x20:
            var p = i + pl + 1
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
                out.append(Float32(atof(String(from_utf8=chars))))
                p = ne + 1
            buf.free()
            return out^
        i = le + 1
    buf.free()
    raise Error(String("ref tag not found: ") + tag)


def main() raises:
    var ctx = DeviceContext()
    var h = ParityHarness()
    var all_pass = True

    # ── Case A: small magnitude, S6 x D16 (no clip) ──
    var S_a = 6
    var D_a = 16
    var n_a = S_a * D_a
    var posA = Tensor.from_host(_fill_pos(n_a), _shape3(1, S_a, D_a), STDtype.F32, ctx)
    var negA = Tensor.from_host(_fill_neg(n_a), _shape3(1, S_a, D_a), STDtype.F32, ctx)
    var outA = nag_combine(posA, negA, SCALE, ALPHA, TAU, ctx)
    var rA = h.compare(outA, _read_ref(String("caseA")), ctx)
    print("A) NAG combine small (no clip)  :", rA)
    all_pass = all_pass and rA.passed

    # ── Case B: large magnitude, S6 x D16 (L1 clip active) ──
    var posB = Tensor.from_host(_fill_pos_big(n_a), _shape3(1, S_a, D_a), STDtype.F32, ctx)
    var negB = Tensor.from_host(_fill_neg_big(n_a), _shape3(1, S_a, D_a), STDtype.F32, ctx)
    var outB = nag_combine(posB, negB, SCALE, ALPHA, TAU, ctx)
    var rB = h.compare(outB, _read_ref(String("caseB")), ctx)
    print("B) NAG combine large (L1 clip)  :", rB)
    all_pass = all_pass and rB.passed

    # ── Case C: video-like width S4 x D128 ──
    var S_c = 4
    var D_c = 128
    var n_c = S_c * D_c
    var posC = Tensor.from_host(_fill_pos(n_c), _shape3(1, S_c, D_c), STDtype.F32, ctx)
    var negC = Tensor.from_host(_fill_neg(n_c), _shape3(1, S_c, D_c), STDtype.F32, ctx)
    var outC = nag_combine(posC, negC, SCALE, ALPHA, TAU, ctx)
    var rC = h.compare(outC, _read_ref(String("caseC")), ctx)
    print("C) NAG combine D128 width       :", rC)
    all_pass = all_pass and rC.passed

    print("")
    if all_pass:
        print("ALL NAG COMBINE GATES PASSED (cos >= 0.999)")
    else:
        print("NAG COMBINE PARITY FAILURE")
        raise Error("ltx2_nag_parity gate failed")
