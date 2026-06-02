# conv2d_bwd_parity.mojo — GPU verification of the naive conv2d BACKWARD.
#
# Tier-5 gate: grad-parity cos >= 0.999 of d_x / d_w / d_b vs a PyTorch reference
# (conv2d_bwd_oracle.py -> conv2d_bwd_ref.txt).
#
# Inputs use the SAME deterministic fills as conv2d_bwd_oracle.py (fill_x/fill_w/
# fill_gy), built in MOJO layout (NHWC x, RSCF w, NHWC grad_y). Only the
# reference GRADIENTS are read from the ref file (already permuted back to mojo
# layout by the oracle).
#
# Shape: N=2, Cin=3, Cout=4, H=W=8, K=3, stride=1, pad=1.
#
# Run the oracle first, then:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/ops/parity/conv2d_bwd_oracle.py
#   pixi run mojo run -I . serenitymojo/ops/parity/conv2d_bwd_parity.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.ops.conv2d_backward import conv2d_backward
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc


comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/ops/parity/conv2d_bwd_ref.txt"
)


# ── deterministic fills — MUST match conv2d_bwd_oracle.py ─────────────────────
def _fill_x(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 7) % 13) - 6.0) * 0.05)
    return out^


def _fill_w(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 5) % 11) - 5.0) * 0.05)
    return out^


def _fill_gy(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 3) % 9) - 4.0) * 0.05)
    return out^


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a); s.append(b); s.append(c); s.append(d)
    return s^


# ── read one tagged space-separated float line (mirrors sdpa_bwd_parity) ──────
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


def main() raises:
    var ctx = DeviceContext()
    var h = ParityHarness()
    var all_pass = True

    # ── compile-time shape (MUST match the oracle) ───────────────────────────
    comptime N = 2
    comptime Hi = 8
    comptime Wi = 8
    comptime Cin = 3
    comptime Kh = 3
    comptime Kw = 3
    comptime Cout = 4
    comptime SH = 1
    comptime SW = 1
    comptime PH = 1
    comptime PW = 1
    comptime Ho = (Hi + 2 * PH - Kh) // SH + 1
    comptime Wo = (Wi + 2 * PW - Kw) // SW + 1

    comptime nx = N * Hi * Wi * Cin
    comptime nw = Kh * Kw * Cin * Cout
    comptime ng = N * Ho * Wo * Cout
    _ = Wo  # Wo only needed via ng above; silence unused-comptime warning

    var x = Tensor.from_host(_fill_x(nx), _shape4(N, Hi, Wi, Cin), STDtype.F32, ctx)
    var w = Tensor.from_host(_fill_w(nw), _shape4(Kh, Kw, Cin, Cout), STDtype.F32, ctx)
    var gy = Tensor.from_host(_fill_gy(ng), _shape4(N, Ho, Wo, Cout), STDtype.F32, ctx)

    var grads = conv2d_backward[
        N, Hi, Wi, Cin, Kh, Kw, Cout, SH, SW, PH, PW
    ](x, w, gy, ctx)

    var dx = grads.d_x.to_host(ctx)
    var dw = grads.d_w.to_host(ctx)
    var db = grads.d_b.to_host(ctx)

    var r_dx = h.compare_host(dx, _read_ref(String("conv_dx")))
    var r_dw = h.compare_host(dw, _read_ref(String("conv_dw")))
    var r_db = h.compare_host(db, _read_ref(String("conv_db")))
    print("conv2d d_x vs torch:", r_dx)
    print("conv2d d_w vs torch:", r_dw)
    print("conv2d d_b vs torch:", r_db)
    all_pass = all_pass and r_dx.passed and r_dw.passed and r_db.passed

    print("")
    if all_pass:
        print("ALL CONV2D BACKWARD GATES PASSED (cos >= 0.999 vs PyTorch)")
    else:
        print("CONV2D BACKWARD PARITY FAILURE")
        raise Error("conv2d_bwd_parity gate failed")
