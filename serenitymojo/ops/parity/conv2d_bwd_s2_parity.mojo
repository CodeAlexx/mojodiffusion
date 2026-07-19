# conv2d_bwd_s2_parity.mojo — GPU verification of the naive conv2d BACKWARD at
# STRIDE 2. Closes the SKEPTIC [MED] gap (ATTACK 1b): the conv2d_backward kernel
# implements general stride (d_x gates on num_h % sh, d_w uses oh*sh-ph+kh) but
# only stride-1 was gated. SDXL Downsample `.op` is a stride-2 Conv3x3.
#
# Shape: N=2, Cin=4, Cout=8, Hi=Wi=8, K=3, stride=2, pad=1 -> Ho=Wo=4.
# Same deterministic fills as conv2d_bwd_oracle.py (shared fills), built in MOJO
# layout; only the reference GRADIENTS cross the boundary.
#
# Run the oracle first, then:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/ops/parity/conv2d_bwd_s2_oracle.py
#   pixi run mojo build -I . -Xlinker -lm serenitymojo/ops/parity/conv2d_bwd_s2_parity.mojo -o /tmp/conv_s2
#   /tmp/conv_s2

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.ops.conv2d_backward import conv2d_backward
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc


comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/ops/parity/conv2d_bwd_s2_ref.txt"
)


# ── deterministic fills — MUST match conv2d_bwd_s2_oracle.py ─────────────────
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

    # ── compile-time shape (MUST match the oracle) — STRIDE 2 ────────────────
    comptime N = 2
    comptime Hi = 8
    comptime Wi = 8
    comptime Cin = 4
    comptime Kh = 3
    comptime Kw = 3
    comptime Cout = 8
    comptime SH = 2
    comptime SW = 2
    comptime PH = 1
    comptime PW = 1
    comptime Ho = (Hi + 2 * PH - Kh) // SH + 1
    comptime Wo = (Wi + 2 * PW - Kw) // SW + 1

    comptime nx = N * Hi * Wi * Cin
    comptime nw = Kh * Kw * Cin * Cout
    comptime ng = N * Ho * Wo * Cout
    _ = Wo

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
    print("conv2d-s2 d_x vs torch:", r_dx)
    print("conv2d-s2 d_w vs torch:", r_dw)
    print("conv2d-s2 d_b vs torch:", r_db)
    all_pass = all_pass and r_dx.passed and r_dw.passed and r_db.passed

    print("")
    if all_pass:
        print("ALL CONV2D STRIDE-2 BACKWARD GATES PASSED (cos >= 0.999 vs PyTorch)")
    else:
        print("CONV2D STRIDE-2 BACKWARD PARITY FAILURE")
        raise Error("conv2d_bwd_s2_parity gate failed")
