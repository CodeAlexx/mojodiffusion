# sampling_parity.mojo — GPU gate for SDXL Down/Up sampling fwd+bwd
# (models/sdxl/sampling.mojo) vs torch autograd (sampling_oracle.py).
# GATE: out + d_x + d_w + d_b at cos >= 0.999 for BOTH samplers.
#
# Run the oracle first, then:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/sdxl/parity/sampling_oracle.py
#   pixi run mojo build -I . -Xlinker -lm serenitymojo/models/sdxl/parity/sampling_parity.mojo -o /tmp/sdxl_samp
#   /tmp/sdxl_samp

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.models.sdxl.sampling import (
    downsample_forward, downsample_backward,
    upsample_forward, upsample_backward, SampleGrads, UpsampleFwd,
)
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc


comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/models/sdxl/parity/sampling_ref.txt"
)

comptime N = 2
comptime C = 16
comptime DHi = 8   # downsample input
comptime UHi = 4   # upsample input


def _fill(n: Int, a: Int, b: Int, c: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * a) % b) - c) * 0.05)
    return out^

def _fx(n: Int) -> List[Float32]:  return _fill(n, 7, 13, 6.0)
def _fw(n: Int) -> List[Float32]:  return _fill(n, 5, 11, 5.0)
def _fb(n: Int) -> List[Float32]:  return _fill(n, 4, 10, 5.0)
def _fgo(n: Int) -> List[Float32]: return _fill(n, 2, 7, 3.0)

def _sh1(a: Int) -> List[Int]:
    var s = List[Int](); s.append(a); return s^
def _sh4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); s.append(d); return s^


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

    # ── DOWNSAMPLE ────────────────────────────────────────────────────────────
    comptime DHo = (DHi + 2 * 1 - 3) // 2 + 1   # 4
    var dx_in = Tensor.from_host(_fx(N * DHi * DHi * C), _sh4(N, DHi, DHi, C), STDtype.F32, ctx)
    var dop_w = Tensor.from_host(_fw(3 * 3 * C * C), _sh4(3, 3, C, C), STDtype.F32, ctx)
    var dop_b = Tensor.from_host(_fb(C), _sh1(C), STDtype.F32, ctx)

    var d_out = downsample_forward[N, DHi, DHi, C](dx_in, dop_w, dop_b, ctx)
    var r_dout = h.compare_host(d_out.to_host(ctx), _read_ref(String("down_out")))
    print("downsample out vs torch:", r_dout)
    all_pass = all_pass and r_dout.passed

    var d_go = Tensor.from_host(_fgo(N * DHo * DHo * C), _sh4(N, DHo, DHo, C), STDtype.F32, ctx)
    var dg = downsample_backward[N, DHi, DHi, C](d_go, dx_in, dop_w, ctx)
    var r_ddx = h.compare_host(dg.d_x.to_host(ctx), _read_ref(String("down_dx")))
    var r_ddw = h.compare_host(dg.d_w.to_host(ctx), _read_ref(String("down_dw")))
    var r_ddb = h.compare_host(dg.d_b.to_host(ctx), _read_ref(String("down_db")))
    print("downsample d_x vs torch:", r_ddx)
    print("downsample d_w vs torch:", r_ddw)
    print("downsample d_b vs torch:", r_ddb)
    all_pass = all_pass and r_ddx.passed and r_ddw.passed and r_ddb.passed

    # ── UPSAMPLE ──────────────────────────────────────────────────────────────
    comptime UHo = 2 * UHi   # 8 (conv stride1 pad1 keeps size)
    var ux_in = Tensor.from_host(_fx(N * UHi * UHi * C), _sh4(N, UHi, UHi, C), STDtype.F32, ctx)
    var uc_w = Tensor.from_host(_fw(3 * 3 * C * C), _sh4(3, 3, C, C), STDtype.F32, ctx)
    var uc_b = Tensor.from_host(_fb(C), _sh1(C), STDtype.F32, ctx)

    var ufwd = upsample_forward[N, UHi, UHi, C](ux_in, uc_w, uc_b, ctx)
    var r_uout = h.compare_host(ufwd.out.to_host(ctx), _read_ref(String("up_out")))
    print("upsample out vs torch:", r_uout)
    all_pass = all_pass and r_uout.passed

    var u_go = Tensor.from_host(_fgo(N * UHo * UHo * C), _sh4(N, UHo, UHo, C), STDtype.F32, ctx)
    var ug = upsample_backward[N, UHi, UHi, C](u_go, ufwd.up, uc_w, ctx)
    var r_udx = h.compare_host(ug.d_x.to_host(ctx), _read_ref(String("up_dx")))
    var r_udw = h.compare_host(ug.d_w.to_host(ctx), _read_ref(String("up_dw")))
    var r_udb = h.compare_host(ug.d_b.to_host(ctx), _read_ref(String("up_db")))
    print("upsample d_x vs torch:", r_udx)
    print("upsample d_w vs torch:", r_udw)
    print("upsample d_b vs torch:", r_udb)
    all_pass = all_pass and r_udx.passed and r_udw.passed and r_udb.passed

    print("")
    if all_pass:
        print("ALL SDXL DOWN/UP SAMPLING FWD+BWD GATES PASSED (cos >= 0.999 vs PyTorch)")
    else:
        print("SDXL SAMPLING PARITY FAILURE")
        raise Error("sampling_parity gate failed")
