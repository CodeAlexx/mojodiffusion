# geglu_parity.mojo — GPU gate for SDXL SpatialTransformer FF GEGLU fwd+bwd
# (models/sdxl/geglu.mojo) vs torch autograd, TANH-approx GELU (geglu_oracle.py).
# GATE: out + d_x + d_proj_w + d_proj_b at cos >= 0.999.
#
# Run the oracle first, then:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/sdxl/parity/geglu_oracle.py
#   pixi run mojo build -I . -Xlinker -lm serenitymojo/models/sdxl/parity/geglu_parity.mojo -o /tmp/sdxl_geglu
#   /tmp/sdxl_geglu

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.models.sdxl.geglu import (
    geglu_forward, geglu_backward, GegluActs, GegluFwd, GegluGrads,
)
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc


comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/models/sdxl/parity/geglu_ref.txt"
)

comptime M = 6
comptime Cin = 8
comptime Cff = 5


def _fill(n: Int, a: Int, b: Int, c: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * a) % b) - c) * 0.05)
    return out^

def _sh1(a: Int) -> List[Int]:
    var s = List[Int](); s.append(a); return s^
def _sh2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^


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

    var x = Tensor.from_host(_fill(M * Cin, 7, 13, 6.0), _sh2(M, Cin), STDtype.F32, ctx)
    var proj_w = Tensor.from_host(_fill(2 * Cff * Cin, 5, 11, 5.0), _sh2(2 * Cff, Cin), STDtype.F32, ctx)
    var proj_b = Tensor.from_host(_fill(2 * Cff, 4, 10, 5.0), _sh1(2 * Cff), STDtype.F32, ctx)

    var fwd = geglu_forward[M, Cin, Cff](x, proj_w, proj_b, ctx)
    var r_out = h.compare_host(fwd.out.to_host(ctx), _read_ref(String("out")))
    print("geglu out      vs torch:", r_out)
    all_pass = all_pass and r_out.passed

    var go = Tensor.from_host(_fill(M * Cff, 2, 7, 3.0), _sh2(M, Cff), STDtype.F32, ctx)
    var g = geglu_backward[M, Cin, Cff](go, fwd.acts, proj_w, ctx)

    var r_dx = h.compare_host(g.d_x.to_host(ctx), _read_ref(String("d_x")))
    var r_dw = h.compare_host(g.d_proj_w.to_host(ctx), _read_ref(String("d_proj_w")))
    var r_db = h.compare_host(g.d_proj_b.to_host(ctx), _read_ref(String("d_proj_b")))
    print("geglu d_x      vs torch:", r_dx)
    print("geglu d_proj_w vs torch:", r_dw)
    print("geglu d_proj_b vs torch:", r_db)
    all_pass = all_pass and r_dx.passed and r_dw.passed and r_db.passed

    print("")
    if all_pass:
        print("ALL SDXL GEGLU FWD+BWD GATES PASSED (cos >= 0.999 vs PyTorch tanh-GELU)")
    else:
        print("SDXL GEGLU PARITY FAILURE")
        raise Error("geglu_parity gate failed")
