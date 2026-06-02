# cat_nhwc_bwd_parity.mojo — GPU gate for the NHWC channel-axis (axis=3) cat
# backward, exercising the SDXL decoder skip-concat path. cat_backward already
# exists (shape_backward.mojo:310); this gates the channel-concat NHWC case
# (TRAINING_PLAN_sdxl.md:90) against torch autograd.
#
# Forward: y = concat(axis=3, h[N,H,W,C0], skip[N,H,W,C1]). Backward splits
# d_y -> (d_h, d_skip). Verifies the split lands the right channel ranges.
#
# Run the oracle first, then:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/ops/parity/cat_nhwc_bwd_oracle.py
#   pixi run mojo build -I . -Xlinker -lm serenitymojo/ops/parity/cat_nhwc_bwd_parity.mojo -o /tmp/cat_nhwc
#   /tmp/cat_nhwc

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.ops.shape_backward import cat_backward
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc


comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/ops/parity/cat_nhwc_bwd_ref.txt"
)

comptime N = 2
comptime H = 4
comptime W = 4
comptime C0 = 6
comptime C1 = 10


def _fill(n: Int, a: Int, b: Int, c: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * a) % b) - c) * 0.05)
    return out^

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

    comptime C = C0 + C1
    var gy = Tensor.from_host(
        _fill(N * H * W * C, 3, 9, 4.0), _sh4(N, H, W, C), STDtype.F32, ctx
    )

    var g = cat_backward(gy, C0, C1, 3, ctx)   # axis=3 = NHWC channel axis
    var r_dh = h.compare_host(g.d_0.to_host(ctx), _read_ref(String("d_h")))
    var r_ds = h.compare_host(g.d_1.to_host(ctx), _read_ref(String("d_skip")))
    print("cat-nhwc d_h    vs torch:", r_dh)
    print("cat-nhwc d_skip vs torch:", r_ds)
    all_pass = all_pass and r_dh.passed and r_ds.passed

    print("")
    if all_pass:
        print("CAT NHWC CHANNEL-AXIS BACKWARD GATE PASSED (cos >= 0.999 vs PyTorch)")
    else:
        print("CAT NHWC BACKWARD PARITY FAILURE")
        raise Error("cat_nhwc_bwd_parity gate failed")
