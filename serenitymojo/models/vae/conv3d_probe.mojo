# conv3d_probe.mojo — standalone parity for the NEW conv3d op vs torch.
#
# Reads parity/c3d_{x,w,b,y}.bin (from gen_conv3d_oracle.py) and runs the Mojo
# conv3d, comparing the NDHWC output to torch's F.conv3d (cos + max_abs). F32
# storage so the only deltas are GPU FMA-order vs torch — gate cos >= 0.999.
#
# Run: pixi run mojo run -I . serenitymojo/models/vae/conv3d_probe.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.models.vae.conv3d import conv3d


comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/vae/parity"


def _read_f32_bin(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open ") + path)
    var n = file_size(fd)
    if n <= 0 or n % 4 != 0:
        _ = sys_close(fd)
        raise Error(String("bad bin size for ") + path)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var count = n // 4
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(count):
        out.append(fp[i])
    buf.free()
    return out^


def main() raises:
    var ctx = DeviceContext()
    # Shapes hard-coded to match gen_conv3d_oracle.py
    comptime N = 1
    comptime D = 3
    comptime H = 5
    comptime W = 6
    comptime Cin = 4
    comptime Cout = 8
    comptime Kd = 3
    comptime Kh = 3
    comptime Kw = 3

    var xv = _read_f32_bin(String(PARITY_DIR) + "/c3d_x.bin")
    var xs = List[Int]()
    xs.append(N); xs.append(D); xs.append(H); xs.append(W); xs.append(Cin)
    var x = Tensor.from_host(xv, xs^, STDtype.F32, ctx)

    var wv = _read_f32_bin(String(PARITY_DIR) + "/c3d_w.bin")
    var ws = List[Int]()
    ws.append(Kd); ws.append(Kh); ws.append(Kw); ws.append(Cin); ws.append(Cout)
    var w = Tensor.from_host(wv, ws^, STDtype.F32, ctx)

    var bv = _read_f32_bin(String(PARITY_DIR) + "/c3d_b.bin")
    var bs = List[Int]()
    bs.append(Cout)
    var b = Tensor.from_host(bv, bs^, STDtype.F32, ctx)

    var y = conv3d(x, w, Optional[Tensor](b^), 1, 1, 1, 1, 1, 1, ctx)
    var ysh = y.shape()
    print("[conv3d] out shape:", ysh[0], ysh[1], ysh[2], ysh[3], ysh[4])

    var refv = _read_f32_bin(String(PARITY_DIR) + "/c3d_y.bin")
    var harness = ParityHarness(0.999)
    var res = harness.compare(y, refv, ctx)
    print("[conv3d] PARITY:", res)
