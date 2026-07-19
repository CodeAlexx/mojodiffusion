# ltx2_upsampler_smoke.mojo — unit gate for the LTX-2 LatentUpsampler.
#
# Loads spatial-x2 and temporal-x2 weights, runs the BARE upsampler forward on
# the small fixed latents dumped by parity/ref_dump.py, and reports cosine
# similarity vs the BF16 PyTorch reference output. Gate target: cos >= 0.999.
#
# Reference .bin layout is NCDHW (PyTorch). conv3d/group_norm here run NDHWC,
# so we transpose in on load and out before comparison.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.math import sqrt
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.models.upsampler.ltx2_upsampler import LatentUpsampler

comptime MODEL_DTYPE = STDtype.BF16
comptime PD = "/home/alex/mojodiffusion/serenitymojo/models/upsampler/parity"
comptime SPATIAL_W = "/home/alex/.serenity/models/ltx2_upscalers/ltx-2-spatial-upscaler-x2-1.0.safetensors"
comptime TEMPORAL_W = "/home/alex/.serenity/models/ltx2_upscalers/ltx-2-temporal-upscaler-x2-1.0.safetensors"


def _read_f32_bin(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open ") + path)
    var n = file_size(fd)
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


# NCDHW host list -> NDHWC host list.
def _ncdhw_to_ndhwc(x: List[Float32], N: Int, C: Int, D: Int, H: Int, W: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(len(x)):
        out.append(0.0)
    for n in range(N):
        for c in range(C):
            for d in range(D):
                for h in range(H):
                    for w in range(W):
                        var src = ((((n * C + c) * D + d) * H + h) * W + w)
                        var dst = ((((n * D + d) * H + h) * W + w) * C + c)
                        out[dst] = x[src]
    return out^


# NDHWC host list -> NCDHW host list.
def _ndhwc_to_ncdhw(x: List[Float32], N: Int, D: Int, H: Int, W: Int, C: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(len(x)):
        out.append(0.0)
    for n in range(N):
        for d in range(D):
            for h in range(H):
                for w in range(W):
                    for c in range(C):
                        var src = ((((n * D + d) * H + h) * W + w) * C + c)
                        var dst = ((((n * C + c) * D + d) * H + h) * W + w)
                        out[dst] = x[src]
    return out^


def _cosine(a: List[Float32], b: List[Float32]) raises -> Float32:
    if len(a) != len(b):
        raise Error(String("cosine: length mismatch ") + String(len(a)) + " vs " + String(len(b)))
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(len(a)):
        var av = Float64(a[i])
        var bv = Float64(b[i])
        dot += av * bv
        na += av * av
        nb += bv * bv
    if na == 0.0 or nb == 0.0:
        return Float32(0.0)
    return Float32(dot / (sqrt(na) * sqrt(nb)))


def _max_abs_err(a: List[Float32], b: List[Float32]) -> Float32:
    var m = Float32(0.0)
    for i in range(len(a)):
        var e = a[i] - b[i]
        if e < 0.0:
            e = -e
        if e > m:
            m = e
    return m


def _run_one(
    tag: String, wpath: String, is_temporal: Bool,
    N: Int, C: Int, Din: Int, Hin: Int, Win: Int,
    ctx: DeviceContext,
) raises -> Float32:
    print("---- ", tag, " ----")
    var st = ShardedSafeTensors.open(wpath)
    var model = LatentUpsampler(st, is_temporal, ctx)

    var lat_ncdhw = _read_f32_bin(PD + "/" + tag + "_in.bin")
    var lat_ndhwc = _ncdhw_to_ndhwc(lat_ncdhw, N, C, Din, Hin, Win)
    var in_shape = List[Int]()
    in_shape.append(N); in_shape.append(Din); in_shape.append(Hin); in_shape.append(Win); in_shape.append(C)
    var lat_t = Tensor.from_host(lat_ndhwc, in_shape^, MODEL_DTYPE, ctx)

    var out_t = model.forward(lat_t, ctx)
    var osh = out_t.shape()
    var Dout = osh[1]; var Hout = osh[2]; var Wout = osh[3]
    print("    out NDHWC shape: [", osh[0], ",", Dout, ",", Hout, ",", Wout, ",", osh[4], "]")
    var out_host = out_t.to_host(ctx)
    var out_ncdhw = _ndhwc_to_ncdhw(out_host, N, Dout, Hout, Wout, C)

    var refv = _read_f32_bin(PD + "/" + tag + "_out.bin")
    var cos = _cosine(out_ncdhw, refv)
    var mae = _max_abs_err(out_ncdhw, refv)
    print("    NCDHW out shape: [", N, ",", C, ",", Dout, ",", Hout, ",", Wout, "]")
    print("    cosine =", cos, "  max_abs_err =", mae, "  n =", len(refv))
    return cos


def main() raises:
    var ctx = DeviceContext()
    # spatial: in [1,128,2,8,8] -> out [1,128,2,16,16]
    var c_sp = _run_one("spatial", SPATIAL_W, False, 1, 128, 2, 8, 8, ctx)
    # temporal: in [1,128,3,6,6] -> out [1,128,5,6,6]
    var c_tp = _run_one("temporal", TEMPORAL_W, True, 1, 128, 3, 6, 6, ctx)
    print("================ GATE ================")
    print("spatial  cos =", c_sp, "  PASS=", c_sp >= 0.999)
    print("temporal cos =", c_tp, "  PASS=", c_tp >= 0.999)
    if c_sp >= 0.999 and c_tp >= 0.999:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
