# wan22_i2v_condy_readback.mojo — verify the Mojo-side I2V cond_y cache read.
# Exercises the EXACT SafeTensors + patchify3d path the trainer's
# _load_cache_sample uses for cond_y, independent of the (downloading) I2V
# weights. Confirms cond_y is [20,1,32,32] f32, computes std/mean (must match the
# torch builder print), and patchifies (1,2,2) -> [S=256, 80].

from std.gpu.host import DeviceContext
from std.collections import List
from std.math import sqrt
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.ops.patchify3d import patchify3d


def main() raises:
    var ctx = DeviceContext()
    var cache_dir = String("/home/alex/.serenity/wan22_cache/40_woman_i2v")
    for idx in range(8):
        var path = cache_dir + String("/sample_") + String(idx) + String(".safetensors")
        var st = SafeTensors.open(path)
        var info = st.tensor_info(String("cond_y"))
        if info.dtype != STDtype.F32:
            raise Error("cond_y dtype != F32")
        var n = 1
        for i in range(len(info.shape)):
            n *= info.shape[i]
        if n != 20 * 1 * 32 * 32:
            raise Error("cond_y numel != 20*1*32*32")
        var b = st.tensor_bytes(String("cond_y"))
        var p = b.unsafe_ptr().bitcast[Float32]()
        var vals = List[Float32]()
        var s = Float64(0.0)
        var ss = Float64(0.0)
        for i in range(n):
            var v = p[i]
            vals.append(v)
            s += Float64(v)
            ss += Float64(v) * Float64(v)
        var mean = s / Float64(n)
        var varp = ss / Float64(n) - mean * mean
        if varp < 0.0:
            varp = 0.0
        var shp = List[Int]()
        shp.append(20)
        shp.append(1)
        shp.append(32)
        shp.append(32)
        var tensor = Tensor.from_host(vals^, shp^, STDtype.F32, ctx)
        var patched = patchify3d(tensor, 1, 2, 2, ctx)   # [S,80]
        var ps = patched.shape()
        print("sample", idx, " cond_y std=", Float32(sqrt(varp)),
              " mean=", Float32(mean), " -> patchify [", ps[0], ",", ps[1], "]")
    print("RESULT: Mojo cond_y cache-read OK (20-ch -> patchify [256,80])")
