# Full streamed 40-block Bernini-R first-forward parity for one expert.
# argv: bernini_r_forward_parity <fp8_cache_dir> <transformer|transformer_2>

from std.collections import List
from max.gpu.host import DeviceContext
from std.memory import alloc
from std.sys import argv

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import O_RDONLY, file_size, sys_close, sys_open, sys_pread
from serenitymojo.models.wan22.wan22_a14b_streamed_dit import Wan22A14BStreamedDiT
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.tensor import Tensor


comptime REF = "/home/alex/mojodiffusion-sync/output/checks/bernini_r/forward_oracle/"
comptime S = 8
comptime TXT = 512
comptime CTXL = 512
comptime VALID = 8
comptime DIM = 4096


def _read_f32(component: String, name: String) raises -> List[Float32]:
    var path = String(REF) + component + String("/") + name + String(".bin")
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open Bernini forward fixture: ") + path)
    var nbytes = file_size(fd)
    if nbytes <= 0 or nbytes % 4 != 0:
        _ = sys_close(fd)
        raise Error(String("invalid Bernini forward fixture: ") + path)
    var buf = alloc[UInt8](nbytes)
    var done = 0
    while done < nbytes:
        var got = sys_pread(fd, buf + done, nbytes - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(nbytes // 4):
        out.append(fp[i])
    buf.free()
    return out^


def _metrics(actual: List[Float32], reference: List[Float32]) raises -> Tuple[Float64, Float64]:
    if len(actual) != len(reference):
        raise Error("Bernini full-forward parity length mismatch")
    var dot = 0.0
    var aa = 0.0
    var rr = 0.0
    var max_abs = 0.0
    for i in range(len(actual)):
        var a = Float64(actual[i])
        var r = Float64(reference[i])
        dot += a * r
        aa += a * a
        rr += r * r
        var d = a - r
        if d < 0.0:
            d = -d
        if d > max_abs:
            max_abs = d
    return (dot / (aa * rr) ** 0.5, max_abs)


def main() raises:
    var args = argv()
    if len(args) < 3:
        raise Error(
            "usage: bernini_r_forward_parity <fp8_cache_dir> <transformer|transformer_2>"
        )
    var cache = String(args[1])
    var component = String(args[2])
    var timestep = Float32(999.0)
    if component == "transformer_2":
        timestep = 500.0
    elif component != "transformer":
        raise Error("Bernini component must be transformer or transformer_2")

    var ctx = DeviceContext()
    var latent = Tensor.from_host(
        _read_f32(component, String("latent")), [16, 1, 4, 8], STDtype.BF16, ctx
    )
    var context = Tensor.from_host(
        _read_f32(component, String("context")), [CTXL, DIM], STDtype.BF16, ctx
    )
    var model = Wan22A14BStreamedDiT.open(cache, ctx)
    var pair = model.forward_cfg_pair[1, 2, 4, S, TXT, CTXL, 40, 128](
        latent, timestep, context, context, VALID, VALID, ctx
    )
    var cond = cast_tensor(pair.cond, STDtype.F32, ctx).to_host(ctx)
    var uncond = cast_tensor(pair.uncond, STDtype.F32, ctx).to_host(ctx)
    var oracle = _read_f32(component, String("output"))
    var cond_metrics = _metrics(cond, oracle)
    var pair_metrics = _metrics(cond, uncond)
    print("Bernini", component, "full streamed first-forward:")
    print("  cache vs creator cos=", cond_metrics[0], " max_abs=", cond_metrics[1])
    print("  paired identical-context cos=", pair_metrics[0], " max_abs=", pair_metrics[1])
    if cond_metrics[0] < 0.99 or pair_metrics[0] < 0.999999:
        raise Error("Bernini-R full streamed first-forward parity FAIL")
    print("GATE PASS", component, "fullForwardCos=", cond_metrics[0])
