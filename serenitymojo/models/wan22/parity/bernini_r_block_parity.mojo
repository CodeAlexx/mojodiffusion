# Bernini-R real block-0 parity: creator BF16 oracle, official BF16 weights,
# and the persistent Mojo E4M3 cache must all agree before product admission.
#
# Oracle first: scripts/bernini_r_block_oracle.py
# argv: bernini_r_block_parity <official_transformer_dir> <fp8_cache_dir>

from std.collections import Dict, List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer, alloc
from std.sys import argv

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import O_RDONLY, file_size, sys_close, sys_open, sys_pread
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.dit.wan22_dit import (
    Wan22Config,
    wan22_block_forward,
    wan22_canonical_weight_key,
)
from serenitymojo.models.wan22.wan22_fp8_stream import Wan22A14BFP8Stream
from serenitymojo.tensor import Tensor


comptime REF = "/home/alex/mojodiffusion-sync/output/checks/bernini_r/block_oracle/"
comptime S = 8
comptime TXT = 8
comptime H = 40
comptime DH = 128
comptime DIM = 5120
comptime TArc = ArcPointer[Tensor]


def _read_f32(name: String) raises -> List[Float32]:
    var path = String(REF) + name + String(".bin")
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open Bernini block fixture: ") + path)
    var nbytes = file_size(fd)
    if nbytes <= 0 or nbytes % 4 != 0:
        _ = sys_close(fd)
        raise Error(String("invalid Bernini block fixture: ") + path)
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


def _source_block(source_dir: String, ctx: DeviceContext) raises -> Dict[String, TArc]:
    var source = ShardedSafeTensors.open(source_dir)
    var out = Dict[String, TArc]()
    for ref raw_name in source.names():
        var name = String(raw_name)
        if not name.startswith("blocks.0."):
            continue
        var canonical = wan22_canonical_weight_key(name)
        var short_name = canonical.replace("blocks.0.", "")
        out[short_name] = TArc(Tensor.from_view_as_bf16(source.tensor_view(name), ctx))
    if len(out) != 27:
        raise Error(String("Bernini source block tensor count != 27: ") + String(len(out)))
    return out^


def _run_source(
    source_dir: String, x: Tensor, e0: Tensor, context: Tensor,
    cos: Tensor, sin: Tensor, ctx: DeviceContext,
) raises -> List[Float32]:
    var block = _source_block(source_dir, ctx)
    var out = wan22_block_forward[S, TXT, H, DH](
        x, e0, context, cos, sin, block, Wan22Config.t2v_a14b(), ctx
    )
    return out.to_host(ctx)


def _run_cache(
    cache_dir: String, x: Tensor, e0: Tensor, context: Tensor,
    cos: Tensor, sin: Tensor, ctx: DeviceContext,
) raises -> List[Float32]:
    var stream = Wan22A14BFP8Stream.open(cache_dir)
    var block = stream.load_block_bf16(0, ctx)
    var out = wan22_block_forward[S, TXT, H, DH](
        x, e0, context, cos, sin, block, Wan22Config.t2v_a14b(), ctx
    )
    return out.to_host(ctx)


def _metrics(actual: List[Float32], reference: List[Float32]) raises -> Tuple[Float64, Float64]:
    if len(actual) != len(reference):
        raise Error("Bernini block parity length mismatch")
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
    var cos = dot / (aa * rr) ** 0.5
    return (cos, max_abs)


def main() raises:
    var args = argv()
    if len(args) < 3:
        raise Error(
            "usage: bernini_r_block_parity <official_transformer_dir> <fp8_cache_dir>"
        )
    var ctx = DeviceContext()
    var x = Tensor.from_host(_read_f32("x"), [1, S, DIM], STDtype.BF16, ctx)
    var e0 = Tensor.from_host(_read_f32("temb"), [1, S, 6, DIM], STDtype.F32, ctx)
    var context = Tensor.from_host(_read_f32("context"), [1, TXT, DIM], STDtype.BF16, ctx)
    var cos = Tensor.from_host(_read_f32("cos"), [S, DH // 2], STDtype.BF16, ctx)
    var sin = Tensor.from_host(_read_f32("sin"), [S, DH // 2], STDtype.BF16, ctx)
    var oracle = _read_f32("output")

    var source_out = _run_source(
        String(args[1]), x, e0, context, cos, sin, ctx
    )
    var source_metrics = _metrics(source_out, oracle)
    print("  official BF16 vs creator: cos=", source_metrics[0], " max_abs=", source_metrics[1])

    var cache_out = _run_cache(
        String(args[2]), x, e0, context, cos, sin, ctx
    )
    var cache_metrics = _metrics(cache_out, oracle)
    var cache_source = _metrics(cache_out, source_out)
    print("  E4M3 cache vs creator: cos=", cache_metrics[0], " max_abs=", cache_metrics[1])
    print("  E4M3 cache vs BF16: cos=", cache_source[0], " max_abs=", cache_source[1])
    if (
        source_metrics[0] < 0.999 or cache_metrics[0] < 0.99
        or cache_source[0] < 0.99
    ):
        raise Error("Bernini-R real block parity FAIL")
    print("GATE PASS Bernini-R block BF16>=.999 and E4M3>=.99")
