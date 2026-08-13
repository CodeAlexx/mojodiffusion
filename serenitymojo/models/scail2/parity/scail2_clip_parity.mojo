# Deterministic SCAIL-2 CLIP visual block parity against scail2_clip_oracle.py.
# This is a development gate; production math is scail2_clip_vision.mojo.

from max.gpu.host import DeviceContext
from std.math import sqrt
from std.sys import argv

from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.scail2.scail2_clip_vision import scail2_clip_block


comptime S = 5
comptime D = 8
comptime H = 2
comptime DH = 4
comptime FF = 16


def _load(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_view(st.tensor_view(name), ctx)


def _abs(x: Float32) -> Float32:
    return x if x >= 0.0 else -x


def main() raises:
    var args = argv()
    if len(args) != 2:
        raise Error("usage: scail2_clip_parity <scail2_clip_fixture.safetensors>")
    var st = ShardedSafeTensors.open(args[1])
    var ctx = DeviceContext()
    var out = scail2_clip_block[S, D, H, DH, FF](
        _load(st, String("hidden"), ctx),
        _load(st, String("norm1_w"), ctx),
        _load(st, String("norm1_b"), ctx),
        _load(st, String("qkv_w"), ctx),
        _load(st, String("qkv_b"), ctx),
        _load(st, String("proj_w"), ctx),
        _load(st, String("proj_b"), ctx),
        _load(st, String("norm2_w"), ctx),
        _load(st, String("norm2_b"), ctx),
        _load(st, String("fc1_w"), ctx),
        _load(st, String("fc1_b"), ctx),
        _load(st, String("fc2_w"), ctx),
        _load(st, String("fc2_b"), ctx),
        Float32(1.0e-5), ctx,
    )
    var actual = out.to_host(ctx)
    var expected = _load(st, String("expected_output"), ctx).to_host(ctx)
    if len(actual) != len(expected):
        raise Error("SCAIL-2 CLIP parity length mismatch")
    var dot = Float32(0.0)
    var aa = Float32(0.0)
    var bb = Float32(0.0)
    var max_abs = Float32(0.0)
    for i in range(len(actual)):
        var a = actual[i]
        var b = expected[i]
        var delta = _abs(a - b)
        if delta > max_abs:
            max_abs = delta
        dot += a * b
        aa += a * a
        bb += b * b
    var cosine = dot / (sqrt(aa) * sqrt(bb))
    print("SCAIL-2 CLIP synthetic block: cos=", cosine, " max_abs=", max_abs)
    if cosine < Float32(0.9999) or max_abs > Float32(2.0e-4):
        raise Error("SCAIL-2 CLIP synthetic block parity FAIL")
    print("GATE PASS SCAIL-2 CLIP block cos>=0.9999 max_abs<=2e-4")
