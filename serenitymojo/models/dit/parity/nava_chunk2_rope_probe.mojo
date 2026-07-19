# NAVA chunk 2 (video 3D rope): build_multiaxis_rope_tables([44,42,42],θ1e4) +
# rope_interleaved on q [1,320,24,128] vs torch rope_apply_3d (qv_rope).
# grid (f,h,w)=(5,8,8)=320 tokens, 24 heads, head_dim 128. Token order f-major.
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.rope_tables import build_multiaxis_rope_tables
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.cast import cast_tensor

comptime FX = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/nava_fx_chunk2_rope.safetensors"


def main() raises:
    var ctx = DeviceContext()
    var F = 5
    var H = 8
    var W = 8
    var NH = 24
    var L = F * H * W           # 320 tokens
    var rows = L * NH           # 7680 (rope broadcast across heads)
    # positions[r*3+a]: row r -> token = r//NH (f-major over f,h,w)
    var pos = List[Float32]()
    for r in range(rows):
        var tok = r // NH
        var f = tok // (H * W)
        var rem = tok % (H * W)
        var hh = rem // W
        var ww = rem % W
        pos.append(Float32(f))
        pos.append(Float32(hh))
        pos.append(Float32(ww))
    var positions = Tensor.from_host(pos^, [rows * 3], STDtype.F32, ctx)
    var axes = [44, 42, 42]     # full per-axis dims, sum=128=head_dim
    var cs = build_multiaxis_rope_tables(positions, axes, Float32(10000.0), ctx, STDtype.BF16)

    var fx = ShardedSafeTensors.open(FX)
    var qv = cast_tensor(Tensor.from_view(fx.tensor_view("qv"), ctx), STDtype.BF16, ctx)  # [1,320,24,128]
    var out = rope_interleaved(qv, cs[0], cs[1], ctx)
    var ref_host = Tensor.from_view(fx.tensor_view("qv_rope"), ctx).to_host(ctx)
    print("NAVA video 3D rope vs torch:", ParityHarness(0.999).compare(out, ref_host, ctx))
