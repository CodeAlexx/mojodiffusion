# NAVA chunk 2 (audio 1D rope): rotate first 22 complex dims (44 real) of q
# [1,34,24,128], passthrough last 84; positions = token*0.24 (temporal scaling),
# axes_dims=[44], θ1e4. vs torch rope_apply_1d (qa_rope).
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.rope_tables import build_multiaxis_rope_tables
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.tensor_algebra import slice, concat
from serenitymojo.ops.cast import cast_tensor

comptime FX = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/nava_fx_chunk2_rope.safetensors"


def main() raises:
    var ctx = DeviceContext()
    var L = 34
    var NH = 24
    var rows = L * NH
    var pos = List[Float32]()
    for r in range(rows):
        pos.append(Float32(r // NH) * Float32(0.24))   # token*scaling, single axis
    var positions = Tensor.from_host(pos^, [rows], STDtype.F32, ctx)
    var axes = [44]            # rotate 44 real dims (22 complex)
    var cs = build_multiaxis_rope_tables(positions, axes, Float32(10000.0), ctx, STDtype.BF16)

    var fx = ShardedSafeTensors.open(FX)
    var qa = cast_tensor(Tensor.from_view(fx.tensor_view("qa"), ctx), STDtype.BF16, ctx)  # [1,34,24,128]
    var qa_rot = slice(qa, 3, 0, 44, ctx)       # first 44 dims -> rope
    var qa_pass = slice(qa, 3, 44, 84, ctx)     # last 84 dims -> passthrough
    var rotd = rope_interleaved(qa_rot, cs[0], cs[1], ctx)
    var out = concat(3, ctx, rotd, qa_pass)
    var ref_host = Tensor.from_view(fx.tensor_view("qa_rope"), ctx).to_host(ctx)
    print("NAVA audio 1D rope vs torch:", ParityHarness(0.999).compare(out, ref_host, ctx))
