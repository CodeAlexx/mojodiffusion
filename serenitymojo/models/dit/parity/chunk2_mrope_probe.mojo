# chunk2 parity: Ideogram-4 MRoPE cos/sin vs Wave-0 fixture. Gate cos>=0.999.
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.models.dit.ideogram4_mrope import build_ideogram4_mrope

comptime FIXTURE = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_transformer.safetensors"
comptime FIXIN = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_inputs_f32.safetensors"


def main() raises:
    var ctx = DeviceContext()
    var fx = ShardedSafeTensors.open(FIXTURE)
    var fin = ShardedSafeTensors.open(FIXIN)

    var pos = Tensor.from_view(fin.tensor_view("chunk6.in_position_ids_f32"), ctx)
    var sec = [24, 20, 20]
    var cs = build_ideogram4_mrope(pos, 256, sec, Float32(5000000.0), ctx, STDtype.F32)

    var exp_cos = Tensor.from_view(fx.tensor_view("chunk2.mrope_cos"), ctx).to_host(ctx)
    var exp_sin = Tensor.from_view(fx.tensor_view("chunk2.mrope_sin"), ctx).to_host(ctx)

    var rc = ParityHarness(0.999).compare(cs[0], exp_cos, ctx)
    var rs = ParityHarness(0.999).compare(cs[1], exp_sin, ctx)
    print("chunk2 mrope cos:", rc)
    print("chunk2 mrope sin:", rs)
