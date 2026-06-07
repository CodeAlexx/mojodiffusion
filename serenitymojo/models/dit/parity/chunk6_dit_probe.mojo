# chunk6 parity: full 34-layer Ideogram-4 DiT velocity vs Wave-0 fixture.
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.dit.ideogram4_dit import ideogram4_forward
from serenitymojo.models.dit.ideogram4_mrope import build_ideogram4_mrope

comptime T = "/home/alex/.serenity/models/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors"
comptime FX = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_transformer.safetensors"
comptime FIN = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_inputs_f32.safetensors"


def main() raises:
    var ctx = DeviceContext()
    var st = ShardedSafeTensors.open(T)
    var fx = ShardedSafeTensors.open(FX)
    var fin = ShardedSafeTensors.open(FIN)

    var x = cast_tensor(Tensor.from_view(fx.tensor_view("chunk6.in_x"), ctx), STDtype.BF16, ctx)
    var llm = cast_tensor(Tensor.from_view(fx.tensor_view("chunk6.in_llm"), ctx), STDtype.BF16, ctx)
    var t = Tensor.from_view(fx.tensor_view("chunk6.in_t"), ctx)
    var ind = Tensor.from_view(fin.tensor_view("chunk6.in_indicator_f32"), ctx)
    var pos = Tensor.from_view(fin.tensor_view("chunk6.in_position_ids_f32"), ctx)
    var sec = [24, 20, 20]
    var cs = build_ideogram4_mrope(pos, 256, sec, Float32(5000000.0), ctx, STDtype.BF16)

    var out = ideogram4_forward[260](st, x, llm, t, ind, cs[0], cs[1], 34, 18, 256, 4608, ctx)
    var exp_host = Tensor.from_view(fx.tensor_view("chunk6.out_velocity"), ctx).to_host(ctx)
    print("chunk6 full DiT velocity parity:", ParityHarness(0.999).compare(out, exp_host, ctx))
