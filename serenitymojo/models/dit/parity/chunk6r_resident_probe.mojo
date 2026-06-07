# gate: resident fp8 forward (forward_r) vs chunk6 fixture (same as ideogram4_forward).
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.dit.ideogram4_resident import Ideogram4Weights, ideogram4_forward_r, ideogram4_build_masks
from serenitymojo.models.dit.ideogram4_mrope import build_ideogram4_mrope

comptime T = "/home/alex/.serenity/models/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors"
comptime FX = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_transformer.safetensors"
comptime FIN = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_inputs_f32.safetensors"


def main() raises:
    var ctx = DeviceContext()
    var w = Ideogram4Weights.load(ShardedSafeTensors.open(T), ctx)
    var fx = ShardedSafeTensors.open(FX)
    var fin = ShardedSafeTensors.open(FIN)
    var x = cast_tensor(Tensor.from_view(fx.tensor_view("chunk6.in_x"), ctx), STDtype.BF16, ctx)
    var llm = cast_tensor(Tensor.from_view(fx.tensor_view("chunk6.in_llm"), ctx), STDtype.BF16, ctx)
    var t = Tensor.from_view(fx.tensor_view("chunk6.in_t"), ctx)
    var ind = Tensor.from_view(fin.tensor_view("chunk6.in_indicator_f32"), ctx)
    var pos = Tensor.from_view(fin.tensor_view("chunk6.in_position_ids_f32"), ctx)
    var sec = [24, 20, 20]
    var cs = build_ideogram4_mrope(pos, 256, sec, Float32(5000000.0), ctx, STDtype.BF16)
    var masks = ideogram4_build_masks(ind, ctx)
    var out = ideogram4_forward_r[260](w, x, llm, t, masks, cs[0], cs[1], 34, 18, 256, 4608, ctx)
    var exp_host = Tensor.from_view(fx.tensor_view("chunk6.out_velocity"), ctx).to_host(ctx)
    print("resident fp8 DiT vs fixture:", ParityHarness(0.999).compare(out, exp_host, ctx))
