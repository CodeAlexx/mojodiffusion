# chunk5 parity: one Ideogram-4 transformer block (layer 0) vs Wave-0 fixture.
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.dit.ideogram4_dit import ideogram4_block, load_w_fp8, load_w_bf16
from serenitymojo.models.dit.ideogram4_mrope import build_ideogram4_mrope

comptime T = "/home/alex/.serenity/models/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors"
comptime FX = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_transformer.safetensors"
comptime FIN = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_inputs_f32.safetensors"


def main() raises:
    var ctx = DeviceContext()
    var st = ShardedSafeTensors.open(T)
    var p = String("layers.0.")
    var amw = load_w_fp8(st, p + "adaln_modulation.weight", ctx)
    var amb = load_w_bf16(st, p + "adaln_modulation.bias", ctx)
    var an1 = load_w_bf16(st, p + "attention_norm1.weight", ctx)
    var an2 = load_w_bf16(st, p + "attention_norm2.weight", ctx)
    var fn1 = load_w_bf16(st, p + "ffn_norm1.weight", ctx)
    var fn2 = load_w_bf16(st, p + "ffn_norm2.weight", ctx)
    var qkv = load_w_fp8(st, p + "attention.qkv.weight", ctx)
    var ow = load_w_fp8(st, p + "attention.o.weight", ctx)
    var nq = load_w_bf16(st, p + "attention.norm_q.weight", ctx)
    var nk = load_w_bf16(st, p + "attention.norm_k.weight", ctx)
    var w1 = load_w_fp8(st, p + "feed_forward.w1.weight", ctx)
    var w2 = load_w_fp8(st, p + "feed_forward.w2.weight", ctx)
    var w3 = load_w_fp8(st, p + "feed_forward.w3.weight", ctx)

    var fx = ShardedSafeTensors.open(FX)
    var fin = ShardedSafeTensors.open(FIN)
    var x = cast_tensor(Tensor.from_view(fx.tensor_view("chunk5.block0_in_x"), ctx), STDtype.BF16, ctx)
    var ad = cast_tensor(Tensor.from_view(fx.tensor_view("chunk5.adaln_input"), ctx), STDtype.BF16, ctx)
    var pos = Tensor.from_view(fin.tensor_view("chunk6.in_position_ids_f32"), ctx)
    var sec = [24, 20, 20]
    var cs = build_ideogram4_mrope(pos, 256, sec, Float32(5000000.0), ctx, STDtype.BF16)

    var out = ideogram4_block[260](
        x, ad, cs[0], cs[1], amw, amb, an1, an2, fn1, fn2,
        qkv, ow, nq, nk, w1, w2, w3, 18, 256, 4608, ctx,
    )
    var exp_host = Tensor.from_view(fx.tensor_view("chunk5.block0_out"), ctx).to_host(ctx)
    print("chunk5 block0 parity:", ParityHarness(0.999).compare(out, exp_host, ctx))
