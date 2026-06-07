# chunk1 parity probe: fp8 per-row dequant of layers.0.attention.qkv.weight
# vs the Wave-0 oracle fixture. Gate: cos >= 0.999.
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.fp8 import load_fp8_dequant

comptime TRANSFORMER = "/home/alex/.serenity/models/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors"
comptime FIXTURE = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_transformer.safetensors"


def main() raises:
    var ctx = DeviceContext()
    var st = ShardedSafeTensors.open(TRANSFORMER)
    var my = load_fp8_dequant(st, "layers.0.attention.qkv.weight", ctx)
    print("dequant shape:", my.shape()[0], "x", my.shape()[1], "dtype", my.dtype().name())

    var fx = ShardedSafeTensors.open(FIXTURE)
    var exp_view = fx.tensor_view("chunk1.qkv_dequant_expected")
    var expected = Tensor.from_view(exp_view, ctx)
    var exp_host = expected.to_host(ctx)

    var res = ParityHarness(0.999).compare(my, exp_host, ctx)
    print("chunk1 fp8 per-row dequant parity:", res)
