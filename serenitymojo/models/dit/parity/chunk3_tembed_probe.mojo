# chunk3 parity: Ideogram-4 t-embedding (EmbedScalar->MLP) vs Wave-0 fixture.
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.models.dit.ideogram4_dit import (
    ideogram4_t_embedding, load_w_fp8, load_w_bf16,
)

comptime TRANSFORMER = "/home/alex/.serenity/models/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors"
comptime FIXTURE = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_transformer.safetensors"


def main() raises:
    var ctx = DeviceContext()
    var st = ShardedSafeTensors.open(TRANSFORMER)
    var mi_w = load_w_fp8(st, "t_embedding.mlp_in.weight", ctx)
    var mi_b = load_w_bf16(st, "t_embedding.mlp_in.bias", ctx)
    var mo_w = load_w_fp8(st, "t_embedding.mlp_out.weight", ctx)
    var mo_b = load_w_bf16(st, "t_embedding.mlp_out.bias", ctx)

    var fx = ShardedSafeTensors.open(FIXTURE)
    var t = Tensor.from_view(fx.tensor_view("chunk3.t_values"), ctx)
    var out = ideogram4_t_embedding(t, 4608, mi_w, mi_b, mo_w, mo_b, ctx)

    var exp_host = Tensor.from_view(fx.tensor_view("chunk3.t_embed"), ctx).to_host(ctx)
    var res = ParityHarness(0.999).compare(out, exp_host, ctx)
    print("chunk3 t-embedding parity:", res)
