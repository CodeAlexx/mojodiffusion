# ideogram4_sampler_gate.mojo — gate the serenity_trainer Ideogram-4 denoise driver
# (Ideogram4SampleLoop) vs the torch sampler oracle (ideogram4_fx_sampler.safetensors).
#
# Run (GPU free):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . -I trainer/src \
#       trainer/smoke/ideogram4_sampler_gate.mojo
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.cast import cast_tensor
from serenity_trainer.modelSampler.Ideogram4SampleLoop import (
    ideogram4_denoise,
    ideogram4_decode,
)

comptime COND = "/home/alex/.serenity/models/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors"
comptime UNCOND = "/home/alex/.serenity/models/ideogram-4-fp8/unconditional_transformer/diffusion_pytorch_model.safetensors"
comptime VAE = "/home/alex/.serenity/models/ideogram-4-fp8/vae/diffusion_pytorch_model.safetensors"
comptime FX = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_sampler.safetensors"
comptime LN = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_latentnorm.safetensors"

comptime NT = 15
comptime NIMG = 256
comptime TOTAL = 271
comptime GH = 16
comptime GW = 16
comptime STEPS = 8
comptime CFG = Float32(7.0)


def main() raises:
    var ctx = DeviceContext()
    var fx = ShardedSafeTensors.open(FX)
    var ln = ShardedSafeTensors.open(LN)

    var z = Tensor.from_view(fx.tensor_view("z0"), ctx)
    var llm = cast_tensor(Tensor.from_view(fx.tensor_view("llm_full"), ctx), STDtype.BF16, ctx)
    var pos = Tensor.from_view(fx.tensor_view("pos_f32"), ctx)
    var ind = Tensor.from_view(fx.tensor_view("ind_f32"), ctx)
    var npos = Tensor.from_view(fx.tensor_view("neg_pos_f32"), ctx)
    var nind = Tensor.from_view(fx.tensor_view("neg_ind_f32"), ctx)

    var zpad_h = List[Float32]()
    for _ in range(NT * 128):
        zpad_h.append(0.0)
    var text_zpad = Tensor.from_host(zpad_h^, [1, NT, 128], STDtype.F32, ctx)
    var nllm_h = List[Float32]()
    for _ in range(NIMG * 53248):
        nllm_h.append(0.0)
    var neg_llm = Tensor.from_host(nllm_h^, [1, NIMG, 53248], STDtype.BF16, ctx)

    var cond = ShardedSafeTensors.open(COND)
    var uncond = ShardedSafeTensors.open(UNCOND)

    var z_final = ideogram4_denoise[NT, NIMG, TOTAL](
        cond, uncond, z^, llm, neg_llm, pos, npos, ind, nind, text_zpad,
        STEPS, CFG, 256, 256, ctx,
    )
    var fz = Tensor.from_view(fx.tensor_view("final_z"), ctx).to_host(ctx)
    print("sampler final_z parity:", ParityHarness(0.999).compare(z_final, fz, ctx))

    var scale = Tensor.from_view(ln.tensor_view("latent_scale"), ctx)
    var shift = Tensor.from_view(ln.tensor_view("latent_shift"), ctx)
    var out = ideogram4_decode[GH, GW](z_final, scale, shift, VAE, ctx)
    var dec = Tensor.from_view(fx.tensor_view("decoded"), ctx).to_host(ctx)
    print("sampler decoded parity:", ParityHarness(0.999).compare(out.image, dec, ctx))
