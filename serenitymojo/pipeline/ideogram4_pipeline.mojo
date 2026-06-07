# pipeline/ideogram4_pipeline.mojo — Ideogram-4 end-to-end sampler (chunk 9).
# 1:1 port of pipeline_ideogram4.__call__ denoise + _decode (587-637).
# Consumes the Wave-0 sampler fixture (dumped z0 + llm_features + packed inputs,
# decoupling RNG/tokenizer/Qwen — all gated separately) and runs the native
# CFG denoise (cond + uncond transformers, logit-normal schedule, Euler) ->
# latent denorm -> Ideogram unpatch -> Flux2 VAE decode -> PNG, gating final_z /
# final_latent / decoded vs the torch reference.
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import mul, add, mul_scalar, reshape, permute, slice, concat
from serenitymojo.image.png import save_png
from serenitymojo.models.dit.ideogram4_dit import ideogram4_forward
from serenitymojo.models.dit.ideogram4_mrope import build_ideogram4_mrope
from serenitymojo.models.vae.ldm_decoder import load_ideogram4_vae_decoder
from serenitymojo.sampling.ideogram4_schedule import (
    ideogram4_logitnormal, ideogram4_schedule_mean, make_step_intervals,
)

comptime COND = "/home/alex/.serenity/models/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors"
comptime UNCOND = "/home/alex/.serenity/models/ideogram-4-fp8/unconditional_transformer/diffusion_pytorch_model.safetensors"
comptime VAE = "/home/alex/.serenity/models/ideogram-4-fp8/vae/diffusion_pytorch_model.safetensors"
comptime FX = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_sampler.safetensors"
comptime LN = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_latentnorm.safetensors"

# fixed by the fixture (256x256, prompt seq 15): nt=15, nimg=256, total=271, gh=gw=16
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

    var z = Tensor.from_view(fx.tensor_view("z0"), ctx)               # [1,NIMG,128] F32
    var llm = cast_tensor(Tensor.from_view(fx.tensor_view("llm_full"), ctx), STDtype.BF16, ctx)  # [1,TOTAL,53248]
    var pos = Tensor.from_view(fx.tensor_view("pos_f32"), ctx)        # [1,TOTAL,3]
    var ind = Tensor.from_view(fx.tensor_view("ind_f32"), ctx)        # [1,TOTAL]
    var npos = Tensor.from_view(fx.tensor_view("neg_pos_f32"), ctx)   # [1,NIMG,3]
    var nind = Tensor.from_view(fx.tensor_view("neg_ind_f32"), ctx)   # [1,NIMG]

    var sec = [24, 20, 20]
    var cs = build_ideogram4_mrope(pos, 256, sec, Float32(5000000.0), ctx, STDtype.BF16)
    var ncs = build_ideogram4_mrope(npos, 256, sec, Float32(5000000.0), ctx, STDtype.BF16)

    var cond = ShardedSafeTensors.open(COND)
    var uncond = ShardedSafeTensors.open(UNCOND)

    # zeros: text z-padding [1,NT,128] F32; neg llm [1,NIMG,53248] bf16
    var zpad_h = List[Float32]()
    for _ in range(NT * 128):
        zpad_h.append(0.0)
    var text_zpad = Tensor.from_host(zpad_h^, [1, NT, 128], STDtype.F32, ctx)
    var nllm_h = List[Float32]()
    for _ in range(NIMG * 53248):
        nllm_h.append(0.0)
    var neg_llm = Tensor.from_host(nllm_h^, [1, NIMG, 53248], STDtype.BF16, ctx)

    var mean = ideogram4_schedule_mean(256, 256, 0.5)
    var si = make_step_intervals(STEPS)

    for step in range(STEPS - 1, -1, -1):
        var t_val = ideogram4_logitnormal(Float64(si[step + 1]), mean)
        var s_val = ideogram4_logitnormal(Float64(si[step]), mean)
        var tv = [t_val]
        var t = Tensor.from_host(tv^, [1], STDtype.F32, ctx)

        var pos_z = cast_tensor(concat(1, ctx, text_zpad, z), STDtype.BF16, ctx)  # [1,TOTAL,128]
        var cout = ideogram4_forward[TOTAL](cond, pos_z, llm, t, ind, cs[0], cs[1], 34, 18, 256, 4608, ctx)
        var pos_v = slice(cout, 1, NT, NIMG, ctx)                                  # [1,NIMG,128] F32

        var z_bf = cast_tensor(z, STDtype.BF16, ctx)
        var t2 = Tensor.from_host([t_val], [1], STDtype.F32, ctx)
        var nout = ideogram4_forward[NIMG](uncond, z_bf, neg_llm, t2, nind, ncs[0], ncs[1], 34, 18, 256, 4608, ctx)

        var v = add(mul_scalar(pos_v, CFG, ctx), mul_scalar(nout, Float32(1.0) - CFG, ctx), ctx)
        z = add(z, mul_scalar(v, s_val - t_val, ctx), ctx)
        print("  step", step, "t", t_val, "s", s_val)

    # gate final_z
    var fz_host = Tensor.from_view(fx.tensor_view("final_z"), ctx).to_host(ctx)
    print("chunk9 final_z parity:", ParityHarness(0.999).compare(z, fz_host, ctx))

    # denorm: z = z*scale + shift  (128-dim broadcast)
    var scale = reshape(Tensor.from_view(ln.tensor_view("latent_scale"), ctx), [1, 1, 128], ctx)
    var shift = reshape(Tensor.from_view(ln.tensor_view("latent_shift"), ctx), [1, 1, 128], ctx)
    var zd = add(mul(z, scale, ctx), shift, ctx)                       # [1,NIMG,128] F32

    # Ideogram unpatch: [1,gh,gw,2,2,32] -> permute(0,5,1,3,2,4) -> [1,32,2gh,2gw]
    var z6 = reshape(zd, [1, GH, GW, 2, 2, 32], ctx)
    var zp = permute(z6, [0, 5, 1, 3, 2, 4], ctx)                      # [1,32,GH,2,GW,2]
    var latent = reshape(zp, [1, 32, 2 * GH, 2 * GW], ctx)            # [1,32,32,32]
    var fl_host = Tensor.from_view(fx.tensor_view("final_latent"), ctx).to_host(ctx)
    print("chunk9 final_latent parity:", ParityHarness(0.999).compare(latent, fl_host, ctx))

    # VAE decode
    var dec = load_ideogram4_vae_decoder[2 * GH, 2 * GW](VAE, ctx)
    var img = dec.decode(cast_tensor(latent, STDtype.BF16, ctx), ctx)  # [1,3,256,256]
    var dec_host = Tensor.from_view(fx.tensor_view("decoded"), ctx).to_host(ctx)
    print("chunk9 decoded parity:", ParityHarness(0.999).compare(img, dec_host, ctx))

    save_png(img, "/home/alex/mojodiffusion/output/ideogram4_256.png", ctx)
    print("saved output/ideogram4_256.png")
