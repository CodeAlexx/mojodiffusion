# pipeline/ideogram4_generate_lora.mojo — Ideogram-4 text->image WITH a trained
# giger LoRA applied (additive overlay on the resident cond transformer) + a real
# giger .json caption prompt. Proves whether the giger LoRA learned the style.
# Same path as ideogram4_generate.mojo; only the prompt + LoRA apply differ.
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import mul, add, mul_scalar, reshape, permute, slice, concat
from serenitymojo.image.png import save_png
from serenitymojo.models.dit.ideogram4_resident import Ideogram4Weights, ideogram4_forward_r, ideogram4_build_masks
from serenitymojo.models.dit.ideogram4_mrope import build_ideogram4_mrope
from serenitymojo.models.vae.ldm_decoder import load_ideogram4_vae_decoder
from serenitymojo.models.text_encoder.ideogram_qwen3vl import load_ideogram_qwen3vl, encode_ideogram_taps
from serenitymojo.sampling.ideogram4_schedule import ideogram4_logitnormal, ideogram4_schedule_mean, make_step_intervals

comptime COND = "/home/alex/.serenity/models/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors"
comptime UNCOND = "/home/alex/.serenity/models/ideogram-4-fp8/unconditional_transformer/diffusion_pytorch_model.safetensors"
comptime TE = "/home/alex/.serenity/models/ideogram-4-fp8/text_encoder/model.safetensors"
comptime VAE = "/home/alex/.serenity/models/ideogram-4-fp8/vae/diffusion_pytorch_model.safetensors"
comptime LORA = "/home/alex/mojodiffusion/output/lora_step_3000.safetensors"

comptime IMG_OFFSET = 65536
comptime NT = 651          # giger 10.json structured caption, chat-templated
comptime QSEQ = 1024       # Qwen sdpa supported seq (pad NT 651 -> 1024)
comptime GH = 64
comptime GW = 64
comptime NIMG = GH * GW
comptime TOTAL = NT + NIMG
comptime STEPS = 20
comptime SEED = 0


def encode_prompt(ctx: DeviceContext) raises -> Tensor:
    var enc = load_ideogram_qwen3vl(TE, ctx)
    var ids = [151644, 872, 198, 4913, 11892, 8274, 11448, 3252, 70, 343, 2836, 18, 1707, 13, 362, 6319, 64523, 33033, 315, 264, 13510, 3850, 85124, 8593, 7071, 448, 264, 27539, 31598, 73544, 3579, 323, 458, 73495, 657, 56490, 291, 1424, 9226, 1573, 1435, 11, 738, 2348, 264, 51987, 296, 1716, 832, 6543, 31598, 323, 3691, 4004, 47891, 3528, 11448, 22317, 64, 477, 48366, 3252, 22624, 11, 64523, 11, 83610, 5658, 938, 11, 728, 81564, 21452, 11, 85124, 11, 85496, 2198, 4145, 287, 3252, 10303, 16173, 448, 5103, 35966, 389, 279, 3579, 11, 5538, 2518, 34188, 11369, 11, 8811, 34512, 2198, 26086, 3252, 33617, 287, 2198, 471, 15117, 3252, 39, 2013, 13, 479, 7272, 3419, 26164, 5767, 6319, 64523, 2142, 11, 3720, 36061, 291, 296, 1716, 832, 29853, 11, 73495, 657, 293, 3549, 61590, 2198, 3423, 66252, 36799, 2, 15, 15, 15, 15, 15, 15, 58324, 16, 32, 15, 21, 15, 21, 58324, 18, 32, 15, 36, 15, 34, 58324, 20, 36, 16, 21, 16, 19, 58324, 23, 21, 17, 15, 16, 34, 58324, 32, 23, 18, 32, 18, 15, 58324, 34, 24, 20, 32, 19, 23, 58324, 22, 32, 19, 32, 19, 15, 1341, 51193, 874, 966, 3005, 2259, 47197, 22317, 6742, 3252, 95971, 14212, 296, 1716, 832, 11369, 315, 6319, 2518, 82, 11, 38977, 323, 2816, 1047, 448, 92278, 8115, 9437, 12681, 323, 15801, 26522, 29853, 47891, 21423, 66582, 1313, 3252, 2295, 2198, 58456, 8899, 17, 15, 15, 11, 17, 21, 15, 11, 22, 17, 15, 11, 21, 19, 15, 28503, 8614, 3252, 80788, 3850, 85124, 8593, 3579, 448, 7015, 2712, 6414, 11, 17232, 61158, 5128, 323, 62144, 812, 296, 1716, 832, 6787, 11, 7493, 18894, 47891, 3423, 66252, 36799, 2, 34, 24, 20, 32, 19, 23, 58324, 32, 23, 18, 32, 18, 15, 58324, 23, 21, 17, 15, 16, 34, 58324, 20, 36, 16, 21, 16, 19, 58324, 16, 32, 15, 21, 15, 21, 1341, 36828, 1313, 3252, 2295, 2198, 58456, 8899, 18, 18, 15, 11, 16, 23, 15, 11, 22, 23, 15, 11, 19, 21, 15, 28503, 8614, 3252, 36, 4825, 657, 293, 3549, 56490, 291, 1424, 448, 36293, 12681, 683, 13595, 19225, 9226, 304, 4065, 315, 279, 3579, 47891, 3423, 66252, 36799, 2, 34, 24, 20, 32, 19, 23, 58324, 32, 23, 18, 32, 18, 15, 58324, 22, 32, 19, 32, 19, 15, 58324, 20, 36, 16, 21, 16, 19, 58324, 18, 32, 15, 36, 15, 34, 1341, 36828, 1313, 3252, 2295, 2198, 58456, 8899, 21, 15, 11, 17, 21, 15, 11, 18, 15, 15, 11, 22, 17, 15, 28503, 8614, 3252, 13218, 50768, 1525, 69881, 83610, 5658, 938, 3072, 315, 2518, 323, 3691, 29064, 279, 7071, 594, 6869, 2899, 4830, 47891, 3423, 66252, 36799, 2, 23, 21, 17, 15, 16, 34, 58324, 20, 36, 16, 21, 16, 19, 58324, 18, 32, 15, 36, 15, 34, 58324, 16, 32, 15, 21, 15, 21, 58324, 32, 23, 18, 32, 18, 15, 1341, 36828, 1313, 3252, 2295, 2198, 58456, 8899, 21, 15, 15, 11, 17, 15, 15, 11, 16, 15, 15, 15, 11, 23, 17, 15, 28503, 8614, 3252, 25830, 62144, 812, 26906, 323, 63200, 315, 279, 7071, 13772, 19648, 1119, 279, 73544, 4004, 47891, 3423, 66252, 36799, 2, 20, 36, 16, 21, 16, 19, 58324, 18, 32, 15, 36, 15, 34, 58324, 16, 32, 15, 21, 15, 21, 58324, 15, 15, 15, 15, 15, 15, 58324, 23, 21, 17, 15, 16, 34, 1341, 36828, 1313, 3252, 2295, 2198, 58456, 8899, 23, 17, 15, 11, 18, 15, 15, 11, 16, 15, 15, 15, 11, 21, 17, 15, 28503, 8614, 3252, 19641, 14283, 36293, 12681, 80358, 35707, 16062, 518, 279, 4722, 4126, 3685, 279, 1424, 47891, 3423, 66252, 36799, 2, 32, 23, 18, 32, 18, 15, 58324, 23, 21, 17, 15, 16, 34, 58324, 20, 36, 16, 21, 16, 19, 58324, 18, 32, 15, 36, 15, 34, 58324, 15, 15, 15, 15, 15, 15, 1341, 25439, 3417, 151645, 198, 151644, 77091, 198]
    for _ in range(QSEQ - NT):
        ids.append(151643)
    var f = encode_ideogram_taps(enc, ids, ctx)
    return slice(f, 1, 0, NT, ctx)


def build_inputs(ctx: DeviceContext) raises -> List[ArcPointer[Tensor]]:
    var pos = List[Float32]()
    var ind = List[Float32]()
    var npos = List[Float32]()
    var nind = List[Float32]()
    for l in range(NT):
        pos.append(Float32(l)); pos.append(Float32(l)); pos.append(Float32(l))
        ind.append(3.0)
    for h in range(GH):
        for w in range(GW):
            var t0 = Float32(IMG_OFFSET); var hh = Float32(IMG_OFFSET + h); var ww = Float32(IMG_OFFSET + w)
            pos.append(t0); pos.append(hh); pos.append(ww)
            npos.append(t0); npos.append(hh); npos.append(ww)
            ind.append(2.0); nind.append(2.0)
    var out = List[ArcPointer[Tensor]]()
    out.append(ArcPointer(Tensor.from_host(pos^, [1, TOTAL, 3], STDtype.F32, ctx)))
    out.append(ArcPointer(Tensor.from_host(ind^, [1, TOTAL], STDtype.F32, ctx)))
    out.append(ArcPointer(Tensor.from_host(npos^, [1, NIMG, 3], STDtype.F32, ctx)))
    out.append(ArcPointer(Tensor.from_host(nind^, [1, NIMG], STDtype.F32, ctx)))
    return out^


def main() raises:
    var ctx = DeviceContext()
    print("encoding giger prompt (Qwen3-VL 13-tap)...")
    var text_features = encode_prompt(ctx)

    var inp = build_inputs(ctx)
    var zllm = List[Float32]()
    for _ in range(NIMG * 53248):
        zllm.append(0.0)
    var img_zeros = Tensor.from_host(zllm^, [1, NIMG, 53248], STDtype.BF16, ctx)
    var llm = concat(1, ctx, text_features, img_zeros)
    var nllm_h = List[Float32]()
    for _ in range(NIMG * 53248):
        nllm_h.append(0.0)
    var neg_llm = Tensor.from_host(nllm_h^, [1, NIMG, 53248], STDtype.BF16, ctx)

    var sec = [24, 20, 20]
    var cs = build_ideogram4_mrope(inp[0][], 256, sec, Float32(5000000.0), ctx, STDtype.BF16)
    var ncs = build_ideogram4_mrope(inp[2][], 256, sec, Float32(5000000.0), ctx, STDtype.BF16)

    print("loading resident fp8 cond+uncond...")
    var cond_w = Ideogram4Weights.load(ShardedSafeTensors.open(COND), ctx)
    var n = cond_w.load_lora(LORA, ctx)
    print("applied LoRA adapters:", n)
    var uncond_w = Ideogram4Weights.load(ShardedSafeTensors.open(UNCOND), ctx)
    var cond_masks = ideogram4_build_masks(inp[1][], ctx)
    var uncond_masks = ideogram4_build_masks(inp[3][], ctx)

    var z = randn([1, NIMG, 128], UInt64(SEED), STDtype.F32, ctx)
    var zpad_h = List[Float32]()
    for _ in range(NT * 128):
        zpad_h.append(0.0)
    var text_zpad = Tensor.from_host(zpad_h^, [1, NT, 128], STDtype.F32, ctx)

    var mean = ideogram4_schedule_mean(1024, 1024, 0.0)
    var si = make_step_intervals(STEPS)
    for step in range(STEPS - 1, -1, -1):
        var t_val = ideogram4_logitnormal(Float64(si[step + 1]), mean, 1.5)
        var s_val = ideogram4_logitnormal(Float64(si[step]), mean, 1.5)
        var gw = Float32(3.0) if step < 3 else Float32(7.0)
        var t = Tensor.from_host([t_val], [1], STDtype.F32, ctx)
        var pos_z = cast_tensor(concat(1, ctx, text_zpad, z), STDtype.BF16, ctx)
        var cout = ideogram4_forward_r[TOTAL](cond_w, pos_z, llm, t, cond_masks, cs[0], cs[1], 34, 18, 256, 4608, ctx)
        var pos_v = slice(cout, 1, NT, NIMG, ctx)
        var t2 = Tensor.from_host([t_val], [1], STDtype.F32, ctx)
        var z_bf = cast_tensor(z, STDtype.BF16, ctx)
        var nout = ideogram4_forward_r[NIMG](uncond_w, z_bf, neg_llm, t2, uncond_masks, ncs[0], ncs[1], 34, 18, 256, 4608, ctx)
        var v = add(mul_scalar(pos_v, gw, ctx), mul_scalar(nout, Float32(1.0) - gw, ctx), ctx)
        z = add(z, mul_scalar(v, s_val - t_val, ctx), ctx)
        print("  step", step, "gw", gw)

    var ln = ShardedSafeTensors.open("/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_latentnorm.safetensors")
    var scale = reshape(Tensor.from_view(ln.tensor_view("latent_scale"), ctx), [1, 1, 128], ctx)
    var shift = reshape(Tensor.from_view(ln.tensor_view("latent_shift"), ctx), [1, 1, 128], ctx)
    var zd = add(mul(z, scale, ctx), shift, ctx)
    var z6 = reshape(zd, [1, GH, GW, 2, 2, 32], ctx)
    var zp = permute(z6, [0, 5, 1, 3, 2, 4], ctx)
    var latent = reshape(zp, [1, 32, 2 * GH, 2 * GW], ctx)
    var dec = load_ideogram4_vae_decoder[2 * GH, 2 * GW](VAE, ctx)
    var img = dec.decode(cast_tensor(latent, STDtype.BF16, ctx), ctx)
    save_png(img, "/home/alex/mojodiffusion/output/ideogram4_giger_lora.png", ctx)
    print("saved output/ideogram4_giger_lora.png")
