# pipeline/ideogram4_generate.mojo — Ideogram-4 NATIVE text->image (no fixtures).
# Full pure-Mojo path: hardcoded prompt token ids -> native Qwen3-VL 13-tap ->
# native packed inputs -> native randn noise -> CFG denoise (cond+uncond,
# logit-normal V4_QUALITY_48 preset) -> latent denorm -> Ideogram unpatch ->
# Flux2 VAE decode -> PNG. Reuses every parity-verified component.
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

comptime IMG_OFFSET = 65536
comptime NT = 648          # detailed structured JSON caption (owl/woman split-face), chat-templated
comptime QSEQ = 1024       # Qwen sdpa supported seq (pad NT 648 -> 1024)
comptime GH = 64           # 1024/16
comptime GW = 64
comptime NIMG = GH * GW    # 4096
comptime TOTAL = NT + NIMG # 4744
comptime STEPS = 20       # real recipe (V4 sample_steps=20), was 4 (turbo, too few)
comptime SEED = 0


# encode the prompt's 13-tap features and FREE the 17GB Qwen encoder on return.
def encode_prompt(ctx: DeviceContext) raises -> Tensor:
    var enc = load_ideogram_qwen3vl(TE, ctx)
    # 15 real ids + 1 pad (151643) -> seq 16 (supported); slice back to 15.
    var ids = [151644, 872, 198, 4913, 11892, 8274, 11448, 3252, 32, 6319, 64523, 6718, 29088, 33033, 25, 279, 2115, 4279, 374, 264, 17071, 1737, 10111, 59889, 52269, 594, 3579, 448, 264, 48492, 67605, 7912, 11, 279, 1290, 4279, 374, 264, 27539, 3908, 5220, 594, 3579, 448, 825, 6303, 7912, 323, 2518, 18778, 15423, 389, 1059, 40703, 11, 279, 1378, 74112, 26001, 60340, 1495, 279, 12140, 4126, 47891, 3528, 11448, 22317, 64, 477, 48366, 3252, 6355, 1076, 11, 17071, 7951, 4532, 11, 6319, 64523, 11, 64665, 11, 7548, 11682, 2198, 4145, 287, 3252, 10303, 16173, 21771, 3108, 17716, 11, 8413, 72845, 34512, 11, 59021, 7010, 41976, 2198, 26086, 3252, 57269, 36932, 2198, 471, 15117, 3252, 68192, 7951, 4532, 6319, 18378, 7377, 18824, 448, 4503, 89768, 6787, 323, 49776, 10434, 11, 6915, 8003, 29572, 11, 25600, 7990, 315, 2070, 2198, 3423, 66252, 36799, 2, 18, 32, 18, 20, 17, 36, 58324, 23, 33, 22, 18, 20, 20, 58324, 34, 24, 33, 22, 24, 34, 58324, 36, 23, 35, 24, 34, 20, 58324, 35, 24, 23, 32, 17, 33, 58324, 20, 33, 22, 34, 24, 24, 58324, 22, 32, 17, 36, 17, 36, 58324, 16, 37, 16, 33, 16, 22, 1341, 51193, 874, 966, 3005, 2259, 47197, 22317, 6742, 3252, 32, 72400, 11, 939, 34367, 19780, 1455, 4830, 38477, 11, 8413, 323, 700, 315, 5244, 448, 264, 6319, 92792, 6535, 2163, 279, 12822, 11, 10282, 678, 6529, 389, 279, 8622, 12300, 47891, 21423, 66582, 1313, 3252, 2295, 2198, 58456, 8899, 19, 15, 11, 15, 11, 23, 15, 15, 11, 19, 24, 15, 28503, 8614, 3252, 785, 2115, 4279, 315, 279, 4034, 374, 264, 17071, 1737, 10111, 3265, 5239, 315, 458, 59889, 52269, 594, 3579, 11, 448, 56116, 13876, 11, 14197, 323, 19780, 55894, 11, 264, 17232, 42670, 387, 585, 518, 4126, 11, 323, 49776, 9765, 44444, 13, 576, 52269, 54986, 288, 4637, 47891, 3423, 66252, 36799, 2, 23, 33, 22, 18, 20, 20, 58324, 18, 32, 18, 20, 17, 36, 58324, 34, 24, 33, 22, 24, 34, 58324, 35, 24, 23, 32, 17, 33, 1341, 36828, 1313, 3252, 2295, 2198, 58456, 8899, 17, 20, 15, 11, 16, 20, 15, 11, 18, 21, 15, 11, 18, 15, 15, 28503, 8614, 3252, 32, 3460, 4778, 48492, 67605, 84383, 52269, 7912, 448, 264, 5538, 3691, 59972, 323, 264, 6915, 10058, 315, 11682, 55894, 2163, 432, 47891, 3423, 66252, 36799, 2, 35, 24, 23, 32, 17, 33, 58324, 16, 32, 16, 32, 16, 32, 1341, 36828, 1313, 3252, 2295, 2198, 58456, 8899, 15, 11, 19, 23, 15, 11, 16, 15, 15, 15, 11, 16, 15, 15, 15, 28503, 8614, 3252, 785, 1290, 4279, 315, 279, 4034, 374, 264, 27539, 6624, 45307, 20561, 3908, 5220, 594, 3579, 448, 1293, 6319, 6869, 11, 264, 19300, 18894, 7493, 11, 2480, 5810, 22877, 11, 323, 10876, 86317, 73544, 6787, 13, 8278, 825, 3108, 315, 1059, 3579, 374, 6839, 11, 8652, 89579, 279, 52269, 47891, 3423, 66252, 36799, 2, 36, 23, 35, 24, 34, 20, 58324, 34, 24, 33, 22, 24, 34, 58324, 20, 33, 22, 34, 24, 24, 58324, 17, 33, 17, 22, 17, 18, 1341, 36828, 1313, 3252, 2295, 2198, 58456, 8899, 18, 15, 15, 11, 20, 21, 15, 11, 19, 15, 15, 11, 22, 19, 15, 28503, 8614, 3252, 32, 3175, 21239, 27539, 6303, 63213, 3738, 7912, 11, 10078, 73056, 323, 33068, 8413, 3100, 11, 3330, 5961, 4637, 47891, 3423, 66252, 36799, 2, 20, 33, 22, 34, 24, 24, 58324, 36, 23, 35, 24, 34, 20, 1341, 36828, 1313, 3252, 2295, 2198, 58456, 8899, 18, 21, 15, 11, 20, 21, 15, 11, 21, 15, 15, 11, 22, 17, 15, 28503, 8614, 3252, 19641, 37236, 6319, 31598, 5312, 2832, 34244, 18778, 15423, 11, 1075, 15430, 56490, 65739, 11, 3941, 279, 5220, 594, 40703, 19756, 47891, 3423, 66252, 36799, 2, 22, 32, 17, 36, 17, 36, 58324, 32, 18, 18, 18, 18, 18, 1341, 25439, 3417, 151645, 198, 151644, 77091, 198]
    for _ in range(QSEQ - NT):
        ids.append(151643)
    var f = encode_ideogram_taps(enc, ids, ctx)   # [1,QSEQ,53248]
    return slice(f, 1, 0, NT, ctx)                 # [1,NT,53248] bf16


# build position_ids/indicator host-side for [text][image] (left-pad=0, 1 prompt).
def build_inputs(ctx: DeviceContext) raises -> List[ArcPointer[Tensor]]:
    var pos = List[Float32]()
    var ind = List[Float32]()
    var npos = List[Float32]()
    var nind = List[Float32]()
    for l in range(NT):
        pos.append(Float32(l)); pos.append(Float32(l)); pos.append(Float32(l))
        ind.append(3.0)  # LLM_TOKEN_INDICATOR
    for h in range(GH):
        for w in range(GW):
            var t0 = Float32(IMG_OFFSET); var hh = Float32(IMG_OFFSET + h); var ww = Float32(IMG_OFFSET + w)
            pos.append(t0); pos.append(hh); pos.append(ww)
            npos.append(t0); npos.append(hh); npos.append(ww)
            ind.append(2.0)  # OUTPUT_IMAGE_INDICATOR
            nind.append(2.0)
    var out = List[ArcPointer[Tensor]]()
    out.append(ArcPointer(Tensor.from_host(pos^, [1, TOTAL, 3], STDtype.F32, ctx)))
    out.append(ArcPointer(Tensor.from_host(ind^, [1, TOTAL], STDtype.F32, ctx)))
    out.append(ArcPointer(Tensor.from_host(npos^, [1, NIMG, 3], STDtype.F32, ctx)))
    out.append(ArcPointer(Tensor.from_host(nind^, [1, NIMG], STDtype.F32, ctx)))
    return out^


def main() raises:
    var ctx = DeviceContext()
    print("encoding prompt (Qwen3-VL 13-tap)...")
    var text_features = encode_prompt(ctx)                       # [1,NT,53248] bf16

    var inp = build_inputs(ctx)  # [pos, ind, npos, nind] (borrow via inp[i][])

    # llm_full = [text_features ; zeros(NIMG)]  -> [1,TOTAL,53248] bf16
    var zllm = List[Float32]()
    for _ in range(NIMG * 53248):
        zllm.append(0.0)
    var img_zeros = Tensor.from_host(zllm^, [1, NIMG, 53248], STDtype.BF16, ctx)
    var llm = concat(1, ctx, text_features, img_zeros)           # [1,TOTAL,53248]
    var nllm_h = List[Float32]()
    for _ in range(NIMG * 53248):
        nllm_h.append(0.0)
    var neg_llm = Tensor.from_host(nllm_h^, [1, NIMG, 53248], STDtype.BF16, ctx)

    var sec = [24, 20, 20]
    var cs = build_ideogram4_mrope(inp[0][], 256, sec, Float32(5000000.0), ctx, STDtype.BF16)
    var ncs = build_ideogram4_mrope(inp[2][], 256, sec, Float32(5000000.0), ctx, STDtype.BF16)

    print("loading resident fp8 transformers (cond+uncond)...")
    var cond_w = Ideogram4Weights.load(ShardedSafeTensors.open(COND), ctx)
    var uncond_w = Ideogram4Weights.load(ShardedSafeTensors.open(UNCOND), ctx)
    var cond_masks = ideogram4_build_masks(inp[1][], ctx)   # hoisted: built once, reused every step
    var uncond_masks = ideogram4_build_masks(inp[3][], ctx)

    var z = randn([1, NIMG, 128], UInt64(SEED), STDtype.F32, ctx)
    var zpad_h = List[Float32]()
    for _ in range(NT * 128):
        zpad_h.append(0.0)
    var text_zpad = Tensor.from_host(zpad_h^, [1, NT, 128], STDtype.F32, ctx)

    # V4_TURBO_12: mu=0.5, std=1.75; guidance loop-index order [3.0]*1+[7.0]*11.
    var mean = ideogram4_schedule_mean(1024, 1024, 0.0)
    var si = make_step_intervals(STEPS)

    for step in range(STEPS - 1, -1, -1):
        # V4_DEFAULT_20 preset (sampler_configs.py / inference-flame scheduler.rs):
        # std=1.75 (NOT the 48-step QUALITY preset's 1.5), guidance = 2 polish
        # steps at 3.0 then 7.0. The std/polish mismatch was softening output.
        var t_val = ideogram4_logitnormal(Float64(si[step + 1]), mean, 1.75)
        var s_val = ideogram4_logitnormal(Float64(si[step]), mean, 1.75)
        var gw = Float32(3.0) if step < 2 else Float32(7.0)
        var t = Tensor.from_host([t_val], [1], STDtype.F32, ctx)
        var pos_z = cast_tensor(concat(1, ctx, text_zpad, z), STDtype.BF16, ctx)
        var cout = ideogram4_forward_r[TOTAL](cond_w, pos_z, llm, t, cond_masks, cs[0], cs[1], 34, 18, 256, 4608, ctx)
        var pos_v = slice(cout, 1, NT, NIMG, ctx)
        var t2 = Tensor.from_host([t_val], [1], STDtype.F32, ctx)
        var z_bf = cast_tensor(z, STDtype.BF16, ctx)
        var nout = ideogram4_forward_r[NIMG](uncond_w, z_bf, neg_llm, t2, uncond_masks, ncs[0], ncs[1], 34, 18, 256, 4608, ctx)
        var v = add(mul_scalar(pos_v, gw, ctx), mul_scalar(nout, Float32(1.0) - gw, ctx), ctx)
        z = add(z, mul_scalar(v, s_val - t_val, ctx), ctx)
        print("  step", step, "gw", gw, "t", t_val, "s", s_val)

    # denorm + unpatch + decode
    var ln = ShardedSafeTensors.open("/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_latentnorm.safetensors")
    var scale = reshape(Tensor.from_view(ln.tensor_view("latent_scale"), ctx), [1, 1, 128], ctx)
    var shift = reshape(Tensor.from_view(ln.tensor_view("latent_shift"), ctx), [1, 1, 128], ctx)
    var zd = add(mul(z, scale, ctx), shift, ctx)
    var z6 = reshape(zd, [1, GH, GW, 2, 2, 32], ctx)
    var zp = permute(z6, [0, 5, 1, 3, 2, 4], ctx)
    var latent = reshape(zp, [1, 32, 2 * GH, 2 * GW], ctx)
    var dec = load_ideogram4_vae_decoder[2 * GH, 2 * GW](VAE, ctx)
    var img = dec.decode(cast_tensor(latent, STDtype.BF16, ctx), ctx)
    save_png(img, "/home/alex/mojodiffusion/output/ideogram4_generated_1024.png", ctx)
    print("saved output/ideogram4_generated_1024.png  1024", img.shape()[2], "x", img.shape()[3])
