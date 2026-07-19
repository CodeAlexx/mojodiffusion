# pipeline/ideogram4_ab_harness.mojo — Rust-vs-Mojo A/B harness (MJ-1047/MJ-1051).
#
# Byte-identical-inputs ideogram4 render: injects the SAME init noise and the SAME
# llm_features conditioning that inference-flame's ideogram4_infer.rs consumes
# (its output/ideogram4_embeddings.safetensors sidecar + --noise-file format:
# tensor "tensor" F32 [1,4096,128]), so any output difference is pipeline-caused,
# not RNG/encoder-caused. 512px (Rust --size 512 default), V4_DEFAULT_20 recipe.
#
# argv:
#   1 noise safetensors        (tensor "tensor" F32 [1,1024,128]; REQUIRED)
#   2 embeddings safetensors   (llm_features F32 [1,18,53248] + num_text; REQUIRED)
#   3 out prefix               (writes <prefix>_whole.png, <prefix>_tiled.png,
#                               <prefix>_final_latent.safetensors)
#   4 cfg mode                 "sched" = V4 polish 3/7 (faithful, ideogram4_generate)
#                              "const7" = constant 7.0 (serve-worker behavior, MJ-1051)
#   5 per-step dumps           "dump" = write <prefix>_step_NN.safetensors each step
#                              "-"    = skip
#
# Everything else (forward, mrope, masks, denorm, unpatchify) is copied verbatim
# from the parity-verified pipeline/ideogram4_generate.mojo.
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.sys import argv
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import mul, add, mul_scalar, reshape, permute, slice, concat
from serenitymojo.image.png import save_png
from serenitymojo.models.dit.ideogram4_resident import Ideogram4Weights, ideogram4_forward_r, ideogram4_build_masks
from serenitymojo.models.dit.ideogram4_mrope import build_ideogram4_mrope
from serenitymojo.models.vae.ldm_decoder import load_ideogram4_vae_decoder
from serenitymojo.models.vae.ideogram4_tiled_decode import ideogram4_tiled_decode
from serenitymojo.sampling.ideogram4_schedule import ideogram4_logitnormal, ideogram4_schedule_mean, make_step_intervals

comptime COND = "/home/alex/.serenity/models/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors"
comptime UNCOND = "/home/alex/.serenity/models/ideogram-4-fp8/unconditional_transformer/diffusion_pytorch_model.safetensors"
comptime VAE = "/home/alex/.serenity/models/ideogram-4-fp8/vae/diffusion_pytorch_model.safetensors"
comptime LATENTNORM = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_latentnorm.safetensors"

comptime IMG_OFFSET = 65536
comptime NT = 18           # matches the Rust sidecar's num_text (fox prompt)
comptime GH = 64           # 1024/16 — Alex: 1024 is the minimum standard
comptime GW = 64
comptime NIMG = GH * GW    # 1024
comptime TOTAL = NT + NIMG # 1042
comptime STEPS = 20
comptime VAE_H = 2 * GH    # 64
comptime VAE_W = 2 * GW


# position_ids/indicator for [text][image], mirrored from ideogram4_generate.mojo.
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
    var a = argv()
    if len(a) < 6:
        raise Error("usage: ideogram4_ab_harness <noise.st> <embeddings.st> <out_prefix> <sched|const7> <dump|->")
    var noise_path = String(a[1])
    var emb_path = String(a[2])
    var prefix = String(a[3])
    var cfg_mode = String(a[4])
    var dump_steps = String(a[5]) == "dump"
    if cfg_mode != "sched" and cfg_mode != "const7":
        raise Error("cfg mode must be 'sched' or 'const7', got: " + cfg_mode)

    var ctx = DeviceContext()

    # --- injected conditioning (bypasses the Mojo inline encoder entirely) ---
    var emb = ShardedSafeTensors.open(emb_path)
    var lf = Tensor.from_view(emb.tensor_view("llm_features"), ctx)  # [1,NT,53248] F32
    if lf.shape()[1] != NT:
        raise Error("embeddings NT=" + String(lf.shape()[1]) + " != comptime NT=" + String(NT))
    var text_features = cast_tensor(lf, STDtype.BF16, ctx)

    var inp = build_inputs(ctx)

    var zllm = List[Float32]()
    for _ in range(NIMG * 53248):
        zllm.append(0.0)
    var img_zeros = Tensor.from_host(zllm^, [1, NIMG, 53248], STDtype.BF16, ctx)
    var llm = concat(1, ctx, text_features, img_zeros)  # [1,TOTAL,53248]
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
    var cond_masks = ideogram4_build_masks(inp[1][], ctx)
    var uncond_masks = ideogram4_build_masks(inp[3][], ctx)

    # --- injected init noise (Rust --noise-file format: "tensor" F32) ---
    var nz = ShardedSafeTensors.open(noise_path)
    var z = cast_tensor(Tensor.from_view(nz.tensor_view("tensor"), ctx), STDtype.F32, ctx)
    z = reshape(z, [1, NIMG, 128], ctx)
    print("init latent: LOADED from", noise_path, "(injected, parity)")

    var zpad_h = List[Float32]()
    for _ in range(NT * 128):
        zpad_h.append(0.0)
    var text_zpad = Tensor.from_host(zpad_h^, [1, NT, 128], STDtype.F32, ctx)

    # V4_DEFAULT_20 at 512px: mean = 0.0 + 0.5*ln(512²/512²) = 0.0, std 1.75.
    var mean = ideogram4_schedule_mean(2 * GH * 8, 2 * GW * 8, 0.0)
    var si = make_step_intervals(STEPS)
    print("schedule mean(mu) =", mean, " std = 1.75  cfg_mode =", cfg_mode)

    for step in range(STEPS - 1, -1, -1):
        var t_val = ideogram4_logitnormal(Float64(si[step + 1]), mean, 1.75)
        var s_val = ideogram4_logitnormal(Float64(si[step]), mean, 1.75)
        var gw = Float32(7.0)
        if cfg_mode == "sched" and step < 2:
            gw = Float32(3.0)  # V4 polish steps (the serve worker skips this drop)
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
        if dump_steps:
            var dn = List[String]()
            dn.append(String("tensor"))
            var dt = List[ArcPointer[Tensor]]()
            dt.append(ArcPointer(z.clone(ctx)))
            var nn = String(step)
            if step < 10:
                nn = String("0") + nn
            save_safetensors(dn, dt, prefix + "_step_" + nn + ".safetensors", ctx)

    # final latent out (same format Rust --latent-out writes: "tensor" F32)
    var fn_ = List[String]()
    fn_.append(String("tensor"))
    var ft = List[ArcPointer[Tensor]]()
    ft.append(ArcPointer(z.clone(ctx)))
    save_safetensors(fn_, ft, prefix + "_final_latent.safetensors", ctx)

    # denorm + unpatch (verbatim from ideogram4_generate.mojo)
    var ln = ShardedSafeTensors.open(LATENTNORM)
    var scale = reshape(Tensor.from_view(ln.tensor_view("latent_scale"), ctx), [1, 1, 128], ctx)
    var shift = reshape(Tensor.from_view(ln.tensor_view("latent_shift"), ctx), [1, 1, 128], ctx)
    var zd = add(mul(z, scale, ctx), shift, ctx)
    var z6 = reshape(zd, [1, GH, GW, 2, 2, 32], ctx)
    var zp = permute(z6, [0, 5, 1, 3, 2, 4], ctx)
    var latent = reshape(zp, [1, 32, VAE_H, VAE_W], ctx)
    var latent_bf = cast_tensor(latent, STDtype.BF16, ctx)

    # decode BOTH ways from the SAME latent: whole-image (Rust behavior) and
    # 3x3 tiled (the serve worker's MJ-1051 path) — isolates tiling quality cost.
    var dec = load_ideogram4_vae_decoder[VAE_H, VAE_W](VAE, ctx)
    var img_whole = dec.decode(latent_bf, ctx)
    save_png(img_whole, prefix + "_whole.png", ctx)
    print("saved", prefix + "_whole.png")
    var img_tiled = ideogram4_tiled_decode[VAE_H, VAE_W](latent_bf, String(VAE), ctx)
    save_png(img_tiled, prefix + "_tiled.png", ctx)
    print("saved", prefix + "_tiled.png")
    print("A/B harness done:", cfg_mode)
