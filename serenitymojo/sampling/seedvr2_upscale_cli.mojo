# sampling/seedvr2_upscale_cli.mojo — SeedVR2-3B END-TO-END upscale pipeline (pure Mojo + MAX).
#
# INFERENCE ONLY. Composes the VERIFIED pieces:
#   VAE encode -> DiT euler v-lerp sampler (8 steps x 2 CFG forwards) -> VAE decode
# and gates the result against the torch reference (pipeline_oracle.safetensors).
#
#   VAE  weights: seedvr2_vae.safetensors  (F32, streamed by name)
#   DiT  weights: seedvr2_dit.safetensors  (bf16, load_dit_weights preload)
#   text:         seedvr2_text_emb.safetensors  pos_emb[58,5120] neg_emb[64,5120] (F32->bf16)
#   input:        real_input.safetensors  real_input[1,3,13,128,128] (F32)
#   oracle:       pipeline_oracle.safetensors  x_T[1024,16] tsched[8] s_all[8]
#                   final_latent[1024,16] decoded[1,3,13,128,128] (F32)
#
# Pipeline (VAE_SCALE=0.9152, T=1000, CFG=7.5, 8 steps):
#   video   = real_input                                     # NCDHW [1,3,13,128,128] F32
#   latent  = encode_seedvr2_vae(video)                      # NCDHW [1,16,4,16,16] F32
#   latent  = mul_scalar(latent, 0.9152)                     # normalize
#   lat_flat= permute[0,2,3,4,1] -> [1,4,16,16,16] -> reshape [1024,16]  # (t,h,w)-major, c inner
#   cond    = concat(lat_flat, ones[1024,1], dim=1)          # [1024,17]  (flag=1.0)
#   x       = x_T                                            # F32 [1024,16]
#   for i in 0..7:
#     vid  = concat(bf16(x), bf16(cond), dim=1)              # [1024,33] bf16
#     pp   = full_dit_forward_pre(vid, pos_emb, 58, tsched[i], w)   # bf16 [1024,16]
#     pn   = full_dit_forward_pre(vid, neg_emb, 64, tsched[i], w)
#     pred = f32(cfg(pp, pn, 7.5))                           # bf16 CFG then upcast
#     x    = euler_vlerp_step(pred, x, tsched[i], s_all[i], 1000)   # x stays F32
#   [gate x vs final_latent]
#   final   = mul_scalar(x, 1/0.9152)                        # un-normalize
#   lat_out = reshape [1,4,16,16,16] -> permute[0,4,1,2,3] -> [1,16,4,16,16]
#   decoded = decode_seedvr2_vae(lat_out)                    # [1,3,13,128,128]
#   [gate decoded vs oracle decoded] ; save frames as PNG
#
# Build (AOT — needs -lpng16 -lturbojpeg for save_png, mirrors realesrgan_cli):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build -I . -I vendor/mojo-libs -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     -Xlinker -lpng16 -Xlinker -lturbojpeg \
#     serenitymojo/sampling/seedvr2_upscale_cli.mojo -o /tmp/seedvr2_up && \
#   env LD_LIBRARY_PATH=.pixi/envs/default/lib /tmp/seedvr2_up

from std.gpu.host import DeviceContext
from math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.sdxl.real_weights import load_bias
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import concat, reshape, permute, mul_scalar, slice
from serenitymojo.models.dit.seedvr2_dit import load_dit_weights, full_dit_forward_pre
from serenitymojo.models.dit.seedvr2_sampler import cfg, euler_vlerp_step
from serenitymojo.models.vae.seedvr2_vae import encode_seedvr2_vae, decode_seedvr2_vae
from serenitymojo.image.png import save_png, ValueRange

comptime VAE_W = "/home/alex/models/seedvr2-3b/seedvr2_vae.safetensors"
comptime DIT_W = "/home/alex/models/seedvr2-3b/seedvr2_dit.safetensors"
comptime TEXT = "/home/alex/models/seedvr2-3b/seedvr2_text_emb.safetensors"
comptime INPUT = "/home/alex/models/seedvr2-3b/real_input.safetensors"
comptime ORACLE = "/home/alex/models/seedvr2-3b/pipeline_oracle.safetensors"
comptime VAE_SCALE = Float32(0.9152)


def _cos_maxabs(a: List[Float32], b: List[Float32]) -> Tuple[Float32, Float32]:
    var n = len(a)
    if len(b) < n:
        n = len(b)
    var dot: Float64 = 0.0
    var na: Float64 = 0.0
    var nb: Float64 = 0.0
    var maxd: Float32 = 0.0
    for i in range(n):
        var x = Float64(a[i])
        var y = Float64(b[i])
        dot += x * y
        na += x * x
        nb += y * y
        var d = a[i] - b[i]
        if d < 0.0:
            d = -d
        if d > maxd:
            maxd = d
    var denom = sqrt(na) * sqrt(nb)
    var cos: Float32 = 0.0
    if denom > 0.0:
        cos = Float32(dot / denom)
    return (cos, maxd)


# Runs the DiT euler v-lerp sampler in its own scope so the ~6.8GB bf16 DiT
# weights are freed on return (before the VAE decode's big activations).
def _run_sampler(cond: Tensor, x_in: Tensor, ctx: DeviceContext) raises -> Tensor:
    var st = SafeTensors.open(DIT_W)
    var txe = SafeTensors.open(TEXT)
    var orc = SafeTensors.open(ORACLE)

    var pos_emb = cast_tensor(load_bias(txe, "pos_emb", ctx), STDtype.BF16, ctx)  # [58,5120]
    var neg_emb = cast_tensor(load_bias(txe, "neg_emb", ctx), STDtype.BF16, ctx)  # [64,5120]
    var tsched = load_bias(orc, "tsched", ctx).to_host(ctx)  # [8]
    var s_all = load_bias(orc, "s_all", ctx).to_host(ctx)    # [8]

    var cond_bf = cast_tensor(cond, STDtype.BF16, ctx)  # [1024,17] bf16 (reused)
    var w = load_dit_weights(st, ctx)
    print("  DiT weights loaded (32 blocks + heads/tail)")

    var x = x_in.clone(ctx)  # F32 [1024,16], mutated across steps
    for i in range(8):
        var x_bf = cast_tensor(x, STDtype.BF16, ctx)          # [1024,16] bf16
        var vid = concat(1, ctx, x_bf, cond_bf)               # [1024,33] bf16
        var pp = full_dit_forward_pre(vid, pos_emb, 58, tsched[i], w, ctx)  # bf16 [1024,16]
        var pn = full_dit_forward_pre(vid, neg_emb, 64, tsched[i], w, ctx)
        var pred_bf = cfg(pp, pn, Float32(7.5), ctx)          # bf16 CFG (torch autocast order)
        var pred = cast_tensor(pred_bf, STDtype.F32, ctx)     # upcast for F32 step
        x = euler_vlerp_step(pred, x, tsched[i], s_all[i], Float32(1000.0), ctx)
        print("  step", i, "t=", tsched[i], "s=", s_all[i], "done")
    return x^


def main() raises:
    var ctx = DeviceContext()
    print("ctx ok", Int(ctx.id()))

    var vae = SafeTensors.open(VAE_W)
    var inp = SafeTensors.open(INPUT)
    var orc = SafeTensors.open(ORACLE)

    # ── 1. VAE ENCODE ────────────────────────────────────────────────────────
    var video = load_bias(inp, "real_input", ctx)  # F32 NCDHW [1,3,13,128,128]
    print("video", video.shape()[0], video.shape()[1], video.shape()[2],
          video.shape()[3], video.shape()[4])
    var latent = encode_seedvr2_vae(video, vae, ctx)  # NCDHW [1,16,4,16,16] F32
    print("latent", latent.shape()[0], latent.shape()[1], latent.shape()[2],
          latent.shape()[3], latent.shape()[4])

    # normalize
    latent = mul_scalar(latent, VAE_SCALE, ctx)

    # ── 2. REARRANGE NCDHW -> flat [1024,16] : (t,h,w)-major, c innermost ─────
    #   [1,16,4,16,16] --permute[0,2,3,4,1]--> [1,4,16,16,16] --reshape--> [1024,16]
    var pperm = List[Int]()
    pperm.append(0); pperm.append(2); pperm.append(3); pperm.append(4); pperm.append(1)
    var lat_thwc = permute(latent, pperm, ctx)  # [1,4,16,16,16] = [N,T,H,W,C]
    var flat_shape = List[Int]()
    flat_shape.append(1024); flat_shape.append(16)
    var lat_flat = reshape(lat_thwc, flat_shape^, ctx)  # [1024,16] F32

    # cond = concat(lat_flat, ones[1024,1], dim=1) -> [1024,17]
    var ones_h = List[Float32](capacity=1024)
    for _ in range(1024):
        ones_h.append(Float32(1.0))
    var ones_shape = List[Int]()
    ones_shape.append(1024); ones_shape.append(1)
    var ones_t = Tensor.from_host(ones_h, ones_shape^, STDtype.F32, ctx)
    var cond = concat(1, ctx, lat_flat, ones_t)  # [1024,17] F32
    print("cond", cond.shape()[0], cond.shape()[1])

    # ── 3. DiT SAMPLER (8 steps x 2 CFG forwards) ────────────────────────────
    var x_T = load_bias(orc, "x_T", ctx)  # F32 [1024,16]
    var x = _run_sampler(cond, x_T, ctx)  # F32 [1024,16]; DiT weights freed on return

    # ── gate x vs final_latent (normalized flat latent) ──────────────────────
    var x_h = x.to_host(ctx)
    var fl_h = load_bias(orc, "final_latent", ctx).to_host(ctx)
    var rfl = _cos_maxabs(x_h, fl_h)
    print("final_latent cos=", rfl[0], " max_abs=", rfl[1])

    # ── 4. un-normalize + rearrange flat -> NCDHW ────────────────────────────
    var final_flat = mul_scalar(x, Float32(1.0) / VAE_SCALE, ctx)  # [1024,16] F32
    var thwc_shape = List[Int]()
    thwc_shape.append(1); thwc_shape.append(4); thwc_shape.append(16)
    thwc_shape.append(16); thwc_shape.append(16)
    var lat_out_thwc = reshape(final_flat, thwc_shape^, ctx)  # [1,4,16,16,16] = [N,T,H,W,C]
    var iperm = List[Int]()
    iperm.append(0); iperm.append(4); iperm.append(1); iperm.append(2); iperm.append(3)
    var lat_out = permute(lat_out_thwc, iperm, ctx)  # [1,16,4,16,16] NCDHW

    # ── 5. VAE DECODE ────────────────────────────────────────────────────────
    var decoded = decode_seedvr2_vae(lat_out, vae, ctx)  # [1,3,13,128,128] ~[-1,1]
    print("decoded", decoded.shape()[0], decoded.shape()[1], decoded.shape()[2],
          decoded.shape()[3], decoded.shape()[4])

    # ── gate decoded vs oracle decoded ───────────────────────────────────────
    var dec_h = decoded.to_host(ctx)
    var ref_h = load_bias(orc, "decoded", ctx).to_host(ctx)
    var rdec = _cos_maxabs(dec_h, ref_h)
    print("decoded cos=", rdec[0], " max_abs=", rdec[1])

    # ── 6. SAVE FRAMES AS PNG (frame is [1,3,128,128] sliced along D=13) ──────
    var frame_ids = List[Int]()
    frame_ids.append(0); frame_ids.append(3); frame_ids.append(6)
    frame_ids.append(9); frame_ids.append(12)
    var frame_shape = List[Int]()
    frame_shape.append(1); frame_shape.append(3); frame_shape.append(128); frame_shape.append(128)
    for fi in range(len(frame_ids)):
        var fidx = frame_ids[fi]
        var fr = slice(decoded, 2, fidx, 1, ctx)  # [1,3,1,128,128]
        var fr2d = reshape(fr, frame_shape.copy(), ctx)  # [1,3,128,128]
        var out_path = String("/tmp/seedvr2_mojo_out_f") + String(fidx) + String(".png")
        save_png(fr2d, out_path, ctx, ValueRange.SIGNED)  # decoded is ~[-1,1]
        print("  wrote", out_path)

    if rdec[0] >= 0.98:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
