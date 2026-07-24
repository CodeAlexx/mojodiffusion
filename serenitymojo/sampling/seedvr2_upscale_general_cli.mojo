# sampling/seedvr2_upscale_general_cli.mojo — SeedVR2-3B ARBITRARY-RESOLUTION
# upscale pipeline (pure Mojo + MAX, INFERENCE ONLY).
#
# Generalizes seedvr2_upscale_cli.mojo to ANY latent grid by driving the
# grid-general DiT forward (full_dit_forward_general) instead of the fixed
# 4x16x16 windowed path. Same VAE encode -> DiT euler v-lerp sampler -> VAE
# decode composition; grid + SD3 shift are computed on the host from the image
# size.
#
# main() runs TWO things:
#   1. VERIFY at 128x128x13 (latent grid Tl=4,Hl=16,Wl=16) using the SAME x_T +
#      real_input the fixed CLI used, gating decoded vs pipeline_oracle. Confirms
#      the GENERAL path reproduces the fixed pipeline (expect cos >= 0.98).
#   2. DEMO at a NEW resolution 96x96x5 (latent grid Tl=2,Hl=12,Wl=12; post-patch
#      2,6,6 — a grid no gated run used) — center-cropped from real_input,
#      deterministic x_T noise. No torch reference; confirm it RUNS and is sane.
#
# Build (AOT — needs -lpng16 -lturbojpeg for save_png, mirrors realesrgan_cli):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build -I . -I vendor/mojo-libs -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     -Xlinker -lpng16 -Xlinker -lturbojpeg \
#     serenitymojo/sampling/seedvr2_upscale_general_cli.mojo -o /tmp/seedvr2_gup && \
#   env LD_LIBRARY_PATH=.pixi/envs/default/lib /tmp/seedvr2_gup

from std.gpu.host import DeviceContext
from math import sqrt, log, cos, sin

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.sdxl.real_weights import load_bias
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import concat, reshape, permute, mul_scalar, slice
from serenitymojo.models.dit.seedvr2_dit import load_dit_weights, full_dit_forward_general
from serenitymojo.models.dit.seedvr2_sampler import cfg, euler_vlerp_step, sampler_timesteps
from serenitymojo.models.vae.seedvr2_vae import encode_seedvr2_vae, decode_seedvr2_vae
from serenitymojo.image.png import save_png, ValueRange

comptime VAE_W = "/home/alex/models/seedvr2-3b/seedvr2_vae.safetensors"
comptime DIT_W = "/home/alex/models/seedvr2-3b/seedvr2_dit.safetensors"
comptime TEXT = "/home/alex/models/seedvr2-3b/seedvr2_text_emb.safetensors"
comptime INPUT = "/home/alex/models/seedvr2-3b/real_input.safetensors"
comptime ORACLE = "/home/alex/models/seedvr2-3b/pipeline_oracle.safetensors"
comptime VAE_SCALE = Float32(0.9152)
comptime PI = Float64(3.141592653589793)


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
    var cos_v: Float32 = 0.0
    if denom > 0.0:
        cos_v = Float32(dot / denom)
    return (cos_v, maxd)


def _minmax(a: List[Float32]) -> Tuple[Float32, Float32]:
    var lo = a[0]
    var hi = a[0]
    for i in range(len(a)):
        if a[i] < lo:
            lo = a[i]
        if a[i] > hi:
            hi = a[i]
    return (lo, hi)


# ── SeedVR timestep_transform SD3 shift from the latent grid (host F64) ──────────
#   frames_px = (Tl-1)*4+1 ; H_px = Hl*8 ; W_px = Wl*8.
#   video (frames_px>1): lin (256*256*37,1.0)->(1280*720*145,5.0) over H*W*F.
#   image (frames_px==1): lin (256*256,1.0)->(1024*1024,3.2) over H*W.
def compute_shift(Tl: Int, Hl: Int, Wl: Int) -> Float32:
    var frames_px = (Tl - 1) * 4 + 1
    var H_px = Hl * 8
    var W_px = Wl * 8
    if frames_px > 1:
        var x1 = 256.0 * 256.0 * 37.0
        var y1 = 1.0
        var x2 = 1280.0 * 720.0 * 145.0
        var y2 = 5.0
        var m = (y2 - y1) / (x2 - x1)
        var b = y1 - m * x1
        var xv = Float64(H_px) * Float64(W_px) * Float64(frames_px)
        return Float32(m * xv + b)
    else:
        var x1 = 256.0 * 256.0
        var y1 = 1.0
        var x2 = 1024.0 * 1024.0
        var y2 = 3.2
        var m = (y2 - y1) / (x2 - x1)
        var b = y1 - m * x1
        var xv = Float64(H_px) * Float64(W_px)
        return Float32(m * xv + b)


# ── deterministic standard-normal noise via LCG + Box-Muller ─────────────────────
def gen_noise(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var s = seed
    while len(out) < n:
        # two uniforms
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u1 = Float64((s >> 40)) / Float64(UInt64(1) << 24)
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u2 = Float64((s >> 40)) / Float64(UInt64(1) << 24)
        if u1 < 1e-7:
            u1 = 1e-7
        var r = sqrt(-2.0 * log(u1))
        var z0 = r * cos(2.0 * PI * u2)
        var z1 = r * sin(2.0 * PI * u2)
        out.append(Float32(z0))
        if len(out) < n:
            out.append(Float32(z1))
    return out^


# ── VAE latent [1,16,Tl,Hl,Wl] -> normalized flat cond [L,17] (L=Tl*Hl*Wl) ──────
def latent_to_cond(latent: Tensor, Tl: Int, Hl: Int, Wl: Int, ctx: DeviceContext) raises -> Tensor:
    var L = Tl * Hl * Wl
    var norm = mul_scalar(latent, VAE_SCALE, ctx)   # [1,16,Tl,Hl,Wl]
    var pperm = List[Int]()
    pperm.append(0); pperm.append(2); pperm.append(3); pperm.append(4); pperm.append(1)
    var thwc = permute(norm, pperm, ctx)            # [1,Tl,Hl,Wl,16]
    var flat_shape = List[Int]()
    flat_shape.append(L); flat_shape.append(16)
    var lat_flat = reshape(thwc, flat_shape^, ctx)  # [L,16]
    var ones_h = List[Float32](capacity=L)
    for _ in range(L):
        ones_h.append(Float32(1.0))
    var ones_shape = List[Int]()
    ones_shape.append(L); ones_shape.append(1)
    var ones_t = Tensor.from_host(ones_h, ones_shape^, STDtype.F32, ctx)
    return concat(1, ctx, lat_flat, ones_t)         # [L,17]


# ── flat normalized latent [L,16] -> un-normalized NCDHW [1,16,Tl,Hl,Wl] ─────────
def flat_to_ncdhw(x: Tensor, Tl: Int, Hl: Int, Wl: Int, ctx: DeviceContext) raises -> Tensor:
    var final_flat = mul_scalar(x, Float32(1.0) / VAE_SCALE, ctx)   # [L,16]
    var thwc_shape = List[Int]()
    thwc_shape.append(1); thwc_shape.append(Tl); thwc_shape.append(Hl)
    thwc_shape.append(Wl); thwc_shape.append(16)
    var thwc = reshape(final_flat, thwc_shape^, ctx)   # [1,Tl,Hl,Wl,16]
    var iperm = List[Int]()
    iperm.append(0); iperm.append(4); iperm.append(1); iperm.append(2); iperm.append(3)
    return permute(thwc, iperm, ctx)                   # [1,16,Tl,Hl,Wl]


# Runs the DiT euler v-lerp sampler (grid-general) in its own scope so the ~6.8GB
# bf16 DiT weights are freed on return (before the VAE decode's big activations).
def _run_sampler_general(
    cond: Tensor,          # [L,17] F32
    x_in: Tensor,          # [L,16] F32
    T: Int, H: Int, W: Int,   # PRE-patch latent grid (Tl,Hl,Wl)
    pos_emb: Tensor,       # [58,5120] bf16
    neg_emb: Tensor,       # [64,5120] bf16
    tsched: List[Float32],
    s_all: List[Float32],
    ctx: DeviceContext,
) raises -> Tensor:
    var st = SafeTensors.open(DIT_W)
    var cond_bf = cast_tensor(cond, STDtype.BF16, ctx)
    var w = load_dit_weights(st, ctx)
    print("  DiT weights loaded (32 blocks + heads/tail)")

    var x = x_in.clone(ctx)
    for i in range(8):
        var x_bf = cast_tensor(x, STDtype.BF16, ctx)          # [L,16] bf16
        var vid = concat(1, ctx, x_bf, cond_bf)               # [L,33] bf16
        var pp = full_dit_forward_general(vid, pos_emb, 58, tsched[i], T, H, W, w, ctx)  # [L,16]
        var pn = full_dit_forward_general(vid, neg_emb, 64, tsched[i], T, H, W, w, ctx)
        var pred_bf = cfg(pp, pn, Float32(7.5), ctx)
        var pred = cast_tensor(pred_bf, STDtype.F32, ctx)
        x = euler_vlerp_step(pred, x, tsched[i], s_all[i], Float32(1000.0), ctx)
        print("  step", i, "t=", tsched[i], "s=", s_all[i], "done")
    return x^


def main() raises:
    var ctx = DeviceContext()
    print("ctx ok", Int(ctx.id()))

    var txe = SafeTensors.open(TEXT)
    var pos_emb = cast_tensor(load_bias(txe, "pos_emb", ctx), STDtype.BF16, ctx)  # [58,5120]
    var neg_emb = cast_tensor(load_bias(txe, "neg_emb", ctx), STDtype.BF16, ctx)  # [64,5120]

    # ════════════════════════════════════════════════════════════════════════════
    # RUN 1 — VERIFY at 128x128x13  (latent grid Tl=4, Hl=16, Wl=16)
    # ════════════════════════════════════════════════════════════════════════════
    print("\n===== VERIFY 128x128x13 (grid 4,16,16) =====")
    var vae = SafeTensors.open(VAE_W)
    var inp = SafeTensors.open(INPUT)
    var orc = SafeTensors.open(ORACLE)

    var video = load_bias(inp, "real_input", ctx)  # F32 NCDHW [1,3,13,128,128]
    var latent = encode_seedvr2_vae(video, vae, ctx)  # [1,16,4,16,16]
    var cond = latent_to_cond(latent, 4, 16, 16, ctx)  # [1024,17]
    print("cond", cond.shape()[0], cond.shape()[1])

    # shift + schedule (host); expect shift ~0.9326
    var shift_v = compute_shift(4, 16, 16)
    print("verify shift=", shift_v)
    var sched = sampler_timesteps(Float32(1000.0), 8, shift_v)
    var tsched = sched[0].copy()
    var s_all = sched[1].copy()

    # sanity: computed tsched must match the oracle tsched (max diff < 1)
    var orc_ts = load_bias(orc, "tsched", ctx).to_host(ctx)
    var ts_maxd: Float32 = 0.0
    for i in range(8):
        var d = tsched[i] - orc_ts[i]
        if d < 0.0:
            d = -d
        if d > ts_maxd:
            ts_maxd = d
    print("tsched max|computed-oracle|=", ts_maxd)

    var x_T = load_bias(orc, "x_T", ctx)  # F32 [1024,16]
    var x = _run_sampler_general(cond, x_T, 4, 16, 16, pos_emb, neg_emb, tsched, s_all, ctx)

    var lat_out = flat_to_ncdhw(x, 4, 16, 16, ctx)  # [1,16,4,16,16]
    var decoded = decode_seedvr2_vae(lat_out, vae, ctx)  # [1,3,13,128,128]
    print("decoded", decoded.shape()[0], decoded.shape()[1], decoded.shape()[2],
          decoded.shape()[3], decoded.shape()[4])

    var dec_h = decoded.to_host(ctx)
    var ref_h = load_bias(orc, "decoded", ctx).to_host(ctx)
    var rdec = _cos_maxabs(dec_h, ref_h)
    print("verify decoded cos=", rdec[0], " max_abs=", rdec[1])
    var verify_ok = rdec[0] >= 0.98

    # ════════════════════════════════════════════════════════════════════════════
    # RUN 2 — DEMO at NEW resolution 96x96x5  (latent grid Tl=2, Hl=12, Wl=12)
    # ════════════════════════════════════════════════════════════════════════════
    print("\n===== DEMO 96x96x5 (grid 2,12,12; post-patch 2,6,6) =====")
    # degraded input: first 5 frames of real_input, center-cropped 128->96 in H,W.
    var v5 = slice(video, 2, 0, 5, ctx)        # [1,3,5,128,128]
    var vH = slice(v5, 3, 16, 96, ctx)         # [1,3,5,96,128]
    var vin = slice(vH, 4, 16, 96, ctx)        # [1,3,5,96,96]
    print("demo input", vin.shape()[0], vin.shape()[1], vin.shape()[2],
          vin.shape()[3], vin.shape()[4])

    var latent2 = encode_seedvr2_vae(vin, vae, ctx)  # [1,16,2,12,12]
    print("demo latent", latent2.shape()[0], latent2.shape()[1], latent2.shape()[2],
          latent2.shape()[3], latent2.shape()[4])
    var cond2 = latent_to_cond(latent2, 2, 12, 12, ctx)  # [288,17]

    var shift2 = compute_shift(2, 12, 12)
    print("demo shift=", shift2)
    var sched2 = sampler_timesteps(Float32(1000.0), 8, shift2)
    var tsched2 = sched2[0].copy()
    var s_all2 = sched2[1].copy()

    # deterministic x_T noise [288,16]
    var Ld = 2 * 12 * 12
    var noise = gen_noise(Ld * 16, UInt64(0x5EED2C0FFEE1))
    var noise_shape = List[Int]()
    noise_shape.append(Ld); noise_shape.append(16)
    var x_T2 = Tensor.from_host(noise, noise_shape^, STDtype.F32, ctx)

    var x2 = _run_sampler_general(cond2, x_T2, 2, 12, 12, pos_emb, neg_emb, tsched2, s_all2, ctx)

    var x2_h = x2.to_host(ctx)
    var mm = _minmax(x2_h)
    print("demo sampler latent range: min=", mm[0], " max=", mm[1])

    var lat_out2 = flat_to_ncdhw(x2, 2, 12, 12, ctx)  # [1,16,2,12,12]
    var decoded2 = decode_seedvr2_vae(lat_out2, vae, ctx)  # [1,3,5,96,96]
    print("demo decoded", decoded2.shape()[0], decoded2.shape()[1], decoded2.shape()[2],
          decoded2.shape()[3], decoded2.shape()[4])
    var dec2_h = decoded2.to_host(ctx)
    var mm2 = _minmax(dec2_h)
    print("demo decoded range: min=", mm2[0], " max=", mm2[1])

    var frame_shape = List[Int]()
    frame_shape.append(1); frame_shape.append(3); frame_shape.append(96); frame_shape.append(96)
    for fidx in range(5):
        var fr = slice(decoded2, 2, fidx, 1, ctx)          # [1,3,1,96,96]
        var fr2d = reshape(fr, frame_shape.copy(), ctx)    # [1,3,96,96]
        var out_path = String("/tmp/seedvr2_general_out_f") + String(fidx) + String(".png")
        save_png(fr2d, out_path, ctx, ValueRange.SIGNED)
        print("  wrote", out_path)

    var demo_ran = decoded2.shape()[2] == 5

    print("\n===== SUMMARY =====")
    print("verify decoded cos=", rdec[0], " (>=0.98:", verify_ok, ")")
    print("demo ran:", demo_ran)
    if verify_ok and demo_ran:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
