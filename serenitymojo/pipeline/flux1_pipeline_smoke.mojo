# flux1_pipeline_smoke.mojo - FLUX.1 Dev end-to-end wiring smoke (COMPILE-ONLY
# for now; GPU wedged). Glues:
#   1. CLIP-L  -> pooled [1,768]          (vector_in / CLIP pooled conditioning)
#   2. T5-XXL  -> hidden [1,512,4096]     (txt conditioning)
#   3. FLUX.1  -> N-step flow-match Euler denoise (Flux1Offloaded, guidance-distilled,
#                 NO CFG: single forward per step, guidance scalar fed as a model input)
#   4. VAE     -> AutoencoderKL decode (Flux1LdmDecoder: LDM/BFL-format ae.safetensors,
#                 16ch, 8x, scale 0.3611, shift 0.1159, no post_quant_conv) -> [1,3,H,W]
#   5. PNG
#
# Mirrors the end-to-end convention in
#   /home/alex/EriDiffusion/inference-flame/src/bin/flux1_infer.rs (read FULL)
# and the FLUX.1 schedule/pack in src/sampling/flux1_sampling.rs.
#
# Conventions taken from the references (NOT inferred):
#   - guidance-distilled: guidance_vec = full((B,), guidance); fed to DiT as a
#     conditioning input. No classifier-free guidance, no negative prompt.
#   - timestep schedule: linspace(1,0,N+1) then BFL time_shift with LINEAR mu
#     (0.5 @ 256 tokens, 1.15 @ 4096 tokens). t[0]=1, t[-1]=0.
#   - Euler step: img = img + (t_prev - t_curr) * pred.  (flux1_denoise)
#   - timestep_embedding scales t by time_factor=1000 INSIDE the sinusoid; the
#     foundation t_embedder does NOT, so we pre-scale t (and guidance) by 1000.
#   - pack [1,16,H,W] -> [1,(H/2)(W/2),64] via reshape [1,16,h2,2,w2,2] ->
#     permute [0,2,4,1,3,5] -> reshape [1,h2*w2,64]; unpack is the inverse.
#   - VAE latent is [1,16, 2*ceil(H/16), 2*ceil(W/16)]; pack halves each spatial
#     dim -> packed grid h2 = ceil(H/16), w2 = ceil(W/16).
#
# This is the wiring/shape proof; tokenization (CLIP BPE + T5 SentencePiece) and
# parity are NOT yet wired (flagged in the report). Token ids are placeholders.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.math import exp as fexp

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.text_encoder.clip_encoder import ClipEncoder, ClipConfig
from serenitymojo.models.text_encoder.t5_encoder import T5Encoder, T5Config
from serenitymojo.models.dit.flux1_dit import (
    Flux1Config,
    Flux1Offloaded,
    build_flux1_rope_tables,
)
from serenitymojo.models.vae.ldm_decoder import load_flux1_ldm_decoder
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import add, mul_scalar, reshape, permute
from serenitymojo.image.png import save_png, ValueRange


# ── paths (mirror flux1_infer.rs) ────────────────────────────────────────────
comptime CLIP_PATH = "/home/alex/.serenity/models/text_encoders/clip_l.safetensors"
comptime T5_PATH = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.safetensors"
comptime DIT_PATH = "/home/alex/.serenity/models/checkpoints/flux1-dev.safetensors"
comptime VAE_PATH = "/home/alex/.serenity/models/vaes/ae.safetensors"
comptime OUT = "/home/alex/mojodiffusion/output/flux1_first.png"

# ── knobs ─────────────────────────────────────────────────────────────────────
comptime HEIGHT = 1024
comptime WIDTH = 1024
comptime AE_IN_CHANNELS = 16
# latent = [1,16, 2*ceil(H/16), 2*ceil(W/16)] = [1,16,128,128] at 1024.
comptime LATENT_H = 2 * ((HEIGHT + 15) // 16)  # 128
comptime LATENT_W = 2 * ((WIDTH + 15) // 16)   # 128
# packed grid h2 = ceil(H/16), w2 = ceil(W/16); N_IMG = h2*w2.
comptime IMG_H2 = (HEIGHT + 15) // 16  # 64
comptime IMG_W2 = (WIDTH + 15) // 16   # 64
comptime N_IMG = IMG_H2 * IMG_W2       # 4096
comptime N_TXT = 512
comptime S = N_IMG + N_TXT
comptime STEPS = 20
comptime GUIDANCE = Float32(3.5)
comptime SEED = UInt64(42)
comptime CLIP_LEN = 77
comptime T5_LEN = 512


# ── FLUX.1 schedule: linspace(1,0,N+1) + BFL time_shift, LINEAR mu ───────────
# flux1_sampling.rs: mu = m*seq + b with (256,0.5)->(4096,1.15); time_shift =
# exp(mu)/(exp(mu)+(1/t-1)^1). Endpoints t=1 and t=0 left untouched.
def _flux1_mu(image_seq_len: Int) -> Float64:
    var x1 = 256.0
    var y1 = 0.5
    var x2 = 4096.0
    var y2 = 1.15
    var m = (y2 - y1) / (x2 - x1)
    var b = y1 - m * x1
    return m * Float64(image_seq_len) + b


def _time_shift(mu: Float64, t: Float64) -> Float64:
    var em = fexp(mu)
    return em / (em + (1.0 / t - 1.0))


def _flux1_schedule(num_steps: Int, image_seq_len: Int) -> List[Float32]:
    var mu = _flux1_mu(image_seq_len)
    var out = List[Float32]()
    for i in range(num_steps + 1):
        var t = 1.0 - Float64(i) / Float64(num_steps)
        if t > 0.0 and t < 1.0:
            t = _time_shift(mu, t)
        out.append(Float32(t))
    return out^


# ── pack [1,16,H,W] -> [1, h2*w2, 64] (flux1_sampling.rs pack_latent) ────────
def _pack_latent(z_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
    # reshape [1,16,h2,2,w2,2] -> permute [0,2,4,1,3,5] -> reshape [1,h2*w2,64]
    var s6 = List[Int]()
    s6.append(1)
    s6.append(AE_IN_CHANNELS)
    s6.append(IMG_H2)
    s6.append(2)
    s6.append(IMG_W2)
    s6.append(2)
    var t6 = reshape(z_nchw, s6^, ctx)
    var p = List[Int]()
    p.append(0)
    p.append(2)
    p.append(4)
    p.append(1)
    p.append(3)
    p.append(5)
    var tp = permute(t6, p^, ctx)
    var sp = List[Int]()
    sp.append(1)
    sp.append(N_IMG)
    sp.append(AE_IN_CHANNELS * 4)
    return reshape(tp, sp^, ctx)


# ── unpack [1, h2*w2, 64] -> [1,16,2*h2,2*w2] (flux1_sampling.rs unpack_latent) ─
def _unpack_latent(packed: Tensor, ctx: DeviceContext) raises -> Tensor:
    # reshape [1,h2,w2,16,2,2] -> permute [0,3,1,4,2,5] -> reshape [1,16,2*h2,2*w2]
    var s6 = List[Int]()
    s6.append(1)
    s6.append(IMG_H2)
    s6.append(IMG_W2)
    s6.append(AE_IN_CHANNELS)
    s6.append(2)
    s6.append(2)
    var t6 = reshape(packed, s6^, ctx)
    var p = List[Int]()
    p.append(0)
    p.append(3)
    p.append(1)
    p.append(4)
    p.append(2)
    p.append(5)
    var tp = permute(t6, p^, ctx)
    var sp = List[Int]()
    sp.append(1)
    sp.append(AE_IN_CHANNELS)
    sp.append(LATENT_H)
    sp.append(LATENT_W)
    return reshape(tp, sp^, ctx)


# ── placeholder token ids (real CLIP/T5 tokenizers not yet in-tree) ──────────
def _clip_ids() -> List[Int]:
    var ids = List[Int]()
    ids.append(49406)  # BOS
    ids.append(49407)  # EOS
    while len(ids) < CLIP_LEN:
        ids.append(49407)  # CLIP pad == EOS
    return ids^


def _t5_ids() -> List[Int]:
    var ids = List[Int]()
    ids.append(1)  # T5 EOS
    while len(ids) < T5_LEN:
        ids.append(0)  # T5 pad
    return ids^


def main() raises:
    var ctx = DeviceContext()
    print("=== FLUX.1 Dev pipeline smoke ===", HEIGHT, "x", WIDTH, STEPS, "steps")

    # 1. CLIP-L pooled [1,768] (encode_sdxl returns (last_hidden, pooled)).
    print("[clip] CLIP-L pooled")
    var clip = ClipEncoder.load(CLIP_PATH, ClipConfig.clip_l(), ctx)
    var clip_out = clip.encode_sdxl[CLIP_LEN](_clip_ids(), ctx)
    # clip_out = (last_hidden, pooled); FLUX.1 uses only the pooled [1,768]
    # (vector_in). Borrow element 1 directly into the BF16 cast (no move bind).
    var vector = cast_tensor(clip_out[1], STDtype.BF16, ctx)

    # 2. T5-XXL hidden [1,512,4096].
    print("[t5] T5-XXL encode")
    var t5 = T5Encoder[T5_LEN].load(T5_PATH, T5Config.t5_xxl(), ctx)
    var t5_hidden = t5.encode(_t5_ids(), ctx)

    # 3. FLUX.1 denoise (guidance-distilled, no CFG).
    print("[dit] FLUX.1 offloaded DiT")
    var model = Flux1Offloaded.load(DIT_PATH, Flux1Config.dev(), ctx)
    # RoPE tables tiled per-head [S*H, 64].
    var rope = build_flux1_rope_tables[N_IMG, N_TXT, 24, 128](
        IMG_H2, IMG_W2, ctx, STDtype.BF16
    )

    # initial noise NCHW [1,16,LATENT_H,LATENT_W] -> pack -> [1,N_IMG,64].
    var noise_shape = List[Int]()
    noise_shape.append(1)
    noise_shape.append(AE_IN_CHANNELS)
    noise_shape.append(LATENT_H)
    noise_shape.append(LATENT_W)
    var noise_nchw = randn(noise_shape^, SEED, STDtype.F32, ctx)
    var img = _pack_latent(noise_nchw, ctx)

    # txt conditioning (BF16) for the DiT math (CLIP pooled cast above).
    var txt = cast_tensor(t5_hidden, STDtype.BF16, ctx)

    var sched = _flux1_schedule(STEPS, N_IMG)
    print("[denoise]", STEPS, "steps, guidance", GUIDANCE)
    for i in range(STEPS):
        var t_curr = sched[i]
        var t_prev = sched[i + 1]
        # t_vec / guidance_vec are [1], pre-scaled by 1000 (BFL time_factor) since
        # the foundation t_embedder does not apply the 1000x factor internally.
        var tvals = List[Float32]()
        tvals.append(t_curr * 1000.0)
        var tsh = List[Int]()
        tsh.append(1)
        var t_vec = Tensor.from_host(tvals, tsh^, STDtype.F32, ctx)

        var gvals = List[Float32]()
        gvals.append(GUIDANCE * 1000.0)
        var gsh = List[Int]()
        gsh.append(1)
        var g_vec = Tensor.from_host(gvals, gsh^, STDtype.F32, ctx)

        var img_bf = cast_tensor(img, STDtype.BF16, ctx)
        var pred = cast_tensor(
            model.forward[N_IMG, N_TXT, S](
                img_bf, txt, t_vec, Optional[Tensor](g_vec^), vector, rope[0], rope[1], ctx
            ),
            STDtype.F32,
            ctx,
        )
        # img = img + (t_prev - t_curr) * pred
        var dt = t_prev - t_curr
        img = add(img, mul_scalar(pred, dt, ctx), ctx)

    # 4. unpack + VAE decode.
    print("[vae] unpack + decode")
    var latent = _unpack_latent(img, ctx)
    var vae = load_flux1_ldm_decoder[LATENT_H, LATENT_W](VAE_PATH, ctx)
    var rgb = vae.decode(latent, ctx)
    var sh = rgb.shape()
    print("  image:", sh[0], sh[1], sh[2], sh[3])

    # 5. PNG.
    save_png(rgb, OUT, ctx, ValueRange.SIGNED)
    print("[done] saved", OUT)
