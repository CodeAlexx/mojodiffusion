# moe_superres_probe.mojo — SUPER-RES refiner (#4): base MoE t2v at LOW res ->
# decode -> 2x bicubic upscale -> temporal VAE encode -> refine at HIGH res.
#
# Composes verified pieces (do NOT reimplement any):
#   * lingbot_t2i_generate      — base t2v denoise (LOW res)
#   * LingBotWanVaeDecoder.decode_video — latent -> pixels (both LOW + HIGH)
#   * resize_video_bicubic      — PIL-exact per-frame bicubic upscale (2x)
#   * LingBotWanVaeEncoder.encode_video — pixels -> normalized latent (HIGH res)
#   * lingbot_refine_generate   — re-noise x_up + refine at HIGH res
#
# Creator refs mirrored: utils.py resize_video_tensor (bicubic+clamp) and
# prepare_refiner_latent ((1-t)*x_up + t*noise — folded inside refine_generate).
#
# Bounded smoke (fast e2e, proves the whole path):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   SR_BASE_STEPS=2 SR_REFINE_STEPS=2 LINGBOT_RESIDENT_BLOCKS=8 pixi run mojo run -I . \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa -Xlinker -lm \
#     serenitymojo/models/lingbotvideo/parity/moe_superres_probe.mojo

from std.pathlib import Path
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.env import env_int
from serenitymojo.ops.random import randn
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.lingbotvideo.lingbot_tokenize import tokenize_lingbot_prompt
from serenitymojo.models.text_encoder.lingbot_qwen3vl import load_lingbot_qwen3vl, encode_lingbot_text
from serenitymojo.models.lingbotvideo.backbone import LingBotAttnConfig, LingBotResidentStore
from serenitymojo.models.lingbotvideo.pipeline import lingbot_t2i_generate, lingbot_refine_generate
from serenitymojo.models.lingbotvideo.vae_decoder import LingBotWanVaeDecoder
from serenitymojo.models.lingbotvideo.vae_encoder import LingBotWanVaeEncoder
from serenitymojo.models.lingbotvideo.lingbot_image_preprocess import resize_video_bicubic

comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
comptime BASE_CKPT = "/mnt/disk1/models/lingbot-video-moe/transformer_mxfp4_w2fp8"
comptime REFINER_CKPT = "/mnt/disk1/models/lingbot-video-moe/refiner_mxfp4_w2fp8"
comptime VAE_FILE = "/mnt/disk1/models/lingbot-video-moe/vae/diffusion_pytorch_model.safetensors"
comptime TE_DIR = "/mnt/disk1/models/lingbot-video-moe/text_encoder"
comptime TOK_JSON = "/mnt/disk1/models/lingbot-video-moe/text_encoder/tokenizer.json"
comptime PROMPT_TXT = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity/t2v_prompt.txt"
comptime NEG_TXT = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity/i2v_neg_prompt.txt"

# ── LOW-res base (same geometry as moe_refine_probe) ────────────────────────
comptime GT = 13
comptime GH = 36
comptime GW = 20
comptime LH = 72
comptime LW = 40
comptime L_COND = 607
comptime L_UNCOND = 234
comptime N_VIDEO = GT * GH * GW
comptime S_COND = N_VIDEO + L_COND
comptime S_UNCOND = N_VIDEO + L_UNCOND
comptime OUT_CH = 16
comptime DEPTH = 48
comptime THETA = Float32(256.0)
comptime SHIFT = 3.0
comptime GUIDANCE = Float32(5.0)

# LOW-res decoded pixel geometry: T_px = (GT-1)*4+1, H = 8*LH, W = 8*LW.
comptime T_PX = (GT - 1) * 4 + 1        # 49
comptime PX_H = LH * 8                   # 576
comptime PX_W = LW * 8                   # 320

# ── HIGH-res (2x) geometry ──────────────────────────────────────────────────
# 2x upscale (576x320 -> 1152x640) via the spatial-tiled VAE encode
# (encode_video_tiled, 256px tiles) — the full-frame encode OOMs 16GB, tiling
# keeps each pass <=256x256. (1.5x/864x480 also works with plain encode_video.)
comptime PX_H_HI = 1152                  # == 8*LH_HI
comptime PX_W_HI = 640                   # == 8*LW_HI
comptime LH_HI = 144
comptime LW_HI = 80
comptime GH_HI = 72
comptime GW_HI = 40
comptime N_VIDEO_HI = GT * GH_HI * GW_HI
comptime S_COND_HI = N_VIDEO_HI + L_COND
comptime S_UNCOND_HI = N_VIDEO_HI + L_UNCOND

# ── refine ──────────────────────────────────────────────────────────────────
comptime T_THRESH = 0.85
comptime TAIL = 2

comptime SEED = UInt64(20260716)
comptime RSEED = UInt64(777)


def _vae_mean() -> List[Float32]:
    var m = List[Float32]()
    for v in [Float32(-0.7571), Float32(-0.7089), Float32(-0.9113), Float32(0.1075),
              Float32(-0.1745), Float32(0.9653), Float32(-0.1517), Float32(1.5508),
              Float32(0.4134), Float32(-0.0715), Float32(0.5517), Float32(-0.3632),
              Float32(-0.1922), Float32(-0.9497), Float32(0.2503), Float32(-0.2921)]:
        m.append(v)
    return m^


def _vae_std() -> List[Float32]:
    var s = List[Float32]()
    for v in [Float32(2.8184), Float32(1.4541), Float32(2.3275), Float32(2.6558),
              Float32(1.2196), Float32(1.7708), Float32(2.6052), Float32(2.0743),
              Float32(3.2687), Float32(2.1526), Float32(2.8652), Float32(1.5579),
              Float32(1.6382), Float32(1.1253), Float32(2.8251), Float32(1.9160)]:
        s.append(v)
    return s^


# Denormalize a normalized latent (per-channel *std + mean) for decode.
# `chstride` = GT * lh * lw (elements per channel).
def _denorm(latent_host: List[Float32], chstride: Int) -> List[Float32]:
    var vmean = _vae_mean()
    var vstd = _vae_std()
    var zhost = List[Float32]()
    zhost.resize(len(latent_host), Float32(0.0))
    for j in range(len(latent_host)):
        var ch = (j // chstride) % OUT_CH
        zhost[j] = latent_host[j] * vstd[ch] + vmean[ch]
    return zhost^


# Decode LOW-res latent -> pixel clip [1,3,T_PX,PX_H,PX_W] F32 in [0,1].
def _decode_lowres(latent_host: List[Float32], ctx: DeviceContext) raises -> Tensor:
    var zhost = _denorm(latent_host, GT * LH * LW)
    var z = Tensor.from_host(zhost, [1, OUT_CH, GT, LH, LW], STDtype.F32, ctx)
    var dec = LingBotWanVaeDecoder[LH, LW].load(VAE_FILE, ctx)
    var pr = dec.decode_video(z, ctx)   # [1,3,T_PX,PX_H,PX_W] in [-1,1]
    var ps = pr.shape()
    var raw = pr.to_host(ctx)
    var px = List[Float32]()
    px.resize(len(raw), Float32(0.0))
    for i in range(len(raw)):
        px[i] = (raw[i] + Float32(1.0)) * Float32(0.5)
    return Tensor.from_host(px, ps^, STDtype.F32, ctx)


# Decode HIGH-res latent -> pixel clip [1,3,T_PX,PX_H_HI,PX_W_HI] in [0,1].
def _decode_highres(latent_host: List[Float32], ctx: DeviceContext) raises -> Tensor:
    var zhost = _denorm(latent_host, GT * LH_HI * LW_HI)
    var z = Tensor.from_host(zhost, [1, OUT_CH, GT, LH_HI, LW_HI], STDtype.F32, ctx)
    var dec = LingBotWanVaeDecoder[LH_HI, LW_HI].load(VAE_FILE, ctx)
    var pr = dec.decode_video(z, ctx)
    var ps = pr.shape()
    var raw = pr.to_host(ctx)
    var px = List[Float32]()
    px.resize(len(raw), Float32(0.0))
    for i in range(len(raw)):
        px[i] = (raw[i] + Float32(1.0)) * Float32(0.5)
    return Tensor.from_host(px, ps^, STDtype.F32, ctx)


def _save_pixels(pix: Tensor, name: String, ctx: DeviceContext) raises:
    var names = List[String]()
    names.append(String("pixels"))
    var p = pix.clone(ctx)
    var tens = List[ArcPointer[Tensor]]()
    tens.append(ArcPointer(p^))
    save_safetensors(names, tens, String(PARITY_DIR) + "/" + name, ctx)
    print("[SR] saved", name)


def _stats(name: String, t: Tensor, ctx: DeviceContext) raises:
    var h = t.to_host(ctx)
    var s = Float64(0.0)
    var s2 = Float64(0.0)
    var amax = Float32(0.0)
    for i in range(len(h)):
        var v = h[i]
        s += Float64(v)
        s2 += Float64(v) * Float64(v)
        var a = v if v >= 0.0 else -v
        if a > amax:
            amax = a
    var n = Float64(len(h))
    var mean = s / n
    var var_ = s2 / n - mean * mean
    var std = var_ ** 0.5 if var_ > 0.0 else 0.0
    print("[SR]", name, "mean", Float32(mean), "std", Float32(std), "absmax", amax)


def main() raises:
    var ctx = DeviceContext()
    var cfg = LingBotAttnConfig.default()
    var base_steps = env_int(String("SR_BASE_STEPS"), 30)
    var refine_steps = env_int(String("SR_REFINE_STEPS"), 15)
    var n_res = env_int(String("LINGBOT_RESIDENT_BLOCKS"), 8)

    # 1. tokenize + encode prompt (shared base + refiner)
    print("[SR] tokenize + encode")
    var tok = Qwen3Tokenizer(String(TOK_JSON))
    var pr = tokenize_lingbot_prompt(tok, Path(String(PROMPT_TXT)).read_text())
    var ng = tokenize_lingbot_prompt(tok, Path(String(NEG_TXT)).read_text())
    var enc = load_lingbot_qwen3vl(String(TE_DIR), ctx)
    var prompt_embeds = encode_lingbot_text(enc, pr[0], pr[1], ctx)
    var neg_embeds = encode_lingbot_text(enc, ng[0], ng[1], ctx)
    _ = enc^

    # 2. BASE t2v at LOW res
    print("[SR] BASE t2v LOW-res (", base_steps, " steps )")
    var init = randn([1, OUT_CH, GT, LH, LW], SEED, STDtype.F32, ctx)
    var base_model = ShardedSafeTensors.open(String(BASE_CKPT))
    var base_store = LingBotResidentStore.load(base_model, n_res, ctx)
    var base_out = lingbot_t2i_generate[S_COND, S_UNCOND](
        init, prompt_embeds, neg_embeds, base_model,
        GT, GH, GW, L_COND, L_UNCOND, 1, 2, 2, OUT_CH, DEPTH, THETA,
        base_steps, SHIFT, GUIDANCE, cfg, base_store, ctx,
    )
    var base_latent_host = base_out.final_latent.copy()
    _ = base_store^
    _ = base_model^

    # decode LOW-res -> pixel clip [1,3,49,576,320] and save
    var low_clip = _decode_lowres(base_latent_host, ctx)
    var lcs = low_clip.shape()
    print("[SR] low_clip shape [", lcs[0], lcs[1], lcs[2], lcs[3], lcs[4], "]")
    _save_pixels(low_clip, String("moe_superres_base_pixels.safetensors"), ctx)

    # 3. 2x bicubic upscale -> [1,3,49,1152,640]  (oh=HEIGHT=8*LH_HI, ow=WIDTH=8*LW_HI)
    print("[SR] bicubic upscale ->", PX_H_HI, "x", PX_W_HI)
    var up_clip = resize_video_bicubic(low_clip, PX_H_HI, PX_W_HI, ctx)
    var ucs = up_clip.shape()
    print("[SR] up_clip shape [", ucs[0], ucs[1], ucs[2], ucs[3], ucs[4], "]")

    # 4. temporal VAE encode at HIGH res -> x_up [1,16,13,144,80]
    print("[SR] VAE encode_video_tiled HIGH-res (256px spatial tiles)")
    var venc = LingBotWanVaeEncoder[PX_H_HI, PX_W_HI].load(VAE_FILE, ctx)
    var x_up = venc.encode_video_tiled(up_clip, ctx)
    var xs = x_up.shape()
    print("[SR] x_up shape [", xs[0], xs[1], xs[2], xs[3], xs[4], "]")
    _stats(String("x_up"), x_up, ctx)
    _ = venc^

    # 5. REFINE at HIGH res
    print("[SR] REFINER pass HIGH-res (", refine_steps, " steps, t_thresh=", T_THRESH, ")")
    var noise = randn([1, OUT_CH, GT, LH_HI, LW_HI], RSEED, STDtype.F32, ctx)
    var ref_model = ShardedSafeTensors.open(String(REFINER_CKPT))
    var ref_store = LingBotResidentStore.load(ref_model, n_res, ctx)
    var ref_out = lingbot_refine_generate[S_COND_HI, S_UNCOND_HI](
        x_up, noise, prompt_embeds, neg_embeds, ref_model,
        GT, GH_HI, GW_HI, L_COND, L_UNCOND, 1, 2, 2, OUT_CH, DEPTH, THETA,
        refine_steps, SHIFT, GUIDANCE, T_THRESH, TAIL, cfg, ref_store, ctx,
    )
    _ = ref_store^
    _ = ref_model^

    # 6. decode refined HIGH-res latent -> pixels and save
    var hi_clip = _decode_highres(ref_out.final_latent.copy(), ctx)
    var hcs = hi_clip.shape()
    print("[SR] refined hi_clip shape [", hcs[0], hcs[1], hcs[2], hcs[3], hcs[4], "]")
    _save_pixels(hi_clip, String("moe_superres_refined_pixels.safetensors"), ctx)
    print("[SR] DONE — base + refined saved")
