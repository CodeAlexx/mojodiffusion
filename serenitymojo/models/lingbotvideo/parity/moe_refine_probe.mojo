# moe_refine_probe.mojo — REFINER (#3): base MoE t2v -> same-res refiner detail pass.
#
# Generates a base clip (base hybrid ckpt), then re-noises its latent to sigma=t_thresh
# and denoises with the REFINER hybrid ckpt on the refiner sigma schedule
# (lingbot_refine_generate). Decodes base + refined -> compare. Same-resolution detail
# refine (super-res upscale needs a temporal VAE encode = follow-up).
#
# Run WITH cshim flags:
#   LINGBOT_RESIDENT_BLOCKS=8 pixi run mojo run -I . \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa -Xlinker -lm \
#     serenitymojo/models/lingbotvideo/parity/moe_refine_probe.mojo

from std.pathlib import Path
from max.gpu.host import DeviceContext
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

comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
comptime BASE_CKPT = "/mnt/disk1/models/lingbot-video-moe/transformer_mxfp4_w2fp8"
comptime REFINER_CKPT = "/mnt/disk1/models/lingbot-video-moe/refiner_mxfp4_w2fp8"
comptime VAE_FILE = "/mnt/disk1/models/lingbot-video-moe/vae/diffusion_pytorch_model.safetensors"
comptime TE_DIR = "/mnt/disk1/models/lingbot-video-moe/text_encoder"
comptime TOK_JSON = "/mnt/disk1/models/lingbot-video-moe/text_encoder/tokenizer.json"
comptime PROMPT_TXT = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity/t2v_prompt.txt"
comptime NEG_TXT = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity/i2v_neg_prompt.txt"

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
comptime BASE_STEPS = 30
comptime SHIFT = 3.0
comptime GUIDANCE = Float32(5.0)
comptime REFINE_STEPS = 15    # truncated to <= t_thresh -> ~a few effective steps
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


def _decode_save(latent_host: List[Float32], name: String, ctx: DeviceContext) raises:
    var chstride = GT * LH * LW
    var vmean = _vae_mean()
    var vstd = _vae_std()
    var zhost = List[Float32]()
    zhost.resize(len(latent_host), Float32(0.0))
    for j in range(len(latent_host)):
        var ch = (j // chstride) % OUT_CH
        zhost[j] = latent_host[j] * vstd[ch] + vmean[ch]
    var z = Tensor.from_host(zhost, [1, OUT_CH, GT, LH, LW], STDtype.F32, ctx)
    var dec = LingBotWanVaeDecoder[LH, LW].load(VAE_FILE, ctx)
    var pr = dec.decode_video(z, ctx)
    var ps = pr.shape()
    var raw = pr.to_host(ctx)
    var px = List[Float32]()
    px.resize(len(raw), Float32(0.0))
    for i in range(len(raw)):
        px[i] = (raw[i] + Float32(1.0)) * Float32(0.5)
    var pix = Tensor.from_host(px, ps.copy(), STDtype.F32, ctx)
    var names = List[String]()
    names.append(String("pixels"))
    var tens = List[ArcPointer[Tensor]]()
    tens.append(ArcPointer(pix^))
    save_safetensors(names, tens, String(PARITY_DIR) + "/" + name, ctx)
    print("[REFINE] saved", name)


def main() raises:
    var ctx = DeviceContext()
    var cfg = LingBotAttnConfig.default()

    # embeds (typed prompt, reused for base + refiner)
    print("[REFINE] tokenize + encode")
    var tok = Qwen3Tokenizer(String(TOK_JSON))
    var pr = tokenize_lingbot_prompt(tok, Path(String(PROMPT_TXT)).read_text())
    var ng = tokenize_lingbot_prompt(tok, Path(String(NEG_TXT)).read_text())
    var enc = load_lingbot_qwen3vl(String(TE_DIR), ctx)
    var prompt_embeds = encode_lingbot_text(enc, pr[0], pr[1], ctx)
    var neg_embeds = encode_lingbot_text(enc, ng[0], ng[1], ctx)
    _ = enc^

    var n_res = env_int(String("LINGBOT_RESIDENT_BLOCKS"), 8)

    # ── BASE t2v ─────────────────────────────────────────────────────────────
    print("[REFINE] BASE t2v (", BASE_STEPS, " steps)")
    var init = randn([1, OUT_CH, GT, LH, LW], SEED, STDtype.F32, ctx)
    var base_model = ShardedSafeTensors.open(String(BASE_CKPT))
    var base_store = LingBotResidentStore.load(base_model, n_res, ctx)
    var base_out = lingbot_t2i_generate[S_COND, S_UNCOND](
        init, prompt_embeds, neg_embeds, base_model,
        GT, GH, GW, L_COND, L_UNCOND, 1, 2, 2, OUT_CH, DEPTH, THETA,
        BASE_STEPS, SHIFT, GUIDANCE, cfg, base_store, ctx,
    )
    var base_latent_host = base_out.final_latent.copy()
    _ = base_store^
    _ = base_model^
    _decode_save(base_latent_host, String("moe_refine_base_pixels.safetensors"), ctx)

    # ── REFINER pass (same res) ──────────────────────────────────────────────
    print("[REFINE] REFINER pass (t_thresh=", T_THRESH, ")")
    var base_latent = Tensor.from_host(base_latent_host, [1, OUT_CH, GT, LH, LW], STDtype.F32, ctx)
    var noise = randn([1, OUT_CH, GT, LH, LW], RSEED, STDtype.F32, ctx)
    var ref_model = ShardedSafeTensors.open(String(REFINER_CKPT))
    var ref_store = LingBotResidentStore.load(ref_model, n_res, ctx)
    var ref_out = lingbot_refine_generate[S_COND, S_UNCOND](
        base_latent, noise, prompt_embeds, neg_embeds, ref_model,
        GT, GH, GW, L_COND, L_UNCOND, 1, 2, 2, OUT_CH, DEPTH, THETA,
        REFINE_STEPS, SHIFT, GUIDANCE, T_THRESH, TAIL, cfg, ref_store, ctx,
    )
    _ = ref_store^
    _ = ref_model^
    _decode_save(ref_out.final_latent.copy(), String("moe_refine_refined_pixels.safetensors"), ctx)
    print("[REFINE] DONE — base + refined saved")
