# lingbot_t2v_mojo.mojo — FULLY PURE-MOJO text-to-video (typed prompt, no Python).
#
# Same pure-Mojo path as lingbot_i2v_mojo but WITHOUT image conditioning: a typed
# prompt (text file) -> Qwen3Tokenizer -> Qwen3-VL text encode -> quantized-MoE
# denoise (lingbot_t2i_generate, grid_t>1) -> temporal VAE decode. Proves t2v takes
# arbitrary typed prompts natively (no captured embeds).
#
# Run:
#   LINGBOT_CKPT=/mnt/disk1/models/lingbot-video-moe/transformer_mxfp4_w2fp8 \
#   LINGBOT_RESIDENT_BLOCKS=8 pixi run mojo run -I . \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa -Xlinker -lm \
#     serenitymojo/models/lingbotvideo/lingbot_t2v_mojo.mojo

from std.pathlib import Path
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.env import env_or, env_int
from serenitymojo.ops.random import randn
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.lingbotvideo.lingbot_tokenize import tokenize_lingbot_prompt
from serenitymojo.models.text_encoder.lingbot_qwen3vl import (
    load_lingbot_qwen3vl, encode_lingbot_text,
)
from serenitymojo.models.lingbotvideo.backbone import (
    LingBotAttnConfig, LingBotResidentStore,
)
from serenitymojo.models.lingbotvideo.pipeline import lingbot_t2i_generate
from serenitymojo.models.lingbotvideo.vae_decoder import LingBotWanVaeDecoder

comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
comptime MODEL_DIR_DEFAULT = "/mnt/disk1/models/lingbot-video-moe/transformer_mxfp4_w2fp8"
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
comptime L_COND = 607     # set to runtime print on first run
comptime L_UNCOND = 234
comptime N_VIDEO = GT * GH * GW
comptime S_COND = N_VIDEO + L_COND
comptime S_UNCOND = N_VIDEO + L_UNCOND
comptime OUT_CH = 16
comptime DEPTH = 48
comptime THETA = Float32(256.0)
comptime NUM_STEPS = 30
comptime SHIFT = 3.0
comptime GUIDANCE = Float32(5.0)
comptime SEED = UInt64(20260714)


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


def main() raises:
    var ctx = DeviceContext()
    var cfg = LingBotAttnConfig.default()

    print("[T2V-MOJO] tokenize (Qwen3Tokenizer)")
    var tok = Qwen3Tokenizer(String(TOK_JSON))
    var pr = tokenize_lingbot_prompt(tok, Path(String(PROMPT_TXT)).read_text())
    var ng = tokenize_lingbot_prompt(tok, Path(String(NEG_TXT)).read_text())
    var p_ids = pr[0].copy()
    var p_crop = pr[1]
    var n_ids = ng[0].copy()
    var n_crop = ng[1]
    print("[T2V-MOJO] L_COND(runtime) =", len(p_ids) - p_crop, " L_UNCOND =", len(n_ids) - n_crop)
    if (len(p_ids) - p_crop) != L_COND or (len(n_ids) - n_crop) != L_UNCOND:
        raise Error(String("recompile with L_COND=") + String(len(p_ids) - p_crop)
                    + " L_UNCOND=" + String(len(n_ids) - n_crop))

    print("[T2V-MOJO] Qwen3-VL text encode")
    var te = load_lingbot_qwen3vl(String(TE_DIR), ctx)
    var prompt_embeds = encode_lingbot_text(te, p_ids, p_crop, ctx)
    var neg_embeds = encode_lingbot_text(te, n_ids, n_crop, ctx)
    _ = te^

    var init_latent = randn([1, OUT_CH, GT, LH, LW], SEED, STDtype.F32, ctx)
    var model_dir = env_or(String("LINGBOT_CKPT"), String(MODEL_DIR_DEFAULT))
    var model = ShardedSafeTensors.open(model_dir)
    var n_res = env_int(String("LINGBOT_RESIDENT_BLOCKS"), 8)
    var store = LingBotResidentStore.load(model, n_res, ctx) if n_res > 0 else LingBotResidentStore.empty()
    print("[T2V-MOJO] t2v denoise grid_t =", GT, " S_cond =", S_COND, " steps =", NUM_STEPS)
    var res = lingbot_t2i_generate[S_COND, S_UNCOND](
        init_latent, prompt_embeds, neg_embeds, model,
        GT, GH, GW, L_COND, L_UNCOND, 1, 2, 2, OUT_CH, DEPTH, THETA,
        NUM_STEPS, SHIFT, GUIDANCE, cfg, store, ctx,
    )

    var chstride = GT * LH * LW
    var vmean = _vae_mean()
    var vstd = _vae_std()
    var zhost = List[Float32]()
    zhost.resize(len(res.final_latent), Float32(0.0))
    for j in range(len(res.final_latent)):
        var ch = (j // chstride) % OUT_CH
        zhost[j] = res.final_latent[j] * vstd[ch] + vmean[ch]
    var z = Tensor.from_host(zhost, [1, OUT_CH, GT, LH, LW], STDtype.F32, ctx)

    print("[T2V-MOJO] temporal VAE decode")
    var dec = LingBotWanVaeDecoder[LH, LW].load(VAE_FILE, ctx)
    var pixels_raw = dec.decode_video(z, ctx)
    var ps = pixels_raw.shape()
    var raw = pixels_raw.to_host(ctx)
    var px01 = List[Float32]()
    px01.resize(len(raw), Float32(0.0))
    for i in range(len(raw)):
        px01[i] = (raw[i] + Float32(1.0)) * Float32(0.5)
    var pix = Tensor.from_host(px01, ps.copy(), STDtype.F32, ctx)
    var names = List[String]()
    names.append(String("pixels"))
    var tens = List[ArcPointer[Tensor]]()
    tens.append(ArcPointer(pix^))
    save_safetensors(names, tens, String(PARITY_DIR) + "/lingbot_t2v_mojo_pixels.safetensors", ctx)
    print("[T2V-MOJO] SAVED lingbot_t2v_mojo_pixels.safetensors — typed prompt, NO PYTHON")
