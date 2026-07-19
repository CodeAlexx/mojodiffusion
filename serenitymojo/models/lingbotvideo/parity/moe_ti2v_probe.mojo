# moe_ti2v_probe.mojo — TI2V: text+image conditioning (VLM-grounded).
#
# The image is used TWICE (creator ti2v): (a) fed through the Qwen3-VL VISION tower
# and FUSED into the prompt embeds (semantic grounding), and (b) VAE-encoded as the
# clean frame-0 latent seed (appearance anchor). Uses the ported vision tower +
# fusion + the latent-seed i2v. Comptime from ti2v_meta.json: vision_seq=1792
# (grid 1x56x32), L_COND=1057, L_UNCOND=234.
#
# Prep: prep_ti2v.py (processor image-template tokenization -> ti2v_inputs.safetensors).
# Run WITH cshim flags (S~10.4K >= flash):
#   LINGBOT_CKPT=.../transformer_mxfp4_w2fp8 LINGBOT_RESIDENT_BLOCKS=8 pixi run mojo run -I . \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa -Xlinker -lm \
#     serenitymojo/models/lingbotvideo/parity/moe_ti2v_probe.mojo

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.env import env_or, env_int
from serenitymojo.ops.random import randn
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer, _read_utf8_file
from serenitymojo.models.text_encoder.lingbot_qwen3vl import load_lingbot_qwen3vl, encode_lingbot_text
from serenitymojo.models.text_encoder.lingbot_qwen3vl_vision import Qwen3VLVisionModel
from serenitymojo.models.text_encoder.lingbot_qwen3vl_fuse import encode_lingbot_image_text
from serenitymojo.models.lingbotvideo.lingbot_image_preprocess import load_condition_image
from serenitymojo.models.lingbotvideo.lingbot_vision_preprocess import lingbot_vision_preprocess
from serenitymojo.models.lingbotvideo.lingbot_tokenize import (
    tokenize_lingbot_image_prompt,
    tokenize_lingbot_prompt,
    LINGBOT_DEFAULT_NEGATIVE_PROMPT,
)
from serenitymojo.models.lingbotvideo.backbone import LingBotAttnConfig, LingBotResidentStore
from serenitymojo.models.lingbotvideo.pipeline import lingbot_i2v_generate
from serenitymojo.models.lingbotvideo.vae_encoder import LingBotWanVaeEncoder
from serenitymojo.models.lingbotvideo.vae_decoder import LingBotWanVaeDecoder

comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
comptime MODEL_DIR_DEFAULT = "/mnt/disk1/models/lingbot-video-moe/transformer_mxfp4_w2fp8"
comptime VAE_FILE = "/mnt/disk1/models/lingbot-video-moe/vae/diffusion_pytorch_model.safetensors"
comptime TE_DIR = "/mnt/disk1/models/lingbot-video-moe/text_encoder"
comptime TOK_JSON = "/mnt/disk1/models/lingbot-video-moe/text_encoder/tokenizer.json"
comptime IMG_PATH = "/home/alex/.claude/uploads/6fd4828e-ac2b-4c91-a10b-c346b1b61e18/8280dc5c-1000004560.webp"

comptime VS = 1792        # vision patch seq (grid_t*h*w = 1*56*32)
comptime VGT = 1
comptime VGH = 56
comptime VGW = 32
comptime H = 576
comptime W = 320
comptime GT = 13
comptime GH = 36
comptime GW = 20
comptime LH = 72
comptime LW = 40
comptime L_COND = 1057    # fused (image tokens + text) post-crop
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
comptime SEED = UInt64(20260715)


def _read_i32(st: ShardedSafeTensors, name: String) raises -> List[Int]:
    var tv = st.tensor_view(name)
    var p = tv.data.unsafe_ptr().bitcast[Int32]()
    var out = List[Int]()
    for i in range(tv.numel()):
        out.append(Int(p[i]))
    return out^


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

    # ── PURE-MOJO ti2v prep: vision preprocess + image-template tokenization ──
    # (replaces prep_ti2v.py — no Python in the conditioning path). Parity-gated
    # by lingbot_ti2v_prep_probe (ids EXACT, pixel_values cos 0.99999).
    print("[TI2V] pure-Mojo prep: vision preprocess + image-template tokenize")
    var tok = Qwen3Tokenizer(String(TOK_JSON))
    var vprep = lingbot_vision_preprocess(String(IMG_PATH), ctx)
    if vprep.grid_t != VGT or vprep.grid_h != VGH or vprep.grid_w != VGW:
        raise Error(String("vision grid mismatch: got ") + String(vprep.grid_t)
            + String("x") + String(vprep.grid_h) + String("x") + String(vprep.grid_w))
    var n_image = vprep.seq // 4
    var prompt = _read_utf8_file(String(PARITY_DIR) + "/i2v_prompt.txt")
    var tok_res = tokenize_lingbot_image_prompt(tok, prompt, n_image)
    var ids = tok_res[0].copy()
    var crop_start = tok_res[1]
    var neg_ids = tokenize_lingbot_prompt(tok, String(LINGBOT_DEFAULT_NEGATIVE_PROMPT))[0].copy()
    _ = tok^
    # comptime template params (S_COND/S_UNCOND) assume these fixed lengths.
    if (len(ids) - crop_start) != L_COND:
        raise Error(String("L_COND mismatch: got ") + String(len(ids) - crop_start)
            + String(" want ") + String(L_COND))
    if (len(neg_ids) - crop_start) != L_UNCOND:
        raise Error(String("L_UNCOND mismatch: got ") + String(len(neg_ids) - crop_start)
            + String(" want ") + String(L_UNCOND))

    print("[TI2V] load vision tower + text encoder; fuse image+text")
    var vision = Qwen3VLVisionModel.load(String(TE_DIR), ctx)
    var enc = load_lingbot_qwen3vl(String(TE_DIR), ctx)
    var prompt_embeds = encode_lingbot_image_text[VS](
        vision, enc, ids, VGT, VGH, VGW, vprep.pixel_values, crop_start, ctx, True
    )  # [1,L_COND,2560] — f32_stream for stability
    var neg_embeds = encode_lingbot_text(enc, neg_ids, crop_start, ctx)  # text-only
    _ = enc^
    _ = vision^
    var pe = prompt_embeds.shape()
    print("[TI2V] prompt_embeds", pe[0], pe[1], pe[2])

    # ── VAE-encode the image as the frame-0 seed (appearance anchor) ─────────
    var image = load_condition_image(String(IMG_PATH), H, W, ctx)  # [1,3,1,H,W]
    var enc_vae = LingBotWanVaeEncoder[H, W].load(VAE_FILE, ctx)
    var cond_latent = enc_vae.encode(image, ctx)
    _ = enc_vae^

    # ── i2v denoise with the grounded embeds + frame-0 seed ──────────────────
    var init_latent = randn([1, OUT_CH, GT, LH, LW], SEED, STDtype.F32, ctx)
    var model_dir = env_or(String("LINGBOT_CKPT"), String(MODEL_DIR_DEFAULT))
    var model = ShardedSafeTensors.open(model_dir)
    var n_res = env_int(String("LINGBOT_RESIDENT_BLOCKS"), 8)
    var store = LingBotResidentStore.load(model, n_res, ctx) if n_res > 0 else LingBotResidentStore.empty()
    print("[TI2V] denoise grid_t =", GT, " S_cond =", S_COND, " steps =", NUM_STEPS)
    var res = lingbot_i2v_generate[S_COND, S_UNCOND](
        init_latent, cond_latent, prompt_embeds, neg_embeds, model,
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

    print("[TI2V] temporal VAE decode")
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
    save_safetensors(names, tens, String(PARITY_DIR) + "/moe_ti2v_pixels.safetensors", ctx)
    print("[TI2V] SAVED moe_ti2v_pixels.safetensors — VLM-grounded (vision tokens + frame-0 seed)")
