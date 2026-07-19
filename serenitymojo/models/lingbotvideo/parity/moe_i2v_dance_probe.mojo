# moe_i2v_dance_probe.mojo — custom i2v: uploaded cyborg image + dance prompt.
#
# Full native path: VAE-encode the condition image + Mojo Qwen3-VL text-encode the
# dance prompt (no torch capture) -> latent-seed i2v -> temporal decode. Geometry
# + comptime L come from dance_meta.json (H512 W352, L_COND=58, L_UNCOND=234).
#
# Run WITH cshim flags (S~9210 >= flash):
#   LINGBOT_CKPT=/mnt/disk1/models/lingbot-video-moe/transformer_mxfp4_w2fp8 \
#   LINGBOT_RESIDENT_BLOCKS=20 pixi run mojo run -I . \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa -Xlinker -lm \
#     serenitymojo/models/lingbotvideo/parity/moe_i2v_dance_probe.mojo

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.env import env_or, env_int
from serenitymojo.ops.random import randn
from serenitymojo.models.text_encoder.lingbot_qwen3vl import (
    load_lingbot_qwen3vl, encode_lingbot_text,
)
from serenitymojo.models.lingbotvideo.backbone import (
    LingBotAttnConfig, LingBotResidentStore,
)
from serenitymojo.models.lingbotvideo.pipeline import lingbot_i2v_generate
from serenitymojo.models.lingbotvideo.vae_encoder import LingBotWanVaeEncoder
from serenitymojo.models.lingbotvideo.vae_decoder import LingBotWanVaeDecoder

comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
comptime MODEL_DIR_DEFAULT = "/mnt/disk1/models/lingbot-video-moe/transformer_mxfp4_w2fp8"
comptime VAE_FILE = "/mnt/disk1/models/lingbot-video-moe/vae/diffusion_pytorch_model.safetensors"
comptime TE_DIR = "/mnt/disk1/models/lingbot-video-moe/text_encoder"

comptime H = 576
comptime W = 320
comptime GT = 31          # 5s @ 24fps -> 121 output frames ((31-1)*4+1)
comptime GH = 36
comptime GW = 20
comptime LH = 72
comptime LW = 40
comptime L_COND = 607      # dance prompt post-crop (structured caption, dance_meta.json)
comptime L_UNCOND = 234    # DEFAULT_NEGATIVE_PROMPT post-crop
comptime N_VIDEO = GT * GH * GW
comptime S_COND = N_VIDEO + L_COND
comptime S_UNCOND = N_VIDEO + L_UNCOND
comptime OUT_CH = 16
comptime DEPTH = 48
comptime THETA = Float32(256.0)
comptime NUM_STEPS = 30
comptime SHIFT = 3.0
comptime GUIDANCE = Float32(5.0)
comptime SEED = UInt64(20260712)


def _load_f32(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_view_as_f32(st.tensor_view(name), ctx)


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

    # ── condition image -> VAE encode ────────────────────────────────────────
    print("[DANCE] VAE-encode condition image (", H, "x", W, ")")
    var cimg = ShardedSafeTensors.open(String(PARITY_DIR) + "/dance_condition.safetensors")
    var image = _load_f32(cimg, "image", ctx)              # [1,3,1,H,W]
    var enc_vae = LingBotWanVaeEncoder[H, W].load(VAE_FILE, ctx)
    var cond_latent = enc_vae.encode(image, ctx)           # [1,16,1,LH,LW]
    _ = enc_vae^                                           # free VAE encoder weights

    # ── Mojo Qwen3-VL text encode (prompt + negative) ────────────────────────
    print("[DANCE] Mojo Qwen3-VL encode dance prompt + negative")
    var ids_st = ShardedSafeTensors.open(String(PARITY_DIR) + "/dance_ids.safetensors")
    var p_ids = _read_i32(ids_st, "prompt_ids")
    var n_ids = _read_i32(ids_st, "neg_ids")
    var crop_start = _read_i32(ids_st, "crop_start")[0]
    var te = load_lingbot_qwen3vl(String(TE_DIR), ctx)
    var prompt_embeds = encode_lingbot_text(te, p_ids, crop_start, ctx)   # [1,L_COND,2560]
    var neg_embeds = encode_lingbot_text(te, n_ids, crop_start, ctx)      # [1,L_UNCOND,2560]
    _ = te^   # free the 8GB Qwen3-VL weights before the resident store + denoise (VRAM)
    var pe = prompt_embeds.shape()
    var ne = neg_embeds.shape()
    print("[DANCE] prompt_embeds", pe[0], pe[1], pe[2], " neg_embeds", ne[0], ne[1], ne[2])

    # ── i2v denoise ──────────────────────────────────────────────────────────
    var init_latent = randn([1, OUT_CH, GT, LH, LW], SEED, STDtype.F32, ctx)
    var model_dir = env_or(String("LINGBOT_CKPT"), String(MODEL_DIR_DEFAULT))
    var model = ShardedSafeTensors.open(model_dir)
    var n_res = env_int(String("LINGBOT_RESIDENT_BLOCKS"), 20)
    var store = LingBotResidentStore.load(model, n_res, ctx) if n_res > 0 else LingBotResidentStore.empty()
    print("[DANCE] i2v denoise grid_t =", GT, " S_cond =", S_COND, " steps =", NUM_STEPS)
    var res = lingbot_i2v_generate[S_COND, S_UNCOND](
        init_latent, cond_latent, prompt_embeds, neg_embeds, model,
        GT, GH, GW, L_COND, L_UNCOND, 1, 2, 2, OUT_CH, DEPTH, THETA,
        NUM_STEPS, SHIFT, GUIDANCE, cfg, store, ctx,
    )

    # ── denorm + temporal decode ─────────────────────────────────────────────
    var chstride = GT * LH * LW
    var vmean = _vae_mean()
    var vstd = _vae_std()
    var zhost = List[Float32]()
    zhost.resize(len(res.final_latent), Float32(0.0))
    for j in range(len(res.final_latent)):
        var ch = (j // chstride) % OUT_CH
        zhost[j] = res.final_latent[j] * vstd[ch] + vmean[ch]
    var z = Tensor.from_host(zhost, [1, OUT_CH, GT, LH, LW], STDtype.F32, ctx)

    print("[DANCE] temporal VAE decode")
    var dec = LingBotWanVaeDecoder[LH, LW].load(VAE_FILE, ctx)
    var pixels_raw = dec.decode_video(z, ctx)
    var ps = pixels_raw.shape()
    print("[DANCE] pixels", ps[0], ps[1], ps[2], ps[3], ps[4])

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
    save_safetensors(names, tens, String(PARITY_DIR) + "/moe_i2v_dance_pixels.safetensors", ctx)
    print("[DANCE] SAVED moe_i2v_dance_pixels.safetensors [0,1]")
