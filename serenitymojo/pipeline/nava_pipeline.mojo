# nava_pipeline.mojo — NAVA end-to-end: denoise + video decode + audio decode + WAV write.
#
# Faithful implementation of pipeline_nava.py:458-552 sample() loop.
# 3-forward CFG per step (cond, uncond, masking_modality=True) × 25 steps.
# CFG formula (align_3d_cfg=True):
#   eps_v = cond_v + 3.0*(cond_v - unco_v) + 3.0*(cond_v - mmask_v)
#   eps_a = cond_a + 2.0*(cond_a - unco_a) + 2.0*(cond_a - mmask_a)
#
# Memory plan (24 GB GPU):
#   Phase A: NavaDiT (~13 GB) loaded inside _denoise(), drops on return.
#   Phase B: Wan22VaeImageDecoder (~1 GB) + LTX2 audio VAE + vocoder (~2 GB).
#
# Inputs: serenitymojo/pipeline/parity/nava_first_inputs.safetensors
#   pos_text     [122,4096] F32
#   init_lat_vid [1280,48]  F32
#   init_lat_aud [34,128]   F32
#   (neg_text: zeros [122,4096] — unconditioned)
#
# Outputs: /home/alex/mojodiffusion/output/nava_first/
#   frame_00.png .. frame_16.png  (17 frames, 256x256 RGB)
#   audio.wav                      (16 kHz 16-bit stereo PCM)
#
# Run:
#   mkdir -p output/nava_first
#   pixi run mojo run -I . serenitymojo/pipeline/nava_pipeline.mojo
#
# NOTE: 3 forwards/step × 25 steps = 75 forwards total. Expected ~10-30 min.

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import (
    add, sub, mul_scalar, reshape, permute, slice, zeros_device,
)
from serenitymojo.ops.reduce import reduce_std_f32
from serenitymojo.sampling.unipc import UniPcMultistepScheduler
from serenitymojo.models.dit.nava_dit import NavaDiT
from serenitymojo.models.vae.wan22_decoder import Wan22VaeImageDecoder
from serenitymojo.models.vae.ltx2_audio_vae import LTX2AudioVaeDecoderWeights, decode
from serenitymojo.models.vocoder.ltx2_vocoder import LTX2VocoderWithBWE
from serenitymojo.ops.resample import resample_hann
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.audio.wav import save_wav


# ── paths ─────────────────────────────────────────────────────────────────────
comptime DIT_CKPT = "/home/alex/.serenity/models/checkpoints/NAVA/NAVA_fp8.safetensors"
comptime VID_VAE_CKPT = "/home/alex/.serenity/models/checkpoints/NAVA/Wan2.2_VAE.safetensors"
comptime AUD_VAE_CKPT = "/home/alex/.serenity/models/checkpoints/NAVA/params/LTX2/ltx-2.3-22b-dev_audio_vae.safetensors"
comptime INPUTS_FX = "/home/alex/mojodiffusion/serenitymojo/pipeline/parity/nava_first_inputs.safetensors"
comptime OUT_DIR = "/home/alex/mojodiffusion/output/nava_first"

comptime NSTEP = 25
comptime VID_G = Float32(3.0)
comptime AUD_G = Float32(2.0)


# ── 2-field Movable struct for latent pair (Tensor is Movable-not-Copyable) ───
struct LatentPair(Movable):
    var vid: Tensor  # [1280, 48] F32
    var aud: Tensor  # [34, 128]  F32

    def __init__(out self, var v: Tensor, var a: Tensor):
        self.vid = v^
        self.aud = a^


# ── zero-pad integer to 2-digit string ───────────────────────────────────────
def _pad2(n: Int) -> String:
    if n < 10:
        return String("0") + String(n)
    return String(n)


# ── helper: build a [1] F32 timestep tensor from a float value ────────────────
def _make_t(val: Float32, ctx: DeviceContext) raises -> Tensor:
    var t_list = List[Float32]()
    t_list.append(val)
    var t_shape = List[Int]()
    t_shape.append(1)
    return Tensor.from_host(t_list, t_shape^, STDtype.F32, ctx)


# ── Phase A: denoise — NavaDiT lives only inside this def ────────────────────
def _denoise(
    pos_text: Tensor,     # [122,4096] F32 (on device)
    neg_text: Tensor,     # [126,4096] F32 — video negative prompt (uncond)
    init_lat_vid: Tensor, # [1280,48] F32
    init_lat_aud: Tensor, # [34,128]  F32
    ctx: DeviceContext,
) raises -> LatentPair:
    """Run NSTEP-step faithful 3-forward CFG denoise with NavaDiT.

    3 forwards per step (cond, uncond, masking_modality) × NSTEP steps.
    CFG formula (align_3d_cfg=True, vision_guidance_scale=3, audio_guidance_scale=2):
      eps_v = cond_v + 3*(cond_v - unco_v) + 3*(cond_v - mmask_v)
      eps_a = cond_a + 2*(cond_a - unco_a) + 2*(cond_a - mmask_a)
    DiT is freed on return (scoped inside this def).
    """
    print("  Loading NavaDiT...")
    var st = ShardedSafeTensors.open(DIT_CKPT)
    var dit = NavaDiT.load(st, ctx)
    print("  NavaDiT loaded.")

    # Condition: cast pos_text to BF16.
    var pos = cast_tensor(pos_text, STDtype.BF16, ctx)
    # Unconditioned: the video NEGATIVE PROMPT (as sample() uses; NOT zeros).
    var neg = cast_tensor(neg_text, STDtype.BF16, ctx)

    # F32-resident latents (updated each step).
    var lv = init_lat_vid.clone(ctx)
    var la = init_lat_aud.clone(ctx)

    # Two UniPC schedulers — one per latent modality.
    var sv = UniPcMultistepScheduler(1000, NSTEP, 5.0, 2)
    var sa = UniPcMultistepScheduler(1000, NSTEP, 5.0, 2)

    # Pre-compute sigma schedule for timestep derivation.
    var sigmas = sv.sigmas()  # List[Float64] length NSTEP+1, last==0

    for i in range(NSTEP):
        # Timestep = sigma[i] * 1000 (flow-matching convention).
        var t_val = Float32(sigmas[i] * 1000.0)

        # ── Forward 1: cond (masking_modality=False) ──────────────────────────
        var lv_b1 = cast_tensor(lv, STDtype.BF16, ctx)
        var la_b1 = cast_tensor(la, STDtype.BF16, ctx)
        var t1 = _make_t(t_val, ctx)
        var cond = dit.forward(lv_b1, la_b1, pos, t1, ctx, masking_modality=False)

        # ── Forward 2: uncond (masking_modality=False) ────────────────────────
        var lv_b2 = cast_tensor(lv, STDtype.BF16, ctx)
        var la_b2 = cast_tensor(la, STDtype.BF16, ctx)
        var t2 = _make_t(t_val, ctx)
        var unco = dit.forward(lv_b2, la_b2, neg, t2, ctx, masking_modality=False)

        # ── Forward 3: masking_modality=True (cond, non-joint) ───────────────
        var lv_b3 = cast_tensor(lv, STDtype.BF16, ctx)
        var la_b3 = cast_tensor(la, STDtype.BF16, ctx)
        var t3 = _make_t(t_val, ctx)
        var mmask = dit.forward(lv_b3, la_b3, pos, t3, ctx, masking_modality=True)

        # ── CFG in F32 ────────────────────────────────────────────────────────
        # Cast all velocities to F32.
        var cond_v  = cast_tensor(cond.vel_vid,   STDtype.F32, ctx)
        var unco_v  = cast_tensor(unco.vel_vid,   STDtype.F32, ctx)
        var mmask_v = cast_tensor(mmask.vel_vid,  STDtype.F32, ctx)
        var cond_a  = cast_tensor(cond.vel_aud,   STDtype.F32, ctx)
        var unco_a  = cast_tensor(unco.vel_aud,   STDtype.F32, ctx)
        var mmask_a = cast_tensor(mmask.vel_aud,  STDtype.F32, ctx)

        # eps_v = cond_v + 3*(cond_v - unco_v) + 3*(cond_v - mmask_v)
        var eps_v = add(
            add(cond_v, mul_scalar(sub(cond_v, unco_v,  ctx), VID_G, ctx), ctx),
            mul_scalar(sub(cond_v, mmask_v, ctx), VID_G, ctx),
            ctx,
        )
        # eps_a = cond_a + 2*(cond_a - unco_a) + 2*(cond_a - mmask_a)
        var eps_a = add(
            add(cond_a, mul_scalar(sub(cond_a, unco_a,  ctx), AUD_G, ctx), ctx),
            mul_scalar(sub(cond_a, mmask_a, ctx), AUD_G, ctx),
            ctx,
        )

        # ── Scheduler step ────────────────────────────────────────────────────
        lv = sv.step(eps_v, lv, ctx)
        la = sa.step(eps_a, la, ctx)

        # ── Print per-step std for sanity ─────────────────────────────────────
        var all_dims_v = List[Int]()
        for d in range(len(lv.shape())):
            all_dims_v.append(d)
        var std_v = reduce_std_f32(lv, all_dims_v, False, ctx).to_host(ctx)
        var all_dims_a = List[Int]()
        for d in range(len(la.shape())):
            all_dims_a.append(d)
        var std_a = reduce_std_f32(la, all_dims_a, False, ctx).to_host(ctx)
        print(
            "  step", i,
            "t=", t_val,
            "lv_std=", std_v[0],
            "la_std=", std_a[0],
        )

    print("  Denoise complete. Returning latents (DiT will free).")
    return LatentPair(lv^, la^)


# ── main ──────────────────────────────────────────────────────────────────────
def main() raises:
    var ctx = DeviceContext()
    print("=== NAVA first-run pipeline (", NSTEP, "steps, 3-fwd CFG vid=3 aud=2) ===")

    # Load inputs fixture.
    print("Loading inputs fixture...")
    var fx = ShardedSafeTensors.open(INPUTS_FX)
    var pos_text    = Tensor.from_view_as_f32(fx.tensor_view("pos_text"),     ctx)  # [122,4096]
    var neg_text    = Tensor.from_view_as_f32(fx.tensor_view("neg_text"),     ctx)  # [126,4096]
    var init_lat_vid = Tensor.from_view_as_f32(fx.tensor_view("init_lat_vid"), ctx)  # [1280,48]
    var init_lat_aud = Tensor.from_view_as_f32(fx.tensor_view("init_lat_aud"), ctx)  # [34,128]
    print("  pos_text:", pos_text.shape(),
          "  init_lat_vid:", init_lat_vid.shape(),
          "  init_lat_aud:", init_lat_aud.shape())

    # Phase A: denoise (NavaDiT scoped inside _denoise, freed on return).
    print("Phase A: denoise...")
    var latents = _denoise(pos_text, neg_text, init_lat_vid, init_lat_aud, ctx)
    print("Phase A done. lv:", latents.vid.shape(), " la:", latents.aud.shape())

    # Phase B: video decode + 17 frames.
    print("Phase B: video decode...")
    var vdec = Wan22VaeImageDecoder[16, 16].load(VID_VAE_CKPT, ctx, f32=True)
    var frames = vdec.decode_video_tokens(latents.vid, 5, ctx)  # [1,3,17,256,256]
    var fs = frames.shape()
    print("  frames out: [", fs[0], ",", fs[1], ",", fs[2], ",", fs[3], ",", fs[4], "]")

    for f in range(17):
        # Slice frame f along temporal dim (dim 2) → [1,3,1,256,256].
        var frm_5d = slice(frames, 2, f, 1, ctx)
        # Reshape to [1,3,256,256].
        var frm_shape = List[Int]()
        frm_shape.append(1)
        frm_shape.append(fs[1])
        frm_shape.append(fs[3])
        frm_shape.append(fs[4])
        var frm = reshape(frm_5d, frm_shape^, ctx)
        var out_path = OUT_DIR + "/frame_" + _pad2(f) + ".png"
        save_png(frm, out_path, ctx, ValueRange.SIGNED)
        print("  wrote", out_path)
    print("Phase B done.")

    # Phase C: audio decode + WAV.
    print("Phase C: audio decode + WAV...")

    # Reshape la [34,128] -> [1,34,8,16] -> permute [0,2,1,3] -> [1,8,34,16].
    var la_r1_shape = List[Int]()
    la_r1_shape.append(1)
    la_r1_shape.append(34)
    la_r1_shape.append(8)
    la_r1_shape.append(16)
    var la_r1 = reshape(latents.aud, la_r1_shape^, ctx)  # [1,34,8,16]
    var perm = List[Int]()
    perm.append(0)
    perm.append(2)
    perm.append(1)
    perm.append(3)
    var la_perm = permute(la_r1, perm, ctx)     # [1,8,34,16]
    var la_f32 = cast_tensor(la_perm, STDtype.F32, ctx)

    # Audio VAE decode: [1,8,34,16] -> mel [1,2,133,64].
    var adw = LTX2AudioVaeDecoderWeights.load(AUD_VAE_CKPT, ctx, f32=True)
    var mel = decode(adw, la_f32, ctx)
    var ms = mel.shape()
    print("  mel: [", ms[0], ",", ms[1], ",", ms[2], ",", ms[3], "]")

    # Vocoder: mel [1,2,133,64] -> wav48 [1,2,63840].
    var voc = LTX2VocoderWithBWE.from_file(AUD_VAE_CKPT, ctx)
    var wav48 = voc.forward(mel, ctx)
    var ws = wav48.shape()
    print("  wav48: [", ws[0], ",", ws[1], ",", ws[2], "]")

    # Resample 48kHz -> 16kHz: [1,2,63840] -> [1,2,21280].
    var wav16 = resample_hann(wav48, 48000, 16000, ctx)
    var ws2 = wav16.shape()
    print("  wav16: [", ws2[0], ",", ws2[1], ",", ws2[2], "]")

    # Write WAV.
    var wav_path = OUT_DIR + "/audio.wav"
    save_wav(wav16, wav_path, 16000, ctx)
    print("  wrote", wav_path)
    print("Phase C done.")

    print("WROTE output/nava_first/ : 17 frames + audio.wav")
