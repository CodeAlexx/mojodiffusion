# pipeline/boogu_pipeline.mojo — full Boogu-Image T2I generator (Chunk C8).
#
# Inference-only, GPU-only. WIRES the verified C1–C7 components into an
# end-to-end text -> image pipeline. NO new model math: every numeric piece
# (tokenizer, Qwen3-VL encoder, BooguDiT, flow-match scheduler, LDM VAE decoder,
# PNG writer) is built & parity-gated upstream. This file is assembly + the CFG
# denoise loop + the VAE decode + the PNG.
#
# Reference (read-only): /home/alex/Boogu-Image/boogu/pipelines/boogu/pipeline_boogu.py
#   denoise loop / CFG branch and the VAE prep, faithfully mirrored by
#   serenitymojo/models/dit/parity/boogu_c8_reference.py (the torch parity oracle).
#
# T2I, batch=1, demo res 256x256:
#   latent [1,16,32,32]  => h_tok=w_tok=16, img_len=256.
#   cond  = SYSTEM_PROMPT_4_T2I + INSTRUCTION  (45 tokens => CAP_LEN_COND).
#   uncond= SYSTEM_PROMPT_DROP  + ""           (66 tokens => CAP_LEN_UNCOND).
#   num_inference_steps = 20, text_guidance_scale = 4.0.
#
# CFG (pipeline_boogu.py text_guidance_scale>1 branch; c8 ref line 99):
#   pc = DiT(cond); pu = DiT(uncond);  model_pred = pc + (cfg-1)*(pc-pu)
#   latent = scheduler.step(latent, model_pred, i)              # Euler, dt>0
# After loop: VAE.decode(latent) -> [1,3,256,256] -> PNG (clamp[-1,1] => SIGNED).
#
# VRAM staging (24GB GPU; encoder ~16GB, DiT ~20GB can't co-reside): each big
# model is loaded + consumed inside its OWN helper `def`, returning only host
# F32 (the encoded feats / the final latent). The model struct is destroyed at
# the helper's scope exit, freeing its device buffers BEFORE the next loads.
#   _encode_both  -> (feats_cond_host, feats_uncond_host)   [encoder freed]
#   _denoise      -> final_latent_host                       [DiT freed]
#   _decode_and_save                                         [VAE freed]
#
# Initial latent: loaded from boogu_dumps/c8_init_latent.bin (torch-seeded noise,
# for parity vs the torch reference) if present, else ops/random.randn. Kept F32
# across the loop; RNE-cast to BF16 only to feed the DiT (denoise bf16-rounding).
#
# Run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg \
#     && pixi run mojo run -I . serenitymojo/pipeline/boogu_pipeline.mojo
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.memory import alloc
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import (
    sys_open,
    sys_close,
    sys_pread,
    sys_pwrite,
    file_size,
    BytePtr,
    O_RDONLY,
    O_WRONLY,
    O_CREAT,
    O_TRUNC,
)
from serenitymojo.ops.tensor_algebra import sub, add, mul_scalar
from serenitymojo.ops.torch_bf16 import torch_f32_to_bf16_rne
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn

from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.boogu_tokenizer import (
    boogu_tokenize,
    boogu_tokenize_uncond,
)
from serenitymojo.models.text_encoder.boogu_qwen3vl import (
    load_boogu_qwen3vl,
    boogu_encode,
)
from serenitymojo.models.dit.boogu_dit import BooguDiT
from serenitymojo.sampling.flow_match import Scheduler
from serenitymojo.models.vae.zimage_decoder import ZImageDecoder
from serenitymojo.image.png import save_png, ValueRange


# ── paths + config (match boogu_c8_reference.py) ─────────────────────────────
comptime ROOT = "/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base"
comptime MLLM_DIR = ROOT + "/mllm"
comptime TOK_JSON = MLLM_DIR + "/tokenizer.json"
comptime TF_DIR = ROOT + "/transformer"
comptime VAE_DIR = ROOT + "/vae"
comptime DUMP = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/boogu_dumps/"
comptime INIT_LATENT_BIN = DUMP + "c8_init_latent_1024.bin"   # absent -> randn(seed)
comptime FINAL_LATENT_BIN = DUMP + "c8_final_latent_1024_mojo.bin"
comptime OUT_PNG = "/home/alex/mojodiffusion/output/boogu_t2i_1024_mojo.png"

comptime INSTRUCTION: StaticString = (
    "Abstract realism and expressionism style, dark-toned muted palette with subtle tonal variations, pronounced painterly texture with visible brushstrokes and layered paint traces emulating the physical process of multi-layered painting. The figure and the smoky background together fill the entire square frame, extending all the way to every edge. Depth is created through varying densities of brushwork. Diffused lighting casts a cool, mysterious atmosphere. Free, gestural brushstrokes convey movement and dynamism. Full-body close-up: she kneels with one leg forward, torso twisting, hands sliding down the outside of her thighs. Platinum hair fans outward in airy, weightless strands. She wears an open lattice of stiff paper bars, arranged like a ribcage reimagined as art, the bars flexing with her motion, widening, narrowing, bending in smooth arcs without creasing, as if the armor breathes with her. Sections of her waist and hips glow through the openings, surrounded by shifting geometry. The background erupts in smoky layers of ruby, ink-blue, and almond-white that bleed off all four edges, bending as though reacting to her twist. Dynamic, dramatic."
)

# Demo res 256x256 -> latent [1,16,32,32], h_tok=w_tok=16.
comptime LAT_C = 16
comptime LAT_H = 128   # 1024x1024 image -> latent 128x128 (VAE 8x)
comptime LAT_W = 128
comptime H_TOK = 64    # LAT/patch_size(2) -> 64x64 token grid -> 4096 img tokens
comptime W_TOK = 64
comptime NUM_STEPS = 20
comptime CFG_SCALE = Float32(4.0)
comptime SEED: UInt64 = 0

# Caption lengths (comptime: BooguDiT.forward needs CAP_LEN at compile time).
# Verified against the real Qwen3-VL processor:
#   cond  = SYSTEM_PROMPT_4_T2I + INSTRUCTION  => 267 tokens (first prompt, full-bleed variant).
#   uncond= SYSTEM_PROMPT_DROP  + ""           => 66 tokens.
comptime CAP_LEN_COND = 267
comptime CAP_LEN_UNCOND = 66


# ── tiny F32 .bin IO (reused idiom from boogu_c6_parity / zimage_generate) ────
def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("boogu_pipeline: cannot open ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("boogu_pipeline: empty/missing ") + path)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var nf = n // 4
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(nf):
        out.append(fp[i])
    buf.free()
    return out^


def _write_bin_f32(path: String, values: List[Float32]) raises:
    var nbytes = len(values) * 4
    var buf = alloc[UInt8](nbytes if nbytes > 0 else 1)
    var fp = buf.bitcast[Float32]()
    for i in range(len(values)):
        fp[i] = values[i]
    var fd = sys_open(path, O_CREAT | O_WRONLY | O_TRUNC, Int32(0o644))
    if fd < 0:
        buf.free()
        raise Error(String("boogu_pipeline: cannot create ") + path)
    var wrote = sys_pwrite(fd, BytePtr(unsafe_from_address=Int(buf)), nbytes, 0)
    buf.free()
    _ = sys_close(fd)
    if wrote != nbytes:
        raise Error(String("boogu_pipeline: short write to ") + path)


def _std_f32(values: List[Float32]) -> Float32:
    var n = len(values)
    if n == 0:
        return Float32(0.0)
    var mean = Float32(0.0)
    for i in range(n):
        mean += values[i]
    mean /= Float32(n)
    var var_sum = Float32(0.0)
    for i in range(n):
        var d = values[i] - mean
        var_sum += d * d
    return sqrt(var_sum / Float32(n))


# ── stage 1: encoder (loaded + freed inside this helper) ─────────────────────
# Returns the cond + uncond RAW last-hidden feats as host F32 (cond [1,45,4096],
# uncond [1,66,4096]). The Qwen3Encoder struct is destroyed at scope exit, so its
# ~16GB of device weights free BEFORE the DiT loads.
def _encode_both(ctx: DeviceContext) raises -> Tuple[List[Float32], List[Float32]]:
    print("[c8] stage1: tokenize + encode (Qwen3-VL text path)…")
    var tok = Qwen3Tokenizer(String(TOK_JSON))
    var cond_ids = boogu_tokenize(tok, String(INSTRUCTION))
    var uncond_ids = boogu_tokenize_uncond(tok)
    print("  cond L=", len(cond_ids), " uncond L=", len(uncond_ids))
    if len(cond_ids) != CAP_LEN_COND:
        raise Error(
            String("boogu_pipeline: cond L=") + String(len(cond_ids))
            + " != CAP_LEN_COND=" + String(CAP_LEN_COND)
        )
    if len(uncond_ids) != CAP_LEN_UNCOND:
        raise Error(
            String("boogu_pipeline: uncond L=") + String(len(uncond_ids))
            + " != CAP_LEN_UNCOND=" + String(CAP_LEN_UNCOND)
        )

    var enc = load_boogu_qwen3vl(String(MLLM_DIR), ctx)
    var feats_cond = boogu_encode(enc, cond_ids, ctx)      # [1,45,4096] BF16
    var feats_uncond = boogu_encode(enc, uncond_ids, ctx)  # [1,66,4096] BF16
    var cond_h = feats_cond.to_host(ctx)                   # F32 host
    var uncond_h = feats_uncond.to_host(ctx)
    print(
        "  feats_cond std=", _std_f32(cond_h),
        " feats_uncond std=", _std_f32(uncond_h),
    )
    # enc (and feats tensors) drop here -> encoder VRAM freed before return.
    return (cond_h^, uncond_h^)


# ── stage 2: DiT denoise loop (loaded + freed inside this helper) ────────────
# Returns the final latent as host F32 [1,16,32,32]. The BooguDiT struct (~20GB)
# is destroyed at scope exit, freeing its device weights BEFORE the VAE loads.
def _denoise(
    cond_h: List[Float32],
    uncond_h: List[Float32],
    init_latent_h: List[Float32],
    ctx: DeviceContext,
) raises -> List[Float32]:
    print("[c8] stage2: DiT denoise (", NUM_STEPS, " steps, cfg=", CFG_SCALE, ")…")
    # Conditioning tensors (BF16 — the DiT's caption path / forward consume BF16,
    # matching the c6 parity gate which fed instruction as BF16).
    var feats_cond = Tensor.from_host(
        cond_h, [1, CAP_LEN_COND, 4096], STDtype.BF16, ctx
    )
    var feats_uncond = Tensor.from_host(
        uncond_h, [1, CAP_LEN_UNCOND, 4096], STDtype.BF16, ctx
    )

    # Accumulating latent stays F32 across the loop.
    var latent = Tensor.from_host(
        init_latent_h, [1, LAT_C, LAT_H, LAT_W], STDtype.F32, ctx
    )

    var sched = Scheduler.boogu(NUM_STEPS)
    var ts = sched.timesteps()  # NUM_STEPS model timesteps (ascending 0->1)

    var dit = BooguDiT.load(String(TF_DIR), ctx)

    for i in range(NUM_STEPS):
        # Per-step model timestep as a [1] F32 tensor (the DiT scales by
        # timestep_scale=1000 internally).
        var t_val = List[Float32]()
        t_val.append(ts[i])
        var t = Tensor.from_host(t_val^, [1], STDtype.F32, ctx)

        # RNE-cast the F32 latent to BF16 to feed the model (denoise bf16-round).
        var latent_bf16 = torch_f32_to_bf16_rne(latent, ctx)

        # pc = DiT(cond), pu = DiT(uncond). Different CAP_LEN -> two
        # `forward[...]` comptime instantiations.
        var pred_cond = dit.forward[CAP_LEN_COND, H_TOK, W_TOK](
            latent_bf16, t, feats_cond, ctx
        )                                                # [1,16,32,32]
        var pred_uncond = dit.forward[CAP_LEN_UNCOND, H_TOK, W_TOK](
            latent_bf16, t, feats_uncond, ctx
        )

        # CFG: model_pred = pc + (cfg-1)*(pc-pu)  (pipeline_boogu.py / c8 ref:99).
        # pred_cond/pred_uncond are BF16 (DiT out); combine in BF16, exactly as the
        # torch reference (which combines the bf16-cast predictions), then upcast
        # to F32 for the Euler step (c8 ref: `model_pred.float()`, `latent.float()`).
        var delta = sub(pred_cond, pred_uncond, ctx)
        var scaled = mul_scalar(delta, CFG_SCALE - Float32(1.0), ctx)
        var model_pred = add(pred_cond, scaled, ctx)     # [1,16,32,32] BF16

        # Euler step on the F32 latent: x_next = x + v*(t[i+1]-t[i]). add() requires
        # matching dtypes, so GPU-upcast the velocity to F32 first.
        var mp_f32 = cast_tensor(model_pred, STDtype.F32, ctx)
        latent = sched.step(latent, mp_f32, i, ctx)      # F32

        if i % 5 == 0 or i == NUM_STEPS - 1:
            var lh = latent.to_host(ctx)
            print("  step ", i, " t=", ts[i], " latent.std=", _std_f32(lh))

    var final_h = latent.to_host(ctx)
    # dit drops here -> DiT VRAM freed before return.
    return final_h^


# ── stage 3: VAE decode + PNG (loaded + freed inside this helper) ────────────
# ZImageDecoder is the diffusers-format 16-channel AutoencoderKL decoder with the
# FLUX VAE rescale (scale 0.3611, shift 0.1159) folded INSIDE decode — exactly the
# Boogu VAE config (use_post_quant_conv=false, latent_channels=16). Feed the RAW
# final latent. (NOTE: load_flux1_ldm_decoder uses LDM-format keys which are NOT in
# this diffusers checkpoint — see report.)
def _decode_and_save(final_h: List[Float32], ctx: DeviceContext) raises:
    print("[c8] stage3: VAE decode + PNG…")
    var latent = Tensor.from_host(
        final_h, [1, LAT_C, LAT_H, LAT_W], STDtype.F32, ctx
    )
    var vae = ZImageDecoder[LAT_H, LAT_W].load(String(VAE_DIR), ctx)
    var img = vae.decode(latent, ctx)  # [1,3,256,256] F32 (NCHW)
    var ish = img.shape()
    print("  decoded image shape [", ish[0], ",", ish[1], ",", ish[2], ",", ish[3], "]")
    var img_h = img.to_host(ctx)
    # image stats (pre-quantization, in [-1,1]).
    var mn = img_h[0]
    var mx = img_h[0]
    for i in range(len(img_h)):
        if img_h[i] < mn:
            mn = img_h[i]
        if img_h[i] > mx:
            mx = img_h[i]
    print("  image min=", mn, " max=", mx, " std=", _std_f32(img_h))
    save_png(img, String(OUT_PNG), ctx, ValueRange.SIGNED)
    print("  wrote ", OUT_PNG)
    # vae drops here.


def main() raises:
    var ctx = DeviceContext()
    print("=== Boogu-Image T2I (C8 full pipeline) ===")
    print("  instruction:", String(INSTRUCTION))
    print("  res", LAT_H * 8, "x", LAT_W * 8, " latent [1,16,", LAT_H, ",", LAT_W, "] steps", NUM_STEPS, "cfg", CFG_SCALE)

    # ── initial latent: prefer the torch-seeded dump for parity, else randn. ──
    var init_h: List[Float32]
    var init_fd = sys_open(String(INIT_LATENT_BIN), O_RDONLY)
    if init_fd >= 0:
        _ = sys_close(init_fd)
        init_h = _read_bin_f32(String(INIT_LATENT_BIN))
        if len(init_h) != LAT_C * LAT_H * LAT_W:
            raise Error(
                String("boogu_pipeline: c8_init_latent.bin numel=")
                + String(len(init_h)) + " != " + String(LAT_C * LAT_H * LAT_W)
            )
        print("  initial latent: LOADED from c8_init_latent.bin (torch seed)")
    else:
        var noise = randn(
            [1, LAT_C, LAT_H, LAT_W], SEED, STDtype.F32, ctx
        )
        init_h = noise.to_host(ctx)
        print("  initial latent: GENERATED via ops/random.randn (seed", SEED, ")")
    print("  init latent std=", _std_f32(init_h))

    # ── stage 1 (encoder freed on return). ──
    var both = _encode_both(ctx)
    var cond_h = both[0].copy()
    var uncond_h = both[1].copy()

    # ── stage 2 (DiT freed on return). ──
    var final_h = _denoise(cond_h, uncond_h, init_h, ctx)
    print("  final latent std=", _std_f32(final_h))
    _write_bin_f32(String(FINAL_LATENT_BIN), final_h)
    print("  dumped final latent ->", FINAL_LATENT_BIN)

    # ── stage 3 (VAE freed on return). ──
    _decode_and_save(final_h, ctx)

    print("=== C8 DONE ===")
