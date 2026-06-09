# serenitymojo/pipeline/flux_sample_cli.mojo
#
# UI-driven CLI adapter for FLUX Dev (FLUX.1-dev) text→image generation.
# Mirrors the qwenimage_sample_cli.mojo pattern verbatim; only the model
# import block, encode, denoise, and VAE-decode sections differ.
#
# Contract (the UI bridge calls it exactly this way):
#
#   flux_sample_cli  <config.json>  <lora|->  <sample_prompts.json>  <prompt_id>  <out.png>
#
#   argv[1]  config JSON path — ACCEPTED BUT IGNORED TODAY (model dirs are
#            comptime constants below).  To override model paths, edit the
#            comptime block in this file or add a config-reader call.
#
#   argv[2]  LoRA safetensors path, or "-"/"base"/"" for base model.
#            FLUX Dev LoRA support is not yet wired in this adapter; the value
#            is ACCEPTED AND IGNORED.  When FLUX LoRA lands, thread it into the
#            denoise loop via a LoRA overlay mechanism analogous to zimage_generate.
#
#   argv[3]  sample_prompts JSON (serenity.sample_prompts.v1 schema).
#            Read with `read_sample_prompt_config`.
#
#   argv[4]  Prompt id/label to select from the JSON, or "" for the first entry.
#
#   argv[5]  Output PNG path.  Written via save_png(…, ValueRange.SIGNED).
#
# ──────────────────────────────────────────────────────────────────────────────
# Request fields honored vs fixed:
#
#   HONORED at runtime:
#     • prompt    — runtime String threaded through T5 + CLIP encode path.
#
#   FIXED at comptime (from flux1_pipeline_smoke.mojo conventions):
#     • steps     = STEPS   (20)
#     • guidance  = GUIDANCE (3.5, guidance-distilled scalar fed to DiT)
#     • seed      = SEED    (UInt64(42))
#     • width     = WIDTH   (1024)
#     • height    = HEIGHT  (1024)
#
#   UNUSED by the model:
#     • negative  — FLUX Dev is guidance-distilled (single forward per step;
#                   guidance_vec is a MODEL INPUT, not a CFG multiplier).
#                   There is NO negative prompt path in the DiT.  The negative
#                   string from the JSON is read and acknowledged but discarded.
#
# ──────────────────────────────────────────────────────────────────────────────
# Generate path: FULL pure-Mojo, prompt-driven (2026-06-09).
#
#   The CLIP-L and T5-XXL weights are loaded from disk and the DiT / VAE decode
#   path is the REAL Mojo implementation (same as flux1_pipeline_smoke.mojo).
#
#   Tokenization is now REAL and pure-Mojo (no Rust/sidecar/placeholder):
#     • CLIP-L BPE     — `ClipTokenizer` (tokenizer/clip_tokenizer.mojo), bit-exact vs HF.
#     • T5-XXL Unigram — `T5Tokenizer`   (tokenizer/t5_tokenizer.mojo),   bit-exact vs HF.
#   encode_text() tokenizes the prompt at runtime, fits CLIP→77 / T5→512, and
#   feeds the real ids to ClipEncoder / T5Encoder. The prompt now drives the image.
#
# Build:
#   cd /home/alex/mojodiffusion && pixi run mojo build -I . \
#     -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/pipeline/flux_sample_cli.mojo \
#     -o /tmp/flux_sample_cli

from std.sys import argv
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.text_encoder.clip_encoder import ClipEncoder, ClipConfig
from serenitymojo.models.text_encoder.t5_encoder import T5Encoder, T5Config
from serenitymojo.tokenizer.clip_tokenizer import ClipTokenizer
from serenitymojo.tokenizer.t5_tokenizer import T5Tokenizer
from serenitymojo.models.dit.flux1_dit import (
    Flux1Config,
    Flux1Offloaded,
    build_flux1_rope_tables,
)
from serenitymojo.models.vae.ldm_decoder import load_flux1_ldm_decoder
from serenitymojo.registry.checkpoints import default_manifest_by_id
from serenitymojo.sampling.flux1_dev import (
    build_flux1_packed_latent_plan,
    build_flux1_sigma_schedule,
)
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import add, mul, mul_scalar, reshape, permute, slice, concat
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.training.sample_prompt_config import (
    SamplePrompt, SamplePromptConfig, read_sample_prompt_config,
)

# ── Model paths (from manifest; comptime for zero runtime overhead) ───────────
comptime FLUX1_ROOT    = "/home/alex/.serenity/models"
comptime TEXT_ENC_ROOT = FLUX1_ROOT + "/text_encoders"
comptime CLIP_PATH     = TEXT_ENC_ROOT + "/clip_l.safetensors"
comptime T5_PATH       = TEXT_ENC_ROOT + "/t5xxl_fp16.safetensors"
comptime CLIP_TOK_JSON = TEXT_ENC_ROOT + "/clip_l.tokenizer.json"
comptime T5_TOK_JSON   = TEXT_ENC_ROOT + "/t5xxl_fp16.tokenizer.json"
comptime DIT_PATH      = FLUX1_ROOT + "/checkpoints/flux1-dev.safetensors"
comptime VAE_PATH      = FLUX1_ROOT + "/vaes/ae.safetensors"

# ── Shape constants (verbatim from flux1_pipeline_smoke.mojo) ─────────────────
comptime HEIGHT        = 1024
comptime WIDTH         = 1024
comptime AE_IN_CHANNELS = 16
comptime LATENT_H      = 2 * ((HEIGHT + 15) // 16)   # 128
comptime LATENT_W      = 2 * ((WIDTH  + 15) // 16)   # 128
comptime IMG_H2        = (HEIGHT + 15) // 16          # 64
comptime IMG_W2        = (WIDTH  + 15) // 16          # 64
comptime N_IMG         = IMG_H2 * IMG_W2              # 4096
comptime N_TXT         = 512
comptime S             = N_IMG + N_TXT
# VAE tiled-decode: 2x2 latent quadrants. After the offloaded DiT, the caching
# allocator pool is at a high water mark and a single 1024² decode buffer OOMs
# (measured: VAE-alone fits, VAE-after-DiT does not). Each tile decodes a
# LATENT/2 quadrant (small alloc that fits the pool) → assembled output.
comptime TILE_H        = LATENT_H // 2               # 64 @1024²
comptime TILE_W        = LATENT_W // 2

# ── Sampler constants (comptime-fixed today; see header for rationale) ────────
comptime STEPS    = 20
comptime GUIDANCE = Float32(3.5)
comptime SEED     = UInt64(42)
comptime CLIP_LEN = 77
comptime T5_LEN   = 512


# ── pack [1,16,H,W] -> [1, h2*w2, 64] (mirrors flux1_pipeline_smoke.mojo) ────
def _pack_latent(z_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
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


# ── unpack [1, h2*w2, 64] -> [1,16,2*h2,2*w2] ────────────────────────────────
def _unpack_latent(packed: Tensor, ctx: DeviceContext) raises -> Tensor:
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


# ── feathered cross-fade weight (ramps 1→0 or 0→1 along `dim`), F32 ──────────
# Shape [1,1,n,1] for a height(dim 2) blend, [1,1,1,n] for a width(dim 3) blend,
# so it broadcasts over channels and the non-blend spatial axis.
def _weight_tensor(n: Int, dim: Int, ascending: Bool, ctx: DeviceContext) raises -> Tensor:
    var h = List[Float32]()
    for i in range(n):
        var t = (Float32(i) + 0.5) / Float32(n)
        h.append(t if ascending else (1.0 - t))
    var sh = List[Int]()
    sh.append(1)
    sh.append(1)
    if dim == 2:
        sh.append(n)
        sh.append(1)
    else:
        sh.append(1)
        sh.append(n)
    return Tensor.from_host(h^, sh^, STDtype.F32, ctx)


# ── cross-fade two equal-shaped overlap slabs along `dim` (left fades out) ────
def _xfade(left: Tensor, right: Tensor, dim: Int, ctx: DeviceContext) raises -> Tensor:
    var n = left.shape()[dim]
    var wl = _weight_tensor(n, dim, False, ctx)
    var wr = _weight_tensor(n, dim, True, ctx)
    return add(mul(left, wl, ctx), mul(right, wr, ctx), ctx)


# ── blend 3 equal tiles (size T along `dim`) placed at offsets 0, T/2, T ──────
# Output size 2T: [pure t0 | xfade(t0,t1) | xfade(t1,t2) | pure t2].
def _blend3(t0: Tensor, t1: Tensor, t2: Tensor, dim: Int, ctx: DeviceContext) raises -> Tensor:
    var t = t0.shape()[dim]
    var s = t // 2
    var ov = t - s
    var a = slice(t0, dim, 0, s, ctx)
    var b = _xfade(slice(t0, dim, s, ov, ctx), slice(t1, dim, 0, ov, ctx), dim, ctx)
    var c = _xfade(slice(t1, dim, ov, ov, ctx), slice(t2, dim, 0, ov, ctx), dim, ctx)
    var d = slice(t2, dim, ov, s, ctx)
    return concat(dim, ctx, a, b, c, d)


# ── tiled VAE decode — 3x3 OVERLAPPING latent crops + feathered blend ────────
# Decoder is instantiated once at the proven TILE shape (64² latent → 512² image,
# the size that fits the post-DiT allocator pool) and reused for all 9 crops.
# Crops sit at latent stride TILE/2 (positions 0/32/64) so adjacent image tiles
# overlap by 256px; a separable feathered cross-fade (horizontal per row, then
# vertical) erases the seams the non-overlapping 2x2 version could leave. Tile
# vars are reassigned per row so prior tiles free → retained-memory peak stays
# near the 2x2 working version (measured: 72²/2x2 OOM'd, 64² fits).
def _tiled_decode(latent: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dec = load_flux1_ldm_decoder[TILE_H, TILE_W](VAE_PATH, ctx)
    var half = TILE_H // 2                          # 32 latent stride
    # row 0 crop [0:64], blend its 3 columns
    var r = slice(latent, 2, 0, TILE_H, ctx)
    var a = cast_tensor(dec.decode(slice(r, 3, 0, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var b = cast_tensor(dec.decode(slice(r, 3, half, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var c = cast_tensor(dec.decode(slice(r, 3, TILE_W, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var row0 = _blend3(a, b, c, 3, ctx)
    # row 1 crop [32:96] (reassign a/b/c → prior tiles freed)
    r = slice(latent, 2, half, TILE_H, ctx)
    a = cast_tensor(dec.decode(slice(r, 3, 0, TILE_W, ctx), ctx), STDtype.F32, ctx)
    b = cast_tensor(dec.decode(slice(r, 3, half, TILE_W, ctx), ctx), STDtype.F32, ctx)
    c = cast_tensor(dec.decode(slice(r, 3, TILE_W, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var row1 = _blend3(a, b, c, 3, ctx)
    # row 2 crop [64:128]
    r = slice(latent, 2, TILE_H, TILE_H, ctx)
    a = cast_tensor(dec.decode(slice(r, 3, 0, TILE_W, ctx), ctx), STDtype.F32, ctx)
    b = cast_tensor(dec.decode(slice(r, 3, half, TILE_W, ctx), ctx), STDtype.F32, ctx)
    c = cast_tensor(dec.decode(slice(r, 3, TILE_W, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var row2 = _blend3(a, b, c, 3, ctx)
    # vertical feathered blend of the 3 full-width rows
    return _blend3(row0, row1, row2, 2, ctx)


def _to_bf16(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    if x.dtype() == STDtype.BF16:
        return cast_tensor(x, STDtype.BF16, ctx)
    if x.dtype() == STDtype.F16:
        var x_f32 = cast_tensor(x, STDtype.F32, ctx)
        return cast_tensor(x_f32, STDtype.BF16, ctx)
    return cast_tensor(x, STDtype.BF16, ctx)


# ── pad/truncate token ids to a fixed length, keeping EOS at the tail ─────────
# (HF CLIP: max 77, pad==eos; HF T5: max 512, pad 0, eos 1. Both encode() calls
#  already include the special tokens — see clip/t5 tokenizer encode().)
def _fit(var ids: List[Int], target: Int, pad_id: Int, eos_id: Int) -> List[Int]:
    var out = List[Int]()
    if len(ids) >= target:
        for i in range(target - 1):
            out.append(ids[i])
        out.append(eos_id)
    else:
        for i in range(len(ids)):
            out.append(ids[i])
        while len(out) < target:
            out.append(pad_id)
    return out^


# ── Encode text → (clip_pooled [1,768], t5_hidden [1,512,4096]) ───────────────
# REAL pure-Mojo tokenization (CLIP BPE + T5 Unigram, both bit-exact vs HF).
@fieldwise_init
struct FluxCaps(Movable):
    var vector: Tensor   # CLIP-L pooled [1,768] cast to BF16
    var txt: Tensor      # T5-XXL hidden  [1,512,4096] cast to BF16


def encode_text(prompt: String, ctx: DeviceContext) raises -> FluxCaps:
    # CLIP-L: encode() adds BOS(49406)+EOS(49407); fit to 77 (pad==eos).
    var clip_tok = ClipTokenizer(CLIP_TOK_JSON)
    var clip_ids = _fit(clip_tok.encode(prompt), CLIP_LEN, 49407, 49407)
    print("[text] CLIP-L encode (", len(clip_ids), "ids )")
    var clip = ClipEncoder.load(CLIP_PATH, ClipConfig.clip_l(), ctx)
    var clip_out = clip.encode_sdxl[CLIP_LEN](clip_ids^, ctx)
    # encode_sdxl returns (last_hidden, pooled); FLUX.1 uses the pooled [1,768].
    var vector = _to_bf16(clip_out[1], ctx)

    # T5-XXL: encode() adds EOS(1); fit to 512 (pad 0).
    var t5_tok = T5Tokenizer(T5_TOK_JSON)
    var t5_ids = _fit(t5_tok.encode(prompt), T5_LEN, 0, 1)
    print("[text] T5-XXL encode (", len(t5_ids), "ids )")
    var t5 = T5Encoder[T5_LEN].load(T5_PATH, T5Config.t5_xxl(), ctx)
    var t5_hidden = t5.encode(t5_ids^, ctx)
    var txt = _to_bf16(t5_hidden, ctx)

    return FluxCaps(vector^, txt^)


# ── FLUX Dev denoise loop (STEPS / GUIDANCE / SEED are comptime-fixed) ────────
# Guidance-distilled: single DiT forward per step; guidance_vec is a MODEL
# INPUT scalar, not a CFG multiplier.  No negative prompt is used.
#
# STAGED LOADING: returns the packed latent on the HOST so that on return the DiT
# (offloaded shared weights + loader + the GPU latent + rope) all drop, leaving a
# clean GPU for the VAE decode stage (the 1024² decode needs a large contiguous
# allocation; cross-stage residency + offload-churn fragmentation caused OOM).
def denoise(caps: FluxCaps, ctx: DeviceContext) raises -> List[Float32]:
    print("[dit] loading FLUX.1-dev DiT (offloaded)")
    var model = Flux1Offloaded.load(DIT_PATH, Flux1Config.dev(), ctx)
    var rope = build_flux1_rope_tables[N_IMG, N_TXT, 24, 128](
        IMG_H2, IMG_W2, ctx, STDtype.BF16
    )

    # initial noise NCHW [1,16,LATENT_H,LATENT_W] -> pack -> [1,N_IMG,64]
    var noise_shape = List[Int]()
    noise_shape.append(1)
    noise_shape.append(AE_IN_CHANNELS)
    noise_shape.append(LATENT_H)
    noise_shape.append(LATENT_W)
    var noise_nchw = randn(noise_shape^, SEED, STDtype.F32, ctx)
    var img = _pack_latent(noise_nchw, ctx)

    var sched = build_flux1_sigma_schedule(STEPS, N_IMG)
    print("[denoise]", STEPS, "steps, guidance", GUIDANCE, "seed", SEED)
    for i in range(STEPS):
        var t_curr = sched[i]
        var t_prev = sched[i + 1]
        print("  step", i + 1, "/", STEPS, "t_curr", t_curr, "->", t_prev)
        # t_vec / guidance_vec pre-scaled by 1000 (BFL time_factor convention;
        # the foundation t_embedder does NOT apply the 1000x internally).
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
                img_bf, caps.txt, t_vec, Optional[Tensor](g_vec^),
                caps.vector, rope[0], rope[1], ctx,
            ),
            STDtype.F32,
            ctx,
        )
        # Euler step: img = img + (t_prev - t_curr) * pred
        var dt = t_prev - t_curr
        img = add(img, mul_scalar(pred, dt, ctx), ctx)

    # to HOST: drops the DiT-stage GPU state on return (staged loading).
    return cast_tensor(img, STDtype.F32, ctx).to_host(ctx)


# ── Prompt selection helpers (verbatim from qwenimage_sample_cli.mojo) ─────────

def _select_prompt(sample_cfg: SamplePromptConfig, wanted: String) raises -> SamplePrompt:
    if len(sample_cfg.prompts) == 0:
        raise Error("flux_sample_cli: sample prompt JSON has no prompts")
    if wanted == String(""):
        return sample_cfg.prompts[0].copy()
    for i in range(len(sample_cfg.prompts)):
        if sample_cfg.prompts[i].label == wanted:
            return sample_cfg.prompts[i].copy()
    raise Error(String("flux_sample_cli: prompt id not found: ") + wanted)


def _load_prompt_json(
    path: String, wanted: String,
    mut prompt: String, mut negative: String,
) raises:
    var sample_cfg = read_sample_prompt_config(path)
    var p = _select_prompt(sample_cfg, wanted)
    if p.frames != 1:
        raise Error("flux_sample_cli: only image prompts (frames=1) are supported")
    prompt = p.prompt.copy()
    negative = p.negative.copy()
    # steps/cfg/seed/width/height are comptime-fixed today; log what the JSON
    # requested so the caller knows what was ignored.
    print(
        "  [info] sample prompt requests steps=", p.steps, "guidance=", p.cfg,
        "seed=", p.seed, "size=", p.width, "x", p.height,
        "→ all ignored (comptime fixed); prompt honored (real CLIP+T5 tokenization),",
        "negative discarded (FLUX Dev is guidance-distilled, no CFG).",
    )


# ── Main entry ──────────────────────────────────────────────────────────────

def main() raises:
    var a = argv()
    if len(a) < 6:
        print(
            "usage: flux_sample_cli <config.json> <lora|-> <sample_prompts.json>"
            " <prompt_id> <out.png>"
        )
        print("  argv[1] config   — accepted, ignored (model dirs are comptime)")
        print("  argv[2] lora     — accepted, ignored (FLUX LoRA not yet wired)")
        print("  argv[3] prompts  — serenity.sample_prompts.v1 JSON")
        print("  argv[4] id       — prompt label, or '' for first")
        print("  argv[5] out.png  — output image path")
        raise Error("flux_sample_cli: need exactly 5 arguments")

    # argv[1]: config — accepted, not used today.
    var _config_path = String(a[1])

    # argv[2]: lora path or sentinel; accepted, not used today.
    var lora_raw = String(a[2])
    var _lora_path = String("")
    if lora_raw != String("-") and lora_raw != String("base") and lora_raw != String(""):
        _lora_path = lora_raw
        print("[lora] path provided but ignored (FLUX Dev LoRA not yet wired):", _lora_path)

    # argv[3]: sample prompts JSON
    var prompts_json = String(a[3])

    # argv[4]: prompt id
    var prompt_id = String(a[4])

    # argv[5]: output PNG
    var out_png = String(a[5])

    # Load prompt + negative from the JSON.
    var prompt = String("")
    var negative = String("")
    _load_prompt_json(prompts_json, prompt_id, prompt, negative)

    print("=== FLUX Dev sample CLI ===")
    print("  prompts:", prompts_json, " id:", prompt_id)
    print("  output:", out_png)
    print("  [prompt]", prompt)
    if negative != String(""):
        print("  [negative] (discarded — FLUX Dev is guidance-distilled, no CFG):", negative)

    var ctx = DeviceContext()

    # Encode text (prompt logged; placeholder token ids used — see header).
    var caps = encode_text(prompt, ctx)

    # Denoise (STEPS/GUIDANCE/SEED are comptime-fixed; see file header).
    # Returns the packed latent on HOST → the DiT stage's GPU is freed here,
    # giving the VAE decode a clean GPU (staged loading).
    var packed_h = denoise(caps, ctx)

    # VAE decode (fresh GPU stage, TILED). Re-upload the host latent [1, N_IMG, 64].
    print("[vae] unpack + tiled decode (3x3 overlap+blend)")
    var psh = List[Int]()
    psh.append(1)
    psh.append(N_IMG)
    psh.append(AE_IN_CHANNELS * 4)
    var packed = Tensor.from_host(packed_h, psh^, STDtype.F32, ctx)
    var latent = _unpack_latent(packed, ctx)
    var img = _tiled_decode(latent, ctx)
    var sh = img.shape()
    print("  image shape:", sh[0], sh[1], sh[2], sh[3])

    # Save.
    save_png(img, out_png, ctx, ValueRange.SIGNED)
    print("[done] saved:", out_png)
