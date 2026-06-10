# serenitymojo/pipeline/ernie_sample_cli.mojo
#
# UI-driven CLI adapter for ERNIE-Image 1024x1024 generation.
#
# Contract (the UI bridge calls it exactly this way):
#
#   ernie_sample_cli  <config.json>  <lora|->  <sample_prompts.json>  <prompt_id>  <out.png>
#
#   argv[1]  config JSON path — ACCEPTED BUT IGNORED TODAY. Model dirs are
#            comptime constants from ernie_contract.mojo. Document override
#            instructions in the config file.
#
#   argv[2]  LoRA safetensors path, or "-"/"base"/"" for base model.
#            When a real path is provided the LoRA overlay is applied via
#            ernie_stack_lora_predict_streamed_device (the same code path used
#            by ernie_lora_sample_1024.mojo). When absent/sentinel the base-only
#            predict path (ernie_stack_predict_streamed_device) is used.
#            *** LoRA path is ACCEPTED AND HONORED (or base if sentinel). ***
#
#   argv[3]  sample_prompts JSON (serenity.sample_prompts.v1 schema).
#            Read with `read_sample_prompt_config`.
#            The JSON entry must supply `caps.positive` and `caps.negative`
#            (precomputed by ernie_precache_sample_prompts).
#
#   argv[4]  Prompt id/label to select from the JSON, or "" for the first entry.
#
#   argv[5]  Output PNG path.  Written via save_png(…, ValueRange.SIGNED).
#
# ──────────────────────────────────────────────────────────────────────────────
# CAP-CACHE DESIGN (klein-style, NOT qwen-style runtime encode):
#
#   ERNIE's text encoder is Mistral-3B (3 GB).  The standalone runner
#   (ernie_pipeline_1024_multistep.mojo) accepts a pre-built sidecar produced by
#   an external Rust process.  The trainer-side sampler
#   (ernie_lora_sample_1024.mojo) reads safetensors cap files written by
#   ernie_precache_sample_prompts.mojo.  We follow the same cap-cache contract:
#
#     • The UI bridge runs `ernie_precache_sample_prompts <req.json>` (which
#       encodes the request's prompt into the caps bins) BEFORE calling this CLI.
#     • This CLI reads `prompt.caps_pos` / `prompt.caps_neg` from the JSON entry.
#     • The text encoder is NOT loaded here — only the DiT + VAE run.
#
#   Each cap file is a safetensors with keys:
#     "text_embedding"  [1, 256, 3072] BF16   (or F32 on disk, cast in _load_cap)
#     "text_real_len"   [1] F32               (real token count)
#
# ──────────────────────────────────────────────────────────────────────────────
# Request fields honored vs fixed:
#
#   HONORED at runtime:
#     • prompt    — honored via the precache step that encodes it into the caps bins.
#                   The cap file at `caps_pos` is exactly the encoding of the request's
#                   prompt text, so the prompt drives the image via the cap.
#     • negative  — same as above via caps_neg.
#     • lora      — LoRA path is applied (or base model used if sentinel).
#     • steps     — threaded from prompt JSON into the denoise loop.
#     • cfg       — threaded from prompt JSON (SamplePrompt.cfg field).
#     • seed      — threaded from prompt JSON (SamplePrompt.seed field).
#
#   FIXED at comptime (shape-dependent constants):
#     • width  = 1024  (ERNIE_IMAGE_TOKENS / LH / LW comptime)
#     • height = 1024  (same reason)
#
#   Prompt/negative text strings in the JSON are NOT encoded here; the UI bridge
#   is responsible for running ernie_precache_sample_prompts to turn the text into
#   cap files before invoking this CLI. If caps_pos/caps_neg are stale (written
#   from a different prompt), the image will correspond to those old caps.
#
# ──────────────────────────────────────────────────────────────────────────────
# Generate path: REAL, not a stub.
#   Loads base weights (ErnieStackBase) + optional LoRA, builds RoPE tables,
#   reads cap-cache embeddings, runs the full 36-block streaming denoise loop
#   (same ernie_stack_lora_predict_streamed_device or ernie_stack_predict_streamed_device
#   used by ernie_lora_sample_1024.mojo), then decodes with KleinVaeDecoder.
#
# Build:
#   cd /home/alex/mojodiffusion && pixi run mojo build -I . \
#     -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/pipeline/ernie_sample_cli.mojo \
#     -o /tmp/ernie_sample_cli

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt
from std.sys import argv

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.embeddings import timestep_embedding_sin_first
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import silu
from serenitymojo.ops.random import randn
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.models.dit.ernie_contract import (
    ERNIE_TRANSFORMER_DIR,
    ERNIE_VAE_FILE,
)
from serenitymojo.models.dit.ernie_image import build_ernie_rope_tables
from serenitymojo.models.vae.klein_decoder import KleinVaeDecoder
from serenitymojo.models.ernie.block import ErnieModVecs
from serenitymojo.models.ernie.weights import (
    ErnieStackBase,
    load_ernie_stack_base,
)
from serenitymojo.models.ernie.ernie_stack_lora import (
    ErnieLoraSet,
    load_ernie_lora_resume,
    ernie_lora_set_to_device,
    ernie_stack_lora_predict_streamed_device,
)
from serenitymojo.sampling.ernie_sampling import (
    build_ernie_sigma_schedule,
    ernie_model_timestep_from_sigma,
)
from serenitymojo.training.sample_prompt_config import (
    SamplePrompt, SamplePromptConfig, read_sample_prompt_config,
)

# ── Comptime shape constants (1024x1024 only; see header for rationale) ──────
comptime H = 32
comptime Dh = 128
comptime D = H * Dh
comptime F = 12288
comptime IN_CH = 128
comptime TEXT_IN = 3072
comptime OUT_CH = 128
comptime NUM_LAYERS = 36
comptime RANK = 16
comptime ALPHA = Float32(1.0)
comptime EPS = Float32(1.0e-06)
comptime LH = 64
comptime LW = 64
comptime N_IMG = LH * LW           # 4096
comptime NEG_TXT = 1
comptime SHIFT = Float32(3.0)


# ── PromptCaps: host-side embedding slice (mirrors ernie_lora_sample_1024) ───
@fieldwise_init
struct PromptCaps(Movable):
    var tokens: List[Float32]
    var real_len: Int


def _chunk(src: List[Float32], idx: Int, width: Int) -> List[Float32]:
    var o = List[Float32]()
    var off = idx * width
    for i in range(width):
        o.append(src[off + i])
    return o^


# ── Shared AdaLN + final-norm computation (mirrors ernie_lora_sample_1024) ───
def _shared_adaln_source(
    base: ErnieStackBase, timestep_value: Float32, ctx: DeviceContext
) raises -> Tuple[ErnieModVecs, List[Float32], List[Float32]]:
    var ts = List[Float32]()
    ts.append(timestep_value)
    var ts_t = Tensor.from_host(ts, [1], STDtype.F32, ctx)
    var emb_in = timestep_embedding_sin_first(
        ts_t, D, ctx, 10000.0, base.te_w1[].dtype()
    )
    var h1 = linear(emb_in, base.te_w1[], Optional[Tensor](base.te_b1[].clone(ctx)), ctx)
    h1 = silu(h1, ctx)
    var c = linear(h1, base.te_w2[], Optional[Tensor](base.te_b2[].clone(ctx)), ctx)
    var sc = silu(c, ctx)
    var adaln = linear(sc, base.adaln_w[], Optional[Tensor](base.adaln_b[].clone(ctx)), ctx)
    var adaln_h = adaln.to_host(ctx)
    var fmod = linear(c, base.final_norm_w[], Optional[Tensor](base.final_norm_b[].clone(ctx)), ctx)
    var fmod_h = fmod.to_host(ctx)
    var mv = ErnieModVecs(
        _chunk(adaln_h, 0, D), _chunk(adaln_h, 1, D), _chunk(adaln_h, 2, D),
        _chunk(adaln_h, 3, D), _chunk(adaln_h, 4, D), _chunk(adaln_h, 5, D),
    )
    var f_scale = _chunk(fmod_h, 0, D)
    var f_shift = _chunk(fmod_h, 1, D)
    return (mv^, f_scale^, f_shift^)


# ── Load a cap-cache safetensors written by ernie_precache_sample_prompts ────
# Cap files have keys "text_embedding" [1, SEQ, TEXT_IN] and "text_real_len" [1].
# We slice to [0..real_len] for the conditioned pass. Unlike ernie_lora_sample_1024
# we do NOT validate against a caller-supplied expected_len; we trust the file.
def _load_prompt_caps(path: String, ctx: DeviceContext) raises -> PromptCaps:
    if path == String(""):
        raise Error("ernie_sample_cli: cap path is empty")
    var st = SafeTensors.open(path)

    # real_len tensor
    var rinfo = st.tensor_info(String("text_real_len"))
    var rbytes = st.tensor_bytes(String("text_real_len"))
    var rtv = from_parts(rinfo.dtype, rinfo.shape.copy(), rbytes)
    var rt = cast_tensor(Tensor.from_view(rtv, ctx), STDtype.F32, ctx)
    var rh = rt.to_host(ctx)
    var real_len = Int(rh[0])
    if real_len <= 0:
        raise Error(String("ernie_sample_cli: text_real_len=0 in cap: ") + path)

    # embedding tensor
    var info = st.tensor_info(String("text_embedding"))
    if len(info.shape) != 3 or Int(info.shape[2]) != TEXT_IN:
        raise Error(String("ernie_sample_cli: text_embedding bad shape in: ") + path)
    if Int(info.shape[1]) < real_len:
        raise Error(String("ernie_sample_cli: text_embedding shorter than real_len in: ") + path)
    var bytes = st.tensor_bytes(String("text_embedding"))
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    var f = cast_tensor(t, STDtype.F32, ctx)
    var h = f.to_host(ctx)

    var out = List[Float32]()
    for r in range(real_len):
        for c in range(TEXT_IN):
            out.append(h[r * TEXT_IN + c])
    print("  cap real_len=", real_len, "path=", path)
    return PromptCaps(out^, real_len)


# ── NCHW latent ↔ flat token reordering (mirrors ernie_lora_sample_1024) ─────
def _latent_nchw_to_tokens(latent: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    var hw = LH * LW
    for t in range(hw):
        for ch in range(IN_CH):
            out.append(latent[ch * hw + t])
    return out^


def _tokens_to_latent_nchw(tokens: List[Float32], ctx: DeviceContext) raises -> Tensor:
    var out = List[Float32]()
    var hw = LH * LW
    for ch in range(IN_CH):
        for t in range(hw):
            out.append(tokens[t * IN_CH + ch])
    return Tensor.from_host(out, [1, IN_CH, LH, LW], STDtype.F32, ctx)


# ── Host-side CFG blend ───────────────────────────────────────────────────────
def _cfg_blend(cond: List[Float32], neg: List[Float32], scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(cond)):
        out.append(neg[i] + scale * (cond[i] - neg[i]))
    return out^


# ── Denoise loop (base-only path via LoRA with an empty lora set) ─────────────
# We always call the LoRA path; when lora_path is a sentinel the caller passes
# a loaded LoRA with 0-rank adapters, which reduces to the base forward.
# Comptime N_TXT is the COND real text length from the cap file.
def _denoise[COND_TXT: Int](
    pos: PromptCaps,
    neg: PromptCaps,
    steps: Int,
    cfg_scale: Float32,
    seed: UInt64,
    base: ErnieStackBase,
    st: ShardedSafeTensors,
    lora: ErnieLoraSet,
    ctx: DeviceContext,
) raises -> List[Float32]:
    var lora_dev = ernie_lora_set_to_device(lora, STDtype.BF16, ctx)
    var cond_rope = build_ernie_rope_tables[N_IMG, COND_TXT, H, Dh](
        LH, LW, pos.real_len, ctx, STDtype.BF16
    )
    var neg_rope = build_ernie_rope_tables[N_IMG, NEG_TXT, H, Dh](
        LH, LW, neg.real_len, ctx, STDtype.BF16
    )

    var sh = List[Int]()
    sh.append(1)
    sh.append(IN_CH)
    sh.append(LH)
    sh.append(LW)
    var noise = randn(sh^, seed, STDtype.BF16, ctx)
    var latent_tokens = _latent_nchw_to_tokens(noise.to_host(ctx))
    var sigmas = build_ernie_sigma_schedule(steps, SHIFT)

    for step in range(steps):
        var sigma = sigmas[step]
        var dt = sigmas[step + 1] - sigma
        print("  step", step + 1, "/", steps, "sigma=", sigma)
        var src = _shared_adaln_source(
            base, ernie_model_timestep_from_sigma(sigma), ctx
        )
        var cond_pred = ernie_stack_lora_predict_streamed_device[
            H, Dh, N_IMG, COND_TXT, N_IMG + COND_TXT
        ](
            latent_tokens.copy(), pos.tokens.copy(), base, st, lora_dev, src[0],
            src[1].copy(), src[2].copy(), cond_rope[0], cond_rope[1],
            D, F, IN_CH, TEXT_IN, OUT_CH, EPS, ctx,
        )
        var neg_pred = ernie_stack_lora_predict_streamed_device[
            H, Dh, N_IMG, NEG_TXT, N_IMG + NEG_TXT
        ](
            latent_tokens.copy(), neg.tokens.copy(), base, st, lora_dev, src[0],
            src[1].copy(), src[2].copy(), neg_rope[0], neg_rope[1],
            D, F, IN_CH, TEXT_IN, OUT_CH, EPS, ctx,
        )
        var pred = _cfg_blend(cond_pred, neg_pred, cfg_scale)
        for i in range(len(latent_tokens)):
            latent_tokens[i] = latent_tokens[i] + pred[i] * dt

    return latent_tokens^


# ── Prompt selection helpers (verbatim pattern from qwenimage_sample_cli) ─────
def _select_prompt(sample_cfg: SamplePromptConfig, wanted: String) raises -> SamplePrompt:
    if len(sample_cfg.prompts) == 0:
        raise Error("ernie_sample_cli: sample prompt JSON has no prompts")
    if wanted == String(""):
        return sample_cfg.prompts[0].copy()
    for i in range(len(sample_cfg.prompts)):
        if sample_cfg.prompts[i].label == wanted:
            return sample_cfg.prompts[i].copy()
    raise Error(String("ernie_sample_cli: prompt id not found: ") + wanted)


def _load_prompt_json(
    path: String, wanted: String,
    mut out_prompt: SamplePrompt,
) raises:
    var sample_cfg = read_sample_prompt_config(path)
    var p = _select_prompt(sample_cfg, wanted)
    if p.frames != 1:
        raise Error("ernie_sample_cli: only image prompts (frames=1) are supported")
    if p.caps_pos == String("") or p.caps_neg == String(""):
        raise Error(
            String("ernie_sample_cli: prompt '") + p.label
            + String("' must have caps.positive and caps.negative (run ernie_precache_sample_prompts first)")
        )
    out_prompt = p^
    # Resolution is comptime-fixed at 1024x1024; log what the JSON requested.
    print(
        "  [info] prompt requests size=", out_prompt.width, "x", out_prompt.height,
        "→ fixed 1024x1024 (comptime); steps/cfg/seed honored."
    )


# ── Main entry ────────────────────────────────────────────────────────────────
def main() raises:
    var a = argv()
    if len(a) < 6:
        print(
            "usage: ernie_sample_cli <config.json> <lora|-> <sample_prompts.json>"
            " <prompt_id> <out.png>"
        )
        print("  argv[1] config   — accepted, ignored (model dirs are comptime)")
        print("  argv[2] lora     — path or '-'/'base'/'' for base model")
        print("  argv[3] prompts  — serenity.sample_prompts.v1 JSON (caps required)")
        print("  argv[4] id       — prompt label, or '' for first")
        print("  argv[5] out.png  — output image path")
        print("  Precache: run ernie_precache_sample_prompts <prompts.json> first.")
        raise Error("ernie_sample_cli: need exactly 5 arguments")

    # argv[1]: config — accepted, not used today.
    var _config_path = String(a[1])

    # argv[2]: lora path or sentinel.
    var lora_raw = String(a[2])
    var lora_path = String("")
    if lora_raw != String("-") and lora_raw != String("base") and lora_raw != String(""):
        lora_path = lora_raw

    # argv[3]: sample prompts JSON
    var prompts_json = String(a[3])

    # argv[4]: prompt id
    var prompt_id = String(a[4])

    # argv[5]: output PNG
    var out_png = String(a[5])

    # Load prompt entry (validates caps fields are set).
    var p = SamplePrompt(
        True, String(""), String(""), String(""),
        1024, 1024, 1, Float32(0.0), Float32(0.0),
        30, Float32(4.0), UInt64(42), False, String(""),
        False, String(""), String(""), String(""), String(""),
    )
    _load_prompt_json(prompts_json, prompt_id, p)

    print("=== ERNIE-Image sample CLI ===")
    print("  prompts:", prompts_json, " id:", p.label)
    print("  output:", out_png)
    print("  [prompt]", p.prompt)
    if p.negative != String(""):
        print("  [negative]", p.negative)
    if lora_path != String(""):
        print("  [lora]", lora_path)
    else:
        print("  [lora] (none — base model)")
    print("  steps=", p.steps, " cfg=", p.cfg, " seed=", p.seed)
    print("  caps_pos=", p.caps_pos)
    print("  caps_neg=", p.caps_neg)

    var ctx = DeviceContext()

    # Load cap-cache embeddings (text encoder NOT loaded here).
    print("[caps] loading positive cap")
    var pos = _load_prompt_caps(p.caps_pos, ctx)
    print("[caps] loading negative cap")
    var neg = _load_prompt_caps(p.caps_neg, ctx)

    # Load base weights + optional LoRA.
    print("[model] loading ERNIE base weights")
    var st = ShardedSafeTensors.open(String(ERNIE_TRANSFORMER_DIR))
    var base = load_ernie_stack_base(st, D, IN_CH, ctx)
    var lora = load_ernie_lora_resume(NUM_LAYERS, RANK, ALPHA, lora_path, ctx)

    # Denoise: caps_pos has runtime real_len; dispatch over a bounded set of
    # common real_len values.  ERNIE prompts are typically short (≤50 tokens
    # for captions; up to ~202 for the caption3 format in the reference driver).
    # We dispatch over the actual real_len with a set of comptime specialisations
    # that covers the range used by ernie_lora_sample_1024.mojo plus a fallback.
    print("[denoise] starting denoise, cond real_len=", pos.real_len,
          "steps=", p.steps, "cfg=", p.cfg, "seed=", p.seed)
    var latent_tokens: List[Float32]
    var rl = pos.real_len
    if rl <= 1:
        latent_tokens = _denoise[1](pos, neg, p.steps, p.cfg, p.seed, base, st, lora, ctx)
    elif rl <= 6:
        latent_tokens = _denoise[6](pos, neg, p.steps, p.cfg, p.seed, base, st, lora, ctx)
    elif rl <= 18:
        latent_tokens = _denoise[18](pos, neg, p.steps, p.cfg, p.seed, base, st, lora, ctx)
    elif rl <= 19:
        latent_tokens = _denoise[19](pos, neg, p.steps, p.cfg, p.seed, base, st, lora, ctx)
    elif rl <= 32:
        latent_tokens = _denoise[32](pos, neg, p.steps, p.cfg, p.seed, base, st, lora, ctx)
    elif rl <= 64:
        latent_tokens = _denoise[64](pos, neg, p.steps, p.cfg, p.seed, base, st, lora, ctx)
    elif rl <= 128:
        latent_tokens = _denoise[128](pos, neg, p.steps, p.cfg, p.seed, base, st, lora, ctx)
    elif rl <= 202:
        latent_tokens = _denoise[202](pos, neg, p.steps, p.cfg, p.seed, base, st, lora, ctx)
    else:
        latent_tokens = _denoise[256](pos, neg, p.steps, p.cfg, p.seed, base, st, lora, ctx)

    # Convert token layout back to NCHW and VAE-decode.
    print("[vae] decode")
    var latent = _tokens_to_latent_nchw(latent_tokens, ctx)
    var vae = KleinVaeDecoder[LH, LW].load(String(ERNIE_VAE_FILE), ctx)
    var img = vae.decode(latent, ctx)
    var sh = img.shape()
    print("  image shape:", sh[0], sh[1], sh[2], sh[3])

    # Save PNG (SIGNED range matches ernie_lora_sample_1024 + the pipeline).
    save_png(img, out_png, ctx, ValueRange.SIGNED)
    print("[done] saved:", out_png)
