# serenitymojo/pipeline/qwenimage_sample_cli.mojo
#
# UI-driven CLI adapter for Qwen-Image text→image generation.
# This is the proof-of-pattern adapter; other model adapters (chroma, sd3,
# sdxl, ernie, flux) should copy this file and replace the encode/denoise/vae
# calls with their own model equivalents.
#
# Contract (the UI bridge calls it exactly this way):
#
#   qwenimage_sample_cli  <config.json>  <lora|->  <sample_prompts.json>  <prompt_id>  <out.png>
#
#   argv[1]  config JSON path (model dirs are comptime constants in the runner;
#            this argument is ACCEPTED BUT IGNORED TODAY — document override
#            instructions in the config file).
#
#   argv[2]  LoRA safetensors path, or "-"/"base"/"" for base model.
#            Qwen-Image has no LoRA path today; the value is ACCEPTED AND
#            IGNORED.  When Qwen-Image LoRA lands, wire it into encode_captions.
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
#     • prompt    — threaded through _encode_trimmed (runtime String, not comptime)
#     • negative  — threaded through _encode_trimmed (runtime String, not comptime)
#
#   FIXED at comptime (from qwenimage_pipeline_1024_multistep.mojo):
#     • steps  = STEPS   (30)
#     • cfg    = CFG     (4.0)
#     • seed   = SEED    (UInt64(42))
#     • width  = LW * 8  (1024, latent LW=128)
#     • height = LH * 8  (1024, latent LH=128)
#
#   The Qwen-Image DiT attention shape is a comptime constant (N_IMG, N_TXT_KEPT,
#   S_POS, S_NEG), so resolution changes require a recompile.  Steps/cfg/seed are
#   also fixed in the denoise loop for the same reason (scheduler sigmas are
#   comptime-derived).  When the runner gains runtime dispatch for these fields,
#   remove them from the list above and thread from `req_prompt` below.
#
# ──────────────────────────────────────────────────────────────────────────────
# Generate path: REAL, not a stub.
#   Calls encode_captions_from_strings → denoise → unpatchify → tiled Qwen VAE decode.
#   The only change vs the standalone runner is that the prompt/negative pair
#   comes from the sample_prompts JSON rather than the PROMPT/NEGATIVE comptime.
#
# Build:
#   cd /home/alex/mojodiffusion && pixi run mojo build -I . \
#     -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/pipeline/qwenimage_sample_cli.mojo \
#     -o /tmp/qwenimage_sample_cli
#
# Note for porter agents (chroma/sd3/sdxl/ernie/flux):
#   1. Replace the import block (text encoder, DiT, VAE, ops) with your model's.
#   2. Replace encode_captions_from_strings / denoise / vae decode with your model's.
#   3. Keep the argv contract identical — the UI bridge is the same for all adapters.
#   4. Keep _select_prompt / _load_prompt_json unchanged — they are shared infra.

from std.sys import argv
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.qwen25vl_encoder import (
    Qwen25VLEncoder,
    Qwen25VLConfig,
)
from serenitymojo.models.dit.qwenimage_dit import QwenImageDitOffloaded
from serenitymojo.models.vae.qwenimage_tiled_decode import qwenimage_tiled_decode
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.layout import patchify, unpatchify
from serenitymojo.ops.tensor_algebra import slice
from serenitymojo.sampling.flow_match import Scheduler, cfg_qwen
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.training.sample_prompt_config import (
    SamplePrompt, SamplePromptConfig, read_sample_prompt_config,
)

# ── Model paths (comptime; override by editing this file or adding config support) ──
comptime QWENIMAGE_DIR = "/home/alex/.serenity/models/checkpoints/qwen-image-2512"
comptime TEXT_ENCODER_DIR = QWENIMAGE_DIR + "/text_encoder"
comptime TOK_JSON = QWENIMAGE_DIR + "/tokenizer/tokenizer.json"
comptime DIT_DIR = QWENIMAGE_DIR + "/transformer"
comptime VAE_DIR = QWENIMAGE_DIR + "/vae"

# ── Tokenizer / encoder constants (verbatim from the runner) ──
comptime PAD_ID = 151643
comptime DROP_IDX = 34
comptime N_TXT_KEPT = 512
comptime N_ENC = N_TXT_KEPT + DROP_IDX   # 546
comptime EXTRACT_LAYER = 27

# ── Latent / DiT shape constants (comptime-fixed; see header for rationale) ──
comptime LH = 128
comptime LW = 128
comptime PATCH = 2
comptime N_IMG = (LH // PATCH) * (LW // PATCH)
comptime S_POS = N_IMG + N_TXT_KEPT
comptime S_NEG = N_IMG + N_TXT_KEPT
comptime FRAME = 1
comptime FH = LH // PATCH
comptime FW = LW // PATCH

# ── Sampler constants (comptime-fixed today; see header) ──
comptime STEPS = 30
comptime CFG = Float32(4.0)
comptime SEED = UInt64(42)


# ── Caption pair produced by the text encoder ──
@fieldwise_init
struct QwenCaps(Movable):
    var pos: Tensor
    var neg: Tensor
    var real_pos: Int
    var real_neg: Int


@fieldwise_init
struct EncodedCaption(Movable):
    var hidden: Tensor
    var real_len: Int

    def into_caps(deinit self, deinit neg: EncodedCaption) -> QwenCaps:
        return QwenCaps(self.hidden^, neg.hidden^, self.real_len, neg.real_len)


# ── Qwen chat template ──
def _qwen_template(prompt: String) -> String:
    return (
        String("<|im_start|>system\nDescribe the image by detailing the color,"
        " shape, size, texture, quantity, text, spatial relationships of the"
        " objects and background:<|im_end|>\n<|im_start|>user\n")
        + prompt
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


# Tokenize and pad to N_ENC; return (ids, real_kept_len).
def _tokenize_for_encoder(
    tok: Qwen3Tokenizer, prompt: String
) raises -> Tuple[List[Int], Int]:
    var ids_full = tok.encode(_qwen_template(prompt))
    var real_len = len(ids_full)
    if real_len <= DROP_IDX:
        raise Error(
            String("qwenimage_sample_cli: prompt tokenized to ")
            + String(real_len)
            + " tokens, not enough past DROP_IDX="
            + String(DROP_IDX)
        )
    if real_len > N_ENC:
        raise Error(
            String("qwenimage_sample_cli: prompt tokenized to ")
            + String(real_len)
            + " tokens, exceeding N_ENC="
            + String(N_ENC)
            + " (N_TXT_KEPT="
            + String(N_TXT_KEPT)
            + "); shorten the prompt or raise N_TXT_KEPT"
        )
    var real_kept_len = real_len - DROP_IDX
    var ids = List[Int](capacity=N_ENC)
    for i in range(real_len):
        ids.append(ids_full[i])
    for _ in range(N_ENC - real_len):
        ids.append(PAD_ID)
    print("  tokens:", real_len, "-> drop", DROP_IDX, "-> kept", real_kept_len)
    return (ids^, real_kept_len)


# Encode a single runtime prompt string → [1, N_TXT_KEPT, 3584] trimmed hidden.
def _encode_trimmed(
    enc: Qwen25VLEncoder, tok: Qwen3Tokenizer, prompt: String, ctx: DeviceContext
) raises -> EncodedCaption:
    var tup = _tokenize_for_encoder(tok, prompt)
    var ids = tup[0].copy()
    var real_kept_len = tup[1]
    var pre = enc.encode(ids, EXTRACT_LAYER, ctx)
    var full = enc.final_norm(pre, ctx)
    var hidden = slice(full, 1, DROP_IDX, N_TXT_KEPT, ctx)
    return EncodedCaption(hidden^, real_kept_len)


# Encode runtime prompt + negative into a QwenCaps pair.
# This is the key difference from the standalone runner: prompt and negative
# are runtime String arguments, not the PROMPT/NEGATIVE comptime constants.
def encode_captions_from_strings(
    prompt: String, negative: String, ctx: DeviceContext
) raises -> QwenCaps:
    print("[text] Qwen2.5-VL encoder, N_TXT_KEPT=", N_TXT_KEPT, "DROP_IDX=", DROP_IDX)
    var tok = Qwen3Tokenizer(TOK_JSON)
    var enc = Qwen25VLEncoder.load(
        TEXT_ENCODER_DIR, Qwen25VLConfig.qwen_image(), ctx
    )
    var pos = _encode_trimmed(enc, tok, prompt, ctx)
    var neg = _encode_trimmed(enc, tok, negative, ctx)
    return pos^.into_caps(neg^)


# Build initial latent packed tensor.
def initial_latent_packed(ctx: DeviceContext) raises -> Tensor:
    var nchw_shape = List[Int]()
    nchw_shape.append(1)
    nchw_shape.append(16)
    nchw_shape.append(LH)
    nchw_shape.append(LW)
    var noise = randn(nchw_shape^, SEED, STDtype.BF16, ctx)
    return patchify(noise, PATCH, ctx)


# CFG denoise loop (STEPS/CFG/SEED are comptime-fixed; see header).
def denoise(caps: QwenCaps, ctx: DeviceContext) raises -> Tensor:
    print("[denoise] loading Qwen-Image MMDiT (block-streamed)")
    var model = QwenImageDitOffloaded.load(DIT_DIR, ctx)
    var sched = Scheduler.qwen(STEPS, Float32(N_IMG))
    var sigmas = sched.sigmas()
    print("[denoise]", STEPS, "steps, CFG", CFG, "seed", SEED)
    print("  real_pos=", caps.real_pos, "real_neg=", caps.real_neg)
    var x = initial_latent_packed(ctx)
    for i in range(STEPS):
        print("  step", i + 1, "/", STEPS, "sigma", sigmas[i], "->", sigmas[i + 1])
        var preds = model.forward_cfg_mixed_text[
            N_IMG, N_TXT_KEPT, S_POS, N_TXT_KEPT, S_NEG
        ](
            x, caps.pos, caps.neg, sigmas[i],
            caps.real_pos, caps.real_neg,
            FRAME, FH, FW, ctx,
        )
        var pred = cfg_qwen(preds.pos, preds.neg, CFG, ctx)
        x = sched.step(x, pred, i, ctx)
    return x^


# ── Prompt selection helpers (verbatim pattern from zimage_generate.mojo) ──

def _select_prompt(sample_cfg: SamplePromptConfig, wanted: String) raises -> SamplePrompt:
    if len(sample_cfg.prompts) == 0:
        raise Error("qwenimage_sample_cli: sample prompt JSON has no prompts")
    if wanted == String(""):
        return sample_cfg.prompts[0].copy()
    for i in range(len(sample_cfg.prompts)):
        if sample_cfg.prompts[i].label == wanted:
            return sample_cfg.prompts[i].copy()
    raise Error(String("qwenimage_sample_cli: prompt id not found: ") + wanted)


def _load_prompt_json(
    path: String, wanted: String,
    mut prompt: String, mut negative: String,
) raises:
    var sample_cfg = read_sample_prompt_config(path)
    var p = _select_prompt(sample_cfg, wanted)
    if p.frames != 1:
        raise Error("qwenimage_sample_cli: only image prompts (frames=1) are supported")
    prompt = p.prompt.copy()
    negative = p.negative.copy()
    # steps/cfg/seed/width/height are comptime-fixed today; log what the JSON
    # requested so the caller knows what was ignored.
    print(
        "  [info] sample prompt requests steps=", p.steps, "cfg=", p.cfg,
        "seed=", p.seed, "size=", p.width, "x", p.height,
        "→ all ignored (comptime fixed); prompt + negative honored.",
    )


# ── Main entry ──────────────────────────────────────────────────────────────

def main() raises:
    var a = argv()
    if len(a) < 6:
        print(
            "usage: qwenimage_sample_cli <config.json> <lora|-> <sample_prompts.json>"
            " <prompt_id> <out.png>"
        )
        print("  argv[1] config   — accepted, ignored (model dirs are comptime)")
        print("  argv[2] lora     — accepted, ignored (no LoRA support yet)")
        print("  argv[3] prompts  — serenity.sample_prompts.v1 JSON")
        print("  argv[4] id       — prompt label, or '' for first")
        print("  argv[5] out.png  — output image path")
        raise Error("qwenimage_sample_cli: need exactly 5 arguments")

    # argv[1]: config — accepted, not used today.
    var _config_path = String(a[1])

    # argv[2]: lora path or sentinel; accepted, not used today.
    var lora_raw = String(a[2])
    var _lora_path = String("")
    if lora_raw != String("-") and lora_raw != String("base") and lora_raw != String(""):
        _lora_path = lora_raw
        print("[lora] path provided but ignored (Qwen-Image LoRA not wired yet):", _lora_path)

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

    print("=== Qwen-Image sample CLI ===")
    print("  prompts:", prompts_json, " id:", prompt_id)
    print("  output:", out_png)
    print("  [prompt]", prompt)
    if negative != String(""):
        print("  [negative]", negative)

    var ctx = DeviceContext()

    # Encode runtime prompt + negative (the key difference from the standalone
    # runner — we are NOT using the PROMPT/NEGATIVE comptime constants).
    var caps = encode_captions_from_strings(prompt, negative, ctx)

    # Denoise (STEPS/CFG/SEED are comptime-fixed; see file header).
    var tokens = denoise(caps, ctx)

    # VAE decode.
    print("[vae] unpack + tiled decode")
    var latent = unpatchify(tokens, 16, LH, LW, PATCH, ctx)
    latent = cast_tensor(latent, STDtype.BF16, ctx)
    var img = qwenimage_tiled_decode[LH, LW](latent, VAE_DIR, ctx)
    var sh = img.shape()
    print("  image shape:", sh[0], sh[1], sh[2], sh[3])

    # Save.
    save_png(img, out_png, ctx, ValueRange.SIGNED)
    print("[done] saved:", out_png)
