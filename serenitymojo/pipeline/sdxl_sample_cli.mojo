# serenitymojo/pipeline/sdxl_sample_cli.mojo
#
# UI-driven CLI adapter for SDXL 1024x1024 generation.
# Mirrors the pattern of sd3_sample_cli.mojo — same CLI contract, same
# argument handling, same prompt-selection infra.
#
# Contract (the UI bridge calls it exactly this way):
#
#   sdxl_sample_cli  <config.json>  <lora|->  <sample_prompts.json>  <prompt_id>  <out.png>
#
#   argv[1]  config JSON path — ACCEPTED BUT IGNORED TODAY.  Model paths are
#            the comptime UNET_PATH / VAE_PATH / EMBEDDINGS_PATH constants below.
#
#   argv[2]  LoRA safetensors path, or "-"/"base"/"" for base model.
#            SDXL LoRA inference is not wired here today; the value is ACCEPTED
#            AND IGNORED.  When LoRA lands, load via the SDXLUNet overlay path
#            and remove from the ignored list.
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
#     • prompt    — honored via the embeddings sidecar path (caps_pos in the JSON).
#                   When caps_pos is non-empty, that sidecar encodes the prompt.
#                   When caps_pos is empty, the comptime EMBEDDINGS_PATH is used
#                   (single pre-cached embedding for the default prompt).
#     • negative  — same as prompt via the same sidecar.
#     • lora      — ACCEPTED, IGNORED TODAY (see argv[2] note above).
#
#   FIXED at comptime (shape-dependent constants matching the UNet kernel):
#     • steps  = NUM_STEPS   (30)
#     • cfg    = CFG         (7.5)
#     • seed   = SEED        (UInt64(42))
#     • width  = LW * 8      (1024)
#     • height = LH * 8      (1024)
#
#   The UNet kernel for SDXL is compiled at fixed LH×LW (latent dims) and the
#   Euler scheduler sigma table is built for NUM_STEPS.  Changing any of these
#   requires a recompile.  Steps/cfg/seed could be threaded as runtime values
#   once the scheduler is dynamic; document that TODO here.
#
# ──────────────────────────────────────────────────────────────────────────────
# Text conditioning: SIDECAR ROUTE (caps_pos from sample_prompts JSON).
#
#   SDXL uses two CLIP encoders (CLIP-L + CLIP-G/OpenCLIP) whose outputs are
#   assembled into:
#     context   [1, 77, 2048]  — cat([clip_l_hidden, clip_g_hidden], dim=2)
#     y         [1, 2816]      — cat([clip_l_pool, clip_g_text_embeds, zeros[768]])
#
#   A pure-Mojo CLIP tokenizer for SDXL (vocab 49408, char-level BPE) does not
#   exist in the tree today — the existing Qwen3Tokenizer is byte-level BPE and
#   the two formats are incompatible.  Therefore, runtime text→embedding encoding
#   is NOT performed here.
#
#   Instead, each SamplePrompt in the JSON MAY carry a `caps_pos` field pointing
#   to a pre-generated .safetensors sidecar (produced by inference-flame's
#   sdxl_encode binary or the Python equivalent).  This adapter loads that sidecar.
#   When caps_pos is empty the comptime EMBEDDINGS_PATH default sidecar is used.
#
#   Sidecar key layout (matching sdxl_pipeline_full_smoke.mojo verbatim):
#     "context"        [1, 77, 2048] BF16   — CLIP-L+G cross-attention context (cond)
#     "context_uncond" [1, 77, 2048] BF16   — same for unconditional (negative)
#     "y"              [1, 2816]     BF16   — ADM pooled+time-ids vector (cond)
#     "y_uncond"       [1, 2816]     BF16   — ADM vector for unconditional
#
#   FUTURE: when a pure-Mojo sdxl_encode_runtime(prompt, ctx) entry exists,
#   replace `_load_*` helpers with it and update the header.
#
# ──────────────────────────────────────────────────────────────────────────────
# Generate path: IS THE UNDERLYING PATH REAL OR A SMOKE/STUB?
#
#   sdxl_pipeline_full_smoke.mojo is named "*_full_smoke" but IS a real full
#   30-step generate.  Its header states explicitly: "Same path as
#   sdxl_pipeline_smoke.mojo, but uses the full 30-step SDXL denoise loop and
#   writes a separate PNG artifact."  The smoke qualifier refers to its use as an
#   integration smoke test (long GPU run), NOT to it being a reduced/stubbed path.
#
#   THIS FILE replicates the same stage sequence verbatim:
#     load embeddings sidecar → load SDXLUNet → Euler denoise (NUM_STEPS) →
#     load_sdxl_ldm_decoder → decode → save_png.
#   It is therefore a REAL full SDXL generate, identical in substance to the
#   smoke runner, differing only in that the embeddings sidecar path comes from
#   the sample_prompts JSON rather than the EMBEDDINGS_PATH comptime constant.
#
#   CAVEAT: the underlying path requires pre-generated CLIP embeddings sidecar.
#   A runtime from-string encode step does NOT exist in this adapter today.  If
#   the UI sets `precache_required: false` and does not also provide a caps_pos
#   sidecar in the JSON, the adapter falls back to the comptime EMBEDDINGS_PATH
#   default (which encodes the original developer test prompt, not the user's).
#   This fallback is documented with a printed warning at runtime.
#
# Build:
#   cd /home/alex/mojodiffusion && pixi run mojo build -I . \
#     -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/pipeline/sdxl_sample_cli.mojo \
#     -o /tmp/sdxl_sample_cli

from std.sys import argv
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.registry.checkpoints import default_manifest_by_id
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import mul_scalar
from serenitymojo.models.dit.sdxl_unet import SDXLUNet
from serenitymojo.models.vae.ldm_decoder import load_sdxl_ldm_decoder
from serenitymojo.models.dit.sdxl_contract import (
    sdxl_default_cached_embeddings_path,
    validate_sdxl_pipeline_contract,
)
from serenitymojo.sampling.sdxl_euler import (
    SDXLEulerScheduler,
    sdxl_cfg,
    sdxl_euler_step,
    sdxl_initial_noise_sigma,
    sdxl_input_scale,
)
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.training.sample_prompt_config import (
    SamplePrompt, SamplePromptConfig, read_sample_prompt_config,
)


# ── Model paths (comptime; override by editing this file or adding config support) ──
comptime WIDTH = 1024
comptime HEIGHT = 1024
comptime LH = HEIGHT // 8      # 128
comptime LW = WIDTH // 8       # 128

# ── Sampler constants (comptime-fixed today; see header) ──
comptime NUM_STEPS = 30
comptime CFG = Float32(7.5)
comptime SEED = UInt64(42)


# ── Sidecar helpers ──────────────────────────────────────────────────────────
# Load each of the four embedding tensors from the sidecar safetensors file.
# The sidecar keys match sdxl_pipeline_full_smoke.mojo verbatim.

def _load_named(sidecar_path: String, name: String, ctx: DeviceContext) raises -> Tensor:
    var st = ShardedSafeTensors.open(sidecar_path)
    return cast_tensor(Tensor.from_view(st.tensor_view(name), ctx), STDtype.BF16, ctx)

def _load_context(sidecar_path: String, ctx: DeviceContext) raises -> Tensor:
    return _load_named(sidecar_path, String("context"), ctx)

def _load_context_uncond(sidecar_path: String, ctx: DeviceContext) raises -> Tensor:
    return _load_named(sidecar_path, String("context_uncond"), ctx)

def _load_y(sidecar_path: String, ctx: DeviceContext) raises -> Tensor:
    return _load_named(sidecar_path, String("y"), ctx)

def _load_y_uncond(sidecar_path: String, ctx: DeviceContext) raises -> Tensor:
    return _load_named(sidecar_path, String("y_uncond"), ctx)


# ── SDXL Euler denoise loop ────────────────────────────────────────────────────
# Runs NUM_STEPS of epsilon-prediction Euler CFG using the pre-loaded embeddings.
# NUM_STEPS/CFG/SEED/LH/LW are all comptime; see file header for rationale.
def _denoise(
    context: Tensor,        # [1, 77, 2048] BF16
    context_uncond: Tensor, # [1, 77, 2048] BF16
    y: Tensor,              # [1, 2816] BF16
    y_uncond: Tensor,       # [1, 2816] BF16
    ctx: DeviceContext,
) raises -> Tensor:
    print("[unet] loading SDXLUNet[", LH, ",", LW, "]")
    var manifest = default_manifest_by_id(String("sdxl"))
    var unet = SDXLUNet[LH, LW].load(manifest.denoiser_path, ctx)
    print("  UNet loaded:", manifest.denoiser_path)

    var sched = SDXLEulerScheduler(NUM_STEPS)
    var sigmas = sched.sigmas()

    var nsh = List[Int]()
    nsh.append(1)
    nsh.append(4)
    nsh.append(LH)
    nsh.append(LW)
    var noise = randn(nsh^, SEED, STDtype.F32, ctx)
    var init_sigma = sdxl_initial_noise_sigma(sigmas[0])
    var x = mul_scalar(noise, init_sigma, ctx)

    print("[denoise]", NUM_STEPS, "steps  CFG", CFG, "  seed", SEED)

    for i in range(NUM_STEPS):
        var sigma = sigmas[i]
        var sigma_next = sigmas[i + 1]
        var t_i = sched.timestep(i)

        var c_in = sdxl_input_scale(sigma)
        var x_in_f32 = mul_scalar(x, c_in, ctx)
        var x_in = cast_tensor(x_in_f32, STDtype.BF16, ctx)

        var eps_cond = cast_tensor(unet.forward(x_in, t_i, context, y, ctx), STDtype.F32, ctx)
        var eps_uncond = cast_tensor(
            unet.forward(x_in, t_i, context_uncond, y_uncond, ctx), STDtype.F32, ctx
        )
        var eps = sdxl_cfg(eps_cond, eps_uncond, CFG, ctx)
        x = sdxl_euler_step(x, eps, sigma, sigma_next, ctx)

        if i == 0 or (i + 1) % 5 == 0 or i == NUM_STEPS - 1:
            print("  step", i + 1, "/", NUM_STEPS, " sigma", sigma)

    return x^


# ── Prompt selection helpers (verbatim from sd3_sample_cli.mojo) ──────────────

def _select_prompt(sample_cfg: SamplePromptConfig, wanted: String) raises -> SamplePrompt:
    if len(sample_cfg.prompts) == 0:
        raise Error("sdxl_sample_cli: sample prompt JSON has no prompts")
    if wanted == String(""):
        return sample_cfg.prompts[0].copy()
    for i in range(len(sample_cfg.prompts)):
        if sample_cfg.prompts[i].label == wanted:
            return sample_cfg.prompts[i].copy()
    raise Error(String("sdxl_sample_cli: prompt id not found: ") + wanted)


def _load_prompt_json(
    path: String, wanted: String,
    mut prompt: String, mut negative: String,
    mut sidecar_path: String,
) raises:
    var sample_cfg = read_sample_prompt_config(path)
    var p = _select_prompt(sample_cfg, wanted)
    if p.frames != 1:
        raise Error("sdxl_sample_cli: only image prompts (frames=1) are supported")
    prompt = p.prompt.copy()
    negative = p.negative.copy()
    # Use the per-prompt caps_pos sidecar if present; else fall back to the
    # comptime EMBEDDINGS_PATH default.
    if p.caps_pos != String(""):
        sidecar_path = p.caps_pos.copy()
    else:
        sidecar_path = sdxl_default_cached_embeddings_path()
        print(
            "  [warn] caps_pos is empty — falling back to default embeddings sidecar:",
            sidecar_path,
            "; this encodes the DEVELOPER TEST prompt, not the user request.",
            " To encode a specific prompt, run inference-flame's sdxl_encode and",
            " set caps_pos in the sample_prompts JSON entry.",
        )
    # steps/cfg/seed/width/height are comptime-fixed today; log what the JSON
    # requested so the caller knows what was ignored.
    print(
        "  [info] sample prompt requests steps=", p.steps, "cfg=", p.cfg,
        "seed=", p.seed, "size=", p.width, "x", p.height,
        "→ all ignored (comptime fixed); prompt + negative + sidecar honored.",
    )
    print("  [info] sidecar:", sidecar_path)


# ── Main entry ──────────────────────────────────────────────────────────────

def main() raises:
    var a = argv()
    if len(a) < 6:
        print(
            "usage: sdxl_sample_cli <config.json> <lora|-> <sample_prompts.json>"
            " <prompt_id> <out.png>"
        )
        print("  argv[1] config   — accepted, ignored (model dirs are comptime)")
        print("  argv[2] lora     — accepted, ignored (SDXL LoRA not wired yet)")
        print("  argv[3] prompts  — serenity.sample_prompts.v1 JSON")
        print("  argv[4] id       — prompt label, or '' for first")
        print("  argv[5] out.png  — output image path")
        raise Error("sdxl_sample_cli: need exactly 5 arguments")

    # argv[1]: config — accepted, not used today.
    var _config_path = String(a[1])

    # argv[2]: lora path or sentinel; accepted, not used today.
    var lora_raw = String(a[2])
    var _lora_path = String("")
    if lora_raw != String("-") and lora_raw != String("base") and lora_raw != String(""):
        _lora_path = lora_raw
        print("[lora] path provided but ignored (SDXL LoRA not wired yet):", _lora_path)

    # argv[3]: sample prompts JSON
    var prompts_json = String(a[3])

    # argv[4]: prompt id
    var prompt_id = String(a[4])

    # argv[5]: output PNG
    var out_png = String(a[5])

    # Load prompt + negative + sidecar path from the JSON.
    var prompt = String("")
    var negative = String("")
    var sidecar_path = sdxl_default_cached_embeddings_path()
    _load_prompt_json(prompts_json, prompt_id, prompt, negative, sidecar_path)

    print("=== SDXL sample CLI ===")
    print("  prompts:", prompts_json, " id:", prompt_id)
    print("  output:", out_png)
    print("  [prompt]", prompt)
    if negative != String(""):
        print("  [negative]", negative)
    print("  [sidecar]", sidecar_path)

    var ctx = DeviceContext()

    # Stage 1: contract check (validates UNet + VAE + embeddings sidecar headers).
    var manifest = default_manifest_by_id(String("sdxl"))
    validate_sdxl_pipeline_contract(manifest.denoiser_path, manifest.vae_path, sidecar_path)
    print("  contract OK")

    # Stage 2: Load text conditioning from sidecar.
    # The sidecar key layout matches sdxl_pipeline_full_smoke.mojo verbatim:
    #   context/context_uncond  [1, 77, 2048] BF16  — CLIP-L+G cross-attention
    #   y/y_uncond              [1, 2816]     BF16  — ADM pooled+time-ids vector
    # No runtime CLIP tokenize/encode step occurs here; see file header for rationale.
    print("[text] loading conditioning sidecar:", sidecar_path)
    var context = _load_context(sidecar_path, ctx)
    var context_uncond = _load_context_uncond(sidecar_path, ctx)
    var y = _load_y(sidecar_path, ctx)
    var y_uncond = _load_y_uncond(sidecar_path, ctx)
    var ccs = context.shape()
    print("  context:", ccs[0], ccs[1], ccs[2], " y:", y.shape()[1])

    # Stage 3: Denoise (NUM_STEPS/CFG/SEED/LH/LW are comptime-fixed; see header).
    var latent = _denoise(context, context_uncond, y, y_uncond, ctx)

    # Stage 4: VAE decode.
    print("[vae] decoding latent → image")
    var vae = load_sdxl_ldm_decoder[LH, LW](manifest.vae_path, ctx)
    var image = vae.decode(latent, ctx)
    var imsh = image.shape()
    print("  decoded:", imsh[2], "x", imsh[3])

    # Stage 5: Save PNG.
    save_png(image, out_png, ctx, ValueRange.SIGNED)
    print("[done] saved:", out_png)
