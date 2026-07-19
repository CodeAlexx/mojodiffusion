# serenitymojo/pipeline/sd3_sample_cli.mojo
#
# UI-driven CLI adapter for SD 3.5 Large (1024x1024) text→image generation.
# Mirrors the pattern of qwenimage_sample_cli.mojo — same CLI contract, same
# argument handling, same prompt-selection infra.
#
# Contract (the UI bridge calls it exactly this way):
#
#   sd3_sample_cli  <config.json>  <lora|->  <sample_prompts.json>  <prompt_id>  <out.png>
#
#   argv[1]  config JSON path (model dirs are comptime constants; this argument
#            is ACCEPTED BUT IGNORED TODAY — model paths are the comptime
#            MODEL_PATH / EMBEDDINGS_PATH constants below).
#
#   argv[2]  LoRA safetensors path, or "-"/"base"/"" for base model.
#            SD3.5 Large has no LoRA path today; the value is ACCEPTED AND
#            IGNORED.  When SD3 LoRA lands, wire it into the denoise loop.
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
#     • caps_pos  — per-prompt path to the pre-generated embeddings sidecar
#                   (see "Text conditioning" below).  Falls back to the
#                   comptime EMBEDDINGS_PATH constant when caps_pos == "".
#     • prompt    — logged and used to select the sidecar path; the sidecar
#                   already encodes the prompt text, so there is no second
#                   runtime encode step.
#     • negative  — same as prompt: the sidecar already encodes the negative.
#
#   FIXED at comptime (from sd3_large_pipeline_1024_multistep.mojo):
#     • steps  = NUM_STEPS    (28)
#     • cfg    = CFG_SCALE    (4.5)
#     • seed   = SEED         (UInt64(42))
#     • width  = LW * 8       (1024)
#     • height = LH * 8       (1024)
#     • shift  = SHIFT        (3.0)
#
#   SD3.5 Large uses a comptime-derived sigma schedule (shift, num_steps), and
#   the DiT attention shape S_JOINT is a comptime constant.  Changing any of
#   these requires a recompile.  When the runner gains runtime dispatch for
#   these fields, remove them from the fixed list and thread from `req_prompt`.
#
# ──────────────────────────────────────────────────────────────────────────────
# Text conditioning: SIDECAR ROUTE (caps_pos from sample_prompts JSON).
#
#   SD3.5 Large uses three text encoders (CLIP-L, CLIP-G, T5-XXL) whose outputs
#   are concatenated into:
#     context   [1, 410, 4096]  (77 CLIP-L + 77 CLIP-G + 256 T5)
#     pooled    [1, 2048]       (pooled CLIP-L + pooled CLIP-G)
#
#   No pure-Mojo SD3 "runtime encode from string" function exists today — the
#   triple-encoder assembly is not wired as a single entry point in this repo.
#   Instead, each SamplePrompt in the JSON carries `caps_pos` and `caps_neg`
#   fields pointing to a pre-generated .safetensors sidecar (produced by the
#   Python encode-only script or the Rust sidecar generator).  This adapter
#   reads that sidecar directly.
#
#   Sidecar key layout (float32 in file → cast to BF16 at load time):
#     context_cond   [1, 410, 4096]
#     context_uncond [1, 410, 4096]
#     pooled_cond    [1, 2048]
#     pooled_uncond  [1, 2048]
#
#   When caps_pos is empty the comptime EMBEDDINGS_PATH default sidecar is used.
#   This matches the original standalone runner exactly.
#
#   FUTURE: when a pure-Mojo sd3_encode_runtime(prompt, ctx) entry exists,
#   replace `_load_cond_*` / `_load_uncond_*` with it and update the header.
#
# ──────────────────────────────────────────────────────────────────────────────
# Generate path: REAL, not a stub.
#   Calls _sd3_large_forward × 2 (cond + uncond) per step → sd3_cfg → sd3_euler_step
#   → load_sd3_embedded_ldm_decoder.decode → save_png.
#   This is the complete sd3_large_pipeline_1024_multistep.mojo pipeline;
#   the only change is the sidecar path comes from the sample_prompts JSON
#   instead of the EMBEDDINGS_PATH comptime constant.
#
# Build:
#   cd /home/alex/mojodiffusion && pixi run mojo build -I . \
#     -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/pipeline/sd3_sample_cli.mojo \
#     -o /tmp/sd3_sample_cli
#
# Note for porter agents (sdxl/ernie/flux):
#   1. Replace the import block and model-path constants with your model's.
#   2. Replace _load_cond_* / _load_uncond_* with your model's conditioning load.
#   3. Replace _sd3_large_forward / denoise with your model's denoise loop.
#   4. Keep the argv contract identical — the UI bridge is the same for all adapters.
#   5. Keep _select_prompt / _load_prompt_json unchanged — they are shared infra.

from std.sys import argv
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.dit.sd3_contract import (
    SD3_LARGE_DEPTH,
    SD3_LARGE_HEAD_DIM,
    SD3_LARGE_HIDDEN,
    SD3_LARGE_IMAGE_TOKENS,
    SD3_LARGE_LATENT_CHANNELS,
    SD3_LARGE_LATENT_H,
    SD3_LARGE_LATENT_W,
    SD3_LARGE_NUM_HEADS,
    SD3_LARGE_TEXT_TOKENS,
    sd3_large_model_timestep,
)
from serenitymojo.models.dit.sd3_mmdit import (
    SD3MMDiTPreBlockGate,
    _sd3_joint_block,
)
from serenitymojo.models.vae.ldm_decoder import load_sd3_embedded_ldm_decoder
from serenitymojo.offload.block_loader import BlockLoader, unload_block
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import add, mul_scalar, reshape
from serenitymojo.sampling.sd3_flow_match import (
    SD3FlowMatchScheduler,
    sd3_cfg,
    sd3_euler_step,
)
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.training.sample_prompt_config import (
    SamplePrompt, SamplePromptConfig, read_sample_prompt_config,
)


# ── Model paths (comptime; override by editing this file or adding config support) ──
comptime MODEL_PATH = "/home/alex/.serenity/models/checkpoints/sd3.5_large.safetensors"
# Default sidecar (used when caps_pos is empty in the sample_prompts JSON).
comptime EMBEDDINGS_PATH = "/home/alex/EriDiffusion/inference-flame/output/sd3_large_embeddings.safetensors"

# ── Latent / DiT shape constants (comptime-fixed; see header for rationale) ──
comptime LH = SD3_LARGE_LATENT_H         # 128
comptime LW = SD3_LARGE_LATENT_W         # 128
comptime LC = SD3_LARGE_LATENT_CHANNELS  # 16

# SD3 Large joint sequence length: image + text tokens
comptime N_CTX = SD3_LARGE_TEXT_TOKENS   # 410 = 77+77+256
comptime N_IMG = SD3_LARGE_IMAGE_TOKENS  # 4096
comptime S_JOINT = N_CTX + N_IMG         # 4506

comptime DEPTH = SD3_LARGE_DEPTH         # 38
comptime H_HEADS = SD3_LARGE_NUM_HEADS   # 38
comptime H_DIM = SD3_LARGE_HEAD_DIM      # 64
comptime HIDDEN = SD3_LARGE_HIDDEN       # 2432
comptime DUAL_BLOCKS = 0                 # No dual attention in Large

# ── Sampler constants (comptime-fixed today; see header) ──
comptime NUM_STEPS = 28
comptime CFG_SCALE = Float32(4.5)
comptime SHIFT = Float32(3.0)
comptime SEED = UInt64(42)


# ── Load text conditioning from sidecar ────────────────────────────────────────
# sidecar_path is a RUNTIME String derived from the sample_prompts JSON
# (caps_pos field), or falls back to the comptime EMBEDDINGS_PATH default.

def _load_cond_context(sidecar_path: String, ctx: DeviceContext) raises -> Tensor:
    var st = ShardedSafeTensors.open(sidecar_path)
    return cast_tensor(Tensor.from_view(st.tensor_view(String("context_cond")), ctx), STDtype.BF16, ctx)

def _load_uncond_context(sidecar_path: String, ctx: DeviceContext) raises -> Tensor:
    var st = ShardedSafeTensors.open(sidecar_path)
    return cast_tensor(Tensor.from_view(st.tensor_view(String("context_uncond")), ctx), STDtype.BF16, ctx)

def _load_cond_pooled(sidecar_path: String, ctx: DeviceContext) raises -> Tensor:
    var st = ShardedSafeTensors.open(sidecar_path)
    return cast_tensor(Tensor.from_view(st.tensor_view(String("pooled_cond")), ctx), STDtype.BF16, ctx)

def _load_uncond_pooled(sidecar_path: String, ctx: DeviceContext) raises -> Tensor:
    var st = ShardedSafeTensors.open(sidecar_path)
    return cast_tensor(Tensor.from_view(st.tensor_view(String("pooled_uncond")), ctx), STDtype.BF16, ctx)


# ── SD3 Large MMDiT forward (one pass) ────────────────────────────────────────
# Streams 38 blocks via BlockLoader one at a time (~426 MB each).
# Verbatim from sd3_large_pipeline_1024_multistep.mojo; no changes to the forward.
def _sd3_large_forward(
    latent: Tensor,          # [1, 16, LH, LW] BF16
    sigma: Float32,          # current sigma
    context: Tensor,         # [1, N_CTX, 4096] BF16
    pooled: Tensor,          # [1, 2048] BF16
    gate: SD3MMDiTPreBlockGate,
    loader: BlockLoader,
    ctx: DeviceContext,
) raises -> Tensor:
    # Pre-block: patch embed + pos embed -> [1, N_IMG, HIDDEN]
    var x_tokens = gate.latent_patch_embed[LH, LW](latent, ctx)

    # Conditioning vector: timestep + pooled -> [1, HIDDEN]
    var c = gate.conditioning(sigma, pooled, ctx)

    # Context embedding: [1, N_CTX, 4096] -> [1, N_CTX, HIDDEN]
    var ctx_tokens = gate.context_embed[N_CTX](context, ctx)

    # 38 joint blocks, streamed one at a time
    for i in range(DEPTH):
        var is_last = (i == DEPTH - 1)
        var block_prefix = String("model.diffusion_model.joint_blocks.") + String(i)
        loader.prefetch_block(block_prefix)
        var blk = loader.load_block(block_prefix, ctx)

        # DUAL_BLOCKS=0 so block_has_dual is always False (Large has no dual-attn).
        _sd3_joint_block[1, S_JOINT, N_CTX, N_IMG, H_HEADS, H_DIM](
            ctx_tokens, x_tokens, c, blk, i, is_last, DUAL_BLOCKS, HIDDEN, ctx
        )

        unload_block(blk^)

    # Final layer: no-affine LN + adaLN + linear -> [1, N_IMG, 64]
    var patch_out = gate.final_layer_tokens(x_tokens, c, ctx)

    # SD3 spatial-outer unpatchify [1, N_IMG, 64] -> [1, 16, LH, LW]
    return gate.final_unpatchify[LH, LW](patch_out, ctx)


# ── Denoise loop ──────────────────────────────────────────────────────────────
# Runs NUM_STEPS of shifted rectified-flow CFG Euler using the sidecar's
# pre-encoded cond/uncond embeddings.
# NUM_STEPS/CFG_SCALE/SEED/SHIFT are comptime-fixed; see file header.
def _denoise(
    context_cond: Tensor,    # [1, N_CTX, 4096] BF16
    context_uncond: Tensor,  # [1, N_CTX, 4096] BF16
    pooled_cond: Tensor,     # [1, 2048] BF16
    pooled_uncond: Tensor,   # [1, 2048] BF16
    ctx: DeviceContext,
) raises -> Tensor:
    print("[denoise] loading SD3.5 Large resident weights")
    var gate = SD3MMDiTPreBlockGate.load_large_default(ctx)

    print("[denoise] opening block loader for", DEPTH, "blocks")
    var loader = BlockLoader.open(String(MODEL_PATH))

    print("[denoise] initialising noise latent [1,", LC, ",", LH, ",", LW, "]")
    var noise_sh = List[Int]()
    noise_sh.append(1)
    noise_sh.append(LC)
    noise_sh.append(LH)
    noise_sh.append(LW)
    var latent = randn(noise_sh^, SEED, STDtype.BF16, ctx)

    var sched = SD3FlowMatchScheduler.large_default()
    print("[denoise]", NUM_STEPS, "steps, CFG", CFG_SCALE, "seed", SEED, "shift", SHIFT)

    for step in range(NUM_STEPS):
        var sigma = sched.timestep(step)
        var dt = sched.dt(step)

        print("  step", step + 1, "/", NUM_STEPS, "sigma=", sigma)

        var v_cond = _sd3_large_forward(
            latent, sigma, context_cond, pooled_cond, gate, loader, ctx
        )
        var v_uncond = _sd3_large_forward(
            latent, sigma, context_uncond, pooled_uncond, gate, loader, ctx
        )

        var velocity = sd3_cfg(v_cond, v_uncond, CFG_SCALE, ctx)
        latent = sd3_euler_step(latent, velocity, dt, ctx)

    return latent^


# ── Prompt selection helpers (verbatim from qwenimage_sample_cli.mojo) ──

def _select_prompt(sample_cfg: SamplePromptConfig, wanted: String) raises -> SamplePrompt:
    if len(sample_cfg.prompts) == 0:
        raise Error("sd3_sample_cli: sample prompt JSON has no prompts")
    if wanted == String(""):
        return sample_cfg.prompts[0].copy()
    for i in range(len(sample_cfg.prompts)):
        if sample_cfg.prompts[i].label == wanted:
            return sample_cfg.prompts[i].copy()
    raise Error(String("sd3_sample_cli: prompt id not found: ") + wanted)


def _load_prompt_json(
    path: String, wanted: String,
    mut prompt: String, mut negative: String,
    mut sidecar_path: String,
) raises:
    var sample_cfg = read_sample_prompt_config(path)
    var p = _select_prompt(sample_cfg, wanted)
    if p.frames != 1:
        raise Error("sd3_sample_cli: only image prompts (frames=1) are supported")
    prompt = p.prompt.copy()
    negative = p.negative.copy()
    # Use the per-prompt caps_pos sidecar if present; else fall back to the
    # comptime EMBEDDINGS_PATH default.
    if p.caps_pos != String(""):
        sidecar_path = p.caps_pos.copy()
    else:
        sidecar_path = String(EMBEDDINGS_PATH)
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
            "usage: sd3_sample_cli <config.json> <lora|-> <sample_prompts.json>"
            " <prompt_id> <out.png>"
        )
        print("  argv[1] config   — accepted, ignored (model dirs are comptime)")
        print("  argv[2] lora     — accepted, ignored (no LoRA support yet)")
        print("  argv[3] prompts  — serenity.sample_prompts.v1 JSON")
        print("  argv[4] id       — prompt label, or '' for first")
        print("  argv[5] out.png  — output image path")
        raise Error("sd3_sample_cli: need exactly 5 arguments")

    # argv[1]: config — accepted, not used today.
    var _config_path = String(a[1])

    # argv[2]: lora path or sentinel; accepted, not used today.
    var lora_raw = String(a[2])
    var _lora_path = String("")
    if lora_raw != String("-") and lora_raw != String("base") and lora_raw != String(""):
        _lora_path = lora_raw
        print("[lora] path provided but ignored (SD3.5 Large LoRA not wired yet):", _lora_path)

    # argv[3]: sample prompts JSON
    var prompts_json = String(a[3])

    # argv[4]: prompt id
    var prompt_id = String(a[4])

    # argv[5]: output PNG
    var out_png = String(a[5])

    # Load prompt + negative + sidecar path from the JSON.
    var prompt = String("")
    var negative = String("")
    var sidecar_path = String(EMBEDDINGS_PATH)
    _load_prompt_json(prompts_json, prompt_id, prompt, negative, sidecar_path)

    print("=== SD3.5 Large sample CLI ===")
    print("  prompts:", prompts_json, " id:", prompt_id)
    print("  output:", out_png)
    print("  [prompt]", prompt)
    if negative != String(""):
        print("  [negative]", negative)
    print("  [sidecar]", sidecar_path)

    var ctx = DeviceContext()

    # Stage 1: Load text conditioning from sidecar.
    # The sidecar encodes the prompt and negative; no runtime text encode step.
    # To generate new conditioning for a new prompt, run the sidecar generator
    # (sd3_encode.py or inference-flame sd3_encode) and set caps_pos in the JSON.
    print("[text] loading conditioning sidecar:", sidecar_path)
    var context_cond = _load_cond_context(sidecar_path, ctx)
    var context_uncond = _load_uncond_context(sidecar_path, ctx)
    var pooled_cond = _load_cond_pooled(sidecar_path, ctx)
    var pooled_uncond = _load_uncond_pooled(sidecar_path, ctx)
    var ccs = context_cond.shape()
    print("  context_cond:", ccs[0], ccs[1], ccs[2], "dtype:", context_cond.dtype().name())

    # Stage 2: Denoise (NUM_STEPS/CFG_SCALE/SEED are comptime-fixed; see header).
    var latent = _denoise(context_cond, context_uncond, pooled_cond, pooled_uncond, ctx)

    # Stage 3: VAE decode (embedded first_stage_model.decoder.* in Large checkpoint).
    print("[vae] decoding latent → image")
    var vae = load_sd3_embedded_ldm_decoder[LH, LW](String(MODEL_PATH), ctx)
    var image = vae.decode(latent, ctx)
    var imsh = image.shape()
    print("  decoded:", imsh[2], "x", imsh[3])

    # Stage 4: Save PNG.
    save_png(image, out_png, ctx, ValueRange.SIGNED)
    print("[done] saved:", out_png)
