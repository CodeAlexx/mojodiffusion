# sd3_medium_pipeline_1024_multistep.mojo — SD3.5 Medium 1024x1024 28-step pipeline.
#
# Faithful port of sd3_medium_infer.rs:
#   1. Load cached conditioning sidecar (produced by Rust encode-only run):
#      context_cond/uncond [1, 410, 4096] BF16, pooled_cond/uncond [1, 2048] BF16.
#   2. Init noise latent [1, 16, 128, 128] BF16 (Box-Muller, seed=42).
#   3. 28-step shifted rectified-flow CFG Euler loop:
#      - sigma schedule: shift=3.0, descending
#      - model timestep: sigma * 1000
#      - CFG: uncond + 4.5 * (cond - uncond)
#      - Euler: latent += velocity * dt (dt negative)
#   4. VAE decode via embedded SD3 LDM decoder (first_stage_model.decoder.*).
#   5. Save PNG output/sd3_medium_1024_28step.png.
#
# Text conditioning: sidecar route (skip porting 3 text encoders).
#   Embeddings path: /home/alex/EriDiffusion/inference-flame/output/sd3_medium_embeddings.safetensors
#   Keys: context_cond [1,410,4096], context_uncond [1,410,4096],
#         pooled_cond [1,2048], pooled_uncond [1,2048]
#
# Architecture: SD3.5 Medium = depth 24, hidden 1536, heads 24, head_dim 64
#   blocks 0-12: dual attention (9*H adaLN)
#   blocks 13-23: single attention (6*H adaLN)
#   block 23 context: pre_only (2*H adaLN)
#
# Build:
#   pixi run mojo build -I . -Xlinker -lm \
#     serenitymojo/pipeline/sd3_medium_pipeline_1024_multistep.mojo \
#     -o /tmp/sd3_medium && /tmp/sd3_medium

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.dit.sd3_contract import (
    SD3_MEDIUM_DEPTH,
    SD3_MEDIUM_DUAL_ATTENTION_BLOCKS,
    SD3_MEDIUM_HEAD_DIM,
    SD3_MEDIUM_HIDDEN,
    SD3_MEDIUM_IMAGE_TOKENS,
    SD3_MEDIUM_LATENT_CHANNELS,
    SD3_MEDIUM_LATENT_H,
    SD3_MEDIUM_LATENT_W,
    SD3_MEDIUM_NUM_HEADS,
    SD3_MEDIUM_TEXT_TOKENS,
    sd3_medium_model_timestep,
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


# ── Configuration ──────────────────────────────────────────────────────────────
comptime EMBEDDINGS_PATH = "/home/alex/EriDiffusion/inference-flame/output/sd3_medium_embeddings.safetensors"
comptime MODEL_PATH = "/home/alex/.serenity/models/checkpoints/stablediffusion35_medium.safetensors"
comptime OUT_PNG = "/home/alex/mojodiffusion/output/sd3_medium_1024_28step.png"

comptime NUM_STEPS = 28
comptime CFG_SCALE = Float32(4.5)
comptime SHIFT = Float32(3.0)
comptime SEED = UInt64(42)

comptime LH = SD3_MEDIUM_LATENT_H      # 128
comptime LW = SD3_MEDIUM_LATENT_W      # 128
comptime LC = SD3_MEDIUM_LATENT_CHANNELS  # 16

# SD3 Medium joint sequence length: image + text tokens
comptime N_CTX = SD3_MEDIUM_TEXT_TOKENS   # 410 = 77+77+256
comptime N_IMG = SD3_MEDIUM_IMAGE_TOKENS  # 4096
comptime S_JOINT = N_CTX + N_IMG          # 4506

comptime DEPTH = SD3_MEDIUM_DEPTH         # 24
comptime H_HEADS = SD3_MEDIUM_NUM_HEADS   # 24
comptime H_DIM = SD3_MEDIUM_HEAD_DIM      # 64
comptime HIDDEN = SD3_MEDIUM_HIDDEN       # 1536
comptime DUAL_BLOCKS = SD3_MEDIUM_DUAL_ATTENTION_BLOCKS  # 13


# ── Stats helper ───────────────────────────────────────────────────────────────
def _stats(name: String, t: Tensor, ctx: DeviceContext) raises:
    var h = t.to_host(ctx)
    var n = len(h)
    if n == 0:
        print("  [stat]", name, "EMPTY")
        return
    var s: Float64 = 0.0
    var amax: Float64 = 0.0
    for i in range(n):
        var v = Float64(h[i])
        s += v
        var av = v if v >= 0.0 else -v
        if av > amax:
            amax = av
    print("  [stat]", name, "mean=", Float32(s / Float64(n)), "absmax=", Float32(amax))


# ── Load text conditioning from sidecar ────────────────────────────────────────
def _load_cond_context(ctx: DeviceContext) raises -> Tensor:
    var st = ShardedSafeTensors.open(String(EMBEDDINGS_PATH))
    return cast_tensor(Tensor.from_view(st.tensor_view(String("context_cond")), ctx), STDtype.BF16, ctx)

def _load_uncond_context(ctx: DeviceContext) raises -> Tensor:
    var st = ShardedSafeTensors.open(String(EMBEDDINGS_PATH))
    return cast_tensor(Tensor.from_view(st.tensor_view(String("context_uncond")), ctx), STDtype.BF16, ctx)

def _load_cond_pooled(ctx: DeviceContext) raises -> Tensor:
    var st = ShardedSafeTensors.open(String(EMBEDDINGS_PATH))
    return cast_tensor(Tensor.from_view(st.tensor_view(String("pooled_cond")), ctx), STDtype.BF16, ctx)

def _load_uncond_pooled(ctx: DeviceContext) raises -> Tensor:
    var st = ShardedSafeTensors.open(String(EMBEDDINGS_PATH))
    return cast_tensor(Tensor.from_view(st.tensor_view(String("pooled_uncond")), ctx), STDtype.BF16, ctx)


# ── SD3 MMDiT forward (one pass) ──────────────────────────────────────────────
# Streams 24 blocks via BlockLoader one at a time.
def _sd3_forward(
    latent: Tensor,          # [1, 16, LH, LW] BF16
    sigma: Float32,          # current sigma
    context: Tensor,         # [1, N_CTX, 4096] BF16
    pooled: Tensor,          # [1, 2048] BF16
    gate: SD3MMDiTPreBlockGate,
    loader: BlockLoader,
    ctx: DeviceContext,
) raises -> Tensor:
    # Pre-block: patch embed + pos embed
    var x_tokens = gate.latent_patch_embed[LH, LW](latent, ctx)  # [1, N_IMG, HIDDEN]

    # Conditioning vector: timestep + pooled
    var c = gate.conditioning(sigma, pooled, ctx)  # [1, HIDDEN]

    # Context embedding
    var ctx_tokens = gate.context_embed[N_CTX](context, ctx)  # [1, N_CTX, HIDDEN]

    # 24 joint blocks, streamed
    for i in range(DEPTH):
        var is_last = (i == DEPTH - 1)
        var block_prefix = String("model.diffusion_model.joint_blocks.") + String(i)
        loader.prefetch_block(block_prefix)
        var blk = loader.load_block(block_prefix, ctx)

        # Inout block forward: updates ctx_tokens and x_tokens in place
        _sd3_joint_block[1, S_JOINT, N_CTX, N_IMG, H_HEADS, H_DIM](
            ctx_tokens, x_tokens, c, blk, i, is_last, DUAL_BLOCKS, HIDDEN, ctx
        )

        unload_block(blk^)

    # Final layer: no-affine LN + adaLN + linear -> [1, N_IMG, 64]
    var patch_out = gate.final_layer_tokens(x_tokens, c, ctx)

    # Unpatchify [1, N_IMG, 64] -> [1, 16, LH, LW]
    return gate.final_unpatchify[LH, LW](patch_out, ctx)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() raises:
    var ctx = DeviceContext()
    print("============================================================")
    print("SD3.5 Medium 1024x1024 — 28-step shifted flow CFG 4.5 seed 42")
    print("  Model:", MODEL_PATH)
    print("  Embeddings:", EMBEDDINGS_PATH)
    print("============================================================")

    # Stage 1: Load text conditioning sidecar
    print("\n--- Stage 1: Text Conditioning (sidecar) ---")
    var context_cond = _load_cond_context(ctx)
    var context_uncond = _load_uncond_context(ctx)
    var pooled_cond = _load_cond_pooled(ctx)
    var pooled_uncond = _load_uncond_pooled(ctx)
    var ccs = context_cond.shape()
    print("  context_cond:", ccs[0], ccs[1], ccs[2], "dtype:", context_cond.dtype().name())
    _stats("context_cond", context_cond, ctx)
    _stats("pooled_cond", pooled_cond, ctx)

    # Stage 2: Load SD3 Medium resident weights
    print("\n--- Stage 2: Load SD3.5 Medium resident weights ---")
    var gate = SD3MMDiTPreBlockGate.load_medium_default(ctx)
    print("  hidden=", gate.hidden, " depth=", DEPTH,
          " heads=", H_HEADS, " dual_blocks=", DUAL_BLOCKS)

    # Stage 3: Open block loader
    print("\n--- Stage 3: Open block loader ---")
    var loader = BlockLoader.open(String(MODEL_PATH))
    print("  Block loader ready")

    # Stage 4: Init noise latent
    print("\n--- Stage 4: Init noise latent [1,", LC, ",", LH, ",", LW, "] ---")
    var noise_sh = List[Int]()
    noise_sh.append(1)
    noise_sh.append(LC)
    noise_sh.append(LH)
    noise_sh.append(LW)
    var latent = randn(noise_sh^, SEED, STDtype.BF16, ctx)
    _stats("initial_noise", latent, ctx)

    # Stage 5: Denoise loop
    print("\n--- Stage 5: Denoise (", NUM_STEPS, "steps, CFG=", CFG_SCALE, ") ---")
    var sched = SD3FlowMatchScheduler.medium_default()

    for step in range(NUM_STEPS):
        var sigma = sched.timestep(step)
        var dt = sched.dt(step)  # dt = sigma[step+1] - sigma[step] (negative)

        print("  step", step + 1, "/", NUM_STEPS, "sigma=", sigma)

        # Conditional forward
        var v_cond = _sd3_forward(
            latent, sigma, context_cond, pooled_cond, gate, loader, ctx
        )

        # Unconditional forward
        var v_uncond = _sd3_forward(
            latent, sigma, context_uncond, pooled_uncond, gate, loader, ctx
        )

        # CFG/Euler tensor ops use F32 arithmetic internally and store BF16.
        var velocity = sd3_cfg(v_cond, v_uncond, CFG_SCALE, ctx)

        # Euler step
        latent = sd3_euler_step(latent, velocity, dt, ctx)

        if step == 0 or step == NUM_STEPS - 1:
            _stats("latent_after_step", latent, ctx)

    _stats("final_latent", latent, ctx)

    # Stage 6: VAE decode
    print("\n--- Stage 6: VAE Decode ---")
    var vae = load_sd3_embedded_ldm_decoder[LH, LW](String(MODEL_PATH), ctx)
    var image = vae.decode(latent, ctx)
    var imsh = image.shape()
    print("  decoded:", imsh[2], "x", imsh[3])
    _stats("decoded_image", image, ctx)

    # Stage 7: Save PNG
    print("\n--- Stage 7: Save PNG ---")
    save_png(image, String(OUT_PNG), ctx, ValueRange.SIGNED)
    print("IMAGE SAVED:", OUT_PNG)

    print("\n============================================================")
    print("SD3.5 Medium pipeline COMPLETE")
    print("============================================================")
