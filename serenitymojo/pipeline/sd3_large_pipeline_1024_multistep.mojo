# sd3_large_pipeline_1024_multistep.mojo — SD3.5 Large 1024x1024 28-step pipeline.
#
# Faithful port of sd3_infer.rs:
#   1. Load cached conditioning sidecar (produced by Rust encode-only run):
#      context_cond/uncond [1, 410, 4096] F32→BF16, pooled_cond/uncond [1, 2048] F32→BF16.
#   2. Init noise latent [1, 16, 128, 128] F32 (Box-Muller, seed=42).
#   3. 28-step shifted rectified-flow CFG Euler loop:
#      - sigma schedule: shift=3.0, descending (same as Medium)
#      - model timestep: sigma * 1000
#      - CFG: uncond + 4.5 * (cond - uncond)
#      - Euler: latent += velocity * dt (dt negative)
#   4. VAE decode via embedded SD3 LDM decoder (first_stage_model.decoder.*).
#   5. Save PNG output/sd3_large_1024_28step.png.
#
# Text conditioning: sidecar route.
#   Embeddings: /home/alex/EriDiffusion/inference-flame/output/sd3_large_embeddings.safetensors
#   Keys: context_cond [1,410,4096], context_uncond [1,410,4096],
#         pooled_cond [1,2048], pooled_uncond [1,2048]  (all F32 in file, cast to BF16)
#
# Architecture: SD3.5 Large = depth 38, hidden 2432, heads 38, head_dim 64
#   ALL 38 blocks: single attention only (6*H adaLN) — NO dual attention
#   block 37 context: pre_only (2*H adaLN)
#
# VRAM strategy: BlockLoader streams one block at a time (~426 MB each).
#   Peak = resident (~430 MB) + one_block (426 MB) + SDPA (H=38,S=4506 → 3.1 GB)
#          + working buffers ≈ 4.5 GB well within 24 GB.
#
# Build:
#   pixi run mojo build -I . -Xlinker -lm \
#     serenitymojo/pipeline/sd3_large_pipeline_1024_multistep.mojo \
#     -o /tmp/sd3_large && /tmp/sd3_large

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


# ── Configuration ──────────────────────────────────────────────────────────────
comptime EMBEDDINGS_PATH = "/home/alex/EriDiffusion/inference-flame/output/sd3_large_embeddings.safetensors"
comptime MODEL_PATH = "/home/alex/.serenity/models/checkpoints/sd3.5_large.safetensors"
comptime OUT_PNG = "/home/alex/mojodiffusion/output/sd3_large_1024_28step.png"

comptime NUM_STEPS = 28
comptime CFG_SCALE = Float32(4.5)
comptime SHIFT = Float32(3.0)
comptime SEED = UInt64(42)

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


# ── SD3 Large MMDiT forward (one pass) ────────────────────────────────────────
# Streams 38 blocks via BlockLoader one at a time (~426 MB each).
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

        # Inout block forward: updates ctx_tokens and x_tokens in place.
        # DUAL_BLOCKS=0 so block_has_dual is always False (Large has no dual-attn).
        _sd3_joint_block[1, S_JOINT, N_CTX, N_IMG, H_HEADS, H_DIM](
            ctx_tokens, x_tokens, c, blk, i, is_last, DUAL_BLOCKS, HIDDEN, ctx
        )

        unload_block(blk^)

    # Final layer: no-affine LN + adaLN + linear -> [1, N_IMG, 64]
    var patch_out = gate.final_layer_tokens(x_tokens, c, ctx)

    # SD3 spatial-outer unpatchify [1, N_IMG, 64] -> [1, 16, LH, LW]
    return gate.final_unpatchify[LH, LW](patch_out, ctx)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() raises:
    var ctx = DeviceContext()
    print("============================================================")
    print("SD3.5 Large 1024x1024 — 28-step shifted flow CFG 4.5 seed 42")
    print("  Model:", MODEL_PATH)
    print("  Embeddings:", EMBEDDINGS_PATH)
    print("  Block streaming: 38 blocks x ~426 MB = 16.2 GB streamed")
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

    # Stage 2: Load SD3.5 Large resident weights
    print("\n--- Stage 2: Load SD3.5 Large resident weights ---")
    var gate = SD3MMDiTPreBlockGate.load_large_default(ctx)
    print("  hidden=", gate.hidden, " depth=", DEPTH,
          " heads=", H_HEADS, " dual_blocks=", DUAL_BLOCKS)

    # Stage 3: Open block loader (single-file .safetensors, BlockLoader handles it)
    print("\n--- Stage 3: Open block loader ---")
    var loader = BlockLoader.open(String(MODEL_PATH))
    print("  Block loader ready for 38 x ~426 MB blocks")

    # Stage 4: Init noise latent [1, 16, 128, 128] F32
    print("\n--- Stage 4: Init noise latent [1,", LC, ",", LH, ",", LW, "] ---")
    var noise_sh = List[Int]()
    noise_sh.append(1)
    noise_sh.append(LC)
    noise_sh.append(LH)
    noise_sh.append(LW)
    var latent_f32 = randn(noise_sh^, SEED, STDtype.F32, ctx)
    _stats("initial_noise", latent_f32, ctx)

    # Stage 5: Denoise loop
    print("\n--- Stage 5: Denoise (", NUM_STEPS, "steps, CFG=", CFG_SCALE, ") ---")
    var sched = SD3FlowMatchScheduler.large_default()
    var latent = latent_f32^   # F32 throughout loop

    for step in range(NUM_STEPS):
        var sigma = sched.timestep(step)
        var dt = sched.dt(step)  # dt = sigma[step+1] - sigma[step] (negative)

        print("  step", step + 1, "/", NUM_STEPS, "sigma=", sigma)

        # Cast F32 latent -> BF16 for DiT
        var latent_bf16 = cast_tensor(latent, STDtype.BF16, ctx)

        # Conditional forward (38 blocks streamed)
        var v_cond = _sd3_large_forward(
            latent_bf16, sigma, context_cond, pooled_cond, gate, loader, ctx
        )

        # Unconditional forward (38 blocks streamed again)
        var v_uncond = _sd3_large_forward(
            latent_bf16, sigma, context_uncond, pooled_uncond, gate, loader, ctx
        )

        # CFG (in F32)
        var vc_f32 = cast_tensor(v_cond, STDtype.F32, ctx)
        var vu_f32 = cast_tensor(v_uncond, STDtype.F32, ctx)
        var velocity = sd3_cfg(vc_f32, vu_f32, CFG_SCALE, ctx)

        # Euler step
        latent = sd3_euler_step(latent, velocity, dt, ctx)

        if step == 0 or step == NUM_STEPS - 1:
            _stats("latent_after_step", latent, ctx)

    _stats("final_latent", latent, ctx)

    # Stage 6: VAE decode (uses embedded first_stage_model.decoder.* in Large checkpoint)
    print("\n--- Stage 6: VAE Decode ---")
    var latent_bf16_final = cast_tensor(latent, STDtype.BF16, ctx)
    var vae = load_sd3_embedded_ldm_decoder[LH, LW](String(MODEL_PATH), ctx)
    var image = vae.decode(latent_bf16_final, ctx)
    var imsh = image.shape()
    print("  decoded:", imsh[2], "x", imsh[3])
    _stats("decoded_image", image, ctx)

    # Stage 7: Save PNG
    print("\n--- Stage 7: Save PNG ---")
    save_png(image, String(OUT_PNG), ctx, ValueRange.SIGNED)
    print("IMAGE SAVED:", OUT_PNG)

    print("\n============================================================")
    print("SD3.5 Large pipeline COMPLETE")
    print("============================================================")
