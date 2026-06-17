# anima_pipeline_1024_multistep.mojo — Anima 1024x1024 30-step Euler CFG denoise.
#
# Faithful port of anima_infer.rs:
#   1. Load cached context_cond/context_uncond (sidecar, skip Qwen3/T5 adapter).
#   2. Init latent from seed 42 (Box-Muller, Rust rand StdRng layout).
#   3. 30-step rectified-flow Euler: sigma 1.0->0.0 linear, CFG 4.5,
#      NO timestep*1000 scaling, dt = sigma_next - sigma (negative).
#      x = x + dt * pred   (direct-velocity Euler)
#   4. Parity gate: compare Mojo latent vs Rust oracle (cosine + max-abs-diff).
#   5. Decode via tiled Wan/Qwen image VAE and save PNG.
#
# Build:
#   pixi run mojo build -I . -Xlinker -lm \
#     serenitymojo/pipeline/anima_pipeline_1024_multistep.mojo \
#     -o /tmp/anima_1024 && /tmp/anima_1024

from std.gpu.host import DeviceContext
from std.sys import argv
from std.math import sqrt as fsqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.dit.anima_contract import (
    ANIMA_DIT_PATH,
    ANIMA_VAE_PATH,
    ANIMA_LATENT_H,
    ANIMA_LATENT_W,
    ANIMA_LATENT_CHANNELS,
    ANIMA_LATENT_T,
    ANIMA_NUM_STEPS,
    ANIMA_IMAGE_TOKENS,
    ANIMA_MAX_SEQ_LEN,
    ANIMA_ADAPTER_DIM,
    anima_default_conditioning_path,
    anima_default_rust_latent_path,
)
from serenitymojo.models.dit.anima_dit import AnimaDiT
from serenitymojo.models.vae.qwenimage_tiled_decode import wan21_image_tiled_decode
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import add, mul_scalar, sub, reshape, permute
from serenitymojo.sampling.anima_sampling import AnimaLinearFlowScheduler, anima_cfg
from serenitymojo.image.png import ValueRange, save_png


comptime OUT_PNG = "/home/alex/mojodiffusion/output/anima_1024_30step.png"
comptime RUST_LATENT_PATH = "/home/alex/EriDiffusion/inference-flame/output/anima_rust_latent.safetensors"
comptime EMBEDDINGS_PATH = "/home/alex/EriDiffusion/inference-flame/output/anima_embeddings.safetensors"

comptime CFG_SCALE = Float32(4.5)
comptime SEED = UInt64(42)
comptime NUM_STEPS = 30
comptime LH = ANIMA_LATENT_H  # 128
comptime LW = ANIMA_LATENT_W  # 128
comptime LATENT_C = ANIMA_LATENT_CHANNELS  # 16
comptime LATENT_T_FRAMES = ANIMA_LATENT_T  # 1


def _abs_f32(v: Float32) -> Float32:
    if v < 0.0:
        return -v
    return v


def _mean_abs(t: Tensor, ctx: DeviceContext) raises -> Float64:
    var h = t.to_host(ctx)
    var n = len(h)
    if n == 0:
        return 0.0
    var total: Float64 = 0.0
    for i in range(n):
        var v = h[i]
        if v < 0.0:
            total += Float64(-v)
        else:
            total += Float64(v)
    return total / Float64(n)


def _cosine_similarity(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Float64:
    """Cosine similarity between two tensors (flattened)."""
    var ah = a.to_host(ctx)
    var bh = b.to_host(ctx)
    var n = len(ah)
    if n == 0:
        raise Error("cosine_similarity: empty tensors")
    if len(bh) != n:
        raise Error("cosine_similarity: shape mismatch")
    var dot: Float64 = 0.0
    var norm_a: Float64 = 0.0
    var norm_b: Float64 = 0.0
    for i in range(n):
        var av = Float64(ah[i])
        var bv = Float64(bh[i])
        dot += av * bv
        norm_a += av * av
        norm_b += bv * bv
    var denom = Float64(fsqrt(Float32(norm_a))) * Float64(fsqrt(Float32(norm_b)))
    if denom < 1e-10:
        return 0.0
    return dot / denom


def _max_abs_diff(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Float32:
    var ah = a.to_host(ctx)
    var bh = b.to_host(ctx)
    var n = len(ah)
    if len(bh) != n:
        raise Error("max_abs_diff: shape mismatch")
    var mx: Float32 = 0.0
    for i in range(n):
        var d = _abs_f32(ah[i] - bh[i])
        if d > mx:
            mx = d
    return mx


struct _ContextPair(Movable):
    var cond: Tensor
    var uncond: Tensor

    def __init__(out self, var cond: Tensor, var uncond: Tensor):
        self.cond = cond^
        self.uncond = uncond^

    def __del__(deinit self):
        pass


def _load_context(
    embeddings_path: String, ctx: DeviceContext
) raises -> _ContextPair:
    """Load context_cond and context_uncond from cached safetensors sidecar."""
    var st = ShardedSafeTensors.open(embeddings_path)
    var cond = Tensor.from_view(st.tensor_view(String("context_cond")), ctx)
    var uncond = Tensor.from_view(st.tensor_view(String("context_uncond")), ctx)
    print("  context_cond  shape:", cond.shape()[0], cond.shape()[1], cond.shape()[2],
          "dtype:", cond.dtype().name())
    print("  context_uncond shape:", uncond.shape()[0], uncond.shape()[1], uncond.shape()[2])
    return _ContextPair(cond^, uncond^)


def _init_latent(ctx: DeviceContext) raises -> Tensor:
    """Init BF16 latent [1, T, H, W, 16] with Box-Muller noise."""
    var numel = LATENT_T_FRAMES * LH * LW * LATENT_C  # 1*128*128*16 = 262144
    # randn generates [numel] in layout [T*H*W*C] = [numel].
    var flat_sh = List[Int]()
    flat_sh.append(numel)
    var noise_flat = randn(flat_sh^, SEED, STDtype.BF16, ctx)
    # Reshape to [1, T, H, W, 16] (Anima layout)
    var sh5 = List[Int]()
    sh5.append(1)
    sh5.append(LATENT_T_FRAMES)
    sh5.append(LH)
    sh5.append(LW)
    sh5.append(LATENT_C)
    return reshape(noise_flat, sh5^, ctx)


def _load_rust_latent(ctx: DeviceContext) raises -> Tensor:
    """Load Rust oracle latent [1, 16, 1, 128, 128] F32 and permute to [1, 1, 128, 128, 16]."""
    var st = ShardedSafeTensors.open(String(RUST_LATENT_PATH))
    var latent5 = Tensor.from_view(st.tensor_view(String("latent")), ctx)
    # latent5 is [1, 16, 1, 128, 128] — permute to [1, 1, 128, 128, 16] for comparison
    var perm = List[Int]()
    perm.append(0)  # B
    perm.append(2)  # T (was index 2 = 1)
    perm.append(3)  # H (was index 3 = 128)
    perm.append(4)  # W (was index 4 = 128)
    perm.append(1)  # C (was index 1 = 16)
    var latent_bthwc = permute(latent5, perm^, ctx)
    print("  oracle latent shape (BTHWC):", latent_bthwc.shape()[0],
          latent_bthwc.shape()[1], latent_bthwc.shape()[2],
          latent_bthwc.shape()[3], latent_bthwc.shape()[4])
    print("  oracle latent mean_abs:", _mean_abs(latent_bthwc, ctx))
    return latent_bthwc^


def _latent_to_vae_input(
    latent_bthwc: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """Convert [1, T=1, H, W, 16] -> [1, 16, H, W] for VAE (drop T dim).
    Matches anima_infer.rs: latent.permute([0, 4, 1, 2, 3]) -> [B,C,T,H,W],
    then the VAE decoder takes [B, 16, H, W] (squeeze T).
    The existing anima_vae_latent_smoke.mojo loads the oracle in [1,16,128,128]
    after squeezing T. So we permute [B,T,H,W,C] -> [B,C,T,H,W] then flatten T."""
    # Permute [1, 1, 128, 128, 16] -> [1, 16, 1, 128, 128]
    var perm = List[Int]()
    perm.append(0)  # B
    perm.append(4)  # C
    perm.append(1)  # T
    perm.append(2)  # H
    perm.append(3)  # W
    var latent_bctbw = permute(latent_bthwc, perm^, ctx)  # [1, 16, 1, 128, 128]
    # Flatten to [1, 16, 128, 128] (squeeze T=1)
    var sh4 = List[Int]()
    sh4.append(1)
    sh4.append(LATENT_C)
    sh4.append(LH)
    sh4.append(LW)
    return reshape(latent_bctbw, sh4^, ctx)


def main() raises:
    var ctx = DeviceContext()
    print("============================================================")
    print("Anima 1024x1024 — 30-step Euler CFG 4.5, seed 42")
    print("============================================================")

    # ── Stage 1: Load cached context (argv[1] overrides the path; argv[2]=out png) ─
    print("\n--- Stage 1: Load cached context ---")
    var _a = argv()
    var ctx_path = String(EMBEDDINGS_PATH)
    if len(_a) > 1:
        ctx_path = String(_a[1])
    print("  context path:", ctx_path)
    var ctx_pair = _load_context(ctx_path, ctx)
    # Cast to BF16 if loaded as F32 (the sidecar may be BF16 or F32)
    var context_cond: Tensor
    var context_uncond: Tensor
    if ctx_pair.cond.dtype() == STDtype.F32:
        context_cond = cast_tensor(ctx_pair.cond, STDtype.BF16, ctx)
        context_uncond = cast_tensor(ctx_pair.uncond, STDtype.BF16, ctx)
    else:
        # Already BF16; we need to copy since we can't move struct fields
        # Use cast as a no-op copy path via BF16->BF16
        context_cond = cast_tensor(ctx_pair.cond, STDtype.BF16, ctx)
        context_uncond = cast_tensor(ctx_pair.uncond, STDtype.BF16, ctx)
    print("  context mean_abs:", _mean_abs(context_cond, ctx))

    # ── Stage 2: Load Anima DiT ───────────────────────────────────────────────
    print("\n--- Stage 2: Load Anima DiT (all-on-GPU) ---")
    var model = AnimaDiT.load(String(ANIMA_DIT_PATH), ctx)

    # ── Stage 3: Init noise latent ────────────────────────────────────────────
    print("\n--- Stage 3: Init latent [1,", LATENT_T_FRAMES, ",", LH, ",", LW, ",", LATENT_C, "] ---")
    var x = _init_latent(ctx)
    print("  initial noise mean_abs:", _mean_abs(x, ctx))

    # ── Stage 4: Denoise loop ─────────────────────────────────────────────────
    print("\n--- Stage 4: Denoise (", NUM_STEPS, "steps, CFG=", CFG_SCALE, ") ---")
    var sched = AnimaLinearFlowScheduler.default_30()

    for step_i in range(NUM_STEPS):
        var sigma = sched.sigma(step_i)
        var sigma_next = sched.sigma(step_i + 1)
        var dt = sigma_next - sigma   # negative (sigma decreasing)
        print("  step", step_i + 1, "/", NUM_STEPS, "sigma=", sigma, "->", sigma_next)

        # Timestep: raw sigma value, NOT *1000 (Anima uses sigma directly)
        var tvals = List[Float32]()
        tvals.append(sigma)
        var tsh = List[Int]()
        tsh.append(1)
        var timestep = Tensor.from_host(tvals, tsh^, STDtype.F32, ctx)

        # Conditional prediction
        var pred_cond = model.forward_with_context(x, timestep, context_cond, ctx)

        # Unconditional prediction (need fresh timestep since from_host consumes)
        var tvals2 = List[Float32]()
        tvals2.append(sigma)
        var tsh2 = List[Int]()
        tsh2.append(1)
        var timestep2 = Tensor.from_host(tvals2, tsh2^, STDtype.F32, ctx)
        var pred_uncond = model.forward_with_context(x, timestep2, context_uncond, ctx)

        # CFG/Euler tensor ops use F32 arithmetic internally and store BF16.
        var diff = sub(pred_cond, pred_uncond, ctx)
        var scaled_diff = mul_scalar(diff, CFG_SCALE, ctx)
        var pred = add(pred_uncond, scaled_diff, ctx)

        # Euler step: x = x + dt * pred; storage remains BF16.
        var delta = mul_scalar(pred, dt, ctx)
        x = add(x, delta, ctx)

        if step_i == 0 or step_i == NUM_STEPS - 1:
            print("    pred mean_abs:", _mean_abs(pred, ctx))

    print("\n  Final latent mean_abs:", _mean_abs(x, ctx))

    # ── Stage 5: Parity gate vs Rust oracle ───────────────────────────────────
    print("\n--- Stage 5: Parity gate ---")
    var oracle = _load_rust_latent(ctx)
    var cos_sim = _cosine_similarity(x, oracle, ctx)
    var max_diff = _max_abs_diff(x, oracle, ctx)
    print("  Mojo vs Rust cosine similarity:", cos_sim)
    print("  Mojo vs Rust max-abs-diff:     ", max_diff)
    if cos_sim >= 0.999:
        print("  PARITY PASS (cos >= 0.999)")
    else:
        print("  PARITY WARN: cos =", cos_sim, "< 0.999 — diverged from Rust oracle")

    # ── Stage 6: VAE decode and save PNG ─────────────────────────────────────
    print("\n--- Stage 6: tiled VAE decode ---")
    var latent_bf16 = cast_tensor(x, STDtype.BF16, ctx)
    # Convert [1, T, H, W, 16] -> [1, 16, H, W] for VAE input
    var vae_input = _latent_to_vae_input(latent_bf16, ctx)
    print("  VAE input shape:", vae_input.shape()[0], vae_input.shape()[1],
          vae_input.shape()[2], vae_input.shape()[3])

    var image = wan21_image_tiled_decode[LH, LW](
        vae_input, String(ANIMA_VAE_PATH), ctx
    )
    print("  decoded:", image.shape()[2], "x", image.shape()[3])
    var _out_p = String(OUT_PNG)
    if len(_a) > 2:
        _out_p = String(_a[2])
    save_png(image, _out_p, ctx, ValueRange.SIGNED)
    print("FINAL IMAGE:", _out_p)
    print("IMAGE SAVED:", _out_p)

    print("\n============================================================")
    print("Anima pipeline 1024 complete")
    print("============================================================")
