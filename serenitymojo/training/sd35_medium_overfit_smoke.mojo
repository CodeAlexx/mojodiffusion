# serenitymojo/training/sd35_medium_overfit_smoke.mojo
#
# MEDIUM (sd3.5-medium) 1024^2 INFRA + OVERFIT smoke. Loads the REAL medium
# checkpoint via the block-swap offload loader, builds the LoRA set, and runs a
# few fwd->MSE->bwd->AdamW steps on ONE fixed synthetic batch at the REQUIRED
# 1024^2 / N_TXT=154 shape (N_IMG=4096, S=4250). Purpose THIS run: MEASURE that
# medium loads + FITS in 24GB + a full-depth 1024^2 step runs + the loss
# decreases (LoRA descends). Reports peak behaviour by exit code + loss trace.
#
# NOT-YET-FAITHFUL (measured gaps, see HANDOFF): the current sd35_stack forward
#   (1) omits pos_embed (x_embedder only), and
#   (2) uses the non-dual joint block for ALL 24 blocks (blocks 0-12 are dual).
# So this smoke de-risks INFRA/MEMORY only; faithful training needs pos_embed +
# the dual-block dispatch wired in (dual block fwd/bwd already gate-verified).
#
# Run:
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#       serenitymojo/training/sd35_medium_overfit_smoke.mojo -o /tmp/sd35_medium_overfit
#   /tmp/sd35_medium_overfit

from std.gpu.host import DeviceContext
from std.collections import List
from std.math import sqrt, sin
from std.time import perf_counter_ns
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.models.sd35.weights import load_sd35_stack_base
from serenitymojo.models.sd35.sd35_stack_lora import (
    SD35StackBase, SD35LoraSet, SD35LoraGradSet,
    build_sd35_lora_set, sd35_lora_adamw_step, total_adapters,
    sd35_stack_lora_forward_offload, sd35_stack_lora_backward_offload,
)
from serenitymojo.offload.plan import build_sd35_block_plan, OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader

# ── sd3.5-MEDIUM arch ──
comptime H = 24
comptime Dh = 64
comptime D = H * Dh            # 1536
comptime FMLP = 6144           # mlp_hidden = D*4
comptime IN_CH = 64
comptime TXT_CH = 4096
comptime OUT_CH = 64
comptime NUM_JOINT = 24
comptime NUM_DUAL = 13          # sd3.5-medium: blocks 0-12 are dual-attention
comptime TIMESTEP_DIM = 256
comptime POOLED_DIM = 2048
comptime EPS = Float32(1e-6)
comptime QK_EPS = Float32(1e-6)

# ── 1024^2 tokens (REQUIRED shape) ──
comptime LAT_H = 128
comptime LAT_W = 128
comptime PATCH = 2
comptime N_IMG = (LAT_H // PATCH) * (LAT_W // PATCH)   # 4096
comptime N_TXT = 154
comptime S = N_TXT + N_IMG                             # 4250

comptime LAT_C = 16
comptime HT = LAT_H // PATCH    # 64
comptime WT = LAT_W // PATCH    # 64
comptime VAE_SHIFT = Float32(0.0609)
comptime VAE_SCALE = Float32(1.5305)

comptime RANK = 16
comptime ALPHA = Float32(16.0)
comptime STEPS = 6
comptime LR = Float32(1.0e-3)
comptime CKPT = "/home/alex/.serenity/models/checkpoints/sd3.5_medium.safetensors"
comptime CACHE = "/home/alex/datasets/andrsd35_sd35_cache/10.safetensors"


def _fill(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var a = Float32(0.0)
    for i in range(n):
        # deterministic sinusoid, varies per index + seed
        a = sin(Float32(0.013) * Float32(i) + Float32(0.0007) * Float32(seed) + Float32(0.5))
        out.append(a * scale)
    return out^


def _cache_f32(st: SafeTensors, name: String, ctx: DeviceContext) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    if t.dtype() == STDtype.BF16:
        var bf = t.to_host_bf16(ctx)
        var out = List[Float32]()
        for i in range(len(bf)):
            out.append(bf[i].cast[DType.float32]())
        return out^
    return t.to_host(ctx)


# channel-major patchify [16,128,128] -> [N_IMG=4096, 64] (matches train_sd35_real)
def _pack_latents(lat: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for ih in range(HT):
        for iw in range(WT):
            for c in range(LAT_C):
                for ph in range(PATCH):
                    for pw in range(PATCH):
                        var hh = ih * PATCH + ph
                        var ww = iw * PATCH + pw
                        out.append(lat[c * LAT_H * LAT_W + hh * LAT_W + ww])
    return out^


# deterministic N(0,1)-ish noise (box-muller-free; bounded). Fixed per index.
def _noise(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var st = seed
    for _ in range(n):
        st = st * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(st >> 40)) * Float32(1.0 / 16777216.0)  # [0,1)
        out.append((u - Float32(0.5)) * Float32(2.0))               # [-1,1)
    return out^


def _clip(mut grads: SD35LoraGradSet, max_norm: Float32) -> Float32:
    var ss = Float32(0.0)
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            ss += grads.d_a[i][j] * grads.d_a[i][j]
        for j in range(len(grads.d_b[i])):
            ss += grads.d_b[i][j] * grads.d_b[i][j]
    var gn = sqrt(ss)
    if gn > max_norm and gn > Float32(0.0):
        var s = max_norm / gn
        for i in range(len(grads.d_a)):
            for j in range(len(grads.d_a[i])):
                grads.d_a[i][j] = grads.d_a[i][j] * s
            for j in range(len(grads.d_b[i])):
                grads.d_b[i][j] = grads.d_b[i][j] * s
    return gn


def main() raises:
    var ctx = DeviceContext()
    print("==== sd3.5-MEDIUM 1024^2 infra+overfit smoke (offload, dual 0-12) ====")
    print("D=", D, " H=", H, " Dh=", Dh, " Fmlp=", FMLP, " depth=", NUM_JOINT)
    print("tokens: N_IMG=", N_IMG, " N_TXT=", N_TXT, " S=", S, " (1024^2)")
    print("ckpt:", CKPT)

    print("[load] SD35StackBase ...")
    var base_st = SafeTensors.open(CKPT)
    var base = load_sd35_stack_base(base_st, ctx)
    print("[load] base resident")

    var plan = build_sd35_block_plan(NUM_JOINT)
    var cfg = OffloadConfig.synchronous_single()
    var loader = TurboPlannedLoader.open(CKPT, plan^, cfg, ctx)
    print("[load] offload loader opened (", loader.block_count(), "blocks)")

    var lora = build_sd35_lora_set(NUM_JOINT, D, FMLP, RANK, ALPHA)
    print("[lora] adapters:", total_adapters(lora))

    # ── REAL cached batch (latent + text + pooled), flow-match in packed space ──
    print("[cache]", CACHE)
    var cst = SafeTensors.open(CACHE)
    var lat_raw = _cache_f32(cst, String("latent"), ctx)        # [16*128*128]
    var lat_scaled = List[Float32]()
    for i in range(len(lat_raw)):
        lat_scaled.append((lat_raw[i] - VAE_SHIFT) * VAE_SCALE)
    var latent_packed = _pack_latents(lat_scaled)              # [N_IMG, IN_CH]
    var txt = _cache_f32(cst, String("text_embedding"), ctx)   # [154*4096]
    var pooled = _cache_f32(cst, String("pooled"), ctx)        # [2048]

    var sigma = Float32(0.5)
    var noise = _noise(N_IMG * IN_CH, UInt64(20260609))
    var noisy = List[Float32]()
    var target = List[Float32]()
    for i in range(len(latent_packed)):
        noisy.append(noise[i] * sigma + latent_packed[i] * (Float32(1.0) - sigma))
        target.append(noise[i] - latent_packed[i])
    print("[cache] latent", len(lat_raw), " packed", len(latent_packed),
          " txt", len(txt), " pooled", len(pooled))

    var first_loss = Float32(0.0)
    var min_loss = Float32(1.0e30)
    print("")
    print("step   MSE        grad_norm   sec")

    for step in range(STEPS):
        var t0 = perf_counter_ns()
        var fwd = sd35_stack_lora_forward_offload[H, Dh, N_IMG, N_TXT, S](
            noisy.copy(), txt.copy(), pooled.copy(), sigma,
            base, loader, lora,
            D, FMLP, IN_CH, TXT_CH, OUT_CH, TIMESTEP_DIM, POOLED_DIM,
            EPS, QK_EPS, ctx, NUM_DUAL, True,
        )
        var nout = len(fwd.out)
        var loss = Float32(0.0)
        var d_loss = List[Float32]()
        var inv_n = Float32(2.0) / Float32(nout)
        for i in range(nout):
            var diff = fwd.out[i] - target[i]
            loss += diff * diff
            d_loss.append(inv_n * diff)
        loss = loss / Float32(nout)
        if step == 0:
            first_loss = loss
        if loss < min_loss:
            min_loss = loss

        var grads = sd35_stack_lora_backward_offload[H, Dh, N_IMG, N_TXT, S](
            d_loss, noisy.copy(), txt.copy(),
            base, loader, lora, fwd,
            D, FMLP, IN_CH, TXT_CH, OUT_CH, TIMESTEP_DIM, POOLED_DIM,
            EPS, QK_EPS, ctx, NUM_DUAL, True,
        )
        var gn = _clip(grads, Float32(1.0))
        sd35_lora_adamw_step(lora, grads, step + 1, LR, ctx)
        var secs = Float64(perf_counter_ns() - t0) / 1.0e9
        print(step, "  ", loss, "  ", gn, "  ", secs)
        if grads.nonfinite != 0:
            print("  !! nonfinite grads =", grads.nonfinite)

    print("")
    print("first MSE =", first_loss, "  min MSE =", min_loss,
          "  reduction =", first_loss / (min_loss + Float32(1.0e-30)), "x")
    if min_loss < first_loss:
        print("RESULT: medium 1024^2 (real cache, dual 0-12 + standard + ctxpre 23)",
              "ran full-depth + FIT in memory + loss DECREASED. NOTE: pos_embed still",
              "omitted (forward not yet fully faithful); host-carrier path ~123s/step.")
    else:
        print("RESULT: ran but loss did not decrease — investigate.")
