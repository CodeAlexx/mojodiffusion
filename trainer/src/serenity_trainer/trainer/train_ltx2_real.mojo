# train_ltx2_real.mojo — LEGACY LTX-2 video-only LoRA training loop.
#
# This file is intentionally fail-closed by default. It trains the old
# `models/ltx2/ltx2_stack_lora.mojo` video-only stack, not the production
# full-AV inference spine in `models/dit/ltx2_dit.mojo`. Use it only as a legacy
# backward smoke with `--legacy-video-only`; production LTX-2 training must land
# on the full AV block forward/backward. See
# `training/ltx2_av_training_readiness.mojo` for the executable readiness gate.
#
# TRANSLATION of the musubi-tuner LTX-2 LoRA recipe onto the Mojo LTX-2 LoRA
# OFFLOAD stack (models/ltx2/ltx2_stack_lora.mojo). Streams all 48 identical
# transformer_blocks block-by-block via TurboPlannedLoader; the frozen
# patchify_proj/proj_out + adaln_single conditioning network are resident. No
# synthetic block math — every block fwd/bwd is the proven ltx2_block.mojo arm.
# Mirrors train_chroma_real.mojo's loop (timing, grad clip, shared progress).
#
# LTX-2 vs CHROMA (the deltas; see ltx2_stack_lora.mojo header):
#   - NO double/single split: 48 identical blocks (BlockKind.transformer()).
#   - Modulation: per-block scale_shift_table[9,D] (frozen) + a single global
#     adaln_single(sigma) -> 6*D delta (frozen) ADDED on top. Built once/step.
#   - Legacy narrowed LoRA target set:
#     attn1.to_q/to_k/to_v/to_out.0 -> 4 adapters x 48 = 192 total.
#     musubi's production T2V preset targets all AV attention modules; see
#     ltx2_av_training_readiness.mojo.
#   - RoPE: SPLIT type (rope_halfsplit), tables [S*H, Dh//2] split layout.
#
# Per step (flow-match; musubi ltx2_scheduler.py + ltx2_train.py):
#   1. load cached latent tokens [N, in_ch] (already patchify-ready)
#   2. sigma via logit-normal (shift=1.0 -> identity sigmoid(N(0,1)) clamp);
#      ltx2_scheduler SD3-shift with shift=1.0 is identity.
#   3. noisy = (1-sigma)*latent + sigma*noise ; target = noise - latent
#        (musubi ltx2_train.py _build_noisy_input_for_sigma:
#         noisy_input = (1-sigma)*latents + sigma*noise; velocity = noise - x0)
#   4. adaln_delta = adaln_single(timestep_embed(sigma))   (frozen, once/step)
#   5. ltx2_stack_lora_forward_offload(noisy, adaln_delta, ...) -> pred [N,out_ch]
#   6. loss = MSE(pred, target); d_loss = (2/N)(pred - target)
#   7. ltx2_stack_lora_backward_offload -> LoRA grads; global-norm clip(1.0)
#   8. ltx2_lora_adamw_step; print shared progress display
#
# Recipe scalars (configs/ltx2.json): lr=3e-4, rank=16, alpha=1.0,
#   timestep_shift=1.0, clip_grad_norm=1.0.
#
# FIXED_SIGMA_SMOKE: every step uses the SAME sample + fixed sigma+noise so a
# correct LoRA backward MUST drive loss DOWN monotonically (trainer-correctness
# gate, same probe as train_chroma_real).
#
# Run (legacy smoke only):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#       serenitymojo/training/train_ltx2_real.mojo -o /tmp/train_ltx2_real && \
#     /tmp/train_ltx2_real --legacy-video-only [steps]
#
# Readiness status:
#   pixi run mojo run -I . serenitymojo/training/ltx2_av_training_readiness.mojo --expect-not-ready

from std.sys import argv
from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin, exp as fexp
from std.time import perf_counter_ns
from std.os import listdir

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.cast import cast_tensor

from serenitymojo.models.ltx2.weights import load_ltx2_stack_base
from serenitymojo.models.ltx2.ltx2_stack_lora import (
    LTX2LoraSet, LTX2LoraGradSet, build_ltx2_lora_set, total_ltx2_adapters,
    ltx2_adaln_delta, ltx2_stack_lora_forward_offload,
    ltx2_stack_lora_backward_offload, ltx2_lora_adamw_step, save_ltx2_lora,
    ltx2_lora_adamw_state_init, ltx2_lora_adamw_step_resident,
    save_ltx2_lora_state, load_ltx2_lora_state, load_ltx2_lora_resume,
)
from serenitymojo.training.lora_adamw_plain_fused import (
    LoraAdamWPlainDeviceState,
    lora_adamw_plain_device_state_sync_params,
    lora_adamw_plain_device_state_sync_moments,
)
from serenitymojo.training.lora_save import lora_train_state_has_moments
from serenitymojo.training.trainer_core import (
    trainer_resolve_resume_path, trainer_warn_warm_resume,
)
from serenitymojo.models.dit.ltx2_rope import build_ltx2_rope
from serenitymojo.offload.ltx2_plan import build_ltx2_block_plan
from serenitymojo.offload.plan import OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.training.schedule import sample_timestep_logit_normal
from serenitymojo.training.progress_display import print_trainer_progress


# ── arch (LTX-2 video DiT; from configs/ltx2.json) ───────────────────────────
comptime H = 32
comptime Dh = 128
comptime D = H * Dh            # 4096
comptime FF = 16384            # mlp_hidden
comptime IN_CH = 128           # patchify in_channels
comptime OUT_CH = 128
comptime NUM_LAYERS = 48
comptime EPS = Float32(1e-06)

# ── token grid (smoke): frame x height x width latent positions. S = F*HT*WT.
#   Keep the working set bounded for the correctness gate (real configs are
#   larger; S is the only dim that scales the per-block matmuls).
comptime GRID_F = 1
comptime GRID_H = 16
comptime GRID_W = 16
comptime S = GRID_F * GRID_H * GRID_W   # 256 tokens

# ── recipe (configs/ltx2.json) ──────────────────────────────────────────────
comptime RANK = 16
comptime ALPHA = Float32(1.0)
comptime LR = Float32(3.0e-4)
comptime TIMESTEP_SHIFT = Float32(1.0)   # SD3-shift identity at 1.0
comptime NUM_TRAIN_TIMESTEPS = 1000
comptime CLIP_GRAD_NORM = Float32(1.0)
comptime SEED_BASE = UInt64(42)

comptime FIXED_SIGMA_SMOKE = True
comptime FIXED_SIGMA = Float32(0.5)

# ── adaln timestep embedding dim (adaln_single.emb.timestep_embedder in_dim) ──
comptime TEMB_DIM = 256

comptime CKPT = "/home/alex/.serenity/models/checkpoints/ltx2_video_bf16.safetensors"
comptime CACHE_DIR = "/home/alex/datasets/ltx2_cache_512"
comptime LORA_DIR = "/home/alex/mojodiffusion/output/ltx2_lora"


# ── deterministic host gaussian noise (Box-Muller PCG; per-step seed) ─────────
def _host_noise(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    var i = 0
    while i < n:
        state = state * 6364136223846793005 + 1442695040888963407
        var u1f = Float64(Int((state >> 12) & 0xFFFFFFFFFFFFF)) * (1.0 / 4503599627370496.0)
        state = state * 6364136223846793005 + 1442695040888963407
        var u2f = Float64(Int((state >> 12) & 0xFFFFFFFFFFFFF)) * (1.0 / 4503599627370496.0)
        if u1f < 1.0e-12:
            u1f = 1.0e-12
        var r = sqrt(-2.0 * flog(Float64(u1f)))
        var theta = 6.283185307179586 * u2f
        out.append(Float32(r * fcos(Float64(theta))))
        if i + 1 < n:
            out.append(Float32(r * fsin(Float64(theta))))
        i += 2
    return out^


# sinusoidal timestep embedding [dim] for a scalar t. LTX-2 uses
# Timesteps(... flip_sin_to_cos=True, downscale_freq_shift=0), i.e. cos first.
def _sinusoidal_temb(t: Float32, dim: Int) -> List[Float32]:
    var half = dim // 2
    var out = List[Float32]()
    for _ in range(dim):
        out.append(Float32(0.0))
    for i in range(half):
        var freq = fexp(-flog(10000.0) * Float64(i) / Float64(half))
        var arg = Float64(t) * freq
        out[i] = Float32(fcos(arg))
        out[half + i] = Float32(fsin(arg))
    return out^


def _absum(v: List[Float32]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(v)):
        var x = v[i]
        s += x if x >= 0.0 else -x
    return s


def _absum(v: List[BFloat16]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(v)):
        var x = v[i].cast[DType.float32]()
        s += x if x >= 0.0 else -x
    return s


def _global_norm(grads: LTX2LoraGradSet) -> Float64:
    var ss = 0.0
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            ss += Float64(grads.d_a[i][j]) * Float64(grads.d_a[i][j])
        for j in range(len(grads.d_b[i])):
            ss += Float64(grads.d_b[i][j]) * Float64(grads.d_b[i][j])
    return sqrt(ss)


def _clip(mut grads: LTX2LoraGradSet, max_norm: Float32) -> Float64:
    var gn = _global_norm(grads)
    if gn <= Float64(max_norm) or gn == 0.0:
        return gn
    var s = Float32(Float64(max_norm) / gn)
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            grads.d_a[i][j] = grads.d_a[i][j] * s
        for j in range(len(grads.d_b[i])):
            grads.d_b[i][j] = grads.d_b[i][j] * s
    return gn


def _list_cache(dir: String) raises -> List[String]:
    var raw = listdir(dir)
    var fs = List[String]()
    for i in range(len(raw)):
        if raw[i].endswith(".safetensors"):
            fs.append(dir + String("/") + raw[i])
    if len(fs) == 0:
        raise Error(String("ltx2 cache: no .safetensors in ") + dir)
    for i in range(1, len(fs)):
        var j = i
        while j > 0 and fs[j - 1] > fs[j]:
            var tmp = fs[j - 1]
            fs[j - 1] = fs[j]
            fs[j] = tmp
            j -= 1
    return fs^


def _load_host(st: SafeTensors, name: String, ctx: DeviceContext) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    return cast_tensor(t, STDtype.F32, ctx).to_host(ctx)


# ── FULL-state resume resolution + loud warm-resume guard (MJ-1088 / MJ-1077) ──
# Probe the `<ckpt>.state` sidecar so a user who passes the PEFT weights still gets
# a FULL (moment-preserving) resume; only warm-restart (loud warning) when there is
# genuinely no moment state. The probe prefix matches _ltx2_lora_prefixes[0].
def _ltx2_resolve_resume_path(path: String) raises -> String:
    # Thin wrapper → shared trainer_core (binds ltx2's first-adapter probe prefix;
    # keeps krea2's default `.state` sidecar naming). Probe order + return logic are
    # the trainer_core skeleton, byte-identical to the pre-extraction path.
    var probe = String("transformer_blocks.0.attn1.to_q")
    return trainer_resolve_resume_path(path, probe)


def _ltx2_warn_warm_resume(path: String):
    # Thin wrapper → shared trainer_core (binds the ltx2 log tag; keeps krea2's
    # default `.safetensors.state` FULL-resume sidecar hint). Every banner line is
    # byte-identical to the pre-extraction ltx2 output.
    trainer_warn_warm_resume(String("ltx2"), path)


def main() raises:
    var ctx = DeviceContext()
    var a = argv()
    if len(a) < 2 or String(a[1]) != "--legacy-video-only":
        print("LTX2 trainer guard: this binary is the legacy video-only trainer.")
        print("It is not production LTX2 AV training and is not wired to ltx2_dit.")
        print("The legacy smoke still uses host-side noise/loss bookkeeping; production AV must keep the hot loop on device.")
        print("Run training/ltx2_av_training_readiness.mojo --expect-not-ready for the current contract.")
        print("Run with --legacy-video-only only when intentionally testing the old stack.")
        raise Error("train_ltx2_real: production AV trainer not implemented here")

    var run_steps = 5
    if len(a) >= 3:
        var v = 0
        var bs = String(a[2]).as_bytes()
        for i in range(String(a[2]).byte_length()):
            if bs[i] < 0x30 or bs[i] > 0x39:
                raise Error("train_ltx2_real: steps must be a positive integer")
            v = v * 10 + Int(bs[i] - 0x30)
        if v <= 0:
            raise Error("train_ltx2_real: steps must be a positive integer")
        run_steps = v

    # Optional argv[3]: resume checkpoint. Pass either the `<ckpt>.safetensors.state`
    # sidecar (FULL moment resume) or the plain PEFT `.safetensors` (WARM start —
    # the .state sibling is auto-probed first so the PEFT path still full-resumes).
    var resume_path = String("")
    if len(a) >= 4:
        resume_path = String(a[3])

    print("=== LTX-2 LEGACY video-only LoRA training loop (block-swap offload) ===")
    print("  WARNING: legacy stack; not production full-AV LTX2 training")
    print("  arch: D=", D, " H=", H, " Dh=", Dh, " FF=", FF, " in_ch=", IN_CH, " out_ch=", OUT_CH)
    print("  depth: NUM_LAYERS=", NUM_LAYERS)
    print("  tokens: grid", GRID_F, "x", GRID_H, "x", GRID_W, " S=", S)
    print("  recipe: rank=", RANK, " alpha=", ALPHA, " lr=", LR, " shift=", TIMESTEP_SHIFT)
    print("  fixed_sigma_smoke=", FIXED_SIGMA_SMOKE)
    print("  hot loop note: legacy smoke uses host noise/loss; not production AV device-loop training")
    print("  ckpt:", CKPT)
    print("  cache:", CACHE_DIR)

    # ── stack-level base (frozen; patchify_proj/proj_out/adaln_single) ────────
    print("[load] LTX2StackBase (patchify_proj, proj_out, adaln_single)")
    var base_st = SafeTensors.open(String(CKPT))
    var base = load_ltx2_stack_base(base_st, NUM_LAYERS, D, IN_CH, OUT_CH, ctx)
    print("[load] base resident")

    # ── block-swap offload loader ─────────────────────────────────────────────
    var plan = build_ltx2_block_plan(NUM_LAYERS)
    var cfg = OffloadConfig.synchronous_single()
    var loader = TurboPlannedLoader.open(String(CKPT), plan^, cfg, ctx)
    print("[load] offload loader opened (", loader.block_count(), "blocks)")

    # ── split-rope tables [S*H, Dh//2] (built once, BF16 storage) ─────────────
    var rope = build_ltx2_rope[GRID_F, GRID_H, GRID_W](
        H, Dh, 10000.0, Float64(GRID_F), Float64(GRID_H), Float64(GRID_W),
        STDtype.BF16, ctx,
    )
    var cos = rope[0].to_host(ctx)
    var sin = rope[1].to_host(ctx)
    print("[load] ltx2 split-rope tables built (S*H x Dh/2)")

    # ── LoRA set (B=0 init -> identity at step 0) ─────────────────────────────
    var lora = build_ltx2_lora_set(NUM_LAYERS, D, RANK, ALPHA)
    var n_adapters = total_ltx2_adapters(lora)
    print("[lora] adapters:", n_adapters, " (4 x", NUM_LAYERS, "blocks: q,k,v,out)")

    var files = _list_cache(String(CACHE_DIR))
    print("[cache] samples:", len(files))

    var b_absum_init = Float32(0.0)
    for i in range(n_adapters):
        b_absum_init += _absum(lora.ad[i].b)
    print("[lora] LoRA-B |.|_1 at init =", b_absum_init, " (expect 0.0)")

    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)

    # MJ-1085: resident fused AdamW (persistent device P/M/V, one-time pinned
    # staging). The per-step fused arm allocates fresh pinned staging every step
    # and can hit the MJ-1070 unmapped-buffer segfault under pinned pressure.
    var adamw_dev_state = Optional[LoraAdamWPlainDeviceState](None)
    var adamw_state_ready = False

    # ── optional resume: reload A/B (+ AdamW moments if a FULL `.state` exists) ──
    # BEFORE the loop so the lazy resident-AdamW init (ltx2_lora_adamw_state_init)
    # seeds dev_m/dev_v from the restored moments — full-moment fidelity (MJ-1088).
    if resume_path != String(""):
        var resolved = _ltx2_resolve_resume_path(resume_path)
        var is_full = lora_train_state_has_moments(
            resolved, String("transformer_blocks.0.attn1.to_q")
        )
        if is_full:
            lora = load_ltx2_lora_state(NUM_LAYERS, D, RANK, ALPHA, resolved, ctx)
            print("[ltx2-resume] FULL resume (A/B + AdamW moments) from", resolved)
        else:
            _ltx2_warn_warm_resume(resolved)
            lora = load_ltx2_lora_resume(NUM_LAYERS, D, RANK, ALPHA, resolved, ctx)
        print("[ltx2-resume] reloaded", total_ltx2_adapters(lora), "adapters")

    var train_start = perf_counter_ns()
    for k in range(1, run_steps + 1):
        var t0 = perf_counter_ns()

        var slot = 0 if FIXED_SIGMA_SMOKE else (k - 1) % len(files)
        var step_seed = UInt64(1) if FIXED_SIGMA_SMOKE else UInt64(k)
        var st = SafeTensors.open(files[slot])

        # latent tokens [S, in_ch]: cached patchify-ready (channel-last per token).
        var lat = _load_host(st, String("latent"), ctx)   # expect S*IN_CH flat
        # truncate/pad to S*IN_CH (the cache may carry a larger grid).
        var latent = List[Float32]()
        for i in range(S * IN_CH):
            if i < len(lat):
                latent.append(lat[i])
            else:
                latent.append(Float32(0.0))

        # ── timestep / sigma ──
        var sigma: Float32
        if FIXED_SIGMA_SMOKE:
            sigma = FIXED_SIGMA
        else:
            sigma = sample_timestep_logit_normal(SEED_BASE + step_seed, TIMESTEP_SHIFT)
            if sigma > Float32(1.0):
                sigma = Float32(1.0)
            if sigma < Float32(1.0e-3):
                sigma = Float32(1.0e-3)
        var t_model = sigma * Float32(NUM_TRAIN_TIMESTEPS)   # musubi: timesteps = sigma*1000

        # ── flow-match: noisy=(1-sigma)*latent+sigma*noise ; target=noise-latent ──
        var noise = _host_noise(S * IN_CH, SEED_BASE * UInt64(7919) + step_seed)
        var noisy = List[Float32]()
        var target = List[Float32]()
        for i in range(len(latent)):
            noisy.append(latent[i] * (Float32(1.0) - sigma) + noise[i] * sigma)
            target.append(noise[i] - latent[i])

        # ── adaln_single conditioning delta (frozen, once/step) ──
        var temb = _sinusoidal_temb(t_model, TEMB_DIM)
        var adaln_delta = ltx2_adaln_delta(base, temb, ctx)   # [6*D]

        # ── forward (offload, full depth) -> velocity [S, OUT_CH] ──
        var fwd = ltx2_stack_lora_forward_offload[H, Dh, S](
            noisy.copy(), adaln_delta.copy(), base, loader, lora,
            cos.copy(), sin.copy(), D, FF, IN_CH, OUT_CH, EPS, True, ctx,
        )

        # ── loss = MSE(pred, target) ; d_loss = (2/N)(pred - target) ──
        var nout = len(fwd.out)
        var d_loss = List[Float32]()
        var sse = 0.0
        var inv_n = Float32(2.0) / Float32(nout)
        for i in range(nout):
            var diff = fwd.out[i] - target[i]
            sse += Float64(diff) * Float64(diff)
            d_loss.append(inv_n * diff)
        var loss = Float32(sse / Float64(nout))
        if k == 1:
            first_loss = loss
        last_loss = loss

        # ── backward (offload, full depth) ──
        var grads = ltx2_stack_lora_backward_offload[H, Dh, S](
            d_loss, noisy.copy(), base, loader, lora,
            cos.copy(), sin.copy(), fwd,
            D, FF, IN_CH, OUT_CH, EPS, True, ctx,
        )

        # ── grad norm + clip(1.0) ──
        var gn_before = _clip(grads, CLIP_GRAD_NORM)

        # ── AdamW (resident fused arm; MJ-1085) ──
        if not adamw_state_ready:
            adamw_dev_state = Optional[LoraAdamWPlainDeviceState](
                ltx2_lora_adamw_state_init(lora, ctx)
            )
            adamw_state_ready = True
            print("[ltx2-adamw] resident fused state initialized (",
                  total_ltx2_adapters(lora), "adapters )")
        ltx2_lora_adamw_step_resident(
            adamw_dev_state.value(), lora, grads, k, LR, ctx,
        )

        var t1 = perf_counter_ns()
        var secs = Float64(t1 - t0) / 1.0e9
        print_trainer_progress(
            String("LTX2-lora"), k, run_steps, 1,
            loss, Float64(gn_before), secs, 0.0,
            Float64(t1 - train_start) / 1.0e9,
        )
        if grads.nonfinite_lora_grads != 0:
            print("[LTX2-lora] warning nonfinite_lora_grads=", grads.nonfinite_lora_grads)

    print("")
    print("first_loss=", first_loss, " last_loss=", last_loss)
    var b_absum_final = Float32(0.0)
    for i in range(n_adapters):
        b_absum_final += _absum(lora.ad[i].b)
    var trains = (b_absum_init == 0.0) and (b_absum_final > 0.0)
    if trains and (last_loss == last_loss):
        print("RESULT: REAL run OK — LoRA-B grew 0 ->", b_absum_final,
              "; loss", first_loss, "->", last_loss,
              (" (DECREASED)" if last_loss < first_loss else " (see trajectory)"))
        var peft_path = String(LORA_DIR) + String("/ltx2_lora_smoke.safetensors")
        _ = save_ltx2_lora(lora, peft_path, ctx)
        # MJ-1088: also write the moment-carrying `.state` sidecar so a resume does
        # NOT zero AdamW momentum. Pull the resident device m/v back to host first —
        # the resident step syncs params every step but moments only on demand.
        if adamw_state_ready:
            lora_adamw_plain_device_state_sync_params(
                adamw_dev_state.value(), lora.ad, ctx
            )
            lora_adamw_plain_device_state_sync_moments(
                adamw_dev_state.value(), lora.ad, ctx
            )
        var meta = List[Float32]()
        meta.append(Float32(run_steps))
        meta.append(Float32(Int(SEED_BASE)))
        var state_path = peft_path + String(".state")
        _ = save_ltx2_lora_state(lora, state_path, ctx, meta^)
        print("[ltx2] save_state (A/B + AdamW moments) path=", state_path)
    else:
        print("RESULT: FAIL trains=", trains)
