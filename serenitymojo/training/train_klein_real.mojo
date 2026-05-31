# train_klein_real.mojo — the INTEGRATED Klein-9B LoRA training loop.
#
# Final assembly: every component below is already built + lead-verified. This
# WIRES them into the real loop and MEASURES per-step wall-clock.
#
# Per step (real 512px: N_IMG=1024, N_TXT=512, S=1536):
#   1. pick a cache sample -> latent [1,128,32,32] + text_embedding [1,512,12288]
#   2. latent -> img_tokens [1024,128] (NCHW->NHWC pack, mirrors initial_tokens)
#   3. sample sigma (logit-normal); flow-match in TOKEN space:
#        x_t    = (1-sigma)*latent_tokens + sigma*noise
#        target = noise - latent_tokens                       (v-prediction)
#   4. build per-timestep modulation vecs from sigma*1000 (BFL time_factor)
#   5. klein_stack_lora_forward(x_t, txt_tokens, ...) -> velocity [1024,128]
#   6. loss = MSE(velocity, target);  d_loss = 2/N * (velocity - target)
#   7. klein_stack_lora_backward -> LoRA grads ;  grad_norm = L2(all LoRA grads)
#   8. klein_lora_adamw_step
#   PRINT (machine-parseable, one per completed step):
#     PROG step=<k> total=<MAX> loss=<f> grad=<f> lr=<f> secs=<wallclock>
#
# Cadence (MAX_STEPS=25):
#   step 0 (before training):           sample baseline
#   step 10:  save + sample + reload via load_klein_lora_resume (prove resume)
#   step 25 (FINAL): save + sample, then stop.
#   EVENT markers: `EVENT sample step=k path=...`, `EVENT save step=k path=...`,
#   `EVENT resume step=k`.
#
# MEMORY: this process loads the full resident Klein-9B base stack (8 double +
# 24 single + projections) + the 80-adapter LoRA set. It does NOT import
# Qwen3Encoder (the ~16 GB encoder ran in klein_prepare_alina.mojo, a separate
# process that already exited). The validation SAMPLE path
# (generate_validation) loads a SECOND resident Klein9BDiT + VAE — see the
# MEMORY NOTE at the sample call site.
#
# Run (2-step timed dry run — the lead's launch decision is from this number):
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/training/train_klein_real.mojo

from std.collections import List
from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin, exp as fexp
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors

from serenitymojo.models.klein.double_block import DoubleBlockWeights
from serenitymojo.models.klein.single_block import SingleBlockWeights
from serenitymojo.models.klein.klein_stack import KleinStackBase, KleinStackForward
from serenitymojo.models.klein.klein_stack_lora import (
    KleinLoraSet, build_klein_lora_set,
    klein_stack_lora_forward, klein_stack_lora_forward_device_inputs,
    klein_stack_lora_backward,
    klein_lora_adamw_step, save_klein_lora, load_klein_lora_resume,
)
from serenitymojo.models.klein.weights import (
    load_double_block_weights, load_single_block_weights,
    load_klein_stack_base, build_klein_vec_silu,
    build_klein_double_modvecs, build_klein_single_modvecs,
)
from serenitymojo.models.klein.double_block import ModVecs as ModVecsT
from serenitymojo.models.klein.single_block import SingleModVecs as SingleModVecsT
from serenitymojo.training.klein_dataset import KleinCache, KleinSample
from serenitymojo.training.schedule import sample_timestep_logit_normal, flow_match_noise_target
from serenitymojo.training.validation_sampler import (
    generate_validation, load_caps, pixel_l1,
)
from serenitymojo.ops.cast import cast_tensor_if_needed
from serenitymojo.ops.tensor_algebra import permute, reshape, reshape_owned
from serenitymojo.io.ffi import sys_system


comptime TArc = ArcPointer[Tensor]


# ── paths / dims ─────────────────────────────────────────────────────────────
comptime KLEIN9B_PATH = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-4b.safetensors"
comptime VAE_PATH = "/home/alex/.serenity/models/vaes/flux2-vae.safetensors"
comptime CACHE_DIR = "/home/alex/mojodiffusion/output/alina_cache_4b"
comptime CAPS_POS = "/home/alex/mojodiffusion/output/klein9b_caps_pos.bin"
comptime CAPS_NEG = "/home/alex/mojodiffusion/output/klein9b_caps_neg.bin"
comptime SAMPLE_DIR = "/home/alex/mojodiffusion/output/alina_train"
comptime LORA_DIR = "/home/alex/mojodiffusion/output/alina_train"

# real 512px latent grid: 32x32 packed -> 1024 image tokens.
comptime LH = 32
comptime LW = 32
comptime N_IMG = 1024
comptime N_TXT = 512
comptime S = N_IMG + N_TXT
comptime H = 24
comptime Dh = 128
comptime NUM_DOUBLE = 5
comptime NUM_SINGLE = 20
comptime TIMESTEP_DIM = 256

# recipe
comptime D = 3072
comptime F = 9216
comptime IN_CH = 128
comptime TXT_CH = 7680
comptime OUT_CH = 128
comptime EPS = Float32(1.0e-6)
comptime RANK = 16
comptime ALPHA = Float32(16.0)
comptime LR = Float32(1.0e-4)
comptime MAX_GRAD_NORM = Float32(1.0)   # EDv2 default-ON global-norm grad clip
comptime TIMESTEP_SHIFT = Float32(1.8)   # Klein 9B empirical sweet spot (memory)
comptime SEED_BASE = UInt64(1234)

# Cadence — overridable for the dry run via DRY_STEPS.
comptime MAX_STEPS = 25
# RUN_STEPS controls THIS invocation. The lead launches the full run by editing
# this to MAX_STEPS (25). The 2-step timed dry run keeps it at 2.
comptime RUN_STEPS = 1
# DO_SAMPLE gates the in-process validation sampling (generate_validation loads
# a SECOND resident Klein9BDiT — see the MEMORY NOTE). Off for the timing dry
# run so the per-step TRAINING seconds are measured cleanly; the lead flips it
# on (or runs sampling out-of-process) for the full 25-step launch.
comptime DO_SAMPLE = False
comptime SAMPLE_STEPS = 20
comptime SAMPLE_CFG = Float32(4.0)
comptime SAMPLE_SEED = UInt64(42)


# ─────────────────────────────────────────────────────────────────────────────
# host helpers
# ─────────────────────────────────────────────────────────────────────────────


# Latent [1,128,LH,LW] (F32 device) -> img_tokens host List [N_IMG, IN_CH].
# Mirrors klein9b_pipeline_multistep_smoke.initial_tokens packing exactly:
# NCHW -> permute(0,2,3,1) -> NHWC -> reshape [1,N_IMG,128] -> to_host.
def _latent_to_img_tokens(latent: Tensor, ctx: DeviceContext) raises -> List[Float32]:
    var p = List[Int]()
    p.append(0); p.append(2); p.append(3); p.append(1)
    var nhwc = permute(latent, p^, ctx)
    var sh = List[Int]()
    sh.append(1); sh.append(N_IMG); sh.append(IN_CH)
    var packed = reshape_owned(nhwc^, sh^)
    return packed.to_host(ctx)


def _latent_to_img_tokens_device(latent: Tensor, ctx: DeviceContext) raises -> Tensor:
    var p = List[Int]()
    p.append(0); p.append(2); p.append(3); p.append(1)
    var nhwc = permute(latent, p^, ctx)
    var sh = List[Int]()
    sh.append(N_IMG); sh.append(IN_CH)
    return reshape_owned(nhwc^, sh^)


# Build the Klein rope tables as flat host Lists [S*H*(Dh//2)] — the layout the
# LoRA stack consumes. Replicates build_klein_rope_tables (klein_dit.mojo:522)
# host loop EXACTLY (4-axis position rope, theta=2000, 16 freqs/axis).
def _build_klein_rope_host() raises -> Tuple[List[Float32], List[Float32]]:
    var img_w = 1
    while img_w * img_w < N_IMG:
        img_w += 1
    if img_w * img_w != N_IMG:
        raise Error("N_IMG must be a square grid")
    var cos_vals = List[Float32]()
    var sin_vals = List[Float32]()
    var log_theta = flog(Float32(2000.0))
    for tok in range(S):
        var p0 = 0
        var p1 = 0
        var p2 = 0
        var p3 = 0
        if tok >= N_TXT:
            var idx = tok - N_TXT
            p1 = idx // img_w
            p2 = idx % img_w
        for _h in range(H):
            for axis in range(4):
                var pos = p0
                if axis == 1:
                    pos = p1
                elif axis == 2:
                    pos = p2
                elif axis == 3:
                    pos = p3
                for i in range(16):
                    var inv_freq = fexp(-log_theta * Float32(2 * i) / Float32(32))
                    var angle = Float32(pos) * inv_freq
                    cos_vals.append(fcos(angle))
                    sin_vals.append(fsin(angle))
    return (cos_vals^, sin_vals^)


# Deterministic host gaussian noise of length n (Box-Muller on a PCG stream),
# seeded per step so the flow-match draw is reproducible.
def _host_noise(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    var i = 0
    while i < n:
        state = state * 6364136223846793005 + 1442695040888963407
        var u1f = Float64(Int((state >> 11) & 0xFFFFFFFFFFFFF)) * (1.0 / 9007199254740992.0)
        state = state * 6364136223846793005 + 1442695040888963407
        var u2f = Float64(Int((state >> 11) & 0xFFFFFFFFFFFFF)) * (1.0 / 9007199254740992.0)
        if u1f < 1.0e-12:
            u1f = 1.0e-12
        var r = sqrt(-2.0 * flog(Float64(u1f)))
        var theta = 6.283185307179586 * u2f
        out.append(Float32(r * fcos(Float64(theta))))
        if i + 1 < n:
            out.append(Float32(r * fsin(Float64(theta))))
        i += 2
    return out^


# L2 norm of a List.
def _l2(h: List[Float32]) -> Float64:
    var s = 0.0
    for i in range(len(h)):
        var v = Float64(h[i])
        s += v * v
    return sqrt(s)


# abs-sum of a List (dead-branch check).
def _abs_sum(h: List[Float32]) -> Float64:
    var s = 0.0
    for i in range(len(h)):
        var v = h[i]
        s += Float64(v) if v >= 0.0 else Float64(-v)
    return s


# scale a List in place by `s` (global-norm grad clip).
def _scale_inplace(mut h: List[Float32], s: Float32):
    for i in range(len(h)):
        h[i] = h[i] * s


# ─────────────────────────────────────────────────────────────────────────────
# Per-step modulation vecs from the sampled sigma (timestep = sigma*1000).
# Rebuilt each step because the timestep varies; small linears, cheap.
# ─────────────────────────────────────────────────────────────────────────────
def _build_step_mods(
    st: SafeTensors, sigma: Float32, ctx: DeviceContext
) raises -> Tuple[ModVecsT, ModVecsT, SingleModVecsT]:
    var tvals = List[Float32]()
    tvals.append(sigma * Float32(1000.0))
    var tsh = List[Int]()
    tsh.append(1)
    var ts = Tensor.from_host(tvals, tsh^, STDtype.F32, ctx)
    var vec_silu = build_klein_vec_silu(st, ts, TIMESTEP_DIM, D, ctx)
    var img_mod = build_klein_double_modvecs(st, vec_silu, String("img"), D, ctx)
    var txt_mod = build_klein_double_modvecs(st, vec_silu, String("txt"), D, ctx)
    var single_mod = build_klein_single_modvecs(st, vec_silu, D, ctx)
    return (img_mod^, txt_mod^, single_mod^)


# ─────────────────────────────────────────────────────────────────────────────
# Sample event: drive generate_validation WITH the just-saved LoRA, write a PNG,
# and pixel-diff vs the no-LoRA baseline (the sample-shift gate).
#
# MEMORY NOTE: generate_validation loads a SECOND resident Klein9BDiT
# (load_full) + a VAE decoder. On a 24 GB card this co-resides with the training
# stack already held in THIS process. If it OOMs, the dry run will surface it;
# the fallback (documented in the RETURN) is to run sampling in a separate
# process that reads the saved LoRA from disk — the encode/denoise split already
# proves that pattern.
# ─────────────────────────────────────────────────────────────────────────────
def _do_sample(
    step: Int, lora_path: String, caps_pos: String, caps_neg: String,
    ctx: DeviceContext,
) raises:
    var png = SAMPLE_DIR + String("/sample_step") + String(step) + String(".png")
    var caps = load_caps(caps_pos, caps_neg, ctx)
    var mult = Float32(0.0) if lora_path == String("") else Float32(1.0)
    var _img = generate_validation[N_IMG, N_TXT, S, LH, LW](
        KLEIN9B_PATH, VAE_PATH, caps, lora_path, mult,
        SAMPLE_STEPS, SAMPLE_CFG, SAMPLE_SEED, png, ctx,
    )
    print("EVENT sample step=", step, " path=", png)


def main() raises:
    var ctx = DeviceContext()
    _ = sys_system(String("mkdir -p ") + SAMPLE_DIR)

    # DRY-RUN knob: if env MAX_STEPS_OVERRIDE is set, use it; else MAX_STEPS.
    # (Mojo has no easy getenv wired here, so the dry run is a comptime swap —
    #  see DRY_STEPS below; for the 2-step timed run set RUN_STEPS=2.)
    var run_steps = RUN_STEPS

    print("=== Klein-9B REAL LoRA training loop ===")
    print(
        "  512px latent: N_IMG=", N_IMG, " N_TXT=", N_TXT, " S=", S,
        " rank=", RANK, " alpha=", ALPHA, " lr=", LR, " shift=", TIMESTEP_SHIFT,
    )
    print("  cadence target MAX_STEPS=", MAX_STEPS, "  this run RUN_STEPS=", run_steps)

    # ── load the resident base stack + block weights ONCE ─────────────────────
    print("[load] Klein-9B base stack + block weights")
    var st = SafeTensors.open(KLEIN9B_PATH)
    # vec_silu at sigma=0.5 just to seed the base struct's final-layer mod; the
    # PER-STEP mods (built from the sampled sigma) are what the loop uses.
    var seed_ts = Tensor.from_host([Float32(500.0)], [1], STDtype.F32, ctx)
    var seed_vec_silu = build_klein_vec_silu(st, seed_ts, TIMESTEP_DIM, D, ctx)
    var base = load_klein_stack_base(st, seed_vec_silu, D, ctx)
    var dbw = List[DoubleBlockWeights]()
    for bi in range(NUM_DOUBLE):
        dbw.append(load_double_block_weights(st, bi, ctx))
    var sbw = List[SingleBlockWeights]()
    for bi in range(NUM_SINGLE):
        sbw.append(load_single_block_weights(st, bi, ctx))
    print("  loaded", len(dbw), "double +", len(sbw), "single block weights")

    # ── build LoRA set (80 adapters) + rope tables ────────────────────────────
    var lora = build_klein_lora_set(NUM_DOUBLE, NUM_SINGLE, D, RANK, ALPHA)
    print("  LoRA set:", len(lora.dbl), "double-slot +", len(lora.sgl), "single-slot")
    var rope = _build_klein_rope_host()
    var cos = rope[0].copy()
    var sin = rope[1].copy()
    print("  rope host tables:", len(cos), "cos /", len(sin), "sin (expect", S * H * (Dh // 2), ")")

    # ── open cache ────────────────────────────────────────────────────────────
    var cache = KleinCache(CACHE_DIR)
    print("  cache samples:", cache.count())

    # ── baseline sample (step 0, before any training) ─────────────────────────
    if DO_SAMPLE:
        print("[cadence] step 0 baseline sample (no LoRA)")
        _do_sample(0, String(""), CAPS_POS, CAPS_NEG, ctx)
    else:
        print("[cadence] step 0 baseline sample SKIPPED (DO_SAMPLE=False, timing run)")

    # ── training loop ─────────────────────────────────────────────────────────
    for k in range(1, run_steps + 1):
        var t0 = perf_counter_ns()

        # pick a sample (round-robin)
        var sample = cache.load((k - 1) % cache.count(), ctx)
        var latent_tokens_t = cast_tensor_if_needed(
            _latent_to_img_tokens_device(sample.latent, ctx), STDtype.F32, ctx
        )
        # text_embedding [1,512,TXT_CH] -> device [N_TXT, TXT_CH]
        var txt_sh = List[Int]()
        txt_sh.append(N_TXT); txt_sh.append(TXT_CH)
        var txt_tokens_t = cast_tensor_if_needed(
            reshape(sample.text_embedding, txt_sh^, ctx), STDtype.F32, ctx
        )

        var n_img_vals = N_IMG * IN_CH

        # sample sigma (logit-normal + qwen-shift)
        var sigma = sample_timestep_logit_normal(SEED_BASE + UInt64(k), TIMESTEP_SHIFT)

        # flow-match in token space (GPU arithmetic — matches schedule.mojo math):
        #   x_t    = (1-sigma)*latent + sigma*noise
        #   target = noise - latent
        var noise = _host_noise(n_img_vals, SEED_BASE * UInt64(7919) + UInt64(k))
        var noise_t = Tensor.from_host(noise^, [N_IMG, IN_CH], STDtype.F32, ctx)
        var fm = flow_match_noise_target(latent_tokens_t, sigma, noise_t, ctx)
        var x_t_dev = TArc(fm.x_t.clone(ctx))
        var target = fm.target.to_host(ctx)

        # per-step modulation vecs from this sigma
        var mods = _build_step_mods(st, sigma, ctx)
        var img_mod = mods[0].copy()
        var txt_mod = mods[1].copy()
        var single_mod = mods[2].copy()

        # forward -> velocity [N_IMG, OUT_CH]
        var fwd = klein_stack_lora_forward_device_inputs[H, Dh, N_IMG, N_TXT, S](
            x_t_dev, TArc(txt_tokens_t^), base,
            dbw, sbw, lora, img_mod, txt_mod, single_mod, cos.copy(), sin.copy(),
            D, F, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
        )

        # MSE loss + d_loss = (2/N)*(velocity - target)
        var nout = len(fwd.out)
        var sse = 0.0
        var d_loss = List[Float32]()
        var inv_n = Float32(2.0) / Float32(nout)
        for i in range(nout):
            var diff = fwd.out[i] - target[i]
            sse += Float64(diff) * Float64(diff)
            d_loss.append(inv_n * diff)
        var loss = Float32(sse / Float64(nout))

        # backward -> LoRA grads
        var empty_img = List[Float32]()
        var empty_txt = List[Float32]()
        var g = klein_stack_lora_backward[H, Dh, N_IMG, N_TXT, S](
            d_loss, empty_img^, empty_txt^, base,
            dbw, sbw, lora, img_mod, txt_mod, single_mod, cos.copy(), sin.copy(), fwd,
            D, F, IN_CH, TXT_CH, OUT_CH, EPS, ctx, False, False,
        )

        # grad_norm = L2 of ALL LoRA d_A/d_B
        var gsum = 0.0
        var nd = NUM_DOUBLE * 4
        for i in range(nd):
            var a = _l2(g.dbl_d_a[i]); var b = _l2(g.dbl_d_b[i])
            gsum += a * a + b * b
        var ns = NUM_SINGLE * 2
        for i in range(ns):
            var a = _l2(g.sgl_d_a[i]); var b = _l2(g.sgl_d_b[i])
            gsum += a * a + b * b
        var grad_norm = sqrt(gsum)

        # ── dead-adapter warn (project's #1 silent failure) ───────────────────
        # B legitimately starts at 0, so its grad can be ~0 early; warn when an
        # adapter's TOTAL |d_A|+|d_B| == 0 at step>=1 (a truly dead branch).
        for i in range(nd):
            if _abs_sum(g.dbl_d_a[i]) + _abs_sum(g.dbl_d_b[i]) == 0.0:
                print("EVENT dead_adapter step=", k, " idx=", i, " kind=double")
        for i in range(ns):
            if _abs_sum(g.sgl_d_a[i]) + _abs_sum(g.sgl_d_b[i]) == 0.0:
                print("EVENT dead_adapter step=", k, " idx=", nd + i, " kind=single")

        # ── global-norm grad clip across ALL LoRA grads (EDv2 default-ON) ──────
        var clip_scale = Float32(1.0)
        if grad_norm > Float64(MAX_GRAD_NORM):
            clip_scale = MAX_GRAD_NORM / Float32(grad_norm)
            for i in range(nd):
                _scale_inplace(g.dbl_d_a[i], clip_scale)
                _scale_inplace(g.dbl_d_b[i], clip_scale)
            for i in range(ns):
                _scale_inplace(g.sgl_d_a[i], clip_scale)
                _scale_inplace(g.sgl_d_b[i], clip_scale)

        # AdamW step (on clipped grads)
        klein_lora_adamw_step(lora, g, k, LR, ctx)

        var t1 = perf_counter_ns()
        var secs = Float64(t1 - t0) / 1.0e9

        # machine-parseable progress line (consumed by the tqdm wrapper).
        # clip=<scale> is 1.0 when no clip applied, else MAX_GRAD_NORM/grad_norm.
        print(
            "PROG step=", k, " total=", MAX_STEPS, " loss=", loss,
            " grad=", Float32(grad_norm), " lr=", LR, " clip=", clip_scale,
            " secs=", Float32(secs),
        )
        # ── cadence ───────────────────────────────────────────────────────────
        if k == 10 and run_steps >= 10:
            var p10 = LORA_DIR + String("/alina_lora_step10.safetensors")
            var npairs = save_klein_lora(lora, p10, ctx)
            print("EVENT save step=", k, " path=", p10, " pairs=", npairs)
            if DO_SAMPLE:
                _do_sample(k, p10, CAPS_POS, CAPS_NEG, ctx)
            # prove resume: reload the adapters byte-exact and continue.
            var reloaded = load_klein_lora_resume(NUM_DOUBLE, NUM_SINGLE, RANK, ALPHA, p10, ctx)
            lora = reloaded^
            print("EVENT resume step=", k)

        if (k == MAX_STEPS and run_steps >= MAX_STEPS) or k == run_steps:
            var pf = LORA_DIR + String("/alina_lora_final.safetensors")
            var npairs = save_klein_lora(lora, pf, ctx)
            print("EVENT save step=", k, " path=", pf, " pairs=", npairs)
            if DO_SAMPLE:
                _do_sample(k, pf, CAPS_POS, CAPS_NEG, ctx)

    print("")
    print("DONE: ran", run_steps, "steps of", MAX_STEPS, "target")
