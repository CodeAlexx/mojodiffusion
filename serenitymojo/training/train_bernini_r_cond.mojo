# serenitymojo/training/train_bernini_r_cond.mojo
#
# BERNINI-R conditioning trainer — the reference-guided renderer, ALL 12 tasks.
# Pure Mojo + MAX, GPU, block-swap offload. Sequence-concat of N CLEAN
# conditioning latents (each with its own source-id RoPE) AHEAD of the noised
# TARGET, full self-attention over the packed sequence, velocity-MSE loss on the
# TARGET region ONLY.
#
# ── This file is the ASSEMBLY + LOOP only. It reimplements NO block/stack math ──
# It REUSES verbatim:
#   * the certified Wan2.2-A14B LoRA engine (wan22_stack_lora) — the block runs
#     over WHATEVER sequence + cos/sin it is given, so a packed
#     [cond_1 | ... | cond_N | target] sequence needs no new kernel;
#   * the Tier-2a source-id RoPE (bernini_src_id_rope.build_bernini_src_id_rope);
#   * the proven patchify (ops/patchify3d) and safetensors reader;
#   * the Bernini timestep samplers (schedule.bernini_sample_sigma — logit_normal
#     + mode density, per-shift window) and EMA (schedule.ema_update math).
#
# ── The 12 tasks (task -> recipe in training/bernini_tasks.mojo) ────────────────
#   task selected by env BERNINI_TASK (default "t2v"). The recipe fixes:
#     shift (3/4/5), weighting (logit_normal image / mode video), n_cond segments.
#   n_cond=0  (t2i/t2v)  -> Tier-1 T2V path (packed == target, src_id=0).
#   n_cond>=1            -> N CLEAN conditioning segments src_id 1..N packed AHEAD
#                           of the target (src_id 0), Bernini pack_vae_latents order.
#   BERNINI_NO_COND=1 forces n_cond=0 regardless of task (regression switch).
#
#   Each packed sample shares ONE timestep (data.py:365) applied to all tokens —
#   matches the certified stack's single-temb AdaLN bit-for-bit.
#
# ── BUILD (binary; JIT can't resolve the driver calls) ─────────────────────────
#   cd /home/alex/mojodiffusion; rm -f serenitymojo.mojopkg
#   pixi run mojo build --optimization-level 2 --target-accelerator sm_120 \
#     -I . -I /home/alex/MOJO-libs -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/training/train_bernini_r_cond.mojo -o output/bin/train_bernini_r_cond
# ── RUN ────────────────────────────────────────────────────────────────────────
#   BERNINI_TASK=i2v output/bin/train_bernini_r_cond [config.json]
#
# Mojo 1.0.0b1, NVIDIA GPU (sm_120 / 16GB).

from std.gpu.host import DeviceContext
from std.collections import List
from std.math import sqrt, exp
from std.time import perf_counter_ns
from std.sys import argv
from std.ffi import external_call
from std.memory import alloc
from std.builtin.type_aliases import MutExternalOrigin

from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_open, sys_close, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.ops.patchify3d import patchify3d
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.train_config import TrainConfig

# REUSE: the certified A14B LoRA engine (NO block/stack math re-implemented here).
from serenitymojo.models.wan22.weights import load_wan22_stack_base, detect_wan22_prefix
from serenitymojo.models.wan22.wan22_stack_lora import (
    Wan22StackBase, Wan22LoraSet, Wan22LoraGradSet, Wan22StackForward,
    build_wan22_lora_set, wan22_total_adapters,
    wan22_stack_lora_forward_offload, wan22_stack_lora_backward_offload,
    wan22_lora_adamw_step, save_wan22_lora, save_wan22_lora_state,
)
from serenitymojo.offload.wan22_plan import build_wan22_block_plan
from serenitymojo.offload.plan import OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader

# REUSE: the Tier-2a source-id RoPE (the packed-sequence rope builder).
from serenitymojo.models.wan22.bernini_src_id_rope import (
    BerniniRopeSegment, build_bernini_src_id_rope,
)

# REUSE: the Bernini timestep samplers (logit_normal + mode, per-shift window).
from serenitymojo.training.schedule import (
    BerniniWindow, bernini_task_window, bernini_sample_sigma,
)

# REUSE: the LoRA adapter carrier (to build the EMA-shadow set for saving).
from serenitymojo.training.train_step import LoraAdapter

# The per-task recipe table (shift / weighting / n_cond / prompt) + smoke geometry.
from serenitymojo.training.bernini_tasks import (
    BerniniTaskRecipe, bernini_recipe_for, BerniniCondSeg, bernini_smoke_cond_segments,
)


# ── A14B architecture (comptime; the wired engine's dims — same as Tier-1) ─────
comptime H = 40
comptime Dh = 128
comptime DIM = H * Dh            # 5120
comptime FFN = 13824
comptime IN_CH = 64              # VAE latent 16ch * patch(1,2,2)
comptime OUT_CH = 64             # velocity of the 16-ch latent
comptime TEXT_DIM = 4096
comptime FREQ_DIM = 256
comptime NUM_BLOCKS = 40
comptime EPS = Float32(1.0e-6)
comptime ROPE_THETA = Float32(10000.0)

# ── Target geometry (fixed): latent [16,1,32,32] -> patchify(1,2,2) -> (1,16,16). ─
comptime TGT_FG = 1
comptime TGT_HG = 16
comptime TGT_WG = 16
comptime S_TGT = TGT_FG * TGT_HG * TGT_WG          # 256 target tokens
comptime TXT = 512                                 # umt5 cross-attn kv length
comptime TGT_SRC_ID = Float32(0.0)                 # target == stock wan rope

# Bernini noise window (bernini_renderer_high.yaml).
comptime NOISE_TMIN = Float64(0.875)
comptime NOISE_TMAX = Float64(1.0)
comptime LOGIT_MEAN = Float64(0.5)
comptime LOGIT_STD = Float64(1.0)
comptime MODE_SCALE = Float64(1.29)

# ── optimizer / recipe defaults (config overrides where present) ───────────────
comptime DEFAULT_LR = Float32(2.0e-4)
comptime DEFAULT_MAX_GRAD_NORM = Float32(1.0)
comptime DEFAULT_EMA_DECAY = Float32(0.9999)       # bernini_renderer_high.yaml ema_decay
comptime DEFAULT_CKPT_LOW =
    "/home/alex/.serenity/models/checkpoints/Bernini-R-Diffusers/serenity_fp8_e4m3_de8c462/low"
comptime DEFAULT_CONFIG =
    "serenitymojo/configs/bernini_r_t2v_cond_smoke.json"


# ══════════════════════════════════════════════════════════════════════════════
# small host helpers (self-contained; the RNG mirrors train_wan22_real exactly)
# ══════════════════════════════════════════════════════════════════════════════
def _file_exists(path: String) -> Bool:
    var fd = sys_open(path, O_RDONLY, Int32(0))
    if fd < 0:
        return False
    _ = sys_close(fd)
    return True


comptime _EnvPtr = UnsafePointer[UInt8, MutExternalOrigin]


def _env_is_set(name: String) -> Bool:
    var n = name.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = name.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    var cname = _EnvPtr(unsafe_from_address=Int(buf))
    var ret = external_call["getenv", _EnvPtr](cname)
    buf.free()
    if Int(ret) == 0:
        return False
    return ret[0] == UInt8(49) and ret[1] == UInt8(0)


# Read an env var STRING value (empty String if unset). Small fixed buffer read.
def _env_str(name: String) -> String:
    var n = name.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = name.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    var cname = _EnvPtr(unsafe_from_address=Int(buf))
    var ret = external_call["getenv", _EnvPtr](cname)
    buf.free()
    if Int(ret) == 0:
        return String("")
    var out = String("")
    var i = 0
    while ret[i] != 0:
        out += chr(Int(ret[i]))
        i += 1
    return out


def _base_file_for(ckpt: String) -> String:
    var shared = ckpt + String("/shared.safetensors")
    if _file_exists(shared):
        return shared
    return ckpt


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def _u01(seed: UInt64) -> Float32:
    var w = _splitmix64(seed)
    return Float32(Float64(w >> 40) * (1.0 / 16777216.0))


def _fcos(x: Float32) -> Float32:
    from std.math import cos as _c
    return Float32(_c(Float64(x)))


def _flog(x: Float32) -> Float32:
    from std.math import log as _l
    return Float32(_l(Float64(x)))


def _randn(seed: UInt64) -> Float32:
    var u1 = _u01(seed * UInt64(2654435761) + 1)
    var u2 = _u01(seed * UInt64(1442695040888963407) + 2)
    if u1 < Float32(1.0e-7):
        u1 = Float32(1.0e-7)
    var r = sqrt(Float32(-2.0) * _flog(u1))
    var ang = Float32(6.2831853071795864769) * u2
    return r * _fcos(ang)


def _noise(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(_randn(seed + UInt64(i) * 2 + 1))
    return out^


def _synth(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(_randn(seed * UInt64(1099511628211) + UInt64(i) * 3 + 5))
    return out^


# global grad clip over all LoRA d_A/d_B (musubi max_grad_norm).
def _clip(mut grads: Wan22LoraGradSet, max_norm: Float32) -> Float32:
    var ss = Float32(0.0)
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            ss += grads.d_a[i][j] * grads.d_a[i][j]
        for j in range(len(grads.d_b[i])):
            ss += grads.d_b[i][j] * grads.d_b[i][j]
    var gn = sqrt(ss)
    if gn > max_norm and gn > Float32(0.0):
        var scl = max_norm / gn
        for i in range(len(grads.d_a)):
            for j in range(len(grads.d_a[i])):
                grads.d_a[i][j] = grads.d_a[i][j] * scl
            for j in range(len(grads.d_b[i])):
                grads.d_b[i][j] = grads.d_b[i][j] * scl
    return gn


def _grad_abs_sum(grads: Wan22LoraGradSet) -> Float32:
    var s = Float32(0.0)
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            s += abs(grads.d_a[i][j])
        for j in range(len(grads.d_b[i])):
            s += abs(grads.d_b[i][j])
    return s


# ── Load a CLEAN latent [16, ff, fh, fw] and patchify(1,2,2) -> [S, IN_CH] flat. ─
# ff = temporal frames (1 for images, >1 for video conditioning). Returns
# (flat [S*IN_CH]) and writes std/mean of the source latent.
def _patchify_latent(
    vals: List[Float32], ff: Int, fh: Int, fw: Int, ctx: DeviceContext,
    mut out_std: Float32, mut out_mean: Float32,
) raises -> List[Float32]:
    var n = 16 * ff * fh * fw
    if len(vals) != n:
        raise Error("latent numel mismatch in _patchify_latent")
    var s = Float64(0.0)
    var ss = Float64(0.0)
    for i in range(n):
        s += Float64(vals[i])
        ss += Float64(vals[i]) * Float64(vals[i])
    var mean = s / Float64(n)
    var varp = ss / Float64(n) - mean * mean
    if varp < 0.0:
        varp = 0.0
    out_mean = Float32(mean)
    out_std = Float32(sqrt(varp))
    var shp = List[Int]()
    shp.append(16)
    shp.append(ff)
    shp.append(fh)
    shp.append(fw)
    var t = Tensor.from_host(vals.copy(), shp^, STDtype.F32, ctx)
    var patched = patchify3d(t, 1, 2, 2, ctx)   # [S, IN_CH]
    return patched.to_host(ctx)


# ── Read the cached target latent [16,1,32,32] f32 from sample_{idx}. ──────────
def _load_cache_target(
    cache_dir: String, idx: Int, ctx: DeviceContext,
    mut out_std: Float32, mut out_mean: Float32,
) raises -> List[Float32]:
    var path = cache_dir + String("/sample_") + String(idx) + String(".safetensors")
    var st = SafeTensors.open(path)
    var info = st.tensor_info(String("latent"))
    if info.dtype != STDtype.F32:
        raise Error(String("cache latent dtype != F32 in ") + path)
    var nlat = 1
    for i in range(len(info.shape)):
        nlat *= info.shape[i]
    if nlat != 16 * 1 * 32 * 32:
        raise Error(String("cache latent numel != 16*1*32*32 in ") + path)
    var lb = st.tensor_bytes(String("latent"))
    var fp = lb.unsafe_ptr().bitcast[Float32]()
    var vals = List[Float32]()
    for i in range(nlat):
        vals.append(fp[i])
    return _patchify_latent(vals^, 1, 32, 32, ctx, out_std, out_mean)


# ── Save an EMA-shadow LoRA sibling (host F32 ema_a/ema_b -> PEFT safetensors). ─
def _save_ema_sibling(
    lora: Wan22LoraSet, ema_a: List[List[Float32]], ema_b: List[List[Float32]],
    path: String, ctx: DeviceContext,
) raises -> Int:
    var ad = List[LoraAdapter]()
    for i in range(len(lora.ad)):
        var na = len(ema_a[i])
        var nb = len(ema_b[i])
        var za = List[Float32]()
        var va = List[Float32]()
        for _ in range(na):
            za.append(Float32(0.0))
            va.append(Float32(0.0))
        var zb = List[Float32]()
        var vb = List[Float32]()
        for _ in range(nb):
            zb.append(Float32(0.0))
            vb.append(Float32(0.0))
        ad.append(LoraAdapter(
            ema_a[i].copy(), ema_b[i].copy(),
            lora.ad[i].rank, lora.ad[i].in_f, lora.ad[i].out_f, lora.ad[i].scale,
            za^, va^, zb^, vb^,
        ))
    var ema_set = Wan22LoraSet(ad^, lora.num_blocks, lora.rank)
    return save_wan22_lora(ema_set, path, ctx)


# ══════════════════════════════════════════════════════════════════════════════
# The step loop, comptime-monomorphized on the packed sequence length SEQ.
# SEQ = S_TGT + sum(conditioning tokens). The TARGET region is ALWAYS the trailing
# S_TGT tokens (offset = SEQ - S_TGT; = 0 when there is no conditioning).
# cond_tokens is empty in no-cond mode, so the prepend is a no-op.
#
# Timestep policy: overfit-pin (pin_sigma fixed, from the TASK's Bernini sampler)
# for a crisp 1-sample monotone drop, or per-step Bernini re-draw when stochastic.
# EMA (decay=ema_decay) shadows the LoRA params each step; saved as a sibling.
# ══════════════════════════════════════════════════════════════════════════════
def _train_loop[SEQ: Int](
    base: Wan22StackBase, mut loader: TurboPlannedLoader, mut lora: Wan22LoraSet,
    cos_h: List[Float32], sin_h: List[Float32],
    cond_tokens: List[Float32], txt_tokens: List[Float32],
    cache_dir: String, use_cache: Bool,
    steps: Int, seed: UInt64, stochastic: Bool,
    is_mode: Bool, shift: Float64, win: BerniniWindow, pin_sigma: Float32,
    lr: Float32, max_grad_norm: Float32,
    beta1: Float32, beta2: Float32, opt_eps: Float32, weight_decay: Float32,
    ema_decay: Float32, out_path: String, no_cond: Bool, ctx: DeviceContext,
) raises:
    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)
    var min_loss = Float32(0.0)
    var have_first = False
    var train_start = perf_counter_ns()

    # ── EMA shadow init (host F32, mirrors each adapter's live BF16 a/b) ─────────
    var ema_a = List[List[Float32]]()
    var ema_b = List[List[Float32]]()
    for i in range(len(lora.ad)):
        var a = List[Float32]()
        for j in range(len(lora.ad[i].a)):
            a.append(Float32(lora.ad[i].a[j]))
        ema_a.append(a^)
        var b = List[Float32]()
        for j in range(len(lora.ad[i].b)):
            b.append(Float32(lora.ad[i].b[j]))
        ema_b.append(b^)

    print("")
    print("step  sigma     target_MSE       grad_norm    grad_absum    sec")
    for step in range(steps):
        var t0 = perf_counter_ns()

        # timestep sigma: overfit pin (fixed pair) or Bernini per-step re-draw.
        var t = pin_sigma
        if stochastic:
            t = bernini_sample_sigma(
                is_mode, shift, seed + UInt64(step) * UInt64(2654435761) + 1,
                LOGIT_MEAN, LOGIT_STD, MODE_SCALE, win,
            )

        # ── TARGET clean latent x0 [S_TGT, IN_CH] ───────────────────────────────
        var tgt_std = Float32(0.0)
        var tgt_mean = Float32(0.0)
        var x0: List[Float32]
        if use_cache:
            x0 = _load_cache_target(cache_dir, 0, ctx, tgt_std, tgt_mean)
        else:
            var tl = _synth(16 * 1 * 32 * 32, seed * 13 + 1)
            x0 = _patchify_latent(tl^, 1, 32, 32, ctx, tgt_std, tgt_mean)
        if len(x0) != S_TGT * IN_CH:
            raise Error("target patch len != S_TGT*IN_CH")
        if step == 0:
            print("  [data] target latent std=", tgt_std, " mean=", tgt_mean)

        # ── flow-match noise on the TARGET only (conditioning is clean) ──────────
        var noise_seed = seed * UInt64(2000003) + 3
        if stochastic:
            noise_seed = seed * UInt64(2000003) + UInt64(step) * 104729 + 3
        var noise = _noise(S_TGT * IN_CH, noise_seed)

        var tgt_in = List[Float32]()      # noised target [S_TGT, IN_CH]
        var target = List[Float32]()      # velocity target = noise - x0
        for i in range(len(x0)):
            tgt_in.append((Float32(1.0) - t) * x0[i] + t * noise[i])
            target.append(noise[i] - x0[i])

        # ── ASSEMBLE the packed model input [cond_1..N (clean) | target(noised)] ─
        var model_in = List[Float32]()
        for i in range(len(cond_tokens)):     # empty in no-cond mode
            model_in.append(cond_tokens[i])
        for i in range(len(tgt_in)):
            model_in.append(tgt_in[i])
        if len(model_in) != SEQ * IN_CH:
            raise Error("packed model_in len != SEQ*IN_CH")

        var t_model = t * Float32(1000.0) + Float32(1.0)

        # ── FORWARD over the PACKED sequence (existing certified stack) ──────────
        var fwd = wan22_stack_lora_forward_offload[H, Dh, SEQ, TXT](
            model_in.copy(), txt_tokens.copy(), t_model,
            base, loader, lora, cos_h.copy(), sin_h.copy(),
            DIM, FFN, IN_CH, TEXT_DIM, OUT_CH, FREQ_DIM, EPS, ctx,
        )

        # ── velocity-MSE on the TARGET region ONLY ──────────────────────────────
        var nout = len(fwd.out)                     # SEQ * OUT_CH
        var tgt_off = (SEQ - S_TGT) * OUT_CH
        var ntgt = S_TGT * OUT_CH
        var inv_n = Float32(2.0) / Float32(ntgt)
        var d_out = List[Float32]()
        for _ in range(tgt_off):
            d_out.append(Float32(0.0))              # conditioning region: no grad
        var loss = Float32(0.0)
        for i in range(ntgt):
            var pred = fwd.out[tgt_off + i]
            var diff = pred - target[i]
            loss += diff * diff
            d_out.append(inv_n * diff)
        loss = loss / Float32(ntgt)
        if len(d_out) != nout:
            raise Error("d_out len != stack out len")
        if not have_first:
            first_loss = loss
            min_loss = loss
            have_first = True
        if loss < min_loss:
            min_loss = loss
        last_loss = loss

        # ── BACKWARD over the whole packed sequence; LoRA grads only ─────────────
        var grads = wan22_stack_lora_backward_offload[H, Dh, SEQ, TXT](
            d_out, model_in.copy(), txt_tokens.copy(),
            base, loader, lora, cos_h.copy(), sin_h.copy(), fwd,
            DIM, FFN, IN_CH, TEXT_DIM, OUT_CH, FREQ_DIM, EPS, ctx,
        )
        var absum = _grad_abs_sum(grads)
        var gn = _clip(grads, max_grad_norm)
        wan22_lora_adamw_step(lora, grads, step + 1, lr, ctx,
                              beta1, beta2, opt_eps, weight_decay)

        # ── EMA shadow: ema = decay*ema + (1-decay)*live (host F32) ──────────────
        var one_m = Float32(1.0) - ema_decay
        for i in range(len(lora.ad)):
            for j in range(len(lora.ad[i].a)):
                ema_a[i][j] = ema_decay * ema_a[i][j] + one_m * Float32(lora.ad[i].a[j])
            for j in range(len(lora.ad[i].b)):
                ema_b[i][j] = ema_decay * ema_b[i][j] + one_m * Float32(lora.ad[i].b[j])

        var secs = Float64(perf_counter_ns() - t0) / 1.0e9
        print(step, "  ", t, "  ", loss, "  ", gn, "  ", absum, "  ", secs)
        if grads.nonfinite_lora_grads != 0:
            print("  !! nonfinite lora grads =", grads.nonfinite_lora_grads)

    var npairs = save_wan22_lora(lora, out_path, ctx)
    var meta = List[Float32]()
    meta.append(Float32(steps))
    meta.append(Float32(Int(seed)))
    var state_path = out_path + String(".state")
    var nstate = save_wan22_lora_state(lora, state_path, ctx, meta^)
    var ema_path = out_path + String(".ema.safetensors")
    var nema = _save_ema_sibling(lora, ema_a, ema_b, ema_path, ctx)
    print("")
    print("[save] wrote", npairs, "PEFT pairs ->", out_path)
    print("[save] wrote", nstate, "state tensors ->", state_path)
    print("[save] wrote", nema, "EMA (decay=", ema_decay, ") PEFT pairs ->", ema_path)
    print("trained steps=", steps, " first target_MSE=", first_loss,
          " last=", last_loss, " min=", min_loss,
          " total sec=", Float64(perf_counter_ns() - train_start) / 1.0e9)
    if no_cond:
        print("RESULT: bernini-r NO-COND (Tier-1 T2V; packed==target) ran.")
    else:
        print("RESULT: bernini-r CONDITIONED (packed [cond_1..N | target], src-id rope,",
              "velocity-MSE on target region only; real fp8 base) ran.")


def main() raises:
    var ctx = DeviceContext()

    var args = argv()
    var config_path = String(DEFAULT_CONFIG)
    if len(args) > 1:
        config_path = String(args[1])
    if not _file_exists(config_path):
        raise Error(String("config not found: ") + config_path)
    var cfg = read_model_config(config_path)

    # ── task selection ──────────────────────────────────────────────────────────
    var task = _env_str(String("BERNINI_TASK"))
    if task.byte_length() == 0:
        task = String("t2v")
    var recipe = bernini_recipe_for(task)

    # regression switch: BERNINI_NO_COND=1 -> force n_cond=0 (Tier-1) for any task.
    var no_cond = _env_is_set(String("BERNINI_NO_COND")) or recipe.n_cond == 0
    var stochastic = _env_is_set(String("BERNINI_STOCHASTIC"))

    var rank = cfg.lora_rank if cfg.lora_rank > 0 else 16
    var alpha = cfg.lora_alpha if cfg.lora_alpha > Float32(0.0) else Float32(16.0)
    var lr = cfg.lr if cfg.lr > Float32(0.0) else DEFAULT_LR
    var max_grad_norm = cfg.max_grad_norm if cfg.max_grad_norm > Float32(0.0) \
        else DEFAULT_MAX_GRAD_NORM
    var steps = cfg.max_steps if cfg.max_steps > 0 else 8
    var seed = cfg.seed
    var beta1 = cfg.beta1 if cfg.beta1 > Float32(0.0) else Float32(0.9)
    var beta2 = cfg.beta2 if cfg.beta2 > Float32(0.0) else Float32(0.999)
    var opt_eps = cfg.eps if cfg.eps > Float32(0.0) else Float32(1.0e-8)
    var weight_decay = cfg.weight_decay if cfg.weight_decay >= Float32(0.0) \
        else Float32(0.01)

    var ema_decay = DEFAULT_EMA_DECAY

    var ckpt = cfg.checkpoint if cfg.checkpoint.byte_length() > 0 \
        else String(DEFAULT_CKPT_LOW)

    var out_path = cfg.output_model_destination
    if out_path.byte_length() == 0:
        out_path = String(
            "/home/alex/mojodiffusion/output/bernini_r_lora/bernini_r_cond_lora.safetensors"
        )

    # ── recipe -> shift / weighting / window ────────────────────────────────────
    var shift = recipe.shift
    var is_mode = recipe.is_mode
    var win = bernini_task_window(shift, NOISE_TMIN, NOISE_TMAX)
    # overfit-pin sigma: one deterministic Bernini draw for the whole run.
    var pin_sigma = bernini_sample_sigma(
        is_mode, shift, seed, LOGIT_MEAN, LOGIT_STD, MODE_SCALE, win
    )

    # ── conditioning segments (skip when no_cond) ───────────────────────────────
    var segs = List[BerniniCondSeg]()
    if not no_cond:
        segs = bernini_smoke_cond_segments(task)
    var n_cond_tokens = 0
    for i in range(len(segs)):
        n_cond_tokens += segs[i].f * segs[i].h * segs[i].w
    var seq_len = S_TGT + n_cond_tokens

    print("==== Bernini-R CONDITIONED LoRA trainer (12-task renderer) ====")
    print("TASK:", task, "  shift=", shift,
          "  weighting=", ("mode(video)" if is_mode else "logit_normal(image)"),
          "  n_cond=", len(segs), "(", recipe.cond_kind, ")")
    print("system_prompt:", recipe.system_prompt)
    print("timestep window (u in [tmin,tmax]): tmin=", win.tmin, " tmax=", win.tmax,
          "  pin_sigma=", pin_sigma)
    if no_cond:
        print("MODE: NO-COND (Tier-1 T2V; packed == target, src_id=0 == stock wan rope)")
    else:
        print("MODE: CONDITIONED  packed order [cond_1..N | target] (pack_vae_latents)")
    for i in range(len(segs)):
        print("  cond seg", i, "src_id=", i + 1, " grid(f,h,w)=(", segs[i].f, ",",
              segs[i].h, ",", segs[i].w, ") kind=", segs[i].kind,
              " tokens=", segs[i].f * segs[i].h * segs[i].w)
    print("packed sequence: cond=", n_cond_tokens, " + target=", S_TGT, " = S=", seq_len)
    print("arch: dim=", DIM, " blocks=", NUM_BLOCKS, " heads=", H, " head_dim=", Dh,
          " ffn=", FFN, " in_ch=", IN_CH, " out_ch=", OUT_CH, " TXT=", TXT)
    print("lora: rank=", rank, " alpha=", alpha, " (10/block: 8 attn + ffn.0 + ffn.2)")
    print("optim: lr=", lr, " max_grad_norm=", max_grad_norm, " ema_decay=", ema_decay)
    if stochastic:
        print("timestep/noise: STOCHASTIC Bernini per-step (sigma re-drawn each step)")
    else:
        print("timestep/noise: OVERFIT PIN (fixed sigma=pin_sigma + fixed noise)")
    print("steps:", steps, " seed:", seed)
    print("ckpt (low):", ckpt)

    # ── checkpoint-absent guard ────────────────────────────────────────────────
    if not _file_exists(_base_file_for(ckpt)):
        print("")
        print("[bernini-r] fp8 base not present, skipping real step:")
        print("        ", ckpt)
        print("        (recipe/geometry/packing + the S=", seq_len,
              "monomorphization type-checked; re-run once the fp8 cache lands.)")
        print("RESULT: bernini-r cond trainer wired OK (weights-absent path, exit 0)")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # REAL WEIGHTED STEP — build resident base + block-swap loader (single low expert).
    # ══════════════════════════════════════════════════════════════════════════
    print("")
    print("[bernini-r] fp8 base present — building resident base + block-swap loader")
    var off_cfg = OffloadConfig.synchronous_single()
    var base_st = SafeTensors.open(_base_file_for(ckpt))
    var ckpt_prefix = detect_wan22_prefix(base_st)
    print("[bernini-r] detected checkpoint key prefix: '", ckpt_prefix, "'", sep="")
    var base = load_wan22_stack_base(base_st, ctx, ckpt_prefix)
    var plan = build_wan22_block_plan(NUM_BLOCKS, ckpt_prefix)
    var loader = TurboPlannedLoader.open(ckpt, plan^, off_cfg, ctx, False)

    var lora = build_wan22_lora_set(NUM_BLOCKS, DIM, FFN, rank, alpha)
    print("[lora] adapters:", wan22_total_adapters(lora))

    # ── build the PACKED source-id RoPE for [cond_1..N (src 1..N) | target (0)] ──
    var rope_segments = List[BerniniRopeSegment]()
    for i in range(len(segs)):
        rope_segments.append(
            BerniniRopeSegment(segs[i].f, segs[i].h, segs[i].w, Float32(i + 1))
        )
    rope_segments.append(BerniniRopeSegment(TGT_FG, TGT_HG, TGT_WG, TGT_SRC_ID))
    var rope = build_bernini_src_id_rope(rope_segments, Dh, ROPE_THETA, STDtype.F32, ctx)
    var cos_h = rope[0].to_host(ctx)   # [seq_len * Dh/2]
    var sin_h = rope[1].to_host(ctx)
    var expect_rope = seq_len * (Dh // 2)
    if len(cos_h) != expect_rope or len(sin_h) != expect_rope:
        raise Error("packed rope length != seq_len*Dh/2 — segment/geometry mismatch")
    print("[rope] packed src-id rope built: segments=", len(rope_segments),
          " rows=", seq_len, " cols(Dh/2)=", Dh // 2)

    # ── conditioning tokens: N CLEAN patchified latents concatenated in order ────
    var cond_tokens = List[Float32]()
    for i in range(len(segs)):
        var seg = segs[i].copy()
        var cs = Float32(0.0)
        var cm = Float32(0.0)
        var cl = _synth(16 * seg.f * (seg.h * 2) * (seg.w * 2),
                        seed * 7 + UInt64(101 + i * 13))
        var toks = _patchify_latent(cl^, seg.f, seg.h * 2, seg.w * 2, ctx, cs, cm)
        var want = seg.f * seg.h * seg.w * IN_CH
        if len(toks) != want:
            raise Error("conditioning seg patch len mismatch")
        for j in range(len(toks)):
            cond_tokens.append(toks[j])
        print("[data] cond seg", i, "(clean, src_id=", i + 1, "): std=", cs,
              " mean=", cm, " tokens=", seg.f * seg.h * seg.w)
    if len(cond_tokens) != n_cond_tokens * IN_CH:
        raise Error("total cond tokens mismatch")

    # ── data: real cached TARGET latent when present, else synthetic ─────────────
    var cache_dir = cfg.dataset_cache_dir
    var use_cache = False
    if cache_dir.byte_length() > 0 and _file_exists(
        cache_dir + String("/sample_0.safetensors")
    ):
        use_cache = True
    if use_cache:
        print("[data] REAL cached target latent:", cache_dir,
              " (synthetic conditioning + synthetic text)")
    else:
        print("[data] SYNTHETIC target/conditioning/text (no cache_dir sample_0)")

    var txt_tokens = _synth(TXT * TEXT_DIM, seed * 777 + 9)

    # ── comptime dispatch on the packed sequence length ──────────────────────────
    if seq_len == 256:
        _train_loop[256](base, loader, lora, cos_h, sin_h, cond_tokens, txt_tokens,
            cache_dir, use_cache, steps, seed, stochastic, is_mode, shift, win,
            pin_sigma, lr, max_grad_norm, beta1, beta2, opt_eps, weight_decay,
            ema_decay, out_path, no_cond, ctx)
    elif seq_len == 320:
        _train_loop[320](base, loader, lora, cos_h, sin_h, cond_tokens, txt_tokens,
            cache_dir, use_cache, steps, seed, stochastic, is_mode, shift, win,
            pin_sigma, lr, max_grad_norm, beta1, beta2, opt_eps, weight_decay,
            ema_decay, out_path, no_cond, ctx)
    elif seq_len == 384:
        _train_loop[384](base, loader, lora, cos_h, sin_h, cond_tokens, txt_tokens,
            cache_dir, use_cache, steps, seed, stochastic, is_mode, shift, win,
            pin_sigma, lr, max_grad_norm, beta1, beta2, opt_eps, weight_decay,
            ema_decay, out_path, no_cond, ctx)
    elif seq_len == 448:
        _train_loop[448](base, loader, lora, cos_h, sin_h, cond_tokens, txt_tokens,
            cache_dir, use_cache, steps, seed, stochastic, is_mode, shift, win,
            pin_sigma, lr, max_grad_norm, beta1, beta2, opt_eps, weight_decay,
            ema_decay, out_path, no_cond, ctx)
    elif seq_len == 512:
        _train_loop[512](base, loader, lora, cos_h, sin_h, cond_tokens, txt_tokens,
            cache_dir, use_cache, steps, seed, stochastic, is_mode, shift, win,
            pin_sigma, lr, max_grad_norm, beta1, beta2, opt_eps, weight_decay,
            ema_decay, out_path, no_cond, ctx)
    else:
        raise Error(String("unsupported packed seq_len ") + String(seq_len) +
                    String(" (task ") + task + String("); add a monomorphization"))
