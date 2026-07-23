# serenitymojo/training/train_bernini_r_cond.mojo
#
# BERNINI-R Tier-2b: the reference-CONDITIONING training path.
# Pure Mojo + MAX, GPU, block-swap offload. Sequence-concat of CLEAN conditioning
# latents (each with its own source-id RoPE) AHEAD of the noised TARGET, full
# self-attention over the packed sequence, velocity-MSE loss on the TARGET region
# ONLY. This completes the reference-guided renderer.
#
# ── This file is the ASSEMBLY + LOOP only. It reimplements NO block/stack math ──
# It REUSES verbatim:
#   * the certified Wan2.2-A14B LoRA engine
#     (models/wan22/wan22_stack_lora.mojo::wan22_stack_lora_forward_offload /
#      _backward_offload / wan22_lora_adamw_step / save_wan22_lora) — the block
#     runs over WHATEVER sequence + cos/sin it is given, so a packed
#     [conditioning | target] sequence needs no new kernel;
#   * the Tier-2a source-id RoPE
#     (models/wan22/bernini_src_id_rope.mojo::build_bernini_src_id_rope) — builds
#     the per-segment src-id rope tables concatenated along the token axis;
#   * the proven patchify (ops/patchify3d) and safetensors reader.
#
# ── Bernini packing ORDER (matched; cite Bernini/bernini/training/data.py) ─────
#   encode_renderer_messages (:171-190) walks the conversation IN ORDER, appending
#   a vae entry per image/video message; `has_loss` marks the GENERATION target.
#   pack_vae_latents (:255-278) then, for each entry in that order:
#     source_id = 0 if vae_mask(=target) else idx+1
#     - conditioning (vae_mask False): patchify the CLEAN latent, source_id=idx+1,
#       NO noise, NO target_velocity.
#     - target (vae_mask True): x_t=(1-σ)x0+σ·noise, source_id=0, target=noise-x0.
#   and cats input_vae_latents / input_vae_rope in that SAME order. In a renderer
#   sample the reference messages precede the generation message, so the packed
#   order is  [conditioning refs (src_id 1,2,…, clean) | target (src_id 0, noised)]
#   with the target the trailing contiguous region. vae_latents_mask (:270) marks
#   exactly those trailing target tokens → the loss slice here.
#
#   The whole packed sample shares ONE timestep (data.py:365 noise_timestep is a
#   scalar; transformer_wan.py:551-557 expands ONE temb over every token) — so the
#   existing stack, which applies one t_model's AdaLN to all S tokens, matches
#   Bernini bit-for-bit; the conditioning tokens receive the target's timestep
#   modulation exactly as Bernini does.
#
# ── Regression to Tier-1 T2V ───────────────────────────────────────────────────
#   With ZERO conditioning segments the packed sequence is just the target
#   (source_id=0). build_bernini_src_id_rope over one src_id=0 segment is
#   BIT-IDENTICAL to wan22_build_rope (Tier-2a identity property), S_packed =
#   S_target, and the loop is byte-for-byte the Tier-1 path. Env BERNINI_NO_COND=1
#   exercises this regression from the same binary.
#
# ── BUILD / RUN ────────────────────────────────────────────────────────────────
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/training/train_bernini_r_cond.mojo [config.json]
#   Default config: serenitymojo/configs/bernini_r_t2v_cond_smoke.json
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

# ── Packed-sequence smoke geometry ─────────────────────────────────────────────
# TARGET latent [16,1,32,32] -> patchify(1,2,2) -> grid (1,16,16) -> S_TGT=256.
# One CONDITIONING reference latent [16,1,16,16] -> grid (1,8,8) -> S_COND=64,
# source_id=1 (a distinct, smaller reference resolution — exercises fractional-
# capable src-id rope with a different grid AND a nonzero id). Both patchify to
# IN_CH=64 so the SAME [5120,64] patch-embed serves both.
# Packed order [cond | target]:  S_PACKED = S_COND + S_TGT.
comptime TGT_FG = 1
comptime TGT_HG = 16
comptime TGT_WG = 16
comptime S_TGT = TGT_FG * TGT_HG * TGT_WG          # 256 target tokens

comptime COND_FG = 1
comptime COND_HG = 8
comptime COND_WG = 8
comptime S_COND = COND_FG * COND_HG * COND_WG      # 64 conditioning tokens

comptime S_PACKED = S_COND + S_TGT                 # 320 packed tokens
comptime S_TGT_ONLY = S_TGT                        # regression (no-cond) length
comptime TXT = 512                                 # umt5 cross-attn kv length

comptime COND_SRC_ID = Float32(1.0)                # first (only) reference id
comptime TGT_SRC_ID = Float32(0.0)                 # target == stock wan rope

# ── optimizer / recipe defaults (config overrides where present) ───────────────
comptime DEFAULT_LR = Float32(2.0e-4)
comptime DEFAULT_MAX_GRAD_NORM = Float32(1.0)
comptime DEFAULT_DISCRETE_FLOW_SHIFT = Float32(3.0)
comptime DEFAULT_SIGMOID_SCALE = Float32(1.0)
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


# Resident-base file for an fp8 per-block CACHE DIR (shared.safetensors sibling)
# vs a monolithic checkpoint (returned unchanged). Same rule as train_wan22_real.
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


def _sigmoid(x: Float32) -> Float32:
    return Float32(1.0) / (Float32(1.0) + exp(-x))


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


# musubi timestep sampler (both modes) — identical math to train_wan22_real.
def _sample_t(
    shift_mode: Bool, step: Int, seed: UInt64,
    sigmoid_scale: Float32, discrete_flow_shift: Float32,
) -> Float32:
    var base = seed * UInt64(6364136223846793005) + UInt64(step) * UInt64(7919) + 17
    if not shift_mode:
        var t = _u01(base)
        t = Float32(1.0) - t
        return t
    var z = _randn(base)
    var t = _sigmoid(z * sigmoid_scale)
    var num = t * discrete_flow_shift
    var den = Float32(1.0) + (discrete_flow_shift - Float32(1.0)) * t
    return num / den


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


# sum |d_A|+|d_B| over all adapters (nonzero-grad witness).
def _grad_abs_sum(grads: Wan22LoraGradSet) -> Float32:
    var s = Float32(0.0)
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            s += abs(grads.d_a[i][j])
        for j in range(len(grads.d_b[i])):
            s += abs(grads.d_b[i][j])
    return s


# ── Load a CLEAN latent [16,1,fh,fw] and patchify(1,2,2) -> [S, IN_CH] flat. ──
# Used for the real cached target latent. Returns (flat [S*IN_CH], std, mean).
def _patchify_latent(
    vals: List[Float32], fh: Int, fw: Int, ctx: DeviceContext,
    mut out_std: Float32, mut out_mean: Float32,
) raises -> List[Float32]:
    var n = 16 * 1 * fh * fw
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
    shp.append(1)
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
    return _patchify_latent(vals^, 32, 32, ctx, out_std, out_mean)


# ══════════════════════════════════════════════════════════════════════════════
# The step loop, comptime-monomorphized on the packed sequence length SEQ.
# SEQ = S_PACKED (cond+target) in conditioned mode, or S_TGT_ONLY in the no-cond
# regression. The TARGET region is ALWAYS the trailing S_TGT tokens
# (offset = SEQ - S_TGT; = 0 when there is no conditioning). cond_tokens is empty
# in no-cond mode, so the prepend is a no-op and model_in is exactly SEQ*IN_CH.
# ══════════════════════════════════════════════════════════════════════════════
def _train_loop[SEQ: Int](
    base: Wan22StackBase, mut loader: TurboPlannedLoader, mut lora: Wan22LoraSet,
    cos_h: List[Float32], sin_h: List[Float32],
    cond_tokens: List[Float32], txt_tokens: List[Float32],
    cache_dir: String, use_cache: Bool,
    steps: Int, seed: UInt64, stochastic: Bool, shift_mode: Bool,
    discrete_flow_shift: Float32, lr: Float32, max_grad_norm: Float32,
    beta1: Float32, beta2: Float32, opt_eps: Float32, weight_decay: Float32,
    out_path: String, no_cond: Bool, ctx: DeviceContext,
) raises:
    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)
    var min_loss = Float32(0.0)
    var have_first = False
    var train_start = perf_counter_ns()

    print("")
    print("step  t        target_MSE       grad_norm    grad_absum    sec")
    for step in range(steps):
        var t0 = perf_counter_ns()

        # timestep: overfit pin (fixed) or musubi sampler.
        var t = Float32(0.5)
        if stochastic:
            t = _sample_t(shift_mode, step, seed,
                          DEFAULT_SIGMOID_SCALE, discrete_flow_shift)

        # ── TARGET clean latent x0 [S_TGT, IN_CH] ───────────────────────────────
        var tgt_std = Float32(0.0)
        var tgt_mean = Float32(0.0)
        var x0: List[Float32]
        if use_cache:
            x0 = _load_cache_target(cache_dir, 0, ctx, tgt_std, tgt_mean)
        else:
            var tl = _synth(16 * 1 * 32 * 32, seed * 13 + 1)
            x0 = _patchify_latent(tl^, 32, 32, ctx, tgt_std, tgt_mean)
        if len(x0) != S_TGT * IN_CH:
            raise Error("target patch len != S_TGT*IN_CH")
        if step == 0:
            print("  [data] target latent std=", tgt_std, " mean=", tgt_mean,
                  " (round-trip gate; must match torch)")

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

        # ── ASSEMBLE the packed model input [cond(clean) | target(noised)] ───────
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
    print("")
    print("[save] wrote", npairs, "PEFT pairs ->", out_path)
    print("[save] wrote", nstate, "state tensors ->", state_path)
    print("trained steps=", steps, " first target_MSE=", first_loss,
          " last=", last_loss, " min=", min_loss,
          " total sec=", Float64(perf_counter_ns() - train_start) / 1.0e9)
    if no_cond:
        print("RESULT: bernini-r NO-COND regression ran (Tier-1 T2V path; packed==target).")
    else:
        print("RESULT: bernini-r Tier-2b CONDITIONED LoRA trainer ran (packed [cond|target],",
              "src-id rope, velocity-MSE on target region only; real fp8 base).")


def main() raises:
    var ctx = DeviceContext()

    var args = argv()
    var config_path = String(DEFAULT_CONFIG)
    if len(args) > 1:
        config_path = String(args[1])
    if not _file_exists(config_path):
        raise Error(String("config not found: ") + config_path)
    var cfg = read_model_config(config_path)

    # regression switch: BERNINI_NO_COND=1 -> zero conditioning segments (Tier-1).
    var no_cond = _env_is_set(String("BERNINI_NO_COND"))
    # overfit smoke pins (default ON for this smoke trainer): fixed t + fixed noise
    # so a 1-sample loop makes an identical (x_t, target) pair every step and the
    # target-region MSE must fall monotonically. BERNINI_STOCHASTIC=1 -> musubi.
    var stochastic = _env_is_set(String("BERNINI_STOCHASTIC"))
    var shift_mode = _env_is_set(String("WAN22_SHIFT_TIMESTEPS"))

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
    var discrete_flow_shift = cfg.timestep_shift if cfg.timestep_shift > Float32(0.0) \
        else DEFAULT_DISCRETE_FLOW_SHIFT

    var ckpt = cfg.checkpoint if cfg.checkpoint.byte_length() > 0 \
        else String(DEFAULT_CKPT_LOW)

    var out_path = cfg.output_model_destination
    if out_path.byte_length() == 0:
        out_path = String(
            "/home/alex/mojodiffusion/output/bernini_r_lora/bernini_r_cond_lora.safetensors"
        )

    var seq_len = S_TGT_ONLY if no_cond else S_PACKED

    print("==== Bernini-R Tier-2b CONDITIONED LoRA trainer (reference-guided renderer) ====")
    if no_cond:
        print("MODE: NO-COND regression (Tier-1 T2V; packed == target, src_id=0 == stock wan rope)")
        print("packed sequence: [target(", S_TGT, ", src_id=0)]  S=", seq_len)
    else:
        print("MODE: CONDITIONED  packed order [cond | target] (Bernini pack_vae_latents)")
        print("packed sequence: [cond(", S_COND, ", src_id=", COND_SRC_ID,
              ", CLEAN) | target(", S_TGT, ", src_id=", TGT_SRC_ID, ", noised)]  S=", seq_len)
    print("arch: dim=", DIM, " blocks=", NUM_BLOCKS, " heads=", H, " head_dim=", Dh,
          " ffn=", FFN, " in_ch=", IN_CH, " out_ch=", OUT_CH, " TXT=", TXT)
    print("lora: rank=", rank, " alpha=", alpha, " (10/block: 8 attn + ffn.0 + ffn.2)")
    print("optim: lr=", lr, " max_grad_norm=", max_grad_norm, " betas=(", beta1, ",", beta2,
          ") eps=", opt_eps, " wd=", weight_decay)
    if stochastic:
        print("timestep/noise: STOCHASTIC musubi (per-step t + noise)")
    else:
        print("timestep/noise: OVERFIT PIN (fixed t=0.5 + fixed noise; identical pair/step)")
    print("steps:", steps, " seed:", seed)
    print("ckpt (low):", ckpt)

    # ── checkpoint-absent guard ────────────────────────────────────────────────
    if not _file_exists(_base_file_for(ckpt)):
        print("")
        print("[bernini-r] fp8 base not present, skipping real step:")
        print("        ", ckpt)
        print("        (config/geometry/packing + the S=", seq_len,
              "monomorphization type-checked; re-run once the fp8 cache lands.)")
        print("RESULT: bernini-r cond trainer wired OK (weights-absent path, exit 0)")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # REAL WEIGHTED STEP — build resident base + block-swap loader (single low expert).
    # Dual high/low switching is orthogonal and already certified in Tier-1; the
    # conditioned smoke pins the LOW expert for a crisp overfit.
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

    # ── build the PACKED source-id RoPE for the segments [cond | target] ────────
    # Each segment carries its own src-id rope (Tier-2a); concatenated along tokens
    # in packing order. Target src_id=0 => that region is bit-identical to stock
    # wan rope. NO-COND regression => single src_id=0 segment == Tier-1 rope.
    var segments = List[BerniniRopeSegment]()
    if not no_cond:
        segments.append(BerniniRopeSegment(COND_FG, COND_HG, COND_WG, COND_SRC_ID))
    segments.append(BerniniRopeSegment(TGT_FG, TGT_HG, TGT_WG, TGT_SRC_ID))
    var rope = build_bernini_src_id_rope(segments, Dh, ROPE_THETA, STDtype.F32, ctx)
    var cos_h = rope[0].to_host(ctx)   # [S_PACKED * Dh/2]
    var sin_h = rope[1].to_host(ctx)   # [S_PACKED * Dh/2]
    var expect_rope = seq_len * (Dh // 2)
    if len(cos_h) != expect_rope or len(sin_h) != expect_rope:
        raise Error("packed rope length != seq_len*Dh/2 — segment/geometry mismatch")
    print("[rope] packed src-id rope built: segments=", len(segments),
          " rows=", seq_len, " cols(Dh/2)=", Dh // 2)

    # ── data: real cached TARGET latent (round-trip gate) + synthetic CLEAN cond +
    #    synthetic text. Conditioning latent is NEVER noised.
    var cache_dir = cfg.dataset_cache_dir
    var use_cache = False
    if cache_dir.byte_length() > 0 and _file_exists(
        cache_dir + String("/sample_0.safetensors")
    ):
        use_cache = True
    if use_cache:
        print("[data] REAL cached target latent:", cache_dir,
              " + synthetic CLEAN conditioning + synthetic text")
    else:
        print("[data] SYNTHETIC target/conditioning/text (no cache_dir sample_0)")

    # Fixed CLEAN conditioning latent [16,1,16,16] -> patchify -> [S_COND, IN_CH].
    var cond_std = Float32(0.0)
    var cond_mean = Float32(0.0)
    var cond_tokens = List[Float32]()
    if not no_cond:
        var cond_latent = _synth(16 * 1 * COND_HG * 2 * COND_WG * 2, seed * 7 + 101)
        cond_tokens = _patchify_latent(
            cond_latent^, COND_HG * 2, COND_WG * 2, ctx, cond_std, cond_mean
        )
        if len(cond_tokens) != S_COND * IN_CH:
            raise Error("conditioning patch len != S_COND*IN_CH")
        print("[data] conditioning (clean, src_id=", COND_SRC_ID, "): std=", cond_std,
              " mean=", cond_mean, " tokens=", S_COND)

    # synthetic text (Tier-2b conditions on the VAE references; text kept synthetic).
    var txt_tokens = _synth(TXT * TEXT_DIM, seed * 777 + 9)

    # ── comptime dispatch on the packed sequence length ──────────────────────────
    if no_cond:
        _train_loop[S_TGT_ONLY](
            base, loader, lora, cos_h, sin_h, cond_tokens, txt_tokens,
            cache_dir, use_cache, steps, seed, stochastic, shift_mode,
            discrete_flow_shift, lr, max_grad_norm, beta1, beta2, opt_eps,
            weight_decay, out_path, no_cond, ctx,
        )
    else:
        _train_loop[S_PACKED](
            base, loader, lora, cos_h, sin_h, cond_tokens, txt_tokens,
            cache_dir, use_cache, steps, seed, stochastic, shift_mode,
            discrete_flow_shift, lr, max_grad_norm, beta1, beta2, opt_eps,
            weight_decay, out_path, no_cond, ctx,
        )
