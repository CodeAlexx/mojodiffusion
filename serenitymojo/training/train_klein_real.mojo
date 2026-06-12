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
#   PRINT (human display, one per completed step):
#     [Klein-lora] step k/total | epoch e/E | loss ... | grad_norm ... | ...s/step | elapsed ... | ETA ...
#   Optional machine `PROG` output stays behind MACHINE_PROGRESS_LOG=False.
#
# Cadence (Klein production):
#   Production cadence is driven by train_klein_cadence.mojo. This file is the
#   worker: it trains a global step range, saves PEFT LoRA at the end, and exits.
#   Keeping sampler and trainer in separate processes is required for Klein 9B
#   1024px validation because one Mojo process can otherwise retain CUDA memory
#   from the training stack/scratch slabs while the sampler tries to load the
#   inference stack.
#
# MEMORY: this process streams Klein-9B transformer blocks through the turbo
# loader and keeps only shared projections, LoRA adapters, cached training
# tensors, and scratch slabs resident. It does NOT import Qwen3Encoder (the
# ~16 GB encoder ran in klein_prepare_alina.mojo, a separate process that
# already exited). Production validation runs through train_klein_cadence.mojo
# so the sampler gets a fresh process and the VAE never co-resides with the
# training stack.
#
# Run (2-step timed dry run — the lead's launch decision is from this number):
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/training/train_klein_real.mojo

from std.sys import argv
from std.collections import List, Optional
from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin, exp as fexp
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.cap_cache import validate_klein_cap_cache_header
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.scratch_ring import ScratchRingAllocator

from serenitymojo.models.klein.klein_stack_lora import (
    KleinLoraSet, KleinLoraGrads, KleinLoraDeviceSet, build_klein_lora_set,
    klein_lora_set_to_device,
    klein_stack_lora_forward, klein_stack_lora_forward_device_inputs,
    klein_stack_lora_forward_device_inputs_resident,
    klein_stack_lora_forward_device_inputs_resident_moddev,
    klein_stack_lora_forward_device_inputs_resident_moddev_rope,
    klein_stack_lora_forward_device_inputs_resident_moddev_rope_scratch,
    klein_stack_lora_forward_device_inputs_offload_turbo_moddev_rope_scratch,
    klein_stack_lora_forward_device_inputs_offload_turbo_moddev_rope_scratch_offloaded_tape,
    klein_stack_lora_backward, klein_stack_lora_backward_resident,
    klein_stack_lora_backward_resident_moddev,
    klein_stack_lora_backward_resident_moddev_rope,
    klein_stack_lora_backward_resident_moddev_rope_scratch,
    klein_stack_lora_backward_offload_turbo_moddev_rope_scratch,
    klein_stack_lora_backward_graph,
    klein_stack_lora_backward_offloaded_tape_turbo_moddev_rope_scratch,
    klein_lora_adamw_step, save_klein_lora, load_klein_lora_resume,
    save_klein_lora_state, load_klein_lora_state, save_klein_lora_ema,
    klein_lora_set_to_device_resident, klein_lora_adamw_step_resident,
    DBL_SLOTS,
)
from serenitymojo.models.klein.weights import (
    load_double_block_weights, load_single_block_weights,
    load_klein_stack_base_training,
    build_klein_double_modvecs, build_klein_single_modvecs,
    load_klein_step_mod_weights, build_klein_step_mods_device_cached,
)
from serenitymojo.training.klein_dataset import KleinCache, KleinSample
from serenitymojo.training.schedule import (
    sample_timestep_logit_normal, flow_match_noise_target,
    sample_timestep_uniform, sample_timestep_sigmoid,
    TSD_UNIFORM, TSD_SIGMOID, TSD_LOGIT_NORMAL,
)
from serenitymojo.training.timestep_bias import apply_bias
from serenitymojo.training.loss_weight import (
    apply_loss_weight, combined_loss_grad_elem,
)
from serenitymojo.training.levers import (
    caption_dropout_pick,
    levers_loss_active, levers_loss_grad,
    LeversOptimizerState, levers_optimizer_active, levers_optimizer_validate,
    levers_optimizer_step_host, levers_optimizer_sync_resident_ot,
    levers_optimizer_eval_for_save, levers_optimizer_train_after_save,
)
from serenitymojo.training.noise_modifiers import apply_noise_modifiers_host
from serenitymojo.training.ema_schedule import ema_decay_at_step, ema_update_host
from serenitymojo.training.grad_accum import (
    accumulate_grad_group, scale_grad_group, zeros_like_group,
)
from serenitymojo.training.validation_sampler import load_caps
from serenitymojo.training.progress_display import print_trainer_progress
from serenitymojo.training.sample_prompt_config import (
    SamplePrompt, SamplePromptConfig, SampleCadence,
    read_sample_prompt_config, read_sample_cadence_config,
    validate_step_sample_cadence, should_sample_completed_step,
    next_sample_completed_step, sample_time_unit_name,
    SAMPLE_UNIT_NEVER, SAMPLE_UNIT_STEP,
)
from serenitymojo.training.onetrainer_train_loop_policy import (
    OT_GRAD_POLICY_ON_OR_CPU_OFFLOADED,
    ot_cache_dir_from_train_config,
    ot_final_or_step_lora_path,
    ot_sample_cadence_from_train_config,
    ot_sampling_enabled,
    ot_should_save_before_sample,
    ot_should_save_checkpoint,
    ot_state_path_for_lora,
    ot_lr_for_optimizer_step,
    validate_ot_gradient_checkpointing_policy,
    validate_ot_lora_adamw_loop_policy,
    validate_ot_train_math_policy,
)
from serenitymojo.training.serenityboard import SerenityBoardWriter
from serenitymojo.sampling.klein_sampler import klein_sample
from serenitymojo.offload.plan import build_klein_block_plan, OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.training.train_config import (
    TrainConfig, TRAIN_OPTIMIZER_ADAMW,
)
from serenitymojo.training.onetrainer_cache_preflight import (
    create_onetrainer_cache_preflight_plan,
    validate_onetrainer_cache_preflight_plan,
)
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.ops.tensor_algebra import permute, reshape, reshape_owned
from serenitymojo.io.ffi import sys_system, sys_open, sys_close, O_RDONLY


comptime TArc = ArcPointer[Tensor]

# v2 ENGINE SWAP (maintainer mandate 2026-06-11, serenitymojo/docs/
# MOJO_V2_ENGINE_PLAN.md): resident-set workstream — persistent device
# OT-AdamW P/M/V whose param buffer the model's device LoRA set views
# (no per-step set upload, no per-step P/M/V round trip). Bit-identical
# math (same kernel, same SR stream). False = previous per-step path.
comptime KLEIN_V2_ENGINE = True
# P6 GRAPH SWAP (AUTOGRAD_V2_MOJO_DESIGN.md P6): the per-block recompute +
# hand-chain backward pair is driven by the autograd_v2 graph engine
# (klein_stack_lora_backward_graph — same conductor loop, same scratch rings,
# per-block mini-graphs whose arms call the hand-chain's own backward
# helpers; SAME-PROCESS bit gate: autograd_v2/tests/klein_block_parity.mojo).
# Only active when KLEIN_V2_ENGINE is also True, and only on the resident
# (non-offloaded-tape) path; the CPU_OFFLOADED activation-tape path keeps the
# hand-chain. False = previous hand-chain path (C13 gate-don't-delete).
comptime KLEIN_V2_GRAPH = True
comptime KLEIN_V2_GRAPH_PATH = KLEIN_V2_ENGINE and KLEIN_V2_GRAPH

from serenitymojo.training.lora_adamw_ot_fused import (
    LoraAdamWOTDeviceState, lora_adamw_ot_device_state_init,
    lora_adamw_ot_device_state_sync_moments,
)
# T2.G LoKr e2e training (adapter_algo==4): SimpleTuner-parity LoKr masters
# trained through the existing stack via the Kronecker carrier representation
# (see training/lokr_stack.mojo header for the math + the ST knob mapping).
from serenitymojo.training.lokr_stack import (
    KleinLoKrSet, KleinLoKrGrads, build_klein_lokr_set, empty_klein_lokr_set,
    klein_lokr_carrier_lists, lokr_carrier_total_bytes,
    LOKR_CARRIER_MAX_DEVICE_BYTES, klein_lokr_chain_all, klein_lokr_grad_norm,
    klein_lokr_clip_grads, klein_lokr_adamw_step, klein_lokr_trainable_l1,
    klein_lokr_zero_leg_l1, save_klein_lokr, klein_lokr_apply_perturbed_init,
)


# ── config file (binding rule 2026-05-31: arch + recipe come from the FILE) ──
# Pass `train_klein_real <config.json>` to pick a variant; defaults to 9B.
comptime DEFAULT_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/klein9b.json"

# ── dataset / output paths (run wiring, not model arch) ──────────────────────
comptime CACHE_DIR = "/home/alex/EriDiffusion/EriDiffusion-v2/cache/alina_klein9b"
comptime SAMPLE_DIR = "/home/alex/mojodiffusion/output/alina_train"
comptime LORA_DIR = "/home/alex/mojodiffusion/output/alina_train"

# ── attention SHAPE + resolution — Mojo COMPTIME generic params ──────────────
# These five (H, Dh, N_IMG, N_TXT, S) are compile-time generic args of the
# klein_stack_lora_forward/backward functions and CANNOT be purely file-driven
# without making every [H,Dh,N_IMG,N_TXT,S]-generic function runtime-generic (a
# much larger refactor). They are ASSERTED against the config at main() start
# (H==n_heads, Dh==head_dim, H*Dh==d_model). N_IMG/N_TXT/LH/LW encode the
# resolution + caption budget (512px: 32x32 packed -> 1024 image tokens; 512 txt).
comptime LH = 32
comptime LW = 32
comptime N_IMG = 1024
comptime N_TXT = 512
comptime S = N_IMG + N_TXT
comptime H = 32
comptime Dh = 128

# ── run knobs (NOT model params; cadence target comes from cfg.max_steps) ─────
comptime SEED_BASE = UInt64(1234)
# RUN_STEPS controls THIS invocation (<= cfg.max_steps). The full run sets this
# to cfg.max_steps; the timed dry run keeps it small.
comptime RUN_STEPS = 0
# DO_SAMPLE is kept for short/manual runs. Production runs pass `nosample` and
# use train_klein_cadence.mojo to launch sampler processes at cadence boundaries.
comptime DO_SAMPLE = True
# Baseline sampling is enabled for production validation runs. Runtime 50-step
# smoke overrides skip it because an undertrained LoRA sample is not useful.
comptime SAMPLE_BASELINE = True
comptime SAMPLE_PROMPTS = 2
comptime SAMPLE_LH = 64
comptime SAMPLE_LW = 64
comptime SAMPLE_N_IMG = 4096
comptime SAMPLE_S = SAMPLE_N_IMG + N_TXT
comptime SAMPLE_STEPS = 20
comptime SAMPLE_CFG = Float32(4.0)
comptime SAMPLE_SEED = UInt64(42)
# Phase-4 residency budget: bytes of transformer blocks pinned permanently on
# device (rest stream). MEASURED 2026-06-11 at 9 GiB (14/32 blocks): step
# 4.5 -> 3.2 s, VRAM peak 21.3/24.5 GiB (nosample — LOWER this if sampling
# in-process). The 8e-5 step-1 loss shift was GATED then CLEARED same day:
# zero-pin run reproduces the old anchor EXACTLY (0.5414262) and
# offload/resident_byte_identity_smoke.mojo proves resident bytes == streamed
# bytes — the shift is pointer-alignment GEMM algo selection (same accepted
# class as the fused-AdamW m/v ties). Anchors re-recorded below.
comptime RESIDENT_BUDGET_BYTES = 9 * 1024 * 1024 * 1024
comptime SCRATCH_FWD_SLAB_BYTES = 512 * 1024 * 1024
comptime SCRATCH_FWD_SLABS = 2
comptime SCRATCH_BWD_SLAB_BYTES = 1024 * 1024 * 1024
comptime SCRATCH_BWD_SLABS = 3
comptime VERBOSE_STAGE_LOG = False
comptime MACHINE_PROGRESS_LOG = False


# ─────────────────────────────────────────────────────────────────────────────
# host helpers
# ─────────────────────────────────────────────────────────────────────────────


# Latent [1,in_ch,LH,LW] -> img_tokens device [N_IMG, in_ch], preserving
# cache storage dtype. Flow-match kernels do any needed F32 arithmetic inside
# the op and return the input/storage dtype.
# NCHW -> permute(0,2,3,1) -> NHWC -> reshape [N_IMG, in_ch]. `in_ch` is config-
# driven (cfg.in_channels); N_IMG stays comptime (resolution).
def _latent_to_img_tokens_device(
    latent: Tensor, in_ch: Int, ctx: DeviceContext
) raises -> Tensor:
    var p = List[Int]()
    p.append(0); p.append(2); p.append(3); p.append(1)
    var nhwc = permute(latent, p^, ctx)
    var sh = List[Int]()
    sh.append(N_IMG); sh.append(in_ch)
    return reshape_owned(nhwc^, sh^)


def _host_f32_for_step_math(t: Tensor, ctx: DeviceContext) raises -> List[Float32]:
    """Stage tensors through their stored dtype before host loss/stat math."""
    if t.dtype() == STDtype.BF16:
        var bf = t.to_host_bf16(ctx)
        var out = List[Float32]()
        for i in range(len(bf)):
            out.append(bf[i].cast[DType.float32]())
        return out^
    if t.dtype() == STDtype.F16:
        var hf = t.to_host_f16(ctx)
        var out = List[Float32]()
        for i in range(len(hf)):
            out.append(hf[i].cast[DType.float32]())
        return out^
    return t.to_host(ctx)


struct KleinLossGrad(Movable):
    var loss: Float32
    var d_loss: List[Float32]

    def __init__(out self, loss: Float32, var d_loss: List[Float32]):
        self.loss = loss
        self.d_loss = d_loss^


def _klein_loss_grad(
    pred: List[Float32],
    target: List[Float32],
    sigma: Float32,
    cfg: TrainConfig,
) raises -> KleinLossGrad:
    if len(pred) != len(target):
        raise Error("Klein loss: predicted/target length mismatch")
    # ── T1.A levers loss dispatch (default-off; levers.mojo) ─────────────────
    # PRECEDENCE DECISION (documented): when ANY levers loss lever is set
    # (cfg.loss_fn != mse OR cfg.min_snr_gamma_flow > 0), the levers path
    # REPLACES klein's own scheme for the run — klein's combined
    # mse/mae/huber strengths AND its apply_loss_weight min-snr weighting
    # (the (SNR+1)-divisor cfg.min_snr_gamma form) are IGNORED; the
    # SimpleTuner flow form min(SNR,γ)/SNR (cfg.min_snr_gamma_flow) applies
    # instead. A setup-time warning prints when both are set
    # (validate_klein_train_config). Levers off (the config default) falls
    # through to the literal klein block below UNCHANGED — C13: the 1-step
    # anchors (0.5414x/0.2154x/0.7809x) cannot move.
    if levers_loss_active(cfg):
        var lev = levers_loss_grad(pred, target, sigma, cfg)
        # .copy(): Mojo forbids moving a field out of `lev` here (partial
        # destruction); levers-path-only host copy, ~N_OUT floats.
        return KleinLossGrad(lev.loss, lev.d_pred.copy())
    var nout = len(pred)
    var w = apply_loss_weight(
        sigma, cfg.min_snr_gamma, cfg.debiased, True
    )
    var mse_s = cfg.loss_mse_strength
    var mae_s = cfg.loss_mae_strength
    var huber_s = cfg.loss_huber_strength
    var combined_levers_on = (
        mae_s != Float32(0.0) or huber_s != Float32(0.0)
        or mse_s != Float32(1.0)
    )
    var loss_default_path = (w == Float32(1.0)) and (not combined_levers_on)
    var d_loss = List[Float32]()
    var loss: Float32
    if loss_default_path:
        var sse = 0.0
        var inv_n = Float32(2.0) / Float32(nout)
        for i in range(nout):
            var diff = pred[i] - target[i]
            sse += Float64(diff) * Float64(diff)
            d_loss.append(inv_n * diff)
        loss = Float32(sse / Float64(nout))
    else:
        var sum_sq = 0.0
        var sum_abs = 0.0
        var sum_hub = 0.0
        for i in range(nout):
            var diff = pred[i] - target[i]
            var fd = Float64(diff)
            sum_sq += fd * fd
            var ad = fd if fd >= 0.0 else -fd
            sum_abs += ad
            var ac = ad if ad <= 1.0 else 1.0
            var lin = ad - 1.0
            if lin < 0.0:
                lin = 0.0
            sum_hub += 0.5 * ac * ac + lin
            d_loss.append(w * combined_loss_grad_elem(diff, nout, mse_s, mae_s, huber_s))
        var invn = 1.0 / Float64(nout)
        var combined = (
            Float64(mse_s) * (sum_sq * invn)
            + Float64(mae_s) * (sum_abs * invn)
            + Float64(huber_s) * (sum_hub * invn)
        )
        loss = Float32(Float64(w) * combined)
    return KleinLossGrad(loss, d_loss^)


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
        else:
            # text-token RoPE = [0,0,0,k] (upstream Flux2 prepare_text_ids:
            # cartesian_prod(arange(1)x3, arange(L))). Axis-3 carries the L-axis
            # rotary freqs, so each text token gets a distinct phase. Was all-zero
            # (collapsed text positions) — the bug the Rust port already fixed
            # (EDv2 klein.rs KLEIN_VERIFY §H2; txt_ids_data[k*4+3]=k).
            p3 = tok
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


# Host List[Float32] of n zeros (uncond caption embedding seed).
def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
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


def _compatible_cache_indices(
    cache: KleinCache, cfg: TrainConfig, ctx: DeviceContext
) raises -> List[Int]:
    """Return cache indices that match this compile-time 512px bucket."""
    var out = List[Int]()
    for i in range(cache.count()):
        var key = cache.peek_key(i, ctx)
        if (
            key.c == cfg.in_channels
            and key.h == LH
            and key.w == LW
            and key.seq == N_TXT
        ):
            out.append(i)
    if len(out) == 0:
        raise Error(
            String("no compatible cache samples for latent [1,")
            + String(cfg.in_channels) + String(",") + String(LH)
            + String(",") + String(LW) + String("] text_seq=")
            + String(N_TXT)
        )
    return out^


def _parse_nonnegative_int(s: String) raises -> Int:
    var out = 0
    var bs = s.as_bytes()
    for i in range(s.byte_length()):
        var ch = bs[i]
        if ch < 0x30 or ch > 0x39:
            raise Error(String("expected integer, got ") + s)
        out = out * 10 + Int(ch - 0x30)
    return out


def _close_f32(a: Float32, b: Float32, tol: Float32 = Float32(1.0e-7)) -> Bool:
    var d = a - b
    if d < Float32(0.0):
        d = -d
    return d <= tol


def _mode_disables_sampling(mode: String) -> Bool:
    return mode == String("nosample") or mode == String("nosample_profile")


def _mode_enables_profile(mode: String) -> Bool:
    return mode == String("profile") or mode == String("nosample_profile")


def _path_exists(path: String) -> Bool:
    var fd = sys_open(path, O_RDONLY, 0)
    if fd < 0:
        return False
    _ = sys_close(fd)
    return True


def _validate_precached_caps(
    sample_cfg: SamplePromptConfig, expected_joint_dim: Int
) raises:
    for i in range(len(sample_cfg.prompts)):
        var p = sample_cfg.prompts[i].copy()
        validate_klein_cap_cache_header(p.caps_pos, expected_joint_dim)
        validate_klein_cap_cache_header(p.caps_neg, expected_joint_dim)


def _sample_png_path(step: Int, label: String) -> String:
    return (
        String(SAMPLE_DIR) + String("/sample_step") + String(step)
        + String("_") + label + String(".png")
    )


def _state_path_for_lora(lora_path: String) -> String:
    return ot_state_path_for_lora(lora_path)


# EMA shadow checkpoint sibling path: alina_lora_step100.safetensors ->
# alina_lora_step100_ema.safetensors. Written only when cfg.ema_enabled.
def _ema_path_for_lora(lora_path: String) -> String:
    var suffix = String(".safetensors")
    if lora_path.endswith(suffix):
        return lora_path.removesuffix(suffix) + String("_ema.safetensors")
    return lora_path + String("_ema.safetensors")


def _lora_path_for_step(base_path: String, step: Int, max_steps: Int) -> String:
    return ot_final_or_step_lora_path(
        base_path,
        String(LORA_DIR),
        String("alina_lora_final.safetensors"),
        String("alina_lora_step"),
        step,
        max_steps,
    )


def validate_klein_train_config(cfg: TrainConfig) raises:
    if cfg.checkpoint == String(""):
        raise Error("Klein trainer config must set checkpoint")
    if not cfg.checkpoint.endswith(String(".safetensors")):
        raise Error(
            String("Klein trainer currently requires a single safetensors checkpoint; ")
            + String("sharded transformer dirs need a dedicated product loader")
        )
    if cfg.n_heads != H:
        raise Error(String("config n_heads ") + String(cfg.n_heads) + " != comptime H " + String(H))
    if cfg.head_dim != Dh:
        raise Error(String("config head_dim ") + String(cfg.head_dim) + " != comptime Dh " + String(Dh))
    if cfg.d_model != H * Dh:
        raise Error(String("config d_model ") + String(cfg.d_model) + " != H*Dh " + String(H * Dh))
    if cfg.in_channels != 128:
        raise Error(String("Klein 9B trainer requires in_channels=128; parsed ") + String(cfg.in_channels))
    if cfg.joint_attention_dim != 12288:
        raise Error(String("Klein 9B trainer requires joint_attention_dim=12288; parsed ") + String(cfg.joint_attention_dim))
    if cfg.out_channels != 128:
        raise Error(String("Klein 9B trainer requires out_channels=128; parsed ") + String(cfg.out_channels))
    if cfg.num_double != 8 or cfg.num_single != 24:
        raise Error(
            String("Klein 9B trainer requires double=8 single=24; got double=")
            + String(cfg.num_double) + String(" single=") + String(cfg.num_single)
        )
    if cfg.mlp_hidden != 12288:
        raise Error(String("Klein 9B trainer requires mlp_hidden=12288; parsed ") + String(cfg.mlp_hidden))
    if cfg.timestep_dim != 256:
        raise Error(String("Klein 9B trainer requires timestep_dim=256; parsed ") + String(cfg.timestep_dim))
    if not _close_f32(Float32(cfg.rope_theta), Float32(2000.0)):
        raise Error(String("Klein 9B trainer requires rope_theta=2000; parsed ") + String(cfg.rope_theta))
    # T1 lever fan-out: klein wires the levers optimizer dispatch
    # (training/levers.mojo T1.C), so the supported non-AdamW optimizers
    # (ADAFACTOR / SCHEDULE_FREE_ADAMW) must pass the shared ADAMW-only
    # loop-policy checks: levers_optimizer_validate re-asserts the supported
    # set, then the shared policies run on a tag-neutralized copy (zimage
    # precedent, train_zimage_real.mojo validate). Default optimizer=ADAMW
    # keeps policy_cfg == cfg — checks byte-identical to before (C13).
    levers_optimizer_validate(cfg, String("Klein trainer"))
    var policy_cfg = cfg.copy()
    if levers_optimizer_active(cfg):
        policy_cfg.optimizer = TRAIN_OPTIMIZER_ADAMW
    validate_ot_lora_adamw_loop_policy(policy_cfg, String("Klein trainer"))
    validate_ot_train_math_policy(policy_cfg, String("Klein trainer"))
    validate_ot_gradient_checkpointing_policy(
        cfg, String("Klein trainer"), OT_GRAD_POLICY_ON_OR_CPU_OFFLOADED
    )
    # T1.A loss-lever precedence warning (decision: LEVERS WIN — see
    # _klein_loss_grad): if a run sets BOTH klein's own combined-loss /
    # min-snr keys AND a levers loss key, say so loudly once at setup.
    if levers_loss_active(cfg):
        var klein_loss_keys_set = (
            cfg.loss_mae_strength != Float32(0.0)
            or cfg.loss_huber_strength != Float32(0.0)
            or cfg.loss_mse_strength != Float32(1.0)
            or cfg.min_snr_gamma >= Float32(0.0)
            or cfg.debiased
        )
        if klein_loss_keys_set:
            print(
                "[Klein-lora] WARNING: levers loss keys (loss_fn/",
                "min_snr_gamma_flow) are set — they REPLACE klein's",
                " combined mse/mae/huber strengths, min_snr_gamma",
                " ((SNR+1) divisor) and debiased for this run; those",
                " keys are IGNORED.",
            )


def klein_cache_dir_from_train_config(cfg: TrainConfig) -> String:
    return ot_cache_dir_from_train_config(cfg, String(CACHE_DIR))


def klein_output_lora_path_from_train_config(
    cfg: TrainConfig, completed_step: Int,
) -> String:
    return _lora_path_for_step(cfg.output_model_destination, completed_step, cfg.max_steps)


def klein_sample_cadence_from_train_config(
    cfg_path: String, cfg: TrainConfig,
) raises -> SampleCadence:
    return ot_sample_cadence_from_train_config(cfg_path, cfg)


def klein_sampling_enabled(cadence: SampleCadence) -> Bool:
    return ot_sampling_enabled(cadence)


def klein_should_save_checkpoint(cfg: TrainConfig, completed_step: Int) -> Bool:
    return ot_should_save_checkpoint(cfg, completed_step)


def klein_should_save_before_sample(
    cadence: SampleCadence, completed_step: Int, saved_this_step: Bool,
) raises -> Bool:
    return ot_should_save_before_sample(cadence, completed_step, saved_this_step)


def _do_sample_prompt(
    cfg: TrainConfig,
    p: SamplePrompt,
    step: Int,
    lora_path: String,
    ctx: DeviceContext,
) raises -> String:
    if p.frames != 1:
        raise Error(String("Klein image sampler expects frames=1 for ") + p.label)
    var png = _sample_png_path(step, p.label)
    var caps = load_caps(p.caps_pos, p.caps_neg, ctx)
    var txt_sh = List[Int]()
    txt_sh.append(N_TXT)
    txt_sh.append(cfg.joint_attention_dim)
    var pos_txt = reshape(caps.pos, txt_sh.copy(), ctx)
    var neg_txt = reshape(caps.neg, txt_sh^, ctx)
    if p.width == 1024 and p.height == 1024:
        var _img1024 = klein_sample[SAMPLE_N_IMG, N_TXT, SAMPLE_S, SAMPLE_LH, SAMPLE_LW, H, Dh](
            cfg, lora_path, pos_txt, neg_txt, p.cfg, p.steps, p.seed, png, ctx,
        )
    elif p.width == 512 and p.height == 512:
        var _img512 = klein_sample[N_IMG, N_TXT, S, LH, LW, H, Dh](
            cfg, lora_path, pos_txt, neg_txt, p.cfg, p.steps, p.seed, png, ctx,
        )
    else:
        raise Error(
            String("Klein sampler unsupported sample size ")
            + String(p.width) + String("x") + String(p.height)
            + String(" for ") + p.label + String(" (supported: 512, 1024)")
        )
    print("[Klein-lora] sample step=", step, " label=", p.label, " path=", png)
    return png^


def _do_sample_all(
    cfg: TrainConfig,
    sample_cfg: SamplePromptConfig,
    step: Int,
    lora_path: String,
    board: SerenityBoardWriter,
    ctx: DeviceContext,
) raises:
    for i in range(len(sample_cfg.prompts)):
        var p = sample_cfg.prompts[i].copy()
        var png = _do_sample_prompt(cfg, p, step, lora_path, ctx)
        board.log_image_png(String("samples/") + p.label, step, i, png)
        board.log_text(String("prompts/") + p.label, step, p.prompt)


def main() raises:
    # ── read the model config FILE (arch + recipe + paths). argv[1] overrides. ─
    var a = argv()
    var cfg_path = String(DEFAULT_CONFIG)
    if len(a) >= 2:
        cfg_path = String(a[1])
    var cfg = read_model_config(cfg_path)
    validate_klein_train_config(cfg)
    # T2.G: LoKr (adapter_algo==4) routes the adapter set through the Kronecker
    # carrier path. Default (0) leaves every branch below byte-unchanged.
    var lokr_active = cfg.adapter_algo == 4
    var cache_preflight = create_onetrainer_cache_preflight_plan(cfg)
    validate_onetrainer_cache_preflight_plan(cache_preflight)
    var output_lora_path = String("")
    if cfg.output_model_destination != String(""):
        output_lora_path = cfg.output_model_destination.copy()
    var run_steps = cfg.max_steps
    if RUN_STEPS > 0:
        run_steps = RUN_STEPS
    if len(a) >= 3:
        run_steps = _parse_nonnegative_int(String(a[2]))
    if run_steps > cfg.max_steps:
        run_steps = cfg.max_steps
    var start_step = 0
    if len(a) >= 4:
        start_step = _parse_nonnegative_int(String(a[3]))
    if start_step > run_steps:
        raise Error(
            String("start_step ") + String(start_step)
            + String(" > run_until_step ") + String(run_steps)
        )
    var resume_lora = String("")
    if len(a) >= 5:
        var rp = String(a[4])
        if rp != String("-") and rp != String(""):
            resume_lora = rp^
    var mode = String("")
    if len(a) >= 6:
        mode = String(a[5])
    var runtime_profile = VERBOSE_STAGE_LOG or _mode_enables_profile(mode)
    var cache_dir = klein_cache_dir_from_train_config(cfg)

    print("=== Klein REAL LoRA training loop:", cfg.name, "===")
    print("  config:", cfg_path)
    print("  cache:", cache_dir)
    print("  checkpoint:", cfg.checkpoint)
    print("  output_lora:", klein_output_lora_path_from_train_config(cfg, run_steps))

    if cfg.only_cache:
        print("[Klein] only_cache requested; no CUDA context or train steps will run in this trainer")
        return

    if cfg.validation_prompts_file == String(""):
        raise Error("Klein trainer config must set validation_prompts_file")
    var sample_cadence = klein_sample_cadence_from_train_config(cfg_path, cfg)
    if sample_cadence.sample_definition_file_name == String(""):
        raise Error("Klein trainer sampling requires sample_definition_file_name or validation_prompts_file")
    var sample_cfg = read_sample_prompt_config(sample_cadence.sample_definition_file_name)
    var runtime_sample_enabled = (
        DO_SAMPLE and not _mode_disables_sampling(mode)
        and klein_sampling_enabled(sample_cadence)
    )
    var sample_every = sample_cadence.sample_every_steps(sample_cfg.every_steps)
    if runtime_sample_enabled:
        _validate_precached_caps(sample_cfg, cfg.joint_attention_dim)

    var ctx = DeviceContext()
    _ = sys_system(String("mkdir -p ") + SAMPLE_DIR)
    var board = SerenityBoardWriter.open(String(SAMPLE_DIR), String("klein_lora_mojo"), start_step)
    board.log_hparams(
        String("{\"model\":\"") + cfg.name + String("\",\"lr\":") + String(cfg.lr)
        + String(",\"max_steps\":") + String(cfg.max_steps)
        + String(",\"save_every\":") + String(cfg.save_every)
        + String(",\"sample_every\":") + String(sample_every) + String("}")
    )
    board.log_text(String("config/train"), 0, cfg_path)
    board.log_text(String("config/sample_prompts"), 0, sample_cadence.sample_definition_file_name)

    print("  sample prompts:", sample_cadence.sample_definition_file_name, " count=", len(sample_cfg.prompts))
    print(
        "  512px latent: N_IMG=", N_IMG, " N_TXT=", N_TXT, " S=", S,
        " rank=", cfg.lora_rank, " alpha=", cfg.lora_alpha, " lr=", cfg.lr,
        " shift=", cfg.timestep_shift,
    )
    print("  arch: d_model=", cfg.d_model, " double=", cfg.num_double, " single=", cfg.num_single)
    print(
        "  cadence target max_steps=", cfg.max_steps,
        " worker range=", start_step + 1, "..", run_steps,
        " sample_every=", sample_every,
        " sample_unit=", sample_time_unit_name(sample_cadence.sample_after_unit),
        " sample_skip_first=", sample_cadence.sample_skip_first,
        " mode=", mode,
    )
    var next_sample = next_sample_completed_step(sample_cadence, start_step, cfg.max_steps)
    print("[cadence] next sample completed_step=", next_sample)

    # Step-0 baseline samples run before the training stack loads. A short smoke
    # (< sample_every) skips sampling so we do not waste time judging a 50-step LoRA.
    if (
        runtime_sample_enabled and start_step == 0
        and should_sample_completed_step(sample_cadence, 0)
        and run_steps >= sample_every
    ):
        print("[cadence] step 0 baseline samples (no LoRA)")
        _do_sample_all(cfg, sample_cfg, 0, String(""), board, ctx)
    else:
        print("[cadence] step 0 baseline sample skipped")

    # ── load shared projections/modulation; stream transformer blocks on demand ─
    print("[load] Klein base projections + turbo block loader")
    var st = SafeTensors.open(cfg.checkpoint)
    # Training overwrites final_shift/final_scale from per-step timestep mods
    # before every forward, so avoid the old seed final-mod GEMM at startup.
    var base = load_klein_stack_base_training(st, cfg.d_model, ctx)
    if runtime_profile:
        print("PROG_STAGE step=0 total=", run_steps, " phase=load_base")
    var mod_weights = load_klein_step_mod_weights(st, cfg.d_model, ctx)
    if runtime_profile:
        print("PROG_STAGE step=0 total=", run_steps, " phase=load_step_mod_weights")
    var plan = build_klein_block_plan(cfg.num_double, cfg.num_single)
    var loader = TurboPlannedLoader.open(
        cfg.checkpoint, plan^, OffloadConfig.synchronous_single(), ctx
    )
    if runtime_profile:
        print("PROG_STAGE step=0 total=", run_steps, " phase=open_turbo_loader")
    print("  block stream:", cfg.num_double, "double +", cfg.num_single, "single blocks")
    # ── Phase-4 residency (2026-06-11): pin blocks permanently on device up to
    # a VRAM budget; the rest keep streaming through the 2 turbo slots. MEASURED
    # basis: 12.4 GiB peak of 24.5 GiB during a streamed step → ~9 GiB headroom.
    # Byte-identical weights (same pinned block store bytes) → loss unchanged.
    # T2.G: LoKr runs pin NOTHING — the carrier adapter set takes the VRAM
    # headroom the pinned blocks would use (full_matrix carriers especially).
    var resident_budget_bytes = 0 if lokr_active else RESIDENT_BUDGET_BYTES
    var pinned_blocks = loader.pin_residents(resident_budget_bytes, ctx)
    print(
        "  resident blocks pinned:", pinned_blocks, "of",
        cfg.num_double + cfg.num_single,
        " budget_bytes=", resident_budget_bytes,
    )
    var use_activation_tape_offload = cfg.activation_offload_enabled()
    if cfg.gradient_checkpointing_offload():
        print(
            "  cpu_offloaded:",
            " activation_offload=", use_activation_tape_offload,
            " layer_offload_fraction=", Float32(cfg.layer_offload_fraction),
        )

    # ── adapter algo selector (Wave 2B item 2j; default 0 = plain LoRA) ───────
    # adapter_algo==1 selects LyCORIS Full (full-shape weight delta). The Full
    # PRIMITIVE + .diff.weight save convention ship in training/full_adapter.mojo
    # and are gated by full_adapter_smoke.mojo. The Klein stack
    # forward/backward is currently low-rank-only (KleinLoraSet), so wiring the
    # full-delta weights through the stack is a flagged follow-up; we fail loud
    # rather than silently train the wrong thing. Default (0) leaves the proven
    # LoRA path byte-unchanged.
    if cfg.adapter_algo == 1:
        raise Error(
            "adapter_algo=1 (LyCORIS Full) selected: the Full adapter primitive "
            + "+ .diff.weight save ship in training/full_adapter.mojo, but the "
            + "Klein stack forward/backward is low-rank-only — wiring full-delta "
            + "weights through the stack is a tracked follow-up. Use adapter_algo=0 "
            + "(plain LoRA) for now."
        )
    elif cfg.adapter_algo == 2:
        # adapter_algo==2 selects LyCORIS LoHa. The LoHa PRIMITIVE (4-factor
        # forward/backward/AdamW) + hada_w1/w2 + .alpha save convention ship in
        # training/loha_adapter.mojo + training/loha_save.mojo and are gated by
        # loha_adapter_smoke.mojo (parity + 4-factor grad-flow + save round-trip,
        # incl. a deliberate dead-factor abort). The Klein stack
        # forward/backward (klein_stack_lora_*) is HARDWIRED to the 2-factor
        # KleinLoraSet A/B path across the offloaded-scratch arms, so swapping
        # every adapted linear onto the 4-factor LoHa delta is a larger
        # follow-up. We fail loud rather than silently train the wrong thing.
        raise Error(
            "adapter_algo=2 (LyCORIS LoHa) selected: the LoHa adapter primitive "
            + "(4-factor fwd/bwd/AdamW) + hada_w1/w2 + .alpha save ship in "
            + "training/loha_adapter.mojo + training/loha_save.mojo and are gated "
            + "by loha_adapter_smoke.mojo, but the Klein stack forward/backward is "
            + "2-factor A/B only — wiring the 4-factor LoHa delta through the stack "
            + "is a tracked follow-up. Use adapter_algo=0 (plain LoRA) for now."
        )
    elif cfg.adapter_algo == 3:
        # adapter_algo==3 selects DoRA (weight-decomposed LoRA). The DoRA
        # PRIMITIVE (effective-weight fwd, 3-grad bwd over lora_down/lora_up/
        # magnitude, AdamW) + lora_down/lora_up/.dora_scale/.alpha save convention
        # ship in training/dora_adapter.mojo + training/dora_save.mojo and are
        # gated by dora_adapter_smoke.mojo (effective-weight + FD-detached-norm
        # parity + 3-grad grad-flow + save round-trip, incl. a deliberate
        # dead-grad abort). The Klein stack forward/backward (klein_stack_lora_*)
        # is hardwired to the additive 2-factor KleinLoraSet A/B delta; DoRA
        # REPLACES the effective weight (m × normalized WP) rather than adding a
        # delta, so routing every adapted linear through WP_dora is a larger
        # follow-up. We fail loud rather than silently train the wrong thing.
        raise Error(
            "adapter_algo=3 (DoRA) selected: the DoRA adapter primitive "
            + "(effective-weight fwd, 3-grad bwd, AdamW) + lora_down/lora_up/"
            + ".dora_scale/.alpha save ship in training/dora_adapter.mojo + "
            + "training/dora_save.mojo and are gated by dora_adapter_smoke.mojo, "
            + "but the Klein stack is additive-LoRA-only — wiring the "
            + "magnitude/direction WP_dora through the stack is a tracked "
            + "follow-up. Use adapter_algo=0 (plain LoRA) for now."
        )
    elif cfg.adapter_algo == 4:
        # adapter_algo==4: LyCORIS LoKr e2e TRAINING (T2.G 2026-06-11,
        # SimpleTuner-parity). LoKr masters (training/lokr_adapter.mojo,
        # upstream lycoris_lora 3.4.0 semantics: factorization(dim,factor),
        # decompose_both/full_matrix factor selection, zero-leg or
        # --init_lokr_norm perturbed-normal init, both-full forced-scale=1
        # quirk) train through the EXISTING stack via the Kronecker carrier
        # representation (training/lokr_stack.mojo header). Host AdamW on the
        # masters; save in the upstream lycoris key convention (lokr_save).
        # CONSTRAINTS this wave (fail loud, honest scope):
        if cfg.grad_accum_steps > 1:
            raise Error("adapter_algo=4 (LoKr): grad_accum_steps>1 not wired this wave")
        if cfg.ema_enabled:
            raise Error("adapter_algo=4 (LoKr): EMA shadows not wired this wave")
        if levers_optimizer_active(cfg):
            raise Error("adapter_algo=4 (LoKr): levers optimizers not wired (host AdamW only)")
        if resume_lora != String(""):
            raise Error("adapter_algo=4 (LoKr): resume/init_lora warm-start not wired this wave")
        if cfg.init_lokr_norm > 0.0 and not cfg.lokr_full_matrix:
            raise Error(
                "adapter_algo=4 (LoKr): init_lokr_norm requires lokr_full_matrix "
                + "(SimpleTuner's init_lokr_network_with_perturbed_normal indexes "
                + "lokr_w1/lokr_w2 directly — full_matrix configs only)"
            )
        # In-process sampling loads PEFT LoRA files; the LoKr product file is
        # lycoris-format. Sampling is disabled for LoKr runs this wave.
        runtime_sample_enabled = False
        print(
            "[Klein-lokr] LoKr training: factor=", cfg.lokr_factor,
            " (attn=", cfg.lokr_factor_attn, " ff=", cfg.lokr_factor_ff,
            " single=", cfg.lokr_factor_single, ")",
            " decompose_both=", cfg.lokr_decompose_both,
            " full_matrix=", cfg.lokr_full_matrix,
            " targets=", cfg.lokr_targets,
            " init_lokr_norm=", cfg.init_lokr_norm,
            " rank=", cfg.lora_rank, " alpha=", cfg.lora_alpha,
        )
    elif cfg.adapter_algo == 5:
        # adapter_algo==5 selects Diag-OFT (Orthogonal Fine-Tuning). The OFT
        # PRIMITIVE (exact-Cayley R=(I+Q)⁻¹(I-Q) per output block, W_eff=R@W
        # forward, analytic d_S backward through the Cayley + skew chain, AdamW)
        # + oft_blocks/.alpha save convention ship in training/oft_adapter.mojo
        # + training/oft_save.mojo and are gated by oft_adapter_smoke.mojo
        # (RᵀR≈I orthogonality + identity-at-init + FD parity + grad-flow +
        # save round-trip, incl. a deliberate dead-grad abort). OFT is
        # MULTIPLICATIVE (W_eff = R·W), not an additive low-rank delta — the
        # Klein stack forward/backward is additive 2-factor A/B only, so routing
        # every adapted linear through the block-diagonal rotation is a tracked
        # follow-up. We fail loud rather than silently train the wrong thing.
        raise Error(
            "adapter_algo=5 (Diag-OFT) selected: the OFT adapter primitive "
            + "(exact-Cayley R, W_eff=R·W fwd, analytic d_S bwd, AdamW) + "
            + "oft_blocks/.alpha save ship in training/oft_adapter.mojo + "
            + "training/oft_save.mojo and are gated by oft_adapter_smoke.mojo, "
            + "but the Klein stack is additive-LoRA-only — wiring the "
            + "multiplicative block-diagonal rotation through the stack is a "
            + "tracked follow-up. Use adapter_algo=0 (plain LoRA) for now."
        )
    elif cfg.adapter_algo == 6:
        # adapter_algo==6 selects BOFT (Butterfly OFT). The BOFT PRIMITIVE
        # (m-stage butterfly product of block-diagonal exact-Cayley rotations
        # with butterfly permutation between stages, analytic per-stage d_S
        # backward via reverse-order accumulation, AdamW) + oft_blocks(4D)/.alpha
        # save convention ship in training/boft_adapter.mojo +
        # training/boft_save.mojo and are gated by boft_adapter_smoke.mojo
        # (overall-transform orthogonality + identity-at-init + FD parity for
        # ALL factor params + grad-flow + save round-trip, incl. a deliberate
        # dead-grad abort). Like OFT, BOFT is MULTIPLICATIVE — the Klein stack is
        # additive-LoRA-only, so wiring the butterfly rotation chain through the
        # stack is a tracked follow-up. We fail loud rather than silently train
        # the wrong thing.
        raise Error(
            "adapter_algo=6 (BOFT) selected: the BOFT adapter primitive "
            + "(m-stage butterfly Cayley rotation chain, analytic per-stage bwd, "
            + "AdamW) + oft_blocks(4D)/.alpha save ship in "
            + "training/boft_adapter.mojo + training/boft_save.mojo and are "
            + "gated by boft_adapter_smoke.mojo, but the Klein stack is "
            + "additive-LoRA-only — wiring the butterfly rotation chain through "
            + "the stack is a tracked follow-up. Use adapter_algo=0 (plain LoRA) for now."
        )
    elif cfg.adapter_algo != 0:
        raise Error(String("unknown adapter_algo ") + String(cfg.adapter_algo) + " (0=LoRA, 1=Full, 2=LoHa, 3=DoRA, 4=LoKr, 5=OFT, 6=BOFT)")

    # ── build LoRA set (OneTrainer split slots: 12 double + 2 single) ─────────
    var lora = build_klein_lora_set(
        cfg.num_double, cfg.num_single, cfg.d_model, cfg.mlp_hidden,
        cfg.lora_rank, cfg.lora_alpha
    )
    if resume_lora != String(""):
        var state_path = _state_path_for_lora(resume_lora)
        var resumed: KleinLoraSet
        if _path_exists(state_path):
            print("[Klein-lora] loading resume state:", state_path)
            resumed = load_klein_lora_state(
                cfg.num_double, cfg.num_single, cfg.lora_rank, cfg.lora_alpha,
                state_path, ctx,
            )
        else:
            print("[Klein-lora] loading resume LoRA without optimizer state:", resume_lora)
            resumed = load_klein_lora_resume(
                cfg.num_double, cfg.num_single, cfg.lora_rank, cfg.lora_alpha,
                resume_lora, ctx,
            )
        lora = resumed^
        board.log_text(String("events/resume"), start_step, resume_lora)
    print("  LoRA set:", len(lora.dbl), "double-slot +", len(lora.sgl), "single-slot")

    # ── T2.G LoKr masters + carrier set (adapter_algo==4 only) ────────────────
    # The plain-LoRA `lora` set built above is REPLACED by the LoKr carrier
    # set: same KleinLoraSet type, same stack forward/backward, but every
    # targeted slot carries the (a_c, b_c) Kronecker factorization of the LoKr
    # delta (lokr_stack.mojo L1/L2/L3). The LoKr masters own init, optimizer
    # state and the saved checkpoint.
    var lokr_masters = empty_klein_lokr_set()
    if lokr_active:
        lokr_masters = build_klein_lokr_set(
            cfg.num_double, cfg.num_single, cfg.d_model, cfg.mlp_hidden,
            cfg.lora_rank, cfg.lora_alpha,
            cfg.lokr_factor, cfg.lokr_factor_attn, cfg.lokr_factor_ff,
            cfg.lokr_factor_single,
            cfg.lokr_decompose_both, cfg.lokr_full_matrix, cfg.lokr_targets,
            SEED_BASE * UInt64(53) + UInt64(11),
        )
        if cfg.init_lokr_norm > 0.0:
            print("[Klein-lokr] perturbed-normal init (init_lokr_norm=", cfg.init_lokr_norm, ") — org stats pass")
            klein_lokr_apply_perturbed_init(
                lokr_masters, st, cfg.d_model, cfg.mlp_hidden,
                cfg.init_lokr_norm, SEED_BASE * UInt64(97) + UInt64(3),
            )
        var carrier_bytes = lokr_carrier_total_bytes(lokr_masters)
        print("[Klein-lokr] carrier device bytes:", carrier_bytes, " budget:", LOKR_CARRIER_MAX_DEVICE_BYTES)
        if carrier_bytes > LOKR_CARRIER_MAX_DEVICE_BYTES:
            raise Error(
                String("adapter_algo=4 (LoKr): carrier set needs ")
                + String(carrier_bytes) + " bytes on device (> budget "
                + String(LOKR_CARRIER_MAX_DEVICE_BYTES) + "). full_matrix or "
                + "factor=-1 at klein-9B dims produces dense/near-dense "
                + "carriers; use an explicit small lokr_factor, restrict "
                + "lokr_targets, or wait for the structured-kron kernel "
                + "follow-up."
            )
        var carriers = klein_lokr_carrier_lists(lokr_masters, cfg.d_model, cfg.mlp_hidden)
        lora = KleinLoraSet(
            carriers[0].copy(), carriers[1].copy(),
            cfg.num_double, cfg.num_single, cfg.lora_rank,
        )
        print("[Klein-lokr] carrier set materialized:", len(lora.dbl), "double-slot +", len(lora.sgl), "single-slot")

    # v2 engine: persistent device OT-AdamW state (dbl + sgl lists) + a
    # resident device LoRA set viewing the live param buffers. Built ONCE
    # after any resume; only used when KLEIN_V2_ENGINE. The LoKr path never
    # touches the resident OT state (host AdamW on the masters + per-step
    # carrier re-upload), so it gets a TINY dummy state instead of allocating
    # P/M/V for the full carrier set.
    var dbl_state: LoraAdamWOTDeviceState
    var sgl_state: LoraAdamWOTDeviceState
    var resident_lora_dev: KleinLoraDeviceSet
    if lokr_active:
        var dummy = build_klein_lora_set(1, 1, 8, 8, 1, Float32(1.0))
        dbl_state = lora_adamw_ot_device_state_init(dummy.dbl, ctx)
        sgl_state = lora_adamw_ot_device_state_init(dummy.sgl, ctx)
        resident_lora_dev = klein_lora_set_to_device_resident(
            dummy, dbl_state, sgl_state, ctx,
        )
    else:
        dbl_state = lora_adamw_ot_device_state_init(lora.dbl, ctx)
        sgl_state = lora_adamw_ot_device_state_init(lora.sgl, ctx)
        resident_lora_dev = klein_lora_set_to_device_resident(
            lora, dbl_state, sgl_state, ctx,
        )

    # ── T1.C levers optimizer state (default-off => lazily empty, no alloc) ──
    # One LeversOptimizerState per adapter list, mirroring klein's dbl/sgl
    # OT-state split; each covers its WHOLE list ([0, len)). NOTE: levers
    # optimizer state has no save/resume sidecar — resuming a levers-optimizer
    # run fails loud at the first step (levers.mojo RESUME contract).
    var lev_opt_dbl = LeversOptimizerState()
    var lev_opt_sgl = LeversOptimizerState()

    # ── EMA shadow params (Wave 2B item 2i; default-off => NO allocation) ─────
    # Shadow copies of every trainable LoRA A/B, host-side (LoRA params are host
    # List[BFloat16]). Updated post-AdamW with F32 math and BF16 storage. When
    # ema_enabled=False these Lists stay empty and the update loop is skipped =>
    # zero shadow allocation, baseline byte-unchanged.
    var ema_dbl_a = List[List[BFloat16]]()
    var ema_dbl_b = List[List[BFloat16]]()
    var ema_sgl_a = List[List[BFloat16]]()
    var ema_sgl_b = List[List[BFloat16]]()
    if cfg.ema_enabled:
        for i in range(len(lora.dbl)):
            ema_dbl_a.append(lora.dbl[i].a.copy())
            ema_dbl_b.append(lora.dbl[i].b.copy())
        for i in range(len(lora.sgl)):
            ema_sgl_a.append(lora.sgl[i].a.copy())
            ema_sgl_b.append(lora.sgl[i].b.copy())
        print("  EMA shadows: enabled inv_gamma=", cfg.ema_inv_gamma,
              " power=", cfg.ema_power, " max_decay=", cfg.ema_max_decay,
              " (", len(ema_dbl_a) + len(ema_sgl_a), "adapters)")
    var rope = _build_klein_rope_host()
    var cos = rope[0].copy()
    var sin = rope[1].copy()
    # RoPE tables are compute constants. The train stream is currently F32, and
    # ops.rope allows F32 tables as the diffusers Flux/Klein compute boundary.
    var cos_dev = TArc(Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.F32, ctx))
    var sin_dev = TArc(Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.F32, ctx))
    print("  rope host tables:", len(cos), "cos /", len(sin), "sin (expect", S * H * (Dh // 2), ")")

    # ── open cache ────────────────────────────────────────────────────────────
    var cache = KleinCache(cache_dir)
    print("  cache samples:", cache.count())
    var compatible = _compatible_cache_indices(cache, cfg, ctx)
    print("  cache compatible samples:", len(compatible), "of", cache.count(), "for", LH, "x", LW)
    print("  preloading compatible cache tensors:", len(compatible))
    var cached_img_tokens = List[TArc]()
    var cached_txt_tokens = List[TArc]()
    var preload_txt_sh = List[Int]()
    preload_txt_sh.append(N_TXT); preload_txt_sh.append(cfg.joint_attention_dim)
    for ci in range(len(compatible)):
        var sample = cache.load(compatible[ci], ctx)
        var img_tok = _latent_to_img_tokens_device(sample.latent, cfg.in_channels, ctx)
        var txt_tok = reshape(sample.text_embedding, preload_txt_sh.copy(), ctx)
        cached_img_tokens.append(TArc(img_tok^))
        cached_txt_tokens.append(TArc(txt_tok^))
    print("  preloaded cache tensors:", len(cached_img_tokens), "samples")

    # ── caption-dropout uncond embedding (Wave 2B item 2d; default-off) ───────
    # Default-off must not allocate an unconditional tensor. When enabled, use a
    # ZERO text-token tensor [N_TXT, joint_attention_dim] as the reproducible
    # uncond/empty-caption embedding until Klein precaches a real empty prompt.
    # Keep the carrier dtype at the cache/model input boundary; the Float32
    # zero list below is only a host scalar source for Tensor.from_host.
    var uncond_txt = Optional[TArc](None)
    if cfg.caption_dropout_prob > Float32(0.0):
        var uncond_dtype = STDtype.BF16
        if len(cached_txt_tokens) > 0:
            uncond_dtype = cached_txt_tokens[0][].dtype()
        uncond_txt = Optional[TArc](
            TArc(
                Tensor.from_host(
                    List[Float32]() if N_TXT * cfg.joint_attention_dim == 0 else _zeros(N_TXT * cfg.joint_attention_dim),
                    [N_TXT, cfg.joint_attention_dim], uncond_dtype, ctx,
                )
            )
        )
        print("  caption_dropout: prob=", cfg.caption_dropout_prob, " (uncond = zero embedding)")

    var scratch_fwd = ScratchRingAllocator(ctx, SCRATCH_FWD_SLAB_BYTES, SCRATCH_FWD_SLABS)
    var scratch_bwd = ScratchRingAllocator(ctx, SCRATCH_BWD_SLAB_BYTES, SCRATCH_BWD_SLABS)
    print(
        "  scratch fwd:", SCRATCH_FWD_SLAB_BYTES, "bytes x", SCRATCH_FWD_SLABS,
        "slabs; bwd:", SCRATCH_BWD_SLAB_BYTES, "bytes x", SCRATCH_BWD_SLABS, "slabs",
    )

    # ── gradient accumulation buffers (Wave 2B item 2h; default-off == 1) ─────
    # Each loop iteration is one MICRO-step (one sample). We accumulate (SUM) the
    # four AdamW-fed LoRA grad groups across `grad_accum_steps` micro-steps, then
    # MEAN (÷N) and run clip+AdamW once on accumulation boundaries. With
    # accum_steps=1 every step is a boundary, the buffer holds one grad, mean=÷1
    # => byte-identical to the current per-step path. Buffers are lazily sized on
    # the first micro-step of each accumulation window.
    var accum_steps = cfg.grad_accum_steps
    if accum_steps < 1:
        accum_steps = 1
    var use_grad_accum = accum_steps > 1
    var acc_dbl_a = List[List[Float32]]()
    var acc_dbl_b = List[List[Float32]]()
    var acc_sgl_a = List[List[Float32]]()
    var acc_sgl_b = List[List[Float32]]()
    var micro_in_window = 0
    if use_grad_accum:
        print("  grad accumulation: accum_steps=", accum_steps, " (mean over micro-steps)")

    # ── training loop ─────────────────────────────────────────────────────────
    var train_start = perf_counter_ns()
    for k in range(start_step + 1, run_steps + 1):
        scratch_fwd.reset()
        scratch_bwd.reset()
        var t0 = perf_counter_ns()
        if runtime_profile:
            print("PROG_STAGE step=", k, " total=", run_steps, " phase=load_sample_begin")

        # pick a sample (round-robin)
        var t_load0 = perf_counter_ns()
        var cache_slot = (k - 1) % len(cached_img_tokens)
        var latent_tokens_t = cached_img_tokens[cache_slot].copy()
        var txt_tokens_t = cached_txt_tokens[cache_slot].copy()
        # ── caption dropout (Wave 2B item 2d; default-off when prob<=0) ────────
        # Per-step Bernoulli on the same uniform draw as the Rust StdRng path.
        # When it fires, swap the conditional text tokens for the cached uncond
        # (zero) embedding. prob<=0 never draws => baseline byte-unchanged.
        # T1.D: routed through the shared levers caption_dropout_pick — the seed
        # expression (SEED_BASE * 31 + k) is IDENTICAL, behavior unchanged.
        if cfg.caption_dropout_prob > Float32(0.0):
            if caption_dropout_pick(UInt64(k), SEED_BASE, cfg.caption_dropout_prob):
                if not uncond_txt:
                    raise Error("caption_dropout enabled but uncond text tensor was not initialized")
                txt_tokens_t = uncond_txt.value().copy()
                if runtime_profile:
                    print("PROG_STAGE step=", k, " phase=caption_dropout dropped=1")
        var t_load1 = perf_counter_ns()
        if runtime_profile:
            print(
                "PROG_STAGE step=", k, " total=", run_steps, " phase=load_sample",
                " secs=", Float32(Float64(t_load1 - t_load0) / 1.0e9),
            )

        var n_img_vals = N_IMG * cfg.in_channels

        # ── (Wave 2D wire 1) timestep-distribution selector ───────────────────
        # Default field -1 => the production logit-normal+qwen-shift path,
        # BYTE-IDENTICAL to before this wave. 0=Uniform, 1=Sigmoid select the
        # alternative draws (same ChaCha12 stream, same seed derivation).
        var sigma_seed = SEED_BASE + UInt64(k)
        var sigma: Float32
        if cfg.timestep_distribution == TSD_UNIFORM:
            sigma = sample_timestep_uniform(sigma_seed)
        elif cfg.timestep_distribution == TSD_SIGMOID:
            sigma = sample_timestep_sigmoid(
                sigma_seed, cfg.timestep_noising_weight, cfg.timestep_noising_bias
            )
        else:
            # -1 (production default) and 2 (explicit LogitNormal) -> UNCHANGED path.
            sigma = sample_timestep_logit_normal(sigma_seed, cfg.timestep_shift)
        # ── (Wave 2D wire 2) timestep bias ────────────────────────────────────
        # Reshape the sampled sigma in [0,1] (total=1.0 for the flow-match sigma
        # path). Strategy 0=None => identity, sigma BYTE-UNCHANGED (default-off).
        sigma = apply_bias(
            sigma, Float32(1.0), cfg.timestep_bias_strategy,
            cfg.timestep_bias_multiplier,
            cfg.timestep_bias_range_min, cfg.timestep_bias_range_max,
        )
        if runtime_profile:
            print(
                "PROG_STAGE step=", k, " total=", run_steps, " phase=noise_begin",
                " sigma=", sigma, " elems=", n_img_vals,
            )
        var t_noise0 = perf_counter_ns()

        # flow-match in token space (GPU arithmetic — matches schedule.mojo math):
        #   x_t    = (1-sigma)*latent + sigma*noise
        #   target = noise - latent
        var noise = _host_noise(n_img_vals, SEED_BASE * UInt64(7919) + UInt64(k))
        # ── noise modifiers (Wave 2B item 2e; ALL default-off) ────────────────
        # offset noise + input perturbation applied IN PLACE on the host list
        # before upload. All-off (weights/gamma/iterations 0) leaves `noise`
        # byte-identical => pure-Gaussian baseline unchanged.
        var multires_skipped = apply_noise_modifiers_host(
            noise, N_IMG, cfg.in_channels,
            cfg.offset_noise_weight, cfg.offset_noise_prob,
            cfg.input_perturbation,
            cfg.multires_iterations, cfg.multires_discount,
            SEED_BASE * UInt64(7919) + UInt64(k),
        )
        if multires_skipped and runtime_profile:
            print("PROG_STAGE step=", k, " phase=multires_noise skipped=token_space_2d")
        var noise_t = Tensor.from_host(
            noise^, [N_IMG, cfg.in_channels], latent_tokens_t[].dtype(), ctx
        )
        var fm = flow_match_noise_target(latent_tokens_t[], sigma, noise_t, ctx)
        var x_t_dev = TArc(fm.x_t.clone(ctx))
        var target = _host_f32_for_step_math(fm.target, ctx)
        var t_noise1 = perf_counter_ns()
        var noise_secs = Float64(t_noise1 - t_noise0) / 1.0e9
        var noise_speed = Float64(n_img_vals) / noise_secs if noise_secs > 0.0 else Float64(0.0)
        if runtime_profile:
            print(
                "PROG_STAGE step=", k, " total=", run_steps, " phase=noise",
                " sigma=", sigma, " secs=", Float32(noise_secs),
                " elems_per_sec=", Float32(noise_speed),
            )

        # per-step modulation vecs from this sigma
        var mods = build_klein_step_mods_device_cached(
            mod_weights, sigma, cfg.timestep_dim, cfg.d_model, ctx
        )
        var img_mod = mods[0].copy()
        var txt_mod = mods[1].copy()
        var single_mod = mods[2].copy()
        # FIX: overwrite the resident base final-layer adaLN mod with THIS step's
        # sigma (was static sigma=0.5 — diffusers-parity localized the loss here).
        base.final_shift = mods[3].copy()
        base.final_scale = mods[4].copy()
        var lora_dev: KleinLoraDeviceSet
        if lokr_active:
            # LoKr: the host carrier set is re-materialized after every master
            # AdamW step, so it is re-uploaded per step (the pre-v2 path).
            lora_dev = klein_lora_set_to_device(lora, ctx)
        else:
            comptime if KLEIN_V2_ENGINE:
                lora_dev = resident_lora_dev.copy()
            else:
                lora_dev = klein_lora_set_to_device(lora, ctx)

        # forward -> loss/d_loss -> backward. Loss math is shared between the
        # resident checkpoint-input path and the CPU_OFFLOADED activation tape
        # path so the offload branch cannot drift numerically.
        var empty_img = List[Float32]()
        var empty_txt = List[Float32]()
        var loss: Float32
        var g: KleinLoraGrads
        var t_bwd0 = UInt(0)
        if runtime_profile:
            print("PROG_STAGE step=", k, " total=", run_steps, " phase=forward_begin")
        var t_fwd0 = perf_counter_ns()
        if use_activation_tape_offload:
            var fwd_tape = klein_stack_lora_forward_device_inputs_offload_turbo_moddev_rope_scratch_offloaded_tape[H, Dh, N_IMG, N_TXT, S](
                x_t_dev, txt_tokens_t, base,
                loader, lora_dev, img_mod, txt_mod, single_mod, cos_dev[], sin_dev[],
                cfg.d_model, cfg.mlp_hidden, cfg.in_channels, cfg.joint_attention_dim,
                cfg.out_channels, cfg.eps, ctx, scratch_fwd,
            )
            var t_fwd1 = perf_counter_ns()
            if runtime_profile:
                print(
                    "PROG_STAGE step=", k, " total=", run_steps, " phase=forward",
                    " secs=", Float32(Float64(t_fwd1 - t_fwd0) / 1.0e9),
                    " activation_tape_host_bytes=", fwd_tape.total_host_bytes(),
                )
            var lg = _klein_loss_grad(fwd_tape.out, target, sigma, cfg)
            loss = lg.loss
            if runtime_profile:
                print("PROG_STAGE step=", k, " total=", run_steps, " phase=backward_begin", " loss=", loss)
            t_bwd0 = perf_counter_ns()
            g = klein_stack_lora_backward_offloaded_tape_turbo_moddev_rope_scratch[H, Dh, N_IMG, N_TXT, S](
                lg.d_loss.copy(), empty_img.copy(), empty_txt.copy(), base,
                loader, lora_dev, img_mod, txt_mod, single_mod, cos_dev[], sin_dev[], fwd_tape,
                cfg.d_model, cfg.mlp_hidden, cfg.in_channels, cfg.joint_attention_dim,
                cfg.out_channels, cfg.eps, ctx, scratch_bwd, False, False,
            )
        else:
            var fwd = klein_stack_lora_forward_device_inputs_offload_turbo_moddev_rope_scratch[H, Dh, N_IMG, N_TXT, S](
                x_t_dev, txt_tokens_t, base,
                loader, lora_dev, img_mod, txt_mod, single_mod, cos_dev[], sin_dev[],
                cfg.d_model, cfg.mlp_hidden, cfg.in_channels, cfg.joint_attention_dim,
                cfg.out_channels, cfg.eps, ctx, scratch_fwd,
            )
            var t_fwd1 = perf_counter_ns()
            if runtime_profile:
                print(
                    "PROG_STAGE step=", k, " total=", run_steps, " phase=forward",
                    " secs=", Float32(Float64(t_fwd1 - t_fwd0) / 1.0e9),
                )
            var lg = _klein_loss_grad(fwd.out, target, sigma, cfg)
            loss = lg.loss
            if runtime_profile:
                print("PROG_STAGE step=", k, " total=", run_steps, " phase=backward_begin", " loss=", loss)
            t_bwd0 = perf_counter_ns()
            comptime if KLEIN_V2_GRAPH_PATH:
                # P6: per-block graph-engine backward (same conductor loop,
                # same scratch ring, same arg list — drop-in for the
                # hand-chain call below; bit gate = klein_block_parity).
                # T2.G: LoKr keeps the C13-gated hand-chain (per-step fresh
                # carrier device sets; the graph arm is validated on the
                # resident plain-LoRA set only).
                if lokr_active:
                    g = klein_stack_lora_backward_offload_turbo_moddev_rope_scratch[H, Dh, N_IMG, N_TXT, S](
                        lg.d_loss.copy(), empty_img.copy(), empty_txt.copy(), base,
                        loader, lora_dev, img_mod, txt_mod, single_mod, cos_dev[], sin_dev[], fwd,
                        cfg.d_model, cfg.mlp_hidden, cfg.in_channels, cfg.joint_attention_dim,
                        cfg.out_channels, cfg.eps, ctx, scratch_bwd, False, False,
                    )
                else:
                    g = klein_stack_lora_backward_graph[H, Dh, N_IMG, N_TXT, S](
                        lg.d_loss.copy(), empty_img.copy(), empty_txt.copy(), base,
                        loader, lora_dev, img_mod, txt_mod, single_mod, cos_dev[], sin_dev[], fwd,
                        cfg.d_model, cfg.mlp_hidden, cfg.in_channels, cfg.joint_attention_dim,
                        cfg.out_channels, cfg.eps, ctx, scratch_bwd, False, False,
                    )
            else:
                g = klein_stack_lora_backward_offload_turbo_moddev_rope_scratch[H, Dh, N_IMG, N_TXT, S](
                    lg.d_loss.copy(), empty_img.copy(), empty_txt.copy(), base,
                    loader, lora_dev, img_mod, txt_mod, single_mod, cos_dev[], sin_dev[], fwd,
                    cfg.d_model, cfg.mlp_hidden, cfg.in_channels, cfg.joint_attention_dim,
                    cfg.out_channels, cfg.eps, ctx, scratch_bwd, False, False,
                )
        var t_bwd1 = perf_counter_ns()
        if runtime_profile:
            print(
                "PROG_STAGE step=", k, " total=", run_steps, " phase=backward",
                " secs=", Float32(Float64(t_bwd1 - t_bwd0) / 1.0e9),
            )

        # ── gradient accumulation (Wave 2B item 2h; default-off when N==1) ─────
        # Fast path: accum_steps==1 uses `g` directly and avoids zero-cloning,
        # accumulating, scaling by 1, and copying buffers back into `g`.
        if use_grad_accum:
            # Treat this loop iteration as one micro-step. SUM its four LoRA
            # grad groups into the window buffers; on the accumulation boundary
            # MEAN (÷N) and copy the meaned grads back into `g`.
            if micro_in_window == 0:
                acc_dbl_a = zeros_like_group(g.dbl_d_a)
                acc_dbl_b = zeros_like_group(g.dbl_d_b)
                acc_sgl_a = zeros_like_group(g.sgl_d_a)
                acc_sgl_b = zeros_like_group(g.sgl_d_b)
            accumulate_grad_group(acc_dbl_a, g.dbl_d_a)
            accumulate_grad_group(acc_dbl_b, g.dbl_d_b)
            accumulate_grad_group(acc_sgl_a, g.sgl_d_a)
            accumulate_grad_group(acc_sgl_b, g.sgl_d_b)
            micro_in_window += 1
            var is_boundary = micro_in_window >= accum_steps or k == run_steps
            if not is_boundary:
                # mid-window: skip optimizer this micro-step, keep accumulating.
                if runtime_profile:
                    print("PROG_STAGE step=", k, " phase=grad_accum micro=", micro_in_window, "/", accum_steps)
                var t1m = perf_counter_ns()
                var secsm = Float64(t1m - t0) / 1.0e9
                print_trainer_progress(
                    String("Klein-lora"),
                    k, cfg.max_steps, len(compatible), loss, 0.0, secsm, noise_speed,
                    Float64(t1m - train_start) / 1.0e9,
                )
                var pending_optimizer_step = ((k - 1) // accum_steps) + 1
                var pending_lr = ot_lr_for_optimizer_step(cfg, pending_optimizer_step)
                board.log_train_step(k, loss, 0.0, pending_lr, secsm, noise_speed)
                continue
            # boundary: MEAN the window then overwrite g's four groups with it.
            var inv_micro = Float32(1.0) / Float32(micro_in_window)
            scale_grad_group(acc_dbl_a, inv_micro)
            scale_grad_group(acc_dbl_b, inv_micro)
            scale_grad_group(acc_sgl_a, inv_micro)
            scale_grad_group(acc_sgl_b, inv_micro)
            for i in range(len(g.dbl_d_a)):
                g.dbl_d_a[i] = acc_dbl_a[i].copy()
                g.dbl_d_b[i] = acc_dbl_b[i].copy()
            for i in range(len(g.sgl_d_a)):
                g.sgl_d_a[i] = acc_sgl_a[i].copy()
                g.sgl_d_b[i] = acc_sgl_b[i].copy()
            micro_in_window = 0

        # grad_norm = L2 of ALL LoRA d_A/d_B
        var gsum = 0.0
        var nd = cfg.num_double * DBL_SLOTS
        for i in range(nd):
            var a = _l2(g.dbl_d_a[i]); var b = _l2(g.dbl_d_b[i])
            gsum += a * a + b * b
        var ns = cfg.num_single * 2
        for i in range(ns):
            var a = _l2(g.sgl_d_a[i]); var b = _l2(g.sgl_d_b[i])
            gsum += a * a + b * b
        var grad_norm = sqrt(gsum)

        # ── dead-adapter warn (project's #1 silent failure) ───────────────────
        # B legitimately starts at 0, so its grad can be ~0 early; warn when an
        # adapter's TOTAL |d_A|+|d_B| == 0 at step>=1 (a truly dead branch).
        # (LoKr: untargeted slots are deliberate zero carriers — warn suppressed.)
        for i in range(nd):
            if (not lokr_active) and _abs_sum(g.dbl_d_a[i]) + _abs_sum(g.dbl_d_b[i]) == 0.0:
                print("[Klein-lora] dead_adapter step=", k, " idx=", i, " kind=double")
        for i in range(ns):
            if (not lokr_active) and _abs_sum(g.sgl_d_a[i]) + _abs_sum(g.sgl_d_b[i]) == 0.0:
                print("[Klein-lora] dead_adapter step=", k, " idx=", nd + i, " kind=single")

        # ── global-norm grad clip across ALL LoRA grads (EDv2 default-ON) ──────
        # (LoKr clips on the MASTER grads inside its optimizer branch below —
        # carrier grads pass through unclipped, mirroring ST clipping the
        # actual lycoris trainables.)
        var clip_scale = Float32(1.0)
        if (not lokr_active) and grad_norm > Float64(cfg.max_grad_norm):
            clip_scale = cfg.max_grad_norm / Float32(grad_norm)
            for i in range(nd):
                _scale_inplace(g.dbl_d_a[i], clip_scale)
                _scale_inplace(g.dbl_d_b[i], clip_scale)
            for i in range(ns):
                _scale_inplace(g.sgl_d_a[i], clip_scale)
                _scale_inplace(g.sgl_d_b[i], clip_scale)

        # AdamW step (on clipped grads)
        if runtime_profile:
            print("PROG_STAGE step=", k, " total=", run_steps, " phase=optim_begin")
        var t_optim0 = perf_counter_ns()
        # Wave 2A item 2a: scheduled lr. Default-off (lr_scheduler=0 Constant +
        # lr_warmup_steps=0) returns cfg.lr for every step => baseline unchanged.
        var optimizer_step = ((k - 1) // accum_steps) + 1
        var step_lr = ot_lr_for_optimizer_step(cfg, optimizer_step)
        # T1.C optimizer lever (default-off; C13: optimizer=ADAMW routes to the
        # existing literal fused calls below, untouched). Active path: HOST
        # adafactor/schedule-free step on the dbl+sgl host a/b mirrors (the
        # mirrors are fresh — at step 1 they are the built/resumed params, and
        # afterwards the levers writeback itself is the only writer because
        # the resident OT kernel does not run on this branch), then — on the
        # KLEIN_V2_ENGINE resident path — push the stepped params into the
        # live OT dev_p buffers so the device LoRA sub-buffer views see them
        # next step (levers_optimizer_sync_resident_ot = the inverse of the
        # resident step's P readback). On the non-resident path no sync is
        # needed: the next loop iteration re-uploads the host set via
        # klein_lora_set_to_device.
        if lokr_active:
            # ── T2.G LoKr optimizer path ──────────────────────────────────────
            # 1) chain carrier grads → master grads (exact bilinear chain rule,
            #    gated vs the upstream-parity lokr_backward in lokr_st_parity);
            # 2) global-norm clip on the MASTERS (the actual trainables);
            # 3) host AdamW on the masters (cfg betas/eps/wd, scheduled lr);
            # 4) re-materialize the carriers — next step re-uploads them.
            var mg = klein_lokr_chain_all(
                lokr_masters, g.dbl_d_a, g.dbl_d_b, g.sgl_d_a, g.sgl_d_b
            )
            var mnorm = klein_lokr_grad_norm(mg)
            grad_norm = mnorm  # logged value = master grad norm for LoKr
            if mnorm > Float64(cfg.max_grad_norm):
                clip_scale = cfg.max_grad_norm / Float32(mnorm)
                klein_lokr_clip_grads(mg, clip_scale)
            klein_lokr_adamw_step(
                lokr_masters, mg, optimizer_step, step_lr,
                cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
            )
            var carriers_k = klein_lokr_carrier_lists(
                lokr_masters, cfg.d_model, cfg.mlp_hidden
            )
            lora = KleinLoraSet(
                carriers_k[0].copy(), carriers_k[1].copy(),
                cfg.num_double, cfg.num_single, cfg.lora_rank,
            )
            print(
                "[Klein-lokr] step=", k,
                " master_grad_norm=", Float32(mnorm),
                " factor_l1=", klein_lokr_trainable_l1(lokr_masters),
                " zero_leg_l1=", klein_lokr_zero_leg_l1(lokr_masters),
            )
        elif levers_optimizer_active(cfg):
            levers_optimizer_step_host(
                cfg, lora.dbl, g.dbl_d_a, g.dbl_d_b, optimizer_step,
                step_lr, 0, len(lora.dbl), lev_opt_dbl,
            )
            levers_optimizer_step_host(
                cfg, lora.sgl, g.sgl_d_a, g.sgl_d_b, optimizer_step,
                step_lr, 0, len(lora.sgl), lev_opt_sgl,
            )
            comptime if KLEIN_V2_ENGINE:
                levers_optimizer_sync_resident_ot(dbl_state, lora.dbl, ctx)
                levers_optimizer_sync_resident_ot(sgl_state, lora.sgl, ctx)
        else:
            comptime if KLEIN_V2_ENGINE:
                klein_lora_adamw_step_resident(
                    dbl_state, sgl_state, lora, g, optimizer_step, step_lr, ctx,
                    cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
                )
            else:
                klein_lora_adamw_step(
                    lora, g, optimizer_step, step_lr, ctx, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay
                )
        # ── EMA shadow update post-AdamW (Wave 2B item 2i; default-off skip) ──
        # decay schedule returns 0.0 before update_after_step (skip). Off =>
        # shadows empty => loop body never runs => baseline unchanged.
        if cfg.ema_enabled:
            var ema_decay = ema_decay_at_step(
                k, cfg.ema_update_after_step, cfg.ema_inv_gamma,
                cfg.ema_power, cfg.ema_min_decay, cfg.ema_max_decay,
            )
            if ema_decay > Float32(0.0):
                for i in range(len(lora.dbl)):
                    ema_update_host(ema_dbl_a[i], lora.dbl[i].a, ema_decay)
                    ema_update_host(ema_dbl_b[i], lora.dbl[i].b, ema_decay)
                for i in range(len(lora.sgl)):
                    ema_update_host(ema_sgl_a[i], lora.sgl[i].a, ema_decay)
                    ema_update_host(ema_sgl_b[i], lora.sgl[i].b, ema_decay)
        var t_optim1 = perf_counter_ns()
        if runtime_profile:
            print(
                "PROG_STAGE step=", k, " total=", run_steps, " phase=optim",
                " secs=", Float32(Float64(t_optim1 - t_optim0) / 1.0e9),
            )

        var t1 = perf_counter_ns()
        var secs = Float64(t1 - t0) / 1.0e9

        # machine-parseable progress line (consumed by the tqdm wrapper).
        # clip=<scale> is 1.0 when no clip applied, else MAX_GRAD_NORM/grad_norm.
        if MACHINE_PROGRESS_LOG:
            print(
                "PROG step=", k, " total=", run_steps, " loss=", loss,
                " grad=", Float32(grad_norm), " lr=", step_lr, " clip=", clip_scale,
                " secs=", Float32(secs),
            )
        print_trainer_progress(
            String("Klein-lora"),
            k, cfg.max_steps, len(compatible), loss, grad_norm, secs, noise_speed,
            Float64(t1 - train_start) / 1.0e9,
        )
        board.log_train_step(k, loss, grad_norm, step_lr, secs, noise_speed)
        # ── cadence ───────────────────────────────────────────────────────────
        var save_due = klein_should_save_checkpoint(cfg, k)
        var sample_due = (
            runtime_sample_enabled
            and should_sample_completed_step(sample_cadence, k)
            and run_steps >= sample_every
        )
        var chunk_final_due = k == run_steps
        if save_due or sample_due or chunk_final_due:
            # T1.C schedule-free SAVE BRACKET (levers.mojo SAVE CONTRACT):
            # eval mode for the save + validation sample, train mode after.
            # No-op for every other optimizer, including the default AdamW.
            levers_optimizer_eval_for_save(cfg, lev_opt_dbl)
            levers_optimizer_eval_for_save(cfg, lev_opt_sgl)
            var ckpt = _lora_path_for_step(output_lora_path, k, cfg.max_steps)
            if lokr_active:
                # T2.G: LoKr product file in the upstream lycoris key
                # convention (lycoris_<module>.lokr_w1[_a/_b]/lokr_w2[_a/_b]/
                # .alpha). No optimizer-state sidecar this wave (LoKr resume
                # is not wired — fail-loud at startup).
                var nmods = save_klein_lokr(lokr_masters, ckpt, ctx)
                print("[Klein-lokr] save step=", k, " path=", ckpt, " modules=", nmods)
                board.log_text(String("events/save"), k, ckpt)
            else:
                var npairs = save_klein_lora(lora, ckpt, ctx)
                print("[Klein-lora] save step=", k, " path=", ckpt, " pairs=", npairs)
                board.log_text(String("events/save"), k, ckpt)
                var state_path = _state_path_for_lora(ckpt)
                comptime if KLEIN_V2_ENGINE:
                    # Levers optimizers never run the resident OT kernel, so the
                    # device M/V are stale init values — skip the pull (the saved
                    # state's AdamW moments are NOT levers state; levers resume
                    # fails loud at the first step by contract).
                    if not levers_optimizer_active(cfg):
                        lora_adamw_ot_device_state_sync_moments(dbl_state, lora.dbl, ctx)
                        lora_adamw_ot_device_state_sync_moments(sgl_state, lora.sgl, ctx)
                var nstate = save_klein_lora_state(lora, state_path, ctx)
                print("[Klein-lora] save_state step=", k, " path=", state_path, " pairs=", nstate)
                board.log_text(String("events/save_state"), k, state_path)
            # ── EMA shadow checkpoint (Wave 2B item 2i) ──────────────────────
            # When ema_enabled, the EMA shadow is the smoothed weight average and
            # is the checkpoint you sample/deploy from. Write it as a SIBLING file
            # so the live (training-state) checkpoint above is untouched. Off =>
            # shadows empty => this block is skipped => baseline byte-unchanged.
            if cfg.ema_enabled:
                var ema_path = _ema_path_for_lora(ckpt)
                var nema = save_klein_lora_ema(
                    lora, ema_dbl_a, ema_dbl_b, ema_sgl_a, ema_sgl_b, ema_path, ctx
                )
                print("[Klein-lora] save_ema step=", k, " path=", ema_path, " pairs=", nema)
                board.log_text(String("events/save_ema"), k, ema_path)
            if sample_due:
                _do_sample_all(cfg, sample_cfg, k, ckpt, board, ctx)
            # Do not reload PEFT LoRA into the live trainer here. PEFT files do
            # not carry AdamW moments, so reloading them mid-run resets optimizer
            # state and hurts convergence. Resume checks belong to the external
            # cadence supervisor or dedicated parity smokes.
            levers_optimizer_train_after_save(cfg, lev_opt_dbl)
            levers_optimizer_train_after_save(cfg, lev_opt_sgl)

    print("")
    print("DONE: worker reached step", run_steps, "of", cfg.max_steps, "target")
    board.set_status(String("complete"))
    board.close()
