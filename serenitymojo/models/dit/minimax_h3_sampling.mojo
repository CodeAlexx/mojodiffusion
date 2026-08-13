# serenitymojo/models/dit/minimax_h3_sampling.mojo
#
# MiniMax-H3 t2va DEVICE-side sampling scaffold: the dual-schedule (video
# shift 12.0 / audio shift 3.0) Euler loop and the packed-sequence layout
# builder (per-row token_tags / timestep_indices / (t,h,w) position_ids).
#
# ─────────────────────────────────────────────────────────────────────────────
# ORACLE BOUNDARY. This is the RUNTIME twin of two ALREADY-GATED host-float32
# oracles:
#
#   models/minimax_h3/scheduler.mojo   (22/22 bit-exact vs diffusers, full
#                                        trajectories at shifts 12.0 and 3.0)
#   models/minimax_h3/packing.mojo     (63/63 bit-exact vs diffusers, incl.
#                                        the 1344x768/124-frame product shape)
#
# Per this port's own convention (models/dit/minimax_h3_dit.mojo header:
# "block_forward.mojo is the host-float32 ORACLE this is gated against ...
# Nothing here may call it"), this file does NOT import scheduler.mojo or
# packing.mojo — every function below independently REPRODUCES their math
# (same formulas, same evaluation order, same named traps) rather than calling
# it. `models/dit/minimax_h3_dit.py::minimax_h3_adaln_row(s)` is the existing
# precedent for this exact split: it duplicates `models/minimax_h3/
# dit_frontend.mojo::minimax_h3_adaln_indices` byte-for-byte rather than
# importing it, for the same reason. Parity against the two oracles is proven
# in `models/dit/parity/minimax_h3_sampling_probe.mojo` by loading the SAME
# safetensors fixtures the oracle gates already consume
# (output/minimax_h3_scheduler/scheduler_ref.safetensors,
# output/minimax_h3_packing/packing_ref.safetensors) and comparing bit-exact —
# so the probe never calls the oracle module either, only its data.
#
# ─────────────────────────────────────────────────────────────────────────────
# THE DUAL SCHEDULE.
#
# H3 predicts video and audio on separate rectified-flow schedules that share
# ONE progress grid: `linspace(1, 0, N)` (torch's both-ends-fill semantics,
# `_linspace_1_to_0_sampling` below) is shifted through the SAME exponential
# formula `sigma' = s*sigma/(1+(s-1)*sigma)` twice, once with s=12.0 (video)
# and once with s=3.0 (audio). Concretely: the "video sigma" and "audio sigma"
# at step i are both DERIVED from the same base[i] via that one time-shift
# function, just with a different shift constant — this is what
# `MiniMaxH3DualSchedule.set_timesteps` builds explicitly (one shared `base`
# List, fed to two independent `MiniMaxH3EulerSchedule` instances). One
# forward call therefore carries (at least) two distinct row timesteps, which
# is why `models/minimax_h3/fp8_policy.mojo::minimax_h3_adaln_distinct_
# timesteps` sizes the AdaLN modulation cache at `2*steps` rows for a
# conditioning-free t2va run, not `steps` — confirmed against that file's own
# accounting (video_timestep and audio_timestep both "varies per step, its
# OWN schedule", fp8_policy.mojo:210-214).
#
# On "the audio velocity is scaled by d(sigma_a)/d(sigma_v)": every row in the
# packed sequence is conditioned (via AdaLN) on ITS OWN modality's timestep —
# `minimax_h3_adaln_rows` addresses the table by `timestep_index*3+tag`, and
# `models/minimax_h3/packing.mojo::minimax_h3_build_row_timesteps` assigns
# audio rows the AUDIO schedule's timestep, never the video one. So the
# transformer's audio-row output is already expressed in the audio schedule's
# OWN units, and `MiniMaxH3EulerSchedule.step_device` applied to the AUDIO
# instance with the AUDIO (sigma, sigma_next) pair reproduces the correct
# audio update with no separate derivative kernel: the `sigma_from_timestep`
# factor multiplying that row's velocity IS the audio schedule's own sigma,
# so the chain-rule scaling is realized implicitly by using two independent
# per-modality schedules rather than needing an explicit d(sigma_a)/d(sigma_v)
# tensor op. This reasoning is NOT independently confirmed against a live
# combined video+audio diffusers sampling-loop trace — no such fixture exists
# in this repo yet (grepped; only the per-modality scalar scheduler oracle
# does) — flagged in this port's HANDOFF as an open item for whoever wires the
# first end-to-end t2va sample.
#
# ─────────────────────────────────────────────────────────────────────────────
# NUMERICAL TRAPS reproduced from the oracle, do not "clean up":
#
#   * Mojo contracts `a + b*c` into a fused multiply-add. Torch runs the
#     multiply and the add as separate kernels with a materialized temporary
#     between them — a fused form misses the reference trajectory by 1 ulp on
#     67 of 464 values (scheduler.mojo:52-67). This file's `_sched_round_f32`
#     forces one float32 rounding per elementary op — but NOT via the
#     oracle's original `@no_inline`-identity-function trick, which MEASURED
#     does not hold at -O3 (this toolchain's DEFAULT optimization level: a
#     bare `mojo run`/`mojo build` with no explicit `-O`). See
#     `_sched_round_f32`'s own docstring for the measured mechanism (an
#     identity function is fusable via its CALL SITE's `contract` fast-math
#     flag regardless of `@no_inline`) and why `std.benchmark.black_box`'s
#     real inline-asm memory-clobber is used instead — proven bit-identical
#     at -O0/-O2/-O3 by `models/dit/parity/minimax_h3_sampling_probe.mojo`.
#     The DEVICE blend below (`MiniMaxH3EulerSchedule.step_device`) does not
#     need this trick: each elementary op is its OWN `ops/tensor_algebra`
#     call, i.e. its own GPU kernel launch with a real device-memory round
#     trip in between (`mul_scalar` then `add`, never fused into one
#     expression) — the same materialize-between-ops discipline
#     `ops/torch_bf16.mojo` already uses for exactly this reason (see e.g.
#     `torch_bf16_eager_add_scaled`, which stages the scaled product through
#     its OWN buffer before the add). A real kernel-launch boundary on a
#     SEPARATE (NVPTX) compilation pipeline is not exposed to the host -O3
#     risk above at all — `step_device`'s only exposure to it is indirectly,
#     through the HOST-computed `sigma`/`ratio` scalars `advance()` reads out
#     of `self.sigmas` (built by the shift formula this same fix protects).
#   * `torch.linspace` fills from BOTH ends with an FMA — `_linspace_1_to_0_
#     sampling` mirrors the halfway split exactly (scheduler.mojo:70-93). The
#     `fma()` HERE is an explicit intrinsic call, not compiler-introduced
#     contraction, and MEASURED bit-identical at -O0/-O2/-O3 as-is — it does
#     not need the `_sched_round_f32` treatment.
#
# ─────────────────────────────────────────────────────────────────────────────
# DTYPE RULE (repo-wide, re-affirmed here because a prior port — NAVA —
# violated it and produced a visibly distorted decode while every single-step
# gate still read cos ~0.9998): the ACCUMULATING latent that carries across
# sampling steps stays F32 the entire loop. It is cast to BF16 only at the
# moment it is fed to the block stack, and that cast MUST go through
# `ops/torch_bf16.torch_f32_to_bf16_rne` (see `minimax_h3_prepare_model_
# input` below), never Mojo's native `.cast[DType.bfloat16]()` — the two
# differ by ~1 BF16 quantum on some values, invisible on one cast, and
# compounding + decorrelating from torch over N steps.
#
# H3's own checkpoint makes the OTHER half of this easy: `proj_out` /
# `audio_proj_out` (the velocity output heads) are two of the twelve tensors
# the checkpoint itself stores in FP32 (models/minimax_h3/fp8_policy.mojo:
# 110-117, F32_KEEP class) — so `model_output` arriving at
# `MiniMaxH3EulerSchedule.step_device` is ALREADY F32-native, no upcast
# needed on that side. The whole Euler blend below is therefore F32 tensors
# in and out, matching the oracle's own all-F32 scalar math exactly.
#
# ─────────────────────────────────────────────────────────────────────────────
# REUSE (do NOT reimplement, imported below):
#   models/dit/minimax_h3_dit.mojo :: MiniMaxH3DiTConfig, minimax_h3_released_
#     config, minimax_h3_adaln_rows — the config and the already-built AdaLN
#     row indexer this file's (token_tags, timestep_indices) feed.
#   ops/patchify3d.mojo :: patchify3d — H3's video patch-embed unfold
#     (`minimax_h3_patchify_video` below is a one-line named wrapper pinning
#     H3's fixed (1,2,2) patch and the config's channel count).
#   ops/torch_bf16.mojo :: torch_f32_to_bf16_rne — the model-input cast.
#   ops/tensor_algebra.mojo :: add, mul_scalar — the device Euler blend's
#     elementary ops (each already its own kernel launch).
#   serenitymojo/tensor.mojo :: Tensor.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.collections import List
from std.math import fma, sqrt
from max.gpu.host import DeviceContext
from std.benchmark import black_box

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import add, mul_scalar
from serenitymojo.ops.torch_bf16 import torch_f32_to_bf16_rne
from serenitymojo.ops.patchify3d import patchify3d
from serenitymojo.models.dit.minimax_h3_dit import (
    MiniMaxH3DiTConfig,
    minimax_h3_released_config,
    minimax_h3_adaln_rows,
)

# ── Constants (mirrored, not imported — see header) ──────────────────────────
# Released checkpoint shifts (model_index.json `sigma_shift_scales`), same
# source as scheduler.mojo:46-49.
comptime MINIMAX_H3_VIDEO_SHIFT = Float32(12.0)
comptime MINIMAX_H3_AUDIO_SHIFT = Float32(3.0)

# Modality tags, same source as packing.mojo:52-56 (a checkpoint contract: they
# index the AdaLN table).
comptime MINIMAX_H3_VIDEO_TAG = 0
comptime MINIMAX_H3_TEXT_TAG = 1
comptime MINIMAX_H3_AUDIO_TAG = 2

# Keyframe anchor codes, same source as packing.mojo:84-85.
comptime MINIMAX_H3_ANCHOR_FIRST = 0
comptime MINIMAX_H3_ANCHOR_LAST = 1

# Audio is channel-major stereo, same source as packing.mojo:75.
comptime MINIMAX_H3_AUDIO_CHANNELS = 2

# Rotary-time constants, same source as packing.mojo:80-81.
comptime ROPE_FRAME_RESCALE = Float64(5.0) / Float64(3.0)
comptime ROPE_SPATIAL_SCALE = Float64(32.0)

# numpy's pairwise-summation block size, same source as packing.mojo:88.
comptime _PW_BLOCKSIZE = 128

# H3's released video patch: (1, 2, 2) over 24-channel latents
# (models/dit/minimax_h3_dit.mojo :: MiniMaxH3DiTConfig.video_patch_dim).
comptime MINIMAX_H3_VIDEO_PATCH_F = 1
comptime MINIMAX_H3_VIDEO_PATCH_H = 2
comptime MINIMAX_H3_VIDEO_PATCH_W = 2


# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — the dual-schedule Euler loop.
# Mirrors models/minimax_h3/scheduler.mojo. Modality-agnostic, like the
# oracle: MiniMaxH3EulerSchedule is used for BOTH video (shift 12.0) and audio
# (shift 3.0); MiniMaxH3DualSchedule composes one of each from a shared base
# grid.
# ═════════════════════════════════════════════════════════════════════════════


def _sched_round_f32(var x: Float32) -> Float32:
    """Optimization barrier forcing one float32 rounding, so the compiler
    cannot contract a neighbouring `a + b*c` into a fused multiply-add.
    Mirrors scheduler.mojo::_round_f32's INTENT (67/464 mismatched values on
    `video_30` with a fully fused form) but NOT its `@no_inline`-identity-
    function mechanism, which MEASURED does not hold at -O3 (this toolchain's
    DEFAULT: plain `mojo run`/`mojo build` with no explicit `-O`).

    Evidence (`--emit llvm` on a standalone repro): the call site is tagged
    `call contract float @"..."` at BOTH -O0 and -O3 with an IDENTICAL total
    `contract`-flag count (27 at each) — Mojo's front end treats a provably-
    `readnone` identity function (`_round_f32(x) { return x }`) as fusable
    with surrounding arithmetic regardless of `@no_inline`, which only blocks
    code duplication, not this value-forwarding. Only -O3's more aggressive
    SelectionDAG/MachineCombiner passes actually exploit that permission into
    real hardware FMA; -O0 does not, which is why the old barrier "worked" at
    -O0. Confirmed with an unwrapped ("naive") sibling expression: it produced
    BIT-IDENTICAL output to the `@no_inline`-protected version at every
    optimization level tested — the old barrier was doing nothing.

    `std.benchmark.black_box` is a REAL inline-asm memory-clobber
    (`inlined_assembly[..., constraints="r,~{memory}", has_side_effect=True]`)
    rather than a function the optimizer can prove is pure identity, so it
    cannot be reasoned through the same way. MEASURED bit-identical at -O0,
    -O2, AND -O3 on both this file's shift formula (`build_from_base` below)
    and this function's own callers (`step_host`/`step_device`'s Euler
    blend) — see models/dit/parity/minimax_h3_sampling_probe.mojo, 38/38
    at all three levels after this change (was 24/38 at -O3 before it, with
    the divergence appearing in exactly the same values the upstream oracle's
    OWN gate diverges on at -O3 — this is not a new bug, it is the
    `@no_inline` trick's pre-existing hole, reproduced faithfully)."""
    return black_box(x)


def _linspace_1_to_0_sampling(steps: Int) -> List[Float32]:
    """`torch.linspace(1.0, 0.0, steps, dtype=float32)`, CPU semantics: fills
    from BOTH ends around `halfway = steps // 2` with a fused multiply-add
    (one rounding per element). Mirrors scheduler.mojo::_linspace_1_to_0
    (lines 70-93) exactly — verified there against torch at 2, 8, 29, 30, 50,
    100, 400 steps."""
    var start = Float32(1.0)
    var end = Float32(0.0)
    var step = (end - start) / Float32(steps - 1)
    var halfway = steps // 2
    var out = List[Float32]()
    for i in range(steps):
        if i < halfway:
            out.append(fma(step, Float32(i), start))
        else:
            out.append(fma(-step, Float32(steps - i - 1), end))
    return out^


struct MiniMaxH3EulerSchedule(Movable):
    """Rectified-flow Euler schedule with an exponential sigma shift — one
    modality's half of the dual schedule. Mirrors scheduler.mojo::
    MiniMaxH3Scheduler exactly (fields, `set_timesteps`, `index_for_timestep`,
    the step() bookkeeping); `step_host` mirrors `step()`'s elementwise blend
    on host Lists (probe/self-check use only), `step_device` is the
    production Tensor path. Both funnel through `advance` so they can never
    disagree on which (sigma, sigma_next) pair a given call used.

    `step_index`/`begin_index` use -1 for the reference's `None`, matching the
    oracle."""

    var shift: Float32
    var sigmas: List[Float32]
    var timesteps: List[Float32]
    var step_index: Int
    var begin_index: Int

    def __init__(out self, shift: Float32) raises:
        if shift <= 0.0:
            raise Error("MiniMax-H3 sampling schedule: `shift` must be positive")
        self.shift = shift
        self.sigmas = List[Float32]()
        self.timesteps = List[Float32]()
        self.step_index = -1
        self.begin_index = -1

    def set_begin_index(mut self, begin_index: Int):
        self.begin_index = begin_index

    def build_from_base(mut self, base: List[Float32]) raises:
        """Shift + collapse + derive timesteps from a PRECOMPUTED base
        linspace grid — the shared-progress-grid half of the dual schedule
        (see file header). Mirrors the second half of scheduler.mojo::
        MiniMaxH3Scheduler.set_timesteps (lines 137-158) exactly; `set_
        timesteps` below is a thin wrapper that builds `base` itself for
        standalone (single-modality) use."""
        var shifted = List[Float32]()
        var shift_minus_one = self.shift - Float32(1.0)
        for i in range(len(base)):
            var b = base[i]
            # Same -O3 contraction risk as the step() blend below, on
            # `1.0 + shift_minus_one*b` — MEASURED to only manifest when
            # `shift - 1` is not an exact power of two (12.0 -> 11.0,
            # affected; 3.0 -> 2.0, exact power of two, nothing to fuse —
            # why the audio schedule was fine even unprotected while the
            # video one was not). See `_sched_round_f32`'s docstring.
            var num = _sched_round_f32(self.shift * b)
            var denom_term = _sched_round_f32(shift_minus_one * b)
            var denom = _sched_round_f32(Float32(1.0) + denom_term)
            shifted.append(num / denom)

        # Collapse float32 collisions the shift can create near sigma=1
        # (`torch.unique_consecutive`).
        var sigmas = List[Float32]()
        for i in range(len(shifted)):
            if i == 0 or shifted[i] != shifted[i - 1]:
                sigmas.append(shifted[i])

        var timesteps = List[Float32]()
        for i in range(len(sigmas) - 1):
            timesteps.append(Float32(1.0) - sigmas[i])

        self.sigmas = sigmas^
        self.timesteps = timesteps^
        self.step_index = -1
        self.begin_index = -1

    def set_timesteps(mut self, num_inference_steps: Int) raises:
        """Standalone entry point (builds its own base grid). Mirrors
        scheduler.mojo::MiniMaxH3Scheduler.set_timesteps exactly."""
        if num_inference_steps < 2:
            raise Error(
                "MiniMax-H3 sampling schedule: `num_inference_steps` must be >= 2"
            )
        self.build_from_base(_linspace_1_to_0_sampling(num_inference_steps))

    def num_inference_steps(self) -> Int:
        """Model evaluations the schedule drives — one fewer than the sigmas."""
        return len(self.timesteps)

    def index_for_timestep(self, timestep: Float32) raises -> Int:
        """Map a timestep value to its index. The schedule is strictly
        increasing in t, so the match is unique."""
        for i in range(len(self.timesteps)):
            if self.timesteps[i] == timestep:
                return i
        raise Error(
            "MiniMax-H3 sampling schedule: timestep is not in `timesteps`; pass"
            " a value taken from the schedule"
        )

    def advance(mut self, timestep: Float32) raises -> Tuple[Float32, Float32]:
        """Bookkeeping-only half of the oracle's `step()`: resolves
        `step_index` (via `index_for_timestep` on the first call, mirroring
        `begin_index is None` in the reference), checks the schedule is not
        exhausted, advances `step_index`, and returns
        `(sigma_from_timestep, ratio)` for the caller's own elementwise blend
        (scheduler.mojo::step, lines 206-226, minus the per-element loop).
        `step_host`/`step_device` below share this so the host self-check and
        the device production path can never use different sigmas."""
        if self.step_index < 0:
            if self.begin_index < 0:
                self.step_index = self.index_for_timestep(timestep)
            else:
                self.step_index = self.begin_index
        if self.step_index + 1 >= len(self.sigmas):
            raise Error("MiniMax-H3 sampling schedule: schedule exhausted")
        # sigma for the denoised estimate comes from the TIMESTEP the
        # transformer was conditioned on, not from the grid (scheduler.mojo
        # header, trap 3).
        var sigma_from_timestep = Float32(1.0) - timestep
        var sigma = self.sigmas[self.step_index]
        var sigma_next = self.sigmas[self.step_index + 1]
        var ratio = sigma_next / sigma
        self.step_index += 1
        return (sigma_from_timestep, ratio)

    def step_host(
        mut self,
        model_output: List[Float32],
        timestep: Float32,
        sample: List[Float32],
    ) raises -> List[Float32]:
        """Host-scalar Euler blend, byte-for-byte scheduler.mojo::step
        including the FMA optimization barrier. Probe/self-check use on tiny
        vectors — `step_device` is the production path."""
        if len(model_output) != len(sample):
            raise Error(
                "MiniMax-H3 sampling schedule: model_output/sample length mismatch"
            )
        var factors = self.advance(timestep)
        var sigma_from_timestep = factors[0]
        var ratio = factors[1]
        var one_minus_ratio = Float32(1.0) - ratio
        var out = List[Float32]()
        for i in range(len(sample)):
            var scaled_velocity = _sched_round_f32(sigma_from_timestep * model_output[i])
            var denoised = _sched_round_f32(sample[i] + scaled_velocity)
            var kept = _sched_round_f32(ratio * sample[i])
            var moved = _sched_round_f32(one_minus_ratio * denoised)
            out.append(_sched_round_f32(kept + moved))
        return out^

    def step_device(
        mut self,
        model_output: Tensor,
        timestep: Float32,
        sample: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        """Device-tensor Euler blend — the production path.

        `model_output` MUST be F32: H3's velocity output heads (`proj_out` /
        `audio_proj_out`) are checkpoint-native FP32 (fp8_policy.mojo F32_KEEP
        class), so no upcast is needed on that side. `sample` MUST be the F32
        ACCUMULATING latent (see file header DTYPE RULE) — never the BF16
        tensor fed to the block stack.

        Composed from ops/tensor_algebra's `mul_scalar`/`add`, each its own
        kernel launch: `scaled_velocity, denoised, kept, moved, out` — one
        launch per one of the oracle's five `_round_f32` calls, so neither
        path can silently fuse a multiply and an add across a rounding
        boundary (file header FMA note)."""
        if model_output.dtype() != STDtype.F32:
            raise Error("MiniMax-H3 sampling schedule: model_output must be F32")
        if sample.dtype() != STDtype.F32:
            raise Error(
                "MiniMax-H3 sampling schedule: sample must be the F32 accumulating"
                " latent, not the BF16 model-input tensor"
            )
        if model_output.numel() != sample.numel():
            raise Error(
                "MiniMax-H3 sampling schedule: model_output/sample numel mismatch"
            )
        var factors = self.advance(timestep)
        var sigma_from_timestep = factors[0]
        var ratio = factors[1]
        var one_minus_ratio = Float32(1.0) - ratio
        var scaled_velocity = mul_scalar(model_output, sigma_from_timestep, ctx)
        var denoised = add(sample, scaled_velocity, ctx)
        var kept = mul_scalar(sample, ratio, ctx)
        var moved = mul_scalar(denoised, one_minus_ratio, ctx)
        return add(kept, moved, ctx)


struct MiniMaxH3DualSchedule(Movable):
    """The video (shift 12.0) + audio (shift 3.0) schedule pair, built from
    ONE shared base progress grid (see file header, "THE DUAL SCHEDULE")."""

    var video: MiniMaxH3EulerSchedule
    var audio: MiniMaxH3EulerSchedule

    def __init__(
        out self,
        video_shift: Float32 = MINIMAX_H3_VIDEO_SHIFT,
        audio_shift: Float32 = MINIMAX_H3_AUDIO_SHIFT,
    ) raises:
        self.video = MiniMaxH3EulerSchedule(video_shift)
        self.audio = MiniMaxH3EulerSchedule(audio_shift)

    def set_timesteps(mut self, num_inference_steps: Int) raises:
        if num_inference_steps < 2:
            raise Error(
                "MiniMax-H3 dual schedule: `num_inference_steps` must be >= 2"
            )
        var base = _linspace_1_to_0_sampling(num_inference_steps)
        self.video.build_from_base(base.copy())
        self.audio.build_from_base(base.copy())
        # The released shifts (12.0, 3.0) never collapse different numbers of
        # sigmas at any step count exercised by the gated oracle fixture (30,
        # 50 both give num_inference_steps=29/49 on both schedules) — but a
        # collapse-count mismatch would silently desync "one forward, two
        # timesteps" into "one forward, two DIFFERENT step counts", so this
        # is checked rather than assumed.
        if self.video.num_inference_steps() != self.audio.num_inference_steps():
            raise Error(
                "MiniMax-H3 dual schedule: video/audio step counts diverged after"
                " shift + collapse — the two shifts produced a different number of"
                " distinct sigmas from the same base grid at this step count"
            )

    def num_inference_steps(self) -> Int:
        return self.video.num_inference_steps()

    def video_timestep(self, index: Int) -> Float32:
        return self.video.timesteps[index]

    def audio_timestep(self, index: Int) -> Float32:
        return self.audio.timesteps[index]

    def step_video_host(
        mut self, model_output: List[Float32], timestep: Float32, sample: List[Float32]
    ) raises -> List[Float32]:
        return self.video.step_host(model_output, timestep, sample)

    def step_audio_host(
        mut self, model_output: List[Float32], timestep: Float32, sample: List[Float32]
    ) raises -> List[Float32]:
        return self.audio.step_host(model_output, timestep, sample)

    def step_video_device(
        mut self, model_output: Tensor, timestep: Float32, sample: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        return self.video.step_device(model_output, timestep, sample, ctx)

    def step_audio_device(
        mut self, model_output: Tensor, timestep: Float32, sample: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        return self.audio.step_device(model_output, timestep, sample, ctx)


def minimax_h3_prepare_model_input(sample_f32: Tensor, ctx: DeviceContext) raises -> Tensor:
    """The ONLY sanctioned way to turn the F32 accumulating latent into the
    BF16 tensor the block stack consumes (see file header DTYPE RULE). A
    named wrapper so a future call site cannot reach for Mojo's native
    `.cast[DType.bfloat16]()` or `ops/cast.cast_tensor` by habit — both round
    ~1 BF16 quantum away from PyTorch/CUDA on some values, invisible on one
    step, compounding over N (this exact class of bug previously produced a
    visibly distorted NAVA decode with every single-step parity gate still
    green)."""
    return torch_f32_to_bf16_rne(sample_f32, ctx)


def minimax_h3_patchify_video(
    latents: Tensor, config: MiniMaxH3DiTConfig, ctx: DeviceContext
) raises -> Tensor:
    """H3's fixed (1,2,2) video patch-embed unfold, reusing `ops/patchify3d.
    patchify3d` (the proven Conv3d stride==kernel unfold) instead of
    re-deriving the index arithmetic. `latents` is `[C,F,H,W]` with
    `C == config.latents_dim` (24, released config) — checked here so a
    channel-count typo fails loudly instead of quietly feeding 50 blocks a
    wrong-shaped tensor.

    Token order is F-major-then-H-then-W (`patch = fi*HO*WO + hi*WO + wi`,
    ops/patchify3d.mojo:59-64) — BY CONSTRUCTION the same order
    `minimax_h3_build_sampling_geometry` below assigns to the target video
    rows (`video_start + frame*rows_per_frame + row`, `row` walking the (h,w)
    meshgrid h-major-then-w). A caller can zip this function's output rows
    directly with `geometry.video_indices[geometry.num_condition_video_rows:]`
    with no re-sort. NOT cross-checked here against a live tensor — no real
    H3 video latent exists in this repo yet; the agreement is the two
    functions' documented index formulas (algebraically identical), not a
    runtime probe. Flagged as unverified in the report."""
    var shape = latents.shape()
    if len(shape) != 4:
        raise Error("minimax_h3_patchify_video: latents must be rank-4 [C,F,H,W]")
    if shape[0] != config.latents_dim:
        raise Error(
            String("minimax_h3_patchify_video: channel dim ")
            + String(shape[0])
            + " != config.latents_dim "
            + String(config.latents_dim)
        )
    return patchify3d(
        latents,
        MINIMAX_H3_VIDEO_PATCH_F,
        MINIMAX_H3_VIDEO_PATCH_H,
        MINIMAX_H3_VIDEO_PATCH_W,
        ctx,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — the packed-sequence layout builder.
# Mirrors models/minimax_h3/packing.mojo's `minimax_h3_build_packed_sequence`
# + `minimax_h3_build_row_timesteps` (and their spatial/temporal-grid
# helpers). Row order: [text (L) | keyframe conditions (C) | target audio (A)
# | target video (V)] — for pure t2va, C=0 (no keyframe_anchors).
# ═════════════════════════════════════════════════════════════════════════════


@fieldwise_init
struct MiniMaxH3SamplingGeometry(Copyable, Movable):
    """The structural description of one packed MiniMax-H3 sequence — built
    ONCE per request (does not change across sampling steps). Mirrors
    packing.mojo::MiniMaxH3PackedSequence field-for-field.

    `position_ids` is flat row-major `[sequence_length * 3]`: the (t, h, w)
    rotary coordinate of row `i` is `position_ids[3*i : 3*i+3]`. `token_tags`
    is 0 video / 1 text / 2 audio for every REAL row; `< 0` marks a PADDING
    row (never produced by `minimax_h3_build_sampling_geometry` itself — see
    `minimax_h3_pad_sampling_geometry` — but downstream code, including the
    REUSED `minimax_h3_adaln_rows`, must treat it as a live possibility and
    never index the AdaLN table with it)."""

    var sequence_length: Int
    var position_ids: List[Float64]
    var token_tags: List[Int]
    var video_indices: List[Int]
    var audio_indices: List[Int]
    var text_indices: List[Int]
    var num_condition_video_rows: Int
    var num_condition_audio_rows: Int


@fieldwise_init
struct MiniMaxH3SamplingRowTimesteps(Copyable, Movable):
    """The distinct row timesteps for ONE forward call, sorted, and every
    row's index into them — rebuilt every sampling step (video_timestep and
    audio_timestep both change). `indices` is exactly what the task calls
    `timestep_indices`; feed it plus `MiniMaxH3SamplingGeometry.token_tags`
    into the REUSED `minimax_h3_adaln_rows` (see
    `minimax_h3_adaln_rows_for_step` below)."""

    var values: List[Float32]
    var indices: List[Int]


def _sampling_pairwise_sum(a: List[Float64], start: Int, n: Int) -> Float64:
    """numpy's pairwise summation. Mirrors packing.mojo::_pairwise_sum (lines
    146-188) exactly — 8-accumulator balanced tree up to a 128-element block,
    recursive split above it. Only reached by the keyframe "last" anchor path
    below (`minimax_h3_sampling_temporal_span`); pure t2va (no keyframe
    conditioning) never calls it, but a t2va-only implementation that dropped
    it would silently break fl2va reuse, so it is kept faithful."""
    if n < 8:
        var res = Float64(0.0)
        for i in range(n):
            res += a[start + i]
        return res

    if n <= _PW_BLOCKSIZE:
        var r0 = a[start + 0]
        var r1 = a[start + 1]
        var r2 = a[start + 2]
        var r3 = a[start + 3]
        var r4 = a[start + 4]
        var r5 = a[start + 5]
        var r6 = a[start + 6]
        var r7 = a[start + 7]
        var i = 8
        var limit = n - (n % 8)
        while i < limit:
            r0 += a[start + i + 0]
            r1 += a[start + i + 1]
            r2 += a[start + i + 2]
            r3 += a[start + i + 3]
            r4 += a[start + i + 4]
            r5 += a[start + i + 5]
            r6 += a[start + i + 6]
            r7 += a[start + i + 7]
            i += 8
        var res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res += a[start + i]
            i += 1
        return res

    var n2 = n // 2
    n2 -= n2 % 8
    return _sampling_pairwise_sum(a, start, n2) + _sampling_pairwise_sum(a, start + n2, n - n2)


def minimax_h3_sampling_spatial_grid(
    dim: Int, patch: Int, sqrt_area: Float64
) -> List[Float64]:
    """One aspect-normalized spatial rotary axis. Mirrors packing.mojo::
    minimax_h3_spatial_position_grid (lines 259-278) exactly: numpy's
    `linspace(start, stop, num, endpoint=False)` semantics, `stop` formed
    first and the difference taken from it (NOT `start + i*ratio/n`)."""
    var ratio = Float64(dim) / sqrt_area
    var left = (1.0 - ratio) / 2.0
    var stop = left + ratio
    var num = dim // patch
    var step = (stop - left) / Float64(num)
    var out = List[Float64]()
    for i in range(num):
        out.append((Float64(i) * step + left) * ROPE_SPATIAL_SCALE)
    return out^


def _sampling_temporal_spans(num_latent_frames: Int) -> List[Float64]:
    """`5/3 * (1, 4, 4, 4, 4)[i % 5]` per latent frame. Mirrors packing.mojo::
    _temporal_spans (lines 281-288)."""
    var spans = List[Float64]()
    for index in range(num_latent_frames):
        var slot = index % 5
        var frames_per_latent = Float64(1.0) if slot == 0 else Float64(4.0)
        spans.append(ROPE_FRAME_RESCALE * frames_per_latent)
    return spans^


def minimax_h3_sampling_temporal_grid(
    num_latent_frames: Int, origin: Float64
) -> List[Float64]:
    """The rotary time of every latent frame, starting at `origin`. Mirrors
    packing.mojo::minimax_h3_temporal_position_grid (lines 291-308): an
    EXCLUSIVE sequential cumulative sum, `origin` added to each element —
    NOT numpy's pairwise summation (that is `_sampling_temporal_span`'s job,
    for the keyframe "last" anchor only)."""
    var spans = _sampling_temporal_spans(num_latent_frames)
    var out = List[Float64]()
    var running = Float64(0.0)
    for index in range(num_latent_frames):
        if index == 0:
            out.append(origin)
        else:
            running += spans[index - 1]
            out.append(origin + running)
    return out^


def minimax_h3_sampling_temporal_span(num_latent_frames: Int) -> Float64:
    """The rotary time spanned by `num_latent_frames` latent frames, summed
    numpy-PAIRWISE (not sequentially). Mirrors packing.mojo::
    minimax_h3_temporal_position_span (lines 311-322). Drives the keyframe
    "last" anchor time; unreached by pure t2va (no anchors)."""
    var spans = List[Float64]()
    for index in range(num_latent_frames):
        var slot = index % 5
        var frames_per_latent = Float64(1.0) if slot == 0 else Float64(4.0)
        spans.append(ROPE_FRAME_RESCALE * frames_per_latent)
    return _sampling_pairwise_sum(spans, 0, num_latent_frames)


def minimax_h3_build_sampling_geometry(
    text_token_tags: List[Int],
    num_latent_frames: Int,
    latent_height: Int,
    latent_width: Int,
    num_audio_latents: Int,
    patch_h: Int,
    patch_w: Int,
    keyframe_anchors: List[Int],
) raises -> MiniMaxH3SamplingGeometry:
    """Build the [text | keyframe conditions | target audio | target video]
    layout. Mirrors packing.mojo::minimax_h3_build_packed_sequence (lines
    325-446) exactly, including the row-order/index formulas
    `minimax_h3_patchify_video`'s docstring above cites. For pure t2va, pass
    `keyframe_anchors = List[Int]()` (empty) — `num_condition_video_rows`
    comes back 0 and `num_condition_audio_rows` is always 0 on this builder
    (ref2va, which conditions on audio, is a separate builder in the oracle
    too — not reproduced here, out of this file's t2va scope)."""
    var rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    var num_text_tokens = len(text_token_tags)
    var num_condition_rows = len(keyframe_anchors) * rows_per_frame
    var num_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    var num_video_rows = num_latent_frames * rows_per_frame
    var sequence_length = (
        num_text_tokens + num_condition_rows + num_audio_rows + num_video_rows
    )

    var condition_start = num_text_tokens
    var audio_start = condition_start + num_condition_rows
    var video_start = audio_start + num_audio_rows

    var position_ids = List[Float64]()
    for _ in range(sequence_length * 3):
        position_ids.append(Float64(0.0))
    for i in range(num_text_tokens):
        position_ids[3 * i] = Float64(i)

    var sqrt_area = sqrt(Float64(latent_height * latent_width))
    var height_grid = minimax_h3_sampling_spatial_grid(latent_height, patch_h, sqrt_area)
    var width_grid = minimax_h3_sampling_spatial_grid(latent_width, patch_w, sqrt_area)
    var frame_h = List[Float64]()
    var frame_w = List[Float64]()
    for h in range(len(height_grid)):
        for w in range(len(width_grid)):
            frame_h.append(height_grid[h])
            frame_w.append(width_grid[w])

    var text_time = Float64(num_text_tokens)
    for index in range(len(keyframe_anchors)):
        var anchor = keyframe_anchors[index]
        var anchor_time: Float64
        if anchor == MINIMAX_H3_ANCHOR_FIRST:
            anchor_time = text_time
        elif anchor == MINIMAX_H3_ANCHOR_LAST:
            anchor_time = (
                text_time
                + minimax_h3_sampling_temporal_span(num_latent_frames)
                - ROPE_FRAME_RESCALE
            )
        else:
            raise Error("MiniMax-H3 sampling geometry: a keyframe anchor must be first or last")
        var row_start = condition_start + index * rows_per_frame
        for row in range(rows_per_frame):
            var base = 3 * (row_start + row)
            position_ids[base] = anchor_time
            position_ids[base + 1] = frame_h[row]
            position_ids[base + 2] = frame_w[row]

    var width_low = width_grid[0]
    var width_high = width_grid[len(width_grid) - 1]
    for row in range(num_audio_rows):
        var base = 3 * (audio_start + row)
        position_ids[base] = text_time + Float64(row % num_audio_latents)
        position_ids[base + 2] = width_low if row < num_audio_latents else width_high

    var temporal_grid = minimax_h3_sampling_temporal_grid(num_latent_frames, text_time)
    for frame in range(num_latent_frames):
        var frame_time = temporal_grid[frame]
        for row in range(rows_per_frame):
            var base = 3 * (video_start + frame * rows_per_frame + row)
            position_ids[base] = frame_time
            position_ids[base + 1] = frame_h[row]
            position_ids[base + 2] = frame_w[row]

    var video_indices = List[Int]()
    for i in range(condition_start, audio_start):
        video_indices.append(i)
    for i in range(video_start, sequence_length):
        video_indices.append(i)
    var audio_indices = List[Int]()
    for i in range(audio_start, video_start):
        audio_indices.append(i)
    var text_indices = List[Int]()
    for i in range(num_text_tokens):
        text_indices.append(i)

    var token_tags = List[Int]()
    for _ in range(sequence_length):
        token_tags.append(0)
    for i in range(num_text_tokens):
        token_tags[i] = text_token_tags[i]
    for i in range(len(audio_indices)):
        token_tags[audio_indices[i]] = MINIMAX_H3_AUDIO_TAG
    for i in range(len(video_indices)):
        token_tags[video_indices[i]] = MINIMAX_H3_VIDEO_TAG

    return MiniMaxH3SamplingGeometry(
        sequence_length,
        position_ids^,
        token_tags^,
        video_indices^,
        audio_indices^,
        text_indices^,
        num_condition_rows,
        0,
    )


def minimax_h3_build_sampling_row_timesteps(
    geometry: MiniMaxH3SamplingGeometry,
    video_timestep: Float32,
    audio_timestep: Float32,
    condition_video_timestep: Float32,
    condition_audio_timestep: Float32,
) raises -> MiniMaxH3SamplingRowTimesteps:
    """Assign a timestep to every row and reduce to (distinct sorted values,
    per-row index) — REBUILT EVERY SAMPLING STEP (video_timestep/
    audio_timestep vary; condition_* stay pinned for the whole run and are
    unused when `geometry.num_condition_*_rows == 0`, the pure-t2va case).
    Mirrors packing.mojo::minimax_h3_build_row_timesteps (lines 449-503)
    exactly, including the insertion-sort reduction (at most 4 distinct
    values ever occur)."""
    var row_timesteps = List[Float32]()
    for _ in range(geometry.sequence_length):
        row_timesteps.append(video_timestep)
    for i in range(geometry.num_condition_video_rows):
        row_timesteps[geometry.video_indices[i]] = condition_video_timestep
    for i in range(geometry.num_condition_audio_rows, len(geometry.audio_indices)):
        row_timesteps[geometry.audio_indices[i]] = audio_timestep
    for i in range(geometry.num_condition_audio_rows):
        row_timesteps[geometry.audio_indices[i]] = condition_audio_timestep

    var values = List[Float32]()
    for i in range(len(row_timesteps)):
        var value = row_timesteps[i]
        var seen = False
        for j in range(len(values)):
            if values[j] == value:
                seen = True
                break
        if not seen:
            values.append(value)
    for i in range(1, len(values)):
        var key = values[i]
        var j = i - 1
        while j >= 0 and values[j] > key:
            values[j + 1] = values[j]
            j -= 1
        values[j + 1] = key

    var indices = List[Int]()
    for i in range(len(row_timesteps)):
        var value = row_timesteps[i]
        var slot = 0
        for j in range(len(values)):
            if values[j] == value:
                slot = j
                break
        indices.append(slot)

    return MiniMaxH3SamplingRowTimesteps(values^, indices^)


def minimax_h3_adaln_rows_for_step(
    geometry: MiniMaxH3SamplingGeometry, row_timesteps: MiniMaxH3SamplingRowTimesteps
) raises -> List[Int]:
    """Per-row AdaLN table index for one forward: composes this file's own
    (token_tags, timestep_indices) with the REUSED runtime indexer
    (models/dit/minimax_h3_dit.minimax_h3_adaln_rows), not dit_frontend.
    mojo's oracle twin (minimax_h3_adaln_indices) — the whole point of this
    file's layout builder is to feed that function."""
    if len(row_timesteps.indices) != geometry.sequence_length:
        raise Error(
            "minimax_h3_adaln_rows_for_step: row_timesteps length != geometry.sequence_length"
        )
    return minimax_h3_adaln_rows(row_timesteps.indices, geometry.token_tags)


# ── Padding (multi-request batching; not exercised by the diffusers oracle,
# which never emits it for a single request — see minimax_h3_dit.mojo's own
# `minimax_h3_adaln_rows` docstring for the contract this satisfies) ────────
def minimax_h3_pad_sampling_geometry(
    geometry: MiniMaxH3SamplingGeometry, target_length: Int
) raises -> MiniMaxH3SamplingGeometry:
    """Extend a built geometry to `target_length` rows with PADDING rows:
    tag -1, zero position. A server batching multiple requests of different
    packed lengths into one attention batch needs this; `minimax_h3_dit.
    minimax_h3_adaln_rows` already commits to the tag<0-means-padding
    contract these rows satisfy (it returns row 0 for them and never faults
    the gather). video/audio/text index lists are unchanged — padding rows
    are appended past every real row and are never an Euler-step target."""
    if target_length < geometry.sequence_length:
        raise Error(
            "minimax_h3_pad_sampling_geometry: target_length shorter than the built sequence"
        )
    var position_ids = geometry.position_ids.copy()
    var token_tags = geometry.token_tags.copy()
    for _ in range(geometry.sequence_length, target_length):
        position_ids.append(Float64(0.0))
        position_ids.append(Float64(0.0))
        position_ids.append(Float64(0.0))
        token_tags.append(-1)
    return MiniMaxH3SamplingGeometry(
        target_length,
        position_ids^,
        token_tags^,
        geometry.video_indices.copy(),
        geometry.audio_indices.copy(),
        geometry.text_indices.copy(),
        geometry.num_condition_video_rows,
        geometry.num_condition_audio_rows,
    )


def minimax_h3_extend_row_timestep_indices(
    indices: List[Int], target_length: Int
) raises -> List[Int]:
    """Extend a row-timesteps index array to `target_length`, filling new
    (padding-row) entries with 0 — a placeholder never actually READ:
    `minimax_h3_adaln_rows` short-circuits on the row's token_tag, not its
    timestep index, once the tag is < 0."""
    if target_length < len(indices):
        raise Error(
            "minimax_h3_extend_row_timestep_indices: target_length shorter than input"
        )
    var out = indices.copy()
    for _ in range(len(indices), target_length):
        out.append(0)
    return out^
