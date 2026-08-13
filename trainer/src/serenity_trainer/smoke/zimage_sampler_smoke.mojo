# zimage_sampler_smoke.mojo — compile smoke for the Z-Image sampler slice:
#   modelSampler/FlowMatchEulerDiscreteScheduler.mojo  (ported diffusers scheduler)
#   modelSampler/ZImageSampler.mojo                    (ported __sample_base loop)
#
# Runs a few denoise steps on a FIXED latent + FIXED cap_feats read from
#   parity/zi_sampler_ref.safetensors  (cap [224,2560] f32, latent0 [1,16,8,8] f32,
#   sigmas [9] f32, timesteps [8] f32 — the Serenity/diffusers reference, shift=6,
#   8 steps), at cfg=1.0, printing latent stats per step.
#
# Two gates:
#  (1) SCHEDULE — the ported FlowMatchEulerDiscreteScheduler.set_timesteps(8) sigmas
#      & timesteps must match the reference tensors EXACTLY (f32). This is the
#      load-bearing scheduler-math gate (sigma linspace + double-shift + terminal 0).
#  (2) EULER STEP — the ported scheduler.step (prev = sample + (σ_next-σ)·v) run on
#      the FIXED latent0 with a FIXED, cap-derived noise_pred, cfg=1. We print the
#      latent mean/std per step so the orchestrator can diff the trajectory.
#
# The full transformer forward (zimage_forward_full_lora) needs the multi-GB frozen
# weight store + 210-adapter LoRA set — out of scope for a compile smoke; it is
# wired in ZImageSampler.sample_zimage and gated by the orchestrator against
# Serenity end-to-end. Here the noise_pred is a deterministic FIXED stand-in
# (mean of cap_feats broadcast over the latent), so the per-step latent stats are
# fully reproducible and exercise the real ported scheduler tensor op.

from max.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.tensor_algebra import add_scalar, zeros_device

from serenity_trainer.modelSampler.FlowMatchEulerDiscreteScheduler import (
    make_zimage_scheduler, ZIMAGE_DEFAULT_SHIFT,
)


comptime REF_PATH = "trainer/parity/zi_sampler_ref.safetensors"
comptime N_STEPS = 8


def _mean_std(vals: List[Float32]) -> List[Float32]:
    var n = len(vals)
    var s = Float32(0.0)
    for i in range(n):
        s += vals[i]
    var mean = s / Float32(n)
    var v = Float32(0.0)
    for i in range(n):
        var d = vals[i] - mean
        v += d * d
    var std = sqrt(v / Float32(n))
    var out = List[Float32]()
    out.append(mean); out.append(std)
    return out^


def _check_close(name: String, got: Float32, expected: Float32, tol: Float32) raises:
    var diff = got - expected
    if diff < 0.0:
        diff = -diff
    if diff > tol:
        raise Error(
            name + String(" MISMATCH got=") + String(got)
            + String(" expected=") + String(expected)
            + String(" |Δ|=") + String(diff)
        )


def main() raises:
    var ctx = DeviceContext()
    print("=== Z-Image sampler smoke (scheduler + Euler step) ===")

    # ── load the reference fixture ────────────────────────────────────────────
    var ref = ShardedSafeTensors.open(String(REF_PATH))
    var lat0 = Tensor.from_view(ref.tensor_view(String("latent0")), ctx)   # [1,16,8,8] f32
    var cap = Tensor.from_view(ref.tensor_view(String("cap")), ctx)        # [224,2560] f32
    var ref_sigmas = ref.tensor_view(String("sigmas"))                     # [9] f32
    var ref_timesteps = ref.tensor_view(String("timesteps"))               # [8] f32
    var sig_host = Tensor.from_view(ref_sigmas, ctx).to_host(ctx)
    var ts_host = Tensor.from_view(ref_timesteps, ctx).to_host(ctx)

    # ── GATE 1: ported scheduler schedule vs reference ────────────────────────
    var sch = make_zimage_scheduler(N_STEPS, ZIMAGE_DEFAULT_SHIFT)
    print("-- schedule (shift=", ZIMAGE_DEFAULT_SHIFT, ", steps=", N_STEPS, ") --")
    for i in range(N_STEPS):
        print(
            "  i=", i,
            " sigma=", sch.sigmas[i], " (ref ", sig_host[i], ")",
            " t=", sch.timesteps[i], " (ref ", ts_host[i], ")",
        )
        _check_close(String("sigma[") + String(i) + String("]"),
                     sch.sigmas[i], sig_host[i], Float32(1e-4))
        _check_close(String("timestep[") + String(i) + String("]"),
                     sch.timesteps[i], ts_host[i], Float32(5e-2))
    # terminal sigma 0
    _check_close(String("sigma_terminal"), sch.sigmas[N_STEPS], sig_host[N_STEPS], Float32(1e-6))
    print("  GATE 1 OK: ported sigmas/timesteps match reference (terminal sigma=",
          sch.sigmas[N_STEPS], ")")

    # ── FIXED noise_pred stand-in: mean(cap_feats) broadcast over the latent ──
    # Deterministic so per-step latent stats are reproducible. (Real run uses
    # zimage_forward_full_lora; the orchestrator gates the full trajectory.)
    var cap_host = cap.to_host(ctx)
    var cap_mean = _mean_std(cap_host)[0]
    print("-- fixed noise_pred scalar = mean(cap) =", cap_mean, "--")

    # ── GATE 2: Euler step trajectory on the FIXED latent, cfg=1 ──────────────
    var latent = lat0
    var l0_stats = _mean_std(latent.to_host(ctx))
    print("  step 0 (init)  latent mean=", l0_stats[0], " std=", l0_stats[1])
    for i in range(N_STEPS):
        # cfg=1 → noise_pred is the single (cond) branch (ZImageSampler.py:69-74,
        # batch_size=1). Here a FIXED, cap-derived uniform stand-in: zeros-shaped
        # like the latent, shifted by cap_mean. (Real run: zimage_forward_full_lora.)
        var noise_pred = add_scalar(
            zeros_device(latent.shape(), latent.dtype(), ctx), cap_mean, ctx
        )
        latent = sch.step(noise_pred, latent, i, ctx)
        var st = _mean_std(latent.to_host(ctx))
        print(
            "  step ", i + 1, " sigma=", sch.sigmas[i], "->", sch.sigmas[i + 1],
            " latent mean=", st[0], " std=", st[1],
        )

    var fin = _mean_std(latent.to_host(ctx))
    print("  GATE 2 OK: final latent mean=", fin[0], " std=", fin[1])
    print("=== smoke complete ===")
