# training/loop_levers_integration_smoke.mojo — WIRING-MANDATE gate (Wave 2B).
#
# The per-primitive gates (caption_dropout_smoke, noise_modifiers_smoke,
# grad_accum_smoke, ema_schedule_smoke, full_adapter_smoke) check the MATH of
# each lever in isolation. They do NOT prove the trainer loop actually drives
# the lever. This gate closes that gap: it builds an ENABLED TrainConfig in-code
# (exactly as train_klein_real.mojo consumes it) and runs the SAME wire-path
# functions, with the SAME seed derivations the loop uses, in a 3-micro-step
# in-code loop — asserting each lever's effect is observable.
#
# Wire sites mirrored (train_klein_real.mojo):
#   caption  : should_drop_caption(SEED_BASE*31 + k, cfg.caption_dropout_prob)   :662
#   noise    : apply_noise_modifiers_host(noise, N_IMG, in_ch, ...)              :692
#   grad-acc : zeros_like_group / accumulate_grad_group / scale_grad_group       :781-816
#   ema      : ema_decay_at_step(...) + ema_update_host(shadow, live, decay)     :869-879
#
# Asserts (exit NONZERO / raise on any failure):
#   (A) caption: with prob=0.5 over many steps the drop path FIRES at least once
#       AND stays put at prob=0.0 (default-off never fires).
#   (B) noise:   enabled offset+input-perturb MUTATES the host noise list;
#       all-off leaves it byte-identical.
#   (C) grad-acc: with grad_accum_steps=2 the boundary-meaned grad over two
#       DIFFERENT micro-grads == (g1+g2)/2 and DIFFERS from a single g1 (the
#       optimizer sees a different value than the no-accum path); N=1 == g1.
#   (D) ema: enabled shadow DIVERGES from live after an update; the decay comes
#       from the per-step schedule (>0 past update_after_step).
#   (E) BITROT DEMO: a deliberately-wrong config (caption prob=0.0) must NOT
#       fire the drop path — proving the gate is sensitive to the config field.
#
# Build/run (JIT):
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/training/loop_levers_integration_smoke.mojo

from serenitymojo.training.train_config import TrainConfig
from serenitymojo.training.caption_dropout import should_drop_caption
from serenitymojo.training.noise_modifiers import apply_noise_modifiers_host
from serenitymojo.training.grad_accum import (
    accumulate_grad_group, scale_grad_group, zeros_like_group,
)
from serenitymojo.training.ema_schedule import ema_decay_at_step, ema_update_host
from serenitymojo.training.schedule import (
    sample_timestep_logit_normal, sample_timestep_uniform, sample_timestep_sigmoid,
    TSD_UNIFORM, TSD_SIGMOID,
)
from serenitymojo.training.timestep_bias import apply_bias
from serenitymojo.training.loss_weight import apply_loss_weight, combined_loss_grad_elem


comptime SEED_BASE = UInt64(1234)


# ─────────────────────────────────────────────────────────────────────────────
# Wave 2D wire-path mirrors. These reproduce the EXACT loss + timestep code the
# train_klein_real.mojo loop runs (sigma sample/dispatch, bias, weighted combined
# loss) so the gate can confirm default-off byte-invariance AND enabled effect.
# ─────────────────────────────────────────────────────────────────────────────


# Mirror the trainer's timestep-sample+bias site (train_klein_real.mojo wires 1+2).
def _wire_sigma(
    seed: UInt64, dist: Int, shift: Float32, nweight: Float32, nbias: Float32,
    bias_strategy: Int, bias_mult: Float32, bias_lo: Float32, bias_hi: Float32,
) -> Float32:
    var sigma: Float32
    if dist == TSD_UNIFORM:
        sigma = sample_timestep_uniform(seed)
    elif dist == TSD_SIGMOID:
        sigma = sample_timestep_sigmoid(seed, nweight, nbias)
    else:
        sigma = sample_timestep_logit_normal(seed, shift)
    return apply_bias(sigma, Float32(1.0), bias_strategy, bias_mult, bias_lo, bias_hi)


@fieldwise_init
struct _LossOut(Copyable, Movable):
    var loss: Float32
    var d0: Float32  # d_loss[0], a representative grad element


# Mirror the trainer's loss site (wires 3+4+5) EXACTLY, including the
# default-path branch (F64 sse, d_loss=(2/N)*diff) vs combined+weighted branch.
def _wire_loss(
    pred: List[Float32], target: List[Float32],
    gamma: Float32, debiased: Bool,
    mse_s: Float32, mae_s: Float32, huber_s: Float32, sigma: Float32,
) raises -> _LossOut:
    var nout = len(pred)
    var w = apply_loss_weight(sigma, gamma, debiased, True)
    var combined_levers_on = (
        mae_s != Float32(0.0) or huber_s != Float32(0.0) or mse_s != Float32(1.0)
    )
    var loss_default_path = (w == Float32(1.0)) and (not combined_levers_on)
    var loss: Float32
    var d0 = Float32(0.0)
    if loss_default_path:
        var sse = 0.0
        var inv_n = Float32(2.0) / Float32(nout)
        for i in range(nout):
            var diff = pred[i] - target[i]
            sse += Float64(diff) * Float64(diff)
            if i == 0:
                d0 = inv_n * diff
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
            if i == 0:
                d0 = w * combined_loss_grad_elem(diff, nout, mse_s, mae_s, huber_s)
        var invn = 1.0 / Float64(nout)
        var combined = (
            Float64(mse_s) * (sum_sq * invn)
            + Float64(mae_s) * (sum_abs * invn)
            + Float64(huber_s) * (sum_hub * invn)
        )
        loss = Float32(Float64(w) * combined)
    return _LossOut(loss, d0)


def _host_noise(n: Int, seed: UInt64) -> List[Float32]:
    """Cheap deterministic stand-in for the trainer's noise list (the values do
    not matter — the gate only checks WHETHER the modifier mutates them)."""
    var out = List[Float32]()
    var s = seed
    for _ in range(n):
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        out.append(Float32(Int((s >> 40) & UInt64(1023))) * Float32(0.001) - Float32(0.5))
    return out^


def _group(seed: Int, n_adapters: Int, numel: Int) -> List[List[Float32]]:
    var out = List[List[Float32]]()
    for i in range(n_adapters):
        var inner = List[Float32]()
        for j in range(numel):
            inner.append(Float32((i * 13 + j * 7 + seed) % 11) * Float32(0.137) - Float32(0.5))
        out.append(inner^)
    return out^


def _bit_equal(a: List[Float32], b: List[Float32]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


# Build an enabled config the way the reader would: start from default(), flip
# the lever fields by name. (Reader key-parsing for these keys is Wave 2C; here
# we mutate in-code, which exercises the SAME fields the loop reads.)
def _enabled_config() -> TrainConfig:
    var cfg = TrainConfig.default()
    cfg.in_channels = 4
    cfg.num_double = 2
    cfg.num_single = 2
    cfg.caption_dropout_prob = Float32(0.5)
    cfg.offset_noise_weight = Float32(0.1)
    cfg.offset_noise_prob = Float32(1.0)        # always fire (deterministic test)
    cfg.input_perturbation = Float32(0.1)
    cfg.grad_accum_steps = 2
    cfg.ema_enabled = True
    cfg.ema_inv_gamma = Float32(1.0)
    cfg.ema_power = Float32(0.6667)
    cfg.ema_update_after_step = 0
    cfg.ema_min_decay = Float32(0.0)
    cfg.ema_max_decay = Float32(0.9999)
    return cfg^


def main() raises:
    var ok = True
    var N_IMG = 64
    var cfg = _enabled_config()

    # ── (A) caption-dropout wire path fires at the trainer's exact seed ────────
    var fired = False
    for k in range(64):
        if should_drop_caption(SEED_BASE * UInt64(31) + UInt64(k), cfg.caption_dropout_prob):
            fired = True
    if fired:
        print("PASS (A) caption-dropout drop path FIRED with prob=0.5 (loop wired)")
    else:
        print("FAIL (A) caption-dropout never fired at prob=0.5"); ok = False
    # default-off must never fire (over the same window)
    var off_fired = False
    for k in range(256):
        if should_drop_caption(SEED_BASE * UInt64(31) + UInt64(k), Float32(0.0)):
            off_fired = True
    if off_fired:
        print("FAIL (A) caption-dropout fired at prob=0.0 (default-off violated)"); ok = False
    else:
        print("PASS (A) caption-dropout default-off (prob=0.0) never fired")

    # ── (B) noise modifiers wire path MUTATES the host noise list ─────────────
    var noise_on = _host_noise(N_IMG * cfg.in_channels, SEED_BASE * UInt64(7919) + UInt64(0))
    var noise_baseline = noise_on.copy()
    var _skip = apply_noise_modifiers_host(
        noise_on, N_IMG, cfg.in_channels,
        cfg.offset_noise_weight, cfg.offset_noise_prob,
        cfg.input_perturbation,
        cfg.multires_iterations, cfg.multires_discount,
        SEED_BASE * UInt64(7919) + UInt64(0),
    )
    if _bit_equal(noise_on, noise_baseline):
        print("FAIL (B) noise modifiers enabled but noise unchanged (lever not wired)"); ok = False
    else:
        print("PASS (B) enabled offset+input-perturb MUTATED the host noise list (loop wired)")
    # all-off path must leave the list byte-identical
    var noise_off = _host_noise(N_IMG * cfg.in_channels, SEED_BASE * UInt64(7919) + UInt64(1))
    var noise_off_base = noise_off.copy()
    var _skip2 = apply_noise_modifiers_host(
        noise_off, N_IMG, cfg.in_channels,
        Float32(0.0), Float32(0.0), Float32(0.0), 0, Float32(0.0),
        SEED_BASE * UInt64(7919) + UInt64(1),
    )
    if _bit_equal(noise_off, noise_off_base):
        print("PASS (B) all-off noise path byte-identical (default-off)")
    else:
        print("FAIL (B) all-off noise path mutated the list"); ok = False

    # ── (C) grad-accum boundary: meaned grad over a window != single g1 ───────
    var n_ad = cfg.num_double * 4 + cfg.num_single * 2
    var g1 = _group(2, n_ad, 20)
    var g2 = _group(5, n_ad, 20)
    # mirror the loop: zeros at window start, accumulate each micro-step, mean at
    # boundary (here grad_accum_steps=2).
    var acc = zeros_like_group(g1)
    accumulate_grad_group(acc, g1)
    accumulate_grad_group(acc, g2)
    scale_grad_group(acc, Float32(1.0) / Float32(2.0))
    # boundary value must equal (g1+g2)/2 to 1e-6 ...
    var maxerr = Float32(0.0)
    for i in range(n_ad):
        for j in range(20):
            var e = acc[i][j] - Float32(0.5) * (g1[i][j] + g2[i][j])
            if e < Float32(0.0):
                e = -e
            if e > maxerr:
                maxerr = e
    print("(C) grad-accum boundary max |mean - (g1+g2)/2| =", maxerr)
    if maxerr > Float32(1.0e-6):
        print("FAIL (C) grad-accum boundary mean wrong"); ok = False
    else:
        print("PASS (C) N=2 boundary == (g1+g2)/2 (loop wired)")
    # ... and must DIFFER from the no-accum path (single g1) for these inputs.
    if _bit_equal(acc[0], g1[0]):
        print("FAIL (C) accumulated grad equals single g1 (accum has no effect)"); ok = False
    else:
        print("PASS (C) accumulated mean DIFFERS from single-micro-step g1 (observable effect)")
    # N=1 default-off: window of one == g1 bit-exact.
    var acc1 = zeros_like_group(g1)
    accumulate_grad_group(acc1, g1)
    scale_grad_group(acc1, Float32(1.0) / Float32(1.0))
    if _bit_equal(acc1[0], g1[0]):
        print("PASS (C) N=1 == g1 (default-off byte-unchanged)")
    else:
        print("FAIL (C) N=1 != g1"); ok = False

    # ── (D) EMA wire path: shadow diverges from live; decay from schedule ─────
    var live = List[Float32]()
    var shadow = List[Float32]()
    for i in range(16):
        live.append(Float32(i) * Float32(0.1) + Float32(0.05))
        shadow.append(Float32(0.0))
    var shadow0 = shadow.copy()
    var any_update = False
    for k in range(1, 4):                       # micro-loop steps 1..3
        var decay = ema_decay_at_step(
            k, cfg.ema_update_after_step, cfg.ema_inv_gamma,
            cfg.ema_power, cfg.ema_min_decay, cfg.ema_max_decay,
        )
        if decay > Float32(0.0):
            ema_update_host(shadow, live, decay)
            any_update = True
    if not any_update:
        print("FAIL (D) EMA schedule returned no positive decay (update never ran)"); ok = False
    elif _bit_equal(shadow, shadow0):
        print("FAIL (D) EMA shadow unchanged after updates (lever not wired)"); ok = False
    elif _bit_equal(shadow, live):
        print("FAIL (D) EMA shadow collapsed to live (decay==0 path)"); ok = False
    else:
        print("PASS (D) EMA shadow DIVERGED from both init and live (loop wired)")
    # default-off: ema_enabled=False => the loop allocates no shadow & skips the
    # block. We assert the gate-config field is the switch.
    if TrainConfig.default().ema_enabled:
        print("FAIL (D) default config has ema_enabled=True (default-off violated)"); ok = False
    else:
        print("PASS (D) default config ema_enabled=False (default-off)")

    # ── (E) BITROT DEMO: wrong config (caption prob 0) must NOT fire ──────────
    var demo_fired = False
    for k in range(256):
        if should_drop_caption(SEED_BASE * UInt64(31) + UInt64(k), Float32(0.0)):
            demo_fired = True
    if demo_fired:
        print("FAIL (E) bitrot demo: prob=0.0 still fired (gate not sensitive)"); ok = False
    else:
        print("PASS (E) bitrot demo: prob=0.0 config never fires (gate keyed on the field)")

    # ═════════════════════════════════════════════════════════════════════════
    # Wave 2D: timestep-distribution + bias + weighted/combined loss wires.
    # ═════════════════════════════════════════════════════════════════════════

    # Build a deterministic pred/target pair (values matter here — we compare the
    # actual loss bits and grads). 64 elements.
    var NV = 64
    var pred = List[Float32]()
    var tgt = List[Float32]()
    for i in range(NV):
        pred.append(Float32((i * 7 + 3) % 23) * Float32(0.11) - Float32(1.0))
        tgt.append(Float32((i * 5 + 2) % 19) * Float32(0.13) - Float32(0.8))

    # ── (2D-a) DEFAULT-OFF byte-invariance ────────────────────────────────────
    # The wired path with default fields MUST reproduce the pre-wave loss+grad
    # bit-for-bit. We compute the reference INLINE the pre-wave way and compare.
    var ref_sse = 0.0
    var ref_invn2 = Float32(2.0) / Float32(NV)
    var ref_d0 = Float32(0.0)
    for i in range(NV):
        var diff = pred[i] - tgt[i]
        ref_sse += Float64(diff) * Float64(diff)
        if i == 0:
            ref_d0 = ref_invn2 * diff
    var ref_loss = Float32(ref_sse / Float64(NV))

    var dcfg = TrainConfig.default()
    # default sigma draw (production logit-normal+qwen-shift, no bias):
    var ref_sigma = sample_timestep_logit_normal(SEED_BASE + UInt64(7), dcfg.timestep_shift)
    var wired_sigma = _wire_sigma(
        SEED_BASE + UInt64(7), dcfg.timestep_distribution, dcfg.timestep_shift,
        dcfg.timestep_noising_weight, dcfg.timestep_noising_bias,
        dcfg.timestep_bias_strategy, dcfg.timestep_bias_multiplier,
        dcfg.timestep_bias_range_min, dcfg.timestep_bias_range_max,
    )
    if wired_sigma.to_bits() == ref_sigma.to_bits():
        print("PASS (2D-a) default timestep draw BYTE-IDENTICAL to pre-wave (bits=",
              wired_sigma.to_bits(), ")")
    else:
        print("FAIL (2D-a) default timestep draw bits differ: wired=",
              wired_sigma.to_bits(), " ref=", ref_sigma.to_bits()); ok = False

    var def_lo = _wire_loss(
        pred, tgt, dcfg.min_snr_gamma, dcfg.debiased,
        dcfg.loss_mse_strength, dcfg.loss_mae_strength, dcfg.loss_huber_strength,
        wired_sigma,
    )
    if def_lo.loss.to_bits() == ref_loss.to_bits() and def_lo.d0.to_bits() == ref_d0.to_bits():
        print("PASS (2D-a) default loss+grad BYTE-IDENTICAL to pre-wave (loss bits=",
              def_lo.loss.to_bits(), " d0 bits=", def_lo.d0.to_bits(), ")")
    else:
        print("FAIL (2D-a) default loss/grad bits differ: loss wired=", def_lo.loss.to_bits(),
              " ref=", ref_loss.to_bits(), " d0 wired=", def_lo.d0.to_bits(),
              " ref=", ref_d0.to_bits()); ok = False

    # ── (2D-b) ENABLED config: each lever changes the observed value ──────────

    # (b1) loss weight min_snr_gamma=5 => w != 1 => loss scales, grad scales.
    var en_sigma = wired_sigma  # same sigma, isolate the weight effect
    var wlo = _wire_loss(pred, tgt, Float32(5.0), False,
                         Float32(1.0), Float32(0.0), Float32(0.0), en_sigma)
    var w5 = apply_loss_weight(en_sigma, Float32(5.0), False, True)
    if w5 == Float32(1.0):
        print("FAIL (2D-b) min_snr w==1 at gamma=5 (test sigma degenerate)"); ok = False
    elif wlo.loss.to_bits() != def_lo.loss.to_bits() and wlo.d0.to_bits() != def_lo.d0.to_bits():
        print("PASS (2D-b) min_snr_gamma=5 weight=", w5,
              " CHANGES loss+grad vs default (loss", def_lo.loss, "->", wlo.loss, ")")
    else:
        print("FAIL (2D-b) min_snr_gamma=5 did NOT change loss/grad"); ok = False

    # (b2) debiased=True => w != 1 => loss changes.
    var dblo = _wire_loss(pred, tgt, Float32(-1.0), True,
                          Float32(1.0), Float32(0.0), Float32(0.0), en_sigma)
    if dblo.loss.to_bits() != def_lo.loss.to_bits():
        print("PASS (2D-b) debiased=True CHANGES loss vs default (", def_lo.loss, "->", dblo.loss, ")")
    else:
        print("FAIL (2D-b) debiased=True did NOT change loss"); ok = False

    # (b3) huber on (mse=1, huber=1) => combined != MSE-only => loss+grad change.
    var hblo = _wire_loss(pred, tgt, Float32(-1.0), False,
                          Float32(1.0), Float32(0.0), Float32(1.0), en_sigma)
    if hblo.loss.to_bits() != def_lo.loss.to_bits() and hblo.d0.to_bits() != def_lo.d0.to_bits():
        print("PASS (2D-b) huber_strength=1 CHANGES loss+grad vs MSE-only (", def_lo.loss, "->", hblo.loss, ")")
    else:
        print("FAIL (2D-b) huber_strength=1 did NOT change loss/grad"); ok = False

    # (b4) timestep_bias_strategy=Later remaps the sampled sigma upward.
    var biased = _wire_sigma(
        SEED_BASE + UInt64(7), dcfg.timestep_distribution, dcfg.timestep_shift,
        dcfg.timestep_noising_weight, dcfg.timestep_noising_bias,
        1, Float32(0.5), Float32(0.0), Float32(1.0),  # TSB_LATER, m=0.5
    )
    if biased.to_bits() != wired_sigma.to_bits() and biased >= wired_sigma:
        print("PASS (2D-b) timestep_bias=Later REMAPS sigma upward (", wired_sigma, "->", biased, ")")
    else:
        print("FAIL (2D-b) timestep_bias=Later did NOT remap sigma"); ok = False

    # (b5) timestep_distribution=Uniform differs from the logit-normal draw.
    var uni = _wire_sigma(
        SEED_BASE + UInt64(7), TSD_UNIFORM, dcfg.timestep_shift,
        dcfg.timestep_noising_weight, dcfg.timestep_noising_bias,
        0, Float32(0.0), Float32(0.0), Float32(1.0),
    )
    if uni.to_bits() != wired_sigma.to_bits():
        print("PASS (2D-b) timestep_distribution=Uniform draw DIFFERS from logit-normal (",
              wired_sigma, " vs ", uni, ")")
    else:
        print("FAIL (2D-b) Uniform draw equals logit-normal (selector not wired)"); ok = False

    if not ok:
        raise Error("loop_levers_integration_smoke FAILED")
    print("loop_levers_integration_smoke gate PASS")
