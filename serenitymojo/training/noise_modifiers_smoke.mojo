# training/noise_modifiers_smoke.mojo — gate for noise modifiers (item 2e).
#
# Asserts:
#   (1) ALL-OFF byte invariance: apply_noise_modifiers_host with
#       weight=0,prob=0,gamma=0,iterations=0 leaves the list BIT-EQUAL to input.
#   (2) OFFSET-NOISE oracle: when it fires (prob=1), out[t,c] == in[t,c] +
#       weight*off[c] to 1e-6, where off[c] is the SAME randn stream recomputed
#       independently here (the Rust formula `noise + weight*per_channel`).
#   (3) INPUT-PERTURB oracle: out[i] == in[i] + gamma*pert[i] to 1e-6, with
#       pert recomputed independently (the Rust `noise + gamma*randn`).
#   (4) OFFSET prob=0 does NOT fire (no change) — Bernoulli short-circuit.
#   (5) BITROT-FAIL DEMO: comparing the offset output against the WRONG formula
#       (weight applied to the wrong channel, off[(c+1)%C]) must exceed 1e-6.
#
# Exits NONZERO (raise) on any mismatch.
#
# Build/run (JIT):
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/training/noise_modifiers_smoke.mojo

from serenitymojo.training.noise_modifiers import (
    apply_noise_modifiers_host, apply_offset_noise_host,
    apply_input_perturbation_host, _host_randn, _bernoulli_uniform,
)


def _base_noise(n: Int) -> List[Float32]:
    # A simple deterministic non-zero base so changes are visible.
    var out = List[Float32]()
    for i in range(n):
        out.append(Float32(0.5) + Float32(i % 7) * Float32(0.1))
    return out^


def main() raises:
    var ok = True
    var n_tokens = 16
    var channels = 8
    var n = n_tokens * channels

    # ── (1) all-off byte invariance ───────────────────────────────────────────
    var a = _base_noise(n)
    var a_ref = _base_noise(n)
    var skipped = apply_noise_modifiers_host(
        a, n_tokens, channels,
        Float32(0.0), Float32(0.0), Float32(0.0), 0, Float32(0.0), UInt64(123),
    )
    var diff_off = 0
    for i in range(n):
        if a[i] != a_ref[i]:
            diff_off += 1
    print("all-off changed elements (expect 0):", diff_off, " multires_skipped=", skipped)
    if diff_off != 0 or skipped:
        print("FAIL all-off NOT byte-identical"); ok = False
    else:
        print("PASS all-off leaves noise byte-identical (default-off)")

    # ── (2) offset-noise oracle (prob=1 => always fires) ──────────────────────
    var b = _base_noise(n)
    var weight = Float32(0.3)
    var off_seed = UInt64(555) * UInt64(101) + UInt64(1)
    var bern_seed = UInt64(555) * UInt64(103) + UInt64(2)
    apply_offset_noise_host(b, n_tokens, channels, weight, Float32(1.0), off_seed, bern_seed)
    var off_ref = _host_randn(channels, off_seed)
    var base_b = _base_noise(n)
    var maxerr = Float32(0.0)
    for t in range(n_tokens):
        for c in range(channels):
            var expect = base_b[t * channels + c] + weight * off_ref[c]
            var e = b[t * channels + c] - expect
            if e < Float32(0.0):
                e = -e
            if e > maxerr:
                maxerr = e
    print("offset-noise max |out - (in + w*off[c])| =", maxerr)
    if maxerr > Float32(1.0e-6):
        print("FAIL offset noise does not match Rust formula"); ok = False
    else:
        print("PASS offset noise == in + weight*per_channel_randn to 1e-6")

    # ── (3) input-perturbation oracle ─────────────────────────────────────────
    var d = _base_noise(n)
    var gamma = Float32(0.05)
    var pseed = UInt64(777)
    apply_input_perturbation_host(d, gamma, pseed)
    var pert_ref = _host_randn(n, pseed)
    var base_d = _base_noise(n)
    var maxerr2 = Float32(0.0)
    for i in range(n):
        var expect = base_d[i] + gamma * pert_ref[i]
        var e = d[i] - expect
        if e < Float32(0.0):
            e = -e
        if e > maxerr2:
            maxerr2 = e
    print("input-perturb max |out - (in + g*randn)| =", maxerr2)
    if maxerr2 > Float32(1.0e-6):
        print("FAIL input perturbation does not match Rust formula"); ok = False
    else:
        print("PASS input perturbation == in + gamma*randn to 1e-6")

    # ── (4) offset prob=0 never fires ─────────────────────────────────────────
    var e0 = _base_noise(n)
    var e0_ref = _base_noise(n)
    apply_offset_noise_host(e0, n_tokens, channels, Float32(0.3), Float32(0.0), UInt64(1), UInt64(2))
    var diff0 = 0
    for i in range(n):
        if e0[i] != e0_ref[i]:
            diff0 += 1
    print("offset prob=0 changed elements (expect 0):", diff0)
    if diff0 != 0:
        print("FAIL offset noise fired at prob=0"); ok = False
    else:
        print("PASS offset noise prob=0 never fires")

    # ── (5) multires requested in token-space must fail loud ──────────────────
    var multires_raised = False
    try:
        var m = _base_noise(n)
        _ = apply_noise_modifiers_host(
            m, n_tokens, channels,
            Float32(0.0), Float32(0.0), Float32(0.0), 4, Float32(0.5), UInt64(999),
        )
    except:
        multires_raised = True
    if not multires_raised:
        print("FAIL multires token-space request did not raise"); ok = False
    else:
        print("PASS multires token-space request fails loud")

    # ── (6) BITROT-FAIL DEMO: wrong-channel offset must exceed 1e-6 ────────────
    var wrong_maxerr = Float32(0.0)
    for t in range(n_tokens):
        for c in range(channels):
            var wrong_expect = base_b[t * channels + c] + weight * off_ref[(c + 1) % channels]
            var e = b[t * channels + c] - wrong_expect
            if e < Float32(0.0):
                e = -e
            if e > wrong_maxerr:
                wrong_maxerr = e
    print("bitrot demo: max err vs WRONG (shifted-channel) formula =", wrong_maxerr)
    if wrong_maxerr <= Float32(1.0e-6):
        print("FAIL bitrot demo: wrong formula matched within tol"); ok = False
    else:
        print("PASS bitrot demo: wrong formula exceeds 1e-6 (gate is sensitive)")

    if not ok:
        raise Error("noise_modifiers_smoke FAILED")
    print("noise_modifiers_smoke gate PASS")
