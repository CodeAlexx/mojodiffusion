# serenitymojo/models/minimax_h3_device/parity/minimax_h3_audio_decoder_device_parity.mojo
#
# GATE: the DEVICE BigVGAN audio decoder against the gated HOST ORACLE.
#
# The reference is `serenitymojo/models/minimax_h3/audio_decoder.mojo`, which is
# itself gated to ~1e-6 against the vendor's `dac_bigvgan.py` (12 staged checks,
# minimax_h3_audio_real_weights_parity.mojo). So this file does NOT re-derive
# correctness from the vendor — it proves a host->device MOVE preserved it.
# Both sides run:
#     * the SAME real weights (FL2VA/audio_vae/model.safetensors, F32), loaded
#       once by the host oracle's own loader and handed to the device uploader,
#       so "different weights" cannot be the explanation for a divergence;
#     * the SAME fixed synthetic latent (deterministic LCG below, no RNG op,
#       no file) — byte-identical input on both sides.
#
# STAGE-BY-STAGE, mirroring the host oracle's own staged entry points, so a
# divergence lands on a stage instead of on a single number at the waveform:
#     check  1      pre          dec_in_proj + conv_pre   [1, 1024, T]
#     checks 2-8    stages 1..7  each upsample + 3 averaged AMP blocks
#     check  9      tail         activation_post + conv_post  (UNCLAMPED)
#     check 10      end-to-end   full decode, clamped, as the pipeline calls it
# The stages are NOT interchangeable — rates (5,5,2,2,2,2,2) with kernels
# (9,9,4,4,4,4,4) — so "it matches at stage 0" does not imply stage 2. Each
# stage is re-run from the latent on both sides (the host oracle exposes
# cumulative `..._stages(n)`, not a resumable one); that is O(n^2) host work and
# is why the gate runs at 4 latents, not 37.
#
# BAR: cos >= 0.9999 per stage, with max_abs reported alongside. Both sides are
# F32 with F32 accumulation; the device convolution sums in a different ORDER
# than the host's serial loop, so bit-equality is not the right bar and a
# too-tight bar would fail for arithmetic reasons and teach nothing. What this
# must catch is STRUCTURAL error — a wrong pad, a missed weight-norm fold, the
# wrong snake, an unflipped kernel — and those move cosine to <0.99, not to
# 0.99999 (the host oracle's own control experiments moved it to 0.275).
#
# CHECK 0 is not numeric: `ops/activation1d` decomposes the depthwise
# ConvTranspose1d upsampler into zero-insert + plain conv1d WITHOUT flipping
# the kernel, which equals a true transposed convolution only for a SYMMETRIC
# filter. Derivation: conv_transpose puts x[i]*w[k] at p = i*stride + k, while
# conv1d(zero_insert(x), wd) with side pad 11 puts x[i]*wd[k] at o = 2i + 11 - k;
# equating gives wd[k] = w[11-k] = w[K-1-k], the flip. So the reuse is valid iff
# w is palindromic. That is a PRECONDITION of the whole device path, it is
# cheap, and it is checked here before any numbers are compared.
#
# Run (needs the GPU free — see the chain note in the report):
#   pixi run mojo build -O2 -j 1 -I . -I vendor/mojo-libs -Xlinker -lm \
#     serenitymojo/models/minimax_h3_device/parity/minimax_h3_audio_decoder_device_parity.mojo \
#     -o output/checks/minimax_h3_audio_decoder_device_parity \
#   && output/checks/minimax_h3_audio_decoder_device_parity
#
# Optional trailing arg `perf` adds the 37-latent host-vs-device timing (the
# standard 0.925 s decode). It is opt-in because the host side of it is ~87 s.

from std.collections import List
from std.math import sqrt
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.minimax_h3.audio_decoder import (
    MiniMaxH3AudioDecoderConfig,
    MiniMaxH3AudioSignal,
    MiniMaxH3AudioWeights,
    minimax_h3_audio_decode,
    minimax_h3_audio_decode_pre,
    minimax_h3_audio_decode_stages,
    minimax_h3_audio_decode_tail,
)
from serenitymojo.models.minimax_h3_device.audio_decoder_device import (
    MiniMaxH3AudioDeviceWeights,
    minimax_h3_audio_decode_device,
    minimax_h3_audio_decode_device_pre,
    minimax_h3_audio_decode_device_stages,
    minimax_h3_audio_decode_device_tail,
    minimax_h3_audio_device_weights,
)

comptime AUDIO_VAE_PATH = (
    "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/audio_vae/model.safetensors"
)
comptime COS_BAR = Float64(0.9999)
comptime GATE_LATENTS = 4
comptime PERF_LATENTS = 37

comptime LATENT_CHANNELS = 32
comptime LATENT_DIM = 2048
comptime DECODER_DIM = 1024


def _config() -> MiniMaxH3AudioDecoderConfig:
    """The released decoder config, verbatim from the pipeline's own
    `_minimax_h3_audio_decoder_config` (t2va: rates/kernels/resblocks)."""
    var rates: List[Int] = [5, 5, 2, 2, 2, 2, 2]
    var kernels: List[Int] = [9, 9, 4, 4, 4, 4, 4]
    var rb_kernels: List[Int] = [3, 7, 11]
    var d: List[Int] = [1, 3, 5]
    var dilations: List[List[Int]] = [d.copy(), d.copy(), d.copy()]
    return MiniMaxH3AudioDecoderConfig(
        LATENT_CHANNELS, LATENT_DIM, DECODER_DIM,
        rates^, kernels^, rb_kernels^, dilations^,
    )


def _load_host_weights(path: String) raises -> MiniMaxH3AudioWeights:
    """The pipeline's own loading loop (`_minimax_h3_load_audio_vae_weights`),
    including its exclusion of `zero_k_bias` — a frozen zero buffer the decoder
    deliberately never reads."""
    var st = SafeTensors.open(path)
    var all_names = st.names()
    var names = List[String]()
    var values = List[List[Float32]]()
    for i in range(len(all_names)):
        ref n = all_names[i]
        if n.find("zero_k_bias") >= 0:
            continue
        var info = st.tensor_info(n)
        var bytes = st.tensor_bytes(n)
        var tv = from_parts(info.dtype, info.shape.copy(), bytes)
        if tv.dtype != STDtype.F32:
            raise Error(String("audio_vae tensor ") + n + " is not F32")
        var p = tv.data.unsafe_ptr().bitcast[Float32]()
        var v = List[Float32](capacity=tv.numel())
        for j in range(tv.numel()):
            v.append(p[j])
        names.append(n)
        values.append(v^)
    return MiniMaxH3AudioWeights(names^, values^)


def _synthetic_latents(channels: Int, num_latents: Int) -> List[Float32]:
    """Deterministic `[channels, num_latents]` latent — a 64-bit LCG mapped to
    [-3, 3]. Fixed and self-contained so the gate needs no oracle file and both
    sides see byte-identical input. Scale is latent-like: the real audio latents
    are denormalized by a per-channel std of ~1.7-2.8."""
    var out = List[Float32](capacity=channels * num_latents)
    var state = UInt64(0x2545F4914F6CDD1D)
    for _ in range(channels * num_latents):
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u = Float64((state >> 11) & UInt64(0x1FFFFFFFFFFFFF)) / Float64(
            9007199254740992.0
        )
        out.append(Float32(u * 6.0 - 3.0))
    return out^


def _cosine(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error(
            String("length mismatch: ") + String(len(a)) + " vs " + String(len(b))
        )
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(len(a)):
        var x = Float64(a[i])
        var y = Float64(b[i])
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        raise Error("cosine: a zero vector — a stage collapsed to all zeros")
    return dot / (sqrt(na) * sqrt(nb))


def _max_abs(a: List[Float32], b: List[Float32]) raises -> Float32:
    if len(a) != len(b):
        raise Error("max_abs: length mismatch")
    var worst = Float32(0.0)
    for i in range(len(a)):
        var d = a[i] - b[i]
        if d < 0:
            d = -d
        if d > worst:
            worst = d
    return worst


def _rms(a: List[Float32]) -> Float64:
    var acc = Float64(0.0)
    for i in range(len(a)):
        acc += Float64(a[i]) * Float64(a[i])
    return sqrt(acc / Float64(len(a)))


def _report(
    label: String, host: List[Float32], dev: List[Float32]
) raises -> Bool:
    """One stage's verdict. Prints cos, max_abs and the host RMS — the RMS is
    there so max_abs can be read RELATIVE to the signal's own scale instead of
    as a bare number."""
    var cos = _cosine(host, dev)
    var mx = _max_abs(host, dev)
    var scale = _rms(host)
    var ok = cos >= COS_BAR
    var verdict = "PASS" if ok else "FAIL"
    print(
        "  ", verdict, " ", label,
        "  n=", len(host),
        "  cos=", cos,
        "  max_abs=", mx,
        "  host_rms=", scale,
    )
    return ok


def _check_filter_symmetry(ref weights: MiniMaxH3AudioWeights) raises -> Bool:
    """PRECONDITION for reusing `ops/activation1d`: it skips the transposed-conv
    kernel flip, which is only valid for a palindromic filter. See file header
    for the alignment derivation."""
    var names = weights.names.copy()
    var count = 0
    var worst = Float32(0.0)
    for i in range(len(names)):
        if names[i].find("filter") < 0:
            continue
        var f = weights.get(names[i])
        if len(f) != 12:
            print("   FAIL filter", names[i], "has length", len(f), "!= 12")
            return False
        count += 1
        for k in range(len(f)):
            var d = f[k] - f[len(f) - 1 - k]
            if d < 0:
                d = -d
            if d > worst:
                worst = d
    if count == 0:
        print("   FAIL: no Kaiser filter tensors found in the checkpoint")
        return False
    var ok = worst == Float32(0.0)
    var verdict = "PASS" if ok else "FAIL"
    print(
        "  ", verdict, "  filter symmetry   ", count,
        "filters, all len 12, worst |w - reverse(w)| =", worst,
    )
    return ok


def main() raises:
    var args = argv()
    var do_perf = False
    for i in range(len(args)):
        if String(args[i]) == "perf":
            do_perf = True

    print("=== MiniMax-H3 audio decoder: DEVICE vs HOST ORACLE ===")
    print("  oracle:  serenitymojo/models/minimax_h3/audio_decoder.mojo")
    print("  weights:", AUDIO_VAE_PATH)
    print("  bar:     cos >=", COS_BAR, " (max_abs reported, not gated)")
    print("")

    var t0 = perf_counter_ns()
    var weights = _load_host_weights(String(AUDIO_VAE_PATH))
    print(
        "  loaded", len(weights.names), "F32 tensors in",
        Float64(perf_counter_ns() - t0) / 1.0e9, "s",
    )
    var config = _config()

    var checks = 0
    var failures = 0

    # ── check 0: the reuse precondition, before any numbers ────────────────
    print("")
    print("  -- precondition --")
    checks += 1
    if not _check_filter_symmetry(weights):
        failures += 1
        print("")
        print("  filter symmetry FAILED — ops/activation1d's no-flip upsample")
        print("  decomposition is INVALID for this checkpoint. Every numeric")
        print("  result below would be meaningless. Stopping.")
        print("FAIL: 1 of 1 checks")
        return

    var ctx = DeviceContext()
    var t_up = perf_counter_ns()
    var dev_weights = minimax_h3_audio_device_weights(weights, config, ctx)
    print(
        "   uploaded", len(dev_weights.names),
        "device tensors (weight-norm folded) in",
        Float64(perf_counter_ns() - t_up) / 1.0e9, "s",
    )

    var latents = _synthetic_latents(LATENT_CHANNELS, GATE_LATENTS)
    var latents_dev = Tensor.from_host(
        latents, [1, LATENT_CHANNELS, GATE_LATENTS], STDtype.F32, ctx
    )
    print("")
    print("  -- staged parity, T =", GATE_LATENTS, "latents --")

    # ── check 1: pre ───────────────────────────────────────────────────────
    var host_pre = minimax_h3_audio_decode_pre(
        weights, config, latents, GATE_LATENTS
    )
    var dev_pre = minimax_h3_audio_decode_device_pre(
        dev_weights, config, latents_dev, ctx
    )
    checks += 1
    if not _report("pre (dec_in_proj+conv_pre)", host_pre.data, dev_pre.to_host(ctx)):
        failures += 1

    # ── checks 2-8: the seven upsample stages ──────────────────────────────
    for s in range(1, len(config.decoder_rates) + 1):
        var host_stage = minimax_h3_audio_decode_stages(
            weights, config, latents, GATE_LATENTS, s
        )
        var dev_stage = minimax_h3_audio_decode_device_stages(
            dev_weights, config, latents_dev, s, ctx
        )
        var label = (
            String("stage ") + String(s) + " [C=" + String(host_stage.channels)
            + ", L=" + String(host_stage.length) + "]"
        )
        checks += 1
        if not _report(label, host_stage.data, dev_stage.to_host(ctx)):
            failures += 1

    # ── check 9: tail (unclamped) ──────────────────────────────────────────
    var host_full_stage = minimax_h3_audio_decode_stages(
        weights, config, latents, GATE_LATENTS, -1
    )
    var dev_full_stage = minimax_h3_audio_decode_device_stages(
        dev_weights, config, latents_dev, -1, ctx
    )
    var host_tail = minimax_h3_audio_decode_tail(weights, host_full_stage)
    var dev_tail = minimax_h3_audio_decode_device_tail(
        dev_weights, dev_full_stage, ctx
    )
    checks += 1
    if not _report("tail (activation_post+conv_post)", host_tail, dev_tail.to_host(ctx)):
        failures += 1

    # ── check 10: end to end, clamped, exactly as the pipeline calls it ────
    var host_wave = minimax_h3_audio_decode(weights, config, latents, GATE_LATENTS)
    var dev_wave = minimax_h3_audio_decode_device(
        dev_weights, config, latents, GATE_LATENTS, ctx
    )
    checks += 1
    if not _report("end-to-end waveform (clamped)", host_wave, dev_wave):
        failures += 1
    print(
        "   waveform length", len(host_wave), "=", GATE_LATENTS, "latents x 800",
    )

    # ── optional: the standard 37-latent decode, timed ─────────────────────
    if do_perf:
        print("")
        print("  -- perf, T =", PERF_LATENTS, "latents (0.925 s of audio) --")
        var perf_latents = _synthetic_latents(LATENT_CHANNELS, PERF_LATENTS)

        var th0 = perf_counter_ns()
        var host_perf = minimax_h3_audio_decode(
            weights, config, perf_latents, PERF_LATENTS
        )
        var host_s = Float64(perf_counter_ns() - th0) / 1.0e9

        var td0 = perf_counter_ns()
        var dev_perf = minimax_h3_audio_decode_device(
            dev_weights, config, perf_latents, PERF_LATENTS, ctx
        )
        var dev_s = Float64(perf_counter_ns() - td0) / 1.0e9

        checks += 1
        if not _report("37-latent waveform", host_perf, dev_perf):
            failures += 1
        print("   host  ", host_s, "s")
        print("   device", dev_s, "s")
        print("   speedup", host_s / dev_s, "x")

    print("")
    if failures == 0:
        print("PASS:", checks, "checks")
    else:
        print("FAIL:", failures, "of", checks, "checks")
