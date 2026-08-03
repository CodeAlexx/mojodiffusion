# serenitymojo/models/minimax_h3/parity/minimax_h3_audio_real_weights_parity.mojo
#
# FIRST REAL-WEIGHT parity gate for the MiniMax-H3 port.
#
# Units 10 and 15 were gated on tiny random-weight fixtures because no
# checkpoint existed. This runs the same Mojo code against the ACTUAL released
# audio VAE at its ACTUAL config — 151.3 M parameters, 1087 tensors, read
# straight out of the released file rather than a re-exported copy.
#
# WHAT ONLY FULL SCALE CAN CATCH:
#   * five encoder stages at rates (2, 4, 4, 5, 5), three with ODD strides
#     whose `k=2*stride, padding=ceil(stride/2)` geometry is not L/stride. The
#     fixture had two stages.
#   * 172 real weight_norm pairs to fold, against six in the fixture. The
#     released file has NO plain `weight` tensor anywhere — measured — so the
#     fold is not optional.
#   * the causal attention runs 2048 -> 32 with 8 heads: head_dim 256, so the
#     adaptive average pool is an exact 8:1 divide. The fixture deliberately
#     used an UNEVEN 6 -> 8 to catch a uniform-stride bug; this is the other
#     side of that check.
#   * seven decoder stages of 18 convolutions each, with the alias-free
#     SnakeBeta activations and their 254 Kaiser filter buffers.
#
# BAR: 2e-4 absolute. Deliberately looser than the 2e-5 of the fixture gates,
# and the reason is scale, not indulgence: this stack is ~230 sequential
# convolutions deep at up to 2048 channels, all in host float32 with a
# different summation order from torch's. The fixture was ~20 ops deep. A bar
# that does not move with depth is a bar that fails for arithmetic reasons and
# teaches nothing. What it must still catch is any STRUCTURAL error — a wrong
# pad, a missed fold, the wrong snake — and those move the output by 1e-1, not
# 1e-4, as unit 15's bidirectional-attention control showed at 0.275.
#
# Oracle: python3 scripts/minimax_h3_audio_real_weights_oracle.py
# Run:
#   pixi run mojo build -O2 -j 1 -I . -I vendor/mojo-libs -Xlinker -lm \
#     serenitymojo/models/minimax_h3/parity/minimax_h3_audio_real_weights_parity.mojo \
#     -o output/checks/minimax_h3_audio_real_weights_parity \
#   && output/checks/minimax_h3_audio_real_weights_parity

from std.collections import List
from std.time import perf_counter_ns

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.minimax_h3.audio_decoder import (
    MiniMaxH3AudioDecoderConfig,
    MiniMaxH3AudioWeights,
    minimax_h3_audio_decode,
)
from serenitymojo.models.minimax_h3.audio_encoder import (
    MiniMaxH3AudioEncoderConfig,
    MiniMaxH3AudioEncoderWeights,
    minimax_h3_audio_encode,
    minimax_h3_audio_encode_preblock,
    minimax_h3_audio_encode_trunk,
)

comptime CKPT = "/home/alex/Downloads/MiniMax-H3-audio_vae.safetensors"
comptime REF = "/home/alex/mojodiffusion/output/minimax_h3_audio/audio_real_weights_ref.safetensors"
comptime TOL = Float32(2.0e-4)

# The released config, confirmed by an exact state-dict match against the
# reference's defaults (1087 tensors, 0 missing, 0 extra, 0 shape mismatches).
comptime ENCODER_DIM = 64
comptime LATENT_DIM = 2048
comptime LATENT_CHANNELS = 32
comptime NUM_HEADS = 8
comptime DECODER_DIM = 1024


def _load_f32(ref st: SafeTensors, name: String) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.dtype != STDtype.F32:
        raise Error(String("_load_f32: unexpected dtype for ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32](capacity=tv.numel())
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


def _max_abs(got: List[Float32], want: List[Float32]) raises -> Float32:
    if len(got) != len(want):
        raise Error("length mismatch")
    var worst = Float32(0.0)
    for i in range(len(got)):
        var diff = got[i] - want[i]
        var mag = -diff if diff < 0.0 else diff
        if mag > worst:
            worst = mag
    return worst


def _rms(x: List[Float32]) raises -> Float32:
    var acc = Float64(0.0)
    for i in range(len(x)):
        acc += Float64(x[i]) * Float64(x[i])
    return Float32((acc / Float64(len(x))) ** 0.5)


def main() raises:
    print("MiniMax-H3 audio VAE — REAL WEIGHT parity gate")
    print("  checkpoint:", CKPT)
    print("  reference :", REF)

    var ck = SafeTensors.open(String(CKPT))
    var orc = SafeTensors.open(String(REF))
    var checks = 0
    var failures = 0

    var t0 = perf_counter_ns()
    var all_names = ck.names()
    var names = List[String]()
    var values = List[List[Float32]]()
    var params = 0
    for i in range(len(all_names)):
        ref n = all_names[i]
        # zero_k_bias is a frozen zero buffer the port deliberately never reads;
        # loading it would be harmless but it is excluded to keep the contract
        # visible — measured max|.| = 0.0 in this very file.
        if n.find("zero_k_bias") >= 0:
            continue
        var v = _load_f32(ck, String(n))
        params += len(v)
        names.append(String(n))
        values.append(v^)
    var load_ms = Float64(perf_counter_ns() - t0) / 1.0e6
    print(
        "  loaded", len(names), "tensors,", Float64(params) / 1.0e6,
        "M params in", load_ms, "ms",
    )

    var samples = _load_f32(orc, "in.samples")
    print("  input", len(samples), "samples")

    # ── encoder ──────────────────────────────────────────────────────────────
    var enc_rates = [2, 4, 4, 5, 5]
    var enc_cfg = MiniMaxH3AudioEncoderConfig(
        ENCODER_DIM, enc_rates^, LATENT_DIM, LATENT_CHANNELS, NUM_HEADS,
        Float32(1.0e-5),
    )
    var enc_w = MiniMaxH3AudioEncoderWeights(names.copy(), values.copy())

    print("")
    print("[1] DAC trunk at full width  (5 stages, rates 2/4/4/5/5, hop 800)")
    t0 = perf_counter_ns()
    var trunk = minimax_h3_audio_encode_trunk(enc_w, enc_cfg, samples)
    var trunk_ms = Float64(perf_counter_ns() - t0) / 1.0e6
    var want_trunk = _load_f32(orc, "out.trunk")
    checks += 1
    if trunk.channels != LATENT_DIM or len(trunk.data) != len(want_trunk):
        failures += 1
        print(
            "  FAIL [", trunk.channels, ",", trunk.frames, "]",
            len(trunk.data), "vs", len(want_trunk),
        )
    else:
        var w = _max_abs(trunk.data, want_trunk)
        var rel = w / _rms(want_trunk)
        if w <= TOL:
            print(
                "  ok   [", trunk.channels, ",", trunk.frames, "] max_abs", w,
                " rel_rms", rel, " in", trunk_ms, "ms",
            )
        else:
            failures += 1
            print("  FAIL trunk max_abs", w, "rel_rms", rel, "> tol", TOL)

    print("")
    print("[2] pre_block at full width  (2048 -> 32, 8 heads, head_dim 256)")
    t0 = perf_counter_ns()
    var pre = minimax_h3_audio_encode_preblock(enc_w, enc_cfg, samples)
    var pre_ms = Float64(perf_counter_ns() - t0) / 1.0e6
    var want_pre = _load_f32(orc, "out.pre_block")
    checks += 1
    if pre.channels != LATENT_CHANNELS or len(pre.data) != len(want_pre):
        failures += 1
        print("  FAIL [", pre.channels, ",", pre.frames, "]")
    else:
        var w = _max_abs(pre.data, want_pre)
        var rel = w / _rms(want_pre)
        if w <= TOL:
            print(
                "  ok   [", pre.channels, ",", pre.frames, "] max_abs", w,
                " rel_rms", rel, " in", pre_ms, "ms",
            )
        else:
            failures += 1
            print("  FAIL pre_block max_abs", w, "rel_rms", rel, "> tol", TOL)

    print("")
    print("[3] posterior mean — the latents H3 actually consumes")
    var latents = minimax_h3_audio_encode(enc_w, enc_cfg, samples)
    var want_mean = _load_f32(orc, "out.mean")
    checks += 1
    if len(latents.data) != len(want_mean):
        failures += 1
        print("  FAIL length", len(latents.data), "!=", len(want_mean))
    else:
        var w = _max_abs(latents.data, want_mean)
        var rel = w / _rms(want_mean)
        if w <= TOL:
            print(
                "  ok  ", len(latents.data), "values, max_abs", w,
                " rel_rms", rel,
            )
        else:
            failures += 1
            print("  FAIL mean max_abs", w, "rel_rms", rel, "> tol", TOL)

    # ── decoder ──────────────────────────────────────────────────────────────
    print("")
    print("[4] BigVGAN decoder at full width  (7 stages x 18 convs, 254 Kaiser)")
    var dec_rates = [5, 5, 2, 2, 2, 2, 2]
    var dec_kernels = [9, 9, 4, 4, 4, 4, 4]
    var res_kernels = [3, 7, 11]
    var res_dils = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    var dec_cfg = MiniMaxH3AudioDecoderConfig(
        LATENT_CHANNELS, LATENT_DIM, DECODER_DIM, dec_rates^, dec_kernels^,
        res_kernels^, res_dils^,
    )
    var dec_w = MiniMaxH3AudioWeights(names^, values^)

    # Decode the REFERENCE's own mean, not ours, so a decoder failure cannot be
    # blamed on the encoder — the two halves stay independently attributable.
    t0 = perf_counter_ns()
    var wave = minimax_h3_audio_decode(dec_w, dec_cfg, want_mean, 4)
    var dec_ms = Float64(perf_counter_ns() - t0) / 1.0e6
    var want_wave = _load_f32(orc, "out.waveform")
    checks += 1
    if len(wave) != len(want_wave):
        failures += 1
        print("  FAIL length", len(wave), "!=", len(want_wave))
    else:
        var w = _max_abs(wave, want_wave)
        var rel = w / _rms(want_wave)
        if w <= TOL:
            print(
                "  ok  ", len(wave), "samples, max_abs", w, " rel_rms", rel,
                " in", dec_ms, "ms",
            )
        else:
            failures += 1
            print("  FAIL waveform max_abs", w, "rel_rms", rel, "> tol", TOL)

    print("")
    if failures == 0:
        print("PASS:", checks, "checks — the port matches the RELEASED weights")
    else:
        print("FAIL:", failures, "of", checks, "checks")
        raise Error("MiniMax-H3 audio real-weight parity gate failed")
