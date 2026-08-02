# serenitymojo/models/minimax_h3/parity/minimax_h3_audio_encoder_parity.mojo
#
# MiniMax-H3 audio encoder parity gate: the DAC trunk and the causal-attention
# projection, on a tiny random-weight model, with no checkpoint in existence.
#
# Reference: diffusers PR huggingface/diffusers#14355 at head e1b518df, run by
# scripts/minimax_h3_audio_encoder_oracle.py against the reference's own
# classes with every encoder-side parameter re-randomized. The defaults would
# make this gate vacuous three times over: Snake1d alpha is ones (so a wrong
# per-channel broadcast passes), q_bias/v_bias are zeros (so the assembled
# cat(q_bias, ZERO, v_bias) is invisible), and LayerNorm is weight-1/bias-0 (so
# norm2 and mlp.norm cannot be told apart).
#
# STAGED, so a failure names the stage rather than the model:
#   [1] geometry      47 samples right-padded to 50, hop 10, 5 latent frames
#   [2] adaptive pool 6 -> 8 on uneven windows, checked analytically
#   [3] DAC trunk     both strides, all three dilations, the weight-norm fold
#   [4] pre_block     causal attention + GeGLU
#   [5] posterior mean, which is what H3 actually consumes
#
# BAR: 2e-5 absolute, as for the other forward gates.
#
# NEGATIVE CONTROL RUN, not assumed: making the attention bidirectional (the
# one wrong version that still produces entirely plausible numbers) fails [4] at
# max_abs 0.275 and [5] at 0.450 — about 13,000x the tolerance. The gate bites.
#
# Oracle: python3 scripts/minimax_h3_audio_encoder_oracle.py
# Run:
#   pixi run mojo build -O2 -j 1 -I . -I vendor/mojo-libs -Xlinker -lm \
#     serenitymojo/models/minimax_h3/parity/minimax_h3_audio_encoder_parity.mojo \
#     -o output/checks/minimax_h3_audio_encoder_parity \
#   && output/checks/minimax_h3_audio_encoder_parity

from std.collections import List
from std.pathlib import Path

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.minimax_h3.audio_encoder import (
    MiniMaxH3AudioEncoderConfig,
    MiniMaxH3AudioEncoderWeights,
    minimax_h3_audio_adaptive_avg_pool1d,
    minimax_h3_audio_encode,
    minimax_h3_audio_encode_preblock,
    minimax_h3_audio_encode_trunk,
)

comptime REF = "/home/alex/mojodiffusion/output/minimax_h3_audio/audio_encoder_ref.safetensors"
comptime KEYS = "/home/alex/mojodiffusion/output/minimax_h3_audio/audio_encoder_keys.txt"
comptime TOL = Float32(2.0e-5)

comptime ENCODER_DIM = 4
comptime LATENT_DIM = 24
comptime LATENT_CHANNELS = 8
comptime NUM_HEADS = 4


def _load_f32(ref st: SafeTensors, name: String) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.dtype != STDtype.F32:
        raise Error(String("_load_f32: unexpected dtype for ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32]()
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


def main() raises:
    print("MiniMax-H3 audio encoder parity gate")
    print("  reference:", REF)
    var st = SafeTensors.open(String(REF))
    var checks = 0
    var failures = 0

    # Load every key the oracle wrote, by the names it wrote. If the reference
    # renames anything, this fails here with the name rather than deep inside a
    # forward pass.
    var names = List[String]()
    var values = List[List[Float32]]()
    var key_lines = Path(String(KEYS)).read_text().split("\n")
    for i in range(len(key_lines)):
        ref line = key_lines[i]
        if line.byte_length() == 0:
            continue
        var name = String(line)
        names.append(name)
        values.append(_load_f32(st, String("w.") + name))
    print("  loaded", len(names), "weight tensors")
    var weights = MiniMaxH3AudioEncoderWeights(names^, values^)

    var rates = [2, 5]
    var config = MiniMaxH3AudioEncoderConfig(
        ENCODER_DIM, rates^, LATENT_DIM, LATENT_CHANNELS, NUM_HEADS, Float32(1.0e-5)
    )

    var samples = _load_f32(st, "in.samples")
    var latents = minimax_h3_audio_encode(weights, config, samples)

    print("")
    print("[1] geometry")
    checks += 1
    # 47 samples is deliberately NOT a multiple of the hop: the encoder must
    # right-pad to 50 and produce exactly 5 frames. A port that forgets the pad
    # produces 4 and silently drops the tail of every clip.
    if latents.channels == LATENT_CHANNELS and latents.frames == 5:
        print(
            "  ok   47 samples -> pad 50 -> [", latents.channels, ",",
            latents.frames, "] at hop", config.hop_length(),
        )
    else:
        failures += 1
        print("  FAIL [", latents.channels, ",", latents.frames, "], want [ 8 , 5 ]")

    print("")
    print("[2] adaptive average pool, 6 -> 8 on uneven windows")
    # Gated on its own and BEFORE the trunk, because it is the single most
    # likely thing to be written as a uniform stride: with head_dim 6 and
    # out_dim 8 the windows are lengths 1,2,2,1,1,2,2,1 and overlap. A uniform
    # implementation agrees whenever out divides in, which is exactly the case
    # a smaller fixture would have chosen.
    checks += 1
    var probe = List[Float32]()
    for i in range(6):
        probe.append(Float32(i + 1))
    var pooled = minimax_h3_audio_adaptive_avg_pool1d(probe, 1, 6, 8)
    # windows: [0,1) [0,2) [1,3) [2,3) [3,4) [3,5) [4,6) [5,6)
    var want_pool = [
        Float32(1.0), Float32(1.5), Float32(2.5), Float32(3.0),
        Float32(4.0), Float32(4.5), Float32(5.5), Float32(6.0),
    ]
    var pool_worst = _max_abs(pooled, want_pool)
    if pool_worst <= Float32(1.0e-6):
        print("  ok   windows 1,2,2,1,1,2,2,1 — not a uniform stride")
    else:
        failures += 1
        print("  FAIL pool max_abs", pool_worst)
        for i in range(8):
            print("       ", i, "got", pooled[i], "want", want_pool[i])

    print("")
    print("[3] DAC trunk — convolutions only")
    var trunk = minimax_h3_audio_encode_trunk(weights, config, samples)
    var want_trunk = _load_f32(st, "out.trunk")
    checks += 1
    if trunk.channels != LATENT_DIM or len(trunk.data) != len(want_trunk):
        failures += 1
        print(
            "  FAIL trunk [", trunk.channels, ",", trunk.frames, "],",
            len(trunk.data), "vs", len(want_trunk),
        )
    else:
        var tw = _max_abs(trunk.data, want_trunk)
        if tw <= TOL:
            print(
                "  ok   [", trunk.channels, ",", trunk.frames, "] max_abs", tw,
                "— both strides, dilations 1/3/9, weight-norm fold",
            )
        else:
            failures += 1
            print("  FAIL trunk max_abs", tw, "> tol", TOL)

    print("")
    print("[4] pre_block — causal attention + GeGLU")
    var want_pre = _load_f32(st, "out.pre_block")
    var pre = minimax_h3_audio_encode_preblock(weights, config, samples)
    checks += 1
    if pre.channels != LATENT_CHANNELS or len(pre.data) != len(want_pre):
        failures += 1
        print(
            "  FAIL pre_block [", pre.channels, ",", pre.frames, "],",
            len(pre.data), "vs", len(want_pre),
        )
    else:
        var pw = _max_abs(pre.data, want_pre)
        if pw <= TOL:
            print(
                "  ok   [", pre.channels, ",", pre.frames, "] max_abs", pw,
                "— zero k-bias, head mean-pool, uneven pool, double LayerNorm",
            )
        else:
            failures += 1
            print("  FAIL pre_block max_abs", pw, "> tol", TOL)

    print("")
    print("[5] posterior mean — the whole encode")
    var want_mean = _load_f32(st, "out.mean")
    checks += 1
    if len(latents.data) != len(want_mean):
        failures += 1
        print("  FAIL length", len(latents.data), "!=", len(want_mean))
    else:
        var worst = _max_abs(latents.data, want_mean)
        if worst <= TOL:
            print("  ok  ", len(latents.data), "values, max_abs", worst)
        else:
            failures += 1
            print("  FAIL mean max_abs", worst, "> tol", TOL)

    print("")
    if failures == 0:
        print("PASS:", checks, "checks")
    else:
        print("FAIL:", failures, "of", checks, "checks")
        raise Error("MiniMax-H3 audio encoder parity gate failed")
