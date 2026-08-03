# serenitymojo/models/minimax_h3/parity/minimax_h3_audio_decoder_parity.mojo
#
# MiniMax-H3 audio decoder parity gate: the BigVGAN waveform path, on a tiny
# random-weight model, with no checkpoint in existence.
#
# Reference: diffusers PR huggingface/diffusers#14355 at head e1b518df, run by
# scripts/minimax_h3_audio_decoder_oracle.py against the reference's OWN test
# fixture config. Both sides consume the same parameters and the same latents.
#
# Two checks, in this order, because the second is meaningless without the
# first:
#   1. the weight-norm FOLD — `g * v / ||v||` — reproduces torch's effective
#      weight for every one of the 36 parametrized convolutions. The released
#      checkpoint stores `weight_g` / `weight_v`, NOT a folded `weight`; a
#      loader written against ComfyUI's converted description would find no
#      `weight` tensor at all.
#   2. the decoded waveform matches.
#
# BAR: 2e-5 absolute. Both sides are float32 but the convolution summation
# orders differ, and the alias-free activations run a resample-activate-resample
# round trip per block, so bit-exactness would be a test of BLAS ordering.
#
# Oracle: python3 scripts/minimax_h3_audio_decoder_oracle.py
# Run:
#   pixi run mojo build -O2 -j 1 -I . -I vendor/mojo-libs -Xlinker -lm \
#     serenitymojo/models/minimax_h3/parity/minimax_h3_audio_decoder_parity.mojo \
#     -o output/checks/minimax_h3_audio_decoder_parity \
#   && output/checks/minimax_h3_audio_decoder_parity

from std.collections import List

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.minimax_h3.audio_decoder import (
    MiniMaxH3AudioDecoderConfig,
    MiniMaxH3AudioWeights,
    fold_weight_norm,
    minimax_h3_audio_decode,
)

comptime REF = "/home/alex/mojodiffusion/output/minimax_h3_audio/audio_decoder_ref.safetensors"
comptime TOL = Float32(2.0e-5)
comptime NUM_LATENTS = 6


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


def _decoder_tensor_names() -> List[String]:
    """Every tensor the decoder path reads, in the reference's own naming."""
    var names = List[String]()
    names.append(String("dec_in_proj.weight"))
    names.append(String("dec_in_proj.bias"))
    names.append(String("decoder.conv_pre.weight_g"))
    names.append(String("decoder.conv_pre.weight_v"))
    names.append(String("decoder.conv_pre.bias"))
    names.append(String("decoder.conv_post.weight_g"))
    names.append(String("decoder.conv_post.weight_v"))
    names.append(String("decoder.activation_post.act.alpha"))
    names.append(String("decoder.activation_post.act.beta"))
    names.append(String("decoder.activation_post.upsample.filter"))
    names.append(String("decoder.activation_post.downsample.lowpass.filter"))
    for i in range(2):
        var up = String("decoder.ups.") + String(i) + ".0"
        names.append(up + ".weight_g")
        names.append(up + ".weight_v")
        names.append(up + ".bias")
    for block in range(4):
        var p = String("decoder.resblocks.") + String(block)
        for d in range(2):
            names.append(p + ".convs1." + String(d) + ".weight_g")
            names.append(p + ".convs1." + String(d) + ".weight_v")
            names.append(p + ".convs1." + String(d) + ".bias")
            names.append(p + ".convs2." + String(d) + ".weight_g")
            names.append(p + ".convs2." + String(d) + ".weight_v")
            names.append(p + ".convs2." + String(d) + ".bias")
        for a in range(4):
            var act = p + ".activations." + String(a)
            names.append(act + ".act.alpha")
            names.append(act + ".act.beta")
            names.append(act + ".upsample.filter")
            names.append(act + ".downsample.lowpass.filter")
    return names^


def main() raises:
    print("MiniMax-H3 audio decoder parity gate")
    print("  reference:", REF)
    var st = SafeTensors.open(String(REF))

    var checks = 0
    var failures = 0

    print("")
    print("[1] weight-norm fold vs torch's effective weight")
    var fold_cases = List[String]()
    fold_cases.append(String("decoder.conv_pre"))
    fold_cases.append(String("decoder.conv_post"))
    fold_cases.append(String("decoder.ups.0.0"))
    fold_cases.append(String("decoder.ups.1.0"))
    fold_cases.append(String("decoder.resblocks.0.convs1.0"))
    fold_cases.append(String("decoder.resblocks.3.convs2.1"))
    var worst_fold = Float32(0.0)
    for i in range(len(fold_cases)):
        var name = fold_cases[i]
        var g = _load_f32(st, String("wn.") + name + ".weight_g")
        var v = _load_f32(st, String("wn.") + name + ".weight_v")
        var want = _load_f32(st, String("wn.") + name + ".weight")
        var dim0 = len(g)
        var per_channel = len(v) // dim0
        var got = fold_weight_norm(g, v, dim0, per_channel)
        var worst = _max_abs(got, want)
        if worst > worst_fold:
            worst_fold = worst
        checks += 1
        if worst <= TOL:
            print("  ok  ", name, "max_abs", worst)
        else:
            failures += 1
            print("  FAIL", name, "max_abs", worst)
    print("  worst fold deviation:", worst_fold)

    print("")
    print("[2] decoded waveform")
    # The Kaiser resampling filters are registered BUFFERS, so they live under
    # `b.` in the dump while every learned tensor is under `w.`.
    var names = _decoder_tensor_names()
    var values = List[List[Float32]]()
    for i in range(len(names)):
        if names[i].endswith("filter"):
            values.append(_load_f32(st, String("b.") + names[i]))
        else:
            values.append(_load_f32(st, String("w.") + names[i]))
    var weights = MiniMaxH3AudioWeights(names^, values^)

    var rates = [2, 2]
    var kernels = [4, 4]
    var resblock_kernels = [3, 7]
    var dilations = List[List[Int]]()
    dilations.append([1, 3])
    dilations.append([1, 3])
    var config = MiniMaxH3AudioDecoderConfig(
        8, 32, 16, rates^, kernels^, resblock_kernels^, dilations^
    )

    var decoded = minimax_h3_audio_decode(
        weights, config, _load_f32(st, "in.latents"), NUM_LATENTS
    )
    var want_wave = _load_f32(st, "out.sample")
    checks += 1
    if len(decoded) != len(want_wave):
        failures += 1
        print("  FAIL waveform length", len(decoded), "!=", len(want_wave))
    else:
        var worst = _max_abs(decoded, want_wave)
        if worst <= TOL:
            print("  ok   waveform", len(decoded), "samples, max_abs", worst)
        else:
            failures += 1
            print("  FAIL waveform max_abs", worst)

    print("")
    if failures == 0:
        print("PASS:", checks, "checks")
    else:
        print("FAIL:", failures, "of", checks, "checks")
        raise Error("MiniMax-H3 audio decoder parity gate failed")
