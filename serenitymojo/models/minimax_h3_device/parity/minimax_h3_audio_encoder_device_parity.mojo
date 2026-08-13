# GPU-only MiniMax-H3 reference AudioVAE encoder gate.
# Uses the already accepted vendor-derived real-weight oracle; no host model
# inference runs in this gate.

from std.collections import List
from max.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.minimax_h3.audio_encoder import (
    MiniMaxH3AudioEncoderConfig,
    MiniMaxH3AudioEncoderWeights,
)
from serenitymojo.models.minimax_h3_device.audio_encoder_device import (
    minimax_h3_audio_encode_device,
    minimax_h3_audio_encode_preblock_device,
    minimax_h3_audio_encode_trunk_device,
    minimax_h3_audio_encoder_device_weights,
)

comptime CKPT = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/audio_vae/model.safetensors"
comptime ORACLE = "/home/alex/mojodiffusion/output/minimax_h3_audio/audio_real_weights_ref.safetensors"
comptime COS_BAR = Float64(0.999)


def _load_f32(st: SafeTensors, name: String) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var view = from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name))
    if view.dtype != STDtype.F32:
        raise Error(String("audio encoder gate: non-F32 tensor ") + name)
    var ptr = view.data.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32](capacity=view.numel())
    for i in range(view.numel()):
        out.append(ptr[i])
    return out^


def _host_weights() raises -> MiniMaxH3AudioEncoderWeights:
    var st = SafeTensors.open(String(CKPT))
    var all = st.names()
    var names = List[String]()
    var values = List[List[Float32]]()
    for i in range(len(all)):
        if all[i].find("zero_k_bias") >= 0:
            continue
        if not (
            all[i].startswith("encoder.")
            or all[i].startswith("pre_block.")
            or all[i].startswith("mean_proj.")
        ):
            continue
        names.append(String(all[i]))
        values.append(_load_f32(st, String(all[i])))
    return MiniMaxH3AudioEncoderWeights(names^, values^)


def _config() -> MiniMaxH3AudioEncoderConfig:
    return MiniMaxH3AudioEncoderConfig(
        64, [2, 4, 4, 5, 5], 2048, 32, 8, Float32(1.0e-5)
    )


def _report(label: String, got: List[Float32], want: List[Float32]) raises -> Bool:
    if len(got) != len(want):
        print("  FAIL", label, "length", len(got), "!=", len(want))
        return False
    var dot = Float64(0.0)
    var ng = Float64(0.0)
    var nw = Float64(0.0)
    var max_abs = Float32(0.0)
    for i in range(len(got)):
        var g = Float64(got[i])
        var w = Float64(want[i])
        dot += g * w
        ng += g * g
        nw += w * w
        var d = got[i] - want[i]
        if d < 0:
            d = -d
        if d > max_abs:
            max_abs = d
    var cosine = dot / (sqrt(ng) * sqrt(nw))
    var ok = cosine >= COS_BAR
    print(
        " ", "PASS" if ok else "FAIL", label,
        "cosine=", cosine, "max_abs=", max_abs,
    )
    return ok


def main() raises:
    var oracle = SafeTensors.open(String(ORACLE))
    var samples = _load_f32(oracle, "in.samples")
    var host = _host_weights()
    var config = _config()
    var ctx = DeviceContext()
    var weights = minimax_h3_audio_encoder_device_weights(host, config, ctx)
    var failures = 0

    var trunk = minimax_h3_audio_encode_trunk_device(
        weights, config, samples, ctx
    ).to_host(ctx)
    if not _report("trunk", trunk, _load_f32(oracle, "out.trunk")):
        failures += 1

    var pre = minimax_h3_audio_encode_preblock_device(
        weights, config, samples, ctx
    ).to_host(ctx)
    if not _report("pre_block", pre, _load_f32(oracle, "out.pre_block")):
        failures += 1

    var mean = minimax_h3_audio_encode_device(
        weights, config, samples, ctx
    )
    if not _report("mean", mean.data, _load_f32(oracle, "out.mean")):
        failures += 1
    if failures != 0:
        raise Error(String("MiniMax-H3 audio encoder device failures=") + String(failures))
    print("PASS: MiniMax-H3 reference AudioVAE encoder is GPU-only")
