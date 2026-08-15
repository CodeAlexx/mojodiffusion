# minimax_h3_final_train_parity — gate the H3 final-layer training twin
# (fwd + frozen backward d_hidden) against torch autograd on REAL FL2VA
# final_layer weights (h3_final_oracle.py). Bars: outputs cos >= 0.999,
# d_hidden cos >= 0.999.
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.tensor import Tensor
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.minimax_h3.h3_final_train import (
    H3FinalTrainWeights, h3_final_train_forward, h3_final_train_backward,
)

comptime ORACLE = "/home/alex/mojodiffusion/output/checks/h3_final_oracle.safetensors"
comptime CKPT = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/transformer"
comptime S = 384
comptime EPS = Float32(1.0e-5)

comptime TArc = ArcPointer[Tensor]


def _load(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    return Tensor.from_view(
        from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name)), ctx
    )


def _bf16_shard(
    sh: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> TArc:
    return TArc(cast_tensor(
        Tensor.from_view(sh.tensor_view(name), ctx), STDtype.BF16, ctx
    ))


def _ints(st: SafeTensors, name: String, ctx: DeviceContext) raises -> List[Int]:
    var h = _load(st, name, ctx).to_host(ctx)
    var out = List[Int]()
    for i in range(len(h)):
        out.append(Int(h[i]))
    return out^


def _cos(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Float64:
    var ah = a.to_host(ctx)
    var bh = b.to_host(ctx)
    if len(ah) != len(bh):
        raise Error("cos: length mismatch")
    var dot = Float64(0)
    var na = Float64(0)
    var nb = Float64(0)
    for i in range(len(ah)):
        var x = Float64(ah[i])
        var y = Float64(bh[i])
        dot += x * y
        na += x * x
        nb += y * y
    return dot / ((na**0.5) * (nb**0.5) + 1e-30)


def main() raises:
    var ctx = DeviceContext()
    var orc = SafeTensors.open(String(ORACLE))
    var sh = ShardedSafeTensors.open(String(CKPT))

    var w = H3FinalTrainWeights(
        _bf16_shard(sh, String("final_layer.norm.weight"), ctx),
        _bf16_shard(sh, String("final_layer.video_out.weight"), ctx),
        _bf16_shard(sh, String("final_layer.video_out.bias"), ctx),
        _bf16_shard(sh, String("final_layer.audio_out.weight"), ctx),
        _bf16_shard(sh, String("final_layer.audio_out.bias"), ctx),
    )

    var hidden = _load(orc, String("hidden"), ctx)
    var mod = _load(orc, String("mod"), ctx)
    var ts_idx = _ints(orc, String("ts_idx"), ctx)
    var video_idx = _ints(orc, String("video_idx"), ctx)
    var audio_idx = _ints(orc, String("audio_idx"), ctx)

    var fwd = h3_final_train_forward(
        hidden, w, mod, ts_idx, video_idx, audio_idx, EPS, ctx
    )
    ctx.synchronize()

    var ok = True
    var c_v = _cos(fwd.video[], _load(orc, String("video_out"), ctx), ctx)
    print("cos(video_out) =", c_v)
    if c_v < 0.999:
        ok = False
    var c_a = _cos(fwd.audio[], _load(orc, String("audio_out"), ctx), ctx)
    print("cos(audio_out) =", c_a)
    if c_a < 0.999:
        ok = False

    var d_video = _load(orc, String("d_video"), ctx)
    var d_audio = _load(orc, String("d_audio"), ctx)
    var d_hidden = h3_final_train_backward(
        d_video, d_audio, fwd.saved, w, ts_idx, video_idx, audio_idx,
        S, EPS, ctx,
    )
    ctx.synchronize()
    var c_dh = _cos(d_hidden, _load(orc, String("d_hidden"), ctx), ctx)
    print("cos(d_hidden) =", c_dh)
    if c_dh < 0.999:
        ok = False

    if ok:
        print("PASS: h3 final layer train fwd+bwd matches torch on real weights")
    else:
        print("FAIL: h3 final layer parity below bar")
        raise Error("minimax_h3_final_train_parity failed")
