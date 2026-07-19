# LTX-2 joint-AV transformer block-0 parity smoke.
#
# This gates the current inference spine:
#   serenitymojo.models.dit.ltx2_dit.ltx2_block_forward_av
#
# The oracle is the production-shaped Rust-port block dump:
#   scripts/ltx2_av_block0_parity.py ->
#   output/ltx2_av_block0/av_block0_ref.safetensors
#
# The oracle is generated from the distilled-FP8 checkpoint. This is
# intentionally a block-math gate, not the inner FP8 streaming gate; block 0 is
# a boundary block stored BF16.
#
# Run:
#   pixi run mojo run -I . -Xlinker -lm -Xlinker -lcuda \
#       serenitymojo/pipeline/ltx2_av_block_parity_smoke.mojo

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.dit.ltx2_dit import (
    LTX2Config,
    LTX2AVBlockWeights,
    ltx2_block_forward_av,
)


comptime CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors"
comptime REF = "/home/alex/mojodiffusion/output/ltx2_av_block0/av_block0_ref.safetensors"

comptime S_V = 16
comptime S_A = 8
comptime N_TXT = 12
comptime S_VPAD = 16
comptime S_APAD = 16
comptime EPS = Float32(1e-6)


def _shape2(a: Int, b: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    return s^


def _load_f32(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    return cast_tensor(Tensor.from_view_as_bf16(st.tensor_view(name), ctx), STDtype.F32, ctx)


def _load_rope(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    # Stored as [H, S, Dh/2]. The DIT block expects [S * H, Dh/2] in (s,h)
    # row order, matching the full-forward velocity smoke.
    var tv = st.tensor_view(name)
    var sh = tv.shape.copy()
    if len(sh) != 3:
        raise Error(String("rope table ") + name + " not rank-3")
    var H = sh[0]
    var S = sh[1]
    var hrd = sh[2]
    var t = cast_tensor(Tensor.from_view_as_bf16(tv, ctx), STDtype.F32, ctx)
    var host = t.to_host(ctx)
    var out = List[Float32]()
    for _ in range(S * H * hrd):
        out.append(Float32(0.0))
    for h in range(H):
        for s in range(S):
            for j in range(hrd):
                out[(s * H + h) * hrd + j] = host[(h * S + s) * hrd + j]
    return Tensor.from_host(out, _shape2(S * H, hrd), STDtype.F32, ctx)


def _cosine(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Float64:
    var ha = a.to_host(ctx)
    var hb = b.to_host(ctx)
    if len(ha) != len(hb):
        raise Error("cosine: length mismatch")
    var dot = 0.0
    var na = 0.0
    var nb = 0.0
    for i in range(len(ha)):
        var x = Float64(ha[i])
        var y = Float64(hb[i])
        if x != x or y != y:
            raise Error("cosine: NaN")
        dot += x * y
        na += x * x
        nb += y * y
    return dot / (sqrt(na) * sqrt(nb) + 1e-30)


def _rel_l2(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Float64:
    var ha = a.to_host(ctx)
    var hb = b.to_host(ctx)
    if len(ha) != len(hb):
        raise Error("rel_l2: length mismatch")
    var diff_sq = 0.0
    var ref_sq = 0.0
    for i in range(len(ha)):
        var d = Float64(ha[i]) - Float64(hb[i])
        var r = Float64(hb[i])
        diff_sq += d * d
        ref_sq += r * r
    return sqrt(diff_sq) / (sqrt(ref_sq) + 1e-30)


def _stats(name: String, t: Tensor, ctx: DeviceContext) raises:
    var h = t.to_host(ctx)
    var s = 0.0
    var amax = 0.0
    for i in range(len(h)):
        var v = Float64(h[i])
        s += v
        var av = v if v >= 0.0 else -v
        if av > amax:
            amax = av
    print("  ", name, "mean:", Float32(s / Float64(len(h))),
          "absmax:", Float32(amax))


def main() raises:
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()

    print("=== LTX-2 joint-AV block-0 parity smoke ===")
    print("  S_V/S_A/N_TXT:", S_V, S_A, N_TXT)
    print("  oracle:", REF)
    print("  checkpoint:", CKPT)

    var dump = ShardedSafeTensors.open(String(REF))

    var hidden = _load_f32(dump, "hs", ctx)
    var ahs = _load_f32(dump, "ahs", ctx)
    var enc = _load_f32(dump, "enc_hs", ctx)
    var aenc = _load_f32(dump, "audio_enc_hs", ctx)
    var v_temb = _load_f32(dump, "v_timestep", ctx)
    var a_temb = _load_f32(dump, "a_timestep", ctx)
    var v_ca_ss = _load_f32(dump, "v_ca_ss", ctx)
    var a_ca_ss = _load_f32(dump, "a_ca_ss", ctx)
    var v_ca_gate = _load_f32(dump, "v_ca_gate", ctx)
    var a_ca_gate = _load_f32(dump, "a_ca_gate", ctx)
    var v_prompt_ts = _load_f32(dump, "video_prompt_ts", ctx)
    var a_prompt_ts = _load_f32(dump, "audio_prompt_ts", ctx)

    var v_cos = _load_rope(dump, "v_cos", ctx)
    var v_sin = _load_rope(dump, "v_sin", ctx)
    var a_cos = _load_rope(dump, "a_cos", ctx)
    var a_sin = _load_rope(dump, "a_sin", ctx)
    var ca_v_cos = _load_rope(dump, "ca_v_cos", ctx)
    var ca_v_sin = _load_rope(dump, "ca_v_sin", ctx)
    var ca_a_cos = _load_rope(dump, "ca_a_cos", ctx)
    var ca_a_sin = _load_rope(dump, "ca_a_sin", ctx)

    var video_ref = _load_f32(dump, "video_out", ctx)
    var audio_ref = _load_f32(dump, "audio_out", ctx)
    var v2a_delta_ref = _load_f32(dump, "v2a_delta", ctx)

    _stats(String("hidden_in"), hidden, ctx)
    _stats(String("ahs_in"), ahs, ctx)

    print("  [load] block-0 AV weights")
    var weights = LTX2AVBlockWeights.load(String(CKPT), 0, cfg, ctx).to_f32(ctx)
    print("  [forward] full dual-stream AV block-0")
    var outs = ltx2_block_forward_av[S_V, S_A, N_TXT, S_VPAD, S_APAD](
        weights,
        hidden, ahs, enc, aenc,
        v_temb, a_temb,
        v_ca_ss, a_ca_ss, v_ca_gate, a_ca_gate,
        v_prompt_ts, a_prompt_ts,
        v_cos, v_sin, a_cos, a_sin,
        ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin,
        EPS, ctx,
    )
    ref video_out = outs[0]
    ref audio_out = outs[1]
    ref v2a_delta_mojo = outs[2]

    _stats(String("video_out"), video_out, ctx)
    _stats(String("audio_out"), audio_out, ctx)
    _stats(String("video_ref"), video_ref, ctx)
    _stats(String("audio_ref"), audio_ref, ctx)
    _stats(String("v2a_delta_mojo"), v2a_delta_mojo, ctx)
    _stats(String("v2a_delta_ref"), v2a_delta_ref, ctx)

    var v_cos_sim = _cosine(video_out, video_ref, ctx)
    var a_cos_sim = _cosine(audio_out, audio_ref, ctx)
    var v2a_cos = _cosine(v2a_delta_mojo, v2a_delta_ref, ctx)
    var v_rl2 = _rel_l2(video_out, video_ref, ctx)
    var a_rl2 = _rel_l2(audio_out, audio_ref, ctx)
    var v2a_rl2 = _rel_l2(v2a_delta_mojo, v2a_delta_ref, ctx)
    print("  >>> VIDEO cos:", Float32(v_cos_sim), " relL2:", Float32(v_rl2))
    print("  >>> AUDIO cos:", Float32(a_cos_sim), " relL2:", Float32(a_rl2))
    print("  >>> V2A-delta cos:", Float32(v2a_cos), " relL2:",
          Float32(v2a_rl2))

    if v_cos_sim < 0.999:
        raise Error(String("VIDEO parity FAIL: cos=") + String(v_cos_sim))
    if a_cos_sim < 0.999:
        raise Error(String("AUDIO parity FAIL: cos=") + String(a_cos_sim))
    if v2a_cos < 0.999:
        raise Error(String("V2A-DELTA parity FAIL: cos=") + String(v2a_cos))
    if v2a_rl2 >= 1e-2:
        raise Error(String("V2A-DELTA parity FAIL: relL2=") + String(v2a_rl2))
    print("LTX-2 joint-AV block-0 PARITY PASS")
