# LTX-2 joint-AV transformer block-0 PARITY smoke (Plan P3 — keystone gate).
#
# Loads the Python oracle dump (scripts/ltx2_av_block0_parity.py ->
# output/ltx2_av_block0/av_block0_ref.safetensors) — deterministic block-0
# inputs + precomputed per-token / global / prompt modulation tensors + RoPE
# cos/sin tables + the gate targets video_out / audio_out — plus the REAL
# block-0 weights from the distilled-fp8 checkpoint (block 0 is a boundary block
# stored BF16). Runs the FULL dual-stream AV block forward
# (ltx2_block_forward_av: 6 attention paths) and GATES cosine_similarity >=
# 0.999 vs the oracle on BOTH video_out and audio_out.
#
# The oracle dumps the RoPE tables in the Rust per-head layout [H, S, head_dim/2]
# (b,h,s) row order. The Mojo block applies RoPE on BSHD [1,S,H,Dh] which the
# rope_halfsplit kernel flattens in (s,h) row order, so we transpose each table
# to [S*H, head_dim/2] (s,h) on load.
#
# Run:  pixi run mojo run serenitymojo/pipeline/ltx2_av_block_parity_smoke.mojo

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.dit.ltx2_dit import (
    LTX2Config,
    LTX2AVBlockWeights,
    ltx2_block_forward_av,
)


comptime CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors"
comptime REF = "/home/alex/mojodiffusion/output/ltx2_av_block0/av_block0_ref.safetensors"

# Must match scripts/ltx2_av_block0_parity.py shape constants.
comptime S_V = 16
comptime S_A = 8
comptime N_TXT = 12
comptime S_VPAD = 16   # max(S_V, N_TXT)
comptime S_APAD = 16   # max(S_A, N_TXT, S_V)
comptime EPS = Float32(1e-6)


def _shape2(a: Int, b: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    return s^


# Load a named F32 tensor from the dump, cast to BF16 on the device.
def _load_bf16(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var tv = st.tensor_view(name)
    return Tensor.from_view_as_bf16(tv, ctx)


# Load a rope table dumped as [H, S, hrd] (F32) and transpose to [S*H, hrd]
# (s,h) row order as BF16, matching the BSHD rope_halfsplit flatten.
def _load_rope(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var tv = st.tensor_view(name)
    var sh = tv.shape.copy()
    if len(sh) != 3:
        raise Error(String("rope table ") + name + " not rank-3")
    var H = sh[0]
    var S = sh[1]
    var hrd = sh[2]
    # Read F32 host data.
    var t = Tensor.from_view_as_bf16(tv, ctx)  # [H,S,hrd] BF16 on device
    var host = t.to_host(ctx)                  # F32 host, row-major (h,s,j)
    var out = List[Float32]()
    for _ in range(S * H * hrd):
        out.append(Float32(0.0))
    for h in range(H):
        for s in range(S):
            for j in range(hrd):
                var src = (h * S + s) * hrd + j
                var dst = (s * H + h) * hrd + j
                out[dst] = host[src]
    return Tensor.from_host(out, _shape2(S * H, hrd), STDtype.BF16, ctx)


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
    """Relative L2: ||a - b|| / (||b|| + 1e-30)."""
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
    print("  ", name, "mean:", Float32(s / Float64(len(h))), "absmax:", Float32(amax))


def main() raises:
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()

    print("=== LTX-2 joint-AV block-0 PARITY smoke (P3 keystone) ===")
    print("  S_V/S_A/N_TXT:", S_V, S_A, N_TXT)

    print("  [load] oracle dump:", REF)
    var dump = ShardedSafeTensors.open(String(REF))

    var hidden = _load_bf16(dump, "hs", ctx)
    var ahs = _load_bf16(dump, "ahs", ctx)
    var enc = _load_bf16(dump, "enc_hs", ctx)
    var aenc = _load_bf16(dump, "audio_enc_hs", ctx)
    var v_temb = _load_bf16(dump, "v_timestep", ctx)
    var a_temb = _load_bf16(dump, "a_timestep", ctx)
    var v_ca_ss = _load_bf16(dump, "v_ca_ss", ctx)
    var a_ca_ss = _load_bf16(dump, "a_ca_ss", ctx)
    var v_ca_gate = _load_bf16(dump, "v_ca_gate", ctx)
    var a_ca_gate = _load_bf16(dump, "a_ca_gate", ctx)
    var v_prompt_ts = _load_bf16(dump, "video_prompt_ts", ctx)
    var a_prompt_ts = _load_bf16(dump, "audio_prompt_ts", ctx)

    var v_cos = _load_rope(dump, "v_cos", ctx)
    var v_sin = _load_rope(dump, "v_sin", ctx)
    var a_cos = _load_rope(dump, "a_cos", ctx)
    var a_sin = _load_rope(dump, "a_sin", ctx)
    var ca_v_cos = _load_rope(dump, "ca_v_cos", ctx)
    var ca_v_sin = _load_rope(dump, "ca_v_sin", ctx)
    var ca_a_cos = _load_rope(dump, "ca_a_cos", ctx)
    var ca_a_sin = _load_rope(dump, "ca_a_sin", ctx)

    var video_ref = _load_bf16(dump, "video_out", ctx)
    var audio_ref = _load_bf16(dump, "audio_out", ctx)
    var v2a_delta_ref = _load_bf16(dump, "v2a_delta", ctx)

    _stats(String("hidden_in"), hidden, ctx)
    _stats(String("ahs_in"), ahs, ctx)

    print("  [load] block-0 AV weights")
    var weights = LTX2AVBlockWeights.load(String(CKPT), 0, cfg, ctx)
    print("  [load] done")

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
    var v2a_rl2 = _rel_l2(v2a_delta_mojo, v2a_delta_ref, ctx)
    print("  >>> VIDEO cos:", Float32(v_cos_sim))
    print("  >>> AUDIO cos:", Float32(a_cos_sim))
    print("  >>> V2A-delta cos:", Float32(v2a_cos), "  relL2:", Float32(v2a_rl2))

    if v_cos_sim < 0.999:
        raise Error(String("VIDEO parity FAIL: cos=") + String(v_cos_sim))
    if a_cos_sim < 0.999:
        raise Error(String("AUDIO parity FAIL: cos=") + String(a_cos_sim))
    # V2A-DELTA gate: directly verifies the video-to-audio cross-modal sub-path.
    # cos < 0.999 or relL2 >= 1e-2 means the v2a path is corrupted.
    if v2a_cos < 0.999:
        raise Error(String("V2A-DELTA parity FAIL: cos=") + String(v2a_cos)
                    + String(" (< 0.999)"))
    if v2a_rl2 >= 1e-2:
        raise Error(String("V2A-DELTA parity FAIL: relL2=") + String(v2a_rl2)
                    + String(" (>= 1e-2)"))
    print("LTX-2 joint-AV block-0 PARITY PASS (video & audio cos >= 0.999, v2a-delta cos >= 0.999 & relL2 < 1e-2)")
