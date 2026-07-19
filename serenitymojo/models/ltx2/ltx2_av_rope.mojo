# serenitymojo/models/ltx2/ltx2_av_rope.mojo
#
# LTX-2.3 joint-AV RoPE table builder (P6.2 runner prerequisite). Builds the 4
# rope pairs the AV stack consumes (video self-3D, audio self-1D, cross-modal
# video-side, cross-modal audio-side) from the comptime geometry — the runner
# needs these but they lived only in pipeline/ltx2_t2v_av_mvp.mojo run() (a
# main() module, un-importable), so ported here op-for-op.
#
# Source (byte-for-byte port): pipeline/ltx2_t2v_av_mvp.mojo
#   _compute_rope           :302-367
#   _build_video_coords     :370-393
#   _build_audio_coords     :396-411
#   _video_temporal_coords  :414-421
#   _mp1/_mp3               :424-429
#   4-pair assembly (run()) :655-666
# CAUSAL_OFFSET = 1 is MUSUBI-CORRECT: musubi ltx_2/components/patchifiers.py:240
# ("+1 ensures the timestamp corresponds to the first fully-available sample"),
# official ltx2_python_reference.py:111, MVP ltx2_t2v_av_mvp.mojo:128, SimpleTuner
# ltxvideo2 default. (The AV oracle's prior CAUSAL_OFFSET=0 was musubi-wrong and
# was corrected 2026-07-18; see scripts/ltx2_av_block_bwd_oracle.py:65.)
#
# F32 output (the training stack runs F32 per block, to_f32; the oracle rope dump
# is F32 too — gate parity/ltx2_av_rope_parity.mojo, cos>=0.999999 + max_abs
# ~1e-7). Emits [P*num_heads, half_head] rows (token-major, head-minor) — the
# SAME layout the committed stack gates feed the block forward (their _load_rope
# transposes the oracle's [H,P,hrd] to [P*H,hrd]).

from std.gpu.host import DeviceContext
from std.collections import List
from std.math import cos as fcos, sin as fsin, pow as fpow, pi

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

# model constants (MVP :115-133 — LTX-2.3 22B AV)
comptime VD = 4096          # video inner_dim
comptime AD = 2048          # audio inner_dim
comptime V_HEADS = 32
comptime A_HEADS = 32
comptime CA_DIM = 2048      # audio_cross_attention_dim
comptime ROPE_THETA = Float64(10000.0)
comptime POS_EMBED_MAX_POS = Float64(20.0)
comptime BASE_HW = Float64(2048.0)
comptime CAUSAL_OFFSET = Float64(1.0)   # MUSUBI-CORRECT (see header)
comptime VAE_SF0 = Float64(8.0)
comptime VAE_SF1 = Float64(32.0)
comptime VAE_SF2 = Float64(32.0)
comptime AUDIO_SCALE_FACTOR = Float64(4.0)
comptime FRAME_RATE = Float64(25.0)


def _sh2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^


# ── _compute_rope: coords [num_pos_dims, P, 2] -> (cos, sin) [P*num_heads, hrd] ─
# op-for-op port of the MVP _compute_rope; F32 output.
def _compute_rope(
    coords: List[Float64],      # [num_pos_dims * P * 2] row-major [d,p,2]
    num_pos_dims: Int, P: Int, dim: Int,
    max_pos: List[Float64], theta: Float64, num_heads: Int, ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    var num_rope_elems = num_pos_dims * 2
    var freq_count = dim // num_rope_elems
    var half_dim = dim // 2
    var rope_freqs = freq_count * num_pos_dims
    var pad = half_dim - rope_freqs
    if pad < 0:
        raise Error("rope: rope_freqs > half_dim")

    var denom = Float64(freq_count - 1)
    if denom < 1.0:
        denom = 1.0
    var freq = List[Float64]()
    for i in range(freq_count):
        freq.append(fpow(theta, Float64(i) / denom) * pi / 2.0)

    var cos_tok = List[Float32]()
    var sin_tok = List[Float32]()
    cos_tok.resize(P * half_dim, Float32(0.0))
    sin_tok.resize(P * half_dim, Float32(0.0))
    var half_head = (dim // 2) // num_heads

    for p in range(P):
        var scaled = List[Float64]()
        for d in range(num_pos_dims):
            var s0 = coords[(d * P + p) * 2 + 0]
            var e0 = coords[(d * P + p) * 2 + 1]
            var mid = (s0 + e0) * 0.5
            var g = mid / max_pos[d]
            scaled.append(2.0 * g - 1.0)
        var base = p * half_dim
        for q in range(pad):
            cos_tok[base + q] = Float32(1.0)
            sin_tok[base + q] = Float32(0.0)
        var off = base + pad
        for i in range(freq_count):
            for d in range(num_pos_dims):
                var a = scaled[d] * freq[i]
                cos_tok[off] = Float32(fcos(a))
                sin_tok[off] = Float32(fsin(a))
                off += 1

    # relayout half_dim -> [num_heads, head_dim/2], emit (token,head) rows
    var cos_rows = List[Float32]()
    var sin_rows = List[Float32]()
    for p in range(P):
        for h in range(num_heads):
            var src = p * half_dim + h * half_head
            for j in range(half_head):
                cos_rows.append(cos_tok[src + j])
                sin_rows.append(sin_tok[src + j])
    var sh = _sh2(P * num_heads, half_head)
    var cos_t = Tensor.from_host(cos_rows, sh.copy(), STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin_rows, sh^, STDtype.F32, ctx)
    return (cos_t^, sin_t^)


def _build_video_coords[NF: Int, NH: Int, NW: Int]() -> List[Float64]:
    comptime S_V = NF * NH * NW
    var out = List[Float64]()
    out.resize(3 * S_V * 2, Float64(0.0))
    var vae_t = VAE_SF0
    for f in range(NF):
        for h in range(NH):
            for w in range(NW):
                var tok = f * NH * NW + h * NW + w
                var fs = Float64(f) * VAE_SF0
                var fe = Float64(f + 1) * VAE_SF0
                var fsc = fs + CAUSAL_OFFSET - vae_t
                var fec = fe + CAUSAL_OFFSET - vae_t
                if fsc < 0.0: fsc = 0.0
                if fec < 0.0: fec = 0.0
                fsc = fsc / FRAME_RATE
                fec = fec / FRAME_RATE
                out[(0 * S_V + tok) * 2 + 0] = fsc
                out[(0 * S_V + tok) * 2 + 1] = fec
                out[(1 * S_V + tok) * 2 + 0] = Float64(h) * VAE_SF1
                out[(1 * S_V + tok) * 2 + 1] = Float64(h + 1) * VAE_SF1
                out[(2 * S_V + tok) * 2 + 0] = Float64(w) * VAE_SF2
                out[(2 * S_V + tok) * 2 + 1] = Float64(w + 1) * VAE_SF2
    return out^


def _build_audio_coords[S_A: Int]() -> List[Float64]:
    var out = List[Float64]()
    out.resize(S_A * 2, Float64(0.0))
    var mel_to_sec = 16000.0 / 160.0
    var scale = AUDIO_SCALE_FACTOR
    for t in range(S_A):
        var ms = Float64(t) * scale
        var me = Float64(t + 1) * scale
        var msc = ms + CAUSAL_OFFSET - scale
        var mec = me + CAUSAL_OFFSET - scale
        if msc < 0.0: msc = 0.0
        if mec < 0.0: mec = 0.0
        out[t * 2 + 0] = msc / mel_to_sec
        out[t * 2 + 1] = mec / mel_to_sec
    return out^


def _video_temporal_coords[NF: Int, NH: Int, NW: Int](vc: List[Float64]) -> List[Float64]:
    comptime S_V = NF * NH * NW
    var out = List[Float64]()
    out.resize(S_V * 2, Float64(0.0))
    for p in range(S_V):
        out[p * 2 + 0] = vc[(0 * S_V + p) * 2 + 0]
        out[p * 2 + 1] = vc[(0 * S_V + p) * 2 + 1]
    return out^


def _mp1() -> List[Float64]:
    var m = List[Float64](); m.append(POS_EMBED_MAX_POS); return m^


def _mp3() -> List[Float64]:
    var m = List[Float64]()
    m.append(POS_EMBED_MAX_POS); m.append(BASE_HW); m.append(BASE_HW)
    return m^


# ── the 4 AV rope pairs (MVP run() :655-666) ──────────────────────────────────
struct LTX2AVRope(Movable):
    var v_cos: Tensor
    var v_sin: Tensor
    var a_cos: Tensor
    var a_sin: Tensor
    var ca_v_cos: Tensor
    var ca_v_sin: Tensor
    var ca_a_cos: Tensor
    var ca_a_sin: Tensor

    def __init__(
        out self, var v_cos: Tensor, var v_sin: Tensor,
        var a_cos: Tensor, var a_sin: Tensor,
        var ca_v_cos: Tensor, var ca_v_sin: Tensor,
        var ca_a_cos: Tensor, var ca_a_sin: Tensor,
    ):
        self.v_cos = v_cos^; self.v_sin = v_sin^
        self.a_cos = a_cos^; self.a_sin = a_sin^
        self.ca_v_cos = ca_v_cos^; self.ca_v_sin = ca_v_sin^
        self.ca_a_cos = ca_a_cos^; self.ca_a_sin = ca_a_sin^


def ltx2_av_build_rope[NF: Int, NH: Int, NW: Int, S_A: Int](
    ctx: DeviceContext
) raises -> LTX2AVRope:
    comptime S_V = NF * NH * NW
    var vc = _build_video_coords[NF, NH, NW]()
    var ac = _build_audio_coords[S_A]()
    var vtc = _video_temporal_coords[NF, NH, NW](vc)

    var vrope = _compute_rope(vc, 3, S_V, VD, _mp3(), ROPE_THETA, V_HEADS, ctx)
    var arope = _compute_rope(ac, 1, S_A, AD, _mp1(), ROPE_THETA, A_HEADS, ctx)
    var cavrope = _compute_rope(vtc, 1, S_V, CA_DIM, _mp1(), ROPE_THETA, V_HEADS, ctx)
    var caarope = _compute_rope(ac, 1, S_A, CA_DIM, _mp1(), ROPE_THETA, A_HEADS, ctx)

    return LTX2AVRope(
        vrope[0].clone(ctx), vrope[1].clone(ctx),
        arope[0].clone(ctx), arope[1].clone(ctx),
        cavrope[0].clone(ctx), cavrope[1].clone(ctx),
        caarope[0].clone(ctx), caarope[1].clone(ctx),
    )
