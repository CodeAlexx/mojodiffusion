# LTX-2 resident-fp8 vs dequant-BF16 AV-block equivalence smoke (GATE 1 of the
# resident-fp8 speed build).
#
# Loads ONE fp8 inner block BOTH ways:
#   A) existing streamed path: LTX2BlockStream.load_block_bf16 (fp8 -> dequant
#      to BF16 once) -> LTX2AVBlockWeights.from_fp8_block -> vendor-BLAS linear
#   B) new resident path: LTX2BlockStream.load_block_fp8_resident (raw F8_E4M3
#      stays on GPU + per-row F32 scales) -> from_fp8_resident -> fused
#      linear_fp8 at every matmul call site
# then runs ltx2_block_forward_av on IDENTICAL non-degenerate (randn /
# sinusoidal-rope) inputs and compares video/audio/v2a outputs.
#
# NOTE: block 4 (not 0) — blocks 0-3 and 47 are BF16 boundary blocks with ZERO
# fp8 tensors, so block 0 would compare the bf16 path against itself and never
# exercise linear_fp8. Block 4 is the first inner block (34 fp8 matmuls).
#
# Expected: fp8->linear_fp8 (exact fp8 decode, F32 acc) vs fp8->dequant->BF16
# GEMM agree to bf16 weight rounding -> cos >= 0.9999 per output.
#
# Run:
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#       serenitymojo/pipeline/ltx2_fp8_block_equiv_smoke.mojo -o /tmp/ltx2_fp8_equiv
#   LD_LIBRARY_PATH=/home/alex/libtorch-cu124/libtorch/lib /tmp/ltx2_fp8_equiv

from std.gpu.host import DeviceContext
from std.math import sqrt, cos as fcos, sin as fsin

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import mul_scalar
from serenitymojo.offload.ltx2_block_stream import LTX2BlockStream, FP8Block
from serenitymojo.models.dit.ltx2_dit import (
    LTX2Config,
    LTX2AVBlockWeights,
    ltx2_block_forward_av,
)


comptime CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors"
comptime BLOCK_IDX = 4   # first fp8 inner block (0-3,47 are BF16 boundary)

comptime S_V = 128
comptime S_A = 16
comptime N_TXT = 128
comptime S_VPAD = 128
comptime S_APAD = 128
comptime EPS = Float32(1e-6)
comptime SEED = UInt64(20260609)

comptime VD = 4096
comptime AD = 2048
comptime CA_DIM = 2048
comptime HEADS = 32


def _sh2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^


def _sh3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); return s^


# Sinusoidal RoPE-shaped table: rows = (token, head) pairs, cols = head_dim/2.
# cos/sin of the SAME angle per cell (valid rotation), angle varies smoothly
# and non-degenerately with (row, col).
def _rope_pair(
    rows: Int, cols: Int, phase: Float64, ctx: DeviceContext
) raises -> Tuple[Tensor, Tensor]:
    var c = List[Float32]()
    var s = List[Float32]()
    for r in range(rows):
        for j in range(cols):
            var a = phase + 0.37 * Float64(r) * (Float64(j) + 1.0) / Float64(cols)
            c.append(Float32(fcos(a)))
            s.append(Float32(fsin(a)))
    var ct = Tensor.from_host(c, _sh2(rows, cols), STDtype.BF16, ctx)
    var st = Tensor.from_host(s, _sh2(rows, cols), STDtype.BF16, ctx)
    return (ct^, st^)


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


def _max_abs_diff(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Float64:
    var ha = a.to_host(ctx)
    var hb = b.to_host(ctx)
    if len(ha) != len(hb):
        raise Error("max_abs_diff: length mismatch")
    var m = 0.0
    for i in range(len(ha)):
        var d = Float64(ha[i]) - Float64(hb[i])
        if d < 0.0:
            d = -d
        if d > m:
            m = d
    return m


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

    print("=== LTX-2 resident-fp8 vs dequant-BF16 block equivalence smoke ===")
    print("  block:", BLOCK_IDX, " S_V/S_A/N_TXT:", S_V, S_A, N_TXT)
    print("  checkpoint:", CKPT)

    var stream = LTX2BlockStream.open(String(CKPT))
    var n_fp8 = stream.fp8_tensor_count(BLOCK_IDX)
    print("  fp8 tensors in block", BLOCK_IDX, ":", n_fp8)
    if n_fp8 == 0:
        raise Error("equiv smoke: chosen block has no fp8 tensors (boundary?)")

    # ── identical non-degenerate inputs (randn latents/contexts, scaled randn
    # modulation, sinusoidal rope) — BF16, the production activation dtype ──
    var hidden = randn(_sh3(1, S_V, VD), SEED + 0, STDtype.BF16, ctx)
    var ahs = randn(_sh3(1, S_A, AD), SEED + 1, STDtype.BF16, ctx)
    var enc = randn(_sh3(1, N_TXT, VD), SEED + 2, STDtype.BF16, ctx)
    var aenc = randn(_sh3(1, N_TXT, AD), SEED + 3, STDtype.BF16, ctx)
    var v_temb = mul_scalar(
        randn(_sh3(1, S_V, 9 * VD), SEED + 4, STDtype.BF16, ctx),
        Float32(0.2), ctx)
    var a_temb = mul_scalar(
        randn(_sh3(1, S_A, 9 * AD), SEED + 5, STDtype.BF16, ctx),
        Float32(0.2), ctx)
    var v_ca_ss = mul_scalar(
        randn(_sh3(1, 1, 4 * VD), SEED + 6, STDtype.BF16, ctx),
        Float32(0.2), ctx)
    var a_ca_ss = mul_scalar(
        randn(_sh3(1, 1, 4 * AD), SEED + 7, STDtype.BF16, ctx),
        Float32(0.2), ctx)
    var v_ca_gate = mul_scalar(
        randn(_sh3(1, 1, VD), SEED + 8, STDtype.BF16, ctx),
        Float32(0.2), ctx)
    var a_ca_gate = mul_scalar(
        randn(_sh3(1, 1, AD), SEED + 9, STDtype.BF16, ctx),
        Float32(0.2), ctx)
    var v_prompt_ts = mul_scalar(
        randn(_sh3(1, N_TXT, 2 * VD), SEED + 10, STDtype.BF16, ctx),
        Float32(0.2), ctx)
    var a_prompt_ts = mul_scalar(
        randn(_sh3(1, N_TXT, 2 * AD), SEED + 11, STDtype.BF16, ctx),
        Float32(0.2), ctx)

    var vrope = _rope_pair(S_V * HEADS, (VD // 2) // HEADS, 0.1, ctx)
    var arope = _rope_pair(S_A * HEADS, (AD // 2) // HEADS, 0.4, ctx)
    var cavrope = _rope_pair(S_V * HEADS, (CA_DIM // 2) // HEADS, 0.7, ctx)
    var caarope = _rope_pair(S_A * HEADS, (CA_DIM // 2) // HEADS, 1.1, ctx)

    _stats(String("hidden_in"), hidden, ctx)
    _stats(String("ahs_in"), ahs, ctx)

    # ── A) existing streamed dequant-BF16 path ──
    print("  [load A] load_block_bf16 (fp8 -> dequant BF16) -> from_fp8_block")
    var blk_a = stream.load_block_bf16(BLOCK_IDX, ctx)
    var w_deq = LTX2AVBlockWeights.from_fp8_block(blk_a^, cfg, ctx)

    # ── B) new resident-fp8 path ──
    print("  [load B] load_block_fp8_resident (raw fp8 + per-row scales)")
    var sc = FP8Block()
    var blk_b = stream.load_block_fp8_resident(BLOCK_IDX, sc, ctx)
    var w_fp8 = LTX2AVBlockWeights.from_fp8_resident(blk_b^, sc^, cfg, ctx)
    print("  [load B] per-row scale tensors:", len(w_fp8.scales))

    print("  [forward A] dequant-BF16 weights")
    var o1 = ltx2_block_forward_av[S_V, S_A, N_TXT, S_VPAD, S_APAD](
        w_deq, hidden, ahs, enc, aenc,
        v_temb, a_temb, v_ca_ss, a_ca_ss, v_ca_gate, a_ca_gate,
        v_prompt_ts, a_prompt_ts,
        vrope[0], vrope[1], arope[0], arope[1],
        cavrope[0], cavrope[1], caarope[0], caarope[1], EPS, ctx,
    )
    print("  [forward B] resident-fp8 weights (linear_fp8)")
    var o2 = ltx2_block_forward_av[S_V, S_A, N_TXT, S_VPAD, S_APAD](
        w_fp8, hidden, ahs, enc, aenc,
        v_temb, a_temb, v_ca_ss, a_ca_ss, v_ca_gate, a_ca_gate,
        v_prompt_ts, a_prompt_ts,
        vrope[0], vrope[1], arope[0], arope[1],
        cavrope[0], cavrope[1], caarope[0], caarope[1], EPS, ctx,
    )

    _stats(String("video A"), o1[0], ctx)
    _stats(String("video B"), o2[0], ctx)
    _stats(String("audio A"), o1[1], ctx)
    _stats(String("audio B"), o2[1], ctx)

    var v_cos = _cosine(o1[0], o2[0], ctx)
    var a_cos = _cosine(o1[1], o2[1], ctx)
    var v2a_cos = _cosine(o1[2], o2[2], ctx)
    var v_mad = _max_abs_diff(o1[0], o2[0], ctx)
    var a_mad = _max_abs_diff(o1[1], o2[1], ctx)
    var v2a_mad = _max_abs_diff(o1[2], o2[2], ctx)
    print("  >>> VIDEO  cos:", v_cos, " max_abs:", v_mad)
    print("  >>> AUDIO  cos:", a_cos, " max_abs:", a_mad)
    print("  >>> V2A    cos:", v2a_cos, " max_abs:", v2a_mad)

    if v_cos < 0.9999:
        raise Error(String("VIDEO equiv FAIL: cos=") + String(v_cos))
    if a_cos < 0.9999:
        raise Error(String("AUDIO equiv FAIL: cos=") + String(a_cos))
    if v2a_cos < 0.9999:
        raise Error(String("V2A equiv FAIL: cos=") + String(v2a_cos))
    print("LTX-2 resident-fp8 block EQUIVALENCE PASS")
