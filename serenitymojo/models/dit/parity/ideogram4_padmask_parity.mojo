# ideogram4_padmask_parity.mojo — Ideogram-4 product-head padding gate.
#
# Exercises the exact H=18, Dh=256 conditional-attention geometry on a small
# aligned non-square sequence: 32 text slots + an 8x12 image grid = 128 tokens.
# The production helper must match a directly packed cuDNN tail-mask call on all
# semantic rows and leave the original padded-text query rows zero after scatter.

from max.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.tensor import Tensor
from serenitymojo.ops.attention import sdpa_qwen_flash_padmask
from serenitymojo.ops.attention_flash import sdpa_flash_fwd_padmask


def _shape(B: Int, S: Int, H: Int, Dh: Int) -> List[Int]:
    return [B, S, H, Dh]


def _fill(n: Int, mul: Int, modulus: Int, center: Float32, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * mul) % modulus) - center) * scale)
    return out^


def _compact(
    src: List[Float32], S: Int, H: Int, Dh: Int,
    n_txt: Int, real_txt_len: Int,
) -> List[Float32]:
    var out = List[Float32]()
    var real_total = real_txt_len + (S - n_txt)
    for compact_s in range(S):
        var src_s: Int
        if compact_s < real_txt_len:
            src_s = compact_s
        elif compact_s < real_total:
            src_s = n_txt + compact_s - real_txt_len
        else:
            src_s = real_txt_len + compact_s - real_total
        for h in range(H):
            var base = (src_s * H + h) * Dh
            for d in range(Dh):
                out.append(src[base + d])
    return out^


def _semantic_original(
    src: List[Float32], S: Int, H: Int, Dh: Int,
    n_txt: Int, real_txt_len: Int,
) -> List[Float32]:
    var out = List[Float32]()
    for s in range(S):
        if s >= real_txt_len and s < n_txt:
            continue
        for h in range(H):
            var base = (s * H + h) * Dh
            for d in range(Dh):
                out.append(src[base + d])
    return out^


def _semantic_compact_prefix(
    src: List[Float32], real_total: Int, H: Int, Dh: Int
) -> List[Float32]:
    var out = List[Float32]()
    for s in range(real_total):
        for h in range(H):
            var base = (s * H + h) * Dh
            for d in range(Dh):
                out.append(src[base + d])
    return out^


def main() raises:
    comptime B = 1
    comptime S = 128
    comptime H = 18
    comptime Dh = 256
    comptime N_TXT = 32
    var real_txt_len = 19
    var real_total = real_txt_len + (S - N_TXT)
    var n = B * S * H * Dh
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ctx = DeviceContext()

    var q_src = _fill(n, 7, 31, 15.0, 0.025)
    var k_src = _fill(n, 11, 37, 18.0, 0.02)
    var v_src = _fill(n, 13, 41, 20.0, 0.015)

    var packed = sdpa_qwen_flash_padmask[B, S, H, Dh, N_TXT](
        Tensor.from_host(q_src.copy(), _shape(B, S, H, Dh), STDtype.BF16, ctx),
        Tensor.from_host(k_src.copy(), _shape(B, S, H, Dh), STDtype.BF16, ctx),
        Tensor.from_host(v_src.copy(), _shape(B, S, H, Dh), STDtype.BF16, ctx),
        real_txt_len, scale, ctx,
    )
    var direct = sdpa_flash_fwd_padmask[B, S, H, Dh](
        Tensor.from_host(
            _compact(q_src^, S, H, Dh, N_TXT, real_txt_len),
            _shape(B, S, H, Dh), STDtype.BF16, ctx,
        ),
        Tensor.from_host(
            _compact(k_src^, S, H, Dh, N_TXT, real_txt_len),
            _shape(B, S, H, Dh), STDtype.BF16, ctx,
        ),
        Tensor.from_host(
            _compact(v_src^, S, H, Dh, N_TXT, real_txt_len),
            _shape(B, S, H, Dh), STDtype.BF16, ctx,
        ),
        real_total, scale, ctx,
    )

    var packed_h = packed.to_host(ctx)
    var direct_h = direct.o.to_host(ctx)
    var harness = ParityHarness()
    var result = harness.compare_host(
        _semantic_original(packed_h.copy(), S, H, Dh, N_TXT, real_txt_len),
        _semantic_compact_prefix(direct_h^, real_total, H, Dh),
    )
    print("ideogram4 H18/D256 padmask semantic rows:", result)

    var pad_nonzero = 0
    for s in range(real_txt_len, N_TXT):
        for h in range(H):
            var base = (s * H + h) * Dh
            for d in range(Dh):
                if packed_h[base + d] != 0.0:
                    pad_nonzero += 1
    print("ideogram4 padded query nonzero elements:", pad_nonzero)
    if not result.passed or pad_nonzero != 0:
        raise Error("ideogram4 product-head padding parity failed")
    print("PASS: ideogram4 product-head padding parity")
