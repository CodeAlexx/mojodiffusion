# sdpa_qwen_keymask_parity.mojo - gate Qwen's no-square-mask attention path.
#
# The production Qwen path masks contiguous padded text key columns inside the
# online-softmax kernel. This gate compares it against the old full [B,H,S,S]
# additive-mask SDPA on tiny F32 and BF16 cases, without loading Qwen weights.

from std.math import sqrt
from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.tensor import Tensor
from serenitymojo.ops.attention import sdpa, sdpa_qwen_keymask


def _bshd(B: Int, S: Int, H: Int, Dh: Int) -> List[Int]:
    var sh = List[Int]()
    sh.append(B)
    sh.append(S)
    sh.append(H)
    sh.append(Dh)
    return sh^


def _mask_shape(B: Int, H: Int, S: Int) -> List[Int]:
    var sh = List[Int]()
    sh.append(B)
    sh.append(H)
    sh.append(S)
    sh.append(S)
    return sh^


def _fill_q(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 7) % 17) - 8.0) * Float32(0.07))
    return out^


def _fill_k(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 5) % 19) - 9.0) * Float32(0.06))
    return out^


def _fill_v(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 3) % 13) - 6.0) * Float32(0.05))
    return out^


def _qwen_full_mask(
    B: Int, H: Int, S: Int, N_TXT: Int, real_txt_len: Int
) raises -> List[Float32]:
    if real_txt_len < 0 or real_txt_len > N_TXT:
        raise Error("real_txt_len out of range")
    var out = List[Float32]()
    var total = B * H * S * S
    for _ in range(total):
        out.append(Float32(0.0))
    for b in range(B):
        for h in range(H):
            var head_base = ((b * H + h) * S) * S
            for q in range(S):
                var row_base = head_base + q * S
                for k in range(real_txt_len, N_TXT):
                    out[row_base + k] = Float32(-1.0e4)
    return out^


def main() raises:
    var ctx = DeviceContext()
    var h = ParityHarness()
    var all_pass = True

    comptime B = 1
    comptime S = 7
    comptime H = 2
    comptime Dh = 8
    comptime N_TXT = 4
    var real_txt_len = 2
    var n = B * S * H * Dh
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var full_f32 = sdpa[B, S, H, Dh](
        Tensor.from_host(_fill_q(n), _bshd(B, S, H, Dh), STDtype.F32, ctx),
        Tensor.from_host(_fill_k(n), _bshd(B, S, H, Dh), STDtype.F32, ctx),
        Tensor.from_host(_fill_v(n), _bshd(B, S, H, Dh), STDtype.F32, ctx),
        Tensor.from_host(
            _qwen_full_mask(B, H, S, N_TXT, real_txt_len),
            _mask_shape(B, H, S),
            STDtype.F32,
            ctx,
        ),
        scale,
        ctx,
    )
    var key_f32 = sdpa_qwen_keymask[B, S, H, Dh, N_TXT](
        Tensor.from_host(_fill_q(n), _bshd(B, S, H, Dh), STDtype.F32, ctx),
        Tensor.from_host(_fill_k(n), _bshd(B, S, H, Dh), STDtype.F32, ctx),
        Tensor.from_host(_fill_v(n), _bshd(B, S, H, Dh), STDtype.F32, ctx),
        real_txt_len,
        scale,
        ctx,
    )
    var r1 = h.compare_host(key_f32.to_host(ctx), full_f32.to_host(ctx))
    print("qwen keymask f32 vs full-mask sdpa:", r1)
    all_pass = all_pass and r1.passed

    var full_bf16 = sdpa[B, S, H, Dh](
        Tensor.from_host(_fill_q(n), _bshd(B, S, H, Dh), STDtype.BF16, ctx),
        Tensor.from_host(_fill_k(n), _bshd(B, S, H, Dh), STDtype.BF16, ctx),
        Tensor.from_host(_fill_v(n), _bshd(B, S, H, Dh), STDtype.BF16, ctx),
        Tensor.from_host(
            _qwen_full_mask(B, H, S, N_TXT, real_txt_len),
            _mask_shape(B, H, S),
            STDtype.BF16,
            ctx,
        ),
        scale,
        ctx,
    )
    var key_bf16 = sdpa_qwen_keymask[B, S, H, Dh, N_TXT](
        Tensor.from_host(_fill_q(n), _bshd(B, S, H, Dh), STDtype.BF16, ctx),
        Tensor.from_host(_fill_k(n), _bshd(B, S, H, Dh), STDtype.BF16, ctx),
        Tensor.from_host(_fill_v(n), _bshd(B, S, H, Dh), STDtype.BF16, ctx),
        real_txt_len,
        scale,
        ctx,
    )
    var r2 = h.compare_host(key_bf16.to_host(ctx), full_bf16.to_host(ctx))
    print("qwen keymask bf16 vs full-mask sdpa:", r2)
    all_pass = all_pass and r2.passed

    var no_pad = N_TXT
    var zero_full = sdpa[B, S, H, Dh](
        Tensor.from_host(_fill_q(n), _bshd(B, S, H, Dh), STDtype.F32, ctx),
        Tensor.from_host(_fill_k(n), _bshd(B, S, H, Dh), STDtype.F32, ctx),
        Tensor.from_host(_fill_v(n), _bshd(B, S, H, Dh), STDtype.F32, ctx),
        Tensor.from_host(
            _qwen_full_mask(B, H, S, N_TXT, no_pad),
            _mask_shape(B, H, S),
            STDtype.F32,
            ctx,
        ),
        scale,
        ctx,
    )
    var zero_key = sdpa_qwen_keymask[B, S, H, Dh, N_TXT](
        Tensor.from_host(_fill_q(n), _bshd(B, S, H, Dh), STDtype.F32, ctx),
        Tensor.from_host(_fill_k(n), _bshd(B, S, H, Dh), STDtype.F32, ctx),
        Tensor.from_host(_fill_v(n), _bshd(B, S, H, Dh), STDtype.F32, ctx),
        no_pad,
        scale,
        ctx,
    )
    var r3 = h.compare_host(zero_key.to_host(ctx), zero_full.to_host(ctx))
    print("qwen keymask no-pad f32 vs zero-mask sdpa:", r3)
    all_pass = all_pass and r3.passed

    if all_pass:
        print("PASS: qwen keymask attention matches full-mask SDPA gates")
    else:
        raise Error("sdpa_qwen_keymask_parity failed")
