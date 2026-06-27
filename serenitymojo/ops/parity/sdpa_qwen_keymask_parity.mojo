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
from serenitymojo.ops.attention import sdpa, sdpa_qwen_keymask, sdpa_qwen_flash_padmask


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


def _semantic_qwen_rows(
    data: List[Float32], B: Int, S: Int, H: Int, Dh: Int, N_TXT: Int, real_txt_len: Int
) -> List[Float32]:
    """Keep real text plus image-token rows, skipping unused padded text queries."""
    var out = List[Float32]()
    for b in range(B):
        for s in range(S):
            if s >= real_txt_len and s < N_TXT:
                continue
            for h in range(H):
                var base = ((b * S + s) * H + h) * Dh
                for d in range(Dh):
                    out.append(data[base + d])
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

    # Aligned BF16 product shape for the new cuDNN repack path. Padded text query
    # rows are intentionally not compared: they are not keys in later attention
    # and never contribute to the final image prediction.
    comptime S_FLASH = 128
    comptime H_FLASH = 2
    comptime Dh_FLASH = 128
    comptime N_TXT_FLASH = 32
    var real_txt_len_flash = 19
    var n_flash = B * S_FLASH * H_FLASH * Dh_FLASH
    var scale_flash = Float32(1.0) / sqrt(Float32(Dh_FLASH))
    var key_flash_ref = sdpa_qwen_keymask[B, S_FLASH, H_FLASH, Dh_FLASH, N_TXT_FLASH](
        Tensor.from_host(_fill_q(n_flash), _bshd(B, S_FLASH, H_FLASH, Dh_FLASH), STDtype.BF16, ctx),
        Tensor.from_host(_fill_k(n_flash), _bshd(B, S_FLASH, H_FLASH, Dh_FLASH), STDtype.BF16, ctx),
        Tensor.from_host(_fill_v(n_flash), _bshd(B, S_FLASH, H_FLASH, Dh_FLASH), STDtype.BF16, ctx),
        real_txt_len_flash,
        scale_flash,
        ctx,
    )
    var flash = sdpa_qwen_flash_padmask[B, S_FLASH, H_FLASH, Dh_FLASH, N_TXT_FLASH](
        Tensor.from_host(_fill_q(n_flash), _bshd(B, S_FLASH, H_FLASH, Dh_FLASH), STDtype.BF16, ctx),
        Tensor.from_host(_fill_k(n_flash), _bshd(B, S_FLASH, H_FLASH, Dh_FLASH), STDtype.BF16, ctx),
        Tensor.from_host(_fill_v(n_flash), _bshd(B, S_FLASH, H_FLASH, Dh_FLASH), STDtype.BF16, ctx),
        real_txt_len_flash,
        scale_flash,
        ctx,
    )
    var r4 = h.compare_host(
        _semantic_qwen_rows(
            flash.to_host(ctx), B, S_FLASH, H_FLASH, Dh_FLASH, N_TXT_FLASH, real_txt_len_flash
        ),
        _semantic_qwen_rows(
            key_flash_ref.to_host(ctx), B, S_FLASH, H_FLASH, Dh_FLASH, N_TXT_FLASH, real_txt_len_flash
        ),
    )
    print("qwen flash bf16 semantic rows vs keymask:", r4)
    all_pass = all_pass and r4.passed

    if all_pass:
        print("PASS: qwen keymask/flash attention matches parity gates")
    else:
        raise Error("sdpa_qwen_keymask_parity failed")
