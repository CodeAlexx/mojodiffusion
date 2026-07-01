# sdpa_bwd_batched_mask_parity.mojo -- shared training SDPA batched-mask gate.
#
# Run:
#   pixi run mojo run -I . serenitymojo/ops/parity/sdpa_bwd_batched_mask_parity.mojo

from std.gpu.host import DeviceContext
from std.math import cos, isfinite, sin, sqrt

from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.attention_backward import sdpa_backward_masked
from serenitymojo.ops.attention_train import training_sdpa_backward_masked_batched_strict
from serenitymojo.parity import ParityHarness
from serenitymojo.tensor import Tensor


def _bshd(B: Int, S: Int, H: Int, Dh: Int) -> List[Int]:
    var shape = List[Int]()
    shape.append(B)
    shape.append(S)
    shape.append(H)
    shape.append(Dh)
    return shape^


def _bhss(B: Int, H: Int, S: Int) -> List[Int]:
    var shape = List[Int]()
    shape.append(B)
    shape.append(H)
    shape.append(S)
    shape.append(S)
    return shape^


def _bhs_rows(B: Int, H: Int, S: Int) -> List[Int]:
    var shape = List[Int]()
    shape.append(B * H * S)
    shape.append(S)
    return shape^


def _hs_rows(H: Int, S: Int) -> List[Int]:
    var shape = List[Int]()
    shape.append(H * S)
    shape.append(S)
    return shape^


def _q(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(sin(Float32(i) * 0.13 + 0.1) * 0.2)
    return out^


def _k(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(cos(Float32(i) * 0.11 + 0.3) * 0.2)
    return out^


def _v(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(sin(Float32(i) * 0.17 + 0.5) * 0.2)
    return out^


def _dout(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(cos(Float32(i) * 0.19 + 0.7) * 0.2)
    return out^


def _legacy_causal_tail_mask(H: Int, S: Int, ar_len: Int) -> List[Float32]:
    var out = List[Float32]()
    for _h in range(H):
        for q in range(S):
            for k in range(S):
                if q >= ar_len and k > q and k >= ar_len:
                    out.append(-10000.0)
                else:
                    out.append(0.0)
    return out^


def _batched_same_causal_tail_mask(B: Int, H: Int, S: Int, ar_len: Int) -> List[Float32]:
    var out = List[Float32]()
    for _b in range(B):
        for _h in range(H):
            for q in range(S):
                for k in range(S):
                    if q >= ar_len and k > q and k >= ar_len:
                        out.append(-10000.0)
                    else:
                        out.append(0.0)
    return out^


def _batched_key_tail_mask(B: Int, H: Int, S: Int, valid0: Int, valid1: Int) -> List[Float32]:
    var out = List[Float32]()
    for b in range(B):
        var valid = valid0
        if b == 1:
            valid = valid1
        for _h in range(H):
            for _q in range(S):
                for k in range(S):
                    if k >= valid:
                        out.append(-10000.0)
                    else:
                        out.append(0.0)
    return out^


def _check_finite(values: List[Float32], name: String) raises:
    for i in range(len(values)):
        if not isfinite(values[i]):
            raise Error(name + ": non-finite gradient value")


def main() raises:
    comptime B = 2
    comptime S = 6
    comptime H = 2
    comptime Dh = 8
    var ctx = DeviceContext()
    var h = ParityHarness(0.999999)
    var n = B * S * H * Dh
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var legacy = sdpa_backward_masked[B, S, H, Dh](
        Tensor.from_host(_q(n), _bshd(B, S, H, Dh), STDtype.F32, ctx),
        Tensor.from_host(_k(n), _bshd(B, S, H, Dh), STDtype.F32, ctx),
        Tensor.from_host(_v(n), _bshd(B, S, H, Dh), STDtype.F32, ctx),
        Tensor.from_host(_legacy_causal_tail_mask(H, S, 3), _hs_rows(H, S), STDtype.F32, ctx),
        Tensor.from_host(_dout(n), _bshd(B, S, H, Dh), STDtype.F32, ctx),
        scale,
        ctx,
    )
    var batched = training_sdpa_backward_masked_batched_strict[B, S, H, Dh](
        Tensor.from_host(_q(n), _bshd(B, S, H, Dh), STDtype.F32, ctx),
        Tensor.from_host(_k(n), _bshd(B, S, H, Dh), STDtype.F32, ctx),
        Tensor.from_host(_v(n), _bshd(B, S, H, Dh), STDtype.F32, ctx),
        Tensor.from_host(_batched_same_causal_tail_mask(B, H, S, 3), _bhss(B, H, S), STDtype.F32, ctx),
        Tensor.from_host(_dout(n), _bshd(B, S, H, Dh), STDtype.F32, ctx),
        scale,
        ctx,
    )
    var rdq = h.compare_host(batched.d_q.to_host(ctx), legacy.d_q.to_host(ctx))
    var rdk = h.compare_host(batched.d_k.to_host(ctx), legacy.d_k.to_host(ctx))
    var rdv = h.compare_host(batched.d_v.to_host(ctx), legacy.d_v.to_host(ctx))
    print("same-mask batched vs broadcast d_q:", rdq)
    print("same-mask batched vs broadcast d_k:", rdk)
    print("same-mask batched vs broadcast d_v:", rdv)
    if not rdq.passed or not rdk.passed or not rdv.passed:
        raise Error("sdpa batched-mask backward did not match broadcast mask")

    var zlike = training_sdpa_backward_masked_batched_strict[B, S, H, Dh](
        Tensor.from_host(_q(n), _bshd(B, S, H, Dh), STDtype.F32, ctx),
        Tensor.from_host(_k(n), _bshd(B, S, H, Dh), STDtype.F32, ctx),
        Tensor.from_host(_v(n), _bshd(B, S, H, Dh), STDtype.F32, ctx),
        Tensor.from_host(_batched_key_tail_mask(B, H, S, 5, 4), _bhs_rows(B, H, S), STDtype.F32, ctx),
        Tensor.from_host(_dout(n), _bshd(B, S, H, Dh), STDtype.F32, ctx),
        scale,
        ctx,
    )
    _check_finite(zlike.d_q.to_host(ctx), "zimage-like d_q")
    _check_finite(zlike.d_k.to_host(ctx), "zimage-like d_k")
    _check_finite(zlike.d_v.to_host(ctx), "zimage-like d_v")
    print("differing per-sample key-tail mask: PASS")
    print("ALL PASS")
