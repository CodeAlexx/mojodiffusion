# ZImage B2 masked stack compile smoke.
#
# Evidence level: compile/runtime wiring smoke for the new non-graph masked B2
# stack APIs. Uses zero transformer blocks and tiny tensors; not model parity.
#
# Run:
#   pixi run mojo run -I . serenitymojo/models/zimage/parity/zimage_b2_masked_stack_compile_smoke.mojo

from std.gpu.host import DeviceContext
from std.math import cos, isfinite, sin

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.zimage.block import ZImageModVecs
from serenitymojo.models.zimage.weights import ZImageBlockWeights
from serenitymojo.models.zimage.zimage_stack_lora import (
    ZImageModVecsDevice,
    build_zimage_zero_lora_device_set,
    zimage_stack_lora_backward_main_device_b2_masked,
    zimage_stack_lora_forward_main_device_b2_masked,
)
from serenitymojo.tensor import Tensor


comptime H = 1
comptime Dh = 4
comptime D = H * Dh
comptime N_IMG = 2
comptime N_TXT = 2
comptime S = N_IMG + N_TXT
comptime F = 6
comptime OUT_CH = 3
comptime EPS = Float32(1.0e-5)
comptime FINAL_EPS = Float32(1.0e-6)


def _fill(n: Int, a: Float32, b: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(sin(Float32(i) * a + b) * Float32(0.05))
    return out^


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _i in range(n):
        out.append(Float32(0.0))
    return out^


def _rope(rows: Int, kind: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(rows * (Dh // 2)):
        if kind == 0:
            out.append(cos(Float32(i) * Float32(0.03)))
        else:
            out.append(sin(Float32(i) * Float32(0.03)))
    return out^


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("zimage_b2_masked_stack_compile_smoke FAILED: ") + msg)


def _all_finite(v: List[Float32]) -> Bool:
    for i in range(len(v)):
        if not isfinite(v[i]):
            return False
    return True


def main() raises:
    var ctx = DeviceContext()
    var empty_blocks = List[ZImageBlockWeights]()
    var empty_mod = List[ZImageModVecs]()
    var empty_moddev = List[ZImageModVecsDevice]()
    var lora = build_zimage_zero_lora_device_set(0, 0, 0, ctx)
    var final_lin_w = Tensor.from_host(
        _fill(OUT_CH * D, Float32(0.11), Float32(0.3)),
        [OUT_CH, D],
        STDtype.F32,
        ctx,
    )
    var final_lin_b = Tensor.from_host(_zeros(OUT_CH), [OUT_CH], STDtype.F32, ctx)
    var x_cos = Tensor.from_host(_rope(N_IMG * H, 0), [N_IMG * H, Dh // 2], STDtype.F32, ctx)
    var x_sin = Tensor.from_host(_rope(N_IMG * H, 1), [N_IMG * H, Dh // 2], STDtype.F32, ctx)
    var cap_cos0 = Tensor.from_host(_rope(N_TXT * H, 0), [N_TXT * H, Dh // 2], STDtype.F32, ctx)
    var cap_sin0 = Tensor.from_host(_rope(N_TXT * H, 1), [N_TXT * H, Dh // 2], STDtype.F32, ctx)
    var cap_cos1 = Tensor.from_host(_rope(N_TXT * H, 0), [N_TXT * H, Dh // 2], STDtype.F32, ctx)
    var cap_sin1 = Tensor.from_host(_rope(N_TXT * H, 1), [N_TXT * H, Dh // 2], STDtype.F32, ctx)
    var uni_cos2 = Tensor.from_host(_rope(2 * S * H, 0), [2 * S * H, Dh // 2], STDtype.F32, ctx)
    var uni_sin2 = Tensor.from_host(_rope(2 * S * H, 1), [2 * S * H, Dh // 2], STDtype.F32, ctx)

    var fwd = zimage_stack_lora_forward_main_device_b2_masked[H, Dh, N_IMG, N_TXT, S](
        _fill(N_IMG * D, Float32(0.03), Float32(0.1)),
        _fill(N_TXT * D, Float32(0.05), Float32(0.2)),
        _fill(N_IMG * D, Float32(0.07), Float32(0.3)),
        _fill(N_TXT * D, Float32(0.09), Float32(0.4)),
        N_TXT,
        N_TXT,
        S,
        S,
        empty_blocks,
        empty_mod,
        empty_mod,
        empty_blocks,
        empty_blocks,
        empty_moddev,
        lora,
        _zeros(2 * D),
        final_lin_w,
        final_lin_b,
        x_cos,
        x_sin,
        cap_cos0,
        cap_sin0,
        cap_cos1,
        cap_sin1,
        uni_cos2,
        uni_sin2,
        D,
        F,
        OUT_CH,
        EPS,
        FINAL_EPS,
        ctx,
    )
    _check(len(fwd.out0) == N_IMG * OUT_CH, "out0 length")
    _check(len(fwd.out1) == N_IMG * OUT_CH, "out1 length")
    _check(_all_finite(fwd.out0), "out0 finite")
    _check(_all_finite(fwd.out1), "out1 finite")

    var grads = zimage_stack_lora_backward_main_device_b2_masked[H, Dh, N_IMG, N_TXT, S](
        _fill(N_IMG * OUT_CH, Float32(0.13), Float32(0.5)),
        _fill(N_IMG * OUT_CH, Float32(0.17), Float32(0.6)),
        S,
        S,
        empty_blocks,
        empty_moddev,
        lora,
        _zeros(2 * D),
        final_lin_w,
        uni_cos2,
        uni_sin2,
        fwd,
        D,
        F,
        OUT_CH,
        EPS,
        FINAL_EPS,
        ctx,
    )
    _check(grads.nonfinite_lora_grads == 0, "no nonfinite grads")
    _check(len(grads.d_a) == 0 and len(grads.d_b) == 0, "zero-block grads empty")
    print("PASS: ZImage masked B2 stack APIs compile and run zero-block smoke")
