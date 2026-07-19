# zimage_device_grad_optimizer_smoke.mojo — ZImage v5 device-grad optimizer smoke.
#
# Run:
#   pixi run mojo run -I . serenitymojo/models/zimage/parity/zimage_device_grad_optimizer_smoke.mojo

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.models.zimage.lora_block import ZIMAGE_SLOTS
from serenitymojo.models.zimage.zimage_stack_lora import (
    build_zimage_lora_set,
    zimage_lora_adamw_step_main_only_device_grads,
    zimage_step_io_init,
)
from serenitymojo.training.lora_adamw_plain_fused import (
    lora_adamw_plain_device_state_init,
)


comptime TArc = ArcPointer[Tensor]


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("zimage_device_grad_optimizer_smoke FAILED: ") + msg)


def _grad_vals(n: Int, base: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        var sign = Float32(1.0) if i % 2 == 0 else Float32(-1.0)
        out.append(sign * (base + Float32(i + 1) * Float32(0.002)))
    return out^


def _absum_bf16(v: List[BFloat16]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(v)):
        var x = v[i].cast[DType.float32]()
        s += x if x >= Float32(0.0) else -x
    return s


def _adapter_absum(lora_a: List[BFloat16], lora_b: List[BFloat16]) -> Float32:
    return _absum_bf16(lora_a) + _absum_bf16(lora_b)


def main() raises:
    var ctx = DeviceContext()
    var final_bias = Tensor.from_host(
        [Float32(0.0), Float32(0.0)], [2], STDtype.F32, ctx
    )
    var io = zimage_step_io_init(
        1, 0, 4, 2,
        1, 1, 1, 4,
        final_bias, ctx,
    )
    var lora = build_zimage_lora_set(
        1, 1, 1, 4, 6, 2, Float32(4.0)
    )
    var start = lora.main_base() * ZIMAGE_SLOTS
    var end = lora.num_blocks() * ZIMAGE_SLOTS
    var state = lora_adamw_plain_device_state_init(
        lora.ad, start, end, ctx
    )

    var frozen_before = Float32(0.0)
    for i in range(start):
        frozen_before += _adapter_absum(lora.ad[i].a, lora.ad[i].b)

    for i in range(ZIMAGE_SLOTS):
        var flat = start + i
        io.grad_indices.append(flat)
        var na = len(lora.ad[flat].a)
        var nb = len(lora.ad[flat].b)
        io.grad_a.append(TArc(Tensor.from_host(
            _grad_vals(na, Float32(i + 1) * Float32(0.01)),
            [na],
            STDtype.F32,
            ctx,
        )))
        io.grad_b.append(TArc(Tensor.from_host(
            _grad_vals(nb, Float32(i + 1) * Float32(0.02)),
            [nb],
            STDtype.F32,
            ctx,
        )))

    var b0 = Float32(0.0)
    for i in range(start, end):
        b0 += _absum_bf16(lora.ad[i].b)
    _check(b0 == Float32(0.0), "B starts at zero")

    var dev_norm = zimage_lora_adamw_step_main_only_device_grads(
        lora,
        state,
        io,
        1,
        Float32(1.0e-3),
        ctx,
        Float32(0.9),
        Float32(0.999),
        Float32(1.0e-8),
        Float32(0.01),
        Float32(1.0),
        True,
        Float32(0.5),
    )
    _check(dev_norm > Float32(0.0), "device grad norm should be positive")

    var b1 = Float32(0.0)
    var moved = 0
    for i in range(start, end):
        var s = _absum_bf16(lora.ad[i].b)
        b1 += s
        if s > Float32(0.0):
            moved += 1
    _check(b1 > Float32(0.0), "B should move after device-grad AdamW")
    _check(moved == ZIMAGE_SLOTS, "all main slots should receive device grads")
    var frozen_after = Float32(0.0)
    for i in range(start):
        frozen_after += _adapter_absum(lora.ad[i].a, lora.ad[i].b)
    _check(frozen_after == frozen_before, "NR/CR adapters must remain unchanged")
    print("PASS: ZImage v5 device grads feed resident AdamW without host grad lists")
