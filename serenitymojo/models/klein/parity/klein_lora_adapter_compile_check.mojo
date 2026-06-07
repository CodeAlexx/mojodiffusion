# Tiny compile/run check for the core Klein OneTrainer-style LoRA adapter helper.

from std.collections import List
from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.klein.lora_adapter import (
    _lora_adamw,
    lora_backward,
    lora_forward,
    make_lora_adapter,
)
from serenitymojo.tensor import Tensor


def _fill(n: Int, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32(i % 7) - Float32(3.0)) * scale)
    return out^


def main() raises:
    var ctx = DeviceContext()
    var lo = make_lora_adapter(2, Float32(4.0), 4, 3, UInt64(7))
    var x = Tensor.from_host(_fill(8, Float32(0.01)), [2, 4], STDtype.BF16, ctx)
    var y = lora_forward(x, lo, 2, ctx)
    var dy = Tensor.from_host(_fill(6, Float32(0.02)), [2, 3], STDtype.BF16, ctx)
    var grads = lora_backward(x, dy, lo, 2, ctx)
    _lora_adamw(lo, grads, 1, Float32(1.0e-3), ctx)
    print(
        "[klein-lora-adapter] PASS",
        "y_numel=", y.numel(),
        "d_a=", len(grads.d_a),
        "d_b=", len(grads.d_b),
    )
