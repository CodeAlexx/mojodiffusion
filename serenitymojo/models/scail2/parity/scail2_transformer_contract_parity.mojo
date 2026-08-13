# Numeric gates for SCAIL-only contracts that intentionally differ from the
# shared Wan path: exact-erf image MLP GELU and retained-F32 residual streams.

from std.collections import List
from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.scail2.scail2_block import (
    scail2_gated_residual_f32,
    scail2_residual_f32,
)
from serenitymojo.models.scail2.scail2_streamed_dit import (
    scail2_image_projection_activation,
)
from serenitymojo.ops.activations import gelu
from serenitymojo.tensor import Tensor


def _abs(x: Float32) -> Float32:
    return -x if x < 0.0 else x


def _require_close(
    got: List[Float32], expected: List[Float32], tolerance: Float32, label: String
) raises:
    if len(got) != len(expected):
        raise Error(String("SCAIL-2 parity length mismatch: ") + label)
    for i in range(len(got)):
        if _abs(got[i] - expected[i]) > tolerance:
            raise Error(
                String("SCAIL-2 parity mismatch: ") + label
                + String(" index=") + String(i)
            )


def _gelu_gate(ctx: DeviceContext) raises:
    # torch.nn.GELU() (approximate='none') oracle values, F32 tolerance.
    var inputs = List[Float32]()
    inputs.append(-3.0)
    inputs.append(-1.5)
    inputs.append(-0.5)
    inputs.append(0.0)
    inputs.append(0.5)
    inputs.append(1.5)
    inputs.append(3.0)
    var expected = List[Float32]()
    expected.append(-0.0040496941)
    expected.append(-0.1002108019)
    expected.append(-0.1542687694)
    expected.append(0.0)
    expected.append(0.3457312306)
    expected.append(1.3997891981)
    expected.append(2.9959503059)
    var exact = scail2_image_projection_activation(
        Tensor.from_host(inputs.copy(), [7], STDtype.F32, ctx), ctx
    )
    _require_close(exact.to_host(ctx), expected, 2.0e-6, "image exact GELU")

    # Prove this gate discriminates the pre-fix tanh approximation.
    var approximate = gelu(
        Tensor.from_host(inputs^, [7], STDtype.F32, ctx), ctx
    ).to_host(ctx)
    var distinguished = False
    for i in range(len(approximate)):
        if _abs(approximate[i] - expected[i]) > 1.0e-4:
            distinguished = True
    if not distinguished:
        raise Error("SCAIL-2 exact GELU gate did not reject tanh approximation")


def _residual_gate(ctx: DeviceContext) raises:
    var x = Tensor.from_host([1.0, -2.0], [2], STDtype.BF16, ctx)
    var y = Tensor.from_host([0.5, 4.0], [2], STDtype.BF16, ctx)
    var gate = Tensor.from_host([2.0, -0.25], [2], STDtype.F32, ctx)
    var gated = scail2_gated_residual_f32(x, y, gate, ctx)
    if gated.dtype() != STDtype.F32:
        raise Error("SCAIL-2 gated residual did not retain F32")
    _require_close(gated.to_host(ctx), [2.0, -3.0], 0.0, "gated residual")

    var plain = scail2_residual_f32(
        Tensor.from_host([1.0, -2.0], [2], STDtype.BF16, ctx),
        Tensor.from_host([0.5, 4.0], [2], STDtype.BF16, ctx),
        ctx,
    )
    if plain.dtype() != STDtype.F32:
        raise Error("SCAIL-2 cross residual did not retain F32")
    _require_close(plain.to_host(ctx), [1.5, 2.0], 0.0, "plain residual")


def main() raises:
    var ctx = DeviceContext()
    _gelu_gate(ctx)
    _residual_gate(ctx)
    print("GATE PASS SCAIL-2 exact image GELU + F32 residual contracts")
