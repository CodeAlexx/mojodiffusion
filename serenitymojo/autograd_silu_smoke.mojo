# autograd_silu_smoke.mojo — tape-level gate for OP_SILU (1-input op).
# Chain: y = silu(x) ; loss seed d_y = ones[N]. Tape backward must produce
#   d_x = silu'(x) = s*(1 + x*(1-s)),  s = sigmoid(x)          (since d_y = ones)
# computed INDEPENDENTLY on the host (closed form) — proves the tape RECORDS +
# DISPATCHES silu through silu_backward (saved0=x; rhs unused), not just that the
# kernel works in isolation.
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . serenitymojo/autograd_silu_smoke.mojo

from std.math import exp
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.autograd import Tape, backward


def main() raises:
    var ctx = DeviceContext()
    var h = ParityHarness()
    comptime N = 12

    var x_vals = List[Float32]()
    for i in range(N):
        x_vals.append((Float32((i * 7) % 13) - 6.0) * 0.3)

    var xsh = List[Int](); xsh.append(N)
    var x = Tensor.from_host(x_vals, xsh.copy(), STDtype.F32, ctx)

    var tape = Tape()
    tape.track(x)
    var y = tape.record_silu(x, ctx)        # [N]

    var grads = backward(tape, y, ctx)       # seeds d_y = ones[N]

    # Closed form (d_y = ones): d_x = s*(1 + x*(1-s)), s = sigmoid(x).
    var exp_dx = List[Float32]()
    for i in range(N):
        var xv = x_vals[i]
        var s = Float32(1.0) / (Float32(1.0) + exp(-xv))
        exp_dx.append(s * (Float32(1.0) + xv * (Float32(1.0) - s)))

    var dx = grads[x.id][].to_host(ctx)
    var r_dx = h.compare_host(dx, exp_dx)
    print("tape silu d_x:", r_dx)
    if r_dx.passed:
        print("TAPE OP_SILU GATE PASSED (1-input op, cos >= 0.999)")
    else:
        print("TAPE OP_SILU GATE FAILED")
        raise Error("tape silu gate failed")
