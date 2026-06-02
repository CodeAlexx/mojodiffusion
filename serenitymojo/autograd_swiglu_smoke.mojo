# autograd_swiglu_smoke.mojo — tape-level gate for OP_SWIGLU (2-input op).
# Chain: y = silu(gate) * up ; loss seed d_y = ones[N]. Tape backward must
# produce (since d_y = ones):
#   d_up   = silu(gate)                 , silu(g) = g*sigmoid(g)
#   d_gate = up * silu'(gate)           , silu'(g) = s*(1 + g*(1-s)), s=sigmoid(g)
# computed INDEPENDENTLY on the host — proves the tape RECORDS + DISPATCHES
# swiglu through swiglu_backward (saved0=gate, saved1=up → d_gate, d_up).
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . serenitymojo/autograd_swiglu_smoke.mojo

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

    var gate_vals = List[Float32]()
    for i in range(N):
        gate_vals.append((Float32((i * 7) % 13) - 6.0) * 0.3)
    var up_vals = List[Float32]()
    for i in range(N):
        up_vals.append((Float32((i * 5) % 11) - 5.0) * 0.2)

    var sh = List[Int](); sh.append(N)
    var gate = Tensor.from_host(gate_vals, sh.copy(), STDtype.F32, ctx)
    var up = Tensor.from_host(up_vals, sh.copy(), STDtype.F32, ctx)

    var tape = Tape()
    tape.track(gate)
    tape.track(up)
    var y = tape.record_swiglu(gate, up, ctx)   # [N]

    var grads = backward(tape, y, ctx)           # seeds d_y = ones[N]

    # Closed form (d_y = ones):
    var exp_dgate = List[Float32]()
    var exp_dup = List[Float32]()
    for i in range(N):
        var gv = gate_vals[i]
        var uv = up_vals[i]
        var s = Float32(1.0) / (Float32(1.0) + exp(-gv))
        var silu_g = gv * s
        var dsilu = s * (Float32(1.0) + gv * (Float32(1.0) - s))
        exp_dup.append(silu_g)
        exp_dgate.append(uv * dsilu)

    var dgate = grads[gate.id][].to_host(ctx)
    var dup = grads[up.id][].to_host(ctx)
    var r_dgate = h.compare_host(dgate, exp_dgate)
    var r_dup = h.compare_host(dup, exp_dup)
    print("tape swiglu d_gate:", r_dgate)
    print("tape swiglu d_up:", r_dup)
    if r_dgate.passed and r_dup.passed:
        print("TAPE OP_SWIGLU GATE PASSED (2-input op, cos >= 0.999)")
    else:
        print("TAPE OP_SWIGLU GATE FAILED")
        raise Error("tape swiglu gate failed")
