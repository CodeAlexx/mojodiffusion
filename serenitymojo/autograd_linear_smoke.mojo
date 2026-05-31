# autograd_linear_smoke.mojo — tape-level gate for OP_LINEAR (3-input op).
# Chain: y = linear(x, W, b) ; loss seed d_y = ones[M,out]. Tape backward must
# produce d_x, d_W, d_b matching the closed form (d_y=ones):
#   d_x[m,i]  = sum_o W[o,i]              (grad_y @ W)
#   d_W[o,i]  = sum_m x[m,i]              (grad_yᵀ @ x)
#   d_b[o]    = M                          (colsum of ones[M,out])
# Proves the tape RECORDS + DISPATCHES a 3-trainable-input op through
# linear_backward (the third_id/saved2 slot), not just 2-input ops.
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . serenitymojo/autograd_linear_smoke.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.autograd import Tape, backward


def main() raises:
    var ctx = DeviceContext()
    var h = ParityHarness()
    comptime M = 4
    comptime IN = 3
    comptime OUT = 2

    var x_vals = List[Float32]()
    for i in range(M * IN):
        x_vals.append((Float32((i * 7) % 13) - 6.0) * 0.05)
    var w_vals = List[Float32]()
    for i in range(OUT * IN):
        w_vals.append((Float32((i * 5) % 11) - 5.0) * 0.05)
    var b_vals = List[Float32]()
    for i in range(OUT):
        b_vals.append((Float32((i * 3) % 9) - 4.0) * 0.05)

    var xsh = List[Int](); xsh.append(M); xsh.append(IN)
    var wsh = List[Int](); wsh.append(OUT); wsh.append(IN)
    var bsh = List[Int](); bsh.append(OUT)
    var x = Tensor.from_host(x_vals, xsh.copy(), STDtype.F32, ctx)
    var w = Tensor.from_host(w_vals, wsh.copy(), STDtype.F32, ctx)
    var b = Tensor.from_host(b_vals, bsh.copy(), STDtype.F32, ctx)

    var tape = Tape()
    tape.track(x)
    tape.track(w)
    tape.track(b)
    var y = tape.record_linear(x, w, b, ctx)   # [M,OUT]

    var grads = backward(tape, y, ctx)           # seeds d_y = ones[M,OUT]

    # Expected (d_y = ones[M,OUT]):
    var exp_dx = List[Float32]()
    for _m in range(M):
        for i in range(IN):
            var s: Float32 = 0.0
            for o in range(OUT):
                s += w_vals[o * IN + i]
            exp_dx.append(s)
    var exp_dw = List[Float32]()
    for o in range(OUT):
        for i in range(IN):
            var s: Float32 = 0.0
            for m in range(M):
                s += x_vals[m * IN + i]
            _ = o
            exp_dw.append(s)
    var exp_db = List[Float32]()
    for _o in range(OUT):
        exp_db.append(Float32(M))

    var dx = grads[x.id][].to_host(ctx)
    var dw = grads[w.id][].to_host(ctx)
    var db = grads[b.id][].to_host(ctx)
    var r_dx = h.compare_host(dx, exp_dx)
    var r_dw = h.compare_host(dw, exp_dw)
    var r_db = h.compare_host(db, exp_db)
    print("tape linear d_x:", r_dx)
    print("tape linear d_W:", r_dw)
    print("tape linear d_b:", r_db)
    if r_dx.passed and r_dw.passed and r_db.passed:
        print("TAPE OP_LINEAR GATE PASSED (3-input op, cos >= 0.999)")
    else:
        print("TAPE OP_LINEAR GATE FAILED")
        raise Error("tape linear gate failed")
