# autograd_rmsnorm_smoke.mojo — tape-level gate for OP_RMSNORM (2-input op).
# Chain: y = rms_norm(x, gamma) over last dim ; loss seed d_y = ones[rows,D].
# Tape backward must produce d_x, d_gamma matching rms_norm_backward run
# DIRECTLY (the verified kernel) on the SAME inputs with d_y = ones. This proves
# the tape RECORDS + DISPATCHES rms_norm through rms_norm_backward (the saved0/
# saved1 slots + shared _RMS_EPS), not just that the kernel works in isolation.
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . serenitymojo/autograd_rmsnorm_smoke.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.autograd import Tape, backward, ones_like
from serenitymojo.ops.norm_backward import rms_norm_backward

# Must match autograd._RMS_EPS (the eps record_rms_norm + the OP_RMSNORM arm
# use). Kept in sync by hand — a single 1e-6 literal in both places.
comptime _RMS_EPS = Float32(1e-6)


def main() raises:
    var ctx = DeviceContext()
    var h = ParityHarness()
    comptime ROWS = 4
    comptime D = 6

    var x_vals = List[Float32]()
    for i in range(ROWS * D):
        x_vals.append((Float32((i * 7) % 13) - 6.0) * 0.1)
    var g_vals = List[Float32]()
    for i in range(D):
        g_vals.append(1.0 + (Float32((i * 5) % 11) - 5.0) * 0.05)

    var xsh = List[Int](); xsh.append(ROWS); xsh.append(D)
    var gsh = List[Int](); gsh.append(D)
    var x = Tensor.from_host(x_vals, xsh.copy(), STDtype.F32, ctx)
    var gamma = Tensor.from_host(g_vals, gsh.copy(), STDtype.F32, ctx)

    var tape = Tape()
    tape.track(x)
    tape.track(gamma)
    var y = tape.record_rms_norm(x, gamma, ctx)   # [ROWS,D]

    var grads = backward(tape, y, ctx)             # seeds d_y = ones[ROWS,D]

    # Reference: run the verified kernel directly with d_y = ones (same as the
    # tape's seed). The tape must reproduce this exactly.
    var dy = ones_like(y, ctx)
    var ref_grads = rms_norm_backward(dy, x, gamma, _RMS_EPS, ctx)
    var exp_dx = ref_grads.d_x.to_host(ctx)
    var exp_dg = ref_grads.d_g.to_host(ctx)

    var dx = grads[x.id][].to_host(ctx)
    var dg = grads[gamma.id][].to_host(ctx)
    var r_dx = h.compare_host(dx, exp_dx)
    var r_dg = h.compare_host(dg, exp_dg)
    print("tape rmsnorm d_x:", r_dx)
    print("tape rmsnorm d_gamma:", r_dg)
    if r_dx.passed and r_dg.passed:
        print("TAPE OP_RMSNORM GATE PASSED (2-input op, cos >= 0.999)")
    else:
        print("TAPE OP_RMSNORM GATE FAILED")
        raise Error("tape rmsnorm gate failed")
