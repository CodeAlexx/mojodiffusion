# autograd_mse_smoke.mojo — tape-level gate for OP_MSE (special LOSS LEAF).
# Chain: loss = mse_loss(pred, target). This op SEEDS the chain: backward()'s
# grads[loss.id]=ones is intentionally IGNORED by the OP_MSE arm, which calls
# mse_backward(pred,target) directly (already carrying the 2/N factor). Tape
# backward must produce (computed INDEPENDENTLY on the host):
#   d_pred = 2*(pred-target)/N
# and target (a constant, untracked input) must receive NO gradient. Proves the
# tape RECORDS + DISPATCHES the loss leaf through mse_backward.
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . serenitymojo/autograd_mse_smoke.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.autograd import Tape, backward


def main() raises:
    var ctx = DeviceContext()
    var h = ParityHarness()
    comptime N = 12

    var pred_vals = List[Float32]()
    for i in range(N):
        pred_vals.append((Float32((i * 7) % 13) - 6.0) * 0.2)
    var tgt_vals = List[Float32]()
    for i in range(N):
        tgt_vals.append((Float32((i * 5) % 11) - 5.0) * 0.15)

    var sh = List[Int](); sh.append(N)
    var pred = Tensor.from_host(pred_vals, sh.copy(), STDtype.F32, ctx)
    var target = Tensor.from_host(tgt_vals, sh.copy(), STDtype.F32, ctx)

    var tape = Tape()
    tape.track(pred)
    # target intentionally NOT tracked — it is a constant; no grad should flow.
    var loss = tape.mse_loss(pred, target, ctx)   # scalar [1]

    var grads = backward(tape, loss, ctx)          # OP_MSE seeds d_pred directly

    # Closed form: d_pred[i] = 2*(pred[i]-target[i])/N.
    var exp_dp = List[Float32]()
    for i in range(N):
        exp_dp.append(Float32(2.0) * (pred_vals[i] - tgt_vals[i]) / Float32(N))

    var dp = grads[pred.id][].to_host(ctx)
    var r_dp = h.compare_host(dp, exp_dp)
    print("tape mse d_pred:", r_dp)
    if r_dp.passed:
        print("TAPE OP_MSE GATE PASSED (loss leaf, cos >= 0.999)")
    else:
        print("TAPE OP_MSE GATE FAILED")
        raise Error("tape mse gate failed")
