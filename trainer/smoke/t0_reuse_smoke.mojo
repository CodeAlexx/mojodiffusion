# t0_reuse_smoke.mojo — T0 gate for the Serenity→Mojo port.
#
# PROVES three things the whole port depends on, with ZERO new model code:
#   1. Cross-repo reuse wiring works: this file (in serenity-trainer) imports
#      serenitymojo's autograd/tensor/ops from /home/alex/mojodiffusion.
#   2. The reused autograd tape RUNS end-to-end: matmul -> MSE -> backward.
#   3. The dtype policy holds at the boundary: BF16 in -> BF16 grads out
#      (storage stays BF16; F32 only inside compute).
#
# Run (only when the GPU is free — JIT executes a kernel):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . -I trainer/src \
#       trainer/smoke/t0_reuse_smoke.mojo
#   expect: "T0 OK" with grad dtype=BF16, finite=true for both inputs.

from std.builtin.dtype import DType
from max.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.autograd import Tape, backward


def _all_finite(v: List[Float32]) -> Bool:
    for i in range(len(v)):
        var x = v[i]
        # NaN != itself; Inf has abs > 3.0e38
        if x != x:
            return False
        var a = x if x >= 0.0 else -x
        if a > 3.0e38:
            return False
    return True


def _require(cond: Bool, msg: String) raises:
    if not cond:
        raise Error("T0 FAIL: " + msg)


def main() raises:
    var ctx = DeviceContext()

    comptime M = 4
    comptime K = 3
    comptime N = 2

    # BF16 inputs (storage dtype = BF16, per the port dtype policy).
    var a_vals = List[Float32]()
    for i in range(M * K):
        a_vals.append((Float32((i * 7) % 13) - 6.0) * 0.05)
    var b_vals = List[Float32]()
    for i in range(K * N):
        b_vals.append((Float32((i * 5) % 11) - 5.0) * 0.05)
    var t_vals = List[Float32]()
    for i in range(M * N):
        t_vals.append((Float32((i * 3) % 7) - 3.0) * 0.05)

    var ash = List[Int](); ash.append(M); ash.append(K)
    var bsh = List[Int](); bsh.append(K); bsh.append(N)
    var tsh = List[Int](); tsh.append(M); tsh.append(N)

    var a = Tensor.from_host(a_vals, ash.copy(), STDtype.BF16, ctx)
    var b = Tensor.from_host(b_vals, bsh.copy(), STDtype.BF16, ctx)
    var target = Tensor.from_host(t_vals, tsh.copy(), STDtype.BF16, ctx)

    # Build the tape: pred = a @ b ; loss = mse(pred, target).
    var tape = Tape()
    tape.track(a)
    tape.track(b)
    var pred = tape.record_matmul(a, b, ctx)
    _require(pred.dtype() == STDtype.BF16, "forward matmul did not preserve BF16")

    var loss = tape.mse_loss(pred, target, ctx)

    # Reverse pass through the reused tape.
    var grads = backward(tape, loss, ctx)

    _require(grads.__contains__(a.id), "no grad for a")
    _require(grads.__contains__(b.id), "no grad for b")

    var ga = grads[a.id]
    var gb = grads[b.id]

    _require(ga[].dtype() == STDtype.BF16, "grad(a) not BF16: " + ga[].dtype().name())
    _require(gb[].dtype() == STDtype.BF16, "grad(b) not BF16: " + gb[].dtype().name())

    var ga_host = ga[].to_host(ctx)
    var gb_host = gb[].to_host(ctx)
    _require(_all_finite(ga_host), "grad(a) has NaN/Inf")
    _require(_all_finite(gb_host), "grad(b) has NaN/Inf")

    print("T0 OK — reuse wiring + bf16 tape backward verified")
    print("  pred dtype =", pred.dtype().name(), " shape =", M, "x", N)
    print("  grad(a) dtype =", ga[].dtype().name(), " n =", len(ga_host), " finite = True")
    print("  grad(b) dtype =", gb[].dtype().name(), " n =", len(gb_host), " finite = True")
