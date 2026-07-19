# autograd_matmul_smoke.mojo — tape-level gate for OP_MATMUL.
# Chain: c = matmul(a,b) ; loss = sum(c) (seed d=ones). Then d_a/d_b via the
# TAPE's backward(), compared to the closed form: with d_c = ones[M,N],
#   d_a = ones @ bᵀ  (row r = sum over n of b[:,n])  -> [M,K]
#   d_b = aᵀ @ ones                                   -> [K,N]
# This proves the tape RECORDS + DISPATCHES matmul through mm_backward, not just
# that mm_backward works in isolation (already gated separately).
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . serenitymojo/autograd_matmul_smoke.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.autograd import Tape, backward


def main() raises:
    var ctx = DeviceContext()
    var h = ParityHarness()
    comptime M = 4
    comptime K = 3
    comptime N = 2

    var a_vals = List[Float32]()
    for i in range(M * K):
        a_vals.append((Float32((i * 7) % 13) - 6.0) * 0.05)
    var b_vals = List[Float32]()
    for i in range(K * N):
        b_vals.append((Float32((i * 5) % 11) - 5.0) * 0.05)

    var ash = List[Int](); ash.append(M); ash.append(K)
    var bsh = List[Int](); bsh.append(K); bsh.append(N)
    var a = Tensor.from_host(a_vals, ash.copy(), STDtype.F32, ctx)
    var b = Tensor.from_host(b_vals, bsh.copy(), STDtype.F32, ctx)

    var tape = Tape()
    tape.track(a)
    tape.track(b)
    var c = tape.record_matmul(a, b, ctx)   # [M,N]

    var grads = backward(tape, c, ctx)       # seeds d_c = ones[M,N]

    # Expected closed form with d_c = ones[M,N]:
    #   d_a[m,k] = sum_n b[k,n]
    #   d_b[k,n] = sum_m a[m,k]
    var exp_da = List[Float32]()
    for _m in range(M):
        for k in range(K):
            var s: Float32 = 0.0
            for n in range(N):
                s += b_vals[k * N + n]
            exp_da.append(s)
    var exp_db = List[Float32]()
    for k in range(K):
        for n in range(N):
            var s: Float32 = 0.0
            for m in range(M):
                s += a_vals[m * K + k]
            _ = n
            exp_db.append(s)

    var da = grads[a.id][].to_host(ctx)
    var db = grads[b.id][].to_host(ctx)
    var r_da = h.compare_host(da, exp_da)
    var r_db = h.compare_host(db, exp_db)
    print("tape matmul d_a:", r_da)
    print("tape matmul d_b:", r_db)
    if r_da.passed and r_db.passed:
        print("TAPE OP_MATMUL GATE PASSED (cos >= 0.999)")
    else:
        print("TAPE OP_MATMUL GATE FAILED")
        raise Error("tape matmul gate failed")
