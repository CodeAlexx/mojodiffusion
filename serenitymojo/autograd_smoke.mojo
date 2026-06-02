# autograd_smoke.mojo — Phase T1 gate: prove the tape engine backprops a small
# op chain correctly vs PyTorch (FULL_PORT_TRAINING_PLAN §4 Phase T1).
#
# Graph:  s = a + b ;  y = s * c   (elementwise, F32, n=64)
# Analytic grads (d(sum y)=1):
#   dy/dc = s = a+b
#   dy/ds = c  → dy/da = c, dy/db = c
# Reference is the exact closed-form grad computed in-Mojo (no torch oracle for
# THIS trivial case — d/da=c, d/db=c, d/dc=a+b are exact; cos=1.0 confirms the
# tape wiring is correct). NOT a torch parity test; the per-op torch gates live
# in serenitymojo/ops/parity/*_bwd_parity.mojo.
#
# Run: cd /home/alex/mojodiffusion
#      pixi run mojo run -I . serenitymojo/autograd_smoke.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.autograd import Tape, backward


def _fill_a(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 7) % 13) - 6.0) * 0.05)
    return out^


def _fill_b(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 5) % 11) - 5.0) * 0.05)
    return out^


def _fill_c(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 3) % 9) - 4.0) * 0.05)
    return out^


def main() raises:
    var ctx = DeviceContext()
    var h = ParityHarness()
    comptime N = 64
    var shape = List[Int]()
    shape.append(N)

    var a = Tensor.from_host(_fill_a(N), shape.copy(), STDtype.F32, ctx)
    var b = Tensor.from_host(_fill_b(N), shape.copy(), STDtype.F32, ctx)
    var c = Tensor.from_host(_fill_c(N), shape.copy(), STDtype.F32, ctx)

    var tape = Tape()
    tape.track(a)
    tape.track(b)
    tape.track(c)

    # Forward: s = a+b ; y = s*c
    var s = tape.record_add(a, b, ctx)
    var y = tape.record_mul(s, c, ctx)

    var grads = backward(tape, y, ctx)

    # Expected (closed form): da = c, db = c, dc = a+b.
    var ah = _fill_a(N)
    var bh = _fill_b(N)
    var ch = _fill_c(N)
    var exp_da = List[Float32]()
    var exp_db = List[Float32]()
    var exp_dc = List[Float32]()
    for i in range(N):
        exp_da.append(ch[i])           # dy/da = c
        exp_db.append(ch[i])           # dy/db = c
        exp_dc.append(ah[i] + bh[i])   # dy/dc = a+b
    _ = ah
    _ = bh
    _ = ch

    var da = grads[a.id][].to_host(ctx)   # ArcPointer deref -> Tensor
    var db = grads[b.id][].to_host(ctx)
    var dc = grads[c.id][].to_host(ctx)

    var r_da = h.compare_host(da, exp_da)
    var r_db = h.compare_host(db, exp_db)
    var r_dc = h.compare_host(dc, exp_dc)
    print("d/da (expect c)   :", r_da)
    print("d/db (expect c)   :", r_db)
    print("d/dc (expect a+b) :", r_dc)

    var all_pass = r_da.passed and r_db.passed and r_dc.passed
    print("")
    if all_pass:
        print("T1 TAPE ENGINE GATE PASSED (add+mul backprop correct, cos >= 0.999)")
    else:
        print("T1 TAPE ENGINE FAILURE")
        raise Error("autograd_smoke gate failed")
