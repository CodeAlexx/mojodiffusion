# Parity vs Serenity adamw_ref.json (deterministic, SR off). Same fixed inputs.
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenity_trainer.util.optimizer.adamw_extensions import adamw_step

def main() raises:
    var ctx = DeviceContext()
    comptime N = 8
    var pv = List[Float32](); var gv = List[Float32](); var zv = List[Float32]()
    for i in range(N):
        pv.append(Float32(i)*0.1 - 0.4); gv.append(0.05); zv.append(0.0)
    var sh = List[Int](); sh.append(N)
    var p = Tensor.from_host(pv, sh.copy(), STDtype.BF16, ctx)
    var m = Tensor.from_host(zv.copy(), sh.copy(), STDtype.BF16, ctx)
    var v = Tensor.from_host(zv.copy(), sh.copy(), STDtype.BF16, ctx)
    var g = Tensor.from_host(gv, sh.copy(), STDtype.BF16, ctx)
    var p0 = p.to_host(ctx)
    # SR OFF → deterministic, comparable to Serenity stochastic_rounding=False
    adamw_step(p, m, v, g, 1, 1e-3, 0.9, 0.999, 1e-8, 0.01, False, UInt32(0), ctx)
    adamw_step(p, m, v, g, 2, 1e-3, 0.9, 0.999, 1e-8, 0.01, False, UInt32(0), ctx)
    var p2 = p.to_host(ctx)
    var tot = Float32(0.0)
    print("p_after2(MOJO)=")
    for i in range(N):
        print("  ", p2[i])
        var d = p2[i]-p0[i]; tot += d if d>=0.0 else -d
    print("total|dp|(MOJO)=", tot)
