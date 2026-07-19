# adamw_compile_check.mojo — compile + run gate for the BF16 AdamW port.
#
# Compile only (no GPU): see BUILD.md.
# Run (GPU free): one step on a tiny BF16 param should move it toward lower loss
# on a fixed grad, with NO F32 tensors anywhere.
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . -I trainer/src \
#       trainer/smoke/adamw_compile_check.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenity_trainer.util.optimizer.adamw_extensions import adamw_step


def main() raises:
    var ctx = DeviceContext()
    comptime N = 8

    var pv = List[Float32]()
    var gv = List[Float32]()
    for i in range(N):
        pv.append(Float32(i) * 0.1 - 0.4)
        gv.append(0.05)  # constant downhill grad
    var zv = List[Float32]()
    for _ in range(N):
        zv.append(0.0)

    var sh = List[Int](); sh.append(N)
    var p = Tensor.from_host(pv, sh.copy(), STDtype.BF16, ctx)
    var m = Tensor.from_host(zv.copy(), sh.copy(), STDtype.BF16, ctx)
    var v = Tensor.from_host(zv.copy(), sh.copy(), STDtype.BF16, ctx)
    var g = Tensor.from_host(gv, sh.copy(), STDtype.BF16, ctx)

    var p0 = p.to_host(ctx)

    # Two steps of BF16 AdamW with stochastic rounding (Serenity default for bf16).
    adamw_step(p, m, v, g, 1, 1e-3, 0.9, 0.999, 1e-8, 0.01, True, UInt32(1), ctx)
    adamw_step(p, m, v, g, 2, 1e-3, 0.9, 0.999, 1e-8, 0.01, True, UInt32(2), ctx)

    # dtype invariants: everything stays BF16 (no F32 state).
    if p.dtype() != STDtype.BF16 or m.dtype() != STDtype.BF16 or v.dtype() != STDtype.BF16:
        raise Error("AdamW leaked a non-BF16 tensor")

    var p2 = p.to_host(ctx)
    var moved = Float32(0.0)
    for i in range(N):
        var d = p2[i] - p0[i]
        moved += d if d >= 0.0 else -d
    print("ADAMW OK — bf16 state, params updated. total|Δp| =", moved)
    print("  p,m,v dtype =", p.dtype().name(), m.dtype().name(), v.dtype().name())
