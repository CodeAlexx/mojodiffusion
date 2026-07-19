# swiglu_packed_smoke.mojo — gate swiglu_packed([gate|up]) AGAINST the trusted
# swiglu(gate, up) on identical values. Reference = GPU scalar swiglu (MJ-1006).
#
# Run: cd /home/alex/md-perf && pixi run mojo run -I serenitymojo \
#      serenitymojo/ops/parity/swiglu_packed_smoke.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.activations import swiglu, swiglu_packed


def _fill(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def main() raises:
    var ctx = DeviceContext()
    var rows = 4096
    var f = 4096
    print("=== swiglu_packed parity vs swiglu(gate,up) (rows=", rows, " f=", f, ") ===")

    var gate_vals = _fill(rows * f, 11, 4.0)
    var up_vals = _fill(rows * f, 22, 4.0)

    # Build the packed [rows, 2f] buffer: per row, [gate(f) | up(f)].
    var gu_vals = List[Float32]()
    for r in range(rows):
        for c in range(f):
            gu_vals.append(gate_vals[r * f + c])
        for c in range(f):
            gu_vals.append(up_vals[r * f + c])

    var gate_t = Tensor.from_host(gate_vals, [rows, f], STDtype.F32, ctx)
    var up_t = Tensor.from_host(up_vals, [rows, f], STDtype.F32, ctx)
    var gate_up = Tensor.from_host(gu_vals, [rows, 2 * f], STDtype.F32, ctx)

    var refh = swiglu(gate_t, up_t, ctx).to_host(ctx)
    var got = swiglu_packed(gate_up, ctx)

    var h = ParityHarness(0.999)
    var r = h.compare(got, refh, ctx)
    print("    swiglu_packed:", r)

    if r.passed:
        print("PASS: swiglu_packed matches swiglu(gate,up) cos>=0.999")
    else:
        raise Error("swiglu_packed_smoke gate FAILED")
