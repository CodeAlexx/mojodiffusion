# Numeric block-4 (SHARED layer) parity gate for magihuman_dit.
# Loads magihuman_block4_fixture.safetensors (built by gen_magihuman_block4_oracle.py
# from the REAL distilled bf16 checkpoint block.layers.4 + a faithful torch recompute
# of SharedTransformerLayer::forward). Runs magihuman_shared_block_forward, compares
# to `expected`. Gate cos >= 0.999 at a SMALL grid (L=128, Dh=128) to dodge the
# math-mode SDPA Dh=128 OOM at full sequence length.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import add_scalar
from serenitymojo.parity import ParityHarness
from serenitymojo.models.dit.magihuman_dit import (
    MagiHumanConfig, magihuman_shared_block_forward,
)

comptime FIX = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/magihuman_block4_fixture.safetensors"
comptime L = 128
comptime H = 40
comptime HKV = 8
comptime DH = 128


def _load(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var tv = st.tensor_view(name)
    return Tensor.from_view(tv, ctx)


def main() raises:
    var ctx = DeviceContext()
    var cfg = MagiHumanConfig.magihuman_15b()
    var st = ShardedSafeTensors.open(FIX)

    var x_in = _load(st, "input", ctx)                       # [L,5120] F32
    var cos_b = cast_tensor(_load(st, "cos", ctx), STDtype.BF16, ctx)  # [L,48]
    var sin_b = cast_tensor(_load(st, "sin", ctx), STDtype.BF16, ctx)
    var expected = _load(st, "expected", ctx)               # [L,5120] F32

    # Weights -> BF16. Norm gains get a pre-added (weight + 1) variant under .p1.
    var bw = Dict[String, ArcPointer[Tensor]]()
    var big = [
        "attention.linear_qkv.weight", "attention.linear_proj.weight",
        "mlp.up_gate_proj.weight", "mlp.down_proj.weight",
    ]
    for sfx in big:
        var s = String(sfx)
        var t = cast_tensor(_load(st, String("w_") + s, ctx), STDtype.BF16, ctx)
        bw[s] = ArcPointer(t^)

    var norms = [
        "attention.pre_norm.weight", "attention.q_norm.weight",
        "attention.k_norm.weight", "mlp.pre_norm.weight",
    ]
    for sfx in norms:
        var s = String(sfx)
        var t = cast_tensor(_load(st, String("w_") + s, ctx), STDtype.BF16, ctx)
        var p1 = add_scalar(t, 1.0, ctx)   # (weight + 1) gain, bf16
        bw[s + ".p1"] = ArcPointer(p1^)

    var out = magihuman_shared_block_forward[L, H, HKV, DH](
        x_in, cos_b, sin_b, bw, cfg, ctx
    )

    var expected_h = expected.to_host(ctx)
    var harness = ParityHarness(0.999)
    var res = harness.compare(out, expected_h, ctx)
    print("MagiHuman shared block-4 gate:")
    print("  cos     =", res.cos)
    print("  max_abs =", res.max_abs)
    print("  n       =", res.n)
    if res.passed:
        print("  GATE: PASS (cos >= 0.999)")
    else:
        print("  GATE: FAIL (cos < 0.999)")
