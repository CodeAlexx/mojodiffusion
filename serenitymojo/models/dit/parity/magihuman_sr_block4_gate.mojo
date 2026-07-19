# Numeric SR block-4 (SHARED layer) parity gate for magihuman_sr_dit.
# Loads magihuman_sr_block4_fixture.safetensors (built by
# gen_magihuman_sr_block4_oracle.py from the REAL SR1080 bf16 checkpoint
# block.layers.4 SPLIT linears + a faithful torch recompute of the SR shared
# layer forward — 4 separate Q/K/V/G matmuls). Feeds the SR split weights through
# sr_fuse_shared_weights (concat q/k/v/g -> fused linear_qkv) then runs the base
# magihuman_shared_block_forward and compares to `expected`. Gate cos >= 0.999 at
# L=128, Dh=128 (dodges the math-mode SDPA Dh=128 OOM at full sequence).

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.parity import ParityHarness
from serenitymojo.models.dit.magihuman_dit import MagiHumanConfig
from serenitymojo.models.dit.magihuman_sr_dit import (
    sr_fuse_shared_weights, magihuman_sr_shared_block_forward,
)

comptime FIX = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/magihuman_sr_block4_fixture.safetensors"
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

    var x_in = _load(st, "input", ctx)                                # [L,5120] F32
    var cos_b = cast_tensor(_load(st, "cos", ctx), STDtype.BF16, ctx)  # [L,48]
    var sin_b = cast_tensor(_load(st, "sin", ctx), STDtype.BF16, ctx)
    var expected = _load(st, "expected", ctx)                         # [L,5120] F32

    # Raw SR split weights -> BF16 (keys WITHOUT block prefix).
    var raw = Dict[String, ArcPointer[Tensor]]()
    var split = [
        "attention.pre_norm.weight", "attention.q_norm.weight",
        "attention.k_norm.weight",
        "attention.linear_q.weight", "attention.linear_k.weight",
        "attention.linear_v.weight", "attention.linear_g.weight",
        "attention.linear_proj.weight",
        "mlp.pre_norm.weight", "mlp.up_gate_proj.weight", "mlp.down_proj.weight",
    ]
    for sfx in split:
        var s = String(sfx)
        var t = cast_tensor(_load(st, String("w_") + s, ctx), STDtype.BF16, ctx)
        raw[s] = ArcPointer(t^)

    # Load-time adapter: fuse q/k/v/g -> linear_qkv, pre-add 1 to norms.
    var bw = sr_fuse_shared_weights(raw, ctx)

    var out = magihuman_sr_shared_block_forward[L, H, HKV, DH](
        x_in, cos_b, sin_b, bw, cfg, ctx
    )

    var expected_h = expected.to_host(ctx)
    var harness = ParityHarness(0.999)
    var res = harness.compare(out, expected_h, ctx)
    print("MagiHuman SR shared block-4 gate:")
    print("  cos     =", res.cos)
    print("  max_abs =", res.max_abs)
    print("  n       =", res.n)
    if res.passed:
        print("  GATE: PASS (cos >= 0.999)")
    else:
        print("  GATE: FAIL (cos < 0.999)")
