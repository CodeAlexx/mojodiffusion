# Full-forward parity gate for acestep_dit (ACE-Step-1.5 turbo DiT).
#
# Loads acestep_full_fixture.safetensors (built by the oracle script from the
# REAL acestep-v15-turbo checkpoint: canonical AceStepDiTModel.forward run
# eager-mode bf16 GPU at T=200 -> cat(context,x_t)=[1,200,192], patched seq
# SP=100 <= sliding_window(128) so all self-attn masks are all-zeros). Runs the
# full Mojo acestep_forward (timestep MLP x2, proj_in conv1d, 24 DiT layers,
# final AdaLN, proj_out convT1d, crop). Gate cos >= 0.99 (deep 24-layer chain).

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.parity import ParityHarness
from serenitymojo.models.dit.acestep_dit import (
    AceStepDiTConfig, acestep_forward,
)

comptime FIX = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/acestep_full_fixture.safetensors"
comptime SP = 100   # patched seq = T/patch = 200/2
comptime L = 48


def _load_bf16(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return cast_tensor(Tensor.from_view(st.tensor_view(name), ctx), STDtype.BF16, ctx)


def _scalar0(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Float32:
    var h = Tensor.from_view(st.tensor_view(name), ctx).to_host(ctx)
    return h[0]


def main() raises:
    var ctx = DeviceContext()
    var cfg = AceStepDiTConfig.turbo()
    var st = ShardedSafeTensors.open(FIX)

    var x_t = _load_bf16(st, "hidden", ctx)      # [1,T,64]
    var ctxt = _load_bf16(st, "context", ctx)    # [1,T,128]
    var enc = _load_bf16(st, "enc", ctx)         # [1,L,2048]
    var timestep = _scalar0(st, "timestep", ctx)
    var timestep_r = _scalar0(st, "timestep_r", ctx)

    # All decoder.* weights into a single dict.
    var full = Dict[String, ArcPointer[Tensor]]()
    var names = st.names()
    for nm in names:
        var n = String(nm)
        if n.startswith("w_"):
            # fixture keys are "w_<rel>" where <rel> dropped the "decoder."
            # prefix; the Mojo forward looks up "decoder.<rel>", so add it back.
            var key = String("decoder.") + String(n[byte=2:])
            full[key] = ArcPointer(_load_bf16(st, n, ctx))

    var out = acestep_forward[SP, L](
        x_t, ctxt, enc, timestep, timestep_r, full, cfg, ctx
    )

    var ph = ParityHarness(0.99)
    var expected = Tensor.from_view(st.tensor_view("expected"), ctx)  # F32
    var res = ph.compare(out, expected.to_host(ctx), ctx)
    print("acestep full-forward parity:", res)
    if res.passed:
        print("GATE: PASS")
    else:
        print("GATE: FAIL")
