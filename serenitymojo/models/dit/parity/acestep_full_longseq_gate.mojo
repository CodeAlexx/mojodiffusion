# Long-seq full-forward parity gate for acestep_dit (ACE-Step-1.5 turbo DiT).
#
# Loads acestep_full_longseq_fixture.safetensors (built by the oracle from the
# REAL acestep-v15-turbo checkpoint: canonical AceStepDiTModel.forward run
# eager-mode bf16 GPU at T=600 -> patched seq SP=300 > sliding_window(128)).
# At SP=300 the 12 sliding-attention layers (layer_types[i]="sliding" when
# (i+1)%2) use a NON-trivial |i-j|<=128 bidirectional mask, so this gate
# exercises the sliding-mask fix (sdpa_tiled w/ full mask). Gate cos >= 0.99.

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

comptime FIX = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/acestep_full_longseq_fixture.safetensors"
comptime SP = 300   # patched seq = T/patch = 600/2 ; > sliding_window(128)
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

    var full = Dict[String, ArcPointer[Tensor]]()
    var names = st.names()
    for nm in names:
        var n = String(nm)
        if n.startswith("w_"):
            var key = String("decoder.") + String(n[byte=2:])
            full[key] = ArcPointer(_load_bf16(st, n, ctx))

    var out = acestep_forward[SP, L](
        x_t, ctxt, enc, timestep, timestep_r, full, cfg, ctx
    )

    var ph = ParityHarness(0.99)
    var expected = Tensor.from_view(st.tensor_view("expected"), ctx)  # F32
    var res = ph.compare(out, expected.to_host(ctx), ctx)
    print("acestep long-seq (SP=300) full-forward parity:", res)
    if res.passed:
        print("GATE: PASS")
    else:
        print("GATE: FAIL")
