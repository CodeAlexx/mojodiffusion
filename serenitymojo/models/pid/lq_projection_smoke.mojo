# models/pid/lq_projection_smoke.mojo — unit-gate PiD LQProjection2D (latent
# branch) against the PyTorch reference.
#
# Loads parity/lq_projection_ref.safetensors (weights + input + ref output dumped
# by gen_lq_projection_reference.py with seeded random weights), runs the Mojo
# LQProjection2D.forward on GPU, and compares the [B,N,OUT_DIM] tokens to the
# reference via ParityHarness. Gate: cos >= 0.999.
#
# Config: B=1, LATENT_CH=16, HIDDEN=512, OUT_DIM=1536, PH=PW=4 (N=16),
# NUM_RES=4. F32 storage throughout.
#
# Run: pixi run mojo run -I . serenitymojo/models/pid/lq_projection_smoke.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.models.pid.lq_projection import LQProjection2D


comptime _REF = (
    "/home/alex/mojodiffusion/serenitymojo/models/pid/parity/"
    "lq_projection_ref.safetensors"
)
comptime _B = 1
comptime _LATENT_CH = 16
comptime _HIDDEN = 512
comptime _OUT_DIM = 1536
comptime _PH = 4
comptime _PW = 4
comptime _NUM_RES = 4


def main() raises:
    var ctx = DeviceContext()
    var st = ShardedSafeTensors.open(String(_REF))

    var model = LQProjection2D(st, ctx)

    # Load input (NCHW [B,16,PH,PW]) and reference output ([B,N,OUT_DIM]).
    var in_tv = st.tensor_view(String("lq_latent"))
    var lq_latent = Tensor.from_view(in_tv, ctx)
    var ref_tv = st.tensor_view(String("ref_output"))
    var ref_out = Tensor.from_view(ref_tv, ctx)
    var ref_host = ref_out.to_host(ctx)

    var out = model.forward[_B, _LATENT_CH, _HIDDEN, _PH, _PW](lq_latent, ctx)

    var osh = out.shape()
    print(
        "out shape: [",
        osh[0], ",", osh[1], ",", osh[2], "]  (expect [",
        _B, ",", _PH * _PW, ",", _OUT_DIM, "])",
    )

    var h = ParityHarness()
    var r = h.compare(out, ref_host, ctx)
    print("LQProjection2D (latent) ", r)

    if r.passed:
        print("LQ_PROJ GATE PASSED (cos >= 0.999)")
    else:
        print("LQ_PROJ GATE FAILED")
        raise Error("lq_projection unit-gate failed")
