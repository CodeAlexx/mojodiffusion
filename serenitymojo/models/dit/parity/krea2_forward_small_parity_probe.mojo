# models/dit/parity/krea2_forward_small_parity_probe.mojo — PARITY GATE for krea2
# chunk 7a: SingleStreamDiT.forward WIRING with REDUCED-block (4) seeded-random
# weights, RESIDENT. vs the reference's own (cuDNN, natural-path) velocity.
# FAIL-LOUD: non-zero exit on cos < 0.999.
#
# Oracle: krea2_forward_small_oracle.safetensors — ALL weights (w.* keys) + inputs
#   img [1,40,64] bf16, context [1,20,12,2560] bf16, t [1] f32, pos [1,60,3] f32,
#   mask [1,60] f32 (all-ones), velocity [1,40,64] f32 (the reference output).
#   TXTLEN=20, IMGLEN=40, L_FULL=60 -> pad-to-256 (exercises the masked main blocks).
#
# Run: cd /home/alex/mojodiffusion && \
#   pixi run mojo run -I . serenitymojo/models/dit/parity/krea2_forward_small_parity_probe.mojo
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.models.dit.krea2_dit import krea2_forward


comptime ORACLE = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/krea2_forward_small_oracle.safetensors"


def main() raises:
    var ctx = DeviceContext()
    var st = ShardedSafeTensors.open(ORACLE)

    # comptime seq dims (must match the gen): L_FULL=60, pad-to-256, txtlen=20, 4 blocks.
    comptime LFULL = 60
    comptime LPAD = 256
    comptime LT = 20
    comptime NBLOCKS = 4

    var img = Tensor.from_view(st.tensor_view("img"), ctx)          # bf16 [1,40,64]
    var context = Tensor.from_view(st.tensor_view("context"), ctx)  # bf16 [1,20,12,2560]
    var t = Tensor.from_view_as_f32(st.tensor_view("t"), ctx)       # f32 [1]
    var pos = Tensor.from_view_as_f32(st.tensor_view("pos"), ctx)   # f32 [1,60,3]

    var velocity = krea2_forward[LFULL, LPAD, LT, NBLOCKS](
        st, img, context, t, pos, ctx
    )                                                               # [1, 40, 64]

    var vel_ref = Tensor.from_view_as_f32(st.tensor_view("velocity"), ctx).to_host(ctx)
    var res = ParityHarness(0.999).compare(velocity, vel_ref, ctx)
    print("krea2_forward (4-block wiring) parity:", res)
    if res.cos < 0.999:
        raise Error(
            "krea2_forward FAILED: cos " + String(res.cos) + " < 0.999"
        )
    print("krea2_forward WIRING PASS")
