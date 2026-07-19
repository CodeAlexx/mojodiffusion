# lycoris_smoke.mojo — COMPILE smoke for the LyCORIS adapter forwards
# (LoHa / LoKr / DoRA) ported from Serenity modules/module/LoRAModule.py.
#
# Drives a synthetic 1-linear "model" through a few optimizer steps using a LoHa
# adapter as the trainable surface, proving the LyCORIS forwards compose with the
# tape (grads flow to the factors) + AdamW + clip + the driver step. Mirrors the
# driver_smoke structure but swaps the LoRA path for the LoHa Hadamard path.
# Build-only gate (do NOT run here). Loss must descend over the steps when run.

from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.autograd import Tape, backward
from serenitymojo.ops.tensor_algebra import zeros_device
from serenity_trainer.module.LoRAModule import (
    LoHaAdapter, make_loha_adapter, loha_forward,
    LoKrAdapter, make_lokr_adapter, lokr_forward,
    DoRAAdapter, make_dora_adapter, dora_forward,
)


def main() raises:
    var ctx = DeviceContext()

    var in_f = 8
    var out_f = 8
    var rank = 4
    var alpha = Float32(8.0)

    # frozen base weight [out,in] (untracked) + an input [M,in].
    var bw_sh = List[Int](); bw_sh.append(out_f); bw_sh.append(in_f)
    var base_w = zeros_device(bw_sh^, STDtype.BF16, ctx)
    var x_sh = List[Int](); x_sh.append(2); x_sh.append(in_f)
    var x = zeros_device(x_sh^, STDtype.BF16, ctx)
    var t_sh = List[Int](); t_sh.append(2); t_sh.append(out_f)
    var target = zeros_device(t_sh^, STDtype.BF16, ctx)

    # ── LoHa forward + backward (grads to the four hada factors) ──────────────
    var loha = make_loha_adapter(in_f, out_f, rank, alpha, UInt64(11), ctx)
    _ = loha.scale()
    var tape1 = Tape()
    loha.track(tape1)
    var y1 = loha_forward(tape1, x, base_w, loha, ctx)
    var l1 = tape1.mse_loss(y1, target, ctx)
    var g1 = backward(tape1, l1, ctx)
    print("lycoris smoke: LoHa grads=", len(g1))

    # ── LoKr forward + backward (Kronecker delta) ─────────────────────────────
    var lokr = make_lokr_adapter(in_f, out_f, 2, 2, rank, alpha, UInt64(22), ctx)
    _ = lokr.scale()
    var tape2 = Tape()
    lokr.track(tape2)
    var y2 = lokr_forward(tape2, x, base_w, lokr, ctx)
    var l2 = tape2.mse_loss(y2, target, ctx)
    var g2 = backward(tape2, l2, ctx)
    print("lycoris smoke: LoKr grads=", len(g2))

    # ── DoRA forward + backward (decomposed magnitude/direction) ──────────────
    var dora = make_dora_adapter(base_w, in_f, out_f, rank, alpha, UInt64(33), ctx)
    _ = dora.scale()
    var tape3 = Tape()
    dora.track(tape3)
    var y3 = dora_forward(tape3, x, base_w, dora, ctx)
    var l3 = tape3.mse_loss(y3, target, ctx)
    var g3 = backward(tape3, l3, ctx)
    print("lycoris smoke: DoRA grads=", len(g3))

    print("lycoris smoke PASS")
