# driver_smoke.mojo — COMPILE smoke for the model-agnostic driver. Drives a tiny
# synthetic 1-linear-layer "model" (a single trainable weight + a frozen input)
# through a few optimizer steps. This is a *build* gate (do not run here): it
# proves the driver + train_step + ParamSlot compose against a real ModelSpec.
#
# The synthetic model conforms to `trait ModelSpec`. Its trainable param is a
# single linear weight W [OUT, IN]; predict() does y = x @ Wᵀ on the tape and a
# fixed target. The SAME W tensor lives both in the model (as a shared TArc) and
# in the driver's ParamSlot, so track_params() stamping W's id is visible to
# predict() (shared ArcPointer → one buffer, one id).

from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.autograd import Tape
from serenitymojo.ops.tensor_algebra import zeros_device
from serenity_trainer.util.config.TrainConfig import TrainConfig
from serenity_trainer.modelSetup.BaseModelSetup import StepOutput, ModelSpec
from serenity_trainer.trainer.train_step import ParamSlot, TArc
from serenity_trainer.trainer.GenericTrainer import train, DriverResult


# ── synthetic 1-linear-layer model ───────────────────────────────────────────
# Holds the trainable weight as a SHARED TArc (the driver's slot wraps the same
# ArcPointer). x and target are constant BF16 tensors regenerated each predict.
struct LinearModel(Movable, ModelSpec):
    var w: TArc          # [OUT, IN] trainable weight (shared with the slot)
    var in_f: Int
    var out_f: Int
    var rows: Int

    def __init__(out self, var w: TArc, in_f: Int, out_f: Int, rows: Int):
        self.w = w^
        self.in_f = in_f
        self.out_f = out_f
        self.rows = rows

    # predict — record y = x @ Wᵀ on the tape, return (predicted, target, t).
    # W's id was already stamped by the driver's track_params (shared TArc), so
    # record_linear wires grads back to the slot's param.
    def predict(
        mut self, mut tape: Tape, config: TrainConfig, step: Int, ctx: DeviceContext
    ) raises -> StepOutput:
        # constant input x [rows, in] and a no-bias zero bias for the linear.
        var xsh = List[Int](); xsh.append(self.rows); xsh.append(self.in_f)
        var xvals = List[Float32]()
        for i in range(self.rows * self.in_f):
            xvals.append(Float32(0.1) * Float32((i % 7) + 1))
        var x = Tensor.from_host(xvals^, xsh^, STDtype.BF16, ctx)

        var bsh = List[Int](); bsh.append(self.out_f)
        var zero_bias = zeros_device(bsh^, STDtype.BF16, ctx)

        # y = x @ Wᵀ + 0  (W = self.w, id already stamped → grads flow to slot)
        var y = tape.record_linear(x, self.w[], zero_bias, ctx)   # [rows, out]

        # constant target [rows, out]
        var tsh = List[Int](); tsh.append(self.rows); tsh.append(self.out_f)
        var tvals = List[Float32]()
        for _j in range(self.rows * self.out_f):
            tvals.append(Float32(0.5))
        var target = Tensor.from_host(tvals^, tsh^, STDtype.BF16, ctx)

        # timestep scalar (constant weighting kind ignores it; supplied for parity)
        return StepOutput(y^, target^, Float32(0.5))


def main() raises:
    var ctx = DeviceContext()

    comptime IN = 4
    comptime OUT = 3
    comptime ROWS = 2

    # build the shared trainable weight W [OUT, IN] (small nonzero init).
    var wsh = List[Int](); wsh.append(OUT); wsh.append(IN)
    var wvals = List[Float32]()
    for i in range(OUT * IN):
        wvals.append(Float32(0.02) * Float32((i % 5) + 1))
    var w_t = Tensor.from_host(wvals^, wsh^, STDtype.BF16, ctx)
    var w_arc = TArc(w_t^)

    # AdamW state zeros_like(W) (BF16, per port policy — no F32 moments).
    var m_arc = TArc(zeros_device([OUT, IN], STDtype.BF16, ctx))
    var v_arc = TArc(zeros_device([OUT, IN], STDtype.BF16, ctx))
    var accum_arc = TArc(zeros_device([OUT, IN], STDtype.BF16, ctx))

    # the driver's slot SHARES w_arc with the model (same ArcPointer → same id).
    var slots = List[ParamSlot]()
    slots.append(ParamSlot(w_arc, m_arc^, v_arc^, accum_arc^))

    # the model wraps the SAME w_arc handle.
    var model = LinearModel(w_arc, IN, OUT, ROWS)

    # config: AdamW LoRA defaults, but force a few steps and accum=1.
    var cfg = TrainConfig.adamw_lora_defaults()
    cfg.epochs = 1

    var steps_per_epoch = 4
    var base_lr = cfg.learning_rate

    var res: DriverResult = train[LinearModel](
        model, slots, cfg, steps_per_epoch, base_lr, ctx
    )

    print("driver smoke: opt_steps=", res.optimizer_steps,
          " micro_steps=", res.micro_steps,
          " ema_loss=", res.ema_loss,
          " last_grad_norm=", res.last_grad_norm)
    print("driver smoke PASS")
