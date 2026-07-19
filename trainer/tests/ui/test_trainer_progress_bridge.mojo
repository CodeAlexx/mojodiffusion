"""Real trainer-loop progress bridge test for the Serenity UI."""

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.autograd import Tape
from serenitymojo.ops.tensor_algebra import zeros_device
from serenity_trainer.modelSetup.BaseModelSetup import StepOutput, ModelSpec
from serenity_trainer.trainer.train_step import ParamSlot, TArc
from serenity_trainer.trainer.GenericTrainer import train_with_progress_file
from serenity_trainer.ui.TrainerRuntimeBridge import (
    TrainerUIRuntime,
    trainer_ui_poll_progress_file,
)
from serenity_trainer.util.config.TrainConfig import TrainConfig


def _expect(cond: Bool, msg: String) raises:
    if not cond:
        print("FAIL:", msg)
        raise Error(msg)


struct BridgeLinearModel(Movable, ModelSpec):
    var w: TArc
    var in_f: Int
    var out_f: Int
    var rows: Int

    def __init__(out self, var w: TArc, in_f: Int, out_f: Int, rows: Int):
        self.w = w^
        self.in_f = in_f
        self.out_f = out_f
        self.rows = rows

    def predict(
        mut self, mut tape: Tape, config: TrainConfig, step: Int, ctx: DeviceContext
    ) raises -> StepOutput:
        var xsh = List[Int]()
        xsh.append(self.rows)
        xsh.append(self.in_f)
        var xvals = List[Float32]()
        for i in range(self.rows * self.in_f):
            xvals.append(Float32(0.1) * Float32((i % 7) + 1))
        var x = Tensor.from_host(xvals^, xsh^, STDtype.BF16, ctx)

        var bsh = List[Int]()
        bsh.append(self.out_f)
        var zero_bias = zeros_device(bsh^, STDtype.BF16, ctx)
        var y = tape.record_linear(x, self.w[], zero_bias, ctx)

        var tsh = List[Int]()
        tsh.append(self.rows)
        tsh.append(self.out_f)
        var tvals = List[Float32]()
        for _j in range(self.rows * self.out_f):
            tvals.append(Float32(0.5))
        var target = Tensor.from_host(tvals^, tsh^, STDtype.BF16, ctx)
        return StepOutput(y^, target^, Float32(0.5))


def main() raises:
    comptime IN = 4
    comptime OUT = 3
    comptime ROWS = 2

    var path = String("/tmp/serenity_real_train_bridge.log")
    var f = open(path, "w")
    f.write(String(""))
    f.close()

    var ctx = DeviceContext()

    var wsh = List[Int]()
    wsh.append(OUT)
    wsh.append(IN)
    var wvals = List[Float32]()
    for i in range(OUT * IN):
        wvals.append(Float32(0.02) * Float32((i % 5) + 1))
    var w_t = Tensor.from_host(wvals^, wsh^, STDtype.BF16, ctx)
    var w_arc = TArc(w_t^)

    var m_arc = TArc(zeros_device([OUT, IN], STDtype.BF16, ctx))
    var v_arc = TArc(zeros_device([OUT, IN], STDtype.BF16, ctx))
    var accum_arc = TArc(zeros_device([OUT, IN], STDtype.BF16, ctx))

    var slots = List[ParamSlot]()
    slots.append(ParamSlot(w_arc, m_arc^, v_arc^, accum_arc^))
    var model = BridgeLinearModel(w_arc, IN, OUT, ROWS)

    var cfg = TrainConfig.adamw_lora_defaults()
    cfg.epochs = 1

    var result = train_with_progress_file[BridgeLinearModel](
        model,
        slots,
        cfg,
        4,
        cfg.learning_rate,
        path.copy(),
        ctx,
    )
    _expect(result.micro_steps == 4, "trainer ran real micro-steps")

    var rt = TrainerUIRuntime()
    rt.progress_file_path = path.copy()
    _expect(trainer_ui_poll_progress_file(rt), "UI runtime consumed emitted trainer events")
    _expect(rt.using_callback_progress, "runtime source is Serenity callback bridge")
    _expect(rt.live.global_step == 4, "final global step from real trainer")
    _expect(rt.live.total_steps == 4, "max step from bridge")
    _expect(rt.live.total_epochs == 1, "max epoch from bridge")
    _expect(rt.live.loss > 0.0, "real loss emitted")
    _expect(rt.live.smooth_loss > 0.0, "real smooth loss emitted")
    _expect(not rt.has_running, "final callback marks run complete")
    print("PASS: real trainer progress bridge")
