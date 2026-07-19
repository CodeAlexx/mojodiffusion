# Phase 5 gate: save_train_state → load_train_state round-trips resume state.
from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenity_trainer.trainer.train_step import ParamSlot
from serenity_trainer.trainer.TrainState import save_train_state, load_train_state, TrainProgress

comptime TArc = ArcPointer[Tensor]

def mk(v: Float32, ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_host([v, v, v, v], [4], STDtype.BF16, ctx))

def main() raises:
    var ctx = DeviceContext()
    var slots = List[ParamSlot]()
    slots.append(ParamSlot(mk(0.0, ctx), mk(1.0, ctx), mk(2.0, ctx), mk(0.0, ctx)))
    slots.append(ParamSlot(mk(0.0, ctx), mk(3.0, ctx), mk(4.0, ctx), mk(0.0, ctx)))
    var prog = TrainProgress()
    prog.epoch = 2; prog.epoch_step = 5; prog.epoch_sample = 7; prog.global_step = 40
    var ema = List[TArc]()
    save_train_state(String("/tmp/ts_test"), slots, ema, prog, 10, ctx)

    var loaded = load_train_state(String("/tmp/ts_test"), ctx)
    print("global_step =", loaded.prog.global_step, "(40)")
    print("opt_step =", loaded.opt_step, "(10)")
    print("n_slots =", loaded.n_slots, "(2)")
    var ok = loaded.prog.global_step == 40 and loaded.opt_step == 10 and loaded.n_slots == 2
    print("TRAIN STATE ROUNDTRIP", "PASS" if ok else "FAIL")
