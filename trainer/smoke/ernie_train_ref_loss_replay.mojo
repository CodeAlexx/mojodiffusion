# Replay the real Serenity Ernie train-step dump through the Mojo loss path.
#
# This is intentionally narrow: it verifies Mojo computes the same F32 MSE loss
# from Serenity's dumped `output.predicted` and `output.target`. It does not
# claim Ernie transformer forward, backward, optimizer, full-finetune, or
# sampler parity.

from max.gpu.host import DeviceContext
from std.math import abs
from std.time import perf_counter_ns

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.json_header import _Cursor, _parse_string, _skip_value
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.reduce import reduce_mean_f32
from serenitymojo.ops.tensor_algebra import mul, sub
from serenitymojo.registry.checkpoints import path_exists
from serenitymojo.tensor import Tensor

from serenity_trainer.util.config.TrainConfigReader import (
    _read_file_bytes,
    _read_scalar,
)


comptime PARITY = "trainer/parity/ernie_train_ref_step000.safetensors"
comptime META = "trainer/parity/ernie_train_ref_meta.json"
# PyTorch's `mean((bf16_pred.float() - f32_target) ** 2)` gives the dump value
# exactly on CPU. Mojo uses a different F32 reduction tree over 286,720 elements;
# the verified 2026-06-05 replay differed by 3.2424927e-05. Keep the tolerance
# narrow and Ernie-specific until the reducer is made PyTorch-order equivalent.
comptime LOSS_EPS = Float32(0.00005)


struct ErnieTrainRefMeta(Copyable, Movable):
    var has_step_index: Bool
    var step_index: Int
    var has_loss_pre_scale: Bool
    var loss_pre_scale: Float32
    var has_loss_for_backward: Bool
    var loss_for_backward: Float32

    def __init__(out self):
        self.has_step_index = False
        self.step_index = 0
        self.has_loss_pre_scale = False
        self.loss_pre_scale = Float32(0.0)
        self.has_loss_for_backward = False
        self.loss_for_backward = Float32(0.0)


def _sec(ns0: UInt, ns1: UInt) -> Float64:
    return Float64(ns1 - ns0) / Float64(1000000000.0)


def _require_artifacts(parity_path: String, meta_path: String) raises:
    var missing = String()
    if not path_exists(parity_path):
        missing += String(" safetensors=") + parity_path
    if not path_exists(meta_path):
        missing += String(" metadata=") + meta_path
    if missing != String():
        raise Error(
            String("Ernie train-ref loss replay missing required artifact(s):")
            + missing
        )


def _parse_step_object(mut cur: _Cursor, mut meta: ErnieTrainRefMeta) raises:
    cur.expect(0x7B)
    cur.skip_ws()
    if cur.peek() == 0x7D:
        cur.advance()
        return
    while True:
        var key = _parse_string(cur)
        cur.expect(0x3A)
        if key == "step_index":
            meta.step_index = Int(_read_scalar(cur).num)
            meta.has_step_index = True
        elif key == "loss_pre_scale":
            meta.loss_pre_scale = Float32(_read_scalar(cur).num)
            meta.has_loss_pre_scale = True
        elif key == "loss_for_backward":
            meta.loss_for_backward = Float32(_read_scalar(cur).num)
            meta.has_loss_for_backward = True
        else:
            _skip_value(cur)
        cur.skip_ws()
        var ch = cur.peek()
        if ch == 0x2C:
            cur.advance()
            continue
        if ch == 0x7D:
            cur.advance()
            break
        raise Error(String("Ernie train-ref meta: bad step object at byte ") + String(cur.pos))


def _parse_steps(mut cur: _Cursor, mut meta: ErnieTrainRefMeta) raises:
    cur.expect(0x5B)
    cur.skip_ws()
    if cur.peek() == 0x5D:
        cur.advance()
        return

    var first = True
    while True:
        if first:
            _parse_step_object(cur, meta)
            first = False
        else:
            _skip_value(cur)
        cur.skip_ws()
        var ch = cur.peek()
        if ch == 0x2C:
            cur.advance()
            continue
        if ch == 0x5D:
            cur.advance()
            break
        raise Error(String("Ernie train-ref meta: bad steps array at byte ") + String(cur.pos))


def read_ernie_train_ref_meta(path: String) raises -> ErnieTrainRefMeta:
    var bytes = _read_file_bytes(path)
    var cur = _Cursor(bytes^)
    var meta = ErnieTrainRefMeta()

    cur.expect(0x7B)
    cur.skip_ws()
    if cur.peek() == 0x7D:
        cur.advance()
        raise Error(String("Ernie train-ref meta has no steps: ") + path)

    while True:
        var key = _parse_string(cur)
        cur.expect(0x3A)
        if key == "steps":
            _parse_steps(cur, meta)
        else:
            _skip_value(cur)
        cur.skip_ws()
        var ch = cur.peek()
        if ch == 0x2C:
            cur.advance()
            continue
        if ch == 0x7D:
            cur.advance()
            break
        raise Error(String("Ernie train-ref meta: bad top-level object at byte ") + String(cur.pos))

    if not meta.has_loss_pre_scale:
        raise Error(String("Ernie train-ref meta missing steps[0].loss_pre_scale: ") + path)
    return meta^


def _require_tensor(have: Dict[String, Int], name: String) raises:
    if not (name in have):
        raise Error(String("Ernie train-ref safetensors missing tensor: ") + name)


def main() raises:
    var all0 = perf_counter_ns()

    print("=== Ernie train-ref loss replay ===")
    print("[parity]", PARITY)
    print("[meta]  ", META)

    _require_artifacts(String(PARITY), String(META))

    var meta0 = perf_counter_ns()
    var meta = read_ernie_train_ref_meta(String(META))
    var meta1 = perf_counter_ns()

    var open0 = perf_counter_ns()
    var st = ShardedSafeTensors.open(String(PARITY))
    var open1 = perf_counter_ns()

    var have = Dict[String, Int]()
    for ref nm in st.names():
        have[nm] = 1
    _require_tensor(have, String("output.predicted"))
    _require_tensor(have, String("output.target"))
    _require_tensor(have, String("output.loss_pre_scale"))

    var predicted_view = st.tensor_view(String("output.predicted"))
    var target_view = st.tensor_view(String("output.target"))
    var loss_view = st.tensor_view(String("output.loss_pre_scale"))
    print(
        "predicted dtype =", predicted_view.dtype.name(),
        " target dtype =", target_view.dtype.name(),
        " loss dtype =", loss_view.dtype.name(),
    )

    var ctx = DeviceContext()
    var predicted_src = Tensor.from_view(predicted_view, ctx)
    var target_src = Tensor.from_view(target_view, ctx)
    var loss_ref = Tensor.from_view(loss_view, ctx)

    # Serenity ModelSetupDiffusionLossMixin.__unmasked_losses casts both
    # tensors to torch.float32 before F.mse_loss(..., reduction='none').
    var cast0 = perf_counter_ns()
    var predicted = cast_tensor(predicted_src, STDtype.F32, ctx)
    var target = cast_tensor(target_src, STDtype.F32, ctx)
    var cast1 = perf_counter_ns()

    var loss0 = perf_counter_ns()
    var diff = sub(predicted, target, ctx)
    var sq = mul(diff, diff, ctx)
    var dims = List[Int]()
    for i in range(len(sq.shape())):
        dims.append(i)
    var loss = reduce_mean_f32(sq, dims^, False, ctx).to_host(ctx)[0]
    var loss1 = perf_counter_ns()

    var ref_host = loss_ref.to_host(ctx)[0]
    var err_dump = abs(loss - ref_host)
    var err_meta = abs(loss - meta.loss_pre_scale)
    var err_dump_meta = abs(ref_host - meta.loss_pre_scale)

    print("Mojo loss =", loss)
    print("dump loss =", ref_host)
    print("meta loss =", meta.loss_pre_scale)
    print(
        "abs_err_dump =", err_dump,
        " abs_err_meta =", err_meta,
        " abs_err_dump_meta =", err_dump_meta,
    )
    print(
        "time_s: meta =", _sec(meta0, meta1),
        " open =", _sec(open0, open1),
        " cast =", _sec(cast0, cast1),
        " loss =", _sec(loss0, loss1),
        " total =", _sec(all0, perf_counter_ns()),
    )

    if err_dump > LOSS_EPS or err_meta > LOSS_EPS or err_dump_meta > LOSS_EPS:
        raise Error("Ernie train-ref loss replay mismatch")

    print("ERNIE TRAIN REF LOSS REPLAY PASS")
