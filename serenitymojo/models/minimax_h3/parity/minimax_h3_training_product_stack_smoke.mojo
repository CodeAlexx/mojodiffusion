# Structural smoke for the reusable MiniMax-H3 product training core/stack.
#
# Uses the reduced 50-block Torch fixture, maps raw FC1 base/LoRA-B into the
# single product runtime `[value;gate]` convention, assembles a synthetic AdaLN
# table and exercises the product gather, then gates final y, initial dx, and
# all 400 gradients. This is not a released-checkpoint or trainer smoke.
# Required invocation (fixture preflight + SHA receipt + build + run):
#   bash serenitymojo/models/minimax_h3/parity/run_minimax_h3_training_product_stack_smoke.sh

from max.gpu.host import DeviceContext
from std.collections import List
from std.math import isfinite
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.minimax_h3.training_core import (
    MiniMaxH3TrainingBlockLoraDevice,
    MiniMaxH3TrainingLoraAdapterDevice,
    MiniMaxH3TrainingModulationDevice,
    MiniMaxH3TrainingWeightsDevice,
    minimax_h3_training_modulation_from_table,
)
from serenitymojo.models.minimax_h3.training_stack import (
    MiniMaxH3TrainingLoraDeviceSet,
    minimax_h3_training_stack_backward_resident,
    minimax_h3_training_stack_forward_resident,
)
from serenitymojo.training.minimax_h3.contract import MiniMaxH3TrainingContract
from serenitymojo.training.minimax_h3.device_loop import (
    validate_minimax_h3_device_training_contract,
)
from serenitymojo.models.minimax_h3.parity.training_block_device_reference import (
    minimax_h3_training_swap_fc1_rows_device,
)
from serenitymojo.models.minimax_h3.parity.minimax_h3_training_stack50_bf16_flash_parity import (
    _bf16_as_f32,
    _check,
    _load_bf16,
    _load_f32,
    _t,
    _t_f32,
    _ta,
    _ta_f32,
)


comptime TArc = ArcPointer[Tensor]
comptime FIXTURE = "serenitymojo/models/minimax_h3/parity/fixtures/minimax_h3_training_stack50_bf16_flash.safetensors"
comptime FIXTURE_SHA256 = "7f98af1639d600776c3aea8e51f846dbf2208ab6b4713c949eb32114b5c80b6c"
comptime S = 3
comptime H = 56
comptime DH = 8
comptime D = 8
comptime FF = 12
comptime ROT = 4
comptime RANK = 2


def _prefix(block: Int) -> String:
    return String("block.") + String(block)


def _adapter(
    ref st: SafeTensors,
    prefix: String,
    name: String,
    inf: Int,
    outf: Int,
    scale: Float32,
    ctx: DeviceContext,
    runtime_fc1_b: Bool = False,
) raises -> MiniMaxH3TrainingLoraAdapterDevice:
    var p = prefix + String(".lora.") + name
    var a = _ta_f32(st, p + String(".a"), [RANK, inf], ctx)
    var b: TArc
    if runtime_fc1_b:
        var raw = _t_f32(st, p + String(".b"), [outf, RANK], ctx)
        b = TArc(minimax_h3_training_swap_fc1_rows_device(raw, FF, ctx))
    else:
        b = _ta_f32(st, p + String(".b"), [outf, RANK], ctx)
    return MiniMaxH3TrainingLoraAdapterDevice(a, b, RANK, inf, outf, scale)


def _weights(
    ref st: SafeTensors, prefix: String, ctx: DeviceContext
) raises -> MiniMaxH3TrainingWeightsDevice:
    comptime I = H * DH
    var p = prefix + String(".w.")
    var raw_fc1 = _t(st, p + String("fc1"), [2 * FF, D], ctx)
    var runtime_fc1 = minimax_h3_training_swap_fc1_rows_device(raw_fc1, FF, ctx)
    return MiniMaxH3TrainingWeightsDevice(
        _ta(st, p + String("norm1"), [D], ctx),
        _ta(st, p + String("qkv"), [3 * I, D], ctx),
        _ta(st, p + String("q_norm"), [DH], ctx),
        _ta(st, p + String("k_norm"), [DH], ctx),
        _ta(st, p + String("out_proj"), [D, I], ctx),
        _ta(st, p + String("norm2"), [D], ctx),
        TArc(runtime_fc1^),
        _ta(st, p + String("fc2"), [D, FF], ctx),
    )


def _modulation_table(
    ref st: SafeTensors, prefix: String, ctx: DeviceContext
) raises -> Tensor:
    var chunks = List[List[BFloat16]]()
    for name in [
        String("shift_msa"), String("scale_msa"), String("gate_msa"),
        String("shift_mlp"), String("scale_mlp"), String("gate_mlp"),
    ]:
        chunks.append(_load_bf16(st, prefix + String(".mod.") + name, [S, D]))
    var table = List[BFloat16](capacity=S * 6 * D)
    for row in range(S):
        for chunk in range(6):
            for column in range(D):
                table.append(chunks[chunk][row * D + column])
    return Tensor.from_host_bf16(table^, [S, 6 * D], ctx)


def _check_grad(
    label: String,
    got: Tensor,
    ref st: SafeTensors,
    name: String,
    var shape: List[Int],
    slot: Int,
    arm_b: Bool,
    ctx: DeviceContext,
) raises -> Bool:
    var value = got.clone(ctx)
    if slot == 2 and arm_b:
        # Product returns runtime [value;gate]; oracle is raw [gate;value].
        value = minimax_h3_training_swap_fc1_rows_device(value, FF, ctx)
    if slot == 0 and not arm_b:
        return _check(label, value, _load_f32(st, name, shape.copy()), shape^, STDtype.F32, ctx, 0.99972, 2.6e-5, 2.55e-2, 1.9e-2)
    if slot == 0:
        return _check(label, value, _load_f32(st, name, shape.copy()), shape^, STDtype.F32, ctx, 0.99952, 1.45e-5, 4.3e-2, 2.9e-2)
    if slot == 1 and not arm_b:
        return _check(label, value, _load_f32(st, name, shape.copy()), shape^, STDtype.F32, ctx, 0.99975, 4.5e-6, 2.9e-2, 2.4e-2)
    if slot == 1:
        return _check(label, value, _load_f32(st, name, shape.copy()), shape^, STDtype.F32, ctx, 0.9992, 1.8e-5, 5.3e-2, 3.4e-2)
    if slot == 2 and not arm_b:
        return _check(label, value, _load_f32(st, name, shape.copy()), shape^, STDtype.F32, ctx, 0.99955, 7.0e-7, 3.3e-2, 1.9e-2)
    if slot == 2:
        return _check(label, value, _load_f32(st, name, shape.copy()), shape^, STDtype.F32, ctx, 0.99966, 8.5e-7, 2.8e-2, 2.4e-2)
    if not arm_b:
        return _check(label, value, _load_f32(st, name, shape.copy()), shape^, STDtype.F32, ctx, 0.99962, 8.5e-7, 3.15e-2, 2.6e-2)
    return _check(label, value, _load_f32(st, name, shape.copy()), shape^, STDtype.F32, ctx, 0.99952, 5.8e-7, 3.65e-2, 2.0e-2)


def _policy_rejections() raises:
    var full = MiniMaxH3TrainingContract.intake_default()
    full.task = String("t2va")
    full.full_finetune = True
    var rejected_full = False
    try:
        validate_minimax_h3_device_training_contract(full)
    except:
        rejected_full = True
    if not rejected_full:
        raise Error("product smoke: full finetune did not fail loud")
    var int8 = MiniMaxH3TrainingContract.intake_default()
    int8.task = String("t2va")
    int8.base_storage = String("convrot_int8")
    var rejected_int8 = False
    try:
        validate_minimax_h3_device_training_contract(int8)
    except:
        rejected_int8 = True
    if not rejected_int8:
        raise Error("product smoke: ConvRot INT8 backward did not fail loud")


def main() raises:
    print("MiniMax-H3 reusable product training core/50-stack structural smoke")
    print("  reduced fixture; runtime QKV + FC1[value;gate]; AdaLN table gather")
    print("  no released checkpoint/base, no dataset, no trainer launch")
    print("  validated fixture sha256", FIXTURE_SHA256)
    _policy_rejections()
    var ctx = DeviceContext()
    var st = SafeTensors.open(String(FIXTURE))
    var multiplier = _load_f32(st, "meta.lora_multiplier", [1])[0]
    var alpha = _load_f32(st, "meta.lora_alpha", [1])[0]
    var rank = _load_f32(st, "meta.lora_rank", [1])[0]
    if not isfinite(rank) or rank != Float32(RANK):
        raise Error("product smoke: invalid exact rank")
    var scale = multiplier * alpha / rank

    var weights = List[MiniMaxH3TrainingWeightsDevice](capacity=50)
    var mods = List[MiniMaxH3TrainingModulationDevice](capacity=50)
    var blocks = List[MiniMaxH3TrainingBlockLoraDevice](capacity=50)
    var ids: List[Int] = [0, 1, 2]
    comptime I = H * DH
    for block in range(50):
        var p = _prefix(block)
        weights.append(_weights(st, p, ctx))
        var table = _modulation_table(st, p, ctx)
        mods.append(minimax_h3_training_modulation_from_table[S, D](table, ids, ctx))
        blocks.append(MiniMaxH3TrainingBlockLoraDevice(
            _adapter(st, p, "qkv", D, 3 * I, scale, ctx),
            _adapter(st, p, "out_proj", I, D, scale, ctx),
            _adapter(st, p, "fc1", D, 2 * FF, scale, ctx, True),
            _adapter(st, p, "fc2", FF, D, scale, ctx),
        ))
    var lora = MiniMaxH3TrainingLoraDeviceSet(blocks^)
    var x = _t(st, "in.x", [S, D], ctx)
    var cos = _t(st, "in.cos", [S, ROT], ctx)
    var sin = _t(st, "in.sin", [S, ROT], ctx)
    var tape = minimax_h3_training_stack_forward_resident[S, H, DH, D, FF, ROT](
        x, weights, mods, lora, cos, sin, ctx
    )
    var final_y = tape.output()
    var ok = _check(
        "product.final.y", final_y[], _bf16_as_f32(st, "out.y", [S, D]), [S, D],
        STDtype.BF16, ctx, 0.99996, 3.3e-3, 8.7e-3, 2.5e-3,
    )
    var d_y = _t(st, "in.dy", [S, D], ctx)
    var grads = minimax_h3_training_stack_backward_resident[S, H, DH, D, FF, ROT](
        d_y, weights, mods, lora, cos, sin, tape, ctx
    )
    ok = _check(
        "product.initial.dx", grads.d_x[], _bf16_as_f32(st, "grad.input.0", [S, D]), [S, D],
        STDtype.BF16, ctx, 0.999955, 1.7e-3, 8.9e-3, 2.3e-3,
    ) and ok
    var names: List[String] = [
        String("qkv"), String("out_proj"), String("fc1"), String("fc2")
    ]
    for block in range(50):
        var p = _prefix(block)
        for slot in range(4):
            var index = block * 4 + slot
            var inf = D
            var outf = 3 * I
            if slot == 1:
                inf = I
                outf = D
            elif slot == 2:
                inf = D
                outf = 2 * FF
            elif slot == 3:
                inf = FF
                outf = D
            var base = p + String(".grad.") + names[slot]
            ok = _check_grad(
                String("product.grad.") + names[slot] + String(".A.") + String(block),
                grads.d_a[index][], st, base + String(".a"), [RANK, inf],
                slot, False, ctx,
            ) and ok
            ok = _check_grad(
                String("product.grad.") + names[slot] + String(".B.") + String(block),
                grads.d_b[index][], st, base + String(".b"), [outf, RANK],
                slot, True, ctx,
            ) and ok
    if grads.adapter_count() != 200:
        raise Error("product smoke: adapter-gradient count mismatch")
    if ok:
        print("PASS: product core + AdaLN gather + exact 50 reverse + 400 grads")
        print("PASS: full-FT and ConvRot INT8 backward rejected")
    else:
        raise Error("MiniMax-H3 product training structural smoke failed")
