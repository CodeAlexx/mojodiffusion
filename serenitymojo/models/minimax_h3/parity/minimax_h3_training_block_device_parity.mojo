# MiniMax-H3 reduced F32 PARITY-ONLY DEVICE block gate.
#
# Uses the existing Torch-autograd fixture generated from pinned Musubi
# b8717864713c9e4e7ef3d56eba1fc695a9b626a5.  It gates the Mojo/MAX device
# composition's forward, d_x, and all eight LoRA A/B gradients at H=56 with
# reduced S/Dh/D/F/rank.  It also exercises the explicit Serenity-runtime
# `[value;gate]` <-> Musubi-training `[gate;value]` FC1 boundary for the frozen
# weight and LoRA B rows. This module cannot be imported by a product trainer.
#
# Evidence level: reduced-shape F32 per-block device parity only.
# Fixture receipt is checked before build with:
#   (cd serenitymojo/models/minimax_h3/parity/fixtures && sha256sum -c minimax_h3_training_block.sha256)

from max.gpu.host import DeviceContext
from std.collections import List
from std.math import sqrt
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.minimax_h3.parity.training_block_device_reference import (
    MiniMaxH3BlockModulationDevice,
    MiniMaxH3LoraAdapterDevice,
    MiniMaxH3TrainingBlockLoraDevice,
    MiniMaxH3TrainingBlockWeightsDevice,
    minimax_h3_training_block_backward_device,
    minimax_h3_training_block_forward_device,
    minimax_h3_training_swap_fc1_rows_device,
)


comptime TArc = ArcPointer[Tensor]
comptime REF = "serenitymojo/models/minimax_h3/parity/fixtures/minimax_h3_training_block.safetensors"
comptime S = 3
comptime H = 56
comptime DH = 4
comptime D = 8
comptime FF = 12
comptime ROT = 2
comptime RANK = 2
comptime MAX_ABS_TOL = Float32(1.0e-6)
comptime COS_TOL = Float32(0.999999)
comptime REL_L2_TOL = Float64(1.0e-5)
comptime MAG_RATIO_TOL = Float64(1.0e-5)


def _load(ref st: SafeTensors, name: String) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.dtype != STDtype.F32:
        raise Error(String("expected F32 fixture tensor: ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32](capacity=tv.numel())
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


def _t(
    ref st: SafeTensors,
    name: String,
    var shape: List[Int],
    ctx: DeviceContext,
) raises -> Tensor:
    var info = st.tensor_info(name)
    if info.shape != shape:
        raise Error(String("fixture shape mismatch for ") + name)
    return Tensor.from_host(_load(st, name), shape^, STDtype.F32, ctx)


def _ta(
    ref st: SafeTensors,
    name: String,
    var shape: List[Int],
    ctx: DeviceContext,
) raises -> TArc:
    return TArc(_t(st, name, shape^, ctx))


def _adapter(
    ref st: SafeTensors,
    name: String,
    inf: Int,
    outf: Int,
    ctx: DeviceContext,
    mapped_fc1_b: Bool = False,
) raises -> MiniMaxH3LoraAdapterDevice:
    var multiplier = _load(st, "meta.lora_multiplier")[0]
    var alpha = _load(st, "meta.lora_alpha")[0]
    var rankf = _load(st, "meta.lora_rank")[0]
    var scale = multiplier * alpha / rankf
    var expected = _load(st, "meta.lora_scale")[0]
    if abs(scale - expected) > 1.0e-7:
        raise Error("MiniMax-H3 LoRA multiplier*alpha/rank mapping mismatch")
    var a = _ta(st, String("lora.") + name + ".a", [RANK, inf], ctx)
    var b: TArc
    if mapped_fc1_b:
        # Simulate an inference-runtime adapter: raw Musubi [gate;value] B is
        # swapped to runtime [value;gate], then mapped back at the train seam.
        var raw_b = _t(st, String("lora.") + name + ".b", [outf, RANK], ctx)
        var runtime_b = minimax_h3_training_swap_fc1_rows_device(raw_b, FF, ctx)
        var train_b = minimax_h3_training_swap_fc1_rows_device(runtime_b, FF, ctx)
        b = TArc(train_b^)
    else:
        b = _ta(st, String("lora.") + name + ".b", [outf, RANK], ctx)
    return MiniMaxH3LoraAdapterDevice(a, b, RANK, inf, outf, scale)


def _check(
    label: String,
    got: Tensor,
    want: List[Float32],
    ctx: DeviceContext,
) raises -> Bool:
    var actual = got.to_host(ctx)
    if len(actual) != len(want):
        print("  FAIL", label, "length", len(actual), "!=", len(want))
        return False
    var dot = Float64(0.0)
    var ng = Float64(0.0)
    var nw = Float64(0.0)
    var nd = Float64(0.0)
    var worst = Float32(0.0)
    for i in range(len(actual)):
        var g = Float64(actual[i])
        var w = Float64(want[i])
        dot += g * w
        ng += g * g
        nw += w * w
        var diff64 = g - w
        nd += diff64 * diff64
        var delta = actual[i] - want[i]
        var ad = -delta if delta < 0.0 else delta
        if ad > worst:
            worst = ad
    if ng == 0.0 or nw == 0.0:
        print("  FAIL", label, "degenerate norm", ng, nw)
        return False
    var cos = Float32(dot / sqrt(ng * nw))
    var rel_l2 = sqrt(nd / nw)
    var mag_ratio = sqrt(ng / nw)
    var mag_error = abs(mag_ratio - 1.0)
    var ok = (
        cos >= COS_TOL and worst <= MAX_ABS_TOL
        and rel_l2 <= REL_L2_TOL and mag_error <= MAG_RATIO_TOL
    )
    print(
        "  ", "ok" if ok else "FAIL", label, "cos", cos,
        "max_abs", worst, "rel_l2", rel_l2, "mag_ratio", mag_ratio,
    )
    return ok


def main() raises:
    print("MiniMax-H3 reduced F32 DEVICE training-block parity")
    print("  H=56; S/Dh/D/F/rank reduced; no BF16/released-stack claim")
    print("  qkv=[all-q;all-k;all-v], train fc1=[gate;value]")
    print("  exact one-way inference-runtime FC1 + LoRA-B mapping exercised")
    var ctx = DeviceContext()
    var st = SafeTensors.open(String(REF))
    comptime I = H * DH

    # Simulate current inference runtime's transformed FC1 then map it back at
    # the explicit training boundary.  The block never accepts it silently.
    var raw_fc1 = _t(st, "w.fc1", [2 * FF, D], ctx)
    var runtime_fc1 = minimax_h3_training_swap_fc1_rows_device(raw_fc1, FF, ctx)
    if not _check(
        "boundary.fc1_runtime", runtime_fc1,
        _load(st, "boundary.fc1_runtime"), ctx,
    ):
        raise Error("MiniMax-H3 one-way frozen FC1 boundary mapping failed")
    var train_fc1 = minimax_h3_training_swap_fc1_rows_device(runtime_fc1, FF, ctx)
    var weights = MiniMaxH3TrainingBlockWeightsDevice(
        _ta(st, "w.norm1", [D], ctx),
        _ta(st, "w.qkv", [3 * I, D], ctx),
        _ta(st, "w.q_norm", [DH], ctx),
        _ta(st, "w.k_norm", [DH], ctx),
        _ta(st, "w.out_proj", [D, I], ctx),
        _ta(st, "w.norm2", [D], ctx),
        TArc(train_fc1^),
        _ta(st, "w.fc2", [D, FF], ctx),
    )
    var mod = MiniMaxH3BlockModulationDevice(
        _ta(st, "mod.shift_msa", [S, D], ctx),
        _ta(st, "mod.scale_msa", [S, D], ctx),
        _ta(st, "mod.gate_msa", [S, D], ctx),
        _ta(st, "mod.shift_mlp", [S, D], ctx),
        _ta(st, "mod.scale_mlp", [S, D], ctx),
        _ta(st, "mod.gate_mlp", [S, D], ctx),
    )
    var lora = MiniMaxH3TrainingBlockLoraDevice(
        _adapter(st, "qkv", D, 3 * I, ctx),
        _adapter(st, "out_proj", I, D, ctx),
        _adapter(st, "fc1", D, 2 * FF, ctx, True),
        _adapter(st, "fc2", FF, D, ctx),
    )
    var raw_fc1_b = _t(st, "lora.fc1.b", [2 * FF, RANK], ctx)
    var runtime_fc1_b = minimax_h3_training_swap_fc1_rows_device(
        raw_fc1_b, FF, ctx,
    )
    if not _check(
        "boundary.lora_fc1_b_runtime", runtime_fc1_b,
        _load(st, "boundary.lora_fc1_b_runtime"), ctx,
    ):
        raise Error("MiniMax-H3 one-way LoRA FC1-B boundary mapping failed")
    var x = _t(st, "in.x", [S, D], ctx)
    var cos = _t(st, "in.cos", [S, ROT], ctx)
    var sin = _t(st, "in.sin", [S, ROT], ctx)
    var dy = _t(st, "in.dy", [S, D], ctx)

    var y = minimax_h3_training_block_forward_device[
        S, H, DH, D, FF, ROT
    ](x, weights, mod, lora, cos, sin, 1.0e-5, 1.0e-5, ctx)
    var b = minimax_h3_training_block_backward_device[
        S, H, DH, D, FF, ROT
    ](dy, x, weights, mod, lora, cos, sin, 1.0e-5, 1.0e-5, ctx)

    var ok = True
    ok = _check("forward.y", y, _load(st, "out.y"), ctx) and ok
    ok = _check("grad.x", b.d_x[], _load(st, "grad.x"), ctx) and ok
    ok = _check("grad.qkv.A", b.qkv.d_a[], _load(st, "grad.qkv.a"), ctx) and ok
    ok = _check("grad.qkv.B", b.qkv.d_b[], _load(st, "grad.qkv.b"), ctx) and ok
    ok = _check("grad.out_proj.A", b.out_proj.d_a[], _load(st, "grad.out_proj.a"), ctx) and ok
    ok = _check("grad.out_proj.B", b.out_proj.d_b[], _load(st, "grad.out_proj.b"), ctx) and ok
    ok = _check("grad.fc1.A", b.fc1.d_a[], _load(st, "grad.fc1.a"), ctx) and ok
    ok = _check("grad.fc1.B", b.fc1.d_b[], _load(st, "grad.fc1.b"), ctx) and ok
    ok = _check("grad.fc2.A", b.fc2.d_a[], _load(st, "grad.fc2.a"), ctx) and ok
    ok = _check("grad.fc2.B", b.fc2.d_b[], _load(st, "grad.fc2.b"), ctx) and ok
    if ok:
        print("PASS: MiniMax-H3 device block forward, d_x, and all LoRA grads")
    else:
        raise Error("MiniMax-H3 device training-block parity failed")
