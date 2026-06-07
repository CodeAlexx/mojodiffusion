# Bounded Mojo AdamW zero-lr state-init smoke for the Klein model-level LoRA
# AdamW helper.
#
# This calls the same `klein_lora_adamw_step` entry used by train_klein_real,
# but on a tiny synthetic 1-double/1-single LoRA set. It is optimizer-path
# support evidence only: it does not consume OneTrainer train-ref tensors, does
# not execute Klein predict/backward_lora, does not prove full Mojo
# predict/backward/AdamW parity, does not prove nonzero update parity, and does
# not prove low-memory offload/checkpoint backward parity.
#
# CONTRACT MARKERS:
# optimizer-path support evidence only
# does not execute Klein predict/backward_lora
# does not prove full Mojo predict/backward/AdamW parity
# does not prove nonzero update parity
# does not prove low-memory offload/checkpoint backward parity

from std.collections import List
from std.builtin.dtype import DType
from std.gpu.host import DeviceContext

from serenitymojo.models.klein.klein_stack_lora import (
    DBL_SLOTS,
    SGL_SLOTS,
    KleinLoraGrads,
    build_klein_lora_set,
    klein_lora_adamw_step,
)
from serenitymojo.ops.torch_bf16 import torch_bf16_rne_value
from serenitymojo.training.train_step import LoraAdapter


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def _abs(x: Float32) -> Float32:
    if x < Float32(0.0):
        return -x
    return x


def _grad(n: Int, seed: Int, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        var centered = Float32(((i + seed * 7) % 17) - 8)
        out.append(centered * scale)
    return out^


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _snapshot_params(adapters: List[LoraAdapter], use_a: Bool) -> List[List[BFloat16]]:
    var out = List[List[BFloat16]]()
    for i in range(len(adapters)):
        if use_a:
            out.append(adapters[i].a.copy())
        else:
            out.append(adapters[i].b.copy())
    return out^


def _assert_same_bf16(actual: List[BFloat16], expected: List[BFloat16], label: String) raises:
    _require(len(actual) == len(expected), label + String(" length mismatch"))
    for i in range(len(actual)):
        var av = actual[i].cast[DType.float32]()
        var ev = expected[i].cast[DType.float32]()
        _require(av == ev, label + String(" BF16 payload changed at index ") + String(i))


def _assert_param_snapshots_unchanged(
    adapters: List[LoraAdapter],
    before_a: List[List[BFloat16]],
    before_b: List[List[BFloat16]],
    label: String,
) raises:
    _require(len(adapters) == len(before_a), label + String(" A snapshot count mismatch"))
    _require(len(adapters) == len(before_b), label + String(" B snapshot count mismatch"))
    for i in range(len(adapters)):
        _assert_same_bf16(adapters[i].a, before_a[i], label + String(".a[") + String(i) + String("]"))
        _assert_same_bf16(adapters[i].b, before_b[i], label + String(".b[") + String(i) + String("]"))


def _count_changed(
    adapters: List[LoraAdapter],
    before_a: List[List[BFloat16]],
    before_b: List[List[BFloat16]],
) -> Int:
    var changed = 0
    for i in range(len(adapters)):
        for j in range(len(adapters[i].a)):
            if adapters[i].a[j].cast[DType.float32]() != before_a[i][j].cast[DType.float32]():
                changed += 1
        for j in range(len(adapters[i].b)):
            if adapters[i].b[j].cast[DType.float32]() != before_b[i][j].cast[DType.float32]():
                changed += 1
    return changed


def _assert_moments_for_adapter(
    lo: LoraAdapter,
    g_a: List[Float32],
    g_b: List[Float32],
    beta1: Float32,
    beta2: Float32,
    label: String,
) raises -> Float32:
    _require(len(lo.ma) == len(g_a), label + String(" ma len mismatch"))
    _require(len(lo.va) == len(g_a), label + String(" va len mismatch"))
    _require(len(lo.mb) == len(g_b), label + String(" mb len mismatch"))
    _require(len(lo.vb) == len(g_b), label + String(" vb len mismatch"))
    var max_err = Float32(0.0)
    for i in range(len(g_a)):
        var expected_m = torch_bf16_rne_value((Float32(1.0) - beta1) * g_a[i]).cast[DType.float32]()
        var expected_v = torch_bf16_rne_value((Float32(1.0) - beta2) * g_a[i] * g_a[i]).cast[DType.float32]()
        var err_m = _abs(lo.ma[i] - expected_m)
        var err_v = _abs(lo.va[i] - expected_v)
        if err_m > max_err:
            max_err = err_m
        if err_v > max_err:
            max_err = err_v
        _require(err_m == Float32(0.0), label + String(" ma mismatch at ") + String(i))
        _require(err_v == Float32(0.0), label + String(" va mismatch at ") + String(i))
    for i in range(len(g_b)):
        var expected_m = torch_bf16_rne_value((Float32(1.0) - beta1) * g_b[i]).cast[DType.float32]()
        var expected_v = torch_bf16_rne_value((Float32(1.0) - beta2) * g_b[i] * g_b[i]).cast[DType.float32]()
        var err_m = _abs(lo.mb[i] - expected_m)
        var err_v = _abs(lo.vb[i] - expected_v)
        if err_m > max_err:
            max_err = err_m
        if err_v > max_err:
            max_err = err_v
        _require(err_m == Float32(0.0), label + String(" mb mismatch at ") + String(i))
        _require(err_v == Float32(0.0), label + String(" vb mismatch at ") + String(i))
    return max_err


def _build_synthetic_grads(lora_dbl: List[LoraAdapter], lora_sgl: List[LoraAdapter]) -> KleinLoraGrads:
    var dbl_d_a = List[List[Float32]]()
    var dbl_d_b = List[List[Float32]]()
    for i in range(len(lora_dbl)):
        dbl_d_a.append(_grad(len(lora_dbl[i].a), i + 1, Float32(0.001)))
        dbl_d_b.append(_grad(len(lora_dbl[i].b), i + 11, Float32(0.0015)))
    var sgl_d_a = List[List[Float32]]()
    var sgl_d_b = List[List[Float32]]()
    for i in range(len(lora_sgl)):
        sgl_d_a.append(_grad(len(lora_sgl[i].a), i + 101, Float32(0.00125)))
        sgl_d_b.append(_grad(len(lora_sgl[i].b), i + 111, Float32(0.00175)))
    return KleinLoraGrads(
        dbl_d_a^, dbl_d_b^, sgl_d_a^, sgl_d_b^,
        _zeros(0), _zeros(0), _zeros(0), _zeros(0), _zeros(0),
        _zeros(0), _zeros(0), _zeros(0), _zeros(0), _zeros(0),
    )


def _assert_all_state_init_moments(
    lora_dbl: List[LoraAdapter],
    lora_sgl: List[LoraAdapter],
    grads: KleinLoraGrads,
    beta1: Float32,
    beta2: Float32,
) raises -> Float32:
    var max_err = Float32(0.0)
    for i in range(len(lora_dbl)):
        var e = _assert_moments_for_adapter(
            lora_dbl[i], grads.dbl_d_a[i], grads.dbl_d_b[i], beta1, beta2,
            String("dbl[") + String(i) + String("]"),
        )
        if e > max_err:
            max_err = e
    for i in range(len(lora_sgl)):
        var e = _assert_moments_for_adapter(
            lora_sgl[i], grads.sgl_d_a[i], grads.sgl_d_b[i], beta1, beta2,
            String("sgl[") + String(i) + String("]"),
        )
        if e > max_err:
            max_err = e
    return max_err


def main() raises:
    comptime NUM_DOUBLE = 1
    comptime NUM_SINGLE = 1
    comptime D = 8
    comptime F = 16
    comptime RANK = 2
    comptime ALPHA = Float32(2.0)
    var beta1 = Float32(0.9)
    var beta2 = Float32(0.999)
    var eps = Float32(1.0e-8)
    var weight_decay = Float32(0.01)
    var ctx = DeviceContext()

    var lora = build_klein_lora_set(NUM_DOUBLE, NUM_SINGLE, D, F, RANK, ALPHA)
    _require(len(lora.dbl) == NUM_DOUBLE * DBL_SLOTS, String("double adapter count mismatch"))
    _require(len(lora.sgl) == NUM_SINGLE * SGL_SLOTS, String("single adapter count mismatch"))
    var grads = _build_synthetic_grads(lora.dbl, lora.sgl)

    var dbl_a_before = _snapshot_params(lora.dbl, True)
    var dbl_b_before = _snapshot_params(lora.dbl, False)
    var sgl_a_before = _snapshot_params(lora.sgl, True)
    var sgl_b_before = _snapshot_params(lora.sgl, False)

    klein_lora_adamw_step(
        lora, grads, 1, Float32(0.0), ctx, beta1, beta2, eps, weight_decay
    )
    _assert_param_snapshots_unchanged(lora.dbl, dbl_a_before, dbl_b_before, String("dbl zero-lr"))
    _assert_param_snapshots_unchanged(lora.sgl, sgl_a_before, sgl_b_before, String("sgl zero-lr"))
    var max_moment_err = _assert_all_state_init_moments(lora.dbl, lora.sgl, grads, beta1, beta2)

    var dbl_a_step1 = _snapshot_params(lora.dbl, True)
    var dbl_b_step1 = _snapshot_params(lora.dbl, False)
    var sgl_a_step1 = _snapshot_params(lora.sgl, True)
    var sgl_b_step1 = _snapshot_params(lora.sgl, False)
    klein_lora_adamw_step(
        lora, grads, 2, Float32(1.0e-3), ctx, beta1, beta2, eps, weight_decay
    )
    var changed = _count_changed(lora.dbl, dbl_a_step1, dbl_b_step1)
    changed += _count_changed(lora.sgl, sgl_a_step1, sgl_b_step1)
    _require(changed > 0, String("positive-lr optimizer step did not change any BF16 adapter value"))

    print(
        "[klein-lora-adamw-state-init] PASS",
        " dbl=", len(lora.dbl),
        " sgl=", len(lora.sgl),
        " max_moment_err=", max_moment_err,
        " positive_lr_changed=", changed,
    )
