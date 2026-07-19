# lora_adamw_plain_device_grads_smoke.mojo — resident LoRA AdamW device-grad smoke.
#
# Run:
#   pixi run mojo run -I . serenitymojo/training/lora_adamw_plain_device_grads_smoke.mojo

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.lora_adamw_plain_fused import (
    fused_lora_adamw_plain_step_resident,
    fused_lora_adamw_plain_step_resident_device_grads,
    fused_lora_adamw_plain_step_resident_preloaded_grads,
    lora_adamw_plain_device_state_init,
    lora_adamw_plain_device_state_copy_device_grad_pair,
    lora_adamw_plain_device_state_sync_moments,
    lora_adamw_plain_device_state_sync_params,
    lora_adamw_plain_preloaded_shared_abi_train_step,
)
from serenitymojo.training.training_arena import TrainingArena


comptime TArc = ArcPointer[Tensor]


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("lora_adamw_plain_device_grads_smoke FAILED: ") + msg)


def _abs(x: Float32) -> Float32:
    if x < Float32(0.0):
        return -x
    return x


def _close(a: Float32, b: Float32, tol: Float32) -> Bool:
    return _abs(a - b) <= tol


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _vals(n: Int, scale: Float32, offset: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        var sign = Float32(1.0) if i % 2 == 0 else Float32(-1.0)
        out.append(sign * (offset + Float32(i + 1) * scale))
    return out^


def _mk_adapter(rank: Int, in_f: Int, out_f: Int, seed: Int) -> LoraAdapter:
    return LoraAdapter(
        _vals(rank * in_f, Float32(0.001), Float32(seed) * Float32(0.01)),
        _zeros(out_f * rank),
        rank,
        in_f,
        out_f,
        Float32(1.0) / Float32(rank),
        _zeros(rank * in_f),
        _zeros(rank * in_f),
        _zeros(out_f * rank),
        _zeros(out_f * rank),
    )


def _compare_adapters(a: List[LoraAdapter], b: List[LoraAdapter]) raises:
    _check(len(a) == len(b), "adapter count mismatch")
    for i in range(len(a)):
        _check(len(a[i].a) == len(b[i].a), "A len mismatch")
        _check(len(a[i].b) == len(b[i].b), "B len mismatch")
        for j in range(len(a[i].a)):
            _check(
                Int(a[i].a[j].to_bits[DType.uint16]())
                == Int(b[i].a[j].to_bits[DType.uint16]()),
                String("A bit mismatch at adapter ") + String(i),
            )
        for j in range(len(a[i].b)):
            _check(
                Int(a[i].b[j].to_bits[DType.uint16]())
                == Int(b[i].b[j].to_bits[DType.uint16]()),
                String("B bit mismatch at adapter ") + String(i),
            )
        for j in range(len(a[i].ma)):
            _check(a[i].ma[j] == b[i].ma[j], "ma mismatch")
            _check(a[i].va[j] == b[i].va[j], "va mismatch")
        for j in range(len(a[i].mb)):
            _check(a[i].mb[j] == b[i].mb[j], "mb mismatch")
            _check(a[i].vb[j] == b[i].vb[j], "vb mismatch")


def _scaled_grads(src: List[List[Float32]], scale: Float32) -> List[List[Float32]]:
    var out = List[List[Float32]]()
    for i in range(len(src)):
        var row = List[Float32]()
        for j in range(len(src[i])):
            row.append(src[i][j] * scale)
        out.append(row^)
    return out^


def main() raises:
    var ctx = DeviceContext()
    var host_ads = List[LoraAdapter]()
    var dev_ads = List[LoraAdapter]()
    host_ads.append(_mk_adapter(2, 3, 4, 1))
    host_ads.append(_mk_adapter(2, 5, 3, 2))
    for i in range(len(host_ads)):
        dev_ads.append(host_ads[i].copy())

    var d_a = List[List[Float32]]()
    var d_b = List[List[Float32]]()
    d_a.append(_vals(len(host_ads[0].a), Float32(0.003), Float32(0.02)))
    d_b.append(_vals(len(host_ads[0].b), Float32(0.004), Float32(0.03)))
    d_a.append(_vals(len(host_ads[1].a), Float32(0.005), Float32(0.04)))
    d_b.append(_vals(len(host_ads[1].b), Float32(0.006), Float32(0.05)))

    var host_state = lora_adamw_plain_device_state_init(host_ads, 0, len(host_ads), ctx)
    var dev_state = lora_adamw_plain_device_state_init(dev_ads, 0, len(dev_ads), ctx)

    fused_lora_adamw_plain_step_resident(
        host_state,
        host_ads,
        d_a,
        d_b,
        1,
        Float32(1.0e-3),
        Float32(0.9),
        Float32(0.999),
        Float32(1.0e-8),
        Float32(0.01),
        ctx,
    )

    var grad_indices = List[Int]()
    var d_a_t = List[TArc]()
    var d_b_t = List[TArc]()
    for i in range(len(dev_ads)):
        grad_indices.append(i)
        d_a_t.append(TArc(Tensor.from_host(d_a[i].copy(), [len(d_a[i])], STDtype.F32, ctx)))
        d_b_t.append(TArc(Tensor.from_host(d_b[i].copy(), [len(d_b[i])], STDtype.F32, ctx)))

    var dev_norm = fused_lora_adamw_plain_step_resident_device_grads(
        dev_state,
        dev_ads,
        grad_indices,
        d_a_t,
        d_b_t,
        1,
        Float32(1.0e-3),
        Float32(0.9),
        Float32(0.999),
        Float32(1.0e-8),
        Float32(0.01),
        ctx,
        Float32(1.0),
        True,
        Float32(10.0),
    )
    _check(dev_norm > Float32(0.0), "device grad norm should be positive")
    lora_adamw_plain_device_state_sync_moments(host_state, host_ads, ctx)
    lora_adamw_plain_device_state_sync_moments(dev_state, dev_ads, ctx)
    _compare_adapters(host_ads, dev_ads)

    var host_clip_ads = List[LoraAdapter]()
    var dev_clip_ads = List[LoraAdapter]()
    host_clip_ads.append(_mk_adapter(2, 3, 4, 3))
    host_clip_ads.append(_mk_adapter(2, 5, 3, 4))
    for i in range(len(host_clip_ads)):
        dev_clip_ads.append(host_clip_ads[i].copy())
    var host_clip_state = lora_adamw_plain_device_state_init(
        host_clip_ads, 0, len(host_clip_ads), ctx
    )
    var dev_clip_state = lora_adamw_plain_device_state_init(
        dev_clip_ads, 0, len(dev_clip_ads), ctx
    )
    var clip_norm = Float32(0.025)
    var clip_dev_norm = fused_lora_adamw_plain_step_resident_device_grads(
        dev_clip_state,
        dev_clip_ads,
        grad_indices,
        d_a_t,
        d_b_t,
        1,
        Float32(1.0e-3),
        Float32(0.9),
        Float32(0.999),
        Float32(1.0e-8),
        Float32(0.01),
        ctx,
        Float32(1.0),
        True,
        clip_norm,
    )
    _check(clip_dev_norm > clip_norm, "clip case should exceed max norm")
    var clip_scale = clip_norm / clip_dev_norm
    var d_a_clip = _scaled_grads(d_a, clip_scale)
    var d_b_clip = _scaled_grads(d_b, clip_scale)
    fused_lora_adamw_plain_step_resident(
        host_clip_state,
        host_clip_ads,
        d_a_clip,
        d_b_clip,
        1,
        Float32(1.0e-3),
        Float32(0.9),
        Float32(0.999),
        Float32(1.0e-8),
        Float32(0.01),
        ctx,
    )
    lora_adamw_plain_device_state_sync_moments(host_clip_state, host_clip_ads, ctx)
    lora_adamw_plain_device_state_sync_moments(dev_clip_state, dev_clip_ads, ctx)
    _compare_adapters(host_clip_ads, dev_clip_ads)

    var direct_shared_ads = List[LoraAdapter]()
    var abi_shared_ads = List[LoraAdapter]()
    direct_shared_ads.append(_mk_adapter(2, 3, 4, 5))
    direct_shared_ads.append(_mk_adapter(2, 5, 3, 6))
    for i in range(len(direct_shared_ads)):
        abi_shared_ads.append(direct_shared_ads[i].copy())
    var direct_shared_state = lora_adamw_plain_device_state_init(
        direct_shared_ads, 0, len(direct_shared_ads), ctx
    )
    var abi_shared_state = lora_adamw_plain_device_state_init(
        abi_shared_ads, 0, len(abi_shared_ads), ctx
    )
    for i in range(len(direct_shared_ads)):
        lora_adamw_plain_device_state_copy_device_grad_pair(
            direct_shared_state, i, d_a_t[i], d_b_t[i], ctx
        )
        lora_adamw_plain_device_state_copy_device_grad_pair(
            abi_shared_state, i, d_a_t[i], d_b_t[i], ctx
        )
    var direct_preloaded_norm = fused_lora_adamw_plain_step_resident_preloaded_grads(
        direct_shared_state,
        direct_shared_ads,
        1,
        Float32(1.0e-3),
        Float32(0.9),
        Float32(0.999),
        Float32(1.0e-8),
        Float32(0.01),
        ctx,
        Float32(1.0),
        True,
        Float32(10.0),
    )
    var abi_arena = TrainingArena(ctx, 8192, 1)
    var abi_result = lora_adamw_plain_preloaded_shared_abi_train_step(
        abi_shared_state,
        Float32(0.125),
        1,
        Float32(1.0e-3),
        Float32(0.9),
        Float32(0.999),
        Float32(1.0e-8),
        Float32(0.01),
        abi_arena,
        ctx,
        Float32(10.0),
    )
    abi_result.validate()
    _check(abi_result.is_fast_path(), "shared ABI resident LoRA result should be device fast")
    _check(
        abi_result.optimizer_backend == String("fused_adamw_multitensor-arena-grad-stats-adamw-descriptors"),
        "shared ABI resident LoRA backend label",
    )
    _check(
        _close(abi_result.grad_norm, direct_preloaded_norm, Float32(1.0e-6)),
        "shared ABI resident LoRA grad norm",
    )
    var abi_stats = abi_arena.stats()
    _check(abi_stats.current_used_bytes == 0, "shared ABI arena should rewind optimizer scratch")
    _check(abi_stats.host_device_transfer_count == 9, "shared ABI arena transfer accounting")
    _check(abi_stats.sync_count == 2, "shared ABI arena sync accounting")
    lora_adamw_plain_device_state_sync_params(abi_shared_state, abi_shared_ads, ctx)
    lora_adamw_plain_device_state_sync_moments(direct_shared_state, direct_shared_ads, ctx)
    lora_adamw_plain_device_state_sync_moments(abi_shared_state, abi_shared_ads, ctx)
    _compare_adapters(direct_shared_ads, abi_shared_ads)

    var missing_raised = False
    try:
        var one_idx = List[Int]()
        one_idx.append(0)
        var one_da = List[TArc]()
        var one_db = List[TArc]()
        one_da.append(d_a_t[0].copy())
        one_db.append(d_b_t[0].copy())
        _ = fused_lora_adamw_plain_step_resident_device_grads(
            dev_state,
            dev_ads,
            one_idx,
            one_da,
            one_db,
            2,
            Float32(1.0e-3),
            Float32(0.9),
            Float32(0.999),
            Float32(1.0e-8),
            Float32(0.01),
            ctx,
        )
    except e:
        missing_raised = True
        print("  raised as expected [missing device grad]:", String(e))
    _check(missing_raised, "missing device grad must fail loud")

    var out_of_range_raised = False
    try:
        var bad_idx = List[Int]()
        bad_idx.append(0)
        bad_idx.append(len(dev_ads))
        _ = fused_lora_adamw_plain_step_resident_device_grads(
            dev_state,
            dev_ads,
            bad_idx,
            d_a_t,
            d_b_t,
            2,
            Float32(1.0e-3),
            Float32(0.9),
            Float32(0.999),
            Float32(1.0e-8),
            Float32(0.01),
            ctx,
        )
    except e:
        out_of_range_raised = True
        print("  raised as expected [out-of-range device grad]:", String(e))
    _check(out_of_range_raised, "out-of-range device grad must fail loud")

    var wrong_dtype_raised = False
    try:
        var bad_da = List[TArc]()
        var bad_db = List[TArc]()
        bad_da.append(TArc(Tensor.from_host(d_a[0].copy(), [len(d_a[0])], STDtype.BF16, ctx)))
        bad_db.append(d_b_t[0].copy())
        bad_da.append(d_a_t[1].copy())
        bad_db.append(d_b_t[1].copy())
        _ = fused_lora_adamw_plain_step_resident_device_grads(
            dev_state,
            dev_ads,
            grad_indices,
            bad_da,
            bad_db,
            2,
            Float32(1.0e-3),
            Float32(0.9),
            Float32(0.999),
            Float32(1.0e-8),
            Float32(0.01),
            ctx,
        )
    except e:
        wrong_dtype_raised = True
        print("  raised as expected [wrong dtype device grad]:", String(e))
    _check(wrong_dtype_raised, "wrong dtype device grad must fail loud")

    var nonfinite_raised = False
    var before_bad = List[LoraAdapter]()
    for i in range(len(dev_ads)):
        before_bad.append(dev_ads[i].copy())
    try:
        var bad_vals = d_a[0].copy()
        bad_vals[0] = Float32(0.0) / Float32(0.0)
        var bad_da2 = List[TArc]()
        var bad_db2 = List[TArc]()
        bad_da2.append(TArc(Tensor.from_host(bad_vals^, [len(d_a[0])], STDtype.F32, ctx)))
        bad_db2.append(d_b_t[0].copy())
        bad_da2.append(d_a_t[1].copy())
        bad_db2.append(d_b_t[1].copy())
        _ = fused_lora_adamw_plain_step_resident_device_grads(
            dev_state,
            dev_ads,
            grad_indices,
            bad_da2,
            bad_db2,
            2,
            Float32(1.0e-3),
            Float32(0.9),
            Float32(0.999),
            Float32(1.0e-8),
            Float32(0.01),
            ctx,
        )
    except e:
        nonfinite_raised = True
        print("  raised as expected [nonfinite device grad]:", String(e))
    _check(nonfinite_raised, "nonfinite device grad must fail loud")
    lora_adamw_plain_device_state_sync_params(dev_state, dev_ads, ctx)
    lora_adamw_plain_device_state_sync_moments(dev_state, dev_ads, ctx)
    _compare_adapters(before_bad, dev_ads)

    print("PASS: resident LoRA AdamW device-grad path matches host-grad resident path")
