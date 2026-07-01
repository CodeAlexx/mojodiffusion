# ZImage B2 masked LoRA block smoke.
#
# Evidence level: bounded equivalence smoke. All-valid masks must match the
# existing no-mask B2 LoRA block path; this is not OneTrainer gradient parity.
#
# Run:
#   pixi run mojo run -I . serenitymojo/models/zimage/parity/zimage_b2_masked_lora_block_smoke.mojo

from std.gpu.host import DeviceContext
from std.math import cos, sin
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.zimage.block import ZImageModVecs
from serenitymojo.models.zimage.lora_block import (
    ZImageBlockLoraDevice,
    zimage_block_lora_backward_device_tensors_batch,
    zimage_block_lora_backward_device_tensors_batch_masked,
    zimage_block_lora_forward_device_tensor_batch,
    zimage_block_lora_forward_device_tensor_batch_masked,
    zimage_lora_adapter_to_device,
    zimage_modvecs_pack2_to_device,
)
from serenitymojo.models.zimage.weights import ZImageBlockWeights
from serenitymojo.models.zimage.zimage_stack_lora import zimage_key_tail_mask_f32
from serenitymojo.parity import ParityHarness
from serenitymojo.tensor import Tensor
from serenitymojo.training.train_step import LoraAdapter


comptime TArc = ArcPointer[Tensor]
comptime B = 2
comptime H = 1
comptime Dh = 4
comptime D = H * Dh
comptime S = 3
comptime F = 6
comptime RANK = 2
comptime EPS = Float32(1.0e-5)


def _fill(n: Int, a: Float32, b: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(sin(Float32(i) * a + b) * Float32(0.05))
    return out^


def _scale(n: Int, a: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(Float32(1.0) + cos(Float32(i) * a) * Float32(0.01))
    return out^


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _i in range(n):
        out.append(Float32(0.0))
    return out^


def _t1(vals: List[Float32], n: Int, ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_host(vals, [n], STDtype.F32, ctx))


def _t2(vals: List[Float32], a: Int, b: Int, ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_host(vals, [a, b], STDtype.F32, ctx))


def _weights(ctx: DeviceContext) raises -> ZImageBlockWeights:
    return ZImageBlockWeights(
        _t1(_scale(D, Float32(0.07)), D, ctx),
        _t2(_fill(D * D, Float32(0.11), Float32(0.1)), D, D, ctx),
        _t2(_fill(D * D, Float32(0.13), Float32(0.2)), D, D, ctx),
        _t2(_fill(D * D, Float32(0.17), Float32(0.3)), D, D, ctx),
        _t2(_fill(D * D, Float32(0.19), Float32(0.4)), D, D, ctx),
        _t1(_scale(Dh, Float32(0.23)), Dh, ctx),
        _t1(_scale(Dh, Float32(0.29)), Dh, ctx),
        _t1(_scale(D, Float32(0.31)), D, ctx),
        _t1(_scale(D, Float32(0.37)), D, ctx),
        _t2(_fill(F * D, Float32(0.41), Float32(0.5)), F, D, ctx),
        _t2(_fill(F * D, Float32(0.43), Float32(0.6)), F, D, ctx),
        _t2(_fill(D * F, Float32(0.47), Float32(0.7)), D, F, ctx),
        _t1(_scale(D, Float32(0.53)), D, ctx),
    )


def _mod(seed: Float32) -> ZImageModVecs:
    return ZImageModVecs(
        _fill(D, Float32(0.05), seed),
        _fill(D, Float32(0.07), seed + Float32(0.1)),
        _fill(D, Float32(0.09), seed + Float32(0.2)),
        _fill(D, Float32(0.11), seed + Float32(0.3)),
    )


def _adapter(in_f: Int, out_f: Int, seed: Float32) -> LoraAdapter:
    return LoraAdapter(
        _fill(RANK * in_f, Float32(0.13), seed),
        _fill(out_f * RANK, Float32(0.17), seed + Float32(0.4)),
        RANK,
        in_f,
        out_f,
        Float32(1.0),
        _zeros(RANK * in_f),
        _zeros(RANK * in_f),
        _zeros(out_f * RANK),
        _zeros(out_f * RANK),
    )


def _lora(ctx: DeviceContext) raises -> ZImageBlockLoraDevice:
    return ZImageBlockLoraDevice(
        zimage_lora_adapter_to_device(_adapter(D, D, Float32(0.1)), ctx),
        zimage_lora_adapter_to_device(_adapter(D, D, Float32(0.2)), ctx),
        zimage_lora_adapter_to_device(_adapter(D, D, Float32(0.3)), ctx),
        zimage_lora_adapter_to_device(_adapter(D, D, Float32(0.4)), ctx),
        zimage_lora_adapter_to_device(_adapter(D, F, Float32(0.5)), ctx),
        zimage_lora_adapter_to_device(_adapter(D, F, Float32(0.6)), ctx),
        zimage_lora_adapter_to_device(_adapter(F, D, Float32(0.7)), ctx),
    )


def _rope(kind: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(B * S * H * (Dh // 2)):
        if kind == 0:
            out.append(cos(Float32(i) * Float32(0.03)))
        else:
            out.append(sin(Float32(i) * Float32(0.03)))
    return out^


def _check(h: ParityHarness, name: String, got: Tensor, expected: Tensor, ctx: DeviceContext) raises:
    var r = h.compare_host(got.to_host(ctx), expected.to_host(ctx))
    print(name, ":", r)
    if not r.passed:
        raise Error(name + " mismatch")


def main() raises:
    var ctx = DeviceContext()
    var h = ParityHarness(0.999999)
    var w = _weights(ctx)
    var mv = zimage_modvecs_pack2_to_device(_mod(Float32(0.1)), _mod(Float32(0.9)), D, ctx)
    var lora = _lora(ctx)
    var x = TArc(Tensor.from_host(
        _fill(B * S * D, Float32(0.03), Float32(0.2)),
        [B * S, D],
        STDtype.F32,
        ctx,
    ))
    var cos_t = Tensor.from_host(_rope(0), [B * S * H, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(_rope(1), [B * S * H, Dh // 2], STDtype.F32, ctx)
    var mask = zimage_key_tail_mask_f32[B, H, S](S, S, ctx)

    var no_mask = zimage_block_lora_forward_device_tensor_batch[B, H, Dh, S](
        x.copy(), w, mv, lora, cos_t, sin_t, D, F, EPS, ctx,
    )
    var masked = zimage_block_lora_forward_device_tensor_batch_masked[B, H, Dh, S](
        x.copy(), w, mv, lora, cos_t, sin_t, mask, D, F, EPS, ctx,
    )
    _check(h, "forward all-valid mask equals no-mask", masked.out[], no_mask.out[], ctx)

    var d_out = Tensor.from_host(
        _fill(B * S * D, Float32(0.05), Float32(0.8)),
        [B * S, D],
        STDtype.F32,
        ctx,
    )
    var g0 = zimage_block_lora_backward_device_tensors_batch[B, H, Dh, S](
        d_out, w, mv, lora, no_mask.saved, cos_t, sin_t, D, F, EPS, ctx,
    )
    var g1 = zimage_block_lora_backward_device_tensors_batch_masked[B, H, Dh, S](
        d_out, w, mv, lora, masked.saved, cos_t, sin_t, mask, D, F, EPS, ctx,
    )
    _check(h, "backward d_x all-valid mask equals no-mask", g1.d_x[], g0.d_x[], ctx)
    for i in range(7):
        _check(h, String("backward d_a slot ") + String(i), g1.d_a[i][], g0.d_a[i][], ctx)
        _check(h, String("backward d_b slot ") + String(i), g1.d_b[i][], g0.d_b[i][], ctx)
    print("PASS: ZImage B2 masked LoRA block all-valid mask matches no-mask")
