from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.krea2_dit import Krea2TextFusionWeights
from serenitymojo.models.krea2.krea2_block import Krea2BlockLora
from serenitymojo.models.krea2.krea2_text_fusion_lora import (
    Krea2TextFusionLora,
    krea2_text_fusion_lora_forward,
    krea2_text_fusion_lora_backward_dev,
    krea2_text_fusion_grads_to_adamw_state,
)
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.lora_adamw_plain_fused import lora_adamw_plain_device_state_init
from serenitymojo.models.klein.lora_adapter import make_lora_adapter
from serenitymojo.models.klein.lora_block import lora_adapter_to_device, LoraAdapterDevice

comptime TArc = ArcPointer[Tensor]


def _vals(n: Int, scale: Float32) -> List[Float32]:
    var xs = List[Float32]()
    for i in range(n):
        xs.append((Float32((i % 17) - 8)) * scale)
    return xs^


def _tensor(n: Int, var shape: List[Int], dtype: STDtype, scale: Float32, ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(_vals(n, scale), shape^, dtype, ctx)


def _weights(ctx: DeviceContext) raises -> Krea2TextFusionWeights:
    var f = 4
    var m = 8
    return Krea2TextFusionWeights(
        TArc(_tensor(f, [f], STDtype.F32, Float32(0.0), ctx)),
        TArc(_tensor(f, [f], STDtype.F32, Float32(0.0), ctx)),
        TArc(_tensor(f * f, [f, f], STDtype.BF16, Float32(0.01), ctx)),
        TArc(_tensor(f * f, [f, f], STDtype.BF16, Float32(0.012), ctx)),
        TArc(_tensor(f * f, [f, f], STDtype.BF16, Float32(0.014), ctx)),
        TArc(_tensor(f * f, [f, f], STDtype.BF16, Float32(0.016), ctx)),
        TArc(_tensor(f * f, [f, f], STDtype.BF16, Float32(0.018), ctx)),
        TArc(_tensor(f, [f], STDtype.F32, Float32(0.0), ctx)),
        TArc(_tensor(f, [f], STDtype.F32, Float32(0.0), ctx)),
        TArc(_tensor(m * f, [m, f], STDtype.BF16, Float32(0.01), ctx)),
        TArc(_tensor(m * f, [m, f], STDtype.BF16, Float32(0.012), ctx)),
        TArc(_tensor(f * m, [f, m], STDtype.BF16, Float32(0.014), ctx)),
    )


def _empty_lora() -> Krea2BlockLora:
    return Krea2BlockLora(None, None, None, None, None, None, None, None)


def _dev_lora(rank: Int, in_f: Int, out_f: Int, seed: UInt64, ctx: DeviceContext) raises -> LoraAdapterDevice:
    return lora_adapter_to_device(make_lora_adapter(rank, Float32(rank), in_f, out_f, seed), ctx)


def _full_block_lora(seed: UInt64, ctx: DeviceContext) raises -> Krea2BlockLora:
    return Krea2BlockLora(
        Optional[LoraAdapterDevice](_dev_lora(2, 4, 4, seed + 0, ctx)),
        Optional[LoraAdapterDevice](_dev_lora(2, 4, 4, seed + 1, ctx)),
        Optional[LoraAdapterDevice](_dev_lora(2, 4, 4, seed + 2, ctx)),
        Optional[LoraAdapterDevice](_dev_lora(2, 4, 4, seed + 3, ctx)),
        Optional[LoraAdapterDevice](_dev_lora(2, 4, 4, seed + 4, ctx)),
        Optional[LoraAdapterDevice](_dev_lora(2, 4, 8, seed + 5, ctx)),
        Optional[LoraAdapterDevice](_dev_lora(2, 4, 8, seed + 6, ctx)),
        Optional[LoraAdapterDevice](_dev_lora(2, 8, 4, seed + 7, ctx)),
    )


def _append_full_block_host(mut ads: List[LoraAdapter], seed: UInt64):
    ads.append(make_lora_adapter(2, Float32(2.0), 4, 4, seed + 0))
    ads.append(make_lora_adapter(2, Float32(2.0), 4, 4, seed + 1))
    ads.append(make_lora_adapter(2, Float32(2.0), 4, 4, seed + 2))
    ads.append(make_lora_adapter(2, Float32(2.0), 4, 4, seed + 3))
    ads.append(make_lora_adapter(2, Float32(2.0), 4, 4, seed + 4))
    ads.append(make_lora_adapter(2, Float32(2.0), 4, 8, seed + 5))
    ads.append(make_lora_adapter(2, Float32(2.0), 4, 8, seed + 6))
    ads.append(make_lora_adapter(2, Float32(2.0), 8, 4, seed + 7))


def main() raises:
    var ctx = DeviceContext()
    comptime LT = 2
    comptime NLAYERS = 3
    comptime HEADS = 1
    comptime HEADDIM = 4

    var shape = List[Int]()
    shape.append(1); shape.append(LT); shape.append(NLAYERS); shape.append(HEADS * HEADDIM)
    var context = _tensor(1 * LT * NLAYERS * HEADS * HEADDIM, shape^, STDtype.BF16, Float32(0.02), ctx)

    var proj_shape = List[Int]()
    proj_shape.append(1); proj_shape.append(NLAYERS)
    var projector = _tensor(NLAYERS, proj_shape^, STDtype.BF16, Float32(0.05), ctx)

    var tf_lora = Krea2TextFusionLora(
        _full_block_lora(UInt64(1000), ctx),
        _full_block_lora(UInt64(2000), ctx),
        _full_block_lora(UInt64(3000), ctx),
        _full_block_lora(UInt64(4000), ctx),
    )
    var fwd = krea2_text_fusion_lora_forward[LT, NLAYERS, HEADS, HEADDIM](
        context,
        _weights(ctx), _weights(ctx), projector,
        _weights(ctx), _weights(ctx),
        tf_lora, None, ctx,
    )
    ctx.synchronize()
    if fwd.out[].dtype() != STDtype.BF16:
        raise Error("krea2_text_fusion_lora_smoke: output boundary is not BF16")
    if fwd.out[].shape()[0] != 1 or fwd.out[].shape()[1] != LT or fwd.out[].shape()[2] != HEADS * HEADDIM:
        raise Error("krea2_text_fusion_lora_smoke: bad output shape")

    var d_shape = List[Int]()
    d_shape.append(1); d_shape.append(LT); d_shape.append(HEADS * HEADDIM)
    var d_out = _tensor(1 * LT * HEADS * HEADDIM, d_shape^, STDtype.BF16, Float32(0.01), ctx)
    var bwd = krea2_text_fusion_lora_backward_dev[LT, NLAYERS, HEADS, HEADDIM](
        d_out, fwd,
        _weights(ctx), _weights(ctx), projector,
        _weights(ctx), _weights(ctx),
        tf_lora, None, ctx,
    )
    ctx.synchronize()
    if bwd.d_context[].dtype() != STDtype.BF16:
        raise Error("krea2_text_fusion_lora_smoke: d_context boundary is not BF16")
    if not bwd.layerwise0.wq.d_a or not bwd.layerwise0.wq.d_b:
        raise Error("krea2_text_fusion_lora_smoke: missing layerwise0 wq device grads")

    var mask_shape = List[Int]()
    mask_shape.append(1); mask_shape.append(HEADS); mask_shape.append(LT); mask_shape.append(LT)
    var mask = _tensor(1 * HEADS * LT * LT, mask_shape^, STDtype.BF16, Float32(0.0), ctx)
    var fwd_masked = krea2_text_fusion_lora_forward[LT, NLAYERS, HEADS, HEADDIM](
        context,
        _weights(ctx), _weights(ctx), projector,
        _weights(ctx), _weights(ctx),
        tf_lora, Optional[Tensor](mask.clone(ctx)), ctx,
    )
    var bwd_masked = krea2_text_fusion_lora_backward_dev[LT, NLAYERS, HEADS, HEADDIM](
        d_out, fwd_masked,
        _weights(ctx), _weights(ctx), projector,
        _weights(ctx), _weights(ctx),
        tf_lora, Optional[Tensor](mask.clone(ctx)), ctx,
    )
    ctx.synchronize()
    if bwd_masked.d_context[].dtype() != STDtype.BF16:
        raise Error("krea2_text_fusion_lora_smoke: masked d_context boundary is not BF16")

    var ads = List[LoraAdapter]()
    for i in range(28):
        _append_full_block_host(ads, UInt64(10000 + i * 100))
    _append_full_block_host(ads, UInt64(1000))
    _append_full_block_host(ads, UInt64(2000))
    _append_full_block_host(ads, UInt64(3000))
    _append_full_block_host(ads, UInt64(4000))
    var state = lora_adamw_plain_device_state_init(ads, 0, len(ads), ctx)
    var copied = krea2_text_fusion_grads_to_adamw_state(bwd, 224, state, ctx)
    ctx.synchronize()
    if copied.grad_count != 32:
        raise Error("krea2_text_fusion_lora_smoke: txtfusion grad copy count != 32")
    _ = len(copied.grads)
    print("PASS: Krea2 txtfusion LoRA forward/backward/masked/device-grad-copy smoke BF16 boundary base=224")
