# krea2_live_devp_view_smoke.mojo — Krea2 resident dev_p LoRA view smoke.
#
# Run:
#   pixi run mojo run -I . serenitymojo/models/krea2/parity/krea2_live_devp_view_smoke.mojo

from std.gpu.host import DeviceContext
from std.builtin.dtype import DType
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.lora_adamw_plain_fused import (
    fused_lora_adamw_plain_step_resident_preloaded_grads,
    lora_adamw_plain_device_state_copy_device_grad_pair,
    lora_adamw_plain_device_state_init,
    lora_adamw_plain_device_state_sync_params,
)
from serenitymojo.models.krea2.krea2_stack import KREA2_SLOTS_PER_BLOCK
from serenitymojo.models.krea2.train_krea2 import _host_to_device_lora_resident


comptime TArc = ArcPointer[Tensor]
comptime N_BLOCKS = 28
comptime RANK = 1


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("krea2_live_devp_view_smoke FAILED: ") + msg)


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _vals(n: Int, scale: Float32, offset: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(offset + Float32(i + 1) * scale)
    return out^


def _mk_adapter(seed: Int) -> LoraAdapter:
    var in_f = 3 + (seed % 3)
    var out_f = 2 + (seed % 4)
    return LoraAdapter(
        _vals(RANK * in_f, Float32(0.01), Float32(seed) * Float32(0.001)),
        _zeros(out_f * RANK),
        RANK,
        in_f,
        out_f,
        Float32(1.0),
        _zeros(RANK * in_f),
        _zeros(RANK * in_f),
        _zeros(out_f * RANK),
        _zeros(out_f * RANK),
    )


def _bf16_absum(v: List[BFloat16]) -> Float32:
    var out = Float32(0.0)
    for i in range(len(v)):
        var x = v[i].cast[DType.float32]()
        out += x if x >= Float32(0.0) else -x
    return out


def _f32_absum(v: List[Float32]) -> Float32:
    var out = Float32(0.0)
    for i in range(len(v)):
        var x = v[i]
        out += x if x >= Float32(0.0) else -x
    return out


def main() raises:
    _check(KREA2_SLOTS_PER_BLOCK == 8, "Krea2 slot count changed")
    var ctx = DeviceContext()
    var host_ads = List[LoraAdapter]()
    for i in range(N_BLOCKS * KREA2_SLOTS_PER_BLOCK):
        host_ads.append(_mk_adapter(i + 1))

    var state = lora_adamw_plain_device_state_init(
        host_ads, 0, len(host_ads), ctx
    )
    var dev_lora = _host_to_device_lora_resident(host_ads, state)

    var b0_before = dev_lora.blocks[0].wq.value().b[].to_host_bf16(ctx)
    _check(_bf16_absum(b0_before) == Float32(0.0), "initial resident B view should be zero")

    for i in range(len(host_ads)):
        var ga = _vals(
            len(host_ads[i].a),
            Float32(0.002),
            Float32(i + 1) * Float32(0.01),
        )
        var gb = _vals(
            len(host_ads[i].b),
            Float32(0.003),
            Float32(i + 1) * Float32(0.02),
        )
        var ta = TArc(Tensor.from_host(ga^, [len(host_ads[i].a)], STDtype.F32, ctx))
        var tb = TArc(Tensor.from_host(gb^, [len(host_ads[i].b)], STDtype.F32, ctx))
        lora_adamw_plain_device_state_copy_device_grad_pair(
            state, i, ta, tb, ctx
        )

    var norm = fused_lora_adamw_plain_step_resident_preloaded_grads(
        state,
        host_ads,
        1,
        Float32(1.0e-2),
        Float32(0.9),
        Float32(0.999),
        Float32(1.0e-8),
        Float32(0.01),
        ctx,
        Float32(1.0),
        False,
        Float32(10.0),
    )
    _check(norm > Float32(0.0), "device norm should be positive")

    var b0_live = dev_lora.blocks[0].wq.value().b[].to_host_bf16(ctx)
    _check(_bf16_absum(b0_live) > Float32(0.0), "resident B view should see dev_p update")
    _check(_bf16_absum(host_ads[0].b) == Float32(0.0), "host mirror should not update before sync")

    lora_adamw_plain_device_state_sync_params(state, host_ads, ctx)
    _check(_bf16_absum(host_ads[0].b) > Float32(0.0), "host mirror should update after explicit sync")
    _check(_f32_absum(dev_lora.blocks[0].wq.value().b[].to_host(ctx)) > Float32(0.0), "F32 inspection of live view should be nonzero")
    print("PASS: Krea2 resident dev_p LoRA views update without per-step host sync")
