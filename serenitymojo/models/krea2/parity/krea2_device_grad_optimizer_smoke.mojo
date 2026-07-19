# krea2_device_grad_optimizer_smoke.mojo — Krea2 flat slot device-grad optimizer smoke.
#
# Run:
#   pixi run mojo run -I . serenitymojo/models/krea2/parity/krea2_device_grad_optimizer_smoke.mojo

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.krea2.krea2_block import Krea2LoraGradT
from serenitymojo.models.krea2.krea2_stack import KREA2_SLOTS_PER_BLOCK
from serenitymojo.training.lora_adamw_plain_fused import (
    fused_lora_adamw_plain_step_resident,
    fused_lora_adamw_plain_step_resident_device_grads,
    fused_lora_adamw_plain_step_resident_preloaded_grads,
    lora_adamw_plain_device_state_copy_device_grad_pair,
    lora_adamw_plain_device_state_init,
    lora_adamw_plain_device_state_sync_moments,
)


comptime TArc = ArcPointer[Tensor]
comptime N_BLOCKS = 2
comptime FEATURES = 6
comptime QDIM = 4
comptime KVDIM = 3
comptime MLPDIM = 8
comptime RANK = 2


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("krea2_device_grad_optimizer_smoke FAILED: ") + msg)


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


def _mk_adapter(in_f: Int, out_f: Int, seed: Int) -> LoraAdapter:
    return LoraAdapter(
        _vals(RANK * in_f, Float32(0.001), Float32(seed) * Float32(0.01)),
        _zeros(out_f * RANK),
        RANK,
        in_f,
        out_f,
        Float32(1.0) / Float32(RANK),
        _zeros(RANK * in_f),
        _zeros(RANK * in_f),
        _zeros(out_f * RANK),
        _zeros(out_f * RANK),
    )


def _append_krea2_block_slots(mut out: List[LoraAdapter], block: Int):
    var seed = block * KREA2_SLOTS_PER_BLOCK + 1
    out.append(_mk_adapter(FEATURES, QDIM, seed + 0))       # wq
    out.append(_mk_adapter(FEATURES, KVDIM, seed + 1))      # wk
    out.append(_mk_adapter(FEATURES, KVDIM, seed + 2))      # wv
    out.append(_mk_adapter(FEATURES, FEATURES, seed + 3))   # gate
    out.append(_mk_adapter(FEATURES, FEATURES, seed + 4))   # wo
    out.append(_mk_adapter(FEATURES, MLPDIM, seed + 5))     # mlp_gate
    out.append(_mk_adapter(FEATURES, MLPDIM, seed + 6))     # mlp_up
    out.append(_mk_adapter(MLPDIM, FEATURES, seed + 7))     # mlp_down


def _absum_bf16(v: List[BFloat16]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(v)):
        var x = v[i].cast[DType.float32]()
        s += x if x >= Float32(0.0) else -x
    return s


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


def main() raises:
    _check(KREA2_SLOTS_PER_BLOCK == 8, "Krea2 slot count changed")
    var ctx = DeviceContext()

    var host_ads = List[LoraAdapter]()
    var dev_ads = List[LoraAdapter]()
    var preloaded_ads = List[LoraAdapter]()
    for bi in range(N_BLOCKS):
        _append_krea2_block_slots(host_ads, bi)
    for i in range(len(host_ads)):
        dev_ads.append(host_ads[i].copy())
        preloaded_ads.append(host_ads[i].copy())
    _check(len(host_ads) == N_BLOCKS * KREA2_SLOTS_PER_BLOCK, "flat adapter count")

    var d_a = List[List[Float32]]()
    var d_b = List[List[Float32]]()
    var grad_indices = List[Int]()
    var d_a_t = List[TArc]()
    var d_b_t = List[TArc]()
    var krea2_carrier = List[Krea2LoraGradT]()
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
        d_a.append(ga.copy())
        d_b.append(gb.copy())
        var ta = TArc(Tensor.from_host(ga^, [len(host_ads[i].a)], STDtype.F32, ctx))
        var tb = TArc(Tensor.from_host(gb^, [len(host_ads[i].b)], STDtype.F32, ctx))
        krea2_carrier.append(Krea2LoraGradT(Optional[TArc](ta.copy()), Optional[TArc](tb.copy())))

    for i in range(len(krea2_carrier)):
        var g = krea2_carrier[i].copy()
        if not g.d_a:
            raise Error("krea2_device_grad_optimizer_smoke FAILED: missing dA carrier")
        if not g.d_b:
            raise Error("krea2_device_grad_optimizer_smoke FAILED: missing dB carrier")
        grad_indices.append(i)
        d_a_t.append(g.d_a.value().copy())
        d_b_t.append(g.d_b.value().copy())

    var host_state = lora_adamw_plain_device_state_init(
        host_ads, 0, len(host_ads), ctx
    )
    var dev_state = lora_adamw_plain_device_state_init(
        dev_ads, 0, len(dev_ads), ctx
    )
    var preloaded_state = lora_adamw_plain_device_state_init(
        preloaded_ads, 0, len(preloaded_ads), ctx
    )
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
    for i in range(len(krea2_carrier)):
        var g = krea2_carrier[i].copy()
        if not g.d_a or not g.d_b:
            raise Error("krea2_device_grad_optimizer_smoke FAILED: missing preloaded grad carrier")
        lora_adamw_plain_device_state_copy_device_grad_pair(
            preloaded_state, i, g.d_a.value(), g.d_b.value(), ctx
        )
    var preloaded_norm = fused_lora_adamw_plain_step_resident_preloaded_grads(
        preloaded_state,
        preloaded_ads,
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
    _check(preloaded_norm > Float32(0.0), "preloaded device grad norm should be positive")
    lora_adamw_plain_device_state_sync_moments(host_state, host_ads, ctx)
    lora_adamw_plain_device_state_sync_moments(dev_state, dev_ads, ctx)
    lora_adamw_plain_device_state_sync_moments(
        preloaded_state, preloaded_ads, ctx
    )
    _compare_adapters(host_ads, dev_ads)
    _compare_adapters(host_ads, preloaded_ads)

    var b_absum = Float32(0.0)
    for i in range(len(dev_ads)):
        b_absum += _absum_bf16(dev_ads[i].b)
    _check(b_absum > Float32(0.0), "Krea2 slot B tensors should move")
    print("PASS: Krea2 flat device-grad slots feed shared resident AdamW")
