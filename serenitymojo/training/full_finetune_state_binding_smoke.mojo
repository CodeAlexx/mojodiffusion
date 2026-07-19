# training/full_finetune_state_binding_smoke.mojo -- bounded TrainState binding smoke.
#
# This proves only the shared full-finetune manifest -> TrainState sidecar order.
# It is not a model product-loop or full-finetune parity gate.

from std.collections import List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.training.full_finetune_save import FullFinetuneTensor
from serenitymojo.training.full_finetune_state_binding import (
    assert_full_finetune_train_state_sidecar_binding,
    full_finetune_train_state_from_payload,
    full_finetune_train_state_sidecar_names,
)


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def _shape1(n: Int) -> List[Int]:
    var out = List[Int]()
    out.append(n)
    return out^


def _bf16_tensor(n: Int, base: Float32, ctx: DeviceContext) raises -> Tensor:
    var values = List[BFloat16]()
    for i in range(n):
        values.append((base + Float32(i) * Float32(0.125)).cast[DType.bfloat16]())
    return Tensor.from_host_bf16(values^, _shape1(n), ctx)


def main() raises:
    var ctx = DeviceContext()

    var names = List[String]()
    names.append(String("block.0.weight"))
    names.append(String("block.1.weight"))
    names.append(String("norm.scale"))

    var payload = List[FullFinetuneTensor]()
    payload.append(
        FullFinetuneTensor(
            names[0],
            ArcPointer[Tensor](_bf16_tensor(4, Float32(1.0), ctx)),
        )
    )
    payload.append(
        FullFinetuneTensor(
            names[1],
            ArcPointer[Tensor](_bf16_tensor(4, Float32(2.0), ctx)),
        )
    )
    payload.append(
        FullFinetuneTensor(
            names[2],
            ArcPointer[Tensor](_bf16_tensor(4, Float32(3.0), ctx)),
        )
    )

    var state = full_finetune_train_state_from_payload(
        payload, names, String("shared smoke"), ctx
    )
    assert_full_finetune_train_state_sidecar_binding(
        state, names, String("shared smoke")
    )

    var sidecars = full_finetune_train_state_sidecar_names(len(names))
    _require(len(sidecars) == 10, String("sidecar count mismatch"))
    _require(sidecars[0] == String("param.0"), String("first param key mismatch"))
    _require(sidecars[3] == String("adam_m.0"), String("first adam_m key mismatch"))
    _require(sidecars[4] == String("adam_v.0"), String("first adam_v key mismatch"))
    _require(sidecars[9] == String("__meta__"), String("meta key mismatch"))
    _require(state.num_params() == 3, String("TrainState param count mismatch"))
    _require(state.masters[0][].dtype() == STDtype.F32, String("master dtype mismatch"))
    _require(state.m[0][].dtype() == STDtype.F32, String("adam_m dtype mismatch"))
    _require(state.v[0][].dtype() == STDtype.F32, String("adam_v dtype mismatch"))
    _require(state.accum[0][].dtype() == STDtype.BF16, String("accum dtype mismatch"))

    var compute = state.compute_weight(0, ctx)
    _require(compute.dtype() == STDtype.BF16, String("compute dtype mismatch"))

    print("full_finetune_state_binding smoke PASS count=", state.num_params())
