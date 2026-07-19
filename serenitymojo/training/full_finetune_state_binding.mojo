# training/full_finetune_state_binding.mojo -- full-finetune TrainState binding.
#
# This is shared resume scaffolding only. It binds an ordered model tensor
# manifest to loop.mojo's opaque optimizer sidecar order:
#   param.N, adam_m.N, adam_v.N, __meta__
#
# Dtype contract:
#   Model payload tensors keep their original storage dtype. The only F32
#   boundary introduced here is the optimizer/master sidecar required by
#   TrainState and AdamW resume.

from std.collections import List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.tensor import Tensor
from serenitymojo.training.full_finetune_contract import (
    FULL_FINETUNE_META_KEY,
    full_finetune_optimizer_adam_m_key,
    full_finetune_optimizer_adam_v_key,
    full_finetune_optimizer_param_key,
)
from serenitymojo.training.full_finetune_save import FullFinetuneTensor
from serenitymojo.training.loop import TrainState


comptime TArc = ArcPointer[Tensor]


def _has_name(names: List[String], name: String) -> Bool:
    for i in range(len(names)):
        if names[i] == name:
            return True
    return False


def validate_full_finetune_payload_manifest_order(
    payload: List[FullFinetuneTensor], expected_names: List[String], label: String
) raises:
    if len(payload) == 0:
        raise Error(label + String(": empty full-finetune payload"))
    if len(payload) != len(expected_names):
        raise Error(
            label
            + String(": full-finetune payload count ")
            + String(len(payload))
            + String(" != manifest count ")
            + String(len(expected_names))
        )

    var seen = List[String]()
    for i in range(len(expected_names)):
        var name = expected_names[i]
        if name.byte_length() == 0:
            raise Error(label + String(": empty manifest name"))
        if _has_name(seen, name):
            raise Error(label + String(": duplicate manifest name: ") + name)
        seen.append(name)
        if payload[i].name != name:
            raise Error(
                label
                + String(": payload/manifest order mismatch at index ")
                + String(i)
                + String(": expected ")
                + name
                + String(" got ")
                + payload[i].name
            )


def full_finetune_train_state_sidecar_names(param_count: Int) raises -> List[String]:
    if param_count <= 0:
        raise Error("full-finetune sidecar names require at least one parameter")

    var out = List[String]()
    for i in range(param_count):
        out.append(full_finetune_optimizer_param_key(i))
    for i in range(param_count):
        out.append(full_finetune_optimizer_adam_m_key(i))
        out.append(full_finetune_optimizer_adam_v_key(i))
    out.append(String(FULL_FINETUNE_META_KEY))
    return out^


def full_finetune_train_state_from_payload(
    payload: List[FullFinetuneTensor],
    expected_names: List[String],
    label: String,
    ctx: DeviceContext,
) raises -> TrainState:
    """Create TrainState masters from model tensors in manifest order.

    This does not mutate or rebind the model payload. It creates F32 optimizer
    masters/moments only, which are the explicit resume sidecar format used by
    training/loop.mojo.
    """

    validate_full_finetune_payload_manifest_order(payload, expected_names, label)

    var masters = List[TArc]()
    for i in range(len(payload)):
        masters.append(TArc(cast_tensor(payload[i].tensor[], STDtype.F32, ctx)))
    return TrainState(masters^, ctx)


def assert_full_finetune_train_state_sidecar_binding(
    state: TrainState, expected_names: List[String], label: String
) raises:
    var n = len(expected_names)
    if n <= 0:
        raise Error(label + String(": empty manifest"))
    if state.num_params() != n:
        raise Error(
            label
            + String(": TrainState param count ")
            + String(state.num_params())
            + String(" != manifest count ")
            + String(n)
        )

    var sidecars = full_finetune_train_state_sidecar_names(n)
    if len(sidecars) != (n * 3 + 1):
        raise Error(label + String(": sidecar name count mismatch"))
    if sidecars[0] != String("param.0"):
        raise Error(label + String(": first sidecar key is not param.0"))
    if sidecars[n] != String("adam_m.0"):
        raise Error(label + String(": first Adam m sidecar key is not adam_m.0"))
    if sidecars[n + 1] != String("adam_v.0"):
        raise Error(label + String(": first Adam v sidecar key is not adam_v.0"))
    if sidecars[len(sidecars) - 1] != String(FULL_FINETUNE_META_KEY):
        raise Error(label + String(": final sidecar key is not __meta__"))

    for i in range(n):
        if state.masters[i][].dtype() != STDtype.F32:
            raise Error(label + String(": TrainState master is not F32"))
        if state.m[i][].dtype() != STDtype.F32:
            raise Error(label + String(": Adam m sidecar is not F32"))
        if state.v[i][].dtype() != STDtype.F32:
            raise Error(label + String(": Adam v sidecar is not F32"))
        if state.accum[i][].dtype() != STDtype.BF16:
            raise Error(label + String(": grad accumulator is not BF16"))
