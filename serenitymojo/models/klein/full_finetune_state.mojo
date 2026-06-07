# Klein full-finetune TrainState sidecar binding.
#
# Scope: model-specific wrapper only. This binds Klein's full-weight manifest
# order to training/loop.mojo's opaque optimizer sidecar keys:
#   param.N, adam_m.N, adam_v.N, __meta__
# It does not wire a product full-finetune loop or model-struct resume rebind.

from std.collections import List
from std.gpu.host import DeviceContext

from serenitymojo.models.klein.full_finetune_inventory import (
    klein_full_finetune_checkpoint_key_manifest,
    klein_full_finetune_inventory_expected_count,
)
from serenitymojo.training.full_finetune_contract import (
    full_finetune_optimizer_adam_m_key,
    full_finetune_optimizer_adam_v_key,
    full_finetune_optimizer_param_key,
)
from serenitymojo.training.full_finetune_save import FullFinetuneTensor
from serenitymojo.training.full_finetune_state_binding import (
    assert_full_finetune_train_state_sidecar_binding,
    full_finetune_train_state_from_payload,
    full_finetune_train_state_sidecar_names,
    validate_full_finetune_payload_manifest_order,
)
from serenitymojo.training.loop import TrainState, load_checkpoint, save_checkpoint


comptime KLEIN_FULL_FINETUNE_STATE_LABEL = "Klein full-finetune TrainState"


@fieldwise_init
struct KleinFullFinetuneTrainStateSidecarSlot(Copyable, Movable):
    var index: Int
    var model_tensor_name: String
    var param_key: String
    var adam_m_key: String
    var adam_v_key: String


def _klein_full_finetune_manifest() raises -> List[String]:
    var names = klein_full_finetune_checkpoint_key_manifest()
    var expected = klein_full_finetune_inventory_expected_count()
    if len(names) != expected:
        raise Error(
            String(KLEIN_FULL_FINETUNE_STATE_LABEL)
            + String(": manifest count ")
            + String(len(names))
            + String(" != inventory expected count ")
            + String(expected)
        )
    return names^


def klein_full_finetune_train_state_param_count() raises -> Int:
    return len(_klein_full_finetune_manifest())


def klein_full_finetune_train_state_sidecar_prefix_markers() -> List[String]:
    """Return the literal TrainState sidecar prefixes checked by source guards."""

    var out = List[String]()
    out.append(String("param."))
    out.append(String("adam_m."))
    out.append(String("adam_v."))
    return out^


def klein_full_finetune_train_state_sidecar_slots(
) raises -> List[KleinFullFinetuneTrainStateSidecarSlot]:
    """Return the per-parameter binding from Klein keys to TrainState sidecars."""

    var names = _klein_full_finetune_manifest()
    var out = List[KleinFullFinetuneTrainStateSidecarSlot]()
    for i in range(len(names)):
        out.append(
            KleinFullFinetuneTrainStateSidecarSlot(
                i,
                names[i],
                full_finetune_optimizer_param_key(i),
                full_finetune_optimizer_adam_m_key(i),
                full_finetune_optimizer_adam_v_key(i),
            )
        )
    return out^


def klein_full_finetune_train_state_sidecar_names() raises -> List[String]:
    """Return TrainState checkpoint keys in loop.mojo save order."""

    return full_finetune_train_state_sidecar_names(
        klein_full_finetune_train_state_param_count()
    )


def validate_klein_full_finetune_payload_manifest_order(
    payload: List[FullFinetuneTensor],
) raises:
    validate_full_finetune_payload_manifest_order(
        payload,
        _klein_full_finetune_manifest(),
        String(KLEIN_FULL_FINETUNE_STATE_LABEL),
    )


def klein_full_finetune_train_state_from_payload(
    payload: List[FullFinetuneTensor], ctx: DeviceContext
) raises -> TrainState:
    """Create F32 TrainState masters from Klein payload tensors in manifest order.

    Model payload tensors keep their original storage dtype. The F32 tensors
    created here are optimizer masters/moments, which are the explicit sidecar
    boundary used by the shared TrainState checkpoint path.
    """

    return full_finetune_train_state_from_payload(
        payload,
        _klein_full_finetune_manifest(),
        String(KLEIN_FULL_FINETUNE_STATE_LABEL),
        ctx,
    )


def assert_klein_full_finetune_train_state_sidecar_binding(
    state: TrainState,
) raises:
    assert_full_finetune_train_state_sidecar_binding(
        state,
        _klein_full_finetune_manifest(),
        String(KLEIN_FULL_FINETUNE_STATE_LABEL),
    )


def save_klein_full_finetune_train_state_sidecar(
    state: TrainState, path: String, ctx: DeviceContext
) raises -> Int:
    """Save the Klein TrainState sidecar after verifying manifest binding."""

    assert_klein_full_finetune_train_state_sidecar_binding(state)
    save_checkpoint(state, path, ctx)
    return state.num_params()


def load_klein_full_finetune_train_state_sidecar(
    path: String, ctx: DeviceContext
) raises -> TrainState:
    """Load the generic TrainState sidecar and verify Klein param count."""

    var state = load_checkpoint(path, ctx)
    assert_klein_full_finetune_train_state_sidecar_binding(state)
    return state^
