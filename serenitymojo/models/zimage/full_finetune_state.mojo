# models/zimage/full_finetune_state.mojo -- Z-Image TrainState sidecar binding.
#
# Scope: model-specific optimizer/master sidecar binding only. This does not
# implement a product full-finetune loop or loaded tensor rebind. The model
# tensor payload order remains zimage_full_finetune_checkpoint_key_manifest();
# this wrapper binds that order to training.loop.TrainState's opaque
# param.N/adam_m.N/adam_v.N/__meta__ sidecar keys.
#
# Dtype contract: model payload tensors stay in their storage dtype elsewhere.
# The tensors passed here are optimizer master tensors, so F32 storage is
# intentional and limited to TrainState masters plus Adam moments.

from std.collections import List
from std.gpu.host import DeviceContext

from serenitymojo.models.zimage.full_finetune_inventory import (
    zimage_full_finetune_checkpoint_key_manifest,
    zimage_full_finetune_inventory_expected_count,
)
from serenitymojo.training.full_finetune_contract import (
    FULL_FINETUNE_META_KEY,
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


comptime ZIMAGE_FULL_FINETUNE_STATE_LABEL = "Z-Image full-finetune TrainState"


@fieldwise_init
struct ZImageFullFinetuneTrainStateBinding(Copyable, Movable):
    """One manifest tensor name and its TrainState sidecar keys."""

    var model_name: String
    var param_key: String
    var adam_m_key: String
    var adam_v_key: String


def _zimage_full_finetune_manifest() raises -> List[String]:
    var names = zimage_full_finetune_checkpoint_key_manifest()
    var expected = zimage_full_finetune_inventory_expected_count()
    if len(names) != expected:
        raise Error(
            String(ZIMAGE_FULL_FINETUNE_STATE_LABEL)
            + String(": manifest count ")
            + String(len(names))
            + String(" != inventory expected count ")
            + String(expected)
        )
    return names^


def zimage_full_finetune_train_state_sidecar_prefix_markers() -> List[String]:
    """Return the literal TrainState sidecar prefixes checked by source guards."""

    var out = List[String]()
    out.append(String("param."))
    out.append(String("adam_m."))
    out.append(String("adam_v."))
    return out^


def zimage_full_finetune_train_state_bindings(
) raises -> List[ZImageFullFinetuneTrainStateBinding]:
    """Bind Z-Image full-finetune manifest indices to TrainState keys."""

    var names = _zimage_full_finetune_manifest()
    var out = List[ZImageFullFinetuneTrainStateBinding]()
    for i in range(len(names)):
        out.append(
            ZImageFullFinetuneTrainStateBinding(
                names[i],
                full_finetune_optimizer_param_key(i),
                full_finetune_optimizer_adam_m_key(i),
                full_finetune_optimizer_adam_v_key(i),
            )
        )
    return out^


def zimage_full_finetune_train_state_param_count() raises -> Int:
    return len(_zimage_full_finetune_manifest())


def zimage_full_finetune_train_state_sidecar_names() raises -> List[String]:
    """Return sidecar keys in training.loop.save_checkpoint write order."""

    return full_finetune_train_state_sidecar_names(
        zimage_full_finetune_train_state_param_count()
    )


def validate_zimage_full_finetune_payload_manifest_order(
    payload: List[FullFinetuneTensor],
) raises:
    validate_full_finetune_payload_manifest_order(
        payload,
        _zimage_full_finetune_manifest(),
        String(ZIMAGE_FULL_FINETUNE_STATE_LABEL),
    )


def zimage_full_finetune_train_state_from_payload(
    payload: List[FullFinetuneTensor], ctx: DeviceContext
) raises -> TrainState:
    """Create F32 TrainState masters from Z-Image payload tensors in order."""

    return full_finetune_train_state_from_payload(
        payload,
        _zimage_full_finetune_manifest(),
        String(ZIMAGE_FULL_FINETUNE_STATE_LABEL),
        ctx,
    )


def assert_zimage_full_finetune_train_state_sidecar_binding(
    state: TrainState,
) raises:
    """Fail if a TrainState is not shaped for the Z-Image manifest."""

    assert_full_finetune_train_state_sidecar_binding(
        state,
        _zimage_full_finetune_manifest(),
        String(ZIMAGE_FULL_FINETUNE_STATE_LABEL),
    )


def save_zimage_full_finetune_train_state_sidecar(
    state: TrainState, path: String, ctx: DeviceContext
) raises -> Int:
    """Save the TrainState sidecar after validating Z-Image param count."""

    assert_zimage_full_finetune_train_state_sidecar_binding(state)
    save_checkpoint(state, path, ctx)
    return state.num_params()


def load_zimage_full_finetune_train_state_sidecar(
    path: String, ctx: DeviceContext
) raises -> TrainState:
    """Load the generic TrainState sidecar and verify Z-Image param count."""

    var state = load_checkpoint(path, ctx)
    assert_zimage_full_finetune_train_state_sidecar_binding(state)
    return state^
