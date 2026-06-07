# models/chroma/full_finetune_state.mojo -- Chroma TrainState sidecar binding.
#
# Scope: Chroma transformer full-finetune sidecar scaffold only. Text encoder
# and embeddings are excluded. This does not wire a product full-finetune loop,
# runtime model rebind, or OneTrainer parity.

from std.collections import List
from std.gpu.host import DeviceContext

from serenitymojo.models.chroma.full_finetune_inventory import (
    chroma_full_finetune_checkpoint_key_manifest,
    chroma_full_finetune_inventory_expected_count,
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


comptime CHROMA_FULL_FINETUNE_STATE_LABEL = "Chroma full-finetune TrainState"


@fieldwise_init
struct ChromaFullFinetuneTrainStateSidecarSlot(Copyable, Movable):
    var index: Int
    var model_tensor_name: String
    var param_key: String
    var adam_m_key: String
    var adam_v_key: String


def _chroma_full_finetune_manifest() raises -> List[String]:
    var names = chroma_full_finetune_checkpoint_key_manifest()
    var expected = chroma_full_finetune_inventory_expected_count()
    if len(names) != expected:
        raise Error(
            String(CHROMA_FULL_FINETUNE_STATE_LABEL)
            + String(": manifest count ")
            + String(len(names))
            + String(" != inventory expected count ")
            + String(expected)
        )
    return names^


def chroma_full_finetune_train_state_sidecar_prefix_markers() -> List[String]:
    """Return literal sidecar prefixes checked by source guards."""

    var out = List[String]()
    out.append(String("param."))
    out.append(String("adam_m."))
    out.append(String("adam_v."))
    return out^


def chroma_full_finetune_train_state_param_count() raises -> Int:
    return len(_chroma_full_finetune_manifest())


def chroma_full_finetune_train_state_sidecar_slots(
) raises -> List[ChromaFullFinetuneTrainStateSidecarSlot]:
    var names = _chroma_full_finetune_manifest()
    var out = List[ChromaFullFinetuneTrainStateSidecarSlot]()
    for i in range(len(names)):
        out.append(
            ChromaFullFinetuneTrainStateSidecarSlot(
                i,
                names[i],
                full_finetune_optimizer_param_key(i),
                full_finetune_optimizer_adam_m_key(i),
                full_finetune_optimizer_adam_v_key(i),
            )
        )
    return out^


def chroma_full_finetune_train_state_sidecar_names() raises -> List[String]:
    return full_finetune_train_state_sidecar_names(
        chroma_full_finetune_train_state_param_count()
    )


def validate_chroma_full_finetune_payload_manifest_order(
    payload: List[FullFinetuneTensor],
) raises:
    validate_full_finetune_payload_manifest_order(
        payload,
        _chroma_full_finetune_manifest(),
        String(CHROMA_FULL_FINETUNE_STATE_LABEL),
    )


def chroma_full_finetune_train_state_from_payload(
    payload: List[FullFinetuneTensor], ctx: DeviceContext
) raises -> TrainState:
    """Create F32 TrainState masters from Chroma payload tensors in order."""

    return full_finetune_train_state_from_payload(
        payload,
        _chroma_full_finetune_manifest(),
        String(CHROMA_FULL_FINETUNE_STATE_LABEL),
        ctx,
    )


def assert_chroma_full_finetune_train_state_sidecar_binding(
    state: TrainState,
) raises:
    assert_full_finetune_train_state_sidecar_binding(
        state,
        _chroma_full_finetune_manifest(),
        String(CHROMA_FULL_FINETUNE_STATE_LABEL),
    )


def save_chroma_full_finetune_train_state_sidecar(
    state: TrainState, path: String, ctx: DeviceContext
) raises -> Int:
    assert_chroma_full_finetune_train_state_sidecar_binding(state)
    save_checkpoint(state, path, ctx)
    return state.num_params()


def load_chroma_full_finetune_train_state_sidecar(
    path: String, ctx: DeviceContext
) raises -> TrainState:
    var state = load_checkpoint(path, ctx)
    assert_chroma_full_finetune_train_state_sidecar_binding(state)
    return state^
