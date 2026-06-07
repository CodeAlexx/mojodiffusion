# Klein full-finetune TrainState sidecar smoke.
#
# Synthetic, bounded gate only: builds 201 tiny BF16 model tensors in Klein
# manifest order, creates a generic TrainState through the Klein wrapper, and
# verifies sidecar key order plus F32 optimizer/BF16 compute-storage boundaries.
# This is not a product full-finetune or resume-rebind proof.

from std.collections import List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.klein.full_finetune_inventory import (
    klein_full_finetune_checkpoint_key_manifest,
    klein_full_finetune_inventory_expected_count,
)
from serenitymojo.models.klein.full_finetune_state import (
    assert_klein_full_finetune_train_state_sidecar_binding,
    klein_full_finetune_train_state_from_payload,
    klein_full_finetune_train_state_param_count,
    klein_full_finetune_train_state_sidecar_names,
    klein_full_finetune_train_state_sidecar_slots,
    load_klein_full_finetune_train_state_sidecar,
    save_klein_full_finetune_train_state_sidecar,
)
from serenitymojo.tensor import Tensor
from serenitymojo.training.full_finetune_save import FullFinetuneTensor


comptime TArc = ArcPointer[Tensor]
comptime SIDE_OUT = "/tmp/klein_full_finetune_state_smoke.sidecar.safetensors"


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def _shape1(n: Int) -> List[Int]:
    var shape = List[Int]()
    shape.append(n)
    return shape^


def _tiny_bf16_tensor(idx: Int, ctx: DeviceContext) raises -> Tensor:
    var values = List[BFloat16]()
    values.append((Float32(idx) * Float32(0.01)).cast[DType.bfloat16]())
    return Tensor.from_host_bf16(values^, _shape1(1), ctx)


def _synthetic_payload(ctx: DeviceContext) raises -> List[FullFinetuneTensor]:
    var names = klein_full_finetune_checkpoint_key_manifest()
    var out = List[FullFinetuneTensor]()
    for i in range(len(names)):
        out.append(
            FullFinetuneTensor(
                names[i],
                TArc(_tiny_bf16_tensor(i, ctx)),
            )
        )
    return out^


def _check_sidecar_names(names: List[String], param_count: Int) raises:
    _require(
        len(names) == param_count * 3 + 1,
        String("sidecar key count mismatch"),
    )
    for i in range(param_count):
        _require(
            names[i] == String("param.") + String(i),
            String("param sidecar order mismatch at ") + String(i),
        )
        var moment_offset = param_count + i * 2
        _require(
            names[moment_offset] == String("adam_m.") + String(i),
            String("adam_m sidecar order mismatch at ") + String(i),
        )
        _require(
            names[moment_offset + 1] == String("adam_v.") + String(i),
            String("adam_v sidecar order mismatch at ") + String(i),
        )
    _require(
        names[len(names) - 1] == String("__meta__"),
        String("final sidecar key mismatch"),
    )


def _check_sidecar_slots(param_count: Int) raises:
    var manifest = klein_full_finetune_checkpoint_key_manifest()
    var slots = klein_full_finetune_train_state_sidecar_slots()
    _require(len(slots) == param_count, String("slot count mismatch"))
    for i in range(param_count):
        _require(slots[i].index == i, String("slot index mismatch"))
        _require(
            slots[i].model_tensor_name == manifest[i],
            String("slot model key mismatch at ") + String(i),
        )
        _require(
            slots[i].param_key == String("param.") + String(i),
            String("slot param key mismatch at ") + String(i),
        )
        _require(
            slots[i].adam_m_key == String("adam_m.") + String(i),
            String("slot adam_m key mismatch at ") + String(i),
        )
        _require(
            slots[i].adam_v_key == String("adam_v.") + String(i),
            String("slot adam_v key mismatch at ") + String(i),
        )


def main() raises:
    print("=== Klein full-finetune TrainState sidecar smoke ===")
    var ctx = DeviceContext()

    var expected = klein_full_finetune_inventory_expected_count()
    _require(expected == 201, String("Klein full-finetune expected count changed"))
    _require(
        klein_full_finetune_train_state_param_count() == expected,
        String("Klein TrainState param count wrapper mismatch"),
    )

    _check_sidecar_slots(expected)
    var sidecar_names = klein_full_finetune_train_state_sidecar_names()
    _check_sidecar_names(sidecar_names, expected)

    var payload = _synthetic_payload(ctx)
    _require(len(payload) == expected, String("synthetic payload count mismatch"))
    _require(
        payload[0].tensor[].dtype() == STDtype.BF16,
        String("synthetic payload must be BF16"),
    )

    var state = klein_full_finetune_train_state_from_payload(payload, ctx)
    _require(state.num_params() == 201, String("TrainState param count mismatch"))
    assert_klein_full_finetune_train_state_sidecar_binding(state)

    for i in range(state.num_params()):
        _require(state.masters[i][].dtype() == STDtype.F32, String("master not F32"))
        _require(state.m[i][].dtype() == STDtype.F32, String("adam_m not F32"))
        _require(state.v[i][].dtype() == STDtype.F32, String("adam_v not F32"))
        _require(state.accum[i][].dtype() == STDtype.BF16, String("accum not BF16"))
        var compute = state.compute_weight(i, ctx)
        _require(compute.dtype() == STDtype.BF16, String("compute view not BF16"))

    var saved = save_klein_full_finetune_train_state_sidecar(
        state, String(SIDE_OUT), ctx
    )
    _require(saved == expected, String("saved sidecar param count mismatch"))

    var st = SafeTensors.open(String(SIDE_OUT))
    _require(st.count() == len(sidecar_names), String("saved sidecar key count mismatch"))
    _require(String("param.0") in st.tensors, String("missing param.0"))
    _require(String("adam_m.0") in st.tensors, String("missing adam_m.0"))
    _require(String("adam_v.0") in st.tensors, String("missing adam_v.0"))
    _require(String("__meta__") in st.tensors, String("missing __meta__"))
    _require(st.tensor_info(String("param.0")).dtype == STDtype.F32, String("param.0 dtype"))
    _require(st.tensor_info(String("adam_m.0")).dtype == STDtype.F32, String("adam_m.0 dtype"))
    _require(st.tensor_info(String("adam_v.0")).dtype == STDtype.F32, String("adam_v.0 dtype"))
    _require(st.tensor_info(String("__meta__")).dtype == STDtype.F32, String("__meta__ dtype"))

    var loaded = load_klein_full_finetune_train_state_sidecar(String(SIDE_OUT), ctx)
    _require(loaded.num_params() == expected, String("loaded param count mismatch"))

    print(
        "Klein full-finetune TrainState sidecar smoke PASS params=",
        expected,
        " sidecar_keys=",
        len(sidecar_names),
        " path=",
        SIDE_OUT,
    )
