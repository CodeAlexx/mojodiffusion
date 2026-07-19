# zimage_full_finetune_state_smoke.mojo -- Z-Image TrainState sidecar smoke.
#
# Synthetic, bounded gate only: creates one-element F32 optimizer masters in
# Z-Image full-finetune manifest order, constructs TrainState, verifies the
# model-specific sidecar binding/order, checks optimizer dtype boundaries, and
# writes a /tmp state sidecar. This is not full-finetune training support.
#
# Run:
#   pixi run mojo run --target-accelerator sm_86 -I . \
#       serenitymojo/models/zimage/parity/zimage_full_finetune_state_smoke.mojo

from std.collections import List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.zimage.full_finetune_inventory import (
    zimage_full_finetune_checkpoint_key_manifest,
)
from serenitymojo.models.zimage.full_finetune_state import (
    assert_zimage_full_finetune_train_state_sidecar_binding,
    load_zimage_full_finetune_train_state_sidecar,
    save_zimage_full_finetune_train_state_sidecar,
    zimage_full_finetune_train_state_from_payload,
    zimage_full_finetune_train_state_bindings,
    zimage_full_finetune_train_state_param_count,
    zimage_full_finetune_train_state_sidecar_names,
)
from serenitymojo.tensor import Tensor
from serenitymojo.training.full_finetune_save import FullFinetuneTensor
from serenitymojo.training.full_finetune_contract import (
    FULL_FINETUNE_META_KEY,
    full_finetune_optimizer_adam_m_key,
    full_finetune_optimizer_adam_v_key,
    full_finetune_optimizer_param_key,
)


comptime TArc = ArcPointer[Tensor]
comptime STATE_OUT = "/tmp/zimage_full_finetune_state_smoke.safetensors"


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def _shape1(n: Int) -> List[Int]:
    var shape = List[Int]()
    shape.append(n)
    return shape^


def _tiny_bf16_tensor(index: Int, ctx: DeviceContext) raises -> TArc:
    var values = List[BFloat16]()
    values.append((Float32(index) * Float32(0.01)).cast[DType.bfloat16]())
    var tensor = Tensor.from_host_bf16(values^, _shape1(1), ctx)
    return TArc(tensor^)


def _synthetic_payload(ctx: DeviceContext) raises -> List[FullFinetuneTensor]:
    var names = zimage_full_finetune_checkpoint_key_manifest()
    var out = List[FullFinetuneTensor]()
    for i in range(len(names)):
        out.append(FullFinetuneTensor(names[i], _tiny_bf16_tensor(i, ctx)))
    return out^


def _require_key(st: SafeTensors, key: String, dtype: STDtype) raises:
    _require(key in st.tensors, String("missing key ") + key)
    var info = st.tensor_info(key)
    _require(info.dtype == dtype, String("dtype mismatch for ") + key)


def _check_binding_order() raises -> Int:
    var manifest = zimage_full_finetune_checkpoint_key_manifest()
    var expected = len(manifest)
    _require(expected == 521, String("Z-Image manifest expected count changed"))
    _require(
        zimage_full_finetune_train_state_param_count() == expected,
        String("state param count helper mismatch"),
    )

    var bindings = zimage_full_finetune_train_state_bindings()
    _require(len(bindings) == expected, String("binding count mismatch"))
    for i in range(expected):
        _require(
            bindings[i].model_name == manifest[i],
            String("binding manifest name mismatch at index ") + String(i),
        )
        _require(
            bindings[i].param_key == full_finetune_optimizer_param_key(i),
            String("binding param key mismatch at index ") + String(i),
        )
        _require(
            bindings[i].adam_m_key == full_finetune_optimizer_adam_m_key(i),
            String("binding adam_m key mismatch at index ") + String(i),
        )
        _require(
            bindings[i].adam_v_key == full_finetune_optimizer_adam_v_key(i),
            String("binding adam_v key mismatch at index ") + String(i),
        )

    var sidecar_names = zimage_full_finetune_train_state_sidecar_names()
    _require(
        len(sidecar_names) == expected * 3 + 1,
        String("sidecar name count mismatch"),
    )
    for i in range(expected):
        _require(
            sidecar_names[i] == full_finetune_optimizer_param_key(i),
            String("param sidecar order mismatch at index ") + String(i),
        )
    for i in range(expected):
        var j = expected + i * 2
        _require(
            sidecar_names[j] == full_finetune_optimizer_adam_m_key(i),
            String("adam_m sidecar order mismatch at index ") + String(i),
        )
        _require(
            sidecar_names[j + 1] == full_finetune_optimizer_adam_v_key(i),
            String("adam_v sidecar order mismatch at index ") + String(i),
        )
    _require(
        sidecar_names[len(sidecar_names) - 1] == String(FULL_FINETUNE_META_KEY),
        String("sidecar meta key mismatch"),
    )
    return expected


def main() raises:
    var ctx = DeviceContext()
    var expected = _check_binding_order()

    var payload = _synthetic_payload(ctx)
    _require(len(payload) == expected, String("synthetic payload count mismatch"))
    _require(
        payload[0].tensor[].dtype() == STDtype.BF16,
        String("synthetic payload must be BF16"),
    )

    var state = zimage_full_finetune_train_state_from_payload(payload, ctx)
    _require(state.num_params() == 521, String("TrainState param count mismatch"))
    assert_zimage_full_finetune_train_state_sidecar_binding(state)

    for i in range(state.num_params()):
        _require(
            state.masters[i][].dtype() == STDtype.F32,
            String("master dtype mismatch at index ") + String(i),
        )
        _require(
            state.m[i][].dtype() == STDtype.F32,
            String("adam_m dtype mismatch at index ") + String(i),
        )
        _require(
            state.v[i][].dtype() == STDtype.F32,
            String("adam_v dtype mismatch at index ") + String(i),
        )
        _require(
            state.accum[i][].dtype() == STDtype.BF16,
            String("accum dtype mismatch at index ") + String(i),
        )
        var compute = state.compute_weight(i, ctx)
        _require(
            compute.dtype() == STDtype.BF16,
            String("compute view dtype mismatch at index ") + String(i),
        )

    var saved = save_zimage_full_finetune_train_state_sidecar(
        state, String(STATE_OUT), ctx
    )
    _require(saved == expected, String("state sidecar saved count mismatch"))

    var st = SafeTensors.open(String(STATE_OUT))
    var sidecar_names = zimage_full_finetune_train_state_sidecar_names()
    _require(st.count() == len(sidecar_names), String("saved sidecar tensor count mismatch"))
    _require_key(st, String("param.0"), STDtype.F32)
    _require_key(st, String("param.520"), STDtype.F32)
    _require_key(st, String("adam_m.0"), STDtype.F32)
    _require_key(st, String("adam_v.520"), STDtype.F32)
    _require_key(st, String(FULL_FINETUNE_META_KEY), STDtype.F32)
    for i in range(len(sidecar_names)):
        _require(sidecar_names[i] in st.tensors, String("saved sidecar missing ") + sidecar_names[i])

    var loaded = load_zimage_full_finetune_train_state_sidecar(String(STATE_OUT), ctx)
    _require(loaded.num_params() == expected, String("loaded param count mismatch"))

    print("Z-Image full-finetune TrainState sidecar smoke PASS count=", expected)
