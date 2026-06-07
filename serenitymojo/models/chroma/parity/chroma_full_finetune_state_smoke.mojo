# Chroma full-finetune TrainState sidecar smoke.
#
# Synthetic, bounded gate only: creates tiny BF16 model payload tensors in the
# 1023-key Chroma transformer manifest order, then verifies TrainState
# optimizer sidecar names and dtypes. Not product full-finetune parity.

from std.collections import List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.chroma.full_finetune_inventory import (
    chroma_full_finetune_checkpoint_key_manifest,
    chroma_full_finetune_inventory_expected_count,
)
from serenitymojo.models.chroma.full_finetune_state import (
    assert_chroma_full_finetune_train_state_sidecar_binding,
    chroma_full_finetune_train_state_from_payload,
    chroma_full_finetune_train_state_param_count,
    chroma_full_finetune_train_state_sidecar_names,
    chroma_full_finetune_train_state_sidecar_slots,
    load_chroma_full_finetune_train_state_sidecar,
    save_chroma_full_finetune_train_state_sidecar,
)
from serenitymojo.tensor import Tensor
from serenitymojo.training.full_finetune_save import FullFinetuneTensor


comptime TArc = ArcPointer[Tensor]
comptime SIDE_OUT = "/tmp/chroma_full_finetune_state_smoke.sidecar.safetensors"


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
    var names = chroma_full_finetune_checkpoint_key_manifest()
    var out = List[FullFinetuneTensor]()
    for i in range(len(names)):
        out.append(FullFinetuneTensor(names[i], TArc(_tiny_bf16_tensor(i, ctx))))
    return out^


def _check_sidecar_names(names: List[String], param_count: Int) raises:
    _require(len(names) == param_count * 3 + 1, String("sidecar count mismatch"))
    for i in range(param_count):
        _require(names[i] == String("param.") + String(i), String("param key mismatch"))
        var j = param_count + i * 2
        _require(names[j] == String("adam_m.") + String(i), String("adam_m key mismatch"))
        _require(names[j + 1] == String("adam_v.") + String(i), String("adam_v key mismatch"))
    _require(names[len(names) - 1] == String("__meta__"), String("meta key mismatch"))


def main() raises:
    print("=== Chroma full-finetune TrainState sidecar smoke ===")
    var ctx = DeviceContext()
    var expected = chroma_full_finetune_inventory_expected_count()
    _require(expected == 1023, String("Chroma full-finetune expected count changed"))
    _require(
        chroma_full_finetune_train_state_param_count() == expected,
        String("param count wrapper mismatch"),
    )

    var slots = chroma_full_finetune_train_state_sidecar_slots()
    _require(len(slots) == expected, String("slot count mismatch"))
    _require(slots[0].model_tensor_name == String("x_embedder.weight"), String("first slot mismatch"))
    _require(
        slots[len(slots) - 1].model_tensor_name
        == String("single_transformer_blocks.37.attn.norm_k.weight"),
        String("last slot mismatch"),
    )

    var sidecar_names = chroma_full_finetune_train_state_sidecar_names()
    _check_sidecar_names(sidecar_names, expected)

    var payload = _synthetic_payload(ctx)
    _require(len(payload) == expected, String("payload count mismatch"))
    _require(payload[0].tensor[].dtype() == STDtype.BF16, String("payload dtype mismatch"))

    var state = chroma_full_finetune_train_state_from_payload(payload, ctx)
    _require(state.num_params() == expected, String("TrainState param count mismatch"))
    assert_chroma_full_finetune_train_state_sidecar_binding(state)

    for i in range(state.num_params()):
        _require(state.masters[i][].dtype() == STDtype.F32, String("master not F32"))
        _require(state.m[i][].dtype() == STDtype.F32, String("adam_m not F32"))
        _require(state.v[i][].dtype() == STDtype.F32, String("adam_v not F32"))
        _require(state.accum[i][].dtype() == STDtype.BF16, String("accum not BF16"))
        var compute = state.compute_weight(i, ctx)
        _require(compute.dtype() == STDtype.BF16, String("compute view not BF16"))

    var saved = save_chroma_full_finetune_train_state_sidecar(
        state, String(SIDE_OUT), ctx
    )
    _require(saved == expected, String("saved sidecar count mismatch"))
    var st = SafeTensors.open(String(SIDE_OUT))
    _require(st.count() == len(sidecar_names), String("saved sidecar tensor count mismatch"))
    _require(st.tensor_info(String("param.0")).dtype == STDtype.F32, String("param dtype"))
    _require(st.tensor_info(String("adam_m.0")).dtype == STDtype.F32, String("adam_m dtype"))
    _require(st.tensor_info(String("adam_v.0")).dtype == STDtype.F32, String("adam_v dtype"))
    _require(st.tensor_info(String("__meta__")).dtype == STDtype.F32, String("meta dtype"))

    var loaded = load_chroma_full_finetune_train_state_sidecar(String(SIDE_OUT), ctx)
    _require(loaded.num_params() == expected, String("loaded param count mismatch"))

    print(
        "Chroma full-finetune TrainState sidecar smoke PASS params=",
        expected,
        " sidecar_keys=",
        len(sidecar_names),
    )
