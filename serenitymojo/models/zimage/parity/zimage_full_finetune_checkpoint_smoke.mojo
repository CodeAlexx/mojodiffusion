# zimage_full_finetune_checkpoint_smoke.mojo -- Z-Image full-finetune surface smoke.
#
# Synthetic structural gate only: builds tiny BF16 tensors in the live
# ZImageRealAux/ZImageBlockWeights carriers, verifies collector order/count, and
# round-trips the model payload plus manifest through the shared full-finetune
# save/load scaffold. This does not claim full-finetune training support.
#
# Run:
#   pixi run mojo run --target-accelerator sm_86 -I . \
#       serenitymojo/models/zimage/parity/zimage_full_finetune_checkpoint_smoke.mojo

from std.collections import List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.zimage.full_finetune_checkpoint import (
    load_zimage_full_finetune_payload_only,
    save_zimage_full_finetune_checkpoint,
    zimage_full_finetune_model_tensors,
)
from serenitymojo.models.zimage.full_finetune_inventory import (
    ZIMAGE_FULL_FT_MAIN_DEPTH,
    ZIMAGE_FULL_FT_NUM_CR,
    ZIMAGE_FULL_FT_NUM_NR,
    zimage_full_finetune_inventory_expected_count,
)
from serenitymojo.models.zimage.real_weights import ZImageRealAux
from serenitymojo.models.zimage.weights import ZImageBlockWeights
from serenitymojo.tensor import Tensor
from serenitymojo.training.full_finetune_save import FullFinetuneTensor


comptime TArc = ArcPointer[Tensor]
comptime MODEL_OUT = "/tmp/zimage_full_finetune_checkpoint_smoke.safetensors"
comptime MANIFEST_OUT = "/tmp/zimage_full_finetune_checkpoint_smoke.names.safetensors"


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def _shape1(n: Int) -> List[Int]:
    var shape = List[Int]()
    shape.append(n)
    return shape^


def _tiny_bf16(ctx: DeviceContext) raises -> TArc:
    var values = List[BFloat16]()
    values.append(BFloat16(Float32(1.0)))
    var tensor = Tensor.from_host_bf16(values^, _shape1(1), ctx)
    return TArc(tensor^)


def _arc_list(tensor: TArc, count: Int) -> List[TArc]:
    var out = List[TArc]()
    for _ in range(count):
        out.append(tensor.copy())
    return out^


def _block(tensor: TArc) -> ZImageBlockWeights:
    return ZImageBlockWeights(
        tensor.copy(),
        tensor.copy(),
        tensor.copy(),
        tensor.copy(),
        tensor.copy(),
        tensor.copy(),
        tensor.copy(),
        tensor.copy(),
        tensor.copy(),
        tensor.copy(),
        tensor.copy(),
        tensor.copy(),
        tensor.copy(),
    )


def _blocks(tensor: TArc, count: Int) -> List[ZImageBlockWeights]:
    var out = List[ZImageBlockWeights]()
    for _ in range(count):
        out.append(_block(tensor))
    return out^


def _aux(tensor: TArc) -> ZImageRealAux:
    return ZImageRealAux(
        tensor.copy(),
        tensor.copy(),
        tensor.copy(),
        tensor.copy(),
        tensor.copy(),
        tensor.copy(),
        tensor.copy(),
        tensor.copy(),
        tensor.copy(),
        tensor.copy(),
        tensor.copy(),
        _arc_list(tensor, ZIMAGE_FULL_FT_NUM_NR),
        _arc_list(tensor, ZIMAGE_FULL_FT_NUM_NR),
        _arc_list(tensor, ZIMAGE_FULL_FT_MAIN_DEPTH),
        _arc_list(tensor, ZIMAGE_FULL_FT_MAIN_DEPTH),
        tensor.copy(),
        tensor.copy(),
        tensor.copy(),
        tensor.copy(),
    )


def _has_name(tensors: List[FullFinetuneTensor], name: String) -> Bool:
    for i in range(len(tensors)):
        if tensors[i].name == name:
            return True
    return False


def main() raises:
    var ctx = DeviceContext()
    var tiny = _tiny_bf16(ctx)
    var aux = _aux(tiny)
    var nr_blocks = _blocks(tiny, ZIMAGE_FULL_FT_NUM_NR)
    var cr_blocks = _blocks(tiny, ZIMAGE_FULL_FT_NUM_CR)
    var main_blocks = _blocks(tiny, ZIMAGE_FULL_FT_MAIN_DEPTH)

    var tensors = zimage_full_finetune_model_tensors(
        aux, nr_blocks, cr_blocks, main_blocks
    )
    var expected = zimage_full_finetune_inventory_expected_count()
    _require(len(tensors) == expected, String("collector count mismatch"))
    _require(expected == 521, String("Z-Image inventory expected count changed"))

    _require(
        tensors[0].name == String("t_embedder.mlp.0.weight"),
        String("first key mismatch"),
    )
    _require(
        tensors[14].name == String("all_final_layer.2-1.linear.bias"),
        String("aux boundary key mismatch"),
    )
    _require(
        tensors[15].name == String("noise_refiner.0.attention_norm1.weight"),
        String("noise_refiner start key mismatch"),
    )
    _require(
        tensors[44].name == String("noise_refiner.1.adaLN_modulation.0.bias"),
        String("noise_refiner end key mismatch"),
    )
    _require(
        tensors[45].name == String("context_refiner.0.attention_norm1.weight"),
        String("context_refiner start key mismatch"),
    )
    _require(
        tensors[70].name == String("context_refiner.1.ffn_norm2.weight"),
        String("context_refiner end key mismatch"),
    )
    _require(
        tensors[71].name == String("layers.0.attention_norm1.weight"),
        String("main layer start key mismatch"),
    )
    _require(
        tensors[len(tensors) - 1].name == String("layers.29.adaLN_modulation.0.bias"),
        String("last key mismatch"),
    )
    for i in range(ZIMAGE_FULL_FT_NUM_CR):
        _require(
            not _has_name(
                tensors,
                String("context_refiner.")
                + String(i)
                + String(".adaLN_modulation.0.weight"),
            ),
            String("context_refiner adaLN weight key must be absent"),
        )
        _require(
            not _has_name(
                tensors,
                String("context_refiner.")
                + String(i)
                + String(".adaLN_modulation.0.bias"),
            ),
            String("context_refiner adaLN bias key must be absent"),
        )

    var saved = save_zimage_full_finetune_checkpoint(
        aux,
        nr_blocks,
        cr_blocks,
        main_blocks,
        String(MODEL_OUT),
        String(MANIFEST_OUT),
        ctx,
    )
    _require(saved == expected, String("saved tensor count mismatch"))

    var loaded = load_zimage_full_finetune_payload_only(
        String(MODEL_OUT), String(MANIFEST_OUT), ctx
    )
    _require(len(loaded) == expected, String("payload-only load count mismatch"))
    _require(
        loaded[0].name == String("t_embedder.mlp.0.weight"),
        String("payload-only load first key mismatch"),
    )
    _require(
        loaded[len(loaded) - 1].name == String("layers.29.adaLN_modulation.0.bias"),
        String("payload-only load last key mismatch"),
    )
    _require(
        loaded[0].tensor[].dtype() == STDtype.BF16,
        String("payload-only load dtype mismatch"),
    )
    print("Z-Image full-finetune checkpoint smoke PASS count=", len(loaded))
