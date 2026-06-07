# Klein full-finetune checkpoint collector.
#
# Scope: model-tensor save/manifest payload surface only. This file does not
# wire full-finetune training, optimizer state, or runtime struct rebind.

from std.collections import List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.models.klein.double_block import (
    DoubleBlockWeights,
    StreamWeights,
)
from serenitymojo.models.klein.full_finetune_inventory import (
    KLEIN_FULL_FT_NUM_DOUBLE,
    KLEIN_FULL_FT_NUM_SINGLE,
    klein_full_finetune_checkpoint_key_manifest,
)
from serenitymojo.models.klein.klein_stack import KleinStackBase
from serenitymojo.models.klein.single_block import SingleBlockWeights
from serenitymojo.models.klein.weights import KleinStepModWeights
from serenitymojo.training.full_finetune_save import (
    FullFinetuneTensor,
    assert_full_finetune_name_manifest_matches,
    load_full_finetune_model_tensors,
    save_full_finetune_model_tensors,
    save_full_finetune_name_manifest,
)


comptime TArc = ArcPointer[Tensor]


def _append_arc(mut out: List[FullFinetuneTensor], name: String, tensor: TArc):
    out.append(FullFinetuneTensor(name, tensor.copy()))


def _append_tensor_clone(
    mut out: List[FullFinetuneTensor], name: String, tensor: Tensor, ctx: DeviceContext
) raises:
    # KleinStepModWeights currently stores owned Tensor fields in this tree.
    # Clone is a device-to-device raw-byte copy that preserves storage dtype; it
    # does not materialize host Float32 or call Tensor.to_host().
    out.append(FullFinetuneTensor(name, TArc(tensor.clone(ctx))))


def _append_stream(
    mut out: List[FullFinetuneTensor], prefix: String, stream: StreamWeights
):
    _append_arc(out, prefix + String("_attn.qkv.weight"), stream.wqkv)
    _append_arc(out, prefix + String("_attn.proj.weight"), stream.wproj)
    _append_arc(out, prefix + String("_attn.norm.query_norm.scale"), stream.q_norm)
    _append_arc(out, prefix + String("_attn.norm.key_norm.scale"), stream.k_norm)
    _append_arc(out, prefix + String("_mlp.0.weight"), stream.wgu)
    _append_arc(out, prefix + String("_mlp.2.weight"), stream.wd)


def _append_double_block(
    mut out: List[FullFinetuneTensor], block_idx: Int, block: DoubleBlockWeights
):
    var p = String("double_blocks.") + String(block_idx) + String(".")
    _append_stream(out, p + String("img"), block.img)
    _append_stream(out, p + String("txt"), block.txt)


def _append_single_block(
    mut out: List[FullFinetuneTensor], block_idx: Int, block: SingleBlockWeights
):
    var p = String("single_blocks.") + String(block_idx)
    _append_arc(out, p + String(".linear1.weight"), block.w1)
    _append_arc(out, p + String(".linear2.weight"), block.w2)
    _append_arc(out, p + String(".norm.query_norm.scale"), block.q_norm)
    _append_arc(out, p + String(".norm.key_norm.scale"), block.k_norm)


def _assert_matches_manifest(tensors: List[FullFinetuneTensor]) raises:
    var expected = klein_full_finetune_checkpoint_key_manifest()
    if len(tensors) != len(expected):
        raise Error(
            String("Klein full-finetune tensor count mismatch: got ")
            + String(len(tensors))
            + String(" expected ")
            + String(len(expected))
        )
    for i in range(len(expected)):
        if tensors[i].name != expected[i]:
            raise Error(
                String("Klein full-finetune key-order mismatch at index ")
                + String(i)
                + String(": expected ")
                + expected[i]
                + String(" got ")
                + tensors[i].name
            )


def collect_klein_full_finetune_tensors(
    base: KleinStackBase,
    step_mod: KleinStepModWeights,
    double_blocks: List[DoubleBlockWeights],
    single_blocks: List[SingleBlockWeights],
    ctx: DeviceContext,
) raises -> List[FullFinetuneTensor]:
    """Collect live Klein runtime tensors in OneTrainer full-weight key order.

    The order is the manifest order from full_finetune_inventory: shared
    projections/modulation tensors, all double blocks, then all single blocks.
    This is a save payload collector only; it does not imply full-finetune
    training or resume rebind support.
    """
    if len(double_blocks) != KLEIN_FULL_FT_NUM_DOUBLE:
        raise Error(
            String("Klein full-finetune expected ")
            + String(KLEIN_FULL_FT_NUM_DOUBLE)
            + String(" double blocks, got ")
            + String(len(double_blocks))
        )
    if len(single_blocks) != KLEIN_FULL_FT_NUM_SINGLE:
        raise Error(
            String("Klein full-finetune expected ")
            + String(KLEIN_FULL_FT_NUM_SINGLE)
            + String(" single blocks, got ")
            + String(len(single_blocks))
        )

    var out = List[FullFinetuneTensor]()

    _append_arc(out, String("img_in.weight"), base.img_in)
    _append_arc(out, String("txt_in.weight"), base.txt_in)
    _append_tensor_clone(out, String("time_in.in_layer.weight"), step_mod.t_in, ctx)
    _append_tensor_clone(out, String("time_in.out_layer.weight"), step_mod.t_out, ctx)
    _append_tensor_clone(
        out,
        String("double_stream_modulation_img.lin.weight"),
        step_mod.img_mod,
        ctx,
    )
    _append_tensor_clone(
        out,
        String("double_stream_modulation_txt.lin.weight"),
        step_mod.txt_mod,
        ctx,
    )
    _append_tensor_clone(
        out,
        String("single_stream_modulation.lin.weight"),
        step_mod.single_mod,
        ctx,
    )
    _append_tensor_clone(
        out,
        String("final_layer.adaLN_modulation.1.weight"),
        step_mod.final_mod,
        ctx,
    )
    _append_arc(out, String("final_layer.linear.weight"), base.final_lin)

    for i in range(len(double_blocks)):
        _append_double_block(out, i, double_blocks[i])
    for i in range(len(single_blocks)):
        _append_single_block(out, i, single_blocks[i])

    _assert_matches_manifest(out)
    return out^


def save_klein_full_finetune_checkpoint(
    base: KleinStackBase,
    step_mod: KleinStepModWeights,
    double_blocks: List[DoubleBlockWeights],
    single_blocks: List[SingleBlockWeights],
    model_path: String,
    manifest_path: String,
    ctx: DeviceContext,
) raises -> Int:
    """Save Klein full-finetune model tensors and matching name manifest.

    Returns the number of model tensors written. The companion manifest binds
    opaque trainer parameter indices to the exact OneTrainer tensor-name order.
    Optimizer/master state remains a separate artifact.
    """
    var tensors = collect_klein_full_finetune_tensors(
        base, step_mod, double_blocks, single_blocks, ctx
    )
    var saved = save_full_finetune_model_tensors(tensors, model_path, ctx)
    var manifest_saved = save_full_finetune_name_manifest(
        klein_full_finetune_checkpoint_key_manifest(), manifest_path, ctx
    )
    if manifest_saved != saved:
        raise Error("Klein full-finetune manifest/model tensor count mismatch")
    return saved


def load_klein_full_finetune_payload_only(
    model_path: String, manifest_path: String, ctx: DeviceContext
) raises -> List[FullFinetuneTensor]:
    """Load and validate the flat Klein full-finetune payload only.

    This deliberately does not rebind loaded tensors into KleinStackBase,
    KleinStepModWeights, DoubleBlockWeights, or SingleBlockWeights.
    """
    var names = klein_full_finetune_checkpoint_key_manifest()
    var payload = load_full_finetune_model_tensors(names, model_path, ctx)
    assert_full_finetune_name_manifest_matches(names, manifest_path)
    return payload^
