# models/chroma/full_finetune_checkpoint.mojo -- Chroma transformer payload surface.
#
# Scope: current local Chroma transformer only. This includes stack-level
# embedders/proj_out, distilled_guidance_layer, double blocks, and single
# blocks. Text encoder and embeddings are not covered here.
#
# Runtime Chroma block weights are fused for compute; this collector slices
# those fused tensors back into exact checkpoint keys before saving. It does not
# implement product full-finetune, runtime rebind, or OneTrainer parity.

from std.collections import List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.models.chroma.chroma_block import (
    ChromaDoubleBlockWeights,
    ChromaSingleBlockWeights,
    ChromaStreamWeights,
)
from serenitymojo.models.chroma.chroma_stack_lora import ChromaStackBase
from serenitymojo.models.chroma.full_finetune_inventory import (
    CHROMA_FULL_FT_DISTILLED_COUNT,
    CHROMA_FULL_FT_NUM_DOUBLE,
    CHROMA_FULL_FT_NUM_SINGLE,
    CHROMA_FULL_FT_STACK_COUNT,
    chroma_full_finetune_checkpoint_key_manifest,
    chroma_full_finetune_inventory_expected_count,
)
from serenitymojo.models.dit.chroma_dit import ChromaDitCache
from serenitymojo.ops.tensor_algebra import slice
from serenitymojo.tensor import Tensor
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


def _append_slice(
    mut out: List[FullFinetuneTensor],
    name: String,
    tensor: Tensor,
    dim: Int,
    start: Int,
    length: Int,
    ctx: DeviceContext,
) raises:
    out.append(FullFinetuneTensor(name, TArc(slice(tensor, dim, start, length, ctx))))


def _append_split_linear3(
    mut out: List[FullFinetuneTensor],
    prefix: String,
    names0: String,
    names1: String,
    names2: String,
    weight: Tensor,
    bias: Tensor,
    d: Int,
    ctx: DeviceContext,
) raises:
    _append_slice(out, prefix + names0 + String(".weight"), weight, 0, 0, d, ctx)
    _append_slice(out, prefix + names0 + String(".bias"), bias, 0, 0, d, ctx)
    _append_slice(out, prefix + names1 + String(".weight"), weight, 0, d, d, ctx)
    _append_slice(out, prefix + names1 + String(".bias"), bias, 0, d, d, ctx)
    _append_slice(out, prefix + names2 + String(".weight"), weight, 0, d * 2, d, ctx)
    _append_slice(out, prefix + names2 + String(".bias"), bias, 0, d * 2, d, ctx)


def _append_double_stream(
    mut out: List[FullFinetuneTensor],
    prefix: String,
    stream: ChromaStreamWeights,
    q: String,
    k: String,
    v: String,
    proj: String,
    mlp0: String,
    mlp2: String,
    q_norm: String,
    k_norm: String,
    ctx: DeviceContext,
) raises:
    var d = stream.wproj[].shape()[0]
    _append_split_linear3(out, prefix, q, k, v, stream.wqkv[], stream.bqkv[], d, ctx)
    _append_arc(out, prefix + proj + String(".weight"), stream.wproj)
    _append_arc(out, prefix + proj + String(".bias"), stream.bproj)
    _append_arc(out, prefix + mlp0 + String(".weight"), stream.wmlp0)
    _append_arc(out, prefix + mlp0 + String(".bias"), stream.bmlp0)
    _append_arc(out, prefix + mlp2 + String(".weight"), stream.wmlp2)
    _append_arc(out, prefix + mlp2 + String(".bias"), stream.bmlp2)
    _append_arc(out, prefix + q_norm + String(".weight"), stream.q_norm)
    _append_arc(out, prefix + k_norm + String(".weight"), stream.k_norm)


def _append_double_block(
    mut out: List[FullFinetuneTensor],
    block_idx: Int,
    block: ChromaDoubleBlockWeights,
    ctx: DeviceContext,
) raises:
    var p = String("transformer_blocks.") + String(block_idx) + String(".")
    _append_double_stream(
        out,
        p,
        block.img,
        String("attn.to_q"),
        String("attn.to_k"),
        String("attn.to_v"),
        String("attn.to_out.0"),
        String("ff.net.0.proj"),
        String("ff.net.2"),
        String("attn.norm_q"),
        String("attn.norm_k"),
        ctx,
    )
    _append_double_stream(
        out,
        p,
        block.txt,
        String("attn.add_q_proj"),
        String("attn.add_k_proj"),
        String("attn.add_v_proj"),
        String("attn.to_add_out"),
        String("ff_context.net.0.proj"),
        String("ff_context.net.2"),
        String("attn.norm_added_q"),
        String("attn.norm_added_k"),
        ctx,
    )


def _append_single_block(
    mut out: List[FullFinetuneTensor],
    block_idx: Int,
    block: ChromaSingleBlockWeights,
    ctx: DeviceContext,
) raises:
    var p = String("single_transformer_blocks.") + String(block_idx) + String(".")
    var d = block.w2[].shape()[0]
    var rows = block.w1[].shape()[0]
    var mlp = rows - d * 3
    if mlp <= 0:
        raise Error("Chroma full-finetune single block has invalid proj_mlp rows")
    _append_split_linear3(
        out,
        p,
        String("attn.to_q"),
        String("attn.to_k"),
        String("attn.to_v"),
        block.w1[],
        block.b1[],
        d,
        ctx,
    )
    _append_slice(out, p + String("proj_mlp.weight"), block.w1[], 0, d * 3, mlp, ctx)
    _append_slice(out, p + String("proj_mlp.bias"), block.b1[], 0, d * 3, mlp, ctx)
    _append_arc(out, p + String("proj_out.weight"), block.w2)
    _append_arc(out, p + String("proj_out.bias"), block.b2)
    _append_arc(out, p + String("attn.norm_q.weight"), block.q_norm)
    _append_arc(out, p + String("attn.norm_k.weight"), block.k_norm)


def _append_distilled_guidance(
    mut out: List[FullFinetuneTensor], dit: ChromaDitCache
) raises:
    var names = chroma_full_finetune_checkpoint_key_manifest()
    var start = CHROMA_FULL_FT_STACK_COUNT
    var stop = start + CHROMA_FULL_FT_DISTILLED_COUNT
    for i in range(start, stop):
        var name = names[i]
        if name not in dit.name_to_idx:
            raise Error(String("Chroma full-finetune missing distilled tensor ") + name)
        var idx = dit.name_to_idx[name]
        _append_arc(out, name, dit.weights[idx])


def _assert_matches_manifest(tensors: List[FullFinetuneTensor]) raises:
    var expected = chroma_full_finetune_checkpoint_key_manifest()
    if len(tensors) != len(expected):
        raise Error(
            String("Chroma full-finetune tensor count mismatch: got ")
            + String(len(tensors))
            + String(" expected ")
            + String(len(expected))
        )
    for i in range(len(expected)):
        if tensors[i].name != expected[i]:
            raise Error(
                String("Chroma full-finetune key-order mismatch at index ")
                + String(i)
                + String(": expected ")
                + expected[i]
                + String(" got ")
                + tensors[i].name
            )


def collect_chroma_full_finetune_tensors(
    base: ChromaStackBase,
    dit: ChromaDitCache,
    double_blocks: List[ChromaDoubleBlockWeights],
    single_blocks: List[ChromaSingleBlockWeights],
    ctx: DeviceContext,
) raises -> List[FullFinetuneTensor]:
    """Collect Chroma transformer tensors in checkpoint key order."""

    if len(double_blocks) != CHROMA_FULL_FT_NUM_DOUBLE:
        raise Error("Chroma full-finetune double block count mismatch")
    if len(single_blocks) != CHROMA_FULL_FT_NUM_SINGLE:
        raise Error("Chroma full-finetune single block count mismatch")

    var out = List[FullFinetuneTensor]()
    _append_arc(out, String("x_embedder.weight"), base.x_embedder_w)
    _append_arc(out, String("x_embedder.bias"), base.x_embedder_b)
    _append_arc(out, String("context_embedder.weight"), base.context_embedder_w)
    _append_arc(out, String("context_embedder.bias"), base.context_embedder_b)
    _append_arc(out, String("proj_out.weight"), base.proj_out_w)
    _append_arc(out, String("proj_out.bias"), base.proj_out_b)

    _append_distilled_guidance(out, dit)

    for i in range(len(double_blocks)):
        _append_double_block(out, i, double_blocks[i], ctx)
    for i in range(len(single_blocks)):
        _append_single_block(out, i, single_blocks[i], ctx)

    if len(out) != chroma_full_finetune_inventory_expected_count():
        raise Error("Chroma full-finetune collected tensor count mismatch")
    _assert_matches_manifest(out)
    return out^


def save_chroma_full_finetune_checkpoint(
    base: ChromaStackBase,
    dit: ChromaDitCache,
    double_blocks: List[ChromaDoubleBlockWeights],
    single_blocks: List[ChromaSingleBlockWeights],
    model_path: String,
    manifest_path: String,
    ctx: DeviceContext,
) raises -> Int:
    var tensors = collect_chroma_full_finetune_tensors(
        base, dit, double_blocks, single_blocks, ctx
    )
    var saved = save_full_finetune_model_tensors(tensors, model_path, ctx)
    var manifest_saved = save_full_finetune_name_manifest(
        chroma_full_finetune_checkpoint_key_manifest(), manifest_path, ctx
    )
    if manifest_saved != saved:
        raise Error("Chroma full-finetune manifest/model count mismatch")
    return saved


def load_chroma_full_finetune_payload_only(
    model_path: String, manifest_path: String, ctx: DeviceContext
) raises -> List[FullFinetuneTensor]:
    var names = chroma_full_finetune_checkpoint_key_manifest()
    var payload = load_full_finetune_model_tensors(names, model_path, ctx)
    assert_full_finetune_name_manifest_matches(names, manifest_path)
    return payload^
