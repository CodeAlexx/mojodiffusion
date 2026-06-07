# models/chroma/full_finetune_inventory.mojo -- Chroma transformer tensor order.
#
# Scope: current local Chroma transformer surface only. OneTrainer
# ChromaFineTuneSetup can also train text encoder and embeddings when the config
# enables them; those are not covered by this transformer manifest.
#
# The Chroma transformer checkpoint has 1023 tensors:
#   * 6 stack-level embed/proj tensors
#   * 29 distilled_guidance_layer tensors
#   * 19 double blocks * 28 tensors
#   * 38 single blocks * 12 tensors
#
# This is inventory metadata only. It does not claim product full-finetune,
# model rebind, resume parity, or OneTrainer numeric parity.

from std.collections import List


comptime CHROMA_FULL_FT_NUM_DOUBLE = 19
comptime CHROMA_FULL_FT_NUM_SINGLE = 38
comptime CHROMA_FULL_FT_DISTILLED_COUNT = 29
comptime CHROMA_FULL_FT_STACK_COUNT = 6
comptime CHROMA_FULL_FT_DOUBLE_PER_BLOCK = 28
comptime CHROMA_FULL_FT_SINGLE_PER_BLOCK = 12
comptime CHROMA_FULL_FT_EXPECTED_COUNT = (
    CHROMA_FULL_FT_STACK_COUNT
    + CHROMA_FULL_FT_DISTILLED_COUNT
    + CHROMA_FULL_FT_NUM_DOUBLE * CHROMA_FULL_FT_DOUBLE_PER_BLOCK
    + CHROMA_FULL_FT_NUM_SINGLE * CHROMA_FULL_FT_SINGLE_PER_BLOCK
)


def _append_linear(mut out: List[String], prefix: String):
    out.append(prefix + String(".weight"))
    out.append(prefix + String(".bias"))


def _append_double_stream(
    mut out: List[String],
    prefix: String,
    q: String,
    k: String,
    v: String,
    proj: String,
    mlp0: String,
    mlp2: String,
    q_norm: String,
    k_norm: String,
):
    _append_linear(out, prefix + q)
    _append_linear(out, prefix + k)
    _append_linear(out, prefix + v)
    _append_linear(out, prefix + proj)
    _append_linear(out, prefix + mlp0)
    _append_linear(out, prefix + mlp2)
    out.append(prefix + q_norm + String(".weight"))
    out.append(prefix + k_norm + String(".weight"))


def _append_double_block(mut out: List[String], block_idx: Int):
    var p = String("transformer_blocks.") + String(block_idx) + String(".")
    _append_double_stream(
        out,
        p,
        String("attn.to_q"),
        String("attn.to_k"),
        String("attn.to_v"),
        String("attn.to_out.0"),
        String("ff.net.0.proj"),
        String("ff.net.2"),
        String("attn.norm_q"),
        String("attn.norm_k"),
    )
    _append_double_stream(
        out,
        p,
        String("attn.add_q_proj"),
        String("attn.add_k_proj"),
        String("attn.add_v_proj"),
        String("attn.to_add_out"),
        String("ff_context.net.0.proj"),
        String("ff_context.net.2"),
        String("attn.norm_added_q"),
        String("attn.norm_added_k"),
    )


def _append_single_block(mut out: List[String], block_idx: Int):
    var p = String("single_transformer_blocks.") + String(block_idx) + String(".")
    _append_linear(out, p + String("attn.to_q"))
    _append_linear(out, p + String("attn.to_k"))
    _append_linear(out, p + String("attn.to_v"))
    _append_linear(out, p + String("proj_mlp"))
    _append_linear(out, p + String("proj_out"))
    out.append(p + String("attn.norm_q.weight"))
    out.append(p + String("attn.norm_k.weight"))


def _append_distilled_guidance(mut out: List[String]):
    _append_linear(out, String("distilled_guidance_layer.in_proj"))
    for i in range(5):
        var p = String("distilled_guidance_layer.layers.") + String(i) + String(".")
        _append_linear(out, p + String("linear_1"))
        _append_linear(out, p + String("linear_2"))
    for i in range(5):
        out.append(
            String("distilled_guidance_layer.norms.") + String(i) + String(".weight")
        )
    _append_linear(out, String("distilled_guidance_layer.out_proj"))


def chroma_full_finetune_checkpoint_key_manifest() -> List[String]:
    var out = List[String]()

    out.append(String("x_embedder.weight"))
    out.append(String("x_embedder.bias"))
    out.append(String("context_embedder.weight"))
    out.append(String("context_embedder.bias"))
    out.append(String("proj_out.weight"))
    out.append(String("proj_out.bias"))

    _append_distilled_guidance(out)

    for i in range(CHROMA_FULL_FT_NUM_DOUBLE):
        _append_double_block(out, i)
    for i in range(CHROMA_FULL_FT_NUM_SINGLE):
        _append_single_block(out, i)

    return out^


def chroma_full_finetune_inventory_expected_count() -> Int:
    return CHROMA_FULL_FT_EXPECTED_COUNT


def chroma_full_finetune_scope_note() -> String:
    return String(
        "Chroma transformer-only full-finetune scaffold; text encoder and embeddings excluded"
    )
