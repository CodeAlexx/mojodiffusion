# Chroma full-finetune transformer inventory smoke.
#
# No CUDA: validates the deterministic 1023-key transformer manifest only.
# This is not product full-finetune or OneTrainer parity.

from std.collections import List

from serenitymojo.models.chroma.full_finetune_inventory import (
    CHROMA_FULL_FT_DISTILLED_COUNT,
    CHROMA_FULL_FT_DOUBLE_PER_BLOCK,
    CHROMA_FULL_FT_NUM_DOUBLE,
    CHROMA_FULL_FT_NUM_SINGLE,
    CHROMA_FULL_FT_SINGLE_PER_BLOCK,
    CHROMA_FULL_FT_STACK_COUNT,
    chroma_full_finetune_checkpoint_key_manifest,
    chroma_full_finetune_inventory_expected_count,
)


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def _has_name(names: List[String], name: String) -> Bool:
    for i in range(len(names)):
        if names[i] == name:
            return True
    return False


def main() raises:
    var names = chroma_full_finetune_checkpoint_key_manifest()
    var expected = chroma_full_finetune_inventory_expected_count()
    _require(expected == 1023, String("Chroma expected count changed"))
    _require(len(names) == expected, String("Chroma manifest count mismatch"))
    _require(
        expected
        == (
            CHROMA_FULL_FT_STACK_COUNT
            + CHROMA_FULL_FT_DISTILLED_COUNT
            + CHROMA_FULL_FT_NUM_DOUBLE * CHROMA_FULL_FT_DOUBLE_PER_BLOCK
            + CHROMA_FULL_FT_NUM_SINGLE * CHROMA_FULL_FT_SINGLE_PER_BLOCK
        ),
        String("Chroma count formula mismatch"),
    )

    var seen = List[String]()
    for i in range(len(names)):
        _require(names[i].byte_length() > 0, String("empty Chroma key"))
        _require(not _has_name(seen, names[i]), String("duplicate Chroma key ") + names[i])
        seen.append(names[i])

    _require(names[0] == String("x_embedder.weight"), String("first key mismatch"))
    _require(names[6] == String("distilled_guidance_layer.in_proj.weight"), String("distilled first key mismatch"))
    _require(
        names[34] == String("distilled_guidance_layer.out_proj.bias"),
        String("distilled last key mismatch"),
    )
    _require(
        names[35] == String("transformer_blocks.0.attn.to_q.weight"),
        String("double block first key mismatch"),
    )
    _require(
        names[34 + CHROMA_FULL_FT_NUM_DOUBLE * CHROMA_FULL_FT_DOUBLE_PER_BLOCK + 1]
        == String("single_transformer_blocks.0.attn.to_q.weight"),
        String("single block first key mismatch"),
    )
    _require(
        names[len(names) - 1]
        == String("single_transformer_blocks.37.attn.norm_k.weight"),
        String("last key mismatch"),
    )
    _require(
        _has_name(names, String("transformer_blocks.18.attn.norm_added_k.weight")),
        String("missing representative double txt norm"),
    )
    _require(
        _has_name(names, String("single_transformer_blocks.37.proj_out.bias")),
        String("missing representative single proj_out bias"),
    )

    print("Chroma full-finetune transformer inventory smoke PASS count=", expected)
