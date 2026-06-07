# Klein/Flux2 full-finetune trainable checkpoint-key inventory.
#
# Scope: inventory only. This does not implement full-weight save/load/rebind,
# optimizer sidecars, product resume, or parity. It gives the full-finetune
# port a deterministic OneTrainer-key order to build those pieces against.

from std.collections import List


comptime KLEIN_FULL_FT_NUM_DOUBLE = 8
comptime KLEIN_FULL_FT_NUM_SINGLE = 24


def _append_klein_double_stream(mut out: List[String], prefix: String):
    out.append(prefix + String("_attn.qkv.weight"))
    out.append(prefix + String("_attn.proj.weight"))
    out.append(prefix + String("_attn.norm.query_norm.scale"))
    out.append(prefix + String("_attn.norm.key_norm.scale"))
    out.append(prefix + String("_mlp.0.weight"))
    out.append(prefix + String("_mlp.2.weight"))


def _append_klein_double_block(mut out: List[String], block_idx: Int):
    var p = String("double_blocks.") + String(block_idx) + String(".")
    _append_klein_double_stream(out, p + String("img"))
    _append_klein_double_stream(out, p + String("txt"))


def _append_klein_single_block(mut out: List[String], block_idx: Int):
    var p = String("single_blocks.") + String(block_idx)
    out.append(p + String(".linear1.weight"))
    out.append(p + String(".linear2.weight"))
    out.append(p + String(".norm.query_norm.scale"))
    out.append(p + String(".norm.key_norm.scale"))


def klein_full_finetune_tensor_names() -> List[String]:
    """Deterministic full_weight_inventory for current Klein 9B transformer.

    OneTrainer Flux2/Klein full finetune trains the transformer while VAE and
    text encoder are frozen. The order follows the Mojo loader/product forward:
    shared projections/modulation, double blocks, then single blocks.
    """
    var out = List[String]()
    out.append(String("img_in.weight"))
    out.append(String("txt_in.weight"))
    out.append(String("time_in.in_layer.weight"))
    out.append(String("time_in.out_layer.weight"))
    out.append(String("double_stream_modulation_img.lin.weight"))
    out.append(String("double_stream_modulation_txt.lin.weight"))
    out.append(String("single_stream_modulation.lin.weight"))
    out.append(String("final_layer.adaLN_modulation.1.weight"))
    out.append(String("final_layer.linear.weight"))
    for i in range(KLEIN_FULL_FT_NUM_DOUBLE):
        _append_klein_double_block(out, i)
    for i in range(KLEIN_FULL_FT_NUM_SINGLE):
        _append_klein_single_block(out, i)
    return out^


def klein_full_finetune_checkpoint_key_manifest() -> List[String]:
    """Ordered full_finetune_name_manifest source for later save/load work."""
    return klein_full_finetune_tensor_names()


def klein_full_finetune_inventory_expected_count() -> Int:
    return 9 + KLEIN_FULL_FT_NUM_DOUBLE * 12 + KLEIN_FULL_FT_NUM_SINGLE * 4
