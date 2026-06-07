# Z-Image full-finetune trainable checkpoint-key inventory.
#
# Scope: inventory only. This does not implement full-weight save/load/rebind,
# optimizer sidecars, product resume, or parity. It gives the full-finetune
# port a deterministic OneTrainer-key order to build those pieces against.

from std.collections import List


comptime ZIMAGE_FULL_FT_NUM_NR = 2
comptime ZIMAGE_FULL_FT_NUM_CR = 2
comptime ZIMAGE_FULL_FT_MAIN_DEPTH = 30


def _append_zimage_block(mut out: List[String], prefix: String):
    out.append(prefix + String(".attention_norm1.weight"))
    out.append(prefix + String(".attention.to_q.weight"))
    out.append(prefix + String(".attention.to_k.weight"))
    out.append(prefix + String(".attention.to_v.weight"))
    out.append(prefix + String(".attention.to_out.0.weight"))
    out.append(prefix + String(".attention.norm_q.weight"))
    out.append(prefix + String(".attention.norm_k.weight"))
    out.append(prefix + String(".attention_norm2.weight"))
    out.append(prefix + String(".ffn_norm1.weight"))
    out.append(prefix + String(".feed_forward.w1.weight"))
    out.append(prefix + String(".feed_forward.w3.weight"))
    out.append(prefix + String(".feed_forward.w2.weight"))
    out.append(prefix + String(".ffn_norm2.weight"))
    out.append(prefix + String(".adaLN_modulation.0.weight"))
    out.append(prefix + String(".adaLN_modulation.0.bias"))


def _append_zimage_unmodulated_refiner(mut out: List[String], prefix: String):
    out.append(prefix + String(".attention_norm1.weight"))
    out.append(prefix + String(".attention.to_q.weight"))
    out.append(prefix + String(".attention.to_k.weight"))
    out.append(prefix + String(".attention.to_v.weight"))
    out.append(prefix + String(".attention.to_out.0.weight"))
    out.append(prefix + String(".attention.norm_q.weight"))
    out.append(prefix + String(".attention.norm_k.weight"))
    out.append(prefix + String(".attention_norm2.weight"))
    out.append(prefix + String(".ffn_norm1.weight"))
    out.append(prefix + String(".feed_forward.w1.weight"))
    out.append(prefix + String(".feed_forward.w3.weight"))
    out.append(prefix + String(".feed_forward.w2.weight"))
    out.append(prefix + String(".ffn_norm2.weight"))


def _append_zimage_aux(mut out: List[String]):
    out.append(String("t_embedder.mlp.0.weight"))
    out.append(String("t_embedder.mlp.0.bias"))
    out.append(String("t_embedder.mlp.2.weight"))
    out.append(String("t_embedder.mlp.2.bias"))
    out.append(String("cap_embedder.0.weight"))
    out.append(String("cap_embedder.1.weight"))
    out.append(String("cap_embedder.1.bias"))
    out.append(String("all_x_embedder.2-1.weight"))
    out.append(String("all_x_embedder.2-1.bias"))
    out.append(String("x_pad_token"))
    out.append(String("cap_pad_token"))
    out.append(String("all_final_layer.2-1.adaLN_modulation.1.weight"))
    out.append(String("all_final_layer.2-1.adaLN_modulation.1.bias"))
    out.append(String("all_final_layer.2-1.linear.weight"))
    out.append(String("all_final_layer.2-1.linear.bias"))


def zimage_full_finetune_tensor_names() -> List[String]:
    """Deterministic full_weight_inventory for current Z-Image transformer.

    OneTrainer Z-Image full finetune trains the transformer while VAE and text
    encoder are frozen. The order follows the Mojo loader/product forward:
    aux/embedder/final tensors, modulated noise refiners, unmodulated context
    refiners, then modulated main layers.
    """
    var out = List[String]()
    _append_zimage_aux(out)
    for i in range(ZIMAGE_FULL_FT_NUM_NR):
        _append_zimage_block(out, String("noise_refiner.") + String(i))
    for i in range(ZIMAGE_FULL_FT_NUM_CR):
        _append_zimage_unmodulated_refiner(out, String("context_refiner.") + String(i))
    for i in range(ZIMAGE_FULL_FT_MAIN_DEPTH):
        _append_zimage_block(out, String("layers.") + String(i))
    return out^


def zimage_full_finetune_checkpoint_key_manifest() -> List[String]:
    """Ordered full_finetune_name_manifest source for later save/load work."""
    return zimage_full_finetune_tensor_names()


def zimage_full_finetune_inventory_expected_count() -> Int:
    return (
        15
        + ZIMAGE_FULL_FT_NUM_NR * 15
        + ZIMAGE_FULL_FT_NUM_CR * 13
        + ZIMAGE_FULL_FT_MAIN_DEPTH * 15
    )
