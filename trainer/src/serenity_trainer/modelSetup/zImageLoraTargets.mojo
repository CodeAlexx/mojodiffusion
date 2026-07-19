# zImageLoraTargets.mojo — Z-Image LoRA target metadata (LEAF module, no
# serenity_trainer imports, no Tensor). This module exists to BREAK the
# model⇄setup comptime/type import cycle:
#   * model/ZImageModel.mojo needs the slot constants (LORA_*, ZSLOTS, ZN_MAIN)
#     and the host helpers (lora_module_prefix, …) at comptime + runtime.
#   * modelSetup/ZImageLoRASetup.mojo needs the runtime ZImageLoraSet from model.
# Previously BOTH lived split across model and setup, importing each other →
# neither could elaborate first. The shared, leaf-pure consts/helpers now live
# HERE; model imports them from this leaf (no setup dep), and setup imports
# ZImageLoraSet from model in ONE direction only.
#
# PORT SPEC: Serenity modules/modelSetup/BaseZImageSetup.py LAYER_PRESETS
# (lines 38-43) + ZImageLoRASetup.py (setup_model, uses config.layer_filter
# against the transformer module names).
#
# LAYER_PRESETS (BaseZImageSetup.py:38-43):
#   "full":      []
#   "blocks":    ["layers"]
#   "attn-mlp":  regex ["^(?=.*attention)(?!.*refiner).*",
#                       "^(?=.*feed_forward)(?!.*refiner).*"]
#   "attn-only": regex ["^(?=.*attention)(?!.*refiner).*"]
#
# The default training preset wraps every Linear inside the 30 MAIN `layers.<i>`
# blocks whose name contains `attention` or `feed_forward`, and EXCLUDES the
# refiners (noise_refiner.*, context_refiner.*) via the negative lookahead
# `(?!.*refiner)`. Inside a main block the trainable Linears are (matching the
# block math in serenitymojo/models/dit/zimage_dit.mojo):
#     attention.to_q,  attention.to_k,  attention.to_v,  attention.to_out.0
#     feed_forward.w1, feed_forward.w2, feed_forward.w3
# (norm_q / norm_k are RMSNorm weights, not Linears → not LoRA targets;
#  attention_norm1/2, ffn_norm1/2, adaLN_modulation are likewise excluded.)
#
# Pure host metadata; no tensors.


# ── per-block LoRA slot indices (stable order, used by block fwd/bwd) ──────────
comptime LORA_TO_Q   = 0   # attention.to_q.weight     [dim, dim]
comptime LORA_TO_K   = 1   # attention.to_k.weight     [dim, dim]
comptime LORA_TO_V   = 2   # attention.to_v.weight     [dim, dim]
comptime LORA_TO_OUT = 3   # attention.to_out.0.weight [dim, dim]
comptime LORA_FF_W1  = 4   # feed_forward.w1.weight    [ff_hidden, dim]
comptime LORA_FF_W3  = 5   # feed_forward.w3.weight    [ff_hidden, dim]
comptime LORA_FF_W2  = 6   # feed_forward.w2.weight    [dim, ff_hidden]
comptime LORA_SLOTS_PER_BLOCK = 7

comptime ZIMAGE_N_MAIN_LAYERS = 30


# Relative MODULE path (no ".weight" suffix) for each LoRA slot. PEFT save keys
# are "<module>.lora_A.weight" / "<module>.lora_B.weight" (see adapters/lora.mojo).
def lora_slot_module(slot: Int) raises -> String:
    if slot == LORA_TO_Q:
        return String("attention.to_q")
    if slot == LORA_TO_K:
        return String("attention.to_k")
    if slot == LORA_TO_V:
        return String("attention.to_v")
    if slot == LORA_TO_OUT:
        return String("attention.to_out.0")
    if slot == LORA_FF_W1:
        return String("feed_forward.w1")
    if slot == LORA_FF_W3:
        return String("feed_forward.w3")
    if slot == LORA_FF_W2:
        return String("feed_forward.w2")
    raise Error(String("lora_slot_module: bad slot ") + String(slot))


# Relative key suffix (the FROZEN base weight) for a slot: "<module>.weight".
def lora_slot_base_suffix(slot: Int) raises -> String:
    return lora_slot_module(slot) + String(".weight")


# Full module prefix for a (block_idx, slot) LoRA pair, e.g.
#   "layers.7.attention.to_q".
def lora_module_prefix(block_idx: Int, slot: Int) raises -> String:
    return String("layers.") + String(block_idx) + String(".") + lora_slot_module(slot)


# Total number of LoRA adapters for the default Z-Image preset (30 blocks × 7).
def zimage_lora_count(n_layers: Int = ZIMAGE_N_MAIN_LAYERS) -> Int:
    return n_layers * LORA_SLOTS_PER_BLOCK


# Build the flat list of every LoRA module prefix (drives adapter allocation +
# safetensors save/load). Order: block-major, slot-minor.
def zimage_lora_target_prefixes(n_layers: Int = ZIMAGE_N_MAIN_LAYERS) raises -> List[String]:
    var out = List[String]()
    for b in range(n_layers):
        for s in range(LORA_SLOTS_PER_BLOCK):
            out.append(lora_module_prefix(b, s))
    return out^
