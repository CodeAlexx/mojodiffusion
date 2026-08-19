# MiniMax-H3 LoRA artifact naming shared by training export and inference load.
#
# Canonical stack / AI Toolkit-compatible external artifacts use PEFT keys:
#   diffusion_model.blocks.{i}.<module>.lora_A.weight
#   diffusion_model.blocks.{i}.<module>.lora_B.weight
#   diffusion_model.token_refiner.blocks.{i}.<module>.lora_A.weight
#   diffusion_model.token_refiner.blocks.{i}.<module>.lora_B.weight
#
# Older H3 artifacts used the Musubi/Kohya lora_unet encoding. The loader keeps
# accepting that format, but new trainer outputs use only the canonical keys.


def h3_lora_module_path(slot: Int) raises -> String:
    if slot == 0:
        return String("attn.qkv_proj")
    if slot == 1:
        return String("attn.out_proj")
    if slot == 2:
        return String("mlp.fc1")
    if slot == 3:
        return String("mlp.fc2")
    raise Error("h3 LoRA format: bad slot")


def h3_lora_peft_prefix(block: Int, slot: Int) raises -> String:
    return (
        String("diffusion_model.blocks.") + String(block) + "."
        + h3_lora_module_path(slot)
    )


def h3_lora_bare_peft_prefix(block: Int, slot: Int) raises -> String:
    """Bare PEFT is accepted because the shared stack loader accepts it too."""
    return String("blocks.") + String(block) + "." + h3_lora_module_path(slot)


def h3_lora_token_refiner_peft_prefix(block: Int, slot: Int) raises -> String:
    return (
        String("diffusion_model.token_refiner.blocks.") + String(block) + "."
        + h3_lora_module_path(slot)
    )


def h3_lora_token_refiner_bare_peft_prefix(
    block: Int, slot: Int
) raises -> String:
    return (
        String("token_refiner.blocks.") + String(block) + "."
        + h3_lora_module_path(slot)
    )


def h3_lora_legacy_slot_key(slot: Int) raises -> String:
    if slot == 0:
        return String("attn_qkv_proj")
    if slot == 1:
        return String("attn_out_proj")
    if slot == 2:
        return String("mlp_fc1")
    if slot == 3:
        return String("mlp_fc2")
    raise Error("h3 LoRA format: bad legacy slot")


def h3_lora_legacy_prefix(block: Int, slot: Int) raises -> String:
    return (
        String("lora_unet_blocks_") + String(block) + "_"
        + h3_lora_legacy_slot_key(slot)
    )
