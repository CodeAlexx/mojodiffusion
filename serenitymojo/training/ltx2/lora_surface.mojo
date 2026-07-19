# lora_surface.mojo -- AV LoRA target surface definitions for LTX-2.
#
# Musubi's LTX2 t2v preset targets Q/K/V/Out projections in all six AV
# attention families inside BasicAVTransformerBlock. FFN targets are preset
# extensions, not part of the default t2v attention surface.

from std.collections import List
from serenitymojo.training.ltx2.config import (
    PRESET_AUDIO,
    PRESET_AUDIO_REF_ONLY_IC,
    PRESET_FULL,
    PRESET_T2V,
    PRESET_V2V,
)


comptime LTX2_NUM_BLOCKS = 48
comptime ATTENTION_PROJECTIONS_PER_MODULE = 4
comptime ATTENTION_PROJECTIONS_WITH_GATE_PER_MODULE = 5
comptime AV_ATTENTION_MODULES_PER_BLOCK = 6
comptime DEFAULT_T2V_TARGETS_PER_BLOCK = AV_ATTENTION_MODULES_PER_BLOCK * ATTENTION_PROJECTIONS_PER_MODULE
comptime DEFAULT_T2V_TARGETS_TOTAL = LTX2_NUM_BLOCKS * DEFAULT_T2V_TARGETS_PER_BLOCK


def attention_families() -> List[String]:
    var out = List[String]()
    out.append("attn1")
    out.append("attn2")
    out.append("audio_attn1")
    out.append("audio_attn2")
    out.append("audio_to_video_attn")
    out.append("video_to_audio_attn")
    return out^


def default_attention_projections() -> List[String]:
    var out = List[String]()
    out.append("to_k")
    out.append("to_q")
    out.append("to_v")
    out.append("to_out.0")
    return out^


def attention_projections(include_gate_logits: Bool) -> List[String]:
    var out = default_attention_projections()
    if include_gate_logits:
        out.append("to_gate_logits")
    return out^


def video_ffn_modules() -> List[String]:
    var out = List[String]()
    out.append("ff.net.0.proj")
    out.append("ff.net.2")
    return out^


def audio_ffn_modules() -> List[String]:
    var out = List[String]()
    out.append("audio_ff.net.0.proj")
    out.append("audio_ff.net.2")
    return out^


def block_prefix(block_idx: Int) -> String:
    return String("transformer_blocks.") + String(block_idx)


def base_weight_key(block_idx: Int, module_path: String) -> String:
    return block_prefix(block_idx) + String(".") + module_path + String(".weight")


def diffusion_lora_prefix(block_idx: Int, module_path: String) -> String:
    return String("diffusion_model.") + block_prefix(block_idx) + String(".") + module_path


def diffusion_lora_a_key(block_idx: Int, module_path: String) -> String:
    return diffusion_lora_prefix(block_idx, module_path) + String(".lora_A.weight")


def diffusion_lora_b_key(block_idx: Int, module_path: String) -> String:
    return diffusion_lora_prefix(block_idx, module_path) + String(".lora_B.weight")


def bare_lora_a_key(block_idx: Int, module_path: String) -> String:
    return block_prefix(block_idx) + String(".") + module_path + String(".lora_A.weight")


def bare_lora_b_key(block_idx: Int, module_path: String) -> String:
    return block_prefix(block_idx) + String(".") + module_path + String(".lora_B.weight")


def t2v_attention_modules_per_block() -> List[String]:
    var out = List[String]()
    var fam = attention_families()
    var proj = default_attention_projections()
    for i in range(len(fam)):
        for j in range(len(proj)):
            out.append(fam[i] + String(".") + proj[j])
    return out^


def v2v_modules_per_block() -> List[String]:
    var out = t2v_attention_modules_per_block()
    var vf = video_ffn_modules()
    for i in range(len(vf)):
        out.append(vf[i])
    var af = audio_ffn_modules()
    for i in range(len(af)):
        out.append(af[i])
    return out^


def audio_modules_per_block() -> List[String]:
    var out = List[String]()
    var fam = List[String]()
    fam.append("audio_attn1")
    fam.append("audio_attn2")
    fam.append("video_to_audio_attn")
    var proj = default_attention_projections()
    for i in range(len(fam)):
        for j in range(len(proj)):
            out.append(fam[i] + String(".") + proj[j])
    var af = audio_ffn_modules()
    for i in range(len(af)):
        out.append(af[i])
    return out^


def audio_ref_only_ic_modules_per_block() -> List[String]:
    var out = List[String]()
    var fam = List[String]()
    fam.append("audio_attn1")
    fam.append("audio_attn2")
    fam.append("audio_to_video_attn")
    fam.append("video_to_audio_attn")
    var proj = default_attention_projections()
    for i in range(len(fam)):
        for j in range(len(proj)):
            out.append(fam[i] + String(".") + proj[j])
    var af = audio_ffn_modules()
    for i in range(len(af)):
        out.append(af[i])
    return out^


def modules_per_block_for_preset(preset: Int) -> List[String]:
    if preset == PRESET_T2V:
        return t2v_attention_modules_per_block()
    if preset == PRESET_V2V:
        return v2v_modules_per_block()
    if preset == PRESET_AUDIO:
        return audio_modules_per_block()
    if preset == PRESET_AUDIO_REF_ONLY_IC:
        return audio_ref_only_ic_modules_per_block()
    return List[String]()


def target_count_for_preset(preset: Int, num_blocks: Int = LTX2_NUM_BLOCKS) -> Int:
    if preset == PRESET_FULL:
        return -1
    return len(modules_per_block_for_preset(preset)) * num_blocks


def lora_surface_summary(preset: Int) -> String:
    if preset == PRESET_T2V:
        return String("t2v: 6 AV attention modules x 4 projections x 48 blocks = 1152 adapters")
    if preset == PRESET_V2V:
        return String("v2v: t2v attention plus video/audio FFN modules = 1344 adapters")
    if preset == PRESET_AUDIO:
        return String("audio: audio attention/FFN plus video_to_audio attention = 672 adapters")
    if preset == PRESET_AUDIO_REF_ONLY_IC:
        return String("audio_ref_only_ic: audio attention/FFN plus both AV cross-modal directions = 864 adapters")
    return String("full: all Linear layers in BasicAVTransformerBlock; count requires checkpoint inspection")
