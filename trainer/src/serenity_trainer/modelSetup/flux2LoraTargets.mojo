# flux2LoraTargets.mojo — Klein (FLUX.2) LoRA target metadata (LEAF module: no
# serenity_trainer model/setup imports, no Tensor). Breaks the model⇄setup comptime
# import cycle, exactly like modelSetup/zImageLoraTargets.mojo does for Z-Image.
#
# ── PORT SPEC (1:1) ───────────────────────────────────────────────────────────
#   * modules/modelSetup/Flux2LoRASetup.py::setup_model (:57-58):
#       model.transformer_lora = LoRAModuleWrapper(
#           model.transformer, "transformer", config, config.layer_filter.split(","))
#   * modules/modelSetup/BaseFlux2Setup.py::LAYER_PRESETS (:38-41):
#       "blocks": ["transformer_block"]
#       "full":   []
#   * modules/util/config/TrainConfig.py: layer_filter default "" (:338,1126).
#     config.layer_filter.split(",") of "" → [""]; ModuleFilter("") matches ALL
#     (ModuleFilter.py:56-57: empty pattern ⇒ is_match=True). So the DEFAULT Klein
#     LoRA run wraps EVERY nn.Linear in the diffusers Flux2Transformer2DModel.
#   * modules/module/LoRAModule.py::LoRAModuleWrapper.__create_modules (:638-692):
#     iterates orig_module.named_modules(), wraps each Linear/Conv2d whose name
#     matches a filter; the saved key prefix is "transformer." + module_name.
#
# ── THE DIFFUSERS MODULE NAMES (the SAVE-KEY layout) ──────────────────────────
# Serenity's Flux2 saver does NOT convert keys (Flux2LoRASaver._get_convert_key_sets
# returns None, :17-18 → LoRASaverMixin routes through __save_legacy_safetensors
# which, with key_sets=None, writes the raw module state_dict). So the on-disk keys
# are EXACTLY the diffusers module paths, prefixed "transformer." :
#   transformer.<diffusers_name>.lora_down.weight   (= LoRAModule.lora_down.weight, [rank,in])
#   transformer.<diffusers_name>.lora_up.weight     (= LoRAModule.lora_up.weight,   [out,rank])
#   transformer.<diffusers_name>.alpha              (LoRAModule.alpha scalar, :303)
#
# diffusers transformer_flux2.py module names (verified against the installed
# diffusers Flux2Transformer2DModel, c8656ed):
#   Per Flux2TransformerBlock i  (diffusers "transformer_blocks.{i}"):
#     attn.to_q  attn.to_k  attn.to_v  attn.to_out.0          (img stream q/k/v/proj)
#     attn.add_q_proj  attn.add_k_proj  attn.add_v_proj  attn.to_add_out  (txt stream)
#     ff.linear_in  ff.linear_out                              (img mlp)
#     ff_context.linear_in  ff_context.linear_out             (txt mlp)
#   Per Flux2SingleTransformerBlock i (diffusers "single_transformer_blocks.{i}"):
#     attn.to_qkv_mlp_proj  attn.to_out                        (fused qkv+mlp in / out)
#   (norm_q/norm_k are RMSNorm weights, not Linears → NOT LoRA targets.)
#   With layer_filter="blocks"→["transformer_block"] (substring), ONLY the two block
#   families above are matched ("transformer_blocks" and "single_transformer_blocks"
#   both contain "transformer_block"); the top-level x_embedder/context_embedder/
#   proj_out/time_guidance_embed/*modulation* Linears are EXCLUDED. The default ""
#   filter would also wrap those top-level Linears; we expose BOTH the block set
#   (the documented "blocks" preset, the recommended Klein LoRA) and the count.
#
# ── SERENITYMOJO FORWARD ↔ DIFFUSERS KEY DIVERGENCE (load-bearing note) ────────
# The borrowed Klein weight loader still has ORIGINAL FLUX.2 fused-qkv name helpers
# (img_attn.qkv, txt_attn.qkv), because base checkpoint weights are fused from the
# diffusers q/k/v tensors. The LoRA carrier is NOT fused: it has one adapter per
# Serenity-wrapped diffusers Linear, matching LoRAModuleWrapper exactly.
#
# Pure host metadata; no tensors, no device imports.


# ── flat LoRA carrier slot scheme ─────────────────────────────────────────────
# Current KleinLoraSet:
#   DBL_SLOTS=12:
#     0-3 img attn q/k/v/out, 4-5 img ff in/out,
#     6-9 txt attn q/k/v/out, 10-11 txt ff in/out
#   SGL_SLOTS=2: qkv+mlp in, out.
comptime DBL_SLOTS = 12
comptime SGL_SLOTS = 2
comptime FLUX2_FUSED_DBL_SLOTS = 4
comptime FLUX2_BK_DOUBLE = 0
comptime FLUX2_BK_SINGLE = 1

# Klein 4B arch (serenitymojo configs/klein4b.json, verified vs checkpoint header):
#   num_double = 5, num_single = 20, inner_dim(D) = 3072, num_heads = 24,
#   head_dim = 128, joint_attention_dim = 7680, in/out_channels = 128.
comptime KLEIN4B_NUM_DOUBLE = 5
comptime KLEIN4B_NUM_SINGLE = 20
comptime KLEIN4B_DIM        = 3072
comptime KLEIN4B_NUM_HEADS  = 24
comptime KLEIN4B_HEAD_DIM   = 128


# ── diffusers per-Linear key suffixes (the Serenity save names) ─────────────
# One DOUBLE block contributes 12 diffusers Linears; one SINGLE block contributes 2.
# Listed in the order diffusers_to_original (Flux2Model.py:52-71) walks them.
def double_block_diffusers_suffixes() -> List[String]:
    var o = List[String]()
    o.append(String("attn.to_q"))           # img q
    o.append(String("attn.to_k"))           # img k
    o.append(String("attn.to_v"))           # img v
    o.append(String("attn.to_out.0"))       # img proj
    o.append(String("attn.add_q_proj"))     # txt q
    o.append(String("attn.add_k_proj"))     # txt k
    o.append(String("attn.add_v_proj"))     # txt v
    o.append(String("attn.to_add_out"))     # txt proj
    o.append(String("ff.linear_in"))        # img mlp in
    o.append(String("ff.linear_out"))       # img mlp out
    o.append(String("ff_context.linear_in"))   # txt mlp in
    o.append(String("ff_context.linear_out"))  # txt mlp out
    return o^


def single_block_diffusers_suffixes() -> List[String]:
    var o = List[String]()
    o.append(String("attn.to_qkv_mlp_proj"))   # fused qkv+mlp in
    o.append(String("attn.to_out"))            # fused out
    return o^


# Full diffusers module name for a double-block Linear:
#   "transformer_blocks.<i>.<suffix>"
def flux2_double_module(block_idx: Int, suffix: String) -> String:
    return String("transformer_blocks.") + String(block_idx) + String(".") + suffix


# Full diffusers module name for a single-block Linear:
#   "single_transformer_blocks.<i>.<suffix>"
def flux2_single_module(block_idx: Int, suffix: String) -> String:
    return String("single_transformer_blocks.") + String(block_idx) + String(".") + suffix


# The Serenity on-disk PEFT prefix for a diffusers module name:
#   "transformer.<diffusers_name>"  (LoRAModuleWrapper prefix "transformer", :57)
def flux2_lora_save_prefix(diffusers_module_name: String) -> String:
    return String("transformer.") + diffusers_module_name


# Enumerate EVERY Serenity save prefix (block set, the "blocks" preset) in the
# order LoRAModuleWrapper.named_modules() would visit them: all double blocks
# (block-major, suffix-minor) then all single blocks. This is the canonical key
# order the saver writes and the loader reads.
def flux2_lora_save_prefixes(num_double: Int, num_single: Int) -> List[String]:
    var out = List[String]()
    var dsuf = double_block_diffusers_suffixes()
    for bi in range(num_double):
        for s in range(len(dsuf)):
            out.append(flux2_lora_save_prefix(flux2_double_module(bi, dsuf[s])))
    var ssuf = single_block_diffusers_suffixes()
    for bi in range(num_single):
        for s in range(len(ssuf)):
            out.append(flux2_lora_save_prefix(flux2_single_module(bi, ssuf[s])))
    return out^


# Total number of LoRA-wrapped Linears (the "blocks" preset, diffusers granularity):
#   num_double*12 + num_single*2.
def flux2_lora_count(num_double: Int, num_single: Int) -> Int:
    return num_double * 12 + num_single * 2


# ── original fused-weight slot prefixes ───────────────────────────────────────
# Kept only for the base-weight name bridge. LoRA save/load uses the diffusers
# per-Linear prefixes above.
# serenitymojo klein_stack_lora.mojo::_klein_lora_prefix (:1765-1779).
def flux2_fused_slot_prefix(block_kind: Int, block_idx: Int, slot: Int) -> String:
    if block_kind == FLUX2_BK_DOUBLE:
        var b = String("double_blocks.") + String(block_idx)
        if slot == 0:
            return b + String(".img_attn.qkv_proj")
        elif slot == 1:
            return b + String(".img_attn.out_proj")
        elif slot == 2:
            return b + String(".txt_attn.qkv_proj")
        else:
            return b + String(".txt_attn.out_proj")
    var s = String("single_blocks.") + String(block_idx)
    if slot == 0:
        return s + String(".qkv_proj")
    return s + String(".out_proj")
