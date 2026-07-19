# lensLoraTargets.mojo — Lens LoRA target metadata (LEAF module, no
# serenity_trainer imports, no Tensor). Mirrors modelSetup/zImageLoraTargets.mojo.
# Pure host metadata; the model⇄setup cycle-break leaf for Lens. THIS LIST IS THE
# ADAPTER-COUNT FIDELITY GATE — every adapter is a SEPARATE Linear, NOT fused.
#
# ── PORT SPEC (1:1) ───────────────────────────────────────────────────────────
# The EXACT set of nn.Linear modules LoRA wraps is the SHIPPED Lens LoRA preset,
# NOT the global default. The training preset that ships with the Lens PR sets:
#   pr-1510 training_presets/'#lens LoRA 16GB.json':
#       "training_method": "LORA",
#       "layer_filter": "attn,mlp",
#       "layer_filter_preset": "attn-mlp"
# So setup_model receives config.layer_filter="attn,mlp", and:
#   * Serenity pr-1510 modules/modelSetup/LensLoRASetup.py::setup_model:
#       model.transformer_lora = LoRAModuleWrapper(
#           model.transformer, "transformer", config, config.layer_filter.split(",")
#       )                                                  (LensLoRASetup.py:64-66)
#     ⇒ module_filters = [ModuleFilter("attn"), ModuleFilter("mlp")].
#   * Serenity pr-1510 modules/module/LoRAModule.py::LoRAModuleWrapper.__create_modules
#       for name, child in orig_module.named_modules():
#           if not isinstance(child, Linear | Conv2d): continue   (LoRAModule.py:898-900)
#           if len(module_filters)==0 or any(f.matches(name)): WRAP (LoRAModule.py:901-907)
#   * Serenity pr-1510 modules/util/ModuleFilter.py::matches — SUBSTRING match:
#       `self._pattern in module_name`  (empty pattern ⇒ matches all).
#     ⇒ "attn" wraps every Linear whose name contains "attn"; "mlp" every Linear
#       whose name contains "mlp". Non-Linear/Conv2d modules are rejected
#       (PeftBase.__init__ raises otherwise, LoRAModule.py:50-52).
#
# This MATCHES the Z-Image precedent (zImageLoraTargets), which follows the SHIPPED
# preset (attn-mlp), not the global TrainConfig default. (Supersedes the earlier
# draft of this file that used the global default "full" (12 slots/block + 6
# top-level = 582); the shipped Lens preset overrides that default, so "full" was
# wrong and over-counted the adapters.)
#
# ── what "attn,mlp" SUBSTRING-matches (lens/transformer.py) ───────────────────
# attn = LensJointAttention (transformer.py:41-46), name contains "attn":
#   attn.img_qkv  = nn.Linear(dim, 3*inner, bias=True)          (:42)
#   attn.txt_qkv  = nn.Linear(dim, 3*inner, bias=True)          (:43)
#   attn.to_out   = ModuleList([Linear(inner,out,bias=True), Identity()]) (:45) → to_out.0
#   attn.to_add_out = nn.Linear(inner, dim, bias=True)          (:46)
#   attn.norm_q/norm_k/norm_added_q/norm_added_k = RMSNorm  → NOT Linear → rejected.
# mlp = GateMLP (transformer.py:13-23); the two block instances are img_mlp /
#   txt_mlp, names contain "mlp":
#   img_mlp.w1 = nn.Linear(dim, hidden, bias=False)             (:18)
#   img_mlp.w2 = nn.Linear(hidden, dim, bias=False)             (:19)
#   img_mlp.w3 = nn.Linear(dim, hidden, bias=False)             (:20)
#   txt_mlp.w1/w2/w3 = same                                     (:78)
#
# EXCLUDED (neither "attn" nor "mlp" in the name):
#   * img_mod.1 / txt_mod.1 (Sequential(SiLU, Linear), name contains "mod") — NOT
#     wrapped by attn-mlp (this is the key difference vs the "full" preset).
#   * ALL top-level Linears: img_in, txt_in, time_text_embed.timestep_embedder.
#     linear_1/linear_2, norm_out.linear, proj_out — none contain "attn"/"mlp".
#   * img/txt_norm1/2, attn QK RMSNorm, txt_norm[i] — RMSNorm, not Linear anyway.
#
# ⇒ 10 Linears/block: img_qkv, txt_qkv, to_out.0, to_add_out,
#   img_mlp.{w1,w2,w3}, txt_mlp.{w1,w2,w3}.  NO mod, NO top-level.
#
# ── ADAPTER COUNT (fidelity gate) ─────────────────────────────────────────────
#   48 × 10 = 480 SEPARATE LoRA adapters (NOT fused).


# ── per-block LoRA slot indices (stable order; block fwd/bwd + saver agree) ────
comptime LORA_IMG_QKV    = 0   # attn.img_qkv.weight     [3*DIM, DIM]   bias=True
comptime LORA_TXT_QKV    = 1   # attn.txt_qkv.weight     [3*DIM, DIM]   bias=True
comptime LORA_TO_OUT     = 2   # attn.to_out.0.weight    [DIM, DIM]     bias=True
comptime LORA_TO_ADD_OUT = 3   # attn.to_add_out.weight  [DIM, DIM]     bias=True
comptime LORA_IMG_MLP_W1 = 4   # img_mlp.w1.weight       [FF, DIM]      bias=False
comptime LORA_IMG_MLP_W2 = 5   # img_mlp.w2.weight       [DIM, FF]      bias=False
comptime LORA_IMG_MLP_W3 = 6   # img_mlp.w3.weight       [FF, DIM]      bias=False
comptime LORA_TXT_MLP_W1 = 7   # txt_mlp.w1.weight       [FF, DIM]      bias=False
comptime LORA_TXT_MLP_W2 = 8   # txt_mlp.w2.weight       [DIM, FF]      bias=False
comptime LORA_TXT_MLP_W3 = 9   # txt_mlp.w3.weight       [FF, DIM]      bias=False
comptime LORA_SLOTS_PER_BLOCK = 10

comptime LENS_N_BLOCKS   = 48          # transformer.py:391 num_layers
comptime LENS_INNER_DIM  = 1536        # num_attention_heads(24)*attention_head_dim(64)
comptime LENS_FF_HIDDEN  = 4096        # int(dim/3*8) = int(1536/3*8) (transformer.py:313)


# Relative MODULE path (no ".weight" suffix) for a per-block slot, e.g.
# "attn.img_qkv". PEFT save keys are "<module>.lora_down.weight" /
# "<module>.lora_up.weight" (LoRAModule.py:563); the saver prepends
# "transformer.transformer_blocks.<i>.".
def lora_slot_module(slot: Int) raises -> String:
    if slot == LORA_IMG_QKV:    return String("attn.img_qkv")
    if slot == LORA_TXT_QKV:    return String("attn.txt_qkv")
    if slot == LORA_TO_OUT:     return String("attn.to_out.0")
    if slot == LORA_TO_ADD_OUT: return String("attn.to_add_out")
    if slot == LORA_IMG_MLP_W1: return String("img_mlp.w1")
    if slot == LORA_IMG_MLP_W2: return String("img_mlp.w2")
    if slot == LORA_IMG_MLP_W3: return String("img_mlp.w3")
    if slot == LORA_TXT_MLP_W1: return String("txt_mlp.w1")
    if slot == LORA_TXT_MLP_W2: return String("txt_mlp.w2")
    if slot == LORA_TXT_MLP_W3: return String("txt_mlp.w3")
    raise Error(String("lora_slot_module: bad slot ") + String(slot))


# Relative key suffix (the FROZEN base weight) for a per-block slot: "<module>.weight".
def lora_slot_base_suffix(slot: Int) raises -> String:
    return lora_slot_module(slot) + String(".weight")


# Module prefix for a (block_idx, slot) per-block LoRA pair (NO "transformer."
# host prefix), e.g. "transformer_blocks.7.attn.img_qkv". The saver prepends
# "transformer.".
def lora_module_prefix(block_idx: Int, slot: Int) raises -> String:
    return String("transformer_blocks.") + String(block_idx) + String(".") + lora_slot_module(slot)


# Total number of LoRA adapters for the shipped Lens attn-mlp preset
#   48 blocks × 10 = 480.
def lens_lora_count(n_blocks: Int = LENS_N_BLOCKS) -> Int:
    return n_blocks * LORA_SLOTS_PER_BLOCK


# Flat list of every LoRA module prefix WITH the "transformer." host prefix
# (drives adapter allocation + safetensors save/load). Order: per-block
# (block-major, slot-minor). Matches LensLoRASaver / load_lens_lora key naming.
def lens_lora_target_prefixes(n_blocks: Int = LENS_N_BLOCKS) raises -> List[String]:
    var out = List[String]()
    for b in range(n_blocks):
        for s in range(LORA_SLOTS_PER_BLOCK):
            out.append(String("transformer.") + lora_module_prefix(b, s))
    return out^
