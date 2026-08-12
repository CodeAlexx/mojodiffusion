# Gemma-4-12B-it GPU-bf16 oracle dump for the gemma4_ltx_streamed.mojo port.
#
# Run:
#   PYTHONPATH=/home/alex/ltx25-parity-pkgs \
#     /home/alex/serenityflow-v2/.venv/bin/python \
#     serenitymojo/models/text_encoder/parity/gemma4_oracle_dump.py
#
# transformers 5.15.0 (gemma4_unified) layered over the serenityflow venv's
# CUDA torch via PYTHONPATH; the venv itself is untouched.
#
# Dumps (all bf16-on-GPU truth, saved to OUT as safetensors):
#   gemma4_oracle_meta.safetensors      token ids, real_len, layer_scalars
#   gemma4_oracle_states.safetensors    all 49 hidden states [1, 1024, 3840]
#   gemma4_oracle_layer0.safetensors    sliding layer 0: input/output + attn in/out
#   gemma4_oracle_layer5.safetensors    global  layer 5: input/output + attn in/out
#   gemma4_oracle_embed.safetensors     scaled embedding output (= state 0)
#
# Conventions mirrored from the gemma3 LTX path (HYPOTHESIS for the exact
# LTX-2.5 recipe until Lightricks ships; the per-layer hook gates are
# convention-independent): LEFT-pad to 1024 with pad id 0, causal attention,
# position ids preserved via the attention mask.

import os
import torch
from safetensors.torch import save_file

CKPT = "/home/alex/.serenity/models/text_encoders/gemma-4-12b-it-standalone"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemma4_refs")
MAX_TOKENS = 1024
PROMPT = (
    "A woman with long brown hair walks through a sunlit forest, leaves "
    "crunching underfoot as birdsong echoes between the tall pines."
)

os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)

from transformers import AutoTokenizer, Gemma4UnifiedForConditionalGeneration  # noqa: E402

tok = AutoTokenizer.from_pretrained(CKPT)
# BOS MUST be prepended by hand. This checkpoint ships `add_bos_token=False`
# and a tokenizer.json post_processor whose TemplateProcessing adds nothing, so
# `add_special_tokens=True` is silently a no-op (measured: identical ids either
# way, first id 236776 == "A"). The reference LTX path does prepend it —
# ComfyUI's LTX gemma tokenizer passes `add_bos: True` (comfy/text_encoders/
# lt.py) and its gemma4 model declares `special_tokens={"start": 2}` — and so
# does our own Mojo tokenizer (`tokenizer.mojo::encode_gemma` ends with
# `ids.append(2)`). Gating against a BOS-less sequence would gate an input the
# encoder never actually receives, and Gemma parks an attention sink on token 0.
BOS_ID = tok.bos_token_id
assert BOS_ID == 2, BOS_ID
ids = tok(PROMPT, add_special_tokens=False)["input_ids"]
assert ids[0] != BOS_ID, "tokenizer already added BOS; drop the manual prepend"
ids = [BOS_ID] + ids
real_len = len(ids)
assert real_len <= MAX_TOKENS
# LEFT-pad to the full 1024 window with pad id 0, mask marks the valid suffix.
pad = MAX_TOKENS - real_len
input_ids = torch.tensor([[0] * pad + ids], dtype=torch.long, device="cuda")
attention_mask = torch.tensor([[0] * pad + [1] * real_len], dtype=torch.long, device="cuda")

# MEMORY SAFETY: the text tower alone is 23.81 GB bf16 (measured from the
# checkpoint header) against a 24.56 GB card — a plain device_map="cuda" OOMs.
# accelerate's cpu_offload keeps the weights in host RAM and streams each module
# to the GPU for its forward, so the math is still GPU-bf16 (the oracle contract)
# with only ~1 layer resident. Do NOT "fix" an OOM here by loading fp32 on host.
model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
    CKPT, dtype=torch.bfloat16, device_map="cpu"
)
model.eval()

text_model = model.model.language_model  # Gemma4UnifiedTextModel
from accelerate import cpu_offload  # noqa: E402

cpu_offload(text_model, execution_device=torch.device("cuda:0"))
layers = text_model.layers
print("layers:", len(layers), "| layer types:", model.config.text_config.layer_types[:12], "...")

layer_scalars = torch.stack([l.layer_scalar.detach().float().cpu().flatten() for l in layers]).flatten()
print("layer_scalar[0:8]:", layer_scalars[:8].tolist())

captures = {}


def cap_layer(idx):
    layer = layers[idx]

    def pre(mod, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        captures[f"layer{idx}_in"] = h.detach().clone()

    def post(mod, args, kwargs, out):
        o = out[0] if isinstance(out, tuple) else out
        captures[f"layer{idx}_out"] = o.detach().clone()

    def attn_pre(mod, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        captures[f"layer{idx}_attn_in"] = h.detach().clone()
        cos, sin = kwargs["position_embeddings"]
        captures[f"layer{idx}_cos"] = cos.detach().clone()
        captures[f"layer{idx}_sin"] = sin.detach().clone()

    def attn_post(mod, args, kwargs, out):
        captures[f"layer{idx}_attn_out"] = out[0].detach().clone()

    layer.register_forward_pre_hook(pre, with_kwargs=True)
    layer.register_forward_hook(post, with_kwargs=True)
    layer.self_attn.register_forward_pre_hook(attn_pre, with_kwargs=True)
    layer.self_attn.register_forward_hook(attn_post, with_kwargs=True)


cap_layer(0)   # sliding_attention
cap_layer(5)   # full_attention (global): head_dim 512, MQA 1 kv head, k_eq_v

with torch.no_grad():
    out = text_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )

states = out.hidden_states
print("num hidden states:", len(states), "| shape:", tuple(states[0].shape))
assert len(states) == 49

save_file(
    {
        "input_ids": input_ids.cpu(),
        "attention_mask": attention_mask.cpu(),
        "real_len": torch.tensor([real_len]),
        "layer_scalars": layer_scalars,
    },
    os.path.join(OUT, "gemma4_oracle_meta.safetensors"),
)
save_file(
    {f"state_{i:02d}": s.detach().cpu() for i, s in enumerate(states)},
    os.path.join(OUT, "gemma4_oracle_states.safetensors"),
)
save_file(
    {k.replace("layer0_", ""): v.cpu() for k, v in captures.items() if k.startswith("layer0_")},
    os.path.join(OUT, "gemma4_oracle_layer0.safetensors"),
)
save_file(
    {k.replace("layer5_", ""): v.cpu() for k, v in captures.items() if k.startswith("layer5_")},
    os.path.join(OUT, "gemma4_oracle_layer5.safetensors"),
)
save_file(
    {"embed": states[0].detach().cpu()},
    os.path.join(OUT, "gemma4_oracle_embed.safetensors"),
)

for i in (0, 1, 5, 6, 24, 48):
    s = states[i].float()
    print(f"state_{i:02d} std {s.std().item():.6e} absmax {s.abs().max().item():.6e}")
print("token ids:", ids[:16], "... real_len", real_len)
print("DONE ->", OUT)
