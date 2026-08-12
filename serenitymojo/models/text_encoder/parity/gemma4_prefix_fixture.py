# Build two tiny safetensors fixtures carrying the EXACT key set
# `gemma4_ltx_streamed._load_layer` requests, one per shipping naming layout:
#
#   google/gemma-4-12B-it            -> model.language_model.*
#   LTX-2.5 gemma4-12b-with-proj-*   -> model.*
#
# Both key layouts were read off the real files' safetensors headers. Shapes are
# deliberately tiny: `_load_bf16` reads whatever shape is present, so this
# isolates NAME RESOLUTION, which is the thing the prefix fix changes. Loading
# 26 GB to test a string prefix would prove nothing extra.
#
# Run:
#   PYTHONPATH=/home/alex/ltx25-parity-pkgs /home/alex/serenityflow-v2/.venv/bin/python \
#     serenitymojo/models/text_encoder/parity/gemma4_prefix_fixture.py

import os
import torch
from safetensors.torch import save_file

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemma4_prefix_fixtures")
os.makedirs(OUT, exist_ok=True)

H = 8            # stand-in hidden
GLOBALS = {5, 11}  # (li+1)%6==0 within the range we build


def layer_tensors(p, li, is_global):
    """Exactly the keys _load_layer asks for. Global layers carry NO v_proj —
    `attention_k_eq_v` removes the tensor entirely in the real checkpoint."""
    t = {
        f"{p}layers.{li}.input_layernorm.weight": torch.ones(H),
        f"{p}layers.{li}.post_attention_layernorm.weight": torch.ones(H),
        f"{p}layers.{li}.pre_feedforward_layernorm.weight": torch.ones(H),
        f"{p}layers.{li}.post_feedforward_layernorm.weight": torch.ones(H),
        f"{p}layers.{li}.self_attn.q_norm.weight": torch.ones(H),
        f"{p}layers.{li}.self_attn.k_norm.weight": torch.ones(H),
        f"{p}layers.{li}.self_attn.q_proj.weight": torch.ones(H, H),
        f"{p}layers.{li}.self_attn.k_proj.weight": torch.ones(H, H),
        f"{p}layers.{li}.self_attn.o_proj.weight": torch.ones(H, H),
        f"{p}layers.{li}.mlp.gate_proj.weight": torch.ones(H, H),
        f"{p}layers.{li}.mlp.up_proj.weight": torch.ones(H, H),
        f"{p}layers.{li}.mlp.down_proj.weight": torch.ones(H, H),
        # distinctive value so the probe can prove it read the right tensor
        f"{p}layers.{li}.layer_scalar": torch.tensor([0.375 + li]),
    }
    if not is_global:
        t[f"{p}layers.{li}.self_attn.v_proj.weight"] = torch.ones(H, H)
    return t


for name, p in [("google_style", "model.language_model."), ("ltx25_style", "model.")]:
    sd = {
        f"{p}embed_tokens.weight": torch.ones(16, H),
        f"{p}norm.weight": torch.ones(H),
    }
    for li in (0, 5):
        sd.update(layer_tensors(p, li, li in GLOBALS))
    if name == "ltx25_style":
        # The real LTX file also carries these; the text path must ignore them.
        sd["text_embedding_projection.video_aggregate_embed.weight"] = torch.ones(4, H)
        sd["text_embedding_projection.audio_aggregate_embed.weight"] = torch.ones(2, H)
        sd["vision_model.patch_dense.bias"] = torch.ones(H)
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    sd = {k: v.to(torch.bfloat16) for k, v in sd.items()}
    save_file(sd, os.path.join(d, "model.safetensors"))
    print(f"{name:14s} prefix={p!r:26s} tensors={len(sd)}")

print("OUT:", OUT)
