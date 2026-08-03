#!/usr/bin/env python3
"""
serenitymojo/models/text_encoder/parity/minimax_h3_conditioner_real_weight_oracle.py

MiniMax-H3 conditioner parity oracle — instantiates transformers' OWN
Qwen3VLTextModel (not a transcription), loads it with REAL MiniMax-H3
checkpoint tensors (text_encoder/, 13 of 14 shards on disk right now; shard
6 — layer 23's weights — is the only gap), and runs it on GPU in bf16 (the
checkpoint's own storage dtype, per config.json's "dtype": "bfloat16").
Never CPU/fp32: a CPU-fp32 reference diverges from a GPU-bf16 implementation
by more than the port itself, which would make the "reference" useless.

WHAT THIS TESTS: two depths that need only the shards already on disk
(shards 1-5 cover layers 0-22; shard 6, layer 23, is missing, so 23 layers
— 0..22 — is as deep as a real-weight gate can currently go):

    hidden_states[1]  = the raw state after running ONLY layer 0   (isolates one layer)
    hidden_states[23] = the raw state after running layers 0..22   (23 layers, compounding)

INDEXING: this does NOT use `output_hidden_states=True` at all. Empirically
verified (interactively, against this exact transformers 4.57.6 install)
that transformers' own `check_model_inputs(tie_last_hidden_states=True)`
decorator records index k (k>=1) as the RAW output after k layers, and
OVERWRITES only the very last tuple entry with the post-final-norm
`last_hidden_state`. Getting a genuine `hidden_states[23]` that way would
require configuring the model with `num_hidden_layers >= 24` — i.e. loading
layer 23's REAL weights too, which do not exist on disk (shard 6). So this
script sidesteps the tied-last-entry mechanism entirely: it manually loops
`model.layers[i](...)` for i in range(depth) and reads the running hidden
state directly, using the model's OWN `rotary_emb` and
`transformers.masking_utils.create_causal_mask` — i.e. still transformers'
real math for every op, just orchestrated by hand instead of through
`forward()`'s hidden-state bookkeeping. No dummy/stand-in weights anywhere
in this script (contrast the DiT-side modcache gate, which needed one for a
tensor genuinely out of scope — this script only ever touches layers whose
real bytes exist).

TEXT-ONLY SIMPLIFICATION, stated plainly: this calls `Qwen3VLTextModel`
directly, not the top-level `Qwen3VLModel` the real diffusers pipeline
calls (`components.text_encoder.model(...)` in
modular_pipelines/minimax_h3/encoders.py). `Qwen3VLTextModel.forward`'s own
position_ids fallback (`cache_position` expanded identically across the 3
MRoPE axes when `position_ids` is None) is used here instead of
`Qwen3VLModel.get_rope_index`. For an all-text prompt with no
vision_start/image/video tokens (this script's fixed ids has none), these
should be the same computation — `get_rope_index` also emits identical
sequential per-axis positions once no image/video grid is present — but
that specific equivalence is NOT itself re-verified by this script;
flagged for whoever builds the vision/keyframe path later.

Output: output/minimax_h3_conditioner/conditioner_real_weight_ref.safetensors
  ids               int64 [8]              the fixed token ids (record only)
  hidden_states_1   float32 [1, 8, 5120]    depth=1  (upcast from bf16, exact)
  hidden_states_23  float32 [1, 8, 5120]    depth=23 (upcast from bf16, exact)

Run: python3 serenitymojo/models/text_encoder/parity/minimax_h3_conditioner_real_weight_oracle.py
"""

import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextConfig, Qwen3VLTextModel

TEXT_ENCODER_DIR = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/text_encoder"
OUT_DIR = "/home/alex/mojodiffusion/output/minimax_h3_conditioner"
DEVICE = "cuda"
DTYPE = torch.bfloat16
LAYER_PREFIX = "model.language_model."  # measured against the real index — see .mojo gate header

# Fixed ids — the SAME ones the Mojo gate feeds `_h3_load_layer`/`_layer`
# (see minimax_h3_qwen3vl_streamed_probe.mojo). No tokenizer involved on
# either side of this gate; the token-ids-are-identical contract is the
# whole point.
IDS = [9906, 1917, 0, 1, 2, 3, 4, 5]
MAX_DEPTH = 23


def load_text_config() -> Qwen3VLTextConfig:
    with open(f"{TEXT_ENCODER_DIR}/config.json") as f:
        cfg = json.load(f)["text_config"]
    return Qwen3VLTextConfig(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        rope_theta=cfg["rope_theta"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_scaling=cfg["rope_scaling"],
    )


def load_weight_map() -> dict:
    with open(f"{TEXT_ENCODER_DIR}/model.safetensors.index.json") as f:
        return json.load(f)["weight_map"]


_shard_handles: dict = {}


def get_real_tensor(name: str, weight_map: dict) -> torch.Tensor:
    fname = weight_map[name]
    if fname not in _shard_handles:
        _shard_handles[fname] = safe_open(
            os.path.join(TEXT_ENCODER_DIR, fname), framework="pt", device="cpu"
        )
    return _shard_handles[fname].get_tensor(name)


def load_real_layers(model: Qwen3VLTextModel, weight_map: dict, num_layers: int) -> None:
    """Overwrite embed_tokens + layers 0..num_layers-1 with REAL checkpoint
    bytes, cast to the checkpoint's native bf16. Every other parameter in
    `model` (layers num_layers..num_hidden_layers-1, model.norm) stays
    random-init and is NEVER read by `run_depth` below."""
    with torch.no_grad():
        model.embed_tokens.weight.copy_(
            get_real_tensor(LAYER_PREFIX + "embed_tokens.weight", weight_map).to(DTYPE)
        )
        for i in range(num_layers):
            p = f"{LAYER_PREFIX}layers.{i}."
            layer = model.layers[i]
            layer.input_layernorm.weight.copy_(get_real_tensor(p + "input_layernorm.weight", weight_map).to(DTYPE))
            layer.post_attention_layernorm.weight.copy_(
                get_real_tensor(p + "post_attention_layernorm.weight", weight_map).to(DTYPE)
            )
            layer.self_attn.q_proj.weight.copy_(get_real_tensor(p + "self_attn.q_proj.weight", weight_map).to(DTYPE))
            layer.self_attn.k_proj.weight.copy_(get_real_tensor(p + "self_attn.k_proj.weight", weight_map).to(DTYPE))
            layer.self_attn.v_proj.weight.copy_(get_real_tensor(p + "self_attn.v_proj.weight", weight_map).to(DTYPE))
            layer.self_attn.o_proj.weight.copy_(get_real_tensor(p + "self_attn.o_proj.weight", weight_map).to(DTYPE))
            layer.self_attn.q_norm.weight.copy_(get_real_tensor(p + "self_attn.q_norm.weight", weight_map).to(DTYPE))
            layer.self_attn.k_norm.weight.copy_(get_real_tensor(p + "self_attn.k_norm.weight", weight_map).to(DTYPE))
            layer.mlp.gate_proj.weight.copy_(get_real_tensor(p + "mlp.gate_proj.weight", weight_map).to(DTYPE))
            layer.mlp.up_proj.weight.copy_(get_real_tensor(p + "mlp.up_proj.weight", weight_map).to(DTYPE))
            layer.mlp.down_proj.weight.copy_(get_real_tensor(p + "mlp.down_proj.weight", weight_map).to(DTYPE))


def run_depth(model: Qwen3VLTextModel, ids_t: torch.Tensor, depth: int) -> torch.Tensor:
    """Raw (pre-norm) hidden state after running layers 0..depth-1 —
    `hidden_states[depth]` in HF's own convention, computed by hand (see
    module docstring for why)."""
    with torch.no_grad():
        emb = model.embed_tokens(ids_t)
        seq = emb.shape[1]
        cache_position = torch.arange(0, seq, device=DEVICE)
        position_ids = cache_position.view(1, 1, -1).expand(3, ids_t.shape[0], -1)
        text_position_ids = position_ids[0]
        attn_mask = create_causal_mask(
            config=model.config,
            input_embeds=emb,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        position_embeddings = model.rotary_emb(emb, position_ids)

        hidden = emb
        for i in range(depth):
            hidden = model.layers[i](
                hidden,
                attention_mask=attn_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        return hidden


def main() -> None:
    print("MiniMax-H3 conditioner real-weight oracle (transformers 4.57.6 Qwen3VLTextModel)")
    text_config = load_text_config()
    print(
        "  text_config: hidden=", text_config.hidden_size,
        " layers(total)=", text_config.num_hidden_layers,
        " heads=", text_config.num_attention_heads,
        " kv_heads=", text_config.num_key_value_heads,
        " head_dim=", text_config.head_dim,
    )
    weight_map = load_weight_map()
    print("  index tensors:", len(weight_map))

    model = Qwen3VLTextModel(text_config).to(DEVICE, dtype=DTYPE).eval()
    load_real_layers(model, weight_map, MAX_DEPTH)
    print("  loaded REAL bf16 bytes for embed_tokens + layers 0..", MAX_DEPTH - 1)

    ids_t = torch.tensor([IDS], device=DEVICE, dtype=torch.long)

    h1 = run_depth(model, ids_t, 1)
    h23 = run_depth(model, ids_t, MAX_DEPTH)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "conditioner_real_weight_ref.safetensors")
    save_file(
        {
            "ids": torch.tensor(IDS, dtype=torch.int64),
            "hidden_states_1": h1.float().cpu().contiguous(),
            "hidden_states_23": h23.float().cpu().contiguous(),
        },
        out_path,
    )

    print("  depth=1  shape=", tuple(h1.shape), " |h|=", h1.float().norm().item())
    print("  depth=23 shape=", tuple(h23.shape), " |h|=", h23.float().norm().item())
    print("  wrote", out_path)


if __name__ == "__main__":
    main()
