#!/usr/bin/env python3
"""MiniMax-H3 2-LAYER STACK REAL-WEIGHT oracle (DEV-ONLY, never imported by
Mojo). Extends minimax_h3_block_real_weight_oracle.py (block 0 alone) to
blocks 0 AND 1 run SEQUENTIALLY — the composition minimax_h3_run_stack
(models/dit/minimax_h3_stack.mojo) performs, not just isolated blocks.

WHAT IS REAL vs SYNTHETIC vs OUT OF SCOPE — unchanged from the two prior
real-weight oracles (block gate, modcache gate):
  REAL:      the 8 checkpoint tensors EACH of blocks 0 and 1 need (norm1,
             norm2, attn.qkv_proj, attn.q_norm, attn.k_norm, attn.out_proj,
             mlp.fc1, mlp.fc2) — read off model-00001-of-00013.safetensors.
  SYNTHETIC: hidden_states (block-0 input) and temb (shared timestep
             embedding, same temb both layers read — matches production:
             one time_embedder output, every block's own adaln_proj reads
             it). adaln_proj for BOTH blocks is left at its diffusers
             default random init (never loaded from the checkpoint here —
             see the block/modcache oracles' headers for why that is the
             correct thing to test, not a shortcut) — this script captures
             EACH layer's own adaln_proj(temb) output and dumps it as that
             layer's modulation table, exactly what a real
             MiniMaxH3ModCache.block_mod[layer] holds.
  NOT TESTED HERE: modcache construction itself (minimax_h3_build_modulation_
             cache) — already verified separately
             (minimax_h3_modcache_real_weight_gate.mojo). This oracle's
             dumped mod tensors are fed DIRECTLY into a hand-built
             MiniMaxH3ModCache on the Mojo side (h3-block's suggestion), so
             this gate isolates STACK COMPOSITION + BLOCK MATH across two
             real-weight layers, not modcache-building as a side effect.

ORACLE: diffusers' OWN MiniMaxH3TransformerBlock, instantiated TWICE (one
per real layer), each loaded with that layer's real weights, run
sequentially: h1 = block0(h0), h2 = block1(h1). GPU, bf16.

Run: python3 serenitymojo/models/dit/parity/minimax_h3_stack_real_weight_oracle.py
Writes: output/minimax_h3_block/stack2_real_weight_ref.safetensors
"""

import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DIFFUSERS_SRC = "/home/alex/minimax_h3_ref/diffusers-src/src"
sys.path.insert(0, DIFFUSERS_SRC)

from diffusers.models.transformers.transformer_minimax_h3 import (  # noqa: E402
    MiniMaxH3RotaryPosEmbed,
    MiniMaxH3TransformerBlock,
)

CKPT_DIR = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/transformer"
SHARD = os.path.join(CKPT_DIR, "model-00001-of-00013.safetensors")
OUT_DIR = "/home/alex/mojodiffusion/output/minimax_h3_block"
OUT_PATH = os.path.join(OUT_DIR, "stack2_real_weight_ref.safetensors")

HIDDEN = 5376
HEADS = 56
HEAD_DIM = 128
INNER = HEADS * HEAD_DIM
FFN = 14336
TIME_EMBED_DIM = 2688
ROPE_FREQ_DIM = 16
ROPE_THETA = 10000.0
NORM_EPS = 1e-5
QK_NORM_EPS = 1e-5

NUM_LAYERS = 2  # blocks 0 and 1 -- the only fully-present real block bytes

NUM_TEXT_TOKENS = 4
NUM_AUDIO_TOKENS = 6
NUM_VIDEO_TOKENS = 8


def packed_layout():
    sequence_length = NUM_TEXT_TOKENS + NUM_AUDIO_TOKENS + NUM_VIDEO_TOKENS
    text_indices = torch.arange(NUM_TEXT_TOKENS)
    audio_indices = torch.arange(NUM_TEXT_TOKENS, NUM_TEXT_TOKENS + NUM_AUDIO_TOKENS)
    video_indices = torch.arange(NUM_TEXT_TOKENS + NUM_AUDIO_TOKENS, sequence_length)

    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = 1
    token_tags[audio_indices] = 2
    token_tags[video_indices] = 0

    timestep_indices = torch.zeros(sequence_length, dtype=torch.long)
    timestep_indices[audio_indices] = 1

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float32)
    position_ids[:, 0] = torch.arange(sequence_length, dtype=torch.float32)
    position_ids[video_indices, 1] = torch.arange(NUM_VIDEO_TOKENS, dtype=torch.float32) % 4
    position_ids[video_indices, 2] = torch.arange(NUM_VIDEO_TOKENS, dtype=torch.float32) % 2

    return sequence_length, token_tags, timestep_indices, position_ids


def deinterleave_qkv(raw: torch.Tensor) -> torch.Tensor:
    hidden = raw.shape[1]
    return raw.reshape(HEADS, 3, HEAD_DIM, hidden).permute(1, 0, 2, 3).reshape(3, INNER, hidden)


def swap_fc1(raw: torch.Tensor) -> torch.Tensor:
    return torch.cat([raw[FFN:], raw[:FFN]], dim=0)


def load_block(layer: int, device: str, dtype: torch.dtype) -> MiniMaxH3TransformerBlock:
    names = [
        f"blocks.{layer}.norm1.weight",
        f"blocks.{layer}.norm2.weight",
        f"blocks.{layer}.attn.qkv_proj.weight",
        f"blocks.{layer}.attn.q_norm.weight",
        f"blocks.{layer}.attn.k_norm.weight",
        f"blocks.{layer}.attn.out_proj.weight",
        f"blocks.{layer}.mlp.fc1.weight",
        f"blocks.{layer}.mlp.fc2.weight",
    ]
    raw = {}
    with safe_open(SHARD, framework="pt", device="cpu") as f:
        for n in names:
            t = f.get_tensor(n)
            assert t.dtype == torch.bfloat16, f"{n} is {t.dtype}"
            raw[n] = t.to(device=device)

    to_q, to_k, to_v = deinterleave_qkv(raw[f"blocks.{layer}.attn.qkv_proj.weight"])
    fc1_value_gate = swap_fc1(raw[f"blocks.{layer}.mlp.fc1.weight"])

    block = (
        MiniMaxH3TransformerBlock(
            hidden_size=HIDDEN,
            num_attention_heads=HEADS,
            attention_head_dim=HEAD_DIM,
            ffn_dim=FFN,
            time_embed_dim=TIME_EMBED_DIM,
            norm_eps=NORM_EPS,
            qk_norm_eps=QK_NORM_EPS,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    with torch.no_grad():
        block.norm1.weight.copy_(raw[f"blocks.{layer}.norm1.weight"])
        block.norm2.weight.copy_(raw[f"blocks.{layer}.norm2.weight"])
        block.attn.to_q.weight.copy_(to_q)
        block.attn.to_k.weight.copy_(to_k)
        block.attn.to_v.weight.copy_(to_v)
        block.attn.norm_q.weight.copy_(raw[f"blocks.{layer}.attn.q_norm.weight"])
        block.attn.norm_k.weight.copy_(raw[f"blocks.{layer}.attn.k_norm.weight"])
        block.attn.to_out[0].weight.copy_(raw[f"blocks.{layer}.attn.out_proj.weight"])
        block.ff.net[0].proj.weight.copy_(fc1_value_gate)
        block.ff.net[2].weight.copy_(raw[f"blocks.{layer}.mlp.fc2.weight"])
    return block


def main() -> None:
    device = "cuda"
    dtype = torch.bfloat16

    print(f"[1] loading {NUM_LAYERS} real MiniMaxH3TransformerBlocks from {SHARD}")
    blocks = [load_block(layer, device, dtype) for layer in range(NUM_LAYERS)]
    print(f"    {NUM_LAYERS}/{NUM_LAYERS} blocks loaded, real norm/attn/ff weights each")

    print("[2] packed layout + rope (same S=18 layout as the block-0 gate)")
    sequence_length, token_tags, timestep_indices, position_ids = packed_layout()
    num_timesteps = int(timestep_indices.max().item()) + 1
    adaln_indices = (timestep_indices * 3 + token_tags.clamp(min=0)).to(device=device)

    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA).to(device)
    cos, sin = rope(position_ids.to(device))
    rotary_dim = cos.shape[-1]

    gen = torch.Generator(device=device).manual_seed(7)
    hidden_states = torch.randn(
        1, sequence_length, HIDDEN, generator=gen, device=device, dtype=torch.float32
    ).to(dtype)
    gen2 = torch.Generator(device=device).manual_seed(11)
    # ONE temb, shared by both layers -- matches production: one time_embedder
    # output read by every block's own adaln_proj.
    temb = torch.randn(num_timesteps, TIME_EMBED_DIM, generator=gen2, device=device, dtype=torch.float32)

    print("[3] capturing each layer's own adaln_proj(temb) as that layer's modulation table")
    tensors = {
        "in.hidden_states": hidden_states.float().cpu().contiguous(),
        "in.cos": cos.float().cpu().contiguous(),
        "in.sin": sin.float().cpu().contiguous(),
        "in.adaln_indices": adaln_indices.to(torch.int64).cpu().contiguous(),
    }
    for layer in range(NUM_LAYERS):
        with torch.no_grad():
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = blocks[layer].adaln_proj(temb)
        mod = torch.cat([shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp], dim=-1).float()
        tensors[f"in.mod.{layer}"] = mod.cpu().contiguous()
        print(f"    layer {layer} mod {tuple(mod.shape)}")

    print(f"[4] running {NUM_LAYERS} real blocks SEQUENTIALLY (the stack composition), GPU bf16")
    h = hidden_states
    with torch.no_grad():
        for layer in range(NUM_LAYERS):
            h = blocks[layer](h, temb, adaln_indices, (cos, sin))
    print(f"    final {tuple(h.shape)} mean {h.float().mean().item():.6f} std {h.float().std().item():.6f}")

    tensors["out.hidden_states"] = h.float().cpu().contiguous()

    os.makedirs(OUT_DIR, exist_ok=True)
    meta = {
        "num_layers": str(NUM_LAYERS),
        "sequence_length": str(sequence_length),
        "hidden_size": str(HIDDEN),
        "num_attention_heads": str(HEADS),
        "attention_head_dim": str(HEAD_DIM),
        "rotary_dim": str(rotary_dim),
        "num_timesteps": str(num_timesteps),
        "shard": SHARD,
        "note": (
            "blocks 0/1 norm/attn/ff weights are REAL; hidden_states/temb are "
            "seeded random; each layer's adaln_proj is NOT loaded from the "
            "checkpoint (production contract) -- its own random-init output "
            "is captured per layer as in.mod.<layer>, matching what a real "
            "MiniMaxH3ModCache.block_mod[layer] holds"
        ),
    }
    save_file(tensors, OUT_PATH, metadata=meta)
    print(f"[5] wrote {OUT_PATH}")
    print("DONE")


if __name__ == "__main__":
    main()
