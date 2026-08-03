#!/usr/bin/env python3
"""MiniMax-H3 block-0 REAL-WEIGHT oracle (DEV-ONLY, never imported by Mojo).

Companion to scripts/minimax_h3_block_oracle.py (which gates the block math
on random weights at a TINY 2-head/16-head_dim geometry). This one gates the
SAME block math at the RELEASED geometry (56 heads x 128 head_dim, hidden
5376, ffn 14336) using REAL bf16 bytes read from the actual FL2VA checkpoint
shard on disk — the first time in this port that anything runs the model's
own math on the model's own weights, as opposed to an oracle or a fixture.

WHAT IS REAL vs SYNTHETIC:
  REAL:      the 8 checkpoint tensors block 0 needs (norm1, norm2,
             attn.qkv_proj, attn.q_norm, attn.k_norm, attn.out_proj,
             mlp.fc1, mlp.fc2) — read directly off
             model-00001-of-00013.safetensors, header-driven (safe_open +
             get_tensor per name), never loading the other 12 shards or the
             other 49 blocks.
  SYNTHETIC: hidden_states (block-0 input) and temb (timestep embedding) —
             both fixed-seed random activations, since gating BLOCK MATH
             does not require a real denoising trajectory (the tiny fixture
             does the same). adaln_proj is NOT loaded from the checkpoint —
             production never loads it into the block either (13.04B params
             over the stack, see minimax_h3_dit.mojo's header) — so this
             oracle leaves MiniMaxH3TransformerBlock's own adaln_proj at its
             default random init and captures ITS OWN output as the
             modulation table `mod`, which is exactly the block's real
             contract: `mod` is always an externally-supplied opaque input,
             never something the block computes from a loaded weight.

ORACLE: NOT a hand transcription of the math — this instantiates and runs
diffusers' OWN `MiniMaxH3TransformerBlock` and `MiniMaxH3RotaryPosEmbed`
classes directly (from the pinned diffusers checkout this whole port is
gated against, huggingface/diffusers#14355 @ e1b518df). That class's forward
is exactly what models/minimax_h3/block_forward.mojo::_transformer_block was
already gated against at max_abs 5.96e-08
(models/minimax_h3/parity/minimax_h3_block_parity.mojo) — running it directly
here removes even the risk of THIS script re-transcribing that math wrong.

WEIGHT-LAYOUT CONVERSION: the checkpoint's `attn.qkv_proj.weight` is
per-head-interleaved ([3*inner,hidden], q/k/v interleaved per head) and
`mlp.fc1.weight` is `[gate;value]` — the SAME two fusions
models/minimax_h3/loader.mojo (`minimax_h3_deinterleave_qkv`/
`minimax_h3_swap_fc1`) and models/dit/minimax_h3_loader_device.mojo
(`minimax_h3_deinterleave_qkv_bf16`/`minimax_h3_swap_fc1_bf16`) undo before
handing weights to minimax_h3_block_forward. `deinterleave_qkv`/`swap_fc1`
below apply the IDENTICAL index math in torch so this oracle's diffusers
call sees exactly what the Mojo device path's REAL loader also produces —
the Mojo side loads through `minimax_h3_load_block_device` on the same
shard, not through anything this script writes.

DTYPE: real weights loaded and kept BF16 throughout — GPU bf16, matching the
house rule (no CPU-fp32-vs-GPU-bf16 comparison, which diverges too much to
be a useful reference). hidden_states is generated fp32 then cast to bf16
(clean seeded RNG); temb stays fp32 (mirrors the real model: time_embedder
is a fp32 module, only its silu(temb) result gets cast down for the AdaLN
matmul — see MiniMaxH3AdaLayerNormModulation.forward's own comment).

Run: python3 serenitymojo/models/dit/parity/minimax_h3_block_real_weight_oracle.py
Writes: output/minimax_h3_block/block0_real_weight_ref.safetensors
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
OUT_PATH = os.path.join(OUT_DIR, "block0_real_weight_ref.safetensors")

# Released FL2VA/transformer/config.json geometry — same numbers
# minimax_h3_released_config() (minimax_h3_dit.mojo) hardcodes.
HIDDEN = 5376
HEADS = 56
HEAD_DIM = 128
INNER = HEADS * HEAD_DIM  # 7168
FFN = 14336
TIME_EMBED_DIM = 2688
ROPE_FREQ_DIM = 16
ROPE_THETA = 10000.0
NORM_EPS = 1e-5
QK_NORM_EPS = 1e-5

# Identical packed layout to the tiny fixture (scripts/minimax_h3_block_oracle.py
# packed_layout()) — S=18, two distinct timesteps — so weights/geometry are the
# ONLY variable that changed relative to that gate, not the packing.
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
    """checkpoint per-head-interleaved [3*inner, hidden] -> diffusers split
    [3, inner, hidden] (to_q, to_k, to_v). Mirrors minimax_h3_deinterleave_qkv's
    index math exactly: source_row = h*3*head_dim + part*head_dim + d, which is
    row-major [heads,3,head_dim]; the target dest_row = part*heads*head_dim +
    h*head_dim + d is row-major [3,heads,head_dim] — a permute of the same
    axes, not a data-dependent gather."""
    hidden = raw.shape[1]
    return raw.reshape(HEADS, 3, HEAD_DIM, hidden).permute(1, 0, 2, 3).reshape(3, INNER, hidden)


def swap_fc1(raw: torch.Tensor) -> torch.Tensor:
    """checkpoint [gate;value] ([2*ffn,hidden], gate rows first) -> diffusers
    [value;gate] (SwiGLU.forward chunks its proj output as (value, gate) in
    that order — activations.py:145, "hidden_states, gate = ...chunk(2)")."""
    return torch.cat([raw[FFN:], raw[:FFN]], dim=0)


def main() -> None:
    device = "cuda"
    dtype = torch.bfloat16

    print(f"[1] reading block-0's 8 tensors from {SHARD} (header-driven, no full-shard load)")
    names = [
        "blocks.0.norm1.weight",
        "blocks.0.norm2.weight",
        "blocks.0.attn.qkv_proj.weight",
        "blocks.0.attn.q_norm.weight",
        "blocks.0.attn.k_norm.weight",
        "blocks.0.attn.out_proj.weight",
        "blocks.0.mlp.fc1.weight",
        "blocks.0.mlp.fc2.weight",
    ]
    raw = {}
    with safe_open(SHARD, framework="pt", device="cpu") as f:
        for n in names:
            t = f.get_tensor(n)
            assert t.dtype == torch.bfloat16, f"{n} is {t.dtype}, expected bf16"
            raw[n] = t.to(device=device)
    for n, t in raw.items():
        print(f"    {n:<32} {tuple(t.shape)} {t.dtype}")

    to_q, to_k, to_v = deinterleave_qkv(raw["blocks.0.attn.qkv_proj.weight"])
    fc1_value_gate = swap_fc1(raw["blocks.0.mlp.fc1.weight"])

    print("[2] building diffusers MiniMaxH3TransformerBlock at released geometry")
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
    # adaln_proj is left at its (random, bf16-cast) default init — see module
    # docstring: production never loads it into the block either, and its
    # output is captured below as an opaque `mod` input, matching the real
    # contract exactly.
    with torch.no_grad():
        block.norm1.weight.copy_(raw["blocks.0.norm1.weight"])
        block.norm2.weight.copy_(raw["blocks.0.norm2.weight"])
        block.attn.to_q.weight.copy_(to_q)
        block.attn.to_k.weight.copy_(to_k)
        block.attn.to_v.weight.copy_(to_v)
        block.attn.norm_q.weight.copy_(raw["blocks.0.attn.q_norm.weight"])
        block.attn.norm_k.weight.copy_(raw["blocks.0.attn.k_norm.weight"])
        block.attn.to_out[0].weight.copy_(raw["blocks.0.attn.out_proj.weight"])
        block.ff.net[0].proj.weight.copy_(fc1_value_gate)
        block.ff.net[2].weight.copy_(raw["blocks.0.mlp.fc2.weight"])
    print("    9/9 real-weight submodules loaded (adaln_proj intentionally left random)")

    print("[3] packed layout + rope + synthetic activations (same S=18 layout as the tiny fixture)")
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
    temb = torch.randn(num_timesteps, TIME_EMBED_DIM, generator=gen2, device=device, dtype=torch.float32)

    print("[4] capturing this block's own AdaLN modulation table (adaln_proj(temb))")
    with torch.no_grad():
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(temb)
    mod = torch.cat([shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp], dim=-1).to(torch.float32)

    print("[5] running the REAL diffusers MiniMaxH3TransformerBlock.forward, GPU bf16")
    with torch.no_grad():
        out = block(hidden_states, temb, adaln_indices, (cos, sin))
    print(f"    out {tuple(out.shape)} mean {out.float().mean().item():.6f} std {out.float().std().item():.6f}")

    print(f"[6] writing {OUT_PATH}")
    os.makedirs(OUT_DIR, exist_ok=True)
    tensors = {
        "in.hidden_states": hidden_states.float().cpu().contiguous(),
        "in.mod": mod.cpu().contiguous(),
        "in.cos": cos.float().cpu().contiguous(),
        "in.sin": sin.float().cpu().contiguous(),
        "in.adaln_indices": adaln_indices.to(torch.int64).cpu().contiguous(),
        "out.hidden_states": out.float().cpu().contiguous(),
    }
    meta = {
        "sequence_length": str(sequence_length),
        "hidden_size": str(HIDDEN),
        "num_attention_heads": str(HEADS),
        "attention_head_dim": str(HEAD_DIM),
        "ffn_dim": str(FFN),
        "rotary_dim": str(rotary_dim),
        "num_timesteps": str(num_timesteps),
        "shard": SHARD,
        "note": (
            "weights (norm1/norm2/attn.*/ff.*) are REAL MiniMax-H3 FL2VA block-0 "
            "bf16 bytes; hidden_states/temb are seeded random activations, not "
            "from a real denoising trajectory; adaln_proj is NOT loaded "
            "(production contract) — its own random-init output is captured as "
            "in.mod"
        ),
    }
    save_file(tensors, OUT_PATH, metadata=meta)
    print("DONE")


if __name__ == "__main__":
    main()
