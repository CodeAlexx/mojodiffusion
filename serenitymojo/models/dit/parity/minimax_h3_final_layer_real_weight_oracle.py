#!/usr/bin/env python3
"""MiniMax-H3 FINAL-LAYER REAL-WEIGHT oracle (DEV-ONLY, never imported by
Mojo). Shard 13 (the last missing shard) landed, so this is the first time
anything in this port has run real `final_layer.*` bytes.

Verifies TWO things, both against diffusers' OWN classes/layers (never a
hand transcription), GPU bf16 where the checkpoint stores bf16, F32 where it
stores F32:

  [1] the "final modulation" PROJECTION alone: `final_layer.adaln_proj.linear`
      applied to a shared temb — the piece `minimax_h3_build_modulation_cache`
      computes as `final_mod` (models/dit/minimax_h3_modcache.mojo), which
      this port's modcache gate previously had to stub with random bytes
      (shard 13 didn't exist yet). Isolated by calling
      `MiniMaxH3AdaLayerNormOut.linear` directly rather than its whole
      `forward` (which also needs hidden_states/timestep_indices — those
      belong to check [2]).
  [2] the FULL final-layer path: diffusers' own
      `MiniMaxH3AdaLayerNormOut.forward(hidden_states, temb, timestep_indices)`
      (real norm.weight + real adaln_proj bytes) — RMSNorm, per-row
      timestep-indexed shift/scale, both in BF16 (temb.dtype()==bf16 output
      here, matching the checkpoint's bf16-stored adaln_proj/norm) — THEN
      the model's own `.to(proj_out.weight.dtype)` cast to F32
      (transformer_minimax_h3.py:638) BEFORE the two REAL F32 output heads
      (`proj_out`/`audio_proj_out`, i.e. `final_layer.video_out`/
      `final_layer.audio_out` in checkpoint naming).

WHY THE CAST ORDER MATTERS (this is the specific risk team-lead flagged):
diffusers does ALL of RMSNorm + shift/scale modulation in BF16 and casts to
F32 ONLY AFTER modulation, right before the F32 output heads (line 638:
`hidden_states = self.norm_out(...).to(self.proj_out.weight.dtype)`). If
`minimax_h3_final_layer` instead upcasts to F32 BEFORE combining with the
modulation table (its own code casts `final_normed` to F32 right after the
RMSNorm, before the shift/scale add), it is doing different arithmetic —
modulation in F32 instead of BF16 — even if every shape and every weight
byte is identical. This oracle's dump lets the Mojo gate catch that
difference directly rather than asserting it away.

WHAT IS REAL vs SYNTHETIC:
  REAL:      final_layer.adaln_proj.linear.{weight,bias} (BF16),
             final_layer.norm.weight (BF16), final_layer.video_out.{weight,bias}
             (F32, NOT downcast), final_layer.audio_out.{weight,bias} (F32,
             NOT downcast) -- all read directly off
             model-00013-of-00013.safetensors.
  SYNTHETIC: hidden_states (the "50-block stack output" this layer consumes),
             temb, and the video/audio/timestep index bookkeeping -- gating
             this layer's math does not need a real denoising trajectory,
             same reasoning as every other real-weight oracle in this port.

Run: python3 serenitymojo/models/dit/parity/minimax_h3_final_layer_real_weight_oracle.py
Writes: output/minimax_h3_block/final_layer_real_weight_ref.safetensors
"""

import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DIFFUSERS_SRC = "/home/alex/minimax_h3_ref/diffusers-src/src"
sys.path.insert(0, DIFFUSERS_SRC)

from diffusers.models.transformers.transformer_minimax_h3 import (  # noqa: E402
    MiniMaxH3AdaLayerNormOut,
)

CKPT_DIR = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/transformer"
SHARD13 = os.path.join(CKPT_DIR, "model-00013-of-00013.safetensors")
OUT_DIR = "/home/alex/mojodiffusion/output/minimax_h3_block"
OUT_PATH = os.path.join(OUT_DIR, "final_layer_real_weight_ref.safetensors")

HIDDEN = 5376
TIME_EMBED_DIM = 2688
VIDEO_PATCH_DIM = 96   # in_channels(24) * patch(1*2*2)
AUDIO_CHANNELS = 32
FINAL_NORM_EPS = 1e-5

NUM_TEXT_TOKENS = 4
NUM_AUDIO_TOKENS = 6
NUM_VIDEO_TOKENS = 8


def packed_layout():
    """Same packed layout as the other real-weight oracles (S=18, two
    distinct timesteps) — this layer only needs timestep_indices/video_indices
    /audio_indices, not token_tags (the final AdaLN has no modality axis)."""
    sequence_length = NUM_TEXT_TOKENS + NUM_AUDIO_TOKENS + NUM_VIDEO_TOKENS
    text_indices = torch.arange(NUM_TEXT_TOKENS)
    audio_indices = torch.arange(NUM_TEXT_TOKENS, NUM_TEXT_TOKENS + NUM_AUDIO_TOKENS)
    video_indices = torch.arange(NUM_TEXT_TOKENS + NUM_AUDIO_TOKENS, sequence_length)

    timestep_indices = torch.zeros(sequence_length, dtype=torch.long)
    timestep_indices[audio_indices] = 1
    return sequence_length, timestep_indices, video_indices, audio_indices


def main() -> None:
    device = "cuda"

    print(f"[1] reading final_layer.* real bytes from {SHARD13}")
    with safe_open(SHARD13, framework="pt", device="cpu") as f:
        adaln_w = f.get_tensor("final_layer.adaln_proj.linear.weight").to(device=device)
        adaln_b = f.get_tensor("final_layer.adaln_proj.linear.bias").to(device=device)
        norm_w = f.get_tensor("final_layer.norm.weight").to(device=device)
        video_out_w = f.get_tensor("final_layer.video_out.weight").to(device=device)
        video_out_b = f.get_tensor("final_layer.video_out.bias").to(device=device)
        audio_out_w = f.get_tensor("final_layer.audio_out.weight").to(device=device)
        audio_out_b = f.get_tensor("final_layer.audio_out.bias").to(device=device)

    assert adaln_w.dtype == torch.bfloat16 and norm_w.dtype == torch.bfloat16
    assert video_out_w.dtype == torch.float32 and audio_out_w.dtype == torch.float32
    print(f"    adaln_proj.linear.weight {tuple(adaln_w.shape)} {adaln_w.dtype}")
    print(f"    norm.weight              {tuple(norm_w.shape)} {norm_w.dtype}")
    print(f"    video_out.weight         {tuple(video_out_w.shape)} {video_out_w.dtype}  (F32, NOT downcast)")
    print(f"    audio_out.weight         {tuple(audio_out_w.shape)} {audio_out_w.dtype}  (F32, NOT downcast)")

    print("[2] building diffusers' own MiniMaxH3AdaLayerNormOut + the two real F32 output heads")
    norm_out = MiniMaxH3AdaLayerNormOut(hidden_size=HIDDEN, time_embed_dim=TIME_EMBED_DIM, eps=FINAL_NORM_EPS)
    norm_out = norm_out.to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        norm_out.norm.weight.copy_(norm_w)
        norm_out.linear.weight.copy_(adaln_w)
        norm_out.linear.bias.copy_(adaln_b)

    proj_out = torch.nn.Linear(HIDDEN, VIDEO_PATCH_DIM, bias=True).to(device=device, dtype=torch.float32)
    audio_proj_out = torch.nn.Linear(HIDDEN, AUDIO_CHANNELS, bias=True).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        proj_out.weight.copy_(video_out_w)
        proj_out.bias.copy_(video_out_b)
        audio_proj_out.weight.copy_(audio_out_w)
        audio_proj_out.bias.copy_(audio_out_b)

    print("[3] packed layout + synthetic stack-output activations")
    sequence_length, timestep_indices, video_indices, audio_indices = packed_layout()
    num_timesteps = int(timestep_indices.max().item()) + 1
    timestep_indices = timestep_indices.to(device)

    gen = torch.Generator(device=device).manual_seed(31)
    hidden_states = torch.randn(
        sequence_length, HIDDEN, generator=gen, device=device, dtype=torch.float32
    ).to(torch.bfloat16)
    gen2 = torch.Generator(device=device).manual_seed(37)
    temb = torch.randn(num_timesteps, TIME_EMBED_DIM, generator=gen2, device=device, dtype=torch.float32)

    print("[4] isolating the modcache-equivalent projection alone (final_mod BEFORE norm/modulate)")
    with torch.no_grad():
        activated = torch.nn.functional.silu(temb).to(norm_out.linear.weight.dtype)
        final_mod_ref = norm_out.linear(activated)  # [num_timesteps, 2*hidden], BF16 (matches checkpoint storage)
    print(f"    final_mod {tuple(final_mod_ref.shape)} {final_mod_ref.dtype} mean {final_mod_ref.float().mean().item():.6f}")

    print("[5] running the REAL diffusers final-layer chain: norm_out(...).to(F32) -> proj_out / audio_proj_out")
    with torch.no_grad():
        modulated_bf16 = norm_out(hidden_states, temb, timestep_indices)  # BF16 (matches transformer_minimax_h3.py:638's PRE-cast value)
        modulated_f32 = modulated_bf16.to(proj_out.weight.dtype)          # the EXPLICIT cast at line 638, AFTER modulation
        video_all = proj_out(modulated_f32)
        audio_all = audio_proj_out(modulated_f32)
        video_out = video_all.index_select(0, video_indices.to(device))
        audio_out = audio_all.index_select(0, audio_indices.to(device))
    print(f"    video_out {tuple(video_out.shape)} mean {video_out.mean().item():.6f} std {video_out.std().item():.6f}")
    print(f"    audio_out {tuple(audio_out.shape)} mean {audio_out.mean().item():.6f} std {audio_out.std().item():.6f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    tensors = {
        "in.hidden_states": hidden_states.float().cpu().contiguous(),
        "in.temb": temb.cpu().contiguous(),
        "in.timestep_indices": timestep_indices.to(torch.int64).cpu().contiguous(),
        "in.video_indices": video_indices.to(torch.int64).cpu().contiguous(),
        "in.audio_indices": audio_indices.to(torch.int64).cpu().contiguous(),
        "out.final_mod": final_mod_ref.float().cpu().contiguous(),
        "out.modulated_bf16_then_f32": modulated_f32.cpu().contiguous(),
        "out.video_out": video_out.cpu().contiguous(),
        "out.audio_out": audio_out.cpu().contiguous(),
    }
    meta = {
        "sequence_length": str(sequence_length),
        "hidden_size": str(HIDDEN),
        "time_embed_dim": str(TIME_EMBED_DIM),
        "video_patch_dim": str(VIDEO_PATCH_DIM),
        "audio_channels": str(AUDIO_CHANNELS),
        "num_timesteps": str(num_timesteps),
        "shard": SHARD13,
        "note": (
            "final_layer.adaln_proj.linear.*/norm.weight are REAL bf16 bytes; "
            "video_out/audio_out are REAL f32 bytes; hidden_states/temb are "
            "seeded random; modulation math (shift/scale combine) happens in "
            "BF16 here, matching transformer_minimax_h3.py -- F32 cast is "
            "AFTER modulation (line 638), not before"
        ),
    }
    save_file(tensors, OUT_PATH, metadata=meta)
    print(f"[6] wrote {OUT_PATH}")
    print("DONE")


if __name__ == "__main__":
    main()
