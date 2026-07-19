#!/usr/bin/env python3
"""LTX-2.3 22B AV transformer block-0 ground-truth oracle.

Uses the REAL musubi BasicAVTransformerBlock._forward with real block-0 weights
from ltx-2.3-22b-dev.safetensors.  This is the oracle the Mojo AV block is
measured against — NOT diffusers, NOT the FP8 distilled ckpt, NOT a hand-ported
reimplementation.

CKPT strategy: load ONLY the 86 block-0 tensors (prefix
  model.diffusion_model.transformer_blocks.0.)
via safe_open + get_tensor, keeping the 46 GB checkpoint file nearly closed.
This avoids the full-model stall that killed the previous agent.

Block-0 config (from model_configurator.py defaults and weight-shape inspection):
  video:  dim=4096, heads=32, d_head=128, context_dim=4096,
          apply_gated_attention=True, cross_attention_adaln=True  → scale_shift_table [9,4096]
  audio:  dim=2048, heads=32, d_head=64,  context_dim=2048,
          apply_gated_attention=True, cross_attention_adaln=True  → audio_scale_shift_table [9,2048]
  A2V cross-modal: audio_to_video_attn (Q=video/4096, KV=audio/2048, out=4096)
  V2A cross-modal: video_to_audio_attn (Q=audio/2048, KV=video/4096, out=2048)

Run:
  /home/alex/musubi-tuner/venv/bin/python scripts/ltx2_av_oracle.py

Outputs:
  output/ltx2_av/block0_ref.safetensors   — all inputs + outputs as float32
  output/ltx2_av/block0_ref_meta.json     — shapes, dtypes, seed, config, sublayer order
"""

import json
import os
import struct
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev.safetensors"
OUT_DIR = "/home/alex/mojodiffusion/output/ltx2_av"
OUT_ST = os.path.join(OUT_DIR, "block0_ref.safetensors")
OUT_META = os.path.join(OUT_DIR, "block0_ref_meta.json")

BLOCK0_PREFIX = "model.diffusion_model.transformer_blocks.0."

# ---------------------------------------------------------------------------
# Block-0 config (confirmed from weight shapes above)
# ---------------------------------------------------------------------------
VIDEO_DIM       = 4096
VIDEO_HEADS     = 32
VIDEO_HEAD_DIM  = 128        # 4096 / 32
VIDEO_CTX_DIM   = 4096       # attn2 context (same as video dim in this model)

AUDIO_DIM       = 2048
AUDIO_HEADS     = 32
AUDIO_HEAD_DIM  = 64         # 2048 / 32
AUDIO_CTX_DIM   = 2048       # audio_attn2 context

NORM_EPS        = 1e-6
APPLY_GATED     = True       # to_gate_logits present in block-0
CA_ADALN        = True       # scale_shift_table [9,*] → cross_attention_adaln=True

# adaln_embedding_coefficient = 6 + 3 (CA) = 9
N_ADA_PARAMS_VIDEO = 9
N_ADA_PARAMS_AUDIO = 9

# Synthetic input sizes (small so the forward is fast, large enough to be
# non-degenerate and exercise all code paths)
S_V    = 64    # video tokens
S_A    = 32    # audio tokens
N_TXT  = 16    # text context length
B      = 1     # batch size = 1
SEED   = 0     # deterministic

# ---------------------------------------------------------------------------
# musubi sys.path setup
# ---------------------------------------------------------------------------
MUSUBI_SRC = "/home/alex/musubi-tuner/src"
if MUSUBI_SRC not in sys.path:
    sys.path.insert(0, MUSUBI_SRC)

# ---------------------------------------------------------------------------
# Load block-0 weights — header-first, pull only block-0 tensors
# ---------------------------------------------------------------------------

def load_block0_weights() -> dict[str, torch.Tensor]:
    """Pull only the 86 block-0 tensors from the 46 GB checkpoint."""
    print(f"[oracle] scanning header of {os.path.basename(CKPT)} ...")
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        all_keys = list(f.keys())
        block0_keys = [k for k in all_keys if k.startswith(BLOCK0_PREFIX)]
        print(f"[oracle] found {len(block0_keys)} block-0 keys (total keys: {len(all_keys)})")
        weights = {}
        for k in sorted(block0_keys):
            t = f.get_tensor(k)
            short = k[len(BLOCK0_PREFIX):]
            weights[short] = t
            print(f"  {short}: {list(t.shape)} {t.dtype}")
    return weights


# ---------------------------------------------------------------------------
# Build musubi BasicAVTransformerBlock and load the real block-0 weights
# ---------------------------------------------------------------------------

def build_block(weights: dict[str, torch.Tensor]) -> "BasicAVTransformerBlock":
    from musubi_tuner.ltx_2.model.transformer.transformer import (
        BasicAVTransformerBlock,
        TransformerConfig,
    )
    from musubi_tuner.ltx_2.model.transformer.attention import AttentionFunction
    from musubi_tuner.ltx_2.model.transformer.rope import LTXRopeType

    video_cfg = TransformerConfig(
        dim=VIDEO_DIM,
        heads=VIDEO_HEADS,
        d_head=VIDEO_HEAD_DIM,
        context_dim=VIDEO_CTX_DIM,
        apply_gated_attention=APPLY_GATED,
        cross_attention_adaln=CA_ADALN,
    )
    audio_cfg = TransformerConfig(
        dim=AUDIO_DIM,
        heads=AUDIO_HEADS,
        d_head=AUDIO_HEAD_DIM,
        context_dim=AUDIO_CTX_DIM,
        apply_gated_attention=APPLY_GATED,
        cross_attention_adaln=CA_ADALN,
    )

    block = BasicAVTransformerBlock(
        idx=0,
        video=video_cfg,
        audio=audio_cfg,
        rope_type=LTXRopeType.INTERLEAVED,
        norm_eps=NORM_EPS,
        # Force pytorch attention so there's no xformers/flash dependency
        attention_function=AttentionFunction.PYTORCH,
    )

    # Build a state dict from the block-0 weights.  The keys in `weights` are
    # already the short names (without the global prefix), which match exactly
    # what PyTorch's state_dict() would produce for this block.
    #
    # Cast everything to float32 for the oracle math (block-0 is bf16/f32 in
    # the checkpoint; operating in f32 gives a numerically clean reference).
    sd = {k: v.to(torch.float32) for k, v in weights.items()}
    missing, unexpected = block.load_state_dict(sd, strict=True)
    if missing:
        print(f"[oracle] WARNING: missing keys: {missing}")
    if unexpected:
        print(f"[oracle] WARNING: unexpected keys: {unexpected}")
    block.eval()
    block = block.to(torch.float32)
    return block


# ---------------------------------------------------------------------------
# Build synthetic but non-degenerate TransformerArgs
# ---------------------------------------------------------------------------

def make_inputs() -> tuple[dict, "TransformerArgs", "TransformerArgs"]:
    """Return (raw_tensors_for_saving, video_args, audio_args).

    TransformerArgs are assembled manually without going through the full
    model preprocessor (which requires patchify, adaln, RoPE etc.) — instead
    we inject the pre-computed tensors directly.  This is valid because
    BasicAVTransformerBlock._forward only reads:
      .x, .context, .context_mask, .timesteps, .prompt_timestep,
      .positional_embeddings, .cross_positional_embeddings,
      .cross_scale_shift_timestep, .cross_gate_timestep,
      .enabled, .self_attention_mask, .a2v_cross_attention_mask,
      .v2a_cross_attention_mask
    """
    from dataclasses import replace as dc_replace
    from musubi_tuner.ltx_2.model.transformer.transformer_args import TransformerArgs

    g = torch.Generator(device="cpu").manual_seed(SEED)

    def rnd(*shape, scale=0.1):
        return torch.randn(*shape, generator=g) * scale

    # --- video stream ---
    # x: [B, S_V, VIDEO_DIM]
    video_x = rnd(B, S_V, VIDEO_DIM)

    # context (text): [B, N_TXT, VIDEO_CTX_DIM]
    video_ctx = rnd(B, N_TXT, VIDEO_CTX_DIM)

    # timesteps: the block's get_ada_values() expects
    #   timestep.reshape(B, T, N_ADA_PARAMS, dim)[:, :, indices, :]
    # where T is num tokens in the timestep batch (broadcast).
    # For simplicity, we pass a per-sample (T=1) timestep that gets broadcast
    # to S_V tokens via the repeat_interleave path in get_ada_values.
    # Shape: [B, 1, N_ADA_PARAMS * VIDEO_DIM]
    video_timesteps = rnd(B, 1, N_ADA_PARAMS_VIDEO * VIDEO_DIM, scale=0.05)

    # prompt_timestep (for cross_attention_adaln KV modulation): [B, N_TXT, 2*dim]
    # This is the per-context-token modulation; produced by prompt_adaln in the real model.
    video_prompt_ts = rnd(B, N_TXT, 2 * VIDEO_DIM, scale=0.05)

    # positional_embeddings (RoPE, interleaved):
    # musubi apply_rotary_emb expects freqs_cis = (cos, sin) where each is
    # [B, S, inner_dim] or broadcastable. The Attention.forward receives pe
    # and passes it to apply_rotary_emb(q, pe, rope_type).
    # For interleaved RoPE, input is (cos_freqs, sin_freqs) with shape
    # matching q: [B*H or B, S, d] — the exact shape is arbitrary for the
    # oracle as long as it's consistent. We use [B, S_V, VIDEO_DIM] (same as q).
    video_cos = rnd(B, S_V, VIDEO_DIM, scale=0.3)
    video_sin = rnd(B, S_V, VIDEO_DIM, scale=0.3)
    video_pe = (video_cos, video_sin)

    # cross_positional_embeddings for the A2V/V2A cross-modal attention.
    # Shape should match the audio cross-attention dim (inner = AUDIO_HEADS * AUDIO_HEAD_DIM = 2048)
    # projected to [B, S_V, AUDIO_DIM] for video side.
    av_cross_cos_v = rnd(B, S_V, AUDIO_DIM, scale=0.3)
    av_cross_sin_v = rnd(B, S_V, AUDIO_DIM, scale=0.3)
    video_cross_pe = (av_cross_cos_v, av_cross_sin_v)

    # cross_scale_shift_timestep: used in get_av_ca_ada_values() for video stream.
    # Shape: [B, 1, 4 * VIDEO_DIM] (4-param scale+shift for a2v and v2a)
    video_cross_ss_ts = rnd(B, 1, 4 * VIDEO_DIM, scale=0.05)

    # cross_gate_timestep: [B, 1, VIDEO_DIM]
    video_cross_gate_ts = rnd(B, 1, VIDEO_DIM, scale=0.05)

    # --- audio stream ---
    audio_x = rnd(B, S_A, AUDIO_DIM)
    audio_ctx = rnd(B, N_TXT, AUDIO_CTX_DIM)
    audio_timesteps = rnd(B, 1, N_ADA_PARAMS_AUDIO * AUDIO_DIM, scale=0.05)
    audio_prompt_ts = rnd(B, N_TXT, 2 * AUDIO_DIM, scale=0.05)

    audio_cos = rnd(B, S_A, AUDIO_DIM, scale=0.3)
    audio_sin = rnd(B, S_A, AUDIO_DIM, scale=0.3)
    audio_pe = (audio_cos, audio_sin)

    av_cross_cos_a = rnd(B, S_A, AUDIO_DIM, scale=0.3)
    av_cross_sin_a = rnd(B, S_A, AUDIO_DIM, scale=0.3)
    audio_cross_pe = (av_cross_cos_a, av_cross_sin_a)

    # audio cross_scale_shift_timestep: [B, 1, 4 * AUDIO_DIM]
    audio_cross_ss_ts = rnd(B, 1, 4 * AUDIO_DIM, scale=0.05)
    # audio cross_gate_timestep: [B, 1, AUDIO_DIM]
    audio_cross_gate_ts = rnd(B, 1, AUDIO_DIM, scale=0.05)

    # Build TransformerArgs (frozen dataclass)
    video_args = TransformerArgs(
        x=video_x,
        context=video_ctx,
        context_mask=None,           # no padding mask for the oracle
        timesteps=video_timesteps,
        embedded_timestep=video_timesteps,  # not used by block._forward
        positional_embeddings=video_pe,
        cross_positional_embeddings=video_cross_pe,
        cross_scale_shift_timestep=video_cross_ss_ts,
        cross_gate_timestep=video_cross_gate_ts,
        enabled=True,
        prompt_timestep=video_prompt_ts,
        self_attention_mask=None,
        a2v_cross_attention_mask=None,
        v2a_cross_attention_mask=None,
    )

    audio_args = TransformerArgs(
        x=audio_x,
        context=audio_ctx,
        context_mask=None,
        timesteps=audio_timesteps,
        embedded_timestep=audio_timesteps,
        positional_embeddings=audio_pe,
        cross_positional_embeddings=audio_cross_pe,
        cross_scale_shift_timestep=audio_cross_ss_ts,
        cross_gate_timestep=audio_cross_gate_ts,
        enabled=True,
        prompt_timestep=audio_prompt_ts,
        self_attention_mask=None,
        a2v_cross_attention_mask=None,
        v2a_cross_attention_mask=None,
    )

    # Collect all raw tensors for saving (flattened; tuples split to _cos/_sin)
    raw = {
        "video_x":              video_x,
        "video_ctx":            video_ctx,
        "video_timesteps":      video_timesteps,
        "video_prompt_ts":      video_prompt_ts,
        "video_pe_cos":         video_cos,
        "video_pe_sin":         video_sin,
        "video_cross_pe_cos":   av_cross_cos_v,
        "video_cross_pe_sin":   av_cross_sin_v,
        "video_cross_ss_ts":    video_cross_ss_ts,
        "video_cross_gate_ts":  video_cross_gate_ts,
        "audio_x":              audio_x,
        "audio_ctx":            audio_ctx,
        "audio_timesteps":      audio_timesteps,
        "audio_prompt_ts":      audio_prompt_ts,
        "audio_pe_cos":         audio_cos,
        "audio_pe_sin":         audio_sin,
        "audio_cross_pe_cos":   av_cross_cos_a,
        "audio_cross_pe_sin":   av_cross_sin_a,
        "audio_cross_ss_ts":    audio_cross_ss_ts,
        "audio_cross_gate_ts":  audio_cross_gate_ts,
    }

    return raw, video_args, audio_args


# ---------------------------------------------------------------------------
# Run block._forward and collect outputs
# ---------------------------------------------------------------------------

def run_oracle():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Load block-0 weights
    weights = load_block0_weights()

    # 2. Build the real musubi BasicAVTransformerBlock with real weights
    print("[oracle] building BasicAVTransformerBlock with real block-0 weights ...")
    block = build_block(weights)
    print(f"[oracle] block parameters: {sum(p.numel() for p in block.parameters()):,}")

    # 3. Build synthetic inputs
    print("[oracle] building synthetic inputs ...")
    raw_inputs, video_args, audio_args = make_inputs()

    # 4. Forward pass through the REAL musubi _forward (not .forward — no checkpointing)
    print("[oracle] running block._forward ...")
    with torch.no_grad():
        video_out_args, audio_out_args = block._forward(video_args, audio_args, perturbations=None)

    video_out = video_out_args.x   # [B, S_V, VIDEO_DIM]
    audio_out = audio_out_args.x   # [B, S_A, AUDIO_DIM]

    print(f"[oracle] video_out: {tuple(video_out.shape)} "
          f"mean={float(video_out.mean()):.4f} "
          f"std={float(video_out.std()):.4f} "
          f"absmax={float(video_out.abs().max()):.4f}")
    print(f"[oracle] audio_out: {tuple(audio_out.shape)} "
          f"mean={float(audio_out.mean()):.4f} "
          f"std={float(audio_out.std()):.4f} "
          f"absmax={float(audio_out.abs().max()):.4f}")

    # Sanity: check finite
    if not torch.isfinite(video_out).all():
        print("[oracle] ERROR: video_out contains non-finite values!")
        sys.exit(1)
    if not torch.isfinite(audio_out).all():
        print("[oracle] ERROR: audio_out contains non-finite values!")
        sys.exit(1)

    # 5. Assemble output dict — everything as float32, contiguous CPU tensors
    save_dict: dict[str, torch.Tensor] = {}
    for k, v in raw_inputs.items():
        save_dict[k] = v.detach().to(torch.float32).cpu().contiguous()
    save_dict["video_out"] = video_out.detach().to(torch.float32).cpu().contiguous()
    save_dict["audio_out"] = audio_out.detach().to(torch.float32).cpu().contiguous()

    # Also save the block-0 modulation tables (useful for the Mojo builder to
    # understand the adaln decomposition)
    save_dict["block0_scale_shift_table"] = (
        weights["scale_shift_table"].detach().to(torch.float32).cpu().contiguous()
    )
    save_dict["block0_audio_scale_shift_table"] = (
        weights["audio_scale_shift_table"].detach().to(torch.float32).cpu().contiguous()
    )
    save_dict["block0_scale_shift_table_a2v_ca_video"] = (
        weights["scale_shift_table_a2v_ca_video"].detach().to(torch.float32).cpu().contiguous()
    )
    save_dict["block0_scale_shift_table_a2v_ca_audio"] = (
        weights["scale_shift_table_a2v_ca_audio"].detach().to(torch.float32).cpu().contiguous()
    )
    save_dict["block0_prompt_scale_shift_table"] = (
        weights["prompt_scale_shift_table"].detach().to(torch.float32).cpu().contiguous()
    )
    save_dict["block0_audio_prompt_scale_shift_table"] = (
        weights["audio_prompt_scale_shift_table"].detach().to(torch.float32).cpu().contiguous()
    )

    # 6. Save safetensors
    save_file(save_dict, OUT_ST)
    print(f"[oracle] saved {len(save_dict)} tensors -> {OUT_ST}")

    # 7. Write metadata JSON
    shapes_dtypes = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in save_dict.items()}

    meta = {
        "generated_by": "scripts/ltx2_av_oracle.py",
        "checkpoint": CKPT,
        "seed": SEED,
        "block_idx": 0,
        "block_config": {
            "video_dim":                VIDEO_DIM,
            "video_heads":              VIDEO_HEADS,
            "video_head_dim":           VIDEO_HEAD_DIM,
            "video_context_dim":        VIDEO_CTX_DIM,
            "audio_dim":                AUDIO_DIM,
            "audio_heads":              AUDIO_HEADS,
            "audio_head_dim":           AUDIO_HEAD_DIM,
            "audio_context_dim":        AUDIO_CTX_DIM,
            "apply_gated_attention":    APPLY_GATED,
            "cross_attention_adaln":    CA_ADALN,
            "norm_eps":                 NORM_EPS,
            "rope_type":                "interleaved",
            "n_ada_params_video":       N_ADA_PARAMS_VIDEO,
            "n_ada_params_audio":       N_ADA_PARAMS_AUDIO,
        },
        "input_shapes": {
            "B": B, "S_V": S_V, "S_A": S_A, "N_TXT": N_TXT,
        },
        "tensors": shapes_dtypes,
        "sublayer_order_in_musubi_forward": [
            "1. video self-attn (attn1) — AdaLN table[0:3] for shift/scale/gate_msa; "
               "RoPE on Q and K; gated attn; residual vx += attn1_out * gate_msa",
            "2. video text cross-attn (attn2) — AdaLN table[6:9] for shift_q/scale_q/gate "
               "(cross_attention_adaln path); prompt_scale_shift_table modulates K/V of context; "
               "gated attn; residual vx += attn2_out * gate",
            "3. audio self-attn (audio_attn1) — same structure with audio_scale_shift_table[0:3]; "
               "RoPE on Q and K; gated attn; residual ax += audio_attn1_out * gate_msa",
            "4. audio text cross-attn (audio_attn2) — audio_scale_shift_table[6:9]; "
               "audio_prompt_scale_shift_table modulates K/V; gated attn; residual ax += ...",
            "5. shared rms_norm of vx and ax before A2V/V2A (no table, just rms_norm)",
            "6. get_av_ca_ada_values on scale_shift_table_a2v_ca_audio (5 rows) "
               "with audio.cross_scale_shift_timestep (4 params) and audio.cross_gate_timestep (1 gate) "
               "=> (scale_a_a2v, shift_a_a2v, scale_a_v2a, shift_a_v2a, gate_v2a)",
            "7. get_av_ca_ada_values on scale_shift_table_a2v_ca_video (5 rows) "
               "with video.cross_scale_shift_timestep (4 params) and video.cross_gate_timestep (1 gate) "
               "=> (scale_v_a2v, shift_v_a2v, scale_v_v2a, shift_v_v2a, gate_a2v)",
            "8. A2V cross-attn (audio_to_video_attn) — Q=video (mod by scale_v_a2v/shift_v_a2v), "
               "KV=audio (mod by scale_a_a2v/shift_a_a2v); cross-modal RoPE on Q (video cross_pe) "
               "and K (audio cross_pe); gated attn; residual vx += a2v_out * gate_a2v",
            "9. V2A cross-attn (video_to_audio_attn) — Q=audio (mod by scale_a_v2a/shift_a_v2a), "
               "KV=video (mod by scale_v_v2a/shift_v_v2a); cross-modal RoPE on Q (audio cross_pe) "
               "and K (video cross_pe); gated attn; residual ax += v2a_out * gate_v2a",
            "10. video FFN — AdaLN table[3:6] for shift_mlp/scale_mlp/gate_mlp; "
                "GELUApprox (tanh approx); clamp +-60000; residual vx += ff_out * gate_mlp",
            "11. audio FFN — audio_scale_shift_table[3:6]; same structure; residual ax += ...",
        ],
        "key_divergences_from_diffusers": [
            "diffusers LTXTransformerBlock has only attn1+ff (6-param AdaLN) — video-only, no audio",
            "real model has 9-param AdaLN (6 base + 3 cross_attention_adaln) per stream",
            "real model has audio_to_video_attn and video_to_audio_attn cross-modal paths",
            "real model has gated attention (to_gate_logits, gate=2*sigmoid) on every attn module",
            "real model uses prompt_scale_shift_table to modulate context K/V in cross-attn",
            "rms_norm always runs in float32 then casts back (musubi stability fix)",
            "AdaLN modulation runs in float32 then casts back (musubi stability fix for bf16 overflow)",
            "get_av_ca_ada_values uses two separate adaln streams: scale_shift (4 params) "
               "and gate (1 param), sourced from different timestep embeddings "
               "(cross_scale_shift_timestep vs cross_gate_timestep)",
        ],
    }

    with open(OUT_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[oracle] metadata -> {OUT_META}")

    # 8. Quick self-consistency check: run again with same inputs
    with torch.no_grad():
        video_out2, audio_out2 = block._forward(video_args, audio_args, perturbations=None)
    video_out2 = video_out2.x
    audio_out2 = audio_out2.x

    v_cos = float((video_out.flatten() @ video_out2.flatten()) /
                  (video_out.flatten().norm() * video_out2.flatten().norm() + 1e-30))
    a_cos = float((audio_out.flatten() @ audio_out2.flatten()) /
                  (audio_out.flatten().norm() * audio_out2.flatten().norm() + 1e-30))
    print(f"[oracle] self-check: video_cos={v_cos:.8f}  audio_cos={a_cos:.8f}")
    if v_cos < 0.9999 or a_cos < 0.9999:
        print("[oracle] WARNING: self-check failed — forward is non-deterministic!")
    else:
        print("[oracle] self-check PASSED (deterministic)")

    print("[oracle] DONE")
    return save_dict


if __name__ == "__main__":
    run_oracle()
