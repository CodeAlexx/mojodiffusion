#!/usr/bin/env python3
"""LTX-2.3 22B AV transformer block-0 PER-SUBLAYER intermediate oracle.

Captures the activation AFTER each of the 11 sublayer steps of
BasicAVTransformerBlock._forward on block-0.  Uses forward hooks on a
patched copy of _forward so we can capture at exact step boundaries.

Same seed=0 / Sv=64 / Sa=32 inputs as block0_ref.py so outputs align.

Run:
  /home/alex/musubi-tuner/venv/bin/python scripts/ltx2_av_oracle_intermediates.py

Outputs:
  output/ltx2_av/block0_intermediates.safetensors
  output/ltx2_av/block0_intermediates_meta.json

Step keys (11 steps, matching musubi _forward lines 595-791):
  step01_video_self_attn     — vx after attn1 + residual              (line 610)
  step02_video_cross_attn    — vx after attn2 + residual              (line 613-627)
  step03_audio_self_attn     — ax after audio_attn1 + residual        (line 648)
  step04_audio_cross_attn    — ax after audio_attn2 + residual        (line 651-667)
  step05_cross_modal_prenorm — (vx_norm3, ax_norm3) before a2v/v2a    (line 674-675)
  step06_av_adaln_audio      — 5 av-ca ada values from audio table    (line 683-688)
  step07_av_adaln_video      — 5 av-ca ada values from video table    (line 697-703)
  step08_a2v_attn            — vx after audio_to_video_attn + gate    (line 710-723)
  step09_v2a_attn            — ax after video_to_audio_attn + gate    (line 734-746)
  step10_video_ffn           — vx after ff + gate + residual          (line 769-773)
  step11_audio_ffn           — ax after audio_ff + gate + residual    (final)

NOTE: steps 05/06/07 are intermediate *values* (not final hidden states) saved
as separate keys.  Steps 01-04 and 08-11 save the updated vx or ax directly.
"""

import json
import os
import sys
import types

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# Paths / config (identical to ltx2_av_oracle.py)
# ---------------------------------------------------------------------------
CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev.safetensors"
OUT_DIR = "/home/alex/mojodiffusion/output/ltx2_av"
OUT_ST = os.path.join(OUT_DIR, "block0_intermediates.safetensors")
OUT_META = os.path.join(OUT_DIR, "block0_intermediates_meta.json")

BLOCK0_PREFIX = "model.diffusion_model.transformer_blocks.0."

VIDEO_DIM       = 4096
VIDEO_HEADS     = 32
VIDEO_HEAD_DIM  = 128
VIDEO_CTX_DIM   = 4096

AUDIO_DIM       = 2048
AUDIO_HEADS     = 32
AUDIO_HEAD_DIM  = 64
AUDIO_CTX_DIM   = 2048

NORM_EPS        = 1e-6
APPLY_GATED     = True
CA_ADALN        = True

N_ADA_PARAMS_VIDEO = 9
N_ADA_PARAMS_AUDIO = 9

S_V    = 64
S_A    = 32
N_TXT  = 16
B      = 1
SEED   = 0

MUSUBI_SRC = "/home/alex/musubi-tuner/src"
if MUSUBI_SRC not in sys.path:
    sys.path.insert(0, MUSUBI_SRC)

# ---------------------------------------------------------------------------
# Load block-0 weights (header-first, pull only block-0 tensors)
# ---------------------------------------------------------------------------

def load_block0_weights() -> dict[str, torch.Tensor]:
    print(f"[intermediates] scanning header of {os.path.basename(CKPT)} ...")
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        all_keys = list(f.keys())
        block0_keys = [k for k in all_keys if k.startswith(BLOCK0_PREFIX)]
        print(f"[intermediates] found {len(block0_keys)} block-0 keys (total: {len(all_keys)})")
        weights = {}
        for k in sorted(block0_keys):
            t = f.get_tensor(k)
            short = k[len(BLOCK0_PREFIX):]
            weights[short] = t
    return weights


# ---------------------------------------------------------------------------
# Build musubi block
# ---------------------------------------------------------------------------

def build_block(weights: dict[str, torch.Tensor]):
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
        attention_function=AttentionFunction.PYTORCH,
    )

    sd = {k: v.to(torch.float32) for k, v in weights.items()}
    missing, unexpected = block.load_state_dict(sd, strict=True)
    if missing:
        print(f"[intermediates] WARNING: missing keys: {missing}")
    if unexpected:
        print(f"[intermediates] WARNING: unexpected keys: {unexpected}")
    block.eval()
    block = block.to(torch.float32)
    return block


# ---------------------------------------------------------------------------
# Build inputs (identical generator sequence to ltx2_av_oracle.py)
# ---------------------------------------------------------------------------

def make_inputs():
    from musubi_tuner.ltx_2.model.transformer.transformer_args import TransformerArgs

    g = torch.Generator(device="cpu").manual_seed(SEED)

    def rnd(*shape, scale=0.1):
        return torch.randn(*shape, generator=g) * scale

    video_x          = rnd(B, S_V, VIDEO_DIM)
    video_ctx        = rnd(B, N_TXT, VIDEO_CTX_DIM)
    video_timesteps  = rnd(B, 1, N_ADA_PARAMS_VIDEO * VIDEO_DIM, scale=0.05)
    video_prompt_ts  = rnd(B, N_TXT, 2 * VIDEO_DIM, scale=0.05)
    video_cos        = rnd(B, S_V, VIDEO_DIM, scale=0.3)
    video_sin        = rnd(B, S_V, VIDEO_DIM, scale=0.3)
    video_pe         = (video_cos, video_sin)
    av_cross_cos_v   = rnd(B, S_V, AUDIO_DIM, scale=0.3)
    av_cross_sin_v   = rnd(B, S_V, AUDIO_DIM, scale=0.3)
    video_cross_pe   = (av_cross_cos_v, av_cross_sin_v)
    video_cross_ss_ts  = rnd(B, 1, 4 * VIDEO_DIM, scale=0.05)
    video_cross_gate_ts = rnd(B, 1, VIDEO_DIM, scale=0.05)

    audio_x          = rnd(B, S_A, AUDIO_DIM)
    audio_ctx        = rnd(B, N_TXT, AUDIO_CTX_DIM)
    audio_timesteps  = rnd(B, 1, N_ADA_PARAMS_AUDIO * AUDIO_DIM, scale=0.05)
    audio_prompt_ts  = rnd(B, N_TXT, 2 * AUDIO_DIM, scale=0.05)
    audio_cos        = rnd(B, S_A, AUDIO_DIM, scale=0.3)
    audio_sin        = rnd(B, S_A, AUDIO_DIM, scale=0.3)
    audio_pe         = (audio_cos, audio_sin)
    av_cross_cos_a   = rnd(B, S_A, AUDIO_DIM, scale=0.3)
    av_cross_sin_a   = rnd(B, S_A, AUDIO_DIM, scale=0.3)
    audio_cross_pe   = (av_cross_cos_a, av_cross_sin_a)
    audio_cross_ss_ts  = rnd(B, 1, 4 * AUDIO_DIM, scale=0.05)
    audio_cross_gate_ts = rnd(B, 1, AUDIO_DIM, scale=0.05)

    video_args = TransformerArgs(
        x=video_x,
        context=video_ctx,
        context_mask=None,
        timesteps=video_timesteps,
        embedded_timestep=video_timesteps,
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

    return video_args, audio_args


# ---------------------------------------------------------------------------
# Instrumented _forward: captures intermediates at exact step boundaries.
# We shadow _forward with a patched version that records tensors AFTER each
# of the 11 sublayer steps (using the same logic as the original _forward but
# with capture points inserted).
# ---------------------------------------------------------------------------

def run_instrumented_forward(block, video_args, audio_args) -> dict[str, torch.Tensor]:
    """Run block._forward with step-by-step capture.

    Returns dict of per-step tensors saved as float32 CPU.
    """
    from musubi_tuner.ltx_2.model.transformer.transformer import (
        apply_cross_attention_adaln,
        BatchedPerturbationConfig,
    )
    from musubi_tuner.ltx_2.utils import rms_norm
    from dataclasses import replace as dc_replace

    captures: dict[str, torch.Tensor] = {}

    def cap(key: str, t: torch.Tensor) -> torch.Tensor:
        """Save a clone of t under key, return t unchanged."""
        captures[key] = t.detach().to(torch.float32).cpu().contiguous()
        return t

    def stats(t: torch.Tensor, name: str) -> str:
        return (f"{name}: shape={tuple(t.shape)} "
                f"mean={float(t.mean()):.4f} "
                f"std={float(t.std()):.4f} "
                f"absmax={float(t.abs().max()):.4f}")

    video  = video_args
    audio  = audio_args
    perturbations = BatchedPerturbationConfig.empty(B)

    vx = video.x
    ax = audio.x
    norm_eps = block.norm_eps

    # ── STEP 1: Video self-attn ──────────────────────────────────────────────
    # musubi _forward lines 597-611
    vshift_msa, vscale_msa, vgate_msa = block.get_ada_values(
        block.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3),
        num_tokens=vx.shape[1]
    )
    norm_vx = (
        rms_norm(vx, eps=norm_eps).to(torch.float32) * (1 + vscale_msa.to(torch.float32))
        + vshift_msa.to(torch.float32)
    ).to(vx.dtype)
    attn1_out = block.attn1(norm_vx, pe=video.positional_embeddings, mask=video.self_attention_mask)
    vx = vx + attn1_out * vgate_msa
    cap("step01_video_self_attn", vx)
    print(f"  {stats(vx, 'step01_video_self_attn')}")

    # ── STEP 2: Video cross-attn (text) ─────────────────────────────────────
    # musubi _forward lines 613-627 via _apply_text_cross_attention
    shift_q_v, scale_q_v, gate_v = block.get_ada_values(
        block.scale_shift_table, vx.shape[0], video.timesteps, slice(6, 9),
        num_tokens=vx.shape[1]
    )
    ca2_delta = apply_cross_attention_adaln(
        vx,
        video.context,
        lambda q, context=None, mask=None: block.attn2(q, context=context, mask=mask),
        shift_q_v, scale_q_v, gate_v,
        block.prompt_scale_shift_table,
        video.prompt_timestep,
        video.context_mask,
        norm_eps,
    )
    vx = vx + ca2_delta
    cap("step02_video_cross_attn", vx)
    print(f"  {stats(vx, 'step02_video_cross_attn')}")

    del vshift_msa, vscale_msa, vgate_msa, shift_q_v, scale_q_v, gate_v

    # ── STEP 3: Audio self-attn ──────────────────────────────────────────────
    # musubi _forward lines 634-649
    ashift_msa, ascale_msa, agate_msa = block.get_ada_values(
        block.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3),
        num_tokens=ax.shape[1]
    )
    norm_ax = (
        rms_norm(ax, eps=norm_eps).to(torch.float32) * (1 + ascale_msa.to(torch.float32))
        + ashift_msa.to(torch.float32)
    ).to(ax.dtype)
    audio_attn1_out = block.audio_attn1(norm_ax, pe=audio.positional_embeddings, mask=audio.self_attention_mask)
    ax = ax + audio_attn1_out * agate_msa
    cap("step03_audio_self_attn", ax)
    print(f"  {stats(ax, 'step03_audio_self_attn')}")

    # ── STEP 4: Audio cross-attn (text) ─────────────────────────────────────
    # musubi _forward lines 651-667 via _apply_text_cross_attention
    shift_q_a, scale_q_a, gate_a = block.get_ada_values(
        block.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(6, 9),
        num_tokens=ax.shape[1]
    )
    ca2_audio_delta = apply_cross_attention_adaln(
        ax,
        audio.context,
        lambda q, context=None, mask=None: block.audio_attn2(q, context=context, mask=mask),
        shift_q_a, scale_q_a, gate_a,
        block.audio_prompt_scale_shift_table,
        audio.prompt_timestep,
        audio.context_mask,
        norm_eps,
    )
    ax = ax + ca2_audio_delta
    cap("step04_audio_cross_attn", ax)
    print(f"  {stats(ax, 'step04_audio_cross_attn')}")

    del ashift_msa, ascale_msa, agate_msa, shift_q_a, scale_q_a, gate_a

    # ── STEP 5: Pre-normalize both streams for cross-modal ───────────────────
    # musubi _forward lines 674-675
    vx_norm3 = rms_norm(vx, eps=norm_eps)
    ax_norm3 = rms_norm(ax, eps=norm_eps)
    cap("step05_vx_norm3", vx_norm3)
    cap("step05_ax_norm3", ax_norm3)
    print(f"  {stats(vx_norm3, 'step05_vx_norm3')}")
    print(f"  {stats(ax_norm3, 'step05_ax_norm3')}")

    # ── STEP 6: AV AdaLN from audio table ────────────────────────────────────
    # musubi _forward lines 683-688 (scale_shift_table_a2v_ca_audio)
    (
        scale_ca_audio_a2v,
        shift_ca_audio_a2v,
        scale_ca_audio_v2a,
        shift_ca_audio_v2a,
        gate_out_v2a,
    ) = block.get_av_ca_ada_values(
        block.scale_shift_table_a2v_ca_audio,
        ax.shape[0],
        audio.cross_scale_shift_timestep,
        audio.cross_gate_timestep,
        num_tokens=ax.shape[1],
    )
    # Save all 5 per-step values
    cap("step06_scale_ca_audio_a2v", scale_ca_audio_a2v.squeeze(1) if scale_ca_audio_a2v.dim() == 3 else scale_ca_audio_a2v)
    cap("step06_shift_ca_audio_a2v", shift_ca_audio_a2v.squeeze(1) if shift_ca_audio_a2v.dim() == 3 else shift_ca_audio_a2v)
    cap("step06_scale_ca_audio_v2a", scale_ca_audio_v2a.squeeze(1) if scale_ca_audio_v2a.dim() == 3 else scale_ca_audio_v2a)
    cap("step06_shift_ca_audio_v2a", shift_ca_audio_v2a.squeeze(1) if shift_ca_audio_v2a.dim() == 3 else shift_ca_audio_v2a)
    cap("step06_gate_v2a", gate_out_v2a.squeeze(1) if gate_out_v2a.dim() == 3 else gate_out_v2a)
    print(f"  step06: audio ada values captured (5 tensors, shape={tuple(scale_ca_audio_a2v.shape)})")

    # ── STEP 7: AV AdaLN from video table ────────────────────────────────────
    # musubi _forward lines 697-703 (scale_shift_table_a2v_ca_video)
    (
        scale_ca_video_a2v,
        shift_ca_video_a2v,
        scale_ca_video_v2a,
        shift_ca_video_v2a,
        gate_out_a2v,
    ) = block.get_av_ca_ada_values(
        block.scale_shift_table_a2v_ca_video,
        vx.shape[0],
        video.cross_scale_shift_timestep,
        video.cross_gate_timestep,
        num_tokens=vx.shape[1],
    )
    cap("step07_scale_ca_video_a2v", scale_ca_video_a2v.squeeze(1) if scale_ca_video_a2v.dim() == 3 else scale_ca_video_a2v)
    cap("step07_shift_ca_video_a2v", shift_ca_video_a2v.squeeze(1) if shift_ca_video_a2v.dim() == 3 else shift_ca_video_a2v)
    cap("step07_scale_ca_video_v2a", scale_ca_video_v2a.squeeze(1) if scale_ca_video_v2a.dim() == 3 else scale_ca_video_v2a)
    cap("step07_shift_ca_video_v2a", shift_ca_video_v2a.squeeze(1) if shift_ca_video_v2a.dim() == 3 else shift_ca_video_v2a)
    cap("step07_gate_a2v", gate_out_a2v.squeeze(1) if gate_out_a2v.dim() == 3 else gate_out_a2v)
    print(f"  step07: video ada values captured (5 tensors, shape={tuple(scale_ca_video_a2v.shape)})")

    # ── STEP 8: A2V cross-attn (audio_to_video_attn) ─────────────────────────
    # musubi _forward lines 705-723
    vx_scaled = (
        vx_norm3.to(torch.float32) * (1 + scale_ca_video_a2v.to(torch.float32))
        + shift_ca_video_a2v.to(torch.float32)
    ).to(vx.dtype)
    ax_scaled = (
        ax_norm3.to(torch.float32) * (1 + scale_ca_audio_a2v.to(torch.float32))
        + shift_ca_audio_a2v.to(torch.float32)
    ).to(ax.dtype)
    a2v_out = block.audio_to_video_attn(
        vx_scaled,
        context=ax_scaled,
        mask=None,
        pe=video.cross_positional_embeddings,
        k_pe=audio.cross_positional_embeddings,
    )
    vx = vx + a2v_out * gate_out_a2v
    cap("step08_a2v_attn", vx)
    print(f"  {stats(vx, 'step08_a2v_attn')}")

    # ── STEP 9: V2A cross-attn (video_to_audio_attn) ─────────────────────────
    # musubi _forward lines 726-746
    ax_scaled2 = (
        ax_norm3.to(torch.float32) * (1 + scale_ca_audio_v2a.to(torch.float32))
        + shift_ca_audio_v2a.to(torch.float32)
    ).to(ax.dtype)
    vx_scaled2 = (
        vx_norm3.to(torch.float32) * (1 + scale_ca_video_v2a.to(torch.float32))
        + shift_ca_video_v2a.to(torch.float32)
    ).to(vx.dtype)
    v2a_out = block.video_to_audio_attn(
        ax_scaled2,
        context=vx_scaled2,
        mask=None,
        pe=audio.cross_positional_embeddings,
        k_pe=video.cross_positional_embeddings,
    )
    ax = ax + v2a_out * gate_out_v2a
    cap("step09_v2a_attn", ax)
    print(f"  {stats(ax, 'step09_v2a_attn')}")

    del gate_out_a2v, gate_out_v2a

    # ── STEP 10: Video FFN ────────────────────────────────────────────────────
    # musubi _forward lines 763-774
    mlp_slice = slice(3, 6)   # cross_attention_adaln=True always in block-0
    vshift_mlp, vscale_mlp, vgate_mlp = block.get_ada_values(
        block.scale_shift_table, vx.shape[0], video.timesteps, mlp_slice,
        num_tokens=vx.shape[1]
    )
    vx_scaled_ff = (
        rms_norm(vx, eps=norm_eps).to(torch.float32) * (1 + vscale_mlp.to(torch.float32))
        + vshift_mlp.to(torch.float32)
    ).to(vx.dtype)
    ff_out = block.ff(vx_scaled_ff) * vgate_mlp
    ff_out = ff_out.clamp(-60000.0, 60000.0)
    vx = vx + ff_out
    cap("step10_video_ffn", vx)
    print(f"  {stats(vx, 'step10_video_ffn')}")

    del vshift_mlp, vscale_mlp, vgate_mlp

    # ── STEP 11: Audio FFN ────────────────────────────────────────────────────
    # musubi _forward lines 778-789
    ashift_mlp, ascale_mlp, agate_mlp = block.get_ada_values(
        block.audio_scale_shift_table, ax.shape[0], audio.timesteps, mlp_slice,
        num_tokens=ax.shape[1]
    )
    ax_scaled_ff = (
        rms_norm(ax, eps=norm_eps).to(torch.float32) * (1 + ascale_mlp.to(torch.float32))
        + ashift_mlp.to(torch.float32)
    ).to(ax.dtype)
    audio_ff_out = block.audio_ff(ax_scaled_ff) * agate_mlp
    audio_ff_out = audio_ff_out.clamp(-60000.0, 60000.0)
    ax = ax + audio_ff_out
    cap("step11_audio_ffn", ax)
    print(f"  {stats(ax, 'step11_audio_ffn')}")

    del ashift_mlp, ascale_mlp, agate_mlp

    # Also capture final video_out / audio_out for cross-check against block0_ref.safetensors.
    # These are the same tensors as step10/step11 but stored as fresh copies so
    # safetensors doesn't complain about shared memory.
    captures["video_out"] = captures["step10_video_ffn"].clone()
    captures["audio_out"] = captures["step11_audio_ffn"].clone()

    return captures


# ---------------------------------------------------------------------------
# Cross-check against block0_ref.safetensors
# ---------------------------------------------------------------------------

def cross_check(captures: dict[str, torch.Tensor]) -> None:
    """Verify that our step11 final outputs match the original oracle."""
    ref_path = os.path.join(OUT_DIR, "block0_ref.safetensors")
    if not os.path.exists(ref_path):
        print("[intermediates] WARNING: block0_ref.safetensors not found; skipping cross-check")
        return

    with safe_open(ref_path, framework="pt", device="cpu") as f:
        ref_vout = f.get_tensor("video_out")
        ref_aout = f.get_tensor("audio_out")

    our_vout = captures["video_out"]
    our_aout = captures["audio_out"]

    def cos_sim(a, b):
        a = a.flatten().float()
        b = b.flatten().float()
        return float((a @ b) / (a.norm() * b.norm() + 1e-30))

    def max_absdiff(a, b):
        return float((a.float() - b.float()).abs().max())

    v_cos = cos_sim(our_vout, ref_vout)
    a_cos = cos_sim(our_aout, ref_aout)
    v_diff = max_absdiff(our_vout, ref_vout)
    a_diff = max_absdiff(our_aout, ref_aout)

    print(f"\n[intermediates] cross-check vs block0_ref.safetensors:")
    print(f"  video_out: cos={v_cos:.8f}  max_absdiff={v_diff:.6f}")
    print(f"  audio_out: cos={a_cos:.8f}  max_absdiff={a_diff:.6f}")

    if v_cos > 0.9999 and a_cos > 0.9999:
        print("[intermediates] cross-check PASSED — intermediates oracle matches block0_ref")
    else:
        print("[intermediates] WARNING: cross-check below 0.9999 — possible input mismatch!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Load block-0 weights
    weights = load_block0_weights()

    # 2. Build block
    print("[intermediates] building BasicAVTransformerBlock ...")
    block = build_block(weights)
    print(f"[intermediates] block parameters: {sum(p.numel() for p in block.parameters()):,}")

    # 3. Build inputs (same seed=0 sequence as block0_ref)
    print("[intermediates] building inputs (seed=0, Sv=64, Sa=32) ...")
    video_args, audio_args = make_inputs()

    # 4. Run instrumented forward
    print("[intermediates] running instrumented _forward ...")
    with torch.no_grad():
        captures = run_instrumented_forward(block, video_args, audio_args)

    # 5. Sanity: all finite
    for k, t in captures.items():
        if not torch.isfinite(t).all():
            print(f"[intermediates] ERROR: {k} has non-finite values!")
            sys.exit(1)
    print(f"\n[intermediates] all {len(captures)} captured tensors are finite")

    # 6. Cross-check final outputs vs block0_ref
    cross_check(captures)

    # 7. Save safetensors
    save_file(captures, OUT_ST)
    print(f"\n[intermediates] saved {len(captures)} tensors -> {OUT_ST}")

    # 8. Write metadata JSON
    shapes_dtypes = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in captures.items()}
    meta = {
        "generated_by": "scripts/ltx2_av_oracle_intermediates.py",
        "checkpoint": CKPT,
        "seed": SEED,
        "block_idx": 0,
        "input_shapes": {"B": B, "S_V": S_V, "S_A": S_A, "N_TXT": N_TXT},
        "sublayer_step_keys": [
            "step01_video_self_attn       — vx after attn1 + gate_msa residual [B,Sv,4096]",
            "step02_video_cross_attn      — vx after attn2 (text) + gate_ca residual [B,Sv,4096]",
            "step03_audio_self_attn       — ax after audio_attn1 + gate_msa residual [B,Sa,2048]",
            "step04_audio_cross_attn      — ax after audio_attn2 (text) + gate_ca residual [B,Sa,2048]",
            "step05_vx_norm3              — rms_norm(vx) before cross-modal [B,Sv,4096]",
            "step05_ax_norm3              — rms_norm(ax) before cross-modal [B,Sa,2048]",
            "step06_scale_ca_audio_a2v    — AdaLN from audio table, a2v scale for audio KV [B,Sa,2048]",
            "step06_shift_ca_audio_a2v    — AdaLN from audio table, a2v shift for audio KV [B,Sa,2048]",
            "step06_scale_ca_audio_v2a    — AdaLN from audio table, v2a scale for audio Q [B,Sa,2048]",
            "step06_shift_ca_audio_v2a    — AdaLN from audio table, v2a shift for audio Q [B,Sa,2048]",
            "step06_gate_v2a              — AdaLN from audio table, gate for V2A output [B,Sa,2048]",
            "step07_scale_ca_video_a2v    — AdaLN from video table, a2v scale for video Q [B,Sv,4096]",
            "step07_shift_ca_video_a2v    — AdaLN from video table, a2v shift for video Q [B,Sv,4096]",
            "step07_scale_ca_video_v2a    — AdaLN from video table, v2a scale for video KV [B,Sv,4096]",
            "step07_shift_ca_video_v2a    — AdaLN from video table, v2a shift for video KV [B,Sv,4096]",
            "step07_gate_a2v              — AdaLN from video table, gate for A2V output [B,Sv,4096]",
            "step08_a2v_attn              — vx after audio_to_video_attn + gate_a2v residual [B,Sv,4096]",
            "step09_v2a_attn              — ax after video_to_audio_attn + gate_v2a residual [B,Sa,2048]",
            "step10_video_ffn             — vx after ff + gate_mlp + residual [B,Sv,4096]",
            "step11_audio_ffn             — ax after audio_ff + gate_mlp + residual [B,Sa,2048]",
            "video_out                    — alias for step10_video_ffn (final video hidden state)",
            "audio_out                    — alias for step11_audio_ffn (final audio hidden state)",
        ],
        "tensors": shapes_dtypes,
    }
    with open(OUT_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[intermediates] metadata -> {OUT_META}")
    print("[intermediates] DONE")


if __name__ == "__main__":
    run()
