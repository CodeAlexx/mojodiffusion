#!/usr/bin/env python3
"""MiniMax-H3 AdaLN modulation-cache REAL-WEIGHT oracle (DEV-ONLY, never
imported by Mojo).

Verifies `minimax_h3_build_modulation_cache` (models/dit/minimax_h3_modcache.mojo)
against REAL `blocks.0/1.adaln_proj.linear.{weight,bias}` bytes — the last
39%-of-the-model surface this port had never run real bytes through
(everything else in adaln_proj is either math this port doesn't touch at all,
or already covered by minimax_h3_block_real_weight_gate.mojo for the OTHER
8 block tensors).

WHAT IS REAL vs SYNTHETIC vs OUT OF SCOPE:
  REAL:         blocks.0.adaln_proj.linear.{weight,bias} and
                blocks.1.adaln_proj.linear.{weight,bias} — read directly off
                model-00001-of-00013.safetensors (the only shard with any
                block's adaln_proj currently on disk).
  SYNTHETIC:    temb (fixed-seed random F32) — gating this math does not need
                a real timestep embedding, same reasoning as the block gate.
  OUT OF SCOPE: `final_layer.adaln_proj.linear.*` lives in
                model-00013-of-00013.safetensors, which is NOT on disk. This
                script does not read it and does not compare it. The
                Mojo-side gate needs a shape/dtype-correct STAND-IN so it can
                call the real, unmodified `minimax_h3_build_modulation_cache`
                entry point (whose preflight checks the final layer
                unconditionally) without faking the block-0/1 result — see
                minimax_h3_modcache_real_weight_gate.mojo's header for why a
                stand-in is honest here and this script's
                write_dummy_final_layer_shard() below for what it contains
                (random bf16, right shape, NEVER compared against anything).

ORACLE: diffusers' OWN `MiniMaxH3AdaLayerNormModulation` class (not a hand
transcription), GPU bf16, loaded with the real per-block weight.

ACTIVATION-PRECISION TRAP (models/dit/minimax_h3_modcache.mojo header): SiLU
runs at temb's OWN F32 precision; only the result is cast to bf16 before the
projection. Section [2] below demonstrates this is a DISCRIMINATING test, not
a tautological one — SiLU-then-cast and cast-then-SiLU visibly disagree on a
probe grid before ever touching a real weight, and temb is deliberately
scaled into SiLU's high-curvature range so the effect is not lost when it
composes with real weights.

Run: python3 serenitymojo/models/dit/parity/minimax_h3_modcache_real_weight_oracle.py
Writes:
  output/minimax_h3_block/modcache_real_weight_ref.safetensors
  output/minimax_h3_block/h3_dummy_final_layer.safetensors   (stand-in, see above)
"""

import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DIFFUSERS_SRC = "/home/alex/minimax_h3_ref/diffusers-src/src"
sys.path.insert(0, DIFFUSERS_SRC)

from diffusers.models.transformers.transformer_minimax_h3 import (  # noqa: E402
    MiniMaxH3AdaLayerNormModulation,
)

CKPT_DIR = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/transformer"
SHARD = os.path.join(CKPT_DIR, "model-00001-of-00013.safetensors")
OUT_DIR = "/home/alex/mojodiffusion/output/minimax_h3_block"
OUT_PATH = os.path.join(OUT_DIR, "modcache_real_weight_ref.safetensors")
DUMMY_FINAL_LAYER_PATH = os.path.join(OUT_DIR, "h3_dummy_final_layer.safetensors")

HIDDEN = 5376
TIME_EMBED_DIM = 2688
ADALN_OUT_FEATURES = 6 * HIDDEN * 3  # 96768
FINAL_ADALN_OUT_FEATURES = 2 * HIDDEN  # 10752
NUM_LAYERS_TESTED = 2  # blocks 0 and 1 -- the only complete real adaln bytes on disk


def load_block_adaln(layer: int, device: str):
    names = [f"blocks.{layer}.adaln_proj.linear.weight", f"blocks.{layer}.adaln_proj.linear.bias"]
    with safe_open(SHARD, framework="pt", device="cpu") as f:
        w = f.get_tensor(names[0]).to(device=device)
        b = f.get_tensor(names[1]).to(device=device)
    assert w.dtype == torch.bfloat16 and b.dtype == torch.bfloat16, (w.dtype, b.dtype)
    assert tuple(w.shape) == (ADALN_OUT_FEATURES, TIME_EMBED_DIM), w.shape
    assert tuple(b.shape) == (ADALN_OUT_FEATURES,), b.shape
    return w, b


def write_dummy_final_layer_shard() -> None:
    """Shape/dtype-correct STAND-IN for final_layer.adaln_proj.linear.* — real
    bytes are in the missing shard 13. Random, fixed-seed, bf16, right shape;
    never read back by anything that compares values. Exists ONLY so the Mojo
    gate can call minimax_h3_build_modulation_cache's real unmodified entry
    point (its preflight unconditionally checks the final layer) without
    faking the block-0/1 result this test actually measures."""
    g = torch.Generator().manual_seed(999)
    w = (torch.randn(FINAL_ADALN_OUT_FEATURES, TIME_EMBED_DIM, generator=g) * 0.02).to(torch.bfloat16)
    b = (torch.randn(FINAL_ADALN_OUT_FEATURES, generator=g) * 0.02).to(torch.bfloat16)
    save_file(
        {
            "final_layer.adaln_proj.linear.weight": w,
            "final_layer.adaln_proj.linear.bias": b,
        },
        DUMMY_FINAL_LAYER_PATH,
        metadata={"note": "STAND-IN ONLY, not real MiniMax-H3 bytes; shard 13 is missing on disk"},
    )
    print(f"    wrote stand-in {DUMMY_FINAL_LAYER_PATH} (NOT real weights, preflight-shape-only)")


def main() -> None:
    device = "cuda"

    print("[1] activation-precision-order discriminating check (SiLU-then-cast vs cast-then-SiLU)")
    probe_x = torch.linspace(-6.0, 6.0, steps=4096, dtype=torch.float32, device=device)
    silu_f32_then_bf16 = torch.nn.functional.silu(probe_x).to(torch.bfloat16)
    bf16_then_silu = torch.nn.functional.silu(probe_x.to(torch.bfloat16))
    order_diff = (silu_f32_then_bf16.float() - bf16_then_silu.float()).abs()
    print(
        f"    max |silu(x)_f32->bf16 - silu(x_bf16)| over {probe_x.numel()} probe points in [-6,6]:"
        f" {order_diff.max().item():.6f} (nonzero -> order is NOT tautological)"
    )
    if order_diff.max().item() == 0.0:
        print("    WARNING: probe range did not discriminate the two orders; widen it")

    print(f"[2] reading blocks.0/1 adaln_proj.linear.{{weight,bias}} from {SHARD}")
    for layer in range(NUM_LAYERS_TESTED):
        w, b = load_block_adaln(layer, device)
        print(f"    blocks.{layer}.adaln_proj.linear.weight {tuple(w.shape)} {w.dtype}")

    write_dummy_final_layer_shard()

    print("[3] two distinct timesteps, seeded random temb (F32, matches time_embedder's own dtype),")
    print("    scaled into SiLU's high-curvature range so the activation-order effect survives")
    num_timesteps = 2
    gen = torch.Generator(device=device).manual_seed(21)
    temb = torch.randn(num_timesteps, TIME_EMBED_DIM, generator=gen, device=device, dtype=torch.float32) * 2.0

    tensors = {"in.temb": temb.cpu().contiguous()}
    print("[4] running the REAL diffusers MiniMaxH3AdaLayerNormModulation.forward per layer, GPU bf16")
    for layer in range(NUM_LAYERS_TESTED):
        w, b = load_block_adaln(layer, device)
        module = (
            MiniMaxH3AdaLayerNormModulation(time_embed_dim=TIME_EMBED_DIM, hidden_size=HIDDEN)
            .to(device=device, dtype=torch.bfloat16)
        )
        with torch.no_grad():
            module.linear.weight.copy_(w)
            module.linear.bias.copy_(b)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = module(temb)
        mod = torch.cat([shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp], dim=-1).float()
        print(f"    layer {layer}: mod {tuple(mod.shape)} mean {mod.mean().item():.6f} std {mod.std().item():.6f}")

        # [3-in-brief.md]-style row differentiation on REAL bytes: video(tag0) vs
        # text(tag1) vs audio(tag2) rows at timestep 0 must genuinely differ.
        row_video = mod[0]
        row_text = mod[1]
        row_audio = mod[2]
        d_vt = (row_video - row_text).norm().item()
        d_va = (row_video - row_audio).norm().item()
        d_ta = (row_text - row_audio).norm().item()
        print(
            f"    layer {layer} row L2 distances @ t0: video<->text={d_vt:.4f}"
            f" video<->audio={d_va:.4f} text<->audio={d_ta:.4f}"
        )

        tensors[f"out.block_mod.{layer}"] = mod.cpu().contiguous()
        del w, b, module
        torch.cuda.empty_cache()

    os.makedirs(OUT_DIR, exist_ok=True)
    meta = {
        "num_layers_tested": str(NUM_LAYERS_TESTED),
        "num_timesteps": str(num_timesteps),
        "hidden_size": str(HIDDEN),
        "time_embed_dim": str(TIME_EMBED_DIM),
        "shard": SHARD,
        "note": (
            "block_mod rows = num_timesteps*3 (row = timestep_index*3+tag, "
            "tag 0=video 1=text 2=audio), cols = 6*hidden_size; final_layer "
            "NOT tested here (real bytes missing on disk, out of scope)"
        ),
    }
    save_file(tensors, OUT_PATH, metadata=meta)
    print(f"[5] wrote {OUT_PATH}")
    print("DONE")


if __name__ == "__main__":
    main()
