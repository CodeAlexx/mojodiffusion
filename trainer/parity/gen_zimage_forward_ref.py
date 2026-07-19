#!/usr/bin/env python
"""
Numeric forward reference for Serenity's REAL Z-Image transformer.

Reference source: Serenity + diffusers ZImageTransformer2DModel ONLY.
Run with: /home/alex/Serenity/venv/bin/python
cwd:      /home/alex/Serenity   (so `modules` imports resolve)

What this does
--------------
Loads the diffusers ZImageTransformer2DModel from the local checkpoint, builds
FIXED reproducible inputs (numpy seed 1234), runs the transformer forward in the
EXACT way Serenity does (modules/modelSetup/BaseZImageSetup.py:125-135), and
dumps raw little-endian float32 bins + a JSON meta describing the layout.

text_encoder_output structure
------------------------------
Serenity (ZImageModel.encode_text) returns `embeddings_list`:
    a Python list, one entry per batch sample, each entry a 2-D tensor
    [caption_len, feat_dim] = [64, 2560]  (Qwen3 hidden_states[-2], masked).
diffusers forward signature:
    forward(self, x, t, cap_feats, ...)
      x        : list of image tensors, each [C, F, H, W] = [16, 1, 16, 16]
      t        : scalar/0-d (or [B]) flow time = (1000-timestep)/1000
      cap_feats: list of caption tensors, each [cap_len, feat_dim] = [64, 2560]
`omni_mode = isinstance(x[0], list)`. Here x[0] is a Tensor -> basic (non-omni)
mode, single image + single caption per sample.

Velocity convention
--------------------
We dump the RAW transformer `.sample` (stacked over the output list, then
squeezed on dim=2): velocity[1,16,16,16]. The Mojo wrapper returns this same RAW
velocity. Serenity's `predicted_flow = -velocity` -- the negation is applied
OUTSIDE the transformer and is NOT baked into the dumped bin.
"""

import json
import os

import numpy as np
import torch

from diffusers import ZImageTransformer2DModel  # same import Serenity uses

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
MODEL_DIR = "/home/alex/.serenity/models/zimage_base/transformer"
OUT_DIR = "/home/alex/serenity-trainer/parity"
SEED = 1234

B = 1
C = 16          # in_channels
F = 1           # frame dim (latent_input.unsqueeze(2))
H = 16
W = 16
CAP_LEN = 64
CAP_FEAT_DIM = 2560

TIMESTEP = 250
T_MODEL = (1000 - TIMESTEP) / 1000.0   # = 0.75 ; Serenity's (1000-timestep)/1000

DEVICE = "cuda"
DTYPE = torch.bfloat16


def dump_f32(path: str, arr_f32_rowmajor: np.ndarray):
    """Write a contiguous, row-major (C-order) little-endian float32 bin."""
    a = np.ascontiguousarray(arr_f32_rowmajor, dtype="<f4")
    a.tofile(path)
    return a.nbytes


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- FIXED inputs from numpy seed ----
    rng = np.random.default_rng(SEED)
    # Latent in EXACT transformer-input layout [B, C, F, H, W] = [1,16,1,16,16]
    latent_np = rng.standard_normal((B, C, F, H, W)).astype(np.float32)
    # Caption hidden states [CAP_LEN, CAP_FEAT_DIM] = [64, 2560]
    cap_np = rng.standard_normal((CAP_LEN, CAP_FEAT_DIM)).astype(np.float32)

    # ---- Load REAL transformer ----
    print(f"[load] {MODEL_DIR}")
    model = ZImageTransformer2DModel.from_pretrained(MODEL_DIR, torch_dtype=DTYPE)
    model = model.to(DEVICE).eval()

    # ---- Build inputs the Serenity way ----
    # latent_input = scaled_noisy_latent_image.unsqueeze(2) -> [B,16,1,H,W]
    # latent_input_list = list(latent_input.unbind(dim=0))  -> list of [16,1,H,W]
    latent_t = torch.from_numpy(latent_np).to(device=DEVICE, dtype=DTYPE)  # [1,16,1,16,16]
    latent_input_list = list(latent_t.unbind(dim=0))                       # [ [16,1,16,16] ]

    # cap_feats: list of per-sample [cap_len, feat_dim] tensors (embeddings_list)
    cap_t = torch.from_numpy(cap_np).to(device=DEVICE, dtype=DTYPE)        # [64,2560]
    cap_feats_list = [cap_t]                                               # one sample

    # t passed to forward: (1000-timestep)/1000, as a tensor (Serenity passes a
    # tensor of shape [B]; t_embedder broadcasts). Use [B] float for safety.
    t_model = torch.full((B,), T_MODEL, device=DEVICE, dtype=DTYPE)

    # ---- Forward ----
    with torch.no_grad():
        out = model(
            latent_input_list,
            t_model,
            cap_feats_list,
            return_dict=True,
        )
    output_list = out.sample                       # list of [16,1,16,16] per sample
    velocity = torch.stack(output_list, dim=0).squeeze(dim=2)   # [B,16,16,16]  RAW .sample
    # NOTE: Serenity's predicted_flow = -velocity (negation NOT applied here).

    velocity_f32 = velocity.to(torch.float32).cpu().numpy()    # [1,16,16,16]

    # ---- Stats ----
    v = velocity_f32
    nonfinite = int(np.count_nonzero(~np.isfinite(v)))
    print("[velocity] shape:", v.shape)
    print(f"[velocity] mean={v.mean():.8f} std={v.std():.8f} "
          f"min={v.min():.8f} max={v.max():.8f} nonfinite={nonfinite}")

    # ---- Dump bins (all float32, row-major / C-order) ----
    # latent dumped as [1,16,16,16] (drop the singleton F dim) to match velocity layout
    latent_dump = latent_np.reshape(B, C, H, W)    # [1,16,16,16]
    p_lat = os.path.join(OUT_DIR, "zi_fwd_latent.bin")
    p_cap = os.path.join(OUT_DIR, "zi_fwd_cap.bin")
    p_vel = os.path.join(OUT_DIR, "zi_fwd_velocity.bin")
    n_lat = dump_f32(p_lat, latent_dump)
    n_cap = dump_f32(p_cap, cap_np)
    n_vel = dump_f32(p_vel, velocity_f32)
    print(f"[dump] {p_lat}  {latent_dump.shape}  {n_lat} bytes")
    print(f"[dump] {p_cap}  {cap_np.shape}  {n_cap} bytes")
    print(f"[dump] {p_vel}  {velocity_f32.shape}  {n_vel} bytes")

    # ---- Meta ----
    meta = {
        "seed": SEED,
        "timestep": TIMESTEP,
        "t_model": T_MODEL,
        "t_model_formula": "(1000 - timestep) / 1000",
        "device": DEVICE,
        "compute_dtype": "bfloat16",
        "model_dir": MODEL_DIR,
        "velocity_convention": (
            "RAW transformer .sample (stacked over output_list, squeezed dim=2). "
            "predicted_flow = -velocity is applied OUTSIDE the transformer and is "
            "NOT baked into zi_fwd_velocity.bin."
        ),
        "text_encoder_output_structure": {
            "description": (
                "Serenity passes `cap_feats` (the 3rd positional arg to "
                "ZImageTransformer2DModel.forward) as a Python list with one entry "
                "per batch sample. Each entry is a 2-D tensor [caption_len, feat_dim]. "
                "This is ZImageModel.encode_text's `embeddings_list` = the Qwen3 "
                "hidden_states[-2] masked per-sample. omni_mode is False because "
                "x[0] and cap_feats[0] are Tensors, not lists."
            ),
            "python_type": "list[torch.Tensor]",
            "num_entries": B,
            "entry_shape": [CAP_LEN, CAP_FEAT_DIM],
            "entry_dtype": "bfloat16",
        },
        "x_input_structure": {
            "description": (
                "First positional arg `x`: list of image tensors, one per sample, "
                "each [C, F, H, W]. Built by latent.unsqueeze(2).unbind(dim=0)."
            ),
            "python_type": "list[torch.Tensor]",
            "num_entries": B,
            "entry_shape": [C, F, H, W],
            "entry_dtype": "bfloat16",
        },
        "files": {
            "zi_fwd_latent.bin": {
                "shape": [B, C, H, W], "dtype": "float32", "order": "row-major (C)",
                "bytes": n_lat,
                "note": "Latent as [1,16,16,16] (F=1 squeezed). The [1,16,1,16,16] tensor fed to transformer is this reshaped.",
            },
            "zi_fwd_cap.bin": {
                "shape": [CAP_LEN, CAP_FEAT_DIM], "dtype": "float32", "order": "row-major (C)",
                "bytes": n_cap,
                "note": "Caption hidden states fed as cap_feats[0].",
            },
            "zi_fwd_velocity.bin": {
                "shape": [B, C, H, W], "dtype": "float32", "order": "row-major (C)",
                "bytes": n_vel,
                "note": "RAW transformer .sample. predicted = -velocity (separate).",
            },
        },
        "velocity_stats": {
            "shape": list(v.shape),
            "mean": float(v.mean()), "std": float(v.std()),
            "min": float(v.min()), "max": float(v.max()),
            "nonfinite": nonfinite,
        },
    }
    p_meta = os.path.join(OUT_DIR, "zi_fwd_meta.json")
    with open(p_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[dump] {p_meta}")
    print("[done]")


if __name__ == "__main__":
    main()
