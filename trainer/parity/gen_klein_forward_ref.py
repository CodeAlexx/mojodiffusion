#!/usr/bin/env python
"""
Numeric forward reference for Serenity's REAL Klein (Flux2) transformer.

Reference source: Serenity + diffusers Flux2Transformer2DModel ONLY.
  - diffusers forward:  venv/src/diffusers/.../transformers/transformer_flux2.py
                        Flux2Transformer2DModel.forward (lines 1178-1381)
  - reference trainer input assembly:  modules/modelSetup/BaseFlux2Setup.py::predict (lines 82-179)
  - reference trainer pack/patchify:   modules/model/Flux2Model.py
                          patchify_latents (297-302), scale_latents (313-318),
                          pack_latents (255-257), unpack_latents (260-262),
                          unpatchify_latents (305-310),
                          prepare_text_ids (281-294), prepare_latent_image_ids (240-251),
                          calculate_timestep_shift (267-278)

Run with: /home/alex/Serenity/venv/bin/python
          (cwd does NOT matter; this script does not import `modules`)

What this does
--------------
Loads the diffusers Flux2Transformer2DModel from the local FLUX.2-klein-base-9B
snapshot (transformer/) + the VAE bn stats (vae/), builds FIXED reproducible
inputs (numpy seed 1234) at a small 256px resolution, runs the transformer forward
the EXACT way BaseFlux2Setup.predict does, and dumps a float32 safetensors + JSON
meta per the Klein PARITY CONTRACT.

RNG-INDEPENDENT contract (avoids the torch-vs-Mojo generator mismatch the port
documents at Flux2LoRASetup.mojo:234-238)
----------------------------------------------------------------------------------
We do NOT rely on either side reproducing the other's RNG stream. Instead we dump
the actual NOISED transformer image input (`scaled_noisy`, pre-patchify [1,32,32,32])
AND its derivation pieces, so the Mojo verification smoke can feed byte-identical
inputs and compare the transformer output directly. We ALSO dump the raw clean
latent + timestep so a full-`predict` path can be checked if both sides agree on
the noise (the port can be fed the dumped noise).

Klein dims (measured from the snapshot config)
----------------------------------------------
  VAE: latent_channels=32, vae_scale_factor=8, batch_norm_eps=1e-4, bn stats [128]
  transformer: in_channels=128, out_channels=128, num_layers=8, num_single_layers=24,
               num_attention_heads=32, attention_head_dim=128 (inner_dim=4096=KDIM),
               joint_attention_dim=12288=KTXT_CH, axes_dims_rope=[32,32,32,32],
               rope_theta=2000, guidance_embeds=False, eps=1e-6
  Qwen3 text encoder hidden_size=4096; 3 layers concatenated -> 12288 = txt feat dim

  256px image -> raw VAE latent [1,32,32,32]; patchify/2 -> [1,128,16,16];
  pack -> [1, N_IMG=256, 128].  NTXT chosen small/fixed.

Velocity convention
-------------------
Flux2 does NOT negate the model output (UNLIKE Z-Image; see
Flux2LoRASetup.mojo:264-266). The dumped `velocity` is the diffusers transformer
`.sample` run through Serenity's unpack_latents + unpatchify_latents, i.e. the
`predicted_flow` in BaseFlux2Setup.predict (line 153-164) BEFORE the
`predicted = unpatchify_latents(predicted_flow)` — actually we apply unpatchify too,
so `velocity` is exactly reference trainer's `model_output_data['predicted']` shape [1,32,32,32].
"""

import json
import os

import numpy as np
import torch
from safetensors.torch import save_file

from diffusers import Flux2Transformer2DModel  # same import Serenity uses
from safetensors import safe_open

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
SNAP = (
    "/home/alex/.cache/huggingface/hub/"
    "models--black-forest-labs--FLUX.2-klein-base-9B/snapshots/"
    "32773329fbe7e81a90ef971740e8ba4b0364ecf3"
)
TRANSFORMER_DIR = os.path.join(SNAP, "transformer")
VAE_DIR = os.path.join(SNAP, "vae")
OUT_DIR = "/home/alex/serenity-trainer/parity"
SEED = 1234

# 256px -> raw VAE latent 32x32 (latent_channels=32); patchify/2 -> 16x16
B = 1
LATENT_CH = 32          # VAE latent_channels (pre-patchify)
HL_IMG = 32             # raw latent height (diffusers in_channels=16 in the contract
WL_IMG = 32             # is a misnomer; Flux2 Klein VAE has 32 latent channels)
PATCH = 2               # Flux2Model.patchify_latents folds 2x2 -> *4 channels
HL = HL_IMG // PATCH    # 16  (patchified height)
WL = WL_IMG // PATCH    # 16  (patchified width)
IN_CH = LATENT_CH * 4   # 128 (patchified channel count = transformer in_channels)
N_IMG = HL * WL         # 256 image tokens

NTXT = 48               # fixed small text sequence length
TXT_DIM = 12288         # joint_attention_dim (= Qwen3 3x4096)

BATCH_NORM_EPS = 1e-4

# Timestep: BaseFlux2Setup.predict passes timestep/1000 to the transformer
# (line 144); the transformer re-scales x1000 internally (line 1231). We pick a
# fixed integer timestep and a fixed sigma-derived noised input.
TIMESTEP = 250          # integer discrete timestep (0..999)
T_MODEL = TIMESTEP / 1000.0   # value handed to transformer `timestep=` arg
# FlowMatch sigma for noising: Serenity _add_noise_discrete uses
# sigma = sigmas[idx] with sigmas = (timesteps+1)/N for the discrete schedule
# (ModelSetupFlowMatchingMixin). For the parity dump we fix sigma directly so the
# noised input is deterministic and RNG-free.
SIGMA = (TIMESTEP + 1) / 1000.0   # = 0.251

GUIDANCE_SCALE = 1.0
GUIDANCE_EMBEDS = False   # FLUX.2-klein-base-9B

DEVICE = "cuda"
DTYPE = torch.bfloat16


def load_bn_stats():
    """Read VAE bn.running_mean / running_var ([128] each) for scale_latents."""
    f = os.path.join(VAE_DIR, "diffusion_pytorch_model.safetensors")
    with safe_open(f, "pt") as g:
        mean = g.get_tensor("bn.running_mean").float()      # [128]
        var = g.get_tensor("bn.running_var").float()        # [128]
    return mean, var


# ── reference trainer Flux2Model input-assembly helpers, ported 1:1 ────────────────────────
def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    # Flux2Model.patchify_latents (297-302)
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(b, c * 4, h // 2, w // 2)
    return latents


def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    # Flux2Model.unpatchify_latents (305-310)
    b, c, h, w = latents.shape
    latents = latents.reshape(b, c // 4, 2, 2, h, w)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(b, c // 4, h * 2, w * 2)
    return latents


def scale_latents(latents, bn_mean, bn_var) -> torch.Tensor:
    # Flux2Model.scale_latents (313-318)
    m = bn_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    s = torch.sqrt(bn_var.view(1, -1, 1, 1) + BATCH_NORM_EPS).to(latents.device, latents.dtype)
    return (latents - m) / s


def pack_latents(latents) -> torch.Tensor:
    # Flux2Model.pack_latents (255-257)
    b, c, h, w = latents.shape
    return latents.reshape(b, c, h * w).permute(0, 2, 1)


def unpack_latents(latents, height, width) -> torch.Tensor:
    # Flux2Model.unpack_latents (260-262)
    b, s, c = latents.shape
    return latents.reshape(b, height, width, c).permute(0, 3, 1, 2)


def prepare_latent_image_ids(latents) -> torch.Tensor:
    # Flux2Model.prepare_latent_image_ids (240-251) — on the PACKED latent
    # (called with latent_input = scaled_noisy [1,128,HL,WL])
    b, _, height, width = latents.shape
    t = torch.arange(1, device=latents.device)
    h = torch.arange(height, device=latents.device)
    w = torch.arange(width, device=latents.device)
    l_ = torch.arange(1, device=latents.device)
    latent_ids = torch.cartesian_prod(t, h, w, l_)
    latent_ids = latent_ids.unsqueeze(0).expand(b, -1, -1)
    return latent_ids


def prepare_text_ids(x) -> torch.Tensor:
    # Flux2Model.prepare_text_ids (281-294) — x is encoder_hidden_states [B,L,_]
    B_, L, _ = x.shape
    out_ids = []
    for _ in range(B_):
        t = torch.arange(1, device=x.device)
        h = torch.arange(1, device=x.device)
        w = torch.arange(1, device=x.device)
        l_ = torch.arange(L, device=x.device)
        coords = torch.cartesian_prod(t, h, w, l_)
        out_ids.append(coords)
    return torch.stack(out_ids)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- FIXED inputs from numpy seed ----
    rng = np.random.default_rng(SEED)
    # Raw VAE mean latent [1,32,32,32] (pre-patchify/pre-scale). This is exactly
    # what Flux2LoRASpec.latent consumes (Flux2LoRASetup.mojo:136-138,157).
    latent_np = rng.standard_normal((B, LATENT_CH, HL_IMG, WL_IMG)).astype(np.float32)
    # Fixed noise [1,32,32,32] (same layout as the clean latent).
    noise_np = rng.standard_normal((B, LATENT_CH, HL_IMG, WL_IMG)).astype(np.float32)
    # Text hidden states [NTXT, TXT_DIM].
    txt_np = rng.standard_normal((NTXT, TXT_DIM)).astype(np.float32)

    bn_mean, bn_var = load_bn_stats()
    print(f"[bn] mean[:3]={bn_mean[:3].tolist()}  var[:3]={bn_var[:3].tolist()}")

    # ---- Load REAL transformer ----
    print(f"[load] {TRANSFORMER_DIR}")
    model = Flux2Transformer2DModel.from_pretrained(TRANSFORMER_DIR, torch_dtype=DTYPE)
    model = model.to(DEVICE).eval()
    cfg = model.config
    assert cfg.in_channels == IN_CH, (cfg.in_channels, IN_CH)
    assert cfg.guidance_embeds == GUIDANCE_EMBEDS, cfg.guidance_embeds
    assert cfg.joint_attention_dim == TXT_DIM, (cfg.joint_attention_dim, TXT_DIM)

    # ---- Build inputs the Serenity (BaseFlux2Setup.predict) way ----
    latent_t = torch.from_numpy(latent_np).to(device=DEVICE, dtype=DTYPE)  # [1,32,32,32]
    noise_t = torch.from_numpy(noise_np).to(device=DEVICE, dtype=DTYPE)    # [1,32,32,32]
    txt_t = torch.from_numpy(txt_np).to(device=DEVICE, dtype=DTYPE)        # [NTXT,12288]

    # (1) patchify + scale clean latent (BaseFlux2Setup.py:107,110)
    patchified = patchify_latents(latent_t.float()).to(DTYPE)              # [1,128,16,16]
    scaled_latent = scale_latents(patchified, bn_mean, bn_var)            # [1,128,16,16]

    # Noise must also be patchified to the [1,128,16,16] space to combine with
    # scaled_latent (reference trainer creates noise on the SCALED latent shape, NoiseMixin).
    noise_patch = patchify_latents(noise_t.float()).to(DTYPE)             # [1,128,16,16]

    # (5) noised input + flow target (_add_noise_discrete + flow):
    #     x_t  = noise*sigma + scaled_latent*(1-sigma)  (FlowMatchingMixin:36-37)
    #     flow = noise - scaled_latent                   (BaseFlux2Setup.py:159)
    scaled_noisy = noise_patch * SIGMA + scaled_latent * (1.0 - SIGMA)    # [1,128,16,16]
    flow = noise_patch - scaled_latent                                   # [1,128,16,16]
    latent_input = scaled_noisy                                          # reference trainer line 130

    # guidance: None for klein-base-9B (guidance_embeds=False, reference trainer line 132-136)
    guidance = None

    # ids + pack (BaseFlux2Setup.py:138-140)
    text_ids = prepare_text_ids(txt_t.unsqueeze(0))          # [1, NTXT, 4]
    image_ids = prepare_latent_image_ids(latent_input)       # [1, N_IMG, 4]
    packed_latent_input = pack_latents(latent_input)         # [1, N_IMG, 128]

    timestep = torch.tensor([TIMESTEP], device=DEVICE, dtype=torch.float32)

    # ---- Forward (BaseFlux2Setup.py:142-151) ----
    with torch.no_grad():
        packed_predicted_flow = model(
            hidden_states=packed_latent_input.to(dtype=DTYPE),
            timestep=timestep / 1000,                         # reference trainer line 144
            guidance=guidance,
            encoder_hidden_states=txt_t.unsqueeze(0).to(dtype=DTYPE),
            txt_ids=text_ids,
            img_ids=image_ids,
            joint_attention_kwargs=None,
            return_dict=True,
        ).sample                                              # [1, N_IMG, 128]

    # PACKED transformer output [1,N_IMG,128] — the RAW model `.sample` BEFORE
    # unpack/unpatchify. This is the byte-identical layout the Mojo
    # klein_inference_forward returns (packed flow [N_IMG,128], no unpatchify),
    # so the parity smoke compares against THIS to avoid any layout drift.
    velocity_packed_f32 = packed_predicted_flow.to(torch.float32).cpu().numpy()  # [1,256,128]

    # unpack -> [1,128,16,16] (BaseFlux2Setup.py:153-157)
    predicted_flow = unpack_latents(
        packed_predicted_flow, latent_input.shape[2], latent_input.shape[3]
    )
    # unpatchify -> [1,32,32,32] (the 'predicted' reference trainer puts in model_output_data, :164)
    velocity = unpatchify_latents(predicted_flow)            # [1,32,32,32]
    target_unpatch = unpatchify_latents(flow)               # [1,32,32,32]

    velocity_f32 = velocity.to(torch.float32).cpu().numpy()
    v = velocity_f32
    nonfinite = int(np.count_nonzero(~np.isfinite(v)))
    print("[velocity] shape:", v.shape)
    print(f"[velocity] mean={v.mean():.8f} std={v.std():.8f} "
          f"min={v.min():.8f} max={v.max():.8f} nonfinite={nonfinite}")

    # reference trainer-loss-style sanity: flow-matching MSE(predicted, target) (calculate_loss
    # -> _flow_matching_losses is an MSE on (predicted - target); we report the
    # raw MSE here as a magnitude sanity, NOT the exact weighted reference trainer loss).
    tgt = target_unpatch.to(torch.float32).cpu().numpy()
    mse = float(np.mean((v - tgt) ** 2))
    print(f"[sanity] raw MSE(velocity, flow_target) = {mse:.6f}")

    # ---- Dump safetensors (float32) per PARITY CONTRACT ----
    # latent: raw VAE mean latent [1,32,32,32] (what Flux2LoRASpec.latent consumes)
    # txt:    text hidden states [NTXT,12288]
    # velocity: diffusers transformer output (unpatchified) [1,32,32,32]
    # PLUS the RNG-free intermediates so the Mojo smoke can bypass torch RNG:
    #   noise (raw [1,32,32,32]), scaled_noisy (patchified [1,128,16,16]),
    #   flow_target (unpatchified [1,32,32,32]).
    tensors = {
        "latent": torch.from_numpy(latent_np).contiguous(),                       # [1,32,32,32] f32
        "txt": torch.from_numpy(txt_np).contiguous(),                             # [NTXT,12288] f32
        "velocity": torch.from_numpy(velocity_f32).contiguous(),                  # [1,32,32,32] f32
        "velocity_packed": torch.from_numpy(velocity_packed_f32).contiguous(),    # [1,256,128] f32 (raw .sample, packed)
        "noise": torch.from_numpy(noise_np).contiguous(),                         # [1,32,32,32] f32
        "scaled_noisy_patched": scaled_noisy.to(torch.float32).cpu().contiguous(),# [1,128,16,16] f32
        "flow_target": torch.from_numpy(tgt).contiguous(),                        # [1,32,32,32] f32
        "bn_mean": bn_mean.contiguous(),                                          # [128] f32
        "bn_var": bn_var.contiguous(),                                            # [128] f32
    }
    p_st = os.path.join(OUT_DIR, "klein_fwd.safetensors")
    save_file(tensors, p_st)
    print(f"[dump] {p_st}")

    # ---- Meta ----
    meta = {
        "seed": SEED,
        "timestep": TIMESTEP,
        "t_model": T_MODEL,
        "t_model_formula": "timestep / 1000 (BaseFlux2Setup.py:144)",
        "sigma": SIGMA,
        "sigma_formula": "(timestep + 1) / 1000  (FlowMatch discrete; fixed RNG-free)",
        "guidance_scale": GUIDANCE_SCALE,
        "guidance_embeds": GUIDANCE_EMBEDS,
        "device": DEVICE,
        "compute_dtype": "bfloat16",
        "transformer_dir": TRANSFORMER_DIR,
        "vae_dir": VAE_DIR,
        "batch_norm_eps": BATCH_NORM_EPS,
        # contract shapes
        "HL_img": HL_IMG, "WL_img": WL_IMG,
        "HL_patch": HL, "WL_patch": WL,
        "LATENT_CH": LATENT_CH, "IN_CH": IN_CH,
        "N_IMG": N_IMG, "NTXT": NTXT, "txt_dim": TXT_DIM,
        "velocity_convention": (
            "diffusers Flux2 transformer .sample, unpacked (unpack_latents) then "
            "unpatchified (unpatchify_latents) -> reference trainer model_output_data['predicted'] "
            "shape [1,32,32,32]. Flux2 does NOT negate the model output (unlike "
            "Z-Image): velocity is the raw predicted flow."
        ),
        "input_assembly": {
            "description": (
                "BaseFlux2Setup.predict order: patchify_latents(latent.float) -> "
                "scale_latents (VAE bn) -> noise*sigma + scaled*(1-sigma) -> "
                "pack_latents -> transformer(hidden_states=packed, timestep/1000, "
                "encoder_hidden_states=txt, txt_ids, img_ids, guidance=None) -> "
                "unpack_latents -> unpatchify_latents."
            ),
            "img_ids": "prepare_latent_image_ids on [1,128,16,16] -> [1,256,4]",
            "txt_ids": "prepare_text_ids on [1,NTXT,12288] -> [1,NTXT,4]",
            "guidance": "None (guidance_embeds=False)",
        },
        "files": {
            "klein_fwd.safetensors": {
                "latent": {"shape": [B, LATENT_CH, HL_IMG, WL_IMG], "dtype": "float32",
                           "note": "raw VAE mean latent (pre-patchify/scale); = Flux2LoRASpec.latent"},
                "txt": {"shape": [NTXT, TXT_DIM], "dtype": "float32",
                        "note": "text hidden states fed as encoder_hidden_states[0]"},
                "velocity": {"shape": [B, LATENT_CH, HL_IMG, WL_IMG], "dtype": "float32",
                             "note": "transformer output, unpacked+unpatchified (reference trainer 'predicted')"},
                "velocity_packed": {"shape": [B, N_IMG, IN_CH], "dtype": "float32",
                                    "note": "RAW transformer .sample BEFORE unpack/unpatchify; "
                                            "byte-identical to klein_inference_forward output "
                                            "(packed [N_IMG,128]); compare the smoke against THIS"},
                "noise": {"shape": [B, LATENT_CH, HL_IMG, WL_IMG], "dtype": "float32",
                          "note": "raw noise (pre-patchify); patchify+combine on Mojo side"},
                "scaled_noisy_patched": {"shape": [B, IN_CH, HL, WL], "dtype": "float32",
                                         "note": "RNG-free transformer image input BEFORE pack_latents; "
                                                 "feed this to bypass torch RNG entirely"},
                "flow_target": {"shape": [B, LATENT_CH, HL_IMG, WL_IMG], "dtype": "float32",
                                "note": "unpatchified flow target (noise - scaled_latent)"},
                "bn_mean": {"shape": [IN_CH], "dtype": "float32",
                            "note": "VAE bn.running_mean for scale_latents"},
                "bn_var": {"shape": [IN_CH], "dtype": "float32",
                           "note": "VAE bn.running_var for scale_latents (eps=1e-4)"},
            },
        },
        "velocity_stats": {
            "shape": list(v.shape),
            "mean": float(v.mean()), "std": float(v.std()),
            "min": float(v.min()), "max": float(v.max()),
            "nonfinite": nonfinite,
            "raw_mse_vs_flow_target": mse,
        },
    }
    p_meta = os.path.join(OUT_DIR, "klein_fwd_meta.json")
    with open(p_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[dump] {p_meta}")
    print("[done]")


if __name__ == "__main__":
    main()
