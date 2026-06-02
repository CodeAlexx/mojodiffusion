#!/usr/bin/env python3
# gen_lq_projection_reference.py — PyTorch reference for PiD LQProjection2D
# (latent-only branch). Builds the REAL repo class with seeded RANDOM weights,
# runs a tiny latent input, and dumps weights + input + reference output to a
# .safetensors for the Mojo unit-gate.
#
# Config (matches the memory's PiD config, latent branch):
#   in_channels=0, latent_channels=16, hidden_dim=512, out_dim=1536,
#   patch_size=16, sr_scale=4, latent_spatial_down_factor=8 -> z_to_patch_ratio=2.0
#   num_res_blocks=4, num_outputs=1, zero_init=False (so output head is nonzero),
#   pit_output=False.
#
# To make the spatial alignment a numerical no-op (isolate the conv stack + head),
# we feed the latent ALREADY at the patch grid (zH=pH, zW=pW). The z_to_patch_ratio
# is 2.0 (>1) so the latent branch does F.interpolate(..., mode="nearest",
# size=(pH,pW)); with zH==pH and zW==pW that interpolate is identity.
#
# Run with SYSTEM python3 (has torch); the pixi env lacks torch.
#   /usr/bin/python3 serenitymojo/models/pid/parity/gen_lq_projection_reference.py

import os
import sys

import torch

# Make the cloned PiD repo importable.
REPO = "/tmp/PiD_repo"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from pid._src.networks.lq_projection_2d import LQProjection2D  # noqa: E402

try:
    from safetensors.torch import save_file
except Exception as e:  # pragma: no cover
    print("ERROR: safetensors not available:", e)
    sys.exit(1)

SEED = 1234
torch.manual_seed(SEED)

# --- config (latent-only branch) ---
LATENT_CHANNELS = 16
HIDDEN_DIM = 512
OUT_DIM = 1536
PATCH_SIZE = 16
SR_SCALE = 4
LSDF = 8
NUM_RES_BLOCKS = 4
NUM_OUTPUTS = 1
NUM_GROUPS = 4  # ResBlock GroupNorm default

# tiny patch grid
pH = 4
pW = 4
B = 1

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(OUT_DIR, "lq_projection_ref.safetensors")


def main():
    model = LQProjection2D(
        in_channels=0,
        latent_channels=LATENT_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        out_dim=OUT_DIM,
        patch_size=PATCH_SIZE,
        sr_scale=SR_SCALE,
        latent_spatial_down_factor=LSDF,
        num_res_blocks=NUM_RES_BLOCKS,
        num_outputs=NUM_OUTPUTS,
        zero_init=False,
        pit_output=False,
    )
    # init_weights() uses zero_init=False -> output head gets small trunc_normal;
    # but it leaves Conv biases zeroed and GroupNorm at default (gamma=1,beta=0).
    # To make the gate meaningful (nonzero GN affine, nonzero conv bias) we
    # randomize ALL parameters AFTER constructing, so every op is exercised.
    model.eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            # small-ish random so activations stay bounded
            p.copy_(torch.randn_like(p) * 0.1)

    # z_to_patch_ratio = (sr_scale*lsdf)/patch_size = (4*8)/16 = 2.0
    z_ratio = (SR_SCALE * LSDF) / PATCH_SIZE
    assert abs(z_ratio - 2.0) < 1e-9, z_ratio
    # latent fed already at the patch grid -> nearest interpolate is identity.
    lq_latent = torch.randn(B, LATENT_CHANNELS, pH, pW)

    with torch.no_grad():
        outputs = model(lq_latent=lq_latent, target_pH=pH, target_pW=pW)
    assert len(outputs) == NUM_OUTPUTS, len(outputs)
    out = outputs[0]  # [B, N, OUT_DIM], N = pH*pW
    assert out.shape == (B, pH * pW, OUT_DIM), out.shape

    # --- collect tensors to dump ---
    tensors = {}
    # input latent in NCHW (as PyTorch holds it).
    tensors["lq_latent"] = lq_latent.contiguous().float()
    tensors["ref_output"] = out.contiguous().float()

    # latent_proj is an nn.Sequential:
    #   [0] Conv2d(16->512), [1] SiLU, [2] Conv2d(512->512),
    #   [3..6] ResBlock (each: GN, SiLU, Conv, GN, SiLU, Conv)
    lp = model.latent_proj
    tensors["proj.conv0.weight"] = lp[0].weight.contiguous().float()  # [512,16,3,3] OIHW
    tensors["proj.conv0.bias"] = lp[0].bias.contiguous().float()
    tensors["proj.conv1.weight"] = lp[2].weight.contiguous().float()  # [512,512,3,3]
    tensors["proj.conv1.bias"] = lp[2].bias.contiguous().float()
    for i in range(NUM_RES_BLOCKS):
        rb = lp[3 + i].block  # Sequential(GN, SiLU, Conv, GN, SiLU, Conv)
        tensors[f"proj.res{i}.gn0.weight"] = rb[0].weight.contiguous().float()
        tensors[f"proj.res{i}.gn0.bias"] = rb[0].bias.contiguous().float()
        tensors[f"proj.res{i}.conv0.weight"] = rb[2].weight.contiguous().float()
        tensors[f"proj.res{i}.conv0.bias"] = rb[2].bias.contiguous().float()
        tensors[f"proj.res{i}.gn1.weight"] = rb[3].weight.contiguous().float()
        tensors[f"proj.res{i}.gn1.bias"] = rb[3].bias.contiguous().float()
        tensors[f"proj.res{i}.conv1.weight"] = rb[5].weight.contiguous().float()
        tensors[f"proj.res{i}.conv1.bias"] = rb[5].bias.contiguous().float()

    # output head (Linear 512->1536)
    head = model.output_heads[0]
    tensors["head.weight"] = head.weight.contiguous().float()  # [1536,512]
    tensors["head.bias"] = head.bias.contiguous().float()

    save_file(tensors, OUT_PATH)
    print("WROTE", OUT_PATH)
    print("config: latent_channels=%d hidden=%d out=%d res_blocks=%d groups=%d" % (
        LATENT_CHANNELS, HIDDEN_DIM, OUT_DIM, NUM_RES_BLOCKS, NUM_GROUPS))
    print("pH=%d pW=%d N=%d  z_to_patch_ratio=%.1f (identity nearest)" % (
        pH, pW, pH * pW, z_ratio))
    print("ref_output shape", tuple(out.shape),
          "mean %.5f std %.5f" % (out.mean().item(), out.std().item()))


if __name__ == "__main__":
    main()
