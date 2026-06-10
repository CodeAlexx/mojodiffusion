#!/usr/bin/env python3
# flux_dit_oracle.py — torch reference for the Flux.1-dev DiT forward.
#
# Reference = the black-forest-labs `Flux` model (the implementation the Mojo
# Flux1Offloaded was ported from), loading the REAL flux1-dev.safetensors. The
# weights are bf16 on disk; we upcast EXACTLY to fp32 (bf16 ⊂ fp32, lossless) and
# compute in fp32 — i.e. the "bf16 storage" the Mojo uses, with the cleanest
# accumulation. The Mojo computes blocks in bf16, so expect a bf16-compute floor:
# cos ~0.99 over the 57-block stack is PASS; >=0.999 is exact-level.
#
# Tiny token grid (H2=W2=4 -> N_IMG=16, N_TXT=16, S=32): the DiT is sequence-
# length agnostic, so a tiny grid exercises the FULL block stack with REAL
# weights while staying fast on CPU and avoiding the 24GB-weights OOM on GPU.
#
# Convention match with the Mojo CLI / forward:
#   * timesteps/guidance passed RAW (0.5 / 3.5); BFL timestep_embedding applies
#     the x1000 time_factor internally. The Mojo caller pre-scales by 1000 and
#     its embedder uses factor 1 -> identical embedding. The probe must feed
#     t*1000, g*1000.
#   * img_ids = (0, row, col) row-major; txt_ids = 0; ids = cat(txt_ids, img_ids).
#
# Dumps (raw F32 LE, no length prefix) into this dir:
#   flux_dit_img.bin  [1,16,64]   flux_dit_txt.bin [1,16,4096]
#   flux_dit_vec.bin  [1,768]     flux_dit_tg.bin  [2] = (t, guidance) RAW
#   flux_dit_pred.bin [1,16,64]   flux_dit_meta.txt
#
# Usage: python3 flux_dit_oracle.py [H2] [W2] [N_TXT]

import sys, os
import torch

sys.path.insert(0, "/home/alex/black-forest-labs-flux/src")
from flux.model import Flux, FluxParams
from safetensors.torch import load_file

DIT_PATH = "/home/alex/.serenity/models/checkpoints/flux1-dev.safetensors"
OUT_DIR = "/home/alex/mojodiffusion/serenitymojo/models/flux/parity"
os.makedirs(OUT_DIR, exist_ok=True)

PARAMS = FluxParams(
    in_channels=64, out_channels=64, vec_in_dim=768, context_in_dim=4096,
    hidden_size=3072, mlp_ratio=4.0, num_heads=24, depth=19,
    depth_single_blocks=38, axes_dim=[16, 56, 56], theta=10_000,
    qkv_bias=True, guidance_embed=True,
)


def dump(path, t):
    v = t.detach().to(torch.float32).contiguous().cpu().numpy().ravel()
    with open(path, "wb") as f:
        f.write(v.astype("<f4").tobytes())


def main():
    H2 = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    W2 = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    N_TXT = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    N_IMG = H2 * W2

    torch.manual_seed(0)
    DEV = "cpu"  # 24GB bf16 weights won't fit a 24GB card alongside CUDA ctx.

    # Avoid double-peak RAM: meta-init the model (no param alloc), convert the
    # state dict to fp32 IN PLACE (bf16 freed per-tensor), then assign=True so the
    # fp32 sd tensors BECOME the params (single 48GB copy, fits 55GB free).
    sd = load_file(DIT_PATH)                       # ~24GB bf16
    for k in list(sd.keys()):
        sd[k] = sd[k].to(torch.float32)           # in place -> ~48GB, bf16 freed
    with torch.device("meta"):
        model = Flux(PARAMS)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    model = model.eval()
    print(f"[oracle] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing[:5]:", missing[:5])
    if unexpected:
        print("  unexpected[:5]:", unexpected[:5])

    g = torch.Generator().manual_seed(1234)
    img = torch.randn(1, N_IMG, 64, generator=g, dtype=torch.float32)
    txt = torch.randn(1, N_TXT, 4096, generator=g, dtype=torch.float32)
    y = torch.randn(1, 768, generator=g, dtype=torch.float32)
    t_raw = torch.tensor([0.5], dtype=torch.float32)
    guid_raw = torch.tensor([3.5], dtype=torch.float32)

    # ids: img (0, row, col) row-major; txt all zeros.
    img_ids = torch.zeros(1, N_IMG, 3, dtype=torch.float32)
    rows = torch.arange(H2).view(H2, 1).expand(H2, W2).reshape(-1)
    cols = torch.arange(W2).view(1, W2).expand(H2, W2).reshape(-1)
    img_ids[0, :, 1] = rows.to(torch.float32)
    img_ids[0, :, 2] = cols.to(torch.float32)
    txt_ids = torch.zeros(1, N_TXT, 3, dtype=torch.float32)

    with torch.no_grad():
        pred = model(img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids,
                     timesteps=t_raw, y=y, guidance=guid_raw)
    print(f"[oracle] pred shape {list(pred.shape)} "
          f"mean={pred.mean().item():.6f} std={pred.std().item():.6f} "
          f"min={pred.min().item():.6f} max={pred.max().item():.6f}")

    dump(f"{OUT_DIR}/flux_dit_img.bin", img)
    dump(f"{OUT_DIR}/flux_dit_txt.bin", txt)
    dump(f"{OUT_DIR}/flux_dit_vec.bin", y)
    dump(f"{OUT_DIR}/flux_dit_tg.bin", torch.tensor([t_raw.item(), guid_raw.item()]))
    dump(f"{OUT_DIR}/flux_dit_pred.bin", pred)
    with open(f"{OUT_DIR}/flux_dit_meta.txt", "w") as f:
        f.write(f"H2={H2} W2={W2} N_IMG={N_IMG} N_TXT={N_TXT} S={N_IMG+N_TXT}\n")
        f.write(f"t_raw=0.5 guidance_raw=3.5 (probe must feed *1000)\n")
        f.write(f"pred_shape={list(pred.shape)} mean={pred.mean().item():.6f} "
                f"std={pred.std().item():.6f}\n")
    print("[oracle] dumped img/txt/vec/tg/pred")


if __name__ == "__main__":
    main()
