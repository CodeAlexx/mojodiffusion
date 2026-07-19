#!/usr/bin/env python3
# flux_denoise_oracle.py — BFL torch reference for the FULL FLUX.1-dev denoise
# loop (20 Euler steps), the capstone integration gate.
#
# Pins the initial noise latent + text embeds (so the RNG-stream difference
# between the Mojo custom randn and torch is removed — see the caveat below) and
# runs the SAME 20-step flow-match Euler loop the Mojo CLI runs, with the REAL
# flux1-dev DiT (bf16 weights upcast to fp32). Tiny 4x4 token grid keeps it fast
# on CPU and exercises the full block stack + schedule + Euler integration.
#
# RNG CAVEAT: a real end-to-end "same seed" image diff is impossible — the Mojo
# randn (custom PCG/Box-Muller) and torch generators produce different noise
# streams. So we PIN the noise here and feed identical bytes to both sides; this
# gate measures the denoise math (DiT x20 + schedule + Euler), not RNG.
#
# Convention: timesteps/guidance passed RAW; BFL timestep_embedding applies the
# x1000 internally; the Mojo probe pre-scales by 1000. Euler: img += (t_prev -
# t_curr) * pred. Schedule = get_schedule(20, image_seq_len=N_IMG).
#
# Dumps: flux_dn_noise.bin [1,16,64], flux_dn_txt.bin [1,16,4096],
#        flux_dn_vec.bin [1,768], flux_dn_final.bin [1,16,64], flux_dn_meta.txt
#
# Usage: python3 flux_denoise_oracle.py [H2] [W2] [N_TXT] [STEPS]

import sys, os, math
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


def get_schedule(num_steps, image_seq_len, base_shift=0.5, max_shift=1.15):
    # BFL get_schedule (linear mu, exp time_shift, sigma=1).
    m = (max_shift - base_shift) / (4096 - 256)
    b = base_shift - m * 256
    mu = m * image_seq_len + b
    ts = [1 - i / num_steps for i in range(num_steps + 1)]
    def shift(t):
        if t <= 0 or t >= 1:
            return t
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1))
    return [shift(t) for t in ts]


def main():
    H2 = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    W2 = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    N_TXT = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    N_IMG = H2 * W2
    GUID = 3.5

    sd = load_file(DIT_PATH)
    for k in list(sd.keys()):
        sd[k] = sd[k].to(torch.float32)
    with torch.device("meta"):
        model = Flux(PARAMS)
    model.load_state_dict(sd, strict=False, assign=True)
    model = model.eval()

    g = torch.Generator().manual_seed(7)
    noise = torch.randn(1, N_IMG, 64, generator=g, dtype=torch.float32)
    txt = torch.randn(1, N_TXT, 4096, generator=g, dtype=torch.float32)
    vec = torch.randn(1, 768, generator=g, dtype=torch.float32)

    img_ids = torch.zeros(1, N_IMG, 3, dtype=torch.float32)
    rows = torch.arange(H2).view(H2, 1).expand(H2, W2).reshape(-1)
    cols = torch.arange(W2).view(1, W2).expand(H2, W2).reshape(-1)
    img_ids[0, :, 1] = rows.to(torch.float32)
    img_ids[0, :, 2] = cols.to(torch.float32)
    txt_ids = torch.zeros(1, N_TXT, 3, dtype=torch.float32)

    sched = get_schedule(STEPS, N_IMG)
    print(f"[oracle] schedule[0..3]={[round(x,5) for x in sched[:4]]} ... [-1]={sched[-1]}")
    img = noise.clone()
    with torch.no_grad():
        for i in range(STEPS):
            t_curr, t_prev = sched[i], sched[i + 1]
            t_vec = torch.tensor([t_curr], dtype=torch.float32)
            g_vec = torch.tensor([GUID], dtype=torch.float32)
            pred = model(img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids,
                         timesteps=t_vec, y=vec, guidance=g_vec)
            img = img + (t_prev - t_curr) * pred
    print(f"[oracle] final latent {list(img.shape)} mean={img.mean().item():.6f} "
          f"std={img.std().item():.6f} min={img.min().item():.6f} max={img.max().item():.6f}")

    dump(f"{OUT_DIR}/flux_dn_noise.bin", noise)
    dump(f"{OUT_DIR}/flux_dn_txt.bin", txt)
    dump(f"{OUT_DIR}/flux_dn_vec.bin", vec)
    dump(f"{OUT_DIR}/flux_dn_final.bin", img)
    with open(f"{OUT_DIR}/flux_dn_meta.txt", "w") as f:
        f.write(f"H2={H2} W2={W2} N_IMG={N_IMG} N_TXT={N_TXT} STEPS={STEPS} guidance={GUID}\n")
        f.write(f"final mean={img.mean().item():.6f} std={img.std().item():.6f}\n")
    print("[oracle] dumped noise/txt/vec/final")


if __name__ == "__main__":
    main()
