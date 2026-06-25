"""Krea-2 SingleStreamDiT.forward WIRING parity oracle — REDUCED-block, RESIDENT.

Chunk 7a: gates the FORWARD WIRING (krea2_first -> temb/tmlp/tproj -> txtfusion ->
txtmlp -> cat(context,img) -> pad-to-256 -> _mask -> rope -> N x SingleStreamBlock
WITH the pad-to-256 mask -> last_layer -> slice image tokens) with a REDUCED config
(layers=4, else identical to KREA2_MMDIT_CONFIG) and SEEDED RANDOM weights,
RESIDENT (no offload, no real weights — that's 7b).

Sequence EXERCISES the pad-to-256 branch: TXTLEN=20, IMGLEN=40 (GH=5,GW=8) ->
L_FULL=60 -> padded to 256. The main blocks run WITH the bf16 pad-to-256 mask
(chunk 4 gated mask=None; this exercises the now-working masked path).

Inputs are built exactly like the reference's pipeline.prepare (latent ->
patchify -> img; context [B,TXTLEN,12,2560]; imgids -> pos; mask all-ones; t).
Reference is run on its NATURAL path (cuDNN SDPA, no backend substitution).

Run (GPU, ~3 GB resident, display-safe):
    /home/alex/serenityflow-v2/.venv/bin/python \
        serenitymojo/models/dit/parity/gen_krea2_forward_small.py
"""

import sys

import torch
from einops import rearrange, repeat
from safetensors.torch import save_file

# Import mmdit.py DIRECTLY (torch+einops only) to bypass the ai-toolkit package
# __init__ chain (which pulls torchao/quanto, absent here).
sys.path.insert(0, "/home/alex/ai-toolkit/extensions_built_in/diffusion_models/krea2/src")
from mmdit import SingleMMDiTConfig, SingleStreamDiT  # noqa: E402

# KREA2_MMDIT_CONFIG copied verbatim from krea2.py:55-68 (the "single_mmdit_large_wide"
# arch) — inlined to avoid importing krea2.py (which pulls the toolkit package).
KREA2_MMDIT_CONFIG = dict(
    features=6144,
    tdim=256,
    txtdim=2560,
    heads=48,
    kvheads=12,
    multiplier=4,
    layers=28,
    patch=2,
    channels=16,
    txtheads=20,
    txtkvheads=20,
    txtlayers=12,
)

OUT = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/krea2_forward_small_oracle.safetensors"
DEV = "cuda"
DTYPE = torch.bfloat16
LAYERS = 4


def main() -> None:
    assert torch.cuda.is_available(), "gen_krea2_forward_small.py MUST run on CUDA."

    cfg_d = dict(KREA2_MMDIT_CONFIG)
    cfg_d["layers"] = LAYERS
    config = SingleMMDiTConfig(**cfg_d)
    print(
        f"config: features={config.features} heads={config.heads} "
        f"kvheads={config.kvheads} layers={config.layers} patch={config.patch} "
        f"channels={config.channels} txtdim={config.txtdim} "
        f"txtlayers={config.txtlayers} theta={config.theta}",
        flush=True,
    )

    # ── reduced model, SEEDED RANDOM weights (scales/mod random *0.1, rest default) ──
    torch.manual_seed(1234)
    model = SingleStreamDiT(config)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(".scale") or name.endswith("mod.lin") or name.endswith("modulation.lin"):
                p.copy_(torch.randn_like(p) * 0.1)
    model = model.to(DEV, dtype=DTYPE)
    model.eval()
    model.disable_gradient_checkpointing()

    # dump ALL weights
    sd = {
        f"w.{k}": (v.detach().to(DTYPE) if v.is_floating_point() else v.detach()).cpu().contiguous()
        for k, v in model.state_dict().items()
    }

    # ── fixed seeded inputs: TXTLEN=20, IMGLEN=40 (GH=5,GW=8), L_FULL=60 -> pad 256 ──
    torch.manual_seed(20240624)
    B = 1
    PATCH = config.patch  # 2
    C = config.channels  # 16
    AE_SCALE = 8
    GH, GW = 5, 8
    IMGLEN = GH * GW  # 40
    HEIGHT = GH * AE_SCALE * PATCH  # 80
    WIDTH = GW * AE_SCALE * PATCH   # 128
    TXTLEN = 20
    N = config.txtlayers  # 12
    D = config.txtdim  # 2560

    latent = torch.randn(B, C, HEIGHT // AE_SCALE, WIDTH // AE_SCALE, device=DEV, dtype=DTYPE)
    img = rearrange(latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=PATCH, pw=PATCH)
    in_dim = C * PATCH * PATCH  # 64
    assert img.shape == (B, IMGLEN, in_dim), img.shape

    context = torch.randn(B, TXTLEN, N, D, device=DEV, dtype=DTYPE)

    imgids = torch.zeros((GH, GW, 3), device=DEV)
    imgids[..., 1] = torch.arange(GH, device=DEV)[:, None]
    imgids[..., 2] = torch.arange(GW, device=DEV)[None, :]
    imgpos = repeat(imgids, "h w three -> b (h w) three", b=B, three=3)
    txtpos = torch.zeros(B, TXTLEN, 3, device=DEV)
    pos = torch.cat((txtpos, imgpos), dim=1)
    L_FULL = TXTLEN + IMGLEN  # 60
    assert pos.shape == (B, L_FULL, 3), pos.shape
    assert L_FULL < 256, "small variant: L_FULL must be < 256 so the pad-to-256 branch runs"

    mask = torch.ones(B, L_FULL, device=DEV, dtype=torch.bool)  # all-ones (b==1 inference)
    t = torch.rand(B, device=DEV, dtype=torch.float32)

    print(
        f"inputs: img={tuple(img.shape)} context={tuple(context.shape)} "
        f"t={tuple(t.shape)} pos={tuple(pos.shape)} mask={tuple(mask.shape)} "
        f"(GH={GH} GW={GW} IMGLEN={IMGLEN} TXTLEN={TXTLEN} L_FULL={L_FULL} -> pad 256)",
        flush=True,
    )

    with torch.no_grad():
        velocity = model(img=img, context=context, t=t, pos=pos, mask=mask)
    assert velocity.shape == (B, IMGLEN, in_dim), velocity.shape
    print(f"velocity: {tuple(velocity.shape)}", flush=True)

    out = dict(sd)
    out["img"] = img.detach().to(DTYPE).cpu().contiguous()
    out["context"] = context.detach().to(DTYPE).cpu().contiguous()
    out["t"] = t.detach().float().cpu().contiguous()
    out["pos"] = pos.detach().float().cpu().contiguous()
    out["mask"] = mask.detach().to(torch.float32).cpu().contiguous()  # [B, L_FULL] 1/0
    out["velocity"] = velocity.detach().float().cpu().contiguous()
    out["meta_txtlen"] = torch.tensor([TXTLEN], dtype=torch.int32)
    out["meta_imglen"] = torch.tensor([IMGLEN], dtype=torch.int32)
    out["meta_lfull"] = torch.tensor([L_FULL], dtype=torch.int32)
    save_file(out, OUT)
    import os
    print(f"OK dumped {len(out)} tensors -> {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
