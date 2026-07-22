"""MageVAE ENCODE parity oracle (offline reference for the Mojo probe).

Loads the real MageVAE from the downloaded Mage-Flow-Edit-Turbo vae safetensors,
feeds a FIXED non-degenerate image y[1,3,64,64], runs the DETERMINISTIC encode
(sample_posterior=False -> the posterior MEAN; mage_vae.py:597-623) at the fixed
one-step conditioning (z_t=0, t=0; mage_vae.py:431-441 _DConvEncoder.forward_pred),
and dumps:
  - the input image y
  - the latent mean [1,128,4,4]
  - stage taps: cond_head0 (post head_blocks[0]), cond_projdown (post proj_down),
    s_fuse (post fuse_proj — the DiCoBlock input), s_block0 (post blocks[0]),
    s_blocklast (post blocks[20]), s_normout (post norm_out), out_proj (post
    proj_out, 256ch)
  - c_tembed: the encoder t_embedder(0) vector

Dumped for BOTH f32 (clean gate) and bf16 (ship dtype — the pipeline casts the
encoded latent to bf16, pipeline.py:503). The Mojo probe reads these + loads the
SAME real weights and compares each stage independently.

Run: /home/alex/OneTrainer/venv/bin/python \
     serenitymojo/models/vae/parity/mageflow_vae_encode_oracle.py
"""
import os
import torch
from safetensors.torch import save_file
from mage_flow.models.modules.mage_vae import MageVAE

CKPT = "/home/alex/.serenity/models/checkpoints/Mage-Flow-Edit-Turbo/vae/diffusion_pytorch_model.safetensors"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda"
IH = IW = 64  # image -> latent [1,128,4,4]


def _taps_for_dtype(dtype):
    torch.manual_seed(0)
    # Non-degenerate image. Real refs are in [-1,1]; randn is a representative,
    # non-degenerate proxy (the probe feeds the SAME tensor, so scale is moot).
    y = torch.randn(1, 3, IH, IW)

    model = MageVAE(CKPT, sample_posterior=False)
    model = model.to(DEV).to(dtype).eval()
    enc = model.dconv_encoder

    yc = y.to(DEV).to(dtype)
    taps = {}

    def hook(name):
        def _h(mod, inp, out):
            taps[name] = out.detach().float().cpu().contiguous()
        return _h

    handles = [
        enc.head_blocks[0].register_forward_hook(hook("cond_head0")),
        enc.proj_down.register_forward_hook(hook("cond_projdown")),
        enc.fuse_proj.register_forward_hook(hook("s_fuse")),
        enc.blocks[0].register_forward_hook(hook("s_block0")),
        enc.blocks[len(enc.blocks) - 1].register_forward_hook(hook("s_blocklast")),
        enc.norm_out.register_forward_hook(hook("s_normout")),
        enc.proj_out.register_forward_hook(hook("out_proj")),
    ]

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype == torch.bfloat16)):
            mean = model.encode(yc)  # sample_posterior=False -> deterministic mean

    for h in handles:
        h.remove()

    with torch.no_grad():
        t0 = torch.zeros(1, device=DEV, dtype=dtype)
        c = enc.t_embedder(t0).detach().float().cpu().contiguous()

    out = {
        "y": y.float().contiguous(),
        "mean": mean.detach().float().cpu().contiguous(),
        "c_tembed": c,
    }
    for k, v in taps.items():
        out[k] = v

    del model
    torch.cuda.empty_cache()
    return out


def main():
    for dtype, tag in [(torch.float32, "f32"), (torch.bfloat16, "bf16")]:
        taps = _taps_for_dtype(dtype)
        path = os.path.join(OUT_DIR, f"mageflow_vae_encode_oracle_{tag}.safetensors")
        save_file({k: v.to(torch.float32).contiguous() for k, v in taps.items()}, path)
        print(f"[{tag}] wrote {path}")
        for k, v in taps.items():
            print(f"    {k:14s} {tuple(v.shape)}  mean={v.float().mean():+.4f} std={v.float().std():.4f}")


if __name__ == "__main__":
    main()
