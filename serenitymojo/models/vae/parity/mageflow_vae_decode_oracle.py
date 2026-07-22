"""MageVAE DECODE parity oracle (offline reference for the Mojo probe).

Loads the real MageVAE from the downloaded Mage-Flow-Edit-Turbo vae safetensors,
feeds a FIXED non-degenerate latent z[1,128,4,4] at the pipeline's decode timestep
(t=0, noise=0 — see mage_vae.py:625-633 MageVAE.decode), and dumps:
  - the input latent z
  - the decoded RGB output [1,3,64,64]
  - stage taps: cond (CoD decoder out), s_after_sembed, s_after_block0,
    s_after_block20, x_after_xembed, x_after_decnet
  - t_embedder(0) vector c and block0's folded adaLN modulation [1,2304]

Written for BOTH f32 (clean gate) and bf16 (ship dtype — the pipeline casts the
vae to bf16, pipeline.py:757 + autocast bf16 at _decode_one). The Mojo probe reads
these + loads the SAME real weights and compares each stage independently.

Run: pixi run python \
     serenitymojo/models/vae/parity/mageflow_vae_decode_oracle.py
"""
import os
import torch
from safetensors.torch import save_file
from mage_flow.models.modules.mage_vae import MageVAE

CKPT = "/home/alex/.serenity/models/checkpoints/Mage-Flow-Edit-Turbo/vae/diffusion_pytorch_model.safetensors"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda"


def _taps_for_dtype(dtype):
    torch.manual_seed(0)
    # Non-degenerate latent. Encoder-mean latents are O(1); randn is representative.
    z = torch.randn(1, 128, 4, 4)

    model = MageVAE(CKPT, sample_posterior=False)
    model = model.to(DEV).to(dtype).eval()
    dm = model.decoder_model

    zc = z.to(DEV).to(dtype)

    taps = {}

    def hook(name):
        def _h(mod, inp, out):
            taps[name] = out.detach().float().cpu().contiguous()
        return _h

    handles = [
        dm.y_embedder.decoder.register_forward_hook(hook("cond")),
        # CoD-decoder internal taps for the patched-attn isolation gate:
        #   block[0]=ResnetBlock output == input to block[1]=AttnBlock(patch32)
        #   block[1]=AttnBlock output   (patched self-attention, replicate-pad)
        dm.y_embedder.decoder.block[0].register_forward_hook(hook("cod_block0_out")),
        dm.y_embedder.decoder.block[1].register_forward_hook(hook("cod_attn1_out")),
        dm.s_embedder.register_forward_hook(hook("s_sembed")),
        dm.blocks[0].register_forward_hook(hook("s_block0")),
        dm.blocks[len(dm.blocks) - 1].register_forward_hook(hook("s_blocklast")),
        dm.x_embedder.register_forward_hook(hook("x_xembed")),
        dm.dec_net.register_forward_hook(hook("x_decnet")),
    ]

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype == torch.bfloat16)):
            rgb = model.decode(zc)

    for h in handles:
        h.remove()

    # t_embedder(0) and block0 folded modulation (constant at t=0).
    with torch.no_grad():
        t0 = torch.zeros(1, device=DEV, dtype=dtype)
        c = dm.t_embedder(t0).detach().float().cpu().contiguous()
        # after _freeze_adaln_cache, adaLN_modulation is _ConstAdaLN(mod)
        mod0 = dm.blocks[0].adaLN_modulation.modulation.detach().float().cpu().contiguous()

    out = {
        "z": z.float().contiguous(),
        "rgb": rgb.detach().float().cpu().contiguous(),
        "c_tembed": c,
        "mod_block0": mod0,
    }
    for k, v in taps.items():
        out[k] = v

    del model
    torch.cuda.empty_cache()
    return out


def main():
    for dtype, tag in [(torch.float32, "f32"), (torch.bfloat16, "bf16")]:
        taps = _taps_for_dtype(dtype)
        # save as f32 for portability (Mojo probe upcasts anyway)
        path = os.path.join(OUT_DIR, f"mageflow_vae_decode_oracle_{tag}.safetensors")
        save_file({k: v.to(torch.float32).contiguous() for k, v in taps.items()}, path)
        print(f"[{tag}] wrote {path}")
        for k, v in taps.items():
            print(f"    {k:14s} {tuple(v.shape)}  mean={v.float().mean():+.4f} std={v.float().std():.4f}")


if __name__ == "__main__":
    main()
