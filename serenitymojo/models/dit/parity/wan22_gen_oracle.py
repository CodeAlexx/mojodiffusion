#!/usr/bin/env python3
# Wan2.2-TI2V-5B DiT oracle generator (DEV-ONLY, not imported at runtime).
#
# Loads the original-format WanModel (dim=3072, num_layers=30, num_heads=24,
# head_dim=128, in_dim=48, ffn_dim=14336, patch (1,2,2), theta=10000) from the
# bf16 diffusers-named single-stack checkpoint, runs ONE forward on a tiny grid
# (bf16 GPU), hooks block 0 to capture its input + output, and dumps:
#   - block0_in.bin       block 0 input x  [1, seq_len, dim] bf16->f32
#   - block0_e0.bin       per-token modulation e0 [1, seq_len, 6, dim] f32
#   - block0_context.bin  text embedding (after text_embedding MLP) [1,512,dim] f32
#   - block0_out.bin      block 0 output  [1, seq_len, dim] f32
#   - grid.txt            "F H W seq_len text_len ctx_len" + dims
#   - full_x_in.bin       raw latent input [in_dim, F, H, W] f32 (CHUNK B)
#   - full_t.txt          scalar timestep
#   - full_context_raw.bin raw text tokens [ctx_len, text_dim] f32 (CHUNK B)
#   - full_out.bin        final unpatchified output [out_dim, F, H, W] f32 (CHUNK B)
#
# Run: /home/alex/SimpleTuner/.venv/bin/python wan22_gen_oracle.py
import os, sys, math
import numpy as np
import torch

# Import the model module directly, bypassing wan/__init__ (needs easydict)
# and wan.modules.__init__ chain. Load model.py as a standalone module, but it
# does `from .attention import flash_attention` — provide a stub package.
import importlib.util, types
WAN_ROOT = "/home/alex/Wan2.2"
# Build a minimal `wan.modules` package namespace with a stub `attention`.
pkg_wan = types.ModuleType("wan"); pkg_wan.__path__ = [os.path.join(WAN_ROOT, "wan")]
pkg_mods = types.ModuleType("wan.modules"); pkg_mods.__path__ = [os.path.join(WAN_ROOT, "wan", "modules")]
sys.modules["wan"] = pkg_wan
sys.modules["wan.modules"] = pkg_mods
# Real attention module (it only imports torch/flash_attn-optional).
spec_a = importlib.util.spec_from_file_location(
    "wan.modules.attention", os.path.join(WAN_ROOT, "wan", "modules", "attention.py"))
mod_a = importlib.util.module_from_spec(spec_a)
sys.modules["wan.modules.attention"] = mod_a
spec_a.loader.exec_module(mod_a)
spec_m = importlib.util.spec_from_file_location(
    "wan.modules.model", os.path.join(WAN_ROOT, "wan", "modules", "model.py"))
mod_m = importlib.util.module_from_spec(spec_m)
sys.modules["wan.modules.model"] = mod_m
spec_m.loader.exec_module(mod_m)
WanModel = mod_m.WanModel

# flash-attn isn't installed; replace flash_attention with an SDPA equivalent.
# Inputs are [B, L, N, C] (B=1 here). k_lens (self-attn=seq_lens, cross=None).
def _sdpa_flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
                          softmax_scale=None, q_scale=None, causal=False,
                          window_size=(-1, -1), deterministic=False,
                          dtype=torch.bfloat16, version=None):
    out_dtype = q.dtype
    q = q.to(dtype); k = k.to(dtype); v = v.to(dtype)
    if q_scale is not None:
        q = q * q_scale
    # [B, L, N, C] -> [B, N, L, C]
    qt = q.transpose(1, 2)
    kt = k.transpose(1, 2)
    vt = v.transpose(1, 2)
    attn_mask = None
    if k_lens is not None:
        # Build a key-padding mask [B, 1, 1, Lk]; our seq has no padding so this
        # is all-valid, but honor it generally.
        b, _, lk, _ = kt.shape
        mask = torch.zeros(b, 1, 1, lk, dtype=torch.bool, device=kt.device)
        for i, kl in enumerate(k_lens.tolist()):
            mask[i, :, :, int(kl):] = True
        if mask.any():
            attn_mask = torch.zeros(b, 1, 1, lk, dtype=qt.dtype, device=kt.device)
            attn_mask.masked_fill_(mask, float("-inf"))
    o = torch.nn.functional.scaled_dot_product_attention(
        qt, kt, vt, attn_mask=attn_mask, dropout_p=0.0, scale=softmax_scale)
    o = o.transpose(1, 2).contiguous()  # [B, L, N, C]
    return o.to(out_dtype)

mod_m.flash_attention = _sdpa_flash_attention

torch.manual_seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = "/home/alex/.serenity/models/checkpoints/Wan2.2-TI2V-5B-bf16"
DEV = "cuda"

# Tiny grid: F=1 frame (patch_f=1), H=8, W=8 latent -> patch (1,2,2) -> grid (1,4,4) -> 16 tokens
IN_DIM = 48
F_LAT, H_LAT, W_LAT = 1, 8, 8
CTX_LEN = 12          # raw text token count (< text_len=512)
TEXT_DIM = 4096
TIMESTEP = 500.0

def dump(name, t):
    a = t.detach().to(torch.float32).cpu().contiguous().numpy().astype(np.float32)
    a.tofile(os.path.join(HERE, name + ".bin"))
    with open(os.path.join(HERE, name + ".shape"), "w") as f:
        f.write(" ".join(str(d) for d in a.shape))
    return a

def main():
    cfg = dict(model_type="ti2v", patch_size=(1, 2, 2), text_len=512,
               in_dim=IN_DIM, dim=3072, ffn_dim=14336, freq_dim=256,
               text_dim=TEXT_DIM, out_dim=IN_DIM, num_heads=24, num_layers=30,
               qk_norm=True, cross_attn_norm=True, eps=1e-6)
    model = WanModel(**cfg)

    # Load weights from the sharded bf16 checkpoint (diffusers-named, same keys).
    from safetensors.torch import load_file
    sd = {}
    import glob
    for shard in sorted(glob.glob(os.path.join(CKPT, "*.safetensors"))):
        sd.update(load_file(shard))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # WanModel has a non-persistent `freqs` buffer (rebuilt in __init__) — ignore it.
    miss_real = [k for k in missing if k != "freqs"]
    print("missing (non-freqs):", miss_real)
    print("unexpected:", unexpected[:5], "..." if len(unexpected) > 5 else "")
    assert not miss_real, f"missing weights: {miss_real}"
    del sd

    # Convert to bf16 on CPU BEFORE moving to GPU (the model was built in f32;
    # moving the f32 copy to GPU would need ~2x the VRAM).
    model = model.to(torch.bfloat16).eval()
    model = model.to(DEV)

    # Inputs
    g = torch.Generator(device="cpu").manual_seed(1234)
    x_lat = torch.randn(IN_DIM, F_LAT, H_LAT, W_LAT, generator=g, dtype=torch.float32)
    ctx = torch.randn(CTX_LEN, TEXT_DIM, generator=g, dtype=torch.float32) * 0.5

    x_in = [x_lat.to(DEV).to(torch.bfloat16)]
    ctx_in = [ctx.to(DEV).to(torch.bfloat16)]
    t = torch.tensor([TIMESTEP], device=DEV, dtype=torch.float32)

    F_p = F_LAT // 1; H_p = H_LAT // 2; W_p = W_LAT // 2
    n_patches = F_p * H_p * W_p
    seq_len = n_patches  # no padding for the tiny case (seq_len == n_patches)

    # Hook block 0 to capture input x and modulation e0 and context.
    cap = {}
    blk0 = model.blocks[0]
    def hook(mod, args, kwargs, output):
        # WanAttentionBlock.forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens)
        a = list(args)
        # x is first positional
        cap["x"] = a[0].detach().clone()
        # e is kwarg 'e'
        cap["e0"] = kwargs["e"].detach().clone() if "e" in kwargs else a[1].detach().clone()
        cap["context"] = kwargs["context"].detach().clone() if "context" in kwargs else None
        cap["out"] = output.detach().clone()
    h = blk0.register_forward_hook(hook, with_kwargs=True)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(x_in, t, ctx_in, seq_len=seq_len)
    h.remove()

    # out is a list of [out_dim, F, H, W]
    full_out = out[0]

    dump("wan22_block0_in", cap["x"])
    dump("wan22_block0_e0", cap["e0"])
    dump("wan22_block0_context", cap["context"])
    dump("wan22_block0_out", cap["out"])
    dump("wan22_full_x_in", x_lat)
    dump("wan22_full_context_raw", ctx)
    dump("wan22_full_out", full_out)

    with open(os.path.join(HERE, "wan22_grid.txt"), "w") as f:
        f.write(f"F={F_p} H={H_p} W={W_p} seq_len={seq_len} n_patches={n_patches} "
                f"text_len=512 ctx_len={CTX_LEN} dim=3072 nheads=24 headdim=128 "
                f"in_dim={IN_DIM} ffn=14336 timestep={TIMESTEP}\n")
        f.write(f"axes_dims=[44,42,42] theta=10000\n")

    print("seq_len", seq_len, "block0_in", tuple(cap["x"].shape),
          "e0", tuple(cap["e0"].shape), "context", tuple(cap["context"].shape),
          "block0_out", tuple(cap["out"].shape), "full_out", tuple(full_out.shape))
    print("OK")

if __name__ == "__main__":
    main()
