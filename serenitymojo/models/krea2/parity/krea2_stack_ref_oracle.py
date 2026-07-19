#!/usr/bin/env python3
# serenitymojo/models/krea2/parity/krea2_stack_ref_oracle.py
#
# Torch oracle for the KREA2 img-EDIT reference-conditioning projection (PORT of
# klein's img_in_ref) wired into the SINGLE-STREAM STACK, at REDUCED depth
# (NBLOCKS=4). This is krea2_stack_oracle.py + the ADDITIVE ref branch:
#
#     combined_base = cat(txtfusion(ctx), first(img))          # the existing input
#     img_in_ref    = leaf [features, in_ch]  (NONZERO)        # the NEW trained param
#     ref_tokens    = [1, imglen, in_ch]      (NONZERO)        # the reference image
#     ref_proj      = ref_tokens @ img_in_refᵀ                 # [1, imglen, features]
#     x0            = combined_leaf + scatter(ref_proj)        # img rows get the add
#     ... blocks ×NBLOCKS ... last ... velocity ... loss.backward()
#
# The Mojo stack takes `combined` (= combined_base, the SAME tensor the existing
# gate feeds) plus ref_tokens + img_in_ref_w, adds the ref projection to the image
# rows internally, and returns d_combined (grad into the block-stack input) +
# d_img_in_ref. This oracle dumps torch's d_combined (= combined_leaf.grad) and
# d_img_in_ref (= img_in_ref.grad) so the Mojo gate can compare cos>=0.999.
#
# NON-DEGENERATE: img_in_ref NONZERO (randn*0.02) so d_img_in_ref is well-defined;
# ref_tokens NONZERO; LoRA B NONZERO (as in the parent oracle).
#
# The ref branch is a PURE INPUT LINEAR (dimension-independent): reduced depth /
# real head config (HEADS=48,KVHEADS=12,HEADDIM=128,features=6144,in_ch=64) is a
# faithful oracle for the composition — the ref math does not depend on NBLOCKS.
#
# Run (SEPARATE command, never chained with && after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/SerenityTrainer/venv/bin/python \
#       serenitymojo/models/krea2/parity/krea2_stack_ref_oracle.py

import os
import sys

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from safetensors.torch import save_file
from torch.nn.attention import sdpa_kernel, SDPBackend

sys.path.insert(0, "/home/alex/ai-toolkit/extensions_built_in/diffusion_models/krea2/src")
import mmdit  # noqa: E402
from mmdit import SingleMMDiTConfig, SingleStreamDiT, temb  # noqa: E402


def _attention_math(q, k, v, mask=None, scale=None, gqa=False):
    with sdpa_kernel(SDPBackend.MATH):
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=scale, enable_gqa=gqa
        )
    return rearrange(x, "B H L D -> B L (H D)")


mmdit.attention = _attention_math

OUT = "/home/alex/mojodiffusion/serenitymojo/models/krea2/parity/krea2_stack_ref_oracle.safetensors"
DEV = "cuda"
DTYPE = torch.float32
NBLOCKS = 4

KREA2_MMDIT_CONFIG = dict(
    features=6144, tdim=256, txtdim=2560, heads=48, kvheads=12, multiplier=4,
    layers=28, patch=2, channels=16, txtheads=20, txtkvheads=20, txtlayers=12,
)

RANK = 8
ALPHA = 16.0
LSCALE = ALPHA / RANK  # 2.0
FEATURES = KREA2_MMDIT_CONFIG["features"]            # 6144
HEADS = KREA2_MMDIT_CONFIG["heads"]                  # 48
KVHEADS = KREA2_MMDIT_CONFIG["kvheads"]              # 12
HEADDIM = FEATURES // HEADS                          # 128
MLPDIM = 16384
SLOT_PATHS = [
    ("wq", "attn.wq"),
    ("wk", "attn.wk"),
    ("wv", "attn.wv"),
    ("gate", "attn.gate"),
    ("wo", "attn.wo"),
    ("mlp_gate", "mlp.gate"),
    ("mlp_up", "mlp.up"),
    ("mlp_down", "mlp.down"),
]


def _get_module(root, dotted):
    m = root
    for part in dotted.split("."):
        m = getattr(m, part)
    return m


class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, A: torch.Tensor, B: torch.Tensor):
        super().__init__()
        self.base = base
        self.A = torch.nn.Parameter(A)   # [rank, in]
        self.B = torch.nn.Parameter(B)   # [out, rank]

    def forward(self, x):
        return self.base(x) + LSCALE * ((x @ self.A.t()) @ self.B.t())


def mmdit_mask(mask: torch.Tensor) -> torch.Tensor:
    return mask.unsqueeze(1).unsqueeze(2) * mask.unsqueeze(1).unsqueeze(3)


def main() -> None:
    assert torch.cuda.is_available(), "krea2_stack_ref_oracle.py MUST run on CUDA."
    torch.manual_seed(1234)

    cfg_d = dict(KREA2_MMDIT_CONFIG)
    cfg_d["layers"] = NBLOCKS
    config = SingleMMDiTConfig(**cfg_d)
    print(
        f"config: features={config.features} heads={config.heads} "
        f"kvheads={config.kvheads} layers={config.layers} in_ch={config.channels*config.patch*config.patch} "
        f"mlpdim={MLPDIM}",
        flush=True,
    )

    model = SingleStreamDiT(config)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(".scale") or name.endswith("mod.lin") or name.endswith("modulation.lin"):
                p.copy_(torch.randn_like(p) * 0.1)
    model = model.to(DEV, dtype=DTYPE)
    model.eval()
    model.disable_gradient_checkpointing()

    for p in model.parameters():
        p.requires_grad_(False)

    # ── inject LoRA on the 8 Linears of each single-stream block ──
    g = torch.Generator(device="cpu").manual_seed(7)
    loras = {}
    mlp_gate_out = None
    for bi in range(NBLOCKS):
        block = model.blocks[bi]
        for (slot, path) in SLOT_PATHS:
            base = _get_module(block, path)
            in_f, out_f = base.in_features, base.out_features
            if slot == "mlp_gate":
                mlp_gate_out = out_f
            A = (torch.randn(RANK, in_f, generator=g) * 0.02).to(DEV, DTYPE)
            B = (torch.randn(out_f, RANK, generator=g) * 0.02).to(DEV, DTYPE)
            wrapped = LoRALinear(base, A, B).to(DEV, DTYPE)
            parent = _get_module(block, path.rsplit(".", 1)[0]) if "." in path else block
            setattr(parent, path.rsplit(".", 1)[-1], wrapped)
            loras[(bi, slot)] = (wrapped.A, wrapped.B)
    assert mlp_gate_out == MLPDIM, f"mlp hidden {mlp_gate_out} != expected {MLPDIM}"

    # ── fixed seeded inputs: TXTLEN + IMGLEN == 256 EXACTLY → no pad ──
    torch.manual_seed(20240625)
    B = 1
    PATCH = config.patch        # 2
    C = config.channels         # 16
    AE_SCALE = 8
    GH, GW = 14, 15
    IMGLEN = GH * GW            # 210
    HEIGHT = GH * AE_SCALE * PATCH
    WIDTH = GW * AE_SCALE * PATCH
    TXTLEN = 256 - IMGLEN      # 46
    N = config.txtlayers        # 12
    D = config.txtdim           # 2560
    in_dim = C * PATCH * PATCH  # 64

    latent = torch.randn(B, C, HEIGHT // AE_SCALE, WIDTH // AE_SCALE, device=DEV, dtype=DTYPE)
    img = rearrange(latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=PATCH, pw=PATCH)
    assert img.shape == (B, IMGLEN, in_dim), img.shape

    context = torch.randn(B, TXTLEN, N, D, device=DEV, dtype=DTYPE)

    imgids = torch.zeros((GH, GW, 3), device=DEV)
    imgids[..., 1] = torch.arange(GH, device=DEV)[:, None]
    imgids[..., 2] = torch.arange(GW, device=DEV)[None, :]
    imgpos = repeat(imgids, "h w three -> b (h w) three", b=B, three=3)
    txtpos = torch.zeros(B, TXTLEN, 3, device=DEV)
    pos = torch.cat((txtpos, imgpos), dim=1)
    L_FULL = TXTLEN + IMGLEN  # 256

    mask = torch.ones(B, L_FULL, device=DEV, dtype=torch.bool)
    t = torch.rand(B, device=DEV, dtype=torch.float32)

    # ── NEW: nonzero img_in_ref (the trained param) + nonzero ref_tokens ──
    torch.manual_seed(424242)
    img_in_ref = (torch.randn(FEATURES, in_dim, device=DEV, dtype=DTYPE) * 0.02)
    img_in_ref = img_in_ref.clone().requires_grad_(True)     # leaf → d_img_in_ref
    ref_tokens = (torch.randn(B, IMGLEN, in_dim, device=DEV, dtype=DTYPE) * 0.7)

    print(
        f"inputs: img={tuple(img.shape)} ref_tokens={tuple(ref_tokens.shape)} "
        f"img_in_ref={tuple(img_in_ref.shape)} "
        f"(IMGLEN={IMGLEN} TXTLEN={TXTLEN} L_FULL={L_FULL} -> NO pad)",
        flush=True,
    )

    with sdpa_kernel(SDPBackend.MATH):
        img_e = model.first(img)
        t_full = model.tmlp(temb(t, config.tdim, device=img.device, dtype=img.dtype))
        tvec = model.tproj(t_full)
        txtmask = mmdit_mask(mask[:, : context.shape[1]])
        ctx_fused = model.txtfusion(context, mask=txtmask)
        ctx_proj = model.txtmlp(ctx_fused)
        txtlen, imglen = ctx_proj.shape[1], img_e.shape[1]
        combined = torch.cat((ctx_proj, img_e), dim=1)                 # [B,256,F] the base input
        full_mask = mmdit_mask(mask)
        freqs = model.posemb(pos)

        # combined_base = the SAME block-stack input the existing gate feeds the Mojo
        # stack (cat(ctx, first(img))). A leaf clone gives us d_combined.
        combined_base = combined.detach()
        combined_leaf = combined_base.clone().requires_grad_(True)

        # ref projection scattered onto the IMAGE rows [txtlen:txtlen+imglen].
        ref_proj = ref_tokens @ img_in_ref.t()                          # [B, imglen, F]
        head = txtlen
        tail = L_FULL - txtlen - imglen
        parts = []
        if head > 0:
            parts.append(torch.zeros(B, head, FEATURES, device=DEV, dtype=DTYPE))
        parts.append(ref_proj)
        if tail > 0:
            parts.append(torch.zeros(B, tail, FEATURES, device=DEV, dtype=DTYPE))
        ref_scatter = torch.cat(parts, dim=1)                           # [B, L, F]

        x = combined_leaf + ref_scatter                                 # effective input
        for bi in range(NBLOCKS):
            x = model.blocks[bi](x, tvec.detach(), freqs, full_mask)
        final = model.last(x, t_full.detach())
        velocity = final[:, txtlen : txtlen + imglen, :]                # [B,IMGLEN,64]
        assert velocity.shape == (B, IMGLEN, in_dim), velocity.shape

        d_velocity = torch.sin(
            torch.arange(velocity.numel(), device=DEV, dtype=DTYPE) * 0.0011 + 0.07
        ).reshape_as(velocity) * 0.05
        (velocity * d_velocity).sum().backward()

    # ── dump everything the Mojo ref gate reconstructs ──
    out = {}

    def put(name, tensor):
        out[name] = tensor.detach().to(torch.float32).cpu().contiguous()

    # prepared single-stream INPUTS (fed verbatim by the gate)
    put("combined", combined_base)               # [1,256,6144]  block-stack input (base)
    put("tvec", tvec)                            # [1,1,6*6144]
    put("tmlp_out", t_full)                      # [1,1,6144]
    put("pos", pos)                              # [1,256,3]
    put("d_velocity", d_velocity)                # [1,IMGLEN,64]
    put("velocity", velocity)                    # [1,IMGLEN,64] reference output (WITH ref)

    # NEW: the ref inputs + the two deliverable grads
    put("ref_tokens", ref_tokens)               # [1,IMGLEN,64]
    put("img_in_ref", img_in_ref)               # [6144,64]  the trained param (nonzero)
    put("kref_ref.d_combined", combined_leaf.grad)   # [1,256,6144] input-token grads
    put("kref_ref.d_img_in_ref", img_in_ref.grad)    # [6144,64]    the load-bearing grad

    # per-block frozen weights + LoRA A/B (so the gate can build the stack)
    for bi in range(NBLOCKS):
        block = model.blocks[bi]
        for (slot, path) in SLOT_PATHS:
            wmod = _get_module(block, path)
            put(f"blk{bi}.{slot}.W", wmod.base.weight)
        put(f"blk{bi}.prenorm", block.prenorm.scale)
        put(f"blk{bi}.postnorm", block.postnorm.scale)
        put(f"blk{bi}.qnorm", block.attn.qknorm.qnorm.scale)
        put(f"blk{bi}.knorm", block.attn.qknorm.knorm.scale)
        put(f"blk{bi}.mod_lin", block.mod.lin)
        for (slot, _path) in SLOT_PATHS:
            A, Bp = loras[(bi, slot)]
            put(f"blk{bi}.{slot}.A", A)
            put(f"blk{bi}.{slot}.B", Bp)

    put("last.norm", model.last.norm.scale)
    put("last.mod_lin", model.last.modulation.lin)
    put("last.lin_w", model.last.linear.weight)
    put("last.lin_b", model.last.linear.bias)

    out["meta_txtlen"] = torch.tensor([TXTLEN], dtype=torch.int32)
    out["meta_imglen"] = torch.tensor([IMGLEN], dtype=torch.int32)
    out["meta_lfull"] = torch.tensor([L_FULL], dtype=torch.int32)
    out["meta_nblocks"] = torch.tensor([NBLOCKS], dtype=torch.int32)
    out["meta_mlpdim"] = torch.tensor([MLPDIM], dtype=torch.int32)
    out["meta_in_ch"] = torch.tensor([in_dim], dtype=torch.int32)

    save_file(out, OUT)
    print(
        f"velocity mean={float(velocity.detach().mean()):.6f} "
        f"std={float(velocity.detach().std()):.6f}  "
        f"d_img_in_ref |.| mean={float(img_in_ref.grad.abs().mean()):.6e}",
        flush=True,
    )
    print(f"OK dumped {len(out)} tensors -> {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
