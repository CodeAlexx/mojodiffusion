#!/usr/bin/env python3
# serenitymojo/models/krea2/parity/krea2_stack_oracle.py
#
# Torch oracle for the Krea-2-Raw SINGLE-STREAM STACK + final layer WITH LoRA on
# the 8 per-block nn.Linears, at REDUCED depth (NBLOCKS=4). This is the Phase-2
# (stack) counterpart of krea2_block_oracle.py (Phase-1, one block): it drives the
# REAL ai-toolkit module (SingleStreamDiT, mmdit.py) so the stack-forward velocity
# AND the per-block LoRA dA/dB are produced by the reference's own autograd.
#
# WHAT IS GATED (the NEW Phase-2 code): the N single-stream blocks (the verified
# Phase-1 block, composed) + the final layer (last). The embedders / text-fusion
# preamble is ALREADY parity-verified (chunks 5-7 + krea2_forward_small gate), so
# we run the reference's full forward to get the prepared single-stream input
# `combined` (= cat(txtmlp(txtfusion(context)), first(img)) ), the block modulation
# vector `tvec` (=tproj(t)), the final-layer conditioning `tmlp_out` (=tmlp(temb)),
# and the rope table `freqs`, then DUMP THOSE so the Mojo stack gate feeds byte-
# identical single-stream inputs and the comparison isolates the new composition.
#
# KEY DESIGN — NO PADDING so block SDPA == sdpa_nomask (the Phase-1 LoRA block path):
#   The reference pads `combined` to a multiple of 256 (mmdit.py:435
#   `_padlen = (-fulllen) % 256`). We pick TXTLEN + IMGLEN == 256 exactly so
#   _padlen == 0, no pad tokens are added, and `mask = _mask(all-ones)` is all-true
#   → the per-block masked SDPA is identical to no-mask attention. The Mojo Phase-1
#   block uses sdpa_nomask, so this is the faithful match (no block-0-tiled-vs-flash
#   divergence — that only exists to handle the pad region, which we avoid here).
#
# PRECISION: run the reference in F32 on GPU under the SDPA MATH backend
# (sdpa_kernel(SDPBackend.MATH)) — the flash/cuDNN kernels have no F32 path for
# these shapes ("No available kernel"), but MATH is a full-precision softmax that
# supports F32 + GQA + backward. The Mojo Phase-1 block runs F32 internally with a
# plain math-softmax sdpa, so F32+MATH is the faithful oracle (no bf16 rounding on
# the gated single-stream path; the Phase-1 block oracle used f64 manual softmax).
#
# LoRA math (ai-toolkit lora_special.py): y' = linear(x,W) + scale*((x@Aᵀ)@Bᵀ),
#   A=[rank,in] (lora_down), B=[out,rank] (lora_up), scale=alpha/rank. NONZERO B so
#   dA is non-degenerate (unlike production zero-init). Real HEADS=48/KVHEADS=12.
#
# Run (SEPARATE command, never chained with && after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/krea2/parity/krea2_stack_oracle.py

import os
import sys

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from safetensors.torch import save_file
from torch.nn.attention import sdpa_kernel, SDPBackend

# Import mmdit.py DIRECTLY (torch+einops only) to bypass the ai-toolkit package
# __init__ chain (torchao/quanto, absent here). Same path as gen_krea2_forward_small.py.
sys.path.insert(0, "/home/alex/ai-toolkit/extensions_built_in/diffusion_models/krea2/src")
import mmdit  # noqa: E402
from mmdit import SingleMMDiTConfig, SingleStreamDiT, temb  # noqa: E402


# The module's attention() hardcodes `with sdpa_kernel(SDPBackend.CUDNN_ATTENTION)`
# (mmdit.py:59), which has NO F32 kernel for these shapes ("No available kernel").
# Monkeypatch it to the MATH backend (full-precision F32 softmax, supports GQA +
# bool mask + backward) — same math as the Mojo block's sdpa_nomask. Identical to
# the original otherwise (same enable_gqa, same B H L D -> B L (H D) rearrange).
def _attention_math(q, k, v, mask=None, scale=None, gqa=False):
    with sdpa_kernel(SDPBackend.MATH):
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=scale, enable_gqa=gqa
        )
    return rearrange(x, "B H L D -> B L (H D)")


mmdit.attention = _attention_math

OUT = "/home/alex/mojodiffusion/serenitymojo/models/krea2/parity/krea2_stack_oracle.safetensors"
DEV = "cuda"
DTYPE = torch.float32
NBLOCKS = 4

# KREA2_MMDIT_CONFIG (krea2.py:55-68, "single_mmdit_large_wide"), inlined.
KREA2_MMDIT_CONFIG = dict(
    features=6144, tdim=256, txtdim=2560, heads=48, kvheads=12, multiplier=4,
    layers=28, patch=2, channels=16, txtheads=20, txtkvheads=20, txtlayers=12,
)

# LoRA recipe (non-degenerate; B NONZERO).
RANK = 8
ALPHA = 16.0
LSCALE = ALPHA / RANK  # 2.0
# The 8 LoRA-wrapped Linears per single-stream block + their (in, out) features.
# (mmdit.py SingleStreamBlock: attn.{wq,wk,wv,gate,wo}, mlp.{gate,up,down}.)
FEATURES = KREA2_MMDIT_CONFIG["features"]            # 6144
HEADS = KREA2_MMDIT_CONFIG["heads"]                  # 48
KVHEADS = KREA2_MMDIT_CONFIG["kvheads"]              # 12
HEADDIM = FEATURES // HEADS                          # 128
# SwiGLU hidden = multiple*ceil(int(2*features/3)*multiplier / multiple) (mmdit.py:186-187)
# = 128*ceil(int(2*6144/3)*4 / 128) = 16384. Derive at runtime from the built model
# (mlp_gate_out below) so this can never drift; the value here is just documentation.
MLPDIM = 16384
# slot name -> attribute path under block. (in,out) features are read from the
# actual nn.Linear shapes after the model is built (so mlp dims can't drift).
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
    """Wrap an nn.Linear so y = base(x) + LSCALE*((x@Aᵀ)@Bᵀ). A/B are leaf params
    we backprop into (the deliverable). Base weight stays frozen (no grad needed)."""

    def __init__(self, base: torch.nn.Linear, A: torch.Tensor, B: torch.Tensor):
        super().__init__()
        self.base = base
        self.A = torch.nn.Parameter(A)   # [rank, in]
        self.B = torch.nn.Parameter(B)   # [out, rank]

    def forward(self, x):
        return self.base(x) + LSCALE * ((x @ self.A.t()) @ self.B.t())


def main() -> None:
    assert torch.cuda.is_available(), "krea2_stack_oracle.py MUST run on CUDA."
    torch.manual_seed(1234)

    cfg_d = dict(KREA2_MMDIT_CONFIG)
    cfg_d["layers"] = NBLOCKS
    config = SingleMMDiTConfig(**cfg_d)
    print(
        f"config: features={config.features} heads={config.heads} "
        f"kvheads={config.kvheads} layers={config.layers} patch={config.patch} "
        f"channels={config.channels} txtdim={config.txtdim} "
        f"txtlayers={config.txtlayers} theta={config.theta} mlpdim={MLPDIM}",
        flush=True,
    )

    # ── reduced model, seeded-random scales/mod (like the forward-small oracle) ──
    model = SingleStreamDiT(config)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(".scale") or name.endswith("mod.lin") or name.endswith("modulation.lin"):
                p.copy_(torch.randn_like(p) * 0.1)
    model = model.to(DEV, dtype=DTYPE)
    model.eval()
    model.disable_gradient_checkpointing()

    # freeze EVERYTHING (we only train the LoRA A/B we inject below).
    for p in model.parameters():
        p.requires_grad_(False)

    # ── inject LoRA on the 8 Linears of each of the NBLOCKS single-stream blocks ──
    # (in,out) features are READ from each nn.Linear (in_features/out_features) so the
    # mlp hidden dim is exactly the model's, never a hardcoded guess.
    g = torch.Generator(device="cpu").manual_seed(7)
    loras = {}  # (block, slot) -> (A_param, B_param)
    mlp_gate_out = None
    for bi in range(NBLOCKS):
        block = model.blocks[bi]
        for (slot, path) in SLOT_PATHS:
            base = _get_module(block, path)
            assert isinstance(base, torch.nn.Linear), (bi, slot, type(base))
            in_f, out_f = base.in_features, base.out_features
            if slot == "mlp_gate":
                mlp_gate_out = out_f
            A = (torch.randn(RANK, in_f, generator=g) * 0.02).to(DEV, DTYPE)
            B = (torch.randn(out_f, RANK, generator=g) * 0.02).to(DEV, DTYPE)
            wrapped = LoRALinear(base, A, B).to(DEV, DTYPE)
            # replace the attribute (e.g. block.attn.wq) with the wrapper.
            parent = _get_module(block, path.rsplit(".", 1)[0]) if "." in path else block
            setattr(parent, path.rsplit(".", 1)[-1], wrapped)
            loras[(bi, slot)] = (wrapped.A, wrapped.B)
    assert mlp_gate_out == MLPDIM, f"mlp hidden {mlp_gate_out} != expected {MLPDIM}"

    # ── fixed seeded inputs: TXTLEN + IMGLEN == 256 EXACTLY → no pad ─────────────
    # GH=14, GW=15 → IMGLEN=210; TXTLEN=46 → L_FULL=256 (mult of 256, _padlen=0).
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
    assert TXTLEN > 0 and (TXTLEN + IMGLEN) % 256 == 0, (TXTLEN, IMGLEN)

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
    L_FULL = TXTLEN + IMGLEN  # 256
    assert pos.shape == (B, L_FULL, 3), pos.shape

    mask = torch.ones(B, L_FULL, device=DEV, dtype=torch.bool)
    t = torch.rand(B, device=DEV, dtype=torch.float32)

    print(
        f"inputs: img={tuple(img.shape)} context={tuple(context.shape)} "
        f"t={tuple(t.shape)} pos={tuple(pos.shape)} "
        f"(GH={GH} GW={GW} IMGLEN={IMGLEN} TXTLEN={TXTLEN} L_FULL={L_FULL} -> NO pad)",
        flush=True,
    )

    # ── capture the prepared single-stream inputs by re-running the preamble ────
    # (the same ops the reference forward does before the block loop). We run with
    # grad ENABLED through the blocks+last so the LoRA A/B accumulate grad. ALL
    # SDPA under the MATH backend (F32 full-precision softmax; the only F32 path).
    with sdpa_kernel(SDPBackend.MATH):
        img_e = model.first(img)
        t_full = model.tmlp(temb(t, config.tdim, device=img.device, dtype=img.dtype))  # [B,1,F] tmlp out
        tvec = model.tproj(t_full)                                                     # [B,1,6F] block mod
        txtmask = mmdit_mask(mask[:, : context.shape[1]])
        ctx_fused = model.txtfusion(context, mask=txtmask)
        ctx_proj = model.txtmlp(ctx_fused)
        txtlen, imglen = ctx_proj.shape[1], img_e.shape[1]
        combined = torch.cat((ctx_proj, img_e), dim=1)                                 # [B,256,F]
        assert (-combined.shape[1]) % 256 == 0, "design invariant: combined len mult of 256"
        full_mask = mmdit_mask(mask)                                                   # all-true
        freqs = model.posemb(pos)                                                      # rope table

        # detach the prepared inputs into fresh leaves the GATE will feed verbatim; the
        # block+last graph re-derives from them so the LoRA grads are well-defined.
        combined_in = combined.detach().clone().requires_grad_(True)
        x = combined_in
        for bi in range(NBLOCKS):
            x = model.blocks[bi](x, tvec.detach(), freqs, full_mask)
        final = model.last(x, t_full.detach())
        velocity = final[:, txtlen : txtlen + imglen, :]                              # [B,IMGLEN,64]
        assert velocity.shape == (B, IMGLEN, in_dim), velocity.shape

        # ── backward via a fixed upstream grad on velocity ──────────────────────
        d_velocity = torch.sin(
            torch.arange(velocity.numel(), device=DEV, dtype=DTYPE) * 0.0011 + 0.07
        ).reshape_as(velocity) * 0.05
        (velocity * d_velocity).sum().backward()

    # ── dump everything the Mojo gate reconstructs ──────────────────────────────
    out = {}

    def put(name, tensor):
        out[name] = tensor.detach().to(torch.float32).cpu().contiguous()

    # prepared single-stream INPUTS (fed verbatim by the gate)
    put("combined", combined_in)                 # [1,256,6144]  block-stack input
    put("tvec", tvec)                            # [1,1,6*6144]  block modulation vec
    put("tmlp_out", t_full)                      # [1,1,6144]    final-layer tvec
    put("pos", pos)                              # [1,256,3]     for the Mojo rope build
    put("d_velocity", d_velocity)                # [1,IMGLEN,64] upstream grad
    put("velocity", velocity)                    # [1,IMGLEN,64] reference output

    # per-block frozen weights (8 Linears + 2 norms + 2 qknorm + mod.lin)
    for bi in range(NBLOCKS):
        block = model.blocks[bi]
        for (slot, path) in SLOT_PATHS:
            wmod = _get_module(block, path)       # LoRALinear; .base is the nn.Linear
            put(f"blk{bi}.{slot}.W", wmod.base.weight)
        put(f"blk{bi}.prenorm", block.prenorm.scale)
        put(f"blk{bi}.postnorm", block.postnorm.scale)
        put(f"blk{bi}.qnorm", block.attn.qknorm.qnorm.scale)
        put(f"blk{bi}.knorm", block.attn.qknorm.knorm.scale)
        put(f"blk{bi}.mod_lin", block.mod.lin)
        # per-block LoRA A/B (inputs) + grads (the deliverable)
        for (slot, _path) in SLOT_PATHS:
            A, Bp = loras[(bi, slot)]
            put(f"blk{bi}.{slot}.A", A)
            put(f"blk{bi}.{slot}.B", Bp)
            put(f"kref.blk{bi}.{slot}.dA", A.grad)
            put(f"kref.blk{bi}.{slot}.dB", Bp.grad)

    # final-layer (last) frozen weights
    put("last.norm", model.last.norm.scale)
    put("last.mod_lin", model.last.modulation.lin)   # [2, features]
    put("last.lin_w", model.last.linear.weight)      # [64, 6144]
    put("last.lin_b", model.last.linear.bias)        # [64]

    # meta
    out["meta_txtlen"] = torch.tensor([TXTLEN], dtype=torch.int32)
    out["meta_imglen"] = torch.tensor([IMGLEN], dtype=torch.int32)
    out["meta_lfull"] = torch.tensor([L_FULL], dtype=torch.int32)
    out["meta_nblocks"] = torch.tensor([NBLOCKS], dtype=torch.int32)
    out["meta_mlpdim"] = torch.tensor([MLPDIM], dtype=torch.int32)

    save_file(out, OUT)
    print(f"forward velocity mean={float(velocity.mean()):.6f} std={float(velocity.std()):.6f}", flush=True)
    print(f"OK dumped {len(out)} tensors -> {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)", flush=True)


def mmdit_mask(mask: torch.Tensor) -> torch.Tensor:
    """_mask (mmdit.py:66-68): (B,L) key-padding -> (B,1,L,L) attn mask."""
    return mask.unsqueeze(1).unsqueeze(2) * mask.unsqueeze(1).unsqueeze(3)


if __name__ == "__main__":
    main()
