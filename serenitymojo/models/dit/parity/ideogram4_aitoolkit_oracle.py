#!/usr/bin/env python3
# DEV-ONLY parity oracle for the serenitymojo Ideogram-4 LoRA BLOCK trainer.
#
# Source of truth: ai-toolkit's production Ideogram-4 implementation
#   /home/alex/ai-toolkit/extensions_built_in/diffusion_models/ideogram4/src/transformer.py
# This REPLACES the earlier (invalid) ideogram4-ref forward oracle. It captures the
# per-block (layer 0) FORWARD and BACKWARD that the mojo
#   serenitymojo/models/ideogram4/block.mojo : ideogram4_block_lora_forward / _backward
# (and the autograd_v2 graph adapter) must match.
#
# Dtype: bf16 — the dtype the ai-toolkit production path actually runs the block in
# (_dequantize_fp8_state_dict folds the fp8 weights to bf16; the model runs bf16).
# This is NOT "lowering to match a bf16 port" — bf16 IS production here.
#
# Inputs: NON-DEGENERATE (randn, fixed seed). GH=GW=16, NTEXT=4 -> L=260, t=0.7.
# All tokens share segment_id=1 (single packed sample) so the native SDPA block
# mask is all-True == the mojo block's sdpa_nomask. Native (SDPA) backend only;
# flash_attn is not installed and the production default backend is "native".
#
# LoRA: rank 16 on each of the 6 block linears (qkv, o, w1, w2, w3, adaln), with
# the mojo convention  A=[rank,in] (lora_down), B=[out,rank] (lora_up),
# out = base(x) + (x @ A^T @ B^T) * (alpha/rank). Both A and B are seeded NONZERO
# so dA and dB are both meaningful references (PEFT's B=0 init would zero dA).
#
# Run:
#   /home/alex/serenityflow-v2/.venv/bin/python \
#     serenitymojo/models/dit/parity/ideogram4_aitoolkit_oracle.py
#
# Output: serenitymojo/models/dit/parity/ideogram4_aitoolkit_block0.safetensors

import os
import sys
import gc

import torch
from safetensors import safe_open
from safetensors.torch import save_file

AITK = "/home/alex/ai-toolkit/extensions_built_in/diffusion_models/ideogram4/src"
sys.path.insert(0, AITK)

# Import ai-toolkit's PRODUCTION transformer module (authoritative forward).
import transformer as i4  # noqa: E402

ROOT = "/home/alex/.serenity/models/ideogram-4-fp8/transformer"
CKPT = os.path.join(ROOT, "diffusion_pytorch_model.safetensors")
OUT_DIR = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity"
OUT = os.path.join(OUT_DIR, "ideogram4_aitoolkit_block0.safetensors")

DEV = torch.device("cuda")
DT = torch.bfloat16  # production dtype (fp8 -> bf16)

# Parity input geometry (mirrors the prior oracle so the gate inputs line up).
GH = GW = 16
NIMG = GH * GW          # 256
NTEXT = 4
L = NTEXT + NIMG        # 260
TVAL = 0.7

LORA_RANK = 16
LORA_ALPHA = 16.0       # alpha/rank = 1.0 (dyadic -> exact bf16 scale)
SEED = 1234

LLM_TOKEN_INDICATOR = i4.LLM_TOKEN_INDICATOR          # 3
OUTPUT_IMAGE_INDICATOR = i4.OUTPUT_IMAGE_INDICATOR    # 2
IMAGE_POSITION_OFFSET = i4.IMAGE_POSITION_OFFSET      # 65536


def _dequant_fp8(w_fp8, scale):
    # Production fold (_dequantize_fp8_state_dict): w_fp8.float() * scale[:, None] -> bf16.
    return (w_fp8.to(torch.float32) * scale.to(torch.float32).unsqueeze(1)).to(DT)


def load_block0_weights():
    """Load layer-0 weights from the fp8 checkpoint, dequantized to bf16, exactly
    as the ai-toolkit production loader does (fp8 .weight + sibling .weight_scale)."""
    cfg = i4.Ideogram4Config()
    block = i4.Ideogram4TransformerBlock(
        hidden_size=cfg.emb_dim,
        intermediate_size=cfg.intermediate_size,
        num_heads=cfg.num_heads,
        norm_eps=cfg.norm_eps,
        adanln_dim=cfg.adanln_dim,
    )
    sd = {}
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        keys = [k for k in f.keys() if k.startswith("layers.0.")]
        raw = {k: f.get_tensor(k) for k in keys}

    # Fold fp8 weights, cast everything else to bf16, strip the "layers.0." prefix.
    P = "layers.0."
    for k, v in raw.items():
        if k.endswith(".weight_scale"):
            continue
        scale_key = k + "_scale"
        local = k[len(P):]
        if k.endswith(".weight") and scale_key in raw:
            sd[local] = _dequant_fp8(v, raw[scale_key])
        elif v.is_floating_point():
            sd[local] = v.to(DT)
        else:
            sd[local] = v

    missing, unexpected = block.load_state_dict(sd, strict=False, assign=True)
    # The block has no non-persistent buffers; load must be exact.
    assert not unexpected, f"unexpected keys: {unexpected}"
    # attention_backend default attr is set in __init__; nothing else missing.
    assert all("attention_backend" not in m for m in missing), f"missing: {missing}"
    block.to(DEV)
    block.attention.attention_backend = "native"
    return cfg, block


def build_inputs(cfg):
    """NON-DEGENERATE packed-sequence block inputs at the mojo parity shapes.

    Returns the exact tensors the block forward consumes:
      x          : (1, L, emb_dim) bf16  -- block input hidden state h
      adaln_input: (1, 1,  adaln)  bf16  -- silu(adaln_proj(t_embedding(t)))
      cos, sin   : (1, L, head_dim) bf16 -- from the transformer MRoPE
      attn_mask  : (1, 1, L, L)    bool  -- block-diagonal (all True, one segment)
    plus the raw position_ids/segment_ids for the report.
    """
    g = torch.Generator(device=DEV).manual_seed(SEED)

    # Block input hidden state (post input_proj + llm cond + indicator embed in the
    # full model). For a per-block parity fixture this is a non-degenerate randn at
    # the block input dim, matching how the mojo block-graph gate seeds x_in.
    x = torch.randn(1, L, cfg.emb_dim, device=DEV, dtype=torch.float32, generator=g).to(DT)

    # position_ids (t,h,w): text rows 0..NTEXT-1 broadcast on all 3 axes; image rows
    # (0, h, w) + IMAGE_POSITION_OFFSET. Exactly the prior oracle's packing.
    pos = torch.zeros(1, L, 3, dtype=torch.long, device=DEV)
    tp = torch.arange(NTEXT, device=DEV)
    pos[0, :NTEXT] = torch.stack([tp, tp, tp], dim=1)
    hi = torch.arange(GH, device=DEV).view(-1, 1).expand(GH, GW).reshape(-1)
    wi = torch.arange(GW, device=DEV).view(1, -1).expand(GH, GW).reshape(-1)
    ti = torch.zeros_like(hi)
    pos[0, NTEXT:] = torch.stack([ti, hi, wi], dim=1) + IMAGE_POSITION_OFFSET

    # segment_ids: one packed sample -> all 1 -> block-diagonal mask is all True.
    seg = torch.ones(1, L, dtype=torch.long, device=DEV)

    # t -> adaln_input via the transformer's OWN t_embedding + adaln_proj (so the
    # block's adaLN input is produced by the production code path, not by hand).
    cfg2 = cfg
    t_embedding = i4.Ideogram4EmbedScalar(cfg2.emb_dim, input_range=(0.0, 1.0)).to(DEV, DT)
    adaln_proj = torch.nn.Linear(cfg2.emb_dim, cfg2.adanln_dim, bias=True).to(DEV, DT)
    # Deterministic, non-degenerate params for the t-path (the parity fixture only
    # needs a fixed adaln_input vector; its exact provenance is captured + saved).
    with torch.no_grad():
        gp = torch.Generator(device=DEV).manual_seed(SEED + 7)
        for p in list(t_embedding.parameters()) + list(adaln_proj.parameters()):
            p.copy_(torch.randn(p.shape, device=DEV, dtype=torch.float32, generator=gp).to(DT) * 0.02)
    t = torch.full((1,), TVAL, device=DEV, dtype=DT)
    with torch.no_grad():
        t_cond = t_embedding(t)            # (1, emb_dim)
        t_cond = t_cond.unsqueeze(1)       # (1, 1, emb_dim)
        adaln_input = torch.nn.functional.silu(adaln_proj(t_cond))  # (1,1,adaln)
    adaln_input = adaln_input.to(DT)

    # cos/sin from the transformer's MRoPE (head_dim = emb_dim/num_heads = 256).
    head_dim = cfg.emb_dim // cfg.num_heads
    rope = i4.Ideogram4MRoPE(head_dim=head_dim, base=cfg.rope_theta,
                             mrope_section=cfg.mrope_section).to(DEV)
    with torch.no_grad():
        cos, sin = rope(pos)               # (1, L, head_dim) float32
    cos = cos.to(DT)
    sin = sin.to(DT)

    # native SDPA block-diagonal mask (all-True for one segment).
    attn_mask = (seg.unsqueeze(2) == seg.unsqueeze(1)).unsqueeze(1)
    return x, adaln_input, cos, sin, attn_mask, pos, seg


class LoraLinear(torch.nn.Module):
    """Wrap a frozen base nn.Linear with a mojo-convention LoRA:
       A=[rank,in] (down), B=[out,rank] (up),  y = base(x) + (x @ A^T @ B^T)*scale.
    Both A and B are nonzero so dA and dB are meaningful. base weight is frozen."""

    def __init__(self, base: torch.nn.Linear, rank: int, alpha: float, seed: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_f = base.in_features
        out_f = base.out_features
        self.scale = alpha / rank
        g = torch.Generator(device="cpu").manual_seed(seed)
        # std=1/sqrt(3*in) matches the mojo make_lora_adapter A init scale; B uses a
        # small but NONZERO std so dA != 0.
        std_a = 1.0 / (3.0 * in_f) ** 0.5
        std_b = 0.02
        A = (torch.randn(rank, in_f, generator=g) * std_a).to(DEV, DT)
        B = (torch.randn(out_f, rank, generator=g) * std_b).to(DEV, DT)
        self.A = torch.nn.Parameter(A.clone().requires_grad_(True))
        self.B = torch.nn.Parameter(B.clone().requires_grad_(True))

    def forward(self, x):
        base = self.base(x)
        # x @ A^T -> (..., rank);  @ B^T -> (..., out)
        down = torch.nn.functional.linear(x, self.A)   # uses A as [rank,in]
        up = torch.nn.functional.linear(down, self.B)  # uses B as [out,rank]
        return base + up * self.scale


# The 6 LoRA slots, in the mojo ideogram4LoraTargets / I4_SLOT_* order.
LORA_SLOTS = [
    ("qkv", "attention.qkv"),
    ("o", "attention.o"),
    ("w1", "feed_forward.w1"),
    ("w2", "feed_forward.w2"),
    ("w3", "feed_forward.w3"),
    ("adaln", "adaln_modulation"),
]


def attach_loras(block):
    """Replace each of the 6 target nn.Linear submodules with a LoraLinear and
    return the adapters in slot order."""
    adapters = {}
    seed = SEED + 100
    for slot, path in LORA_SLOTS:
        parent = block
        parts = path.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        leaf = parts[-1]
        base = getattr(parent, leaf)
        assert isinstance(base, torch.nn.Linear), f"{path} is not a Linear"
        lin = LoraLinear(base, LORA_RANK, LORA_ALPHA, seed)
        setattr(parent, leaf, lin)
        adapters[slot] = lin
        seed += 1
    return adapters


def main():
    torch.manual_seed(SEED)
    cfg, block = load_block0_weights()
    print(f"[oracle] layer-0 block loaded ({DT}); VRAM {torch.cuda.memory_allocated()/1e9:.2f}GB")

    adapters = attach_loras(block)
    print(f"[oracle] attached {len(adapters)} LoRA adapters (rank {LORA_RANK}, "
          f"alpha {LORA_ALPHA}, scale {LORA_ALPHA/LORA_RANK})")

    x, adaln_input, cos, sin, attn_mask, pos, seg = build_inputs(cfg)
    print(f"[oracle] inputs: x{tuple(x.shape)} adaln{tuple(adaln_input.shape)} "
          f"cos{tuple(cos.shape)} mask{tuple(attn_mask.shape)} L={L}")

    # --- FORWARD (production block) ---
    x_in = x.detach().clone().requires_grad_(True)
    adaln_in = adaln_input.detach().clone().requires_grad_(True)
    block.train()  # enable grad flow (no dropout in this block)
    out = block(x_in, attn_mask, cos, sin, adaln_in)  # (1, L, emb_dim) bf16
    print(f"[oracle] forward out {tuple(out.shape)} {out.dtype} "
          f"mean {out.float().mean():.5f} std {out.float().std():.5f}")

    # --- BACKWARD (torch autograd, fixed seeded upstream grad) ---
    g = torch.Generator(device=DEV).manual_seed(SEED + 999)
    d_out = torch.randn(out.shape, device=DEV, dtype=torch.float32, generator=g).to(DT)
    out.backward(d_out)

    fx = {}
    # forward fixtures
    fx["fwd.block0_in_x"] = x_in.detach().float().cpu()
    fx["fwd.adaln_input"] = adaln_in.detach().float().cpu()
    fx["fwd.cos"] = cos.float().cpu()
    fx["fwd.sin"] = sin.float().cpu()
    fx["fwd.attn_mask"] = attn_mask.to(torch.int32).cpu()
    fx["fwd.position_ids"] = pos.to(torch.int32).cpu()
    fx["fwd.segment_ids"] = seg.to(torch.int32).cpu()
    fx["fwd.block0_out"] = out.detach().float().cpu()
    # backward fixtures
    fx["bwd.d_out"] = d_out.float().cpu()
    fx["bwd.d_x"] = x_in.grad.detach().float().cpu()
    fx["bwd.d_adaln_input"] = adaln_in.grad.detach().float().cpu()

    # per-weight grads (frozen base weights have no grad; norms/adaln-bias do)
    for name, p in block.named_parameters():
        if p.grad is not None:
            fx[f"bwd.grad.{name}"] = p.grad.detach().float().cpu()

    # LoRA factor grads + the A/B values used (so the mojo gate loads the SAME LoRA)
    for slot, _ in LORA_SLOTS:
        ad = adapters[slot]
        fx[f"lora.{slot}.A"] = ad.A.detach().float().cpu()
        fx[f"lora.{slot}.B"] = ad.B.detach().float().cpu()
        fx[f"bwd.lora.{slot}.dA"] = ad.A.grad.detach().float().cpu()
        fx[f"bwd.lora.{slot}.dB"] = ad.B.grad.detach().float().cpu()

    os.makedirs(OUT_DIR, exist_ok=True)
    save_file(fx, OUT)
    print(f"[oracle] saved {len(fx)} tensors -> {OUT}")
    # quick sanity dump
    for slot, _ in LORA_SLOTS:
        dA = fx[f"bwd.lora.{slot}.dA"]
        dB = fx[f"bwd.lora.{slot}.dB"]
        print(f"  lora.{slot}: A{tuple(fx[f'lora.{slot}.A'].shape)} "
              f"dA|std={dA.std():.3e} nonzero={int((dA != 0).sum())}  "
              f"dB|std={dB.std():.3e} nonzero={int((dB != 0).sum())}")
    print(f"  d_x std {fx['bwd.d_x'].std():.5f}  "
          f"d_adaln_input std {fx['bwd.d_adaln_input'].std():.5f}")
    ngrad = sum(1 for k in fx if k.startswith("bwd.grad."))
    print(f"  per-weight base/norm grads captured: {ngrad}")
    del block
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
