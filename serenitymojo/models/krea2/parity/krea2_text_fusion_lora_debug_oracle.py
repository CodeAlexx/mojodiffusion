#!/usr/bin/env python3
"""Debug-only Krea2 TextFusion BF16 oracle with block intermediates.

This does not replace krea2_text_fusion_lora_oracle.py. It dumps selected
TextFusion block outputs and input gradients to isolate replay drift while
preserving the production BF16 oracle artifact unchanged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import save_file
from torch.nn.attention import SDPBackend, sdpa_kernel

from krea2_text_fusion_lora_oracle import (
    BATCH,
    DEV,
    DTYPE,
    HEADS,
    LT,
    MULTIPLIER,
    NLAYERS,
    TXTDIM,
    TextFusionTransformer,
    _blocks,
    _init_reference_weights,
    _inject_lora,
    _pattern,
    LSCALE,
    LoRALinear,
)


OUT = "/home/alex/mojodiffusion/serenitymojo/models/krea2/parity/krea2_text_fusion_lora_debug_oracle.safetensors"


def _dump_block_forward(saved: dict[str, torch.Tensor], label: str, block, x: torch.Tensor) -> torch.Tensor:
    """Replay one TextFusionBlock and dump Mojo-aligned forward intermediates.

    Shapes are stored in Mojo order: q/k/v are BSHD, attention is flattened BLD.
    """
    saved[f"{label}.in"] = x.detach()
    xn = block.prenorm(x)
    q = block.attn.wq(xn)
    k = block.attn.wk(xn)
    v_lin = block.attn.wv(xn)
    gate_pre = block.attn.gate(xn)

    q_bhld = rearrange(q, "B L (H D) -> B H L D", H=HEADS)
    k_bhld = rearrange(k, "B L (H D) -> B H L D", H=HEADS)
    v_bhld = rearrange(v_lin, "B L (H D) -> B H L D", H=HEADS)
    q_norm_bhld, k_norm_bhld, v_bhld = block.attn.qknorm(q_bhld, k_bhld, v_bhld)
    scale = 1.0 / (q_norm_bhld.shape[-1] ** 0.5)
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        att_bhld = F.scaled_dot_product_attention(
            q_norm_bhld, k_norm_bhld, v_bhld, attn_mask=None, scale=scale
        )
    attn_flat = rearrange(att_bhld, "B H L D -> B L (H D)")
    sg = F.sigmoid(gate_pre)
    gated = attn_flat * sg
    a = block.attn.wo(gated)
    x1 = x + a

    xn2 = block.postnorm(x1)
    mlp_gate = block.mlp.gate(xn2)
    mlp_up = block.mlp.up(xn2)
    sw = F.silu(mlp_gate) * mlp_up
    m = block.mlp.down(sw)
    out = x1 + m

    saved[f"{label}.xn"] = xn.detach()
    saved[f"{label}.q_pre"] = rearrange(q_bhld, "B H L D -> B L H D").detach()
    saved[f"{label}.k_pre"] = rearrange(k_bhld, "B H L D -> B L H D").detach()
    saved[f"{label}.v"] = rearrange(v_bhld, "B H L D -> B L H D").detach()
    saved[f"{label}.q_norm"] = rearrange(q_norm_bhld, "B H L D -> B L H D").detach()
    saved[f"{label}.k_norm"] = rearrange(k_norm_bhld, "B H L D -> B L H D").detach()
    saved[f"{label}.attn_flat"] = attn_flat.detach()
    saved[f"{label}.gate_pre"] = gate_pre.detach()
    saved[f"{label}.sg"] = sg.detach()
    saved[f"{label}.gated"] = gated.detach()
    saved[f"{label}.a"] = a.detach()
    saved[f"{label}.x1"] = x1.detach()
    saved[f"{label}.xn2"] = xn2.detach()
    saved[f"{label}.mlp_gate"] = mlp_gate.detach()
    saved[f"{label}.mlp_up"] = mlp_up.detach()
    saved[f"{label}.sw"] = sw.detach()
    saved[f"{label}.m"] = m.detach()
    saved[f"{label}.out"] = out.detach()
    return out


def _dump_lora_linear(saved: dict[str, torch.Tensor], label: str, slot: str, module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(module, LoRALinear):
        base = module.base(x)
        t = x @ module.A.t()
        mid = t @ module.B.t()
        delta = LSCALE * mid
        out = base + delta
        saved[f"{label}.{slot}.base"] = base.detach()
        saved[f"{label}.{slot}.lora_t"] = t.detach()
        saved[f"{label}.{slot}.delta"] = delta.detach()
        saved[f"{label}.{slot}.out"] = out.detach()
        return out
    out = module(x)
    saved[f"{label}.{slot}.base"] = out.detach()
    saved[f"{label}.{slot}.out"] = out.detach()
    return out


def main() -> None:
    assert torch.cuda.is_available(), "krea2_text_fusion_lora_debug_oracle.py requires CUDA."
    torch.manual_seed(20260630)

    model = TextFusionTransformer(
        num_txt_layers=NLAYERS,
        txt_dim=TXTDIM,
        heads=HEADS,
        multiplier=MULTIPLIER,
        bias=False,
        kvheads=HEADS,
    ).to(DEV, DTYPE)
    model.eval()
    _init_reference_weights(model)
    for p in model.parameters():
        p.requires_grad_(False)
    _inject_lora(model)

    saved: dict[str, torch.Tensor] = {}

    def make_hook(label: str):
        def hook(_module, args, output):
            x = args[0]
            x.retain_grad()
            output.retain_grad()
            saved[f"{label}.in"] = x
            saved[f"{label}.out"] = output

        return hook

    for label, block in _blocks(model):
        block.register_forward_hook(make_hook(label))

    context = _pattern((BATCH, LT, NLAYERS, TXTDIM), 0.015, 200000).requires_grad_(True)
    d_out = _pattern((BATCH, LT, TXTDIM), 0.02, 300000)
    out = model(context, mask=None)
    out.backward(d_out)

    tensors: dict[str, torch.Tensor] = {
        "out": out.detach().cpu(),
        "d_context": context.grad.detach().cpu(),
    }
    for label, tensor in saved.items():
        tensors[label] = tensor.detach().cpu()
        if tensor.grad is not None:
            tensors[f"d_{label}"] = tensor.grad.detach().cpu()

    # Forward-internal replay for first-mismatch localization.  Run separately
    # under no_grad so the backward debug artifact above remains unchanged.
    forward_saved: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        x = context.detach().reshape(BATCH * LT, NLAYERS, TXTDIM)
        x = _dump_block_forward(forward_saved, "layerwise0", model.layerwise_blocks[0], x.contiguous())
        x = _dump_block_forward(forward_saved, "layerwise1", model.layerwise_blocks[1], x.contiguous())
        x = rearrange(x, "(b l) n d -> b l d n", b=BATCH, l=LT)
        x = model.projector(x.reshape(BATCH * LT, TXTDIM, NLAYERS))
        x = x.reshape(BATCH, LT, TXTDIM)
        x = _dump_block_forward(forward_saved, "refiner0", model.refiner_blocks[0], x)
        _ = _dump_block_forward(forward_saved, "refiner1", model.refiner_blocks[1], x)
        for label, block, bx in (
            ("refiner0", model.refiner_blocks[0], forward_saved["refiner0.xn"]),
            ("refiner1", model.refiner_blocks[1], forward_saved["refiner1.xn"]),
        ):
            _dump_lora_linear(forward_saved, label, "wq", block.attn.wq, bx)
            _dump_lora_linear(forward_saved, label, "wk", block.attn.wk, bx)
            _dump_lora_linear(forward_saved, label, "wv", block.attn.wv, bx)
            _dump_lora_linear(forward_saved, label, "gate", block.attn.gate, bx)
    for label, tensor in forward_saved.items():
        tensors[label] = tensor.cpu().contiguous()

    save_file(tensors, OUT)
    print(
        "OK dumped Krea2 txtfusion debug BF16 oracle",
        f"tensors={len(tensors)}",
        f"dtypes={sorted({str(t.dtype) for t in tensors.values()})}",
        f"-> {OUT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
