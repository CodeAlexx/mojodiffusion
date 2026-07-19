#!/usr/bin/env python3
"""ai-toolkit BF16 oracle for Krea2 TextFusion LoRA.

Runs ai-toolkit's real `TextFusionTransformer` module at the configured Krea2
training dtype (`bf16`) and dumps forward output, input gradient, and all
txtfusion LoRA dA/dB tensors. This is reference behavior for the txtfusion
submodule; it does not replace full real-cache product trainer parity.
"""

from __future__ import annotations

import sys

import torch
from safetensors.torch import save_file


sys.path.insert(0, "/home/alex/ai-toolkit/extensions_built_in/diffusion_models/krea2/src")
from mmdit import TextFusionTransformer  # noqa: E402


OUT = "/home/alex/mojodiffusion/serenitymojo/models/krea2/parity/krea2_text_fusion_lora_oracle.safetensors"
DEV = "cuda"
DTYPE = torch.bfloat16

BATCH = 1
LT = 16
NLAYERS = 12
TXTDIM = 2560
HEADS = 20
HEADDIM = TXTDIM // HEADS
MULTIPLIER = 4
RANK = 2
ALPHA = 2.0
LSCALE = ALPHA / RANK

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


def _get_module(root, dotted: str):
    cur = root
    for part in dotted.split("."):
        cur = getattr(cur, part)
    return cur


class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, a: torch.Tensor, b: torch.Tensor):
        super().__init__()
        self.base = base
        self.A = torch.nn.Parameter(a)
        self.B = torch.nn.Parameter(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + LSCALE * ((x @ self.A.t()) @ self.B.t())


def _pattern(shape: tuple[int, ...], scale: float, offset: int) -> torch.Tensor:
    n = 1
    for dim in shape:
        n *= dim
    vals = torch.arange(offset, offset + n, device=DEV)
    vals = (((vals % 23) - 11).to(dtype=DTYPE) * scale).to(dtype=DTYPE)
    return vals.reshape(shape).contiguous()


def _blocks(model: TextFusionTransformer):
    return [
        ("layerwise0", model.layerwise_blocks[0]),
        ("layerwise1", model.layerwise_blocks[1]),
        ("refiner0", model.refiner_blocks[0]),
        ("refiner1", model.refiner_blocks[1]),
    ]


def _init_reference_weights(model: TextFusionTransformer) -> None:
    with torch.no_grad():
        offset = 1
        for _, block in _blocks(model):
            for module in (
                block.prenorm,
                block.postnorm,
                block.attn.qknorm.qnorm,
                block.attn.qknorm.knorm,
            ):
                module.scale.copy_(_pattern(tuple(module.scale.shape), 0.003, offset))
                offset += module.scale.numel()
            for _, path in SLOT_PATHS:
                linear = _get_module(block, path)
                assert isinstance(linear, torch.nn.Linear), (path, type(linear))
                linear.weight.copy_(_pattern(tuple(linear.weight.shape), 0.006, offset))
                offset += linear.weight.numel()
                if linear.bias is not None:
                    linear.bias.zero_()
        model.projector.weight.copy_(_pattern(tuple(model.projector.weight.shape), 0.02, offset))


def _inject_lora(model: TextFusionTransformer):
    loras: dict[tuple[str, str], tuple[torch.nn.Parameter, torch.nn.Parameter]] = {}
    offset = 100000
    for label, block in _blocks(model):
        for slot, path in SLOT_PATHS:
            base = _get_module(block, path)
            assert isinstance(base, torch.nn.Linear), (label, slot, type(base))
            a = _pattern((RANK, base.in_features), 0.01, offset)
            offset += a.numel()
            b = _pattern((base.out_features, RANK), 0.01, offset)
            offset += b.numel()
            wrapped = LoRALinear(base, a, b).to(DEV, DTYPE)
            parent = _get_module(block, path.rsplit(".", 1)[0])
            setattr(parent, path.rsplit(".", 1)[-1], wrapped)
            loras[(label, slot)] = (wrapped.A, wrapped.B)
    return loras


def _dump_block(out: dict[str, torch.Tensor], label: str, block) -> None:
    out[f"{label}.prenorm"] = block.prenorm.scale.detach().cpu()
    out[f"{label}.postnorm"] = block.postnorm.scale.detach().cpu()
    out[f"{label}.qnorm"] = block.attn.qknorm.qnorm.scale.detach().cpu()
    out[f"{label}.knorm"] = block.attn.qknorm.knorm.scale.detach().cpu()
    for slot, path in SLOT_PATHS:
        module = _get_module(block, path)
        base = module.base if isinstance(module, LoRALinear) else module
        out[f"{label}.{slot}.W"] = base.weight.detach().cpu()


def main() -> None:
    assert torch.cuda.is_available(), "krea2_text_fusion_lora_oracle.py requires CUDA."
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
    loras = _inject_lora(model)

    context = _pattern((BATCH, LT, NLAYERS, TXTDIM), 0.015, 200000).requires_grad_(True)
    d_out = _pattern((BATCH, LT, TXTDIM), 0.02, 300000)
    out = model(context, mask=None)
    out.backward(d_out)

    tensors: dict[str, torch.Tensor] = {
        "context": context.detach().cpu(),
        "d_out": d_out.detach().cpu(),
        "out": out.detach().cpu(),
        "d_context": context.grad.detach().cpu(),
        "projector.W": model.projector.weight.detach().cpu(),
    }
    for label, block in _blocks(model):
        _dump_block(tensors, label, block)
    for (label, slot), (a, b) in loras.items():
        tensors[f"{label}.{slot}.A"] = a.detach().cpu()
        tensors[f"{label}.{slot}.B"] = b.detach().cpu()
        tensors[f"{label}.{slot}.dA"] = a.grad.detach().cpu()
        tensors[f"{label}.{slot}.dB"] = b.grad.detach().cpu()

    save_file(tensors, OUT)
    gate = model.layerwise_blocks[0].mlp.gate
    mlpdim = gate.base.out_features if isinstance(gate, LoRALinear) else gate.out_features
    dtypes = sorted({str(t.dtype) for t in tensors.values()})
    print(
        "OK dumped Krea2 txtfusion LoRA BF16 oracle",
        f"tensors={len(tensors)}",
        f"LT={LT}",
        f"NLAYERS={NLAYERS}",
        f"TXTDIM={TXTDIM}",
        f"HEADS={HEADS}",
        f"MLPDIM={mlpdim}",
        f"dtypes={dtypes}",
        f"-> {OUT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
