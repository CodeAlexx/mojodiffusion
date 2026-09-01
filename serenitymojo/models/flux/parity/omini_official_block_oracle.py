#!/usr/bin/env python3
"""Pinned official OminiControl FLUX.1 block oracle.

This is deliberately a development-only Python oracle.  It imports the block
functions from the pinned upstream OminiControl checkout instead of copying
their math.  The fixture uses the real local FLUX.1-dev block-0 weights and the
released OminiControl fill adapter, then exercises the exact three branches
used by training:

    text (no adapter), noisy target (no adapter), condition (fill adapter)

It dumps x-embedder outputs, one dual-stream block, one single-stream block,
input gradients, and every LoRA gradient involved in those blocks.  A Mojo
gate must replay this fixture before an OminiControl training path can claim
parity.

The upstream source and released adapter are external evidence.  Do not copy
them into the Mojo runtime or make this script a production dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
from peft import LoraConfig, inject_adapter_in_model
from safetensors import safe_open
from safetensors.torch import save_file


PINNED_OMINI_COMMIT = "65d929e390a3f785b11e71344de082bb58a2c527"
PINNED_FILL_SHA256 = "b139f2c96c67f956f9070607742a78cde6ccfb2091ce1dd37db160eb32869883"

DEFAULT_SOURCE = Path("/tmp/ominicontrol-official-oracle")
DEFAULT_ADAPTER = Path("/tmp/ominicontrol-official-weights/experimental/fill.safetensors")
DEFAULT_BASE = Path("/home/alex/.serenity/models/checkpoints/flux1-dev.safetensors")
DEFAULT_OUTPUT = Path("/tmp/omini_official_block0_fixture.safetensors")

D = 3072
HEADS = 24
HEAD_DIM = 128
MLP = 4 * D
RANK = 4
ADAPTER = "fill"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_provenance(source: Path, adapter: Path) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != PINNED_OMINI_COMMIT:
        raise SystemExit(
            f"wrong OminiControl source revision: {commit}; expected {PINNED_OMINI_COMMIT}"
        )
    adapter_sha = sha256_file(adapter)
    if adapter_sha != PINNED_FILL_SHA256:
        raise SystemExit(
            f"wrong fill adapter sha256: {adapter_sha}; expected {PINNED_FILL_SHA256}"
        )


def split_rows(tensor: torch.Tensor, part: int) -> tuple[torch.Tensor, ...]:
    return tuple(tensor[i * part : (i + 1) * part] for i in range(tensor.shape[0] // part))


def load_double_block(base_path: Path):
    from diffusers.models.transformers.transformer_flux import FluxTransformerBlock

    with safe_open(base_path, framework="pt", device="cpu") as src:
        iq, ik, iv = split_rows(src.get_tensor("double_blocks.0.img_attn.qkv.weight"), D)
        ibq, ibk, ibv = split_rows(src.get_tensor("double_blocks.0.img_attn.qkv.bias"), D)
        tq, tk, tv = split_rows(src.get_tensor("double_blocks.0.txt_attn.qkv.weight"), D)
        tbq, tbk, tbv = split_rows(src.get_tensor("double_blocks.0.txt_attn.qkv.bias"), D)
        state = {
            "norm1.linear.weight": src.get_tensor("double_blocks.0.img_mod.lin.weight"),
            "norm1.linear.bias": src.get_tensor("double_blocks.0.img_mod.lin.bias"),
            "norm1_context.linear.weight": src.get_tensor("double_blocks.0.txt_mod.lin.weight"),
            "norm1_context.linear.bias": src.get_tensor("double_blocks.0.txt_mod.lin.bias"),
            "attn.norm_q.weight": src.get_tensor("double_blocks.0.img_attn.norm.query_norm.scale"),
            "attn.norm_k.weight": src.get_tensor("double_blocks.0.img_attn.norm.key_norm.scale"),
            "attn.to_q.weight": iq,
            "attn.to_q.bias": ibq,
            "attn.to_k.weight": ik,
            "attn.to_k.bias": ibk,
            "attn.to_v.weight": iv,
            "attn.to_v.bias": ibv,
            "attn.to_out.0.weight": src.get_tensor("double_blocks.0.img_attn.proj.weight"),
            "attn.to_out.0.bias": src.get_tensor("double_blocks.0.img_attn.proj.bias"),
            "attn.norm_added_q.weight": src.get_tensor("double_blocks.0.txt_attn.norm.query_norm.scale"),
            "attn.norm_added_k.weight": src.get_tensor("double_blocks.0.txt_attn.norm.key_norm.scale"),
            "attn.add_q_proj.weight": tq,
            "attn.add_q_proj.bias": tbq,
            "attn.add_k_proj.weight": tk,
            "attn.add_k_proj.bias": tbk,
            "attn.add_v_proj.weight": tv,
            "attn.add_v_proj.bias": tbv,
            "attn.to_add_out.weight": src.get_tensor("double_blocks.0.txt_attn.proj.weight"),
            "attn.to_add_out.bias": src.get_tensor("double_blocks.0.txt_attn.proj.bias"),
            "ff.net.0.proj.weight": src.get_tensor("double_blocks.0.img_mlp.0.weight"),
            "ff.net.0.proj.bias": src.get_tensor("double_blocks.0.img_mlp.0.bias"),
            "ff.net.2.weight": src.get_tensor("double_blocks.0.img_mlp.2.weight"),
            "ff.net.2.bias": src.get_tensor("double_blocks.0.img_mlp.2.bias"),
            "ff_context.net.0.proj.weight": src.get_tensor("double_blocks.0.txt_mlp.0.weight"),
            "ff_context.net.0.proj.bias": src.get_tensor("double_blocks.0.txt_mlp.0.bias"),
            "ff_context.net.2.weight": src.get_tensor("double_blocks.0.txt_mlp.2.weight"),
            "ff_context.net.2.bias": src.get_tensor("double_blocks.0.txt_mlp.2.bias"),
        }

    with torch.device("meta"):
        block = FluxTransformerBlock(D, HEADS, HEAD_DIM)
    block.load_state_dict(state, strict=True, assign=True)
    config = LoraConfig(
        r=RANK,
        lora_alpha=RANK,
        lora_dropout=0.0,
        init_lora_weights="gaussian",
        target_modules=[
            "norm1.linear",
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "ff.net.2",
        ],
    )
    return inject_adapter_in_model(config, block, adapter_name=ADAPTER)


def load_single_block(base_path: Path):
    from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock

    with safe_open(base_path, framework="pt", device="cpu") as src:
        linear1_w = src.get_tensor("single_blocks.0.linear1.weight")
        linear1_b = src.get_tensor("single_blocks.0.linear1.bias")
        q, k, v = split_rows(linear1_w[: 3 * D], D)
        bq, bk, bv = split_rows(linear1_b[: 3 * D], D)
        state = {
            "norm.linear.weight": src.get_tensor("single_blocks.0.modulation.lin.weight"),
            "norm.linear.bias": src.get_tensor("single_blocks.0.modulation.lin.bias"),
            "proj_mlp.weight": linear1_w[3 * D :],
            "proj_mlp.bias": linear1_b[3 * D :],
            "proj_out.weight": src.get_tensor("single_blocks.0.linear2.weight"),
            "proj_out.bias": src.get_tensor("single_blocks.0.linear2.bias"),
            "attn.norm_q.weight": src.get_tensor("single_blocks.0.norm.query_norm.scale"),
            "attn.norm_k.weight": src.get_tensor("single_blocks.0.norm.key_norm.scale"),
            "attn.to_q.weight": q,
            "attn.to_q.bias": bq,
            "attn.to_k.weight": k,
            "attn.to_k.bias": bk,
            "attn.to_v.weight": v,
            "attn.to_v.bias": bv,
        }

    with torch.device("meta"):
        block = FluxSingleTransformerBlock(D, HEADS, HEAD_DIM)
    block.load_state_dict(state, strict=True, assign=True)
    config = LoraConfig(
        r=RANK,
        lora_alpha=RANK,
        lora_dropout=0.0,
        init_lora_weights="gaussian",
        target_modules=[
            "norm.linear",
            "proj_mlp",
            "proj_out",
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
        ],
    )
    return inject_adapter_in_model(config, block, adapter_name=ADAPTER)


class XEmbedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x_embedder = torch.nn.Linear(64, D)


def load_x_embedder(base_path: Path):
    with safe_open(base_path, framework="pt", device="cpu") as src:
        state = {
            "x_embedder.weight": src.get_tensor("img_in.weight"),
            "x_embedder.bias": src.get_tensor("img_in.bias"),
        }
    with torch.device("meta"):
        module = XEmbedder()
    module.load_state_dict(state, strict=True, assign=True)
    config = LoraConfig(
        r=RANK,
        lora_alpha=RANK,
        lora_dropout=0.0,
        init_lora_weights="gaussian",
        target_modules=["x_embedder"],
    )
    return inject_adapter_in_model(config, module, adapter_name=ADAPTER)


def load_official_adapters(
    adapter_path: Path, x_embedder: torch.nn.Module, double: torch.nn.Module, single: torch.nn.Module
) -> None:
    modules = (
        (x_embedder, "transformer.x_embedder.", "x_embedder."),
        (double, "transformer.transformer_blocks.0.", ""),
        (single, "transformer.single_transformer_blocks.0.", ""),
    )
    with safe_open(adapter_path, framework="pt", device="cpu") as src:
        keys = set(src.keys())
        for module, source_prefix, module_prefix in modules:
            mapped = {}
            for key in keys:
                if not key.startswith(source_prefix):
                    continue
                tail = key[len(source_prefix) :]
                tail = tail.replace("lora_A.weight", f"lora_A.{ADAPTER}.weight")
                tail = tail.replace("lora_B.weight", f"lora_B.{ADAPTER}.weight")
                mapped[module_prefix + tail] = src.get_tensor(key)
            result = module.load_state_dict(mapped, strict=False, assign=True)
            unexpected = list(result.unexpected_keys)
            missing_lora = [key for key in result.missing_keys if "lora_" in key]
            if unexpected or missing_lora:
                raise RuntimeError(
                    f"adapter mapping failed for {source_prefix}: "
                    f"unexpected={unexpected}, missing_lora={missing_lora}"
                )


def adapter_parameters(prefix: str, module: torch.nn.Module):
    for name, parameter in module.named_parameters():
        if "lora_" in name:
            yield f"grad.{prefix}.{name}", parameter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    require_provenance(args.source, args.adapter)
    sys.path.insert(0, str(args.source))
    from diffusers.models.transformers.transformer_flux import FluxPosEmbed
    from omini.pipeline.flux_omini import block_forward, lora_forward, single_block_forward

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the official OminiControl oracle")
    torch.manual_seed(20260731)
    torch.cuda.manual_seed_all(20260731)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    x_embedder = load_x_embedder(args.base)
    double = load_double_block(args.base)
    single = load_single_block(args.base)
    load_official_adapters(args.adapter, x_embedder, double, single)
    for module in (x_embedder, double, single):
        module.requires_grad_(False)
        for _, parameter in adapter_parameters("unused", module):
            parameter.requires_grad_(True)
        module.to(device=device, dtype=dtype).train()

    generator = torch.Generator(device=device).manual_seed(20260731)
    raw_main = torch.randn(1, 4, 64, generator=generator, device=device, dtype=dtype)
    raw_cond = torch.randn(1, 4, 64, generator=generator, device=device, dtype=dtype)
    text = torch.randn(1, 2, D, generator=generator, device=device, dtype=dtype)
    tembs = [
        torch.randn(1, D, generator=generator, device=device, dtype=dtype)
        for _ in range(3)
    ]
    raw_main.requires_grad_(True)
    raw_cond.requires_grad_(True)
    text.requires_grad_(True)
    for temb in tembs:
        temb.requires_grad_(True)

    main_hidden = lora_forward(x_embedder.x_embedder, raw_main, None)
    cond_hidden = lora_forward(x_embedder.x_embedder, raw_cond, ADAPTER)

    txt_ids = torch.tensor([[0, 0, 0], [0, 0, 1]], device=device, dtype=dtype)
    img_ids = torch.tensor(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]], device=device, dtype=dtype
    )
    pos = FluxPosEmbed(theta=10_000, axes_dim=[16, 56, 56]).to(device)
    position_embs = [pos(txt_ids), pos(img_ids), pos(img_ids)]
    adapters = [None, None, ADAPTER]
    group_mask = torch.ones((3, 3), device=device, dtype=torch.bool)

    double_images, double_text = block_forward(
        self=double,
        image_hidden_states=[main_hidden, cond_hidden],
        text_hidden_states=[text],
        tembs=tembs,
        adapters=adapters,
        position_embs=position_embs,
        group_mask=group_mask,
    )
    single_out = single_block_forward(
        self=single,
        hidden_states=[double_text[0], *double_images],
        tembs=tembs,
        adapters=adapters,
        position_embs=position_embs,
        group_mask=group_mask,
    )

    loss_probe = torch.randn(
        single_out[1].shape, generator=generator, device=device, dtype=dtype
    )
    loss = (single_out[1].float() * loss_probe.float()).mean()
    loss.backward()

    fixture = {
        "input.raw_main": raw_main.detach().float().cpu(),
        "input.raw_cond": raw_cond.detach().float().cpu(),
        "input.text": text.detach().float().cpu(),
        "input.temb_text": tembs[0].detach().float().cpu(),
        "input.temb_main": tembs[1].detach().float().cpu(),
        "input.temb_cond": tembs[2].detach().float().cpu(),
        "input.txt_ids": txt_ids.detach().float().cpu(),
        "input.img_ids": img_ids.detach().float().cpu(),
        "input.loss_probe": loss_probe.detach().float().cpu(),
        "output.x_main": main_hidden.detach().float().cpu(),
        "output.x_cond": cond_hidden.detach().float().cpu(),
        "output.double_text": double_text[0].detach().float().cpu(),
        "output.double_main": double_images[0].detach().float().cpu(),
        "output.double_cond": double_images[1].detach().float().cpu(),
        "output.single_text": single_out[0].detach().float().cpu(),
        "output.single_main": single_out[1].detach().float().cpu(),
        "output.single_cond": single_out[2].detach().float().cpu(),
        "grad.raw_main": raw_main.grad.detach().float().cpu(),
        "grad.raw_cond": raw_cond.grad.detach().float().cpu(),
        "grad.text": text.grad.detach().float().cpu(),
        "grad.temb_text": tembs[0].grad.detach().float().cpu(),
        "grad.temb_main": tembs[1].grad.detach().float().cpu(),
        "grad.temb_cond": tembs[2].grad.detach().float().cpu(),
        "scalar.loss": loss.detach().reshape(1).float().cpu(),
    }
    expected_none_grads = []
    for module_name, module in (
        ("x_embedder", x_embedder),
        ("double", double),
        ("single", single),
    ):
        for name, parameter in adapter_parameters(module_name, module):
            if parameter.grad is None:
                expected_none_grads.append(name)
                fixture[name] = torch.zeros_like(parameter, dtype=torch.float32).cpu()
            else:
                fixture[name] = parameter.grad.detach().float().cpu()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(fixture, str(args.output))
    manifest = {
        "evidence_level": "official CUDA oracle fixture",
        "official_source": str(args.source),
        "official_commit": PINNED_OMINI_COMMIT,
        "official_adapter": str(args.adapter),
        "official_adapter_sha256": PINNED_FILL_SHA256,
        "base_checkpoint": str(args.base),
        "base_checkpoint_size": args.base.stat().st_size,
        "fixture": str(args.output),
        "fixture_sha256": sha256_file(args.output),
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "dtype": "bfloat16",
        "branches": ["text:no_adapter", "main:no_adapter", "condition:fill"],
        "group_mask": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        "loss": float(loss.detach()),
        "tensor_count": len(fixture),
        "expected_none_grads": expected_none_grads,
    }
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
