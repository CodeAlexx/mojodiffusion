#!/usr/bin/env python3
"""First real Bernini-R transformer-block oracle from pinned creator code."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
import types
from pathlib import Path

import torch
from safetensors import safe_open


ORACLE_REVISION = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
MODEL_REVISION = "de8c4621d3ac75cc33efe3db8deaed2023e9ac8c"
DEFAULT_CREATOR = Path("/home/alex/Bernini")
DEFAULT_SOURCE = Path(
    "/home/alex/.serenity/models/checkpoints/Bernini-R-Diffusers/transformer"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "output/checks/bernini_r/block_oracle"


def write_f32(path: Path, tensor: torch.Tensor) -> None:
    values = tensor.detach().float().cpu().contiguous().view(-1).tolist()
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def load_creator_module(creator: Path):
    """Load transformer_wan without executing models/__init__ or veomni paths."""
    sys.path.insert(0, str(creator))
    import bernini  # noqa: F401 - package __init__ is intentionally lazy
    import bernini.attention  # noqa: F401
    import bernini.parallel  # noqa: F401

    models = types.ModuleType("bernini.models")
    models.__path__ = [str(creator / "bernini/models")]
    models.__package__ = "bernini.models"
    sys.modules["bernini.models"] = models
    path = creator / "bernini/models/transformer_wan.py"
    spec = importlib.util.spec_from_file_location("bernini.models.transformer_wan", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load creator transformer module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--creator", type=Path, default=DEFAULT_CREATOR)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    creator = args.creator.resolve(strict=True)
    source = args.source.resolve(strict=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=creator, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=creator, text=True
    ).strip()
    if revision != ORACLE_REVISION or dirty:
        raise RuntimeError(
            f"creator authority mismatch: revision={revision}, dirty={bool(dirty)}"
        )

    module, source_code = load_creator_module(creator)
    index_path = source / "diffusion_pytorch_model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    block_names = sorted(k for k in index["weight_map"] if k.startswith("blocks.0."))
    if len(block_names) != 27:
        raise RuntimeError(f"expected 27 block-0 tensors, found {len(block_names)}")
    shards = {index["weight_map"][key] for key in block_names}
    if len(shards) != 1:
        raise RuntimeError(f"block 0 unexpectedly spans shards: {sorted(shards)}")
    shard_path = source / next(iter(shards))
    if not shard_path.is_file():
        raise RuntimeError(f"official block-0 shard is not downloaded yet: {shard_path}")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    with torch.device("meta"):
        block = module.WanTransformerBlock(
            dim=5120,
            ffn_dim=13824,
            num_heads=40,
            qk_norm="rms_norm_across_heads",
            cross_attn_norm=True,
            eps=1e-6,
            added_kv_proj_dim=None,
        )
    state = {}
    keep_f32 = {"scale_shift_table", "norm1", "norm2", "norm3", "time_embedder"}
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        for full_name in block_names:
            short_name = full_name.removeprefix("blocks.0.")
            target_dtype = (
                torch.float32
                if any(part in keep_f32 for part in short_name.split("."))
                else dtype
            )
            state[short_name] = handle.get_tensor(full_name).to(
                device=device, dtype=target_dtype
            )
    missing, unexpected = block.load_state_dict(state, strict=True, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"creator block state mismatch: missing={missing}, unexpected={unexpected}")
    block.eval()

    generator = torch.Generator(device="cpu").manual_seed(20260601)
    sequence = 8
    text = 8
    x = torch.randn((1, sequence, 5120), generator=generator).to(device, dtype)
    temb = torch.randn((1, sequence, 6, 5120), generator=generator).to(device, torch.float32)
    context = torch.randn((1, text, 5120), generator=generator).to(device, dtype)

    # T2V target source_id=0: creator source-ID phase is exactly one.  Generate
    # the spatial/temporal phase through the untouched creator RoPE class.
    rope_module = module.WanRotaryPosEmbed(
        128, (1, 2, 2), 1024, theta=10000.0, use_src_id_rotary_emb=True
    )
    latent_shape_probe = torch.empty((1, 16, 1, 4, 8), device=device, dtype=dtype)
    rotary = rope_module(latent_shape_probe, source_id=0).transpose(1, 2)
    cos = rotary.real.squeeze(0).squeeze(1)
    sin = rotary.imag.squeeze(0).squeeze(1)

    cu = torch.tensor([0, sequence], device=device, dtype=torch.int32)
    cut = torch.tensor([0, text], device=device, dtype=torch.int32)
    kwargs = {
        "cu_seqlens_q_cache": cu,
        "max_seqlen_q_cache": sequence,
        "cu_seqlens_k_cross_cache": cut,
        "max_seqlen_k_cross_cache": text,
        "cu_seqlens_q_cross_cache": cu,
        "max_seqlen_q_cross_cache": sequence,
        "origin_hidden_states_seq_len": None,
        "split_hidden_states_seq_len": None,
    }
    with torch.inference_mode():
        output = block(
            x,
            context,
            temb,
            rotary,
            batch_image_vae_seqlen=[sequence],
            text_features_length=[text],
            **kwargs,
        )
    torch.cuda.synchronize()

    args.output.mkdir(parents=True, exist_ok=True)
    tensors = {
        "x": x,
        "temb": temb,
        "context": context,
        "cos": cos,
        "sin": sin,
        "output": output,
    }
    files = {}
    for name, tensor in tensors.items():
        path = args.output / f"{name}.bin"
        write_f32(path, tensor)
        files[name] = {
            "path": path.name,
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = {
        "schema": "serenity.bernini_r.block_oracle.v1",
        "creator_revision": revision,
        "creator_source": str(source_code),
        "creator_source_sha256": hashlib.sha256(source_code.read_bytes()).hexdigest(),
        "model_revision": MODEL_REVISION,
        "expert": "high-noise",
        "block": 0,
        "weight_shard": shard_path.name,
        "weight_shard_sha256": hashlib.sha256(shard_path.read_bytes()).hexdigest(),
        "attention_backend": __import__("bernini.attention", fromlist=["get_attention_backend"]).get_attention_backend(),
        "dtype": "bfloat16",
        "seed": 20260601,
        "source_id": 0,
        "sequence": sequence,
        "text": text,
        "files": files,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
