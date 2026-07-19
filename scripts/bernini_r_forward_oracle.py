#!/usr/bin/env python3
"""Full 40-block Bernini-R first-forward oracle with bounded residency."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
import types
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open


ORACLE_REVISION = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
MODEL_REVISION = "de8c4621d3ac75cc33efe3db8deaed2023e9ac8c"
DEFAULT_CREATOR = Path("/home/alex/Bernini")
DEFAULT_MODEL = Path("/home/alex/.serenity/models/checkpoints/Bernini-R-Diffusers")
DEFAULT_ARTIFACT = DEFAULT_MODEL / "serenity_bernini_r_manifest.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "output/checks/bernini_r/forward_oracle"


def write_f32(path: Path, tensor: torch.Tensor) -> None:
    values = tensor.detach().float().cpu().contiguous().view(-1).tolist()
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def load_creator_module(creator: Path):
    sys.path.insert(0, str(creator))
    import bernini  # noqa: F401
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


class StreamingBlocks(nn.Module):
    """Yield exactly one creator block at a time to the untouched forward loop."""

    def __init__(self, loader, count: int):
        super().__init__()
        self.loader = loader
        self.count = count

    def __iter__(self):
        for index in range(self.count):
            yield self.loader(index)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--creator", type=Path, default=DEFAULT_CREATOR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--artifact-manifest", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--component", choices=("transformer", "transformer_2"), required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    creator = args.creator.resolve(strict=True)
    model_root = args.model.resolve(strict=True)
    artifact_path = args.artifact_manifest.resolve(strict=True)
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
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("revision") != MODEL_REVISION or artifact.get("artifact_gate_passed") is not True:
        raise RuntimeError("artifact manifest is not the admitted pinned snapshot")

    module, creator_source = load_creator_module(creator)
    source = model_root / args.component
    index_path = source / "diffusion_pytorch_model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    if len(index) != 1095:
        raise RuntimeError(f"unexpected transformer tensor count: {len(index)}")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    with torch.device("meta"):
        model = module.WanTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=40,
            attention_head_dim=128,
            in_channels=16,
            out_channels=16,
            text_dim=4096,
            freq_dim=256,
            ffn_dim=13824,
            num_layers=40,
            cross_attn_norm=True,
            qk_norm="rms_norm_across_heads",
            eps=1e-6,
            rope_max_seq_len=1024,
            use_src_id_rotary_emb=True,
        )
    # RoPE has no checkpoint parameters and must be created on a real device.
    model.rope = module.WanRotaryPosEmbed(
        128, (1, 2, 2), 1024, theta=10000.0, use_src_id_rotary_emb=True
    )

    grouped: dict[str, list[str]] = defaultdict(list)
    for name, shard in index.items():
        if not name.startswith("blocks."):
            grouped[shard].append(name)
    keep_f32 = {"scale_shift_table", "norm1", "norm2", "norm3", "time_embedder"}

    def target_dtype(name: str):
        return (
            torch.float32
            if any(part in keep_f32 for part in name.split("."))
            else dtype
        )

    shared_state = {}
    for shard, names in grouped.items():
        with safe_open(source / shard, framework="pt", device="cpu") as handle:
            for name in names:
                shared_state[name] = handle.get_tensor(name).to(
                    device=device, dtype=target_dtype(name)
                )
    missing, unexpected = model.load_state_dict(shared_state, strict=False, assign=True)
    if unexpected or any(not name.startswith("blocks.") for name in missing):
        raise RuntimeError(
            f"shared creator state mismatch: non-block missing={missing[:10]}, unexpected={unexpected}"
        )
    del shared_state

    def load_block(block_index: int):
        prefix = f"blocks.{block_index}."
        names = sorted(name for name in index if name.startswith(prefix))
        if len(names) != 27:
            raise RuntimeError(f"block {block_index} tensor count {len(names)} != 27")
        by_shard: dict[str, list[str]] = defaultdict(list)
        for name in names:
            by_shard[index[name]].append(name)
        state = {}
        for shard, shard_names in by_shard.items():
            with safe_open(source / shard, framework="pt", device="cpu") as handle:
                for name in shard_names:
                    short_name = name.removeprefix(prefix)
                    state[short_name] = handle.get_tensor(name).to(
                        device=device, dtype=target_dtype(short_name)
                    )
        with torch.device("meta"):
            block = module.WanTransformerBlock(
                5120, 13824, 40, "rms_norm_across_heads", True, 1e-6, None
            )
        result = block.load_state_dict(state, strict=True, assign=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"block {block_index} state mismatch: {result}")
        block.eval()
        print(f"[creator-stream] {args.component} block {block_index + 1}/40", flush=True)
        return block

    model.blocks = StreamingBlocks(load_block, 40)
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(20260601)
    latent = torch.randn((1, 16, 1, 4, 8), generator=generator).to(device, dtype)
    context = torch.zeros((1, 512, 4096), device=device, dtype=dtype)
    context[:, :8] = torch.randn((1, 8, 4096), generator=generator).to(device, dtype)
    timestep_value = 999 if args.component == "transformer" else 500
    timestep = torch.tensor([timestep_value], device=device, dtype=torch.int64)
    hidden, rotary = model.patch_vae_latent(latent, source_id=0)
    with torch.inference_mode():
        packed = model(
            hidden,
            timestep,
            context,
            rotary,
            batch_image_vae_seqlen=[8],
            text_features_length=[8],
            return_dict=False,
        )[0]
    # [1, t*h*w, pt*ph*pw*c] -> [1,c,t,h*ph,w*pw]
    spatial = (
        packed.view(1, 1, 2, 4, 1, 2, 2, 16)
        .permute(0, 7, 1, 4, 2, 5, 3, 6)
        .reshape(1, 16, 1, 4, 8)
    )
    torch.cuda.synchronize()

    out_dir = args.output / args.component
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = {"latent": latent, "context": context, "output": spatial}
    files = {}
    for name, tensor in tensors.items():
        path = out_dir / f"{name}.bin"
        write_f32(path, tensor)
        files[name] = {
            "path": path.name,
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = {
        "schema": "serenity.bernini_r.forward_oracle.v1",
        "creator_revision": revision,
        "creator_source": str(creator_source),
        "creator_source_sha256": hashlib.sha256(creator_source.read_bytes()).hexdigest(),
        "model_revision": MODEL_REVISION,
        "artifact_manifest_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "component": args.component,
        "timestep": timestep_value,
        "seed": 20260601,
        "source_id": 0,
        "geometry": {
            "latent": [1, 16, 1, 4, 8],
            "patch_grid": [1, 2, 4],
            "tokens": 8,
            "text_buffer_rows": 512,
            "text_valid_rows": 8,
        },
        "attention_backend": __import__("bernini.attention", fromlist=["get_attention_backend"]).get_attention_backend(),
        "dtype": "bfloat16",
        "files": files,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
