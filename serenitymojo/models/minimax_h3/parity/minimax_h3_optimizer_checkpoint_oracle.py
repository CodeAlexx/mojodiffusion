#!/usr/bin/env python3
"""Oracle for H3's 200-target inventory and reduced optimizer/private state.

Musubi pins the target/dtype/layout contract. PyTorch supplies numeric AdamW
trajectories for four synthetic representative adapters. This does not generate
or validate a Musubi-compatible LoRA weight file.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from pathlib import Path
from typing import List, Optional
from urllib.request import urlopen

import torch


COMMIT = "b8717864713c9e4e7ef3d56eba1fc695a9b626a5"
BASE = f"https://raw.githubusercontent.com/kohya-ss/musubi-tuner/{COMMIT}/src/musubi_tuner"
SOURCES = {
    "networks/lora_minimax_h3.py": (
        f"{BASE}/networks/lora_minimax_h3.py",
        "4c2c4c850f2ad6ad901e9d49088b54128f4e17ee09c88564db88602ede71fe17",
    ),
    "networks/lora.py": (
        f"{BASE}/networks/lora.py",
        "694bcf27bebd8911a7868628ac1bc075d07cc8e87fdb289993b17ecab71475d5",
    ),
    "minimax_h3/model.py": (
        f"{BASE}/minimax_h3/model.py",
        "500fcacf93b40fac49b1ccbb21d8b382cb1f1b9fbd7954d1ac08155b2d0d243a",
    ),
}
OUT = Path(__file__).resolve().parent / "fixtures/minimax_h3_optimizer_checkpoint_v1.json"
SCHEMA = "serenity.minimax_h3.target_inventory_reduced_optimizer_private_state.v1"
RANK = 2
LR = 3.0e-4
BETAS = (0.9, 0.999)
EPS = 1.0e-8
WEIGHT_DECAY = 0.01
REDUCED_GEOMETRIES = ((3, 8), (4, 5), (3, 6), (3, 4))


def fetch(name: str) -> str:
    url, expected = SOURCES[name]
    raw = urlopen(url, timeout=30).read()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise RuntimeError(f"pinned source mismatch for {name}: {actual}")
    return raw.decode()


def assignment(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return ast.literal_eval(node.value)
    raise RuntimeError(f"missing assignment {name}")


def config_defaults(tree: ast.Module) -> dict[str, int]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3Config":
            values = {}
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    try:
                        values[item.target.id] = ast.literal_eval(item.value)
                    except Exception:
                        pass
            return values
    raise RuntimeError("missing MiniMaxH3Config")


def execute_lora_module(source: str):
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LoRAModule")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0), cls],
        type_ignores=[],
    )
    namespace = {"torch": torch, "math": math, "Optional": Optional, "List": List}
    exec(compile(ast.fix_missing_locations(module), "<pinned-musubi-lora>", "exec"), namespace)
    base = torch.nn.Linear(3, 5, bias=False, dtype=torch.bfloat16)
    probe = namespace["LoRAModule"]("probe", base, lora_dim=2, alpha=2)
    if base.weight.dtype != torch.bfloat16:
        raise RuntimeError("oracle base dtype drift")
    if probe.lora_down.weight.dtype != torch.float32 or probe.lora_up.weight.dtype != torch.float32:
        raise RuntimeError("Musubi LoRA trainable dtype is no longer F32")
    if torch.count_nonzero(probe.lora_up.weight).item() != 0:
        raise RuntimeError("Musubi LoRA-up initialization is no longer zero")
    return {
        "base": str(base.weight.dtype),
        "lora_down": str(probe.lora_down.weight.dtype),
        "lora_up": str(probe.lora_up.weight.dtype),
        "lora_up_zero_initialized": True,
    }


def f32_scalar(value: float | int) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float32)


def seed_values(n: int, adapter: int, arm: int) -> torch.Tensor:
    values = []
    for i in range(n):
        centered = (i * 17 + adapter * 11 + arm * 7) % 29 - 14
        value = (
            f32_scalar(0.0037) * f32_scalar(adapter + 1)
            + f32_scalar(0.000173) * f32_scalar(centered)
            + f32_scalar(i + 1) * f32_scalar(1.3e-7)
        )
        values.append(value)
    return torch.stack(values)


def gradient(n: int, adapter: int, arm: int, step: int) -> torch.Tensor:
    values = []
    for i in range(n):
        centered = (i * 13 + adapter * 19 + arm * 5 + step * 23) % 37 - 18
        values.append(f32_scalar(centered) * f32_scalar(7.1e-5) + f32_scalar(step) * f32_scalar(2.9e-6))
    return torch.stack(values)


def flat(value: torch.Tensor) -> list[float]:
    return value.detach().to(torch.float32).reshape(-1).tolist()


def optimizer_trajectory(inventory: list[dict]) -> dict:
    params: list[tuple[torch.nn.Parameter, torch.nn.Parameter]] = []
    cases = []
    for index, (in_f, out_f) in enumerate(REDUCED_GEOMETRIES):
        a = torch.nn.Parameter(seed_values(RANK * in_f, index, 0))
        b = torch.nn.Parameter(seed_values(out_f * RANK, index, 1))
        params.append((a, b))
        cases.append(
            {
                "family": inventory[index]["family"],
                "musubi_prefix": inventory[index]["musubi_prefix"],
                "rank": RANK,
                "in_features": in_f,
                "out_features": out_f,
                "initial_a": flat(a),
                "initial_b": flat(b),
                "steps": [],
            }
        )
    optimizer = torch.optim.AdamW(
        [parameter for pair in params for parameter in pair],
        lr=LR,
        betas=BETAS,
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
        foreach=False,
        fused=False,
    )
    for step in range(1, 4):
        for index, (a, b) in enumerate(params):
            a.grad = gradient(a.numel(), index, 0, step)
            b.grad = gradient(b.numel(), index, 1, step)
        optimizer.step()
        for index, (a, b) in enumerate(params):
            a_state = optimizer.state[a]
            b_state = optimizer.state[b]
            cases[index]["steps"].append(
                {
                    "step": step,
                    "a": flat(a), "b": flat(b),
                    "ma": flat(a_state["exp_avg"]), "va": flat(a_state["exp_avg_sq"]),
                    "mb": flat(b_state["exp_avg"]), "vb": flat(b_state["exp_avg_sq"]),
                }
            )
    return {
        "oracle": "torch.optim.AdamW foreach=False fused=False",
        "rank": RANK,
        "learning_rate": LR,
        "beta1": BETAS[0],
        "beta2": BETAS[1],
        "eps": EPS,
        "weight_decay": WEIGHT_DECAY,
        "cases": cases,
    }


def build_document() -> dict:
    sources = {name: fetch(name) for name in SOURCES}
    arch_tree = ast.parse(sources["networks/lora_minimax_h3.py"])
    model_tree = ast.parse(sources["minimax_h3/model.py"])
    pattern = assignment(arch_tree, "MINIMAX_H3_DEFAULT_TARGET_PATTERN")
    replace_modules = assignment(arch_tree, "MINIMAX_H3_TARGET_REPLACE_MODULES")
    if replace_modules != ["DiTBlock"]:
        raise RuntimeError("pinned H3 replacement surface drifted")
    dims = config_defaults(model_tree)
    hidden = dims["hidden_size"]
    inner = dims["num_attention_heads"] * dims["attention_head_dim"]
    ffn = dims["ffn_hidden_size"]
    families = (
        ("qkv_proj", "attn.qkv_proj", hidden, 3 * inner, False),
        ("out_proj", "attn.out_proj", inner, hidden, False),
        ("fc1", "mlp.fc1", hidden, 2 * ffn, True),
        ("fc2", "mlp.fc2", ffn, hidden, False),
    )
    inventory = []
    for layer in range(dims["num_layers"]):
        for family, suffix, in_f, out_f, swap in families:
            module_path = f"blocks.{layer}.{suffix}"
            if re.fullmatch(pattern, module_path) is None:
                raise RuntimeError(f"pinned target regex rejected {module_path}")
            inventory.append(
                {
                    "layer": layer,
                    "family": family,
                    "module_path": module_path,
                    "musubi_prefix": f"lora_unet_{module_path.replace('.', '_')}",
                    "in_features": in_f,
                    "out_features": out_f,
                    "fc1_runtime_swap": swap,
                }
            )
    if len(inventory) != 200 or len({item["musubi_prefix"] for item in inventory}) != 200:
        raise RuntimeError("H3 inventory is not exactly 200 unique targets")
    model_source = sources["minimax_h3/model.py"]
    if "gate, values = self.fc1(hidden_states).chunk(2, dim=-1)" not in model_source:
        raise RuntimeError("pinned raw FC1 [gate;values] contract drifted")
    lora_source = sources["networks/lora.py"]
    if 'lora_name = f"{pfx}.{original_name}".replace(".", "_")' not in lora_source:
        raise RuntimeError("pinned LoRA naming transform drifted")
    return {
        "schema": SCHEMA,
        "oracle_repository": "kohya-ss/musubi-tuner",
        "oracle_commit": COMMIT,
        "source_sha256": {name: expected for name, (_, expected) in SOURCES.items()},
        "source_contracts": [
            "lora_minimax_h3.MINIMAX_H3_TARGET_REPLACE_MODULES",
            "lora_minimax_h3.MINIMAX_H3_DEFAULT_TARGET_PATTERN",
            "lora.LoRAModule.__init__",
            "lora.LoRANetwork.__init__.create_modules",
            "model.MLP.forward",
        ],
        "executed_dtype_probe": execute_lora_module(lora_source),
        "target_pattern": pattern,
        "inventory": inventory,
        "optimizer_trajectory": optimizer_trajectory(inventory),
        "excluded_evidence": [
            "Musubi LoRA weight-file export/import",
            "full 200-target optimizer allocation or update",
            "model forward/backward, dataset, cache, trainer, and product launch",
        ],
    }


def validate_document(document: dict) -> None:
    expected_top = {
        "schema", "oracle_repository", "oracle_commit", "source_sha256",
        "source_contracts", "executed_dtype_probe", "target_pattern",
        "inventory", "optimizer_trajectory", "excluded_evidence",
    }
    if set(document) != expected_top or document.get("schema") != SCHEMA:
        raise RuntimeError("fixture top-level schema/key set mismatch")
    if document.get("oracle_commit") != COMMIT:
        raise RuntimeError("fixture commit mismatch")
    if document.get("source_sha256") != {name: expected for name, (_, expected) in SOURCES.items()}:
        raise RuntimeError("fixture source hashes mismatch")
    inventory = document["inventory"]
    inventory_keys = {
        "layer", "family", "module_path", "musubi_prefix", "in_features",
        "out_features", "fc1_runtime_swap",
    }
    if len(inventory) != 200 or len({item["musubi_prefix"] for item in inventory}) != 200:
        raise RuntimeError("fixture inventory is not 200 unique targets")
    if any(set(item) != inventory_keys for item in inventory):
        raise RuntimeError("fixture inventory entry key set mismatch")
    trajectory = document["optimizer_trajectory"]
    expected_trajectory_keys = {
        "oracle", "rank", "learning_rate", "beta1", "beta2",
        "eps", "weight_decay", "cases",
    }
    if set(trajectory) != expected_trajectory_keys or trajectory["rank"] != RANK:
        raise RuntimeError("optimizer trajectory schema mismatch")
    if len(trajectory["cases"]) != 4:
        raise RuntimeError("optimizer trajectory must contain four representative adapters")
    case_keys = {
        "family", "musubi_prefix", "rank", "in_features", "out_features",
        "initial_a", "initial_b", "steps",
    }
    state_keys = {"step", "a", "b", "ma", "va", "mb", "vb"}
    for index, case in enumerate(trajectory["cases"]):
        if set(case) != case_keys or case["family"] != inventory[index]["family"]:
            raise RuntimeError(f"optimizer case {index} schema/family mismatch")
        in_f, out_f = REDUCED_GEOMETRIES[index]
        if (case["in_features"], case["out_features"]) != (in_f, out_f):
            raise RuntimeError(f"optimizer case {index} reduced geometry mismatch")
        if len(case["initial_a"]) != RANK * in_f or len(case["initial_b"]) != out_f * RANK:
            raise RuntimeError(f"optimizer case {index} initial length mismatch")
        if [state.get("step") for state in case["steps"]] != [1, 2, 3]:
            raise RuntimeError(f"optimizer case {index} step sequence mismatch")
        if any(set(state) != state_keys for state in case["steps"]):
            raise RuntimeError(f"optimizer case {index} state key set mismatch")
    if "Musubi LoRA weight-file export/import" not in document["excluded_evidence"]:
        raise RuntimeError("weight-file export/import exclusion missing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    document = build_document()
    validate_document(document)
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = args.output.with_suffix(".sha256")
    sidecar_text = f"{digest}  {args.output.name}\n"
    if args.check:
        if not args.output.exists() or args.output.read_bytes() != payload:
            raise RuntimeError("fixture bytes differ from deterministic regeneration")
        if not sidecar.exists() or sidecar.read_text() != sidecar_text:
            raise RuntimeError("fixture SHA-256 sidecar mismatch")
        validate_document(json.loads(args.output.read_text()))
        print(f"PASS {args.output} {digest}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    sidecar.write_text(sidecar_text)
    print(f"wrote {args.output}")
    print(f"wrote {sidecar}")
    print(f"generator_torch={torch.__version__} (noncanonical)")
    print(digest)


if __name__ == "__main__":
    main()
