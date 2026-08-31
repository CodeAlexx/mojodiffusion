#!/usr/bin/env python3
"""Pinned Musubi oracle for MiniMax-H3 AdaLN modality segmentation.

This is fixture generation only. It downloads immutable sources, verifies their
exact bytes, executes the selected upstream definitions, and emits small F32
T2VA/FL2VA/Ref2VA cases. It does not import a product trainer or touch a model.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from urllib.request import urlopen

import torch
import torch.nn.functional as F
from torch import nn


COMMIT = "b8717864713c9e4e7ef3d56eba1fc695a9b626a5"
ROOT = f"https://raw.githubusercontent.com/kohya-ss/musubi-tuner/{COMMIT}/src/musubi_tuner/minimax_h3"
SOURCES = {
    "model.py": (
        f"{ROOT}/model.py",
        "500fcacf93b40fac49b1ccbb21d8b382cb1f1b9fbd7954d1ac08155b2d0d243a",
    ),
    "packing.py": (
        f"{ROOT}/packing.py",
        "464371faca4f156de883ce37022533c0fd3e0965723648c904d2f1cc09be2cc3",
    ),
}
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "fixtures/minimax_h3_adaln_modality_v1.json"
D, E, EXPAND, MODALITIES = 4, 3, 6, 3


def fetch_verified(name: str) -> str:
    url, expected = SOURCES[name]
    source = urlopen(url, timeout=30).read()
    actual = hashlib.sha256(source).hexdigest()
    if actual != expected:
        raise RuntimeError(f"pinned {name} SHA-256 mismatch: {actual}")
    return source.decode("utf-8")


def exec_selected(source: str, names: set[str], namespace: dict) -> dict:
    tree = ast.parse(source)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    found = {node.name for node in selected}
    if found != names:
        raise RuntimeError(f"missing pinned definitions: {sorted(names - found)}")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0), *selected],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "<pinned-musubi>", "exec"), namespace)
    return namespace


def f32_list(value: torch.Tensor) -> list[float]:
    return value.detach().to(torch.float32).reshape(-1).tolist()


def build_case(name: str, spec: dict, api: dict, weights: torch.Tensor, bias: torch.Tensor) -> dict:
    rows = api["build_timestep_rows"](
        spec["layout"],
        text_token_tags=torch.tensor([spec["text_token_tags"]], dtype=torch.int64),
        model_t_video=spec["model_t_video"],
        model_t_audio=spec["model_t_audio"],
        visual_condition_clean=spec["visual_condition_clean"],
        audio_condition_clean=spec["audio_condition_clean"],
    )
    unique = rows.unique_timesteps
    temb = torch.stack((unique + 0.07, unique.square() - 0.11, 0.83 - 0.37 * unique), dim=-1)
    proj = api["AdalnProj"](E, D, EXPAND, MODALITIES, dtype=torch.float32, apply_silu=True)
    with torch.no_grad():
        proj.linear.weight.copy_(weights)
        proj.linear.bias.copy_(bias)
    chunks = proj(temb)

    sequence = spec["layout"].row_count
    hidden = torch.arange(sequence * D, dtype=torch.float32).reshape(1, sequence, D)
    hidden = torch.sin(hidden * 0.31 + 0.17) * 0.7
    update = torch.cos(torch.arange(sequence * D, dtype=torch.float32).reshape(1, sequence, D) * 0.23 - 0.29) * 0.4
    modulated = api["_mod_scale_shift"](hidden.clone(), chunks[0], chunks[1], rows.block_segments)
    gated = api["_mod_gate"](hidden, update, chunks[2], rows.block_segments)
    modulated_mlp = api["_mod_scale_shift"](hidden.clone(), chunks[3], chunks[4], rows.block_segments)
    gated_mlp = api["_mod_gate"](hidden, update, chunks[5], rows.block_segments)

    # Materialize all six selected parameters per sequence row. This catches
    # both projection reshape/chunk order and segment expansion errors.
    selected = torch.empty(sequence, EXPAND, D, dtype=torch.float32)
    for start, stop, row in rows.block_segments:
        for parameter, chunk in enumerate(chunks):
            selected[start:stop, parameter] = chunk[row]

    if len(rows.block_segments) < 3 or torch.unique(rows.block_adaln_indices).numel() < 3:
        raise RuntimeError(f"degenerate {name} segmentation fixture")
    if selected.abs().sum() <= 1.0 or modulated.std() <= 0.05 or gated.std() <= 0.05:
        raise RuntimeError(f"degenerate {name} numerical fixture")
    return {
        "task": name,
        "text_token_tags": spec["text_token_tags"],
        "model_t_video": spec["model_t_video"],
        "model_t_audio": spec["model_t_audio"],
        "visual_condition_clean": spec["visual_condition_clean"],
        "audio_condition_clean": spec["audio_condition_clean"],
        "segment_kinds": [segment.kind for segment in spec["layout"].segments],
        "segment_lengths": [segment.stop - segment.start for segment in spec["layout"].segments],
        "unique_timesteps": f32_list(unique),
        "timestep_embeddings": f32_list(temb),
        "block_adaln_indices": rows.block_adaln_indices.reshape(-1).tolist(),
        "block_segments": [list(segment) for segment in rows.block_segments],
        "hidden": f32_list(hidden),
        "update": f32_list(update),
        "selected_parameters": f32_list(selected),
        "mod_scale_shift": f32_list(modulated),
        "mod_gate": f32_list(gated),
        "mod_scale_shift_mlp": f32_list(modulated_mlp),
        "mod_gate_mlp": f32_list(gated_mlp),
    }


def build_document() -> dict:
    model_source = fetch_verified("model.py")
    packing_source = fetch_verified("packing.py")
    model_api = exec_selected(
        model_source,
        {"AdalnProj", "_mod_scale_shift", "_mod_gate"},
        {"torch": torch, "F": F, "nn": nn},
    )
    packing_api = exec_selected(
        packing_source,
        {
            "H3VideoGeometry", "H3ReferenceGeometry", "H3TimeOverrides",
            "H3RowSegment", "H3PackedLayout", "H3TimestepRows",
            "_coerce_video_geometry", "_validate_video_latent_frames",
            "_expected_audio_frames", "_fl_condition_roles", "build_h3_layout",
            "_single_model_time", "_validate_clean_coefficient",
            "_build_modulation_segments", "build_timestep_rows",
        },
        {
            "torch": torch, "math": math,
            "dataclass": __import__("dataclasses").dataclass,
            "H3Task": str, "H3ReferenceKind": str, "H3SegmentKind": str,
            "Sequence": list, "STEREO_CHANNELS": 2,
            "ONE_FRAME_VIDEO_LATENT_FRAMES": 1,
            "ONE_FRAME_AUDIO_LATENT_FRAMES": 2,
        },
    )
    api = {**model_api, **packing_api}

    weight_values = torch.arange(EXPAND * MODALITIES * D * E, dtype=torch.float32)
    weights = torch.sin(weight_values * 0.173 + 0.41).reshape(EXPAND * MODALITIES * D, E) * 0.29
    bias_values = torch.arange(EXPAND * MODALITIES * D, dtype=torch.float32)
    bias = torch.cos(bias_values * 0.119 - 0.23) * 0.17

    geometry = packing_api["H3VideoGeometry"]
    reference = packing_api["H3ReferenceGeometry"]
    make_layout = packing_api["build_h3_layout"]
    target = geometry(2, 2, 2)
    target_audio_frames = packing_api["_expected_audio_frames"](target.frames)
    specs = {
        "t2va": {
            "layout": make_layout(task="t2va", text_length=3, target_video=target, target_audio_frames=target_audio_frames),
            "text_token_tags": [0, 1, 0], "model_t_video": 0.31, "model_t_audio": 0.67,
            "visual_condition_clean": 0.999, "audio_condition_clean": 1.0,
        },
        "fl2va": {
            "layout": make_layout(
                task="fl2va", text_length=2, target_video=target,
                target_audio_frames=target_audio_frames,
                visual_conditions=(geometry(1, 2, 2), geometry(1, 2, 2)),
            ),
            "text_token_tags": [1, 0], "model_t_video": 0.23, "model_t_audio": 0.59,
            "visual_condition_clean": 0.91, "audio_condition_clean": 0.97,
        },
        "ref2va": {
            "layout": make_layout(
                task="ref2va", text_length=3, target_video=target,
                target_audio_frames=target_audio_frames,
                references=(
                    reference("image", video=geometry(1, 2, 2)),
                    reference("audio", audio_frames=2),
                ),
            ),
            "text_token_tags": [1, 0, 1], "model_t_video": 0.18, "model_t_audio": 0.72,
            "visual_condition_clean": 0.93, "audio_condition_clean": 0.98,
        },
    }
    cases = [build_case(name, spec, api, weights, bias) for name, spec in specs.items()]
    return {
        "schema": "serenity.minimax_h3.adaln_modality_oracle.v1",
        "oracle_repository": "kohya-ss/musubi-tuner",
        "oracle_commit": COMMIT,
        "source_sha256": {name: expected for name, (_, expected) in SOURCES.items()},
        "source_contracts": {
            "model.py": ["AdalnProj.forward", "_mod_scale_shift", "_mod_gate", "DiTBlock.forward"],
            "packing.py": ["build_h3_layout", "_build_modulation_segments", "build_timestep_rows"],
        },
        "executed_upstream_definitions": [
            "AdalnProj", "_mod_scale_shift", "_mod_gate", "H3VideoGeometry",
            "H3ReferenceGeometry", "H3RowSegment", "H3PackedLayout",
            "H3TimestepRows", "build_h3_layout", "_build_modulation_segments",
            "build_timestep_rows",
        ],
        "dtype": "torch.float32",
        "dimensions": {"hidden": D, "timestep": E, "expand": EXPAND, "modalities": MODALITIES},
        "adaln_weight": f32_list(weights),
        "adaln_bias": f32_list(bias),
        "cases": cases,
    }


def validate_document(document: dict) -> None:
    if document.get("schema") != "serenity.minimax_h3.adaln_modality_oracle.v1":
        raise RuntimeError("schema mismatch")
    if document.get("oracle_commit") != COMMIT:
        raise RuntimeError("commit mismatch")
    if document.get("source_sha256") != {name: expected for name, (_, expected) in SOURCES.items()}:
        raise RuntimeError("source hash map mismatch")
    if document.get("dimensions") != {"hidden": D, "timestep": E, "expand": EXPAND, "modalities": MODALITIES}:
        raise RuntimeError("dimension contract mismatch")
    cases = document.get("cases", [])
    if [case.get("task") for case in cases] != ["t2va", "fl2va", "ref2va"]:
        raise RuntimeError("task inventory/order mismatch")
    expected = {
        "t2va": (["text", "target_audio", "target_video"], [3, 16, 2]),
        "fl2va": (["text", "visual_condition", "visual_condition", "target_audio", "target_video"], [2, 1, 1, 16, 2]),
        "ref2va": (["text", "visual_condition", "audio_condition", "target_audio", "target_video"], [3, 1, 4, 16, 2]),
    }
    for case in cases:
        kinds, lengths = expected[case["task"]]
        if case.get("segment_kinds") != kinds or case.get("segment_lengths") != lengths:
            raise RuntimeError(f"{case['task']} segment schema mismatch")
        if len(case.get("block_adaln_indices", [])) != sum(lengths):
            raise RuntimeError(f"{case['task']} AdaLN row count mismatch")
    if cases[1]["segment_lengths"][1] != cases[1]["segment_lengths"][2]:
        raise RuntimeError("FL2VA first/last lengths differ")
    if cases[2]["segment_lengths"][2] % 2:
        raise RuntimeError("Ref2VA audio-condition row length must be even")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    document = build_document()
    validate_document(document)
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = args.output.with_suffix(".sha256")
    expected_sidecar = f"{digest}  {args.output.name}\n"
    if args.check:
        if not args.output.exists() or args.output.read_bytes() != payload:
            raise RuntimeError("fixture bytes differ from deterministic regeneration")
        if not sidecar.exists() or sidecar.read_text() != expected_sidecar:
            raise RuntimeError("fixture SHA-256 sidecar mismatch")
        validate_document(json.loads(args.output.read_text()))
        print(f"PASS {args.output} {digest}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    sidecar.write_text(expected_sidecar)
    print(f"wrote {args.output}")
    print(f"wrote {sidecar}")
    print(digest)


if __name__ == "__main__":
    main()
