#!/usr/bin/env python3
"""Guard production inference weight sources.

The product path may load large model artifacts from the local model store
(`.serenity`) or, for explicitly pinned tokenizer/text-encoder snapshots, the
local Hugging Face cache. It must not depend on old implementation repos such as
EriDiffusion, SerenityFlow, SwarmUI, or ComfyUI at runtime.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]

PRODUCTION_SURFACES = [
    "serenitymojo/runtime/model_manifest.mojo",
    "serenitymojo/serve/dispatch_backend.mojo",
    "serenitymojo/serve/model_scan.mojo",
    "serenitymojo/serve/zimage_backend.mojo",
    "serenitymojo/pipeline/zimage_generate.mojo",
    "serenitymojo/serve/qwenimage_backend.mojo",
    "serenitymojo/pipeline/qwenimage_sample_cli.mojo",
    "serenitymojo/configs/qwenimage.json",
    "serenitymojo/serve/ideogram4_backend.mojo",
    "serenitymojo/serve/flux_backend.mojo",
    "serenitymojo/configs/flux.json",
    "serenitymojo/serve/klein_backend.mojo",
    "serenitymojo/serve/klein_runtime_backend.mojo",
    "serenitymojo/configs/klein9b.json",
    "serenitymojo/configs/klein4b.json",
    "serenitymojo/serve/sample_cli_backend.mojo",
    "serenitymojo/serve/sdxl_backend.mojo",
    "serenitymojo/serve/sd3_backend.mojo",
    "serenitymojo/serve/anima_backend.mojo",
    "serenitymojo/configs/anima.json",
    "serenitymojo/serve/sensenova_backend.mojo",
    "serenitymojo/serve/lens_backend.mojo",
    "serenity-server/crates/server/src/main.rs",
    "serenity-server/crates/server/src/block_profiles.rs",
]

BANNED_RUNTIME_ROOTS = [
    "/home/alex/EriDiffusion",
    "/home/alex/SerenityFlow",
    "/home/alex/SwarmUI",
    "/home/alex/ComfyUI",
    "/home/alex/Lance",
]

ALLOWED_RUNTIME_ROOTS = [
    "/home/alex/.serenity/",
    "/home/alex/.cache/huggingface/",
    "/home/alex/mojodiffusion/",
]

EXPECTED_TOKENS = {
    "serenitymojo/runtime/model_manifest.mojo": [
        'var root = String("/home/alex/.serenity/models/hidream_o1_dev")',
        'var root = String("/home/alex/.serenity/models/ernie_image")',
    ],
    "serenitymojo/configs/qwenimage.json": [
        '"checkpoint": "/home/alex/.serenity/models/checkpoints/qwen-image-2512/transformer"',
        '"vae": "/home/alex/.serenity/models/checkpoints/qwen-image-2512/vae"',
    ],
    "serenitymojo/pipeline/qwenimage_sample_cli.mojo": [
        'comptime QWENIMAGE_DIR = "/home/alex/.serenity/models/checkpoints/qwen-image-2512"',
    ],
    "serenitymojo/pipeline/zimage_generate.mojo": [
        'comptime ZROOT = "/home/alex/.serenity/models/zimage_base"',
    ],
    "serenitymojo/serve/ideogram4_backend.mojo": [
        'comptime COND = "/home/alex/.serenity/models/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors"',
    ],
    "serenity-server/crates/server/src/main.rs": [
        '"runtime_dependency_on_external_repos": false',
        'let root = "/home/alex/.serenity/models/checkpoints/qwen-image-2512";',
        'let root = "/home/alex/.serenity/models/zimage_base";',
        'let root = "/home/alex/.serenity/models/ideogram-4-fp8";',
        'let root = "/home/alex/.serenity/models/hidream_o1_dev";',
    ],
}

RUNNABLE_REQUIRED_LOCAL_PATHS = {
    "zimage transformer": "/home/alex/.serenity/models/zimage_base/transformer",
    "zimage vae": "/home/alex/.serenity/models/zimage_base/vae",
    "qwen transformer": "/home/alex/.serenity/models/checkpoints/qwen-image-2512/transformer",
    "qwen vae": "/home/alex/.serenity/models/checkpoints/qwen-image-2512/vae",
    "ideogram cond": "/home/alex/.serenity/models/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors",
    "ideogram uncond": "/home/alex/.serenity/models/ideogram-4-fp8/unconditional_transformer/diffusion_pytorch_model.safetensors",
    "sdxl unet": "/home/alex/.serenity/models/checkpoints/sdxl_unet_bf16.safetensors",
    "sd3 large": "/home/alex/.serenity/models/checkpoints/sd3.5_large.safetensors",
    "flux1 checkpoint": "/home/alex/.serenity/models/checkpoints/flux1-dev.safetensors",
    "klein9b checkpoint": "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors",
    "anima dit": "/home/alex/.serenity/models/anima/split_files/diffusion_models/anima-base-v1.0.safetensors",
    "sensenova weights": "/home/alex/.serenity/models/sensenova_u1/model.safetensors.index.json",
    "lens transformer": "/home/alex/.serenity/models/microsoft_lens/transformer",
}

DECLARED_OPTIONAL_MODELSTORE_PATHS = {
    "hidream root": "/home/alex/.serenity/models/hidream_o1_dev",
    "ernie root": "/home/alex/.serenity/models/ernie_image",
    "lance video root": "/home/alex/.serenity/models/lance/Lance_3B_Video",
}

STRING_RE = re.compile(r"""(["'])([^"']*/home/alex/[^"']*)\1""")


def uncommented_text(path: Path) -> str:
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.lstrip()
        if stripped.startswith("#") or stripped.startswith("//"):
            continue
        lines.append(raw)
    return "\n".join(lines)


def absolute_string_literals(path: Path) -> list[str]:
    text = uncommented_text(path)
    return [match.group(2) for match in STRING_RE.finditer(text)]


def check_surfaces() -> tuple[list[dict[str, Any]], list[str]]:
    findings: list[dict[str, Any]] = []
    blockers: list[str] = []
    for rel in PRODUCTION_SURFACES:
        path = REPO / rel
        if not path.is_file():
            blockers.append(f"production surface missing: {rel}")
            continue
        for literal in absolute_string_literals(path):
            for root in BANNED_RUNTIME_ROOTS:
                if literal.startswith(root):
                    blockers.append(f"{rel}: runtime string points at banned repo root {root}: {literal}")
                    findings.append({"path": rel, "literal": literal, "banned_root": root})
            if not any(literal.startswith(root) for root in ALLOWED_RUNTIME_ROOTS):
                blockers.append(
                    f"{rel}: runtime string is outside allowed local roots "
                    f"(.serenity, Hugging Face cache, repo): {literal}"
                )
                findings.append({"path": rel, "literal": literal, "allowed_root": False})
    return findings, blockers


def check_expected_tokens() -> list[str]:
    blockers: list[str] = []
    for rel, tokens in EXPECTED_TOKENS.items():
        path = REPO / rel
        text = path.read_text(encoding="utf-8")
        for token in tokens:
            if token not in text:
                blockers.append(f"{rel}: missing expected token {token!r}")
    return blockers


def check_required_paths(paths: dict[str, str]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for label, raw in paths.items():
        path = Path(raw)
        ok = path.exists()
        rows.append({"label": label, "path": raw, "exists": ok})
        if not ok:
            blockers.append(f"missing required local artifact path for {label}: {raw}")
    return rows, blockers


def check_optional_paths(paths: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, raw in paths.items():
        path = Path(raw)
        rows.append({"label": label, "path": raw, "exists": path.exists()})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    findings, blockers = check_surfaces()
    blockers.extend(check_expected_tokens())
    required_paths, path_blockers = check_required_paths(RUNNABLE_REQUIRED_LOCAL_PATHS)
    blockers.extend(path_blockers)
    optional_paths = check_optional_paths(DECLARED_OPTIONAL_MODELSTORE_PATHS)

    report = {
        "schema": "serenity.mojo_inference_weight_sources.v1",
        "accepted_runtime_dependency_on_external_repos": not blockers,
        "production_surface_count": len(PRODUCTION_SURFACES),
        "banned_runtime_roots": BANNED_RUNTIME_ROOTS,
        "allowed_runtime_roots": ALLOWED_RUNTIME_ROOTS,
        "banned_findings": findings,
        "required_paths": required_paths,
        "optional_declared_modelstore_paths": optional_paths,
        "blockers": blockers,
        "notes": [
            "Local .serenity model-store paths are accepted runtime artifact sources.",
            "Pinned Hugging Face cache paths remain allowed only where the production source explicitly uses them for tokenizer/text-encoder snapshots.",
            "Optional declared model-store paths are reported but do not make disabled/quarantined families runnable.",
            "This checker does not prove generation quality, speed, or sampler parity.",
        ],
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif blockers:
        print("Mojo inference weight sources: FAIL")
        for blocker in blockers:
            print(f"  - {blocker}")
    else:
        print("Mojo inference weight sources: PASS")
        print(f"  production surfaces checked: {len(PRODUCTION_SURFACES)}")
        print(f"  required local artifact paths checked: {len(RUNNABLE_REQUIRED_LOCAL_PATHS)}")
        print(f"  optional declared model-store paths checked: {len(DECLARED_OPTIONAL_MODELSTORE_PATHS)}")
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
