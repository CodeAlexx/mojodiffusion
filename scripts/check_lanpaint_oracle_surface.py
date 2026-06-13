#!/usr/bin/env python3
"""Check the local LanPaint/Comfy inpaint oracle surface.

This is a no-heavy-model contract checker. It reads the local LanPaint Python
nodes, representative LanPaint workflow exports, SerenityFlow's
SetLatentNoiseMask node, and the current Mojo boundary. The goal is to preserve
the actual oracle semantics before implementing real mask-aware daemon runtime.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LANPAINT_NODES = Path("/home/alex/LanPaint/src/LanPaint/nodes.py")
SERENITYFLOW_LATENT = Path("/home/alex/serenityflow-v2/serenityflow/nodes/latent.py")
WORKFLOW_GRAPH = REPO / "serenitymojo/serve/workflow_graph.mojo"
BACKEND = REPO / "serenitymojo/serve/backend.mojo"
IMAGE_IO = REPO / "serenitymojo/serve/image_io.mojo"
INPAINT_MOJO = REPO / "serenitymojo/sampling/inpaint.mojo"
INPAINT_PARITY = REPO / "serenitymojo/sampling/parity/inpaint_parity.mojo"
NODE_SURFACE = REPO / "scripts/check_workflow_node_surface.py"
DEFAULT_REPORT = REPO / "output/checks/lanpaint_oracle_surface.json"

WORKFLOWS = {
    "zimage_inpaint": {
        "path": Path("/home/alex/LanPaint/example_workflows/Z_image_Inpaint.json"),
        "required_types": {
            "LanPaint_KSampler": 1,
            "SetLatentNoiseMask": 1,
            "LanPaint_MaskBlend": 1,
            "MaskToImage": 1,
            "ImageToMask": 1,
            "LoadImage": 1,
            "VAEEncode": 2,
        },
        "lanpaint_node": "LanPaint_KSampler",
        "lanpaint_widgets": {
            "steps": 9,
            "cfg": 1,
            "sampler": "euler",
            "scheduler": "simple",
            "denoise": 1,
            "lanpaint_num_steps": 5,
            "prompt_mode": "Image First",
            "inpainting_mode": "image",
        },
        "mask_blend_overlap": 9,
        "image_to_mask_channel": "red",
    },
    "qwen_image_inpaint": {
        "path": Path("/home/alex/LanPaint/example_workflows/Qwen_Image_Inpaint.json"),
        "required_types": {
            "LanPaint_KSampler": 1,
            "SetLatentNoiseMask": 1,
            "LanPaint_MaskBlend": 1,
            "MaskToImage": 1,
            "ImageToMask": 1,
            "LoadImage": 1,
            "VAEEncode": 2,
        },
        "lanpaint_node": "LanPaint_KSampler",
        "lanpaint_widgets": {
            "steps": 20,
            "cfg": 4,
            "sampler": "euler",
            "scheduler": "simple",
            "denoise": 1,
            "lanpaint_num_steps": 5,
            "prompt_mode": "Image First",
            "inpainting_mode": "image",
        },
        "mask_blend_overlap": 9,
        "image_to_mask_channel": "red",
    },
    "flux2_klein_inpainting": {
        "path": Path("/home/alex/LanPaint/example_workflows/Flux2_Klein_inpainting.json"),
        "required_types": {
            "LanPaint_SamplerCustomAdvanced": 1,
            "LanPaint_MaskBlend": 1,
            "MaskToImage": 2,
            "ImageToMask": 2,
            "SetLatentNoiseMask": 1,
            "ReferenceLatent": 2,
            "VAEEncode": 3,
        },
        "lanpaint_node": "LanPaint_SamplerCustomAdvanced",
        "lanpaint_widgets": {
            "lanpaint_num_steps": 2,
            "lanpaint_lambda": 8,
            "lanpaint_step_size": 0.2,
            "lanpaint_beta": 1,
            "lanpaint_friction": 15,
            "prompt_mode": "Image First",
            "early_stop": 1,
            "inner_threshold": 0,
            "inner_patience": 1,
        },
        "mask_blend_overlap": 9,
        "image_to_mask_channel": "red",
        "requires_subgraph_expansion": True,
    },
}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def require(condition: bool, message: str, blockers: list[str]) -> None:
    if not condition:
        blockers.append(message)


def node_type(node: dict[str, Any]) -> str:
    return str(node.get("type") or node.get("class_type") or node.get("type_id") or "")


def walk_nodes(value: Any) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if node_type(value):
            nodes.append(value)
        for child in value.values():
            nodes.extend(walk_nodes(child))
    elif isinstance(value, list):
        for child in value:
            nodes.extend(walk_nodes(child))
    return nodes


def first_node(nodes: list[dict[str, Any]], wanted: str) -> dict[str, Any]:
    for node in nodes:
        if node_type(node) == wanted:
            return node
    return {}


def lanpaint_ksampler_summary(widgets: list[Any]) -> dict[str, Any]:
    return {
        "steps": widgets[2] if len(widgets) > 2 else None,
        "cfg": widgets[3] if len(widgets) > 3 else None,
        "sampler": widgets[4] if len(widgets) > 4 else None,
        "scheduler": widgets[5] if len(widgets) > 5 else None,
        "denoise": widgets[6] if len(widgets) > 6 else None,
        "lanpaint_num_steps": widgets[7] if len(widgets) > 7 else None,
        "prompt_mode": widgets[8] if len(widgets) > 8 else None,
        "inpainting_mode": "video"
        if any(str(item) == "🎬 Video Inpainting" for item in widgets)
        else "image",
    }


def lanpaint_custom_advanced_summary(widgets: list[Any]) -> dict[str, Any]:
    return {
        "lanpaint_num_steps": widgets[0] if len(widgets) > 0 else None,
        "lanpaint_lambda": widgets[1] if len(widgets) > 1 else None,
        "lanpaint_step_size": widgets[2] if len(widgets) > 2 else None,
        "lanpaint_beta": widgets[3] if len(widgets) > 3 else None,
        "lanpaint_friction": widgets[4] if len(widgets) > 4 else None,
        "prompt_mode": widgets[5] if len(widgets) > 5 else None,
        "early_stop": widgets[6] if len(widgets) > 6 else None,
        "inner_threshold": widgets[8] if len(widgets) > 8 else None,
        "inner_patience": widgets[9] if len(widgets) > 9 else None,
    }


def validate_workflow(case_name: str, spec: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    path = Path(spec["path"])
    evidence: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return evidence, [f"missing workflow: {path}"]

    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = walk_nodes(data)
    counts = Counter(node_type(node) for node in nodes if node_type(node))
    evidence["node_type_counts"] = dict(sorted(counts.items()))
    evidence["node_count_seen"] = len(nodes)

    for typ, minimum in dict(spec["required_types"]).items():
        require(counts.get(typ, 0) >= int(minimum), f"{case_name}: missing node type {typ}>={minimum}", blockers)

    lanpaint_node = first_node(nodes, str(spec["lanpaint_node"]))
    widgets = lanpaint_node.get("widgets_values") if isinstance(lanpaint_node, dict) else []
    widgets = widgets if isinstance(widgets, list) else []
    if spec["lanpaint_node"] == "LanPaint_KSampler":
        summary = lanpaint_ksampler_summary(widgets)
    else:
        summary = lanpaint_custom_advanced_summary(widgets)
    evidence["lanpaint_widgets"] = summary
    for key, expected in dict(spec["lanpaint_widgets"]).items():
        require(summary.get(key) == expected, f"{case_name}: {key}={expected!r}", blockers)

    blend = first_node(nodes, "LanPaint_MaskBlend")
    blend_widgets = blend.get("widgets_values") if isinstance(blend, dict) else []
    overlap = blend_widgets[0] if isinstance(blend_widgets, list) and blend_widgets else None
    evidence["mask_blend_overlap"] = overlap
    require(overlap == spec["mask_blend_overlap"], f"{case_name}: mask_blend_overlap={spec['mask_blend_overlap']}", blockers)

    image_to_mask = first_node(nodes, "ImageToMask")
    mask_widgets = image_to_mask.get("widgets_values") if isinstance(image_to_mask, dict) else []
    channel = mask_widgets[0] if isinstance(mask_widgets, list) and mask_widgets else None
    evidence["image_to_mask_channel"] = channel
    require(channel == spec["image_to_mask_channel"], f"{case_name}: ImageToMask channel={spec['image_to_mask_channel']!r}", blockers)
    evidence["requires_subgraph_expansion"] = bool(spec.get("requires_subgraph_expansion", False))
    return evidence, blockers


def contains_all(path: Path, needles: list[str]) -> tuple[dict[str, Any], list[str]]:
    text = read_text(path)
    blockers: list[str] = []
    missing = [needle for needle in needles if needle not in text]
    evidence = {"path": str(path), "exists": path.exists(), "missing": missing}
    for needle in missing:
        blockers.append(f"{path}: missing marker {needle!r}")
    return evidence, blockers


def run() -> dict[str, Any]:
    blockers: list[str] = []
    workflow_evidence: dict[str, Any] = {}
    for case_name, spec in WORKFLOWS.items():
        evidence, case_blockers = validate_workflow(case_name, spec)
        workflow_evidence[case_name] = evidence
        blockers.extend(case_blockers)

    lanpaint_py, lanpaint_blockers = contains_all(
        LANPAINT_NODES,
        [
            "class LanPaint_KSampler",
            "class LanPaint_KSamplerAdvanced",
            "class LanPaint_SamplerCustomAdvanced",
            "class MaskBlend",
            "denoise_mask = (denoise_mask > 0.5).float()",
            "latent_mask = 1 - denoise_mask",
            "model.LanPaint_StepSize = 0.2",
            "model.LanPaint_Lambda = 16.0",
            "model.LanPaint_Beta = 1.",
            "model.LanPaint_Friction = 15.",
            "model.LanPaint_cfg_BIG = cfg",
            "model.LanPaint_cfg_BIG = 0*cfg - 0.5",
            "noise_mask = latent[\"noise_mask\"]",
            "denoise_mask=noise_mask",
            "torch.nn.functional.max_pool2d",
            "torch.nn.functional.conv2d",
            "image1 * (1 - mask[...,None]) + image2 * mask[...,None]",
            "\"LanPaint_KSampler\": LanPaint_KSampler",
            "\"LanPaint_SamplerCustomAdvanced\" : LanPaint_SamplerCustomAdvanced",
            "\"LanPaint_MaskBlend\": MaskBlend",
        ],
    )
    blockers.extend(lanpaint_blockers)

    serenityflow_latent, serenityflow_blockers = contains_all(
        SERENITYFLOW_LATENT,
        [
            "\"SetLatentNoiseMask\"",
            "def set_latent_noise_mask",
            "s[\"noise_mask\"] = mask",
            "return (s,)",
        ],
    )
    blockers.extend(serenityflow_blockers)

    mojo_inpaint, mojo_inpaint_blockers = contains_all(
        INPAINT_MOJO,
        [
            "def mask_blend",
            "mask == 1.0",
            "mask == 0.0",
            "def lanpaint_overdamped_step",
            "AGENT-DEFAULT",
            "score: Tensor",
        ],
    )
    blockers.extend(mojo_inpaint_blockers)

    mojo_boundary, mojo_boundary_blockers = contains_all(
        WORKFLOW_GRAPH,
        [
            'or type_id == "SetLatentNoiseMask"',
            'or type_id == "LanPaint_KSampler"',
            'or type_id == "LanPaint_SamplerCustomAdvanced"',
            'or type_id == "LanPaint_MaskBlend"',
            "comfy_ui_canvas_to_typed_graph",
            "_workflow_source_meta",
            "load_image_mask",
            "_workflow_copy_lanpaint_sampler_fields",
            'or type_id == "SamplerCustomAdvanced"',
            '_set_if_missing(obj, String("mask_image")',
            'raise Error(String("[501] unsupported workflow graph node type: ") + type_id)',
        ],
    )
    blockers.extend(mojo_boundary_blockers)

    mask_io, mask_io_blockers = contains_all(
        IMAGE_IO,
        [
            "decode_comfy_mask",
            "load_image_mask",
            "ImageToMask is raw channel selection",
            "resize_mask_bilinear",
            "resize_mask_nearest_exact",
            "binarize_lanpaint_denoise_mask",
            "load_comfy_latent_preserve_mask",
            "load_lanpaint_latent_preserve_mask",
        ],
    )
    blockers.extend(mask_io_blockers)

    node_surface, node_surface_blockers = contains_all(
        NODE_SURFACE,
        [
            '"LanPaint_KSampler"',
            '"LanPaint_SamplerCustomAdvanced"',
            '"LanPaint_MaskBlend"',
            '"ImageToMask"',
            '"MaskToImage"',
            "LanPaint canvas daemon smoke",
            "lanpaint_num_steps",
        ],
    )
    blockers.extend(node_surface_blockers)

    backend_boundary, backend_blockers = contains_all(
        BACKEND,
        [
            "reject_unsupported_mask_image_params",
            "reject_unsupported_lanpaint_params",
            "Comfy SetLatentNoiseMask/inpaint mask conditioning is not supported",
            "LanPaint inpaint sampler/blend semantics are not supported",
        ],
    )
    blockers.extend(backend_blockers)

    parity_gate, parity_blockers = contains_all(
        INPAINT_PARITY,
        [
            "mask-blend + overdamped LanPaint step",
            "PASS: inpaint mask-blend + LanPaint overdamped step parity",
            "--bitrot",
        ],
    )
    blockers.extend(parity_blockers)

    return {
        "schema": "serenity.lanpaint_oracle_surface.v1",
        "ready": not blockers,
        "blockers": blockers,
        "oracle": {
            "lanpaint_nodes": lanpaint_py,
            "serenityflow_set_latent_noise_mask": serenityflow_latent,
            "workflows": workflow_evidence,
        },
        "mojo": {
            "inpaint_math_substrate": mojo_inpaint,
            "inpaint_parity_gate": parity_gate,
            "mask_io_boundary": mask_io,
            "workflow_fail_loud_boundary": mojo_boundary,
            "node_surface_lanpaint_plumbing": node_surface,
            "backend_fail_loud_boundary": backend_boundary,
        },
        "non_claims": [
            "Z-Image consumes plain SetLatentNoiseMask for img2img; full LanPaint sampler/blend semantics are still fenced.",
            "The current Mojo parity gate covers weight-free mask blend and one supplied-score overdamped step only.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run()
    args.write_report.parent.mkdir(parents=True, exist_ok=True)
    args.write_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[lanpaint-oracle-surface] wrote report: {args.write_report}")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["ready"] else "FAIL"
        print(f"[lanpaint-oracle-surface] {status} workflows={len(WORKFLOWS)}")
        for blocker in report["blockers"]:
            print(f"[lanpaint-oracle-surface] blocker: {blocker}")
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
