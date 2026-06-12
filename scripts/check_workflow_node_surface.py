#!/usr/bin/env python3
"""No-CUDA static guard for Comfy/Swarm workflow node surface.

This checker reads source files only. It validates the constrained daemon
workflow adapter markers and fail-loud unsupported-node behavior. It does not
run CUDA, does not start the daemon, does not execute arbitrary graphs, and
does not claim ComfyUI/SwarmUI graph parity.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[1]

DOC = REPO / "serenitymojo/docs/COMFY_SWARM_WORKFLOW_PARITY_MAP_2026-06-12.md"
DAEMON = REPO / "serenitymojo/serve/serenity_daemon.mojo"
BACKEND = REPO / "serenitymojo/serve/backend.mojo"
ZIMAGE_BACKEND = REPO / "serenitymojo/serve/zimage_backend.mojo"
QWEN_BACKEND = REPO / "serenitymojo/serve/qwenimage_backend.mojo"
IMAGE_IO = REPO / "serenitymojo/serve/image_io.mojo"
MODEL_SCAN = REPO / "serenitymojo/serve/model_scan.mojo"
LEDGER = REPO / "serenitymojo/docs/SWARMUI_PRODUCT_PATH_LEDGER_2026-06-12.md"
ROADMAP = REPO / "serenitymojo/docs/SWARMUI_PARITY_ROADMAP_2026-06-12.md"
WORKFLOW_GRAPH_PRODUCT = REPO / "output/checks/workflow_graph_product_readiness.json"

P0 = "P0"
P1 = "P1"
P2 = "P2"
PASS = "PASS"

SUPPORTED_NODE_TYPES = [
    "CheckpointLoaderSimple",
    "CLIPTextEncode",
    "EmptyLatentImage",
    "KSampler",
    "VAEDecode",
    "SaveImage",
]

UNSUPPORTED_NODE_EXAMPLES = [
    "LoraLoader",
    "LoadImage",
    "VAEEncode",
    "ControlNetApply",
    "IPAdapterApply",
    "ImageUpscaleWithModel",
    "VideoCombine",
]


@dataclass(frozen=True)
class Check:
    ok: bool
    severity: str
    category: str
    label: str
    detail: str
    path: str
    acceptance: str


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def read(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def function_body(text: str, name: str) -> str:
    lines = text.splitlines(keepends=True)
    start = -1
    for idx, line in enumerate(lines):
        if line.startswith(f"def {name}(") or line.startswith(f"def {name}["):
            start = idx
            break
    if start < 0:
        return ""
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        line = lines[idx]
        if line.startswith("def ") or line.startswith("struct ") or line.startswith("trait "):
            end = idx
            break
    return "".join(lines[start:end])


def check_contains(
    path: Path,
    *,
    category: str,
    label: str,
    needles: Iterable[str],
    severity: str,
    acceptance: str,
) -> Check:
    text = read(path)
    if not text:
        return Check(False, severity, category, label, f"missing file: {path}", rel(path), acceptance)
    missing = [needle for needle in needles if needle not in text]
    if missing:
        return Check(
            False,
            severity,
            category,
            label,
            "missing markers: " + ", ".join(repr(item) for item in missing),
            rel(path),
            acceptance,
        )
    return Check(True, PASS, category, label, "required markers present", rel(path), acceptance)


def check_body_contains(
    path: Path,
    function_name: str,
    *,
    category: str,
    label: str,
    needles: Iterable[str],
    severity: str,
    acceptance: str,
) -> Check:
    text = read(path)
    if not text:
        return Check(False, severity, category, label, f"missing file: {path}", rel(path), acceptance)
    body = function_body(text, function_name)
    if not body:
        return Check(
            False,
            severity,
            category,
            label,
            f"missing function body: {function_name}",
            rel(path),
            acceptance,
        )
    missing = [needle for needle in needles if needle not in body]
    if missing:
        return Check(
            False,
            severity,
            category,
            label,
            "missing markers: " + ", ".join(repr(item) for item in missing),
            rel(path),
            acceptance,
        )
    return Check(True, PASS, category, label, "required function markers present", rel(path), acceptance)


def check_unsupported_not_allowlisted() -> Check:
    body = function_body(read(DAEMON), "_apply_workflow_params")
    if not body:
        return Check(
            False,
            P0,
            "workflow",
            "unsupported node examples stay outside allowlist",
            "missing _apply_workflow_params body",
            rel(DAEMON),
            "Advanced node families are not silently accepted by the constrained adapter.",
        )
    accepted = [node for node in UNSUPPORTED_NODE_EXAMPLES if f'type_id == "{node}"' in body]
    if accepted:
        return Check(
            False,
            P0,
            "workflow",
            "unsupported node examples stay outside allowlist",
            "unexpected accepted markers: " + ", ".join(accepted),
            rel(DAEMON),
            "Advanced node families are not silently accepted by the constrained adapter.",
        )
    return Check(
        True,
        PASS,
        "workflow",
        "unsupported node examples stay outside allowlist",
        "advanced examples are not accepted markers",
        rel(DAEMON),
        "Advanced node families are not silently accepted by the constrained adapter.",
    )


def check_supported_nodes() -> list[Check]:
    checks: list[Check] = []
    checks.append(
        check_body_contains(
            DAEMON,
            "_apply_workflow_params",
            category="workflow",
            label="workflow adapter entrypoint",
            needles=[
                "workflow",
                "params",
                "genparams",
                "nodes",
                "Linked `workflow.nodes`/`workflow.edges` bodies are executed",
            ],
            severity=P0,
            acceptance="Daemon exposes flat workflow passthrough plus typed linked-graph execution for supported nodes.",
        )
    )
    checks.append(
        check_body_contains(
            DAEMON,
            "_apply_workflow_graph_ir",
            category="workflow",
            label="typed linked graph IR executor",
            needles=[
                "_workflow_find_input_link",
                "_workflow_add_value",
                "_workflow_require_value_type",
                "edges",
                "MODEL",
                "CLIP",
                "VAE",
                "CONDITIONING",
                "LATENT",
                "IMAGE",
                "unresolved or cyclic typed links",
            ],
            severity=P0,
            acceptance="Supported Comfy/Swarm t2i graphs use typed handles and topological execution instead of field-only flattening.",
        )
    )
    checks.append(
        check_body_contains(
            DAEMON,
            "_apply_workflow_graph_ir",
            category="workflow",
            label="supported Comfy-like node markers",
            needles=SUPPORTED_NODE_TYPES,
            severity=P0,
            acceptance="Only the documented constrained t2i node markers are accepted.",
        )
    )
    checks.append(
        check_body_contains(
            DAEMON,
            "_apply_workflow_graph_ir",
            category="workflow",
            label="supported node field mappings",
            needles=[
                "ckpt_name",
                "model",
                "text",
                "negative",
                "width",
                "height",
                "batch_size",
                "images",
                "steps",
                "seed",
                "cfg",
                "sampler_name",
                "sampler",
                "scheduler",
                "denoise",
                "creativity",
            ],
            severity=P0,
            acceptance="Accepted nodes map explicit fields into flat genparams.",
        )
    )
    checks.append(
        check_contains(
            BACKEND,
            category="workflow",
            label="flat JobParams surface",
            needles=[
                "var model: String",
                "var prompt: String",
                "var negative: String",
                "var width: Int",
                "var height: Int",
                "var steps: Int",
                "var seed: Int",
                "var cfg: Float64",
                "var init_image: String",
                "var creativity: Float64",
                "var loras: List[LoraSpec]",
                "var params_json: String",
            ],
            severity=P1,
            acceptance="Workflow adapter feeds the existing flat product job contract.",
        )
    )
    return checks


def check_fail_loud() -> list[Check]:
    return [
        check_body_contains(
            DAEMON,
            "_apply_workflow_params",
            category="failure",
            label="unsupported graph shapes fail loud",
            needles=[
                "[501] workflow graph body must be an object",
                "[501] workflow graph body needs nodes or params/genparams",
                "[501] workflow graph node must be an object",
                "[501] unsupported workflow graph format: missing type_id",
                "[501] unsupported workflow graph node type: ",
                "[501] workflow graph did not contain a prompt node",
            ],
            severity=P0,
            acceptance="Unsupported graph requests name the unsupported feature instead of silently no-oping.",
        ),
        check_contains(
            DAEMON,
            category="failure",
            label="typed workflow graph validation fails loud",
            needles=[
                "[501] workflow graph body needs edges for typed execution",
                "[501] workflow graph input ",
                "[501] workflow graph has unresolved or cyclic typed links",
            ],
            severity=P0,
            acceptance="Bad typed links and cyclic linked graphs fail before enqueue.",
        ),
        check_contains(
            DAEMON,
            category="failure",
            label="501 sentinel converted to HTTP 501",
            needles=[
                'msg.startswith("[501] ")',
                "return error_response(501",
                "byte_substr(msg, 6, msg.byte_length())",
            ],
            severity=P0,
            acceptance="The daemon preserves unsupported-workflow semantics at the HTTP boundary.",
        ),
        check_unsupported_not_allowlisted(),
    ]


def check_family_surfaces() -> list[Check]:
    return [
        check_contains(
            MODEL_SCAN,
            category="checkpoint-lora",
            label="checkpoint and LoRA disk scan",
            needles=[
                "scan_checkpoints",
                "scan_loras",
                "CHECKPOINTS_DIR",
                "LORAS_DIR",
                "qwen-image",
                "zimage",
                "ltx2",
            ],
            severity=P1,
            acceptance="Model/LoRA browser foundations exist as disk scans, not graph loader parity.",
        ),
        check_contains(
            DAEMON,
            category="lora",
            label="flat LoRA request and prompt tag extraction",
            needles=[
                "'lora' must be an array of {name, weight}",
                "prompt_syntax.loras",
                "lora_tags",
                "conditioning_weights_applied",
            ],
            severity=P1,
            acceptance="Flat LoRA metadata is parsed and recorded while graph LoRA nodes remain unsupported.",
        ),
        check_contains(
            ZIMAGE_BACKEND,
            category="lora",
            label="Z-Image LoRA admission is explicit",
            needles=[
                "merge_zimage_lora_sets_for_inference",
                "rank_concat_scaled_b",
                "load_zimage_lora_main_only_comfy",
                "load_zimage_lora_main_only_resume",
            ],
            severity=P1,
            acceptance="Z-Image LoRA support is explicit and bounded to proven runtime formats.",
        ),
        check_contains(
            QWEN_BACKEND,
            category="lora",
            label="Qwen LoRA and img2img reject loudly",
            needles=[
                "LoRA is not supported for Qwen-Image yet",
                "img2img is not supported for Qwen-Image yet",
            ],
            severity=P1,
            acceptance="Qwen unsupported families fail loud instead of silently no-oping.",
        ),
        check_contains(
            IMAGE_IO,
            category="image",
            label="flat init-image decode formats",
            needles=[
                "decode_png_bytes",
                "decode_jpeg_bytes",
                "decode_webp_bytes",
                "init image format not supported",
            ],
            severity=P1,
            acceptance="Flat img2img path can decode common local image formats without Python.",
        ),
        check_contains(
            ZIMAGE_BACKEND,
            category="image",
            label="Z-Image img2img resize and VAE encode",
            needles=[
                "resize_bilinear",
                "_encode_init_latent",
                "ZImageVaeEncoder",
                "creativity",
                "start step",
            ],
            severity=P1,
            acceptance="Z-Image flat img2img has a Mojo decode/resize/encode path.",
        ),
        check_contains(
            DAEMON,
            category="video",
            label="video has bounded daemon smoke contract",
            needles=[
                "/v1/video",
                "LTX2_VIDEO_SMOKE_RUNNER",
                "_ltx2_staged_smoke_video_result",
                "ltx2_t2v_av_stage2_dev_smoke.mp4",
                "frame_count",
                "duration",
                "muxing",
                "accepted_video_parity",
            ],
            severity=PASS,
            acceptance="Video workflow nodes remain outside arbitrary graph parity, but the daemon has a bounded product smoke runner surface.",
        ),
    ]


def check_workflow_graph_product_report() -> Check:
    report = read_json(WORKFLOW_GRAPH_PRODUCT)
    if not report:
        return Check(
            False,
            P1,
            "workflow",
            "typed workflow graph product smoke",
            f"missing report: {rel(WORKFLOW_GRAPH_PRODUCT)}",
            rel(WORKFLOW_GRAPH_PRODUCT),
            "A daemon product smoke posts a linked graph, emits PNG genparams, and verifies typed-link 501 failures.",
        )
    blockers = report.get("blockers")
    if report.get("ready") is not True:
        return Check(
            False,
            P1,
            "workflow",
            "typed workflow graph product smoke",
            "report not ready: " + json.dumps(blockers),
            rel(WORKFLOW_GRAPH_PRODUCT),
            "A daemon product smoke posts a linked graph, emits PNG genparams, and verifies typed-link 501 failures.",
        )
    job = report.get("job")
    png = report.get("png")
    if not isinstance(job, dict) or not isinstance(png, dict):
        return Check(
            False,
            P1,
            "workflow",
            "typed workflow graph product smoke",
            "report missing job/png evidence",
            rel(WORKFLOW_GRAPH_PRODUCT),
            "A daemon product smoke posts a linked graph, emits PNG genparams, and verifies typed-link 501 failures.",
        )
    return Check(
        True,
        PASS,
        "workflow",
        "typed workflow graph product smoke",
        f"linked graph completed {job.get('id')} and wrote {png.get('path')}; unsupported and wrong-type links returned HTTP 501",
        rel(WORKFLOW_GRAPH_PRODUCT),
        "A daemon product smoke posts a linked graph, emits PNG genparams, and verifies typed-link 501 failures.",
    )


def check_docs() -> list[Check]:
    return [
        check_contains(
            DOC,
            category="docs",
            label="workflow parity map exists",
            needles=[
                "not an arbitrary ComfyUI graph executor",
                "Supported Adapter Markers",
                "Node Surface Table",
                "Exact Current Blockers",
            ],
            severity=P0,
            acceptance="The workflow/node surface limits are documented before readiness claims.",
        ),
        check_contains(
            LEDGER,
            category="docs",
            label="ledger records constrained workflow status",
            needles=[
                "typed linked workflow",
                "CheckpointLoaderSimple",
                "unsupported workflow graph node type",
            ],
            severity=P1,
            acceptance="Existing product ledger names the constrained adapter and unsupported-node behavior.",
        ),
        check_contains(
            ROADMAP,
            category="docs",
            label="roadmap keeps graph gap visible",
            needles=[
                "advanced Comfy/Swarm node families beyond the typed t2i graph",
                "Qwen full daemon generation was not run",
                "Video generation is still not accepted",
                "ltx2_staged_dev_smoke",
            ],
            severity=P1,
            acceptance="Roadmap does not present graph/Qwen/video runtime parity as complete.",
        ),
    ]


def collect_checks() -> list[Check]:
    checks: list[Check] = []
    checks.extend(check_docs())
    checks.extend(check_supported_nodes())
    checks.extend(check_fail_loud())
    checks.extend(check_family_surfaces())
    checks.append(check_workflow_graph_product_report())
    return checks


def build_report(checks: list[Check]) -> dict[str, object]:
    p0 = [check for check in checks if not check.ok and check.severity == P0]
    p1 = [check for check in checks if not check.ok and check.severity == P1]
    p2 = [check for check in checks if not check.ok and check.severity == P2]
    return {
        "schema": "serenity.workflow_node_surface_readiness.v1",
        "scope": "static source markers plus the latest typed workflow graph product smoke report",
        "constrained_workflow_adapter_ready": not p0,
        "arbitrary_comfy_swarm_graph_execution_ready": False,
        "supported_t2i_graph_execution_ready": not p0 and not p1,
        "full_graph_parity_claim": "blocked_for_advanced_node_families",
        "supported_adapter_node_types": SUPPORTED_NODE_TYPES,
        "unsupported_node_examples_expected_501": UNSUPPORTED_NODE_EXAMPLES,
        "summary": {
            "checks": len(checks),
            "passed": sum(1 for check in checks if check.ok),
            "p0_blockers": len(p0),
            "p1_blockers": len(p1),
            "p2_blockers": len(p2),
        },
        "blockers": {
            "p0": [asdict(check) for check in p0],
            "p1": [asdict(check) for check in p1],
            "p2": [asdict(check) for check in p2],
        },
        "checks": [asdict(check) for check in checks],
        "next_command": (
            "python3 scripts/check_workflow_node_surface.py "
            "--write-readiness output/checks/workflow_node_surface_readiness.json"
        ),
    }


def print_text_report(report: dict[str, object]) -> None:
    checks = report["checks"]
    assert isinstance(checks, list)
    for item in checks:
        assert isinstance(item, dict)
        status = "PASS" if item["ok"] else item["severity"]
        print(
            "[workflow-node-surface] "
            f"{status} {item['category']}: {item['label']} "
            f"({item['path']}) - {item['detail']}"
        )
    summary = report["summary"]
    assert isinstance(summary, dict)
    print(
        "[workflow-node-surface] summary "
        f"checks={summary['checks']} passed={summary['passed']} "
        f"p0={summary['p0_blockers']} p1={summary['p1_blockers']} "
        f"p2={summary['p2_blockers']}"
    )
    if report["constrained_workflow_adapter_ready"]:
        print("[workflow-node-surface] constrained adapter markers: READY")
    else:
        print("[workflow-node-surface] constrained adapter markers: BLOCKED")
    print("[workflow-node-surface] arbitrary Comfy/Swarm graph parity: BLOCKED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="exit 2 if P0 marker blockers remain")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON report")
    parser.add_argument("--write-readiness", type=Path, help="write machine-readable readiness JSON")
    args = parser.parse_args()

    report = build_report(collect_checks())

    if args.write_readiness is not None:
        args.write_readiness.parent.mkdir(parents=True, exist_ok=True)
        args.write_readiness.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[workflow-node-surface] wrote readiness report: {args.write_readiness}")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text_report(report)

    if args.strict and not report["constrained_workflow_adapter_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
