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
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[1]

DOC = REPO / "serenitymojo/docs/COMFY_SWARM_WORKFLOW_PARITY_MAP_2026-06-12.md"
DAEMON = REPO / "serenitymojo/serve/serenity_daemon.mojo"
VIDEO_API = REPO / "serenitymojo/serve/video_api.mojo"
WORKFLOW_GRAPH = REPO / "serenitymojo/serve/workflow_graph.mojo"
BACKEND = REPO / "serenitymojo/serve/backend.mojo"
ZIMAGE_BACKEND = REPO / "serenitymojo/serve/zimage_backend.mojo"
IDEOGRAM4_BACKEND = REPO / "serenitymojo/serve/ideogram4_backend.mojo"
QWEN_BACKEND = REPO / "serenitymojo/serve/qwenimage_backend.mojo"
KLEIN_BACKEND = REPO / "serenitymojo/serve/klein_backend.mojo"
EXTERNAL_COMMAND = REPO / "serenitymojo/serve/external_command.mojo"
DISPATCH_BACKEND = REPO / "serenitymojo/serve/dispatch_backend.mojo"
WORKER = REPO / "serenitymojo/serve/worker.mojo"
IPC_CODEC = REPO / "serenitymojo/serve/ipc_codec.mojo"
IMAGE_IO = REPO / "serenitymojo/serve/image_io.mojo"
MODEL_SCAN = REPO / "serenitymojo/serve/model_scan.mojo"
SAMPLER_REGISTRY = REPO / "serenitymojo/sampling/sampler_registry.mojo"
KLEIN_REFERENCE_BRIDGE = REPO / "serenitymojo/sampling/klein_reference_latent_bridge.mojo"
KLEIN_REFERENCE_BRIDGE_SMOKE = REPO / "serenitymojo/sampling/parity/klein_reference_latent_bridge_smoke.mojo"
KLEIN_SAMPLE_CLI = REPO / "serenitymojo/sampling/klein_sample_cli.mojo"
KLEIN_STACK_LORA = REPO / "serenitymojo/models/klein/klein_stack_lora.mojo"
KLEIN_REFERENCE_DAEMON_SMOKE_RUNNER = REPO / "scripts/check_klein_reference_daemon_smoke.py"
KLEIN_LORA_DAEMON_SMOKE_RUNNER = REPO / "scripts/check_klein_lora_daemon_smoke.py"
KLEIN_LORA_REFERENCE_DAEMON_SMOKE_RUNNER = REPO / "scripts/check_klein_lora_reference_daemon_smoke.py"
VISUAL_HEALTH_HELPER = REPO / "scripts/visual_health.py"
KLEIN_REAL_IMAGE_HEALTH_RUNNER = REPO / "scripts/check_klein_real_image_health.py"
LANPAINT_ORACLE_SURFACE_RUNNER = REPO / "scripts/check_lanpaint_oracle_surface.py"
LANPAINT_CANVAS_DAEMON_SMOKE_RUNNER = REPO / "scripts/check_lanpaint_canvas_daemon_smoke.py"
PIXI = REPO / "pixi.toml"
LEDGER = REPO / "serenitymojo/docs/SWARMUI_PRODUCT_PATH_LEDGER_2026-06-12.md"
ROADMAP = REPO / "serenitymojo/docs/SWARMUI_PARITY_ROADMAP_2026-06-12.md"
WORKFLOW_GRAPH_PRODUCT = REPO / "output/checks/workflow_graph_product_readiness.json"
KLEIN4B_REFERENCE_DAEMON_SMOKE = REPO / "output/checks/klein4b_reference_edit_daemon_smoke.json"
KLEIN9B_REFERENCE_DAEMON_SMOKE = REPO / "output/checks/klein9b_reference_edit_daemon_smoke.json"
KLEIN9B_LORA_DAEMON_SMOKE = REPO / "output/checks/klein9b_lora_daemon_smoke.json"
KLEIN9B_LORA_REFERENCE_DAEMON_SMOKE = REPO / "output/checks/klein9b_lora_reference_edit_daemon_smoke.json"
KLEIN_REAL_IMAGE_HEALTH = REPO / "output/checks/klein_real_image_health.json"
LANPAINT_ORACLE_SURFACE = REPO / "output/checks/lanpaint_oracle_surface.json"
LANPAINT_CANVAS_DAEMON_SMOKE = REPO / "output/checks/lanpaint_canvas_daemon_smoke.json"
SERENITYFLOW = Path("/home/alex/serenityflow-v2/serenityflow")
SERENITYFLOW_LATENT_NODES = SERENITYFLOW / "nodes/latent.py"
SERENITYFLOW_BASIC_CONDITIONING_NODES = SERENITYFLOW / "nodes/conditioning.py"
SERENITYFLOW_CONDITIONING_NODES = SERENITYFLOW / "nodes/conditioning_extra.py"
SERENITYFLOW_SAMPLING = SERENITYFLOW / "bridge/sampling.py"
COMFY_NODES = Path("/home/alex/SwarmUI/dlbackend/ComfyUI/nodes.py")

P0 = "P0"
P1 = "P1"
P2 = "P2"
PASS = "PASS"
LANPAINT_AREA_RESIZE_ROLE = "LanPaint ImageScale(area) -> MaskBlend.image1 base/original image resize"

SUPPORTED_NODE_TYPES = [
    "CheckpointLoaderSimple",
    "UNETLoader",
    "DiffusionModelLoader",
    "LoraLoader",
    "LoraLoaderModelOnly",
    "ZImageLoraModelOnly",
    "CLIPLoader",
    "DualCLIPLoader",
    "TripleCLIPLoader",
    "VAELoader",
    "CLIPTextEncode",
    "CLIPTextEncodeFlux",
    "TextEncodeQwenImageEdit",
    "TextEncodeQwenImageEditPlus",
    "ConditioningZeroOut",
    "ConditioningSetMask",
    "Reroute",
    "SetNode",
    "GetNode",
    "LoadImage",
    "ImageToMask",
    "MaskToImage",
    "EmptyLatentImage",
    "EmptySD3LatentImage",
    "EmptyFlux2LatentImage",
    "VAEEncode",
    "SetLatentNoiseMask",
    "GetImageSize",
    "ImageScale",
    "ImageScaleToTotalPixels",
    "ImagePadForOutpaint",
    "ThresholdMask",
    "InpaintModelConditioning",
    "ReferenceLatent",
    "6007e698-2ebd-4917-84d8-299b35d7b7ab",
    "f07d2d08-2bc5-4dd8-a9f0-f2347c6b5cca",
    "ModelSamplingAuraFlow",
    "ModelSamplingSD3",
    "DifferentialDiffusion",
    "KSampler",
    "LanPaint_KSampler",
    "LanPaint_KSamplerAdvanced",
    "CFGGuider",
    "BasicGuider",
    "FluxGuidance",
    "Flux2Scheduler",
    "BasicScheduler",
    "RandomNoise",
    "KSamplerSelect",
    "SamplerCustomAdvanced",
    "LanPaint_SamplerCustomAdvanced",
    "LanPaint_MaskBlend",
    "ComfySwitchNode",
    "VAEDecode",
    "SaveImage",
    "PreviewImage",
    "MarkdownNote",
    "Note",
    "PrimitiveInt",
    "PrimitiveFloat",
    "PrimitiveString",
    "PrimitiveStringMultiline",
    "PrimitiveBoolean",
    "PrimitiveNode",
    "INTConstant",
    "FloatConstant",
    "StringConstant",
    "StringConstantMultiline",
    "BOOLConstant",
    "SeedNode",
    "easy int",
    "easy float",
    "easy string",
]

UNSUPPORTED_NODE_EXAMPLES = [
    "LoraLoaderStack",
    "ControlNetApply",
    "IPAdapterApply",
    "ImageUpscaleWithModel",
    "VideoCombine",
    "LanPaint_SamplerCustom",
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


def check_not_contains(
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
    present = [needle for needle in needles if needle in text]
    if present:
        return Check(
            False,
            severity,
            category,
            label,
            "forbidden markers present: " + ", ".join(repr(item) for item in present),
            rel(path),
            acceptance,
        )
    return Check(True, PASS, category, label, "forbidden markers absent", rel(path), acceptance)


def check_unsupported_not_allowlisted() -> Check:
    text = read(WORKFLOW_GRAPH)
    body = function_body(text, "apply_workflow_params") + function_body(text, "apply_typed_workflow_graph")
    if not body:
        return Check(
            False,
            P0,
            "workflow",
            "unsupported node examples stay outside allowlist",
            "missing workflow graph adapter bodies",
            rel(WORKFLOW_GRAPH),
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
            rel(WORKFLOW_GRAPH),
            "Advanced node families are not silently accepted by the constrained adapter.",
        )
    return Check(
        True,
        PASS,
        "workflow",
        "unsupported node examples stay outside allowlist",
        "advanced examples are not accepted markers",
        rel(WORKFLOW_GRAPH),
        "Advanced node families are not silently accepted by the constrained adapter.",
    )


def check_supported_nodes() -> list[Check]:
    checks: list[Check] = []
    checks.append(
        check_body_contains(
            DAEMON,
            "parse_generate",
            category="workflow",
            label="daemon workflow executor entrypoint",
            needles=[
                "apply_workflow_graph_params",
            ],
            severity=P0,
            acceptance="Daemon routes /v1/generate through the workflow graph module before JobParams validation.",
        )
    )
    checks.append(
        check_contains(
            DAEMON,
            category="workflow",
            label="daemon imports workflow graph module",
            needles=[
                "from serenitymojo.serve.workflow_graph import apply_workflow_params as apply_workflow_graph_params",
            ],
            severity=P0,
            acceptance="Workflow execution is a backend module contract, not daemon-local flat parsing.",
        )
    )
    checks.append(
        check_not_contains(
            DAEMON,
            category="workflow",
            label="daemon has no duplicate workflow executor",
            needles=[
                "def _apply_workflow_params",
                "def _apply_workflow_graph_ir",
                "def _apply_ideogram4_comfy_ui_export",
                "struct WorkflowLink",
            ],
            severity=P0,
            acceptance="The daemon cannot drift from the workflow_graph module by carrying a second graph executor.",
        )
    )
    checks.append(
        check_body_contains(
            DAEMON,
            "parse_generate",
            category="workflow",
            label="Ideogram4 structured JSON prompt bypass",
            needles=[
                'sampler_backend == "ideogram4"',
                "_normalize_ideogram4_structured_prompt(obj)",
                "_looks_like_ideogram4_structured_prompt(prompt_raw)",
                "prompt_syntax.resolved = prompt_raw.copy()",
            ],
            severity=P0,
            acceptance="Ideogram4 structured JSON captions, including bbox layout prompts, reach the backend without generic prompt-syntax rewriting.",
        )
    )
    checks.append(
        check_body_contains(
            DAEMON,
            "_normalize_ideogram4_structured_prompt",
            category="workflow",
            label="Ideogram4 prompt_json normalizer",
            needles=[
                "prompt_json",
                "Bounding boxes stay inside",
                "_set_ideogram_prompt_field(obj, String(\"prompt\"), raw, source)",
                "_set_ideogram_prompt_field(obj, String(\"prompt_raw\"), raw, source)",
            ],
            severity=P0,
            acceptance="Ideogram4 prompt_json objects/arrays are serialized into the authored prompt string and preserved in prompt_raw.",
        )
    )
    checks.append(
        check_body_contains(
            WORKFLOW_GRAPH,
            "apply_workflow_params",
            category="workflow",
            label="workflow graph adapter entrypoint",
            needles=[
                "workflow",
                "params",
                "genparams",
                "nodes",
                "apply_typed_workflow_graph",
                "looks_like_comfy_api_prompt_graph",
                "apply_comfy_api_prompt_graph",
                "looks_like_comfy_ui_canvas_graph",
                "apply_comfy_ui_canvas_graph",
            ],
            severity=P0,
            acceptance="Graph module exposes flat workflow passthrough, Comfy API prompt import, Comfy UI visual canvas import, plus typed linked-graph execution for supported nodes.",
        )
    )
    checks.append(
        check_body_contains(
            WORKFLOW_GRAPH,
            "_workflow_has_prompt_override",
            category="workflow",
            label="Ideogram4 workflow prompt_json override",
            needles=[
                "prompt_json",
                "dumps(obj[\"prompt_json\"])",
                "Ideogram4 Comfy export prompt_json must be a string or JSON object/array",
                "_set_if_missing(obj, String(\"prompt_raw\"), JSONValue.from_string(raw))",
            ],
            severity=P0,
            acceptance="Raw Ideogram4 Comfy exports can provide structured JSON/bbox captions through prompt_json before the bounded importer runs.",
        )
    )
    checks.append(
        check_body_contains(
            WORKFLOW_GRAPH,
            "apply_workflow_params",
            category="workflow",
            label="raw Comfy Ideogram4 export branch",
            needles=[
                "looks_like_ideogram4_comfy_ui_export",
                "apply_ideogram4_comfy_ui_export",
            ],
            severity=P0,
            acceptance="Raw Comfy UI Ideogram4 txt2img exports are routed to a bounded importer before the generic type_id adapter.",
        )
    )
    checks.append(
        check_body_contains(
            WORKFLOW_GRAPH,
            "apply_ideogram4_comfy_ui_export",
            category="workflow",
            label="bounded Ideogram4 Comfy importer",
            needles=[
                "Text to Image (Ideogram v4)",
                "EmptyFlux2LatentImage",
                "UNETLoader",
                "ideogram-4-fp8",
                "KSamplerSelect",
                "BasicScheduler",
                "ModelSamplingAuraFlow",
                "DualModelGuider",
                "CFGOverride",
                "cfg_override_start_percent",
                "prompt-builder subgraph",
                "_workflow_set_seed_from_widget_if_missing",
                "active LoRA nodes",
            ],
            severity=P0,
            acceptance="The Ideogram4 importer extracts the known Comfy txt2img semantics and fails loud for prompt-builder/seed/LoRA gaps.",
        )
    )
    checks.append(
        check_contains(
            WORKFLOW_GRAPH,
            category="workflow",
            label="workflow graph normalizes Comfy UI node ids",
            needles=[
                "def _workflow_type_id",
                'removeprefix("comfy/")',
            ],
            severity=P0,
            acceptance="Frontend and imported Comfy graphs may use comfy/ type_id prefixes while the executor allowlist stays canonical.",
        )
    )
    checks.append(
        check_body_contains(
            WORKFLOW_GRAPH,
            "apply_comfy_api_prompt_graph",
            category="workflow",
            label="Comfy API prompt graph adapter",
            needles=[
                "_comfy_api_prompt_body",
                "_comfy_api_prompt_to_typed_graph",
                "apply_typed_workflow_graph",
                "comfy_api_prompt_graph",
            ],
            severity=P0,
            acceptance="ComfyUI API prompt graphs lower into the typed executor instead of requiring Serenity-only nodes/edges JSON.",
        )
    )
    checks.append(
        check_body_contains(
            WORKFLOW_GRAPH,
            "_comfy_api_prompt_to_typed_graph",
            category="workflow",
            label="Comfy API prompt link lowering",
            needles=[
                "class_type",
                "inputs",
                "_comfy_api_input_is_link",
                "_comfy_api_output_port",
                "_comfy_api_link_output_slot",
                "source_id",
            ],
            severity=P0,
            acceptance="ComfyUI API links [node_id, output_index] are converted into typed graph edges.",
        )
    )
    checks.append(
        check_body_contains(
            WORKFLOW_GRAPH,
            "comfy_ui_canvas_to_typed_graph",
            category="workflow",
            label="Comfy UI visual canvas link lowering",
            needles=[
                "widgets_values",
                "_comfy_ui_widget_fields",
                "_comfy_ui_output_port",
                "_comfy_ui_input_port",
                "source output slot",
                "target input slot",
            ],
            severity=P0,
            acceptance="Comfy UI visual canvas nodes/links lower into the typed executor with widget values mapped into fields.",
        )
    )
    checks.append(
        check_body_contains(
            WORKFLOW_GRAPH,
            "_comfy_ui_widget_fields",
            category="workflow",
            label="Comfy UI visual widget mapping",
            needles=[
                "LanPaint_KSampler",
                "LanPaint_KSamplerAdvanced",
                "LanPaint_SamplerCustomAdvanced",
                "LanPaint_MaskBlend",
                "ImagePadForOutpaint",
                "ThresholdMask",
                "SaveImage",
                "filename_prefix",
                "ImageToMask",
                "BasicScheduler",
                "denoise",
                "LanPaint_NumSteps",
                "LanPaint_PromptMode",
                "LanPaint_InnerThreshold",
                "blend_overlap",
                "feathering",
            ],
            severity=P0,
            acceptance="Comfy UI visual widget arrays map bounded LanPaint nodes into canonical fields before typed execution.",
        )
    )
    checks.append(
        check_body_contains(
            WORKFLOW_GRAPH,
            "apply_typed_workflow_graph",
            category="workflow",
            label="graph LoRA loader lowering",
            needles=[
                "LoraLoader",
                "LoraLoaderModelOnly",
                "ZImageLoraModelOnly",
                "_workflow_append_lora",
                "lora_name",
                "strength_model",
                "strength_clip",
                "CLIP_LORA_UNSUPPORTED",
                "workflow graph LoraLoader missing clip input",
                "workflow graph LoraLoaderModelOnly missing model input",
                "workflow graph ZImageLoraModelOnly missing model input",
                "_workflow_append_lora(obj, lora_name, strength)",
            ],
            severity=P0,
            acceptance="Comfy/SerenityFlow LoRA loader nodes lower model-side adapters into the existing flat LoRA metadata contract and keep CLIP-side LoRA fail-loud unless the clip strength is zero.",
        )
    )
    checks.append(
        check_body_contains(
            WORKFLOW_GRAPH,
            "apply_typed_workflow_graph",
            category="workflow",
            label="LanPaint outpaint preprocessing lowering",
            needles=[
                "ImagePadForOutpaint",
                "ThresholdMask",
                "outpaint_left",
                "outpaint_top",
                "outpaint_right",
                "outpaint_bottom",
                "outpaint_feathering",
                "threshold_mask_value",
                "threshold_mask_operator",
                "image_pad_for_outpaint",
                'JSONValue.from_string(String("gt"))',
                "workflow graph ImagePadForOutpaint missing image input",
                "workflow graph ThresholdMask missing mask input",
            ],
            severity=P0,
            acceptance="Bounded Comfy ImagePadForOutpaint and strict ThresholdMask imports preserve outpaint preprocessing metadata while full sampler/runtime parity remains fail-loud.",
        )
    )
    checks.append(
        check_body_contains(
            WORKFLOW_GRAPH,
            "apply_typed_workflow_graph",
            category="workflow",
            label="InpaintModelConditioning graph lowering",
            needles=[
                "InpaintModelConditioning",
                "inpaint_conditioning_image",
                "inpaint_conditioning_mask",
                "inpaint_conditioning_noise_mask",
                "workflow graph InpaintModelConditioning missing required typed input",
                'String("positive"), String("CONDITIONING")',
                'String("negative"), String("CONDITIONING")',
                'String("LATENT"), String("LATENT")',
                "concat conditioning",
            ],
            severity=P0,
            acceptance="Comfy InpaintModelConditioning imports preserve its conditioning-side concat metadata and do not collapse the node into plain SetLatentNoiseMask runtime parity.",
        )
    )
    checks.append(
        check_body_contains(
            WORKFLOW_GRAPH,
            "apply_typed_workflow_graph",
            category="workflow",
            label="Qwen edit image conditioning graph lowering",
            needles=[
                "TextEncodeQwenImageEdit",
                "TextEncodeQwenImageEditPlus",
                "qwen_edit_conditioning_image",
                '"[501] workflow graph " + type_id + " missing required typed input"',
                'String("CONDITIONING"), String("CONDITIONING")',
                "_workflow_image_path(image_nodes, image_ports, image_paths, image_link)",
            ],
            severity=P0,
            acceptance="SerenityFlow/Comfy Qwen edit text+image conditioning imports preserve the source image as explicit metadata instead of treating it as ordinary img2img or ReferenceLatent.",
        )
    )
    checks.append(
        check_body_contains(
            WORKFLOW_GRAPH,
            "apply_typed_workflow_graph",
            category="workflow",
            label="ConditioningSetMask graph lowering",
            needles=[
                "ConditioningSetMask",
                "conditioning_mask_image",
                "conditioning_mask_channel",
                "conditioning_mask_strength",
                "conditioning_mask_set_area_to_bounds",
                "workflow graph ConditioningSetMask missing required typed input",
                "_workflow_require_value_type(value_nodes, value_ports, value_types, mask_link, String(\"MASK\"), String(\"mask\"))",
                'String("CONDITIONING"), String("CONDITIONING")',
            ],
            severity=P0,
            acceptance="Comfy ConditioningSetMask imports preserve conditioning-side regional mask metadata without collapsing it into latent mask_image.",
        )
    )
    checks.append(
        check_body_contains(
            WORKFLOW_GRAPH,
            "apply_typed_workflow_graph",
            category="workflow",
            label="Reroute graph pass-through lowering",
            needles=[
                "Reroute",
                "_workflow_find_reroute_input_link",
                "workflow graph Reroute missing input",
                'String("REROUTE")',
                "_workflow_copy_value_metadata(",
                "input_link, node_id, String(\"REROUTE\"), actual",
            ],
            severity=P0,
            acceptance="Comfy Reroute imports behave as typed pass-through handles and preserve side-table metadata instead of becoming a synthetic tensor operation.",
        )
    )
    checks.append(
        check_contains(
            WORKFLOW_GRAPH,
            category="workflow",
            label="GetNode/SetNode virtual variable lowering",
            needles=[
                "SetNode",
                "GetNode",
                "_workflow_find_setnode_input_link",
                "_workflow_setget_name",
                "_workflow_setget_supported_type",
                "_workflow_type_accepts",
                "duplicate SetNode name",
                "GetNode missing SetNode",
                "SetNode missing input",
                "SetNode unsupported bus type",
                "GetNode output type mismatch",
                'String("SET")',
                'String("GET")',
                "setget_names",
                "output_type",
                "_workflow_copy_value_metadata(",
            ],
            severity=P0,
            acceptance="KJNodes Get/Set virtual variables lower as bounded typed handle pass-throughs with fail-loud ambiguity/type checks.",
        )
    )
    checks.append(
        check_contains(
            WORKFLOW_GRAPH,
            category="workflow",
            label="primitive scalar constant lowering",
            needles=[
                "PrimitiveInt",
                "PrimitiveFloat",
                "PrimitiveString",
                "PrimitiveStringMultiline",
                "PrimitiveBoolean",
                "INTConstant",
                "FloatConstant",
                "StringConstant",
                "StringConstantMultiline",
                "BOOLConstant",
                "PrimitiveNode",
                "_workflow_is_scalar_node",
                "_workflow_add_scalar",
                "_workflow_scalar_int",
                "_workflow_scalar_float",
                "_workflow_scalar_string",
                "_workflow_require_scalar_type",
                "scalar_nodes",
                "scalar_ports",
                "scalar_types",
                "workflow graph scalar metadata missing source",
                "StringConstantMultiline strip_newlines transform is unsupported",
                "[501] workflow graph input ",
            ],
            severity=P0,
            acceptance="Comfy core/KJ primitive scalar constants lower as first-class INT/FLOAT/STRING/BOOLEAN values and bad scalar consumers fail before enqueue.",
        )
    )
    checks.append(
        check_contains(
            WORKFLOW_GRAPH,
            category="workflow",
            label="UI-only drop and preview sink lowering",
            needles=[
                "PreviewImage",
                "MarkdownNote",
                "Note",
                'elif type_id == "PreviewImage"',
                '_workflow_find_input_link(edges, node_id, String("images"))',
                '_workflow_require_value_type(value_nodes, value_ports, value_types, image_link, String("IMAGE"), String("images"))',
                'elif type_id == "MarkdownNote" or type_id == "Note"',
            ],
            severity=P0,
            acceptance="Swarm/Comfy UI-only notes are inert and PreviewImage is an optional IMAGE sink with fail-loud type checking.",
        )
    )
    checks.append(
        check_contains(
            WORKFLOW_GRAPH,
            category="workflow",
            label="ComfySwitchNode static selector lowering",
            needles=[
                "ComfySwitchNode",
                'fields.set("switch", JSONValue.from_bool(_workflow_widget_bool(widgets, 0, False)))',
                'return String("output")',
                'elif type_id == "ComfySwitchNode"',
                '_workflow_find_input_link(edges, node_id, String("on_false"))',
                '_workflow_find_input_link(edges, node_id, String("on_true"))',
                '_workflow_find_input_link(edges, node_id, String("switch"))',
                "_workflow_scalar_bool",
                "selected = true_link.copy()",
                '_workflow_add_value(value_nodes, value_ports, value_types, node_id, String("output"), actual.copy())',
                "_workflow_copy_value_metadata(",
            ],
            severity=P0,
            acceptance="ComfySwitchNode mirrors Comfy's lazy boolean selector by copying only the selected branch's typed handle and metadata.",
        )
    )
    checks.append(
        check_body_contains(
            WORKFLOW_GRAPH,
            "_comfy_ui_output_port",
            category="workflow",
            label="Comfy UI blank-port Reroute import",
            needles=[
                'node_type == "Reroute"',
                'return String("REROUTE")',
            ],
            severity=P0,
            acceptance="Visual Comfy Reroute nodes with blank output names lower to a stable typed REROUTE port.",
        )
    )
    checks.append(
        check_body_contains(
            WORKFLOW_GRAPH,
            "_comfy_ui_input_port",
            category="workflow",
            label="Comfy UI blank-input Reroute import",
            needles=[
                'node_type == "Reroute" and port == ""',
                'return String("input")',
            ],
            severity=P0,
            acceptance="Visual Comfy Reroute nodes with blank input names lower to a stable input port.",
        )
    )
    checks.append(
        check_body_contains(
            WORKFLOW_GRAPH,
            "apply_typed_workflow_graph",
            category="workflow",
            label="typed linked graph IR executor",
            needles=[
                "_workflow_find_input_link",
                "_workflow_add_value",
                "_workflow_require_value_type",
                "_workflow_type_id",
                "edges",
                "MODEL",
                "CLIP",
                "VAE",
                "CONDITIONING",
                "image_paths",
                "LATENT",
                "latent_init_images",
                "IMAGE",
                "unresolved or cyclic typed links",
            ],
            severity=P0,
            acceptance="Supported Comfy/Swarm t2i graphs use typed handles and topological execution instead of field-only flattening.",
        )
    )
    checks.append(
        check_body_contains(
            WORKFLOW_GRAPH,
            "apply_typed_workflow_graph",
            category="workflow",
            label="bounded Klein edit typed graph markers",
            needles=[
                "ReferenceLatent",
                "SamplerCustomAdvanced",
                "GUIDER",
                "SIGMAS",
                "NOISE",
                "SAMPLER",
                "MASK",
                "COND_LATENT",
            ],
            severity=P0,
            acceptance="Bounded edit/inpaint graphs expose explicit typed handles for guider/sigmas/noise/sampler/mask/reference-latent lowering.",
        )
    )
    checks.append(
        check_body_contains(
            WORKFLOW_GRAPH,
            "apply_typed_workflow_graph",
            category="workflow",
            label="supported Comfy-like node markers",
            needles=SUPPORTED_NODE_TYPES,
            severity=P0,
            acceptance="Only the documented constrained t2i node markers are accepted.",
        )
    )
    checks.append(
        check_contains(
            WORKFLOW_GRAPH,
            category="workflow",
            label="Comfy ImageToMask channel parity guard",
            needles=[
                "_workflow_imagetomask_channel",
                "ImageToMask alpha is unsupported",
                "use LoadImage MASK",
            ],
            severity=P1,
            acceptance="Graph lowering does not treat a source file alpha channel as Comfy IMAGE alpha; alpha masks must come through LoadImage MASK.",
        )
    )
    checks.append(
        check_body_contains(
            WORKFLOW_GRAPH,
            "_workflow_loader_model_name",
            category="workflow",
            label="Comfy loader model-name aliases",
            needles=[
                "ckpt_name",
                "unet_name",
                "model_name",
            ],
            severity=P0,
            acceptance="CheckpointLoaderSimple, UNETLoader, and DiffusionModelLoader model-name fields are normalized before execution.",
        )
    )
    checks.append(
        check_body_contains(
            WORKFLOW_GRAPH,
            "_workflow_conditioning_prompt_text",
            category="workflow",
            label="Comfy text-encoder prompt aliases",
            needles=[
                "text",
                "CLIPTextEncodeFlux",
                "t5xxl",
                "clip_l",
            ],
            severity=P0,
            acceptance="CLIPTextEncode and CLIPTextEncodeFlux prompt fields are normalized before execution.",
        )
    )
    checks.append(
        check_body_contains(
            WORKFLOW_GRAPH,
            "apply_typed_workflow_graph",
            category="workflow",
            label="supported node field mappings",
            needles=[
                "model",
                "negative",
                "width",
                "height",
                "batch_size",
                "image",
                "path",
                "workflow_save_prefix",
                "init_image",
                "mask_image",
                "reference_image",
                "reference_latent_method",
                "reference_latent_count",
                "images",
                "steps",
                "seed",
                "cfg",
                "sampler_name",
                "sampler",
                "scheduler",
                "shift",
                "sigma_shift",
                "denoise",
                "creativity",
            ],
            severity=P0,
            acceptance="Accepted nodes map explicit fields into flat genparams.",
        )
    )
    checks.append(
        check_body_contains(
            WORKFLOW_GRAPH,
            "_workflow_copy_lanpaint_sampler_fields",
            category="workflow",
            label="LanPaint canonical field mappings",
            needles=[
                "lanpaint_num_steps",
                "lanpaint_lambda",
                "lanpaint_step_size",
                "lanpaint_beta",
                "lanpaint_friction",
                "lanpaint_prompt_mode",
                "lanpaint_inpainting_mode",
                "lanpaint_add_noise",
                "lanpaint_noise_seed",
                "lanpaint_start_at_step",
                "lanpaint_end_at_step",
                "lanpaint_return_with_leftover_noise",
                "lanpaint_early_stop",
                "lanpaint_inner_threshold",
                "lanpaint_inner_patience",
            ],
            severity=P0,
            acceptance="LanPaint sampler fields lower into canonical flat genparams before backend admission.",
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
                "var cfg_override: Float64",
                "var cfg_override_start_percent: Float64",
                "var cfg_override_end_percent: Float64",
                "var sampler: String",
                "var scheduler: String",
                "var sigma_shift: Float64",
                "var workflow_save_prefix: String",
                "var init_image: String",
                "var mask_image: String",
                "var lanpaint_mask_channel: String",
                "var lanpaint_mask_blend_overlap: Int",
                "var lanpaint_num_steps: Int",
                "var lanpaint_lambda: Float64",
                "var lanpaint_step_size: Float64",
                "var lanpaint_beta: Float64",
                "var lanpaint_friction: Float64",
                "var lanpaint_prompt_mode: String",
                "var lanpaint_inpainting_mode: String",
                "var lanpaint_add_noise: String",
                "var lanpaint_noise_seed: Int",
                "var lanpaint_start_at_step: Int",
                "var lanpaint_end_at_step: Int",
                "var lanpaint_return_with_leftover_noise: String",
                "var lanpaint_early_stop: Int",
                "var lanpaint_inner_threshold: Float64",
                "var lanpaint_inner_patience: Int",
                "var reference_image: String",
                "var reference_latent_method: String",
                "var reference_latent_count: Int",
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
        check_contains(
            WORKFLOW_GRAPH,
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
            WORKFLOW_GRAPH,
            category="failure",
            label="Ideogram4 Comfy import gaps fail loud",
            needles=[
                "[501] Ideogram4 Comfy export uses a prompt-builder subgraph",
                "[501] Ideogram4 Comfy export uses randomized seed",
                "[501] Ideogram4 Comfy export has active LoRA nodes",
            ],
            severity=P0,
            acceptance="Raw Ideogram4 Comfy exports fail before enqueue when prompt, seed, or LoRA semantics are not executable.",
        ),
        check_contains(
            WORKFLOW_GRAPH,
            category="failure",
            label="typed workflow graph validation fails loud",
            needles=[
                "[501] workflow graph body needs edges for typed execution",
                "[501] workflow graph input ",
                "[501] workflow graph has unresolved or cyclic typed links",
                "def _workflow_reject_multi_output_topology",
                "_workflow_reject_multi_output_topology(nodes_json)",
                "[501] workflow graph has multiple sampler/output branches",
                "[501] workflow graph has multiple SaveImage outputs",
                "[501] workflow graph duplicate SetNode name: ",
                "[501] workflow graph GetNode missing SetNode: ",
                "[501] workflow graph SetNode missing input",
                "[501] workflow graph SetNode unsupported bus type: ",
                "[501] workflow graph GetNode output type mismatch: ",
            ],
            severity=P0,
            acceptance="Bad typed links and cyclic linked graphs fail before enqueue.",
        ),
        check_not_contains(
            WORKFLOW_GRAPH,
            category="failure",
            label="field-only graph fallback removed",
            needles=[
                "field_only_graph_adapter",
            ],
            severity=P0,
            acceptance="Node graphs without typed edges fail loudly instead of guessing graph semantics from node fields.",
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
            acceptance="Flat LoRA metadata is parsed and recorded; graph LoraLoader/LoraLoaderModelOnly lower model-side adapters into the same product contract.",
        ),
        check_contains(
            WORKFLOW_GRAPH,
            category="workflow",
            label="workflow execution metadata",
            needles=[
                "WORKFLOW_GRAPH_EXECUTOR",
                "WORKFLOW_SCHEMA",
                "workflow_executor",
                "workflow_source",
                "workflow_node_count",
                "workflow_edge_count",
            ],
            severity=P1,
            acceptance="Graph execution records metadata for frontend history and artifact reuse.",
        ),
        check_contains(
            BACKEND,
            category="workflow",
            label="ReferenceLatent runtime contract helper",
            needles=[
                "reject_unsupported_reference_image_params",
                "Comfy ReferenceLatent/reference image conditioning is not supported",
            ],
            severity=P1,
            acceptance="ReferenceLatent metadata is explicit and existing non-Klein backends fail loud instead of treating it as ordinary img2img.",
        ),
        check_contains(
            BACKEND,
            category="workflow",
            label="SetLatentNoiseMask runtime contract helper",
            needles=[
                "reject_unsupported_mask_image_params",
                "Comfy SetLatentNoiseMask/inpaint mask conditioning is not supported",
            ],
            severity=P1,
            acceptance="Mask metadata is explicit and real backends fail loud until mask-aware denoise is wired.",
        ),
        check_contains(
            BACKEND,
            category="workflow",
            label="InpaintModelConditioning runtime contract helper",
            needles=[
                "inpaint_conditioning_image",
                "inpaint_conditioning_mask",
                "inpaint_conditioning_noise_mask",
                "reject_unsupported_inpaint_conditioning_params",
                "Comfy InpaintModelConditioning concat conditioning is not supported",
            ],
            severity=P1,
            acceptance="Comfy InpaintModelConditioning concat_latent_image/concat_mask metadata is explicit and real backends fail loud until they implement the conditioning path.",
        ),
        check_contains(
            BACKEND,
            category="workflow",
            label="Qwen edit image conditioning runtime contract helper",
            needles=[
                "qwen_edit_conditioning_image",
                "reject_unsupported_qwen_edit_conditioning_params",
                "Comfy TextEncodeQwenImageEdit image conditioning is not supported",
            ],
            severity=P1,
            acceptance="Comfy/SerenityFlow Qwen edit image-conditioning metadata is explicit and real backends fail loud until they implement the Qwen-Image-Edit source-image path.",
        ),
        check_contains(
            BACKEND,
            category="workflow",
            label="ConditioningSetMask runtime contract helper",
            needles=[
                "conditioning_mask_image",
                "conditioning_mask_channel",
                "conditioning_mask_strength",
                "conditioning_mask_set_area_to_bounds",
                "reject_unsupported_conditioning_mask_params",
                "Comfy ConditioningSetMask/regional conditioning is not supported",
            ],
            severity=P1,
            acceptance="Comfy ConditioningSetMask regional-conditioning metadata is explicit and real backends fail loud until they implement conditioning-side masks.",
        ),
        check_contains(
            BACKEND,
            category="workflow",
            label="LanPaint sampler runtime contract helper",
            needles=[
                "has_lanpaint_runtime_params",
                "has_lanpaint_sampler_runtime_params",
                "reject_unsupported_lanpaint_sampler_params",
                "reject_unsupported_lanpaint_params",
                "LanPaint_MaskBlend can be handled as a",
                "LanPaint inpaint sampler semantics are not supported",
                "LanPaint inpaint sampler/blend semantics are not supported",
            ],
            severity=P1,
            acceptance="LanPaint sampler-loop metadata is explicit and real backends fail loud unless they implement the LanPaint runtime loop; MaskBlend may be a separate final image blend only where a backend opts in.",
        ),
        check_contains(
            IMAGE_IO,
            category="workflow",
            label="LanPaint_MaskBlend pixel helpers",
            needles=[
                "smooth_lanpaint_blend_mask",
                "load_lanpaint_pixel_blend_mask",
                "apply_lanpaint_mask_blend_signed_chw",
                "image_area_resize_to_signed_nchw",
                "max-pool",
                "Gaussian blur",
                "Float64(blend_overlap - 1) / 4.0",
                "image1 * (1-mask) + image2 * mask",
            ],
            severity=P1,
            acceptance="The bounded LanPaint_MaskBlend pixel helper follows the oracle max-pool, Gaussian smooth, and image1/image2 compositing formula.",
        ),
        check_contains(
            DAEMON,
            category="workflow",
            label="daemon parses SaveImage filename prefix genparams",
            needles=[
                'p.workflow_save_prefix = _opt_str(obj, "workflow_save_prefix", String(""))',
                'o.set("workflow_save_prefix", JSONValue.from_string(p.workflow_save_prefix))',
            ],
            severity=P1,
            acceptance="Comfy SaveImage.filename_prefix survives parse_generate and canonical PNG/job metadata as workflow_save_prefix without changing artifact naming semantics.",
        ),
        check_contains(
            DAEMON,
            category="workflow",
            label="daemon parses mask image genparams",
            needles=[
                'p.mask_image = _opt_str(obj, "mask_image", String(""))',
                'o.set("mask_image", JSONValue.from_string(p.mask_image))',
            ],
            severity=P1,
            acceptance="SetLatentNoiseMask metadata survives parse_generate and canonical PNG/job metadata.",
        ),
        check_contains(
            DAEMON,
            category="workflow",
            label="daemon parses InpaintModelConditioning genparams",
            needles=[
                'p.inpaint_conditioning_image = _opt_str(obj, "inpaint_conditioning_image", String(""))',
                'p.inpaint_conditioning_mask = _opt_str(obj, "inpaint_conditioning_mask", String(""))',
                'p.inpaint_conditioning_noise_mask = _opt_bool(obj, "inpaint_conditioning_noise_mask", False)',
                'o.set("inpaint_conditioning_image", JSONValue.from_string(p.inpaint_conditioning_image))',
                'o.set("inpaint_conditioning_mask", JSONValue.from_string(p.inpaint_conditioning_mask))',
                'o.set("inpaint_conditioning_noise_mask", JSONValue.from_bool(p.inpaint_conditioning_noise_mask))',
            ],
            severity=P1,
            acceptance="InpaintModelConditioning metadata survives parse_generate and canonical PNG/job metadata.",
        ),
        check_contains(
            DAEMON,
            category="workflow",
            label="daemon parses Qwen edit image conditioning genparams",
            needles=[
                'p.qwen_edit_conditioning_image = _opt_str(obj, "qwen_edit_conditioning_image", String(""))',
                'o.set("qwen_edit_conditioning_image", JSONValue.from_string(p.qwen_edit_conditioning_image))',
                "TextEncodeQwenImageEdit image conditioning is not supported in this bounded slice",
            ],
            severity=P1,
            acceptance="Qwen edit image-conditioning metadata survives parse_generate/canonical PNG metadata and Ideogram4 rejects it before generic img2img handling.",
        ),
        check_contains(
            DAEMON,
            category="workflow",
            label="daemon parses ConditioningSetMask genparams",
            needles=[
                'p.conditioning_mask_image = _opt_str(obj, "conditioning_mask_image", String(""))',
                'p.conditioning_mask_channel = _opt_str(obj, "conditioning_mask_channel", String(""))',
                'p.conditioning_mask_strength = _opt_num(obj, "conditioning_mask_strength", -1.0, -1.0, 10.0)',
                'p.conditioning_mask_set_area_to_bounds = _opt_bool(obj, "conditioning_mask_set_area_to_bounds", False)',
                'o.set("conditioning_mask_image", JSONValue.from_string(p.conditioning_mask_image))',
                'o.set("conditioning_mask_set_area_to_bounds", JSONValue.from_bool(p.conditioning_mask_set_area_to_bounds))',
                "ConditioningSetMask/regional conditioning is not supported in this bounded slice",
            ],
            severity=P1,
            acceptance="ConditioningSetMask metadata survives parse_generate/canonical PNG metadata and Ideogram4 rejects it before generic mask handling.",
        ),
        check_contains(
            DAEMON,
            category="workflow",
            label="daemon parses LanPaint genparams",
            needles=[
                'p.lanpaint_num_steps = _opt_int(obj, "lanpaint_num_steps", -1, -1, 4096)',
                'p.lanpaint_prompt_mode = _opt_str(obj, "lanpaint_prompt_mode", String(""))',
                'p.lanpaint_mask_blend_overlap = _opt_int(obj, "lanpaint_mask_blend_overlap", -1, -1, 4096)',
                'o.set("lanpaint_num_steps", JSONValue.from_int(p.lanpaint_num_steps))',
                'o.set("lanpaint_prompt_mode", JSONValue.from_string(p.lanpaint_prompt_mode))',
                'o.set("lanpaint_mask_blend_overlap", JSONValue.from_int(p.lanpaint_mask_blend_overlap))',
            ],
            severity=P1,
            acceptance="LanPaint metadata survives parse_generate and canonical PNG/job metadata.",
        ),
        check_contains(
            IPC_CODEC,
            category="workflow",
            label="worker IPC preserves SaveImage filename prefix genparams",
            needles=[
                'o.set("workflow_save_prefix", JSONValue.from_string(p.workflow_save_prefix))',
                'p.workflow_save_prefix = obj["workflow_save_prefix"].as_string()',
            ],
            severity=P1,
            acceptance="Process-isolated worker IPC does not drop SaveImage filename-prefix metadata.",
        ),
        check_contains(
            IPC_CODEC,
            category="workflow",
            label="worker IPC preserves mask image genparams",
            needles=[
                'o.set("mask_image", JSONValue.from_string(p.mask_image))',
                'p.mask_image = obj["mask_image"].as_string()',
            ],
            severity=P1,
            acceptance="Process-isolated worker IPC does not drop SetLatentNoiseMask metadata.",
        ),
        check_contains(
            IPC_CODEC,
            category="workflow",
            label="worker IPC preserves InpaintModelConditioning genparams",
            needles=[
                'o.set("inpaint_conditioning_image", JSONValue.from_string(p.inpaint_conditioning_image))',
                'o.set("inpaint_conditioning_mask", JSONValue.from_string(p.inpaint_conditioning_mask))',
                'o.set("inpaint_conditioning_noise_mask", JSONValue.from_bool(p.inpaint_conditioning_noise_mask))',
                'p.inpaint_conditioning_image = obj["inpaint_conditioning_image"].as_string()',
                'p.inpaint_conditioning_mask = obj["inpaint_conditioning_mask"].as_string()',
                'p.inpaint_conditioning_noise_mask = obj["inpaint_conditioning_noise_mask"].as_bool()',
            ],
            severity=P1,
            acceptance="Process-isolated worker IPC does not drop InpaintModelConditioning metadata.",
        ),
        check_contains(
            IPC_CODEC,
            category="workflow",
            label="worker IPC preserves Qwen edit image conditioning genparams",
            needles=[
                'o.set("qwen_edit_conditioning_image", JSONValue.from_string(p.qwen_edit_conditioning_image))',
                'p.qwen_edit_conditioning_image = obj["qwen_edit_conditioning_image"].as_string()',
            ],
            severity=P1,
            acceptance="Process-isolated worker IPC does not drop Qwen edit image-conditioning metadata.",
        ),
        check_contains(
            IPC_CODEC,
            category="workflow",
            label="worker IPC preserves ConditioningSetMask genparams",
            needles=[
                'o.set("conditioning_mask_image", JSONValue.from_string(p.conditioning_mask_image))',
                'o.set("conditioning_mask_channel", JSONValue.from_string(p.conditioning_mask_channel))',
                'o.set("conditioning_mask_strength", JSONValue.from_float(p.conditioning_mask_strength))',
                'o.set("conditioning_mask_set_area_to_bounds", JSONValue.from_bool(p.conditioning_mask_set_area_to_bounds))',
                'p.conditioning_mask_image = obj["conditioning_mask_image"].as_string()',
                'p.conditioning_mask_set_area_to_bounds = obj["conditioning_mask_set_area_to_bounds"].as_bool()',
            ],
            severity=P1,
            acceptance="Process-isolated worker IPC does not drop ConditioningSetMask regional-conditioning metadata.",
        ),
        check_contains(
            IPC_CODEC,
            category="workflow",
            label="worker IPC preserves LanPaint genparams",
            needles=[
                'o.set("lanpaint_num_steps", JSONValue.from_int(p.lanpaint_num_steps))',
                'o.set("lanpaint_prompt_mode", JSONValue.from_string(p.lanpaint_prompt_mode))',
                'o.set("lanpaint_mask_blend_overlap", JSONValue.from_int(p.lanpaint_mask_blend_overlap))',
                'p.lanpaint_num_steps = Int(obj["lanpaint_num_steps"].as_float())',
                'p.lanpaint_prompt_mode = obj["lanpaint_prompt_mode"].as_string()',
                'p.lanpaint_mask_blend_overlap = Int(obj["lanpaint_mask_blend_overlap"].as_float())',
            ],
            severity=P1,
            acceptance="Process-isolated worker IPC does not drop LanPaint metadata.",
        ),
        check_contains(
            ZIMAGE_BACKEND,
            category="workflow",
            label="Z-Image rejects ReferenceLatent metadata",
            needles=[
                'reject_unsupported_reference_image_params(params, String("zimage"))',
            ],
            severity=P1,
            acceptance="Z-Image img2img does not silently consume Comfy ReferenceLatent/Klein edit metadata.",
        ),
        check_contains(
            ZIMAGE_BACKEND,
            category="workflow",
            label="Z-Image rejects InpaintModelConditioning metadata",
            needles=[
                "reject_unsupported_inpaint_conditioning_params",
                'reject_unsupported_inpaint_conditioning_params(params, String("zimage"))',
            ],
            severity=P1,
            acceptance="Z-Image preserve-mask img2img does not silently consume Comfy InpaintModelConditioning concat conditioning metadata.",
        ),
        check_contains(
            ZIMAGE_BACKEND,
            category="workflow",
            label="Z-Image rejects Qwen edit image conditioning metadata",
            needles=[
                "reject_unsupported_qwen_edit_conditioning_params",
                'reject_unsupported_qwen_edit_conditioning_params(params, String("zimage"))',
            ],
            severity=P1,
            acceptance="Z-Image img2img does not silently consume Comfy TextEncodeQwenImageEdit source-image conditioning metadata.",
        ),
        check_contains(
            ZIMAGE_BACKEND,
            category="workflow",
            label="Z-Image rejects ConditioningSetMask metadata",
            needles=[
                "reject_unsupported_conditioning_mask_params",
                'reject_unsupported_conditioning_mask_params(params, String("zimage"))',
            ],
            severity=P1,
            acceptance="Z-Image preserve-mask img2img does not silently consume Comfy regional-conditioning masks.",
        ),
        check_contains(
            ZIMAGE_BACKEND,
            category="workflow",
            label="Z-Image runs SetLatentNoiseMask img2img slice",
            needles=[
                "load_comfy_latent_preserve_mask",
                "_apply_inpaint_preserve_mask",
                "SetLatentNoiseMask requires an init_image/VAEEncode latent",
                '"inpaint_mask_applied"',
            ],
            severity=P1,
            acceptance="Z-Image no longer treats Comfy inpaint masks as metadata only; it validates a mask source and constrains known latent regions after sampler updates.",
        ),
        check_contains(
            ZIMAGE_BACKEND,
            category="sampler",
            label="Z-Image bounded Comfy sgm_uniform scheduler",
            needles=[
                "zimage_comfy_sgm_uniform_sigmas_with_shift",
                "sgm_uniform_flowmatch",
                "zimage_comfy_sgm_uniform_sigmas",
                "txt2img_initial_noise_scale",
                "generic uni_pc",
                "DISCARD_PENULTIMATE_SIGMA_SAMPLERS",
                "SigmaConvert",
            ],
            severity=P1,
            acceptance="Z-Image may execute the bounded Comfy-current sgm_uniform schedule slice for Euler/DPM++/both UniPC variants through Comfy discard-penultimate prep plus SigmaConvert wording while accepted sampler parity remains false.",
        ),
        check_contains(
            SAMPLER_REGISTRY,
            category="sampler",
            label="Z-Image registry exposes bounded sgm_uniform",
            needles=[
                "sgm_uniform_flowmatch",
                '["simple","flowmatch","flow_match","sgm_uniform"]',
                "bounded sgm_uniform flow-match schedules",
            ],
            severity=P1,
            acceptance="Sampler discovery/admission advertises only bounded Z-Image sgm_uniform support, not full scheduler parity.",
        ),
        check_contains(
            ZIMAGE_BACKEND,
            category="workflow",
            label="Z-Image rejects LanPaint sampler metadata",
            needles=[
                'reject_unsupported_lanpaint_sampler_params(params, String("zimage"))',
            ],
            severity=P1,
            acceptance="Z-Image keeps full LanPaint sampler-loop fields fail-loud instead of treating them as ordinary img2img controls.",
        ),
        check_contains(
            ZIMAGE_BACKEND,
            category="workflow",
            label="Z-Image applies bounded LanPaint_MaskBlend final pixels",
            needles=[
                "_apply_lanpaint_mask_blend",
                "LanPaint_MaskBlend requires init_image and mask_image",
                "decode_image_any(self.params.init_image)",
                "image_area_resize_to_signed_nchw",
                "load_lanpaint_pixel_blend_mask",
                "apply_lanpaint_mask_blend_signed_chw",
                "lanpaint_mask_blend_applied",
                "lanpaint_mask_blend_mean",
                '"lanpaint_mask_blend_applied"',
                '"lanpaint_mask_blend_overlap"',
                '"lanpaint_mask_blend_mean"',
            ],
            severity=P1,
            acceptance="Z-Image may consume only the bounded LanPaint_MaskBlend final decoded image blend, including Comfy/PyTorch area resize of the base image1 role, while full LanPaint sampler-loop fields remain rejected.",
        ),
        check_not_contains(
            ZIMAGE_BACKEND,
            category="workflow",
            label="Z-Image LanPaint_MaskBlend removed base-size precondition",
            needles=[
                "LanPaint_MaskBlend base image resize requires Comfy ImageScale(area) parity",
                "pre-scale init_image to output size for this backend slice",
            ],
            severity=P1,
            acceptance="Bounded Z-Image LanPaint_MaskBlend must no longer require init_image to already match output size when the base/original image can be resized with Comfy/PyTorch area semantics.",
        ),
        check_contains(
            QWEN_BACKEND,
            category="workflow",
            label="Qwen rejects ReferenceLatent metadata",
            needles=[
                'reject_unsupported_reference_image_params(params, String("qwenimage"))',
            ],
            severity=P1,
            acceptance="Qwen-Image does not silently consume Comfy ReferenceLatent/Klein edit metadata.",
        ),
        check_contains(
            QWEN_BACKEND,
            category="workflow",
            label="Qwen rejects SetLatentNoiseMask metadata",
            needles=[
                'reject_unsupported_mask_image_params(params, String("qwenimage"))',
            ],
            severity=P1,
            acceptance="Qwen-Image does not silently consume Comfy inpaint mask metadata.",
        ),
        check_contains(
            QWEN_BACKEND,
            category="workflow",
            label="Qwen rejects InpaintModelConditioning metadata",
            needles=[
                "reject_unsupported_inpaint_conditioning_params",
                'reject_unsupported_inpaint_conditioning_params(params, String("qwenimage"))',
            ],
            severity=P1,
            acceptance="Qwen-Image does not silently consume Comfy InpaintModelConditioning concat conditioning metadata.",
        ),
        check_contains(
            QWEN_BACKEND,
            category="workflow",
            label="Qwen rejects Qwen edit image conditioning metadata",
            needles=[
                "reject_unsupported_qwen_edit_conditioning_params",
                'reject_unsupported_qwen_edit_conditioning_params(params, String("qwenimage"))',
            ],
            severity=P1,
            acceptance="Qwen-Image txt2img backend rejects Qwen-Image-Edit source-image conditioning until the edit runtime path consumes it.",
        ),
        check_contains(
            QWEN_BACKEND,
            category="workflow",
            label="Qwen rejects ConditioningSetMask metadata",
            needles=[
                "reject_unsupported_conditioning_mask_params",
                'reject_unsupported_conditioning_mask_params(params, String("qwenimage"))',
            ],
            severity=P1,
            acceptance="Qwen-Image txt2img backend rejects ConditioningSetMask regional conditioning until that runtime path consumes it.",
        ),
        check_contains(
            QWEN_BACKEND,
            category="workflow",
            label="Qwen rejects LanPaint metadata",
            needles=[
                'reject_unsupported_lanpaint_params(params, String("qwenimage"))',
            ],
            severity=P1,
            acceptance="Qwen-Image does not silently consume LanPaint sampler or blend metadata.",
        ),
        check_contains(
            IDEOGRAM4_BACKEND,
            category="workflow",
            label="Ideogram4 rejects ReferenceLatent metadata",
            needles=[
                'reject_unsupported_reference_image_params(params, String("ideogram4"))',
            ],
            severity=P1,
            acceptance="Ideogram4 does not silently consume Comfy ReferenceLatent/Klein edit metadata.",
        ),
        check_contains(
            IDEOGRAM4_BACKEND,
            category="workflow",
            label="Ideogram4 rejects SetLatentNoiseMask metadata",
            needles=[
                'reject_unsupported_mask_image_params(params, String("ideogram4"))',
            ],
            severity=P1,
            acceptance="Ideogram4 does not silently consume Comfy inpaint mask metadata.",
        ),
        check_contains(
            IDEOGRAM4_BACKEND,
            category="workflow",
            label="Ideogram4 rejects InpaintModelConditioning metadata",
            needles=[
                "reject_unsupported_inpaint_conditioning_params",
                'reject_unsupported_inpaint_conditioning_params(params, String("ideogram4"))',
            ],
            severity=P1,
            acceptance="Ideogram4 does not silently consume Comfy InpaintModelConditioning concat conditioning metadata.",
        ),
        check_contains(
            IDEOGRAM4_BACKEND,
            category="workflow",
            label="Ideogram4 rejects Qwen edit image conditioning metadata",
            needles=[
                "reject_unsupported_qwen_edit_conditioning_params",
                'reject_unsupported_qwen_edit_conditioning_params(params, String("ideogram4"))',
            ],
            severity=P1,
            acceptance="Ideogram4 does not silently consume Comfy TextEncodeQwenImageEdit source-image conditioning metadata.",
        ),
        check_contains(
            IDEOGRAM4_BACKEND,
            category="workflow",
            label="Ideogram4 rejects ConditioningSetMask metadata",
            needles=[
                "reject_unsupported_conditioning_mask_params",
                'reject_unsupported_conditioning_mask_params(params, String("ideogram4"))',
            ],
            severity=P1,
            acceptance="Ideogram4 does not silently consume Comfy regional-conditioning masks.",
        ),
        check_contains(
            IDEOGRAM4_BACKEND,
            category="workflow",
            label="Ideogram4 rejects LanPaint metadata",
            needles=[
                'reject_unsupported_lanpaint_params(params, String("ideogram4"))',
            ],
            severity=P1,
            acceptance="Ideogram4 does not silently consume LanPaint sampler or blend metadata.",
        ),
        check_contains(
            KLEIN_BACKEND,
            category="workflow",
            label="Klein rejects LanPaint metadata",
            needles=[
                'reject_unsupported_lanpaint_params(params, String("klein"))',
            ],
            severity=P1,
            acceptance="Klein ReferenceLatent edit does not silently consume LanPaint sampler or blend metadata.",
        ),
        check_contains(
            KLEIN_BACKEND,
            category="workflow",
            label="Klein rejects InpaintModelConditioning metadata",
            needles=[
                "reject_unsupported_inpaint_conditioning_params",
                'reject_unsupported_inpaint_conditioning_params(params, String("klein"))',
            ],
            severity=P1,
            acceptance="Klein ReferenceLatent edit does not silently consume Comfy InpaintModelConditioning concat conditioning metadata.",
        ),
        check_contains(
            KLEIN_BACKEND,
            category="workflow",
            label="Klein rejects Qwen edit image conditioning metadata",
            needles=[
                "reject_unsupported_qwen_edit_conditioning_params",
                'reject_unsupported_qwen_edit_conditioning_params(params, String("klein"))',
            ],
            severity=P1,
            acceptance="Klein ReferenceLatent edit does not silently consume Comfy TextEncodeQwenImageEdit source-image conditioning metadata.",
        ),
        check_contains(
            KLEIN_BACKEND,
            category="workflow",
            label="Klein rejects ConditioningSetMask metadata",
            needles=[
                "reject_unsupported_conditioning_mask_params",
                'reject_unsupported_conditioning_mask_params(params, String("klein"))',
            ],
            severity=P1,
            acceptance="Klein ReferenceLatent edit does not silently consume Comfy regional-conditioning masks.",
        ),
        check_contains(
            SAMPLER_REGISTRY,
            category="sampler",
            label="Flux2/Klein registry remains metadata-only outside daemon bridge",
            needles=[
                'return String("flux2")',
                "Flux2/Klein daemon backend route exists",
                "cap-cache/ReferenceLatent bridge",
            ],
            severity=P1,
            acceptance="Flux2/Klein workflow imports preserve Swarm/Comfy scheduler metadata while direct in-process sampler registry execution stays blocked outside the staged daemon bridge.",
        ),
        check_contains(
            KLEIN_REFERENCE_BRIDGE,
            category="workflow",
            label="Klein ReferenceLatent bridge contract",
            needles=[
                "struct KleinReferenceLatentPlan",
                "plan_klein_reference_latent_bridge",
                "reference_latent_count",
                "reference_latent_method",
                "edit_sequence_tokens",
                "build_klein_reference_combined_img_ids",
                "build_klein_reference_combined_tokens",
                "prepare_combined_img_ids",
                "prepare_combined_tokens",
            ],
            severity=P1,
            acceptance="Klein edit metadata has a no-heavy bridge plan for target/reference token counts and combined image ids before full sampler execution is enabled.",
        ),
        check_contains(
            KLEIN_REFERENCE_BRIDGE_SMOKE,
            category="workflow",
            label="Klein ReferenceLatent bridge no-heavy smoke",
            needles=[
                "512 edit bridge",
                "1024 edit bridge",
                "plan_klein_reference_latent_bridge",
                "build_klein_reference_combined_img_ids",
                "build_klein_reference_combined_tokens",
                "PASS: Klein ReferenceLatent bridge no-heavy gate",
            ],
            severity=P1,
            acceptance="A focused smoke proves 512 and 1024 Klein edit token/id shapes without loading Qwen, Klein DiT, or the VAE.",
        ),
        check_contains(
            KLEIN_SAMPLE_CLI,
            category="workflow",
            label="Klein ReferenceLatent staged sampler CLI",
            needles=[
                "klein_sample_with_reference_latent",
                "klein_sample_with_reference_latent_initial_noise",
                "_encode_reference_512",
                "_encode_reference_1024",
                "serenity.klein_edit_parity_sidecar.v1",
                "edit_initial_noise_replay",
                "initial_noise_sidecar",
                "load_tensor_bin(initial_noise_path, ctx)",
                "save_tensor_bin(reference_latent, latent_path, ctx)",
                "reference_vae_latent.bin",
                "ReferenceLatent parity sidecar",
                "decode_image_any",
                "resize_bilinear",
                "KleinVaeEncoder",
                "N_EDIT_IMG_512",
                "S_EDIT_1024",
                "reference_t_offset",
            ],
            severity=P1,
            acceptance="The staged Klein sampler can decode/resize a source image, VAE-encode it, save the encoded reference latent for edit parity, optionally replay a supplied target initial-noise sidecar, and dispatch 512/1024 ReferenceLatent edit specializations instead of only txt2img.",
        ),
        check_contains(
            KLEIN_STACK_LORA,
            category="lora",
            label="Klein AI Toolkit LoRA key mapping",
            needles=[
                "_load_klein_flux2_double_blocks_lora",
                "diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight",
                "diffusion_model.single_blocks.",
                "_check_adapter_shape",
                "loaded Flux2/Klein double_blocks adapters",
            ],
            severity=P1,
            acceptance="Klein LoRA loading recognizes AI Toolkit/Comfy Flux2-Klein double_blocks and single_blocks keys and rejects shape-incompatible files before sampling.",
        ),
        check_contains(
            SERENITYFLOW_LATENT_NODES,
            category="workflow",
            label="Python oracle VAEEncode node",
            needles=[
                "@registry.register(",
                '"VAEEncode"',
                "def vae_encode_node",
                "bhwc_to_bchw",
                "latent = vae_encode(vae, image)",
                "return (wrap_latent(latent),)",
            ],
            severity=P1,
            acceptance="The Klein edit oracle starts from SerenityFlow/Comfy VAEEncode, which calls the normal VAE encoder and wraps that latent.",
        ),
        check_contains(
            SERENITYFLOW_CONDITIONING_NODES,
            category="workflow",
            label="Python oracle ReferenceLatent node",
            needles=[
                '"ReferenceLatent"',
                "def reference_latent",
                'n["reference_latent"] = ref',
                "return (out,)",
            ],
            severity=P1,
            acceptance="The Klein edit oracle treats ReferenceLatent as conditioning metadata, not ordinary img2img init noise.",
        ),
        check_contains(
            SERENITYFLOW_CONDITIONING_NODES,
            category="workflow",
            label="Python oracle Qwen edit image conditioning node",
            needles=[
                '"TextEncodeQwenImageEditPlus"',
                "def text_encode_qwen_image_edit_plus",
                'n["edit_image"] = image_bchw',
                "return (out,)",
            ],
            severity=P1,
            acceptance="The Qwen edit oracle treats the source image as conditioning-side edit_image metadata, not ordinary img2img.",
        ),
        check_contains(
            COMFY_NODES,
            category="workflow",
            label="Comfy oracle ConditioningSetMask node",
            needles=[
                "class ConditioningSetMask",
                '"mask": mask',
                '"set_area_to_bounds": set_area_to_bounds',
                '"mask_strength": strength',
            ],
            severity=P1,
            acceptance="The Comfy oracle treats ConditioningSetMask as conditioning-side mask metadata, not latent noise-mask metadata.",
        ),
        check_contains(
            SERENITYFLOW_BASIC_CONDITIONING_NODES,
            category="workflow",
            label="SerenityFlow oracle ConditioningSetMask node",
            needles=[
                '"ConditioningSetMask"',
                "def conditioning_set_mask",
                'n["mask"] = mask',
                'n["strength"] = strength',
                'n["set_area_to_bounds"] = (set_cond_area != "default")',
            ],
            severity=P1,
            acceptance="The SerenityFlow oracle treats ConditioningSetMask as conditioning-side regional mask metadata.",
        ),
        check_contains(
            SERENITYFLOW_SAMPLING,
            category="workflow",
            label="Python oracle Flux2/Klein reference sampler",
            needles=[
                "reference_latent = cond_entry.get(\"reference_latent\")",
                "ref_latent = reference_latent.float()",
                "ref_latent = _patchify_flux2_latents(ref_latent)",
                "flux_condition_tokens = _pack_flux2_latents(ref_latent.to(device))",
                "t_offset=10.0",
                "flux_img_ids = torch.cat([flux_img_ids, ref_img_ids], dim=0)",
                "flux_txt_ids = _prepare_flux2_text_ids(txt_len, device, model_dtype)",
                "txt_ids[:, 3] = torch.linspace(",
            ],
            severity=P1,
            acceptance="The reference-token/id semantics are guarded against SerenityFlow's Python Comfy-compatible sampler, not a Rust oracle.",
        ),
        check_contains(
            KLEIN_REFERENCE_DAEMON_SMOKE_RUNNER,
            category="workflow",
            label="Klein ReferenceLatent daemon smoke runner",
            needles=[
                "serenity.klein_reference_edit_daemon_smoke.v1",
                "serenity_daemon dispatch",
                "klein4b_edit.json",
                "klein9b_edit.json",
                "--case",
                "DEFAULT_WORKFLOWS",
                "reference_image",
                "reference_latent_count",
                "reference_latent_edit",
                "klein_sample_cli",
                "klein_precache_sample_prompts",
                "workflow_source",
                "comfy_api_prompt_graph",
            ],
            severity=P1,
            acceptance="The real Klein ReferenceLatent daemon smoke is reproducible as a checked-in product-path checker instead of only as ad hoc shell history.",
        ),
        check_contains(
            KLEIN_LORA_DAEMON_SMOKE_RUNNER,
            category="lora",
            label="Klein 9B LoRA daemon smoke runner",
            needles=[
                "serenity.klein9b_lora_daemon_smoke.v1",
                "serenity_daemon dispatch",
                "flux2_klein_9b_imperial_historical_lora.safetensors",
                "loaded Flux2/Klein double_blocks adapters: 144",
                "lora_count",
                "lora_weight",
                "GENPARAMS_KEY",
                "read_png_info",
                "log_markers",
            ],
            severity=P1,
            acceptance="The real Klein 9B LoRA txt2img daemon smoke is reproducible as a checked-in product-path checker instead of only as ad hoc shell history.",
        ),
        check_contains(
            KLEIN_LORA_REFERENCE_DAEMON_SMOKE_RUNNER,
            category="lora",
            label="Klein 9B LoRA ReferenceLatent daemon smoke runner",
            needles=[
                "serenity.klein9b_lora_reference_edit_daemon_smoke.v1",
                "klein9b_edit_lora.json",
                "patch_lora_workflow",
                "workflow_edge_count",
                "ReferenceLatent edit",
                "sample_command_has_reference",
                "loaded Flux2/Klein double_blocks adapters: 144",
                "reference_latent_edit",
                "lora_weight",
            ],
            severity=P1,
            acceptance="The real Klein 9B LoRA plus ReferenceLatent edit daemon smoke is reproducible as a checked-in product-path checker instead of only as ad hoc shell history.",
        ),
        check_contains(
            VISUAL_HEALTH_HELPER,
            category="workflow",
            label="real image visual-health helper",
            needles=[
                "compute_visual_health",
                "min_gray_stddev",
                "min_edge_mean",
                "min_edge_stddev",
                "high-frequency noise signature",
                "not judge aesthetics or oracle parity",
            ],
            severity=P1,
            acceptance=KLEIN_REAL_IMAGE_HEALTH_ACCEPTANCE,
        ),
        check_contains(
            KLEIN_REAL_IMAGE_HEALTH_RUNNER,
            category="workflow",
            label="Klein real image visual-health runner",
            needles=[
                "serenity.klein_real_image_health.v1",
                "compute_visual_health",
                "klein4b_reference_edit_daemon_smoke.json",
                "klein9b_reference_edit_daemon_smoke.json",
                "klein9b_lora_daemon_smoke.json",
                "klein9b_lora_reference_edit_daemon_smoke.json",
                "not pixel/latent/trajectory oracle parity",
            ],
            severity=P1,
            acceptance=KLEIN_REAL_IMAGE_HEALTH_ACCEPTANCE,
        ),
        check_contains(
            LANPAINT_ORACLE_SURFACE_RUNNER,
            category="workflow",
            label="LanPaint oracle surface checker",
            needles=[
                "serenity.lanpaint_oracle_surface.v1",
                "Z_image_Inpaint.json",
                "Qwen_Image_Inpaint.json",
                "Flux2_Klein_inpainting.json",
                "LanPaint_KSampler",
                "LanPaint_SamplerCustomAdvanced",
                "LanPaint_MaskBlend",
                "SetLatentNoiseMask",
                "denoise_mask = (denoise_mask > 0.5).float()",
                "latent_mask = 1 - denoise_mask",
                "smooth_lanpaint_blend_mask",
                "load_lanpaint_pixel_blend_mask",
                "apply_lanpaint_mask_blend_signed_chw",
                "image_area_resize_to_signed_nchw",
                "AREA_RESIZE_ROLE",
                LANPAINT_AREA_RESIZE_ROLE,
                "area_resize_2d_scalar",
                "PyTorch/Comfy interpolate(..., mode='area')",
                "forbid_all",
                "reject_unsupported_lanpaint_sampler_params",
                "non_claims",
            ],
            severity=P1,
            acceptance="LanPaint work is pinned to local Python/Comfy oracle semantics and representative workflow exports, with only the bounded final-pixel MaskBlend slice and its base-image area resize separated from full sampler runtime parity.",
        ),
        check_contains(
            LANPAINT_CANVAS_DAEMON_SMOKE_RUNNER,
            category="workflow",
            label="LanPaint canvas daemon smoke runner",
            needles=[
                "serenity.lanpaint_canvas_daemon_smoke.v1",
                "SDXL_Inpaint.json",
                "comfy_ui_canvas_graph",
                "LanPaint Comfy UI canvas",
                "lanpaint_num_steps",
                "lanpaint_prompt_mode",
                "lanpaint_mask_blend_overlap",
                "mask_image",
                "stub",
                "No-heavy product smoke",
            ],
            severity=P1,
            acceptance="A no-heavy product smoke proves visual LanPaint canvas lowering and metadata persistence through the daemon without claiming real denoise parity.",
        ),
        check_contains(
            KLEIN_BACKEND,
            category="workflow",
            label="Flux2/Klein staged daemon bridge",
            needles=[
                "struct KleinBackend",
                "from serenitymojo.serve.external_command import ExternalCommand",
                "self.sidecar.start",
                "self.sidecar.poll",
                "self.sidecar.require_success",
                "KLEIN_PRECACHE_BIN",
                "KLEIN_SAMPLER_BIN",
                "serenity.sample_prompts.v1",
                "from serenitymojo.serve.model_scan import LORAS_DIR",
                "_resolve_klein_lora_path",
                "_klein_lora_path",
                "_sample_prompts_json",
                "_precache_command",
                "_sample_command",
                "_shell_quote(lora_arg)",
                "validate_klein_cap_cache_header",
                "_joint_dim_for_variant",
                "_embed_genparams_in_png",
                ".klein_daemon_result.json",
                '"lora_path"',
                "plan_klein_reference_latent_bridge",
                "KLEIN_REFERENCE_EDIT_SHIFT",
                "_shell_quote(self.params.reference_image)",
                "edit_parity_sidecar_dir",
                "reference_vae_latent.bin",
                "reference_latent_edit",
                "no placeholder image was written",
            ],
            severity=P1,
            acceptance="Flux2/Klein daemon jobs stage per-job sample prompts, run Qwen3 cap-cache precache and route txt2img, single-LoRA txt2img, or bounded ReferenceLatent edit jobs to the existing staged sampler path with edit sidecar capture.",
        ),
        check_contains(
            EXTERNAL_COMMAND,
            category="workflow",
            label="daemon external command runner is pollable",
            needles=[
                "struct ExternalCommand",
                "sys_fork",
                "sys_execv",
                "sys_waitpid",
                "WNOHANG",
                "proc_kill_wait",
                "/bin/sh",
                "def poll",
                "def kill",
                "def require_success",
            ],
            severity=P1,
            acceptance="Long staged model commands run as pollable child processes so the daemon can keep servicing status/cancel while preserving existing shell command semantics.",
        ),
        check_contains(
            PIXI,
            category="workflow",
            label="Flux2/Klein staged binary build tasks",
            needles=[
                "build-klein-precache",
                "klein9b_precache_sample_prompts.mojo",
                "output/bin/klein_precache_sample_prompts",
                "build-klein-sampler",
                "klein_sample_cli.mojo",
                "output/bin/klein_sample_cli",
            ],
            severity=P1,
            acceptance="The daemon's required Klein sidecar binaries have first-class pixi build tasks.",
        ),
        check_contains(
            DISPATCH_BACKEND,
            category="workflow",
            label="Flux2/Klein dispatches to KleinBackend",
            needles=[
                "KIND_KLEIN",
                "from serenitymojo.serve.klein_backend import KleinBackend",
                "return KIND_KLEIN",
                'return String("klein")',
                "Flux2-dev remains explicitly unsupported",
            ],
            severity=P1,
            acceptance="Dispatch/isolated routing recognizes Klein as its own backend kind and does not route Flux2-dev through Klein.",
        ),
        check_not_contains(
            KLEIN_BACKEND,
            category="workflow",
            label="Flux2/Klein backend has no stub output path",
            needles=[
                "StubBackend",
                "_render_stub_png",
                "stub-preview-step",
            ],
            severity=P1,
            acceptance="The Klein backend can return successful sampler outputs, but it must not depend on StubBackend or stub image generation.",
        ),
        check_contains(
            WORKER,
            category="workflow",
            label="Flux2/Klein worker kind",
            needles=[
                "from serenitymojo.serve.klein_backend import KleinBackend",
                'elif kind == "klein"',
                "KleinBackend()",
            ],
            severity=P1,
            acceptance="Process-isolated daemon workers can construct the fail-loud Klein backend contract kind.",
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
            IDEOGRAM4_BACKEND,
            category="sampler",
            label="Ideogram4 simple AuraFlow schedule",
            needles=[
                "_build_ideogram4_simple_sigmas",
                "ideogram4_simple_flowmatch",
                "ideogram4_comfy_simple_aura_flow",
                "cfg_override",
                "sigma_shift",
            ],
            severity=P1,
            acceptance="Imported Ideogram4 Comfy workflows can request the bounded simple AuraFlow scheduler with CFGOverride metadata.",
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
            IMAGE_IO,
            category="image",
            label="Comfy/LanPaint mask decode substrate",
            needles=[
                "decode_comfy_mask",
                "load_image_mask",
                "resize_mask_bilinear",
                "resize_mask_nearest_exact",
                "binarize_lanpaint_denoise_mask",
                "load_comfy_latent_preserve_mask",
                "load_lanpaint_latent_preserve_mask",
            ],
            severity=P1,
            acceptance="Mask-conditioned runtime can extract Comfy LoadImage/ImageToMask masks and prepare both Comfy soft and LanPaint hard latent preserve masks.",
        ),
        check_contains(
            ZIMAGE_BACKEND,
            category="image",
            label="Z-Image init-image encode substrate",
            needles=[
                "resize_bilinear",
                "_encode_init_latent",
                "ZImageVaeEncoder",
                "creativity",
                "start step",
            ],
            severity=P1,
            acceptance="Z-Image has Mojo decode/resize/encode substrate for bounded init-image/LanPaint/refiner experiments; this is not accepted general i2i parity.",
        ),
        check_contains(
            VIDEO_API,
            category="video",
            label="video has bounded daemon smoke contract",
            needles=[
                "/v1/video",
                "LTX2_VIDEO_SMOKE_RUNNER",
                "ltx2_staged_smoke_video_result",
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


WORKFLOW_GRAPH_SMOKE_ACCEPTANCE = (
    "A daemon product smoke posts linked, LoRA, img2img, mask, Comfy API prompt, SerenityFlow t2i API-prompt graphs, "
    "bounded SerenityFlow edit graphs, and the Ideogram4 visual export, emits PNG genparams, "
    "and verifies typed-link/unsupported-node 501 failures."
)

KLEIN_REFERENCE_DAEMON_SMOKE_ACCEPTANCE = (
    "A real dispatch-mode daemon smoke runs a bounded Klein 4B ReferenceLatent edit through "
    "Qwen3 cap-cache precache plus the staged Klein sampler, emits a PNG with genparams, "
    "and writes the Klein daemon result manifest."
)

KLEIN_LORA_DAEMON_SMOKE_ACCEPTANCE = (
    "A real dispatch-mode daemon smoke runs bounded Klein 9B LoRA txt2img through "
    "Qwen3 cap-cache precache plus the staged Klein sampler, emits a PNG with genparams, "
    "writes the Klein daemon result manifest, and proves the AI Toolkit/Comfy LoRA loader path."
)

KLEIN_LORA_REFERENCE_DAEMON_SMOKE_ACCEPTANCE = (
    "A real dispatch-mode daemon smoke runs SerenityFlow's Klein 9B edit-LoRA Comfy graph through "
    "graph LoRA lowering, ReferenceLatent metadata lowering, Qwen3 cap-cache precache, and the "
    "staged Klein ReferenceLatent sampler with the AI Toolkit/Comfy LoRA loaded."
)

KLEIN_REAL_IMAGE_HEALTH_ACCEPTANCE = (
    "A no-heavy checker reads existing real Klein daemon PNG artifacts and rejects blank, flat, "
    "stub-like, or obvious high-frequency-noise images. This is a visual-health guard only, "
    "not aesthetic scoring or Python/Comfy pixel parity."
)

LANPAINT_ORACLE_SURFACE_ACCEPTANCE = (
    "A no-heavy checker pins representative LanPaint workflow exports, Python node semantics, "
    "SetLatentNoiseMask noise-mask behavior, Mojo mask math substrate, the bounded Z-Image "
    "LanPaint_MaskBlend final-pixel slice including its base-image area resize role, and the current fail-loud sampler boundary before "
    "full LanPaint runtime parity is claimed."
)

LANPAINT_CANVAS_DAEMON_SMOKE_ACCEPTANCE = (
    "A no-heavy daemon product smoke posts a real LanPaint Comfy UI canvas to the stub backend, "
    "lowers visual nodes/links through the typed graph executor, and proves mask/LanPaint metadata "
    "survives into PNG genparams without claiming real mask-aware denoise."
)


def dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def genparams_from_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    return dict_or_empty(dict_or_empty(evidence.get("png")).get("genparams"))


def serenityflow_t2i_case(report: dict[str, Any], case_name: str) -> dict[str, Any]:
    cases = report.get("serenityflow_t2i")
    if cases is not None and not isinstance(cases, dict):
        return {"_shape_error": "serenityflow_t2i must be an object"}
    if isinstance(cases, dict):
        case = cases.get(case_name)
        if case is not None and not isinstance(case, dict):
            return {"_shape_error": f"serenityflow_t2i.{case_name} must be an object"}
        if isinstance(case, dict):
            return case

    legacy_prefix = f"serenityflow_{case_name}"
    legacy = {
        "generate": report.get(f"{legacy_prefix}_generate"),
        "job": report.get(f"{legacy_prefix}_job"),
        "png": report.get(f"{legacy_prefix}_png"),
    }
    return legacy if any(isinstance(value, dict) for value in legacy.values()) else {}


def serenityflow_edit_case(report: dict[str, Any], case_name: str) -> dict[str, Any]:
    cases = report.get("serenityflow_edit")
    if not isinstance(cases, dict):
        return {"_shape_error": "serenityflow_edit must be an object"}
    case = cases.get(case_name)
    if not isinstance(case, dict):
        return {"_shape_error": f"serenityflow_edit.{case_name} must be an object"}
    return case


def serenityflow_qwen_edit_case(report: dict[str, Any], case_name: str) -> dict[str, Any]:
    cases = report.get("serenityflow_qwen_edit")
    if not isinstance(cases, dict):
        return {"_shape_error": "serenityflow_qwen_edit must be an object"}
    case = cases.get(case_name)
    if not isinstance(case, dict):
        return {"_shape_error": f"serenityflow_qwen_edit.{case_name} must be an object"}
    return case


def prefixed_evidence(report: dict[str, Any], prefix: str) -> dict[str, Any]:
    direct = report.get(prefix)
    if isinstance(direct, dict):
        return direct
    if direct is not None:
        return {"_shape_error": f"{prefix} must be an object"}

    keys = [key for key in report if key.startswith(f"{prefix}_")]
    if not keys:
        return {}
    return {
        "generate": report.get(f"{prefix}_generate"),
        "job": report.get(f"{prefix}_job"),
        "png": report.get(f"{prefix}_png"),
    }


def evidence_has_job_png(evidence: dict[str, Any]) -> bool:
    return isinstance(evidence.get("job"), dict) and isinstance(evidence.get("png"), dict)


def genparam_matches(genparams: dict[str, Any], key: str, expected: Any) -> bool:
    return key in genparams and genparams.get(key) == expected


def evidence_count_matches(
    evidence: dict[str, Any],
    genparams: dict[str, Any],
    workflow_key: str,
    flat_key: str,
    expected: int,
) -> bool:
    values = []
    if workflow_key in genparams:
        values.append(genparams.get(workflow_key))
    if flat_key in evidence:
        values.append(evidence.get(flat_key))
    return bool(values) and all(value == expected for value in values)


def validate_zimage_t2i_evidence(evidence: dict[str, Any]) -> str:
    if evidence.get("_shape_error"):
        return str(evidence["_shape_error"])
    if not evidence_has_job_png(evidence):
        return "SerenityFlow zimage_t2i job/png evidence missing from product report"
    genparams = genparams_from_evidence(evidence)
    expected = {
        "workflow_source": "comfy_api_prompt_graph",
        "model": "z_image_turbo_bf16.safetensors",
        "prompt": "a stunning landscape photograph",
        "negative": "",
    }
    missing = [
        f"{key}={value!r}"
        for key, value in expected.items()
        if not genparam_matches(genparams, key, value)
    ]
    if missing:
        return "SerenityFlow zimage_t2i metadata mismatch: " + ", ".join(missing)
    if not evidence_count_matches(evidence, genparams, "workflow_node_count", "node_count", 10):
        return "SerenityFlow zimage_t2i node count missing or mismatched"
    if not evidence_count_matches(evidence, genparams, "workflow_edge_count", "edge_count", 10):
        return "SerenityFlow zimage_t2i edge count missing or mismatched"
    return ""


def validate_klein_edit_evidence(evidence: dict[str, Any], case_name: str, model: str) -> str:
    if evidence.get("_shape_error"):
        return str(evidence["_shape_error"])
    if not evidence_has_job_png(evidence):
        return f"SerenityFlow edit {case_name} job/png evidence missing from product report"
    genparams = genparams_from_evidence(evidence)
    expected = {
        "model": model,
        "prompt": "change the dress to blue",
        "negative": "",
        "width": 1024,
        "height": 1024,
        "images": 1,
        "steps": 35,
        "seed": 42,
        "cfg": 3.5,
        "sampler": "euler",
        "scheduler": "flux2",
        "init_image": "input.png",
        "reference_image": "input.png",
        "reference_latent_method": "index",
        "reference_latent_count": 2,
        "workflow_node_count": 18,
        "workflow_edge_count": 21,
    }
    missing = [
        f"{key}={value!r}"
        for key, value in expected.items()
        if not genparam_matches(genparams, key, value)
    ]
    if missing:
        return f"SerenityFlow edit {case_name} metadata mismatch: " + ", ".join(missing)
    return ""


def validate_qwen_edit_evidence(evidence: dict[str, Any], case_name: str, has_lora: bool) -> str:
    if evidence.get("_shape_error"):
        return str(evidence["_shape_error"])
    if not evidence_has_job_png(evidence):
        return f"SerenityFlow Qwen edit {case_name} job/png evidence missing from product report"
    genparams = genparams_from_evidence(evidence)
    expected = {
        "workflow_source": "comfy_api_prompt_graph",
        "model": "qwen_image_edit.safetensors",
        "prompt": "change the background to a beach",
        "negative": "",
        "steps": 20,
        "seed": 42,
        "cfg": 1,
        "sampler": "euler",
        "scheduler": "simple",
        "creativity": 0.75,
        "sigma_shift": 3,
        "init_image": "input.png",
        "qwen_edit_conditioning_image": "input.png",
        "workflow_save_prefix": "qwen_edit_lora" if has_lora else "qwen_edit",
        "workflow_node_count": 13 if has_lora else 12,
        "workflow_edge_count": 15 if has_lora else 14,
    }
    missing = [
        f"{key}={value!r}"
        for key, value in expected.items()
        if not genparam_matches(genparams, key, value)
    ]
    loras = genparams.get("lora")
    if has_lora:
        if loras != [{"name": "lora.safetensors", "weight": 1.0}]:
            missing.append("lora=[lora.safetensors:1.0]")
    elif loras != []:
        missing.append("lora=[]")
    if missing:
        return f"SerenityFlow Qwen edit {case_name} metadata mismatch: " + ", ".join(missing)
    return ""


def validate_ideogram4_visual_export_evidence(evidence: dict[str, Any]) -> str:
    if not evidence:
        return ""
    if evidence.get("_shape_error"):
        return str(evidence["_shape_error"])
    if not evidence_has_job_png(evidence):
        return "Ideogram4 visual export job/png evidence missing from product report"
    genparams = genparams_from_evidence(evidence)
    expected = {
        "workflow_source": "ideogram4_comfy_ui_export",
        "model": "ideogram-4-fp8",
        "prompt": "a surreal streetwear collage poster with blue sky and large COMFY letters",
        "width": 1024,
        "height": 1024,
        "seed": 424242,
        "sampler": "euler",
        "scheduler": "simple",
        "sigma_shift": 5,
        "cfg": 7,
        "workflow_node_count": 28,
        "workflow_edge_count": 16,
    }
    missing = [
        f"{key}={value!r}"
        for key, value in expected.items()
        if not genparam_matches(genparams, key, value)
    ]
    if missing:
        return "Ideogram4 visual export metadata mismatch: " + ", ".join(missing)
    return ""


def _evidence_path(path_value: Any) -> Path:
    path = Path(str(path_value or ""))
    return path if path.is_absolute() else REPO / path


def check_klein_reference_daemon_smoke_report(
    smoke_path: Path, expected_variant: str, expected_model: str, expected_config: str,
) -> Check:
    report = read_json(smoke_path)
    label = f"Klein {expected_variant} ReferenceLatent real daemon smoke"
    if not report:
        return Check(
            False,
            P1,
            "workflow",
            label,
            f"missing report: {rel(smoke_path)}",
            rel(smoke_path),
            KLEIN_REFERENCE_DAEMON_SMOKE_ACCEPTANCE,
        )

    generate = dict_or_empty(report.get("generate"))
    job = dict_or_empty(report.get("job"))
    genparams = dict_or_empty(report.get("genparams"))
    manifest = dict_or_empty(report.get("manifest"))
    request = dict_or_empty(report.get("request"))
    output_path = _evidence_path(report.get("output_path"))
    manifest_path = _evidence_path(report.get("manifest_path"))
    expected_steps = request.get("steps") if isinstance(request.get("steps"), int) else 1
    expected_creativity = (
        request.get("creativity") if isinstance(request.get("creativity"), (int, float)) else 0.45
    )
    expected_reference = str(request.get("reference_image") or genparams.get("reference_image") or "")
    expected_reference_path = _evidence_path(expected_reference)

    missing = []
    if report.get("ready") is not True:
        missing.append("report.ready=True")
    if report.get("blockers") not in ([], None):
        missing.append("report.blockers=[]")
    if generate.get("status") != 200:
        missing.append("generate.status=200")
    if job.get("state") != "done":
        missing.append("job.state='done'")
    if job.get("step") != expected_steps or job.get("total") != expected_steps:
        missing.append(f"job.step/total={expected_steps}/{expected_steps}")
    if not output_path.is_file():
        missing.append(f"output PNG exists: {output_path}")
    elif output_path.stat().st_size < 100_000:
        missing.append("output PNG is nontrivial")
    if not manifest_path.is_file():
        missing.append(f"Klein manifest exists: {manifest_path}")
    if not expected_reference:
        missing.append("request.reference_image is nonempty")
    elif not expected_reference_path.is_file():
        missing.append(f"reference image exists: {expected_reference_path}")

    expected_genparams = {
        "model": expected_model,
        "prompt": "change the dress to blue",
        "width": 512,
        "height": 512,
        "steps": expected_steps,
        "seed": 42,
        "cfg": 3.5,
        "sampler": "euler",
        "scheduler": "flux2",
        "creativity": expected_creativity,
        "reference_image": expected_reference,
        "reference_latent_method": "index",
        "reference_latent_count": 2,
        "workflow_schema": "serenity.workflow_graph.v1",
        "workflow_executor": "serenity.workflow_graph.executor.v1",
        "workflow_source": "comfy_api_prompt_graph",
        "workflow_node_count": 18,
        "workflow_edge_count": 21,
    }
    missing.extend(
        f"genparams.{key}={expected!r}"
        for key, expected in expected_genparams.items()
        if genparams.get(key) != expected
    )

    expected_manifest = {
        "schema": "serenity.klein_daemon_result.v1",
        "backend": "klein",
        "variant": expected_variant,
        "model": expected_model,
        "config_path": expected_config,
        "mode": "reference_latent_edit",
        "reference_image": expected_reference,
        "reference_latent_count": 2,
        "edit_denoise": expected_creativity,
        "edit_shift": 2.02,
        "reference_t_offset": 10.0,
        "metadata_key": "serenity.genparams.v1",
    }
    missing.extend(
        f"manifest.{key}={expected!r}"
        for key, expected in expected_manifest.items()
        if manifest.get(key) != expected
    )
    if not str(manifest.get("sampler_binary") or "").endswith("/output/bin/klein_sample_cli"):
        missing.append("manifest.sampler_binary=output/bin/klein_sample_cli")
    if not str(manifest.get("precache_binary") or "").endswith("/output/bin/klein_precache_sample_prompts"):
        missing.append("manifest.precache_binary=output/bin/klein_precache_sample_prompts")

    if missing:
        return Check(
            False,
            P1,
            "workflow",
            label,
            "missing evidence: " + ", ".join(missing),
            rel(smoke_path),
            KLEIN_REFERENCE_DAEMON_SMOKE_ACCEPTANCE,
        )

    return Check(
        True,
        PASS,
        "workflow",
        label,
        (
            f"dispatch job {job.get('id')} wrote {rel(output_path)} with "
            f"manifest mode {manifest.get('mode')}"
        ),
        rel(smoke_path),
        KLEIN_REFERENCE_DAEMON_SMOKE_ACCEPTANCE,
    )


def check_klein_lora_daemon_smoke_report(smoke_path: Path) -> Check:
    report = read_json(smoke_path)
    label = "Klein 9B LoRA real daemon smoke"
    if not report:
        return Check(
            False,
            P1,
            "lora",
            label,
            f"missing report: {rel(smoke_path)}",
            rel(smoke_path),
            KLEIN_LORA_DAEMON_SMOKE_ACCEPTANCE,
        )

    expected_lora = "/home/alex/Downloads/flux2_klein_9b_imperial_historical_lora.safetensors"
    expected_prompt = (
        "srx_ottoman, desert caravan approaching an ancient city, camel riders and "
        "walking travelers in traditional robes and turbans, rocky desert road, "
        "fortified historical city in the distance, domes and a tall minaret, warm "
        "sunlight, dusty air, epic cinematic historical mood, highly detailed"
    )
    generate = dict_or_empty(report.get("generate"))
    job = dict_or_empty(report.get("job"))
    png = dict_or_empty(report.get("png"))
    genparams = dict_or_empty(report.get("genparams"))
    manifest = dict_or_empty(report.get("manifest"))
    log_markers = dict_or_empty(report.get("log_markers"))
    output_path = _evidence_path(report.get("output_path"))
    manifest_path = _evidence_path(report.get("manifest_path"))

    missing = []
    if report.get("ready") is not True:
        missing.append("report.ready=True")
    if report.get("blockers") not in ([], None):
        missing.append("report.blockers=[]")
    if generate.get("status") != 200:
        missing.append("generate.status=200")
    if job.get("state") != "done":
        missing.append("job.state='done'")
    if job.get("step") != 4 or job.get("total") != 4:
        missing.append("job.step/total=4/4")
    if not output_path.is_file():
        missing.append(f"output PNG exists: {output_path}")
    elif output_path.stat().st_size < 100_000:
        missing.append("output PNG is nontrivial")
    if png.get("width") != 512 or png.get("height") != 512:
        missing.append("png.width/height=512/512")
    text_keys = png.get("text_keys")
    if not isinstance(text_keys, list) or "serenity.genparams.v1" not in text_keys:
        missing.append("png.text_keys includes serenity.genparams.v1")
    if not str(report.get("idat_sha256") or ""):
        missing.append("idat_sha256 is present")
    if not manifest_path.is_file():
        missing.append(f"Klein manifest exists: {manifest_path}")

    expected_genparams = {
        "schema": "serenity.genparams.v1",
        "model": "flux2-klein-9b.safetensors",
        "prompt": expected_prompt,
        "prompt_raw": expected_prompt,
        "negative": "",
        "width": 512,
        "height": 512,
        "steps": 4,
        "seed": 424242,
        "cfg": 3.5,
        "sampler": "euler",
        "scheduler": "simple",
        "images": 1,
        "creativity": 0.5,
        "init_image": "",
        "mask_image": "",
        "reference_image": "",
        "reference_latent_method": "",
        "reference_latent_count": 0,
    }
    missing.extend(
        f"genparams.{key}={expected!r}"
        for key, expected in expected_genparams.items()
        if genparams.get(key) != expected
    )
    loras = genparams.get("lora")
    if not isinstance(loras, list) or len(loras) != 1:
        missing.append("genparams.lora has one entry")
    else:
        lora = dict_or_empty(loras[0])
        if lora.get("name") != expected_lora:
            missing.append(f"genparams.lora[0].name={expected_lora!r}")
        if lora.get("weight") != 1.0:
            missing.append("genparams.lora[0].weight=1.0")

    expected_manifest = {
        "schema": "serenity.klein_daemon_result.v1",
        "backend": "klein",
        "variant": "9b",
        "model": "flux2-klein-9b.safetensors",
        "config_path": "/home/alex/mojodiffusion/serenitymojo/configs/klein9b.json",
        "lora_count": 1,
        "lora_name": expected_lora,
        "lora_path": expected_lora,
        "lora_weight": 1.0,
        "metadata_key": "serenity.genparams.v1",
    }
    missing.extend(
        f"manifest.{key}={expected!r}"
        for key, expected in expected_manifest.items()
        if manifest.get(key) != expected
    )
    if not str(manifest.get("sampler_binary") or "").endswith("/output/bin/klein_sample_cli"):
        missing.append("manifest.sampler_binary=output/bin/klein_sample_cli")
    if not str(manifest.get("precache_binary") or "").endswith("/output/bin/klein_precache_sample_prompts"):
        missing.append("manifest.precache_binary=output/bin/klein_precache_sample_prompts")
    if not Path(str(manifest.get("lora_path") or "")).is_file():
        missing.append("manifest.lora_path exists on disk")
    if not str(manifest.get("output_png") or "").endswith(output_path.name):
        missing.append("manifest.output_png matches report output")
    if log_markers.get("sample_command_has_lora") is not True:
        missing.append("log_markers.sample_command_has_lora=True")
    if log_markers.get("loaded_adapter_count") is not True:
        missing.append("log_markers.loaded_adapter_count=True")
    if log_markers.get("done_staged_sample") is not True:
        missing.append("log_markers.done_staged_sample=True")

    if missing:
        return Check(
            False,
            P1,
            "lora",
            label,
            "missing evidence: " + ", ".join(missing),
            rel(smoke_path),
            KLEIN_LORA_DAEMON_SMOKE_ACCEPTANCE,
        )

    return Check(
        True,
        PASS,
        "lora",
        label,
        (
            f"dispatch job {job.get('id')} wrote {rel(output_path)} with "
            f"LoRA {Path(expected_lora).name}"
        ),
        rel(smoke_path),
        KLEIN_LORA_DAEMON_SMOKE_ACCEPTANCE,
    )


def check_klein_lora_reference_daemon_smoke_report(smoke_path: Path) -> Check:
    report = read_json(smoke_path)
    label = "Klein 9B LoRA ReferenceLatent real daemon smoke"
    if not report:
        return Check(
            False,
            P1,
            "lora",
            label,
            f"missing report: {rel(smoke_path)}",
            rel(smoke_path),
            KLEIN_LORA_REFERENCE_DAEMON_SMOKE_ACCEPTANCE,
        )

    expected_lora = "/home/alex/Downloads/flux2_klein_9b_imperial_historical_lora.safetensors"
    generate = dict_or_empty(report.get("generate"))
    job = dict_or_empty(report.get("job"))
    png = dict_or_empty(report.get("png"))
    genparams = dict_or_empty(report.get("genparams"))
    manifest = dict_or_empty(report.get("manifest"))
    workflow_metadata = dict_or_empty(report.get("workflow_metadata"))
    log_markers = dict_or_empty(report.get("log_markers"))
    request = dict_or_empty(report.get("request"))
    output_path = _evidence_path(report.get("output_path"))
    manifest_path = _evidence_path(report.get("manifest_path"))
    expected_steps = request.get("steps") if isinstance(request.get("steps"), int) else 1
    expected_creativity = (
        request.get("creativity") if isinstance(request.get("creativity"), (int, float)) else 0.45
    )
    expected_reference = str(request.get("reference_image") or genparams.get("reference_image") or "")
    expected_reference_path = _evidence_path(expected_reference)

    missing = []
    if report.get("ready") is not True:
        missing.append("report.ready=True")
    if report.get("blockers") not in ([], None):
        missing.append("report.blockers=[]")
    if workflow_metadata.get("workflow_path") != "/home/alex/serenityflow-v2/serenityflow/workflows/klein9b_edit_lora.json":
        missing.append("workflow_metadata.workflow_path=klein9b_edit_lora.json")
    if workflow_metadata.get("patched_lora_nodes") != 1:
        missing.append("workflow_metadata.patched_lora_nodes=1")
    if workflow_metadata.get("workflow_node_count") != 19:
        missing.append("workflow_metadata.workflow_node_count=19")
    if workflow_metadata.get("workflow_edge_count") != 22:
        missing.append("workflow_metadata.workflow_edge_count=22")
    if generate.get("status") != 200:
        missing.append("generate.status=200")
    if job.get("state") != "done":
        missing.append("job.state='done'")
    if job.get("step") != expected_steps or job.get("total") != expected_steps:
        missing.append(f"job.step/total={expected_steps}/{expected_steps}")
    if not output_path.is_file():
        missing.append(f"output PNG exists: {output_path}")
    elif output_path.stat().st_size < 100_000:
        missing.append("output PNG is nontrivial")
    if png.get("width") != 512 or png.get("height") != 512:
        missing.append("png.width/height=512/512")
    text_keys = png.get("text_keys")
    if not isinstance(text_keys, list) or "serenity.genparams.v1" not in text_keys:
        missing.append("png.text_keys includes serenity.genparams.v1")
    if not str(report.get("idat_sha256") or ""):
        missing.append("idat_sha256 is present")
    if not manifest_path.is_file():
        missing.append(f"Klein manifest exists: {manifest_path}")
    if not expected_reference:
        missing.append("request.reference_image is nonempty")
    elif not expected_reference_path.is_file():
        missing.append(f"reference image exists: {expected_reference_path}")

    expected_genparams = {
        "schema": "serenity.genparams.v1",
        "model": "flux2-klein-9b.safetensors",
        "prompt": "change the dress to blue",
        "negative": "",
        "width": 512,
        "height": 512,
        "steps": expected_steps,
        "seed": 42,
        "cfg": 3.5,
        "sampler": "euler",
        "scheduler": "flux2",
        "creativity": expected_creativity,
        "reference_image": expected_reference,
        "reference_latent_method": "index",
        "reference_latent_count": 2,
        "workflow_schema": "serenity.workflow_graph.v1",
        "workflow_executor": "serenity.workflow_graph.executor.v1",
        "workflow_source": "comfy_api_prompt_graph",
        "workflow_node_count": 19,
        "workflow_edge_count": 22,
    }
    missing.extend(
        f"genparams.{key}={expected!r}"
        for key, expected in expected_genparams.items()
        if genparams.get(key) != expected
    )
    loras = genparams.get("lora")
    if not isinstance(loras, list) or len(loras) != 1:
        missing.append("genparams.lora has one entry")
    else:
        lora = dict_or_empty(loras[0])
        if lora.get("name") != expected_lora:
            missing.append(f"genparams.lora[0].name={expected_lora!r}")
        if lora.get("weight") != 1.0:
            missing.append("genparams.lora[0].weight=1.0")

    expected_manifest = {
        "schema": "serenity.klein_daemon_result.v1",
        "backend": "klein",
        "variant": "9b",
        "model": "flux2-klein-9b.safetensors",
        "config_path": "/home/alex/mojodiffusion/serenitymojo/configs/klein9b.json",
        "lora_count": 1,
        "lora_name": expected_lora,
        "lora_path": expected_lora,
        "lora_weight": 1.0,
        "mode": "reference_latent_edit",
        "reference_image": expected_reference,
        "reference_latent_count": 2,
        "edit_denoise": expected_creativity,
        "edit_shift": 2.02,
        "reference_t_offset": 10.0,
        "metadata_key": "serenity.genparams.v1",
    }
    missing.extend(
        f"manifest.{key}={expected!r}"
        for key, expected in expected_manifest.items()
        if manifest.get(key) != expected
    )
    if not str(manifest.get("sampler_binary") or "").endswith("/output/bin/klein_sample_cli"):
        missing.append("manifest.sampler_binary=output/bin/klein_sample_cli")
    if not str(manifest.get("precache_binary") or "").endswith("/output/bin/klein_precache_sample_prompts"):
        missing.append("manifest.precache_binary=output/bin/klein_precache_sample_prompts")
    if not Path(str(manifest.get("lora_path") or "")).is_file():
        missing.append("manifest.lora_path exists on disk")
    if not str(manifest.get("output_png") or "").endswith(output_path.name):
        missing.append("manifest.output_png matches report output")
    if log_markers.get("sample_command_has_lora") is not True:
        missing.append("log_markers.sample_command_has_lora=True")
    if log_markers.get("sample_command_has_reference") is not True:
        missing.append("log_markers.sample_command_has_reference=True")
    if log_markers.get("loaded_adapter_count") is not True:
        missing.append("log_markers.loaded_adapter_count=True")
    if log_markers.get("reference_edit") is not True:
        missing.append("log_markers.reference_edit=True")
    if log_markers.get("done_staged_sample") is not True:
        missing.append("log_markers.done_staged_sample=True")

    if missing:
        return Check(
            False,
            P1,
            "lora",
            label,
            "missing evidence: " + ", ".join(missing),
            rel(smoke_path),
            KLEIN_LORA_REFERENCE_DAEMON_SMOKE_ACCEPTANCE,
        )

    return Check(
        True,
        PASS,
        "lora",
        label,
        (
            f"dispatch job {job.get('id')} wrote {rel(output_path)} with "
            f"ReferenceLatent edit plus LoRA {Path(expected_lora).name}"
        ),
        rel(smoke_path),
        KLEIN_LORA_REFERENCE_DAEMON_SMOKE_ACCEPTANCE,
    )


def check_klein_real_image_health_report(report_path: Path) -> Check:
    report = read_json(report_path)
    label = "Klein real image visual health"
    if not report:
        return Check(
            False,
            P1,
            "workflow",
            label,
            f"missing report: {rel(report_path)}",
            rel(report_path),
            KLEIN_REAL_IMAGE_HEALTH_ACCEPTANCE,
        )
    if report.get("ready") is not True:
        return Check(
            False,
            P1,
            "workflow",
            label,
            "report not ready: " + json.dumps(report.get("blockers")),
            rel(report_path),
            KLEIN_REAL_IMAGE_HEALTH_ACCEPTANCE,
        )
    expected_cases = [
        "klein4b_reference_edit",
        "klein9b_reference_edit",
        "klein9b_lora_txt2img",
        "klein9b_lora_reference_edit",
    ]
    cases = dict_or_empty(report.get("cases"))
    missing = []
    details = []
    for case_name in expected_cases:
        case = dict_or_empty(cases.get(case_name))
        health = dict_or_empty(case.get("visual_health"))
        output_path = _evidence_path(case.get("output_path"))
        if not case:
            missing.append(f"cases.{case_name}")
            continue
        if case.get("ready") is not True:
            missing.append(f"{case_name}.ready=True")
        if health.get("ready") is not True:
            missing.append(f"{case_name}.visual_health.ready=True")
        if not output_path.is_file():
            missing.append(f"{case_name}.output_path exists")
        if not isinstance(health.get("gray_stddev"), (int, float)) or health.get("gray_stddev") < 20.0:
            missing.append(f"{case_name}.gray_stddev>=20")
        if not isinstance(health.get("edge_mean"), (int, float)) or health.get("edge_mean") < 8.0:
            missing.append(f"{case_name}.edge_mean>=8")
        if not isinstance(health.get("edge_stddev"), (int, float)) or health.get("edge_stddev") < 20.0:
            missing.append(f"{case_name}.edge_stddev>=20")
        if health.get("blockers") not in ([], None):
            missing.append(f"{case_name}.visual_health.blockers=[]")
        if health:
            details.append(
                f"{case_name} gray_stddev={health.get('gray_stddev')} edge_mean={health.get('edge_mean')}"
            )
    if missing:
        return Check(
            False,
            P1,
            "workflow",
            label,
            "missing evidence: " + ", ".join(missing),
            rel(report_path),
            KLEIN_REAL_IMAGE_HEALTH_ACCEPTANCE,
        )
    return Check(
        True,
        PASS,
        "workflow",
        label,
        "; ".join(details),
        rel(report_path),
        KLEIN_REAL_IMAGE_HEALTH_ACCEPTANCE,
    )


def check_lanpaint_oracle_surface_report(report_path: Path) -> Check:
    report = read_json(report_path)
    if not report:
        return Check(
            False,
            P1,
            "workflow",
            "LanPaint oracle surface report",
            f"missing report: {rel(report_path)}",
            rel(report_path),
            LANPAINT_ORACLE_SURFACE_ACCEPTANCE,
        )
    if report.get("ready") is not True:
        return Check(
            False,
            P1,
            "workflow",
            "LanPaint oracle surface report",
            "report not ready: " + json.dumps(report.get("blockers")),
            rel(report_path),
            LANPAINT_ORACLE_SURFACE_ACCEPTANCE,
        )
    workflows = dict_or_empty(dict_or_empty(report.get("oracle")).get("workflows"))
    missing = []
    for case_name in ["zimage_inpaint", "qwen_image_inpaint", "flux2_klein_inpainting"]:
        case = dict_or_empty(workflows.get(case_name))
        if not case:
            missing.append(f"oracle.workflows.{case_name}")
            continue
        counts = dict_or_empty(case.get("node_type_counts"))
        if case_name == "flux2_klein_inpainting":
            if counts.get("LanPaint_SamplerCustomAdvanced") != 1:
                missing.append(f"{case_name}.LanPaint_SamplerCustomAdvanced=1")
        else:
            if counts.get("LanPaint_KSampler") != 1:
                missing.append(f"{case_name}.LanPaint_KSampler=1")
        if counts.get("LanPaint_MaskBlend") != 1:
            missing.append(f"{case_name}.LanPaint_MaskBlend=1")
        if case.get("mask_blend_overlap") != 9:
            missing.append(f"{case_name}.mask_blend_overlap=9")
        if case.get("image_to_mask_channel") != "red":
            missing.append(f"{case_name}.image_to_mask_channel='red'")
    area_resize = dict_or_empty(dict_or_empty(report.get("oracle")).get("pytorch_area_resize"))
    if area_resize.get("role") != LANPAINT_AREA_RESIZE_ROLE:
        missing.append("oracle.pytorch_area_resize.role")
    if area_resize.get("cases_passed") != 3:
        missing.append("oracle.pytorch_area_resize.cases_passed=3")
    if missing:
        return Check(
            False,
            P1,
            "workflow",
            "LanPaint oracle surface report",
            "missing evidence: " + ", ".join(missing),
            rel(report_path),
            LANPAINT_ORACLE_SURFACE_ACCEPTANCE,
        )
    return Check(
        True,
        PASS,
        "workflow",
        "LanPaint oracle surface report",
        "oracle workflows, Python/Comfy mask semantics, and PyTorch area resize cases are pinned; source checks pin bounded Z-Image MaskBlend markers",
        rel(report_path),
        LANPAINT_ORACLE_SURFACE_ACCEPTANCE,
    )


def check_lanpaint_canvas_daemon_smoke_report(report_path: Path) -> Check:
    report = read_json(report_path)
    if not report:
        return Check(
            False,
            P1,
            "workflow",
            "LanPaint canvas daemon smoke",
            f"missing report: {rel(report_path)}",
            rel(report_path),
            LANPAINT_CANVAS_DAEMON_SMOKE_ACCEPTANCE,
        )
    if report.get("ready") is not True:
        return Check(
            False,
            P1,
            "workflow",
            "LanPaint canvas daemon smoke",
            "report not ready: " + json.dumps(report.get("blockers")),
            rel(report_path),
            LANPAINT_CANVAS_DAEMON_SMOKE_ACCEPTANCE,
        )
    job = dict_or_empty(report.get("job"))
    png = dict_or_empty(report.get("png"))
    genparams = dict_or_empty(png.get("genparams"))
    expected = {
        "workflow_source": "comfy_ui_canvas_graph",
        "workflow_node_count": 11,
        "workflow_edge_count": 17,
        "model": "animagineXL40_v4Opt.safetensors",
        "steps": 30,
        "seed": 0,
        "cfg": 5.0,
        "sampler": "euler",
        "scheduler": "karras",
        "lanpaint_num_steps": 5,
        "lanpaint_prompt_mode": "Image First",
        "lanpaint_mask_blend_overlap": 9,
    }
    missing = [
        f"genparams.{key}={value!r}"
        for key, value in expected.items()
        if genparams.get(key) != value
    ]
    if genparams.get("init_image") != genparams.get("mask_image"):
        missing.append("init_image equals mask_image for SDXL LanPaint smoke")
    if job.get("state") != "done":
        missing.append("job.state='done'")
    if not png.get("idat_sha256"):
        missing.append("png.idat_sha256")
    if missing:
        return Check(
            False,
            P1,
            "workflow",
            "LanPaint canvas daemon smoke",
            "missing evidence: " + ", ".join(missing),
            rel(report_path),
            LANPAINT_CANVAS_DAEMON_SMOKE_ACCEPTANCE,
        )
    return Check(
        True,
        PASS,
        "workflow",
        "LanPaint canvas daemon smoke",
        f"LanPaint canvas completed {job.get('id')} with PNG metadata",
        rel(report_path),
        LANPAINT_CANVAS_DAEMON_SMOKE_ACCEPTANCE,
    )


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
            WORKFLOW_GRAPH_SMOKE_ACCEPTANCE,
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
            WORKFLOW_GRAPH_SMOKE_ACCEPTANCE,
        )
    job = report.get("job")
    png = report.get("png")
    api_job = report.get("comfy_api_job")
    api_png = report.get("comfy_api_png")
    reroute_api_job = report.get("reroute_api_job")
    reroute_api_png = report.get("reroute_api_png")
    reroute_canvas_job = report.get("reroute_canvas_job")
    reroute_canvas_png = report.get("reroute_canvas_png")
    getset_canvas_job = report.get("getset_canvas_job")
    getset_canvas_png = report.get("getset_canvas_png")
    scalar_canvas_job = report.get("scalar_canvas_job")
    scalar_canvas_png = report.get("scalar_canvas_png")
    ui_drop_canvas_job = report.get("ui_drop_canvas_job")
    ui_drop_canvas_png = report.get("ui_drop_canvas_png")
    switch_canvas_job = report.get("switch_canvas_job")
    switch_canvas_png = report.get("switch_canvas_png")
    outpaint_threshold_api_job = report.get("outpaint_threshold_api_job")
    outpaint_threshold_api_png = report.get("outpaint_threshold_api_png")
    inpaint_conditioning_api_job = report.get("inpaint_conditioning_api_job")
    inpaint_conditioning_api_png = report.get("inpaint_conditioning_api_png")
    inpaint_conditioning_no_noise_mask_api_job = report.get("inpaint_conditioning_no_noise_mask_api_job")
    inpaint_conditioning_no_noise_mask_api_png = report.get("inpaint_conditioning_no_noise_mask_api_png")
    conditioning_set_mask_api_job = report.get("conditioning_set_mask_api_job")
    conditioning_set_mask_api_png = report.get("conditioning_set_mask_api_png")
    img_job = report.get("img2img_job")
    img_png = report.get("img2img_png")
    lora_job = report.get("lora_job")
    lora_png = report.get("lora_png")
    zimage_lora_alias_job = report.get("zimage_lora_alias_job")
    zimage_lora_alias_png = report.get("zimage_lora_alias_png")
    mask_job = report.get("mask_job")
    mask_png = report.get("mask_png")
    outpaint_preprocess_job = report.get("outpaint_preprocess_job")
    outpaint_preprocess_png = report.get("outpaint_preprocess_png")
    basic_scheduler_job = report.get("basic_scheduler_job")
    basic_scheduler_png = report.get("basic_scheduler_png")
    sf_evidence = serenityflow_t2i_case(report, "zimage_t2i")
    edit_klein9b_evidence = serenityflow_edit_case(report, "klein9b_edit")
    edit_klein4b_evidence = serenityflow_edit_case(report, "klein4b_edit")
    qwen_edit_evidence = serenityflow_qwen_edit_case(report, "qwen_edit")
    qwen_edit_lora_evidence = serenityflow_qwen_edit_case(report, "qwen_edit_lora")
    ideogram4_evidence = prefixed_evidence(report, "ideogram4_visual_export")
    unsupported_api = report.get("unsupported_comfy_api_node")
    lora_clip_unsupported = report.get("lora_clip_unsupported")
    reroute_missing_input = report.get("reroute_missing_input")
    getset_duplicate_setnode = report.get("getset_duplicate_setnode")
    getset_missing_setnode = report.get("getset_missing_setnode")
    getset_missing_input = report.get("getset_missing_input")
    getset_unsupported_type = report.get("getset_unsupported_type")
    getset_type_mismatch = report.get("getset_type_mismatch")
    scalar_type_mismatch = report.get("scalar_type_mismatch")
    preview_type_mismatch = report.get("preview_type_mismatch")
    switch_type_mismatch = report.get("switch_type_mismatch")
    inpaint_conditioning_missing_mask = report.get("inpaint_conditioning_missing_mask")
    conditioning_set_mask_missing_mask = report.get("conditioning_set_mask_missing_mask")
    if (
        not isinstance(job, dict)
        or not isinstance(png, dict)
        or not isinstance(api_job, dict)
        or not isinstance(api_png, dict)
        or not isinstance(reroute_api_job, dict)
        or not isinstance(reroute_api_png, dict)
        or not isinstance(reroute_canvas_job, dict)
        or not isinstance(reroute_canvas_png, dict)
        or not isinstance(getset_canvas_job, dict)
        or not isinstance(getset_canvas_png, dict)
        or not isinstance(scalar_canvas_job, dict)
        or not isinstance(scalar_canvas_png, dict)
        or not isinstance(ui_drop_canvas_job, dict)
        or not isinstance(ui_drop_canvas_png, dict)
        or not isinstance(switch_canvas_job, dict)
        or not isinstance(switch_canvas_png, dict)
        or not isinstance(outpaint_threshold_api_job, dict)
        or not isinstance(outpaint_threshold_api_png, dict)
        or not isinstance(inpaint_conditioning_api_job, dict)
        or not isinstance(inpaint_conditioning_api_png, dict)
        or not isinstance(inpaint_conditioning_no_noise_mask_api_job, dict)
        or not isinstance(inpaint_conditioning_no_noise_mask_api_png, dict)
        or not isinstance(conditioning_set_mask_api_job, dict)
        or not isinstance(conditioning_set_mask_api_png, dict)
        or not isinstance(img_job, dict)
        or not isinstance(img_png, dict)
        or not isinstance(lora_job, dict)
        or not isinstance(lora_png, dict)
        or not isinstance(zimage_lora_alias_job, dict)
        or not isinstance(zimage_lora_alias_png, dict)
        or not isinstance(mask_job, dict)
        or not isinstance(mask_png, dict)
        or not isinstance(outpaint_preprocess_job, dict)
        or not isinstance(outpaint_preprocess_png, dict)
        or not isinstance(basic_scheduler_job, dict)
        or not isinstance(basic_scheduler_png, dict)
        or not isinstance(unsupported_api, dict)
        or not isinstance(lora_clip_unsupported, dict)
        or not isinstance(reroute_missing_input, dict)
        or not isinstance(getset_duplicate_setnode, dict)
        or not isinstance(getset_missing_setnode, dict)
        or not isinstance(getset_missing_input, dict)
        or not isinstance(getset_unsupported_type, dict)
        or not isinstance(getset_type_mismatch, dict)
        or not isinstance(scalar_type_mismatch, dict)
        or not isinstance(preview_type_mismatch, dict)
        or not isinstance(switch_type_mismatch, dict)
        or not isinstance(inpaint_conditioning_missing_mask, dict)
        or not isinstance(conditioning_set_mask_missing_mask, dict)
    ):
        return Check(
            False,
            P1,
            "workflow",
            "typed workflow graph product smoke",
            "report missing linked graph, Comfy API prompt, Reroute API/canvas import, Get/Set canvas import, scalar canvas import, UI/drop canvas import, ComfySwitchNode canvas import, outpaint ThresholdMask API import, InpaintModelConditioning API import, ConditioningSetMask API import, Qwen edit import, LoRA, ZImageLoraModelOnly, LoRA CLIP reject, img2img, mask, outpaint preprocessing, BasicScheduler, or unsupported-node evidence",
            rel(WORKFLOW_GRAPH_PRODUCT),
            WORKFLOW_GRAPH_SMOKE_ACCEPTANCE,
        )
    if lora_clip_unsupported.get("status") != 501:
        return Check(
            False,
            P1,
            "workflow",
            "typed workflow graph product smoke",
            "LoraLoader CLIP-side unsupported report did not return HTTP 501",
            rel(WORKFLOW_GRAPH_PRODUCT),
            WORKFLOW_GRAPH_SMOKE_ACCEPTANCE,
        )
    if inpaint_conditioning_missing_mask.get("status") != 501:
        return Check(
            False,
            P1,
            "workflow",
            "typed workflow graph product smoke",
            "InpaintModelConditioning missing-mask unsupported report did not return HTTP 501",
            rel(WORKFLOW_GRAPH_PRODUCT),
            WORKFLOW_GRAPH_SMOKE_ACCEPTANCE,
        )
    if conditioning_set_mask_missing_mask.get("status") != 501:
        return Check(
            False,
            P1,
            "workflow",
            "typed workflow graph product smoke",
            "ConditioningSetMask missing-mask unsupported report did not return HTTP 501",
            rel(WORKFLOW_GRAPH_PRODUCT),
            WORKFLOW_GRAPH_SMOKE_ACCEPTANCE,
        )
    if reroute_missing_input.get("status") != 501:
        return Check(
            False,
            P1,
            "workflow",
            "typed workflow graph product smoke",
            "Reroute missing-input unsupported report did not return HTTP 501",
            rel(WORKFLOW_GRAPH_PRODUCT),
            WORKFLOW_GRAPH_SMOKE_ACCEPTANCE,
        )
    for label, item in (
        ("Get/Set duplicate SetNode", getset_duplicate_setnode),
        ("Get/Set missing SetNode", getset_missing_setnode),
        ("Get/Set missing SetNode input", getset_missing_input),
        ("Get/Set unsupported bus type", getset_unsupported_type),
        ("Get/Set output type mismatch", getset_type_mismatch),
        ("scalar consumer type mismatch", scalar_type_mismatch),
        ("PreviewImage type mismatch", preview_type_mismatch),
        ("ComfySwitchNode selected type mismatch", switch_type_mismatch),
    ):
        if item.get("status") != 501:
            return Check(
                False,
                P1,
                "workflow",
                "typed workflow graph product smoke",
                f"{label} unsupported report did not return HTTP 501",
                rel(WORKFLOW_GRAPH_PRODUCT),
                WORKFLOW_GRAPH_SMOKE_ACCEPTANCE,
            )
    linked_genparams = png.get("genparams")
    api_genparams = api_png.get("genparams")
    if (
        not isinstance(linked_genparams, dict)
        or linked_genparams.get("workflow_save_prefix") != "typed-graph"
        or not isinstance(api_genparams, dict)
        or api_genparams.get("workflow_save_prefix") != "comfy-api"
    ):
        return Check(
            False,
            P1,
            "workflow",
            "typed workflow graph product smoke",
            "SaveImage filename_prefix metadata missing from typed or Comfy API product report",
            rel(WORKFLOW_GRAPH_PRODUCT),
            WORKFLOW_GRAPH_SMOKE_ACCEPTANCE,
        )
    lora_genparams = lora_png.get("genparams")
    lora_items = lora_genparams.get("lora") if isinstance(lora_genparams, dict) else None
    if (
        not isinstance(lora_items, list)
        or len(lora_items) != 1
        or not isinstance(lora_items[0], dict)
        or lora_items[0].get("name") != "graph_lora.safetensors"
        or lora_items[0].get("weight") != 0.8
    ):
        return Check(
            False,
            P1,
            "workflow",
            "typed workflow graph product smoke",
            "LoraLoader metadata missing from product report",
            rel(WORKFLOW_GRAPH_PRODUCT),
            WORKFLOW_GRAPH_SMOKE_ACCEPTANCE,
        )
    z_lora_genparams = zimage_lora_alias_png.get("genparams")
    z_lora_items = z_lora_genparams.get("lora") if isinstance(z_lora_genparams, dict) else None
    if (
        not isinstance(z_lora_items, list)
        or len(z_lora_items) != 2
        or not isinstance(z_lora_items[0], dict)
        or not isinstance(z_lora_items[1], dict)
        or z_lora_items[0].get("name") != "zimage_first.safetensors"
        or z_lora_items[0].get("weight") != 0.65
        or z_lora_items[1].get("name") != "zimage_second.safetensors"
        or z_lora_items[1].get("weight") != 0.4
    ):
        return Check(
            False,
            P1,
            "workflow",
            "typed workflow graph product smoke",
            "ZImageLoraModelOnly metadata missing from product report",
            rel(WORKFLOW_GRAPH_PRODUCT),
            WORKFLOW_GRAPH_SMOKE_ACCEPTANCE,
        )
    img_genparams = img_png.get("genparams")
    if (
        not isinstance(img_genparams, dict)
        or img_genparams.get("init_image") != "/tmp/serenity_graph_init.png"
    ):
        return Check(
            False,
            P1,
            "workflow",
            "typed workflow graph product smoke",
            "img2img LoadImage/VAEEncode metadata missing from product report",
            rel(WORKFLOW_GRAPH_PRODUCT),
            WORKFLOW_GRAPH_SMOKE_ACCEPTANCE,
        )
    mask_genparams = mask_png.get("genparams")
    if (
        not isinstance(mask_genparams, dict)
        or mask_genparams.get("init_image") != "/tmp/serenity_graph_init.png"
        or mask_genparams.get("mask_image") != "/tmp/serenity_graph_mask.png"
    ):
        return Check(
            False,
            P1,
            "workflow",
            "typed workflow graph product smoke",
            "SetLatentNoiseMask metadata missing from product report",
            rel(WORKFLOW_GRAPH_PRODUCT),
            WORKFLOW_GRAPH_SMOKE_ACCEPTANCE,
        )
    outpaint_genparams = outpaint_preprocess_png.get("genparams")
    if (
        not isinstance(outpaint_genparams, dict)
        or outpaint_genparams.get("init_image") != "/tmp/serenity_outpaint_source.png"
        or outpaint_genparams.get("mask_image") != "/tmp/serenity_outpaint_source.png"
        or outpaint_genparams.get("lanpaint_mask_channel") != "image_pad_for_outpaint"
        or outpaint_genparams.get("outpaint_left") != 200
        or outpaint_genparams.get("outpaint_top") != 200
        or outpaint_genparams.get("outpaint_right") != 200
        or outpaint_genparams.get("outpaint_bottom") != 200
        or outpaint_genparams.get("outpaint_feathering") != 20
        or outpaint_genparams.get("threshold_mask_value") != 0.01
        or outpaint_genparams.get("threshold_mask_operator") != "gt"
        or outpaint_genparams.get("lanpaint_mask_blend_overlap") != 9
    ):
        return Check(
            False,
            P1,
            "workflow",
            "typed workflow graph product smoke",
            "ImagePadForOutpaint/ThresholdMask metadata missing from product report",
            rel(WORKFLOW_GRAPH_PRODUCT),
            WORKFLOW_GRAPH_SMOKE_ACCEPTANCE,
        )
    api_genparams = api_png.get("genparams")
    if (
        not isinstance(api_genparams, dict)
        or api_genparams.get("workflow_source") != "comfy_api_prompt_graph"
    ):
        return Check(
            False,
            P1,
            "workflow",
            "typed workflow graph product smoke",
            "Comfy API prompt graph metadata missing from product report",
            rel(WORKFLOW_GRAPH_PRODUCT),
            WORKFLOW_GRAPH_SMOKE_ACCEPTANCE,
        )
    reroute_api_genparams = reroute_api_png.get("genparams")
    if (
        not isinstance(reroute_api_genparams, dict)
        or reroute_api_genparams.get("workflow_source") != "comfy_api_prompt_graph"
        or reroute_api_genparams.get("workflow_save_prefix") != "reroute-api"
        or reroute_api_genparams.get("prompt") != "reroute api positive prompt"
        or reroute_api_genparams.get("negative") != "reroute api negative prompt"
        or reroute_api_genparams.get("model") != "stub"
        or reroute_api_genparams.get("width") != 832
        or reroute_api_genparams.get("height") != 512
        or reroute_api_genparams.get("seed") != 88901
        or reroute_api_genparams.get("workflow_node_count") != 11
        or reroute_api_genparams.get("workflow_edge_count") != 13
    ):
        return Check(
            False,
            P1,
            "workflow",
            "typed workflow graph product smoke",
            "Reroute Comfy API import metadata missing from product report",
            rel(WORKFLOW_GRAPH_PRODUCT),
            WORKFLOW_GRAPH_SMOKE_ACCEPTANCE,
        )
    reroute_canvas_genparams = reroute_canvas_png.get("genparams")
    if (
        not isinstance(reroute_canvas_genparams, dict)
        or reroute_canvas_genparams.get("workflow_source") != "comfy_ui_canvas_graph"
        or reroute_canvas_genparams.get("workflow_save_prefix") != "canvas-reroute"
        or reroute_canvas_genparams.get("prompt") != "reroute canvas positive prompt"
        or reroute_canvas_genparams.get("negative") != "reroute canvas negative prompt"
        or reroute_canvas_genparams.get("model") != "stub"
        or reroute_canvas_genparams.get("init_image") != "/tmp/serenity_canvas_reroute.png"
        or reroute_canvas_genparams.get("seed") != 99012
        or reroute_canvas_genparams.get("workflow_node_count") != 9
        or reroute_canvas_genparams.get("workflow_edge_count") != 12
    ):
        return Check(
            False,
            P1,
            "workflow",
            "typed workflow graph product smoke",
            "Reroute Comfy UI canvas import metadata missing from product report",
            rel(WORKFLOW_GRAPH_PRODUCT),
            WORKFLOW_GRAPH_SMOKE_ACCEPTANCE,
        )
    getset_canvas_genparams = getset_canvas_png.get("genparams")
    if (
        not isinstance(getset_canvas_genparams, dict)
        or getset_canvas_genparams.get("workflow_source") != "comfy_ui_canvas_graph"
        or getset_canvas_genparams.get("workflow_save_prefix") != "canvas-getset"
        or getset_canvas_genparams.get("prompt") != "getset canvas positive prompt"
        or getset_canvas_genparams.get("negative") != "getset canvas negative prompt"
        or getset_canvas_genparams.get("model") != "stub"
        or getset_canvas_genparams.get("init_image") != "/tmp/serenity_getset_image.png"
        or getset_canvas_genparams.get("seed") != 12321
        or getset_canvas_genparams.get("workflow_node_count") != 16
        or getset_canvas_genparams.get("workflow_edge_count") != 15
    ):
        return Check(
            False,
            P1,
            "workflow",
            "typed workflow graph product smoke",
            "Get/Set Comfy UI canvas import metadata missing from product report",
            rel(WORKFLOW_GRAPH_PRODUCT),
            WORKFLOW_GRAPH_SMOKE_ACCEPTANCE,
        )
    scalar_canvas_genparams = scalar_canvas_png.get("genparams")
    if (
        not isinstance(scalar_canvas_genparams, dict)
        or scalar_canvas_genparams.get("workflow_source") != "comfy_ui_canvas_graph"
        or scalar_canvas_genparams.get("workflow_save_prefix") != "scalar-canvas-prefix"
        or scalar_canvas_genparams.get("prompt") != "scalar linked positive prompt"
        or scalar_canvas_genparams.get("negative") != "scalar canvas negative prompt"
        or scalar_canvas_genparams.get("model") != "stub"
        or scalar_canvas_genparams.get("width") != 672
        or scalar_canvas_genparams.get("height") != 544
        or scalar_canvas_genparams.get("images") != 1
        or scalar_canvas_genparams.get("steps") != 6
        or scalar_canvas_genparams.get("seed") != 24680
        or scalar_canvas_genparams.get("cfg") != 2.125
        or scalar_canvas_genparams.get("sampler") != "euler"
        or scalar_canvas_genparams.get("scheduler") != "simple"
        or scalar_canvas_genparams.get("creativity") != 0.625
        or scalar_canvas_genparams.get("workflow_node_count") != 20
        or scalar_canvas_genparams.get("workflow_edge_count") != 21
    ):
        return Check(
            False,
            P1,
            "workflow",
            "typed workflow graph product smoke",
            "scalar Comfy UI canvas primitive/link metadata missing from product report",
            rel(WORKFLOW_GRAPH_PRODUCT),
            WORKFLOW_GRAPH_SMOKE_ACCEPTANCE,
        )
    ui_drop_genparams = ui_drop_canvas_png.get("genparams")
    if (
        not isinstance(ui_drop_genparams, dict)
        or ui_drop_genparams.get("workflow_source") != "comfy_ui_canvas_graph"
        or ui_drop_genparams.get("workflow_save_prefix") != "ui-drop-canvas"
        or ui_drop_genparams.get("prompt") != "ui drop positive prompt"
        or ui_drop_genparams.get("negative") != "ui drop negative prompt"
        or ui_drop_genparams.get("model") != "stub"
        or ui_drop_genparams.get("width") != 576
        or ui_drop_genparams.get("height") != 448
        or ui_drop_genparams.get("images") != 1
        or ui_drop_genparams.get("steps") != 4
        or ui_drop_genparams.get("seed") != 13579
        or ui_drop_genparams.get("cfg") != 2.25
        or ui_drop_genparams.get("sampler") != "euler"
        or ui_drop_genparams.get("scheduler") != "simple"
        or ui_drop_genparams.get("creativity") != 0.6
        or ui_drop_genparams.get("workflow_node_count") != 10
        or ui_drop_genparams.get("workflow_edge_count") != 10
    ):
        return Check(
            False,
            P1,
            "workflow",
            "typed workflow graph product smoke",
            "UI-only Note/MarkdownNote/PreviewImage canvas metadata missing from product report",
            rel(WORKFLOW_GRAPH_PRODUCT),
            WORKFLOW_GRAPH_SMOKE_ACCEPTANCE,
        )
    switch_genparams = switch_canvas_png.get("genparams")
    if (
        not isinstance(switch_genparams, dict)
        or switch_genparams.get("workflow_source") != "comfy_ui_canvas_graph"
        or switch_genparams.get("workflow_save_prefix") != "switch-canvas"
        or switch_genparams.get("prompt") != "switch canvas positive prompt"
        or switch_genparams.get("negative") != "switch canvas negative prompt"
        or switch_genparams.get("model") != "stub"
        or switch_genparams.get("width") != 608
        or switch_genparams.get("height") != 512
        or switch_genparams.get("images") != 1
        or switch_genparams.get("steps") != 6
        or switch_genparams.get("seed") != 97531
        or switch_genparams.get("cfg") != 2.875
        or switch_genparams.get("sampler") != "euler"
        or switch_genparams.get("scheduler") != "simple"
        or switch_genparams.get("creativity") != 0.7
        or switch_genparams.get("workflow_node_count") != 18
        or switch_genparams.get("workflow_edge_count") != 20
    ):
        return Check(
            False,
            P1,
            "workflow",
            "typed workflow graph product smoke",
            "ComfySwitchNode selected-branch metadata missing from product report",
            rel(WORKFLOW_GRAPH_PRODUCT),
            WORKFLOW_GRAPH_SMOKE_ACCEPTANCE,
        )
    outpaint_api_genparams = outpaint_threshold_api_png.get("genparams")
    if (
        not isinstance(outpaint_api_genparams, dict)
        or outpaint_api_genparams.get("workflow_source") != "comfy_api_prompt_graph"
        or outpaint_api_genparams.get("workflow_save_prefix") != "outpaint-threshold-graph"
        or outpaint_api_genparams.get("init_image") != "/tmp/serenity_graph_init.png"
        or outpaint_api_genparams.get("mask_image") != "/tmp/serenity_graph_init.png"
        or outpaint_api_genparams.get("lanpaint_mask_channel") != "image_pad_for_outpaint"
        or outpaint_api_genparams.get("outpaint_left") != 16
        or outpaint_api_genparams.get("outpaint_top") != 8
        or outpaint_api_genparams.get("outpaint_right") != 16
        or outpaint_api_genparams.get("outpaint_bottom") != 8
        or outpaint_api_genparams.get("outpaint_feathering") != 0
        or outpaint_api_genparams.get("threshold_mask_value") != 0.5
        or outpaint_api_genparams.get("threshold_mask_operator") != "gt"
        or outpaint_api_genparams.get("workflow_node_count") != 11
        or outpaint_api_genparams.get("workflow_edge_count") != 15
    ):
        return Check(
            False,
            P1,
            "workflow",
            "typed workflow graph product smoke",
            "ImagePadForOutpaint/ThresholdMask Comfy API import metadata missing from product report",
            rel(WORKFLOW_GRAPH_PRODUCT),
            WORKFLOW_GRAPH_SMOKE_ACCEPTANCE,
        )
    inpaint_genparams = inpaint_conditioning_api_png.get("genparams")
    if (
        not isinstance(inpaint_genparams, dict)
        or inpaint_genparams.get("workflow_source") != "comfy_api_prompt_graph"
        or inpaint_genparams.get("workflow_save_prefix") != "inpaint-conditioning-graph"
        or inpaint_genparams.get("prompt") != "inpaint conditioning positive prompt"
        or inpaint_genparams.get("negative") != "inpaint conditioning negative prompt"
        or inpaint_genparams.get("init_image") != "/tmp/serenity_inpaint_conditioning.png"
        or inpaint_genparams.get("mask_image") != "/tmp/serenity_inpaint_conditioning.png"
        or inpaint_genparams.get("lanpaint_mask_channel") != "load_image_mask"
        or inpaint_genparams.get("inpaint_conditioning_image") != "/tmp/serenity_inpaint_conditioning.png"
        or inpaint_genparams.get("inpaint_conditioning_mask") != "/tmp/serenity_inpaint_conditioning.png"
        or inpaint_genparams.get("inpaint_conditioning_noise_mask") is not True
        or inpaint_genparams.get("workflow_node_count") != 8
        or inpaint_genparams.get("workflow_edge_count") != 14
    ):
        return Check(
            False,
            P1,
            "workflow",
            "typed workflow graph product smoke",
            "InpaintModelConditioning default-noise-mask Comfy API import metadata missing from product report",
            rel(WORKFLOW_GRAPH_PRODUCT),
            WORKFLOW_GRAPH_SMOKE_ACCEPTANCE,
        )
    inpaint_no_noise_genparams = inpaint_conditioning_no_noise_mask_api_png.get("genparams")
    if (
        not isinstance(inpaint_no_noise_genparams, dict)
        or inpaint_no_noise_genparams.get("workflow_source") != "comfy_api_prompt_graph"
        or inpaint_no_noise_genparams.get("workflow_save_prefix") != "inpaint-conditioning-no-noise-mask"
        or inpaint_no_noise_genparams.get("init_image") != "/tmp/serenity_inpaint_conditioning.png"
        or inpaint_no_noise_genparams.get("mask_image") != ""
        or inpaint_no_noise_genparams.get("lanpaint_mask_channel") != ""
        or inpaint_no_noise_genparams.get("inpaint_conditioning_image") != "/tmp/serenity_inpaint_conditioning.png"
        or inpaint_no_noise_genparams.get("inpaint_conditioning_mask") != "/tmp/serenity_inpaint_conditioning.png"
        or inpaint_no_noise_genparams.get("inpaint_conditioning_noise_mask") is not False
        or inpaint_no_noise_genparams.get("workflow_node_count") != 8
        or inpaint_no_noise_genparams.get("workflow_edge_count") != 14
    ):
        return Check(
            False,
            P1,
            "workflow",
            "typed workflow graph product smoke",
            "InpaintModelConditioning noise_mask=false Comfy API import metadata missing or collapsed into mask_image",
            rel(WORKFLOW_GRAPH_PRODUCT),
            WORKFLOW_GRAPH_SMOKE_ACCEPTANCE,
        )
    conditioning_mask_genparams = conditioning_set_mask_api_png.get("genparams")
    if (
        not isinstance(conditioning_mask_genparams, dict)
        or conditioning_mask_genparams.get("workflow_source") != "comfy_api_prompt_graph"
        or conditioning_mask_genparams.get("workflow_save_prefix") != "conditioning-mask-graph"
        or conditioning_mask_genparams.get("prompt") != "conditioning mask positive prompt"
        or conditioning_mask_genparams.get("negative") != "conditioning mask negative prompt"
        or conditioning_mask_genparams.get("conditioning_mask_image") != "/tmp/serenity_conditioning_mask.png"
        or conditioning_mask_genparams.get("conditioning_mask_channel") != "load_image_mask"
        or conditioning_mask_genparams.get("conditioning_mask_strength") != 0.42
        or conditioning_mask_genparams.get("conditioning_mask_set_area_to_bounds") is not True
        or conditioning_mask_genparams.get("mask_image") != ""
        or conditioning_mask_genparams.get("lanpaint_mask_channel") != ""
        or conditioning_mask_genparams.get("workflow_node_count") != 9
        or conditioning_mask_genparams.get("workflow_edge_count") != 11
    ):
        return Check(
            False,
            P1,
            "workflow",
            "typed workflow graph product smoke",
            "ConditioningSetMask Comfy API import metadata missing or collapsed into latent mask_image",
            rel(WORKFLOW_GRAPH_PRODUCT),
            WORKFLOW_GRAPH_SMOKE_ACCEPTANCE,
        )
    basic_scheduler_genparams = basic_scheduler_png.get("genparams")
    if (
        not isinstance(basic_scheduler_genparams, dict)
        or basic_scheduler_genparams.get("scheduler") != "simple"
        or basic_scheduler_genparams.get("steps") != 8
        or basic_scheduler_genparams.get("creativity") != 0.33
        or basic_scheduler_genparams.get("sampler") != "euler"
        or basic_scheduler_genparams.get("seed") != 67890
        or basic_scheduler_genparams.get("workflow_node_count") != 14
        or basic_scheduler_genparams.get("workflow_edge_count") != 16
    ):
        return Check(
            False,
            P1,
            "workflow",
            "typed workflow graph product smoke",
            "BasicScheduler/SamplerCustomAdvanced metadata missing from product report",
            rel(WORKFLOW_GRAPH_PRODUCT),
            WORKFLOW_GRAPH_SMOKE_ACCEPTANCE,
        )
    sf_error = validate_zimage_t2i_evidence(sf_evidence)
    if sf_error:
        return Check(
            False,
            P1,
            "workflow",
            "typed workflow graph product smoke",
            sf_error,
            rel(WORKFLOW_GRAPH_PRODUCT),
            WORKFLOW_GRAPH_SMOKE_ACCEPTANCE,
        )
    edit_errors = [
        validate_klein_edit_evidence(edit_klein9b_evidence, "klein9b_edit", "flux2-klein-9b.safetensors"),
        validate_klein_edit_evidence(edit_klein4b_evidence, "klein4b_edit", "flux2-klein-4b.safetensors"),
    ]
    for edit_error in edit_errors:
        if edit_error:
            return Check(
                False,
                P1,
                "workflow",
                "typed workflow graph product smoke",
                edit_error,
                rel(WORKFLOW_GRAPH_PRODUCT),
                WORKFLOW_GRAPH_SMOKE_ACCEPTANCE,
            )
    qwen_edit_errors = [
        validate_qwen_edit_evidence(qwen_edit_evidence, "qwen_edit", False),
        validate_qwen_edit_evidence(qwen_edit_lora_evidence, "qwen_edit_lora", True),
    ]
    for qwen_edit_error in qwen_edit_errors:
        if qwen_edit_error:
            return Check(
                False,
                P1,
                "workflow",
                "typed workflow graph product smoke",
                qwen_edit_error,
                rel(WORKFLOW_GRAPH_PRODUCT),
                WORKFLOW_GRAPH_SMOKE_ACCEPTANCE,
            )
    ideogram4_error = validate_ideogram4_visual_export_evidence(ideogram4_evidence)
    if ideogram4_error:
        return Check(
            False,
            P1,
            "workflow",
            "typed workflow graph product smoke",
            ideogram4_error,
            rel(WORKFLOW_GRAPH_PRODUCT),
            WORKFLOW_GRAPH_SMOKE_ACCEPTANCE,
        )
    sf_job = dict_or_empty(sf_evidence.get("job"))
    edit_jobs = [
        ("klein9b_edit", dict_or_empty(edit_klein9b_evidence.get("job"))),
        ("klein4b_edit", dict_or_empty(edit_klein4b_evidence.get("job"))),
    ]
    qwen_edit_jobs = [
        ("qwen_edit", dict_or_empty(qwen_edit_evidence.get("job"))),
        ("qwen_edit_lora", dict_or_empty(qwen_edit_lora_evidence.get("job"))),
    ]
    ideogram4_job = dict_or_empty(ideogram4_evidence.get("job"))
    detail = (
        f"linked graph completed {job.get('id')}; "
        f"img2img graph completed {img_job.get('id')}; "
        f"LoRA graph completed {lora_job.get('id')}; "
        f"mask graph completed {mask_job.get('id')}; "
        f"outpaint preprocess graph completed {outpaint_preprocess_job.get('id')}; "
        f"BasicScheduler graph completed {basic_scheduler_job.get('id')}; "
        f"Comfy API prompt completed {api_job.get('id')}; "
        f"Reroute API completed {reroute_api_job.get('id')}; "
        f"Reroute canvas completed {reroute_canvas_job.get('id')}; "
        f"Get/Set canvas completed {getset_canvas_job.get('id')}; "
        f"scalar canvas completed {scalar_canvas_job.get('id')}; "
        f"UI/drop canvas completed {ui_drop_canvas_job.get('id')}; "
        f"ComfySwitchNode canvas completed {switch_canvas_job.get('id')}; "
        f"outpaint ThresholdMask API completed {outpaint_threshold_api_job.get('id')}; "
        f"InpaintModelConditioning API completed {inpaint_conditioning_api_job.get('id')}; "
        f"InpaintModelConditioning noise_mask=false API completed {inpaint_conditioning_no_noise_mask_api_job.get('id')}; "
        f"ConditioningSetMask API completed {conditioning_set_mask_api_job.get('id')}; "
        "unsupported and wrong-type links returned HTTP 501"
    )
    sf_cases = report.get("serenityflow_t2i")
    ignored_qwen_historical = False
    if isinstance(sf_cases, dict):
        completed_cases = []
        for case_name in sorted(sf_cases):
            if "qwen" in case_name.lower():
                ignored_qwen_historical = True
                continue
            case = sf_cases.get(case_name)
            if isinstance(case, dict):
                case_job = dict_or_empty(case.get("job"))
                if case_job.get("id"):
                    completed_cases.append(f"{case_name} {case_job.get('id')}")
        if completed_cases:
            detail += "; SerenityFlow t2i completed " + ", ".join(completed_cases)
        else:
            detail += f"; SerenityFlow zimage_t2i completed {sf_job.get('id')}"
    else:
        detail += f"; SerenityFlow zimage_t2i completed {sf_job.get('id')}"
    completed_edit_cases = [
        f"{case_name} {case_job.get('id')}"
        for case_name, case_job in edit_jobs
        if case_job.get("id")
    ]
    if completed_edit_cases:
        detail += "; SerenityFlow edit completed " + ", ".join(completed_edit_cases)
    completed_qwen_edit_cases = [
        f"{case_name} {case_job.get('id')}"
        for case_name, case_job in qwen_edit_jobs
        if case_job.get("id")
    ]
    if completed_qwen_edit_cases:
        ignored_qwen_historical = True
    if ignored_qwen_historical:
        detail += "; historical Qwen/Qwen-edit workflow entries ignored while Qwen is metadata/preflight-only"
    if ideogram4_evidence:
        detail += f"; Ideogram4 visual export completed {ideogram4_job.get('id')}"
    return Check(
        True,
        PASS,
        "workflow",
        "typed workflow graph product smoke",
        detail,
        rel(WORKFLOW_GRAPH_PRODUCT),
        WORKFLOW_GRAPH_SMOKE_ACCEPTANCE,
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
    checks.append(
        check_klein_reference_daemon_smoke_report(
            KLEIN4B_REFERENCE_DAEMON_SMOKE,
            "4b",
            "flux2-klein-4b.safetensors",
            "/home/alex/mojodiffusion/serenitymojo/configs/klein4b.json",
        )
    )
    checks.append(
        check_klein_reference_daemon_smoke_report(
            KLEIN9B_REFERENCE_DAEMON_SMOKE,
            "9b",
            "flux2-klein-9b.safetensors",
            "/home/alex/mojodiffusion/serenitymojo/configs/klein9b.json",
        )
    )
    checks.append(check_klein_lora_daemon_smoke_report(KLEIN9B_LORA_DAEMON_SMOKE))
    checks.append(
        check_klein_lora_reference_daemon_smoke_report(
            KLEIN9B_LORA_REFERENCE_DAEMON_SMOKE
        )
    )
    checks.append(check_klein_real_image_health_report(KLEIN_REAL_IMAGE_HEALTH))
    checks.append(check_lanpaint_oracle_surface_report(LANPAINT_ORACLE_SURFACE))
    checks.append(check_lanpaint_canvas_daemon_smoke_report(LANPAINT_CANVAS_DAEMON_SMOKE))
    return checks


def build_report(checks: list[Check]) -> dict[str, object]:
    p0 = [check for check in checks if not check.ok and check.severity == P0]
    p1 = [check for check in checks if not check.ok and check.severity == P1]
    p2 = [check for check in checks if not check.ok and check.severity == P2]
    return {
        "schema": "serenity.workflow_node_surface_readiness.v1",
        "scope": "workflow graph module markers plus the latest typed workflow graph product smoke report",
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
