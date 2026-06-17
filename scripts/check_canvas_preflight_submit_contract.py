#!/usr/bin/env python3
"""Guard the canvas generate path preflight contract.

The browser adapter must call `/v1/preflight` with the exact body it will submit
to `/v1/generate`, and the generate UI must distinguish preflight rejection from
backend/network failure.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
API = ROOT / "serenity-server/canvas/js/api.js"
CANVAS_CSS = ROOT / "serenity-server/canvas/css/styles.css"
GENERATE_WS = ROOT / "serenity-server/canvas/js/modules/generate_ws.js"
PARAM_RAIL = ROOT / "serenity-server/canvas/js/modules/param_rail.js"
REFINER_UPSCALE = ROOT / "serenity-server/canvas/js/modules/refiner_upscale.js"
PROMPT_BAR = ROOT / "serenity-server/canvas/js/modules/prompt_bar.js"
GALLERY_PRO = ROOT / "serenity-server/canvas/js/modules/gallery_pro.js"
PROMPT_SYNTAX = ROOT / "serenity-server/canvas/js/modules/prompt_syntax.js"
WORKFLOWS = ROOT / "serenity-server/canvas/js/modules/workflows.js"
GRID_XYZ = ROOT / "serenity-server/canvas/js/modules/grid_xyz.js"
IDEOGRAM4_NODES = ROOT / "serenity-server/canvas/js/modules/ideogram4_nodes.js"
BBOX_BUILDER = ROOT / "serenity-server/canvas/js/modules/bbox_builder.js"


API_REQUIRED = (
    "function requirePreflight(body)",
    "postJSON('/v1/preflight', body)",
    "err.name = \"PreflightBlocked\"",
    "err.preflight = r || null",
    "err.name = \"GenerateRejected\"",
    "err.generateError = parsed",
    "err.capabilityProfile = parsed.capability_profile || null",
    "function hasWorkflowGraph(graph)",
    "function workflowSubmitBody(graph)",
    "function workflowParamsSubmitBody(params)",
    "function submitBody(graph)",
    "workflow: graph",
    "workflow: { params: params }",
    "workflow_client: \"serenity.canvas.generate_ws\"",
    "workflowParamsSubmitBody(generateBody())",
    "var body = submitBody(_graph);",
    "return requirePreflight(body).then(function (pf)",
    "return postJSON('/v1/generate', body);",
    "return postJSON('/v1/preflight', body || submitBody(null));",
    "window.Serenity.bus.emit(\"generate:preflight\", pf)",
    "var capabilityCache = null",
    "function capabilities(force)",
    "capabilities: capabilities",
    "function backendForModelName(model)",
    "function defaultsForBackend(backend)",
    "function defaultsForModel(model)",
    "m.indexOf(\"flux2\") >= 0 || m.indexOf(\"flux-2\") >= 0",
    "m.indexOf(\"flux_2\") >= 0 || m.indexOf(\"klein\") >= 0) return \"flux2\"",
    "m.indexOf(\"sensenova\") >= 0 || m.indexOf(\"sense_nova\") >= 0",
    "m.indexOf(\"sdxl\") >= 0 || m.indexOf(\"sd_xl\") >= 0",
    "m.indexOf(\"sd-xl\") >= 0 || m.indexOf(\"sd xl\") >= 0",
    "function normalizeModelName(model)",
    "normalizeModelName: normalizeModelName",
    "function featureSupportedForModel(model, featureName, fallback)",
    "function anyFeatureSupportedForModel(model, featureNames, fallback)",
    "prompt_raw: p.prompt_raw || '', negative_raw: p.negative_raw || ''",
    "denoise: p.denoise != null ? p.denoise : 1.0",
    "refiner: null",
    "hires_scale: 1.0, hires_denoise: 0.4",
    "init_image: '', mask_image: ''",
    "lanpaint_mask_channel: ''",
    "outpaint: null, outpaint_enabled: false",
    "refiner_model: '', refiner_steps: 0",
    "refiner_cfg: -1, refiner_method: ''",
    "refiner_control: -1, refiner_tiling: false",
    "upscaler: null, upscaler_model: '', upscale_by: 1.0",
)

API_FORBIDDEN = (
    "hires_scale: p.hires_scale",
    "hires_denoise: p.hires_denoise",
    "init_image: p.init_image",
    "mask_image: p.mask_image",
    "lanpaint_mask_channel: p.lanpaint_mask_channel",
    "outpaint: p.outpaint",
    "outpaint_enabled: p.outpaint_enabled",
    "refiner: p.refiner",
    "refiner_model: p.refiner_model",
    "refiner_steps: p.refiner_steps",
    "refiner_cfg: p.refiner_cfg",
    "refiner_method: p.refiner_method",
    "refiner_control: p.refiner_control",
    "refiner_tiling: p.refiner_tiling",
    "upscaler: p.upscaler",
    "upscaler_model: p.upscaler_model",
    "upscale_by: p.upscale_by",
    "body || generateBody()",
    "? workflowSubmitBody(graph) : generateBody()",
)

GENERATE_WS_REQUIRED = (
    "e.name === \"PreflightBlocked\"",
    "e.name === \"GenerateRejected\"",
    "ui.setStatus(e.message || \"preflight blocked\")",
    "ui.setStatus(\"generate blocked: \"",
    "bus.emit(\"generate:preflight_blocked\"",
    "bus.emit(\"generate:rejected\"",
    "graph that was blocked before enqueue",
    "graph rejected by /v1/generate",
    "assemble a ComfyUI workflow GRAPH from state.params + state.layers",
    "the Rust workflow lowerer and capability gate decide admission",
    "api.normalizeModelName(p.model)",
    "inputs: { ckpt_name: modelName || \"z-image\" }",
    "let imgRef = cl.uploadedPath || cl.uploadedName || null;",
    "let initRef = rasterLayer.uploadedPath || rasterLayer.uploadedName || null;",
    "let maskRef = ml.uploadedPath || ml.uploadedName || null;",
    "function maybeUploadLayerImage(layer, kind)",
    "function maybeUploadLayerMask(layer, rasterLayer)",
    "res.path || res.name || res.filename",
    "layer.uploadedPath = ref",
    "layer.uploadedPath = out",
    "class_type: \"SerenityRefinerUpscaleIntent\"",
    "function hasRefinerUpscaleIntent(p)",
    "function refinerUpscaleIntentInputs(p)",
)

GENERATE_WS_FORBIDDEN = (
    "function unsupportedCanvasConditioning()",
    "const unsupported = unsupportedCanvasConditioning();",
    "generate:unsupported_canvas_conditioning",
    "function prepareMaskAndInit()",
    "await prepareMaskAndInit();",
    "image conditioning is not admitted by the current production route",
)

PARAM_RAIL_REQUIRED = (
    "ControlNet is not production-admitted in this route.",
    "Refiner and hires two-pass are not production-admitted in this route.",
    "cnEnable.disabled = true",
    "cnStrength.disabled = true",
    "refEnable.disabled = true",
    "refModel.disabled = true",
    "refSwitch.disabled = true",
    'set("params.controlnet", null)',
    'set("params.refiner", null)',
    'set("params.hires_scale", 1.0)',
    'set("params.hires_denoise", 0.4)',
    "function capabilityBackendEntry(backend)",
    "api.capabilityForBackend(backend)",
    'bus.on("capabilities:loaded", applyModelCaps)',
    "capSamplers && capSamplers.supported_samplers",
    "function capabilityDefaults(backend)",
    "function admittedDefaultSize(backend, sizes)",
    "function applyCapabilityDefaults(backend, backendChanged)",
    "flux2: [[512, 512]]",
    "sensenova: [[1024, 1024], [512, 512]]",
    "sensenova: { width: 1024, height: 1024, steps: 30",
    "m.indexOf(\"flux2\") >= 0 || m.indexOf(\"flux-2\") >= 0 || m.indexOf(\"flux_2\") >= 0 || m.indexOf(\"klein\") >= 0) return \"flux2\"",
    "m.indexOf(\"sensenova\") >= 0 || m.indexOf(\"sense_nova\") >= 0 || m.indexOf(\"sense-nova\") >= 0) return \"sensenova\"",
    "m.indexOf(\"sdxl\") >= 0 || m.indexOf(\"sd_xl\") >= 0 || m.indexOf(\"sd-xl\") >= 0 || m.indexOf(\"sd xl\") >= 0",
)

REFINER_UPSCALE_REQUIRED = (
    "This panel authors workflow intent.",
    "SerenityRefinerUpscaleIntent graph node",
    "Workflow preflight decides whether this graph is admitted.",
    "Unsupported two-pass graphs fail at Rust workflow preflight before enqueue.",
    'badge("wired", "workflow")',
    'set("params.refiner", enabled ? {',
    'set("params.upscaler", (enabled && factor > 1.0) ? {',
    'set("params.upscale_by", (enabled && factor > 1.0) ? factor : 1.0)',
    'set("params.hires_scale", factor)',
    'set("params.hires_denoise", control)',
)

REFINER_UPSCALE_FORBIDDEN = (
    "PRODUCT_ROUTE_REFINER_UPSCALE_DISABLED",
    "function routeRefinerUpscaleDisabled()",
    "function clearProductDisabledState()",
    "api.anyFeatureSupportedForModel(",
    'bus.on("capabilities:loaded", syncProductRouteCapability)',
    "Refine/Upscale is disabled while /v1/generate is txt2img-only.",
    "Hires two-pass is rejected by Rust prequeue",
    'panel.classList.add("ru-disabled")',
    "n.disabled = disabled",
)

PROMPT_BAR_REQUIRED = (
    "function negativeDisabled()",
    'api.featureSupportedForModel(model, "negative_prompt", fallbackSupported)',
    'bus.on("capabilities:loaded", syncNegativeAvailability)',
    'set("params.negative", "")',
    'bus.on("change:params.model", syncNegativeAvailability)',
)

PROMPT_BAR_FORBIDDEN = (
    "uploadImage",
    "uploadMask",
    "attach image",
    "paste image",
)

GALLERY_PRO_REQUIRED = (
    "img2img/upscale actions create layer workflow state; preflight decides",
    "function sendToImg2img(it)",
    "return new Promise((resolve) => {",
    "img.onload = () => {",
    "bus.emit('layers:changed', state.layers.slice())",
    "bus.emit('img2img:source'",
    "resolve(true);",
    "async function upscale(it)",
    "const ok = await sendToImg2img(it);",
    "if (!ok) return;",
    "bus.emit('generate:request')",
    "add('Send to img2img', () => sendToImg2img(it));",
    "add('Upscale 2×', () => upscale(it));",
    "actions.appendChild(actBtn('img2img', () => sendToImg2img(current())));",
    "actions.appendChild(actBtn('Upscale', () => upscale(current())));",
)

GALLERY_PRO_FORBIDDEN = (
    "txt2imgOnlyRoute",
    "function refreshRouteCapabilities()",
    "function routeIsTxt2ImgOnly()",
    "routeIsTxt2ImgOnly()",
    "api.anyFeatureSupportedForModel(",
    "function clearImageConditioningState()",
    "img2img is not admitted in the current production route",
    "Upscale is not admitted in the current production route",
    "if (!routeIsTxt2ImgOnly())",
)

PROMPT_SYNTAX_REQUIRED = (
    "function resolveParamsForSubmit(params, bus, opts)",
    "function patchWorkflowGraphForSubmit(graph, rPos, rNeg, seed, promptLoras)",
    "function patchWorkflowParamsForSubmit(graph, rPos, rNeg, seed, promptLoras)",
    "resolveParamsForSubmit: function (params, opts)",
    "params.seed = seed",
    "params[key] = mergePromptLoras(params[key], lorasForParams)",
    "patchWorkflowGraphForSubmit(",
    "resolved.lorasForParams",
    "samplerInputs.seed = seed",
    "patchClipText(posNode, rPos.resolved)",
    "patchClipText(negNode, rNeg.resolved)",
    "class_type: \"LoraLoader\"",
    "samplerInputs.model = modelSlot",
    "posNode.inputs.clip = clipSlot",
    "negNode.inputs.clip = clipSlot",
)

WORKFLOWS_REQUIRED = (
    "function compileToComfyPrompt(uiGraph)",
    "function workflowDefaults(modelValue)",
    "api.defaultsForModel(modelValue)",
    "steps: Number(p.steps) || defaults.steps",
    "class_type: \"ConditioningZeroOut\"",
    "function validateComfyPrompt(out)",
    "if (!hasSave) throw new Error(\"workflow needs a SaveImage terminal\")",
    "compile: function () { if (!built) build(); return compileToComfyPrompt(graph()); }",
    "? { params: lowered }",
    "api.submitPrompt(submitGraph, ctx.clientId)",
)

WORKFLOWS_FORBIDDEN = (
    "api.submitPrompt(null",
)

GRID_XYZ_REQUIRED = (
    "var params = {",
    "S.api.normalizeModelName(p.model)",
    "S.promptSyntax.resolveParamsForSubmit(params, { emitEvents: true })",
    "workflow: { params: params }",
    "workflow_client: \"serenity.canvas.grid_xyz\"",
    "fetch(apiBase + \"/v1/grid\"",
)

IDEOGRAM4_NODES_REQUIRED = (
    "function promptJsonFromCaption(caption)",
    "prompt_json: promptJsonFromCaption(caption)",
    "prompt_raw: caption",
    "negative: \"\"",
)

BBOX_BUILDER_REQUIRED = (
    "function buildCaptionObject()",
    "prompt_json: promptJson",
    "api.submitPrompt({ params: params }, ctx.clientId)",
)

BBOX_BUILDER_FORBIDDEN = (
    "api.submitPrompt(null",
)

CANVAS_CSS_REQUIRED = (
    "#queue-strip{position:fixed;right:12px;bottom:236px;z-index:20}",
)


def require(path: Path, tokens: tuple[str, ...]) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [token for token in tokens if token not in text]


def forbid(path: Path, tokens: tuple[str, ...]) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [token for token in tokens if token in text]


def require_order(path: Path, before: str, after: str) -> str | None:
    text = path.read_text(encoding="utf-8")
    before_idx = text.find(before)
    after_idx = text.find(after)
    if before_idx < 0 or after_idx < 0:
        return None
    if before_idx > after_idx:
        return (
            f"{path.relative_to(ROOT)} order: {before!r} must appear before "
            f"{after!r}"
        )
    return None


def main() -> int:
    failures: list[str] = []
    for path, tokens in (
        (API, API_REQUIRED),
        (GENERATE_WS, GENERATE_WS_REQUIRED),
        (PARAM_RAIL, PARAM_RAIL_REQUIRED),
        (REFINER_UPSCALE, REFINER_UPSCALE_REQUIRED),
        (PROMPT_BAR, PROMPT_BAR_REQUIRED),
        (GALLERY_PRO, GALLERY_PRO_REQUIRED),
        (PROMPT_SYNTAX, PROMPT_SYNTAX_REQUIRED),
        (WORKFLOWS, WORKFLOWS_REQUIRED),
        (GRID_XYZ, GRID_XYZ_REQUIRED),
        (IDEOGRAM4_NODES, IDEOGRAM4_NODES_REQUIRED),
        (BBOX_BUILDER, BBOX_BUILDER_REQUIRED),
        (CANVAS_CSS, CANVAS_CSS_REQUIRED),
    ):
        missing = require(path, tokens)
        if missing:
            failures.append(f"{path.relative_to(ROOT)} missing: {', '.join(missing)}")

    forbidden = forbid(PROMPT_BAR, PROMPT_BAR_FORBIDDEN)
    if forbidden:
        failures.append(
            f"{PROMPT_BAR.relative_to(ROOT)} forbidden txt2img-only affordance: "
            + ", ".join(forbidden)
        )

    api_forbidden = forbid(API, API_FORBIDDEN)
    if api_forbidden:
        failures.append(
            f"{API.relative_to(ROOT)} forwards disabled img2img/refine/upscale params: "
            + ", ".join(api_forbidden)
        )

    generate_ws_forbidden = forbid(GENERATE_WS, GENERATE_WS_FORBIDDEN)
    if generate_ws_forbidden:
        failures.append(
            f"{GENERATE_WS.relative_to(ROOT)} contains browser-side workflow shape gate: "
            + ", ".join(generate_ws_forbidden)
        )

    refiner_upscale_forbidden = forbid(REFINER_UPSCALE, REFINER_UPSCALE_FORBIDDEN)
    if refiner_upscale_forbidden:
        failures.append(
            f"{REFINER_UPSCALE.relative_to(ROOT)} contains browser-side route-shape gate: "
            + ", ".join(refiner_upscale_forbidden)
        )

    gallery_pro_forbidden = forbid(GALLERY_PRO, GALLERY_PRO_FORBIDDEN)
    if gallery_pro_forbidden:
        failures.append(
            f"{GALLERY_PRO.relative_to(ROOT)} contains browser-side route-shape gate: "
            + ", ".join(gallery_pro_forbidden)
        )

    workflows_forbidden = forbid(WORKFLOWS, WORKFLOWS_FORBIDDEN)
    if workflows_forbidden:
        failures.append(
            f"{WORKFLOWS.relative_to(ROOT)} bypasses workflow submit: "
            + ", ".join(workflows_forbidden)
        )

    bbox_builder_forbidden = forbid(BBOX_BUILDER, BBOX_BUILDER_FORBIDDEN)
    if bbox_builder_forbidden:
        failures.append(
            f"{BBOX_BUILDER.relative_to(ROOT)} bypasses workflow submit: "
            + ", ".join(bbox_builder_forbidden)
        )

    if failures:
        print("canvas preflight submit contract: FAIL")
        for failure in failures:
            print(failure)
        return 1

    print("canvas preflight submit contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
