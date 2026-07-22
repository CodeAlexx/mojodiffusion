#!/usr/bin/env python3
"""Guard the deployed Serenity Canvas submission and pre-GPU admission seam."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CONTRACTS: dict[Path, tuple[str, ...]] = {
    ROOT / "serenity-server/canvas/js/api.js": (
        "function videoRequestFromWorkflow(workflow)",
        "runner: 'ltx2_mojo_request'",
        "checkpoint: ltxCheckpoint",
        "lora: loras",
        "return fetch('/v1/video'",
        "return fetch('/prompt'",
    ),
    ROOT / "serenity-server/canvas/js/canvas-tab.js": (
        "function validateCanvasGenerationFeatures(hasContent, hasMask)",
        "canvasFeature('controlnet')",
        "canvasFeature('image_conditioning')",
        "canvasFeature('inpaint')",
        "canvasFeature('image_to_image')",
        "function enabledCanvasLoras()",
        "function loadLtx2TemplateProfile()",
        "SerenityAPI.postPrompt(workflow",
        "CanvasStaging.activate(",
        "function cancelCanvasGeneration()",
        "checkCanvasSamAvailability()",
        "function prefer1024FlowEditForNewImage()",
        "imageDataUrlAsPngBase64(editSourceDataUrl)",
        "function prepareFlowEditCaptionPair(imagePath, sourceDescription, editInstruction)",
        "function syncCanvasModelFromFlowEditEngine()",
        "function syncFlowEditEngineFromCanvasModel(modelName)",
        "Describing source and target",
        "flowEditTargetPrompt: prepared.targetDescription",
        "FlowEdit preparation failed:",
    ),
    ROOT / "serenity-server/canvas/js/workflow-builder.js": (
        "function buildLTXV(p)",
        "krea2_turbo.safetensors",
        "class_type: 'LTXVSampler'",
        "prevClipRef ? 'LoraLoader' : 'LoraLoaderModelOnly'",
        "caps_positive: p.capsPositive || ''",
        "noise_fixture: p.noiseFixture || ''",
        "include_audio: p.includeAudio === true",
        "throw new Error('The LTX2 Mojo request runner is text-to-video only')",
    ),
    ROOT / "serenity-server/canvas/js/model-utils.js": (
        "fetch('/v1/capabilities'",
        "function backendForArch(arch)",
        "backendForArch: backendForArch",
    ),
    ROOT / "serenity-server/crates/server/src/video.rs": (
        "fn validate_ltx2_mojo_request(",
        '"ltx-2.3-22b-dev-fp8"',
        "lora_path_and_arch(name)",
        'if arch != "ltx2"',
        '== Some("ltx2_mojo_request")',
    ),
    ROOT / "serenity-server/crates/server/src/models.rs": (
        "pub fn lora_path_and_arch(",
    ),
    ROOT / "serenitymojo/sampling/ltx2_request_cli.mojo": (
        "def _configure_loras(obj: JSONValue) raises:",
        'String("LTX2_TRAINED_LORA_COUNT")',
        'String("LTX2_TRAINED_LORA_NAME_")',
        '_require_string(obj, String("checkpoint"))',
        "run_request_profile(",
    ),
    ROOT / "serenitymojo/pipeline/ltx2_t2v_av_hq.mojo": (
        'body += String(\'  "request_loras":[\')',
        'String("LTX2_TRAINED_LORA_NAME_")',
        'String("LTX2_TRAINED_LORA_MULT_")',
    ),
}

FORBIDDEN: dict[Path, tuple[str, ...]] = {
    ROOT / "serenity-server/canvas/js/workflow-builder.js": (
        "width: 1920",
        "height: 1088",
        "num_frames: 9",
    ),
    ROOT / "serenitymojo/sampling/ltx2_request_cli.mojo": (
        "import Python",
    ),
}


def main() -> int:
    failures: list[str] = []
    for path, required in CONTRACTS.items():
        if not path.is_file():
            failures.append(f"missing file: {path.relative_to(ROOT)}")
            continue
        text = path.read_text(encoding="utf-8")
        missing = [token for token in required if token not in text]
        if missing:
            failures.append(
                f"{path.relative_to(ROOT)} missing: {', '.join(repr(token) for token in missing)}"
            )

    for path, forbidden in FORBIDDEN.items():
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        present = [token for token in forbidden if token in text]
        if present:
            failures.append(
                f"{path.relative_to(ROOT)} contains forbidden substitution: "
                + ", ".join(repr(token) for token in present)
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
