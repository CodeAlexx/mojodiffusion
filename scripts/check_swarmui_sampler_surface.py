#!/usr/bin/env python3
"""Static SwarmUI/ComfyUI sampler surface readiness checker.

This checker intentionally does not import Mojo modules, allocate CUDA, or run
generation. It verifies request-surface markers and writes a JSON readiness
report for the current sampler parity audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[1]
SWARMUI = Path("/home/alex/SwarmUI")


@dataclass(frozen=True)
class Marker:
    feature: str
    label: str
    path: str
    ok: bool
    severity: str
    detail: str
    acceptance_gate: str


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def has_all(text: str, needles: Iterable[str]) -> bool:
    return all(needle in text for needle in needles)


def has_any(text: str, needles: Iterable[str]) -> bool:
    return any(needle in text for needle in needles)


def marker(
    feature: str,
    label: str,
    path: Path,
    ok: bool,
    detail: str,
    gate: str,
    severity: str = "blocker",
) -> Marker:
    return Marker(
        feature=feature,
        label=label,
        path=rel(path),
        ok=ok,
        severity="ok" if ok else severity,
        detail=detail,
        acceptance_gate=gate,
    )


def source_markers() -> list[Marker]:
    markers: list[Marker] = []

    comfy_samplers = SWARMUI / "dlbackend/ComfyUI/comfy/samplers.py"
    comfy_nodes = SWARMUI / "dlbackend/ComfyUI/nodes.py"
    swarm_backend = (
        SWARMUI
        / "src/BuiltinExtensions/ComfyUIBackend/ComfyUIBackendExtension.cs"
    )
    workflow_generator = (
        SWARMUI / "src/BuiltinExtensions/ComfyUIBackend/WorkflowGenerator.cs"
    )
    swarm_ksampler = (
        SWARMUI
        / "src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyCommon/SwarmKSampler.py"
    )
    t2i_params = SWARMUI / "src/Text2Image/T2IParamTypes.cs"

    samplers_text = read_text(comfy_samplers)
    nodes_text = read_text(comfy_nodes)
    backend_text = read_text(swarm_backend)
    workflow_text = read_text(workflow_generator)
    swarm_ksampler_text = read_text(swarm_ksampler)
    t2i_text = read_text(t2i_params)

    markers.append(
        marker(
            "SwarmUI/Comfy expectation source",
            "Comfy sampler catalog found",
            comfy_samplers,
            has_all(
                samplers_text,
                [
                    "KSAMPLER_NAMES",
                    "SAMPLER_NAMES = KSAMPLER_NAMES +",
                    '"dpmpp_2m"',
                    '"uni_pc_bh2"',
                ],
            ),
            "Comfy sampler names include KSampler names plus DDIM/UniPC variants.",
            "Keep this checker pinned to the local Comfy catalog when auditing supported sampler names.",
        )
    )
    markers.append(
        marker(
            "SwarmUI/Comfy expectation source",
            "Comfy scheduler catalog found",
            comfy_samplers,
            has_all(
                samplers_text,
                [
                    "SCHEDULER_HANDLERS",
                    '"simple"',
                    '"karras"',
                    '"exponential"',
                    '"kl_optimal"',
                ],
            ),
            "Comfy scheduler handlers are present in the local backend.",
            "Mojo must validate requested schedulers against a model-compatible support matrix.",
        )
    )
    markers.append(
        marker(
            "SwarmUI/Comfy expectation source",
            "Swarm scheduler extensions found",
            swarm_backend,
            has_all(
                backend_text,
                [
                    "align_your_steps",
                    "ltxv",
                    "ltxv-image",
                    "flux2",
                ],
            ),
            "SwarmUI exposes extra scheduler names beyond base Comfy.",
            "Unsupported Swarm scheduler names must fail loud or be hidden per model.",
        )
    )
    markers.append(
        marker(
            "KSampler request mapping",
            "Comfy KSampler inputs found",
            comfy_nodes,
            has_all(
                nodes_text,
                [
                    '"sampler_name"',
                    '"scheduler"',
                    '"denoise"',
                    '"cfg"',
                    '"seed"',
                ],
            ),
            "Comfy KSampler exposes sampler, scheduler, CFG, seed, and denoise inputs.",
            "Daemon graph adapter must preserve these fields and backends must execute or reject them.",
        )
    )
    markers.append(
        marker(
            "KSampler request mapping",
            "Swarm family defaults found",
            workflow_generator,
            has_all(
                workflow_text,
                [
                    'DefaultSampler = "euler"',
                    'DefaultScheduler = "normal"',
                    'defscheduler ??= "flux2"',
                    'defsampler ??= "er_sde"',
                ],
            ),
            "SwarmUI applies family-specific default sampler/scheduler choices.",
            "Mojo must expose per-family defaults and not silently replace requested values.",
        )
    )
    markers.append(
        marker(
            "Variation seed and strength",
            "Swarm variation noise blend found",
            swarm_ksampler,
            has_all(
                swarm_ksampler_text,
                [
                    "var_seed",
                    "var_seed_strength",
                    "slerp(",
                    "seed + i",
                ],
            ),
            "SwarmKSampler blends variation noise and increments seeds per batch element.",
            "Mojo variation support must affect noise, not just metadata.",
        )
    )
    markers.append(
        marker(
            "Img2img denoise / creativity",
            "Swarm init image creativity found",
            t2i_params,
            has_all(
                t2i_text,
                [
                    "Init Image Creativity",
                    "fraction of steps",
                    "Variation Seed",
                    "Images",
                ],
            ),
            "SwarmUI user params include images, variation seed, and init-image creativity.",
            "Mojo must connect these controls to runtime behavior or reject unsupported cases.",
        )
    )

    return markers


def mojo_markers() -> list[Marker]:
    markers: list[Marker] = []

    daemon = REPO / "serenitymojo/serve/serenity_daemon.mojo"
    backend = REPO / "serenitymojo/serve/backend.mojo"
    zimage = REPO / "serenitymojo/serve/zimage_backend.mojo"
    qwen = REPO / "serenitymojo/serve/qwenimage_backend.mojo"
    sampler_registry = REPO / "serenitymojo/sampling/sampler_registry.mojo"
    variation_noise = REPO / "serenitymojo/sampling/variation_noise.mojo"
    dispatch = REPO / "serenitymojo/serve/dispatch_backend.mojo"
    stub = REPO / "serenitymojo/serve/stub_backend.mojo"
    harness = REPO / "serenitymojo/sampling/product_sampler_harness.mojo"
    harness_doc = REPO / "serenitymojo/docs/SAMPLER_PRODUCT_HARNESS_2026-06-05.md"
    parity_doc = (
        REPO / "serenitymojo/docs/SWARMUI_SAMPLER_PARITY_MAP_2026-06-12.md"
    )

    daemon_text = read_text(daemon)
    backend_text = read_text(backend)
    zimage_text = read_text(zimage)
    qwen_text = read_text(qwen)
    sampler_registry_text = read_text(sampler_registry)
    variation_noise_text = read_text(variation_noise)
    dispatch_text = read_text(dispatch)
    stub_text = read_text(stub)
    harness_text = read_text(harness)
    harness_doc_text = read_text(harness_doc)
    parity_doc_text = read_text(parity_doc)

    markers.append(
        marker(
            "Flat request parsing",
            "Daemon preserves sampler surface params",
            daemon,
            has_all(
                daemon_text,
                [
                    '"sampler"',
                    '"scheduler"',
                    '"variation_seed"',
                    '"variation_strength"',
                    '"images"',
                    '"init_image"',
                    '"creativity"',
                    "GENPARAMS_TEXT_KEY",
                ],
            ),
            "Daemon parses and stores sampler-facing fields in canonical metadata.",
            "Backend execution and metadata must distinguish requested versus executed values.",
            severity="warning",
        )
    )
    markers.append(
        marker(
            "Workflow graph coverage",
            "Constrained KSampler adapter found",
            daemon,
            has_all(
                daemon_text,
                [
                    "_apply_workflow_params",
                    '"KSampler"',
                    '"sampler_name"',
                    '"denoise"',
                    '"EmptyLatentImage"',
                    "unsupported workflow graph node",
                ],
            ),
            "Daemon maps a constrained Comfy KSampler-like graph and rejects unknown nodes.",
            "Every supported node must have product tests; unsupported nodes must remain loud failures.",
            severity="warning",
        )
    )
    markers.append(
        marker(
            "Backend typed request",
            "JobParams carries typed sampler admission fields",
            backend,
            has_all(
                backend_text,
                [
                    "struct JobParams",
                    "var sampler: String",
                    "var scheduler: String",
                    "var variation_seed: Int",
                    "var variation_strength: Float64",
                    "var images: Int",
                    "var image_index: Int",
                    "var image_count: Int",
                    "var init_image: String",
                    "var creativity: Float64",
                    "var params_json: String",
                    "reject_unsupported_common_runtime_params",
                ],
            ),
            "Executable JobParams carry sampler, scheduler, variation, and image count fields with common fail-loud guards.",
            "Backend support still needs per-algorithm artifacts before sampler parity is accepted.",
            severity="warning",
        )
    )
    markers.append(
        marker(
            "Sampler/scheduler registry",
            "Mojo exposes a SwarmUI/Comfy sampler support matrix",
            sampler_registry,
            has_all(
                sampler_registry_text,
                [
                    "serenity.samplers.v1",
                    "sampler_admission_for_backend",
                    "scheduler_admission_for_backend",
                    "unsupported_policy",
                    "fail_loud",
                    "blocked_samplers",
                    '"not_alias_of":"uni_pc_bh2"',
                    '"required_variant":"bh1"',
                    '"required_schedule":"SigmaConvert"',
                    "dpmpp_2m",
                    "uni_pc_bh2",
                    "align_your_steps",
                    "ltxv-image",
                ],
            )
            and has_all(
                daemon_text,
                [
                    'path == "/v1/samplers"',
                    "swarmui_sampler_registry_json",
                    "default_sampler_for_backend",
                    "default_scheduler_for_backend",
                ],
            ),
            "Daemon exposes a pure-Mojo /v1/samplers registry with backend support, unsupported-policy, and non-claim labels.",
            "Accepted sampler parity still requires one artifact-backed denoise loop per accepted sampler/scheduler pair.",
            severity="warning",
        )
    )
    markers.append(
        marker(
            "Variation seed and strength",
            "Swarm-style variation noise is wired into image backends",
            variation_noise,
            has_all(
                variation_noise_text,
                [
                    "swarm_variation_noise_chw",
                    "SwarmKSampler-compatible slerp",
                    "acos",
                    "sin",
                ],
            )
            and has_all(
                zimage_text + qwen_text,
                [
                    "swarm_variation_noise_chw",
                    "self.params.variation_seed + self.params.image_index",
                    "variation_strength > 0.0",
                ],
            ),
            "Z-Image and Qwen apply variation_seed/variation_strength to initial latent noise; Z-Image also records variation_applied in its manifest.",
            "Runtime acceptance still requires artifact evidence that variation changes the output payload.",
            severity="warning",
        )
    )
    markers.append(
        marker(
            "Z-Image backend subset",
            "Z-Image executes seed/CFG/negative/img2img subset",
            zimage,
            has_all(
                zimage_text,
                [
                    "only 512x512 is served",
                    "len(params.loras) > 1",
                    "self.params.init_image",
                    "self.params.creativity",
                    "_build_sigmas(self.params.steps)",
                    "self.params.seed",
                    "self.params.negative",
                    "_cfg_pred_overlay",
                    "sampler_admission_for_backend",
                    "scheduler_admission_for_backend",
                    "unsupported sampler",
                    "executed_sampler",
                    "dpmpp_2m_step",
                    "UniPcMultistepScheduler",
                    "from_sigmas",
                    "sampler_trace",
                    "dpmpp_update_steps",
                    "dpmpp_second_order_steps",
                    "unipc_update_steps",
                    "unipc_second_order_steps",
                    "schedule_source",
                    "self.params.image_index",
                    "self.params.image_count",
                ],
            ),
            "Z-Image has real subset behavior, registry-backed admission, fixed Euler/simple flow-match, bounded DPM++ 2M/simple flow-match execution, and bounded UniPC bh2/simple flow-match execution.",
            "Accepted sampler parity needs per-sampler artifact evidence and executed sampler metadata.",
            severity="warning",
        )
    )
    markers.append(
        marker(
            "Qwen backend subset",
            "Qwen executes seed/CFG/negative subset and rejects img2img/LoRA",
            qwen,
            has_all(
                qwen_text,
                [
                    "only 1024x1024 is served",
                    "LoRA is not supported",
                    "img2img is not supported",
                    "Scheduler.qwen(self.params.steps",
                    "self.params.seed",
                    "self.params.negative",
                    "cfg_qwen",
                    "sampler_admission_for_backend",
                    "scheduler_admission_for_backend",
                    "unsupported sampler",
                ],
            ),
            "Qwen has a model-specific schedule and registry-backed sampler/scheduler admission.",
            "Full Qwen generation remains out of scope until separate product gates pass.",
            severity="warning",
        )
    )
    markers.append(
        marker(
            "Model/backend mapping",
            "Dispatch surface is limited to Z-Image and Qwen real backends",
            dispatch,
            has_all(dispatch_text, ["KIND_ZIMAGE", "KIND_QWEN"])
            and not has_any(dispatch_text, ["KIND_KLEIN", "KIND_SDXL", "KIND_SD3"]),
            "Daemon real dispatch currently covers Z-Image and Qwen, not Klein/SDXL/SD3/etc.",
            "Add dispatch entries only after model-specific artifact, timing, VRAM, and sampler evidence exists.",
            severity="blocker",
        )
    )
    markers.append(
        marker(
            "Output metadata and reuse",
            "PNG metadata and gallery markers found",
            daemon,
            has_all(
                daemon_text,
                [
                    "read_png_text",
                    "GENPARAMS_TEXT_KEY",
                ],
            )
            and has_all(
                stub_text + zimage_text + qwen_text,
                [
                    "encode_png_with_text",
                    "serenity.genparams.v1",
                ],
            )
            and "jobs.db" in daemon_text,
            "Daemon can write/read canonical generation metadata and persist job/gallery state.",
            "Artifacts must include requested and executed sampler/scheduler plus acceptance booleans.",
            severity="warning",
        )
    )
    markers.append(
        marker(
            "Readiness labels",
            "Product harness remains a measurement scaffold",
            harness,
            has_all(
                harness_text,
                [
                    "sample_config_ready",
                    "transformer_denoise_ready",
                    "timing_ready",
                    "vram_ready",
                ],
            )
            and has_any(
                harness_doc_text,
                [
                    "not a speed parity proof",
                    "not accepted speed parity",
                    "measurement scaffold",
                ],
            ),
            "Harness markers exist, but docs say they are not acceptance proof.",
            "Runtime acceptance still requires real artifacts and backend manifests.",
            severity="warning",
        )
    )
    markers.append(
        marker(
            "Sampler parity map doc",
            "Owned parity map document exists",
            parity_doc,
            has_all(
                parity_doc_text,
                [
                    "| Feature | SwarmUI/Comfy expectation | Current Mojo surface | Blocker | Acceptance gate |",
                    "Sampler name catalog",
                    "Scheduler name catalog",
                    "Variation seed and strength",
                    "accepted_sampler_parity",
                ],
            ),
            "Sampler parity map includes the required table and non-acceptance language.",
            "Keep the map updated as backend support changes.",
        )
    )

    sampling_files = {
        "Z-Image flow match": REPO / "serenitymojo/sampling/flow_match.mojo",
        "Qwen flow match": REPO / "serenitymojo/sampling/qwenimage_sampling.mojo",
        "Flux2/Klein": REPO / "serenitymojo/sampling/flux2_klein.mojo",
        "SDXL Euler": REPO / "serenitymojo/sampling/sdxl_euler.mojo",
        "SD15 Euler": REPO / "serenitymojo/sampling/sd15_euler.mojo",
        "SD3 flow match": REPO / "serenitymojo/sampling/sd3_flow_match.mojo",
        "DPM++ 2M": REPO / "serenitymojo/sampling/dpmpp_2m.mojo",
        "UniPC": REPO / "serenitymojo/sampling/unipc.mojo",
        "LTX2": REPO / "serenitymojo/sampling/ltx2_sampling.mojo",
        "img2img refpack": REPO / "serenitymojo/sampling/img2img_refpack.mojo",
        "inpaint": REPO / "serenitymojo/sampling/inpaint.mojo",
    }
    for label, path in sampling_files.items():
        markers.append(
            marker(
                "Sampling module inventory",
                f"{label} sampler module present",
                path,
                path.is_file(),
                f"{label} source file is present.",
                "Presence is only inventory; daemon dispatch and artifact gates are still required.",
                severity="warning",
            )
        )

    return markers


def feature_status(markers: list[Marker]) -> list[dict[str, object]]:
    grouped: dict[str, list[Marker]] = {}
    for item in markers:
        grouped.setdefault(item.feature, []).append(item)

    features: list[dict[str, object]] = []
    for feature in sorted(grouped):
        items = grouped[feature]
        failed = [item for item in items if not item.ok]
        features.append(
            {
                "feature": feature,
                "markers": len(items),
                "passed_markers": sum(1 for item in items if item.ok),
                "status": "blocked" if failed else "surface_marker_present",
                "blockers": [item.detail for item in failed],
                "acceptance_gates": sorted({item.acceptance_gate for item in items}),
            }
        )
    return features


def surface_blockers() -> list[dict[str, str]]:
    return [
        {
            "id": "sampler_scheduler_dispatch",
            "severity": "P1",
            "blocker": "Z-Image now has bounded DPM++ 2M and UniPC bh2/simple flow-match paths, but generic UniPC/order-3, ancestral, SDE, CFG++, Karras, and the rest of the SwarmUI/Comfy sampler catalog still lack distinct daemon denoise loops.",
            "acceptance_gate": "Wire each accepted sampler/scheduler pair into a backend denoise loop and record requested versus executed values with artifact/timing/VRAM evidence.",
        },
        {
            "id": "latent_batch_execution",
            "severity": "P1",
            "blocker": "images=N is expanded into indexed serial daemon jobs, but Comfy-style batched latent execution is not implemented as a single backend batch.",
            "acceptance_gate": "If the UI exposes Comfy batch-size semantics separately from Images count, add a batched latent path or fail loud per backend.",
        },
        {
            "id": "advanced_surfaces",
            "severity": "P1",
            "blocker": "Hires/upscale/refiner/control/regional/mask surfaces are absent or only helper-level inventory, not accepted daemon product paths.",
            "acceptance_gate": "Expose validated request fields, fail loud on unsupported model pairs, and prove final dimensions, stage metadata, timings, and VRAM.",
        },
        {
            "id": "model_dispatch_coverage",
            "severity": "P1",
            "blocker": "Sampling modules exist for Klein/Flux2, SDXL, SD15, SD3, Chroma, ERNIE, Anima, and LTX2, but real daemon dispatch is limited to Z-Image and Qwen.",
            "acceptance_gate": "Add real dispatch only after each model has artifact, timing, VRAM, metadata, and sampler/scheduler failure-mode evidence.",
        },
        {
            "id": "conditioning_parity",
            "severity": "P1",
            "blocker": "Negative prompt and CFG are model-specific and prompt weights are parsed/persisted but not proven applied to conditioning math.",
            "acceptance_gate": "Per model, prove negative prompt, CFG scale, and weighted prompt behavior with artifact and metadata checks.",
        },
        {
            "id": "qwen_video_quarantine",
            "severity": "P0",
            "blocker": "Full Qwen generation and video generation are not accepted targets for this sampler task.",
            "acceptance_gate": "Keep Qwen full-generation and video sampler parity blocked until separate product gates provide real artifacts and resource evidence.",
        },
    ]


def build_report() -> dict[str, object]:
    markers = source_markers() + mojo_markers()
    failed = [item for item in markers if not item.ok]
    marker_blockers = [item for item in failed if item.severity == "blocker"]
    warnings = [item for item in failed if item.severity != "blocker"]
    blockers = surface_blockers()

    return {
        "checker": "check_swarmui_sampler_surface",
        "schema_version": 1,
        "repo": str(REPO),
        "swarmui_repo": str(SWARMUI),
        "surface_audit_only": True,
        "cuda_required": False,
        "runtime_generation_run": False,
        "accepted_runtime_parity": False,
        "accepted_sampler_parity": False,
        "readiness_label": "blocked",
        "summary": {
            "markers_total": len(markers),
            "markers_passed": sum(1 for item in markers if item.ok),
            "markers_failed": len(failed),
            "marker_blockers": len(marker_blockers),
            "marker_warnings": len(warnings),
            "surface_blockers": len(blockers),
        },
        "surface_blockers": blockers,
        "features": feature_status(markers),
        "markers": [asdict(item) for item in markers],
        "non_claims": [
            "This checker does not run generation.",
            "This checker does not prove sampler runtime parity.",
            "This checker does not prove Qwen full-generation readiness.",
            "This checker does not prove video readiness.",
        ],
        "next_command": "python3 scripts/check_swarmui_sampler_surface.py --write-readiness output/checks/swarmui_sampler_surface_readiness.json",
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-readiness",
        type=Path,
        help="Write the JSON readiness report. Use a path under output/.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if required surface markers are missing.",
    )
    args = parser.parse_args(argv)

    report = build_report()
    if args.write_readiness is not None:
        out_path = args.write_readiness
        if not out_path.is_absolute():
            out_path = REPO / out_path
        try:
            out_path.relative_to(REPO / "output")
        except ValueError:
            print(
                f"refusing to write readiness report outside output/: {out_path}",
                file=sys.stderr,
            )
            return 2
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.strict and (
        report["summary"]["marker_blockers"] or report["summary"]["surface_blockers"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
