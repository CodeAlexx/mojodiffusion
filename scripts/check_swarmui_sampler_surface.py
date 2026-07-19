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


def read_json(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def dict_or_empty(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def positive_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0.0


def evidence_path(value: object) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else REPO / path


def klein_report_ready(path: Path, expected_model: str, *, expected_mode: str = "") -> bool:
    report = read_json(path)
    if report.get("ready") is not True:
        return False
    if report.get("blockers") not in ([], None):
        return False
    output_path = evidence_path(report.get("output_path"))
    manifest_path = evidence_path(report.get("manifest_path"))
    if not output_path.is_file() or output_path.stat().st_size < 100_000:
        return False
    if not manifest_path.is_file():
        return False
    genparams = dict_or_empty(report.get("genparams"))
    manifest = dict_or_empty(report.get("manifest"))
    visual_health = dict_or_empty(report.get("visual_health"))
    if genparams.get("model") != expected_model:
        return False
    if manifest.get("schema") != "serenity.klein_daemon_result.v1":
        return False
    if manifest.get("backend") != "klein":
        return False
    if manifest.get("model") != expected_model:
        return False
    if expected_mode and manifest.get("mode") != expected_mode:
        return False
    return visual_health.get("ready") is True


def zimage_conditioning_report_ready(path: Path) -> bool:
    report = read_json(path)
    if report.get("ready") is not True:
        return False
    if report.get("blockers") not in ([], None):
        return False
    if report.get("accepted_conditioning_parity") is not False:
        return False
    hashes = report.get("idat_sha256")
    if not isinstance(hashes, dict):
        return False
    required = (
        "cfg_low_empty_negative",
        "cfg_high_empty_negative",
        "cfg_high_with_negative",
    )
    values = [str(hashes.get(label) or "") for label in required]
    if any(not value for value in values) or len(set(values)) != len(values):
        return False
    cases = report.get("cases")
    if not isinstance(cases, dict):
        return False
    for label in required:
        case = dict_or_empty(cases.get(label))
        evidence = dict_or_empty(case.get("evidence"))
        png = dict_or_empty(evidence.get("png"))
        manifest = dict_or_empty(evidence.get("manifest"))
        run_identity = dict_or_empty(manifest.get("run_identity"))
        visual_health = dict_or_empty(case.get("visual_health"))
        png_path = evidence_path(png.get("path"))
        if not png_path.is_file() or visual_health.get("ready") is not True:
            return False
        if run_identity.get("executed_sampler") != "flowmatch_euler":
            return False
        if run_identity.get("executed_scheduler") != "simple_flowmatch":
            return False
    return True


def server_conditioning_report_ready(
    path: Path,
    model: str,
    *,
    schema: str,
    executed_sampler: str,
    executed_scheduler: str,
    width: int,
    height: int,
) -> bool:
    report = read_json(path)
    summary = dict_or_empty(report.get("summary"))
    if summary.get("exit_ok") is not True:
        return False
    if summary.get("accepted_conditioning_parity") is not False:
        return False
    if summary.get("accepted_sampler_parity") is not False:
        return False
    ready_models = summary.get("ready_models")
    if not isinstance(ready_models, list) or model not in ready_models:
        return False
    models = dict_or_empty(report.get("models"))
    model_report = dict_or_empty(models.get(model))
    if model_report.get("ready") is not True:
        return False
    if model_report.get("blockers") not in ([], None):
        return False
    if model_report.get("accepted_conditioning_parity") is not False:
        return False
    if model_report.get("accepted_sampler_parity") is not False:
        return False
    hashes = model_report.get("idat_sha256")
    if not isinstance(hashes, dict):
        return False
    required = (
        "cfg_low_empty_negative",
        "cfg_high_empty_negative",
        "cfg_high_with_negative",
    )
    values = [str(hashes.get(label) or "") for label in required]
    if any(not value for value in values) or len(set(values)) != len(values):
        return False
    cases = model_report.get("cases")
    if not isinstance(cases, dict):
        return False
    for label in required:
        case = dict_or_empty(cases.get(label))
        if case.get("ok") is not True or case.get("blockers") not in ([], None):
            return False
        request = dict_or_empty(case.get("request"))
        job = dict_or_empty(case.get("job"))
        png = dict_or_empty(job.get("png"))
        manifest = dict_or_empty(case.get("manifest"))
        run_identity = dict_or_empty(manifest.get("run_identity"))
        mojo = dict_or_empty(manifest.get("mojo"))
        png_path = evidence_path(job.get("output_path"))
        manifest_path = evidence_path(job.get("manifest_path"))
        if not png_path.is_file() or not manifest_path.is_file():
            return False
        if png.get("width") != width or png.get("height") != height:
            return False
        if png.get("genparams_present") is not True:
            return False
        if manifest.get("schema") != schema:
            return False
        if manifest.get("accepted_sampler_parity") is not False:
            return False
        if manifest.get("accepted_speed_parity") is not False:
            return False
        if run_identity.get("prompt") != request.get("prompt"):
            return False
        if run_identity.get("negative") != request.get("negative"):
            return False
        if run_identity.get("requested_sampler") != request.get("sampler"):
            return False
        if run_identity.get("requested_scheduler") != request.get("scheduler"):
            return False
        if run_identity.get("executed_sampler") != executed_sampler:
            return False
        if run_identity.get("executed_scheduler") != executed_scheduler:
            return False
        if not positive_number(mojo.get("peak_vram_mib")):
            return False
        if not positive_number(mojo.get("total_wall_seconds")):
            return False
    return True


def zimage_dpmpp2m_sgm_uniform_report_ready(path: Path) -> bool:
    report = read_json(path)
    if report.get("ready") is not True:
        return False
    if report.get("blockers") not in ([], None):
        return False
    if report.get("accepted_sampler_parity") is not False:
        return False
    smoke = dict_or_empty(report.get("dpmpp2m_sgm_uniform_smoke"))
    request = dict_or_empty(smoke.get("request"))
    evidence = dict_or_empty(smoke.get("evidence"))
    png = dict_or_empty(evidence.get("png"))
    manifest = dict_or_empty(evidence.get("manifest"))
    run_identity = dict_or_empty(manifest.get("run_identity"))
    mojo = dict_or_empty(manifest.get("mojo"))
    trace = dict_or_empty(run_identity.get("sampler_trace"))
    png_path = evidence_path(png.get("path"))
    sigma_trace = run_identity.get("sigma_trace")
    if not png_path.is_file():
        return False
    if request.get("sampler") != "dpmpp_2m" or request.get("scheduler") != "sgm_uniform":
        return False
    if png.get("width") != 512 or png.get("height") != 512:
        return False
    if run_identity.get("requested_sampler") != "dpmpp_2m":
        return False
    if run_identity.get("requested_scheduler") != "sgm_uniform":
        return False
    if run_identity.get("executed_sampler") != "dpmpp_2m":
        return False
    if run_identity.get("executed_scheduler") != "sgm_uniform_flowmatch":
        return False
    if not isinstance(sigma_trace, list) or len(sigma_trace) != 5:
        return False
    if trace.get("algorithm") != "dpmpp_2m":
        return False
    if trace.get("schedule_source") != "zimage_comfy_sgm_uniform_sigmas":
        return False
    if int(trace.get("dpmpp_update_steps") or 0) < 1:
        return False
    if int(trace.get("dpmpp_second_order_steps") or 0) < 1:
        return False
    if not positive_number(mojo.get("peak_vram_mib")):
        return False
    return positive_number(mojo.get("denoise_seconds_per_step"))


def weighted_prompt_fail_loud_report_ready(path: Path) -> bool:
    report = read_json(path)
    if report.get("ready") is not True:
        return False
    if report.get("blockers") not in ([], None):
        return False
    if report.get("accepted_prompt_weight_parity") is not False:
        return False
    if report.get("accepted_conditioning_parity") is not False:
        return False
    if report.get("fail_loud_prompt_weight_gate") is not True:
        return False
    if report.get("prequeue_rejection") is not True:
        return False
    if report.get("job_count_unchanged") is not True:
        return False
    response = dict_or_empty(report.get("response"))
    if response.get("http_status") != 422:
        return False
    error = str(response.get("error") or response.get("text") or "")
    return (
        "weighted prompt syntax is not supported" in error
        and "conditioning_weights_applied=false" in error
    )


def latent_batch_fail_loud_report_ready(path: Path) -> bool:
    report = read_json(path)
    if report.get("ready") is not True:
        return False
    if report.get("blockers") not in ([], None):
        return False
    if report.get("accepted_latent_batch_parity") is not False:
        return False
    if report.get("flat_images_serial_fanout_only") is not True:
        return False
    if report.get("prequeue_rejection") is not True:
        return False
    if report.get("job_count_unchanged") is not True:
        return False
    cases = report.get("cases")
    if not isinstance(cases, list) or len(cases) < 2:
        return False
    seen: set[str] = set()
    for item in cases:
        if not isinstance(item, dict):
            return False
        case_id = str(item.get("case") or "")
        error = str(item.get("error") or "")
        if item.get("http_status") != 501:
            return False
        if "latent-batch execution" not in error:
            return False
        if case_id == "empty_latent_batch_size" and "EmptyLatentImage" not in error:
            return False
        if case_id == "repeat_latent_batch" and "RepeatLatentBatch" not in error:
            return False
        seen.add(case_id)
    return {"empty_latent_batch_size", "repeat_latent_batch"}.issubset(seen)


def disabled_model_fail_loud_report_ready(path: Path) -> bool:
    report = read_json(path)
    if report.get("ready") is not True:
        return False
    if report.get("blockers") not in ([], None):
        return False
    if report.get("accepted_disabled_family_runtime") is not False:
        return False
    if report.get("prequeue_rejection") is not True:
        return False
    if report.get("job_count_unchanged") is not True:
        return False
    cases = report.get("cases")
    if not isinstance(cases, list) or len(cases) < 2:
        return False
    seen: set[str] = set()
    for item in cases:
        if not isinstance(item, dict):
            return False
        case_id = str(item.get("case") or "")
        error = str(item.get("error") or "")
        if item.get("http_status") != 501:
            return False
        if "not runnable" not in error:
            return False
        seen.add(case_id)
    return {"qwen_image_edit", "ltx2_video"}.issubset(seen)


def server_qwen_product_report_ready(path: Path) -> bool:
    report = read_json(path)
    summary = report.get("summary")
    if not isinstance(summary, dict) or summary.get("exit_ok") is not True:
        return False
    preflight_profiles = report.get("preflight_capability_profiles")
    if not isinstance(preflight_profiles, list):
        return False
    profile_seen = False
    for item in preflight_profiles:
        if not isinstance(item, dict) or item.get("case") != "qwen_admitted_profile":
            continue
        profile = item.get("capability_profile")
        if item.get("ok") is not True or item.get("http_status") != 200:
            return False
        if not isinstance(profile, dict):
            return False
        if profile.get("backend") != "qwenimage" or profile.get("production_status") != "admitted":
            return False
        features = profile.get("features")
        if not isinstance(features, dict):
            return False
        text_to_image = features.get("text_to_image")
        if not isinstance(text_to_image, dict) or text_to_image.get("supported") is not True:
            return False
        profile_seen = True
    if not profile_seen:
        return False
    samplers = report.get("samplers")
    if not isinstance(samplers, dict) or samplers.get("http_status") != 200:
        return False
    body = samplers.get("body")
    if not isinstance(body, dict):
        return False
    backends = body.get("backends")
    if not isinstance(backends, list):
        return False
    for entry in backends:
        if not isinstance(entry, dict) or entry.get("backend") != "qwenimage":
            continue
        return (
            "euler" in list(entry.get("supported_samplers") or [])
            and "flowmatch_euler" in list(entry.get("supported_samplers") or [])
            and "simple" in list(entry.get("supported_schedulers") or [])
            and entry.get("executed_sampler") == "qwenimage_flowmatch_euler"
            and entry.get("executed_scheduler") == "qwenimage_simple_flowmatch"
            and "bounded 1024x1024" in str(entry.get("reason") or "")
        )
    return False


def server_zimage_karras_prequeue_report_ready(path: Path) -> bool:
    report = read_json(path)
    summary = report.get("summary")
    if not isinstance(summary, dict) or summary.get("exit_ok") is not True:
        return False
    prequeue = report.get("prequeue_rejections")
    if not isinstance(prequeue, list):
        return False
    case_seen = False
    for item in prequeue:
        if not isinstance(item, dict):
            return False
        if item.get("case") != "zimage_karras_scheduler":
            continue
        error = str(item.get("error") or "")
        if item.get("ok") is not True:
            return False
        if item.get("http_status") != 400:
            return False
        if item.get("job_count_before") != item.get("job_count_after"):
            return False
        for part in ("zimage", "unsupported scheduler", "karras"):
            if part not in error.lower():
                return False
        case_seen = True
    if not case_seen:
        return False

    samplers = report.get("samplers")
    if not isinstance(samplers, dict) or samplers.get("http_status") != 200:
        return False
    body = samplers.get("body")
    if not isinstance(body, dict):
        return False
    backends = body.get("backends")
    if not isinstance(backends, list):
        return False
    for entry in backends:
        if not isinstance(entry, dict) or entry.get("backend") != "zimage":
            continue
        schedulers = entry.get("supported_schedulers")
        return (
            isinstance(schedulers, list)
            and "sgm_uniform" in schedulers
            and "karras" not in schedulers
            and entry.get("unsupported_policy") == "fail_loud"
            and entry.get("accepted_sampler_parity") is False
        )
    return False


def server_capabilities_report_ready(path: Path) -> bool:
    report = read_json(path)
    summary = report.get("summary")
    if not isinstance(summary, dict) or summary.get("exit_ok") is not True:
        return False
    if summary.get("failed_capability_cases") not in ([], None):
        return False
    if summary.get("failed_preflight_capability_profile_cases") not in ([], None):
        return False
    capabilities = report.get("capabilities")
    if not isinstance(capabilities, dict) or capabilities.get("http_status") != 200:
        return False
    coverage = capabilities.get("coverage")
    if not isinstance(coverage, dict) or coverage.get("ok") is not True:
        return False
    if summary.get("failed_capability_rejection_cases") not in ([], None):
        return False
    preflight_profiles = report.get("preflight_capability_profiles")
    if not isinstance(preflight_profiles, list) or len(preflight_profiles) < 6:
        return False
    profile_cases = {str(item.get("case") or ""): item for item in preflight_profiles if isinstance(item, dict)}
    for case_name, backend, status in (
        ("zimage_admitted_profile", "zimage", "admitted"),
        ("qwen_admitted_profile", "qwenimage", "admitted"),
        ("klein9b_admitted_profile", "flux2", "admitted"),
        ("klein4b_blocked_profile", "flux2", "blocked"),
        ("ideogram_negative_prompt_profile", "ideogram4", "admitted"),
        ("zimage_raw_controlnet_profile", "zimage", "admitted"),
        ("zimage_workflow_unsupported_node_profile", "zimage", "admitted"),
    ):
        item = profile_cases.get(case_name)
        if not isinstance(item, dict) or item.get("ok") is not True:
            return False
        if case_name == "zimage_workflow_unsupported_node_profile":
            if item.get("http_status") != 501 or item.get("rejection_stage") != "workflow_lowering":
                return False
        profile = item.get("capability_profile")
        if not isinstance(profile, dict):
            return False
        if profile.get("schema") != "serenity.capability_profile.v1":
            return False
        if profile.get("backend") != backend or profile.get("production_status") != status:
            return False
    capability_rejections = report.get("capability_rejections")
    if not isinstance(capability_rejections, list) or len(capability_rejections) < 30:
        return False
    rejection_cases = {str(item.get("case") or ""): item for item in capability_rejections if isinstance(item, dict)}
    for case_name in (
        "zimage_image_to_image_disabled",
        "zimage_controlnet_disabled",
        "zimage_bbox_prompt_json_disabled",
        "sdxl_vae_override_disabled",
        "ideogram4_negative_prompt_disabled",
        "flux_multi_lora_disabled",
    ):
        item = rejection_cases.get(case_name)
        if not isinstance(item, dict) or item.get("ok") is not True:
            return False
        if item.get("job_count_before") != item.get("job_count_after"):
            return False
    body = capabilities.get("body")
    if not isinstance(body, dict):
        return False
    if body.get("schema") != "serenity.capabilities.v1":
        return False
    global_limits = body.get("global_limits")
    if not isinstance(global_limits, dict):
        return False
    if global_limits.get("txt2img_only") is not True:
        return False
    if global_limits.get("image_to_image") is not False:
        return False
    if global_limits.get("runtime_dependency_on_external_repos") is not False:
        return False
    backends = body.get("backends")
    if not isinstance(backends, list):
        return False
    found = {entry.get("backend"): entry for entry in backends if isinstance(entry, dict)}
    for backend in ("zimage", "ideogram4", "sdxl", "anima", "sd3", "flux2", "sensenova"):
        entry = found.get(backend)
        if not isinstance(entry, dict):
            return False
        features = entry.get("features")
        if not isinstance(features, dict):
            return False
        for name in ("image_to_image", "inpaint", "image_conditioning", "vae_override", "outpaint"):
            feature = features.get(name)
            if (
                not isinstance(feature, dict)
                or feature.get("supported") is not False
                or feature.get("policy") != "fail_loud"
            ):
                return False
    zimage = found["zimage"]
    z_samplers = zimage.get("samplers")
    if not isinstance(z_samplers, dict):
        return False
    z_sched = z_samplers.get("supported_schedulers")
    if not isinstance(z_sched, list) or "sgm_uniform" not in z_sched or "karras" in z_sched:
        return False
    ideogram_features = found["ideogram4"].get("features")
    if not isinstance(ideogram_features, dict):
        return False
    if ideogram_features.get("bbox_prompt_json", {}).get("supported") is not True:
        return False
    blocked = body.get("blocked_families")
    if not isinstance(blocked, list):
        return False
    flux_block = next(
        (entry for entry in blocked if isinstance(entry, dict) and entry.get("backend") == "flux"),
        None,
    )
    if not isinstance(flux_block, dict) or flux_block.get("production_status") != "blocked":
        return False
    if "6/20" not in str(flux_block.get("reason") or ""):
        return False
    return True


def server_sd3_worker_and_flux_block_report_ready(path: Path) -> bool:
    report = read_json(path)
    summary = report.get("summary")
    if not isinstance(summary, dict) or summary.get("exit_ok") is not True:
        return False

    samplers = report.get("samplers")
    if not isinstance(samplers, dict) or samplers.get("http_status") != 200:
        return False
    body = samplers.get("body")
    if not isinstance(body, dict):
        return False
    backends = body.get("backends")
    if not isinstance(backends, list):
        return False

    expected = {"sd3": ("sd3_flowmatch_euler", "sd3_simple_flowmatch")}
    seen: set[str] = set()
    flux_blocked = False
    for entry in backends:
        if not isinstance(entry, dict):
            return False
        backend = str(entry.get("backend") or "")
        if backend == "flux":
            flux_blocked = (
                entry.get("production_status") == "blocked"
                and entry.get("supported_samplers") == []
                and entry.get("supported_schedulers") == []
                and "6/20" in str(entry.get("reason") or "")
            )
            continue
        if backend not in expected:
            continue
        want_sampler, want_scheduler = expected[backend]
        supported_samplers = entry.get("supported_samplers")
        supported_schedulers = entry.get("supported_schedulers")
        reason = str(entry.get("reason") or "")
        if not isinstance(supported_samplers, list) or not isinstance(
            supported_schedulers, list
        ):
            return False
        if entry.get("accepted_sampler_parity") is not False:
            return False
        if entry.get("unsupported_policy") != "fail_loud":
            return False
        if entry.get("executed_sampler") != want_sampler:
            return False
        if entry.get("executed_scheduler") != want_scheduler:
            return False
        if "euler" not in supported_samplers or "flowmatch_euler" not in supported_samplers:
            return False
        if "simple" not in supported_schedulers:
            return False
        if "bounded 1024x1024" not in reason:
            return False
        seen.add(backend)

    if seen != set(expected) or not flux_blocked:
        return False

    prequeue = report.get("prequeue_rejections")
    if not isinstance(prequeue, list):
        return False
    for item in prequeue:
        if not isinstance(item, dict) or item.get("case") != "flux_blocked":
            continue
        error = str(item.get("error") or "")
        return (
            item.get("ok") is True
            and item.get("http_status") == 400
            and item.get("job_count_before") == item.get("job_count_after")
            and "Flux.1-dev" in error
            and "6/20" in error
        )
    return False


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
    workflow_graph = REPO / "serenitymojo/serve/workflow_graph.mojo"
    backend = REPO / "serenitymojo/serve/backend.mojo"
    zimage = REPO / "serenitymojo/serve/zimage_backend.mojo"
    qwen = REPO / "serenitymojo/serve/qwenimage_backend.mojo"
    ideogram = REPO / "serenitymojo/serve/ideogram4_backend.mojo"
    klein = REPO / "serenitymojo/serve/klein_backend.mojo"
    sampler_registry = REPO / "serenitymojo/sampling/sampler_registry.mojo"
    variation_noise = REPO / "serenitymojo/sampling/variation_noise.mojo"
    dispatch = REPO / "serenitymojo/serve/dispatch_backend.mojo"
    sd3_backend = REPO / "serenitymojo/serve/sd3_backend.mojo"
    flux_backend = REPO / "serenitymojo/serve/flux_backend.mojo"
    sd3_worker = REPO / "serenitymojo/serve/serenity_worker_sd3.mojo"
    flux_worker = REPO / "serenitymojo/serve/serenity_worker_flux.mojo"
    stub = REPO / "serenitymojo/serve/stub_backend.mojo"
    harness = REPO / "serenitymojo/sampling/product_sampler_harness.mojo"
    harness_doc = REPO / "serenitymojo/docs/SAMPLER_PRODUCT_HARNESS_2026-06-05.md"
    parity_doc = (
        REPO / "serenitymojo/docs/SWARMUI_SAMPLER_PARITY_MAP_2026-06-12.md"
    )
    server_main = REPO / "serenity-server/crates/server/src/main.rs"
    server_capabilities = REPO / "serenity-server/crates/server/src/capabilities.rs"
    samplers_asset = REPO / "serenity-server/crates/server/src/assets/samplers_v1.json"
    zimage_conditioning_report = REPO / "output/checks/zimage_conditioning_readiness.json"
    sdxl_conditioning_report = REPO / "output/checks/sdxl_conditioning_gate.json"
    sd3_conditioning_report = REPO / "output/checks/sd3_conditioning_gate.json"
    anima_conditioning_report = REPO / "output/checks/anima_conditioning_gate.json"
    weighted_prompt_report = REPO / "output/checks/weighted_prompt_fail_loud.json"
    latent_batch_report = REPO / "output/checks/latent_batch_fail_loud.json"
    disabled_model_report = REPO / "output/checks/disabled_model_fail_loud.json"
    server_prequeue_report = REPO / "output/checks/serenity_server_t2i_product_gate_prequeue_latest.json"
    zimage_dpmpp2m_sgm_report = REPO / "output/checks/zimage_dpmpp2m_sgm_uniform_readiness.json"

    daemon_text = read_text(daemon)
    workflow_graph_text = read_text(workflow_graph)
    backend_text = read_text(backend)
    zimage_text = read_text(zimage)
    qwen_text = read_text(qwen)
    ideogram_text = read_text(ideogram)
    klein_text = read_text(klein)
    sampler_registry_text = read_text(sampler_registry)
    variation_noise_text = read_text(variation_noise)
    dispatch_text = read_text(dispatch)
    server_main_text = read_text(server_main)
    server_capabilities_text = read_text(server_capabilities)
    samplers_asset_text = read_text(samplers_asset)
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
            workflow_graph,
            has_all(
                workflow_graph_text,
                [
                    "apply_workflow_params",
                    "apply_typed_workflow_graph",
                    '"KSampler"',
                    '"sampler_name"',
                    '"denoise"',
                    '"EmptyLatentImage"',
                    "unsupported workflow graph node",
                ],
            ),
            "Workflow graph module maps a constrained Comfy KSampler-like graph and rejects unknown nodes.",
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
                    "uni_pc",
                    "uni_pc_bh2",
                    "ideogram4_logitnormal_euler",
                    "ideogram4_logitnormal",
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
            "Supported feature foundation",
            "Rust server exposes a shared /v1/capabilities product contract",
            server_prequeue_report,
            server_capabilities_report_ready(server_prequeue_report),
            "The Rust-server product gate fetches /v1/capabilities and verifies admitted backends, dimensions, sampler/scheduler subsets, disabled img2img/inpaint/image-conditioning/VAE/refiner/upscale/control features, Ideogram bbox prompt JSON, and fail-loud policy.",
            "Keep UI controls and /v1/preflight wired to the same capability contract before broadening admitted features.",
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
            "Z-Image and Qwen apply variation_seed/variation_strength to initial latent noise; accepted variation parity still requires artifact evidence per backend.",
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
                    "supported sizes are 512x512 and 1024x1024",
                    "len(params.loras)",
                    "merge_zimage_lora_sets_for_inference",
                    "self.params.init_image",
                    "self.params.creativity",
                    "img2img_applied",
                    "denoise_start_step",
                    "_build_zimage_sigmas",
                    "self.params.sigma_shift",
                    "zimage_comfy_simple_sigmas",
                    "zimage_comfy_sgm_uniform_sigmas",
                    "self.params.seed",
                    "self.params.negative",
                    "_cfg_pred_overlay",
                    "sampler_admission_for_backend",
                    "scheduler_admission_for_backend",
                    "unsupported sampler",
                    "executed_sampler",
                    "dpmpp_2m_step",
                    "UniPcMultistepScheduler",
                    "ComfyUniPcMultistepScheduler",
                    "_build_comfy_unipc_sigmas",
                    "DISCARD_PENULTIMATE_SIGMA_SAMPLERS",
                    "from_sigmas",
                    "sampler_trace",
                    "solver_variant",
                    "SigmaConvert",
                    "dpmpp_update_steps",
                    "dpmpp_second_order_steps",
                    "unipc_update_steps",
                    "unipc_second_order_steps",
                    "unipc_third_order_steps",
                    "schedule_source",
                    "self.params.image_index",
                    "self.params.image_count",
                ],
            ),
            "Z-Image has real subset behavior, registry-backed admission, SwarmUI/Comfy-aligned Euler/simple and sgm_uniform flow-match sigmas with sigma_shift, bounded DPM++ 2M execution, bounded generic UniPC bh1/order<=3 execution, bounded UniPC bh2 execution, Comfy DISCARD_PENULTIMATE sigma prep plus SigmaConvert wording for UniPC schedules, and bounded flat img2img/creativity artifact evidence.",
            "Accepted sampler parity needs per-sampler artifact evidence and executed sampler metadata.",
            severity="warning",
        )
    )
    markers.append(
        marker(
            "Z-Image backend subset",
            "Z-Image DPM++ 2M executes on sgm_uniform",
            zimage_dpmpp2m_sgm_report,
            zimage_dpmpp2m_sgm_uniform_report_ready(zimage_dpmpp2m_sgm_report),
            "Z-Image DPM++ 2M has a dedicated daemon artifact for the admitted sgm_uniform flow-match schedule, with Comfy sigma-source metadata, DPM++ update trace, timing, and VRAM evidence.",
            "Keep sampler parity blocked until every exposed sampler/scheduler pair has comparable artifact evidence or fails loud.",
            severity="warning",
        )
    )
    markers.append(
        marker(
            "Z-Image backend subset",
            "Z-Image Karras scheduler fails loud before enqueue",
            server_prequeue_report,
            server_zimage_karras_prequeue_report_ready(server_prequeue_report),
            "The Rust-server product gate rejects a Z-Image Karras scheduler request before job fanout and /v1/samplers keeps Karras out of Z-Image supported_schedulers.",
            "Keep sampler parity blocked until Karras has a real Z-Image scheduler builder and artifact/timing/VRAM evidence.",
            severity="warning",
        )
    )
    markers.append(
        marker(
            "Qwen backend subset",
            "Qwen bounded txt2img route is product-admitted",
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
                    "cfg_qwen_device",
                    "sampler_admission_for_backend",
                    "scheduler_admission_for_backend",
                    "unsupported sampler",
                ],
            ),
            "Qwen txt2img has a bounded 1024x1024 Euler/simple route with device CFG, negative prompt plumbing, and local fail-loud checks.",
            "Do not treat the bounded route as full Qwen sampler parity until separate artifact, timing, VRAM, quality, and workflow gates pass.",
            severity="warning",
        )
    )
    markers.append(
        marker(
            "Conditioning parity evidence",
            "Z-Image CFG and negative prompt affect output payload",
            zimage_conditioning_report,
            zimage_conditioning_report_ready(zimage_conditioning_report),
            "Z-Image same-seed artifact smoke proves CFG and negative-prompt changes alter the PNG payload while preserving manifest metadata.",
            "Keep conditioning parity blocked for prompt weights and unproven model families until each has artifact evidence.",
            severity="warning",
        )
    )
    markers.append(
        marker(
            "Conditioning parity evidence",
            "SDXL CFG and negative prompt affect Rust-server output payload",
            sdxl_conditioning_report,
            server_conditioning_report_ready(
                sdxl_conditioning_report,
                "sdxl",
                schema="serenity.sdxl.daemon_result.v1",
                executed_sampler="sdxl_euler",
                executed_scheduler="normal",
                width=1024,
                height=1024,
            ),
            "SDXL same-seed Rust-server artifact gate proves CFG and negative-prompt changes alter the PNG payload while preserving manifest timing, VRAM, and sampler metadata.",
            "Keep conditioning parity blocked for prompt weights and model families that still lack artifact evidence.",
            severity="warning",
        )
    )
    markers.append(
        marker(
            "Conditioning parity evidence",
            "SD3 CFG and negative prompt affect Rust-server output payload",
            sd3_conditioning_report,
            server_conditioning_report_ready(
                sd3_conditioning_report,
                "sd3",
                schema="serenity.sd3.daemon_result.v1",
                executed_sampler="sd3_flowmatch_euler",
                executed_scheduler="sd3_simple_flowmatch",
                width=1024,
                height=1024,
            ),
            "SD3 same-seed Rust-server artifact gate proves CFG and negative-prompt changes alter the PNG payload while preserving manifest timing, VRAM, sampler metadata, and 5x5 low-memory VAE decode metadata.",
            "Keep conditioning parity blocked for prompt weights and model families that still lack artifact evidence.",
            severity="warning",
        )
    )
    markers.append(
        marker(
            "Conditioning parity evidence",
            "Anima CFG and negative prompt affect Rust-server output payload",
            anima_conditioning_report,
            server_conditioning_report_ready(
                anima_conditioning_report,
                "anima",
                schema="serenity.anima.daemon_result.v1",
                executed_sampler="anima_euler",
                executed_scheduler="normal",
                width=1024,
                height=1024,
            ),
            "Anima same-seed Rust-server artifact gate proves CFG and negative-prompt changes alter the PNG payload while preserving manifest timing, VRAM, and sampler metadata.",
            "Keep conditioning parity blocked for prompt-weight math and broad prompt-weight syntax parity.",
            severity="warning",
        )
    )
    markers.append(
        marker(
            "Conditioning parity evidence",
            "Weighted prompt syntax fails loud before enqueue",
            weighted_prompt_report,
            weighted_prompt_fail_loud_report_ready(weighted_prompt_report)
            and has_all(
                daemon_text,
                [
                    "_reject_unapplied_prompt_weights",
                    "weighted prompt syntax is not supported",
                    "conditioning_weights_applied=false",
                ],
            ),
            "Weighted prompt syntax is rejected with HTTP 422 before job fanout while prompt-syntax metadata still records conditioning_weights_applied=false.",
            "Do not accept prompt-weight parity until conditioning-weight math has artifact evidence; until then, weighted syntax must remain a prequeue fail-loud path.",
            severity="warning",
        )
    )
    markers.append(
        marker(
            "Images and latent batch",
            "Comfy latent batch syntax fails loud before enqueue",
            latent_batch_report,
            latent_batch_fail_loud_report_ready(latent_batch_report)
            and has_all(
                workflow_graph_text,
                [
                    "EmptyLatentImage batch_size>1",
                    "RepeatLatentBatch requires real Comfy",
                    "latent-batch execution",
                    "flat images=N",
                    "product fanout",
                ],
            ),
            "Workflow graph EmptyLatentImage.batch_size>1 and RepeatLatentBatch are rejected with HTTP 501 before job fanout; flat images=N remains the serial output path.",
            "Do not accept Comfy latent-batch parity until a backend denoises real batched latents in one execution path.",
            severity="warning",
        )
    )
    markers.append(
        marker(
            "Qwen edit and video quarantine",
            "Qwen txt2img is bounded while edit/video families fail loud before enqueue",
            disabled_model_report,
            disabled_model_fail_loud_report_ready(disabled_model_report)
            and server_qwen_product_report_ready(server_prequeue_report)
            and has_all(
                daemon_text + sampler_registry_text + dispatch_text,
                [
                    'sampler_backend == "disabled"',
                    "qwenimage_flowmatch_euler",
                    "m.find(\"qwen\")",
                    "Qwen-Image-Edit execution is disabled",
                    "LTX/LTX2 execution is disabled",
                ],
            ),
            "Mojo daemon admits bounded Qwen txt2img, rejects Qwen-Edit/LTX-style known-disabled models before job fanout, and publishes Qwen sampler inventory with narrow support.",
            "Keep Qwen edit/video parity blocked until separate artifact, timing, VRAM, quality, and workflow gates pass.",
            severity="warning",
        )
    )
    markers.append(
        marker(
            "Ideogram4 backend subset",
            "Ideogram4 executes bounded logit-normal/simple Euler and rejects wider controls",
            ideogram,
            has_all(
                ideogram_text,
                [
                    "ideogram4_logitnormal_euler",
                    "ideogram4_logitnormal",
                    "ideogram4_simple_flowmatch",
                    "_build_ideogram4_simple_sigmas",
                    "ideogram4_comfy_simple_aura_flow",
                    "cfg_override",
                    "sigma_shift",
                    "accepted_sampler_parity",
                    "accepted_speed_parity",
                    "negative prompt is not supported",
                    "LoRA is not supported",
                    "init image is not supported",
                    "creativity/denoise control is not supported",
                    "variation noise is not supported",
                    "cfg must be positive",
                    "1024x1024",
                ],
            ),
            "Ideogram4 has bounded native inference, a logit-normal path, and a Comfy simple AuraFlow scheduler path for the imported workflow, but accepted sampler/speed parity remains false.",
            "Full Ideogram4 parity still needs paired Comfy artifact evidence, prompt-builder coverage, and broader request-surface coverage.",
            severity="warning",
        )
    )
    markers.append(
        marker(
            "Model/backend mapping",
            "Standalone dispatch has bounded real routes and loud disabled families",
            dispatch,
            has_all(
                dispatch_text,
                [
                    "KIND_ZIMAGE",
                    "KIND_QWEN",
                    "KIND_IDEOGRAM4",
                    "KIND_KLEIN",
                    "KIND_SDXL",
                    "KIND_ANIMA",
                    "KleinBackend",
                    "QwenImageBackend",
                    "SampleCliBackend(String(\"sdxl\"))",
                    "SampleCliBackend(String(\"anima\"))",
                    "_known_disabled_model_reason",
                    "Qwen-Image-Edit execution is disabled",
                    "SD3/SD3.5 execution is metadata-only",
                    "LTX/LTX2 execution is disabled",
                ],
            ),
            "Standalone Mojo dispatch includes bounded Z-Image, Ideogram4, Flux2/Klein, SDXL CLI, and Anima CLI routes, while keeping heavyweight SD3/Flux1 worker stacks out of the monolithic dispatch path.",
            "Only widen standalone dispatch after model-specific artifact, timing, VRAM, build-memory, and sampler evidence exists.",
            severity="warning",
        )
    )
    markers.append(
        marker(
            "Model/backend mapping",
            "Rust server maps SD3 and Flux to per-kind Mojo workers",
            server_capabilities,
            has_all(
                server_main_text + server_capabilities_text,
                [
                    "ModelFamily::Sd3",
                    "ModelFamily::Flux",
                    "serenity_worker_sd3",
                    "serenity_worker_flux",
                    'production_entry: "serenitymojo/serve/sd3_backend.mojo"',
                    'production_entry: "serenitymojo/serve/flux_backend.mojo"',
                    "fn reject_negative",
                    "negative prompt is not supported by this production route",
                    "worker_dispatch_uses_admitted_model_family_classifier",
                ],
            )
            and has_all(
                samplers_asset_text,
                [
                    '"backend": "sd3"',
                    '"backend": "flux"',
                    '"executed_sampler": "sd3_flowmatch_euler"',
                    '"production_status": "blocked"',
                ],
            )
            and sd3_backend.is_file()
            and flux_backend.is_file()
            and sd3_worker.is_file()
            and flux_worker.is_file()
            and server_sd3_worker_and_flux_block_report_ready(server_prequeue_report),
            "Rust control plane admits SD3 through a per-kind process worker and keeps Flux.1-dev visible but blocked until the real memory gate passes.",
            "Run scripts/check_serenity_server_t2i_product_gate.py --all-admitted --strict-production and require manifest-backed artifacts, timings, VRAM, and metadata for every admitted family before accepting full model coverage.",
            severity="warning",
        )
    )
    markers.append(
        marker(
            "Flux2/Klein backend subset",
            "Klein staged daemon route has real artifact evidence",
            klein,
            has_all(
                klein_text,
                [
                    "KleinBackend",
                    "process-separated Qwen3 cap-cache precache",
                    "KLEIN_PRECACHE_BIN",
                    "KLEIN_SAMPLER_BIN",
                    "unsupported sampler",
                    "unsupported scheduler",
                    "reference_latent_count",
                    "encode_png_with_text",
                    "serenity.genparams.v1",
                ],
            )
            and klein_report_ready(
                REPO / "output/checks/klein9b_lora_daemon_smoke.json",
                "flux2-klein-9b.safetensors",
            )
            and klein_report_ready(
                REPO / "output/checks/klein4b_reference_edit_daemon_smoke.json",
                "flux2-klein-4b.safetensors",
                expected_mode="reference_latent_edit",
            )
            and klein_report_ready(
                REPO / "output/checks/klein9b_reference_edit_daemon_smoke.json",
                "flux2-klein-9b.safetensors",
                expected_mode="reference_latent_edit",
            )
            and klein_report_ready(
                REPO / "output/checks/klein9b_lora_reference_edit_daemon_smoke.json",
                "flux2-klein-9b.safetensors",
                expected_mode="reference_latent_edit",
            ),
            "Klein has bounded daemon artifacts for 9B LoRA txt2img, 4B ReferenceLatent, 9B ReferenceLatent, and 9B LoRA ReferenceLatent.",
            "Keep Flux2/Klein marked bounded until sampler/scheduler, timing, VRAM, and CLI limits are accepted per route.",
            severity="warning",
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
                stub_text + zimage_text + qwen_text + klein_text,
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
            "blocker": "Z-Image now has bounded DPM++ 2M, generic UniPC bh1/order<=3, and UniPC bh2 paths on admitted simple and sgm_uniform flow-match schedules, including dedicated DPM++ 2M + sgm_uniform artifact evidence, but Karras, ancestral, SDE, CFG++, and the rest of the SwarmUI/Comfy sampler catalog still lack distinct daemon denoise loops.",
            "acceptance_gate": "Wire each accepted sampler/scheduler pair into a backend denoise loop and record requested versus executed values with artifact/timing/VRAM evidence.",
        },
        {
            "id": "latent_batch_execution",
            "severity": "P1",
            "blocker": "Flat images=N is expanded into indexed serial daemon jobs, and Comfy workflow latent-batch syntax now fails loud before enqueue instead of being treated as serial images. True Comfy-style batched latent execution is still not implemented as a single backend batch.",
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
            "blocker": "Standalone Mojo dispatch covers bounded Z-Image, Qwen txt2img, Ideogram4, Flux2/Klein staged artifacts, and fixed SDXL/Anima sample-CLI wrappers. The Rust control plane also maps SD3/SD3.5 and Flux1-dev to per-kind Mojo workers with sampler inventory and fail-loud prequeue checks. A strict all-admitted Rust-server gate needs current artifact, timing, and VRAM evidence for every admitted family; SD15, Chroma, ERNIE, full LTX2/video, HiDream, Qwen-Edit, and wider sampler variants remain missing or explicitly disabled.",
            "acceptance_gate": "Add or accept model routes only after each admitted model has manifest-backed artifact, timing, VRAM, metadata, sampler/scheduler, and failure-mode evidence.",
        },
        {
            "id": "conditioning_parity",
            "severity": "P1",
            "blocker": "Z-Image, SDXL, SD3, and Anima now have bounded artifact evidence that CFG and negative-prompt changes affect output payloads, and weighted prompt syntax now fails loud before enqueue instead of being persisted as a silent no-op. Prompt-weight conditioning math is still missing; Flux and Ideogram intentionally do not accept negative prompts in the current bounded production routes.",
            "acceptance_gate": "Per model, prove negative prompt, CFG scale, and weighted prompt behavior with artifact and metadata checks.",
        },
        {
            "id": "qwen_video_quarantine",
            "severity": "P0",
            "blocker": "Qwen txt2img is accepted only for the bounded 1024x1024 Euler/simple route. Qwen-Edit and video-generation requests remain rejected before enqueue instead of becoming accepted jobs, while video remains bounded DEV-smoke evidence only.",
            "acceptance_gate": "Keep Qwen-Edit, video sampler parity, and wider Qwen sampler aliases blocked until separate product gates provide real artifacts and resource evidence.",
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
