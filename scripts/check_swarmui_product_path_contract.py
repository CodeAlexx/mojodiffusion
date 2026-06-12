#!/usr/bin/env python3
"""No-CUDA guard for SwarmUI product-path parity.

Default mode is a report and exits 0. Use --strict to fail on P0 product-path
blockers, or --strict-all to fail on every remaining SwarmUI-level blocker.

This checker does not run CUDA, generate images, or claim parity. It makes the
current product-path gaps machine-readable so the daemon/UI path cannot quietly
claim SwarmUI parity while image generation still bypasses the fast runtime path
or video remains dtype-broken.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MOJO_LIBS = Path("/home/alex/MOJO-libs")

AUDIT_DOC = REPO / "SWARMUI_PARITY_AUDIT_MOJOLIB_MOJODIFFUSION_2026-06-12.md"
LEDGER_DOC = REPO / "serenitymojo/docs/SWARMUI_PRODUCT_PATH_LEDGER_2026-06-12.md"
SAMPLER_MAP_DOC = REPO / "serenitymojo/docs/SWARMUI_SAMPLER_PARITY_MAP_2026-06-12.md"
WORKFLOW_MAP_DOC = REPO / "serenitymojo/docs/COMFY_SWARM_WORKFLOW_PARITY_MAP_2026-06-12.md"
MODEL_GALLERY_LORA_MAP_DOC = REPO / "serenitymojo/docs/SWARMUI_MODEL_GALLERY_LORA_PARITY_MAP_2026-06-12.md"

DAEMON = REPO / "serenitymojo/serve/serenity_daemon.mojo"
MODEL_SCAN = REPO / "serenitymojo/serve/model_scan.mojo"
PROCESS_ISOLATED = REPO / "serenitymojo/serve/process_isolated_backend.mojo"
ZIMAGE_BACKEND = REPO / "serenitymojo/serve/zimage_backend.mojo"
QWEN_BACKEND = REPO / "serenitymojo/serve/qwenimage_backend.mojo"
STUB_BACKEND = REPO / "serenitymojo/serve/stub_backend.mojo"
ZIMAGE_DAEMON_PRODUCT_CHECK = REPO / "scripts/check_zimage_daemon_product_contract.py"
SAMPLER_SURFACE_CHECK = REPO / "scripts/check_swarmui_sampler_surface.py"
WORKFLOW_NODE_SURFACE_CHECK = REPO / "scripts/check_workflow_node_surface.py"
MODEL_GALLERY_LORA_SURFACE_CHECK = REPO / "scripts/check_model_gallery_lora_surface.py"
SAMPLER_REGISTRY = REPO / "serenitymojo/sampling/sampler_registry.mojo"
VARIATION_NOISE = REPO / "serenitymojo/sampling/variation_noise.mojo"
ZIMAGE_GENERATE = REPO / "serenitymojo/pipeline/zimage_generate.mojo"
ZIMAGE_LORA_BLOCK = REPO / "serenitymojo/models/zimage/lora_block.mojo"
ZIMAGE_STACK_LORA = REPO / "serenitymojo/models/zimage/zimage_stack_lora.mojo"
QWEN_DIT = REPO / "serenitymojo/models/dit/qwenimage_dit.mojo"
ATTENTION = REPO / "serenitymojo/ops/attention.mojo"
ATTENTION_FLASH = REPO / "serenitymojo/ops/attention_flash.mojo"
LTX2_HQ = REPO / "serenitymojo/pipeline/ltx2_t2v_av_hq.mojo"
TODO_DOC = REPO / "TODO.md"
HANDOFF_DOC = REPO / "serenitymojo/docs/HANDOFF_2026-06-12.md"
SAMPLER_HARNESS_DOC = REPO / "serenitymojo/docs/SAMPLER_PRODUCT_HARNESS_2026-06-05.md"
TRAINER_USE_STATUS = REPO / "MOJO_TRAINER_USE_STATUS.md"

P0 = "P0"
P1 = "P1"
P2 = "P2"
PASS = "PASS"


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
        return Check(
            False,
            severity,
            category,
            label,
            f"missing file: {path}",
            rel(path),
            acceptance,
        )
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


def check_absent(
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
    found = [needle for needle in needles if needle in text]
    if found:
        return Check(
            False,
            severity,
            category,
            label,
            "forbidden/current-blocker markers present: " + ", ".join(repr(item) for item in found),
            rel(path),
            acceptance,
        )
    return Check(True, PASS, category, label, "blocker markers absent", rel(path), acceptance)


def _function_body(text: str, name: str) -> str:
    lines = text.splitlines(keepends=True)
    start = None
    for idx, line in enumerate(lines):
        if line.startswith(f"def {name}[") or line.startswith(f"def {name}("):
            start = idx
            break
    if start is None:
        return ""
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        line = lines[idx]
        if line.startswith("def ") or line.startswith("struct "):
            end = idx
            break
    return "".join(lines[start:end])


def check_absent_in_functions(
    path: Path,
    *,
    function_names: Iterable[str],
    category: str,
    label: str,
    needles: Iterable[str],
    severity: str,
    acceptance: str,
) -> Check:
    text = read(path)
    if not text:
        return Check(False, severity, category, label, f"missing file: {path}", rel(path), acceptance)
    missing_functions: list[str] = []
    found: list[str] = []
    for function_name in function_names:
        body = _function_body(text, function_name)
        if not body:
            missing_functions.append(function_name)
            continue
        for needle in needles:
            if needle in body:
                found.append(f"{function_name}: {needle!r}")
    if missing_functions or found:
        details: list[str] = []
        if missing_functions:
            details.append("missing functions: " + ", ".join(missing_functions))
        if found:
            details.append("forbidden/current-blocker markers present: " + ", ".join(found))
        return Check(False, severity, category, label, "; ".join(details), rel(path), acceptance)
    return Check(True, PASS, category, label, "function bodies avoid blocker markers", rel(path), acceptance)


def check_regex_absent(
    path: Path,
    *,
    category: str,
    label: str,
    pattern: str,
    severity: str,
    acceptance: str,
) -> Check:
    text = read(path)
    if not text:
        return Check(False, severity, category, label, f"missing file: {path}", rel(path), acceptance)
    matches = re.findall(pattern, text, flags=re.MULTILINE | re.DOTALL)
    if matches:
        return Check(
            False,
            severity,
            category,
            label,
            f"matched blocker pattern {pattern!r} count={len(matches)}",
            rel(path),
            acceptance,
        )
    return Check(True, PASS, category, label, "blocker regex absent", rel(path), acceptance)


def check_swarmui_audit_doc() -> list[Check]:
    return [
        check_contains(
            AUDIT_DOC,
            category="docs",
            label="SwarmUI parity audit exists",
            needles=[
                "Image Model Generation Speed",
                "Video / Audio",
                "Current P0 Build Order",
            ],
            severity=P0,
            acceptance="Audit doc names image speed, broken video, and build order.",
        ),
        check_contains(
            LEDGER_DOC,
            category="docs",
            label="SwarmUI product-path ledger exists",
            needles=[
                "Acceptance Gates",
                "P0.1 Image Fast Path",
                "P0.2 Daemon Product Gate",
            ],
            severity=P0,
            acceptance="Ledger names concrete work packets and acceptance gates.",
        ),
        check_contains(
            SAMPLER_MAP_DOC,
            category="docs",
            label="SwarmUI sampler parity map exists",
            needles=[
                "`accepted_sampler_parity` must remain false",
                "Z-Image and Qwen now apply",
                "`/v1/samplers` support matrix",
            ],
            severity=P0,
            acceptance="Sampler map names current sampler/scheduler blockers and implemented variation behavior.",
        ),
        check_contains(
            WORKFLOW_MAP_DOC,
            category="docs",
            label="Comfy/Swarm workflow parity map exists",
            needles=[
                "not an arbitrary ComfyUI graph executor",
                "No graph IR",
                "Unsupported graph nodes",
            ],
            severity=P1,
            acceptance="Workflow map distinguishes constrained adapter support from arbitrary graph parity.",
        ),
        check_contains(
            MODEL_GALLERY_LORA_MAP_DOC,
            category="docs",
            label="model/gallery/LoRA parity map exists",
            needles=[
                "Latest Z-Image multi-LoRA product smoke",
                "rank_concat_scaled_b",
                "Gallery thumbnails",
                "Model search/filter/sort",
            ],
            severity=P1,
            acceptance="Model/gallery/LoRA map names browser, gallery, and current multi-LoRA evidence/limits.",
        ),
    ]


def check_specialized_surface_blockers() -> list[Check]:
    return [
        check_contains(
            SAMPLER_SURFACE_CHECK,
            category="sampler",
            label="sampler surface checker exists",
            needles=[
                "accepted_sampler_parity",
                "surface_blockers",
                "variation_seed",
                "images",
            ],
            severity=P0,
            acceptance="Sampler surface has a no-CUDA checker and JSON blocker report.",
        ),
        check_contains(
            DAEMON,
            category="sampler",
            label="sampler registry discovery endpoint",
            needles=[
                "swarmui_sampler_registry_json",
                'path == "/v1/samplers"',
                "default_sampler_for_backend",
                "default_scheduler_for_backend",
            ],
            severity=P0,
            acceptance="/v1/samplers exposes model-aware sampler/scheduler support instead of relying on docs-only inventory.",
        ),
        check_contains(
            SAMPLER_REGISTRY,
            category="sampler",
            label="SwarmUI/Comfy sampler registry exists",
            needles=[
                "serenity.samplers.v1",
                "sampler_admission_for_backend",
                "scheduler_admission_for_backend",
                "unsupported_policy",
                "fail_loud",
                "align_your_steps",
                "ltxv-image",
                "accepted_sampler_parity",
            ],
            severity=P0,
            acceptance="Daemon requests select from a SwarmUI/Comfy-compatible sampler/scheduler registry with fail-loud unsupported policy.",
        ),
        check_contains(
            REPO / "serenitymojo/serve/backend.mojo",
            category="sampler",
            label="typed sampler request fields reach backend contract",
            needles=[
                "var sampler: String",
                "var scheduler: String",
                "var variation_seed: Int",
                "var variation_strength: Float64",
                "var images: Int",
                "var image_index: Int",
                "var image_count: Int",
                "reject_unsupported_common_runtime_params",
            ],
            severity=P0,
            acceptance="Daemon requests validate or execute sampler/scheduler names per backend instead of storing metadata only.",
        ),
        check_contains(
            REPO / "serenitymojo/serve/ipc_codec.mojo",
            category="sampler",
            label="typed sampler request fields cross process isolation",
            needles=[
                'o.set("sampler"',
                'o.set("scheduler"',
                'o.set("variation_seed"',
                'o.set("variation_strength"',
                'o.set("images"',
                'o.set("image_index"',
                'o.set("image_count"',
                'p.sampler = obj["sampler"].as_string()',
                'p.scheduler = obj["scheduler"].as_string()',
                'p.image_index = Int(obj["image_index"].as_float())',
            ],
            severity=P0,
            acceptance="Process-isolated workers receive sampler/scheduler/variation/images as typed JobParams fields.",
        ),
        check_contains(
            ZIMAGE_BACKEND,
            category="sampler",
            label="Z-Image rejects unsupported sampler controls",
            needles=[
                "reject_unsupported_common_runtime_params",
                "sampler_admission_for_backend",
                "scheduler_admission_for_backend",
                "unsupported sampler",
                "unsupported scheduler",
            ],
            severity=P0,
            acceptance="Z-Image fails loud for unsupported sampler/scheduler/image/variation settings before model work.",
        ),
        check_contains(
            QWEN_BACKEND,
            category="sampler",
            label="Qwen rejects unsupported sampler controls",
            needles=[
                "reject_unsupported_common_runtime_params",
                "sampler_admission_for_backend",
                "scheduler_admission_for_backend",
                "unsupported sampler",
                "unsupported scheduler",
            ],
            severity=P0,
            acceptance="Qwen fails loud for unsupported sampler/scheduler/image/variation settings before model work.",
        ),
        check_contains(
            VARIATION_NOISE,
            category="sampler",
            label="Swarm variation noise helper exists",
            needles=[
                "swarm_variation_noise_chw",
                "acos",
                "sin",
                "SwarmKSampler-compatible slerp",
            ],
            severity=P0,
            acceptance="Variation noise has a pure-Mojo Swarm-style slerp helper instead of metadata-only handling.",
        ),
        check_contains(
            ZIMAGE_BACKEND,
            category="sampler",
            label="Z-Image applies variation noise",
            needles=[
                "swarm_variation_noise_chw",
                "self.params.variation_seed + self.params.image_index",
                '"variation_applied"',
            ],
            severity=P0,
            acceptance="Z-Image variation_seed/variation_strength affect initial latent noise and are recorded in the manifest.",
        ),
        check_contains(
            QWEN_BACKEND,
            category="sampler",
            label="Qwen applies variation noise",
            needles=[
                "swarm_variation_noise_chw",
                "self.params.variation_seed + self.params.image_index",
                "variation_strength > 0.0",
            ],
            severity=P0,
            acceptance="Qwen variation_seed/variation_strength affect initial latent noise instead of silently no-oping.",
        ),
        check_absent(
            SAMPLER_MAP_DOC,
            category="sampler",
            label="variation seed affects runtime noise",
            needles=["still not noise-path behavior"],
            severity=P0,
            acceptance="variation_seed and variation_strength affect the Mojo noise path or fail loud.",
        ),
        check_contains(
            DAEMON,
            category="sampler",
            label="images count emits multiple artifacts",
            needles=[
                "requested_images = p.images",
                "params_json_for_image_job",
                'o.set("job_ids"',
                'o.set("image_index"',
                'o.set("image_count"',
            ],
            severity=P0,
            acceptance="images=N creates N artifacts, metadata records, progress totals, and gallery entries.",
        ),
        check_contains(
            WORKFLOW_NODE_SURFACE_CHECK,
            category="workflow",
            label="workflow node surface checker exists",
            needles=[
                "constrained_workflow_adapter_ready",
                "arbitrary_comfy_swarm_graph_execution_ready",
                "unsupported graph",
            ],
            severity=P1,
            acceptance="Workflow node surface has a static checker for constrained support and blocked graph parity.",
        ),
        check_absent(
            WORKFLOW_MAP_DOC,
            category="workflow",
            label="arbitrary Comfy graph execution is implemented",
            needles=["No graph IR"],
            severity=P1,
            acceptance="A typed graph IR/topological executor exists for accepted Comfy/Swarm workflows.",
        ),
        check_contains(
            MODEL_GALLERY_LORA_SURFACE_CHECK,
            category="models-gallery-lora",
            label="model/gallery/LoRA surface checker exists",
            needles=[
                "claims_ux_parity",
                "claims_multi_lora_runtime_parity",
                "model_search_filter_sort",
                "multi_lora_runtime_parity",
            ],
            severity=P1,
            acceptance="Model/gallery/LoRA surface has a static checker and blocker report.",
        ),
        check_absent(
            MODEL_GALLERY_LORA_MAP_DOC,
            category="models-gallery-lora",
            label="multi-LoRA runtime parity is implemented",
            needles=["No accepted real multi-LoRA runtime path"],
            severity=P1,
            acceptance="At least one real backend proves a compatible multi-LoRA stack with artifact metadata.",
        ),
        check_absent(
            MODEL_GALLERY_LORA_MAP_DOC,
            category="models-gallery-lora",
            label="model and LoRA browser search/filter/sort exists",
            needles=["No model or LoRA browser search/filter/sort"],
            severity=P1,
            acceptance="Model and LoRA browsing supports search/filter/sort and compatibility metadata.",
        ),
        check_absent(
            MODEL_GALLERY_LORA_MAP_DOC,
            category="models-gallery-lora",
            label="gallery utility parity is implemented",
            needles=["No gallery thumbnails"],
            severity=P1,
            acceptance="Gallery supports thumbnails, favorite/star, delete/rename, sort/filter, and reuse workflows.",
        ),
    ]


def check_foundation(mojo_libs: Path) -> list[Check]:
    png = mojo_libs / "image/png.mojo"
    return [
        check_contains(
            png,
            category="foundation",
            label="MOJO-libs PNG tEXt metadata",
            needles=[
                "encode_png_bytes_with_text",
                "encode_png_with_text",
                "read_png_text_bytes",
                "read_png_text",
            ],
            severity=P0,
            acceptance="Generated images can carry and read serenity.genparams.v1 metadata.",
        ),
        check_contains(
            mojo_libs / "http/README.md",
            category="foundation",
            label="MOJO-libs HTTP/WebSocket stack",
            needles=["WebSocket", "streamed", "binary-safe"],
            severity=P0,
            acceptance="Native daemon can use Mojo HTTP/WebSocket primitives.",
        ),
        check_contains(
            mojo_libs / "sqlite/README.md",
            category="foundation",
            label="MOJO-libs SQLite-format index",
            needles=["read", "write", "sqlite3"],
            severity=P1,
            acceptance="Gallery/job index can be local and Mojo-native, with subset limits documented.",
        ),
    ]


def check_daemon_surface() -> list[Check]:
    return [
        check_contains(
            DAEMON,
            category="daemon",
            label="localhost generation API",
            needles=[
                "POST /v1/generate",
                "GET  /v1/jobs",
                "GET  /v1/job/<id>",
                "POST /v1/cancel/<id>",
                "WS   /v1/progress",
                "GET  /v1/health",
            ],
            severity=P0,
            acceptance="Daemon exposes the core single-user SwarmUI generation loop.",
        ),
        check_contains(
            DAEMON,
            category="daemon",
            label="canonical genparams preservation",
            needles=[
                "serenity.genparams.v1",
                "prompt_raw",
                "sampler",
                "scheduler",
                "variation_seed",
                "variation_strength",
                "images",
                "params_json",
            ],
            severity=P1,
            acceptance="Flat UI state survives through daemon params, PNG metadata, and DB rows.",
        ),
        check_contains(
            DAEMON,
            category="daemon",
            label="jobs DB gallery-index seam",
            needles=["DB_PATH", "jobs.db", "DbWriter", "save_jobs_db_safe"],
            severity=P1,
            acceptance="Finished jobs are indexed without relying on Python or external SQLite bindings.",
        ),
        check_contains(
            PROCESS_ISOLATED,
            category="daemon",
            label="process-isolated model workers",
            needles=["SIGKILL", "OS reclaims", "consecutive jobs REUSE the resident child"],
            severity=P0,
            acceptance="Model switching has a credible VRAM-reclaim path.",
        ),
        check_absent(
            DAEMON,
            category="daemon",
            label="daemon header is not stale skeleton prose",
            needles=["skeleton"],
            severity=P2,
            acceptance="Source comments match the current substantial daemon implementation.",
        ),
        check_contains(
            ZIMAGE_DAEMON_PRODUCT_CHECK,
            category="daemon",
            label="Z-Image daemon product smoke script",
            needles=[
                "serenity.zimage.daemon_product_smoke.v1",
                "/v1/generate",
                "/v1/progress",
                "jobs.db",
                "serenity.genparams.v1",
                "peak_vram_mib",
                "unsupported_sampler_smoke",
                "unipc_bh2_smoke",
                "multi_image_smoke",
                "variation_smoke",
                "executed_sampler",
            ],
            severity=P0,
            acceptance="Real Z-Image daemon generation has a repeatable artifact/timing/VRAM gate.",
        ),
    ]


def check_model_gallery_surface() -> list[Check]:
    return [
        check_contains(
            MODEL_SCAN,
            category="models",
            label="checkpoint and LoRA scanner",
            needles=[
                "scan_checkpoints",
                "scan_loras",
                "zimage",
                "qwen-image",
                "ltx2",
                "flux-2/klein",
                "sdxl",
            ],
            severity=P1,
            acceptance="Model browser can be backed by real disk scans and family tags.",
        ),
        check_contains(
            STUB_BACKEND,
            category="gallery",
            label="stub backend writes PNG metadata",
            needles=["encode_png_with_text", "serenity.genparams.v1"],
            severity=P1,
            acceptance="Metadata path is testable without CUDA.",
        ),
        check_contains(
            ZIMAGE_BACKEND,
            category="gallery",
            label="Z-Image backend writes PNG metadata",
            needles=["encode_png_with_text", "serenity.genparams.v1"],
            severity=P1,
            acceptance="Real image backend persists reusable params in PNG tEXt.",
        ),
        check_contains(
            QWEN_BACKEND,
            category="gallery",
            label="Qwen backend writes PNG metadata",
            needles=["encode_png_with_text", "serenity.genparams.v1"],
            severity=P1,
            acceptance="Second real image backend persists reusable params in PNG tEXt.",
        ),
        check_contains(
            DAEMON,
            category="gallery",
            label="gallery import/read-params endpoint",
            needles=["/v1/gallery", "read_png_text"],
            severity=P1,
            acceptance="SwarmUI-style gallery can import/read existing PNG params, not only list jobs.",
        ),
    ]


def check_image_fast_path() -> list[Check]:
    return [
        check_contains(
            ATTENTION_FLASH,
            category="image-fast-path",
            label="fast SDPA shim exists",
            needles=["cuDNN v9 Flash SDPA", "sdpa_flash_train_fwd", "sdpa_flash_backward"],
            severity=P0,
            acceptance="A fast attention primitive exists in-tree.",
        ),
        check_contains(
            ATTENTION,
            category="image-fast-path",
            label="current sdpa_nomask is identifiable",
            needles=["def sdpa_nomask", "return _sdpa_math", "def sdpa_nomask_slab"],
            severity=P0,
            acceptance="Checker can prove whether product paths still route to math-mode fallback.",
        ),
        check_absent(
            ZIMAGE_LORA_BLOCK,
            category="image-fast-path",
            label="Z-Image product forward no longer keeps math SDPA",
            needles=["the v3 forward keeps math sdpa"],
            severity=P0,
            acceptance="Z-Image generation forward is routed to the accepted fast path, not only graph backward.",
        ),
        check_contains(
            ZIMAGE_LORA_BLOCK,
            category="image-fast-path",
            label="Z-Image no-saved forward has flash dispatch",
            needles=[
                "def _zimage_sdpa_product_fwd",
                "sdpa_flash_train_fwd",
                "comptime ZIMAGE_SDPA_FLASH = True",
            ],
            severity=P0,
            acceptance="Z-Image product forward has a BF16 flash SDPA dispatch point.",
        ),
        check_contains(
            ZIMAGE_STACK_LORA,
            category="image-fast-path",
            label="Z-Image stack uses device no-saved refiners",
            needles=[
                "zimage_block_forward_device_moddev[H, Dh, N_IMG]",
                "zimage_refiner_forward_device[H, Dh, N_TXT]",
            ],
            severity=P0,
            acceptance="Z-Image generation stack routes frozen refiners through no-saved device forwards.",
        ),
        check_absent_in_functions(
            ZIMAGE_LORA_BLOCK,
            category="image-fast-path",
            label="Z-Image product forward avoids sdpa_nomask math fallback",
            function_names=[
                "zimage_block_lora_predict_device_tensor_moddev",
                "zimage_block_forward_device_moddev",
                "zimage_refiner_forward_device",
            ],
            needles=["var att = sdpa_nomask["],
            severity=P0,
            acceptance="Z-Image denoise does not call the old math-mode sampler attention in product forward.",
        ),
        check_absent(
            QWEN_DIT,
            category="image-fast-path",
            label="Qwen image product forward avoids math SDPA fallback",
            needles=["var attn = sdpa["],
            severity=P0,
            acceptance="Qwen image denoise routes masked joint attention to an accepted fast path.",
        ),
        check_absent(
            ZIMAGE_GENERATE,
            category="image-fast-path",
            label="Z-Image result manifest records real peak VRAM",
            needles=['"peak_vram_mib":0'],
            severity=P0,
            acceptance="Generation result manifests carry positive measured VRAM, not a placeholder.",
        ),
        check_contains(
            TRAINER_USE_STATUS,
            category="image-fast-path",
            label="status doc keeps speed claim honest",
            needles=[
                "Z-Image generation is still too slow for production speed parity",
                "two serial CFG main-stack passes",
            ],
            severity=P0,
            acceptance="Docs must not claim image generation speed parity until product path evidence exists.",
        ),
    ]


def check_video_path() -> list[Check]:
    return [
        check_regex_absent(
            LTX2_HQ,
            category="video",
            label="LTX2 HQ randn fallback is not hardcoded F32",
            pattern=r"s2n_[va]\s*=\s*randn\([^\n]*STDtype\.F32",
            severity=P0,
            acceptance="Video/audio random/noise generation returns BF16/F16/FP8 storage dtype unless a reference requires otherwise.",
        ),
        check_contains(
            DAEMON,
            category="video",
            label="daemon video generation endpoint",
            needles=["/v1/video", "mp4", "frame_count", "duration"],
            severity=P0,
            acceptance="SwarmUI-level video path reports real duration, frames, muxing, and audio behavior.",
        ),
        check_contains(
            DAEMON,
            category="video",
            label="bounded daemon video smoke is wired",
            needles=[
                "LTX2_VIDEO_SMOKE_RUNNER",
                "_ltx2_staged_smoke_video_result",
                "ltx2_t2v_av_stage2_dev_smoke.mp4",
                "accepted_video_parity",
                "total_wall_seconds",
                "build-video-smoke",
            ],
            severity=P0,
            acceptance="A bounded daemon video runner can emit MP4 artifacts with measured frame count, duration, muxing, audio behavior, and timings; full parity still requires runtime evidence.",
        ),
    ]


def check_prompt_queue_workflow() -> list[Check]:
    return [
        check_absent(
            DAEMON,
            category="prompt",
            label="prompt syntax is parsed in product path",
            needles=["the UI resolves prompt syntax at submit"],
            severity=P1,
            acceptance="Weighted prompts, LoRA tags, random syntax, and wildcards have a gated parser or a documented UI parser gate.",
        ),
        check_contains(
            DAEMON,
            category="queue",
            label="queue reorder/remove endpoints",
            needles=["/v1/reorder", "/v1/remove"],
            severity=P1,
            acceptance="Queued jobs can be reordered or removed before execution.",
        ),
        check_absent(
            DAEMON,
            category="workflow",
            label="workflow graph bodies are implemented",
            needles=["'workflow' (graph body) is reserved and not implemented"],
            severity=P2,
            acceptance="Workflow graph requests execute or fail with a documented feature gate outside SwarmUI parity claims.",
        ),
        check_contains(
            DAEMON,
            category="presets",
            label="presets and UI state endpoints",
            needles=["/v1/presets", "/v1/state"],
            severity=P1,
            acceptance="Named presets and last-state restoration are product APIs, not only local UI state.",
        ),
    ]


def check_docs_alignment() -> list[Check]:
    return [
        check_contains(
            TODO_DOC,
            category="docs",
            label="active task board tracks inference speed",
            needles=["inference speed sweep", "P7 zimage B2"],
            severity=P1,
            acceptance="The central task board keeps product generation speed visible.",
        ),
        check_contains(
            HANDOFF_DOC,
            category="docs",
            label="handoff states sampler fast-path work",
            needles=[
                "samplers still run math-mode SDPA",
                "switching sampler attention to flash",
            ],
            severity=P1,
            acceptance="Handoff explicitly distinguishes trainer flash work from sampler/product path work.",
        ),
        check_contains(
            SAMPLER_HARNESS_DOC,
            category="docs",
            label="sampler harness records timing/VRAM evidence contract",
            needles=["seconds/step", "peak VRAM", "speed parity is not accepted"],
            severity=P1,
            acceptance="Speed evidence remains measured and non-theatrical.",
        ),
    ]


def collect_checks(mojo_libs: Path) -> list[Check]:
    checks: list[Check] = []
    checks.extend(check_swarmui_audit_doc())
    checks.extend(check_foundation(mojo_libs))
    checks.extend(check_daemon_surface())
    checks.extend(check_model_gallery_surface())
    checks.extend(check_image_fast_path())
    checks.extend(check_video_path())
    checks.extend(check_prompt_queue_workflow())
    checks.extend(check_specialized_surface_blockers())
    checks.extend(check_docs_alignment())
    return checks


def build_report(checks: list[Check]) -> dict[str, object]:
    p0 = [check for check in checks if not check.ok and check.severity == P0]
    p1 = [check for check in checks if not check.ok and check.severity == P1]
    p2 = [check for check in checks if not check.ok and check.severity == P2]
    return {
        "schema": "serenity.swarmui.product_path_readiness.v1",
        "scope": "no-CUDA static product-path gate; no generation, no image comparison",
        "product_path_p0_ready": not p0,
        "swarmui_all_levels_ready": not p0 and not p1 and not p2,
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
        "strict_command": "python3 scripts/check_swarmui_product_path_contract.py --strict",
        "strict_all_command": "python3 scripts/check_swarmui_product_path_contract.py --strict-all",
    }


def print_text_report(report: dict[str, object]) -> None:
    checks = report["checks"]
    assert isinstance(checks, list)
    for item in checks:
        assert isinstance(item, dict)
        status = "PASS" if item["ok"] else item["severity"]
        print(
            "[swarmui-product] "
            f"{status} {item['category']}: {item['label']} "
            f"({item['path']}) - {item['detail']}"
        )
    summary = report["summary"]
    assert isinstance(summary, dict)
    print(
        "[swarmui-product] summary "
        f"checks={summary['checks']} passed={summary['passed']} "
        f"p0={summary['p0_blockers']} p1={summary['p1_blockers']} "
        f"p2={summary['p2_blockers']}"
    )
    if report["product_path_p0_ready"]:
        print("[swarmui-product] P0 product path: READY")
    else:
        print("[swarmui-product] P0 product path: BLOCKED")
    if report["swarmui_all_levels_ready"]:
        print("[swarmui-product] SwarmUI all-level parity: READY")
    else:
        print("[swarmui-product] SwarmUI all-level parity: BLOCKED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mojo-libs", type=Path, default=DEFAULT_MOJO_LIBS)
    parser.add_argument("--strict", action="store_true", help="exit 2 if P0 product-path blockers remain")
    parser.add_argument("--strict-all", action="store_true", help="exit 2 if any SwarmUI-level blocker remains")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON report")
    parser.add_argument("--write-readiness", type=Path, help="write machine-readable readiness JSON")
    args = parser.parse_args()

    report = build_report(collect_checks(args.mojo_libs))

    if args.write_readiness is not None:
        args.write_readiness.parent.mkdir(parents=True, exist_ok=True)
        args.write_readiness.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[swarmui-product] wrote readiness report: {args.write_readiness}")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text_report(report)

    if args.strict_all and not report["swarmui_all_levels_ready"]:
        return 2
    if args.strict and not report["product_path_p0_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
