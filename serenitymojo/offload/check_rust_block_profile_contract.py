#!/usr/bin/env python3
"""No-CUDA guard for Rust-owned model block/memory profiles.

The Rust server owns admission, preflight, and block/memory reporting. Mojo owns
the model runtime and the concrete offload/VMM primitives. This checker keeps
`serenity-server/crates/server/src/block_profiles.rs` aligned with local Mojo
offload plan builders and rejects any slide back toward outside-repo runtime
dependencies.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
BLOCK_PROFILES = REPO / "serenity-server/crates/server/src/block_profiles.rs"
PLAN = REPO / "serenitymojo/offload/plan.mojo"
VMM_MANAGER = REPO / "serenitymojo/offload/vmm_manager.mojo"
SOURCE_MAP = REPO / "serenitymojo/offload/STAGEHAND_TURBO_SOURCE_MAP_2026-06-16.md"

BANNED_RUNTIME_ROOTS = (
    "/home/alex/EriDiffusion",
    "/home/alex/SerenityFlow",
    "/home/alex/SwarmUI",
    "/home/alex/ComfyUI",
)


@dataclass(frozen=True)
class ProfileExpectation:
    name: str
    rust_const: str
    profile: str
    family: str
    plan_builder: str
    block_count: int
    kind_counts: dict[str, int]
    vmm: bool
    tensor_count_hint: int | None = None
    byte_count_hint_per_block: int | None = None
    byte_count_hint_total: int | None = None


EXPECTATIONS = (
    ProfileExpectation(
        "qwenimage",
        "QWEN_IMAGE",
        "qwen_image_transformer",
        "qwenimage",
        "build_qwenimage_block_plan",
        60,
        {"double_stream": 60},
        True,
        tensor_count_hint=1920,
        byte_count_hint_per_block=679_662_592,
        byte_count_hint_total=40_779_755_520,
    ),
    ProfileExpectation(
        "klein9b",
        "KLEIN9B_FLUX2",
        "klein9b_flux2_dit",
        "flux2",
        "build_klein9b_block_plan",
        32,
        {"double_stream": 8, "single_stream": 24},
        True,
    ),
    ProfileExpectation(
        "flux1",
        "FLUX1_DEV",
        "flux1_dev_dit",
        "flux",
        "build_flux1_dev_block_plan",
        57,
        {"double_stream": 19, "single_stream": 38},
        True,
    ),
    ProfileExpectation(
        "sd35",
        "SD35_LARGE",
        "sd35_large_mmdit",
        "sd3",
        "build_sd35_large_block_plan",
        38,
        {"joint_double_stream": 38},
        True,
    ),
    ProfileExpectation(
        "hidream",
        "HIDREAM_O1",
        "hidream_o1",
        "hidream",
        "build_hidream_o1_block_plan",
        36,
        {"transformer": 36},
        True,
    ),
    ProfileExpectation(
        "sensenova",
        "SENSENOVA_U1",
        "sensenova_u1",
        "sensenova",
        "build_sensenova_u1_block_plan",
        42,
        {"transformer": 42},
        True,
    ),
    ProfileExpectation(
        "lance",
        "LANCE_T2V",
        "lance_t2v",
        "lance",
        "build_lance_t2v_block_plan",
        36,
        {"transformer": 36},
        True,
    ),
    ProfileExpectation(
        "zimage",
        "ZIMAGE_NEXTDIT",
        "zimage_nextdit",
        "zimage",
        "NextDiTConfig.zimage",
        34,
        {"noise_refiner": 2, "context_refiner": 2, "main_layers": 30},
        False,
    ),
    ProfileExpectation(
        "ideogram4",
        "IDEOGRAM4_FP8",
        "ideogram4_fp8_resident",
        "ideogram4",
        "ideogram4_forward",
        34,
        {"transformer_layers": 34},
        False,
    ),
    ProfileExpectation(
        "sdxl",
        "SDXL_UNET",
        "sdxl_unet",
        "sdxl",
        "SDXLUNet",
        21,
        {"input_blocks": 9, "middle_blocks": 3, "output_blocks": 9},
        False,
    ),
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def fail(failures: list[str], msg: str) -> None:
    failures.append(msg)


def require(cond: bool, failures: list[str], msg: str) -> None:
    if not cond:
        fail(failures, msg)


def rust_const_body(text: str, const_name: str) -> str:
    match = re.search(
        rf"const\s+{re.escape(const_name)}:\s+BlockProfileSpec\s*=\s*BlockProfileSpec\s*\{{(?P<body>.*?)\n\}};",
        text,
        flags=re.DOTALL,
    )
    return match.group("body") if match else ""


def normalize_int_literal(value: int) -> str:
    return f"{value:,}".replace(",", "_")


def require_field(body: str, field: str, value: str, failures: list[str], name: str) -> None:
    require(f"{field}: {value}" in body, failures, f"{name}: {field} != {value}")


def check_rust_profile(text: str, exp: ProfileExpectation, failures: list[str]) -> None:
    body = rust_const_body(text, exp.rust_const)
    require(bool(body), failures, f"{exp.name}: missing Rust const {exp.rust_const}")
    if not body:
        return
    require_field(body, "profile", f'"{exp.profile}"', failures, exp.name)
    require_field(body, "family", f'"{exp.family}"', failures, exp.name)
    require(exp.plan_builder in body, failures, f"{exp.name}: source missing {exp.plan_builder}")
    require_field(body, "block_count", f"Some({exp.block_count})", failures, exp.name)
    require_field(body, "vmm_handle_available", str(exp.vmm).lower(), failures, exp.name)
    require_field(body, "turbo_hot_path", "false", failures, exp.name)
    for kind, count in exp.kind_counts.items():
        require(
            f'("{kind}", {count})' in body,
            failures,
            f"{exp.name}: missing block kind {kind}={count}",
        )
    if exp.tensor_count_hint is not None:
        require_field(body, "tensor_count_hint", f"Some({exp.tensor_count_hint})", failures, exp.name)
    if exp.byte_count_hint_per_block is not None:
        require_field(
            body,
            "byte_count_hint_per_block",
            f"Some({normalize_int_literal(exp.byte_count_hint_per_block)})",
            failures,
            exp.name,
        )
    if exp.byte_count_hint_total is not None:
        require_field(
            body,
            "byte_count_hint_total",
            f"Some({normalize_int_literal(exp.byte_count_hint_total)})",
            failures,
            exp.name,
        )


def check_mojo_plan(plan: str, failures: list[str]) -> None:
    plan_tokens = {
        "qwen": ["def build_qwenimage_block_plan()", "for i in range(60)", "BlockKind.double_stream()", "32", "679662592"],
        "klein": ["def build_klein9b_block_plan()", "return build_klein_block_plan(8, 24)"],
        "flux": ["def build_flux1_dev_block_plan()", "return build_flux_block_plan(19, 38)"],
        "sd35": ["def build_sd35_large_block_plan()", "return build_sd35_block_plan(38)"],
        "hidream": ["def build_hidream_o1_block_plan()", "for i in range(36)", "BlockKind.transformer()"],
        "sensenova": ["def build_sensenova_u1_block_plan()", "for i in range(42)", "BlockKind.transformer()"],
        "lance": ["def build_lance_t2v_block_plan()", "for i in range(36)", "BlockKind.transformer()"],
    }
    for name, tokens in plan_tokens.items():
        for token in tokens:
            require(token in plan, failures, f"plan.mojo missing {name} token: {token}")


def main() -> int:
    failures: list[str] = []
    rust = read(BLOCK_PROFILES)
    plan = read(PLAN)
    vmm = read(VMM_MANAGER)
    source_map = read(SOURCE_MAP)

    for token in (
        "control_plane_owner",
        "runtime_owner",
        "memory_block_policy_owner",
        "rust_preflight_plus_mojo_runtime",
        "runtime_dependency_on_external_repos",
        "serenitymojo/offload/vmm_manager.mojo",
        "unknown_profile",
    ):
        require(token in rust, failures, f"block_profiles.rs missing contract token: {token}")
    for banned in BANNED_RUNTIME_ROOTS:
        require(banned not in rust, failures, f"block_profiles.rs contains banned runtime root: {banned}")

    for exp in EXPECTATIONS:
        check_rust_profile(rust, exp, failures)

    check_mojo_plan(plan, failures)
    for token in (
        "struct VmmModelHandle",
        "struct VmmModelManager",
        "ensure_block_resident",
        "evict_block",
        "resident_bytes",
    ):
        require(token in vmm, failures, f"vmm_manager.mojo missing token: {token}")
    for token in (
        "Runtime code",
        "should not call or import outside repos",
        "VmmModelHandle",
        "Rust Turbo Concepts",
    ):
        require(token in source_map, failures, f"source map missing token: {token}")

    report = {
        "schema": "serenity.rust_block_profiles.v1",
        "contract_source": "serenitymojo/offload/check_rust_block_profile_contract.py",
        "block_profiles": str(BLOCK_PROFILES.relative_to(REPO)),
        "mojo_plan": str(PLAN.relative_to(REPO)),
        "vmm_manager": str(VMM_MANAGER.relative_to(REPO)),
        "profiles_checked": [exp.name for exp in EXPECTATIONS],
        "failures": failures,
        "ok": not failures,
    }

    if failures:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1
    print("Rust block profile contract: PASS")
    print(f"  profiles checked: {len(EXPECTATIONS)}")
    print("  schema: serenity.rust_block_profiles.v1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
