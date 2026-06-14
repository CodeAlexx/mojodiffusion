#!/usr/bin/env python3
"""Static guard for the allowed daemon CLI-backed backend surface.

This checker is intentionally lightweight: it reads source text only. It does
not build Mojo, start the daemon, run models, invoke CUDA, or spawn
subprocesses. Its job is to fail when the planned SDXL/Anima SampleCliBackend
daemon plumbing is missing or regresses.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class MarkerGroup:
    path: str
    description: str
    markers: tuple[str, ...]


MARKER_GROUPS: tuple[MarkerGroup, ...] = (
    MarkerGroup(
        "serenitymojo/serve/sample_cli_backend.mojo",
        "Sample CLI backend exists and exposes SDXL/Anima CLI/caption markers",
        (
            "SampleCliBackend",
            "SAMPLE_CLI_SDXL_BIN",
            "SAMPLE_CLI_ANIMA_BIN",
            "sample_caps_pos",
            "sample_caps_neg",
            "sdxl",
            "anima",
        ),
    ),
    MarkerGroup(
        "serenitymojo/serve/dispatch_backend.mojo",
        "DispatchBackend imports and routes SDXL/Anima SampleCliBackend kinds",
        (
            "SampleCliBackend",
            "KIND_SDXL",
            "KIND_ANIMA",
            "sdxl",
            "anima",
        ),
    ),
    MarkerGroup(
        "serenitymojo/serve/worker.mojo",
        'Worker imports SampleCliBackend and handles kind == "sdxl"/"anima"',
        (
            "SampleCliBackend",
            'kind == "sdxl"',
            'kind == "anima"',
            "sdxl",
            "anima",
        ),
    ),
    MarkerGroup(
        "serenitymojo/serve/ipc_codec.mojo",
        "Worker IPC encodes/decodes sample caption fields",
        (
            "sample_caps_pos",
            "sample_caps_neg",
        ),
    ),
    MarkerGroup(
        "serenitymojo/serve/backend.mojo",
        "JobParams carries sample caption fields",
        (
            "sample_caps_pos",
            "sample_caps_neg",
        ),
    ),
    MarkerGroup(
        "serenitymojo/serve/serenity_daemon.mojo",
        "Daemon parses and persists sample caption fields",
        (
            "sample_caps_pos",
            "sample_caps_neg",
        ),
    ),
    MarkerGroup(
        "serenitymojo/serve/workflow_graph.mojo",
        "Workflow graph forwards sample caption fields through flat params/genparams",
        (
            "sample_caps_pos",
            "sample_caps_neg",
            "flat",
            "genparams",
        ),
    ),
    MarkerGroup(
        "serenitymojo/sampling/sampler_registry.mojo",
        "Sampler registry recognizes SDXL/Anima backend/model routing",
        (
            "sdxl",
            "anima",
        ),
    ),
    MarkerGroup(
        "pixi.toml",
        "Pixi exposes CLI backend build tasks",
        (
            "build-sdxl-cli",
            "build-anima-cli",
        ),
    ),
)


BANNED_SELF_MARKERS: tuple[str, ...] = (
    "import " + "subprocess",
    "from " + "subprocess",
    "sub" + "process.",
    "os." + "system(",
    "os." + "popen(",
    "pty." + "spawn(",
    "nvidia" + "-smi",
)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def read_text(path: Path) -> tuple[str, str | None]:
    try:
        return path.read_text(encoding="utf-8"), None
    except OSError as exc:
        return "", str(exc)


def check_group(group: MarkerGroup) -> list[str]:
    path = REPO / group.path
    if not path.is_file():
        return [f"{group.path}: file is missing"]
    text, error = read_text(path)
    if error is not None:
        return [f"{group.path}: cannot read file: {error}"]
    missing = [marker for marker in group.markers if marker not in text]
    return [f"{group.path}: missing marker {marker!r}" for marker in missing]


def check_self_static_only() -> list[str]:
    """Fail if this guard grows runtime/process/CUDA behavior."""
    text, error = read_text(Path(__file__).resolve())
    if error is not None:
        return [f"{rel(Path(__file__).resolve())}: cannot read self for runtime guard: {error}"]

    findings: list[str] = []
    for marker in BANNED_SELF_MARKERS:
        if marker in text:
            findings.append(
                f"{rel(Path(__file__).resolve())}: banned runtime marker present in guard source: {marker!r}"
            )
    return findings


def main() -> int:
    print("allowed-daemon-cli-backends: static source-text guard")
    print("runtime: PASS static/no CUDA/no model execution/no subprocess invocation")

    failures: list[str] = []
    self_failures = check_self_static_only()
    if self_failures:
        failures.extend(self_failures)
        print("FAIL runtime guard: script contains banned runtime/process/CUDA markers")
    else:
        print("PASS runtime guard: script reads files only")

    for group in MARKER_GROUPS:
        group_failures = check_group(group)
        if group_failures:
            failures.extend(group_failures)
            print(f"FAIL {group.path}: {group.description}")
            for failure in group_failures:
                print(f"  - {failure}")
        else:
            print(f"PASS {group.path}: {group.description}")

    if failures:
        print()
        print(f"FAIL allowed-daemon-cli-backends: {len(failures)} missing/banned marker(s)")
        return 1

    print()
    print("PASS allowed-daemon-cli-backends: SDXL/Anima CLI-backed daemon surface is present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
