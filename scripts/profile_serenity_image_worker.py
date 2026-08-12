#!/usr/bin/env python3
"""Capture one real Serenity image-worker denoise update with Nsight Systems."""

from __future__ import annotations

import argparse
import json
import os
import selectors
import socket
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
NSYS = Path("/opt/nvidia/nsight-compute/2024.1.1/host/target-linux-x64/nsys")
ROOTLESS_SCOPE_MARKER = "/app.slice/serenity-runtime-memory-"
ROOTLESS_MAX_BYTES = 24 * 1024 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--steps-sequence",
        help="comma-separated per-job step counts (for example 1,35); implies repeat length",
    )
    parser.add_argument("--no-nsys", action="store_true")
    parser.add_argument(
        "--assert-no-read-during-sampling",
        action="store_true",
        help=(
            "fail if /proc/<worker>/io read_bytes increases after sampling "
            "starts and before its final denoise progress event"
        ),
    )
    parser.add_argument(
        "--disk-read-warmup-jobs",
        type=int,
        default=0,
        help=(
            "number of leading jobs excluded from the read_bytes assertion; "
            "use only to warm lazy driver/library code before gated jobs"
        ),
    )
    parser.add_argument("--timeout", type=float, default=1800.0)
    return parser.parse_args()


def start_message(
    request: dict[str, object], out_dir: Path, steps: int, job_number: int = 1
) -> dict[str, object]:
    request = dict(request)
    request["steps"] = steps
    request.setdefault("checkpoint_path", "")
    request.setdefault("negative", "")
    request.setdefault("sigma_shift", 3.0)
    request.setdefault("variation_seed", 0)
    request.setdefault("variation_strength", 0.0)
    request.setdefault("images", 1)
    request.setdefault("image_index", 0)
    request.setdefault("image_count", 1)
    request.setdefault("workflow_save_prefix", "")
    request.setdefault("init_image", "")
    request.setdefault("mask_image", "")
    request.setdefault("creativity", 0.5)
    request["cmd"] = "start"
    request["job_id"] = f"job-{job_number:04d}"
    request["out_dir"] = str(out_dir.resolve())
    request["params_json"] = json.dumps(request, separators=(",", ":"))
    request["lora"] = []
    return request


def require_isolated_scope() -> None:
    cgroup_lines = Path("/proc/self/cgroup").read_text().splitlines()
    unified = next((line.split(":", 2)[2] for line in cgroup_lines if line.startswith("0::")), "")
    is_rootless_scope = ROOTLESS_SCOPE_MARKER in unified and unified.endswith(".service")
    if is_rootless_scope:
        cgroup_dir = Path("/sys/fs/cgroup") / unified.removeprefix("/")
        memory_max_raw = (cgroup_dir / "memory.max").read_text().strip()
        oom_group_raw = (cgroup_dir / "memory.oom.group").read_text().strip()
        memory_high_raw = (cgroup_dir / "memory.high").read_text().strip()
        try:
            memory_max = int(memory_max_raw)
        except ValueError as exc:
            raise SystemExit(
                f"refusing rootless runtime with non-finite memory.max={memory_max_raw}"
            ) from exc
        if memory_max > ROOTLESS_MAX_BYTES or oom_group_raw != "1":
            raise SystemExit(
                "refusing rootless runtime without the 24 GiB hard ceiling and "
                f"OOM group: memory.max={memory_max_raw}, memory.oom.group={oom_group_raw}"
            )
        if memory_high_raw not in {"max", memory_max_raw}:
            raise SystemExit(
                "refusing rootless runtime with a reclaim-pressure threshold below "
                f"the hard ceiling: memory.high={memory_high_raw}, memory.max={memory_max_raw}"
            )
        return
    raise SystemExit(
        "refusing to launch an image worker outside the passwordless rootless "
        f"runtime cgroup; current cgroup is {unified or '<unknown>'}. Run this "
        "command through scripts/mem_safe_runtime.sh."
    )


def process_io(pid: int) -> dict[str, int]:
    """Return the kernel I/O counters for one live worker process."""
    counters: dict[str, int] = {}
    for line in Path(f"/proc/{pid}/io").read_text().splitlines():
        name, raw_value = line.split(":", 1)
        counters[name] = int(raw_value.strip())
    return counters


def main() -> int:
    args = parse_args()
    require_isolated_scope()
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")
    steps_sequence = [args.steps] * args.repeat
    if args.steps_sequence:
        if args.repeat != 1:
            raise SystemExit("--steps-sequence cannot be combined with --repeat")
        try:
            steps_sequence = [int(raw) for raw in args.steps_sequence.split(",")]
        except ValueError as exc:
            raise SystemExit("--steps-sequence values must be integers") from exc
        if not steps_sequence or any(steps < 1 for steps in steps_sequence):
            raise SystemExit("--steps-sequence values must all be >= 1")
    repeat_count = len(steps_sequence)
    if args.disk_read_warmup_jobs < 0:
        raise SystemExit("--disk-read-warmup-jobs must be >= 0")
    if args.disk_read_warmup_jobs >= repeat_count:
        raise SystemExit(
            "--disk-read-warmup-jobs must leave at least one measured job"
        )
    if repeat_count > 1 and not args.no_nsys:
        raise SystemExit("--repeat > 1 is supported only with --no-nsys")
    if args.assert_no_read_during_sampling and not args.no_nsys:
        raise SystemExit(
            "--assert-no-read-during-sampling requires --no-nsys so the "
            "observed PID is the image worker"
        )
    worker = args.worker.resolve()
    request_path = args.request.resolve()
    report = args.report.resolve()
    out_dir = args.out_dir.resolve()
    if not worker.is_file():
        raise SystemExit(f"worker missing: {worker}")
    if not request_path.is_file():
        raise SystemExit(f"request missing: {request_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    loaded = json.loads(request_path.read_text())
    if isinstance(loaded, dict) and isinstance(loaded.get("request"), dict):
        loaded = loaded["request"]
    if not isinstance(loaded, dict):
        raise SystemExit(f"request JSON is not an object: {request_path}")
    base_request = loaded
    request = start_message(base_request, out_dir, steps_sequence[0])

    parent, child = socket.socketpair()
    child.set_inheritable(True)
    env = dict(os.environ)
    lib_paths = [
        REPO / "serenitymojo/ops/cshim/lib",
        REPO / ".pixi/envs/default/lib",
        REPO / ".pixi/envs/default/targets/x86_64-linux/lib/stubs",
    ]
    env["LD_LIBRARY_PATH"] = ":".join(map(str, lib_paths))
    if args.no_nsys:
        cmd = [str(worker), str(child.fileno())]
    else:
        cmd = [
            str(NSYS),
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--force-overwrite=true",
            "--wait=all",
            f"--output={report}",
            str(worker),
            str(child.fileno()),
            "profile-oneshot",
        ]
    proc = subprocess.Popen(
        cmd,
        cwd=REPO,
        env=env,
        pass_fds=(child.fileno(),),
    )
    child.close()
    parent.setblocking(False)
    selector = selectors.DefaultSelector()
    selector.register(parent, selectors.EVENT_READ)
    pending = b""
    terminal = False
    sent = False
    completed = 0
    sampling_io_start: dict[str, int] | None = None
    sampling_io_passed = False
    deadline = time.monotonic() + args.timeout
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None and not terminal:
                raise RuntimeError(f"profiled worker exited early with code {proc.returncode}")
            events = selector.select(timeout=0.1)
            for _key, _mask in events:
                chunk = parent.recv(65536)
                if not chunk:
                    if terminal:
                        break
                    raise RuntimeError("worker IPC closed before a terminal event")
                pending += chunk
                while b"\n" in pending:
                    raw, pending = pending.split(b"\n", 1)
                    event = json.loads(raw)
                    print(json.dumps(event, separators=(",", ":")), flush=True)
                    if (
                        args.assert_no_read_during_sampling
                        and completed >= args.disk_read_warmup_jobs
                        and event.get("ev") == "progress"
                        and event.get("phase") == "sampling"
                    ):
                        step = int(event.get("step", 0))
                        total = int(event.get("total", 0))
                        current_io = process_io(proc.pid)
                        if step == 0:
                            sampling_io_start = current_io
                            print(
                                json.dumps(
                                    {
                                        "gate": "denoise_disk_read",
                                        "job": completed + 1,
                                        "state": "armed",
                                        "read_bytes": current_io["read_bytes"],
                                        "rchar": current_io["rchar"],
                                    },
                                    separators=(",", ":"),
                                ),
                                flush=True,
                            )
                        elif sampling_io_start is None:
                            raise RuntimeError(
                                "denoise disk-read gate saw a sampling step before "
                                "the step-0 sampling boundary"
                            )
                        else:
                            read_delta = (
                                current_io["read_bytes"]
                                - sampling_io_start["read_bytes"]
                            )
                            if read_delta != 0:
                                raise RuntimeError(
                                    "denoise disk-read gate FAILED for job "
                                    f"{completed + 1} at step {step}/{total}: "
                                    f"read_bytes increased by {read_delta}"
                                )
                            if step == total:
                                sampling_io_passed = True
                                print(
                                    json.dumps(
                                        {
                                            "gate": "denoise_disk_read",
                                            "job": completed + 1,
                                            "state": "passed",
                                            "steps": total,
                                            "read_bytes_delta": read_delta,
                                            "rchar_delta": (
                                                current_io["rchar"]
                                                - sampling_io_start["rchar"]
                                            ),
                                        },
                                        separators=(",", ":"),
                                    ),
                                    flush=True,
                                )
                    if event.get("ev") == "ready" and not sent:
                        parent.sendall(
                            (json.dumps(request, separators=(",", ":")) + "\n").encode()
                        )
                        sent = True
                        deadline = time.monotonic() + args.timeout
                    if event.get("ev") in {"done", "failed", "cancelled"}:
                        if event.get("ev") != "done":
                            raise RuntimeError(
                                f"worker terminal event: {event.get('ev')}: "
                                f"{event.get('error', '')}"
                            )
                        if (
                            args.assert_no_read_during_sampling
                            and completed >= args.disk_read_warmup_jobs
                            and not sampling_io_passed
                        ):
                            raise RuntimeError(
                                "denoise disk-read gate did not observe a complete "
                                "step-0-to-final-step sampling interval"
                            )
                        if (
                            args.assert_no_read_during_sampling
                            and completed < args.disk_read_warmup_jobs
                        ):
                            print(
                                json.dumps(
                                    {
                                        "gate": "denoise_disk_read",
                                        "job": completed + 1,
                                        "state": "warmup_skipped",
                                    },
                                    separators=(",", ":"),
                                ),
                                flush=True,
                            )
                        completed += 1
                        sampling_io_start = None
                        sampling_io_passed = False
                        if completed >= repeat_count:
                            terminal = True
                        else:
                            request = start_message(
                                base_request,
                                out_dir,
                                steps_sequence[completed],
                                completed + 1,
                            )
                            parent.sendall(
                                (json.dumps(request, separators=(",", ":")) + "\n").encode()
                            )
                            deadline = time.monotonic() + args.timeout
            if terminal:
                break
        if not terminal:
            raise TimeoutError(f"worker did not finish within {args.timeout:.0f}s")
        if args.no_nsys:
            selector.unregister(parent)
            parent.shutdown(socket.SHUT_RDWR)
            parent.close()
        return_code = proc.wait(timeout=60)
        if return_code != 0:
            raise RuntimeError(f"nsys/worker exited with code {return_code}")
        return 0
    finally:
        selector.close()
        parent.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"profile failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
