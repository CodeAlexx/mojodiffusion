#!/usr/bin/env python3
"""Run the LTX2 parity and readiness gates as an explicit matrix.

This is dev/oracle tooling only. The LTX2 runtime stays Mojo-only; this script
orchestrates existing Mojo gates and, when requested, Python oracle generators.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CUDNN_LIB = Path(
    "/home/alex/musubi-tuner-ref/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib"
)

STAGED_PHASE_REQUIRED: tuple[str, ...] = tuple(
    f"output/ltx2_staged_phase/{name}.bin"
    for name in (
        "stage1_video_x",
        "stage1_video_vel",
        "stage1_video_final_ref",
        "stage1_audio_x",
        "stage1_audio_vel",
        "stage1_audio_final_ref",
        "stage2_video_upscaled_ref",
        "stage2_video_init_noise",
        "stage2_audio_init_noise",
        "stage2_video_x_ref",
        "stage2_audio_x_ref",
        "stage2_video_vel1",
        "stage2_video_vel2",
        "stage2_audio_vel1",
        "stage2_audio_vel2",
        "stage2_video_noise_sub",
        "stage2_video_noise_step",
        "stage2_audio_noise_sub",
        "stage2_audio_noise_step",
        "stage2_video_next_ref",
        "stage2_audio_next_ref",
    )
)

CREATOR_PHASE_REQUIRED: tuple[str, ...] = (
    "output/ltx2_creator_phase_dumps/creator_960x512_121f_seed42/creator_phase_tensors.safetensors",
    "output/ltx2_creator_phase_dumps/creator_960x512_121f_seed42/manifest.json",
)


@dataclasses.dataclass(frozen=True)
class Gate:
    gate_id: str
    area: str
    tier: str
    kind: str
    commands: tuple[tuple[str, ...], ...]
    required_paths: tuple[str, ...] = ()
    oracle_commands: tuple[tuple[str, ...], ...] = ()
    note: str = ""


def p(path: str) -> str:
    return str(REPO / path)


GATES: tuple[Gate, ...] = (
    Gate(
        "dtype_rng_contract",
        "inference/runtime",
        "fast",
        "static-guard",
        (("python3", "scripts/check_ltx2_dtype_contract.py", "--scope", "sidecar"),),
        note=(
            "Pipeline/sampling BF16 boundary guard. Mojo RNG is accepted only "
            "when explicitly marked not-PyTorch-parity, oracle-noise, or proven."
        ),
    ),
    Gate(
        "attention_tiled_cross",
        "ops/attention",
        "fast",
        "math-and-memory",
        (("pixi", "run", "mojo", "run", "-I", ".", "serenitymojo/ops/sdpa_tiled_probe.mojo"),),
        note="Online-softmax SDPA, including rectangular cross-attention vs padded-mask oracle.",
    ),
    Gate(
        "sampler_schedule_guidance",
        "inference/sampler",
        "fast",
        "self-parity",
        (("pixi", "run", "mojo", "run", "-I", ".", "serenitymojo/pipeline/ltx2_sampler_smoke.mojo"),),
        note="Distilled schedules, CFG-star, STG masks/rescale.",
    ),
    Gate(
        "sampler_res2s_core",
        "inference/sampler",
        "fast",
        "host-parity",
        (("pixi", "run", "mojo", "run", "-I", ".", "serenitymojo/pipeline/ltx2_res2s_smoke.mojo"),),
        note="Deterministic res_2s coefficients/substep/combine against host F64.",
    ),
    Gate(
        "sampler_res2s_hq_sde_bong",
        "inference/sampler",
        "fast",
        "host-parity",
        (("pixi", "run", "mojo", "run", "-I", ".", "serenitymojo/pipeline/ltx2_res2s_hq_step_smoke.mojo"),),
        note="Full interior HQ step: substep SDE, bongmath, step SDE.",
    ),
    Gate(
        "nag_combine",
        "inference/guidance",
        "fast",
        "python-oracle-parity",
        (("pixi", "run", "mojo", "run", "-I", ".", "serenitymojo/models/dit/parity/ltx2_nag_parity.mojo"),),
        required_paths=("serenitymojo/models/dit/parity/ltx2_nag_ref.txt",),
        oracle_commands=(("python3", "serenitymojo/models/dit/parity/ltx2_nag_oracle.py"),),
        note="NAG combine, including L1 clip-active case.",
    ),
    Gate(
        "audiosync_profile_contract",
        "inference/workflow",
        "fast",
        "mojo-contract",
        (("pixi", "run", "mojo", "run", "-I", ".", "serenitymojo/pipeline/ltx2_audiosync_profile_smoke.mojo"),),
        note="Mojo shape/scheduler contract for 97-frame 24fps AudioSync with 6.5s audio and staged token pressure.",
    ),
    Gate(
        "audiosync_runner_profile",
        "inference/workflow",
        "fast",
        "mojo-contract",
        (("pixi", "run", "mojo", "run", "-I", ".", "serenitymojo/pipeline/ltx2_runner_profile_smoke.mojo"),),
        note="Renderer-side AudioSync profile: non-square geometry and LoRA strengths used by the Mojo binary.",
    ),
    Gate(
        "staged_hq_phase_handoff",
        "inference/workflow",
        "full",
        "python-oracle-parity",
        (
            (
                "pixi",
                "run",
                "mojo",
                "run",
                "-I",
                ".",
                "serenitymojo/pipeline/ltx2_staged_phase_parity_smoke.mojo",
            ),
        ),
        required_paths=STAGED_PHASE_REQUIRED,
        oracle_commands=(("python3", "scripts/ltx2_staged_phase_parity_ref.py"),),
        note="Stage-1 final denoise, upsampler handoff, stage-2 noiser using PyTorch oracle noise tensors, and first video/audio HQ res2s step.",
    ),
    Gate(
        "creator_fast_phase_handoff",
        "inference/workflow",
        "full",
        "creator-oracle-parity",
        (
            (
                "pixi",
                "run",
                "mojo",
                "run",
                "-I",
                ".",
                "serenitymojo/pipeline/ltx2_creator_phase_parity_smoke.mojo",
            ),
        ),
        required_paths=CREATOR_PHASE_REQUIRED,
        oracle_commands=(("python3", "scripts/ltx2_creator_phase_dump.py"),),
        note=(
            "Real Desktop creator fast-distilled schedules, PyTorch BF16 noiser "
            "handoff, first/last transformer raw-velocity capture, and first/last "
            "Euler updates against captured creator tensors."
        ),
    ),
    Gate(
        "lora_distilled_surface",
        "inference/lora",
        "full",
        "coverage-and-math",
        (("pixi", "run", "mojo", "run", "-I", ".", "serenitymojo/pipeline/ltx2_lora_smoke.mojo"),),
        note="Distilled rank-384 LoRA mapping, fail-closed apply, add math.",
    ),
    Gate(
        "lora_hq_stack_surface",
        "inference/lora",
        "full",
        "coverage-and-apply",
        (
            (
                "pixi",
                "run",
                "mojo",
                "run",
                "-I",
                ".",
                "-Xlinker",
                "-lm",
                "-Xlinker",
                "-lcuda",
                "serenitymojo/pipeline/ltx2_hq_lora_stack_smoke.mojo",
            ),
        ),
        note="Distilled + camera-static + detailer + local Musubi LoRA block apply.",
    ),
    Gate(
        "lora_factorized_surface",
        "inference/lora",
        "full",
        "coverage-and-runtime-surface",
        (
            (
                "pixi",
                "run",
                "mojo",
                "run",
                "-I",
                ".",
                "serenitymojo/pipeline/ltx2_factorized_lora_smoke.mojo",
            ),
        ),
        note="HQ LoRA stack attaches as low-rank A/B factors, avoiding full-rank deltas.",
    ),
    Gate(
        "lora_factorized_math",
        "inference/lora",
        "fast",
        "math-parity",
        (
            (
                "pixi",
                "run",
                "mojo",
                "run",
                "-I",
                ".",
                "serenitymojo/pipeline/ltx2_factorized_lora_math_parity.mojo",
            ),
        ),
        note="Synthetic factorized LoRA equals materialized W + scale*(B@A), with and without bias.",
    ),
    Gate(
        "av_block0",
        "inference/dit",
        "full",
        "python-oracle-parity",
        (
            (
                "pixi",
                "run",
                "mojo",
                "run",
                "-I",
                ".",
                "-Xlinker",
                "-lm",
                "-Xlinker",
                "-lcuda",
                "serenitymojo/pipeline/ltx2_av_block_parity_smoke.mojo",
            ),
        ),
        required_paths=("output/ltx2_av_block0/av_block0_ref.safetensors",),
        oracle_commands=(("python3", "scripts/ltx2_av_block0_parity.py"),),
        note="Block-0 joint video/audio transformer forward vs reference dump.",
    ),
    Gate(
        "dit_forward_48_block",
        "inference/dit",
        "expensive",
        "python-oracle-parity",
        (
            (
                "pixi",
                "run",
                "mojo",
                "run",
                "-I",
                ".",
                "-Xlinker",
                "-lm",
                "-Xlinker",
                "-lcuda",
                "serenitymojo/pipeline/ltx2_dit_forward_smoke.mojo",
            ),
        ),
        required_paths=("output/ltx2_dit_forward/dit_forward_ref.safetensors",),
        oracle_commands=(("python3", "scripts/ltx2_dit_forward_parity_ref.py"),),
        note="Full 48-block AV DiT velocity gate. VRAM/time heavy.",
    ),
    Gate(
        "latent_upsampler",
        "inference/upsampler",
        "full",
        "python-oracle-parity",
        (
            (
                "pixi",
                "run",
                "mojo",
                "run",
                "-I",
                ".",
                "-Xlinker",
                "-lm",
                "serenitymojo/models/upsampler/ltx2_upsampler_smoke.mojo",
            ),
        ),
        required_paths=(
            "serenitymojo/models/upsampler/parity/spatial_in.bin",
            "serenitymojo/models/upsampler/parity/spatial_out.bin",
            "serenitymojo/models/upsampler/parity/temporal_in.bin",
            "serenitymojo/models/upsampler/parity/temporal_out.bin",
        ),
        oracle_commands=(("python3", "serenitymojo/models/upsampler/parity/ref_dump.py"),),
        note="LTX spatial-x2 and temporal-x2 latent upsamplers vs Lightricks/PyTorch reference.",
    ),
    Gate(
        "video_vae_decode",
        "inference/video-vae",
        "full",
        "python-oracle-parity",
        (
            (
                "pixi",
                "run",
                "mojo",
                "run",
                "-I",
                ".",
                "-Xlinker",
                "-lm",
                "serenitymojo/pipeline/ltx2_video_vae_decode_smoke.mojo",
            ),
        ),
        required_paths=("output/ltx2_video_vae/video_vae_ref.safetensors",),
        oracle_commands=(("python3", "scripts/ltx2_video_vae_decode_ref.py"),),
        note="Full video VAE decode against fixed latent oracle and PNG artifact.",
    ),
    Gate(
        "audio_vae_decode",
        "inference/audio-vae",
        "full",
        "python-oracle-parity",
        (
            (
                "pixi",
                "run",
                "mojo",
                "run",
                "-I",
                ".",
                "-Xlinker",
                "-lm",
                "serenitymojo/pipeline/ltx2_audio_vae_smoke.mojo",
            ),
        ),
        required_paths=("output/ltx2_audio_vae/audio_vae_ref.safetensors",),
        oracle_commands=(("python3", "scripts/ltx2_audio_vae_ref.py"),),
        note="Full audio VAE decode against fixed latent oracle.",
    ),
    Gate(
        "vocoder_bwe",
        "inference/audio",
        "full",
        "python-oracle-parity",
        (
            (
                "pixi",
                "run",
                "mojo",
                "run",
                "-I",
                ".",
                "-Xlinker",
                "-lm",
                "serenitymojo/pipeline/ltx2_vocoder_smoke.mojo",
            ),
        ),
        required_paths=("output/ltx2_vocoder/vocoder_ref.safetensors",),
        oracle_commands=(("python3", "scripts/ltx2_vocoder_ref.py"),),
        note="BigVGAN+BWE waveform parity and WAV artifact.",
    ),
    Gate(
        "stream_ceiling",
        "inference/offload",
        "full",
        "contract",
        (("pixi", "run", "mojo", "run", "-I", ".", "serenitymojo/pipeline/ltx2_stream_ceiling_smoke.mojo"),),
        note="Single-resident stream/offload ceiling contract.",
    ),
    Gate(
        "render_staged_hq_smoke",
        "inference/render",
        "manual-smoke",
        "artifact-smoke",
        (
            (
                "pixi",
                "run",
                "mojo",
                "build",
                "-I",
                ".",
                "-Xlinker",
                "-lm",
                "serenitymojo/pipeline/ltx2_t2v_av_hq.mojo",
                "-o",
                "/tmp/ltx2_hq_staged_gate",
            ),
            (
                "/tmp/ltx2_hq_staged_gate",
                "staged",
                "lora",
                "stream",
                "noaudio",
                "output/ltx2_hq_staged_gate",
                "1",
            ),
        ),
        note="Manual video-only smoke. Not a production acceptance gate.",
    ),
    Gate(
        "render_video_nag_smoke",
        "inference/render",
        "manual-smoke",
        "artifact-smoke",
        (
            (
                "pixi",
                "run",
                "mojo",
                "build",
                "-I",
                ".",
                "-Xlinker",
                "-lm",
                "serenitymojo/pipeline/ltx2_t2v_av_hq.mojo",
                "-o",
                "/tmp/ltx2_hq_nag_gate",
            ),
            (
                "/tmp/ltx2_hq_nag_gate",
                "single",
                "lora",
                "stream",
                "noaudio",
                "nag",
                "output/ltx2_hq_nag_gate",
                "1",
            ),
        ),
        note="Manual video-only NAG smoke. Not a production acceptance gate.",
    ),
    Gate(
        "trainer_readiness_fail_closed",
        "training",
        "manual",
        "contract",
        (
            (
                "pixi",
                "run",
                "mojo",
                "run",
                "-I",
                ".",
                "serenitymojo/training/ltx2_av_training_readiness.mojo",
                "--expect-not-ready",
            ),
        ),
        note="Production trainer must remain fail-closed until AV backward/surface are real.",
    ),
    Gate(
        "trainer_foundation_acceptance",
        "training",
        "manual",
        "contract",
        (
            (
                "pixi",
                "run",
                "mojo",
                "build",
                "-I",
                ".",
                "serenitymojo/training/train_ltx2_av.mojo",
                "-o",
                "/tmp/train_ltx2_av_contract",
            ),
            ("/tmp/train_ltx2_av_contract", "--acceptance"),
        ),
        note="Modular AV trainer foundation and acceptance contract.",
    ),
    Gate(
        "trainer_masked_av_loss",
        "training",
        "manual",
        "contract",
        (
            (
                "pixi",
                "run",
                "mojo",
                "run",
                "-I",
                ".",
                "serenitymojo/training/parity/ltx2_masked_av_loss_parity.mojo",
            ),
        ),
        note="Musubi masked video/audio loss parity: broadcast masks, all-false fallback, MSE/MAE/Huber.",
    ),
    Gate(
        "trainer_audio_ref_ic",
        "training",
        "manual",
        "contract",
        (
            (
                "pixi",
                "run",
                "mojo",
                "run",
                "-I",
                ".",
                "serenitymojo/training/parity/ltx2_audio_ref_ic_contract.mojo",
            ),
        ),
        note="Musubi audio_ref_only_ic contract: ref concat, zero timesteps/targets, loss mask, positions, attention masks.",
    ),
    Gate(
        "trainer_parity_audit",
        "training",
        "manual",
        "contract",
        (
            (
                "pixi",
                "run",
                "mojo",
                "run",
                "-I",
                ".",
                "serenitymojo/training/parity/ltx2_trainer_parity_audit.mojo",
            ),
        ),
        note="Musubi/LTX2 trainer foundation audit across config/cache/schedule/loss/LoRA/checkpoint/validation.",
    ),
    Gate(
        "trainer_real_build",
        "training",
        "manual",
        "build",
        (
            (
                "pixi",
                "run",
                "mojo",
                "build",
                "-I",
                ".",
                "-Xlinker",
                "-lm",
                "-Xlinker",
                "-lcuda",
                "serenitymojo/training/train_ltx2_real.mojo",
                "-o",
                "/tmp/train_ltx2_real",
            ),
        ),
        note="Legacy/real trainer entrypoint still builds after config surface changes.",
    ),
)


KNOWN_GAPS: tuple[str, ...] = (
    "No full production artifact acceptance gate yet: prompt/negative context, NAG wiring, multi-LoRA 48-block forward, VAE/vocoder, mux, duration, frames, audio, VRAM, and runtime in one long-video run.",
    "No full numeric multi-LoRA 48-block forward parity yet; current gates cover mapping/apply, factor attachment, and synthetic factorized math.",
    "Audio-reference IC conditioning is contract-gated; image/control IC conditioning and full runtime integration are not done.",
    "The 97-frame AudioSync profile and single-stage runner shape are gated, but the full render oracle and automated visual/audio-semantic acceptance are not done.",
    "AV trainer production parity is fail-closed: full AV backward/update parity is not implemented.",
)


def selected_tiers(mode: str) -> set[str]:
    if mode == "fast":
        return {"fast"}
    if mode == "full":
        return {"fast", "full"}
    if mode == "expensive":
        return {"fast", "full", "expensive"}
    if mode == "all":
        return {"fast", "full", "expensive"}
    raise ValueError(mode)


def parse_csv(value: str | None) -> set[str]:
    if not value:
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def area_matches(gate: Gate, filters: set[str]) -> bool:
    if not filters:
        return True
    gate_root = gate.area.split("/", 1)[0]
    for value in filters:
        if gate.area == value or gate_root == value or gate.area.startswith(value + "/"):
            return True
    return False


def resolve_gates(
    mode: str, only: set[str], skip: set[str], area_filters: set[str]
) -> list[Gate]:
    known_ids = {gate.gate_id for gate in GATES}
    unknown = (only | skip) - known_ids
    if unknown:
        joined = ", ".join(sorted(unknown))
        raise SystemExit(f"unknown gate id(s): {joined}")
    tiers = selected_tiers(mode)
    out: list[Gate] = []
    for gate in GATES:
        if only and gate.gate_id not in only:
            continue
        if gate.gate_id in skip:
            continue
        if not area_matches(gate, area_filters):
            continue
        if not only and gate.tier not in tiers:
            continue
        out.append(gate)
    return out


def missing_required(gate: Gate) -> list[Path]:
    missing: list[Path] = []
    for rel in gate.required_paths:
        candidate = REPO / rel
        if not candidate.exists():
            missing.append(candidate)
    return missing


def fmt_cmd(cmd: tuple[str, ...]) -> str:
    return " ".join(cmd)


def run_cmd(cmd: tuple[str, ...], dry_run: bool, timeout: int | None) -> int:
    print(f"$ {fmt_cmd(cmd)}", flush=True)
    if dry_run:
        return 0
    start = time.monotonic()
    env = os.environ.copy()
    if CUDNN_LIB.exists():
        current_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            str(CUDNN_LIB) if not current_ld else str(CUDNN_LIB) + ":" + current_ld
        )
    try:
        proc = subprocess.run(cmd, cwd=REPO, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after {timeout}s: {fmt_cmd(cmd)}", flush=True)
        return 124
    elapsed = time.monotonic() - start
    print(f"[elapsed] {elapsed:.1f}s", flush=True)
    return proc.returncode


def print_list() -> None:
    print("LTX2 parity gates:")
    width = max(len(g.gate_id) for g in GATES)
    for gate in GATES:
        req = " req-oracle" if gate.required_paths else ""
        print(
            f"  {gate.gate_id:<{width}}  {gate.tier:<9} {gate.kind:<22} "
            f"{gate.area}{req}"
        )
        if gate.note:
            print(f"    {gate.note}")
    print("")
    print("Known missing gates:")
    for gap in KNOWN_GAPS:
        print(f"  - {gap}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("fast", "full", "expensive", "all"),
        default="fast",
        help="Gate tier to run. full includes fast; expensive/all include all tiers.",
    )
    parser.add_argument("--only", help="Comma-separated gate ids to run.")
    parser.add_argument("--skip", help="Comma-separated gate ids to skip.")
    parser.add_argument(
        "--area",
        help="Comma-separated area filters, e.g. inference, training, inference/lora.",
    )
    parser.add_argument(
        "--refresh-oracles",
        action="store_true",
        help="Run Python oracle generators before gates with required dumps.",
    )
    parser.add_argument("--list", action="store_true", help="List gates and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure.")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip gates with missing required oracle dumps instead of failing.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Per-command timeout in seconds; 0 disables timeout.",
    )
    parser.add_argument(
        "--hide-gaps",
        action="store_true",
        help="Do not print known missing gates at the end.",
    )
    args = parser.parse_args()

    if args.list:
        print_list()
        return 0

    gates = resolve_gates(
        args.mode, parse_csv(args.only), parse_csv(args.skip), parse_csv(args.area)
    )
    timeout = args.timeout if args.timeout > 0 else None
    if not gates:
        print("No gates selected.")
        return 2

    print(f"LTX2 parity sweep: mode={args.mode} gates={len(gates)} repo={REPO}")
    print("Python use here is dev/oracle orchestration only; runtime gates are Mojo.")
    print("")

    failures: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []

    for index, gate in enumerate(gates, start=1):
        print("=" * 88)
        print(
            f"[{index}/{len(gates)}] {gate.gate_id} "
            f"({gate.area}, {gate.tier}, {gate.kind})"
        )
        if gate.note:
            print(gate.note)

        missing = missing_required(gate)
        if missing and args.refresh_oracles:
            print("Refreshing oracle dump(s):")
            for cmd in gate.oracle_commands:
                rc = run_cmd(cmd, args.dry_run, timeout)
                if rc != 0:
                    failures.append((gate.gate_id, f"oracle command failed: {rc}"))
                    if args.fail_fast:
                        return summarize(failures, skipped, args.hide_gaps)
                    break
            missing = missing_required(gate)

        if missing:
            msg = "missing required oracle/artifact: " + ", ".join(str(x) for x in missing)
            if args.allow_missing:
                print(f"SKIP {gate.gate_id}: {msg}")
                print("     rerun with --refresh-oracles to regenerate when supported")
                skipped.append((gate.gate_id, msg))
            else:
                print(f"FAIL {gate.gate_id}: {msg}")
                print("     rerun with --refresh-oracles to regenerate when supported")
                failures.append((gate.gate_id, msg))
                if args.fail_fast:
                    return summarize(failures, skipped, args.hide_gaps)
            continue

        gate_failed = False
        for cmd in gate.commands:
            rc = run_cmd(cmd, args.dry_run, timeout)
            if rc != 0:
                failures.append((gate.gate_id, f"command failed: {rc}"))
                gate_failed = True
                break

        if gate_failed:
            print(f"FAIL {gate.gate_id}")
            if args.fail_fast:
                return summarize(failures, skipped, args.hide_gaps)
        else:
            print(f"PASS {gate.gate_id}")

    return summarize(failures, skipped, args.hide_gaps)


def summarize(
    failures: list[tuple[str, str]], skipped: list[tuple[str, str]], hide_gaps: bool
) -> int:
    print("=" * 88)
    print("LTX2 parity summary")
    print(f"  failures: {len(failures)}")
    for gate_id, reason in failures:
        print(f"    FAIL {gate_id}: {reason}")
    print(f"  skipped:  {len(skipped)}")
    for gate_id, reason in skipped:
        print(f"    SKIP {gate_id}: {reason}")
    if not hide_gaps:
        print("")
        print("Known missing gates before production acceptance:")
        for gap in KNOWN_GAPS:
            print(f"  - {gap}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
