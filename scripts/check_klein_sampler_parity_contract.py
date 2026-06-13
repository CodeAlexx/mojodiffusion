#!/usr/bin/env python3
"""Static Klein/Flux2 sampler parity evidence guard.

This is intentionally no-CUDA. It inspects OneTrainer reference files and local
Mojo sampler/conditioning surfaces, then reports whether Klein sampler parity is
production-ready. Default mode exits 0 as a report. Use --strict to exit 2 while
the current known blockers remain.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ONETRAINER = Path("/home/alex/OneTrainer")

KLEIN9B_CONFIG = REPO / "serenitymojo/configs/klein9b.json"
OT_FLUX2_MODEL = ONETRAINER / "modules/model/Flux2Model.py"
OT_FLUX2_SAMPLER = ONETRAINER / "modules/modelSampler/Flux2Sampler.py"
OT_FLUX2_DATALOADER = ONETRAINER / "modules/dataLoader/Flux2BaseDataLoader.py"

KLEIN_SAMPLER = REPO / "serenitymojo/sampling/klein_sampler.mojo"
KLEIN_SAMPLE_CLI = REPO / "serenitymojo/sampling/klein_sample_cli.mojo"
KLEIN_PARITY_DUMP = REPO / "serenitymojo/sampling/klein_sampler_parity_dump.mojo"
KLEIN_PARITY_DUMP_CLI = REPO / "serenitymojo/sampling/klein_sampler_parity_dump_cli.mojo"
VALIDATION_SAMPLER = REPO / "serenitymojo/training/validation_sampler.mojo"
KLEIN_TRAIN = REPO / "serenitymojo/training/train_klein_real.mojo"
CAP_CACHE_SRC = REPO / "serenitymojo/io/cap_cache.mojo"
CAP_CACHE_HEADER_SMOKE = REPO / "serenitymojo/io/cap_cache_header_smoke.mojo"
PRECACHE_PROMPTS = REPO / "serenitymojo/pipeline/klein9b_precache_sample_prompts.mojo"
QWEN3_ENCODER = REPO / "serenitymojo/models/text_encoder/qwen3_encoder.mojo"
KLEIN_DECODER = REPO / "serenitymojo/models/vae/klein_decoder.mojo"
FLUX2_KLEIN = REPO / "serenitymojo/sampling/flux2_klein.mojo"
FLUX_FAMILY_GUARD = REPO / "scripts/check_flux_family_sampler_contracts.py"
PRODUCT_HARNESS = REPO / "serenitymojo/sampling/product_sampler_harness.mojo"
KLEIN4B_CONFIG = REPO / "serenitymojo/configs/klein4b.json"
CAP_CACHE_GUARD = REPO / "scripts/check_klein_cap_cache_contract.py"
CONDITIONING_TEMPLATE_GUARD = REPO / "scripts/check_klein_conditioning_template_contract.py"
INITIAL_NOISE_SIDECAR_GUARD = REPO / "scripts/check_klein_initial_noise_sidecar_contract.py"
SAMPLER_ARTIFACT_GUARD = REPO / "scripts/check_klein_sampler_artifact_manifest.py"
USE_STATUS = REPO / "MOJO_TRAINER_USE_STATUS.md"
PORT_STATUS = REPO / "OT_MOJO_PORT_REMAINING.md"

KNOWN_BLOCKERS: tuple[str, ...] = (
    "No accepted token/template parity gate compares OneTrainer Qwen3 Klein chat text and token ids against the local Mojo template.",
    "Configured Klein sample cap-cache files are not proven present as BF16 [1,512,12288] or [512,12288] headers.",
    "No accepted seeded trajectory parity artifact exists yet; the sampler has an explicit initial-noise sidecar entry, but still needs matched OneTrainer post-patch/post-pack noise and latent trajectory evidence.",
    "No accepted numeric VAE/final-PNG parity gate compares inverse-BN, unpatchify, VAE decode, and image output against OneTrainer.",
    "No accepted Klein sampler speed/VRAM parity artifact records matched OneTrainer and Mojo prompt, seed, resolution, steps, CFG, dtype, trajectory, denoise seconds/step, VAE time, and peak VRAM.",
)


@dataclass(frozen=True)
class Source:
    path: Path
    text: str | None

    @property
    def exists(self) -> bool:
        return self.text is not None

    @property
    def rel(self) -> str:
        try:
            return str(self.path.relative_to(REPO))
        except ValueError:
            return str(self.path)

    def has(self, needle: str) -> bool:
        return self.text is not None and needle in self.text

    def line(self, needle: str) -> int | None:
        if self.text is None:
            return None
        index = self.text.find(needle)
        if index < 0:
            return None
        return self.text.count("\n", 0, index) + 1


@dataclass(frozen=True)
class Fact:
    status: str
    label: str
    detail: str
    refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContractReport:
    facts: tuple[Fact, ...]
    known_blockers: tuple[str, ...]
    strict_blockers: tuple[str, ...]
    accepted_sampler_parity: bool
    accepted_speed_parity: bool
    accepted_conditioning_template_parity: bool
    missing_sources: tuple[str, ...]

    @property
    def production_ready(self) -> bool:
        return (
            self.accepted_sampler_parity
            and self.accepted_speed_parity
            and not self.strict_blockers
            and not self.missing_sources
        )


def read_source(path: Path) -> Source:
    try:
        return Source(path, path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Source(path, None)


def ref(src: Source, line: int | None) -> str:
    if line is None:
        return f"{src.rel}:missing"
    return f"{src.rel}:{line}"


def fact_for_needles(
    *,
    status_if_ok: str,
    status_if_missing: str,
    label: str,
    source: Source,
    needles: tuple[str, ...],
    detail_ok: str,
    detail_missing: str,
) -> Fact:
    if not source.exists:
        return Fact("MISSING", label, f"missing source file {source.rel}", (ref(source, None),))
    missing = [needle for needle in needles if not source.has(needle)]
    refs = tuple(ref(source, source.line(needle)) for needle in needles if source.has(needle))
    if missing:
        return Fact(
            status_if_missing,
            label,
            detail_missing + " missing markers: " + ", ".join(repr(item) for item in missing),
            refs,
        )
    return Fact(status_if_ok, label, detail_ok, refs)


def any_negative_near(text: str, term: str) -> bool:
    lower = text.lower()
    term_l = term.lower()
    for match in re.finditer(re.escape(term_l), lower):
        start = max(0, match.start() - 100)
        end = min(len(lower), match.end() + 100)
        context = lower[start:end]
        if re.search(r"\b(no|not|missing|blocked|still needs|must still fail|not accepted)\b", context):
            return True
    return False


def positive_claim_present(sources: tuple[Source, ...], terms: tuple[str, ...]) -> bool:
    for source in sources:
        if source.text is None:
            continue
        for term in terms:
            if term.lower() in source.text.lower() and not any_negative_near(source.text, term):
                return True
    return False


def status_counts(facts: tuple[Fact, ...]) -> dict[str, int]:
    out: dict[str, int] = {}
    for fact in facts:
        out[fact.status] = out.get(fact.status, 0) + 1
    return out


def run_conditioning_template_guard() -> tuple[bool, str]:
    if not CONDITIONING_TEMPLATE_GUARD.exists():
        return (False, f"missing {CONDITIONING_TEMPLATE_GUARD}")
    try:
        result = subprocess.run(
            [sys.executable, str(CONDITIONING_TEMPLATE_GUARD), "--quiet"],
            cwd=str(REPO),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=90,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - report local oracle failures.
        return (False, str(exc))
    detail = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        return (False, detail if detail else f"exit {result.returncode}")
    return (True, "check_klein_conditioning_template_contract.py --quiet passed")


def run_initial_noise_sidecar_guard() -> tuple[bool, str]:
    if not INITIAL_NOISE_SIDECAR_GUARD.exists():
        return (False, f"missing {INITIAL_NOISE_SIDECAR_GUARD}")
    try:
        result = subprocess.run(
            [sys.executable, str(INITIAL_NOISE_SIDECAR_GUARD), "--strict"],
            cwd=str(REPO),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - report local guard failures.
        return (False, str(exc))
    detail = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        return (False, detail if detail else f"exit {result.returncode}")
    return (True, "check_klein_initial_noise_sidecar_contract.py --strict passed")


def run_cap_cache_guard() -> tuple[bool, str]:
    if not CAP_CACHE_GUARD.exists():
        return (False, f"missing {CAP_CACHE_GUARD}")
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(CAP_CACHE_GUARD),
                "--strict",
                "--config",
                str(KLEIN9B_CONFIG),
            ],
            cwd=str(REPO),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - report local guard failures.
        return (False, str(exc))
    detail = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        return (False, detail if detail else f"exit {result.returncode}")
    return (True, "check_klein_cap_cache_contract.py --strict --config serenitymojo/configs/klein9b.json passed")


def run_sampler_artifact_guard() -> tuple[bool, str]:
    if not SAMPLER_ARTIFACT_GUARD.exists():
        return (False, f"missing {SAMPLER_ARTIFACT_GUARD}")
    try:
        result = subprocess.run(
            [sys.executable, str(SAMPLER_ARTIFACT_GUARD), "--strict"],
            cwd=str(REPO),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - report local guard failures.
        return (False, str(exc))
    detail = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        return (False, detail if detail else f"exit {result.returncode}")
    return (True, "check_klein_sampler_artifact_manifest.py --strict passed")


def gather_report() -> ContractReport:
    sources = {
        "ot_model": read_source(OT_FLUX2_MODEL),
        "ot_sampler": read_source(OT_FLUX2_SAMPLER),
        "ot_dataloader": read_source(OT_FLUX2_DATALOADER),
        "klein_sampler": read_source(KLEIN_SAMPLER),
        "sample_cli": read_source(KLEIN_SAMPLE_CLI),
        "parity_dump": read_source(KLEIN_PARITY_DUMP),
        "parity_dump_cli": read_source(KLEIN_PARITY_DUMP_CLI),
        "validation": read_source(VALIDATION_SAMPLER),
        "klein_train": read_source(KLEIN_TRAIN),
        "cap_cache_src": read_source(CAP_CACHE_SRC),
        "cap_header_smoke": read_source(CAP_CACHE_HEADER_SMOKE),
        "precache": read_source(PRECACHE_PROMPTS),
        "qwen": read_source(QWEN3_ENCODER),
        "decoder": read_source(KLEIN_DECODER),
        "flux2": read_source(FLUX2_KLEIN),
        "flux_guard": read_source(FLUX_FAMILY_GUARD),
        "harness": read_source(PRODUCT_HARNESS),
        "klein4b": read_source(KLEIN4B_CONFIG),
        "cap_guard": read_source(CAP_CACHE_GUARD),
        "conditioning_guard": read_source(CONDITIONING_TEMPLATE_GUARD),
        "initial_noise_guard": read_source(INITIAL_NOISE_SIDECAR_GUARD),
        "sampler_artifact_guard": read_source(SAMPLER_ARTIFACT_GUARD),
        "use_status": read_source(USE_STATUS),
        "port_status": read_source(PORT_STATUS),
    }
    missing_sources = tuple(src.rel for src in sources.values() if not src.exists)
    facts: list[Fact] = []
    conditioning_guard_ok, conditioning_guard_detail = run_conditioning_template_guard()
    initial_noise_guard_ok, initial_noise_guard_detail = run_initial_noise_sidecar_guard()
    cap_cache_guard_ok, cap_cache_guard_detail = run_cap_cache_guard()
    sampler_artifact_guard_ok, sampler_artifact_guard_detail = run_sampler_artifact_guard()

    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="OneTrainer Klein conditioning contract",
            source=sources["ot_model"],
            needles=(
                "QWEN3_HIDDEN_STATES_LAYERS = [9, 18, 27]",
                "add_generation_prompt=True",
                "enable_thinking=False",
                "QWEN3_HIDDEN_STATES_LAYERS",
            ),
            detail_ok="OneTrainer Klein uses Qwen3 chat formatting and hidden-state layers [9,18,27].",
            detail_missing="OneTrainer Flux2Model conditioning markers changed; inspect reference manually.",
        )
    )
    facts.append(
        fact_for_needles(
            status_if_ok="PASS" if conditioning_guard_ok else "WARN",
            status_if_missing="WARN",
            label="local conditioning is static/precached only",
            source=sources["precache"],
            needles=("_klein_template", "tok.encode(_klein_template(prompt))", "enc.encode_klein"),
            detail_ok=(
                "Local sample conditioning uses the hand-written template and precached caps; "
                + conditioning_guard_detail
            ),
            detail_missing="local Klein precache markers changed; inspect conditioning parity manually.",
        )
    )
    facts.append(
        fact_for_needles(
            status_if_ok="PASS" if conditioning_guard_ok else "WARN",
            status_if_missing="WARN",
            label="Klein conditioning template/token guard",
            source=sources["conditioning_guard"],
            needles=("AutoTokenizer", "Tokenizer.from_file", "apply_chat_template", "enable_thinking=False", "PAD_ID = 151643"),
            detail_ok=conditioning_guard_detail,
            detail_missing="Klein conditioning template guard is incomplete.",
        )
    )
    facts.append(
        fact_for_needles(
            status_if_ok="PASS" if cap_cache_guard_ok else "WARN",
            status_if_missing="WARN",
            label="Klein cap-cache readiness guard",
            source=sources["cap_guard"],
            needles=("CAP_MAGIC", "STD_BF16_TAG", "KLEIN9B_CAP_SHAPES", "--strict"),
            detail_ok=cap_cache_guard_detail,
            detail_missing="Klein cap-cache guard markers are incomplete.",
        )
    )
    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="Klein cap-cache header validator is no-CUDA",
            source=sources["cap_cache_src"],
            needles=(
                "def validate_klein_cap_cache_header",
                "STDtype.BF16.tag",
                "file_size(fd)",
                "expected BF16 [512,",
            ),
            detail_ok="io/cap_cache validates Klein cap metadata, BF16 dtype, accepted text shapes, and exact file size without a DeviceContext.",
            detail_missing="cap-cache header validator markers changed; inspect pre-CUDA sample cap validation manually.",
        )
    )
    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="Klein train preflights sample caps before CUDA",
            source=sources["klein_train"],
            needles=(
                "validate_klein_cap_cache_header",
                "_validate_precached_caps(sample_cfg, cfg.joint_attention_dim)",
                "var ctx = DeviceContext()",
            ),
            detail_ok="Klein training validates configured sample cap headers before creating DeviceContext when runtime sampling is enabled.",
            detail_missing="Klein train-loop cap preflight markers changed; inspect CUDA-before-cache failure risk manually.",
        )
    )
    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="Klein cap-cache header smoke rejects false dtype/shape",
            source=sources["cap_header_smoke"],
            needles=(
                "valid BF16 [1,512,12288]",
                "valid BF16 [512,12288]",
                "F32 dtype boundary",
                "4B text width for 9B sampler",
            ),
            detail_ok="The no-CUDA smoke covers accepted 2D/3D BF16 9B caps and rejects F32 or 4B-width sample caps.",
            detail_missing="cap-cache header smoke markers changed; inspect no-CUDA validation coverage manually.",
        )
    )
    facts.append(
        fact_for_needles(
            status_if_ok="PASS" if initial_noise_guard_ok else "WARN",
            status_if_missing="WARN",
            label="Klein initial-noise sidecar guard",
            source=sources["initial_noise_guard"],
            needles=("sidecar shape validation", "sidecar dtype preservation", "default production BF16 randn path", "--strict"),
            detail_ok=initial_noise_guard_detail,
            detail_missing="Klein initial-noise sidecar guard is incomplete.",
        )
    )
    facts.append(
        fact_for_needles(
            status_if_ok="PASS" if sampler_artifact_guard_ok else "WARN",
            status_if_missing="WARN",
            label="Klein sampler artifact manifest guard",
            source=sources["sampler_artifact_guard"],
            needles=(
                "Klein/Flux2 sampler artifact manifest guard",
                "onetrainer_initial_noise_raw_nchw",
                "onetrainer_initial_noise_post_patch_nchw",
                "onetrainer_initial_noise_post_pack",
                "onetrainer_latent_trajectory",
                "mojo_latent_trajectory",
                "onetrainer_final_packed_latent",
                "mojo_final_packed_latent",
                "final_unscaled_unpatchified_latent",
                "vae_decoded_tensor",
                "denoise_seconds_per_step",
                "peak_vram_mib",
                "--strict",
            ),
            detail_ok=sampler_artifact_guard_detail,
            detail_missing="Klein sampler artifact manifest guard is incomplete.",
        )
    )
    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="Qwen3 encoder extracts Klein layers",
            source=sources["qwen"],
            needles=("def encode_klein", "states[8]", "states[17]", "states[26]"),
            detail_ok="Mojo Qwen3 encoder exposes the Klein hidden-state concatenation path.",
            detail_missing="Qwen3 Klein layer extraction markers are incomplete.",
        )
    )
    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="initial noise sidecar path exists",
            source=sources["klein_sampler"],
            needles=(
                "randn(nchw^, seed, STDtype.BF16, ctx)",
                "_initial_noise_tokens_from_sidecar",
                "klein_sample_with_initial_noise",
                "post-patch/post-pack",
                "initial_noise.dtype().to_mojo_dtype()",
                "return initial_noise^",
            ),
            detail_ok="Default production sampling still uses Mojo BF16 randn, and the explicit parity entry can consume a supplied OT-equivalent post-patch/post-pack initial-noise tensor without dtype casting.",
            detail_missing="local Klein initial-noise markers changed; inspect RNG parity manually.",
        )
    )
    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="sample CLI accepts parity initial-noise sidecar",
            source=sources["sample_cli"],
            needles=("argv[6] optional post-patch/post-pack initial-noise tensor bin", "load_tensor_bin(initial_noise_path", "parity initial noise (post-patch/post-pack):"),
            detail_ok="Standalone Klein sample CLI can feed a raw tensor-bin post-patch/post-pack initial-noise sidecar into the parity sampler entry.",
            detail_missing="sample CLI initial-noise sidecar markers changed; inspect trajectory replay wiring manually.",
        )
    )
    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="ReferenceLatent edit accepts parity initial-noise sidecar",
            source=sources["sample_cli"],
            needles=(
                "klein_sample_with_reference_latent_initial_noise",
                "edit_initial_noise_replay",
                "initial_noise_sidecar",
                "load_tensor_bin(initial_noise_path, ctx)",
            ),
            detail_ok="Standalone Klein sample CLI can combine ReferenceLatent edit replay with a supplied target post-patch/post-pack initial-noise sidecar.",
            detail_missing="ReferenceLatent edit initial-noise replay markers changed; inspect edit trajectory replay wiring manually.",
        )
    )
    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="ReferenceLatent edit replay dtype boundary",
            source=sources["klein_sampler"],
            needles=(
                "klein_sample_with_reference_latent_initial_noise",
                "if x.dtype() != ref_tokens.dtype():",
                "x = cast_tensor(x, ref_tokens.dtype(), ctx)",
            ),
            detail_ok="ReferenceLatent edit replay casts target sidecar tokens to the reference/model token dtype before concat.",
            detail_missing="ReferenceLatent edit replay dtype-boundary markers changed; inspect F32 oracle sidecar handling manually.",
        )
    )
    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="ReferenceLatent edit parity dump artifacts",
            source=sources["parity_dump"],
            needles=(
                "dump_klein_reference_latent_parity_artifacts",
                "_denoise_lora_reference_from_initial_with_trajectory",
                "build_flux2_img2img_sigmas",
                "prepare_combined_img_ids",
                "mojo_reference_combined_img_ids.bin",
                "mojo_edit_initial_noise_target_tokens.bin",
                "mojo_edit_effective_initial_target_tokens.bin",
                "mojo_edit_combined_tokens_step0.bin",
                "mojo_edit_target_latent_trajectory.bin",
                "\\\"mode\\\":\\\"reference_latent_edit\\\"",
            ),
            detail_ok="Mojo-side ReferenceLatent edit parity dump can emit reference tokens/ids, effective step-0 tokens, edit trajectory, VAE tensor, PNG, and a no-claim manifest.",
            detail_missing="ReferenceLatent edit parity dump markers changed; inspect Mojo edit artifact production manually.",
        )
    )
    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="ReferenceLatent edit parity dump CLI",
            source=sources["parity_dump_cli"],
            needles=(
                "dump_klein_reference_latent_parity_artifacts",
                "_dump_reference_512",
                "_dump_reference_1024",
                "N_EDIT_IMG_512",
                "S_EDIT_1024",
                "reference_vae_latent.bin",
                "denoise_strength",
                "reference_t_offset",
            ),
            detail_ok="Mojo parity dump CLI keeps txt2img replay compatible and adds bounded 512/1024 ReferenceLatent edit dump dispatch for 9B/4B heads.",
            detail_missing="ReferenceLatent edit parity dump CLI markers changed; inspect edit dump dispatch manually.",
        )
    )
    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="OneTrainer sampler RNG and decode path",
            source=sources["ot_sampler"],
            needles=("torch.randn", "dtype=torch.float32", "noise_scheduler.set_timesteps", "self.model.unscale_latents", "vae.decode"),
            detail_ok="OneTrainer Flux2Sampler uses PyTorch F32 noise, scheduler timesteps, unscale/unpatchify, and VAE decode.",
            detail_missing="OneTrainer Flux2Sampler markers changed; inspect reference manually.",
        )
    )
    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="cap-cache sampler boundary preserves cached dtype",
            source=sources["sample_cli"],
            needles=("reshape(caps.pos", "reshape(caps.neg"),
            detail_ok="Standalone Klein sample CLI reshapes cached cap tensors without an F32 storage-boundary cast.",
            detail_missing="sample CLI cap-cache reshape markers changed; inspect dtype boundary manually.",
        )
    )
    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="validation sampler preserves caps and threads multiplier",
            source=sources["validation"],
            needles=("reshape(caps.pos", "reshape(caps.neg", "cfg.n_heads == 24", "lora_multiplier"),
            detail_ok="Validation sampler reshapes cached caps without an F32 boundary cast, dispatches the 4B H=24 specialization, and passes lora_multiplier through to the Klein sampler.",
            detail_missing="validation sampler markers changed; inspect sample parity manually.",
        )
    )
    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="standalone sample CLI dispatches Klein 9B and 4B heads",
            source=sources["sample_cli"],
            needles=("comptime H_9B = 32", "comptime H_4B = 24", "comptime Dh = 128", "klein_sample[N_IMG_1024"),
            detail_ok="The CLI dispatches both 9B H=32 and 4B H=24 specializations at runtime from the config.",
            detail_missing="sample CLI shape markers changed; inspect head-specialization dispatch manually.",
        )
    )
    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="Klein 4B config dimensions are covered by sampler dispatch",
            source=sources["klein4b"],
            needles=('"num_heads": 24', '"joint_attention_dim": 7680'),
            detail_ok="klein4b.json carries 4B dimensions and the sample wrappers now dispatch H=24/Dh=128.",
            detail_missing="klein4b config markers changed; inspect 4B sampler readiness manually.",
        )
    )
    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="Flux2/Klein scalar scheduler helper",
            source=sources["flux2"],
            needles=("compute_empirical_mu", "build_flux2_sigma_schedule", "Flux2KleinScheduler", "flux2_cfg"),
            detail_ok="Mojo has the shared Flux2/Klein scalar scheduler and CFG helper.",
            detail_missing="Flux2/Klein scalar scheduler markers are incomplete.",
        )
    )
    facts.append(
        fact_for_needles(
            status_if_ok="WARN",
            status_if_missing="WARN",
            label="VAE path exists without accepted numeric image parity",
            source=sources["decoder"],
            needles=("_inverse_bn", "_unpatchify_packed", "def decode", "KleinVaeDecoder"),
            detail_ok="Local VAE inverse-BN/unpatchify/decode path exists, but no accepted numeric VAE/final PNG parity gate is present.",
            detail_missing="Klein VAE decode markers are incomplete.",
        )
    )
    facts.append(
        fact_for_needles(
            status_if_ok="PASS",
            status_if_missing="WARN",
            label="Flux-family speed claim guard covers Flux2/Klein",
            source=sources["flux_guard"],
            needles=("FLUX2_KLEIN_CLAIM_ALIASES", "require_flux2_klein_speed_claim_evidence", "Flux2/Klein speed-parity evidence gate"),
            detail_ok="Static guard rejects accepted Flux2/Klein speed-parity claims without matched timing/VRAM/run identity evidence.",
            detail_missing="Flux2/Klein speed-claim guard markers are incomplete.",
        )
    )
    facts.append(
        fact_for_needles(
            status_if_ok="WARN",
            status_if_missing="WARN",
            label="product sampler harness is still scaffold",
            source=sources["harness"],
            needles=("transformer_denoise_ready", "vae_decode_ready", "measurement scaffold only, speed parity not accepted"),
            detail_ok="Product sampler harness has measurement fields, but the current status is scaffold/not accepted speed parity.",
            detail_missing="product sampler harness markers changed; inspect sampler measurement readiness manually.",
        )
    )

    claim_sources = (
        sources["klein_sampler"],
        sources["sample_cli"],
        sources["validation"],
        sources["flux_guard"],
        sources["harness"],
        sources["use_status"],
        sources["port_status"],
    )
    accepted_sampler = positive_claim_present(
        claim_sources,
        ("accepted Klein sampler parity", "Klein sampler parity accepted", "accepted Flux2/Klein sampler parity"),
    )
    accepted_speed = positive_claim_present(
        claim_sources,
        ("accepted Klein sampler speed parity", "accepted Flux2/Klein speed parity", "Flux2/Klein speed parity accepted"),
    )

    known_blockers = list(KNOWN_BLOCKERS)
    if conditioning_guard_ok:
        known_blockers = [blocker for blocker in known_blockers if blocker != KNOWN_BLOCKERS[0]]
    if cap_cache_guard_ok:
        known_blockers = [blocker for blocker in known_blockers if blocker != KNOWN_BLOCKERS[1]]

    strict_blockers: list[str] = []
    if not accepted_sampler:
        if not conditioning_guard_ok:
            strict_blockers.append(KNOWN_BLOCKERS[0])
        if not cap_cache_guard_ok:
            strict_blockers.append(KNOWN_BLOCKERS[1])
        strict_blockers.extend(KNOWN_BLOCKERS[2:4])
    if not accepted_speed:
        strict_blockers.append(KNOWN_BLOCKERS[4])
    if missing_sources:
        strict_blockers.append(
            "Required source/reference files were missing, so this static guard cannot verify Klein sampler evidence: "
            + ", ".join(missing_sources)
        )

    return ContractReport(
        facts=tuple(facts),
        known_blockers=tuple(known_blockers),
        strict_blockers=tuple(strict_blockers),
        accepted_sampler_parity=accepted_sampler,
        accepted_speed_parity=accepted_speed,
        accepted_conditioning_template_parity=conditioning_guard_ok,
        missing_sources=missing_sources,
    )


def print_report(report: ContractReport, *, ref_limit: int) -> None:
    print("Klein sampler parity contract report")
    print(f"repo: {REPO}")
    print("mode: report-only by default; pass --strict to exit 2 for known blockers")
    print("scope: no-CUDA static source/reference inspection; does not run DiT, VAE, or torch")
    print("")

    for fact in report.facts:
        print(f"{fact.status} {fact.label}")
        print(f"  {fact.detail}")
        if fact.refs:
            shown = fact.refs[:ref_limit]
            suffix = "" if len(fact.refs) <= ref_limit else f" ... +{len(fact.refs) - ref_limit} more"
            print(f"  refs: {', '.join(shown)}{suffix}")
        print("")

    print("verdict")
    print(f"  production_ready_klein_sampler_parity: {report.production_ready}")
    print(f"  accepted_sampler_parity: {report.accepted_sampler_parity}")
    print(f"  accepted_speed_parity: {report.accepted_speed_parity}")
    print(f"  accepted_conditioning_template_parity: {report.accepted_conditioning_template_parity}")
    print("")
    print("known blockers")
    for blocker in report.known_blockers:
        print(f"  - {blocker}")
    print("")
    counts = status_counts(report.facts)
    print(
        "summary: "
        f"facts={len(report.facts)} "
        + " ".join(f"{key.lower()}={value}" for key, value in sorted(counts.items()))
        + f" strict_blockers={len(report.strict_blockers)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="exit 2 for known sampler parity blockers")
    parser.add_argument("--ref-limit", type=int, default=5)
    args = parser.parse_args()

    report = gather_report()
    print_report(report, ref_limit=max(0, args.ref_limit))
    if args.strict and report.strict_blockers:
        print("")
        print("STRICT FAIL")
        for blocker in report.strict_blockers:
            print(f"  - {blocker}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
