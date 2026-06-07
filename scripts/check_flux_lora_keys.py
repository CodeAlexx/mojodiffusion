#!/usr/bin/env python3
"""Static/header guard for Flux.1-dev OneTrainer LoRA key parity."""

from __future__ import annotations

import argparse
import json
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ONETRAINER = Path("/home/alex/OneTrainer")

OT_SETUP = ONETRAINER / "modules/modelSetup/FluxLoRASetup.py"
OT_SAVER = ONETRAINER / "modules/modelSaver/flux/FluxLoRASaver.py"
OT_SAVER_MIXIN = ONETRAINER / "modules/modelSaver/mixin/LoRASaverMixin.py"
OT_CONVERTER = ONETRAINER / "modules/util/convert/lora/convert_flux_lora.py"
OT_CONVERT_UTIL = ONETRAINER / "modules/util/convert/lora/convert_lora_util.py"
OT_BASELINE = ONETRAINER / "output/flux1_100step_baseline/lora_last.safetensors"
STACK = REPO / "serenitymojo/models/flux/flux_stack_lora.mojo"
LORA_SAVE = REPO / "serenitymojo/training/lora_save.mojo"

SAVE_SUFFIXES = ("alpha", "lora_down.weight", "lora_up.weight")
DEFAULT_NUM_DOUBLE = 19
DEFAULT_NUM_SINGLE = 38
DEFAULT_D = 3072
DEFAULT_FMLP = 12288
DEFAULT_CONTEXT_DIM = 4096
DEFAULT_VECTOR_DIM = 768
DEFAULT_TIME_DIM = 256
DEFAULT_IN_CH = 64
DEFAULT_OUT_CH = 64
SAMPLE_OT_KEYS = (
    "lora_transformer_x_embedder",
    "lora_transformer_transformer_blocks_0_attn_to_q",
    "lora_transformer_transformer_blocks_0_attn_add_q_proj",
    "lora_transformer_single_transformer_blocks_0_norm_linear",
    "lora_transformer_proj_out",
)
SAMPLE_MOJO_SUPPORTED_KEYS = (
    "lora_transformer_transformer_blocks_0_attn_to_q",
    "lora_transformer_transformer_blocks_0_attn_add_q_proj",
    "lora_transformer_transformer_blocks_0_ff_context_net_2",
    "lora_transformer_single_transformer_blocks_0_proj_out",
)
FLUX_DEV_SUPPORTED_TRANSFORMER_ADAPTERS = 19 * 12 + 38 * 5
FLUX_DEV_OT_TRANSFORMER_ADAPTERS = 10 + 19 * 14 + 38 * 6
FLUX_DEV_MISSING_TRANSFORMER_ADAPTERS = (
    FLUX_DEV_OT_TRANSFORMER_ADAPTERS - FLUX_DEV_SUPPORTED_TRANSFORMER_ADAPTERS
)


@dataclass(frozen=True)
class PrefixSpec:
    prefix: str
    in_f: int
    out_f: int
    group: str
    implemented: bool
    ot_module: str = ""


STACK_TARGETS = (
    ("context_embedder", "context_embedder", DEFAULT_CONTEXT_DIM, DEFAULT_D),
    ("norm_out_linear", "norm_out.linear", DEFAULT_D, 2 * DEFAULT_D),
    ("proj_out", "proj_out", DEFAULT_D, DEFAULT_OUT_CH),
    (
        "time_text_embed_guidance_embedder_linear_1",
        "time_text_embed.guidance_embedder.linear_1",
        DEFAULT_TIME_DIM,
        DEFAULT_D,
    ),
    (
        "time_text_embed_guidance_embedder_linear_2",
        "time_text_embed.guidance_embedder.linear_2",
        DEFAULT_D,
        DEFAULT_D,
    ),
    (
        "time_text_embed_text_embedder_linear_1",
        "time_text_embed.text_embedder.linear_1",
        DEFAULT_VECTOR_DIM,
        DEFAULT_D,
    ),
    (
        "time_text_embed_text_embedder_linear_2",
        "time_text_embed.text_embedder.linear_2",
        DEFAULT_D,
        DEFAULT_D,
    ),
    (
        "time_text_embed_timestep_embedder_linear_1",
        "time_text_embed.timestep_embedder.linear_1",
        DEFAULT_TIME_DIM,
        DEFAULT_D,
    ),
    (
        "time_text_embed_timestep_embedder_linear_2",
        "time_text_embed.timestep_embedder.linear_2",
        DEFAULT_D,
        DEFAULT_D,
    ),
    ("x_embedder", "x_embedder", DEFAULT_IN_CH, DEFAULT_D),
)

def read_text(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"missing required file: {path}")
    return path.read_text(encoding="utf-8")


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise SystemExit(f"[flux-lora-keys] missing {label}: {needle}")


def read_safetensors_header(path: Path) -> dict:
    raw = path.read_bytes()
    if len(raw) < 8:
        raise SystemExit(f"safetensors too small: {path}")
    header_len = struct.unpack("<Q", raw[:8])[0]
    return json.loads(raw[8 : 8 + header_len].decode("utf-8"))


def suffix_for(key: str) -> str:
    for suffix in SAVE_SUFFIXES:
        if key.endswith(f".{suffix}"):
            return suffix
    return "<other>"


def adapter_prefix(key: str) -> str | None:
    for suffix in SAVE_SUFFIXES:
        trailer = f".{suffix}"
        if key.endswith(trailer):
            return key[: -len(trailer)]
    return None


def ordered(counter: Counter) -> dict:
    return {key: counter[key] for key in sorted(counter)}


def _stack_specs() -> list[PrefixSpec]:
    return [
        PrefixSpec(
            f"lora_transformer_{legacy_name}",
            in_f,
            out_f,
            "stack",
            False,
            f"lora_transformer.{ot_module}",
        )
        for legacy_name, ot_module, in_f, out_f in STACK_TARGETS
    ]


def _double_image_targets(d_model: int, f_mlp: int) -> tuple[tuple[str, str, int, int], ...]:
    return (
        ("attn_to_q", "attn.to_q", d_model, d_model),
        ("attn_to_k", "attn.to_k", d_model, d_model),
        ("attn_to_v", "attn.to_v", d_model, d_model),
        ("attn_to_out_0", "attn.to_out.0", d_model, d_model),
        ("ff_net_0_proj", "ff.net.0.proj", d_model, f_mlp),
        ("ff_net_2", "ff.net.2", f_mlp, d_model),
    )


def _double_text_targets(d_model: int, f_mlp: int) -> tuple[tuple[str, str, int, int], ...]:
    return (
        ("attn_add_q_proj", "attn.add_q_proj", d_model, d_model),
        ("attn_add_k_proj", "attn.add_k_proj", d_model, d_model),
        ("attn_add_v_proj", "attn.add_v_proj", d_model, d_model),
        ("attn_to_add_out", "attn.to_add_out", d_model, d_model),
        ("ff_context_net_0_proj", "ff_context.net.0.proj", d_model, f_mlp),
        ("ff_context_net_2", "ff_context.net.2", f_mlp, d_model),
    )


def _single_targets(d_model: int, f_mlp: int) -> tuple[tuple[str, str, int, int], ...]:
    return (
        ("attn_to_q", "attn.to_q", d_model, d_model),
        ("attn_to_k", "attn.to_k", d_model, d_model),
        ("attn_to_v", "attn.to_v", d_model, d_model),
        ("proj_mlp", "proj_mlp", d_model, f_mlp),
        ("proj_out", "proj_out", d_model + f_mlp, d_model),
    )


def _double_supported_specs(
    bi: int, d_model: int = DEFAULT_D, f_mlp: int = DEFAULT_FMLP
) -> list[PrefixSpec]:
    specs: list[PrefixSpec] = []
    base = f"lora_transformer_transformer_blocks_{bi}_"
    ot_base = f"lora_transformer.transformer_blocks.{bi}."
    for legacy_name, ot_suffix, in_f, out_f in _double_image_targets(d_model, f_mlp):
        specs.append(PrefixSpec(base + legacy_name, in_f, out_f, "double", True, ot_base + ot_suffix))
    for legacy_name, ot_suffix, in_f, out_f in _double_text_targets(d_model, f_mlp):
        specs.append(PrefixSpec(base + legacy_name, in_f, out_f, "double", True, ot_base + ot_suffix))
    return specs


def _double_missing_specs(bi: int, d_model: int = DEFAULT_D) -> list[PrefixSpec]:
    base = f"lora_transformer_transformer_blocks_{bi}_"
    ot_base = f"lora_transformer.transformer_blocks.{bi}."
    return [
        PrefixSpec(
            base + "norm1_linear",
            d_model,
            6 * d_model,
            "double_norm",
            False,
            ot_base + "norm1.linear",
        ),
        PrefixSpec(
            base + "norm1_context_linear",
            d_model,
            6 * d_model,
            "double_norm",
            False,
            ot_base + "norm1_context.linear",
        ),
    ]


def _single_supported_specs(
    bi: int, d_model: int = DEFAULT_D, f_mlp: int = DEFAULT_FMLP
) -> list[PrefixSpec]:
    base = f"lora_transformer_single_transformer_blocks_{bi}_"
    ot_base = f"lora_transformer.single_transformer_blocks.{bi}."
    return [
        PrefixSpec(base + legacy_name, in_f, out_f, "single", True, ot_base + ot_suffix)
        for legacy_name, ot_suffix, in_f, out_f in _single_targets(d_model, f_mlp)
    ]


def _single_missing_specs(bi: int, d_model: int = DEFAULT_D) -> list[PrefixSpec]:
    base = f"lora_transformer_single_transformer_blocks_{bi}_"
    return [
        PrefixSpec(
            base + "norm_linear",
            d_model,
            3 * d_model,
            "single_norm",
            False,
            f"lora_transformer.single_transformer_blocks.{bi}.norm.linear",
        )
    ]


def build_supported_specs(
    num_double: int,
    num_single: int,
    d_model: int = DEFAULT_D,
    f_mlp: int = DEFAULT_FMLP,
) -> list[PrefixSpec]:
    specs: list[PrefixSpec] = []
    for bi in range(num_double):
        specs.extend(_double_supported_specs(bi, d_model, f_mlp))
    for bi in range(num_single):
        specs.extend(_single_supported_specs(bi, d_model, f_mlp))
    return specs


def build_missing_specs(
    num_double: int, num_single: int, d_model: int = DEFAULT_D
) -> list[PrefixSpec]:
    specs = _stack_specs()
    for bi in range(num_double):
        specs.extend(_double_missing_specs(bi, d_model))
    for bi in range(num_single):
        specs.extend(_single_missing_specs(bi, d_model))
    return specs


def build_ot_transformer_specs(
    num_double: int,
    num_single: int,
    d_model: int = DEFAULT_D,
    f_mlp: int = DEFAULT_FMLP,
) -> list[PrefixSpec]:
    return build_supported_specs(num_double, num_single, d_model, f_mlp) + build_missing_specs(
        num_double, num_single, d_model
    )


def expected_specs_for_adapter_count(expected_adapters: int | None) -> list[PrefixSpec] | None:
    if expected_adapters == FLUX_DEV_OT_TRANSFORMER_ADAPTERS:
        return build_ot_transformer_specs(DEFAULT_NUM_DOUBLE, DEFAULT_NUM_SINGLE)
    if expected_adapters == FLUX_DEV_SUPPORTED_TRANSFORMER_ADAPTERS:
        return build_supported_specs(DEFAULT_NUM_DOUBLE, DEFAULT_NUM_SINGLE)
    if expected_adapters == 17:
        return build_supported_specs(1, 1, d_model=8, f_mlp=16)
    return None


def check_expected_partition() -> None:
    supported = build_supported_specs(DEFAULT_NUM_DOUBLE, DEFAULT_NUM_SINGLE)
    missing = build_missing_specs(DEFAULT_NUM_DOUBLE, DEFAULT_NUM_SINGLE)
    full = build_ot_transformer_specs(DEFAULT_NUM_DOUBLE, DEFAULT_NUM_SINGLE)
    supported_set = {spec.prefix for spec in supported}
    missing_set = {spec.prefix for spec in missing}
    full_set = {spec.prefix for spec in full}
    if len(supported) != FLUX_DEV_SUPPORTED_TRANSFORMER_ADAPTERS:
        raise SystemExit(
            "[flux-lora-keys] checker supported inventory mismatch: "
            f"{len(supported)} != {FLUX_DEV_SUPPORTED_TRANSFORMER_ADAPTERS}"
        )
    if len(full) != FLUX_DEV_OT_TRANSFORMER_ADAPTERS:
        raise SystemExit(
            "[flux-lora-keys] checker OT inventory mismatch: "
            f"{len(full)} != {FLUX_DEV_OT_TRANSFORMER_ADAPTERS}"
        )
    if len(missing) != FLUX_DEV_MISSING_TRANSFORMER_ADAPTERS:
        raise SystemExit(
            "[flux-lora-keys] checker missing inventory mismatch: "
            f"{len(missing)} != {FLUX_DEV_MISSING_TRANSFORMER_ADAPTERS}"
        )
    overlap = sorted(supported_set & missing_set)
    if overlap:
        raise SystemExit(f"[flux-lora-keys] implemented/missing overlap: {overlap[:8]}")
    if supported_set | missing_set != full_set:
        raise SystemExit("[flux-lora-keys] supported+missing does not equal full OT inventory")


def _shape_text(spec: PrefixSpec) -> str:
    return f"in={spec.in_f} out={spec.out_f}"


def _print_grouped_missing_targets(num_double: int, num_single: int) -> None:
    missing = build_missing_specs(num_double, num_single)
    counts = Counter(spec.group for spec in missing)
    if len(missing) != FLUX_DEV_MISSING_TRANSFORMER_ADAPTERS:
        raise SystemExit(
            "[flux-lora-keys] missing-target report count mismatch: "
            f"{len(missing)} != {FLUX_DEV_MISSING_TRANSFORMER_ADAPTERS}"
        )

    print(
        "[flux-lora-keys] Missing OT transformer targets by OneTrainer module path "
        f"(fail-loud, not saveable yet): count={len(missing)} groups={ordered(counts)}"
    )
    for group in ("stack", "double_norm", "single_norm"):
        group_specs = [spec for spec in missing if spec.group == group]
        if not group_specs:
            continue
        print(f"[flux-lora-keys]   {group}: {len(group_specs)}")
        for spec in group_specs:
            print(
                "[flux-lora-keys]     "
                f"{spec.ot_module} -> {spec.prefix} ({_shape_text(spec)})"
            )


def check_shape(
    header: dict, prefix: str, suffix: str, expected_rank: int | None, spec: PrefixSpec | None
) -> None:
    key = f"{prefix}.{suffix}"
    shape = header[key].get("shape", [])
    if suffix == "alpha":
        if shape != []:
            raise SystemExit(f"[flux-lora-keys] {key} alpha shape {shape} != []")
        return

    if len(shape) != 2:
        raise SystemExit(f"[flux-lora-keys] {key} rank {len(shape)} != 2")

    if expected_rank is not None:
        rank_axis = 0 if suffix == "lora_down.weight" else 1
        if shape[rank_axis] != expected_rank:
            raise SystemExit(
                f"[flux-lora-keys] {key} rank axis {shape[rank_axis]} != {expected_rank}"
            )
        if spec is not None:
            expected_shape = (
                [expected_rank, spec.in_f]
                if suffix == "lora_down.weight"
                else [spec.out_f, expected_rank]
            )
            if shape != expected_shape:
                raise SystemExit(f"[flux-lora-keys] {key} shape {shape} != {expected_shape}")


def check_ot_source() -> None:
    setup = read_text(OT_SETUP)
    saver = read_text(OT_SAVER)
    saver_mixin = read_text(OT_SAVER_MIXIN)
    converter = read_text(OT_CONVERTER)
    convert_util = read_text(OT_CONVERT_UTIL)

    require(setup, "LoRAModuleWrapper", "OT LoRA wrapper")
    require(setup, 'model.transformer, "lora_transformer"', "OT transformer prefix")
    require(setup, 'model.text_encoder_1, "lora_te1"', "OT text encoder 1 prefix")
    require(setup, 'model.text_encoder_2, "lora_te2"', "OT text encoder 2 prefix")
    require(setup, "config.layer_filter.split", "OT layer filter")
    require(setup, 'state_dict_has_prefix(model.lora_state_dict, "lora_te1")', "OT TE1 resume trigger")
    require(setup, 'state_dict_has_prefix(model.lora_state_dict, "lora_te2")', "OT TE2 resume trigger")

    require(saver, "convert_flux_lora_key_sets", "Flux saver converter")
    require(saver, "model.text_encoder_1_lora.state_dict()", "Flux TE1 state save")
    require(saver, "model.text_encoder_2_lora.state_dict()", "Flux TE2 state save")
    require(saver, "model.transformer_lora.state_dict()", "Flux transformer state save")
    require(saver_mixin, "case ModelFormat.SAFETENSORS", "SAFETENSORS save branch")
    require(saver_mixin, "self.__save_legacy_safetensors", "default legacy safetensors save")

    require(converter, 'LoraConversionKeySet("transformer", "lora_transformer")', "Flux transformer conversion root")
    require(converter, 'map_clip(LoraConversionKeySet("clip_l", "lora_te1"))', "Flux CLIP conversion root")
    require(converter, 'map_t5(LoraConversionKeySet("t5", "lora_te2"))', "Flux T5 conversion root")
    for target in (
        "context_embedder",
        "norm_out.linear",
        "proj_out",
        "time_text_embed.guidance_embedder.linear_1",
        "time_text_embed.text_embedder.linear_1",
        "time_text_embed.timestep_embedder.linear_1",
        "x_embedder",
        "norm1.linear",
        "norm1_context.linear",
        "norm.linear",
        "ff_context.net.2",
    ):
        require(converter, target, f"Flux converter target {target}")
    require(convert_util, "legacy_diffusers_prefix = self.diffusers_prefix.replace('.', '_')", "legacy underscore key conversion")

    print("[flux-lora-keys] OneTrainer source contract: PASS")


def check_saved_header(path: Path, expected_adapters: int | None, expected_rank: int | None) -> None:
    header = read_safetensors_header(path)
    keys = sorted(k for k in header if k != "__metadata__")
    suffix_counts = Counter(suffix_for(k) for k in keys)
    dtype_counts = Counter(header[k].get("dtype") for k in keys)
    prefixes = sorted({prefix for key in keys if (prefix := adapter_prefix(key)) is not None})
    expected_specs = expected_specs_for_adapter_count(expected_adapters)
    expected_by_prefix = {spec.prefix: spec for spec in expected_specs or []}
    group_counts = Counter(
        expected_by_prefix[prefix].group if prefix in expected_by_prefix else "<unknown>"
        for prefix in prefixes
    )

    if suffix_counts.get("<other>", 0) != 0:
        raise SystemExit(f"[flux-lora-keys] non-LoRA suffixes in saved file: {ordered(suffix_counts)}")

    if expected_specs is not None:
        actual_prefixes = set(prefixes)
        expected_prefixes = set(expected_by_prefix)
        missing = sorted(expected_prefixes - actual_prefixes)
        extra = sorted(actual_prefixes - expected_prefixes)
        if missing:
            raise SystemExit(f"[flux-lora-keys] missing expected prefixes: {missing[:8]}")
        if extra:
            raise SystemExit(f"[flux-lora-keys] unexpected prefixes: {extra[:8]}")

    if expected_adapters is not None:
        expected_tensors = expected_adapters * len(SAVE_SUFFIXES)
        if len(keys) != expected_tensors:
            raise SystemExit(
                f"[flux-lora-keys] tensor count mismatch: got {len(keys)} expected {expected_tensors}"
            )
        if len(prefixes) != expected_adapters:
            raise SystemExit(
                f"[flux-lora-keys] adapter count mismatch: got {len(prefixes)} expected {expected_adapters}"
            )
    if dtype_counts != {"BF16": len(keys)}:
        raise SystemExit(f"[flux-lora-keys] dtype mismatch: {ordered(dtype_counts)}")
    for suffix in SAVE_SUFFIXES:
        if suffix_counts[suffix] != len(prefixes):
            raise SystemExit(f"[flux-lora-keys] suffix count mismatch: {ordered(suffix_counts)}")

    sample_keys = SAMPLE_OT_KEYS
    if expected_adapters == FLUX_DEV_SUPPORTED_TRANSFORMER_ADAPTERS or expected_adapters == 17:
        sample_keys = SAMPLE_MOJO_SUPPORTED_KEYS

    for prefix in sample_keys:
        for suffix in SAVE_SUFFIXES:
            key = f"{prefix}.{suffix}"
            if key not in header:
                raise SystemExit(f"[flux-lora-keys] missing saved key: {key}")
            check_shape(header, prefix, suffix, expected_rank, expected_by_prefix.get(prefix))

    for prefix in prefixes:
        spec = expected_by_prefix.get(prefix)
        for suffix in SAVE_SUFFIXES:
            key = f"{prefix}.{suffix}"
            if key not in header:
                raise SystemExit(f"[flux-lora-keys] missing saved key: {key}")
            check_shape(header, prefix, suffix, expected_rank, spec)

    print(
        "[flux-lora-keys] OneTrainer saved inventory: PASS "
        f"tensors={len(keys)} adapters={len(prefixes)} "
        f"groups={ordered(group_counts)} suffixes={ordered(suffix_counts)} "
        f"dtypes={ordered(dtype_counts)}"
    )


def check_mojo_sources(strict_port: bool, report_missing: bool) -> None:
    check_expected_partition()
    stack = read_text(STACK)
    lora_save = read_text(LORA_SAVE)

    require(stack, "save_flux_lora", "Mojo Flux save hook")
    require(stack, "DBL_SLOTS_PER_BLOCK", "Mojo Flux double slot count")
    require(stack, "SGL_SLOTS", "Mojo Flux single slot count")
    require(lora_save, "def save_lora_onetrainer", "shared OneTrainer saver")
    require(lora_save, ".lora_down.weight", "shared OT down suffix")
    require(lora_save, ".lora_up.weight", "shared OT up suffix")
    require(lora_save, ".alpha", "shared OT alpha suffix")
    require(stack, "flux_lora_ot_transformer_prefixes", "Flux full OT transformer inventory")
    require(stack, "flux_lora_missing_ot_transformer_prefixes", "Flux missing OT transformer inventory")
    require(stack, "require_flux_lora_ot_transformer_complete", "Flux fail-loud transformer guard")
    require(stack, "require_flux_lora_text_encoder_disabled", "Flux fail-loud TE guard")
    require(stack, "lora_transformer_context_embedder", "OT stack context embedder key")
    require(stack, "lora_transformer_x_embedder", "OT stack x embedder key")
    require(stack, "lora_transformer_norm_out_linear", "OT final norm modulation key")
    require(stack, "lora_transformer_proj_out", "OT final projection key")
    require(stack, "time_text_embed_guidance_embedder_linear_1", "OT guidance embedder key")
    require(stack, "time_text_embed_text_embedder_linear_1", "OT pooled text embedder key")
    require(stack, "time_text_embed_timestep_embedder_linear_1", "OT timestep embedder key")
    require(stack, "norm1_context_linear", "OT double context norm modulation key")
    require(stack, "norm1_linear", "OT double image norm modulation key")
    require(stack, "norm_linear", "OT single norm modulation key")
    require(stack, "lora_te1", "explicit TE1 unsupported guard")
    require(stack, "lora_te2", "explicit TE2 unsupported guard")

    blockers: list[str] = []
    if "save_lora_peft(named, path, ctx)" in stack:
        blockers.append("Mojo Flux save path still writes PEFT lora_A/lora_B keys")
    if "save_lora_onetrainer" not in stack:
        blockers.append("Mojo Flux save hook does not call save_lora_onetrainer")
    if "double_blocks." in stack or "single_blocks." in stack:
        blockers.append("Mojo Flux prefixes are BFL/local double_blocks/single_blocks, not OneTrainer lora_transformer_* keys")
    if "Flux LoRA transformer surface is not full OneTrainer parity" not in stack:
        blockers.append("Mojo Flux missing OT transformer targets are not fail-loud")
    if "Flux LoRA lora_te1 save/resume surface is not implemented" not in stack:
        blockers.append("Mojo Flux TE1-enabled configs are not fail-loud")
    if "Flux LoRA lora_te2 save/resume surface is not implemented" not in stack:
        blockers.append("Mojo Flux TE2-enabled configs are not fail-loud")

    if blockers:
        for blocker in blockers:
            print(f"[flux-lora-keys] WARN: {blocker}")
        if report_missing:
            _print_grouped_missing_targets(DEFAULT_NUM_DOUBLE, DEFAULT_NUM_SINGLE)
        if strict_port:
            raise SystemExit("[flux-lora-keys] strict port requested and Flux save-key blockers remain")
    else:
        print(
            "[flux-lora-keys] Mojo source scaffold: PASS "
            f"supported_transformer={FLUX_DEV_SUPPORTED_TRANSFORMER_ADAPTERS} "
            f"ot_transformer={FLUX_DEV_OT_TRANSFORMER_ADAPTERS} "
            f"fail_loud_missing={FLUX_DEV_MISSING_TRANSFORMER_ADAPTERS} "
            "te_surface=fail-loud"
        )
        if report_missing:
            _print_grouped_missing_targets(DEFAULT_NUM_DOUBLE, DEFAULT_NUM_SINGLE)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-port", action="store_true")
    parser.add_argument(
        "--report-missing",
        action="store_true",
        help="Print the fail-loud missing Flux transformer targets grouped by OneTrainer module path.",
    )
    parser.add_argument("--safetensors", type=Path, default=OT_BASELINE if OT_BASELINE.exists() else None)
    parser.add_argument("--expected-adapters", type=int, default=504)
    parser.add_argument("--expected-rank", type=int, default=16)
    args = parser.parse_args()

    check_ot_source()
    if args.safetensors is not None:
        check_saved_header(args.safetensors, args.expected_adapters, args.expected_rank)
    check_mojo_sources(args.strict_port, args.report_missing or args.strict_port)
    print("[flux-lora-keys] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
