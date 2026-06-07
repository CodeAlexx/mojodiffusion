#!/usr/bin/env python3
"""Report-first guard for Chroma OneTrainer LoRA key parity."""

from __future__ import annotations

import argparse
import json
import re
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ONETRAINER = Path("/home/alex/OneTrainer")
PARITY = Path("/home/alex/onetrainer-mojo/parity")

OT_SETUP = ONETRAINER / "modules/modelSetup/ChromaLoRASetup.py"
OT_SAVER = ONETRAINER / "modules/modelSaver/chroma/ChromaLoRASaver.py"
OT_SAVER_MIXIN = ONETRAINER / "modules/modelSaver/mixin/LoRASaverMixin.py"
OT_CONVERTER = ONETRAINER / "modules/util/convert/lora/convert_chroma_lora.py"
OT_CONVERT_UTIL = ONETRAINER / "modules/util/convert/lora/convert_lora_util.py"
OT_CONFIG = ONETRAINER / "configs/chroma_100step_baseline.json"
OT_BASELINE = ONETRAINER / "output/chroma_100step_baseline/lora.safetensors"
CHROMA_STEP_DUMP = PARITY / "chroma_train_ref_step000.safetensors"
CHROMA_ADAPTER_DUMP = PARITY / "chroma_train_ref_step000_adapters.safetensors"

STACK = REPO / "serenitymojo/models/chroma/chroma_stack_lora.mojo"
CHROMA_ARTIFACT_GATE = REPO / "serenitymojo/models/chroma/parity/chroma_train_ref_artifact_smoke.mojo"
CHROMA_ADAPTER_UPDATE_GATE = REPO / "scripts/check_chroma_adapter_update_replay.py"

SAVE_SUFFIXES = ("alpha", "lora_down.weight", "lora_up.weight")
DEFAULT_NUM_DOUBLE = 19
DEFAULT_NUM_SINGLE = 38
DEFAULT_D = 3072
DEFAULT_FMLP = 12288


@dataclass(frozen=True)
class Target:
    raw_name: str
    legacy_name: str
    in_f: int | None
    out_f: int | None


@dataclass(frozen=True)
class PrefixSpec:
    prefix: str
    raw_name: str
    in_f: int | None
    out_f: int | None
    group: str


DOUBLE_TARGETS = (
    Target("attn.to_q", "attn_to_q", DEFAULT_D, DEFAULT_D),
    Target("attn.to_k", "attn_to_k", DEFAULT_D, DEFAULT_D),
    Target("attn.to_v", "attn_to_v", DEFAULT_D, DEFAULT_D),
    Target("attn.add_q_proj", "attn_add_q_proj", DEFAULT_D, DEFAULT_D),
    Target("attn.add_k_proj", "attn_add_k_proj", DEFAULT_D, DEFAULT_D),
    Target("attn.add_v_proj", "attn_add_v_proj", DEFAULT_D, DEFAULT_D),
    Target("attn.to_out.0", "attn_to_out_0", DEFAULT_D, DEFAULT_D),
    Target("ff.net.0.proj", "ff_net_0_proj", DEFAULT_D, DEFAULT_FMLP),
    Target("ff.net.2", "ff_net_2", DEFAULT_FMLP, DEFAULT_D),
    Target("attn.to_add_out", "attn_to_add_out", DEFAULT_D, DEFAULT_D),
    Target("ff_context.net.0.proj", "ff_context_net_0_proj", DEFAULT_D, DEFAULT_FMLP),
    Target("ff_context.net.2", "ff_context_net_2", DEFAULT_FMLP, DEFAULT_D),
)

SINGLE_TARGETS = (
    Target("attn.to_q", "attn_to_q", DEFAULT_D, DEFAULT_D),
    Target("attn.to_k", "attn_to_k", DEFAULT_D, DEFAULT_D),
    Target("attn.to_v", "attn_to_v", DEFAULT_D, DEFAULT_D),
    Target("proj_mlp", "proj_mlp", DEFAULT_D, DEFAULT_FMLP),
    Target("proj_out", "proj_out", DEFAULT_D + DEFAULT_FMLP, DEFAULT_D),
)

STACK_TARGETS = (
    Target("context_embedder", "context_embedder", None, None),
    Target("proj_out", "proj_out", None, None),
    Target("x_embedder", "x_embedder", None, None),
)

DISTILLED_TARGETS = (
    Target("distilled_guidance_layer.layers.*.linear_1", "distilled_guidance_layer_layers_*_linear_1", None, None),
    Target("distilled_guidance_layer.layers.*.linear_2", "distilled_guidance_layer_layers_*_linear_2", None, None),
)


def read_text(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"missing required file: {path}")
    return path.read_text(encoding="utf-8")


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise SystemExit(f"[chroma-lora-keys] missing {label}: {needle}")


def read_safetensors_header(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"[chroma-lora-keys] missing safetensors file: {path}")
    with path.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise SystemExit(f"[chroma-lora-keys] safetensors too small: {path}")
        header_len = struct.unpack("<Q", raw_len)[0]
        header_raw = f.read(header_len)
    if len(header_raw) != header_len:
        raise SystemExit(f"[chroma-lora-keys] truncated safetensors header: {path}")
    return json.loads(header_raw.decode("utf-8"))


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


def dtype_from_config(value: str | None) -> str | None:
    if value is None:
        return None
    return {
        "BFLOAT_16": "BF16",
        "FLOAT_16": "F16",
        "FLOAT_32": "F32",
    }.get(value)


def target_matches(raw_name: str, filters: list[str], use_regex: bool) -> bool:
    for pattern in filters:
        pattern = pattern.strip()
        if pattern == "":
            return True
        if use_regex:
            if re.search(pattern, raw_name):
                return True
        elif pattern in raw_name:
            return True
    return False


def compile_regex_filters(filters: list[str]) -> None:
    for pattern in filters:
        pattern = pattern.strip()
        if pattern:
            re.compile(pattern)


def build_expected_specs(
    filters: list[str],
    use_regex: bool,
    transformer_train: bool,
    num_double: int,
    num_single: int,
) -> tuple[list[PrefixSpec], list[str]]:
    if use_regex:
        compile_regex_filters(filters)

    specs: list[PrefixSpec] = []
    notes: list[str] = []

    if transformer_train:
        for target in STACK_TARGETS:
            if target_matches(target.raw_name, filters, use_regex):
                specs.append(
                    PrefixSpec(
                        f"lora_transformer_{target.legacy_name}",
                        target.raw_name,
                        target.in_f,
                        target.out_f,
                        "stack",
                    )
                )

        for bi in range(num_double):
            for target in DOUBLE_TARGETS:
                raw_name = f"transformer_blocks.{bi}.{target.raw_name}"
                if target_matches(raw_name, filters, use_regex):
                    specs.append(
                        PrefixSpec(
                            f"lora_transformer_transformer_blocks_{bi}_{target.legacy_name}",
                            raw_name,
                            target.in_f,
                            target.out_f,
                            "double",
                        )
                    )

        for bi in range(num_single):
            for target in SINGLE_TARGETS:
                raw_name = f"single_transformer_blocks.{bi}.{target.raw_name}"
                if target_matches(raw_name, filters, use_regex):
                    specs.append(
                        PrefixSpec(
                            f"lora_transformer_single_transformer_blocks_{bi}_{target.legacy_name}",
                            raw_name,
                            target.in_f,
                            target.out_f,
                            "single",
                        )
                    )

        if any(target_matches(t.raw_name.replace("*", "0"), filters, use_regex) for t in DISTILLED_TARGETS):
            notes.append(
                "distilled_guidance_layer LoRA targets are selected, but this static guard does not "
                "instantiate Chroma to count actual guidance layers"
            )

    return specs, notes


def load_expected_from_config(
    path: Path | None, num_double: int, num_single: int
) -> tuple[list[PrefixSpec] | None, int | None, str | None, str, list[str]]:
    if path is None or not path.exists():
        return None, None, None, "<none>", []

    cfg = json.loads(path.read_text(encoding="utf-8"))
    filters = [part.strip() for part in cfg.get("layer_filter", "").split(",")]
    if not filters:
        filters = [""]
    use_regex = bool(cfg.get("layer_filter_regex", False))
    transformer_train = bool(cfg.get("transformer", {}).get("train", False))
    text_encoder_train = bool(cfg.get("text_encoder", {}).get("train", False))

    specs, notes = build_expected_specs(filters, use_regex, transformer_train, num_double, num_single)
    if text_encoder_train:
        notes.append("text_encoder.train is true; static expected inventory excludes T5 lora_te target count")

    rank = cfg.get("lora_rank")
    expected_rank = int(rank) if rank is not None else None
    expected_dtype = dtype_from_config(cfg.get("output_dtype"))
    filter_label = ",".join(filters)
    return specs, expected_rank, expected_dtype, filter_label, notes


def categorize_prefix(prefix: str) -> str:
    if prefix.startswith("lora_transformer_transformer_blocks_"):
        return "double"
    if prefix.startswith("lora_transformer_single_transformer_blocks_"):
        return "single"
    if prefix.startswith("lora_transformer_distilled_guidance_layer"):
        return "distilled_guidance_layer"
    if prefix.startswith("lora_te"):
        return "text_encoder"
    if prefix.startswith("lora_transformer_"):
        return "stack"
    if prefix.startswith("bundle_emb."):
        return "bundle_embedding"
    return "other"


def check_shape(
    header: dict,
    prefix: str,
    suffix: str,
    expected_rank: int | None,
    in_f: int | None,
    out_f: int | None,
) -> None:
    key = f"{prefix}.{suffix}"
    shape = header[key].get("shape", [])
    if suffix == "alpha":
        if shape != []:
            raise SystemExit(f"[chroma-lora-keys] {key} alpha shape {shape} != []")
        return

    if len(shape) != 2:
        raise SystemExit(f"[chroma-lora-keys] {key} shape rank {len(shape)} != 2")

    if expected_rank is not None:
        rank_axis = 0 if suffix == "lora_down.weight" else 1
        if shape[rank_axis] != expected_rank:
            raise SystemExit(
                f"[chroma-lora-keys] {key} rank axis {shape[rank_axis]} != {expected_rank}"
            )

    if expected_rank is not None and in_f is not None and out_f is not None:
        expected_shape = [expected_rank, in_f] if suffix == "lora_down.weight" else [out_f, expected_rank]
        if shape != expected_shape:
            raise SystemExit(f"[chroma-lora-keys] {key} shape {shape} != {expected_shape}")


def check_ot_source() -> None:
    setup = read_text(OT_SETUP)
    saver = read_text(OT_SAVER)
    saver_mixin = read_text(OT_SAVER_MIXIN)
    converter = read_text(OT_CONVERTER)
    convert_util = read_text(OT_CONVERT_UTIL)

    require(setup, "LoRAModuleWrapper", "OT LoRA wrapper")
    require(setup, 'model.text_encoder, "lora_te"', "OT text encoder prefix")
    require(setup, 'model.transformer, "lora_transformer"', "OT transformer prefix")
    require(setup, "config.layer_filter.split", "OT layer filter")
    require(setup, 'state_dict_has_prefix(model.lora_state_dict, "lora_te")', "OT text encoder resume trigger")

    require(saver, "convert_chroma_lora_key_sets", "Chroma saver converter")
    require(saver, "model.text_encoder_lora.state_dict()", "Chroma text encoder state save")
    require(saver, "model.transformer_lora.state_dict()", "Chroma transformer state save")
    require(saver_mixin, "case ModelFormat.SAFETENSORS", "SAFETENSORS save branch")
    require(saver_mixin, "self.__save_legacy_safetensors", "default legacy safetensors save")

    require(converter, 'LoraConversionKeySet("transformer", "lora_transformer")', "Chroma transformer conversion root")
    require(converter, 'map_t5(LoraConversionKeySet("t5", "lora_te"))', "Chroma T5 conversion root")
    for target in (
        "context_embedder",
        "proj_out",
        "x_embedder",
        "attn.to_q",
        "attn.add_q_proj",
        "ff.net.0.proj",
        "ff_context.net.0.proj",
        "proj_mlp",
        "distilled_guidance_layer.layers",
    ):
        require(converter, target, f"Chroma converter target {target}")
    require(convert_util, "legacy_diffusers_prefix = self.diffusers_prefix.replace('.', '_')", "legacy underscore key conversion")

    print("[chroma-lora-keys] OneTrainer source contract: PASS")
    print(
        "[chroma-lora-keys] OneTrainer setup inventory: "
        "raw prefixes transformer=lora_transformer text_encoder=lora_te "
        "layer_filter=config.layer_filter.split(',')"
    )
    print(
        "[chroma-lora-keys] OneTrainer conversion inventory: "
        "default SAFETENSORS save uses legacy-diffusers underscore prefixes; "
        "block roots lora_transformer_transformer_blocks_{i}_* and "
        "lora_transformer_single_transformer_blocks_{i}_*"
    )
    print(
        "[chroma-lora-keys] OneTrainer Chroma targets: "
        "stack=context_embedder,proj_out,x_embedder; "
        "double=attn q/k/v, add_q/add_k/add_v, to_out.0, to_add_out, ff.net, ff_context.net; "
        "single=attn q/k/v, proj_mlp, proj_out; optional text_encoder=lora_te"
    )


def check_saved_header(
    path: Path,
    expected_specs: list[PrefixSpec] | None,
    expected_adapters: int | None,
    expected_rank: int | None,
    expected_dtype: str | None,
) -> None:
    header = read_safetensors_header(path)
    keys = sorted(k for k in header if k != "__metadata__")
    suffix_counts = Counter(suffix_for(k) for k in keys)
    dtype_counts = Counter(header[k].get("dtype") for k in keys)
    prefixes = sorted({prefix for key in keys if (prefix := adapter_prefix(key)) is not None})
    group_counts = Counter(categorize_prefix(prefix) for prefix in prefixes)

    if suffix_counts.get("<other>", 0) != 0:
        raise SystemExit(f"[chroma-lora-keys] non-LoRA suffixes in saved file: {ordered(suffix_counts)}")

    if expected_specs is not None:
        expected_adapters = len(expected_specs)
        expected_prefixes = {spec.prefix: spec for spec in expected_specs}
        actual_prefixes = set(prefixes)
        missing = sorted(set(expected_prefixes) - actual_prefixes)
        extra = sorted(actual_prefixes - set(expected_prefixes))
        if missing:
            raise SystemExit(f"[chroma-lora-keys] missing expected prefixes: {missing[:8]}")
        if extra:
            raise SystemExit(f"[chroma-lora-keys] unexpected prefixes: {extra[:8]}")
    else:
        expected_prefixes = {}

    if expected_adapters is not None and len(prefixes) != expected_adapters:
        raise SystemExit(
            f"[chroma-lora-keys] adapter count mismatch: got {len(prefixes)} expected {expected_adapters}"
        )

    for suffix in SAVE_SUFFIXES:
        if suffix_counts[suffix] != len(prefixes):
            raise SystemExit(
                f"[chroma-lora-keys] suffix count mismatch for {suffix}: {ordered(suffix_counts)}"
            )

    if expected_dtype is not None and dtype_counts != {expected_dtype: len(keys)}:
        raise SystemExit(
            f"[chroma-lora-keys] dtype mismatch: got {ordered(dtype_counts)} expected all {expected_dtype}"
        )

    for prefix in prefixes:
        spec = expected_prefixes.get(prefix)
        in_f = spec.in_f if spec is not None else None
        out_f = spec.out_f if spec is not None else None
        for suffix in SAVE_SUFFIXES:
            key = f"{prefix}.{suffix}"
            if key not in header:
                raise SystemExit(f"[chroma-lora-keys] missing saved key: {key}")
            check_shape(header, prefix, suffix, expected_rank, in_f, out_f)

    print(
        "[chroma-lora-keys] OneTrainer saved inventory: PASS "
        f"path={path} tensors={len(keys)} adapters={len(prefixes)} "
        f"groups={ordered(group_counts)} suffixes={ordered(suffix_counts)} dtypes={ordered(dtype_counts)}"
    )


def check_mojo_sources(
    strict_port: bool,
    baseline_adapter_count: int | None,
    expected_specs: list[PrefixSpec] | None,
) -> None:
    stack = read_text(STACK)

    blockers: list[str] = []
    warnings: list[str] = []
    if "save_flux_lora" in stack or "load_flux_lora_resume" in stack:
        blockers.append(
            "Chroma stack imports Flux save/load entrypoints; Chroma has no owned "
            "chroma_lora_prefixes/save_chroma_lora/load_chroma_lora_resume contract."
        )

    layer_filter_tokens = (
        "chroma_lora_prefixes_for_layer_filter",
        "save_chroma_lora_for_layer_filter",
        "save_chroma_lora_state_for_layer_filter",
        "load_chroma_lora_resume_for_layer_filter",
        "load_chroma_lora_state_for_layer_filter",
        "_chroma_layer_filter_matches",
    )
    if not all(token in stack for token in layer_filter_tokens):
        blockers.append("Chroma stack is missing layer_filter-aware prefix/save/resume/state entrypoints.")

    if baseline_adapter_count == 304 and "attn,ff.net" not in stack:
        blockers.append("Chroma stack does not document or smoke the local layer_filter='attn,ff.net' 304-adapter baseline.")

    selected_groups = {spec.group for spec in expected_specs or []}
    if expected_specs is not None:
        unsupported = selected_groups - {"double", "single"}
        if unsupported:
            blockers.append(
                "selected Chroma OneTrainer baseline includes unsupported Mojo LoRA groups: "
                + ",".join(sorted(unsupported))
            )

    full_block_count = DEFAULT_NUM_DOUBLE * len(DOUBLE_TARGETS) + DEFAULT_NUM_SINGLE * len(SINGLE_TARGETS)
    if "build_flux_lora_set" in stack or "FluxLoraSet" in stack:
        if baseline_adapter_count is not None:
            warnings.append(
                f"Chroma full block carrier remains available ({full_block_count} adapters); "
                f"selected local baseline is {baseline_adapter_count} via layer_filter-aware save/resume."
            )
        else:
            warnings.append(
                f"Chroma full block carrier remains available ({full_block_count} adapters); "
                "no local baseline inventory was selected."
            )
    if "lora_te" not in stack or "save_chroma_text_encoder" not in stack:
        warnings.append("Chroma stack has no text-encoder lora_te save/resume/training surface for TE-enabled OneTrainer configs.")
    if not all(token in stack for token in ("lora_transformer_x_embedder", "lora_transformer_context_embedder", "lora_transformer_proj_out")):
        warnings.append(
            "Chroma stack freezes stack-level OneTrainer transformer targets "
            "x_embedder/context_embedder/proj_out; this is outside the selected local baseline unless those targets match layer_filter."
        )
    if "distilled_guidance_layer" in stack and "lora_transformer_distilled_guidance_layer" not in stack:
        warnings.append(
            "Chroma stack documents the distilled_guidance_layer approximator as frozen; "
            "this is outside the selected local baseline unless those targets match layer_filter."
        )

    if blockers:
        for blocker in blockers:
            print(f"[chroma-lora-keys] WARN: {blocker}")
        if strict_port:
            raise SystemExit("[chroma-lora-keys] strict port requested and Chroma Mojo blockers remain")
    for warning in warnings:
        print(f"[chroma-lora-keys] WARN: {warning}")
    else:
        if not warnings:
            print("[chroma-lora-keys] Mojo source scaffold: PASS")
    if not blockers:
        print("[chroma-lora-keys] Mojo source selected baseline scaffold: PASS")


def check_no_false_artifact_claims() -> None:
    claim_re = re.compile(
        r"Chroma[^\n]{0,80}(?:one-step|replay|train ref|artifact)[^\n]{0,80}(?:PASS|consumer|consumes)",
        re.I,
    )
    offenders: list[Path] = []
    for root, suffix in (
        (REPO / "serenitymojo/models/chroma", ".mojo"),
        (REPO / "scripts", ".py"),
    ):
        if not root.exists():
            continue
        for path in root.rglob(f"*{suffix}"):
            text = path.read_text(encoding="utf-8", errors="replace")
            if claim_re.search(text) and "chroma_train_ref" not in text:
                offenders.append(path)

    if offenders:
        rels = [str(path.relative_to(REPO)) for path in offenders]
        raise SystemExit(
            "[chroma-lora-keys] false Chroma one-step artifact consumer claim(s): "
            + ", ".join(rels)
        )
    print("[chroma-lora-keys] Chroma train-ref false-claim scan: PASS")


def check_train_ref_artifact_consumer() -> None:
    artifacts = (CHROMA_STEP_DUMP, CHROMA_ADAPTER_DUMP)
    missing = [path for path in artifacts if not path.exists()]
    if missing:
        check_no_false_artifact_claims()
        print("[chroma-lora-keys] Chroma train-ref artifacts: MISSING")
        for path in artifacts:
            exists = path.exists()
            print(f"  {path} exists={str(exists).lower()}")
        raise SystemExit(
            "[chroma-lora-keys] missing local Chroma OneTrainer train-ref artifact(s): "
            + ", ".join(str(path) for path in missing)
        )

    if not CHROMA_ARTIFACT_GATE.exists():
        raise SystemExit(
            "[chroma-lora-keys] no in-repo Chroma artifact consumer found; "
            f"expected {CHROMA_ARTIFACT_GATE}"
        )

    text = read_text(CHROMA_ARTIFACT_GATE)
    required_needles = (
        str(CHROMA_STEP_DUMP),
        str(CHROMA_ADAPTER_DUMP),
        "SafeTensors.open",
        "_check_step_dump",
        "_check_adapter_dump",
        "no transformer/backward/AdamW parity",
    )
    missing_needles = [needle for needle in required_needles if needle not in text]
    if missing_needles:
        raise SystemExit(
            "[chroma-lora-keys] Chroma artifact consumer does not prove it consumes the local dump: "
            + ", ".join(missing_needles)
        )

    print(
        "[chroma-lora-keys] Chroma train-ref artifacts: PRESENT "
        f"step={CHROMA_STEP_DUMP} adapter={CHROMA_ADAPTER_DUMP}"
    )
    print(
        "[chroma-lora-keys] Chroma train-ref in-repo consumer: PASS "
        f"{CHROMA_ARTIFACT_GATE.relative_to(REPO)}"
    )

    if not CHROMA_ADAPTER_UPDATE_GATE.exists():
        raise SystemExit(
            "[chroma-lora-keys] missing Chroma adapter update oracle gate: "
            f"{CHROMA_ADAPTER_UPDATE_GATE}"
        )
    update_gate_text = read_text(CHROMA_ADAPTER_UPDATE_GATE)
    for needle in (
        "check_adapter_update_replay",
        '"chroma"',
    ):
        if needle not in update_gate_text:
            raise SystemExit(
                "[chroma-lora-keys] Chroma adapter update gate is not wired to the "
                f"shared replay checker; missing {needle!r}"
            )
    print(
        "[chroma-lora-keys] Chroma adapter update oracle gate: PASS "
        f"{CHROMA_ADAPTER_UPDATE_GATE.relative_to(REPO)}"
    )
    check_no_false_artifact_claims()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-port", action="store_true")
    parser.add_argument("--safetensors", type=Path, default=OT_BASELINE if OT_BASELINE.exists() else None)
    parser.add_argument("--config", type=Path, default=OT_CONFIG if OT_CONFIG.exists() else None)
    parser.add_argument("--expected-adapters", type=int, default=None)
    parser.add_argument("--expected-rank", type=int, default=None)
    parser.add_argument("--expected-dtype", type=str, default=None)
    parser.add_argument("--num-double", type=int, default=DEFAULT_NUM_DOUBLE)
    parser.add_argument("--num-single", type=int, default=DEFAULT_NUM_SINGLE)
    args = parser.parse_args()

    check_ot_source()

    expected_specs, config_rank, config_dtype, filter_label, notes = load_expected_from_config(
        args.config, args.num_double, args.num_single
    )
    expected_rank = args.expected_rank if args.expected_rank is not None else config_rank
    expected_dtype = args.expected_dtype if args.expected_dtype is not None else config_dtype
    baseline_adapter_count = len(expected_specs) if expected_specs is not None else args.expected_adapters

    if expected_specs is not None:
        group_counts = Counter(spec.group for spec in expected_specs)
        print(
            "[chroma-lora-keys] Expected baseline from config: "
            f"config={args.config} layer_filter={filter_label!r} "
            f"rank={expected_rank} dtype={expected_dtype} adapters={len(expected_specs)} "
            f"groups={ordered(group_counts)}"
        )
    else:
        print("[chroma-lora-keys] Expected baseline from config: SKIP (config missing)")

    for note in notes:
        print(f"[chroma-lora-keys] NOTE: {note}")

    if args.safetensors is not None:
        check_saved_header(
            args.safetensors,
            expected_specs,
            args.expected_adapters,
            expected_rank,
            expected_dtype,
        )
    else:
        print("[chroma-lora-keys] OneTrainer saved inventory: SKIP (baseline safetensors not present)")

    check_mojo_sources(args.strict_port, baseline_adapter_count, expected_specs)
    check_train_ref_artifact_consumer()
    print("[chroma-lora-keys] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
