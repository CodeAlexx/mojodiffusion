#!/usr/bin/env python3
"""Static/header guard for ERNIE OneTrainer LoRA key parity.

Checks the local OneTrainer Ernie LoRA wrapper/saver source, the Mojo Ernie
OneTrainer save path, and an optional safetensors header. The saved-header path
is intentionally exact: every expected implemented Ernie adapter must have BF16
``.alpha`` / ``.lora_down.weight`` / ``.lora_up.weight`` tensors with the
expected dimensions, and no unsupported extra surfaces may be present.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ONETRAINER = Path("/home/alex/OneTrainer")

OT_SETUP = ONETRAINER / "modules/modelSetup/ErnieLoRASetup.py"
OT_MODEL = ONETRAINER / "modules/model/ErnieModel.py"
OT_SAVER = ONETRAINER / "modules/modelSaver/ernie/ErnieLoRASaver.py"
OT_SAVER_MIXIN = ONETRAINER / "modules/modelSaver/mixin/LoRASaverMixin.py"
OT_LORA_MODULE = ONETRAINER / "modules/module/LoRAModule.py"
OT_CONFIG = ONETRAINER / "configs/ernie_eri2_100step_baseline.json"
OT_BASELINE = ONETRAINER / "output/ernie_eri2_100step_baseline/lora.safetensors"

STACK = REPO / "serenitymojo/models/ernie/ernie_stack_lora.mojo"
LORA_SAVE = REPO / "serenitymojo/training/lora_save.mojo"
SMOKE = REPO / "serenitymojo/models/ernie/parity/ernie_lora_ot_save_key_smoke.mojo"

DEFAULT_NUM_LAYERS = 36
DEFAULT_RANK = 16
DEFAULT_HIDDEN_SIZE = 4096
DEFAULT_FFN_SIZE = 12288
DEFAULT_DTYPE = "BF16"

SMOKE_NUM_LAYERS = 1
SMOKE_RANK = 2
SMOKE_HIDDEN_SIZE = 8
SMOKE_FFN_SIZE = 16
SMOKE_ADAPTERS = 7

OT_MODULES = (
    "self_attention.to_q",
    "self_attention.to_k",
    "self_attention.to_v",
    "self_attention.to_out.0",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.linear_fc2",
)
SAVE_SUFFIXES = ("alpha", "lora_down.weight", "lora_up.weight")


def die(msg: str) -> None:
    raise SystemExit(f"[ernie-lora-keys] FAIL: {msg}")


def read_text(path: Path) -> str:
    if not path.exists():
        die(f"missing required file: {path}")
    return path.read_text(encoding="utf-8")


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        die(f"missing {label}: {needle}")


def strip_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def function_body(text: str, name: str) -> str:
    match = re.search(rf"^def {re.escape(name)}\b", text, flags=re.MULTILINE)
    if match is None:
        die(f"missing def {name}")
    boundary = re.search(
        r"^(?:def|struct|comptime|from|import) \w+\b",
        text[match.end() :],
        flags=re.MULTILINE,
    )
    end = len(text) if boundary is None else match.end() + boundary.start()
    return strip_comments(text[match.start() : end])


def read_safetensors_header(path: Path) -> dict:
    if not path.exists():
        die(f"missing safetensors file: {path}")
    with path.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            die(f"safetensors too small: {path}")
        header_len = struct.unpack("<Q", raw_len)[0]
        header_raw = f.read(header_len)
    if len(header_raw) != header_len:
        die(f"truncated safetensors header: {path}")
    return json.loads(header_raw.decode("utf-8"))


def saved_prefix(layer: int, module: str) -> str:
    return f"transformer.layers.{layer}.{module}"


def saved_key(layer: int, module: str, suffix: str) -> str:
    return f"{saved_prefix(layer, module)}.{suffix}"


def suffix_for(key: str) -> str | None:
    for suffix in SAVE_SUFFIXES:
        if key.endswith(f".{suffix}"):
            return suffix
    return None


def adapter_prefix(key: str) -> str | None:
    suffix = suffix_for(key)
    if suffix is None:
        return None
    return key[: -len(f".{suffix}")]


def expected_prefixes(num_layers: int) -> list[str]:
    return [saved_prefix(layer, module) for layer in range(num_layers) for module in OT_MODULES]


def module_dims(module: str, hidden_size: int, ffn_size: int) -> tuple[int, int]:
    if module.startswith("self_attention."):
        return hidden_size, hidden_size
    if module in {"mlp.gate_proj", "mlp.up_proj"}:
        return hidden_size, ffn_size
    if module == "mlp.linear_fc2":
        return ffn_size, hidden_size
    die(f"unhandled Ernie module shape: {module}")


def expected_shape(module: str, suffix: str, rank: int, hidden_size: int, ffn_size: int) -> list[int]:
    if suffix == "alpha":
        return []
    in_f, out_f = module_dims(module, hidden_size, ffn_size)
    if suffix == "lora_down.weight":
        return [rank, in_f]
    if suffix == "lora_up.weight":
        return [out_f, rank]
    die(f"unhandled suffix shape: {suffix}")


def dtype_from_config(value: str | None) -> str | None:
    if value is None:
        return None
    return {
        "BFLOAT_16": "BF16",
        "FLOAT_16": "F16",
        "FLOAT_32": "F32",
    }.get(value)


def categorize_prefix(prefix: str) -> str:
    if prefix.startswith("transformer.layers."):
        return "unsupported_transformer_layer_module"
    if prefix.startswith("transformer."):
        return "unsupported_transformer_nonlayer_module"
    if prefix.startswith("text_encoder."):
        return "unsupported_text_encoder"
    if prefix.startswith("vae."):
        return "unsupported_vae"
    if prefix.startswith("bundle_emb."):
        return "unsupported_embedding"
    return "unknown"


def ordered(counter: Counter) -> dict:
    return {key: counter[key] for key in sorted(counter)}


def check_ot_source(config_path: Path | None) -> tuple[int | None, str | None]:
    setup = read_text(OT_SETUP)
    model = read_text(OT_MODEL)
    saver = read_text(OT_SAVER)
    saver_mixin = read_text(OT_SAVER_MIXIN)
    lora_module = read_text(OT_LORA_MODULE)

    require(setup, "LoRAModuleWrapper", "OT LoRA wrapper")
    require(setup, 'model.transformer, "transformer"', "OT transformer wrapper prefix")
    require(setup, "config.layer_filter.split(\",\")", "OT layer filter")
    require(setup, "model.text_encoder.requires_grad_(False)", "OT text encoder frozen")
    require(setup, "model.vae.requires_grad_(False)", "OT VAE frozen")
    require(model, "return [a for a in [\n            self.transformer_lora,", "OT Ernie adapters list")

    require(lora_module, "module.state_dict(prefix=module.prefix)", "OT state_dict prefix")
    require(lora_module, "self.register_buffer(\"alpha\"", "OT alpha buffer")
    require(lora_module, "lora_down.weight", "OT lora_down suffix")
    require(lora_module, "lora_up.weight", "OT lora_up suffix")

    require(saver, "return None", "OT Ernie no legacy conversion keyset")
    require(saver, "state_dict |= model.transformer_lora.state_dict()", "OT transformer LoRA save")
    require(saver, "state_dict |= model.lora_state_dict", "OT carried LoRA state save")
    require(saver_mixin, "self.__save_legacy_safetensors", "default legacy safetensors save")

    config_rank: int | None = None
    config_dtype: str | None = None
    if config_path is not None and config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        if cfg.get("model_type") != "ERNIE":
            die(f"baseline config model_type={cfg.get('model_type')!r}, expected 'ERNIE'")
        if cfg.get("training_method") != "LORA":
            die(f"baseline config training_method={cfg.get('training_method')!r}, expected 'LORA'")
        if cfg.get("layer_filter") != "self_attention,mlp":
            die(f"baseline config layer_filter={cfg.get('layer_filter')!r}, expected 'self_attention,mlp'")
        if bool(cfg.get("layer_filter_regex", False)):
            die("baseline config layer_filter_regex must be false for exact Ernie inventory")
        if not bool(cfg.get("transformer", {}).get("train", False)):
            die("baseline config transformer.train must be true")
        if bool(cfg.get("text_encoder", {}).get("train", False)):
            die("baseline config selects unsupported text_encoder LoRA surface")
        if "lora_rank" in cfg:
            config_rank = int(cfg["lora_rank"])
        config_dtype = dtype_from_config(cfg.get("output_dtype"))
        print(
            "[ernie-lora-keys] OneTrainer baseline config: PASS "
            f"rank={config_rank or DEFAULT_RANK} dtype={config_dtype or DEFAULT_DTYPE} "
            "filter=self_attention,mlp"
        )

    print("[ernie-lora-keys] OneTrainer source contract: PASS")
    return config_rank, config_dtype


def check_mojo_sources(strict_port: bool) -> None:
    stack = read_text(STACK)
    lora_save = read_text(LORA_SAVE)
    smoke = read_text(SMOKE)
    save_body = function_body(lora_save, "save_lora_onetrainer")

    require(stack, "ERNIE_SLOTS", "Mojo ERNIE slot count")
    require(stack, "build_ernie_lora_set", "Mojo ERNIE LoRA set builder")
    require(stack, "ernie_lora_prefixes", "Mojo ERNIE prefix inventory")
    require(stack, "save_ernie_lora", "Mojo ERNIE save hook")
    require(stack, "save_lora_onetrainer", "Mojo OneTrainer save call")
    require(stack, 'String("transformer.layers.")', "Mojo ERNIE wrapper prefix")
    for module in OT_MODULES:
        require(stack, f".{module}", f"Mojo ERNIE module {module}")

    require(lora_save, "def save_lora_onetrainer", "shared OneTrainer saver")
    require(save_body, ".lora_down.weight", "shared OT down suffix")
    require(save_body, ".lora_up.weight", "shared OT up suffix")
    require(save_body, ".alpha", "shared OT alpha suffix")
    require(save_body, "_bf16_scalar", "shared BF16 alpha helper")
    require(save_body, "_bf16_2d(a.a.copy(), a.rank, a.in_f, ctx)", "shared BF16 down tensor")
    require(save_body, "_bf16_2d(a.b.copy(), a.out_f, a.rank, ctx)", "shared BF16 up tensor")
    if "_f32_2d" in save_body or "STDtype.F32" in save_body:
        die("save_lora_onetrainer stores OneTrainer LoRA tensors through an F32 boundary")

    load_body = function_body(lora_save, "load_lora_for_resume")
    require(load_body, "adapter_scale = alpha_h[0] / Float32(rank)", "raw-key resume alpha scale")

    require(smoke, "SMOKE_NUM_LAYERS", "smoke reduced layer count assertion")
    require(smoke, "SMOKE_RANK", "smoke reduced rank assertion")
    require(smoke, "SMOKE_HIDDEN", "smoke reduced hidden dimension assertion")
    require(smoke, "SMOKE_FFN", "smoke reduced FFN dimension assertion")
    require(smoke, "_require_exact_inventory", "smoke exact saved inventory guard")

    gaps: list[str] = []
    if "save_lora_peft(named" in stack:
        gaps.append("Mojo ERNIE stack delegates product save to generic PEFT naming")
    if ".lora_down.weight" not in save_body or ".lora_up.weight" not in save_body:
        gaps.append("shared saver does not emit raw OneTrainer lora_down/lora_up")
    if ".alpha" not in save_body:
        gaps.append("shared saver does not emit raw OneTrainer alpha")
    if "_f32_2d" in save_body or "STDtype.F32" in save_body:
        gaps.append("shared OneTrainer saver stores production LoRA tensors as F32")
    if gaps:
        for gap in gaps:
            print(f"[ernie-lora-keys] WARN: {gap}")
        if strict_port:
            die("strict port requested and Ernie LoRA save/key gaps remain")

    print("[ernie-lora-keys] Mojo source scaffold: PASS")


def check_shape(
    header: dict,
    key: str,
    module: str,
    suffix: str,
    rank: int,
    hidden_size: int,
    ffn_size: int,
    expected_dtype: str,
) -> None:
    info = header[key]
    dtype = info.get("dtype")
    if dtype != expected_dtype:
        die(f"{key} dtype {dtype} != {expected_dtype}")
    shape = info.get("shape")
    want_shape = expected_shape(module, suffix, rank, hidden_size, ffn_size)
    if shape != want_shape:
        die(f"{key} shape {shape} != {want_shape}")
    offsets = info.get("data_offsets")
    if suffix == "alpha":
        if offsets is not None and len(offsets) == 2 and offsets[1] - offsets[0] != 2:
            die(f"{key} BF16 scalar byte span {offsets[1] - offsets[0]} != 2")


def check_saved_header(
    path: Path,
    num_layers: int,
    expected_rank: int,
    hidden_size: int,
    ffn_size: int,
    expected_dtype: str,
    expected_adapters: int | None,
) -> None:
    header = read_safetensors_header(path)
    keys = sorted(k for k in header if k != "__metadata__")
    expected = expected_prefixes(num_layers)
    expected_prefix_set = set(expected)
    expected_adapter_count = num_layers * len(OT_MODULES)
    if expected_adapters is not None and expected_adapters != expected_adapter_count:
        die(
            f"expected-adapters={expected_adapters} does not match implemented "
            f"Ernie surface {expected_adapter_count}"
        )

    bad_suffix = [key for key in keys if suffix_for(key) is None]
    if bad_suffix:
        die(f"unsupported saved suffixes/keys present: {bad_suffix[:8]}")

    prefixes = {adapter_prefix(key) for key in keys}
    if None in prefixes:
        die("internal prefix parse failed")
    actual_prefixes = {p for p in prefixes if p is not None}
    missing = sorted(expected_prefix_set - actual_prefixes)
    extra = sorted(actual_prefixes - expected_prefix_set)
    if missing:
        die(f"missing expected prefixes: {missing[:8]}")
    if extra:
        categories = Counter(categorize_prefix(prefix) for prefix in extra)
        die(f"unsupported/unexpected prefixes: {extra[:8]} categories={ordered(categories)}")

    expected_tensor_count = expected_adapter_count * len(SAVE_SUFFIXES)
    if len(keys) != expected_tensor_count:
        die(f"tensor count mismatch: got {len(keys)} expected {expected_tensor_count}")
    if len(actual_prefixes) != expected_adapter_count:
        die(f"adapter count mismatch: got {len(actual_prefixes)} expected {expected_adapter_count}")

    suffix_counts = Counter(suffix_for(key) for key in keys)
    if suffix_counts != Counter({suffix: expected_adapter_count for suffix in SAVE_SUFFIXES}):
        die(f"suffix inventory mismatch: got {ordered(suffix_counts)}")

    dtype_counts = Counter(header[key].get("dtype") for key in keys)
    if dtype_counts != Counter({expected_dtype: len(keys)}):
        die(f"dtype mismatch: got {ordered(dtype_counts)} expected all {expected_dtype}")

    for layer in range(num_layers):
        for module in OT_MODULES:
            for suffix in SAVE_SUFFIXES:
                key = saved_key(layer, module, suffix)
                if key not in header:
                    die(f"missing saved key: {key}")
                check_shape(header, key, module, suffix, expected_rank, hidden_size, ffn_size, expected_dtype)

    print(
        "[ernie-lora-keys] saved inventory: PASS "
        f"layers={num_layers} adapters={expected_adapter_count} tensors={len(keys)} "
        f"rank={expected_rank} hidden={hidden_size} ffn={ffn_size} dtype={expected_dtype}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-port", action="store_true")
    parser.add_argument("--safetensors", type=Path, default=OT_BASELINE if OT_BASELINE.exists() else None)
    parser.add_argument("--config", type=Path, default=OT_CONFIG if OT_CONFIG.exists() else None)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--expected-adapters", type=int)
    parser.add_argument("--expected-rank", type=int)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--ffn-size", type=int, default=DEFAULT_FFN_SIZE)
    parser.add_argument("--expected-dtype", type=str)
    parser.add_argument(
        "--smoke-dims",
        action="store_true",
        help="use the reduced 1-layer smoke dimensions/rank/adapter count",
    )
    args = parser.parse_args()

    config_rank, config_dtype = check_ot_source(args.config)
    check_mojo_sources(args.strict_port)

    expected_rank = args.expected_rank if args.expected_rank is not None else config_rank or DEFAULT_RANK
    expected_dtype = args.expected_dtype or config_dtype or DEFAULT_DTYPE
    expected_adapters = args.expected_adapters
    if args.smoke_dims:
        args.num_layers = SMOKE_NUM_LAYERS
        expected_rank = SMOKE_RANK
        args.hidden_size = SMOKE_HIDDEN_SIZE
        args.ffn_size = SMOKE_FFN_SIZE
        expected_adapters = SMOKE_ADAPTERS

    if args.safetensors is not None:
        check_saved_header(
            args.safetensors,
            args.num_layers,
            expected_rank,
            args.hidden_size,
            args.ffn_size,
            expected_dtype,
            expected_adapters,
        )
    else:
        print("[ernie-lora-keys] saved inventory: SKIP (no --safetensors)")

    print("[ernie-lora-keys] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
