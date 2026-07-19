# Flux.1 real LoRA file key/shape/dtype gate.
#
# Serenity reference file:
#   /home/alex/Serenity/output/flux1_100step_baseline/lora_last.safetensors
#
# Serenity source:
#   modules/modelSetup/FluxLoRASetup.py
#   modules/modelSaver/flux/FluxLoRASaver.py
#   modules/modelLoader/flux/FluxLoRALoader.py
#   modules/util/convert/lora/convert_flux_lora.py
#
# Gate:
#   Inspect the real Serenity-saved Flux LoRA file with safetensors views only.
#   Verify complete key triplets, BF16 storage dtype, rank 16, scalar alpha
#   tensors, legacy Flux key spelling, and representative transformer shapes.
#   This is key/shape/dtype evidence only; it does not claim Flux train parity.

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.registry.checkpoints import path_exists

from serenity_trainer.util.convert.lora.convert_flux_lora import (
    FLUX_REAL_LORA_EXPECTED_ADAPTERS,
    FLUX_REAL_LORA_EXPECTED_KEYS,
    FLUX_REAL_LORA_EXPECTED_RANK,
    flux_lora_alpha_key,
    flux_lora_candidate_files,
    flux_lora_conversion_summary,
    flux_lora_down_key,
    flux_lora_legacy_prefixed_module,
    flux_lora_up_key,
    flux_representative_lora_target_specs,
)


struct FluxRealFileSummary(Copyable, Movable):
    var key_count: Int
    var adapter_count: Int
    var rank: Int
    var bundle_key_count: Int

    def __init__(
        out self,
        key_count: Int,
        adapter_count: Int,
        rank: Int,
        bundle_key_count: Int,
    ):
        self.key_count = key_count
        self.adapter_count = adapter_count
        self.rank = rank
        self.bundle_key_count = bundle_key_count


def _expect_int(name: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(
            name + String(": got ") + String(got)
            + String(", expected ") + String(expected)
        )


def _expect_prefix(name: String, got: String, prefix: String) raises:
    if not got.startswith(prefix):
        raise Error(name + String(": got ") + got + String(", expected prefix ") + prefix)


def _expect_suffix(name: String, got: String, suffix: String) raises:
    if not got.endswith(suffix):
        raise Error(name + String(": got ") + got + String(", expected suffix ") + suffix)


def _expect_dtype(name: String, got: STDtype, expected: STDtype) raises:
    if got != expected:
        raise Error(
            name + String(": dtype got ") + got.name()
            + String(", expected ") + expected.name()
        )


def _expect_positive(name: String, got: Int) raises:
    if got <= 0:
        raise Error(name + String(": got non-positive dimension ") + String(got))


def _expect_shape_2d(name: String, sh: List[Int], rows: Int, cols: Int) raises:
    _expect_int(name + String(".rank"), len(sh), 2)
    _expect_int(name + String(".rows"), sh[0], rows)
    _expect_int(name + String(".cols"), sh[1], cols)


def _substr_bytes(s: String, start: Int, end: Int) -> String:
    var out = String("")
    var bytes = s.as_bytes()
    for i in range(start, end):
        out += chr(Int(bytes[i]))
    return out


def _strip_suffix(s: String, suffix: String) -> String:
    if not s.endswith(suffix):
        return String()
    if s.byte_length() <= suffix.byte_length():
        return String()
    return _substr_bytes(s, 0, s.byte_length() - suffix.byte_length())


def _is_flux_lora_prefix(prefix: String) -> Bool:
    return (
        prefix.startswith(String("lora_transformer_"))
        or prefix.startswith(String("lora_te1_"))
        or prefix.startswith(String("lora_te2_"))
        or prefix.startswith(String("lora_transformer."))
        or prefix.startswith(String("lora_te1."))
        or prefix.startswith(String("lora_te2."))
        or prefix.startswith(String("transformer."))
        or prefix.startswith(String("clip_l."))
        or prefix.startswith(String("t5."))
    )


def _find_real_file(paths: List[String]) raises -> String:
    for i in range(len(paths)):
        if path_exists(paths[i]):
            return paths[i].copy()
    return String()


def _check_conversion_metadata() raises:
    var summary = flux_lora_conversion_summary()
    _expect_int("flux conversion range", summary.range_upper_bound, 100)
    _expect_int("flux root rules", summary.transformer_root_rule_count, 10)
    _expect_int("flux double rules", summary.double_block_rule_count, 14)
    _expect_int("flux single rules", summary.single_block_rule_count, 6)
    if not summary.has_qkv_split_rules:
        raise Error("Flux conversion metadata missing qkv split rules")
    if not summary.has_swap_chunks_rules:
        raise Error("Flux conversion metadata missing swap-chunks rules")

    var targets = flux_representative_lora_target_specs()
    _expect_int("representative flux target count", targets.len(), 11)
    for i in range(targets.len()):
        _expect_prefix(
            String("representative diffusers prefix ") + String(i),
            targets.diffusers_prefixes[i],
            String("lora_"),
        )
        _expect_suffix(
            String("representative down key ") + String(i),
            flux_lora_down_key(targets.legacy_prefixes[i]),
            String(".lora_down.weight"),
        )
        _expect_suffix(
            String("representative up key ") + String(i),
            flux_lora_up_key(targets.legacy_prefixes[i]),
            String(".lora_up.weight"),
        )
        _expect_suffix(
            String("representative alpha key ") + String(i),
            flux_lora_alpha_key(targets.legacy_prefixes[i]),
            String(".alpha"),
        )


def _check_representative_shape(
    ref st: ShardedSafeTensors,
    prefix: String,
    down_rows: Int,
    down_cols: Int,
    up_rows: Int,
    up_cols: Int,
) raises:
    var down_key = flux_lora_down_key(prefix)
    var up_key = flux_lora_up_key(prefix)
    var down = st.tensor_view(down_key)
    var up = st.tensor_view(up_key)
    _expect_shape_2d(down_key, down.shape, down_rows, down_cols)
    _expect_shape_2d(up_key, up.shape, up_rows, up_cols)


def _check_real_file(path: String, expected_dtype: STDtype) raises -> FluxRealFileSummary:
    var st = ShardedSafeTensors.open(path)
    var have = Dict[String, Int]()
    var key_count = 0
    var down_count = 0
    var up_count = 0
    var alpha_count = 0
    var bundle_key_count = 0

    for ref nm in st.names():
        have[nm] = 1
        key_count += 1
        if nm.endswith(".lora_down.weight"):
            var prefix = _strip_suffix(nm, String(".lora_down.weight"))
            if not _is_flux_lora_prefix(prefix):
                raise Error(String("real Flux LoRA down key has unexpected namespace: ") + nm)
            down_count += 1
        elif nm.endswith(".lora_up.weight"):
            var prefix = _strip_suffix(nm, String(".lora_up.weight"))
            if not _is_flux_lora_prefix(prefix):
                raise Error(String("real Flux LoRA up key has unexpected namespace: ") + nm)
            up_count += 1
        elif nm.endswith(".alpha"):
            var prefix = _strip_suffix(nm, String(".alpha"))
            if not _is_flux_lora_prefix(prefix):
                raise Error(String("real Flux LoRA alpha key has unexpected namespace: ") + nm)
            alpha_count += 1
        elif nm.startswith(String("bundle_emb.")):
            bundle_key_count += 1
        else:
            raise Error(String("real Flux LoRA file has unexpected non-LoRA key: ") + nm)

    _expect_int("real flux key count", key_count, FLUX_REAL_LORA_EXPECTED_KEYS)
    _expect_int("real flux down count", down_count, FLUX_REAL_LORA_EXPECTED_ADAPTERS)
    _expect_int("real flux up count", up_count, down_count)
    _expect_int("real flux alpha count", alpha_count, down_count)

    var rank = -1
    for ref nm in st.names():
        if not nm.endswith(".lora_down.weight"):
            continue

        var prefix = _strip_suffix(nm, String(".lora_down.weight"))
        var down_key = flux_lora_down_key(prefix)
        var up_key = flux_lora_up_key(prefix)
        var alpha_key = flux_lora_alpha_key(prefix)
        if not (up_key in have):
            raise Error(String("missing ") + up_key)
        if not (alpha_key in have):
            raise Error(String("missing ") + alpha_key)

        var down = st.tensor_view(down_key)
        var up = st.tensor_view(up_key)
        var alpha = st.tensor_view(alpha_key)
        _expect_dtype(down_key, down.dtype, expected_dtype)
        _expect_dtype(up_key, up.dtype, expected_dtype)
        _expect_dtype(alpha_key, alpha.dtype, expected_dtype)
        _expect_int(down_key + String(".shape_rank"), len(down.shape), 2)
        _expect_int(up_key + String(".shape_rank"), len(up.shape), 2)
        _expect_int(alpha_key + String(".shape_rank"), len(alpha.shape), 0)
        _expect_positive(down_key + String(".in_features"), down.shape[1])
        _expect_positive(up_key + String(".out_features"), up.shape[0])
        if rank < 0:
            rank = down.shape[0]
        _expect_int(down_key + String(".rank_dim"), down.shape[0], rank)
        _expect_int(up_key + String(".rank_dim"), up.shape[1], rank)

    _expect_int("real flux rank", rank, FLUX_REAL_LORA_EXPECTED_RANK)

    _check_representative_shape(
        st,
        flux_lora_legacy_prefixed_module(String("lora_transformer"), String("context_embedder")),
        16,
        4096,
        3072,
        16,
    )
    _check_representative_shape(
        st,
        flux_lora_legacy_prefixed_module(String("lora_transformer"), String("norm_out.linear")),
        16,
        3072,
        6144,
        16,
    )
    _check_representative_shape(
        st,
        flux_lora_legacy_prefixed_module(String("lora_transformer"), String("x_embedder")),
        16,
        64,
        3072,
        16,
    )
    _check_representative_shape(
        st,
        flux_lora_legacy_prefixed_module(String("lora_transformer"), String("single_transformer_blocks.0.proj_out")),
        16,
        15360,
        3072,
        16,
    )
    _check_representative_shape(
        st,
        flux_lora_legacy_prefixed_module(String("lora_transformer"), String("transformer_blocks.0.ff.net.0.proj")),
        16,
        3072,
        12288,
        16,
    )

    return FluxRealFileSummary(key_count, down_count, rank, bundle_key_count)


def main() raises:
    _check_conversion_metadata()

    var paths = flux_lora_candidate_files()
    var real_file = _find_real_file(paths)
    if real_file == String():
        raise Error("Flux real LoRA file missing from expected Serenity output path")

    var summary = _check_real_file(real_file, STDtype.BF16)
    print(
        "FLUX REAL LORA FILE PARITY PARTIAL OK: path=", real_file,
        " keys=", summary.key_count,
        " adapters=", summary.adapter_count,
        " rank=", summary.rank,
        " dtype=BF16 bundle_keys=", summary.bundle_key_count,
        " alpha_scalar=checked",
    )
