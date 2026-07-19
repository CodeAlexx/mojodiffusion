# Chroma LoRA key construction and real-file blocker gate.
#
# Serenity source:
#   modules/modelSetup/ChromaLoRASetup.py
#   modules/modelSaver/chroma/ChromaLoRASaver.py
#   modules/util/convert/lora/convert_chroma_lora.py
#   modules/module/LoRAModule.py
#
# Gate:
#   If a real local Serenity Chroma LoRA exists at one of the candidate
#   paths, inspect file-level LoRA key suffix counts and dtype/rank/alpha shape
#   basics. If no real file exists, report the missing paths and still verify
#   the bounded Serenity-derived raw/OMI/legacy key construction. This is not
#   Chroma transformer forward/backward parity.

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.registry.checkpoints import path_exists

from serenity_trainer.modelSetup.chromaLoraTargets import (
    ChromaLoraTargetSpecs,
    chroma_lora_alpha_key,
    chroma_lora_candidate_files,
    chroma_lora_down_key,
    chroma_lora_up_key,
    chroma_representative_lora_target_specs,
)


struct ChromaRealFileSummary(Copyable, Movable):
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


def _is_chroma_lora_prefix(prefix: String) -> Bool:
    return (
        prefix.startswith(String("lora_transformer."))
        or prefix.startswith(String("lora_transformer_"))
        or prefix.startswith(String("lora_te."))
        or prefix.startswith(String("lora_te_"))
        or prefix.startswith(String("transformer."))
        or prefix.startswith(String("t5."))
        or prefix.startswith(String("transformer_"))
        or prefix.startswith(String("t5_"))
    )


def _check_target_key_construction(targets: ChromaLoraTargetSpecs) raises:
    _expect_int("representative chroma target count", targets.len(), 23)
    for i in range(targets.len()):
        var raw_prefix = targets.raw_prefixes[i]
        var omi_prefix = targets.omi_prefixes[i]
        var legacy_prefix = targets.legacy_prefixes[i]
        _expect_prefix(String("raw prefix ") + String(i), raw_prefix, String("lora_"))
        if raw_prefix.startswith("lora_transformer."):
            _expect_prefix(String("omi transformer prefix ") + String(i), omi_prefix, String("transformer."))
            _expect_prefix(String("legacy transformer prefix ") + String(i), legacy_prefix, String("transformer_"))
        elif raw_prefix.startswith("lora_te."):
            _expect_prefix(String("omi t5 prefix ") + String(i), omi_prefix, String("t5."))
            _expect_prefix(String("legacy t5 prefix ") + String(i), legacy_prefix, String("t5_"))
        else:
            raise Error(String("unexpected Chroma raw prefix ") + raw_prefix)

        _expect_suffix(String("down key ") + String(i), chroma_lora_down_key(raw_prefix), String(".lora_down.weight"))
        _expect_suffix(String("up key ") + String(i), chroma_lora_up_key(raw_prefix), String(".lora_up.weight"))
        _expect_suffix(String("alpha key ") + String(i), chroma_lora_alpha_key(raw_prefix), String(".alpha"))
        _expect_suffix(String("omi down key ") + String(i), chroma_lora_down_key(omi_prefix), String(".lora_down.weight"))
        _expect_suffix(String("legacy down key ") + String(i), chroma_lora_down_key(legacy_prefix), String(".lora_down.weight"))


def _find_real_file(paths: List[String]) raises -> String:
    for i in range(len(paths)):
        if path_exists(paths[i]):
            return paths[i].copy()
    return String()


def _check_real_file(path: String, expected_dtype: STDtype) raises -> ChromaRealFileSummary:
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
            if not _is_chroma_lora_prefix(prefix):
                raise Error(String("real Chroma LoRA down key has unexpected namespace: ") + nm)
            down_count += 1
        elif nm.endswith(".lora_up.weight"):
            var prefix = _strip_suffix(nm, String(".lora_up.weight"))
            if not _is_chroma_lora_prefix(prefix):
                raise Error(String("real Chroma LoRA up key has unexpected namespace: ") + nm)
            up_count += 1
        elif nm.endswith(".alpha"):
            var prefix = _strip_suffix(nm, String(".alpha"))
            if not _is_chroma_lora_prefix(prefix):
                raise Error(String("real Chroma LoRA alpha key has unexpected namespace: ") + nm)
            alpha_count += 1
        elif nm.startswith(String("bundle_emb.")):
            bundle_key_count += 1
        else:
            raise Error(String("real Chroma LoRA file has unexpected non-LoRA key: ") + nm)

    if key_count == 0:
        raise Error(String("real Chroma LoRA file has no tensors: ") + path)
    if down_count == 0:
        raise Error(String("real Chroma LoRA file has no lora_down weights: ") + path)
    _expect_int("real chroma up count", up_count, down_count)
    _expect_int("real chroma alpha count", alpha_count, down_count)

    var rank = -1
    for ref nm in st.names():
        if not nm.endswith(".lora_down.weight"):
            continue

        var prefix = _strip_suffix(nm, String(".lora_down.weight"))
        var down_key = chroma_lora_down_key(prefix)
        var up_key = chroma_lora_up_key(prefix)
        var alpha_key = chroma_lora_alpha_key(prefix)
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
            _expect_positive("real chroma rank", rank)
        _expect_int(down_key + String(".rank_dim"), down.shape[0], rank)
        _expect_int(up_key + String(".rank_dim"), up.shape[1], rank)

    return ChromaRealFileSummary(key_count, down_count, rank, bundle_key_count)


def main() raises:
    var targets = chroma_representative_lora_target_specs()
    _check_target_key_construction(targets)

    var paths = chroma_lora_candidate_files()
    var real_file = _find_real_file(paths)
    if real_file == String():
        print("CHROMA REAL LORA FILE MISSING")
        print("checked target/key construction: representative_targets=", targets.len())
        for i in range(len(paths)):
            print("missing candidate: ", paths[i])
        return

    var summary = _check_real_file(real_file, STDtype.BF16)
    print(
        "CHROMA REAL LORA FILE PARITY PARTIAL OK: path=", real_file,
        " keys=", summary.key_count,
        " adapters=", summary.adapter_count,
        " rank=", summary.rank,
        " dtype=BF16 bundle_keys=", summary.bundle_key_count,
        " checked_targets=", targets.len(),
    )
