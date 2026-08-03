# SKEPTIC PROBE (not part of the port) — HEADER-ONLY inspection of the REAL
# MiniMax-H3 transformer shards that have landed on disk. Opens each shard
# via chunk-1 `SafeTensors.open()` directly (mmap + 8-byte length + JSON
# header parse only — the ~4.9 GB data segment per shard is mapped, not
# read), and cross-checks every block-related tensor's shape/dtype against
# the contract in minimax_h3_dit.mojo. Does NOT go through
# `ShardedSafeTensors.open()` (no index.json exists yet — see report) and
# does NOT load any tensor bytes onto the GPU.

from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.minimax_h3_dit import (
    minimax_h3_released_config,
    minimax_h3_block_tensor_names,
    minimax_h3_expected_shape,
)

comptime SHARD_DIR = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/transformer/"


def _shape_str(shape: List[Int]) -> String:
    var s = String("[")
    for i in range(len(shape)):
        if i > 0:
            s += ", "
        s += String(shape[i])
    s += "]"
    return s^


def _inspect_shard(path: String) raises:
    print("== ", path, " ==")
    var st = SafeTensors.open(path)
    print("  tensor count:", st.count())
    var names = st.names_storage_order()

    # bucket names by "kind": block tensors (blocks.N.*), refiner
    # (token_refiner.*), everything else (patch proj / time embed / final
    # layer / condition proj).
    var block_prefixes = Dict[Int, Bool]()  # layer index -> seen
    var other_count = 0
    var refiner_count = 0
    var adaln_count = 0

    for i in range(len(names)):
        var n = names[i]
        if n.startswith("blocks."):
            # parse the layer index out of "blocks.<N>.<rest>" via split(".")
            var parts = n.split(".")
            if len(parts) >= 2:
                try:
                    var layer = Int(parts[1])
                    block_prefixes[layer] = True
                except:
                    pass
            if n.find("adaln_proj") >= 0:
                adaln_count += 1
        elif n.startswith("token_refiner."):
            refiner_count += 1
        else:
            other_count += 1

    var layers = List[Int]()
    for ref e in block_prefixes.items():
        layers.append(e.key)
    # simple insertion sort (small N)
    for i in range(1, len(layers)):
        var v = layers[i]
        var j = i - 1
        while j >= 0 and layers[j] > v:
            layers[j + 1] = layers[j]
            j -= 1
        layers[j + 1] = v

    print("  distinct block layers present:", len(layers))
    var layer_list = String("  layers: [")
    for i in range(len(layers)):
        if i > 0:
            layer_list += ", "
        layer_list += String(layers[i])
    layer_list += "]"
    print(layer_list)
    print("  token_refiner tensors:", refiner_count)
    print("  adaln_proj tensors (blocks.*):", adaln_count)
    print("  other (patch/time/final/condition) tensors:", other_count)

    # For each present layer, check whether ALL 8 non-adaln block tensor
    # names (minimax_h3_block_tensor_names) are present IN THIS SHARD.
    var config = minimax_h3_released_config()
    print("")
    print("  per-layer completeness (all 8 non-adaln tensors in THIS shard?):")
    for i in range(len(layers)):
        var layer = layers[i]
        var required = minimax_h3_block_tensor_names(layer)
        var have = 0
        var missing = List[String]()
        for j in range(len(required)):
            if st.has_tensor(required[j]):
                have += 1
            else:
                missing.append(required[j])
        var complete = have == len(required)
        print(
            "    layer", layer, ": ", have, "/", len(required),
            "complete=", complete,
        )
        if not complete:
            var miss_str = String("      missing: ")
            for k in range(len(missing)):
                if k > 0:
                    miss_str += ", "
                miss_str += missing[k]
            print(miss_str)

    # Shape/dtype contract check for every block tensor present in this shard.
    print("")
    print("  shape/dtype contract check (all block tensors in this shard):")
    var checked = 0
    var mismatches = 0
    for i in range(len(names)):
        var n = names[i]
        if not n.startswith("blocks."):
            continue
        if n.find("adaln_proj") >= 0:
            continue  # not covered by minimax_h3_expected_shape (by design)
        var info = st.tensor_info(n)
        var want_shape: List[Int]
        try:
            want_shape = minimax_h3_expected_shape(n, config)
        except:
            continue  # e.g. norm1/norm2/q_norm/k_norm handled; skip anything else
        checked += 1
        var shape_ok = len(info.shape) == len(want_shape)
        if shape_ok:
            for d in range(len(want_shape)):
                if info.shape[d] != want_shape[d]:
                    shape_ok = False
                    break
        var dtype_ok = info.dtype == STDtype.BF16
        if not shape_ok or not dtype_ok:
            mismatches += 1
            print(
                "    MISMATCH", n, " got_shape=", _shape_str(info.shape),
                " want_shape=", _shape_str(want_shape),
                " dtype=", info.dtype.name(),
            )
    print("  checked", checked, "block tensors against minimax_h3_expected_shape, mismatches:", mismatches)
    print("")


def main() raises:
    print("MiniMax-H3 REAL shard header probe (header-only, no data load)")
    print("")
    _inspect_shard(SHARD_DIR + "model-00001-of-00013.safetensors")
    _inspect_shard(SHARD_DIR + "model-00006-of-00013.safetensors")
    print("DONE")
