# Exact-output dumper for EVG's request-static ragged 2-D partition.
# The companion Python gate imports EVG's own module and compares these lines.

from std.sys import argv

from serenitymojo.ops.evg_ragged_attention import (
    EVGH3SparsePolicy,
    evg_route_counts,
    make_evg_ragged_2d_partition,
)


def _parse_positive_int(text: String) raises -> Int:
    var bytes = text.as_bytes()
    if len(bytes) == 0:
        raise Error("expected a positive integer")
    var value = 0
    for i in range(len(bytes)):
        if bytes[i] < UInt8(48) or bytes[i] > UInt8(57):
            raise Error("expected a positive integer")
        value = value * 10 + Int(bytes[i] - UInt8(48))
    if value <= 0:
        raise Error("expected a positive integer")
    return value


def main() raises:
    var args = argv()
    if len(args) != 4:
        raise Error("usage: evg_ragged_partition_dump HEIGHT WIDTH CAPACITY")
    var height = _parse_positive_int(String(args[1]))
    var width = _parse_positive_int(String(args[2]))
    var capacity = _parse_positive_int(String(args[3]))
    var partition = make_evg_ragged_2d_partition(height, width, capacity)
    print("shape|", height, "|", width, "|", capacity)
    print("block_count|", partition.block_count())
    for block_id in range(partition.block_count()):
        var line = String("block|") + String(block_id)
        for token_index in range(len(partition.blocks[block_id])):
            line += String("|") + String(partition.blocks[block_id][token_index])
        print(line)
    var mapping = String("mapping")
    for token in range(len(partition.token_to_block)):
        mapping += String("|") + String(partition.token_to_block[token])
    print(mapping)
    var adjacency = String("adjacency")
    for index in range(len(partition.adjacency)):
        adjacency += String("|") + (String("1") if partition.adjacency[index] else String("0"))
    print(adjacency)

    # The released policy/count arithmetic is included in the same CPU gate.
    var policy = EVGH3SparsePolicy(0.88, 0.80, 0.20, 10, 2, 50)
    policy.validate()
    var counts = evg_route_counts(
        partition.block_count() * partition.block_count(),
        policy.sparsity_ratio,
        policy.retained_int8_ratio,
        policy.retained_fp16_ratio,
    )
    print("route_counts|", counts.fp16, "|", counts.int8)
    var schedule = String("schedule")
    for layer in range(policy.layers_per_step):
        schedule += String("|") + String(policy.layer_sparsity(layer))
    print(schedule)
