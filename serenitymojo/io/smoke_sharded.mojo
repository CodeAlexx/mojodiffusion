# smoke_sharded.mojo — opens the Z-Image *transformer* directory (2 shards +
# index) via ShardedSafeTensors and prints total tensor count + 3 sample
# tensors, including at least one resident in shard 00002 (proves cross-shard
# resolution through the weight_map).
#
# Run: pixi run mojo run -I . serenitymojo/io/smoke_sharded.mojo
#
# Snapshot dir resolved 2026-05-25 via:
#   ls ~/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/snapshots/*/transformer/
# (one snapshot) -> hardcoded below. Expect 521 tensors (weight_map size):
# 423 in shard 00001, 98 in shard 00002.

from serenitymojo.io.sharded import ShardedSafeTensors


comptime TRANSFORMER_DIR = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/transformer"
)


def _print_tensor(ref st: ShardedSafeTensors, name: String) raises:
    var info = st.tensor_info(name)
    var view = st.tensor_view(name)
    var shape_str = String("[")
    for i in range(len(info.shape)):
        if i > 0:
            shape_str += ", "
        shape_str += String(info.shape[i])
    shape_str += "]"
    # Dereference the origin-bound view (proves the mmap'd pages are readable
    # AND that the byte span resolves to the correct owning shard).
    var first_byte = Int(view.data[0])
    print(
        "  ",
        name,
        "| shard=",
        st.shard_index(name),
        "| dtype=",
        view.dtype.name(),
        "| shape=",
        shape_str,
        "| nbytes=",
        view.nbytes(),
        "| first_byte=",
        first_byte,
    )


def main() raises:
    var st = ShardedSafeTensors.open(String(TRANSFORMER_DIR))
    print("Z-Image transformer (sharded) opened.")
    print("num_shards:", st.num_shards())
    print("total tensor count:", st.num_tensors())

    print("sample tensors:")
    # shard 00001 tensors:
    _print_tensor(st, String("cap_embedder.0.weight"))
    _print_tensor(st, String("t_embedder.mlp.0.weight"))
    # shard 00002 tensor (proves cross-shard resolution):
    _print_tensor(st, String("layers.23.adaLN_modulation.0.bias"))
