# No-CUDA guard for the revision-bound Wan 2.2 Diffusers -> Mojo artifact view.

from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype


comptime ROOT = (
    "/home/alex/.serenity/models/checkpoints/Wan2.2-TI2V-5B-Mojo"
)


def _check_tensor(
    ref st: ShardedSafeTensors,
    name: String,
    dtype: STDtype,
    expected_numel: Int,
) raises:
    var info = st.tensor_info(name)
    if info.dtype != dtype:
        raise Error(
            name + String(": dtype ") + info.dtype.name()
            + String(" != ") + dtype.name()
        )
    var n = 1
    for d in info.shape:
        n *= d
    if n != expected_numel:
        raise Error(
            name + String(": numel ") + String(n)
            + String(" != ") + String(expected_numel)
        )
    # Touch the mmap span, proving index -> shard resolution, not just key text.
    var view = st.tensor_view(name)
    _ = view.data[0]


def main() raises:
    var dit = ShardedSafeTensors.open(String(ROOT))
    if dit.num_shards() != 5 or dit.num_tensors() != 825:
        raise Error(
            String("Wan DiT inventory mismatch: shards=")
            + String(dit.num_shards()) + String(" tensors=")
            + String(dit.num_tensors())
        )
    _check_tensor(
        dit, String("blocks.0.attn1.to_q.weight"), STDtype.F32,
        3072 * 3072,
    )
    _check_tensor(
        dit, String("blocks.29.ffn.net.2.weight"), STDtype.F32,
        3072 * 14336,
    )
    _check_tensor(
        dit, String("scale_shift_table"), STDtype.F32, 2 * 3072,
    )

    var umt5 = ShardedSafeTensors.open(String(ROOT) + String("/umt5"))
    if umt5.num_shards() != 3 or umt5.num_tensors() != 242:
        raise Error(
            String("Wan UMT5 inventory mismatch: shards=")
            + String(umt5.num_shards()) + String(" tensors=")
            + String(umt5.num_tensors())
        )
    _check_tensor(
        umt5, String("shared.weight"), STDtype.BF16,
        256384 * 4096,
    )
    _check_tensor(
        umt5, String("encoder.block.23.layer.1.DenseReluDense.wo.weight"), STDtype.BF16,
        4096 * 10240,
    )
    _check_tensor(
        umt5, String("encoder.block.17.layer.0.SelfAttention.relative_attention_bias.weight"),
        STDtype.BF16, 32 * 64,
    )
    print("wan22_artifact_view_smoke: PASS (5/825 DiT, 3/242 UMT5)")
