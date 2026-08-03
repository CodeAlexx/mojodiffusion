# SKEPTIC PROBE (not part of the port) — runs the REAL device loader
# (`minimax_h3_load_block_device`) against REAL MiniMax-H3 checkpoint bytes
# for the first time. No index.json has landed yet (see SKEPTIC_H3_LOADER.md
# update), so `ShardedSafeTensors.open(dir)` cannot open the checkpoint set —
# this probe builds a `ShardedSafeTensors` DIRECTLY via its public
# constructor over ONE already-complete real shard file
# (model-00001-of-00013.safetensors), which the header probe confirmed
# contains layers 0 and 1 complete (all 8 non-adaln tensors each). This does
# NOT modify anything under /home/alex/.serenity/... and does NOT claim
# `.open()` works — it is an explicit, documented bypass of the missing-index
# limitation, done ONLY to get a first real-bytes numeric check.
#
# What this checks, for real layer 0 (and layer 1 as a second data point):
#   1. `minimax_h3_load_block_device` runs end to end on real bytes and
#      preflight (`minimax_h3_check_block_weights`) passes.
#   2. The qkv de-interleave and fc1 swap the loader produced on the GPU,
#      read back host-side, match the GATED F32 ORACLE
#      (`models/minimax_h3/loader.mojo`) run on the SAME real raw bytes
#      (upcast to F32, oracle transform, cast back to BF16) -- the first
#      real-weight correctness check of these two transforms.
#   3. The six unrewritten tensors round-trip verbatim (spot-checked).
#
# Run: cd /home/alex/mojodiffusion && pixi run mojo run -I . \
#   serenitymojo/models/dit/parity/minimax_h3_real_block_device_probe.mojo

from std.collections import Dict, List
from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.models.dit.minimax_h3_dit import (
    MiniMaxH3DiTConfig,
    minimax_h3_released_config,
    minimax_h3_block_tensor_names,
    minimax_h3_block_prefix,
)
from serenitymojo.models.dit.minimax_h3_loader_device import (
    minimax_h3_load_block_device,
)
from serenitymojo.models.minimax_h3.loader import (
    minimax_h3_deinterleave_qkv,
    minimax_h3_swap_fc1,
)

comptime SHARD_PATH = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/transformer/model-00001-of-00013.safetensors"


def _open_single_shard_no_index(path: String) raises -> ShardedSafeTensors:
    """Manual bypass of the missing index.json: wraps ONE real shard file in
    a ShardedSafeTensors via the public constructor, mapping every tensor
    name it contains to shard index 0. Does not touch the checkpoint dir."""
    var st = SafeTensors.open(path)
    var names = st.names()
    var name_to_shard = Dict[String, Int]()
    for i in range(len(names)):
        name_to_shard[names[i]] = 0
    var shards = List[ArcPointer[SafeTensors]]()
    shards.append(ArcPointer(st^))
    return ShardedSafeTensors(shards^, name_to_shard^)


def _raw_bf16_to_f32(st: SafeTensors, name: String) raises -> List[Float32]:
    """Read a real BF16 tensor's raw bytes straight off the mmap and upcast
    to F32 (host-side, no GPU) -- the input format the f32 oracle expects."""
    var info = st.tensor_info(name)
    if info.dtype != STDtype.BF16:
        raise Error(String("expected BF16 for ") + name)
    var bytes = st.tensor_bytes(name)
    var n = info.size // 2
    var p = bytes.unsafe_ptr().bitcast[BFloat16]()
    var out = List[Float32](capacity=n)
    for i in range(n):
        out.append(p[i].cast[DType.float32]())
    return out^


def _max_abs_bf16(got: List[BFloat16], want_f32: List[Float32]) raises -> Float32:
    if len(got) != len(want_f32):
        raise Error(
            String("length mismatch: got ") + String(len(got)) + " want "
            + String(len(want_f32))
        )
    var worst = Float32(0.0)
    for i in range(len(got)):
        var g = got[i].cast[DType.float32]()
        var w = want_f32[i].cast[DType.bfloat16]().cast[DType.float32]()
        var d = g - w
        var ad = d if d >= Float32(0.0) else -d
        if ad > worst:
            worst = ad
    return worst


def _check_layer(st_raw: SafeTensors, sharded: ShardedSafeTensors, layer: Int, config: MiniMaxH3DiTConfig, ctx: DeviceContext) raises:
    print("== layer", layer, "==")
    var weights = minimax_h3_load_block_device(sharded, layer, config, ctx)
    print("  minimax_h3_load_block_device: OK,", len(weights), "tensors, preflight passed")

    var prefix = minimax_h3_block_prefix(layer)

    # --- qkv de-interleave vs f32 oracle, on REAL bytes ---
    var qkv_name = prefix + "attn.qkv_proj.weight"
    var raw_qkv_f32 = _raw_bf16_to_f32(st_raw, qkv_name)
    var heads = config.num_attention_heads
    var head_dim = config.attention_head_dim
    var hidden = config.hidden_size
    var want_qkv = minimax_h3_deinterleave_qkv(raw_qkv_f32, heads, head_dim, hidden)
    ref got_qkv_tensor = weights[qkv_name][]
    var got_qkv = got_qkv_tensor.to_host_bf16(ctx)
    var qkv_worst = _max_abs_bf16(got_qkv, want_qkv)
    print("  qkv de-interleave vs f32 oracle on REAL bytes: max_abs =", qkv_worst, " (", len(want_qkv), "values)")

    # --- fc1 swap vs f32 oracle, on REAL bytes ---
    var fc1_name = prefix + "mlp.fc1.weight"
    var raw_fc1_f32 = _raw_bf16_to_f32(st_raw, fc1_name)
    var ffn = config.ffn_hidden_size
    var want_fc1 = minimax_h3_swap_fc1(raw_fc1_f32, ffn, hidden)
    ref got_fc1_tensor = weights[fc1_name][]
    var got_fc1 = got_fc1_tensor.to_host_bf16(ctx)
    var fc1_worst = _max_abs_bf16(got_fc1, want_fc1)
    print("  fc1 swap vs f32 oracle on REAL bytes:         max_abs =", fc1_worst, " (", len(want_fc1), "values)")

    # --- spot-check one unrewritten tensor (norm1.weight) round-trips verbatim ---
    var norm1_name = prefix + "norm1.weight"
    var raw_norm1_f32 = _raw_bf16_to_f32(st_raw, norm1_name)
    var got_norm1 = weights[norm1_name][].to_host_bf16(ctx)
    var norm1_worst = Float32(0.0)
    for i in range(len(got_norm1)):
        var g = got_norm1[i].cast[DType.float32]()
        var w = raw_norm1_f32[i]
        var d = g - w
        var ad = d if d >= Float32(0.0) else -d
        if ad > norm1_worst:
            norm1_worst = ad
    print("  norm1.weight verbatim round-trip:             max_abs =", norm1_worst, " (", len(got_norm1), "values)")
    print("")


def main() raises:
    print("MiniMax-H3 REAL block device-loader probe")
    print("  shard:", SHARD_PATH)
    print("  (index.json absent -- ShardedSafeTensors built manually over this ONE shard)")
    print("")

    # Two independent handles to the SAME shard file: one wrapped for the
    # loader (moved into ShardedSafeTensors), one kept raw for the oracle
    # comparison reads. Both are separate mmaps of the same file -- cheap,
    # read-only, no interaction.
    var st_raw = SafeTensors.open(String(SHARD_PATH))
    var sharded = _open_single_shard_no_index(String(SHARD_PATH))

    var config = minimax_h3_released_config()
    config.validate()
    var ctx = DeviceContext()

    _check_layer(st_raw, sharded, 0, config, ctx)
    _check_layer(st_raw, sharded, 1, config, ctx)

    print("DONE")
