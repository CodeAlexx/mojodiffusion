# Real Ernie LoRA file key/shape/dtype gate.
#
# Serenity reference file:
#   /home/alex/Serenity/output/ernie_eri2_100step_baseline/lora.safetensors
#
# Serenity source:
#   modules/modelSetup/ErnieLoRASetup.py
#   modules/modelSaver/ernie/ErnieLoRASaver.py
#
# Gate:
#   load the real Serenity-saved LoRA file with raw keys, verify the full
#   transformer inventory used by the baseline: 36 layers * 7 targets = 252
#   adapters, 756 keys, BF16 tensors, rank 16, alpha 1.0. This is key/shape/dtype
#   parity only; it does not claim Ernie transformer forward/backward parity.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.tensor import Tensor

from serenity_trainer.modelSetup.ernieLoraTargets import (
    ErnieLoraTargetSpecs,
    ernie_lora_alpha_key,
    ernie_lora_down_key,
    ernie_lora_up_key,
    ernie_transformer_attn_mlp_target_specs,
)


def _expect_int(name: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(
            name + String(": got ") + String(got)
            + String(", expected ") + String(expected)
        )


def _expect_float(name: String, got: Float32, expected: Float32) raises:
    var diff = got - expected
    if diff < Float32(0.0):
        diff = -diff
    if diff > Float32(0.001):
        raise Error(
            name + String(": got ") + String(got)
            + String(", expected ") + String(expected)
        )


def _expect_shape(name: String, sh: List[Int], rows: Int, cols: Int) raises:
    _expect_int(name + String(".rank"), len(sh), 2)
    _expect_int(name + String(".rows"), sh[0], rows)
    _expect_int(name + String(".cols"), sh[1], cols)


def _expect_dtype(name: String, tensor: Tensor, expected: STDtype) raises:
    if tensor.dtype() != expected:
        raise Error(
            name + String(": dtype got ") + tensor.dtype().name()
            + String(", expected ") + expected.name()
        )


def _load_and_check_targets(
    path: String,
    targets: ErnieLoraTargetSpecs,
    ctx: DeviceContext,
    expected_dtype: STDtype,
) raises -> Int:
    var st = ShardedSafeTensors.open(path)
    var have = Dict[String, Int]()
    var key_count = 0
    for ref nm in st.names():
        have[nm] = 1
        key_count += 1

    _expect_int("real ernie lora key count", key_count, 756)

    var rank = -1
    for i in range(targets.len()):
        var prefix = targets.prefixes[i]
        var down_key = ernie_lora_down_key(prefix)
        var up_key = ernie_lora_up_key(prefix)
        var alpha_key = ernie_lora_alpha_key(prefix)
        if not (down_key in have):
            raise Error(String("missing ") + down_key)
        if not (up_key in have):
            raise Error(String("missing ") + up_key)
        if not (alpha_key in have):
            raise Error(String("missing ") + alpha_key)

        var down = Tensor.from_view(st.tensor_view(down_key), ctx)
        var up = Tensor.from_view(st.tensor_view(up_key), ctx)
        var alpha = Tensor.from_view(st.tensor_view(alpha_key), ctx)
        _expect_dtype(down_key, down, expected_dtype)
        _expect_dtype(up_key, up, expected_dtype)
        _expect_dtype(alpha_key, alpha, expected_dtype)

        var dsh = down.shape()
        var ush = up.shape()
        _expect_shape(down_key, dsh, 16, targets.in_features[i])
        _expect_shape(up_key, ush, targets.out_features[i], 16)
        if rank < 0:
            rank = dsh[0]
        _expect_int(down_key + String(".rank_dim"), dsh[0], rank)
        _expect_int(up_key + String(".rank_dim"), ush[1], rank)

        var ash = alpha.shape()
        _expect_int(alpha_key + String(".rank"), len(ash), 0)
        var host = alpha.to_host(ctx)
        if len(host) > 0:
            _expect_float(alpha_key, host[0], Float32(1.0))

    _expect_int("rank", rank, 16)
    return key_count


def main() raises:
    var ctx = DeviceContext()
    var path = String("/home/alex/Serenity/output/ernie_eri2_100step_baseline/lora.safetensors")
    var targets = ernie_transformer_attn_mlp_target_specs(36, 4096, 12288)
    _expect_int("target count", targets.len(), 252)

    var key_count = _load_and_check_targets(path, targets, ctx, STDtype.BF16)

    print("ERNIE REAL LORA FILE PARITY OK: keys=", key_count, " adapters=252 rank=16 dtype=BF16 alpha=1.0")
