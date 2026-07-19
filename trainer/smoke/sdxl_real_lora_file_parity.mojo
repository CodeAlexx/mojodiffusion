# Real SDXL LoRA file key/shape/dtype gate.
#
# Serenity reference file:
#   /home/alex/Serenity/output/sdxl_100step_baseline/lora_last.safetensors
#
# Serenity source:
#   modules/modelSetup/StableDiffusionXLLoRASetup.py
#   modules/modelSaver/stableDiffusionXL/StableDiffusionXLLoRASaver.py
#   modules/modelSaver/mixin/LoRASaverMixin.py
#   modules/util/convert/lora/convert_sdxl_lora.py
#   modules/module/LoRAModule.py
#
# Gate:
#   load the real Serenity-saved SDXL output LoRA file with converted keys,
#   verify
#   the file-level inventory count (794 UNet adapters, 2382 tensors), and check
#   a representative target slice for key names, Linear/Conv2d shapes, BF16
#   storage dtype, rank 16, and alpha 16.0. This is key/shape/dtype parity only;
#   it does not claim SDXL UNet forward/backward parity or text-encoder LoRA
#   coverage for this file.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.tensor import Tensor

from serenity_trainer.modelSetup.stableDiffusionXLLoraTargets import (
    SDXL_REAL_OMI_UNET_KEY_COUNT,
    SDXL_REAL_OMI_UNET_RANK,
    SDXL_REAL_OMI_UNET_TARGET_COUNT,
    StableDiffusionXLLoraTargetSpecs,
    sdxl_lora_alpha_key,
    sdxl_lora_down_key,
    sdxl_lora_up_key,
    sdxl_real_omi_unet_alpha,
    sdxl_real_omi_unet_lora_file,
    sdxl_real_omi_unet_representative_target_specs,
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


def _expect_shape_2d(name: String, sh: List[Int], rows: Int, cols: Int) raises:
    _expect_int(name + String(".rank"), len(sh), 2)
    _expect_int(name + String(".rows"), sh[0], rows)
    _expect_int(name + String(".cols"), sh[1], cols)


def _expect_shape_4d(
    name: String,
    sh: List[Int],
    dim0: Int,
    dim1: Int,
    dim2: Int,
    dim3: Int,
) raises:
    _expect_int(name + String(".rank"), len(sh), 4)
    _expect_int(name + String(".dim0"), sh[0], dim0)
    _expect_int(name + String(".dim1"), sh[1], dim1)
    _expect_int(name + String(".dim2"), sh[2], dim2)
    _expect_int(name + String(".dim3"), sh[3], dim3)


def _expect_dtype(name: String, tensor: Tensor, expected: STDtype) raises:
    if tensor.dtype() != expected:
        raise Error(
            name + String(": dtype got ") + tensor.dtype().name()
            + String(", expected ") + expected.name()
        )


def _check_lora_shape_pair(
    down_key: String,
    up_key: String,
    down: Tensor,
    up: Tensor,
    in_features: Int,
    out_features: Int,
    kernel_h: Int,
    kernel_w: Int,
) raises:
    var dsh = down.shape()
    var ush = up.shape()
    if kernel_h == 0 and kernel_w == 0:
        _expect_shape_2d(down_key, dsh, SDXL_REAL_OMI_UNET_RANK, in_features)
        _expect_shape_2d(up_key, ush, out_features, SDXL_REAL_OMI_UNET_RANK)
    else:
        _expect_shape_4d(
            down_key,
            dsh,
            SDXL_REAL_OMI_UNET_RANK,
            in_features,
            kernel_h,
            kernel_w,
        )
        _expect_shape_4d(
            up_key,
            ush,
            out_features,
            SDXL_REAL_OMI_UNET_RANK,
            1,
            1,
        )


def _load_and_check_targets(
    path: String,
    targets: StableDiffusionXLLoraTargetSpecs,
    ctx: DeviceContext,
    expected_dtype: STDtype,
) raises -> Int:
    var st = ShardedSafeTensors.open(path)
    var have = Dict[String, Int]()
    var key_count = 0
    var down_count = 0
    var up_count = 0
    var alpha_count = 0

    for ref nm in st.names():
        have[nm] = 1
        key_count += 1
        if nm.endswith(".lora_down.weight"):
            down_count += 1
        elif nm.endswith(".lora_up.weight"):
            up_count += 1
        elif nm.endswith(".alpha"):
            alpha_count += 1
        if not nm.startswith("lora_unet_"):
            raise Error(String("real SDXL output LoRA key is not converted UNet namespace: ") + nm)

    _expect_int("real sdxl lora key count", key_count, SDXL_REAL_OMI_UNET_KEY_COUNT)
    _expect_int("real sdxl lora down count", down_count, SDXL_REAL_OMI_UNET_TARGET_COUNT)
    _expect_int("real sdxl lora up count", up_count, SDXL_REAL_OMI_UNET_TARGET_COUNT)
    _expect_int("real sdxl lora alpha count", alpha_count, SDXL_REAL_OMI_UNET_TARGET_COUNT)

    for i in range(targets.len()):
        var prefix = targets.prefixes[i]
        var down_key = sdxl_lora_down_key(prefix)
        var up_key = sdxl_lora_up_key(prefix)
        var alpha_key = sdxl_lora_alpha_key(prefix)
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
        _check_lora_shape_pair(
            down_key,
            up_key,
            down,
            up,
            targets.in_features[i],
            targets.out_features[i],
            targets.kernel_h[i],
            targets.kernel_w[i],
        )

        var ash = alpha.shape()
        _expect_int(alpha_key + String(".rank"), len(ash), 0)
        var host = alpha.to_host(ctx)
        if len(host) > 0:
            _expect_float(alpha_key, host[0], sdxl_real_omi_unet_alpha())

    return key_count


def main() raises:
    var ctx = DeviceContext()
    var path = sdxl_real_omi_unet_lora_file()
    var targets = sdxl_real_omi_unet_representative_target_specs()
    _expect_int("representative target count", targets.len(), 17)

    var key_count = _load_and_check_targets(path, targets, ctx, STDtype.BF16)

    print("SDXL REAL LORA FILE PARITY OK: keys=", key_count, " adapters=794 rank=16 dtype=BF16 alpha=16.0 checked_targets=17")
