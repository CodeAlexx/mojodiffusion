# Synthetic device smoke for the released MiniMax-H3 VideoVAE encoder loaders.
#
# This gate writes tiny one-element tensors under every expected native encoder
# key. It proves the existing loader preserves F16/BF16/F32 storage, the
# explicit cache-encode loader makes every resident tensor F32, and both paths
# reject an unsupported storage dtype. It does not execute the VAE or use the
# released 5.2 GB checkpoint.

from max.gpu.host import DeviceContext
from std.collections import List

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_remove
from serenitymojo.io.safetensors_writer import HostTensorDesc, save_safetensors_host
from serenitymojo.models.vae.minimax_h3_video_encoder_device import (
    MiniMaxH3VideoEncoderDevice,
    MiniMaxH3VideoEncoderDeviceConfig,
    minimax_h3_video_encoder_key_names,
)


comptime MIXED = "/tmp/minimax_h3_video_encoder_f32_loader_mixed.safetensors"
comptime INVALID = "/tmp/minimax_h3_video_encoder_f32_loader_invalid.safetensors"


def _check(condition: Bool, label: String) raises:
    if not condition:
        raise Error(String("MiniMax-H3 VideoVAE F32 loader smoke failed: ") + label)


def _config() -> MiniMaxH3VideoEncoderDeviceConfig:
    # Loader-only reduced topology. Tensor shapes are deliberately [1] because
    # this smoke exercises storage contracts, not encoder geometry or forward.
    return MiniMaxH3VideoEncoderDeviceConfig(
        4, [1, 2], [2, 1], [1, 1], 1, 2, 3, 2, Float32(1.0e-6),
    )


def _supported_dtype(index: Int) -> STDtype:
    var lane = index % 3
    if lane == 0:
        return STDtype.F16
    if lane == 1:
        return STDtype.BF16
    return STDtype.F32


def _one_bytes(dtype: STDtype) raises -> List[UInt8]:
    if dtype == STDtype.F16:
        return [UInt8(0x00), UInt8(0x3C)]
    if dtype == STDtype.BF16:
        return [UInt8(0x80), UInt8(0x3F)]
    if dtype == STDtype.F32:
        return [UInt8(0x00), UInt8(0x00), UInt8(0x80), UInt8(0x3F)]
    if dtype == STDtype.I8:
        return [UInt8(1)]
    raise Error("loader smoke byte fixture has an unsupported dtype")


def _write_checkpoint(
    path: String, names: List[String], invalid_first: Bool,
) raises:
    var descs = List[HostTensorDesc]()
    for i in range(len(names)):
        var dtype = _supported_dtype(i)
        if invalid_first and i == 0:
            dtype = STDtype.I8
        descs.append(HostTensorDesc(dtype, [1], _one_bytes(dtype)))
    save_safetensors_host(names, descs^, path)


def _check_default_dtypes(
    encoder: MiniMaxH3VideoEncoderDevice, names: List[String],
) raises:
    for i in range(len(names)):
        var name = names[i]
        var got = encoder.weights[encoder.name_to_idx[name]][].dtype()
        _check(got == _supported_dtype(i), String("default dtype for ") + name)


def _check_f32_dtypes_and_values(
    encoder: MiniMaxH3VideoEncoderDevice, names: List[String],
    ctx: DeviceContext,
) raises:
    for name in names:
        var index = encoder.name_to_idx[name]
        _check(
            encoder.weights[index][].dtype() == STDtype.F32,
            String("F32 dtype for ") + name,
        )
        var value = encoder.weights[index][].to_host(ctx)
        _check(len(value) == 1 and value[0] == Float32(1.0), String("F32 value for ") + name)


def _check_invalid_rejections(
    config: MiniMaxH3VideoEncoderDeviceConfig, ctx: DeviceContext,
) raises:
    var default_rejected = False
    try:
        _ = MiniMaxH3VideoEncoderDevice.load(String(INVALID), config, ctx)
    except error:
        default_rejected = True
        _check(
            String(error).find(String("unsupported storage dtype I8")) >= 0,
            String("default unsupported-dtype diagnostic"),
        )
    _check(default_rejected, String("default loader rejects I8"))

    var f32_rejected = False
    try:
        _ = MiniMaxH3VideoEncoderDevice.load_f32_compute(
            String(INVALID), config, ctx,
        )
    except error:
        f32_rejected = True
        _check(
            String(error).find(String("unsupported storage dtype I8")) >= 0,
            String("F32 unsupported-dtype diagnostic"),
        )
    _check(f32_rejected, String("F32 loader rejects I8"))


def main() raises:
    var config = _config()
    var names = minimax_h3_video_encoder_key_names(config)
    _write_checkpoint(String(MIXED), names, False)
    _write_checkpoint(String(INVALID), names, True)

    var ctx = DeviceContext()
    var storage_encoder = MiniMaxH3VideoEncoderDevice.load(
        String(MIXED), config, ctx,
    )
    _check_default_dtypes(storage_encoder, names)

    var f32_encoder = MiniMaxH3VideoEncoderDevice.load_f32_compute(
        String(MIXED), config, ctx,
    )
    _check_f32_dtypes_and_values(f32_encoder, names, ctx)
    _check_invalid_rejections(config, ctx)

    _ = sys_remove(String(MIXED))
    _ = sys_remove(String(INVALID))
    print("PASS MiniMax-H3 VideoVAE F32 loader synthetic device smoke")
    print("  default loader: F16/BF16/F32 storage preserved")
    print("  F32 compute loader: every resident weight/bias is F32")
    print("  unsupported I8 storage: rejected before weight upload")
    print("  released 5.2 GB VideoVAE forward/cache generation: NOT TESTED")
