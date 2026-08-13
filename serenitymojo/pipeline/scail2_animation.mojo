# SCAIL-2 14B single-segment animation denoise stage, pure Mojo.
#
# Exact creator defaults for generate.py: 896x512, 65 input frames, UniPC,
# shift=3, CFG=5, seed=42. This process exits after writing the normalized Wan
# latent; `scail2_decode.mojo` loads the VAE in a fresh process.
#
# argv:
#   scail2_animation <stage.safetensors> <reference_latent.safetensors>
#       <pose_latent.safetensors> <text_context.safetensors>
#       <clip_context.safetensors> <fp8_cache_dir> <out_latent.safetensors>
#       [steps=40] [seed=42] [mode=animation] [additional-reference-latent|-]

from std.collections import List
from std.ffi import external_call
from max.gpu.host import DeviceContext
from std.memory import ArcPointer, UnsafePointer, alloc
from std.sys import argv

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.scail2.scail2_streamed_dit import (
    Scail2PredictionPair,
    Scail2StreamedDiT,
)
from serenitymojo.models.scail2.scail2_manifest import (
    preflight_scail2_additional_manifest,
    preflight_scail2_animation_manifests,
    write_scail2_animation_manifest,
)
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random_torch import randn_torch
from serenitymojo.ops.tensor_algebra import (
    add,
    concat,
    full_device,
    mul_scalar,
    reshape,
    sub,
)
from serenitymojo.sampling.scail2_unipc import (
    build_scail2_unipc_sigma_schedule,
    build_scail2_unipc_timesteps,
)
from serenitymojo.sampling.unipc import UniPcMultistepScheduler
from serenitymojo.tensor import Tensor


comptime HEIGHT = 512
comptime WIDTH = 896
comptime FRAMES = 65
comptime FT = 17
comptime H_LAT = 64
comptime W_LAT = 112
comptime GH = 32
comptime GW = 56
comptime S = 39872
comptime S_AR1 = 41664
comptime S_AR2 = 43456
comptime S_AR3 = 45248
comptime TXT = 512
comptime CTXL = 512
comptime IMG = 257
comptime HEADS = 40
comptime HEAD_DIM = 128
comptime CHANNELS = 16
comptime CFG = Float32(5.0)
comptime SHIFT = Float64(3.0)
comptime _CStr = UnsafePointer[UInt8, MutExternalOrigin]


def _cstr(s: String) -> _CStr:
    var n = s.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = s.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    return _CStr(unsafe_from_address=Int(buf))


def _enable_bounded_device_allocations():
    # The default async pool retains the cumulative 39,872-token transient
    # peak and OOMs on a 16 GB RTX 5080.  Synchronous device allocation frees
    # each streamed-block temporary back to the driver.  This must be set
    # before the first DeviceContext is constructed.
    _ = external_call["setenv", Int32](
        _cstr(String("MODULAR_DEVICE_CONTEXT_SYNC_MODE")),
        _cstr(String("true")),
        Int32(1),
    )


def _require(st: ShardedSafeTensors, key: String) raises:
    if key not in st.names():
        raise Error(String("SCAIL-2 input missing tensor '") + key + String("'"))


def _shape_matches(actual: List[Int], expected: List[Int]) -> Bool:
    if len(actual) != len(expected):
        return False
    for i in range(len(actual)):
        if actual[i] != expected[i]:
            return False
    return True


def _expect_input_tensor(
    st: ShardedSafeTensors,
    key: String,
    expected_shape: List[Int],
    expected_dtype: STDtype,
) raises:
    """Validate source metadata before device loading or shape adaptation."""
    _require(st, key)
    var tv = st.tensor_view(key)
    if tv.dtype != expected_dtype:
        raise Error(
            String("SCAIL-2 tensor dtype mismatch: ") + key
            + String(" got=") + tv.dtype.name()
            + String(" expected=") + expected_dtype.name()
        )
    if not _shape_matches(tv.shape, expected_shape):
        raise Error(String("SCAIL-2 tensor rank/shape mismatch: ") + key)


def validate_scail2_animation_input(
    path: String,
    key: String,
    expected_shape: List[Int],
    expected_dtype: STDtype,
) raises:
    """Public metadata-only gate used by negative loader parity tests."""
    var st = ShardedSafeTensors.open(path)
    _expect_input_tensor(st, key, expected_shape, expected_dtype)


def _load_rank4(
    path: String,
    key: String,
    source_shape: List[Int],
    output_shape: List[Int],
    source_dtype: STDtype,
    output_dtype: STDtype,
    ctx: DeviceContext,
) raises -> Tensor:
    var st = ShardedSafeTensors.open(path)
    _expect_input_tensor(st, key, source_shape, source_dtype)
    var value = Tensor.from_view(st.tensor_view(key), ctx)
    value = cast_tensor(value, output_dtype, ctx)
    return reshape(value, output_shape.copy(), ctx)


def _load_text(
    path: String, key: String, ctx: DeviceContext,
) raises -> Tensor:
    var st = ShardedSafeTensors.open(path)
    _expect_input_tensor(st, key, [1, TXT, 4096], STDtype.BF16)
    var value = Tensor.from_view(st.tensor_view(key), ctx)
    return reshape(value, [TXT, 4096], ctx)


def _load_clip(path: String, ctx: DeviceContext) raises -> Tensor:
    var st = ShardedSafeTensors.open(path)
    var key = String("clip_context")
    _expect_input_tensor(st, key, [1, IMG, 1280], STDtype.F16)
    var value = Tensor.from_view(st.tensor_view(key), ctx)
    return value^


def _channels20(
    latent16: Tensor, marker: Float32, t: Int, h: Int, w: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var marker4 = full_device([4, t, h, w], marker, STDtype.F32, ctx)
    return concat(0, ctx, latent16, marker4)


def main() raises:
    var args = argv()
    if len(args) < 8 or len(args) > 12:
        raise Error(
            "usage: scail2_animation <stage> <reference_latent> <pose_latent> "
            "<text_context> <clip_context> <fp8_cache_dir> <out_latent> "
            "[steps=40] [seed=42] [mode=animation|replacement] "
            "[additional-reference-latent|-]"
        )
    var stage_path = String(args[1])
    var reference_path = String(args[2])
    var pose_path = String(args[3])
    var text_path = String(args[4])
    var clip_path = String(args[5])
    var cache_dir = String(args[6])
    var output_path = String(args[7])
    var steps = 40
    if len(args) >= 9:
        steps = atol(String(args[8]))
    var seed = UInt64(42)
    if len(args) >= 10:
        seed = UInt64(atol(String(args[9])))
    var mode = String("animation")
    if len(args) >= 11:
        mode = String(args[10])
    if mode != String("animation") and mode != String("replacement"):
        raise Error("SCAIL-2 mode must be animation or replacement")
    var replace_flag = mode == String("replacement")
    var additional_path = String("-")
    if len(args) >= 12:
        additional_path = String(args[11])
    if steps < 1:
        raise Error("SCAIL-2 steps must be >= 1")

    preflight_scail2_animation_manifests(
        stage_path, reference_path, pose_path, text_path, clip_path, cache_dir,
    )
    var additional_count = 0
    if additional_path != String("-"):
        preflight_scail2_additional_manifest(additional_path)
        var additional_meta = ShardedSafeTensors.open(additional_path)
        var additional_key = String("additional_reference_latent")
        _require(additional_meta, additional_key)
        var additional_shape = additional_meta.tensor_view(additional_key).shape.copy()
        if (
            len(additional_shape) != 5 or additional_shape[0] != 1
            or additional_shape[1] != CHANNELS or additional_shape[2] < 1
            or additional_shape[2] > 3 or additional_shape[3] != H_LAT
            or additional_shape[4] != W_LAT
        ):
            raise Error("SCAIL-2 additional reference latent shape mismatch")
        additional_count = additional_shape[2]
    _enable_bounded_device_allocations()
    var ctx = DeviceContext()
    var reference16 = _load_rank4(
        reference_path, String("reference_latent"),
        [1, CHANNELS, 1, H_LAT, W_LAT],
        [CHANNELS, 1, H_LAT, W_LAT],
        STDtype.F32, STDtype.F32, ctx,
    )
    var pose16 = _load_rank4(
        pose_path, String("pose_latent"),
        [1, CHANNELS, FT, H_LAT // 2, W_LAT // 2],
        [CHANNELS, FT, H_LAT // 2, W_LAT // 2],
        STDtype.F32, STDtype.F32, ctx,
    )
    var reference20 = _channels20(
        reference16^, 1.0, 1, H_LAT, W_LAT, ctx
    )
    var pose20 = _channels20(
        pose16^, 1.0, FT, H_LAT // 2, W_LAT // 2, ctx
    )
    var ref_masks = _load_rank4(
        stage_path, String("ref_masks"),
        [28, FT + 1, H_LAT, W_LAT], [28, FT + 1, H_LAT, W_LAT],
        STDtype.F32, STDtype.F32, ctx,
    )
    var driving_masks = _load_rank4(
        stage_path, String("driving_masks"),
        [28, FT, H_LAT // 2, W_LAT // 2],
        [28, FT, H_LAT // 2, W_LAT // 2],
        STDtype.F32, STDtype.F32, ctx,
    )
    var positive = _load_text(text_path, String("pos_embed"), ctx)
    var negative = _load_text(text_path, String("neg_embed"), ctx)
    var clip = _load_clip(clip_path, ctx)
    # One-element sentinels satisfy definite initialization; additional-count
    # dispatch never passes them to a compiled multi-reference branch at zero.
    var additional20 = full_device([1], 0.0, STDtype.F32, ctx)
    var additional_masks = full_device([1], 0.0, STDtype.F32, ctx)
    if additional_count > 0:
        var additional16 = _load_rank4(
            additional_path, String("additional_reference_latent"),
            [1, CHANNELS, additional_count, H_LAT, W_LAT],
            [CHANNELS, additional_count, H_LAT, W_LAT],
            STDtype.F32, STDtype.F32, ctx,
        )
        additional20 = _channels20(
            additional16^, 1.0, additional_count, H_LAT, W_LAT, ctx
        )
        additional_masks = _load_rank4(
            stage_path, String("additional_ref_masks"),
            [28, additional_count, H_LAT, W_LAT],
            [28, additional_count, H_LAT, W_LAT],
            STDtype.F32, STDtype.F32, ctx,
        )

    var sigmas = build_scail2_unipc_sigma_schedule(steps, SHIFT, 1000)
    var timesteps = build_scail2_unipc_timesteps(steps, SHIFT, 1000)
    var scheduler = UniPcMultistepScheduler.from_sigmas(sigmas.copy(), 2)
    var x = randn_torch([CHANNELS, FT, H_LAT, W_LAT], seed, ctx)
    var model = Scail2StreamedDiT.open(cache_dir, ctx)
    var sequence_tokens = S
    if additional_count == 1:
        sequence_tokens = S_AR1
    elif additional_count == 2:
        sequence_tokens = S_AR2
    elif additional_count == 3:
        sequence_tokens = S_AR3
    print(
        "=== SCAIL-2 animation === geometry=896x512x65 steps=", steps,
        " seed=", seed, " mode=", mode, " shift=3 cfg=5 tokens=",
        sequence_tokens,
    )
    for step in range(steps):
        var video20 = _channels20(x.clone(ctx), 0.0, FT, H_LAT, W_LAT, ctx)
        var pair: Scail2PredictionPair
        if additional_count == 0:
            pair = model.forward_cfg_pair[
                FT, GH, GW, S, TXT, CTXL, IMG, HEADS, HEAD_DIM,
            ](
                video20^, reference20, pose20, ref_masks, driving_masks,
                timesteps[step], positive, negative, clip,
                TXT, TXT, replace_flag, ctx,
            )
        elif additional_count == 1:
            pair = model.forward_cfg_pair_with_additional[
                FT, GH, GW, 1, S_AR1, TXT, CTXL, IMG, HEADS, HEAD_DIM,
            ](
                video20^, reference20, pose20, ref_masks, driving_masks,
                additional20, additional_masks, timesteps[step], positive,
                negative, clip, TXT, TXT, replace_flag, ctx,
            )
        elif additional_count == 2:
            pair = model.forward_cfg_pair_with_additional[
                FT, GH, GW, 2, S_AR2, TXT, CTXL, IMG, HEADS, HEAD_DIM,
            ](
                video20^, reference20, pose20, ref_masks, driving_masks,
                additional20, additional_masks, timesteps[step], positive,
                negative, clip, TXT, TXT, replace_flag, ctx,
            )
        else:
            pair = model.forward_cfg_pair_with_additional[
                FT, GH, GW, 3, S_AR3, TXT, CTXL, IMG, HEADS, HEAD_DIM,
            ](
                video20^, reference20, pose20, ref_masks, driving_masks,
                additional20, additional_masks, timesteps[step], positive,
                negative, clip, TXT, TXT, replace_flag, ctx,
            )
        var cond = cast_tensor(pair.cond, STDtype.F32, ctx)
        var uncond = cast_tensor(pair.uncond, STDtype.F32, ctx)
        var guided = add(
            uncond, mul_scalar(sub(cond, uncond, ctx), CFG, ctx), ctx
        )
        x = scheduler.step(guided, x, ctx)
        print(
            "  step", step + 1, "/", steps,
            " sigma=", sigmas[step], " t=", timesteps[step],
        )

    var names = List[String]()
    names.append(String("latent"))
    var tensors = List[ArcPointer[Tensor]]()
    tensors.append(ArcPointer(x^))
    save_safetensors(names, tensors, output_path, ctx)
    var run_manifest = write_scail2_animation_manifest(
        String(args[0]), stage_path, reference_path, pose_path, text_path,
        clip_path, cache_dir, output_path, steps, seed, mode,
        additional_path,
    )
    print("GATE SCAIL-2 denoise complete:", output_path)
    print("GATE SCAIL-2 run manifest:", run_manifest)
