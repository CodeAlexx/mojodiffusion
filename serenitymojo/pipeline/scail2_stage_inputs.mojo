# Pure-Mojo SCAIL-2 image/video input staging. FFmpeg is used only as the media
# decoder; all resize, normalization, mask compression, tensor layout, atomic
# safetensors publication, and provenance sidecars are owned by Mojo.
#
# Usage:
#   scail2_stage_inputs <image> <image-mask> <pose-video> <driving-mask-video>
#                       <output.safetensors> <height> <width> [max-frames]
#                       [additional-image additional-mask]...

from std.collections import List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.sys import argv

from serenitymojo.components.artifacts import shell_quote
from serenitymojo.image.decode import decode_image
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_system
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.models.scail2.scail2_manifest import (
    scail2_path_exists,
    write_scail2_stage_manifest,
)
from serenitymojo.models.scail2.scail2_preprocess import (
    scail2_binary7_area_from_signed_chw,
    scail2_binary7_area_from_unit_chw,
    scail2_clip_from_reference_chw,
    scail2_half_bilinear_u8_to_signed_chw,
    scail2_half_bilinear_u8_to_unit_chw,
    scail2_hwc_f32_to_chw,
    scail2_pack_temporal_mask28,
    scail2_resize_center_f32,
    scail2_resize_center_u8,
)
from serenitymojo.tensor import Tensor


def _parse_positive_int(value: String, label: String) raises -> Int:
    if value.byte_length() == 0:
        raise Error(String("SCAIL-2 empty integer: ") + label)
    var out = 0
    for byte in value.as_bytes():
        if byte < UInt8(ord("0")) or byte > UInt8(ord("9")):
            raise Error(String("SCAIL-2 invalid integer: ") + label)
        out = out * 10 + Int(byte - UInt8(ord("0")))
    if out <= 0:
        raise Error(String("SCAIL-2 integer must be positive: ") + label)
    return out


def _six_digits(index: Int) -> String:
    var value = String(index)
    while value.byte_length() < 6:
        value = String("0") + value
    return value^


def _frame_path(root: String, kind: String, index: Int) -> String:
    return (
        root + String("/") + kind + String("/frame_")
        + _six_digits(index) + String(".png")
    )


def _prepare_frame_root(root: String) raises:
    # Delete only decoder-owned frame_*.png files beneath the exact derived
    # staging root. Other files are neither matched nor removed.
    var command = String("mkdir -p -- ") + shell_quote(root + String("/pose"))
    command += String(" ") + shell_quote(root + String("/mask"))
    command += String(" && find ") + shell_quote(root)
    command += String(" -mindepth 2 -maxdepth 2 -type f -name 'frame_*.png' -delete")
    if sys_system(command) != 0:
        raise Error(String("SCAIL-2 cannot prepare frame staging root: ") + root)


def _extract_frames(video: String, pattern: String) raises:
    var command = String("ffmpeg -hide_banner -loglevel error -y -i ")
    command += shell_quote(video)
    command += String(" -vsync 0 -start_number 0 ") + shell_quote(pattern)
    if sys_system(command) != 0:
        raise Error(String("SCAIL-2 ffmpeg frame decode failed: ") + video)


def _frame_count(root: String, maximum: Int) raises -> Int:
    var count = 0
    while True:
        if maximum > 0 and count >= maximum:
            break
        var pose = _frame_path(root, String("pose"), count)
        var mask = _frame_path(root, String("mask"), count)
        var has_pose = scail2_path_exists(pose)
        var has_mask = scail2_path_exists(mask)
        if has_pose != has_mask:
            raise Error(String("SCAIL-2 pose/mask frame mismatch at index ") + String(count))
        if not has_pose:
            break
        count += 1
    if count < 1 or (count - 1) % 4 != 0:
        raise Error(
            String("SCAIL-2 staged segment must have 4n+1 frames, got ")
            + String(count)
        )
    # Detect mismatches beyond an explicit truncation only when the full input
    # was requested; max-frames intentionally ignores later frames.
    if maximum <= 0:
        var next_pose = scail2_path_exists(_frame_path(root, String("pose"), count))
        var next_mask = scail2_path_exists(_frame_path(root, String("mask"), count))
        if next_pose != next_mask:
            raise Error("SCAIL-2 pose/mask decoded frame counts differ")
    return count


def _cleanup_frames(root: String) raises:
    var command = String("find ") + shell_quote(root)
    command += String(" -mindepth 2 -maxdepth 2 -type f -name 'frame_*.png' -delete")
    command += String(" && rmdir -- ") + shell_quote(root + String("/pose"))
    command += String(" ") + shell_quote(root + String("/mask"))
    command += String(" ") + shell_quote(root)
    if sys_system(command) != 0:
        raise Error(String("SCAIL-2 cannot clean frame staging root: ") + root)


def _i32_scalar(value: Int, ctx: DeviceContext) raises -> Tensor:
    var host = ctx.enqueue_create_host_buffer[DType.uint8](4)
    host.unsafe_ptr().bitcast[Int32]()[0] = Int32(value)
    var device = ctx.enqueue_create_buffer[DType.uint8](4)
    ctx.enqueue_copy(dst_buf=device, src_buf=host)
    ctx.synchronize()
    var shape: List[Int] = [1]
    return Tensor(device^, shape^, STDtype.I32)


def main() raises:
    var args = argv()
    if len(args) < 8 or len(args) > 15 or (len(args) >= 10 and (len(args) - 9) % 2 != 0):
        raise Error(
            "usage: scail2_stage_inputs <image> <image-mask> <pose-video> "
            "<driving-mask-video> <output.safetensors> <height> <width> "
            "[max-frames] [additional-image additional-mask]... (up to 3)"
        )
    var image_path = String(args[1])
    var image_mask_path = String(args[2])
    var pose_path = String(args[3])
    var driving_mask_path = String(args[4])
    var output_path = String(args[5])
    var height = _parse_positive_int(String(args[6]), String("height"))
    var width = _parse_positive_int(String(args[7]), String("width"))
    var maximum = 0
    if len(args) >= 9:
        maximum = _parse_positive_int(String(args[8]), String("max-frames"))
    var additional_images = List[String]()
    var additional_masks = List[String]()
    var extra = 9
    while extra + 1 < len(args):
        additional_images.append(String(args[extra]))
        additional_masks.append(String(args[extra + 1]))
        extra += 2
    if len(additional_images) > 3:
        raise Error("SCAIL-2 supports at most 3 additional references per run")
    if height % 16 != 0 or width % 16 != 0:
        raise Error("SCAIL-2 height and width must be multiples of 16")
    if (
        output_path == image_path or output_path == image_mask_path
        or output_path == pose_path or output_path == driving_mask_path
    ):
        raise Error("SCAIL-2 stage output must not overwrite an input")
    for path in additional_images:
        if output_path == path:
            raise Error("SCAIL-2 stage output must not overwrite an additional image")
    for path in additional_masks:
        if output_path == path:
            raise Error("SCAIL-2 stage output must not overwrite an additional mask")

    var mkdir_command = String("mkdir -p -- \"$(dirname -- ")
    mkdir_command += shell_quote(output_path) + String(")\"")
    if sys_system(mkdir_command) != 0:
        raise Error(String("SCAIL-2 cannot create output parent: ") + output_path)
    var frame_root = output_path + String(".scail2_frames.tmp")
    _prepare_frame_root(frame_root)
    _extract_frames(pose_path, frame_root + String("/pose/frame_%06d.png"))
    _extract_frames(driving_mask_path, frame_root + String("/mask/frame_%06d.png"))
    var frames = _frame_count(frame_root, maximum)
    var latent_frames = (frames - 1) // 4 + 1
    var half_height = height // 2
    var half_width = width // 2
    var latent_height = height // 16
    var latent_width = width // 16

    var reference_hwc = scail2_resize_center_f32(
        decode_image(image_path, drop_alpha=True), height, width
    )
    var reference = scail2_hwc_f32_to_chw(reference_hwc, height, width)
    var reference_mask_hwc = scail2_resize_center_f32(
        decode_image(image_mask_path, drop_alpha=True), height, width
    )
    var reference_mask = scail2_hwc_f32_to_chw(
        reference_mask_hwc, height, width
    )
    var reference_binary = scail2_binary7_area_from_unit_chw(
        reference_mask, height, width, 8
    )
    var reference_mask28 = scail2_pack_temporal_mask28(
        reference_binary, 1, height // 8, width // 8
    )
    var ref_mask_plane = (height // 8) * (width // 8)
    var ref_masks = List[Float32]()
    ref_masks.resize(
        28 * (1 + latent_frames) * ref_mask_plane, Float32(0.0)
    )
    for q in range(28):
        for i in range(ref_mask_plane):
            ref_masks[(q * (1 + latent_frames)) * ref_mask_plane + i] = (
                reference_mask28[q * ref_mask_plane + i]
            )

    var pose_pixels = List[Float32]()
    pose_pixels.resize(
        3 * frames * half_height * half_width, Float32(0.0)
    )
    var binary_by_frame = List[Float32]()
    binary_by_frame.resize(
        frames * 7 * latent_height * latent_width, Float32(0.0)
    )
    var half_plane = half_height * half_width
    var latent_plane = latent_height * latent_width
    for frame in range(frames):
        var pose_u8 = scail2_resize_center_u8(
            decode_image(_frame_path(frame_root, String("pose"), frame), drop_alpha=True),
            height, width,
        )
        var pose_unit = scail2_half_bilinear_u8_to_unit_chw(
            pose_u8, height, width
        )
        for c in range(3):
            for i in range(half_plane):
                pose_pixels[(c * frames + frame) * half_plane + i] = (
                    pose_unit[c * half_plane + i]
                )

        var mask_u8 = scail2_resize_center_u8(
            decode_image(_frame_path(frame_root, String("mask"), frame), drop_alpha=True),
            height, width,
        )
        var mask_signed = scail2_half_bilinear_u8_to_signed_chw(
            mask_u8, height, width
        )
        var binary = scail2_binary7_area_from_signed_chw(
            mask_signed, half_height, half_width, 8
        )
        for q in range(7):
            for i in range(latent_plane):
                binary_by_frame[(frame * 7 + q) * latent_plane + i] = (
                    binary[q * latent_plane + i]
                )
        if frame == 0 or frame + 1 == frames or (frame + 1) % 16 == 0:
            print("[scail2-stage] processed frame", frame + 1, "/", frames)

    var driving_masks = scail2_pack_temporal_mask28(
        binary_by_frame, frames, latent_height, latent_width
    )
    var clip_pixel = scail2_clip_from_reference_chw(
        reference, height, width
    )

    var additional_reference = List[Float32]()
    var additional_masks28 = List[Float32]()
    var additional_count = len(additional_images)
    var full_plane = height * width
    additional_reference.resize(
        3 * additional_count * full_plane, Float32(0.0)
    )
    additional_masks28.resize(
        28 * additional_count * ref_mask_plane, Float32(0.0)
    )
    for ai in range(additional_count):
        var image_hwc = scail2_resize_center_f32(
            decode_image(additional_images[ai], drop_alpha=True), height, width
        )
        var image_chw = scail2_hwc_f32_to_chw(image_hwc, height, width)
        for c in range(3):
            for i in range(full_plane):
                additional_reference[(c * additional_count + ai) * full_plane + i] = (
                    image_chw[c * full_plane + i]
                )
        var mask_hwc = scail2_resize_center_f32(
            decode_image(additional_masks[ai], drop_alpha=True), height, width
        )
        var mask_chw = scail2_hwc_f32_to_chw(mask_hwc, height, width)
        var mask_binary = scail2_binary7_area_from_unit_chw(
            mask_chw, height, width, 8
        )
        # Creator parity: each additional image is an independent one-frame
        # reference. Compress its mask independently, then concatenate the
        # resulting [28,1,H/8,W/8] tensors along the temporal/reference axis.
        var packed_mask = scail2_pack_temporal_mask28(
            mask_binary, 1, height // 8, width // 8
        )
        for q in range(28):
            for i in range(ref_mask_plane):
                additional_masks28[(q * additional_count + ai) * ref_mask_plane + i] = (
                    packed_mask[q * ref_mask_plane + i]
                )

    var ctx = DeviceContext()
    var reference_tensor = Tensor.from_host(
        reference, [1, 3, 1, height, width], STDtype.F32, ctx
    )
    var pose_tensor = Tensor.from_host(
        pose_pixels, [1, 3, frames, half_height, half_width], STDtype.F32, ctx
    )
    var ref_masks_tensor = Tensor.from_host(
        ref_masks, [28, 1 + latent_frames, height // 8, width // 8],
        STDtype.F32, ctx,
    )
    var driving_masks_tensor = Tensor.from_host(
        driving_masks, [28, latent_frames, latent_height, latent_width],
        STDtype.F32, ctx,
    )
    var clip_tensor = Tensor.from_host(
        clip_pixel, [1, 3, 224, 224], STDtype.F32, ctx
    )
    var frames_tensor = _i32_scalar(frames, ctx)
    var additional_count_tensor = _i32_scalar(additional_count, ctx)
    var names: List[String] = [
        String("reference_pixel"), String("pose_pixel"), String("ref_masks"),
        String("driving_masks"), String("clip_pixel"), String("frames"),
    ]
    var tensors = List[ArcPointer[Tensor]]()
    tensors.append(ArcPointer(reference_tensor^))
    tensors.append(ArcPointer(pose_tensor^))
    tensors.append(ArcPointer(ref_masks_tensor^))
    tensors.append(ArcPointer(driving_masks_tensor^))
    tensors.append(ArcPointer(clip_tensor^))
    tensors.append(ArcPointer(frames_tensor^))
    names.append(String("additional_ref_count"))
    tensors.append(ArcPointer(additional_count_tensor^))
    if additional_count > 0:
        var additional_reference_tensor = Tensor.from_host(
            additional_reference,
            [1, 3, additional_count, height, width], STDtype.F32, ctx,
        )
        var additional_masks_tensor = Tensor.from_host(
            additional_masks28,
            [28, additional_count, height // 8, width // 8], STDtype.F32, ctx,
        )
        names.append(String("additional_reference_pixel"))
        tensors.append(ArcPointer(additional_reference_tensor^))
        names.append(String("additional_ref_masks"))
        tensors.append(ArcPointer(additional_masks_tensor^))
    save_safetensors(names, tensors, output_path, ctx)
    _ = write_scail2_stage_manifest(
        String(args[0]), image_path, image_mask_path, pose_path,
        driving_mask_path, output_path, height, width, frames,
        additional_images, additional_masks,
    )
    _cleanup_frames(frame_root)
    print(
        "[scail2-stage] GATE frames=", frames,
        " reference=[1,3,1,", height, ",", width,
        "] pose=[1,3,", frames, ",", half_height, ",", half_width,
        "] output=", output_path,
    )
