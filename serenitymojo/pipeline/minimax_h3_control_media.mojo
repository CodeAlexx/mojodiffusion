# Native media preparation for MiniMax-H3 Fun ControlNet-Union.
#
# SerenityFlow is the behavioral oracle only.  This module owns the equivalent
# work in the Mojo stack: ffmpeg decode/resize at H3's 24-fps clock, optional
# OpenCV-compatible 3x3 Canny semantics, H3 video-VAE posterior MEAN encoding,
# and 24/49-channel Union guide packing.  It never imports or launches Python.

from std.collections import List
from std.memory import ArcPointer, alloc
from std.math import abs, floor
from max.gpu.host import DeviceContext

from serenitymojo.components.artifacts import shell_quote
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import (
    BytePtr,
    O_RDONLY,
    file_size,
    sys_close,
    sys_open,
    sys_pread,
    sys_system,
)
from serenitymojo.tensor import Tensor
from serenitymojo.models.dit.minimax_h3_controlnet import (
    MiniMaxH3ControlInput,
)
from serenitymojo.models.minimax_h3.packing import (
    minimax_h3_video_latent_num_frames,
)
from serenitymojo.models.minimax_h3.rearrange import (
    minimax_h3_patchify_video,
)
from serenitymojo.models.vae.minimax_h3_ref_encode import (
    MINIMAX_H3_VIDEO_LATENT_CHANNELS,
    minimax_h3_encode_reference_visual_moments,
    minimax_h3_normalize_video_latents,
    minimax_h3_pixel_normalize_frames,
)
from serenitymojo.models.vae.minimax_h3_video_encoder_device import (
    MiniMaxH3VideoEncoderDevice,
    minimax_h3_video_released_encoder_config,
)
from serenitymojo.pipeline.minimax_h3_video_vae_spatial_tiling import (
    minimax_h3_video_released_tiling_config,
)


comptime MINIMAX_H3_CONTROL_FPS = 24


@fieldwise_init
struct MiniMaxH3ControlMediaSpec(Copyable, Movable):
    var path: String
    var preprocessor: String
    var resize_mode: String
    var canny_low: Int
    var canny_high: Int
    var strength: Float32
    var start_percent: Float32
    var end_percent: Float32
    var source_path: String
    var mask_path: String
    var invert_mask: Bool


def _control_read_bytes(path: String) raises -> List[UInt8]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("MiniMax-H3 ControlNet cannot open ") + path)
    var size = file_size(fd)
    if size < 0:
        _ = sys_close(fd)
        raise Error(String("MiniMax-H3 ControlNet cannot stat ") + path)
    var raw = alloc[UInt8](size)
    var ptr = BytePtr(unsafe_from_address=Int(raw))
    var done = 0
    while done < size:
        var got = sys_pread(fd, ptr + done, size - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    if done != size:
        raw.free()
        raise Error(String("MiniMax-H3 ControlNet short read from ") + path)
    var data = List[UInt8]()
    data.resize(size, UInt8(0))
    for i in range(size):
        data[i] = raw[i]
    raw.free()
    return data^


def _control_resize_filter(mode: String, width: Int, height: Int) raises -> String:
    if mode == String("stretch"):
        return (
            String("scale=") + String(width) + String(":") + String(height)
            + String(":flags=bilinear")
        )
    if mode == String("crop"):
        return (
            String("scale=") + String(width) + String(":") + String(height)
            + String(":force_original_aspect_ratio=increase:flags=bilinear,")
            + String("crop=") + String(width) + String(":") + String(height)
        )
    if mode == String("pad"):
        return (
            String("scale=") + String(width) + String(":") + String(height)
            + String(":force_original_aspect_ratio=decrease:flags=bilinear,")
            + String("pad=") + String(width) + String(":") + String(height)
            + String(":(ow-iw)/2:(oh-ih)/2:black")
        )
    raise Error("MiniMax-H3 ControlNet resize must be crop, pad, or stretch")


def _control_decode_media(
    path: String,
    raw_path: String,
    width: Int,
    height: Int,
    frames: Int,
    resize_mode: String,
) raises -> List[UInt8]:
    """Decode on the true H3 clock; short/static inputs hold their last frame."""
    var vf = (
        String("fps=") + String(MINIMAX_H3_CONTROL_FPS) + String(",")
        + _control_resize_filter(resize_mode, width, height)
        + String(",tpad=stop_mode=clone:stop_duration=3600")
    )
    var command = (
        String("ffmpeg -v error -y -i ") + shell_quote(path)
        + String(" -an -vf ") + shell_quote(vf)
        + String(" -frames:v ") + String(frames)
        + String(" -f rawvideo -pix_fmt rgb24 ") + shell_quote(raw_path)
    )
    if sys_system(command) != 0:
        raise Error(String("MiniMax-H3 ControlNet could not decode ") + path)
    var pixels = _control_read_bytes(raw_path)
    var expected = frames * height * width * 3
    if len(pixels) != expected:
        raise Error(
            String("MiniMax-H3 ControlNet decoded ") + String(len(pixels))
            + String(" bytes from ") + path + String(", expected ")
            + String(expected)
        )
    return pixels^


def _control_gray(rgb: List[UInt8], frames: Int, height: Int, width: Int) -> List[Int]:
    # OpenCV COLOR_RGB2GRAY's 14-bit fixed-point coefficients.
    var n = frames * height * width
    var gray = List[Int]()
    gray.resize(n, 0)
    for i in range(n):
        var at = 3 * i
        gray[i] = (
            4899 * Int(rgb[at]) + 9617 * Int(rgb[at + 1])
            + 1868 * Int(rgb[at + 2]) + 8192
        ) >> 14
    return gray^


def _control_clamp(v: Int, lo: Int, hi: Int) -> Int:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _control_mag_at(
    mag: List[Int], base: Int, height: Int, width: Int, y: Int, x: Int
) -> Int:
    # OpenCV surrounds the non-maximum-suppression magnitude ring with zeros.
    # This is distinct from the replicated source border used by Sobel itself.
    if y < 0 or y >= height or x < 0 or x >= width:
        return 0
    return mag[base + y * width + x]


def _control_canny(
    rgb: List[UInt8], frames: Int, height: Int, width: Int,
    low: Int, high: Int,
) raises -> List[UInt8]:
    """Native 3x3 Sobel, L1-NMS, double-threshold, 8-connected hysteresis.

    This is the algorithm selected by `cv2.Canny(gray, low, high)` with the
    default aperture=3 and L2gradient=false.  Frame boundaries never mix.
    """
    if low < 0 or high > 255 or low >= high:
        raise Error("MiniMax-H3 ControlNet Canny requires 0 <= low < high <= 255")
    var gray = _control_gray(rgb, frames, height, width)
    var plane = height * width
    var n = frames * plane
    var gx = List[Int](); gx.resize(n, 0)
    var gy = List[Int](); gy.resize(n, 0)
    var mag = List[Int](); mag.resize(n, 0)
    for f in range(frames):
        var base = f * plane
        for y in range(height):
            var ym = _control_clamp(y - 1, 0, height - 1)
            var yp = _control_clamp(y + 1, 0, height - 1)
            for x in range(width):
                var xm = _control_clamp(x - 1, 0, width - 1)
                var xp = _control_clamp(x + 1, 0, width - 1)
                var a = gray[base + ym * width + xm]
                var b = gray[base + ym * width + x]
                var c = gray[base + ym * width + xp]
                var d = gray[base + y * width + xm]
                var e = gray[base + y * width + xp]
                var g = gray[base + yp * width + xm]
                var h = gray[base + yp * width + x]
                var j = gray[base + yp * width + xp]
                var at = base + y * width + x
                gx[at] = -a + c - 2 * d + 2 * e - g + j
                gy[at] = -a - 2 * b - c + g + 2 * h + j
                mag[at] = abs(gx[at]) + abs(gy[at])

    # state: 0 suppressed, 1 weak, 2 strong. OpenCV Canny uses TG22=13573
    # on 15-bit axes and derives tan(67.5) as TG22 + 2.0 in that same scale.
    var state = List[UInt8](); state.resize(n, UInt8(0))
    var stack = List[Int]()
    for f in range(frames):
        var base = f * plane
        for y in range(height):
            for x in range(width):
                var at = base + y * width + x
                var ax = abs(gx[at]); var ay = abs(gy[at])
                var m0 = mag[at]
                if m0 <= low:
                    continue
                var m1 = 0; var m2 = 0
                var keep = False
                if ay * 32768 < ax * 13573:
                    m1 = _control_mag_at(mag, base, height, width, y, x - 1)
                    m2 = _control_mag_at(mag, base, height, width, y, x + 1)
                    keep = m0 > m1 and m0 >= m2
                elif ay * 32768 > ax * 79109:
                    m1 = _control_mag_at(mag, base, height, width, y - 1, x)
                    m2 = _control_mag_at(mag, base, height, width, y + 1, x)
                    keep = m0 > m1 and m0 >= m2
                elif gx[at] * gy[at] >= 0:
                    m1 = _control_mag_at(mag, base, height, width, y - 1, x - 1)
                    m2 = _control_mag_at(mag, base, height, width, y + 1, x + 1)
                    keep = m0 > m1 and m0 > m2
                else:
                    m1 = _control_mag_at(mag, base, height, width, y - 1, x + 1)
                    m2 = _control_mag_at(mag, base, height, width, y + 1, x - 1)
                    keep = m0 > m1 and m0 > m2
                if not keep:
                    continue
                if m0 > high:
                    state[at] = UInt8(2); stack.append(at)
                elif m0 > low:
                    state[at] = UInt8(1)
    while len(stack) > 0:
        var at = stack.pop()
        var local = at % plane
        var y = local // width; var x = local % width
        var base = (at // plane) * plane
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                var yy = y + dy; var xx = x + dx
                if yy < 0 or yy >= height or xx < 0 or xx >= width:
                    continue
                var other = base + yy * width + xx
                if state[other] == UInt8(1):
                    state[other] = UInt8(2); stack.append(other)
    var out = List[UInt8](); out.resize(3 * n, UInt8(0))
    for i in range(n):
        if state[i] == UInt8(2):
            out[3 * i] = UInt8(255)
            out[3 * i + 1] = UInt8(255)
            out[3 * i + 2] = UInt8(255)
    return out^


def _control_encode_mean(
    encoder: MiniMaxH3VideoEncoderDevice,
    rgb: List[UInt8], frames: Int, height: Int, width: Int,
    ctx: DeviceContext,
) raises -> List[Float32]:
    var pixels = minimax_h3_pixel_normalize_frames(rgb, frames, height, width)
    var moments = minimax_h3_encode_reference_visual_moments(
        encoder, pixels, 3, frames, height, width,
        minimax_h3_video_released_tiling_config(), ctx,
    )
    var latent_t = minimax_h3_video_latent_num_frames(frames)
    var latent_h = height // 16; var latent_w = width // 16
    var n = MINIMAX_H3_VIDEO_LATENT_CHANNELS * latent_t * latent_h * latent_w
    var mean = List[Float32](); mean.resize(n, Float32(0.0))
    for i in range(n):
        mean[i] = moments[i]
    # Fun ControlNet uses posterior.mode() and does not perform the Ref2VA
    # fp16 round trip before the released latent normalization.
    return minimax_h3_normalize_video_latents(
        mean, MINIMAX_H3_VIDEO_LATENT_CHANNELS,
        latent_t, latent_h, latent_w,
    )


def _control_visibility_latent(
    mask_rgb: List[UInt8], frames: Int, height: Int, width: Int,
    latent_t: Int, latent_h: Int, latent_w: Int, invert: Bool,
) -> List[Float32]:
    # Binary mask: source visibility = 1 - masked_region.  Trilinear resize
    # uses align_corners=False, matching torch interpolate in SerenityFlow.
    var source = List[Float32](); source.resize(frames * height * width, Float32(0.0))
    for f in range(frames):
        for y in range(height):
            for x in range(width):
                var at = (f * height + y) * width + x
                var rgb_at = 3 * at
                var masked = (
                    Int(mask_rgb[rgb_at]) + Int(mask_rgb[rgb_at + 1])
                    + Int(mask_rgb[rgb_at + 2])
                ) > 382
                if invert:
                    masked = not masked
                source[at] = Float32(0.0) if masked else Float32(1.0)
    var out = List[Float32](); out.resize(latent_t * latent_h * latent_w, Float32(0.0))
    for ot in range(latent_t):
        var ft = (Float32(ot) + Float32(0.5)) * Float32(frames) / Float32(latent_t) - Float32(0.5)
        var t0 = Int(floor(ft)); var wt = ft - Float32(t0)
        if t0 < 0: t0 = 0; wt = Float32(0.0)
        var t1 = t0 + 1
        if t1 >= frames: t1 = frames - 1
        for oy in range(latent_h):
            var fy = (Float32(oy) + Float32(0.5)) * Float32(height) / Float32(latent_h) - Float32(0.5)
            var y0 = Int(floor(fy)); var wy = fy - Float32(y0)
            if y0 < 0: y0 = 0; wy = Float32(0.0)
            var y1 = y0 + 1
            if y1 >= height: y1 = height - 1
            for ox in range(latent_w):
                var fx = (Float32(ox) + Float32(0.5)) * Float32(width) / Float32(latent_w) - Float32(0.5)
                var x0 = Int(floor(fx)); var wx = fx - Float32(x0)
                if x0 < 0: x0 = 0; wx = Float32(0.0)
                var x1 = x0 + 1
                if x1 >= width: x1 = width - 1
                var v000 = source[(t0 * height + y0) * width + x0]
                var v001 = source[(t0 * height + y0) * width + x1]
                var v010 = source[(t0 * height + y1) * width + x0]
                var v011 = source[(t0 * height + y1) * width + x1]
                var v100 = source[(t1 * height + y0) * width + x0]
                var v101 = source[(t1 * height + y0) * width + x1]
                var v110 = source[(t1 * height + y1) * width + x0]
                var v111 = source[(t1 * height + y1) * width + x1]
                var a0 = v000 + (v001 - v000) * wx
                var a1 = v010 + (v011 - v010) * wx
                var b0 = v100 + (v101 - v100) * wx
                var b1 = v110 + (v111 - v110) * wx
                var a = a0 + (a1 - a0) * wy
                var b = b0 + (b1 - b0) * wy
                out[(ot * latent_h + oy) * latent_w + ox] = a + (b - a) * wt
    return out^


def _control_mask_source(
    source_rgb: List[UInt8], visibility: List[Float32],
) -> List[UInt8]:
    var out = List[UInt8](); out.resize(len(source_rgb), UInt8(0))
    for i in range(len(visibility)):
        for c in range(3):
            out[3 * i + c] = UInt8(
                Int(Float32(Int(source_rgb[3 * i + c])) * visibility[i])
            )
    return out^


def minimax_h3_prepare_control_inputs(
    specs: List[MiniMaxH3ControlMediaSpec],
    video_vae_dir: String,
    out_dir: String,
    width: Int,
    height: Int,
    frames: Int,
    ctx: DeviceContext,
) raises -> List[MiniMaxH3ControlInput]:
    if len(specs) < 1 or len(specs) > 4:
        raise Error("MiniMax-H3 ControlNet requires one through four media inputs")
    if frames % 17 != 5 or width % 32 != 0 or height % 32 != 0:
        raise Error("MiniMax-H3 ControlNet media geometry is not H3-aligned")
    var encoder = MiniMaxH3VideoEncoderDevice.load(
        video_vae_dir, minimax_h3_video_released_encoder_config(), ctx
    )
    var latent_t = minimax_h3_video_latent_num_frames(frames)
    var latent_h = height // 16; var latent_w = width // 16
    var latent_volume = latent_t * latent_h * latent_w
    var output = List[MiniMaxH3ControlInput]()
    for index in range(len(specs)):
        ref spec = specs[index]
        var tag = out_dir + String("/control_") + String(index)
        var rgb = _control_decode_media(
            spec.path, tag + String(".rgb"), width, height, frames,
            spec.resize_mode,
        )
        if spec.preprocessor == String("canny"):
            rgb = _control_canny(
                rgb, frames, height, width, spec.canny_low, spec.canny_high
            )
        elif spec.preprocessor != String("prepared"):
            raise Error("MiniMax-H3 ControlNet preprocessor must be prepared or canny")
        var control_latents = _control_encode_mean(
            encoder, rgb, frames, height, width, ctx
        )
        var union = List[Float32]()
        union.resize(49 * latent_volume, Float32(0.0))
        for i in range(24 * latent_volume):
            union[i] = control_latents[i]
        var paired = (
            spec.source_path != String("") and spec.mask_path != String("")
        )
        if (spec.source_path == String("")) != (spec.mask_path == String("")):
            raise Error("MiniMax-H3 ControlNet source and mask must be supplied together")
        if paired:
            var source_rgb = _control_decode_media(
                spec.source_path, tag + String("_source.rgb"), width, height,
                frames, spec.resize_mode,
            )
            var mask_rgb = _control_decode_media(
                spec.mask_path, tag + String("_mask.rgb"), width, height,
                frames, spec.resize_mode,
            )
            var visibility = _control_visibility_latent(
                mask_rgb, frames, height, width, latent_t, latent_h, latent_w,
                spec.invert_mask,
            )
            # Mask at pixel resolution before encoding the inpaint source.
            var pixel_visibility = _control_visibility_latent(
                mask_rgb, frames, height, width, frames, height, width,
                spec.invert_mask,
            )
            var masked_source = _control_mask_source(source_rgb, pixel_visibility)
            var source_latents = _control_encode_mean(
                encoder, masked_source, frames, height, width, ctx
            )
            for i in range(latent_volume):
                union[24 * latent_volume + i] = visibility[i]
            for i in range(24 * latent_volume):
                union[25 * latent_volume + i] = source_latents[i]
        var packed = minimax_h3_patchify_video(
            union, 49, latent_t, latent_h, latent_w, 1, 2, 2
        )
        var rows = Tensor.from_host(
            packed, [latent_t * (latent_h // 2) * (latent_w // 2), 196],
            STDtype.F32, ctx,
        )
        output.append(
            MiniMaxH3ControlInput(
                ArcPointer[Tensor](rows^),
                spec.strength, spec.start_percent, spec.end_percent,
            )
        )
        _ = sys_system(
            String("rm -f ") + shell_quote(tag + String(".rgb"))
            + String(" ") + shell_quote(tag + String("_source.rgb"))
            + String(" ") + shell_quote(tag + String("_mask.rgb"))
        )
        ctx.synchronize()
    return output^
