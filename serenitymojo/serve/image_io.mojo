# serenitymojo.serve.image_io — init-image decode helpers (plan P7, img2img).
#
# Pure-Mojo decode of a user-supplied init/mask image via MOJO-libs image
# (png / jpeg / webp), shared by:
#   * serve/zimage_backend.mojo — decode + resize + [-1,1] NCHW for the
#     Z-Image VAE encoder (the real img2img path), and
#   * the SerenityUI params column — decode + resize + RGBA8 for the
#     init-image thumbnail texture.
#
# Format detection is MAGIC-BYTE based (extension only as a tiebreaker is
# unnecessary): PNG \x89PNG, JPEG \xFF\xD8, WebP RIFF....WEBP.

from std.io.file import open
from std.math import exp, floor

from image.buffer import Image
from image.jpeg import decode_jpeg_bytes
from image.png import decode_png_bytes
from image.webp import decode_webp_bytes


def _read_file_bytes(path: String) raises -> List[UInt8]:
    var f = open(path, String("r"))
    var d = f.read_bytes()
    f.close()
    var data = List[UInt8](capacity=len(d))
    for i in range(len(d)):
        data.append(d[i])
    return data^


def decode_image_any(path: String) raises -> Image:
    """Decode a PNG / JPEG / WebP file (magic-byte sniffed). Raises with a
    clear message for missing files and unsupported formats."""
    var data: List[UInt8]
    try:
        data = _read_file_bytes(path)
    except:
        raise Error(String("init image not readable: ") + path)
    if len(data) < 12:
        raise Error(String("init image too small / empty: ") + path)
    if data[0] == 0x89 and data[1] == 0x50 and data[2] == 0x4E and data[3] == 0x47:
        return decode_png_bytes(data)
    if data[0] == 0xFF and data[1] == 0xD8:
        return decode_jpeg_bytes(data)
    if (
        data[0] == 0x52 and data[1] == 0x49 and data[2] == 0x46 and data[3] == 0x46
        and data[8] == 0x57 and data[9] == 0x45 and data[10] == 0x42 and data[11] == 0x50
    ):
        return decode_webp_bytes(data)
    raise Error(
        String("init image format not supported (need png/jpeg/webp): ") + path
    )


def image_rgb_at(img: Image, x: Int, y: Int, mut rgb: List[Int]) raises:
    """8-bit RGB triple at (x,y) for gray(1) / RGB(3) / RGBA(4) images,
    written into rgb[0..2] (caller provides a 3-slot list)."""
    if img.channels == 1:
        var g = Int(img.get(x, y, 0))
        rgb[0] = g
        rgb[1] = g
        rgb[2] = g
        return
    rgb[0] = Int(img.get(x, y, 0))
    rgb[1] = Int(img.get(x, y, 1))
    rgb[2] = Int(img.get(x, y, 2))


struct ComfyMaskImage(Copyable, Movable):
    """Row-major single-channel Comfy/LanPaint mask values.

    Values use Comfy's denoise-mask convention: 1.0 means repaint/unknown,
    0.0 means preserve/known. Use `comfy_mask_to_preserve_mask()` when calling
    helpers that expect the inverse preserve-mask convention.
    """

    var width: Int
    var height: Int
    var values: List[Float32]

    def __init__(out self, width: Int, height: Int, var values: List[Float32]):
        self.width = width
        self.height = height
        self.values = values^


def _image_alpha_at(img: Image, x: Int, y: Int) raises -> Float32:
    if img.channels < 4:
        return -1.0
    return Float32(Int(img.get(x, y, 3))) / 255.0


def _image_rgb_channel_at(img: Image, x: Int, y: Int, channel: Int) raises -> Float32:
    if img.channels == 1:
        return Float32(Int(img.get(x, y, 0))) / 255.0
    if channel >= img.channels:
        return 0.0
    return Float32(Int(img.get(x, y, channel))) / 255.0


def _comfy_mask_value_at(img: Image, x: Int, y: Int, channel: String) raises -> Float32:
    # LoadImage's MASK output is inverted alpha. If the file has no alpha,
    # Comfy emits an all-zero mask.
    if channel == "" or channel == "load_image_mask":
        var alpha = _image_alpha_at(img, x, y)
        if alpha < 0.0:
            return 0.0
        return 1.0 - alpha

    # Low-level explicit ImageToMask file-channel extraction; it does not
    # threshold. The graph importer enforces Comfy IMAGE tensor channel parity.
    if channel == "red" or channel == "r":
        return _image_rgb_channel_at(img, x, y, 0)
    if channel == "green" or channel == "g":
        return _image_rgb_channel_at(img, x, y, 1)
    if channel == "blue" or channel == "b":
        return _image_rgb_channel_at(img, x, y, 2)
    if channel == "alpha" or channel == "a":
        var alpha = _image_alpha_at(img, x, y)
        if alpha < 0.0:
            raise Error("ImageToMask alpha requested but source image has no alpha channel")
        return alpha
    if channel == "luminance" or channel == "gray" or channel == "grey":
        var r = _image_rgb_channel_at(img, x, y, 0)
        var g = _image_rgb_channel_at(img, x, y, 1)
        var b = _image_rgb_channel_at(img, x, y, 2)
        return r * 0.299 + g * 0.587 + b * 0.114

    raise Error(String("unsupported Comfy mask channel: ") + channel)


def decode_comfy_mask(path: String, channel: String) raises -> ComfyMaskImage:
    """Decode a Comfy mask source without resizing or thresholding.

    `channel == "load_image_mask"` mirrors LoadImage's MASK output
    (1-alpha, or zeros without alpha). Other channels are explicit raw file
    channel extraction helpers; workflow graph lowering only admits Comfy's RGB
    ImageToMask channels.
    """
    var img = decode_image_any(path)
    var out = List[Float32](capacity=img.width * img.height)
    for y in range(img.height):
        for x in range(img.width):
            out.append(_comfy_mask_value_at(img, x, y, channel))
    return ComfyMaskImage(img.width, img.height, out^)


def resize_mask_nearest_exact(mask: ComfyMaskImage, width: Int, height: Int) raises -> ComfyMaskImage:
    """Resize a mask with PyTorch/Comfy nearest-exact index selection."""
    if width <= 0 or height <= 0:
        raise Error("resize_mask_nearest_exact: target dimensions must be positive")
    if mask.width <= 0 or mask.height <= 0:
        raise Error("resize_mask_nearest_exact: source dimensions must be positive")
    var out = List[Float32](capacity=width * height)
    for y in range(height):
        var sy = ((2 * y + 1) * mask.height) // (2 * height)
        if sy >= mask.height:
            sy = mask.height - 1
        for x in range(width):
            var sx = ((2 * x + 1) * mask.width) // (2 * width)
            if sx >= mask.width:
                sx = mask.width - 1
            out.append(mask.values[sy * mask.width + sx])
    return ComfyMaskImage(width, height, out^)


def _mask_clamped(mask: ComfyMaskImage, x: Int, y: Int) -> Float32:
    var cx = x
    var cy = y
    if cx < 0:
        cx = 0
    elif cx >= mask.width:
        cx = mask.width - 1
    if cy < 0:
        cy = 0
    elif cy >= mask.height:
        cy = mask.height - 1
    return mask.values[cy * mask.width + cx]


def resize_mask_bilinear(mask: ComfyMaskImage, width: Int, height: Int) raises -> ComfyMaskImage:
    """Resize a mask with PyTorch interpolate(..., mode='bilinear',
    align_corners=False) coordinates."""
    if width <= 0 or height <= 0:
        raise Error("resize_mask_bilinear: target dimensions must be positive")
    if mask.width <= 0 or mask.height <= 0:
        raise Error("resize_mask_bilinear: source dimensions must be positive")
    var out = List[Float32](capacity=width * height)
    for y in range(height):
        var src_y = (Float64(y) + 0.5) * Float64(mask.height) / Float64(height) - 0.5
        var y0 = Int(floor(src_y))
        var wy = Float32(src_y - Float64(y0))
        if y0 < 0:
            y0 = 0
            wy = 0.0
        elif y0 >= mask.height - 1:
            y0 = mask.height - 1
            wy = 0.0
        var y1 = y0 + 1
        if y1 >= mask.height:
            y1 = mask.height - 1
        for x in range(width):
            var src_x = (Float64(x) + 0.5) * Float64(mask.width) / Float64(width) - 0.5
            var x0 = Int(floor(src_x))
            var wx = Float32(src_x - Float64(x0))
            if x0 < 0:
                x0 = 0
                wx = 0.0
            elif x0 >= mask.width - 1:
                x0 = mask.width - 1
                wx = 0.0
            var x1 = x0 + 1
            if x1 >= mask.width:
                x1 = mask.width - 1
            var v00 = _mask_clamped(mask, x0, y0)
            var v01 = _mask_clamped(mask, x1, y0)
            var v10 = _mask_clamped(mask, x0, y1)
            var v11 = _mask_clamped(mask, x1, y1)
            var top = v00 * (1.0 - wx) + v01 * wx
            var bottom = v10 * (1.0 - wx) + v11 * wx
            out.append(top * (1.0 - wy) + bottom * wy)
    return ComfyMaskImage(width, height, out^)


def binarize_lanpaint_denoise_mask(mask: ComfyMaskImage) raises -> ComfyMaskImage:
    """LanPaint hard mask convention: denoise_mask = (mask > 0.5).float()."""
    var out = List[Float32](capacity=len(mask.values))
    for i in range(len(mask.values)):
        if mask.values[i] > 0.5:
            out.append(1.0)
        else:
            out.append(0.0)
    return ComfyMaskImage(mask.width, mask.height, out^)


def comfy_mask_to_preserve_mask(mask: ComfyMaskImage) raises -> ComfyMaskImage:
    """Invert Comfy denoise mask to the preserve-mask convention.

    Result: 1.0 means preserve/known, 0.0 means repaint/unknown.
    """
    var out = List[Float32](capacity=len(mask.values))
    for i in range(len(mask.values)):
        out.append(1.0 - mask.values[i])
    return ComfyMaskImage(mask.width, mask.height, out^)


def load_lanpaint_latent_preserve_mask(
    path: String, channel: String, latent_width: Int, latent_height: Int
) raises -> ComfyMaskImage:
    """Load the mask shape LanPaint uses inside sampling.

    Pipeline: Comfy source mask -> nearest-exact latent resize -> hard
    denoise_mask > 0.5 -> invert to preserve-mask.
    """
    var raw = decode_comfy_mask(path, channel)
    var resized = resize_mask_nearest_exact(raw, latent_width, latent_height)
    var hard = binarize_lanpaint_denoise_mask(resized)
    return comfy_mask_to_preserve_mask(hard)


def smooth_lanpaint_blend_mask(mask: ComfyMaskImage, blend_overlap: Int) raises -> ComfyMaskImage:
    """LanPaint_MaskBlend mask smoothing: max_pool2d then Gaussian conv2d."""
    if blend_overlap <= 1:
        return ComfyMaskImage(mask.width, mask.height, mask.values.copy())
    if blend_overlap % 2 == 0:
        raise Error("LanPaint_MaskBlend blend_overlap must be odd")
    if mask.width <= 0 or mask.height <= 0:
        raise Error("smooth_lanpaint_blend_mask: source dimensions must be positive")
    var radius = blend_overlap // 2
    var pooled = List[Float32](capacity=mask.width * mask.height)
    for y in range(mask.height):
        for x in range(mask.width):
            var max_v: Float32 = 0.0
            for ky in range(blend_overlap):
                var sy = y + ky - radius
                if sy < 0 or sy >= mask.height:
                    continue
                for kx in range(blend_overlap):
                    var sx = x + kx - radius
                    if sx < 0 or sx >= mask.width:
                        continue
                    var v = mask.values[sy * mask.width + sx]
                    if v > max_v:
                        max_v = v
            pooled.append(max_v)

    var sigma = Float64(blend_overlap - 1) / 4.0
    if sigma <= 0.0:
        return ComfyMaskImage(mask.width, mask.height, pooled^)
    var kernel = List[Float64](capacity=blend_overlap * blend_overlap)
    var total: Float64 = 0.0
    for ky in range(blend_overlap):
        var dy = Float64(ky - radius)
        for kx in range(blend_overlap):
            var dx = Float64(kx - radius)
            var w = exp(-((dx * dx + dy * dy) / (2.0 * sigma * sigma)))
            kernel.append(w)
            total += w
    var out = List[Float32](capacity=mask.width * mask.height)
    for y in range(mask.height):
        for x in range(mask.width):
            var acc: Float64 = 0.0
            for ky in range(blend_overlap):
                var sy = y + ky - radius
                if sy < 0 or sy >= mask.height:
                    continue
                for kx in range(blend_overlap):
                    var sx = x + kx - radius
                    if sx < 0 or sx >= mask.width:
                        continue
                    var w = kernel[ky * blend_overlap + kx] / total
                    acc += Float64(pooled[sy * mask.width + sx]) * w
            out.append(Float32(acc))
    return ComfyMaskImage(mask.width, mask.height, out^)


def load_lanpaint_pixel_blend_mask(
    path: String, channel: String, width: Int, height: Int, blend_overlap: Int
) raises -> ComfyMaskImage:
    """Load the image-space mask used by LanPaint_MaskBlend.

    Pipeline: Comfy source mask -> nearest-exact image resize -> max-pool ->
    Gaussian blur. The resulting denoise mask convention is 1.0 = use image2
    / inpainted pixels, 0.0 = use image1 / original pixels.
    """
    var raw = decode_comfy_mask(path, channel)
    var resized = resize_mask_nearest_exact(raw, width, height)
    return smooth_lanpaint_blend_mask(resized, blend_overlap)


def apply_lanpaint_mask_blend_signed_chw(
    base: List[Float32], painted: List[Float32], mask: ComfyMaskImage
) raises -> List[Float32]:
    """Apply LanPaint_MaskBlend to signed CHW RGB arrays.

    Formula matches Python LanPaint_MaskBlend:
    image1 * (1-mask) + image2 * mask.
    """
    var plane = mask.width * mask.height
    if plane <= 0:
        raise Error("apply_lanpaint_mask_blend_signed_chw: invalid mask shape")
    if len(base) != 3 * plane or len(painted) != 3 * plane:
        raise Error("apply_lanpaint_mask_blend_signed_chw: image/mask size mismatch")
    var out = List[Float32](capacity=3 * plane)
    for c in range(3):
        var c_off = c * plane
        for p in range(plane):
            var m = mask.values[p]
            out.append(base[c_off + p] * (1.0 - m) + painted[c_off + p] * m)
    return out^


def load_comfy_latent_preserve_mask(
    path: String, channel: String, latent_width: Int, latent_height: Int
) raises -> ComfyMaskImage:
    """Load the mask shape Comfy's standard sampler uses.

    Pipeline: Comfy source mask -> bilinear latent resize -> invert to
    preserve-mask. No thresholding.
    """
    var raw = decode_comfy_mask(path, channel)
    var resized = resize_mask_bilinear(raw, latent_width, latent_height)
    return comfy_mask_to_preserve_mask(resized)


def mask_active_count(mask: ComfyMaskImage) -> Int:
    var count = 0
    for i in range(len(mask.values)):
        if mask.values[i] > 0.5:
            count += 1
    return count


def mask_mean(mask: ComfyMaskImage) -> Float32:
    if len(mask.values) == 0:
        return 0.0
    var total: Float32 = 0.0
    for i in range(len(mask.values)):
        total += mask.values[i]
    return total / Float32(len(mask.values))


def image_to_rgba_bytes(img: Image) raises -> List[UInt8]:
    """Row-major RGBA8 bytes (alpha forced opaque) — the Backend
    make_texture_rgba upload format."""
    var out = List[UInt8](capacity=img.width * img.height * 4)
    var px: List[Int] = [0, 0, 0]
    for y in range(img.height):
        for x in range(img.width):
            image_rgb_at(img, x, y, px)
            out.append(UInt8(px[0]))
            out.append(UInt8(px[1]))
            out.append(UInt8(px[2]))
            out.append(UInt8(255))
    return out^


def image_to_signed_nchw(img: Image) raises -> List[Float32]:
    """[1,3,H,W] flat host floats in [-1,1] (v = px/127.5 - 1) — the VAE
    encoder input convention (staged images are [-1,1] RGB; see
    pipeline/zimage_prepare.mojo stage contract)."""
    var plane = img.width * img.height
    var out = List[Float32](capacity=3 * plane)
    for _ in range(3 * plane):
        out.append(Float32(0.0))
    var px: List[Int] = [0, 0, 0]
    for y in range(img.height):
        for x in range(img.width):
            image_rgb_at(img, x, y, px)
            var off = y * img.width + x
            out[0 * plane + off] = Float32(px[0]) / 127.5 - 1.0
            out[1 * plane + off] = Float32(px[1]) / 127.5 - 1.0
            out[2 * plane + off] = Float32(px[2]) / 127.5 - 1.0
    return out^
