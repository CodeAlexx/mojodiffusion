# Creator-faithful host preprocessing for SCAIL-2 production input ingest.
#
# The pinned creator has two distinct resize paths:
#   * reference images: float32 antialiased Keys bicubic;
#   * decoded video frames: torchvision casts uint8 to float32, applies the
#     antialiased Keys bicubic path, clamps, then rounds back to uint8.
# Keeping those paths distinct is required for parity at mask boundaries.

from std.collections import List
from std.math import ceil, floor, round

from serenitymojo.image.decode import DecodedImage


comptime SCAIL2_MASK_THRESHOLD = Float32((225.0 - 127.5) / 127.5)


def _cubic_keys_f32(xin: Float32) -> Float32:
    var x = xin if xin >= 0.0 else -xin
    comptime a = Float32(-0.5)
    if x < 1.0:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    if x < 2.0:
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a
    return 0.0


@fieldwise_init
struct _F32Coeffs(Movable):
    var kmin: List[Int]
    var kcount: List[Int]
    var weights: List[Float32]
    var ksize: Int


def _f32_coeffs(in_size: Int, out_size: Int) raises -> _F32Coeffs:
    var scale = Float32(in_size) / Float32(out_size)
    var support = Float32(2.0) * (scale if scale >= 1.0 else Float32(1.0))
    var invscale = Float32(1.0) / scale if scale >= 1.0 else Float32(1.0)
    var ksize = Int(ceil(Float64(support))) * 2 + 1
    var kmin = List[Int]()
    var kcount = List[Int]()
    var weights = List[Float32]()
    kmin.resize(out_size, 0)
    kcount.resize(out_size, 0)
    weights.resize(out_size * ksize, Float32(0.0))
    for i in range(out_size):
        var center = scale * (Float32(i) + 0.5)
        var xmin = Int(center - support + 0.5)
        if xmin < 0:
            xmin = 0
        var xmax = Int(center + support + 0.5)
        if xmax > in_size:
            xmax = in_size
        var count = xmax - xmin
        if count > ksize:
            count = ksize
        var total = Float32(0.0)
        for j in range(count):
            var weight = _cubic_keys_f32(
                (Float32(j + xmin) - center + 0.5) * invscale
            )
            weights[i * ksize + j] = weight
            total += weight
        if total != 0.0:
            for j in range(count):
                weights[i * ksize + j] /= total
        kmin[i] = xmin
        kcount[i] = count
    return _F32Coeffs(kmin^, kcount^, weights^, ksize)


def _scail2_resize_center_f32_scaled(
    img: DecodedImage, target_height: Int, target_width: Int,
    normalize_input: Bool,
) raises -> List[Float32]:
    if img.width <= 0 or img.height <= 0:
        raise Error("SCAIL-2 decoded reference has invalid dimensions")
    var resized_height = target_height
    var resized_width = target_width
    if Float64(img.width) / Float64(img.height) > Float64(target_width) / Float64(target_height):
        resized_width = (img.width * target_height) // img.height
    else:
        resized_height = (img.height * target_width) // img.width
    var hc = _f32_coeffs(img.width, resized_width)
    var vc = _f32_coeffs(img.height, resized_height)
    var horizontal = List[Float32]()
    horizontal.resize(img.height * resized_width * 3, Float32(0.0))
    for y in range(img.height):
        for x in range(resized_width):
            for c in range(3):
                var acc = Float32(0.0)
                for k in range(hc.kcount[x]):
                    var source = Float32(
                        Int(img.rgb[(y * img.width + hc.kmin[x] + k) * 3 + c])
                    )
                    if normalize_input:
                        source = source / 255.0
                    acc += source * hc.weights[x * hc.ksize + k]
                horizontal[(y * resized_width + x) * 3 + c] = acc
    var crop_left = (resized_width - target_width) // 2
    var crop_top = (resized_height - target_height) // 2
    var full = List[Float32]()
    full.resize(resized_height * target_width * 3, Float32(0.0))
    for y in range(resized_height):
        for x in range(target_width):
            var source_x = crop_left + x
            for c in range(3):
                var acc = Float32(0.0)
                for k in range(vc.kcount[y]):
                    acc += (
                        horizontal[((vc.kmin[y] + k) * resized_width + source_x) * 3 + c]
                        * vc.weights[y * vc.ksize + k]
                    )
                full[(y * target_width + x) * 3 + c] = acc
    if crop_top == 0 and resized_height == target_height:
        return full^
    var cropped = List[Float32]()
    cropped.resize(target_height * target_width * 3, Float32(0.0))
    for y in range(target_height):
        for x in range(target_width):
            for c in range(3):
                cropped[(y * target_width + x) * 3 + c] = full[
                    ((crop_top + y) * target_width + x) * 3 + c
                ]
    return cropped^


def scail2_resize_center_f32(
    img: DecodedImage, target_height: Int, target_width: Int
) raises -> List[Float32]:
    return _scail2_resize_center_f32_scaled(
        img, target_height, target_width, True
    )


def scail2_resize_center_u8(
    img: DecodedImage, target_height: Int, target_width: Int
) raises -> List[UInt8]:
    var resized = _scail2_resize_center_f32_scaled(
        img, target_height, target_width, False
    )
    var out = List[UInt8]()
    out.resize(len(resized), UInt8(0))
    for i in range(len(resized)):
        var value = Int(round(resized[i]))
        if value < 0:
            value = 0
        elif value > 255:
            value = 255
        out[i] = UInt8(value)
    return out^


def scail2_hwc_f32_to_chw(
    pixels: List[Float32], height: Int, width: Int
) -> List[Float32]:
    var out = List[Float32]()
    out.resize(3 * height * width, Float32(0.0))
    var plane = height * width
    for y in range(height):
        for x in range(width):
            for c in range(3):
                out[c * plane + y * width + x] = pixels[(y * width + x) * 3 + c]
    return out^


def _cubic_a_f32(xin: Float32, a: Float32) -> Float32:
    var x = xin if xin >= 0.0 else -xin
    if x < 1.0:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    if x < 2.0:
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a
    return 0.0


def _clamp_index(value: Int, size: Int) -> Int:
    if value < 0:
        return 0
    if value >= size:
        return size - 1
    return value


def scail2_clip_from_reference_chw(
    reference: List[Float32], height: Int, width: Int
) raises -> List[Float32]:
    """Creator CLIP pixel path: signed reference -> bicubic 224 -> unit -> norm."""
    comptime out_size = 224
    comptime cubic_a = Float32(-0.75)
    comptime mean0 = Float32(0.48145466)
    comptime mean1 = Float32(0.4578275)
    comptime mean2 = Float32(0.40821073)
    comptime std0 = Float32(0.26862954)
    comptime std1 = Float32(0.26130258)
    comptime std2 = Float32(0.27577711)
    if len(reference) != 3 * height * width:
        raise Error("SCAIL-2 reference CHW size mismatch")
    var out = List[Float32]()
    out.resize(3 * out_size * out_size, Float32(0.0))
    var scale_y = Float32(height) / Float32(out_size)
    var scale_x = Float32(width) / Float32(out_size)
    var source_plane = height * width
    var output_plane = out_size * out_size
    for oy in range(out_size):
        var source_y = (Float32(oy) + 0.5) * scale_y - 0.5
        var ybase = Int(floor(Float64(source_y)))
        var ty = source_y - Float32(ybase)
        var wy: List[Float32] = [
            _cubic_a_f32(ty + 1.0, cubic_a),
            _cubic_a_f32(ty, cubic_a),
            _cubic_a_f32(1.0 - ty, cubic_a),
            _cubic_a_f32(2.0 - ty, cubic_a),
        ]
        for ox in range(out_size):
            var source_x = (Float32(ox) + 0.5) * scale_x - 0.5
            var xbase = Int(floor(Float64(source_x)))
            var tx = source_x - Float32(xbase)
            var wx: List[Float32] = [
                _cubic_a_f32(tx + 1.0, cubic_a),
                _cubic_a_f32(tx, cubic_a),
                _cubic_a_f32(1.0 - tx, cubic_a),
                _cubic_a_f32(2.0 - tx, cubic_a),
            ]
            for c in range(3):
                var rows = List[Float32]()
                rows.resize(4, Float32(0.0))
                for ky in range(4):
                    var sy = _clamp_index(ybase - 1 + ky, height)
                    var v0 = reference[c * source_plane + sy * width + _clamp_index(xbase - 1, width)] * 2.0 - 1.0
                    var v1 = reference[c * source_plane + sy * width + _clamp_index(xbase, width)] * 2.0 - 1.0
                    var v2 = reference[c * source_plane + sy * width + _clamp_index(xbase + 1, width)] * 2.0 - 1.0
                    var v3 = reference[c * source_plane + sy * width + _clamp_index(xbase + 2, width)] * 2.0 - 1.0
                    rows[ky] = v0 * wx[0] + v1 * wx[1] + v2 * wx[2] + v3 * wx[3]
                var signed = rows[0] * wy[0] + rows[1] * wy[1] + rows[2] * wy[2] + rows[3] * wy[3]
                var unit = signed * 0.5 + 0.5
                var mean = mean0 if c == 0 else (mean1 if c == 1 else mean2)
                var std = std0 if c == 0 else (std1 if c == 1 else std2)
                out[c * output_plane + oy * out_size + ox] = (unit - mean) / std
    return out^


def scail2_half_bilinear_u8_to_signed_chw(
    pixels: List[UInt8], height: Int, width: Int
) raises -> List[Float32]:
    if height % 2 != 0 or width % 2 != 0:
        raise Error("SCAIL-2 half-resolution input dimensions must be even")
    var oh = height // 2
    var ow = width // 2
    var out = List[Float32]()
    out.resize(3 * oh * ow, Float32(0.0))
    var plane = oh * ow
    for c in range(3):
        for y in range(oh):
            for x in range(ow):
                var v00 = (Float32(Int(pixels[((2 * y) * width + 2 * x) * 3 + c])) - 127.5) / 127.5
                var v01 = (Float32(Int(pixels[((2 * y) * width + 2 * x + 1) * 3 + c])) - 127.5) / 127.5
                var v10 = (Float32(Int(pixels[((2 * y + 1) * width + 2 * x) * 3 + c])) - 127.5) / 127.5
                var v11 = (Float32(Int(pixels[((2 * y + 1) * width + 2 * x + 1) * 3 + c])) - 127.5) / 127.5
                out[c * plane + y * ow + x] = (
                    v00 * 0.25 + v01 * 0.25 + v10 * 0.25 + v11 * 0.25
                )
    return out^


def scail2_half_bilinear_u8_to_unit_chw(
    pixels: List[UInt8], height: Int, width: Int
) raises -> List[Float32]:
    var signed = scail2_half_bilinear_u8_to_signed_chw(pixels, height, width)
    for i in range(len(signed)):
        signed[i] = (signed[i] + 1.0) * 0.5
    return signed^


def scail2_binary7_area_from_signed_chw(
    signed_chw: List[Float32], height: Int, width: Int, downsample: Int
) raises -> List[Float32]:
    if height % downsample != 0 or width % downsample != 0:
        raise Error("SCAIL-2 mask area downsample must divide dimensions")
    var oh = height // downsample
    var ow = width // downsample
    var out = List[Float32]()
    out.resize(7 * oh * ow, Float32(0.0))
    var source_plane = height * width
    var target_plane = oh * ow
    var denom = Float32(downsample * downsample)
    for oy in range(oh):
        for ox in range(ow):
            for sy in range(oy * downsample, (oy + 1) * downsample):
                for sx in range(ox * downsample, (ox + 1) * downsample):
                    var off = sy * width + sx
                    var red = 1 if signed_chw[off] > SCAIL2_MASK_THRESHOLD else 0
                    var green = 1 if signed_chw[source_plane + off] > SCAIL2_MASK_THRESHOLD else 0
                    var blue = 1 if signed_chw[2 * source_plane + off] > SCAIL2_MASK_THRESHOLD else 0
                    var values: List[Int] = [
                        red * green * blue,
                        red * (1 - green) * (1 - blue),
                        (1 - red) * green * (1 - blue),
                        (1 - red) * (1 - green) * blue,
                        red * green * (1 - blue),
                        red * (1 - green) * blue,
                        (1 - red) * green * blue,
                    ]
                    for q in range(7):
                        out[q * target_plane + oy * ow + ox] += Float32(values[q]) / denom
    return out^


def scail2_binary7_area_from_unit_chw(
    unit_chw: List[Float32], height: Int, width: Int, downsample: Int
) raises -> List[Float32]:
    var signed = unit_chw.copy()
    for i in range(len(signed)):
        signed[i] = signed[i] * 2.0 - 1.0
    return scail2_binary7_area_from_signed_chw(
        signed^, height, width, downsample
    )


def scail2_pack_temporal_mask28(
    binary7_by_frame: List[Float32], frames: Int, height: Int, width: Int
) raises -> List[Float32]:
    if frames < 1 or (frames - 1) % 4 != 0:
        raise Error("SCAIL-2 temporal mask requires 4n+1 frames")
    var latent_frames = (frames - 1) // 4 + 1
    var plane = height * width
    var out = List[Float32]()
    out.resize(28 * latent_frames * plane, Float32(0.0))
    for lf in range(latent_frames):
        for q in range(28):
            var padded_frame = lf * 4 + q // 7
            var source_frame = 0 if padded_frame < 4 else padded_frame - 3
            var source_channel = q % 7
            for i in range(plane):
                out[(q * latent_frames + lf) * plane + i] = binary7_by_frame[
                    (source_frame * 7 + source_channel) * plane + i
                ]
    return out^
