# pipeline/zimage_stage_alina.mojo — local Mojo raw-image stage for Z-Image.
#
# This is the first leg of the production LoRA pipeline:
#
#   raw Alina PNG/JPEG + txt captions
#     -> output/alina_zimage_stage/alina_NNN.safetensors + alina_NNN.txt
#
# The stage tensor contract matches zimage_prepare.mojo:
#
#   image: [1,3,H,W] BF16, RGB, values in [-1,1]
#
# OneTrainer's Alina "512" baseline uses aspect buckets, not square 512 crops.
# The main portrait bucket is 576x448 -> latent [16,72,56]; the video-frame
# sample lands in 704x384 -> latent [16,88,48].
#
# No Python. No Rust. No EriDiffusion cache. The compressed image formats are
# decoded by Mojo FFI into local system image libraries, then resize/crop,
# normalize, tensor creation, and safetensor writing all happen here.
#
# Build/run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo build -I . \
#       -Xlinker -lpng16 -Xlinker /lib/x86_64-linux-gnu/libturbojpeg.so.0 \
#       serenitymojo/pipeline/zimage_stage_alina.mojo -o /tmp/zimage_stage_alina && \
#     /tmp/zimage_stage_alina

from std.collections import List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer, alloc
from std.os import listdir

from serenitymojo.image.decode import decode_image
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import (
    BytePtr,
    O_CREAT,
    O_RDONLY,
    O_TRUNC,
    O_WRONLY,
    sys_close,
    sys_open,
    sys_pread,
    sys_pwrite,
    sys_system,
)
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.tensor import Tensor


comptime DATA_DIR = "/home/alex/datasets/AlinaAignatova"
comptime STAGE_DIR = "/home/alex/mojodiffusion/output/alina_zimage_stage"
comptime SCALE = 1024
comptime SHIFT = 10


def _sort_strings(mut xs: List[String]):
    for i in range(1, len(xs)):
        var key = xs[i]
        var j = i - 1
        while j >= 0 and xs[j] > key:
            xs[j + 1] = xs[j]
            j -= 1
        xs[j + 1] = key


def _is_image_name(s: String) -> Bool:
    return (
        s.endswith(".png")
        or s.endswith(".PNG")
        or s.endswith(".jpg")
        or s.endswith(".JPG")
        or s.endswith(".jpeg")
        or s.endswith(".JPEG")
    )


def _drop_suffix(s: String, suffix_len: Int) raises -> String:
    var n = s.byte_length() - suffix_len
    if n < 0:
        raise Error("_drop_suffix: suffix longer than string")
    var src = s.as_bytes()
    var out = List[UInt8]()
    for i in range(n):
        out.append(src[i])
    return String(unsafe_from_utf8=out)


def _stem_for_image(name: String) raises -> String:
    if name.endswith(".jpeg") or name.endswith(".JPEG"):
        return _drop_suffix(name, 5)
    return _drop_suffix(name, 4)


def _image_names() raises -> List[String]:
    var raw = listdir(String(DATA_DIR))
    var xs = List[String]()
    for i in range(len(raw)):
        if _is_image_name(raw[i]):
            xs.append(raw[i])
    _sort_strings(xs)
    return xs^


def _read_text(path: String) raises -> List[UInt8]:
    var fd = sys_open(path, O_RDONLY, Int32(0))
    if fd < 0:
        raise Error(String("caption missing: ") + path)
    var out = List[UInt8]()
    comptime CHUNK = 65536
    var buf = alloc[UInt8](CHUNK)
    var offset = 0
    while True:
        var n = sys_pread(fd, BytePtr(unsafe_from_address=Int(buf)), CHUNK, offset)
        if n < 0:
            buf.free()
            _ = sys_close(fd)
            raise Error(String("caption read failed: ") + path)
        if n == 0:
            break
        for i in range(n):
            out.append(buf[i])
        offset += n
        if n < CHUNK:
            break
    buf.free()
    _ = sys_close(fd)
    return out^


def _write_all(fd: Int, buf: BytePtr, count: Int, offset: Int) raises:
    var done = 0
    while done < count:
        var n = sys_pwrite(fd, buf + done, count - done, offset + done)
        if n < 0:
            raise Error("pwrite failed")
        if n == 0:
            raise Error("pwrite wrote 0 bytes")
        done += n


def _write_text(path: String, bytes: List[UInt8]) raises:
    var fd = sys_open(path, O_WRONLY | O_CREAT | O_TRUNC, Int32(0o644))
    if fd < 0:
        raise Error(String("open caption for write failed: ") + path)
    if len(bytes) > 0:
        var buf = alloc[UInt8](len(bytes))
        for i in range(len(bytes)):
            buf[i] = bytes[i]
        try:
            _write_all(fd, BytePtr(unsafe_from_address=Int(buf)), len(bytes), 0)
        except e:
            buf.free()
            _ = sys_close(fd)
            raise Error(String("caption write failed: ") + String(e))
        buf.free()
    _ = sys_close(fd)


def _stage_name(idx: Int) -> String:
    if idx < 10:
        return String("alina_00") + String(idx)
    if idx < 100:
        return String("alina_0") + String(idx)
    return String("alina_") + String(idx)


def _min_int(a: Int, b: Int) -> Int:
    if a < b:
        return a
    return b


def _max_int(a: Int, b: Int) -> Int:
    if a > b:
        return a
    return b


def _round_div(numer: Int, denom: Int) -> Int:
    return (numer + denom // 2) // denom


def _abs64(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def _bucket_for(height: Int, width: Int) -> Tuple[Int, Int]:
    # OneTrainer 512 target, quantization=64 automatic buckets.
    var hs = List[Int]()
    var ws = List[Int]()
    hs.append(256); ws.append(1024)
    hs.append(256); ws.append(960)
    hs.append(320); ws.append(896)
    hs.append(320); ws.append(832)
    hs.append(384); ws.append(704)
    hs.append(448); ws.append(640)
    hs.append(448); ws.append(576)
    hs.append(512); ws.append(512)
    hs.append(576); ws.append(448)
    hs.append(640); ws.append(448)
    hs.append(704); ws.append(384)
    hs.append(832); ws.append(320)
    hs.append(896); ws.append(320)
    hs.append(960); ws.append(256)
    hs.append(1024); ws.append(256)

    var aspect = Float64(height) / Float64(width)
    var best = 0
    var best_diff = _abs64(Float64(hs[0]) / Float64(ws[0]) - aspect)
    for i in range(1, len(hs)):
        var d = _abs64(Float64(hs[i]) / Float64(ws[i]) - aspect)
        if d < best_diff:
            best = i
            best_diff = d
    return (hs[best], ws[best])


def _bilinear_u8(
    rgb: List[UInt8],
    width: Int,
    sx0: Int,
    sy0: Int,
    sx1: Int,
    sy1: Int,
    fx: Int,
    fy: Int,
    c: Int,
) -> UInt8:
    var invx = SCALE - fx
    var invy = SCALE - fy
    var p00 = Int(rgb[(sy0 * width + sx0) * 3 + c])
    var p01 = Int(rgb[(sy0 * width + sx1) * 3 + c])
    var p10 = Int(rgb[(sy1 * width + sx0) * 3 + c])
    var p11 = Int(rgb[(sy1 * width + sx1) * 3 + c])
    var top = p00 * invx + p01 * fx
    var bot = p10 * invx + p11 * fx
    var v = (top * invy + bot * fy + (SCALE * SCALE // 2)) // (SCALE * SCALE)
    if v < 0:
        v = 0
    if v > 255:
        v = 255
    return UInt8(v)


def _resize_center_crop_to_tensor_values(
    rgb: List[UInt8],
    width: Int,
    height: Int,
    out_h: Int,
    out_w: Int,
) raises -> List[Float32]:
    if width <= 0 or height <= 0:
        raise Error("decoded image has invalid dimensions")
    if len(rgb) != width * height * 3:
        raise Error("decoded image byte count does not match dimensions")

    # Match OneTrainer's ScaleCropImage behavior without importing MGDS:
    #   1. resize the full source so the selected bucket fits
    #   2. center-crop from that resized image
    # For a 1080x1350 portrait this gives scale_resolution 576x461 and
    # crop_offset (y=0, x=6), matching the OneTrainer cache metadata.
    var scale_h: Int
    var scale_w: Int
    if height * out_w > width * out_h:
        scale_h = _round_div(height * out_w, width)
        scale_w = out_w
    else:
        scale_h = out_h
        scale_w = _round_div(width * out_h, height)
    if scale_h < out_h or scale_w < out_w:
        raise Error("decoded image scale resolution smaller than target bucket")
    var y_offset = (scale_h - out_h) // 2
    var x_offset = (scale_w - out_w) // 2
    var values = List[Float32](capacity=3 * out_h * out_w)

    # NCHW order. Use fixed-point bilinear; this avoids depending on a runtime
    # floating floor implementation and keeps staging deterministic.
    for c in range(3):
        for oy in range(out_h):
            var iy = oy + y_offset
            var src_y_fp = (((2 * iy + 1) * height * SCALE) // (2 * scale_h)) - (SCALE // 2)
            if src_y_fp < 0:
                src_y_fp = 0
            var max_y_fp = (height - 1) * SCALE
            if src_y_fp > max_y_fp:
                src_y_fp = max_y_fp
            var y0 = src_y_fp >> SHIFT
            var fy = src_y_fp - y0 * SCALE
            var y1 = _min_int(y0 + 1, height - 1)
            for ox in range(out_w):
                var ix = ox + x_offset
                var src_x_fp = (((2 * ix + 1) * width * SCALE) // (2 * scale_w)) - (SCALE // 2)
                if src_x_fp < 0:
                    src_x_fp = 0
                var max_x_fp = (width - 1) * SCALE
                if src_x_fp > max_x_fp:
                    src_x_fp = max_x_fp
                var x0 = src_x_fp >> SHIFT
                var fx = src_x_fp - x0 * SCALE
                var x1 = _min_int(x0 + 1, width - 1)
                var px = _bilinear_u8(
                    rgb,
                    width,
                    x0,
                    y0,
                    x1,
                    y1,
                    fx,
                    fy,
                    c,
                )
                values.append(Float32(px) * Float32(2.0 / 255.0) - Float32(1.0))
    return values^


def _save_stage_tensor(values: List[Float32], out_path: String, out_h: Int, out_w: Int, ctx: DeviceContext) raises:
    var image = Tensor.from_host(values, [1, 3, out_h, out_w], STDtype.BF16, ctx)
    var names = List[String]()
    names.append(String("image"))
    var tensors = List[ArcPointer[Tensor]]()
    tensors.append(ArcPointer(image^))
    save_safetensors(names, tensors, out_path, ctx)


def main() raises:
    var ctx = DeviceContext()
    print("=== Z-Image Alina stage: raw PNG/JPEG -> image safetensors ===")
    print("  data: ", DATA_DIR)
    print("  stage:", STAGE_DIR)
    _ = sys_system(String("mkdir -p ") + String(STAGE_DIR))
    _ = sys_system(String("rm -f ") + String(STAGE_DIR) + String("/*.safetensors"))
    _ = sys_system(String("rm -f ") + String(STAGE_DIR) + String("/*.txt"))

    var names = _image_names()
    if len(names) == 0:
        raise Error(String("no PNG/JPEG images found in ") + String(DATA_DIR))

    for i in range(len(names)):
        var src_name = names[i]
        var stem = _stem_for_image(src_name)
        var src_path = String(DATA_DIR) + String("/") + src_name
        var cap_path = String(DATA_DIR) + String("/") + stem + String(".txt")
        var out_stem = _stage_name(i)
        var out_img = String(STAGE_DIR) + String("/") + out_stem + String(".safetensors")
        var out_txt = String(STAGE_DIR) + String("/") + out_stem + String(".txt")

        print("-- stage", i + 1, "/", len(names), src_name)
        var decoded = decode_image(src_path)
        var bucket = _bucket_for(decoded.height, decoded.width)
        var out_h = bucket[0]
        var out_w = bucket[1]
        print("   decoded:", decoded.width, "x", decoded.height, " bucket:", out_h, "x", out_w)
        var values = _resize_center_crop_to_tensor_values(decoded.rgb, decoded.width, decoded.height, out_h, out_w)
        _save_stage_tensor(values, out_img, out_h, out_w, ctx)
        var caption = _read_text(cap_path)
        _write_text(out_txt, caption)
        print("   wrote:", out_img)

    print("PASS: wrote", len(names), "Z-Image stage samples to", STAGE_DIR)
