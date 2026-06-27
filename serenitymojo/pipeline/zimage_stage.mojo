# pipeline/zimage_stage.mojo — config-driven Mojo raw-image stage for Z-Image.
#
# First leg of the production LoRA precache pipeline (config is the SINGLE
# SOURCE OF TRUTH — no hardcoded dataset/stage paths):
#
#   raw PNG/JPEG + .txt captions   (cfg.dataset_path)
#     -> <cfg.cache_dir>_stage/sample_NNNNN.safetensors + .txt
#
# The stage tensor contract matches zimage_prepare.mojo:
#
#   image: [1,3,H,W] BF16, RGB, values in [-1,1]
#
# Aspect bucketing uses the GENERATED ladder (training/aspect_buckets.mojo,
# SimpleTuner semantics: closest ladder aspect -> resize to an intermediary
# that keeps the ORIGINAL aspect -> center-crop to the bucket canvas). These
# are exactly the buckets the Z-Image trainer iterates, so the prepared cache
# is trainer-consumable by construction.
#
# No Python. No Rust. No EriDiffusion cache. Compressed image formats are
# decoded by Mojo FFI into local system image libraries; resize/crop,
# normalize, tensor creation and safetensor writing all happen here.
#
# Build/run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo build --optimization-level 2 -I . \
#       -Xlinker -lpng16 -Xlinker /lib/x86_64-linux-gnu/libturbojpeg.so.0 \
#       serenitymojo/pipeline/zimage_stage.mojo -o /tmp/zimage_stage && \
#     /tmp/zimage_stage serenitymojo/configs/zimage_eri_2000.json [max_images]

from std.collections import List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer, alloc
from std.os import listdir
from std.sys import argv

from serenitymojo.image.decode import decode_image
from serenitymojo.training.aspect_buckets import (
    AspectBucket, AspectBucketAssignment, default_aspect_ladder,
    generate_aspect_buckets, assign_aspect_bucket,
)
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
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.tensor import Tensor
from serenitymojo.training.onetrainer_train_loop_policy import (
    ot_stage_dir_from_train_config,
    ot_dataset_path_from_train_config,
)


comptime SCALE = 1024
comptime SHIFT = 10
comptime MEGAPIXELS = Float64(0.262144)  # 512x512 budget
comptime ALIGN = 64  # SimpleTuner aspect_bucket_alignment default; 64 % 16 == 0
                     # satisfies the Z-Image VAE stride(8) x patch(2) constraint


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


def _image_names_in(data_dir: String) raises -> List[String]:
    var raw = listdir(data_dir)
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


def _out_name(idx: Int) -> String:
    # zero-padded for stable lexical sort in the prepare step (KleinCache sorts).
    var s = String(idx)
    var pad = String("")
    for _ in range(5 - s.byte_length() if s.byte_length() < 5 else 0):
        pad += String("0")
    return String("sample_") + pad + s


def _min_int(a: Int, b: Int) -> Int:
    if a < b:
        return a
    return b


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


def _resize_crop_explicit_to_tensor_values(
    rgb: List[UInt8],
    width: Int,
    height: Int,
    scale_w: Int,
    scale_h: Int,
    crop_x: Int,
    crop_y: Int,
    out_w: Int,
    out_h: Int,
) raises -> List[Float32]:
    """Resize the source to an EXPLICIT intermediary (scale_w x scale_h,
    SimpleTuner's original-aspect intermediary) and crop the (out_w x out_h)
    bucket canvas at the given offsets. Deterministic fixed-point bilinear."""
    if width <= 0 or height <= 0:
        raise Error("decoded image has invalid dimensions")
    if len(rgb) != width * height * 3:
        raise Error("decoded image byte count does not match dimensions")
    if scale_w < out_w or scale_h < out_h:
        raise Error("stage: intermediary smaller than bucket canvas")
    if crop_x < 0 or crop_y < 0 or crop_x + out_w > scale_w or crop_y + out_h > scale_h:
        raise Error("stage: crop window outside the intermediary")

    var values = List[Float32](capacity=3 * out_h * out_w)
    for c in range(3):
        for oy in range(out_h):
            var iy = oy + crop_y
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
                var ix = ox + crop_x
                var src_x_fp = (((2 * ix + 1) * width * SCALE) // (2 * scale_w)) - (SCALE // 2)
                if src_x_fp < 0:
                    src_x_fp = 0
                var max_x_fp = (width - 1) * SCALE
                if src_x_fp > max_x_fp:
                    src_x_fp = max_x_fp
                var x0 = src_x_fp >> SHIFT
                var fx = src_x_fp - x0 * SCALE
                var x1 = _min_int(x0 + 1, width - 1)
                var px = _bilinear_u8(rgb, width, x0, y0, x1, y1, fx, fy, c)
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
    var a = argv()
    if len(a) < 2 or not String(a[1]).endswith(String(".json")):
        raise Error(
            "usage: zimage_stage <config.json> [max_images]  "
            "(config must set dataset_path + cache_dir; no hardcoded dataset)"
        )
    var cfg = read_model_config(String(a[1]))
    var data_dir = ot_dataset_path_from_train_config(cfg)
    var stage_dir = ot_stage_dir_from_train_config(cfg, String(""))

    var max_images = 0
    if len(a) >= 3:
        var s = String(a[2])
        var bs = s.as_bytes()
        for i in range(s.byte_length()):
            if bs[i] < 0x30 or bs[i] > 0x39:
                raise Error("max_images must be a non-negative integer")
            max_images = max_images * 10 + Int(bs[i] - 0x30)

    print("=== Z-Image stage: raw PNG/JPEG -> bucketed image safetensors ===")
    print("  config:", String(a[1]))
    print("  data:  ", data_dir)
    print("  stage: ", stage_dir)

    var ladder = default_aspect_ladder()
    var buckets = generate_aspect_buckets(MEGAPIXELS, ALIGN, ladder)
    print("  ladder buckets (area=262144 px, align=", ALIGN, "):")
    for i in range(len(buckets)):
        print(
            "    [", i, "] aspect_x100=", Int(buckets[i].aspect * 100.0 + 0.5),
            " canvas=", buckets[i].width, "x", buckets[i].height,
            " latent=", buckets[i].height // 8, "x", buckets[i].width // 8,
        )
    _ = sys_system(String("mkdir -p ") + stage_dir)
    _ = sys_system(String("rm -f ") + stage_dir + String("/*.safetensors"))
    _ = sys_system(String("rm -f ") + stage_dir + String("/*.txt"))
    _ = sys_system(String("rm -f ") + stage_dir + String("/aspect_buckets.json"))

    var names = _image_names_in(data_dir)
    if len(names) == 0:
        raise Error(String("no PNG/JPEG images found in ") + data_dir)
    var n = len(names)
    if max_images > 0 and max_images < n:
        n = max_images

    var manifest = String("{\n")
    manifest += String('  "schema":"serenity.aspect_buckets.v1",\n')
    manifest += String('  "semantics":"simpletuner closest-aspect + pixel-area + center-crop",\n')
    manifest += String('  "pixel_area":262144,\n')
    manifest += String('  "alignment":') + String(ALIGN) + String(",\n")
    manifest += String('  "buckets":[')
    for i in range(len(buckets)):
        if i > 0:
            manifest += String(",")
        manifest += (
            String('{"aspect_x100":') + String(Int(buckets[i].aspect * 100.0 + 0.5))
            + String(',"width":') + String(buckets[i].width)
            + String(',"height":') + String(buckets[i].height) + String("}")
        )
    manifest += String("],\n")
    manifest += String('  "samples":[\n')

    for i in range(n):
        var src_name = names[i]
        var stem = _stem_for_image(src_name)
        var src_path = data_dir + String("/") + src_name
        var cap_path = data_dir + String("/") + stem + String(".txt")
        var out_stem = _out_name(i)
        var out_img = stage_dir + String("/") + out_stem + String(".safetensors")
        var out_txt = stage_dir + String("/") + out_stem + String(".txt")

        print("-- stage", i + 1, "/", n, src_name)
        var decoded = decode_image(src_path)
        var asn = assign_aspect_bucket(
            decoded.width, decoded.height, ladder, MEGAPIXELS, ALIGN,
        )
        print(
            "   decoded:", decoded.width, "x", decoded.height,
            " -> bucket", asn.bucket_index,
            " canvas:", asn.target_w, "x", asn.target_h,
            " inter:", asn.inter_w, "x", asn.inter_h,
            " crop:(", asn.crop_x, ",", asn.crop_y, ")",
        )
        var values = _resize_crop_explicit_to_tensor_values(
            decoded.rgb, decoded.width, decoded.height,
            asn.inter_w, asn.inter_h, asn.crop_x, asn.crop_y,
            asn.target_w, asn.target_h,
        )
        _save_stage_tensor(values, out_img, asn.target_h, asn.target_w, ctx)
        var caption = _read_text(cap_path)
        _write_text(out_txt, caption)

        if i > 0:
            manifest += String(",\n")
        manifest += (
            String('    {"file":"') + out_stem + String('.safetensors"')
            + String(',"source":"') + src_name + String('"')
            + String(',"source_w":') + String(decoded.width)
            + String(',"source_h":') + String(decoded.height)
            + String(',"bucket":') + String(asn.bucket_index)
            + String(',"target_w":') + String(asn.target_w)
            + String(',"target_h":') + String(asn.target_h)
            + String(',"inter_w":') + String(asn.inter_w)
            + String(',"inter_h":') + String(asn.inter_h)
            + String(',"crop_x":') + String(asn.crop_x)
            + String(',"crop_y":') + String(asn.crop_y) + String("}")
        )

    manifest += String("\n  ]\n}\n")
    var manifest_bytes = List[UInt8]()
    var mb = manifest.as_bytes()
    for i in range(manifest.byte_length()):
        manifest_bytes.append(mb[i])
    _write_text(stage_dir + String("/aspect_buckets.json"), manifest_bytes)
    print("PASS: wrote", n, "Z-Image stage samples to", stage_dir)
