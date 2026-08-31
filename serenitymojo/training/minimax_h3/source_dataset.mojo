# MiniMax-H3 source-dataset intake and deterministic receipt.
#
# This is a host-only boundary.  It enumerates authored source pairs and proves
# their bytes before any cache builder or DeviceContext may be entered.  The
# logical run identity and physical path are intentionally separate: the
# requested run is `eri_with_trigger`, while its directory is supplied by the
# caller and is never inferred from a similarly named directory.

from std.collections import List
from std.memory import alloc
from std.os import listdir

from serenitymojo.io.ffi import (
    O_RDONLY,
    file_size,
    sys_close,
    sys_open,
    sys_pread,
)
from serenitymojo.training.minimax_h3.sha256 import (
    minimax_h3_sha256_file,
    minimax_h3_sha256_text,
)
from serenitymojo.training.minimax_h3.contract import (
    MINIMAX_H3_INTAKE_DATASET_IDENTITY,
)


comptime MINIMAX_H3_SOURCE_RECEIPT_SCHEMA = (
    "serenity.minimax_h3.source_dataset_receipt.v2"
)


@fieldwise_init
struct MiniMaxH3SourceSample(Copyable, Movable):
    var stem: String
    var image_relative_path: String
    var caption_relative_path: String
    var image_sha256: String
    var caption_sha256: String
    # Deterministic content identity for the exact relative pair.  This is the
    # sample ID accepted by the H3 cache manifest; ordinal cache names cannot
    # silently retarget an artifact after source order changes.
    var pair_fingerprint: String
    var original_width: Int
    var original_height: Int
    # Exact authored UTF-8 bytes, including authored leading/trailing whitespace
    # and line endings.  Intake never parses, strips, prepends, or rewrites it.
    var caption: String
    var trigger_present: Bool


@fieldwise_init
struct MiniMaxH3SourceReceipt(Copyable, Movable):
    var schema: String
    var dataset_identity: String
    var physical_path: String
    var trigger: String
    var trigger_present_count: Int
    var receipt: String
    var samples: List[MiniMaxH3SourceSample]


def _sort_strings(mut values: List[String]):
    for index in range(1, len(values)):
        var key = values[index]
        var cursor = index - 1
        while cursor >= 0 and values[cursor] > key:
            values[cursor + 1] = values[cursor]
            cursor -= 1
        values[cursor + 1] = key


def _extension_length(name: String) -> Int:
    var lower = String(name.lower())
    if lower.endswith(String(".jpeg")) or lower.endswith(String(".webp")):
        return 5
    if lower.endswith(String(".jpg")) or lower.endswith(String(".png")):
        return 4
    return 0


def _is_unsupported_image(name: String) -> Bool:
    var lower = String(name.lower())
    return (
        lower.endswith(String(".bmp"))
        or lower.endswith(String(".gif"))
        or lower.endswith(String(".tif"))
        or lower.endswith(String(".tiff"))
        or lower.endswith(String(".avif"))
        or lower.endswith(String(".heic"))
        or lower.endswith(String(".heif"))
    )


def _prefix_bytes(value: String, count: Int) -> String:
    var out = String("")
    var source = value.as_bytes()
    for index in range(count):
        out += chr(Int(source[index]))
    return out^


def _contains(values: List[String], needle: String) -> Bool:
    for value in values:
        if value == needle:
            return True
    return False


def _read_authored_utf8(path: String) raises -> String:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("MiniMax H3 source caption is missing: ") + path)
    var size = file_size(fd)
    if size < 0:
        _ = sys_close(fd)
        raise Error(String("MiniMax H3 source caption cannot be sized: ") + path)
    var buf = alloc[UInt8](size if size > 0 else 1)
    var done = 0
    while done < size:
        var count = sys_pread(fd, buf + done, size - done, done)
        if count <= 0:
            buf.free()
            _ = sys_close(fd)
            raise Error(String("MiniMax H3 source caption short read: ") + path)
        done += count
    _ = sys_close(fd)
    var value = String(
        StringSlice(unsafe_from_utf8=Span(unsafe_ptr=buf, length=size))
    )
    buf.free()
    return value^


def _read_bytes(path: String) raises -> List[UInt8]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("MiniMax H3 source image cannot open: ") + path)
    var size = file_size(fd)
    if size <= 0:
        _ = sys_close(fd)
        raise Error(String("MiniMax H3 source image is empty: ") + path)
    var buf = alloc[UInt8](size)
    var done = 0
    while done < size:
        var count = sys_pread(fd, buf + done, size - done, done)
        if count <= 0:
            buf.free()
            _ = sys_close(fd)
            raise Error(String("MiniMax H3 source image short read: ") + path)
        done += count
    _ = sys_close(fd)
    var out = List[UInt8](capacity=size)
    for index in range(size):
        out.append(buf[index])
    buf.free()
    return out^


def _u16_be(data: List[UInt8], offset: Int) -> Int:
    return (Int(data[offset]) << 8) | Int(data[offset + 1])


def _u16_le(data: List[UInt8], offset: Int) -> Int:
    return Int(data[offset]) | (Int(data[offset + 1]) << 8)


def _u24_le(data: List[UInt8], offset: Int) -> Int:
    return (
        Int(data[offset])
        | (Int(data[offset + 1]) << 8)
        | (Int(data[offset + 2]) << 16)
    )


@fieldwise_init
struct _ImageSize(Copyable, Movable):
    var width: Int
    var height: Int


def _positive_image_size(width: Int, height: Int, path: String) raises -> _ImageSize:
    if width <= 0 or height <= 0:
        raise Error(String("MiniMax H3 source image has invalid dimensions: ") + path)
    return _ImageSize(width, height)


def _image_dimensions(path: String, relative_path: String) raises -> _ImageSize:
    """Read original PNG/JPEG/WebP dimensions without decoding pixels."""
    var data = _read_bytes(path)
    var lower = String(relative_path.lower())
    if lower.endswith(String(".png")):
        if (
            len(data) < 24
            or data[0] != UInt8(0x89) or data[1] != UInt8(0x50)
            or data[2] != UInt8(0x4E) or data[3] != UInt8(0x47)
            or data[12] != UInt8(0x49) or data[13] != UInt8(0x48)
            or data[14] != UInt8(0x44) or data[15] != UInt8(0x52)
        ):
            raise Error(String("MiniMax H3 source PNG header is malformed: ") + path)
        var width = (
            (Int(data[16]) << 24) | (Int(data[17]) << 16)
            | (Int(data[18]) << 8) | Int(data[19])
        )
        var height = (
            (Int(data[20]) << 24) | (Int(data[21]) << 16)
            | (Int(data[22]) << 8) | Int(data[23])
        )
        return _positive_image_size(width, height, path)

    if lower.endswith(String(".jpg")) or lower.endswith(String(".jpeg")):
        if len(data) < 4 or data[0] != UInt8(0xFF) or data[1] != UInt8(0xD8):
            raise Error(String("MiniMax H3 source JPEG header is malformed: ") + path)
        var pos = 2
        while pos + 3 < len(data):
            while pos < len(data) and data[pos] != UInt8(0xFF):
                pos += 1
            while pos < len(data) and data[pos] == UInt8(0xFF):
                pos += 1
            if pos >= len(data):
                break
            var marker = Int(data[pos])
            pos += 1
            if marker == 0xD9 or marker == 0xDA:
                break
            if marker == 0x01 or (marker >= 0xD0 and marker <= 0xD7):
                continue
            if pos + 1 >= len(data):
                break
            var segment_length = _u16_be(data, pos)
            if segment_length < 2 or pos + segment_length > len(data):
                raise Error(String("MiniMax H3 source JPEG segment is malformed: ") + path)
            var is_sof = (
                (marker >= 0xC0 and marker <= 0xC3)
                or (marker >= 0xC5 and marker <= 0xC7)
                or (marker >= 0xC9 and marker <= 0xCB)
                or (marker >= 0xCD and marker <= 0xCF)
            )
            if is_sof:
                if segment_length < 7:
                    raise Error(String("MiniMax H3 source JPEG SOF is malformed: ") + path)
                return _positive_image_size(
                    _u16_be(data, pos + 5), _u16_be(data, pos + 3), path,
                )
            pos += segment_length
        raise Error(String("MiniMax H3 source JPEG has no size frame: ") + path)

    if lower.endswith(String(".webp")):
        if (
            len(data) < 30
            or data[0] != UInt8(0x52) or data[1] != UInt8(0x49)
            or data[2] != UInt8(0x46) or data[3] != UInt8(0x46)
            or data[8] != UInt8(0x57) or data[9] != UInt8(0x45)
            or data[10] != UInt8(0x42) or data[11] != UInt8(0x50)
        ):
            raise Error(String("MiniMax H3 source WebP header is malformed: ") + path)
        var chunk = String("")
        for index in range(12, 16):
            chunk += chr(Int(data[index]))
        if chunk == String("VP8X"):
            return _positive_image_size(
                _u24_le(data, 24) + 1, _u24_le(data, 27) + 1, path,
            )
        if chunk == String("VP8L"):
            if data[20] != UInt8(0x2F):
                raise Error(String("MiniMax H3 source WebP lossless header is malformed: ") + path)
            var width = 1 + Int(data[21]) + ((Int(data[22]) & 0x3F) << 8)
            var height = 1 + ((Int(data[22]) & 0xC0) >> 6) \
                + (Int(data[23]) << 2) + ((Int(data[24]) & 0x0F) << 10)
            return _positive_image_size(width, height, path)
        if chunk == String("VP8 "):
            if (
                data[23] != UInt8(0x9D) or data[24] != UInt8(0x01)
                or data[25] != UInt8(0x2A)
            ):
                raise Error(String("MiniMax H3 source WebP lossy header is malformed: ") + path)
            return _positive_image_size(
                _u16_le(data, 26) & 0x3FFF,
                _u16_le(data, 28) & 0x3FFF,
                path,
            )
        raise Error(String("MiniMax H3 source WebP chunk is unsupported: ") + path)

    raise Error(String("MiniMax H3 source image extension is unsupported: ") + path)


def _receipt_line(relative_path: String, digest: String) -> String:
    # Exact GNU sha256sum spelling used by the measured source receipt:
    #   <64 lowercase hex><two spaces><./relative-name>\n
    return (
        String(digest.removeprefix(String("sha256:")))
        + String("  ./") + relative_path + String("\n")
    )


def _pair_fingerprint(
    image_relative_path: String,
    image_sha256: String,
    caption_relative_path: String,
    caption_sha256: String,
) -> String:
    return minimax_h3_sha256_text(
        String("serenity.minimax_h3.source_pair.v1\n")
        + String("image=") + image_relative_path + String("\n")
        + String("image_sha256=") + image_sha256 + String("\n")
        + String("caption=") + caption_relative_path + String("\n")
        + String("caption_sha256=") + caption_sha256 + String("\n")
    )


def intake_minimax_h3_source_dataset(
    dataset_identity: String,
    physical_path: String,
    trigger: String,
    expected_receipt: String = String(""),
) raises -> MiniMaxH3SourceReceipt:
    """Enumerate and receipt a top-level still-image dataset before device use.

    Extra non-pair sidecars (for example authored `.json` and run logs) are not
    part of the receipt.  Orphan `.txt` files are likewise not training pairs;
    every enumerated image itself must have exactly the lowercase same-stem
    `.txt` caption.  Known unsupported image formats fail loud.
    """
    if dataset_identity != String(MINIMAX_H3_INTAKE_DATASET_IDENTITY):
        raise Error(
            String("MiniMax H3 source identity must be exactly ")
            + String(MINIMAX_H3_INTAKE_DATASET_IDENTITY)
        )
    if physical_path.byte_length() == 0 or physical_path.as_bytes()[0] != 0x2F:
        raise Error("MiniMax H3 source physical path must be absolute")
    if trigger.byte_length() == 0:
        raise Error("MiniMax H3 source trigger report requires a nonempty trigger")

    var entries = listdir(physical_path)
    var images = List[String]()
    for entry in entries:
        var name = String(entry)
        if _extension_length(name) > 0:
            images.append(name)
        elif _is_unsupported_image(name):
            raise Error(String("MiniMax H3 source unsupported image format: ") + name)
    _sort_strings(images)
    if len(images) == 0:
        raise Error("MiniMax H3 source contains no supported top-level images")

    var stems = List[String]()
    var samples = List[MiniMaxH3SourceSample]()
    var trigger_count = 0
    var receipt_lines = List[String]()
    for image_name in images:
        var ext_len = _extension_length(image_name)
        var stem = _prefix_bytes(image_name, image_name.byte_length() - ext_len)
        if _contains(stems, stem):
            raise Error(String("MiniMax H3 source duplicate image stem: ") + stem)
        stems.append(stem)
        var caption_name = stem + String(".txt")
        var caption_matches = 0
        for entry in entries:
            var candidate = String(entry)
            if String(candidate.lower()) == String(caption_name.lower()):
                caption_matches += 1
                if candidate != caption_name:
                    raise Error(
                        String("MiniMax H3 source caption must be exact same-stem .txt: ")
                        + candidate
                    )
        if caption_matches == 0:
            raise Error(String("MiniMax H3 source missing caption for ") + image_name)
        if caption_matches != 1:
            raise Error(String("MiniMax H3 source duplicate captions for stem ") + stem)

        var image_path = physical_path + String("/") + image_name
        var caption_path = physical_path + String("/") + caption_name
        var caption = _read_authored_utf8(caption_path)
        var image_sha256 = minimax_h3_sha256_file(image_path)
        var caption_sha256 = minimax_h3_sha256_file(caption_path)
        var pair_fingerprint = _pair_fingerprint(
            image_name, image_sha256, caption_name, caption_sha256,
        )
        var dimensions = _image_dimensions(image_path, image_name)
        var present = caption.find(trigger) >= 0
        if present:
            trigger_count += 1
        receipt_lines.append(_receipt_line(image_name, image_sha256))
        receipt_lines.append(_receipt_line(caption_name, caption_sha256))
        samples.append(
            MiniMaxH3SourceSample(
                stem^,
                image_name,
                caption_name,
                image_sha256^,
                caption_sha256^,
                pair_fingerprint^,
                dimensions.width,
                dimensions.height,
                caption^,
                present,
            )
        )

    # Compatibility receipt, byte-for-byte equivalent to:
    #   for img in ./*.jpg ./*.png; do stem=${img%.*};
    #     sha256sum "$img" "$stem.txt"; done | LC_ALL=C sort | sha256sum
    # Generalized enumeration additionally admits JPEG/WEBP while retaining the
    # exact sha256sum line framing and C-byte lexical sort.
    _sort_strings(receipt_lines)
    var receipt_stream = String("")
    for line in receipt_lines:
        receipt_stream += line
    var receipt = minimax_h3_sha256_text(receipt_stream)
    if expected_receipt.byte_length() > 0 and receipt != expected_receipt:
        raise Error(
            String("MiniMax H3 source receipt mismatch: expected ")
            + expected_receipt + String(", got ") + receipt
        )
    return MiniMaxH3SourceReceipt(
        String(MINIMAX_H3_SOURCE_RECEIPT_SCHEMA),
        dataset_identity,
        physical_path,
        trigger,
        trigger_count,
        receipt^,
        samples^,
    )
