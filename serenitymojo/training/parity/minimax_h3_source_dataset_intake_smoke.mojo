# Host-only MiniMax-H3 source-dataset intake/receipt gate.
# No cache, model, DeviceContext, or training surface is imported.

from std.memory import alloc
from std.sys import argv

from serenitymojo.io.ffi import (
    BytePtr,
    O_CREAT,
    O_TRUNC,
    O_WRONLY,
    sys_close,
    sys_mkdirs,
    sys_open,
    sys_pwrite,
    sys_remove,
)
from serenitymojo.training.minimax_h3.source_dataset import (
    MINIMAX_H3_SOURCE_RECEIPT_SCHEMA,
    intake_minimax_h3_source_dataset,
)


comptime ROOT = "/tmp/serenity_h3_source_intake_v1"
comptime GOOD = ROOT + "/good"
comptime MISSING = ROOT + "/missing"
comptime DUPLICATE = ROOT + "/duplicate"
comptime UNSUPPORTED = ROOT + "/unsupported"
comptime FIXTURE_RECEIPT = (
    "sha256:aa8f4006c5e0db59339d570d2ea7b71a5c4f586b571c609efb00556725a80b0c"
)
comptime REAL_RECEIPT = (
    "sha256:34952653c587db734f673dbac6c8b513e5a9ed2741e35bd78d4e47c47c16be7b"
)


def _require(condition: Bool, message: String) raises:
    if not condition:
        raise Error(message)


def _write(path: String, value: String) raises:
    var fd = sys_open(path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
    if fd < 0:
        raise Error(String("fixture cannot open ") + path)
    var size = value.byte_length()
    var buf = alloc[UInt8](size if size > 0 else 1)
    var source = value.as_bytes()
    for index in range(size):
        buf[index] = source[index]
    var wrote = sys_pwrite(
        fd, BytePtr(unsafe_from_address=Int(buf)), size, 0,
    )
    buf.free()
    _ = sys_close(fd)
    if wrote != size:
        raise Error(String("fixture short write: ") + path)


def _write_bytes(path: String, data: List[UInt8]) raises:
    var fd = sys_open(path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
    if fd < 0:
        raise Error(String("fixture cannot open ") + path)
    var buf = alloc[UInt8](len(data) if len(data) > 0 else 1)
    for index in range(len(data)):
        buf[index] = data[index]
    var wrote = sys_pwrite(
        fd, BytePtr(unsafe_from_address=Int(buf)), len(data), 0,
    )
    buf.free()
    _ = sys_close(fd)
    if wrote != len(data):
        raise Error(String("fixture short write: ") + path)


def _jpeg_header() -> List[UInt8]:
    return [
        UInt8(0xFF), UInt8(0xD8), UInt8(0xFF), UInt8(0xC0),
        UInt8(0x00), UInt8(0x11), UInt8(0x08),
        UInt8(0x00), UInt8(0x10), UInt8(0x00), UInt8(0x20),
        UInt8(0x03), UInt8(0x01), UInt8(0x11), UInt8(0x00),
        UInt8(0x02), UInt8(0x11), UInt8(0x00),
        UInt8(0x03), UInt8(0x11), UInt8(0x00),
    ]


def _png_header() -> List[UInt8]:
    return [
        UInt8(0x89), UInt8(0x50), UInt8(0x4E), UInt8(0x47),
        UInt8(0x0D), UInt8(0x0A), UInt8(0x1A), UInt8(0x0A),
        UInt8(0), UInt8(0), UInt8(0), UInt8(13),
        UInt8(0x49), UInt8(0x48), UInt8(0x44), UInt8(0x52),
        UInt8(0), UInt8(0), UInt8(0), UInt8(48),
        UInt8(0), UInt8(0), UInt8(0), UInt8(24),
    ]


def _webp_header() -> List[UInt8]:
    return [
        UInt8(0x52), UInt8(0x49), UInt8(0x46), UInt8(0x46),
        UInt8(22), UInt8(0), UInt8(0), UInt8(0),
        UInt8(0x57), UInt8(0x45), UInt8(0x42), UInt8(0x50),
        UInt8(0x56), UInt8(0x50), UInt8(0x38), UInt8(0x58),
        UInt8(10), UInt8(0), UInt8(0), UInt8(0),
        UInt8(0), UInt8(0), UInt8(0), UInt8(0),
        UInt8(63), UInt8(0), UInt8(0),
        UInt8(31), UInt8(0), UInt8(0),
    ]


def _prepare_fixtures() raises:
    _ = sys_mkdirs(String(GOOD))
    _ = sys_mkdirs(String(MISSING))
    _ = sys_mkdirs(String(DUPLICATE))
    _ = sys_mkdirs(String(UNSUPPORTED))
    _write_bytes(String(GOOD) + String("/alpha.jpg"), _jpeg_header())
    _write(
        String(GOOD) + String("/alpha.txt"),
        String("authored caption with spaces  \r\n"),
    )
    _write_bytes(String(GOOD) + String("/zeta.png"), _png_header())
    _write(
        String(GOOD) + String("/zeta.txt"),
        String("vrtlEri2, authored\nsecond line\n"),
    )
    _write_bytes(String(GOOD) + String("/beta.JPEG"), _jpeg_header())
    _write(String(GOOD) + String("/beta.txt"), String("vrtlEri2 beta\n"))
    _write_bytes(String(GOOD) + String("/omega.WEBP"), _webp_header())
    _write(String(GOOD) + String("/omega.txt"), String("omega authored\n"))
    # Ignored sidecars and orphan non-pair captions are outside the receipt.
    _write(String(GOOD) + String("/alpha.json"), String("{\"sidecar\":true}\n"))
    _write(String(GOOD) + String("/notes.txt"), String("not an image pair\n"))

    _write(String(MISSING) + String("/alone.webp"), String("image\n"))

    _write_bytes(String(DUPLICATE) + String("/dup.jpg"), _jpeg_header())
    _write_bytes(String(DUPLICATE) + String("/dup.png"), _png_header())
    _write(String(DUPLICATE) + String("/dup.txt"), String("caption\n"))

    _write(String(UNSUPPORTED) + String("/bad.bmp"), String("bitmap\n"))
    _write(String(UNSUPPORTED) + String("/bad.txt"), String("caption\n"))


def _cleanup():
    for path in [
        String(GOOD) + String("/alpha.jpg"),
        String(GOOD) + String("/alpha.txt"),
        String(GOOD) + String("/zeta.png"),
        String(GOOD) + String("/zeta.txt"),
        String(GOOD) + String("/beta.JPEG"),
        String(GOOD) + String("/beta.txt"),
        String(GOOD) + String("/omega.WEBP"),
        String(GOOD) + String("/omega.txt"),
        String(GOOD) + String("/alpha.json"),
        String(GOOD) + String("/notes.txt"),
        String(MISSING) + String("/alone.webp"),
        String(DUPLICATE) + String("/dup.jpg"),
        String(DUPLICATE) + String("/dup.png"),
        String(DUPLICATE) + String("/dup.txt"),
        String(UNSUPPORTED) + String("/bad.bmp"),
        String(UNSUPPORTED) + String("/bad.txt"),
    ]:
        _ = sys_remove(path)
    _ = sys_remove(String(GOOD))
    _ = sys_remove(String(MISSING))
    _ = sys_remove(String(DUPLICATE))
    _ = sys_remove(String(UNSUPPORTED))
    _ = sys_remove(String(ROOT))


def main() raises:
    var args = argv()
    if len(args) != 2:
        raise Error(
            "usage: minimax_h3_source_dataset_intake_smoke <physical-dataset-path>"
        )
    _cleanup()
    _prepare_fixtures()

    var fixture = intake_minimax_h3_source_dataset(
        String("eri_with_trigger"),
        String(GOOD),
        String("vrtlEri2"),
        String(FIXTURE_RECEIPT),
    )
    _require(
        fixture.schema == String(MINIMAX_H3_SOURCE_RECEIPT_SCHEMA),
        String("fixture schema mismatch"),
    )
    _require(len(fixture.samples) == 4, String("fixture pair count mismatch"))
    _require(
        fixture.samples[0].image_relative_path == String("alpha.jpg")
        and fixture.samples[1].image_relative_path == String("beta.JPEG")
        and fixture.samples[2].image_relative_path == String("omega.WEBP")
        and fixture.samples[3].image_relative_path == String("zeta.png"),
        String("fixture image order is not deterministic"),
    )
    _require(
        fixture.samples[0].caption
        == String("authored caption with spaces  \r\n"),
        String("authored caption bytes were rewritten"),
    )
    _require(
        fixture.samples[0].original_width == 32
        and fixture.samples[0].original_height == 16
        and fixture.samples[2].original_width == 64
        and fixture.samples[2].original_height == 32
        and fixture.samples[3].original_width == 48
        and fixture.samples[3].original_height == 24,
        String("source image dimensions were not read from exact headers"),
    )
    _require(
        fixture.samples[0].image_sha256
            == String("sha256:237f09ad0cec5bf75ad788b753aff391d584982f109e544724fdbe0e8193923c")
        and fixture.samples[0].pair_fingerprint.startswith(String("sha256:")),
        String("source pair content identity mismatch"),
    )
    _require(
        fixture.trigger_present_count == 2
        and not fixture.samples[0].trigger_present
        and fixture.samples[1].trigger_present
        and not fixture.samples[2].trigger_present
        and fixture.samples[3].trigger_present,
        String("fixture trigger-presence report mismatch"),
    )

    var rejected = False
    try:
        _ = intake_minimax_h3_source_dataset(
            String("eri2_with_trigger"), String(GOOD), String("vrtlEri2")
        )
    except:
        rejected = True
    _require(rejected, String("wrong logical identity was accepted"))

    rejected = False
    try:
        _ = intake_minimax_h3_source_dataset(
            String("eri_with_trigger"), String(MISSING), String("vrtlEri2")
        )
    except:
        rejected = True
    _require(rejected, String("missing caption was accepted"))

    rejected = False
    try:
        _ = intake_minimax_h3_source_dataset(
            String("eri_with_trigger"), String(DUPLICATE), String("vrtlEri2")
        )
    except:
        rejected = True
    _require(rejected, String("duplicate image stem was accepted"))

    rejected = False
    try:
        _ = intake_minimax_h3_source_dataset(
            String("eri_with_trigger"), String(UNSUPPORTED), String("vrtlEri2")
        )
    except:
        rejected = True
    _require(rejected, String("unsupported image pair was accepted"))

    _write(
        String(GOOD) + String("/zeta.txt"),
        String("vrtlEri2, mutated\nsecond line\n"),
    )
    rejected = False
    try:
        _ = intake_minimax_h3_source_dataset(
            String("eri_with_trigger"),
            String(GOOD),
            String("vrtlEri2"),
            String(FIXTURE_RECEIPT),
        )
    except:
        rejected = True
    _require(rejected, String("caption-byte mutation did not change receipt"))

    var real = intake_minimax_h3_source_dataset(
        String("eri_with_trigger"),
        String(args[1]),
        String("vrtlEri2"),
        String(REAL_RECEIPT),
    )
    _require(len(real.samples) == 118, String("measured pair count must be 118"))
    _require(
        real.trigger_present_count == 116,
        String("measured trigger-presence count must be 116"),
    )
    _cleanup()
    print("PASS MiniMax H3 source dataset intake/receipt host gate")
    print("  logical identity: eri_with_trigger")
    print("  measured pairs: 118; trigger present: 116")
    print("  receipt:", real.receipt)
    print("  DeviceContext/cache/training: NOT ENTERED")
