# serenitymojo/io/parquet/extract.mojo — extract a Parquet dataset shard into the
# folder layout the serenity/SerenityTrainer cache builders consume. Two shard shapes
# (both supported, mirroring SimpleTuner's parquet backend):
#
#   (a) INLINE-BLOB shard (HF-style): a BYTE_ARRAY column holds the image/video
#       bytes. → write <out_dir>/<NNNNN>.<ext> (magic-sniffed) + <NNNNN>.txt.
#   (b) METADATA shard (SimpleTuner-style): a filename column + caption column;
#       images already on disk. → write a <stem>.txt caption sidecar next to each
#       image (in --image-dir), NO blob path. Nothing is decoded from pixels.
#
# Column mapping is CONFIG, not guessed (SimpleTuner semantics):
#   --caption-column (default: first of caption/text/prompt present; loud report)
#   --fallback-caption-column (used when the primary caption is NULL/empty)
#   --filename-column (its presence selects METADATA mode)
#   --media-column / --media-col (the blob column for inline mode)
# Empty-after-fallback rows are skipped and counted (SimpleTuner drops them).
#
# Build (native, -O2 per repo build-memory finding):
#   pixi run mojo build --optimization-level 2 -I . \
#       serenitymojo/io/parquet/extract.mojo -o output/bin/parquet_extract

from std.sys import argv

from serenitymojo.io.parquet.reader import (
    read_file_bytes, parse_metadata, read_byte_array_column,
    read_byte_array_column_aligned, AlignedColumn, ColumnChunkMeta, ParquetMeta,
)
from serenitymojo.io.ffi import (
    BytePtr, O_RDONLY, O_WRONLY, O_CREAT, O_TRUNC,
    sys_open, sys_pwrite, sys_close, sys_mkdirs,
)


# ── media magic sniff ────────────────────────────────────────────────────────
def _ext_for(v: List[UInt8]) -> String:
    var n = len(v)
    if n >= 3 and Int(v[0]) == 0xFF and Int(v[1]) == 0xD8 and Int(v[2]) == 0xFF:
        return "jpg"
    if n >= 8 and Int(v[0]) == 0x89 and Int(v[1]) == ord("P") and Int(v[2]) == ord("N") and Int(v[3]) == ord("G"):
        return "png"
    if n >= 6 and Int(v[0]) == ord("G") and Int(v[1]) == ord("I") and Int(v[2]) == ord("F"):
        return "gif"
    if n >= 12 and Int(v[0]) == ord("R") and Int(v[1]) == ord("I") and Int(v[2]) == ord("F") and Int(v[3]) == ord("F") and Int(v[8]) == ord("W") and Int(v[9]) == ord("E") and Int(v[10]) == ord("B") and Int(v[11]) == ord("P"):
        return "webp"
    if n >= 12 and Int(v[4]) == ord("f") and Int(v[5]) == ord("t") and Int(v[6]) == ord("y") and Int(v[7]) == ord("p"):
        return "mp4"
    if n >= 4 and Int(v[0]) == 0x1A and Int(v[1]) == 0x45 and Int(v[2]) == 0xDF and Int(v[3]) == 0xA3:
        return "webm"
    return "bin"


# ── small string/path helpers (ASCII-path assumption, matches io/disk_check) ──
def _zpad(i: Int, width: Int) -> String:
    var s = String(i)
    while s.byte_length() < width:
        s = "0" + s
    return s


def _ends_with(name: String, suffix: String) -> Bool:
    var nb = name.as_bytes()
    var sb = suffix.as_bytes()
    if len(nb) < len(sb):
        return False
    var off = len(nb) - len(sb)
    for i in range(len(sb)):
        if nb[off + i] != sb[i]:
            return False
    return True


def _bytes_to_path(v: List[UInt8]) -> String:
    # For filenames (assumed ASCII path bytes) — byte→char, as io/disk_check does.
    var s = String("")
    for i in range(len(v)):
        s += chr(Int(v[i]))
    return s^


def _basename(s: String) -> String:
    var b = s.as_bytes()
    var cut = -1
    for i in range(len(b)):
        if b[i] == UInt8(47):  # '/'
            cut = i
    if cut < 0:
        return s
    var out_dir = String("")
    for i in range(cut + 1, len(b)):
        out_dir += chr(Int(b[i]))
    return out_dir^


def _strip_ext(s: String) -> String:
    var b = s.as_bytes()
    var dot = -1
    for i in range(len(b)):
        if b[i] == UInt8(46):  # '.'
            dot = i
    if dot <= 0:  # no extension (or a leading-dot name) → keep as-is
        return s
    var out_dir = String("")
    for i in range(dot):
        out_dir += chr(Int(b[i]))
    return out_dir^


# ── file I/O via io/ffi (never the builtin open) ─────────────────────────────
def _write_bytes(path: String, data: List[UInt8]) raises:
    var fd = sys_open(path, O_WRONLY | O_CREAT | O_TRUNC, Int32(0o644))
    if fd < 0:
        raise Error("parquet: cannot open for write '" + path + "'")
    var n = len(data)
    if n > 0:
        var bp = BytePtr(unsafe_from_address=Int(data.unsafe_ptr()))
        var done = 0
        while done < n:
            var w = sys_pwrite(fd, bp + done, n - done, done)
            if w <= 0:
                break
            done += w
        _ = sys_close(fd)
        if done != n:
            raise Error("parquet: short write on '" + path + "'")
    else:
        _ = sys_close(fd)


def _exists(path: String) -> Bool:
    var fd = sys_open(path, O_RDONLY, 0)
    if fd < 0:
        return False
    _ = sys_close(fd)
    return True


def _arg(a: List[String], flag: String, default: String) raises -> String:
    for i in range(1, len(a) - 1):
        if a[i] == flag:
            return a[i + 1]
    return default


def _has_flag(a: List[String], flag: String) -> Bool:
    for i in range(1, len(a)):
        if a[i] == flag:
            return True
    return False


# ── caption column resolution (config, not guessed; loud report) ─────────────
def _resolve_caption_col(meta: ParquetMeta, explicit: String) raises -> Int:
    if explicit != "":
        return meta.col_index(explicit)
    var defaults = ["caption", "text", "prompt"]
    for d in defaults:
        for c in range(len(meta.columns)):
            if meta.columns[c].name == d:
                print("caption column: '" + d + "' (auto-matched from caption/text/prompt)")
                return c
    return -1


# ── manifest JSON (byte-level escape so UTF-8 captions survive) ───────────────
def _json_append(mut buf: List[UInt8], s: String):
    for i in range(len(s.as_bytes())):
        buf.append(s.as_bytes()[i])


def _json_append_escaped(mut buf: List[UInt8], v: List[UInt8]):
    for i in range(len(v)):
        var c = Int(v[i])
        if c == 0x22:  # "
            buf.append(UInt8(0x5C)); buf.append(UInt8(0x22))
        elif c == 0x5C:  # backslash
            buf.append(UInt8(0x5C)); buf.append(UInt8(0x5C))
        elif c == 0x0A:  # \n
            buf.append(UInt8(0x5C)); buf.append(UInt8(0x6E))
        elif c == 0x0D:  # \r
            buf.append(UInt8(0x5C)); buf.append(UInt8(0x72))
        elif c == 0x09:  # \t
            buf.append(UInt8(0x5C)); buf.append(UInt8(0x74))
        elif c < 0x20:  # other control → \u00XX
            var hexd = "0123456789abcdef"
            buf.append(UInt8(0x5C)); buf.append(UInt8(0x75))
            buf.append(UInt8(0x30)); buf.append(UInt8(0x30))
            buf.append(hexd.as_bytes()[(c >> 4) & 0xF])
            buf.append(hexd.as_bytes()[c & 0xF])
        else:
            buf.append(v[i])  # UTF-8 continuation bytes pass through verbatim


def main() raises:
    var raw = argv()
    var a = List[String]()
    for i in range(len(raw)):
        a.append(String(raw[i]))
    var parquet = _arg(a, "--parquet", "")
    var out_dir = _arg(a, "--out", "")
    if parquet == "" or out_dir == "":
        print("usage: parquet_extract --parquet FILE --out_dir DIR")
        print("  inline-blob shard:  [--media-column NAME] [--caption-column NAME] [--prefix S] [--pad N] [--limit N]")
        print("  metadata shard:     --filename-column NAME [--image-dir DIR] [--caption-column NAME] [--fallback-caption-column NAME] [--overwrite] [--limit N]")
        return

    var caption_col = _arg(a, "--caption-column", _arg(a, "--caption-col", ""))
    var fallback_col = _arg(a, "--fallback-caption-column", "")
    var filename_col = _arg(a, "--filename-column", "")
    var media_col = _arg(a, "--media-column", _arg(a, "--media-col", ""))
    var image_dir = _arg(a, "--image-dir", "")
    var prefix = _arg(a, "--prefix", "")
    var pad = Int(_arg(a, "--pad", "5"))
    var limit = Int(_arg(a, "--limit", "0"))
    var overwrite = _has_flag(a, "--overwrite")

    var data = read_file_bytes(parquet)
    var meta = parse_metadata(data)
    print("parquet:", parquet, "| rows:", meta.num_rows, "| columns:", len(meta.columns))

    if filename_col != "":
        _run_metadata(data, meta, out_dir, filename_col, caption_col, fallback_col, image_dir, limit, overwrite)
    else:
        _run_inline(data, meta, out_dir, media_col, caption_col, prefix, pad, limit)


# ── (a) inline-blob shard ─────────────────────────────────────────────────────
def _run_inline(
    data: List[UInt8], meta: ParquetMeta, out_dir: String,
    media_col: String, caption_col: String, prefix: String, pad: Int, limit: Int,
) raises:
    var media_idx = -1
    if media_col != "":
        media_idx = meta.col_index(media_col)
    else:
        for c in range(len(meta.columns)):
            if _ends_with(meta.columns[c].name, "_bytes"):
                media_idx = c
                break
    if media_idx < 0:
        print("could not auto-detect a media column (no column ends in '_bytes'); pass --media-column NAME")
        return

    var caption_idx = -1
    if caption_col != "":
        caption_idx = meta.col_index(caption_col)
    else:
        caption_idx = _resolve_caption_col(meta, "")
        if caption_idx < 0:
            for c in range(len(meta.columns)):
                if c != media_idx:
                    caption_idx = c
                    break

    print("mode: inline-blob | media column:", meta.columns[media_idx].name)
    if caption_idx >= 0:
        print("caption column:", meta.columns[caption_idx].name)

    _ = sys_mkdirs(out_dir)
    var media = read_byte_array_column(data, meta.columns[media_idx])
    var have_caps = caption_idx >= 0
    var caps = List[List[UInt8]]()
    if have_caps:
        caps = read_byte_array_column(data, meta.columns[caption_idx])

    var count = len(media)
    if limit > 0 and limit < count:
        count = limit

    var manifest = List[UInt8]()
    var written = 0
    for i in range(count):
        var stem = out_dir + "/" + prefix + _zpad(i, pad)
        var ext = _ext_for(media[i])
        var fname = prefix + _zpad(i, pad) + "." + ext
        _write_bytes(stem + "." + ext, media[i])
        if have_caps:
            _write_bytes(stem + ".txt", caps[i])
        written += 1
        _json_append(manifest, '{"file": "' + fname + '", "caption": "')
        if have_caps:
            _json_append_escaped(manifest, caps[i])
        _json_append(manifest, '"}\n')
    _write_bytes(out_dir + "/manifest.jsonl", manifest)
    print("wrote", written, "media files (+captions) +manifest.jsonl to", out_dir)


# ── (b) SimpleTuner-style metadata shard ─────────────────────────────────────
def _run_metadata(
    data: List[UInt8], meta: ParquetMeta, out_dir: String,
    filename_col: String, caption_col: String, fallback_col: String,
    image_dir: String, limit: Int, overwrite: Bool,
) raises:
    var fn_idx = meta.col_index(filename_col)
    var cap_idx = -1
    if caption_col != "":
        cap_idx = meta.col_index(caption_col)
        print("caption column:", caption_col)
    else:
        cap_idx = _resolve_caption_col(meta, "")
    if cap_idx < 0:
        print("no caption column found (tried caption/text/prompt); pass --caption-column NAME")
        return
    var fb_idx = -1
    if fallback_col != "":
        fb_idx = meta.col_index(fallback_col)

    print("mode: metadata | filename column:", filename_col, "| fallback:", fallback_col if fallback_col != "" else String("(none)"))

    var fnames = read_byte_array_column_aligned(data, meta.columns[fn_idx])
    var caps = read_byte_array_column_aligned(data, meta.columns[cap_idx])
    var have_fb = fb_idx >= 0
    var fbs = AlignedColumn(List[List[UInt8]](), List[Bool]())
    if have_fb:
        fbs = read_byte_array_column_aligned(data, meta.columns[fb_idx])

    var base = image_dir if image_dir != "" else out_dir
    _ = sys_mkdirs(base)
    if out_dir != base:
        _ = sys_mkdirs(out_dir)

    var count = meta.num_rows
    if limit > 0 and limit < count:
        count = limit

    var manifest = List[UInt8]()
    var written = 0
    var skipped_missing = 0   # empty caption after fallback (SimpleTuner drops)
    var skipped_exists = 0
    var skipped_nofile = 0    # null/empty filename
    for i in range(count):
        if not fnames.present[i] or len(fnames.values[i]) == 0:
            skipped_nofile += 1
            continue
        # caption with fallback: SimpleTuner uses fallback when primary null/empty
        var cap = caps.values[i].copy()
        if len(cap) == 0 and have_fb:
            cap = fbs.values[i].copy()
        if len(cap) == 0:
            skipped_missing += 1
            continue

        var stem = _strip_ext(_basename(_bytes_to_path(fnames.values[i])))
        var txt_path = base + "/" + stem + ".txt"
        if _exists(txt_path) and not overwrite:
            skipped_exists += 1
            continue
        _write_bytes(txt_path, cap)
        written += 1
        _json_append(manifest, '{"filename": "' + stem + '", "caption": "')
        _json_append_escaped(manifest, cap)
        _json_append(manifest, '"}\n')

    _write_bytes(out_dir + "/manifest.jsonl", manifest)
    print("wrote", written, ".txt sidecars to", base, "+manifest.jsonl to", out_dir)
    print("skipped:", skipped_missing, "no-caption |", skipped_exists, "already-exist |", skipped_nofile, "no-filename")
