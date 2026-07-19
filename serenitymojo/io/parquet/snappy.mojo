# serenitymojo/io/parquet/snappy.mojo — raw Snappy decompression (100% Mojo).
#
# "Raw" Snappy is the format Parquet stores column pages in (NOT the framed
# stream format): a varint of the uncompressed length, then a sequence of
# elements. Each element's tag byte carries a 2-bit type in its low bits:
#   00 literal            — upper 6 bits are (len-1), or a 1..4 byte extension
#   01 copy, 1-byte off   — len = tag[2:5]+4 (4..11), off = tag[5:8]<<8 | next
#   02 copy, 2-byte off   — len = tag[2:]+1  (1..64), off = next 2 LE
#   03 copy, 4-byte off   — len = tag[2:]+1  (1..64), off = next 4 LE
# Copies may overlap the output tail (offset < length) — that is the RLE-style
# run, so we copy one byte at a time from already-written output.
#
# Reference: https://github.com/google/snappy/blob/main/format_description.txt


def _read_varint(src: List[UInt8], mut pos: Int) raises -> Int:
    """LEB128 varint, 7 bits/byte, low byte first, MSB = continuation."""
    var value = 0
    var shift = 0
    while True:
        if pos >= len(src):
            raise Error("snappy: truncated varint")
        var b = Int(src[pos])
        pos += 1
        value |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            break
        shift += 7
    return value


def snappy_decompress(src: List[UInt8]) raises -> List[UInt8]:
    """Decompress a raw-Snappy block into its original bytes."""
    var pos = 0
    var ulen = _read_varint(src, pos)
    var out = List[UInt8](capacity=ulen)

    while pos < len(src):
        var tag = Int(src[pos])
        pos += 1
        var etype = tag & 0x3

        if etype == 0:
            # ── literal ──────────────────────────────────────────────────
            var lenm1 = tag >> 2
            if lenm1 >= 60:
                # 60/61/62/63 → (len-1) is in the next 1/2/3/4 bytes, LE.
                var nbytes = lenm1 - 59
                var v = 0
                for i in range(nbytes):
                    if pos >= len(src):
                        raise Error("snappy: truncated literal length")
                    v |= Int(src[pos]) << (8 * i)
                    pos += 1
                lenm1 = v
            var length = lenm1 + 1
            if pos + length > len(src):
                raise Error("snappy: literal overruns input")
            for i in range(length):
                out.append(src[pos + i])
            pos += length
        else:
            # ── copy ─────────────────────────────────────────────────────
            var length: Int
            var offset: Int
            if etype == 1:
                length = ((tag >> 2) & 0x7) + 4
                if pos >= len(src):
                    raise Error("snappy: truncated 1-byte copy")
                offset = ((tag >> 5) << 8) | Int(src[pos])
                pos += 1
            elif etype == 2:
                length = (tag >> 2) + 1
                if pos + 2 > len(src):
                    raise Error("snappy: truncated 2-byte copy")
                offset = Int(src[pos]) | (Int(src[pos + 1]) << 8)
                pos += 2
            else:  # etype == 3
                length = (tag >> 2) + 1
                if pos + 4 > len(src):
                    raise Error("snappy: truncated 4-byte copy")
                offset = (
                    Int(src[pos])
                    | (Int(src[pos + 1]) << 8)
                    | (Int(src[pos + 2]) << 16)
                    | (Int(src[pos + 3]) << 24)
                )
                pos += 4
            if offset <= 0 or offset > len(out):
                raise Error("snappy: copy offset out of range")
            var start = len(out) - offset
            # One byte at a time: overlapping copies (offset < length) must see
            # the bytes this same copy just appended.
            for i in range(length):
                out.append(out[start + i])

    if len(out) != ulen:
        raise Error("snappy: decoded length disagrees with header")
    return out^
