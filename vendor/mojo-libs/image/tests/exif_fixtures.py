#!/usr/bin/env python3
# EXIF fixtures + round-trip checker, PIL oracle.
#
# Modes:
#   (default / "gen")  -> write reference EXIF byte blobs to /tmp/exif_fix/ for
#                         the Mojo parse_exif test:
#                           ref_le.exif  little-endian (II), with an ExifIFD
#                                        sub-IFD holding RATIONAL FNumber/etc.
#                           ref_be.exif  big-endian (MM), same tags.
#                         Both are bare TIFF streams (no "Exif\0\0" APP1 prefix).
#   "verify <path>"    -> load a Mojo-written EXIF blob with PIL and confirm the
#                         tags PIL reads back (the round-trip step).
#
# We hand-build the TIFF because PIL's Image.Exif.tobytes() does not emit a real
# ExifIFD sub-IFD (it flattens, and writes rationals as DOUBLE). Hand-building
# gives a genuine sub-IFD + RATIONAL fields to exercise the Mojo parser, and PIL
# reads our hand-built blobs back correctly (verified below).
import os
import sys
import struct
from PIL import Image
from PIL.ExifTags import Base as B, IFD

OUT = "/tmp/exif_fix"

MAKE = b"MojoCam\x00"
MODEL = b"X1\x00"
DT = b"2026:06:10 12:34:56\x00"


def _build(endian):
    """endian: '<' little (II) or '>' big (MM)."""
    bom = b"II" if endian == "<" else b"MM"
    hdr = bom + struct.pack(endian + "H", 42) + struct.pack(endian + "I", 8)

    n0 = 5                       # IFD0 entries (incl. ExifIFD pointer)
    ifd0_start = 8
    ifd0_size = 2 + n0 * 12 + 4
    exif_start = ifd0_start + ifd0_size
    n1 = 5                       # ExifIFD entries
    exif_size = 2 + n1 * 12 + 4
    data_start = exif_start + exif_size

    # data area: three RATIONALs (8 bytes each) then the out-of-line ASCII
    # payloads (MAKE and DT; MODEL is <=4 bytes so it is stored inline).
    exp_off = data_start          # ExposureTime 1/200
    fn_off = exp_off + 8          # FNumber 28/10
    fl_off = fn_off + 8           # FocalLength 50/1
    make_off = fl_off + 8
    model_off = 0                 # unused: MODEL stored inline
    dt_off = make_off + len(MAKE)

    def short_inline(val):
        # SHORT value occupies the first 2 bytes of the 4-byte value field;
        # the position is the same in the field for both endians since the
        # field is itself written in `endian`.
        return struct.pack(endian + "H", val) + b"\x00\x00"

    def ascii_field(tag, payload, off):
        # ASCII payloads <=4 bytes live INLINE in the value field; otherwise
        # the value field is an offset.
        if len(payload) <= 4:
            val = payload + b"\x00" * (4 - len(payload))
            return struct.pack(endian + "HHI", tag, 2, len(payload)) + val
        return struct.pack(endian + "HHII", tag, 2, len(payload), off)

    ifd0 = struct.pack(endian + "H", n0)
    ifd0 += ascii_field(0x010F, MAKE, make_off)                            # Make ASCII
    ifd0 += ascii_field(0x0110, MODEL, model_off)                          # Model ASCII (inline)
    ifd0 += struct.pack(endian + "HHI", 0x0112, 3, 1) + short_inline(6)     # Orientation
    ifd0 += ascii_field(0x0132, DT, dt_off)                                # DateTime ASCII
    ifd0 += struct.pack(endian + "HHII", 0x8769, 4, 1, exif_start)         # ExifIFD ptr
    ifd0 += struct.pack(endian + "I", 0)

    exif = struct.pack(endian + "H", n1)
    exif += struct.pack(endian + "HHII", 0x829A, 5, 1, exp_off)            # ExposureTime R
    exif += struct.pack(endian + "HHII", 0x829D, 5, 1, fn_off)             # FNumber R
    exif += struct.pack(endian + "HHI", 0x8827, 3, 1) + short_inline(100)  # ISO SHORT
    exif += struct.pack(endian + "HHII", 0x920A, 5, 1, fl_off)             # FocalLength R
    exif += struct.pack(endian + "HHI", 0xA002, 3, 1) + short_inline(640)  # PixelXDim SHORT
    exif += struct.pack(endian + "I", 0)

    data = struct.pack(endian + "II", 1, 200)
    data += struct.pack(endian + "II", 28, 10)
    data += struct.pack(endian + "II", 50, 1)
    data += MAKE + DT

    return hdr + ifd0 + exif + data


def _pil_check(label, blob):
    r = Image.Exif()
    r.load(blob)
    s = r.get_ifd(IFD.Exif)
    print(label, "tags PIL reads:")
    print("  Make=", r.get(B.Make.value))
    print("  Model=", r.get(B.Model.value))
    print("  Orientation=", r.get(B.Orientation.value))
    print("  DateTime=", r.get(B.DateTime.value))
    print("  FNumber=", s.get(B.FNumber.value))
    print("  ISO=", s.get(B.ISOSpeedRatings.value))
    print("  FocalLength=", s.get(B.FocalLength.value))
    assert r.get(B.Make.value) == "MojoCam", "Make"
    assert r.get(B.Model.value) == "X1", "Model"
    assert int(r.get(B.Orientation.value)) == 6, "Orientation"
    assert r.get(B.DateTime.value) == "2026:06:10 12:34:56", "DateTime"
    assert abs(float(s.get(B.FNumber.value)) - 2.8) < 1e-6, "FNumber"
    assert int(s.get(B.ISOSpeedRatings.value)) == 100, "ISO"
    assert abs(float(s.get(B.FocalLength.value)) - 50.0) < 1e-6, "FocalLength"


def gen():
    os.makedirs(OUT, exist_ok=True)
    le = _build("<")
    be = _build(">")
    with open(os.path.join(OUT, "ref_le.exif"), "wb") as f:
        f.write(le)
    with open(os.path.join(OUT, "ref_be.exif"), "wb") as f:
        f.write(be)
    _pil_check("REF_LE", le)
    _pil_check("REF_BE", be)
    print("FIXTURES_OK")


def verify(path):
    with open(path, "rb") as f:
        data = f.read()
    e = Image.Exif()
    e.load(data)
    make = e.get(B.Make.value)
    model = e.get(B.Model.value)
    orient = e.get(B.Orientation.value)
    dt = e.get(B.DateTime.value)
    print("PIL reads Mojo-built EXIF blob:")
    print("  Make=", make)
    print("  Model=", model)
    print("  Orientation=", orient)
    print("  DateTime=", dt)
    ok = (make == "MojoCam" and model == "X1" and int(orient) == 6
          and dt == "2026:06:10 12:00:00")
    print("PIL_ROUNDTRIP_OK" if ok else "PIL_ROUNDTRIP_FAIL")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "verify":
        verify(sys.argv[2])
    else:
        gen()
