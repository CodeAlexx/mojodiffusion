# pdf/reader.mojo
# Minimal PDF reader/parser — enough to read the classic-xref PDFs that this
# library (pdf/document.mojo) generates: a single `xref` table with subsections,
# a `trailer << ... >>`, and `startxref` pointing at the table.
#
# Access model: offset-based. We parse the xref to learn each object's byte
# offset, then on demand re-scan from "N 0 obj" to "endobj" to return that
# object's raw body bytes. We do NOT build a full object graph; that is enough
# for trailer lookups and FlateDecode stream extraction.
#
# NOT handled (by design — a separate component does these): cross-reference
# STREAMS (PDF 1.5+, "/Type /XRef"), object streams (/ObjStm), incremental
# updates with multiple xref sections chained via /Prev, and encryption.

from graphics.inflate import zlib_inflate
from pdf.md5 import md5
from pdf.rc4 import rc4
from pdf.aes import aes_cbc_decrypt, aes_decrypt_block, aes_encrypt_block, _key_expansion, _sbox
from pdf.sha256 import sha256


# ── small byte/string helpers ─────────────────────────────────────────────────
def _is_digit(c: Int) -> Bool:
    return c >= ord("0") and c <= ord("9")


def _is_ws(c: Int) -> Bool:
    # PDF whitespace: NUL, TAB, LF, FF, CR, SPACE.
    return c == 0 or c == 9 or c == 10 or c == 12 or c == 13 or c == 32


def _bytes_to_str(data: List[UInt8], start: Int, count: Int) -> String:
    var s = String("")
    var end = start + count
    if end > len(data):
        end = len(data)
    for i in range(start, end):
        s += chr(Int(data[i]))
    return s


def _match_at(data: List[UInt8], pos: Int, needle: String) -> Bool:
    var nb = needle.as_bytes()
    if pos < 0 or pos + len(nb) > len(data):
        return False
    for k in range(len(nb)):
        if data[pos + k] != nb[k]:
            return False
    return True


def _find_last(data: List[UInt8], needle: String) -> Int:
    # Last occurrence of `needle` in `data`; -1 if not found.
    var nb = needle.as_bytes()
    var nlen = len(nb)
    if nlen == 0 or nlen > len(data):
        return -1
    var i = len(data) - nlen
    while i >= 0:
        var ok = True
        for k in range(nlen):
            if data[i + k] != nb[k]:
                ok = False
                break
        if ok:
            return i
        i -= 1
    return -1


def _find_from(data: List[UInt8], needle: String, from_pos: Int) -> Int:
    # First occurrence of `needle` at or after from_pos; -1 if not found.
    var nb = needle.as_bytes()
    var nlen = len(nb)
    if nlen == 0:
        return -1
    var i = from_pos
    var last = len(data) - nlen
    while i <= last:
        var ok = True
        for k in range(nlen):
            if data[i + k] != nb[k]:
                ok = False
                break
        if ok:
            return i
        i += 1
    return -1


def _skip_ws(data: List[UInt8], pos: Int) -> Int:
    var i = pos
    while i < len(data) and _is_ws(Int(data[i])):
        i += 1
    return i


def _read_int(data: List[UInt8], pos: Int, mut out_val: Int) -> Int:
    # Parse a non-negative integer starting at pos (after skipping leading ws);
    # returns the index just past the digits. out_val gets the value.
    var i = _skip_ws(data, pos)
    var v = 0
    var any = False
    while i < len(data) and _is_digit(Int(data[i])):
        v = v * 10 + (Int(data[i]) - ord("0"))
        i += 1
        any = True
    out_val = v if any else -1
    return i


# ── helpers used by the xref-stream / ObjStm paths ─────────────────────────────
def _skip_ws_list(data: List[UInt8], pos: Int) -> Int:
    var i = pos
    while i < len(data) and _is_ws(Int(data[i])):
        i += 1
    return i


def _read_int_list(data: List[UInt8], pos: Int, mut out_val: Int) -> Int:
    var i = _skip_ws_list(data, pos)
    var v = 0
    var any = False
    while i < len(data) and _is_digit(Int(data[i])):
        v = v * 10 + (Int(data[i]) - ord("0"))
        i += 1
        any = True
    out_val = v if any else -1
    return i


def _str_to_int(s: String) -> Int:
    # Parse leading non-negative integer out of a token; -1 if none.
    var v = 0
    var any = False
    for j in range(len(s)):
        var c = ord(s[byte=j])
        if c >= ord("0") and c <= ord("9"):
            v = v * 10 + (c - ord("0"))
            any = True
        else:
            break
    return v if any else -1


def _bytes_index_of(data: List[UInt8], needle: String) -> Int:
    var nb = needle.as_bytes()
    var nlen = len(nb)
    if nlen == 0 or nlen > len(data):
        return -1
    var i = 0
    var last = len(data) - nlen
    while i <= last:
        var ok = True
        for k in range(nlen):
            if data[i + k] != nb[k]:
                ok = False
                break
        if ok:
            return i
        i += 1
    return -1


def _bytes_contains(data: List[UInt8], needle: String) -> Bool:
    return _bytes_index_of(data, needle) >= 0


def _find_matching_gt(data: List[UInt8], open_pos: Int) -> Int:
    # Given open_pos pointing at "<<", return the index just past the matching
    # ">>" (handles nesting). -1 if unbalanced.
    var depth = 0
    var i = open_pos
    while i + 1 < len(data):
        if Int(data[i]) == ord("<") and Int(data[i + 1]) == ord("<"):
            depth += 1
            i += 2
            continue
        if Int(data[i]) == ord(">") and Int(data[i + 1]) == ord(">"):
            depth -= 1
            i += 2
            if depth == 0:
                return i
            continue
        i += 1
    return -1


def _find_matching_gt_list(data: List[UInt8], open_pos: Int) -> Int:
    return _find_matching_gt(data, open_pos)


def _undo_png_predictor(data: List[UInt8], rowlen: Int, bpp: Int) raises -> List[UInt8]:
    # Reverse PNG row filters (RFC 2083 §6) used by /Predictor >= 10. Each row in
    # the input is rowlen+1 bytes: a 1-byte filter tag then rowlen data bytes.
    var stride = rowlen + 1
    var nrows = len(data) // stride
    var out = List[UInt8]()
    var prev = List[UInt8]()
    for _ in range(rowlen):
        prev.append(0)
    for r in range(nrows):
        var base = r * stride
        var ftype = Int(data[base])
        var cur = List[UInt8]()
        for c in range(rowlen):
            var raw = Int(data[base + 1 + c])
            var a = Int(cur[c - bpp]) if c >= bpp else 0   # left
            var b = Int(prev[c])                            # up
            var cc = Int(prev[c - bpp]) if c >= bpp else 0  # upper-left
            var val = 0
            if ftype == 0:        # None
                val = raw
            elif ftype == 1:      # Sub
                val = raw + a
            elif ftype == 2:      # Up
                val = raw + b
            elif ftype == 3:      # Average
                val = raw + ((a + b) // 2)
            elif ftype == 4:      # Paeth
                val = raw + _paeth(a, b, cc)
            else:
                raise Error("PdfReader: unknown PNG predictor filter " + String(ftype))
            cur.append(UInt8(val & 0xFF))
        for c in range(rowlen):
            out.append(cur[c])
        prev = cur^
    return out^


def _paeth(a: Int, b: Int, c: Int) -> Int:
    var p = a + b - c
    var pa = p - a if p >= a else a - p
    var pb = p - b if p >= b else b - p
    var pc = p - c if p >= c else c - p
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _undo_tiff_pred2(data: List[UInt8], rowlen: Int, bpp: Int) raises -> List[UInt8]:
    # TIFF predictor 2: horizontal differencing (no per-row filter byte). Rows
    # are exactly rowlen bytes.
    var nrows = len(data) // rowlen
    var out = List[UInt8]()
    for r in range(nrows):
        var base = r * rowlen
        var cur = List[UInt8]()
        for c in range(rowlen):
            var raw = Int(data[base + c])
            var left = Int(cur[c - bpp]) if c >= bpp else 0
            cur.append(UInt8((raw + left) & 0xFF))
        for c in range(rowlen):
            out.append(cur[c])
    return out^


# ── the reader ────────────────────────────────────────────────────────────────
struct PdfReader(Movable):
    var data: List[UInt8]
    # offsets[k] = byte offset of object number k ("k 0 obj"); -1 if free/absent.
    var offsets: List[Int]
    var size: Int          # trailer /Size (object count incl. free object 0)
    var trailer_start: Int # byte offset of "trailer" keyword (classic only; -1 for xref streams)
    var trailer_end: Int   # byte offset just past the trailer dict's ">>" (classic only)

    # ── xref-stream / object-stream support (PDF 1.5+) ──────────────────────────
    # When the xref representation is itself a stream (/Type /XRef) we do not have
    # a textual "trailer" keyword. Instead we keep the merged xref dict bytes
    # (from the newest XRef stream; /Prev sections are merged into the offset/
    # objstm tables but the dict we expose for trailer_value() is the newest one).
    var xref_is_stream: Bool
    var trailer_dict: List[UInt8]   # the XRef stream's dict bytes (incl. << >>)
    # For objects stored inside object streams: objstm_container[n] = the obj
    # number of the /ObjStm that holds object n (>=0), else -1; objstm_index[n] =
    # the 0-based index of object n within that stream, else -1. An object is
    # "compressed" iff objstm_container[n] >= 0 (in that case offsets[n] is -1).
    var objstm_container: List[Int]
    var objstm_index: List[Int]

    # ── encryption state (set by open_encrypted) ────────────────────────────────
    var is_encrypted: Bool       # True once a /Encrypt dict was authenticated.
    var enc_file_key: List[UInt8]  # the file encryption key (16 for V<=4, 32 for V5).
    var enc_cfm: Int             # 0 = RC4 (V1/V2), 1 = AESV2, 2 = AESV3.
    var enc_encrypt_num: Int     # object number of the /Encrypt dict (NOT decrypted).

    def __init__(out self):
        self.data = List[UInt8]()
        self.offsets = List[Int]()
        self.size = 0
        self.trailer_start = -1
        self.trailer_end = -1
        self.xref_is_stream = False
        self.trailer_dict = List[UInt8]()
        self.objstm_container = List[Int]()
        self.objstm_index = List[Int]()
        self.is_encrypted = False
        self.enc_file_key = List[UInt8]()
        self.enc_cfm = 0
        self.enc_encrypt_num = -1

    def open(mut self, path: String) raises:
        var f = open(path, "r")
        var d = f.read_bytes()
        f.close()
        self.data = List[UInt8]()
        for i in range(len(d)):
            self.data.append(d[i])
        # Dispatch: peek at what startxref points to. A classic table starts with
        # the literal "xref"; otherwise it is a cross-reference stream object.
        var sx = _find_last(self.data, "startxref")
        if sx < 0:
            raise Error("PdfReader: no startxref found")
        var xref_off = 0
        _ = _read_int(self.data, sx + len("startxref"), xref_off)
        if xref_off < 0 or xref_off >= len(self.data):
            raise Error("PdfReader: bad startxref offset")
        var p = _skip_ws(self.data, xref_off)
        if _match_at(self.data, p, "xref"):
            self.xref_is_stream = False
            self._parse_xref()
        else:
            self.xref_is_stream = True
            self._parse_xref_stream(xref_off)

    def _parse_xref(mut self) raises:
        # 1) startxref -> byte offset of the xref table.
        var sx = _find_last(self.data, "startxref")
        if sx < 0:
            raise Error("PdfReader: no startxref found")
        var xref_off = 0
        _ = _read_int(self.data, sx + len("startxref"), xref_off)
        if xref_off < 0 or xref_off >= len(self.data):
            raise Error("PdfReader: bad startxref offset")

        # 2) the table must begin with "xref" (classic). xref STREAMS unsupported.
        var p = _skip_ws(self.data, xref_off)
        if not _match_at(self.data, p, "xref"):
            raise Error(
                "PdfReader: classic 'xref' table not found at startxref "
                "(xref streams are not supported)"
            )
        p += len("xref")

        # 3) parse subsections: "<start> <count>\n" then `count` 20-byte entries.
        # First determine the max object number to size the offsets array; we do
        # this lazily by growing the array as we read entries.
        self.offsets = List[Int]()
        # grow offsets to index n with -1 fill
        while True:
            p = _skip_ws(self.data, p)
            # A subsection header is "<int> <int>"; the trailer keyword ends it.
            if _match_at(self.data, p, "trailer"):
                break
            var start_obj = 0
            var np = _read_int(self.data, p, start_obj)
            if start_obj < 0:
                break  # not a subsection header -> done
            var count = 0
            np = _read_int(self.data, np, count)
            if count < 0:
                raise Error("PdfReader: malformed xref subsection header")
            # advance to start of entries: skip to end of the header line
            var q = np
            while q < len(self.data) and Int(self.data[q]) != 10 and Int(self.data[q]) != 13:
                q += 1
            q = _skip_ws(self.data, q)
            # ensure offsets array can hold start_obj..start_obj+count-1
            var need = start_obj + count
            while len(self.offsets) < need:
                self.offsets.append(-1)
            # each entry: 10-digit offset, space, 5-digit gen, space, 'n'/'f'.
            for k in range(count):
                var off_val = 0
                var ep = _read_int(self.data, q, off_val)
                var gen_val = 0
                ep = _read_int(self.data, ep, gen_val)
                ep = _skip_ws(self.data, ep)
                var ty = ord("n")
                if ep < len(self.data):
                    ty = Int(self.data[ep])
                    ep += 1
                var obj_num = start_obj + k
                if ty == ord("n"):
                    self.offsets[obj_num] = off_val
                else:
                    self.offsets[obj_num] = -1
                q = ep
            p = q

        # 4) trailer << ... >>
        var tr = _find_from(self.data, "trailer", p - 8 if p > 8 else 0)
        if tr < 0:
            tr = _find_last(self.data, "trailer")
        if tr < 0:
            raise Error("PdfReader: no trailer found")
        self.trailer_start = tr
        var dd = _find_from(self.data, "<<", tr)
        if dd < 0:
            raise Error("PdfReader: trailer has no dict")
        # find matching ">>" (trailer dicts here are flat — no nested <<).
        var ee = _find_from(self.data, ">>", dd)
        if ee < 0:
            raise Error("PdfReader: trailer dict not closed")
        self.trailer_end = ee + 2

        # 5) /Size from the trailer.
        var sz = self.trailer_value("/Size")
        if len(sz) > 0:
            var v = 0
            for j in range(len(sz)):
                var c = ord(sz[byte=j])
                if c >= ord("0") and c <= ord("9"):
                    v = v * 10 + (c - ord("0"))
            self.size = v

    # ── cross-reference STREAM parsing (PDF 1.5+, /Type /XRef) ──────────────────
    def _grow_tables(mut self, need: Int):
        # Ensure offsets / objstm_* arrays can index 0..need-1 (fill with -1).
        while len(self.offsets) < need:
            self.offsets.append(-1)
        while len(self.objstm_container) < need:
            self.objstm_container.append(-1)
        while len(self.objstm_index) < need:
            self.objstm_index.append(-1)

    def _parse_xref_stream(mut self, first_off: Int) raises:
        # Follow the chain of XRef streams starting at byte offset first_off,
        # merging older (/Prev) sections beneath newer ones (newer wins). We keep
        # the newest stream's dict bytes for trailer_value().
        self.offsets = List[Int]()
        self.objstm_container = List[Int]()
        self.objstm_index = List[Int]()

        var cur = first_off
        var first = True
        # Guard against cyclic /Prev chains.
        var seen = List[Int]()
        while cur >= 0:
            # cycle check
            var dup = False
            for s in range(len(seen)):
                if seen[s] == cur:
                    dup = True
                    break
            if dup:
                break
            seen.append(cur)

            var dict_bytes = self._read_obj_dict_at(cur)
            if first:
                self.trailer_dict = dict_bytes.copy()
                self.trailer_start = -1
                self.trailer_end = -1
                first = False
            self._apply_xref_stream_section(cur, dict_bytes)

            # follow /Prev (older section) if present
            var prev_s = self._dict_value_str(dict_bytes, "/Prev")
            if len(prev_s) == 0:
                cur = -1
            else:
                var pv = _str_to_int(prev_s)
                cur = pv if pv >= 0 else -1

        # /Size from the newest dict.
        var sz = self._dict_value_str(self.trailer_dict, "/Size")
        if len(sz) > 0:
            self.size = _str_to_int(sz)

    def _read_obj_dict_at(self, off: Int) raises -> List[UInt8]:
        # At byte offset `off` there is "N G obj << ... >> stream ...". Return the
        # bytes of the dict including the outer "<<" ".. ">>" (nested dicts ok).
        var p = off
        var n0 = 0
        p = _read_int(self.data, p, n0)
        var g0 = 0
        p = _read_int(self.data, p, g0)
        p = _skip_ws(self.data, p)
        if not _match_at(self.data, p, "obj"):
            raise Error("PdfReader: 'obj' keyword not at xref-stream offset")
        p += 3
        var ds = _find_from(self.data, "<<", p)
        if ds < 0:
            raise Error("PdfReader: xref-stream object has no dict")
        var de = _find_matching_gt(self.data, ds)
        if de < 0:
            raise Error("PdfReader: xref-stream dict not closed")
        var out = List[UInt8]()
        for i in range(ds, de):
            out.append(self.data[i])
        return out^

    def _apply_xref_stream_section(mut self, obj_off: Int, dict_bytes: List[UInt8]) raises:
        # Inflate the XRef stream body, reverse any PNG/TIFF predictor, then walk
        # the entries using /W field widths and /Index sub-ranges. Only fill a
        # table slot if not already set (newer sections win because they are
        # applied first; here we test for "unset" before writing).
        var raw = self._read_stream_bytes_at(obj_off, dict_bytes)
        var data = self._maybe_undo_predictor(raw, dict_bytes)

        # /W [w1 w2 w3]
        var w = self._dict_int_array(dict_bytes, "/W")
        if len(w) != 3:
            raise Error("PdfReader: XRef /W must have 3 widths")
        var w1 = w[0]
        var w2 = w[1]
        var w3 = w[2]
        var row = w1 + w2 + w3
        if row <= 0:
            raise Error("PdfReader: XRef /W row width is zero")

        # /Index [start count start count ...]; default [0 Size].
        var index = self._dict_int_array(dict_bytes, "/Index")
        if len(index) == 0:
            var sz = self._dict_value_str(dict_bytes, "/Size")
            var size_v = _str_to_int(sz) if len(sz) > 0 else (len(data) // row)
            index = List[Int]()
            index.append(0)
            index.append(size_v)

        var pos = 0  # byte cursor within decoded entry data
        var pair = 0
        while pair + 1 < len(index):
            var start = index[pair]
            var count = index[pair + 1]
            pair += 2
            for k in range(count):
                if pos + row > len(data):
                    break
                var f1 = self._be(data, pos, w1)
                # Default type is 1 when w1 == 0 (per spec).
                var typ = f1 if w1 > 0 else 1
                var f2 = self._be(data, pos + w1, w2)
                var f3 = self._be(data, pos + w1 + w2, w3)
                pos += row
                var obj_num = start + k
                self._grow_tables(obj_num + 1)
                # Newer section already populated? Skip (older must not override).
                var already = (self.offsets[obj_num] >= 0) or (self.objstm_container[obj_num] >= 0)
                if already:
                    continue
                if typ == 1:
                    # f2 = byte offset, f3 = generation (ignored).
                    self.offsets[obj_num] = f2
                elif typ == 2:
                    # f2 = ObjStm object number, f3 = index within that stream.
                    self.objstm_container[obj_num] = f2
                    self.objstm_index[obj_num] = f3
                # typ == 0 -> free; leave -1.

    def _read_stream_bytes_at(self, obj_off: Int, dict_bytes: List[UInt8]) raises -> List[UInt8]:
        # Read & (if /FlateDecode) inflate the stream of the object at obj_off.
        # Uses /Length from the dict when present to bound the raw bytes; falls
        # back to scanning for "endstream".
        var sb = _find_from(self.data, "stream", obj_off)
        if sb < 0:
            raise Error("PdfReader: xref-stream object has no 'stream'")
        var ds = sb + len("stream")
        if ds < len(self.data) and Int(self.data[ds]) == 13:
            ds += 1
        if ds < len(self.data) and Int(self.data[ds]) == 10:
            ds += 1
        var de = -1
        var len_s = self._dict_value_str(dict_bytes, "/Length")
        if len(len_s) > 0:
            var L = _str_to_int(len_s)
            # /Length may be an indirect ref ("N 0 R"); detect by trailing R.
            var is_ref = False
            for ci in range(len(len_s)):
                if len_s[byte=ci] == "R":
                    is_ref = True
                    break
            if L >= 0 and not is_ref and ds + L <= len(self.data):
                de = ds + L
        if de < 0:
            de = _find_from(self.data, "endstream", ds)
            if de < 0:
                raise Error("PdfReader: xref-stream object has no endstream")
            while de > ds and (Int(self.data[de - 1]) == 10 or Int(self.data[de - 1]) == 13):
                de -= 1
        var raw = List[UInt8]()
        for i in range(ds, de):
            raw.append(self.data[i])
        if _bytes_contains(dict_bytes, "/FlateDecode"):
            return zlib_inflate(raw)
        return raw^

    def _maybe_undo_predictor(self, data: List[UInt8], dict_bytes: List[UInt8]) raises -> List[UInt8]:
        # Reverse a PNG/TIFF predictor declared in /DecodeParms. Only PNG
        # predictors (>=10) with the per-row filter byte are emitted by qpdf/
        # poppler for XRef streams; Predictor 1/absent -> no change.
        var dp = self._extract_decodeparms(dict_bytes)
        if len(dp) == 0:
            return data.copy()
        var pred_s = self._dict_value_str(dp, "/Predictor")
        if len(pred_s) == 0:
            return data.copy()
        var pred = _str_to_int(pred_s)
        if pred <= 1:
            return data.copy()
        var cols_s = self._dict_value_str(dp, "/Columns")
        var columns = _str_to_int(cols_s) if len(cols_s) > 0 else 1
        var colors_s = self._dict_value_str(dp, "/Colors")
        var colors = _str_to_int(colors_s) if len(colors_s) > 0 else 1
        var bpc_s = self._dict_value_str(dp, "/BitsPerComponent")
        var bpc = _str_to_int(bpc_s) if len(bpc_s) > 0 else 8
        var bpp = (colors * bpc + 7) // 8  # bytes per pixel (min 1)
        if bpp < 1:
            bpp = 1
        var rowlen = (colors * bpc * columns + 7) // 8  # data bytes per row
        if rowlen < 1:
            raise Error("PdfReader: bad predictor /Columns")
        if pred == 2:
            # TIFF predictor 2 (rare for XRef): horizontal differencing.
            return _undo_tiff_pred2(data, rowlen, bpp)
        # PNG predictors (10..15): each row prefixed with a 1-byte filter type.
        return _undo_png_predictor(data, rowlen, bpp)

    # ── dict helpers (operate on raw dict bytes, handle nesting) ────────────────
    def _be(self, data: List[UInt8], start: Int, width: Int) -> Int:
        # Big-endian unsigned integer of `width` bytes from data[start..].
        var v = 0
        for i in range(width):
            v = (v << 8) | Int(data[start + i])
        return v

    def _extract_decodeparms(self, dict_bytes: List[UInt8]) -> List[UInt8]:
        # Return the bytes of the /DecodeParms (or /DP) sub-dict, or empty.
        var key = _bytes_index_of(dict_bytes, "/DecodeParms")
        if key < 0:
            key = _bytes_index_of(dict_bytes, "/DP")
        if key < 0:
            return List[UInt8]()
        var i = key
        # find the "<<" that opens the sub-dict
        var open_pos = -1
        var j = i
        while j + 1 < len(dict_bytes):
            if dict_bytes[j] == ord("<") and dict_bytes[j + 1] == ord("<"):
                open_pos = j
                break
            # stop if we hit another '/' key before a dict opens (DP could be a ref)
            j += 1
            if j - i > 40:
                break
        if open_pos < 0:
            return List[UInt8]()
        var close_pos = _find_matching_gt_list(dict_bytes, open_pos)
        if close_pos < 0:
            return List[UInt8]()
        var out = List[UInt8]()
        for k in range(open_pos, close_pos):
            out.append(dict_bytes[k])
        return out^

    def _dict_value_str(self, dict_bytes: List[UInt8], key: String) raises -> String:
        # Public-style value extractor over arbitrary dict bytes (top level only;
        # skips nested sub-dicts so e.g. /Size in the outer dict isn't shadowed by
        # a key inside /DecodeParms). Returns name/number/"N G R" token.
        var kb = key.as_bytes()
        var i = 0
        var depth = 0
        var stop = len(dict_bytes)
        # We search at depth 1 (inside the outer << >>). The supplied bytes start
        # with "<<"; treat the outer as depth 0->1.
        while i + len(kb) <= stop:
            if dict_bytes[i] == ord("<") and i + 1 < stop and dict_bytes[i + 1] == ord("<"):
                depth += 1
                i += 2
                continue
            if dict_bytes[i] == ord(">") and i + 1 < stop and dict_bytes[i + 1] == ord(">"):
                depth -= 1
                i += 2
                continue
            # match key only at the outer level (depth == 1)
            var ok = depth == 1
            if ok:
                for k in range(len(kb)):
                    if dict_bytes[i + k] != kb[k]:
                        ok = False
                        break
            if ok:
                var p = _skip_ws_list(dict_bytes, i + len(kb))
                if p < stop and _is_digit(Int(dict_bytes[p])):
                    var v1 = 0
                    var p2 = _read_int_list(dict_bytes, p, v1)
                    var v2 = 0
                    var p3 = _read_int_list(dict_bytes, p2, v2)
                    var p4 = _skip_ws_list(dict_bytes, p3)
                    if v2 >= 0 and p4 < stop and Int(dict_bytes[p4]) == ord("R"):
                        return String(v1) + " " + String(v2) + " R"
                    return String(v1)
                elif p < stop and Int(dict_bytes[p]) == ord("/"):
                    var e = p + 1
                    while e < stop and not _is_ws(Int(dict_bytes[e])) and Int(dict_bytes[e]) != ord("/") and Int(dict_bytes[e]) != ord(">") and Int(dict_bytes[e]) != ord("[") and Int(dict_bytes[e]) != ord("]"):
                        e += 1
                    return _bytes_to_str(dict_bytes, p, e - p)
                else:
                    var e = p
                    while e < stop and not _is_ws(Int(dict_bytes[e])) and Int(dict_bytes[e]) != ord(">"):
                        e += 1
                    return _bytes_to_str(dict_bytes, p, e - p)
            i += 1
        return String("")

    def _dict_int_array(self, dict_bytes: List[UInt8], key: String) raises -> List[Int]:
        # Parse "key [ a b c ... ]" into a List[Int] (top-level key only).
        var kb = key.as_bytes()
        var i = 0
        var depth = 0
        var stop = len(dict_bytes)
        while i + len(kb) <= stop:
            if dict_bytes[i] == ord("<") and i + 1 < stop and dict_bytes[i + 1] == ord("<"):
                depth += 1; i += 2; continue
            if dict_bytes[i] == ord(">") and i + 1 < stop and dict_bytes[i + 1] == ord(">"):
                depth -= 1; i += 2; continue
            var ok = depth == 1
            if ok:
                for k in range(len(kb)):
                    if dict_bytes[i + k] != kb[k]:
                        ok = False
                        break
            if ok:
                var p = _skip_ws_list(dict_bytes, i + len(kb))
                if p < stop and Int(dict_bytes[p]) == ord("["):
                    p += 1
                    var out = List[Int]()
                    while p < stop:
                        p = _skip_ws_list(dict_bytes, p)
                        if p < stop and Int(dict_bytes[p]) == ord("]"):
                            break
                        if p < stop and _is_digit(Int(dict_bytes[p])):
                            var v = 0
                            p = _read_int_list(dict_bytes, p, v)
                            out.append(v)
                        else:
                            break
                    return out^
                return List[Int]()
            i += 1
        return List[Int]()

    def num_objects(self) -> Int:
        # Count of in-use objects discovered in the xref (excludes free entries).
        # Includes both directly-stored and ObjStm-compressed objects.
        var c = 0
        for i in range(len(self.offsets)):
            if self.offsets[i] >= 0:
                c += 1
        for i in range(len(self.objstm_container)):
            if self.objstm_container[i] >= 0:
                c += 1
        return c

    def trailer_value(self, key: String) raises -> String:
        # Return the token following `key` inside the trailer dict. Handles
        # indirect refs ("N 0 R" -> "N 0 R"), names ("/Foo"), and numbers.
        # For xref-stream PDFs the "trailer" lives in the XRef stream's dict.
        if self.xref_is_stream:
            return self._dict_value_str(self.trailer_dict, key)
        if self.trailer_start < 0:
            raise Error("PdfReader: trailer not parsed")
        var kb = key.as_bytes()
        var i = self.trailer_start
        var stop = self.trailer_end
        while i + len(kb) <= stop:
            var ok = True
            for k in range(len(kb)):
                if self.data[i + k] != kb[k]:
                    ok = False
                    break
            if ok:
                var p = _skip_ws(self.data, i + len(kb))
                # Read one value token. If it's an integer possibly followed by
                # "<gen> R", capture the whole "N G R" reference.
                if p < stop and _is_digit(Int(self.data[p])):
                    var v1 = 0
                    var p2 = _read_int(self.data, p, v1)
                    var save = p2
                    var v2 = 0
                    var p3 = _read_int(self.data, p2, v2)
                    var p4 = _skip_ws(self.data, p3)
                    if v2 >= 0 and p4 < stop and Int(self.data[p4]) == ord("R"):
                        return String(v1) + " " + String(v2) + " R"
                    return String(v1)
                elif p < stop and Int(self.data[p]) == ord("/"):
                    var e = p + 1
                    while e < stop and not _is_ws(Int(self.data[e])) and Int(self.data[e]) != ord("/") and Int(self.data[e]) != ord(">"):
                        e += 1
                    return _bytes_to_str(self.data, p, e - p)
                else:
                    # generic token until whitespace or delimiter
                    var e = p
                    while e < stop and not _is_ws(Int(self.data[e])) and Int(self.data[e]) != ord(">"):
                        e += 1
                    return _bytes_to_str(self.data, p, e - p)
            i += 1
        return String("")

    def get_object(self, n: Int) raises -> List[UInt8]:
        # Resolve object n to its raw body bytes. Works for objects stored
        # directly (offset-based, classic or xref-stream type-1) AND for objects
        # compressed inside an object stream (/ObjStm, xref-stream type-2).
        #
        # When the file is encrypted (open_encrypted authenticated), the returned
        # body has its literal/hex strings and (if present) stream payload
        # DECRYPTED with object n's key, EXCEPT for the /Encrypt dict (exempt) and
        # objects living inside an /ObjStm (those were decrypted as part of the
        # container's stream and must not be re-decrypted).
        var body: List[UInt8]
        var in_objstm = False
        if n >= 0 and n < len(self.objstm_container) and self.objstm_container[n] >= 0:
            body = self._get_object_from_objstm(n)
            in_objstm = True
        else:
            body = self._get_object_direct(n)
        if self.is_encrypted and not in_objstm and n != self.enc_encrypt_num:
            return self._decrypt_object_body(n, body)
        return body^

    def _get_object_from_objstm(self, n: Int) raises -> List[UInt8]:
        # The object lives at index objstm_index[n] inside ObjStm object
        # objstm_container[n]. Inflate the ObjStm, read its "N" and /First, parse
        # the N (objnum, reloffset) header pairs, and slice out object n's bytes.
        var container = self.objstm_container[n]
        var idx = self.objstm_index[n]
        # Read the container's dict to get /N and /First.
        if container < 0 or container >= len(self.offsets) or self.offsets[container] < 0:
            raise Error("PdfReader: ObjStm container " + String(container) + " not directly addressable")
        var c_off = self.offsets[container]
        var c_dict = self._read_obj_dict_at(c_off)
        var n_count = _str_to_int(self._dict_value_str(c_dict, "/N"))
        var first = _str_to_int(self._dict_value_str(c_dict, "/First"))
        if n_count < 0 or first < 0:
            raise Error("PdfReader: ObjStm missing /N or /First")
        var body = self._read_stream_bytes_at(c_off, c_dict)
        if idx < 0 or idx >= n_count:
            raise Error("PdfReader: ObjStm index out of range")
        # Header: n_count pairs of "objnum reloffset" (whitespace separated),
        # all located in body[0 .. first).
        var hp = 0
        var off_at = -1
        var off_next = len(body) - first  # default end = end of stream data
        for k in range(n_count):
            var onum = 0
            hp = _read_int_list(body, hp, onum)
            var roff = 0
            hp = _read_int_list(body, hp, roff)
            if k == idx:
                off_at = roff
            if k == idx + 1:
                off_next = roff
        if off_at < 0:
            raise Error("PdfReader: ObjStm object offset not found")
        var s = first + off_at
        var e = first + off_next
        if e > len(body):
            e = len(body)
        var out = List[UInt8]()
        for i in range(s, e):
            out.append(body[i])
        # trim trailing whitespace
        while len(out) > 0 and _is_ws(Int(out[len(out) - 1])):
            _ = out.pop()
        return out^

    def _get_object_direct(self, n: Int) raises -> List[UInt8]:
        # Return the raw body bytes of object n: everything between the
        # "n 0 obj\n" header and the trailing "endobj".
        if n < 0 or n >= len(self.offsets) or self.offsets[n] < 0:
            raise Error("PdfReader: object " + String(n) + " not in xref")
        var off = self.offsets[n]
        # Skip "n 0 obj" header.
        var p = off
        var got = 0
        p = _read_int(self.data, p, got)  # object number
        var gen = 0
        p = _read_int(self.data, p, gen)  # generation
        p = _skip_ws(self.data, p)
        if not _match_at(self.data, p, "obj"):
            raise Error("PdfReader: 'obj' keyword not at offset for object " + String(n))
        p += 3
        # body starts after the header; trim a single leading newline if present.
        if p < len(self.data) and (Int(self.data[p]) == 10 or Int(self.data[p]) == 13):
            p += 1
        var end = _find_from(self.data, "endobj", p)
        if end < 0:
            raise Error("PdfReader: no 'endobj' for object " + String(n))
        # trim a trailing newline before endobj
        var body_end = end
        while body_end > p and (Int(self.data[body_end - 1]) == 10 or Int(self.data[body_end - 1]) == 13):
            body_end -= 1
        var out = List[UInt8]()
        for i in range(p, body_end):
            out.append(self.data[i])
        return out^

    def _dict_has_flate(self, body: List[UInt8], dict_end: Int) -> Bool:
        # True if "/FlateDecode" appears in the object dict (before `stream`).
        var fb = String("/FlateDecode").as_bytes()
        var i = 0
        var last = dict_end - len(fb)
        while i <= last:
            var ok = True
            for k in range(len(fb)):
                if body[i + k] != fb[k]:
                    ok = False
                    break
            if ok:
                return True
            i += 1
        return False

    def get_stream_data(self, n: Int) raises -> List[UInt8]:
        # Locate the object's stream...endstream. If its dict declares
        # /Filter /FlateDecode, return zlib_inflate of the raw stream bytes;
        # otherwise return the raw stream bytes unchanged.
        var body = self.get_object(n)
        var sb = _find_from(body, "stream", 0)
        if sb < 0:
            raise Error("PdfReader: object " + String(n) + " has no stream")
        var is_flate = self._dict_has_flate(body, sb)
        # Per spec, "stream" is followed by CRLF or LF; stream data starts after.
        var ds = sb + len("stream")
        if ds < len(body) and Int(body[ds]) == 13:  # CR
            ds += 1
        if ds < len(body) and Int(body[ds]) == 10:  # LF
            ds += 1
        var de = _find_from(body, "endstream", ds)
        if de < 0:
            raise Error("PdfReader: object " + String(n) + " has no endstream")
        # trim a single newline immediately before endstream (added by writer).
        var data_end = de
        if data_end > ds and Int(body[data_end - 1]) == 10:
            data_end -= 1
            if data_end > ds and Int(body[data_end - 1]) == 13:
                data_end -= 1
        var raw = List[UInt8]()
        for i in range(ds, data_end):
            raw.append(body[i])
        if is_flate:
            return zlib_inflate(raw)
        return raw^

    # ── high-level helpers ──────────────────────────────────────────────────────
    def dict_value(self, obj_bytes: List[UInt8], key: String) raises -> String:
        # Public helper: extract `key`'s value from an object's dict bytes.
        # Handles nested sub-dicts (only matches the outermost dict's keys).
        return self._dict_value_str(obj_bytes, key)

    def _resolve_ref(self, token: String) raises -> Int:
        # Turn a "N G R" reference (or bare "N") into an object number; -1 if not.
        if len(token) == 0:
            return -1
        var v = 0
        var any = False
        var i = 0
        while i < len(token) and token[byte=i] >= "0" and token[byte=i] <= "9":
            v = v * 10 + (Int(ord(token[byte=i])) - ord("0"))
            i += 1
            any = True
        return v if any else -1

    def page_count(self) raises -> Int:
        # Resolve /Root -> /Pages -> /Count.
        var root_tok = self.trailer_value("/Root")
        var root_n = self._resolve_ref(root_tok)
        if root_n < 0:
            raise Error("PdfReader: cannot resolve /Root")
        var root_obj = self.get_object(root_n)
        var pages_tok = self.dict_value(root_obj, "/Pages")
        var pages_n = self._resolve_ref(pages_tok)
        if pages_n < 0:
            raise Error("PdfReader: cannot resolve /Pages")
        var pages_obj = self.get_object(pages_n)
        var count_s = self.dict_value(pages_obj, "/Count")
        if len(count_s) == 0:
            raise Error("PdfReader: /Pages has no /Count")
        return _str_to_int(count_s)

    # ════════════════════════════════════════════════════════════════════════════
    # ENCRYPTION / DECRYPTION (Standard Security Handler: RC4, AESV2, AESV3).
    # Mirrors the key-derivation in pdf/document.mojo's save_encrypted* (reversed).
    # ════════════════════════════════════════════════════════════════════════════

    # ── small crypto/byte helpers ───────────────────────────────────────────────
    def _enc_pad(self) raises -> List[UInt8]:
        # The 32-byte standard padding string (ISO 32000-1, 7.6.3.3).
        var p = List[UInt8]()
        p.append(UInt8(0x28)); p.append(UInt8(0xBF)); p.append(UInt8(0x4E)); p.append(UInt8(0x5E))
        p.append(UInt8(0x4E)); p.append(UInt8(0x75)); p.append(UInt8(0x8A)); p.append(UInt8(0x41))
        p.append(UInt8(0x64)); p.append(UInt8(0x00)); p.append(UInt8(0x4E)); p.append(UInt8(0x56))
        p.append(UInt8(0xFF)); p.append(UInt8(0xFA)); p.append(UInt8(0x01)); p.append(UInt8(0x08))
        p.append(UInt8(0x2E)); p.append(UInt8(0x2E)); p.append(UInt8(0x00)); p.append(UInt8(0xB6))
        p.append(UInt8(0xD0)); p.append(UInt8(0x68)); p.append(UInt8(0x3E)); p.append(UInt8(0x80))
        p.append(UInt8(0x2F)); p.append(UInt8(0x0C)); p.append(UInt8(0xA9)); p.append(UInt8(0xFE))
        p.append(UInt8(0x64)); p.append(UInt8(0x53)); p.append(UInt8(0x69)); p.append(UInt8(0x7A))
        return p^

    def _pad_pw(self, pw: List[UInt8]) raises -> List[UInt8]:
        var pad = self._enc_pad()
        var out = List[UInt8]()
        var take = len(pw)
        if take > 32:
            take = 32
        for i in range(take):
            out.append(pw[i])
        var k = 0
        while len(out) < 32:
            out.append(pad[k])
            k = k + 1
        return out^

    def _first_n(self, data: List[UInt8], n: Int) raises -> List[UInt8]:
        var out = List[UInt8]()
        var lim = n
        if lim > len(data):
            lim = len(data)
        for i in range(lim):
            out.append(data[i])
        return out^

    def _slice_bytes(self, data: List[UInt8], start: Int, end: Int) raises -> List[UInt8]:
        var out = List[UInt8]()
        var e = end
        if e > len(data):
            e = len(data)
        for i in range(start, e):
            out.append(data[i])
        return out^

    def _bytes_of_str(self, s: String) raises -> List[UInt8]:
        var src = s.as_bytes()
        var out = List[UInt8]()
        for i in range(len(src)):
            out.append(src[i])
        return out^

    def _bytes_eq(self, a: List[UInt8], b: List[UInt8], count: Int) -> Bool:
        if len(a) < count or len(b) < count:
            return False
        for i in range(count):
            if a[i] != b[i]:
                return False
        return True

    def _hex_decode(self, hex_str: List[UInt8]) raises -> List[UInt8]:
        # Decode ASCII hex digit bytes (ignoring whitespace) to raw bytes.
        var nibs = List[Int]()
        for i in range(len(hex_str)):
            var c = Int(hex_str[i])
            var v = -1
            if c >= ord("0") and c <= ord("9"):
                v = c - ord("0")
            elif c >= ord("a") and c <= ord("f"):
                v = c - ord("a") + 10
            elif c >= ord("A") and c <= ord("F"):
                v = c - ord("A") + 10
            if v >= 0:
                nibs.append(v)
        var out = List[UInt8]()
        var j = 0
        while j + 1 < len(nibs):
            out.append(UInt8(((nibs[j] << 4) | nibs[j + 1]) & 0xFF))
            j = j + 2
        if (len(nibs) % 2) == 1:
            out.append(UInt8((nibs[len(nibs) - 1] << 4) & 0xFF))
        return out^

    def _hex_encode_upper(self, data: List[UInt8]) raises -> List[UInt8]:
        var HEX = String("0123456789ABCDEF")
        var hb = HEX.as_bytes()
        var out = List[UInt8]()
        for i in range(len(data)):
            var byte = Int(data[i])
            out.append(hb[(byte >> 4) & 0xF])
            out.append(hb[byte & 0xF])
        return out^

    def _xor_each(self, key: List[UInt8], v: Int) raises -> List[UInt8]:
        var out = List[UInt8]()
        for i in range(len(key)):
            out.append(key[i] ^ UInt8(v & 0xFF))
        return out^

    # ── AES no-pad primitives (needed for AESV3 hash2b + UE/Perms) ───────────────
    def _aes_cbc_nopad_enc(self, key: List[UInt8], iv: List[UInt8], data: List[UInt8]) raises -> List[UInt8]:
        if (len(data) % 16) != 0:
            raise Error("_aes_cbc_nopad_enc: data not a multiple of 16")
        var rk = _key_expansion(key, _sbox())
        var prev = List[UInt8]()
        for i in range(16):
            prev.append(iv[i])
        var out = List[UInt8]()
        var off = 0
        while off < len(data):
            var block = List[UInt8]()
            for i in range(16):
                block.append(data[off + i] ^ prev[i])
            var enc = aes_encrypt_block(rk, block)
            for i in range(16):
                out.append(enc[i])
            prev = enc.copy()
            off += 16
        return out^

    def _aes_cbc_nopad_dec(self, key: List[UInt8], iv: List[UInt8], data: List[UInt8]) raises -> List[UInt8]:
        # AES-CBC decrypt, NO padding strip. len(data) multiple of 16.
        if (len(data) % 16) != 0:
            raise Error("_aes_cbc_nopad_dec: data not a multiple of 16")
        var rk = _key_expansion(key, _sbox())
        var prev = List[UInt8]()
        for i in range(16):
            prev.append(iv[i])
        var out = List[UInt8]()
        var off = 0
        while off < len(data):
            var block = List[UInt8]()
            for i in range(16):
                block.append(data[off + i])
            var dec = aes_decrypt_block(rk, block)
            for i in range(16):
                out.append(dec[i] ^ prev[i])
            prev = block.copy()
            off += 16
        return out^

    # ── SHA-512 / SHA-384 (FIPS 180-4) for the R6 hash2b loop ───────────────────
    def _rotr64(self, x: UInt64, n: UInt64) -> UInt64:
        return ((x >> n) | (x << (UInt64(64) - n)))

    def _sha512_core(self, data: List[UInt8], is384: Bool) raises -> List[UInt8]:
        var K = List[UInt64]()
        K.append(0x428a2f98d728ae22); K.append(0x7137449123ef65cd)
        K.append(0xb5c0fbcfec4d3b2f); K.append(0xe9b5dba58189dbbc)
        K.append(0x3956c25bf348b538); K.append(0x59f111f1b605d019)
        K.append(0x923f82a4af194f9b); K.append(0xab1c5ed5da6d8118)
        K.append(0xd807aa98a3030242); K.append(0x12835b0145706fbe)
        K.append(0x243185be4ee4b28c); K.append(0x550c7dc3d5ffb4e2)
        K.append(0x72be5d74f27b896f); K.append(0x80deb1fe3b1696b1)
        K.append(0x9bdc06a725c71235); K.append(0xc19bf174cf692694)
        K.append(0xe49b69c19ef14ad2); K.append(0xefbe4786384f25e3)
        K.append(0x0fc19dc68b8cd5b5); K.append(0x240ca1cc77ac9c65)
        K.append(0x2de92c6f592b0275); K.append(0x4a7484aa6ea6e483)
        K.append(0x5cb0a9dcbd41fbd4); K.append(0x76f988da831153b5)
        K.append(0x983e5152ee66dfab); K.append(0xa831c66d2db43210)
        K.append(0xb00327c898fb213f); K.append(0xbf597fc7beef0ee4)
        K.append(0xc6e00bf33da88fc2); K.append(0xd5a79147930aa725)
        K.append(0x06ca6351e003826f); K.append(0x142929670a0e6e70)
        K.append(0x27b70a8546d22ffc); K.append(0x2e1b21385c26c926)
        K.append(0x4d2c6dfc5ac42aed); K.append(0x53380d139d95b3df)
        K.append(0x650a73548baf63de); K.append(0x766a0abb3c77b2a8)
        K.append(0x81c2c92e47edaee6); K.append(0x92722c851482353b)
        K.append(0xa2bfe8a14cf10364); K.append(0xa81a664bbc423001)
        K.append(0xc24b8b70d0f89791); K.append(0xc76c51a30654be30)
        K.append(0xd192e819d6ef5218); K.append(0xd69906245565a910)
        K.append(0xf40e35855771202a); K.append(0x106aa07032bbd1b8)
        K.append(0x19a4c116b8d2d0c8); K.append(0x1e376c085141ab53)
        K.append(0x2748774cdf8eeb99); K.append(0x34b0bcb5e19b48a8)
        K.append(0x391c0cb3c5c95a63); K.append(0x4ed8aa4ae3418acb)
        K.append(0x5b9cca4f7763e373); K.append(0x682e6ff3d6b2b8a3)
        K.append(0x748f82ee5defb2fc); K.append(0x78a5636f43172f60)
        K.append(0x84c87814a1f0ab72); K.append(0x8cc702081a6439ec)
        K.append(0x90befffa23631e28); K.append(0xa4506cebde82bde9)
        K.append(0xbef9a3f7b2c67915); K.append(0xc67178f2e372532b)
        K.append(0xca273eceea26619c); K.append(0xd186b8c721c0c207)
        K.append(0xeada7dd6cde0eb1e); K.append(0xf57d4f7fee6ed178)
        K.append(0x06f067aa72176fba); K.append(0x0a637dc5a2c898a6)
        K.append(0x113f9804bef90dae); K.append(0x1b710b35131c471b)
        K.append(0x28db77f523047d84); K.append(0x32caab7b40c72493)
        K.append(0x3c9ebe0a15c9bebc); K.append(0x431d67c49c100d4c)
        K.append(0x4cc5d4becb3e42b6); K.append(0x597f299cfc657e2a)
        K.append(0x5fcb6fab3ad6faec); K.append(0x6c44198c4a475817)

        var h0: UInt64
        var h1: UInt64
        var h2: UInt64
        var h3: UInt64
        var h4: UInt64
        var h5: UInt64
        var h6: UInt64
        var h7: UInt64
        if is384:
            h0 = 0xcbbb9d5dc1059ed8; h1 = 0x629a292a367cd507
            h2 = 0x9159015a3070dd17; h3 = 0x152fecd8f70e5939
            h4 = 0x67332667ffc00b31; h5 = 0x8eb44a8768581511
            h6 = 0xdb0c2e0d64f98fa7; h7 = 0x47b5481dbefa4fa4
        else:
            h0 = 0x6a09e667f3bcc908; h1 = 0xbb67ae8584caa73b
            h2 = 0x3c6ef372fe94f82b; h3 = 0xa54ff53a5f1d36f1
            h4 = 0x510e527fade682d1; h5 = 0x9b05688c2b3e6c1f
            h6 = 0x1f83d9abfb41bd6b; h7 = 0x5be0cd19137e2179

        var msg = List[UInt8]()
        for i in range(len(data)):
            msg.append(data[i])
        var bit_len = UInt64(len(data)) * UInt64(8)
        msg.append(UInt8(0x80))
        while (len(msg) % 128) != 112:
            msg.append(UInt8(0))
        for _q in range(8):
            msg.append(UInt8(0))
        for i in range(8):
            var shift = UInt64(8) * UInt64(7 - i)
            msg.append(UInt8((bit_len >> shift) & UInt64(0xFF)))

        var chunk = 0
        while chunk < len(msg):
            var W = List[UInt64]()
            for w in range(16):
                var base = chunk + w * 8
                var word = UInt64(0)
                for bidx in range(8):
                    word = (word << 8) | UInt64(msg[base + bidx])
                W.append(word)
            for i in range(16, 80):
                var w15 = W[i - 15]
                var w2 = W[i - 2]
                var s0 = self._rotr64(w15, 1) ^ self._rotr64(w15, 8) ^ (w15 >> 7)
                var s1 = self._rotr64(w2, 19) ^ self._rotr64(w2, 61) ^ (w2 >> 6)
                W.append(W[i - 16] + s0 + W[i - 7] + s1)

            var a = h0
            var b = h1
            var c = h2
            var d = h3
            var e = h4
            var f = h5
            var g = h6
            var h = h7
            for i in range(80):
                var S1 = self._rotr64(e, 14) ^ self._rotr64(e, 18) ^ self._rotr64(e, 41)
                var ch = (e & f) ^ ((~e) & g)
                var temp1 = h + S1 + ch + K[i] + W[i]
                var S0 = self._rotr64(a, 28) ^ self._rotr64(a, 34) ^ self._rotr64(a, 39)
                var maj = (a & b) ^ (a & c) ^ (b & c)
                var temp2 = S0 + maj
                h = g
                g = f
                f = e
                e = d + temp1
                d = c
                c = b
                b = a
                a = temp1 + temp2

            h0 = h0 + a
            h1 = h1 + b
            h2 = h2 + c
            h3 = h3 + d
            h4 = h4 + e
            h5 = h5 + f
            h6 = h6 + g
            h7 = h7 + h
            chunk = chunk + 128

        var words = List[UInt64]()
        words.append(h0); words.append(h1); words.append(h2); words.append(h3)
        words.append(h4); words.append(h5); words.append(h6); words.append(h7)
        var out = List[UInt8]()
        var nwords = 8
        if is384:
            nwords = 6  # SHA-384 = first 48 bytes
        for wi in range(nwords):
            var v = words[wi]
            for bidx in range(8):
                var shift = UInt64(8) * UInt64(7 - bidx)
                out.append(UInt8((v >> shift) & UInt64(0xFF)))
        return out^

    def _sha384(self, data: List[UInt8]) raises -> List[UInt8]:
        return self._sha512_core(data, True)

    def _sha512(self, data: List[UInt8]) raises -> List[UInt8]:
        return self._sha512_core(data, False)

    def _concat3(self, a: List[UInt8], b: List[UInt8], c: List[UInt8]) raises -> List[UInt8]:
        var out = List[UInt8]()
        for i in range(len(a)):
            out.append(a[i])
        for i in range(len(b)):
            out.append(b[i])
        for i in range(len(c)):
            out.append(c[i])
        return out^

    # Algorithm 2.B (ISO 32000-2, R6 hardened hash).
    def _hash2b(self, password: List[UInt8], salt: List[UInt8], udata: List[UInt8]) raises -> List[UInt8]:
        var K = sha256(self._concat3(password, salt, udata))  # 32 bytes
        var round = 0
        while True:
            round = round + 1
            var seq = self._concat3(password, K, udata)
            var K1 = List[UInt8]()
            for _r in range(64):
                for i in range(len(seq)):
                    K1.append(seq[i])
            var ek = self._slice_bytes(K, 0, 16)
            var iv = self._slice_bytes(K, 16, 32)
            var E = self._aes_cbc_nopad_enc(ek, iv, K1)
            var ssum = 0
            for i in range(16):
                ssum = ssum + Int(E[i])
            var modv = ssum % 3
            if modv == 0:
                K = sha256(E)
            elif modv == 1:
                K = self._sha384(E)
            else:
                K = self._sha512(E)
            var last = Int(E[len(E) - 1])
            if round >= 64 and last <= (round - 32):
                break
        return self._slice_bytes(K, 0, 32)

    # ── /Encrypt dict + /ID parsing ─────────────────────────────────────────────
    def _trailer_id0(self) raises -> List[UInt8]:
        # Extract the first element of /ID [ <hex0> <hex1> ] from the trailer
        # (classic) or XRef-stream dict. Returns the raw bytes of <hex0>.
        var src: List[UInt8]
        var start: Int
        var stop: Int
        if self.xref_is_stream:
            src = self.trailer_dict.copy()
            start = 0
            stop = len(src)
        else:
            src = self.data.copy()
            start = self.trailer_start
            stop = self.trailer_end
        # find "/ID"
        var idkw = String("/ID").as_bytes()
        var i = start
        var pos = -1
        while i + 3 <= stop:
            if src[i] == idkw[0] and src[i + 1] == idkw[1] and src[i + 2] == idkw[2]:
                pos = i
                break
            i += 1
        if pos < 0:
            return List[UInt8]()  # no /ID (allowed; e.g. AESV3 files still have one here)
        # find first '<' after /ID, then '>'
        var j = pos + 3
        while j < stop and Int(src[j]) != ord("<"):
            j += 1
        if j >= stop:
            return List[UInt8]()
        var hs = j + 1
        var he = hs
        while he < stop and Int(src[he]) != ord(">"):
            he += 1
        var hx = List[UInt8]()
        for k in range(hs, he):
            hx.append(src[k])
        return self._hex_decode(hx)

    def _enc_dict_hexval(self, enc: List[UInt8], key: String) raises -> List[UInt8]:
        # Return the raw bytes of a hex-string value "key <ABCD...>" in the
        # /Encrypt dict bytes. Empty if not found.
        var kb = key.as_bytes()
        var stop = len(enc)
        var i = 0
        while i + len(kb) <= stop:
            var ok = True
            for k in range(len(kb)):
                if enc[i + k] != kb[k]:
                    ok = False
                    break
            if ok:
                var p = i + len(kb)
                while p < stop and _is_ws(Int(enc[p])):
                    p += 1
                if p < stop and Int(enc[p]) == ord("<"):
                    var hs = p + 1
                    var he = hs
                    while he < stop and Int(enc[he]) != ord(">"):
                        he += 1
                    var hx = List[UInt8]()
                    for q in range(hs, he):
                        hx.append(enc[q])
                    return self._hex_decode(hx)
                elif p < stop and Int(enc[p]) == ord("("):
                    # literal string form (rare for O/U) — decode minimally.
                    var ls = p + 1
                    var le = ls
                    while le < stop and Int(enc[le]) != ord(")"):
                        if Int(enc[le]) == ord("\\"):
                            le += 1
                        le += 1
                    var out = List[UInt8]()
                    var z = ls
                    while z < le:
                        if Int(enc[z]) == ord("\\") and z + 1 < le:
                            out.append(enc[z + 1])
                            z += 2
                        else:
                            out.append(enc[z])
                            z += 1
                    return out^
                return List[UInt8]()
            i += 1
        return List[UInt8]()

    def _enc_dict_int(self, enc: List[UInt8], key: String) raises -> Int:
        var s = self._dict_value_str(enc, key)
        if len(s) == 0:
            return -1
        # handle negative
        var neg = False
        var start = 0
        if len(s) > 0 and s[byte=0] == "-":
            neg = True
            start = 1
        var v = 0
        var any = False
        for j in range(start, len(s)):
            var c = ord(s[byte=j])
            if c >= ord("0") and c <= ord("9"):
                v = v * 10 + (c - ord("0"))
                any = True
            else:
                break
        if not any:
            return -1
        return -v if neg else v

    def _detect_cfm(self, enc: List[UInt8], V: Int) raises -> Int:
        # Decide cipher: 0=RC4, 1=AESV2, 2=AESV3. For V<=3 always RC4. For V>=4
        # read /CF /StdCF /CFM.
        if V <= 3:
            return 0
        if _bytes_index_of(enc, "AESV3") >= 0:
            return 2
        if _bytes_index_of(enc, "AESV2") >= 0:
            return 1
        if _bytes_index_of(enc, "/V2") >= 0:
            return 0
        # default for V4 with no recognizable CFM -> RC4
        return 0

    # ── open + authenticate ─────────────────────────────────────────────────────
    def open_encrypted(mut self, path: String, password: String) raises:
        # Open the file, parse the trailer, read /Encrypt, authenticate the user
        # password, and compute the file encryption key. After this, get_object /
        # get_stream_data / get_string transparently decrypt.
        self.open(path)
        var enc_tok = self.trailer_value("/Encrypt")
        if len(enc_tok) == 0:
            raise Error("PdfReader.open_encrypted: file has no /Encrypt (not encrypted)")
        var enc_num = self._resolve_ref(enc_tok)
        if enc_num < 0:
            raise Error("PdfReader.open_encrypted: cannot resolve /Encrypt reference")
        self.enc_encrypt_num = enc_num
        var enc = self.get_object(enc_num)  # not flagged encrypted yet -> raw

        # Required common fields.
        var filter = self._dict_value_str(enc, "/Filter")
        if filter != "/Standard":
            raise Error("PdfReader.open_encrypted: only /Standard handler supported, got " + filter)
        var V = self._enc_dict_int(enc, "/V")
        var R = self._enc_dict_int(enc, "/R")
        if V < 1 or R < 2:
            raise Error("PdfReader.open_encrypted: bad /V or /R")
        var cfm = self._detect_cfm(enc, V)
        self.enc_cfm = cfm

        var pw = self._bytes_of_str(password)

        if V >= 5:
            # AESV3 (R6): authenticate via validation salt + hash2b vs /U.
            self._auth_aesv3(enc, pw)
        else:
            # RC4 / AESV2 (R2..R4): Algorithm 2 key + Algorithm 6 /U check.
            self._auth_rc4_aes128(enc, pw, V, R)
        self.is_encrypted = True

    def _auth_rc4_aes128(mut self, enc: List[UInt8], pw: List[UInt8], V: Int, R: Int) raises:
        var O = self._enc_dict_hexval(enc, "/O")
        var U = self._enc_dict_hexval(enc, "/U")
        if len(O) < 32 or len(U) < 16:
            raise Error("PdfReader.open_encrypted: /O or /U missing/short")
        var P = self._enc_dict_int(enc, "/P")
        var length_bits = self._enc_dict_int(enc, "/Length")
        var keylen = 5  # 40-bit default (V1)
        if V >= 2:
            if length_bits > 0:
                keylen = length_bits // 8
            else:
                keylen = 16
        var id0 = self._trailer_id0()

        # Algorithm 2: encryption key from the USER password.
        var m = self._pad_pw(pw)
        for q in range(len(O)):
            if q < 32:
                m.append(O[q])
        var pmask = P & 0xFFFFFFFF
        m.append(UInt8(pmask & 0xFF))
        m.append(UInt8((pmask >> 8) & 0xFF))
        m.append(UInt8((pmask >> 16) & 0xFF))
        m.append(UInt8((pmask >> 24) & 0xFF))
        for q in range(len(id0)):
            m.append(id0[q])
        # (Skip the R4 EncryptMetadata=false 0xFFFFFFFF append; our writer keeps
        #  metadata encrypted so it is omitted.)
        var key = md5(m)
        if R >= 3:
            for _r in range(50):
                key = md5(self._first_n(key, keylen))
        var enc_key = self._first_n(key, keylen)

        # Algorithm 6: verify the USER password against /U.
        var ok = False
        if R == 2:
            var u_calc = rc4(enc_key, self._enc_pad())
            ok = self._bytes_eq(u_calc, U, 32)
        else:
            var um = List[UInt8]()
            var pad = self._enc_pad()
            for q in range(len(pad)):
                um.append(pad[q])
            for q in range(len(id0)):
                um.append(id0[q])
            var u = md5(um)
            u = rc4(enc_key, u)
            for i in range(1, 20):
                u = rc4(self._xor_each(enc_key, i), u)
            # Compare first 16 bytes (the rest of /U is arbitrary padding).
            ok = self._bytes_eq(u, U, 16)
        if not ok:
            raise Error("PdfReader.open_encrypted: authentication failed (wrong password)")
        self.enc_file_key = enc_key.copy()

    def _auth_aesv3(mut self, enc: List[UInt8], pw: List[UInt8]) raises:
        var U = self._enc_dict_hexval(enc, "/U")
        var UE = self._enc_dict_hexval(enc, "/UE")
        if len(U) < 48 or len(UE) < 32:
            raise Error("PdfReader.open_encrypted: AESV3 /U or /UE missing/short")
        var vsalt = self._slice_bytes(U, 32, 40)  # validation salt
        var ksalt = self._slice_bytes(U, 40, 48)  # key salt
        var empty = List[UInt8]()
        # Validate the user password.
        var hv = self._hash2b(pw, vsalt, empty)
        var ok = self._bytes_eq(hv, self._first_n(U, 32), 32)
        if not ok:
            raise Error("PdfReader.open_encrypted: authentication failed (wrong password)")
        # Derive the intermediate key, then AES-256-CBC-no-pad decrypt UE -> FK.
        var ik = self._hash2b(pw, ksalt, empty)
        var zero_iv = List[UInt8]()
        for _z in range(16):
            zero_iv.append(UInt8(0))
        var FK = self._aes_cbc_nopad_dec(ik, zero_iv, self._first_n(UE, 32))
        self.enc_file_key = FK.copy()

    # ── per-object key + body decryption ────────────────────────────────────────
    def _obj_key(self, onum: Int) raises -> List[UInt8]:
        # RC4 (V1/V2): md5(key + obj3LE + gen2LE)[:keylen+5] (capped at 16).
        var m = List[UInt8]()
        for i in range(len(self.enc_file_key)):
            m.append(self.enc_file_key[i])
        m.append(UInt8(onum & 0xFF))
        m.append(UInt8((onum >> 8) & 0xFF))
        m.append(UInt8((onum >> 16) & 0xFF))
        m.append(UInt8(0))  # gen low
        m.append(UInt8(0))  # gen high
        var d = md5(m)
        var n = len(self.enc_file_key) + 5
        if n > 16:
            n = 16
        return self._first_n(d, n)

    def _obj_key_aes(self, onum: Int) raises -> List[UInt8]:
        # AESV2: md5(key + obj3LE + gen2LE + "sAlT")[:16].
        var m = List[UInt8]()
        for i in range(len(self.enc_file_key)):
            m.append(self.enc_file_key[i])
        m.append(UInt8(onum & 0xFF))
        m.append(UInt8((onum >> 8) & 0xFF))
        m.append(UInt8((onum >> 16) & 0xFF))
        m.append(UInt8(0))  # gen low
        m.append(UInt8(0))  # gen high
        m.append(UInt8(0x73))  # 's'
        m.append(UInt8(0x41))  # 'A'
        m.append(UInt8(0x6C))  # 'l'
        m.append(UInt8(0x54))  # 'T'
        var d = md5(m)
        return self._first_n(d, 16)

    def _decrypt_token(self, raw: List[UInt8], onum: Int) raises -> List[UInt8]:
        # Decrypt one string/stream payload according to the active cipher.
        if self.enc_cfm == 0:
            var key = self._obj_key(onum)
            return rc4(key, raw)  # RC4 is symmetric
        elif self.enc_cfm == 1:
            # AESV2: first 16 bytes = IV, rest = ciphertext (PKCS#7 padded).
            if len(raw) < 16:
                return List[UInt8]()
            var key = self._obj_key_aes(onum)
            var iv = self._first_n(raw, 16)
            var ct = self._slice_bytes(raw, 16, len(raw))
            if len(ct) == 0 or (len(ct) % 16) != 0:
                return List[UInt8]()
            return aes_cbc_decrypt(key, iv, ct)
        else:
            # AESV3: file key directly, first 16 bytes = IV.
            if len(raw) < 16:
                return List[UInt8]()
            var iv = self._first_n(raw, 16)
            var ct = self._slice_bytes(raw, 16, len(raw))
            if len(ct) == 0 or (len(ct) % 16) != 0:
                return List[UInt8]()
            return aes_cbc_decrypt(self.enc_file_key, iv, ct)

    def _reescape_literal(self, raw: List[UInt8]) raises -> List[UInt8]:
        # Re-escape raw bytes for a PDF literal string body (between the parens).
        var out = List[UInt8]()
        for i in range(len(raw)):
            var c = raw[i]
            if c == UInt8(ord("(")) or c == UInt8(ord(")")) or c == UInt8(ord("\\")):
                out.append(UInt8(ord("\\")))
                out.append(c)
            elif Int(c) < 0x20:
                var v = Int(c)
                out.append(UInt8(ord("\\")))
                out.append(UInt8(ord("0") + ((v >> 6) & 0x7)))
                out.append(UInt8(ord("0") + ((v >> 3) & 0x7)))
                out.append(UInt8(ord("0") + (v & 0x7)))
            else:
                out.append(c)
        return out^

    def _decode_literal(self, esc: List[UInt8]) raises -> List[UInt8]:
        # Decode a PDF literal-string BODY (bytes between '(' and ')').
        var out = List[UInt8]()
        var i = 0
        while i < len(esc):
            var c = esc[i]
            if c == UInt8(ord("\\")) and i + 1 < len(esc):
                var nch = esc[i + 1]
                if nch >= UInt8(ord("0")) and nch <= UInt8(ord("7")):
                    var val = 0
                    var cnt = 0
                    var j = i + 1
                    while j < len(esc) and cnt < 3 and esc[j] >= UInt8(ord("0")) and esc[j] <= UInt8(ord("7")):
                        val = (val << 3) | (Int(esc[j]) - ord("0"))
                        j = j + 1
                        cnt = cnt + 1
                    out.append(UInt8(val & 0xFF))
                    i = j
                else:
                    if nch == UInt8(ord("n")):
                        out.append(UInt8(10))
                    elif nch == UInt8(ord("r")):
                        out.append(UInt8(13))
                    elif nch == UInt8(ord("t")):
                        out.append(UInt8(9))
                    elif nch == UInt8(ord("b")):
                        out.append(UInt8(8))
                    elif nch == UInt8(ord("f")):
                        out.append(UInt8(12))
                    else:
                        out.append(nch)
                    i = i + 2
            else:
                out.append(c)
                i = i + 1
        return out^

    def _match_body(self, body: List[UInt8], pos: Int, s: String) raises -> Bool:
        var sb = s.as_bytes()
        if pos + len(sb) > len(body):
            return False
        for k in range(len(sb)):
            if body[pos + k] != sb[k]:
                return False
        return True

    def _decrypt_object_body(self, onum: Int, body: List[UInt8]) raises -> List[UInt8]:
        # Walk one object body, decrypting every literal string ( … ), hex string
        # < … > (NOT << >>), and the stream payload between "stream"+nl and
        # "\nendstream". Mirrors document.mojo's _encrypt_body walkers, reversed.
        var out = List[UInt8]()
        var i = 0
        var n = len(body)
        while i < n:
            var c = body[i]
            if c == UInt8(ord("s")) and self._match_body(body, i, "stream"):
                for k in range(6):
                    out.append(body[i + k])
                var p = i + 6
                if p < n and body[p] == UInt8(13):
                    out.append(body[p]); p = p + 1
                if p < n and body[p] == UInt8(10):
                    out.append(body[p]); p = p + 1
                var es = _find_from(body, "endstream", p)
                if es < 0:
                    while p < n:
                        out.append(body[p]); p = p + 1
                    i = p
                    continue
                var pe = es
                if pe > p and body[pe - 1] == UInt8(10):
                    pe = pe - 1
                    if pe > p and body[pe - 1] == UInt8(13):
                        pe = pe - 1
                var payload = List[UInt8]()
                for q in range(p, pe):
                    payload.append(body[q])
                var dec = self._decrypt_token(payload, onum)
                for q in range(len(dec)):
                    out.append(dec[q])
                for q in range(pe, es):
                    out.append(body[q])
                for k in range(len("endstream")):
                    out.append(body[es + k])
                i = es + len("endstream")
                continue
            elif c == UInt8(ord("(")):
                var depth = 1
                var j = i + 1
                while j < n and depth > 0:
                    var cj = body[j]
                    if cj == UInt8(ord("\\")):
                        j = j + 2
                        continue
                    if cj == UInt8(ord("(")):
                        depth = depth + 1
                    elif cj == UInt8(ord(")")):
                        depth = depth - 1
                        if depth == 0:
                            break
                    j = j + 1
                var inner = List[UInt8]()
                for q in range(i + 1, j):
                    inner.append(body[q])
                var rawe = self._decode_literal(inner)
                var decs = self._decrypt_token(rawe, onum)
                var resc = self._reescape_literal(decs)
                out.append(UInt8(ord("(")))
                for q in range(len(resc)):
                    out.append(resc[q])
                out.append(UInt8(ord(")")))
                i = j + 1
                continue
            elif c == UInt8(ord("<")):
                if i + 1 < n and body[i + 1] == UInt8(ord("<")):
                    out.append(c)
                    out.append(body[i + 1])
                    i = i + 2
                    continue
                var j = i + 1
                while j < n and body[j] != UInt8(ord(">")):
                    j = j + 1
                var hx = List[UInt8]()
                for q in range(i + 1, j):
                    hx.append(body[q])
                var rawh = self._hex_decode(hx)
                var dech = self._decrypt_token(rawh, onum)
                var hexout = self._hex_encode_upper(dech)
                out.append(UInt8(ord("<")))
                for q in range(len(hexout)):
                    out.append(hexout[q])
                out.append(UInt8(ord(">")))
                i = j + 1
                continue
            else:
                out.append(c)
                i = i + 1
        return out^

    # ── decryption-aware public accessor ────────────────────────────────────────
    def get_string(self, obj_bytes: List[UInt8], key: String) raises -> String:
        # Return a (decoded, plaintext) string value for `key` from an object's
        # body. When the object body was produced by get_object on an encrypted
        # file, the literal/hex string is already decrypted; here we just decode
        # the literal escapes / hex into raw chars.
        var kb = key.as_bytes()
        var stop = len(obj_bytes)
        var i = 0
        while i + len(kb) <= stop:
            var ok = True
            for k in range(len(kb)):
                if obj_bytes[i + k] != kb[k]:
                    ok = False
                    break
            if ok:
                var p = i + len(kb)
                while p < stop and _is_ws(Int(obj_bytes[p])):
                    p += 1
                if p < stop and Int(obj_bytes[p]) == ord("("):
                    var depth = 1
                    var j = p + 1
                    while j < stop and depth > 0:
                        var cj = obj_bytes[j]
                        if cj == UInt8(ord("\\")):
                            j = j + 2
                            continue
                        if cj == UInt8(ord("(")):
                            depth = depth + 1
                        elif cj == UInt8(ord(")")):
                            depth = depth - 1
                            if depth == 0:
                                break
                        j = j + 1
                    var inner = List[UInt8]()
                    for q in range(p + 1, j):
                        inner.append(obj_bytes[q])
                    var raws = self._decode_literal(inner)
                    var s = String("")
                    for q in range(len(raws)):
                        s += chr(Int(raws[q]))
                    return s^
                elif p < stop and Int(obj_bytes[p]) == ord("<"):
                    var hs = p + 1
                    var he = hs
                    while he < stop and Int(obj_bytes[he]) != ord(">"):
                        he += 1
                    var hxb = List[UInt8]()
                    for q in range(hs, he):
                        hxb.append(obj_bytes[q])
                    var rawh = self._hex_decode(hxb)
                    var s2 = String("")
                    for q in range(len(rawh)):
                        s2 += chr(Int(rawh[q]))
                    return s2^
            i += 1
        return String("")

    def _kids_obj_nums(self, pages_obj: List[UInt8]) raises -> List[Int]:
        # Parse "/Kids [ N 0 R N 0 R ... ]" -> [N, N, ...] (object numbers).
        var kb = String("/Kids").as_bytes()
        var stop = len(pages_obj)
        var i = 0
        var out = List[Int]()
        while i + len(kb) <= stop:
            var ok = True
            for k in range(len(kb)):
                if pages_obj[i + k] != kb[k]:
                    ok = False
                    break
            if ok:
                var p = i + len(kb)
                while p < stop and _is_ws(Int(pages_obj[p])):
                    p += 1
                if p < stop and Int(pages_obj[p]) == ord("["):
                    p += 1
                    while p < stop:
                        p = _skip_ws(pages_obj, p)
                        if p < stop and Int(pages_obj[p]) == ord("]"):
                            break
                        if p < stop and _is_digit(Int(pages_obj[p])):
                            var nval = 0
                            p = _read_int(pages_obj, p, nval)
                            var gval = 0
                            p = _read_int(pages_obj, p, gval)
                            p = _skip_ws(pages_obj, p)
                            if p < stop and Int(pages_obj[p]) == ord("R"):
                                p += 1
                            out.append(nval)
                        else:
                            break
                return out^
            i += 1
        return out^

    def extract_page_text_ish(self, page_index: Int) raises -> String:
        # Resolve the page object at page_index (0-based), decrypt its /Contents
        # stream, and return the (decompressed) content-stream bytes as a String.
        # Not a full text extractor — exposes the drawn "( … ) Tj" operators.
        var root_n = self._resolve_ref(self.trailer_value("/Root"))
        if root_n < 0:
            raise Error("extract_page_text_ish: cannot resolve /Root")
        var root_obj = self.get_object(root_n)
        var pages_n = self._resolve_ref(self.dict_value(root_obj, "/Pages"))
        if pages_n < 0:
            raise Error("extract_page_text_ish: cannot resolve /Pages")
        var pages_obj = self.get_object(pages_n)
        # Find the Kids array and pick page_index. Kids entries are "N 0 R"
        # references; collect the object numbers (every 3rd token: N, gen, R).
        var kids = self._kids_obj_nums(pages_obj)
        if page_index < 0 or page_index >= len(kids):
            raise Error("extract_page_text_ish: page index out of range")
        var page_n = kids[page_index]
        var page_obj = self.get_object(page_n)
        var contents_n = self._resolve_ref(self.dict_value(page_obj, "/Contents"))
        if contents_n < 0:
            raise Error("extract_page_text_ish: cannot resolve /Contents")
        var stream = self.get_stream_data(contents_n)
        var s = String("")
        for i in range(len(stream)):
            s += chr(Int(stream[i]))
        return s^
