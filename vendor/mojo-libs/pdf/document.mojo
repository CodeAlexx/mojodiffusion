# pdf/document.mojo
# Pure-Mojo PDF 1.7 document assembler.
#
# Strategy: hold each object's body bytes (the bytes BETWEEN "N 0 obj\n" and
# "\nendobj"). On save(), assemble the entire file into ONE List[UInt8],
# recording the exact byte offset at the moment each "N 0 obj\n" header is
# written. This guarantees the xref offsets match the produced bytes.
#
# Catalog and Pages are appended as the LAST two objects (we know every page
# object number by then), so the Kids array and /Count are correct.

from pdf.objects import (
    append_str,
    append_int,
    append_real,
    pdf_name,
    pdf_literal_string,
)
from graphics.deflate import deflate
from pdf.md5 import md5
from pdf.rc4 import rc4
from pdf.aes import aes_cbc_encrypt, aes_encrypt_block, _key_expansion, _sbox
from pdf.sha256 import sha256
from std.random import random_ui64


struct PdfDoc(Movable):
    # objects[i] = raw bytes between "N 0 obj\n" and "\nendobj" for object i+1.
    var objects: List[List[UInt8]]
    # 1-based object numbers of the page objects, in page order.
    var page_numbers: List[Int]
    # Document /Info metadata. Only emitted if has_info is True.
    var has_info: Bool
    var info_title: String
    var info_author: String
    var info_subject: String
    var info_creator: String
    # Document outline (bookmarks). 0 means "no outline".
    var outlines_obj: Int
    # Active AES cipher key used by the modern-encryption body walker.
    # (Per-object key for AESV2, or the 32-byte file key for AESV3.)
    var _aes_active_key: List[UInt8]
    # When True, save()/save_compressed() emit an XMP /Metadata stream and a
    # document /ID. (Added feature; off by default to preserve existing tests.)
    var xmp_enabled: Bool

    def __init__(out self):
        self.objects = List[List[UInt8]]()
        self.page_numbers = List[Int]()
        self.has_info = False
        self.info_title = String("")
        self.info_author = String("")
        self.info_subject = String("")
        self.info_creator = String("")
        self.outlines_obj = 0
        self._aes_active_key = List[UInt8]()
        self.xmp_enabled = False

    def set_info(
        mut self,
        title: String,
        author: String,
        subject: String,
        creator: String,
    ):
        # Record document metadata; emitted as an /Info dict in save().
        self.has_info = True
        self.info_title = title
        self.info_author = author
        self.info_subject = subject
        self.info_creator = creator

    def set_outlines(mut self, outlines_obj: Int):
        # Record the object number of the document outline (bookmarks) tree.
        # save()'s Catalog will reference it via /Outlines.
        self.outlines_obj = outlines_obj

    def _zlib_wrap(self, raw: List[UInt8]) raises -> List[UInt8]:
        # Wrap a raw DEFLATE stream in a zlib (RFC 1950) container for PDF's
        # /FlateDecode: 2-byte header (0x78 0x9C) + raw deflate body + 4-byte
        # big-endian Adler-32 of the UNcompressed data. (Mirrors pdf/image.mojo.)
        var compressed = deflate(raw)
        var z = List[UInt8]()
        z.append(UInt8(0x78))
        z.append(UInt8(0x9C))
        for i in range(len(compressed)):
            z.append(compressed[i])
        var a: Int = 1
        var b2: Int = 0
        var MOD: Int = 65521
        for i in range(len(raw)):
            a = (a + Int(raw[i])) % MOD
            b2 = (b2 + a) % MOD
        var adler = (b2 << 16) | a
        z.append(UInt8((adler >> 24) & 0xFF))
        z.append(UInt8((adler >> 16) & 0xFF))
        z.append(UInt8((adler >> 8) & 0xFF))
        z.append(UInt8(adler & 0xFF))
        return z.copy()

    def add_object(mut self, body: List[UInt8]) raises -> Int:
        # Append an object body; return its 1-based object number.
        self.objects.append(body.copy())
        return len(self.objects)

    def add_stream_object(
        mut self, dict_bytes: List[UInt8], stream_data: List[UInt8]
    ) raises -> Int:
        # Builds:
        #   << ...dict... /Length N >>\nstream\n<stream_data>\nendstream
        # where N = len(stream_data). dict_bytes must be the INNER dict content
        # WITHOUT the surrounding "<<" ">>" (e.g. "/Type /XObject ...") OR may
        # be empty. We always wrap with "<<" ... " /Length N >>".
        var body = List[UInt8]()
        append_str(body, "<< ")
        for i in range(len(dict_bytes)):
            body.append(dict_bytes[i])
        if len(dict_bytes) > 0:
            body.append(UInt8(ord(" ")))
        append_str(body, "/Length ")
        append_int(body, len(stream_data))
        append_str(body, " >>\nstream\n")
        for i in range(len(stream_data)):
            body.append(stream_data[i])
        append_str(body, "\nendstream")
        return self.add_object(body)

    def add_page(
        mut self,
        width: Float64,
        height: Float64,
        content: List[UInt8],
        resources: List[UInt8],
    ) raises -> Int:
        # Backward-compatible thin wrapper: no annotations, uncompressed stream.
        var no_annots = List[Int]()
        return self.add_page_ext(
            width, height, content, resources, no_annots, False
        )

    def add_page_flate(
        mut self,
        width: Float64,
        height: Float64,
        content: List[UInt8],
        resources: List[UInt8],
    ) raises -> Int:
        # Like add_page but the content stream is zlib(deflate(content)) with
        # /Filter /FlateDecode. pdftotext still extracts text (it decodes Flate).
        var no_annots = List[Int]()
        return self.add_page_ext(
            width, height, content, resources, no_annots, True
        )

    def add_page_ext(
        mut self,
        width: Float64,
        height: Float64,
        content: List[UInt8],
        resources: List[UInt8],
        annot_refs: List[Int],
        flate: Bool,
    ) raises -> Int:
        # 1) content stream object (optionally FlateDecode-compressed).
        var content_obj: Int
        if flate:
            var stream_dict = List[UInt8]()
            append_str(stream_dict, "/Filter /FlateDecode")
            var compressed = self._zlib_wrap(content)
            content_obj = self.add_stream_object(stream_dict, compressed)
        else:
            var empty_dict = List[UInt8]()
            content_obj = self.add_stream_object(empty_dict, content)

        # 2) page object. /Parent points at the Pages object, whose number we
        #    don't know until save() — emit a "@@PARENT@@" sentinel that save()
        #    rewrites once the final layout (and thus Pages number) is known.
        var body = List[UInt8]()
        append_str(body, "<< /Type /Page /Parent ")
        append_str(body, "@@PARENT@@")
        append_str(body, " 0 R /MediaBox [ 0 0 ")
        append_real(body, width)
        body.append(UInt8(ord(" ")))
        append_real(body, height)
        append_str(body, " ] /Contents ")
        append_int(body, content_obj)
        append_str(body, " 0 R /Resources ")
        if len(resources) > 0:
            for i in range(len(resources)):
                body.append(resources[i])
        else:
            append_str(body, "<< >>")
        # Optional annotation references (links, etc., owned by other modules).
        if len(annot_refs) > 0:
            append_str(body, " /Annots [ ")
            for i in range(len(annot_refs)):
                append_int(body, annot_refs[i])
                append_str(body, " 0 R ")
            append_str(body, "]")
        append_str(body, " >>")

        var page_obj = self.add_object(body)
        self.page_numbers.append(page_obj)
        return page_obj

    def save(mut self, path: String) raises:
        # Determine final object numbers.
        var num_user = len(self.objects)
        var catalog_num = num_user + 1
        var pages_num = num_user + 2
        var info_num = 0
        var total = pages_num  # N total objects
        if self.has_info:
            info_num = num_user + 3
            total = info_num

        # Patch each page body: replace "@@PARENT@@" sentinel with pages_num.
        # The sentinel is exactly 10 bytes; replace in-place by rebuilding body.
        var sentinel = String("@@PARENT@@").as_bytes()
        for p in range(len(self.page_numbers)):
            var pn = self.page_numbers[p]
            var idx = pn - 1
            var old_body = self.objects[idx].copy()
            var new_body = List[UInt8]()
            var i = 0
            while i < len(old_body):
                # Try to match sentinel at position i.
                var matched = False
                if i + len(sentinel) <= len(old_body):
                    var ok = True
                    for k in range(len(sentinel)):
                        if old_body[i + k] != sentinel[k]:
                            ok = False
                            break
                    if ok:
                        matched = True
                if matched:
                    append_int(new_body, pages_num)
                    i = i + len(sentinel)
                else:
                    new_body.append(old_body[i])
                    i = i + 1
            self.objects[idx] = new_body.copy()

        # Build Catalog body.
        var catalog = List[UInt8]()
        append_str(catalog, "<< /Type /Catalog /Pages ")
        append_int(catalog, pages_num)
        append_str(catalog, " 0 R")
        if self.outlines_obj > 0:
            append_str(catalog, " /Outlines ")
            append_int(catalog, self.outlines_obj)
            append_str(catalog, " 0 R")
        append_str(catalog, " >>")

        # Build /Info body (only if set_info was called).
        var info = List[UInt8]()
        if self.has_info:
            append_str(info, "<< /Title ")
            pdf_literal_string(info, self.info_title)
            append_str(info, " /Author ")
            pdf_literal_string(info, self.info_author)
            append_str(info, " /Subject ")
            pdf_literal_string(info, self.info_subject)
            append_str(info, " /Creator ")
            pdf_literal_string(info, self.info_creator)
            append_str(info, " /Producer ")
            pdf_literal_string(info, String("MojoPDF"))
            append_str(info, " /CreationDate (D:20260609120000Z) >>")

        # Build Pages body.
        var pages = List[UInt8]()
        append_str(pages, "<< /Type /Pages /Kids [ ")
        for p in range(len(self.page_numbers)):
            append_int(pages, self.page_numbers[p])
            append_str(pages, " 0 R ")
        append_str(pages, "] /Count ")
        append_int(pages, len(self.page_numbers))
        append_str(pages, " >>")

        # Assemble the entire file into one buffer, recording offsets.
        var buf = List[UInt8]()
        # offsets[n-1] = byte offset of object n's "n 0 obj" header.
        var offsets = List[Int]()
        for _i in range(total):
            offsets.append(0)

        # Header + binary comment.
        append_str(buf, "%PDF-1.7\n")
        # Binary marker comment: % followed by 4 high bytes.
        buf.append(UInt8(ord("%")))
        buf.append(UInt8(0xE2))
        buf.append(UInt8(0xE3))
        buf.append(UInt8(0xCF))
        buf.append(UInt8(0xD3))
        buf.append(UInt8(ord("\n")))

        # Helper inlined: write one object n (1-based) with given body.
        # User objects 1..num_user.
        for n in range(1, num_user + 1):
            offsets[n - 1] = len(buf)
            append_int(buf, n)
            append_str(buf, " 0 obj\n")
            var b = self.objects[n - 1].copy()
            for j in range(len(b)):
                buf.append(b[j])
            append_str(buf, "\nendobj\n")

        # Catalog object.
        offsets[catalog_num - 1] = len(buf)
        append_int(buf, catalog_num)
        append_str(buf, " 0 obj\n")
        for j in range(len(catalog)):
            buf.append(catalog[j])
        append_str(buf, "\nendobj\n")

        # Pages object.
        offsets[pages_num - 1] = len(buf)
        append_int(buf, pages_num)
        append_str(buf, " 0 obj\n")
        for j in range(len(pages)):
            buf.append(pages[j])
        append_str(buf, "\nendobj\n")

        # Info object (only if set_info was called).
        if self.has_info:
            offsets[info_num - 1] = len(buf)
            append_int(buf, info_num)
            append_str(buf, " 0 obj\n")
            for j in range(len(info)):
                buf.append(info[j])
            append_str(buf, "\nendobj\n")

        # Cross-reference table.
        var xref_offset = len(buf)
        append_str(buf, "xref\n0 ")
        append_int(buf, total + 1)
        buf.append(UInt8(ord("\n")))
        # Free entry for object 0: "0000000000 65535 f \n" (20 bytes).
        append_str(buf, "0000000000 65535 f \n")
        # In-use entries, object 1..total: "%010d 00000 n \n" (20 bytes each).
        for n in range(1, total + 1):
            self._write_offset10(buf, offsets[n - 1])
            append_str(buf, " 00000 n \n")

        # Trailer.
        append_str(buf, "trailer\n<< /Size ")
        append_int(buf, total + 1)
        append_str(buf, " /Root ")
        append_int(buf, catalog_num)
        append_str(buf, " 0 R")
        if self.has_info:
            append_str(buf, " /Info ")
            append_int(buf, info_num)
            append_str(buf, " 0 R")
        append_str(buf, " >>\nstartxref\n")
        append_int(buf, xref_offset)
        append_str(buf, "\n%%EOF\n")

        # Write the buffer to disk as binary.
        with open(path, "w") as f:
            f.write_bytes(Span(buf))

    def _write_offset10(mut self, mut buf: List[UInt8], value: Int) raises:
        # Write exactly 10 ASCII digits, zero-padded.
        var digits = List[UInt8]()
        var n = value
        if n < 0:
            n = 0
        # Produce digits least-significant first.
        if n == 0:
            digits.append(UInt8(ord("0")))
        else:
            while n > 0:
                digits.append(UInt8(ord("0") + (n % 10)))
                n = n // 10
        # Pad to 10.
        while len(digits) < 10:
            digits.append(UInt8(ord("0")))
        # Emit reversed (most-significant first).
        var k = len(digits)
        while k > 0:
            k = k - 1
            buf.append(digits[k])

    # ------------------------------------------------------------------
    # Shared object-list builder (used by save_encrypted + save_xref_stream).
    # Produces the FINAL ordered list of object BODIES (the bytes between
    # "N 0 obj\n" and "\nendobj") for objects 1..total, with @@PARENT@@
    # already patched. Object N's body is bodies[N-1]. Also returns the
    # catalog/pages/info object numbers. Does NOT touch self.objects or save().
    # ------------------------------------------------------------------
    def _build_bodies(
        self,
        mut bodies: List[List[UInt8]],
        mut catalog_num: Int,
        mut pages_num: Int,
        mut info_num: Int,
    ) raises:
        var num_user = len(self.objects)
        catalog_num = num_user + 1
        pages_num = num_user + 2
        info_num = 0
        if self.has_info:
            info_num = num_user + 3

        # Copy + patch user object bodies (replace @@PARENT@@ with pages_num).
        var sentinel = String("@@PARENT@@").as_bytes()
        # Pre-fill bodies with user objects (patched only for page objects).
        for n in range(1, num_user + 1):
            bodies.append(self.objects[n - 1].copy())
        for p in range(len(self.page_numbers)):
            var pn = self.page_numbers[p]
            var idx = pn - 1
            var old_body = bodies[idx].copy()
            var new_body = List[UInt8]()
            var i = 0
            while i < len(old_body):
                var matched = False
                if i + len(sentinel) <= len(old_body):
                    var ok = True
                    for kk in range(len(sentinel)):
                        if old_body[i + kk] != sentinel[kk]:
                            ok = False
                            break
                    if ok:
                        matched = True
                if matched:
                    append_int(new_body, pages_num)
                    i = i + len(sentinel)
                else:
                    new_body.append(old_body[i])
                    i = i + 1
            bodies[idx] = new_body.copy()

        # Catalog body.
        var catalog = List[UInt8]()
        append_str(catalog, "<< /Type /Catalog /Pages ")
        append_int(catalog, pages_num)
        append_str(catalog, " 0 R")
        if self.outlines_obj > 0:
            append_str(catalog, " /Outlines ")
            append_int(catalog, self.outlines_obj)
            append_str(catalog, " 0 R")
        append_str(catalog, " >>")
        bodies.append(catalog.copy())

        # Pages body.
        var pages = List[UInt8]()
        append_str(pages, "<< /Type /Pages /Kids [ ")
        for p in range(len(self.page_numbers)):
            append_int(pages, self.page_numbers[p])
            append_str(pages, " 0 R ")
        append_str(pages, "] /Count ")
        append_int(pages, len(self.page_numbers))
        append_str(pages, " >>")
        bodies.append(pages.copy())

        # Info body (only if set).
        if self.has_info:
            var info = List[UInt8]()
            append_str(info, "<< /Title ")
            pdf_literal_string(info, self.info_title)
            append_str(info, " /Author ")
            pdf_literal_string(info, self.info_author)
            append_str(info, " /Subject ")
            pdf_literal_string(info, self.info_subject)
            append_str(info, " /Creator ")
            pdf_literal_string(info, self.info_creator)
            append_str(info, " /Producer ")
            pdf_literal_string(info, String("MojoPDF"))
            append_str(info, " /CreationDate (D:20260609120000Z) >>")
            bodies.append(info.copy())

    # ------------------------------------------------------------------
    # PART 2: Cross-reference STREAM save (PDF 1.5).
    # ------------------------------------------------------------------
    def save_xref_stream(mut self, path: String) raises:
        var bodies = List[List[UInt8]]()
        var catalog_num = 0
        var pages_num = 0
        var info_num = 0
        self._build_bodies(bodies, catalog_num, pages_num, info_num)
        var total = len(bodies)  # objects 1..total
        var xref_num = total + 1  # the XRef stream object's own number

        var buf = List[UInt8]()
        var offsets = List[Int]()
        for _i in range(total + 1):  # offsets[0..total] (1-based: obj n at idx n-1)
            offsets.append(0)

        # Header (1.5) + binary marker.
        append_str(buf, "%PDF-1.5\n")
        buf.append(UInt8(ord("%")))
        buf.append(UInt8(0xE2))
        buf.append(UInt8(0xE3))
        buf.append(UInt8(0xCF))
        buf.append(UInt8(0xD3))
        buf.append(UInt8(ord("\n")))

        # Write objects 1..total recording offsets.
        for n in range(1, total + 1):
            offsets[n - 1] = len(buf)
            append_int(buf, n)
            append_str(buf, " 0 obj\n")
            var b = bodies[n - 1].copy()
            for j in range(len(b)):
                buf.append(b[j])
            append_str(buf, "\nendobj\n")

        # The XRef stream object's offset (it is object xref_num).
        var xref_obj_offset = len(buf)
        offsets[xref_num - 1] = xref_obj_offset

        # Build raw xref entries. Size = total + 2 (objects 0..xref_num).
        var size = total + 2
        var entries = List[UInt8]()
        # Object 0: free, type 0, offset 0, gen 65535.
        entries.append(UInt8(0))
        for _q in range(4):
            entries.append(UInt8(0))
        entries.append(UInt8(0xFF))
        entries.append(UInt8(0xFF))
        # Objects 1..total: type 1, byte offset BE(4), gen 0.
        for n in range(1, total + 1):
            entries.append(UInt8(1))
            var off = offsets[n - 1]
            entries.append(UInt8((off >> 24) & 0xFF))
            entries.append(UInt8((off >> 16) & 0xFF))
            entries.append(UInt8((off >> 8) & 0xFF))
            entries.append(UInt8(off & 0xFF))
            entries.append(UInt8(0))
            entries.append(UInt8(0))
        # The XRef stream object itself: type 1, its own offset, gen 0.
        entries.append(UInt8(1))
        entries.append(UInt8((xref_obj_offset >> 24) & 0xFF))
        entries.append(UInt8((xref_obj_offset >> 16) & 0xFF))
        entries.append(UInt8((xref_obj_offset >> 8) & 0xFF))
        entries.append(UInt8(xref_obj_offset & 0xFF))
        entries.append(UInt8(0))
        entries.append(UInt8(0))

        var compressed = self._zlib_wrap(entries)

        # Emit the XRef stream object.
        append_int(buf, xref_num)
        append_str(buf, " 0 obj\n")
        append_str(buf, "<< /Type /XRef /Size ")
        append_int(buf, size)
        append_str(buf, " /Root ")
        append_int(buf, catalog_num)
        append_str(buf, " 0 R")
        if self.has_info:
            append_str(buf, " /Info ")
            append_int(buf, info_num)
            append_str(buf, " 0 R")
        append_str(buf, " /W [ 1 4 2 ] /Index [ 0 ")
        append_int(buf, size)
        append_str(buf, " ] /Filter /FlateDecode /Length ")
        append_int(buf, len(compressed))
        append_str(buf, " >>\nstream\n")
        for j in range(len(compressed)):
            buf.append(compressed[j])
        append_str(buf, "\nendstream\nendobj\n")

        # startxref points at the XRef stream object header. No trailer keyword.
        append_str(buf, "startxref\n")
        append_int(buf, xref_obj_offset)
        append_str(buf, "\n%%EOF\n")

        with open(path, "w") as f:
            f.write_bytes(Span(buf))

    # ------------------------------------------------------------------
    # PART 1: RC4 Standard Security Handler encryption (V=2, R=3, 128-bit).
    # ------------------------------------------------------------------
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
        # Take up to 32 bytes of pw, append PAD to reach exactly 32.
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

    def _xor_each(self, key: List[UInt8], v: Int) raises -> List[UInt8]:
        var out = List[UInt8]()
        for i in range(len(key)):
            out.append(key[i] ^ UInt8(v & 0xFF))
        return out^

    def _first16(self, data: List[UInt8]) raises -> List[UInt8]:
        var out = List[UInt8]()
        for i in range(16):
            out.append(data[i])
        return out^

    def _obj_key(self, enc_key: List[UInt8], onum: Int) raises -> List[UInt8]:
        # Algorithm 1: per-object key = md5(enc_key + onum_LE3 + gen_LE2)[0:16].
        var m = List[UInt8]()
        for i in range(len(enc_key)):
            m.append(enc_key[i])
        m.append(UInt8(onum & 0xFF))
        m.append(UInt8((onum >> 8) & 0xFF))
        m.append(UInt8((onum >> 16) & 0xFF))
        m.append(UInt8(0))  # gen low byte
        m.append(UInt8(0))  # gen high byte
        var d = md5(m)
        # n + 5 = 21 > 16, so take 16 bytes.
        return self._first16(d)

    def _bytes_of_str(self, s: String) raises -> List[UInt8]:
        var src = s.as_bytes()
        var out = List[UInt8]()
        for i in range(len(src)):
            out.append(src[i])
        return out^

    def _hex_lower(self, data: List[UInt8]) raises -> String:
        var HEX = String("0123456789abcdef")
        var hb = HEX.as_bytes()
        var s = String("")
        for i in range(len(data)):
            var byte = Int(data[i])
            s += chr(Int(hb[(byte >> 4) & 0xF]))
            s += chr(Int(hb[byte & 0xF]))
        return s^

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
        # Odd trailing nibble: pad low (per PDF spec).
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

    def _reescape_literal(self, raw: List[UInt8]) raises -> List[UInt8]:
        # Re-escape raw bytes for a PDF literal string body (between the parens).
        # Escape ( ) \ ; octal-escape bytes < 0x20.
        var out = List[UInt8]()
        for i in range(len(raw)):
            var c = raw[i]
            if c == UInt8(ord("(")) or c == UInt8(ord(")")) or c == UInt8(ord("\\")):
                out.append(UInt8(ord("\\")))
                out.append(c)
            elif Int(c) < 0x20:
                # \ooo octal, 3 digits.
                var v = Int(c)
                out.append(UInt8(ord("\\")))
                out.append(UInt8(ord("0") + ((v >> 6) & 0x7)))
                out.append(UInt8(ord("0") + ((v >> 3) & 0x7)))
                out.append(UInt8(ord("0") + (v & 0x7)))
            else:
                out.append(c)
        return out^

    def _decode_literal(self, esc: List[UInt8]) raises -> List[UInt8]:
        # Decode a PDF literal-string BODY (the bytes between '(' and ')').
        # Handles \\ \( \) \n \r \t \b \f \( \) and \ooo octal.
        var out = List[UInt8]()
        var i = 0
        while i < len(esc):
            var c = esc[i]
            if c == UInt8(ord("\\")) and i + 1 < len(esc):
                var n = esc[i + 1]
                if n >= UInt8(ord("0")) and n <= UInt8(ord("7")):
                    # up to 3 octal digits
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
                    if n == UInt8(ord("n")):
                        out.append(UInt8(10))
                    elif n == UInt8(ord("r")):
                        out.append(UInt8(13))
                    elif n == UInt8(ord("t")):
                        out.append(UInt8(9))
                    elif n == UInt8(ord("b")):
                        out.append(UInt8(8))
                    elif n == UInt8(ord("f")):
                        out.append(UInt8(12))
                    else:
                        out.append(n)  # \( \) \\ and others -> literal char
                    i = i + 2
            else:
                out.append(c)
                i = i + 1
        return out^

    def _encrypt_body(self, body: List[UInt8], objkey: List[UInt8]) raises -> List[UInt8]:
        # Walk one object body, RC4-encrypting every literal string ( … ),
        # every hex string < … > (NOT << >>), and the stream payload between
        # "stream\n" and "\nendstream". RC4 preserves length.
        var out = List[UInt8]()
        var i = 0
        var n = len(body)
        while i < n:
            var c = body[i]
            # Stream payload detection: look for "stream\n" or "stream\r\n".
            if c == UInt8(ord("s")) and self._match_at(body, i, "stream"):
                # Confirm it is the stream keyword (preceded by '>' or space/newline).
                # Append "stream" then the newline(s), then encrypt up to endstream.
                # Copy "stream"
                for k in range(6):
                    out.append(body[i + k])
                var p = i + 6
                # newline after stream: either \n or \r\n
                if p < n and body[p] == UInt8(13):
                    out.append(body[p]); p = p + 1
                if p < n and body[p] == UInt8(10):
                    out.append(body[p]); p = p + 1
                # find "\nendstream"  -> the payload ends before the newline that
                # precedes "endstream". Search for "endstream".
                var es = self._find(body, p, "endstream")
                if es < 0:
                    # No endstream; copy rest verbatim.
                    while p < n:
                        out.append(body[p]); p = p + 1
                    i = p
                    continue
                # payload is body[p : es_payload_end] where there's a preceding
                # newline before "endstream". Determine payload end (strip the
                # single \n or \r\n that precedes endstream).
                var pe = es
                # back up over the newline(s) that separate payload from endstream
                if pe > p and body[pe - 1] == UInt8(10):
                    pe = pe - 1
                    if pe > p and body[pe - 1] == UInt8(13):
                        pe = pe - 1
                var payload = List[UInt8]()
                for q in range(p, pe):
                    payload.append(body[q])
                var enc = rc4(objkey, payload)
                for q in range(len(enc)):
                    out.append(enc[q])
                # copy the separator newline(s) + "endstream"
                for q in range(pe, es):
                    out.append(body[q])
                for k in range(len("endstream")):
                    out.append(body[es + k])
                i = es + len("endstream")
                continue
            elif c == UInt8(ord("(")):
                # Literal string. Find matching ')' honoring escapes + nesting.
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
                # body[i+1 : j] is the escaped content; body[j] == ')'
                var inner = List[UInt8]()
                for q in range(i + 1, j):
                    inner.append(body[q])
                var raw = self._decode_literal(inner)
                var encs = rc4(objkey, raw)
                var resc = self._reescape_literal(encs)
                out.append(UInt8(ord("(")))
                for q in range(len(resc)):
                    out.append(resc[q])
                out.append(UInt8(ord(")")))
                i = j + 1
                continue
            elif c == UInt8(ord("<")):
                # Could be "<<" (dict) or "<hex>".
                if i + 1 < n and body[i + 1] == UInt8(ord("<")):
                    out.append(c)
                    out.append(body[i + 1])
                    i = i + 2
                    continue
                # Hex string: find '>'.
                var j = i + 1
                while j < n and body[j] != UInt8(ord(">")):
                    j = j + 1
                var hx = List[UInt8]()
                for q in range(i + 1, j):
                    hx.append(body[q])
                var rawh = self._hex_decode(hx)
                var ench = rc4(objkey, rawh)
                var hexout = self._hex_encode_upper(ench)
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

    def _match_at(self, body: List[UInt8], pos: Int, s: String) raises -> Bool:
        var sb = s.as_bytes()
        if pos + len(sb) > len(body):
            return False
        for k in range(len(sb)):
            if body[pos + k] != sb[k]:
                return False
        return True

    def _find(self, body: List[UInt8], start: Int, s: String) raises -> Int:
        var sb = s.as_bytes()
        var i = start
        while i + len(sb) <= len(body):
            var ok = True
            for k in range(len(sb)):
                if body[i + k] != sb[k]:
                    ok = False
                    break
            if ok:
                return i
            i = i + 1
        return -1

    def save_encrypted(
        mut self, path: String, user_password: String, owner_password: String
    ) raises:
        # Build the ordered object bodies (user + Catalog + Pages + Info).
        var bodies = List[List[UInt8]]()
        var catalog_num = 0
        var pages_num = 0
        var info_num = 0
        self._build_bodies(bodies, catalog_num, pages_num, info_num)
        var num_docobjs = len(bodies)  # objects 1..num_docobjs (the encryptable set)
        var encrypt_num = num_docobjs + 1  # /Encrypt object number (exempt)
        var total = encrypt_num

        var upw = self._bytes_of_str(user_password)
        var opw = self._bytes_of_str(owner_password)
        var pad = self._enc_pad()

        # --- Deterministic 16-byte file ID = md5(all bodies concatenated + len) ---
        var idsrc = List[UInt8]()
        for bi in range(len(bodies)):
            var b = bodies[bi].copy()
            for q in range(len(b)):
                idsrc.append(b[q])
        var tl = len(idsrc)
        idsrc.append(UInt8(tl & 0xFF))
        idsrc.append(UInt8((tl >> 8) & 0xFF))
        idsrc.append(UInt8((tl >> 16) & 0xFF))
        idsrc.append(UInt8((tl >> 24) & 0xFF))
        var id0 = md5(idsrc)  # 16 raw bytes

        # P = -44 (allow print/copy). 32-bit signed, stored little-endian.
        var P = -44

        # --- Algorithm 3: /O ---
        var key0 = md5(self._pad_pw(opw))
        for _r in range(50):
            key0 = md5(self._first16(key0))
        var ownerkey = self._first16(key0)
        var o = self._pad_pw(upw)
        o = rc4(ownerkey, o)
        for i in range(1, 20):
            o = rc4(self._xor_each(ownerkey, i), o)
        var O = o.copy()  # 32 bytes

        # --- Algorithm 2: encryption key ---
        var m = self._pad_pw(upw)
        for q in range(len(O)):
            m.append(O[q])
        # P as 4 bytes little-endian (mask to 32 bits).
        var pmask = P & 0xFFFFFFFF
        m.append(UInt8(pmask & 0xFF))
        m.append(UInt8((pmask >> 8) & 0xFF))
        m.append(UInt8((pmask >> 16) & 0xFF))
        m.append(UInt8((pmask >> 24) & 0xFF))
        for q in range(len(id0)):
            m.append(id0[q])
        var key = md5(m)
        for _r in range(50):
            key = md5(self._first16(key))
        var enc_key = self._first16(key)  # 16 bytes

        # --- Algorithm 5 (R>=3): /U ---
        var um = List[UInt8]()
        for q in range(len(pad)):
            um.append(pad[q])
        for q in range(len(id0)):
            um.append(id0[q])
        var u = md5(um)  # 16 bytes
        u = rc4(enc_key, u)
        for i in range(1, 20):
            u = rc4(self._xor_each(enc_key, i), u)
        # U = 16-byte result + 16 padding bytes (zeros).
        var U = List[UInt8]()
        for q in range(len(u)):
            U.append(u[q])
        while len(U) < 32:
            U.append(UInt8(0))

        # --- Encrypt strings/streams in objects 1..num_docobjs (NOT /Encrypt) ---
        var enc_bodies = List[List[UInt8]]()
        for n in range(1, num_docobjs + 1):
            var objkey = self._obj_key(enc_key, n)
            enc_bodies.append(self._encrypt_body(bodies[n - 1], objkey))

        # --- /Encrypt object body (exempt from encryption) ---
        var encdict = List[UInt8]()
        append_str(encdict, "<< /Filter /Standard /V 2 /R 3 /Length 128 /P ")
        append_int(encdict, P)
        append_str(encdict, " /O ")
        var ohex = self._hex_encode_upper(O)
        encdict.append(UInt8(ord("<")))
        for q in range(len(ohex)):
            encdict.append(ohex[q])
        encdict.append(UInt8(ord(">")))
        append_str(encdict, " /U ")
        var uhex = self._hex_encode_upper(U)
        encdict.append(UInt8(ord("<")))
        for q in range(len(uhex)):
            encdict.append(uhex[q])
        encdict.append(UInt8(ord(">")))
        append_str(encdict, " >>")

        # --- Assemble file with classic xref + trailer (/Encrypt + /ID) ---
        var buf = List[UInt8]()
        var offsets = List[Int]()
        for _i in range(total):
            offsets.append(0)

        append_str(buf, "%PDF-1.7\n")
        buf.append(UInt8(ord("%")))
        buf.append(UInt8(0xE2))
        buf.append(UInt8(0xE3))
        buf.append(UInt8(0xCF))
        buf.append(UInt8(0xD3))
        buf.append(UInt8(ord("\n")))

        # Objects 1..num_docobjs (encrypted bodies).
        for n in range(1, num_docobjs + 1):
            offsets[n - 1] = len(buf)
            append_int(buf, n)
            append_str(buf, " 0 obj\n")
            var b = enc_bodies[n - 1].copy()
            for j in range(len(b)):
                buf.append(b[j])
            append_str(buf, "\nendobj\n")

        # /Encrypt object.
        offsets[encrypt_num - 1] = len(buf)
        append_int(buf, encrypt_num)
        append_str(buf, " 0 obj\n")
        for j in range(len(encdict)):
            buf.append(encdict[j])
        append_str(buf, "\nendobj\n")

        # Classic xref.
        var xref_offset = len(buf)
        append_str(buf, "xref\n0 ")
        append_int(buf, total + 1)
        buf.append(UInt8(ord("\n")))
        append_str(buf, "0000000000 65535 f \n")
        for n in range(1, total + 1):
            self._write_offset10(buf, offsets[n - 1])
            append_str(buf, " 00000 n \n")

        # Trailer with /Encrypt and /ID.
        var id0hex = self._hex_lower(id0)
        append_str(buf, "trailer\n<< /Size ")
        append_int(buf, total + 1)
        append_str(buf, " /Root ")
        append_int(buf, catalog_num)
        append_str(buf, " 0 R")
        if self.has_info:
            append_str(buf, " /Info ")
            append_int(buf, info_num)
            append_str(buf, " 0 R")
        append_str(buf, " /Encrypt ")
        append_int(buf, encrypt_num)
        append_str(buf, " 0 R")
        append_str(buf, " /ID [ <")
        append_str(buf, id0hex)
        append_str(buf, "> <")
        append_str(buf, id0hex)
        append_str(buf, "> ]")
        append_str(buf, " >>\nstartxref\n")
        append_int(buf, xref_offset)
        append_str(buf, "\n%%EOF\n")

        with open(path, "w") as f:
            f.write_bytes(Span(buf))

    # ==================================================================
    # PART 3: MODERN AES ENCRYPTION (AESV2 = V4/R4 128-bit,
    #                                AESV3 = V5/R6 256-bit).
    # Added alongside the RC4 save_encrypted above (which is untouched).
    # ==================================================================

    # ---- low-level AES no-pad helpers (the public aes_cbc_encrypt adds
    # ---- PKCS#7, which we MUST NOT use for the R6 K1/UE/OE/Perms steps). ----
    def _aes_cbc_nopad(self, key: List[UInt8], iv: List[UInt8], data: List[UInt8]) raises -> List[UInt8]:
        # AES-CBC, NO padding. len(data) must be a multiple of 16.
        if (len(data) % 16) != 0:
            raise Error("_aes_cbc_nopad: data not a multiple of 16")
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

    def _aes_ecb_nopad(self, key: List[UInt8], data: List[UInt8]) raises -> List[UInt8]:
        # AES-ECB, NO padding, NO IV. len(data) must be a multiple of 16.
        if (len(data) % 16) != 0:
            raise Error("_aes_ecb_nopad: data not a multiple of 16")
        var rk = _key_expansion(key, _sbox())
        var out = List[UInt8]()
        var off = 0
        while off < len(data):
            var block = List[UInt8]()
            for i in range(16):
                block.append(data[off + i])
            var enc = aes_encrypt_block(rk, block)
            for i in range(16):
                out.append(enc[i])
            off += 16
        return out^

    def _rand_bytes(self, count: Int) raises -> List[UInt8]:
        # Real randomness via std.random.random_ui64(), split into bytes.
        var out = List[UInt8]()
        while len(out) < count:
            var r = random_ui64(0, UInt64(0xFFFFFFFFFFFFFFFF))
            for k in range(8):
                if len(out) >= count:
                    break
                out.append(UInt8((r >> (UInt64(8) * UInt64(k))) & UInt64(0xFF)))
        return out^

    # ------------------------------------------------------------------
    # SHA-512 / SHA-384 (FIPS 180-4) — needed for the R6 hash2b loop.
    # 64-bit words, 80 rounds, BIG-endian. SHA-384 = same compression
    # with a different IV and 48-byte truncation.
    # ------------------------------------------------------------------
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

        # Padding: 0x80, zeros until len ≡ 112 (mod 128), then 128-bit length BE.
        var msg = List[UInt8]()
        for i in range(len(data)):
            msg.append(data[i])
        var bit_len = UInt64(len(data)) * UInt64(8)
        msg.append(UInt8(0x80))
        while (len(msg) % 128) != 112:
            msg.append(UInt8(0))
        # High 64 bits of length are 0 for our small inputs.
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

    def _slice(self, data: List[UInt8], start: Int, end: Int) raises -> List[UInt8]:
        var out = List[UInt8]()
        for i in range(start, end):
            out.append(data[i])
        return out^

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
            # K1 = (password + K + udata) repeated 64 times.
            var seq = self._concat3(password, K, udata)
            var K1 = List[UInt8]()
            for _r in range(64):
                for i in range(len(seq)):
                    K1.append(seq[i])
            # E = AES-128-CBC-no-pad encrypt(K1) with key=K[0:16], iv=K[16:32].
            var ek = self._slice(K, 0, 16)
            var iv = self._slice(K, 16, 32)
            var E = self._aes_cbc_nopad(ek, iv, K1)
            # mod = (sum of first 16 bytes of E) % 3.
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
        return self._slice(K, 0, 32)

    # ------------------------------------------------------------------
    # AES body walker: like _encrypt_body but uses AES-CBC (IV||ct) per
    # string/stream, and patches the stream object's /Length to the new
    # ciphertext length (AES is NOT length-preserving). objkey is the
    # 16-byte per-object key for AESV2 or the 32-byte file key for AESV3.
    # ------------------------------------------------------------------
    def _aes_string(self, plain: List[UInt8]) raises -> List[UInt8]:
        # Reads the active cipher key from a field set by the caller.
        var iv = self._rand_bytes(16)
        var ct = aes_cbc_encrypt(self._aes_active_key, iv, plain)
        var out = List[UInt8]()
        for i in range(16):
            out.append(iv[i])
        for i in range(len(ct)):
            out.append(ct[i])
        return out^

    def _encrypt_body_aes(self, body: List[UInt8]) raises -> List[UInt8]:
        # Stage 1: encrypt the stream payload (if any) and remember its new len.
        var has_stream = False
        var new_stream_len = 0
        var enc_payload = List[UInt8]()
        var stream_kw = self._find(body, 0, "stream")
        var es = -1
        var payload_end = 0
        if stream_kw >= 0:
            var p = stream_kw + 6
            if p < len(body) and body[p] == UInt8(13):
                p = p + 1
            if p < len(body) and body[p] == UInt8(10):
                p = p + 1
            es = self._find(body, p, "endstream")
            if es >= 0:
                var pe = es
                if pe > p and body[pe - 1] == UInt8(10):
                    pe = pe - 1
                    if pe > p and body[pe - 1] == UInt8(13):
                        pe = pe - 1
                var payload = List[UInt8]()
                for q in range(p, pe):
                    payload.append(body[q])
                enc_payload = self._aes_string(payload)
                new_stream_len = len(enc_payload)
                has_stream = True
                payload_end = pe

        # Stage 2: walk body. Patch /Length in the dict, encrypt strings,
        # and substitute the encrypted stream payload.
        var out = List[UInt8]()
        var i = 0
        var n = len(body)
        # Boundary that separates dict (where /Length lives) from payload.
        while i < n:
            var c = body[i]
            # Stream payload region: emit "stream"+newline + enc payload + rest.
            if has_stream and i == stream_kw:
                for k in range(6):
                    out.append(body[i + k])
                var p = i + 6
                if p < n and body[p] == UInt8(13):
                    out.append(body[p]); p = p + 1
                if p < n and body[p] == UInt8(10):
                    out.append(body[p])
                for q in range(len(enc_payload)):
                    out.append(enc_payload[q])
                # copy separator newline(s) between original payload and endstream
                for q in range(payload_end, es):
                    out.append(body[q])
                for k in range(len("endstream")):
                    out.append(body[es + k])
                i = es + len("endstream")
                continue
            # /Length token (only patch when this object has a stream).
            if has_stream and c == UInt8(ord("/")) and self._match_at(body, i, "/Length"):
                # Make sure it's followed by whitespace then digits (the stream
                # length), not "/LengthX". Emit "/Length " then patched number.
                var after = i + len("/Length")
                # require a delimiter (space/newline) right after.
                if after < n and (body[after] == UInt8(32) or body[after] == UInt8(10) or body[after] == UInt8(13)):
                    # copy "/Length"
                    for k in range(len("/Length")):
                        out.append(body[i + k])
                    var j = after
                    # copy whitespace
                    while j < n and (body[j] == UInt8(32) or body[j] == UInt8(10) or body[j] == UInt8(13)):
                        out.append(body[j]); j = j + 1
                    # skip the old digits
                    while j < n and body[j] >= UInt8(ord("0")) and body[j] <= UInt8(ord("9")):
                        j = j + 1
                    append_int(out, new_stream_len)
                    i = j
                    continue
            if c == UInt8(ord("(")):
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
                var raw = self._decode_literal(inner)
                var encs = self._aes_string(raw)
                var resc = self._reescape_literal(encs)
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
                var ench = self._aes_string(rawh)
                var hexout = self._hex_encode_upper(ench)
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

    # ------------------------------------------------------------------
    # PART A: AESV2 (V=4, R=4, 128-bit). Same O/U/key derivation as RC4 R3,
    # but per-object key gets the "sAlT" suffix and per-object cipher is
    # AES-128-CBC with a random IV prepended.
    # ------------------------------------------------------------------
    def _obj_key_aes(self, enc_key: List[UInt8], onum: Int) raises -> List[UInt8]:
        # Algorithm 1 (AESV2): md5(enc_key + onum_LE3 + gen_LE2 + "sAlT")[0:16].
        var m = List[UInt8]()
        for i in range(len(enc_key)):
            m.append(enc_key[i])
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
        return self._first16(d)

    def save_encrypted_aes128(
        mut self, path: String, user_password: String, owner_password: String
    ) raises:
        var bodies = List[List[UInt8]]()
        var catalog_num = 0
        var pages_num = 0
        var info_num = 0
        self._build_bodies(bodies, catalog_num, pages_num, info_num)
        var num_docobjs = len(bodies)
        var encrypt_num = num_docobjs + 1
        var total = encrypt_num

        var upw = self._bytes_of_str(user_password)
        var opw = self._bytes_of_str(owner_password)
        var pad = self._enc_pad()

        # Deterministic 16-byte /ID (same scheme as RC4 path).
        var idsrc = List[UInt8]()
        for bi in range(len(bodies)):
            var b = bodies[bi].copy()
            for q in range(len(b)):
                idsrc.append(b[q])
        var tl = len(idsrc)
        idsrc.append(UInt8(tl & 0xFF))
        idsrc.append(UInt8((tl >> 8) & 0xFF))
        idsrc.append(UInt8((tl >> 16) & 0xFF))
        idsrc.append(UInt8((tl >> 24) & 0xFF))
        var id0 = md5(idsrc)

        var P = -44

        # Algorithm 3: /O (R3 logic).
        var key0 = md5(self._pad_pw(opw))
        for _r in range(50):
            key0 = md5(self._first16(key0))
        var ownerkey = self._first16(key0)
        var o = self._pad_pw(upw)
        o = rc4(ownerkey, o)
        for i in range(1, 20):
            o = rc4(self._xor_each(ownerkey, i), o)
        var O = o.copy()

        # Algorithm 2: encryption key.
        var m = self._pad_pw(upw)
        for q in range(len(O)):
            m.append(O[q])
        var pmask = P & 0xFFFFFFFF
        m.append(UInt8(pmask & 0xFF))
        m.append(UInt8((pmask >> 8) & 0xFF))
        m.append(UInt8((pmask >> 16) & 0xFF))
        m.append(UInt8((pmask >> 24) & 0xFF))
        for q in range(len(id0)):
            m.append(id0[q])
        var key = md5(m)
        for _r in range(50):
            key = md5(self._first16(key))
        var enc_key = self._first16(key)

        # Algorithm 5 (R>=3): /U.
        var um = List[UInt8]()
        for q in range(len(pad)):
            um.append(pad[q])
        for q in range(len(id0)):
            um.append(id0[q])
        var u = md5(um)
        u = rc4(enc_key, u)
        for i in range(1, 20):
            u = rc4(self._xor_each(enc_key, i), u)
        var U = List[UInt8]()
        for q in range(len(u)):
            U.append(u[q])
        while len(U) < 32:
            U.append(UInt8(0))

        # Encrypt strings/streams (objects 1..num_docobjs) with AES-128-CBC.
        var enc_bodies = List[List[UInt8]]()
        for n in range(1, num_docobjs + 1):
            self._aes_active_key = self._obj_key_aes(enc_key, n)
            enc_bodies.append(self._encrypt_body_aes(bodies[n - 1]))

        # /Encrypt dict (AESV2 crypt filter).
        var encdict = List[UInt8]()
        append_str(encdict, "<< /Filter /Standard /V 4 /R 4 /Length 128")
        append_str(encdict, " /CF << /StdCF << /Type /CryptFilter /CFM /AESV2 /Length 16 /AuthEvent /DocOpen >> >>")
        append_str(encdict, " /StmF /StdCF /StrF /StdCF /P ")
        append_int(encdict, P)
        append_str(encdict, " /O ")
        var ohex = self._hex_encode_upper(O)
        encdict.append(UInt8(ord("<")))
        for q in range(len(ohex)):
            encdict.append(ohex[q])
        encdict.append(UInt8(ord(">")))
        append_str(encdict, " /U ")
        var uhex = self._hex_encode_upper(U)
        encdict.append(UInt8(ord("<")))
        for q in range(len(uhex)):
            encdict.append(uhex[q])
        encdict.append(UInt8(ord(">")))
        append_str(encdict, " >>")

        self._write_encrypted_file(path, enc_bodies, encdict, encrypt_num, total, catalog_num, info_num, id0)

    # ------------------------------------------------------------------
    # PART B: AESV3 (V=5, R=6, 256-bit). No per-object key; the 32-byte
    # file key encrypts everything (Algorithm 2.B for U/O/UE/OE/Perms).
    # ------------------------------------------------------------------
    def save_encrypted_aes256(
        mut self, path: String, user_password: String, owner_password: String
    ) raises:
        var bodies = List[List[UInt8]]()
        var catalog_num = 0
        var pages_num = 0
        var info_num = 0
        self._build_bodies(bodies, catalog_num, pages_num, info_num)
        var num_docobjs = len(bodies)
        var encrypt_num = num_docobjs + 1
        var total = encrypt_num

        var upw = self._bytes_of_str(user_password)
        var opw = self._bytes_of_str(owner_password)

        # Deterministic /ID (kept for completeness / readers that want it).
        var idsrc = List[UInt8]()
        for bi in range(len(bodies)):
            var b = bodies[bi].copy()
            for q in range(len(b)):
                idsrc.append(b[q])
        var tl = len(idsrc)
        idsrc.append(UInt8(tl & 0xFF))
        idsrc.append(UInt8((tl >> 8) & 0xFF))
        idsrc.append(UInt8((tl >> 16) & 0xFF))
        idsrc.append(UInt8((tl >> 24) & 0xFF))
        var id0 = md5(idsrc)

        var P = -44

        # 32-byte file key + per-purpose salts.
        var FK = self._rand_bytes(32)
        var VS_u = self._rand_bytes(8)
        var KS_u = self._rand_bytes(8)
        var VS_o = self._rand_bytes(8)
        var KS_o = self._rand_bytes(8)
        var empty = List[UInt8]()
        var zero_iv = List[UInt8]()
        for _z in range(16):
            zero_iv.append(UInt8(0))

        # U (48) = hash2b(user_pw, VS_u, empty)[0:32] || VS_u || KS_u.
        var u_hash = self._hash2b(upw, VS_u, empty)
        var U = List[UInt8]()
        for q in range(32):
            U.append(u_hash[q])
        for q in range(8):
            U.append(VS_u[q])
        for q in range(8):
            U.append(KS_u[q])

        # UE = AES-256-CBC-no-pad(key=hash2b(user_pw, KS_u, empty), iv=0, FK).
        var uek = self._hash2b(upw, KS_u, empty)
        var UE = self._aes_cbc_nopad(uek, zero_iv, FK)

        # O (48) = hash2b(owner_pw, VS_o, U_48)[0:32] || VS_o || KS_o.
        var o_hash = self._hash2b(opw, VS_o, U)
        var O = List[UInt8]()
        for q in range(32):
            O.append(o_hash[q])
        for q in range(8):
            O.append(VS_o[q])
        for q in range(8):
            O.append(KS_o[q])

        # OE = AES-256-CBC-no-pad(key=hash2b(owner_pw, KS_o, U_48), iv=0, FK).
        var oek = self._hash2b(opw, KS_o, U)
        var OE = self._aes_cbc_nopad(oek, zero_iv, FK)

        # Perms (16): P(4 LE) + FFFFFFFF(4) + 'T'/'F' + 'adb' + 4 random,
        # then AES-256-ECB-no-pad with FK.
        var pmask = P & 0xFFFFFFFF
        var permblk = List[UInt8]()
        permblk.append(UInt8(pmask & 0xFF))
        permblk.append(UInt8((pmask >> 8) & 0xFF))
        permblk.append(UInt8((pmask >> 16) & 0xFF))
        permblk.append(UInt8((pmask >> 24) & 0xFF))
        permblk.append(UInt8(0xFF))
        permblk.append(UInt8(0xFF))
        permblk.append(UInt8(0xFF))
        permblk.append(UInt8(0xFF))
        permblk.append(UInt8(ord("T")))  # EncryptMetadata = true
        permblk.append(UInt8(ord("a")))
        permblk.append(UInt8(ord("d")))
        permblk.append(UInt8(ord("b")))
        var prand = self._rand_bytes(4)
        for q in range(4):
            permblk.append(prand[q])
        var Perms = self._aes_ecb_nopad(FK, permblk)

        # Encrypt strings/streams with the file key directly (AES-256-CBC).
        var enc_bodies = List[List[UInt8]]()
        self._aes_active_key = FK.copy()
        for n in range(1, num_docobjs + 1):
            enc_bodies.append(self._encrypt_body_aes(bodies[n - 1]))

        # /Encrypt dict (AESV3 crypt filter).
        var encdict = List[UInt8]()
        append_str(encdict, "<< /Filter /Standard /V 5 /R 6 /Length 256")
        append_str(encdict, " /CF << /StdCF << /Type /CryptFilter /CFM /AESV3 /Length 32 /AuthEvent /DocOpen >> >>")
        append_str(encdict, " /StmF /StdCF /StrF /StdCF /P ")
        append_int(encdict, P)
        self._append_hexstr(encdict, " /O ", O)
        self._append_hexstr(encdict, " /U ", U)
        self._append_hexstr(encdict, " /OE ", OE)
        self._append_hexstr(encdict, " /UE ", UE)
        self._append_hexstr(encdict, " /Perms ", Perms)
        append_str(encdict, " >>")

        self._write_encrypted_file(path, enc_bodies, encdict, encrypt_num, total, catalog_num, info_num, id0)

    def _append_hexstr(self, mut dst: List[UInt8], key: String, data: List[UInt8]) raises:
        append_str(dst, key)
        var hx = self._hex_encode_upper(data)
        dst.append(UInt8(ord("<")))
        for q in range(len(hx)):
            dst.append(hx[q])
        dst.append(UInt8(ord(">")))

    # Shared file assembler for both AES paths (classic xref + /Encrypt + /ID).
    def _write_encrypted_file(
        mut self,
        path: String,
        enc_bodies: List[List[UInt8]],
        encdict: List[UInt8],
        encrypt_num: Int,
        total: Int,
        catalog_num: Int,
        info_num: Int,
        id0: List[UInt8],
    ) raises:
        var num_docobjs = encrypt_num - 1
        var buf = List[UInt8]()
        var offsets = List[Int]()
        for _i in range(total):
            offsets.append(0)

        append_str(buf, "%PDF-1.7\n")
        buf.append(UInt8(ord("%")))
        buf.append(UInt8(0xE2))
        buf.append(UInt8(0xE3))
        buf.append(UInt8(0xCF))
        buf.append(UInt8(0xD3))
        buf.append(UInt8(ord("\n")))

        for n in range(1, num_docobjs + 1):
            offsets[n - 1] = len(buf)
            append_int(buf, n)
            append_str(buf, " 0 obj\n")
            var b = enc_bodies[n - 1].copy()
            for j in range(len(b)):
                buf.append(b[j])
            append_str(buf, "\nendobj\n")

        offsets[encrypt_num - 1] = len(buf)
        append_int(buf, encrypt_num)
        append_str(buf, " 0 obj\n")
        for j in range(len(encdict)):
            buf.append(encdict[j])
        append_str(buf, "\nendobj\n")

        var xref_offset = len(buf)
        append_str(buf, "xref\n0 ")
        append_int(buf, total + 1)
        buf.append(UInt8(ord("\n")))
        append_str(buf, "0000000000 65535 f \n")
        for n in range(1, total + 1):
            self._write_offset10(buf, offsets[n - 1])
            append_str(buf, " 00000 n \n")

        var id0hex = self._hex_lower(id0)
        append_str(buf, "trailer\n<< /Size ")
        append_int(buf, total + 1)
        append_str(buf, " /Root ")
        append_int(buf, catalog_num)
        append_str(buf, " 0 R")
        if self.has_info:
            append_str(buf, " /Info ")
            append_int(buf, info_num)
            append_str(buf, " 0 R")
        append_str(buf, " /Encrypt ")
        append_int(buf, encrypt_num)
        append_str(buf, " 0 R")
        append_str(buf, " /ID [ <")
        append_str(buf, id0hex)
        append_str(buf, "> <")
        append_str(buf, id0hex)
        append_str(buf, "> ]")
        append_str(buf, " >>\nstartxref\n")
        append_int(buf, xref_offset)
        append_str(buf, "\n%%EOF\n")

        with open(path, "w") as f:
            f.write_bytes(Span(buf))

    # ==================================================================
    # ADDED FEATURE 1: enable_xmp toggle.
    # ADDED FEATURE 2: save_compressed (ObjStm + XRef stream).
    # ADDED FEATURE 3: XMP /Metadata stream + deterministic /ID helpers.
    # These ADD-ONLY methods do not touch save()/save_encrypted*/
    # save_xref_stream above.
    # ==================================================================

    def enable_xmp(mut self):
        # Opt in to XMP /Metadata + document /ID on the save paths that honor
        # it (currently save_with_xmp and save_compressed()).
        self.xmp_enabled = True

    # ---- small helpers -------------------------------------------------

    def _body_has_stream(self, body: List[UInt8]) raises -> Bool:
        # An object is "stream-bearing" if its body contains the token
        # "stream" (content streams, images, FontFile2, ToUnicode, etc.).
        # Such objects cannot live inside an /ObjStm.
        return self._find(body, 0, "stream") >= 0

    def _xml_escape(self, s: String) raises -> String:
        # Escape &, <, > for embedding inside XMP packet text.
        var out = String("")
        var b = s.as_bytes()
        for i in range(len(b)):
            var c = Int(b[i])
            if c == ord("&"):
                out += "&amp;"
            elif c == ord("<"):
                out += "&lt;"
            elif c == ord(">"):
                out += "&gt;"
            else:
                out += chr(c)
        return out^

    def _build_xmp_packet(self) raises -> List[UInt8]:
        # Build an archival XMP packet from set_info fields (escaped).
        var title = self._xml_escape(self.info_title)
        var author = self._xml_escape(self.info_author)
        var creator = self._xml_escape(self.info_creator)
        var s = String("")
        s += '<?xpacket begin="\xEF\xBB\xBF" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        s += '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        s += '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        s += '<rdf:Description rdf:about=""'
        s += ' xmlns:dc="http://purl.org/dc/elements/1.1/"'
        s += ' xmlns:pdf="http://ns.adobe.com/pdf/1.3/"'
        s += ' xmlns:xmp="http://ns.adobe.com/xap/1.0/">\n'
        s += "<dc:title><rdf:Alt><rdf:li xml:lang=\"x-default\">"
        s += title
        s += "</rdf:li></rdf:Alt></dc:title>\n"
        s += "<dc:creator><rdf:Seq><rdf:li>"
        s += author
        s += "</rdf:li></rdf:Seq></dc:creator>\n"
        s += "<pdf:Producer>MojoPDF</pdf:Producer>\n"
        s += "<xmp:CreatorTool>"
        s += creator
        s += "</xmp:CreatorTool>\n"
        s += "</rdf:Description>\n</rdf:RDF>\n</x:xmpmeta>\n"
        s += '<?xpacket end="w"?>'
        var out = List[UInt8]()
        var sb = s.as_bytes()
        for i in range(len(sb)):
            out.append(sb[i])
        return out^

    def _metadata_stream_body(self, packet: List[UInt8]) raises -> List[UInt8]:
        # Build a /Type /Metadata /Subtype /XML object body (uncompressed).
        var body = List[UInt8]()
        append_str(body, "<< /Type /Metadata /Subtype /XML /Length ")
        append_int(body, len(packet))
        append_str(body, " >>\nstream\n")
        for i in range(len(packet)):
            body.append(packet[i])
        append_str(body, "\nendstream")
        return body^

    def _doc_id(self, bodies: List[List[UInt8]]) raises -> List[UInt8]:
        # Deterministic 16-byte document /ID = md5(all bodies + total length).
        var idsrc = List[UInt8]()
        for bi in range(len(bodies)):
            var b = bodies[bi].copy()
            for q in range(len(b)):
                idsrc.append(b[q])
        var tl = len(idsrc)
        idsrc.append(UInt8(tl & 0xFF))
        idsrc.append(UInt8((tl >> 8) & 0xFF))
        idsrc.append(UInt8((tl >> 16) & 0xFF))
        idsrc.append(UInt8((tl >> 24) & 0xFF))
        return md5(idsrc)

    def _inject_metadata_ref(
        self, catalog_body: List[UInt8], meta_num: Int
    ) raises -> List[UInt8]:
        # Insert " /Metadata <meta_num> 0 R " just before the FINAL ">>" of a
        # Catalog dict body.
        var last = -1
        var i = 0
        while i + 1 < len(catalog_body):
            if (
                catalog_body[i] == UInt8(ord(">"))
                and catalog_body[i + 1] == UInt8(ord(">"))
            ):
                last = i
            i = i + 1
        if last < 0:
            return catalog_body.copy()  # defensive: no closing ">>"
        var out = List[UInt8]()
        for k in range(last):
            out.append(catalog_body[k])
        append_str(out, " /Metadata ")
        append_int(out, meta_num)
        append_str(out, " 0 R ")
        for k in range(last, len(catalog_body)):
            out.append(catalog_body[k])
        return out^

    # ------------------------------------------------------------------
    # ADDED save() variant that emits XMP /Metadata + /ID when xmp_enabled.
    # Falls back to the original (untouched) save() when xmp is off. The
    # Catalog here gains /Metadata <n> 0 R.
    # ------------------------------------------------------------------
    def save_with_xmp(mut self, path: String) raises:
        if not self.xmp_enabled:
            self.save(path)
            return

        var bodies = List[List[UInt8]]()
        var catalog_num = 0
        var pages_num = 0
        var info_num = 0
        self._build_bodies(bodies, catalog_num, pages_num, info_num)
        var num_docobjs = len(bodies)

        var id0 = self._doc_id(bodies)
        var id0hex = self._hex_lower(id0)

        var meta_num = num_docobjs + 1
        var packet = self._build_xmp_packet()
        var cat_idx = catalog_num - 1
        bodies[cat_idx] = self._inject_metadata_ref(bodies[cat_idx], meta_num)
        bodies.append(self._metadata_stream_body(packet))
        var total = len(bodies)

        var buf = List[UInt8]()
        var offsets = List[Int]()
        for _i in range(total):
            offsets.append(0)

        append_str(buf, "%PDF-1.7\n")
        buf.append(UInt8(ord("%")))
        buf.append(UInt8(0xE2))
        buf.append(UInt8(0xE3))
        buf.append(UInt8(0xCF))
        buf.append(UInt8(0xD3))
        buf.append(UInt8(ord("\n")))

        for n in range(1, total + 1):
            offsets[n - 1] = len(buf)
            append_int(buf, n)
            append_str(buf, " 0 obj\n")
            var b = bodies[n - 1].copy()
            for j in range(len(b)):
                buf.append(b[j])
            append_str(buf, "\nendobj\n")

        var xref_offset = len(buf)
        append_str(buf, "xref\n0 ")
        append_int(buf, total + 1)
        buf.append(UInt8(ord("\n")))
        append_str(buf, "0000000000 65535 f \n")
        for n in range(1, total + 1):
            self._write_offset10(buf, offsets[n - 1])
            append_str(buf, " 00000 n \n")

        append_str(buf, "trailer\n<< /Size ")
        append_int(buf, total + 1)
        append_str(buf, " /Root ")
        append_int(buf, catalog_num)
        append_str(buf, " 0 R")
        if self.has_info:
            append_str(buf, " /Info ")
            append_int(buf, info_num)
            append_str(buf, " 0 R")
        append_str(buf, " /ID [ <")
        append_str(buf, id0hex)
        append_str(buf, "> <")
        append_str(buf, id0hex)
        append_str(buf, "> ]")
        append_str(buf, " >>\nstartxref\n")
        append_int(buf, xref_offset)
        append_str(buf, "\n%%EOF\n")

        with open(path, "w") as f:
            f.write_bytes(Span(buf))

    # ------------------------------------------------------------------
    # ADDED FEATURE: save_compressed — PDF 1.5 object streams (/ObjStm) +
    # cross-reference stream. Packs all non-stream objects into one ObjStm
    # to shrink small files. When xmp_enabled, also emits an XMP /Metadata
    # stream (kept as a regular object, since it IS a stream) and a /ID in
    # the XRef-stream dict.
    # ------------------------------------------------------------------
    def save_compressed(mut self, path: String) raises:
        # 1) Build the final ordered object bodies (objects 1..num_docobjs).
        var bodies = List[List[UInt8]]()
        var catalog_num = 0
        var pages_num = 0
        var info_num = 0
        self._build_bodies(bodies, catalog_num, pages_num, info_num)
        var num_docobjs = len(bodies)

        var id0 = self._doc_id(bodies)
        var id0hex = self._hex_lower(id0)

        # 2) Optional XMP metadata: a stream object kept as a regular object.
        var have_meta = self.xmp_enabled
        var meta_num = 0
        if have_meta:
            meta_num = num_docobjs + 1
            var cat_idx = catalog_num - 1
            bodies[cat_idx] = self._inject_metadata_ref(
                bodies[cat_idx], meta_num
            )
            var packet = self._build_xmp_packet()
            bodies.append(self._metadata_stream_body(packet))

        var num_objs = len(bodies)  # final document objects 1..num_objs
        var objstm_num = num_objs + 1
        var xref_num = num_objs + 2
        var max_num = xref_num

        # 3) Classify: stream-bearing -> regular; else packable.
        var is_packable = List[Bool]()
        for _i in range(num_objs + 1):  # index by 1-based object number
            is_packable.append(False)
        var packable_nums = List[Int]()
        for n in range(1, num_objs + 1):
            if not self._body_has_stream(bodies[n - 1]):
                is_packable[n] = True
                packable_nums.append(n)

        # 4) Build the ObjStm: header "onum off ..." then concatenated bodies.
        var index_in_objstm = List[Int]()
        for _i in range(num_objs + 1):
            index_in_objstm.append(-1)

        var concat = List[UInt8]()
        var rel_offsets = List[Int]()
        for k in range(len(packable_nums)):
            var n = packable_nums[k]
            rel_offsets.append(len(concat))
            index_in_objstm[n] = k
            var b = bodies[n - 1].copy()
            for j in range(len(b)):
                concat.append(b[j])
            concat.append(UInt8(ord("\n")))  # separate bodies

        var header = List[UInt8]()
        for k in range(len(packable_nums)):
            append_int(header, packable_nums[k])
            header.append(UInt8(ord(" ")))
            append_int(header, rel_offsets[k])
            header.append(UInt8(ord(" ")))

        var first = len(header)  # /First = header byte length
        var objstm_plain = List[UInt8]()
        for j in range(len(header)):
            objstm_plain.append(header[j])
        for j in range(len(concat)):
            objstm_plain.append(concat[j])
        var objstm_comp = self._zlib_wrap(objstm_plain)

        # 5) Assemble file.
        var buf = List[UInt8]()
        var obj_offset = List[Int]()
        for _i in range(max_num + 1):
            obj_offset.append(0)

        append_str(buf, "%PDF-1.5\n")
        buf.append(UInt8(ord("%")))
        buf.append(UInt8(0xE2))
        buf.append(UInt8(0xE3))
        buf.append(UInt8(0xCF))
        buf.append(UInt8(0xD3))
        buf.append(UInt8(ord("\n")))

        # 5a) Stream-bearing regular objects (object-number order).
        for n in range(1, num_objs + 1):
            if is_packable[n]:
                continue
            obj_offset[n] = len(buf)
            append_int(buf, n)
            append_str(buf, " 0 obj\n")
            var b = bodies[n - 1].copy()
            for j in range(len(b)):
                buf.append(b[j])
            append_str(buf, "\nendobj\n")

        # 5b) ObjStm object.
        obj_offset[objstm_num] = len(buf)
        append_int(buf, objstm_num)
        append_str(buf, " 0 obj\n")
        append_str(buf, "<< /Type /ObjStm /N ")
        append_int(buf, len(packable_nums))
        append_str(buf, " /First ")
        append_int(buf, first)
        append_str(buf, " /Filter /FlateDecode /Length ")
        append_int(buf, len(objstm_comp))
        append_str(buf, " >>\nstream\n")
        for j in range(len(objstm_comp)):
            buf.append(objstm_comp[j])
        append_str(buf, "\nendstream\nendobj\n")

        # 5c) XRef stream object.
        var xref_obj_offset = len(buf)
        obj_offset[xref_num] = xref_obj_offset

        var size = max_num + 1  # /Index covers objects 0..max_num
        var entries = List[UInt8]()
        # Object 0: free, type 0.
        entries.append(UInt8(0))
        entries.append(UInt8(0)); entries.append(UInt8(0))
        entries.append(UInt8(0)); entries.append(UInt8(0))
        entries.append(UInt8(0xFF)); entries.append(UInt8(0xFF))

        for n in range(1, num_objs + 1):
            if is_packable[n]:
                # type 2: (objstm number, index within objstm).
                entries.append(UInt8(2))
                entries.append(UInt8((objstm_num >> 24) & 0xFF))
                entries.append(UInt8((objstm_num >> 16) & 0xFF))
                entries.append(UInt8((objstm_num >> 8) & 0xFF))
                entries.append(UInt8(objstm_num & 0xFF))
                var idx = index_in_objstm[n]
                entries.append(UInt8((idx >> 8) & 0xFF))
                entries.append(UInt8(idx & 0xFF))
            else:
                # type 1: byte offset, gen 0.
                var off = obj_offset[n]
                entries.append(UInt8(1))
                entries.append(UInt8((off >> 24) & 0xFF))
                entries.append(UInt8((off >> 16) & 0xFF))
                entries.append(UInt8((off >> 8) & 0xFF))
                entries.append(UInt8(off & 0xFF))
                entries.append(UInt8(0)); entries.append(UInt8(0))

        # ObjStm object: type 1 at its offset.
        var ostmoff = obj_offset[objstm_num]
        entries.append(UInt8(1))
        entries.append(UInt8((ostmoff >> 24) & 0xFF))
        entries.append(UInt8((ostmoff >> 16) & 0xFF))
        entries.append(UInt8((ostmoff >> 8) & 0xFF))
        entries.append(UInt8(ostmoff & 0xFF))
        entries.append(UInt8(0)); entries.append(UInt8(0))

        # XRef object itself: type 1 at its own offset.
        entries.append(UInt8(1))
        entries.append(UInt8((xref_obj_offset >> 24) & 0xFF))
        entries.append(UInt8((xref_obj_offset >> 16) & 0xFF))
        entries.append(UInt8((xref_obj_offset >> 8) & 0xFF))
        entries.append(UInt8(xref_obj_offset & 0xFF))
        entries.append(UInt8(0)); entries.append(UInt8(0))

        var xref_comp = self._zlib_wrap(entries)

        append_int(buf, xref_num)
        append_str(buf, " 0 obj\n")
        append_str(buf, "<< /Type /XRef /Size ")
        append_int(buf, size)
        append_str(buf, " /Root ")
        append_int(buf, catalog_num)
        append_str(buf, " 0 R")
        if self.has_info:
            append_str(buf, " /Info ")
            append_int(buf, info_num)
            append_str(buf, " 0 R")
        if self.xmp_enabled:
            append_str(buf, " /ID [ <")
            append_str(buf, id0hex)
            append_str(buf, "> <")
            append_str(buf, id0hex)
            append_str(buf, "> ]")
        append_str(buf, " /W [ 1 4 2 ] /Index [ 0 ")
        append_int(buf, size)
        append_str(buf, " ] /Filter /FlateDecode /Length ")
        append_int(buf, len(xref_comp))
        append_str(buf, " >>\nstream\n")
        for j in range(len(xref_comp)):
            buf.append(xref_comp[j])
        append_str(buf, "\nendstream\nendobj\n")

        # startxref -> XRef stream object offset. No trailer keyword.
        append_str(buf, "startxref\n")
        append_int(buf, xref_obj_offset)
        append_str(buf, "\n%%EOF\n")

        with open(path, "w") as f:
            f.write_bytes(Span(buf))

    # ==================================================================
    # ADDED FEATURE: PDF/A-2b conformant save path (PDF 1.7 base).
    #
    # ADD-ONLY: builds on the existing _build_bodies / _zlib_wrap / _doc_id /
    # _inject_metadata_ref helpers and mirrors save_with_xmp's classic-xref +
    # trailer assembly. Produces the structural elements of PDF/A-2b that can
    # be constructed without a validator:
    #   * all fonts embedded (caller's responsibility; uses embed_font),
    #   * an /OutputIntent with an embedded sRGB ICC profile (FlateDecode),
    #   * an XMP /Metadata stream carrying the pdfaid identification schema
    #     (<pdfaid:part>2</pdfaid:part> <pdfaid:conformance>B</...>),
    #   * a deterministic document /ID in the trailer.
    # No encryption, no /JavaScript; DeviceRGB is allowed given the RGB intent.
    #
    # NOTE: veraPDF was NOT available in the build environment, so full PDF/A-2b
    # conformance was NOT machine-verified. The above are STRUCTURAL guarantees.
    # ==================================================================

    def _read_file_bytes(self, p: String) raises -> List[UInt8]:
        # Binary read NON-with (per environment idiom): open "r", read_bytes.
        var f = open(p, "r")
        var d = f.read_bytes()
        f.close()
        var out = List[UInt8]()
        for i in range(len(d)):
            out.append(d[i])
        return out^

    def _load_srgb_icc(self) raises -> List[UInt8]:
        # Read an sRGB ICC profile from a known system location. Tries the
        # colord profile first, then the ghostscript fallback.
        var primary = String("/usr/share/color/icc/colord/sRGB.icc")
        var fallback = String("/usr/share/color/icc/ghostscript/srgb.icc")
        try:
            return self._read_file_bytes(primary)
        except:
            return self._read_file_bytes(fallback)

    def _build_xmp_packet_pdfa(self) raises -> List[UInt8]:
        # Like _build_xmp_packet but ADDS the PDF/A identification schema
        # (xmlns:pdfaid + pdfaid:part 2 / pdfaid:conformance B). Standalone so
        # the existing _build_xmp_packet stays untouched.
        var title = self._xml_escape(self.info_title)
        var author = self._xml_escape(self.info_author)
        var creator = self._xml_escape(self.info_creator)
        var s = String("")
        s += '<?xpacket begin="\xEF\xBB\xBF" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        s += '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        s += '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        s += '<rdf:Description rdf:about=""'
        s += ' xmlns:dc="http://purl.org/dc/elements/1.1/"'
        s += ' xmlns:pdf="http://ns.adobe.com/pdf/1.3/"'
        s += ' xmlns:xmp="http://ns.adobe.com/xap/1.0/"'
        s += ' xmlns:pdfaid="http://www.aiim.org/pdfa/ns/id/">\n'
        s += "<pdfaid:part>2</pdfaid:part>\n"
        s += "<pdfaid:conformance>B</pdfaid:conformance>\n"
        s += "<dc:title><rdf:Alt><rdf:li xml:lang=\"x-default\">"
        s += title
        s += "</rdf:li></rdf:Alt></dc:title>\n"
        s += "<dc:creator><rdf:Seq><rdf:li>"
        s += author
        s += "</rdf:li></rdf:Seq></dc:creator>\n"
        s += "<pdf:Producer>MojoPDF</pdf:Producer>\n"
        s += "<xmp:CreatorTool>"
        s += creator
        s += "</xmp:CreatorTool>\n"
        s += "</rdf:Description>\n</rdf:RDF>\n</x:xmpmeta>\n"
        s += '<?xpacket end="w"?>'
        var out = List[UInt8]()
        var sb = s.as_bytes()
        for i in range(len(sb)):
            out.append(sb[i])
        return out^

    def _icc_stream_body(self, icc: List[UInt8]) raises -> List[UInt8]:
        # ICC profile stream: << /N 3 /Alternate /DeviceRGB /Length .. >>
        # with FlateDecode. /N 3 == 3 colour components (RGB).
        var comp = self._zlib_wrap(icc)
        var body = List[UInt8]()
        append_str(body, "<< /N 3 /Alternate /DeviceRGB /Filter /FlateDecode /Length ")
        append_int(body, len(comp))
        append_str(body, " >>\nstream\n")
        for i in range(len(comp)):
            body.append(comp[i])
        append_str(body, "\nendstream")
        return body^

    def _inject_outputintents(
        self, catalog_body: List[UInt8], icc_num: Int
    ) raises -> List[UInt8]:
        # Insert an /OutputIntents array (with one GTS_PDFA1 OutputIntent whose
        # /DestOutputProfile references the embedded ICC stream object icc_num)
        # just before the FINAL ">>" of a Catalog dict body.
        var last = -1
        var i = 0
        while i + 1 < len(catalog_body):
            if (
                catalog_body[i] == UInt8(ord(">"))
                and catalog_body[i + 1] == UInt8(ord(">"))
            ):
                last = i
            i = i + 1
        if last < 0:
            return catalog_body.copy()  # defensive: no closing ">>"
        var out = List[UInt8]()
        for k in range(last):
            out.append(catalog_body[k])
        append_str(
            out,
            " /OutputIntents [ << /Type /OutputIntent /S /GTS_PDFA1"
            " /OutputConditionIdentifier (sRGB) /Info (sRGB)"
            " /DestOutputProfile ",
        )
        append_int(out, icc_num)
        append_str(out, " 0 R >> ] ")
        for k in range(last, len(catalog_body)):
            out.append(catalog_body[k])
        return out^

    def save_pdfa(mut self, path: String) raises:
        # Build a PDF/A-2b conformant document (structural). The caller MUST
        # have added pages whose fonts are all embedded (via embed_font) — this
        # method introduces NO non-embedded font and NO forbidden feature.

        # 1) Ordered document object bodies (user + Catalog + Pages + Info).
        var bodies = List[List[UInt8]]()
        var catalog_num = 0
        var pages_num = 0
        var info_num = 0
        self._build_bodies(bodies, catalog_num, pages_num, info_num)
        var num_docobjs = len(bodies)

        # 2) Deterministic /ID over the base document bodies.
        var id0 = self._doc_id(bodies)
        var id0hex = self._hex_lower(id0)

        # 3) Append the ICC profile stream object (icc_num) and the XMP
        #    /Metadata stream object (meta_num).
        var icc_num = num_docobjs + 1
        var meta_num = num_docobjs + 2

        # 4) Patch the Catalog: add /OutputIntents (-> icc_num) and /Metadata.
        var cat_idx = catalog_num - 1
        bodies[cat_idx] = self._inject_outputintents(bodies[cat_idx], icc_num)
        bodies[cat_idx] = self._inject_metadata_ref(bodies[cat_idx], meta_num)

        # 5) ICC stream body (object icc_num).
        var icc = self._load_srgb_icc()
        bodies.append(self._icc_stream_body(icc))

        # 6) XMP /Metadata body with the pdfaid schema (object meta_num).
        var packet = self._build_xmp_packet_pdfa()
        bodies.append(self._metadata_stream_body(packet))

        var total = len(bodies)

        # 7) Assemble file: classic xref + trailer (mirror save_with_xmp).
        var buf = List[UInt8]()
        var offsets = List[Int]()
        for _i in range(total):
            offsets.append(0)

        append_str(buf, "%PDF-1.7\n")
        buf.append(UInt8(ord("%")))
        buf.append(UInt8(0xE2))
        buf.append(UInt8(0xE3))
        buf.append(UInt8(0xCF))
        buf.append(UInt8(0xD3))
        buf.append(UInt8(ord("\n")))

        for n in range(1, total + 1):
            offsets[n - 1] = len(buf)
            append_int(buf, n)
            append_str(buf, " 0 obj\n")
            var b = bodies[n - 1].copy()
            for j in range(len(b)):
                buf.append(b[j])
            append_str(buf, "\nendobj\n")

        var xref_offset = len(buf)
        append_str(buf, "xref\n0 ")
        append_int(buf, total + 1)
        buf.append(UInt8(ord("\n")))
        append_str(buf, "0000000000 65535 f \n")
        for n in range(1, total + 1):
            self._write_offset10(buf, offsets[n - 1])
            append_str(buf, " 00000 n \n")

        append_str(buf, "trailer\n<< /Size ")
        append_int(buf, total + 1)
        append_str(buf, " /Root ")
        append_int(buf, catalog_num)
        append_str(buf, " 0 R")
        if self.has_info:
            append_str(buf, " /Info ")
            append_int(buf, info_num)
            append_str(buf, " 0 R")
        append_str(buf, " /ID [ <")
        append_str(buf, id0hex)
        append_str(buf, "> <")
        append_str(buf, id0hex)
        append_str(buf, "> ]")
        append_str(buf, " >>\nstartxref\n")
        append_int(buf, xref_offset)
        append_str(buf, "\n%%EOF\n")

        with open(path, "w") as f:
            f.write_bytes(Span(buf))
