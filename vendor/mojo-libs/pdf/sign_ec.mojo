# pdf/sign_ec.mojo
# Pure-Mojo ECDSA (NIST P-256) signing + an ECDSA-signed PDF path.
#
# Two layers:
#   1. Raw ECDSA over a 32-byte SHA-256 hash:
#        ecdsa_sign(d_hex, hash32) -> EcdsaSig {r, s} (each 32-byte big-endian)
#        ecdsa_sig_der(r, s)       -> DER SEQUENCE { INTEGER r, INTEGER s }
#   2. sign_pdf_ecdsa(...) — the SAME incremental-update + detached CMS
#      structure as pdf/sign.mojo, but SignerInfo.signatureAlgorithm is
#      ecdsa-with-SHA256 (1.2.840.10045.4.3.2) and signature OCTET STRING is the
#      ECDSA DER over sha256(DER(SET of signedAttrs)).
#
# Crypto primitives: pdf/sha256.mojo, pdf/ec.mojo, pdf/bigint.mojo, pdf/asn1.mojo.

from std.random import random_ui64

from pdf.sha256 import sha256
from pdf.bigint import BigInt
from pdf.ec import (
    Point,
    scalar_mul,
    generator,
    p256_n,
    field_inverse,
)
from pdf.asn1 import (
    der_sequence,
    der_set,
    der_integer,
    der_octet_string,
    der_oid,
    der_null,
    der_explicit,
    der_length,
)


# ----------------------------------------------------------------------
# ECDSA signature value
# ----------------------------------------------------------------------
struct EcdsaSig(Copyable, Movable):
    var r: List[UInt8]   # 32-byte big-endian
    var s: List[UInt8]   # 32-byte big-endian

    def __init__(out self, var r: List[UInt8], var s: List[UInt8]):
        self.r = r^
        self.s = s^

    def __init__(out self, *, copy: Self):
        self.r = copy.r.copy()
        self.s = copy.s.copy()


def _random_scalar_mod_n(n: BigInt) raises -> BigInt:
    # Build a 256-bit candidate from random 64-bit words, reduce mod n, retry
    # until in [1, n-1].
    while True:
        var bytes = List[UInt8]()
        for _w in range(4):
            var r = random_ui64(0, UInt64(0xFFFFFFFFFFFFFFFF))
            for b in range(8):
                var shift = UInt64(8) * UInt64(7 - b)
                bytes.append(UInt8((r >> shift) & UInt64(0xFF)))
        var cand = BigInt.from_bytes_be(bytes)
        var k = cand.mod(n)
        if not k.is_zero():
            return k^


def ecdsa_sign(d_hex: String, hash32: List[UInt8]) raises -> EcdsaSig:
    # ECDSA sign per FIPS 186-4 over the 32-byte hash (z = the whole hash, since
    # qlen == hlen == 256 bits for P-256 + SHA-256).
    var n = p256_n()
    var d = BigInt.from_hex(d_hex)
    var z = BigInt.from_bytes_be(hash32).mod(n)
    var G = generator()

    while True:
        var k = _random_scalar_mod_n(n)
        var R = scalar_mul(k, G)
        if R.is_inf():
            continue
        var r = R.x.mod(n)
        if r.is_zero():
            continue
        # s = k^-1 * (z + r*d) mod n
        var kinv = field_inverse(k, n)         # n is prime -> Fermat inverse
        var rd = r.mul(d).mod(n)
        var zrd = z.add(rd).mod(n)
        var s = kinv.mul(zrd).mod(n)
        if s.is_zero():
            continue
        var r_bytes = r.to_bytes_be_padded(32)
        var s_bytes = s.to_bytes_be_padded(32)
        return EcdsaSig(r_bytes^, s_bytes^)


def ecdsa_sig_der(r: List[UInt8], s: List[UInt8]) raises -> List[UInt8]:
    # ECDSA-Sig-Value ::= SEQUENCE { r INTEGER, s INTEGER }
    var children = List[List[UInt8]]()
    children.append(der_integer(r))
    children.append(der_integer(s))
    return der_sequence(children)


# ======================================================================
# PDF signing — mirrors pdf/sign.mojo but with ECDSA SignerInfo.
# ======================================================================

# ---- byte helpers (mirrors sign.mojo) ----
def _append_bytes(mut dst: List[UInt8], src: List[UInt8]):
    for i in range(len(src)):
        dst.append(src[i])


def _append_ascii(mut dst: List[UInt8], s: String) raises:
    var src = s.as_bytes()
    for i in range(len(src)):
        dst.append(src[i])


def _append_int(mut dst: List[UInt8], v: Int) raises:
    if v == 0:
        dst.append(UInt8(ord("0")))
        return
    var n = v
    var digits = List[UInt8]()
    while n > 0:
        digits.append(UInt8(ord("0") + (n % 10)))
        n = n // 10
    var k = len(digits)
    while k > 0:
        k -= 1
        dst.append(digits[k])


def _read_file_bytes(path: String) raises -> List[UInt8]:
    var f = open(path, "r")
    var d = f.read_bytes()
    f.close()
    var out = List[UInt8]()
    for i in range(len(d)):
        out.append(d[i])
    return out^


def _find_subseq(hay: List[UInt8], needle: List[UInt8], start: Int) raises -> Int:
    var n = len(hay)
    var m = len(needle)
    if m == 0:
        return start
    var i = start
    while i + m <= n:
        var ok = True
        for k in range(m):
            if hay[i + k] != needle[k]:
                ok = False
                break
        if ok:
            return i
        i += 1
    return -1


def _rfind_subseq(hay: List[UInt8], needle: List[UInt8]) raises -> Int:
    var n = len(hay)
    var m = len(needle)
    if m == 0 or m > n:
        return -1
    var i = n - m
    while i >= 0:
        var ok = True
        for k in range(m):
            if hay[i + k] != needle[k]:
                ok = False
                break
        if ok:
            return i
        i -= 1
    return -1


# ---- DER walking to extract IssuerAndSerialNumber from the cert ----
struct _Tlv(Copyable, Movable):
    var tag: Int
    var hdr_len: Int
    var content_off: Int
    var content_len: Int

    def __init__(out self, tag: Int, hdr_len: Int, content_off: Int, content_len: Int):
        self.tag = tag
        self.hdr_len = hdr_len
        self.content_off = content_off
        self.content_len = content_len

    def total_len(self) -> Int:
        return self.hdr_len + self.content_len


def _read_tlv(data: List[UInt8], off: Int) raises -> _Tlv:
    if off >= len(data):
        raise Error("der: off past end")
    var tag = Int(data[off])
    var p = off + 1
    if p >= len(data):
        raise Error("der: truncated length")
    var l0 = Int(data[p])
    p += 1
    var clen = 0
    if l0 < 0x80:
        clen = l0
    else:
        var nbytes = l0 & 0x7F
        if nbytes == 0 or nbytes > 4:
            raise Error("der: unsupported length form")
        for _i in range(nbytes):
            if p >= len(data):
                raise Error("der: truncated long length")
            clen = (clen << 8) | Int(data[p])
            p += 1
    var hdr = p - off
    return _Tlv(tag, hdr, off + hdr, clen)


def _slice(data: List[UInt8], start: Int, length: Int) raises -> List[UInt8]:
    var out = List[UInt8]()
    for i in range(start, start + length):
        out.append(data[i])
    return out^


def _slice_tlv(data: List[UInt8], t: _Tlv) raises -> List[UInt8]:
    return _slice(data, t.content_off - t.hdr_len, t.total_len())


def _extract_issuer_and_serial(cert: List[UInt8]) raises -> List[UInt8]:
    var cert_seq = _read_tlv(cert, 0)
    if cert_seq.tag != 0x30:
        raise Error("cert: expected outer SEQUENCE")
    var tbs = _read_tlv(cert, cert_seq.content_off)
    if tbs.tag != 0x30:
        raise Error("cert: expected tbsCertificate SEQUENCE")

    var p = tbs.content_off
    var first = _read_tlv(cert, p)
    if first.tag == 0xA0:
        p = first.content_off + first.content_len
        first = _read_tlv(cert, p)
    if first.tag != 0x02:
        raise Error("cert: expected serialNumber INTEGER")
    var serial_tlv = _slice_tlv(cert, first)
    p = first.content_off + first.content_len

    var sigalg = _read_tlv(cert, p)
    if sigalg.tag != 0x30:
        raise Error("cert: expected signature alg SEQUENCE")
    p = sigalg.content_off + sigalg.content_len

    var issuer = _read_tlv(cert, p)
    if issuer.tag != 0x30:
        raise Error("cert: expected issuer Name SEQUENCE")
    var issuer_tlv = _slice_tlv(cert, issuer)

    var children = List[List[UInt8]]()
    children.append(issuer_tlv^)
    children.append(serial_tlv^)
    return der_sequence(children)


# ---- OIDs ----
comptime OID_SIGNED_DATA = "1.2.840.113549.1.7.2"
comptime OID_PKCS7_DATA = "1.2.840.113549.1.7.1"
comptime OID_SHA256 = "2.16.840.1.101.3.4.2.1"
comptime OID_ECDSA_SHA256 = "1.2.840.10045.4.3.2"   # ecdsa-with-SHA256
comptime OID_CT_TYPE = "1.2.840.113549.1.9.3"
comptime OID_MSG_DIGEST = "1.2.840.113549.1.9.4"
comptime OID_SIGNING_TIME = "1.2.840.113549.1.9.5"


def _algid_sha256() raises -> List[UInt8]:
    # AlgorithmIdentifier { sha256, NULL }
    var ch = List[List[UInt8]]()
    ch.append(der_oid(OID_SHA256))
    ch.append(der_null())
    return der_sequence(ch)


def _algid_ecdsa_sha256() raises -> List[UInt8]:
    # AlgorithmIdentifier { ecdsa-with-SHA256 } — no parameters (per RFC 5758).
    var ch = List[List[UInt8]]()
    ch.append(der_oid(OID_ECDSA_SHA256))
    return der_sequence(ch)


def _utctime(yymmddhhmmssZ: String) raises -> List[UInt8]:
    var content = List[UInt8]()
    _append_ascii(content, yymmddhhmmssZ)
    var out = List[UInt8]()
    out.append(UInt8(0x17))
    var lb = der_length(len(content))
    _append_bytes(out, lb)
    _append_bytes(out, content)
    return out^


def _attribute(oid: String, value: List[UInt8]) raises -> List[UInt8]:
    var setch = List[List[UInt8]]()
    setch.append(value.copy())
    var attrvals = der_set(setch)
    var ch = List[List[UInt8]]()
    ch.append(der_oid(oid))
    ch.append(attrvals^)
    return der_sequence(ch)


def _build_signed_attrs_children(
    content_digest: List[UInt8], signing_time: String
) raises -> List[List[UInt8]]:
    var attrs = List[List[UInt8]]()
    var ct_val = der_oid(OID_PKCS7_DATA)
    attrs.append(_attribute(OID_CT_TYPE, ct_val))
    var md_val = der_octet_string(content_digest)
    attrs.append(_attribute(OID_MSG_DIGEST, md_val))
    var st_val = _utctime(signing_time)
    attrs.append(_attribute(OID_SIGNING_TIME, st_val))
    return attrs^


def _build_pkcs7_ecdsa(
    content_digest: List[UInt8],
    cert: List[UInt8],
    d_hex: String,
    signing_time: String,
) raises -> List[UInt8]:
    # Detached CMS SignedData with an ECDSA SignerInfo.
    var attr_children = _build_signed_attrs_children(content_digest, signing_time)

    # signedAttrs: signed bytes are the SET (0x31) re-encoding; the message
    # carries them as [0] IMPLICIT (tag 0xA0) with identical content.
    var signed_attrs_set = der_set(attr_children)
    var attrs_for_msg = List[UInt8]()
    attrs_for_msg.append(UInt8(0xA0))
    var attrs_content = List[UInt8]()
    for i in range(len(attr_children)):
        _append_bytes(attrs_content, attr_children[i])
    var attrs_lb = der_length(len(attrs_content))
    _append_bytes(attrs_for_msg, attrs_lb)
    _append_bytes(attrs_for_msg, attrs_content)

    # signature = ECDSA-SHA256 over sha256( DER(SET of signedAttrs) ).
    var attrs_digest = sha256(signed_attrs_set)
    var sig = ecdsa_sign(d_hex, attrs_digest)
    var sig_der = ecdsa_sig_der(sig.r, sig.s)

    var ias = _extract_issuer_and_serial(cert)

    # SignerInfo ::= SEQUENCE {
    #   version INTEGER 1,
    #   sid IssuerAndSerialNumber,
    #   digestAlgorithm AlgorithmIdentifier(sha256),
    #   signedAttrs [0] IMPLICIT SET OF Attribute,
    #   signatureAlgorithm AlgorithmIdentifier(ecdsa-with-SHA256),
    #   signature OCTET STRING (ECDSA DER)
    # }
    var one_b = List[UInt8](); one_b.append(UInt8(0x01))
    var si_children = List[List[UInt8]]()
    si_children.append(der_integer(one_b))
    si_children.append(ias^)
    si_children.append(_algid_sha256())
    si_children.append(attrs_for_msg^)
    si_children.append(_algid_ecdsa_sha256())
    si_children.append(der_octet_string(sig_der))
    var signer_info = der_sequence(si_children)

    # SignedData ::= SEQUENCE { ... }
    var sd_children = List[List[UInt8]]()
    var v1_b = List[UInt8](); v1_b.append(UInt8(0x01))
    sd_children.append(der_integer(v1_b))

    var dalgs = List[List[UInt8]]()
    dalgs.append(_algid_sha256())
    sd_children.append(der_set(dalgs))

    var encap_children = List[List[UInt8]]()
    encap_children.append(der_oid(OID_PKCS7_DATA))
    sd_children.append(der_sequence(encap_children))

    # certificates [0] IMPLICIT.
    var certs_content = cert.copy()
    var certs_field = List[UInt8]()
    certs_field.append(UInt8(0xA0))
    var certs_lb = der_length(len(certs_content))
    _append_bytes(certs_field, certs_lb)
    _append_bytes(certs_field, certs_content)
    sd_children.append(certs_field^)

    var siset = List[List[UInt8]]()
    siset.append(signer_info^)
    sd_children.append(der_set(siset))

    var signed_data = der_sequence(sd_children)

    # ContentInfo ::= SEQUENCE { contentType signedData, [0] EXPLICIT SignedData }
    var ci_children = List[List[UInt8]]()
    ci_children.append(der_oid(OID_SIGNED_DATA))
    ci_children.append(der_explicit(0, signed_data))
    return der_sequence(ci_children)


def _hex_lower_into(mut dst: List[UInt8], data: List[UInt8]) raises:
    comptime HEX = "0123456789abcdef"
    var hb = HEX.as_bytes()
    for i in range(len(data)):
        var b = Int(data[i])
        dst.append(hb[(b >> 4) & 0xF])
        dst.append(hb[b & 0xF])


# ---- original-PDF inspection (mirrors sign.mojo) ----
def _scan_max_objnum(data: List[UInt8]) raises -> Int:
    var n = len(data)
    var maxn = 0
    var i = 0
    while i < n:
        if data[i] >= UInt8(ord("0")) and data[i] <= UInt8(ord("9")):
            var j = i
            var val = 0
            while j < n and data[j] >= UInt8(ord("0")) and data[j] <= UInt8(ord("9")):
                val = val * 10 + (Int(data[j]) - ord("0"))
                j += 1
            var pat = List[UInt8]()
            _append_ascii(pat, " 0 obj")
            var ok = True
            if j + len(pat) <= n:
                for k in range(len(pat)):
                    if data[j + k] != pat[k]:
                        ok = False
                        break
            else:
                ok = False
            if ok and val > maxn:
                maxn = val
            i = j
        else:
            i += 1
    return maxn


def _parse_root_num(data: List[UInt8]) raises -> Int:
    var pat = List[UInt8]()
    _append_ascii(pat, "/Root")
    var pos = -1
    var search = 0
    while True:
        var f = _find_subseq(data, pat, search)
        if f < 0:
            break
        pos = f
        search = f + 1
    if pos < 0:
        raise Error("pdf: no /Root in trailer")
    var p = pos + len(pat)
    while p < len(data) and (data[p] == UInt8(ord(" ")) or data[p] == UInt8(10) or data[p] == UInt8(13)):
        p += 1
    var val = 0
    var got = False
    while p < len(data) and data[p] >= UInt8(ord("0")) and data[p] <= UInt8(ord("9")):
        val = val * 10 + (Int(data[p]) - ord("0"))
        p += 1
        got = True
    if not got:
        raise Error("pdf: malformed /Root")
    return val


def _parse_last_startxref(data: List[UInt8]) raises -> Int:
    var pat = List[UInt8]()
    _append_ascii(pat, "startxref")
    var pos = _rfind_subseq(data, pat)
    if pos < 0:
        raise Error("pdf: no startxref")
    var p = pos + len(pat)
    while p < len(data) and (data[p] == UInt8(ord(" ")) or data[p] == UInt8(10) or data[p] == UInt8(13)):
        p += 1
    var val = 0
    var got = False
    while p < len(data) and data[p] >= UInt8(ord("0")) and data[p] <= UInt8(ord("9")):
        val = val * 10 + (Int(data[p]) - ord("0"))
        p += 1
        got = True
    if not got:
        raise Error("pdf: malformed startxref")
    return val


def _find_first_page(data: List[UInt8]) raises -> Int:
    var pat = List[UInt8]()
    _append_ascii(pat, "/Type /Page")
    var search = 0
    while True:
        var f = _find_subseq(data, pat, search)
        if f < 0:
            return 0
        var after = f + len(pat)
        if after < len(data) and data[after] == UInt8(ord("s")):
            search = f + 1
            continue
        var objpat = List[UInt8]()
        _append_ascii(objpat, " 0 obj")
        var lo = f - 200
        if lo < 0:
            lo = 0
        var found = -1
        var k = f
        while k >= lo:
            var ok = True
            if k + len(objpat) <= len(data):
                for q in range(len(objpat)):
                    if data[k + q] != objpat[q]:
                        ok = False
                        break
            else:
                ok = False
            if ok:
                found = k
                break
            k -= 1
        if found >= 0:
            var e = found
            var s = e - 1
            while s >= 0 and data[s] >= UInt8(ord("0")) and data[s] <= UInt8(ord("9")):
                s -= 1
            var val = 0
            for q in range(s + 1, e):
                val = val * 10 + (Int(data[q]) - ord("0"))
            return val
        search = f + 1


def _parse_catalog_pages(data: List[UInt8], root_num: Int) raises -> Int:
    var hdr = List[UInt8]()
    _append_int(hdr, root_num)
    _append_ascii(hdr, " 0 obj")
    var start = _find_subseq(data, hdr, 0)
    if start < 0:
        raise Error("pdf: catalog object not found")
    var endpat = List[UInt8]()
    _append_ascii(endpat, "endobj")
    var endp = _find_subseq(data, endpat, start)
    if endp < 0:
        endp = len(data)
    var pat = List[UInt8]()
    _append_ascii(pat, "/Pages")
    var fp = _find_subseq(data, pat, start)
    if fp < 0 or fp > endp:
        raise Error("pdf: /Pages not found in catalog")
    var p = fp + len(pat)
    while p < len(data) and (data[p] == UInt8(ord(" ")) or data[p] == UInt8(10) or data[p] == UInt8(13)):
        p += 1
    var val = 0
    var got = False
    while p < len(data) and data[p] >= UInt8(ord("0")) and data[p] <= UInt8(ord("9")):
        val = val * 10 + (Int(data[p]) - ord("0"))
        p += 1
        got = True
    if not got:
        raise Error("pdf: malformed /Pages ref")
    return val


def _xref_entry(mut buf: List[UInt8], off: Int) raises:
    _patch_or_write10(buf, off)
    _append_ascii(buf, " 00000 n \n")


def _patch_or_write10(mut buf: List[UInt8], value: Int) raises:
    var digits = List[UInt8]()
    var n = value
    if n == 0:
        digits.append(UInt8(ord("0")))
    else:
        while n > 0:
            digits.append(UInt8(ord("0") + (n % 10)))
            n = n // 10
    while len(digits) < 10:
        digits.append(UInt8(ord("0")))
    var k = len(digits)
    while k > 0:
        k -= 1
        buf.append(digits[k])


def _patch_num10(mut buf: List[UInt8], pos: Int, value: Int) raises:
    var digits = List[UInt8]()
    var n = value
    if n == 0:
        digits.append(UInt8(ord("0")))
    else:
        while n > 0:
            digits.append(UInt8(ord("0") + (n % 10)))
            n = n // 10
    while len(digits) < 10:
        digits.append(UInt8(ord("0")))
    var k = len(digits)
    var idx = pos
    while k > 0:
        k -= 1
        buf[idx] = digits[k]
        idx += 1


# ----------------------------------------------------------------------
# Main entry: ECDSA-signed PDF
# ----------------------------------------------------------------------
def sign_pdf_ecdsa(
    in_path: String,
    out_path: String,
    d_hex: String,
    cert_der: List[UInt8],
    reason: String,
) raises:
    var orig = _read_file_bytes(in_path)

    var max_obj = _scan_max_objnum(orig)
    var root_num = _parse_root_num(orig)
    var prev_xref = _parse_last_startxref(orig)
    var first_page = _find_first_page(orig)

    var sig_num = max_obj + 1
    var widget_num = max_obj + 2
    var acro_num = max_obj + 3
    var newcat_num = root_num

    comptime CONTENTS_HEX_LEN = 9000
    comptime SIGNING_TIME = "260609120000Z"
    comptime M_DATE = "D:20260609120000Z"

    var buf = List[UInt8]()
    _append_bytes(buf, orig)
    if buf[len(buf) - 1] != UInt8(10):
        buf.append(UInt8(10))

    # ---- Signature dictionary ----
    var sig_obj_off = len(buf)
    _append_int(buf, sig_num)
    _append_ascii(buf, " 0 obj\n")
    _append_ascii(buf, "<< /Type /Sig /Filter /Adobe.PPKLite /SubFilter /adbe.pkcs7.detached")
    _append_ascii(buf, " /Reason (")
    _append_ascii(buf, reason)
    _append_ascii(buf, ") /M (")
    _append_ascii(buf, M_DATE)
    _append_ascii(buf, ")")
    _append_ascii(buf, " /ByteRange [ ")
    _append_ascii(buf, "0 ")
    var br_pos_a = len(buf)
    _append_ascii(buf, "0000000000 ")
    var br_pos_b = len(buf)
    _append_ascii(buf, "0000000000 ")
    var br_pos_c = len(buf)
    _append_ascii(buf, "0000000000")
    _append_ascii(buf, " ]")
    _append_ascii(buf, " /Contents <")
    var contents_value_start = len(buf)
    for _i in range(CONTENTS_HEX_LEN):
        buf.append(UInt8(ord("0")))
    var contents_value_end = len(buf)
    _append_ascii(buf, "> >>\nendobj\n")

    var lt_off = contents_value_start - 1
    var gt_off = contents_value_end

    # ---- Widget annotation ----
    var widget_obj_off = len(buf)
    _append_int(buf, widget_num)
    _append_ascii(buf, " 0 obj\n")
    _append_ascii(buf, "<< /Type /Annot /Subtype /Widget /FT /Sig /T (Signature1) /V ")
    _append_int(buf, sig_num)
    _append_ascii(buf, " 0 R /Rect [ 0 0 0 0 ]")
    if first_page > 0:
        _append_ascii(buf, " /P ")
        _append_int(buf, first_page)
        _append_ascii(buf, " 0 R")
    _append_ascii(buf, " /F 132 >>\nendobj\n")

    # ---- AcroForm ----
    var acro_obj_off = len(buf)
    _append_int(buf, acro_num)
    _append_ascii(buf, " 0 obj\n")
    _append_ascii(buf, "<< /Fields [ ")
    _append_int(buf, widget_num)
    _append_ascii(buf, " 0 R ] /SigFlags 3 >>\nendobj\n")

    # ---- updated Catalog (overrides root_num) ----
    var cat_obj_off = len(buf)
    _append_int(buf, newcat_num)
    _append_ascii(buf, " 0 obj\n")
    _append_ascii(buf, "<< /Type /Catalog /Pages ")
    var pages_num = _parse_catalog_pages(orig, root_num)
    _append_int(buf, pages_num)
    _append_ascii(buf, " 0 R /AcroForm ")
    _append_int(buf, acro_num)
    _append_ascii(buf, " 0 R >>\nendobj\n")

    # ---- new xref section ----
    var new_xref_off = len(buf)
    _append_ascii(buf, "xref\n")

    _append_int(buf, newcat_num)
    _append_ascii(buf, " 1\n")
    _xref_entry(buf, cat_obj_off)

    _append_int(buf, sig_num)
    _append_ascii(buf, " 3\n")
    _xref_entry(buf, sig_obj_off)
    _xref_entry(buf, widget_obj_off)
    _xref_entry(buf, acro_obj_off)

    # ---- trailer ----
    var size = acro_num + 1
    _append_ascii(buf, "trailer\n<< /Size ")
    _append_int(buf, size)
    _append_ascii(buf, " /Root ")
    _append_int(buf, newcat_num)
    _append_ascii(buf, " 0 R /Prev ")
    _append_int(buf, prev_xref)
    _append_ascii(buf, " >>\nstartxref\n")
    _append_int(buf, new_xref_off)
    _append_ascii(buf, "\n%%EOF\n")

    # ---- finalize ByteRange and patch ----
    var seg1_start = 0
    var seg1_len = lt_off
    var seg2_start = gt_off + 1
    var seg2_len = len(buf) - seg2_start

    _patch_num10(buf, br_pos_a, seg1_len)
    _patch_num10(buf, br_pos_b, seg2_start)
    _patch_num10(buf, br_pos_c, seg2_len)

    var signed_region = List[UInt8]()
    for i in range(seg1_start, seg1_start + seg1_len):
        signed_region.append(buf[i])
    for i in range(seg2_start, seg2_start + seg2_len):
        signed_region.append(buf[i])
    var content_digest = sha256(signed_region)

    var p7 = _build_pkcs7_ecdsa(content_digest, cert_der, d_hex, SIGNING_TIME)

    var p7_hex = List[UInt8]()
    _hex_lower_into(p7_hex, p7)
    if len(p7_hex) > CONTENTS_HEX_LEN:
        raise Error("sign_ec: PKCS7 too large for /Contents placeholder")
    for i in range(len(p7_hex)):
        buf[contents_value_start + i] = p7_hex[i]

    with open(out_path, "w") as f:
        f.write_bytes(Span(buf))
