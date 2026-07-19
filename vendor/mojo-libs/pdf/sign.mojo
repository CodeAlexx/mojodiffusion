# pdf/sign.mojo
# Pure-Mojo PDF digital-signature post-processor.
#
# Produces an Adobe-compatible detached PKCS#7 (CMS) signature embedded in a
# PDF via an INCREMENTAL UPDATE (append-only): the original file bytes are left
# untouched and a new revision (signature dict + signature widget + AcroForm +
# updated catalog + new xref/trailer with /Prev) is appended.
#
# SubFilter /adbe.pkcs7.detached. The /Contents value is a fixed-size hex
# placeholder; ByteRange spans the whole file except that placeholder. After the
# byte layout is final we hash the two ByteRange segments, build the CMS
# SignedData (signedAttrs: contentType + messageDigest + signingTime, signed
# with RSA-PKCS1v1.5-SHA256), DER-encode it, hex it, and overwrite the
# placeholder zeros IN PLACE (same length) so no offsets shift.
#
# Crypto primitives: pdf/sha256.mojo, pdf/rsa.mojo, pdf/asn1.mojo (all verified).

from pdf.sha256 import sha256
from pdf.rsa import rsa_sign_pkcs1v15_sha256
from pdf.asn1 import (
    der_sequence,
    der_set,
    der_integer,
    der_octet_string,
    der_oid,
    der_null,
    der_explicit,
    der_context,
    der_length,
)


# ----------------------------------------------------------------------
# Small byte helpers
# ----------------------------------------------------------------------
def _append_bytes(mut dst: List[UInt8], src: List[UInt8]):
    for i in range(len(src)):
        dst.append(src[i])


def _append_ascii(mut dst: List[UInt8], s: String) raises:
    var src = s.as_bytes()
    for i in range(len(src)):
        dst.append(src[i])


def _append_int(mut dst: List[UInt8], v: Int) raises:
    # decimal int (non-negative expected here)
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
    # First index >= start where needle occurs in hay, else -1.
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
    # Last index where needle occurs in hay, else -1.
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


# ----------------------------------------------------------------------
# DER walking (to extract serial + issuer from a certificate)
# ----------------------------------------------------------------------
struct _Tlv(Copyable, Movable):
    var tag: Int
    var hdr_len: Int      # bytes of tag+length
    var content_off: Int  # absolute offset of content start
    var content_len: Int

    def __init__(out self, tag: Int, hdr_len: Int, content_off: Int, content_len: Int):
        self.tag = tag
        self.hdr_len = hdr_len
        self.content_off = content_off
        self.content_len = content_len

    def total_len(self) -> Int:
        return self.hdr_len + self.content_len


def _read_tlv(data: List[UInt8], off: Int) raises -> _Tlv:
    # Parse one DER TLV header at off. Tag assumed single-byte (true for the
    # tags present in an X.509 cert's top structure).
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
    # The full TLV bytes (tag+len+content).
    return _slice(data, t.content_off - t.hdr_len, t.total_len())


def _extract_issuer_and_serial(cert: List[UInt8]) raises -> List[UInt8]:
    # Parse an X.509 Certificate to build the CMS IssuerAndSerialNumber:
    #   SEQUENCE { issuer Name (DER as-is), serialNumber INTEGER (DER as-is) }
    #
    # Certificate ::= SEQ { tbsCertificate SEQ { ... }, sigAlg, sigValue }
    # tbsCertificate ::= SEQ {
    #     [0] version (optional, EXPLICIT, tag 0xA0),
    #     serialNumber INTEGER,
    #     signature AlgorithmIdentifier SEQ,
    #     issuer Name SEQ,
    #     ... }
    var cert_seq = _read_tlv(cert, 0)            # outer Certificate SEQUENCE
    if cert_seq.tag != 0x30:
        raise Error("cert: expected outer SEQUENCE")
    var tbs = _read_tlv(cert, cert_seq.content_off)  # tbsCertificate SEQUENCE
    if tbs.tag != 0x30:
        raise Error("cert: expected tbsCertificate SEQUENCE")

    var p = tbs.content_off
    var first = _read_tlv(cert, p)
    # Optional [0] EXPLICIT version (context constructed tag 0xA0).
    if first.tag == 0xA0:
        p = first.content_off + first.content_len  # skip version
        first = _read_tlv(cert, p)
    # Now `first` is the serialNumber INTEGER.
    if first.tag != 0x02:
        raise Error("cert: expected serialNumber INTEGER")
    var serial_tlv = _slice_tlv(cert, first)
    p = first.content_off + first.content_len

    # signature AlgorithmIdentifier SEQUENCE — skip.
    var sigalg = _read_tlv(cert, p)
    if sigalg.tag != 0x30:
        raise Error("cert: expected signature alg SEQUENCE")
    p = sigalg.content_off + sigalg.content_len

    # issuer Name SEQUENCE.
    var issuer = _read_tlv(cert, p)
    if issuer.tag != 0x30:
        raise Error("cert: expected issuer Name SEQUENCE")
    var issuer_tlv = _slice_tlv(cert, issuer)

    # IssuerAndSerialNumber ::= SEQUENCE { issuer, serial }
    var children = List[List[UInt8]]()
    children.append(issuer_tlv^)
    children.append(serial_tlv^)
    return der_sequence(children)


# ----------------------------------------------------------------------
# OIDs
# ----------------------------------------------------------------------
comptime OID_SIGNED_DATA = "1.2.840.113549.1.7.2"
comptime OID_PKCS7_DATA = "1.2.840.113549.1.7.1"
comptime OID_SHA256 = "2.16.840.1.101.3.4.2.1"
comptime OID_RSA_ENC = "1.2.840.113549.1.1.1"
comptime OID_CT_TYPE = "1.2.840.113549.1.9.3"   # contentType
comptime OID_MSG_DIGEST = "1.2.840.113549.1.9.4"  # messageDigest
comptime OID_SIGNING_TIME = "1.2.840.113549.1.9.5"  # signingTime


def _algid_sha256() raises -> List[UInt8]:
    # AlgorithmIdentifier { sha256, NULL }
    var ch = List[List[UInt8]]()
    ch.append(der_oid(OID_SHA256))
    ch.append(der_null())
    return der_sequence(ch)


def _algid_rsa() raises -> List[UInt8]:
    # AlgorithmIdentifier { rsaEncryption, NULL }
    var ch = List[List[UInt8]]()
    ch.append(der_oid(OID_RSA_ENC))
    ch.append(der_null())
    return der_sequence(ch)


def _utctime(yymmddhhmmssZ: String) raises -> List[UInt8]:
    # UTCTime, tag 0x17. Content is the ASCII e.g. "260609120000Z".
    var content = List[UInt8]()
    _append_ascii(content, yymmddhhmmssZ)
    var out = List[UInt8]()
    out.append(UInt8(0x17))
    var lb = der_length(len(content))
    _append_bytes(out, lb)
    _append_bytes(out, content)
    return out^


def _attribute(oid: String, value: List[UInt8]) raises -> List[UInt8]:
    # Attribute ::= SEQUENCE { attrType OID, attrValues SET { value } }
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
    # The set of authenticated attributes, as a list of Attribute DERs.
    # DER requires SET OF elements to be sorted, but CMS verifiers accept the
    # canonical content; we order by attrType OID byte value to be safe:
    #   contentType (..9.3), messageDigest (..9.4), signingTime (..9.5).
    var attrs = List[List[UInt8]]()
    # contentType = pkcs7-data
    var ct_val = der_oid(OID_PKCS7_DATA)
    attrs.append(_attribute(OID_CT_TYPE, ct_val))
    # messageDigest = OCTET STRING(sha256(ByteRange))
    var md_val = der_octet_string(content_digest)
    attrs.append(_attribute(OID_MSG_DIGEST, md_val))
    # signingTime
    var st_val = _utctime(signing_time)
    attrs.append(_attribute(OID_SIGNING_TIME, st_val))
    return attrs^


def _build_pkcs7(
    content_digest: List[UInt8],
    cert: List[UInt8],
    n_hex: String,
    e_hex: String,
    d_hex: String,
    signing_time: String,
) raises -> List[UInt8]:
    # Build a complete detached CMS SignedData ContentInfo (DER).
    var attr_children = _build_signed_attrs_children(content_digest, signing_time)

    # signedAttrs appear in the message as [0] IMPLICIT (tag 0xA0). But the
    # bytes that are SIGNED are the same attributes re-tagged as a SET (0x31).
    var signed_attrs_set = der_set(attr_children)         # tag 0x31 (to be signed)
    # rebuild [0] IMPLICIT version (same content, tag 0xA0) for the message
    var attrs_for_msg = List[UInt8]()
    attrs_for_msg.append(UInt8(0xA0))
    # content of signed_attrs_set is everything after its TL header; easiest:
    # re-encode the children directly under 0xA0.
    var attrs_content = List[UInt8]()
    for i in range(len(attr_children)):
        _append_bytes(attrs_content, attr_children[i])
    var attrs_lb = der_length(len(attrs_content))
    _append_bytes(attrs_for_msg, attrs_lb)
    _append_bytes(attrs_for_msg, attrs_content)

    # signature = RSA-PKCS1v1.5-SHA256 over sha256( DER(SET of signedAttrs) )
    var attrs_digest = sha256(signed_attrs_set)
    var sig = rsa_sign_pkcs1v15_sha256(n_hex, d_hex, attrs_digest)

    # IssuerAndSerialNumber
    var ias = _extract_issuer_and_serial(cert)

    # SignerInfo ::= SEQUENCE {
    #   version INTEGER 1,
    #   sid IssuerAndSerialNumber,
    #   digestAlgorithm AlgorithmIdentifier(sha256),
    #   signedAttrs [0] IMPLICIT SET OF Attribute,
    #   signatureAlgorithm AlgorithmIdentifier(rsaEncryption),
    #   signature OCTET STRING
    # }
    var one_b = List[UInt8](); one_b.append(UInt8(0x01))
    var si_children = List[List[UInt8]]()
    si_children.append(der_integer(one_b))
    si_children.append(ias^)
    si_children.append(_algid_sha256())
    si_children.append(attrs_for_msg^)
    si_children.append(_algid_rsa())
    si_children.append(der_octet_string(sig))
    var signer_info = der_sequence(si_children)

    # SignedData ::= SEQUENCE {
    #   version INTEGER 1,
    #   digestAlgorithms SET OF AlgorithmIdentifier,
    #   encapContentInfo SEQUENCE { eContentType OID pkcs7-data },  (detached)
    #   certificates [0] IMPLICIT SET OF Certificate,
    #   signerInfos SET OF SignerInfo
    # }
    var sd_children = List[List[UInt8]]()
    var v1_b = List[UInt8](); v1_b.append(UInt8(0x01))
    sd_children.append(der_integer(v1_b))

    var dalgs = List[List[UInt8]]()
    dalgs.append(_algid_sha256())
    sd_children.append(der_set(dalgs))

    var encap_children = List[List[UInt8]]()
    encap_children.append(der_oid(OID_PKCS7_DATA))
    sd_children.append(der_sequence(encap_children))

    # certificates [0] IMPLICIT: content is the raw certificate DER(s).
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

    # ContentInfo ::= SEQUENCE { contentType OID signedData, [0] EXPLICIT SignedData }
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


# ----------------------------------------------------------------------
# Original-PDF inspection: object count, /Root number, startxref offset
# ----------------------------------------------------------------------
def _scan_max_objnum(data: List[UInt8]) raises -> Int:
    # Scan for "<N> 0 obj" tokens, return max N.
    var n = len(data)
    var maxn = 0
    var i = 0
    while i < n:
        # digit run
        if data[i] >= UInt8(ord("0")) and data[i] <= UInt8(ord("9")):
            var j = i
            var val = 0
            while j < n and data[j] >= UInt8(ord("0")) and data[j] <= UInt8(ord("9")):
                val = val * 10 + (Int(data[j]) - ord("0"))
                j += 1
            # expect " 0 obj"
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
    # Find the LAST "/Root <N> 0 R" and return N.
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
    # skip "/Root", whitespace, read integer
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
    # Find an object that is a Page: look for "/Type /Page" (not /Pages) and
    # walk back to its object number. Best-effort; used only for /P widget ref.
    var pat = List[UInt8]()
    _append_ascii(pat, "/Type /Page")
    var search = 0
    while True:
        var f = _find_subseq(data, pat, search)
        if f < 0:
            return 0
        # reject "/Type /Pages"
        var after = f + len(pat)
        if after < len(data) and data[after] == UInt8(ord("s")):
            search = f + 1
            continue
        # walk backwards to find "<N> 0 obj"
        var objpat = List[UInt8]()
        _append_ascii(objpat, " 0 obj")
        var b = f
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
            # read the number ending at found (digits immediately before space)
            var e = found
            var s = e - 1
            while s >= 0 and data[s] >= UInt8(ord("0")) and data[s] <= UInt8(ord("9")):
                s -= 1
            var val = 0
            for q in range(s + 1, e):
                val = val * 10 + (Int(data[q]) - ord("0"))
            return val
        search = f + 1


# ----------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------
def sign_pdf(
    in_path: String,
    out_path: String,
    n_hex: String,
    e_hex: String,
    d_hex: String,
    cert_der: List[UInt8],
    reason: String,
) raises:
    var orig = _read_file_bytes(in_path)
    var orig_len = len(orig)

    var max_obj = _scan_max_objnum(orig)
    var root_num = _parse_root_num(orig)
    var prev_xref = _parse_last_startxref(orig)
    var first_page = _find_first_page(orig)

    # New object numbers.
    var sig_num = max_obj + 1
    var widget_num = max_obj + 2
    var acro_num = max_obj + 3
    var newcat_num = root_num  # we OVERRIDE the catalog (same obj num, gen 0)

    # Fixed-size /Contents hex placeholder. 9000 hex digits => 4500 DER bytes.
    comptime CONTENTS_HEX_LEN = 9000

    # Signing time (fixed for determinism).
    comptime SIGNING_TIME = "260609120000Z"
    comptime M_DATE = "D:20260609120000Z"

    # Assemble the appended revision into `buf`, on top of the original bytes.
    var buf = List[UInt8]()
    _append_bytes(buf, orig)
    # Ensure a newline boundary before the appended revision.
    if buf[len(buf) - 1] != UInt8(10):
        buf.append(UInt8(10))

    # ---- Object: Signature dictionary ----
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
    # ByteRange placeholder: fixed-width fields we will overwrite in place.
    # Format: "0 AAAAAAAAAA BBBBBBBBBB CCCCCCCCCC" -> use 10-digit zero-padded.
    # We record the position of each field to patch later.
    var br_pos0 = len(buf)
    _append_ascii(buf, "0 ")
    var br_pos_a = len(buf)
    _append_ascii(buf, "0000000000 ")
    var br_pos_b = len(buf)
    _append_ascii(buf, "0000000000 ")
    var br_pos_c = len(buf)
    _append_ascii(buf, "0000000000")
    _append_ascii(buf, " ]")
    _append_ascii(buf, " /Contents <")
    var contents_value_start = len(buf)  # offset of '<'? No: this is AFTER '<'
    # contents_value_start currently points to first hex digit (we just wrote '<').
    # Fill placeholder zeros.
    for _i in range(CONTENTS_HEX_LEN):
        buf.append(UInt8(ord("0")))
    var contents_value_end = len(buf)  # offset just past last hex digit (the '>')
    _append_ascii(buf, "> >>\nendobj\n")

    # The '<' is at contents_value_start - 1; the '>' is at contents_value_end.
    var lt_off = contents_value_start - 1
    var gt_off = contents_value_end

    # ---- Object: Signature field / Widget annotation ----
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

    # ---- Object: AcroForm ----
    var acro_obj_off = len(buf)
    _append_int(buf, acro_num)
    _append_ascii(buf, " 0 obj\n")
    _append_ascii(buf, "<< /Fields [ ")
    _append_int(buf, widget_num)
    _append_ascii(buf, " 0 R ] /SigFlags 3 >>\nendobj\n")

    # ---- Object: updated Catalog (overrides root_num) ----
    var cat_obj_off = len(buf)
    _append_int(buf, newcat_num)
    _append_ascii(buf, " 0 obj\n")
    _append_ascii(buf, "<< /Type /Catalog /Pages ")
    # We must preserve the original /Pages reference. Find it in the original
    # catalog object. Simplest robust approach: reuse original Pages number by
    # scanning the original catalog. Fall back: leave as-is is unsafe, so parse.
    var pages_num = _parse_catalog_pages(orig, root_num)
    _append_int(buf, pages_num)
    _append_ascii(buf, " 0 R /AcroForm ")
    _append_int(buf, acro_num)
    _append_ascii(buf, " 0 R >>\nendobj\n")

    # ---- New classic xref section for the new/changed objects ----
    # Changed/new objects: sig_num, widget_num, acro_num, newcat_num(=root_num).
    # They are not contiguous (root_num is small), so emit multiple subsections.
    var new_xref_off = len(buf)
    _append_ascii(buf, "xref\n")

    # Subsection 1: the catalog override (object root_num, count 1).
    _append_int(buf, newcat_num)
    _append_ascii(buf, " 1\n")
    _xref_entry(buf, cat_obj_off)

    # Subsection 2: sig_num..acro_num (3 contiguous new objects).
    _append_int(buf, sig_num)
    _append_ascii(buf, " 3\n")
    _xref_entry(buf, sig_obj_off)
    _xref_entry(buf, widget_obj_off)
    _xref_entry(buf, acro_obj_off)

    # ---- Trailer with /Prev ----
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

    # ----------------------------------------------------------------
    # Layout is now FINAL. Compute ByteRange and patch in place.
    # ByteRange covers everything EXCEPT the /Contents value INCLUDING the
    # angle brackets:
    #   segment1 = [0 .. lt_off)        (up to and including byte before '<')
    #   segment2 = [gt_off+1 .. EOF)    (from byte after '>' to EOF)
    # Convention used widely (and by poppler): the gap is the bytes between the
    # '<' and '>' INCLUSIVE, i.e. ByteRange = [0, lt_off, gt_off+1, len-(gt_off+1)].
    # ----------------------------------------------------------------
    var seg1_start = 0
    var seg1_len = lt_off                 # bytes [0, lt_off)
    var seg2_start = gt_off + 1           # byte after '>'
    var seg2_len = len(buf) - seg2_start

    # Patch the ByteRange numeric fields (10-digit zero padded) in place.
    _patch_num10(buf, br_pos_a, seg1_len)
    _patch_num10(buf, br_pos_b, seg2_start)
    _patch_num10(buf, br_pos_c, seg2_len)
    # br_pos0 already holds "0 " (seg1_start == 0) — leave it.

    # Build the ByteRange byte stream (seg1 ++ seg2) and hash it.
    var signed_region = List[UInt8]()
    for i in range(seg1_start, seg1_start + seg1_len):
        signed_region.append(buf[i])
    for i in range(seg2_start, seg2_start + seg2_len):
        signed_region.append(buf[i])
    var content_digest = sha256(signed_region)

    # Build the PKCS#7 over that digest.
    var p7 = _build_pkcs7(content_digest, cert_der, n_hex, e_hex, d_hex, SIGNING_TIME)

    # Hex-encode the DER; must fit in CONTENTS_HEX_LEN, pad with trailing zeros.
    var p7_hex = List[UInt8]()
    _hex_lower_into(p7_hex, p7)
    if len(p7_hex) > CONTENTS_HEX_LEN:
        raise Error("sign: PKCS7 too large for /Contents placeholder")
    # Overwrite placeholder zeros in place.
    for i in range(len(p7_hex)):
        buf[contents_value_start + i] = p7_hex[i]
    # remaining bytes stay '0' (already zeros) — they are valid hex padding.

    # Write the signed file.
    with open(out_path, "w") as f:
        f.write_bytes(Span(buf))


def _xref_entry(mut buf: List[UInt8], off: Int) raises:
    # "%010d 00000 n \n"
    _patch_or_write10(buf, off)
    _append_ascii(buf, " 00000 n \n")


def _patch_or_write10(mut buf: List[UInt8], value: Int) raises:
    # Append a 10-digit zero-padded decimal.
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
    # Overwrite 10 bytes at pos with the zero-padded decimal of value.
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
    # digits is least-significant first; write most-significant first.
    var k = len(digits)
    var idx = pos
    while k > 0:
        k -= 1
        buf[idx] = digits[k]
        idx += 1


def _parse_catalog_pages(data: List[UInt8], root_num: Int) raises -> Int:
    # Locate object "<root_num> 0 obj", then find "/Pages <N> 0 R" within it.
    var hdr = List[UInt8]()
    _append_int(hdr, root_num)
    _append_ascii(hdr, " 0 obj")
    var start = _find_subseq(data, hdr, 0)
    if start < 0:
        raise Error("pdf: catalog object not found")
    # bound the search to this object (until "endobj")
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
