# pdf/x509.mojo
# Pure-Mojo self-signed X.509 v3 certificate generation.
#
# Builds a minimal but openssl-valid certificate:
#   Certificate ::= SEQUENCE {
#     tbsCertificate       TBSCertificate,
#     signatureAlgorithm   AlgorithmIdentifier (sha256WithRSAEncryption),
#     signatureValue       BIT STRING (RSA signature over DER(tbs))
#   }
# Issuer == Subject (self-signed). The signature is produced with the matching
# private key via pdf.rsa, so the certificate validates as a self-signed root.

from pdf.bigint import BigInt
from pdf.asn1 import (
    der_sequence,
    der_set,
    der_integer,
    der_oid,
    der_null,
    der_explicit,
)
from pdf.rsa import rsa_sign_pkcs1v15_sha256
from pdf.sha256 import sha256
from pdf.keygen import RsaKey


comptime OID_RSA_ENCRYPTION = "1.2.840.113549.1.1.1"
comptime OID_SHA256_WITH_RSA = "1.2.840.113549.1.1.11"
comptime OID_COMMON_NAME = "2.5.4.3"


# ---- DER helpers missing from asn1.mojo (kept local per ownership rules) ----

def der_bitstring(bytes: List[UInt8]) raises -> List[UInt8]:
    # BIT STRING with zero unused bits: tag 0x03, content = 0x00 || bytes.
    var content = List[UInt8]()
    content.append(UInt8(0))  # unused-bits count
    for i in range(len(bytes)):
        content.append(bytes[i])
    var out = List[UInt8]()
    out.append(UInt8(0x03))
    var lenb = _der_length(len(content))
    for i in range(len(lenb)):
        out.append(lenb[i])
    for i in range(len(content)):
        out.append(content[i])
    return out^


def der_utf8_string(s: String) raises -> List[UInt8]:
    # UTF8String, tag 0x0C.
    var content = List[UInt8]()
    var b = s.as_bytes()
    for i in range(len(b)):
        content.append(b[i])
    return _tlv(UInt8(0x0C), content)


def der_utc_time(s: String) raises -> List[UInt8]:
    # UTCTime, tag 0x17, e.g. "260101000000Z".
    var content = List[UInt8]()
    var b = s.as_bytes()
    for i in range(len(b)):
        content.append(b[i])
    return _tlv(UInt8(0x17), content)


def _der_length(n: Int) raises -> List[UInt8]:
    var out = List[UInt8]()
    if n < 128:
        out.append(UInt8(n))
        return out^
    var tmp = List[UInt8]()
    var v = n
    while v > 0:
        tmp.append(UInt8(v & 0xFF))
        v = v >> 8
    out.append(UInt8(0x80 | len(tmp)))
    var i = len(tmp) - 1
    while i >= 0:
        out.append(tmp[i])
        i -= 1
    return out^


def _tlv(tag: UInt8, content: List[UInt8]) raises -> List[UInt8]:
    var out = List[UInt8]()
    out.append(tag)
    var lenb = _der_length(len(content))
    for i in range(len(lenb)):
        out.append(lenb[i])
    for i in range(len(content)):
        out.append(content[i])
    return out^


# ---- structural pieces ----

def _alg_id(oid: String) raises -> List[UInt8]:
    # AlgorithmIdentifier ::= SEQ( OID, NULL )
    var c = List[List[UInt8]]()
    c.append(der_oid(oid))
    c.append(der_null())
    return der_sequence(c)


def _name(common_name: String) raises -> List[UInt8]:
    # Name ::= SEQ( SET( SEQ( OID(commonName), UTF8String(cn) ) ) )
    var atv = List[List[UInt8]]()
    atv.append(der_oid(OID_COMMON_NAME))
    atv.append(der_utf8_string(common_name))
    var atv_seq = der_sequence(atv)

    var set_children = List[List[UInt8]]()
    set_children.append(atv_seq^)
    var rdn = der_set(set_children)

    var name_children = List[List[UInt8]]()
    name_children.append(rdn^)
    return der_sequence(name_children)


def _spki(key: RsaKey) raises -> List[UInt8]:
    # SubjectPublicKeyInfo ::= SEQ( AlgId(rsaEncryption, NULL),
    #                               BITSTRING( SEQ( INTEGER n, INTEGER e ) ) )
    var n = BigInt.from_hex(key.n_hex)
    var e = BigInt.from_hex(key.e_hex)

    var pub_children = List[List[UInt8]]()
    pub_children.append(der_integer(n.to_bytes_be()))
    pub_children.append(der_integer(e.to_bytes_be()))
    var pub_seq = der_sequence(pub_children)

    var spki_children = List[List[UInt8]]()
    spki_children.append(_alg_id(OID_RSA_ENCRYPTION))
    spki_children.append(der_bitstring(pub_seq))
    return der_sequence(spki_children)


def _tbs(key: RsaKey, common_name: String) raises -> List[UInt8]:
    # TBSCertificate (v3)
    var children = List[List[UInt8]]()

    # version [0] EXPLICIT INTEGER 2  (v3)
    var ver_int_bytes = List[UInt8]()
    ver_int_bytes.append(UInt8(2))
    children.append(der_explicit(0, der_integer(ver_int_bytes)))

    # serialNumber INTEGER (small positive)
    var serial_bytes = List[UInt8]()
    serial_bytes.append(UInt8(0x01))
    children.append(der_integer(serial_bytes))

    # signature AlgorithmIdentifier (sha256WithRSAEncryption)
    children.append(_alg_id(OID_SHA256_WITH_RSA))

    # issuer Name
    children.append(_name(common_name))

    # validity SEQ( UTCTime notBefore, UTCTime notAfter )
    var val_children = List[List[UInt8]]()
    val_children.append(der_utc_time("260101000000Z"))
    val_children.append(der_utc_time("360101000000Z"))
    children.append(der_sequence(val_children))

    # subject Name (== issuer)
    children.append(_name(common_name))

    # subjectPublicKeyInfo
    children.append(_spki(key))

    return der_sequence(children)


def make_self_signed_cert(key: RsaKey, common_name: String) raises -> List[UInt8]:
    var tbs = _tbs(key, common_name)
    var digest = sha256(tbs)
    var sig = rsa_sign_pkcs1v15_sha256(key.n_hex, key.d_hex, digest)

    var cert_children = List[List[UInt8]]()
    cert_children.append(tbs^)
    cert_children.append(_alg_id(OID_SHA256_WITH_RSA))
    cert_children.append(der_bitstring(sig))
    return der_sequence(cert_children)
