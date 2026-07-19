# pdf/rsa.mojo
# RSA PKCS#1 v1.5 signing over a precomputed SHA-256 digest, plus a raw public
# operation for self-verification. Built on pdf.bigint (modexp) and pdf.asn1.
#
# EMSA-PKCS1-v1_5 encoding (RFC 8017 / PKCS#1):
#   DigestInfo = SEQ( SEQ( OID(sha256), NULL ), OCTETSTRING(digest) )
#   EM = 0x00 || 0x01 || PS(0xFF...) || 0x00 || DigestInfo   (length k = modlen)
#   signature = (EM as integer)^d mod n , emitted as k big-endian bytes.

from pdf.bigint import BigInt
from pdf.asn1 import der_sequence, der_octet_string, der_oid, der_null


comptime SHA256_OID = "2.16.840.1.101.3.4.2.1"


def _digest_info_sha256(digest: List[UInt8]) raises -> List[UInt8]:
    # DigestInfo for SHA-256.
    var alg_children = List[List[UInt8]]()
    alg_children.append(der_oid(SHA256_OID))
    alg_children.append(der_null())
    var alg_id = der_sequence(alg_children)

    var di_children = List[List[UInt8]]()
    di_children.append(alg_id^)
    di_children.append(der_octet_string(digest))
    return der_sequence(di_children)


def emsa_pkcs1v15_sha256(digest: List[UInt8], k: Int) raises -> List[UInt8]:
    # Build the k-byte encoded message EM.
    var di = _digest_info_sha256(digest)
    var tlen = len(di)
    # Need at least: 0x00 0x01 + (8 bytes min PS) + 0x00 + DigestInfo
    if k < tlen + 11:
        raise Error("emsa_pkcs1v15: modulus too short for digest")
    var em = List[UInt8]()
    em.append(UInt8(0x00))
    em.append(UInt8(0x01))
    var ps_len = k - tlen - 3
    for _ in range(ps_len):
        em.append(UInt8(0xFF))
    em.append(UInt8(0x00))
    for i in range(len(di)):
        em.append(di[i])
    return em^


def rsa_sign_pkcs1v15_sha256(
    n_hex: String, d_hex: String, digest: List[UInt8]
) raises -> List[UInt8]:
    var n = BigInt.from_hex(n_hex)
    var d = BigInt.from_hex(d_hex)
    # modulus byte length k
    var nbytes = n.to_bytes_be()
    var k = len(nbytes)
    var em = emsa_pkcs1v15_sha256(digest, k)
    var m = BigInt.from_bytes_be(em)
    var sig = m.modexp(d, n)
    return sig.to_bytes_be_padded(k)


def rsa_public_raw(n_hex: String, e_hex: String, sig: List[UInt8]) raises -> List[UInt8]:
    # Recover EM = sig^e mod n, padded to modulus length, for self-check.
    var n = BigInt.from_hex(n_hex)
    var e = BigInt.from_hex(e_hex)
    var k = len(n.to_bytes_be())
    var s = BigInt.from_bytes_be(sig)
    var em = s.modexp(e, n)
    return em.to_bytes_be_padded(k)
