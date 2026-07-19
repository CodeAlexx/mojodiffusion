# pdf/tests/rsa_test.mojo
# Tests for the pure-Mojo bigint + RSA (PKCS#1 v1.5) + ASN.1/DER toolkit.
#
# Run from /home/alex/MOJO-libs:
#   pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . pdf/tests/rsa_test.mojo
#
# This:
#   1. Self-checks BigInt (modexp small known cases + big mul/mod cross-checked).
#   2. Self-checks ASN.1 DER (sha256 OID exact bytes + a SEQUENCE/INTEGER).
#   3. Signs the SHA-256 digest of /tmp/msg.txt with an openssl-generated key
#      whose n/e/d are embedded below, writes /tmp/mojo_sig.bin, and self-checks
#      via rsa_public_raw. The decisive openssl `dgst -verify` is run by the
#      companion shell driver (pdf/tests/rsa_verify.sh).

from pdf.bigint import BigInt
from pdf.asn1 import (
    der_oid,
    der_sequence,
    der_integer,
    der_octet_string,
    der_null,
    der_length,
)
from pdf.rsa import rsa_sign_pkcs1v15_sha256, rsa_public_raw

# --- openssl-generated RSA-2048 key (n,e,d), from /tmp/rsa_test.pem ---
comptime N_HEX = "cc25befba7cbf165995906e20c39c5d9075e758e05ada352cd9e88b550207d8bba395b3bdb684b3c74f201aed2b867e24fcc8aeb18f7433a44ae7a0da9ecf161bcad1d168160cd825f1c9f1da063448079172725ebdbc5512f531955aa43f4f4595891ddc200c1e2b72c03a7ba6eb486a75586dc8b15a74e9cfc1cd54ff8ad4ff9c36df9ac4e35eee40dba0e78f9b0da23b4b571852bc7754c9982a2cbcad3498cce2ef2bd04aca6545065edf599476b0ac1c3159d8ada4f9ede2b7c8986bba4da33ef0f8781c4d39a353ce67239a71391d15c8f096c9dc34e9d0804ec3f590797c1537473cbc83e20b0d092321e0aa8e0f1919f294f0510dff75db6ef7ccd0b"
comptime E_HEX = "10001"
comptime D_HEX = "2049da0e988807aaaf99e69e4b1bba20aceb32419fc14a6336d55bbefda8dde283363e2955f7056b4efdd5e94e37cf6a7a7f99fb3c2c238c6c3f825b75e45d7b3d69cdff78c0145109f50f6f92a610b8172ee3c8ba28bce92dc88169ccafc9e6f9d8a9dd7ea93b013e426e639177a002ea257b5a977ef9c2d3ce864af0c69eb68eb40ea22a786bc9fd8ebe6d1702ba41699072e9567e94caa3c6a76a4b109f224ef3170877b47c026201482fe59762231e4f727a8b5542b7a4b7f773db0a4226a2f837eb511d605f7650e81a35d307696b6c95a8b8d07a1085f6013457563ba67e3410a1f85f23005fa4f431c0f29d375f261d24bb0bfc1cb3d149d59ff24f2d"

# SHA-256("The quick brown fox jumps over the lazy dog")
comptime MSG_DIGEST_HEX = "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592"


def to_hex(bs: List[UInt8]) raises -> String:
    comptime HEX = "0123456789abcdef"
    var hexb = HEX.as_bytes()
    var s = String("")
    for i in range(len(bs)):
        var b = Int(bs[i])
        s += chr(Int(hexb[(b >> 4) & 0xF]))
        s += chr(Int(hexb[b & 0xF]))
    return s^


def bytes_eq(a: List[UInt8], b: List[UInt8]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def hex_to_bytes(s: String) raises -> List[UInt8]:
    var b = BigInt.from_hex(s)
    return b.to_bytes_be()


def write_file(path: String, data: List[UInt8]) raises:
    # Write RAW bytes (Span), NOT a String — chr() would UTF-8-expand bytes >=0x80.
    with open(path, "w") as f:
        f.write_bytes(Span(data))


def main() raises:
    var passed = 0
    var failed = 0

    print("=== BigInt checks ===")

    # modexp: 4^13 mod 497 = 445
    var b1 = BigInt(UInt64(4))
    var e1 = BigInt(UInt64(13))
    var m1 = BigInt(UInt64(497))
    var r1 = b1.modexp(e1, m1)
    var exp1 = BigInt(UInt64(445))
    if r1.compare(exp1) == 0:
        print("PASS  4^13 mod 497 == 445  (got ", r1.to_hex(), ")")
        passed += 1
    else:
        print("FAIL  4^13 mod 497  got 0x", r1.to_hex(), " want 0x1bd")
        failed += 1

    # modexp: 2^10 mod 1000 = 24
    var r2 = BigInt(UInt64(2)).modexp(BigInt(UInt64(10)), BigInt(UInt64(1000)))
    if r2.compare(BigInt(UInt64(24))) == 0:
        print("PASS  2^10 mod 1000 == 24")
        passed += 1
    else:
        print("FAIL  2^10 mod 1000  got 0x", r2.to_hex())
        failed += 1

    # modexp: 3^644 mod 645 = 36 (classic)
    var r3 = BigInt(UInt64(3)).modexp(BigInt(UInt64(644)), BigInt(UInt64(645)))
    if r3.compare(BigInt(UInt64(36))) == 0:
        print("PASS  3^644 mod 645 == 36")
        passed += 1
    else:
        print("FAIL  3^644 mod 645  got 0x", r3.to_hex())
        failed += 1

    # big mul cross-check: (2^64 - 1) * (2^64 - 1)
    #   = 0xfffffffffffffffe0000000000000001
    var big = BigInt.from_hex("ffffffffffffffff")
    var prod = big.mul(big)
    if prod.to_hex() == "fffffffffffffffe0000000000000001":
        print("PASS  (2^64-1)^2 == 0x", prod.to_hex())
        passed += 1
    else:
        print("FAIL  (2^64-1)^2  got 0x", prod.to_hex())
        failed += 1

    # big mod cross-check (Python: pow(7,1000,13)=9)
    var r4 = BigInt(UInt64(7)).modexp(BigInt(UInt64(1000)), BigInt(UInt64(13)))
    if r4.compare(BigInt(UInt64(9))) == 0:
        print("PASS  7^1000 mod 13 == 9")
        passed += 1
    else:
        print("FAIL  7^1000 mod 13  got 0x", r4.to_hex())
        failed += 1

    # hex round-trip
    var rt = BigInt.from_hex(N_HEX)
    if rt.to_hex() == N_HEX:
        print("PASS  N hex round-trip (2048-bit)")
        passed += 1
    else:
        print("FAIL  N hex round-trip")
        failed += 1

    # decimal parse: "1000000000000000000000" -> hex 3635c9adc5dea00000
    var dec = BigInt.from_decimal("1000000000000000000000")
    if dec.to_hex() == "3635c9adc5dea00000":
        print("PASS  decimal 1e21 -> hex 0x", dec.to_hex())
        passed += 1
    else:
        print("FAIL  decimal 1e21  got 0x", dec.to_hex())
        failed += 1

    print("")
    print("=== ASN.1 DER checks ===")

    # sha256 OID: 06 09 60 86 48 01 65 03 04 02 01
    var oid = der_oid("2.16.840.1.101.3.4.2.1")
    var expect_oid = List[UInt8]()
    expect_oid.append(UInt8(0x06)); expect_oid.append(UInt8(0x09))
    expect_oid.append(UInt8(0x60)); expect_oid.append(UInt8(0x86))
    expect_oid.append(UInt8(0x48)); expect_oid.append(UInt8(0x01))
    expect_oid.append(UInt8(0x65)); expect_oid.append(UInt8(0x03))
    expect_oid.append(UInt8(0x04)); expect_oid.append(UInt8(0x02))
    expect_oid.append(UInt8(0x01))
    if bytes_eq(oid, expect_oid):
        print("PASS  sha256 OID == 06 09 60 86 48 01 65 03 04 02 01")
        passed += 1
    else:
        print("FAIL  sha256 OID  got ", to_hex(oid))
        failed += 1

    # der_length long form: 200 -> 81 c8 ; 300 -> 82 01 2c
    var l200 = der_length(200)
    var l300 = der_length(300)
    if to_hex(l200) == "81c8" and to_hex(l300) == "82012c":
        print("PASS  der_length(200)=81c8  der_length(300)=82012c")
        passed += 1
    else:
        print("FAIL  der_length 200=", to_hex(l200), " 300=", to_hex(l300))
        failed += 1

    # INTEGER with high bit set gets 0x00 prefix: 0xFF -> 02 02 00 ff
    var ib = List[UInt8]()
    ib.append(UInt8(0xFF))
    var di = der_integer(ib)
    if to_hex(di) == "020200ff":
        print("PASS  der_integer(0xff) == 02 02 00 ff")
        passed += 1
    else:
        print("FAIL  der_integer(0xff)  got ", to_hex(di))
        failed += 1

    # SEQUENCE { INTEGER 1, INTEGER 256 } -> 30 06 02 01 01 02 02 01 00
    var i1b = List[UInt8](); i1b.append(UInt8(0x01))
    var i256b = List[UInt8](); i256b.append(UInt8(0x01)); i256b.append(UInt8(0x00))
    var seq_children = List[List[UInt8]]()
    seq_children.append(der_integer(i1b))
    seq_children.append(der_integer(i256b))
    var seq = der_sequence(seq_children)
    if to_hex(seq) == "300702010102020100":
        print("PASS  SEQ{INT 1, INT 256} == 30 07 02 01 01 02 02 01 00")
        passed += 1
    else:
        print("FAIL  SEQ{INT 1, INT 256}  got ", to_hex(seq))
        failed += 1

    # DigestInfo AlgorithmIdentifier: SEQ{ OID sha256, NULL }
    #   -> 30 0d 06 09 60 86 48 01 65 03 04 02 01 05 00
    var alg_children = List[List[UInt8]]()
    alg_children.append(der_oid("2.16.840.1.101.3.4.2.1"))
    alg_children.append(der_null())
    var algid = der_sequence(alg_children)
    if to_hex(algid) == "300d06096086480165030402010500":
        print("PASS  AlgorithmIdentifier(sha256,NULL) DER exact")
        passed += 1
    else:
        print("FAIL  AlgorithmIdentifier  got ", to_hex(algid))
        failed += 1

    print("")
    print("=== RSA sign + self-verify ===")

    var digest = hex_to_bytes(MSG_DIGEST_HEX)
    # ensure 32 bytes (leading-zero strip in hex_to_bytes won't happen here)
    if len(digest) != 32:
        print("WARN  digest length =", len(digest), " (expected 32)")

    var sig = rsa_sign_pkcs1v15_sha256(N_HEX, D_HEX, digest)
    print("      signature length =", len(sig), " bytes")
    write_file("/tmp/mojo_sig.bin", sig)
    print("      wrote /tmp/mojo_sig.bin")

    # Self-check: recovered EM via public op must equal the EM we constructed.
    from pdf.rsa import emsa_pkcs1v15_sha256
    var em_built = emsa_pkcs1v15_sha256(digest, len(sig))
    var em_recovered = rsa_public_raw(N_HEX, E_HEX, sig)
    if bytes_eq(em_built, em_recovered):
        print("PASS  rsa_public_raw(n,e,sig) recovers constructed EM")
        passed += 1
    else:
        print("FAIL  EM mismatch")
        print("      built     =", to_hex(em_built))
        print("      recovered =", to_hex(em_recovered))
        failed += 1

    # Sanity: EM begins 00 01 ff ... and ends with the DigestInfo
    var em_ok = (len(em_built) == len(sig)) and (em_built[0] == UInt8(0x00)) \
        and (em_built[1] == UInt8(0x01)) and (em_built[2] == UInt8(0xFF))
    if em_ok:
        print("PASS  EM padding starts 00 01 ff ...")
        passed += 1
    else:
        print("FAIL  EM padding malformed")
        failed += 1

    print("")
    print("=== SUMMARY ===")
    print("passed =", passed, " failed =", failed)
    if failed == 0:
        print("ALL MOJO CHECKS PASS")
        print("Now run pdf/tests/rsa_verify.sh for the openssl dgst -verify oracle.")
    else:
        print("THERE ARE FAILURES")
