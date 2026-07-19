# pdf/tests/ecdsa_test.mojo
# Drives the pure-Mojo ECDSA (P-256) signer and the ECDSA-signed-PDF path.
#
# Sanity (self-contained, no external tools):
#   * 2*G via point_double and via scalar_mul agree.
#   * n*G == infinity  AND  (n-1)*G == -G.
#
# Decisive crypto (requires files prepared by ecdsa_verify.sh):
#   /tmp/ec_d.hex          -- P-256 private scalar d (lowercase hex)
#   /tmp/ec_cert.der       -- signer certificate, DER
#   /tmp/msg.txt           -- a message to sign (raw ECDSA path)
#   /tmp/pdf_showcase.pdf  -- an existing simple PDF to sign
#
# Produces:
#   /tmp/ec_sig.der        -- DER ECDSA signature over sha256(/tmp/msg.txt)
#   /tmp/pdf_ecsigned.pdf  -- the ECDSA-signed PDF
#
# Run from /home/alex/MOJO-libs:
#   pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . pdf/tests/ecdsa_test.mojo

from pdf.bigint import BigInt
from pdf.sha256 import sha256
from pdf.ec import (
    Point,
    generator,
    scalar_mul,
    point_double,
    point_negate,
    p256_n,
    p256_gx,
    p256_gy,
)
from pdf.sign_ec import ecdsa_sign, ecdsa_sig_der, sign_pdf_ecdsa


def _read_text(path: String) raises -> String:
    var f = open(path, "r")
    var d = f.read_bytes()
    f.close()
    var s = String("")
    for i in range(len(d)):
        var c = Int(d[i])
        if c == 10 or c == 13 or c == 32 or c == 9:
            continue
        s += chr(c)
    return s^


def _read_bytes(path: String) raises -> List[UInt8]:
    var f = open(path, "r")
    var d = f.read_bytes()
    f.close()
    var out = List[UInt8]()
    for i in range(len(d)):
        out.append(d[i])
    return out^


def _write_bytes(path: String, buf: List[UInt8]) raises:
    with open(path, "w") as f:
        f.write_bytes(Span(buf))


def main() raises:
    print("=== ECDSA P-256 (pure Mojo) ===")

    # -------- EC sanity --------
    var G = generator()

    # 2*G two ways.
    var twoG_dbl = point_double(G)
    var twoG_smul = scalar_mul(BigInt(UInt64(2)), G)
    if twoG_dbl.equals(twoG_smul):
        print("OK  2*G: double == scalar_mul(2)")
        print("    2G.x =", twoG_dbl.x.to_hex())
        print("    2G.y =", twoG_dbl.y.to_hex())
    else:
        print("FAIL 2*G mismatch")

    # Known-answer for 2*G (SEC test vector).
    comptime TWOG_X = "7cf27b188d034f7e8a52380304b51ac3c08969e277f21b35a60b48fc47669978"
    comptime TWOG_Y = "07775510db8ed040293d9ac69f7430dbba7dade63ce982299e04b79d227873d1"
    if twoG_dbl.x.to_hex() == String(TWOG_X) and twoG_dbl.y.to_hex() == String(TWOG_Y):
        print("OK  2*G matches SEC known-answer vector")
    else:
        print("FAIL 2*G != known-answer vector")

    # n*G == infinity.
    var n = p256_n()
    var nG = scalar_mul(n, G)
    if nG.is_inf():
        print("OK  n*G == point at infinity")
    else:
        print("FAIL n*G != infinity")

    # (n-1)*G == -G.
    var nm1 = n.sub(BigInt(UInt64(1)))
    var nm1G = scalar_mul(nm1, G)
    var negG = point_negate(G)
    if nm1G.equals(negG):
        print("OK  (n-1)*G == -G")
    else:
        print("FAIL (n-1)*G != -G")

    # -------- Raw ECDSA over sha256(/tmp/msg.txt) --------
    print("--- raw ECDSA ---")
    var d_hex = _read_text("/tmp/ec_d.hex")
    print("  d_hex len =", len(d_hex))
    var msg = _read_bytes("/tmp/msg.txt")
    var h = sha256(msg)
    var sig = ecdsa_sign(d_hex, h)
    var der = ecdsa_sig_der(sig.r, sig.s)
    _write_bytes("/tmp/ec_sig.der", der)
    print("  sha256(msg) over", len(msg), "bytes -> ECDSA DER", len(der), "bytes")
    print("  wrote /tmp/ec_sig.der")

    # -------- ECDSA-signed PDF --------
    print("--- ECDSA PDF ---")
    var cert = _read_bytes("/tmp/ec_cert.der")
    print("  cert DER bytes =", len(cert))
    sign_pdf_ecdsa(
        "/tmp/pdf_showcase.pdf",
        "/tmp/pdf_ecsigned.pdf",
        d_hex,
        cert,
        String("Mojo pure-Mojo ECDSA signature"),
    )
    var signed = _read_bytes("/tmp/pdf_ecsigned.pdf")
    print("  wrote /tmp/pdf_ecsigned.pdf  (", len(signed), "bytes )")
    print("DONE - run pdf/tests/ecdsa_verify.sh for the openssl / pdfsig oracle.")
