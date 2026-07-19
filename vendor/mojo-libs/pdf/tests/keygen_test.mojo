# pdf/tests/keygen_test.mojo
# Drives pure-Mojo RSA key generation + self-signed X.509 certificate.
#
# Run from /home/alex/MOJO-libs:
#   pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . pdf/tests/keygen_test.mojo
#
# Produces:
#   /tmp/mojo_rsa.der   PKCS#1 RSAPrivateKey DER   (openssl rsa -check)
#   /tmp/mojo_cert.der  self-signed X.509 v3 DER   (openssl x509 / verify)
#   /tmp/mojo_n.hex /tmp/mojo_e.hex /tmp/mojo_d.hex  key components
#   /tmp/mojo_msg.txt   a message
#   /tmp/mojo_msgsig.bin RSA PKCS#1v1.5-SHA256 signature over the message
#
# The decisive cryptographic validation (openssl rsa -check, x509 -text,
# verify, dgst -verify) is run by the companion shell driver keygen_verify.sh.

from pdf.keygen import gen_rsa, rsa_private_der, RsaKey
from pdf.x509 import make_self_signed_cert
from pdf.rsa import rsa_sign_pkcs1v15_sha256
from pdf.sha256 import sha256


def write_bytes(path: String, data: List[UInt8]) raises:
    # Write RAW bytes (Span), NOT a String — chr() would UTF-8-expand >=0x80.
    with open(path, "w") as f:
        f.write_bytes(Span(data))


def write_text(path: String, s: String) raises:
    var b = s.as_bytes()
    var data = List[UInt8]()
    for i in range(len(b)):
        data.append(b[i])
    with open(path, "w") as f:
        f.write_bytes(Span(data))


def main() raises:
    print("=== pure-Mojo RSA keygen + self-signed cert ===")

    comptime BITS = 512
    print("  generating", BITS, "-bit RSA key (pure-Mojo bigint; this is slow)...")

    var key = gen_rsa(BITS)
    print("  n bits (hex chars) =", len(key.n_hex))
    print("  e =", key.e_hex)
    print("  d hex chars =", len(key.d_hex))

    # --- 1. PKCS#1 RSAPrivateKey DER ---
    var priv = rsa_private_der(key)
    write_bytes("/tmp/mojo_rsa.der", priv)
    print("  wrote /tmp/mojo_rsa.der  (", len(priv), " bytes )")

    # key component hex files (for openssl dgst -verify pubkey extraction)
    write_text("/tmp/mojo_n.hex", key.n_hex)
    write_text("/tmp/mojo_e.hex", key.e_hex)
    write_text("/tmp/mojo_d.hex", key.d_hex)

    # --- 2. self-signed certificate ---
    var cert = make_self_signed_cert(key, String("Mojo Test CA"))
    write_bytes("/tmp/mojo_cert.der", cert)
    print("  wrote /tmp/mojo_cert.der  (", len(cert), " bytes )")

    # --- 3. sign a message with the generated key ---
    var msg_text = String("pure-Mojo RSA signing works")
    write_text("/tmp/mojo_msg.txt", msg_text)
    var msg_bytes = List[UInt8]()
    var mb = msg_text.as_bytes()
    for i in range(len(mb)):
        msg_bytes.append(mb[i])
    var digest = sha256(msg_bytes)
    var sig = rsa_sign_pkcs1v15_sha256(key.n_hex, key.d_hex, digest)
    write_bytes("/tmp/mojo_msgsig.bin", sig)
    print("  wrote /tmp/mojo_msgsig.bin  (", len(sig), " bytes )")

    print("")
    print("DONE — now run pdf/tests/keygen_verify.sh for the openssl oracle.")
