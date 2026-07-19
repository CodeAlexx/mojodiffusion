# pdf/tests/sign_test.mojo
# Drives the pure-Mojo PDF digital-signature post-processor.
#
# Prerequisites (created by the companion shell driver, sign_verify.sh, OR run
# manually before this test):
#   /tmp/n.hex /tmp/e.hex /tmp/d.hex  -- RSA private key components (lowercase hex)
#   /tmp/sc.der                       -- the signer certificate, DER
#   /tmp/pdf_showcase.pdf             -- an existing simple PDF to sign
#
# This test reads those, calls sign_pdf(...) to produce /tmp/pdf_signed.pdf,
# and prints layout info. The DECISIVE crypto validation is run by the shell
# driver via `pdfsig` (poppler) and/or `openssl cms -verify`.
#
# Run from /home/alex/MOJO-libs:
#   pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . pdf/tests/sign_test.mojo

from pdf.sign import sign_pdf


def _read_text(path: String) raises -> String:
    var f = open(path, "r")
    var d = f.read_bytes()
    f.close()
    var s = String("")
    for i in range(len(d)):
        var c = Int(d[i])
        # strip trailing whitespace/newlines
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


def main() raises:
    print("=== PDF digital signature (pure Mojo) ===")

    var n_hex = _read_text("/tmp/n.hex")
    var e_hex = _read_text("/tmp/e.hex")
    var d_hex = _read_text("/tmp/d.hex")
    print("  n_hex len =", len(n_hex), " e_hex =", e_hex, " d_hex len =", len(d_hex))

    var cert = _read_bytes("/tmp/sc.der")
    print("  cert DER bytes =", len(cert))

    sign_pdf(
        "/tmp/pdf_showcase.pdf",
        "/tmp/pdf_signed.pdf",
        n_hex,
        e_hex,
        d_hex,
        cert,
        String("Mojo pure-Mojo test signature"),
    )

    var signed = _read_bytes("/tmp/pdf_signed.pdf")
    print("  wrote /tmp/pdf_signed.pdf  (", len(signed), " bytes )")
    print("DONE — now run pdf/tests/sign_verify.sh for the pdfsig / openssl oracle.")
