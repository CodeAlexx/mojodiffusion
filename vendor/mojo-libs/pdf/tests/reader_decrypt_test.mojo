# pdf/tests/reader_decrypt_test.mojo
# Verifies pdf/reader.mojo's open_encrypted + decryption-aware accessors against
# this library's OWN encrypted outputs (RC4, AESV2, AESV3).
#
# For each algorithm:
#   1) ensure the encrypted file exists (regenerate via the document API if not).
#   2) open_encrypted(path, "userpw"), decrypt page 0's content stream, assert it
#      contains the expected "( … ) Tj" text literal, print the snippet.
#   3) confirm a WRONG password raises.
#
# Run from /home/alex/MOJO-libs:
#   pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . pdf/tests/reader_decrypt_test.mojo

from pdf.reader import PdfReader
from pdf.document import PdfDoc
from pdf.objects import append_str
from pdf.text import standard_font_object


struct TT(Copyable, Movable):
    var p: Int
    var f: Int
    def __init__(out self):
        self.p = 0; self.f = 0
    def ck(mut self, cond: Bool, name: String):
        if cond:
            self.p += 1
            print("  PASS:", name)
        else:
            self.f += 1
            print("  FAIL:", name)


def _file_exists(path: String) -> Bool:
    try:
        var f = open(path, "r")
        f.close()
        return True
    except:
        return False


def _contains(hay: String, needle: String) -> Bool:
    var hb = hay.as_bytes()
    var nb = needle.as_bytes()
    if len(nb) == 0 or len(nb) > len(hb):
        return False
    var i = 0
    var last = len(hb) - len(nb)
    while i <= last:
        var ok = True
        for k in range(len(nb)):
            if hb[i + k] != nb[k]:
                ok = False
                break
        if ok:
            return True
        i += 1
    return False


def _build_doc(text: String) raises -> PdfDoc:
    var doc = PdfDoc()
    doc.set_info(
        String("Encrypted Doc Title"),
        String("Mojo"),
        String("Encryption test"),
        String("MojoPDF"),
    )
    var font_obj = standard_font_object(String("Helvetica"))
    var font_num = doc.add_object(font_obj)
    var content = List[UInt8]()
    append_str(content, "BT /F1 24 Tf 72 700 Td (")
    append_str(content, text)
    append_str(content, ") Tj ET\n")
    var resources = List[UInt8]()
    append_str(resources, "<< /Font << /F1 ")
    append_str(resources, String(font_num))
    append_str(resources, " 0 R >> >>")
    var _page = doc.add_page(612.0, 792.0, content, resources)
    return doc^


def _ensure_rc4(path: String) raises:
    if _file_exists(path):
        return
    var doc = _build_doc(String("Encrypted Mojo PDF 2026"))
    doc.save_encrypted(path, "userpw", "ownerpw")
    print("  (regenerated", path, ")")


def _ensure_aes128(path: String) raises:
    if _file_exists(path):
        return
    var doc = _build_doc(String("AES Encrypted Mojo 2026"))
    doc.save_encrypted_aes128(path, "userpw", "ownerpw")
    print("  (regenerated", path, ")")


def _ensure_aes256(path: String) raises:
    if _file_exists(path):
        return
    var doc = _build_doc(String("AES Encrypted Mojo 2026"))
    doc.save_encrypted_aes256(path, "userpw", "ownerpw")
    print("  (regenerated", path, ")")


def _wrong_pw_raises(path: String) -> Bool:
    try:
        var r = PdfReader()
        r.open_encrypted(path, "WRONGPW")
        # If it did not raise, force-touch the result so it isn't optimized away.
        _ = r.page_count()
        return False
    except:
        return True


def _run_one(mut t: TT, label: String, path: String, expect: String) raises:
    print("== ", label, " (", path, ") ==")
    var r = PdfReader()
    r.open_encrypted(path, "userpw")
    var content = r.extract_page_text_ish(0)
    var snippet_ok = _contains(content, expect)
    # Print a verbatim snippet around the Tj operator.
    print("  decrypted content stream:")
    print("    ", content)
    t.ck(snippet_ok, label + ": decrypted content contains '(" + expect + ") Tj'")
    t.ck(_wrong_pw_raises(path), label + ": wrong password raises")


def main() raises:
    var rc4_path = String("/tmp/pdf_enc.pdf")
    var aes128_path = String("/tmp/pdf_aes128.pdf")
    var aes256_path = String("/tmp/pdf_aes256.pdf")

    _ensure_rc4(rc4_path)
    _ensure_aes128(aes128_path)
    _ensure_aes256(aes256_path)

    var t = TT()

    _run_one(t, String("RC4 (V2/R3)"), rc4_path, String("Encrypted Mojo PDF 2026"))
    _run_one(t, String("AESV2 (V4/R4)"), aes128_path, String("AES Encrypted Mojo 2026"))
    _run_one(t, String("AESV3 (V5/R6)"), aes256_path, String("AES Encrypted Mojo 2026"))

    print("")
    print("RESULTS: ", t.p, " passed, ", t.f, " failed")
    if t.f != 0:
        raise Error("reader_decrypt_test: failures present")
    print("ALL GREEN")
