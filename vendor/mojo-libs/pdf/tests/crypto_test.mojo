# pdf/tests/crypto_test.mojo
# Verifies MD5 and RC4 primitives against published test vectors.
# Checks EXACT bytes (not asserts) and prints PASS/FAIL per case.
#
# Run from /home/alex/MOJO-libs:
#   pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . pdf/tests/crypto_test.mojo

from pdf.md5 import md5, md5_hex
from pdf.rc4 import rc4


def bytes_of(s: String) raises -> List[UInt8]:
    # ASCII string -> byte list.
    var src = s.as_bytes()
    var out = List[UInt8]()
    for i in range(len(src)):
        out.append(src[i])
    return out^


def to_hex(data: List[UInt8]) raises -> String:
    comptime HEX = "0123456789abcdef"
    var hexb = HEX.as_bytes()
    var s = String("")
    for i in range(len(data)):
        var byte = Int(data[i])
        s += chr(Int(hexb[(byte >> 4) & 0xF]))
        s += chr(Int(hexb[byte & 0xF]))
    return s^


def main() raises:
    var passed = 0
    var failed = 0

    print("=== MD5 vectors (RFC 1321) ===")

    # (input, expected lowercase hex)
    var md5_in = List[String]()
    var md5_exp = List[String]()
    md5_in.append(""); md5_exp.append("d41d8cd98f00b204e9800998ecf8427e")
    md5_in.append("abc"); md5_exp.append("900150983cd24fb0d6963f7d28e17f72")
    md5_in.append("message digest"); md5_exp.append("f96b697d7cb7938d525a2f31aaf161d0")
    md5_in.append("The quick brown fox jumps over the lazy dog")
    md5_exp.append("9e107d9d372bb6826bd81d3542a419d6")

    for i in range(len(md5_in)):
        var got = md5_hex(bytes_of(md5_in[i]))
        var want = md5_exp[i]
        var ok = (got == want)
        var label = String("FAIL")
        if ok:
            label = String("PASS")
            passed += 1
        else:
            failed += 1
        print(label, ' md5("', md5_in[i], '")')
        print("      got =", got)
        print("      want=", want)

    print("")
    print("=== RC4 vectors (classic) ===")

    # (key, plaintext, expected ciphertext hex lowercase)
    var rc4_key = List[String]()
    var rc4_pt = List[String]()
    var rc4_ct = List[String]()
    rc4_key.append("Key"); rc4_pt.append("Plaintext"); rc4_ct.append("bbf316e8d940af0ad3")
    rc4_key.append("Wiki"); rc4_pt.append("pedia"); rc4_ct.append("1021bf0420")
    rc4_key.append("Secret"); rc4_pt.append("Attack at dawn")
    rc4_ct.append("45a01f645fc35b383552544b9bf5")

    for i in range(len(rc4_key)):
        var ct = rc4(bytes_of(rc4_key[i]), bytes_of(rc4_pt[i]))
        var got = to_hex(ct)
        var want = rc4_ct[i]
        var ok = (got == want)
        var label = String("FAIL")
        if ok:
            label = String("PASS")
            passed += 1
        else:
            failed += 1
        print(label, ' rc4(key="', rc4_key[i], '", pt="', rc4_pt[i], '")')
        print("      got =", got)
        print("      want=", want)

    print("")
    print("=== RC4 round-trip ===")
    var rt_key = bytes_of("hunter2-secret-key")
    var rt_msg = bytes_of("The quick brown fox jumps over the lazy dog 0123456789!@#")
    var enc = rc4(rt_key, rt_msg)
    var dec = rc4(rt_key, enc)
    var rt_ok = (len(dec) == len(rt_msg))
    if rt_ok:
        for i in range(len(rt_msg)):
            if dec[i] != rt_msg[i]:
                rt_ok = False
    # Also confirm ciphertext actually differs from plaintext.
    var differs = (len(enc) == len(rt_msg))
    if differs:
        var same = True
        for i in range(len(rt_msg)):
            if enc[i] != rt_msg[i]:
                same = False
        differs = not same
    if rt_ok and differs:
        print("PASS  rc4(key, rc4(key, msg)) == msg  (and ciphertext != plaintext)")
        passed += 1
    else:
        print("FAIL  round-trip rt_ok=", rt_ok, " ct_differs=", differs)
        failed += 1
    print("      plaintext  =", to_hex(rt_msg))
    print("      ciphertext =", to_hex(enc))
    print("      decrypted  =", to_hex(dec))

    print("")
    print("=== SUMMARY ===")
    print("passed =", passed, " failed =", failed)
    if failed == 0:
        print("ALL PASS")
    else:
        print("THERE ARE FAILURES")
