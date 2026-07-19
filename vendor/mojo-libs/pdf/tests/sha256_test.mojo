# pdf/tests/sha256_test.mojo
# Verifies SHA-256 against published FIPS 180-4 / NIST test vectors.
# Checks EXACT hex (not asserts) and prints PASS/FAIL + actual hex per case.
#
# Run from /home/alex/MOJO-libs:
#   pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . pdf/tests/sha256_test.mojo

from pdf.sha256 import sha256, sha256_hex


def bytes_of(s: String) raises -> List[UInt8]:
    # ASCII string -> byte list.
    var src = s.as_bytes()
    var out = List[UInt8]()
    for i in range(len(src)):
        out.append(src[i])
    return out^


def main() raises:
    var passed = 0
    var failed = 0

    print("=== SHA-256 vectors (FIPS 180-4) ===")

    var sha_in = List[String]()
    var sha_exp = List[String]()
    sha_in.append("")
    sha_exp.append("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    sha_in.append("abc")
    sha_exp.append("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    sha_in.append("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")
    sha_exp.append("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1")

    for i in range(len(sha_in)):
        var got = sha256_hex(bytes_of(sha_in[i]))
        var want = sha_exp[i]
        var ok = (got == want)
        var label = String("FAIL")
        if ok:
            label = String("PASS")
            passed += 1
        else:
            failed += 1
        print(label, ' sha256("', sha_in[i], '")')
        print("      got =", got)
        print("      want=", want)

    # Long message vector: 1,000,000 'a' characters.
    # Expected: cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0
    print("")
    print("=== SHA-256 one-million 'a' vector ===")
    var million = List[UInt8]()
    for _ in range(1000000):
        million.append(UInt8(0x61))  # 'a'
    var got_m = sha256_hex(million)
    var want_m = String("cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0")
    var ok_m = (got_m == want_m)
    if ok_m:
        print("PASS  sha256( 'a' x 1000000 )")
        passed += 1
    else:
        print("FAIL  sha256( 'a' x 1000000 )")
        failed += 1
    print("      got =", got_m)
    print("      want=", want_m)

    print("")
    print("=== SUMMARY ===")
    print("passed =", passed, " failed =", failed)
    if failed == 0:
        print("ALL PASS")
    else:
        print("THERE ARE FAILURES")
