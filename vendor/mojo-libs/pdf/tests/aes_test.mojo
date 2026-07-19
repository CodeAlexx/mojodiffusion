# pdf/tests/aes_test.mojo
# Verifies AES-128/256 against FIPS-197 / NIST known-answer block vectors,
# plus CBC round-trip for 16- and 32-byte keys.
# Checks EXACT hex (not asserts) and prints PASS/FAIL + actual hex per case.
#
# Run from /home/alex/MOJO-libs:
#   pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . pdf/tests/aes_test.mojo

from pdf.aes import (
    aes_encrypt_block,
    aes_decrypt_block,
    aes_cbc_encrypt,
    aes_cbc_decrypt,
    _key_expansion,
    _sbox,
)


def from_hex(s: String) raises -> List[UInt8]:
    # Parse a lowercase/uppercase hex string into bytes.
    var src = s.as_bytes()
    var n = len(src)
    var out = List[UInt8]()
    var i = 0
    while i < n:
        var hi = _hex_val(src[i])
        var lo = _hex_val(src[i + 1])
        out.append(UInt8((hi << 4) | lo))
        i += 2
    return out^


def _hex_val(b: UInt8) raises -> Int:
    var c = Int(b)
    if c >= 48 and c <= 57:
        return c - 48          # '0'..'9'
    if c >= 97 and c <= 102:
        return c - 97 + 10     # 'a'..'f'
    if c >= 65 and c <= 70:
        return c - 65 + 10     # 'A'..'F'
    raise Error("from_hex: bad hex digit")


def to_hex(data: List[UInt8]) raises -> String:
    comptime HEX = "0123456789abcdef"
    var hexb = HEX.as_bytes()
    var s = String("")
    for i in range(len(data)):
        var byte = Int(data[i])
        s += chr(Int(hexb[(byte >> 4) & 0xF]))
        s += chr(Int(hexb[byte & 0xF]))
    return s^


def bytes_of(s: String) raises -> List[UInt8]:
    var src = s.as_bytes()
    var out = List[UInt8]()
    for i in range(len(src)):
        out.append(src[i])
    return out^


def eq(a: List[UInt8], b: List[UInt8]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def main() raises:
    var passed = 0
    var failed = 0
    var sbox = _sbox()

    print("=== AES single-block vectors (FIPS-197) ===")

    # --- AES-128 ---
    var k128 = from_hex("000102030405060708090a0b0c0d0e0f")
    var pt = from_hex("00112233445566778899aabbccddeeff")
    var exp128 = String("69c4e0d86a7b0430d8cdb78070b4c55a")
    var rk128 = _key_expansion(k128, sbox)
    var ct128 = aes_encrypt_block(rk128, pt)
    var got128 = to_hex(ct128)
    if got128 == exp128:
        print("PASS  AES-128 encrypt block")
        passed += 1
    else:
        print("FAIL  AES-128 encrypt block")
        failed += 1
    print("      got =", got128)
    print("      want=", exp128)

    # AES-128 decrypt back to plaintext.
    var dec128 = aes_decrypt_block(rk128, ct128)
    if eq(dec128, pt):
        print("PASS  AES-128 decrypt block (round-trip)")
        passed += 1
    else:
        print("FAIL  AES-128 decrypt block (round-trip)")
        failed += 1
    print("      got =", to_hex(dec128))
    print("      want=", to_hex(pt))

    # --- AES-256 ---
    var k256 = from_hex("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f")
    var exp256 = String("8ea2b7ca516745bfeafc49904b496089")
    var rk256 = _key_expansion(k256, sbox)
    var ct256 = aes_encrypt_block(rk256, pt)
    var got256 = to_hex(ct256)
    if got256 == exp256:
        print("PASS  AES-256 encrypt block")
        passed += 1
    else:
        print("FAIL  AES-256 encrypt block")
        failed += 1
    print("      got =", got256)
    print("      want=", exp256)

    var dec256 = aes_decrypt_block(rk256, ct256)
    if eq(dec256, pt):
        print("PASS  AES-256 decrypt block (round-trip)")
        passed += 1
    else:
        print("FAIL  AES-256 decrypt block (round-trip)")
        failed += 1
    print("      got =", to_hex(dec256))
    print("      want=", to_hex(pt))

    print("")
    print("=== AES-CBC round-trip (PKCS#7) ===")

    var iv = from_hex("000102030405060708090a0b0c0d0e0f")

    # Test messages: empty, partial, exact-block-multiple, longer multi-block.
    var msgs = List[String]()
    msgs.append("")
    msgs.append("hello")
    msgs.append("YELLOW SUBMARINE")  # exactly 16 bytes
    msgs.append("The quick brown fox jumps over the lazy dog 0123456789!@#")
    msgs.append("exactly thirty-two bytes long!!!")  # 32 bytes

    for ki in range(2):
        var key: List[UInt8]
        var klabel: String
        if ki == 0:
            key = from_hex("000102030405060708090a0b0c0d0e0f")
            klabel = String("AES-128 (16-byte key)")
        else:
            key = from_hex("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f")
            klabel = String("AES-256 (32-byte key)")

        print("--", klabel, "--")
        for mi in range(len(msgs)):
            var msg = bytes_of(msgs[mi])
            var ct = aes_cbc_encrypt(key, iv, msg)
            var rt = aes_cbc_decrypt(key, iv, ct)
            var ok_rt = eq(rt, msg)
            # ciphertext must be block-aligned and (for non-empty) differ from plaintext
            var ok_len = (len(ct) % 16) == 0 and len(ct) >= 16
            if ok_rt and ok_len:
                print("PASS  cbc round-trip msg[", mi, "] len=", len(msg))
                passed += 1
            else:
                print("FAIL  cbc round-trip msg[", mi, "] ok_rt=", ok_rt, " ok_len=", ok_len)
                failed += 1
            print("        msg = '", msgs[mi], "'")
            print("        ct  =", to_hex(ct))
            print("        dec = '", recover_str(rt), "'")

    print("")
    print("=== AES-128-CBC known ciphertext (cross-check vs Python) ===")
    # key=all-zero 16B, iv=all-zero 16B, plaintext = 16 bytes 0x00.
    # Python cryptography produces this exact ciphertext (first block of the
    # padded output). See aes_test cross-check note.
    var zk = from_hex("00000000000000000000000000000000")
    var ziv = from_hex("00000000000000000000000000000000")
    var zpt = from_hex("00000000000000000000000000000000")
    var zct = aes_cbc_encrypt(zk, ziv, zpt)
    # zpt is exactly one block -> PKCS#7 adds a full padding block; first 16
    # bytes are the encryption of the zero block under the zero key.
    var first16 = List[UInt8]()
    for i in range(16):
        first16.append(zct[i])
    var zexp = String("66e94bd4ef8a2c3b884cfa59ca342b2e")
    if to_hex(first16) == zexp:
        print("PASS  AES-128-CBC zero-block first ciphertext block")
        passed += 1
    else:
        print("FAIL  AES-128-CBC zero-block first ciphertext block")
        failed += 1
    print("      got =", to_hex(first16))
    print("      want=", zexp)
    print("      full ct (incl pad block) =", to_hex(zct))

    print("")
    print("=== SUMMARY ===")
    print("passed =", passed, " failed =", failed)
    if failed == 0:
        print("ALL PASS")
    else:
        print("THERE ARE FAILURES")


def recover_str(data: List[UInt8]) raises -> String:
    # Best-effort ASCII rendering for display only.
    var s = String("")
    for i in range(len(data)):
        s += chr(Int(data[i]))
    return s^
