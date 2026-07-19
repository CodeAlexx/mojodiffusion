from pdf.bigint import BigInt, bi_modinv

def check(a: Int, m: Int, want: Int) raises:
    var r = bi_modinv(BigInt(UInt64(a)), BigInt(UInt64(m)))
    var got = r.to_hex()
    var w = BigInt(UInt64(want)).to_hex()
    if got == w:
        print("OK  modinv(", a, ",", m, ") =", want)
    else:
        print("FAIL modinv(", a, ",", m, ") got 0x", got, " want 0x", w)

def main() raises:
    check(3, 11, 4)
    check(17, 3120, 2753)
    check(7, 26, 15)
    # large-ish: inverse of 65537 mod 999999937 (prime); verify by multiply
    var a = BigInt(UInt64(65537))
    var m = BigInt(UInt64(999999937))
    var inv = bi_modinv(a, m)
    var prod = a.mul(inv).mod(m)
    if prod.to_hex() == BigInt(UInt64(1)).to_hex():
        print("OK  65537 * inv mod 999999937 == 1")
    else:
        print("FAIL big modinv, prod=0x", prod.to_hex())
