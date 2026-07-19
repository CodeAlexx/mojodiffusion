# pdf/keygen.mojo
# Pure-Mojo RSA key generation.
#
# Generates two random probable primes p, q of bits/2 each, forms the RSA
# modulus n = p*q, public exponent e = 65537, and the private exponent
# d = e^-1 mod lambda(n) where lambda = lcm(p-1, q-1). The resulting key is
# emitted as a PKCS#1 RSAPrivateKey DER blob that openssl accepts.
#
# Primality testing: cheap trial division against a small-prime table, then
# Miller-Rabin (>=16 rounds) using BigInt.modexp.
#
# Performance note: pure-Mojo bigint divmod is bit-by-bit and slow, so larger
# moduli grow superlinearly in wall time. gen_rsa(bits) is fully general (the
# key size is a parameter; openssl accepts whatever size we produce), but the
# verified test uses a small size so it finishes quickly.

from std.random import random_ui64

from pdf.bigint import BigInt, bi_modinv
from pdf.asn1 import der_integer, der_sequence


struct RsaKey(Copyable, Movable):
    var n_hex: String
    var e_hex: String
    var d_hex: String
    var p_hex: String
    var q_hex: String

    def __init__(
        out self,
        var n_hex: String,
        var e_hex: String,
        var d_hex: String,
        var p_hex: String,
        var q_hex: String,
    ):
        self.n_hex = n_hex^
        self.e_hex = e_hex^
        self.d_hex = d_hex^
        self.p_hex = p_hex^
        self.q_hex = q_hex^

    def __init__(out self, *, copy: Self):
        self.n_hex = copy.n_hex
        self.e_hex = copy.e_hex
        self.d_hex = copy.d_hex
        self.p_hex = copy.p_hex
        self.q_hex = copy.q_hex


# ---- small constant BigInts ----

def _one() raises -> BigInt:
    return BigInt(UInt64(1))


def _two() raises -> BigInt:
    return BigInt(UInt64(2))


# ---- small prime table for trial division / Miller-Rabin bases ----

def _small_primes() raises -> List[UInt64]:
    # Odd primes from 3 up to ~300. Used for trial division and as MR bases.
    var sieve_max = 300
    var is_comp = List[Bool]()
    for _ in range(sieve_max + 1):
        is_comp.append(False)
    var primes = List[UInt64]()
    var i = 2
    while i <= sieve_max:
        if not is_comp[i]:
            if i >= 3:
                primes.append(UInt64(i))
            var j = i * i
            while j <= sieve_max:
                is_comp[j] = True
                j += i
        i += 1
    return primes^


# ---- gcd on BigInt ----

def _gcd(var a: BigInt, var b: BigInt) raises -> BigInt:
    while not b.is_zero():
        var r = a.mod(b)
        a = b.copy()
        b = r.copy()
    return a^


# ---- Miller-Rabin probable-prime test ----

def _is_probable_prime(n: BigInt, rounds: Int) raises -> Bool:
    var one = _one()
    var two = _two()

    # n < 2 -> not prime; n == 2 or 3 -> prime; even -> composite.
    if n.compare(two) < 0:
        return False
    if n.compare(two) == 0:
        return True
    var three = BigInt(UInt64(3))
    if n.compare(three) == 0:
        return True
    if not n.test_bit(0):
        return False

    # Trial division by small primes.
    var sp = _small_primes()
    for k in range(len(sp)):
        var p = BigInt(sp[k])
        if n.compare(p) == 0:
            return True
        var r = n.mod(p)
        if r.is_zero():
            return False

    # Write n-1 = d * 2^s with d odd.
    var n_minus_1 = n.sub(one)
    var d = n_minus_1.copy()
    var s = 0
    while not d.test_bit(0):
        d = d.divmod(two).q.copy()
        s += 1

    # Witness loop using small-prime bases (deterministic small bases plus a
    # couple of randoms drawn from the small set), at least `rounds` of them.
    var bases = List[UInt64]()
    # First add fixed small bases known to be strong witnesses.
    var fixed = [UInt64(2), UInt64(3), UInt64(5), UInt64(7), UInt64(11),
                 UInt64(13), UInt64(17), UInt64(19), UInt64(23), UInt64(29),
                 UInt64(31), UInt64(37), UInt64(41), UInt64(43), UInt64(47),
                 UInt64(53)]
    for fi in range(len(fixed)):
        bases.append(fixed[fi])
    # Pad with extra randomized small odd bases if rounds demands more.
    while len(bases) < rounds:
        var rb = random_ui64(UInt64(2), UInt64(201))
        bases.append(rb)

    var n_minus_3 = n.sub(three)
    for bi in range(len(bases)):
        var a = BigInt(bases[bi])
        # Ensure 2 <= a <= n-2; if a too big, reduce.
        if a.compare(n_minus_1) >= 0:
            a = a.mod(n_minus_3)
            a = a.add(two)
        var x = a.modexp(d, n)
        if x.compare(one) == 0 or x.compare(n_minus_1) == 0:
            continue
        var composite = True
        for _ in range(s - 1):
            x = x.mul(x).mod(n)
            if x.compare(n_minus_1) == 0:
                composite = False
                break
        if composite:
            return False
    return True


# ---- random odd candidate of exactly `bits` bits ----

def _random_candidate(bits: Int) raises -> BigInt:
    # Build limbs (32-bit) covering `bits`, fill with random data, then set the
    # top two bits (size guarantee) and the low bit (odd).
    var nlimbs = (bits + 31) // 32
    var limbs = List[UInt32]()
    for _ in range(nlimbs):
        var r = random_ui64(UInt64(0), UInt64(0xFFFFFFFFFFFFFFFF))
        limbs.append(UInt32(r & 0xFFFFFFFF))
    # Mask the top limb so the value is exactly `bits` bits wide.
    var top_bits = bits - (nlimbs - 1) * 32  # 1..32
    var top = limbs[nlimbs - 1]
    if top_bits < 32:
        var mask = (UInt32(1) << UInt32(top_bits)) - UInt32(1)
        top = top & mask
    # Set the top two bits within the `bits` window:
    #   bit (bits-1) guarantees full bit-length;
    #   bit (bits-2) makes p*q have the full 2*bits length.
    var hb = top_bits - 1          # index within top limb of the MSB
    top = top | (UInt32(1) << UInt32(hb))
    if hb >= 1:
        top = top | (UInt32(1) << UInt32(hb - 1))
    limbs[nlimbs - 1] = top
    # Make it odd.
    limbs[0] = limbs[0] | UInt32(1)
    return BigInt(limbs^)


def _gen_prime(bits: Int) raises -> BigInt:
    var two = _two()
    while True:
        var cand = _random_candidate(bits)
        # Step through odd candidates until one passes.
        var tries = 0
        while tries < 5000:
            if _is_probable_prime(cand, 20):
                return cand^
            cand = cand.add(two)
            tries += 1
        # Exhausted local window; draw a fresh candidate.


# ---- main key generation ----

def gen_rsa(bits: Int) raises -> RsaKey:
    if bits < 16:
        raise Error("gen_rsa: bits too small")
    var half = bits // 2
    var e = BigInt(UInt64(65537))
    var one = _one()

    while True:
        var p = _gen_prime(half)
        var q = _gen_prime(half)
        # Require p != q.
        if p.compare(q) == 0:
            continue
        var p1 = p.sub(one)
        var q1 = q.sub(one)
        # e must be coprime with (p-1) and (q-1).
        if not _gcd(BigInt(copy=e), BigInt(copy=p1)).compare(one) == 0:
            continue
        if not _gcd(BigInt(copy=e), BigInt(copy=q1)).compare(one) == 0:
            continue
        var n = p.mul(q)
        # lambda = lcm(p-1, q-1) = (p-1)*(q-1) / gcd(p-1, q-1)
        var g = _gcd(BigInt(copy=p1), BigInt(copy=q1))
        var lam = p1.mul(q1).divmod(g).q.copy()
        var d = bi_modinv(e, lam)
        if d.is_zero() or d.compare(one) == 0:
            continue
        return RsaKey(
            n.to_hex(),
            e.to_hex(),
            d.to_hex(),
            p.to_hex(),
            q.to_hex(),
        )


# ---- PKCS#1 RSAPrivateKey DER ----

def rsa_private_der(key: RsaKey) raises -> List[UInt8]:
    # RSAPrivateKey ::= SEQUENCE {
    #   version           INTEGER (0),
    #   modulus           INTEGER (n),
    #   publicExponent    INTEGER (e),
    #   privateExponent   INTEGER (d),
    #   prime1            INTEGER (p),
    #   prime2            INTEGER (q),
    #   exponent1         INTEGER (d mod (p-1)),
    #   exponent2         INTEGER (d mod (q-1)),
    #   coefficient       INTEGER (q^-1 mod p)
    # }
    var n = BigInt.from_hex(key.n_hex)
    var e = BigInt.from_hex(key.e_hex)
    var d = BigInt.from_hex(key.d_hex)
    var p = BigInt.from_hex(key.p_hex)
    var q = BigInt.from_hex(key.q_hex)
    var one = _one()

    var p1 = p.sub(one)
    var q1 = q.sub(one)
    var dp = d.mod(p1)
    var dq = d.mod(q1)
    var qinv = bi_modinv(q, p)

    var ver = List[UInt8]()
    ver.append(UInt8(0))

    var children = List[List[UInt8]]()
    children.append(der_integer(ver))
    children.append(der_integer(n.to_bytes_be()))
    children.append(der_integer(e.to_bytes_be()))
    children.append(der_integer(d.to_bytes_be()))
    children.append(der_integer(p.to_bytes_be()))
    children.append(der_integer(q.to_bytes_be()))
    children.append(der_integer(dp.to_bytes_be()))
    children.append(der_integer(dq.to_bytes_be()))
    children.append(der_integer(qinv.to_bytes_be()))
    return der_sequence(children)
