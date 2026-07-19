# pdf/sha256.mojo
# Pure-Mojo SHA-256 (FIPS 180-4). Returns the 32 raw digest bytes.
# Used by the modern PDF security handlers (AESV3 / PDF 2.0 key derivation).
#
# All arithmetic is done on UInt32 with explicit & 0xFFFFFFFF masking.
# SHA-256 is BIG-endian: message words and the appended bit-length are
# packed/unpacked most-significant byte first.


def _rotr32(x: UInt32, n: UInt32) -> UInt32:
    # Rotate-right within 32 bits.
    var v = x & 0xFFFFFFFF
    return ((v >> n) | (v << (UInt32(32) - n))) & 0xFFFFFFFF


def sha256(data: List[UInt8]) raises -> List[UInt8]:
    # Round constants K[0..63] = first 32 bits of the fractional parts of the
    # cube roots of the first 64 primes (FIPS 180-4).
    var K = List[UInt32]()
    K.append(0x428a2f98); K.append(0x71374491); K.append(0xb5c0fbcf); K.append(0xe9b5dba5)
    K.append(0x3956c25b); K.append(0x59f111f1); K.append(0x923f82a4); K.append(0xab1c5ed5)
    K.append(0xd807aa98); K.append(0x12835b01); K.append(0x243185be); K.append(0x550c7dc3)
    K.append(0x72be5d74); K.append(0x80deb1fe); K.append(0x9bdc06a7); K.append(0xc19bf174)
    K.append(0xe49b69c1); K.append(0xefbe4786); K.append(0x0fc19dc6); K.append(0x240ca1cc)
    K.append(0x2de92c6f); K.append(0x4a7484aa); K.append(0x5cb0a9dc); K.append(0x76f988da)
    K.append(0x983e5152); K.append(0xa831c66d); K.append(0xb00327c8); K.append(0xbf597fc7)
    K.append(0xc6e00bf3); K.append(0xd5a79147); K.append(0x06ca6351); K.append(0x14292967)
    K.append(0x27b70a85); K.append(0x2e1b2138); K.append(0x4d2c6dfc); K.append(0x53380d13)
    K.append(0x650a7354); K.append(0x766a0abb); K.append(0x81c2c92e); K.append(0x92722c85)
    K.append(0xa2bfe8a1); K.append(0xa81a664b); K.append(0xc24b8b70); K.append(0xc76c51a3)
    K.append(0xd192e819); K.append(0xd6990624); K.append(0xf40e3585); K.append(0x106aa070)
    K.append(0x19a4c116); K.append(0x1e376c08); K.append(0x2748774c); K.append(0x34b0bcb5)
    K.append(0x391c0cb3); K.append(0x4ed8aa4a); K.append(0x5b9cca4f); K.append(0x682e6ff3)
    K.append(0x748f82ee); K.append(0x78a5636f); K.append(0x84c87814); K.append(0x8cc70208)
    K.append(0x90befffa); K.append(0xa4506ceb); K.append(0xbef9a3f7); K.append(0xc67178f2)

    # --- Build padded message ---
    var msg = List[UInt8]()
    var orig_len = len(data)
    for i in range(orig_len):
        msg.append(data[i])

    # Original message length in BITS, modulo 2^64.
    var bit_len = UInt64(orig_len) * UInt64(8)

    # Append 0x80 then 0x00 until length ≡ 56 (mod 64).
    msg.append(UInt8(0x80))
    while (len(msg) % 64) != 56:
        msg.append(UInt8(0))

    # Append the 64-bit length, BIG-endian.
    for i in range(8):
        var shift = UInt64(8) * UInt64(7 - i)
        msg.append(UInt8((bit_len >> shift) & UInt64(0xFF)))

    # --- Initial hash state: first 32 bits of fractional parts of square
    # roots of the first 8 primes (FIPS 180-4). ---
    var h0 = UInt32(0x6a09e667)
    var h1 = UInt32(0xbb67ae85)
    var h2 = UInt32(0x3c6ef372)
    var h3 = UInt32(0xa54ff53a)
    var h4 = UInt32(0x510e527f)
    var h5 = UInt32(0x9b05688c)
    var h6 = UInt32(0x1f83d9ab)
    var h7 = UInt32(0x5be0cd19)

    var total = len(msg)
    var chunk = 0
    while chunk < total:
        # Decode 16 BIG-endian 32-bit words from this 64-byte chunk into the
        # first 16 entries of the 64-entry message schedule.
        var W = List[UInt32]()
        for w in range(16):
            var base = chunk + w * 4
            var word = (UInt32(msg[base]) << 24) \
                | (UInt32(msg[base + 1]) << 16) \
                | (UInt32(msg[base + 2]) << 8) \
                | UInt32(msg[base + 3])
            W.append(word & 0xFFFFFFFF)

        # Extend the schedule to 64 words.
        for i in range(16, 64):
            var w15 = W[i - 15]
            var w2 = W[i - 2]
            var s0 = _rotr32(w15, 7) ^ _rotr32(w15, 18) ^ ((w15 >> 3) & 0xFFFFFFFF)
            var s1 = _rotr32(w2, 17) ^ _rotr32(w2, 19) ^ ((w2 >> 10) & 0xFFFFFFFF)
            var nw = (W[i - 16] + s0 + W[i - 7] + s1) & 0xFFFFFFFF
            W.append(nw)

        var a = h0
        var b = h1
        var c = h2
        var d = h3
        var e = h4
        var f = h5
        var g = h6
        var h = h7

        for i in range(64):
            var S1 = _rotr32(e, 6) ^ _rotr32(e, 11) ^ _rotr32(e, 25)
            var ch = (e & f) ^ ((~e) & g)
            var temp1 = (h + S1 + ch + K[i] + W[i]) & 0xFFFFFFFF
            var S0 = _rotr32(a, 2) ^ _rotr32(a, 13) ^ _rotr32(a, 22)
            var maj = (a & b) ^ (a & c) ^ (b & c)
            var temp2 = (S0 + maj) & 0xFFFFFFFF

            h = g
            g = f
            f = e
            e = (d + temp1) & 0xFFFFFFFF
            d = c
            c = b
            b = a
            a = (temp1 + temp2) & 0xFFFFFFFF

        h0 = (h0 + a) & 0xFFFFFFFF
        h1 = (h1 + b) & 0xFFFFFFFF
        h2 = (h2 + c) & 0xFFFFFFFF
        h3 = (h3 + d) & 0xFFFFFFFF
        h4 = (h4 + e) & 0xFFFFFFFF
        h5 = (h5 + f) & 0xFFFFFFFF
        h6 = (h6 + g) & 0xFFFFFFFF
        h7 = (h7 + h) & 0xFFFFFFFF

        chunk = chunk + 64

    # Output digest: h0..h7 each BIG-endian.
    var out = List[UInt8]()
    var words = List[UInt32]()
    words.append(h0); words.append(h1); words.append(h2); words.append(h3)
    words.append(h4); words.append(h5); words.append(h6); words.append(h7)
    for wi in range(8):
        var v = words[wi]
        out.append(UInt8((v >> 24) & 0xFF))
        out.append(UInt8((v >> 16) & 0xFF))
        out.append(UInt8((v >> 8) & 0xFF))
        out.append(UInt8(v & 0xFF))
    return out^


def sha256_hex(data: List[UInt8]) raises -> String:
    # Lowercase hex of the 32-byte digest.
    comptime HEX = "0123456789abcdef"
    var hexb = HEX.as_bytes()
    var digest = sha256(data)
    var s = String("")
    for i in range(len(digest)):
        var byte = Int(digest[i])
        s += chr(Int(hexb[(byte >> 4) & 0xF]))
        s += chr(Int(hexb[byte & 0xF]))
    return s^
