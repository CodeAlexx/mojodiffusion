# pdf/md5.mojo
# Pure-Mojo MD5 (RFC 1321). Returns the 16 raw digest bytes.
# Used by PDF standard security handler (RC4 key derivation).
#
# All arithmetic is done on UInt32 with explicit & 0xFFFFFFFF masking.
# MD5 is little-endian: message words and the appended bit-length are
# packed/unpacked least-significant byte first.


def _rotl32(x: UInt32, n: UInt32) -> UInt32:
    # Rotate-left within 32 bits.
    var v = x & 0xFFFFFFFF
    return ((v << n) | (v >> (UInt32(32) - n))) & 0xFFFFFFFF


def md5(data: List[UInt8]) raises -> List[UInt8]:
    # Per-round left-rotate amounts (RFC 1321).
    var s = List[UInt32]()
    # Round 1
    s.append(7); s.append(12); s.append(17); s.append(22)
    s.append(7); s.append(12); s.append(17); s.append(22)
    s.append(7); s.append(12); s.append(17); s.append(22)
    s.append(7); s.append(12); s.append(17); s.append(22)
    # Round 2
    s.append(5); s.append(9); s.append(14); s.append(20)
    s.append(5); s.append(9); s.append(14); s.append(20)
    s.append(5); s.append(9); s.append(14); s.append(20)
    s.append(5); s.append(9); s.append(14); s.append(20)
    # Round 3
    s.append(4); s.append(11); s.append(16); s.append(23)
    s.append(4); s.append(11); s.append(16); s.append(23)
    s.append(4); s.append(11); s.append(16); s.append(23)
    s.append(4); s.append(11); s.append(16); s.append(23)
    # Round 4
    s.append(6); s.append(10); s.append(15); s.append(21)
    s.append(6); s.append(10); s.append(15); s.append(21)
    s.append(6); s.append(10); s.append(15); s.append(21)
    s.append(6); s.append(10); s.append(15); s.append(21)

    # K table: K[i] = floor(abs(sin(i+1)) * 2^32), i = 0..63 (RFC 1321 constants).
    var K = List[UInt32]()
    K.append(0xd76aa478); K.append(0xe8c7b756); K.append(0x242070db); K.append(0xc1bdceee)
    K.append(0xf57c0faf); K.append(0x4787c62a); K.append(0xa8304613); K.append(0xfd469501)
    K.append(0x698098d8); K.append(0x8b44f7af); K.append(0xffff5bb1); K.append(0x895cd7be)
    K.append(0x6b901122); K.append(0xfd987193); K.append(0xa679438e); K.append(0x49b40821)
    K.append(0xf61e2562); K.append(0xc040b340); K.append(0x265e5a51); K.append(0xe9b6c7aa)
    K.append(0xd62f105d); K.append(0x02441453); K.append(0xd8a1e681); K.append(0xe7d3fbc8)
    K.append(0x21e1cde6); K.append(0xc33707d6); K.append(0xf4d50d87); K.append(0x455a14ed)
    K.append(0xa9e3e905); K.append(0xfcefa3f8); K.append(0x676f02d9); K.append(0x8d2a4c8a)
    K.append(0xfffa3942); K.append(0x8771f681); K.append(0x6d9d6122); K.append(0xfde5380c)
    K.append(0xa4beea44); K.append(0x4bdecfa9); K.append(0xf6bb4b60); K.append(0xbebfbc70)
    K.append(0x289b7ec6); K.append(0xeaa127fa); K.append(0xd4ef3085); K.append(0x04881d05)
    K.append(0xd9d4d039); K.append(0xe6db99e5); K.append(0x1fa27cf8); K.append(0xc4ac5665)
    K.append(0xf4292244); K.append(0x432aff97); K.append(0xab9423a7); K.append(0xfc93a039)
    K.append(0x655b59c3); K.append(0x8f0ccc92); K.append(0xffeff47d); K.append(0x85845dd1)
    K.append(0x6fa87e4f); K.append(0xfe2ce6e0); K.append(0xa3014314); K.append(0x4e0811a1)
    K.append(0xf7537e82); K.append(0xbd3af235); K.append(0x2ad7d2bb); K.append(0xeb86d391)

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

    # Append the 64-bit length, little-endian.
    for i in range(8):
        msg.append(UInt8((bit_len >> (UInt64(8) * UInt64(i))) & UInt64(0xFF)))

    # --- Initial state (little-endian magic constants) ---
    var a0 = UInt32(0x67452301)
    var b0 = UInt32(0xefcdab89)
    var c0 = UInt32(0x98badcfe)
    var d0 = UInt32(0x10325476)

    var total = len(msg)
    var chunk = 0
    while chunk < total:
        # Decode 16 little-endian 32-bit words from this 64-byte chunk.
        var M = List[UInt32]()
        for w in range(16):
            var base = chunk + w * 4
            var word = UInt32(msg[base]) \
                | (UInt32(msg[base + 1]) << 8) \
                | (UInt32(msg[base + 2]) << 16) \
                | (UInt32(msg[base + 3]) << 24)
            M.append(word & 0xFFFFFFFF)

        var A = a0
        var B = b0
        var C = c0
        var D = d0

        for i in range(64):
            var F: UInt32
            var g: Int
            if i < 16:
                F = (B & C) | ((~B) & D)
                g = i
            elif i < 32:
                F = (D & B) | ((~D) & C)
                g = (5 * i + 1) % 16
            elif i < 48:
                F = B ^ C ^ D
                g = (3 * i + 5) % 16
            else:
                F = C ^ (B | (~D))
                g = (7 * i) % 16

            F = (F + A + K[i] + M[g]) & 0xFFFFFFFF
            A = D
            D = C
            C = B
            B = (B + _rotl32(F, s[i])) & 0xFFFFFFFF

        a0 = (a0 + A) & 0xFFFFFFFF
        b0 = (b0 + B) & 0xFFFFFFFF
        c0 = (c0 + C) & 0xFFFFFFFF
        d0 = (d0 + D) & 0xFFFFFFFF

        chunk = chunk + 64

    # Output digest: a0,b0,c0,d0 each little-endian.
    var out = List[UInt8]()
    var words = List[UInt32]()
    words.append(a0); words.append(b0); words.append(c0); words.append(d0)
    for wi in range(4):
        var v = words[wi]
        out.append(UInt8(v & 0xFF))
        out.append(UInt8((v >> 8) & 0xFF))
        out.append(UInt8((v >> 16) & 0xFF))
        out.append(UInt8((v >> 24) & 0xFF))
    return out^


def md5_hex(data: List[UInt8]) raises -> String:
    # Lowercase hex of the 16-byte digest.
    comptime HEX = "0123456789abcdef"
    var hexb = HEX.as_bytes()
    var digest = md5(data)
    var s = String("")
    for i in range(len(digest)):
        var byte = Int(digest[i])
        s += chr(Int(hexb[(byte >> 4) & 0xF]))
        s += chr(Int(hexb[byte & 0xF]))
    return s^
