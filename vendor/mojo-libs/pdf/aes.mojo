# pdf/aes.mojo
# Pure-Mojo AES-128 / AES-256 (FIPS-197) with CBC mode + PKCS#7 padding.
# These are the symmetric primitives for modern PDF (AESV2 = AES-128-CBC,
# AESV3 = AES-256-CBC) encryption.
#
# All byte arithmetic is over GF(2^8) with the AES reduction polynomial
# 0x11B. The S-box and Rcon are the fixed FIPS-197 tables, hardcoded below.
# Key length (16 or 32 bytes) selects AES-128 (10 rounds) or AES-256 (14).


def _sbox() raises -> List[UInt8]:
    # FIPS-197 forward S-box (256 entries).
    var s = List[UInt8]()
    s.append(0x63); s.append(0x7c); s.append(0x77); s.append(0x7b); s.append(0xf2); s.append(0x6b); s.append(0x6f); s.append(0xc5)
    s.append(0x30); s.append(0x01); s.append(0x67); s.append(0x2b); s.append(0xfe); s.append(0xd7); s.append(0xab); s.append(0x76)
    s.append(0xca); s.append(0x82); s.append(0xc9); s.append(0x7d); s.append(0xfa); s.append(0x59); s.append(0x47); s.append(0xf0)
    s.append(0xad); s.append(0xd4); s.append(0xa2); s.append(0xaf); s.append(0x9c); s.append(0xa4); s.append(0x72); s.append(0xc0)
    s.append(0xb7); s.append(0xfd); s.append(0x93); s.append(0x26); s.append(0x36); s.append(0x3f); s.append(0xf7); s.append(0xcc)
    s.append(0x34); s.append(0xa5); s.append(0xe5); s.append(0xf1); s.append(0x71); s.append(0xd8); s.append(0x31); s.append(0x15)
    s.append(0x04); s.append(0xc7); s.append(0x23); s.append(0xc3); s.append(0x18); s.append(0x96); s.append(0x05); s.append(0x9a)
    s.append(0x07); s.append(0x12); s.append(0x80); s.append(0xe2); s.append(0xeb); s.append(0x27); s.append(0xb2); s.append(0x75)
    s.append(0x09); s.append(0x83); s.append(0x2c); s.append(0x1a); s.append(0x1b); s.append(0x6e); s.append(0x5a); s.append(0xa0)
    s.append(0x52); s.append(0x3b); s.append(0xd6); s.append(0xb3); s.append(0x29); s.append(0xe3); s.append(0x2f); s.append(0x84)
    s.append(0x53); s.append(0xd1); s.append(0x00); s.append(0xed); s.append(0x20); s.append(0xfc); s.append(0xb1); s.append(0x5b)
    s.append(0x6a); s.append(0xcb); s.append(0xbe); s.append(0x39); s.append(0x4a); s.append(0x4c); s.append(0x58); s.append(0xcf)
    s.append(0xd0); s.append(0xef); s.append(0xaa); s.append(0xfb); s.append(0x43); s.append(0x4d); s.append(0x33); s.append(0x85)
    s.append(0x45); s.append(0xf9); s.append(0x02); s.append(0x7f); s.append(0x50); s.append(0x3c); s.append(0x9f); s.append(0xa8)
    s.append(0x51); s.append(0xa3); s.append(0x40); s.append(0x8f); s.append(0x92); s.append(0x9d); s.append(0x38); s.append(0xf5)
    s.append(0xbc); s.append(0xb6); s.append(0xda); s.append(0x21); s.append(0x10); s.append(0xff); s.append(0xf3); s.append(0xd2)
    s.append(0xcd); s.append(0x0c); s.append(0x13); s.append(0xec); s.append(0x5f); s.append(0x97); s.append(0x44); s.append(0x17)
    s.append(0xc4); s.append(0xa7); s.append(0x7e); s.append(0x3d); s.append(0x64); s.append(0x5d); s.append(0x19); s.append(0x73)
    s.append(0x60); s.append(0x81); s.append(0x4f); s.append(0xdc); s.append(0x22); s.append(0x2a); s.append(0x90); s.append(0x88)
    s.append(0x46); s.append(0xee); s.append(0xb8); s.append(0x14); s.append(0xde); s.append(0x5e); s.append(0x0b); s.append(0xdb)
    s.append(0xe0); s.append(0x32); s.append(0x3a); s.append(0x0a); s.append(0x49); s.append(0x06); s.append(0x24); s.append(0x5c)
    s.append(0xc2); s.append(0xd3); s.append(0xac); s.append(0x62); s.append(0x91); s.append(0x95); s.append(0xe4); s.append(0x79)
    s.append(0xe7); s.append(0xc8); s.append(0x37); s.append(0x6d); s.append(0x8d); s.append(0xd5); s.append(0x4e); s.append(0xa9)
    s.append(0x6c); s.append(0x56); s.append(0xf4); s.append(0xea); s.append(0x65); s.append(0x7a); s.append(0xae); s.append(0x08)
    s.append(0xba); s.append(0x78); s.append(0x25); s.append(0x2e); s.append(0x1c); s.append(0xa6); s.append(0xb4); s.append(0xc6)
    s.append(0xe8); s.append(0xdd); s.append(0x74); s.append(0x1f); s.append(0x4b); s.append(0xbd); s.append(0x8b); s.append(0x8a)
    s.append(0x70); s.append(0x3e); s.append(0xb5); s.append(0x66); s.append(0x48); s.append(0x03); s.append(0xf6); s.append(0x0e)
    s.append(0x61); s.append(0x35); s.append(0x57); s.append(0xb9); s.append(0x86); s.append(0xc1); s.append(0x1d); s.append(0x9e)
    s.append(0xe1); s.append(0xf8); s.append(0x98); s.append(0x11); s.append(0x69); s.append(0xd9); s.append(0x8e); s.append(0x94)
    s.append(0x9b); s.append(0x1e); s.append(0x87); s.append(0xe9); s.append(0xce); s.append(0x55); s.append(0x28); s.append(0xdf)
    s.append(0x8c); s.append(0xa1); s.append(0x89); s.append(0x0d); s.append(0xbf); s.append(0xe6); s.append(0x42); s.append(0x68)
    s.append(0x41); s.append(0x99); s.append(0x2d); s.append(0x0f); s.append(0xb0); s.append(0x54); s.append(0xbb); s.append(0x16)
    return s^


def _inv_sbox(sbox: List[UInt8]) raises -> List[UInt8]:
    # Inverse S-box derived from the forward table.
    var inv = List[UInt8]()
    for _ in range(256):
        inv.append(UInt8(0))
    for i in range(256):
        inv[Int(sbox[i])] = UInt8(i)
    return inv^


def _xtime(b: UInt8) -> UInt8:
    # Multiply by x (i.e. 0x02) in GF(2^8) with reduction poly 0x11B.
    var v = Int(b)
    var shifted = (v << 1) & 0xFF
    if (v & 0x80) != 0:
        shifted = shifted ^ 0x1B
    return UInt8(shifted & 0xFF)


def _gmul(a: UInt8, b: UInt8) -> UInt8:
    # General GF(2^8) multiply.
    var p = 0
    var aa = Int(a)
    var bb = Int(b)
    for _ in range(8):
        if (bb & 1) != 0:
            p = p ^ aa
        var hi = aa & 0x80
        aa = (aa << 1) & 0xFF
        if hi != 0:
            aa = aa ^ 0x1B
        bb = bb >> 1
    return UInt8(p & 0xFF)


def _key_expansion(key: List[UInt8], sbox: List[UInt8]) raises -> List[UInt8]:
    # Expand the cipher key into round keys.
    # Nk = key words (4 or 8); Nr rounds (10 or 14); output = 16*(Nr+1) bytes.
    var Nk = len(key) // 4
    var Nr: Int
    if Nk == 4:
        Nr = 10
    else:
        Nr = 14
    var total_words = 4 * (Nr + 1)

    # Rcon (round constants), only the high byte is used.
    var rcon = List[UInt8]()
    rcon.append(0x01); rcon.append(0x02); rcon.append(0x04); rcon.append(0x08)
    rcon.append(0x10); rcon.append(0x20); rcon.append(0x40); rcon.append(0x80)
    rcon.append(0x1b); rcon.append(0x36); rcon.append(0x6c); rcon.append(0xd8)
    rcon.append(0xab); rcon.append(0x4d)

    # w[] stored as a flat byte list, 4 bytes per word.
    var w = List[UInt8]()
    for i in range(len(key)):
        w.append(key[i])

    var wi = Nk
    while wi < total_words:
        # temp = previous word.
        var t0 = w[(wi - 1) * 4 + 0]
        var t1 = w[(wi - 1) * 4 + 1]
        var t2 = w[(wi - 1) * 4 + 2]
        var t3 = w[(wi - 1) * 4 + 3]

        if (wi % Nk) == 0:
            # RotWord
            var r0 = t1
            var r1 = t2
            var r2 = t3
            var r3 = t0
            # SubWord
            t0 = sbox[Int(r0)]
            t1 = sbox[Int(r1)]
            t2 = sbox[Int(r2)]
            t3 = sbox[Int(r3)]
            # XOR Rcon (high byte only).
            t0 = t0 ^ rcon[(wi // Nk) - 1]
        elif Nk > 6 and (wi % Nk) == 4:
            # AES-256 extra SubWord step.
            t0 = sbox[Int(t0)]
            t1 = sbox[Int(t1)]
            t2 = sbox[Int(t2)]
            t3 = sbox[Int(t3)]

        # w[wi] = w[wi-Nk] XOR temp.
        w.append(w[(wi - Nk) * 4 + 0] ^ t0)
        w.append(w[(wi - Nk) * 4 + 1] ^ t1)
        w.append(w[(wi - Nk) * 4 + 2] ^ t2)
        w.append(w[(wi - Nk) * 4 + 3] ^ t3)
        wi += 1

    return w^


def _add_round_key(mut state: List[UInt8], rk: List[UInt8], rk_off: Int):
    for i in range(16):
        state[i] = state[i] ^ rk[rk_off + i]


def _sub_bytes(mut state: List[UInt8], sbox: List[UInt8]):
    for i in range(16):
        state[i] = sbox[Int(state[i])]


def _inv_sub_bytes(mut state: List[UInt8], inv: List[UInt8]):
    for i in range(16):
        state[i] = inv[Int(state[i])]


def _shift_rows(mut state: List[UInt8]):
    # State is column-major: state[r + 4*c]. Row r is cyclically left-shifted r.
    var tmp = state.copy()
    for r in range(1, 4):
        for c in range(4):
            state[r + 4 * c] = tmp[r + 4 * ((c + r) % 4)]


def _inv_shift_rows(mut state: List[UInt8]):
    var tmp = state.copy()
    for r in range(1, 4):
        for c in range(4):
            state[r + 4 * c] = tmp[r + 4 * ((c - r + 4) % 4)]


def _mix_columns(mut state: List[UInt8]):
    for c in range(4):
        var a0 = state[0 + 4 * c]
        var a1 = state[1 + 4 * c]
        var a2 = state[2 + 4 * c]
        var a3 = state[3 + 4 * c]
        state[0 + 4 * c] = _xtime(a0) ^ (_xtime(a1) ^ a1) ^ a2 ^ a3
        state[1 + 4 * c] = a0 ^ _xtime(a1) ^ (_xtime(a2) ^ a2) ^ a3
        state[2 + 4 * c] = a0 ^ a1 ^ _xtime(a2) ^ (_xtime(a3) ^ a3)
        state[3 + 4 * c] = (_xtime(a0) ^ a0) ^ a1 ^ a2 ^ _xtime(a3)


def _inv_mix_columns(mut state: List[UInt8]):
    for c in range(4):
        var a0 = state[0 + 4 * c]
        var a1 = state[1 + 4 * c]
        var a2 = state[2 + 4 * c]
        var a3 = state[3 + 4 * c]
        state[0 + 4 * c] = _gmul(a0, 0x0e) ^ _gmul(a1, 0x0b) ^ _gmul(a2, 0x0d) ^ _gmul(a3, 0x09)
        state[1 + 4 * c] = _gmul(a0, 0x09) ^ _gmul(a1, 0x0e) ^ _gmul(a2, 0x0b) ^ _gmul(a3, 0x0d)
        state[2 + 4 * c] = _gmul(a0, 0x0d) ^ _gmul(a1, 0x09) ^ _gmul(a2, 0x0e) ^ _gmul(a3, 0x0b)
        state[3 + 4 * c] = _gmul(a0, 0x0b) ^ _gmul(a1, 0x0d) ^ _gmul(a2, 0x09) ^ _gmul(a3, 0x0e)


def aes_encrypt_block(round_keys: List[UInt8], block16: List[UInt8]) raises -> List[UInt8]:
    # Encrypt one 16-byte block with the expanded round keys.
    var sbox = _sbox()
    var Nr = (len(round_keys) // 16) - 1

    # Load state column-major from the 16 input bytes.
    var state = List[UInt8]()
    for i in range(16):
        state.append(block16[i])

    _add_round_key(state, round_keys, 0)
    for rnd in range(1, Nr):
        _sub_bytes(state, sbox)
        _shift_rows(state)
        _mix_columns(state)
        _add_round_key(state, round_keys, 16 * rnd)
    # Final round (no MixColumns).
    _sub_bytes(state, sbox)
    _shift_rows(state)
    _add_round_key(state, round_keys, 16 * Nr)

    var out = List[UInt8]()
    for i in range(16):
        out.append(state[i])
    return out^


def aes_decrypt_block(round_keys: List[UInt8], block16: List[UInt8]) raises -> List[UInt8]:
    # Decrypt one 16-byte block with the expanded round keys.
    var sbox = _sbox()
    var inv = _inv_sbox(sbox)
    var Nr = (len(round_keys) // 16) - 1

    var state = List[UInt8]()
    for i in range(16):
        state.append(block16[i])

    _add_round_key(state, round_keys, 16 * Nr)
    var rnd = Nr - 1
    while rnd >= 1:
        _inv_shift_rows(state)
        _inv_sub_bytes(state, inv)
        _add_round_key(state, round_keys, 16 * rnd)
        _inv_mix_columns(state)
        rnd -= 1
    # Final (first) round.
    _inv_shift_rows(state)
    _inv_sub_bytes(state, inv)
    _add_round_key(state, round_keys, 0)

    var out = List[UInt8]()
    for i in range(16):
        out.append(state[i])
    return out^


def aes_cbc_encrypt(key: List[UInt8], iv: List[UInt8], data: List[UInt8]) raises -> List[UInt8]:
    # AES-CBC encrypt with PKCS#7 padding. Key length 16 or 32 selects AES-128/256.
    if len(key) != 16 and len(key) != 32:
        raise Error("aes_cbc_encrypt: key must be 16 or 32 bytes")
    if len(iv) != 16:
        raise Error("aes_cbc_encrypt: iv must be 16 bytes")

    var sbox = _sbox()
    var round_keys = _key_expansion(key, sbox)

    # PKCS#7 pad: append N bytes each equal to N, where N in [1,16].
    var padded = List[UInt8]()
    for i in range(len(data)):
        padded.append(data[i])
    var pad = 16 - (len(data) % 16)
    if pad == 0:
        pad = 16
    for _ in range(pad):
        padded.append(UInt8(pad))

    var prev = List[UInt8]()
    for i in range(16):
        prev.append(iv[i])

    var out = List[UInt8]()
    var off = 0
    while off < len(padded):
        var block = List[UInt8]()
        for i in range(16):
            block.append(padded[off + i] ^ prev[i])
        var enc = aes_encrypt_block(round_keys, block)
        for i in range(16):
            out.append(enc[i])
        prev = enc.copy()
        off += 16

    return out^


def aes_cbc_decrypt(key: List[UInt8], iv: List[UInt8], data: List[UInt8]) raises -> List[UInt8]:
    # AES-CBC decrypt and strip PKCS#7 padding. Key length 16 or 32 selects AES-128/256.
    if len(key) != 16 and len(key) != 32:
        raise Error("aes_cbc_decrypt: key must be 16 or 32 bytes")
    if len(iv) != 16:
        raise Error("aes_cbc_decrypt: iv must be 16 bytes")
    if len(data) == 0 or (len(data) % 16) != 0:
        raise Error("aes_cbc_decrypt: ciphertext must be a non-zero multiple of 16 bytes")

    var sbox = _sbox()
    var round_keys = _key_expansion(key, sbox)

    var prev = List[UInt8]()
    for i in range(16):
        prev.append(iv[i])

    var plain = List[UInt8]()
    var off = 0
    while off < len(data):
        var block = List[UInt8]()
        for i in range(16):
            block.append(data[off + i])
        var dec = aes_decrypt_block(round_keys, block)
        for i in range(16):
            plain.append(dec[i] ^ prev[i])
        prev = block.copy()
        off += 16

    # Strip PKCS#7 padding.
    var pad = Int(plain[len(plain) - 1])
    if pad < 1 or pad > 16 or pad > len(plain):
        raise Error("aes_cbc_decrypt: invalid PKCS#7 padding")
    for i in range(pad):
        if Int(plain[len(plain) - 1 - i]) != pad:
            raise Error("aes_cbc_decrypt: corrupt PKCS#7 padding")
    var out = List[UInt8]()
    for i in range(len(plain) - pad):
        out.append(plain[i])
    return out^
