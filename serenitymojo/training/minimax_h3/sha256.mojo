# Host-only streaming SHA-256 used by H3 source and cache receipts.
# No MAX model, tensor, or DeviceContext dependency belongs in this module.

from std.collections import List
from std.math import min
from std.memory import alloc

from serenitymojo.io.ffi import (
    BytePtr,
    O_RDONLY,
    file_size,
    sys_close,
    sys_open,
    sys_pread,
)


comptime _SHA_CHUNK = 65536


def _sha_rotr(value: UInt32, amount: UInt32) -> UInt32:
    return (
        (value >> amount) | (value << (UInt32(32) - amount))
    ) & UInt32(0xFFFFFFFF)


def _sha_constants() -> List[UInt32]:
    return [
        UInt32(0x428a2f98), UInt32(0x71374491), UInt32(0xb5c0fbcf), UInt32(0xe9b5dba5),
        UInt32(0x3956c25b), UInt32(0x59f111f1), UInt32(0x923f82a4), UInt32(0xab1c5ed5),
        UInt32(0xd807aa98), UInt32(0x12835b01), UInt32(0x243185be), UInt32(0x550c7dc3),
        UInt32(0x72be5d74), UInt32(0x80deb1fe), UInt32(0x9bdc06a7), UInt32(0xc19bf174),
        UInt32(0xe49b69c1), UInt32(0xefbe4786), UInt32(0x0fc19dc6), UInt32(0x240ca1cc),
        UInt32(0x2de92c6f), UInt32(0x4a7484aa), UInt32(0x5cb0a9dc), UInt32(0x76f988da),
        UInt32(0x983e5152), UInt32(0xa831c66d), UInt32(0xb00327c8), UInt32(0xbf597fc7),
        UInt32(0xc6e00bf3), UInt32(0xd5a79147), UInt32(0x06ca6351), UInt32(0x14292967),
        UInt32(0x27b70a85), UInt32(0x2e1b2138), UInt32(0x4d2c6dfc), UInt32(0x53380d13),
        UInt32(0x650a7354), UInt32(0x766a0abb), UInt32(0x81c2c92e), UInt32(0x92722c85),
        UInt32(0xa2bfe8a1), UInt32(0xa81a664b), UInt32(0xc24b8b70), UInt32(0xc76c51a3),
        UInt32(0xd192e819), UInt32(0xd6990624), UInt32(0xf40e3585), UInt32(0x106aa070),
        UInt32(0x19a4c116), UInt32(0x1e376c08), UInt32(0x2748774c), UInt32(0x34b0bcb5),
        UInt32(0x391c0cb3), UInt32(0x4ed8aa4a), UInt32(0x5b9cca4f), UInt32(0x682e6ff3),
        UInt32(0x748f82ee), UInt32(0x78a5636f), UInt32(0x84c87814), UInt32(0x8cc70208),
        UInt32(0x90befffa), UInt32(0xa4506ceb), UInt32(0xbef9a3f7), UInt32(0xc67178f2),
    ]


def _sha_initial() -> List[UInt32]:
    return [
        UInt32(0x6a09e667), UInt32(0xbb67ae85),
        UInt32(0x3c6ef372), UInt32(0xa54ff53a),
        UInt32(0x510e527f), UInt32(0x9b05688c),
        UInt32(0x1f83d9ab), UInt32(0x5be0cd19),
    ]


def _sha_compress(data: List[UInt8], offset: Int, mut state: List[UInt32]):
    var constants = _sha_constants()
    var words = List[UInt32]()
    for index in range(16):
        var base = offset + 4 * index
        words.append(
            (UInt32(data[base]) << UInt32(24))
            | (UInt32(data[base + 1]) << UInt32(16))
            | (UInt32(data[base + 2]) << UInt32(8))
            | UInt32(data[base + 3])
        )
    for index in range(16, 64):
        var first = words[index - 15]
        var second = words[index - 2]
        var sigma0 = _sha_rotr(first, 7) ^ _sha_rotr(first, 18) ^ (first >> UInt32(3))
        var sigma1 = _sha_rotr(second, 17) ^ _sha_rotr(second, 19) ^ (second >> UInt32(10))
        words.append(
            (words[index - 16] + sigma0 + words[index - 7] + sigma1)
            & UInt32(0xFFFFFFFF)
        )
    var a = state[0]
    var b = state[1]
    var c = state[2]
    var d = state[3]
    var e = state[4]
    var f = state[5]
    var g = state[6]
    var h = state[7]
    for index in range(64):
        var sum1 = _sha_rotr(e, 6) ^ _sha_rotr(e, 11) ^ _sha_rotr(e, 25)
        var choose = (e & f) ^ ((~e) & g)
        var temp1 = (h + sum1 + choose + constants[index] + words[index]) & UInt32(0xFFFFFFFF)
        var sum0 = _sha_rotr(a, 2) ^ _sha_rotr(a, 13) ^ _sha_rotr(a, 22)
        var majority = (a & b) ^ (a & c) ^ (b & c)
        var temp2 = (sum0 + majority) & UInt32(0xFFFFFFFF)
        h = g
        g = f
        f = e
        e = (d + temp1) & UInt32(0xFFFFFFFF)
        d = c
        c = b
        b = a
        a = (temp1 + temp2) & UInt32(0xFFFFFFFF)
    state[0] = (state[0] + a) & UInt32(0xFFFFFFFF)
    state[1] = (state[1] + b) & UInt32(0xFFFFFFFF)
    state[2] = (state[2] + c) & UInt32(0xFFFFFFFF)
    state[3] = (state[3] + d) & UInt32(0xFFFFFFFF)
    state[4] = (state[4] + e) & UInt32(0xFFFFFFFF)
    state[5] = (state[5] + f) & UInt32(0xFFFFFFFF)
    state[6] = (state[6] + g) & UInt32(0xFFFFFFFF)
    state[7] = (state[7] + h) & UInt32(0xFFFFFFFF)


def _sha_hex(state: List[UInt32]) -> String:
    comptime HEX = "0123456789abcdef"
    var alphabet = HEX.as_bytes()
    var out = String("")
    for word in state:
        for byte_index in range(4):
            var shift = UInt32(24 - 8 * byte_index)
            var byte = Int((word >> shift) & UInt32(0xFF))
            out += chr(Int(alphabet[(byte >> 4) & 0xF]))
            out += chr(Int(alphabet[byte & 0xF]))
    return out^


def minimax_h3_sha256_bytes(data: List[UInt8]) -> String:
    var state = _sha_initial()
    var padded = data.copy()
    var bit_length = UInt64(len(padded)) * UInt64(8)
    padded.append(UInt8(0x80))
    while len(padded) % 64 != 56:
        padded.append(UInt8(0))
    for index in range(8):
        padded.append(UInt8((bit_length >> UInt64(8 * (7 - index))) & UInt64(0xFF)))
    for block in range(0, len(padded), 64):
        _sha_compress(padded, block, state)
    return String("sha256:") + _sha_hex(state)


def minimax_h3_sha256_text(value: String) -> String:
    """SHA-256 receipt for deterministic UTF-8 fixture/provenance strings."""
    var bytes = List[UInt8]()
    var source = value.as_bytes()
    for index in range(value.byte_length()):
        bytes.append(source[index])
    return minimax_h3_sha256_bytes(bytes^)


def minimax_h3_sha256_file(path: String) raises -> String:
    """Streaming `sha256:<lowercase hex>` with at most ~64 KiB buffered."""
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("MiniMax H3 checksum cannot open ") + path)
    var declared_size = file_size(fd)
    if declared_size < 0:
        _ = sys_close(fd)
        raise Error(String("MiniMax H3 checksum cannot size ") + path)
    var state = _sha_initial()
    var pending = List[UInt8]()
    var buf = alloc[UInt8](_SHA_CHUNK)
    var file_offset = 0
    while file_offset < declared_size:
        var want = min(_SHA_CHUNK, declared_size - file_offset)
        var count = sys_pread(
            fd, BytePtr(unsafe_from_address=Int(buf)), want, file_offset,
        )
        if count <= 0:
            buf.free()
            _ = sys_close(fd)
            raise Error(String("MiniMax H3 checksum short read: ") + path)
        var merged = pending^
        for index in range(count):
            merged.append(buf[index])
        var complete = (len(merged) // 64) * 64
        for block in range(0, complete, 64):
            _sha_compress(merged, block, state)
        pending = List[UInt8]()
        for index in range(complete, len(merged)):
            pending.append(merged[index])
        file_offset += count
    buf.free()
    _ = sys_close(fd)
    if file_offset != declared_size:
        raise Error(String("MiniMax H3 checksum size changed while reading: ") + path)
    var bit_length = UInt64(declared_size) * UInt64(8)
    pending.append(UInt8(0x80))
    while len(pending) % 64 != 56:
        pending.append(UInt8(0))
    for index in range(8):
        pending.append(
            UInt8((bit_length >> UInt64(8 * (7 - index))) & UInt64(0xFF))
        )
    for block in range(0, len(pending), 64):
        _sha_compress(pending, block, state)
    return String("sha256:") + _sha_hex(state)
