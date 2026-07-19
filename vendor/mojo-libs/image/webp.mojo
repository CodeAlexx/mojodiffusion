# image/webp.mojo — a WebP VP8L (lossless) codec, 100% Mojo.
#
# Scope: LOSSLESS VP8L decode + encode. Lossy VP8 is a LATER phase — if a "VP8 "
# (lossy) chunk is encountered we raise a clear error. A "VP8X" extended container
# wrapping a VP8L subchunk is supported by scanning to the VP8L chunk.
#
# Decoder (decode_webp / decode_webp_bytes):
#   - RIFF container: "RIFF" <size LE> "WEBP", then chunk walk. We locate the
#     "VP8L" chunk (directly, or inside a "VP8X" container).
#   - VP8L bitstream (LSB-first within each byte):
#       signature byte 0x2F, 14-bit (width-1), 14-bit (height-1), 1-bit alpha_used,
#       3-bit version.
#   - Transforms (read while transform-present bit set, applied in REVERSE on
#     decode): PREDICTOR, COLOR, SUBTRACT_GREEN, COLOR_INDEXING. All four
#     implemented.
#   - Entropy-coded ARGB image: optional color cache, meta-Huffman (entropy image),
#     5 Huffman codes per meta-group (green+length+cache prefix / red / blue /
#     alpha / distance), LZ77 backward references (length & distance prefix coding,
#     2-D distance mapping), literals, and color-cache lookups.
#   - Canonical Huffman built via the two-level scheme: read code-length-code
#     lengths in the fixed order, build that code, then read symbol code lengths
#     using repeat codes 16/17/18.
#   Output: RGBA Image (channels = 4).
#
# Encoder (encode_webp / encode_webp_bytes):
#   - MINIMAL but spec-valid lossless VP8L: no transforms, no color cache, no
#     LZ77. The full ARGB stream is emitted as literals using per-channel
#     canonical Huffman codes built from the image's own symbol histograms (so it
#     is valid and reasonably compact, and decodable by libwebp/Pillow).
#   - Wrapped in a correct RIFF/VP8L container (sizes little-endian, even padding).
#
# VP8L multi-bit values are LSB-first (low bits read first within each byte) —
# unlike PNG/JPEG. RIFF container integers are little-endian.

from image.buffer import Image
from std.memory import alloc, UnsafePointer


# ─────────────────────────────────────────────────────────────────────────────
# LSB-first bit reader
# ─────────────────────────────────────────────────────────────────────────────
struct BitReader(Movable):
    var data: List[UInt8]
    var pos: Int          # byte position
    var bit_buf: UInt64   # buffered bits (LSB-first)
    var bit_cnt: Int      # number of valid bits in bit_buf

    def __init__(out self, var data: List[UInt8]):
        self.data = data^
        self.pos = 0
        self.bit_buf = 0
        self.bit_cnt = 0

    @always_inline
    def _fill(mut self):
        # Pull bytes into the buffer until we have >= 56 bits (or run out).
        while self.bit_cnt <= 56 and self.pos < len(self.data):
            self.bit_buf |= UInt64(Int(self.data[self.pos])) << UInt64(self.bit_cnt)
            self.bit_cnt += 8
            self.pos += 1

    def read_bits(mut self, n: Int) raises -> Int:
        """Read n bits LSB-first. Returns them as an Int (low bit first)."""
        if n == 0:
            return 0
        if n > 32:
            raise Error("webp: read_bits n>32")
        if self.bit_cnt < n:
            self._fill()
        if self.bit_cnt < n:
            # Out of data: pad with zeros (libwebp tolerates trailing reads).
            # Return what we have, treating missing as 0.
            var have = self.bit_cnt
            var v = Int(self.bit_buf & ((UInt64(1) << UInt64(have)) - UInt64(1))) if have > 0 else 0
            self.bit_buf = 0
            self.bit_cnt = 0
            # remaining (n-have) bits are zero
            return v
        var mask = (UInt64(1) << UInt64(n)) - UInt64(1)
        var v = Int(self.bit_buf & mask)
        self.bit_buf = self.bit_buf >> UInt64(n)
        self.bit_cnt -= n
        return v

    @always_inline
    def read_one(mut self) raises -> Int:
        return self.read_bits(1)


# ─────────────────────────────────────────────────────────────────────────────
# Canonical Huffman decoder (LSB-first, like libwebp's HuffmanTree but simple).
#
# We build a per-length code table and decode bit-by-bit. Codes are assigned in
# canonical order: shorter lengths first, then by symbol order. Bits are read
# LSB-first but Huffman codes in VP8L are read MSB-first into the code value
# (libwebp reads one bit at a time and shifts the accumulator left). We replicate
# that: accumulate code = (code<<1) | bit, length grows, look up.
# ─────────────────────────────────────────────────────────────────────────────
struct HuffTree(Copyable, Movable):
    # For decoding we store, for each length L (1..15), the first canonical code
    # at that length and the symbol index offset, plus a sorted symbol array.
    var counts: List[Int]        # counts[L] = number of codes of length L (L in 0..15)
    var symbols: List[Int]       # symbols sorted by (length, symbol)
    var num_symbols: Int
    var single_symbol: Int       # if only one symbol total, its value (>=0); else -1

    def __init__(out self, lengths: List[Int]) raises:
        # lengths[s] = code length of symbol s (0 = unused)
        var n = len(lengths)
        var counts = List[Int]()
        for _ in range(16):
            counts.append(0)
        var used = 0
        var last_sym = -1
        for s in range(n):
            var L = lengths[s]
            if L > 0:
                if L > 15:
                    raise Error("webp: huffman code length >15")
                counts[L] += 1
                used += 1
                last_sym = s
        self.counts = counts^
        self.num_symbols = used
        self.single_symbol = -1
        if used == 1:
            self.single_symbol = last_sym
        # Build the sorted symbol list (offsets per length).
        var offsets = List[Int]()
        for _ in range(16):
            offsets.append(0)
        for L in range(1, 16):
            offsets[L] = offsets[L - 1] + self.counts[L - 1]
        var symbols = List[Int]()
        for _ in range(used):
            symbols.append(0)
        # second pass: place symbols
        var next_off = offsets.copy()
        for s in range(n):
            var L = lengths[s]
            if L > 0:
                symbols[next_off[L]] = s
                next_off[L] += 1
        self.symbols = symbols^

    def decode(self, mut br: BitReader) raises -> Int:
        if self.single_symbol >= 0:
            # Zero-length code: emits the single symbol without consuming bits.
            return self.single_symbol
        var code = 0
        var first = 0       # first canonical code of current length
        var index = 0       # index into symbols of first code of current length
        for L in range(1, 16):
            code |= br.read_one()
            var count = self.counts[L]
            if code - first < count:
                return self.symbols[index + (code - first)]
            index += count
            first = (first + count) << 1
            code = code << 1
        raise Error("webp: invalid huffman code")


# ─────────────────────────────────────────────────────────────────────────────
# Code-length-code order (the fixed order in which the 19 code-length code
# lengths are transmitted).
# ─────────────────────────────────────────────────────────────────────────────
def _code_length_code_order() -> List[Int]:
    return [17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


# Read a single Huffman code's symbol lengths and build the tree.
# alphabet_size: number of symbols in this alphabet.
def _read_huffman_tree(
    mut br: BitReader, alphabet_size: Int
) raises -> HuffTree:
    var simple = br.read_one()
    if simple == 1:
        # Simple code length code: 1 or 2 symbols.
        var num_symbols = br.read_one() + 1
        var lengths = List[Int]()
        for _ in range(alphabet_size):
            lengths.append(0)
        var first_is_8 = br.read_one()  # is the first symbol 8-bit?
        var sym0 = 0
        if first_is_8 == 1:
            sym0 = br.read_bits(8)
        else:
            sym0 = br.read_bits(1)
        if sym0 < alphabet_size:
            lengths[sym0] = 1
        if num_symbols == 2:
            var sym1 = br.read_bits(8)
            if sym1 < alphabet_size:
                lengths[sym1] = 1
        return HuffTree(lengths)
    else:
        # Normal code length code.
        var num_code_lengths = br.read_bits(4) + 4
        var order = _code_length_code_order()
        var clc_lengths = List[Int]()
        for _ in range(19):
            clc_lengths.append(0)
        for i in range(num_code_lengths):
            clc_lengths[order[i]] = br.read_bits(3)
        var clc_tree = HuffTree(clc_lengths)

        # Now read the actual symbol code lengths using the clc_tree.
        var lengths = List[Int]()
        for _ in range(alphabet_size):
            lengths.append(0)

        var max_symbol = alphabet_size
        var use_length = br.read_one()
        if use_length == 1:
            var length_nbits = 2 + 2 * br.read_bits(3)
            max_symbol = 2 + br.read_bits(length_nbits)

        var symbol = 0
        var prev_code_len = 8
        while symbol < alphabet_size:
            if max_symbol <= 0:
                break
            max_symbol -= 1
            var code_len = clc_tree.decode(br)
            if code_len < 16:
                lengths[symbol] = code_len
                symbol += 1
                if code_len != 0:
                    prev_code_len = code_len
            else:
                var repeat = 0
                var rep_value = 0
                if code_len == 16:
                    repeat = 3 + br.read_bits(2)
                    rep_value = prev_code_len
                elif code_len == 17:
                    repeat = 3 + br.read_bits(3)
                    rep_value = 0
                else:  # 18
                    repeat = 11 + br.read_bits(7)
                    rep_value = 0
                if symbol + repeat > alphabet_size:
                    raise Error("webp: huffman repeat overflow")
                for _ in range(repeat):
                    lengths[symbol] = rep_value
                    symbol += 1
        return HuffTree(lengths)


# ─────────────────────────────────────────────────────────────────────────────
# Distance / length prefix coding.
# ─────────────────────────────────────────────────────────────────────────────
@always_inline
def _get_extra(mut br: BitReader, prefix_code: Int) raises -> Int:
    """Given a prefix code symbol (for length or distance), return the actual
    value. (prefix coding: codes 0..3 map to 1..4; higher codes carry extra
    bits.)"""
    if prefix_code < 4:
        return prefix_code + 1
    var extra_bits = (prefix_code - 2) >> 1
    var offset = (2 + (prefix_code & 1)) << extra_bits
    return offset + br.read_bits(extra_bits) + 1


# Distance mapping: codes 1..120 map to 2-D plane offsets (closer neighbours).
def _distance_map() -> List[Int]:
    # Each entry is (dx + dy*xsize_base) encoded as the libwebp table where
    # dy in high nibble offset, dx (signed) in low. We store as packed:
    #   value = (yoffset << 4) | (xoffset + 8)? -> simpler: store dx in -7..8 and dy.
    # libwebp's kCodeToPlane (120 entries):
    return [
        0x18, 0x07, 0x17, 0x19, 0x28, 0x06, 0x27, 0x29, 0x16, 0x1a,
        0x26, 0x2a, 0x38, 0x05, 0x37, 0x39, 0x15, 0x1b, 0x36, 0x3a,
        0x25, 0x2b, 0x48, 0x04, 0x47, 0x49, 0x14, 0x1c, 0x35, 0x3b,
        0x46, 0x4a, 0x24, 0x2c, 0x58, 0x45, 0x4b, 0x34, 0x3c, 0x03,
        0x57, 0x59, 0x13, 0x1d, 0x56, 0x5a, 0x23, 0x2d, 0x44, 0x4c,
        0x55, 0x5b, 0x33, 0x3d, 0x68, 0x02, 0x67, 0x69, 0x12, 0x1e,
        0x66, 0x6a, 0x22, 0x2e, 0x54, 0x5c, 0x43, 0x4d, 0x65, 0x6b,
        0x32, 0x3e, 0x78, 0x01, 0x77, 0x79, 0x53, 0x5d, 0x11, 0x1f,
        0x64, 0x6c, 0x42, 0x4e, 0x76, 0x7a, 0x21, 0x2f, 0x75, 0x7b,
        0x31, 0x3f, 0x63, 0x6d, 0x52, 0x5e, 0x00, 0x74, 0x7c, 0x41,
        0x4f, 0x10, 0x20, 0x62, 0x6e, 0x30, 0x73, 0x7d, 0x51, 0x5f,
        0x40, 0x72, 0x7e, 0x61, 0x6f, 0x50, 0x71, 0x7f, 0x60, 0x70,
    ]


def _plane_to_distance(xsize: Int, dist_code: Int) raises -> Int:
    """Convert a transmitted distance value to a real pixel distance.
    dist_code is the value AFTER prefix decoding (1-based)."""
    if dist_code > 120:
        return dist_code - 120
    var table = _distance_map()
    var packed = table[dist_code - 1]
    var xoffset = 8 - (packed & 0x0F)
    var yoffset = packed >> 4
    var dist = yoffset * xsize + xoffset
    if dist < 1:
        dist = 1
    return dist


# ─────────────────────────────────────────────────────────────────────────────
# Color cache
# ─────────────────────────────────────────────────────────────────────────────
@always_inline
def _cache_hash(argb: UInt32, bits: Int) -> Int:
    return Int((argb * UInt32(0x1e35a7bd)) >> UInt32(32 - bits))


# ─────────────────────────────────────────────────────────────────────────────
# Huffman group: 5 trees (green, red, blue, alpha, distance).
# ─────────────────────────────────────────────────────────────────────────────
struct HuffGroup(Copyable, Movable):
    var green: HuffTree
    var red: HuffTree
    var blue: HuffTree
    var alpha: HuffTree
    var dist: HuffTree

    def __init__(
        out self,
        var green: HuffTree,
        var red: HuffTree,
        var blue: HuffTree,
        var alpha: HuffTree,
        var dist: HuffTree,
    ):
        self.green = green^
        self.red = red^
        self.blue = blue^
        self.alpha = alpha^
        self.dist = dist^


comptime NUM_LITERAL = 256
comptime NUM_LENGTH = 24
comptime NUM_DISTANCE = 40
comptime CODE_LENGTH_CODES = 19


# ─────────────────────────────────────────────────────────────────────────────
# Decode the entropy-coded ARGB image into a flat List[UInt32] (length w*h).
# This is recursive-capable: meta-Huffman uses an entropy image (the "Huffman
# image") at a reduced resolution mapping each block to a Huffman group index.
# ─────────────────────────────────────────────────────────────────────────────
def _decode_image(
    mut br: BitReader,
    xsize: Int,
    ysize: Int,
    allow_meta: Bool,
) raises -> List[UInt32]:
    # Color cache.
    var color_cache_bits = 0
    var has_cache = br.read_one()
    if has_cache == 1:
        color_cache_bits = br.read_bits(4)
        if color_cache_bits < 1 or color_cache_bits > 11:
            raise Error("webp: invalid color cache bits")
    var cache_size = 0
    if color_cache_bits > 0:
        cache_size = 1 << color_cache_bits
    var color_cache = List[UInt32]()
    for _ in range(cache_size):
        color_cache.append(0)

    # Meta-Huffman (entropy image).
    var huffman_bits = 0
    var huffman_xsize = 1
    var num_huff_groups = 1
    var huffman_image = List[Int]()  # per-block group index
    if allow_meta:
        var use_meta = br.read_one()
        if use_meta == 1:
            huffman_bits = br.read_bits(3) + 2
            huffman_xsize = (xsize + (1 << huffman_bits) - 1) >> huffman_bits
            var huffman_ysize = (ysize + (1 << huffman_bits) - 1) >> huffman_bits
            # The entropy image is itself an ARGB image of size
            # huffman_xsize x huffman_ysize; the group index is in the red/green.
            var entropy = _decode_image(br, huffman_xsize, huffman_ysize, False)
            var max_group = 0
            for i in range(len(entropy)):
                var px = entropy[i]
                # group index = (red << 8) | green  (bytes 16..8)
                var g = Int((px >> UInt32(8)) & UInt32(0xFFFF))
                huffman_image.append(g)
                if g > max_group:
                    max_group = g
            num_huff_groups = max_group + 1

    # Read all Huffman groups (5 trees each).
    var groups = List[HuffGroup]()
    for _ in range(num_huff_groups):
        var green_alpha = NUM_LITERAL + NUM_LENGTH + cache_size
        var green = _read_huffman_tree(br, green_alpha)
        var red = _read_huffman_tree(br, NUM_LITERAL)
        var blue = _read_huffman_tree(br, NUM_LITERAL)
        var alpha = _read_huffman_tree(br, NUM_LITERAL)
        var dist = _read_huffman_tree(br, NUM_DISTANCE)
        groups.append(HuffGroup(green^, red^, blue^, alpha^, dist^))

    # Output buffer.
    var out = List[UInt32]()
    for _ in range(xsize * ysize):
        out.append(0)

    var px = 0
    var total = xsize * ysize
    while px < total:
        var x = px % xsize
        var y = px // xsize
        # Select Huffman group for this pixel.
        var gi = 0
        if num_huff_groups > 1:
            var hx = x >> huffman_bits
            var hy = y >> huffman_bits
            gi = huffman_image[hy * huffman_xsize + hx]
        ref grp = groups[gi]

        var sym = grp.green.decode(br)
        if sym < NUM_LITERAL:
            # Literal pixel: green=sym, then read red, blue, alpha.
            var g = sym
            var r = grp.red.decode(br)
            var b = grp.blue.decode(br)
            var a = grp.alpha.decode(br)
            var argb = (UInt32(a) << UInt32(24)) | (UInt32(r) << UInt32(16)) | (
                UInt32(g) << UInt32(8)
            ) | UInt32(b)
            out[px] = argb
            if cache_size > 0:
                color_cache[_cache_hash(argb, color_cache_bits)] = argb
            px += 1
        elif sym < NUM_LITERAL + NUM_LENGTH:
            # Backward reference (LZ77).
            var length_code = sym - NUM_LITERAL
            var length = _get_extra(br, length_code)
            var dist_sym = grp.dist.decode(br)
            var dist_val = _get_extra(br, dist_sym)
            var dist = _plane_to_distance(xsize, dist_val)
            if dist > px:
                # Can't reference before start; clamp (shouldn't happen on valid).
                raise Error("webp: backref distance too large")
            for _ in range(length):
                if px >= total:
                    break
                var argb = out[px - dist]
                out[px] = argb
                if cache_size > 0:
                    color_cache[_cache_hash(argb, color_cache_bits)] = argb
                px += 1
        else:
            # Color cache reference.
            var idx = sym - NUM_LITERAL - NUM_LENGTH
            if idx >= cache_size:
                raise Error("webp: color cache index out of range")
            out[px] = color_cache[idx]
            px += 1

    return out^


# ─────────────────────────────────────────────────────────────────────────────
# Transforms (applied in REVERSE order on decode).
# Each transform reads its parameters from the bitstream when present.
# ─────────────────────────────────────────────────────────────────────────────
comptime XF_PREDICTOR = 0
comptime XF_COLOR = 1
comptime XF_SUBTRACT_GREEN = 2
comptime XF_COLOR_INDEXING = 3


struct Transform(Copyable, Movable):
    var kind: Int
    var bits: Int                 # block size bits (predictor/color)
    var xsize: Int                # transform image xsize (predictor/color), or palette size
    var ysize: Int
    var data: List[UInt32]        # predictor/color/palette data

    def __init__(out self, kind: Int, bits: Int, xsize: Int, ysize: Int, var data: List[UInt32]):
        self.kind = kind
        self.bits = bits
        self.xsize = xsize
        self.ysize = ysize
        self.data = data^


@always_inline
def _add_byte(a: Int, b: Int) -> Int:
    return (a + b) & 0xFF


@always_inline
def _argb_a(p: UInt32) -> Int:
    return Int((p >> UInt32(24)) & UInt32(0xFF))


@always_inline
def _argb_r(p: UInt32) -> Int:
    return Int((p >> UInt32(16)) & UInt32(0xFF))


@always_inline
def _argb_g(p: UInt32) -> Int:
    return Int((p >> UInt32(8)) & UInt32(0xFF))


@always_inline
def _argb_b(p: UInt32) -> Int:
    return Int(p & UInt32(0xFF))


@always_inline
def _make_argb(a: Int, r: Int, g: Int, b: Int) -> UInt32:
    return (UInt32(a & 0xFF) << UInt32(24)) | (UInt32(r & 0xFF) << UInt32(16)) | (
        UInt32(g & 0xFF) << UInt32(8)
    ) | UInt32(b & 0xFF)


# Predictor helpers (the 14 predictors).
@always_inline
def _avg2(a: Int, b: Int) -> Int:
    return (a + b) // 2


def _clamp_add_subtract_full(a: Int, b: Int, c: Int) -> Int:
    var v = a + b - c
    if v < 0:
        return 0
    if v > 255:
        return 255
    return v


def _clamp_add_subtract_half(a: Int, b: Int) -> Int:
    var v = a + (a - b) // 2
    if v < 0:
        return 0
    if v > 255:
        return 255
    return v


def _predict(mode: Int, left: UInt32, top: UInt32, tl: UInt32, tr: UInt32) -> UInt32:
    if mode == 0:
        return _make_argb(255, 0, 0, 0)  # opaque black (handled specially)
    if mode == 1:
        return left
    if mode == 2:
        return top
    if mode == 3:
        return tr
    if mode == 4:
        return tl
    if mode == 5:
        # Average2(Average2(left, tr), top)
        var a = _avg2(_argb_a(left), _argb_a(tr))
        var r = _avg2(_argb_r(left), _argb_r(tr))
        var g = _avg2(_argb_g(left), _argb_g(tr))
        var b = _avg2(_argb_b(left), _argb_b(tr))
        var a2 = _avg2(a, _argb_a(top))
        var r2 = _avg2(r, _argb_r(top))
        var g2 = _avg2(g, _argb_g(top))
        var b2 = _avg2(b, _argb_b(top))
        return _make_argb(a2, r2, g2, b2)
    if mode == 6:
        return _make_argb(
            _avg2(_argb_a(left), _argb_a(tl)),
            _avg2(_argb_r(left), _argb_r(tl)),
            _avg2(_argb_g(left), _argb_g(tl)),
            _avg2(_argb_b(left), _argb_b(tl)),
        )
    if mode == 7:
        return _make_argb(
            _avg2(_argb_a(left), _argb_a(top)),
            _avg2(_argb_r(left), _argb_r(top)),
            _avg2(_argb_g(left), _argb_g(top)),
            _avg2(_argb_b(left), _argb_b(top)),
        )
    if mode == 8:
        return _make_argb(
            _avg2(_argb_a(tl), _argb_a(top)),
            _avg2(_argb_r(tl), _argb_r(top)),
            _avg2(_argb_g(tl), _argb_g(top)),
            _avg2(_argb_b(tl), _argb_b(top)),
        )
    if mode == 9:
        return _make_argb(
            _avg2(_argb_a(top), _argb_a(tr)),
            _avg2(_argb_r(top), _argb_r(tr)),
            _avg2(_argb_g(top), _argb_g(tr)),
            _avg2(_argb_b(top), _argb_b(tr)),
        )
    if mode == 10:
        # Average2(Average2(left, tl), Average2(top, tr))
        var la = _avg2(_argb_a(left), _argb_a(tl))
        var lr = _avg2(_argb_r(left), _argb_r(tl))
        var lg = _avg2(_argb_g(left), _argb_g(tl))
        var lb = _avg2(_argb_b(left), _argb_b(tl))
        var ta = _avg2(_argb_a(top), _argb_a(tr))
        var tr_ = _avg2(_argb_r(top), _argb_r(tr))
        var tg = _avg2(_argb_g(top), _argb_g(tr))
        var tb = _avg2(_argb_b(top), _argb_b(tr))
        return _make_argb(_avg2(la, ta), _avg2(lr, tr_), _avg2(lg, tg), _avg2(lb, tb))
    if mode == 11:
        # Select: pick left or top based on gradient.
        var pa = _argb_a(left) + _argb_a(top) - _argb_a(tl)
        var pr = _argb_r(left) + _argb_r(top) - _argb_r(tl)
        var pg = _argb_g(left) + _argb_g(top) - _argb_g(tl)
        var pb = _argb_b(left) + _argb_b(top) - _argb_b(tl)
        # |p - top| using L1 over channels vs |p - left|
        var pa_l = _abs(pa - _argb_a(left)) + _abs(pr - _argb_r(left)) + _abs(pg - _argb_g(left)) + _abs(pb - _argb_b(left))
        var pa_t = _abs(pa - _argb_a(top)) + _abs(pr - _argb_r(top)) + _abs(pg - _argb_g(top)) + _abs(pb - _argb_b(top))
        if pa_t <= pa_l:
            return top
        return left
    if mode == 12:
        return _make_argb(
            _clamp_add_subtract_full(_argb_a(left), _argb_a(top), _argb_a(tl)),
            _clamp_add_subtract_full(_argb_r(left), _argb_r(top), _argb_r(tl)),
            _clamp_add_subtract_full(_argb_g(left), _argb_g(top), _argb_g(tl)),
            _clamp_add_subtract_full(_argb_b(left), _argb_b(top), _argb_b(tl)),
        )
    # mode 13: ClampAddSubtractHalf(Average2(left, top), tl)
    var av_a = _avg2(_argb_a(left), _argb_a(top))
    var av_r = _avg2(_argb_r(left), _argb_r(top))
    var av_g = _avg2(_argb_g(left), _argb_g(top))
    var av_b = _avg2(_argb_b(left), _argb_b(top))
    return _make_argb(
        _clamp_add_subtract_half(av_a, _argb_a(tl)),
        _clamp_add_subtract_half(av_r, _argb_r(tl)),
        _clamp_add_subtract_half(av_g, _argb_g(tl)),
        _clamp_add_subtract_half(av_b, _argb_b(tl)),
    )


@always_inline
def _abs(v: Int) -> Int:
    return -v if v < 0 else v


def _apply_predictor(
    mut pixels: List[UInt32], xsize: Int, ysize: Int, xf: Transform
) raises:
    var bits = xf.bits
    var tile_xsize = (xsize + (1 << bits) - 1) >> bits
    for y in range(ysize):
        for x in range(xsize):
            var idx = y * xsize + x
            var mode = 0
            if x == 0 and y == 0:
                # top-left: predictor 0 (opaque black) added
                var pred = _make_argb(255, 0, 0, 0)
                pixels[idx] = _add_pixels(pixels[idx], pred)
                continue
            elif y == 0:
                # first row: predict from left
                pixels[idx] = _add_pixels(pixels[idx], pixels[idx - 1])
                continue
            elif x == 0:
                # first column: predict from top
                pixels[idx] = _add_pixels(pixels[idx], pixels[idx - xsize])
                continue
            # interior: get mode from transform image
            var tx = x >> bits
            var ty = y >> bits
            var tval = xf.data[ty * tile_xsize + tx]
            mode = _argb_g(tval)  # mode stored in green channel
            var left = pixels[idx - 1]
            var top = pixels[idx - xsize]
            var tl = pixels[idx - xsize - 1]
            var tr = pixels[idx - xsize + 1] if x + 1 < xsize else pixels[idx - xsize]
            var pred = _predict(mode, left, top, tl, tr)
            pixels[idx] = _add_pixels(pixels[idx], pred)


@always_inline
def _add_pixels(a: UInt32, b: UInt32) -> UInt32:
    return _make_argb(
        _add_byte(_argb_a(a), _argb_a(b)),
        _add_byte(_argb_r(a), _argb_r(b)),
        _add_byte(_argb_g(a), _argb_g(b)),
        _add_byte(_argb_b(a), _argb_b(b)),
    )


@always_inline
def _color_transform_delta(t: Int, c: Int) -> Int:
    # t is a signed 8-bit multiplier (stored as 0..255), c is signed 8-bit color.
    var ts = t if t < 128 else t - 256
    var cs = c if c < 128 else c - 256
    return (ts * cs) >> 5


def _apply_color(
    mut pixels: List[UInt32], xsize: Int, ysize: Int, xf: Transform
) raises:
    var bits = xf.bits
    var tile_xsize = (xsize + (1 << bits) - 1) >> bits
    for y in range(ysize):
        for x in range(xsize):
            var idx = y * xsize + x
            var tx = x >> bits
            var ty = y >> bits
            var m = xf.data[ty * tile_xsize + tx]
            # color_code is the ARGB; multipliers are (libwebp):
            #   green_to_red  = bits 0..7  = BLUE channel
            #   green_to_blue = bits 8..15 = GREEN channel
            #   red_to_blue   = bits 16..23 = RED channel
            var green_to_red = _argb_b(m)
            var green_to_blue = _argb_g(m)
            var red_to_blue = _argb_r(m)
            var p = pixels[idx]
            var a = _argb_a(p)
            var red = _argb_r(p)
            var green = _argb_g(p)
            var blue = _argb_b(p)
            red = (red + _color_transform_delta(green_to_red, green)) & 0xFF
            blue = (blue + _color_transform_delta(green_to_blue, green)) & 0xFF
            blue = (blue + _color_transform_delta(red_to_blue, red)) & 0xFF
            pixels[idx] = _make_argb(a, red & 0xFF, green, blue & 0xFF)


def _apply_subtract_green(mut pixels: List[UInt32], xsize: Int, ysize: Int) raises:
    for i in range(xsize * ysize):
        var p = pixels[i]
        var g = _argb_g(p)
        var r = (_argb_r(p) + g) & 0xFF
        var b = (_argb_b(p) + g) & 0xFF
        pixels[i] = _make_argb(_argb_a(p), r, g, b)


def _apply_color_indexing(
    mut pixels: List[UInt32], xsize: Int, ysize: Int, xf: Transform
) raises -> List[UInt32]:
    # Palette lookup; xf.data is the palette (size xf.xsize), and pixels are
    # stored as indices in the green channel. If palette small, multiple pixels
    # are bit-packed per byte (in green). xsize here is the PACKED width.
    var palette_size = xf.xsize
    # Determine bundling factor (how many pixels packed into one).
    var width_bits = 0
    if palette_size <= 2:
        width_bits = 3
    elif palette_size <= 4:
        width_bits = 2
    elif palette_size <= 16:
        width_bits = 1
    else:
        width_bits = 0
    var bundle = 1 << width_bits
    var out_xsize = xsize  # already the real width (caller passes real)
    # Build the palette as cumulative? No: VP8L stores palette delta-coded; the
    # decoder must reconstruct via prefix sum. We do this in build step instead.
    var out = List[UInt32]()
    for _ in range(out_xsize * ysize):
        out.append(0)
    if bundle == 1:
        for i in range(xsize * ysize):
            var idx = _argb_g(pixels[i])
            if idx >= palette_size:
                idx = 0
            out[i] = xf.data[idx]
        return out^
    else:
        # Packed: each source pixel's green holds `bundle` indices.
        var packed_xsize = (xsize + bundle - 1) // bundle
        for y in range(ysize):
            for x in range(xsize):
                var bx = x // bundle
                var sub = x % bundle
                var packed = _argb_g(pixels[y * packed_xsize + bx])
                var bits_per = width_bits if False else (8 // bundle)
                # indices packed low-to-high
                var idx = (packed >> (sub * bits_per)) & ((1 << bits_per) - 1)
                if idx >= palette_size:
                    idx = 0
                out[y * out_xsize + x] = xf.data[idx]
        return out^


# ─────────────────────────────────────────────────────────────────────────────
# Read the transforms list, returning the (possibly reduced) working xsize.
# ─────────────────────────────────────────────────────────────────────────────
def _read_transforms(
    mut br: BitReader, mut xsize: Int, ysize: Int, mut transforms: List[Transform]
) raises:
    var seen = List[Int]()
    for _ in range(4):
        seen.append(0)
    while br.read_one() == 1:
        var kind = br.read_bits(2)
        if seen[kind] == 1:
            raise Error("webp: transform repeated")
        seen[kind] = 1
        if kind == XF_PREDICTOR or kind == XF_COLOR:
            var size_bits = br.read_bits(3) + 2
            var bw = (xsize + (1 << size_bits) - 1) >> size_bits
            var bh = (ysize + (1 << size_bits) - 1) >> size_bits
            var tdata = _decode_image(br, bw, bh, False)
            transforms.append(Transform(kind, size_bits, bw, bh, tdata^))
        elif kind == XF_SUBTRACT_GREEN:
            transforms.append(Transform(kind, 0, 0, 0, List[UInt32]()))
        else:  # XF_COLOR_INDEXING
            var num_colors = br.read_bits(8) + 1
            var palette_raw = _decode_image(br, num_colors, 1, False)
            # Palette is delta-coded: prefix-sum across entries.
            var palette = List[UInt32]()
            var prev = UInt32(0)
            for i in range(num_colors):
                var cur = _add_pixels(prev, palette_raw[i])
                palette.append(cur)
                prev = cur
            transforms.append(Transform(kind, 0, num_colors, 1, palette^))
            # Color indexing reduces the working width if palette is small.
            var width_bits = 0
            if num_colors <= 2:
                width_bits = 3
            elif num_colors <= 4:
                width_bits = 2
            elif num_colors <= 16:
                width_bits = 1
            if width_bits > 0:
                var bundle = 1 << width_bits
                xsize = (xsize + bundle - 1) // bundle


# ─────────────────────────────────────────────────────────────────────────────
# VP8L bitstream decode -> List[UInt32] ARGB (real width x height).
# ─────────────────────────────────────────────────────────────────────────────
def _decode_vp8l(var data: List[UInt8]) raises -> Image:
    var br = BitReader(data^)
    var sig = br.read_bits(8)
    if sig != 0x2F:
        raise Error("webp: bad VP8L signature 0x" + String(sig))
    var width = br.read_bits(14) + 1
    var height = br.read_bits(14) + 1
    var _alpha_used = br.read_one()
    var version = br.read_bits(3)
    if version != 0:
        raise Error("webp: unsupported VP8L version " + String(version))

    # Read transforms (modifies working xsize for color indexing).
    var work_xsize = width
    var transforms = List[Transform]()
    _read_transforms(br, work_xsize, height, transforms)

    # Decode the entropy-coded image at the (possibly reduced) working width.
    var pixels = _decode_image(br, work_xsize, height, True)

    # Apply transforms in REVERSE order.
    var cur_xsize = work_xsize
    var n = len(transforms)
    for ti in range(n - 1, -1, -1):
        ref xf = transforms[ti]
        if xf.kind == XF_PREDICTOR:
            _apply_predictor(pixels, cur_xsize, height, xf)
        elif xf.kind == XF_COLOR:
            _apply_color(pixels, cur_xsize, height, xf)
        elif xf.kind == XF_SUBTRACT_GREEN:
            _apply_subtract_green(pixels, cur_xsize, height)
        else:  # color indexing: expands width back to real width
            pixels = _apply_color_indexing(pixels, width, height, xf)
            cur_xsize = width

    # Materialize RGBA image. ARGB -> RGBA channels.
    var img = Image.new(width, height, 4)
    for y in range(height):
        for x in range(width):
            var p = pixels[y * width + x]
            img.set(x, y, 0, UInt8(_argb_r(p)))
            img.set(x, y, 1, UInt8(_argb_g(p)))
            img.set(x, y, 2, UInt8(_argb_b(p)))
            img.set(x, y, 3, UInt8(_argb_a(p)))
    return img^


# ─────────────────────────────────────────────────────────────────────────────
# RIFF container parsing.
# ─────────────────────────────────────────────────────────────────────────────
@always_inline
def _u32le(data: List[UInt8], off: Int) raises -> Int:
    if off + 4 > len(data):
        raise Error("webp: truncated u32 at " + String(off))
    return Int(data[off]) | (Int(data[off + 1]) << 8) | (Int(data[off + 2]) << 16) | (
        Int(data[off + 3]) << 24
    )


def _fourcc(data: List[UInt8], off: Int) raises -> String:
    return chr(Int(data[off])) + chr(Int(data[off + 1])) + chr(Int(data[off + 2])) + chr(Int(data[off + 3]))


def decode_webp_bytes(data: List[UInt8]) raises -> Image:
    if len(data) < 12:
        raise Error("webp: file too small")
    if _fourcc(data, 0) != "RIFF":
        raise Error("webp: not a RIFF file")
    if _fourcc(data, 8) != "WEBP":
        raise Error("webp: not a WEBP file")

    # Walk chunks starting at offset 12.
    var pos = 12
    while pos + 8 <= len(data):
        var cc = _fourcc(data, pos)
        var clen = _u32le(data, pos + 4)
        var dstart = pos + 8
        if dstart + clen > len(data):
            raise Error("webp: chunk overruns file: " + cc)
        if cc == "VP8L":
            var sub = List[UInt8]()
            for i in range(clen):
                sub.append(data[dstart + i])
            return _decode_vp8l(sub^)
        elif cc == "VP8 ":
            raise Error("webp: lossy VP8 not supported (lossless VP8L only)")
        elif cc == "VP8X":
            # Extended container: skip its 10-byte header and keep scanning for a
            # VP8L subchunk inside the file (subchunks follow VP8X at top level).
            pass
        # advance (chunks are padded to even length)
        var advance = clen + (clen & 1)
        pos = dstart + advance

    raise Error("webp: no VP8L chunk found (lossless VP8L only)")


def decode_webp(path: String) raises -> Image:
    var f = open(path, "r")
    var d = f.read_bytes()
    f.close()
    var data = List[UInt8]()
    for i in range(len(d)):
        data.append(d[i])
    return decode_webp_bytes(data)


# ─────────────────────────────────────────────────────────────────────────────
# ENCODER — minimal but spec-valid lossless VP8L.
#   No transforms, no color cache, no LZ77. Every pixel emitted as a literal
#   (green, red, blue, alpha) using per-channel canonical Huffman codes built
#   from the image's own histograms. One Huffman group, no meta-Huffman.
# ─────────────────────────────────────────────────────────────────────────────
struct BitWriter(Movable):
    var bytes: List[UInt8]
    var bit_buf: UInt64
    var bit_cnt: Int

    def __init__(out self):
        self.bytes = List[UInt8]()
        self.bit_buf = 0
        self.bit_cnt = 0

    def write_bits(mut self, value: Int, n: Int):
        # LSB-first.
        self.bit_buf |= UInt64(value & ((1 << n) - 1)) << UInt64(self.bit_cnt)
        self.bit_cnt += n
        while self.bit_cnt >= 8:
            self.bytes.append(UInt8(Int(self.bit_buf & UInt64(0xFF))))
            self.bit_buf = self.bit_buf >> UInt64(8)
            self.bit_cnt -= 8

    def flush(mut self):
        if self.bit_cnt > 0:
            self.bytes.append(UInt8(Int(self.bit_buf & UInt64(0xFF))))
            self.bit_buf = 0
            self.bit_cnt = 0


# Build canonical Huffman code lengths from symbol histogram, capped at 15 bits.
def _build_code_lengths(histo: List[Int], max_len: Int) raises -> List[Int]:
    var n = len(histo)
    var lengths = List[Int]()
    for _ in range(n):
        lengths.append(0)
    # Count used symbols.
    var used = 0
    for i in range(n):
        if histo[i] > 0:
            used += 1
    if used == 0:
        return lengths^
    if used == 1:
        for i in range(n):
            if histo[i] > 0:
                lengths[i] = 1
        return lengths^

    # Package-merge would be ideal; use a simple length-limited approach:
    # build a Huffman tree, then if any length > max_len, fall back to a flat
    # length assignment. For our small alphabets this is fine.
    # Simple approach: iterative two-lowest-weight merge to get lengths.
    # nodes: (weight, list-of-symbols)
    var weights = List[Int]()
    var node_syms = List[List[Int]]()
    for i in range(n):
        if histo[i] > 0:
            weights.append(histo[i])
            var s = List[Int]()
            s.append(i)
            node_syms.append(s^)

    # Repeatedly merge the two lowest-weight nodes, incrementing the length of
    # every symbol in those nodes.
    while len(weights) > 1:
        # find two smallest
        var i0 = 0
        for i in range(1, len(weights)):
            if weights[i] < weights[i0]:
                i0 = i
        var i1 = -1
        for i in range(len(weights)):
            if i == i0:
                continue
            if i1 == -1 or weights[i] < weights[i1]:
                i1 = i
        # increment lengths for all symbols in both nodes
        for k in range(len(node_syms[i0])):
            lengths[node_syms[i0][k]] += 1
        for k in range(len(node_syms[i1])):
            lengths[node_syms[i1][k]] += 1
        # merge i1 into i0
        var merged_w = weights[i0] + weights[i1]
        var merged = node_syms[i0].copy()
        for k in range(len(node_syms[i1])):
            merged.append(node_syms[i1][k])
        # remove the higher index first
        var hi = i0 if i0 > i1 else i1
        var lo = i0 if i0 < i1 else i1
        # rebuild lists without hi and lo, then append merged
        var nw = List[Int]()
        var nn = List[List[Int]]()
        for i in range(len(weights)):
            if i == hi or i == lo:
                continue
            nw.append(weights[i])
            nn.append(node_syms[i].copy())
        nw.append(merged_w)
        nn.append(merged^)
        weights = nw^
        node_syms = nn^

    # Enforce max_len: if any length exceeds, clamp and (mildly) rebalance.
    var over = False
    for i in range(n):
        if lengths[i] > max_len:
            over = True
    if over:
        # Fallback: assign all used symbols a flat length large enough to be a
        # complete code. Find smallest L with 2^L >= used.
        var L = 1
        while (1 << L) < used:
            L += 1
        if L > max_len:
            L = max_len
        for i in range(n):
            lengths[i] = L if histo[i] > 0 else 0
    return lengths^


# Build canonical codes (the bit patterns) from lengths. Returns codes list.
# VP8L canonical codes are MSB-first; we generate code values then must emit them
# bit-reversed because the decoder reads bit-by-bit MSB-into-accumulator while
# our writer is LSB-first. We return (code_value, length); emit helper reverses.
def _build_codes(lengths: List[Int]) raises -> List[Int]:
    var n = len(lengths)
    var bl_count = List[Int]()
    for _ in range(16):
        bl_count.append(0)
    for i in range(n):
        bl_count[lengths[i]] += 1
    bl_count[0] = 0
    var next_code = List[Int]()
    for _ in range(16):
        next_code.append(0)
    var code = 0
    for bits in range(1, 16):
        code = (code + bl_count[bits - 1]) << 1
        next_code[bits] = code
    var codes = List[Int]()
    for _ in range(n):
        codes.append(0)
    for i in range(n):
        var L = lengths[i]
        if L > 0:
            codes[i] = next_code[L]
            next_code[L] += 1
    return codes^


@always_inline
def _reverse_bits(v: Int, n: Int) -> Int:
    var r = 0
    var x = v
    for _ in range(n):
        r = (r << 1) | (x & 1)
        x = x >> 1
    return r


# Emit a Huffman code for `symbol` (MSB-first canonical code -> reversed for the
# LSB-first writer, matching the decoder's bit-by-bit MSB accumulation).
@always_inline
def _emit_code(mut bw: BitWriter, codes: List[Int], lengths: List[Int], symbol: Int):
    var L = lengths[symbol]
    if L == 0:
        return
    var c = codes[symbol]
    bw.write_bits(_reverse_bits(c, L), L)


# Count how many symbols a length table actually uses (length > 0).
@always_inline
def _count_used(lengths: List[Int]) -> Int:
    var used = 0
    for i in range(len(lengths)):
        if lengths[i] > 0:
            used += 1
    return used


# Emit a literal symbol for a channel. If the channel's Huffman code has only a
# SINGLE used symbol, the decoder (HuffTree.single_symbol path) consumes ZERO
# bits — so the encoder must likewise emit nothing, or the bitstream desyncs.
@always_inline
def _emit_literal(
    mut bw: BitWriter, codes: List[Int], lengths: List[Int], symbol: Int, single: Bool
):
    if single:
        return
    _emit_code(bw, codes, lengths, symbol)


# Write a full Huffman code definition into the stream using the "normal" path
# with code-length codes 16/17/18 NOT used (we send each length directly via the
# code-length-code alphabet). For simplicity & validity we always use the normal
# (non-simple) coding with a complete code-length-code that can express 0..15.
def _write_huffman_code(mut bw: BitWriter, lengths: List[Int]) raises:
    # We choose the simplest valid encoding: NOT simple-code (bit 0), then a
    # code-length-code that assigns each of the lengths we actually use.
    # Build histogram of the code-length symbols (0..15) actually present.
    var n = len(lengths)
    var clc_histo = List[Int]()
    for _ in range(CODE_LENGTH_CODES):
        clc_histo.append(0)
    for i in range(n):
        clc_histo[lengths[i]] += 1
    # Build code-length-code lengths (a Huffman over the 0..15 + repeat symbols,
    # but we don't use repeats here, so symbols 16/17/18 stay unused).
    var clc_lengths = _build_code_lengths(clc_histo, 7)
    var clc_codes = _build_codes(clc_lengths)

    # simple-code bit = 0
    bw.write_bits(0, 1)
    # num_code_lengths - 4, in the fixed order; we must send all positions up to
    # the highest used in the fixed order. To be safe, send all 19.
    var order = _code_length_code_order()
    # find the highest index in `order` that has a nonzero length
    var max_idx = CODE_LENGTH_CODES - 1
    while max_idx >= 4:
        if clc_lengths[order[max_idx]] != 0:
            break
        max_idx -= 1
    var num_code_lengths = max_idx + 1
    if num_code_lengths < 4:
        num_code_lengths = 4
    bw.write_bits(num_code_lengths - 4, 4)
    for i in range(num_code_lengths):
        bw.write_bits(clc_lengths[order[i]], 3)
    # use_length = 0 (read lengths for all symbols)
    bw.write_bits(0, 1)
    # Now emit each symbol's code length using the clc code (no repeats).
    for i in range(n):
        _emit_code(bw, clc_codes, clc_lengths, lengths[i])


def encode_webp_bytes(img: Image) raises -> List[UInt8]:
    if img.bit_depth != 8:
        raise Error("encode_webp: only 8-bit images supported")
    var w = img.width
    var h = img.height
    var ch = img.channels

    # Gather ARGB pixels and per-channel histograms.
    var green_histo = List[Int]()
    for _ in range(NUM_LITERAL + NUM_LENGTH):  # green alphabet (no cache)
        green_histo.append(0)
    var red_histo = List[Int]()
    var blue_histo = List[Int]()
    var alpha_histo = List[Int]()
    for _ in range(NUM_LITERAL):
        red_histo.append(0)
        blue_histo.append(0)
        alpha_histo.append(0)

    var argb = List[UInt32]()
    for y in range(h):
        for x in range(w):
            var r = 0
            var g = 0
            var b = 0
            var a = 255
            if ch == 1:
                var gg = Int(img.get(x, y, 0))
                r = gg; g = gg; b = gg
            elif ch == 3:
                r = Int(img.get(x, y, 0))
                g = Int(img.get(x, y, 1))
                b = Int(img.get(x, y, 2))
            else:
                r = Int(img.get(x, y, 0))
                g = Int(img.get(x, y, 1))
                b = Int(img.get(x, y, 2))
                a = Int(img.get(x, y, 3))
            argb.append(_make_argb(a, r, g, b))
            green_histo[g] += 1
            red_histo[r] += 1
            blue_histo[b] += 1
            alpha_histo[a] += 1

    # Distance alphabet unused (no LZ77) but must be a valid (complete) code.
    var dist_histo = List[Int]()
    for _ in range(NUM_DISTANCE):
        dist_histo.append(0)
    dist_histo[0] = 1  # one phantom symbol so the code is well-formed

    var green_len = _build_code_lengths(green_histo, 15)
    var red_len = _build_code_lengths(red_histo, 15)
    var blue_len = _build_code_lengths(blue_histo, 15)
    var alpha_len = _build_code_lengths(alpha_histo, 15)
    var dist_len = _build_code_lengths(dist_histo, 15)

    var green_codes = _build_codes(green_len)
    var red_codes = _build_codes(red_len)
    var blue_codes = _build_codes(blue_len)
    var alpha_codes = _build_codes(alpha_len)
    var _dist_codes = _build_codes(dist_len)

    # Single-symbol channels emit zero bits on decode (HuffTree.single_symbol).
    var green_single = _count_used(green_len) == 1
    var red_single = _count_used(red_len) == 1
    var blue_single = _count_used(blue_len) == 1
    var alpha_single = _count_used(alpha_len) == 1

    # Build the VP8L bitstream.
    var bw = BitWriter()
    bw.write_bits(0x2F, 8)             # signature
    bw.write_bits(w - 1, 14)           # width-1
    bw.write_bits(h - 1, 14)           # height-1
    bw.write_bits(1, 1)                # alpha_used = 1
    bw.write_bits(0, 3)                # version 0

    bw.write_bits(0, 1)                # no transforms (transform-present = 0)

    # Entropy-coded image header:
    bw.write_bits(0, 1)                # no color cache
    bw.write_bits(0, 1)                # no meta-Huffman (single group)

    # Emit the 5 Huffman code definitions for the single group.
    _write_huffman_code(bw, green_len)
    _write_huffman_code(bw, red_len)
    _write_huffman_code(bw, blue_len)
    _write_huffman_code(bw, alpha_len)
    _write_huffman_code(bw, dist_len)

    # Emit all pixels as literals. Single-symbol channels emit no bits (the
    # decoder's single_symbol path consumes none) to keep the stream in sync.
    for i in range(len(argb)):
        var p = argb[i]
        _emit_literal(bw, green_codes, green_len, _argb_g(p), green_single)
        _emit_literal(bw, red_codes, red_len, _argb_r(p), red_single)
        _emit_literal(bw, blue_codes, blue_len, _argb_b(p), blue_single)
        _emit_literal(bw, alpha_codes, alpha_len, _argb_a(p), alpha_single)

    bw.flush()
    var vp8l = bw.bytes.copy()

    # Wrap in RIFF/WEBP container.
    var out = List[UInt8]()
    _append_str(out, "RIFF")
    var riff_size = 4 + 8 + len(vp8l) + (len(vp8l) & 1)  # "WEBP" + chunk hdr + data + pad
    _append_u32le(out, riff_size)
    _append_str(out, "WEBP")
    _append_str(out, "VP8L")
    _append_u32le(out, len(vp8l))
    for i in range(len(vp8l)):
        out.append(vp8l[i])
    if (len(vp8l) & 1) == 1:
        out.append(0)  # padding to even
    return out^


def _append_str(mut out: List[UInt8], s: String):
    var b = s.as_bytes()
    for i in range(len(b)):
        out.append(b[i])


def _append_u32le(mut out: List[UInt8], v: Int):
    out.append(UInt8(v & 0xFF))
    out.append(UInt8((v >> 8) & 0xFF))
    out.append(UInt8((v >> 16) & 0xFF))
    out.append(UInt8((v >> 24) & 0xFF))


def encode_webp(img: Image, path: String) raises:
    var bytes = encode_webp_bytes(img)
    with open(path, "w") as f:
        f.write_bytes(Span(bytes))
