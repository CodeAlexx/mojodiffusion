# graphics.inflate — a real DEFLATE decompressor (RFC 1951), 100% Mojo.
#
# Inverse of graphics/deflate.mojo. Decodes all three block types:
#   BTYPE=00 stored (uncompressed), BTYPE=01 fixed Huffman, BTYPE=10 dynamic
#   Huffman. Full LZ77 length/distance back-reference machinery is implemented
#   (copy from the already-produced output window, byte-by-byte so overlapping
#   copies — distance < length — work correctly).
#
# Bit order matches the compressor: DEFLATE packs bits LSB-first into bytes
# (BitReader.read_bits), but Huffman codes are read MSB-first one bit at a time
# while walking the canonical-code tables (decode_symbol). The length/distance
# base+extra tables mirror deflate.mojo exactly.
#
# zlib_inflate() additionally strips the RFC-1950 zlib container (2-byte header
# + 4-byte big-endian Adler-32 trailer) and verifies the Adler-32 of the output.


struct BitReader(Movable):
    var data: List[UInt8]
    var pos: Int      # current byte index
    var cur: Int      # bit buffer (low `nbits` bits valid)
    var nbits: Int    # number of valid bits currently buffered

    def __init__(out self, data: List[UInt8]):
        self.data = data.copy()
        self.pos = 0
        self.cur = 0
        self.nbits = 0

    def read_bit(mut self) raises -> Int:
        if self.nbits == 0:
            if self.pos >= len(self.data):
                raise Error("inflate: unexpected end of input (read_bit)")
            self.cur = Int(self.data[self.pos])
            self.pos += 1
            self.nbits = 8
        var b = self.cur & 1
        self.cur >>= 1
        self.nbits -= 1
        return b

    def read_bits(mut self, n: Int) raises -> Int:
        # LSB-first: first bit read is the least-significant bit of the value.
        var v = 0
        for i in range(n):
            v |= self.read_bit() << i
        return v

    def align_to_byte(mut self):
        # Discard remaining bits in the current partial byte (for stored blocks).
        self.cur = 0
        self.nbits = 0

    def read_byte(mut self) raises -> Int:
        # Must be byte-aligned (align_to_byte called first). Reads a raw byte.
        if self.pos >= len(self.data):
            raise Error("inflate: unexpected end of input (read_byte)")
        var b = Int(self.data[self.pos])
        self.pos += 1
        return b


# ── code tables (length / distance base + extra bits) — mirror deflate.mojo ────
def _len_base() -> List[Int]:
    return [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43,
            51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258]


def _len_extra() -> List[Int]:
    return [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4,
            4, 4, 4, 5, 5, 5, 5, 0]


def _dist_base() -> List[Int]:
    return [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257,
            385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289,
            16385, 24577]


def _dist_extra() -> List[Int]:
    return [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9,
            9, 10, 10, 11, 11, 12, 12, 13, 13]


# ── canonical Huffman decode table ────────────────────────────────────────────
# Built from per-symbol code lengths (the canonical form deflate.mojo emits).
# For decoding we keep, per code length L: the first canonical code at that
# length and the symbol index offset, so a code value can be mapped back to a
# symbol. This is the classic "first_code / first_symbol per length" scheme.
struct HuffTable(Movable):
    var counts: List[Int]       # counts[L] = number of codes of length L
    var symbols: List[Int]      # symbols sorted by (length, code) order
    var maxbits: Int

    def __init__(out self, lengths: List[Int], maxbits: Int) raises:
        self.maxbits = maxbits
        self.counts = List[Int]()
        for _ in range(maxbits + 1):
            self.counts.append(0)
        var n = len(lengths)
        for s in range(n):
            var l = lengths[s]
            if l < 0 or l > maxbits:
                raise Error("inflate: invalid code length")
            self.counts[l] += 1
        self.counts[0] = 0  # length-0 symbols are absent from the code

        # offsets[L] = index into `symbols` of the first symbol with length L
        var offsets = List[Int]()
        for _ in range(maxbits + 2):
            offsets.append(0)
        for l in range(1, maxbits + 1):
            offsets[l + 1] = offsets[l] + self.counts[l]

        self.symbols = List[Int]()
        for _ in range(n):
            self.symbols.append(0)
        for s in range(n):
            var l = lengths[s]
            if l != 0:
                self.symbols[offsets[l]] = s
                offsets[l] += 1


def decode_symbol(mut br: BitReader, tab: HuffTable) raises -> Int:
    # Walk one bit at a time (MSB-first within the canonical code), tracking the
    # running code value and the count/first-code/first-symbol per length.
    var code = 0
    var first = 0   # first canonical code at the current length
    var index = 0   # index of the first symbol at the current length
    for length in range(1, tab.maxbits + 1):
        code |= br.read_bit()
        var count = tab.counts[length]
        if code - first < count:
            return tab.symbols[index + (code - first)]
        index += count
        first += count
        first <<= 1
        code <<= 1
    raise Error("inflate: invalid Huffman code (ran off the table)")


# ── fixed Huffman tables (RFC 1951 §3.2.6) ────────────────────────────────────
def _fixed_litlen_table() raises -> HuffTable:
    var lengths = List[Int]()
    for s in range(288):
        if s <= 143:
            lengths.append(8)
        elif s <= 255:
            lengths.append(9)
        elif s <= 279:
            lengths.append(7)
        else:
            lengths.append(8)
    return HuffTable(lengths, 9)


def _fixed_dist_table() raises -> HuffTable:
    var lengths = List[Int]()
    for _ in range(30):
        lengths.append(5)
    return HuffTable(lengths, 5)


# ── dynamic Huffman header parsing (RFC 1951 §3.2.7) ──────────────────────────
def _read_dynamic_tables(
    mut br: BitReader, mut lit_out: HuffTable, mut dist_out: HuffTable
) raises:
    var hlit = br.read_bits(5) + 257
    var hdist = br.read_bits(5) + 1
    var hclen = br.read_bits(4) + 4

    var order = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15]
    var cl_lengths = List[Int]()
    for _ in range(19):
        cl_lengths.append(0)
    for i in range(hclen):
        cl_lengths[order[i]] = br.read_bits(3)
    var cl_table = HuffTable(cl_lengths, 7)

    # Decode the combined (litlen + dist) code-length sequence with the cl table.
    var total = hlit + hdist
    var all_lengths = List[Int]()
    while len(all_lengths) < total:
        var sym = decode_symbol(br, cl_table)
        if sym < 16:
            all_lengths.append(sym)
        elif sym == 16:
            # repeat previous length 3..6 times
            if len(all_lengths) == 0:
                raise Error("inflate: repeat (16) with no previous length")
            var prev = all_lengths[len(all_lengths) - 1]
            var rep = br.read_bits(2) + 3
            for _ in range(rep):
                all_lengths.append(prev)
        elif sym == 17:
            # repeat zero 3..10 times
            var rep = br.read_bits(3) + 3
            for _ in range(rep):
                all_lengths.append(0)
        elif sym == 18:
            # repeat zero 11..138 times
            var rep = br.read_bits(7) + 11
            for _ in range(rep):
                all_lengths.append(0)
        else:
            raise Error("inflate: invalid code-length symbol")

    if len(all_lengths) != total:
        raise Error("inflate: code-length sequence overran the table sizes")

    var lit_lengths = List[Int]()
    for i in range(hlit):
        lit_lengths.append(all_lengths[i])
    var dist_lengths = List[Int]()
    for i in range(hdist):
        dist_lengths.append(all_lengths[hlit + i])

    lit_out = HuffTable(lit_lengths, 15)
    dist_out = HuffTable(dist_lengths, 15)


# ── block decoders ────────────────────────────────────────────────────────────
def _inflate_stored(mut br: BitReader, mut out: List[UInt8]) raises:
    br.align_to_byte()
    var len_lo = br.read_byte()
    var len_hi = br.read_byte()
    var blklen = len_lo | (len_hi << 8)
    var nlen_lo = br.read_byte()
    var nlen_hi = br.read_byte()
    var nlen = nlen_lo | (nlen_hi << 8)
    if (blklen ^ 0xFFFF) != nlen:
        raise Error("inflate: stored-block LEN/NLEN mismatch")
    for _ in range(blklen):
        out.append(UInt8(br.read_byte()))


def _inflate_huffman(
    mut br: BitReader, lit: HuffTable, dist: HuffTable, mut out: List[UInt8]
) raises:
    var lbase = _len_base()
    var lextra = _len_extra()
    var dbase = _dist_base()
    var dextra = _dist_extra()
    while True:
        var sym = decode_symbol(br, lit)
        if sym == 256:
            break  # end of block
        if sym < 256:
            out.append(UInt8(sym))
        else:
            var lc = sym - 257
            if lc < 0 or lc >= len(lbase):
                raise Error("inflate: invalid length symbol")
            var length = lbase[lc]
            if lextra[lc] > 0:
                length += br.read_bits(lextra[lc])
            var dsym = decode_symbol(br, dist)
            if dsym < 0 or dsym >= len(dbase):
                raise Error("inflate: invalid distance symbol")
            var distance = dbase[dsym]
            if dextra[dsym] > 0:
                distance += br.read_bits(dextra[dsym])
            if distance > len(out):
                raise Error("inflate: distance points before start of output")
            # Copy byte-by-byte so overlapping copies (distance < length) work.
            var start = len(out) - distance
            for k in range(length):
                out.append(out[start + k])


# ── public API ────────────────────────────────────────────────────────────────
def inflate(data: List[UInt8]) raises -> List[UInt8]:
    """RFC 1951 raw DEFLATE decompressor. Decodes a sequence of blocks until the
    BFINAL bit is set, returning the reconstructed bytes."""
    var br = BitReader(data)
    var out = List[UInt8]()
    var final = 0
    while final == 0:
        final = br.read_bit()
        var btype = br.read_bits(2)
        if btype == 0:
            _inflate_stored(br, out)
        elif btype == 1:
            var lit = _fixed_litlen_table()
            var dist = _fixed_dist_table()
            _inflate_huffman(br, lit, dist, out)
        elif btype == 2:
            var lit = _fixed_litlen_table()   # placeholder, overwritten below
            var dist = _fixed_dist_table()
            _read_dynamic_tables(br, lit, dist)
            _inflate_huffman(br, lit, dist, out)
        else:
            raise Error("inflate: reserved block type (BTYPE=3)")
    return out^


def _adler32(data: List[UInt8]) -> Int:
    var a = 1
    var b = 0
    var MOD = 65521
    for i in range(len(data)):
        a = (a + Int(data[i])) % MOD
        b = (b + a) % MOD
    return (b << 16) | a


def zlib_inflate(data: List[UInt8]) raises -> List[UInt8]:
    """Strip the RFC-1950 zlib container (2-byte header + 4-byte big-endian
    Adler-32 trailer), inflate the raw DEFLATE body, and verify the Adler-32 of
    the output before returning it."""
    if len(data) < 6:
        raise Error("zlib_inflate: stream too short for a zlib container")
    # CMF/FLG header check: CM (low nibble of CMF) must be 8 (deflate); the
    # (CMF*256+FLG) value must be a multiple of 31 per RFC 1950.
    var cmf = Int(data[0])
    var flg = Int(data[1])
    if (cmf & 0x0F) != 8:
        raise Error("zlib_inflate: not a deflate stream (CM != 8)")
    if ((cmf << 8) | flg) % 31 != 0:
        raise Error("zlib_inflate: bad zlib header check bits")
    if (flg & 0x20) != 0:
        raise Error("zlib_inflate: preset dictionary not supported")

    # Raw deflate body = bytes [2 .. len-4); trailing 4 bytes = Adler-32 (BE).
    var body = List[UInt8]()
    for i in range(2, len(data) - 4):
        body.append(data[i])
    var out = inflate(body)

    var stored = (Int(data[len(data) - 4]) << 24) | (
        Int(data[len(data) - 3]) << 16
    ) | (Int(data[len(data) - 2]) << 8) | Int(data[len(data) - 1])
    var got = _adler32(out)
    if stored != got:
        raise Error("zlib_inflate: Adler-32 mismatch (corrupt stream)")
    return out^
