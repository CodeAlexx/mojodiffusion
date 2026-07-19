# graphics.deflate — a real DEFLATE compressor (RFC 1951), 100% Mojo.
#
# LZ77 (hash-chain greedy matching) produces a token stream; it is then encoded
# with BOTH the FIXED Huffman table (BTYPE=01) and a frequency-optimized DYNAMIC
# Huffman table (BTYPE=10), and the smaller result is emitted (zlib's strategy).
# Dynamic Huffman uses length-limited codes (<=15 bits) built via a Huffman tree
# + zlib's bit-length overflow redistribution, canonical code assignment, and the
# RLE-coded code-length header (symbols 16/17/18). Output is one final block; any
# inflater (zlib, PIL, browsers) decodes it.
#
# Bit order: DEFLATE packs bits LSB-first into bytes, but Huffman codes are
# emitted MSB-first — write_bits() is LSB-first (headers + extra bits),
# write_code() is MSB-first (Huffman codes + the fixed 5-bit distance codes).


struct BitWriter(Movable):
    var out: List[UInt8]
    var cur: Int
    var nbits: Int

    def __init__(out self):
        self.out = List[UInt8]()
        self.cur = 0
        self.nbits = 0

    def put_bit(mut self, b: Int):
        self.cur |= (b & 1) << self.nbits
        self.nbits += 1
        if self.nbits == 8:
            self.out.append(UInt8(self.cur & 0xFF))
            self.cur = 0
            self.nbits = 0

    def write_bits(mut self, value: Int, n: Int):
        for i in range(n):
            self.put_bit((value >> i) & 1)

    def write_code(mut self, code: Int, n: Int):
        for i in range(n):
            self.put_bit((code >> (n - 1 - i)) & 1)

    def flush(mut self):
        if self.nbits > 0:
            self.out.append(UInt8(self.cur & 0xFF))
            self.cur = 0
            self.nbits = 0


# ── code tables (length / distance base + extra bits) ─────────────────────────
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


def _hash3(raw: List[UInt8], i: Int) -> Int:
    return ((Int(raw[i]) << 10) ^ (Int(raw[i + 1]) << 5) ^ Int(raw[i + 2])) & 0x7FFF


# ── LZ77 token stream ─────────────────────────────────────────────────────────
# Per token: lsym = literal byte (0-255) | length code (257-285); lval = match
# length (matches); dsym = distance code (0-29) or -1 for a literal; dval =
# distance value. The end-of-block symbol 256 is appended as a final token.
struct Tokens(Movable):
    var lsym: List[Int]
    var lval: List[Int]
    var dsym: List[Int]
    var dval: List[Int]
    var has_match: Bool

    def __init__(out self):
        self.lsym = List[Int]()
        self.lval = List[Int]()
        self.dsym = List[Int]()
        self.dval = List[Int]()
        self.has_match = False


def _find_match(raw: List[UInt8], head: List[Int], prev: List[Int], i: Int, n: Int, mut blen: Int, mut bdist: Int):
    """Longest match for position i among earlier window positions (read-only:
    does not modify the hash chains). Sets blen/bdist (blen=0 if none)."""
    blen = 0
    bdist = 0
    if i + 3 > n:
        return
    var j = head[_hash3(raw, i)]
    var chain = 0
    while j >= 0 and chain < 128 and (i - j) <= 32768:
        var l = 0
        while l < 258 and i + l < n and raw[j + l] == raw[i + l]:
            l += 1
        if l > blen:
            blen = l
            bdist = i - j
            if l >= 258:
                break
        j = prev[j]
        chain += 1


def _emit_match(mut tok: Tokens, lbase: List[Int], dbase: List[Int], mlen: Int, mdist: Int):
    var lc = 28
    for k in range(28, -1, -1):
        if lbase[k] <= mlen:
            lc = k
            break
    var dc = 29
    for k in range(29, -1, -1):
        if dbase[k] <= mdist:
            dc = k
            break
    tok.lsym.append(257 + lc)
    tok.lval.append(mlen)
    tok.dsym.append(dc)
    tok.dval.append(mdist)
    tok.has_match = True


def _lz77(raw: List[UInt8]) raises -> Tokens:
    """LZ77 with LAZY matching: a match at i is deferred if i+1 has a longer one
    (then i is emitted as a literal). Hash chains are inserted AFTER each search."""
    var lbase = _len_base()
    var dbase = _dist_base()
    var tok = Tokens()
    var n = len(raw)
    comptime HSIZE = 1 << 15
    var head = List[Int]()
    for _ in range(HSIZE):
        head.append(-1)
    var prev = List[Int]()
    for _ in range(n if n > 0 else 1):
        prev.append(-1)

    var i = 0
    while i < n:
        var l1 = 0
        var d1 = 0
        _find_match(raw, head, prev, i, n, l1, d1)   # search uses pre-insert head
        if i + 3 <= n:                               # insert i AFTER its search
            var hi = _hash3(raw, i)
            prev[i] = head[hi]
            head[hi] = i
        if l1 >= 3:
            var l2 = 0
            var d2 = 0
            if i + 1 < n:
                _find_match(raw, head, prev, i + 1, n, l2, d2)  # i now in chain
            if l2 > l1:
                # defer: emit literal at i, take the longer match next iteration
                tok.lsym.append(Int(raw[i]))
                tok.lval.append(0); tok.dsym.append(-1); tok.dval.append(0)
                i += 1
            else:
                _emit_match(tok, lbase, dbase, l1, d1)
                var k = i + 1
                var end = i + l1
                while k < end:
                    if k + 3 <= n:
                        var hk = _hash3(raw, k)
                        prev[k] = head[hk]
                        head[hk] = k
                    k += 1
                i = end
        else:
            tok.lsym.append(Int(raw[i]))
            tok.lval.append(0); tok.dsym.append(-1); tok.dval.append(0)
            i += 1
    tok.lsym.append(256)  # end-of-block
    tok.lval.append(0); tok.dsym.append(-1); tok.dval.append(0)
    return tok^


# ── length-limited Huffman (tree + zlib bit-length redistribution) ────────────
def code_lengths(freq: List[Int], maxsym: Int, L: Int) raises -> List[Int]:
    var lengths = List[Int]()
    for _ in range(maxsym + 1):
        lengths.append(0)
    var syms = List[Int]()
    for s in range(maxsym + 1):
        if freq[s] > 0:
            syms.append(s)
    var n = len(syms)
    if n == 0:
        return lengths^
    if n == 1:
        lengths[syms[0]] = 1
        return lengths^

    # Huffman tree via repeated extract-two-min (n small: O(n^2))
    var nf = List[Int](); var nsym = List[Int](); var nl = List[Int](); var nr = List[Int]()
    for i in range(n):
        nf.append(freq[syms[i]]); nsym.append(syms[i]); nl.append(-1); nr.append(-1)
    var active = List[Int]()
    for i in range(n):
        active.append(i)
    while len(active) > 1:
        var i1 = 0
        for k in range(1, len(active)):
            if nf[active[k]] < nf[active[i1]]:
                i1 = k
        var a = active[i1]
        active[i1] = active[len(active) - 1]; _ = active.pop()
        var i2 = 0
        for k in range(1, len(active)):
            if nf[active[k]] < nf[active[i2]]:
                i2 = k
        var b = active[i2]
        active[i2] = active[len(active) - 1]; _ = active.pop()
        var ni = len(nf)
        nf.append(nf[a] + nf[b]); nsym.append(-1); nl.append(a); nr.append(b)
        active.append(ni)
    var root = active[0]

    # raw depths (= code lengths) via an explicit stack
    var raw = List[Int]()
    for _ in range(maxsym + 1):
        raw.append(0)
    var stn = List[Int](); var std = List[Int]()
    stn.append(root); std.append(0)
    while len(stn) > 0:
        var node = stn[len(stn) - 1]; _ = stn.pop()
        var dep = std[len(std) - 1]; _ = std.pop()
        if nsym[node] >= 0:
            raw[nsym[node]] = dep if dep > 0 else 1
        else:
            stn.append(nl[node]); std.append(dep + 1)
            stn.append(nr[node]); std.append(dep + 1)

    # bl_count with clamp to L + zlib overflow redistribution
    var bl = List[Int]()
    for _ in range(L + 1):
        bl.append(0)
    var overflow = 0
    for i in range(n):
        var b = raw[syms[i]]
        if b > L:
            b = L
            overflow += 1
        bl[b] += 1
    while overflow > 0:
        var bits = L - 1
        while bits >= 1 and bl[bits] == 0:
            bits -= 1
        if bits < 1:
            break  # cannot redistribute further (caller validates result)
        bl[bits] -= 1
        bl[bits + 1] += 2
        bl[L] -= 1
        overflow -= 2

    # assign: most-frequent symbols get the shortest lengths
    var order = List[Int]()
    for i in range(n):
        order.append(syms[i])
    for a in range(1, n):
        var key = order[a]
        var b = a - 1
        while b >= 0 and (freq[order[b]] < freq[key] or (freq[order[b]] == freq[key] and order[b] > key)):
            order[b + 1] = order[b]
            b -= 1
        order[b + 1] = key
    var idx = 0
    for length in range(1, L + 1):
        for _ in range(bl[length]):
            lengths[order[idx]] = length
            idx += 1
    return lengths^


def canonical_codes(lengths: List[Int], maxsym: Int, maxbits: Int) -> List[Int]:
    var blc = List[Int]()
    for _ in range(maxbits + 1):
        blc.append(0)
    for s in range(maxsym + 1):
        if lengths[s] > 0:
            blc[lengths[s]] += 1
    var next_code = List[Int]()
    for _ in range(maxbits + 1):
        next_code.append(0)
    var code = 0
    for bits in range(1, maxbits + 1):
        code = (code + blc[bits - 1]) << 1
        next_code[bits] = code
    var codes = List[Int]()
    for _ in range(maxsym + 1):
        codes.append(0)
    for s in range(maxsym + 1):
        var l = lengths[s]
        if l > 0:
            codes[s] = next_code[l]
            next_code[l] += 1
    return codes^


# ── fixed-Huffman encoding ────────────────────────────────────────────────────
def _write_litlen_fixed(mut bw: BitWriter, sym: Int):
    if sym <= 143:
        bw.write_code(0x30 + sym, 8)
    elif sym <= 255:
        bw.write_code(0x190 + (sym - 144), 9)
    elif sym <= 279:
        bw.write_code(sym - 256, 7)
    else:
        bw.write_code(0xC0 + (sym - 280), 8)


def _encode_fixed(tok: Tokens) raises -> List[UInt8]:
    var lbase = _len_base(); var lextra = _len_extra()
    var dbase = _dist_base(); var dextra = _dist_extra()
    var bw = BitWriter()
    bw.write_bits(1, 1)   # BFINAL
    bw.write_bits(1, 2)   # BTYPE = 01
    var m = len(tok.lsym)
    for t in range(m):
        var ls = tok.lsym[t]
        _write_litlen_fixed(bw, ls)
        if tok.dsym[t] >= 0:
            var lc = ls - 257
            if lextra[lc] > 0:
                bw.write_bits(tok.lval[t] - lbase[lc], lextra[lc])
            var dc = tok.dsym[t]
            bw.write_code(dc, 5)
            if dextra[dc] > 0:
                bw.write_bits(tok.dval[t] - dbase[dc], dextra[dc])
    bw.flush()
    return bw.out.copy()


# ── dynamic-Huffman encoding ──────────────────────────────────────────────────
struct ClTokens(Movable):
    var sym: List[Int]
    var exval: List[Int]
    var exn: List[Int]
    def __init__(out self):
        self.sym = List[Int]()
        self.exval = List[Int]()
        self.exn = List[Int]()
    def add(mut self, s: Int, v: Int, n: Int):
        self.sym.append(s); self.exval.append(v); self.exn.append(n)


def _rle_lengths(seq: List[Int]) -> ClTokens:
    var out = ClTokens()
    var n = len(seq)
    var i = 0
    while i < n:
        var v = seq[i]
        var run = 1
        while i + run < n and seq[i + run] == v:
            run += 1
        if v == 0:
            var rem = run
            while rem >= 11:
                var r = rem if rem < 138 else 138
                out.add(18, r - 11, 7); rem -= r
            while rem >= 3:
                var r = rem if rem < 10 else 10
                out.add(17, r - 3, 3); rem -= r
            while rem > 0:
                out.add(0, 0, 0); rem -= 1
        else:
            out.add(v, 0, 0)
            var rem = run - 1
            while rem >= 3:
                var r = rem if rem < 6 else 6
                out.add(16, r - 3, 2); rem -= r
            while rem > 0:
                out.add(v, 0, 0); rem -= 1
        i += run
    return out^


def _kraft_ok(freq: List[Int], lengths: List[Int], maxsym: Int, maxbits: Int) -> Bool:
    """True if `lengths` is a valid prefix code for the used symbols: every
    freq>0 symbol has a length in [1,maxbits], and the code is complete
    (Kraft sum == 2^maxbits) — or a lone single code. Guards against a buggy
    length-limit pass shipping an invalid (zlib-rejected) Huffman table."""
    var nonzero = 0
    var kraft = 0
    for s in range(maxsym + 1):
        var l = lengths[s]
        if freq[s] > 0 and l == 0:
            return False
        if l > 0:
            if l > maxbits:
                return False
            nonzero += 1
            kraft += (1 << (maxbits - l))
    if nonzero == 0:
        return True
    if nonzero == 1:
        return kraft <= (1 << maxbits)
    return kraft == (1 << maxbits)


def _encode_dynamic(tok: Tokens) raises -> List[UInt8]:
    var lbase = _len_base(); var lextra = _len_extra()
    var dbase = _dist_base(); var dextra = _dist_extra()
    var m = len(tok.lsym)

    # frequencies
    var lfreq = List[Int]()
    for _ in range(286):
        lfreq.append(0)
    var dfreq = List[Int]()
    for _ in range(30):
        dfreq.append(0)
    for t in range(m):
        lfreq[tok.lsym[t]] += 1
        if tok.dsym[t] >= 0:
            dfreq[tok.dsym[t]] += 1

    var llen = code_lengths(lfreq, 285, 15)
    var dlen = code_lengths(dfreq, 29, 15)
    # ensure at least one distance code (RFC: a block with no distances still
    # carries one dummy distance code)
    var any_d = False
    for s in range(30):
        if dlen[s] > 0:
            any_d = True
    if not any_d:
        dlen[0] = 1

    # bail to fixed Huffman if the length-limited codes aren't valid prefix codes
    if not _kraft_ok(lfreq, llen, 285, 15):
        return List[UInt8]()
    if not _kraft_ok(dfreq, dlen, 29, 15):
        return List[UInt8]()

    var lcode = canonical_codes(llen, 285, 15)
    var dcode = canonical_codes(dlen, 29, 15)

    # HLIT / HDIST counts
    var hlit = 286
    while hlit > 257 and llen[hlit - 1] == 0:
        hlit -= 1
    var hdist = 30
    while hdist > 1 and dlen[hdist - 1] == 0:
        hdist -= 1

    # combined code-length sequence -> RLE
    var seq = List[Int]()
    for s in range(hlit):
        seq.append(llen[s])
    for s in range(hdist):
        seq.append(dlen[s])
    var cl = _rle_lengths(seq)

    # code-length-alphabet Huffman (symbols 0..18, max 7 bits)
    var clfreq = List[Int]()
    for _ in range(19):
        clfreq.append(0)
    for t in range(len(cl.sym)):
        clfreq[cl.sym[t]] += 1
    var cllen = code_lengths(clfreq, 18, 7)
    if not _kraft_ok(clfreq, cllen, 18, 7):
        return List[UInt8]()
    var clcode = canonical_codes(cllen, 18, 7)

    var order = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15]
    var hclen = 19
    while hclen > 4 and cllen[order[hclen - 1]] == 0:
        hclen -= 1

    var bw = BitWriter()
    bw.write_bits(1, 1)   # BFINAL
    bw.write_bits(2, 2)   # BTYPE = 10 (dynamic)
    bw.write_bits(hlit - 257, 5)
    bw.write_bits(hdist - 1, 5)
    bw.write_bits(hclen - 4, 4)
    for i in range(hclen):
        bw.write_bits(cllen[order[i]], 3)
    # emit the RLE code-length tokens with the cl Huffman + their extra bits
    for t in range(len(cl.sym)):
        var s = cl.sym[t]
        bw.write_code(clcode[s], cllen[s])
        if cl.exn[t] > 0:
            bw.write_bits(cl.exval[t], cl.exn[t])
    # emit the data with the dynamic litlen / dist Huffman
    for t in range(m):
        var ls = tok.lsym[t]
        bw.write_code(lcode[ls], llen[ls])
        if tok.dsym[t] >= 0:
            var lc = ls - 257
            if lextra[lc] > 0:
                bw.write_bits(tok.lval[t] - lbase[lc], lextra[lc])
            var dc = tok.dsym[t]
            bw.write_code(dcode[dc], dlen[dc])
            if dextra[dc] > 0:
                bw.write_bits(tok.dval[t] - dbase[dc], dextra[dc])
    bw.flush()
    return bw.out.copy()


def deflate(raw: List[UInt8]) raises -> List[UInt8]:
    """Compress `raw` to a raw DEFLATE stream — the smaller of a fixed-Huffman
    and a dynamic-Huffman block (both over the same LZ77 token stream)."""
    var tok = _lz77(raw)
    var fixed = _encode_fixed(tok)
    if not tok.has_match:
        return fixed^  # no distances -> skip dynamic (avoids the dummy-dist case)
    var dyn = _encode_dynamic(tok)  # empty if the dynamic codes failed validation
    if len(dyn) > 0 and len(dyn) < len(fixed):
        return dyn^
    return fixed^
