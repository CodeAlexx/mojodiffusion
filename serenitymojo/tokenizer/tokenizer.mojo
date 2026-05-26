# Pure-Mojo byte-level BPE tokenizer for the Qwen3 text encoder.
#
# Replaces the Rust `tokenizers` crate (which loads tokenizer.json at runtime).
# This is a CPU-only string-processing module: no Python, no Rust crate at
# runtime. tokenizer.json is parsed ONCE at startup by a minimal hand-rolled
# JSON reader (sufficient for vocab + merges + added_tokens).
#
# Source of truth: the Z-Image / Klein Qwen3 tokenizer.json (BPE, byte-level).
#   model.type = BPE, byte_fallback=False, ignore_merges=False
#   normalizer = NFC
#   pre_tokenizer = Sequence[ Split(<Qwen2 regex>, Isolated), ByteLevel(add_prefix_space=false) ]
#   post_processor = ByteLevel  (adds NO special tokens)
#   vocab size 151643, merges 151387, 26 added/special tokens (151643..151668)
#
# The pre-tokenizer regex (replicated as a hand-rolled scanner below):
#   (?i:'s|'t|'re|'ve|'m|'ll|'d)
#   | [^\r\n\p{L}\p{N}]?\p{L}+
#   | \p{N}
#   |  ?[^\s\p{L}\p{N}]+[\r\n]*
#   | \s*[\r\n]+
#   | \s+(?!\S)
#   | \s+
# Matched leftmost-first (Perl/Rust-regex alternation order), NOT POSIX-longest.

from std.memory import alloc
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY


# ----------------------------------------------------------------------------
# Unicode category approximation  (\p{L}, \p{N})
# ----------------------------------------------------------------------------
# Mojo stdlib has NO Unicode category tables, so \p{L}/\p{N} are approximated by
# codepoint ranges. This is EXACT for ASCII + common Latin/Greek/Cyrillic/CJK
# and is FLAGGED as an approximation for rare scripts. See report SKEPTIC-BAIT.

@always_inline
def is_digit(cp: Int) -> Bool:
    # \p{N} approximation: ASCII digits only. The Qwen2 regex uses \p{N} which
    # matches a SINGLE numeral (no '+'); other-script digits are not covered.
    return cp >= 48 and cp <= 57


@always_inline
def is_letter(cp: Int) -> Bool:
    # \p{L} approximation covering the scripts the encoder realistically sees.
    if cp < 0x80:
        # ASCII letters
        return (cp >= 65 and cp <= 90) or (cp >= 97 and cp <= 122)
    # Latin-1 supplement letters: U+00C0..U+00FF excluding × (0xD7) and ÷ (0xF7)
    if cp >= 0x00C0 and cp <= 0x00FF:
        return cp != 0x00D7 and cp != 0x00F7
    if cp == 0x00AA or cp == 0x00B5 or cp == 0x00BA:
        return True
    # Latin Extended-A/B + IPA + spacing modifiers + Greek/Coptic/Cyrillic
    if cp >= 0x0100 and cp <= 0x02AF:
        return True
    if cp >= 0x0370 and cp <= 0x03FF:
        # Greek/Coptic letters (skip punctuation/symbols in this block roughly)
        return True
    if cp >= 0x0400 and cp <= 0x04FF:
        return True  # Cyrillic
    if cp >= 0x0530 and cp <= 0x058F:
        return True  # Armenian
    if cp >= 0x0590 and cp <= 0x05FF:
        return True  # Hebrew
    if cp >= 0x0600 and cp <= 0x06FF:
        return True  # Arabic
    if cp >= 0x0900 and cp <= 0x097F:
        return True  # Devanagari
    if cp >= 0x0E00 and cp <= 0x0E7F:
        return True  # Thai
    if cp >= 0x1100 and cp <= 0x11FF:
        return True  # Hangul Jamo
    if cp >= 0x3040 and cp <= 0x30FF:
        return True  # Hiragana + Katakana
    if cp >= 0x3400 and cp <= 0x4DBF:
        return True  # CJK Ext A
    if cp >= 0x4E00 and cp <= 0x9FFF:
        return True  # CJK Unified Ideographs
    if cp >= 0xAC00 and cp <= 0xD7A3:
        return True  # Hangul Syllables
    if cp >= 0xF900 and cp <= 0xFAFF:
        return True  # CJK Compatibility Ideographs
    if cp >= 0x20000 and cp <= 0x2FA1F:
        return True  # CJK Ext B..F + compat supplement
    return False


@always_inline
def is_whitespace(cp: Int) -> Bool:
    # \s for the regex engine: ASCII ws + Unicode separators the engine treats
    # as whitespace. Covers space/tab/newline/CR/FF/VT used by the prompts.
    if cp == 0x20 or cp == 0x09 or cp == 0x0A or cp == 0x0D or cp == 0x0B or cp == 0x0C:
        return True
    if cp == 0x85 or cp == 0xA0 or cp == 0x1680:
        return True
    if cp >= 0x2000 and cp <= 0x200A:
        return True
    if cp == 0x2028 or cp == 0x2029 or cp == 0x202F or cp == 0x205F or cp == 0x3000:
        return True
    return False


@always_inline
def is_newline(cp: Int) -> Bool:
    return cp == 0x0A or cp == 0x0D


# ----------------------------------------------------------------------------
# Byte -> Unicode-codepoint table (GPT-2 bytes_to_unicode)
# ----------------------------------------------------------------------------
def build_byte_to_unicode() -> List[Int]:
    # Reversible map: every byte 0..255 -> a printable Unicode codepoint, so the
    # byte stream becomes a string over a 256-char alphabet that the BPE vocab
    # is expressed in. Values match GPT-2/Qwen exactly.
    var table = List[Int]()
    for _ in range(256):
        table.append(0)
    # "printable" bytes map to themselves
    var n = 0
    for b in range(256):
        var printable = (b >= 33 and b <= 126) or (b >= 161 and b <= 172) or (b >= 174 and b <= 255)
        if printable:
            table[b] = b
        else:
            table[b] = 256 + n
            n += 1
    return table^


# ----------------------------------------------------------------------------
# Minimal JSON reader  (vocab object + merges array + added_tokens array)
# ----------------------------------------------------------------------------
# Only the escapes that actually occur in this tokenizer.json are handled:
#   \"  ->  "      \\  ->  \
# All vocab/merge tokens are composed solely of the 256 byte-level codepoints
# (<= U+0142), stored as literal UTF-8 in the file (no \u escapes). The parser
# returns each JSON string as a List[Int] of Unicode codepoints, which is the
# native key form for BPE lookups.

struct JsonString(Copyable, Movable):
    # A decoded JSON string represented as its Unicode codepoints.
    var cps: List[Int]

    def __init__(out self, var cps: List[Int]):
        self.cps = cps^


@always_inline
def _utf8_decode_at(bytes: Span[Byte, _], i: Int, mut cp: Int) -> Int:
    # Decode one UTF-8 codepoint starting at byte i. Returns #bytes consumed and
    # writes the codepoint into `cp`. Assumes valid UTF-8 (tokenizer.json is).
    var b0 = Int(bytes[i])
    if b0 < 0x80:
        cp = b0
        return 1
    elif (b0 & 0xE0) == 0xC0:
        cp = ((b0 & 0x1F) << 6) | (Int(bytes[i + 1]) & 0x3F)
        return 2
    elif (b0 & 0xF0) == 0xE0:
        cp = ((b0 & 0x0F) << 12) | ((Int(bytes[i + 1]) & 0x3F) << 6) | (Int(bytes[i + 2]) & 0x3F)
        return 3
    else:
        cp = (
            ((b0 & 0x07) << 18)
            | ((Int(bytes[i + 1]) & 0x3F) << 12)
            | ((Int(bytes[i + 2]) & 0x3F) << 6)
            | (Int(bytes[i + 3]) & 0x3F)
        )
        return 4


def _parse_json_string(bytes: Span[Byte, _], mut pos: Int) -> List[Int]:
    # `pos` must point at the opening quote. Advances past the closing quote.
    # Returns the decoded codepoints.
    var out = List[Int]()
    pos += 1  # skip opening quote
    var n = len(bytes)
    while pos < n:
        var b = Int(bytes[pos])
        if b == 0x22:  # closing quote "
            pos += 1
            return out^
        if b == 0x5C:  # backslash escape
            var e = Int(bytes[pos + 1])
            if e == 0x22:
                out.append(0x22); pos += 2; continue
            elif e == 0x5C:
                out.append(0x5C); pos += 2; continue
            elif e == 0x6E:  # \n
                out.append(0x0A); pos += 2; continue
            elif e == 0x74:  # \t
                out.append(0x09); pos += 2; continue
            elif e == 0x72:  # \r
                out.append(0x0D); pos += 2; continue
            elif e == 0x2F:  # \/
                out.append(0x2F); pos += 2; continue
            elif e == 0x75:  # \uXXXX (not expected in vocab, but handle)
                var hv = 0
                for k in range(4):
                    var hc = Int(bytes[pos + 2 + k])
                    var d = 0
                    if hc >= 48 and hc <= 57: d = hc - 48
                    elif hc >= 97 and hc <= 102: d = hc - 87
                    elif hc >= 65 and hc <= 70: d = hc - 55
                    hv = hv * 16 + d
                out.append(hv); pos += 6; continue
            else:
                out.append(e); pos += 2; continue
        # raw (possibly multi-byte UTF-8) codepoint
        var cp = 0
        var adv = _utf8_decode_at(bytes, pos, cp)
        out.append(cp)
        pos += adv
    return out^


def _find_key(bytes: Span[Byte, _], key: String, start: Int) -> Int:
    # Find the byte offset of a top-of-section quoted key like `"vocab":`.
    # Returns the index just past the colon, or -1.
    var needle = String('"') + key + String('"')
    # naive substring search over bytes
    var nb = needle.as_bytes()
    var nlen = len(nb)
    var n = len(bytes)
    var i = start
    while i + nlen <= n:
        var matched = True
        for j in range(nlen):
            if bytes[i + j] != nb[j]:
                matched = False
                break
        if matched:
            # advance past key, whitespace, colon
            var p = i + nlen
            while p < n and (bytes[p] == 0x20 or bytes[p] == 0x09 or bytes[p] == 0x0A or bytes[p] == 0x0D):
                p += 1
            if p < n and bytes[p] == 0x3A:  # ':'
                return p + 1
        i += 1
    return -1


@always_inline
def _skip_ws(bytes: Span[Byte, _], mut pos: Int):
    var n = len(bytes)
    while pos < n:
        var b = Int(bytes[pos])
        if b == 0x20 or b == 0x09 or b == 0x0A or b == 0x0D:
            pos += 1
        else:
            return


def _parse_int(bytes: Span[Byte, _], mut pos: Int) -> Int:
    var n = len(bytes)
    var neg = False
    if pos < n and bytes[pos] == 0x2D:
        neg = True; pos += 1
    var v = 0
    while pos < n:
        var b = Int(bytes[pos])
        if b >= 48 and b <= 57:
            v = v * 10 + (b - 48)
            pos += 1
        else:
            break
    return -v if neg else v


# ----------------------------------------------------------------------------
# codepoint-list key helpers for Dict
# ----------------------------------------------------------------------------
def _cps_to_key(cps: List[Int]) -> String:
    # Pack a codepoint list into a String usable as a Dict key. We join the
    # decimal codepoints with a separator that cannot appear (','), giving a
    # collision-free, hashable key without needing the codepoints to be valid
    # UTF-8 (they always are <= U+0142 here, but this stays robust).
    var s = String("")
    for i in range(len(cps)):
        if i != 0:
            s += String(",")
        s += String(cps[i])
    return s^


# ----------------------------------------------------------------------------
# The tokenizer
# ----------------------------------------------------------------------------
struct _Segment(Copyable, Movable):
    var text: String
    var is_special: Bool
    var special_id: Int

    def __init__(out self, var text: String, is_special: Bool, special_id: Int):
        self.text = text^
        self.is_special = is_special
        self.special_id = special_id


struct Qwen3Tokenizer(Movable):
    # vocab: byte-level token (as codepoint-key string) -> id
    var vocab: Dict[String, Int]
    # merge rank: "tokenA|tokenB" (codepoint-key strings joined by 0x1F) -> rank
    var merge_rank: Dict[String, Int]
    # added/special tokens: literal content string -> id
    var special_tokens: List[String]
    var special_ids: List[Int]
    # byte -> unicode codepoint table
    var byte_to_uni: List[Int]
    # reverse vocab for decode: id -> codepoint-key string (built lazily)
    var id_to_token: Dict[Int, String]

    def __init__(out self, json_path: String) raises:
        self.vocab = Dict[String, Int]()
        self.merge_rank = Dict[String, Int]()
        self.special_tokens = List[String]()
        self.special_ids = List[Int]()
        self.byte_to_uni = build_byte_to_unicode()
        self.id_to_token = Dict[Int, String]()

        # Read the tokenizer JSON via ffi (NOT std Path.read_text — its builtin
        # `open` collides with the lib's external_call open; see io/ffi.sys_open).
        var fd = sys_open(json_path, O_RDONLY)
        if fd < 0:
            raise Error(String("tokenizer: cannot open ") + json_path)
        var fsz = file_size(fd)
        var rbuf = alloc[UInt8](fsz)
        var rdone = 0
        while rdone < fsz:
            var got = sys_pread(fd, rbuf + rdone, fsz - rdone, rdone)
            if got <= 0:
                break
            rdone += got
        _ = sys_close(fd)
        var jbytes = List[UInt8](capacity=fsz)
        for i in range(fsz):
            jbytes.append(rbuf[i])
        rbuf.free()
        var text = String(unsafe_from_utf8=jbytes)
        var bytes = text.as_bytes()

        self._parse_vocab(bytes)
        self._parse_merges(bytes)
        self._parse_added_tokens(bytes)

    def _parse_vocab(mut self, bytes: Span[Byte, _]) raises:
        var p = _find_key(bytes, String("vocab"), 0)
        if p < 0:
            return
        _skip_ws(bytes, p)
        # expect '{'
        if bytes[p] != 0x7B:
            return
        p += 1
        var n = len(bytes)
        while p < n:
            _skip_ws(bytes, p)
            if bytes[p] == 0x7D:  # '}'
                p += 1
                return
            if bytes[p] == 0x2C:  # ','
                p += 1
                continue
            # key string
            var key_cps = _parse_json_string(bytes, p)
            _skip_ws(bytes, p)
            # ':'
            if bytes[p] == 0x3A:
                p += 1
            _skip_ws(bytes, p)
            var id = _parse_int(bytes, p)
            var k = _cps_to_key(key_cps)
            self.vocab[k] = id
            self.id_to_token[id] = k

    def _parse_merges(mut self, bytes: Span[Byte, _]) raises:
        var p = _find_key(bytes, String("merges"), 0)
        if p < 0:
            return
        _skip_ws(bytes, p)
        if bytes[p] != 0x5B:  # '['
            return
        p += 1
        var n = len(bytes)
        var rank = 0
        while p < n:
            _skip_ws(bytes, p)
            if bytes[p] == 0x5D:  # ']'
                p += 1
                return
            if bytes[p] == 0x2C:
                p += 1
                continue
            if bytes[p] == 0x5B:  # inner '['  -> a ["a","b"] pair
                p += 1
                _skip_ws(bytes, p)
                var a = _parse_json_string(bytes, p)
                _skip_ws(bytes, p)
                if bytes[p] == 0x2C:
                    p += 1
                _skip_ws(bytes, p)
                var b = _parse_json_string(bytes, p)
                _skip_ws(bytes, p)
                if bytes[p] == 0x5D:  # inner ']'
                    p += 1
                var mk = _cps_to_key(a) + String(chr(0x1F)) + _cps_to_key(b)
                self.merge_rank[mk] = rank
                rank += 1
            else:
                # unexpected; bail
                return

    def _parse_added_tokens(mut self, bytes: Span[Byte, _]) raises:
        # added_tokens is an array of objects with "id" and "content".
        var p = _find_key(bytes, String("added_tokens"), 0)
        if p < 0:
            return
        _skip_ws(bytes, p)
        if bytes[p] != 0x5B:
            return
        p += 1
        var n = len(bytes)
        while p < n:
            _skip_ws(bytes, p)
            if bytes[p] == 0x5D:
                p += 1
                return
            if bytes[p] == 0x2C:
                p += 1
                continue
            if bytes[p] == 0x7B:  # '{'  one token object
                # scan inside for "id" and "content"
                var depth = 1
                p += 1
                var tok_id = -1
                var content_cps = List[Int]()
                while p < n and depth > 0:
                    _skip_ws(bytes, p)
                    var b = Int(bytes[p])
                    if b == 0x7D:
                        depth -= 1; p += 1; continue
                    if b == 0x2C:
                        p += 1; continue
                    if b == 0x22:  # a key
                        var key = _parse_json_string(bytes, p)
                        _skip_ws(bytes, p)
                        if bytes[p] == 0x3A:
                            p += 1
                        _skip_ws(bytes, p)
                        # value
                        var kk = _cps_to_key(key)
                        if kk == _cps_to_key(_str_to_cps(String("id"))):
                            tok_id = _parse_int(bytes, p)
                        elif kk == _cps_to_key(_str_to_cps(String("content"))):
                            content_cps = _parse_json_string(bytes, p)
                        else:
                            # skip a scalar value (string / bool / int)
                            if bytes[p] == 0x22:
                                _ = _parse_json_string(bytes, p)
                            else:
                                # bool/int/null token
                                while p < n:
                                    var c = Int(bytes[p])
                                    if c == 0x2C or c == 0x7D:
                                        break
                                    p += 1
                    else:
                        p += 1
                if tok_id >= 0 and len(content_cps) > 0:
                    self.special_tokens.append(_cps_to_string(content_cps))
                    self.special_ids.append(tok_id)
            else:
                p += 1

    # ------------------------------------------------------------------------
    # Pre-tokenizer scanner: replicate the Qwen2 regex, leftmost-first.
    # Operates on the original-text codepoints. Returns a list of pre-token
    # spans (each a List[Int] of codepoints).
    # ------------------------------------------------------------------------
    def _pretokenize(self, cps: List[Int]) -> List[List[Int]]:
        var out = List[List[Int]]()
        var n = len(cps)
        var i = 0
        while i < n:
            # 1) contractions (?i:'s|'t|'re|'ve|'m|'ll|'d)  -- apostrophe = 0x27
            if cps[i] == 0x27 and i + 1 < n:
                var c1 = _lower(cps[i + 1])
                var matched = 0
                if c1 == ord('s') or c1 == ord('t') or c1 == ord('m') or c1 == ord('d'):
                    matched = 2
                elif c1 == ord('r') and i + 2 < n and _lower(cps[i + 2]) == ord('e'):
                    matched = 3
                elif c1 == ord('v') and i + 2 < n and _lower(cps[i + 2]) == ord('e'):
                    matched = 3
                elif c1 == ord('l') and i + 2 < n and _lower(cps[i + 2]) == ord('l'):
                    matched = 3
                if matched > 0:
                    var seg = List[Int]()
                    for k in range(matched):
                        seg.append(cps[i + k])
                    out.append(seg^)
                    i += matched
                    continue

            # 2) [^\r\n\p{L}\p{N}]? \p{L}+
            #    optional single non-(\r\n\p{L}\p{N}) char, then >=1 letters
            var j = i
            var took_prefix = False
            if not is_newline(cps[j]) and not is_letter(cps[j]) and not is_digit(cps[j]):
                # candidate prefix; only valid if a letter follows
                if j + 1 < n and is_letter(cps[j + 1]):
                    took_prefix = True
                    j += 1
            if is_letter(cps[j]):
                var k = j
                while k < n and is_letter(cps[k]):
                    k += 1
                var seg = List[Int]()
                for m in range(i, k):
                    seg.append(cps[m])
                out.append(seg^)
                i = k
                continue
            # if we tentatively took a prefix but no letters, undo (j unused)
            _ = took_prefix

            # 3) \p{N}  (single digit)
            if is_digit(cps[i]):
                var seg = List[Int]()
                seg.append(cps[i])
                out.append(seg^)
                i += 1
                continue

            # 4)  ?[^\s\p{L}\p{N}]+[\r\n]*
            #    optional single leading space, then >=1 non-(space/letter/digit),
            #    then any trailing \r\n run.
            var s = i
            if cps[s] == 0x20 and s + 1 < n and not is_whitespace(cps[s + 1]) and not is_letter(cps[s + 1]) and not is_digit(cps[s + 1]):
                # leading space allowed only if a punct char follows
                s += 1
            if not is_whitespace(cps[s]) and not is_letter(cps[s]) and not is_digit(cps[s]):
                var k = s
                while k < n and not is_whitespace(cps[k]) and not is_letter(cps[k]) and not is_digit(cps[k]):
                    k += 1
                # trailing [\r\n]*
                while k < n and is_newline(cps[k]):
                    k += 1
                var seg = List[Int]()
                for m in range(i, k):
                    seg.append(cps[m])
                out.append(seg^)
                i = k
                continue

            # 5) \s*[\r\n]+   (whitespace run that contains a newline)
            #    greedily: any \s, but must include >=1 newline; matches \s* then [\r\n]+
            #    Implementation: scan a whitespace run; if it contains a newline,
            #    the match ends at the LAST newline in the leading \s*[\r\n]+ block.
            if is_whitespace(cps[i]):
                # find extent of whitespace run
                var we = i
                while we < n and is_whitespace(cps[we]):
                    we += 1
                # does the run contain a newline?
                var last_nl = -1
                for m in range(i, we):
                    if is_newline(cps[m]):
                        last_nl = m
                if last_nl >= 0:
                    # \s*[\r\n]+ matches from i up to and including the last
                    # contiguous newline block. Per regex: \s* (any ws incl nl)
                    # then [\r\n]+ . Greedy \s* consumes through the final newline;
                    # trailing non-newline ws after the last newline is NOT part
                    # of this match (it falls to branch 6/7).
                    var end = last_nl + 1
                    var seg = List[Int]()
                    for m in range(i, end):
                        seg.append(cps[m])
                    out.append(seg^)
                    i = end
                    continue
                else:
                    # 6) \s+(?!\S)  : trailing whitespace run (no non-ws follows)
                    # 7) \s+        : otherwise leave ONE trailing ws for the next
                    #                 token's optional leading space.
                    if we == n:
                        # whole rest is whitespace -> \s+(?!\S) takes it all
                        var seg = List[Int]()
                        for m in range(i, we):
                            seg.append(cps[m])
                        out.append(seg^)
                        i = we
                        continue
                    else:
                        # non-ws follows -> \s+ takes the run minus the last ws
                        var end = we - 1
                        if end <= i:
                            # single ws followed by non-ws: \s+ matches just it
                            end = i + 1
                            # but if that single ws will be the next token's
                            # leading space, \s+ still must match >=1: it matches
                            # the one space. (Matches oracle 'x  y' -> x,' ',' y'
                            # where the middle two spaces: first by \s+, ...).
                        var seg = List[Int]()
                        for m in range(i, end):
                            seg.append(cps[m])
                        out.append(seg^)
                        i = end
                        continue

            # Fallback (should not happen): emit one codepoint.
            var seg = List[Int]()
            seg.append(cps[i])
            out.append(seg^)
            i += 1

        return out^

    # ------------------------------------------------------------------------
    # BPE over one pre-token's byte-level codepoints.
    # ------------------------------------------------------------------------
    def _bpe(self, units: List[String]) raises -> List[String]:
        # `units` is a list of single byte-level tokens (each a codepoint-key
        # string, here each representing exactly ONE unicode char). Greedily
        # merge the lowest-rank adjacent pair until no merge applies.
        var word = units.copy()
        if len(word) < 2:
            return word^
        while True:
            var best_rank = -1
            var best_i = -1
            for i in range(len(word) - 1):
                var mk = word[i] + String(chr(0x1F)) + word[i + 1]
                if mk in self.merge_rank:
                    var r = self.merge_rank[mk]
                    if best_rank == -1 or r < best_rank:
                        best_rank = r
                        best_i = i
            if best_i == -1:
                break
            # merge at best_i (lowest rank). On rank ties we pick the FIRST (lowest
            # index) occurrence, matching the HF/GPT-2 reference loop.
            # reconstruct codepoint-key concat: keys are "c1,c2,..." decimal lists
            var merged = _merge_keys(word[best_i], word[best_i + 1])
            var new_word = List[String]()
            for k in range(len(word)):
                if k == best_i:
                    new_word.append(merged)
                elif k == best_i + 1:
                    continue
                else:
                    new_word.append(word[k])
            word = new_word^
        return word^

    # ------------------------------------------------------------------------
    # encode
    # ------------------------------------------------------------------------
    def encode(self, text: String) raises -> List[Int]:
        var ids = List[Int]()
        # NFC normalization is approximated as a no-op pass-through. This is
        # EXACT for NFC-stable input (all ASCII + precomposed accents); FLAGGED
        # for decomposed input (combining marks). See report.
        # Split out special tokens first (split_special_tokens=false: specials
        # are matched as atomic units when present in the text).
        var segments = self._split_on_specials(text)
        for seg_idx in range(len(segments)):
            ref seg = segments[seg_idx]
            if seg.is_special:
                ids.append(seg.special_id)
                continue
            # ordinary text segment -> codepoints -> pretokenize -> bytelevel -> bpe
            var cps = _str_to_cps(seg.text)
            var pretoks = self._pretokenize(cps)
            for pt_idx in range(len(pretoks)):
                ref pt = pretoks[pt_idx]
                # byte-level expand: each codepoint -> its UTF-8 bytes -> mapped
                # to byte-level unicode codepoints, one unit per byte.
                var units = List[String]()
                for ci in range(len(pt)):
                    var b = List[Int]()
                    _cp_to_utf8(pt[ci], b)
                    for bi in range(len(b)):
                        var mapped = self.byte_to_uni[b[bi]]
                        var one = List[Int]()
                        one.append(mapped)
                        units.append(_cps_to_key(one))
                var merged = self._bpe(units)
                for mi in range(len(merged)):
                    if merged[mi] in self.vocab:
                        ids.append(self.vocab[merged[mi]])
                    # (byte_fallback=False, but every single byte-level char is
                    #  in vocab, so unmatched should never occur.)
        return ids^

    def _split_on_specials(self, text: String) raises -> List[_Segment]:
        # Find earliest-occurring special token at each scan position. Specials
        # are matched as literal substrings (normalized=false, special=true).
        var out = List[_Segment]()
        var cps = _str_to_cps(text)
        # Precompute special token codepoint lists.
        var spec_cps = List[List[Int]]()
        for s in range(len(self.special_tokens)):
            spec_cps.append(_str_to_cps(self.special_tokens[s]))
        var n = len(cps)
        var i = 0
        var buf = List[Int]()
        while i < n:
            var hit = -1
            var hit_len = 0
            # try each special; pick the longest match starting at i (tie ->
            # first by list order, which is id order)
            for s in range(len(spec_cps)):
                ref sc = spec_cps[s]
                var L = len(sc)
                if L == 0 or i + L > n:
                    continue
                var ok = True
                for k in range(L):
                    if cps[i + k] != sc[k]:
                        ok = False
                        break
                if ok and L > hit_len:
                    hit = s
                    hit_len = L
            if hit >= 0:
                if len(buf) > 0:
                    out.append(_Segment(_cps_to_string(buf), False, -1))
                    buf = List[Int]()
                out.append(_Segment(self.special_tokens[hit], True, self.special_ids[hit]))
                i += hit_len
            else:
                buf.append(cps[i])
                i += 1
        if len(buf) > 0:
            out.append(_Segment(_cps_to_string(buf), False, -1))
        return out^

    # ------------------------------------------------------------------------
    # decode (nice-to-have): id -> byte-level token -> bytes -> UTF-8 string
    # ------------------------------------------------------------------------
    def decode(self, ids: List[Int]) raises -> String:
        # Build reverse map for specials.
        var out_bytes = List[Byte]()
        # inverse byte_to_uni: codepoint -> byte
        var inv = Dict[Int, Int]()
        for b in range(256):
            inv[self.byte_to_uni[b]] = b
        for idx in range(len(ids)):
            var id = ids[idx]
            # special?
            var is_spec = False
            for s in range(len(self.special_ids)):
                if self.special_ids[s] == id:
                    var sb = self.special_tokens[s].as_bytes()
                    for bi in range(len(sb)):
                        out_bytes.append(sb[bi])
                    is_spec = True
                    break
            if is_spec:
                continue
            if id in self.id_to_token:
                # id_to_token value is a codepoint-key string "c1,c2,..."
                var key = self.id_to_token[id]
                var cps = _key_to_cps(key)
                for c in range(len(cps)):
                    if cps[c] in inv:
                        out_bytes.append(Byte(inv[cps[c]]))
        # out_bytes is a UTF-8 stream
        return String(unsafe_from_utf8=out_bytes)


# ----------------------------------------------------------------------------
# free helpers
# ----------------------------------------------------------------------------
@always_inline
def _lower(cp: Int) -> Int:
    if cp >= 65 and cp <= 90:
        return cp + 32
    return cp


def _str_to_cps(s: String) -> List[Int]:
    var out = List[Int]()
    for cp in s.codepoints():
        out.append(Int(cp))
    return out^


def _cp_to_utf8(cp: Int, mut out: List[Int]):
    if cp < 0x80:
        out.append(cp)
    elif cp < 0x800:
        out.append(0xC0 | (cp >> 6))
        out.append(0x80 | (cp & 0x3F))
    elif cp < 0x10000:
        out.append(0xE0 | (cp >> 12))
        out.append(0x80 | ((cp >> 6) & 0x3F))
        out.append(0x80 | (cp & 0x3F))
    else:
        out.append(0xF0 | (cp >> 18))
        out.append(0x80 | ((cp >> 12) & 0x3F))
        out.append(0x80 | ((cp >> 6) & 0x3F))
        out.append(0x80 | (cp & 0x3F))


def _cps_to_string(cps: List[Int]) -> String:
    var b = List[Byte]()
    for i in range(len(cps)):
        var u = List[Int]()
        _cp_to_utf8(cps[i], u)
        for k in range(len(u)):
            b.append(Byte(u[k]))
    return String(unsafe_from_utf8=b)


def _key_to_cps(key: String) -> List[Int]:
    # inverse of _cps_to_key: parse "c1,c2,..." decimal list
    var out = List[Int]()
    var cur = 0
    var have = False
    var kb = key.as_bytes()
    for i in range(len(kb)):
        var c = Int(kb[i])
        if c >= 48 and c <= 57:
            cur = cur * 10 + (c - 48)
            have = True
        elif c == 0x2C:  # ','
            if have:
                out.append(cur)
            cur = 0
            have = False
    if have:
        out.append(cur)
    return out^


def _merge_keys(a: String, b: String) -> String:
    # concat two codepoint-key strings into one codepoint-key string
    return a + String(",") + b
