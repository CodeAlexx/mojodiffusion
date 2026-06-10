# Pure-Mojo CLIP BPE tokenizer (text -> token ids) for CLIP-L and CLIP-G.
#
# Drives real prompt conditioning for SDXL / SD3.5 / FLUX, which all use one or
# two CLIP text encoders. CPU-only string processing: no Python, no Rust crate.
# tokenizer.json is parsed ONCE at load by a minimal hand-rolled JSON reader.
#
# Source of truth (replicated EXACTLY where stated):
#   openai/clip-vit-large-patch14            (CLIP-L, vocab 49408)
#   laion/CLIP-ViT-bigG-14-laion2B-39B-b160k (CLIP-G, same 49408 BPE algorithm)
#   tokenizer.json: model.type = BPE, end_of_word_suffix = "</w>",
#                   continuing_subword_prefix = "", unk = <|endoftext|>,
#                   BOS <|startoftext|> = 49406, EOS <|endoftext|> = 49407.
#
# This mirrors HuggingFace `transformers.CLIPTokenizer` (the SLOW tokenizer used
# by SDXL/SD3/FLUX pipelines), whose `_tokenize` is:
#     text = whitespace_clean(fix_text(text)).lower()
#     for token in re.findall(PAT, text):
#         token = "".join(byte_encoder[b] for b in token.encode("utf-8"))
#         bpe_tokens += bpe(token).split(" ")
#   PAT = <\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d
#         |[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+        (re.IGNORECASE)
#   bpe(token): word = tuple(token[:-1]) + (token[-1] + "</w>",); greedily merge
#               the lowest-rank adjacent bigram (all occurrences per round).
#   Final ids wrapped with BOS ... EOS (RobertaProcessing post_processor).
#
# ============================ FIDELITY NOTES ===============================
# FAITHFULLY REPLICATED (bit-exact vs HF for ASCII / Latin text):
#   * whitespace_clean: collapse runs of \s -> single space, strip ends.
#   * lowercase before byte-encoding.
#   * CLIP word-split regex, in alternation order, incl. the 7 contractions,
#     letter-runs [\p{L}]+, SINGLE-digit [\p{N}] (so "33" -> "3","3"), and
#     punctuation runs [^\s\p{L}\p{N}]+.
#   * GPT-2 bytes_to_unicode byte map (every UTF-8 byte -> one printable cp).
#   * BPE with the </w> end-of-word marker on the last symbol of each word,
#     lowest-rank-first, merge-all-occurrences-per-round (HF bpe() semantics).
#   * BOS=49406 prepend, EOS=49407 append, NO padding (caller pads to 77).
#
# APPROXIMATED (possible divergence vs HF on non-ASCII input -- PARITY RISK):
#   * fix_text / ftfy: NOT run. HF's ftfy path fixes mojibake; for clean
#     ASCII/UTF-8 prompts this is a no-op, so it does not affect the common
#     case. (If HF was installed WITHOUT ftfy it uses BasicTokenizer instead;
#     for ASCII both paths agree with this implementation.)
#   * Unicode \p{L} / \p{N}: approximated by codepoint ranges (see _clip_is_*).
#     EXACT for ASCII + common Latin/Greek/Cyrillic/CJK; rare scripts may split
#     differently.
#   * Unicode NFC normalization: pass-through (no-op). EXACT for NFC-stable
#     input (all ASCII + precomposed accents); decomposed combining marks differ.
#   * Lowercasing: ASCII A-Z plus Latin-1 supplement A-grave..Thorn. Full
#     Unicode case-folding (Greek/Cyrillic/etc. uppercase) is NOT applied.
#   * Literal <|startoftext|>/<|endoftext|> appearing INSIDE the user text are
#     NOT matched as atomic regex alternatives (they are byte-encoded like any
#     other run). Real prompts never contain them; BOS/EOS are added explicitly.
# ===========================================================================

from std.memory import alloc
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY


# ----------------------------------------------------------------------------
# Unicode category approximation  (\p{L}, \p{N}, \s) -- copies of the shared
# tokenizer.mojo helpers, kept local so this file stands alone.
# ----------------------------------------------------------------------------
@always_inline
def _clip_is_digit(cp: Int) -> Bool:
    # \p{N} approximation: ASCII digits. The CLIP regex matches a SINGLE numeral.
    return cp >= 48 and cp <= 57


@always_inline
def _clip_is_letter(cp: Int) -> Bool:
    # \p{L} approximation covering the scripts the encoder realistically sees.
    if cp < 0x80:
        return (cp >= 65 and cp <= 90) or (cp >= 97 and cp <= 122)
    if cp >= 0x00C0 and cp <= 0x00FF:
        return cp != 0x00D7 and cp != 0x00F7
    if cp == 0x00AA or cp == 0x00B5 or cp == 0x00BA:
        return True
    if cp >= 0x0100 and cp <= 0x02AF:
        return True
    if cp >= 0x0370 and cp <= 0x03FF:
        return True  # Greek/Coptic
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
        return True  # CJK Ext B..F
    return False


@always_inline
def _clip_is_whitespace(cp: Int) -> Bool:
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
def _clip_lower(cp: Int) -> Int:
    # ASCII A-Z + Latin-1 supplement uppercase (A-grave..Thorn, excluding U+00D7).
    if cp >= 65 and cp <= 90:
        return cp + 32
    if cp >= 0x00C0 and cp <= 0x00DE and cp != 0x00D7:
        return cp + 32
    return cp


# ----------------------------------------------------------------------------
# Byte -> Unicode-codepoint table (GPT-2 bytes_to_unicode)
# ----------------------------------------------------------------------------
def _clip_byte_to_unicode() -> List[Int]:
    var table = List[Int]()
    for _ in range(256):
        table.append(0)
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
# UTF-8 helpers
# ----------------------------------------------------------------------------
@always_inline
def _clip_utf8_decode_at(bytes: Span[Byte, _], i: Int, mut cp: Int) -> Int:
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


def _clip_cp_to_utf8(cp: Int, mut out: List[Int]):
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


def _clip_cps_to_string(cps: List[Int]) -> String:
    var b = List[Byte]()
    for i in range(len(cps)):
        var u = List[Int]()
        _clip_cp_to_utf8(cps[i], u)
        for k in range(len(u)):
            b.append(Byte(u[k]))
    return String(unsafe_from_utf8=b)


def _clip_str_to_cps(s: String) -> List[Int]:
    var out = List[Int]()
    for cp in s.codepoints():
        out.append(Int(cp))
    return out^


# ----------------------------------------------------------------------------
# Minimal JSON reader  (vocab object + merges array)
# ----------------------------------------------------------------------------
def _clip_parse_json_string(bytes: Span[Byte, _], mut pos: Int) -> List[Int]:
    # `pos` must point at the opening quote. Advances past the closing quote.
    var out = List[Int]()
    pos += 1
    var n = len(bytes)
    while pos < n:
        var b = Int(bytes[pos])
        if b == 0x22:  # closing quote
            pos += 1
            return out^
        if b == 0x5C:  # backslash escape
            var e = Int(bytes[pos + 1])
            if e == 0x22:
                out.append(0x22); pos += 2; continue
            elif e == 0x5C:
                out.append(0x5C); pos += 2; continue
            elif e == 0x6E:
                out.append(0x0A); pos += 2; continue
            elif e == 0x74:
                out.append(0x09); pos += 2; continue
            elif e == 0x72:
                out.append(0x0D); pos += 2; continue
            elif e == 0x2F:
                out.append(0x2F); pos += 2; continue
            elif e == 0x75:  # \uXXXX
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
        var cp = 0
        var adv = _clip_utf8_decode_at(bytes, pos, cp)
        out.append(cp)
        pos += adv
    return out^


def _clip_find_key(bytes: Span[Byte, _], key: String, start: Int) -> Int:
    var needle = String('"') + key + String('"')
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
            var p = i + nlen
            while p < n and (bytes[p] == 0x20 or bytes[p] == 0x09 or bytes[p] == 0x0A or bytes[p] == 0x0D):
                p += 1
            if p < n and bytes[p] == 0x3A:
                return p + 1
        i += 1
    return -1


@always_inline
def _clip_skip_ws(bytes: Span[Byte, _], mut pos: Int):
    var n = len(bytes)
    while pos < n:
        var b = Int(bytes[pos])
        if b == 0x20 or b == 0x09 or b == 0x0A or b == 0x0D:
            pos += 1
        else:
            return


def _clip_parse_int(bytes: Span[Byte, _], mut pos: Int) -> Int:
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


def _clip_read_utf8_file(path: String) raises -> String:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("clip_tokenizer: cannot open ") + path)
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
    return String(unsafe_from_utf8=jbytes)


# ----------------------------------------------------------------------------
# The CLIP tokenizer
# ----------------------------------------------------------------------------
struct ClipTokenizer(Movable):
    # vocab: byte-level BPE token (literal string, e.g. "cat</w>") -> id
    var vocab: Dict[String, Int]
    # merge rank: "left\x1Fright" (literal strings) -> rank
    var merge_rank: Dict[String, Int]
    # byte -> unicode codepoint table (GPT-2)
    var byte_to_uni: List[Int]
    var bos_id: Int
    var eos_id: Int

    def __init__(out self, json_path: String) raises:
        self.vocab = Dict[String, Int]()
        self.merge_rank = Dict[String, Int]()
        self.byte_to_uni = _clip_byte_to_unicode()
        self.bos_id = 49406
        self.eos_id = 49407

        var text = _clip_read_utf8_file(json_path)
        var bytes = text.as_bytes()
        self._parse_vocab(bytes)
        self._parse_merges(bytes)

        # Prefer ids straight from the vocab if present (robust across CLIP-L/G).
        var bos_key = String("<|startoftext|>")
        var eos_key = String("<|endoftext|>")
        if bos_key in self.vocab:
            self.bos_id = self.vocab[bos_key]
        if eos_key in self.vocab:
            self.eos_id = self.vocab[eos_key]

    def _parse_vocab(mut self, bytes: Span[Byte, _]) raises:
        var p = _clip_find_key(bytes, String("vocab"), 0)
        if p < 0:
            return
        _clip_skip_ws(bytes, p)
        if bytes[p] != 0x7B:  # '{'
            return
        p += 1
        var n = len(bytes)
        while p < n:
            _clip_skip_ws(bytes, p)
            if bytes[p] == 0x7D:  # '}'
                p += 1
                return
            if bytes[p] == 0x2C:  # ','
                p += 1
                continue
            var key_cps = _clip_parse_json_string(bytes, p)
            _clip_skip_ws(bytes, p)
            if bytes[p] == 0x3A:  # ':'
                p += 1
            _clip_skip_ws(bytes, p)
            var id = _clip_parse_int(bytes, p)
            self.vocab[_clip_cps_to_string(key_cps)] = id

    def _parse_merges(mut self, bytes: Span[Byte, _]) raises:
        var p = _clip_find_key(bytes, String("merges"), 0)
        if p < 0:
            return
        _clip_skip_ws(bytes, p)
        if bytes[p] != 0x5B:  # '['
            return
        p += 1
        var n = len(bytes)
        var rank = 0
        var sep = String(chr(0x1F))
        while p < n:
            _clip_skip_ws(bytes, p)
            if bytes[p] == 0x5D:  # ']'
                p += 1
                return
            if bytes[p] == 0x2C:
                p += 1
                continue
            if bytes[p] == 0x5B:  # inner '[' -> ["a","b"] pair form
                p += 1
                _clip_skip_ws(bytes, p)
                var a = _clip_parse_json_string(bytes, p)
                _clip_skip_ws(bytes, p)
                if bytes[p] == 0x2C:
                    p += 1
                _clip_skip_ws(bytes, p)
                var b = _clip_parse_json_string(bytes, p)
                _clip_skip_ws(bytes, p)
                if bytes[p] == 0x5D:
                    p += 1
                var mk = _clip_cps_to_string(a) + sep + _clip_cps_to_string(b)
                if mk not in self.merge_rank:
                    self.merge_rank[mk] = rank
                rank += 1
            elif bytes[p] == 0x22:  # "left right" string form (CLIP tokenizer.json)
                var pair = _clip_parse_json_string(bytes, p)
                # split on FIRST ASCII space (0x20); symbols never contain 0x20.
                var spos = -1
                for k in range(len(pair)):
                    if pair[k] == 0x20:
                        spos = k
                        break
                if spos > 0 and spos < len(pair) - 1:
                    var la = List[Int]()
                    for k in range(spos):
                        la.append(pair[k])
                    var rb = List[Int]()
                    for k in range(spos + 1, len(pair)):
                        rb.append(pair[k])
                    var mk = _clip_cps_to_string(la) + sep + _clip_cps_to_string(rb)
                    if mk not in self.merge_rank:
                        self.merge_rank[mk] = rank
                rank += 1
            else:
                return

    # ------------------------------------------------------------------------
    # normalization: whitespace_clean(text).lower()
    # ------------------------------------------------------------------------
    def _normalize(self, cps: List[Int]) -> List[Int]:
        var out = List[Int]()
        var pending_space = False
        for i in range(len(cps)):
            var cp = cps[i]
            if _clip_is_whitespace(cp):
                pending_space = True
            else:
                if pending_space and len(out) > 0:
                    out.append(0x20)
                pending_space = False
                out.append(_clip_lower(cp))
        return out^

    # ------------------------------------------------------------------------
    # CLIP word-split regex emulation (alternation order preserved)
    # ------------------------------------------------------------------------
    def _split(self, cps: List[Int]) -> List[List[Int]]:
        var toks = List[List[Int]]()
        var n = len(cps)
        var i = 0
        while i < n:
            var cp = cps[i]
            # 1) contractions: 's|'t|'re|'ve|'m|'ll|'d
            if cp == 0x27:  # apostrophe
                var c1 = cps[i + 1] if i + 1 < n else -1
                var c2 = cps[i + 2] if i + 2 < n else -1
                var clen = 0
                if c1 == 0x73 or c1 == 0x74 or c1 == 0x6D or c1 == 0x64:
                    clen = 2  # 's 't 'm 'd
                elif c1 == 0x72 and c2 == 0x65:
                    clen = 3  # 're
                elif c1 == 0x76 and c2 == 0x65:
                    clen = 3  # 've
                elif c1 == 0x6C and c2 == 0x6C:
                    clen = 3  # 'll
                if clen > 0:
                    var t = List[Int]()
                    for k in range(clen):
                        t.append(cps[i + k])
                    toks.append(t^)
                    i += clen
                    continue
                # else: apostrophe handled by punctuation rule below
            # 2) letter run: [\p{L}]+
            if _clip_is_letter(cp):
                var t = List[Int]()
                while i < n and _clip_is_letter(cps[i]):
                    t.append(cps[i])
                    i += 1
                toks.append(t^)
                continue
            # 3) single digit: [\p{N}]
            if _clip_is_digit(cp):
                var t = List[Int]()
                t.append(cp)
                toks.append(t^)
                i += 1
                continue
            # 4) whitespace: not captured by any alternative -> skip
            if _clip_is_whitespace(cp):
                i += 1
                continue
            # 5) punctuation / other run: [^\s\p{L}\p{N}]+
            # HF's `regex` engine matches this greedily, so an apostrophe inside a
            # punctuation run is swallowed here (e.g. "x!'s" -> "x","!'","s"); the
            # contraction branch above only fires when ' follows a letter/digit/space.
            var t = List[Int]()
            while i < n and not _clip_is_whitespace(cps[i]) and not _clip_is_letter(cps[i]) and not _clip_is_digit(cps[i]):
                t.append(cps[i])
                i += 1
            toks.append(t^)
        return toks^

    # ------------------------------------------------------------------------
    # BPE on one pre-token (HF CLIPTokenizer.bpe semantics)
    # `chars`: list of single-codepoint byte-level strings.
    # ------------------------------------------------------------------------
    def _bpe_word(self, chars: List[String]) raises -> List[String]:
        var n = len(chars)
        var word = List[String]()
        if n == 0:
            return word^
        for i in range(n - 1):
            word.append(chars[i])
        word.append(chars[n - 1] + String("</w>"))
        if len(word) < 2:
            return word^  # token + "</w>"
        var sep = String(chr(0x1F))
        while True:
            var best_rank = -1
            var best_i = -1
            for i in range(len(word) - 1):
                var mk = word[i] + sep + word[i + 1]
                if mk in self.merge_rank:
                    var r = self.merge_rank[mk]
                    if best_rank == -1 or r < best_rank:
                        best_rank = r
                        best_i = i
            if best_i == -1:
                break
            var first = word[best_i]
            var second = word[best_i + 1]
            var merged = first + second
            var new_word = List[String]()
            var i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word^
            if len(word) < 2:
                break
        return word^

    # ------------------------------------------------------------------------
    # encode: text -> [BOS, ids..., EOS]   (NO padding; caller pads to 77)
    # ------------------------------------------------------------------------
    def encode(self, text: String) raises -> List[Int]:
        var ids = List[Int]()
        ids.append(self.bos_id)
        var cps = _clip_str_to_cps(text)
        var norm = self._normalize(cps)
        var toks = self._split(norm)
        for ti in range(len(toks)):
            ref pt = toks[ti]
            # byte-level expand: each codepoint -> UTF-8 bytes -> byte-level chars.
            var chars = List[String]()
            for ci in range(len(pt)):
                var b = List[Int]()
                _clip_cp_to_utf8(pt[ci], b)
                for bi in range(len(b)):
                    chars.append(String(chr(self.byte_to_uni[b[bi]])))
            var merged = self._bpe_word(chars)
            for mi in range(len(merged)):
                if merged[mi] in self.vocab:
                    ids.append(self.vocab[merged[mi]])
                # byte-level: every single char is in vocab, so this never misses.
        ids.append(self.eos_id)
        return ids^


# ----------------------------------------------------------------------------
# loader
# ----------------------------------------------------------------------------
def load(json_path: String) raises -> ClipTokenizer:
    return ClipTokenizer(json_path)
