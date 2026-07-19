# Pure-Mojo T5 SentencePiece-Unigram tokenizer (text -> token ids).
#
# Drop-in replacement for the HF `tokenizers` T5TokenizerFast used by the T5
# text encoder of Chroma / SD3.5 / FLUX / Anima. CPU-only string processing:
# no Python, no Rust crate at runtime. tokenizer.json is parsed ONCE at startup
# by a minimal hand-rolled JSON reader (reusing helpers from tokenizer.mojo).
#
# Source of truth: the standard T5 `tokenizer.json` (vocab 32100/32128):
#   model.type      = "Unigram"  (vocab = JSON array of [piece, score] pairs)
#   model.unk_id    = 2          (<unk>); <pad>=0, </s>=1 (EOS)
#   normalizer      = {"type":"Precompiled", precompiled_charsmap: ...}
#   pre_tokenizer   = Sequence[ WhitespaceSplit,
#                               Metaspace(replacement="▁", add_prefix_space=true) ]
#   post_processor  = TemplateProcessing single: A </s>   (append EOS id 1)
#
# ============================================================================
# REPLICATED FAITHFULLY (verified id-for-id against T5TokenizerFast on CPU):
#   * Pre-tokenization: WhitespaceSplit (split on whitespace runs, discard the
#     whitespace, collapse repeats, drop leading/trailing) then Metaspace
#     (prepend U+2581 "lower one eighth block" to every word). add_prefix_space
#     therefore applies per-word.
#   * Unigram inference = SentencePiece/HF lattice VITERBI best-path over the
#     word's codepoints, maximizing the SUM of piece log-scores. (NOT greedy
#     longest-match: e.g. "▁a" segments to "▁"+"a" because "▁a"
#     is absent from the vocab and that 2-piece path wins.)
#   * Unknown handling: at any codepoint position with NO single-codepoint piece
#     match, a 1-codepoint unk node is inserted with score = min_vocab_score
#     - 10.0 (HF's K_UNK_PENALTY). One <unk> (unk_id) is emitted per unknown
#     codepoint. NO byte-fallback (this T5 json has none).
#   * EOS: id 1 ("</s>") is appended. NO padding (caller pads).
#
# APPROXIMATED (documented divergence sources):
#   * NORMALIZATION: the "Precompiled" precompiled_charsmap (an NMT/NFKC charmap)
#     is NOT applied. Input codepoints pass through unchanged (identity). This is
#     EXACT for ASCII and for NFC/NFKC-stable text (incl. precomposed Latin
#     accents like e-acute U+00E9, which T5's map leaves untouched). It will
#     DIVERGE for inputs the charmap rewrites: NFKC-decomposable sequences
#     (e.g. "e"+combining-acute U+0301 -> must compose to U+00E9), full-width
#     forms, ligatures, some control/space characters, certain CJK compat chars.
#   * \s for WhitespaceSplit is approximated by is_whitespace() from tokenizer.mojo
#     (ASCII ws + the common Unicode separators) rather than the exact regex \s.
#   * Special/added tokens embedded literally in the INPUT text (e.g. a literal
#     "</s>" substring) are NOT matched as atomic specials; they are tokenized as
#     ordinary text. T5 conditioning prompts do not contain these, so this is a
#     non-issue for the intended use; documented for completeness.
# ============================================================================

from serenitymojo.tokenizer.tokenizer import (
    _read_utf8_file,
    _str_to_cps,
    _cps_to_string,
    is_whitespace,
    _find_key,
    _skip_ws,
    _parse_int,
    _parse_json_string,
)

comptime METASPACE: Int = 0x2581       # "▁" U+2581
comptime UNK_PENALTY: Float64 = 10.0   # HF tokenizers K_UNK_PENALTY
comptime NEG_INF: Float64 = -1.0e30


# ----------------------------------------------------------------------------
# float parser (scores like "-9.648370742797852", "0.0", optional exponent)
# ----------------------------------------------------------------------------
def _parse_float(bytes: Span[Byte, _], mut pos: Int) -> Float64:
    var n = len(bytes)
    var sign = 1.0
    if pos < n and Int(bytes[pos]) == 0x2D:      # '-'
        sign = -1.0
        pos += 1
    elif pos < n and Int(bytes[pos]) == 0x2B:    # '+'
        pos += 1
    var intpart = 0.0
    while pos < n:
        var b = Int(bytes[pos])
        if b >= 48 and b <= 57:
            intpart = intpart * 10.0 + Float64(b - 48)
            pos += 1
        else:
            break
    var frac = 0.0
    var scale = 0.1
    if pos < n and Int(bytes[pos]) == 0x2E:      # '.'
        pos += 1
        while pos < n:
            var b = Int(bytes[pos])
            if b >= 48 and b <= 57:
                frac = frac + Float64(b - 48) * scale
                scale = scale * 0.1
                pos += 1
            else:
                break
    var val = intpart + frac
    # optional exponent
    if pos < n and (Int(bytes[pos]) == 0x65 or Int(bytes[pos]) == 0x45):  # e/E
        pos += 1
        var esign = 1
        if pos < n and Int(bytes[pos]) == 0x2D:
            esign = -1
            pos += 1
        elif pos < n and Int(bytes[pos]) == 0x2B:
            pos += 1
        var e = 0
        while pos < n:
            var b = Int(bytes[pos])
            if b >= 48 and b <= 57:
                e = e * 10 + (b - 48)
                pos += 1
            else:
                break
        var p10 = 1.0
        for _ in range(e):
            p10 = p10 * 10.0
        if esign > 0:
            val = val * p10
        else:
            val = val / p10
    return sign * val


# ----------------------------------------------------------------------------
# The tokenizer
# ----------------------------------------------------------------------------
struct T5Tokenizer(Movable):
    # piece (literal UTF-8 string) -> id and -> log-score
    var piece_id: Dict[String, Int]
    var piece_score: Dict[String, Float64]
    var unk_id: Int
    var eos_id: Int
    var min_score: Float64
    var max_piece_len: Int   # in codepoints

    def __init__(out self, json_path: String) raises:
        self.piece_id = Dict[String, Int]()
        self.piece_score = Dict[String, Float64]()
        self.unk_id = 2
        self.eos_id = 1
        self.min_score = 0.0
        self.max_piece_len = 1

        var text = _read_utf8_file(json_path)
        var bytes = text.as_bytes()

        # unk_id (single occurrence in the file, inside the model object)
        var up = _find_key(bytes, String("unk_id"), 0)
        if up >= 0:
            _skip_ws(bytes, up)
            self.unk_id = _parse_int(bytes, up)

        self._parse_vocab(bytes)

    @staticmethod
    def load(json_path: String) raises -> T5Tokenizer:
        return T5Tokenizer(json_path)

    def _parse_vocab(mut self, bytes: Span[Byte, _]) raises:
        # model.vocab : [ [piece, score], [piece, score], ... ]
        var p = _find_key(bytes, String("vocab"), 0)
        if p < 0:
            raise Error("T5Tokenizer: 'vocab' not found in tokenizer.json")
        _skip_ws(bytes, p)
        if Int(bytes[p]) != 0x5B:   # '['
            raise Error("T5Tokenizer: malformed vocab (expected '[')")
        p += 1
        var n = len(bytes)
        var idx = 0
        var first = True
        while p < n:
            _skip_ws(bytes, p)
            if Int(bytes[p]) == 0x5D:   # ']'  -> end of vocab array
                p += 1
                break
            if Int(bytes[p]) == 0x2C:   # ','
                p += 1
                continue
            if Int(bytes[p]) != 0x5B:   # inner '['  -> [piece, score]
                raise Error("T5Tokenizer: malformed vocab pair")
            p += 1
            _skip_ws(bytes, p)
            var piece_cps = _parse_json_string(bytes, p)
            _skip_ws(bytes, p)
            if Int(bytes[p]) == 0x2C:
                p += 1
            _skip_ws(bytes, p)
            var score = _parse_float(bytes, p)
            _skip_ws(bytes, p)
            if Int(bytes[p]) == 0x5D:   # inner ']'
                p += 1
            var piece = _cps_to_string(piece_cps)
            self.piece_id[piece] = idx
            self.piece_score[piece] = score
            if first or score < self.min_score:
                self.min_score = score
                first = False
            var L = len(piece_cps)
            if L > self.max_piece_len:
                self.max_piece_len = L
            idx += 1
        if idx == 0:
            raise Error("T5Tokenizer: empty vocab")

    # ------------------------------------------------------------------------
    # Unigram Viterbi best-path over one word's codepoints (incl. leading ▁).
    # Appends the resulting piece ids to `out`.
    # ------------------------------------------------------------------------
    def _viterbi(self, w: List[Int], mut out: List[Int]) raises:
        var n = len(w)
        if n == 0:
            return
        var best = List[Float64](capacity=n + 1)
        var bp_start = List[Int](capacity=n + 1)
        var bp_id = List[Int](capacity=n + 1)
        for _ in range(n + 1):
            best.append(NEG_INF)
            bp_start.append(-1)
            bp_id.append(-1)
        best[0] = 0.0
        var unk_score = self.min_score - UNK_PENALTY

        for i in range(n):
            if best[i] <= NEG_INF * 0.5:
                continue
            var has_single = False
            var maxL = self.max_piece_len
            if n - i < maxL:
                maxL = n - i
            var cand = List[Int]()
            for L in range(1, maxL + 1):
                cand.append(w[i + L - 1])
                var cand_str = _cps_to_string(cand)
                if cand_str in self.piece_id:
                    var sc = self.piece_score[cand_str]
                    var total = best[i] + sc
                    var end = i + L
                    if total > best[end]:
                        best[end] = total
                        bp_start[end] = i
                        bp_id[end] = self.piece_id[cand_str]
                    if L == 1:
                        has_single = True
            if not has_single:
                var total = best[i] + unk_score
                var end = i + 1
                if total > best[end]:
                    best[end] = total
                    bp_start[end] = i
                    bp_id[end] = self.unk_id

        # backtrack n -> 0
        var rev = List[Int]()
        var j = n
        while j > 0:
            var st = bp_start[j]
            if st < 0:
                # unreachable safety net: emit unk and stop
                rev.append(self.unk_id)
                break
            rev.append(bp_id[j])
            j = st
        for k in range(len(rev) - 1, -1, -1):
            out.append(rev[k])

    # ------------------------------------------------------------------------
    # encode: text -> ids (+ trailing EOS=1, no padding)
    # ------------------------------------------------------------------------
    def encode(self, text: String) raises -> List[Int]:
        var ids = List[Int]()
        # normalization: identity passthrough (see top-of-file note).
        var cps = _str_to_cps(text)
        var n = len(cps)
        # WhitespaceSplit: split into words on whitespace runs (discarded).
        var i = 0
        while i < n:
            if is_whitespace(cps[i]):
                i += 1
                continue
            var start = i
            while i < n and not is_whitespace(cps[i]):
                i += 1
            # Metaspace: prepend ▁ to the word, then Viterbi-segment it.
            var word = List[Int]()
            word.append(METASPACE)
            for k in range(start, i):
                word.append(cps[k])
            self._viterbi(word, ids)
        ids.append(self.eos_id)
        return ids^
