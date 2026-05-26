# SKEPTIC FINDINGS — Serenitymojo Qwen3 byte-level BPE tokenizer
Date: 2026-05-25
Reviewer stance: adversarial ("assume it lies").
Scope reviewed (read-only): `serenitymojo/tokenizer/tokenizer.mojo`, `tokenizer_smoke.mojo`,
`tokenizer_fuzz.mojo`, `parity/`. Source of truth: the Z-Image Qwen3
`tokenizer.json` (`...snapshots/04cc4abb.../tokenizer/tokenizer.json`).
Oracle: HF `tokenizers` 0.22.2 (the lib the builder used), loaded from the SAME file.

All probe scripts/cases live in `parity/` (see "Reproduce" at the bottom). I did
NOT edit scope code or any plan/state docs.

---

## Bottom line

- **The gate is NOT circular.** I regenerated the oracle FRESH from HF and the
  hardcoded smoke ids match HF exactly (13/13), and the committed
  `fuzz_cases.tsv` is byte-identical to a fresh HF regeneration (genuine HF
  output, not Mojo output). See Clean Check C1/C2.
- **Fresh-oracle parity on my NEW prompts: 256/271 exact-match.**
  - Adversarial image-gen prompts (item 2): **77/77**
  - Regex tie-break stress (item 4): **119/119**
  - Pathological ASCII/Unicode whitespace (item 4): **60/60**
  - The 15 mismatches are all in ONE class: non-Latin scripts / non-ASCII
    numbers that hit the `\p{L}`/`\p{N}` approximation (**8/15** pass, 7 fail).
- **NFC verdict: CORRECT-BUT-FRAGILE** (confirmed no-op; only bites decomposed
  input, which is rare because keyboard/web/copy-paste deliver precomposed NFC).
- **One real BLOCKER-class gap the builder UNDER-documented (F1):** the
  `\p{L}`/`\p{N}` approximation diverges on Vietnamese, Thai-with-tone-marks,
  superscript digits, Roman numerals, and circled numbers — not just the "rare
  scripts" / "`\p{N}` = ASCII" the builder flagged. Severity depends on whether
  the Z-Image encoder is expected to see multilingual prompts.

---

## Findings

### F1 — `\p{L}`/`\p{N}` approximation diverges on common-ish multilingual input (UNDER-DOCUMENTED)
- **Where:** `tokenizer.mojo` `is_letter` (lines ~42–85) and `is_digit` (~35–39),
  used by `_pretokenize`.
- **What:** The pre-tokenizer regex uses Unicode `\p{L}` (letters) and `\p{N}`
  (numbers). The Mojo approximations are (a) **incomplete** — they miss whole
  letter blocks; and (b) **over-inclusive** — covered blocks include combining
  marks (category Mn/Mc) that `\p{L}` excludes, and non-ASCII numbers are
  mis-routed. Both change pre-token boundaries → different ids.
- **Expected (HF):** `\p{L}` = Unicode letter categories only (Lu/Ll/Lt/Lm/Lo),
  excluding marks; `\p{N}` = all number categories (Nd/Nl/No) as single chars.
- **Why it matters:** demonstrated divergences (Mojo vs fresh HF), all from
  `parity/skeptic_scripts.tsv`:
  | input | Mojo ids | HF ids | mechanism |
  |---|---|---|---|
  | `Tôi yêu mèo đẹp` (Vietnamese) | `…78,14854,124635,79` | `…78,128501` | `ẹ` U+1EB9 (Latin Ext Additional, a real Ll) NOT in `is_letter` → punct branch |
  | `cà phê sữa đá ngon` | `…134634,274,124843,…` | `…134634,133044,…` | `ữ` U+1EEF uncovered |
  | `แมวน่ารัก` (Thai) | `124840,124700,64741,124854` | `…64741,22287,123888` | Thai tone marks U+0E48/U+0E31 (Mn) treated as letters by Mojo (whole 0E00–0E7F block returns True) but excluded by `\p{L}` |
  | `E=mc² and x²+y²` | `…29456,10,88,29456` | `…29456,43010,29456` | `²` U+00B2 (No) is `\p{N}` to HF but punct to Mojo |
  | `Chapter Ⅳ and Ⅻ` | `…2858,227,96,…` | `…220,70467,96,…` | Roman numerals U+2160+ (Nl) mis-routed |
  | `① ② ③ steps` | `…2858,239,94,…` | `…220,48312,94,…` | circled numbers U+2460+ (No) mis-routed |
  | `সুন্দর বিড়াল` (Bengali) | `…49128,110` | `…41312,146227` | Bengali combining marks treated as letters |
- **Severity:** **HIGH if the encoder is expected to handle multilingual /
  scientific prompts; MEDIUM-LOW if prompts are ASCII/CJK/Latin-1 only.** This
  is the one place the documentation oversells: the code comment says
  `is_letter` "covers the scripts the encoder realistically sees" and flags only
  "rare scripts," but Vietnamese, Thai (any toned word), and superscript-² are
  not rare for a general image model. The two distinct defects (coverage gap +
  combining-mark over-inclusion) are not separately documented.
- **Evidence:** `parity/skeptic_scripts.tsv` (HF oracle), run via
  `skeptic_run.mojo` → 8 passed / 7 failed. Mechanism confirmed by
  `pre_tokenizer.pre_tokenize_str` showing HF splits at the combining marks.
- **NOT manufactured:** these are concrete inputs with two divergent id
  sequences each. Whether they constitute a blocker is a product decision about
  the target prompt distribution — I am flagging the divergence + its breadth,
  not asserting the encoder will see them.

### F2 — NFC normalizer is a no-op; decomposed input diverges (DOCUMENTED, CORRECT-BUT-FRAGILE)
- **Where:** `encode` (lines ~642–646): "NFC normalization is approximated as a
  no-op pass-through."
- **What:** `tokenizer.json` has `normalizer = NFC`. Mojo applies no
  normalization. Precomposed input is unaffected; decomposed (combining-mark)
  input diverges.
- **Expected:** HF runs NFC, so decomposed and precomposed forms collapse to the
  same ids.
- **Why / severity verdict — CORRECT-BUT-FRAGILE (not a blocker):**
  - Direct probes (`parity/skeptic_edge*.{mojo,py}`):
    | input | Mojo | HF | |
    |---|---|---|---|
    | `café` precomposed (U+00E9) | `[924,58858]` | `[924,58858]` | OK |
    | `café` decomposed (e+U+0301) | `[924,1859,53839]` | `[924,58858]` | DIVERGE |
    | `ü` precomposed (U+00FC) | `[2391]` | `[2391]` | OK |
    | `ü` decomposed (u+U+0308) | `[84,136,230]` | `[2391]` | DIVERGE |
    | `한국` precomposed | `[23573,124785]` | `[23573,124785]` | OK |
    | `한국` jamo-decomposed | 6 ids | `[23573,124785]` | DIVERGE |
  - Quantification: of 20 typical accented prompt words (café, résumé, Zürich,
    한국어, Pokémon, déjà vu, …), **20/20 are already in NFC form** as normally
    typed/pasted, so the no-op produces correct ids for all of them. Decomposed
    forms arise from specific sources (macOS HFS+ filenames, some IME pipelines),
    not normal keyboard/web entry.
  - Verdict: rare in practice for this use case; precomposed is the norm. NOT a
    blocker, but a genuine fragility worth a one-line warning at the API surface
    (or a future NFC pass) since the failure is silent.
- **Doc nit:** the smoke test comment claims the decomposed-`café` oracle is
  `[924,58858]` and prints it — fresh HF agrees (`[924,58858]`), so the comment
  is accurate. Good.

### F3 (MINOR / doc inaccuracy) — Arabic-Indic digit "known divergence" does NOT actually diverge
- **Where:** `tokenizer_smoke.mojo` lines ~91–94: prints Arabic-Indic digits
  U+0664–0666 and claims "differs: `\p{N}` approximated as ASCII 0-9 only."
- **What:** Both Mojo and fresh HF produce `[149,97,149,98,149,99]` — they
  AGREE. The arabic-indic digits aren't ASCII so Mojo routes them through the
  punct branch, and HF's `\p{N}` matches each as a single number; both end up
  isolating each char and byte-level-expanding to the same ids here.
- **Severity:** cosmetic. The "known divergence" note is misleading (it's a
  case that happens to MATCH), which slightly inflates the appearance of
  honesty. Real `\p{N}` divergences are the superscript/Roman/circled cases in
  F1, which the doc does not list.
- **Evidence:** `parity/skeptic_inspect.py` "documented known-divergence" block.

---

## Hammered items — results

1. **Circular gate?** NO. Hardcoded smoke ids re-derived FRESH from HF: 13/13
   match. `fuzz_cases.tsv` re-generated fresh = byte-identical to committed.
   (Clean C1/C2.)
2. **Adversarial NEW prompts:** 77 fresh image-gen prompts (long descriptive,
   `(word:1.2)` weighting, `{a|b}` wildcards, CJK+Latin, `!!!`, `4k 8K
   1920x1080`, hashtags, curly vs straight quotes, URLs, Windows paths,
   leading/trailing/empty/single, ZWJ/flag/skin-tone emoji, specials inline).
   **77/77 exact-match.** Plus 119 regex + 60 whitespace = **256/271 overall**;
   the 15 misses are all F1.
3. **NFC severity:** see F2 — CORRECT-BUT-FRAGILE. Quantified 20/20 typical
   accented words are NFC-stable as typed; only explicitly-decomposed input
   diverges.
4. **Regex tie-breaks at scale:** punct-then-letter (every ASCII punct),
   multi-space runs len 1–8, leading/trailing space runs, `\s+(?!\S)` vs `\s+`,
   `\s*[\r\n]+` with trailing non-newline ws, CR/LF/CRLF/LFCR, VT/FF, NBSP/U+2009/
   U+2028/U+2029/U+202F/U+3000, contractions-with-capitals (`DON'T`, `It'S`,
   `cAn'T`, `'tis`, `'TIS`, `y'all'd've`), non-contraction apostrophes
   (`o'clock`, `'x`, `'1`), digit runs len 1–10. **All match HF** (119/119 +
   60/60). The empirically-pinned whitespace branches are faithful.
5. **Special-token matching:** `<|im_start|>user`, mid-word, back-to-back
   specials, special-inside-text, partials (`<|im_`, `<|im_start|`), `special=
   False` added tokens (`<tool_call>`, `<think>`, `<|fim_prefix|>`). **All match
   HF.** Confirmed HF matches BOTH the 14 `special=True` and 12 `special=False`
   added tokens atomically (id-order longest-match), which is what the Mojo
   `_split_on_specials` does (all 26 atomic, longest-match, id-order tie-break).
6. **JSON parse robustness:** 1666 vocab keys contain `"` or `\` (must be JSON-
   escaped). Built 12 inputs whose correct tokenization REQUIRES those escape-
   heavy keys (`"`, `\`, `=\"`, `(\"`, `\"\n`, `\\`, JSON/HTML/Windows-path
   snippets). **All 12 match HF** → `\"` and `\\` handling in `_parse_json_string`
   is correct. (The `\u` branch is unexercised because this vocab stores all
   byte-level chars as literal UTF-8, but the bare-`"`/`\` tests prove the escape
   machinery; `\u` path is untested.)

---

## Clean checks (verified, no issue found)

- **C1** Hardcoded smoke ids == fresh HF (13/13). Gate not circular.
- **C2** Committed `fuzz_cases.tsv` == fresh HF regeneration (byte-identical, 88
  cases). Fuzz oracle is genuine.
- **C3** Config claims all true: model BPE, byte_fallback=False,
  ignore_merges=False, vocab 151643, merges 151387, normalizer NFC,
  pre_tokenizer Sequence[Split(Qwen2 regex, Isolated), ByteLevel(add_prefix_space
  =false, use_regex=false)], post_processor ByteLevel (no specials added), 26
  added_tokens ids 151643–151668.
- **C4** Byte-level (GPT-2 `bytes_to_unicode`) + BPE merge-rank + lowest-rank-
  first / lowest-index-on-tie loop: 256/271 fresh-oracle matches, all ASCII/
  Latin-1/CJK/emoji/punct/whitespace correct.
- **C5** Special-token longest-match + id-order tie-break == HF (incl.
  `special=False` added tokens).
- **C6** Hand-rolled JSON `\"`/`\\` escape handling == HF on 12 escape-forcing
  inputs.
- **C7** Empty string → `[]`, single space → `[220]`, single `!` → `[0]`: match.
- **C8** Decode round-trips for `[14990,1879]`, `[924,58858]`, `[151644,872]`
  produce the expected text (with the byte-level ` ` prefix-space artifact, which
  is HF-faithful for byte-level decode).

---

## Couldn't verify / out of scope

- **`\u` escape decode path** in `_parse_json_string` (lines ~197–206): not
  exercised because this `tokenizer.json` has zero `\u` escapes in vocab/merges
  (all byte-level chars stored literally). The code exists and looks correct but
  is untested against this file. (Low risk for THIS file; would matter if a
  different tokenizer.json used `\u`.)
- **Throughput / perf:** not measured (skeptic is correctness-only).
- **Full vocab/merge dump diff:** I verified vocab correctness transitively via
  encode round-trips (incl. escape-heavy keys) rather than dumping all 151k
  entries; Mojo `Dict` iteration order made a direct full dump awkward. The
  escape-forcing inputs (F6/C6) + 256-case parity give high confidence the
  parse is complete, but I did not byte-diff all 151643 keys.
- **NFC for input that is decomposed by the caller upstream:** I tested the
  tokenizer in isolation; whether the Z-Image pipeline ever hands it decomposed
  text depends on the caller (not in scope here).

---

## Reproduce

```bash
cd /home/alex/mojodiffusion
# 1. Fresh oracle + config inspection + re-derive hardcoded smoke ids
pixi run python serenitymojo/tokenizer/parity/skeptic_inspect.py
# 2. HF special-token behavior probe
pixi run python serenitymojo/tokenizer/parity/skeptic_specials.py
# 3. Generate fresh adversarial / regex / ws / script oracles (TSV)
pixi run python serenitymojo/tokenizer/parity/skeptic_oracle.py     # 77 cases
pixi run python serenitymojo/tokenizer/parity/skeptic_regex.py      # 119 cases
pixi run python serenitymojo/tokenizer/parity/skeptic_ws.py         # 51 cases
pixi run python serenitymojo/tokenizer/parity/skeptic_scripts.py    # 15 cases (F1)
pixi run python serenitymojo/tokenizer/parity/skeptic_ws_extra.py   # 9 unicode-ws
# 4. Run Mojo tokenizer vs each oracle (parametrized by TSV path)
for f in skeptic_cases skeptic_regex_cases skeptic_ws_cases skeptic_ws_extra skeptic_scripts; do
  pixi run mojo run -I . serenitymojo/tokenizer/parity/skeptic_run.mojo \
      serenitymojo/tokenizer/parity/$f.tsv
done
# 5. Edge/NFC/escape spot checks (direct, no TSV)
pixi run mojo run -I . serenitymojo/tokenizer/parity/skeptic_edge.mojo
pixi run python  serenitymojo/tokenizer/parity/skeptic_edge_oracle.py
pixi run mojo run -I . serenitymojo/tokenizer/parity/skeptic_escape.mojo
# 6. Baseline gates (builder's)
pixi run mojo run -I . serenitymojo/tokenizer/tokenizer_smoke.mojo   # 13/13
pixi run mojo run -I . serenitymojo/tokenizer/tokenizer_fuzz.mojo    # 88/88
```
