# Skeptic Findings — pure-Mojo Ideogram B/C captioner GLUE

Target: `serenitymojo/captioner/ideogram_bc_glue.mojo`
Reference: ai-toolkit `extensions_built_in/captioner/Ideogram4Captioner.py` (B),
`ui_scripts/upsample_ideogram4_caption.py` (C), `toolkit/ideogram_caption.py` (module A).
Method: independent Python oracle (the REAL ai-toolkit glue fns, no GPU/LLM, via
`/home/alex/serenityflow-v2/.venv/bin/python`) vs the Mojo glue, run through purpose-built
Mojo harnesses comparing BYTES. Every claim below is backed by a real run, not a read.

**VERDICT: 1 FRAGILE divergence (non-numeric bbox coercion), 1 FRAGILE edge
(non-string truthy aspect_ratio). 0 BLOCKERS.** The committed probe passes 79/79; my
out-of-distribution battery (95 extra adversarial cases) found exactly two divergences,
both confined to malformed model output the system prompt steers against.

---

## What I ran (all PASS unless noted)

| Surface | Cases | Result |
|---|---|---|
| Templates B/C embedded vs Python | 2 | byte-exact: B sha `9e9ce2a3…` 22986B, C sha `7bf215b2…` 7966B |
| Directives FAITHFUL/CREATIVE vs Python | 2 | byte-exact (340 / 579 chars) |
| `dumps_default` / `dumps_pretty` vs CPython `json.dumps` | 13 shapes | ALL byte-identical |
| `extract_json` fence/brace battery | 37 + 14 + 11 = 62 | ALL match |
| `b_convert_bbox` / `c_sanitize_bbox` standard battery | 38 | 36 match, **2 mismatch (string-num)** |
| `b_convert_bbox` / `c_sanitize_bbox` coercion battery | 12 | **0 match (all bool/string)** |
| `b_full_glue` structural edges | 6 | ALL match |
| `b_full_glue` string-num bbox (integration) | 1 | **MISMATCH (drops bbox)** |
| `compute_aspect_ratio` (incl. search + banker ties) | 26 | ALL match |
| `normalize_item` (realistic) | 14 | ALL match |
| `normalize_item` non-string truthy AR | 7 | **5 diverge (FRAGILE)** |
| `b_build_prompt` / `c_build_prompt` incl. placeholders-in-values | 12 | ALL match (sha256+codepoint-len) |

The committed `ideogram_bc_glue_parity_probe.mojo` itself: **79/79 PASS** (re-ran, exit 0).

---

## FRAGILE #1 — bbox `float()` coercion: Mojo rejects bools & numeric strings; Python coerces

`_bbox_floats` (ideogram_bc_glue.mojo:356) accepts only JSON number kinds:
```mojo
if not (e.kind == 2 or e.kind == 3):  # J_INT or J_FLOAT
    return _Floats4(False, ...)       # -> caller returns None (drop bbox)
```
Python's `[float(v) for v in bbox]` (Ideogram4Captioner.py:98 / upsample:122) coerces:
`float(True)==1.0`, `float("100")==100.0`, `float("1.5")==1.5` — NO error. Only
`None`/dict/list raise `TypeError` (those two agree: both → None).

### Proven byte divergence (element-level, `c_sanitize_bbox`, NO swap):

| input bbox | Python out | Mojo out |
|---|---|---|
| `["100","200","400","800"]` | `[100,200,400,800]` | `null` |
| `["10.5","20.5","300.5","400.5"]` | `[10,20,300,400]` | `null` |
| `[false,false,true,true]` | `[0,0,1,1]` | `null` |
| `[false,100,true,800]` | `[0,100,1,800]` | `null` |
| `[100,"200",400,"800"]` (mixed) | `[100,200,400,800]` | `null` |

(B `_convert_bbox` diverges identically, with its x/y swap, e.g. `["100","200","400","800"]`
→ Python `[200,100,800,400]`, Mojo `null`.) All 12 coercion cases + the 2 string-num cases
in the standard battery mismatch; every OTHER bbox case (38 incl. banker `.5` ties
`[0.5,1.5,2.5,3.5]`→`[0,2,2,4]`, clamping, degenerate-drop, reversed-sort, wrong-len,
wrong-type, None/dict/list element) matches.

### Proven INTEGRATION divergence (`b_full_glue`, real saved-JSON bytes):
Input: `{"high_level_description":"A","compositional_deconstruction":{"background":"bg",`
`"elements":[{"type":"obj","bbox":["100","200","400","800"],"desc":"d"}]}}`

Python keeps the box:
```
      {
        "type": "obj",
        "bbox": [
          200,
          100,
          800,
          400
        ],
        "desc": "d"
      }
```
Mojo drops it:
```
      {
        "type": "obj",
        "desc": "d"
      }
```

### Severity = FRAGILE (not BLOCKER):
- The B/C **system prompts instruct the model to emit numeric** `[x1,y1,x2,y2]`
  (caption template line 70/198: "Format `[x1, y1, x2, y2]` … `x1 < x2`"). A well-formed
  model never produces string/bool coordinates, so the happy path is unaffected.
- The consequence is itself a degradation/recovery (drop the bad box), not a crash.
- BUT it IS a real, reproducible byte divergence on malformed output, and a fine-tuned
  model that quotes coordinates (`"100"`) would silently lose its boxes in Mojo while
  Python keeps them. Worth a fix or an explicit documented carve-out.
- **Fix**: in `_bbox_floats`, also accept `J_BOOL` (→ 1.0/0.0) and `J_STR` (parse with
  Python-`float()` grammar via `atof`, returning None only on a genuine ValueError),
  matching `float(v)`. Then re-run the coercion battery to gate it.

---

## FRAGILE #2 — `normalize_item`: non-string truthy `aspect_ratio` → default in Mojo, raw value in Python

`normalize_item` (ideogram_bc_glue.mojo:706) only takes `aspect_ratio` when it is a
non-empty **string**; otherwise default. Python is `item.get("aspect_ratio") or default`
— ANY truthy value passes through:

| input | Python | Mojo |
|---|---|---|
| `{"prompt":"x","aspect_ratio":16}` | `("x", 16)` | `("x", default)` |
| `{"prompt":"x","aspect_ratio":true}` | `("x", True)` | `("x", default)` |
| `{"prompt":"x","aspect_ratio":[1,9]}` | `("x", [1,9])` | `("x", default)` |
| `{"prompt":"x","aspect_ratio":false}` | `("x", default)` | `("x", default)` ✓ |
| `{"prompt":"x","aspect_ratio":""}`/`0`/`[]` | `("x", default)` | `("x", default)` ✓ |

Severity = FRAGILE: in Python a non-string `ar` then flows into
`template.replace("{{aspect_ratio}}", ar)` which **raises `TypeError`** (replace needs a
str) — so Python produces no usable caption for these inputs anyway; it crashes the item.
Mojo's stricter behavior (fall back to the default string) is arguably more robust but is a
genuine boundary divergence. The realistic string/empty/None/0/missing cases all match
(14/14). No fix strictly required; note it as an intentional hardening if kept.

---

## Surfaces I tried hard to break and could NOT (clean)

- **Hand-rolled 2-mode serializer** — byte-identical to CPython `json.dumps(…,
  ensure_ascii=False)` (default `", "`/`": "`) and `indent=2` across: empty `{}`/`[]`
  (stay inline both modes), nested-empty, single-key, arrays-of-objects, 3-deep nesting,
  bool/null, unicode (CJK/emoji/accented), control-char escapes (`\t \n `),
  `"`/`\`/`/` escaping, floats (`1.5`, `100.0`, `-0.0`, `1e2→100.0`, `3.25`). The pretty
  array-element-per-line form matches the adv fixture exactly.
- **Hand-written fence scanner (`extract_json`)** — matches the lazy-DOTALL
  `` ```(?:json)?\s*(.*?)``` `` + first-`{`/last-`}` regex across 62 adversarial inputs:
  case-sensitive `json` tag (`JSON`/`Json`/`js`/`jsonX`/`jsonl` all handled like Python),
  bare vs tagged fence, `\s*` eating `\t\f\v`/CRLF, NO whitespace after tag, 4/5/6
  backtick runs, multiple fences→FIRST, fence-without-close→brace-fallback, closing-fence-
  then-stray-`}`, lazy stop at a ``` INSIDE a JSON string (`{"a":"\`\`\`"}`→None),
  `} {` wrong-order→None, two-objects `{…} … {…}`→None, brace-inside-string-value,
  whitespace-only→None, prose-wrapped, leading/trailing ws + fence.
- **NUL-safety (Phase-1 concern)** — NOT reintroduced. Raw `\x00` inside a JSON string
  value flows through `_read_file`/`_bytes_to_string` (length-based) un-truncated and is
  correctly REJECTED by the strict parser as a control char `<0x20` — matching Python's
  `json.loads` (`{"a":"x\x00y"}`→None and bare `{"a":"p\x00q"}`→None both match).
- **bbox banker's rounding** (`_py_round` half-to-even): `[0.5,1.5,2.5,3.5]`,
  `[10.6,20.4,300.5,400.5]`, `[0.4,0.6,1000.4,1000.6]`, `[2.5,2.5,7.5,7.5]` all match.
- **`compute_aspect_ratio`** — clean-ratio early-return AND the q∈1..16 search (incl.
  `1023×768→4:3`, `33×17→31:16`, `1×1000→1:16`, `1366×769→16:9`, `2001×1000→2:1`,
  degenerate `0×100`/`100×0`/`-5×10→1:1`) and banker ties inside the search (`33×2`
  target 16.5) all match.
- **`build_prompt` substitution order** — B subs `{{aspect_ratio}}` then
  `{{user_instructions}}`; C subs `{{mode_directive}}`,`{{user_instructions}}`,
  `{{aspect_ratio}}`,`{{original_prompt}}` in that order. A placeholder that appears
  INSIDE a substituted value stays literal (not re-scanned) — verified for both B and C
  via sha256 + codepoint-len, incl. unicode and multiline instructions, None/empty/spaced.
- **module-A tail** (`normalize_caption_dict`, key order, palette upper/cap, medium canon,
  swap-in-text fallback) — exercised through full-glue structural edges (bbox:null,
  non-dict element preserved verbatim, no-bbox, decon-not-dict, elements-not-list→`{}`,
  degenerate-drop) all byte-identical; module A itself is separately gated.
- **B vs C wiring** — B swaps x/y, C does not (proven per-case); B malformed→
  `swap_bbox_xy_in_text(raw)` vs C→None (proven); B pretty `indent=2` vs C default-sep
  `", "`/`": "` (proven).

---

## One-line verdict
**BLOCKERS: 0. FRAGILE: 2 (bbox non-numeric coercion drops boxes Python keeps; non-string
truthy aspect_ratio).** Both are off the prompt-steered happy path; everything load-bearing
(serializer, fence scanner, bbox math/rounding, build_prompt order, NUL-safety, B/C wiring,
templates, module-A tail) is byte-identical to real CPython. Probe 79/79 PASS.
