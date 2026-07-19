# ideogram_bc_glue.mojo — pure-Mojo 1:1 port of the DETERMINISTIC glue around
# module A for the two Ideogram-4 captioner pipelines:
#   B = Ideogram4Captioner (image → JSON)   — ai-toolkit extensions_built_in/captioner/Ideogram4Captioner.py
#   C = upsample_ideogram4_caption (idea → JSON) — ai-toolkit ui_scripts/upsample_ideogram4_caption.py
#
# Spec (read line-by-line): /home/alex/EriTrainer/IDEOGRAM_BC_GLUE_PORT_SPEC.md
# Oracle generator (verbatim glue): /home/alex/EriTrainer/trainer/parity/ideogram_bc_glue/gen_glue_fixtures.py
#
# This is HOST string/JSON logic only — NO GPU, NO LLM, NO torch. The single
# non-deterministic box (model.generate) is a pluggable backend (see the
# CaptionGenerator trait at the bottom) and is NOT byte-gated.
#
# The pipeline (both B and C):
#   build_prompt (template substitution) → [LLM.generate] → extract_json →
#     per-element bbox fix (B SWAPS x/y, C does NOT) → module-A
#     normalize_caption_dict → json.dumps (B indent=2 pretty, C indent=None default-sep)
#
# Reuses module A verbatim: parse_json (strict), JValue (ordered), normalize_caption_dict,
# swap_bbox_xy_in_text, _py_round (banker's rounding). Adds two NEW serializer modes
# (default-separators ", "/": " and pretty indent=2) matching CPython byte-for-byte.

from std.memory import alloc

from serenitymojo.io.ffi import (
    sys_open,
    sys_close,
    sys_pread,
    file_size,
    BytePtr,
    O_RDONLY,
)
from serenitymojo.captioner.ideogram_caption import (
    JValue,
    parse_json,
    normalize_caption_dict,
    swap_bbox_xy_in_text,
    _py_round,
    _escape_string_into,
    _append_ascii,
    _bytes_to_string,
)


# ════════════════════════════ template embedding ══════════════════════════════
# The two system prompts are embedded as data files read at runtime (the spec's
# "generated file + hash-assert" approach — far less error-prone than a 22 831 /
# 7 892-char Mojo string literal with em-dashes + literal \uNNNN / \n). Both were
# produced exactly as the Python does:
#   B: `from .prompts.ideogram4_caption_prompt import ideogram4_caption_prompt`
#      (the POST-IMPORT string) → ideogram4_caption_prompt.txt (22 831 chars).
#   C: src[src.find('"""')+3 : src.rfind('"""')] (the verbatim between-triple-quote
#      slice, literal \uNNNN/\n unresolved) → ideogram4_upsample_prompt.txt (7 892 chars).
comptime B_TEMPLATE_PATH = "/home/alex/mojodiffusion/serenitymojo/captioner/prompts/ideogram4_caption_prompt.txt"
comptime C_TEMPLATE_PATH = "/home/alex/mojodiffusion/serenitymojo/captioner/prompts/ideogram4_upsample_prompt.txt"

# C-only mode directives (verbatim from upsample_ideogram4_caption.py lines 40–56).
comptime FAITHFUL_DIRECTIVE = (
    "- **Fill in ONLY what the structure needs.** Add a concrete background shell, "
    "bounding boxes, and the required elements/text -- nothing else. Do NOT add new "
    "subjects, props, narrative, mood, or a setting the user did not specify. If the "
    "prompt names no location, keep the background minimal. If the prompt is sparse, "
    "the scene stays sparse."
)
comptime CREATIVE_DIRECTIVE = (
    "- **Expand the scene while keeping the user's idea intact.** Place the subject in "
    "a specific, believable setting and build a real background environment with fitting "
    "secondary details (props, depth layers, atmosphere) that serve the idea -- never a "
    "blank or 'plain' background when a setting can be implied. Everything you add must "
    "support, never replace or contradict, what the user asked for, and you must not "
    "introduce a different main subject. The FIDELITY rules above still hold: triggers "
    "verbatim, no invented appearance for a named person, no elaboration of a named style."
)


def _read_file(path: String) raises -> String:
    """Length-aware read via io/ffi (never builtin open, never NUL-terminated)."""
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error("cannot open template/file: " + path)
    var sz = file_size(fd)
    var tmp = alloc[UInt8](sz)
    var bp = BytePtr(unsafe_from_address=Int(tmp))
    var done = 0
    while done < sz:
        var got = sys_pread(fd, bp + done, sz - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var buf = List[UInt8]()
    for i in range(done):
        buf.append(tmp[i])
    tmp.free()
    return _bytes_to_string(buf)


def load_b_template() raises -> String:
    return _read_file(B_TEMPLATE_PATH)


def load_c_template() raises -> String:
    return _read_file(C_TEMPLATE_PATH)


# ════════════════════════════ string helpers ══════════════════════════════════

def _is_py_space(c: Int) -> Bool:
    # Python str.strip() default whitespace: space, \t, \n, \r, \f, \v.
    return (
        c == 0x20 or c == 0x09 or c == 0x0A or c == 0x0D or c == 0x0C or c == 0x0B
    )


def _strip(s: String) -> String:
    """Python str.strip(): trim leading/trailing ASCII whitespace."""
    var b = s.as_bytes()
    var n = len(b)
    var start = 0
    while start < n and _is_py_space(Int(b[start])):
        start += 1
    var end = n
    while end > start and _is_py_space(Int(b[end - 1])):
        end -= 1
    var out = List[UInt8]()
    for i in range(start, end):
        out.append(b[i])
    return _bytes_to_string(out)


def str_replace_all(haystack: String, needle: String, replacement: String) -> String:
    """Python str.replace(needle, replacement) — replace EVERY non-overlapping
    occurrence, left to right (the substituted text is never re-scanned)."""
    var hb = haystack.as_bytes()
    var nb = needle.as_bytes()
    var hn = len(hb)
    var nn = len(nb)
    if nn == 0:
        return haystack
    var out = List[UInt8]()
    var i = 0
    while i < hn:
        # Try to match `needle` at position i.
        var matched = False
        if i + nn <= hn:
            matched = True
            for k in range(nn):
                if hb[i + k] != nb[k]:
                    matched = False
                    break
        if matched:
            var rb = replacement.as_bytes()
            for k in range(len(rb)):
                out.append(rb[k])
            i += nn  # skip the needle; do NOT rescan the replacement
        else:
            out.append(hb[i])
            i += 1
    return _bytes_to_string(out)


# ═══════════════════════════ B+C: build_prompt ════════════════════════════════

def b_build_prompt(template: String, caption_prompt: String, has_caption_prompt: Bool, aspect_ratio: String) -> String:
    """B Ideogram4Captioner.build_prompt: substitute {{aspect_ratio}} then
    {{user_instructions}}. user_instructions = caption_prompt.strip(); empty/None →
    the literal "None.". `has_caption_prompt`=False models Python's None (→ "")."""
    var user_instructions = _strip(caption_prompt) if has_caption_prompt else String("")
    if user_instructions.byte_length() == 0:
        user_instructions = String("None.")
    var prompt = str_replace_all(template, "{{aspect_ratio}}", aspect_ratio)
    prompt = str_replace_all(prompt, "{{user_instructions}}", user_instructions)
    return prompt


def c_build_prompt(template: String, aspect_ratio: String, original_prompt: String, creative: Bool, instructions: String) -> String:
    """C upsample build_prompt: substitute {{mode_directive}} (FAITHFUL/CREATIVE),
    {{user_instructions}} (instructions.strip() or "None."), {{aspect_ratio}},
    {{original_prompt}} — in THIS order. original_prompt is passed through verbatim
    (the caller upstream strips the idea, not build_prompt)."""
    var directive = String(CREATIVE_DIRECTIVE) if creative else String(FAITHFUL_DIRECTIVE)
    var prompt = str_replace_all(template, "{{mode_directive}}", directive)
    var instr = _strip(instructions)
    if instr.byte_length() == 0:
        instr = String("None.")
    prompt = str_replace_all(prompt, "{{user_instructions}}", instr)
    prompt = str_replace_all(prompt, "{{aspect_ratio}}", aspect_ratio)
    prompt = str_replace_all(prompt, "{{original_prompt}}", original_prompt)
    return prompt


# ═══════════════════════════ shared: extract_json ═════════════════════════════
# Python:
#   text = raw.strip()
#   fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)   # FIRST, lazy
#   if fence: text = fence.group(1).strip()
#   start = text.find("{"); end = text.rfind("}")
#   if start==-1 or end==-1 or end<=start: return None
#   try: return json.loads(text[start:end+1]) except: return None
# Mojo has no regex → hand-write the lazy-DOTALL fence scanner (same pattern as
# module A's bbox scanner). Returns a JValue (or null-with-ok=False meaning None).

struct ExtractResult(Copyable, Movable):
    var ok: Bool       # True if a JSON object was parsed (Python: not None)
    var value: JValue  # the parsed value when ok

    def __init__(out self, ok: Bool, var value: JValue):
        self.ok = ok
        self.value = value^


def _re_ws_run_len(b: Span[UInt8, _], n: Int, p: Int) -> Int:
    """Length of the \\s* run (regex whitespace [ \\t\\n\\r\\f\\v]) starting at p."""
    var q = p
    while q < n:
        var c = Int(b[q])
        if (
            c == 0x20 or c == 0x09 or c == 0x0A or c == 0x0D or c == 0x0C or c == 0x0B
        ):
            q += 1
        else:
            break
    return q - p


def _find_first_fence(text: String) -> Int:
    """Replicate re.search(r"```(?:json)?\\s*(.*?)```", DOTALL): find the FIRST
    position where ```[json]?\\s* opens AND a closing ``` exists after the lazy
    (.*?). Returns the byte index of the group(1) START, or -1 if no fence matches.
    Stores the group end via the module-level _fence_group_end side channel is
    avoided — instead we return start and the caller rescans for the closing ```.
    Here we return -1 (no match) or the group-1 start index; the matched group end
    is the next ``` at or after the group start (lazy → first one)."""
    var b = text.as_bytes()
    var n = len(b)
    var i = 0
    while i < n:
        # need an opening ``` at i
        if i + 3 <= n and Int(b[i]) == 0x60 and Int(b[i + 1]) == 0x60 and Int(b[i + 2]) == 0x60:
            var p = i + 3
            # optional literal "json"
            if (
                p + 4 <= n
                and Int(b[p]) == 0x6A
                and Int(b[p + 1]) == 0x73
                and Int(b[p + 2]) == 0x6F
                and Int(b[p + 3]) == 0x6E
            ):
                p += 4
            # \s*
            p += _re_ws_run_len(b, n, p)
            # group(1) is lazy (.*?) up to the FIRST closing ```. Does one exist?
            var close = _find_seq(b, n, p, 0x60)
            if close >= 0:
                return p  # group(1) starts at p; caller takes [p, close)
        i += 1
    return -1


def _find_seq(b: Span[UInt8, _], n: Int, start: Int, _backtick: Int) -> Int:
    """Index of the next ``` (three backticks) at or after `start`, or -1."""
    var q = start
    while q + 3 <= n:
        if Int(b[q]) == 0x60 and Int(b[q + 1]) == 0x60 and Int(b[q + 2]) == 0x60:
            return q
        q += 1
    return -1


def _fence_group(text: String) -> String:
    """If a fence matches, return group(1).strip(); else return the input unchanged.
    group(1) = the lazy (.*?) span from after ```[json]?\\s* up to the FIRST ```."""
    var gstart = _find_first_fence(text)
    if gstart < 0:
        return text
    var b = text.as_bytes()
    var n = len(b)
    var gend = _find_seq(b, n, gstart, 0x60)
    if gend < 0:
        return text  # shouldn't happen (find_first_fence checked), defensive
    var inner = List[UInt8]()
    for i in range(gstart, gend):
        inner.append(b[i])
    return _strip(_bytes_to_string(inner))


def _find_char(s: String, ch: Int) -> Int:
    var b = s.as_bytes()
    for i in range(len(b)):
        if Int(b[i]) == ch:
            return i
    return -1


def _rfind_char(s: String, ch: Int) -> Int:
    var b = s.as_bytes()
    var i = len(b) - 1
    while i >= 0:
        if Int(b[i]) == ch:
            return i
        i -= 1
    return -1


def _slice(s: String, start: Int, end: Int) -> String:
    var b = s.as_bytes()
    var out = List[UInt8]()
    for i in range(start, end):
        out.append(b[i])
    return _bytes_to_string(out)


def extract_json(raw: String) -> ExtractResult:
    """Shared B/C extract_json (behaviour-identical). Returns ok=False for Python's
    None (no JSON / unparseable / degenerate span)."""
    var text = _strip(raw)
    text = _fence_group(text)  # strip the first ```...``` fence if present
    var start = _find_char(text, 0x7B)  # first '{'
    var end = _rfind_char(text, 0x7D)   # last '}'
    if start == -1 or end == -1 or end <= start:
        return ExtractResult(False, JValue())
    var candidate = _slice(text, start, end + 1)
    try:
        var d = parse_json(candidate)
        return ExtractResult(True, d^)
    except:
        return ExtractResult(False, JValue())


# ═══════════════════════ bbox fix: B swaps, C does not ═════════════════════════
# clamp0_1000(v) = max(0, min(1000, round(float(v)))); round() = banker's (module A _py_round).

struct BBoxResult(Copyable, Movable):
    var ok: Bool       # False → Python None (drop the bbox key, keep element)
    var y1: Int
    var x1: Int
    var y2: Int
    var x2: Int

    def __init__(out self):
        self.ok = False
        self.y1 = 0
        self.x1 = 0
        self.y2 = 0
        self.x2 = 0


@fieldwise_init
struct _Floats4(Copyable, Movable):
    var ok: Bool
    var a: Float64  # position 0
    var b: Float64  # position 1
    var c: Float64  # position 2
    var d: Float64  # position 3


def _coerce_float(e: JValue) raises -> Float64:
    """Python `float(v)` coercion for one bbox element. Matches the real
    `[float(v) for v in bbox]` in Ideogram4Captioner._convert_bbox / sanitize_bbox:
      - JSON number (J_INT/J_FLOAT) → its value;
      - bool (J_BOOL)              → float(True)=1.0 / float(False)=0.0;
      - string (J_STR)             → float(str) via atof (handles ' 100 ', '1.5',
                                      '1e2', '+5', '.5', '5.', 'inf', 'nan'); a
                                      non-numeric string raises ValueError (atof
                                      raises) → caller drops the box, like Python.
      - None / list / dict         → TypeError → raise (caller drops the box).
    Residual (documented, off the prompt-steered path): Python `float('1_000')`
    accepts digit-group underscores (→1000.0) while atof raises; bbox strings from a
    model are plain digits, so this never occurs in practice."""
    if e.kind == 2 or e.kind == 3:  # J_INT / J_FLOAT
        return atof(e.num_text)
    if e.kind == 1:  # J_BOOL → 1.0 / 0.0
        return 1.0 if e.b else 0.0
    if e.kind == 4:  # J_STR → float(str); non-numeric → atof raises (ValueError)
        return atof(e.s)
    # J_NULL / J_ARR / J_OBJ → Python float() raises TypeError
    raise Error("float(): non-coercible element")


def _bbox_floats(bbox: JValue) -> _Floats4:
    """Mirror `x1,y1,x2,y2 = [float(v) for v in bbox]` with Python's float() coercion
    (numbers, bools, and numeric strings all coerce; None/list/dict or a non-numeric
    string raise → caller returns None). Returns ok + the 4 floats by position."""
    if not bbox.is_array():
        return _Floats4(False, 0.0, 0.0, 0.0, 0.0)
    if len(bbox.arr) != 4:
        return _Floats4(False, 0.0, 0.0, 0.0, 0.0)
    var vals = List[Float64]()
    for i in range(4):
        var e = bbox.arr[i].copy()
        try:
            vals.append(_coerce_float(e))
        except:
            return _Floats4(False, 0.0, 0.0, 0.0, 0.0)
    return _Floats4(True, vals[0], vals[1], vals[2], vals[3])


def _clamp0_1000_round(v: Float64) -> Int:
    var r = _py_round(v)  # banker's rounding (module A)
    if r < 0:
        return 0
    if r > 1000:
        return 1000
    return r


@fieldwise_init
struct _Pair2(Copyable, Movable):
    var lo: Int
    var hi: Int


def _sort2(a: Int, b: Int) -> _Pair2:
    if a <= b:
        return _Pair2(a, b)
    return _Pair2(b, a)


def b_convert_bbox(bbox: JValue) -> BBoxResult:
    """B _convert_bbox: input [x1,y1,x2,y2], clamp/round/sort each axis, SWAP to
    stored [y1,x1,y2,x2]. Degenerate (y2<=y1 or x2<=x1) → None."""
    var res = BBoxResult()
    var f = _bbox_floats(bbox)
    if not f.ok:
        return res^
    # x1,y1,x2,y2 = floats by position (a,b,c,d)
    var sx = _sort2(_clamp0_1000_round(f.a), _clamp0_1000_round(f.c))  # x-axis: positions 0,2
    var sy = _sort2(_clamp0_1000_round(f.b), _clamp0_1000_round(f.d))  # y-axis: positions 1,3
    var x1 = sx.lo
    var x2 = sx.hi
    var y1 = sy.lo
    var y2 = sy.hi
    if y2 <= y1 or x2 <= x1:
        return res^
    res.ok = True
    res.y1 = y1
    res.x1 = x1
    res.y2 = y2
    res.x2 = x2
    return res^


def c_sanitize_bbox(bbox: JValue) -> BBoxResult:
    """C sanitize_bbox: input ALREADY [y1,x1,y2,x2], clamp/round/sort each axis in
    place — NO swap. Degenerate → None."""
    var res = BBoxResult()
    var f = _bbox_floats(bbox)
    if not f.ok:
        return res^
    # y1,x1,y2,x2 = floats by position (a,b,c,d)
    var sy = _sort2(_clamp0_1000_round(f.a), _clamp0_1000_round(f.c))  # y-axis: positions 0,2
    var sx = _sort2(_clamp0_1000_round(f.b), _clamp0_1000_round(f.d))  # x-axis: positions 1,3
    var y1 = sy.lo
    var y2 = sy.hi
    var x1 = sx.lo
    var x2 = sx.hi
    if y2 <= y1 or x2 <= x1:
        return res^
    res.ok = True
    res.y1 = y1
    res.x1 = x1
    res.y2 = y2
    res.x2 = x2
    return res^


def _bbox_to_jarray(r: BBoxResult) -> JValue:
    # Result is always stored order [y1,x1,y2,x2] as JSON INTS.
    var arr = JValue.new_array()
    arr.append(JValue.from_int_text(String(r.y1)))
    arr.append(JValue.from_int_text(String(r.x1)))
    arr.append(JValue.from_int_text(String(r.y2)))
    arr.append(JValue.from_int_text(String(r.x2)))
    return arr^


def _apply_bbox_fix(var data: JValue, is_b: Bool) -> JValue:
    """Application loop shared by B _normalize_caption / C sanitize_caption:
    walk compositional_deconstruction.elements; for each dict element with a "bbox",
    replace with the fixed bbox (B swap / C no-swap) or DROP the bbox key (keep el)
    when the fix returns None. Then call module-A normalize_caption_dict."""
    # decon = data.get("compositional_deconstruction", {}); elements if decon is dict
    if data.is_object() and data.has_key("compositional_deconstruction"):
        var decon = data.get("compositional_deconstruction")
        if decon.is_object() and decon.has_key("elements"):
            var els = decon.get("elements")
            if els.is_array():
                # Rebuild data with mutated elements (JValue is value-typed). We
                # construct a fresh object preserving order, swapping in fixed elements.
                var new_els = JValue.new_array()
                for i in range(len(els.arr)):
                    var el = els.arr[i].copy()
                    if el.is_object() and el.has_key("bbox"):
                        var bb = el.get("bbox")
                        var fixed = b_convert_bbox(bb) if is_b else c_sanitize_bbox(bb)
                        new_els.append(_el_with_bbox(el^, fixed))
                    else:
                        new_els.append(el^)
                # rebuild decon with elements replaced (preserve key order)
                var new_decon = JValue.new_object()
                for j in range(len(decon.obj)):
                    var k = decon.obj[j].key
                    if k == "elements":
                        new_decon.set("elements", new_els.copy())
                    else:
                        new_decon.set(k, decon.obj[j].value.copy())
                # rebuild data with decon replaced
                var new_data = JValue.new_object()
                for j in range(len(data.obj)):
                    var k = data.obj[j].key
                    if k == "compositional_deconstruction":
                        new_data.set(k, new_decon.copy())
                    else:
                        new_data.set(k, data.obj[j].value.copy())
                return normalize_caption_dict(new_data)
    return normalize_caption_dict(data)


def _el_with_bbox(var el: JValue, fixed: BBoxResult) -> JValue:
    """Return `el` with its bbox replaced (ok) or dropped (None), preserving the
    order of every other key. Mirrors el["bbox"]=cleaned / el.pop("bbox")."""
    var out = JValue.new_object()
    for i in range(len(el.obj)):
        var k = el.obj[i].key
        if k == "bbox":
            if fixed.ok:
                out.set("bbox", _bbox_to_jarray(fixed))
            # else: drop the bbox key (do not append)
        else:
            out.set(k, el.obj[i].value.copy())
    return out^


# ════════════════════ NEW serializer modes (CPython json.dumps) ════════════════
# Module A's compact serializer uses separators=(",",":"). The glue needs TWO more:
#   - DEFAULT (indent=None): item_sep ", "  key_sep ": "   (single line, SPACES)
#   - PRETTY (indent=2): newline + 2-space-per-level indent; each item on its own
#     line; ',' immediately before the newline; key_sep ": "; EMPTY {}/[] stay on
#     one line. (Verified vs CPython json.dumps.)
# ensure_ascii=False in all modes → raw UTF-8 passthrough (reuse _escape_string_into).

def _ser_default(mut out: List[UInt8], v: JValue):
    """json.dumps(v, ensure_ascii=False) default form: ', ' and ': ' separators."""
    if v.kind == 5:  # J_ARR
        out.append(0x5B)  # [
        for i in range(len(v.arr)):
            if i > 0:
                out.append(0x2C)  # ,
                out.append(0x20)  # space
            _ser_default(out, v.arr[i])
        out.append(0x5D)  # ]
    elif v.kind == 6:  # J_OBJ
        out.append(0x7B)  # {
        for i in range(len(v.obj)):
            if i > 0:
                out.append(0x2C)  # ,
                out.append(0x20)  # space
            _escape_string_into(out, v.obj[i].key)
            out.append(0x3A)  # :
            out.append(0x20)  # space
            _ser_default(out, v.obj[i].value)
        out.append(0x7D)  # }
    else:
        _ser_scalar(out, v)


def _ser_pretty(mut out: List[UInt8], v: JValue, level: Int):
    """json.dumps(v, ensure_ascii=False, indent=2). `level` = current nesting depth
    (indent = 2*level spaces). Empty containers stay on one line."""
    if v.kind == 5:  # J_ARR
        if len(v.arr) == 0:
            out.append(0x5B)
            out.append(0x5D)  # []
            return
        out.append(0x5B)  # [
        for i in range(len(v.arr)):
            if i > 0:
                out.append(0x2C)  # ,
            out.append(0x0A)  # newline
            _indent(out, level + 1)
            _ser_pretty(out, v.arr[i], level + 1)
        out.append(0x0A)
        _indent(out, level)
        out.append(0x5D)  # ]
    elif v.kind == 6:  # J_OBJ
        if len(v.obj) == 0:
            out.append(0x7B)
            out.append(0x7D)  # {}
            return
        out.append(0x7B)  # {
        for i in range(len(v.obj)):
            if i > 0:
                out.append(0x2C)  # ,
            out.append(0x0A)  # newline
            _indent(out, level + 1)
            _escape_string_into(out, v.obj[i].key)
            out.append(0x3A)  # :
            out.append(0x20)  # space (key_separator ": ")
            _ser_pretty(out, v.obj[i].value, level + 1)
        out.append(0x0A)
        _indent(out, level)
        out.append(0x7D)  # }
    else:
        _ser_scalar(out, v)


def _ser_scalar(mut out: List[UInt8], v: JValue):
    if v.kind == 0:  # J_NULL
        _append_ascii(out, "null")
    elif v.kind == 1:  # J_BOOL
        _append_ascii(out, "true" if v.b else "false")
    elif v.kind == 2 or v.kind == 3:  # J_INT / J_FLOAT
        _append_ascii(out, v.num_text)
    elif v.kind == 4:  # J_STR
        _escape_string_into(out, v.s)


def _indent(mut out: List[UInt8], level: Int):
    for _i in range(level * 2):
        out.append(0x20)  # 2 spaces per level


def dumps_default(data: JValue) -> String:
    """json.dumps(data, ensure_ascii=False) — single line, ', '/': ' separators."""
    var out = List[UInt8]()
    _ser_default(out, data)
    return _bytes_to_string(out)


def dumps_pretty(data: JValue) -> String:
    """json.dumps(data, ensure_ascii=False, indent=2) — 2-space pretty."""
    var out = List[UInt8]()
    _ser_pretty(out, data, 0)
    return _bytes_to_string(out)


# ═══════════════════════════ full glue pipelines ══════════════════════════════

def b_full_glue(raw_model_string: String) -> String:
    """B post-LLM glue: extract_json → (None? swap_bbox_xy_in_text(raw)) else
    bbox-fix(swap) + normalize → pretty indent=2 JSON. Mirrors get_caption_for_file."""
    var ex = extract_json(raw_model_string)
    if not ex.ok:
        return swap_bbox_xy_in_text(raw_model_string)  # B malformed fallback
    var fixed = _apply_bbox_fix(ex.value.copy(), True)  # B SWAP
    return dumps_pretty(fixed)


struct CGlueResult(Copyable, Movable):
    var ok: Bool      # False → Python None (item becomes null)
    var value: String

    def __init__(out self, ok: Bool, value: String):
        self.ok = ok
        self.value = value


def c_full_glue(raw_model_string: String) -> CGlueResult:
    """C post-LLM glue: extract_json → None? None else bbox-fix(NO swap) + normalize
    → compact-default (indent=None) JSON. C has NO malformed fallback."""
    var ex = extract_json(raw_model_string)
    if not ex.ok:
        return CGlueResult(False, String(""))  # C returns None
    var fixed = _apply_bbox_fix(ex.value.copy(), False)  # C NO swap
    return CGlueResult(True, dumps_default(fixed))


# ═══════════════════════ B-only: compute_aspect_ratio ═════════════════════════

comptime MAX_AR_DENOMINATOR = 16


def _gcd(a: Int, b: Int) -> Int:
    var x = a
    var y = b
    while y != 0:
        var t = y
        y = x % y
        x = t
    return x


def compute_aspect_ratio(width: Int, height: Int) -> String:
    """B compute_aspect_ratio: gcd-reduce; if both ≤16 return rw:rh; else search
    q∈1..16, p=max(1,round(target*q)), keep min |p/q − target|. round() = banker's."""
    if width <= 0 or height <= 0:
        return String("1:1")
    var g = _gcd(width, height)
    var rw = width // g
    var rh = height // g
    if rw <= MAX_AR_DENOMINATOR and rh <= MAX_AR_DENOMINATOR:
        return String(rw) + ":" + String(rh)
    var target = Float64(width) / Float64(height)
    var have_best = False
    var best_err = Float64(0)
    var best_p = 1
    var best_q = 1
    for q in range(1, MAX_AR_DENOMINATOR + 1):
        var p = _py_round(target * Float64(q))
        if p < 1:
            p = 1
        var err = _fabs(Float64(p) / Float64(q) - target)
        if (not have_best) or err < best_err:
            have_best = True
            best_err = err
            best_p = p
            best_q = q
    return String(best_p) + ":" + String(best_q)


def _fabs(x: Float64) -> Float64:
    return -x if x < 0.0 else x


# ═══════════════════════════ C-only: normalize_item ═══════════════════════════

struct NormItemResult(Copyable, Movable):
    var ok: Bool       # False → Python None (skipped)
    var idea: String
    var aspect_ratio: String

    def __init__(out self, ok: Bool, idea: String, aspect_ratio: String):
        self.ok = ok
        self.idea = idea
        self.aspect_ratio = aspect_ratio


def normalize_item(item: JValue, default_aspect_ratio: String) -> NormItemResult:
    """C normalize_item: bare string → (string, default_ar); dict with str prompt →
    (prompt, aspect_ratio-if-truthy-else-default); else → None. Empty-after-strip → None."""
    var idea: String
    var ar: String
    if item.is_string():
        idea = item.s
        ar = default_aspect_ratio
    elif item.is_object() and item.has_key("prompt") and item.get("prompt").is_string():
        idea = item.get("prompt").s
        # Python: ar = item.get("aspect_ratio") or default — ANY truthy value passes
        # through (a number / true / non-empty list). We carry through only a non-empty
        # STRING and fall back to the default otherwise.
        #
        # DOCUMENTED DEVIATION (FRAGILE #2, off the prompt-steered happy path): for a
        # NON-STRING truthy aspect_ratio (e.g. {"aspect_ratio":16} or true or [1,9])
        # Python would carry the raw value, but it then flows into
        # template.replace("{{aspect_ratio}}", ar) which RAISES TypeError (str.replace
        # needs a str) → Python produces no usable caption for that item anyway (it
        # crashes). Our result type is (idea, ar:String), so we cannot carry a non-str
        # value AND it would be unusable downstream regardless; falling back to the
        # default string is the strictly-more-robust, non-crashing choice. The
        # realistic cases — string AR, empty/0/false/[]/missing AR (→ default), and a
        # bare-string item — all MATCH Python (skeptic: 14/14). aspect_ratio is a
        # string in every real call site.
        var ar_v = item.get("aspect_ratio")
        if item.has_key("aspect_ratio") and ar_v.is_string() and ar_v.s.byte_length() > 0:
            ar = ar_v.s
        else:
            ar = default_aspect_ratio
    else:
        return NormItemResult(False, String(""), String(""))
    if _strip(idea).byte_length() == 0:
        return NormItemResult(False, String(""), String(""))
    return NormItemResult(True, idea, ar)


# ═══════════════════════ LLM boundary (smoke-only) ════════════════════════════
# The single non-deterministic box. A pluggable backend: given a built prompt (and,
# for B, an optional base64 image), return the raw model string. NOT byte-gated —
# the glue around it (everything above) is. A real host = an OpenAI-compatible
# Qwen3-VL-8B server (HTTP); the StubGenerator returns a fixed canned reply so the
# end-to-end glue can be smoke-exercised without a model.

trait CaptionGenerator:
    def generate(self, prompt: String, image_b64: String, has_image: Bool) raises -> String:
        ...


@fieldwise_init
struct StubGenerator(CaptionGenerator, Copyable, Movable):
    """Smoke stub: echoes a fixed structured caption regardless of input. Lets the
    full B/C glue be exercised end-to-end with no LLM. NEVER used for real captions."""

    var canned: String

    def generate(self, prompt: String, image_b64: String, has_image: Bool) raises -> String:
        return self.canned
