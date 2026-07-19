# json.schema — JSON Schema validation (a practical Draft-07 subset). 100% Mojo, no FFI.
#
# validate(instance, schema) -> ValidationResult { valid: Bool, errors: List[String] }
# collects ALL errors, each tagged with a JSON-Pointer location (RFC 6901).
#
# ── Implemented keyword subset (Draft-07) ───────────────────────────────────────
#   type                 : string or array of: null|boolean|object|array|number|
#                          integer|string
#   const                : exact JSON-value equality
#   enum                 : membership in a list of JSON values
#   required             : array of required property names
#   properties           : per-key subschema
#   additionalProperties : bool (allow/deny extras) OR subschema applied to extras
#   minProperties / maxProperties
#   numbers : minimum / maximum / exclusiveMinimum / exclusiveMaximum / multipleOf
#   strings : minLength / maxLength (codepoint counts) / pattern (small regex subset)
#   arrays  : items (single subschema, applied to every element) /
#             minItems / maxItems / uniqueItems
#   combinators : allOf / anyOf / oneOf / not
#
# ── NOT implemented (silently ignored if present) ───────────────────────────────
#   $ref / $defs / definitions, if/then/else, dependencies, propertyNames,
#   patternProperties, contains, tuple-form `items` (array) + additionalItems,
#   format, $schema/$id, default, etc.
#
# ── pattern regex subset (HONEST LIMITS) ────────────────────────────────────────
#   Supported metachars : .  *  +  ?  ^  $  and char classes [...] (incl. ranges
#                         a-z and negation [^...]).
#   `*` `+` `?` quantify the SINGLE preceding atom (a literal char, `.`, or a
#   class). There is NO grouping `()`, NO alternation `|`, NO `{m,n}`, NO escape
#   sequences like \d \w \s, NO backreferences. A backslash escapes the next
#   char to a literal. An UNANCHORED pattern matches if it occurs ANYWHERE in the
#   string (substring search, like Python's re.search, which jsonschema uses).
#   This is approximate: patterns using unsupported features will mis-validate.

from json.value import JSONValue


# ── result type ─────────────────────────────────────────────────────────────────
struct ValidationResult(Copyable, Movable):
    var valid: Bool
    var errors: List[String]

    def __init__(out self):
        self.valid = True
        self.errors = List[String]()

    def add(mut self, path: String, msg: String):
        self.valid = False
        self.errors.append(String("at ") + (path if path.byte_length() > 0 else String("/")) + ": " + msg)


# ── JSON-Pointer helpers (RFC 6901 escaping) ─────────────────────────────────────
def _esc_token(tok: String) -> String:
    # ~ -> ~0, / -> ~1
    var out = String("")
    for cp in tok.codepoint_slices():
        var s = String(cp)
        if s == "~":
            out += "~0"
        elif s == "/":
            out += "~1"
        else:
            out += s
    return out


def _join(path: String, token: String) -> String:
    return path + "/" + _esc_token(token)


def _join_idx(path: String, idx: Int) -> String:
    return path + "/" + String(idx)


# ── deep JSON equality (for const / enum / uniqueItems) ──────────────────────────
def _json_equal(a: JSONValue, b: JSONValue) raises -> Bool:
    # numeric: compare int/float across kinds by value
    if a.is_number() and b.is_number():
        # treat 1 and 1.0 as equal (JSON Schema const/enum use value equality)
        if a.is_int() and b.is_int():
            return a.as_int() == b.as_int()
        return a.as_float() == b.as_float()
    if a.kind != b.kind:
        return False
    if a.is_null():
        return True
    if a.is_bool():
        return a.as_bool() == b.as_bool()
    if a.is_string():
        return a.as_string() == b.as_string()
    if a.is_array():
        if a.length() != b.length():
            return False
        for i in range(a.length()):
            if not _json_equal(a[i], b[i]):
                return False
        return True
    if a.is_object():
        if a.length() != b.length():
            return False
        var ak = a.keys()
        for ki in range(len(ak)):
            var k = ak[ki]
            if not b.contains(k):
                return False
            if not _json_equal(a[k], b[k]):
                return False
        return True
    return False


# ── type-name matching for `type` keyword ────────────────────────────────────────
def _matches_type(inst: JSONValue, tname: String) -> Bool:
    if tname == "null":
        return inst.is_null()
    if tname == "boolean":
        return inst.is_bool()
    if tname == "object":
        return inst.is_object()
    if tname == "array":
        return inst.is_array()
    if tname == "string":
        return inst.is_string()
    if tname == "number":
        return inst.is_number()
    if tname == "integer":
        # integer = an int value, or a float with no fractional part
        if inst.is_int():
            return True
        if inst.is_float():
            var f = inst.as_float()
            return f == Float64(Int(f))
        return False
    return False  # unknown type name → no match


# ── small regex engine (subset) ──────────────────────────────────────────────────
# Atom kinds
comptime AT_LIT = 0    # literal codepoint
comptime AT_DOT = 1    # .
comptime AT_CLASS = 2  # [...]


struct _Atom(Copyable, Movable):
    var kind: Int
    var lit: Int             # codepoint for AT_LIT
    var negate: Bool         # for AT_CLASS
    var cls_singles: List[Int]
    var cls_lo: List[Int]    # range lows
    var cls_hi: List[Int]    # range highs
    var quant: Int           # 0=none(exactly 1), 1='*', 2='+', 3='?'

    def __init__(out self):
        self.kind = AT_LIT
        self.lit = 0
        self.negate = False
        self.cls_singles = List[Int]()
        self.cls_lo = List[Int]()
        self.cls_hi = List[Int]()
        self.quant = 0


struct _Regex(Copyable, Movable):
    var anchored_start: Bool
    var anchored_end: Bool
    var atoms: List[_Atom]

    def __init__(out self):
        self.anchored_start = False
        self.anchored_end = False
        self.atoms = List[_Atom]()


def _cps(s: String) -> List[Int]:
    var out = List[Int]()
    for cp in s.codepoints():
        out.append(Int(cp))
    return out^


def _compile_regex(pattern: String) raises -> _Regex:
    var rx = _Regex()
    var p = _cps(pattern)
    var n = len(p)
    var i = 0
    if n > 0 and p[0] == 0x5E:  # ^
        rx.anchored_start = True
        i = 1
    while i < n:
        var c = p[i]
        if c == 0x24 and i == n - 1:  # $ only meaningful at end
            rx.anchored_end = True
            i += 1
            continue
        var atom = _Atom()
        if c == 0x2E:  # .
            atom.kind = AT_DOT
            i += 1
        elif c == 0x5B:  # [
            atom.kind = AT_CLASS
            i += 1
            if i < n and p[i] == 0x5E:  # [^
                atom.negate = True
                i += 1
            # a leading ] is a literal ] in a class
            var first = True
            while i < n and (p[i] != 0x5D or first):
                first = False
                var ch = p[i]
                if ch == 0x5C and i + 1 < n:  # escape inside class
                    i += 1
                    atom.cls_singles.append(p[i])
                    i += 1
                    continue
                # range?  a-z   (need next two: '-' then a non-']')
                if i + 2 < n and p[i + 1] == 0x2D and p[i + 2] != 0x5D:
                    atom.cls_lo.append(ch)
                    atom.cls_hi.append(p[i + 2])
                    i += 3
                else:
                    atom.cls_singles.append(ch)
                    i += 1
            if i >= n:
                raise Error("unterminated character class in pattern")
            i += 1  # consume ]
        elif c == 0x5C:  # backslash → literal next char
            if i + 1 >= n:
                raise Error("trailing backslash in pattern")
            atom.kind = AT_LIT
            atom.lit = p[i + 1]
            i += 2
        else:
            atom.kind = AT_LIT
            atom.lit = c
            i += 1
        # quantifier?
        if i < n:
            var q = p[i]
            if q == 0x2A:  # *
                atom.quant = 1
                i += 1
            elif q == 0x2B:  # +
                atom.quant = 2
                i += 1
            elif q == 0x3F:  # ?
                atom.quant = 3
                i += 1
        rx.atoms.append(atom^)
    return rx^


def _atom_matches(atom: _Atom, ch: Int) -> Bool:
    if atom.kind == AT_DOT:
        return True
    if atom.kind == AT_LIT:
        return atom.lit == ch
    # AT_CLASS
    var hit = False
    for k in range(len(atom.cls_singles)):
        if atom.cls_singles[k] == ch:
            hit = True
            break
    if not hit:
        for k in range(len(atom.cls_lo)):
            if ch >= atom.cls_lo[k] and ch <= atom.cls_hi[k]:
                hit = True
                break
    return hit != atom.negate


# Backtracking matcher: returns True if atoms[ai:] match text[ti:] with end-anchor.
def _match_here(rx: _Regex, text: List[Int], atoms: List[_Atom], ai: Int, ti: Int) -> Bool:
    var nat = len(atoms)
    var ntx = len(text)
    if ai == nat:
        if rx.anchored_end:
            return ti == ntx
        return True
    var atom = atoms[ai].copy()
    var q = atom.quant
    if q == 0:  # exactly one
        if ti < ntx and _atom_matches(atom, text[ti]):
            return _match_here(rx, text, atoms, ai + 1, ti + 1)
        return False
    if q == 3:  # ? (0 or 1) — greedy
        if ti < ntx and _atom_matches(atom, text[ti]):
            if _match_here(rx, text, atoms, ai + 1, ti + 1):
                return True
        return _match_here(rx, text, atoms, ai + 1, ti)
    # q == 1 (*) or q == 2 (+): greedy, consume as many as possible then backtrack
    var count = 0
    var t = ti
    while t < ntx and _atom_matches(atom, text[t]):
        t += 1
        count += 1
    var min_needed = 1 if q == 2 else 0
    while count >= min_needed:
        if _match_here(rx, text, atoms, ai + 1, ti + count):
            return True
        count -= 1
    return False


def _regex_search(rx: _Regex, s: String) -> Bool:
    var text = _cps(s)
    var ntx = len(text)
    if rx.anchored_start:
        return _match_here(rx, text, rx.atoms, 0, 0)
    # unanchored: try every start position (re.search semantics)
    for start in range(ntx + 1):
        if _match_here(rx, text, rx.atoms, 0, start):
            return True
    return False


# ── core validation ──────────────────────────────────────────────────────────────
def validate(instance: JSONValue, schema: JSONValue) raises -> ValidationResult:
    var res = ValidationResult()
    _validate(instance, schema, String(""), res)
    return res^


def _validate(inst: JSONValue, schema: JSONValue, path: String, mut res: ValidationResult) raises:
    # A boolean schema: true = always valid, false = always invalid.
    if schema.is_bool():
        if not schema.as_bool():
            res.add(path, "schema is `false`: nothing is valid here")
        return
    if not schema.is_object():
        return  # non-object, non-bool schema → treat as permissive

    # ── type ──
    if schema.contains("type"):
        var t = schema["type"]
        var ok = False
        if t.is_string():
            ok = _matches_type(inst, t.as_string())
            if not ok:
                res.add(path, "type mismatch: expected " + t.as_string())
        elif t.is_array():
            var names = String("")
            for ti in range(t.length()):
                var nm = t[ti].as_string()
                if ti > 0:
                    names += "|"
                names += nm
                if _matches_type(inst, nm):
                    ok = True
            if not ok:
                res.add(path, "type mismatch: expected one of [" + names + "]")

    # ── const ──
    if schema.contains("const"):
        if not _json_equal(inst, schema["const"]):
            res.add(path, "const mismatch: value not equal to required const")

    # ── enum ──
    if schema.contains("enum"):
        var en = schema["enum"]
        var found = False
        if en.is_array():
            for ei in range(en.length()):
                if _json_equal(inst, en[ei]):
                    found = True
                    break
        if not found:
            res.add(path, "enum mismatch: value not in allowed set")

    # ── numbers ──
    if inst.is_number():
        var x = inst.as_float()
        if schema.contains("minimum"):
            var m = schema["minimum"].as_float()
            if x < m:
                res.add(path, "minimum: " + String(x) + " < " + String(m))
        if schema.contains("maximum"):
            var m = schema["maximum"].as_float()
            if x > m:
                res.add(path, "maximum: " + String(x) + " > " + String(m))
        if schema.contains("exclusiveMinimum"):
            var m = schema["exclusiveMinimum"].as_float()
            if x <= m:
                res.add(path, "exclusiveMinimum: " + String(x) + " <= " + String(m))
        if schema.contains("exclusiveMaximum"):
            var m = schema["exclusiveMaximum"].as_float()
            if x >= m:
                res.add(path, "exclusiveMaximum: " + String(x) + " >= " + String(m))
        if schema.contains("multipleOf"):
            var d = schema["multipleOf"].as_float()
            if d <= 0.0:
                res.add(path, "multipleOf: divisor must be > 0")
            else:
                var ratio = x / d
                var rounded = Float64(Int(ratio + (0.5 if ratio >= 0.0 else -0.5)))
                var diff = ratio - rounded
                var ad = diff if diff >= 0.0 else -diff
                if ad > 1e-9:
                    res.add(path, "multipleOf: " + String(x) + " is not a multiple of " + String(d))

    # ── strings ──
    if inst.is_string():
        var s = inst.as_string()
        var slen = s.count_codepoints()
        if schema.contains("minLength"):
            var ml = schema["minLength"].as_int()
            if slen < ml:
                res.add(path, "minLength: length " + String(slen) + " < " + String(ml))
        if schema.contains("maxLength"):
            var ml = schema["maxLength"].as_int()
            if slen > ml:
                res.add(path, "maxLength: length " + String(slen) + " > " + String(ml))
        if schema.contains("pattern"):
            var pat = schema["pattern"].as_string()
            var rx = _compile_regex(pat)
            if not _regex_search(rx, s):
                res.add(path, "pattern: value does not match /" + pat + "/")

    # ── arrays ──
    if inst.is_array():
        var alen = inst.length()
        if schema.contains("minItems"):
            var mi = schema["minItems"].as_int()
            if alen < mi:
                res.add(path, "minItems: " + String(alen) + " < " + String(mi))
        if schema.contains("maxItems"):
            var mi = schema["maxItems"].as_int()
            if alen > mi:
                res.add(path, "maxItems: " + String(alen) + " > " + String(mi))
        if schema.contains("uniqueItems") and schema["uniqueItems"].as_bool():
            var dup = False
            for ai in range(alen):
                for bj in range(ai + 1, alen):
                    if _json_equal(inst[ai], inst[bj]):
                        dup = True
                        res.add(path, "uniqueItems: duplicate at indices " + String(ai) + " and " + String(bj))
                        break
                if dup:
                    break
        if schema.contains("items"):
            var isub = schema["items"]
            for ai in range(alen):
                _validate(inst[ai], isub, _join_idx(path, ai), res)

    # ── objects ──
    if inst.is_object():
        var nprops = inst.length()
        if schema.contains("minProperties"):
            var mp = schema["minProperties"].as_int()
            if nprops < mp:
                res.add(path, "minProperties: " + String(nprops) + " < " + String(mp))
        if schema.contains("maxProperties"):
            var mp = schema["maxProperties"].as_int()
            if nprops > mp:
                res.add(path, "maxProperties: " + String(nprops) + " > " + String(mp))

        if schema.contains("required"):
            var req = schema["required"]
            if req.is_array():
                for ri in range(req.length()):
                    var rname = req[ri].as_string()
                    if not inst.contains(rname):
                        res.add(_join(path, rname), "required: property is missing")

        # properties + additionalProperties
        var has_props = schema.contains("properties")
        var props = schema["properties"] if has_props else JSONValue.new_object()
        var has_addl = schema.contains("additionalProperties")
        var addl = schema["additionalProperties"] if has_addl else JSONValue.from_bool(True)

        var ikeys = inst.keys()
        for ki in range(len(ikeys)):
            var k = ikeys[ki]
            if has_props and props.contains(k):
                _validate(inst[k], props[k], _join(path, k), res)
            else:
                # additional property
                if has_addl:
                    if addl.is_bool():
                        if not addl.as_bool():
                            res.add(_join(path, k), "additionalProperties: property not allowed")
                    else:
                        _validate(inst[k], addl, _join(path, k), res)

    # ── combinators ──
    if schema.contains("allOf"):
        var lst = schema["allOf"]
        if lst.is_array():
            for ci in range(lst.length()):
                _validate(inst, lst[ci], path, res)

    if schema.contains("anyOf"):
        var lst = schema["anyOf"]
        if lst.is_array():
            var any_ok = False
            for ci in range(lst.length()):
                var sub = ValidationResult()
                _validate(inst, lst[ci], path, sub)
                if sub.valid:
                    any_ok = True
                    break
            if not any_ok:
                res.add(path, "anyOf: value did not match any subschema")

    if schema.contains("oneOf"):
        var lst = schema["oneOf"]
        if lst.is_array():
            var match_count = 0
            for ci in range(lst.length()):
                var sub = ValidationResult()
                _validate(inst, lst[ci], path, sub)
                if sub.valid:
                    match_count += 1
            if match_count != 1:
                res.add(path, "oneOf: value matched " + String(match_count) + " subschemas (must be exactly 1)")

    if schema.contains("not"):
        var sub = ValidationResult()
        _validate(inst, schema["not"], path, sub)
        if sub.valid:
            res.add(path, "not: value must NOT match the subschema, but it did")
