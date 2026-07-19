# json.patch — JSON Merge Patch (RFC 7386) + JSON Patch (RFC 6902). 100% Mojo.
#
# Both operations are implemented FUNCTIONALLY: they never mutate the input doc
# in place. Containers along the path to a change are rebuilt as new JSONValues
# (structural sharing of untouched subtrees via copy). RFC 6902 ops supported:
# add, remove, replace, move, copy, test. A failing "test" op aborts the whole
# patch (apply_patch raises).

from json.value import JSONValue
from json.serialize import dumps
from json.pointer import split_pointer


# ─────────────────────────────────────────────────────────────────────────────
# RFC 7386 — JSON Merge Patch
# ─────────────────────────────────────────────────────────────────────────────
def apply_merge_patch(doc: JSONValue, patch: JSONValue) raises -> JSONValue:
    """RFC 7386: merge `patch` into `doc`.

    If patch is not an object, it replaces the target entirely. Otherwise each
    member is merged recursively; a member whose value is `null` deletes the
    corresponding key from the target.
    """
    if not patch.is_object():
        return patch.copy()
    # Result starts as a copy of doc's object members (if doc is an object),
    # otherwise an empty object (non-object targets become objects).
    var result = JSONValue.new_object()
    if doc.is_object():
        var dkeys = doc.keys()
        for i in range(len(dkeys)):
            result.set(dkeys[i], doc[dkeys[i]])
    var pkeys = patch.keys()
    for i in range(len(pkeys)):
        var k = pkeys[i]
        var pv = patch[k]
        if pv.is_null():
            # delete: rebuild result without this key
            result = _object_without(result, k)
        else:
            var existing: JSONValue
            if result.is_object() and result.contains(k):
                existing = result[k]
            else:
                existing = JSONValue.null()
            result.set(k, apply_merge_patch(existing, pv))
    return result^


def _object_without(obj: JSONValue, drop_key: String) raises -> JSONValue:
    var out = JSONValue.new_object()
    var ks = obj.keys()
    for i in range(len(ks)):
        if ks[i] != drop_key:
            out.set(ks[i], obj[ks[i]])
    return out^


# ─────────────────────────────────────────────────────────────────────────────
# Functional container rebuild helpers
# ─────────────────────────────────────────────────────────────────────────────
def _array_with_set(arr: JSONValue, idx: Int, var v: JSONValue) raises -> JSONValue:
    var out = JSONValue.new_array()
    var n = arr.length()
    for i in range(n):
        if i == idx:
            out.append(v.copy())
        else:
            out.append(arr[i])
    return out^


def _array_with_insert(arr: JSONValue, idx: Int, var v: JSONValue) raises -> JSONValue:
    # Insert v at position idx (shifting the rest right). idx == length appends.
    var out = JSONValue.new_array()
    var n = arr.length()
    for i in range(n):
        if i == idx:
            out.append(v.copy())
        out.append(arr[i])
    if idx == n:
        out.append(v.copy())
    return out^


def _array_without(arr: JSONValue, idx: Int) raises -> JSONValue:
    var out = JSONValue.new_array()
    var n = arr.length()
    for i in range(n):
        if i != idx:
            out.append(arr[i])
    return out^


def _parse_index(tok: String) raises -> Int:
    var sb = tok.as_bytes()
    var n = tok.byte_length()
    if n == 0:
        raise Error("empty array index token")
    if n > 1 and Int(sb[0]) == 0x30:
        raise Error("array index has leading zero: " + tok)
    var v = 0
    for i in range(n):
        var c = Int(sb[i])
        if c < 0x30 or c > 0x39:
            raise Error("non-numeric array index: " + tok)
        v = v * 10 + (c - 0x30)
    return v


# ─────────────────────────────────────────────────────────────────────────────
# RFC 6902 — JSON Patch primitive operations (functional)
# ─────────────────────────────────────────────────────────────────────────────
def _do_add(doc: JSONValue, toks: List[String], depth: Int, var v: JSONValue) raises -> JSONValue:
    # Add v at the location addressed by toks[depth:].
    if depth == len(toks):
        # Adding at root replaces the whole document.
        return v^
    var tok = toks[depth]
    if doc.is_object():
        var child: JSONValue
        if doc.contains(tok):
            child = doc[tok]
        else:
            child = JSONValue.null()
        if depth + 1 == len(toks):
            # final segment: set/overwrite the member
            var out = JSONValue.new_object()
            var ks = doc.keys()
            var replaced = False
            for i in range(len(ks)):
                if ks[i] == tok:
                    out.set(ks[i], v.copy())
                    replaced = True
                else:
                    out.set(ks[i], doc[ks[i]])
            if not replaced:
                out.set(tok, v.copy())
            return out^
        else:
            if not doc.contains(tok):
                raise Error("add: path not found, missing key '" + tok + "'")
            var newchild = _do_add(child, toks, depth + 1, v^)
            var out = JSONValue.new_object()
            var ks = doc.keys()
            for i in range(len(ks)):
                if ks[i] == tok:
                    out.set(ks[i], newchild.copy())
                else:
                    out.set(ks[i], doc[ks[i]])
            return out^
    elif doc.is_array():
        if depth + 1 == len(toks):
            var idx: Int
            if tok == "-":
                idx = doc.length()
            else:
                idx = _parse_index(tok)
                if idx < 0 or idx > doc.length():
                    raise Error("add: array index out of range: " + tok)
            return _array_with_insert(doc, idx, v^)
        else:
            var idx = _parse_index(tok)
            if idx < 0 or idx >= doc.length():
                raise Error("add: array index out of range: " + tok)
            var newchild = _do_add(doc[idx], toks, depth + 1, v^)
            return _array_with_set(doc, idx, newchild^)
    else:
        raise Error("add: cannot descend into scalar at token '" + tok + "'")


def _do_remove(doc: JSONValue, toks: List[String], depth: Int) raises -> JSONValue:
    if depth == len(toks):
        raise Error("remove: cannot remove the root document")
    var tok = toks[depth]
    if doc.is_object():
        if not doc.contains(tok):
            raise Error("remove: object has no key '" + tok + "'")
        if depth + 1 == len(toks):
            return _object_without(doc, tok)
        else:
            var newchild = _do_remove(doc[tok], toks, depth + 1)
            var out = JSONValue.new_object()
            var ks = doc.keys()
            for i in range(len(ks)):
                if ks[i] == tok:
                    out.set(ks[i], newchild.copy())
                else:
                    out.set(ks[i], doc[ks[i]])
            return out^
    elif doc.is_array():
        if tok == "-":
            raise Error("remove: '-' is not a valid index for removal")
        var idx = _parse_index(tok)
        if idx < 0 or idx >= doc.length():
            raise Error("remove: array index out of range: " + tok)
        if depth + 1 == len(toks):
            return _array_without(doc, idx)
        else:
            var newchild = _do_remove(doc[idx], toks, depth + 1)
            return _array_with_set(doc, idx, newchild^)
    else:
        raise Error("remove: cannot descend into scalar at token '" + tok + "'")


def _do_replace(doc: JSONValue, toks: List[String], depth: Int, var v: JSONValue) raises -> JSONValue:
    # replace requires the target location to already exist.
    if depth == len(toks):
        return v^
    var tok = toks[depth]
    if doc.is_object():
        if not doc.contains(tok):
            raise Error("replace: object has no key '" + tok + "'")
        var newchild: JSONValue
        if depth + 1 == len(toks):
            newchild = v^
        else:
            newchild = _do_replace(doc[tok], toks, depth + 1, v^)
        var out = JSONValue.new_object()
        var ks = doc.keys()
        for i in range(len(ks)):
            if ks[i] == tok:
                out.set(ks[i], newchild.copy())
            else:
                out.set(ks[i], doc[ks[i]])
        return out^
    elif doc.is_array():
        var idx = _parse_index(tok)
        if idx < 0 or idx >= doc.length():
            raise Error("replace: array index out of range: " + tok)
        var newchild: JSONValue
        if depth + 1 == len(toks):
            newchild = v^
        else:
            newchild = _do_replace(doc[idx], toks, depth + 1, v^)
        return _array_with_set(doc, idx, newchild^)
    else:
        raise Error("replace: cannot descend into scalar at token '" + tok + "'")


def _resolve_strict(doc: JSONValue, toks: List[String]) raises -> JSONValue:
    var cur = doc.copy()
    for i in range(len(toks)):
        var tok = toks[i]
        if cur.is_object():
            if not cur.contains(tok):
                raise Error("path: object has no key '" + tok + "'")
            cur = cur[tok]
        elif cur.is_array():
            if tok == "-":
                raise Error("path: '-' references nonexistent element")
            var idx = _parse_index(tok)
            if idx < 0 or idx >= cur.length():
                raise Error("path: array index out of range: " + tok)
            cur = cur[idx]
        else:
            raise Error("path: cannot descend into scalar at token '" + tok + "'")
    return cur^


def _values_equal(a: JSONValue, b: JSONValue) raises -> Bool:
    # Structural equality via canonical serialization. dumps() emits object keys
    # in insertion order; for the test op we compare deep structure, and RFC 6902
    # "test" is defined as equality of values (order-insensitive for objects).
    # We do an explicit recursive compare to be order-insensitive for objects.
    if a.kind != b.kind:
        # int vs float numeric equality (e.g. 1 vs 1.0) — RFC treats them by
        # JSON type, but be lenient for numbers with equal value.
        if a.is_number() and b.is_number():
            return a.as_float() == b.as_float()
        return False
    if a.is_null():
        return True
    if a.is_bool():
        return a.as_bool() == b.as_bool()
    if a.is_int():
        return a.as_int() == b.as_int()
    if a.is_float():
        return a.as_float() == b.as_float()
    if a.is_string():
        return a.as_string() == b.as_string()
    if a.is_array():
        if a.length() != b.length():
            return False
        for i in range(a.length()):
            if not _values_equal(a[i], b[i]):
                return False
        return True
    if a.is_object():
        if a.length() != b.length():
            return False
        var ks = a.keys()
        for i in range(len(ks)):
            if not b.contains(ks[i]):
                return False
            if not _values_equal(a[ks[i]], b[ks[i]]):
                return False
        return True
    return False


def apply_patch(doc: JSONValue, ops: JSONValue) raises -> JSONValue:
    """RFC 6902: apply an array of patch operations, returning a new document.

    Each op is an object with "op" and "path"; add/replace/test also carry
    "value"; move/copy carry "from". A failing "test" raises and the whole
    patch is considered failed (the original doc is untouched since we work on
    copies).
    """
    if not ops.is_array():
        raise Error("apply_patch: operations must be a JSON array")
    var cur = doc.copy()
    for oi in range(ops.length()):
        var op = ops[oi]
        if not op.is_object():
            raise Error("apply_patch: each operation must be an object")
        if not op.contains("op"):
            raise Error("apply_patch: operation missing 'op'")
        var name = op["op"].as_string()
        if not op.contains("path"):
            raise Error("apply_patch: operation missing 'path'")
        var path = op["path"].as_string()
        var toks = split_pointer(path)

        if name == "add":
            if not op.contains("value"):
                raise Error("add: missing 'value'")
            cur = _do_add(cur, toks, 0, op["value"])
        elif name == "remove":
            cur = _do_remove(cur, toks, 0)
        elif name == "replace":
            if not op.contains("value"):
                raise Error("replace: missing 'value'")
            cur = _do_replace(cur, toks, 0, op["value"])
        elif name == "test":
            if not op.contains("value"):
                raise Error("test: missing 'value'")
            var actual = _resolve_strict(cur, toks)
            if not _values_equal(actual, op["value"]):
                raise Error(
                    "test failed at '" + path + "': expected "
                    + dumps(op["value"]) + " got " + dumps(actual)
                )
        elif name == "move":
            if not op.contains("from"):
                raise Error("move: missing 'from'")
            var frompath = op["from"].as_string()
            var fromtoks = split_pointer(frompath)
            # Cannot move a location into one of its own children.
            if _is_prefix(fromtoks, toks):
                raise Error("move: cannot move a value into its own child")
            var moved = _resolve_strict(cur, fromtoks)
            cur = _do_remove(cur, fromtoks, 0)
            cur = _do_add(cur, toks, 0, moved^)
        elif name == "copy":
            if not op.contains("from"):
                raise Error("copy: missing 'from'")
            var frompath = op["from"].as_string()
            var fromtoks = split_pointer(frompath)
            var copied = _resolve_strict(cur, fromtoks)
            cur = _do_add(cur, toks, 0, copied^)
        else:
            raise Error("apply_patch: unknown op '" + name + "'")
    return cur^


def _is_prefix(prefix: List[String], full: List[String]) -> Bool:
    # True if `prefix` is a (proper or equal) prefix of `full`.
    if len(prefix) > len(full):
        return False
    for i in range(len(prefix)):
        if prefix[i] != full[i]:
            return False
    return True
