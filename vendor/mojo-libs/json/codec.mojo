# json.codec — reflection-driven validation + typed extraction for JSON models.
# 100% Mojo, no FFI. Built on json.value + json.parser + std.reflection.
#
# WHAT'S REFLECTION-DRIVEN AND GENERIC (works over ANY struct T):
#   validate_required[T](obj)  — every field of T must be present in the JSON
#   reject_unknown[T](obj)     — the JSON has no keys that aren't fields of T
#   field_list[T]()            — the model's field names (schema building block)
# These iterate reflect[T]().field_names() (uniform StaticString) at comptime.
#
# TYPED VALUE BINDING uses the req_*/opt_* validated extractors below. A model's
# `from_json` validates with reflection, then binds each field with one line.
#
# NOTE (measured, Mojo 1.0.0b1): a fully-automatic per-field-type codec — looping
# over fields and coercing each by its own type with no per-field code — is NOT
# expressible here: inside a comptime-for / generic body the field's type stays
# symbolic, so type-specific dispatch (overload / assignment) won't resolve; only
# literal-index access at a concrete type does. Newer Mojo reflection (field types
# usable in type position) appears to lift this. Until then, value binding is the
# one-line-per-field step below; validation is fully generic.

from std.reflection import reflect
from json.value import JSONValue


# ── generic, reflection-driven validation ────────────────────────────────────
def validate_required[T: AnyType](obj: JSONValue) raises:
    """Raise unless every field of T is present in `obj`."""
    var names = reflect[T]().field_names()
    comptime for i in range(reflect[T]().field_count()):
        var nm = String(names[i])
        if not obj.contains(nm):
            raise Error("missing required field: " + nm)


def reject_unknown[T: AnyType](obj: JSONValue) raises:
    """Raise if `obj` has any key that is not a field of T."""
    var names = reflect[T]().field_names()
    var keys = obj.keys()
    for ki in range(len(keys)):
        var k = keys[ki]
        var ok = False
        comptime for i in range(reflect[T]().field_count()):
            if String(names[i]) == k:
                ok = True
        if not ok:
            raise Error("unknown field: " + k)


def field_list[T: AnyType]() -> String:
    """The model's field names as a JSON array string (schema building block)."""
    var names = reflect[T]().field_names()
    var out = String("[")
    comptime for i in range(reflect[T]().field_count()):
        if i > 0:
            out += ","
        out += '"' + String(names[i]) + '"'
    out += "]"
    return out


# ── typed validated extractors (raise a 422-style message on violation) ──────
def verr(err_type: String, field: String, msg: String) -> String:
    """A FastAPI-shaped validation-error entry (type/loc/msg). The raised Error's
    message IS this JSON object; a handler wraps it as {"detail":[ ... ]}."""
    return (
        '{"type":"' + err_type + '","loc":["body","' + field + '"],"msg":"' + msg + '"}'
    )


def req_int(obj: JSONValue, key: String) raises -> Int:
    if not obj.contains(key):
        raise Error(verr("missing", key, "Field required"))
    var v = obj[key]
    if not v.is_number():
        raise Error(verr("int_parsing", key, "Input should be a valid integer"))
    return v.as_int()


def req_float(obj: JSONValue, key: String) raises -> Float64:
    if not obj.contains(key):
        raise Error(verr("missing", key, "Field required"))
    var v = obj[key]
    if not v.is_number():
        raise Error(verr("float_parsing", key, "Input should be a valid number"))
    return v.as_float()


def req_str(obj: JSONValue, key: String) raises -> String:
    if not obj.contains(key):
        raise Error(verr("missing", key, "Field required"))
    var v = obj[key]
    if not v.is_string():
        raise Error(verr("string_type", key, "Input should be a valid string"))
    return v.as_string()


def req_bool(obj: JSONValue, key: String) raises -> Bool:
    if not obj.contains(key):
        raise Error(verr("missing", key, "Field required"))
    var v = obj[key]
    if not v.is_bool():
        raise Error(verr("bool_type", key, "Input should be a valid boolean"))
    return v.as_bool()


def opt_int(obj: JSONValue, key: String, default: Int) raises -> Int:
    if not obj.contains(key):
        return default
    return req_int(obj, key)


def opt_str(obj: JSONValue, key: String, default: String) raises -> String:
    if not obj.contains(key):
        return default
    return req_str(obj, key)


def opt_bool(obj: JSONValue, key: String, default: Bool) raises -> Bool:
    if not obj.contains(key):
        return default
    return req_bool(obj, key)
