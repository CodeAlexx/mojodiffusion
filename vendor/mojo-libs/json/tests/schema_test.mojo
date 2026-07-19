# json.schema tests — build schema+instance pairs (valid & invalid), print PASS/FAIL
# per case against a hand-verified expected verdict, then CROSS-CHECK every pair
# against Python `jsonschema` (Draft-07) if importable.

from json.parser import loads
from json.value import JSONValue
from json.schema import validate, ValidationResult
from std.python import Python, PythonObject


struct Tally(Copyable, Movable):
    var passed: Int
    var failed: Int
    # cross-check accounting
    var xc_avail: Bool
    var xc_checked: Int
    var xc_match: Int

    def __init__(out self):
        self.passed = 0
        self.failed = 0
        self.xc_avail = False
        self.xc_checked = 0
        self.xc_match = 0


def chk(
    mut t: Tally,
    name: String,
    schema_src: String,
    inst_src: String,
    expect_valid: Bool,
    py: PythonObject,
    have_py: Bool,
) raises:
    var schema = loads(schema_src)
    var inst = loads(inst_src)
    var res = validate(inst, schema)

    var ok = res.valid == expect_valid
    if ok:
        t.passed += 1
        print("  PASS:", name, "(valid =", res.valid, ")")
    else:
        t.failed += 1
        print("  FAIL:", name, "expected valid =", expect_valid, "got", res.valid)
    if not res.valid:
        for ei in range(len(res.errors)):
            print("        err:", res.errors[ei])

    # ── cross-check against Python jsonschema ──
    if have_py:
        t.xc_avail = True
        var py_valid = _py_validate(py, schema_src, inst_src)
        t.xc_checked += 1
        if py_valid == res.valid:
            t.xc_match += 1
            print("        xcheck: jsonschema agrees (valid =", py_valid, ")")
        else:
            print("        xcheck MISMATCH: mojo valid =", res.valid, " jsonschema valid =", py_valid)


def _py_validate(py: PythonObject, schema_src: String, inst_src: String) raises -> Bool:
    # py is the `jsonschema` module; use the json module too.
    var json_mod = Python.import_module("json")
    var schema_obj = json_mod.loads(schema_src.to_python_object())
    var inst_obj = json_mod.loads(inst_src.to_python_object())
    var validator_cls = py.Draft7Validator
    var validator = validator_cls(schema_obj)
    var is_valid = validator.is_valid(inst_obj)
    return Bool(py=is_valid)


def main() raises:
    var t = Tally()

    # Try to import jsonschema for cross-checking
    var have_py: Bool
    var py = Python.none()
    try:
        py = Python.import_module("jsonschema")
        have_py = True
        print("jsonschema importable -> cross-checking every case against Draft7Validator")
    except e:
        have_py = False
        print("jsonschema NOT importable (", String(e), ") -> relying on hand-verified verdicts")

    print("")

    # ── type ──
    chk(t, "type string ok", '{"type":"string"}', '"hello"', True, py, have_py)
    chk(t, "type string fail", '{"type":"string"}', '42', False, py, have_py)
    chk(t, "type integer ok (int)", '{"type":"integer"}', '7', True, py, have_py)
    chk(t, "type integer ok (5.0)", '{"type":"integer"}', '5.0', True, py, have_py)
    chk(t, "type integer fail (5.5)", '{"type":"integer"}', '5.5', False, py, have_py)
    chk(t, "type number ok (float)", '{"type":"number"}', '3.14', True, py, have_py)
    chk(t, "type array of types ok", '{"type":["string","null"]}', 'null', True, py, have_py)
    chk(t, "type array of types fail", '{"type":["string","null"]}', '5', False, py, have_py)
    chk(t, "type boolean ok", '{"type":"boolean"}', 'true', True, py, have_py)
    chk(t, "type object ok", '{"type":"object"}', '{}', True, py, have_py)

    # ── const ──
    chk(t, "const ok", '{"const":42}', '42', True, py, have_py)
    chk(t, "const fail", '{"const":42}', '43', False, py, have_py)
    chk(t, "const object ok", '{"const":{"a":1,"b":[2,3]}}', '{"b":[2,3],"a":1}', True, py, have_py)

    # ── enum ──
    chk(t, "enum ok", '{"enum":["red","green","blue"]}', '"green"', True, py, have_py)
    chk(t, "enum MISS", '{"enum":["red","green","blue"]}', '"yellow"', False, py, have_py)
    chk(t, "enum mixed ok", '{"enum":[1,"two",true,null]}', 'null', True, py, have_py)

    # ── numbers ──
    chk(t, "minimum ok", '{"minimum":10}', '10', True, py, have_py)
    chk(t, "minimum fail", '{"minimum":10}', '9', False, py, have_py)
    chk(t, "maximum fail", '{"maximum":5}', '6', False, py, have_py)
    chk(t, "exclusiveMinimum fail (eq)", '{"exclusiveMinimum":10}', '10', False, py, have_py)
    chk(t, "exclusiveMaximum ok", '{"exclusiveMaximum":10}', '9.99', True, py, have_py)
    chk(t, "multipleOf ok", '{"multipleOf":3}', '9', True, py, have_py)
    chk(t, "multipleOf fail", '{"multipleOf":3}', '10', False, py, have_py)
    chk(t, "multipleOf 0.5 ok", '{"multipleOf":0.5}', '2.5', True, py, have_py)

    # ── strings ──
    chk(t, "minLength ok", '{"minLength":3}', '"abc"', True, py, have_py)
    chk(t, "minLength fail", '{"minLength":3}', '"ab"', False, py, have_py)
    chk(t, "maxLength fail", '{"maxLength":3}', '"abcd"', False, py, have_py)
    # pattern subset:
    chk(t, "pattern literal+class ok", '{"pattern":"^a[0-9]+z$"}', '"a123z"', True, py, have_py)
    chk(t, "pattern literal+class fail", '{"pattern":"^a[0-9]+z$"}', '"az"', False, py, have_py)
    chk(t, "pattern dot-star ok", '{"pattern":"^h.*o$"}', '"hello"', True, py, have_py)
    chk(t, "pattern unanchored search", '{"pattern":"b[0-9]c"}', '"xxb7cyy"', True, py, have_py)
    chk(t, "pattern negated class", '{"pattern":"^[^0-9]+$"}', '"abc"', True, py, have_py)
    chk(t, "pattern negated class fail", '{"pattern":"^[^0-9]+$"}', '"ab2c"', False, py, have_py)
    chk(t, "pattern optional ok", '{"pattern":"^colou?r$"}', '"color"', True, py, have_py)
    chk(t, "pattern optional ok2", '{"pattern":"^colou?r$"}', '"colour"', True, py, have_py)

    # ── arrays ──
    chk(t, "items ok", '{"items":{"type":"integer"}}', '[1,2,3]', True, py, have_py)
    chk(t, "items fail nested", '{"items":{"type":"integer"}}', '[1,"x",3]', False, py, have_py)
    chk(t, "minItems fail", '{"minItems":2}', '[1]', False, py, have_py)
    chk(t, "maxItems fail", '{"maxItems":2}', '[1,2,3]', False, py, have_py)
    chk(t, "uniqueItems ok", '{"uniqueItems":true}', '[1,2,3]', True, py, have_py)
    chk(t, "uniqueItems FAIL", '{"uniqueItems":true}', '[1,2,2,3]', False, py, have_py)
    chk(t, "uniqueItems obj FAIL", '{"uniqueItems":true}', '[{"a":1},{"a":1}]', False, py, have_py)

    # ── objects ──
    chk(t, "required ok", '{"required":["name"]}', '{"name":"x"}', True, py, have_py)
    chk(t, "required MISSING", '{"required":["name","age"]}', '{"name":"x"}', False, py, have_py)
    chk(t, "minProperties fail", '{"minProperties":2}', '{"a":1}', False, py, have_py)
    chk(t, "maxProperties fail", '{"maxProperties":1}', '{"a":1,"b":2}', False, py, have_py)
    chk(
        t,
        "additionalProperties false fail",
        '{"properties":{"a":{}},"additionalProperties":false}',
        '{"a":1,"b":2}',
        False,
        py,
        have_py,
    )
    chk(
        t,
        "additionalProperties schema fail",
        '{"properties":{"a":{}},"additionalProperties":{"type":"string"}}',
        '{"a":1,"b":2}',
        False,
        py,
        have_py,
    )
    chk(
        t,
        "properties typed ok",
        '{"properties":{"a":{"type":"integer"},"b":{"type":"string"}}}',
        '{"a":1,"b":"x"}',
        True,
        py,
        have_py,
    )
    chk(
        t,
        "properties typed fail",
        '{"properties":{"a":{"type":"integer"}}}',
        '{"a":"not-int"}',
        False,
        py,
        have_py,
    )

    # ── nested object schema ──
    var nested_schema = String(
        '{"type":"object","required":["user"],"properties":{'
        + '"user":{"type":"object","required":["name","age"],"properties":{'
        + '"name":{"type":"string","minLength":1},'
        + '"age":{"type":"integer","minimum":0,"maximum":150},'
        + '"tags":{"type":"array","items":{"type":"string"},"uniqueItems":true}'
        + '}}}}'
    )
    chk(
        t,
        "nested object ok",
        nested_schema,
        '{"user":{"name":"Ada","age":36,"tags":["a","b"]}}',
        True,
        py,
        have_py,
    )
    chk(
        t,
        "nested object fail (age type + dup tag)",
        nested_schema,
        '{"user":{"name":"Ada","age":"old","tags":["a","a"]}}',
        False,
        py,
        have_py,
    )

    # ── combinators ──
    chk(t, "allOf ok", '{"allOf":[{"type":"integer"},{"minimum":5}]}', '7', True, py, have_py)
    chk(t, "allOf fail", '{"allOf":[{"type":"integer"},{"minimum":5}]}', '3', False, py, have_py)
    chk(
        t,
        "anyOf ok",
        '{"anyOf":[{"type":"string"},{"type":"integer"}]}',
        '"hi"',
        True,
        py,
        have_py,
    )
    chk(
        t,
        "anyOf FAIL",
        '{"anyOf":[{"type":"string"},{"type":"integer"}]}',
        '3.14',
        False,
        py,
        have_py,
    )
    chk(
        t,
        "oneOf ok (exactly 1)",
        '{"oneOf":[{"type":"string"},{"type":"integer"}]}',
        '5',
        True,
        py,
        have_py,
    )
    chk(
        t,
        "oneOf fail (matches 2)",
        '{"oneOf":[{"minimum":0},{"maximum":100}]}',
        '50',
        False,
        py,
        have_py,
    )
    chk(t, "not ok", '{"not":{"type":"string"}}', '5', True, py, have_py)
    chk(t, "not FAIL", '{"not":{"type":"string"}}', '"x"', False, py, have_py)

    # ── pointer-location check on a required-missing error ──
    print("")
    print("pointer-location check (required-missing):")
    var ps = loads(String('{"properties":{"user":{"required":["age"]}}}'))
    var pi = loads(String('{"user":{"name":"x"}}'))
    var pr = validate(pi, ps)
    var want_ptr = String("at /user/age: required: property is missing")
    var found_ptr = False
    for ei in range(len(pr.errors)):
        print("  error:", pr.errors[ei])
        if pr.errors[ei] == want_ptr:
            found_ptr = True
    if found_ptr:
        t.passed += 1
        print("  PASS: pointer path is /user/age")
    else:
        t.failed += 1
        print("  FAIL: expected pointer", want_ptr)

    # ── summary ──
    print("")
    print("passed:", t.passed, "failed:", t.failed)
    if t.xc_avail:
        print(
            "jsonschema cross-check: matched",
            t.xc_match,
            "of",
            t.xc_checked,
            "(",
            t.xc_match,
            "/",
            t.xc_checked,
            ")",
        )
        if t.xc_match != t.xc_checked:
            print("CROSS-CHECK HAD MISMATCHES")
    else:
        print("jsonschema cross-check: UNAVAILABLE (hand-verified verdicts only)")

    if t.failed == 0 and (not t.xc_avail or t.xc_match == t.xc_checked):
        print("ALL SCHEMA TESTS PASSED")
    else:
        print("SOME SCHEMA TESTS FAILED")
