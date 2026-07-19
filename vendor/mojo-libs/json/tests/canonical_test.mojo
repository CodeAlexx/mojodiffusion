# Tests for json.canonical — minify / dumps_sorted / canonicalize.
# Prints REAL output (no asserts that hide values). Python cross-check is done
# externally (see final report); here we print our outputs so they can be diffed.

from json.value import JSONValue
from json.parser import loads
from json.canonical import minify, dumps_sorted, canonicalize


def _check(mut passed: Int, mut failed: Int, name: String, got: String, want: String):
    if got == want:
        passed += 1
        print("PASS:", name)
        print("    got == want:", got)
    else:
        failed += 1
        print("FAIL:", name)
        print("    got :", got)
        print("    want:", want)


def main() raises:
    var passed = 0
    var failed = 0

    print("=== json.canonical tests ===")
    print("")

    # 1. Two docs differ only in key order + whitespace -> same canonical string.
    var a = String('{"b": 2, "a": 1}')
    var b = String('{  "a" :1,\n  "b":   2 }')
    var ca = canonicalize(a)
    var cb = canonicalize(b)
    print("--- test 1: key-order + whitespace invariance ---")
    print("    canon(a):", ca)
    print("    canon(b):", cb)
    _check(passed, failed, "canonicalize order/ws invariant", ca, cb)
    _check(passed, failed, "canonical value is sorted compact", ca, String('{"a":1,"b":2}'))
    print("")

    # 2. Nested objects sort at every level.
    var nested = String('{"z":{"y":2,"x":1},"a":{"d":4,"c":3}}')
    var cn = canonicalize(nested)
    print("--- test 2: recursive nested sort ---")
    print("    canon:", cn)
    _check(passed, failed, "nested sort all levels", cn,
           String('{"a":{"c":3,"d":4},"z":{"x":1,"y":2}}'))
    print("")

    # 3. Arrays preserve element order; objects inside arrays still sort.
    var arr = String('[{"b":1,"a":2},{"d":3,"c":4}]')
    var carr = canonicalize(arr)
    print("--- test 3: arrays keep order, inner objects sort ---")
    print("    canon:", carr)
    _check(passed, failed, "array order kept, inner sorted", carr,
           String('[{"a":2,"b":1},{"c":4,"d":3}]'))
    print("")

    # 4. minify == dumps_sorted when keys already sorted.
    var sorted_doc = loads(String('{"a":1,"b":[1,2,3],"c":"x"}'))
    var mn = minify(sorted_doc)
    var ds = dumps_sorted(sorted_doc)
    print("--- test 4: minify vs dumps_sorted (pre-sorted) ---")
    print("    minify      :", mn)
    print("    dumps_sorted:", ds)
    _check(passed, failed, "minify==dumps_sorted when pre-sorted", mn, ds)
    print("")

    # 5. minify preserves insertion order; dumps_sorted reorders.
    var unsorted = loads(String('{"b":1,"a":2}'))
    var mu = minify(unsorted)
    var du = dumps_sorted(unsorted)
    print("--- test 5: minify keeps order, dumps_sorted sorts ---")
    print("    minify      :", mu)
    print("    dumps_sorted:", du)
    _check(passed, failed, "minify keeps insertion order", mu, String('{"b":1,"a":2}'))
    _check(passed, failed, "dumps_sorted sorts", du, String('{"a":2,"b":1}'))
    print("")

    # 6. Unicode key/string passes through as raw UTF-8.
    var uni = String('{"ü":"café"}')
    var cu = canonicalize(uni)
    print("--- test 6: unicode passthrough (UTF-8) ---")
    print("    canon:", cu)
    _check(passed, failed, "unicode utf-8 passthrough", cu, uni)
    print("")

    # 7. Escaping: required control chars + quote + backslash, '/' left alone.
    var esc = loads(String('{"k":"a\\"b\\\\c/d\\n\\t"}'))
    var ce = dumps_sorted(esc)
    print("--- test 7: escaping rules ---")
    print("    canon:", ce)
    _check(passed, failed, "escape quote/backslash/ctrl, keep slash", ce,
           String('{"k":"a\\"b\\\\c/d\\n\\t"}'))
    print("")

    # 8. Numbers: integers verbatim, simple floats round-trip-ish.
    var nums = loads(String('{"i":42,"neg":-7,"f":1.5,"zero":0}'))
    var cnums = dumps_sorted(nums)
    print("--- test 8: number formatting ---")
    print("    canon:", cnums)
    _check(passed, failed, "integer + simple float formatting", cnums,
           String('{"f":1.5,"i":42,"neg":-7,"zero":0}'))
    print("")

    # 9. Mixed top-level types + nested arrays/objects, full canonical re-emit.
    var big = String('{ "name":"x" , "list":[3,2,1] , "meta":{"v":1,"k":2} , "ok":true , "nil":null }')
    var cbig = canonicalize(big)
    print("--- test 9: mixed full document ---")
    print("    canon:", cbig)
    _check(passed, failed, "mixed full doc canonical", cbig,
           String('{"list":[3,2,1],"meta":{"k":2,"v":1},"name":"x","nil":null,"ok":true}'))
    print("")

    # 10. Empty object / empty array.
    var empties = canonicalize(String('{"o":{},"a":[]}'))
    print("--- test 10: empties ---")
    print("    canon:", empties)
    _check(passed, failed, "empty obj/arr", empties, String('{"a":[],"o":{}}'))
    print("")

    print("passed:", passed, "failed:", failed)
    if failed == 0:
        print("ALL CANONICAL TESTS PASSED")
    else:
        print("SOME CANONICAL TESTS FAILED")
