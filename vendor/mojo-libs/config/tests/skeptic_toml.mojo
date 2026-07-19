# Adversarial probes for config/toml.mojo. Reports observed behavior verbatim.

from config.value import ConfigTree, ConfigValue, CV_INT, CV_FLOAT, CV_STR, CV_BOOL, CV_ARRAY
from config.toml import parse_toml


def _kind_name(k: Int) -> String:
    if k == CV_INT: return String("int")
    if k == CV_FLOAT: return String("float")
    if k == CV_STR: return String("str")
    if k == CV_BOOL: return String("bool")
    if k == CV_ARRAY: return String("array")
    return String("null")


def _show(label: String, text: String, section: String, key: String) raises:
    try:
        var t = parse_toml(text)
        if t.has(section, key):
            var v = t.get(section, key)
            var rep = String("")
            if v.kind == CV_INT: rep = String(v.i)
            elif v.kind == CV_FLOAT: rep = String(v.f)
            elif v.kind == CV_STR: rep = v.s
            elif v.kind == CV_BOOL: rep = String("true") if v.b else String("false")
            elif v.kind == CV_ARRAY: rep = String("arr-len=") + String(len(v.arr))
            print(label, ": kind=", _kind_name(v.kind), " val='", rep, "'")
        else:
            print(label, ": MISSING [", section, "]", key)
    except e:
        print(label, ": RAISED:", String(e))


def main() raises:
    # T1: underscores in floats 1_000.5
    _show(String("T1 float-underscore"), String("x = 1_000.5\n"), String(""), String("x"))
    # T2: negative float
    _show(String("T2 neg-float"), String("x = -0.5\n"), String(""), String("x"))
    # T3: exponent 1e10
    _show(String("T3 exp"), String("x = 1e10\n"), String(""), String("x"))
    # T4: +0
    _show(String("T4 plus-zero"), String("x = +0\n"), String(""), String("x"))
    # T5: leading-zero int 01 (tomllib REJECTS)
    _show(String("T5 leading-zero"), String("x = 01\n"), String(""), String("x"))
    # T5b: 007
    _show(String("T5b leading-zero-007"), String("x = 007\n"), String(""), String("x"))
    # T6: empty array
    _show(String("T6 empty-array"), String("x = []\n"), String(""), String("x"))
    # T7: array with trailing comma
    _show(String("T7 trailing-comma"), String("x = [1, 2, 3,]\n"), String(""), String("x"))
    # T8: nested arrays
    _show(String("T8 nested-array"), String("x = [[1,2],[3]]\n"), String(""), String("x"))
    # T9: escaped unicode é (é)
    _show(String("T9 unicode-escape"), String('x = "caf\\u00e9"\n'), String(""), String("x"))
    # T10: literal string with backslashes
    _show(String("T10 literal-backslash"), String("x = 'C:\\\\path'\n"), String(""), String("x"))
    # T11: '#' inside quoted string must NOT be a comment
    _show(String("T11 hash-in-string"), String('x = "a#b"\n'), String(""), String("x"))
    # T11b: '#' inside literal string
    _show(String("T11b hash-in-literal"), String("x = 'a#b'\n"), String(""), String("x"))
    # T12: dotted key under a table
    var t12 = parse_toml(String("[srv]\na.b = 1\n"))
    print("T12 dotted-under-table: [srv.a] has b? ", t12.has(String("srv.a"), String("b")))
    # T13: top-level dotted
    var t13 = parse_toml(String("a.b.c = 1\n"))
    print("T13 top-dotted: [a.b] has c? ", t13.has(String("a.b"), String("c")))

    # ── OUT OF SCOPE: must RAISE, not silently mis-parse ──
    _show(String("T14 inline-table"), String("x = { a = 1 }\n"), String(""), String("x"))
    # array-of-tables
    try:
        var t = parse_toml(String("[[x]]\na = 1\n"))
        print("T15 array-of-tables: NO RAISE; sections=", len(t.sections()))
    except e:
        print("T15 array-of-tables: RAISED:", String(e))
    _show(String("T16 multiline-basic"), String('x = """hi"""\n'), String(""), String("x"))
    _show(String("T17 datetime"), String("x = 1979-05-27T07:32:00Z\n"), String(""), String("x"))
    _show(String("T17b date"), String("x = 1979-05-27\n"), String(""), String("x"))
    _show(String("T17c time"), String("x = 07:32:00\n"), String(""), String("x"))

    # ── extra TOML conformance hunts ──
    # T18: bare 'true'/'false' as part of larger token -> truex
    _show(String("T18 truex"), String("x = truex\n"), String(""), String("x"))
    # T19: hex/octal/binary ints (TOML 1.0 supports 0x 0o 0b) — does lib?
    _show(String("T19 hex-int"), String("x = 0xFF\n"), String(""), String("x"))
    # T20: int with double underscore 1__0 (tomllib rejects)
    _show(String("T20 double-underscore"), String("x = 1__0\n"), String(""), String("x"))
    # T21: underscore at start/end _1 / 1_ (tomllib rejects)
    _show(String("T21 trailing-underscore"), String("x = 1_\n"), String(""), String("x"))
    # T22: leading-zero float 0.5 ok, but 01.5 reject
    _show(String("T22 leading-zero-float"), String("x = 01.5\n"), String(""), String("x"))
    # T23: bare token that is garbage 'abc'
    _show(String("T23 bare-garbage"), String("x = abc\n"), String(""), String("x"))
    # T24: duplicate key in TOML (tomllib RAISES)
    _show(String("T24 dup-key"), String("x = 1\nx = 2\n"), String(""), String("x"))
    # T25: '.' inside a float exponent? 1.5e2
    _show(String("T25 float-exp"), String("x = 1.5e2\n"), String(""), String("x"))

    print("PROBES DONE")
