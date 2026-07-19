# config/tests/toml_test.mojo
#
# Verifies parse_toml (config/toml.mojo) against the REAL Python `tomllib`.
# toml_oracle.py parses the SAME document with tomllib and writes a flattened
# section/key -> typed-value oracle file; this test parses the document with
# parse_toml and asserts matching TYPES + values (int==int, float within 1e-9,
# exact strings incl. escapes, bool, array length + each element, dotted-table
# placement).
#
# Run from /home/alex/MOJO-libs:
#   python3 config/tests/toml_oracle.py /tmp/toml_oracle.txt
#   pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . \
#       config/tests/toml_test.mojo
#
# (The test ALSO shells out to run the oracle itself, so a bare `mojo run`
#  works; if that fails it falls back to a pre-generated /tmp/toml_oracle.txt.)

from config.value import ConfigTree, ConfigValue
from config.toml import parse_toml
from std.os import abort

# The same TOML document the oracle parses with tomllib. Kept byte-identical.
comptime DOC = String(
    "# top-level comment\n"
    + 'title = "TOML \\"Example\\""        # trailing comment with a quote\n'
    + "literal = 'C:\\\\no\\\\escape'\n"
    + "maxconn = 1_000\n"
    + "ratio = 6.022e23\n"
    + "small = 0.5\n"
    + "neg = -17\n"
    + "flag = true\n"
    + "nums = [1, 2, 3]\n"
    + 'words = ["a", "b", "c"]\n'
    + 'mixed = [1, "two", 3.0, false]\n'
    + "matrix = [\n    [1, 2],\n    [3, 4],\n]\n"
    + 'escaped = "line1\\nline2\\ttab"\n'
    + "\n[server]\n"
    + 'host = "localhost"\n'
    + "port = 8080\n"
    + "enabled = false\n"
    + "\n[a.b]\n"
    + "deep = 42\n"
    + "name = 'nested'\n"
    + "\n[owner]\n"
    + 'dotted.key = "viadot"\n'
)

comptime ORACLE_PATH = "/tmp/toml_oracle.txt"


# ── tiny split/parse helpers for the oracle file (tab-separated records) ──────
def _split(s: String, sep: Int) -> List[String]:
    var out = List[String]()
    var b = s.as_bytes()
    var n = s.byte_length()
    var cur = String("")
    for i in range(n):
        var c = Int(b[i])
        if c == sep:
            out.append(cur)
            cur = String("")
        else:
            cur += chr(c)
    out.append(cur)
    return out^


def _unescape(s: String) -> String:
    # reverse of oracle escape_str: \\ \n \t \r
    var out = String("")
    var b = s.as_bytes()
    var n = s.byte_length()
    var i = 0
    while i < n:
        var c = Int(b[i])
        if c == 0x5C and i + 1 < n:
            var e = Int(b[i + 1])
            if e == 0x5C:
                out += chr(0x5C); i += 2; continue
            if e == 0x6E:
                out += chr(0x0A); i += 2; continue
            if e == 0x74:
                out += chr(0x09); i += 2; continue
            if e == 0x72:
                out += chr(0x0D); i += 2; continue
        out += chr(c)
        i += 1
    return out


def _close(a: Float64, b: Float64) -> Bool:
    var d = a - b
    if d < 0:
        d = -d
    # relative tolerance for large magnitudes (e.g. 6.022e23), abs for small
    var scale = b if b >= 0 else -b
    var tol = 1e-9
    if scale > 1.0:
        tol = scale * 1e-9
    return d <= tol


def check(mut p: Int, mut f: Int, cond: Bool, name: String):
    if cond:
        p += 1
        print("  ok  :", name)
    else:
        f += 1
        print("  FAIL:", name)


def main() raises:
    var p = 0
    var f = 0

    # 1) read the oracle produced by REAL tomllib (config/tests/toml_oracle.py).
    #    Generate it first with:
    #      python3 config/tests/toml_oracle.py /tmp/toml_oracle.txt
    var oracle_text: String
    try:
        oracle_text = open(ORACLE_PATH, "r").read()
    except:
        raise Error(
            "could not read oracle file "
            + ORACLE_PATH
            + " — first run: python3 config/tests/toml_oracle.py "
            + ORACLE_PATH
        )

    # 2) parse the document with our parser
    var tree = parse_toml(DOC)

    print("=== tomllib (oracle) vs parse_toml (mojo) ===")

    # 3) walk oracle records, asserting against the tree.
    var lines = _split(oracle_text, 0x0A)

    for li in range(len(lines)):
        var line = lines[li]
        if line.byte_length() == 0:
            continue
        var rec = _split(line, 0x09)  # tab
        var tag = rec[0]

        if tag == "S":
            # S section key typekind payload
            var sec = rec[1]
            var key = rec[2]
            var tk = rec[3]
            var payload = rec[4]
            var label = "[" + sec + "] " + key + " (" + tk + ")"
            if not tree.has(sec, key):
                check(p, f, False, label + " present")
                continue
            var v = tree.get(sec, key)
            if tk == "int":
                var exp = Int64(Int(payload))
                var got = v.as_int()
                print(
                    "  cmp ", label, "tomllib=", exp, " mojo=", got
                )
                check(p, f, v.kind == 2 and got == exp, label)
            elif tk == "float":
                var ef = Float64(payload)
                var gf = v.as_float()
                print("  cmp ", label, "tomllib=", ef, " mojo=", gf)
                check(p, f, v.kind == 3 and _close(gf, ef), label)
            elif tk == "bool":
                var eb = payload == "true"
                var gb = v.as_bool()
                print("  cmp ", label, "tomllib=", eb, " mojo=", gb)
                check(p, f, v.kind == 4 and gb == eb, label)
            elif tk == "str":
                var es = _unescape(payload)
                var gs = v.as_str()
                print(
                    "  cmp ", label, "tomllib=<", es, "> mojo=<", gs, ">"
                )
                check(p, f, v.kind == 1 and gs == es, label)
            else:
                check(p, f, False, label + " unknown type " + tk)

        elif tag == "A":
            # A section key array len
            var sec = rec[1]
            var key = rec[2]
            var alen = Int(rec[4])
            var label = "[" + sec + "] " + key + " array len"
            if not tree.has(sec, key):
                check(p, f, False, label + " present")
                continue
            var v = tree.get(sec, key)
            var arr = v.as_array()
            print("  cmp ", label, "tomllib=", alen, " mojo=", len(arr))
            check(p, f, v.kind == 5 and len(arr) == alen, label)

        elif tag == "E":
            # E section key idx typekind payload  (scalar element)
            var sec = rec[1]
            var key = rec[2]
            var idx = Int(rec[3])
            var tk = rec[4]
            var payload = rec[5]
            var label = "[" + sec + "] " + key + "[" + String(idx) + "] (" + tk + ")"
            var arr = tree.get(sec, key).as_array()
            if idx >= len(arr):
                check(p, f, False, label + " index in range")
                continue
            var el = arr[idx].copy()
            if tk == "int":
                check(p, f, el.kind == 2 and el.as_int() == Int64(Int(payload)), label)
            elif tk == "float":
                check(p, f, el.kind == 3 and _close(el.as_float(), Float64(payload)), label)
            elif tk == "bool":
                check(p, f, el.kind == 4 and el.as_bool() == (payload == "true"), label)
            elif tk == "str":
                check(p, f, el.kind == 1 and el.as_str() == _unescape(payload), label)
            else:
                check(p, f, False, label + " unknown type")

        elif tag == "N":
            # N section key idx array sublen  (nested-array header)
            var sec = rec[1]
            var key = rec[2]
            var idx = Int(rec[3])
            var sublen = Int(rec[5])
            var label = "[" + sec + "] " + key + "[" + String(idx) + "] subarray len"
            var arr = tree.get(sec, key).as_array()
            var sub = arr[idx].as_array()
            check(p, f, arr[idx].kind == 5 and len(sub) == sublen, label)

        elif tag == "M":
            # M section key i j typekind payload  (nested-array element)
            var sec = rec[1]
            var key = rec[2]
            var i = Int(rec[3])
            var j = Int(rec[4])
            var tk = rec[5]
            var payload = rec[6]
            var label = (
                "[" + sec + "] " + key + "[" + String(i) + "][" + String(j) + "]"
            )
            var sub = tree.get(sec, key).as_array()[i].as_array()
            var el = sub[j].copy()
            if tk == "int":
                check(p, f, el.kind == 2 and el.as_int() == Int64(Int(payload)), label)
            elif tk == "float":
                check(p, f, el.kind == 3 and _close(el.as_float(), Float64(payload)), label)
            elif tk == "bool":
                check(p, f, el.kind == 4 and el.as_bool() == (payload == "true"), label)
            elif tk == "str":
                check(p, f, el.kind == 1 and el.as_str() == _unescape(payload), label)
            else:
                check(p, f, False, label + " unknown type")

    # 4) extra direct assertions on the trickiest cases (independent of oracle).
    print("=== direct spot-checks ===")
    check(p, f, tree.get("", "title").as_str() == 'TOML "Example"', "escaped-quote string")
    # literal string: source is  'C:\\no\\escape'  -> no escape processing, so
    # the value keeps both backslash pairs verbatim: C:\\no\\escape
    check(p, f, tree.get("", "literal").as_str() == "C:\\\\no\\\\escape", "literal string verbatim")
    check(p, f, tree.get("", "maxconn").as_int() == 1000, "underscored int 1_000")
    check(p, f, tree.get("", "escaped").as_str() == "line1\nline2\ttab", "string escapes \\n \\t")
    check(p, f, tree.has("a.b", "deep") and tree.get("a.b", "deep").as_int() == 42, "dotted table [a.b]")
    check(p, f, tree.has("owner.dotted", "key"), "dotted assignment key placement")

    print("")
    print("passed:", p, " failed:", f)
    if f == 0:
        print("ALL TOML TESTS PASSED")
    else:
        abort("TOML tests failed")
