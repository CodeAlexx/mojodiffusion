# config/tests/ini_test.mojo — verifies parse_ini against a REAL Python
# configparser oracle (no hand-asserted expected values).
#
# Flow:
#   1. Run config/tests/ini_oracle.py (writes ini_oracle_out.txt from the SAME
#      INI text that is embedded below, parsed by stdlib configparser).
#   2. parse_ini() the identical INI text in Mojo.
#   3. Compare: same set of sections, same keys per section, same string values.
#
# The INI text here MUST stay byte-identical to ini_oracle.py's INI_TEXT.

from config.value import ConfigTree, ConfigValue
from config.ini import parse_ini, parse_ini_file
from std.os import abort

# Keep in sync with ini_oracle.py INI_TEXT.
comptime INI_TEXT = """[DEFAULT]
shared = base
timeout : 30

[server]
Host = localhost
Port : 8080
empty =
KeyWithSpace = v1
padded   =   trimmed value
colonkey:novalue_sep

[db]
url = postgres://x
shared = override
; this is a full-line comment
# so is this
nums = 1,2,3
"""


def check(mut p: Int, mut f: Int, cond: Bool, name: String):
    if cond:
        p += 1
    else:
        f += 1
        print("  FAIL:", name)


# ---- minimal escaping decoder matching ini_oracle.py esc() -------------------
def _unesc(s: String) -> String:
    var b = s.as_bytes()
    var n = s.byte_length()
    var out = String("")
    var i = 0
    while i < n:
        var c = Int(b[i])
        if c == 0x5C and i + 1 < n:  # backslash
            var nxt = Int(b[i + 1])
            if nxt == 0x6E:  # 'n'
                out += chr(0x0A)
                i += 2
                continue
            if nxt == 0x74:  # 't'
                out += chr(0x09)
                i += 2
                continue
            if nxt == 0x5C:  # backslash
                out += chr(0x5C)
                i += 2
                continue
        out += chr(c)
        i += 1
    return out^


def _split_tabs(s: String) -> List[String]:
    var out = List[String]()
    var b = s.as_bytes()
    var n = s.byte_length()
    var cur = String("")
    for i in range(n):
        var c = Int(b[i])
        if c == 0x09:  # tab
            out.append(cur)
            cur = String("")
        else:
            cur += chr(c)
    out.append(cur)
    return out^


def main() raises:
    var p = 0
    var f = 0

    # ---- Mojo parse ----
    var tree = parse_ini(INI_TEXT)

    # ---- read the oracle dump ----
    var oracle_path = String("config/tests/ini_oracle_out.txt")
    var oracle_text: String
    try:
        with open(oracle_path, "r") as fh:
            oracle_text = fh.read()
    except:
        print("ERROR: could not read", oracle_path)
        print("Run first:  python3 config/tests/ini_oracle.py")
        abort()

    # Build the oracle's section/key/value model.
    var oracle_sections = List[String]()
    # Parallel lists of (section, key, value) for KV records.
    var ksec = List[String]()
    var kkey = List[String]()
    var kval = List[String]()

    var orac_lines = List[String]()
    var ob = oracle_text.as_bytes()
    var on = oracle_text.byte_length()
    var cur = String("")
    for i in range(on):
        var c = Int(ob[i])
        if c == 0x0A:
            orac_lines.append(cur)
            cur = String("")
        elif c == 0x0D:
            continue
        else:
            cur += chr(c)
    if cur.byte_length() > 0:
        orac_lines.append(cur)

    for li in range(len(orac_lines)):
        var parts = _split_tabs(orac_lines[li])
        if len(parts) == 0:
            continue
        if parts[0] == "SECTION" and len(parts) >= 2:
            oracle_sections.append(_unesc(parts[1]))
        elif parts[0] == "KV" and len(parts) >= 4:
            ksec.append(_unesc(parts[1]))
            kkey.append(_unesc(parts[2]))
            kval.append(_unesc(parts[3]))

    # ---- Compare: same set of sections ----
    var mojo_sections = tree.sections()
    check(p, f, len(mojo_sections) == len(oracle_sections),
          "section count (mojo=" + String(len(mojo_sections)) +
          " oracle=" + String(len(oracle_sections)) + ")")
    for i in range(len(oracle_sections)):
        var found = False
        for j in range(len(mojo_sections)):
            if mojo_sections[j] == oracle_sections[i]:
                found = True
                break
        check(p, f, found, "section present: [" + oracle_sections[i] + "]")

    # ---- Compare: every oracle KV exists in the Mojo tree with same value ----
    print("--- configparser-vs-mojo comparison ---")
    for i in range(len(ksec)):
        var sec = ksec[i]
        var key = kkey[i]
        var oval = kval[i]
        var has = tree.has(sec, key)
        var mval = String("")
        if has:
            mval = tree.get(sec, key).as_str()
        var ok = has and (mval == oval)
        var status = "OK " if ok else "DIFF"
        print(status, "[" + sec + "]", key,
              "  oracle=" + _q(oval) + "  mojo=" + (_q(mval) if has else "<MISSING>"))
        check(p, f, ok, "[" + sec + "] " + key + " value match")

    # ---- Compare: per-section key counts match (no extra Mojo keys) ----
    for si in range(len(oracle_sections)):
        var sec = oracle_sections[si]
        # count oracle keys for this section
        var ocount = 0
        for i in range(len(ksec)):
            if ksec[i] == sec:
                ocount += 1
        var mkeys = tree.keys(sec)
        check(p, f, len(mkeys) == ocount,
              "[" + sec + "] key count (mojo=" + String(len(mkeys)) +
              " oracle=" + String(ocount) + ")")

    # ---- Mojo-specific divergence: pre-header keys -> global section "" ----
    # (configparser would RAISE MissingSectionHeaderError here.)
    var g = parse_ini("preheader = top\n[s]\na = 1\n")
    check(p, f, g.has("", "preheader") and g.get("", "preheader").as_str() == "top",
          "divergence: pre-header key -> global section \"\"")
    check(p, f, g.has("s", "a"), "after-pre-header section parsed")

    # ---- last-wins duplicate keys ----
    var d = parse_ini("[x]\nk = 1\nk = 2\nk = 3\n")
    check(p, f, d.get("x", "k").as_str() == "3", "duplicate key last-wins")

    # ---- empty value + ':' separator + verbatim inline comment ----
    var e = parse_ini("[x]\nempty=\nc : v ; inline stays\n")
    check(p, f, e.get("x", "empty").as_str() == "", "empty value -> \"\"")
    check(p, f, e.get("x", "c").as_str() == "v ; inline stays",
          "inline comment NOT stripped (matches configparser default)")

    # ---- parse_ini_file round-trip ----
    var tmp = String("config/tests/_ini_tmp.ini")
    with open(tmp, "w") as wf:
        wf.write("[f]\nalpha = beta\n")
    var ft = parse_ini_file(tmp)
    check(p, f, ft.get("f", "alpha").as_str() == "beta", "parse_ini_file round-trip")

    print("passed:", p, "failed:", f)
    if f == 0:
        print("ALL INI TESTS PASSED")
    else:
        abort()


def _q(s: String) -> String:
    return "'" + s + "'"
