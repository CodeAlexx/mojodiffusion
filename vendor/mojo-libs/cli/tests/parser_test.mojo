# Tests for cli.parser + cli.help — verifies every documented form with concrete
# checks via check(mut p, mut f, cond, name) (no hidden asserts). Prints per-test
# results, a "passed: N failed: M" line, and "ALL CLI TESTS PASSED" when M == 0.
#
# Cross-check vs Python argparse: the EXPECTED_* constants below are the verbatim
# stdout of cli/tests/argparse_oracle.py for three representative argv inputs
# (regenerate with `python3 cli/tests/argparse_oracle.py -- <argv>`; mirrored by
# run_oracle.sh). The Mojo test builds an equivalent ArgParser, parses the same
# argv, renders the same key=value lines, and asserts byte-equality with the
# argparse oracle for the overlapping semantics.

from cli.parser import (
    ArgParser, ParseResult, TYPE_STR, TYPE_INT, TYPE_FLOAT,
)
from cli.help import render_help, render_version


def check(mut passed: Int, mut failed: Int, cond: Bool, name: String):
    if cond:
        passed += 1
        print("PASS:", name)
    else:
        failed += 1
        print("FAIL:", name)


# General-purpose parser used by most feature tests.
def _build() raises -> ArgParser:
    var p = ArgParser(String("app"), String("test app"))
    p.set_version(String("2.4.6"))
    p.add_flag(String("verbose"), String("v"), String("verbose output"))
    p.add_flag(String("alpha"), String("a"), String("flag a"))
    p.add_flag(String("beta"), String("b"), String("flag b"))
    p.add_flag(String("gamma"), String("c"), String("flag c"))
    p.add_count_flag(String("debug"), String("d"), String("increase debug"))
    p.add_option(String("name"), String("n"), String("the name"))
    p.add_option(String("count"), String("k"), String("a count"), String("7"), True, False, String(""), TYPE_INT)
    p.add_option(String("ratio"), String("R"), String("a ratio"), String(""), False, False, String(""), TYPE_FLOAT)
    p.add_option(String("region"), String("r"), String("region"), String(""), False, False, String("CLI_REGION"))
    p.add_option(String("token"), String("t"), String("required token"), String(""), False, True, String(""))
    var levels = List[String]()
    levels.append(String("low"))
    levels.append(String("mid"))
    levels.append(String("high"))
    p.add_option(String("level"), String("l"), String("log level"), String(""), False, False, String(""), TYPE_STR, levels^)
    p.add_option(String("inc"), String("i"), String("includes"), String(""), False, False, String(""), TYPE_STR, List[String](), True, False)
    p.add_option(String("tags"), String("T"), String("comma tags"), String(""), False, False, String(""), TYPE_STR, List[String](), False, True)
    p.add_positional(String("src"), String("source path"))
    p.add_positional(String("dst"), String("dest path"))
    p.add_rest(String("extra"), String("extra args"))
    return p^


# Build a parser equivalent to argparse_oracle.py's spec.
def _build_oracle() raises -> ArgParser:
    var p = ArgParser(String("app"), String(""))
    p.add_flag(String("verbose"), String("v"), String(""))
    p.add_option(String("name"), String("n"), String(""))
    p.add_option(String("count"), String("k"), String(""), String("7"), True, False, String(""), TYPE_INT)
    var levels = List[String]()
    levels.append(String("low"))
    levels.append(String("mid"))
    levels.append(String("high"))
    p.add_option(String("level"), String("l"), String(""), String(""), False, False, String(""), TYPE_STR, levels^)
    p.add_option(String("inc"), String(""), String(""), String(""), False, False, String(""), TYPE_STR, List[String](), True, False)
    p.add_positional(String("src"), String(""))
    p.add_positional(String("dst"), String(""))
    p.add_rest(String("rest"), String(""))
    return p^


# Render a ParseResult into the same key=value line format as the oracle.
def _render_oracle(r: ParseResult) raises -> String:
    var lines = List[String]()
    if r.get_bool(String("verbose")):
        lines.append(String("verbose=true"))
    else:
        lines.append(String("verbose=false"))
    if r.has(String("name")):
        lines.append(String("name=") + r.get_str(String("name")))
    lines.append(String("count=") + r.get_str(String("count")))
    if r.has(String("level")):
        lines.append(String("level=") + r.get_str(String("level")))
    if r.has(String("inc")):
        var inc = r.get_list(String("inc"))
        var joined = String("")
        for j in range(len(inc)):
            if j > 0:
                joined += "|"
            joined += inc[j]
        lines.append(String("inc=") + joined)
    lines.append(String("src=") + r.get_str(String("src")))
    lines.append(String("dst=") + r.get_str(String("dst")))
    var rest = r.rest()
    if len(rest) > 0:
        var jr = String("")
        for j in range(len(rest)):
            if j > 0:
                jr += "|"
            jr += rest[j]
        lines.append(String("rest=") + jr)
    var out = String("")
    for j in range(len(lines)):
        if j > 0:
            out += "\n"
        out += lines[j]
    return out^


def main() raises:
    var passed = 0
    var failed = 0
    print("=== cli.parser tests ===")

    # 1. mixed long flag + long option value + short -n2 + positionals
    var p1 = _build()
    var a1: List[String] = [
        String("--verbose"), String("--name"), String("alice"),
        String("-n2"), String("--token"), String("tk"),
        String("pos1"), String("pos2"),
    ]
    var r1 = p1.parse(a1)
    check(passed, failed, r1.get_bool(String("verbose")) == True, "1a verbose flag true")
    check(passed, failed, r1.get_str(String("name")) == "2", "1b -nX short value overrides")
    var pos1 = r1.positionals()
    check(passed, failed, len(pos1) == 2 and pos1[0] == "pos1" and pos1[1] == "pos2",
          "1c positionals collected in order")
    check(passed, failed, r1.get_str(String("src")) == "pos1" and r1.get_str(String("dst")) == "pos2",
          "1d named positionals assigned")

    # get_int / get_float coercion
    var p1b = _build()
    var a1b: List[String] = [String("--name"), String("42"), String("--ratio"), String("3.5"),
                             String("--token"), String("x")]
    var r1b = p1b.parse(a1b)
    check(passed, failed, r1b.get_int(String("name")) == 42, "1e get_int coercion")
    check(passed, failed, r1b.get_float(String("ratio")) == 3.5, "1f get_float coercion")

    # 2. value forms: --o=v, -ov, -o=v, -o v
    var p2 = _build()
    var a2: List[String] = [String("--name=bob"), String("--token"), String("x")]
    var r2 = p2.parse(a2)
    check(passed, failed, r2.get_str(String("name")) == "bob", "2a --name=value form")

    var p2b = _build()
    var a2b: List[String] = [String("-nbob"), String("--token"), String("x")]
    var r2b = p2b.parse(a2b)
    check(passed, failed, r2b.get_str(String("name")) == "bob", "2b -nValue form")

    var p2c = _build()
    var a2c: List[String] = [String("-n=carol"), String("--token"), String("x")]
    var r2c = p2c.parse(a2c)
    check(passed, failed, r2c.get_str(String("name")) == "carol", "2c -n=value form")

    var p2d = _build()
    var a2d: List[String] = [String("-n"), String("dave"), String("--token"), String("x")]
    var r2d = p2d.parse(a2d)
    check(passed, failed, r2d.get_str(String("name")) == "dave", "2d -n value (separate)")

    # 3. combined short flags -abc
    var p3 = _build()
    var a3: List[String] = [String("-abc"), String("--token"), String("x")]
    var r3 = p3.parse(a3)
    check(passed, failed,
          r3.get_bool(String("alpha")) and r3.get_bool(String("beta")) and r3.get_bool(String("gamma")),
          "3a -abc sets three flags")
    check(passed, failed, r3.get_bool(String("verbose")) == False, "3b unrelated flag stays false")

    # 3c count flags: -ddd -> 3
    var p3c = _build()
    var a3c: List[String] = [String("-ddd"), String("--token"), String("x")]
    var r3c = p3c.parse(a3c)
    check(passed, failed, r3c.get_count(String("debug")) == 3, "3c -ddd count is 3")

    # 3d count via repeated long flags
    var p3d = _build()
    var a3d: List[String] = [String("--debug"), String("--debug"), String("--token"), String("x")]
    var r3d = p3d.parse(a3d)
    check(passed, failed, r3d.get_count(String("debug")) == 2, "3d --debug x2 count is 2")

    # 3e count flag combined in a cluster with bool flags: -avd
    var p3e = _build()
    var a3e: List[String] = [String("-avd"), String("--token"), String("x")]
    var r3e = p3e.parse(a3e)
    check(passed, failed,
          r3e.get_bool(String("alpha")) and r3e.get_bool(String("verbose")) and r3e.get_count(String("debug")) == 1,
          "3e mixed cluster -avd")

    # 4. -- stops option parsing
    var p4 = _build()
    var a4: List[String] = [String("--token"), String("x"), String("--"), String("--not-a-flag")]
    var r4 = p4.parse(a4)
    var pos4 = r4.positionals()
    check(passed, failed, len(pos4) == 1 and pos4[0] == "--not-a-flag",
          "4a -- makes --not-a-flag a positional")

    # 5. defaults
    var p5 = _build()
    var a5: List[String] = [String("--token"), String("x")]
    var r5 = p5.parse(a5)
    check(passed, failed, r5.get_str(String("count")) == "7", "5a default value returned")
    check(passed, failed, r5.has(String("count")) == False, "5b default is not 'present'")
    check(passed, failed, r5.get_str(String("name")) == "", "5c no default -> empty string")

    # 5d required-but-missing raises
    var p5d = _build()
    var a5d: List[String] = [String("--name"), String("z")]
    var raised_req = False
    try:
        var _r = p5d.parse(a5d)
    except e:
        raised_req = True
    check(passed, failed, raised_req, "5d missing required option raises")

    # 5e..5h M1: an option must NOT consume a flag-looking next token as its
    # value (matches argparse "expected one argument"). A negative number IS a
    # valid value.
    var p5e = _build()
    var a5e: List[String] = [String("--name"), String("--verbose"), String("--token"), String("x")]
    var raised_5e = False
    try:
        var _r = p5e.parse(a5e)
    except e:
        raised_5e = True
    check(passed, failed, raised_5e, "5e --name --verbose raises (no flag-as-value)")

    var p5f = _build()
    var a5f: List[String] = [String("--name"), String("-v"), String("--token"), String("x")]
    var raised_5f = False
    try:
        var _r = p5f.parse(a5f)
    except e:
        raised_5f = True
    check(passed, failed, raised_5f, "5f --name -v raises (no short-flag-as-value)")

    # required --name not satisfied by swallowing --verbose
    var p5g = ArgParser(String("app"), String(""))
    p5g.add_flag(String("verbose"), String("v"), String(""))
    p5g.add_option(String("name"), String("n"), String(""), String(""), False, True, String(""))
    var a5g: List[String] = [String("--name"), String("--verbose")]
    var raised_5g = False
    try:
        var _r = p5g.parse(a5g)
    except e:
        raised_5g = True
    check(passed, failed, raised_5g, "5g required --name + --name --verbose raises")

    # negative number IS accepted as an option value
    var p5h = _build()
    var a5h: List[String] = [String("--ratio"), String("-3.5"), String("--token"), String("x")]
    var r5h = p5h.parse(a5h)
    check(passed, failed, r5h.get_float(String("ratio")) == -3.5, "5h negative number is a valid value")

    # 6. error cases
    var p6 = _build()
    var a6: List[String] = [String("--bogus"), String("--token"), String("x")]
    var raised_unknown = False
    try:
        var _r = p6.parse(a6)
    except e:
        raised_unknown = True
    check(passed, failed, raised_unknown, "6a unknown long option raises")

    var p6b = _build()
    var a6b: List[String] = [String("-z"), String("--token"), String("x")]
    var raised_unknown_short = False
    try:
        var _r = p6b.parse(a6b)
    except e:
        raised_unknown_short = True
    check(passed, failed, raised_unknown_short, "6b unknown short flag raises")

    var p6c = _build()
    var a6c: List[String] = [String("--token"), String("x"), String("--name")]
    var raised_missing_val = False
    try:
        var _r = p6c.parse(a6c)
    except e:
        raised_missing_val = True
    check(passed, failed, raised_missing_val, "6c missing option value raises")

    # 6d bad int coercion (typed option) raises at parse time
    var p6d = _build()
    var a6d: List[String] = [String("--count"), String("abc"), String("--token"), String("x")]
    var raised_bad_int = False
    try:
        var _r = p6d.parse(a6d)
    except e:
        raised_bad_int = True
    check(passed, failed, raised_bad_int, "6d bad int coercion raises (typed)")

    # 6e bad float coercion raises at parse time
    var p6e = _build()
    var a6e: List[String] = [String("--ratio"), String("xyz"), String("--token"), String("x")]
    var raised_bad_float = False
    try:
        var _r = p6e.parse(a6e)
    except e:
        raised_bad_float = True
    check(passed, failed, raised_bad_float, "6e bad float coercion raises (typed)")

    # 6f get_int on untyped str value also raises on bad input
    var p6f = _build()
    var a6f: List[String] = [String("--name"), String("abc"), String("--token"), String("x")]
    var r6f = p6f.parse(a6f)
    var raised_get_int = False
    try:
        var _i = r6f.get_int(String("name"))
    except e:
        raised_get_int = True
    check(passed, failed, raised_get_int, "6f get_int on bad str raises")

    # 7. env fallback
    var p7 = _build()
    var env7 = Dict[String, String]()
    env7[String("CLI_REGION")] = String("eu-west")
    var a7: List[String] = [String("--token"), String("x")]
    var r7 = p7.parse(a7, env7)
    check(passed, failed, r7.get_str(String("region")) == "eu-west", "7a env fallback used")
    check(passed, failed, r7.has(String("region")), "7b env value marks present")

    var p7b = _build()
    var a7b: List[String] = [String("--region"), String("us-east"), String("--token"), String("x")]
    var r7b = p7b.parse(a7b, env7)
    check(passed, failed, r7b.get_str(String("region")) == "us-east", "7c command-line overrides env")

    # 8. subcommand
    var sub = ArgParser(String("build"), String("build the thing"))
    sub.add_option(String("opt"), String("o"), String("an opt"))
    sub.add_flag(String("clean"), String("C"), String("clean first"))
    sub.add_positional(String("target"), String("what to build"))
    var p8 = ArgParser(String("app"), String("root"))
    p8.add_flag(String("verbose"), String("v"), String("verbose"))
    p8.add_subcommand(String("build"), sub^)
    var a8: List[String] = [String("build"), String("--opt"), String("x"), String("--clean"), String("mytarget")]
    var r8 = p8.parse(a8)
    check(passed, failed, r8.subcommand() == "build", "8a subcommand name")
    var sr8 = r8.sub()
    check(passed, failed, sr8.get_str(String("opt")) == "x", "8b sub option parsed")
    check(passed, failed, sr8.get_bool(String("clean")), "8c sub flag parsed")
    check(passed, failed, sr8.get_str(String("target")) == "mytarget", "8d sub positional parsed")
    check(passed, failed, r8.subcommand() == "build", "8e parent still reports subcommand")

    # 9. choices
    var p9 = _build()
    var a9: List[String] = [String("--level"), String("mid"), String("--token"), String("x")]
    var r9 = p9.parse(a9)
    check(passed, failed, r9.get_str(String("level")) == "mid", "9a valid choice accepted")

    var p9b = _build()
    var a9b: List[String] = [String("--level"), String("nope"), String("--token"), String("x")]
    var raised_choice = False
    try:
        var _r = p9b.parse(a9b)
    except e:
        raised_choice = True
    check(passed, failed, raised_choice, "9b invalid choice raises")

    # 10. multi-value / repeatable + comma-split
    var p10 = _build()
    var a10: List[String] = [String("--inc"), String("a"), String("--inc"), String("b"),
                             String("-i"), String("c"), String("--token"), String("x")]
    var r10 = p10.parse(a10)
    var inc10 = r10.get_list(String("inc"))
    check(passed, failed, len(inc10) == 3 and inc10[0] == "a" and inc10[1] == "b" and inc10[2] == "c",
          "10a repeatable option collects list")

    var p10b = _build()
    var a10b: List[String] = [String("--tags"), String("x,y,z"), String("--token"), String("x")]
    var r10b = p10b.parse(a10b)
    var tags = r10b.get_list(String("tags"))
    check(passed, failed, len(tags) == 3 and tags[0] == "x" and tags[1] == "y" and tags[2] == "z",
          "10b comma-split into list")

    # 11. variadic rest
    var p11 = _build()
    var a11: List[String] = [String("--token"), String("x"), String("s"), String("d"),
                             String("r1"), String("r2"), String("r3")]
    var r11 = p11.parse(a11)
    var rest11 = r11.rest()
    check(passed, failed, r11.get_str(String("src")) == "s" and r11.get_str(String("dst")) == "d",
          "11a named positionals before rest")
    check(passed, failed, len(rest11) == 3 and rest11[0] == "r1" and rest11[2] == "r3",
          "11b rest collects leftovers")

    # 12. mutually-exclusive group
    var p12 = ArgParser(String("app"), String(""))
    p12.add_flag(String("json"), String("j"), String("json output"))
    p12.add_flag(String("yaml"), String("y"), String("yaml output"))
    var ex = List[String]()
    ex.append(String("json"))
    ex.append(String("yaml"))
    p12.set_exclusive(ex^)
    # only one given -> ok
    var a12ok: List[String] = [String("--json")]
    var r12ok = p12.parse(a12ok)
    check(passed, failed, r12ok.get_bool(String("json")), "12a one of exclusive group ok")
    # both given -> raises
    var p12b = ArgParser(String("app"), String(""))
    p12b.add_flag(String("json"), String("j"), String("json output"))
    p12b.add_flag(String("yaml"), String("y"), String("yaml output"))
    var ex2 = List[String]()
    ex2.append(String("json"))
    ex2.append(String("yaml"))
    p12b.set_exclusive(ex2^)
    var a12bad: List[String] = [String("--json"), String("--yaml")]
    var raised_excl = False
    try:
        var _r = p12b.parse(a12bad)
    except e:
        raised_excl = True
    check(passed, failed, raised_excl, "12b both exclusive flags raises")

    # 13. help / version intents
    var p13 = _build()
    var a13: List[String] = [String("--help")]
    var r13 = p13.parse(a13)
    check(passed, failed, r13.wants_help(), "13a --help sets wants_help")
    # -h short help (no -h flag registered)
    var p13b = _build()
    var a13b: List[String] = [String("-h")]
    var r13b = p13b.parse(a13b)
    check(passed, failed, r13b.wants_help(), "13b -h sets wants_help")
    # help skips required validation
    check(passed, failed, not r13.wants_version(), "13c help-only does not set version")
    var p13c = _build()
    var a13c: List[String] = [String("--version")]
    var r13c = p13c.parse(a13c)
    check(passed, failed, r13c.wants_version(), "13d --version sets wants_version")

    # 14. help() text + version
    var ph = _build()
    var h = render_help(ph)
    check(passed, failed, h.find("--verbose") >= 0 and h.find("verbose output") >= 0,
          "14a help has flag name + text")
    check(passed, failed, h.find("--name") >= 0 and h.find("the name") >= 0,
          "14b help has option name + text")
    check(passed, failed, h.find("src") >= 0 and h.find("source path") >= 0,
          "14c help has positional name + text")
    check(passed, failed, h.find("[required]") >= 0, "14d help marks required")
    check(passed, failed, h.find("[default: 7]") >= 0, "14e help shows default")
    check(passed, failed, h.find("CLI_REGION") >= 0, "14f help shows env var")
    check(passed, failed, h.find("[choices: low, mid, high]") >= 0, "14g help shows choices")
    check(passed, failed, h.find("(repeatable)") >= 0 or h.find("[repeatable]") >= 0,
          "14h help marks repeatable")
    check(passed, failed, h.find("extra...") >= 0, "14i help shows variadic rest")
    var ver = render_version(ph)
    check(passed, failed, ver.find("2.4.6") >= 0 and ver.find("app") >= 0, "14j version text")

    var h8 = render_help(p8)
    check(passed, failed, h8.find("build") >= 0, "14k help lists subcommand")

    # ── 15. cross-check vs Python argparse (3 representative inputs) ──
    print("")
    print("--- argparse cross-check ---")

    # Input 1: -v --name alice --count 3 src1 dst1
    var expect1 = String("verbose=true\nname=alice\ncount=3\nsrc=src1\ndst=dst1")
    var oc1 = _build_oracle()
    var oa1: List[String] = [String("-v"), String("--name"), String("alice"),
                             String("--count"), String("3"), String("src1"), String("dst1")]
    var orr1 = oc1.parse(oa1)
    var got1 = _render_oracle(orr1)
    print("input1 argparse:", expect1.replace(String("\n"), String(" | ")))
    print("input1 mojo    :", got1.replace(String("\n"), String(" | ")))
    check(passed, failed, got1 == expect1, "15a matches argparse (flags+name+count+pos)")

    # Input 2: --level mid --inc a --inc b s d extra1 extra2
    var expect2 = String("verbose=false\ncount=7\nlevel=mid\ninc=a|b\nsrc=s\ndst=d\nrest=extra1|extra2")
    var oc2 = _build_oracle()
    var oa2: List[String] = [String("--level"), String("mid"), String("--inc"), String("a"),
                             String("--inc"), String("b"), String("s"), String("d"),
                             String("extra1"), String("extra2")]
    var orr2 = oc2.parse(oa2)
    var got2 = _render_oracle(orr2)
    print("input2 argparse:", expect2.replace(String("\n"), String(" | ")))
    print("input2 mojo    :", got2.replace(String("\n"), String(" | ")))
    check(passed, failed, got2 == expect2, "15b matches argparse (choice+append+rest)")

    # Input 3: -n bob s2 d2
    var expect3 = String("verbose=false\nname=bob\ncount=7\nsrc=s2\ndst=d2")
    var oc3 = _build_oracle()
    var oa3: List[String] = [String("-n"), String("bob"), String("s2"), String("d2")]
    var orr3 = oc3.parse(oa3)
    var got3 = _render_oracle(orr3)
    print("input3 argparse:", expect3.replace(String("\n"), String(" | ")))
    print("input3 mojo    :", got3.replace(String("\n"), String(" | ")))
    check(passed, failed, got3 == expect3, "15c matches argparse (short opt + defaults)")

    print("")
    print("passed:", passed, "failed:", failed)
    if failed == 0:
        print("ALL CLI TESTS PASSED")
