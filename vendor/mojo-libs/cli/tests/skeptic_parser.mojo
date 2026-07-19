# Adversarial probes for cli.parser. Each probe prints PROBE name + observed
# behavior. We do NOT assert pass/fail here — we report verbatim what happens so
# the skeptic can compare against argparse semantics.

from cli.parser import (
    ArgParser, ParseResult, TYPE_STR, TYPE_INT, TYPE_FLOAT,
)


def _mk() raises -> ArgParser:
    var p = ArgParser(String("app"), String("probe"))
    p.add_flag(String("verbose"), String("v"))
    p.add_count_flag(String("count"), String("c"))
    p.add_option(String("name"), String("n"))
    p.add_option(String("num"), String("N"), String(""), String(""), False, False, String(""), TYPE_INT)
    p.add_positional(String("src"))
    return p^


def _argv(items: List[String]) -> List[String]:
    return items.copy()


def main() raises:
    # ---- P1: --opt with MISSING value at end of args -> must raise ----
    var p1 = ArgParser(String("app"))
    p1.add_option(String("name"), String("n"))
    var a1 = List[String](); a1.append(String("--name"))
    try:
        var r = p1.parse(a1)
        print("P1 missing-long-value: NO RAISE; name='", r.get_str(String("name")), "' present=", r.has(String("name")))
    except e:
        print("P1 missing-long-value: RAISED:", String(e))

    # ---- P1b: -n with missing value at end ----
    var a1b = List[String](); a1b.append(String("-n"))
    try:
        var r = p1.parse(a1b)
        print("P1b missing-short-value: NO RAISE; name='", r.get_str(String("name")), "'")
    except e:
        print("P1b missing-short-value: RAISED:", String(e))

    # ---- P2: -o=v glued form ----
    var p2 = ArgParser(String("app"))
    p2.add_option(String("name"), String("n"))
    var a2 = List[String](); a2.append(String("-n=hello"))
    try:
        var r = p2.parse(a2)
        print("P2 -n=hello: name='", r.get_str(String("name")), "'")
    except e:
        print("P2 -n=hello: RAISED:", String(e))

    # ---- P2b: -nVALUE glued ----
    var a2b = List[String](); a2b.append(String("-nhello"))
    try:
        var r = p2.parse(a2b)
        print("P2b -nhello: name='", r.get_str(String("name")), "'")
    except e:
        print("P2b -nhello: RAISED:", String(e))

    # ---- P3: '--' as very first arg, then a positional ----
    var p3 = _mk()
    var a3 = List[String](); a3.append(String("--")); a3.append(String("foo"))
    try:
        var r = p3.parse(a3)
        print("P3 -- first: src='", r.get_str(String("src")), "' positionals=", len(r.positionals()))
    except e:
        print("P3 -- first: RAISED:", String(e))

    # ---- P3b: '--' as very last arg ----
    var a3b = List[String](); a3b.append(String("foo")); a3b.append(String("--"))
    try:
        var r = p3.parse(a3b)
        print("P3b -- last: src='", r.get_str(String("src")), "' positionals=", len(r.positionals()))
    except e:
        print("P3b -- last: RAISED:", String(e))

    # ---- P4: empty arg "" ----
    var p4 = _mk()
    var a4 = List[String](); a4.append(String(""))
    try:
        var r = p4.parse(a4)
        print("P4 empty-arg: src='", r.get_str(String("src")), "' positionals=", len(r.positionals()))
    except e:
        print("P4 empty-arg: RAISED:", String(e))

    # ---- P5: value that looks like a flag: --name --verbose ----
    var p5 = _mk()
    var a5 = List[String](); a5.append(String("--name")); a5.append(String("--verbose"))
    try:
        var r = p5.parse(a5)
        print("P5 --name --verbose: name='", r.get_str(String("name")), "' verbose=", r.get_bool(String("verbose")))
    except e:
        print("P5 --name --verbose: RAISED:", String(e))

    # ---- P5b: --name -v (short flag as value) ----
    var a5b = List[String](); a5b.append(String("--name")); a5b.append(String("-v"))
    try:
        var r = p5.parse(a5b)
        print("P5b --name -v: name='", r.get_str(String("name")), "' verbose=", r.get_bool(String("verbose")))
    except e:
        print("P5b --name -v: RAISED:", String(e))

    # ---- P6: env value that fails int coercion ----
    var p6 = ArgParser(String("app"))
    p6.add_option(String("num"), String("N"), String(""), String(""), False, True, String("MYNUM"), TYPE_INT)
    var env6 = Dict[String, String](); env6[String("MYNUM")] = String("notanint")
    var a6 = List[String]()
    try:
        var r = p6.parse(a6, env6)
        print("P6 env-int-bad: NO RAISE; num='", r.get_str(String("num")), "'")
    except e:
        print("P6 env-int-bad: RAISED:", String(e))

    # ---- P6b: required satisfied ONLY via env ----
    var env6b = Dict[String, String](); env6b[String("MYNUM")] = String("42")
    try:
        var r = p6.parse(a6, env6b)
        print("P6b env-satisfies-required: num='", r.get_str(String("num")), "' present=", r.has(String("num")))
    except e:
        print("P6b env-satisfies-required: RAISED:", String(e))

    # ---- P6c: required missing entirely ----
    var env6c = Dict[String, String]()
    try:
        var r = p6.parse(a6, env6c)
        print("P6c required-missing: NO RAISE num='", r.get_str(String("num")), "'")
    except e:
        print("P6c required-missing: RAISED:", String(e))

    # ---- P7: choices not in set raises; case sensitivity ----
    var p7 = ArgParser(String("app"))
    var ch = List[String](); ch.append(String("red")); ch.append(String("green"))
    p7.add_option(String("color"), String("C"), String(""), String(""), False, False, String(""), TYPE_STR, ch^)
    var a7 = List[String](); a7.append(String("--color")); a7.append(String("blue"))
    try:
        var r = p7.parse(a7)
        print("P7 bad-choice: NO RAISE color='", r.get_str(String("color")), "'")
    except e:
        print("P7 bad-choice: RAISED:", String(e))
    var a7b = List[String](); a7b.append(String("--color")); a7b.append(String("RED"))
    try:
        var r = p7.parse(a7b)
        print("P7b choice-case RED: NO RAISE color='", r.get_str(String("color")), "'")
    except e:
        print("P7b choice-case RED: RAISED:", String(e))

    # ---- P8: count via -ccc vs -c -c -c ----
    var p8 = ArgParser(String("app"))
    p8.add_count_flag(String("count"), String("c"))
    var a8 = List[String](); a8.append(String("-ccc"))
    try:
        var r = p8.parse(a8)
        print("P8 -ccc: count=", r.get_count(String("count")))
    except e:
        print("P8 -ccc: RAISED:", String(e))
    var a8b = List[String](); a8b.append(String("-c")); a8b.append(String("-c")); a8b.append(String("-c"))
    var r8b = p8.parse(a8b)
    print("P8b -c -c -c: count=", r8b.get_count(String("count")))

    # ---- P9: unknown subcommand (no subs defined) treated as positional? ----
    var p9 = _mk()
    var a9 = List[String](); a9.append(String("bogussub"))
    try:
        var r = p9.parse(a9)
        print("P9 unknown-token: src='", r.get_str(String("src")), "' sub='", r.subcommand(), "'")
    except e:
        print("P9 unknown-token: RAISED:", String(e))

    # ---- P10: negative-number positional -5 ----
    var p10 = _mk()
    var a10 = List[String](); a10.append(String("-5"))
    try:
        var r = p10.parse(a10)
        print("P10 -5 positional: src='", r.get_str(String("src")), "' positionals=", len(r.positionals()))
    except e:
        print("P10 -5 positional: RAISED:", String(e))

    # ---- P11: duplicate option, non-repeatable -> last wins or error? ----
    var p11 = ArgParser(String("app"))
    p11.add_option(String("name"), String("n"))
    var a11 = List[String](); a11.append(String("--name")); a11.append(String("first")); a11.append(String("--name")); a11.append(String("second"))
    try:
        var r = p11.parse(a11)
        print("P11 dup-option: name='", r.get_str(String("name")), "'")
    except e:
        print("P11 dup-option: RAISED:", String(e))

    # ---- P12: unknown long option ----
    var p12 = _mk()
    var a12 = List[String](); a12.append(String("--unknownopt"))
    try:
        var r = p12.parse(a12)
        print("P12 unknown-long: NO RAISE")
    except e:
        print("P12 unknown-long: RAISED:", String(e))

    # ---- P13: option value '=' empty (--name=) ----
    var p13 = ArgParser(String("app"))
    p13.add_option(String("name"), String("n"))
    var a13 = List[String](); a13.append(String("--name="))
    try:
        var r = p13.parse(a13)
        print("P13 --name= empty: name='", r.get_str(String("name")), "' present=", r.has(String("name")))
    except e:
        print("P13 --name= empty: RAISED:", String(e))

    # ---- P14: combined cluster where a middle char is an option needing a value: -vn val ----
    var p14 = ArgParser(String("app"))
    p14.add_flag(String("verbose"), String("v"))
    p14.add_option(String("name"), String("n"))
    var a14 = List[String](); a14.append(String("-vn")); a14.append(String("bob"))
    try:
        var r = p14.parse(a14)
        print("P14 -vn bob: verbose=", r.get_bool(String("verbose")), " name='", r.get_str(String("name")), "'")
    except e:
        print("P14 -vn bob: RAISED:", String(e))

    # ---- P15: lone '-' as positional ----
    var p15 = _mk()
    var a15 = List[String](); a15.append(String("-"))
    try:
        var r = p15.parse(a15)
        print("P15 lone-dash: src='", r.get_str(String("src")), "' positionals=", len(r.positionals()))
    except e:
        print("P15 lone-dash: RAISED:", String(e))

    print("PROBES DONE")
