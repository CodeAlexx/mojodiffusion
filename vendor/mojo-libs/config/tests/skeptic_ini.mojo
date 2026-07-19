# Adversarial probes for config/ini.mojo. Reports observed behavior verbatim.

from config.value import ConfigTree, ConfigValue
from config.ini import parse_ini


def _show(label: String, text: String, section: String, key: String) raises:
    try:
        var t = parse_ini(text)
        if t.has(section, key):
            print(label, ": [", section, "]", key, "='", t.get(section, key).as_str(), "'")
        else:
            print(label, ": MISSING [", section, "]", key)
    except e:
        print(label, ": RAISED:", String(e))


def main() raises:
    # I1: pre-header keys -> global section "" (documented divergence from configparser)
    _show(String("I1 pre-header"), String("foo = bar\n[s]\nx=1\n"), String(""), String("foo"))

    # I2: [DEFAULT] kept as plain section (NOT merged into others)
    var t2 = parse_ini(String("[DEFAULT]\nx = 1\n[s]\ny = 2\n"))
    print("I2 DEFAULT-merge: [s] has x? ", t2.has(String("s"), String("x")), " [DEFAULT] x='", t2.get(String("DEFAULT"), String("x")).as_str(), "'")

    # I3: duplicate section headers -> merge or replace?
    var t3 = parse_ini(String("[s]\na = 1\n[s]\nb = 2\n"))
    print("I3 dup-section: [s] has a? ", t3.has(String("s"), String("a")), " has b? ", t3.has(String("s"), String("b")))

    # I4: duplicate keys -> last wins
    _show(String("I4 dup-key"), String("[s]\nx = 1\nx = 2\n"), String("s"), String("x"))

    # I5: value containing '=' -> split on FIRST separator only
    _show(String("I5 eq-in-value"), String("[s]\nurl = a=b:c\n"), String("s"), String("url"))

    # I5b: value containing ':' when '=' present later -> first separator wins (':' before '=')
    _show(String("I5b colon-first"), String("[s]\nkey: a=b\n"), String("s"), String("key"))

    # I5c: '=' before ':'
    _show(String("I5c eq-first"), String("[s]\nkey = a:b\n"), String("s"), String("key"))

    # I6: empty value
    _show(String("I6 empty-val"), String("[s]\nx =\n"), String("s"), String("x"))

    # I7: key with no separator -> raises
    _show(String("I7 no-sep"), String("[s]\njustakey\n"), String("s"), String("justakey"))

    # I8: inline ';' NOT stripped (claimed) -> preserved verbatim
    _show(String("I8 inline-semicolon"), String("[s]\nx = val ; trailing\n"), String("s"), String("x"))

    # I8b: inline '#' NOT stripped
    _show(String("I8b inline-hash"), String("[s]\nx = val # trailing\n"), String("s"), String("x"))

    # I9: colon-key (key with ':' separator)
    _show(String("I9 colon-key"), String("[s]\nhost : localhost\n"), String("s"), String("host"))

    # I10: whitespace-only value
    _show(String("I10 ws-val"), String("[s]\nx =    \n"), String("s"), String("x"))

    # I11: section name with surrounding spaces
    var t11 = parse_ini(String("[ sec ]\nx = 1\n"))
    print("I11 spaced-section: has [sec]? ", t11.has(String("sec"), String("x")), " has [ sec ]? ", t11.has(String(" sec "), String("x")))

    print("PROBES DONE")
