#!/usr/bin/env python3
# Oracle for config/ini.mojo — dumps Python's stdlib configparser parse of a
# representative INI to a flat text file the Mojo test reads back and compares.
#
# Configuration choices made so the comparison is apples-to-apples with the
# Mojo parser (see ini.mojo "DIVERGENCE" notes):
#   * optionxform = str        -> preserve key case (configparser lowercases by default)
#   * use c._sections / c._defaults raw maps -> NO DEFAULT-fallback merge into
#     each section (configparser's items()/[] view merges DEFAULT keys into every
#     section; the Mojo parser stores DEFAULT as its own plain section instead).
#   * the [DEFAULT] section is emitted under the literal name "DEFAULT".
#
# Output format (one record per line), written to ini_oracle_out.txt:
#   SECTION\t<section-name>
#   KV\t<section-name>\t<key>\t<value>
# values are escaped: \n -> \\n, \t -> \\t, \\ -> \\\\

import configparser
import os
import sys

# The single representative INI shared by oracle + Mojo test.
INI_TEXT = """[DEFAULT]
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


def esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, "ini_oracle_out.txt")

    c = configparser.ConfigParser()
    c.optionxform = str  # preserve key case to match the Mojo parser
    c.read_string(INI_TEXT)

    # Build a unified section map: literal "DEFAULT" + each real section,
    # each with ONLY its own raw keys (no DEFAULT fallback merge).
    sections = {}
    if c._defaults:
        sections["DEFAULT"] = dict(c._defaults)
    for s in c.sections():
        sections[s] = dict(c._sections[s])

    lines = []
    for sec, kv in sections.items():
        lines.append("SECTION\t" + esc(sec))
        for k, v in kv.items():
            lines.append("KV\t" + esc(sec) + "\t" + esc(k) + "\t" + esc(v))

    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # Human-readable dump to stdout for the report.
    print("=== configparser oracle (optionxform=str, raw sections, no DEFAULT merge) ===")
    for sec, kv in sections.items():
        print("[%s]" % sec)
        for k, v in kv.items():
            print("  %r = %r" % (k, v))
    print("wrote:", out_path)


if __name__ == "__main__":
    main()
