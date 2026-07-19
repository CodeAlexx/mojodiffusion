#!/usr/bin/env python3
# argparse_oracle.py — a small Python argparse driver used as a cross-check
# oracle for cli.parser (the Mojo library). It defines a fixed spec that mirrors
# the Mojo ArgParser used in the Mojo test, parses argv passed after a "--"
# sentinel, and dumps the resolved values as `key=value` lines on stdout so the
# Mojo test can compare its own parse for the overlapping semantics.
#
# Spec (mirrors the Mojo `_build_oracle()` parser):
#   --verbose / -v   : bool flag
#   --name    / -n   : str option
#   --count   / -k   : int option, default 7
#   --level   / -l   : choice in {low, mid, high}
#   --inc            : repeatable (append) -> list
#   src, dst         : two positionals
#   rest...          : variadic remainder
#
# Output lines (only for what resolved):
#   verbose=true|false
#   name=<str>          (omitted if not given)
#   count=<int>
#   level=<str>         (omitted if not given)
#   inc=a|b|c           ('|'-joined; omitted if empty)
#   src=<str> / dst=<str>
#   rest=x|y            ('|'-joined; omitted if empty)
#
# Usage: argparse_oracle.py -- <argv...>

import argparse
import sys


def build():
    p = argparse.ArgumentParser(prog="app", add_help=False)
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--name", "-n")
    p.add_argument("--count", "-k", type=int, default=7)
    p.add_argument("--level", "-l", choices=["low", "mid", "high"])
    p.add_argument("--inc", action="append", default=[])
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("rest", nargs="*")
    return p


def main():
    argv = sys.argv[1:]
    if "--" in argv:
        idx = argv.index("--")
        argv = argv[idx + 1:]
    p = build()
    ns = p.parse_args(argv)
    out = []
    out.append("verbose=" + ("true" if ns.verbose else "false"))
    if ns.name is not None:
        out.append("name=" + ns.name)
    out.append("count=" + str(ns.count))
    if ns.level is not None:
        out.append("level=" + ns.level)
    if ns.inc:
        out.append("inc=" + "|".join(ns.inc))
    out.append("src=" + ns.src)
    out.append("dst=" + ns.dst)
    if ns.rest:
        out.append("rest=" + "|".join(ns.rest))
    print("\n".join(out))


if __name__ == "__main__":
    main()
