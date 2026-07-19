#!/usr/bin/env python3
# toml_oracle.py — the REAL oracle for toml.mojo.
#
# Parses a fixed TOML document with Python's stdlib `tomllib` (the reference),
# flattens the nested dict to dotted-section/key -> typed value using the SAME
# section mapping toml.mojo uses (a `[a.b]` table or a dotted key `a.b.c = v`
# flattens to section "a.b", key "c"), and emits a simple line-based oracle file
# that toml_test.mojo reads back and asserts against parse_toml's output.
#
# Output format (one record per line), fields tab-separated:
#   S\t<section>\t<key>\t<typekind>\t<payload>
#   A\t<section>\t<key>\tarray\t<len>            (array header)
#   E\t<section>\t<key>\t<idx>\t<typekind>\t<payload>   (array element, scalars)
# typekind in: int float bool str
# payloads: int -> decimal; float -> repr; bool -> true/false; str -> raw bytes
#   with \n \t \r \\ escaped so it stays single-line and round-trips exactly.
#
# We intentionally only emit scalar array elements and ONE level of array
# nesting flattened (nested arrays are emitted as nested via N records); see
# below. The test document keeps nesting to depth 2 to keep the oracle simple.

import sys
import tomllib

# The representative TOML document (also embedded verbatim in toml_test.mojo).
TOML = b"""# top-level comment
title = "TOML \\"Example\\""        # trailing comment with a quote
literal = 'C:\\\\no\\\\escape'
maxconn = 1_000
ratio = 6.022e23
small = 0.5
neg = -17
flag = true
nums = [1, 2, 3]
words = ["a", "b", "c"]
mixed = [1, "two", 3.0, false]
matrix = [
    [1, 2],
    [3, 4],
]
escaped = "line1\\nline2\\ttab"

[server]
host = "localhost"
port = 8080
enabled = false

[a.b]
deep = 42
name = 'nested'

[owner]
dotted.key = "viadot"
"""


def escape_str(s: str) -> str:
    out = []
    for ch in s:
        if ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        else:
            out.append(ch)
    return "".join(out)


def typekind(v):
    # bool must precede int (bool is subclass of int in Python)
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    return None


def scalar_payload(v):
    k = typekind(v)
    if k == "bool":
        return "true" if v else "false"
    if k == "int":
        return str(v)
    if k == "float":
        return repr(v)
    if k == "str":
        return escape_str(v)
    raise SystemExit(f"unsupported scalar in oracle: {v!r}")


def emit_array(lines, section, key, arr):
    lines.append(f"A\t{section}\t{key}\tarray\t{len(arr)}")
    for i, el in enumerate(arr):
        if isinstance(el, list):
            # nested array: emit as N record with its own length, then sub elems
            lines.append(f"N\t{section}\t{key}\t{i}\tarray\t{len(el)}")
            for j, sub in enumerate(el):
                lines.append(
                    f"M\t{section}\t{key}\t{i}\t{j}\t{typekind(sub)}\t{scalar_payload(sub)}"
                )
        else:
            lines.append(
                f"E\t{section}\t{key}\t{i}\t{typekind(el)}\t{scalar_payload(el)}"
            )


def flatten(lines, section, table):
    for k, v in table.items():
        if isinstance(v, dict):
            sub = k if not section else f"{section}.{k}"
            flatten(lines, sub, v)
        elif isinstance(v, list):
            emit_array(lines, section, k, v)
        else:
            tk = typekind(v)
            if tk is None:
                # datetimes etc. — out of scope for toml.mojo, skip in oracle
                continue
            lines.append(f"S\t{section}\t{k}\t{tk}\t{scalar_payload(v)}")


def main():
    parsed = tomllib.loads(TOML.decode("utf-8"))
    lines = []
    flatten(lines, "", parsed)
    out = "\n".join(lines) + "\n"
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            f.write(out)
    else:
        sys.stdout.write(out)


if __name__ == "__main__":
    main()
