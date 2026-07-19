# config/ini.mojo — an INI parser. 100% Mojo, no FFI.
#
# Produces a `ConfigTree` (from config/value.mojo). All values are stored as
# strings via `ConfigValue.str_` — INI is untyped; typed access happens later
# through the layered `Config` wrapper.
#
# Grammar (deliberately small; tracks Python's stdlib `configparser` defaults
# for the common cases):
#   [section]        section header
#   key = value      assignment ('=' separator)
#   key : value      assignment (':' separator) — whichever appears FIRST wins
#   ; comment        full-line comment (only when ';' is the first non-space char)
#   # comment        full-line comment (only when '#' is the first non-space char)
#   <blank line>     ignored
#
# Whitespace surrounding both key and value is trimmed. Values may be empty
# (`key =` -> "").
#
# ---- DEFAULT / global-section semantics (chosen, and where it diverges) ------
# * A literal `[DEFAULT]` header parses into a section named "DEFAULT", stored
#   as a *plain* section — its keys are NOT auto-merged as fallbacks into other
#   sections. (configparser DOES merge DEFAULT keys into every section's view;
#   we keep DEFAULT as an ordinary section so the layered Config can decide
#   overlay policy itself.)
# * Keys appearing BEFORE any `[section]` header are placed in section "" (the
#   ConfigTree global scope). configparser instead RAISES MissingSectionHeaderError
#   for pre-header keys — this is a deliberate, documented divergence (we are
#   more permissive). To stay oracle-comparable, the verification INI puts its
#   shared keys under an explicit `[DEFAULT]` header rather than pre-header.
#
# ---- Unsupported (documented) -----------------------------------------------
# * Interpolation `%(name)s` / `${name}` — values are returned verbatim.
# * Line continuations (indented continuation lines) — each physical line is a
#   record; an indented line with no separator raises.
# * Inline comments (`key = val ; trailing`) — matches configparser's default
#   (inline `;`/`#` are part of the value, NOT stripped).
#
# ---- Duplicate keys ----------------------------------------------------------
# Last assignment wins (the ConfigTree.set call overwrites).

from config.value import ConfigTree, ConfigValue

comptime DEFAULT_SECTION = "DEFAULT"


def parse_ini(text: String) raises -> ConfigTree:
    """Parse INI text into a ConfigTree. All values are stored as strings."""
    var tree = ConfigTree()
    # Pre-header keys go into the global scope (section "").
    var current = String("")

    var lines = _split_lines(text)
    for li in range(len(lines)):
        var line = _trim(lines[li])
        var n = line.byte_length()
        if n == 0:
            continue
        var b = line.as_bytes()
        var first = Int(b[0])

        # Full-line comments: ';' or '#' as the first non-space char.
        if first == 0x3B or first == 0x23:  # ';' or '#'
            continue

        # Section header: [name]
        if first == 0x5B:  # '['
            var close = -1
            for i in range(1, n):
                if Int(b[i]) == 0x5D:  # ']'
                    close = i
                    break
            if close < 0:
                raise Error("unterminated section header: " + line)
            var name = String("")
            for i in range(1, close):
                name += chr(Int(b[i]))
            current = _trim(name)
            continue

        # Assignment: split on the FIRST '=' or ':' (whichever comes first).
        var sep = -1
        for i in range(n):
            var c = Int(b[i])
            if c == 0x3D or c == 0x3A:  # '=' or ':'
                sep = i
                break
        if sep < 0:
            raise Error("invalid INI line (no '=' or ':'): " + line)

        var key = String("")
        for i in range(sep):
            key += chr(Int(b[i]))
        var val = String("")
        for i in range(sep + 1, n):
            val += chr(Int(b[i]))

        var k = _trim(key)
        if k.byte_length() == 0:
            raise Error("empty key in INI line: " + line)
        # Last-wins: set overwrites any prior value for this (section, key).
        tree.set(current, k, ConfigValue.str_(_trim(val)))

    return tree^


def parse_ini_file(path: String) raises -> ConfigTree:
    """Read a file at `path` and parse its contents as INI."""
    var text: String
    with open(path, "r") as fh:
        text = fh.read()
    return parse_ini(text)


def _is_space(c: Int) -> Bool:
    # space, tab, CR, LF, vertical tab, form feed
    return c == 0x20 or c == 0x09 or c == 0x0D or c == 0x0A or c == 0x0B or c == 0x0C


def _trim(s: String) -> String:
    var b = s.as_bytes()
    var n = s.byte_length()
    var start = 0
    while start < n and _is_space(Int(b[start])):
        start += 1
    var end = n
    while end > start and _is_space(Int(b[end - 1])):
        end -= 1
    var out = String("")
    for i in range(start, end):
        out += chr(Int(b[i]))
    return out^


def _split_lines(text: String) -> List[String]:
    var out = List[String]()
    var b = text.as_bytes()
    var n = text.byte_length()
    var cur = String("")
    for i in range(n):
        var c = Int(b[i])
        if c == 0x0A:  # '\n'
            out.append(cur)
            cur = String("")
        elif c == 0x0D:  # '\r' (drop; handles CRLF and lone CR)
            continue
        else:
            cur += chr(c)
    out.append(cur)
    return out^
