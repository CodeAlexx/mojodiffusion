# 1:1 port of Serenity modules/util/path_util.py (the subset the cadence needs)
#
# canonical_join (path_util.py:24-30): os.path.join(base, *paths) then replace
# '\\' with '/'. On Linux os.path.join just inserts '/' between non-empty
# segments (and discards `base` if a later segment is absolute). We reproduce
# the common case the cadence exercises (relative segments joined with '/').
#
# Mojo String has no positional s[i] indexing (only byte=/codepoint= keyword
# forms), so the leading/trailing '/' tests use byte comparison via as_bytes().
# ASCII '/' is byte 47.

comptime _SLASH: UInt8 = 47   # ord('/')


# _join2 — os.path.join(a, b) for the two-arg case (the cadence only joins pairs).
# - empty a            -> b
# - b starts with '/'  -> b           (os.path.join discards `a` for absolute b)
# - else               -> a (sans one trailing '/') + '/' + b
def _join2(a: String, b: String) -> String:
    if a.byte_length() == 0:
        return b

    var bb = b.as_bytes()
    if b.byte_length() > 0 and bb[0] == _SLASH:
        return b

    # strip a single trailing '/' on `a` so we don't emit "a//b"
    var ab = a.as_bytes()
    var an = a.byte_length()
    var base = String("")
    var keep = an
    if an > 0 and ab[an - 1] == _SLASH:
        keep = an - 1
    for i in range(keep):
        base += chr(Int(ab[i]))

    return base + "/" + b


# canonical_join(base, *paths) — path_util.canonical_join. On Linux '\\'→'/' is a
# no-op (no backslashes in joined POSIX paths), so the join IS the canonical form.
def canonical_join(base: String, p1: String) -> String:
    return _join2(base, p1)


def canonical_join(base: String, p1: String, p2: String) -> String:
    return _join2(_join2(base, p1), p2)
