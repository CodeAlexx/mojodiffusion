# sqlite/tests/query_test.mojo — SELECT-engine parity against Python sqlite3.
#
# Run query_fixtures.py FIRST to build /tmp/sq_people.db and dump expected
# result sets to /tmp/sq_query_expect.txt. Then:
#   pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . \
#       sqlite/tests/query_test.mojo
#
# For each query in the dump we run execute_select() and compare the result set
# cell-by-cell (and row order) against Python's. REAL cells compare as exact
# Float64 (both read the same stored doubles); INT/TEXT compare exactly.

from sqlite.value import Value, VT_NULL, VT_INT, VT_REAL, VT_TEXT, VT_BLOB
from sqlite.db import Database
from sqlite.query import execute_select, ResultSet

comptime B_NL = UInt8(10)
comptime B_SP = UInt8(32)
comptime B_US = UInt8(31)    # unit separator between cells


# ─── read the whole expect file into byte-delimited lines ────────────────────
def _read_lines(path: String) raises -> List[String]:
    var fh = open(path, "r")
    var txt = fh.read()
    fh.close()
    var tb = txt.as_bytes()
    var n = txt.byte_length()
    var lines = List[String]()
    var cur = String("")
    var any = False
    for i in range(n):
        var ch = tb[i]
        if ch == B_NL:
            lines.append(cur)
            cur = String("")
            any = False
        else:
            cur += chr(Int(ch))
            any = True
    if any:
        lines.append(cur)
    return lines^


# Split a line on the US (0x1f) separator into cells.
def _split_us(s: String) raises -> List[String]:
    var sb = s.as_bytes()
    var n = s.byte_length()
    var out = List[String]()
    var cur = String("")
    for i in range(n):
        if sb[i] == B_US:
            out.append(cur)
            cur = String("")
        else:
            cur += chr(Int(sb[i]))
    out.append(cur)
    return out^


# Strip a leading "<tag> " prefix, returning the remainder of the line.
def _after_first_space(s: String) raises -> String:
    var sb = s.as_bytes()
    var n = s.byte_length()
    for i in range(n):
        if sb[i] == B_SP:
            var out = String("")
            for j in range(i + 1, n):
                out += chr(Int(sb[j]))
            return out^
    return String("")


# Format a Mojo Value the way query_fixtures.py's cell_repr() formats it, for
# INT and TEXT. REAL is compared numerically (see below), not by string.
def _cell_int_text(v: Value) raises -> String:
    if v.kind == VT_INT:
        return String(Int(v.as_int()))
    if v.kind == VT_TEXT:
        return v.as_text()
    return String("")


def main() raises:
    var p = 0
    var f = 0

    var db = Database.open("/tmp/sq_people.db")
    var lines = _read_lines("/tmp/sq_query_expect.txt")

    # parse: first line "count N"
    var li = 0
    var qcount = Int(_after_first_space(lines[li]))
    li += 1

    for _q in range(qcount):
        # Q <idx> <sql>
        ref qline = lines[li]
        # the SQL is everything after "Q <idx> "
        var rest = _after_first_space(qline)          # "<idx> <sql>"
        var sql = _after_first_space(rest)            # "<sql>"
        li += 1
        # C <ncols>
        var ncols = Int(_after_first_space(lines[li]))
        li += 1
        # N <nrows>
        var nrows = Int(_after_first_space(lines[li]))
        li += 1

        # collect expected rows
        var exp_rows = List[List[String]]()
        for _r in range(nrows):
            var rowtxt = _after_first_space(lines[li])   # after "R "
            li += 1
            exp_rows.append(_split_us(rowtxt))

        # run our engine
        var rs = execute_select(db, sql)

        var ok = True
        # column count
        if len(rs.columns) != ncols:
            ok = False
        # row count
        if len(rs.rows) != nrows:
            ok = False
        if ok:
            for r in range(nrows):
                ref got = rs.rows[r]
                ref exp = exp_rows[r]
                if len(got) != len(exp):
                    ok = False
                    break
                for c in range(len(got)):
                    ref gv = got[c]
                    ref ec = exp[c]
                    if gv.kind == VT_REAL:
                        # exact double compare — both from same stored bytes
                        var want = Float64(ec)
                        if gv.as_real() != want:
                            ok = False
                            break
                    else:
                        if _cell_int_text(gv) != ec:
                            ok = False
                            break
                if not ok:
                    break

        if ok:
            p += 1
            print("  PASS Q" + String(_q) + " rows=" + String(nrows) + " cols=" + String(ncols) + " (matched Python) : " + sql)
        else:
            f += 1
            print("  FAIL Q" + String(_q) + " : " + sql)
            print("       expected rows=" + String(nrows) + " cols=" + String(ncols) + " | got rows=" + String(len(rs.rows)) + " cols=" + String(len(rs.columns)))

    print("passed:", p, " failed:", f)
    if f == 0:
        print("ALL QUERY TESTS PASSED")
