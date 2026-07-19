# sqlite/tests/reader_test.mojo — read-side parity against Python sqlite3.
#
# Run reader_fixtures.py FIRST to build /tmp/sq_*.db and /tmp/sq_expect.txt.
# Build/run:
#   pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . \
#       sqlite/tests/reader_test.mojo

from sqlite.value import Value
from sqlite.db import Database
from sqlite.btree import walk_table, Record
from sqlite.pager import Pager


def check(mut p: Int, mut f: Int, cond: Bool, name: String):
    if cond:
        p += 1
    else:
        f += 1
        print("  FAIL:", name)


# ─── tiny key/value reader for /tmp/sq_expect.txt (ASCII, byte-level) ────────
def _read_expect(path: String) raises -> List[String]:
    var fh = open(path, "r")
    var txt = fh.read()
    fh.close()
    var tb = txt.as_bytes()
    var n = txt.byte_length()
    var lines = List[String]()
    var cur = String("")
    for i in range(n):
        var ch = tb[i]
        if ch == UInt8(10):  # \n
            if len(cur) > 0:
                lines.append(cur)
            cur = String("")
        else:
            cur += chr(Int(ch))
    if len(cur) > 0:
        lines.append(cur)
    return lines^


def expect_str(lines: List[String], key: String) raises -> String:
    for i in range(len(lines)):
        ref ln = lines[i]
        var lb = ln.as_bytes()
        var m = ln.byte_length()
        var sp = -1
        for j in range(m):
            if lb[j] == UInt8(32):  # space
                sp = j
                break
        if sp < 0:
            continue
        var k = String("")
        for j in range(sp):
            k += chr(Int(lb[j]))
        if k == key:
            var v = String("")
            for j in range(sp + 1, m):
                v += chr(Int(lb[j]))
            return v^
    raise Error("expect key not found: " + key)


def expect_int(lines: List[String], key: String) raises -> Int:
    return Int(expect_str(lines, key))


def main() raises:
    var p = 0
    var f = 0

    # ════ CASE 1: small (mixed types, 3 rows) ════
    print("== small ==")
    var dbs = Database.open("/tmp/sq_small.db")
    var rows = dbs.read_table("t")
    check(p, f, len(rows) == 3, "small row count == 3 (got " + String(len(rows)) + ")")
    # row 0: (1, 3.5, "hello", b"\x01\x02\x03", NULL)
    ref r0 = rows[0].values
    check(p, f, len(r0) == 5, "small row0 col count 5")
    check(p, f, r0[0].as_int() == 1, "small r0 a==1")
    check(p, f, r0[1].as_real() == 3.5, "small r0 b==3.5")
    check(p, f, r0[2].as_text() == "hello", "small r0 c==hello")
    var b0 = r0[3].as_blob()
    check(p, f, len(b0) == 3 and b0[0] == UInt8(1) and b0[1] == UInt8(2) and b0[2] == UInt8(3), "small r0 d blob 01 02 03")
    check(p, f, r0[4].is_null(), "small r0 e NULL")
    # row 1: (-7, 2.25, "world", b"\xff\x00\xaa", 99)
    ref r1 = rows[1].values
    check(p, f, r1[0].as_int() == -7, "small r1 a==-7")
    check(p, f, r1[1].as_real() == 2.25, "small r1 b==2.25")
    check(p, f, r1[2].as_text() == "world", "small r1 c==world")
    var b1 = r1[3].as_blob()
    check(p, f, len(b1) == 3 and b1[0] == UInt8(0xFF) and b1[1] == UInt8(0) and b1[2] == UInt8(0xAA), "small r1 d blob ff 00 aa")
    check(p, f, r1[4].as_int() == 99, "small r1 e==99")
    # row 2: (1000000, 0.0, "", b"", NULL)
    ref r2 = rows[2].values
    check(p, f, r2[0].as_int() == 1000000, "small r2 a==1000000")
    check(p, f, r2[2].as_text() == "", "small r2 c empty text")
    check(p, f, len(r2[3].as_blob()) == 0, "small r2 d empty blob")
    check(p, f, r2[4].is_null(), "small r2 e NULL")
    print("  small rows:", len(rows), "| r0.a=", r0[0].as_int(), "r0.c=", r0[2].as_text(), "| r1.a=", r1[0].as_int(), "| r2.a=", r2[0].as_int())

    # ════ CASE 2: multipage (interior + multiple leaves) ════
    print("== multipage ==")
    var exp = _read_expect("/tmp/sq_expect.txt")
    var pgr = Pager.open("/tmp/sq_multi.db")
    var expected_pc = expect_int(exp, "multi_page_count")
    check(p, f, pgr.page_count() == expected_pc, "multi page_count==" + String(expected_pc) + " (got " + String(pgr.page_count()) + ")")
    check(p, f, expected_pc > 2, "multi spans >2 pages (interior page implied): " + String(expected_pc))

    var dbm = Database.open("/tmp/sq_multi.db")
    var mrows = dbm.read_table("big")
    var ecnt = expect_int(exp, "multi_count")
    check(p, f, len(mrows) == ecnt, "multi row count==" + String(ecnt) + " (got " + String(len(mrows)) + ")")

    # rows are (id INTEGER PRIMARY KEY -> rowid alias, name, val)
    # first
    ref mf = mrows[0].values
    check(p, f, Int(mf[0].as_int()) == expect_int(exp, "multi_first_id"), "multi first id")
    check(p, f, mf[1].as_text() == expect_str(exp, "multi_first_name"), "multi first name")
    check(p, f, Int(mf[2].as_int()) == expect_int(exp, "multi_first_val"), "multi first val")
    # middle (id==1000 -> index 999)
    ref mm = mrows[999].values
    check(p, f, Int(mm[0].as_int()) == expect_int(exp, "multi_mid_id"), "multi mid id")
    check(p, f, mm[1].as_text() == expect_str(exp, "multi_mid_name"), "multi mid name")
    check(p, f, Int(mm[2].as_int()) == expect_int(exp, "multi_mid_val"), "multi mid val")
    # last
    ref ml = mrows[len(mrows) - 1].values
    check(p, f, Int(ml[0].as_int()) == expect_int(exp, "multi_last_id"), "multi last id")
    check(p, f, ml[1].as_text() == expect_str(exp, "multi_last_name"), "multi last name")
    check(p, f, Int(ml[2].as_int()) == expect_int(exp, "multi_last_val"), "multi last val")
    # rowid-alias correctness: id column equals rowid for every spot-checked row
    print("  multi rows:", len(mrows), "| first id/name/val=", Int(mf[0].as_int()), mf[1].as_text(), Int(mf[2].as_int()))
    print("  multi mid id/name/val=", Int(mm[0].as_int()), mm[1].as_text(), Int(mm[2].as_int()))
    print("  multi last id/name/val=", Int(ml[0].as_int()), ml[1].as_text(), Int(ml[2].as_int()))

    # ════ CASE 3: overflow (long text + long blob spill) ════
    print("== overflow ==")
    var dbo = Database.open("/tmp/sq_overflow.db")
    var orows = dbo.read_table("ov")
    check(p, f, len(orows) == 2, "overflow row count==2")
    ref o0 = orows[0].values
    var etlen = expect_int(exp, "overflow_text_len")
    var eblen = expect_int(exp, "overflow_blob_len")
    var txt = o0[1].as_text()
    check(p, f, txt.byte_length() == etlen, "overflow text len==" + String(etlen) + " (got " + String(txt.byte_length()) + ")")
    # byte-exact: all 'X' (0x58)
    var tb2 = txt.as_bytes()
    var all_x = True
    for i in range(txt.byte_length()):
        if tb2[i] != UInt8(88):
            all_x = False
            break
    check(p, f, all_x, "overflow text byte-exact (all 'X')")
    var ob = o0[2].as_blob()
    check(p, f, len(ob) == eblen, "overflow blob len==" + String(eblen) + " (got " + String(len(ob)) + ")")
    var blob_ok = True
    for i in range(len(ob)):
        if Int(ob[i]) != (i % 256):
            blob_ok = False
            break
    check(p, f, blob_ok, "overflow blob byte-exact (i%256)")
    # the short second row should be intact too
    ref o1 = orows[1].values
    check(p, f, o1[1].as_text() == "short", "overflow row1 short text")
    print("  overflow text_len:", len(txt), "blob_len:", len(ob), "| all_x:", all_x, "blob_ok:", blob_ok)

    # ════ CASE 4: schema ════
    print("== schema ==")
    var dbsc = Database.open("/tmp/sq_schema.db")
    var names = dbsc.table_names()
    check(p, f, len(names) == 2, "schema table count==2 (got " + String(len(names)) + ")")
    var has_users = False
    var has_posts = False
    for i in range(len(names)):
        if names[i] == "users":
            has_users = True
        if names[i] == "posts":
            has_posts = True
    check(p, f, has_users and has_posts, "schema has users+posts")
    var ucols = dbsc.columns("users")
    check(p, f, len(ucols) == 3, "users col count==3 (got " + String(len(ucols)) + ")")
    check(p, f, ucols[0] == "uid" and ucols[1] == "email" and ucols[2] == "age", "users cols uid/email/age")
    var pcols = dbsc.columns("posts")
    check(p, f, len(pcols) == 4 and pcols[0] == "pid" and pcols[3] == "author", "posts cols pid..author")
    # rowid-alias surfaced for users.uid
    var urows = dbsc.read_table("users")
    check(p, f, len(urows) == 1 and Int(urows[0].values[0].as_int()) == 1, "users uid rowid-alias==1")
    var sqlu = dbsc.schema_sql("users")
    check(p, f, len(sqlu) > 0, "schema_sql(users) non-empty")
    print("  tables:", names[0], names[1])
    print("  users cols:", ucols[0], ucols[1], ucols[2], "| posts cols:", pcols[0], pcols[1], pcols[2], pcols[3])
    print("  schema_sql(users):", sqlu)

    print("passed:", p, " failed:", f)
    if f == 0:
        print("ALL READER TESTS PASSED")
