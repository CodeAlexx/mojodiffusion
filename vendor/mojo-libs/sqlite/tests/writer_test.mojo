# sqlite/tests/writer_test.mojo — write SQLite DBs, verify with our own reader.
#
# This writes /tmp/sq_w_*.db with DbWriter, then:
#   (a) reads them back with our own Database reader (round-trip), and
#   (b) leaves the files for writer_check.py to open with the REAL Python
#       sqlite3 (the decisive oracle).
#
# Build/run from /home/alex/MOJO-libs:
#   pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . \
#       sqlite/tests/writer_test.mojo

from sqlite.value import Value
from sqlite.writer import DbWriter
from sqlite.db import Database
from sqlite.pager import Pager


def check(mut p: Int, mut f: Int, cond: Bool, name: String):
    if cond:
        p += 1
    else:
        f += 1
        print("  FAIL:", name)


def main() raises:
    var p = 0
    var f = 0

    # ════ CASE 1: small mixed-type table, single leaf ════
    print("== write small ==")
    var w = DbWriter.create()
    var cols = List[String]()
    cols.append(String("a"))
    cols.append(String("b"))
    cols.append(String("c"))
    w.create_table(
        String("t"), String("CREATE TABLE t(a INTEGER, b TEXT, c REAL)"), cols^
    )

    # row 1: (1, "hello", 3.5)
    var r1 = List[Value]()
    r1.append(Value.integer(1))
    r1.append(Value.text(String("hello")))
    r1.append(Value.real(3.5))
    w.insert(String("t"), r1)
    # row 2: (-7, "world", 2.25)  -- negative int
    var r2 = List[Value]()
    r2.append(Value.integer(-7))
    r2.append(Value.text(String("world")))
    r2.append(Value.real(2.25))
    w.insert(String("t"), r2)
    # row 3: (0, "", 0.0)  -- empty string, zero real
    var r3 = List[Value]()
    r3.append(Value.integer(0))
    r3.append(Value.text(String("")))
    r3.append(Value.real(0.0))
    w.insert(String("t"), r3)
    # row 4: (1000000, "tail", -1.5)
    var r4 = List[Value]()
    r4.append(Value.integer(1000000))
    r4.append(Value.text(String("tail")))
    r4.append(Value.real(-1.5))
    w.insert(String("t"), r4)

    w.save(String("/tmp/sq_w_small.db"))

    # round-trip with our own reader.
    var db = Database.open(String("/tmp/sq_w_small.db"))
    var names = db.table_names()
    check(p, f, len(names) == 1 and names[0] == "t", "small: sqlite_master lists t")
    var rows = db.read_table(String("t"))
    check(p, f, len(rows) == 4, "small: 4 rows (got " + String(len(rows)) + ")")
    ref a0 = rows[0].values
    check(p, f, a0[0].as_int() == 1 and a0[1].as_text() == "hello" and a0[2].as_real() == 3.5, "small r0 (1,hello,3.5)")
    ref a1 = rows[1].values
    check(p, f, a1[0].as_int() == -7 and a1[1].as_text() == "world" and a1[2].as_real() == 2.25, "small r1 (-7,world,2.25)")
    ref a2 = rows[2].values
    check(p, f, a2[0].as_int() == 0 and a2[1].as_text() == "" and a2[2].as_real() == 0.0, "small r2 (0,'',0.0)")
    ref a3 = rows[3].values
    check(p, f, a3[0].as_int() == 1000000 and a3[1].as_text() == "tail" and a3[2].as_real() == -1.5, "small r3 (1000000,tail,-1.5)")
    print("  small round-trip rows:", len(rows), "| r0=", a0[0].as_int(), a0[1].as_text(), a0[2].as_real())

    # ════ CASE 2: big table forcing multiple leaves + interior root ════
    print("== write big (500 rows) ==")
    var w2 = DbWriter.create()
    var bcols = List[String]()
    bcols.append(String("id"))
    bcols.append(String("name"))
    bcols.append(String("val"))
    w2.create_table(
        String("big"),
        String("CREATE TABLE big(id INTEGER, name TEXT, val INTEGER)"),
        bcols^,
    )
    var N = 500
    for i in range(N):
        var row = List[Value]()
        row.append(Value.integer(Int64(i + 1)))
        row.append(Value.text(String("name_") + String(i + 1)))
        row.append(Value.integer(Int64((i + 1) * 7)))
        w2.insert(String("big"), row)
    w2.save(String("/tmp/sq_w_big.db"))

    var pgr = Pager.open(String("/tmp/sq_w_big.db"))
    check(p, f, pgr.page_count() > 2, "big spans >2 pages (interior root implied): " + String(pgr.page_count()))
    var dbb = Database.open(String("/tmp/sq_w_big.db"))
    var brows = dbb.read_table(String("big"))
    check(p, f, len(brows) == N, "big: " + String(N) + " rows (got " + String(len(brows)) + ")")
    # first / mid / last
    ref bf0 = brows[0].values
    check(p, f, bf0[0].as_int() == 1 and bf0[1].as_text() == "name_1" and bf0[2].as_int() == 7, "big first row")
    ref bm = brows[249].values
    check(p, f, bm[0].as_int() == 250 and bm[1].as_text() == "name_250" and bm[2].as_int() == 1750, "big mid row")
    ref bl = brows[N - 1].values
    check(p, f, bl[0].as_int() == Int64(N) and bl[1].as_text() == (String("name_") + String(N)) and bl[2].as_int() == Int64(N * 7), "big last row")
    print("  big page_count:", pgr.page_count(), "| rows:", len(brows))
    print("  big first/mid/last id:", bf0[0].as_int(), bm[0].as_int(), bl[0].as_int())

    print("passed:", p, " failed:", f)
    if f == 0:
        print("ALL WRITER TESTS PASSED")
