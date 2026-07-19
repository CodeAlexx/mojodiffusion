#!/usr/bin/env python3
# sqlite/tests/writer_check.py — open the Mojo-written DBs with the REAL
# system sqlite3 (Python's built-in, libsqlite3 3.45). This is the decisive
# oracle: integrity_check must say "ok" and SELECT must return the exact rows.
#
# Run writer_test.mojo FIRST to produce /tmp/sq_w_small.db and /tmp/sq_w_big.db.

import sqlite3
import sys


def main():
    passed = 0
    failed = 0

    def check(cond, name):
        nonlocal passed, failed
        if cond:
            passed += 1
        else:
            failed += 1
            print("  FAIL:", name)

    # ════ small ════
    print("== check small ==")
    con = sqlite3.connect("/tmp/sq_w_small.db")
    cur = con.cursor()

    ic = cur.execute("PRAGMA integrity_check").fetchall()
    print("  integrity_check:", ic)
    check(ic == [("ok",)], "small integrity_check == ok")

    names = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print("  sqlite_master tables:", names)
    check(names == ["t"], "small sqlite_master lists t")

    expected = [
        (1, "hello", 3.5),
        (-7, "world", 2.25),
        (0, "", 0.0),
        (1000000, "tail", -1.5),
    ]
    got = cur.execute("SELECT a,b,c FROM t").fetchall()
    print("  inserted:", expected)
    print("  SELECT  :", got)
    check(got == expected, "small SELECT a,b,c matches inserted rows exactly")
    con.close()

    # ════ big (multi-leaf / interior) ════
    print("== check big ==")
    con = sqlite3.connect("/tmp/sq_w_big.db")
    cur = con.cursor()

    ic = cur.execute("PRAGMA integrity_check").fetchall()
    print("  integrity_check:", ic)
    check(ic == [("ok",)], "big integrity_check == ok")

    cnt = cur.execute("SELECT count(*) FROM big").fetchone()[0]
    print("  SELECT count(*) FROM big:", cnt)
    check(cnt == 500, "big count(*) == 500")

    first = cur.execute("SELECT id,name,val FROM big ORDER BY id LIMIT 1").fetchone()
    last = cur.execute("SELECT id,name,val FROM big ORDER BY id DESC LIMIT 1").fetchone()
    mid = cur.execute("SELECT id,name,val FROM big WHERE id=250").fetchone()
    print("  first:", first, "mid:", mid, "last:", last)
    check(first == (1, "name_1", 7), "big first row")
    check(mid == (250, "name_250", 1750), "big mid row")
    check(last == (500, "name_500", 3500), "big last row")

    # full ordered scan equals the generated sequence
    allrows = cur.execute("SELECT id,name,val FROM big ORDER BY id").fetchall()
    want = [(i + 1, "name_%d" % (i + 1), (i + 1) * 7) for i in range(500)]
    check(allrows == want, "big full ordered scan matches generated rows")
    con.close()

    print("passed:", passed, "failed:", failed)
    if failed == 0:
        print("ALL WRITER CHECKS PASSED")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
