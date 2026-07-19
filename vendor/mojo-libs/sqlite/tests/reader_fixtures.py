#!/usr/bin/env python3
# sqlite/tests/reader_fixtures.py
#
# Build the SQLite databases (the oracle) that reader_test.mojo reads back, and
# dump the expected answers for the multipage spot-checks to a side file the
# Mojo test can compare against. Uses the system libsqlite3 via Python's
# sqlite3 module — this IS the reference implementation.

import os
import sqlite3

SMALL = "/tmp/sq_small.db"
MULTI = "/tmp/sq_multi.db"
OVERFLOW = "/tmp/sq_overflow.db"
SCHEMA = "/tmp/sq_schema.db"
EXPECT = "/tmp/sq_expect.txt"


def fresh(path):
    if os.path.exists(path):
        os.remove(path)
    return sqlite3.connect(path)


def build_small():
    con = fresh(SMALL)
    cur = con.cursor()
    cur.execute("CREATE TABLE t (a INTEGER, b REAL, c TEXT, d BLOB, e INTEGER)")
    # 3 rows mixing INTEGER, REAL, TEXT, BLOB, NULL
    cur.execute("INSERT INTO t VALUES (?,?,?,?,?)", (1, 3.5, "hello", b"\x01\x02\x03", None))
    cur.execute("INSERT INTO t VALUES (?,?,?,?,?)", (-7, 2.25, "world", b"\xff\x00\xaa", 99))
    cur.execute("INSERT INTO t VALUES (?,?,?,?,?)", (1000000, 0.0, "", b"", None))
    con.commit()
    con.close()


def build_multi():
    con = fresh(MULTI)
    cur = con.cursor()
    cur.execute("CREATE TABLE big (id INTEGER PRIMARY KEY, name TEXT, val INTEGER)")
    rows = []
    for i in range(1, 2001):
        rows.append((i, "name_%d" % i, i * 7))
    cur.executemany("INSERT INTO big VALUES (?,?,?)", rows)
    con.commit()
    # confirm the b-tree spans multiple leaves (interior pages used)
    pc = cur.execute("PRAGMA page_count").fetchone()[0]
    ps = cur.execute("PRAGMA page_size").fetchone()[0]
    # spot-check rows
    first = cur.execute("SELECT id,name,val FROM big ORDER BY id LIMIT 1").fetchone()
    last = cur.execute("SELECT id,name,val FROM big ORDER BY id DESC LIMIT 1").fetchone()
    mid = cur.execute("SELECT id,name,val FROM big WHERE id=1000").fetchone()
    cnt = cur.execute("SELECT COUNT(*) FROM big").fetchone()[0]
    con.close()
    return pc, ps, first, mid, last, cnt


def build_overflow():
    con = fresh(OVERFLOW)
    cur = con.cursor()
    cur.execute("CREATE TABLE ov (id INTEGER, body TEXT, blob BLOB)")
    longtext = "X" * 5000  # spills to overflow pages on a 4096-byte page
    longblob = bytes((i % 256) for i in range(6000))
    cur.execute("INSERT INTO ov VALUES (?,?,?)", (1, longtext, longblob))
    cur.execute("INSERT INTO ov VALUES (?,?,?)", (2, "short", b"tiny"))
    con.commit()
    con.close()
    return len(longtext), len(longblob)


def build_schema():
    con = fresh(SCHEMA)
    cur = con.cursor()
    cur.execute("CREATE TABLE users (uid INTEGER PRIMARY KEY, email TEXT, age INTEGER)")
    cur.execute("CREATE TABLE posts (pid INTEGER PRIMARY KEY, title TEXT, body TEXT, author INTEGER)")
    cur.execute("INSERT INTO users VALUES (1,'a@b.c',30)")
    cur.execute("INSERT INTO posts VALUES (1,'hi','hello world',1)")
    con.commit()
    con.close()


def main():
    build_small()
    pc, ps, first, mid, last, cnt = build_multi()
    tlen, blen = build_overflow()
    build_schema()

    with open(EXPECT, "w") as f:
        f.write("multi_page_count %d\n" % pc)
        f.write("multi_page_size %d\n" % ps)
        f.write("multi_count %d\n" % cnt)
        f.write("multi_first_id %d\n" % first[0])
        f.write("multi_first_name %s\n" % first[1])
        f.write("multi_first_val %d\n" % first[2])
        f.write("multi_mid_id %d\n" % mid[0])
        f.write("multi_mid_name %s\n" % mid[1])
        f.write("multi_mid_val %d\n" % mid[2])
        f.write("multi_last_id %d\n" % last[0])
        f.write("multi_last_name %s\n" % last[1])
        f.write("multi_last_val %d\n" % last[2])
        f.write("overflow_text_len %d\n" % tlen)
        f.write("overflow_blob_len %d\n" % blen)

    print("sqlite3 version:", sqlite3.sqlite_version)
    print("built:", SMALL, MULTI, OVERFLOW, SCHEMA)
    print("multi: page_count=%d page_size=%d rows=%d" % (pc, ps, cnt))
    print("  first=%s mid=%s last=%s" % (first, mid, last))
    print("overflow: text_len=%d blob_len=%d" % (tlen, blen))
    # report whether multipage truly used interior pages: a single leaf can
    # hold far fewer than 2000 small rows on a 4096-byte page, so >1 leaf page
    # implies an interior page exists.
    print("expectations written to", EXPECT)


if __name__ == "__main__":
    main()
