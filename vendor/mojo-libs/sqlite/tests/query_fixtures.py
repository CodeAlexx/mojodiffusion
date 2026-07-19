#!/usr/bin/env python3
# sqlite/tests/query_fixtures.py
#
# Build /tmp/sq_people.db (the oracle) with a `people` table of ~50 varied rows,
# then run the SAME set of SELECT queries query_test.mojo will run, via Python's
# sqlite3 (the reference implementation), and dump expected result sets to
# /tmp/sq_query_expect.txt for byte-exact Mojo comparison.
#
# Dump format (line oriented, ASCII):
#   Q <index> <sql>            # the SQL for query #index
#   C <ncols>                  # number of result columns
#   N <nrows>                  # number of result rows
#   R <cell>\x1f<cell>\x1f...  # one line per row; cells separated by US (0x1f)
# A NULL cell is the literal token \x00NULL\x00 (no real data uses it here).
# REAL cells use repr() so the exact stored double round-trips; INT cells are
# decimal; TEXT cells are verbatim.

import os
import sqlite3

DB = "/tmp/sq_people.db"
EXPECT = "/tmp/sq_query_expect.txt"
US = "\x1f"          # unit separator between cells
NULL_TOKEN = "\x00NULL\x00"

QUERIES = [
    "SELECT * FROM people",
    "SELECT name, age FROM people WHERE age > 30",
    "SELECT name FROM people WHERE city = 'Paris' AND age < 40",
    "SELECT * FROM people WHERE score >= 4.5 OR age = 18",
    "SELECT name, score FROM people WHERE age >= 21 ORDER BY score DESC LIMIT 5",
    "SELECT id, name FROM people WHERE name != 'Bob' ORDER BY id ASC LIMIT 10",
    "SELECT name FROM people WHERE (age < 20 OR age > 60) AND score > 3.0",
]


def fresh(path):
    if os.path.exists(path):
        os.remove(path)
    return sqlite3.connect(path)


def build():
    con = fresh(DB)
    cur = con.cursor()
    cur.execute(
        "CREATE TABLE people ("
        "id INTEGER PRIMARY KEY, name TEXT, age INTEGER, score REAL, city TEXT)"
    )
    names = [
        "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi",
        "Ivan", "Judy", "Mallory", "Niaj", "Olivia", "Peggy", "Rupert",
        "Sybil", "Trent", "Victor", "Walter", "Xena", "Yvonne", "Zane",
    ]
    cities = ["Paris", "London", "Tokyo", "Berlin", "Madrid"]
    rows = []
    # 50 deterministic but varied rows. id is the rowid alias (auto from 1).
    for i in range(50):
        name = names[i % len(names)]
        # spread ages 18..67
        age = 18 + (i * 17) % 50
        # scores 0.0 .. ~5.0 with one decimal, varied
        score = round(((i * 31) % 51) / 10.0, 1)
        city = cities[i % len(cities)]
        rows.append((name, age, score, city))
    cur.executemany(
        "INSERT INTO people (name, age, score, city) VALUES (?,?,?,?)", rows
    )
    con.commit()
    return con


def cell_repr(v):
    if v is None:
        return NULL_TOKEN
    if isinstance(v, float):
        return repr(v)          # exact round-trip of the stored double
    if isinstance(v, int):
        return str(v)
    return str(v)               # TEXT verbatim


def main():
    con = build()
    cur = con.cursor()
    with open(EXPECT, "w") as f:
        f.write("count %d\n" % len(QUERIES))
        for idx, sql in enumerate(QUERIES):
            result = cur.execute(sql).fetchall()
            ncols = len(cur.description) if cur.description else 0
            f.write("Q %d %s\n" % (idx, sql))
            f.write("C %d\n" % ncols)
            f.write("N %d\n" % len(result))
            for row in result:
                cells = [cell_repr(v) for v in row]
                f.write("R " + US.join(cells) + "\n")
    con.close()

    print("sqlite3 version:", sqlite3.sqlite_version)
    print("built:", DB)
    print("queries:", len(QUERIES), "-> expectations in", EXPECT)
    # human-readable per-query row counts
    con = sqlite3.connect(DB)
    cur = con.cursor()
    for idx, sql in enumerate(QUERIES):
        n = len(cur.execute(sql).fetchall())
        print("  Q%d rows=%d : %s" % (idx, n, sql))
    con.close()


if __name__ == "__main__":
    main()
