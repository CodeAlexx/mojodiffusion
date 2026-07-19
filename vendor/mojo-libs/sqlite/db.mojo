# sqlite/db.mojo — table-level read API over the pager + b-tree layers.
#
# Opens a SQLite database, reads its schema from sqlite_master (the table
# b-tree rooted at page 1), and lets callers list tables, read columns, and
# read whole tables as rows of Value.
#
# sqlite_master rows are 5-column records:
#   type(text) name(text) tbl_name(text) rootpage(int) sql(text)
# A row with type=="table" describes a user table; its rootpage is the b-tree
# root and its sql is the CREATE TABLE statement (parsed for column names).
#
# INTEGER PRIMARY KEY "rowid alias": when a column is declared
# `<col> INTEGER PRIMARY KEY`, that column is NOT stored in the record (it
# decodes to NULL); its value IS the row's rowid. read_table() detects this
# column from the CREATE sql and substitutes the rowid for that column so the
# returned row matches what `SELECT *` reports. See _rowid_alias_index.

from sqlite.value import Value
from sqlite.format import decode_record
from sqlite.pager import Pager
from sqlite.btree import walk_table, Record


struct Row(Movable, Copyable):
    var values: List[Value]

    def __init__(out self, var values: List[Value]):
        self.values = values^

    def __init__(out self, *, copy: Self):
        self.values = copy.values.copy()


struct TableInfo(Movable, Copyable):
    var name: String
    var rootpage: Int
    var sql: String

    def __init__(out self, name: String, rootpage: Int, sql: String):
        self.name = name
        self.rootpage = rootpage
        self.sql = sql


struct Database(Movable):
    var pager: Pager
    var tables: List[TableInfo]

    def __init__(out self, var pager: Pager, var tables: List[TableInfo]):
        self.pager = pager^
        self.tables = tables^

    @staticmethod
    def open(path: String) raises -> Database:
        var pager = Pager.open(path)
        # sqlite_master lives in the table b-tree rooted at page 1.
        var schema_rows = walk_table(pager, 1)
        var tables = List[TableInfo]()
        for r in range(len(schema_rows)):
            ref rec = schema_rows[r]
            if len(rec.values) < 5:
                continue
            var typ = rec.values[0].as_text()
            if typ != "table":
                continue
            var name = rec.values[1].as_text()
            var rootpage = Int(rec.values[3].as_int())
            var sql = rec.values[4].as_text()
            tables.append(TableInfo(name, rootpage, sql))
        return Database(pager^, tables^)

    def _find(self, table: String) raises -> Int:
        for i in range(len(self.tables)):
            if self.tables[i].name == table:
                return i
        raise Error("no such table: " + table)

    def table_names(self) -> List[String]:
        var out = List[String]()
        for i in range(len(self.tables)):
            out.append(self.tables[i].name)
        return out^

    def schema_sql(self, table: String) raises -> String:
        return self.tables[self._find(table)].sql

    def columns(self, table: String) raises -> List[String]:
        return _parse_columns(self.tables[self._find(table)].sql)

    def read_table(self, table: String) raises -> List[Row]:
        var idx = self._find(table)
        var sql = self.tables[idx].sql
        var rootpage = self.tables[idx].rootpage
        var rowid_col = _rowid_alias_index(sql)
        var recs = walk_table(self.pager, rootpage)
        var rows = List[Row]()
        for r in range(len(recs)):
            ref rec = recs[r]
            var vals = List[Value]()
            for c in range(len(rec.values)):
                vals.append(rec.values[c].copy())
            # Surface the rowid for an INTEGER PRIMARY KEY rowid-alias column,
            # which is stored as NULL in the record.
            if rowid_col >= 0 and rowid_col < len(vals):
                vals[rowid_col] = Value.integer(rec.rowid)
            rows.append(Row(vals^))
        return rows^


# ─── CREATE TABLE parsing (byte-level; CREATE statements are ASCII) ──────────
comptime B_LPAREN = UInt8(40)   # (
comptime B_RPAREN = UInt8(41)   # )
comptime B_COMMA  = UInt8(44)   # ,
comptime B_SPACE  = UInt8(32)
comptime B_TAB    = UInt8(9)
comptime B_NL     = UInt8(10)
comptime B_CR     = UInt8(13)
comptime B_DQUOTE = UInt8(34)   # "
comptime B_BTICK  = UInt8(96)   # `
comptime B_LBRACK = UInt8(91)   # [
comptime B_RBRACK = UInt8(93)   # ]


def _bytes_of(s: String) raises -> List[UInt8]:
    var sb = s.as_bytes()
    var out = List[UInt8]()
    for i in range(s.byte_length()):
        out.append(sb[i])
    return out^


def _is_ws(b: UInt8) -> Bool:
    return b == B_SPACE or b == B_TAB or b == B_NL or b == B_CR


def _to_upper(b: UInt8) -> UInt8:
    if b >= UInt8(97) and b <= UInt8(122):  # a..z
        return b - UInt8(32)
    return b


def _str_from_bytes(b: List[UInt8], start: Int, end: Int) raises -> String:
    var s = String("")
    for i in range(start, end):
        s += chr(Int(b[i]))
    return s^


def _column_defs(sql: String) raises -> List[List[UInt8]]:
    """Split the parenthesised CREATE TABLE body into comma-separated
    definitions (column or table-constraint), respecting nested parens.
    Returns each definition as raw bytes."""
    var s = _bytes_of(sql)
    var n = len(s)
    var defs = List[List[UInt8]]()
    var start = -1
    for i in range(n):
        if s[i] == B_LPAREN:
            start = i + 1
            break
    if start < 0:
        return defs^
    var depth = 0
    var cur = List[UInt8]()
    var i = start
    while i < n:
        var ch = s[i]
        if ch == B_LPAREN:
            depth += 1
            cur.append(ch)
        elif ch == B_RPAREN:
            if depth == 0:
                if len(cur) > 0:
                    defs.append(cur^)
                return defs^
            depth -= 1
            cur.append(ch)
        elif ch == B_COMMA and depth == 0:
            if len(cur) > 0:
                defs.append(cur^)
            cur = List[UInt8]()
        else:
            cur.append(ch)
        i += 1
    if len(cur) > 0:
        defs.append(cur^)
    return defs^


def _first_token(defn: List[UInt8]) raises -> String:
    """First identifier token of a column definition. Handles "x" / `x` / [x]
    and bare identifiers."""
    var n = len(defn)
    var i = 0
    while i < n and _is_ws(defn[i]):
        i += 1
    if i >= n:
        return String("")
    var q = defn[i]
    if q == B_DQUOTE or q == B_BTICK:
        var j = i + 1
        while j < n and defn[j] != q:
            j += 1
        return _str_from_bytes(defn, i + 1, j)
    if q == B_LBRACK:
        var j = i + 1
        while j < n and defn[j] != B_RBRACK:
            j += 1
        return _str_from_bytes(defn, i + 1, j)
    var j = i
    while j < n and not _is_ws(defn[j]):
        j += 1
    return _str_from_bytes(defn, i, j)


def _is_constraint(defn: List[UInt8]) raises -> Bool:
    """Table-level constraints are not columns."""
    var n = len(defn)
    var i = 0
    while i < n and _is_ws(defn[i]):
        i += 1
    var kw = String("")
    while i < n and not _is_ws(defn[i]) and defn[i] != B_LPAREN:
        kw += chr(Int(_to_upper(defn[i])))
        i += 1
    return (
        kw == "PRIMARY" or kw == "UNIQUE" or kw == "CHECK"
        or kw == "FOREIGN" or kw == "CONSTRAINT"
    )


def _parse_columns(sql: String) raises -> List[String]:
    var defs = _column_defs(sql)
    var cols = List[String]()
    for i in range(len(defs)):
        ref d = defs[i]
        if _is_constraint(d):
            continue
        cols.append(_first_token(d))
    return cols^


def _upper_str(defn: List[UInt8]) raises -> String:
    var s = String("")
    for i in range(len(defn)):
        s += chr(Int(_to_upper(defn[i])))
    return s^


def _rowid_alias_index(sql: String) raises -> Int:
    """Column index of an INTEGER PRIMARY KEY rowid-alias column, or -1.
    Such a column is stored as NULL in the record; its value is the rowid."""
    var defs = _column_defs(sql)
    var col_index = -1
    for i in range(len(defs)):
        ref d = defs[i]
        if _is_constraint(d):
            continue
        col_index += 1
        var up = _upper_str(d)
        if _contains(up, "INTEGER") and _contains(up, "PRIMARY") and _contains(up, "KEY"):
            return col_index
    return -1


def _contains(hay: String, needle: String) raises -> Bool:
    var hb = hay.as_bytes()
    var nb = needle.as_bytes()
    var h = hay.byte_length()
    var n = needle.byte_length()
    if n == 0:
        return True
    if n > h:
        return False
    for i in range(h - n + 1):
        var ok = True
        for j in range(n):
            if hb[i + j] != nb[j]:
                ok = False
                break
        if ok:
            return True
    return False
