# sqlite/query.mojo — a read-only SELECT executor over the table reader.
#
# execute_select(db, sql) parses the SQL (sql.mojo), resolves the table and its
# columns from `db`, reads all rows, then applies WHERE → ORDER BY → LIMIT →
# projection, returning a ResultSet of selected columns and matching rows.
#
# Comparison semantics (SQLite-ish, simplified):
#   - INT/REAL operands compare numerically (mixed int/real promoted to real)
#   - TEXT operands compare lexicographically (byte order, ASCII)
#   - any NULL involved in a comparison yields false (row excluded)
#   - col-vs-literal and col-vs-col are both supported

from sqlite.value import Value, VT_NULL, VT_INT, VT_REAL, VT_TEXT, VT_BLOB
from sqlite.db import Database, Row
from sqlite.sql import (
    SelectStmt,
    WhereExpr,
    WhereNode,
    parse_select,
    EX_AND,
    EX_OR,
    EX_CMP,
    OPND_COL,
    OPND_INT,
    OPND_REAL,
    OPND_TEXT,
    OPND_NULL,
)


struct ResultSet(Movable, Copyable):
    var columns: List[String]
    var rows: List[List[Value]]

    def __init__(out self):
        self.columns = List[String]()
        self.rows = List[List[Value]]()

    def __init__(out self, var columns: List[String], var rows: List[List[Value]]):
        self.columns = columns^
        self.rows = rows^

    def __init__(out self, *, copy: Self):
        self.columns = copy.columns.copy()
        var rs = List[List[Value]]()
        for i in range(len(copy.rows)):
            var r = List[Value]()
            for j in range(len(copy.rows[i])):
                r.append(copy.rows[i][j].copy())
            rs.append(r^)
        self.rows = rs^


# ─── REAL column affinity ────────────────────────────────────────────────────
# SQLite applies column affinity on read: a column declared with a REAL-affinity
# type ("REAL"/"FLOA"/"DOUB") stores whole-number values as integers but reads
# them back as floating point. The table reader surfaces the raw stored Value
# (VT_INT for those), so we coerce them to VT_REAL to match sqlite3's output.
#
# We parse declared types directly from the CREATE TABLE sql via the same
# comma/paren split db.mojo uses, but we only need a per-schema-column flag.
comptime B_LP = UInt8(40)    # (
comptime B_RP = UInt8(41)    # )
comptime B_CM = UInt8(44)    # ,
comptime B_WS_SP = UInt8(32)
comptime B_WS_TAB = UInt8(9)
comptime B_WS_NL = UInt8(10)
comptime B_WS_CR = UInt8(13)
comptime B_DQ = UInt8(34)    # "
comptime B_BT = UInt8(96)    # `
comptime B_LB = UInt8(91)    # [
comptime B_RB = UInt8(93)    # ]


def _aff_ws(b: UInt8) -> Bool:
    return b == B_WS_SP or b == B_WS_TAB or b == B_WS_NL or b == B_WS_CR


def _aff_up(b: UInt8) -> UInt8:
    if b >= UInt8(97) and b <= UInt8(122):
        return b - UInt8(32)
    return b


def _contains_ci(hay: String, needle: String) raises -> Bool:
    var hb = hay.as_bytes()
    var nb = needle.as_bytes()
    var h = hay.byte_length()
    var n = needle.byte_length()
    if n == 0 or n > h:
        return n == 0
    for i in range(h - n + 1):
        var ok = True
        for j in range(n):
            if _aff_up(hb[i + j]) != nb[j]:
                ok = False
                break
        if ok:
            return True
    return False


# Returns a bool per schema column: True if the column has REAL affinity.
def _real_affinity(schema_sql: String) raises -> List[Bool]:
    var sb = schema_sql.as_bytes()
    var n = schema_sql.byte_length()
    var flags = List[Bool]()
    # find first '('
    var start = -1
    for i in range(n):
        if sb[i] == B_LP:
            start = i + 1
            break
    if start < 0:
        return flags^
    # walk top-level comma-separated definitions
    var depth = 0
    var cur = String("")
    var i = start
    var defs = List[String]()
    while i < n:
        var ch = sb[i]
        if ch == B_LP:
            depth += 1
            cur += chr(Int(ch))
        elif ch == B_RP:
            if depth == 0:
                if cur.byte_length() > 0:
                    defs.append(cur^)
                break
            depth -= 1
            cur += chr(Int(ch))
        elif ch == B_CM and depth == 0:
            defs.append(cur^)
            cur = String("")
        else:
            cur += chr(Int(ch))
        i += 1
    for d in range(len(defs)):
        ref defn = defs[d]
        # skip leading ws, read first token (the column name) to detect a
        # table-level constraint; if the first token is a constraint keyword,
        # it is not a column.
        var db = defn.as_bytes()
        var m = defn.byte_length()
        var k = 0
        while k < m and _aff_ws(db[k]):
            k += 1
        # first token (handle quoting just enough to skip it)
        var first = String("")
        if k < m and (db[k] == B_DQ or db[k] == B_BT or db[k] == B_LB):
            var close = db[k]
            if db[k] == B_LB:
                close = B_RB
            k += 1
            while k < m and db[k] != close:
                k += 1
            k += 1
        else:
            while k < m and not _aff_ws(db[k]):
                first += chr(Int(_aff_up(db[k])))
                k += 1
        if (
            first == "PRIMARY" or first == "UNIQUE" or first == "CHECK"
            or first == "FOREIGN" or first == "CONSTRAINT"
        ):
            continue
        # column: REAL affinity if declared type contains REAL/FLOA/DOUB
        var is_real = (
            _contains_ci(defn, "REAL")
            or _contains_ci(defn, "FLOA")
            or _contains_ci(defn, "DOUB")
        )
        flags.append(is_real)
    return flags^


def _col_index(cols: List[String], name: String) raises -> Int:
    for i in range(len(cols)):
        if cols[i] == name:
            return i
    raise Error("no such column: " + name)


# ─── byte-wise text comparison: -1 / 0 / 1 ──────────────────────────────────
def _text_cmp(a: String, b: String) raises -> Int:
    var ab = a.as_bytes()
    var bb = b.as_bytes()
    var an = a.byte_length()
    var bn = b.byte_length()
    var m = an
    if bn < m:
        m = bn
    for i in range(m):
        if ab[i] < bb[i]:
            return -1
        if ab[i] > bb[i]:
            return 1
    if an < bn:
        return -1
    if an > bn:
        return 1
    return 0


# ─── apply a canonical operator to an ordering result (-1/0/1) ───────────────
def _apply_op(op: String, c: Int) -> Bool:
    if op == "=":
        return c == 0
    if op == "!=":
        return c != 0
    if op == "<":
        return c < 0
    if op == "<=":
        return c <= 0
    if op == ">":
        return c > 0
    if op == ">=":
        return c >= 0
    return False


# Numeric comparator: returns -1/0/1.
def _num_cmp(a: Float64, b: Float64) -> Int:
    if a < b:
        return -1
    if a > b:
        return 1
    return 0


# Evaluate one comparison node against a row. NULL on either side → false.
def _eval_cmp(node: WhereNode, row: List[Value], cols: List[String]) raises -> Bool:
    var li = _col_index(cols, node.col)
    ref lv = row[li]
    if lv.is_null():
        return False

    # Build the right-hand value's "type intent".
    if node.rhs_kind == OPND_NULL:
        return False  # NULL comparison always false here

    if node.rhs_kind == OPND_COL:
        var ri = _col_index(cols, node.rhs_col)
        ref rv = row[ri]
        if rv.is_null():
            return False
        # both columns present: choose numeric if both numeric, else text
        var l_num = lv.kind == VT_INT or lv.kind == VT_REAL
        var r_num = rv.kind == VT_INT or rv.kind == VT_REAL
        if l_num and r_num:
            return _apply_op(node.op, _num_cmp(lv.as_real(), rv.as_real()))
        if lv.kind == VT_TEXT and rv.kind == VT_TEXT:
            return _apply_op(node.op, _text_cmp(lv.as_text(), rv.as_text()))
        # mixed/blob: SQLite orders by type class; treat as not-equal/false-ish.
        # numeric < text in SQLite type ordering.
        if l_num and rv.kind == VT_TEXT:
            return _apply_op(node.op, -1)
        if lv.kind == VT_TEXT and r_num:
            return _apply_op(node.op, 1)
        return False

    # literal RHS
    if node.rhs_kind == OPND_INT:
        if lv.kind == VT_INT or lv.kind == VT_REAL:
            return _apply_op(node.op, _num_cmp(lv.as_real(), Float64(node.rhs_ival)))
        return False
    if node.rhs_kind == OPND_REAL:
        if lv.kind == VT_INT or lv.kind == VT_REAL:
            return _apply_op(node.op, _num_cmp(lv.as_real(), node.rhs_rval))
        return False
    if node.rhs_kind == OPND_TEXT:
        if lv.kind == VT_TEXT:
            return _apply_op(node.op, _text_cmp(lv.as_text(), node.rhs_text))
        return False
    return False


def _eval(where: WhereExpr, idx: Int, row: List[Value], cols: List[String]) raises -> Bool:
    ref node = where.nodes[idx]
    if node.kind == EX_CMP:
        return _eval_cmp(node, row, cols)
    if node.kind == EX_AND:
        return _eval(where, node.left, row, cols) and _eval(where, node.right, row, cols)
    if node.kind == EX_OR:
        return _eval(where, node.left, row, cols) or _eval(where, node.right, row, cols)
    return False


# ─── ORDER BY: stable insertion sort on a key column ─────────────────────────
# A NULL key sorts before everything (SQLite ascending: NULLs first).
# Returns -1/0/1 ordering of row a vs b on the key column, numeric or text.
def _key_cmp(a: List[Value], b: List[Value], ki: Int) raises -> Int:
    ref av = a[ki]
    ref bv = b[ki]
    if av.is_null() and bv.is_null():
        return 0
    if av.is_null():
        return -1
    if bv.is_null():
        return 1
    var a_num = av.kind == VT_INT or av.kind == VT_REAL
    var b_num = bv.kind == VT_INT or bv.kind == VT_REAL
    if a_num and b_num:
        return _num_cmp(av.as_real(), bv.as_real())
    if av.kind == VT_TEXT and bv.kind == VT_TEXT:
        return _text_cmp(av.as_text(), bv.as_text())
    # type-class ordering: NULL < numeric < text < blob (numerics handled above)
    if a_num and bv.kind == VT_TEXT:
        return -1
    if av.kind == VT_TEXT and b_num:
        return 1
    return 0


# Stable insertion sort on a permutation of row indices, returning the order.
def _stable_order(rows: List[List[Value]], ki: Int, desc: Bool) raises -> List[Int]:
    var order = List[Int]()
    var n = len(rows)
    for i in range(n):
        order.append(i)
    # insertion sort over `order`; stable since equal keys never swap.
    for i in range(1, n):
        var j = i
        while j > 0:
            var c = _key_cmp(rows[order[j - 1]], rows[order[j]], ki)
            var swap = (c > 0) if not desc else (c < 0)
            if not swap:
                break
            var tmp = order[j - 1]
            order[j - 1] = order[j]
            order[j] = tmp
            j -= 1
    return order^


def execute_select(db: Database, sql: String) raises -> ResultSet:
    var stmt = parse_select(sql)
    var schema_cols = db.columns(stmt.table)
    var src = db.read_table(stmt.table)

    # column affinity: REAL-affinity columns surface stored integers as reals
    var real_aff = _real_affinity(db.schema_sql(stmt.table))

    # 1. WHERE filter → list of surviving rows (as List[Value]), with affinity
    #    coercion applied so values match sqlite3's reported types.
    var filtered = List[List[Value]]()
    for ri in range(len(src)):
        ref raw = src[ri].values
        var rv = List[Value]()
        for ci in range(len(raw)):
            var v = raw[ci].copy()
            if ci < len(real_aff) and real_aff[ci] and v.kind == VT_INT:
                v = Value.real(Float64(v.as_int()))
            rv.append(v^)
        var keep = True
        if stmt.has_where:
            keep = _eval(stmt.where, stmt.where.root, rv, schema_cols)
        if keep:
            filtered.append(rv^)

    # 2. ORDER BY (stable) on a resolved key column → produces a row ordering
    var order = List[Int]()
    if stmt.order_by.byte_length() > 0:
        var ki = _col_index(schema_cols, stmt.order_by)
        order = _stable_order(filtered, ki, stmt.order_desc)
    else:
        for i in range(len(filtered)):
            order.append(i)

    # 3. LIMIT
    var limit = len(order)
    if stmt.limit >= 0 and stmt.limit < limit:
        limit = stmt.limit

    # 4. Projection — resolve output columns + indices
    var out_cols = List[String]()
    var out_idx = List[Int]()
    if stmt.star:
        for i in range(len(schema_cols)):
            out_cols.append(schema_cols[i])
            out_idx.append(i)
    else:
        for i in range(len(stmt.columns)):
            var ix = _col_index(schema_cols, stmt.columns[i])
            out_cols.append(stmt.columns[i])
            out_idx.append(ix)

    var out_rows = List[List[Value]]()
    for r in range(limit):
        ref full = filtered[order[r]]
        var proj = List[Value]()
        for k in range(len(out_idx)):
            proj.append(full[out_idx[k]].copy())
        out_rows.append(proj^)

    return ResultSet(out_cols^, out_rows^)
