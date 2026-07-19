# sqlite/writer.mojo — pure-Mojo SQLite database WRITER.
#
# Emits a SQLite-format-3 database file that the system libsqlite3 (Python
# sqlite3 3.45) opens: `SELECT *` returns the exact rows and
# `PRAGMA integrity_check` returns "ok".
#
# Reference: SQLite file format spec (https://www.sqlite.org/fileformat2.html),
# verified byte-for-byte against bytes the real libsqlite3 accepts.
#
# Layout produced by save():
#   page 1            : 100-byte DB header + sqlite_master table-leaf b-tree.
#   pages 2..         : one root b-tree per user table, in CREATE order.
#                       A table's root is a single leaf (0x0D) when all its
#                       rows fit one page; otherwise it's an interior page
#                       (0x05) whose children are leaf pages (one b-tree level
#                       of interior nodes — see SUPPORTED/UNSUPPORTED below).
#
# Cells grow DOWN from the end of each page; the cell-pointer array grows up
# from just after the page header. Multi-byte integers are BIG-endian.
#
# SUPPORTED:
#   - CREATE TABLE + INSERT for INTEGER / REAL / TEXT / BLOB / NULL values.
#   - sqlite_master with the exact CREATE statement text.
#   - Single-leaf tables AND multi-leaf tables via a one-level interior root
#     (0x05 interior page over 0x0D leaves), so large row counts work.
#
# UNSUPPORTED (documented, fail-loud where it matters):
#   - Overflow pages: any single cell that does not fit in one page raises.
#     Keep individual rows modest (payload < ~page_size-35 bytes).
#   - Only ONE level of interior pages. If the leaf count is so large that the
#     interior page's own cell array overflows a page, save() raises. With
#     page_size 4096 that ceiling is ~500 leaves => hundreds of thousands of
#     rows, which is far past anything the tests need.
#   - No indexes, DELETE/UPDATE, freelist, incremental vacuum, or WAL.

from sqlite.value import Value, VT_NULL, VT_INT, VT_REAL, VT_TEXT, VT_BLOB
from sqlite.format import encode_record, write_varint


comptime PAGE_SIZE: Int = 4096
comptime HEADER_LEN: Int = 100
comptime LEAF_HDR: Int = 8        # table-leaf page header
comptime INTERIOR_HDR: Int = 12   # table-interior page header
comptime SQLITE_VERSION_NUMBER: Int = 3045001


# ─── byte helpers ─────────────────────────────────────────────────────────────
def _put_u16(mut buf: List[UInt8], off: Int, v: Int):
    buf[off] = UInt8((v >> 8) & 0xFF)
    buf[off + 1] = UInt8(v & 0xFF)


def _put_u32(mut buf: List[UInt8], off: Int, v: Int):
    buf[off] = UInt8((v >> 24) & 0xFF)
    buf[off + 1] = UInt8((v >> 16) & 0xFF)
    buf[off + 2] = UInt8((v >> 8) & 0xFF)
    buf[off + 3] = UInt8(v & 0xFF)


# ─── a table-leaf cell: varint(payload_len) varint(rowid) payload ──────────────
def _make_leaf_cell(rowid: Int64, record: List[UInt8]) raises -> List[UInt8]:
    var cell = List[UInt8]()
    var plen = write_varint(UInt64(len(record)))
    for i in range(len(plen)):
        cell.append(plen[i])
    var rv = write_varint(UInt64(rowid))
    for i in range(len(rv)):
        cell.append(rv[i])
    for i in range(len(record)):
        cell.append(record[i])
    return cell^


# ─── a pending row (rowid + already-encoded record body) ───────────────────────
struct _PendingRow(Movable, Copyable):
    var rowid: Int64
    var record: List[UInt8]

    def __init__(out self, rowid: Int64, var record: List[UInt8]):
        self.rowid = rowid
        self.record = record^

    def __init__(out self, *, copy: Self):
        self.rowid = copy.rowid
        self.record = copy.record.copy()


# ─── a table being built ───────────────────────────────────────────────────────
struct _Table(Movable, Copyable):
    var name: String
    var create_sql: String
    var columns: List[String]
    var rows: List[_PendingRow]
    var next_rowid: Int64
    var rootpage: Int   # filled in during save()

    def __init__(out self, name: String, create_sql: String, var columns: List[String]):
        self.name = name
        self.create_sql = create_sql
        self.columns = columns^
        self.rows = List[_PendingRow]()
        self.next_rowid = 1
        self.rootpage = 0

    def __init__(out self, *, copy: Self):
        self.name = copy.name
        self.create_sql = copy.create_sql
        self.columns = copy.columns.copy()
        self.rows = copy.rows.copy()
        self.next_rowid = copy.next_rowid
        self.rootpage = copy.rootpage


struct DbWriter(Movable):
    var tables: List[_Table]

    def __init__(out self):
        self.tables = List[_Table]()

    @staticmethod
    def create() -> DbWriter:
        return DbWriter()

    def _find(self, name: String) raises -> Int:
        for i in range(len(self.tables)):
            if self.tables[i].name == name:
                return i
        raise Error("no such table in writer: " + name)

    def create_table(mut self, name: String, create_sql: String, var columns: List[String]) raises:
        for i in range(len(self.tables)):
            if self.tables[i].name == name:
                raise Error("table already exists: " + name)
        self.tables.append(_Table(name, create_sql, columns^))

    def insert(mut self, table: String, values: List[Value]) raises:
        var idx = self._find(table)
        var rowid = self.tables[idx].next_rowid
        var rec = encode_record(values)
        # A row whose payload cannot fit in a page would need overflow pages,
        # which we do not support — fail loud rather than emit a corrupt file.
        var cell_overhead = len(write_varint(UInt64(len(rec)))) + len(write_varint(UInt64(rowid)))
        if len(rec) + cell_overhead + 2 > PAGE_SIZE - LEAF_HDR:
            raise Error(
                "row too large for one page (overflow unsupported): "
                + String(len(rec)) + " payload bytes"
            )
        self.tables[idx].rows.append(_PendingRow(rowid, rec^))
        self.tables[idx].next_rowid = rowid + 1

    # ─── group a table's rows into leaf pages (each <= one page) ────────────────
    def _pack_leaves(self, ref tbl: _Table) raises -> List[List[_PendingRow]]:
        """Greedily split rows into leaf-sized groups in rowid order. Each group
        must fit: LEAF_HDR + sum(2 + cell_len) <= PAGE_SIZE."""
        var groups = List[List[_PendingRow]]()
        var cur = List[_PendingRow]()
        var used = LEAF_HDR
        for r in range(len(tbl.rows)):
            ref row = tbl.rows[r]
            var cell = _make_leaf_cell(row.rowid, row.record)
            var need = 2 + len(cell)   # cell pointer + cell bytes
            if len(cur) > 0 and used + need > PAGE_SIZE:
                groups.append(cur^)
                cur = List[_PendingRow]()
                used = LEAF_HDR
            cur.append(_PendingRow(row.rowid, row.record.copy()))
            used += need
        if len(cur) > 0:
            groups.append(cur^)
        # Empty table => one empty leaf page.
        if len(groups) == 0:
            groups.append(List[_PendingRow]())
        return groups^

    # ─── write a single table-leaf page into the file buffer ────────────────────
    def _write_leaf_page(self, mut buf: List[UInt8], page_no: Int, ref rows: List[_PendingRow]) raises:
        var page_base = (page_no - 1) * PAGE_SIZE
        # page 1 carries the 100-byte DB header before the b-tree header.
        var hdr = page_base + (HEADER_LEN if page_no == 1 else 0)
        var cell_count = len(rows)

        # Cells fill from the end of the page downward.
        var content_start = PAGE_SIZE  # page-relative offset of first cell byte
        var ptr_base = hdr + LEAF_HDR  # absolute offset of cell-pointer array

        # Lay cells out (rowid order ascending == required b-tree key order).
        for c in range(cell_count):
            ref row = rows[c]
            var cell = _make_leaf_cell(row.rowid, row.record)
            content_start -= len(cell)
            var abs_off = page_base + content_start
            for k in range(len(cell)):
                buf[abs_off + k] = cell[k]
            # cell pointer (page-relative), in cell order.
            _put_u16(buf, ptr_base + c * 2, content_start)

        # Leaf b-tree header (8 bytes).
        buf[hdr] = UInt8(0x0D)               # table-leaf
        _put_u16(buf, hdr + 1, 0)            # first freeblock = 0
        _put_u16(buf, hdr + 3, cell_count)  # cell count
        # cell-content-start: 0 means 65536; for an empty page SQLite uses the
        # usable page size. With no reserved bytes that is PAGE_SIZE.
        var cc = content_start if cell_count > 0 else PAGE_SIZE
        _put_u16(buf, hdr + 5, cc & 0xFFFF)
        buf[hdr + 7] = UInt8(0)             # fragmented free bytes

    # ─── write a table-interior page over a list of (leaf page, max-key) ────────
    def _write_interior_page(
        self,
        mut buf: List[UInt8],
        page_no: Int,
        child_pages: List[Int],
        child_max_keys: List[Int64],
    ) raises:
        # Interior cells: all children except the last become (left_child, key)
        # cells; the last child is the rightmost pointer in the header.
        var n = len(child_pages)
        if n == 0:
            raise Error("interior page needs >= 1 child")
        var right_child = child_pages[n - 1]
        var cell_count = n - 1

        var page_base = (page_no - 1) * PAGE_SIZE
        var hdr = page_base + (HEADER_LEN if page_no == 1 else 0)
        var content_start = PAGE_SIZE
        var ptr_base = hdr + INTERIOR_HDR

        for c in range(cell_count):
            # interior cell = u32 left-child + varint key (largest rowid in that
            # child subtree).
            var cell = List[UInt8]()
            var lc = child_pages[c]
            cell.append(UInt8((lc >> 24) & 0xFF))
            cell.append(UInt8((lc >> 16) & 0xFF))
            cell.append(UInt8((lc >> 8) & 0xFF))
            cell.append(UInt8(lc & 0xFF))
            var key = write_varint(UInt64(child_max_keys[c]))
            for k in range(len(key)):
                cell.append(key[k])
            content_start -= len(cell)
            if content_start < (ptr_base - page_base) + cell_count * 2:
                raise Error(
                    "too many leaves for a single interior page (one interior "
                    "level only): " + String(n) + " children"
                )
            var abs_off = page_base + content_start
            for k in range(len(cell)):
                buf[abs_off + k] = cell[k]
            _put_u16(buf, ptr_base + c * 2, content_start)

        # Interior b-tree header (12 bytes).
        buf[hdr] = UInt8(0x05)              # table-interior
        _put_u16(buf, hdr + 1, 0)           # first freeblock
        _put_u16(buf, hdr + 3, cell_count)  # cell count
        var cc = content_start if cell_count > 0 else PAGE_SIZE
        _put_u16(buf, hdr + 5, cc & 0xFFFF)
        buf[hdr + 7] = UInt8(0)             # fragmented free bytes
        _put_u32(buf, hdr + 8, right_child) # rightmost child pointer

    # ─── header ─────────────────────────────────────────────────────────────────
    def _write_header(self, mut buf: List[UInt8], page_count: Int) raises:
        var magic = String("SQLite format 3")
        var mb = magic.as_bytes()
        for i in range(15):
            buf[i] = mb[i]
        buf[15] = UInt8(0)                  # trailing NUL of the magic
        _put_u16(buf, 16, PAGE_SIZE)        # page size
        buf[18] = UInt8(1)                  # file format write version (legacy)
        buf[19] = UInt8(1)                  # file format read version (legacy)
        buf[20] = UInt8(0)                  # reserved space per page
        buf[21] = UInt8(64)                 # max embedded payload fraction
        buf[22] = UInt8(32)                 # min embedded payload fraction
        buf[23] = UInt8(32)                 # leaf payload fraction
        _put_u32(buf, 24, 1)                # file change counter
        _put_u32(buf, 28, page_count)       # size of db file in pages
        _put_u32(buf, 32, 0)                # first freelist trunk page
        _put_u32(buf, 36, 0)                # total freelist pages
        _put_u32(buf, 40, 1)                # schema cookie
        _put_u32(buf, 44, 4)                # schema format number
        _put_u32(buf, 48, 0)                # default page cache size
        _put_u32(buf, 52, 0)                # largest root b-tree (vacuum)
        _put_u32(buf, 56, 1)                # text encoding = UTF-8
        _put_u32(buf, 60, 0)                # user version
        _put_u32(buf, 64, 0)                # incremental vacuum
        _put_u32(buf, 68, 0)                # application id
        # 72..91 reserved => already zero.
        _put_u32(buf, 92, 1)                # version-valid-for
        _put_u32(buf, 96, SQLITE_VERSION_NUMBER)

    # ─── assemble the whole file ─────────────────────────────────────────────────
    def save(self, path: String) raises:
        # 1) Decide per-table page layout. Page 1 = sqlite_master. Then for each
        #    table, allocate leaf pages and (if >1 leaf) an interior root.
        #
        # We assign page numbers in two passes: first count pages so the
        # sqlite_master rootpage values are known, then write everything.

        var n_tables = len(self.tables)

        # Pack each table's rows into leaf groups (does not depend on numbering).
        var packed = List[List[List[_PendingRow]]]()
        for t in range(n_tables):
            ref tbl = self.tables[t]
            packed.append(self._pack_leaves(tbl))

        # Assign page numbers. Page 1 is sqlite_master. For each table, if it has
        # one leaf, that leaf is the root; else allocate the leaves then an
        # interior root page (root must be a stable page, so we place the
        # interior page right after its leaves and record it as rootpage).
        var next_page = 2
        # rootpage per table, plus the concrete page assignments.
        var roots = List[Int]()
        var table_leaf_pages = List[List[Int]]()   # leaf page numbers per table
        var table_root_is_interior = List[Bool]()
        var table_interior_page = List[Int]()

        for t in range(n_tables):
            ref leaves = packed[t]
            var leaf_pages = List[Int]()
            for _ in range(len(leaves)):
                leaf_pages.append(next_page)
                next_page += 1
            if len(leaves) == 1:
                roots.append(leaf_pages[0])
                table_root_is_interior.append(False)
                table_interior_page.append(0)
            else:
                var interior = next_page
                next_page += 1
                roots.append(interior)
                table_root_is_interior.append(True)
                table_interior_page.append(interior)
            table_leaf_pages.append(leaf_pages^)

        var page_count = next_page - 1

        # 2) Allocate the file buffer (all zero) and write the header.
        var total = page_count * PAGE_SIZE
        var buf = List[UInt8]()
        for _ in range(total):
            buf.append(UInt8(0))
        self._write_header(buf, page_count)

        # 3) Build sqlite_master rows: (type,name,tbl_name,rootpage,sql).
        var master_rows = List[_PendingRow]()
        for t in range(n_tables):
            ref tbl = self.tables[t]
            var vals = List[Value]()
            vals.append(Value.text(String("table")))
            vals.append(Value.text(tbl.name))
            vals.append(Value.text(tbl.name))
            vals.append(Value.integer(Int64(roots[t])))
            vals.append(Value.text(tbl.create_sql))
            var rec = encode_record(vals)
            master_rows.append(_PendingRow(Int64(t + 1), rec^))

        # sqlite_master must fit one leaf page (we do not split the schema).
        var master_used = LEAF_HDR
        for r in range(len(master_rows)):
            ref mr = master_rows[r]
            var cell = _make_leaf_cell(mr.rowid, mr.record)
            master_used += 2 + len(cell)
        if HEADER_LEN + master_used > PAGE_SIZE:
            raise Error("schema too large for one page (sqlite_master split unsupported)")

        self._write_leaf_page(buf, 1, master_rows)

        # 4) Write each table's pages.
        for t in range(n_tables):
            ref leaves = packed[t]
            ref leaf_pages = table_leaf_pages[t]
            var child_pages = List[Int]()
            var child_max_keys = List[Int64]()
            for li in range(len(leaves)):
                ref group = leaves[li]
                var pno = leaf_pages[li]
                self._write_leaf_page(buf, pno, group)
                child_pages.append(pno)
                if len(group) > 0:
                    child_max_keys.append(group[len(group) - 1].rowid)
                else:
                    child_max_keys.append(Int64(0))
            if table_root_is_interior[t]:
                self._write_interior_page(
                    buf, table_interior_page[t], child_pages, child_max_keys
                )

        # 5) Flush the whole file.
        with open(path, "w") as f:
            f.write_bytes(Span(buf))
