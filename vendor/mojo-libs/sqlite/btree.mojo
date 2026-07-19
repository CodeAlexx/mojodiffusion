# sqlite/btree.mojo — table b-tree traversal (read-only).
#
# Walks a table b-tree (interior + leaf pages) rooted at a given page and
# returns every row in rowid order. Each leaf cell's payload is decoded via
# format.decode_record; payloads that spill onto overflow pages are reassembled
# by following the overflow page chain first.
#
# Reference: SQLite file format spec, section "B-tree Pages".
#   - page header byte 0: 0x0D table-leaf, 0x05 table-interior
#                         (0x0A/0x02 are index pages; out of scope here)
#   - leaf header: 8 bytes; interior header: 12 bytes (extra 4-byte rightmost
#     child pointer)
#   - cell pointer array: cell_count u16 offsets (relative to page start)
#   - table-leaf cell:     varint payload-len, varint rowid, payload[..]
#   - table-interior cell: u32 left-child page, varint rowid (key)
#
# IMPORTANT: cell-pointer offsets are relative to the PAGE start (byte 0). On
# page 1 the b-tree header itself starts at byte 100 (after the DB header), but
# the cell offsets remain page-relative, so payload reads index into the raw
# page bytes directly.

from sqlite.value import Value
from sqlite.format import read_varint, decode_record
from sqlite.pager import Pager


struct Record(Movable, Copyable):
    var rowid: Int64
    var values: List[Value]

    def __init__(out self, rowid: Int64, var values: List[Value]):
        self.rowid = rowid
        self.values = values^

    def __init__(out self, *, copy: Self):
        self.rowid = copy.rowid
        self.values = copy.values.copy()


def _u16(d: List[UInt8], off: Int) -> Int:
    return (Int(d[off]) << 8) | Int(d[off + 1])


def _u32(d: List[UInt8], off: Int) -> Int:
    return (Int(d[off]) << 24) | (Int(d[off + 1]) << 16) | (Int(d[off + 2]) << 8) | Int(d[off + 3])


def _reassemble_payload(
    pager: Pager,
    page: List[UInt8],
    local_off: Int,
    payload_len: Int,
    page_size: Int,
) raises -> List[UInt8]:
    """Read `payload_len` bytes of record payload starting at `local_off` in
    `page`, following overflow pages when the payload spills.

    SQLite spill formula for a table-leaf cell:
      U = page_size (usable size; assumes no reserved bytes)
      X = U - 35                                  (max local payload)
      M = ((U - 12) * 32 / 255) - 23              (min local payload on spill)
      K = M + ((payload_len - M) % (U - 4))
      local = X                if payload_len <= X
              K                if K <= X
              M                otherwise
    The first `local` bytes live in this page; if local < payload_len, the 4
    bytes immediately after them are the first overflow page number, and each
    overflow page is [u32 next-page][data...]."""
    var U = page_size
    var X = U - 35
    var out = List[UInt8]()

    if payload_len <= X:
        for i in range(payload_len):
            out.append(page[local_off + i])
        return out^

    var M = ((U - 12) * 32 // 255) - 23
    var K = M + ((payload_len - M) % (U - 4))
    var local = K if K <= X else M

    # local bytes from this page
    for i in range(local):
        out.append(page[local_off + i])

    # next 4 bytes = first overflow page number (big-endian)
    var next_page = _u32(page, local_off + local)
    var remaining = payload_len - local

    while remaining > 0 and next_page != 0:
        var ov = pager.read_page(next_page)
        var np = _u32(ov, 0)            # first 4 bytes: next overflow page (0 = last)
        var avail = page_size - 4       # data bytes available on this overflow page
        var take = avail if avail < remaining else remaining
        for i in range(take):
            out.append(ov[4 + i])
        remaining -= take
        next_page = np

    return out^


def _walk_page(
    pager: Pager,
    page_no: Int,
    page_size: Int,
    mut acc: List[Record],
) raises:
    var page = pager.read_page(page_no)
    # On page 1 the b-tree header begins at byte 100; elsewhere at byte 0.
    var hdr = 100 if page_no == 1 else 0
    var ptype = Int(page[hdr])
    var cell_count = _u16(page, hdr + 3)

    if ptype == 0x0D:
        # table leaf: header is 8 bytes; cell pointers follow.
        var ptr_base = hdr + 8
        for c in range(cell_count):
            var cell_off = _u16(page, ptr_base + c * 2)
            var pl = read_varint(page, cell_off)
            var payload_len = Int(pl.value)
            var rv = read_varint(page, cell_off + pl.length)
            var rowid = Int64(rv.value)
            var data_off = cell_off + pl.length + rv.length
            var payload = _reassemble_payload(
                pager, page, data_off, payload_len, page_size
            )
            var vals = decode_record(payload)
            acc.append(Record(rowid, vals^))
    elif ptype == 0x05:
        # table interior: header is 12 bytes; rightmost child at hdr+8.
        var right_child = _u32(page, hdr + 8)
        var ptr_base = hdr + 12
        for c in range(cell_count):
            var cell_off = _u16(page, ptr_base + c * 2)
            var left_child = _u32(page, cell_off)
            # (varint rowid key follows but is not needed for a full walk)
            _walk_page(pager, left_child, page_size, acc)
        _walk_page(pager, right_child, page_size, acc)
    else:
        raise Error(
            "unexpected b-tree page type 0x" + hex(ptype)
            + " on page " + String(page_no) + " (index b-trees not supported)"
        )


def walk_table(pager: Pager, rootpage: Int) raises -> List[Record]:
    """Traverse the table b-tree rooted at `rootpage`, returning all rows in
    rowid order."""
    var acc = List[Record]()
    _walk_page(pager, rootpage, pager.page_size(), acc)
    return acc^
