# sqlite/pager.mojo — page-level reader for a SQLite database file.
#
# A SQLite file is a sequence of fixed-size pages (page_size bytes each). Page
# numbers are 1-based: page N occupies bytes (N-1)*page_size .. N*page_size.
# Page 1 begins with the 100-byte database header. The pager loads the whole
# file into memory once (read-only) and hands out page byte-slices on request.

from sqlite.format import parse_header, DbHeader


struct Pager(Movable):
    var data: List[UInt8]
    var _page_size: Int
    var _page_count: Int

    def __init__(out self, var data: List[UInt8], page_size: Int, page_count: Int):
        self.data = data^
        self._page_size = page_size
        self._page_count = page_count

    @staticmethod
    def open(path: String) raises -> Pager:
        var fh = open(path, "r")
        var raw = fh.read_bytes()
        fh.close()
        var data = List[UInt8]()
        for i in range(len(raw)):
            data.append(raw[i])
        var h = parse_header(data)
        # The header's page_count field can be stale (0) in some files; fall back
        # to deriving it from the file length when the field is unusable.
        var pc = h.page_count
        if pc == 0 and h.page_size > 0:
            pc = len(data) // h.page_size
        return Pager(data^, h.page_size, pc)

    def page_size(self) -> Int:
        return self._page_size

    def page_count(self) -> Int:
        return self._page_count

    def read_page(self, n: Int) raises -> List[UInt8]:
        """Return the bytes of page `n` (1-based)."""
        if n < 1:
            raise Error("page number must be >= 1 (got " + String(n) + ")")
        var start = (n - 1) * self._page_size
        var end = start + self._page_size
        if end > len(self.data):
            raise Error("page " + String(n) + " out of range")
        var out = List[UInt8]()
        for i in range(start, end):
            out.append(self.data[i])
        return out^
