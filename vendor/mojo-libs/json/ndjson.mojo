# json.ndjson — NDJSON / JSON Lines support. 100% Mojo, no FFI.
#
# NDJSON is one JSON value per line, '\n'-separated. This module provides:
#   - parse_ndjson(text)   -> List[JSONValue]   (blank lines skipped)
#   - write_ndjson(values) -> String            (one compact dumps() per line)
#   - NdjsonReader                              (pull one value at a time)
#
# The eager parse_ndjson/write_ndjson buffer the whole result, but NdjsonReader
# scans line-by-line: each call to next_value() parses exactly ONE line into a
# JSONValue, so a huge file can be processed without ever holding more than the
# current line's value at once (the input String itself is the only doc-sized
# buffer — see the honest note in the test report).

from std.memory import UnsafePointer
from json.value import JSONValue
from json.parser import loads
from json.serialize import dumps

comptime BytePtr = UnsafePointer[UInt8, MutExternalOrigin]


def _is_blank(line: String) -> Bool:
    """A line is blank if it is empty or only whitespace."""
    var b = line.as_bytes()
    for i in range(line.byte_length()):
        var c = Int(b[i])
        if c != 0x20 and c != 0x09 and c != 0x0D and c != 0x0A:
            return False
    return True


def parse_ndjson(text: String) raises -> List[JSONValue]:
    """Parse NDJSON text into a list of JSONValues, one per non-blank line.
    Blank / whitespace-only lines are skipped (lenient, per common practice)."""
    var out = List[JSONValue]()
    var lines = text.split("\n")
    for i in range(len(lines)):
        var line = String(lines[i])
        if _is_blank(line):
            continue
        out.append(loads(line))
    return out^


def write_ndjson(values: List[JSONValue]) raises -> String:
    """Serialize a list of JSONValues to NDJSON: one compact dumps() per line,
    '\\n'-separated. No trailing newline is appended (callers that want one can
    add it; round-trip via parse_ndjson tolerates either)."""
    var out = String("")
    for i in range(len(values)):
        if i > 0:
            out += "\n"
        out += dumps(values[i])
    return out^


struct NdjsonReader(Movable):
    """Streaming NDJSON reader. Holds the source buffer and a cursor; each
    next_value() advances past blank lines, slices the next line, and parses just
    that one line into a JSONValue. `has_next()` reports whether more non-blank
    content remains. Bounded extra memory: only the current line + its value."""

    var src: String
    var n: Int
    var pos: Int

    def __init__(out self, var src: String):
        self.src = src^
        self.n = self.src.byte_length()
        self.pos = 0

    fn _p(self) -> BytePtr:
        return BytePtr(unsafe_from_address=Int(self.src.unsafe_ptr()))

    def _skip_blank_lines(mut self):
        # advance pos to the first byte of the next non-blank line (or to EOF).
        var p = self._p()
        while self.pos < self.n:
            # find end of this line
            var ls = self.pos
            var le = self.pos
            while le < self.n and Int(p[le]) != 0x0A:
                le += 1
            # is [ls, le) blank?
            var blank = True
            for i in range(ls, le):
                var c = Int(p[i])
                if c != 0x20 and c != 0x09 and c != 0x0D:
                    blank = False
                    break
            if blank:
                # skip past the newline (or to EOF) and continue
                self.pos = le + 1 if le < self.n else le
            else:
                self.pos = ls
                return

    def has_next(mut self) -> Bool:
        self._skip_blank_lines()
        return self.pos < self.n

    def next_value(mut self) raises -> JSONValue:
        """Parse and return the next non-blank line's JSON value. Raises if there
        is no next value — guard with has_next()."""
        self._skip_blank_lines()
        if self.pos >= self.n:
            raise Error("NdjsonReader: no more values")
        var p = self._p()
        var ls = self.pos
        var le = self.pos
        while le < self.n and Int(p[le]) != 0x0A:
            le += 1
        var line = String(StringSlice(unsafe_from_utf8=Span(unsafe_ptr=p + ls, length=le - ls)))
        self.pos = le + 1 if le < self.n else le
        return loads(line)


def reader(var text: String) raises -> NdjsonReader:
    """Construct a streaming NDJSON reader over `text`."""
    return NdjsonReader(text^)
