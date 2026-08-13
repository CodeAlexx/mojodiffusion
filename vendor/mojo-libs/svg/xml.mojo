# svg/xml.mojo — a minimal XML parser for SVG documents.
#
# Handles: the `<?xml ...?>` prolog, `<!-- comments -->`, `<!DOCTYPE ...>`,
# nested elements, self-closing tags `<tag .../>`, attributes with double or
# single quotes, and a small set of entities. Text content between tags is
# ignored (SVG icons carry geometry in attributes, not text). This is NOT a
# general XML engine — namespaces are flattened (the `svg:` / `xlink:` prefixes
# are kept verbatim in names/attr keys), CDATA and DTD internals are skipped.

from std.collections import Dict


struct XmlNode(Copyable, Movable):
    var name: String
    var attrs: Dict[String, String]
    var children: List[XmlNode]

    def __init__(out self, name: String):
        self.name = name
        self.attrs = Dict[String, String]()
        self.children = List[XmlNode]()

    def __init__(out self, *, copy: Self):
        self.name = copy.name
        self.attrs = Dict[String, String]()
        for ref kv in copy.attrs.items():
            self.attrs[kv.key] = kv.value
        self.children = List[XmlNode]()
        for i in range(len(copy.children)):
            self.children.append(copy.children[i].copy())

    def attr(self, key: String, default: String) raises -> String:
        if key in self.attrs:
            return self.attrs[key]
        return default

    def has(self, key: String) raises -> Bool:
        return key in self.attrs


def _is_ws(c: Int) -> Bool:
    return c == 0x20 or c == 0x09 or c == 0x0A or c == 0x0D


def _is_name_char(c: Int) -> Bool:
    # name chars: letters, digits, '-', '_', '.', ':' (namespace), '%'
    return (
        (c >= 65 and c <= 90) or (c >= 97 and c <= 122) or (c >= 48 and c <= 57)
        or c == 45 or c == 95 or c == 46 or c == 58
    )


def _slice(b: List[UInt8], start: Int, end: Int) -> String:
    if end <= start:
        return String("")
    return String(StringSlice(unsafe_from_utf8=Span(unsafe_ptr=b.unsafe_ptr() + start, length=end - start)))


def _decode_entities(s: String) raises -> String:
    # minimal: &amp; &lt; &gt; &quot; &apos; (icons rarely need more)
    if s.find(String("&")) < 0:
        return s
    var out = s
    out = out.replace(String("&lt;"), String("<"))
    out = out.replace(String("&gt;"), String(">"))
    out = out.replace(String("&quot;"), String("\""))
    out = out.replace(String("&apos;"), String("'"))
    out = out.replace(String("&amp;"), String("&"))
    return out


def _skip_ws(b: List[UInt8], n: Int, mut pos: Int):
    while pos < n and _is_ws(Int(b[pos])):
        pos += 1


def _skip_misc(b: List[UInt8], n: Int, mut pos: Int):
    # skip whitespace, comments, prolog, doctype, processing instructions
    while pos < n:
        _skip_ws(b, n, pos)
        if pos + 3 < n and Int(b[pos]) == 60 and Int(b[pos + 1]) == 33 and Int(b[pos + 2]) == 45 and Int(b[pos + 3]) == 45:
            # <!-- ... -->
            pos += 4
            while pos + 2 < n and not (Int(b[pos]) == 45 and Int(b[pos + 1]) == 45 and Int(b[pos + 2]) == 62):
                pos += 1
            pos += 3
        elif pos + 1 < n and Int(b[pos]) == 60 and Int(b[pos + 1]) == 63:
            # <? ... ?>
            pos += 2
            while pos + 1 < n and not (Int(b[pos]) == 63 and Int(b[pos + 1]) == 62):
                pos += 1
            pos += 2
        elif pos + 1 < n and Int(b[pos]) == 60 and Int(b[pos + 1]) == 33:
            # <! ... > (DOCTYPE) — skip to matching '>', honoring one nesting of [ ]
            pos += 2
            var depth = 0
            while pos < n:
                var c = Int(b[pos])
                if c == 91:
                    depth += 1
                elif c == 93:
                    depth -= 1
                elif c == 62 and depth <= 0:
                    pos += 1
                    break
                pos += 1
        else:
            break


def _parse_attrs(b: List[UInt8], n: Int, mut pos: Int, mut node: XmlNode) raises:
    while True:
        _skip_ws(b, n, pos)
        if pos >= n:
            return
        var c = Int(b[pos])
        if c == 62 or c == 47:   # '>' or '/'
            return
        # attribute name
        var nstart = pos
        while pos < n and _is_name_char(Int(b[pos])):
            pos += 1
        var aname = _slice(b, nstart, pos)
        _skip_ws(b, n, pos)
        if pos < n and Int(b[pos]) == 61:   # '='
            pos += 1
            _skip_ws(b, n, pos)
            var q = Int(b[pos]) if pos < n else 0
            if q == 34 or q == 39:   # " or '
                pos += 1
                var vstart = pos
                while pos < n and Int(b[pos]) != q:
                    pos += 1
                var aval = _slice(b, vstart, pos)
                pos += 1   # closing quote
                node.attrs[aname] = _decode_entities(aval)
            else:
                # unquoted value (rare) — read to whitespace or '>'
                var vstart = pos
                while pos < n and not _is_ws(Int(b[pos])) and Int(b[pos]) != 62:
                    pos += 1
                node.attrs[aname] = _slice(b, vstart, pos)
        else:
            node.attrs[aname] = String("")   # valueless attribute


def _parse_element(b: List[UInt8], n: Int, mut pos: Int) raises -> XmlNode:
    # assumes b[pos] == '<' and not a comment/PI/doctype
    pos += 1   # consume '<'
    var nstart = pos
    while pos < n and _is_name_char(Int(b[pos])):
        pos += 1
    var node = XmlNode(_slice(b, nstart, pos))
    _parse_attrs(b, n, pos, node)
    _skip_ws(b, n, pos)
    if pos < n and Int(b[pos]) == 47:   # '/>' self-closing
        pos += 1
        if pos < n and Int(b[pos]) == 62:
            pos += 1
        return node^
    if pos < n and Int(b[pos]) == 62:   # '>'
        pos += 1

    # children until matching </name>
    while pos < n:
        _skip_misc(b, n, pos)
        if pos >= n:
            break
        if Int(b[pos]) == 60:   # '<'
            if pos + 1 < n and Int(b[pos + 1]) == 47:   # '</' end tag
                pos += 2
                while pos < n and Int(b[pos]) != 62:
                    pos += 1
                if pos < n:
                    pos += 1   # consume '>'
                break
            else:
                var child = _parse_element(b, n, pos)
                node.children.append(child^)
        else:
            # text content — skip to next '<'
            while pos < n and Int(b[pos]) != 60:
                pos += 1
    return node^


def parse_xml(text: String) raises -> XmlNode:
    """Parse an XML/SVG document and return the root element."""
    var src = text.as_bytes()
    var b = List[UInt8]()
    for i in range(len(src)):
        b.append(src[i])
    var n = len(b)
    var pos = 0
    _skip_misc(b, n, pos)
    if pos >= n or Int(b[pos]) != 60:
        raise Error("xml: no root element")
    return _parse_element(b, n, pos)


def find_all(node: XmlNode, name: String, mut out: List[XmlNode]) raises:
    """Collect (copies of) all descendant elements with the given tag name."""
    for i in range(len(node.children)):
        var ch = node.children[i].copy()
        if ch.name == name:
            out.append(ch.copy())
        find_all(ch, name, out)
