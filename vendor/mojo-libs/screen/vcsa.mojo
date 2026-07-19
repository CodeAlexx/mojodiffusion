# screen.vcsa — Linux virtual-console TEXT snapshot (/dev/vcsaN), pure Mojo.
#
# This captures the *text* on a Linux virtual console (the tty grid), not pixels
# — the text complement to the framebuffer. /dev/vcsaN format:
#   byte 0: rows         byte 1: cols
#   byte 2: cursor col   byte 3: cursor row
#   then rows*cols cells of 2 bytes each: [char, attribute]
# We keep the char byte and drop the attribute (color/blink). NUL cells render
# as spaces.
#
# NOTE: only meaningful on the Linux text console; inside X/Wayland terminals,
# xterm, tmux, or over SSH there is no vcsa for your session. /dev/vcsaN is
# root:tty 0660 — reading it needs the `tty` group (or root). grab_vcsa() fails
# loud otherwise. N=0 is the *active* console; N>=1 is tty1, tty2, …


def _read_bytes(path: String) raises -> List[UInt8]:
    var f = open(path, "r")
    var s = f.read()
    f.close()
    var bs = s.as_bytes()
    var out = List[UInt8]()
    for i in range(len(bs)):
        out.append(bs[i])
    return out^


@fieldwise_init
struct TextScreen(Copyable, Movable):
    var rows: Int
    var cols: Int
    var cursor_col: Int
    var cursor_row: Int
    var lines: List[String]


def vcsa_decode(raw: List[UInt8]) raises -> TextScreen:
    """Decode a /dev/vcsa buffer into a TextScreen (rows of text)."""
    if len(raw) < 4:
        raise Error("vcsa: buffer too small")
    var rows = Int(raw[0])
    var cols = Int(raw[1])
    var ccol = Int(raw[2])
    var crow = Int(raw[3])
    var lines = List[String]()
    var pos = 4
    for _r in range(rows):
        var line = String("")
        for _c in range(cols):
            if pos + 1 >= len(raw):
                break
            var ch = Int(raw[pos])
            pos += 2  # skip the attribute byte
            if ch == 0:
                ch = ord(" ")
            line += chr(ch)
        lines.append(line)
    return TextScreen(rows, cols, ccol, crow, lines^)


def render_text(ts: TextScreen) -> String:
    var out = String("")
    for i in range(len(ts.lines)):
        out += ts.lines[i] + "\n"
    return out


def grab_vcsa(n: Int = 0) raises -> TextScreen:
    """Snapshot virtual console N (0 = active). Needs read access to /dev/vcsaN."""
    var dev = String("/dev/vcsa") if n == 0 else String("/dev/vcsa" + String(n))
    var raw = _read_bytes(dev)
    return vcsa_decode(raw)
