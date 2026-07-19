# pdf/metrics.mojo
# Standard-14 AFM glyph metrics (1000-unit em) and text measurement.
#
# The standard-14 PDF fonts have no embedded program; their per-glyph advance
# widths are published in Adobe's AFM (Adobe Font Metrics) files. This module
# hardcodes the canonical ASCII (32..126) widths for Helvetica and Times-Roman,
# and the constant 600 monospace advance for Courier.
#
# All widths are in 1000-unit em (the AFM convention). At a given point size,
# a glyph's advance in points = width_1000 / 1000 * size.


# ---- Helvetica AFM widths, ASCII 32..126 (95 entries) ---------------------
# Canonical Adobe Helvetica.afm advance widths. Index 0 == codepoint 32 (space).
def _helvetica_table() -> List[Int]:
    var t = List[Int]()
    t.append(278)  # 32 space
    t.append(278)  # 33 !
    t.append(355)  # 34 "
    t.append(556)  # 35 #
    t.append(556)  # 36 $
    t.append(889)  # 37 %
    t.append(667)  # 38 &
    t.append(191)  # 39 '
    t.append(333)  # 40 (
    t.append(333)  # 41 )
    t.append(389)  # 42 *
    t.append(584)  # 43 +
    t.append(278)  # 44 ,
    t.append(333)  # 45 -
    t.append(278)  # 46 .
    t.append(278)  # 47 /
    t.append(556)  # 48 0
    t.append(556)  # 49 1
    t.append(556)  # 50 2
    t.append(556)  # 51 3
    t.append(556)  # 52 4
    t.append(556)  # 53 5
    t.append(556)  # 54 6
    t.append(556)  # 55 7
    t.append(556)  # 56 8
    t.append(556)  # 57 9
    t.append(278)  # 58 :
    t.append(278)  # 59 ;
    t.append(584)  # 60 <
    t.append(584)  # 61 =
    t.append(584)  # 62 >
    t.append(556)  # 63 ?
    t.append(1015)  # 64 @
    t.append(667)  # 65 A
    t.append(667)  # 66 B
    t.append(722)  # 67 C
    t.append(722)  # 68 D
    t.append(667)  # 69 E
    t.append(611)  # 70 F
    t.append(778)  # 71 G
    t.append(722)  # 72 H
    t.append(278)  # 73 I
    t.append(500)  # 74 J
    t.append(667)  # 75 K
    t.append(556)  # 76 L
    t.append(833)  # 77 M
    t.append(722)  # 78 N
    t.append(778)  # 79 O
    t.append(667)  # 80 P
    t.append(778)  # 81 Q
    t.append(722)  # 82 R
    t.append(667)  # 83 S
    t.append(611)  # 84 T
    t.append(722)  # 85 U
    t.append(667)  # 86 V
    t.append(944)  # 87 W
    t.append(667)  # 88 X
    t.append(667)  # 89 Y
    t.append(611)  # 90 Z
    t.append(278)  # 91 [
    t.append(278)  # 92 backslash
    t.append(278)  # 93 ]
    t.append(469)  # 94 ^
    t.append(556)  # 95 _
    t.append(333)  # 96 `
    t.append(556)  # 97 a
    t.append(556)  # 98 b
    t.append(500)  # 99 c
    t.append(556)  # 100 d
    t.append(556)  # 101 e
    t.append(278)  # 102 f
    t.append(556)  # 103 g
    t.append(556)  # 104 h
    t.append(222)  # 105 i
    t.append(222)  # 106 j
    t.append(500)  # 107 k
    t.append(222)  # 108 l
    t.append(833)  # 109 m
    t.append(556)  # 110 n
    t.append(556)  # 111 o
    t.append(556)  # 112 p
    t.append(556)  # 113 q
    t.append(333)  # 114 r
    t.append(500)  # 115 s
    t.append(278)  # 116 t
    t.append(556)  # 117 u
    t.append(500)  # 118 v
    t.append(722)  # 119 w
    t.append(500)  # 120 x
    t.append(500)  # 121 y
    t.append(500)  # 122 z
    t.append(334)  # 123 {
    t.append(260)  # 124 |
    t.append(334)  # 125 }
    t.append(584)  # 126 ~
    return t^


# ---- Times-Roman AFM widths, ASCII 32..126 (95 entries) -------------------
# Canonical Adobe Times-Roman.afm advance widths. Index 0 == codepoint 32.
def _times_roman_table() -> List[Int]:
    var t = List[Int]()
    t.append(250)  # 32 space
    t.append(333)  # 33 !
    t.append(408)  # 34 "
    t.append(500)  # 35 #
    t.append(500)  # 36 $
    t.append(833)  # 37 %
    t.append(778)  # 38 &
    t.append(180)  # 39 '
    t.append(333)  # 40 (
    t.append(333)  # 41 )
    t.append(500)  # 42 *
    t.append(564)  # 43 +
    t.append(250)  # 44 ,
    t.append(333)  # 45 -
    t.append(250)  # 46 .
    t.append(278)  # 47 /
    t.append(500)  # 48 0
    t.append(500)  # 49 1
    t.append(500)  # 50 2
    t.append(500)  # 51 3
    t.append(500)  # 52 4
    t.append(500)  # 53 5
    t.append(500)  # 54 6
    t.append(500)  # 55 7
    t.append(500)  # 56 8
    t.append(500)  # 57 9
    t.append(278)  # 58 :
    t.append(278)  # 59 ;
    t.append(564)  # 60 <
    t.append(564)  # 61 =
    t.append(564)  # 62 >
    t.append(444)  # 63 ?
    t.append(921)  # 64 @
    t.append(722)  # 65 A
    t.append(667)  # 66 B
    t.append(667)  # 67 C
    t.append(722)  # 68 D
    t.append(611)  # 69 E
    t.append(556)  # 70 F
    t.append(722)  # 71 G
    t.append(722)  # 72 H
    t.append(333)  # 73 I
    t.append(389)  # 74 J
    t.append(722)  # 75 K
    t.append(611)  # 76 L
    t.append(889)  # 77 M
    t.append(722)  # 78 N
    t.append(722)  # 79 O
    t.append(556)  # 80 P
    t.append(722)  # 81 Q
    t.append(667)  # 82 R
    t.append(556)  # 83 S
    t.append(611)  # 84 T
    t.append(722)  # 85 U
    t.append(722)  # 86 V
    t.append(944)  # 87 W
    t.append(722)  # 88 X
    t.append(722)  # 89 Y
    t.append(611)  # 90 Z
    t.append(333)  # 91 [
    t.append(278)  # 92 backslash
    t.append(333)  # 93 ]
    t.append(469)  # 94 ^
    t.append(500)  # 95 _
    t.append(333)  # 96 `
    t.append(444)  # 97 a
    t.append(500)  # 98 b
    t.append(444)  # 99 c
    t.append(500)  # 100 d
    t.append(444)  # 101 e
    t.append(333)  # 102 f
    t.append(500)  # 103 g
    t.append(500)  # 104 h
    t.append(278)  # 105 i
    t.append(278)  # 106 j
    t.append(500)  # 107 k
    t.append(278)  # 108 l
    t.append(778)  # 109 m
    t.append(500)  # 110 n
    t.append(500)  # 111 o
    t.append(500)  # 112 p
    t.append(500)  # 113 q
    t.append(333)  # 114 r
    t.append(389)  # 115 s
    t.append(278)  # 116 t
    t.append(500)  # 117 u
    t.append(500)  # 118 v
    t.append(722)  # 119 w
    t.append(500)  # 120 x
    t.append(500)  # 121 y
    t.append(444)  # 122 z
    t.append(480)  # 123 {
    t.append(200)  # 124 |
    t.append(480)  # 125 }
    t.append(541)  # 126 ~
    return t^


# Normalize a font name to one of: "helvetica" | "courier" | "times-roman".
# Unknown names fall back to "helvetica".
def _norm_font(font_name: String) raises -> String:
    var lo = font_name.lower()
    if lo == "courier":
        return "courier"
    if lo == "times-roman" or lo == "times" or lo == "timesroman":
        return "times-roman"
    return "helvetica"


# Width of a single codepoint in 1000-unit em for the named font.
# ASCII printable range 32..126 is exact; anything else falls back to ~556.
def char_width_1000(font_name: String, cp: Int) raises -> Int:
    var f = _norm_font(font_name)

    if f == "courier":
        # Monospace: every glyph advances 600.
        return 600

    # Out-of-range codepoints: a reasonable average advance.
    if cp < 32 or cp > 126:
        return 556

    var idx = cp - 32
    if f == "times-roman":
        var tbl = _times_roman_table()
        return tbl[idx]

    # Default: Helvetica.
    var tbl = _helvetica_table()
    return tbl[idx]


# Total advance width of a string in points at the given size.
# width = sum(char_width_1000) / 1000 * size
def text_width(font_name: String, size: Float64, s: String) raises -> Float64:
    var total = 0
    for cp in s.codepoints():
        total += char_width_1000(font_name, Int(cp))
    return Float64(total) / 1000.0 * size
