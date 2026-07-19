# tables.mojo — WORKSTREAM: immediate-mode GUI TABLES for Mojo 1.0.0b1, drawn on the MojoGUI C rendering
# backend (librender.so) via external_call ONLY. No backend change.
#
# A Table struct + the canonical immediate-mode GUI table call sequence:
#   begin_table(id,x,y,w,h,ncols)
#   table_setup_column(name, weight_or_fixed_px, is_fixed)   # declare each column
#   table_headers_row()                                      # draw the header bar w/ labels
#   table_next_row()                                         # advance to a new data row
#   table_next_column() -> Tuple[Float32,Float32]            # advance one cell, return its (x,y)
#   table_cell_rect(col,row) -> Tuple[Float32,Float32,Float32,Float32]  # (x,y,w,h) of any cell
#   end_table()
#
# COLUMN WIDTH MODEL (immediate-mode GUI-faithful):
#   Each column is either FIXED (a literal pixel width) or STRETCH (a weight). Fixed widths are
#   summed and subtracted from the table's content width; the REMAINING width is divided among the
#   stretch columns in proportion to their weights (weights summed, each stretch col gets
#   remaining * weight/total_weight). x-positions are the cumulative prefix sum of resolved widths,
#   measured from the table's left edge (x). This is computed ONCE in table_headers_row()/the first
#   layout call and cached in `col_x` / `col_w`.
#
# VISUALS:
#   - header bar: a distinct (brighter) fill across the table width + each column label, with a
#     vertical 1px separator between header cells.
#   - data rows: ALTERNATING row background (even/odd shades).
#   - 1px borders BETWEEN cells (vertical separators at each column boundary) + a row underline.
#   - each cell's CONTENT is clipped with set_clip_rect(...) before the caller draws, and the table
#     clears the clip with clear_clip_rect() in table_next_column (before returning) / end_table.
#
# FFI idioms mirrored EXACTLY from the framebuffer-verified exemplars
#   /home/alex/framepfx-mojo/src/keyframe_editor.mojo (to_cstr 37-43; col/fill/rect/line_t/txt
#   46-63) and /tmp/mojoui/core_widgets.mojo + /tmp/mojoui2/core_layout.mojo (UiContext concept,
#   measure_text two-out-ptr, point_in_rect/clampf, owned-List .copy()/^ rules).
#
# set_clip_rect / clear_clip_rect signatures are exactly as documented:
#   set_clip_rect(int x, int y, int w, int h)   (top-left rect)
#   clear_clip_rect()
#
# NO `from math` import: the windowed FFI binary cannot link transcendentals (sincos@GLIBC). Only
# builtin min/max/abs/Int()/Float32() arithmetic is used (same constraint as the exemplars). The
# stretch-width division is plain Float32 arithmetic; no trig anywhere.
#
# INJECTABLE INPUT: this module is layout/draw only and has NO mouse-gated logic path, so there is
# nothing to inject (the self-test drives pure geometry). A Table carries no get_mouse_* calls.

from std.ffi import external_call
from std.memory import UnsafePointer, alloc
from builtin.type_aliases import MutExternalOrigin


# ============================================================================
# C backend helpers  (mirror keyframe_editor.mojo:37-63 / core_widgets.mojo:32-82 EXACTLY)
# ============================================================================
def to_cstr(s: String) -> UnsafePointer[Int8, MutExternalOrigin]:
    var b = s.as_bytes()
    var buf = alloc[Int8](len(b) + 1)
    for i in range(len(b)):
        buf[i] = Int8(b[i])
    buf[len(b)] = 0
    return buf


def set_color(r: Float32, g: Float32, b: Float32, a: Float32):
    _ = external_call["set_color", Int32](r, g, b, a)


def col(r: Int, g: Int, b: Int):
    # 0..255 convenience wrapper (keyframe_editor.mojo:46-47)
    _ = external_call["set_color", Int32](
        Float32(r) / 255.0, Float32(g) / 255.0, Float32(b) / 255.0, Float32(1.0)
    )


def fill(x: Float32, y: Float32, w: Float32, h: Float32):
    _ = external_call["draw_filled_rectangle", Int32](x, y, w, h)


def rect(x: Float32, y: Float32, w: Float32, h: Float32):
    _ = external_call["draw_rectangle", Int32](x, y, w, h)


def line_t(x1: Float32, y1: Float32, x2: Float32, y2: Float32, thick: Float32):
    _ = external_call["draw_line", Int32](x1, y1, x2, y2, thick)


def txt(s: String, x: Float32, y: Float32, size: Float32):
    _ = external_call["draw_text", Int32](to_cstr(s), x, y, size)


# get_text_size(const char*, float size, float* outW, float* outH) -> (w, h) in px.
def measure_text(s: String, size: Float32) -> Tuple[Float32, Float32]:
    var wbuf = alloc[Float32](1)
    var hbuf = alloc[Float32](1)
    wbuf[0] = Float32(0.0)
    hbuf[0] = Float32(0.0)
    _ = external_call["get_text_size", Int32](to_cstr(s), size, wbuf, hbuf)
    var w = wbuf[0]
    var h = hbuf[0]
    return Tuple[Float32, Float32](w, h)


# ---- clipping primitives (documented backend signatures) ----
# set_clip_rect(int x, int y, int w, int h) — top-left rect. clear_clip_rect() — pop.
def set_clip_rect_px(x: Float32, y: Float32, w: Float32, h: Float32):
    _ = external_call["set_clip_rect", Int32](Int32(x), Int32(y), Int32(w), Int32(h))


def clear_clip_rect():
    _ = external_call["clear_clip_rect", Int32]()


# ============================================================================
# geometry helpers (mirror core_widgets.mojo:88-97)
# ============================================================================
def point_in_rect(px: Float32, py: Float32, x: Float32, y: Float32, w: Float32, h: Float32) -> Bool:
    return px >= x and px < x + w and py >= y and py < y + h


def clampf(v: Float32, lo: Float32, hi: Float32) -> Float32:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


# ============================================================================
# Column spec — one declared column. ImplicitlyCopyable POD so it lives in a List cheaply.
#   name     : the header label.
#   spec     : if is_fixed, the literal pixel width; else the stretch WEIGHT.
#   is_fixed : True => fixed-px column; False => stretch (weight) column.
# ============================================================================
@fieldwise_init
struct Column(ImplicitlyCopyable, Movable):
    var name: String
    var spec: Float32
    var is_fixed: Bool

    # explicit copy (mirrors the verified Pin struct in node_graph.mojo:77-84 which is ALSO
    # @fieldwise_init + ImplicitlyCopyable + a hand-written copy and holds a String field). Used by
    # .copy()-on-read of List[Column]. `def` not `fn` per the live mojo-syntax skill (`fn` deprecated).
    def copy(self) -> Self:
        return Column(self.name, self.spec, self.is_fixed)


# ============================================================================
# Table — carries the table geometry, the declared columns, the resolved per-column x/width
# cache, and the live cursor (current row / current column).
#
# Owned type (holds List fields), so reads of its List fields use .copy() and write-backs use ^
# (Mojo 1.0.0b1 owned-type rule). It issues NO get_mouse_* calls — pure layout/draw.
#
# Cursor model (immediate-mode GUI semantics):
#   cur_row    : index of the row currently being filled (-1 before the first table_next_row;
#                the header occupies the band ABOVE row 0 and is NOT a data row).
#   cur_col    : index of the column the next table_next_column will RETURN (advances 0,1,2,...,
#                wrapping is the caller's job via table_next_row). Starts at -1 each row so the
#                first table_next_column yields column 0.
#   resolved   : True once column widths have been computed (lazily on first need).
# ============================================================================
struct Table(Copyable, Movable):
    var id: Int
    var x: Float32
    var y: Float32
    var w: Float32
    var h: Float32
    var ncols: Int

    var columns: List[Column]      # declared columns (table_setup_column appends here)
    var col_x: List[Float32]       # resolved left x of each column (prefix sum), relative-to-screen
    var col_w: List[Float32]       # resolved width of each column
    var resolved: Bool

    var row_h: Float32             # height of one row (header + data rows share this)
    var header_drawn: Bool         # has table_headers_row run (header occupies the top band)
    var cur_row: Int               # current data-row index (-1 before first table_next_row)
    var cur_col: Int               # current column index within the row (-1 at row start)

    # ---- begin_table is modelled as the constructor (named begin_table free fn calls it) ----
    def __init__(out self, id: Int, x: Float32, y: Float32, w: Float32, h: Float32, ncols: Int):
        self.id = id
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.ncols = ncols
        self.columns = List[Column]()
        self.col_x = List[Float32]()
        self.col_w = List[Float32]()
        self.resolved = False
        self.row_h = Float32(22.0)     # default immediate-mode GUI-ish row height
        self.header_drawn = False
        self.cur_row = -1
        self.cur_col = -1

    # ------------------------------------------------------------------
    # table_setup_column(name, spec, is_fixed): declare one column. Call ncols times after
    #   begin_table and BEFORE table_headers_row / the first row. `spec` is a fixed px width when
    #   is_fixed, else a stretch weight. Re-declaring is not supported (append-only).
    # ------------------------------------------------------------------
    def table_setup_column(mut self, name: String, spec: Float32, is_fixed: Bool):
        self.columns.append(Column(name, spec, is_fixed))

    # ------------------------------------------------------------------
    # _resolve_columns(): compute col_x (prefix sums, screen-relative) and col_w from the declared
    #   columns. FIXED widths are summed; the REMAINING content width is split among STRETCH columns
    #   in proportion to their weights. Computed once; idempotent (guarded by `resolved`).
    #
    #   total_fixed   = sum(spec for fixed columns)
    #   total_weight  = sum(spec for stretch columns)
    #   remaining     = max(0, w - total_fixed)
    #   stretch col i = remaining * (weight_i / total_weight)      (if total_weight > 0)
    #   col_x[0] = x ; col_x[i] = col_x[i-1] + col_w[i-1]          (prefix sum from table left x)
    # ------------------------------------------------------------------
    def _resolve_columns(mut self):
        if self.resolved:
            return
        var n = len(self.columns)
        var total_fixed = Float32(0.0)
        var total_weight = Float32(0.0)
        for i in range(n):
            var c = self.columns[i].copy()
            if c.is_fixed:
                total_fixed = total_fixed + c.spec
            else:
                total_weight = total_weight + c.spec
        var remaining = self.w - total_fixed
        if remaining < Float32(0.0):
            remaining = Float32(0.0)

        # resolve each width
        self.col_w = List[Float32]()
        for i in range(n):
            var c2 = self.columns[i].copy()
            var cw = Float32(0.0)
            if c2.is_fixed:
                cw = c2.spec
            else:
                if total_weight > Float32(0.0):
                    cw = remaining * (c2.spec / total_weight)
                else:
                    cw = Float32(0.0)
            self.col_w.append(cw)

        # prefix-sum the x positions from the table's left edge
        self.col_x = List[Float32]()
        var acc = self.x
        for i in range(n):
            self.col_x.append(acc)
            acc = acc + self.col_w[i]
        self.resolved = True

    # ------------------------------------------------------------------
    # public read-accessors for resolved geometry (used by the self-test + callers).
    #   column_x(col) : the SCREEN x of column `col`'s left edge.
    #   column_x_rel(col) : the x RELATIVE to the table's left edge (so {0,60,180} for the spec test).
    #   column_w(col) : the resolved pixel width of column `col`.
    # ------------------------------------------------------------------
    def column_x(mut self, c: Int) -> Float32:
        self._resolve_columns()
        return self.col_x[c]

    def column_x_rel(mut self, c: Int) -> Float32:
        self._resolve_columns()
        return self.col_x[c] - self.x

    def column_w(mut self, c: Int) -> Float32:
        self._resolve_columns()
        return self.col_w[c]

    # ------------------------------------------------------------------
    # table_cell_rect(col,row) -> (x,y,w,h): the screen rect of any cell. Row 0 is the FIRST DATA
    #   row, located one row-height BELOW the header band. The header band sits at y .. y+row_h;
    #   data row r sits at y + (r+1)*row_h .. y + (r+2)*row_h. Width is the column's resolved width.
    # ------------------------------------------------------------------
    def table_cell_rect(mut self, c: Int, r: Int) -> Tuple[Float32, Float32, Float32, Float32]:
        self._resolve_columns()
        var cx = self.col_x[c]
        var cw = self.col_w[c]
        var cy = self.y + Float32(r + 1) * self.row_h
        return Tuple[Float32, Float32, Float32, Float32](cx, cy, cw, self.row_h)

    # ------------------------------------------------------------------
    # table_headers_row(): draw the header bar (distinct color) across the table width, each column
    #   label inside its cell (clipped), and a 1px vertical separator at each column boundary.
    #   The header band occupies y .. y + row_h. Marks header_drawn so data rows start below it.
    # ------------------------------------------------------------------
    def table_headers_row(mut self) raises:
        self._resolve_columns()
        var n = len(self.columns)
        var hy = self.y
        # header bar background (distinct, brighter than data rows)
        col(58, 62, 78)
        fill(self.x, hy, self.w, self.row_h)
        # bottom edge of the header band (heavier underline)
        col(20, 20, 26)
        line_t(self.x, hy + self.row_h, self.x + self.w, hy + self.row_h, Float32(1.0))
        # each header label, clipped to its cell, + a vertical separator on the right boundary
        for i in range(n):
            var cx = self.col_x[i]
            var cw = self.col_w[i]
            # clip label to the cell interior (small left pad)
            set_clip_rect_px(cx, hy, cw, self.row_h)
            col(225, 228, 236)
            var hc = self.columns[i].copy()
            txt(hc.name, cx + Float32(5.0), hy + Float32(5.0), Float32(12.0))
            clear_clip_rect()
            # vertical separator at the right edge of this column (skip the table's outer right edge)
            if i < n - 1:
                var bx = cx + cw
                col(20, 20, 26)
                line_t(bx, hy, bx, hy + self.row_h, Float32(1.0))
        # outer border of the whole table
        col(20, 20, 26)
        rect(self.x, self.y, self.w, self.h)
        self.header_drawn = True

    # ------------------------------------------------------------------
    # table_next_row(): advance to the next DATA row. The first call makes cur_row = 0. Draws the
    #   alternating row background across the table width and resets the per-row column cursor so the
    #   next table_next_column returns column 0.
    # ------------------------------------------------------------------
    def table_next_row(mut self) raises:
        self._resolve_columns()
        self.cur_row = self.cur_row + 1
        self.cur_col = -1
        var ry = self.y + Float32(self.cur_row + 1) * self.row_h   # +1 => skip the header band
        # alternating row background: even rows darker, odd rows a touch lighter
        if self.cur_row % 2 == 0:
            col(34, 34, 40)
        else:
            col(42, 42, 50)
        fill(self.x, ry, self.w, self.row_h)
        # 1px row underline (border between rows)
        col(22, 22, 28)
        line_t(self.x, ry + self.row_h, self.x + self.w, ry + self.row_h, Float32(1.0))

    # ------------------------------------------------------------------
    # table_next_column() -> (x,y): advance the column cursor and return the SCREEN (x,y) of the
    #   current cell's top-left so the caller draws content there. Also draws the 1px vertical
    #   border on the LEFT boundary of every column except the first, and sets a clip rect to the
    #   cell interior (the caller draws, then the NEXT table_next_column / end_table clears it).
    #
    #   To keep the clip lifecycle simple and leak-free, we CLEAR any prior cell clip first, then
    #   set the new cell's clip. end_table() issues a final clear_clip_rect().
    # ------------------------------------------------------------------
    def table_next_column(mut self) raises -> Tuple[Float32, Float32]:
        self._resolve_columns()
        # close the previous cell's clip (if any) before opening the new one
        clear_clip_rect()
        self.cur_col = self.cur_col + 1
        var c = self.cur_col
        var cx = self.col_x[c]
        var cw = self.col_w[c]
        var cy = self.y + Float32(self.cur_row + 1) * self.row_h   # +1 => below header band
        # vertical border on the left boundary of every column except column 0
        if c > 0:
            col(22, 22, 28)
            line_t(cx, cy, cx, cy + self.row_h, Float32(1.0))
        # clip subsequent caller draws to this cell's interior (small left pad)
        set_clip_rect_px(cx + Float32(1.0), cy, cw - Float32(1.0), self.row_h)
        return Tuple[Float32, Float32](cx + Float32(5.0), cy + Float32(5.0))

    # ------------------------------------------------------------------
    # end_table(): clear any lingering cell clip and draw the table's outer border (idempotent with
    #   the header's border). No state to pop on the backend beyond the clip.
    # ------------------------------------------------------------------
    def end_table(mut self):
        clear_clip_rect()
        col(20, 20, 26)
        rect(self.x, self.y, self.w, self.h)


# ============================================================================
# Free-function API mirroring the immediate-mode GUI call names. begin_table CONSTRUCTS a Table (returned by
# value, owned ^). The remaining calls are thin forwarders so callers can write the canonical
# immediate-mode GUI sequence without method syntax if they prefer.
# ============================================================================
def begin_table(id: Int, x: Float32, y: Float32, w: Float32, h: Float32, ncols: Int) -> Table:
    return Table(id, x, y, w, h, ncols)


def table_setup_column(mut t: Table, name: String, spec: Float32, is_fixed: Bool):
    t.table_setup_column(name, spec, is_fixed)


def table_headers_row(mut t: Table) raises:
    t.table_headers_row()


def table_next_row(mut t: Table) raises:
    t.table_next_row()


def table_next_column(mut t: Table) raises -> Tuple[Float32, Float32]:
    return t.table_next_column()


def table_cell_rect(mut t: Table, c: Int, r: Int) -> Tuple[Float32, Float32, Float32, Float32]:
    return t.table_cell_rect(c, r)


def end_table(mut t: Table):
    t.end_table()


# ============================================================================
# SELF-TEST  (compiled + run headlessly by the MAIN LOOP — do NOT run it yourself)
#
# Every print line is gated by the main loop; the EXACT expected number is in a comment, derived
# from first principles. Every visual element issues real draw_* calls (the render gate readback).
# ============================================================================
def main() raises:
    var pass_all = True

    # ===================================================================
    # TABLES_SELFTEST — 3 columns in a table of width 300:
    #   col0 FIXED 60px ; col1 STRETCH weight 1 ; col2 STRETCH weight 1.
    #
    # First-principles column math (see _resolve_columns):
    #   total_fixed  = 60                       (only col0 is fixed)
    #   total_weight = 1 + 1 = 2                 (col1 + col2 weights)
    #   remaining    = w - total_fixed = 300 - 60 = 240
    #   col_w[0] = 60                            (fixed)
    #   col_w[1] = remaining * (1/2) = 240 * 0.5 = 120
    #   col_w[2] = remaining * (1/2) = 240 * 0.5 = 120
    #   col_x (relative to table left, x=0):
    #     col_x[0] = 0
    #     col_x[1] = col_x[0] + col_w[0] = 0 + 60 = 60
    #     col_x[2] = col_x[1] + col_w[1] = 60 + 120 = 180
    #   => COL_X relative positions = {0, 60, 180}   <-- the spec's required assertion.
    # ===================================================================
    var t = begin_table(1, Float32(0.0), Float32(0.0), Float32(300.0), Float32(120.0), 3)
    table_setup_column(t, String("ID"), Float32(60.0), True)     # fixed 60px
    table_setup_column(t, String("Name"), Float32(1.0), False)   # stretch weight 1
    table_setup_column(t, String("Value"), Float32(1.0), False)  # stretch weight 1

    var x0 = t.column_x_rel(0)
    var x1 = t.column_x_rel(1)
    var x2 = t.column_x_rel(2)
    var w0 = t.column_w(0)
    var w1 = t.column_w(1)
    var w2 = t.column_w(2)

    print("COL_X0=", x0)          # EXPECT: COL_X0= 0.0
    print("COL_X1=", x1)          # EXPECT: COL_X1= 60.0
    print("COL_X2=", x2)          # EXPECT: COL_X2= 180.0
    print("COL_W0=", w0)          # EXPECT: COL_W0= 60.0
    print("COL_W1=", w1)          # EXPECT: COL_W1= 120.0
    print("COL_W2=", w2)          # EXPECT: COL_W2= 120.0

    if x0 != Float32(0.0):
        pass_all = False
    if x1 != Float32(60.0):
        pass_all = False
    if x2 != Float32(180.0):
        pass_all = False
    if w0 != Float32(60.0):
        pass_all = False
    if w1 != Float32(120.0):
        pass_all = False
    if w2 != Float32(120.0):
        pass_all = False

    # ----- draw the header bar (distinct color) + a cell border, satisfying the render gate -----
    table_headers_row(t)

    # ----- 4 DATA ROWS: each row issues alternating-bg fill + per-cell borders + content draws -----
    # Row geometry check on the way: data row r top y = y + (r+1)*row_h. Default row_h = 22.
    #   row 0 top = 0 + 1*22 = 22 ; row 1 top = 44 ; row 2 top = 66 ; row 3 top = 88.
    # table_cell_rect(col, row) must agree with table_next_column's returned (x,y) (minus the +5 pad).
    for r in range(4):
        table_next_row(t)
        # column 0 cell
        var p0 = table_next_column(t)
        col(220, 220, 226)
        txt(String("r") + String(r) + String("c0"), p0[0], p0[1], Float32(11.0))
        # column 1 cell
        var p1 = table_next_column(t)
        col(220, 220, 226)
        txt(String("r") + String(r) + String("c1"), p1[0], p1[1], Float32(11.0))
        # column 2 cell
        var p2 = table_next_column(t)
        col(220, 220, 226)
        txt(String("r") + String(r) + String("c2"), p2[0], p2[1], Float32(11.0))

    # ----- verify table_cell_rect agrees with the layout (col2,row3): x=180,y=88,w=120,h=22 -----
    var cr = table_cell_rect(t, 2, 3)
    var crx = cr[0]
    var cry = cr[1]
    var crw = cr[2]
    var crh = cr[3]
    print("CELL_2_3_X=", crx)     # EXPECT: CELL_2_3_X= 180.0   (col_x[2], table x=0)
    print("CELL_2_3_Y=", cry)     # EXPECT: CELL_2_3_Y= 88.0    (0 + (3+1)*22)
    print("CELL_2_3_W=", crw)     # EXPECT: CELL_2_3_W= 120.0   (col_w[2])
    print("CELL_2_3_H=", crh)     # EXPECT: CELL_2_3_H= 22.0    (row_h)
    if crx != Float32(180.0):
        pass_all = False
    if cry != Float32(88.0):
        pass_all = False
    if crw != Float32(120.0):
        pass_all = False
    if crh != Float32(22.0):
        pass_all = False

    end_table(t)

    # ----- summary line the main loop gates on -----
    if pass_all:
        print("TABLES_SELFTEST: pass")
    else:
        print("TABLES_SELFTEST: fail")
