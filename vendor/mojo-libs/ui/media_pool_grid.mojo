# media_pool_grid.mojo — a MEDIA-POOL THUMBNAIL GRID widget for Mojo 1.0.0b1, drawn on the
# MojoGUI C rendering backend (librender.so) via external_call ONLY. This is Resolve's "Media Pool":
# a scrollable grid of RGBA8 thumbnails with a name under each; clicking a cell selects it.
#
# SELF-CONTAINED: its own FFI helpers + UiContext (mirrors core_widgets.mojo EXACTLY) so it can be
# compiled/run standalone for the headless self-test (the mojopkg race means we can't link the
# shared toolkit). Idioms mirrored from the verified exemplars:
#   - core_widgets.mojo: to_cstr 32-38; col/fill/rect/txt 45-69; UiContext input model (mouse_x/y:Int,
#     mouse_down/prev/clicked/released:Bool, hot_id/active_id:Int, set_input edge derivation,
#     begin_frame -> hot_id=-1); point_in_rect/clampf.
#   - media_widgets.mojo: filmstrip 151-157 (draw_image(tx,ty,w,h, slot, tw, th) on a packed RGBA8
#     buffer, slot = thumbs + i*tw*th*4); the Toolbar `draw: Bool` HEADLESS pattern (draw=False skips
#     ALL external_call/GL — the gated logic path is pure).
#   - widget_variants.mojo: list_box_scroll 1026-1085 (caller-owned `scroll` px-from-top, content_h,
#     max_scroll clamp, vertical offset iy = base - scroll); comp_cell_x cell-geometry helper.
#
# NO `from math` import (the windowed FFI binary cannot link transcendentals); only builtin
# min/max/abs/Int()/Float32() arithmetic is used. Widget ids are EXPLICIT (passed per call-site).

from std.ffi import external_call
from std.memory import UnsafePointer, alloc
from builtin.type_aliases import MutExternalOrigin


# ============================================================================
# C backend helpers (mirror core_widgets.mojo:32-69 / media_widgets.mojo:26-41 EXACTLY)
# ============================================================================
def to_cstr(s: String) -> UnsafePointer[Int8, MutExternalOrigin]:
    var b = s.as_bytes()
    var buf = alloc[Int8](len(b) + 1)
    for i in range(len(b)):
        buf[i] = Int8(b[i])
    buf[len(b)] = 0
    return buf


def col(r: Int, g: Int, b: Int):
    _ = external_call["set_color", Int32](
        Float32(r) / 255.0, Float32(g) / 255.0, Float32(b) / 255.0, Float32(1.0)
    )


def fill(x: Float32, y: Float32, w: Float32, h: Float32):
    _ = external_call["draw_filled_rectangle", Int32](x, y, w, h)


def rect(x: Float32, y: Float32, w: Float32, h: Float32):
    _ = external_call["draw_rectangle", Int32](x, y, w, h)


def txt(s: String, x: Float32, y: Float32, size: Float32):
    _ = external_call["draw_text", Int32](to_cstr(s), x, y, size)


def draw_image_rgba(x: Float32, y: Float32, w: Float32, h: Float32, px: UnsafePointer[UInt8, MutExternalOrigin], tw: Int, th: Int):
    # draw_image(x, y, w, h, rgba8_ptr, srcW, srcH) — mirrors media_widgets.filmstrip:157 exactly.
    _ = external_call["draw_image", Int32](x, y, w, h, px, Int32(tw), Int32(th))


def push_clip(x: Float32, y: Float32, w: Float32, h: Float32):
    _ = external_call["set_clip_rect", Int32](Int32(x), Int32(y), Int32(w), Int32(h))


def pop_clip():
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
# UiContext — the immediate-mode core (mirrors core_widgets.UiContext EXACTLY; self-contained copy
# so this module compiles standalone for its headless self-test).
# ============================================================================
struct UiContext(Copyable, Movable):
    var mouse_x: Int
    var mouse_y: Int
    var mouse_down: Bool
    var prev_mouse_down: Bool
    var mouse_clicked: Bool
    var mouse_released: Bool
    var hot_id: Int
    var active_id: Int

    def __init__(out self):
        self.mouse_x = 0
        self.mouse_y = 0
        self.mouse_down = False
        self.prev_mouse_down = False
        self.mouse_clicked = False
        self.mouse_released = False
        self.hot_id = -1
        self.active_id = -1

    # ---- INJECTABLE input (mandatory for headless verification) ----
    def set_input(mut self, mx: Int, my: Int, down: Bool):
        self.mouse_clicked = down and not self.prev_mouse_down
        self.mouse_released = self.prev_mouse_down and not down
        self.mouse_x = mx
        self.mouse_y = my
        self.mouse_down = down
        self.prev_mouse_down = down

    # ---- LIVE input from the backend (production path) ----
    def pump_live_input(mut self):
        var mx = Int(external_call["get_mouse_x", Int32]())
        var my = Int(external_call["get_mouse_y", Int32]())
        var bs = Int(external_call["get_mouse_button_state", Int32](Int32(0)))
        var down = bs != 0
        self.mouse_clicked = down and not self.prev_mouse_down
        self.mouse_released = self.prev_mouse_down and not down
        self.mouse_x = mx
        self.mouse_y = my
        self.mouse_down = down
        self.prev_mouse_down = down

    def begin_frame(mut self):
        self.hot_id = -1

    def _mouse_in(self, x: Float32, y: Float32, w: Float32, h: Float32) -> Bool:
        return point_in_rect(Float32(self.mouse_x), Float32(self.mouse_y), x, y, w, h)


# ============================================================================
# grid geometry — pure (no GL), shared by both the gated logic path AND the draw path.
#
# Layout: the grid lays n cells into `cols` columns (rows = ceil(n/cols)). Each cell is a fixed
# THUMB CELL: a tw x th image area + a NAME_H label strip below it, with CELL_PAD around. Cells are
# packed left-to-right, top-to-bottom. A vertical `scroll` (px from top, list_box_scroll style)
# offsets every cell's y. cell_rect(i) returns the cell's (x, y, w, h) in WIDGET-LOCAL UNSCROLLED
# coords (origin = grid x,y; caller subtracts scroll for the on-screen y). This makes index->row/col
# math testable headlessly.
#
#   row = i // cols ; column = i % cols
#   cell_w = tw + 2*CELL_PAD ; cell_h = th + NAME_H + 2*CELL_PAD
#   cell_x = column * cell_w ; cell_y = row * cell_h
# ============================================================================
comptime CELL_PAD = Float32(6.0)
comptime NAME_H = Float32(16.0)


def grid_cell_w(tw: Int) -> Float32:
    return Float32(tw) + Float32(2.0) * CELL_PAD


def grid_cell_h(th: Int) -> Float32:
    return Float32(th) + NAME_H + Float32(2.0) * CELL_PAD


# cell_rect(i) -> (x, y, w, h) in widget-local UNSCROLLED coords (origin at the grid's top-left).
# Maps a flat index to its (row, column) cell box. cols must be >= 1.
def cell_rect(i: Int, tw: Int, th: Int, cols: Int) -> Tuple[Float32, Float32, Float32, Float32]:
    var c = cols
    if c < 1:
        c = 1
    var column = i % c
    var row = i // c
    var cw = grid_cell_w(tw)
    var ch = grid_cell_h(th)
    var cx = Float32(column) * cw
    var cy = Float32(row) * ch
    return Tuple[Float32, Float32, Float32, Float32](cx, cy, cw, ch)


# total content height (px) for n cells across `cols` columns — used for the scroll clamp.
def grid_content_h(n: Int, th: Int, cols: Int) -> Float32:
    var c = cols
    if c < 1:
        c = 1
    var rows = (n + c - 1) // c            # ceil(n/cols)
    return Float32(rows) * grid_cell_h(th)


# ============================================================================
# media_pool_grid — the widget.
#
# Lays names/thumbs into a `cols`-column grid inside (x,y,w,h). Each cell draws its RGBA8 thumbnail
# (tw x th, packed contiguously in `thumbs` at offset i*tw*th*4 — cols-agnostic) plus its name below.
# Clicking a cell sets `sel` to that index (press+release inside, immediate-mode click semantics).
# A caller-owned vertical `scroll` (px from top) offsets the grid (list_box_scroll style); the grid
# is clipped to (x,y,w,h). Returns True iff the selection changed this frame.
#
# draw=False => HEADLESS: skips ALL external_call/GL (the gated hit-test/selection logic still runs),
# for the self-test. n = len(names); `thumbs` may be null when draw=False / n==0.
# ============================================================================
def media_pool_grid(
    mut ui: UiContext,
    id: Int,
    x: Float32,
    y: Float32,
    w: Float32,
    h: Float32,
    names: List[String],
    thumbs: UnsafePointer[UInt8, MutExternalOrigin],
    tw: Int,
    th: Int,
    cols: Int,
    mut sel: Int,
    mut scroll: Float32,
    draw: Bool,
) raises -> Bool:
    var n = len(names)
    var c = cols
    if c < 1:
        c = 1
    var content_h = grid_content_h(n, th, c)
    var max_scroll = content_h - h
    if max_scroll < Float32(0.0):
        max_scroll = Float32(0.0)
    scroll = clampf(scroll, Float32(0.0), max_scroll)

    var cw = grid_cell_w(tw)
    var ch = grid_cell_h(th)
    var changed = False

    # ---- panel background + frame (draw only) ----
    if draw:
        col(24, 24, 28)
        fill(x, y, w, h)
        col(0, 0, 0)
        rect(x, y, w, h)
        push_clip(x, y, w, h)

    # ---- cells: hit-test + select (logic runs headless), then paint (draw only) ----
    for i in range(n):
        var r = cell_rect(i, tw, th, c)
        var cell_x = x + r[0]
        var cell_y = y + r[1] - scroll              # on-screen y (scrolled)

        # cull rows fully outside the viewport (matches list_box_scroll:1066)
        if cell_y + ch < y or cell_y > y + h:
            continue

        var cell_id = id * 8192 + i + 1             # stable per-cell id
        var inside = ui._mouse_in(cell_x, cell_y, cw, ch)
        if inside:
            ui.hot_id = cell_id
        if inside and ui.mouse_clicked:
            ui.active_id = cell_id
        if ui.mouse_released and ui.active_id == cell_id:
            if inside:
                if sel != i:
                    sel = i
                    changed = True
            ui.active_id = -1

        if draw:
            # cell background: selected accent / hover / idle
            if i == sel:
                col(60, 96, 150)
            elif ui.hot_id == cell_id:
                col(48, 48, 56)
            else:
                col(32, 32, 38)
            fill(cell_x, cell_y, cw, ch)
            # thumbnail image (RGBA8, packed contiguously — cols-agnostic offset)
            var slot = thumbs + (i * tw * th * 4)
            var img_x = cell_x + CELL_PAD
            var img_y = cell_y + CELL_PAD
            draw_image_rgba(img_x, img_y, Float32(tw), Float32(th), slot, tw, th)
            # name strip below the thumbnail
            if i == sel:
                col(240, 240, 245)
            else:
                col(210, 210, 216)
            txt(names[i].copy(), img_x, img_y + Float32(th) + Float32(2.0), Float32(11.0))

    if draw:
        pop_clip()
    return changed


# ============================================================================
# SELF-TEST  (compiled + run headlessly by the MAIN LOOP — do NOT run it yourself)
#
# draw=False everywhere: NO GL is touched. Press+release widgets need two frames (down -> up); the
# click fires on the UP frame when the cursor is still inside the cell it pressed.
#
# Every expected number is derived from FIRST PRINCIPLES in the comment beside it.
# ============================================================================
def main() raises:
    var pass_all = True

    # ---- geometry: cell_rect maps index -> row/col correctly. tw=80, th=45, cols=3. ----
    #   cell_w = 80 + 2*6 = 92.0 ; cell_h = 45 + 16 + 2*6 = 73.0
    #   index 4 -> column = 4 % 3 = 1, row = 4 // 3 = 1 -> cell_x = 1*92 = 92.0, cell_y = 1*73 = 73.0
    var tw = 80
    var th = 45
    var cols = 3
    var cwexp = Float32(92.0)
    var chexp = Float32(73.0)
    if grid_cell_w(tw) != cwexp:
        pass_all = False
    if grid_cell_h(th) != chexp:
        pass_all = False

    var g0 = cell_rect(0, tw, th, cols)        # col0,row0 -> (0,0)
    if g0[0] != Float32(0.0) or g0[1] != Float32(0.0) or g0[2] != cwexp or g0[3] != chexp:
        pass_all = False
    var g2 = cell_rect(2, tw, th, cols)        # col2,row0 -> (184, 0)
    if g2[0] != Float32(184.0) or g2[1] != Float32(0.0):
        pass_all = False
    var g4 = cell_rect(4, tw, th, cols)        # col1,row1 -> (92, 73)
    if g4[0] != Float32(92.0) or g4[1] != Float32(73.0):
        pass_all = False
    var g6 = cell_rect(6, tw, th, cols)        # col0,row2 -> (0, 146)
    if g6[0] != Float32(0.0) or g6[1] != Float32(146.0):
        pass_all = False
    print("MEDIA_GRID_SELFTEST: cell0=(", g0[0], ",", g0[1], ") cell4=(", g4[0], ",", g4[1],
          ") cell6=(", g6[0], ",", g6[1], ")")
    # EXPECT: cell0=(0.0,0.0) cell4=(92.0,73.0) cell6=(0.0,146.0)

    # content height for 7 cells / 3 cols -> ceil(7/3)=3 rows -> 3*73 = 219.0
    if grid_content_h(7, th, cols) != Float32(219.0):
        pass_all = False

    # ---- selection: click the cell at row1,col1 (index 4) -> sel becomes 4, returns True. ----
    # Grid at (x=100, y=100, w=300, h=400) — tall enough that no scroll/cull hides any cell.
    # Cell 4 on-screen rect (scroll=0): cell_x = 100 + 92 = 192, cell_y = 100 + 73 = 173, 92 x 73.
    #   center = (192 + 46, 173 + 36) = (238, 209). 238∈[192,284), 209∈[173,246) -> inside cell 4 only.
    var names = List[String]()
    names.append(String("clipA"))
    names.append(String("clipB"))
    names.append(String("clipC"))
    names.append(String("clipD"))
    names.append(String("clipE"))
    names.append(String("clipF"))
    names.append(String("clipG"))                      # 7 names

    var null_thumbs = alloc[UInt8](1)   # repo-proven throwaway buf (cf. to_cstr/media_widgets:562); never read (draw=False)
    var sel = -1
    var scroll = Float32(0.0)
    var gx = Float32(100.0)
    var gy = Float32(100.0)
    var gw = Float32(300.0)
    var gh = Float32(400.0)
    var ccx = 238
    var ccy = 209

    var ui = UiContext()
    # frame 1: press (down) inside cell 4 -> latches active, no change yet.
    ui.begin_frame()
    ui.set_input(ccx, ccy, True)
    var ch1 = media_pool_grid(ui, 1, gx, gy, gw, gh, names, null_thumbs, tw, th, cols, sel, scroll, False)
    # frame 2: release (up) still inside cell 4 -> selection changes to 4, returns True.
    ui.begin_frame()
    ui.set_input(ccx, ccy, False)
    var ch2 = media_pool_grid(ui, 1, gx, gy, gw, gh, names, null_thumbs, tw, th, cols, sel, scroll, False)
    print("MEDIA_GRID_SELFTEST: press_changed=", ch1, " release_changed=", ch2, " sel=", sel)
    # EXPECT: press_changed= False  release_changed= True  sel= 4
    if ch1 != False:
        pass_all = False
    if ch2 != True:
        pass_all = False
    if sel != 4:
        pass_all = False

    # ---- re-click the SAME cell -> no change (sel already 4), returns False. ----
    ui.begin_frame()
    ui.set_input(ccx, ccy, True)
    _ = media_pool_grid(ui, 1, gx, gy, gw, gh, names, null_thumbs, tw, th, cols, sel, scroll, False)
    ui.begin_frame()
    ui.set_input(ccx, ccy, False)
    var ch3 = media_pool_grid(ui, 1, gx, gy, gw, gh, names, null_thumbs, tw, th, cols, sel, scroll, False)
    if ch3 != False:
        pass_all = False
    if sel != 4:
        pass_all = False

    # ---- click cell 0 (top-left) -> sel becomes 0, returns True. ----
    #   cell 0 rect: (100, 100, 92, 73) ; center = (146, 136). 146∈[100,192), 136∈[100,173) -> cell 0.
    ui.begin_frame()
    ui.set_input(146, 136, True)
    _ = media_pool_grid(ui, 1, gx, gy, gw, gh, names, null_thumbs, tw, th, cols, sel, scroll, False)
    ui.begin_frame()
    ui.set_input(146, 136, False)
    var ch4 = media_pool_grid(ui, 1, gx, gy, gw, gh, names, null_thumbs, tw, th, cols, sel, scroll, False)
    if ch4 != True:
        pass_all = False
    if sel != 0:
        pass_all = False

    # ---- scroll clamp: a viewport shorter than content clamps scroll to max_scroll. ----
    # gh=100 -> content 219 -> max_scroll = 119. An over-large scroll is clamped down in-place.
    var sel2 = -1
    var scroll2 = Float32(9999.0)
    ui.begin_frame()
    ui.set_input(5000, 5000, False)
    _ = media_pool_grid(ui, 2, gx, gy, gw, Float32(100.0), names, null_thumbs, tw, th, cols, sel2, scroll2, False)
    print("MEDIA_GRID_SELFTEST: scroll_clamped=", scroll2)
    # EXPECT: scroll_clamped= 119.0   (content 219 - viewport 100)
    if scroll2 != Float32(119.0):
        pass_all = False

    if pass_all:
        print("MEDIA_GRID_SELFTEST: pass (cell4=(92,73), click idx4 -> sel=4, re-click no-op, click idx0 -> sel=0, scroll clamp=119)")
    else:
        print("MEDIA_GRID_SELFTEST: fail")
