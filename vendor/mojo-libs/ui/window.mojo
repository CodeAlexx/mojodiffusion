# window.mojo — WORKSTREAM: REAL movable/resizable WINDOWS for Mojo 1.0.0b1, drawn on the
# MojoGUI C rendering backend (librender.so) via external_call ONLY. No backend change.
#
# A Window struct (id,title,x,y,w,h,collapsed,scroll) plus begin_window/end_window with:
#   - a TITLE BAR (drag it to MOVE the window),
#   - a COLLAPSE toggle (a caret in the title bar; clicking it flips `collapsed`),
#   - a RESIZE GRIP (bottom-right 14x14; drag to resize, clamped to a min size),
#   - a content region CLIPPED via set_clip_rect / clear_clip_rect,
#   - a vertical SCROLLBAR whose thumb reflects scroll/content_height and which also responds to
#     get_scroll_y (injected as `wheel` in tests).
#
# update(mx,my,down,wheel) drives MOVE / RESIZE / SCROLL entirely from INJECTED input — it NEVER
# calls get_mouse_*/get_scroll_y itself (the headless self-test relies on this). A separate
# pump_live() binds the live backend input in production.
#
# FFI idioms mirrored EXACTLY from the framebuffer-verified exemplars
#   /home/alex/framepfx-mojo/src/keyframe_editor.mojo (to_cstr 37-43; col/fill/rect/line_t/txt
#   wrappers) and /tmp/mojoui/core_widgets.mojo + /tmp/mojoui2/widget_pack.mojo (UiContext input
#   model: mouse_x/mouse_y:Int, mouse_down/prev_mouse_down/mouse_clicked/mouse_released:Bool,
#   set_input edge derivation, press_x/press_y drag anchor, point_in_rect/clampf, press+release
#   click semantics). This module carries the SAME context-struct shape (WinCtx) and adds the
#   per-window drag-mode state needed for move/resize.
#
# CLIP + SCROLL FFI (call idiom from the workstream brief):
#   _ = external_call["set_clip_rect", Int32](Int32(x),Int32(y),Int32(w),Int32(h))   # top-left rect
#   _ = external_call["clear_clip_rect", Int32]()
#   var sy = external_call["get_scroll_y", Float64]()                                 # consumed wheel
#
# NO `from math` import: the windowed FFI binary cannot link transcendentals (sincos@GLIBC). Only
# builtin min/max/abs/Int()/Float32() arithmetic is used (same constraint as the exemplars). The
# caret is drawn from draw_line arms (no trig), mirroring widget_pack.tree_node.

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


# ---- clipping wrappers (top-left integer rect) ----
def set_clip(x: Float32, y: Float32, w: Float32, h: Float32):
    _ = external_call["set_clip_rect", Int32](Int32(Int(x)), Int32(Int(y)), Int32(Int(w)), Int32(Int(h)))


def clear_clip():
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
# Layout / theme constants (fixed so move/resize/scroll math is reproducible headless).
# ============================================================================
comptime TITLE_H = Float32(24.0)        # title-bar height
comptime GRIP = Float32(14.0)           # resize grip is GRIP x GRIP at the bottom-right corner
comptime SB_W = Float32(12.0)           # vertical scrollbar width
comptime LINE = Float32(20.0)           # one scroll "line" (wheel notch) in px
comptime MIN_W = Float32(120.0)         # minimum window width
comptime MIN_H = Float32(80.0)          # minimum window height (title bar + a little content)
comptime CARET_BOX = Float32(20.0)      # clickable caret box at the left of the title bar

# drag modes for WinCtx.drag_mode
comptime DRAG_NONE = 0
comptime DRAG_MOVE = 1                   # dragging the title bar -> moving the window
comptime DRAG_RESIZE = 2                 # dragging the grip -> resizing the window
comptime DRAG_SCROLL = 3                 # dragging the scrollbar thumb -> scrolling


# ============================================================================
# Window — persistent per-window state (caller owns one of these per window).
#
#   id        : stable window id (used as the active-capture id in WinCtx)
#   title     : title-bar text
#   x, y      : top-left of the window (the MOVE target)
#   w, h      : full window size including the title bar (the RESIZE target)
#   collapsed : when True the body is hidden (only the title bar shows)
#   scroll    : vertical scroll offset in px (0 .. max_scroll), clamped against content_height
# ============================================================================
struct Window(Copyable, Movable):
    var id: Int
    var title: String
    var x: Float32
    var y: Float32
    var w: Float32
    var h: Float32
    var collapsed: Bool
    var scroll: Float32

    def __init__(out self, id: Int, title: String, x: Float32, y: Float32, w: Float32, h: Float32):
        self.id = id
        self.title = title.copy()
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.collapsed = False
        self.scroll = Float32(0.0)

    # is the body visible this frame? (the COLLAPSE toggle drives this)
    def content_visible(self) -> Bool:
        return not self.collapsed

    # the content viewport rect (below the title bar, left of the scrollbar). Returns (x,y,w,h).
    def content_rect(self) -> Tuple[Float32, Float32, Float32, Float32]:
        var cx = self.x
        var cy = self.y + TITLE_H
        var cw = self.w - SB_W
        var ch = self.h - TITLE_H
        if cw < Float32(0.0):
            cw = Float32(0.0)
        if ch < Float32(0.0):
            ch = Float32(0.0)
        return Tuple[Float32, Float32, Float32, Float32](cx, cy, cw, ch)


# ============================================================================
# WinCtx — the immediate-mode context, SAME shape as core_widgets UiContext / widget_pack Ctx.
#
# Carries the injectable per-frame input (mouse_x/mouse_y:Int; mouse_down/prev_mouse_down/
# mouse_clicked/mouse_released:Bool) plus hot_id/active_id, a press anchor (press_x/press_y) used
# for relative move/resize deltas, the window-origin/size captured at press time
# (orig_x/orig_y/orig_w/orig_h) so resize/move are computed from the press snapshot, the current
# drag_mode, and the injected wheel delta for this frame.
# ============================================================================
struct WinCtx(Copyable, Movable):
    var mouse_x: Int
    var mouse_y: Int
    var mouse_down: Bool
    var prev_mouse_down: Bool
    var mouse_clicked: Bool
    var mouse_released: Bool
    var hot_id: Int
    var active_id: Int
    var press_x: Int            # cursor x when a drag latched
    var press_y: Int            # cursor y when a drag latched
    var orig_x: Float32         # window x at press (for MOVE)
    var orig_y: Float32         # window y at press
    var orig_w: Float32         # window w at press (for RESIZE)
    var orig_h: Float32         # window h at press
    var drag_mode: Int          # DRAG_NONE / DRAG_MOVE / DRAG_RESIZE / DRAG_SCROLL
    var wheel: Float32          # injected scroll-wheel delta for THIS frame (consumed by scroll)

    def __init__(out self):
        self.mouse_x = 0
        self.mouse_y = 0
        self.mouse_down = False
        self.prev_mouse_down = False
        self.mouse_clicked = False
        self.mouse_released = False
        self.hot_id = -1
        self.active_id = -1
        self.press_x = 0
        self.press_y = 0
        self.orig_x = Float32(0.0)
        self.orig_y = Float32(0.0)
        self.orig_w = Float32(0.0)
        self.orig_h = Float32(0.0)
        self.drag_mode = DRAG_NONE
        self.wheel = Float32(0.0)

    # ---- INJECTABLE input (mandatory for headless verification) ----
    # Derives click/release edges from the previous frame, advances prev_mouse_down, and stores the
    # per-frame wheel delta. Identical edge derivation to core_widgets.set_input + widget_pack.
    def set_input(mut self, mx: Int, my: Int, down: Bool, wheel: Float32):
        self.mouse_clicked = down and not self.prev_mouse_down
        self.mouse_released = self.prev_mouse_down and not down
        self.mouse_x = mx
        self.mouse_y = my
        self.mouse_down = down
        self.prev_mouse_down = down
        self.wheel = wheel

    # ---- LIVE input from the backend (production path) ----
    # Binds get_mouse_* + get_scroll_y here ONLY (the gated update path never touches the backend).
    def pump_live(mut self):
        var mx = Int(external_call["get_mouse_x", Int32]())
        var my = Int(external_call["get_mouse_y", Int32]())
        var bs = Int(external_call["get_mouse_button_state", Int32](Int32(0)))
        var down = bs != 0
        var sy = external_call["get_scroll_y", Float64]()
        self.mouse_clicked = down and not self.prev_mouse_down
        self.mouse_released = self.prev_mouse_down and not down
        self.mouse_x = mx
        self.mouse_y = my
        self.mouse_down = down
        self.prev_mouse_down = down
        self.wheel = Float32(sy)

    def begin_frame(mut self):
        self.hot_id = -1

    def _mouse_in(self, x: Float32, y: Float32, w: Float32, h: Float32) -> Bool:
        return point_in_rect(Float32(self.mouse_x), Float32(self.mouse_y), x, y, w, h)

    # ========================================================================
    # max_scroll: how far the content can scroll = content_height - viewport_height (>= 0).
    # ========================================================================
    def _max_scroll(self, win: Window, content_height: Float32) -> Float32:
        var cr = win.content_rect()
        var view_h = cr[3]
        var ms = content_height - view_h
        if ms < Float32(0.0):
            ms = Float32(0.0)
        return ms

    # ========================================================================
    # update(win, content_height, mx, my, down, wheel) — drive MOVE / RESIZE / SCROLL for one
    # window from INJECTED input. Sets edges, latches the appropriate drag_mode on press over the
    # right region, applies the relative delta from the press snapshot on each held frame, and
    # consumes the wheel into scroll. Mutates win in place. Returns nothing.
    #
    # Region priority on a fresh press (mouse_clicked):
    #   1. RESIZE GRIP  (bottom-right GRIPxGRIP)  -> DRAG_RESIZE
    #   2. SCROLLBAR    (right edge, below title) -> DRAG_SCROLL
    #   3. CARET BOX    (left of title bar)       -> COLLAPSE toggle (handled on RELEASE; no drag)
    #   4. TITLE BAR    (rest of the title bar)   -> DRAG_MOVE
    # The caret toggle is a press+release click (like a button), so a press there latches active_id
    # WITHOUT a drag_mode; the flip happens on release if still inside the caret box.
    # ========================================================================
    def update(
        mut self,
        mut win: Window,
        content_height: Float32,
        mx: Int,
        my: Int,
        down: Bool,
        wheel: Float32,
    ):
        # advance synthetic input for this frame
        self.begin_frame()
        self.set_input(mx, my, down, wheel)

        var caret_x = win.x
        var caret_y = win.y
        var grip_x = win.x + win.w - GRIP
        var grip_y = win.y + win.h - GRIP
        var title_x = win.x
        var title_y = win.y

        var in_grip = self._mouse_in(grip_x, grip_y, GRIP, GRIP)
        var in_caret = self._mouse_in(caret_x, caret_y, CARET_BOX, TITLE_H)
        var in_title = self._mouse_in(title_x, title_y, win.w, TITLE_H)

        # ----- hot tracking -----
        if in_grip:
            self.hot_id = win.id
        elif in_title:
            self.hot_id = win.id

        # ----- press: pick a drag mode (or latch the caret click) -----
        if self.mouse_clicked:
            if in_grip:
                self.active_id = win.id
                self.drag_mode = DRAG_RESIZE
                self.press_x = self.mouse_x
                self.press_y = self.mouse_y
                self.orig_w = win.w
                self.orig_h = win.h
            elif self._scrollbar_hit(win, content_height):
                self.active_id = win.id
                self.drag_mode = DRAG_SCROLL
                self.press_x = self.mouse_x
                self.press_y = self.mouse_y
            elif in_caret:
                # caret click: latch active but NO drag (toggle on release).
                self.active_id = win.id
                self.drag_mode = DRAG_NONE
            elif in_title:
                self.active_id = win.id
                self.drag_mode = DRAG_MOVE
                self.press_x = self.mouse_x
                self.press_y = self.mouse_y
                self.orig_x = win.x
                self.orig_y = win.y

        # ----- held: apply the relative delta from the press snapshot -----
        if self.active_id == win.id and self.mouse_down:
            if self.drag_mode == DRAG_MOVE:
                var dx = Float32(self.mouse_x - self.press_x)
                var dy = Float32(self.mouse_y - self.press_y)
                win.x = self.orig_x + dx
                win.y = self.orig_y + dy
            elif self.drag_mode == DRAG_RESIZE:
                var rdx = Float32(self.mouse_x - self.press_x)
                var rdy = Float32(self.mouse_y - self.press_y)
                var nw = self.orig_w + rdx
                var nh = self.orig_h + rdy
                if nw < MIN_W:
                    nw = MIN_W
                if nh < MIN_H:
                    nh = MIN_H
                win.w = nw
                win.h = nh
            elif self.drag_mode == DRAG_SCROLL:
                self._scroll_drag(win, content_height)

        # ----- release: fire caret toggle if it was a caret-click; clear capture -----
        if self.mouse_released and self.active_id == win.id:
            if self.drag_mode == DRAG_NONE and in_caret:
                win.collapsed = not win.collapsed
            self.active_id = -1
            self.drag_mode = DRAG_NONE

        # ----- wheel scroll (independent of any drag): wheel<0 scrolls content DOWN (scroll up in
        # value). scroll += (-wheel) * LINE, clamped to [0, max_scroll]. Only when content visible. -----
        if win.content_visible():
            var ms = self._max_scroll(win, content_height)
            if self.wheel != Float32(0.0) and ms > Float32(0.0):
                win.scroll = clampf(win.scroll + (-self.wheel) * LINE, Float32(0.0), ms)
            else:
                win.scroll = clampf(win.scroll, Float32(0.0), ms)
        else:
            win.scroll = clampf(win.scroll, Float32(0.0), Float32(0.0))

    # is the cursor over the scrollbar thumb-track area (right edge, below the title)?
    def _scrollbar_hit(self, win: Window, content_height: Float32) -> Bool:
        var ms = self._max_scroll(win, content_height)
        if ms <= Float32(0.0):
            return False
        var sb_x = win.x + win.w - SB_W
        var sb_y = win.y + TITLE_H
        var sb_h = win.h - TITLE_H
        return self._mouse_in(sb_x, sb_y, SB_W, sb_h)

    # drag the scrollbar thumb: map the cursor y on the track to a scroll value.
    def _scroll_drag(self, mut win: Window, content_height: Float32):
        var cr = win.content_rect()
        var view_h = cr[3]
        var ms = self._max_scroll(win, content_height)
        if ms <= Float32(0.0):
            win.scroll = Float32(0.0)
            return
        var sb_y = win.y + TITLE_H
        var sb_h = win.h - TITLE_H
        var visible_frac = view_h / content_height
        var thumb_h = sb_h * visible_frac
        if thumb_h < Float32(16.0):
            thumb_h = Float32(16.0)
        var track = sb_h - thumb_h
        if track <= Float32(0.0):
            win.scroll = Float32(0.0)
            return
        var ty = Float32(self.mouse_y) - sb_y - thumb_h / Float32(2.0)
        var t = ty / track
        t = clampf(t, Float32(0.0), Float32(1.0))
        win.scroll = t * ms

    # ========================================================================
    # DRAW — begin_window / end_window. begin_window paints the frame (title bar + caret + body bg
    # + grip + scrollbar) and, when not collapsed, SETS the content clip rect; returns the content
    # viewport rect so the caller draws its content there (offset by -win.scroll). end_window clears
    # the clip. Drawing reads the CURRENT win state already mutated by update().
    # ========================================================================

    # begin_window: returns (cx, cy, cw, ch) = the clipped content viewport.
    def begin_window(mut self, mut win: Window, content_height: Float32) raises -> Tuple[Float32, Float32, Float32, Float32]:
        # ---- title bar ----
        var tb_col_hot = self.hot_id == win.id and self.active_id != win.id
        if self.active_id == win.id and self.drag_mode == DRAG_MOVE:
            col(70, 110, 165)        # actively moving -> accent
        elif tb_col_hot:
            col(58, 64, 86)          # hovered title bar
        else:
            col(46, 50, 66)          # idle title bar
        fill(win.x, win.y, win.w, TITLE_H)
        col(0, 0, 0)
        rect(win.x, win.y, win.w, TITLE_H)

        # ---- collapse caret (left of title) — right-pointing when collapsed, down when open ----
        var cx = win.x + Float32(10.0)
        var cy = win.y + TITLE_H / Float32(2.0)
        col(210, 210, 216)
        if win.collapsed:
            # right-pointing caret
            line_t(cx - Float32(3.0), cy - Float32(4.0), cx + Float32(4.0), cy, Float32(2.0))
            line_t(cx - Float32(3.0), cy + Float32(4.0), cx + Float32(4.0), cy, Float32(2.0))
        else:
            # down-pointing caret
            line_t(cx - Float32(4.0), cy - Float32(3.0), cx, cy + Float32(4.0), Float32(2.0))
            line_t(cx + Float32(4.0), cy - Float32(3.0), cx, cy + Float32(4.0), Float32(2.0))

        # ---- title text (after the caret box) ----
        col(228, 228, 233)
        txt(win.title, win.x + CARET_BOX + Float32(2.0), win.y + Float32(5.0), Float32(13.0))

        var cr = win.content_rect()
        var content_x = cr[0]
        var content_y = cr[1]
        var content_w = cr[2]
        # view_h = the VIEWPORT (visible) height = h - TITLE_H, distinct from `content_height`
        # (the TOTAL scrollable content height passed in by the caller). Kept separate so the
        # thumb math below (view_h / content_height) is unambiguous.
        var view_h = cr[3]

        if not win.content_visible():
            # collapsed: nothing below the title bar; no clip set.
            return Tuple[Float32, Float32, Float32, Float32](content_x, content_y, content_w, Float32(0.0))

        # ---- body background (full window minus title) ----
        col(28, 30, 38)
        fill(win.x, win.y + TITLE_H, win.w, win.h - TITLE_H)
        col(0, 0, 0)
        rect(win.x, win.y + TITLE_H, win.w, win.h - TITLE_H)

        # ---- vertical scrollbar (right edge, below title) ----
        var ms = self._max_scroll(win, content_height)
        var sb_x = win.x + win.w - SB_W
        var sb_y = win.y + TITLE_H
        var sb_h = win.h - TITLE_H
        # track
        col(20, 20, 24)
        fill(sb_x, sb_y, SB_W, sb_h)
        if ms > Float32(0.0):
            var visible_frac = view_h / content_height
            var thumb_h = sb_h * visible_frac
            if thumb_h < Float32(16.0):
                thumb_h = Float32(16.0)
            var track = sb_h - thumb_h
            var thumb_y = sb_y
            if ms > Float32(0.0):
                thumb_y = sb_y + (win.scroll / ms) * track
            if self.active_id == win.id and self.drag_mode == DRAG_SCROLL:
                col(150, 150, 165)
            else:
                col(96, 96, 108)
            fill(sb_x + Float32(2.0), thumb_y, SB_W - Float32(4.0), thumb_h)

        # ---- resize grip (bottom-right) — diagonal hatch lines ----
        var grip_x = win.x + win.w - GRIP
        var grip_y = win.y + win.h - GRIP
        if self.active_id == win.id and self.drag_mode == DRAG_RESIZE:
            col(160, 200, 255)
        else:
            col(150, 150, 162)
        # three short diagonals forming the classic grip
        line_t(grip_x + Float32(3.0), grip_y + GRIP - Float32(1.0), grip_x + GRIP - Float32(1.0), grip_y + Float32(3.0), Float32(1.0))
        line_t(grip_x + Float32(6.0), grip_y + GRIP - Float32(1.0), grip_x + GRIP - Float32(1.0), grip_y + Float32(6.0), Float32(1.0))
        line_t(grip_x + Float32(9.0), grip_y + GRIP - Float32(1.0), grip_x + GRIP - Float32(1.0), grip_y + Float32(9.0), Float32(1.0))

        # ---- set the content clip rect so caller content is scissored to the viewport ----
        set_clip(content_x, content_y, content_w, view_h)
        return Tuple[Float32, Float32, Float32, Float32](content_x, content_y, content_w, view_h)

    # end_window: clear the clip rect (no-op if collapsed but harmless to always clear).
    def end_window(self):
        clear_clip()


# ============================================================================
# SELF-TEST  (compiled + run headlessly by the MAIN LOOP — do NOT run it yourself)
#
# Drag widgets need: frame 1 = press over the region (latch active + snapshot orig), frame 2 = move
# (apply delta from the snapshot). Every print line's EXACT expected value is derived from first
# principles in the comment beside it. Every visual path issues real draw_* calls (render gate).
# ============================================================================
def main() raises:
    var ctx = WinCtx()
    var pass_all = True

    # Layout constants used below: TITLE_H=24, GRIP=14, SB_W=12, LINE=20, MIN_W=120, MIN_H=80,
    # CARET_BOX=20.

    # ===================================================================
    # 1) MOVE: window at (100,100,300,400). Grab the TITLE BAR (avoiding the caret box at
    #    x in [100,120) and the scrollbar; title bar spans y in [100,124)). Press at (150,112),
    #    then move by (+25,+15) to (175,127).
    #    frame 1: press @ (150,112) down -> DRAG_MOVE, orig_x=100, orig_y=100, press=(150,112).
    #    frame 2: move @ (175,127) down -> dx=25, dy=15 -> x=100+25=125, y=100+15=115.
    #    EXPECT: win.x=125.0, win.y=115.0  (moved EXACTLY +25,+15).
    # ===================================================================
    var w1 = Window(1, String("Inspector"), Float32(100.0), Float32(100.0), Float32(300.0), Float32(400.0))
    var content_h1 = Float32(800.0)        # taller than the viewport (376) so a scrollbar exists
    ctx.update(w1, content_h1, 150, 112, True, Float32(0.0))     # press on title bar
    ctx.update(w1, content_h1, 175, 127, True, Float32(0.0))     # drag +25,+15
    print("WINDOW_SELFTEST: move_x=", w1.x)        # EXPECT: move_x= 125.0
    print("WINDOW_SELFTEST: move_y=", w1.y)        # EXPECT: move_y= 115.0
    if w1.x != Float32(125.0):
        pass_all = False
    if w1.y != Float32(115.0):
        pass_all = False
    # release to clear capture (window now at 125,115,300,400)
    ctx.update(w1, content_h1, 175, 127, False, Float32(0.0))

    # ===================================================================
    # 2) RESIZE: window now (125,115,300,400). Grip is the bottom-right 14x14 at
    #    (x+w-14, y+h-14) = (125+300-14, 115+400-14) = (411, 501). Press inside the grip @ (415,505),
    #    then move by (+30,+20) to (445,525).
    #    frame 1: press @ (415,505) down -> DRAG_RESIZE, orig_w=300, orig_h=400, press=(415,505).
    #    frame 2: move @ (445,525) down -> rdx=30, rdy=20 -> w=300+30=330, h=400+20=420.
    #    Both > min (MIN_W=120, MIN_H=80) so NO clamping. EXPECT: w=330.0, h=420.0.
    # ===================================================================
    ctx.update(w1, content_h1, 415, 505, True, Float32(0.0))     # press on grip
    ctx.update(w1, content_h1, 445, 525, True, Float32(0.0))     # drag +30,+20
    print("WINDOW_SELFTEST: resize_w=", w1.w)      # EXPECT: resize_w= 330.0
    print("WINDOW_SELFTEST: resize_h=", w1.h)      # EXPECT: resize_h= 420.0
    if w1.w != Float32(330.0):
        pass_all = False
    if w1.h != Float32(420.0):
        pass_all = False
    ctx.update(w1, content_h1, 445, 525, False, Float32(0.0))    # release

    # ===================================================================
    # 2b) RESIZE CLAMP: a fresh small window (10,10,130,90). Drag the grip far up-left to shrink
    #     past the minimum. Grip at (10+130-14, 10+90-14) = (126, 86). Press @ (130,90), then move
    #     to (20,20): rdx = 20-130 = -110, rdy = 20-90 = -70.
    #     nw = 130 + (-110) = 20  -> clamped up to MIN_W=120.
    #     nh = 90  + (-70)  = 20  -> clamped up to MIN_H=80.
    #     EXPECT: w=120.0 (clamped), h=80.0 (clamped).
    # ===================================================================
    var w2 = Window(2, String("Tiny"), Float32(10.0), Float32(10.0), Float32(130.0), Float32(90.0))
    var content_h2 = Float32(50.0)         # shorter than viewport -> no scrollbar (fine for resize)
    ctx.update(w2, content_h2, 130, 90, True, Float32(0.0))      # press on grip
    ctx.update(w2, content_h2, 20, 20, True, Float32(0.0))       # drag far up-left
    print("WINDOW_SELFTEST: clamp_w=", w2.w)       # EXPECT: clamp_w= 120.0
    print("WINDOW_SELFTEST: clamp_h=", w2.h)       # EXPECT: clamp_h= 80.0
    if w2.w != Float32(120.0):
        pass_all = False
    if w2.h != Float32(80.0):
        pass_all = False
    ctx.update(w2, content_h2, 20, 20, False, Float32(0.0))      # release

    # ===================================================================
    # 3) COLLAPSE: window w1 starts NOT collapsed (content_visible True). Click the caret box
    #    (left of the title bar, x in [125,145), y in [115,139)) once: press @ (130,125) then
    #    release @ (130,125). Press+release inside the caret box -> collapsed flips to True.
    #    EXPECT: content_visible_before=True, content_visible_after=False, collapsed_after=True.
    # ===================================================================
    var vis_before = w1.content_visible()
    ctx.update(w1, content_h1, 130, 125, True, Float32(0.0))     # press caret
    ctx.update(w1, content_h1, 130, 125, False, Float32(0.0))    # release caret -> toggle
    var vis_after = w1.content_visible()
    print("WINDOW_SELFTEST: vis_before=", vis_before)    # EXPECT: vis_before= True
    print("WINDOW_SELFTEST: vis_after=", vis_after)      # EXPECT: vis_after= False
    print("WINDOW_SELFTEST: collapsed_after=", w1.collapsed)   # EXPECT: collapsed_after= True
    if vis_before != True:
        pass_all = False
    if vis_after != False:
        pass_all = False
    if w1.collapsed != True:
        pass_all = False
    # un-collapse for the scroll test (toggle again)
    ctx.update(w1, content_h1, 130, 125, True, Float32(0.0))
    ctx.update(w1, content_h1, 130, 125, False, Float32(0.0))
    if w1.content_visible() != True:
        pass_all = False

    # ===================================================================
    # 4) WHEEL SCROLL: window w1 is (125,115,300,420) now, content_visible. viewport height =
    #    h - TITLE_H = 420 - 24 = 396. content_height=800 > 396, so max_scroll = 800 - 396 = 404 > 0.
    #    scroll starts at 0. Inject wheel = -3 (mouse parked NOT on any drag region, e.g. far away
    #    at (5000,5000) so no drag latches — pure wheel).
    #    scroll += (-wheel) * LINE = -(-3) * 20 = 3 * 20 = 60. Clamped to [0,404] -> 60.
    #    EXPECT: scroll_before=0.0, scroll_after=60.0, delta = 3*LINE = 60.0.
    # ===================================================================
    # NOTE: w1.h was changed to 420 by the resize test above; viewport = 420-24 = 396; max_scroll
    # = 800-396 = 404, so the wheel is NOT clamped and the full 60 applies.
    var scroll_before = w1.scroll
    ctx.update(w1, content_h1, 5000, 5000, False, Float32(-3.0))  # wheel=-3, no mouse region
    var scroll_after = w1.scroll
    print("WINDOW_SELFTEST: scroll_before=", scroll_before)   # EXPECT: scroll_before= 0.0
    print("WINDOW_SELFTEST: scroll_after=", scroll_after)     # EXPECT: scroll_after= 60.0
    print("WINDOW_SELFTEST: scroll_delta=", scroll_after - scroll_before)  # EXPECT: scroll_delta= 60.0
    if scroll_before != Float32(0.0):
        pass_all = False
    if scroll_after != Float32(60.0):
        pass_all = False
    if (scroll_after - scroll_before) != Float32(60.0):
        pass_all = False

    # ===================================================================
    # RENDER-GATE COVERAGE: draw the window frame (title bar + caret + body + scrollbar + grip)
    # via begin_window/end_window so the glReadPixels readback exercises every paint path. Draw it
    # OPEN (content visible) and then a COLLAPSED window so the right-caret + title-only path paints
    # too. The clip rect is set then cleared.
    # ===================================================================
    var cr = ctx.begin_window(w1, content_h1)
    # caller content (one rect inside the clipped viewport, offset by -scroll) — render coverage.
    var cx0 = cr[0]
    var cy0 = cr[1]
    col(120, 90, 60)
    fill(cx0 + Float32(8.0), cy0 + Float32(8.0) - w1.scroll, Float32(120.0), Float32(40.0))
    ctx.end_window()

    var w3 = Window(3, String("Collapsed"), Float32(500.0), Float32(100.0), Float32(220.0), Float32(160.0))
    w3.collapsed = True
    var cr3 = ctx.begin_window(w3, Float32(400.0))   # collapsed -> only title bar paints
    ctx.end_window()
    _ = cr3
    print("WINDOW_SELFTEST: render_coverage_drawn=", True)   # EXPECT: render_coverage_drawn= True

    # ----- summary line the main loop gates on -----
    if pass_all:
        print("WINDOW_SELFTEST: pass (move +25/+15 -> 125.0/115.0, resize +30/+20 -> 330.0/420.0, "
              + "clamp -> 120.0/80.0, collapse True->False, wheel -3 -> scroll 60.0)")
    else:
        print("WINDOW_SELFTEST: fail")
