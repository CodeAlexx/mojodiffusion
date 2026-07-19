# popup.mojo — WORKSTREAM: POPUPS & MODALS for Mojo 1.0.0b1, drawn on the MojoGUI C rendering
# backend (librender.so) via external_call ONLY. No backend change.
#
# Dear-immediate-mode floating popups + modal dialogs:
#   * open_popup(id)              — request a popup to open (latches PopupCtx.open_id = id).
#   * begin_popup(id,x,y,w,h)     — if id is the open popup, draw a framed floating box and return
#                                   True (caller fills it with widgets); else return False.
#   * begin_popup_modal(id,...)   — same, but FIRST paints a dimmed full-screen backdrop (semi-
#                                   transparent black over the whole window) so the modal grabs focus.
#                                   While a modal is open, a click OUTSIDE does NOT close it.
#   * close_current_popup()       — close whatever popup is open (open_id = -1).
#   * is_popup_open(id)->Bool     — query.
#   * end_popup()                 — symmetry / future framing (no backend scissor to pop).
#   * context_menu(id)            — opens the popup `id` on a RIGHT-click (button 1) edge.
#   * update(mx,my,left_down,right_down,win_w,win_h) — inject one frame of input; derives the
#                                   left/right click+release EDGES and remembers window size so the
#                                   modal backdrop can cover it. For a NON-modal popup, a left-click
#                                   OUTSIDE the popup's last-drawn rect closes it.
#
# FFI idioms mirrored EXACTLY from the framebuffer-verified exemplar
#   /home/alex/framepfx-mojo/src/keyframe_editor.mojo  (to_cstr 37-43; col/fill/rect/line_t/txt
#   wrappers 46-63) and the sibling toolkits /tmp/mojoui/core_widgets.mojo and
#   /tmp/mojoui2/widget_pack.mojo (UiContext/Ctx input model: mouse_x/mouse_y:Int,
#   mouse_down/prev_mouse_down/mouse_clicked/mouse_released:Bool, set_input edge derivation,
#   point_in_rect/clampf, begin_frame hot reset). PopupCtx is the SAME context-struct shape and
#   ADDS right-button edge fields + open_id/backdrop/window-size/last-popup-rect state.
#
# INJECTABLE INPUT is mandatory: the gated logic path reads mouse + button state ONLY from
# PopupCtx fields (set by update()). It never calls get_mouse_*/get_mouse_button_state inside the
# gated path. A SEPARATE pump_live() binds the live backend for production frames.
#
# NO `from math` import: the windowed FFI binary cannot link transcendentals (sincos@GLIBC).
# Only builtin min/max/abs/Int()/Float32() arithmetic is used (same constraint as the exemplars).

from std.ffi import external_call
from std.memory import UnsafePointer, alloc
from builtin.type_aliases import MutExternalOrigin


# ============================================================================
# C backend helpers  (mirror keyframe_editor.mojo:37-63 / widget_pack.mojo:32-82 exactly)
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
    # 0..255 opaque convenience wrapper (keyframe_editor.mojo:46-47)
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


# get_text_size(const char*, float size, float* outW, float* outH) -> (w, h) in pixels.
def measure_text(s: String, size: Float32) -> Tuple[Float32, Float32]:
    var wbuf = alloc[Float32](1)
    var hbuf = alloc[Float32](1)
    wbuf[0] = Float32(0.0)
    hbuf[0] = Float32(0.0)
    _ = external_call["get_text_size", Int32](to_cstr(s), size, wbuf, hbuf)
    var w = wbuf[0]
    var h = hbuf[0]
    return Tuple[Float32, Float32](w, h)


# ============================================================================
# geometry helpers (mirror core_widgets.mojo:88-97 / widget_pack.mojo:87-96)
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
# PopupCtx — the immediate-mode popup/modal core.
#
# SAME context-struct shape as core_widgets.UiContext / widget_pack.Ctx (carries the per-frame LEFT
# mouse-button state with press/release edges + hot_id/active_id), PLUS:
#
#   ---- RIGHT button (for context_menu) ----
#   right_down       : Bool  is the right button held THIS frame.
#   prev_right_down  : Bool  was it held LAST frame (for edge detection).
#   right_clicked    : Bool  pressed-this-frame edge = right_down and not prev_right_down.
#   right_released   : Bool  released-this-frame edge = prev_right_down and not right_down.
#
#   ---- popup state ----
#   open_id          : Int   id of the currently OPEN popup, or -1 if none. (spec default -1)
#   modal            : Bool  is the currently open popup a MODAL (drew a backdrop)?
#   backdrop_drawn   : Bool  did the LAST begin_popup_modal paint the full-screen backdrop this
#                            frame? (self-test gate; recomputed each modal begin).
#   win_w, win_h     : Int   window size (from update()) so the modal backdrop covers the whole
#                            window and the "click outside" test can reason about the screen.
#   popup_x/y/w/h    : Float32  rect of the popup as drawn by the LAST begin_popup(_modal) call.
#                            update() uses this to decide whether a left-click landed OUTSIDE a
#                            NON-modal popup (which closes it).
#   has_popup_rect   : Bool  True once a popup rect has been recorded this session (so update()
#                            doesn't treat an as-yet-undrawn popup's stale rect as the screen).
#
# Popup ids are EXPLICIT (passed by the caller per call-site) — never hashed from labels
# (matches core_widgets / widget_pack).
#
# CLOSE-ON-CLICK-OUTSIDE ORDERING CONTRACT (mirrors immediate-mode GUI's deferred close):
#   The "click outside closes a non-modal popup" decision is made in update() using the popup rect
#   recorded by the PREVIOUS frame's begin_popup. So the canonical per-frame order is:
#     update(...)            # may close a non-modal popup if last frame's click was outside its rect
#     if begin_popup(id,...): ...widgets...; end_popup()   # (re)draws + records this frame's rect
#   The self-test below follows that order exactly.
# ============================================================================
struct PopupCtx(Copyable, Movable):
    # ---- LEFT button input (mirrors core_widgets.UiContext) ----
    var mouse_x: Int
    var mouse_y: Int
    var mouse_down: Bool
    var prev_mouse_down: Bool
    var mouse_clicked: Bool
    var mouse_released: Bool
    var hot_id: Int
    var active_id: Int

    # ---- RIGHT button input (context-menu trigger) ----
    var right_down: Bool
    var prev_right_down: Bool
    var right_clicked: Bool
    var right_released: Bool

    # ---- popup / modal state ----
    var open_id: Int
    var modal: Bool
    var backdrop_drawn: Bool
    var win_w: Int
    var win_h: Int
    var popup_x: Float32
    var popup_y: Float32
    var popup_w: Float32
    var popup_h: Float32
    var has_popup_rect: Bool

    def __init__(out self):
        self.mouse_x = 0
        self.mouse_y = 0
        self.mouse_down = False
        self.prev_mouse_down = False
        self.mouse_clicked = False
        self.mouse_released = False
        self.hot_id = -1
        self.active_id = -1

        self.right_down = False
        self.prev_right_down = False
        self.right_clicked = False
        self.right_released = False

        self.open_id = -1                 # spec: PopupCtx(open_id = -1) -> nothing open
        self.modal = False
        self.backdrop_drawn = False
        self.win_w = 0
        self.win_h = 0
        self.popup_x = Float32(0.0)
        self.popup_y = Float32(0.0)
        self.popup_w = Float32(0.0)
        self.popup_h = Float32(0.0)
        self.has_popup_rect = False

    # ------------------------------------------------------------------
    # INJECTABLE input (mandatory for headless verification).
    #
    # update(mx,my,left_down,right_down,win_w,win_h): inject ONE frame of input.
    #   1. derive LEFT click/release edges from prev_mouse_down (mirrors core_widgets.set_input).
    #   2. derive RIGHT click/release edges from prev_right_down (same edge logic).
    #   3. remember window size for the modal backdrop + screen reasoning.
    #   4. CLOSE-ON-CLICK-OUTSIDE: if a NON-modal popup is open AND we have its drawn rect AND a
    #      LEFT click landed this frame OUTSIDE that rect -> close it (open_id = -1). Modals are
    #      immune (a click outside a modal is swallowed, never closes it).
    #   5. advance prev_* so the NEXT update computes edges correctly.
    # ------------------------------------------------------------------
    def update(
        mut self,
        mx: Int,
        my: Int,
        left_down: Bool,
        right_down: Bool,
        win_w: Int,
        win_h: Int,
    ):
        # 1) LEFT edges
        self.mouse_clicked = left_down and not self.prev_mouse_down
        self.mouse_released = self.prev_mouse_down and not left_down
        # 2) RIGHT edges
        self.right_clicked = right_down and not self.prev_right_down
        self.right_released = self.prev_right_down and not right_down
        # store positions/state
        self.mouse_x = mx
        self.mouse_y = my
        self.mouse_down = left_down
        self.right_down = right_down
        # 3) window size
        self.win_w = win_w
        self.win_h = win_h

        # 4) click-outside closes a NON-modal popup (deferred: uses last frame's drawn rect)
        if self.open_id != -1 and not self.modal and self.has_popup_rect:
            if self.mouse_clicked:
                var inside = point_in_rect(
                    Float32(mx), Float32(my),
                    self.popup_x, self.popup_y, self.popup_w, self.popup_h,
                )
                if not inside:
                    self.open_id = -1
                    self.modal = False
                    self.has_popup_rect = False

        # 5) advance edge-detection state
        self.prev_mouse_down = left_down
        self.prev_right_down = right_down
        # hot tracking is per-frame; reset like begin_frame in the exemplars
        self.hot_id = -1

    # ---- LIVE input from the backend (production path; SEPARATE from the gated logic) ----
    # Binds get_mouse_* / get_mouse_button_state(0=left,1=right) and forwards to update(). The
    # gated logic above never calls these directly (headless-verifiable contract).
    def pump_live(mut self, win_w: Int, win_h: Int):
        var mx = Int(external_call["get_mouse_x", Int32]())
        var my = Int(external_call["get_mouse_y", Int32]())
        var lb = Int(external_call["get_mouse_button_state", Int32](Int32(0)))
        var rb = Int(external_call["get_mouse_button_state", Int32](Int32(1)))
        self.update(mx, my, lb != 0, rb != 0, win_w, win_h)

    # ==================================================================
    # popup open / query / close
    # ==================================================================

    # open_popup(id): request popup `id` to open. Latches open_id; clears modal/backdrop so a
    # following begin_popup draws a plain floating box (begin_popup_modal re-sets modal=True).
    def open_popup(mut self, id: Int):
        self.open_id = id
        self.modal = False
        self.backdrop_drawn = False
        self.has_popup_rect = False        # rect not yet drawn this open-session

    # is_popup_open(id): True iff `id` is the currently open popup.
    def is_popup_open(self, id: Int) -> Bool:
        return self.open_id == id

    # close_current_popup(): close whatever popup is open.
    def close_current_popup(mut self):
        self.open_id = -1
        self.modal = False
        self.backdrop_drawn = False
        self.has_popup_rect = False

    # context_menu(id): RIGHT-click opens popup `id`. Returns True the frame it opened (so the
    # caller can, e.g., position the popup at the cursor). Uses the right_clicked EDGE from update()
    # so a held right button opens it exactly once.
    def context_menu(mut self, id: Int) -> Bool:
        if self.right_clicked:
            self.open_popup(id)
            return True
        return False

    # ==================================================================
    # begin_popup / begin_popup_modal / end_popup
    # ==================================================================

    # _draw_frame(x,y,w,h): the floating box visual shared by both popup kinds — a filled panel,
    # a 1px outer border, and a subtle inner top highlight line (so it reads as raised). Records
    # the rect into the ctx so update() can do click-outside hit-testing next frame.
    def _draw_frame(mut self, x: Float32, y: Float32, w: Float32, h: Float32):
        # drop a faint shadow offset (cheap depth cue)
        col(10, 10, 12)
        fill(x + Float32(3.0), y + Float32(3.0), w, h)
        # panel body
        col(46, 46, 54)
        fill(x, y, w, h)
        # outer border
        col(12, 12, 16)
        rect(x, y, w, h)
        # inner top highlight
        col(80, 80, 92)
        line_t(x + Float32(1.0), y + Float32(1.0), x + w - Float32(1.0), y + Float32(1.0), Float32(1.0))
        # record drawn rect for click-outside testing
        self.popup_x = x
        self.popup_y = y
        self.popup_w = w
        self.popup_h = h
        self.has_popup_rect = True

    # begin_popup(id,x,y,w,h) -> Bool. If `id` is open, draw a framed floating box and return True
    # (the caller then fills it and calls end_popup). Otherwise return False (draw nothing).
    # Non-modal: closing on click-outside is handled in update() using the rect recorded here.
    def begin_popup(mut self, id: Int, x: Float32, y: Float32, w: Float32, h: Float32) -> Bool:
        if self.open_id != id:
            return False
        # this begin is non-modal; if it was previously marked modal for some other reason, the
        # explicit non-modal begin wins for THIS popup.
        self.modal = False
        self.backdrop_drawn = False
        self._draw_frame(x, y, w, h)
        return True

    # begin_popup_modal(id,x,y,w,h) -> Bool. Same as begin_popup but FIRST paints a dimmed
    # full-screen backdrop (semi-transparent black over [0,0,win_w,win_h]) so the modal is
    # focus-grabbing, then draws the framed box. Sets modal=True (so update() never closes it on a
    # click outside) and backdrop_drawn=True (self-test gate). Returns True if open.
    def begin_popup_modal(mut self, id: Int, x: Float32, y: Float32, w: Float32, h: Float32) -> Bool:
        if self.open_id != id:
            self.backdrop_drawn = False
            return False
        # ---- dimmed full-screen backdrop (alpha < 1 => semi-transparent black) ----
        # Cover the whole window. If win size is unknown (0), fall back to a generous default so the
        # backdrop is still visible (the render gate must see pixels).
        var bw = Float32(self.win_w)
        var bh = Float32(self.win_h)
        if bw <= Float32(0.0):
            bw = Float32(1920.0)
        if bh <= Float32(0.0):
            bh = Float32(1080.0)
        set_color(Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.55))   # 55% black dim
        fill(Float32(0.0), Float32(0.0), bw, bh)
        self.backdrop_drawn = True
        self.modal = True
        # ---- the modal box on top of the backdrop ----
        self._draw_frame(x, y, w, h)
        return True

    # end_popup(): symmetry / future framing. No backend scissor to pop; present so call sites read
    # like immediate-mode GUI (BeginPopup .. EndPopup).
    def end_popup(self):
        pass

    # ---- helper: is the cursor inside the current popup rect? (caller convenience) ----
    def mouse_in_popup(self) -> Bool:
        if not self.has_popup_rect:
            return False
        return point_in_rect(
            Float32(self.mouse_x), Float32(self.mouse_y),
            self.popup_x, self.popup_y, self.popup_w, self.popup_h,
        )


# ============================================================================
# SELF-TEST  (compiled + run headlessly by the MAIN LOOP — do NOT run it yourself)
#
# Every print line is gated by the main loop; the EXACT expected value is in a comment, derived
# from first principles. Every visual element issues real draw_* calls (the render gate sees the
# backdrop fill + the popup frame fills/rects/lines).
#
# Per-frame order (the close-on-click-outside contract): update(...) FIRST, then begin_popup(...).
# update() uses the popup rect recorded by the PREVIOUS frame's begin_popup, so closing-on-outside
# is observed on the frame AFTER the outside click is injected.
# ============================================================================
def main() raises:
    var ctx = PopupCtx()
    var pass_all = True

    # popup geometry used throughout the non-modal test: rect (300, 200, 160, 120).
    var px = Float32(300.0)
    var py = Float32(200.0)
    var pw = Float32(160.0)
    var ph = Float32(120.0)
    # window size used for the backdrop test.
    var WW = 1280
    var WH = 720

    # ---------------------------------------------------------------
    # 0) FRESH CONTEXT: nothing open.
    #    PopupCtx() sets open_id = -1 -> is_popup_open(7) is False.
    # ---------------------------------------------------------------
    var t0 = ctx.is_popup_open(7)
    print("POPUP_SELFTEST: initial_open=", t0)            # EXPECT False
    if t0 != False:
        pass_all = False

    # ---------------------------------------------------------------
    # 1) OPEN: open_popup(7) -> is_popup_open(7) True.
    # ---------------------------------------------------------------
    ctx.open_popup(7)
    var t1 = ctx.is_popup_open(7)
    print("POPUP_SELFTEST: after_open=", t1)              # EXPECT True
    if t1 != True:
        pass_all = False

    # ---------------------------------------------------------------
    # 2) CLICK INSIDE -> popup STAYS open.
    #    Frame A: draw the popup once so its rect (300,200,160,120) is recorded for next frame's
    #             update() hit-test. No button this frame.
    #    Frame B: update() with a LEFT click INSIDE the rect (center = 300+80=380, 200+60=260).
    #             Because the click is inside (and non-modal), update() does NOT close it.
    #             Then begin_popup re-draws/records the rect again.
    # Frame A: no input, just draw to record rect.
    ctx.update(0, 0, False, False, WW, WH)
    var openA = ctx.begin_popup(7, px, py, pw, ph)
    if openA:
        ctx.end_popup()
    # Frame B: left-click inside center (380, 260).
    ctx.update(380, 260, True, False, WW, WH)
    var insideStays = ctx.is_popup_open(7)
    var openB = ctx.begin_popup(7, px, py, pw, ph)
    if openB:
        ctx.end_popup()
    print("POPUP_SELFTEST: click_inside_stays_open=", insideStays)   # EXPECT True
    if insideStays != True:
        pass_all = False
    # release the left button (edge hygiene; no state change expected since click was inside).
    ctx.update(380, 260, False, False, WW, WH)
    var stillOpen = ctx.begin_popup(7, px, py, pw, ph)
    if stillOpen:
        ctx.end_popup()
    if ctx.is_popup_open(7) != True:
        pass_all = False

    # ---------------------------------------------------------------
    # 3) CLICK OUTSIDE (non-modal) -> popup CLOSES.
    #    The rect (300,200,160,120) was recorded by the begin_popup in step 2's release frame.
    #    Now inject a LEFT click OUTSIDE it (e.g. (50, 50) — well clear of the rect). update()
    #    sees mouse_clicked True + outside -> sets open_id = -1. is_popup_open(7) becomes False.
    #    point (50,50): inside x in [300,460)? 50<300 -> NO. So it is OUTSIDE. ✓
    ctx.update(50, 50, True, False, WW, WH)
    var afterOutside = ctx.is_popup_open(7)
    # begin_popup now returns False (nothing open) and draws nothing.
    var openAfterOutside = ctx.begin_popup(7, px, py, pw, ph)
    if openAfterOutside:
        ctx.end_popup()
    print("POPUP_SELFTEST: click_outside_closes=", afterOutside)     # EXPECT False
    if afterOutside != False:
        pass_all = False
    if openAfterOutside != False:
        pass_all = False
    # release to clear the left edge.
    ctx.update(50, 50, False, False, WW, WH)

    # ---------------------------------------------------------------
    # 4) MODAL: open a modal popup -> backdrop_drawn True; a click OUTSIDE does NOT close it.
    #    open_popup(8); begin_popup_modal(8,...) paints the full-screen backdrop (sets
    #    backdrop_drawn=True, modal=True) then the box. Then a LEFT click OUTSIDE the modal box
    #    (50,50) must NOT close it (modal is immune in update()).
    ctx.open_popup(8)
    # Frame M1: draw the modal (records backdrop_drawn + modal=True + rect).
    ctx.update(0, 0, False, False, WW, WH)
    var modalOpen = ctx.begin_popup_modal(8, px, py, pw, ph)
    var backdrop = ctx.backdrop_drawn
    if modalOpen:
        ctx.end_popup()
    print("POPUP_SELFTEST: modal_open=", modalOpen)        # EXPECT True
    print("POPUP_SELFTEST: backdrop_drawn=", backdrop)     # EXPECT True
    if modalOpen != True:
        pass_all = False
    if backdrop != True:
        pass_all = False

    # Frame M2: LEFT click OUTSIDE the modal box (50,50). Modal -> NOT closed.
    ctx.update(50, 50, True, False, WW, WH)
    var modalStillOpen = ctx.is_popup_open(8)
    var modalRedraw = ctx.begin_popup_modal(8, px, py, pw, ph)
    if modalRedraw:
        ctx.end_popup()
    print("POPUP_SELFTEST: modal_click_outside_stays=", modalStillOpen)  # EXPECT True
    if modalStillOpen != True:
        pass_all = False
    # release + explicitly close the modal via close_current_popup -> now closed.
    ctx.update(50, 50, False, False, WW, WH)
    ctx.close_current_popup()
    var modalClosed = ctx.is_popup_open(8)
    print("POPUP_SELFTEST: modal_after_close=", modalClosed)  # EXPECT False
    if modalClosed != False:
        pass_all = False

    # ---------------------------------------------------------------
    # 5) CONTEXT MENU: a RIGHT-click opens popup id 9.
    #    Start closed. Frame R1: right button DOWN (right_clicked edge True) -> context_menu(9)
    #    returns True and opens popup 9. is_popup_open(9) becomes True.
    var ctxBefore = ctx.is_popup_open(9)
    ctx.update(640, 360, False, True, WW, WH)   # right button pressed (left up)
    var opened = ctx.context_menu(9)
    var ctxAfter = ctx.is_popup_open(9)
    print("POPUP_SELFTEST: ctxmenu_before=", ctxBefore)   # EXPECT False
    print("POPUP_SELFTEST: ctxmenu_opened=", opened)      # EXPECT True
    print("POPUP_SELFTEST: ctxmenu_after=", ctxAfter)     # EXPECT True
    if ctxBefore != False:
        pass_all = False
    if opened != True:
        pass_all = False
    if ctxAfter != True:
        pass_all = False
    # draw the just-opened context popup once (render-gate coverage of begin_popup frame path).
    var ctxDraw = ctx.begin_popup(9, Float32(640.0), Float32(360.0), Float32(140.0), Float32(90.0))
    if ctxDraw:
        ctx.end_popup()
    # right button held next frame must NOT re-open (edge only fires once).
    ctx.update(640, 360, False, True, WW, WH)
    var openedAgain = ctx.context_menu(9)
    print("POPUP_SELFTEST: ctxmenu_no_retrigger=", openedAgain)  # EXPECT False
    if openedAgain != False:
        pass_all = False
    # release right + close.
    ctx.update(640, 360, False, False, WW, WH)
    ctx.close_current_popup()

    # ----- summary line the main loop gates on -----
    if pass_all:
        print("POPUP_SELFTEST: pass (initial False, open True, click-inside True, "
              + "click-outside False, modal True + backdrop True + modal-outside True, "
              + "ctxmenu False->True->True, no-retrigger False)")
    else:
        print("POPUP_SELFTEST: fail")
