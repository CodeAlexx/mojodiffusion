# game_screen_live.mojo — MOJO QUEST, LIVE + interactive.
# FIX: the widget contexts are created ONCE and persist across frames (immediate-mode
# click/drag edge-detection needs prev_mouse_down + active_id to survive frame-to-frame).
from std.ffi import external_call
from std.memory import UnsafePointer, alloc
from builtin.type_aliases import MutExternalOrigin
from core_widgets import UiContext
from widget_pack import Ctx, progress_bar, tab, tree_node, color_button
from widget_variants import VCtx, selectable, list_box, arrow_button
from tables import Table, begin_table, table_setup_column, table_headers_row, table_next_row, table_next_column, end_table
from draw_list_paths import DrawList, Col4
from color_picker import hsv_to_rgb
from input_text import TextInput

comptime W = 1100
comptime H = 680


fn to_cstr(s: String) raises -> UnsafePointer[Int8, MutExternalOrigin]:
    var b = s.as_bytes()
    var buf = alloc[Int8](len(b) + 1)
    for i in range(len(b)):
        buf[i] = Int8(b[i])
    buf[len(b)] = 0
    return buf


fn txt(s: String, x: Float32, y: Float32, size: Float32) raises:
    _ = external_call["draw_text", Int32](to_cstr(s), x, y, size)


fn tcol(r: Float32, g: Float32, b: Float32):
    _ = external_call["set_color", Int32](r, g, b, Float32(1.0))


def main() raises:
    var ir = external_call["initialize_gl_context", Int32](Int32(W), Int32(H), to_cstr(String("MOJO QUEST")))
    if Int(ir) != 0:
        print("FAIL gl init", ir); return
    _ = external_call["load_default_font", Int32]()

    # ---- contexts created ONCE (persist across frames) ----
    var ui = UiContext()
    var wp = Ctx()
    var vc = VCtx()

    # ---- persistent UI state ----
    var s1 = Float32(72.0); var s2 = Float32(48.0); var s3 = Float32(91.0)
    var cb1 = True; var cb2 = False
    var sel = 1
    var ab_open = True
    var active_tab = 0

    # editable text field (seed with "Hero" — click it isn't needed; it's focused)
    var name_field = TextInput()
    var seed = List[Int]()
    var sb = String("Hero").as_bytes()
    for i in range(len(sb)): seed.append(Int(sb[i]))
    name_field.edit(seed, False)

    # player dye colour (recolours the orb rings); -1 = nothing picked yet
    var dye_r = Float32(1.0); var dye_g = Float32(0.5); var dye_b = Float32(0.5)
    var dye_i = -1

    var names = List[String](); names.append(String("Aria")); names.append(String("Borin")); names.append(String("Cyra")); names.append(String("You"))
    var scores = List[String](); scores.append(String("9810")); scores.append(String("8740")); scores.append(String("8120")); scores.append(String("7990"))
    var inv = List[String]()
    inv.append(String("Excalibur")); inv.append(String("Health Potion x5")); inv.append(String("Dragon Scale"))
    inv.append(String("Gold Key")); inv.append(String("Mana Crystal")); inv.append(String("Ancient Map"))

    var frame_no = 0
    while Int(external_call["should_close_window", Int32]()) == 0:
        _ = external_call["poll_events", Int32]()
        var mx = Int(external_call["get_mouse_x", Int32]())
        var my = Int(external_call["get_mouse_y", Int32]())
        var down = Int(external_call["get_mouse_button_state", Int32](Int32(0))) != 0
        ui.begin_frame(); ui.set_input(mx, my, down)
        wp.begin_frame(); wp.set_input(mx, my, down)
        vc.begin_frame(); vc.set_input(mx, my, down)

        _ = external_call["frame_begin", Int32]()

        # ---- background + panels + HUD (one batched DrawList) ----
        var PT = Float32(64.0)              # panel top
        var PB = Float32(H) - 64.0          # panel bottom
        var PH = PB - PT
        var dl = DrawList()
        dl.add_rect_filled(0.0, 0.0, Float32(W), Float32(H), Col4(0.07, 0.08, 0.12, 1.0))
        dl.add_rect_filled(0.0, 0.0, Float32(W), 52.0, Col4(0.16, 0.13, 0.22, 1.0))
        dl.add_rect_filled(16.0, PT, 300.0, PH, Col4(0.11, 0.12, 0.17, 1.0))           # left
        dl.add_rect_filled(332.0, PT, 440.0, PH, Col4(0.11, 0.12, 0.17, 1.0))          # center
        dl.add_rect_filled(788.0, PT, 296.0, PH, Col4(0.11, 0.12, 0.17, 1.0))          # right
        # orbs
        dl.add_circle_filled(80.0, 204.0, 34.0, Col4(0.85, 0.15, 0.15, 1.0), 28)
        dl.add_circle(80.0, 204.0, 34.0, 4.0, Col4(dye_r, dye_g, dye_b, 1.0), 28)      # ring = dye
        dl.add_circle_filled(190.0, 204.0, 34.0, Col4(0.15, 0.35, 0.9, 1.0), 28)
        dl.add_circle(190.0, 204.0, 34.0, 4.0, Col4(dye_r, dye_g, dye_b, 1.0), 28)     # ring = dye
        # minimap top-right
        dl.add_circle_filled(Float32(W) - 56.0, 88.0, 28.0, Col4(0.05, 0.1, 0.08, 1.0), 24)
        dl.add_circle(Float32(W) - 56.0, 88.0, 28.0, 2.0, Col4(0.3, 0.8, 0.4, 1.0), 24)
        var _t = dl.flush()

        tcol(0.95, 0.85, 0.4); txt(String("* MOJO QUEST"), 18.0, 15.0, 22.0)

        # ---- panel headers (aligned baseline y=76) ----
        tcol(0.62, 0.66, 0.82)
        txt(String("VITALS"), 30.0, 78.0, 13.0)
        txt(String("LEADERBOARD"), 800.0, 78.0, 13.0)

        # ---- LEFT: name field (top) + bars + dye ----
        name_field.pump_live()
        tcol(0.62, 0.66, 0.82); txt(String("Character Name  (type / backspace)"), 30.0, 96.0, 12.0)
        name_field.draw(30.0, 108.0, 256.0, 28.0)

        tcol(0.85, 0.88, 0.95)
        txt(String("HP"), 30.0, 260.0, 13.0); txt(String("MP"), 30.0, 286.0, 13.0); txt(String("XP"), 30.0, 312.0, 13.0)
        progress_bar(s1 / 100.0, 64.0, 256.0, 236.0, 16.0)
        progress_bar(s2 / 100.0, 64.0, 282.0, 236.0, 16.0)
        progress_bar(s3 / 100.0, 64.0, 308.0, 236.0, 16.0)
        tcol(0.8, 0.82, 0.9); txt(String("Lv 27   Paladin"), 30.0, 346.0, 14.0)
        tcol(0.62, 0.66, 0.82); txt(String("Dye  (click to recolour the orb rings)"), 30.0, 378.0, 12.0)
        for i in range(8):
            var hsv = hsv_to_rgb(Float32(i) * 45.0, Float32(1.0), Float32(1.0))
            if color_button(wp, 800 + i, hsv[0], hsv[1], hsv[2], 70.0 + Float32(i) * 26.0, 392.0, 22.0, 22.0):
                dye_r = hsv[0]; dye_g = hsv[1]; dye_b = hsv[2]; dye_i = i
        if dye_i >= 0:
            var sx = 70.0 + Float32(dye_i) * 26.0
            _ = external_call["set_color", Int32](Float32(1.0), Float32(1.0), Float32(1.0), Float32(1.0))
            _ = external_call["draw_rectangle", Int32](Float32(sx - 2.0), Float32(390.0), Float32(26.0), Float32(26.0))

        # ---- CENTER: tabs (header row) + sliders + checks + tree ----
        if tab(wp, 10, String("Stats"), 340.0, 74.0, 90.0, 26.0, active_tab == 0): active_tab = 0
        if tab(wp, 11, String("Skills"), 434.0, 74.0, 90.0, 26.0, active_tab == 1): active_tab = 1
        if tab(wp, 12, String("Inventory"), 528.0, 74.0, 104.0, 26.0, active_tab == 2): active_tab = 2

        tcol(0.85, 0.88, 0.95)
        txt(String("STR"), 344.0, 122.0, 13.0); txt(String("DEX"), 344.0, 154.0, 13.0); txt(String("INT"), 344.0, 186.0, 13.0)
        _ = ui.slider_float(20, 388.0, 120.0, 250.0, 16.0, s1, Float32(0.0), Float32(100.0))
        _ = ui.slider_float(21, 388.0, 152.0, 250.0, 16.0, s2, Float32(0.0), Float32(100.0))
        _ = ui.slider_float(22, 388.0, 184.0, 250.0, 16.0, s3, Float32(0.0), Float32(100.0))
        _ = ui.checkbox(30, String("Hardcore"), 344.0, 218.0, cb1)
        _ = ui.checkbox(31, String("Permadeath"), 486.0, 218.0, cb2)
        _ = tree_node(wp, 40, String("v Abilities"), 344.0, 252.0, 280.0, ab_open)
        if ab_open:
            tcol(0.75, 0.78, 0.88)
            txt(String("  - Holy Strike      Lv 5"), 354.0, 280.0, 13.0)
            txt(String("  - Divine Shield    Lv 3"), 354.0, 302.0, 13.0)
            txt(String("  - Lay on Hands     Lv 8"), 354.0, 324.0, 13.0)

        # ---- RIGHT: leaderboard + inventory ----
        var tb = begin_table(50, 800.0, 98.0, 272.0, 140.0, 3)
        table_setup_column(tb, String("#"), Float32(40.0), True)
        table_setup_column(tb, String("Name"), Float32(1.0), False)
        table_setup_column(tb, String("Score"), Float32(1.0), False)
        table_headers_row(tb)
        for r in range(4):
            table_next_row(tb)
            var c0 = table_next_column(tb); tcol(0.7, 0.72, 0.8); txt(String(r + 1), c0[0] + 6.0, c0[1] + 5.0, 12.0)
            var c1 = table_next_column(tb); tcol(0.85, 0.88, 0.95); txt(names[r], c1[0] + 6.0, c1[1] + 5.0, 12.0)
            var c2 = table_next_column(tb); tcol(0.6, 0.9, 0.6); txt(scores[r], c2[0] + 6.0, c2[1] + 5.0, 12.0)
        end_table(tb)
        tcol(0.62, 0.66, 0.82); txt(String("INVENTORY"), 800.0, 262.0, 13.0)
        _ = list_box(vc, 60, 800.0, 282.0, 272.0, 160.0, inv, sel)
        _ = arrow_button(vc, 70, 800.0, 456.0, 24.0, 2)
        _ = arrow_button(vc, 71, 832.0, 456.0, 24.0, 3)

        # ---- bottom buttons ----
        _ = ui.button(80, String("Save"), 18.0, Float32(H) - 46.0, 110.0, 32.0)
        _ = ui.button(81, String("Apply"), 138.0, Float32(H) - 46.0, 110.0, 32.0)
        if ui.button(82, String("Quit"), Float32(W) - 128.0, Float32(H) - 46.0, 110.0, 32.0):
            break

        # ---- popup notification (bottom-center toast — clear of the panels) ----
        var px0 = Float32(W) / 2.0 - 120.0
        var py0 = Float32(H) - 92.0
        var dl2 = DrawList()
        dl2.add_rect_filled(px0, py0, 240.0, 32.0, Col4(0.12, 0.22, 0.14, 0.97))
        dl2.add_rect(px0, py0, 240.0, 32.0, 2.0, Col4(0.3, 0.8, 0.4, 1.0))
        var _u = dl2.flush()
        tcol(0.7, 0.95, 0.7); txt(String("Quest Complete!  +500 XP"), px0 + 12.0, py0 + 21.0, 13.0)

        if frame_no == 0:
            _ = external_call["fpx_dump_fb", Int32](Int32(W), Int32(H), to_cstr(String("/tmp/game_live.ppm")))
        frame_no += 1

        _ = external_call["frame_end", Int32]()

    _ = external_call["cleanup_gl", Int32]()
    print("GAME_LIVE: window closed after", frame_no, "frames")
