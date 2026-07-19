# timeline_tracks.mojo — a multi-track timeline DATA MODEL + GEOMETRY for Mojo 1.0.0b1.
#
# This module is PURE LOGIC / GEOMETRY: a stacked-track layout (video on top, audio below)
# plus the frame<->pixel mapping needed to hit-test clips. It owns NO GL state and makes
# NO external_call / draw calls of any kind — it is fully headless by construction (the
# "draw=False" headless pattern from media_widgets.Toolbar taken to its limit: there is no
# draw path to gate). The interactive UI layer (a timeline widget) consumes these helpers to
# place track rows + clip rects; this file just answers "where is track i?" and "what clip is
# under (mx,my)?".
#
# Coordinate model (matches the surrounding widgets: y grows DOWN, x grows RIGHT):
#   - Tracks are stacked vertically from `base_y` downward, in list order, each `height` px
#     tall, separated by `gap` px. track_top(i) = base_y + sum(height[j]+gap for j<i).
#   - The horizontal axis is FRAMES: x0 is the pixel of frame 0, and `px_per_frame` scales
#     frames -> pixels. A clip on `track` occupies the frame span [t_start, t_start+length)
#     -> pixel span [x0 + t_start*ppf, x0 + (t_start+length)*ppf) on that track's row.
#
# Data model:
#   Track : a horizontal lane (video=0 / audio=1) with mute/lock/solo/enable + pixel height.
#   TClip : a placed media segment (media id, src_in, length frames, t_start frame, track idx).
#
# MOJO 1.0.0b1: `def` (raises only where needed), `comptime` not `alias`, `var`/`mut`, lists
# are move-only (.copy() to read-copy, ^ to move), [a,b] literals (no variadic ctor). Numeric
# conversions explicit (Float32(i), Int(u)). Inline if/else only as a value-ternary.


# ============================================================================
# Track — one horizontal lane in the stack.
#   kind   : 0 = video, 1 = audio  (video lanes are listed first by convention).
#   name   : display label ("V1", "A2", ...).
#   muted  : audio silenced / video hidden in the mix.
#   locked : clips on this track cannot be edited/moved.
#   soloed : only soloed tracks contribute when ANY track is soloed.
#   enabled: lane is active in the render (a disabled lane is skipped entirely).
#   height : the lane's pixel height in the stacked layout.
# ============================================================================
@fieldwise_init
struct Track(Copyable, Movable):
    var kind: Int
    var name: String
    var muted: Bool
    var locked: Bool
    var soloed: Bool
    var enabled: Bool
    var height: Float32


# ============================================================================
# TClip — a media segment placed on a track.
#   media   : id/index of the source media item this clip references.
#   src_in  : in-point (first source frame) inside that media.
#   length  : clip duration in frames (the on-timeline span).
#   t_start : timeline frame where the clip begins.
#   track   : index into the tracks list that this clip lives on.
# A clip covers timeline frames [t_start, t_start + length).
# ============================================================================
@fieldwise_init
struct TClip(Copyable, Movable):
    var media: Int
    var src_in: Int
    var length: Int
    var t_start: Int
    var track: Int


# ============================================================================
# track_top — pixel Y of the TOP edge of track `idx`.
# Sum of (height + gap) of every track ABOVE idx, offset by base_y. The gap is added AFTER
# each track above, so track 0's top is exactly base_y. Out-of-range idx is clamped to the
# end (returns base_y + full stacked extent through the last track) to stay total.
# ============================================================================
def track_top(tracks: List[Track], idx: Int, base_y: Float32, gap: Float32) -> Float32:
    var y = base_y
    var n = len(tracks)
    var stop = idx
    if stop > n:
        stop = n
    if stop < 0:
        stop = 0
    for i in range(stop):
        y = y + tracks[i].height + gap
    return y


# ============================================================================
# total_height — full vertical extent of the stack: sum of all track heights + the gaps
# BETWEEN them ((n-1) gaps for n tracks). Empty stack => 0.
# ============================================================================
def total_height(tracks: List[Track], gap: Float32) -> Float32:
    var n = len(tracks)
    if n == 0:
        return Float32(0.0)
    var h = Float32(0.0)
    for i in range(n):
        h = h + tracks[i].height
    h = h + Float32(n - 1) * gap
    return h


# ============================================================================
# frame_to_x / x_to_frame — the horizontal frame<->pixel mapping (shared with the ruler).
#   x = x0 + frame * px_per_frame ;  frame = (x - x0) / px_per_frame  (truncated)
# ============================================================================
def frame_to_x(frame: Int, x0: Float32, px_per_frame: Float32) -> Float32:
    return x0 + Float32(frame) * px_per_frame


def x_to_frame(x: Float32, x0: Float32, px_per_frame: Float32) -> Int:
    if px_per_frame == Float32(0.0):
        return 0
    return Int((x - x0) / px_per_frame)


# ============================================================================
# track_at — index of the track whose [top, top+height) row contains pixel-y `my`, else -1.
# The inter-track `gap` bands belong to NO track (return -1 there).
# ============================================================================
def track_at(tracks: List[Track], my: Float32, base_y: Float32, gap: Float32) -> Int:
    var y = base_y
    var n = len(tracks)
    for i in range(n):
        var top = y
        var bot = y + tracks[i].height
        if my >= top and my < bot:
            return i
        y = bot + gap
    return -1


# ============================================================================
# clip_at — index into `clips` of the clip under (mx, my), or -1 if none.
# A hit requires BOTH:
#   - my falls inside the clip's track row  [track_top, track_top+height)
#   - mx falls inside the clip's frame span [t_start, t_start+length) mapped to pixels.
# Iterates in list order; LAST match wins so clips drawn on top (later in the list) take
# precedence on overlap. Clips on out-of-range track indices are skipped.
# ============================================================================
def clip_at(
    tracks: List[Track],
    clips: List[TClip],
    mx: Float32,
    my: Float32,
    x0: Float32,
    base_y: Float32,
    gap: Float32,
    px_per_frame: Float32,
) -> Int:
    var hit = -1
    var nt = len(tracks)
    var nc = len(clips)
    for i in range(nc):
        var tr = clips[i].track
        if tr < 0 or tr >= nt:
            continue
        # vertical: inside this clip's track row?
        var top = track_top(tracks, tr, base_y, gap)
        var bot = top + tracks[tr].height
        if my < top or my >= bot:
            continue
        # horizontal: inside [t_start, t_start+length) in pixels?
        var lx = frame_to_x(clips[i].t_start, x0, px_per_frame)
        var rx = frame_to_x(clips[i].t_start + clips[i].length, x0, px_per_frame)
        if mx >= lx and mx < rx:
            hit = i
    return hit


# ============================================================================
# any_soloed — whether ANY track is soloed (gates solo-exclusive mixing).
# ============================================================================
def any_soloed(tracks: List[Track]) -> Bool:
    for i in range(len(tracks)):
        if tracks[i].soloed:
            return True
    return False


# ============================================================================
# track_audible — does track `idx` contribute to the mix? enabled AND not muted, and if any
# track is soloed then this one must be soloed too. Pure boolean policy helper.
# ============================================================================
def track_audible(tracks: List[Track], idx: Int) -> Bool:
    if idx < 0 or idx >= len(tracks):
        return False
    if not tracks[idx].enabled:
        return False
    if tracks[idx].muted:
        return False
    if any_soloed(tracks):
        return tracks[idx].soloed
    return True


# ============================================================================
# SELF-TEST  (compiled + run headlessly by the MAIN LOOP — do NOT run it yourself)
#
# Pure logic / geometry: NO GL, NO external_call anywhere. Builds a known stack of 4 video +
# 3 audio tracks (varied heights) plus a few placed clips and asserts every helper from first
# principles. Each expected number is derived in the comment beside it.
# ============================================================================
def main() raises:
    var pass_all = True

    # ----- build 4 video + 3 audio tracks with VARIED heights -----
    # heights: V1..V4 = 60, 40, 80, 50 ; A1..A3 = 30, 45, 30 ; gap = 4.
    var tracks = List[Track]()
    tracks.append(Track(0, String("V1"), False, False, False, True, Float32(60.0)))
    tracks.append(Track(0, String("V2"), False, False, False, True, Float32(40.0)))
    tracks.append(Track(0, String("V3"), False, False, False, True, Float32(80.0)))
    tracks.append(Track(0, String("V4"), False, False, False, True, Float32(50.0)))
    tracks.append(Track(1, String("A1"), False, False, False, True, Float32(30.0)))
    tracks.append(Track(1, String("A2"), False, False, False, True, Float32(45.0)))
    tracks.append(Track(1, String("A3"), False, False, False, True, Float32(30.0)))

    var base_y = Float32(20.0)
    var gap = Float32(4.0)

    # ----- 1) track_top: STRICTLY INCREASING, and exact per first principles -----
    # top(0) = 20
    # top(1) = 20 + 60 + 4              = 84
    # top(2) = 84 + 40 + 4              = 128
    # top(3) = 128 + 80 + 4             = 212
    # top(4) = 212 + 50 + 4             = 266
    # top(5) = 266 + 30 + 4             = 300
    # top(6) = 300 + 45 + 4             = 349
    var t0 = track_top(tracks, 0, base_y, gap)
    var t1 = track_top(tracks, 1, base_y, gap)
    var t2 = track_top(tracks, 2, base_y, gap)
    var t3 = track_top(tracks, 3, base_y, gap)
    var t4 = track_top(tracks, 4, base_y, gap)
    var t5 = track_top(tracks, 5, base_y, gap)
    var t6 = track_top(tracks, 6, base_y, gap)
    print("TT top0..6 =", t0, t1, t2, t3, t4, t5, t6)
    if t0 != Float32(20.0): pass_all = False
    if t1 != Float32(84.0): pass_all = False
    if t2 != Float32(128.0): pass_all = False
    if t3 != Float32(212.0): pass_all = False
    if t4 != Float32(266.0): pass_all = False
    if t5 != Float32(300.0): pass_all = False
    if t6 != Float32(349.0): pass_all = False
    # strictly increasing across all 7 rows.
    var prev = t0
    var inc_ok = True
    var tops = [t1, t2, t3, t4, t5, t6]
    for i in range(len(tops)):
        if not (tops[i] > prev):
            inc_ok = False
        prev = tops[i]
    if not inc_ok: pass_all = False
    print("TT strictly_increasing =", inc_ok)

    # ----- 2) total_height: sum of heights + (n-1) gaps -----
    # heights sum = 60+40+80+50+30+45+30 = 335 ; gaps = (7-1)*4 = 24 ; total = 359.
    var th = total_height(tracks, gap)
    print("TT total_height =", th, " (expect 359.0)")
    if th != Float32(359.0): pass_all = False
    # bottom of the last track row = top(6) + height(6) = 349 + 30 = 379 = base_y + total (20+359).
    var last_bottom = track_top(tracks, 6, base_y, gap) + tracks[6].height
    if last_bottom != base_y + th: pass_all = False

    # ----- 3) clip_at: build clips and hit-test a known point + empty space -----
    # Horizontal map: x0 = 100, px_per_frame = 2.0  => x = 100 + frame*2.
    var x0 = Float32(100.0)
    var ppf = Float32(2.0)
    var clips = List[TClip]()
    # clip A: media 0, on track 0 (V1, row [20,80)), frames [10, 60)  -> px [120, 220)
    clips.append(TClip(0, 0, 50, 10, 0))
    # clip B: media 1, on track 2 (V3, row [128,208)), frames [100, 130) -> px [300, 360)
    clips.append(TClip(1, 0, 30, 100, 2))
    # clip C: media 2, on track 4 (A1, row [266,296)), frames [200, 240) -> px [500, 580)
    clips.append(TClip(2, 0, 40, 200, 4))

    # point INSIDE clip A: frame 30 -> mx = 160 ; row 0 mid -> my = 50.
    var hA = clip_at(tracks, clips, Float32(160.0), Float32(50.0), x0, base_y, gap, ppf)
    print("clip_at insideA -> idx", hA, " (expect 0)")
    if hA != 0: pass_all = False

    # point INSIDE clip B: frame 110 -> mx = 320 ; row 2 mid -> my = 160.
    var hB = clip_at(tracks, clips, Float32(320.0), Float32(160.0), x0, base_y, gap, ppf)
    print("clip_at insideB -> idx", hB, " (expect 1)")
    if hB != 1: pass_all = False

    # point INSIDE clip C: frame 210 -> mx = 520 ; row 4 mid -> my = 280.
    var hC = clip_at(tracks, clips, Float32(520.0), Float32(280.0), x0, base_y, gap, ppf)
    print("clip_at insideC -> idx", hC, " (expect 2)")
    if hC != 2: pass_all = False

    # EMPTY space (right frame, but vertically in clip A's track) -> mx far right, my in row 0.
    var e1 = clip_at(tracks, clips, Float32(900.0), Float32(50.0), x0, base_y, gap, ppf)
    # EMPTY space (correct x for clip A's frames, but on the WRONG track row, e.g. row 1 [84,124)).
    var e2 = clip_at(tracks, clips, Float32(160.0), Float32(100.0), x0, base_y, gap, ppf)
    # EMPTY space in an inter-track GAP band (between row0 bottom 80 and row1 top 84 -> y=82).
    var e3 = clip_at(tracks, clips, Float32(160.0), Float32(82.0), x0, base_y, gap, ppf)
    print("clip_at empties ->", e1, e2, e3, " (expect -1 -1 -1)")
    if e1 != -1: pass_all = False
    if e2 != -1: pass_all = False
    if e3 != -1: pass_all = False

    # boundary: left edge of clip A (frame 10 -> mx 120) is INSIDE; right edge (frame 60 -> mx 220)
    # is EXCLUSIVE (half-open span) -> -1.
    var bL = clip_at(tracks, clips, Float32(120.0), Float32(50.0), x0, base_y, gap, ppf)
    var bR = clip_at(tracks, clips, Float32(220.0), Float32(50.0), x0, base_y, gap, ppf)
    print("clip_at boundsL/R ->", bL, bR, " (expect 0 -1)")
    if bL != 0: pass_all = False
    if bR != -1: pass_all = False

    # ----- 4) track_at + frame_to_x/x_to_frame round-trip sanity -----
    if track_at(tracks, Float32(50.0), base_y, gap) != 0: pass_all = False     # row 0
    if track_at(tracks, Float32(160.0), base_y, gap) != 2: pass_all = False    # row 2 (V3)
    if track_at(tracks, Float32(82.0), base_y, gap) != -1: pass_all = False    # gap band
    if track_at(tracks, Float32(5.0), base_y, gap) != -1: pass_all = False     # above stack
    if x_to_frame(frame_to_x(123, x0, ppf), x0, ppf) != 123: pass_all = False

    # ----- 5) mute/solo audibility policy -----
    # no solo yet: every enabled, unmuted track is audible.
    if not track_audible(tracks, 0): pass_all = False
    # mute V2 -> not audible.
    tracks[1].muted = True
    if track_audible(tracks, 1): pass_all = False
    # solo A2 (idx 5): now ONLY soloed tracks audible -> V1 (idx0) silenced, A2 audible.
    tracks[5].soloed = True
    if track_audible(tracks, 0): pass_all = False
    if not track_audible(tracks, 5): pass_all = False
    print("TT audibility policy checked")

    if pass_all:
        print("TIMELINE_TRACKS_SELFTEST: pass")
    else:
        print("TIMELINE_TRACKS_SELFTEST: fail")
