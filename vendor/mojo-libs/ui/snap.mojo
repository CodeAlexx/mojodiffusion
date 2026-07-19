# snap.mojo — pure SNAP functions for the timeline (fully headless, no GL).
#
# These are timeline-snapping primitives for the NLE: when the user drags a clip / the
# playhead near another clip's edge (or any marker), the position SNAPS to that edge if it
# falls within a tolerance. No drawing, no FFI, no external_call — pure integer / Float32
# arithmetic, so this module is trivially unit-testable headlessly (the surrounding widgets
# in media_widgets.mojo / widget_variants.mojo carry a 'draw' gate for that; here there is
# simply nothing to gate — every function is pure logic).
#
# MOJO 1.0.0b1 idioms (matched to core_widgets.mojo / media_widgets.mojo / widget_variants.mojo):
#   - 'def' (with 'raises' only where a raising API is called; these are non-raising).
#   - 'var'/'mut'; 'comptime' for constants; multiline 'if'; explicit Float32()/Int() casts.
#   - List is move-only: build with append, return with ^, read elements by index.
#
# Public API:
#   snap_frame(frame, edges, max_dist) -> Int          frame-space snap (nearest edge wins)
#   snap_px(px, edges_px, max_dist_px) -> Float32      pixel-space variant
#   collect_edges(clip_starts, clip_lens) -> List[Int] all clip start + end frames, for snapping


# ============================================================================
# tiny helpers (no math import — only builtin arithmetic, like the exemplars)
# ============================================================================
def iabs(v: Int) -> Int:
    if v < 0:
        return -v
    return v


def fabsf(v: Float32) -> Float32:
    if v < Float32(0.0):
        return -v
    return v


# ============================================================================
# snap_frame — if some edge is within `max_dist` frames of `frame`, snap to the NEAREST such
# edge; otherwise return `frame` unchanged. Ties keep the first encountered (stable).
# ============================================================================
def snap_frame(frame: Int, edges: List[Int], max_dist: Int) -> Int:
    var best = frame
    var best_d = max_dist + 1            # sentinel: nothing within tolerance yet
    var n = len(edges)
    for i in range(n):
        var e = edges[i]
        var d = iabs(e - frame)
        if d <= max_dist and d < best_d:
            best_d = d
            best = e
    return best


# ============================================================================
# snap_px — pixel-space variant of snap_frame. If some edge_px is within `max_dist_px` of
# `px`, snap to the NEAREST one; else return `px`. Same nearest-wins / stable-tie rule.
# ============================================================================
def snap_px(px: Float32, edges_px: List[Float32], max_dist_px: Float32) -> Float32:
    var best = px
    var best_d = max_dist_px + Float32(1.0)    # sentinel just outside tolerance
    var have = False
    var n = len(edges_px)
    for i in range(n):
        var e = edges_px[i]
        var d = fabsf(e - px)
        if d <= max_dist_px:
            if (not have) or d < best_d:
                best_d = d
                best = e
                have = True
    return best


# ============================================================================
# collect_edges — flatten parallel (start, len) arrays into the list of all snap targets:
# each clip contributes its START frame and its END frame (start + len). Mismatched lengths
# are handled by iterating the SHORTER of the two (defensive). Order: start_i, end_i, ...
# ============================================================================
def collect_edges(clip_starts: List[Int], clip_lens: List[Int]) -> List[Int]:
    var out = List[Int]()
    var n = len(clip_starts)
    var m = len(clip_lens)
    if m < n:
        n = m
    for i in range(n):
        var s = clip_starts[i]
        var e = s + clip_lens[i]
        out.append(s)
        out.append(e)
    return out^


# ============================================================================
# SELF-TEST  (compiled + run headlessly by the MAIN LOOP — do NOT run it yourself)
#
# Pure logic: no UiContext / GL needed. Every expected value is derived from first principles
# in the comment beside it.
# ============================================================================
def main() raises:
    var pass_all = True

    # ----- 1) collect_edges: starts + ends interleaved -----
    # clips: (start=0, len=10) -> edges 0,10 ; (start=20, len=5) -> edges 20,25 ;
    #        (start=100, len=30) -> edges 100,130.  Expect [0,10,20,25,100,130].
    var starts = List[Int]()
    starts.append(0); starts.append(20); starts.append(100)
    var lens = List[Int]()
    lens.append(10); lens.append(5); lens.append(30)
    var edges = collect_edges(starts, lens)
    var edges_ok = (len(edges) == 6)
    if edges_ok:
        edges_ok = (
            (edges[0] == 0) and (edges[1] == 10) and (edges[2] == 20)
            and (edges[3] == 25) and (edges[4] == 100) and (edges[5] == 130)
        )
    print("SNAP collect_edges n=", len(edges), " ok=", edges_ok)
    if not edges_ok:
        pass_all = False

    # ----- 2) snap_frame: near edge snaps, far frame is unchanged -----
    # edges = the 6 collected above; max_dist = 3 frames.
    #   frame 22 -> nearest edge 20 (|22-20|=2 <= 3)  => 20.
    #   frame 9  -> nearest edge 10 (|9-10|=1 <= 3)   => 10.
    #   frame 50 -> nearest edge is 25 (|50-25|=25) or 100 (50) — both > 3 => 50 (unchanged).
    var s_near = snap_frame(22, edges, 3)        # EXPECT 20
    var s_near2 = snap_frame(9, edges, 3)        # EXPECT 10
    var s_far = snap_frame(50, edges, 3)         # EXPECT 50 (no edge within tolerance)
    print("SNAP frame 22->", s_near, " 9->", s_near2, " 50->", s_far)
    if s_near != 20:
        pass_all = False
    if s_near2 != 10:
        pass_all = False
    if s_far != 50:
        pass_all = False

    # nearest-wins tie-break: frame 12 with edges {10,15} and tol 5 -> 10 (|2| < |3|).
    var two = List[Int]()
    two.append(10); two.append(15)
    var s_nearest = snap_frame(12, two, 5)       # EXPECT 10 (closer of the two)
    print("SNAP nearest-wins 12->", s_nearest)
    if s_nearest != 10:
        pass_all = False

    # ----- 3) snap_px: pixel-space snap works the same way -----
    # edges_px = [100.0, 250.5, 400.0]; max_dist_px = 8.0.
    #   px 252.0 -> nearest 250.5 (|1.5| <= 8) => 250.5.
    #   px 300.0 -> nearest is 250.5 (49.5) / 400.0 (100) — both > 8 => 300.0 (unchanged).
    var epx = List[Float32]()
    epx.append(Float32(100.0)); epx.append(Float32(250.5)); epx.append(Float32(400.0))
    var p_near = snap_px(Float32(252.0), epx, Float32(8.0))     # EXPECT 250.5
    var p_far = snap_px(Float32(300.0), epx, Float32(8.0))      # EXPECT 300.0
    print("SNAP px 252->", p_near, " 300->", p_far)
    if p_near != Float32(250.5):
        pass_all = False
    if p_far != Float32(300.0):
        pass_all = False

    # ----- summary line the main loop gates on -----
    if pass_all:
        print("SNAP_SELFTEST: pass (edges[0,10,20,25,100,130]; frame 22->20/9->10/50->50; px 252->250.5/300->300)")
    else:
        print("SNAP_SELFTEST: fail")
