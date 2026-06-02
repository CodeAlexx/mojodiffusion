# training/caption_dropout_smoke.mojo — gate for caption dropout (item 2d).
#
# Asserts:
#   (1) should_drop_caption(seed, p) == (sample_timestep_uniform(seed) < p)
#       for a sweep of seeds + p values (decision matches the Rust draw exactly).
#   (2) p=0.0 NEVER drops (default-off; no draw consumed).
#   (3) empirical drop RATE over N=4000 distinct seeds matches p within 0.03
#       for p in {0.1, 0.25, 0.5} (distribution-level parity since the Mojo
#       ChaCha12 stream is seed-keyed, not a single advancing StdRng).
#   (4) BITROT-FAIL DEMO: a deliberately-wrong predicate (draw > p) is shown to
#       disagree with the spec on a known case, proving the gate can fail.
#
# The smoke exits NONZERO (raise) on any mismatch (Tenet 4 / parity-bitrot guard).
#
# Build/run (JIT):
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/training/caption_dropout_smoke.mojo

from std.math import sqrt
from serenitymojo.training.schedule import sample_timestep_uniform
from serenitymojo.training.caption_dropout import should_drop_caption, caption_dropout_draw


def main() raises:
    var ok = True

    # ── (1) decision matches the draw exactly ─────────────────────────────────
    var ps = List[Float32]()
    ps.append(Float32(0.1)); ps.append(Float32(0.25)); ps.append(Float32(0.5))
    ps.append(Float32(0.9))
    var mismatches = 0
    for pi in range(len(ps)):
        var p = ps[pi]
        for s in range(200):
            var seed = UInt64(7000 + s)
            var draw = sample_timestep_uniform(seed)
            var expect = draw < p
            var got = should_drop_caption(seed, p)
            if got != expect:
                mismatches += 1
    print("decision-vs-draw mismatches (expect 0):", mismatches)
    if mismatches != 0:
        print("FAIL should_drop_caption disagrees with (draw < p)"); ok = False
    else:
        print("PASS decision == (sample_timestep_uniform(seed) < p) for all cases")

    # ── (2) p=0 never drops ────────────────────────────────────────────────────
    var dropped0 = 0
    for s in range(1000):
        if should_drop_caption(UInt64(s), Float32(0.0)):
            dropped0 += 1
    print("p=0 drops over 1000 (expect 0):", dropped0)
    if dropped0 != 0:
        print("FAIL p=0 dropped a caption"); ok = False
    else:
        print("PASS p=0 never drops (default-off)")

    # ── (3) empirical rate matches p within 0.03 ──────────────────────────────
    for pi in range(3):  # 0.1, 0.25, 0.5
        var p = ps[pi]
        var n = 4000
        var hits = 0
        for s in range(n):
            if should_drop_caption(UInt64(100000 + s), p):
                hits += 1
        var rate = Float32(hits) / Float32(n)
        var err = rate - p if rate >= p else p - rate
        print("p=", p, " empirical drop rate=", rate, " |err|=", err)
        if err > Float32(0.03):
            print("FAIL empirical rate off by >0.03 for p=", p); ok = False
        else:
            print("PASS empirical rate within 0.03 of p")

    # ── (4) BITROT-FAIL DEMO: wrong predicate must disagree on a known case ────
    # Find a seed whose draw < 0.5 (should drop). The wrong predicate (draw>0.5)
    # would say "keep" — proving the gate is sensitive to the comparison direction.
    var demo_seed = UInt64(0)
    var found = False
    for s in range(500):
        var d = sample_timestep_uniform(UInt64(s))
        if d < Float32(0.5):
            demo_seed = UInt64(s); found = True; break
    if not found:
        print("FAIL bitrot demo: no low draw found"); ok = False
    else:
        var correct = should_drop_caption(demo_seed, Float32(0.5))  # True
        var wrong = caption_dropout_draw(demo_seed) > Float32(0.5)  # False
        print("bitrot demo seed=", demo_seed, " correct(drop)=", correct, " wrong-pred=", wrong)
        if correct == wrong:
            print("FAIL bitrot demo: wrong predicate did NOT disagree"); ok = False
        else:
            print("PASS bitrot demo: wrong predicate disagrees (gate is sensitive)")

    if not ok:
        raise Error("caption_dropout_smoke FAILED")
    print("caption_dropout_smoke gate PASS")
