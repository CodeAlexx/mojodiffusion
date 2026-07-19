# ltx2_grad_accum_counter.mojo — P4 grad-accum counter-math + window unit gate.
#
# CPU-only (pure host math — no GPU, embargo-safe). Verifies the driver's
# micro/update counter split against the REAL GradAccumWindow:
#   micro (loop var)  -> keys sample/sigma/noise/dropout (unchanged derivations)
#   _opt_idx = (micro-1)//accum + 1  -> AdamW t / LR / cadence / save / termination
#   loop runs micro = start..max_updates*accum ; boundary every `accum` micros.
#   resume: start_micro = saved_update*accum + 1  (=> first _opt_idx = saved_update+1)
# Plus the accum==1 byte-identity: the window's sum-of-one/÷1 leaves grads EXACT
# (trainer_core.mojo:251-252) and _opt_idx == micro (=> C13 per-step path).
#
#   pixi run mojo build -O2 -I . serenitymojo/models/ltx2/parity/ltx2_grad_accum_counter.mojo \
#     -o /tmp/ltx2_grad_accum_counter && /tmp/ltx2_grad_accum_counter

from serenitymojo.training.trainer_core import GradAccumWindow


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error("FAIL: " + msg)


def _opt_idx(micro: Int, accum: Int) -> Int:
    return (micro - 1) // accum + 1


# synthetic grad group: n_adapters x width, value = micro-keyed so we can check
# the mean is exact.
def _mk_group(n_ad: Int, width: Int, micro: Int) -> List[List[Float32]]:
    var g = List[List[Float32]]()
    for a in range(n_ad):
        var row = List[Float32]()
        for j in range(width):
            row.append(Float32(micro) * Float32(0.5) + Float32(a) - Float32(j) * Float32(0.25))
        g.append(row^)
    return g^


def _run_counter(accum: Int, max_updates: Int, start_update: Int) raises:
    # start_micro reconstructs from the saved UPDATE index (resume contract).
    var start_micro = start_update * accum + 1
    _require(_opt_idx(start_micro, accum) == start_update + 1,
             "resume: first _opt_idx != saved_update+1 (accum=" + String(accum) + ")")
    var total_micros = max_updates * accum
    var win = GradAccumWindow(accum)
    var boundaries = 0
    var last_update_seen = start_update
    var micro = start_micro
    while micro <= total_micros:
        var ga = _mk_group(3, 4, micro)
        var gb = _mk_group(3, 4, micro)
        var is_boundary = win.accumulate(ga, gb, False)
        # boundary EXACTLY when the window filled `accum` micros == micro % accum == 0
        _require(is_boundary == (micro % accum == 0),
                 "boundary mismatch at micro " + String(micro) + " accum " + String(accum))
        if is_boundary:
            var oi = _opt_idx(micro, accum)
            _require(oi == micro // accum, "boundary _opt_idx != micro//accum at " + String(micro))
            _require(oi == last_update_seen + 1,
                     "update index skipped/repeated at micro " + String(micro))
            last_update_seen = oi
            boundaries += 1
            var _gn = win.finalize_mean(ga, gb)  # resets the window
        micro += 1
    _require(boundaries == max_updates - start_update,
             "boundary count " + String(boundaries) + " != updates "
             + String(max_updates - start_update) + " (accum " + String(accum) + ")")
    _require(last_update_seen == max_updates, "did not terminate on max_updates")
    print("  PASS accum=", accum, " updates=", max_updates, " resume@", start_update,
          " -> ", boundaries, "boundaries, last update", last_update_seen)


# accum==1 byte-identity: window leaves the single micro's grads EXACT (sum/÷1),
# and _opt_idx == micro.
def _accum1_identity() raises:
    var win = GradAccumWindow(1)
    for micro in range(1, 6):
        _require(_opt_idx(micro, 1) == micro, "accum=1 _opt_idx != micro")
        var ga = _mk_group(2, 5, micro)
        var gb = _mk_group(2, 5, micro)
        var ref_a = _mk_group(2, 5, micro)   # a copy of the exact input
        var ref_b = _mk_group(2, 5, micro)
        var is_boundary = win.accumulate(ga, gb, False)
        _require(is_boundary, "accum=1 every micro must be a boundary")
        var _gn = win.finalize_mean(ga, gb)
        # sum-of-one / ÷1 -> grads BIT-EXACT to the input
        for a in range(len(ga)):
            for j in range(len(ga[a])):
                _require(ga[a][j] == ref_a[a][j], "accum=1 d_a mutated (not sum/÷1 identity)")
                _require(gb[a][j] == ref_b[a][j], "accum=1 d_b mutated")
    print("  PASS accum=1 byte-identity (sum-of-one/÷1 leaves grads EXACT; _opt_idx==micro)")


# accum>1 mean correctness: window mean == arithmetic mean of the micros' grads.
def _mean_correctness() raises:
    var accum = 4
    var win = GradAccumWindow(accum)
    var ga = List[List[Float32]]()
    var gb = List[List[Float32]]()
    for micro in range(1, accum + 1):
        ga = _mk_group(2, 3, micro)
        gb = _mk_group(2, 3, micro)
        var _b = win.accumulate(ga, gb, False)
    var _gn = win.finalize_mean(ga, gb)   # ga/gb now hold the MEAN
    # expected mean of _mk_group over micro=1..4 at (a,j): mean(micro)*0.5 + a - j*0.25
    var mean_micro = Float32(2.5)   # (1+2+3+4)/4
    for a in range(2):
        for j in range(3):
            var expect = mean_micro * Float32(0.5) + Float32(a) - Float32(j) * Float32(0.25)
            var d = ga[a][j] - expect
            if d < 0: d = -d
            _require(d < Float32(1e-5), "mean mismatch at (" + String(a) + "," + String(j) + ")")
    print("  PASS accum=4 window MEAN == arithmetic mean of the 4 micros")


def main() raises:
    print("=== LTX2 P4 grad-accum counter + window unit gate (CPU) ===")
    _accum1_identity()
    _mean_correctness()
    # counter/boundary/termination/resume over synthetic (accum, updates, resume@)
    _run_counter(1, 4, 0)     # accum=1 fresh
    _run_counter(4, 4, 0)     # accum=4 fresh (16 micros -> 4 updates)
    _run_counter(4, 6, 2)     # accum=4 resume at update 2 (continue to 6)
    _run_counter(3, 5, 0)
    _run_counter(8, 3, 1)     # accum=8 resume at update 1
    print("ALL PASS")
