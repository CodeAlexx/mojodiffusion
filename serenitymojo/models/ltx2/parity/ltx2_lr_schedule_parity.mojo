# ltx2_lr_schedule_parity.mojo — LTX2 LR-lever parity vs transformers.optimization.
#
# musubi's LR path IS transformers.get_scheduler, so the LTX2 trainer routes its
# scheduled LR through `transformers_lr_for_step` (training/lr_schedule.mojo). This
# gate proves that function reproduces transformers.get_scheduler for the four
# musubi-exposed kinds (constant / constant_with_warmup / linear / cosine) over
# warmups {0, 5, 10} at the probe steps {0, 1, W-1, W, W+1, mid, total-1}.
#
# It ALSO measures the FLAME `lr_for_step` (the SerenityTrainer port) against the same
# oracle and prints its worst error — the (step+1)/W warmup ramp completes one
# step early vs transformers' step/max(1,W), the divergence that made a dedicated
# transformers path necessary (reported, not gated).
#
# Run the oracle first (writes ltx2_lr_schedule_ref.bin), then the gate:
#   /home/alex/serenityflow-v2/.venv/bin/python scripts/ltx2_lr_schedule_oracle.py
#   pixi run mojo run -I . serenitymojo/models/ltx2/parity/ltx2_lr_schedule_parity.mojo
#
# Bar: absolute error <= 1e-5 (base_lr=1.0, factors in [0,1]). f32-vs-f64 cosine
# noise sits ~1e-7; a formula divergence (the flame warmup ramp) is ~0.1-0.2, so
# 1e-5 catches divergence decisively while tolerating f32 transcendental rounding.

from std.memory import alloc
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.training.lr_schedule import (
    transformers_lr_for_step, lr_for_step, LR_CONSTANT, LR_LINEAR, LR_COSINE,
)


comptime DIR = "/home/alex/mojodiffusion/serenitymojo/models/ltx2/parity/"
comptime TOTAL = 100
comptime NROWS = 9
comptime BAR = Float32(1.0e-5)


def _read_f32_bin(name: String) raises -> List[Float32]:
    var path = String(DIR) + name
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open ") + path + " (run the oracle first)")
    var nbytes = file_size(fd)
    if nbytes <= 0:
        _ = sys_close(fd)
        raise Error(String("empty bin: ") + path)
    var buf = alloc[UInt8](nbytes)
    var done = 0
    while done < nbytes:
        var got = sys_pread(fd, buf + done, nbytes - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var nfloats = done // 4
    var out = List[Float32]()
    var fptr = buf.bitcast[Float32]()
    for i in range(nfloats):
        out.append(fptr[i])
    buf.free()
    return out^


# probe steps {0, 1, W-1, W, W+1, mid, total-1} clamped to [0, TOTAL-1], deduped.
def _probe_steps(w: Int) -> List[Int]:
    var raw = [0, 1, w - 1, w, w + 1, TOTAL // 2, TOTAL - 1]
    var out = List[Int]()
    for ref t in raw:
        if t < 0 or t > TOTAL - 1:
            continue
        var dup = False
        for ref u in out:
            if u == t:
                dup = True
        if not dup:
            out.append(t)
    return out^


def main() raises:
    var refv = _read_f32_bin("ltx2_lr_schedule_ref.bin")
    if len(refv) != NROWS * TOTAL:
        raise Error(String("ltx2_lr_schedule_ref.bin size ") + String(len(refv))
                    + " != " + String(NROWS * TOTAL) + " (re-run the oracle)")

    # row order MIRRORS the oracle: constant/0, cwarmup/5, cwarmup/10,
    # linear/{0,5,10}, cosine/{0,5,10}. constant + constant_with_warmup both map
    # to Mojo LR_CONSTANT (warmup>0 ramps, warmup0 is flat).
    var kinds = [
        LR_CONSTANT, LR_CONSTANT, LR_CONSTANT,
        LR_LINEAR, LR_LINEAR, LR_LINEAR,
        LR_COSINE, LR_COSINE, LR_COSINE,
    ]
    var warmups = [0, 5, 10, 0, 5, 10, 0, 5, 10]

    var worst_tf = Float32(0.0)
    var worst_flame = Float32(0.0)
    var checks = 0
    for r in range(NROWS):
        var w = warmups[r]
        var kind = kinds[r]
        var probes = _probe_steps(w)
        for ref t in probes:
            var exp = refv[r * TOTAL + t]
            var got = transformers_lr_for_step(Float32(1.0), t, w, TOTAL, kind)
            var e = abs(got - exp)
            if e > worst_tf:
                worst_tf = e
            checks += 1
            if e > BAR:
                print("  FAIL row", r, "step", t, "kind", kind, "warmup", w,
                      "  transformers_lr=", got, " oracle=", exp, " abs=", e)
                raise Error(String("[ltx2-lr-parity] FAIL at row ") + String(r)
                            + " step " + String(t) + " (abs > 1e-5)")
            # flame SerenityTrainer-port path vs the SAME oracle (evidence, not gated).
            var flame = lr_for_step(
                Float32(1.0), t, w, TOTAL, kind,
                Float32(0.0), Float32(0.0), Float32(2.0))
            var ef = abs(flame - exp)
            if ef > worst_flame:
                worst_flame = ef

    print("[ltx2-lr-parity]", checks,
          "checks (constant/cwarmup/linear/cosine x warmup {0,5,10})")
    print("[ltx2-lr-parity] transformers_lr_for_step worst abs err vs oracle:",
          worst_tf, " (bar 1e-5) -> PASS")
    print("[ltx2-lr-parity] flame lr_for_step worst abs err vs oracle:",
          worst_flame,
          " (the (step+1)/W warmup divergence -> ltx2 uses transformers_lr_for_step)")
    print("ALL PASS")
