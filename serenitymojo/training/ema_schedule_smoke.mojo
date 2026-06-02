# training/ema_schedule_smoke.mojo — gate for EMA wiring (item 2i).
#
# Asserts:
#   (1) HOST ema_update_host hand-check: decay=0.999, shadow=1.0, live=2.0 ->
#       1.001 (matches schedule.mojo:377 / ema.rs hand-check) to 1e-6.
#   (2) DEVICE ema_update primitive: same hand-check on a 1-elem F32 tensor.
#   (3) POWER-DECAY schedule vs the Rust ema_advanced.rs decay_at_step at
#       t in {0,1,2,11,100,1e6} (defaults inv_gamma=1, power=0.6667,
#       max_decay=0.9999) to 1e-5. t<=update_after_step returns 0.0.
#   (4) BITROT-FAIL DEMO: a wrong schedule value (off by 0.1) must exceed 1e-5.
#
# Exits NONZERO (raise) on any mismatch.
#
# Build/run (JIT):
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/training/ema_schedule_smoke.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.ema_schedule import ema_decay_at_step, ema_update_host
from serenitymojo.training.schedule import ema_update


def main() raises:
    var ctx = DeviceContext()
    var ok = True

    # ── (1) host hand-check ───────────────────────────────────────────────────
    var shadow = List[Float32](); shadow.append(Float32(1.0))
    var live = List[Float32](); live.append(Float32(2.0))
    ema_update_host(shadow, live, Float32(0.999))
    print("host ema_update shadow (expect 1.001):", shadow[0])
    var he = Float64(shadow[0]) - 1.001
    if (he if he >= 0.0 else -he) > 1.0e-6:
        print("FAIL host ema_update != 1.001"); ok = False
    else:
        print("PASS host ema_update hand-check (1.001) to 1e-6")

    # ── (2) device primitive hand-check ───────────────────────────────────────
    var sh_t = Tensor.from_host([Float32(1.0)], [1], STDtype.F32, ctx)
    var lv_t = Tensor.from_host([Float32(2.0)], [1], STDtype.F32, ctx)
    ema_update(sh_t, lv_t, Float32(0.999), ctx)
    var sh_h = sh_t.to_host(ctx)
    print("device ema_update shadow (expect 1.001):", sh_h[0])
    var de = Float64(sh_h[0]) - 1.001
    if (de if de >= 0.0 else -de) > 1.0e-6:
        print("FAIL device ema_update != 1.001"); ok = False
    else:
        print("PASS device ema_update hand-check (1.001) to 1e-6")

    # ── (3) power-decay schedule vs Rust ──────────────────────────────────────
    # Rust ema_advanced.rs decay_at_step defaults: inv_gamma=1, power=0.6667,
    # update_after_step=0, min=0, max=0.9999.
    var steps = List[Int](); steps.append(0); steps.append(1); steps.append(2)
    steps.append(11); steps.append(100); steps.append(1000000)
    var refs = List[Float64]()
    refs.append(0.0)
    refs.append(0.3700540300631411)
    refs.append(0.5192677481651926)
    refs.append(0.8092300950757283)
    refs.append(0.9538980877138314)
    refs.append(0.9999)
    for i in range(len(steps)):
        var got = ema_decay_at_step(
            steps[i], 0, Float32(1.0), Float32(0.6667), Float32(0.0), Float32(0.9999)
        )
        var e = Float64(got) - refs[i]
        var ae = e if e >= 0.0 else -e
        print("decay t=", steps[i], " got=", got, " ref=", Float32(refs[i]), " |err|=", Float32(ae))
        if ae > 1.0e-5:
            print("FAIL power-decay mismatch at t=", steps[i]); ok = False
    print("PASS-or-FAIL power-decay sweep complete")

    # ── (3b) update_after_step gate returns 0 before warmup ───────────────────
    var pre = ema_decay_at_step(50, 100, Float32(1.0), Float32(0.6667), Float32(0.0), Float32(0.9999))
    print("decay step=50 update_after=100 (expect 0):", pre)
    if pre != Float32(0.0):
        print("FAIL pre-warmup decay != 0"); ok = False
    else:
        print("PASS pre-warmup decay == 0 (skip)")

    # ── (4) BITROT-FAIL DEMO ──────────────────────────────────────────────────
    var got1 = ema_decay_at_step(1, 0, Float32(1.0), Float32(0.6667), Float32(0.0), Float32(0.9999))
    var wrong_ref = 0.3700540300631411 + 0.1
    var we = Float64(got1) - wrong_ref
    var wae = we if we >= 0.0 else -we
    print("bitrot demo: decay t=1 vs WRONG ref(+0.1) |err|=", Float32(wae))
    if wae <= 1.0e-5:
        print("FAIL bitrot demo: matched wrong ref"); ok = False
    else:
        print("PASS bitrot demo: wrong ref exceeds 1e-5 (gate is sensitive)")

    if not ok:
        raise Error("ema_schedule_smoke FAILED")
    print("ema_schedule_smoke gate PASS")
