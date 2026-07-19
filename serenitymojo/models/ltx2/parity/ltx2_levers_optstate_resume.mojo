# ltx2_levers_optstate_resume.mojo — levers-optimizer resume sidecar gates (P3.3).
#
# CPU-only (the *_step fns + the sidecar are host math — no GPU). Per optimizer
# family (adafactor / schedule_free / adamw_8bit / automagic3):
#   (a) ROUND-TRIP: step 2x on synthetic F32 params, save the sidecar, load into
#       a fresh state, assert every restored F32 buffer + counter is BIT-EXACT.
#   (b) CONTINUATION: an uninterrupted 4-step run vs a run that saves at k=2,
#       reloads into a fresh state, and continues to k=4 — the final params MUST
#       be BIT-EXACT (proves the whole state — moments, counters, ctl lr + RNG,
#       automagic3 sign-history — round-trips faithfully).
#
#   pixi run mojo build -O2 -I . -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/models/ltx2/parity/ltx2_levers_optstate_resume.mojo \
#     -o /tmp/ltx2_levers_optstate_resume && /tmp/ltx2_levers_optstate_resume

from std.math import sin, cos

from serenitymojo.training.levers import (
    F32AdapterView, LeversOptimizerState, levers_optimizer_step_host_f32,
)
from serenitymojo.training.levers_optimizer_sidecar import (
    levers_optimizer_sidecar_save, levers_optimizer_sidecar_load,
)
from serenitymojo.training.train_config import (
    TrainConfig,
    TRAIN_OPTIMIZER_ADAFACTOR, TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW,
    TRAIN_OPTIMIZER_ADAMW_8BIT, TRAIN_OPTIMIZER_AUTOMAGIC3,
)

comptime R = 4
comptime IN = 16
comptime OUT = 16
comptime NAD = 3


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def _sinfill(n: Int, phase: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        var fi = Float32(i)
        out.append(Float32(0.02) * (sin(Float32(0.7) * fi + phase)
                   + Float32(0.3) * cos(Float32(1.3) * fi)))
    return out^


def _mk_views() -> List[F32AdapterView]:
    var v = List[F32AdapterView]()
    for j in range(NAD):
        var fj = Float32(j) * Float32(0.37)
        v.append(F32AdapterView(
            _sinfill(R * IN, fj + Float32(0.1)),
            _sinfill(OUT * R, fj + Float32(0.5)), R, IN, OUT))
    return v^


# deterministic per-step grads (keyed on k so ref and resumed runs match).
def _mk_grads(k: Int) -> Tuple[List[List[Float32]], List[List[Float32]]]:
    var da = List[List[Float32]]()
    var db = List[List[Float32]]()
    for j in range(NAD):
        var fk = Float32(k) * Float32(0.9) + Float32(j) * Float32(0.31)
        da.append(_sinfill(R * IN, fk + Float32(2.0)))
        db.append(_sinfill(OUT * R, fk + Float32(2.7)))
    return (da^, db^)


def _mk_cfg(tag: Int) -> TrainConfig:
    var cfg = TrainConfig.default()
    cfg.optimizer = tag
    cfg.lr = Float32(1.0e-3)
    cfg.weight_decay = Float32(0.0)
    cfg.seed = UInt64(1234567)
    return cfg^


def _step(cfg: TrainConfig, mut views: List[F32AdapterView], k: Int, mut st: LeversOptimizerState) raises:
    var g = _mk_grads(k)
    levers_optimizer_step_host_f32(cfg, views, g[0], g[1], k, Float32(1.0e-3), 0, NAD, st)


def _cmp(a: List[Float32], b: List[Float32], tag: String) raises:
    _require(len(a) == len(b), tag + ": len mismatch")
    for i in range(len(a)):
        if a[i] != b[i]:
            raise Error(tag + ": BIT MISMATCH elem " + String(i))


def _params_equal(va: List[F32AdapterView], vb: List[F32AdapterView], tag: String) raises:
    _require(len(va) == len(vb), tag + ": view count mismatch")
    for j in range(len(va)):
        _cmp(va[j].a, vb[j].a, tag + " A[" + String(j) + "]")
        _cmp(va[j].b, vb[j].b, tag + " B[" + String(j) + "]")


def _test_family(tag: Int, name: String) raises:
    var cfg = _mk_cfg(tag)
    var path = String("/tmp/ltx2_optstate_") + name + ".safetensors"
    var n_states = 2 * NAD

    # ── (a) ROUND-TRIP: 2 steps, save, load, compare the F32 buffers ─────────
    var vr = _mk_views()
    var st_rt = LeversOptimizerState()
    _step(cfg, vr, 1, st_rt)
    _step(cfg, vr, 2, st_rt)
    var n_saved = levers_optimizer_sidecar_save(st_rt, 2, cfg.seed, path)
    _require(n_saved == n_states, name + ": save n_states " + String(n_saved))
    var res = levers_optimizer_sidecar_load(path, tag, n_states)
    _require(res.k == 2, name + ": loaded k != 2")
    var st_ld = res^.take_state()
    # per-family F32-buffer bit-check
    if tag == TRAIN_OPTIMIZER_ADAFACTOR:
        for j in range(n_states):
            _cmp(st_rt.ada[j].row_var, st_ld.ada[j].row_var, name + " rowvar")
            _cmp(st_rt.ada[j].col_var, st_ld.ada[j].col_var, name + " colvar")
            _require(st_rt.ada[j].step == st_ld.ada[j].step, name + " step")
    elif tag == TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW:
        for j in range(n_states):
            _cmp(st_rt.sf[j].exp_avg, st_ld.sf[j].exp_avg, name + " exp_avg")
            _cmp(st_rt.sf[j].exp_avg_sq, st_ld.sf[j].exp_avg_sq, name + " exp_avg_sq")
        _require(st_rt.sf_ctl.k == st_ld.sf_ctl.k, name + " ctl.k")
    elif tag == TRAIN_OPTIMIZER_ADAMW_8BIT:
        for j in range(n_states):
            _require(len(st_rt.a8[j].m_codes) == len(st_ld.a8[j].m_codes), name + " m_codes len")
            for i in range(len(st_rt.a8[j].m_codes)):
                _require(st_rt.a8[j].m_codes[i] == st_ld.a8[j].m_codes[i], name + " m_codes")
            _cmp(st_rt.a8[j].m_absmax, st_ld.a8[j].m_absmax, name + " m_absmax")
            _require(st_rt.a8[j].step == st_ld.a8[j].step, name + " step")
    elif tag == TRAIN_OPTIMIZER_AUTOMAGIC3:
        for j in range(n_states):
            _require(st_rt.auto3[j].hist_fill == st_ld.auto3[j].hist_fill, name + " hist_fill")
            _require(st_rt.auto3[j].hist_idx == st_ld.auto3[j].hist_idx, name + " hist_idx")
        _require(st_rt.auto3_ctl.rng.state == st_ld.auto3_ctl.rng.state, name + " rng.state")
    print("  PASS", name, "round-trip (state bit-exact)")

    # ── (b) CONTINUATION: uninterrupted 4 vs 2-save-load-continue-4 ──────────
    var vref = _mk_views()
    var st_ref = LeversOptimizerState()
    for k in range(1, 5):
        _step(cfg, vref, k, st_ref)

    var vres = _mk_views()
    var st_a = LeversOptimizerState()
    _step(cfg, vres, 1, st_a)
    _step(cfg, vres, 2, st_a)
    var cpath = String("/tmp/ltx2_optstate_cont_") + name + ".safetensors"
    _ = levers_optimizer_sidecar_save(st_a, 2, cfg.seed, cpath)
    var res2 = levers_optimizer_sidecar_load(cpath, tag, n_states)
    var st_b = res2^.take_state()
    _step(cfg, vres, 3, st_b)
    _step(cfg, vres, 4, st_b)

    _params_equal(vref, vres, name + " continuation")
    print("  PASS", name, "continuation (resumed params == uninterrupted, bit-exact)")


def main() raises:
    print("=== LTX2 levers-optimizer resume-sidecar gate (CPU) ===")
    _test_family(TRAIN_OPTIMIZER_ADAFACTOR, String("adafactor"))
    _test_family(TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW, String("schedule_free"))
    _test_family(TRAIN_OPTIMIZER_ADAMW_8BIT, String("adamw_8bit"))
    _test_family(TRAIN_OPTIMIZER_AUTOMAGIC3, String("automagic3"))
    print("ALL PASS")
