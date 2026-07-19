# levers_optimizer_sidecar.mojo — resume sidecar for LeversOptimizerState (P3.3).
#
# The levers optimizers (adafactor/schedule_free/adamw_8bit/automagic3) carry
# rich per-adapter state that the F32Lora ma/va/mb/vb fields (dead under a lever
# optimizer) cannot hold — so a lever-optimizer resume needs its OWN sidecar.
# This is that sidecar: HOST-DIRECT save/load (save_safetensors_host — the state
# NEVER stages through VRAM, the P0 lesson) of the whole LeversOptimizerState,
# per optimizer family, to `<base>.optstate.safetensors`.
#
# Template: training/full_ft_sidecar.mojo (U32[.] meta + fail-loud load), but
# host-direct (HostTensorDesc) and generalized per family. Layout per family:
#   ADAFACTOR:  af.<j>.row_var/col_var (F32) + af.<j>.step (U32)
#   SCHEDULE_FREE: sf.<j>.exp_avg/exp_avg_sq/kahan (F32) + sf.ctl (k U32,
#                  lr_max/last_lr F64, train_mode U8)
#   ADAMW_8BIT: a8.<j>.m_codes/v_codes (U8) + m_absmax/v_absmax (F32) + step (U32)
#               (the 256-entry qmap LUTs are deterministic -> REBUILT on load)
#   AUTOMAGIC3: auto3.<j>.rowvar/colvar/v (F32, per `factored`) + signhist (U8,
#               bit-packed hist_fill planes) + scalars (U32[8]) + auto3.ctl
#               (lr F64, initialized U8, rng.state U64)
# meta __meta__ U32[6] = [optimizer_tag, k, n_states, seed_lo, seed_hi, version=1].
# Fail loud on tag/count/version mismatch at load. j = state index (2 per adapter,
# A then B — LeversOptimizerState ordering).
#
# Mojo 1.0.0b1.

from std.collections import List

from serenitymojo.training.levers import LeversOptimizerState
from serenitymojo.training.adafactor import AdafactorState
from serenitymojo.training.adamw_schedulefree import (
    AdamWScheduleFreeState, AdamWScheduleFreeCtl,
)
from serenitymojo.training.adamw8bit import Adam8bitState, adam8bit_create_dynamic_map
from serenitymojo.training.automagic3 import (
    Automagic3State, Automagic3Ctl, Automagic3Rng,
)
from serenitymojo.training.train_config import (
    TRAIN_OPTIMIZER_ADAFACTOR, TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW,
    TRAIN_OPTIMIZER_ADAMW_8BIT, TRAIN_OPTIMIZER_AUTOMAGIC3,
)
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors_host, HostTensorDesc
from serenitymojo.io.dtype import STDtype

comptime SIDECAR_VERSION = 1


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


# ── little-endian byte packers (native LE on x86-64, via ptr bitcast) ─────────
def _f32_le(v: List[Float32]) -> List[UInt8]:
    var out = List[UInt8]()
    var p = v.unsafe_ptr().bitcast[UInt8]()
    for i in range(len(v) * 4):
        out.append(p[i])
    return out^


def _f64_le(v: List[Float64]) -> List[UInt8]:
    var out = List[UInt8]()
    var p = v.unsafe_ptr().bitcast[UInt8]()
    for i in range(len(v) * 8):
        out.append(p[i])
    return out^


def _u32_le(v: List[UInt32]) -> List[UInt8]:
    var out = List[UInt8]()
    var p = v.unsafe_ptr().bitcast[UInt8]()
    for i in range(len(v) * 4):
        out.append(p[i])
    return out^


def _u64_le(v: List[UInt64]) -> List[UInt8]:
    var out = List[UInt8]()
    var p = v.unsafe_ptr().bitcast[UInt8]()
    for i in range(len(v) * 8):
        out.append(p[i])
    return out^


# ── HostTensorDesc adders ─────────────────────────────────────────────────────
def _shp(n: Int) -> List[Int]:
    var s = List[Int]()
    s.append(n)
    return s^


def _add_f32(mut names: List[String], mut descs: List[HostTensorDesc], name: String, v: List[Float32]):
    names.append(name)
    descs.append(HostTensorDesc(STDtype.F32, _shp(len(v)), _f32_le(v)))


def _add_f64(mut names: List[String], mut descs: List[HostTensorDesc], name: String, v: List[Float64]):
    names.append(name)
    descs.append(HostTensorDesc(STDtype.F64, _shp(len(v)), _f64_le(v)))


def _add_u32(mut names: List[String], mut descs: List[HostTensorDesc], name: String, v: List[UInt32]):
    names.append(name)
    descs.append(HostTensorDesc(STDtype.U32, _shp(len(v)), _u32_le(v)))


def _add_u64(mut names: List[String], mut descs: List[HostTensorDesc], name: String, v: List[UInt64]):
    names.append(name)
    descs.append(HostTensorDesc(STDtype.U64, _shp(len(v)), _u64_le(v)))


def _add_u8(mut names: List[String], mut descs: List[HostTensorDesc], name: String, v: List[UInt8]):
    names.append(name)
    descs.append(HostTensorDesc(STDtype.U8, _shp(len(v)), v.copy()))


# ── read-back helpers (dtype-checked, LE via ptr bitcast) ─────────────────────
def _rd_f32(st: SafeTensors, name: String) raises -> List[Float32]:
    var info = st.tensor_info(name)
    if info.dtype != STDtype.F32:
        raise Error(String("sidecar: ") + name + " not F32")
    var bytes = st.tensor_bytes(name)
    var p = bytes.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32]()
    for i in range(info.size // 4):
        out.append(p[i])
    return out^


def _rd_f64(st: SafeTensors, name: String) raises -> List[Float64]:
    var info = st.tensor_info(name)
    if info.dtype != STDtype.F64:
        raise Error(String("sidecar: ") + name + " not F64")
    var bytes = st.tensor_bytes(name)
    var p = bytes.unsafe_ptr().bitcast[Float64]()
    var out = List[Float64]()
    for i in range(info.size // 8):
        out.append(p[i])
    return out^


def _rd_u32(st: SafeTensors, name: String) raises -> List[UInt32]:
    var info = st.tensor_info(name)
    if info.dtype != STDtype.U32:
        raise Error(String("sidecar: ") + name + " not U32")
    var bytes = st.tensor_bytes(name)
    var p = bytes.unsafe_ptr().bitcast[UInt32]()
    var out = List[UInt32]()
    for i in range(info.size // 4):
        out.append(p[i])
    return out^


def _rd_u64(st: SafeTensors, name: String) raises -> List[UInt64]:
    var info = st.tensor_info(name)
    if info.dtype != STDtype.U64:
        raise Error(String("sidecar: ") + name + " not U64")
    var bytes = st.tensor_bytes(name)
    var p = bytes.unsafe_ptr().bitcast[UInt64]()
    var out = List[UInt64]()
    for i in range(info.size // 8):
        out.append(p[i])
    return out^


def _rd_u8(st: SafeTensors, name: String) raises -> List[UInt8]:
    var info = st.tensor_info(name)
    if info.dtype != STDtype.U8:
        raise Error(String("sidecar: ") + name + " not U8")
    var bytes = st.tensor_bytes(name)
    var out = List[UInt8]()
    for i in range(info.size):
        out.append(bytes[i])
    return out^


# ── automagic3 sign-history bit-pack / unpack (1 bit/element, planes concat) ──
def _pack_signhist(planes: List[List[Bool]], numel: Int) -> List[UInt8]:
    var bpp = (numel + 7) // 8
    var out = List[UInt8]()
    for pi in range(len(planes)):
        var base = len(out)
        for _ in range(bpp):
            out.append(UInt8(0))
        for i in range(numel):
            if planes[pi][i]:
                out[base + (i // 8)] = out[base + (i // 8)] | (UInt8(1) << UInt8(i % 8))
    return out^


def _unpack_signhist(bytes: List[UInt8], hist_fill: Int, numel: Int) -> List[List[Bool]]:
    var bpp = (numel + 7) // 8
    var out = List[List[Bool]]()
    for pi in range(hist_fill):
        var base = pi * bpp
        var plane = List[Bool]()
        for i in range(numel):
            var bit = (bytes[base + (i // 8)] >> UInt8(i % 8)) & UInt8(1)
            plane.append(bit == UInt8(1))
        out.append(plane^)
    return out^


def _k(i: Int) -> String:
    return String(i)


# ═════════════════════════════════════════════════════════════════════════════
# SAVE
# ═════════════════════════════════════════════════════════════════════════════
def levers_optimizer_sidecar_path(base: String) -> String:
    return base + String(".optstate.safetensors")


def levers_optimizer_sidecar_save(
    st: LeversOptimizerState, k: Int, seed: UInt64, path: String
) raises -> Int:
    """Host-direct save of the whole LeversOptimizerState. Returns n_states."""
    _require(st.initialized, "sidecar_save: LeversOptimizerState not initialized")
    var names = List[String]()
    var descs = List[HostTensorDesc]()

    var n_states = 0
    if st.kind == TRAIN_OPTIMIZER_ADAFACTOR:
        n_states = len(st.ada)
        for j in range(n_states):
            _add_f32(names, descs, String("af.") + _k(j) + ".row_var", st.ada[j].row_var)
            _add_f32(names, descs, String("af.") + _k(j) + ".col_var", st.ada[j].col_var)
            var sc = List[UInt32]()
            sc.append(UInt32(st.ada[j].step))
            _add_u32(names, descs, String("af.") + _k(j) + ".step", sc)
    elif st.kind == TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW:
        n_states = len(st.sf)
        for j in range(n_states):
            _add_f32(names, descs, String("sf.") + _k(j) + ".exp_avg", st.sf[j].exp_avg)
            _add_f32(names, descs, String("sf.") + _k(j) + ".exp_avg_sq", st.sf[j].exp_avg_sq)
            _add_f32(names, descs, String("sf.") + _k(j) + ".kahan", st.sf[j].kahan_comp)
        var ck = List[UInt32]()
        ck.append(UInt32(st.sf_ctl.k))
        ck.append(UInt32(1) if st.sf_ctl.train_mode else UInt32(0))
        _add_u32(names, descs, String("sf.ctl.u32"), ck)
        var cf = List[Float64]()
        cf.append(st.sf_ctl.lr_max)
        cf.append(st.sf_ctl.last_lr)
        _add_f64(names, descs, String("sf.ctl.f64"), cf)
    elif st.kind == TRAIN_OPTIMIZER_ADAMW_8BIT:
        n_states = len(st.a8)
        for j in range(n_states):
            _add_u8(names, descs, String("a8.") + _k(j) + ".m_codes", st.a8[j].m_codes)
            _add_u8(names, descs, String("a8.") + _k(j) + ".v_codes", st.a8[j].v_codes)
            _add_f32(names, descs, String("a8.") + _k(j) + ".m_absmax", st.a8[j].m_absmax)
            _add_f32(names, descs, String("a8.") + _k(j) + ".v_absmax", st.a8[j].v_absmax)
            var sc = List[UInt32]()
            sc.append(UInt32(st.a8[j].step))
            _add_u32(names, descs, String("a8.") + _k(j) + ".step", sc)
    elif st.kind == TRAIN_OPTIMIZER_AUTOMAGIC3:
        n_states = len(st.auto3)
        for j in range(n_states):
            ref s = st.auto3[j]
            var scal = List[UInt32]()
            scal.append(UInt32(1) if s.factored else UInt32(0))
            scal.append(UInt32(s.rows))
            scal.append(UInt32(s.cols))
            scal.append(UInt32(s.numel))
            scal.append(UInt32(s.h))
            scal.append(UInt32(s.hist_idx))
            scal.append(UInt32(s.hist_fill))
            scal.append(UInt32(s.step))
            _add_u32(names, descs, String("auto3.") + _k(j) + ".scalars", scal)
            if s.factored:
                _add_f32(names, descs, String("auto3.") + _k(j) + ".rowvar", s.row_var)
                _add_f32(names, descs, String("auto3.") + _k(j) + ".colvar", s.col_var)
            else:
                _add_f32(names, descs, String("auto3.") + _k(j) + ".v", s.v)
            _add_u8(names, descs, String("auto3.") + _k(j) + ".signhist",
                    _pack_signhist(s.sign_history, s.numel))
        # ctl
        var clr = List[Float64]()
        clr.append(st.auto3_ctl.lr)
        _add_f64(names, descs, String("auto3.ctl.lr"), clr)
        var ci = List[UInt32]()
        ci.append(UInt32(1) if st.auto3_ctl.initialized else UInt32(0))
        _add_u32(names, descs, String("auto3.ctl.init"), ci)
        var crng = List[UInt64]()
        crng.append(st.auto3_ctl.rng.state)
        _add_u64(names, descs, String("auto3.ctl.rng"), crng)
    else:
        raise Error(String("sidecar_save: optimizer tag ") + String(st.kind)
                    + " has no levers dispatch")

    # meta LAST (so it exists once): [tag, k, n_states, seed_lo, seed_hi, version]
    var meta = List[UInt32]()
    meta.append(UInt32(st.kind))
    meta.append(UInt32(k))
    meta.append(UInt32(n_states))
    meta.append(UInt32(seed & UInt64(0xFFFFFFFF)))
    meta.append(UInt32((seed >> 32) & UInt64(0xFFFFFFFF)))
    meta.append(UInt32(SIDECAR_VERSION))
    _add_u32(names, descs, String("__meta__"), meta)

    save_safetensors_host(names, descs, path)
    return n_states


# ═════════════════════════════════════════════════════════════════════════════
# LOAD
# ═════════════════════════════════════════════════════════════════════════════
struct LeversSidecarResume(Movable):
    var state: LeversOptimizerState
    var k: Int
    var seed: UInt64

    def __init__(out self, var state: LeversOptimizerState, k: Int, seed: UInt64):
        self.state = state^
        self.k = k
        self.seed = seed

    def take_state(deinit self) -> LeversOptimizerState:
        return self.state^


def levers_optimizer_sidecar_load(
    path: String, expected_kind: Int, expected_n_states: Int
) raises -> LeversSidecarResume:
    """Restore LeversOptimizerState (initialized=True) + k from the sidecar.
    Fails loud on tag / count / version mismatch. Shapes are self-describing."""
    var st_file = SafeTensors.open(path)
    var meta = _rd_u32(st_file, String("__meta__"))
    _require(len(meta) == 6, "sidecar_load: __meta__ must be U32[6]")
    var tag = Int(meta[0])
    var k = Int(meta[1])
    var n_states = Int(meta[2])
    var seed = UInt64(meta[3]) | (UInt64(meta[4]) << 32)
    var version = Int(meta[5])
    _require(version == SIDECAR_VERSION,
             String("sidecar_load: version ") + String(version) + " != " + String(SIDECAR_VERSION))
    _require(tag == expected_kind,
             String("sidecar_load: optimizer tag ") + String(tag)
             + " != expected " + String(expected_kind))
    _require(n_states == expected_n_states,
             String("sidecar_load: n_states ") + String(n_states)
             + " != expected " + String(expected_n_states))

    var out = LeversOptimizerState()
    if tag == TRAIN_OPTIMIZER_ADAFACTOR:
        for j in range(n_states):
            var rv = _rd_f32(st_file, String("af.") + _k(j) + ".row_var")
            var cv = _rd_f32(st_file, String("af.") + _k(j) + ".col_var")
            var stp = _rd_u32(st_file, String("af.") + _k(j) + ".step")
            var s = AdafactorState(len(rv), len(cv))
            s.row_var = rv^
            s.col_var = cv^
            s.step = Int(stp[0])
            out.ada.append(s^)
    elif tag == TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW:
        for j in range(n_states):
            var ea = _rd_f32(st_file, String("sf.") + _k(j) + ".exp_avg")
            var es = _rd_f32(st_file, String("sf.") + _k(j) + ".exp_avg_sq")
            var kh = _rd_f32(st_file, String("sf.") + _k(j) + ".kahan")
            var s = AdamWScheduleFreeState(len(ea))
            s.exp_avg = ea^
            s.exp_avg_sq = es^
            s.kahan_comp = kh^
            out.sf.append(s^)
        var cu = _rd_u32(st_file, String("sf.ctl.u32"))
        var cf = _rd_f64(st_file, String("sf.ctl.f64"))
        out.sf_ctl.k = Int(cu[0])
        out.sf_ctl.train_mode = cu[1] == UInt32(1)
        out.sf_ctl.lr_max = cf[0]
        out.sf_ctl.last_lr = cf[1]
    elif tag == TRAIN_OPTIMIZER_ADAMW_8BIT:
        out.a8_qmap_signed = adam8bit_create_dynamic_map(True)
        out.a8_qmap_unsigned = adam8bit_create_dynamic_map(False)
        for j in range(n_states):
            var mc = _rd_u8(st_file, String("a8.") + _k(j) + ".m_codes")
            var vc = _rd_u8(st_file, String("a8.") + _k(j) + ".v_codes")
            var ma = _rd_f32(st_file, String("a8.") + _k(j) + ".m_absmax")
            var va = _rd_f32(st_file, String("a8.") + _k(j) + ".v_absmax")
            var stp = _rd_u32(st_file, String("a8.") + _k(j) + ".step")
            var s = Adam8bitState(len(mc))
            s.m_codes = mc^
            s.v_codes = vc^
            s.m_absmax = ma^
            s.v_absmax = va^
            s.step = Int(stp[0])
            out.a8.append(s^)
    elif tag == TRAIN_OPTIMIZER_AUTOMAGIC3:
        for j in range(n_states):
            var scal = _rd_u32(st_file, String("auto3.") + _k(j) + ".scalars")
            _require(len(scal) == 8, "sidecar_load: auto3 scalars must be U32[8]")
            var factored = scal[0] == UInt32(1)
            var rows = Int(scal[1])
            var cols = Int(scal[2])
            var numel = Int(scal[3])
            var h = Int(scal[4])
            var hist_idx = Int(scal[5])
            var hist_fill = Int(scal[6])
            var step = Int(scal[7])
            var s: Automagic3State
            if factored:
                s = Automagic3State(rows, cols, h)
                s.row_var = _rd_f32(st_file, String("auto3.") + _k(j) + ".rowvar")
                s.col_var = _rd_f32(st_file, String("auto3.") + _k(j) + ".colvar")
            else:
                s = Automagic3State(numel, h)
                s.v = _rd_f32(st_file, String("auto3.") + _k(j) + ".v")
            var sh = _rd_u8(st_file, String("auto3.") + _k(j) + ".signhist")
            s.sign_history = _unpack_signhist(sh, hist_fill, numel)
            s.hist_idx = hist_idx
            s.hist_fill = hist_fill
            s.step = step
            out.auto3.append(s^)
        var clr = _rd_f64(st_file, String("auto3.ctl.lr"))
        var ci = _rd_u32(st_file, String("auto3.ctl.init"))
        var cr = _rd_u64(st_file, String("auto3.ctl.rng"))
        out.auto3_ctl.lr = clr[0]
        out.auto3_ctl.initialized = ci[0] == UInt32(1)
        out.auto3_ctl.rng = Automagic3Rng(cr[0])
    else:
        raise Error(String("sidecar_load: optimizer tag ") + String(tag)
                    + " has no levers dispatch")

    out.kind = tag
    out.start = 0
    out.end = n_states // 2
    out.initialized = True
    return LeversSidecarResume(out^, k, seed)
