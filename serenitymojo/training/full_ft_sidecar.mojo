# training/full_ft_sidecar.mojo — the fleet FULL-FINETUNE RESUME SIDECAR
# (the last v1 delta owed by reference-mojo-full-finetune / FULL_FINETUNE_ROLLOUT_PLAN
# 2026-07-07: "resume: adafactor row/col sidecar + store reload not wired").
#
# The full-FT arms train through a pinned-host bf16 store (the store IS the
# model) + fused device Adafactor (training/adafactor_device.mojo). At save
# time the store's trained bytes go to a safetensors OVERLAY (original ckpt
# key names), but the optimizer's factored second moments (row_var/col_var,
# ~2*(N+K) F32 per matrix — ~20MB for all of krea2) died with the process, so
# every arm fail-louded on resume. This module makes resume real:
#
#   * full_ft_sidecar_save   — D2H every state's row_var/col_var into ONE
#     safetensors sidecar next to the overlay. Keys: `af.<i>.row_var` (F32
#     [rows]) / `af.<i>.col_var` (F32 [cols]) in the model's FLAT state order
#     (the exact order the trainer built its af_states list — the same index
#     discipline as training/full_finetune_contract.mojo's `param.<i>`/`adam_
#     {m,v}.<i>` scheme), plus ONE meta tensor `__af_meta__` (U32 [5] =
#     [t_step, n_states, seed_lo, seed_hi, contract_version]). U32 (not the
#     contract's F32 `__meta__` pair) because seed halves are exact 32-bit
#     values an F32 mantissa cannot carry; a distinct key name so nothing
#     mistakes this for the AdamW master/moment sidecar spec.
#   * full_ft_sidecar_load   — validates count + per-state shapes + dtype +
#     contract version FAIL-LOUD, uploads the states back to the device, and
#     returns (states, t_step, seed_base).
#   * full_ft_overlay_into_host_store — the store-rebuild half: the store
#     builders already read the BASE checkpoint; resume = base THEN overlay.
#     Generic over models: takes PARALLEL LISTS (names, pinned host buffers,
#     nbytes) in the store's flat order and copies each trained tensor's bytes
#     over the pinned bytes (byte-exact memcpy, dtype/size-checked) — each
#     model's store type calls this with its own flattening, no per-model
#     copies of the loop.
#
# Resume continuity contract (what the caller must keep):
#   * t_step in the sidecar = COMPLETED optimizer steps. The resumed loop
#     continues at global step index t_step with the SAME seed_base — the
#     per-step sigma/noise/SR streams (seed_base+step, seed_base*7919+step,
#     ...) then continue exactly where run A would have been.
#   * seed_base mismatch on load is an ERROR (silently different noise
#     streams would corrupt the continuation).
#
# Mojo 1.0.0b1, NVIDIA GPU (device states; the file I/O itself is host-only).

from std.collections import List
from std.memory import ArcPointer
from max.gpu.host import DeviceContext, HostBuffer
from std.builtin.dtype import DType

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.ffi import BytePtr, sys_memcpy
from serenitymojo.training.adafactor_device import AdafactorDeviceState

comptime TArc = ArcPointer[Tensor]
comptime HArc = ArcPointer[HostBuffer[DType.uint8]]

comptime FULL_FT_SIDECAR_META_KEY = "__af_meta__"
comptime FULL_FT_SIDECAR_CONTRACT_VERSION = 1
comptime FULL_FT_SIDECAR_CONTRACT_VERSION_V2 = 2
comptime FULL_FT_SIDECAR_FACTORED_KEY = "__af_factored__"
comptime _META_FIELDS = 5   # [t_step, n_states, seed_lo, seed_hi, version]

# ── contract v2 (FULL_SURFACE_PLAN_2026-07-08 Phase A: MIXED states) ─────────
# v1 states are all 2D-FACTORED (row_var+col_var). The full-surface arms add
# 1D UNFACTORED states (exp_avg_sq [n], AdafactorDeviceState.factored=False,
# cols==0 sentinel). v2 adds ONE tensor `__af_factored__` (U8 [n_states],
# 1=factored 0=unfactored, flat state order) and saves an unfactored state as
# ONLY `af.<i>.row_var` (= exp_avg_sq [n]); the count check becomes computed
# (2*n_factored + n_unfactored + 2). BACK/FORWARD COMPAT, both fail-loud:
#   * SAVE emits v2 ONLY when any state is unfactored — an all-factored save
#     stays version 1 and BYTE-IDENTICAL to today (same keys, same order, no
#     flag tensor), so the fleet's byte-equal resume gate classes
#     (klein/hidream/i4/zimage) are untouched.
#   * LOAD of a version-1 sidecar (no `__af_factored__`) = all-factored — the
#     exact pre-v2 code path. Expected factoring comes from the caller's
#     shape lists: expected_cols[i] == 0 means state i is expected UNFACTORED
#     (the AdafactorDeviceState sentinel); a v1 file cannot satisfy an
#     unfactored expectation (error, never a silent reinterpret).


def full_ft_sidecar_path_for_overlay(overlay_path: String) -> String:
    """Canonical sidecar location, derived from the overlay path (kept next to
    it, like the LoRA trainers' `.state` files): `<overlay>.afstate.safetensors`.
    """
    return overlay_path + String(".afstate.safetensors")


def _af_row_key(i: Int) -> String:
    return String("af.") + String(i) + String(".row_var")


def _af_col_key(i: Int) -> String:
    return String("af.") + String(i) + String(".col_var")


# ── U32[5] meta tensor (device-resident so save_safetensors D2Hs it like the
# states; 20 bytes) ───────────────────────────────────────────────────────────
def _meta_tensor(
    t_step: Int, n_states: Int, seed_base: UInt64, version: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    if t_step < 0 or t_step > 0xFFFFFFFF:
        raise Error(
            String("full_ft_sidecar: t_step out of u32 range: ") + String(t_step)
        )
    if n_states <= 0 or n_states > 0xFFFFFFFF:
        raise Error(
            String("full_ft_sidecar: n_states out of range: ") + String(n_states)
        )
    var vals = List[UInt32]()
    vals.append(UInt32(t_step))
    vals.append(UInt32(n_states))
    vals.append(UInt32(seed_base & UInt64(0xFFFFFFFF)))
    vals.append(UInt32((seed_base >> 32) & UInt64(0xFFFFFFFF)))
    vals.append(UInt32(version))
    var nbytes = _META_FIELDS * 4
    var host = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
    ctx.synchronize()
    var p = host.unsafe_ptr()
    for i in range(_META_FIELDS):
        var v = vals[i]
        p[i * 4 + 0] = UInt8(v & UInt32(0xFF))
        p[i * 4 + 1] = UInt8((v >> 8) & UInt32(0xFF))
        p[i * 4 + 2] = UInt8((v >> 16) & UInt32(0xFF))
        p[i * 4 + 3] = UInt8((v >> 24) & UInt32(0xFF))
    var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    ctx.synchronize()
    return Tensor(dev^, [_META_FIELDS], STDtype.U32)


def _read_meta_u32(st: SafeTensors, idx: Int) raises -> UInt32:
    """LE u32 field `idx` of the `__af_meta__` tensor (dtype/shape validated by
    the caller once)."""
    var span = st.tensor_bytes(String(FULL_FT_SIDECAR_META_KEY))
    var base = idx * 4
    var v = UInt32(0)
    for b in range(4):
        v = v | (UInt32(span[base + b]) << UInt32(8 * b))
    return v


# ── U8[n_states] per-state factored flags (v2 only; 1=factored 0=unfactored) ─
def _factored_tensor(
    states: List[AdafactorDeviceState], ctx: DeviceContext
) raises -> Tensor:
    var n = len(states)
    var host = ctx.enqueue_create_host_buffer[DType.uint8](n)
    ctx.synchronize()
    var p = host.unsafe_ptr()
    for i in range(n):
        p[i] = UInt8(1) if states[i].factored else UInt8(0)
    var dev = ctx.enqueue_create_buffer[DType.uint8](n)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    ctx.synchronize()
    return Tensor(dev^, [n], STDtype.U8)


def _read_factored_flag(st: SafeTensors, idx: Int) raises -> Bool:
    """Byte `idx` of the `__af_factored__` tensor (dtype/shape validated by
    the caller once). Values other than 0/1 are corruption — fail-loud."""
    var span = st.tensor_bytes(String(FULL_FT_SIDECAR_FACTORED_KEY))
    var b = span[idx]
    if b == UInt8(1):
        return True
    if b == UInt8(0):
        return False
    raise Error(
        String("full_ft_sidecar_load: __af_factored__[") + String(idx)
        + String("] = ") + String(Int(b)) + String(" is not 0/1")
    )


# ── (i) SAVE: states + step + seed → ONE sidecar safetensors ─────────────────
def full_ft_sidecar_save(
    states: List[AdafactorDeviceState],
    t_step: Int,
    sr_seed_base: UInt64,
    path: String,
    ctx: DeviceContext,
) raises:
    """Write every Adafactor state (F32 device) plus the resume meta
    ([t_step, n_states, seed_lo, seed_hi, version] U32) to ONE safetensors
    sidecar at `path`. Factored states save `af.<i>.{row,col}_var`; UNFACTORED
    (1D) states save ONLY `af.<i>.row_var` (= exp_avg_sq [n]). Keys in the
    trainer's FLAT af_states order. Version emission: 2 (+ `__af_factored__`
    U8[n_states]) ONLY if any state is unfactored; an all-factored save stays
    version 1 BYTE-IDENTICAL to the pre-v2 writer (the fleet byte-equal gate
    contract). The writer D2Hs each tensor raw (byte-exact F32) and the write
    is atomic (tmp + rename — save_safetensors)."""
    if len(states) == 0:
        raise Error("full_ft_sidecar_save: empty state list")
    var any_unfactored = False
    for i in range(len(states)):
        if not states[i].factored:
            any_unfactored = True
    var names = List[String]()
    var tensors = List[TArc]()
    for i in range(len(states)):
        if states[i].row_var[].dtype() != STDtype.F32:
            raise Error("full_ft_sidecar_save: row_var must be F32")
        if states[i].row_var[].numel() != states[i].rows:
            raise Error("full_ft_sidecar_save: row_var numel != rows")
        names.append(_af_row_key(i))
        tensors.append(states[i].row_var.copy())
        if states[i].factored:
            if states[i].col_var[].dtype() != STDtype.F32:
                raise Error("full_ft_sidecar_save: col_var must be F32")
            if states[i].col_var[].numel() != states[i].cols:
                raise Error("full_ft_sidecar_save: col_var numel != cols")
            names.append(_af_col_key(i))
            tensors.append(states[i].col_var.copy())
        elif states[i].cols != 0:
            raise Error("full_ft_sidecar_save: unfactored state cols != 0")
    var version = FULL_FT_SIDECAR_CONTRACT_VERSION
    if any_unfactored:
        version = FULL_FT_SIDECAR_CONTRACT_VERSION_V2
        names.append(String(FULL_FT_SIDECAR_FACTORED_KEY))
        tensors.append(TArc(_factored_tensor(states, ctx)))
    names.append(String(FULL_FT_SIDECAR_META_KEY))
    tensors.append(TArc(_meta_tensor(t_step, len(states), sr_seed_base, version, ctx)))
    save_safetensors(names, tensors, path, ctx)
    print(
        "[full-ft-sidecar] wrote", len(states), "adafactor states (t_step",
        t_step, ", contract v", version, ") ->", path,
    )


# ── (ii) LOAD: sidecar → device states + t_step + seed (fail-loud) ───────────
struct FullFTSidecarResume(Movable):
    var states: List[AdafactorDeviceState]
    var t_step: Int
    var seed_base: UInt64

    def __init__(
        out self,
        var states: List[AdafactorDeviceState],
        t_step: Int,
        seed_base: UInt64,
    ):
        self.states = states^
        self.t_step = t_step
        self.seed_base = seed_base

    def take_states(deinit self) -> List[AdafactorDeviceState]:
        return self.states^


def _upload_f32(
    st: SafeTensors, name: String, expect_n: Int, dst: Tensor, ctx: DeviceContext
) raises:
    """H2D one F32 sidecar tensor's bytes into the (already-allocated) device
    tensor `dst`. Dtype/size fail-loud; byte-exact (no cast)."""
    var info = st.tensor_info(name)
    if info.dtype != STDtype.F32:
        raise Error(String("full_ft_sidecar_load: ") + name + String(" is not F32"))
    var nbytes = expect_n * 4
    if info.size != nbytes:
        raise Error(
            String("full_ft_sidecar_load: ") + name + String(" size ")
            + String(info.size) + String(" != expected ") + String(nbytes)
        )
    if dst.nbytes() != nbytes:
        raise Error(String("full_ft_sidecar_load: dst tensor size mismatch for ") + name)
    var span = st.tensor_bytes(name)
    var host = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
    ctx.synchronize()
    var dstp = BytePtr(unsafe_from_address=Int(host.unsafe_ptr()))
    var srcp = BytePtr(unsafe_from_address=Int(span.unsafe_ptr()))
    _ = sys_memcpy(dstp, srcp, nbytes)
    ctx.enqueue_copy(dst_buf=dst.buf, src_buf=host)
    ctx.synchronize()


def full_ft_sidecar_load(
    path: String,
    expected_rows: List[Int],
    expected_cols: List[Int],
    ctx: DeviceContext,
) raises -> FullFTSidecarResume:
    """Load the Adafactor resume sidecar written by full_ft_sidecar_save.
    `expected_rows/cols` are the trainer's FLAT per-state shapes (built off the
    live store, the same loop that builds fresh af_states); expected_cols[i]
    == 0 means state i is expected UNFACTORED (exp_avg_sq [expected_rows[i]]
    — the AdafactorDeviceState sentinel). State count, every shape and every
    factored flag must match FAIL-LOUD. Version dispatch: a version-1 sidecar
    (all-factored, no `__af_factored__`) loads on the exact pre-v2 path; a
    version-2 sidecar dispatches per state on the flag with a COMPUTED count
    check. Returns device-resident states + t_step (completed steps) +
    seed_base."""
    if len(expected_rows) != len(expected_cols) or len(expected_rows) == 0:
        raise Error("full_ft_sidecar_load: bad expected shape lists")
    var n_unf_expected = 0
    for i in range(len(expected_cols)):
        if expected_cols[i] == 0:
            n_unf_expected += 1
    var st = SafeTensors.open(path)

    # meta first (dtype/shape gate, then fields).
    var minfo = st.tensor_info(String(FULL_FT_SIDECAR_META_KEY))
    if minfo.dtype != STDtype.U32:
        raise Error("full_ft_sidecar_load: __af_meta__ is not U32")
    if minfo.size != _META_FIELDS * 4:
        raise Error("full_ft_sidecar_load: __af_meta__ must be U32[5]")
    var t_step = Int(_read_meta_u32(st, 0))
    var n_states = Int(_read_meta_u32(st, 1))
    var seed_lo = UInt64(_read_meta_u32(st, 2))
    var seed_hi = UInt64(_read_meta_u32(st, 3))
    var version = Int(_read_meta_u32(st, 4))
    var seed_base = seed_lo | (seed_hi << 32)
    if (
        version != FULL_FT_SIDECAR_CONTRACT_VERSION
        and version != FULL_FT_SIDECAR_CONTRACT_VERSION_V2
    ):
        raise Error(
            String("full_ft_sidecar_load: contract version ") + String(version)
            + String(" not in {") + String(FULL_FT_SIDECAR_CONTRACT_VERSION)
            + String(", ") + String(FULL_FT_SIDECAR_CONTRACT_VERSION_V2)
            + String("}")
        )
    if n_states != len(expected_rows):
        raise Error(
            String("full_ft_sidecar_load: sidecar has ") + String(n_states)
            + String(" states, model expects ") + String(len(expected_rows))
        )
    if t_step <= 0:
        raise Error("full_ft_sidecar_load: t_step must be >= 1")

    if version == FULL_FT_SIDECAR_CONTRACT_VERSION:
        # ── v1: all-factored — the exact pre-v2 path (back-compat gate class).
        if n_unf_expected != 0:
            raise Error(
                String("full_ft_sidecar_load: version-1 sidecar is all-")
                + String("factored but the model expects ")
                + String(n_unf_expected)
                + String(" unfactored (1D) states — wrong sidecar for this")
                + String(" trained surface")
            )
        # exactly the expected tensor population (2 per state + meta) — a
        # merged or truncated file is an error, not a warning.
        if st.count() != 2 * n_states + 1:
            raise Error(
                String("full_ft_sidecar_load: tensor count ") + String(st.count())
                + String(" != ") + String(2 * n_states + 1)
            )
        var states = List[AdafactorDeviceState]()
        for i in range(n_states):
            var rows = expected_rows[i]
            var cols = expected_cols[i]
            var stt = AdafactorDeviceState(rows, cols, ctx)
            _upload_f32(st, _af_row_key(i), rows, stt.row_var[], ctx)
            _upload_f32(st, _af_col_key(i), cols, stt.col_var[], ctx)
            states.append(stt^)
        print(
            "[full-ft-sidecar] loaded", n_states, "adafactor states (t_step",
            t_step, ") <-", path,
        )
        return FullFTSidecarResume(states^, t_step, seed_base)

    # ── v2: mixed states — per-state dispatch on `__af_factored__`.
    var finfo = st.tensor_info(String(FULL_FT_SIDECAR_FACTORED_KEY))
    if finfo.dtype != STDtype.U8:
        raise Error("full_ft_sidecar_load: __af_factored__ is not U8")
    if finfo.size != n_states:
        raise Error(
            String("full_ft_sidecar_load: __af_factored__ size ")
            + String(finfo.size) + String(" != n_states ") + String(n_states)
        )
    var n_factored = 0
    for i in range(n_states):
        if _read_factored_flag(st, i):
            n_factored += 1
    # computed population: row(+col) per state + flags + meta.
    var expect_count = 2 * n_factored + (n_states - n_factored) + 2
    if st.count() != expect_count:
        raise Error(
            String("full_ft_sidecar_load: tensor count ") + String(st.count())
            + String(" != ") + String(expect_count)
        )
    var states = List[AdafactorDeviceState]()
    for i in range(n_states):
        var rows = expected_rows[i]
        var cols = expected_cols[i]
        var flag = _read_factored_flag(st, i)
        if flag != (cols != 0):
            raise Error(
                String("full_ft_sidecar_load: state ") + String(i)
                + String(" factored flag ") + String(1 if flag else 0)
                + String(" != model expectation (cols=") + String(cols)
                + String(")")
            )
        if flag:
            var stt = AdafactorDeviceState(rows, cols, ctx)
            _upload_f32(st, _af_row_key(i), rows, stt.row_var[], ctx)
            _upload_f32(st, _af_col_key(i), cols, stt.col_var[], ctx)
            states.append(stt^)
        else:
            var stt = AdafactorDeviceState(rows, ctx)   # unfactored [rows]
            _upload_f32(st, _af_row_key(i), rows, stt.row_var[], ctx)
            states.append(stt^)
    print(
        "[full-ft-sidecar] loaded", n_states, "adafactor states (",
        n_factored, "factored,", n_states - n_factored,
        "unfactored; t_step", t_step, ") <-", path,
    )
    return FullFTSidecarResume(states^, t_step, seed_base)


# ── (iii) store rebuild: base ckpt (the builders) THEN this overlay ──────────
def full_ft_overlay_into_host_store(
    overlay_path: String,
    names: List[String],
    bufs: List[HArc],
    nbytes: List[Int],
    expected_total: Int = -1,
) raises:
    """Copy each trained tensor's bytes from the overlay safetensors OVER the
    pinned host store bytes (byte-exact memcpy, BF16 + size checked — the reference trainer
    full-FT contract is bf16 weights). `names`/`bufs`/`nbytes` are PARALLEL
    lists in the model store's flat order; the overlay must contain these
    tensors (the *_host_bf16_save inverse). By default the overlay must hold
    EXACTLY these keys (extra or missing keys are errors). `expected_total`
    overrides the total-count check for models whose save writes MORE tensors
    than land in the host store — e.g. flux full-surface, where the 152
    device-resident mod.lin tensors are saved into the same overlay but loaded
    separately by the caller; pass 760 there so the 608 store names still
    overlay while a v1 (228) file still fails loud (228 != 760). The store
    must already hold the BASE checkpoint (the builders)."""
    if len(names) != len(bufs) or len(names) != len(nbytes) or len(names) == 0:
        raise Error("full_ft_overlay_into_host_store: bad parallel lists")
    var st = SafeTensors.open(overlay_path)
    var want_total = len(names) if expected_total < 0 else expected_total
    if st.count() != want_total:
        raise Error(
            String("full_ft_overlay_into_host_store: overlay has ")
            + String(st.count()) + String(" tensors, expected ")
            + String(want_total) + String(" — wrong overlay for this model? ")
            + overlay_path
        )
    for i in range(len(names)):
        var info = st.tensor_info(names[i])   # raises if missing
        if info.dtype != STDtype.BF16:
            raise Error(
                String("full_ft_overlay_into_host_store: ") + names[i]
                + String(" is not BF16 — quantized/other-dtype full-FT overlays are forbidden")
            )
        if info.size != nbytes[i]:
            raise Error(
                String("full_ft_overlay_into_host_store: ") + names[i]
                + String(" size ") + String(info.size)
                + String(" != store nbytes ") + String(nbytes[i])
            )
        var span = st.tensor_bytes(names[i])
        var dstp = BytePtr(unsafe_from_address=Int(bufs[i][].unsafe_ptr()))
        var srcp = BytePtr(unsafe_from_address=Int(span.unsafe_ptr()))
        _ = sys_memcpy(dstp, srcp, nbytes[i])
    print(
        "[full-ft-sidecar] overlaid", len(names), "trained tensors into the",
        "pinned host store <-", overlay_path,
    )
