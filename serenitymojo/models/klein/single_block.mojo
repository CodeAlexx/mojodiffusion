# serenitymojo/models/klein/single_block.mojo
#
# Klein (FLUX.2) SINGLE-STREAM DiT block: forward (saving activations) +
# hand-chained backward (training), packaged as a reusable unit in the EXACT
# style proven by serenitymojo/models/klein/double_block.mojo (the double-stream
# block, gated 28/28 vs torch). This is that pattern HALVED + flattened: ONE
# stream, a parallel attention+MLP FLUX single block (NO img/txt coupling).
#
# DEVICE-RESIDENT INTERIOR (Increment 1 perf refactor, 2026-05-30)
#   The PUBLIC API is unchanged: `x` enters as host `List[Float32]`, `out` and all
#   grads leave as host `List[Float32]` (so klein_stack_lora.mojo and the parity
#   gates compile with ZERO changes). But INTERNALLY the chain now threads device
#   `Tensor`s op-to-op: `from_host(x)` runs ONCE at the forward entry, `.to_host`
#   runs ONCE on the returned `out`; saved activations are device `Tensor`s moved
#   straight from the producing op into `SingleBlockSaved` (no `.to_host()` /
#   `.clone()` per intermediate). The OLD code bounced EVERY intermediate op's
#   output to host (`.to_host` → `from_host`), forcing ~70 host-stall syncs per
#   block (from_host syncs, to_host syncs). Removing the per-INTERMEDIATE bounce
#   is the entire win; the boundary from_host/to_host stay.
#
#   Tensor is MOVE-ONLY, so `SingleBlockSaved` / `SingleBlockForward` are now
#   Movable-ONLY (Copyable dropped). VERIFIED no caller `.copy()`s a
#   SingleBlockForward or its `.saved`: klein_stack.mojo / klein_stack_lora.mojo
#   and both parity gates construct `var fwd = ...forward(...)` then read
#   `fwd.saved` BY BORROW into backward (they only `.copy()` the `out` List).
#
#   The qkv|gate_up channel split, the q/k/v split, and their backward scatters
#   are now DEVICE slice/concat (ops/tensor_algebra.slice + .concat) instead of
#   host row-loops. reshape [S,D]<->[1,S,H,Dh] is a row-major byte no-op, so it
#   is just a Tensor reshape (ops/tensor_algebra.reshape) — same bytes.
#
# WHY HOST List[Float32] STILL AT THE API BOUNDARY
#   The boundary contract is fixed by the callers (stack + gates pass host lists).
#   The LoRA-delta helpers have device-resident siblings, so LoRA activations and
#   adapter A/B tensors stay on device in the hot trainer path; only d_A/d_B
#   leave for the existing host optimizer state. The base chain is fully
#   device-resident.
#
# FORWARD GRAPH (mirrors models/dit/klein_dit.mojo `_single_block`, lines 354-390)
#   With precomputed AdaLN vectors (shift, scale, gate) each [D] from single_mod:
#     x_norm   = modulate(layer_norm(x,1,0,eps), scale, shift)   # (1+scale)*LN+shift
#     fused    = linear(x_norm, W1)                              # [1,S, 3D+2F]
#     qkv      = fused[:, :, :3D]    ; gate_up = fused[:, :, 3D:3D+2F]  (CHANNEL slice)
#     q,k,v    = split qkv into 3x [1,S,H,Dh]
#     q        = rms_norm(q, q_norm[Dh]) ; k = rms_norm(k, k_norm[Dh])  (eps 1e-6)
#     att      = sdpa_nomask(rope_interleaved(q,cos,sin),
#                            rope_interleaved(k,cos,sin), v, 1/sqrt(Dh))
#     att_flat = reshape(att, [1,S,D])
#     mlp_gate = gate_up[:, :, :F] ; mlp_up = gate_up[:, :, F:2F]
#     mlp      = swiglu(mlp_gate, mlp_up)                        # [1,S,F]
#     out_in   = concat(axis=2, att_flat, mlp)                   # [1,S, D+F]  CHANNEL concat
#     out      = linear(out_in, W2)                              # W2 [D, D+F]
#     result   = residual_gate(x, gate, out)                     # x + gate*out
#
# KEY DIFFERENCES from the double block (all handled below):
#   (1) the att/mlp concat is on the CHANNEL axis (axis=2, sizes D and F), not
#       the sequence axis -> cat_backward(grad, size0=D, size1=F, axis=2).
#   (2) the qkv/gate_up split is a contiguous CHANNEL slice of `fused`.
#   (3) modulate uses the SAME modulate_backward -> layer_norm_backward chain
#       (LN weight=1 bias=0, discard LN d_g/d_b).
#   (4) gate_residual_backward needs the gated `y` = `out`; recompute
#       out = linear(out_in, W2) in the backward (cheap).
#   single_mod's shift/scale/gate are INPUTS; their grads are OUTPUTS but do NOT
#   backprop into the modulation MLP (exactly like the double block's ModVecs).
#
# Mojo 1.0.0b1: `def` not `fn`; Tensor move-only (return Movable structs, never
# store Tensor in a collection); no-bias linear = linear(x, w, Optional(None), ctx).

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.scratch_ring import ScratchRingAllocator


# TArc = the Copyable device carrier (ArcPointer[Tensor]); a copy is a refcount
# bump of the SAME device buffer (no D2D, no sync). Mirrors autograd.mojo:50.
comptime TArc = ArcPointer[Tensor]

# ── forward ops (GPU) ────────────────────────────────────────────────────────
from serenitymojo.ops.linear import (
    linear, linear_scratch, linear_rows, linear_rows_scratch,
    linear_two_inputs_scratch,
)
from serenitymojo.ops.norm import rms_norm, layer_norm
from serenitymojo.ops.activations import swiglu
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.attention import sdpa_nomask
# cuDNN flash SDPA (approved numerics change 2026-06-11, memory
# sdpa-flash-signoff): the PRODUCTION resident-scratch recompute+backward
# pair runs attention through cuDNN flash with F32<->bf16 boundary casts
# (Klein S=1536 is 128-aligned — zero-copy path). Old math path stays
# compiled in the non-flag branch + every other fwd/bwd variant (C13).
# Anchors MOVE by design — re-anchor on flip (gate: sdpa_flash_parity).
from serenitymojo.ops.attention_flash import (
    sdpa_flash_train_fwd_f32, sdpa_flash_backward_f32, SdpaFlashF32Fwd,
)

comptime KLEIN_SDPA_FLASH = True
from serenitymojo.ops.tensor_algebra import (
    reshape, reshape_owned, reshape_in_place, slice, concat, add, add_in_place_f32,
    mul, mul_scalar, zeros_device,
)
from serenitymojo.ops.tensor_algebra_scratch import (
    concat2_scratch, concat3_scratch, slice_scratch,
)

# ── backward arms (GPU; all pre-built + gated) ───────────────────────────────
from serenitymojo.ops.linalg_backward import (
    linear_backward, linear_backward_dx, linear_backward_dx_scratch,
    linear_backward_dx_split_scratch, linear_backward_dw, LinearGrads,
)
from serenitymojo.util.bf16_stochastic_rounding import sr_uniform
from serenitymojo.ops.norm_backward import (
    rms_norm_backward, rms_norm_backward_dx, RmsNormBackward,
    layer_norm_backward, layer_norm_backward_dx, LayerNormBackward,
)
from serenitymojo.ops.loss_swiglu_backward import swiglu_backward, SwigluGrads
from serenitymojo.ops.attention_backward import (
    sdpa_backward, sdpa_backward_scratch, SdpaGrads,
)
from serenitymojo.ops.elementwise_backward import modulate_backward, ModulateBackward
from serenitymojo.ops.rope_struct_backward import (
    gate_residual_backward, gate_residual_backward_dxdy, GateResidualGrads,
    rope_backward,
)
from serenitymojo.ops.shape_backward import (
    cat_backward, CatGrads2, slice_backward, reshape_backward,
)


# ── host helpers ─────────────────────────────────────────────────────────────
def _add_lists(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(a)):
        o.append(a[i] + b[i])
    return o^


def _ones(d: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(d):
        o.append(1.0)
    return o^


def _zeros(d: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(d):
        o.append(0.0)
    return o^


# F32 host-list -> device Tensor helper (boundary / weight upload only).
def _t(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.F32, ctx)


def _t_dtype(
    vals: List[Float32], var shape: List[Int], dtype: STDtype, ctx: DeviceContext
) raises -> Tensor:
    return Tensor.from_host(vals, shape^, dtype, ctx)


# ── single modulation vectors (each [D]) ─────────────────────────────────────
struct SingleModVecs(Copyable, Movable):
    var shift: List[Float32]
    var scale: List[Float32]
    var gate: List[Float32]

    def __init__(
        out self,
        var shift: List[Float32], var scale: List[Float32], var gate: List[Float32],
    ):
        self.shift = shift^
        self.scale = scale^
        self.gate = gate^


struct SingleModVecsDevice(Copyable, Movable):
    var shift: TArc
    var scale: TArc
    var gate: TArc

    def __init__(out self, var shift: TArc, var scale: TArc, var gate: TArc):
        self.shift = shift^
        self.scale = scale^
        self.gate = gate^


def single_modvecs_to_device(
    mv: SingleModVecs, D: Int, ctx: DeviceContext
) raises -> SingleModVecsDevice:
    return SingleModVecsDevice(
        TArc(_t(mv.shift.copy(), [D], ctx)),
        TArc(_t(mv.scale.copy(), [D], ctx)),
        TArc(_t(mv.gate.copy(), [D], ctx)),
    )


# ── trainable weights (A2: DEVICE-RESIDENT, uploaded ONCE) ────────────────────
#   w1: [3D+2F, D]   (fused qkv + gate_up projection; "linear1")
#   w2: [D, D+F]     (output projection; "linear2")
#   q_norm/k_norm: [Dh]  (per-head rms scale)
#
# A2 PERF (2026-05-31): the FROZEN base matrices are now device-resident `TArc`
# carriers uploaded EXACTLY ONCE at construction (load time), not host
# List[Float32] re-uploaded by `_t(w.field.copy(), ...)` on every op every step.
# `__init__` takes the host lists (loader + gates pass byte-identical data) + the
# dims/ctx to upload each at its real shape; use-sites pass `w.field[]` (a borrow
# of the SAME resident buffer) — no per-op from_host, no per-op sync.
struct SingleBlockWeights(Copyable, Movable):
    var w1: TArc        # [3D+2F, D]
    var w2: TArc        # [D, D+F]
    var w2_att: TArc    # [D, D] packed w2[:, :D]
    var w2_mlp: TArc    # [D, F] packed w2[:, D:]
    var q_norm: TArc    # [Dh]
    var k_norm: TArc    # [Dh]

    def __init__(
        out self,
        var w1: List[Float32], var w2: List[Float32],
        var q_norm: List[Float32], var k_norm: List[Float32],
        D: Int, F: Int, Dh: Int, ctx: DeviceContext,
        keep_w2: Bool = True,
    ) raises:
        var w2_att = List[Float32]()
        var w2_mlp = List[Float32]()
        for r in range(D):
            var base = r * (D + F)
            for c in range(D):
                w2_att.append(w2[base + c])
            for c in range(F):
                w2_mlp.append(w2[base + D + c])
        self.w1 = TArc(Tensor.from_host(w1^, [3 * D + 2 * F, D], STDtype.F32, ctx))
        if keep_w2:
            self.w2 = TArc(Tensor.from_host(w2^, [D, D + F], STDtype.F32, ctx))
        else:
            var dummy = List[Float32]()
            dummy.append(0.0)
            self.w2 = TArc(Tensor.from_host(dummy^, [1, 1], STDtype.F32, ctx))
        self.w2_att = TArc(Tensor.from_host(w2_att^, [D, D], STDtype.F32, ctx))
        self.w2_mlp = TArc(Tensor.from_host(w2_mlp^, [D, F], STDtype.F32, ctx))
        self.q_norm = TArc(Tensor.from_host(q_norm^, [Dh], STDtype.F32, ctx))
        self.k_norm = TArc(Tensor.from_host(k_norm^, [Dh], STDtype.F32, ctx))

    def __init__(
        out self,
        var w1: TArc, var w2: TArc, var q_norm: TArc, var k_norm: TArc,
        D: Int, F: Int, ctx: DeviceContext,
        keep_w2: Bool = True,
    ) raises:
        self.w1 = w1^
        if keep_w2:
            self.w2 = w2.copy()
        else:
            self.w2 = TArc(zeros_device([1, 1], w2[].dtype(), ctx))
        self.w2_att = TArc(slice(w2[], 1, 0, D, ctx))
        self.w2_mlp = TArc(slice(w2[], 1, D, F, ctx))
        self.q_norm = q_norm^
        self.k_norm = k_norm^


# ── saved activations (DEVICE-RESIDENT via TArc) ─────────────────────────────
# Each field is a refcount handle to a device Tensor. A copy is an Arc bump, not
# a D2D clone, which lets stack-level checkpoint carriers stay device-resident.
struct SingleBlockSaved(Copyable, Movable):
    var x: TArc        # [S,D]      block input
    var ln: TArc       # [S,D]      layer_norm(x)
    var norm: TArc     # [S,D]      modulate(ln, scale, shift)
    var q_pre: TArc    # [1,S,H,Dh] q before rms (post-qkv split)
    var k_pre: TArc    # [1,S,H,Dh]
    var q_rms: TArc    # [1,S,H,Dh] rms_norm(q_pre, q_norm)
    var k_rms: TArc    # [1,S,H,Dh]
    var v: TArc        # [1,S,H,Dh]
    var q_rope: TArc   # [1,S,H,Dh] rope(q_rms)
    var k_rope: TArc   # [1,S,H,Dh] rope(k_rms)
    var att_flat: TArc # [S,D]      reshape(sdpa(...))
    var mlp_gate: TArc # [S,F]      gate_up[:, :F]
    var mlp_up: TArc   # [S,F]      gate_up[:, F:2F]
    var mlp: TArc      # [S,F]      swiglu(mlp_gate, mlp_up)
    var out_in: TArc   # [S, D+F]   concat(axis=1, att_flat, mlp)
    # cos/sin are NOT saved (constant rope tables borrowed by the backward).
    # Flash-SDPA saved set (Optional: only the KLEIN_SDPA_FLASH recompute
    # path fills these; every other constructor site passes nothing):
    # bf16 q_rope/k_rope/v + bf16 O + F32 LSE stats — exactly what
    # sdpa_flash_backward_f32 consumes, no re-casting in backward.
    var flash_q: Optional[TArc]
    var flash_k: Optional[TArc]
    var flash_v: Optional[TArc]
    var flash_o: Optional[TArc]
    var flash_stats: Optional[TArc]

    def __init__(
        out self,
        var x: TArc, var ln: TArc, var norm: TArc,
        var q_pre: TArc, var k_pre: TArc,
        var q_rms: TArc, var k_rms: TArc, var v: TArc,
        var q_rope: TArc, var k_rope: TArc,
        var att_flat: TArc,
        var mlp_gate: TArc, var mlp_up: TArc, var mlp: TArc,
        var out_in: TArc,
        var flash_q: Optional[TArc] = None,
        var flash_k: Optional[TArc] = None,
        var flash_v: Optional[TArc] = None,
        var flash_o: Optional[TArc] = None,
        var flash_stats: Optional[TArc] = None,
    ):
        self.x = x^
        self.ln = ln^
        self.norm = norm^
        self.q_pre = q_pre^
        self.k_pre = k_pre^
        self.q_rms = q_rms^
        self.k_rms = k_rms^
        self.v = v^
        self.q_rope = q_rope^
        self.k_rope = k_rope^
        self.att_flat = att_flat^
        self.mlp_gate = mlp_gate^
        self.mlp_up = mlp_up^
        self.mlp = mlp^
        self.out_in = out_in^
        self.flash_q = flash_q^
        self.flash_k = flash_k^
        self.flash_v = flash_v^
        self.flash_o = flash_o^
        self.flash_stats = flash_stats^


struct SingleBlockForward(Movable):
    var out: List[Float32]   # [S, D]  (host: the block output, boundary readback)
    var saved: SingleBlockSaved

    def __init__(out self, var out: List[Float32], var saved: SingleBlockSaved):
        self.out = out^
        self.saved = saved^


struct SingleBlockDeviceForward(Copyable, Movable):
    var out: TArc            # [S, D]  device-resident block output
    var saved: SingleBlockSaved

    def __init__(out self, var out: TArc, var saved: SingleBlockSaved):
        self.out = out^
        self.saved = saved^


struct SingleBlockDeviceOutput(Copyable, Movable):
    var out: TArc            # [S, D]  device-resident block output

    def __init__(out self, var out: TArc):
        self.out = out^


# ── backward result: input grad + all trainable weight grads + mod-vec grads ─
struct SingleBlockGrads(Copyable, Movable):
    var d_x: List[Float32]
    var d_w1: List[Float32]
    var d_w2: List[Float32]
    var d_q_norm: List[Float32]
    var d_k_norm: List[Float32]
    # modulation-vector grads (block outputs; not backproped into mod MLP)
    var d_shift: List[Float32]
    var d_scale: List[Float32]
    var d_gate: List[Float32]

    def __init__(
        out self,
        var d_x: List[Float32], var d_w1: List[Float32], var d_w2: List[Float32],
        var d_q_norm: List[Float32], var d_k_norm: List[Float32],
        var d_shift: List[Float32], var d_scale: List[Float32], var d_gate: List[Float32],
    ):
        self.d_x = d_x^
        self.d_w1 = d_w1^
        self.d_w2 = d_w2^
        self.d_q_norm = d_q_norm^
        self.d_k_norm = d_k_norm^
        self.d_shift = d_shift^
        self.d_scale = d_scale^
        self.d_gate = d_gate^


# ── FORWARD of one SINGLE block ──────────────────────────────────────────────
# cos/sin: precomputed rope tables for the sequence, [S*H, Dh/2], resident.
def single_block_forward[
    H: Int, Dh: Int, S: Int
](
    x: List[Float32],
    w: SingleBlockWeights, mv: SingleModVecs,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> SingleBlockForward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    # resident layer_norm ones[D]/zeros[D] + qkv weights (uploaded once).
    var ones_t = _t(_ones(D), [D], ctx)
    var zeros_t = _t(_zeros(D), [D], ctx)

    # x ENTERS host -> ONE from_host. Everything below stays on-device.
    # The GPU ops BORROW their Tensor args, so an activation can feed a downstream
    # op AND still be moved into `SingleBlockSaved` afterwards — no per-op
    # .to_host()/from_host() round-trip (that bounce was the whole cost). The only
    # host transfer is the single `x` in and the single `out` readback.
    var x_t = _t(x, [S, D], ctx)

    # x_norm = modulate(layer_norm(x), scale, shift)
    var ln_t = layer_norm(x_t, ones_t, zeros_t, eps, ctx)
    var norm_t = modulate(ln_t, _t(mv.scale.copy(), [D], ctx), _t(mv.shift.copy(), [D], ctx), ctx)

    # fused = linear(x_norm, W1) ; [S, 3D+2F]
    var no_bias = Optional[Tensor](None)
    var fused = linear(norm_t, w.w1[], no_bias^, ctx)

    # channel split: qkv [S,3D] | gate_up [S,2F]  (device slice on dim 1)
    var qkv = slice(fused, 1, 0, 3 * D, ctx)
    var gate_up = slice(fused, 1, 3 * D, 2 * F, ctx)

    # q,k,v: each [S,D] (== [1,S,H,Dh] byte-identical)
    var q_pre_flat = slice(qkv, 1, 0, D, ctx)
    var k_pre_flat = slice(qkv, 1, D, D, ctx)
    var v_flat = slice(qkv, 1, 2 * D, D, ctx)
    # reshape [S,D] -> [1,S,H,Dh] is a row-major byte no-op.
    var q_pre = reshape_owned(q_pre_flat^, [1, S, H, Dh])
    var k_pre = reshape_owned(k_pre_flat^, [1, S, H, Dh])
    var v = reshape_owned(v_flat^, [1, S, H, Dh])

    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)

    # rope then sdpa (cos/sin borrowed — resident)
    var q_rope = rope_interleaved(q_rms, cos, sin, ctx)
    var k_rope = rope_interleaved(k_rms, cos, sin, ctx)
    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
    # reshape [1,S,H,Dh] -> [S,D] is a byte no-op.
    var att_flat = reshape_owned(att^, [S, D])

    # mlp branch
    var mlp_gate = slice(gate_up, 1, 0, F, ctx)
    var mlp_up = slice(gate_up, 1, F, F, ctx)
    var mlp = swiglu(mlp_gate, mlp_up, ctx)

    # concat on CHANNEL axis (dim 1): out_in [S, D+F]
    var out_in = concat(1, ctx, att_flat, mlp)

    # out = linear(out_in, W2) ; W2 [D, D+F]
    var no_bias2 = Optional[Tensor](None)
    var out_proj = linear(out_in, w.w2[], no_bias2^, ctx)

    # result = residual_gate(x, gate, out) ; ONE boundary readback (inline).
    var result = residual_gate(
        x_t, _t(mv.gate.copy(), [D], ctx), out_proj, ctx
    ).to_host(ctx)

    var saved = SingleBlockSaved(
        TArc(x_t^), TArc(ln_t^), TArc(norm_t^), TArc(q_pre^), TArc(k_pre^),
        TArc(q_rms^), TArc(k_rms^), TArc(v^),
        TArc(q_rope^), TArc(k_rope^), TArc(att_flat^),
        TArc(mlp_gate^), TArc(mlp_up^), TArc(mlp^), TArc(out_in^),
    )
    return SingleBlockForward(result^, saved^)


# ── BACKWARD of one SINGLE block (hand-chained) ──────────────────────────────
# d_out: upstream grad of the block output [S,D] (host list in; from_host ONCE).
def single_block_backward[
    H: Int, Dh: Int, S: Int
](
    d_out: List[Float32],
    w: SingleBlockWeights, mv: SingleModVecs, saved: SingleBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> SingleBlockGrads:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t(_ones(D), [D], ctx)
    var scale_t = _t(mv.scale.copy(), [D], ctx)
    var gate_t = _t(mv.gate.copy(), [D], ctx)

    var d_out_t = _t(d_out, [S, D], ctx)

    # result = residual_gate(x, gate, out): o = x + gate*out
    # `out` (the gated `y`) is recomputed = linear(out_in, W2).
    var nb = Optional[Tensor](None)
    var out_y = linear(saved.out_in[], w.w2[], nb^, ctx)
    var grg = gate_residual_backward(d_out_t, saved.x[], gate_t, out_y, ctx)
    # d_x_res (residual branch) and d_out_proj kept device-resident.
    var d_gate = grg.d_g.to_host(ctx)

    # out = linear(out_in, W2)
    var lb_w2 = linear_backward(
        grg.d_y, saved.out_in[], w.w2[], S, D + F, D, ctx,
    )
    # d_out_in = lb_w2.d_x [S, D+F] ; d_w2 = lb_w2.d_w
    var d_w2 = lb_w2.d_w.to_host(ctx)

    # out_in = concat(axis=2, att_flat, mlp) on the CHANNEL axis (sizes D, F).
    # d_out_in is [S, D+F]; reshape to [1,S,D+F] for cat_backward on axis=2.
    reshape_in_place(lb_w2.d_x, [1, S, D + F])
    var cb = cat_backward(lb_w2.d_x, D, F, 2, ctx)
    reshape_in_place(cb.d_0, [1, S, H, Dh])   # [1,S,D] == [1,S,H,Dh]
    reshape_in_place(cb.d_1, [S, F])          # [1,S,F] == [S,F]

    # mlp = swiglu(mlp_gate, mlp_up)
    var sgb = swiglu_backward(cb.d_1, saved.mlp_gate[], saved.mlp_up[], ctx)
    # join gate/up grads back into gate_up [S,2F] (device concat on dim 1)
    var d_gate_up = concat(1, ctx, sgb.d_gate, sgb.d_up)

    # att branch: d_att_flat [1,S,H,Dh] -> sdpa backward.
    var sb = sdpa_backward[1, S, H, Dh](
        saved.q_rope[], saved.k_rope[], saved.v[], cb.d_0, scale, ctx,
    )
    # d_q_rope/d_k_rope/d_v device-resident in sb.

    # rope backward (cos/sin non-learnable -> only d_x); cos/sin borrowed resident
    var d_q_rms = rope_backward(sb.d_q, cos, sin, True, ctx)
    var d_k_rms = rope_backward(sb.d_k, cos, sin, True, ctx)

    # rms_norm backward for q and k
    var rb_q = rms_norm_backward(d_q_rms, saved.q_pre[], w.q_norm[], eps, ctx)
    var d_q_norm = rb_q.d_g.to_host(ctx)
    var rb_k = rms_norm_backward(d_k_rms, saved.k_pre[], w.k_norm[], eps, ctx)
    var d_k_norm = rb_k.d_g.to_host(ctx)

    # join d_q_pre|d_k_pre|d_v into d_qkv [S,3D] (reshape each [1,S,H,Dh]->[S,D],
    # then device concat on dim 1).
    reshape_in_place(rb_q.d_x, [S, D])
    reshape_in_place(rb_k.d_x, [S, D])
    reshape_in_place(sb.d_v, [S, D])
    var d_qkv = concat(1, ctx, rb_q.d_x, rb_k.d_x, sb.d_v)

    # join the qkv grad and gate_up grad back into d_fused [S, 3D+2F]
    var d_fused = concat(1, ctx, d_qkv, d_gate_up)

    # fused = linear(norm, W1)
    var lb_w1 = linear_backward(
        d_fused, saved.norm[], w.w1[], S, D, 3 * D + 2 * F, ctx,
    )
    var d_w1 = lb_w1.d_w.to_host(ctx)
    # d_norm = lb_w1.d_x

    # norm = modulate(ln, scale, shift)
    var mb = modulate_backward(lb_w1.d_x, saved.ln[], scale_t, ctx)
    var d_scale = mb.d_scale.to_host(ctx)
    var d_shift = mb.d_shift.to_host(ctx)

    # ln = layer_norm(x, 1, 0)
    var lnb = layer_norm_backward(mb.d_x, saved.x[], ones_t, eps, ctx)

    # x feeds BOTH the residual (grg.d_x) AND layer_norm(x) -> SUM.
    # gate_residual_backward gives d_x = grad_out (passthrough); sum on host at
    # the boundary readback (both are [S,D] device grads).
    var d_x_res = grg.d_x.to_host(ctx)
    var d_x_norm = lnb.d_x.to_host(ctx)
    var d_x = _add_lists(d_x_res, d_x_norm)

    return SingleBlockGrads(
        d_x^, d_w1^, d_w2^, d_q_norm^, d_k_norm^,
        d_shift^, d_scale^, d_gate^,
    )


# ═══════════════════════════════════════════════════════════════════════════
# LoRA-ON-PROJECTION VARIANT
#
# Targets (matches OneTrainer Flux2 single_blocks):
#   linear1 (w1) to_qkv_mlp slot: the LoRA delta covers the full OneTrainer
#     fused projection [S, 3D + 2F], including Q/K/V and gate/up rows.
#   linear2 (w2) to_out slot: the LoRA input is the full [att_flat, mlp]
#     concatenation [S, D + F], matching OneTrainer's single block target.
#
# When both adapters are absent this REDUCES to the verified base single block.
#
# NOTE: the LoRA-delta helpers have resident device variants. The host-list
# helpers remain for compatibility/parity, while the hot trainer path passes
# `SingleBlockLoraDevice` and avoids per-use A/B uploads.
# ═══════════════════════════════════════════════════════════════════════════

from serenitymojo.models.klein.lora_block import (
    LoraAdapter, LoraAdapterDevice, lora_adapter_to_device,
    klein_lora_fwd_device, klein_lora_bwd_device,
    klein_lora_fwd_device_resident,
    klein_lora_bwd_device_resident, klein_lora_bwd_device_resident_tensors,
    KleinLoraDeviceGrads, KleinLoraDeviceGradTensors,
    klein_take_cols_device, klein_add_cols_device,
)
from serenitymojo.models.klein.klein_direct_lycoris_stack import (
    KleinSingleDirectDoRA, KleinSingleDirectOFT,
    KleinDirectDoRAGradT, KleinDirectOFTGradT,
    klein_direct_dora_projection_forward_optional,
    klein_direct_dora_projection_backward_optional,
    klein_direct_oft_projection_forward_optional,
    klein_direct_oft_projection_backward_optional,
)


struct LoraDropout(ImplicitlyCopyable, Movable):
    var p: Float32
    var seed: UInt32
    var slot: UInt32

    def __init__(out self, p: Float32 = 0.0, seed: UInt32 = 0, slot: UInt32 = 0):
        self.p = p
        self.seed = seed
        self.slot = slot


def _lora_dropout_mask(
    drop: LoraDropout, M: Int, rank: Int, dt: STDtype, ctx: DeviceContext
) raises -> Tensor:
    var n = M * rank
    var inv_keep = Float32(1.0) / (Float32(1.0) - drop.p)
    var vals = List[Float32]()
    var slot_seed = drop.seed ^ (drop.slot * UInt32(2654435761))
    for i in range(n):
        var u = sr_uniform(slot_seed, i)
        if u < drop.p:
            vals.append(Float32(0.0))
        else:
            vals.append(inv_keep)
    return Tensor.from_host(vals^, [M, rank], dt, ctx)


def _klein_lora_fwd_dropout(
    x: Tensor, lo: LoraAdapterDevice, M: Int, drop: LoraDropout, ctx: DeviceContext
) raises -> Tensor:
    var nb1 = Optional[Tensor](None)
    var t = linear(x, lo.a[], nb1^, ctx)
    if drop.p > Float32(0.0):
        var mask = _lora_dropout_mask(drop, M, lo.rank, t.dtype(), ctx)
        t = mul(t, mask, ctx)
    var nb2 = Optional[Tensor](None)
    var dy = linear(t, lo.b[], nb2^, ctx)
    return mul_scalar(dy, lo.scale, ctx)


def _klein_lora_bwd_dropout(
    d_contrib: Tensor, x: Tensor, lo: LoraAdapterDevice,
    M: Int, drop: LoraDropout, ctx: DeviceContext,
) raises -> KleinLoraDeviceGrads:
    if drop.p <= Float32(0.0):
        return klein_lora_bwd_device_resident(d_contrib, x, lo, M, ctx)

    var nb_t = Optional[Tensor](None)
    var t = linear(x, lo.a[], nb_t^, ctx)
    var t_drop = mul(t, _lora_dropout_mask(drop, M, lo.rank, t.dtype(), ctx), ctx)
    var d_dy = mul_scalar(d_contrib, lo.scale, ctx)
    var d_t_drop = linear_backward_dx(d_dy, lo.b[], M, lo.rank, lo.out_f, ctx)
    var d_b_t = linear_backward_dw(
        d_dy, t_drop, M, lo.rank, lo.out_f, ctx, output_dtype=STDtype.F32
    )
    var d_t = mul(
        d_t_drop, _lora_dropout_mask(drop, M, lo.rank, d_t_drop.dtype(), ctx), ctx
    )
    var d_x_lo = linear_backward_dx(d_t, lo.a[], M, lo.in_f, lo.rank, ctx)
    var d_a_t = linear_backward_dw(
        d_t, x, M, lo.in_f, lo.rank, ctx, output_dtype=STDtype.F32
    )
    var d_a = d_a_t.to_host(ctx)
    var d_b = d_b_t.to_host(ctx)
    return KleinLoraDeviceGrads(d_a^, d_b^, d_x_lo^)


def _klein_lora_bwd_dropout_tensors(
    d_contrib: Tensor, x: Tensor, lo: LoraAdapterDevice,
    M: Int, drop: LoraDropout, ctx: DeviceContext,
) raises -> KleinLoraDeviceGradTensors:
    if drop.p <= Float32(0.0):
        return klein_lora_bwd_device_resident_tensors(d_contrib, x, lo, M, ctx)

    var nb_t = Optional[Tensor](None)
    var t = linear(x, lo.a[], nb_t^, ctx)
    var t_drop = mul(t, _lora_dropout_mask(drop, M, lo.rank, t.dtype(), ctx), ctx)
    var d_dy = mul_scalar(d_contrib, lo.scale, ctx)
    var d_t_drop = linear_backward_dx(d_dy, lo.b[], M, lo.rank, lo.out_f, ctx)
    var d_b_t = linear_backward_dw(
        d_dy, t_drop, M, lo.rank, lo.out_f, ctx, output_dtype=STDtype.F32
    )
    var d_t = mul(
        d_t_drop, _lora_dropout_mask(drop, M, lo.rank, d_t_drop.dtype(), ctx), ctx
    )
    var d_x_lo = linear_backward_dx(d_t, lo.a[], M, lo.in_f, lo.rank, ctx)
    var d_a_t = linear_backward_dw(
        d_t, x, M, lo.in_f, lo.rank, ctx, output_dtype=STDtype.F32
    )
    return KleinLoraDeviceGradTensors(TArc(d_a_t^), TArc(d_b_t^), TArc(d_x_lo^))


struct SingleBlockLora(Copyable, Movable):
    var qkv: Optional[LoraAdapter]    # to_qkv_mlp_proj (in=D, out=3D+2F)
    var out: Optional[LoraAdapter]    # to_out          (in=D+F, out=D)

    def __init__(
        out self, var qkv: Optional[LoraAdapter], var out: Optional[LoraAdapter]
    ):
        self.qkv = qkv^
        self.out = out^


struct SingleBlockLoraDevice(Copyable, Movable):
    var qkv: Optional[LoraAdapterDevice]    # to_qkv_mlp_proj (in=D, out=3D+2F)
    var out: Optional[LoraAdapterDevice]    # to_out          (in=D+F, out=D)

    def __init__(
        out self,
        var qkv: Optional[LoraAdapterDevice], var out: Optional[LoraAdapterDevice],
    ):
        self.qkv = qkv^
        self.out = out^


def _optional_lora_to_device(
    lo: Optional[LoraAdapter], ctx: DeviceContext
) raises -> Optional[LoraAdapterDevice]:
    if lo:
        return Optional[LoraAdapterDevice](lora_adapter_to_device(lo.value(), ctx))
    return Optional[LoraAdapterDevice](None)


def single_block_lora_to_device(
    lora: SingleBlockLora, ctx: DeviceContext
) raises -> SingleBlockLoraDevice:
    return SingleBlockLoraDevice(
        _optional_lora_to_device(lora.qkv, ctx),
        _optional_lora_to_device(lora.out, ctx),
    )


struct SingleBlockLoraGrads(Copyable, Movable):
    var base: SingleBlockGrads
    var qkv_d_a: List[Float32]
    var qkv_d_b: List[Float32]
    var out_d_a: List[Float32]
    var out_d_b: List[Float32]

    def __init__(
        out self, var base: SingleBlockGrads,
        var qkv_d_a: List[Float32], var qkv_d_b: List[Float32],
        var out_d_a: List[Float32], var out_d_b: List[Float32],
    ):
        self.base = base^
        self.qkv_d_a = qkv_d_a^
        self.qkv_d_b = qkv_d_b^
        self.out_d_a = out_d_a^
        self.out_d_b = out_d_b^


struct SingleBlockLoraDeviceGrads(Copyable, Movable):
    var d_x: TArc
    var d_shift: List[Float32]
    var d_scale: List[Float32]
    var d_gate: List[Float32]
    var qkv_d_a: List[Float32]
    var qkv_d_b: List[Float32]
    var out_d_a: List[Float32]
    var out_d_b: List[Float32]

    def __init__(
        out self,
        var d_x: TArc,
        var d_shift: List[Float32], var d_scale: List[Float32], var d_gate: List[Float32],
        var qkv_d_a: List[Float32], var qkv_d_b: List[Float32],
        var out_d_a: List[Float32], var out_d_b: List[Float32],
    ):
        self.d_x = d_x^
        self.d_shift = d_shift^
        self.d_scale = d_scale^
        self.d_gate = d_gate^
        self.qkv_d_a = qkv_d_a^
        self.qkv_d_b = qkv_d_b^
        self.out_d_a = out_d_a^
        self.out_d_b = out_d_b^


struct SingleBlockLoraDeviceGradTensors(Copyable, Movable):
    var d_x: TArc
    var d_shift: List[Float32]
    var d_scale: List[Float32]
    var d_gate: List[Float32]
    var qkv_d_a: Optional[TArc]
    var qkv_d_b: Optional[TArc]
    var out_d_a: Optional[TArc]
    var out_d_b: Optional[TArc]

    def __init__(
        out self,
        var d_x: TArc,
        var d_shift: List[Float32], var d_scale: List[Float32], var d_gate: List[Float32],
        var qkv_d_a: Optional[TArc], var qkv_d_b: Optional[TArc],
        var out_d_a: Optional[TArc], var out_d_b: Optional[TArc],
    ):
        self.d_x = d_x^
        self.d_shift = d_shift^
        self.d_scale = d_scale^
        self.d_gate = d_gate^
        self.qkv_d_a = qkv_d_a^
        self.qkv_d_b = qkv_d_b^
        self.out_d_a = out_d_a^
        self.out_d_b = out_d_b^


# ── FORWARD of one SINGLE block WITH LoRA on full linear1 + full linear2 ─────
def single_block_lora_forward_device_resident[
    H: Int, Dh: Int, S: Int
](
    x_t: TArc,
    w: SingleBlockWeights, mv: SingleModVecsDevice, lora: SingleBlockLoraDevice,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
    drop_qkv: LoraDropout = LoraDropout(),
    drop_out: LoraDropout = LoraDropout(),
) raises -> SingleBlockDeviceForward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var norm_dtype = x_t[].dtype()
    var ones_t = _t_dtype(_ones(D), [D], norm_dtype, ctx)
    var zeros_t = _t_dtype(_zeros(D), [D], norm_dtype, ctx)

    var ln_t = layer_norm(x_t[], ones_t, zeros_t, eps, ctx)
    var norm_t = modulate(ln_t, mv.scale[], mv.shift[], ctx)

    var no_bias = Optional[Tensor](None)
    var fused = linear(norm_t, w.w1[], no_bias^, ctx)   # [S, 3D+2F]
    # LoRA on to_qkv_mlp_proj: FULL delta [S,3D+2F] added to all fused cols.
    if lora.qkv:
        var dlt = _klein_lora_fwd_dropout(
            norm_t, lora.qkv.value(), S, drop_qkv, ctx
        )
        fused = add(fused, dlt, ctx)

    var qkv = slice(fused, 1, 0, 3 * D, ctx)
    var gate_up = slice(fused, 1, 3 * D, 2 * F, ctx)

    var q_pre_flat = slice(qkv, 1, 0, D, ctx)
    var k_pre_flat = slice(qkv, 1, D, D, ctx)
    var v_flat = slice(qkv, 1, 2 * D, D, ctx)
    var q_pre = reshape_owned(q_pre_flat^, [1, S, H, Dh])
    var k_pre = reshape_owned(k_pre_flat^, [1, S, H, Dh])
    var v = reshape_owned(v_flat^, [1, S, H, Dh])

    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)

    var q_rope = rope_interleaved(q_rms, cos, sin, ctx)
    var k_rope = rope_interleaved(k_rms, cos, sin, ctx)
    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
    var att_flat = reshape_owned(att^, [S, D])

    var mlp_gate = slice(gate_up, 1, 0, F, ctx)
    var mlp_up = slice(gate_up, 1, F, F, ctx)
    var mlp = swiglu(mlp_gate, mlp_up, ctx)

    var out_in = concat(1, ctx, att_flat, mlp)

    var no_bias2 = Optional[Tensor](None)
    var out_proj = linear(out_in, w.w2[], no_bias2^, ctx)
    # LoRA on to_out: input is the FULL out_in [S,D+F], not attn-only.
    if lora.out:
        var dlt2 = _klein_lora_fwd_dropout(
            out_in, lora.out.value(), S, drop_out, ctx
        )
        out_proj = add(out_proj, dlt2, ctx)

    var result = residual_gate(
        x_t[], mv.gate[], out_proj, ctx
    )

    var saved = SingleBlockSaved(
        x_t.copy(), TArc(ln_t^), TArc(norm_t^), TArc(q_pre^), TArc(k_pre^),
        TArc(q_rms^), TArc(k_rms^), TArc(v^),
        TArc(q_rope^), TArc(k_rope^), TArc(att_flat^),
        TArc(mlp_gate^), TArc(mlp_up^), TArc(mlp^), TArc(out_in^),
    )
    return SingleBlockDeviceForward(TArc(result^), saved^)


def single_block_lora_forward_device_resident_scratch[
    H: Int, Dh: Int, S: Int
](
    x_t: TArc,
    w: SingleBlockWeights, mv: SingleModVecsDevice, lora: SingleBlockLoraDevice,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    norm_ones: Tensor, norm_zeros: Tensor,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
    drop_qkv: LoraDropout = LoraDropout(),
    drop_out: LoraDropout = LoraDropout(),
) raises -> SingleBlockDeviceForward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var ln_t = layer_norm(x_t[], norm_ones, norm_zeros, eps, ctx)
    var norm_t = modulate(ln_t, mv.scale[], mv.shift[], ctx)

    var scratch_mark = scratch.mark()
    var q_pre_flat = linear_rows(norm_t, w.w1[], 0, D, ctx)
    var k_pre_flat = linear_rows(norm_t, w.w1[], D, D, ctx)
    var v_flat = linear_rows(norm_t, w.w1[], 2 * D, D, ctx)
    var gate_up = linear_rows_scratch(norm_t, w.w1[], 3 * D, 2 * F, ctx, scratch)
    if lora.qkv:
        var dlt = _klein_lora_fwd_dropout(
            norm_t, lora.qkv.value(), S, drop_qkv, ctx
        )
        add_in_place_f32(q_pre_flat, slice(dlt, 1, 0, D, ctx), ctx)
        add_in_place_f32(k_pre_flat, slice(dlt, 1, D, D, ctx), ctx)
        add_in_place_f32(v_flat, slice(dlt, 1, 2 * D, D, ctx), ctx)
        add_in_place_f32(gate_up, slice(dlt, 1, 3 * D, 2 * F, ctx), ctx)
    var q_pre = reshape_owned(q_pre_flat^, [1, S, H, Dh])
    var k_pre = reshape_owned(k_pre_flat^, [1, S, H, Dh])
    var v = reshape_owned(v_flat^, [1, S, H, Dh])

    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)

    var q_rope = rope_interleaved(q_rms, cos, sin, ctx)
    var k_rope = rope_interleaved(k_rms, cos, sin, ctx)
    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
    var att_flat = reshape_owned(att^, [S, D])

    var mlp_gate = slice(gate_up, 1, 0, F, ctx)
    var mlp_up = slice(gate_up, 1, F, F, ctx)
    var mlp = swiglu(mlp_gate, mlp_up, ctx)
    scratch.rewind(scratch_mark)

    var out_in_t = concat(1, ctx, att_flat, mlp)
    var proj_mark = scratch.mark()
    var out_proj = linear_two_inputs_scratch(
        att_flat, mlp, w.w2_att[], w.w2_mlp[], ctx, scratch,
    )
    if lora.out:
        var dlt2 = _klein_lora_fwd_dropout(
            out_in_t, lora.out.value(), S, drop_out, ctx
        )
        add_in_place_f32(out_proj, dlt2, ctx)

    var result = residual_gate(
        x_t[], mv.gate[], out_proj, ctx
    )
    scratch.rewind(proj_mark)

    var saved = SingleBlockSaved(
        x_t.copy(), TArc(ln_t^), TArc(norm_t^), TArc(q_pre^), TArc(k_pre^),
        TArc(q_rms^), TArc(k_rms^), TArc(v^),
        TArc(q_rope^), TArc(k_rope^), TArc(att_flat^),
        TArc(mlp_gate^), TArc(mlp_up^), TArc(mlp^), TArc(out_in_t^),
    )
    return SingleBlockDeviceForward(TArc(result^), saved^)


def single_block_lora_predict_device_resident_scratch[
    H: Int, Dh: Int, S: Int
](
    x_t: TArc,
    w: SingleBlockWeights, mv: SingleModVecsDevice, lora: SingleBlockLoraDevice,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    norm_ones: Tensor, norm_zeros: Tensor,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> SingleBlockDeviceOutput:
    """Inference-only LoRA single block: same math, no backward tape."""
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var ln_t = layer_norm(x_t[], norm_ones, norm_zeros, eps, ctx)
    var norm_t = modulate(ln_t, mv.scale[], mv.shift[], ctx)

    var scratch_mark = scratch.mark()
    var q_pre_flat = linear_rows(norm_t, w.w1[], 0, D, ctx)
    var k_pre_flat = linear_rows(norm_t, w.w1[], D, D, ctx)
    var v_flat = linear_rows(norm_t, w.w1[], 2 * D, D, ctx)
    var gate_up = linear_rows_scratch(norm_t, w.w1[], 3 * D, 2 * F, ctx, scratch)
    if lora.qkv:
        var dlt = klein_lora_fwd_device_resident(norm_t, lora.qkv.value(), S, ctx)
        add_in_place_f32(q_pre_flat, slice(dlt, 1, 0, D, ctx), ctx)
        add_in_place_f32(k_pre_flat, slice(dlt, 1, D, D, ctx), ctx)
        add_in_place_f32(v_flat, slice(dlt, 1, 2 * D, D, ctx), ctx)
        add_in_place_f32(gate_up, slice(dlt, 1, 3 * D, 2 * F, ctx), ctx)
    var q_pre = reshape_owned(q_pre_flat^, [1, S, H, Dh])
    var k_pre = reshape_owned(k_pre_flat^, [1, S, H, Dh])
    var v = reshape_owned(v_flat^, [1, S, H, Dh])

    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)

    var q_rope = rope_interleaved(q_rms, cos, sin, ctx)
    var k_rope = rope_interleaved(k_rms, cos, sin, ctx)
    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
    var att_flat = reshape_owned(att^, [S, D])

    var mlp_gate = slice(gate_up, 1, 0, F, ctx)
    var mlp_up = slice(gate_up, 1, F, F, ctx)
    var mlp = swiglu(mlp_gate, mlp_up, ctx)
    scratch.rewind(scratch_mark)

    var proj_mark = scratch.mark()
    var out_proj = linear_two_inputs_scratch(
        att_flat, mlp, w.w2_att[], w.w2_mlp[], ctx, scratch,
    )
    if lora.out:
        var out_in_t = concat(1, ctx, att_flat, mlp)
        var dlt2 = klein_lora_fwd_device_resident(out_in_t, lora.out.value(), S, ctx)
        add_in_place_f32(out_proj, dlt2, ctx)

    var result = residual_gate(
        x_t[], mv.gate[], out_proj, ctx
    )
    scratch.rewind(proj_mark)
    return SingleBlockDeviceOutput(TArc(result^))


def single_block_lora_recompute_saved_device_resident[
    H: Int, Dh: Int, S: Int
](
    x_t: TArc,
    w: SingleBlockWeights, mv: SingleModVecsDevice, lora: SingleBlockLoraDevice,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> SingleBlockSaved:
    """Recompute only the activations needed by backward checkpointing.

    The block output is discarded by stack backward for unsaved single blocks,
    and no-aux LoRA backward no longer needs the gated output value. Stop at
    `out_in` to avoid the final W2/LoRA-out/residual output work.
    """
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var norm_dtype = x_t[].dtype()
    var ones_t = _t_dtype(_ones(D), [D], norm_dtype, ctx)
    var zeros_t = _t_dtype(_zeros(D), [D], norm_dtype, ctx)

    var ln_t = layer_norm(x_t[], ones_t, zeros_t, eps, ctx)
    var norm_t = modulate(ln_t, mv.scale[], mv.shift[], ctx)

    var no_bias = Optional[Tensor](None)
    var fused = linear(norm_t, w.w1[], no_bias^, ctx)
    if lora.qkv:
        var dlt = klein_lora_fwd_device_resident(norm_t, lora.qkv.value(), S, ctx)
        fused = add(fused, dlt, ctx)

    var qkv = slice(fused, 1, 0, 3 * D, ctx)
    var gate_up = slice(fused, 1, 3 * D, 2 * F, ctx)

    var q_pre_flat = slice(qkv, 1, 0, D, ctx)
    var k_pre_flat = slice(qkv, 1, D, D, ctx)
    var v_flat = slice(qkv, 1, 2 * D, D, ctx)
    var q_pre = reshape_owned(q_pre_flat^, [1, S, H, Dh])
    var k_pre = reshape_owned(k_pre_flat^, [1, S, H, Dh])
    var v = reshape_owned(v_flat^, [1, S, H, Dh])

    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)

    var q_rope = rope_interleaved(q_rms, cos, sin, ctx)
    var k_rope = rope_interleaved(k_rms, cos, sin, ctx)
    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
    var att_flat = reshape_owned(att^, [S, D])

    var mlp_gate = slice(gate_up, 1, 0, F, ctx)
    var mlp_up = slice(gate_up, 1, F, F, ctx)
    var mlp = swiglu(mlp_gate, mlp_up, ctx)

    var out_in = concat(1, ctx, att_flat, mlp)

    return SingleBlockSaved(
        x_t.copy(), TArc(ln_t^), TArc(norm_t^), TArc(q_pre^), TArc(k_pre^),
        TArc(q_rms^), TArc(k_rms^), TArc(v^),
        TArc(q_rope^), TArc(k_rope^), TArc(att_flat^),
        TArc(mlp_gate^), TArc(mlp_up^), TArc(mlp^), TArc(out_in^),
    )


def single_block_lora_recompute_saved_device_resident_scratch[
    H: Int, Dh: Int, S: Int
](
    x_t: TArc,
    w: SingleBlockWeights, mv: SingleModVecsDevice, lora: SingleBlockLoraDevice,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    norm_ones: Tensor, norm_zeros: Tensor,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> SingleBlockSaved:
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var ln_t = layer_norm(x_t[], norm_ones, norm_zeros, eps, ctx)
    var norm_t = modulate(ln_t, mv.scale[], mv.shift[], ctx)

    var scratch_mark = scratch.mark()
    var q_pre_flat = linear_rows(norm_t, w.w1[], 0, D, ctx)
    var k_pre_flat = linear_rows(norm_t, w.w1[], D, D, ctx)
    var v_flat = linear_rows(norm_t, w.w1[], 2 * D, D, ctx)
    var gate_up = linear_rows_scratch(norm_t, w.w1[], 3 * D, 2 * F, ctx, scratch)
    if lora.qkv:
        var dlt = klein_lora_fwd_device_resident(norm_t, lora.qkv.value(), S, ctx)
        add_in_place_f32(q_pre_flat, slice(dlt, 1, 0, D, ctx), ctx)
        add_in_place_f32(k_pre_flat, slice(dlt, 1, D, D, ctx), ctx)
        add_in_place_f32(v_flat, slice(dlt, 1, 2 * D, D, ctx), ctx)
        add_in_place_f32(gate_up, slice(dlt, 1, 3 * D, 2 * F, ctx), ctx)
    var q_pre = reshape_owned(q_pre_flat^, [1, S, H, Dh])
    var k_pre = reshape_owned(k_pre_flat^, [1, S, H, Dh])
    var v = reshape_owned(v_flat^, [1, S, H, Dh])

    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)

    var q_rope = rope_interleaved(q_rms, cos, sin, ctx)
    var k_rope = rope_interleaved(k_rms, cos, sin, ctx)
    comptime if KLEIN_SDPA_FLASH:
        # cuDNN flash with F32<->bf16 boundary casts; bf16 q/k/v/o + stats
        # go to the tape for the flash backward (no recompute, no re-cast).
        var ff = sdpa_flash_train_fwd_f32[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
        # zero-copy re-box [1,S,H,Dh] -> [S,D] (no partial move out of ff)
        var af_shape: List[Int] = [S, D]
        var att_flat = Tensor(ff.att.buf.copy(), af_shape^, STDtype.F32)

        var mlp_gate = slice(gate_up, 1, 0, F, ctx)
        var mlp_up = slice(gate_up, 1, F, F, ctx)
        var mlp = swiglu(mlp_gate, mlp_up, ctx)
        scratch.rewind(scratch_mark)

        var out_in = concat(1, ctx, att_flat, mlp)
        return SingleBlockSaved(
            x_t.copy(), TArc(ln_t^), TArc(norm_t^), TArc(q_pre^), TArc(k_pre^),
            TArc(q_rms^), TArc(k_rms^), TArc(v^),
            TArc(q_rope^), TArc(k_rope^), TArc(att_flat^),
            TArc(mlp_gate^), TArc(mlp_up^), TArc(mlp^), TArc(out_in^),
            Optional[TArc](ff.q_bf.copy()), Optional[TArc](ff.k_bf.copy()),
            Optional[TArc](ff.v_bf.copy()), Optional[TArc](ff.o_bf.copy()),
            Optional[TArc](ff.stats.copy()),
        )
    else:
        var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
        var att_flat = reshape_owned(att^, [S, D])

        var mlp_gate = slice(gate_up, 1, 0, F, ctx)
        var mlp_up = slice(gate_up, 1, F, F, ctx)
        var mlp = swiglu(mlp_gate, mlp_up, ctx)
        scratch.rewind(scratch_mark)

        var out_in = concat(1, ctx, att_flat, mlp)
        return SingleBlockSaved(
            x_t.copy(), TArc(ln_t^), TArc(norm_t^), TArc(q_pre^), TArc(k_pre^),
            TArc(q_rms^), TArc(k_rms^), TArc(v^),
            TArc(q_rope^), TArc(k_rope^), TArc(att_flat^),
            TArc(mlp_gate^), TArc(mlp_up^), TArc(mlp^), TArc(out_in^),
        )


def single_block_lora_forward_device[
    H: Int, Dh: Int, S: Int
](
    x_t: TArc,
    w: SingleBlockWeights, mv: SingleModVecsDevice, lora: SingleBlockLora,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> SingleBlockDeviceForward:
    var lora_dev = single_block_lora_to_device(lora, ctx)
    return single_block_lora_forward_device_resident[H, Dh, S](
        x_t, w, mv, lora_dev, cos, sin, D, F, eps, ctx,
    )


def single_block_lora_forward[
    H: Int, Dh: Int, S: Int
](
    x: List[Float32],
    w: SingleBlockWeights, mv: SingleModVecs, lora: SingleBlockLora,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> SingleBlockForward:
    var mv_dev = single_modvecs_to_device(mv, D, ctx)
    var fwd = single_block_lora_forward_device[H, Dh, S](
        TArc(_t(x, [S, D], ctx)), w, mv_dev, lora, cos, sin, D, F, eps, ctx,
    )
    var out = fwd.out[].to_host(ctx)
    return SingleBlockForward(out^, fwd.saved.copy())


# ── BACKWARD of one SINGLE block WITH LoRA ───────────────────────────────────
def single_block_lora_backward_device_resident[
    H: Int, Dh: Int, S: Int
](
    d_out_t: TArc,
    w: SingleBlockWeights, mv: SingleModVecsDevice, lora: SingleBlockLoraDevice,
    saved: SingleBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
    compute_aux_grads: Bool = True,
    drop_qkv: LoraDropout = LoraDropout(),
    drop_out: LoraDropout = LoraDropout(),
) raises -> SingleBlockLoraDeviceGrads:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t_dtype(_ones(D), [D], saved.x[].dtype(), ctx)

    # result = residual_gate(x, gate, out). When aux modulation grads are
    # disabled, d_gate is discarded and `out` is not needed for d_x/d_y.
    var grg: GateResidualGrads
    var d_gate = List[Float32]()
    if compute_aux_grads:
        var nb = Optional[Tensor](None)
        var out_y = linear(saved.out_in[], w.w2[], nb^, ctx)
        if lora.out:
            var dlt2 = _klein_lora_fwd_dropout(
                saved.out_in[], lora.out.value(), S, drop_out, ctx
            )
            out_y = add(out_y, dlt2, ctx)
        grg = gate_residual_backward(
            d_out_t[], saved.x[], mv.gate[], out_y, ctx
        )
        d_gate = grg.d_g.to_host(ctx)
    else:
        grg = gate_residual_backward_dxdy(d_out_t[], mv.gate[], ctx)

    # base w2 backward (frozen W): d_x ONLY — base d_w2 was computed-then-discarded
    # (W2 is frozen; only LoRA trains). Skipping it drops the d_w2 GEMM + readback.
    var d_out_in_t = linear_backward_dx(
        grg.d_y, w.w2[], S, D + F, D, ctx,
    )

    # LoRA on to_out: input = FULL out_in [S,D+F], d_y = d_out_proj.
    # d_x_lo [S,D+F] adds into both attention and MLP portions.
    var out_d_a = List[Float32]()
    var out_d_b = List[Float32]()
    if lora.out:
        var lg2 = _klein_lora_bwd_dropout(
            grg.d_y, saved.out_in[], lora.out.value(), S, drop_out, ctx
        )
        d_out_in_t = add(d_out_in_t, lg2.d_x, ctx)
        out_d_a = lg2.d_a.copy()
        out_d_b = lg2.d_b.copy()

    var d_out_in_3d = reshape_owned(d_out_in_t^, [1, S, D + F])
    var cb = cat_backward(d_out_in_3d, D, F, 2, ctx)
    reshape_in_place(cb.d_0, [1, S, H, Dh])
    reshape_in_place(cb.d_1, [S, F])

    var sgb = swiglu_backward(cb.d_1, saved.mlp_gate[], saved.mlp_up[], ctx)
    var d_gate_up = concat(1, ctx, sgb.d_gate, sgb.d_up)

    var sb = sdpa_backward[1, S, H, Dh](
        saved.q_rope[], saved.k_rope[], saved.v[], cb.d_0, scale, ctx,
    )

    var d_q_rms = rope_backward(sb.d_q, cos, sin, True, ctx)
    var d_k_rms = rope_backward(sb.d_k, cos, sin, True, ctx)

    var d_q_pre_t = rms_norm_backward_dx(d_q_rms, saved.q_pre[], w.q_norm[], eps, ctx)
    var d_k_pre_t = rms_norm_backward_dx(d_k_rms, saved.k_pre[], w.k_norm[], eps, ctx)

    var d_q_pre_flat = reshape_owned(d_q_pre_t^, [S, D])
    var d_k_pre_flat = reshape_owned(d_k_pre_t^, [S, D])
    reshape_in_place(sb.d_v, [S, D])
    var d_qkv = concat(1, ctx, d_q_pre_flat, d_k_pre_flat, sb.d_v)   # [S,3D]

    var d_fused = concat(1, ctx, d_qkv, d_gate_up)   # [S, 3D+2F]

    # base w1 backward (frozen W): d_x ONLY — base d_w1 was computed-then-discarded
    # (W1 is frozen; only LoRA trains). Skipping it drops the d_w1 GEMM + readback.
    var d_norm_t = linear_backward_dx(
        d_fused, w.w1[], S, D, 3 * D + 2 * F, ctx,
    )

    # LoRA on to_qkv_mlp_proj: input = norm, d_y = FULL d_fused [S,3D+2F].
    # This trains q/k/v plus gate/up MLP columns as OneTrainer does.
    var qkv_d_a = List[Float32]()
    var qkv_d_b = List[Float32]()
    if lora.qkv:
        var lg = _klein_lora_bwd_dropout(
            d_fused, saved.norm[], lora.qkv.value(), S, drop_qkv, ctx
        )
        d_norm_t = add(d_norm_t, lg.d_x, ctx)
        qkv_d_a = lg.d_a.copy()
        qkv_d_b = lg.d_b.copy()

    var mb = modulate_backward(d_norm_t, saved.ln[], mv.scale[], ctx, compute_aux_grads)
    var d_scale = List[Float32]()
    var d_shift = List[Float32]()
    if compute_aux_grads:
        d_scale = mb.d_scale.to_host(ctx)
        d_shift = mb.d_shift.to_host(ctx)

    var d_x_norm_t = layer_norm_backward_dx(mb.d_x, saved.x[], ones_t, eps, ctx)

    var d_x_t = add(grg.d_x, d_x_norm_t, ctx)

    return SingleBlockLoraDeviceGrads(
        TArc(d_x_t^), d_shift^, d_scale^, d_gate^,
        qkv_d_a^, qkv_d_b^, out_d_a^, out_d_b^,
    )


def single_block_lora_backward_device_resident_scratch[
    H: Int, Dh: Int, S: Int
](
    d_out_t: TArc,
    w: SingleBlockWeights, mv: SingleModVecsDevice, lora: SingleBlockLoraDevice,
    saved: SingleBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    norm_ones: Tensor,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
    compute_aux_grads: Bool = True,
    drop_qkv: LoraDropout = LoraDropout(),
    drop_out: LoraDropout = LoraDropout(),
) raises -> SingleBlockLoraDeviceGrads:
    var scratch_mark = scratch.mark()
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var grg: GateResidualGrads
    var d_gate = List[Float32]()
    if compute_aux_grads:
        var out_y = linear_two_inputs_scratch(
            saved.att_flat[], saved.mlp[], w.w2_att[], w.w2_mlp[], ctx, scratch,
        )
        if lora.out:
            var out_in_t = concat(1, ctx, saved.att_flat[], saved.mlp[])
            var dlt2 = _klein_lora_fwd_dropout(
                out_in_t, lora.out.value(), S, drop_out, ctx
            )
            add_in_place_f32(out_y, dlt2, ctx)
        grg = gate_residual_backward(
            d_out_t[], saved.x[], mv.gate[], out_y, ctx
        )
        d_gate = grg.d_g.to_host(ctx)
    else:
        grg = gate_residual_backward_dxdy(d_out_t[], mv.gate[], ctx)

    var d_att = linear_backward_dx_scratch(
        grg.d_y, w.w2_att[], S, D, D, ctx, scratch,
    )
    var d_mlp = linear_backward_dx_scratch(
        grg.d_y, w.w2_mlp[], S, F, D, ctx, scratch,
    )

    var out_d_a = List[Float32]()
    var out_d_b = List[Float32]()
    if lora.out:
        var out_in_t = concat(1, ctx, saved.att_flat[], saved.mlp[])
        var lg2 = _klein_lora_bwd_dropout(
            grg.d_y, out_in_t, lora.out.value(), S, drop_out, ctx
        )
        add_in_place_f32(d_att, slice(lg2.d_x, 1, 0, D, ctx), ctx)
        add_in_place_f32(d_mlp, slice(lg2.d_x, 1, D, F, ctx), ctx)
        out_d_a = lg2.d_a.copy()
        out_d_b = lg2.d_b.copy()

    reshape_in_place(d_att, [1, S, H, Dh])

    var sgb = swiglu_backward(d_mlp, saved.mlp_gate[], saved.mlp_up[], ctx)
    var d_gate_up = concat2_scratch(1, ctx, scratch, sgb.d_gate, sgb.d_up)

    var sb = sdpa_backward_scratch[1, S, H, Dh](
        saved.q_rope[], saved.k_rope[], saved.v[], d_att, scale, ctx, scratch,
    )

    var d_q_rms = rope_backward(sb.d_q, cos, sin, True, ctx)
    var d_k_rms = rope_backward(sb.d_k, cos, sin, True, ctx)

    var d_q_pre_t = rms_norm_backward_dx(d_q_rms, saved.q_pre[], w.q_norm[], eps, ctx)
    var d_k_pre_t = rms_norm_backward_dx(d_k_rms, saved.k_pre[], w.k_norm[], eps, ctx)

    var d_q_pre_flat = reshape_owned(d_q_pre_t^, [S, D])
    var d_k_pre_flat = reshape_owned(d_k_pre_t^, [S, D])
    reshape_in_place(sb.d_v, [S, D])
    var d_qkv = concat3_scratch(1, ctx, scratch, d_q_pre_flat, d_k_pre_flat, sb.d_v, True)

    var d_norm_t = linear_backward_dx_split_scratch(
        d_qkv, d_gate_up, w.w1[], S, D, 3 * D, 2 * F, ctx, scratch,
    )

    var d_fused = concat(1, ctx, d_qkv, d_gate_up)
    var qkv_d_a = List[Float32]()
    var qkv_d_b = List[Float32]()
    if lora.qkv:
        var lg = _klein_lora_bwd_dropout(
            d_fused, saved.norm[], lora.qkv.value(), S, drop_qkv, ctx
        )
        d_norm_t = add(d_norm_t, lg.d_x, ctx)
        qkv_d_a = lg.d_a.copy()
        qkv_d_b = lg.d_b.copy()

    var mb = modulate_backward(d_norm_t, saved.ln[], mv.scale[], ctx, compute_aux_grads)
    var d_scale = List[Float32]()
    var d_shift = List[Float32]()
    if compute_aux_grads:
        d_scale = mb.d_scale.to_host(ctx)
        d_shift = mb.d_shift.to_host(ctx)

    var d_x_norm_t = layer_norm_backward_dx(mb.d_x, saved.x[], norm_ones, eps, ctx)

    var d_x_t = add(grg.d_x, d_x_norm_t, ctx)

    var out = SingleBlockLoraDeviceGrads(
        TArc(d_x_t^), d_shift^, d_scale^, d_gate^,
        qkv_d_a^, qkv_d_b^, out_d_a^, out_d_b^,
    )
    scratch.rewind(scratch_mark)
    return out^


def single_block_lora_backward_device_resident_scratch_tensors[
    H: Int, Dh: Int, S: Int
](
    d_out_t: TArc,
    w: SingleBlockWeights, mv: SingleModVecsDevice, lora: SingleBlockLoraDevice,
    saved: SingleBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    norm_ones: Tensor,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
    compute_aux_grads: Bool = True,
    drop_qkv: LoraDropout = LoraDropout(),
    drop_out: LoraDropout = LoraDropout(),
) raises -> SingleBlockLoraDeviceGradTensors:
    var scratch_mark = scratch.mark()
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var grg: GateResidualGrads
    var d_gate = List[Float32]()
    if compute_aux_grads:
        var out_y = linear_two_inputs_scratch(
            saved.att_flat[], saved.mlp[], w.w2_att[], w.w2_mlp[], ctx, scratch,
        )
        if lora.out:
            var out_in_t = concat(1, ctx, saved.att_flat[], saved.mlp[])
            var dlt2 = _klein_lora_fwd_dropout(
                out_in_t, lora.out.value(), S, drop_out, ctx
            )
            add_in_place_f32(out_y, dlt2, ctx)
        grg = gate_residual_backward(
            d_out_t[], saved.x[], mv.gate[], out_y, ctx
        )
        d_gate = grg.d_g.to_host(ctx)
    else:
        grg = gate_residual_backward_dxdy(d_out_t[], mv.gate[], ctx)

    var d_att = linear_backward_dx_scratch(
        grg.d_y, w.w2_att[], S, D, D, ctx, scratch,
    )
    var d_mlp = linear_backward_dx_scratch(
        grg.d_y, w.w2_mlp[], S, F, D, ctx, scratch,
    )

    var out_d_a = Optional[TArc](None)
    var out_d_b = Optional[TArc](None)
    if lora.out:
        var out_in_t = concat(1, ctx, saved.att_flat[], saved.mlp[])
        var lg2 = _klein_lora_bwd_dropout_tensors(
            grg.d_y, out_in_t, lora.out.value(), S, drop_out, ctx
        )
        add_in_place_f32(d_att, slice(lg2.d_x[], 1, 0, D, ctx), ctx)
        add_in_place_f32(d_mlp, slice(lg2.d_x[], 1, D, F, ctx), ctx)
        out_d_a = Optional[TArc](lg2.d_a.copy())
        out_d_b = Optional[TArc](lg2.d_b.copy())

    reshape_in_place(d_att, [1, S, H, Dh])

    var sgb = swiglu_backward(d_mlp, saved.mlp_gate[], saved.mlp_up[], ctx)
    var d_gate_up = concat2_scratch(1, ctx, scratch, sgb.d_gate, sgb.d_up)

    var d_q_sb: Tensor
    var d_k_sb: Tensor
    var d_v_sb: Tensor
    comptime if KLEIN_SDPA_FLASH:
        # flash backward from the tape's bf16 saved set (FAIL-LOUD if the
        # recompute path didn't fill it — fwd/bwd flag mismatch).
        if not saved.flash_stats:
            raise Error(
                "single_block bwd: KLEIN_SDPA_FLASH on but saved tape has"
                " no flash stats (recompute/backward flag mismatch)"
            )
        var fb = sdpa_flash_backward_f32[1, S, H, Dh](
            saved.flash_q.value(), saved.flash_k.value(),
            saved.flash_v.value(), saved.flash_o.value(),
            saved.flash_stats.value(), d_att, scale, ctx,
        )
        d_q_sb = Tensor(fb.d_q.buf.copy(), fb.d_q.shape(), fb.d_q.dtype())
        d_k_sb = Tensor(fb.d_k.buf.copy(), fb.d_k.shape(), fb.d_k.dtype())
        d_v_sb = Tensor(fb.d_v.buf.copy(), fb.d_v.shape(), fb.d_v.dtype())
    else:
        var sb = sdpa_backward_scratch[1, S, H, Dh](
            saved.q_rope[], saved.k_rope[], saved.v[], d_att, scale, ctx, scratch,
        )
        d_q_sb = Tensor(sb.d_q.buf.copy(), sb.d_q.shape(), sb.d_q.dtype())
        d_k_sb = Tensor(sb.d_k.buf.copy(), sb.d_k.shape(), sb.d_k.dtype())
        d_v_sb = Tensor(sb.d_v.buf.copy(), sb.d_v.shape(), sb.d_v.dtype())

    var d_q_rms = rope_backward(d_q_sb, cos, sin, True, ctx)
    var d_k_rms = rope_backward(d_k_sb, cos, sin, True, ctx)

    var d_q_pre_t = rms_norm_backward_dx(d_q_rms, saved.q_pre[], w.q_norm[], eps, ctx)
    var d_k_pre_t = rms_norm_backward_dx(d_k_rms, saved.k_pre[], w.k_norm[], eps, ctx)

    var d_q_pre_flat = reshape_owned(d_q_pre_t^, [S, D])
    var d_k_pre_flat = reshape_owned(d_k_pre_t^, [S, D])
    reshape_in_place(d_v_sb, [S, D])
    var d_qkv = concat3_scratch(1, ctx, scratch, d_q_pre_flat, d_k_pre_flat, d_v_sb, True)

    var d_norm_t = linear_backward_dx_split_scratch(
        d_qkv, d_gate_up, w.w1[], S, D, 3 * D, 2 * F, ctx, scratch,
    )

    var d_fused = concat(1, ctx, d_qkv, d_gate_up)
    var qkv_d_a = Optional[TArc](None)
    var qkv_d_b = Optional[TArc](None)
    if lora.qkv:
        var lg = _klein_lora_bwd_dropout_tensors(
            d_fused, saved.norm[], lora.qkv.value(), S, drop_qkv, ctx
        )
        d_norm_t = add(d_norm_t, lg.d_x[], ctx)
        qkv_d_a = Optional[TArc](lg.d_a.copy())
        qkv_d_b = Optional[TArc](lg.d_b.copy())

    var mb = modulate_backward(d_norm_t, saved.ln[], mv.scale[], ctx, compute_aux_grads)
    var d_scale = List[Float32]()
    var d_shift = List[Float32]()
    if compute_aux_grads:
        d_scale = mb.d_scale.to_host(ctx)
        d_shift = mb.d_shift.to_host(ctx)

    var d_x_norm_t = layer_norm_backward_dx(mb.d_x, saved.x[], norm_ones, eps, ctx)
    var d_x_t = add(grg.d_x, d_x_norm_t, ctx)

    var out = SingleBlockLoraDeviceGradTensors(
        TArc(d_x_t^), d_shift^, d_scale^, d_gate^,
        qkv_d_a^, qkv_d_b^, out_d_a^, out_d_b^,
    )
    scratch.rewind(scratch_mark)
    return out^


struct SingleBlockDirectDoRAGradsT(Copyable, Movable):
    var d_x: TArc
    var d_shift: List[Float32]
    var d_scale: List[Float32]
    var d_gate: List[Float32]
    var qkv: KleinDirectDoRAGradT
    var out: KleinDirectDoRAGradT

    def __init__(
        out self, var d_x: TArc,
        var d_shift: List[Float32], var d_scale: List[Float32], var d_gate: List[Float32],
        var qkv: KleinDirectDoRAGradT, var out_g: KleinDirectDoRAGradT,
    ):
        self.d_x = d_x^
        self.d_shift = d_shift^
        self.d_scale = d_scale^
        self.d_gate = d_gate^
        self.qkv = qkv^
        self.out = out_g^


struct SingleBlockDirectOFTGradsT(Copyable, Movable):
    var d_x: TArc
    var d_shift: List[Float32]
    var d_scale: List[Float32]
    var d_gate: List[Float32]
    var qkv: KleinDirectOFTGradT
    var out: KleinDirectOFTGradT

    def __init__(
        out self, var d_x: TArc,
        var d_shift: List[Float32], var d_scale: List[Float32], var d_gate: List[Float32],
        var qkv: KleinDirectOFTGradT, var out_g: KleinDirectOFTGradT,
    ):
        self.d_x = d_x^
        self.d_shift = d_shift^
        self.d_scale = d_scale^
        self.d_gate = d_gate^
        self.qkv = qkv^
        self.out = out_g^


def single_block_direct_dora_forward_device_resident_scratch[
    H: Int, Dh: Int, S: Int
](
    x_t: TArc,
    w: SingleBlockWeights, mv: SingleModVecsDevice, dora: KleinSingleDirectDoRA,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    norm_ones: Tensor, norm_zeros: Tensor,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> SingleBlockDeviceForward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var ln_t = layer_norm(x_t[], norm_ones, norm_zeros, eps, ctx)
    var norm_t = modulate(ln_t, mv.scale[], mv.shift[], ctx)

    var scratch_mark = scratch.mark()
    var q_pre_flat: Tensor
    var k_pre_flat: Tensor
    var v_flat: Tensor
    var gate_up: Tensor
    if dora.qkv:
        var fused = klein_direct_dora_projection_forward_optional(
            norm_t, w.w1[], dora.qkv, S, ctx,
        )
        q_pre_flat = slice(fused, 1, 0, D, ctx)
        k_pre_flat = slice(fused, 1, D, D, ctx)
        v_flat = slice(fused, 1, 2 * D, D, ctx)
        gate_up = slice(fused, 1, 3 * D, 2 * F, ctx)
    else:
        q_pre_flat = linear_rows(norm_t, w.w1[], 0, D, ctx)
        k_pre_flat = linear_rows(norm_t, w.w1[], D, D, ctx)
        v_flat = linear_rows(norm_t, w.w1[], 2 * D, D, ctx)
        gate_up = linear_rows_scratch(norm_t, w.w1[], 3 * D, 2 * F, ctx, scratch)
    var q_pre = reshape_owned(q_pre_flat^, [1, S, H, Dh])
    var k_pre = reshape_owned(k_pre_flat^, [1, S, H, Dh])
    var v = reshape_owned(v_flat^, [1, S, H, Dh])

    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    var q_rope = rope_interleaved(q_rms, cos, sin, ctx)
    var k_rope = rope_interleaved(k_rms, cos, sin, ctx)

    var flash_q = Optional[TArc](None)
    var flash_k = Optional[TArc](None)
    var flash_v = Optional[TArc](None)
    var flash_o = Optional[TArc](None)
    var flash_stats = Optional[TArc](None)
    var att_flat: Tensor
    comptime if KLEIN_SDPA_FLASH:
        var ff = sdpa_flash_train_fwd_f32[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
        var af_shape: List[Int] = [S, D]
        att_flat = Tensor(ff.att.buf.copy(), af_shape^, STDtype.F32)
        flash_q = Optional[TArc](ff.q_bf.copy())
        flash_k = Optional[TArc](ff.k_bf.copy())
        flash_v = Optional[TArc](ff.v_bf.copy())
        flash_o = Optional[TArc](ff.o_bf.copy())
        flash_stats = Optional[TArc](ff.stats.copy())
    else:
        var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
        att_flat = reshape_owned(att^, [S, D])

    var mlp_gate = slice(gate_up, 1, 0, F, ctx)
    var mlp_up = slice(gate_up, 1, F, F, ctx)
    var mlp = swiglu(mlp_gate, mlp_up, ctx)
    scratch.rewind(scratch_mark)

    var out_in_t = concat(1, ctx, att_flat, mlp)
    var out_proj: Tensor
    if dora.out:
        out_proj = klein_direct_dora_projection_forward_optional(
            out_in_t, w.w2[], dora.out, S, ctx,
        )
    else:
        var proj_mark = scratch.mark()
        out_proj = linear_two_inputs_scratch(
            att_flat, mlp, w.w2_att[], w.w2_mlp[], ctx, scratch,
        )
        scratch.rewind(proj_mark)

    var result = residual_gate(x_t[], mv.gate[], out_proj, ctx)
    var saved = SingleBlockSaved(
        x_t.copy(), TArc(ln_t^), TArc(norm_t^), TArc(q_pre^), TArc(k_pre^),
        TArc(q_rms^), TArc(k_rms^), TArc(v^),
        TArc(q_rope^), TArc(k_rope^), TArc(att_flat^),
        TArc(mlp_gate^), TArc(mlp_up^), TArc(mlp^), TArc(out_in_t^),
        flash_q^, flash_k^, flash_v^, flash_o^, flash_stats^,
    )
    return SingleBlockDeviceForward(TArc(result^), saved^)


def single_block_direct_oft_forward_device_resident_scratch[
    H: Int, Dh: Int, S: Int
](
    x_t: TArc,
    w: SingleBlockWeights, mv: SingleModVecsDevice, oft: KleinSingleDirectOFT,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    norm_ones: Tensor, norm_zeros: Tensor,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> SingleBlockDeviceForward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var ln_t = layer_norm(x_t[], norm_ones, norm_zeros, eps, ctx)
    var norm_t = modulate(ln_t, mv.scale[], mv.shift[], ctx)

    var scratch_mark = scratch.mark()
    var q_pre_flat: Tensor
    var k_pre_flat: Tensor
    var v_flat: Tensor
    var gate_up: Tensor
    if oft.qkv:
        var fused = klein_direct_oft_projection_forward_optional(
            norm_t, w.w1[], oft.qkv, S, ctx,
        )
        q_pre_flat = slice(fused, 1, 0, D, ctx)
        k_pre_flat = slice(fused, 1, D, D, ctx)
        v_flat = slice(fused, 1, 2 * D, D, ctx)
        gate_up = slice(fused, 1, 3 * D, 2 * F, ctx)
    else:
        q_pre_flat = linear_rows(norm_t, w.w1[], 0, D, ctx)
        k_pre_flat = linear_rows(norm_t, w.w1[], D, D, ctx)
        v_flat = linear_rows(norm_t, w.w1[], 2 * D, D, ctx)
        gate_up = linear_rows_scratch(norm_t, w.w1[], 3 * D, 2 * F, ctx, scratch)
    var q_pre = reshape_owned(q_pre_flat^, [1, S, H, Dh])
    var k_pre = reshape_owned(k_pre_flat^, [1, S, H, Dh])
    var v = reshape_owned(v_flat^, [1, S, H, Dh])

    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    var q_rope = rope_interleaved(q_rms, cos, sin, ctx)
    var k_rope = rope_interleaved(k_rms, cos, sin, ctx)

    var flash_q = Optional[TArc](None)
    var flash_k = Optional[TArc](None)
    var flash_v = Optional[TArc](None)
    var flash_o = Optional[TArc](None)
    var flash_stats = Optional[TArc](None)
    var att_flat: Tensor
    comptime if KLEIN_SDPA_FLASH:
        var ff = sdpa_flash_train_fwd_f32[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
        var af_shape: List[Int] = [S, D]
        att_flat = Tensor(ff.att.buf.copy(), af_shape^, STDtype.F32)
        flash_q = Optional[TArc](ff.q_bf.copy())
        flash_k = Optional[TArc](ff.k_bf.copy())
        flash_v = Optional[TArc](ff.v_bf.copy())
        flash_o = Optional[TArc](ff.o_bf.copy())
        flash_stats = Optional[TArc](ff.stats.copy())
    else:
        var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
        att_flat = reshape_owned(att^, [S, D])

    var mlp_gate = slice(gate_up, 1, 0, F, ctx)
    var mlp_up = slice(gate_up, 1, F, F, ctx)
    var mlp = swiglu(mlp_gate, mlp_up, ctx)
    scratch.rewind(scratch_mark)

    var out_in_t = concat(1, ctx, att_flat, mlp)
    var out_proj: Tensor
    if oft.out:
        out_proj = klein_direct_oft_projection_forward_optional(
            out_in_t, w.w2[], oft.out, S, ctx,
        )
    else:
        var proj_mark = scratch.mark()
        out_proj = linear_two_inputs_scratch(
            att_flat, mlp, w.w2_att[], w.w2_mlp[], ctx, scratch,
        )
        scratch.rewind(proj_mark)

    var result = residual_gate(x_t[], mv.gate[], out_proj, ctx)
    var saved = SingleBlockSaved(
        x_t.copy(), TArc(ln_t^), TArc(norm_t^), TArc(q_pre^), TArc(k_pre^),
        TArc(q_rms^), TArc(k_rms^), TArc(v^),
        TArc(q_rope^), TArc(k_rope^), TArc(att_flat^),
        TArc(mlp_gate^), TArc(mlp_up^), TArc(mlp^), TArc(out_in_t^),
        flash_q^, flash_k^, flash_v^, flash_o^, flash_stats^,
    )
    return SingleBlockDeviceForward(TArc(result^), saved^)


def single_block_direct_dora_backward_device_resident_scratch[
    H: Int, Dh: Int, S: Int
](
    d_out_t: TArc,
    w: SingleBlockWeights, mv: SingleModVecsDevice, dora: KleinSingleDirectDoRA,
    saved: SingleBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    norm_ones: Tensor,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
    compute_aux_grads: Bool = True,
) raises -> SingleBlockDirectDoRAGradsT:
    var scratch_mark = scratch.mark()
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var grg: GateResidualGrads
    var d_gate = List[Float32]()
    if compute_aux_grads:
        var out_y: Tensor
        if dora.out:
            out_y = klein_direct_dora_projection_forward_optional(
                saved.out_in[], w.w2[], dora.out, S, ctx,
            )
        else:
            out_y = linear_two_inputs_scratch(
                saved.att_flat[], saved.mlp[], w.w2_att[], w.w2_mlp[], ctx, scratch,
            )
        grg = gate_residual_backward(d_out_t[], saved.x[], mv.gate[], out_y, ctx)
        d_gate = grg.d_g.to_host(ctx)
    else:
        grg = gate_residual_backward_dxdy(d_out_t[], mv.gate[], ctx)

    var out_grad = KleinDirectDoRAGradT(None, None, None)
    var d_att: Tensor
    var d_mlp: Tensor
    if dora.out:
        var bw_out = klein_direct_dora_projection_backward_optional(
            grg.d_y, saved.out_in[], w.w2[], dora.out, S, D + F, D, ctx,
        )
        var d_out_in = bw_out.d_x.clone(ctx)
        d_att = slice(d_out_in, 1, 0, D, ctx)
        d_mlp = slice(d_out_in, 1, D, F, ctx)
        out_grad = bw_out.dora.copy()
    else:
        d_att = linear_backward_dx_scratch(
            grg.d_y, w.w2_att[], S, D, D, ctx, scratch,
        )
        d_mlp = linear_backward_dx_scratch(
            grg.d_y, w.w2_mlp[], S, F, D, ctx, scratch,
        )

    reshape_in_place(d_att, [1, S, H, Dh])
    var sgb = swiglu_backward(d_mlp, saved.mlp_gate[], saved.mlp_up[], ctx)
    var d_gate_up = concat2_scratch(1, ctx, scratch, sgb.d_gate, sgb.d_up)

    var d_q_sb: Tensor
    var d_k_sb: Tensor
    var d_v_sb: Tensor
    comptime if KLEIN_SDPA_FLASH:
        if not saved.flash_stats:
            raise Error("single_block direct DoRA bwd: missing flash stats")
        var fb = sdpa_flash_backward_f32[1, S, H, Dh](
            saved.flash_q.value(), saved.flash_k.value(),
            saved.flash_v.value(), saved.flash_o.value(),
            saved.flash_stats.value(), d_att, scale, ctx,
        )
        d_q_sb = Tensor(fb.d_q.buf.copy(), fb.d_q.shape(), fb.d_q.dtype())
        d_k_sb = Tensor(fb.d_k.buf.copy(), fb.d_k.shape(), fb.d_k.dtype())
        d_v_sb = Tensor(fb.d_v.buf.copy(), fb.d_v.shape(), fb.d_v.dtype())
    else:
        var sb = sdpa_backward_scratch[1, S, H, Dh](
            saved.q_rope[], saved.k_rope[], saved.v[], d_att, scale, ctx, scratch,
        )
        d_q_sb = Tensor(sb.d_q.buf.copy(), sb.d_q.shape(), sb.d_q.dtype())
        d_k_sb = Tensor(sb.d_k.buf.copy(), sb.d_k.shape(), sb.d_k.dtype())
        d_v_sb = Tensor(sb.d_v.buf.copy(), sb.d_v.shape(), sb.d_v.dtype())

    var d_q_rms = rope_backward(d_q_sb, cos, sin, True, ctx)
    var d_k_rms = rope_backward(d_k_sb, cos, sin, True, ctx)
    var d_q_pre_t = rms_norm_backward_dx(d_q_rms, saved.q_pre[], w.q_norm[], eps, ctx)
    var d_k_pre_t = rms_norm_backward_dx(d_k_rms, saved.k_pre[], w.k_norm[], eps, ctx)
    var d_q_pre_flat = reshape_owned(d_q_pre_t^, [S, D])
    var d_k_pre_flat = reshape_owned(d_k_pre_t^, [S, D])
    reshape_in_place(d_v_sb, [S, D])
    var d_qkv = concat3_scratch(1, ctx, scratch, d_q_pre_flat, d_k_pre_flat, d_v_sb, True)
    var d_fused = concat(1, ctx, d_qkv, d_gate_up)

    var qkv_grad = KleinDirectDoRAGradT(None, None, None)
    var d_norm_t: Tensor
    if dora.qkv:
        var bw_qkv = klein_direct_dora_projection_backward_optional(
            d_fused, saved.norm[], w.w1[], dora.qkv, S, D, 3 * D + 2 * F, ctx,
        )
        d_norm_t = bw_qkv.d_x.clone(ctx)
        qkv_grad = bw_qkv.dora.copy()
    else:
        d_norm_t = linear_backward_dx_split_scratch(
            d_qkv, d_gate_up, w.w1[], S, D, 3 * D, 2 * F, ctx, scratch,
        )

    var mb = modulate_backward(d_norm_t, saved.ln[], mv.scale[], ctx, compute_aux_grads)
    var d_scale = List[Float32]()
    var d_shift = List[Float32]()
    if compute_aux_grads:
        d_scale = mb.d_scale.to_host(ctx)
        d_shift = mb.d_shift.to_host(ctx)

    var d_x_norm_t = layer_norm_backward_dx(mb.d_x, saved.x[], norm_ones, eps, ctx)
    var d_x_t = add(grg.d_x, d_x_norm_t, ctx)
    var out = SingleBlockDirectDoRAGradsT(
        TArc(d_x_t^), d_shift^, d_scale^, d_gate^, qkv_grad^, out_grad^,
    )
    scratch.rewind(scratch_mark)
    return out^


def single_block_direct_oft_backward_device_resident_scratch[
    H: Int, Dh: Int, S: Int
](
    d_out_t: TArc,
    w: SingleBlockWeights, mv: SingleModVecsDevice, oft: KleinSingleDirectOFT,
    saved: SingleBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    norm_ones: Tensor,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
    compute_aux_grads: Bool = True,
) raises -> SingleBlockDirectOFTGradsT:
    var scratch_mark = scratch.mark()
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var grg: GateResidualGrads
    var d_gate = List[Float32]()
    if compute_aux_grads:
        var out_y: Tensor
        if oft.out:
            out_y = klein_direct_oft_projection_forward_optional(
                saved.out_in[], w.w2[], oft.out, S, ctx,
            )
        else:
            out_y = linear_two_inputs_scratch(
                saved.att_flat[], saved.mlp[], w.w2_att[], w.w2_mlp[], ctx, scratch,
            )
        grg = gate_residual_backward(d_out_t[], saved.x[], mv.gate[], out_y, ctx)
        d_gate = grg.d_g.to_host(ctx)
    else:
        grg = gate_residual_backward_dxdy(d_out_t[], mv.gate[], ctx)

    var out_grad = KleinDirectOFTGradT(None)
    var d_att: Tensor
    var d_mlp: Tensor
    if oft.out:
        var bw_out = klein_direct_oft_projection_backward_optional(
            grg.d_y, saved.out_in[], w.w2[], oft.out, S, D + F, D, ctx,
        )
        var d_out_in = bw_out.d_x.clone(ctx)
        d_att = slice(d_out_in, 1, 0, D, ctx)
        d_mlp = slice(d_out_in, 1, D, F, ctx)
        out_grad = bw_out.oft.copy()
    else:
        d_att = linear_backward_dx_scratch(
            grg.d_y, w.w2_att[], S, D, D, ctx, scratch,
        )
        d_mlp = linear_backward_dx_scratch(
            grg.d_y, w.w2_mlp[], S, F, D, ctx, scratch,
        )

    reshape_in_place(d_att, [1, S, H, Dh])
    var sgb = swiglu_backward(d_mlp, saved.mlp_gate[], saved.mlp_up[], ctx)
    var d_gate_up = concat2_scratch(1, ctx, scratch, sgb.d_gate, sgb.d_up)

    var d_q_sb: Tensor
    var d_k_sb: Tensor
    var d_v_sb: Tensor
    comptime if KLEIN_SDPA_FLASH:
        if not saved.flash_stats:
            raise Error("single_block direct OFT bwd: missing flash stats")
        var fb = sdpa_flash_backward_f32[1, S, H, Dh](
            saved.flash_q.value(), saved.flash_k.value(),
            saved.flash_v.value(), saved.flash_o.value(),
            saved.flash_stats.value(), d_att, scale, ctx,
        )
        d_q_sb = Tensor(fb.d_q.buf.copy(), fb.d_q.shape(), fb.d_q.dtype())
        d_k_sb = Tensor(fb.d_k.buf.copy(), fb.d_k.shape(), fb.d_k.dtype())
        d_v_sb = Tensor(fb.d_v.buf.copy(), fb.d_v.shape(), fb.d_v.dtype())
    else:
        var sb = sdpa_backward_scratch[1, S, H, Dh](
            saved.q_rope[], saved.k_rope[], saved.v[], d_att, scale, ctx, scratch,
        )
        d_q_sb = Tensor(sb.d_q.buf.copy(), sb.d_q.shape(), sb.d_q.dtype())
        d_k_sb = Tensor(sb.d_k.buf.copy(), sb.d_k.shape(), sb.d_k.dtype())
        d_v_sb = Tensor(sb.d_v.buf.copy(), sb.d_v.shape(), sb.d_v.dtype())

    var d_q_rms = rope_backward(d_q_sb, cos, sin, True, ctx)
    var d_k_rms = rope_backward(d_k_sb, cos, sin, True, ctx)
    var d_q_pre_t = rms_norm_backward_dx(d_q_rms, saved.q_pre[], w.q_norm[], eps, ctx)
    var d_k_pre_t = rms_norm_backward_dx(d_k_rms, saved.k_pre[], w.k_norm[], eps, ctx)
    var d_q_pre_flat = reshape_owned(d_q_pre_t^, [S, D])
    var d_k_pre_flat = reshape_owned(d_k_pre_t^, [S, D])
    reshape_in_place(d_v_sb, [S, D])
    var d_qkv = concat3_scratch(1, ctx, scratch, d_q_pre_flat, d_k_pre_flat, d_v_sb, True)
    var d_fused = concat(1, ctx, d_qkv, d_gate_up)

    var qkv_grad = KleinDirectOFTGradT(None)
    var d_norm_t: Tensor
    if oft.qkv:
        var bw_qkv = klein_direct_oft_projection_backward_optional(
            d_fused, saved.norm[], w.w1[], oft.qkv, S, D, 3 * D + 2 * F, ctx,
        )
        d_norm_t = bw_qkv.d_x.clone(ctx)
        qkv_grad = bw_qkv.oft.copy()
    else:
        d_norm_t = linear_backward_dx_split_scratch(
            d_qkv, d_gate_up, w.w1[], S, D, 3 * D, 2 * F, ctx, scratch,
        )

    var mb = modulate_backward(d_norm_t, saved.ln[], mv.scale[], ctx, compute_aux_grads)
    var d_scale = List[Float32]()
    var d_shift = List[Float32]()
    if compute_aux_grads:
        d_scale = mb.d_scale.to_host(ctx)
        d_shift = mb.d_shift.to_host(ctx)

    var d_x_norm_t = layer_norm_backward_dx(mb.d_x, saved.x[], norm_ones, eps, ctx)
    var d_x_t = add(grg.d_x, d_x_norm_t, ctx)
    var out = SingleBlockDirectOFTGradsT(
        TArc(d_x_t^), d_shift^, d_scale^, d_gate^, qkv_grad^, out_grad^,
    )
    scratch.rewind(scratch_mark)
    return out^


def single_block_lora_backward_device[
    H: Int, Dh: Int, S: Int
](
    d_out_t: TArc,
    w: SingleBlockWeights, mv: SingleModVecsDevice, lora: SingleBlockLora,
    saved: SingleBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
    compute_aux_grads: Bool = True,
) raises -> SingleBlockLoraDeviceGrads:
    var lora_dev = single_block_lora_to_device(lora, ctx)
    return single_block_lora_backward_device_resident[H, Dh, S](
        d_out_t, w, mv, lora_dev, saved, cos, sin, D, F, eps, ctx, compute_aux_grads,
    )


def single_block_lora_backward[
    H: Int, Dh: Int, S: Int
](
    d_out: List[Float32],
    w: SingleBlockWeights, mv: SingleModVecs, lora: SingleBlockLora,
    saved: SingleBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> SingleBlockLoraGrads:
    var mv_dev = single_modvecs_to_device(mv, D, ctx)
    var dg = single_block_lora_backward_device[H, Dh, S](
        TArc(_t(d_out, [S, D], ctx)), w, mv_dev, lora, saved, cos, sin, D, F, eps, ctx,
    )
    var d_x = dg.d_x[].to_host(ctx)
    var base = SingleBlockGrads(
        d_x^, List[Float32](), List[Float32](), List[Float32](), List[Float32](),
        dg.d_shift.copy(), dg.d_scale.copy(), dg.d_gate.copy(),
    )
    return SingleBlockLoraGrads(
        base^,
        dg.qkv_d_a.copy(), dg.qkv_d_b.copy(), dg.out_d_a.copy(), dg.out_d_b.copy(),
    )
