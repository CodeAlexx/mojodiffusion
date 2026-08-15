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

from max.gpu.host import DeviceContext
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
from serenitymojo.ops.norm import rms_norm, layer_norm, layer_norm_modulate, lnmod_placeholder
from serenitymojo.ops.activations import swiglu
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.cast import cast_tensor
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
# b2rs backward A/B (2026-07-06): the batched [2,S] flash BACKWARD measured
# SLOWER than 2x math bwd in the interleaved pair (43.3 vs 37.6 s/12 at
# S=1536) — HYPOTHESIS REFUTED 2026-07-06: math arm measured WORSE
# (bwd 44.9 vs flash 43.3 vs interleaved 37.6 s/12; step-1 loss 0.7715).
# Flash stays the b2rs bwd arm. Next suspect: scratch-ring slot capacity at
# [2S,*] transients (fallback alloc churn) — unmeasured.
comptime KLEIN_B2RS_FLASH_BWD = True
from serenitymojo.ops.tensor_algebra import (
    reshape, reshape_owned, reshape_in_place, slice, concat, add, add_in_place_f32,
    add_band_in_place_f32, mul, mul_scalar, zeros_device,
)
from serenitymojo.ops.tensor_algebra_scratch import (
    concat2_scratch, concat3_scratch, slice_scratch,
)
# int8 W8A8 base ops (cfg.quantized_resident=="int_w8a8"). MIRRORS krea2's int8
# pattern: quantize the frozen base weight TENSORWISE once at load; per step run
# int8×int8→int32 GEMM with per-token activation quant (NO per-step dequant).
from serenitymojo.ops.int8_quant import (
    int8_tensorwise_scale, int8_encode_tensorwise, int8_transpose,
)
from serenitymojo.ops.int8_linear import (
    int8_linear_fwd, int8_linear_bwd, int8_linear_bwd_nn,
    int8_linear_fwd_f32, int8_linear_bwd_nn_f32, int8_linear_bwd_nt_f32,
    int8_linear_fwd_f32_bands, int8_linear_fwd_f32_bands_addfull,
    int8_linear_bwd_nn_f32_bands, int8_linear_bwd_nt_f32_bands,
    int8_band_tensor,
)
# SquareQ NVFP4 native-FP4 base fwd (cfg.quantized_resident=="squareq_nvfp4"):
# frozen-base linears run alpha*fp4gemm(rht_quant(x), nvq) + (x@ld)@lu^T when
# the payload is present. FORWARD-ONLY dispatch — every backward arm keeps
# consuming the reconstructed BF16 W_hat the loader put in the Block under the
# weight's own name (ops/squareq_nvfp4.mojo header: disclosed numerics-mismatch
# STE, same class as torchref's).
from serenitymojo.ops.squareq_nvfp4 import squareq_nvfp4_linear

# ── backward arms (GPU; all pre-built + gated) ───────────────────────────────
from serenitymojo.ops.linalg_backward import (
    linear_backward, linear_backward_dx, linear_backward_dx_scratch,
    linear_backward_dx_split_scratch, linear_backward_dw, LinearGrads,
)
from serenitymojo.util.bf16_stochastic_rounding import sr_uniform
from serenitymojo.ops.norm_backward import (
    rms_norm_backward, rms_norm_backward_dx, rms_norm_backward_dg, RmsNormBackward,
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
    # int8-W8A8 loader sidecar (Klein int8 slice 4): when the block was loaded
    # int8-resident (pin_residents_int8), w1/w2/w2_att/w2_mlp above are BF16 [1,1]
    # dummies (the bf16 GEMM is skipped) and this holds the (int8 weight, F32 scalar
    # scale) payload the int8 forward reads. None on the bf16 path (default).
    var int8: Optional[SingleBlockInt8]
    # squareq_nvfp4 loader sidecar (SquareQ chunk 8): when the block was pinned
    # nvfp4-resident (pin_residents_squareq_nvfp4), w1/w2/w2_att/w2_mlp above
    # hold the RECONSTRUCTED BF16 W_hat (every backward arm reads them
    # unchanged) and this holds the packed (nvq/nvs/ld/lu + nvg) payload the
    # native-FP4 FORWARD reads. None everywhere else (default) — the bf16/int8
    # paths are byte-identical with this None.
    var nvfp4: Optional[SingleBlockNvfp4]

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
        self.int8 = Optional[SingleBlockInt8](None)
        self.nvfp4 = Optional[SingleBlockNvfp4](None)

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
        self.int8 = Optional[SingleBlockInt8](None)
        self.nvfp4 = Optional[SingleBlockNvfp4](None)

    # int8-W8A8 loader constructor (Klein int8 slice 4): the base weights live in
    # the int8 payload; w1/w2/w2_att/w2_mlp are BF16 [1,1] dummies (never read on
    # the int8 forward path — the frozen base runs int8_linear_fwd from the payload,
    # NOT the bf16 `linear`). No slicing of the (int8) w2 — mirrors krea2's
    # dummy-bf16-fields-on-the-int8-path pattern.
    def __init__(
        out self,
        var int8: SingleBlockInt8,
        var q_norm: TArc, var k_norm: TArc,
        ctx: DeviceContext,
    ) raises:
        self.w1 = TArc(zeros_device([1, 1], STDtype.BF16, ctx))
        self.w2 = TArc(zeros_device([1, 1], STDtype.BF16, ctx))
        self.w2_att = TArc(zeros_device([1, 1], STDtype.BF16, ctx))
        self.w2_mlp = TArc(zeros_device([1, 1], STDtype.BF16, ctx))
        self.q_norm = q_norm^
        self.k_norm = k_norm^
        self.int8 = Optional[SingleBlockInt8](int8^)
        self.nvfp4 = Optional[SingleBlockNvfp4](None)


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
    # Flash-SDPA saved set (Optional: the KLEIN_SDPA_FLASH arms of the scratch
    # forward AND the recompute path fill these; legacy math constructor
    # sites pass nothing): bf16 q_rope/k_rope/v + bf16 O + F32 LSE stats —
    # exactly what sdpa_flash_backward_f32 consumes, no re-casting in backward.
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

    # ln = layer_norm(x, 1, 0); weight is frozen ones -> d_x-only backward
    var lnb_dx = layer_norm_backward_dx(mb.d_x, saved.x[], ones_t, eps, ctx)

    # x feeds BOTH the residual (grg.d_x) AND layer_norm(x) -> SUM.
    # gate_residual_backward gives d_x = grad_out (passthrough); sum on host at
    # the boundary readback (both are [S,D] device grads).
    var d_x_res = grg.d_x.to_host(ctx)
    var d_x_norm = lnb_dx.to_host(ctx)
    var d_x = _add_lists(d_x_res, d_x_norm)

    return SingleBlockGrads(
        d_x^, d_w1^, d_w2^, d_q_norm^, d_k_norm^,
        d_shift^, d_scale^, d_gate^,
    )


# ═══════════════════════════════════════════════════════════════════════════
# LoRA-ON-PROJECTION VARIANT
#
# Targets (matches SerenityTrainer Flux2 single_blocks):
#   linear1 (w1) to_qkv_mlp slot: the LoRA delta covers the full SerenityTrainer
#     fused projection [S, 3D + 2F], including Q/K/V and gate/up rows.
#   linear2 (w2) to_out slot: the LoRA input is the full [att_flat, mlp]
#     concatenation [S, D + F], matching SerenityTrainer's single block target.
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
    # ALPHA-FOLD (2026-07-11): for the F32 activation chain the up-GEMM writes
    # the exact F32 accumulator, so mul_scalar computed fl(scale·acc) — the
    # SAME single F32 multiply cuBLAS applies as the GEMM epilogue alpha.
    # linear_rows(t, B, 0, out_f, alpha) is linear(t, B) with alpha folded →
    # BIT-IDENTICAL output, one full-tensor kernel + buffer fewer.
    # Gate: models/klein/parity/klein_lora_scale_band_parity.mojo (max_abs 0.0).
    if t.dtype() == STDtype.F32:
        return linear_rows(t, lo.b[], 0, lo.out_f, ctx, alpha=lo.scale)
    # Narrow (bf16/f16) storage rounds the GEMM output before mul_scalar; the
    # fold would collapse that double rounding → keep the legacy chain there.
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

    var ln_t = lnmod_placeholder(ctx)
    var norm_t = layer_norm_modulate(x_t[], mv.scale[], mv.shift[], eps, ctx, ln_t)

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
    int8: Optional[SingleBlockInt8] = None,
    recompute_only: Bool = False,
) raises -> SingleBlockDeviceForward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var ln_t = lnmod_placeholder(ctx)
    var norm_t = layer_norm_modulate(x_t[], mv.scale[], mv.shift[], eps, ctx, ln_t)

    var scratch_mark = scratch.mark()
    # FROZEN base w1: int8 W8A8 dispatch (krea2 in-function pattern). When the
    # int8 payload is present, run ONE full int8 GEMM norm_t@w1ᵀ → [S,3D+2F] and
    # CHANNEL-slice the q/k/v/gate_up bands — the exact bands the 4 `linear_rows`
    # produce (rows [0:D],[D:2D],[2D:3D],[3D:3D+2F] of w1 == cols 0..3D+2F of the
    # fused GEMM). When None this is BYTE-IDENTICAL to the bf16 linear_rows path.
    var q_pre_flat: Tensor
    var k_pre_flat: Tensor
    var v_flat: Tensor
    var gate_up: Tensor
    # LoRA qkv delta FIRST (it reads only norm_t) so the F32-int8 arm can fold
    # the band add into the dequant epilogue (fusion pass 2). Hoisting changes
    # no value — the delta chain and the base GEMM are independent.
    var qkv_dlt = Optional[Tensor](None)
    if lora.qkv:
        qkv_dlt = Optional[Tensor](_klein_lora_fwd_dropout(
            norm_t, lora.qkv.value(), S, drop_qkv, ctx
        ))
    var delta_fused = False
    if w.nvfp4:
        # SquareQ NVFP4 native-FP4 base w1 (FORWARD-ONLY dispatch): ONE fused
        # fp4 GEMM norm_t@W1_hatᵀ → [S,3D+2F], channel-sliced into the exact
        # q/k/v/gate_up bands the bf16/int8 arms produce. The LoRA delta stays
        # on the unchanged band-add path below (delta_fused stays False).
        var fused_nv = _base_fwd_nv4(norm_t, w.nvfp4.value(), 0, ctx)
        q_pre_flat = slice(fused_nv, 1, 0, D, ctx)
        k_pre_flat = slice(fused_nv, 1, D, D, ctx)
        v_flat = slice(fused_nv, 1, 2 * D, D, ctx)
        gate_up = slice(fused_nv, 1, 3 * D, 2 * F, ctx)
    elif int8 and norm_t.dtype() == STDtype.F32:
        # FUSED F32 boundary + band-split (2026-07-11): quant reads the F32
        # norm_t directly (bf16 rounding in-register == the old cast_tensor),
        # the dequant writes the 4 q/k/v/gate_up column bands as separate F32
        # tensors (== the old dequant→cast→4×slice chain, bit-identically),
        # removing 2 full-[S,3D+2F] casts + 4 slice kernels per call.
        # FUSED delta-add (fusion pass 2, 2026-07-11): a F32 qkv LoRA delta is
        # folded into the dequant band stores — the SAME F32 add that
        # add_band_in_place_f32 performed, minus 4 band-add kernels and a full
        # re-read/re-write of the bands. Gate: klein_int8_f32_fusion_parity.
        ref p8f = int8.value()
        var bands: List[ArcPointer[Tensor]]
        if qkv_dlt and qkv_dlt.value().dtype() == STDtype.F32:
            bands = int8_linear_fwd_f32_bands_addfull(
                norm_t, p8f.w8[0][], p8f.scale[0][], qkv_dlt.value(),
                D, D, D, 2 * F, ctx,
            )
            delta_fused = True
        else:
            bands = int8_linear_fwd_f32_bands(
                norm_t, p8f.w8[0][], p8f.scale[0][], D, D, D, 2 * F, ctx,
            )
        q_pre_flat = int8_band_tensor(bands[0])
        k_pre_flat = int8_band_tensor(bands[1])
        v_flat = int8_band_tensor(bands[2])
        gate_up = int8_band_tensor(bands[3])
    elif int8:
        var fused_i8 = _base_fwd_i8(norm_t, w.w1[], int8, 0, ctx)   # [S,3D+2F]
        q_pre_flat = slice(fused_i8, 1, 0, D, ctx)
        k_pre_flat = slice(fused_i8, 1, D, D, ctx)
        v_flat = slice(fused_i8, 1, 2 * D, D, ctx)
        gate_up = slice(fused_i8, 1, 3 * D, 2 * F, ctx)
    else:
        q_pre_flat = linear_rows(norm_t, w.w1[], 0, D, ctx)
        k_pre_flat = linear_rows(norm_t, w.w1[], D, D, ctx)
        v_flat = linear_rows(norm_t, w.w1[], 2 * D, D, ctx)
        gate_up = linear_rows_scratch(norm_t, w.w1[], 3 * D, 2 * F, ctx, scratch)
    if qkv_dlt and not delta_fused:
        ref dlt = qkv_dlt.value()
        if dlt.dtype() == STDtype.F32 and q_pre_flat.dtype() == STDtype.F32:
            # FUSED band add (2026-07-11): dst += dlt[:, band] straight out of
            # the delta — same F32 add, same slice element mapping, minus the
            # 4 materialized band copies.
            add_band_in_place_f32(q_pre_flat, dlt, 0, ctx)
            add_band_in_place_f32(k_pre_flat, dlt, D, ctx)
            add_band_in_place_f32(v_flat, dlt, 2 * D, ctx)
            add_band_in_place_f32(gate_up, dlt, 3 * D, ctx)
        else:
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
    var att_flat: Tensor
    var flash_q = Optional[TArc](None)
    var flash_k = Optional[TArc](None)
    var flash_v = Optional[TArc](None)
    var flash_o = Optional[TArc](None)
    var flash_stats = Optional[TArc](None)
    comptime if KLEIN_SDPA_FLASH:
        # cuDNN flash with F32<->bf16 boundary casts (same arm the recompute
        # path below uses); bf16 q/k/v/o + stats go to the tape so the flash
        # backward consumes them directly — no recompute, no re-cast.
        var ff = sdpa_flash_train_fwd_f32[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
        # zero-copy re-box [1,S,H,Dh] -> [S,D] (no partial move out of ff)
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
    if att_flat.dtype() != mlp.dtype():
        att_flat = cast_tensor(att_flat, mlp.dtype(), ctx)
    scratch.rewind(scratch_mark)

    var out_in_t = concat(1, ctx, att_flat, mlp)
    if recompute_only:
        # DEAD-TAIL SKIP (fusion pass 2, 2026-07-11): the backward-walk
        # recompute consumes ONLY `saved` — the block output is discarded and
        # the aux-off devgrads backward never reads the out projection VALUE
        # (it reads saved.out_in for the LoRA-out grads). Skipping the
        # w2/LoRA-out/residual_gate tail removes an int8 quant+GEMM+dequant on
        # [S,D+F], the LoRA-out fwd chain and the gate per recomputed block.
        # `saved` is built by the IDENTICAL kernels/order as the full path →
        # the tape is bit-identical to what the full forward would have saved.
        var saved_r = SingleBlockSaved(
            x_t.copy(), TArc(ln_t^), TArc(norm_t^), TArc(q_pre^), TArc(k_pre^),
            TArc(q_rms^), TArc(k_rms^), TArc(v^),
            TArc(q_rope^), TArc(k_rope^), TArc(att_flat^),
            TArc(mlp_gate^), TArc(mlp_up^), TArc(mlp^), TArc(out_in_t^),
            flash_q^, flash_k^, flash_v^, flash_o^, flash_stats^,
        )
        return SingleBlockDeviceForward(x_t.copy(), saved_r^)
    var proj_mark = scratch.mark()
    # FROZEN base w2: int8 W8A8 dispatch. int8_linear_fwd on the FULL out_in
    # [S,D+F] with the full-w2 int8 payload equals att@w2[:,:D]ᵀ + mlp@w2[:,D:]ᵀ
    # (== concat(att,mlp)@w2ᵀ) that linear_two_inputs_scratch computes bf16. When
    # None this is BYTE-IDENTICAL to the bf16 two-input path.
    var out_proj: Tensor
    if w.nvfp4:
        # SquareQ NVFP4 native-FP4 base w2 (FORWARD-ONLY dispatch): fp4 GEMM
        # on the FULL out_in [S,D+F] == concat(att,mlp)@W2_hatᵀ.
        out_proj = _base_fwd_nv4(out_in_t, w.nvfp4.value(), 1, ctx)   # [S,D]
    elif int8:
        out_proj = _base_fwd_i8(out_in_t, w.w2[], int8, 1, ctx)   # [S,D]
    else:
        out_proj = linear_two_inputs_scratch(
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
        flash_q^, flash_k^, flash_v^, flash_o^, flash_stats^,
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

    var ln_t = lnmod_placeholder(ctx)
    var norm_t = layer_norm_modulate(x_t[], mv.scale[], mv.shift[], eps, ctx, ln_t)

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
    var att_flat: Tensor
    comptime if KLEIN_SDPA_FLASH:
        var ff = sdpa_flash_train_fwd_f32[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
        var af_shape: List[Int] = [S, D]
        att_flat = Tensor(ff.att.buf.copy(), af_shape^, STDtype.F32)
    else:
        var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
        att_flat = reshape_owned(att^, [S, D])

    var mlp_gate = slice(gate_up, 1, 0, F, ctx)
    var mlp_up = slice(gate_up, 1, F, F, ctx)
    var mlp = swiglu(mlp_gate, mlp_up, ctx)
    if att_flat.dtype() != mlp.dtype():
        att_flat = cast_tensor(att_flat, mlp.dtype(), ctx)
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

    var ln_t = lnmod_placeholder(ctx)
    var norm_t = layer_norm_modulate(x_t[], mv.scale[], mv.shift[], eps, ctx, ln_t)

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

    var ln_t = lnmod_placeholder(ctx)
    var norm_t = layer_norm_modulate(x_t[], mv.scale[], mv.shift[], eps, ctx, ln_t)

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
    # This trains q/k/v plus gate/up MLP columns as SerenityTrainer does.
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
    int8: Optional[SingleBlockInt8] = None,
) raises -> SingleBlockLoraDeviceGrads:
    var scratch_mark = scratch.mark()
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var grg: GateResidualGrads
    var d_gate = List[Float32]()
    if compute_aux_grads:
        # Aux recompute of out_y (== forward out projection). Only d_gate depends
        # on out_y's VALUE (grg.d_x/d_y do not), so the int8 quant here shifts
        # d_gate to match the int8 forward while leaving the base dX path exact.
        var out_y: Tensor
        if int8:
            # saved.out_in IS concat(att_flat, mlp) — read the tape (pass 2).
            out_y = _base_fwd_i8(saved.out_in[], w.w2[], int8, 1, ctx)   # full-w2 int8
        else:
            out_y = linear_two_inputs_scratch(
                saved.att_flat[], saved.mlp[], w.w2_att[], w.w2_mlp[], ctx, scratch,
            )
        if lora.out:
            var dlt2 = _klein_lora_fwd_dropout(
                saved.out_in[], lora.out.value(), S, drop_out, ctx
            )
            add_in_place_f32(out_y, dlt2, ctx)
        grg = gate_residual_backward(
            d_out_t[], saved.x[], mv.gate[], out_y, ctx
        )
        d_gate = grg.d_g.to_host(ctx)
    else:
        grg = gate_residual_backward_dxdy(d_out_t[], mv.gate[], ctx)

    # base w2 backward (frozen W): d_x ONLY. int8 path runs ONE dX GEMM on the
    # FULL w2 [D, D+F] (grg.d_y[S,D] @ w2[D,D+F] contract N=D -> dX[S,D+F]) then
    # SLICES cols [0:D]=att-in-dX, [D:D+F]=mlp-in-dX. This is byte-equivalent in
    # layout to the two split bf16 GEMMs: w2_att=w2[:,:D], w2_mlp=w2[:,D:D+F], so
    # dX[:, :D] == grg.d_y @ w2_att and dX[:, D:] == grg.d_y @ w2_mlp. Cast the
    # bf16 int8 result back to grg.d_y's dtype so downstream ops are dtype-identical.
    var d_att: Tensor
    var d_mlp: Tensor
    if int8:
        ref p8 = int8.value()
        var g_bf = cast_tensor(grg.d_y, STDtype.BF16, ctx)   # bf16 at GEMM boundary
        var d_w2 = _i8_bwd_dx(g_bf, p8, 1, ctx)              # [S,D+F] bf16
        if d_w2.dtype() != grg.d_y.dtype():
            d_w2 = cast_tensor(d_w2, grg.d_y.dtype(), ctx)
        d_att = slice(d_w2, 1, 0, D, ctx)
        d_mlp = slice(d_w2, 1, D, F, ctx)
    else:
        d_att = linear_backward_dx_scratch(
            grg.d_y, w.w2_att[], S, D, D, ctx, scratch,
        )
        d_mlp = linear_backward_dx_scratch(
            grg.d_y, w.w2_mlp[], S, F, D, ctx, scratch,
        )

    var out_d_a = List[Float32]()
    var out_d_b = List[Float32]()
    if lora.out:
        var lg2 = _klein_lora_bwd_dropout(
            grg.d_y, saved.out_in[], lora.out.value(), S, drop_out, ctx
        )
        add_in_place_f32(d_att, slice(lg2.d_x, 1, 0, D, ctx), ctx)
        add_in_place_f32(d_mlp, slice(lg2.d_x, 1, D, F, ctx), ctx)
        out_d_a = lg2.d_a.copy()
        out_d_b = lg2.d_b.copy()

    reshape_in_place(d_att, [1, S, H, Dh])

    var sgb = swiglu_backward(d_mlp, saved.mlp_gate[], saved.mlp_up[], ctx)
    var d_gate_up = concat2_scratch(1, ctx, scratch, sgb.d_gate, sgb.d_up)

    var d_q_sb: Tensor
    var d_k_sb: Tensor
    var d_v_sb: Tensor
    comptime if KLEIN_SDPA_FLASH:
        # flash backward from the tape's bf16 saved set (FAIL-LOUD if the
        # forward/recompute path didn't fill it — fwd/bwd flag mismatch).
        if not saved.flash_stats:
            raise Error(
                "single_block bwd: KLEIN_SDPA_FLASH on but saved tape has"
                " no flash stats (forward/backward flag mismatch)"
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

    # base w1 backward (frozen W): d_x ONLY. int8 path CONCATS the two output-row
    # grad blocks (d_qkv[S,3D], d_gate_up[S,2F]) -> grad_full[S,3D+2F] and runs ONE
    # dX GEMM on the FULL w1[3D+2F, D] (contract N=3D+2F -> dX[S,D]) — matching
    # linear_backward_dx_split_scratch which sums the same two row-split contractions.
    var d_norm_t: Tensor
    if int8:
        ref p8 = int8.value()
        var grad_full = concat2_scratch(1, ctx, scratch, d_qkv, d_gate_up)  # [S,3D+2F]
        var grad_full_bf = cast_tensor(grad_full, STDtype.BF16, ctx)   # bf16 at GEMM boundary
        d_norm_t = _i8_bwd_dx(grad_full_bf, p8, 0, ctx)                # [S,D] bf16
        if d_norm_t.dtype() != d_qkv.dtype():
            d_norm_t = cast_tensor(d_norm_t, d_qkv.dtype(), ctx)
    else:
        d_norm_t = linear_backward_dx_split_scratch(
            d_qkv, d_gate_up, w.w1[], S, D, 3 * D, 2 * F, ctx, scratch,
        )

    var d_fused = concat2_scratch(1, ctx, scratch, d_qkv, d_gate_up)
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
            var dlt2 = _klein_lora_fwd_dropout(
                saved.out_in[], lora.out.value(), S, drop_out, ctx
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
        # saved.out_in IS concat(att_flat, mlp) on every constructor path —
        # read the tape instead of re-concatenating (fusion pass 2).
        var lg2 = _klein_lora_bwd_dropout_tensors(
            grg.d_y, saved.out_in[], lora.out.value(), S, drop_out, ctx
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

    var d_fused = concat2_scratch(1, ctx, scratch, d_qkv, d_gate_up)
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


# ── KLEIN_RESIDENT_GRADS device-grads backward (2026-07-11) ──────────────────
# EXACT math mirror of `single_block_lora_backward_device_resident_scratch`
# (the product tape path's block backward: flash SDPA backward under
# KLEIN_SDPA_FLASH — same arm as the `_tensors` graph variant above — plus the
# int8 dX GEMMs), with ONE data-movement change: the LoRA d_A/d_B leave as
# DEVICE F32 tensors (`_klein_lora_bwd_dropout_tensors`) instead of host
# lists. Same tensors, same kernels, same values — no per-block D2H.
def single_block_lora_backward_device_resident_scratch_devgrads[
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
    int8: Optional[SingleBlockInt8] = None,
) raises -> SingleBlockLoraDeviceGradTensors:
    var scratch_mark = scratch.mark()
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var grg: GateResidualGrads
    var d_gate = List[Float32]()
    if compute_aux_grads:
        # Aux recompute of out_y (== forward out projection). Only d_gate depends
        # on out_y's VALUE (grg.d_x/d_y do not), so the int8 quant here shifts
        # d_gate to match the int8 forward while leaving the base dX path exact.
        var out_y: Tensor
        if int8:
            # saved.out_in IS concat(att_flat, mlp) — read the tape (pass 2).
            out_y = _base_fwd_i8(saved.out_in[], w.w2[], int8, 1, ctx)   # full-w2 int8
        else:
            out_y = linear_two_inputs_scratch(
                saved.att_flat[], saved.mlp[], w.w2_att[], w.w2_mlp[], ctx, scratch,
            )
        if lora.out:
            var dlt2 = _klein_lora_fwd_dropout(
                saved.out_in[], lora.out.value(), S, drop_out, ctx
            )
            add_in_place_f32(out_y, dlt2, ctx)
        grg = gate_residual_backward(
            d_out_t[], saved.x[], mv.gate[], out_y, ctx
        )
        d_gate = grg.d_g.to_host(ctx)
    else:
        grg = gate_residual_backward_dxdy(d_out_t[], mv.gate[], ctx)

    # base w2 backward (frozen W): d_x ONLY — int8/bf16 arms byte-identical to
    # the host-grads variant above (see its comment for the slicing proof).
    var d_att: Tensor
    var d_mlp: Tensor
    if int8 and grg.d_y.dtype() == STDtype.F32:
        # FUSED F32 boundary + band-split (2026-07-11): quant reads the F32
        # grad directly and the dequant writes the [d_att|d_mlp] bands —
        # bit-identical to cast→bwd_nn→cast→2×slice, minus 4 kernels.
        ref p8f = int8.value()
        var dbands = _i8_bwd_dx_f32_bands(
            grg.d_y, p8f, 1, D, F, 0, 0, ctx,
        )
        d_att = int8_band_tensor(dbands[0])
        d_mlp = int8_band_tensor(dbands[1])
    elif int8:
        ref p8 = int8.value()
        var g_bf = cast_tensor(grg.d_y, STDtype.BF16, ctx)   # bf16 at GEMM boundary
        var d_w2 = _i8_bwd_dx(g_bf, p8, 1, ctx)              # [S,D+F] bf16
        if d_w2.dtype() != grg.d_y.dtype():
            d_w2 = cast_tensor(d_w2, grg.d_y.dtype(), ctx)
        d_att = slice(d_w2, 1, 0, D, ctx)
        d_mlp = slice(d_w2, 1, D, F, ctx)
    else:
        d_att = linear_backward_dx_scratch(
            grg.d_y, w.w2_att[], S, D, D, ctx, scratch,
        )
        d_mlp = linear_backward_dx_scratch(
            grg.d_y, w.w2_mlp[], S, F, D, ctx, scratch,
        )

    var out_d_a = Optional[TArc](None)
    var out_d_b = Optional[TArc](None)
    if lora.out:
        # saved.out_in IS concat(att_flat, mlp) on every constructor path —
        # read the tape instead of re-concatenating (fusion pass 2).
        var lg2 = _klein_lora_bwd_dropout_tensors(
            grg.d_y, saved.out_in[], lora.out.value(), S, drop_out, ctx
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
        # forward/recompute path didn't fill it — fwd/bwd flag mismatch).
        if not saved.flash_stats:
            raise Error(
                "single_block devgrads bwd: KLEIN_SDPA_FLASH on but saved tape"
                " has no flash stats (forward/backward flag mismatch)"
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

    # ONE [S,3D+2F] concat (fusion pass 2): both the int8 dX GEMM and the LoRA
    # qkv backward consume concat(d_qkv, d_gate_up) — build it once instead of
    # twice (identical inputs → identical bytes).
    var d_fused = concat2_scratch(1, ctx, scratch, d_qkv, d_gate_up)

    # base w1 backward (frozen W): d_x ONLY — int8/bf16 arms byte-identical to
    # the host-grads variant above.
    var d_norm_t: Tensor
    if int8 and d_qkv.dtype() == STDtype.F32 and d_gate_up.dtype() == STDtype.F32:
        # FUSED F32 boundary (2026-07-11): bit-identical to cast→bwd_nn→cast.
        ref p8f = int8.value()
        d_norm_t = _i8_bwd_dx_f32(d_fused, p8f, 0, ctx)
    elif int8:
        ref p8 = int8.value()
        var grad_full_bf = cast_tensor(d_fused, STDtype.BF16, ctx)   # bf16 at GEMM boundary
        d_norm_t = _i8_bwd_dx(grad_full_bf, p8, 0, ctx)              # [S,D] bf16
        if d_norm_t.dtype() != d_qkv.dtype():
            d_norm_t = cast_tensor(d_norm_t, d_qkv.dtype(), ctx)
    else:
        d_norm_t = linear_backward_dx_split_scratch(
            d_qkv, d_gate_up, w.w1[], S, D, 3 * D, 2 * F, ctx, scratch,
        )
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


# ═══════════════════════════════════════════════════════════════════════════
# FULL-FINETUNE block backward (klein full-FT rollout item 2, 2026-07-07)
#
# Same hand-chain as the LoRA device-resident backward above, but the base
# matmul weights are TRAINABLE: each projection site emits d_W = d_yᵀ @ x_saved
# via ops/linalg_backward.linear_backward_dw with EXPLICIT STDtype.F32 output
# (the BOOL default sentinel means "match input dtype" -> bf16 grads the F32
# optimizer rejects — krea2 trap), alongside the plain d_x carry
# (linear_backward_dx). NO LoRA anywhere, NO int8 (reference trainer full-FT forbids
# quantized linears — Flux2FineTuneSetup trains bf16 weights).
#
# v1 trainable surface = the 2 fused matmuls (w1 [3D+2F,D], w2 [D,D+F]).
# FROZEN (documented delta vs reference trainer, which trains all transformer params):
#   q_norm/k_norm rms scales, the modulation vectors (shift/scale/gate), and
#   the LN (non-learnable by construction). Their grads are SKIPPED
#   (gate_residual_backward_dxdy / modulate_backward(compute_param_grads=False))
#   — extending the surface = more dW arms + host-store entries (P-later).
#
# dX fold order (C15) matches single_block_lora_backward_device_resident
# EXACTLY: d_out_in via w2-dx -> cat_backward(D,F) -> swiglu/sdpa arms ->
# d_qkv concat(q,k,v) -> d_fused concat(d_qkv, d_gate_up) -> w1-dx -> modulate
# -> layer_norm -> d_x = add(residual_branch, norm_branch).
#
# SDPA arm: v1 consumes the always-saved math tape (saved.q_rope/k_rope/v via
# sdpa_backward) — every fwd variant fills those, including the flash
# recompute. A flash-tape arm (saved.flash_*) is P3 trainer wiring; adding it
# untested here would be an ungated branch.
#
# Oracle: single_block_oracle.py is base-only (no LoRA) and dumps ref_d_w1 /
# ref_d_w2 (torch W.grad) + ref_d_x — the FT gate
# (parity/single_block_ft_parity.mojo) compares both at cos >= 0.999.
# ═══════════════════════════════════════════════════════════════════════════
struct SingleBlockFTGrads(Movable):
    var d_x: TArc          # [S,D] input grad (the inter-block carry)
    var dw: List[TArc]     # len 2, F32 device: [0]=d_w1 [3D+2F,D], [1]=d_w2 [D,D+F]
    # SURF q/k rms-scale grads (F32 [Dh]): [0]=d_qnorm, [1]=d_knorm. EMPTY off.
    var d_qk: List[TArc]
    # SURF SHARED single_stream_modulation.lin FLAT grad (F32 [1,3D]), packed
    #   [shift|scale|gate] (the single_stream_modulation.lin output layout). The
    #   STACK sums this across single blocks then matmuls silu(vec). EMPTY off.
    var mod_flat: List[TArc]

    def __init__(
        out self, var d_x: TArc, var dw: List[TArc],
        var d_qk: List[TArc], var mod_flat: List[TArc],
    ):
        self.d_x = d_x^
        self.dw = dw^
        self.d_qk = d_qk^
        self.mod_flat = mod_flat^


def single_block_ft_backward_dev[
    H: Int, Dh: Int, S: Int, SURF: Bool = False
](
    d_out_t: TArc,
    w: SingleBlockWeights, mv: SingleModVecsDevice,
    saved: SingleBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> SingleBlockFTGrads:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t_dtype(_ones(D), [D], saved.x[].dtype(), ctx)
    var modp = List[TArc]()   # SURF: [0]=d_gate, [1]=d_shift, [2]=d_scale (F32)
    var d_qk = List[TArc]()

    # result = residual_gate(x, gate, out); mod grads frozen (v1) OR trainable.
    var grg = gate_residual_backward_dxdy(d_out_t[], mv.gate[], ctx)

    comptime if SURF:
        # d_gate = sum_rows d_out * out; out = linear(out_in, W2) not saved ->
        # recompute (no bias). FULL gate_residual_backward d_g (d_x/d_y match the
        # dxdy chain above so the dX path is byte-unchanged).
        var nb_out = Optional[Tensor](None)
        var out_y = linear(saved.out_in[], w.w2[], nb_out^, ctx)
        var grgf = gate_residual_backward(d_out_t[], saved.x[], mv.gate[], out_y, ctx, True)
        modp.append(TArc(cast_tensor(grgf.d_g, STDtype.F32, ctx)))   # [0] d_gate

    # out = linear(out_in, W2): TRAINABLE -> d_W (F32) + d_x carry.
    var dw_w2 = linear_backward_dw(
        grg.d_y, saved.out_in[], S, D + F, D, ctx, STDtype.F32,
    )
    var d_out_in_t = linear_backward_dx(grg.d_y, w.w2[], S, D + F, D, ctx)

    # out_in = concat(att_flat, mlp) on the CHANNEL axis (sizes D, F).
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

    # d_x arm: keep the v1 dx-only rms backward BYTE-UNCHANGED. Under SURF add
    # the weight-free d_g arm (rms_norm_backward_dg, F32) — d_x stays identical.
    var d_q_pre_t = rms_norm_backward_dx(d_q_rms, saved.q_pre[], w.q_norm[], eps, ctx)
    var d_k_pre_t = rms_norm_backward_dx(d_k_rms, saved.k_pre[], w.k_norm[], eps, ctx)
    comptime if SURF:
        d_qk.append(TArc(rms_norm_backward_dg(d_q_rms, saved.q_pre[], eps, ctx)))   # [0] d_qnorm
        d_qk.append(TArc(rms_norm_backward_dg(d_k_rms, saved.k_pre[], eps, ctx)))   # [1] d_knorm

    var d_q_pre_flat = reshape_owned(d_q_pre_t^, [S, D])
    var d_k_pre_flat = reshape_owned(d_k_pre_t^, [S, D])
    reshape_in_place(sb.d_v, [S, D])
    var d_qkv = concat(1, ctx, d_q_pre_flat, d_k_pre_flat, sb.d_v)   # [S,3D]

    var d_fused = concat(1, ctx, d_qkv, d_gate_up)   # [S, 3D+2F]

    # fused = linear(norm, W1): TRAINABLE -> d_W (F32) + d_x carry.
    var dw_w1 = linear_backward_dw(
        d_fused, saved.norm[], S, D, 3 * D + 2 * F, ctx, STDtype.F32,
    )
    var d_norm_t = linear_backward_dx(d_fused, w.w1[], S, D, 3 * D + 2 * F, ctx)

    # norm = modulate(ln, scale, shift); mod grads frozen (v1) OR trainable.
    var mb = modulate_backward(d_norm_t, saved.ln[], mv.scale[], ctx, SURF)
    comptime if SURF:
        modp.append(TArc(cast_tensor(mb.d_shift, STDtype.F32, ctx)))   # [1] d_shift
        modp.append(TArc(cast_tensor(mb.d_scale, STDtype.F32, ctx)))   # [2] d_scale
    var d_x_norm_t = layer_norm_backward_dx(mb.d_x, saved.x[], ones_t, eps, ctx)

    # x feeds BOTH the residual (grg.d_x) AND layer_norm(x) -> SUM (C15 order).
    var d_x_t = add(grg.d_x, d_x_norm_t, ctx)

    var dw = List[TArc]()
    dw.append(TArc(dw_w1^))
    dw.append(TArc(dw_w2^))

    var mod_flat = List[TArc]()
    comptime if SURF:
        # single_stream_modulation.lin output layout [shift|scale|gate]; modp =
        # [d_gate, d_shift, d_scale].
        var m_flat = concat(0, ctx, modp[1][], modp[2][], modp[0][])
        reshape_in_place(m_flat, [1, 3 * D])
        mod_flat.append(TArc(m_flat^))
    return SingleBlockFTGrads(TArc(d_x_t^), dw^, d_qk^, mod_flat^)


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

    var ln_t = lnmod_placeholder(ctx)
    var norm_t = layer_norm_modulate(x_t[], mv.scale[], mv.shift[], eps, ctx, ln_t)

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

    var ln_t = lnmod_placeholder(ctx)
    var norm_t = layer_norm_modulate(x_t[], mv.scale[], mv.shift[], eps, ctx, ln_t)

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


# ── TRUE-batch row-stacked (b2rs) single block ───────────────────────────────
# Row-stacked batch>1 pair (fleet TRUE-batching, klein rung: replace the
# interleaved b2 that computes every block TWICE per pair with ONE pass over
# [B*S, D] rows):
#   x            [B*S, D]   sample rows contiguous (sample 0 first)
#   mv           SingleModVecsDevice whose shift/scale/gate are [B, D] stacked
#                per-sample vecs (modulate/residual_gate row-block broadcast)
#   attention    REAL batch-B cuDNN flash (stats saved on the tape) —
#                samples cannot cross-attend; math sdpa_nomask[B] fallback
#   LoRA         GEMMs run at M=B*S, so shared-adapter d_A/d_B sum over the
#                batch inside the GEMM (equivalent to summed per-sample grads)
#   cos/sin      the B-stacked rope tables [B*S*H, Dh/2] (concat of the b1
#                table with itself when the samples share positions)
# AdaLN aux grads (shift/scale/gate) are NOT produced: the [B,D] kernels do
# not reduce per-vec param grads (LoRA training discards them — frozen adaLN).


def single_block_lora_forward_device_resident_scratch_batch[
    B: Int, H: Int, Dh: Int, S: Int
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
    comptime ROWS = B * S
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var ln_t = lnmod_placeholder(ctx)
    var norm_t = layer_norm_modulate(x_t[], mv.scale[], mv.shift[], eps, ctx, ln_t)

    var scratch_mark = scratch.mark()
    var q_pre_flat = linear_rows(norm_t, w.w1[], 0, D, ctx)
    var k_pre_flat = linear_rows(norm_t, w.w1[], D, D, ctx)
    var v_flat = linear_rows(norm_t, w.w1[], 2 * D, D, ctx)
    var gate_up = linear_rows_scratch(norm_t, w.w1[], 3 * D, 2 * F, ctx, scratch)
    if lora.qkv:
        var dlt = _klein_lora_fwd_dropout(
            norm_t, lora.qkv.value(), ROWS, drop_qkv, ctx
        )
        add_in_place_f32(q_pre_flat, slice(dlt, 1, 0, D, ctx), ctx)
        add_in_place_f32(k_pre_flat, slice(dlt, 1, D, D, ctx), ctx)
        add_in_place_f32(v_flat, slice(dlt, 1, 2 * D, D, ctx), ctx)
        add_in_place_f32(gate_up, slice(dlt, 1, 3 * D, 2 * F, ctx), ctx)
    var q_pre = reshape_owned(q_pre_flat^, [B, S, H, Dh])
    var k_pre = reshape_owned(k_pre_flat^, [B, S, H, Dh])
    var v = reshape_owned(v_flat^, [B, S, H, Dh])

    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)

    var q_rope = rope_interleaved(q_rms, cos, sin, ctx)
    var k_rope = rope_interleaved(k_rms, cos, sin, ctx)

    var att_flat: Tensor
    var flash_q = Optional[TArc](None)
    var flash_k = Optional[TArc](None)
    var flash_v = Optional[TArc](None)
    var flash_o = Optional[TArc](None)
    var flash_stats = Optional[TArc](None)
    comptime if KLEIN_SDPA_FLASH:
        var ff = sdpa_flash_train_fwd_f32[B, S, H, Dh](q_rope, k_rope, v, scale, ctx)
        var af_shape: List[Int] = [ROWS, D]
        att_flat = Tensor(ff.att.buf.copy(), af_shape^, STDtype.F32)
        flash_q = Optional[TArc](ff.q_bf.copy())
        flash_k = Optional[TArc](ff.k_bf.copy())
        flash_v = Optional[TArc](ff.v_bf.copy())
        flash_o = Optional[TArc](ff.o_bf.copy())
        flash_stats = Optional[TArc](ff.stats.copy())
    else:
        var att = sdpa_nomask[B, S, H, Dh](q_rope, k_rope, v, scale, ctx)
        att_flat = reshape_owned(att^, [ROWS, D])

    var mlp_gate = slice(gate_up, 1, 0, F, ctx)
    var mlp_up = slice(gate_up, 1, F, F, ctx)
    var mlp = swiglu(mlp_gate, mlp_up, ctx)
    if att_flat.dtype() != mlp.dtype():
        att_flat = cast_tensor(att_flat, mlp.dtype(), ctx)
    scratch.rewind(scratch_mark)

    var out_in_t = concat(1, ctx, att_flat, mlp)
    var proj_mark = scratch.mark()
    var out_proj = linear_two_inputs_scratch(
        att_flat, mlp, w.w2_att[], w.w2_mlp[], ctx, scratch,
    )
    if lora.out:
        var dlt2 = _klein_lora_fwd_dropout(
            out_in_t, lora.out.value(), ROWS, drop_out, ctx
        )
        add_in_place_f32(out_proj, dlt2, ctx)

    var result = residual_gate(x_t[], mv.gate[], out_proj, ctx)
    scratch.rewind(proj_mark)

    var saved = SingleBlockSaved(
        x_t.copy(), TArc(ln_t^), TArc(norm_t^), TArc(q_pre^), TArc(k_pre^),
        TArc(q_rms^), TArc(k_rms^), TArc(v^),
        TArc(q_rope^), TArc(k_rope^), TArc(att_flat^),
        TArc(mlp_gate^), TArc(mlp_up^), TArc(mlp^), TArc(out_in_t^),
        flash_q^, flash_k^, flash_v^, flash_o^, flash_stats^,
    )
    return SingleBlockDeviceForward(TArc(result^), saved^)


def single_block_lora_backward_device_resident_scratch_tensors_batch[
    B: Int, H: Int, Dh: Int, S: Int
](
    d_out_t: TArc,
    w: SingleBlockWeights, mv: SingleModVecsDevice, lora: SingleBlockLoraDevice,
    saved: SingleBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    norm_ones: Tensor,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
    drop_qkv: LoraDropout = LoraDropout(),
    drop_out: LoraDropout = LoraDropout(),
) raises -> SingleBlockLoraDeviceGradTensors:
    # No compute_aux_grads arm: adaLN shift/scale/gate grads are per-vec
    # reductions the [B,D] kernels do not produce (frozen in LoRA training).
    comptime ROWS = B * S
    var scratch_mark = scratch.mark()
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var grg = gate_residual_backward_dxdy(d_out_t[], mv.gate[], ctx)
    var d_gate = List[Float32]()

    var d_att = linear_backward_dx_scratch(
        grg.d_y, w.w2_att[], ROWS, D, D, ctx, scratch,
    )
    var d_mlp = linear_backward_dx_scratch(
        grg.d_y, w.w2_mlp[], ROWS, F, D, ctx, scratch,
    )

    var out_d_a = Optional[TArc](None)
    var out_d_b = Optional[TArc](None)
    if lora.out:
        var lg2 = _klein_lora_bwd_dropout_tensors(
            grg.d_y, saved.out_in[], lora.out.value(), ROWS, drop_out, ctx
        )
        add_in_place_f32(d_att, slice(lg2.d_x[], 1, 0, D, ctx), ctx)
        add_in_place_f32(d_mlp, slice(lg2.d_x[], 1, D, F, ctx), ctx)
        out_d_a = Optional[TArc](lg2.d_a.copy())
        out_d_b = Optional[TArc](lg2.d_b.copy())

    reshape_in_place(d_att, [B, S, H, Dh])

    var sgb = swiglu_backward(d_mlp, saved.mlp_gate[], saved.mlp_up[], ctx)
    var d_gate_up = concat2_scratch(1, ctx, scratch, sgb.d_gate, sgb.d_up)

    var d_q_sb: Tensor
    var d_k_sb: Tensor
    var d_v_sb: Tensor
    comptime if KLEIN_SDPA_FLASH and KLEIN_B2RS_FLASH_BWD:
        if not saved.flash_stats:
            raise Error(
                "single_block b2rs bwd: KLEIN_SDPA_FLASH on but saved tape"
                " has no flash stats (pair with the _batch forward)"
            )
        var fb = sdpa_flash_backward_f32[B, S, H, Dh](
            saved.flash_q.value(), saved.flash_k.value(),
            saved.flash_v.value(), saved.flash_o.value(),
            saved.flash_stats.value(), d_att, scale, ctx,
        )
        d_q_sb = Tensor(fb.d_q.buf.copy(), fb.d_q.shape(), fb.d_q.dtype())
        d_k_sb = Tensor(fb.d_k.buf.copy(), fb.d_k.shape(), fb.d_k.dtype())
        d_v_sb = Tensor(fb.d_v.buf.copy(), fb.d_v.shape(), fb.d_v.dtype())
    else:
        var sb = sdpa_backward_scratch[B, S, H, Dh](
            saved.q_rope[], saved.k_rope[], saved.v[], d_att, scale, ctx, scratch,
        )
        d_q_sb = Tensor(sb.d_q.buf.copy(), sb.d_q.shape(), sb.d_q.dtype())
        d_k_sb = Tensor(sb.d_k.buf.copy(), sb.d_k.shape(), sb.d_k.dtype())
        d_v_sb = Tensor(sb.d_v.buf.copy(), sb.d_v.shape(), sb.d_v.dtype())

    var d_q_rms = rope_backward(d_q_sb, cos, sin, True, ctx)
    var d_k_rms = rope_backward(d_k_sb, cos, sin, True, ctx)

    var d_q_pre_t = rms_norm_backward_dx(d_q_rms, saved.q_pre[], w.q_norm[], eps, ctx)
    var d_k_pre_t = rms_norm_backward_dx(d_k_rms, saved.k_pre[], w.k_norm[], eps, ctx)

    var d_q_pre_flat = reshape_owned(d_q_pre_t^, [ROWS, D])
    var d_k_pre_flat = reshape_owned(d_k_pre_t^, [ROWS, D])
    reshape_in_place(d_v_sb, [ROWS, D])
    var d_qkv = concat3_scratch(1, ctx, scratch, d_q_pre_flat, d_k_pre_flat, d_v_sb, True)

    var d_norm_t = linear_backward_dx_split_scratch(
        d_qkv, d_gate_up, w.w1[], ROWS, D, 3 * D, 2 * F, ctx, scratch,
    )

    var d_fused = concat2_scratch(1, ctx, scratch, d_qkv, d_gate_up)
    var qkv_d_a = Optional[TArc](None)
    var qkv_d_b = Optional[TArc](None)
    if lora.qkv:
        var lg = _klein_lora_bwd_dropout_tensors(
            d_fused, saved.norm[], lora.qkv.value(), ROWS, drop_qkv, ctx
        )
        d_norm_t = add(d_norm_t, lg.d_x[], ctx)
        qkv_d_a = Optional[TArc](lg.d_a.copy())
        qkv_d_b = Optional[TArc](lg.d_b.copy())

    var mb = modulate_backward(d_norm_t, saved.ln[], mv.scale[], ctx, False)
    var d_scale = List[Float32]()
    var d_shift = List[Float32]()

    var d_x_norm_t = layer_norm_backward_dx(mb.d_x, saved.x[], norm_ones, eps, ctx)
    var d_x_t = add(grg.d_x, d_x_norm_t, ctx)

    var out = SingleBlockLoraDeviceGradTensors(
        TArc(d_x_t^), d_shift^, d_scale^, d_gate^,
        qkv_d_a^, qkv_d_b^, out_d_a^, out_d_b^,
    )
    scratch.rewind(scratch_mark)
    return out^


def single_block_lora_recompute_saved_device_resident_scratch_batch[
    B: Int, H: Int, Dh: Int, S: Int
](
    x_t: TArc,
    w: SingleBlockWeights, mv: SingleModVecsDevice, lora: SingleBlockLoraDevice,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    norm_ones: Tensor, norm_zeros: Tensor,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> SingleBlockSaved:
    """Batched LEAN recompute (b2rs backward): rebuilds ONLY the tape fields —
    no out-projection GEMM, no residual, no block output. Mirror of
    single_block_lora_recompute_saved_device_resident_scratch at B rows."""
    comptime ROWS = B * S
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var ln_t = lnmod_placeholder(ctx)
    var norm_t = layer_norm_modulate(x_t[], mv.scale[], mv.shift[], eps, ctx, ln_t)

    var scratch_mark = scratch.mark()
    var q_pre_flat = linear_rows(norm_t, w.w1[], 0, D, ctx)
    var k_pre_flat = linear_rows(norm_t, w.w1[], D, D, ctx)
    var v_flat = linear_rows(norm_t, w.w1[], 2 * D, D, ctx)
    var gate_up = linear_rows_scratch(norm_t, w.w1[], 3 * D, 2 * F, ctx, scratch)
    if lora.qkv:
        var dlt = klein_lora_fwd_device_resident(norm_t, lora.qkv.value(), ROWS, ctx)
        add_in_place_f32(q_pre_flat, slice(dlt, 1, 0, D, ctx), ctx)
        add_in_place_f32(k_pre_flat, slice(dlt, 1, D, D, ctx), ctx)
        add_in_place_f32(v_flat, slice(dlt, 1, 2 * D, D, ctx), ctx)
        add_in_place_f32(gate_up, slice(dlt, 1, 3 * D, 2 * F, ctx), ctx)
    var q_pre = reshape_owned(q_pre_flat^, [B, S, H, Dh])
    var k_pre = reshape_owned(k_pre_flat^, [B, S, H, Dh])
    var v = reshape_owned(v_flat^, [B, S, H, Dh])

    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)

    var q_rope = rope_interleaved(q_rms, cos, sin, ctx)
    var k_rope = rope_interleaved(k_rms, cos, sin, ctx)
    comptime if KLEIN_SDPA_FLASH:
        var ff = sdpa_flash_train_fwd_f32[B, S, H, Dh](q_rope, k_rope, v, scale, ctx)
        var af_shape: List[Int] = [ROWS, D]
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
        var att = sdpa_nomask[B, S, H, Dh](q_rope, k_rope, v, scale, ctx)
        var att_flat = reshape_owned(att^, [ROWS, D])

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


# ══════════════════════════════════════════════════════════════════════════════
# int8 W8A8 QUANTIZED-RESIDENT BASE PATH (slice 1: single block forward only)
#
# MIRRORS krea2's PROVEN int8 pattern (models/krea2/krea2_block.mojo:251 payload,
# krea2_stack.mojo:671 builder, :187 `_base_fwd` dispatch). Klein's single block
# has TWO frozen base matmuls (vs krea2's 8):
#   idx 0 = w1 [3D+2F, D]  (to_qkv_mlp_proj / "linear1") — contracts K=D
#   idx 1 = w2 [D,   D+F]  (to_out          / "linear2") — contracts K=D+F
# Each is quantized TENSORWISE (one scalar F32 scale) ONCE at load, held int8
# resident. Per step the base runs int8×int8→int32 GEMM with per-token activation
# quant (int8_linear_fwd) — NO per-step dequant. The LoRA adapters stay bf16 and
# UNCHANGED (only the frozen base matmul becomes int8).
#
# KLEIN-vs-krea2 STRUCTURAL DIFFERENCE (needed judgment):
#   krea2's block runs bf16 activations end-to-end, so `_base_fwd` returns the
#   int8 GEMM's bf16 output straight into the bf16 chain. Klein's trainer runs
#   F32 activations with BF16 weights (mixed GEMM — klein_stack_lora.mojo:473-475),
#   so `_base_fwd_i8` here casts the F32 activation → BF16 at the matmul boundary
#   (== reference trainer's bf16 activations, exactly krea2's cast) AND casts the bf16 GEMM
#   output back to the activation dtype so the REST of the block stays
#   dtype-identical to the bf16 base path (the int8 quant error is already baked
#   into the bf16 output; bf16→F32 is lossless). This keeps the surrounding
#   graph bit-identical between the bf16 and int8 forwards, isolating the int8
#   quant error at the two base matmuls.
# ══════════════════════════════════════════════════════════════════════════════

# int8 W8A8 payload for ONE Klein single block's 2 frozen base matmul weights.
# Mirror of krea2_block.mojo's Krea2BlockInt8. Field/list order = base-matmul
# index: [0]=w1 (to_qkv_mlp_proj), [1]=w2 (to_out). Per weight i:
#   w8[i]    = int8 [N,K] (fwd orientation; fwd contracts K),
#   w8t[i]   = int8 [K,N] PRE-TRANSPOSED copy for the backward's dX NT GEMM
#              (SAME int8 values, transposed ONCE at quantize time — the NN
#              layout ran ~226 TOPS forwardCompat kernels, NT runs ~450 TOPS
#              i16832; dX is BIT-IDENTICAL, exact int32 accumulate). May be
#              EMPTY (len 0) → the bwd falls back to the NN path unchanged.
#   scale[i] = F32 scalar tensorwise scale [1].
# Base weight is FROZEN → no weight grad; only the bf16 LoRA adapters train (==reference trainer).
struct SingleBlockInt8(Copyable, Movable):
    var w8: List[TArc]     # len 2: int8 [N,K] (w1, w2)
    var scale: List[TArc]  # len 2: F32 scalar [1]
    var w8t: List[TArc]    # len 2 (or 0 = NN fallback): int8 [K,N]

    def __init__(out self, var w8: List[TArc], var scale: List[TArc]):
        self.w8 = w8^
        self.scale = scale^
        self.w8t = List[TArc]()

    def __init__(
        out self, var w8: List[TArc], var scale: List[TArc], var w8t: List[TArc]
    ):
        self.w8 = w8^
        self.scale = scale^
        self.w8t = w8t^


# Builder: quantize the two frozen bf16 base weights TENSORWISE → int8 payload,
# ONCE at load. Mirror of krea2_stack.mojo `_quantize_wb_int8`
# (int8_tensorwise_scale → int8_encode_tensorwise). w1_bf/w2_bf are BF16 [N,K].
# Also pre-transposes each int8 weight ONCE ([N,K] → [K,N], same values) so the
# backward's dX GEMMs run the fast NT entry point (bit-identical dX).
def quantize_single_block_int8(
    w1_bf: Tensor, w2_bf: Tensor, ctx: DeviceContext
) raises -> SingleBlockInt8:
    var s1 = int8_tensorwise_scale(w1_bf, ctx)          # F32 [1]
    var q1 = int8_encode_tensorwise(w1_bf, s1, ctx)     # I8 [3D+2F, D]
    var s2 = int8_tensorwise_scale(w2_bf, ctx)          # F32 [1]
    var q2 = int8_encode_tensorwise(w2_bf, s2, ctx)     # I8 [D, D+F]
    var w8t = List[TArc]()
    w8t.append(TArc(int8_transpose(q1, ctx)))           # I8 [D, 3D+2F]
    w8t.append(TArc(int8_transpose(q2, ctx)))           # I8 [D+F, D]
    var w8 = List[TArc]()
    w8.append(TArc(q1^))
    w8.append(TArc(q2^))
    var scale = List[TArc]()
    scale.append(TArc(s1^))
    scale.append(TArc(s2^))
    return SingleBlockInt8(w8^, scale^, w8t^)


# ── bwd dX dispatch (Klein bwd NT lever, 2026-07-11): the NN stored-orientation
# GEMM lands on ~226 TOPS forwardCompat wmma kernels on sm_120; the NT layout
# runs ~450 TOPS i16832 (nsys census: 195 ms/step pool). Route every dX through
# NT: use the payload's pre-transposed W8T when present (block-direct quantize /
# parity drivers), else TRANSPOSE-ON-VISIT — one vectorized int8 [N,K]→[K,N]
# device transpose per weight per bwd block visit (MEASURED 21.3 ms/step total,
# n=112), reading the already-staged W8 (NO extra PCIe). The staged-host-W8T design was
# MEASURED WORSE: doubling the per-block H2D (290→580 MB) left the bwd walk
# copy-bound (GPU idle 3.4%→26.8%, step 0.82→0.90 s). dX is BIT-IDENTICAL in
# every arm (same int8 values, exact int32 accumulate; gate:
# ops/tests/int8_linear_parity.mojo max_abs 0.0). ─────────────────────────────
def _i8_bwd_dx(
    g_bf: Tensor, p8: SingleBlockInt8, idx: Int, ctx: DeviceContext
) raises -> Tensor:
    if len(p8.w8t) == len(p8.w8):
        return int8_linear_bwd(g_bf, p8.w8t[idx][], p8.scale[idx][], ctx)
    var w8t = int8_transpose(p8.w8[idx][], ctx)
    return int8_linear_bwd(g_bf, w8t, p8.scale[idx][], ctx)


def _i8_bwd_dx_f32(
    g: Tensor, p8: SingleBlockInt8, idx: Int, ctx: DeviceContext
) raises -> Tensor:
    if len(p8.w8t) == len(p8.w8):
        return int8_linear_bwd_nt_f32(g, p8.w8t[idx][], p8.scale[idx][], ctx)
    var w8t = int8_transpose(p8.w8[idx][], ctx)
    return int8_linear_bwd_nt_f32(g, w8t, p8.scale[idx][], ctx)


def _i8_bwd_dx_f32_bands(
    g: Tensor, p8: SingleBlockInt8, idx: Int,
    w0: Int, w1: Int, w2: Int, w3: Int, ctx: DeviceContext,
) raises -> List[TArc]:
    if len(p8.w8t) == len(p8.w8):
        return int8_linear_bwd_nt_f32_bands(
            g, p8.w8t[idx][], p8.scale[idx][], w0, w1, w2, w3, ctx
        )
    var w8t = int8_transpose(p8.w8[idx][], ctx)
    return int8_linear_bwd_nt_f32_bands(
        g, w8t, p8.scale[idx][], w0, w1, w2, w3, ctx
    )


# Frozen base forward y = x @ W[idx]ᵀ (no bias). int8 W8A8 when the payload is
# present, else the unchanged bf16 `linear`. Mirror of krea2_block `_base_fwd`,
# adapted for Klein's F32-activation chain (cast bf16 at the GEMM, cast the bf16
# result back to x's dtype — see the header note).
def _base_fwd_i8(
    x: Tensor, w_bf: Tensor, int8: Optional[SingleBlockInt8], idx: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    if int8:
        ref p = int8.value()
        var o: Tensor
        if x.dtype() == STDtype.BF16:
            o = int8_linear_fwd(x, p.w8[idx][], p.scale[idx][], ctx)
        elif x.dtype() == STDtype.F32:
            # FUSED F32 boundary (2026-07-11): bit-identical to
            # cast(x,BF16) → int8_linear_fwd → cast(out,F32), minus 2 kernels.
            return int8_linear_fwd_f32(x, p.w8[idx][], p.scale[idx][], ctx)
        else:
            var xb = cast_tensor(x, STDtype.BF16, ctx)
            o = int8_linear_fwd(xb, p.w8[idx][], p.scale[idx][], ctx)
        if o.dtype() != x.dtype():
            return cast_tensor(o, x.dtype(), ctx)
        return o^
    var nb = Optional[Tensor](None)
    return linear(x, w_bf, nb^, ctx)


# ══════════════════════════════════════════════════════════════════════════════
# SquareQ NVFP4 NATIVE-FP4 BASE PATH (chunk 8: forward dispatch only)
#
# MIRRORS the SingleBlockInt8 payload pattern above. Per base-matmul slot
# ([0]=w1 to_qkv_mlp_proj, [1]=w2 to_out — the SingleBlockInt8 index order):
#   nvq[i] = U8 [out, in/2]  e2m1 codes of the rotated residual,
#   nvs[i] = U8 tiled ue4m3 weight scales (cuBLASLt layout),
#   ld[i]/lu[i] = BF16 [in,R]/[out,R] low-rank factors (BASE quantization
#                 correction, NOT the trained LoRA),
#   nvg[i] = F32 per-tensor global scale (HOST scalar — rides cublasLt alpha;
#            read once at pin time, threaded through the block builders).
# FORWARD-ONLY: the backward keeps the reconstructed BF16 W_hat that the
# nvfp4-resident loader placed in the Block under the weight's own name
# (squareq_nvfp4_reconstruct_weight) — a disclosed numerics-mismatch STE
# (ops/squareq_nvfp4.mojo:15). Base weight is FROZEN → no weight grad; only
# the bf16 LoRA adapters train, exactly like the int8 path.
# ══════════════════════════════════════════════════════════════════════════════
struct SingleBlockNvfp4(Copyable, Movable):
    var nvq: List[TArc]     # len 2: U8 [out, in/2] (w1, w2)
    var nvs: List[TArc]     # len 2: U8 tiled ue4m3 scales
    var ld: List[TArc]      # len 2: BF16 [in, R]
    var lu: List[TArc]      # len 2: BF16 [out, R]
    var nvg: List[Float32]  # len 2: per-tensor global scale (host)

    def __init__(
        out self,
        var nvq: List[TArc], var nvs: List[TArc],
        var ld: List[TArc], var lu: List[TArc], var nvg: List[Float32],
    ):
        self.nvq = nvq^
        self.nvs = nvs^
        self.ld = ld^
        self.lu = lu^
        self.nvg = nvg^


# Frozen base forward y = x @ W_hat[idx]ᵀ via the NATIVE NVFP4 GEMM. Klein's
# trainer chain runs F32 activations, so cast to BF16 at the GEMM boundary
# (== the reference trainer's bf16 activations — the exact _base_fwd_i8
# boundary discipline) and cast the BF16 result back to x's dtype so the rest
# of the block stays dtype-identical to the bf16 base path.
def _base_fwd_nv4(
    x: Tensor, p: SingleBlockNvfp4, idx: Int, ctx: DeviceContext,
) raises -> Tensor:
    var o: Tensor
    if x.dtype() == STDtype.BF16:
        o = squareq_nvfp4_linear(
            x, p.nvq[idx][], p.nvs[idx][], p.nvg[idx],
            p.ld[idx][], p.lu[idx][], ctx,
        )
    else:
        var xb = cast_tensor(x, STDtype.BF16, ctx)
        o = squareq_nvfp4_linear(
            xb, p.nvq[idx][], p.nvs[idx][], p.nvg[idx],
            p.ld[idx][], p.lu[idx][], ctx,
        )
    if o.dtype() != x.dtype():
        return cast_tensor(o, x.dtype(), ctx)
    return o^


# ── FORWARD of one SINGLE block with the FROZEN BASE matmuls dispatched int8 ────
# GATED: when `int8` is None this is BYTE-IDENTICAL to the bf16 base path (the two
# base matmuls run the unchanged `linear`); when present the two base matmuls run
# int8 W8A8. LoRA adapters (bf16) are UNCHANGED — same _klein_lora_fwd_dropout as
# single_block_lora_forward_device_resident. This is an ADDITIONAL function; the
# existing bf16 trainer forwards are untouched.
def single_block_int8_base_forward_device_resident[
    H: Int, Dh: Int, S: Int
](
    x_t: TArc,
    w: SingleBlockWeights, mv: SingleModVecsDevice, lora: SingleBlockLoraDevice,
    int8: Optional[SingleBlockInt8],
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

    var ln_t = lnmod_placeholder(ctx)
    var norm_t = layer_norm_modulate(x_t[], mv.scale[], mv.shift[], eps, ctx, ln_t)

    # BASE MATMUL 1 (idx 0): fused = norm_t @ w1ᵀ  [S, 3D+2F]  (int8 or bf16)
    var fused = _base_fwd_i8(norm_t, w.w1[], int8, 0, ctx)
    # LoRA on to_qkv_mlp_proj: FULL bf16 delta [S,3D+2F] (UNCHANGED).
    if lora.qkv:
        var dlt = _klein_lora_fwd_dropout(norm_t, lora.qkv.value(), S, drop_qkv, ctx)
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

    # BASE MATMUL 2 (idx 1): out = out_in @ w2ᵀ  [S, D]  (int8 or bf16)
    var out_proj = _base_fwd_i8(out_in, w.w2[], int8, 1, ctx)
    # LoRA on to_out: input is the FULL out_in [S,D+F] (UNCHANGED).
    if lora.out:
        var dlt2 = _klein_lora_fwd_dropout(out_in, lora.out.value(), S, drop_out, ctx)
        out_proj = add(out_proj, dlt2, ctx)

    var result = residual_gate(x_t[], mv.gate[], out_proj, ctx)

    var saved = SingleBlockSaved(
        x_t.copy(), TArc(ln_t^), TArc(norm_t^), TArc(q_pre^), TArc(k_pre^),
        TArc(q_rms^), TArc(k_rms^), TArc(v^),
        TArc(q_rope^), TArc(k_rope^), TArc(att_flat^),
        TArc(mlp_gate^), TArc(mlp_up^), TArc(mlp^), TArc(out_in^),
    )
    return SingleBlockDeviceForward(TArc(result^), saved^)
