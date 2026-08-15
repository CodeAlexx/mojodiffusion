# models/dit/krea2_dit.mojo — Krea-2-Raw (krea2) single-stream MMDiT.
#
# Reference: torchref krea2 src/mmdit.py (`SingleStreamDiT`). This file holds
# the inference port, chunk by chunk: the `Krea2Config` struct + 3-axis
# interleaved RoPE (chunk 1), RMSNorm/SwiGLU/Modulations (chunk 2), Attention
# (chunk 3), SingleStreamBlock (chunk 4), and the embedders + input/output heads
# (chunk 5: temb/tmlp/tproj/txtmlp/first/LastLayer). The top-level forward
# (chunk 6) wires these together.
#
# ── RoPE math (1:1 with mmdit.py rope()/ropeapply()/PositionalEncoding) ───────
#   rope(pos, dim, theta):
#     scale = arange(0, dim, 2, f64) / dim        # dim/2 entries
#     omega = 1 / theta^scale                     # = theta^(-i/(dim/2)) = theta^(-i/half_a)
#     out[n, d] = pos[n] * omega[d]               # d in [0, dim/2)
#     2x2 rotation per (n,d): [[cos, -sin], [sin, cos]]
#   PositionalEncoding.forward(pos):
#     cat over the 3 axes along the freq dim, each axis i using dim=axdims[i].
#     axis order = [global, h, w]; axdims = [32, 48, 48] for headdim=128.
#     -> table covering half = sum(axdims)/2 = 64 = headdim/2 freqs per token.
#   ropeapply(xq, xk, freqs):  xq.reshape(*shape, -1, 1, 2) -> INTERLEAVED pairs
#     (x[2i], x[2i+1]) with angle index i:
#       out0 = freqs[..,0,0]*x0 + freqs[..,0,1]*x1 = cos*x0 - sin*x1
#       out1 = freqs[..,1,0]*x0 + freqs[..,1,1]*x1 = sin*x0 + cos*x1
#     This is EXACTLY ops/rope.rope_interleaved's convention.
#
# inv_freq is theta^(-i/half_a), identical to ops/rope_tables's exponent. We do
# NOT reuse ops/rope_tables.build_multiaxis_rope_tables here, because it runs
# plain F32 trig with no range reduction: krea2's global axis reaches positions
# ~ seq-len (thousands) and theta=1e3 keeps omega ~ 1.0 for the low freqs, so the
# angle hits thousands of radians where F32 sin/cos is inaccurate. We mirror the
# F64-range-reduction idiom from models/dit/ideogram4_mrope.mojo (omega in F64,
# reduce the angle mod 2pi in F64, then F32 trig on the small remainder). The
# torch reference computes omega in F64 and trig in F32 with proper reduction, so
# this matches it. The apply step reuses ops/rope.rope_interleaved unchanged.
#
# Mojo 1.0.0b1, NVIDIA GPU. Inference-only.

from max.gpu.host import DeviceContext, HostBuffer
from std.gpu import global_idx, block_idx, thread_idx
from max.gpu import barrier
from max.gpu.memory import AddressSpace
from std.memory import stack_allocation, ArcPointer
from std.math import cos as fcos, sin as fsin, exp, log, floor, sqrt
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.linear import linear
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.activations import swiglu as swiglu_op, sigmoid, gelu
from serenitymojo.ops.tensor_algebra import (
    add, slice, reshape, mul, mul_scalar, transpose, concat, zeros_device,
)
from serenitymojo.ops.attention import sdpa_nomask, sdpa, sdpa_tiled, sdpa_nomask_tiled
from serenitymojo.ops.attention_small import sdpa_nomask_small
from serenitymojo.ops.attention_flash import sdpa_flash_fwd_padmask
from serenitymojo.ops.gqa_backward import repeat_kv_f32
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.embeddings import timestep_embedding
from serenitymojo.lora import LoraSet
from serenitymojo.ops.fp8 import fp8_e4m3_dequant_perrow_to_bf16
# int8 W8A8-RESIDENT inference base (task #11): the TRAINER's int8 GEMM primitive
# (tensorwise int8 weight scale + per-token int8 activation quant, int8×int8→int32).
from serenitymojo.ops.int8_linear import int8_linear_fwd
# int8 encode (load-once quantize of the pinned-HOST remainder) + mempool trim
# (return the per-weight transient bf16 sources to the OS during the build).
from serenitymojo.ops.int8_quant import int8_tensorwise_scale, int8_encode_tensorwise
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current


comptime _DYN1 = Layout.row_major(-1)
comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256
comptime _MAX_AXES = 4  # global/h/w (+ optional 4th); bounded.


# ── fp8-RESIDENT base store (MOVED here from krea2_stack.mojo to break the
# krea2_stack→krea2_dit import cycle: krea2_forward needs the type, krea2_stack
# already imports krea2_dit, so these structs live in the lower module). Pure data
# holders (ArcPointer carriers). build_krea2_resident_fp8 stays in krea2_stack. ──
struct Krea2BlockResidentFp8(Copyable, Movable):
    var fp8: List[ArcPointer[Tensor]]    # len 8: E4M3 bytes [out,in] per matmul weight
    var scale: List[ArcPointer[Tensor]]  # len 8: F32 per-output-row scale [out]
    var qnorm_scale: ArcPointer[Tensor]  # raw checkpoint dtype [HEADDIM]
    var knorm_scale: ArcPointer[Tensor]  # raw checkpoint dtype [HEADDIM]
    var prenorm_scale: ArcPointer[Tensor]   # raw checkpoint dtype [features]
    var postnorm_scale: ArcPointer[Tensor]  # raw checkpoint dtype [features]
    var mod_lin: ArcPointer[Tensor]         # bf16 [6*features]

    def __init__(
        out self, var fp8: List[ArcPointer[Tensor]], var scale: List[ArcPointer[Tensor]],
        var qnorm_scale: ArcPointer[Tensor], var knorm_scale: ArcPointer[Tensor],
        var prenorm_scale: ArcPointer[Tensor], var postnorm_scale: ArcPointer[Tensor],
        var mod_lin: ArcPointer[Tensor],
    ):
        self.fp8 = fp8^
        self.scale = scale^
        self.qnorm_scale = qnorm_scale^
        self.knorm_scale = knorm_scale^
        self.prenorm_scale = prenorm_scale^
        self.postnorm_scale = postnorm_scale^
        self.mod_lin = mod_lin^


struct Krea2ResidentFp8(Copyable, Movable):
    var blocks: List[Krea2BlockResidentFp8]   # len == nblocks

    def __init__(out self, var blocks: List[Krea2BlockResidentFp8]):
        self.blocks = blocks^


# ── SquareQ W4-RESIDENT base store (cfg.quantized_resident == "squareq_w4").
# Mirrors the fp8 store but holds the PREBUILT sidecar payload per matmul weight
# (scripts/squareq_build_slab.py: int4-g64 of the H256-rotated rank-R residual +
# BF16 low-rank factors, ~0.28x bf16). Per-block load reconstructs the bf16
# weight W_hat = dequant4@H_bd + lora_up@lora_down^T (ops/squareq.mojo) — the
# unchanged block fwd/bwd consume it exactly like the fp8 dequant path. ──
struct Krea2BlockResidentSquareq(Copyable, Movable):
    var qweight: List[ArcPointer[Tensor]]    # len 8: U8 packed int4 [out, in/2]
    var wscales: List[ArcPointer[Tensor]]    # len 8: BF16 [in/64, out]
    var lora_down: List[ArcPointer[Tensor]]  # len 8: BF16 [in, R]
    var lora_up: List[ArcPointer[Tensor]]    # len 8: BF16 [out, R]
    var qnorm_scale: ArcPointer[Tensor]  # raw checkpoint dtype [HEADDIM]
    var knorm_scale: ArcPointer[Tensor]  # raw checkpoint dtype [HEADDIM]
    var prenorm_scale: ArcPointer[Tensor]   # raw checkpoint dtype [features]
    var postnorm_scale: ArcPointer[Tensor]  # raw checkpoint dtype [features]
    var mod_lin: ArcPointer[Tensor]         # bf16 [6*features]

    def __init__(
        out self,
        var qweight: List[ArcPointer[Tensor]], var wscales: List[ArcPointer[Tensor]],
        var lora_down: List[ArcPointer[Tensor]], var lora_up: List[ArcPointer[Tensor]],
        var qnorm_scale: ArcPointer[Tensor], var knorm_scale: ArcPointer[Tensor],
        var prenorm_scale: ArcPointer[Tensor], var postnorm_scale: ArcPointer[Tensor],
        var mod_lin: ArcPointer[Tensor],
    ):
        self.qweight = qweight^
        self.wscales = wscales^
        self.lora_down = lora_down^
        self.lora_up = lora_up^
        self.qnorm_scale = qnorm_scale^
        self.knorm_scale = knorm_scale^
        self.prenorm_scale = prenorm_scale^
        self.postnorm_scale = postnorm_scale^
        self.mod_lin = mod_lin^


struct Krea2ResidentSquareq(Copyable, Movable):
    var blocks: List[Krea2BlockResidentSquareq]   # len == nblocks

    def __init__(out self, var blocks: List[Krea2BlockResidentSquareq]):
        self.blocks = blocks^


# ── int8 W8A8-RESIDENT base store (cfg.quantized_resident == "int_w8a8"). Mirrors
# the fp8 store but holds int8 tensorwise-quantized weights: no per-step dequant
# (the whole point) — the block does int8×int8→int32 GEMM directly. Per matmul
# weight we hold BOTH orientations (fwd contracts K over w_8[N,K]; bwd contracts N
# over w_8T[K,N]) + the SCALAR tensorwise scale (factors out of both). The 5 small
# per-block tensors stay bf16/F32 resident, exactly as the fp8 store. ──
struct Krea2BlockResidentInt8(Copyable, Movable):
    # ONE orientation only (w8[N,K], ~1 byte/param). The backward's w8T[K,N] is
    # transposed on the fly per step (cheap byte-shuffle, transient) — storing both
    # would double the resident footprint to ~24GB and blow 16GB. == reference trainer (one int8
    # weight, transpose in the GEMM).
    var w8: List[ArcPointer[Tensor]]     # len 8: int8 bytes [out,in]=[N,K]
    var scale: List[ArcPointer[Tensor]]  # len 8: F32 scalar tensorwise scale [1]
    var qnorm_scale: ArcPointer[Tensor]  # raw checkpoint dtype [HEADDIM]
    var knorm_scale: ArcPointer[Tensor]  # raw checkpoint dtype [HEADDIM]
    var prenorm_scale: ArcPointer[Tensor]   # raw checkpoint dtype [features]
    var postnorm_scale: ArcPointer[Tensor]  # raw checkpoint dtype [features]
    var mod_lin: ArcPointer[Tensor]         # bf16 [6*features]

    def __init__(
        out self,
        var w8: List[ArcPointer[Tensor]], var scale: List[ArcPointer[Tensor]],
        var qnorm_scale: ArcPointer[Tensor], var knorm_scale: ArcPointer[Tensor],
        var prenorm_scale: ArcPointer[Tensor], var postnorm_scale: ArcPointer[Tensor],
        var mod_lin: ArcPointer[Tensor],
    ):
        self.w8 = w8^
        self.scale = scale^
        self.qnorm_scale = qnorm_scale^
        self.knorm_scale = knorm_scale^
        self.prenorm_scale = prenorm_scale^
        self.postnorm_scale = postnorm_scale^
        self.mod_lin = mod_lin^


struct Krea2ResidentInt8(Copyable, Movable):
    var blocks: List[Krea2BlockResidentInt8]   # len == resident_blocks

    def __init__(out self, var blocks: List[Krea2BlockResidentInt8]):
        self.blocks = blocks^


# ── int8 W8A8 PINNED-HOST store for INFERENCE (task #11). The fully-resident
# 28-block int8 store is ~12.1GB — MEASURED OOM on the 16GB 5080 alongside the
# inference pipeline baseline (VAE-encode remnants + activations + the end-of-run
# VAE decode). Hybrid instead: the FIRST K blocks device-resident
# (Krea2ResidentInt8), the remainder held as PINNED-HOST int8 bytes and H2D'd per
# block per forward (~433MB/block over PCIe, no disk, no decode) — the trainer's
# Krea2HostInt8 pattern (krea2_stack.mojo:876+), re-implemented here because the
# inference forward lives in this lower module (krea2_stack imports krea2_dit).
# The scalar scales + 5 small per-block tensors stay device-resident (tiny). ──
comptime _HArc = ArcPointer[HostBuffer[DType.uint8]]


struct Krea2BlockHostInt8Inf(Copyable, Movable):
    """One NON-resident block as PINNED-HOST int8: 8 matmul weights as host int8
    bytes [N,K] (H2D per forward); scalar scales + 5 small tensors device-resident."""
    var w8_h: List[_HArc]                    # len 8: pinned host int8 bytes [N,K]
    var w8_nbytes: List[Int]                 # len 8
    var w8_shape: List[List[Int]]            # len 8: [N,K]
    var scale: List[ArcPointer[Tensor]]      # len 8: F32 scalar tensorwise scale [1]
    var qnorm_scale: ArcPointer[Tensor]      # bf16 [HEADDIM]
    var knorm_scale: ArcPointer[Tensor]      # bf16 [HEADDIM]
    var prenorm_scale: ArcPointer[Tensor]    # bf16 [features]
    var postnorm_scale: ArcPointer[Tensor]   # bf16 [features]
    var mod_lin: ArcPointer[Tensor]          # bf16 [6*features]

    def __init__(
        out self,
        var w8_h: List[_HArc], var w8_nbytes: List[Int],
        var w8_shape: List[List[Int]], var scale: List[ArcPointer[Tensor]],
        var qnorm_scale: ArcPointer[Tensor], var knorm_scale: ArcPointer[Tensor],
        var prenorm_scale: ArcPointer[Tensor], var postnorm_scale: ArcPointer[Tensor],
        var mod_lin: ArcPointer[Tensor],
    ):
        self.w8_h = w8_h^
        self.w8_nbytes = w8_nbytes^
        self.w8_shape = w8_shape^
        self.scale = scale^
        self.qnorm_scale = qnorm_scale^
        self.knorm_scale = knorm_scale^
        self.prenorm_scale = prenorm_scale^
        self.postnorm_scale = postnorm_scale^
        self.mod_lin = mod_lin^


struct Krea2HostInt8Inf(Copyable, Movable):
    var first: Int                           # global block index of blocks[0]
    var blocks: List[Krea2BlockHostInt8Inf]  # blocks [first : first+len)

    def __init__(out self, first: Int, var blocks: List[Krea2BlockHostInt8Inf]):
        self.first = first
        self.blocks = blocks^


# Load-once-quantize blocks [resident_blocks:nblocks] to PINNED-HOST int8. Same
# tensorwise int8 encode as the trainer's build_krea2_resident_int8 / _host_int8
# (== reference trainer LinearW8A8); D2H the int8 bytes into pinned host buffers and drop the
# device copies, so device cost ≈ 0 (only scalar scales + 5 small tensors resident).
def build_krea2_host_int8_inf(
    st: ShardedSafeTensors, key_prefix: String, nblocks: Int,
    resident_blocks: Int, ctx: DeviceContext,
) raises -> Krea2HostInt8Inf:
    var n_res = resident_blocks if resident_blocks < nblocks else nblocks
    var blocks = List[Krea2BlockHostInt8Inf]()
    for bi in range(n_res, nblocks):
        var p = key_prefix + "blocks." + String(bi) + "."
        var keys = List[String]()
        keys.append(p + "attn.wq.weight")
        keys.append(p + "attn.wk.weight")
        keys.append(p + "attn.wv.weight")
        keys.append(p + "attn.gate.weight")
        keys.append(p + "attn.wo.weight")
        keys.append(p + "mlp.gate.weight")
        keys.append(p + "mlp.up.weight")
        keys.append(p + "mlp.down.weight")
        var w8_h = List[_HArc]()
        var w8_nbytes = List[Int]()
        var w8_shape = List[List[Int]]()
        var scale = List[ArcPointer[Tensor]]()
        for ki in range(8):
            var w_bf = Tensor.from_view_as_bf16(st.tensor_view(keys[ki]), ctx)  # [N,K]
            var sc = int8_tensorwise_scale(w_bf, ctx)                           # F32 [1]
            var w8 = int8_encode_tensorwise(w_bf, sc, ctx)                      # I8 [N,K]
            var bh = ctx.enqueue_create_host_buffer[DType.uint8](w8.nbytes())
            ctx.enqueue_copy(bh, w8.buf)          # D2H into pinned host
            ctx.synchronize()                      # D2H done; fence transients
            cu_mempool_trim_current(0)             # return the bf16 source to the OS
            w8_h.append(_HArc(bh^))
            w8_nbytes.append(w8.nbytes())
            w8_shape.append(w8.shape().copy())
            scale.append(ArcPointer[Tensor](sc^))
        blocks.append(Krea2BlockHostInt8Inf(
            w8_h^, w8_nbytes^, w8_shape^, scale^,
            ArcPointer(_scale(st, p + "attn.qknorm.qnorm.scale", ctx)),
            ArcPointer(_scale(st, p + "attn.qknorm.knorm.scale", ctx)),
            ArcPointer(_scale(st, p + "prenorm.scale", ctx)),
            ArcPointer(_scale(st, p + "postnorm.scale", ctx)),
            ArcPointer(_wb(st, p + "mod.lin", ctx)),
        ))
        if (bi + 1 - n_res) % 7 == 0 or bi + 1 == nblocks:
            print("int8_w8a8 host(inf): pinned block", bi + 1, "/", nblocks,
                  "(", bi + 1 - n_res, "of", nblocks - n_res, "host)")
    return Krea2HostInt8Inf(n_res, blocks^)


# Per-forward H2D reconstruction of one HOST block (local hi = li - store.first):
# fresh transient device I8 tensors (freed when the caller's per-block sync drains)
# + Arc refcount copies of the resident scales/small tensors, packaged as the SAME
# Krea2BlockResidentInt8 the resident path feeds krea2_single_stream_block_i8.
def _krea2_host_i8_block_dev(
    store: Krea2HostInt8Inf, hi: Int, ctx: DeviceContext
) raises -> Krea2BlockResidentInt8:
    ref b = store.blocks[hi]
    var w8 = List[ArcPointer[Tensor]]()
    var scale = List[ArcPointer[Tensor]]()
    for i in range(8):
        var dbuf = ctx.enqueue_create_buffer[DType.uint8](b.w8_nbytes[i])
        ctx.enqueue_copy(dbuf, b.w8_h[i][])       # H2D pinned int8
        var wt = Tensor(dbuf^, b.w8_shape[i].copy(), STDtype.I8)
        w8.append(ArcPointer[Tensor](wt^))
        scale.append(b.scale[i].copy())
    return Krea2BlockResidentInt8(
        w8^, scale^,
        b.qnorm_scale.copy(), b.knorm_scale.copy(),
        b.prenorm_scale.copy(), b.postnorm_scale.copy(), b.mod_lin.copy(),
    )


# ── SHARED (non-block) weights, LOAD-ONCE resident (task #11). krea2_forward
# used to re-load every shared weight from the safetensors PER FORWARD via
# from_view_as_bf16 — a single-threaded host byte-copy + F32→bf16 cast loop over
# ~1.3GB (tproj.1.weight alone is [36864,6144] F32 = 906MB on disk) → MEASURED
# ~4.4s of host time per step (2 forwards) while the GPU sat idle. This store
# loads them once (~1.3GB device bf16); the per-forward cost becomes Arc clones.
# Tensor slots (fixed order, see build_krea2_shared_resident):
#   0 first.weight        1 first.bias
#   2 tmlp.0.weight       3 tmlp.0.bias       4 tmlp.2.weight   5 tmlp.2.bias
#   6 tproj.1.weight      7 tproj.1.bias      8 txtfusion.projector.weight
#   9 txtmlp.0.scale     10 txtmlp.1.weight  11 txtmlp.1.bias
#  12 txtmlp.3.weight    13 txtmlp.3.bias
#  14 last.norm.scale    15 last.modulation.lin
#  16 last.linear.weight 17 last.linear.bias
# txtf slots: 0 layerwise_blocks.0, 1 layerwise_blocks.1, 2 refiner_blocks.0,
# 3 refiner_blocks.1 (each a full Krea2TextFusionWeights bundle). ──
struct Krea2SharedResident(Copyable, Movable):
    var t: List[ArcPointer[Tensor]]           # 18 slots, order above
    var txtf: List[Krea2TextFusionWeights]    # 4 bundles, order above

    def __init__(
        out self, var t: List[ArcPointer[Tensor]],
        var txtf: List[Krea2TextFusionWeights],
    ):
        self.t = t^
        self.txtf = txtf^


def build_krea2_shared_resident(
    st: ShardedSafeTensors, key_prefix: String, ctx: DeviceContext
) raises -> Krea2SharedResident:
    """Load-once the SHARED (non-block) krea2 weights, same loaders/dtypes as the
    per-forward path (_wb/_scale/_txtf_bundle) so values are byte-identical."""
    var t = List[ArcPointer[Tensor]]()
    t.append(ArcPointer(_wb(st, key_prefix + "first.weight", ctx)))
    t.append(ArcPointer(_wb(st, key_prefix + "first.bias", ctx)))
    t.append(ArcPointer(_wb(st, key_prefix + "tmlp.0.weight", ctx)))
    t.append(ArcPointer(_wb(st, key_prefix + "tmlp.0.bias", ctx)))
    t.append(ArcPointer(_wb(st, key_prefix + "tmlp.2.weight", ctx)))
    t.append(ArcPointer(_wb(st, key_prefix + "tmlp.2.bias", ctx)))
    t.append(ArcPointer(_wb(st, key_prefix + "tproj.1.weight", ctx)))
    t.append(ArcPointer(_wb(st, key_prefix + "tproj.1.bias", ctx)))
    t.append(ArcPointer(_wb(st, key_prefix + "txtfusion.projector.weight", ctx)))
    t.append(ArcPointer(_scale(st, key_prefix + "txtmlp.0.scale", ctx)))
    t.append(ArcPointer(_wb(st, key_prefix + "txtmlp.1.weight", ctx)))
    t.append(ArcPointer(_wb(st, key_prefix + "txtmlp.1.bias", ctx)))
    t.append(ArcPointer(_wb(st, key_prefix + "txtmlp.3.weight", ctx)))
    t.append(ArcPointer(_wb(st, key_prefix + "txtmlp.3.bias", ctx)))
    t.append(ArcPointer(_scale(st, key_prefix + "last.norm.scale", ctx)))
    t.append(ArcPointer(_wb(st, key_prefix + "last.modulation.lin", ctx)))
    t.append(ArcPointer(_wb(st, key_prefix + "last.linear.weight", ctx)))
    t.append(ArcPointer(_wb(st, key_prefix + "last.linear.bias", ctx)))
    var txtf = List[Krea2TextFusionWeights]()
    txtf.append(_txtf_bundle(st, key_prefix + "txtfusion.layerwise_blocks.0", ctx))
    txtf.append(_txtf_bundle(st, key_prefix + "txtfusion.layerwise_blocks.1", ctx))
    txtf.append(_txtf_bundle(st, key_prefix + "txtfusion.refiner_blocks.0", ctx))
    txtf.append(_txtf_bundle(st, key_prefix + "txtfusion.refiner_blocks.1", ctx))
    ctx.synchronize()
    return Krea2SharedResident(t^, txtf^)


# Shared-weight source: device clone from the resident store when present, else
# the per-forward disk load (the historical path, byte-identical values).
def _shw(
    shared: Optional[Krea2SharedResident], i: Int,
    st: ShardedSafeTensors, key: String, ctx: DeviceContext,
) raises -> Tensor:
    if shared:
        return shared.value().t[i][].clone(ctx)
    return _wb(st, key, ctx)


def _shs(
    shared: Optional[Krea2SharedResident], i: Int,
    st: ShardedSafeTensors, key: String, ctx: DeviceContext,
) raises -> Tensor:
    if shared:
        return shared.value().t[i][].clone(ctx)
    return _scale(st, key, ctx)


def _shtxtf(
    shared: Optional[Krea2SharedResident], i: Int,
    st: ShardedSafeTensors, prefix: String, ctx: DeviceContext,
) raises -> Krea2TextFusionWeights:
    if shared:
        return shared.value().txtf[i].copy()
    return _txtf_bundle(st, prefix, ctx)


# ── Krea2Config ──────────────────────────────────────────────────────────────
@fieldwise_init
struct Krea2Config(Copyable, Movable):
    """Krea-2-Raw SingleStreamDiT config (KREA2_MMDIT_CONFIG, krea2.py:55-68).

    Defaults are the reference "single_mmdit_large_wide" architecture
    (oss_raw / oss_turbo share it). `headdim = features // heads = 128` and the
    3-axis RoPE split `axes = [headdim - 12*(headdim//16), 6*(headdim//16),
    6*(headdim//16)] = [32, 48, 48]` are derived (see `head_dim()`/`rope_axes()`).
    """

    var features: Int      # 6144  — model (token) width.
    var tdim: Int          # 256   — timestep-embedding sinusoid width.
    var txtdim: Int        # 2560  — Qwen3-VL text-feature width.
    var heads: Int         # 48    — image-stream attention heads.
    var kvheads: Int       # 12    — image-stream KV heads (GQA).
    var multiplier: Int    # 4     — SwiGLU hidden multiplier.
    var layers: Int        # 28    — SingleStreamBlock depth.
    var patch: Int         # 2     — latent patch size.
    var channels: Int      # 16    — latent channels (Qwen-Image VAE z_dim).
    var txtheads: Int      # 20    — TextFusion attention heads.
    var txtkvheads: Int    # 20    — TextFusion KV heads.
    var txtlayers: Int     # 12    — selected encoder hidden-state layers fed in.
    var theta: Float32     # 1e3   — RoPE base (config.theta; overrides rope()'s 1e4 default).
    var bias: Bool         # False — Linear bias.

    @staticmethod
    def default() -> Krea2Config:
        """KREA2_MMDIT_CONFIG defaults (krea2.py:55-68 + SingleMMDiTConfig)."""
        return Krea2Config(
            features=6144,
            tdim=256,
            txtdim=2560,
            heads=48,
            kvheads=12,
            multiplier=4,
            layers=28,
            patch=2,
            channels=16,
            txtheads=20,
            txtkvheads=20,
            txtlayers=12,
            theta=Float32(1.0e3),
            bias=False,
        )

    def head_dim(self) -> Int:
        """Head dim = features // heads = 6144 // 48 = 128 (mmdit.py:202/346)."""
        return self.features // self.heads

    def rope_axes(self) -> List[Int]:
        """Per-axis FULL rotary dims (mmdit.py:347-353).

            axes = [headdim - 12*(headdim//16), 6*(headdim//16), 6*(headdim//16)]

        For headdim=128 -> [32, 48, 48] (sums to headdim, all even). Axis order
        is [global, h, w]; pos[..., a] must follow this order.
        """
        var hd = self.head_dim()
        var unit = hd // 16
        var ax = List[Int]()
        ax.append(hd - 12 * unit)
        ax.append(6 * unit)
        ax.append(6 * unit)
        return ax^


# ── 3-axis interleaved RoPE table builder ────────────────────────────────────
# One GPU thread per (row, col) of the [rows, half] table. The axis-block walk
# finds which of the 3 axes owns column `col` and its local index within that
# axis, exactly like ops/rope_tables but with F64 range reduction.
def _krea2_rope_kernel[out_dtype: DType](
    positions: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [rows*num_axes]
    axes_half: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],    # [num_axes]
    cos_t: LayoutTensor[out_dtype, _DYN2, MutAnyOrigin],          # [rows, half]
    sin_t: LayoutTensor[out_dtype, _DYN2, MutAnyOrigin],          # [rows, half]
    rows_w: Int32,
    half_w: Int32,         # sum(axes_half) == head_dim/2
    num_axes_w: Int32,
    log_theta: Float64,
):
    var rows = Int(rows_w)
    var half = Int(half_w)
    var num_axes = Int(num_axes_w)
    var idx = Int(global_idx.x)
    var total = rows * half
    if idx >= total:
        return
    var row = idx // half
    var col = idx % half

    # Walk axis blocks to find the owning axis `a` and local index `local_i`.
    var off = 0
    var a = 0
    var local_i = col
    var ha = 0
    while a < num_axes:
        ha = Int(rebind[Scalar[DType.int32]](axes_half[a]))
        if col < off + ha:
            local_i = col - off
            break
        off += ha
        a += 1

    # pos for token `row` along axis `a`: positions[row*num_axes + a].
    var pos = rebind[Scalar[DType.float32]](positions[row * num_axes + a])
    # omega = 1/theta^(local_i/half_a) = theta^(-local_i/half_a). Compute in F64
    # for the small-magnitude exponent precision (mirrors torch rope() f64 arange).
    # log_theta is already F64 (computed F64 at the call site) so the whole
    # exponent path stays F64 — no F32 log widening.
    var inv = exp((-Float64(local_i) / Float64(ha)) * log_theta)
    var angle = Float64(pos) * inv
    # GPU has no F64 trig, but F64 arithmetic is fine: reduce the angle mod 2pi in
    # F64 (krea2's global axis hits thousands of radians where F32 trig is wrong),
    # then F32 trig on the small reduced remainder.
    comptime TWO_PI = Float64(6.283185307179586476925286766559)
    var k = floor(angle / TWO_PI + 0.5)
    var reduced = Float32(angle - k * TWO_PI)
    cos_t[row, col] = rebind[cos_t.element_type](fcos(reduced).cast[out_dtype]())
    sin_t[row, col] = rebind[sin_t.element_type](fsin(reduced).cast[out_dtype]())


def build_krea2_rope(
    positions: Tensor,
    axes_dims: List[Int],
    theta: Float32,
    ctx: DeviceContext,
    out_dtype: STDtype,
) raises -> Tuple[Tensor, Tensor]:
    """Krea2 3-axis interleaved RoPE cos/sin tables (mmdit.py PositionalEncoding).

    positions: [rows * num_axes] F32, token-major (index `t*num_axes + a` holds
               token t's grid position along axis a; axis order [global, h, w]).
    axes_dims: per-axis FULL rotary dim (each even); `sum(axes_dims)` must equal
               head_dim, and `sum(axes_dims)/2` (head_dim/2) is the produced
               table width `half`. For headdim=128: [32, 48, 48].
    theta:     RoPE base (krea2 config.theta = 1e3).
    returns (cos, sin), each [rows, half] in out_dtype. Concatenated over the 3
            axes with per-axis omega_i = theta^(-i/half_a). Feed straight into
            ops/rope.rope_interleaved with q/k of shape [..., head_dim].
    Trig is computed with F64 range reduction; storage casts to out_dtype.
    """
    var num_axes = len(axes_dims)
    if num_axes < 1 or num_axes > _MAX_AXES:
        raise Error("build_krea2_rope: num_axes must be 1.._MAX_AXES")
    if positions.dtype() != STDtype.F32:
        raise Error("build_krea2_rope: positions must be F32")
    var pn = positions.numel()
    if pn % num_axes != 0:
        raise Error("build_krea2_rope: positions numel must be rows*num_axes")
    var rows = pn // num_axes

    var half = 0
    var axes_half_host = List[Int32]()
    for a in range(num_axes):
        var da = axes_dims[a]
        if da % 2 != 0:
            raise Error("build_krea2_rope: each axis dim must be even")
        var ha = da // 2
        axes_half_host.append(Int32(ha))
        half += ha

    # Upload axes_half as a true-I32 device buffer (mirrors ops/rope_tables).
    var axes_host = ctx.enqueue_create_host_buffer[DType.uint8](num_axes * 4)
    var axes_hp = axes_host.unsafe_ptr().bitcast[Int32]()
    for a in range(num_axes):
        axes_hp[a] = axes_half_host[a]
    var axes_buf = ctx.enqueue_create_buffer[DType.uint8](num_axes * 4)
    ctx.enqueue_copy(dst_buf=axes_buf, src_buf=axes_host)

    var cos_buf = ctx.enqueue_create_buffer[DType.uint8](
        rows * half * out_dtype.byte_size()
    )
    var sin_buf = ctx.enqueue_create_buffer[DType.uint8](
        rows * half * out_dtype.byte_size()
    )

    var p_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](pn))
    var a_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](num_axes))
    var f_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, half))

    var P = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(positions.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=p_rl,
    )
    var A = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=Int(axes_buf.unsafe_ptr().bitcast[Int32]())
        ),
        runtime_layout=a_rl,
    )
    var total = rows * half
    var grid = (total + _BLOCK - 1) // _BLOCK
    var lt = log(Float64(theta))
    var odt = out_dtype.to_mojo_dtype()
    if odt == DType.float32:
        var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(cos_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
        var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(sin_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
        ctx.enqueue_function[_krea2_rope_kernel[DType.float32]](P, A, C, S, Int32(rows), Int32(half), Int32(num_axes), lt, grid_dim=grid, block_dim=_BLOCK)
    elif odt == DType.bfloat16:
        var C = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(cos_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=f_rl,
    )
        var S = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(sin_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=f_rl,
    )
        ctx.enqueue_function[_krea2_rope_kernel[DType.bfloat16]](P, A, C, S, Int32(rows), Int32(half), Int32(num_axes), lt, grid_dim=grid, block_dim=_BLOCK)
    else:
        var C = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(cos_buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=f_rl,
    )
        var S = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(sin_buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=f_rl,
    )
        ctx.enqueue_function[_krea2_rope_kernel[DType.float16]](P, A, C, S, Int32(rows), Int32(half), Int32(num_axes), lt, grid_dim=grid, block_dim=_BLOCK)
    ctx.synchronize()

    var cos_shape = List[Int]()
    cos_shape.append(rows)
    cos_shape.append(half)
    var sin_shape = List[Int]()
    sin_shape.append(rows)
    sin_shape.append(half)
    var cos_out = Tensor(cos_buf^, cos_shape^, out_dtype)
    var sin_out = Tensor(sin_buf^, sin_shape^, out_dtype)
    return (cos_out^, sin_out^)


def apply_krea2_rope(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, ctx: DeviceContext
) raises -> Tuple[Tensor, Tensor]:
    """Apply krea2 interleaved RoPE to q and k (mmdit.py ropeapply()).

    q, k:    [..., head_dim]  (rows = product of leading dims; here rows = the
             same L the cos/sin table was built for, so each row's q/k share its
             token's freqs).
    cos/sin: [rows, head_dim/2] from build_krea2_rope.
    returns (q_rot, k_rot), same shapes/dtype as q/k. Math is F32 inside the
    interleaved kernel (matches ropeapply's xq.float()). q and k are rotated
    with the SAME freqs table (ropeapply passes one `freqs` to both).
    """
    var q_rot = rope_interleaved(q, cos, sin, ctx)
    var k_rot = rope_interleaved(k, cos, sin, ctx)
    return (q_rot^, k_rot^)


# ══════════════════════════════════════════════════════════════════════════════
# CHUNK 2 — SingleStreamBlock leaf ops (RMSNorm / SwiGLU / Modulations).
# Reference: mmdit.py RMSNorm(163-177), SwiGLU(180-194), SimpleModulation(109-119),
# DoubleSharedModulation(122-133).
# ══════════════════════════════════════════════════════════════════════════════


# ── RMSNorm (mmdit.py:163-177) — F32-INTERNAL with weight = scale + 1.0 ───────
# The reference is precision-critical (the Rust noise-saga root cause):
#   t = x.float()                                            # bf16 -> F32
#   t = F.rms_norm(t, (features,), eps=1e-5, weight=scale.float() + 1.0)
#   return t.to(dtype)                                       # F32 -> bf16
# i.e. the rms reduction AND the weight multiply are F32, the weight is the
# F32 reparam (scale.float() + 1.0), and bf16 is touched ONLY at the x-read
# upcast and final store. ops/norm.rms_norm is NOT reused: its bf16 path reads a
# materialized WEIGHT as bf16 (bf16-rounds scale+1 before the multiply) and
# applies the raw weight (no +1 reparam). We hand-roll F32-internal kernels that
# cast scale to F32 inside the op and add 1.0 in F32 inside the multiply.
# x is read as its storage dtype and upcast to F32.
comptime _RMS_TPB = 256  # threads per block (one block per row)


def _krea2_rmsnorm_kernel[x_dtype: DType, scale_dtype: DType](
    x: LayoutTensor[x_dtype, _DYN2, MutAnyOrigin],
    scale: LayoutTensor[scale_dtype, _DYN1, MutAnyOrigin],  # raw scale (NOT scale+1)
    o: LayoutTensor[x_dtype, _DYN2, MutAnyOrigin],
    cols_w: Int32,
    eps: Float32,
):
    var cols = Int(cols_w)
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var shared = stack_allocation[
        _RMS_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    # Sum of squares in F32 (matches F.rms_norm on x.float()).
    var local: Float32 = 0.0
    var c = tid
    while c < cols:
        var v = rebind[Scalar[x_dtype]](x[row, c]).cast[DType.float32]()
        local += v * v
        c += _RMS_TPB
    shared[tid] = local
    barrier()
    var active = _RMS_TPB // 2
    while active > 0:
        if tid < active:
            shared[tid] = shared[tid] + shared[tid + active]
        barrier()
        active //= 2
    var inv = 1.0 / sqrt(shared[0] / Float32(cols) + eps)
    c = tid
    while c < cols:
        var v = rebind[Scalar[x_dtype]](x[row, c]).cast[DType.float32]()
        # weight = scale + 1.0, kept F32 (the reference's scale.float() + 1.0).
        var w = rebind[Scalar[scale_dtype]](scale[c]).cast[DType.float32]() + Float32(1.0)
        o[row, c] = rebind[o.element_type]((v * inv * w).cast[x_dtype]())
        c += _RMS_TPB


def krea2_rmsnorm(
    x: Tensor, scale: Tensor, eps: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Krea2 RMSNorm (mmdit.py:163-177). F32-internal; weight = scale + 1.0.

    x:     [..., features]  (storage dtype; read upcast to F32 == x.float()).
    scale: [features]       storage dtype. The raw scale is cast to F32 inside
           the op; weight is scale.float()+1.0, added in F32 inside the kernel.
    eps:   1e-5.
    returns [..., features] in x's dtype (F32 math, cast only at store).
    """
    var xshape = x.shape()
    if len(xshape) < 1:
        raise Error("krea2_rmsnorm: x must have rank >= 1")
    var cols = xshape[len(xshape) - 1]
    if scale.numel() != cols:
        raise Error("krea2_rmsnorm: scale numel must equal features")
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]

    var dt = x.dtype().to_mojo_dtype()
    var sdt = scale.dtype().to_mojo_dtype()
    if (
        sdt != DType.float32
        and sdt != DType.bfloat16
        and sdt != DType.float16
    ):
        raise Error("krea2_rmsnorm: scale dtype must be F32/BF16/F16")
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, cols))
    var g_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](cols))
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=x_rl,
    )
        var O = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=x_rl,
    )
        if sdt == DType.float32:
            var SC = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(scale.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=g_rl,
    )
            ctx.enqueue_function[_krea2_rmsnorm_kernel[DType.float32, DType.float32]](X, SC, O, Int32(cols), eps, grid_dim=rows, block_dim=_RMS_TPB)
        elif sdt == DType.bfloat16:
            var SC = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(scale.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=g_rl,
    )
            ctx.enqueue_function[_krea2_rmsnorm_kernel[DType.float32, DType.bfloat16]](X, SC, O, Int32(cols), eps, grid_dim=rows, block_dim=_RMS_TPB)
        else:
            var SC = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(scale.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=g_rl,
    )
            ctx.enqueue_function[_krea2_rmsnorm_kernel[DType.float32, DType.float16]](X, SC, O, Int32(cols), eps, grid_dim=rows, block_dim=_RMS_TPB)
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=x_rl,
    )
        var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=x_rl,
    )
        if sdt == DType.float32:
            var SC = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(scale.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=g_rl,
    )
            ctx.enqueue_function[_krea2_rmsnorm_kernel[DType.bfloat16, DType.float32]](X, SC, O, Int32(cols), eps, grid_dim=rows, block_dim=_RMS_TPB)
        elif sdt == DType.bfloat16:
            var SC = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(scale.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=g_rl,
    )
            ctx.enqueue_function[_krea2_rmsnorm_kernel[DType.bfloat16, DType.bfloat16]](X, SC, O, Int32(cols), eps, grid_dim=rows, block_dim=_RMS_TPB)
        else:
            var SC = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(scale.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=g_rl,
    )
            ctx.enqueue_function[_krea2_rmsnorm_kernel[DType.bfloat16, DType.float16]](X, SC, O, Int32(cols), eps, grid_dim=rows, block_dim=_RMS_TPB)
    else:
        var X = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=x_rl,
    )
        var O = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=x_rl,
    )
        if sdt == DType.float32:
            var SC = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(scale.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=g_rl,
    )
            ctx.enqueue_function[_krea2_rmsnorm_kernel[DType.float16, DType.float32]](X, SC, O, Int32(cols), eps, grid_dim=rows, block_dim=_RMS_TPB)
        elif sdt == DType.bfloat16:
            var SC = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(scale.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=g_rl,
    )
            ctx.enqueue_function[_krea2_rmsnorm_kernel[DType.float16, DType.bfloat16]](X, SC, O, Int32(cols), eps, grid_dim=rows, block_dim=_RMS_TPB)
        else:
            var SC = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(scale.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=g_rl,
    )
            ctx.enqueue_function[_krea2_rmsnorm_kernel[DType.float16, DType.float16]](X, SC, O, Int32(cols), eps, grid_dim=rows, block_dim=_RMS_TPB)
    # No trailing sync (single-stream ordering; downstream .to_host() syncs).
    return Tensor(out_buf^, x.shape(), x.dtype())


def _krea2_rmsnorm_bwd_dx_kernel[x_dtype: DType, scale_dtype: DType](
    go: LayoutTensor[x_dtype, _DYN2, MutAnyOrigin],
    x: LayoutTensor[x_dtype, _DYN2, MutAnyOrigin],
    scale: LayoutTensor[scale_dtype, _DYN1, MutAnyOrigin],
    dx: LayoutTensor[x_dtype, _DYN2, MutAnyOrigin],
    cols_w: Int32,
    eps: Float32,
):
    var cols = Int(cols_w)
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var shared = stack_allocation[
        _RMS_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var local: Float32 = 0.0
    var c = tid
    while c < cols:
        var v = rebind[Scalar[x_dtype]](x[row, c]).cast[DType.float32]()
        local += v * v
        c += _RMS_TPB
    shared[tid] = local
    barrier()
    var active = _RMS_TPB // 2
    while active > 0:
        if tid < active:
            shared[tid] = shared[tid] + shared[tid + active]
        barrier()
        active //= 2
    var inv = 1.0 / sqrt(shared[0] / Float32(cols) + eps)
    barrier()

    var lgwx: Float32 = 0.0
    c = tid
    while c < cols:
        var gov = rebind[Scalar[x_dtype]](go[row, c]).cast[DType.float32]()
        var xv = rebind[Scalar[x_dtype]](x[row, c]).cast[DType.float32]()
        var w = rebind[Scalar[scale_dtype]](scale[c]).cast[DType.float32]() + Float32(1.0)
        lgwx += gov * w * xv
        c += _RMS_TPB
    shared[tid] = lgwx
    barrier()
    active = _RMS_TPB // 2
    while active > 0:
        if tid < active:
            shared[tid] = shared[tid] + shared[tid + active]
        barrier()
        active //= 2
    var sum_gwx = shared[0]
    barrier()

    var inv3 = inv * inv * inv
    c = tid
    while c < cols:
        var xv = rebind[Scalar[x_dtype]](x[row, c]).cast[DType.float32]()
        var gov = rebind[Scalar[x_dtype]](go[row, c]).cast[DType.float32]()
        var w = rebind[Scalar[scale_dtype]](scale[c]).cast[DType.float32]() + Float32(1.0)
        var out = w * gov * inv - xv * inv3 * (sum_gwx / Float32(cols))
        dx[row, c] = rebind[dx.element_type](out.cast[x_dtype]())
        c += _RMS_TPB


def krea2_rmsnorm_backward_dx(
    go: Tensor, x: Tensor, scale: Tensor, eps: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Backward d_x for Krea2 RMSNorm's F32-internal scale.float()+1 contract.

    go/x keep their storage dtype at tensor boundaries. The raw scale tensor may
    be BF16/F16/F32; it is cast to F32 internally before adding 1.0.
    """
    if x.dtype() != go.dtype():
        raise Error("krea2_rmsnorm_backward_dx: go/x dtype mismatch")
    var xshape = x.shape()
    if len(xshape) < 1:
        raise Error("krea2_rmsnorm_backward_dx: x must have rank >= 1")
    var cols = xshape[len(xshape) - 1]
    if scale.numel() != cols:
        raise Error("krea2_rmsnorm_backward_dx: scale numel must equal features")
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]
    var dx_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, cols))
    var s_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](cols))
    var dt = x.dtype().to_mojo_dtype()
    var sdt = scale.dtype().to_mojo_dtype()
    if (
        sdt != DType.float32
        and sdt != DType.bfloat16
        and sdt != DType.float16
    ):
        raise Error("krea2_rmsnorm_backward_dx: scale dtype must be F32/BF16/F16")
    if dt == DType.float32:
        var GO = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(go.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=x_rl,
    )
        var X = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=x_rl,
    )
        var DX = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(dx_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=x_rl,
    )
        if sdt == DType.float32:
            var SC = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(scale.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=s_rl,
    )
            ctx.enqueue_function[_krea2_rmsnorm_bwd_dx_kernel[DType.float32, DType.float32]](GO, X, SC, DX, Int32(cols), eps, grid_dim=rows, block_dim=_RMS_TPB)
        elif sdt == DType.bfloat16:
            var SC = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(scale.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=s_rl,
    )
            ctx.enqueue_function[_krea2_rmsnorm_bwd_dx_kernel[DType.float32, DType.bfloat16]](GO, X, SC, DX, Int32(cols), eps, grid_dim=rows, block_dim=_RMS_TPB)
        else:
            var SC = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(scale.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=s_rl,
    )
            ctx.enqueue_function[_krea2_rmsnorm_bwd_dx_kernel[DType.float32, DType.float16]](GO, X, SC, DX, Int32(cols), eps, grid_dim=rows, block_dim=_RMS_TPB)
    elif dt == DType.bfloat16:
        var GO = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(go.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=x_rl,
    )
        var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=x_rl,
    )
        var DX = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(dx_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=x_rl,
    )
        if sdt == DType.float32:
            var SC = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(scale.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=s_rl,
    )
            ctx.enqueue_function[_krea2_rmsnorm_bwd_dx_kernel[DType.bfloat16, DType.float32]](GO, X, SC, DX, Int32(cols), eps, grid_dim=rows, block_dim=_RMS_TPB)
        elif sdt == DType.bfloat16:
            var SC = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(scale.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=s_rl,
    )
            ctx.enqueue_function[_krea2_rmsnorm_bwd_dx_kernel[DType.bfloat16, DType.bfloat16]](GO, X, SC, DX, Int32(cols), eps, grid_dim=rows, block_dim=_RMS_TPB)
        else:
            var SC = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(scale.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=s_rl,
    )
            ctx.enqueue_function[_krea2_rmsnorm_bwd_dx_kernel[DType.bfloat16, DType.float16]](GO, X, SC, DX, Int32(cols), eps, grid_dim=rows, block_dim=_RMS_TPB)
    else:
        var GO = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(go.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=x_rl,
    )
        var X = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=x_rl,
    )
        var DX = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(dx_buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=x_rl,
    )
        if sdt == DType.float32:
            var SC = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(scale.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=s_rl,
    )
            ctx.enqueue_function[_krea2_rmsnorm_bwd_dx_kernel[DType.float16, DType.float32]](GO, X, SC, DX, Int32(cols), eps, grid_dim=rows, block_dim=_RMS_TPB)
        elif sdt == DType.bfloat16:
            var SC = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(scale.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=s_rl,
    )
            ctx.enqueue_function[_krea2_rmsnorm_bwd_dx_kernel[DType.float16, DType.bfloat16]](GO, X, SC, DX, Int32(cols), eps, grid_dim=rows, block_dim=_RMS_TPB)
        else:
            var SC = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(scale.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=s_rl,
    )
            ctx.enqueue_function[_krea2_rmsnorm_bwd_dx_kernel[DType.float16, DType.float16]](GO, X, SC, DX, Int32(cols), eps, grid_dim=rows, block_dim=_RMS_TPB)
    return Tensor(dx_buf^, xshape^, x.dtype())


def krea2_swiglu_mlpdim(features: Int, multiplier: Int) -> Int:
    """SwiGLU hidden dim (mmdit.py:186-187): int(2*features/3)*multiplier rounded
    UP to a multiple of 128. For features=6144, multiplier=4 -> 4096*4=16384."""
    var mlpdim = (Int(2 * features // 3)) * multiplier
    var multiple = 128
    mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)
    return mlpdim


def krea2_swiglu(
    x: Tensor,
    gate_w: Tensor,
    up_w: Tensor,
    down_w: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """Krea2 SwiGLU (mmdit.py:180-194): down(silu(gate(x)) * up(x)), no bias.

    REUSES ops/linear (x @ Wᵀ, F32 accum, bf16 storage) for the three projections
    and ops/activations.swiglu (= silu(gate)*up elementwise) for the gated core.
    gate_w/up_w: [mlpdim, features]; down_w: [features, mlpdim] (torch Linear
    weight layout). mlpdim is taken from the weight shapes, not recomputed.
    """
    var gate = linear(x, gate_w, None, ctx)      # [..., mlpdim]
    var up = linear(x, up_w, None, ctx)          # [..., mlpdim]
    var gated = swiglu_op(gate, up, ctx)         # silu(gate) * up
    return linear(gated, down_w, None, ctx)      # [..., features]


# ── SimpleModulation (mmdit.py:109-119) ──────────────────────────────────────
# param `lin` is [2, dim] zeros; out = vec + lin[None]; chunk(2, dim=1) ->
# (scale, shift). At inference (b=1) vec [1, dim] broadcasts against lin[1,2,dim]
# -> [1, 2, dim]; scale/shift are each [1, 1, dim]. We add then slice dim=1.
def krea2_simple_modulation(
    vec: Tensor, lin: Tensor, ctx: DeviceContext
) raises -> Tuple[Tensor, Tensor]:
    """SimpleModulation.forward (mmdit.py:116-119). Returns (scale, shift).

    vec: [b, dim]   the (time) conditioning vector.
    lin: [2, dim]   the zeros-init modulation parameter.
    out = vec[:, None, :] + lin[None]  -> [b, 2, dim]; chunk along dim=1.
    Returns scale, shift each [b, 1, dim] (matching torch chunk(2, dim=1)).
    """
    var vshape = vec.shape()
    var b = vshape[0]
    var dim = vshape[len(vshape) - 1]
    # Reshape vec [b, dim] -> [b, 1, dim] so it broadcasts against lin [1, 2, dim].
    var vec3_shape = List[Int]()
    vec3_shape.append(b)
    vec3_shape.append(1)
    vec3_shape.append(dim)
    var vec3 = reshape(vec, vec3_shape^, ctx)
    # lin [2, dim] -> [1, 2, dim] for broadcast add.
    var lin3_shape = List[Int]()
    lin3_shape.append(1)
    lin3_shape.append(2)
    lin3_shape.append(dim)
    var lin3 = reshape(lin, lin3_shape^, ctx)
    var out = add(vec3, lin3, ctx)               # [b, 2, dim]
    var scale = slice(out, 1, 0, 1, ctx)         # [b, 1, dim]
    var shift = slice(out, 1, 1, 1, ctx)         # [b, 1, dim]
    return (scale^, shift^)


# ── DoubleSharedModulation (mmdit.py:122-133) ────────────────────────────────
# param `lin` is [6*dim] zeros; out = vec + lin; chunk(6, dim=-1) ->
# (prescale, preshift, pregate, postscale, postshift, postgate).
def krea2_double_shared_modulation(
    vec: Tensor, lin: Tensor, ctx: DeviceContext
) raises -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """DoubleSharedModulation.forward (mmdit.py:128-133). Returns the 6 chunks
    (prescale, preshift, pregate, postscale, postshift, postgate), each [b, dim].

    vec: [b, 6*dim]   conditioning vector.
    lin: [6*dim]      zeros-init parameter (broadcasts over the batch).
    out = vec + lin; chunk into 6 along the last dim.
    """
    var vshape = vec.shape()
    var last = len(vshape) - 1
    var sixdim = vshape[last]
    if sixdim % 6 != 0:
        raise Error("krea2_double_shared_modulation: last dim must be 6*dim")
    var dim = sixdim // 6
    var out = add(vec, lin, ctx)                 # [b, 6*dim] (lin [6*dim] broadcasts)
    var c0 = slice(out, last, 0 * dim, dim, ctx)
    var c1 = slice(out, last, 1 * dim, dim, ctx)
    var c2 = slice(out, last, 2 * dim, dim, ctx)
    var c3 = slice(out, last, 3 * dim, dim, ctx)
    var c4 = slice(out, last, 4 * dim, dim, ctx)
    var c5 = slice(out, last, 5 * dim, dim, ctx)
    return (c0^, c1^, c2^, c3^, c4^, c5^)


# ══════════════════════════════════════════════════════════════════════════════
# CHUNK 3 — krea2 Attention (GQA + QKNorm + RoPE + sigmoid-gate).
# Reference: mmdit.py Attention(197-228), QKNorm(153-160), attention()(51-63).
# ══════════════════════════════════════════════════════════════════════════════
#
# q/k/v are kept in BSHD layout [1, L, H, Dh] throughout (the serenity sdpa /
# rope_interleaved convention, matching ideogram4_dit). This is numerically
# identical to the reference's [B, H, L, D] rearrange + torch SDPA: SDPA is
# per-head, so the head axis position is immaterial to the math.


# ── Per-head RoPE table tiling for BSHD ──────────────────────────────────────
# build_krea2_rope produces a per-token table [L, half]. In BSHD [1, L, H, Dh],
# rope_interleaved flattens to rows (l*H + h), so every head h of token l must
# read table[l]. Tile [L, half] -> [L*H, half] with row (l*H + h) = table[l].
def _tile_rope_table_bshd[t_dtype: DType](
    table: LayoutTensor[t_dtype, _DYN2, MutAnyOrigin],   # [L, half]
    out_t: LayoutTensor[t_dtype, _DYN2, MutAnyOrigin],   # [L*H, half]
    L_w: Int32,
    H_w: Int32,
    half_w: Int32,
):
    var L = Int(L_w)
    var H = Int(H_w)
    var half = Int(half_w)
    var idx = Int(global_idx.x)
    var total = L * H * half
    if idx >= total:
        return
    var col = idx % half
    var rest = idx // half
    var h = rest % H
    var l = rest // H
    out_t[l * H + h, col] = rebind[out_t.element_type](table[l, col])


def _tile_rope_table(
    table: Tensor, L: Int, H: Int, half: Int, ctx: DeviceContext
) raises -> Tensor:
    """Tile a per-token RoPE table [L, half] -> [L*H, half] for BSHD apply
    (row (l*H + h) = table[l]). table dtype is preserved."""
    var dt = table.dtype().to_mojo_dtype()
    var out_n = L * H * half
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        out_n * table.dtype().byte_size()
    )
    var in_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](L, half))
    var out_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](L * H, half))
    var grid = (out_n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var T = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(table.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=in_rl,
    )
        var O = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=out_rl,
    )
        ctx.enqueue_function[_tile_rope_table_bshd[DType.float32]](T, O, Int32(L), Int32(H), Int32(half), grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var T = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(table.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=in_rl,
    )
        var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=out_rl,
    )
        ctx.enqueue_function[_tile_rope_table_bshd[DType.bfloat16]](T, O, Int32(L), Int32(H), Int32(half), grid_dim=grid, block_dim=_BLOCK)
    else:
        var T = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(table.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=in_rl,
    )
        var O = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=out_rl,
    )
        ctx.enqueue_function[_tile_rope_table_bshd[DType.float16]](T, O, Int32(L), Int32(H), Int32(half), grid_dim=grid, block_dim=_BLOCK)
    var out_shape = List[Int]()
    out_shape.append(L * H)
    out_shape.append(half)
    return Tensor(out_buf^, out_shape^, table.dtype())


def krea2_attention[L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int](
    x: Tensor,             # [1, L, features]
    wq: Tensor,            # [HEADS*HEADDIM, features]   = [6144, 6144]
    wk: Tensor,            # [KVHEADS*HEADDIM, features] = [1536, 6144]
    wv: Tensor,            # [KVHEADS*HEADDIM, features] = [1536, 6144]
    gate_w: Tensor,        # [features, features]        = [6144, 6144]
    wo: Tensor,            # [features, features]        = [6144, 6144]
    qnorm_scale: Tensor,   # [HEADDIM] raw checkpoint dtype (QKNorm.qnorm.scale)
    knorm_scale: Tensor,   # [HEADDIM] raw checkpoint dtype (QKNorm.knorm.scale)
    cos: Tensor,           # [L, HEADDIM/2]  per-token RoPE table (build_krea2_rope)
    sin: Tensor,           # [L, HEADDIM/2]
    mask: Optional[Tensor],  # None, or additive [1, HEADS, L, L] (tiled-path pad mask)
    real_len: Optional[Int], # if set -> cuDNN flash padmask path (real seq len; L is the buffer)
    ctx: DeviceContext,
) raises -> Tensor:
    """Krea2 Attention forward (mmdit.py:212-228), b==1 inference.

    L/HEADS/KVHEADS/HEADDIM are comptime (sdpa needs a comptime H/S/Dh). For the
    single_mmdit_large_wide arch: HEADS=48, KVHEADS=12, HEADDIM=128, features=6144.

    GQA: q has HEADS heads, k/v have KVHEADS; k/v are repeat_kv'd to HEADS
    (PyTorch repeat_interleave: dst head h reads kv head h//n_rep) before SDPA —
    numerically exact for torch's enable_gqa. QKNorm = krea2_rmsnorm over HEADDIM
    on q,k only (v untouched). RoPE applied to q,k (shared per-token table, tiled
    per head for BSHD). mask: None (single-stream call without padding) or the
    additive [1,HEADS,L,L] pad-to-256 mask (the main-block forward path) — when
    present it routes through the masked sdpa path (must be the q/k/v dtype).
    Output = wo(SDPA-out * sigmoid(gate)). Returns [1, L, features].
    """
    comptime heads = HEADS           # 48
    comptime kvheads = KVHEADS       # 12
    comptime headdim = HEADDIM       # 128
    comptime features = HEADS * HEADDIM  # 6144
    comptime half = HEADDIM // 2     # 64
    comptime n_rep = HEADS // KVHEADS  # 4

    # 1) Projections (no bias).
    var q = linear(x, wq, None, ctx)        # [1, L, heads*headdim]
    var k = linear(x, wk, None, ctx)        # [1, L, kvheads*headdim]
    var v = linear(x, wv, None, ctx)        # [1, L, kvheads*headdim]
    var gate = linear(x, gate_w, None, ctx) # [1, L, features]

    # 2) Reshape to BSHD [1, L, H, Dh].
    var q_shape = List[Int]()
    q_shape.append(1); q_shape.append(L); q_shape.append(heads); q_shape.append(headdim)
    q = reshape(q, q_shape^, ctx)
    var k_shape = List[Int]()
    k_shape.append(1); k_shape.append(L); k_shape.append(kvheads); k_shape.append(headdim)
    k = reshape(k, k_shape^, ctx)
    var v_shape = List[Int]()
    v_shape.append(1); v_shape.append(L); v_shape.append(kvheads); v_shape.append(headdim)
    v = reshape(v, v_shape^, ctx)

    # 3) QKNorm over headdim. krea2_rmsnorm uses F32 internal reduction/scale math
    # and returns the input storage dtype; q/k/v remain BF16 at the block boundary.
    q = krea2_rmsnorm(q, qnorm_scale, Float32(1.0e-5), ctx)
    k = krea2_rmsnorm(k, knorm_scale, Float32(1.0e-5), ctx)

    # 4) RoPE on q,k. Both share the per-token table, but q has `heads` heads
    # and k has `kvheads`, so each gets its own per-head BSHD tiling. rope_interleaved
    # applies the exact ropeapply 2x2 form (chunk-1 verified).
    var cos_q = _tile_rope_table(cos, L, heads, half, ctx)
    var sin_q = _tile_rope_table(sin, L, heads, half, ctx)
    var cos_k = _tile_rope_table(cos, L, kvheads, half, ctx)
    var sin_k = _tile_rope_table(sin, L, kvheads, half, ctx)
    var q_rot = rope_interleaved(q, cos_q, sin_q, ctx)
    var k_rot = rope_interleaved(k, cos_k, sin_k, ctx)

    # 5) GQA: repeat_kv k,v from kvheads -> heads (BSHD [1,L,kvheads,Dh]).
    # The helper name is historical; it preserves the input dtype.
    var k_full = repeat_kv_f32(k_rot, L, kvheads, n_rep, headdim, ctx)
    var v_full = repeat_kv_f32(v, L, kvheads, n_rep, headdim, ctx)

    # 6) SDPA. THREE paths (Dh=128):
    #  (a) cuDNN FLASH (real_len set): tensor-core fused flash on bf16 q/k/v — the
    #      reference's OWN backend (SDPBackend.CUDNN_ATTENTION). This is the 1024²
    #      speedup (the tiled F32 SDPA was nsys-measured at 54% of GPU time). cuDNN
    #      masks the [real_len:L] pad rows internally (replacing the additive mask).
    #      BF16 q/k/v match the reference storage boundary; cuDNN uses F32-scale
    #      attention math internally.
    #  (b) TILED + mask (no real_len): online-softmax F32 math, additive pad mask.
    #  (c) TILED nomask (no mask, no real_len): the per-op gates (chunk 3).
    var scale = Float32(1.0) / sqrt(Float32(headdim))
    var attn: Tensor
    if real_len:
        # cuDNN flash on BF16 q/k/v; mask the [real_len:L] pad rows via real_len.
        var fwd = sdpa_flash_fwd_padmask[1, L, HEADS, HEADDIM](
            q_rot, k_full, v_full, real_len.value(), scale, ctx
        )
        attn = fwd.o.clone(ctx)
    elif mask:
        attn = sdpa_tiled[1, L, HEADS, HEADDIM](q_rot, k_full, v_full, mask.value(), scale, ctx)
    else:
        attn = sdpa_nomask_tiled[1, L, HEADS, HEADDIM](q_rot, k_full, v_full, scale, ctx)

    # 7) Match the BF16 product flow: SDPA stores the q/k/v dtype, gate multiply
    # stores that dtype, and wo uses BF16 inputs/weights with F32 GEMM accumulation.
    var merge_shape = List[Int]()
    merge_shape.append(1); merge_shape.append(L); merge_shape.append(features)
    var merged = reshape(attn, merge_shape^, ctx)
    var g = sigmoid(gate, ctx)
    var gated = mul(merged, g, ctx)
    return linear(gated, wo, None, ctx)


# ══════════════════════════════════════════════════════════════════════════════
# CHUNK 4 — krea2 SingleStreamBlock (composes chunks 2+3).
# Reference: mmdit.py SingleStreamBlock (312-337).
# ══════════════════════════════════════════════════════════════════════════════
#
# forward(x, vec, freqs, mask=None)  (mmdit.py:328-337):
#   prescale,preshift,pregate,postscale,postshift,postgate = self.mod(vec)
#   x = x + pregate  * self.attn((1+prescale )*self.prenorm (x) + preshift,  freqs, mask)
#   x = x + postgate * self.mlp ((1+postscale)*self.postnorm(x) + postshift)
# self.mod=DoubleSharedModulation (chunk 2); prenorm/postnorm=krea2_rmsnorm(features);
# self.attn=krea2_attention (chunk 3); self.mlp=krea2_swiglu (chunk 2).
#
# AdaLN broadcast: prescale/preshift/pregate (etc.) are per-channel [features]
# vectors broadcast over the L token axis. ops/elementwise.modulate and
# residual_gate apply a [D] param per-channel over ALL leading rows — exactly the
# AdaLN-over-tokens form — so they are REUSED directly (no broadcast variant
# needed; verified the kernels index param[c] for every (row,c)).
# The +1 reparam lives in modulate's (1+scale); the chunks are RAW (no +1) — so
# we pass the raw modulation chunks straight to modulate (no double-add).


def _reshape_chunk_to_vec(
    chunk: Tensor, features: Int, ctx: DeviceContext
) raises -> Tensor:
    """Reshape a modulation chunk [1, features] -> [features] (a clean [D] param
    for modulate/residual_gate). (b==1 inference.)"""
    var s = List[Int]()
    s.append(features)
    return reshape(chunk, s^, ctx)


def krea2_single_stream_block[L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int](
    x: Tensor,             # [1, L, features]
    vec: Tensor,           # [1, 6*features]  (tproj(t); chunk 5)
    mod_lin: Tensor,       # [6*features]     (DoubleSharedModulation.lin)
    prenorm_scale: Tensor, # [features] F32   (prenorm.scale)
    postnorm_scale: Tensor,# [features] F32   (postnorm.scale)
    wq: Tensor, wk: Tensor, wv: Tensor, gate_w: Tensor, wo: Tensor,  # attn proj
    qnorm_scale: Tensor, knorm_scale: Tensor,                        # attn QKNorm [128] F32
    mlp_gate_w: Tensor, mlp_up_w: Tensor, mlp_down_w: Tensor,        # SwiGLU
    cos: Tensor, sin: Tensor,                                        # rope table [L, headdim/2]
    mask: Optional[Tensor],                                          # None or additive [1,HEADS,L,L] (tiled path)
    real_len: Optional[Int],                                         # if set -> cuDNN flash padmask path
    ctx: DeviceContext,
) raises -> Tensor:
    """Krea2 SingleStreamBlock forward (mmdit.py:328-337), b==1.

    Composes chunk-2 (DoubleSharedModulation, RMSNorm, SwiGLU) + chunk-3
    (Attention). vec is the timestep-derived [1, 6*features] modulation vector;
    its 6 raw chunks gate the two AdaLN-Zero residual branches. mask: None, or the
    additive [1,HEADS,L,L] pad-to-256 mask (tiled-path forward). real_len: if set,
    the attention uses the cuDNN FLASH padmask path (the 1024² speedup) with cuDNN
    masking the [real_len:L] pad rows. Returns [1, L, features].
    """
    comptime features = HEADS * HEADDIM   # 6144

    # mod(vec) -> 6 raw chunks, each [1, features].
    var mods = krea2_double_shared_modulation(vec, mod_lin, ctx)
    var prescale = _reshape_chunk_to_vec(mods[0], features, ctx)
    var preshift = _reshape_chunk_to_vec(mods[1], features, ctx)
    var pregate = _reshape_chunk_to_vec(mods[2], features, ctx)
    var postscale = _reshape_chunk_to_vec(mods[3], features, ctx)
    var postshift = _reshape_chunk_to_vec(mods[4], features, ctx)
    var postgate = _reshape_chunk_to_vec(mods[5], features, ctx)

    # Attention branch: x = x + pregate * attn((1+prescale)*prenorm(x) + preshift).
    var xn = krea2_rmsnorm(x, prenorm_scale, Float32(1.0e-5), ctx)        # [1,L,features]
    var xm = modulate(xn, prescale, preshift, ctx)                       # (1+prescale)*xn + preshift
    var a = krea2_attention[L, HEADS, KVHEADS, HEADDIM](
        xm, wq, wk, wv, gate_w, wo, qnorm_scale, knorm_scale, cos, sin, mask, real_len, ctx
    )
    var x1 = residual_gate(x, pregate, a, ctx)                          # x + pregate*a

    # MLP branch: x = x + postgate * mlp((1+postscale)*postnorm(x) + postshift).
    var xn2 = krea2_rmsnorm(x1, postnorm_scale, Float32(1.0e-5), ctx)
    var xm2 = modulate(xn2, postscale, postshift, ctx)
    var m = krea2_swiglu(xm2, mlp_gate_w, mlp_up_w, mlp_down_w, ctx)
    var x2 = residual_gate(x1, postgate, m, ctx)
    return x2^


# ══════════════════════════════════════════════════════════════════════════════
# CHUNK 5 — embedders + input/output heads.
# Reference: mmdit.py temb(71-88), tmlp(374-378), tproj(395-397),
# txtmlp(387-392), first(358-360), LastLayer(231-242).
# ══════════════════════════════════════════════════════════════════════════════


def krea2_temb(
    t: Tensor, dim: Int, ctx: DeviceContext, out_dtype: STDtype
) raises -> Tensor:
    """Sinusoidal timestep embedding (mmdit.py:71-88). dim=tdim=256.

        half   = dim/2 = 128
        freqs  = exp(-log(1e4) * arange(half)/half)
        args   = (t * 1e3) * freqs          # tfactor=1e3 PRE-SCALE on t
        return cat(cos(args), sin(args), -1) # cos-FIRST, then sin

    REUSES ops/embeddings.timestep_embedding (cos-first, max_period arg) — its
    math is `angle = t_in * freq` with freq = exp(-log(max_period)*i/half), so we
    pass t_in = t * tfactor (=1e3) to fold in the pre-scale. period=1e4 -> the
    max_period arg. The cos-then-sin concat order matches exactly.
    t: [B] (any dtype). Returns [B, dim] (out_dtype). (Reference's extra unit dims
    are layout-only; the caller reshapes to [B,1,dim] as needed.)
    """
    var t_scaled = mul_scalar(t, Float32(1.0e3), ctx)   # tfactor pre-scale
    return timestep_embedding(t_scaled, dim, ctx, Float32(1.0e4), out_dtype)


def krea2_tmlp(
    temb: Tensor,
    w1: Tensor, b1: Tensor,    # Linear(tdim -> features)  (bias=True)
    w2: Tensor, b2: Tensor,    # Linear(features -> features)
    ctx: DeviceContext,
) raises -> Tensor:
    """Tmlp (mmdit.py:374-378): Linear(256->6144) -> GELU(tanh) -> Linear(6144->6144).
    temb [B,1,256] (or [B,256]) -> t [..., features]. Both Linears have bias."""
    var h = linear(temb, w1, Optional[Tensor](b1.clone(ctx)), ctx)
    var hg = gelu(h, ctx)
    return linear(hg, w2, Optional[Tensor](b2.clone(ctx)), ctx)


def krea2_tproj(
    t: Tensor, w: Tensor, b: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """Tproj (mmdit.py:395-397): GELU(tanh) -> Linear(features -> 6*features).
    t [..., features] -> vec [..., 6*features]. Linear has bias."""
    var tg = gelu(t, ctx)
    return linear(tg, w, Optional[Tensor](b.clone(ctx)), ctx)


def krea2_txtmlp(
    context: Tensor,
    rms_scale: Tensor,         # [txtdim] F32  (RMSNorm.scale)
    w1: Tensor, b1: Tensor,    # Linear(txtdim -> features)
    w2: Tensor, b2: Tensor,    # Linear(features -> features)
    ctx: DeviceContext,
) raises -> Tensor:
    """Txtmlp (mmdit.py:387-392): RMSNorm(2560) -> Linear(2560->6144) ->
    GELU(tanh) -> Linear(6144->6144). context [1,L,txtdim] -> [1,L,features].
    RMSNorm = krea2_rmsnorm (F32-internal, scale+1)."""
    var cn = krea2_rmsnorm(context, rms_scale, Float32(1.0e-5), ctx)
    var h = linear(cn, w1, Optional[Tensor](b1.clone(ctx)), ctx)
    var hg = gelu(h, ctx)
    return linear(hg, w2, Optional[Tensor](b2.clone(ctx)), ctx)


def krea2_first(
    x: Tensor, w: Tensor, b: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """First (mmdit.py:358-360): Linear(channels*patch^2 -> features), bias=True.
    Patchified latent [1, N, channels*patch^2 = 64] -> [1, N, features]."""
    return linear(x, w, Optional[Tensor](b.clone(ctx)), ctx)


def krea2_last_layer(
    x: Tensor,             # [1, L, features]
    tvec: Tensor,          # [1, 1, features]  (= t, the tmlp output — NOT tproj's vec)
    norm_scale: Tensor,    # [features] F32   (LastLayer.norm.scale)
    mod_lin: Tensor,       # [2, features]    (SimpleModulation.lin)
    lin_w: Tensor,         # [patch^2*channels, features]  = [64, 6144]
    lin_b: Tensor,         # [patch^2*channels]  = [64]
    features: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """LastLayer (mmdit.py:231-242). forward(x, tvec):
        scale, shift = SimpleModulation(tvec)
        x = (1 + scale) * RMSNorm(x) + shift
        x = Linear(x)                            # bias=True
    tvec = t (tmlp output [1,1,features]), NOT tproj's vec. Returns [1, L, 64].
    SimpleModulation generalizes b>1, but LastLayer (like the whole inference
    path) is b==1; scale/shift come out [1,1,features] -> reshaped to [features]
    for modulate's per-channel broadcast over the L tokens.
    """
    var mods = krea2_simple_modulation(tvec, mod_lin, ctx)  # (scale, shift) each [1,1,features]
    var scale = _reshape_chunk_to_vec(mods[0], features, ctx)  # [features]
    var shift = _reshape_chunk_to_vec(mods[1], features, ctx)
    var xn = krea2_rmsnorm(x, norm_scale, Float32(1.0e-5), ctx)  # [1,L,features]
    var xm = modulate(xn, scale, shift, ctx)                     # (1+scale)*xn + shift
    return linear(xm, lin_w, Optional[Tensor](lin_b.clone(ctx)), ctx)      # [1, L, 64]


# ══════════════════════════════════════════════════════════════════════════════
# CHUNK 6 — TextFusionTransformer (processes Qwen3-VL context before the blocks).
# Reference: mmdit.py TextFusionBlock (245-264), TextFusionTransformer (267-309),
# _mask (66-68). Attention here is NO-rope, NO-GQA (heads==kvheads).
# ══════════════════════════════════════════════════════════════════════════════
#
# SHAPE CONTRACT (derived by RUNNING the reference — the forward's local names are
# MISLEADING): context fed in is [B, Lt, n=txtlayers=12, d=2560] (pipeline.py
# predict_velocity:111-115). In TextFusionTransformer.forward the locals are
# `b, l, n, d = x.shape` so l=Lt (TOKENS), n=12 (LAYERS).
#   reshape(b*l, n, d)        -> [B*Lt, 12, d]     layerwise attends over 12 LAYERS
#   2x layerwise blocks (mask=None)
#   rearrange (b l) n d -> b l d n  -> [B, Lt, d, 12]
#   reshape(b*l, d, n)        -> [B*Lt, d, 12]
#   projector Linear(12->1)   -> [B*Lt, d, 1]      collapses the 12-LAYER axis
#   reshape(b, l, d)          -> [B, Lt, d]
#   2x refiner blocks (mask=txtmask)  attends over Lt TOKENS, masked
#
# _mask (66-68): keep-vector [B,Lt] (BOOL) -> [B,1,Lt,Lt] = keep[i] & keep[j]
# (bool outer product). The reference passes this BOOL mask to F.sdpa, which treats
# a BOOLEAN attn_mask as KEEP/MASK-OUT: True (both real) -> attend, False (either
# padded) -> score set to -inf (the position is masked OUT, not softly biased).
# MEASURED (2026-06-24): the reference's main-block mask is bool (mmdit.py:441
# `mask = _mask(mask)` with a bool padded keep), and additive `-1e4` on pad keys
# reproduces the bool reference EXACTLY (cos 1.0 on real block-0 attn). The earlier
# "+1 additive" reading was WRONG — it only matched chunk-6's gen which fed a FLOAT
# keep (float outer product -> additive), not the bool production path. We build
# the additive equivalent: 0.0 where both keep, -1e4 (bf16-safe, not -inf -> no NaN)
# where either is padded. Feed it as ops/attention.sdpa's additive [B,H,Lt,Lt] mask.
comptime _MASK_NEG = Float32(-1.0e4)  # additive pad penalty (bf16-safe stand-in for -inf)


# ── text key-padding mask -> additive [H, Lt, Lt] (B=1): 0 if both keep else -1e4 ─
def _krea2_text_mask_kernel[out_dtype: DType](
    keep: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [Lt]  (1.0 keep / 0.0 pad)
    out_m: LayoutTensor[out_dtype, _DYN2, MutAnyOrigin],      # [H*Lt, Lt]  (additive)
    H_w: Int32,
    Lt_w: Int32,
):
    var H = Int(H_w)
    var Lt = Int(Lt_w)
    var idx = Int(global_idx.x)
    var total = H * Lt * Lt
    if idx >= total:
        return
    var j = idx % Lt
    var rest = idx // Lt
    var i = rest % Lt
    var ki = rebind[Scalar[DType.float32]](keep[i])
    var kj = rebind[Scalar[DType.float32]](keep[j])
    # bool keep[i] AND keep[j] -> 0.0 (attend); else -1e4 (mask out).
    var v = Float32(0.0) if (ki * kj) > Float32(0.0) else _MASK_NEG
    out_m[rest, j] = rebind[out_m.element_type](v.cast[out_dtype]())


def build_krea2_text_mask(
    keep: Tensor, H: Int, Lt: Int, ctx: DeviceContext, out_dtype: STDtype
) raises -> Tensor:
    """Build the additive attention mask (reference _mask 66-68, BOOL semantics).

    keep: [Lt] F32 (1.0 = real token, 0.0 = padded). Returns [1, H, Lt, Lt] in
    out_dtype, additive mask m[i,j] = 0.0 if (keep[i] AND keep[j]) else -1e4. The
    reference _mask builds a BOOL keep[i]&keep[j] mask that F.sdpa renders as
    -inf-masking on padded positions; -1e4 is the bf16-safe additive equivalent
    (reproduces the bool reference cos 1.0). Broadcast over H for ops/attention.sdpa.

    out_dtype MUST match the q/k/v dtype: ops/attention.sdpa enforces
    q.dtype()==mask.dtype(). 0.0 and -1e4 are bf16-representable. The masked sdpa
    softmax accumulates F32 regardless of mask storage. Pass STDtype.BF16 for the
    bf16 inference/training path (chunk 7's pad-to-256 mask), STDtype.F32 only for
    an F32 model.
    """
    if keep.dtype() != STDtype.F32:
        raise Error("build_krea2_text_mask: keep must be F32")
    var out_n = H * Lt * Lt
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        out_n * out_dtype.byte_size()
    )
    var k_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](Lt))
    var m_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](H * Lt, Lt))
    var K = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(keep.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=k_rl,
    )
    var grid = (out_n + _BLOCK - 1) // _BLOCK
    var odt = out_dtype.to_mojo_dtype()
    if odt == DType.float32:
        var M = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=m_rl,
    )
        ctx.enqueue_function[_krea2_text_mask_kernel[DType.float32]](K, M, Int32(H), Int32(Lt), grid_dim=grid, block_dim=_BLOCK)
    elif odt == DType.bfloat16:
        var M = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=m_rl,
    )
        ctx.enqueue_function[_krea2_text_mask_kernel[DType.bfloat16]](K, M, Int32(H), Int32(Lt), grid_dim=grid, block_dim=_BLOCK)
    else:
        var M = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=m_rl,
    )
        ctx.enqueue_function[_krea2_text_mask_kernel[DType.float16]](K, M, Int32(H), Int32(Lt), grid_dim=grid, block_dim=_BLOCK)
    var shape = List[Int]()
    shape.append(1); shape.append(H); shape.append(Lt); shape.append(Lt)
    return Tensor(out_buf^, shape^, out_dtype)


# ── krea2_mha — no-rope, no-GQA multi-head attention (text path) ──────────────
# Same structure as krea2_attention MINUS rope and MINUS repeat_kv (heads==kvheads).
# QKNorm (over headdim), sigmoid-gate, and wo are STILL present. Optional additive
# mask (None for layerwise, the refiner txtmask for refiner blocks).
def krea2_mha[B: Int, S: Int, HEADS: Int, HEADDIM: Int](
    x: Tensor,             # [B, S, features]
    wq: Tensor, wk: Tensor, wv: Tensor, gate_w: Tensor, wo: Tensor,
    qnorm_scale: Tensor, knorm_scale: Tensor,   # [HEADDIM] F32
    mask: Optional[Tensor],                     # None, or additive [B,HEADS,S,S] F32
    ctx: DeviceContext,
) raises -> Tensor:
    """Krea2 text-path Attention (mmdit.py Attention with freqs=None, gqa=False).

    No RoPE (freqs is None in the TextFusionBlock call), no GQA (heads==kvheads),
    so q/k/v all have HEADS heads. QKNorm over HEADDIM on q,k; sigmoid-gate; wo.
    B is comptime (layerwise batches over the LT tokens with B=LT, S=NLAYERS;
    refiner runs B=1, S=LT). mask: additive [B,HEADS,S,S] (refiner) or None.
    Returns [B, S, features].
    """
    comptime features = HEADS * HEADDIM

    var q = linear(x, wq, None, ctx)         # [B,S,features]
    var k = linear(x, wk, None, ctx)
    var v = linear(x, wv, None, ctx)
    var gate = linear(x, gate_w, None, ctx)

    var q_shape = List[Int]()
    q_shape.append(B); q_shape.append(S); q_shape.append(HEADS); q_shape.append(HEADDIM)
    q = reshape(q, q_shape^, ctx)
    var k_shape = List[Int]()
    k_shape.append(B); k_shape.append(S); k_shape.append(HEADS); k_shape.append(HEADDIM)
    k = reshape(k, k_shape^, ctx)
    var v_shape = List[Int]()
    v_shape.append(B); v_shape.append(S); v_shape.append(HEADS); v_shape.append(HEADDIM)
    v = reshape(v, v_shape^, ctx)

    # QKNorm over headdim (q,k only); v untouched. No rope, no repeat_kv.
    q = krea2_rmsnorm(q, qnorm_scale, Float32(1.0e-5), ctx)
    k = krea2_rmsnorm(k, knorm_scale, Float32(1.0e-5), ctx)

    var scale = Float32(1.0) / sqrt(Float32(HEADDIM))
    var attn: Tensor
    if mask:
        attn = sdpa[B, S, HEADS, HEADDIM](q, k, v, mask.value(), scale, ctx)  # [B,S,H,Dh]
    else:
        # Small-S no-mask (the LAYERWISE txtfusion shape: B=LT, S=NLAYERS=12):
        # ONE fused kernel over the B*H grid instead of _sdpa_math_storage's
        # 2·B·H per-(b,h) GEMM launches (7680+7680 tiny launch-bound kernels
        # per call — the conditioning hotspot, nsys 2026-07-07). Parity-gated
        # vs sdpa_nomask (ops/tests/sdpa_small_parity.mojo, cos>=0.9999).
        comptime if S <= 16 and HEADDIM <= 128:
            if q.dtype() == STDtype.BF16:
                attn = sdpa_nomask_small[B, S, HEADS, HEADDIM](q, k, v, scale, ctx)
            else:
                attn = sdpa_nomask[B, S, HEADS, HEADDIM](q, k, v, scale, ctx)
        else:
            attn = sdpa_nomask[B, S, HEADS, HEADDIM](q, k, v, scale, ctx)

    var merge_shape = List[Int]()
    merge_shape.append(B); merge_shape.append(S); merge_shape.append(features)
    var merged = reshape(attn, merge_shape^, ctx)
    var g = sigmoid(gate, ctx)
    var gated = mul(merged, g, ctx)
    return linear(gated, wo, None, ctx)


# ── krea2_text_fusion_block (mmdit.py TextFusionBlock 245-264) ────────────────
def krea2_text_fusion_block[B: Int, S: Int, HEADS: Int, HEADDIM: Int](
    x: Tensor,             # [B, S, txtdim]
    prenorm_scale: Tensor, postnorm_scale: Tensor,   # [txtdim] F32
    wq: Tensor, wk: Tensor, wv: Tensor, gate_w: Tensor, wo: Tensor,
    qnorm_scale: Tensor, knorm_scale: Tensor,        # [HEADDIM] F32
    mlp_gate_w: Tensor, mlp_up_w: Tensor, mlp_down_w: Tensor,
    mask: Optional[Tensor],
    ctx: DeviceContext,
) raises -> Tensor:
    """TextFusionBlock forward (mmdit.py:260-264):
        x = x + attn(prenorm(x), mask=mask)     # attn = krea2_mha (no rope/GQA)
        x = x + mlp(postnorm(x))                # mlp = krea2_swiglu
    NOTE: plain residual ADD (no AdaLN gate — TextFusionBlock has no modulation).
    """
    var xn = krea2_rmsnorm(x, prenorm_scale, Float32(1.0e-5), ctx)
    var a = krea2_mha[B, S, HEADS, HEADDIM](
        xn, wq, wk, wv, gate_w, wo, qnorm_scale, knorm_scale, mask, ctx
    )
    var x1 = add(x, a, ctx)
    var xn2 = krea2_rmsnorm(x1, postnorm_scale, Float32(1.0e-5), ctx)
    var m = krea2_swiglu(xn2, mlp_gate_w, mlp_up_w, mlp_down_w, ctx)
    return add(x1, m, ctx)


# ── krea2_text_fusion (mmdit.py TextFusionTransformer 267-309) ────────────────
@fieldwise_init
struct Krea2TextFusionWeights(Copyable, Movable):
    """The weight bundle for ONE TextFusionBlock (layerwise or refiner).

    All Tensors are ArcPointer-shared so the struct is Copyable/Movable (Tensor
    itself is move-only). prenorm/postnorm/qnorm/knorm scales are F32; the
    projections are storage-dtype."""

    var prenorm: ArcPointer[Tensor]
    var postnorm: ArcPointer[Tensor]
    var wq: ArcPointer[Tensor]
    var wk: ArcPointer[Tensor]
    var wv: ArcPointer[Tensor]
    var gate_w: ArcPointer[Tensor]
    var wo: ArcPointer[Tensor]
    var qnorm: ArcPointer[Tensor]
    var knorm: ArcPointer[Tensor]
    var mlp_gate: ArcPointer[Tensor]
    var mlp_up: ArcPointer[Tensor]
    var mlp_down: ArcPointer[Tensor]


def _run_text_fusion_block[B: Int, S: Int, HEADS: Int, HEADDIM: Int](
    x: Tensor, w: Krea2TextFusionWeights, mask: Optional[Tensor], ctx: DeviceContext
) raises -> Tensor:
    """Run one TextFusionBlock from a weight bundle (thin wrapper over
    krea2_text_fusion_block)."""
    return krea2_text_fusion_block[B, S, HEADS, HEADDIM](
        x,
        w.prenorm[], w.postnorm[],
        w.wq[], w.wk[], w.wv[], w.gate_w[], w.wo[],
        w.qnorm[], w.knorm[],
        w.mlp_gate[], w.mlp_up[], w.mlp_down[],
        mask, ctx,
    )


def krea2_text_fusion[LT: Int, NLAYERS: Int, HEADS: Int, HEADDIM: Int](
    context: Tensor,             # [1, LT, NLAYERS, txtdim]  (B=1; NLAYERS=12)
    layerwise0: Krea2TextFusionWeights,
    layerwise1: Krea2TextFusionWeights,
    projector_w: Tensor,         # [1, NLAYERS]  (Linear(NLAYERS -> 1), no bias)
    refiner0: Krea2TextFusionWeights,
    refiner1: Krea2TextFusionWeights,
    refiner_mask: Optional[Tensor],   # additive [1,HEADS,LT,LT] (refiner txtmask) or None
    ctx: DeviceContext,
) raises -> Tensor:
    """TextFusionTransformer forward (mmdit.py:294-309), B==1.

    context [1, LT, NLAYERS=12, txtdim]: LT = caption tokens, NLAYERS = 12 stacked
    Qwen3-VL layers, txtdim=2560. The 2 layerwise blocks attend over the 12 LAYERS
    (seq=NLAYERS, batched over the LT tokens), the projector Linear(12->1) collapses
    the layer axis, and the 2 refiner blocks attend over the LT TOKENS (seq=LT) with
    the txtmask. Returns [1, LT, txtdim].
    """
    var cshape = context.shape()
    var txtdim = cshape[len(cshape) - 1]

    # reshape [1, LT, NLAYERS, d] -> [LT, NLAYERS, d]  (reference reshape(b*l, n, d);
    # b*l = 1*LT = LT). Layerwise blocks attend over NLAYERS, batched over LT.
    var lw_shape = List[Int]()
    lw_shape.append(LT); lw_shape.append(NLAYERS); lw_shape.append(txtdim)
    var x = reshape(context, lw_shape^, ctx)   # [LT, NLAYERS, d]

    # 2 layerwise blocks: B=LT, S=NLAYERS, mask=None (block-diagonal over LT — the
    # SDPA's batch axis keeps each token's 12-layer attention independent, exactly
    # the reference's batched [b*l, n, d] call).
    x = _run_text_fusion_block[LT, NLAYERS, HEADS, HEADDIM](x, layerwise0, None, ctx)
    x = _run_text_fusion_block[LT, NLAYERS, HEADS, HEADDIM](x, layerwise1, None, ctx)

    # projector: collapse the NLAYERS axis (mmdit.py:299-304).
    #   rearrange (b l) n d -> b l d n ; reshape(b*l, d, n) ; Linear(n=NLAYERS -> 1)
    # Equivalent batched: x is [LT, NLAYERS, d]; transpose last two -> [LT, d, NLAYERS];
    # Linear acts on the last dim (NLAYERS) -> [LT, d, 1]; reshape -> [1, LT, d].
    var xt = transpose(x, 1, 2, ctx)                 # [LT, d, NLAYERS]
    var proj = linear(xt, projector_w, None, ctx)    # [LT, d, 1]
    var seq_shape = List[Int]()
    seq_shape.append(1); seq_shape.append(LT); seq_shape.append(txtdim)
    var xr = reshape(proj, seq_shape^, ctx)          # [1, LT, d]

    # 2 refiner blocks: B=1, S=LT, with the refiner txtmask.
    xr = _run_text_fusion_block[1, LT, HEADS, HEADDIM](xr, refiner0, refiner_mask, ctx)
    xr = _run_text_fusion_block[1, LT, HEADS, HEADDIM](xr, refiner1, refiner_mask, ctx)
    return xr^


# ══════════════════════════════════════════════════════════════════════════════
# CHUNK 7a — krea2_forward = SingleStreamDiT.forward (WIRING, resident).
# Reference: mmdit.py SingleStreamDiT.forward (413-461).
# ══════════════════════════════════════════════════════════════════════════════


# ── checkpoint weight loaders (mixed-precision raw.safetensors → bf16 runtime) ─
# The real Krea-2-Raw checkpoint is MIXED PRECISION on disk: block matmul weights
# are BF16 but the embedders/heads/norms/mod (`first`, `tmlp/tproj/txtmlp`, `last`,
# `projector`, every `.scale`, every `mod.lin`) are F32. The reference casts ALL
# floating-point params to bf16 at load (krea2.py:190 `v.to("bf16")`), so the model
# runs bf16 throughout — which is what chunks 1-7 gated against. We must do the same:
#   _wb(...)  loads ANY float weight as BF16 (F32/F16->bf16; bf16 is a no-op) — the
#             reference's `v.to(bf16)`. Used for every matmul weight + mod.lin + the
#             linears/embedders (already bf16-loaded).
#   _scale(...) loads a norm `.scale` as BF16 storage. krea2_rmsnorm casts scale to
#             F32 internally and applies the reference `self.scale.float() + 1.0`.
def _wb(st: ShardedSafeTensors, key: String, ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_view_as_bf16(st.tensor_view(key), ctx)


# ── _wb + LoRA OVERLAY ([[feedback_lora_never_fused]]): load the bf16 base weight,
# then if `lora` carries an adapter for `base_key`, ADD scale·(B@A) in-memory.
# OVERLAY only — the saved checkpoint is NEVER modified; the delta is added to the
# freshly-streamed per-block weight, freed when the block iteration ends. base_key
# is the krea2 base weight key (e.g. blocks.0.attn.wq.weight); the LoraSet resolved
# the PEFT prefix diffusion_model.blocks.0.attn.wq → blocks.0.attn.wq.weight at load.
# Apply the LoRA overlay to an already-loaded base weight `w` (W += scale·BA when
# the LoraSet has an adapter for base_key). Shared by the disk-stream (_wb_lora)
# and the fp8-resident (_blk_w8) paths so the overlay math is identical either way.
def _apply_lora(
    var w: Tensor, base_key: String,
    lora: Optional[LoraSet], multiplier: Float32, ctx: DeviceContext,
) raises -> Tensor:
    if not lora:
        return w^
    ref ls = lora.value()
    for ref m in ls.mappings:
        if m.base_key == base_key:
            var scale = ls._module_scale(m, multiplier, ctx)
            var delta = ls._compute_delta(m, scale, w.dtype(), ctx)  # [out,in] in w's dtype
            return add(w, delta, ctx)
    return w^


def _wb_lora(
    st: ShardedSafeTensors, key: String, base_key: String,
    lora: Optional[LoraSet], multiplier: Float32, ctx: DeviceContext,
) raises -> Tensor:
    return _apply_lora(_wb(st, key, ctx), base_key, lora, multiplier, ctx)


# ── per-block weight SOURCE helpers: fp8-resident dequant (NO disk) when `resident`
# is present, else disk-stream. _blk_w8 = the k-th matmul weight (+ LoRA overlay);
# _blk_scale = the k-th F32 norm scale (0=qnorm 1=knorm 2=prenorm 3=postnorm);
# _blk_modlin = mod.lin. The resident tensors match the stream dtypes exactly
# (matmul→bf16 dequant, scales→F32, mod_lin→bf16), so the no-LoRA result is the
# fp8-quantized base (~0.99 cos vs the bf16-streamed base — the documented tradeoff).
def _blk_w8(
    resident: Optional[Krea2ResidentFp8], li: Int, k: Int,
    st: ShardedSafeTensors, stream_key: String, base_key: String,
    lora: Optional[LoraSet], multiplier: Float32, ctx: DeviceContext,
) raises -> Tensor:
    if resident:
        ref b = resident.value().blocks[li]
        var w = fp8_e4m3_dequant_perrow_to_bf16(b.fp8[k][], b.scale[k][], ctx)
        return _apply_lora(w^, base_key, lora, multiplier, ctx)
    return _wb_lora(st, stream_key, base_key, lora, multiplier, ctx)


def _blk_scale(
    resident: Optional[Krea2ResidentFp8], li: Int, k: Int,
    st: ShardedSafeTensors, stream_key: String, ctx: DeviceContext,
) raises -> Tensor:
    if resident:
        ref b = resident.value().blocks[li]
        if k == 0:
            return b.qnorm_scale[].clone(ctx)
        elif k == 1:
            return b.knorm_scale[].clone(ctx)
        elif k == 2:
            return b.prenorm_scale[].clone(ctx)
        return b.postnorm_scale[].clone(ctx)
    return _scale(st, stream_key, ctx)


def _blk_modlin(
    resident: Optional[Krea2ResidentFp8], li: Int,
    st: ShardedSafeTensors, stream_key: String, ctx: DeviceContext,
) raises -> Tensor:
    if resident:
        return resident.value().blocks[li].mod_lin[].clone(ctx)
    return _wb(st, stream_key, ctx)


def _scale(st: ShardedSafeTensors, key: String, ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_view_as_bf16(st.tensor_view(key), ctx)


# ── int8 W8A8-RESIDENT inference path (task #11: fast krea2 EDIT inference) ─────
# Mirrors the TRAINER's int8 base-matmul path EXACTLY (krea2_block._base_fwd /
# _linear_lora → ops/int8_linear.int8_linear_fwd: tensorwise int8 weight scale +
# per-token int8 activation quant, int8×int8→int32 GEMM, bf16 out — == reference trainer
# LinearW8A8, the dtype path the edit adapter was TRAINED against). LoRA here is
# a bf16 SIDE-BRANCH added to the int8 GEMM output — the same math training ran
# (_linear_lora: base int8 + scale·((x@Aᵀ)@Bᵀ)) — NOT a weight overlay (there is
# no bf16 weight to overlay onto on this path).

# DEVICE-RESIDENT LoRA side cache: A/B factors + folded per-module scale, loaded
# ONCE at startup. The naive per-call variant (mmap A/B + a `.alpha` file probe
# per module per block per forward = 448 loads + hidden syncs PER STEP) was
# MEASURED to dominate the int8 step (~4s of the 5.4s). base_keys parallel a/b/s.
struct Krea2LoraSideCache(Copyable, Movable):
    var base_keys: List[String]           # resolved bare keys (blocks.<li>.<mod>.weight)
    var a: List[ArcPointer[Tensor]]       # [rank, in]  bf16 device
    var b: List[ArcPointer[Tensor]]       # [out, rank] bf16 device
    var scale: List[Float32]              # (alpha/rank)*multiplier, folded

    def __init__(
        out self, var base_keys: List[String],
        var a: List[ArcPointer[Tensor]], var b: List[ArcPointer[Tensor]],
        var scale: List[Float32],
    ):
        self.base_keys = base_keys^
        self.a = a^
        self.b = b^
        self.scale = scale^


# Build the device-resident side cache from a loaded LoraSet (multiplier folded
# into the per-module scale — == _module_scale semantics, computed ONCE).
def build_krea2_lora_side_cache(
    lora: LoraSet, multiplier: Float32, ctx: DeviceContext
) raises -> Krea2LoraSideCache:
    var base_keys = List[String]()
    var a_l = List[ArcPointer[Tensor]]()
    var b_l = List[ArcPointer[Tensor]]()
    var s_l = List[Float32]()
    for ref m in lora.mappings:
        var sc = lora._module_scale(m, multiplier, ctx)
        var a = lora._load_lora_tensor(m.prefix + lora.suffix_a, ctx)  # [rank, in]
        var b = lora._load_lora_tensor(m.prefix + lora.suffix_b, ctx)  # [out, rank]
        if a.dtype() != STDtype.BF16:
            a = cast_tensor(a^, STDtype.BF16, ctx)
        if b.dtype() != STDtype.BF16:
            b = cast_tensor(b^, STDtype.BF16, ctx)
        base_keys.append(m.base_key)
        a_l.append(ArcPointer[Tensor](a^))
        b_l.append(ArcPointer[Tensor](b^))
        s_l.append(sc)
    ctx.synchronize()
    return Krea2LoraSideCache(base_keys^, a_l^, b_l^, s_l^)


# One LoRA side-branch delta scale·((x@Aᵀ)@Bᵀ) for base_key from the CACHE, or
# None. Numerically the overlay's x@(scale·B@A)ᵀ term, but computed as two thin
# rank-sized GEMMs instead of materializing the [out,in]=6144² delta each block —
# and with zero per-call file access (the resident A/B).
def _lora_side(
    x: Tensor, base_key: String,
    cache: Optional[Krea2LoraSideCache], ctx: DeviceContext,
) raises -> Optional[Tensor]:
    if not cache:
        return Optional[Tensor](None)
    ref c = cache.value()
    for i in range(len(c.base_keys)):
        if c.base_keys[i] == base_key:
            var t = linear(x, c.a[i][], None, ctx)     # x@Aᵀ  [..., rank]
            var d = linear(t, c.b[i][], None, ctx)     # t@Bᵀ  [..., out]
            return Optional[Tensor](mul_scalar(d, c.scale[i], ctx))
    return Optional[Tensor](None)


# int8 base GEMM + optional LoRA side-branch: y = int8_fwd(x, w8[k]) + Δ_lora.
# k = weight slot in the resident-store order (0=wq 1=wk 2=wv 3=gate 4=wo
# 5=mlp_gate 6=mlp_up 7=mlp_down — Krea2BlockWeights field order).
def _i8_lin(
    x: Tensor, b: Krea2BlockResidentInt8, k: Int, base_key: String,
    lora_cache: Optional[Krea2LoraSideCache], ctx: DeviceContext,
) raises -> Tensor:
    var y = int8_linear_fwd(x, b.w8[k][], b.scale[k][], ctx)
    var d = _lora_side(x, base_key, lora_cache, ctx)
    if d:
        return add(y, d.value(), ctx)
    return y^


def krea2_single_stream_block_i8[L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int](
    x: Tensor,                     # [1, L, features] bf16
    vec: Tensor,                   # [1, 6*features]  (tproj(t))
    blk: Krea2BlockResidentInt8,   # resident int8 block (8× w8+scale, 5 small tensors)
    bk: String,                    # LoRA base-key root "blocks.<li>" (bare, no key_prefix)
    lora_cache: Optional[Krea2LoraSideCache],  # device-resident A/B/scale (or None)
    cos: Tensor, sin: Tensor,      # rope table [L, headdim/2]
    mask: Optional[Tensor],        # None or additive [1,HEADS,L,L] (tiled path, block 0)
    real_len: Optional[Int],       # if set -> cuDNN flash padmask path
    ctx: DeviceContext,
) raises -> Tensor:
    """krea2_single_stream_block on the int8 W8A8-RESIDENT base.

    Identical non-matmul math (rmsnorm/modulate/rope/SDPA path selection /
    sigmoid-gate/residual) to krea2_single_stream_block+krea2_attention; the 8
    matmuls run the trainer's int8 W8A8 GEMM (int8_linear_fwd) with the LoRA
    delta as a bf16 side-branch on the SAME input (== training _linear_lora).
    Returns [1, L, features]."""
    comptime heads = HEADS
    comptime kvheads = KVHEADS
    comptime headdim = HEADDIM
    comptime features = HEADS * HEADDIM
    comptime half = HEADDIM // 2
    comptime n_rep = HEADS // KVHEADS

    # mod(vec) -> 6 raw chunks, each [1, features].
    var mods = krea2_double_shared_modulation(vec, blk.mod_lin[], ctx)
    var prescale = _reshape_chunk_to_vec(mods[0], features, ctx)
    var preshift = _reshape_chunk_to_vec(mods[1], features, ctx)
    var pregate = _reshape_chunk_to_vec(mods[2], features, ctx)
    var postscale = _reshape_chunk_to_vec(mods[3], features, ctx)
    var postshift = _reshape_chunk_to_vec(mods[4], features, ctx)
    var postgate = _reshape_chunk_to_vec(mods[5], features, ctx)

    # ── ATTENTION branch: x1 = x + pregate * attn(modulate(prenorm(x))) ────────
    var xn = krea2_rmsnorm(x, blk.prenorm_scale[], Float32(1.0e-5), ctx)
    var xm = modulate(xn, prescale, preshift, ctx)                 # [1,L,features]

    var q = _i8_lin(xm, blk, 0, bk + ".attn.wq.weight", lora_cache, ctx)
    var k = _i8_lin(xm, blk, 1, bk + ".attn.wk.weight", lora_cache, ctx)
    var v = _i8_lin(xm, blk, 2, bk + ".attn.wv.weight", lora_cache, ctx)
    var gate = _i8_lin(xm, blk, 3, bk + ".attn.gate.weight", lora_cache, ctx)

    q = reshape(q, [1, L, heads, headdim], ctx)
    k = reshape(k, [1, L, kvheads, headdim], ctx)
    v = reshape(v, [1, L, kvheads, headdim], ctx)

    # QKNorm (F32-internal), RoPE (per-head tiled tables), GQA repeat — identical
    # to krea2_attention steps 3-5.
    q = krea2_rmsnorm(q, blk.qnorm_scale[], Float32(1.0e-5), ctx)
    k = krea2_rmsnorm(k, blk.knorm_scale[], Float32(1.0e-5), ctx)
    var cos_q = _tile_rope_table(cos, L, heads, half, ctx)
    var sin_q = _tile_rope_table(sin, L, heads, half, ctx)
    var cos_k = _tile_rope_table(cos, L, kvheads, half, ctx)
    var sin_k = _tile_rope_table(sin, L, kvheads, half, ctx)
    var q_rot = rope_interleaved(q, cos_q, sin_q, ctx)
    var k_rot = rope_interleaved(k, cos_k, sin_k, ctx)
    var k_full = repeat_kv_f32(k_rot, L, kvheads, n_rep, headdim, ctx)
    var v_full = repeat_kv_f32(v, L, kvheads, n_rep, headdim, ctx)

    # SDPA: same three-path selection as krea2_attention (block 0 = tiled+mask,
    # blocks >=1 = cuDNN flash padmask).
    var scale_a = Float32(1.0) / sqrt(Float32(headdim))
    var attn: Tensor
    if real_len:
        var fwd = sdpa_flash_fwd_padmask[1, L, HEADS, HEADDIM](
            q_rot, k_full, v_full, real_len.value(), scale_a, ctx
        )
        attn = fwd.o.clone(ctx)
    elif mask:
        attn = sdpa_tiled[1, L, HEADS, HEADDIM](
            q_rot, k_full, v_full, mask.value(), scale_a, ctx
        )
    else:
        attn = sdpa_nomask_tiled[1, L, HEADS, HEADDIM](
            q_rot, k_full, v_full, scale_a, ctx
        )
    var merged = reshape(attn, [1, L, features], ctx)
    var g = sigmoid(gate, ctx)
    var gated = mul(merged, g, ctx)
    var a_out = _i8_lin(gated, blk, 4, bk + ".attn.wo.weight", lora_cache, ctx)
    var x1 = residual_gate(x, pregate, a_out, ctx)

    # ── MLP branch: x2 = x1 + postgate * mlp(modulate(postnorm(x1))) ───────────
    var xn2 = krea2_rmsnorm(x1, blk.postnorm_scale[], Float32(1.0e-5), ctx)
    var xm2 = modulate(xn2, postscale, postshift, ctx)
    var mg = _i8_lin(xm2, blk, 5, bk + ".mlp.gate.weight", lora_cache, ctx)
    var mu = _i8_lin(xm2, blk, 6, bk + ".mlp.up.weight", lora_cache, ctx)
    var sw = swiglu_op(mg, mu, ctx)
    var m_out = _i8_lin(sw, blk, 7, bk + ".mlp.down.weight", lora_cache, ctx)
    return residual_gate(x1, postgate, m_out, ctx)


def _txtf_bundle(
    st: ShardedSafeTensors, prefix: String, ctx: DeviceContext
) raises -> Krea2TextFusionWeights:
    """Load one TextFusionBlock bundle from the checkpoint, ALL float params bf16
    at runtime (= reference v.to(bf16)). Norm scales stay BF16 at tensor
    boundaries; krea2_rmsnorm casts internally for reference scale.float()+1
    compute. `prefix` is the FULL key prefix (incl. any checkpoint prefix)."""
    return Krea2TextFusionWeights(
        ArcPointer(_scale(st, prefix + ".prenorm.scale", ctx)),
        ArcPointer(_scale(st, prefix + ".postnorm.scale", ctx)),
        ArcPointer(_wb(st, prefix + ".attn.wq.weight", ctx)),
        ArcPointer(_wb(st, prefix + ".attn.wk.weight", ctx)),
        ArcPointer(_wb(st, prefix + ".attn.wv.weight", ctx)),
        ArcPointer(_wb(st, prefix + ".attn.gate.weight", ctx)),
        ArcPointer(_wb(st, prefix + ".attn.wo.weight", ctx)),
        ArcPointer(_scale(st, prefix + ".attn.qknorm.qnorm.scale", ctx)),
        ArcPointer(_scale(st, prefix + ".attn.qknorm.knorm.scale", ctx)),
        ArcPointer(_wb(st, prefix + ".mlp.gate.weight", ctx)),
        ArcPointer(_wb(st, prefix + ".mlp.up.weight", ctx)),
        ArcPointer(_wb(st, prefix + ".mlp.down.weight", ctx)),
    )


def _pad_seq_zeros(
    x: Tensor, L: Int, LPAD: Int, F: Int, ctx: DeviceContext
) raises -> Tensor:
    """Pad x [1, L, F] -> [1, LPAD, F] with zeros on the seq axis (LPAD >= L)."""
    if LPAD == L:
        return x.clone(ctx)
    var pshape = List[Int]()
    pshape.append(1); pshape.append(LPAD - L); pshape.append(F)
    var pad = zeros_device(pshape^, x.dtype(), ctx)
    return concat(1, ctx, x, pad)


def krea2_forward[
    LFULL: Int,   # real combined seq length (txtlen + imglen), before pad-to-256
    LPAD: Int,    # padded seq length = ceil(LFULL/256)*256 (the main-block SDPA S)
    LT: Int,      # text/caption token length (txtfusion seq + the txtlen slice point)
    NBLOCKS: Int, # SingleStreamBlock depth (reduced=4 for 7a; 28 for production)
](
    st: ShardedSafeTensors,
    img: Tensor,        # [1, imglen, channels*patch^2 = 64] bf16
    context: Tensor,    # [1, LT, txtlayers=12, txtdim=2560] bf16
    t: Tensor,          # [1] f32 timestep
    pos: Tensor,        # [1, LFULL, 3] f32 (txt zeros + img grid ids)
    ctx: DeviceContext,
    key_prefix: String = String("w."),
    lora: Optional[LoraSet] = Optional[LoraSet](None),  # OVERLAY: when present, each
    # block's 8 LoRA-target weights get W += scale·(B@A) in-memory (never baked).
    # base key = key_prefix+"blocks.<li>.<mod>.weight"; default None = pristine base.
    lora_mult: Float32 = Float32(1.0),
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    # fp8-RESIDENT base: when present, the per-block weights are DEQUANTED from the
    # ~12GB resident store (built ONCE) instead of re-streamed disk→GPU every forward
    # → ZERO per-step disk reads (the user's no-repetitive-disk-read directive). The
    # LoRA overlay applies ON TOP of the dequant'd weight. Default None = disk stream.
    ref_tokens: Optional[Tensor] = Optional[Tensor](None),
    # OPT-IN img-EDIT reference conditioning (the torchref "krea2 o-edit" additive
    # input projection). ref_tokens [1, imglen, 64] = the SOURCE image VAE-encoded +
    # patchified (SAME layout as `img`). When BOTH ref_tokens AND img_in_ref_w are
    # present the `first` projection becomes  img = first(img) + linear(ref, img_in_ref)
    # — byte-identical to training's _krea2_apply_img_in_ref_fwd. Absent (or a zero
    # weight) => the added term is exactly 0 -> byte-identical text-to-image.
    img_in_ref_w: Optional[Tensor] = Optional[Tensor](None),  # [FEATURES, 64] BF16 trained proj
    resident_i8: Optional[Krea2ResidentInt8] = Optional[Krea2ResidentInt8](None),
    # int8 W8A8 base (task #11, TRAILING so positional call sites keep working):
    # when present, blocks run krea2_single_stream_block_i8 — the trainer's int8
    # GEMM path (tensorwise weight scale + per-token activation quant), LoRA as a
    # bf16 side-branch. resident_i8 = the FIRST K blocks device-resident; host_i8
    # = the PINNED-HOST remainder [K:NBLOCKS], H2D'd per block (fully-resident
    # ~12.1GB MEASURED OOM on 16GB). Together they must cover ALL NBLOCKS
    # (fail-loud below). MUTUALLY EXCLUSIVE with `resident` (fp8).
    host_i8: Optional[Krea2HostInt8Inf] = Optional[Krea2HostInt8Inf](None),
    lora_cache: Optional[Krea2LoraSideCache] = Optional[Krea2LoraSideCache](None),
    # Device-resident LoRA A/B/scale for the int8 side-branch — build ONCE via
    # build_krea2_lora_side_cache and pass here. If absent while `lora` is set on
    # the int8 path, the forward builds it per call (correct, slower). MEASURED:
    # the per-call mmap+alpha-probe variant cost ~4s/step of the 5.4s int8 step.
    shared: Optional[Krea2SharedResident] = Optional[Krea2SharedResident](None),
    # SHARED (non-block) weights, load-once resident (task #11): when present the
    # embedders/txtfusion/txtmlp/last weights come from the ~1.3GB device store
    # instead of the per-forward disk reload (MEASURED ~4.4s/step of host-side
    # byte-copy + F32→bf16 cast). Values byte-identical either way.
    real_text_len: Optional[Int] = Optional[Int](None),
    # VARIABLE-LT support (FlowEdit, task #20): when set (rlt < LT), `context` is a
    # ZERO-ROW-PADDED [1, LT, 12, 2560] stack whose real caption occupies rows
    # [0:rlt] — EXACTLY the trainer's length-bucket contract (krea2_cache_reader.
    # sample_padded → train_krea2._build_conditioning): txtfusion runs unmasked over
    # the padded LT (training-consistent), then the combined sequence is REORDERED
    # to [TXT_real(0:rlt) | IMG | TXT_pad] so the pad is a contiguous tail that the
    # cuDNN flash padmask (real_len = rlt + imglen) masks; block 0's additive mask
    # keep-vector covers the same valid prefix. Absent (or rlt == LT) → byte-
    # identical to the original forward.
) raises -> Tensor:
    """Krea2 SingleStreamDiT.forward (mmdit.py:413-461), b==1 inference.

    LFULL/LPAD/LT/NBLOCKS are comptime. The arch is the single_mmdit_large_wide
    config: features=6144, heads=48, kvheads=12, headdim=128, txtheads=20,
    txtlayers=12, txtdim=2560, patch=2, channels=16, tdim=256, theta=1e3.
    mask is all-ones at b==1 inference (no text pad); the ONLY masked positions are
    the pad-to-LPAD region, which the main blocks mask out via the additive
    [1,heads,LPAD,LPAD] mask. Returns the velocity on the image tokens [1, imglen, 64].

    `key_prefix` is prepended to every checkpoint key. It defaults to "w." (the
    parity-oracle dumps store weights as `w.<torch_key>`; see gen_krea2_*.py). The
    real Krea-2 raw.safetensors stores the bare torch keys (`blocks.0.attn.wq.weight`,
    `first.weight`, ...), so pipeline callers pass key_prefix="" against it. Block
    streaming is implicit: each `_wb`/`_scale` load copies only the active block's
    weights H2D (bf16) and frees them when the loop iteration ends, so the 28-block
    real model never goes fully GPU-resident.
    """
    comptime FEATURES = 6144
    comptime HEADS = 48
    comptime KVHEADS = 12
    comptime HEADDIM = 128
    comptime TXTHEADS = 20
    comptime TXTHD = 128          # txtdim/txtheads = 2560/20
    comptime NLAYERS_TXT = 12
    comptime TDIM = 256
    var imglen = img.shape()[1]

    # int8 sanity (fail-loud): resident_i8 [0:K] + host_i8 [K:NBLOCKS] must cover
    # every block the loop will index; fp8+int8 together is a caller bug.
    var n_res_i8 = 0
    if resident_i8:
        n_res_i8 = len(resident_i8.value().blocks)
    if Bool(resident_i8) or Bool(host_i8):
        if resident:
            raise Error(
                "krea2_forward: pass EITHER `resident` (fp8) OR the int8 stores"
                " (resident_i8/host_i8), not both"
            )
        var cover = n_res_i8
        if host_i8:
            if host_i8.value().first != n_res_i8:
                raise Error(
                    String("krea2_forward: host_i8.first=")
                    + String(host_i8.value().first) + " != resident_i8 blocks="
                    + String(n_res_i8) + " (int8 stores must tile [0:NBLOCKS])"
                )
            cover += len(host_i8.value().blocks)
        if cover < NBLOCKS:
            raise Error(
                String("krea2_forward: int8 stores cover ") + String(cover)
                + " blocks < NBLOCKS=" + String(NBLOCKS)
            )

    # int8-path LoRA side cache: use the caller's (built once at startup), else
    # build from the LoraSet per call (correct fallback; slower).
    var _i8_lc = Optional[Krea2LoraSideCache](None)
    if Bool(resident_i8) or Bool(host_i8):
        if lora_cache:
            _i8_lc = lora_cache.copy()
        elif lora:
            _i8_lc = Optional[Krea2LoraSideCache](
                build_krea2_lora_side_cache(lora.value(), lora_mult, ctx)
            )

    # WEIGHT DTYPE (the real raw.safetensors is MIXED precision): block matmul
    # weights are bf16 on disk, but the embedders/heads/norms/mod (first/tmlp/tproj/
    # txtmlp/last/projector, every .scale, every mod.lin) are F32. The torchref
    # reference casts EVERY floating param to bf16 at load (krea2.py:190 `v.to(bf16)`)
    # → the whole model runs bf16 (what chunks 1-7 gated against). We mirror it via
    # the _wb / _scale loaders (defined above): _wb = every float weight/bias/mod.lin
    # as bf16 (= v.to(bf16)); _scale = norm .scale bf16-ROUNDED then F32 (= reference
    # bf16(scale).float(), consumed by the F32-internal krea2_rmsnorm). NO forward-
    # path float weight is left at un-rounded F32. (On the bf16-saved gate oracles
    # these loaders are value-identical to the old from_view*; on the real F32 disk
    # they bf16-round first — the faithful runtime value. The F32 latent accumulator
    # + F32 intra-attention q/k/v carry are ACTIVATIONS, unchanged.)

    # 1) img = first(img)  -> [1, imglen, FEATURES]. img feed is bf16 -> bf16 head.
    var img_e = krea2_first(
        img,
        _shw(shared, 0, st, key_prefix + "first.weight", ctx),
        _shw(shared, 1, st, key_prefix + "first.bias", ctx),
        ctx,
    )

    # 1b) OPT-IN img-EDIT reference conditioning (mirror training's
    # _krea2_apply_img_in_ref_fwd): when a SOURCE reference is supplied,
    #     img_e += linear(ref_tokens, img_in_ref_w)
    # ref_tokens [1, imglen, 64] is the VAE-encoded source (SAME patchify layout as
    # `img`); img_in_ref_w [FEATURES, 64] is the trained additive projection. Because
    # the inference layout puts the image tokens in their OWN [1,imglen,FEATURES]
    # tensor (before the context||img concat), the scatter/slice the training hook
    # does (image rows of the concatenated `combined`) is unnecessary here — we add
    # directly to the image-token rows. A zero weight => identity (t2i unchanged).
    if Bool(ref_tokens) and Bool(img_in_ref_w):
        var _xdt = img_e.dtype()
        var _refx = cast_tensor(ref_tokens.value(), _xdt, ctx)     # [1, imglen, 64]
        var _wref = cast_tensor(img_in_ref_w.value(), _xdt, ctx)   # [FEATURES, 64]
        var _refproj = linear(_refx, _wref, Optional[Tensor](None), ctx)  # [1, imglen, FEATURES]
        img_e = add(img_e, _refproj, ctx)

    # 2) t = tmlp(temb(t, tdim))  -> [1, 1, FEATURES].
    var te = krea2_temb(t, TDIM, ctx, STDtype.BF16)   # [1, 256]
    var t_vec = krea2_tmlp(
        te,
        _shw(shared, 2, st, key_prefix + "tmlp.0.weight", ctx),
        _shw(shared, 3, st, key_prefix + "tmlp.0.bias", ctx),
        _shw(shared, 4, st, key_prefix + "tmlp.2.weight", ctx),
        _shw(shared, 5, st, key_prefix + "tmlp.2.bias", ctx),
        ctx,
    )
    var tshape = List[Int]()
    tshape.append(1); tshape.append(1); tshape.append(FEATURES)
    var t3 = reshape(t_vec, tshape^, ctx)             # [1, 1, FEATURES]  (= LastLayer tvec)

    # 3) tvec = tproj(t)  -> [1, 1, 6*FEATURES]  (the block modulation vector).
    var blk_vec = krea2_tproj(
        t3,
        _shw(shared, 6, st, key_prefix + "tproj.1.weight", ctx),
        _shw(shared, 7, st, key_prefix + "tproj.1.bias", ctx),
        ctx,
    )
    var bvshape = List[Int]()
    bvshape.append(1); bvshape.append(6 * FEATURES)
    var blk_vec2 = reshape(blk_vec, bvshape^, ctx)    # [1, 6*FEATURES] for the block

    # 4-5) context = txtfusion(context, txtmask). At b==1 the txtmask is all-ones
    # (no caption padding) => refiner runs the no-op path (chunk-6: refiner mask=None).
    var lw0 = _shtxtf(shared, 0, st, key_prefix + "txtfusion.layerwise_blocks.0", ctx)
    var lw1 = _shtxtf(shared, 1, st, key_prefix + "txtfusion.layerwise_blocks.1", ctx)
    var rf0 = _shtxtf(shared, 2, st, key_prefix + "txtfusion.refiner_blocks.0", ctx)
    var rf1 = _shtxtf(shared, 3, st, key_prefix + "txtfusion.refiner_blocks.1", ctx)
    var ctx_fused = krea2_text_fusion[LT, NLAYERS_TXT, TXTHEADS, TXTHD](
        context, lw0, lw1,
        _shw(shared, 8, st, key_prefix + "txtfusion.projector.weight", ctx),
        rf0, rf1, Optional[Tensor](None), ctx,
    )                                                  # [1, LT, txtdim]

    # 6) context = txtmlp(context)  -> [1, LT, FEATURES]. RMSNorm scale = bf16-rounded
    # then F32 (= reference bf16(scale).float()); the rest bf16.
    var ctx_proj = krea2_txtmlp(
        ctx_fused,
        _shs(shared, 9, st, key_prefix + "txtmlp.0.scale", ctx),
        _shw(shared, 10, st, key_prefix + "txtmlp.1.weight", ctx),
        _shw(shared, 11, st, key_prefix + "txtmlp.1.bias", ctx),
        _shw(shared, 12, st, key_prefix + "txtmlp.3.weight", ctx),
        _shw(shared, 13, st, key_prefix + "txtmlp.3.bias", ctx),
        ctx,
    )

    # 7-8) combined = cat(context, img, dim=1)  -> [1, LFULL, FEATURES]  (context THEN img).
    # VARIABLE-LT (real_text_len set, rlt < LT): the trainer's LENGTH-BUCKET REORDER
    # (train_krea2._build_conditioning 838-851) — [TXT_real(0:rlt) | IMG | TXT_pad]
    # so the pad text is a contiguous TAIL the flash padmask can mask.
    var rlt = LT
    if real_text_len:
        rlt = real_text_len.value()
        if rlt < 1 or rlt > LT:
            raise Error(
                String("krea2_forward: real_text_len=") + String(rlt)
                + " out of range [1, LT=" + String(LT) + "]"
            )
    var valid_len = rlt + imglen       # == LFULL when rlt == LT (no reorder)
    var combined: Tensor
    if rlt < LT:
        var real_text = slice(ctx_proj, 1, 0, rlt, ctx)            # [1,rlt,F]
        var pad_text = slice(ctx_proj, 1, rlt, LT - rlt, ctx)      # [1,LT-rlt,F]
        var head = concat(1, ctx, real_text, img_e)                # [1,rlt+imglen,F]
        combined = concat(1, ctx, head, pad_text)                  # [1,LFULL,F]
    else:
        combined = concat(1, ctx, ctx_proj, img_e)                 # [1, LFULL, FEATURES]

    # 9) pad-to-LPAD: combined (zeros), pos (zeros) on the seq axis.
    var combined_p = _pad_seq_zeros(combined, LFULL, LPAD, FEATURES, ctx)  # [1, LPAD, F]

    # 10) BLOCK-0-ONLY additive mask. Blocks >=1 use the cuDNN FLASH path (real_len=
    # LFULL masks the [LFULL:LPAD] pad rows internally). BLOCK 0 stays on the TILED F32
    # path: its QKNorm scale is ~52.9× (near-one-hot softmax) and MEASURED (7b
    # spot-check) cuDNN bf16 diverges there — cos 0.9965 < the bf16-tap floor 0.9978,
    # ch2569/3389 rel 8.0%/11.2% vs floor 4.5%/8.5% — because bf16-rounding q/k before
    # the cuDNN call flips which key wins the sharp softmax. Blocks 1 & 19 cuDNN-match
    # the tap AT floor (cos 0.99999/0.99998). So: block 0 tiled-F32 (faithful), 1..N-1
    # cuDNN-flash (the 1024² speedup on 27/28 blocks). build_krea2_text_mask wants
    # keep [LPAD] F32 = ones[0:LFULL], zeros[LFULL:LPAD].
    # keep = ones over the VALID prefix [0:valid_len] (real text + image); zeros over
    # the pad-text tail (variable-LT reorder) AND the pad-to-256 region.
    var keep_host = List[Float32]()
    for i in range(LPAD):
        keep_host.append(Float32(1.0) if i < valid_len else Float32(0.0))
    var keep_shape = List[Int]()
    keep_shape.append(LPAD)
    var keep = Tensor.from_host(keep_host^, keep_shape^, STDtype.F32, ctx)
    var blk0_mask = build_krea2_text_mask(keep, HEADS, LPAD, ctx, STDtype.BF16)  # [1,HEADS,LPAD,LPAD]
    # Build the two Optionals ONCE (Tensor is move-only) and pass them BORROWED (the
    # block's `mask`/`real_len` args are `read`): block 0 uses blk0_mask_opt + None;
    # blocks >=1 use none_mask + LFULL. The unused one is just an empty/ignored Optional.
    var blk0_mask_opt = Optional[Tensor](blk0_mask^)
    var none_mask = Optional[Tensor](None)
    var rl_none = Optional[Int](None)
    var rl_full = Optional[Int](valid_len)   # == LFULL when rlt == LT

    # 11) freqs = posemb(pos): pad pos to LPAD (zeros), build the rope table [LPAD,64].
    var pos_flat_shape = List[Int]()
    pos_flat_shape.append(LFULL * 3)
    var pos_flat = reshape(pos, pos_flat_shape^, ctx)         # [LFULL*3]
    var pos_pad_host = List[Float32]()
    var pos_host = pos_flat.to_host(ctx)
    if rlt < LT:
        # VARIABLE-LT reorder: [txt_real_zeros(rlt) | img grid | zeros to LPAD].
        # Text pos rows are ALL-ZERO (same rationale as train_krea2 855-858: the
        # reorder changes no token's rotation, it aligns the table to `combined`).
        for _i in range(rlt * 3):
            pos_pad_host.append(Float32(0.0))
        for i in range(imglen * 3):
            pos_pad_host.append(pos_host[LT * 3 + i])
        for _i in range((LPAD - valid_len) * 3):
            pos_pad_host.append(Float32(0.0))
    else:
        for i in range(LFULL * 3):
            pos_pad_host.append(pos_host[i])
        for _i in range((LPAD - LFULL) * 3):
            pos_pad_host.append(Float32(0.0))
    var pos_pad_shape = List[Int]()
    pos_pad_shape.append(LPAD * 3)
    var pos_pad = Tensor.from_host(pos_pad_host^, pos_pad_shape^, STDtype.F32, ctx)
    var axes = List[Int]()
    axes.append(32); axes.append(48); axes.append(48)
    var rope = build_krea2_rope(pos_pad, axes, Float32(1.0e3), ctx, STDtype.F32)  # ([LPAD,64], [LPAD,64])

    # 12) N x SingleStreamBlock. Block 0 = TILED F32 (additive mask, near-one-hot
    # faithful); blocks 1..N-1 = cuDNN FLASH (real_len=LFULL pad-mask, the speedup).
    var x = combined_p^
    for li in range(NBLOCKS):
        var p = key_prefix + "blocks." + String(li)
        # LoRA base-weight key root (bare, no key_prefix — the LoraSet resolved the
        # PEFT prefix diffusion_model.blocks.<li>.<mod> → blocks.<li>.<mod>.weight).
        var bk = String("blocks.") + String(li)
        # int8 W8A8 source (task #11): NO disk, NO dequant — the trainer's int8
        # GEMM per matmul, LoRA as a bf16 side-branch inside the i8 block.
        # Blocks [0:K] come from the device-resident store; [K:NBLOCKS] are H2D'd
        # from the pinned-host store (fresh transient device int8, ~433MB).
        if Bool(resident_i8) and li < n_res_i8:
            ref bi8 = resident_i8.value().blocks[li]
            x = krea2_single_stream_block_i8[LPAD, HEADS, KVHEADS, HEADDIM](
                x, blk_vec2, bi8, bk, _i8_lc,
                rope[0], rope[1],
                blk0_mask_opt if li == 0 else none_mask,
                rl_none if li == 0 else rl_full,
                ctx,
            )
            # Same per-block drain rationale as the LoRA-overlay path below: fence
            # the deferred frees of the block's activation/side-branch transients
            # so peak stays resident-store + ONE block's transients.
            if lora:
                ctx.synchronize()
            continue
        if host_i8:
            var hb = _krea2_host_i8_block_dev(
                host_i8.value(), li - host_i8.value().first, ctx
            )
            x = krea2_single_stream_block_i8[LPAD, HEADS, KVHEADS, HEADDIM](
                x, blk_vec2, hb, bk, _i8_lc,
                rope[0], rope[1],
                blk0_mask_opt if li == 0 else none_mask,
                rl_none if li == 0 else rl_full,
                ctx,
            )
            # ALWAYS drain on the host path: the transient device int8 block
            # (~433MB) must free before the next block's H2D or the loop
            # accumulates NBLOCKS×433MB of deferred frees.
            ctx.synchronize()
            continue
        # Per-block weight SOURCE: fp8-resident dequant (NO disk) when `resident` is
        # present, else disk-stream via _wb/_scale. Then the 8 matmul weights get the
        # LoRA overlay (W += scale·BA) on top. The 5 small tensors (mod.lin + 4 norm
        # scales) come from the same source. _blk_w8 returns the k-th matmul weight
        # (k: 0=wq 1=wk 2=wv 3=gate 4=wo 5=mlp_gate 6=mlp_up 7=mlp_down).
        x = krea2_single_stream_block[LPAD, HEADS, KVHEADS, HEADDIM](
            x,
            blk_vec2,
            _blk_modlin(resident, li, st, p + ".mod.lin", ctx),
            _blk_scale(resident, li, 2, st, p + ".prenorm.scale", ctx),   # 2=prenorm
            _blk_scale(resident, li, 3, st, p + ".postnorm.scale", ctx),  # 3=postnorm
            _blk_w8(resident, li, 0, st, p + ".attn.wq.weight", bk + ".attn.wq.weight", lora, lora_mult, ctx),
            _blk_w8(resident, li, 1, st, p + ".attn.wk.weight", bk + ".attn.wk.weight", lora, lora_mult, ctx),
            _blk_w8(resident, li, 2, st, p + ".attn.wv.weight", bk + ".attn.wv.weight", lora, lora_mult, ctx),
            _blk_w8(resident, li, 3, st, p + ".attn.gate.weight", bk + ".attn.gate.weight", lora, lora_mult, ctx),
            _blk_w8(resident, li, 4, st, p + ".attn.wo.weight", bk + ".attn.wo.weight", lora, lora_mult, ctx),
            _blk_scale(resident, li, 0, st, p + ".attn.qknorm.qnorm.scale", ctx),  # 0=qnorm
            _blk_scale(resident, li, 1, st, p + ".attn.qknorm.knorm.scale", ctx),  # 1=knorm
            _blk_w8(resident, li, 5, st, p + ".mlp.gate.weight", bk + ".mlp.gate.weight", lora, lora_mult, ctx),
            _blk_w8(resident, li, 6, st, p + ".mlp.up.weight", bk + ".mlp.up.weight", lora, lora_mult, ctx),
            _blk_w8(resident, li, 7, st, p + ".mlp.down.weight", bk + ".mlp.down.weight", lora, lora_mult, ctx),
            rope[0], rope[1],
            blk0_mask_opt if li == 0 else none_mask,
            rl_none if li == 0 else rl_full,
            ctx,
        )
        # LoRA-path per-block DRAIN: the overlay adds ~7 transient tensors/module
        # (load A+B, transpose, B@A linear, mul_scalar, cast, add) × 8 modules/block.
        # Without a per-block fence these deferred frees accumulate across all 28
        # blocks × 2 forwards/step ON TOP of the 12GB fp8-resident base → OOM (~step
        # 17, MEASURED). One sync/block reclaims them so peak = resident + ONE block's
        # transients. Guarded by `if lora` so the no-LoRA path (A, already fits) is
        # UNCHANGED. (resident-dequant transients alone don't tip it; the LoRA delta
        # compute is the added pressure that needs the drain.)
        if lora:
            ctx.synchronize()

    # 13) final = last_layer(combined, t)  (tvec = t3, the tmlp output).
    # Norm scale stays raw checkpoint dtype at the boundary; krea2_rmsnorm casts it
    # internally for scale.float()+1 and returns x dtype.
    var final = krea2_last_layer(
        x,
        t3,
        _shs(shared, 14, st, key_prefix + "last.norm.scale", ctx),
        _shw(shared, 15, st, key_prefix + "last.modulation.lin", ctx),
        _shw(shared, 16, st, key_prefix + "last.linear.weight", ctx),
        _shw(shared, 17, st, key_prefix + "last.linear.bias", ctx),
        FEATURES,
        ctx,
    )                                                  # [1, LPAD, 64]

    # 14) output = final[:, txtlen : txtlen+imglen, :]  (the image tokens). Under
    # the variable-LT reorder the image tokens sit at [rlt : rlt+imglen] (rlt == LT
    # when real_text_len is absent — the original slice).
    return slice(final, 1, rlt, imglen, ctx)           # [1, imglen, 64]
