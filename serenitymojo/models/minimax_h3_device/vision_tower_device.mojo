# serenitymojo/models/minimax_h3_device/vision_tower_device.mojo
#
# MiniMax-H3 Qwen3-VL VISION TOWER — DEVICE port.
#
# This is a HOST->DEVICE MOVE of already-verified math, not a re-port. The
# reference is `serenitymojo/models/text_encoder/minimax_h3_qwen3vl_vision.mojo`
# — the ONE host tower (arbitration-kept), whose geometry is gated by 25
# host-side checks and whose weighted forward is gated against a float32 torch
# oracle at derived bars (minimax_h3_qwen3vl_vision_cpu_gate.mojo). Every
# geometric decision below — cu_seqlens, merge-block rotary coordinates, the
# hardcoded inv_freq table, the `_torch_linspace_f32` pos-embed interpolation,
# the two mergers' different norm widths, the two different GELUs — is IMPORTED
# from that file, not re-derived. When the two disagree, the HOST file is right
# and this one is wrong.
#
# WHY: the host tower runs ~21.5 minutes at the real 2304-patch keyframe
# geometry (O(P^2) eager attention in host lists) and is the perf long-pole for
# i2va / ref2va conditioning while every other stage is device-resident.
#
# NO NEW GPU KERNELS. Every numeric operation is an existing, separately gated
# primitive:
#     ops/linear.linear_bias        (bf16 GEMM, F32 accumulation)
#     ops/norm.layer_norm           (F32-accumulated mean/var)
#     ops/attention.sdpa_nomask     (math-mode SDPA, F32 softmax)
#     ops/rope.rope_halfsplit       (rotate-half rotary, F32 math)
#     ops/activations.gelu / gelu_exact
#     ops/tensor_algebra.add / reshape / reshape_owned
# The structure deliberately mirrors `lingbot_qwen3vl_vision.mojo` — the SAME
# Qwen3-VL vision architecture at different dimensions, already device-gated at
# cos>=0.999 against its own bf16 GPU torch oracle — and
# `audio_decoder_device.mojo` (tonight's device BigVGAN move) for the
# name-keyed upload-once weight cache.
#
# DTYPE LAW — BF16 storage END-TO-END, F32 accumulation inside every op,
# exactly torch-CUDA's own bf16 semantics. MEASURED, not assumed:
#   * WEIGHTS stay the checkpoint's native BF16 on device, byte-verbatim.
#   * The ACTIVATION stream is BF16, like torch's. The first build of this
#     file used lingbot's F32-stream recipe (bf16 weights, f32 residual) and
#     the gate MEASURED it failing at exactly the two deepest stages:
#     block_26 cos 0.99724, embeds 0.99807 (bar 0.999) — while every earlier
#     stage passed. The mechanism is the residual CARRIER: by block 26 the
#     activations reach absmax ~10560, where one bf16 ulp is 64 — torch's
#     bf16 residual adds quantize away increments our f32 carrier keeps, so
#     the two implementations track different trajectories at the outlier
#     channels precisely where magnitudes explode. Re-quantizing the stream
#     to bf16 at every op — LN out, GEMM out, gelu out, residual add out —
#     follows torch's own rounding trajectory instead. Each op still
#     ACCUMULATES in F32 (the ops' bf16 kernels all do), which is what
#     torch-CUDA does per-thread with tree reductions, so the host tower's
#     F64-accumulation fix is not "regressed" here — that fix targeted a
#     float32-vs-float32 comparison at derived bars; the reference HERE is
#     the bf16 DEVICE torch run.
#   * The ROTARY APPLICATION is F32 on both sides: torch's
#     `apply_rotary_pos_emb_vision` upcasts q/k AND the bf16 tables to float,
#     rotates, and casts back to bf16 — which is exactly `ops/rope`'s
#     bf16-x/f32-tables kernel.
#   * The ROTARY TABLES replicate the bf16 oracle's quantization chain
#     bit-for-bit: `inv_freq` is a NON-persistent float32 buffer that
#     `model.to(bfloat16)` bf16-ROUNDS (asserted at run time by the oracle),
#     `torch.outer` then computes bf16 freqs, `.cos()/.sin()` produce bf16
#     tables, and `apply_rotary_pos_emb_vision` upcasts them to F32 for the
#     actual rotation. Skipping the bf16 rounding is NOT a smaller error: at
#     row 47 the raw frequency is 47 rad and one bf16 ulp there is 0.25 rad —
#     a per-element phase error torch's own tables carry and ours must carry
#     too (the repo's rope-buffer rule, feedback_bf16_buffer_rounding).
#
# ATTENTION SEGMENTATION: one segment PER FRAME (host header trap 2), via the
# gated `minimax_h3_vision_cu_seqlens`. `sdpa_nomask` needs its S at compile
# time (the foundation-op wall `qwen3_encoder._sdpa_dispatch` also lives
# behind), so segment lengths are dispatched through `_vis_sdpa_segment`
# below. The i2va keyframe path — one still image, one segment — is
# instantiated at S=2304 (the 768x768 canvas, the smallest canvas
# `resolve_canvas_size` can emit and the gated geometry). Ref2VA's released
# 768x1344 reference canvas is one temporal block at S=4032 and has its own
# explicit instantiation. Other lengths raise a NAMED error listing the exact
# instantiation to add; a request with more than the released FL2VA pair
# likewise raises rather than silently attending across frame boundaries.
#
# Gate: models/minimax_h3_device/parity/minimax_h3_vision_tower_device_parity.mojo
# against scripts/minimax_h3_vision_tower_device_oracle.py (torch bf16, GPU,
# real FL2VA weights, the real shopping keyframe at 2304 patches).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.collections import List
from std.math import cos as fcos, sin as fsin, sqrt as fsqrt
from std.memory import ArcPointer
from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.ffi import BytePtr, sys_memcpy
from serenitymojo.ops.linear import linear_bias
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.attention import sdpa_nomask, sdpa_nomask_dynamic
from serenitymojo.ops.activations import gelu, gelu_exact
from serenitymojo.ops.rope import rope_halfsplit
from serenitymojo.ops.tensor_algebra import add, concat, reshape, reshape_owned, slice

from serenitymojo.models.text_encoder.minimax_h3_qwen3vl_vision import (
    H3_VIS_DEPTH,
    H3_VIS_HIDDEN,
    H3_VIS_HEADS,
    H3_VIS_HEAD_DIM,
    H3_VIS_ROTARY_DIM,
    H3_VIS_INTERMEDIATE,
    H3_VIS_OUT_HIDDEN,
    H3_VIS_PATCH_NUMEL,
    H3_VIS_MERGE_UNIT,
    H3_VIS_MERGED_WIDTH,
    H3_VIS_EPS,
    H3_VIS_DEEPSTACK_0,
    H3_VIS_DEEPSTACK_1,
    H3_VIS_DEEPSTACK_2,
    MiniMaxH3VisionGrid,
    MiniMaxH3VisionOutput,
    minimax_h3_vision_tensor_names,
    minimax_h3_vision_cu_seqlens,
    minimax_h3_vision_rotary_inv_freq,
    minimax_h3_vision_rotary_positions,
    minimax_h3_vision_deepstack_tap_blocks,
    _vis_pos_embeds,
)

# The one attention length instantiated so far: 48x48 patches, the 768x768
# canvas (the smallest `resolve_canvas_size` can emit; square keyframes).
comptime H3_VIS_SDPA_S_2304 = 2304
comptime H3_VIS_SDPA_S_4032 = 4032

comptime _ROT_FREQS = H3_VIS_ROTARY_DIM // 2   # 18 inv_freq entries


def _bf16_round(v: Float32) -> Float32:
    """Round-to-nearest-even F32 -> BF16 -> F32, the same widening/narrowing
    pair `Tensor.from_host` / `.to_host` use. Replicates what torch's
    `.to(bfloat16)` does to a buffer value."""
    return v.cast[DType.bfloat16]().cast[DType.float32]()


# ═════════════════════════════════════════════════════════════════════════════
# Device weight cache — name-keyed with a linear scan, deliberately mirroring
# `MiniMaxH3AudioDeviceWeights` (audio_decoder_device.mojo) and the host
# tower's own `MiniMaxH3VisionWeights.get`, so the forward below reads
# structurally identical to the host file. Keys are the FULL `model.visual.*`
# names exactly as `minimax_h3_vision_tensor_names` emits them, with ONE
# load-time transform: each block's fused `attn.qkv.{weight,bias}` is split
# into `attn.q/k/v.{weight,bias}` (contiguous row thirds of the stored tensor
# — the same reshape(seq,3,heads,-1) outer split the host's `_vis_qkv_part`
# gates), because three [1152,1152] GEMMs and one [3456,1152] GEMM are the
# same per-element dot products and the split needs no device slice op.
# ═════════════════════════════════════════════════════════════════════════════
struct MiniMaxH3VisionDeviceWeights(Movable):
    var names: List[String]
    var values: List[ArcPointer[Tensor]]

    def __init__(out self):
        self.names = List[String]()
        self.values = List[ArcPointer[Tensor]]()

    def put(mut self, name: String, var t: Tensor):
        self.names.append(name)
        self.values.append(ArcPointer[Tensor](t^))

    def slot(self, name: String) raises -> Int:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return i
        raise Error(
            String("minimax_h3 vision device: missing device tensor ") + name
        )


def _slice_to_device(
    hbase: Int, byte_off: Int, nbytes: Int,
    var shape: List[Int], dtype: STDtype, ctx: DeviceContext,
) raises -> Tensor:
    """H2D-copy a contiguous byte sub-range of an mmap'd tensor view into a
    fresh device buffer (used to split the fused qkv). Same helper
    `lingbot_qwen3vl_vision.mojo` uses for the same split."""
    var host = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
    var hdst = BytePtr(unsafe_from_address=Int(host.unsafe_ptr()))
    var hsrc = BytePtr(unsafe_from_address=hbase + byte_off)
    _ = sys_memcpy(hdst, hsrc, nbytes)
    var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    ctx.synchronize()
    return Tensor(dev^, shape^, dtype)


def minimax_h3_vision_device_weights(
    text_encoder_dir: String,
    ctx: DeviceContext,
) raises -> MiniMaxH3VisionDeviceWeights:
    """Upload all 351 `model.visual.*` tensors, NATIVE BF16, byte-verbatim.

    Walks the gated manifest (`minimax_h3_vision_tensor_names`) rather than
    whatever the shards happen to contain, so a checkpoint/manifest mismatch
    fails HERE, at load, naming the tensor — not 27 blocks deep. Two
    load-time transforms, both shape-only:
      * `patch_embed.proj.weight` [1152,3,2,16,16] -> [1152,1536]: the conv
        that is really a linear (host header trap 1); the row-major flatten
        (c, t, ph, pw) is exactly the per-row order the gated preprocessor
        writes.
      * each `attn.qkv.{weight,bias}` -> `attn.q/k/v.{weight,bias}` thirds.
    """
    var st = ShardedSafeTensors.open(text_encoder_dir)
    var names = minimax_h3_vision_tensor_names()
    var dev = MiniMaxH3VisionDeviceWeights()

    for i in range(len(names)):
        var name = names[i]
        var tv = st.tensor_view(name)
        if tv.dtype != STDtype.BF16 and tv.dtype != STDtype.F32:
            raise Error(
                String("minimax_h3 vision device: unsupported dtype ")
                + tv.dtype.name() + " for " + name
            )

        if name == String("model.visual.patch_embed.proj.weight"):
            var t = Tensor.from_view(tv, ctx)
            dev.put(name, reshape_owned(t^, [H3_VIS_HIDDEN, H3_VIS_PATCH_NUMEL]))
            continue

        var wstem = String(name.removesuffix("qkv.weight"))
        if wstem.byte_length() != name.byte_length():
            var stem = wstem
            var hbase = Int(tv.data.unsafe_ptr())
            var blk_bytes = H3_VIS_HIDDEN * H3_VIS_HIDDEN * tv.dtype.byte_size()
            dev.put(stem + "q.weight", _slice_to_device(
                hbase, 0 * blk_bytes, blk_bytes,
                [H3_VIS_HIDDEN, H3_VIS_HIDDEN], tv.dtype, ctx))
            dev.put(stem + "k.weight", _slice_to_device(
                hbase, 1 * blk_bytes, blk_bytes,
                [H3_VIS_HIDDEN, H3_VIS_HIDDEN], tv.dtype, ctx))
            dev.put(stem + "v.weight", _slice_to_device(
                hbase, 2 * blk_bytes, blk_bytes,
                [H3_VIS_HIDDEN, H3_VIS_HIDDEN], tv.dtype, ctx))
            continue
        var bstem = String(name.removesuffix("qkv.bias"))
        if bstem.byte_length() != name.byte_length():
            var stem = bstem
            var hbase = Int(tv.data.unsafe_ptr())
            var blk_bytes = H3_VIS_HIDDEN * tv.dtype.byte_size()
            dev.put(stem + "q.bias", _slice_to_device(
                hbase, 0 * blk_bytes, blk_bytes, [H3_VIS_HIDDEN], tv.dtype, ctx))
            dev.put(stem + "k.bias", _slice_to_device(
                hbase, 1 * blk_bytes, blk_bytes, [H3_VIS_HIDDEN], tv.dtype, ctx))
            dev.put(stem + "v.bias", _slice_to_device(
                hbase, 2 * blk_bytes, blk_bytes, [H3_VIS_HIDDEN], tv.dtype, ctx))
            continue

        dev.put(name, Tensor.from_view(tv, ctx))

    return dev^


# ═════════════════════════════════════════════════════════════════════════════
# Rotary tables — host-built, bf16-quantization-replicated (see file header).
# ═════════════════════════════════════════════════════════════════════════════
def minimax_h3_vision_device_rope_host(
    grids: List[MiniMaxH3VisionGrid],
) raises -> Tuple[List[Float32], List[Float32]]:
    """cos/sin, `[num_patches, 72]` host F32 — the F32 UPCAST of the bf16
    tables the bf16 torch model runs with. Layout matches the host tower's
    `_vis_rotary_cos_sin`: per patch `[row 18 | col 18 | row 18 | col 18]`
    (the `emb = cat(rot, rot)` duplication).

    Chain, replicated from the measured torch-4.57.6 bf16 flow:
      inv_bf16 = bf16(inv_f32_hardcoded)      # .to(bfloat16) on the buffer
      freq     = bf16(pos * f32(inv_bf16))    # torch.outer on bf16 CUDA
      table    = bf16(cos/sin(f32(freq)))     # .cos()/.sin() on the bf16 emb
    The gate compares this against the oracle's dump of the model's OWN
    tables, so the replication is proven, not assumed."""
    var inv = minimax_h3_vision_rotary_inv_freq()
    var inv_b = List[Float32]()
    for j in range(len(inv)):
        inv_b.append(_bf16_round(inv[j]))
    if len(inv_b) != _ROT_FREQS:
        raise Error("minimax_h3 vision device: inv_freq table is not 18 long")

    var coords = minimax_h3_vision_rotary_positions(grids)
    var num_patches = len(coords) // 2
    var cos72 = List[Float32]()
    var sin72 = List[Float32]()
    cos72.resize(num_patches * H3_VIS_HEAD_DIM, Float32(0.0))
    sin72.resize(num_patches * H3_VIS_HEAD_DIM, Float32(0.0))

    for p in range(num_patches):
        var row = coords[2 * p]
        var col = coords[2 * p + 1]
        var obase = p * H3_VIS_HEAD_DIM
        for j in range(_ROT_FREQS):
            var fr = _bf16_round(Float32(row) * inv_b[j])
            var fc = _bf16_round(Float32(col) * inv_b[j])
            cos72[obase + j] = _bf16_round(fcos(fr))
            sin72[obase + j] = _bf16_round(fsin(fr))
            cos72[obase + _ROT_FREQS + j] = _bf16_round(fcos(fc))
            sin72[obase + _ROT_FREQS + j] = _bf16_round(fsin(fc))
        # emb = cat(rot, rot): the second 36 duplicate the first 36.
        for j in range(H3_VIS_ROTARY_DIM):
            cos72[obase + H3_VIS_ROTARY_DIM + j] = cos72[obase + j]
            sin72[obase + H3_VIS_ROTARY_DIM + j] = sin72[obase + j]

    return (cos72^, sin72^)


struct _RopeApply(Movable):
    """The two device apply-tables, held in a struct because `Tensor` is
    Movable-only and cannot be moved out of a `Tuple` element (the same reason
    lingbot's `RopeTables` exists)."""

    var cos_apply: Tensor
    var sin_apply: Tensor

    def __init__(out self, var cos_apply: Tensor, var sin_apply: Tensor):
        self.cos_apply = cos_apply^
        self.sin_apply = sin_apply^


def _vis_rope_apply_tables(
    cos72: List[Float32], sin72: List[Float32],
    num_patches: Int, ctx: DeviceContext,
) raises -> _RopeApply:
    """`ops/rope.rope_halfsplit` apply-tables: `[num_patches * heads, 36]` F32,
    the first half of each 72-row broadcast over every head (the reference's
    `cos.unsqueeze(-2)` — host header trap 5). rotate_half + duplicated-half
    tables is exactly the half-split pairing, so only the first 36 are needed."""
    var rows = num_patches * H3_VIS_HEADS
    var cos_a = List[Float32]()
    var sin_a = List[Float32]()
    cos_a.resize(rows * H3_VIS_ROTARY_DIM, Float32(0.0))
    sin_a.resize(rows * H3_VIS_ROTARY_DIM, Float32(0.0))
    for p in range(num_patches):
        var sbase = p * H3_VIS_HEAD_DIM
        for h in range(H3_VIS_HEADS):
            var dbase = (p * H3_VIS_HEADS + h) * H3_VIS_ROTARY_DIM
            for j in range(H3_VIS_ROTARY_DIM):
                cos_a[dbase + j] = cos72[sbase + j]
                sin_a[dbase + j] = sin72[sbase + j]
    var ct = Tensor.from_host(cos_a, [rows, H3_VIS_ROTARY_DIM], STDtype.F32, ctx)
    var st = Tensor.from_host(sin_a, [rows, H3_VIS_ROTARY_DIM], STDtype.F32, ctx)
    return _RopeApply(ct^, st^)


# ═════════════════════════════════════════════════════════════════════════════
# Attention — the comptime-S dispatch wall (see file header).
# ═════════════════════════════════════════════════════════════════════════════
def _vis_sdpa_segment(
    q: Tensor, k: Tensor, v: Tensor, seg: Int, scale: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """Full non-causal MHA over ONE cu_seqlens segment, `[1,seg,16,72]` BSHD.

    Runtime math dispatch keeps inference on GPU for every processor grid while
    preserving the accepted static vision-tower attention arithmetic; `seg` is
    checked against the actual sliced tensor before launch."""
    if q.shape()[1] != seg or k.shape()[1] != seg or v.shape()[1] != seg:
        raise Error("minimax_h3 vision device: segment tensor length mismatch")
    return sdpa_nomask_dynamic(q, k, v, scale, ctx)


# ═════════════════════════════════════════════════════════════════════════════
# Forward. Structure mirrors the host tower's `minimax_h3_vision_forward_traced`
# stage-for-stage so the two can be diffed by eye and gated at the same seams.
# ═════════════════════════════════════════════════════════════════════════════
struct MiniMaxH3VisionDeviceTrace(Movable):
    """Device-resident twins of the host `MiniMaxH3VisionTrace` stages, plus
    the three tap outputs separately (they are separate gate rows). All F32."""

    var after_patch: Tensor   # [num_patches, 1152]
    var block_00: Tensor
    var block_08: Tensor
    var block_16: Tensor
    var block_24: Tensor
    var block_26: Tensor
    var ds0: Tensor           # [num_tokens, 5120]
    var ds1: Tensor
    var ds2: Tensor
    var embeds: Tensor        # [num_tokens, 5120]

    def __init__(
        out self,
        var after_patch: Tensor, var block_00: Tensor, var block_08: Tensor,
        var block_16: Tensor, var block_24: Tensor, var block_26: Tensor,
        var ds0: Tensor, var ds1: Tensor, var ds2: Tensor, var embeds: Tensor,
    ):
        self.after_patch = after_patch^
        self.block_00 = block_00^
        self.block_08 = block_08^
        self.block_16 = block_16^
        self.block_24 = block_24^
        self.block_26 = block_26^
        self.ds0 = ds0^
        self.ds1 = ds1^
        self.ds2 = ds2^
        self.embeds = embeds^


def _vis_device_block(
    ref w: MiniMaxH3VisionDeviceWeights,
    blk: Int,
    var hidden: Tensor,
    cos_apply: Tensor,
    sin_apply: Tensor,
    cu_seqlens: List[Int],
    ctx: DeviceContext,
) raises -> Tensor:
    """One pre-norm vision block:
    h += proj(attn(rope(qkv(LN1 h)))); h += fc2(gelu_tanh(fc1(LN2 h)))."""
    var p = String("model.visual.blocks.") + String(blk) + String(".")
    var seq = hidden.shape()[0]
    var scale = Float32(1.0) / fsqrt(Float32(H3_VIS_HEAD_DIM))

    var normed = layer_norm(
        hidden,
        w.values[w.slot(p + "norm1.weight")][],
        w.values[w.slot(p + "norm1.bias")][],
        H3_VIS_EPS, ctx,
    )
    var q = linear_bias(
        normed, w.values[w.slot(p + "attn.q.weight")][],
        w.values[w.slot(p + "attn.q.bias")][], ctx,
    )
    var k = linear_bias(
        normed, w.values[w.slot(p + "attn.k.weight")][],
        w.values[w.slot(p + "attn.k.bias")][], ctx,
    )
    var v = linear_bias(
        normed, w.values[w.slot(p + "attn.v.weight")][],
        w.values[w.slot(p + "attn.v.bias")][], ctx,
    )

    # 2-D rotary on q/k: [seq,1152] -> [seq*heads,72] rows, tables broadcast
    # per head (host header trap 5), F32 math on both sides.
    q = reshape(q, [seq * H3_VIS_HEADS, H3_VIS_HEAD_DIM], ctx)
    k = reshape(k, [seq * H3_VIS_HEADS, H3_VIS_HEAD_DIM], ctx)
    q = rope_halfsplit(q, cos_apply, sin_apply, ctx)
    k = rope_halfsplit(k, cos_apply, sin_apply, ctx)

    # BSHD for full non-causal MHA, ONE SEGMENT PER FRAME (host header trap 2).
    # Released I2VA/Ref2VA profiles have one segment; FL2VA has two square
    # keyframe segments. Slice each segment before SDPA and concatenate in the
    # original row order, exactly matching the host tower's cu_seqlens loop.
    q = reshape(q, [1, seq, H3_VIS_HEADS, H3_VIS_HEAD_DIM], ctx)
    k = reshape(k, [1, seq, H3_VIS_HEADS, H3_VIS_HEAD_DIM], ctx)
    v = reshape(v, [1, seq, H3_VIS_HEADS, H3_VIS_HEAD_DIM], ctx)
    if len(cu_seqlens) < 2 or cu_seqlens[0] != 0 \
            or cu_seqlens[len(cu_seqlens) - 1] != seq:
        raise Error(
            "minimax_h3 vision device: invalid attention segment boundaries"
        )
    var first_len = cu_seqlens[1] - cu_seqlens[0]
    if first_len <= 0:
        raise Error("minimax_h3 vision device: empty attention segment")
    var q0 = slice(q, 1, 0, first_len, ctx)
    var k0 = slice(k, 1, 0, first_len, ctx)
    var v0 = slice(v, 1, 0, first_len, ctx)
    var attn = _vis_sdpa_segment(q0, k0, v0, first_len, scale, ctx)
    for segment in range(1, len(cu_seqlens) - 1):
        var start = cu_seqlens[segment]
        var stop = cu_seqlens[segment + 1]
        var seg = stop - start
        if seg <= 0:
            raise Error("minimax_h3 vision device: empty attention segment")
        var qs = slice(q, 1, start, seg, ctx)
        var ks = slice(k, 1, start, seg, ctx)
        var vs = slice(v, 1, start, seg, ctx)
        var next = _vis_sdpa_segment(qs, ks, vs, seg, scale, ctx)
        attn = concat(1, ctx, attn, next)
    attn = reshape(attn, [seq, H3_VIS_HIDDEN], ctx)

    var attn_out = linear_bias(
        attn, w.values[w.slot(p + "attn.proj.weight")][],
        w.values[w.slot(p + "attn.proj.bias")][], ctx,
    )
    var h1 = add(hidden, attn_out, ctx)

    var normed2 = layer_norm(
        h1,
        w.values[w.slot(p + "norm2.weight")][],
        w.values[w.slot(p + "norm2.bias")][],
        H3_VIS_EPS, ctx,
    )
    var mh = linear_bias(
        normed2, w.values[w.slot(p + "mlp.linear_fc1.weight")][],
        w.values[w.slot(p + "mlp.linear_fc1.bias")][], ctx,
    )
    mh = gelu(mh, ctx)   # gelu_pytorch_tanh — the BLOCK activation (trap 4)
    var mlp_out = linear_bias(
        mh, w.values[w.slot(p + "mlp.linear_fc2.weight")][],
        w.values[w.slot(p + "mlp.linear_fc2.bias")][], ctx,
    )
    return add(h1, mlp_out, ctx)


def _vis_device_merger_postshuffle(
    ref w: MiniMaxH3VisionDeviceWeights,
    prefix: String, hidden: Tensor, ctx: DeviceContext,
) raises -> Tensor:
    """A DEEPSTACK merger — `use_postshuffle_norm=TRUE` (host header trap 3):
    reshape to [-1, 4608] FIRST, then LayerNorm over 4608, then fc1 ->
    exact-erf GELU (trap 4) -> fc2."""
    var seq = hidden.shape()[0]
    var groups = seq // H3_VIS_MERGE_UNIT
    var x = reshape(hidden, [groups, H3_VIS_MERGED_WIDTH], ctx)
    x = layer_norm(
        x, w.values[w.slot(prefix + "norm.weight")][],
        w.values[w.slot(prefix + "norm.bias")][], H3_VIS_EPS, ctx,
    )
    x = linear_bias(
        x, w.values[w.slot(prefix + "linear_fc1.weight")][],
        w.values[w.slot(prefix + "linear_fc1.bias")][], ctx,
    )
    x = gelu_exact(x, ctx)
    return linear_bias(
        x, w.values[w.slot(prefix + "linear_fc2.weight")][],
        w.values[w.slot(prefix + "linear_fc2.bias")][], ctx,
    )


def _vis_device_merger_preshuffle(
    ref w: MiniMaxH3VisionDeviceWeights,
    hidden: Tensor, ctx: DeviceContext,
) raises -> Tensor:
    """The FINAL merger — `use_postshuffle_norm=FALSE` (host header trap 3):
    LayerNorm over 1152 FIRST (per patch), THEN the reshape to [-1, 4608]."""
    var seq = hidden.shape()[0]
    var groups = seq // H3_VIS_MERGE_UNIT
    var x = layer_norm(
        hidden, w.values[w.slot(String("model.visual.merger.norm.weight"))][],
        w.values[w.slot(String("model.visual.merger.norm.bias"))][],
        H3_VIS_EPS, ctx,
    )
    x = reshape(x, [groups, H3_VIS_MERGED_WIDTH], ctx)
    x = linear_bias(
        x, w.values[w.slot(String("model.visual.merger.linear_fc1.weight"))][],
        w.values[w.slot(String("model.visual.merger.linear_fc1.bias"))][], ctx,
    )
    x = gelu_exact(x, ctx)
    return linear_bias(
        x, w.values[w.slot(String("model.visual.merger.linear_fc2.weight"))][],
        w.values[w.slot(String("model.visual.merger.linear_fc2.bias"))][], ctx,
    )


def minimax_h3_vision_forward_device_traced(
    ref w: MiniMaxH3VisionDeviceWeights,
    pixel_patches: List[Float32],
    grids: List[MiniMaxH3VisionGrid],
    ctx: DeviceContext,
) raises -> MiniMaxH3VisionDeviceTrace:
    """THE device forward, stage-for-stage the host tower's
    `minimax_h3_vision_forward_traced`:
      patch_embed (a LINEAR, trap 1) + interpolated pos embed
      -> 27 x [ LN -> q/k/v -> 2-D rotary (trap 5) -> per-frame attention
                (trap 2) -> proj -> residual -> LN -> fc1 -> gelu_tanh -> fc2
                -> residual ]
         tapping blocks 8/16/24 through the POSTSHUFFLE-norm mergers (trap 3)
      -> final merger (PRESHUFFLE norm, exact-erf GELU, traps 3+4).

    `pixel_patches` is the gated preprocessor's `[num_patches, 1536]` F32 rows
    (pipeline/minimax_h3_vision_preprocess.mojo). They are uploaded as BF16 —
    the same `.to(patch_embed.proj.weight.dtype)` cast torch applies — and the
    whole activation stream stays BF16 (see the file header's DTYPE LAW). NO
    readback here — every stage stays device-resident; the caller reads back
    what it needs."""
    var num_patches = len(pixel_patches) // H3_VIS_PATCH_NUMEL
    var expected_patches = 0
    for g in range(len(grids)):
        expected_patches += grids[g].num_patches()
    if num_patches != expected_patches:
        raise Error(
            String("minimax_h3 vision device: pixel_patches has ")
            + String(num_patches) + " patches, grids imply "
            + String(expected_patches)
        )

    # patch_embed: a LINEAR, 1536 -> 1152 (host header trap 1). BF16 upload =
    # torch's `.to(patch_embed.proj.weight.dtype)` input cast.
    var px = Tensor.from_host(
        pixel_patches, [num_patches, H3_VIS_PATCH_NUMEL], STDtype.BF16, ctx
    )
    var hidden = linear_bias(
        px, w.values[w.slot(String("model.visual.patch_embed.proj.weight"))][],
        w.values[w.slot(String("model.visual.patch_embed.proj.bias"))][], ctx,
    )

    # + interpolated, merge-block-permuted position embed — the HOST tower's
    # own gated `_vis_pos_embeds` (`_torch_linspace_f32` inside), fed the F32
    # upcast of the device table's bf16 bytes. One small D2H of the 2304x1152
    # table; the activations never leave the device.
    var pos_table = w.values[w.slot(String("model.visual.pos_embed.weight"))][].to_host(ctx)
    var pos = _vis_pos_embeds(grids, pos_table)
    if len(pos) != num_patches * H3_VIS_HIDDEN:
        raise Error("minimax_h3 vision device: pos_embed length mismatch")
    # BF16, like torch's own pos_embeds (computed at the pos_embed table's
    # dtype); the add below then rounds bf16 like torch's `hidden + pos`.
    var pos_dev = Tensor.from_host(
        pos, [num_patches, H3_VIS_HIDDEN], STDtype.BF16, ctx
    )
    hidden = add(hidden, pos_dev, ctx)
    var after_patch = hidden.clone(ctx)

    # rotary tables (bf16-replicated chain, see header) + per-frame segments.
    var trig = minimax_h3_vision_device_rope_host(grids)
    var rope = _vis_rope_apply_tables(trig[0], trig[1], num_patches, ctx)
    var cu_seqlens = minimax_h3_vision_cu_seqlens(grids)
    var tap_blocks = minimax_h3_vision_deepstack_tap_blocks()

    var block_00 = Tensor(ctx.enqueue_create_buffer[DType.uint8](4), [1], STDtype.F32)
    var block_08 = Tensor(ctx.enqueue_create_buffer[DType.uint8](4), [1], STDtype.F32)
    var block_16 = Tensor(ctx.enqueue_create_buffer[DType.uint8](4), [1], STDtype.F32)
    var block_24 = Tensor(ctx.enqueue_create_buffer[DType.uint8](4), [1], STDtype.F32)
    var block_26 = Tensor(ctx.enqueue_create_buffer[DType.uint8](4), [1], STDtype.F32)
    var ds0 = Tensor(ctx.enqueue_create_buffer[DType.uint8](4), [1], STDtype.F32)
    var ds1 = Tensor(ctx.enqueue_create_buffer[DType.uint8](4), [1], STDtype.F32)
    var ds2 = Tensor(ctx.enqueue_create_buffer[DType.uint8](4), [1], STDtype.F32)

    for b in range(H3_VIS_DEPTH):
        hidden = _vis_device_block(
            w, b, hidden^, rope.cos_apply, rope.sin_apply, cu_seqlens, ctx
        )
        if b == 0:
            block_00 = hidden.clone(ctx)
        if b == H3_VIS_DEEPSTACK_0:
            block_08 = hidden.clone(ctx)
        if b == H3_VIS_DEEPSTACK_1:
            block_16 = hidden.clone(ctx)
        if b == H3_VIS_DEEPSTACK_2:
            block_24 = hidden.clone(ctx)
        if b == H3_VIS_DEPTH - 1:
            block_26 = hidden.clone(ctx)
        for t in range(len(tap_blocks)):
            if tap_blocks[t] == b:
                var dp = (
                    String("model.visual.deepstack_merger_list.")
                    + String(t) + String(".")
                )
                var tap_out = _vis_device_merger_postshuffle(w, dp, hidden, ctx)
                if t == 0:
                    ds0 = tap_out^
                elif t == 1:
                    ds1 = tap_out^
                else:
                    ds2 = tap_out^

    var embeds = _vis_device_merger_preshuffle(w, hidden, ctx)

    return MiniMaxH3VisionDeviceTrace(
        after_patch^, block_00^, block_08^, block_16^, block_24^, block_26^,
        ds0^, ds1^, ds2^, embeds^,
    )


def minimax_h3_vision_forward_device(
    ref w: MiniMaxH3VisionDeviceWeights,
    pixel_patches: List[Float32],
    grids: List[MiniMaxH3VisionGrid],
    ctx: DeviceContext,
) raises -> MiniMaxH3VisionOutput:
    """Production entry point — the DEVICE twin of the host tower's
    `minimax_h3_vision_forward`: same `pixel_patches`/`grids` inputs, same
    `MiniMaxH3VisionOutput` (embeds `[num_tokens, 5120]` + THREE deepstack
    blocks concatenated in tap order, consumed at LANGUAGE decoder layers
    0/1/2 — see the host file's DEEPSTACK CONTRACT header). A call site swaps
    `minimax_h3_vision_forward(weights, ...)` for
    `minimax_h3_vision_forward_device(device_weights, ..., ctx)`.

    Readbacks happen HERE, once, at the very end: embeds + the three taps —
    exactly the four tensors the streamed text tower needs host-side."""
    var trace = minimax_h3_vision_forward_device_traced(
        w, pixel_patches, grids, ctx
    )
    var embeds = trace.embeds.to_host(ctx)
    var num_tokens = trace.embeds.shape()[0]
    var deepstack = List[Float32](capacity=3 * num_tokens * H3_VIS_OUT_HIDDEN)
    var d0 = trace.ds0.to_host(ctx)
    var d1 = trace.ds1.to_host(ctx)
    var d2 = trace.ds2.to_host(ctx)
    for i in range(len(d0)):
        deepstack.append(d0[i])
    for i in range(len(d1)):
        deepstack.append(d1[i])
    for i in range(len(d2)):
        deepstack.append(d2[i])
    return MiniMaxH3VisionOutput(embeds^, deepstack^, num_tokens)
