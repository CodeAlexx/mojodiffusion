# serenitymojo/models/scail2/scail2_stack_train_full.mojo
#
# SCAIL-2 i2v FULL training driver (CHUNK 2b): wires the frozen embed /
# conditioning / head around the gated 40-block LoRA stack, adds the NEW head
# backward, and STREAMS the real FP8 base one block at a time (16 GB budget).
#
# What lives here (all additive; reuses CHUNK 1 + CHUNK 2a unchanged):
#   1. Base-loader adapter: Dict[String,TArc] from
#      Scail2A14BFP8Stream.load_block_bf16(i) -> WanI2VBlockWeights (32 keys).
#   2. Per-block frozen modulation: e0 [1,1,6,dim] + blocks.{i}.modulation ->
#      WanModVecs (broadcast to [S,dim]; same chunk6 order as scail2_block.mojo).
#   3. NEW head fwd (faithful reproduction of scail2_streamed_dit._head_video)
#      + NEW head backward `scail2_head_video_backward` (unpatchify3d fold-backward
#      -> reshape -> linear_backward dx (head frozen) -> modulate/LN backward ->
#      SCATTER into ONLY the video rows of the block-40 output grad).
#   4. STREAMING stack fwd/bwd: load each block's base on demand, run the CHUNK-1
#      F32-residual LoRA block, keep only a host F32 per-block INPUT tape; the
#      backward RE-loads + recomputes each block (bounds live VRAM to one block).
#
# Reuses (imported, NOT rebuilt): scail2_block_lora_{forward,backward} +
# Scail2Saved (CHUNK 1), the LoRA set/grad-set/scatter/adamw/save (CHUNK 2a),
# _head_video's own ops (wan22_mod_pre/_chunk2/_unsqueeze2, unpatchify3d).
#
# Mojo 0.26.x+: def not fn; Tensor move-only; TArc carriers; structs .copy()/^.

from std.gpu.host import DeviceContext
from std.collections import Dict, List, Optional
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

from serenitymojo.ops.linear import linear
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.tensor_algebra import add, mul, add_scalar, slice, reshape
from serenitymojo.ops.patchify3d import unpatchify3d
from serenitymojo.ops.linalg_backward import linear_backward, LinearGrads
from serenitymojo.ops.norm_backward import layer_norm_backward_dx

from serenitymojo.models.dit.wan22_dit import (
    wan22_mod_pre, _unsqueeze2, _chunk2,
)
from serenitymojo.models.wan22.wan22_block import (
    WanModVecs, WanBlockWeights, WanI2VBlockWeights, WanI2VBlockLoraGrads,
    wan_modulate_backward,
)
from serenitymojo.models.scail2.scail2_block_train import (
    Scail2Saved, Scail2BlockForward,
    scail2_block_lora_forward, scail2_block_lora_backward,
)
from serenitymojo.models.scail2.scail2_fp8_stream import Scail2A14BFP8Stream
from serenitymojo.models.scail2.scail2_stack_lora import (
    Scail2LoraSet, Scail2LoraGradSet, Scail2StackBackward,
    scail2_total_adapters, _scail2_block_lora_for, _scatter_block_grads,
    _sc_block_base,
)


comptime TArc = ArcPointer[Tensor]


# ── host helpers ──────────────────────────────────────────────────────────────
def _ones_h(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(1.0))
    return o^


def _zeros_h(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(0.0))
    return o^


def _nonfinite(v: List[Float32]) -> Int:
    var bad = 0
    for i in range(len(v)):
        var x = v[i]
        if (x != x) or (x - x != Float32(0.0)):
            bad += 1
    return bad


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BASE-LOADER ADAPTER: Dict[String,TArc] (load_block_bf16) -> WanI2VBlockWeights
# The 32 per-block keys are the short names emitted by Scail2A14BFP8Stream (the
# "blocks.{i}." prefix is already stripped). 31 map to the typed weight struct;
# "modulation" is consumed separately (per-block AdaLN, see below).
# ═══════════════════════════════════════════════════════════════════════════════
def _blk_f32(w: Dict[String, TArc], key: String, ctx: DeviceContext) raises -> List[Float32]:
    if key not in w:
        raise Error(String("SCAIL-2 streamed block missing tensor: ") + key)
    return cast_tensor(w[key][], STDtype.F32, ctx).to_host(ctx)


def scail2_block_weights_from_stream(
    w: Dict[String, TArc], dim: Int, ffn: Int, hd: Int, ctx: DeviceContext,
) raises -> WanI2VBlockWeights:
    var base = WanBlockWeights(
        _blk_f32(w, String("self_attn.q.weight"), ctx),
        _blk_f32(w, String("self_attn.k.weight"), ctx),
        _blk_f32(w, String("self_attn.v.weight"), ctx),
        _blk_f32(w, String("self_attn.o.weight"), ctx),
        _blk_f32(w, String("self_attn.q.bias"), ctx),
        _blk_f32(w, String("self_attn.k.bias"), ctx),
        _blk_f32(w, String("self_attn.v.bias"), ctx),
        _blk_f32(w, String("self_attn.o.bias"), ctx),
        _blk_f32(w, String("self_attn.norm_q.weight"), ctx),
        _blk_f32(w, String("self_attn.norm_k.weight"), ctx),
        _blk_f32(w, String("cross_attn.q.weight"), ctx),
        _blk_f32(w, String("cross_attn.k.weight"), ctx),
        _blk_f32(w, String("cross_attn.v.weight"), ctx),
        _blk_f32(w, String("cross_attn.o.weight"), ctx),
        _blk_f32(w, String("cross_attn.q.bias"), ctx),
        _blk_f32(w, String("cross_attn.k.bias"), ctx),
        _blk_f32(w, String("cross_attn.v.bias"), ctx),
        _blk_f32(w, String("cross_attn.o.bias"), ctx),
        _blk_f32(w, String("cross_attn.norm_q.weight"), ctx),
        _blk_f32(w, String("cross_attn.norm_k.weight"), ctx),
        _blk_f32(w, String("norm3.weight"), ctx),
        _blk_f32(w, String("norm3.bias"), ctx),
        _blk_f32(w, String("ffn.0.weight"), ctx),
        _blk_f32(w, String("ffn.0.bias"), ctx),
        _blk_f32(w, String("ffn.2.weight"), ctx),
        _blk_f32(w, String("ffn.2.bias"), ctx),
        dim, ffn, hd, ctx,
    )
    return WanI2VBlockWeights(
        base^,
        _blk_f32(w, String("cross_attn.k_img.weight"), ctx),
        _blk_f32(w, String("cross_attn.k_img.bias"), ctx),
        _blk_f32(w, String("cross_attn.v_img.weight"), ctx),
        _blk_f32(w, String("cross_attn.v_img.bias"), ctx),
        _blk_f32(w, String("cross_attn.norm_k_img.weight"), ctx),
        dim, ctx,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PER-BLOCK FROZEN MODULATION.
# SCAIL-2 block modulation = _chunk6(block.modulation + e0). e0 is the shared
# [1,1,6,dim] time vector (broadcast across all S tokens). block_mod is [1,6,dim].
# Order (scail2_block.mojo:113-118): shift_sa,scale_sa,gate_sa,shift_ffn,scale_ffn,
# gate_ffn. Each field is broadcast to [S,dim] (the CHUNK-1 block consumes [S,dim]).
# ═══════════════════════════════════════════════════════════════════════════════
def scail2_modvecs_from_stream(
    e0_host: List[Float32], block_mod_h: List[Float32], S: Int, dim: Int,
) -> WanModVecs:
    # e0_host, block_mod_h both [6*dim]: group j (axis over the 6 chunks) at
    # [j*dim + d]. Broadcast the shared [dim] row across all S tokens.
    var shift_sa = List[Float32]()
    var scale_sa = List[Float32]()
    var gate_sa = List[Float32]()
    var shift_ffn = List[Float32]()
    var scale_ffn = List[Float32]()
    var gate_ffn = List[Float32]()
    for _ in range(S):
        for d in range(dim):
            shift_sa.append(e0_host[0 * dim + d] + block_mod_h[0 * dim + d])
            scale_sa.append(e0_host[1 * dim + d] + block_mod_h[1 * dim + d])
            gate_sa.append(e0_host[2 * dim + d] + block_mod_h[2 * dim + d])
            shift_ffn.append(e0_host[3 * dim + d] + block_mod_h[3 * dim + d])
            scale_ffn.append(e0_host[4 * dim + d] + block_mod_h[4 * dim + d])
            gate_ffn.append(e0_host[5 * dim + d] + block_mod_h[5 * dim + d])
    return WanModVecs(shift_sa^, scale_sa^, gate_sa^, shift_ffn^, scale_ffn^, gate_ffn^)


# ═══════════════════════════════════════════════════════════════════════════════
# 3a. HEAD FORWARD (faithful reproduction of scail2_streamed_dit._head_video).
# Standalone (explicit weights) so the G1 parity can run without the FP8 cache.
# img_seq [1,S,dim]; e_head [1,1,dim] F32; head_mod [1,2,dim]; head_w [out*4,dim].
# Returns the video velocity [out_dim, FT, GH*2, GW*2].
# ═══════════════════════════════════════════════════════════════════════════════
def scail2_head_video_forward[
    FT: Int, GH: Int, GW: Int, AR: Int, S: Int
](
    img_seq: Tensor, e_head: Tensor, head_mod: Tensor, head_w: Tensor,
    head_b: Tensor, out_dim: Int, dim: Int, eps: Float32, ctx: DeviceContext,
) raises -> Tensor:
    var video_offset = (AR + 1) * GH * GW
    var video_rows = FT * GH * GW
    var video = slice(img_seq, 1, video_offset, video_rows, ctx)     # [1,VR,dim]
    var head_mod_f32 = cast_tensor(head_mod, STDtype.F32, ctx)
    var head_e = add(head_mod_f32, _unsqueeze2(e_head, 1, dim, ctx), ctx)
    var shift = _chunk2(head_e, 0, 1, dim, ctx)
    var scale = _chunk2(head_e, 1, 1, dim, ctx)
    var head_in = cast_tensor(
        wan22_mod_pre(video, scale, shift, dim, eps, ctx), head_w.dtype(), ctx
    )
    var projected = linear(head_in, head_w, Optional(head_b.clone(ctx)), ctx)
    var proj2 = reshape(projected, [video_rows, out_dim * 4], ctx)
    return unpatchify3d(proj2, out_dim, FT, GH * 2, GW * 2, 1, 2, 2, ctx)


# 3b. unpatchify3d fold-BACKWARD (host permutation; exact adjoint of
# ops/patchify3d.mojo::unpatchify3d, which is a bijective gather). d_out is
# [C,F,H,W] row-major host F32; result is d_seq [L, C*pf*ph*pw] host F32.
def _unpatchify3d_backward_host(
    d_out: List[Float32], C: Int, F: Int, Hh: Int, Ww: Int,
    pf: Int, ph: Int, pw: Int,
) raises -> List[Float32]:
    var FO = F // pf
    var HO = Hh // ph
    var WO = Ww // pw
    var L = FO * HO * WO
    var PD = C * pf * ph * pw
    var d_seq = List[Float32]()
    d_seq.resize(L * PD, Float32(0.0))
    var HW = Hh * Ww
    var FHW = F * HW
    for c in range(C):
        for f in range(F):
            var fi = f // pf
            var pfi = f % pf
            for h in range(Hh):
                var hi = h // ph
                var phi = h % ph
                for w in range(Ww):
                    var wi = w // pw
                    var pwi = w % pw
                    var patch = fi * HO * WO + hi * WO + wi
                    var src = pfi * (ph * pw * C) + phi * (pw * C) + pwi * C + c
                    d_seq[patch * PD + src] = d_out[c * FHW + f * HW + h * Ww + w]
    return d_seq^


# 3c. HEAD BACKWARD (NEW). Reverse of scail2_head_video_forward. Head weights
# (modulation + head.head linear) are FROZEN -> only the input-grad path is
# produced. Scatters d_video into ONLY the video rows of the [S,dim] block-40
# output grad; ref/pose/additional rows get ZERO. Returns d_seq host [S*dim] F32.
def scail2_head_video_backward[
    FT: Int, GH: Int, GW: Int, AR: Int, S: Int
](
    d_out: Tensor,               # [out_dim, FT, GH*2, GW*2] velocity grad
    video_region: List[Float32], # [video_rows, dim] F32 (block-40 video rows)
    e_head: Tensor, head_mod: Tensor, head_w: Tensor,
    out_dim: Int, dim: Int, eps: Float32, ctx: DeviceContext,
) raises -> List[Float32]:
    var video_offset = (AR + 1) * GH * GW
    var video_rows = FT * GH * GW

    # per-token head shift/scale broadcast to [video_rows, dim] (frozen).
    var e_head_h = cast_tensor(e_head, STDtype.F32, ctx).to_host(ctx)      # [dim]
    var head_mod_h = cast_tensor(head_mod, STDtype.F32, ctx).to_host(ctx)  # [2*dim]
    var shift_h = List[Float32]()
    var scale_h = List[Float32]()
    for _ in range(video_rows):
        for d in range(dim):
            shift_h.append(head_mod_h[0 * dim + d] + e_head_h[d])
            scale_h.append(head_mod_h[1 * dim + d] + e_head_h[d])
    var scale_t = Tensor.from_host(scale_h.copy(), [video_rows, dim], STDtype.F32, ctx)

    # recompute head_in (F32): (1+scale)*LN_noaffine(video) + shift.
    var video_t = Tensor.from_host(video_region.copy(), [video_rows, dim], STDtype.F32, ctx)
    var ones_t = Tensor.from_host(_ones_h(dim), [dim], STDtype.F32, ctx)
    var zeros_t = Tensor.from_host(_zeros_h(dim), [dim], STDtype.F32, ctx)
    var ln = layer_norm(video_t, ones_t, zeros_t, eps, ctx)               # F32
    var sc1 = add_scalar(scale_t, Float32(1.0), ctx)
    var head_in = add(mul(ln, sc1, ctx),
                      Tensor.from_host(shift_h^, [video_rows, dim], STDtype.F32, ctx), ctx)

    # unpatchify fold-backward -> [video_rows, out*4] grad.
    var d_out_h = cast_tensor(d_out, STDtype.F32, ctx).to_host(ctx)
    var d_proj_h = _unpatchify3d_backward_host(
        d_out_h, out_dim, FT, GH * 2, GW * 2, 1, 2, 2,
    )
    var d_proj = Tensor.from_host(d_proj_h^, [video_rows, out_dim * 4], STDtype.F32, ctx)

    # head.head linear backward (dx only; weights frozen). head_in F32, weight
    # bf16(real)/f32(oracle) -> the mixed-base linear_backward path returns F32 dx.
    var lbh = linear_backward(d_proj, head_in, head_w, video_rows, dim, out_dim * 4, ctx)
    # modulate backward -> d_ln (discard d_scale/d_shift; frozen).
    var mbh = wan_modulate_backward(lbh.d_x, ln, scale_t, ctx)
    var d_video = layer_norm_backward_dx(mbh.d_ln, video_t, ones_t, eps, ctx)
    var d_video_h = d_video.to_host(ctx)                                  # [VR,dim]

    # scatter into ONLY the video rows of the full [S,dim] block-40 output grad.
    var d_seq = List[Float32]()
    d_seq.resize(S * dim, Float32(0.0))
    for r in range(video_rows):
        var dst = (video_offset + r) * dim
        var srcb = r * dim
        for d in range(dim):
            d_seq[dst + d] = d_video_h[srcb + d]
    return d_seq^


# ═══════════════════════════════════════════════════════════════════════════════
# 4. STREAMING FULL-DEPTH STACK (loads one FP8 base block at a time).
# Forward keeps only a host F32 per-block INPUT tape; the backward RE-loads each
# block and RECOMPUTES its forward (bounds live VRAM to a single block).
# ═══════════════════════════════════════════════════════════════════════════════
struct Scail2StreamedStackForward(Movable):
    var x_out: List[Float32]                # [S,dim] F32 (pre-head)
    var block_inputs: List[List[Float32]]   # per-block F32 input tape

    def __init__(
        out self, var x_out: List[Float32], var block_inputs: List[List[Float32]],
    ):
        self.x_out = x_out^
        self.block_inputs = block_inputs^


def scail2_stack_lora_forward_streamed[
    H: Int, DH: Int, S: Int, TXT: Int, IMG: Int
](
    var sequence: List[Float32],
    e0_host: List[Float32],
    context_txt: List[Float32], context_img: List[Float32],
    cos: Tensor, sin: Tensor,
    mut stream: Scail2A14BFP8Stream, num_layers: Int, lora_set: Scail2LoraSet,
    dim: Int, ffn: Int, hd: Int, eps: Float32, ctx: DeviceContext,
) raises -> Scail2StreamedStackForward:
    var x = sequence^
    var block_inputs = List[List[Float32]]()
    for bi in range(num_layers):
        block_inputs.append(x.copy())
        var wd = stream.load_block_bf16(bi, ctx)
        var w = scail2_block_weights_from_stream(wd, dim, ffn, hd, ctx)
        var block_mod_h = cast_tensor(wd["modulation"][], STDtype.F32, ctx).to_host(ctx)
        var mv = scail2_modvecs_from_stream(e0_host, block_mod_h, S, dim)
        var bl = _scail2_block_lora_for(lora_set, bi)
        var fwd = scail2_block_lora_forward[S, TXT, IMG, H, DH](
            x.copy(), context_txt.copy(), context_img.copy(),
            mv, w, bl, cos, sin, dim, ffn, eps, ctx,
        )
        x = fwd.x_out.copy()
        # w / wd / fwd.saved drop here -> block VRAM freed before the next load.
    return Scail2StreamedStackForward(x^, block_inputs^)


def scail2_stack_lora_backward_streamed[
    H: Int, DH: Int, S: Int, TXT: Int, IMG: Int
](
    var d_out: List[Float32],
    e0_host: List[Float32],
    context_txt: List[Float32], context_img: List[Float32],
    cos: Tensor, sin: Tensor,
    mut stream: Scail2A14BFP8Stream, num_layers: Int, lora_set: Scail2LoraSet,
    fwd_state: Scail2StreamedStackForward,
    dim: Int, ffn: Int, hd: Int, eps: Float32, ctx: DeviceContext,
) raises -> Scail2StackBackward:
    var n_adapters = scail2_total_adapters(lora_set)
    var d_a = List[List[Float32]]()
    var d_b = List[List[Float32]]()
    for _ in range(n_adapters):
        d_a.append(List[Float32]())
        d_b.append(List[Float32]())
    var nonfinite = 0

    var d_cur = d_out^
    var bi = num_layers - 1
    while bi >= 0:
        var wd = stream.load_block_bf16(bi, ctx)
        var w = scail2_block_weights_from_stream(wd, dim, ffn, hd, ctx)
        var block_mod_h = cast_tensor(wd["modulation"][], STDtype.F32, ctx).to_host(ctx)
        var mv = scail2_modvecs_from_stream(e0_host, block_mod_h, S, dim)
        var bl = _scail2_block_lora_for(lora_set, bi)

        # RECOMPUTE this block's forward from its saved F32 input, then backward.
        var rc = scail2_block_lora_forward[S, TXT, IMG, H, DH](
            fwd_state.block_inputs[bi].copy(), context_txt.copy(), context_img.copy(),
            mv, w, bl, cos, sin, dim, ffn, eps, ctx,
        )
        var bg = scail2_block_lora_backward[S, TXT, IMG, H, DH](
            d_cur.copy(), mv, w, bl, rc.saved, cos, sin, dim, ffn, eps, ctx,
        )
        d_cur = bg.base.base.d_x.copy()
        nonfinite += _scatter_block_grads(d_a, d_b, _sc_block_base(bi), bg)
        bi -= 1

    var grads = Scail2LoraGradSet(d_a^, d_b^, nonfinite)
    return Scail2StackBackward(d_cur^, grads^)
