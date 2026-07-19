# serenitymojo/models/zimage/controlnet_block.mojo
#
# Z-Image ControlNet CONTROL BLOCK (T2.E): forward + hand-chained backward for
# training, COMPOSING the parity-verified Z-Image main block
# (models/zimage/block.mojo zimage_block_forward/_backward — 19/19 cos>=0.99999
# vs torch) with the two ControlNet projection linears.
#
# REFERENCE (the oracle this mirrors 1:1):
#   diffusers 0.38.0.dev0 models/controlnets/controlnet_z_image.py
#   `ZImageControlTransformerBlock` (the official Alibaba Z-Image ControlNet) —
#   local: /home/alex/.local/lib/python3.12/site-packages/diffusers/models/
#   controlnets/controlnet_z_image.py
#
# This is the DiT ControlNet pattern (Flux/SD3-class), NOT the UNet zero-conv
# pattern: each control block is a COPY of one ZImageTransformerBlock, plus
#   before_proj : nn.Linear(D, D), ZERO-INIT, FIRST control block only
#   after_proj  : nn.Linear(D, D), ZERO-INIT, every control block
# Control stream (diffusers forward, global-modulation branch):
#   block 0:   c = before_proj(c) + x        # x = unified main-stream INPUT
#   body:      c = ZImageTransformerBlock(c) # modulation=True, adaln branch
#   hint:      c_skip = after_proj(c)        # the per-block residual "hint"
# The MAIN transformer consumes hints AFTER each main layer at the places
# listed in control_layers_places (transformer_z_image.py:1032):
#   unified = layer(unified); if layer_idx in samples: unified += hint*scale
#
# CHECKPOINT FORMAT (documented for the loader; diffusers ZImageControlNetModel
# safetensors key layout — `control_layers.{i}` blocks store the SAME inner keys
# as the base `layers.{i}` blocks, so weights.mojo's prefixed loader is reused):
#   control_layers.{i}.attention.to_q.weight            [D, D]
#   control_layers.{i}.attention.to_k.weight            [D, D]
#   control_layers.{i}.attention.to_v.weight            [D, D]
#   control_layers.{i}.attention.to_out.0.weight        [D, D]
#   control_layers.{i}.attention.norm_q.weight          [Dh]
#   control_layers.{i}.attention.norm_k.weight          [Dh]
#   control_layers.{i}.attention_norm1.weight           [D]
#   control_layers.{i}.attention_norm2.weight           [D]
#   control_layers.{i}.ffn_norm1.weight                 [D]
#   control_layers.{i}.ffn_norm2.weight                 [D]
#   control_layers.{i}.feed_forward.w1.weight           [F, D]
#   control_layers.{i}.feed_forward.w3.weight           [F, D]
#   control_layers.{i}.feed_forward.w2.weight           [D, F]
#   control_layers.{i}.adaLN_modulation.0.weight        [4D, min(D,256)]
#   control_layers.{i}.adaLN_modulation.0.bias          [4D]
#   control_layers.{i}.before_proj.weight / .bias       [D, D] / [D]   (i==0 ONLY)
#   control_layers.{i}.after_proj.weight  / .bias       [D, D] / [D]
#   control_all_x_embedder.2-1.weight / .bias           [D, 4*control_in_dim] / [D]
#   control_noise_refiner.{j}.*  (plain ZImageTransformerBlock keys, default
#     add_control_noise_refiner=None config — reuses zimage_block_forward as-is)
# NOTE: `i` indexes control_layers_places (the diffusers ModuleList is built
# from that list); block_id==0 (the before_proj owner) is the FIRST entry and
# the reference asserts 0 in control_layers_places.
#
# TRAINING CONTRACT: in ControlNet training the BASE model is FROZEN; the
# trainable set is exactly { control block copies (incl. their adaLN), the
# before/after projections, control_all_x_embedder, control_noise_refiner }.
# The body grads returned here (ZImageBlockGrads) are therefore TRAINABLE
# grads (unlike the frozen-base main layers).
#
# Mojo 1.0.0b1: `def` + raises; no implicit Tensor copies; host List[Float32]
# carriers at the block boundary (same contract as block.mojo / zimage_stack.mojo).

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import add
from serenitymojo.ops.linalg_backward import linear_backward, LinearGrads
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import from_parts

from serenitymojo.models.zimage.weights import (
    ZImageBlockWeights, load_zimage_block_weights_prefixed_mixed,
)
from serenitymojo.models.zimage.block import (
    ZImageModVecs, ZImageBlockSaved, ZImageBlockGrads,
    zimage_block_forward, zimage_block_backward,
)


comptime TArc = ArcPointer[Tensor]


# ── host helpers (boundary only; same style as block.mojo) ───────────────────
def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(0.0))
    return o^


def _t(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.F32, ctx)


# ── control-block weights: base-block copy + the two projections ─────────────
# before_w/before_b are only READ when is_first (diffusers creates before_proj
# at block_id==0 only); callers pass zero [D,D]/[D] tensors for the rest so the
# struct stays uniform (they are never touched when is_first=False).
struct ZImageControlBlockWeights(Copyable, Movable):
    var base: ZImageBlockWeights   # the copied transformer block (TRAINABLE here)
    var before_w: TArc             # [D, D]  zero-init (fresh) — block 0 only
    var before_b: TArc             # [D]
    var after_w: TArc              # [D, D]  zero-init (fresh)
    var after_b: TArc              # [D]
    var is_first: Bool

    def __init__(
        out self,
        var base: ZImageBlockWeights,
        var before_w: TArc, var before_b: TArc,
        var after_w: TArc, var after_b: TArc,
        is_first: Bool,
    ):
        self.base = base^
        self.before_w = before_w^
        self.before_b = before_b^
        self.after_w = after_w^
        self.after_b = after_b^
        self.is_first = is_first


# Fresh-controlnet projection init (the diffusers `zero_module` convention):
# both projections START AT ZERO so the control branch is a no-op at step 0.
def zero_proj_w(D: Int, ctx: DeviceContext) raises -> TArc:
    return TArc(_t(_zeros(D * D), [D, D], ctx))


def zero_proj_b(D: Int, ctx: DeviceContext) raises -> TArc:
    return TArc(_t(_zeros(D), [D], ctx))


# Load ONE control block from a diffusers ZImageControlNetModel safetensors
# checkpoint. `place_idx` is the index INTO control_layers (the ModuleList
# position, i.e. the i-th entry of control_layers_places). The inner block keys
# are identical to the base model's `layers.{i}` keys, so the proven prefixed
# loader is reused; only before/after_proj are new.
def load_zimage_control_block_weights(
    st: ShardedSafeTensors, place_idx: Int, ctx: DeviceContext
) raises -> ZImageControlBlockWeights:
    var p = String("control_layers.") + String(place_idx)
    var base = load_zimage_block_weights_prefixed_mixed(st, p, ctx)
    var is_first = place_idx == 0
    var before_w: TArc
    var before_b: TArc
    if is_first:
        before_w = TArc(_load_f32(st, p + String(".before_proj.weight"), ctx))
        before_b = TArc(_load_f32(st, p + String(".before_proj.bias"), ctx))
    else:
        # never read; uniform-struct placeholders
        before_w = TArc(_t(_zeros(1), [1, 1], ctx))
        before_b = TArc(_t(_zeros(1), [1], ctx))
    var after_w = TArc(_load_f32(st, p + String(".after_proj.weight"), ctx))
    var after_b = TArc(_load_f32(st, p + String(".after_proj.bias"), ctx))
    return ZImageControlBlockWeights(
        base^, before_w^, before_b^, after_w^, after_b^, is_first
    )


# projections are TRAINED -> load as F32 master copies (unlike the bf16-resident
# frozen base linears).
def _load_f32(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    if t.dtype() == STDtype.F32:
        return t^
    # upcast through host (load-time only)
    var host = t.to_host(ctx)
    var shp = List[Int]()
    for i in range(len(info.shape)):
        shp.append(Int(info.shape[i]))
    return Tensor.from_host(host^, shp^, STDtype.F32, ctx)


# ── saved activations ─────────────────────────────────────────────────────────
struct ZImageControlBlockSaved(Copyable, Movable):
    var c_in: TArc                  # [S,D] control input (pre before_proj)
    var body: ZImageBlockSaved      # body saved (body input is in body.x)
    var c_out: TArc                 # [S,D] body output (after_proj input)

    def __init__(out self, var c_in: TArc, var body: ZImageBlockSaved, var c_out: TArc):
        self.c_in = c_in^
        self.body = body^
        self.c_out = c_out^


struct ZImageControlBlockForward(Movable):
    var hint: List[Float32]     # [S,D] = after_proj(c_out)  (UNSCALED; the
                                # conditioning_scale multiply happens at the
                                # injection site, matching diffusers which
                                # scales in the output dict)
    var c_out: List[Float32]    # [S,D] carried to the next control block
    var saved: ZImageControlBlockSaved

    def __init__(
        out self,
        var hint: List[Float32], var c_out: List[Float32],
        var saved: ZImageControlBlockSaved,
    ):
        self.hint = hint^
        self.c_out = c_out^
        self.saved = saved^


# ── backward result ───────────────────────────────────────────────────────────
struct ZImageControlBlockGrads(Copyable, Movable):
    var d_c: List[Float32]          # [S,D] grad into this block's c input
    var d_x: List[Float32]          # [S,D] grad into the unified input
                                    #       (nonzero ONLY for the first block)
    var body: ZImageBlockGrads      # TRAINABLE copied-block grads (incl. RAW
                                    #       mod-vec grads for its own adaLN)
    var d_before_w: List[Float32]   # [D,D] (empty when not first)
    var d_before_b: List[Float32]   # [D]   (empty when not first)
    var d_after_w: List[Float32]    # [D,D]
    var d_after_b: List[Float32]    # [D]

    def __init__(
        out self,
        var d_c: List[Float32], var d_x: List[Float32],
        var body: ZImageBlockGrads,
        var d_before_w: List[Float32], var d_before_b: List[Float32],
        var d_after_w: List[Float32], var d_after_b: List[Float32],
    ):
        self.d_c = d_c^
        self.d_x = d_x^
        self.body = body^
        self.d_before_w = d_before_w^
        self.d_before_b = d_before_b^
        self.d_after_w = d_after_w^
        self.d_after_b = d_after_b^


# ── FORWARD of one control block ──────────────────────────────────────────────
# c        : [S,D] control stream input (block 0: the embedded+refined control
#            context unified with cap tokens; later blocks: previous c_out).
# x_unified: [S,D] the MAIN stream's unified input (read ONLY when is_first).
# mv       : this control block's OWN RAW modulation chunks (its adaLN output).
# cos/sin  : HALF-WIDTH [S*H, Dh/2] interleaved rope tables (the UNIFIED tables).
def zimage_control_block_forward[
    H: Int, Dh: Int, S: Int
](
    c: List[Float32], x_unified: List[Float32],
    w: ZImageControlBlockWeights, mv: ZImageModVecs,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageControlBlockForward:
    var c_in_t = _t(c.copy(), [S, D], ctx)

    # block 0: c = before_proj(c) + x
    var c_body: List[Float32]
    if w.is_first:
        var bb = Optional[Tensor](w.before_b[].clone(ctx))
        var bp = linear(c_in_t, w.before_w[], bb^, ctx)            # [S,D]
        c_body = add(bp, _t(x_unified.copy(), [S, D], ctx), ctx).to_host(ctx)
    else:
        c_body = c.copy()

    # body = the copied (trainable) Z-Image block, modulated / adaln branch
    var fwd = zimage_block_forward[H, Dh, S](
        c_body, w.base, mv, cos, sin, D, F, eps, ctx,
    )

    # hint = after_proj(c_out)
    var c_out_t = _t(fwd.out.copy(), [S, D], ctx)
    var ab = Optional[Tensor](w.after_b[].clone(ctx))
    var hint = linear(c_out_t, w.after_w[], ab^, ctx).to_host(ctx)

    var saved = ZImageControlBlockSaved(TArc(c_in_t^), fwd.saved.copy(), TArc(c_out_t^))
    return ZImageControlBlockForward(hint^, fwd.out.copy(), saved^)


# ── BACKWARD of one control block ─────────────────────────────────────────────
# d_hint : [S,D] grad into this block's hint (from the main-stream injection
#          site; ALREADY conditioning_scale-scaled by the caller).
# d_c_out: [S,D] grad into c_out from the NEXT control block (zeros for last).
def zimage_control_block_backward[
    H: Int, Dh: Int, S: Int
](
    d_hint: List[Float32], d_c_out: List[Float32],
    w: ZImageControlBlockWeights, mv: ZImageModVecs,
    saved: ZImageControlBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageControlBlockGrads:
    # hint = linear(c_out, after_w) + after_b
    var lb_a = linear_backward(
        _t(d_hint.copy(), [S, D], ctx), saved.c_out[], w.after_w[], S, D, D, ctx,
    )
    var d_after_w = lb_a.d_w.to_host(ctx)
    var d_after_b = lb_a.d_b.to_host(ctx)

    # c_out feeds BOTH after_proj AND the next control block -> SUM
    var d_body_out = add(lb_a.d_x, _t(d_c_out.copy(), [S, D], ctx), ctx).to_host(ctx)

    var bg = zimage_block_backward[H, Dh, S](
        d_body_out, w.base, mv, saved.body, cos, sin, D, F, eps, ctx,
    )
    # bg.d_x = grad into the body input

    var d_c: List[Float32]
    var d_x: List[Float32]
    var d_before_w = List[Float32]()
    var d_before_b = List[Float32]()
    if w.is_first:
        # body_in = linear(c_in, before_w) + before_b + x
        var lb_b = linear_backward(
            _t(bg.d_x.copy(), [S, D], ctx), saved.c_in[], w.before_w[], S, D, D, ctx,
        )
        d_c = lb_b.d_x.to_host(ctx)
        d_before_w = lb_b.d_w.to_host(ctx)
        d_before_b = lb_b.d_b.to_host(ctx)
        d_x = bg.d_x.copy()           # the +x residual passes d straight through
    else:
        d_c = bg.d_x.copy()
        d_x = _zeros(S * D)

    return ZImageControlBlockGrads(
        d_c^, d_x^, bg^, d_before_w^, d_before_b^, d_after_w^, d_after_b^,
    )


# ══════════════════════════════════════════════════════════════════════════════
# CONTROL STACK helpers: run the N control blocks (the diffusers control_layers
# loop) producing the per-place hints, and the reverse pass consuming the
# per-place d_hints. Mirrors the all_c stack/unbind chaining of the reference
# (controlnet_z_image.py:394-429,831-845) without materializing the stack: c is
# carried forward; hints are emitted per block.

struct ZImageControlStackForward(Movable):
    var hints: List[List[Float32]]            # N x [S,D] (UNSCALED)
    var saveds: List[ZImageControlBlockSaved]
    var c_final: List[Float32]                # [S,D]

    def __init__(
        out self,
        var hints: List[List[Float32]],
        var saveds: List[ZImageControlBlockSaved],
        var c_final: List[Float32],
    ):
        self.hints = hints^
        self.saveds = saveds^
        self.c_final = c_final^


struct ZImageControlStackGrads(Movable):
    var blocks: List[ZImageControlBlockGrads]  # in BLOCK ORDER (0..N-1)
    var d_c0: List[Float32]                    # grad into the control input
    var d_x: List[Float32]                     # grad into the unified input

    def __init__(
        out self,
        var blocks: List[ZImageControlBlockGrads],
        var d_c0: List[Float32], var d_x: List[Float32],
    ):
        self.blocks = blocks^
        self.d_c0 = d_c0^
        self.d_x = d_x^


def zimage_control_stack_forward[
    H: Int, Dh: Int, S: Int
](
    c0: List[Float32], x_unified: List[Float32],
    blocks: List[ZImageControlBlockWeights], mods: List[ZImageModVecs],
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageControlStackForward:
    var hints = List[List[Float32]]()
    var saveds = List[ZImageControlBlockSaved]()
    var c = c0.copy()
    for i in range(len(blocks)):
        var fwd = zimage_control_block_forward[H, Dh, S](
            c, x_unified, blocks[i], mods[i], cos, sin, D, F, eps, ctx,
        )
        hints.append(fwd.hint.copy())
        saveds.append(fwd.saved.copy())
        c = fwd.c_out.copy()
    return ZImageControlStackForward(hints^, saveds^, c^)


# d_hints: N x [S,D], each = the grad arriving at that hint's injection site
# (caller applies conditioning_scale before calling, if scale != 1).
def zimage_control_stack_backward[
    H: Int, Dh: Int, S: Int
](
    d_hints: List[List[Float32]],
    blocks: List[ZImageControlBlockWeights], mods: List[ZImageModVecs],
    saveds: List[ZImageControlBlockSaved],
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageControlStackGrads:
    var n = len(blocks)
    var grads_rev = List[ZImageControlBlockGrads]()
    var d_c = _zeros(S * D)                  # no grad into c past the last block
    var d_x_acc = _zeros(S * D)
    var i = n - 1
    while i >= 0:
        var bg = zimage_control_block_backward[H, Dh, S](
            d_hints[i].copy(), d_c.copy(), blocks[i], mods[i], saveds[i],
            cos, sin, D, F, eps, ctx,
        )
        d_c = bg.d_c.copy()
        if blocks[i].is_first:
            for j in range(S * D):
                d_x_acc[j] += bg.d_x[j]
        grads_rev.append(bg^)
        i -= 1
    var ordered = List[ZImageControlBlockGrads]()
    var j = len(grads_rev) - 1
    while j >= 0:
        ordered.append(grads_rev[j].copy())
        j -= 1
    return ZImageControlStackGrads(ordered^, d_c^, d_x_acc^)
