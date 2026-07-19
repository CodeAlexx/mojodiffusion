# serenitymojo/training/controlnet_zimage.mojo — T2.E Z-Image ControlNet
# TRAINER side: parameter store + init + per-step upload + control forward /
# backward COMPOSITION + AdamW group + diffusers-format save/load.
#
# COMPOSES (rebuilds NOTHING):
#   * models/zimage/controlnet_block.mojo — the GATED control stack fwd/bwd
#     (zimage_controlnet_block_parity 46/46 cos>=0.99999 vs diffusers;
#     zimage_controlnet_step_smoke e2e training semantics).
#   * models/zimage/block.mojo zimage_block_forward/_backward — the
#     control_noise_refiner blocks (plain modulated ZImageTransformerBlock,
#     the diffusers default add_control_noise_refiner=None branch).
#   * ops/linear + linalg_backward — control_all_x_embedder.
#
# REFERENCE: diffusers 0.38.0.dev0 ZImageControlNetModel
#   (models/controlnets/controlnet_z_image.py). Forward (training-relevant,
#   default config add_control_noise_refiner=None):
#     control_context = patchify(control_latent)                (:701)
#     control_context = control_all_x_embedder(control_context) (:703)
#     control_context[pad rows] = x_pad_token                   (:705)
#     for layer in control_noise_refiner: c = layer(c, x_mask, x_rope, adaln) (:813-821)
#     c_unified = cat(c[:x_len], cap_feats[:cap_len])           (:823-829)
#     for layer in control_layers: c_unified = layer(c_unified, unified, ...) (:831-839)
#     hints -> {place: hint * conditioning_scale}                (:841-844)
#   The control image enters as a VAE latent normalized (x - shift) * scale —
#   pipeline_z_image_controlnet.py:550-551 (the SAME normalization as the
#   training latent; the trainer applies it).
#
# TRAINABLE SET (the full ZImageControlNetModel param surface):
#   control_layers.{i}.: 13 body tensors + adaLN_modulation.0.{weight,bias}
#                        + before_proj (i==0) + after_proj   (zero-init fresh)
#   control_noise_refiner.{j}.: 13 body tensors + adaLN_modulation.0.{w,b}
#   control_all_x_embedder.2-1.{weight,bias}
# INIT (controlnet_checkpoint == ""): body/adaLN/x_embedder = COPIES of the
# base transformer's layers.{place}/noise_refiner.{j}/all_x_embedder.2-1 (the
# standard ControlNet copy-from-base init; the diffusers blocks are literal
# copies of ZImageTransformerBlock), projections ZERO (zero_module).
#
# CONTROL PLACES: control_layers_places derived as evenly spaced
# [i * depth // n] (includes the mandatory 0 — the reference asserts
# `0 in control_layers_places`). A loaded controlnet_checkpoint must carry the
# same layer count; its places are taken from the trainer config the same way.
#
# OPTIMIZER: host AdamW (bias-corrected, decoupled wd) over F32 masters —
# the control group is ITS OWN optimizer group, fully separate from the LoRA
# fused-AdamW state (which the controlnet driver never touches). v1 host loop
# uses unsafe_ptr inner loops; device-resident control AdamW is a perf
# follow-up (documented in TIER2_PARITY_CAMPAIGN T2.E).
#
# Mojo 1.0.0b1: def + raises; no implicit Tensor copies; host List[Float32]
# carriers at module boundaries.

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer, alloc
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.ffi import (
    sys_system, sys_open, sys_close, sys_pwrite, BytePtr,
    O_WRONLY, O_CREAT, O_TRUNC,
)
from serenitymojo.ops.linear import linear
from serenitymojo.ops.linalg_backward import linear_backward
from serenitymojo.models.zimage.weights import ZImageBlockWeights
from serenitymojo.models.zimage.block import (
    ZImageModVecs, ZImageBlockSaved, ZImageBlockGrads,
    zimage_block_forward, zimage_block_backward,
)
from serenitymojo.models.zimage.controlnet_block import (
    ZImageControlBlockWeights, ZImageControlBlockSaved, ZImageControlBlockGrads,
    zimage_control_stack_forward, zimage_control_stack_backward,
)
from serenitymojo.models.zimage.real_weights import build_block_modvecs


comptime TArc = ArcPointer[Tensor]
comptime ADALN_IN = 256          # min(dim, ADALN_EMBED_DIM) for dim=3840


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32](capacity=n)
    for _ in range(n):
        o.append(Float32(0.0))
    return o^


def zimage_cn_places(n: Int, depth: Int) raises -> List[Int]:
    """control_layers_places for n control blocks over `depth` main layers:
    evenly spaced [i*depth//n], strictly increasing, includes 0 (the reference
    asserts 0 in control_layers_places)."""
    if n <= 0 or n > depth:
        raise Error("zimage_cn_places: need 0 < n <= depth")
    var o = List[Int]()
    for i in range(n):
        o.append(i * depth // n)
    return o^


# ── named F32 master-parameter store (+ AdamW moments) ────────────────────────
struct ZImageCnParams(Movable):
    var names: List[String]      # diffusers ZImageControlNetModel state-dict keys
    var shapes: List[List[Int]]
    var p: List[List[Float32]]
    var m: List[List[Float32]]
    var v: List[List[Float32]]
    var places: List[Int]
    var n_ctl: Int
    var n_nr: Int

    def __init__(out self, var places: List[Int], n_nr: Int):
        self.names = List[String]()
        self.shapes = List[List[Int]]()
        self.p = List[List[Float32]]()
        self.m = List[List[Float32]]()
        self.v = List[List[Float32]]()
        self.n_ctl = len(places)
        self.places = places^
        self.n_nr = n_nr

    def add(mut self, name: String, var shape: List[Int], var vals: List[Float32]) raises:
        var n = 1
        for i in range(len(shape)):
            n *= shape[i]
        if n != len(vals):
            raise Error(String("ZImageCnParams.add: shape/value mismatch for ") + name)
        self.names.append(name)
        self.shapes.append(shape^)
        self.m.append(_zeros(n))
        self.v.append(_zeros(n))
        self.p.append(vals^)

    def idx(self, name: String) raises -> Int:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return i
        raise Error(String("ZImageCnParams.idx: no param named ") + name)

    def l1(self, name: String) raises -> Float64:
        var i = self.idx(name)
        var s = 0.0
        for j in range(len(self.p[i])):
            var x = Float64(self.p[i][j])
            s += x if x >= 0.0 else -x
        return s


def _host_f32_sharded(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tuple[List[Float32], List[Int]]:
    """Load one base-checkpoint tensor to a host F32 list (bf16 upcast at the
    load boundary only — these become TRAINED F32 masters)."""
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    var vals = t.to_host(ctx)
    var shp = List[Int]()
    for i in range(len(info.shape)):
        shp.append(Int(info.shape[i]))
    return (vals^, shp^)


def _body_key_pairs(prefix: String) -> Tuple[List[String], List[String]]:
    """(control-key, base-relative-key) pairs for the 15 per-block tensors, in
    ZImageBlockWeights field order + adaLN. Key names = the diffusers block
    submodule paths (identical inner layout to the base layers.{i} blocks —
    weights.mojo load_zimage_block_weights_prefixed_mixed uses the same)."""
    var rel = List[String]()
    rel.append(String("attention_norm1.weight"))
    rel.append(String("attention.to_q.weight"))
    rel.append(String("attention.to_k.weight"))
    rel.append(String("attention.to_v.weight"))
    rel.append(String("attention.to_out.0.weight"))
    rel.append(String("attention.norm_q.weight"))
    rel.append(String("attention.norm_k.weight"))
    rel.append(String("attention_norm2.weight"))
    rel.append(String("ffn_norm1.weight"))
    rel.append(String("feed_forward.w1.weight"))
    rel.append(String("feed_forward.w3.weight"))
    rel.append(String("feed_forward.w2.weight"))
    rel.append(String("ffn_norm2.weight"))
    rel.append(String("adaLN_modulation.0.weight"))
    rel.append(String("adaLN_modulation.0.bias"))
    var full = List[String]()
    for i in range(len(rel)):
        full.append(prefix + rel[i])
    return (full^, rel^)


def zimage_cn_params_init_from_base(
    st: ShardedSafeTensors, var places: List[Int], n_nr: Int,
    D: Int, ctx: DeviceContext,
) raises -> ZImageCnParams:
    """Fresh controlnet init: control blocks = COPIES of the base
    layers.{place} blocks (incl. their adaLN), control_noise_refiner = copies
    of the base noise_refiner blocks, control_all_x_embedder = copy of the
    base all_x_embedder (control_in_dim == in_channels == 16), projections
    ZERO (the diffusers zero_module convention — control branch is a no-op at
    step 0; gradient cascade gated in zimage_controlnet_step_smoke)."""
    var n_ctl = len(places)
    var params = ZImageCnParams(places.copy(), n_nr)
    for i in range(n_ctl):
        var cp = String("control_layers.") + String(i) + String(".")
        var bp = String("layers.") + String(params.places[i]) + String(".")
        var keys = _body_key_pairs(cp)
        for k in range(len(keys[0])):
            var lv = _host_f32_sharded(st, bp + keys[1][k], ctx)
            params.add(keys[0][k].copy(), lv[1].copy(), lv[0].copy())
        if i == 0:
            params.add(cp + String("before_proj.weight"), [D, D], _zeros(D * D))
            params.add(cp + String("before_proj.bias"), [D], _zeros(D))
        params.add(cp + String("after_proj.weight"), [D, D], _zeros(D * D))
        params.add(cp + String("after_proj.bias"), [D], _zeros(D))
    for j in range(n_nr):
        var cp = String("control_noise_refiner.") + String(j) + String(".")
        var bp = String("noise_refiner.") + String(j) + String(".")
        var keys = _body_key_pairs(cp)
        for k in range(len(keys[0])):
            var lv = _host_f32_sharded(st, bp + keys[1][k], ctx)
            params.add(keys[0][k].copy(), lv[1].copy(), lv[0].copy())
    var xw = _host_f32_sharded(st, String("all_x_embedder.2-1.weight"), ctx)
    params.add(String("control_all_x_embedder.2-1.weight"), xw[1].copy(), xw[0].copy())
    var xb = _host_f32_sharded(st, String("all_x_embedder.2-1.bias"), ctx)
    params.add(String("control_all_x_embedder.2-1.bias"), xb[1].copy(), xb[0].copy())
    return params^


def zimage_cn_params_load_checkpoint(
    mut params: ZImageCnParams, path: String, ctx: DeviceContext
) raises:
    """Overwrite the masters from a saved ZImageControlNetModel safetensors
    (key-for-key; missing keys fail loud, shapes checked)."""
    var st = SafeTensors.open(path)
    for i in range(len(params.names)):
        var info = st.tensor_info(params.names[i])
        var bytes = st.tensor_bytes(params.names[i])
        var tv = from_parts(info.dtype, info.shape.copy(), bytes)
        var t = Tensor.from_view(tv, ctx)
        var vals = t.to_host(ctx)
        if len(vals) != len(params.p[i]):
            raise Error(
                String("controlnet_checkpoint: size mismatch for ") + params.names[i]
            )
        params.p[i] = vals^


# ── per-step device upload ────────────────────────────────────────────────────
def _t_dev(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_host(vals, shape^, STDtype.F32, ctx))


struct ZImageCnDevice(Movable):
    var blocks: List[ZImageControlBlockWeights]
    var nr: List[ZImageBlockWeights]
    var x_emb_w: TArc            # [D, p*p*C]
    var x_emb_b: TArc            # [D]
    var blk_mods: List[ZImageModVecs]   # per control block (its OWN adaLN)
    var nr_mods: List[ZImageModVecs]    # per control NR block

    def __init__(
        out self,
        var blocks: List[ZImageControlBlockWeights],
        var nr: List[ZImageBlockWeights],
        var x_emb_w: TArc, var x_emb_b: TArc,
        var blk_mods: List[ZImageModVecs], var nr_mods: List[ZImageModVecs],
    ):
        self.blocks = blocks^
        self.nr = nr^
        self.x_emb_w = x_emb_w^
        self.x_emb_b = x_emb_b^
        self.blk_mods = blk_mods^
        self.nr_mods = nr_mods^


def _upload_body(
    params: ZImageCnParams, prefix: String, ctx: DeviceContext
) raises -> ZImageBlockWeights:
    var keys = _body_key_pairs(prefix)
    # field order == _body_key_pairs order (the first 13 are the body)
    var ts = List[TArc]()
    for k in range(13):
        var i = params.idx(keys[0][k])
        ts.append(_t_dev(params.p[i].copy(), params.shapes[i].copy(), ctx))
    return ZImageBlockWeights(
        ts[0].copy(), ts[1].copy(), ts[2].copy(), ts[3].copy(), ts[4].copy(),
        ts[5].copy(), ts[6].copy(), ts[7].copy(), ts[8].copy(),
        ts[9].copy(), ts[10].copy(), ts[11].copy(), ts[12].copy(),
    )


def _build_mods(
    params: ZImageCnParams, prefix: String, adaln: Tensor, D: Int,
    ctx: DeviceContext,
) raises -> ZImageModVecs:
    var wi = params.idx(prefix + String("adaLN_modulation.0.weight"))
    var bi = params.idx(prefix + String("adaLN_modulation.0.bias"))
    var w_t = Tensor.from_host(
        params.p[wi].copy(), params.shapes[wi].copy(), STDtype.F32, ctx
    )
    var b_t = Tensor.from_host(
        params.p[bi].copy(), params.shapes[bi].copy(), STDtype.F32, ctx
    )
    return build_block_modvecs(w_t, b_t, adaln, D, ctx)


def zimage_cn_upload(
    params: ZImageCnParams, adaln: Tensor, D: Int, ctx: DeviceContext
) raises -> ZImageCnDevice:
    """Upload the current masters as device weights + this step's mod-vecs
    (each control block runs its OWN adaLN on the shared adaln embedding —
    controlnet_z_image.py:405)."""
    var blocks = List[ZImageControlBlockWeights]()
    var blk_mods = List[ZImageModVecs]()
    for i in range(params.n_ctl):
        var cp = String("control_layers.") + String(i) + String(".")
        var base = _upload_body(params, cp, ctx)
        var before_w: TArc
        var before_b: TArc
        if i == 0:
            var bwi = params.idx(cp + String("before_proj.weight"))
            var bbi = params.idx(cp + String("before_proj.bias"))
            before_w = _t_dev(params.p[bwi].copy(), params.shapes[bwi].copy(), ctx)
            before_b = _t_dev(params.p[bbi].copy(), params.shapes[bbi].copy(), ctx)
        else:
            before_w = _t_dev(_zeros(1), [1, 1], ctx)
            before_b = _t_dev(_zeros(1), [1], ctx)
        var awi = params.idx(cp + String("after_proj.weight"))
        var abi = params.idx(cp + String("after_proj.bias"))
        var after_w = _t_dev(params.p[awi].copy(), params.shapes[awi].copy(), ctx)
        var after_b = _t_dev(params.p[abi].copy(), params.shapes[abi].copy(), ctx)
        blocks.append(ZImageControlBlockWeights(
            base^, before_w^, before_b^, after_w^, after_b^, i == 0,
        ))
        blk_mods.append(_build_mods(params, cp, adaln, D, ctx))
    var nr = List[ZImageBlockWeights]()
    var nr_mods = List[ZImageModVecs]()
    for j in range(params.n_nr):
        var cp = String("control_noise_refiner.") + String(j) + String(".")
        nr.append(_upload_body(params, cp, ctx))
        nr_mods.append(_build_mods(params, cp, adaln, D, ctx))
    var xwi = params.idx(String("control_all_x_embedder.2-1.weight"))
    var xbi = params.idx(String("control_all_x_embedder.2-1.bias"))
    var x_emb_w = _t_dev(params.p[xwi].copy(), params.shapes[xwi].copy(), ctx)
    var x_emb_b = _t_dev(params.p[xbi].copy(), params.shapes[xbi].copy(), ctx)
    return ZImageCnDevice(
        blocks^, nr^, x_emb_w^, x_emb_b^, blk_mods^, nr_mods^,
    )


# ── control forward / backward (the ZImageControlNetModel.forward body) ──────
struct ZImageCnForwardState(Movable):
    var hints: List[List[Float32]]            # n_ctl x [S*D] UNSCALED
    var stack_saveds: List[ZImageControlBlockSaved]
    var nr_saveds: List[ZImageBlockSaved]
    var ctl_patches: TArc                     # [N_IMG, in_dim] embedder input

    def __init__(
        out self,
        var hints: List[List[Float32]],
        var stack_saveds: List[ZImageControlBlockSaved],
        var nr_saveds: List[ZImageBlockSaved],
        var ctl_patches: TArc,
    ):
        self.hints = hints^
        self.stack_saveds = stack_saveds^
        self.nr_saveds = nr_saveds^
        self.ctl_patches = ctl_patches^


def zimage_cn_forward[
    H: Int, Dh: Int, N_IMG: Int, S: Int
](
    ctl_patches_h: List[Float32],     # [N_IMG * in_dim] patchified control latent
    xs: List[Float32],                # [N_IMG*D] NR-refined image stream (frozen base)
    cs: List[Float32],                # [N_TXT*D] CR-refined caption stream (frozen base)
    n_img_real: Int,
    x_pad_h: List[Float32],           # [D] learned x_pad_token (frozen base)
    dev: ZImageCnDevice,
    x_cos: Tensor, x_sin: Tensor,
    uni_cos: Tensor, uni_sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageCnForwardState:
    var in_dim = len(ctl_patches_h) // N_IMG
    var patches_t = Tensor.from_host(
        ctl_patches_h.copy(), [N_IMG, in_dim], STDtype.F32, ctx
    )
    var emb = linear(
        patches_t, dev.x_emb_w[], Optional[Tensor](dev.x_emb_b[].clone(ctx)), ctx
    )
    var c = emb.to_host(ctx)
    # pad rows -> x_pad_token (controlnet_z_image.py:705)
    for r in range(n_img_real, N_IMG):
        for q in range(D):
            c[r * D + q] = x_pad_h[q]

    var nr_saveds = List[ZImageBlockSaved]()
    for j in range(len(dev.nr)):
        var f = zimage_block_forward[H, Dh, N_IMG](
            c.copy(), dev.nr[j], dev.nr_mods[j], x_cos, x_sin, D, F, eps, ctx,
        )
        nr_saveds.append(f.saved.copy())
        c = f.out.copy()

    # unify the refined control context with the (frozen) refined captions
    # (controlnet_z_image.py:823-829), and the main unified input for block 0's
    # before_proj residual.
    var c0 = c.copy()
    for i in range(len(cs)):
        c0.append(cs[i])
    var u0 = xs.copy()
    for i in range(len(cs)):
        u0.append(cs[i])

    var sf = zimage_control_stack_forward[H, Dh, S](
        c0, u0, dev.blocks, dev.blk_mods, uni_cos, uni_sin, D, F, eps, ctx,
    )
    return ZImageCnForwardState(
        sf.hints.copy(), sf.saveds.copy(), nr_saveds^, TArc(patches_t^),
    )


struct ZImageCnGrads(Movable):
    var blocks: List[ZImageControlBlockGrads]   # block order 0..n_ctl-1
    var nr: List[ZImageBlockGrads]              # block order 0..n_nr-1
    var d_x_emb_w: List[Float32]
    var d_x_emb_b: List[Float32]

    def __init__(
        out self,
        var blocks: List[ZImageControlBlockGrads],
        var nr: List[ZImageBlockGrads],
        var d_x_emb_w: List[Float32], var d_x_emb_b: List[Float32],
    ):
        self.blocks = blocks^
        self.nr = nr^
        self.d_x_emb_w = d_x_emb_w^
        self.d_x_emb_b = d_x_emb_b^


def zimage_cn_backward[
    H: Int, Dh: Int, N_IMG: Int, S: Int
](
    d_hints: List[List[Float32]],     # ALREADY conditioning_scale-scaled
    fwdst: ZImageCnForwardState,
    n_img_real: Int,
    dev: ZImageCnDevice,
    x_cos: Tensor, x_sin: Tensor,
    uni_cos: Tensor, uni_sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageCnGrads:
    var sg = zimage_control_stack_backward[H, Dh, S](
        d_hints, dev.blocks, dev.blk_mods, fwdst.stack_saveds,
        uni_cos, uni_sin, D, F, eps, ctx,
    )
    # sg.d_x (grad into the main unified input) feeds the FROZEN base
    # refiners/embedders -> discard. Caption rows of d_c0 likewise (frozen
    # context refiner). The IMAGE rows backprop through the control NR blocks.
    var d_c = List[Float32](capacity=N_IMG * D)
    for j in range(N_IMG * D):
        d_c.append(sg.d_c0[j])

    var nr_rev = List[ZImageBlockGrads]()
    var j = len(dev.nr) - 1
    while j >= 0:
        var bg = zimage_block_backward[H, Dh, N_IMG](
            d_c.copy(), dev.nr[j], dev.nr_mods[j], fwdst.nr_saveds[j],
            x_cos, x_sin, D, F, eps, ctx,
        )
        d_c = bg.d_x.copy()
        nr_rev.append(bg^)
        j -= 1
    var nr_grads = List[ZImageBlockGrads]()
    var jj = len(nr_rev) - 1
    while jj >= 0:
        nr_grads.append(nr_rev[jj].copy())
        jj -= 1

    # pad rows were OVERWRITTEN with the (frozen) x_pad_token -> their grad
    # does not reach the embedder.
    for r in range(n_img_real, N_IMG):
        for q in range(D):
            d_c[r * D + q] = Float32(0.0)
    var in_dim = fwdst.ctl_patches[].shape()[1]
    var lb = linear_backward(
        Tensor.from_host(d_c, [N_IMG, D], STDtype.F32, ctx),
        fwdst.ctl_patches[], dev.x_emb_w[], N_IMG, in_dim, D, ctx,
    )
    var d_w = lb.d_w.to_host(ctx)
    var d_b = lb.d_b.to_host(ctx)
    return ZImageCnGrads(sg.blocks.copy(), nr_grads^, d_w^, d_b^)


# ── AdamW group (host, F32 masters) ───────────────────────────────────────────
def _grad_set(
    mut g: List[List[Float32]], params: ZImageCnParams,
    name: String, vals: List[Float32],
) raises:
    var i = params.idx(name)
    if len(vals) != len(params.p[i]):
        raise Error(String("cn grad size mismatch for ") + name)
    g[i] = vals.copy()


def _adaln_grad_w(
    d_sm: List[Float32], d_gm: List[Float32],
    d_sp: List[Float32], d_gp: List[Float32],
    adaln_h: List[Float32], D: Int,
) -> Tuple[List[Float32], List[Float32]]:
    """adaLN_modulation.0 grads from the RAW mod-vec grads: mod = W@adaln + b
    with raw layout [4D] = scale_msa|gate_msa|scale_mlp|gate_mlp (the
    build_block_modvecs slicing). d_W = d_mod ⊗ adaln, d_b = d_mod."""
    var K = len(adaln_h)
    var d_b = List[Float32](capacity=4 * D)
    for r in range(D):
        d_b.append(d_sm[r])
    for r in range(D):
        d_b.append(d_gm[r])
    for r in range(D):
        d_b.append(d_sp[r])
    for r in range(D):
        d_b.append(d_gp[r])
    var d_w = List[Float32](capacity=4 * D * K)
    for r in range(4 * D):
        var dm = d_b[r]
        for c in range(K):
            d_w.append(dm * adaln_h[c])
    return (d_w^, d_b^)


def _scatter_body_grads(
    mut g: List[List[Float32]], params: ZImageCnParams, prefix: String,
    bg: ZImageBlockGrads, adaln_h: List[Float32], D: Int,
) raises:
    _grad_set(g, params, prefix + String("attention_norm1.weight"), bg.d_n1)
    _grad_set(g, params, prefix + String("attention.to_q.weight"), bg.d_wq)
    _grad_set(g, params, prefix + String("attention.to_k.weight"), bg.d_wk)
    _grad_set(g, params, prefix + String("attention.to_v.weight"), bg.d_wv)
    _grad_set(g, params, prefix + String("attention.to_out.0.weight"), bg.d_wo)
    _grad_set(g, params, prefix + String("attention.norm_q.weight"), bg.d_q_norm)
    _grad_set(g, params, prefix + String("attention.norm_k.weight"), bg.d_k_norm)
    _grad_set(g, params, prefix + String("attention_norm2.weight"), bg.d_n2)
    _grad_set(g, params, prefix + String("ffn_norm1.weight"), bg.d_fn1)
    _grad_set(g, params, prefix + String("feed_forward.w1.weight"), bg.d_w1)
    _grad_set(g, params, prefix + String("feed_forward.w3.weight"), bg.d_w3)
    _grad_set(g, params, prefix + String("feed_forward.w2.weight"), bg.d_w2)
    _grad_set(g, params, prefix + String("ffn_norm2.weight"), bg.d_fn2)
    var aw = _adaln_grad_w(
        bg.d_scale_msa, bg.d_gate_msa, bg.d_scale_mlp, bg.d_gate_mlp,
        adaln_h, D,
    )
    _grad_set(g, params, prefix + String("adaLN_modulation.0.weight"), aw[0])
    _grad_set(g, params, prefix + String("adaLN_modulation.0.bias"), aw[1])


def zimage_cn_apply_step(
    mut params: ZImageCnParams,
    grads: ZImageCnGrads,
    adaln_h: List[Float32],
    t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps_opt: Float32, wd: Float32,
    max_grad_norm: Float32,
    D: Int,
) raises -> Float64:
    """Scatter the control grads into the named store, GLOBAL-norm clip over
    the whole control group, then bias-corrected decoupled-wd AdamW on the F32
    masters. Returns the pre-clip global grad norm."""
    var g = List[List[Float32]]()
    for _ in range(len(params.p)):
        g.append(List[Float32]())
    for i in range(params.n_ctl):
        var cp = String("control_layers.") + String(i) + String(".")
        ref bg = grads.blocks[i]
        _scatter_body_grads(g, params, cp, bg.body, adaln_h, D)
        if i == 0:
            _grad_set(g, params, cp + String("before_proj.weight"), bg.d_before_w)
            _grad_set(g, params, cp + String("before_proj.bias"), bg.d_before_b)
        _grad_set(g, params, cp + String("after_proj.weight"), bg.d_after_w)
        _grad_set(g, params, cp + String("after_proj.bias"), bg.d_after_b)
    for j in range(params.n_nr):
        var cp = String("control_noise_refiner.") + String(j) + String(".")
        _scatter_body_grads(g, params, cp, grads.nr[j], adaln_h, D)
    _grad_set(g, params, String("control_all_x_embedder.2-1.weight"), grads.d_x_emb_w)
    _grad_set(g, params, String("control_all_x_embedder.2-1.bias"), grads.d_x_emb_b)

    for i in range(len(g)):
        if len(g[i]) != len(params.p[i]):
            raise Error(
                String("cn apply_step: param without grad: ") + params.names[i]
            )

    # global norm + clip
    var ss = 0.0
    for i in range(len(g)):
        var gp = g[i].unsafe_ptr()
        for k in range(len(g[i])):
            ss += Float64(gp[k]) * Float64(gp[k])
    var gn = sqrt(ss)
    var cscale = Float32(1.0)
    if gn > Float64(max_grad_norm) and gn > 0.0:
        cscale = Float32(Float64(max_grad_norm) / gn)

    # AdamW (decoupled wd; bias-corrected) — pointer inner loops
    var b1p = 1.0
    var b2p = 1.0
    for _ in range(t):
        b1p *= Float64(beta1)
        b2p *= Float64(beta2)
    var bc1 = Float32(1.0 - b1p)
    var bc2 = Float32(1.0 - b2p)
    for i in range(len(g)):
        var n = len(g[i])
        var gp = g[i].unsafe_ptr()
        var pp = params.p[i].unsafe_ptr()
        var mp = params.m[i].unsafe_ptr()
        var vp = params.v[i].unsafe_ptr()
        for k in range(n):
            var gk = gp[k] * cscale
            mp[k] = beta1 * mp[k] + (Float32(1.0) - beta1) * gk
            vp[k] = beta2 * vp[k] + (Float32(1.0) - beta2) * gk * gk
            var mh = mp[k] / bc1
            var vh = vp[k] / bc2
            pp[k] = pp[k] - lr * (mh / (sqrt(vh) + eps_opt) + wd * pp[k])
    return gn


# ── diffusers-format save ─────────────────────────────────────────────────────
def _write_text_file(path: String, content: String) raises:
    var fd = sys_open(path, O_CREAT | O_WRONLY | O_TRUNC, Int32(0o644))
    if fd < 0:
        raise Error(String("controlnet_zimage: cannot create ") + path)
    var n = content.byte_length()
    var buf = alloc[UInt8](n if n > 0 else 1)
    var src = content.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    var wrote = sys_pwrite(fd, BytePtr(unsafe_from_address=Int(buf)), n, 0)
    buf.free()
    _ = sys_close(fd)
    if wrote != n:
        raise Error(String("controlnet_zimage: short write to ") + path)


def zimage_cn_save(
    params: ZImageCnParams, out_dir: String,
    D: Int, n_heads: Int, control_in_dim: Int,
    ctx: DeviceContext,
) raises -> String:
    """Save the trained controlnet in the diffusers ZImageControlNetModel
    folder format: config.json (register_to_config fields) +
    diffusion_pytorch_model.safetensors with the EXACT state-dict keys (gate d
    diffs keys+shapes vs the reference model)."""
    _ = sys_system(String("mkdir -p ") + out_dir)
    var cfg = String("{\n")
    cfg += String('  "_class_name": "ZImageControlNetModel",\n')
    cfg += String('  "_diffusers_version": "0.38.0.dev0",\n')
    cfg += String('  "control_layers_places": [')
    for i in range(len(params.places)):
        if i > 0:
            cfg += String(", ")
        cfg += String(params.places[i])
    cfg += String("],\n")
    cfg += String('  "control_refiner_layers_places": null,\n')
    cfg += String('  "control_in_dim": ') + String(control_in_dim) + String(",\n")
    cfg += String('  "add_control_noise_refiner": null,\n')
    cfg += String('  "all_patch_size": [2],\n')
    cfg += String('  "all_f_patch_size": [1],\n')
    cfg += String('  "dim": ') + String(D) + String(",\n")
    cfg += String('  "n_refiner_layers": ') + String(params.n_nr) + String(",\n")
    cfg += String('  "n_heads": ') + String(n_heads) + String(",\n")
    cfg += String('  "n_kv_heads": ') + String(n_heads) + String(",\n")
    cfg += String('  "norm_eps": 1e-05,\n')
    cfg += String('  "qk_norm": true\n')
    cfg += String("}\n")
    _write_text_file(out_dir + String("/config.json"), cfg)

    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()
    for i in range(len(params.names)):
        names.append(params.names[i].copy())
        tensors.append(ArcPointer[Tensor](Tensor.from_host(
            params.p[i].copy(), params.shapes[i].copy(), STDtype.F32, ctx,
        )))
    var st_path = out_dir + String("/diffusion_pytorch_model.safetensors")
    save_safetensors(names, tensors, st_path, ctx)
    return st_path^
