# models/dit/wan_vace_dit.mojo — Wan-VACE DiT (VACE control variant), pure Mojo + MAX.
#
# Inference-only, GPU-only. VACE = ControlNet-style conditioning on top of the
# Wan base DiT. This file builds ONLY the VACE control deltas; the entire
# transformer (block + full forward + patchify3d + RoPE + SDPA) is REUSED from
# serenitymojo/models/dit/wan22_dit.mojo (wan22_block_forward, etc.).
#
# References (read line-by-line, READ-ONLY):
#   /home/alex/VACE/vace/models/wan/modules/model.py  (canonical VaceWanModel oracle)
#   /home/alex/EriDiffusion/inference-flame/src/models/wan_vace_dit.rs (Rust port)
#
# Variant: Wan2.1-VACE-14B (the only VACE checkpoint structure on record here):
#   dim=5120, num_layers=40, num_heads=40, head_dim=128, ffn_dim=13824,
#   in_dim=out_dim=16, patch_size=(1,2,2), freq_dim=256, text_dim=4096,
#   text_len=512, eps=1e-6, rope_theta=10000, qk_norm=True, cross_attn_norm=True.
#   8 VACE blocks at base layers [0,2,4,6,8,10,12,14]; vace_in_dim=96.
#
# ── What VACE ADDS over the base Wan DiT (model.py:10-142) ──────────────────
# 1. vace_patch_embedding: Conv3d(96, dim, k=(1,2,2), s=(1,2,2))  — separate
#    patch-embed for the 96-channel control context (z + mask). == patchify3d
#    (pf=1,ph=2,pw=2) + linear(pe_w.reshape[dim, 96*4=384], pe_b). [model.py:128-129]
# 2. 8 VaceWanAttentionBlocks (vace_blocks.{0-7}) — SAME WanAttentionBlock arch
#    as base blocks (reuse wan22_block_forward), PLUS:
#      - block 0 only: before_proj  (Linear dim->dim, zero-init): c = before_proj(c)+x
#      - every block:  after_proj   (Linear dim->dim, zero-init): hint = after_proj(c)
#    The running hidden c is carried block→block; after_proj(c_i) is hint i.
#    [model.py:33-44 — the all_c stack trick; hints = unbind(c)[:-1].]
# 3. Hint injection: base block at layer vace_layers[idx] does
#      x = base_block(x) + hints[idx] * vace_context_scale     [model.py:63-67]
#
# Everything else (time/text embed, RoPE, the 40 base blocks, head, unpatchify)
# is byte-identical to base Wan — reused from Wan22DiT.forward shape. The control
# math here is the ONLY delta.
#
# DTYPE: bf16 weights+activations, F32 accumulate (matches the bf16-GPU oracle).
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.patchify3d import patchify3d, unpatchify3d
from serenitymojo.ops.tensor_algebra import add, mul_scalar, reshape, concat as _ta_concat
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.dit.wan22_dit import (
    Wan22Config, wan22_build_rope, wan22_block_forward,
)


# ── VACE config ──────────────────────────────────────────────────────────────
# Wan2.1-VACE-14B. vace_layers = [0,2,4,...,14] (range(0,num_layers,2)[:8]).
@fieldwise_init
struct WanVaceConfig(Copyable, Movable, ImplicitlyCopyable):
    var base: Wan22Config
    var vace_in_dim: Int        # 96 (z + mask control channels)
    var num_vace_blocks: Int    # 8

    @staticmethod
    def vace_14b() -> WanVaceConfig:
        var base = Wan22Config(
            num_layers=40, dim=5120, ffn_dim=13824, num_heads=40, head_dim=128,
            in_dim=16, out_dim=16, freq_dim=256, text_dim=4096, text_len=512,
            eps=1.0e-6, rope_theta=10000.0,
        )
        return WanVaceConfig(base=base, vace_in_dim=96, num_vace_blocks=8)


# VACE block i ↔ base block at index 2*i (vace_layers = [0,2,4,6,8,10,12,14]).
# Returns base-layer index for vace block i, or -1 (not a VACE layer).
def wan_vace_layer_for_base(base_layer: Int) -> Int:
    # If base_layer is even and base_layer//2 < num_vace_blocks(8) -> vace index.
    if base_layer % 2 == 0 and (base_layer // 2) < 8:
        return base_layer // 2
    return -1


# ── vace_patch_embedding: Conv3d(vace_in_dim, dim, k=(1,2,2), s=(1,2,2)) ──────
# == patchify3d(control, 1,2,2) -> [S, vace_in_dim*4] then linear with
# vace_patch_embedding.weight reshaped [dim, vace_in_dim*4]. (model.py:116-129).
# control: [vace_in_dim, F, H, W] bf16. Returns [1, S_grid, dim] (UNPADDED).
def wan_vace_patch_embed(
    control: Tensor,
    pe_w: Tensor,        # [dim, vace_in_dim, 1, 2, 2]
    pe_b: Tensor,        # [dim]
    dim: Int,
    vace_in_dim: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var patched = patchify3d(control, 1, 2, 2, ctx)   # [S, vace_in_dim*4]
    var patch_dim = vace_in_dim * 1 * 2 * 2           # 96*4 = 384
    var pe_w_flat = reshape(pe_w, [dim, patch_dim], ctx)
    var emb2d = linear(patched, pe_w_flat, Optional(pe_b.clone(ctx)), ctx)  # [S,dim]
    var s = patched.shape()[0]
    return reshape(emb2d, [1, s, dim], ctx)           # [1,S,dim]


# Zero-pad a [1, n, dim] embedding to [1, S, dim] (token axis) — matches the
# `cat([u, zeros(1, seq_len-n, dim)])` in forward_vace (model.py:130-133).
def _pad_tokens(
    emb: Tensor, n: Int, S: Int, dim: Int, dt: STDtype, ctx: DeviceContext
) raises -> Tensor:
    if n >= S:
        return emb.clone(ctx)
    var pad_rows = S - n
    var zh = List[Float32]()
    for _ in range(pad_rows * dim):
        zh.append(0.0)
    var zeros2d = Tensor.from_host(zh^, [pad_rows, dim], dt, ctx)
    var emb2d = reshape(emb, [n, dim], ctx)
    var cat = _ta_concat(0, ctx, emb2d, zeros2d)       # [S, dim]
    return reshape(cat, [1, S, dim], ctx)


# ── ONE VACE control block (the VACE delta on top of wan22_block_forward) ─────
# c_in: running hidden [1,S,dim] (for block 0 this is the padded vace embedding;
#       for i>0 it is the previous block's output hidden).
# img:  base-path patch embedding [1,S,dim] (only used by block 0's before_proj).
# Returns (c_out, hint) where:
#   block 0: c0 = before_proj(c_in) + img ; c_out = base_block(c0); hint=after_proj(c_out)
#   i>0:     c_out = base_block(c_in)      ; hint = after_proj(c_out)
# Reuses wan22_block_forward for the (self_attn/cross_attn/ffn) body.
def wan_vace_block[
    S: Int, TXT: Int, H: Int, DH: Int
](
    is_block0: Bool,
    c_in: Tensor,
    img: Tensor,
    e0: Tensor,
    context: Tensor,
    cos: Tensor,
    sin: Tensor,
    w: Dict[String, ArcPointer[Tensor]],   # vace block weights (stripped prefix)
    cfg: Wan22Config,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    var bf = c_in.dtype()

    # block 0: c = before_proj(c_in) + img  (before_proj is zero-init in fresh
    # weights → c == img; with trained weights it mixes the control embedding).
    var c_pre: Tensor
    if is_block0:
        var bp = linear(
            c_in, w["before_proj.weight"][],
            Optional(w["before_proj.bias"][].clone(ctx)), ctx,
        )
        c_pre = add(
            cast_tensor(bp, STDtype.F32, ctx),
            cast_tensor(img, STDtype.F32, ctx), ctx,
        )
        c_pre = cast_tensor(c_pre, bf, ctx)
    else:
        c_pre = c_in.clone(ctx)

    # Run the SAME WanAttentionBlock body (reuse base forward).
    var c_out = wan22_block_forward[S, TXT, H, DH](
        c_pre, e0, context, cos, sin, w, cfg, ctx
    )

    # after_proj(c_out) -> hint (zero-init in fresh weights → hint == 0).
    var hint = linear(
        c_out, w["after_proj.weight"][],
        Optional(w["after_proj.bias"][].clone(ctx)), ctx,
    )
    return (c_out^, hint^)


# ── Full VACE model ──────────────────────────────────────────────────────────
# Loads ALL weights resident (base blocks.{0-39} + vace_blocks.{0-7} + vace
# patch-embed + shared). forward mirrors VaceWanModel.forward (model.py:144-235):
#   1. patchify noise -> base img [1,S,dim]
#   2. vace_patch_embedding(control) -> vace_emb, pad to S
#   3. time/text embed (shared with base — reuse base path)
#   4. run 8 vace blocks -> hints[0..7]
#   5. run 40 base blocks, at vace_layers[idx] add hints[idx]*scale
#   6. head -> unpatchify
struct WanVaceDit(Movable):
    var weights: Dict[String, ArcPointer[Tensor]]
    var config: WanVaceConfig

    def __init__(out self, var weights: Dict[String, ArcPointer[Tensor]], config: WanVaceConfig):
        self.weights = weights^
        self.config = config

    @staticmethod
    def load(dir: String, cfg: WanVaceConfig, ctx: DeviceContext) raises -> WanVaceDit:
        var st = ShardedSafeTensors.open(dir)
        var w = Dict[String, ArcPointer[Tensor]]()
        var names = st.names()
        for nm in names:
            var key = String(nm)
            var tv = st.tensor_view(key)
            w[key] = ArcPointer(Tensor.from_view(tv, ctx))
        return WanVaceDit(weights=w^, config=cfg)

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        return self.weights[name][]

    # Sub-dict for vace block i (keys stripped of "vace_blocks.i." prefix). Block
    # 0 additionally carries before_proj.*; every block carries after_proj.*.
    def _vace_block_weights(self, i: Int, ctx: DeviceContext) raises -> Dict[String, ArcPointer[Tensor]]:
        var prefix = String("vace_blocks.") + String(i) + "."
        var bw = Dict[String, ArcPointer[Tensor]]()
        var suffixes = [
            "modulation",
            "self_attn.q.weight", "self_attn.q.bias",
            "self_attn.k.weight", "self_attn.k.bias",
            "self_attn.v.weight", "self_attn.v.bias",
            "self_attn.o.weight", "self_attn.o.bias",
            "self_attn.norm_q.weight", "self_attn.norm_k.weight",
            "cross_attn.q.weight", "cross_attn.q.bias",
            "cross_attn.k.weight", "cross_attn.k.bias",
            "cross_attn.v.weight", "cross_attn.v.bias",
            "cross_attn.o.weight", "cross_attn.o.bias",
            "cross_attn.norm_q.weight", "cross_attn.norm_k.weight",
            "norm3.weight", "norm3.bias",
            "ffn.0.weight", "ffn.0.bias",
            "ffn.2.weight", "ffn.2.bias",
            "after_proj.weight", "after_proj.bias",
        ]
        for sfx in suffixes:
            var s = String(sfx)
            bw[s] = ArcPointer(self.weights[prefix + s][].clone(ctx))
        if i == 0:
            bw[String("before_proj.weight")] = ArcPointer(self.weights[prefix + "before_proj.weight"][].clone(ctx))
            bw[String("before_proj.bias")] = ArcPointer(self.weights[prefix + "before_proj.bias"][].clone(ctx))
        return bw^

    # Compute the 8 VACE hints (control branch). control: [vace_in_dim,F,H,W].
    # img: base patch embedding [1,S,dim] (for block-0 before_proj injection).
    def forward_vace[
        FG: Int, HG: Int, WG: Int, S: Int, TXT: Int, H: Int, DH: Int
    ](
        self, control: Tensor, img: Tensor, e0: Tensor, txt: Tensor,
        cos: Tensor, sin: Tensor, ctx: DeviceContext,
    ) raises -> List[ArcPointer[Tensor]]:
        var cfg = self.config.base
        var dim = cfg.dim
        var bf = img.dtype()

        # vace patch embed -> pad to S.
        var vemb = wan_vace_patch_embed(
            control, self._w("vace_patch_embedding.weight"),
            self._w("vace_patch_embedding.bias"), dim, self.config.vace_in_dim, ctx,
        )
        var n_patches = FG * HG * WG
        var c = _pad_tokens(vemb, n_patches, S, dim, bf, ctx)

        var hints = List[ArcPointer[Tensor]]()
        for vi in range(self.config.num_vace_blocks):
            var bw = self._vace_block_weights(vi, ctx)
            var res = wan_vace_block[S, TXT, H, DH](
                vi == 0, c, img, e0, txt, cos, sin, bw, cfg, ctx,
            )
            c = res[0].clone(ctx)
            hints.append(ArcPointer(res[1].clone(ctx)))
        return hints^


# Add a scaled hint into the running image stream at a VACE layer:
#   img = img + hint * scale   (model.py:66). F32 accumulate, cast back.
def wan_vace_inject(
    img: Tensor, hint: Tensor, scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    var bf = img.dtype()
    var hint_f32 = cast_tensor(hint, STDtype.F32, ctx)
    var scaled = mul_scalar(hint_f32, scale, ctx)
    var res = add(cast_tensor(img, STDtype.F32, ctx), scaled, ctx)
    return cast_tensor(res, bf, ctx)
