# models/dit/nava_dit.mojo — NAVA full 30-layer DiT forward (chunk 7).
#
# Wires:
#   PRE  : nava_embed.{video_patch_embed, audio_patch_embed, text_embed}
#          + inline time embedding (to produce both e_head and e0)
#   10x  : nava_block.nava_double_block
#   20x  : nava_block.nava_single_block
#   POST : Head (2-way modulation-add + layer_norm_no_affine + linear) +
#          unpatchify video (fhwpqrc→cfphqwr permute, reshape) +
#          predict_eps flatten (cfhw→flat)
#
# Reference: WanAVModel.forward (model_mm.py:1574) + Head.forward (model_mm.py:1118)
#            + unpatchify (model_mm.py:1693) + predict_eps (model_nava.py:462-486)
#
# Inputs  : in_lat_vid [1280,48] BF16, in_lat_aud [34,128] BF16,
#           in_text [42,4096] BF16, in_t [1] F32
# Outputs : NavaDitOut.vel_vid [1280,48] BF16, NavaDitOut.vel_aud [34,128] BF16
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm_no_affine
from serenitymojo.ops.activations import silu
from serenitymojo.ops.embeddings import timestep_embedding
from serenitymojo.ops.tensor_algebra import (
    add, mul, add_scalar, reshape, permute, concat, slice, zeros_device,
)
from serenitymojo.models.dit.nava_embed import (
    load_nava_embed_weights,
    nava_video_patch_embed,
    nava_video_patch_embed_hires,
    nava_audio_patch_embed,
    nava_text_embed,
)
from serenitymojo.ops.patchify3d import unpatchify3d
from serenitymojo.models.dit.nava_block import (
    load_nava_double_block,
    load_nava_single_block,
    nava_double_block,
    nava_single_block,
    build_nava_rope_tables,
    nava_double_block_hires,
    nava_single_block_hires,
    build_nava_rope_tables_hires,
    NavaRope,
)


# ── Comptime layout constants (matches nava_block.mojo) ───────────────────────
comptime _VID    = 320
comptime _AUD    = 34
comptime _DIM    = 3072
comptime _VID_HR = 1950   # hi-res 832×480: F=5, Hp=15, Wp=26


# ── Output struct (Tensor is Movable-not-Copyable; can't return two separately) ─
struct NavaDitOut(Movable):
    """Output of nava_dit_forward."""
    var vel_vid: Tensor  # [1280, 48] BF16
    var vel_aud: Tensor  # [34, 128]  BF16

    def __init__(out self, var v: Tensor, var a: Tensor):
        self.vel_vid = v^
        self.vel_aud = a^


# ── Head forward ──────────────────────────────────────────────────────────────
# model_mm.py:1118 (Head.forward):
#   modulation [1,2,3072]; e [1,L,3072]
#   he[0] = slice(modulation,1,0,1) [1,1,3072] broadcast+ e [1,L,3072] → [1,L,3072]
#   he[1] = slice(modulation,1,1,1) [1,1,3072] broadcast+ e [1,L,3072] → [1,L,3072]
#   out = head_linear( layer_norm_no_affine(x)*(1+he[1]) + he[0] )
def _nava_head_forward(
    x: Tensor,          # [1, L, 3072] BF16
    e: Tensor,          # [1, L, 3072] BF16 (broadcast of e_head)
    mod: Tensor,        # [1, 2, 3072] BF16
    head_w: Tensor,     # [out_dim, 3072] BF16
    head_b: Tensor,     # [out_dim] BF16
    ctx: DeviceContext,
) raises -> Tensor:
    """Head modulation + norm + linear: [1,L,3072] → [1,L,out_dim]."""
    var EPS = Float32(1e-6)

    # he[i] = mod[:,i,:] [1,1,3072] + e [1,L,3072]  (broadcast)
    var mod0 = slice(mod, 1, 0, 1, ctx)  # [1,1,3072]
    var mod1 = slice(mod, 1, 1, 1, ctx)  # [1,1,3072]
    var he0 = add(mod0, e, ctx)          # [1,L,3072]
    var he1 = add(mod1, e, ctx)          # [1,L,3072]

    # out = head_linear( norm_no_affine(x)*(1+he1) + he0 )
    var xn    = layer_norm_no_affine(x, EPS, ctx)
    var xmod  = add(mul(xn, add_scalar(he1, 1.0, ctx), ctx), he0, ctx)
    return linear(xmod, head_w, Optional(head_b.clone(ctx)), ctx)


# ── Unpatchify video + predict_eps flatten ────────────────────────────────────
# model_mm.py:1714-1716 + predict_eps transform (model_nava.py:469-476):
#   head_out [1,320,192] → [320,192]
#   unpatchify3d([320,192], C=48, F=5, H=16, W=16, pf=1, ph=2, pw=2) → [48,5,16,16]
#     (unpatchify3d kernel reads within-patch as (pf,ph,pw,C) = exactly einsum
#      'fhwpqrc->cfphqwr')
#   predict_eps: permute(c,f,h,w)→(f,h,w,c) = permute[1,2,3,0] → [5,16,16,48]
#               → reshape [1280,48]
def _unpatchify_video(
    head_out: Tensor,  # [1,320,192] BF16
    ctx: DeviceContext,
) raises -> Tensor:
    """Unpatchify video head output → vel_vid [1280,48] BF16."""
    # [1,320,192] → [320,192]
    var flat = reshape(head_out, [320, 192], ctx)

    # unpatchify3d: [320,192] → [48,5,16,16]
    # einsum 'fhwpqrc->cfphqwr' is implemented by the kernel's within-patch
    # order (pf,ph,pw,C) matching PyTorch's view(5,8,8,1,2,2,48) → einsum.
    var vid4d = unpatchify3d(flat, 48, 5, 16, 16, 1, 2, 2, ctx)  # [48,5,16,16]

    # predict_eps: permute (c,f,h,w)→(f,h,w,c) = permute [1,2,3,0]
    var perm2 = List[Int]()
    perm2.append(1)
    perm2.append(2)
    perm2.append(3)
    perm2.append(0)
    var fhwc = permute(vid4d, perm2, ctx)  # [5,16,16,48]

    # reshape [5,16,16,48] → [1280,48]
    return reshape(fhwc, [1280, 48], ctx)


# ── Full forward ──────────────────────────────────────────────────────────────
def nava_dit_forward(
    in_lat_vid: Tensor,  # [1280,48] BF16
    in_lat_aud: Tensor,  # [34,128]  BF16
    in_text: Tensor,     # [42,4096] BF16
    in_t: Tensor,        # [1] F32
    st: ShardedSafeTensors,
    ctx: DeviceContext,
    masking_modality: Bool = False,
) raises -> NavaDitOut:
    """Full NAVA WanAVModel forward, b=1.

    masking_modality=False (default): joint self-attention.
    masking_modality=True: non-joint self-attention (video/audio attend separately).
    Returns NavaDitOut(vel_vid [1280,48] BF16, vel_aud [34,128] BF16).
    """
    # ── Load embed weights (BF16) ─────────────────────────────────────────────
    var we = load_nava_embed_weights(st, "backbone.", ctx)

    # ── PRE: patch-embeds, text ───────────────────────────────────────────────
    # (A) video: [1280,48] → [1,320,3072]
    var x_vid   = nava_video_patch_embed(in_lat_vid, we, ctx)
    # (B) audio: [34,128] → [1,34,3072]
    var x_audio = nava_audio_patch_embed(in_lat_aud, we, ctx)
    # (C) text: [42,4096] → [1,512,3072]
    var context = nava_text_embed(in_text, we, ctx)

    # ── PRE: time embed (inline to get BOTH e_head and e0) ────────────────────
    # time_embedding: sinusoidal(t)→[1,256] → Linear→SiLU→Linear → e_head [1,3072]
    # time_projection: SiLU(e_head) → Linear → reshape → e0 [1,1,6,3072]
    var _se     = timestep_embedding(in_t, 256, ctx, Float32(10000.0), STDtype.BF16)  # [1,256]
    var _h0     = linear(_se, we["tme0.weight"][], Optional(we["tme0.bias"][].clone(ctx)), ctx)
    var _h0s    = silu(_h0, ctx)
    var e_head  = linear(_h0s, we["tme2.weight"][], Optional(we["tme2.bias"][].clone(ctx)), ctx)  # [1,3072]
    var _eps    = silu(e_head, ctx)
    var _eproj  = linear(_eps, we["tp1.weight"][], Optional(we["tp1.bias"][].clone(ctx)), ctx)     # [1,18432]
    var e0      = reshape(_eproj, [1, 1, 6, _DIM], ctx)  # [1,1,6,3072]

    # ── Broadcast e0 to per-token modulation: [1,1,6,3072] → [1,L,6,3072] ────
    # add() broadcasts [1,1,6,3072] against zeros[1,L,6,3072] → [1,L,6,3072]
    var z_vid   = zeros_device([1, _VID, 6, _DIM], STDtype.BF16, ctx)
    var e_vid   = add(e0, z_vid, ctx)  # [1,320,6,3072]
    var z_aud   = zeros_device([1, _AUD, 6, _DIM], STDtype.BF16, ctx)
    var e_audio = add(e0, z_aud, ctx)  # [1,34,6,3072]

    # ── Concat x for blocks: [1,354,3072] ────────────────────────────────────
    var x = concat(1, ctx, x_vid, x_audio)  # [1,354,3072]

    # ── Build RoPE tables once (constant across all 30 blocks) ───────────────
    var rope = build_nava_rope_tables(ctx)

    # ── 10 Double blocks ──────────────────────────────────────────────────────
    for i in range(10):
        var blk_prefix = String("backbone.double_blocks.") + String(i) + "."
        var wb = load_nava_double_block(st, blk_prefix, ctx)
        x = nava_double_block(x, e_vid, e_audio, context, wb, rope, ctx, masking_modality)

    # ── 20 Single blocks ──────────────────────────────────────────────────────
    for i in range(20):
        var blk_prefix = String("backbone.single_blocks.") + String(i) + "."
        var wb = load_nava_single_block(st, blk_prefix, ctx)
        x = nava_single_block(x, e_vid, e_audio, context, wb, rope, ctx, masking_modality)

    # ── POST: split x back into vid and audio streams ─────────────────────────
    var xv_post = slice(x, 1, 0,    _VID, ctx)  # [1,320,3072]
    var xa_post = slice(x, 1, _VID, _AUD, ctx)  # [1,34,3072]

    # ── Build e_head_vid / e_head_audio for head modulation ───────────────────
    # e_head [1,3072] → broadcast to [1,L,3072] via zeros + add
    var z_hv = zeros_device([1, _VID, _DIM], STDtype.BF16, ctx)
    var e_hv = add(e_head, z_hv, ctx)   # [1,320,3072]
    var z_ha = zeros_device([1, _AUD, _DIM], STDtype.BF16, ctx)
    var e_ha = add(e_head, z_ha, ctx)   # [1,34,3072]

    # ── Load head weights ─────────────────────────────────────────────────────
    var hw_vid = Tensor.from_view(st.tensor_view("backbone.head.head.weight"),        ctx)  # [192,3072]
    var hb_vid = Tensor.from_view(st.tensor_view("backbone.head.head.bias"),          ctx)  # [192]
    var hm_vid = Tensor.from_view(st.tensor_view("backbone.head.modulation"),         ctx)  # [1,2,3072]

    var hw_aud = Tensor.from_view(st.tensor_view("backbone.head_audio.head.weight"),  ctx)  # [128,3072]
    var hb_aud = Tensor.from_view(st.tensor_view("backbone.head_audio.head.bias"),    ctx)  # [128]
    var hm_aud = Tensor.from_view(st.tensor_view("backbone.head_audio.modulation"),   ctx)  # [1,2,3072]

    # ── Video head: [1,320,3072] → [1,320,192] ────────────────────────────────
    var vid_head_out = _nava_head_forward(xv_post, e_hv, hm_vid, hw_vid, hb_vid, ctx)

    # ── Audio head: [1,34,3072] → [1,34,128] (no unpatchify for audio) ────────
    var aud_head_out = _nava_head_forward(xa_post, e_ha, hm_aud, hw_aud, hb_aud, ctx)

    # ── Unpatchify video + predict_eps → vel_vid [1280,48] ────────────────────
    var vel_vid = _unpatchify_video(vid_head_out, ctx)

    # ── vel_aud: [1,34,128] → [34,128] ───────────────────────────────────────
    var vel_aud = reshape(aud_head_out, [34, 128], ctx)

    return NavaDitOut(vel_vid^, vel_aud^)


# ══════════════════════════════════════════════════════════════════════════════
# NavaDiT — resident struct: loads ALL weights ONCE, reuses across denoise steps
# ══════════════════════════════════════════════════════════════════════════════
#
# Fields:
#   embed_weights  : embed + time-embed weights (Dict[String, ArcPointer[Tensor]])
#   head_weights   : video + audio head weights (Dict[String, ArcPointer[Tensor]])
#   double_blocks  : List of 10 weight dicts for double blocks
#   single_blocks  : List of 20 weight dicts for single blocks
#   rope           : NavaRope with pre-built vid + aud cos/sin tables
#
# Usage:
#   var dit = NavaDiT.load(st, ctx)          # once at startup
#   var out = dit.forward(lat_v, lat_a, txt, t, ctx)  # each denoise step
#
struct NavaDiT(Movable):
    """Resident NAVA DiT: all weights + RoPE tables live on GPU between steps."""

    var embed_weights: Dict[String, ArcPointer[Tensor]]
    var head_weights:  Dict[String, ArcPointer[Tensor]]
    var double_blocks: List[Dict[String, ArcPointer[Tensor]]]
    var single_blocks: List[Dict[String, ArcPointer[Tensor]]]
    var rope: NavaRope

    def __init__(out self,
                 var ew: Dict[String, ArcPointer[Tensor]],
                 var hw: Dict[String, ArcPointer[Tensor]],
                 var db: List[Dict[String, ArcPointer[Tensor]]],
                 var sb: List[Dict[String, ArcPointer[Tensor]]],
                 var r:  NavaRope):
        self.embed_weights = ew^
        self.head_weights  = hw^
        self.double_blocks = db^
        self.single_blocks = sb^
        self.rope          = r^

    @staticmethod
    def load(st: ShardedSafeTensors, ctx: DeviceContext) raises -> NavaDiT:
        """Load all NAVA weights once and build the RoPE tables.

        Dequantises all FP8 block weights to BF16 at load (GPU-resident).
        ~13GB resident; fits a 24GB GPU with activations.
        """
        # Embed weights
        var ew = load_nava_embed_weights(st, "backbone.", ctx)

        # Head weights (BF16, loaded directly from safetensors)
        var hw = Dict[String, ArcPointer[Tensor]]()
        var _hw_vid  = Tensor.from_view(st.tensor_view("backbone.head.head.weight"),       ctx)
        var _hb_vid  = Tensor.from_view(st.tensor_view("backbone.head.head.bias"),         ctx)
        var _hm_vid  = Tensor.from_view(st.tensor_view("backbone.head.modulation"),        ctx)
        var _hw_aud  = Tensor.from_view(st.tensor_view("backbone.head_audio.head.weight"), ctx)
        var _hb_aud  = Tensor.from_view(st.tensor_view("backbone.head_audio.head.bias"),   ctx)
        var _hm_aud  = Tensor.from_view(st.tensor_view("backbone.head_audio.modulation"),  ctx)
        hw["head.weight"]       = ArcPointer(_hw_vid^)
        hw["head.bias"]         = ArcPointer(_hb_vid^)
        hw["head.modulation"]   = ArcPointer(_hm_vid^)
        hw["head_a.weight"]     = ArcPointer(_hw_aud^)
        hw["head_a.bias"]       = ArcPointer(_hb_aud^)
        hw["head_a.modulation"] = ArcPointer(_hm_aud^)

        # Double block weights (10 blocks)
        var double_blocks = List[Dict[String, ArcPointer[Tensor]]]()
        for i in range(10):
            var blk_prefix = String("backbone.double_blocks.") + String(i) + "."
            double_blocks.append(load_nava_double_block(st, blk_prefix, ctx))

        # Single block weights (20 blocks)
        var single_blocks = List[Dict[String, ArcPointer[Tensor]]]()
        for i in range(20):
            var blk_prefix = String("backbone.single_blocks.") + String(i) + "."
            single_blocks.append(load_nava_single_block(st, blk_prefix, ctx))

        # RoPE tables (constant across all blocks + all steps)
        var rope = build_nava_rope_tables(ctx)

        return NavaDiT(ew^, hw^, double_blocks^, single_blocks^, rope^)

    def forward(
        self,
        in_lat_vid: Tensor,  # [1280,48] BF16
        in_lat_aud: Tensor,  # [34,128]  BF16
        in_text: Tensor,     # [42,4096] BF16
        in_t: Tensor,        # [1] F32
        ctx: DeviceContext,
        masking_modality: Bool = False,
    ) raises -> NavaDitOut:
        """Full NAVA WanAVModel forward using resident weights (no re-load).

        masking_modality=False (default): joint self-attention.
        masking_modality=True: non-joint self-attention (video/audio attend separately).
        Uses self.embed_weights, self.double_blocks[i], self.single_blocks[i], self.rope.
        Returns NavaDitOut(vel_vid [1280,48] BF16, vel_aud [34,128] BF16).
        """
        # ── PRE: patch-embeds, text ───────────────────────────────────────────
        # Access self.embed_weights directly (dict is Copyable via ArcPointer values;
        # each embed function receives a copy with refcount-bumped ArcPointers).
        var x_vid   = nava_video_patch_embed(in_lat_vid, self.embed_weights, ctx)
        var x_audio = nava_audio_patch_embed(in_lat_aud, self.embed_weights, ctx)
        var context = nava_text_embed(in_text, self.embed_weights, ctx)

        # ── PRE: time embed ───────────────────────────────────────────────────
        var _se    = timestep_embedding(in_t, 256, ctx, Float32(10000.0), STDtype.BF16)
        var _h0    = linear(_se,
                            self.embed_weights["tme0.weight"][],
                            Optional(self.embed_weights["tme0.bias"][].clone(ctx)), ctx)
        var _h0s   = silu(_h0, ctx)
        var e_head = linear(_h0s,
                            self.embed_weights["tme2.weight"][],
                            Optional(self.embed_weights["tme2.bias"][].clone(ctx)), ctx)
        var _eps   = silu(e_head, ctx)
        var _eproj = linear(_eps,
                            self.embed_weights["tp1.weight"][],
                            Optional(self.embed_weights["tp1.bias"][].clone(ctx)), ctx)
        var e0     = reshape(_eproj, [1, 1, 6, _DIM], ctx)

        # ── Broadcast e0 ─────────────────────────────────────────────────────
        var z_vid   = zeros_device([1, _VID, 6, _DIM], STDtype.BF16, ctx)
        var e_vid   = add(e0, z_vid, ctx)
        var z_aud   = zeros_device([1, _AUD, 6, _DIM], STDtype.BF16, ctx)
        var e_audio = add(e0, z_aud, ctx)

        # ── Concat x for blocks ───────────────────────────────────────────────
        var x = concat(1, ctx, x_vid, x_audio)

        # ── 10 Double blocks (resident weights, shared rope) ──────────────────
        for i in range(10):
            x = nava_double_block(x, e_vid, e_audio, context, self.double_blocks[i], self.rope, ctx, masking_modality)

        # ── 20 Single blocks (resident weights, shared rope) ──────────────────
        for i in range(20):
            x = nava_single_block(x, e_vid, e_audio, context, self.single_blocks[i], self.rope, ctx, masking_modality)

        # ── POST: split x ─────────────────────────────────────────────────────
        var xv_post = slice(x, 1, 0,    _VID, ctx)
        var xa_post = slice(x, 1, _VID, _AUD, ctx)

        # ── e_head broadcast ──────────────────────────────────────────────────
        var z_hv = zeros_device([1, _VID, _DIM], STDtype.BF16, ctx)
        var e_hv = add(e_head, z_hv, ctx)
        var z_ha = zeros_device([1, _AUD, _DIM], STDtype.BF16, ctx)
        var e_ha = add(e_head, z_ha, ctx)

        # ── Head forward ──────────────────────────────────────────────────────
        var vid_head_out = _nava_head_forward(xv_post, e_hv,
            self.head_weights["head.modulation"][],
            self.head_weights["head.weight"][],
            self.head_weights["head.bias"][], ctx)
        var aud_head_out = _nava_head_forward(xa_post, e_ha,
            self.head_weights["head_a.modulation"][],
            self.head_weights["head_a.weight"][],
            self.head_weights["head_a.bias"][], ctx)

        # ── Unpatchify + flatten ──────────────────────────────────────────────
        var vel_vid = _unpatchify_video(vid_head_out, ctx)
        var vel_aud = reshape(aud_head_out, [34, 128], ctx)

        return NavaDitOut(vel_vid^, vel_aud^)


# ══════════════════════════════════════════════════════════════════════════════
# Hi-res (832×480) unpatchify + NavaDiTHires resident struct
# Latent grid: F=5, Hlat=30, Wlat=52 → patch [1,2,2] → Hp=15, Wp=26
# VID=1950  head_out [1,1950,192] → vel_vid [7800,48]
# ══════════════════════════════════════════════════════════════════════════════

def _unpatchify_video_hires(
    head_out: Tensor,  # [1,1950,192] BF16
    ctx: DeviceContext,
) raises -> Tensor:
    """Unpatchify hi-res video head output → vel_vid [7800,48] BF16.

    [1,1950,192] → [1950,192]
    unpatchify3d([1950,192], C=48, F=5, H=30, W=52, pf=1, ph=2, pw=2) → [48,5,30,52]
    predict_eps: permute (c,f,h,w)→(f,h,w,c) = permute [1,2,3,0] → [5,30,52,48]
    reshape [5,30,52,48] → [7800,48]
    """
    # [1,1950,192] → [1950,192]
    var flat = reshape(head_out, [1950, 192], ctx)

    # unpatchify3d: [1950,192] → [48,5,30,52]
    var vid4d = unpatchify3d(flat, 48, 5, 30, 52, 1, 2, 2, ctx)  # [48,5,30,52]

    # predict_eps: permute (c,f,h,w)→(f,h,w,c) = permute [1,2,3,0]
    var perm2 = List[Int]()
    perm2.append(1)
    perm2.append(2)
    perm2.append(3)
    perm2.append(0)
    var fhwc = permute(vid4d, perm2, ctx)  # [5,30,52,48]

    # reshape [5,30,52,48] → [7800,48]
    return reshape(fhwc, [7800, 48], ctx)


struct NavaDiTHires(Movable):
    """Resident NAVA DiT for hi-res 832×480: all weights + RoPE tables live on GPU.

    VID=1950 tokens (F=5, Hp=15, Wp=26), SEQ=1984, AUD=34.
    Same weights as NavaDiT (model weights are resolution-independent);
    only the rope tables and activation shapes differ.
    """

    var embed_weights: Dict[String, ArcPointer[Tensor]]
    var head_weights:  Dict[String, ArcPointer[Tensor]]
    var double_blocks: List[Dict[String, ArcPointer[Tensor]]]
    var single_blocks: List[Dict[String, ArcPointer[Tensor]]]
    var rope: NavaRope

    def __init__(out self,
                 var ew: Dict[String, ArcPointer[Tensor]],
                 var hw: Dict[String, ArcPointer[Tensor]],
                 var db: List[Dict[String, ArcPointer[Tensor]]],
                 var sb: List[Dict[String, ArcPointer[Tensor]]],
                 var r:  NavaRope):
        self.embed_weights = ew^
        self.head_weights  = hw^
        self.double_blocks = db^
        self.single_blocks = sb^
        self.rope          = r^

    @staticmethod
    def load(st: ShardedSafeTensors, ctx: DeviceContext) raises -> NavaDiTHires:
        """Load all NAVA weights once and build the hi-res RoPE tables."""
        # Embed weights (same safetensors keys as NavaDiT)
        var ew = load_nava_embed_weights(st, "backbone.", ctx)

        # Head weights
        var hw = Dict[String, ArcPointer[Tensor]]()
        var _hw_vid  = Tensor.from_view(st.tensor_view("backbone.head.head.weight"),       ctx)
        var _hb_vid  = Tensor.from_view(st.tensor_view("backbone.head.head.bias"),         ctx)
        var _hm_vid  = Tensor.from_view(st.tensor_view("backbone.head.modulation"),        ctx)
        var _hw_aud  = Tensor.from_view(st.tensor_view("backbone.head_audio.head.weight"), ctx)
        var _hb_aud  = Tensor.from_view(st.tensor_view("backbone.head_audio.head.bias"),   ctx)
        var _hm_aud  = Tensor.from_view(st.tensor_view("backbone.head_audio.modulation"),  ctx)
        hw["head.weight"]       = ArcPointer(_hw_vid^)
        hw["head.bias"]         = ArcPointer(_hb_vid^)
        hw["head.modulation"]   = ArcPointer(_hm_vid^)
        hw["head_a.weight"]     = ArcPointer(_hw_aud^)
        hw["head_a.bias"]       = ArcPointer(_hb_aud^)
        hw["head_a.modulation"] = ArcPointer(_hm_aud^)

        # Double block weights (10 blocks)
        var double_blocks = List[Dict[String, ArcPointer[Tensor]]]()
        for i in range(10):
            var blk_prefix = String("backbone.double_blocks.") + String(i) + "."
            double_blocks.append(load_nava_double_block(st, blk_prefix, ctx))

        # Single block weights (20 blocks)
        var single_blocks = List[Dict[String, ArcPointer[Tensor]]]()
        for i in range(20):
            var blk_prefix = String("backbone.single_blocks.") + String(i) + "."
            single_blocks.append(load_nava_single_block(st, blk_prefix, ctx))

        # Hi-res RoPE tables (Hp=15, Wp=26)
        var rope = build_nava_rope_tables_hires(ctx)

        return NavaDiTHires(ew^, hw^, double_blocks^, single_blocks^, rope^)

    def forward(
        self,
        in_lat_vid: Tensor,  # [7800,48] BF16
        in_lat_aud: Tensor,  # [34,128]  BF16
        in_text: Tensor,     # [N,4096]  BF16  (pads to 512 inside nava_text_embed)
        in_t: Tensor,        # [1] F32
        ctx: DeviceContext,
        masking_modality: Bool = False,
    ) raises -> NavaDitOut:
        """Full NAVA WanAVModel forward at 832×480 using resident weights.

        Returns NavaDitOut(vel_vid [7800,48] BF16, vel_aud [34,128] BF16).
        """
        # ── PRE: patch-embeds, text ───────────────────────────────────────────
        var x_vid   = nava_video_patch_embed_hires(in_lat_vid, self.embed_weights, ctx)
        var x_audio = nava_audio_patch_embed(in_lat_aud, self.embed_weights, ctx)
        var context = nava_text_embed(in_text, self.embed_weights, ctx)

        # ── PRE: time embed ───────────────────────────────────────────────────
        var _se    = timestep_embedding(in_t, 256, ctx, Float32(10000.0), STDtype.BF16)
        var _h0    = linear(_se,
                            self.embed_weights["tme0.weight"][],
                            Optional(self.embed_weights["tme0.bias"][].clone(ctx)), ctx)
        var _h0s   = silu(_h0, ctx)
        var e_head = linear(_h0s,
                            self.embed_weights["tme2.weight"][],
                            Optional(self.embed_weights["tme2.bias"][].clone(ctx)), ctx)
        var _eps   = silu(e_head, ctx)
        var _eproj = linear(_eps,
                            self.embed_weights["tp1.weight"][],
                            Optional(self.embed_weights["tp1.bias"][].clone(ctx)), ctx)
        var e0     = reshape(_eproj, [1, 1, 6, _DIM], ctx)

        # ── Broadcast e0 to per-token modulation ─────────────────────────────
        var z_vid   = zeros_device([1, _VID_HR, 6, _DIM], STDtype.BF16, ctx)
        var e_vid   = add(e0, z_vid, ctx)   # [1,1950,6,3072]
        var z_aud   = zeros_device([1, _AUD, 6, _DIM], STDtype.BF16, ctx)
        var e_audio = add(e0, z_aud, ctx)   # [1,34,6,3072]

        # ── Concat x for blocks ───────────────────────────────────────────────
        var x = concat(1, ctx, x_vid, x_audio)  # [1,1984,3072]

        # ── 10 Double blocks ──────────────────────────────────────────────────
        for i in range(10):
            x = nava_double_block_hires(x, e_vid, e_audio, context,
                                        self.double_blocks[i], self.rope, ctx, masking_modality)

        # ── 20 Single blocks ──────────────────────────────────────────────────
        for i in range(20):
            x = nava_single_block_hires(x, e_vid, e_audio, context,
                                        self.single_blocks[i], self.rope, ctx, masking_modality)

        # ── POST: split x ─────────────────────────────────────────────────────
        var xv_post = slice(x, 1, 0,       _VID_HR, ctx)  # [1,1950,3072]
        var xa_post = slice(x, 1, _VID_HR, _AUD,    ctx)  # [1,34,3072]

        # ── e_head broadcast ──────────────────────────────────────────────────
        var z_hv = zeros_device([1, _VID_HR, _DIM], STDtype.BF16, ctx)
        var e_hv = add(e_head, z_hv, ctx)   # [1,1950,3072]
        var z_ha = zeros_device([1, _AUD, _DIM], STDtype.BF16, ctx)
        var e_ha = add(e_head, z_ha, ctx)   # [1,34,3072]

        # ── Head forward ──────────────────────────────────────────────────────
        var vid_head_out = _nava_head_forward(xv_post, e_hv,
            self.head_weights["head.modulation"][],
            self.head_weights["head.weight"][],
            self.head_weights["head.bias"][], ctx)   # [1,1950,192]
        var aud_head_out = _nava_head_forward(xa_post, e_ha,
            self.head_weights["head_a.modulation"][],
            self.head_weights["head_a.weight"][],
            self.head_weights["head_a.bias"][], ctx)  # [1,34,128]

        # ── Unpatchify + flatten ──────────────────────────────────────────────
        var vel_vid = _unpatchify_video_hires(vid_head_out, ctx)  # [7800,48]
        var vel_aud = reshape(aud_head_out, [34, 128], ctx)

        return NavaDitOut(vel_vid^, vel_aud^)
