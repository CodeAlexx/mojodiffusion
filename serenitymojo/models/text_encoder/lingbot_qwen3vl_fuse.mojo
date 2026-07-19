# lingbot_qwen3vl_fuse.mojo — pure-Mojo Qwen3-VL VISION→TEXT fusion for LingBot.
#
# Fuses the (already parity-passed) Qwen3-VL vision tower's tokens into the
# Qwen3-VL text decoder to produce the IMAGE-GROUNDED conditioning embeds the
# LingBot-Video DiT consumes. Ports the three fusion pieces of
# transformers/models/qwen3_vl/modeling_qwen3_vl.py:
#
#   1. masked_scatter (Qwen3VLModel.forward:1287-1297): write the vision
#      pooler_output rows into the token-embedding rows where input_ids==151655
#      (image_pad). The image span is contiguous, so row order == pooler order.
#
#   2. 3D interleaved M-RoPE (get_rope_index:1031 + get_vision_position_ids:975 +
#      apply_interleaved_mrope:370 + rotary forward:352). Build [3,seq] (T,H,W)
#      positions: text spans get arange shared on all 3 axes; the image span gets
#      t=const, h=row, w=col over the merged H/2xW/2 grid, then the cursor
#      advances by max(H,W)//2. The 64 rope freqs are axis-selected by the
#      interleave of mrope_section=[24,20,20]: freq k -> T if (k>=60 or k%3==0),
#      H if k%3==1, W if k%3==2 (for k<60). Angle = pos_axis(k) * theta^(-k/64).
#      This REPLACES the text-only plain 1-D rope (valid only with no image).
#      The half-split rotate_half kernel (rope_halfsplit) is reused verbatim —
#      only the per-(position,freq) angle table changes.
#
#   3. deepstack injection (Qwen3VLTextModel.forward:927 + _deepstack_process:941):
#      at decoder layers 0,1,2, ADD the 3 vision deepstack features to the hidden
#      state at the visual-token positions.
#
# Parity: gated cos>=0.999 vs HF Qwen3VLModel fused last_hidden_state
# (parity/qwen3vl_fuse_oracle.py + qwen3vl_fuse_parity.mojo).
#
# Mojo 1.0.0b1, NVIDIA GPU. BF16 storage, F32 accumulation.

from std.math import cos as fcos, sin as fsin, exp as fexp, log as flog

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import slice, concat, add
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder
from serenitymojo.models.text_encoder.lingbot_qwen3vl_vision import (
    Qwen3VLVisionModel, VisionOutput,
)


comptime IMAGE_TOKEN_ID = 151655
comptime FUSE_PAD_ID = 151643        # Qwen3 pad/bos; masked out as causal columns
comptime SPATIAL_MERGE = 2
comptime MROPE_T = 24                 # mrope_section (informational; interleave below)
comptime MROPE_H = 20
comptime MROPE_W = 20


# ── SDPA seq bucketing (mirror encode_layer_states' supported comptime seqs) ──
def _pad_bucket(L: Int) raises -> Int:
    if L <= 64: return 64
    if L <= 128: return 128
    if L <= 256: return 256
    if L <= 512: return 512
    if L <= 1024: return 1024
    if L <= 2048: return 2048
    raise Error(String("fuse: L=") + String(L) + " exceeds 2048")


# ── 3D M-RoPE position ids (get_rope_index for a single image) ───────────────
struct Positions(Movable):
    var pt: List[Int]
    var ph: List[Int]
    var pw: List[Int]
    var vis_start: Int
    var nvis: Int

    def __init__(
        out self, var pt: List[Int], var ph: List[Int], var pw: List[Int],
        vis_start: Int, nvis: Int,
    ):
        self.pt = pt^
        self.ph = ph^
        self.pw = pw^
        self.vis_start = vis_start
        self.nvis = nvis


def _build_positions(
    ids: List[Int], L: Int, seq_pad: Int, gh_full: Int, gw_full: Int,
) raises -> Positions:
    """Port get_rope_index / get_vision_position_ids for text-image-text (single
    image). Text tokens share arange on all 3 axes; the contiguous image span
    (image_token rows) gets t=cur (const), h=cur+row, w=cur+col over the merged
    (gh_full/2 x gw_full/2) grid in row-major (h-major) order — matching the
    vision pooler token order — then the cursor advances by max(gh,gw)//2."""
    var merged_h = gh_full // SPATIAL_MERGE
    var merged_w = gw_full // SPATIAL_MERGE
    var nvis = merged_h * merged_w
    var advance = (gh_full if gh_full >= gw_full else gw_full) // SPATIAL_MERGE

    var pt = List[Int](); pt.resize(seq_pad, 0)
    var ph = List[Int](); ph.resize(seq_pad, 0)
    var pw = List[Int](); pw.resize(seq_pad, 0)

    var vis_start = -1
    var cur = 0
    var i = 0
    while i < L:
        if ids[i] == IMAGE_TOKEN_ID:
            if vis_start < 0:
                vis_start = i
            for r in range(merged_h):
                for c in range(merged_w):
                    var idx = i + r * merged_w + c
                    pt[idx] = cur
                    ph[idx] = cur + r
                    pw[idx] = cur + c
            cur += advance
            i += nvis
        else:
            pt[i] = cur
            ph[i] = cur
            pw[i] = cur
            cur += 1
            i += 1
    if vis_start < 0:
        raise Error("fuse: no image_pad (151655) token found in input_ids")
    return Positions(pt^, ph^, pw^, vis_start, nvis)


# ── interleaved-mrope cos/sin tables (rope forward + apply_interleaved_mrope) ──
def _build_mrope_tables(
    pos: Positions, seq: Int, heads: Int, head_dim: Int, theta: Float64,
) raises -> List[List[Float32]]:
    """Build half-split rope cos/sin tables ordered (position, head, freq) for
    `heads` heads. Per position the 64 freqs pick their axis from the
    mrope_section=[24,20,20] interleave, replicated across all heads (rope angle
    depends only on the token's 3D position, not the head)."""
    var half = head_dim // 2         # 64
    var log_theta = flog(Float32(theta))
    var inv = List[Float32]()
    for k in range(half):
        inv.append(fexp(-log_theta * Float32(2 * k) / Float32(head_dim)))

    # Per-position 64 cos/sin, then head-replicate.
    var pc = List[Float32](); pc.resize(seq * half, Float32(0.0))
    var ps = List[Float32](); ps.resize(seq * half, Float32(0.0))
    for t in range(seq):
        for k in range(half):
            var axis_pos: Int
            if k >= 60 or (k % 3 == 0):
                axis_pos = pos.pt[t]
            elif k % 3 == 1:
                axis_pos = pos.ph[t]
            else:
                axis_pos = pos.pw[t]
            var angle = Float32(axis_pos) * inv[k]
            pc[t * half + k] = fcos(angle)
            ps[t * half + k] = fsin(angle)

    var cos_vals = List[Float32](); cos_vals.resize(seq * heads * half, Float32(0.0))
    var sin_vals = List[Float32](); sin_vals.resize(seq * heads * half, Float32(0.0))
    for t in range(seq):
        for hd in range(heads):
            var base = (t * heads + hd) * half
            var pbase = t * half
            for k in range(half):
                cos_vals[base + k] = pc[pbase + k]
                sin_vals[base + k] = ps[pbase + k]

    var out = List[List[Float32]]()
    out.append(cos_vals^)
    out.append(sin_vals^)
    return out^


# additive causal mask [1, heads, seq, seq] (== qwen3_encoder._build_causal_mask).
def _causal_mask_data(seq: Int, heads: Int, real_len: Int) raises -> List[Float32]:
    var neg = Float32(-1.0e4)
    var data = List[Float32]()
    for _hh in range(heads):
        for i in range(seq):
            for j in range(seq):
                if j <= i and j < real_len:
                    data.append(Float32(0.0))
                else:
                    data.append(neg)
    return data^


# ── row-region helpers (contiguous visual span) ──────────────────────────────
def _replace_rows(
    base: Tensor, repl: Tensor, start: Int, ctx: DeviceContext
) raises -> Tensor:
    """masked_scatter over the contiguous rows [start, start+nvis): base and repl
    are [1, seq, H] / [1, nvis, H]; returns base with those rows = repl."""
    var seq = base.shape()[1]
    var nvis = repl.shape()[1]
    var left = slice(base, 1, 0, start, ctx)
    var right = slice(base, 1, start + nvis, seq - start - nvis, ctx)
    return concat(1, ctx, left, repl, right)


def _add_rows(
    hidden: Tensor, feat: Tensor, start: Int, ctx: DeviceContext
) raises -> Tensor:
    """deepstack add: hidden[:, start:start+nvis] += feat. hidden [1,seq,H],
    feat [1,nvis,H]."""
    var seq = hidden.shape()[1]
    var nvis = feat.shape()[1]
    var mid = slice(hidden, 1, start, nvis, ctx)
    var mid2 = add(mid, feat, ctx)
    var left = slice(hidden, 1, 0, start, ctx)
    var right = slice(hidden, 1, start + nvis, seq - start - nvis, ctx)
    return concat(1, ctx, left, mid2, right)


def _to_rows(t: Tensor, nvis: Int, H: Int, dt: STDtype, ctx: DeviceContext) raises -> Tensor:
    """Vision pooler/deepstack are [nvis,H]; cast to [1,nvis,H] in dtype `dt`
    (bf16 == HF image_embeds.to(inputs_embeds.dtype); f32 for the parity
    high-precision stream)."""
    var host = t.to_host(ctx)
    return Tensor.from_host(host, [1, nvis, H], dt, ctx)


def _cast(t: Tensor, dt: STDtype, ctx: DeviceContext) raises -> Tensor:
    """Cast a Tensor to dtype `dt` (via host round-trip; parity/setup only)."""
    if t.dtype() == dt:
        return t.clone(ctx)
    var host = t.to_host(ctx)
    return Tensor.from_host(host, t.shape(), dt, ctx)


# ── full fused forward ───────────────────────────────────────────────────────
def fuse_core(
    enc: Qwen3Encoder,
    pooler: Tensor,       # [nvis, H]   vision pooler_output (F32 or bf16)
    ds0: Tensor, ds1: Tensor, ds2: Tensor,  # [nvis, H] deepstack features
    ids: List[Int],
    grid_h: Int, grid_w: Int,
    crop_start: Int,
    ctx: DeviceContext,
    f32_stream: Bool = False,
) raises -> Tensor:
    """The fusion math given already-computed vision tokens: masked_scatter the
    pooler rows into the token embeddings at image_pad (151655), run the Qwen3-VL
    text decoder with 3D interleaved M-RoPE and the deepstack add at layers
    0/1/2, apply model.norm, crop the leading `crop_start` rows. Returns
    [1, L-crop_start, 2560]. Splitting the vision run out lets the parity probe
    feed bit-exact torch vision tokens to isolate the fusion math.

    `f32_stream`: run the decoder ACTIVATION stream in F32 (weights stay bf16 —
    the same mixed path the vision tower uses). The image-token rows of the fused
    last_hidden_state are numerically ill-conditioned in bf16 (HF's own bf16-vs-
    f32 cos on those rows is ~0.95), so the fusion MATH can only be gated at
    cos>=0.999 in F32. Default False = the native bf16 product path."""
    var cfg = enc.config
    var H = cfg.hidden_size
    var heads = cfg.num_heads
    var kv = cfg.num_kv_heads
    var dh = cfg.head_dim
    var half = dh // 2
    var L = len(ids)
    var act_dt = STDtype.F32 if f32_stream else STDtype.BF16

    # positions + visual span
    var seq_pad = _pad_bucket(L)
    var pos = _build_positions(ids, L, seq_pad, grid_h, grid_w)
    var nvis = pos.nvis
    var vis_start = pos.vis_start

    # padded token ids (image rows overwritten below; pad rows masked)
    var padded = List[Int]()
    for i in range(L):
        padded.append(ids[i])
    for _i in range(seq_pad - L):
        padded.append(FUSE_PAD_ID)

    # 1) token embeddings (bf16 table), then masked_scatter the pooler rows
    var embed = _cast(enc._embed(padded, ctx), act_dt, ctx)   # [1,seq_pad,H]
    var pooler_a = _to_rows(pooler, nvis, H, act_dt, ctx)      # [1,nvis,H]
    var inputs_embeds = _replace_rows(embed, pooler_a, vis_start, ctx)

    # 2) interleaved M-RoPE tables (act dtype; HF computes cos/sin in f32)
    var q_tab = _build_mrope_tables(pos, seq_pad, heads, dh, cfg.rope_theta)
    var k_tab = _build_mrope_tables(pos, seq_pad, kv, dh, cfg.rope_theta)
    var cos_q = Tensor.from_host(q_tab[0], [seq_pad * heads * half], act_dt, ctx)
    var sin_q = Tensor.from_host(q_tab[1], [seq_pad * heads * half], act_dt, ctx)
    var cos_k = Tensor.from_host(k_tab[0], [seq_pad * kv * half], act_dt, ctx)
    var sin_k = Tensor.from_host(k_tab[1], [seq_pad * kv * half], act_dt, ctx)

    # causal mask (real columns only)
    var mask_data = _causal_mask_data(seq_pad, heads, L)
    var mask = Tensor.from_host(mask_data, [1, heads, seq_pad, seq_pad], act_dt, ctx)

    # deepstack features -> act dtype [1,nvis,H]
    var d0 = _to_rows(ds0, nvis, H, act_dt, ctx)
    var d1 = _to_rows(ds1, nvis, H, act_dt, ctx)
    var d2 = _to_rows(ds2, nvis, H, act_dt, ctx)

    # 3) decoder with deepstack add at layers 0/1/2
    var hidden = inputs_embeds^
    for i in range(cfg.num_layers):
        hidden = enc._layer(i, hidden, cos_q, sin_q, cos_k, sin_k, mask, ctx)
        if i == 0:
            hidden = _add_rows(hidden, d0, vis_start, ctx)
        elif i == 1:
            hidden = _add_rows(hidden, d1, vis_start, ctx)
        elif i == 2:
            hidden = _add_rows(hidden, d2, vis_start, ctx)

    # model.norm -> last_hidden_state, drop prefix + padding
    var normed = enc.final_norm(hidden, ctx)
    var keep = L - crop_start
    return slice(normed, 1, crop_start, keep, ctx)


def encode_lingbot_image_text[S: Int](
    vision: Qwen3VLVisionModel,
    enc: Qwen3Encoder,
    ids: List[Int],
    grid_t: Int, grid_h: Int, grid_w: Int,
    pixel_values: Tensor,
    crop_start: Int,
    ctx: DeviceContext,
    f32_stream: Bool = False,
) raises -> Tensor:
    """Image-grounded LingBot conditioning: run the vision tower, then fuse its
    tokens into the text decoder (see fuse_core). `S` is the vision patch seq
    (grid_t*h*w). Returns [1, L-crop_start, 2560]."""
    var vout = vision.forward[S](pixel_values, grid_t, grid_h, grid_w, ctx)
    return fuse_core(
        enc, vout.pooler, vout.ds0, vout.ds1, vout.ds2,
        ids, grid_h, grid_w, crop_start, ctx, f32_stream,
    )
