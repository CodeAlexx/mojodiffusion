# conditioning_mask.mojo -- LTX2 intrinsic-conditioning per-token mask builder
# (P5.5 unit 1). Generalizes the P5 first-frame seam: EVERY condition emits a
# [seq] {0,1} mask; composition = per-token UNION of independently-drawn
# conditions; the combined mask drives the SAME three surfaces the first-frame
# wiring already implements (clean-replace / timestep-0 / loss-off).
#
# Pure HOST index arithmetic, ported 1:1 from the official LTX-2 trainer
# (LTX-2-upstream/packages/ltx-trainer/.../training_strategies/flexible.py):
#   _compute_temporal_mask       :521-539
#   _compute_spatial_crop_mask   :541-570
#   _apply_intrinsic_condition   :464-519 (the 3-surface application + union)
# Token order frame->h->w, tokens_per_frame = H*W, seq = NF*H*W.
#
# ZERO model/DiT change: the per-token timesteps the mask edits are already
# consumed by the head's per-token AdaLN (P5). The per-condition Bernoulli DRAW
# is the caller's (own seed namespace); this module is the deterministic mask
# geometry only.

from std.collections import List

comptime VAE_SF_HW = 32   # official VIDEO_SCALE_FACTORS.height == .width == 32


def _zeros(seq: Int) -> List[Bool]:
    var m = List[Bool]()
    for _ in range(seq):
        m.append(False)
    return m^


# flexible.py:_compute_temporal_mask (:521-539): the first (from_end=False /
# PREFIX) or last (from_end=True / SUFFIX) num_frames*H*W tokens are conditioned.
# FIRST_FRAME = temporal_mask(num_frames=1, from_end=False) (:491-492).
def temporal_mask(nf: Int, nh: Int, nw: Int, num_frames: Int, from_end: Bool) -> List[Bool]:
    var seq = nf * nh * nw
    var tokens_per_frame = nh * nw
    var num_tokens = num_frames * tokens_per_frame
    var m = _zeros(seq)
    if from_end:
        var start = seq - num_tokens
        if start < 0:
            start = 0
        for i in range(start, seq):
            m[i] = True
    else:
        var end = num_tokens
        if end > seq:
            end = seq
        for i in range(end):
            m[i] = True
    return m^


def first_frame_mask(nf: Int, nh: Int, nw: Int) -> List[Bool]:
    return temporal_mask(nf, nh, nw, 1, False)


# max(0, min(v // scale, max_v)) -- official to_latent (:557-558).
def _to_latent(v: Int, scale: Int, max_v: Int) -> Int:
    var lv = v // scale
    if lv < 0:
        lv = 0
    if lv > max_v:
        lv = max_v
    return lv


# flexible.py:_compute_spatial_crop_mask (:541-570): pixel rect (y1,x1,y2,x2) ->
# //32 latent clamp -> [H,W] rectangle spatial_mask[ly1:ly2, lx1:lx2]=1 -> tiled
# (repeated) across all NF frames. VIDEO-ONLY per the official strategy.
def spatial_crop_mask(nf: Int, nh: Int, nw: Int, y1: Int, x1: Int, y2: Int, x2: Int) -> List[Bool]:
    var ly1 = _to_latent(y1, VAE_SF_HW, nh)
    var ly2 = _to_latent(y2, VAE_SF_HW, nh)
    var lx1 = _to_latent(x1, VAE_SF_HW, nw)
    var lx2 = _to_latent(x2, VAE_SF_HW, nw)
    var m = List[Bool]()
    for _f in range(nf):
        for h in range(nh):
            for w in range(nw):
                m.append(h >= ly1 and h < ly2 and w >= lx1 and w < lx2)
    return m^


# Per-token UNION (flexible.py applies conditions sequentially; the net effect on
# a token is the OR of the conditions that fired). Independent per-condition draws
# are the caller's; this ORs the already-drawn masks. Unfired conditions pass an
# all-False mask (== the official `mask * apply_per_sample` zeroing at bs=1).
def union_into(mut acc: List[Bool], m: List[Bool]) raises:
    if len(acc) != len(m):
        raise Error("conditioning_mask.union_into: length mismatch")
    for i in range(len(acc)):
        if m[i]:
            acc[i] = True
