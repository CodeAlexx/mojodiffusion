# serenitymojo/models/minimax_h3/rearrange.mojo
#
# MiniMax-H3 tensor rearrangements — unit 6 of the H3 port.
#
# The permutations that turn latents into packed rows and rows back into
# latents, plus the rotate-half rotary application. Every one is an index
# shuffle with no arithmetic to check, which is exactly what makes them
# dangerous: a transposed pair of axes yields output of the right shape and
# dtype that is silently scrambled. That survives every smoke test and surfaces
# as "the video looks wrong somehow".
#
# Source (op-for-op port, diffusers PR huggingface/diffusers#14355 at head
# e1b518dfd5e390e7ba09a79a1d39fe1c6cb52dc1):
#   src/diffusers/modular_pipelines/minimax_h3/packing.py
#     patchify_video_latents    :246-275
#     unpatchify_video_tokens   :278-312
#     unpack_audio_tokens       :315-328
#   src/diffusers/models/transformers/transformer_minimax_h3.py
#     _apply_rotary_emb         :56-70
#
# Layouts, all flat row-major here:
#   video latents  [C, T, H, W]                (batch 1; H3 is batch-1 only)
#   video rows     [T*(H/ph)*(W/pw), C*pt*ph*pw]
#   audio rows     [num_audio_latents*2, C]    channel-major stereo
#   audio latents  [2, C, num_audio_latents]   one batch item per stereo channel
#
# The video row order is frame-major then row-major then column-major, and
# WITHIN a row the channel varies slowest and the patch offsets fastest —
# `permute(0, 2, 4, 6, 1, 3, 5, 7)` on `[b, c, t, pt, h, ph, w, pw]`.

from std.collections import List


def minimax_h3_patchify_video(
    latents: List[Float32],
    channels: Int,
    num_frames: Int,
    height: Int,
    width: Int,
    patch_t: Int,
    patch_h: Int,
    patch_w: Int,
) raises -> List[Float32]:
    """`[C, T, H, W]` latents -> `[T'*H'*W', C*pt*ph*pw]` rows (packing.py:246).

    Source index is `((c * T + t) * H + h) * W + w`; destination is the
    `permute(0, 2, 4, 6, 1, 3, 5, 7)` of `[b, c, t, pt, h, ph, w, pw]`, i.e.
    row `(t', h', w')` and within it `((c * pt + it) * ph + ih) * pw + iw`."""
    if num_frames % patch_t != 0 or height % patch_h != 0 or width % patch_w != 0:
        raise Error("MiniMax-H3: latent shape is not divisible by the patch")

    var out_t = num_frames // patch_t
    var out_h = height // patch_h
    var out_w = width // patch_w
    var row_width = channels * patch_t * patch_h * patch_w

    var out = List[Float32]()
    for _ in range(out_t * out_h * out_w * row_width):
        out.append(Float32(0.0))

    for t in range(out_t):
        for h in range(out_h):
            for w in range(out_w):
                var row = (t * out_h + h) * out_w + w
                for c in range(channels):
                    for it in range(patch_t):
                        for ih in range(patch_h):
                            for iw in range(patch_w):
                                var column = ((c * patch_t + it) * patch_h + ih) * patch_w + iw
                                var src_t = t * patch_t + it
                                var src_h = h * patch_h + ih
                                var src_w = w * patch_w + iw
                                var src = ((c * num_frames + src_t) * height + src_h) * width + src_w
                                out[row * row_width + column] = latents[src]
    return out^


def minimax_h3_unpatchify_video(
    rows: List[Float32],
    channels: Int,
    num_frames: Int,
    height: Int,
    width: Int,
    patch_t: Int,
    patch_h: Int,
    patch_w: Int,
) raises -> List[Float32]:
    """Inverse of `minimax_h3_patchify_video` (packing.py:278).

    `num_frames`/`height`/`width` are the LATENT dimensions, as in the
    reference — it divides them by the patch internally."""
    if num_frames % patch_t != 0 or height % patch_h != 0 or width % patch_w != 0:
        raise Error("MiniMax-H3: latent shape is not divisible by the patch")

    var out_t = num_frames // patch_t
    var out_h = height // patch_h
    var out_w = width // patch_w
    var row_width = channels * patch_t * patch_h * patch_w

    var out = List[Float32]()
    for _ in range(channels * num_frames * height * width):
        out.append(Float32(0.0))

    for t in range(out_t):
        for h in range(out_h):
            for w in range(out_w):
                var row = (t * out_h + h) * out_w + w
                for c in range(channels):
                    for it in range(patch_t):
                        for ih in range(patch_h):
                            for iw in range(patch_w):
                                var column = ((c * patch_t + it) * patch_h + ih) * patch_w + iw
                                var dst_t = t * patch_t + it
                                var dst_h = h * patch_h + ih
                                var dst_w = w * patch_w + iw
                                var dst = ((c * num_frames + dst_t) * height + dst_h) * width + dst_w
                                out[dst] = rows[row * row_width + column]
    return out^


def minimax_h3_unpack_audio(
    rows: List[Float32], num_audio_latents: Int, channels: Int
) -> List[Float32]:
    """`[num_audio_latents*2, C]` channel-major rows -> `[2, C, T]` latents
    (packing.py:315).

    The row block for stereo channel 0 comes first, then channel 1; the mono
    audio VAE consumes the two as separate batch items."""
    var out = List[Float32]()
    for _ in range(2 * channels * num_audio_latents):
        out.append(Float32(0.0))
    for stereo in range(2):
        for t in range(num_audio_latents):
            var row = stereo * num_audio_latents + t
            for c in range(channels):
                var dst = (stereo * channels + c) * num_audio_latents + t
                out[dst] = rows[row * channels + c]
    return out^


def minimax_h3_apply_rotary(
    hidden: List[Float32],
    cos: List[Float32],
    sin: List[Float32],
    sequence_length: Int,
    heads: Int,
    head_dim: Int,
    rotary_dim: Int,
) raises -> List[Float32]:
    """Rotate the leading `rotary_dim` channels of every head; pass the rest
    through (transformer:56-70).

    `hidden` is flat `[S, H, D]` (batch 1). The rotate-half convention splits
    the rotary block in two and forms `(-x2, x1)`, so channel `i` in the first
    half pairs with channel `i + rotary_dim/2` — NOT with its neighbour. Getting
    that wrong is the classic rope bug and it still produces plausible output.

    cos/sin are `[S, rotary_dim]` and are shared across heads."""
    if rotary_dim > head_dim:
        raise Error("MiniMax-H3: rotary_dim exceeds head_dim")
    if rotary_dim % 2 != 0:
        raise Error("MiniMax-H3: rotary_dim must be even")

    var half = rotary_dim // 2
    var out = List[Float32]()
    for _ in range(sequence_length * heads * head_dim):
        out.append(Float32(0.0))

    for s in range(sequence_length):
        for h in range(heads):
            var base = (s * heads + h) * head_dim
            for i in range(rotary_dim):
                var partner: Float32
                if i < half:
                    # first half pairs with -(second half)
                    partner = -hidden[base + i + half]
                else:
                    partner = hidden[base + i - half]
                var c = cos[s * rotary_dim + i]
                var sn = sin[s * rotary_dim + i]
                out[base + i] = hidden[base + i] * c + partner * sn
            for i in range(rotary_dim, head_dim):
                out[base + i] = hidden[base + i]
    return out^
