# Z-Image L2P 3-axis RoPE — host-built (cos, sin) tables.
#
# Direct port of /home/alex/EriDiffusion/inference-flame/src/models/l2p/rope.rs
# build_3d_rope. All math in Float32 on host; uploaded as BF16.
#
# Layout (matches Rust):
#   axes_dims = (T=32, H=48, W=48), sum = head_dim = 128. Each axis contributes
#   axis_dim/2 frequencies (16 + 24 + 24 = 64 = head_dim/2). Output `[total_seq,
#   head_dim/2]` packs axes contiguously along the last dim in order (T,H,W).
#   theta = 256.0. freq[i] = 1.0 / theta^(i/half_axis), i in [0, half_axis).
#
# Position grid:
#   caption [0..CAP_LEN):           (i+1, 0, 0)
#   image   [CAP_LEN..CAP_LEN+PH*PW): (CAP_LEN+1, ih, iw)
#   img-pad [CAP_LEN+PH*PW..total): (0, 0, 0)   (rows kept zero; Rust inits to
#                                                zeros and only overwrites the
#                                                populated ranges.)
#
# Consumer convention: angles are stored once per (token, half-pair) — when the
# block forward applies RoPE in BSHD `[B,S,H,Dh]` over `rope_interleaved`, the
# cos/sin must be replicated H times per token. That replication lives in the
# block_forward path (small data at smoke sizes), keeping this builder true to
# the Rust signature `[total_seq, head_dim/2]`.

from std.gpu.host import DeviceContext
from std.math import cos as fcos, log as flog, exp as fexp, sin as fsin

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor


# L2P RoPE constants — pinned to the Rust L2pDiTConfig defaults
# (axes_dims_rope=[32,48,48], rope_theta=256.0, head_dim=128).
comptime ZIMAGE_L2P_ROPE_THETA = Float32(256.0)
comptime ZIMAGE_L2P_ROPE_AXIS_T = 32
comptime ZIMAGE_L2P_ROPE_AXIS_H = 48
comptime ZIMAGE_L2P_ROPE_AXIS_W = 48
comptime ZIMAGE_L2P_ROPE_HEAD_DIM = 128
comptime ZIMAGE_L2P_ROPE_HALF_HEAD_DIM = ZIMAGE_L2P_ROPE_HEAD_DIM // 2  # 64


def build_zimage_l2p_3d_rope[CAP_LEN: Int, PH: Int, PW: Int, IMG_PAD: Int](
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    """Build the L2P (cos, sin) tables for caption + image + image-padding.

    Returns (cos, sin) both BF16, shape `[CAP_LEN + PH*PW + IMG_PAD, 64]`.
    Matches Rust `models/l2p/rope.rs::build_3d_rope` exactly.
    """
    comptime assert CAP_LEN >= 0, "CAP_LEN must be non-negative"
    comptime assert PH >= 0 and PW >= 0, "PH/PW must be non-negative"
    comptime assert IMG_PAD >= 0, "IMG_PAD must be non-negative"

    var total_seq = CAP_LEN + PH * PW + IMG_PAD
    var half_head_dim = ZIMAGE_L2P_ROPE_HALF_HEAD_DIM

    # pos_ids[seq][axis], default zeros — img-pad keeps (0,0,0).
    var pos_t = List[Float32]()
    var pos_h = List[Float32]()
    var pos_w = List[Float32]()
    for _ in range(total_seq):
        pos_t.append(Float32(0.0))
        pos_h.append(Float32(0.0))
        pos_w.append(Float32(0.0))

    # Caption: (i+1, 0, 0)
    for i in range(CAP_LEN):
        pos_t[i] = Float32(i + 1)

    # Image: (CAP_LEN+1, ih, iw)
    for ih in range(PH):
        for iw in range(PW):
            var idx = CAP_LEN + ih * PW + iw
            pos_t[idx] = Float32(CAP_LEN + 1)
            pos_h[idx] = Float32(ih)
            pos_w[idx] = Float32(iw)

    # Output buffers — row-major [total_seq, half_head_dim].
    var cos_data = List[Float32]()
    var sin_data = List[Float32]()
    for _ in range(total_seq * half_head_dim):
        cos_data.append(Float32(0.0))
        sin_data.append(Float32(0.0))

    var theta = ZIMAGE_L2P_ROPE_THETA
    var log_theta = flog(theta)

    # Walk axes (T, H, W) and concatenate freqs along the last dim.
    var offset = 0
    for axis_idx in range(3):
        var axis_dim: Int
        if axis_idx == 0:
            axis_dim = ZIMAGE_L2P_ROPE_AXIS_T
        elif axis_idx == 1:
            axis_dim = ZIMAGE_L2P_ROPE_AXIS_H
        else:
            axis_dim = ZIMAGE_L2P_ROPE_AXIS_W
        var half_axis = axis_dim // 2

        # freq[i] = 1 / theta^(i/half_axis) = exp(-log_theta * i/half_axis)
        var freqs = List[Float32]()
        for i in range(half_axis):
            var f = fexp(-log_theta * Float32(i) / Float32(half_axis))
            freqs.append(f)

        for seq_idx in range(total_seq):
            var pos: Float32
            if axis_idx == 0:
                pos = pos_t[seq_idx]
            elif axis_idx == 1:
                pos = pos_h[seq_idx]
            else:
                pos = pos_w[seq_idx]
            for freq_idx in range(half_axis):
                var angle = pos * freqs[freq_idx]
                var out_idx = seq_idx * half_head_dim + offset + freq_idx
                cos_data[out_idx] = fcos(angle)
                sin_data[out_idx] = fsin(angle)

        offset += half_axis

    var shape = List[Int]()
    shape.append(total_seq)
    shape.append(half_head_dim)

    var cos_t = Tensor.from_host(cos_data, shape.copy(), STDtype.BF16, ctx)
    var sin_t = Tensor.from_host(sin_data, shape^, STDtype.BF16, ctx)
    return (cos_t^, sin_t^)
