# models/dit/ltx2_rope.mojo — LTX-2 3D split-RoPE (video path), pure Mojo.
#
# Port of inference-flame/src/models/ltx2_model.rs:
#   compute_rope_frequencies (line 373)  -> build_ltx2_rope
#   apply_rotary_emb         (line 492)  -> apply_ltx2_rope
#
# LTX-2 uses a SPLIT (half-split) rotary over 3 latent axes (frame, height,
# width). The convention (ltx2_model.rs:481-484):
#   first_half_out  = first_half  * cos - second_half * sin
#   second_half_out = second_half * cos + first_half  * sin
# which is EXACTLY serenitymojo `rope_halfsplit` (ops/rope.mojo:364, kernel
# lines 100-118). So we build [rows, head_dim/2] cos/sin tables and feed them
# straight in — no new kernel.
#
# ── Frequency construction (ltx2_model.rs:386-453) ───────────────────────────
#   num_pos_dims = 3 (frame, height, width)
#   num_rope_elems = num_pos_dims * 2 = 6
#   freq_count = dim / num_rope_elems          (dim = inner_dim = H * head_dim)
#   for i in 0..freq_count:
#       t = i / max(freq_count - 1, 1)
#       freq[i] = theta^t * pi/2                (NOTE: theta^t, NOT 1/theta^t)
#   grid[axis] = midpoint[axis] / max_positions[axis]        (fractional pos)
#   angles[axis, i] = (2*grid[axis] - 1) * freq[i]
#   flatten as [.. , freq_count, num_pos_dims] -> per-token row of length
#       rope_freqs = freq_count * num_pos_dims
#   cos/sin of angles; if rope_freqs < half_dim (= dim/2), FRONT-pad with
#       ones(cos)/zeros(sin) to half_dim.
#   reshape half_dim -> [num_heads, head_dim/2] (head_rope_dim = half_dim/H).
#
# For the bounded smoke we materialise per (token, head, half-pair) on the host
# and upload a [B*H*N, head_dim/2] table. Since the LTX-2 angle depends only on
# the token (the head split is just a contiguous slice of the per-token
# half-vector), head h gets half-vector slice [h*head_rope_dim : (h+1)*...].
#
# Coordinates: matching the Lightricks reference, each latent token at
# (f, y, x) uses the midpoint of its [start, end) patch boundary. With unit
# patch_size (patch_size == patch_size_t == 1 in LTX2Config) start==index and
# end==index+1, so midpoint = index + 0.5.  max_positions per axis come from
# the patch-grid extent; we follow the reference normalisation grid/max.
#
# Mojo 1.0.0b1, NVIDIA GPU. *** CODE-ONLY: compile-verified; NOT executed. ***

from std.gpu.host import DeviceContext
from std.math import cos as fcos, sin as fsin, pow as fpow, pi

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.rope import rope_halfsplit


# ── frequency table (per-token half-vector of length dim/2) ──────────────────
# Returns the per-token, per-half-pair cos/sin host vectors, length
# n_tokens * (dim/2), row-major [token, half_idx].
def _ltx2_freq_host(
    n_frame: Int,
    n_h: Int,
    n_w: Int,
    inner_dim: Int,
    theta: Float64,
    max_f: Float64,
    max_h: Float64,
    max_w: Float64,
) raises -> Tuple[List[Float32], List[Float32]]:
    var num_pos_dims = 3
    var num_rope_elems = num_pos_dims * 2          # 6
    var freq_count = inner_dim // num_rope_elems     # e.g. 4096/6 = 682
    var rope_freqs = freq_count * num_pos_dims        # 682*3 = 2046
    var half_dim = inner_dim // 2                     # 2048
    if rope_freqs > half_dim:
        raise Error("ltx2 rope: rope_freqs exceeds half_dim")
    var pad = half_dim - rope_freqs                   # front pad count

    # freq[i] = theta^(i / max(freq_count-1, 1)) * pi/2     (ltx2_model.rs:405-408)
    var denom = Float64(freq_count - 1)
    if denom < 1.0:
        denom = 1.0
    var freq = List[Float32]()
    for i in range(freq_count):
        var t = Float64(i) / denom
        freq.append(Float32(fpow(theta, t) * pi / 2.0))

    var n_tokens = n_frame * n_h * n_w
    var cos_out = List[Float32]()
    var sin_out = List[Float32]()
    # Reserve n_tokens * half_dim entries.
    for _ in range(n_tokens * half_dim):
        cos_out.append(Float32(0.0))
        sin_out.append(Float32(0.0))

    for tok in range(n_tokens):
        var f_idx = tok // (n_h * n_w)
        var rem = tok % (n_h * n_w)
        var y_idx = rem // n_w
        var x_idx = rem % n_w
        # midpoint of [start,end) with unit patch => index + 0.5
        var mid_f = Float64(f_idx) + 0.5
        var mid_y = Float64(y_idx) + 0.5
        var mid_x = Float64(x_idx) + 0.5
        # grid = midpoint / max_positions[axis]
        var g_f = mid_f / max_f
        var g_y = mid_y / max_h
        var g_x = mid_x / max_w
        # scaled = 2*grid - 1   (ltx2_model.rs:415)
        var s_f = 2.0 * g_f - 1.0
        var s_y = 2.0 * g_y - 1.0
        var s_x = 2.0 * g_x - 1.0

        var row_base = tok * half_dim
        # Front padding: cos=1, sin=0 (ltx2_model.rs:429-440).
        for p in range(pad):
            cos_out[row_base + p] = Float32(1.0)
            sin_out[row_base + p] = Float32(0.0)
        # Body order is [freq_count, num_pos_dims] flattened
        # (ltx2_model.rs:418-421: permute to [.., freq_count, num_pos_dims]).
        var off = row_base + pad
        for i in range(freq_count):
            var fi = Float64(freq[i])
            var a_f = s_f * fi
            var a_y = s_y * fi
            var a_x = s_x * fi
            cos_out[off + 0] = Float32(fcos(a_f))
            sin_out[off + 0] = Float32(fsin(a_f))
            cos_out[off + 1] = Float32(fcos(a_y))
            sin_out[off + 1] = Float32(fsin(a_y))
            cos_out[off + 2] = Float32(fcos(a_x))
            sin_out[off + 2] = Float32(fsin(a_x))
            off += num_pos_dims
    return (cos_out^, sin_out^)


# ── public table builder ─────────────────────────────────────────────────────
# Builds per (token, head) cos/sin tables shaped [B*H*N, head_dim/2] for the
# half-split kernel.  inner_dim = num_heads * head_dim.  The per-token half
# vector of length inner_dim/2 is split into num_heads contiguous chunks of
# head_dim/2 (ltx2_model.rs:445-450 reshape half_dim -> [H, head_dim/2]).
def build_ltx2_rope[F: Int, H: Int, W: Int](
    num_heads: Int,
    head_dim: Int,
    theta: Float64,
    max_f: Float64,
    max_h: Float64,
    max_w: Float64,
    dtype: STDtype,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    var inner_dim = num_heads * head_dim
    var half_head = head_dim // 2
    var half_dim = inner_dim // 2
    if num_heads * half_head != half_dim:
        raise Error("ltx2 rope: inner_dim/2 != heads * head_dim/2")

    var freqs = _ltx2_freq_host(
        F, H, W, inner_dim, theta, max_f, max_h, max_w
    )
    ref cos_tok = freqs[0]   # [n_tokens, half_dim]
    ref sin_tok = freqs[1]

    var n_tokens = F * H * W
    # Re-lay into [B*H*N, head_dim/2] = [token*head, half_head]. Batch B == 1.
    var cos_rows = List[Float32]()
    var sin_rows = List[Float32]()
    for tok in range(n_tokens):
        for h in range(num_heads):
            var src = tok * half_dim + h * half_head
            for j in range(half_head):
                cos_rows.append(cos_tok[src + j])
                sin_rows.append(sin_tok[src + j])

    var sh = List[Int]()
    sh.append(n_tokens * num_heads)
    sh.append(half_head)
    var cos_t = Tensor.from_host(cos_rows, sh.copy(), dtype, ctx)
    var sin_t = Tensor.from_host(sin_rows, sh^, dtype, ctx)
    return (cos_t^, sin_t^)


# ── apply ────────────────────────────────────────────────────────────────────
# x is the per-head Q or K in layout [B, H, N, head_dim] (BHND, B==1). The
# half-split kernel flattens leading dims to rows, so cos/sin must be
# [B*H*N, head_dim/2] in the SAME (b,h,n) row order. apply_ltx2_rope therefore
# expects x already permuted to [B, H, N, head_dim].
def apply_ltx2_rope(
    x: Tensor, cos: Tensor, sin: Tensor, ctx: DeviceContext
) raises -> Tensor:
    return rope_halfsplit(x, cos, sin, ctx)
