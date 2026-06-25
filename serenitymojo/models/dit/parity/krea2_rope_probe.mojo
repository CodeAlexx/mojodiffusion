# models/dit/parity/krea2_rope_probe.mojo — smoke probe for build_krea2_rope +
# apply_krea2_rope. Builds the 3-axis table for a tiny pos (L=8) and applies it to
# tiny q/k [B=1, H=2, L=8, 128]. Prints shapes + a few output values and a
# host-recomputed reference for one (head, token, pair) to self-check the math.
#
# Run: cd /home/alex/mojodiffusion && \
#   pixi run mojo run -I . serenitymojo/models/dit/parity/krea2_rope_probe.mojo
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.math import cos as fcos, sin as fsin, exp, log, floor
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.krea2_dit import (
    Krea2Config,
    build_krea2_rope,
    apply_krea2_rope,
)


def main() raises:
    var ctx = DeviceContext()

    # Config sanity: headdim=128, axes=[32,48,48], theta=1e3.
    var cfg = Krea2Config.default()
    var hd = cfg.head_dim()
    var axes = cfg.rope_axes()
    print("Krea2Config: features=", cfg.features, " heads=", cfg.heads,
          " head_dim=", hd, " theta=", cfg.theta)
    print("rope_axes = [", axes[0], ",", axes[1], ",", axes[2], "]  sum=",
          axes[0] + axes[1] + axes[2])
    if axes[0] + axes[1] + axes[2] != hd:
        raise Error("axes do not sum to head_dim")

    # Tiny positions: L=8 tokens, 3 axes (global, h, w), small integer coords.
    # token-major: positions[t*3 + a]. Use a 4x2 image grid (h in 0..3, w in 0..1),
    # global = flat token index.
    comptime L = 8
    var pos_host = List[Float32]()
    for t in range(L):
        var h = t // 2
        var w = t % 2
        pos_host.append(Float32(t))      # global
        pos_host.append(Float32(h))      # h
        pos_host.append(Float32(w))      # w
    var pos_shape = List[Int]()
    pos_shape.append(L * 3)
    var positions = Tensor.from_host(pos_host, pos_shape^, STDtype.F32, ctx)

    # Build cos/sin in F32 (table dtype). Keep the tuple alive and index it
    # (cos/sin are Movable-not-Copyable; the tuple owns them — index by ref).
    var tables = build_krea2_rope(positions, axes, cfg.theta, ctx, STDtype.F32)
    var cshape = tables[0].shape()
    print("cos shape = [", cshape[0], ",", cshape[1], "]  (expect [", L, ",",
          hd // 2, "])")

    # Read a few table values for inspection.
    var cos_host = tables[0].to_host(ctx)
    var sin_host = tables[1].to_host(ctx)
    var half = hd // 2
    print("cos[token0, pair0..3] =", cos_host[0], cos_host[1], cos_host[2],
          cos_host[3])
    print("cos[token5, pair0..3] =", cos_host[5 * half + 0],
          cos_host[5 * half + 1], cos_host[5 * half + 2], cos_host[5 * half + 3])
    print("sin[token5, pair0..3] =", sin_host[5 * half + 0],
          sin_host[5 * half + 1], sin_host[5 * half + 2], sin_host[5 * half + 3])

    # Tiny q/k: [B=1, H=2, L=8, 128]. rows = B*H*L = 16. NOTE: the table has L=8
    # rows, so to apply per-token rope the q/k rows must align with the table
    # rows. rope_interleaved flattens leading dims to rows and pairs row r of x
    # with row r of cos/sin. To keep the probe self-contained we apply rope to a
    # SINGLE head's [L, 128] block (rows == L == 8), which is the per-(B,H) unit
    # the real attention path rotates.
    comptime D = 128
    var q_host = List[Float32]()
    var k_host = List[Float32]()
    for r in range(L):
        for d in range(D):
            # deterministic small values
            q_host.append(Float32(0.01) * Float32((r * D + d) % 17) - Float32(0.05))
            k_host.append(Float32(0.02) * Float32((r * D + d) % 11) - Float32(0.03))
    var q_shape = List[Int]()
    q_shape.append(L)
    q_shape.append(D)
    var k_shape = List[Int]()
    k_shape.append(L)
    k_shape.append(D)
    var q = Tensor.from_host(q_host.copy(), q_shape^, STDtype.F32, ctx)
    var k = Tensor.from_host(k_host.copy(), k_shape^, STDtype.F32, ctx)

    var rotated = apply_krea2_rope(q, k, tables[0], tables[1], ctx)
    var qr_shape = rotated[0].shape()
    print("q_rot shape = [", qr_shape[0], ",", qr_shape[1], "]")
    var q_rot_host = rotated[0].to_host(ctx)
    var k_rot_host = rotated[1].to_host(ctx)
    print("q_rot[token0, 0..3] =", q_rot_host[0], q_rot_host[1], q_rot_host[2],
          q_rot_host[3])
    print("q_rot[token5, 0..3] =", q_rot_host[5 * D + 0], q_rot_host[5 * D + 1],
          q_rot_host[5 * D + 2], q_rot_host[5 * D + 3])
    print("k_rot[token5, 0..3] =", k_rot_host[5 * D + 0], k_rot_host[5 * D + 1],
          k_rot_host[5 * D + 2], k_rot_host[5 * D + 3])

    # ── Host reference self-check ────────────────────────────────────────────
    # Recompute the rope for (token=5, pair=0) and (token=5, pair=20) on host in
    # F64 (matching the kernel's omega + range reduction) and verify q_rot.
    # axes_half = [16, 24, 24]; offsets = [0, 16, 40].
    var axes_half = List[Int]()
    axes_half.append(axes[0] // 2)
    axes_half.append(axes[1] // 2)
    axes_half.append(axes[2] // 2)

    var max_err: Float64 = 0.0
    var t = 5
    var hh = t // 2  # h coord
    var ww = t % 2   # w coord
    var pos_for_axis = List[Float64]()
    pos_for_axis.append(Float64(t))   # global
    pos_for_axis.append(Float64(hh))  # h
    pos_for_axis.append(Float64(ww))  # w
    var lt = Float64(log(cfg.theta))
    comptime TWO_PI = Float64(6.283185307179586476925286766559)
    for col in range(half):
        # find owning axis + local index
        var off = 0
        var a = 0
        var local_i = col
        for ax in range(3):
            var ha = axes_half[ax]
            if col < off + ha:
                a = ax
                local_i = col - off
                break
            off += ha
        var ha_owner = axes_half[a]
        var inv = exp((-Float64(local_i) / Float64(ha_owner)) * lt)
        var angle = pos_for_axis[a] * inv
        var kk = floor(angle / TWO_PI + 0.5)
        var reduced = Float32(angle - kk * TWO_PI)
        var cv = Float64(fcos(reduced))
        var sv = Float64(fsin(reduced))
        # interleaved pair (2*col, 2*col+1) of token t
        var x0 = Float64(q_host[t * D + 2 * col])
        var x1 = Float64(q_host[t * D + 2 * col + 1])
        var ref0 = x0 * cv - x1 * sv
        var ref1 = x0 * sv + x1 * cv
        var got0 = Float64(q_rot_host[t * D + 2 * col])
        var got1 = Float64(q_rot_host[t * D + 2 * col + 1])
        var e0 = abs(ref0 - got0)
        var e1 = abs(ref1 - got1)
        if e0 > max_err:
            max_err = e0
        if e1 > max_err:
            max_err = e1
    print("host-ref max abs err over token5 q_rot pairs =", max_err)
    if max_err > Float64(1.0e-5):
        raise Error("krea2 rope self-check FAILED: max_err too large")
    print("PROBE OK")
