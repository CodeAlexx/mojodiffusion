# models/dit/parity/krea2_rope_parity_probe.mojo — PARITY GATE for krea2 chunk 1
# (Krea2Config + 3-axis interleaved RoPE) vs a real ai-toolkit torch oracle.
#
# Oracle: krea2_rope_oracle.safetensors, generated from the REAL
# PositionalEncoding + ropeapply (mmdit.py). All tensors F32:
#   pos      [1,27,3]      axis order [global,h,w]; token-major (incl. a large
#                          global pos ~2000+ to stress F64 range reduction).
#   q        [1,48,27,128] pre-RoPE (B,H=48,L=27,D=128)
#   k        [1,12,27,128] pre-RoPE (B,KVH=12,L,D)
#   cos,sin  [27,64]       per-pair angle tables (cos=freqs[..,0,0], sin=freqs[..,1,0])
#   q_roped  [1,48,27,128] ropeapply ground truth
#   k_roped  [1,12,27,128] ropeapply ground truth
#
# Gate: cos >= 0.999 on cos_table, sin_table, q_roped, k_roped (expect ~0.9999+).
#
# Run: cd /home/alex/mojodiffusion && \
#   pixi run mojo run -I . serenitymojo/models/dit/parity/krea2_rope_parity_probe.mojo
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.models.dit.krea2_dit import Krea2Config, build_krea2_rope


comptime ORACLE = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/krea2_rope_oracle.safetensors"


def _tile_table_rows(
    table: Tensor, n_rep: Int, L: Int, half: Int, ctx: DeviceContext
) raises -> Tensor:
    """Repeat the whole [L, half] table `n_rep` times along dim0 -> [n_rep*L, half].

    rope_interleaved pairs row r of q with row r of cos/sin. The oracle q has
    `n_rep` heads, flattened to rows (h*L + l), so each head re-uses the same
    per-token table. Row (h*L + l) must equal table[l] -> tile the [L,half]
    block n_rep times. Built host-side in F32, uploaded fresh."""
    var src = table.to_host(ctx)  # [L*half] F32, row-major
    var out = List[Float32]()
    for _h in range(n_rep):
        for r in range(L * half):
            out.append(src[r])
    var shape = List[Int]()
    shape.append(n_rep * L)
    shape.append(half)
    return Tensor.from_host(out^, shape^, STDtype.F32, ctx)


def main() raises:
    var ctx = DeviceContext()
    var fx = ShardedSafeTensors.open(ORACLE)

    comptime L = 27
    comptime D = 128
    comptime HALF = 64
    comptime QH = 48
    comptime KH = 12

    var cfg = Krea2Config.default()
    var axes = cfg.rope_axes()  # [32,48,48]

    # ── Load oracle tensors (all F32) ────────────────────────────────────────
    # pos [1,27,3] -> flatten to [81] (already token-major t*3+a).
    var pos_raw = Tensor.from_view_as_f32(fx.tensor_view("pos"), ctx)
    var pos_flat_shape = List[Int]()
    pos_flat_shape.append(L * 3)
    var pos = reshape(pos_raw, pos_flat_shape^, ctx)

    var q = Tensor.from_view_as_f32(fx.tensor_view("q"), ctx)         # [1,48,27,128]
    var k = Tensor.from_view_as_f32(fx.tensor_view("k"), ctx)         # [1,12,27,128]
    var cos_ref = Tensor.from_view_as_f32(fx.tensor_view("cos"), ctx).to_host(ctx)
    var sin_ref = Tensor.from_view_as_f32(fx.tensor_view("sin"), ctx).to_host(ctx)
    var q_roped_ref = Tensor.from_view_as_f32(fx.tensor_view("q_roped"), ctx).to_host(ctx)
    var k_roped_ref = Tensor.from_view_as_f32(fx.tensor_view("k_roped"), ctx).to_host(ctx)

    # ── Build the table and compare cos/sin directly ─────────────────────────
    var tables = build_krea2_rope(pos, axes, cfg.theta, ctx, STDtype.F32)
    var cos_res = ParityHarness(0.999).compare(tables[0], cos_ref, ctx)
    var sin_res = ParityHarness(0.999).compare(tables[1], sin_ref, ctx)
    print("cos_table parity:", cos_res)
    print("sin_table parity:", sin_res)

    # ── Apply RoPE to q (48 heads) and k (12 heads), compare ─────────────────
    # Reshape q [1,48,27,128] -> [48*27, 128]; tile table -> [48*27, 64].
    var q_rows_shape = List[Int]()
    q_rows_shape.append(QH * L)
    q_rows_shape.append(D)
    var q_rows = reshape(q, q_rows_shape^, ctx)
    var q_cos = _tile_table_rows(tables[0], QH, L, HALF, ctx)
    var q_sin = _tile_table_rows(tables[1], QH, L, HALF, ctx)
    var q_out = rope_interleaved(q_rows, q_cos, q_sin, ctx)
    var q_res = ParityHarness(0.999).compare(q_out, q_roped_ref, ctx)
    print("q_roped parity:", q_res)

    var k_rows_shape = List[Int]()
    k_rows_shape.append(KH * L)
    k_rows_shape.append(D)
    var k_rows = reshape(k, k_rows_shape^, ctx)
    var k_cos = _tile_table_rows(tables[0], KH, L, HALF, ctx)
    var k_sin = _tile_table_rows(tables[1], KH, L, HALF, ctx)
    var k_out = rope_interleaved(k_rows, k_cos, k_sin, ctx)
    var k_res = ParityHarness(0.999).compare(k_out, k_roped_ref, ctx)
    print("k_roped parity:", k_res)
