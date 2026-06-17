# boogu_c2_rope_probe.mojo — compile+run probe for Boogu-Image 3-axis RoPE (C2).
#
# Builds the joint RoPE cos/sin tables for the T2I no-ref batch=1 case
# (cap_len=16, h_tok=16, w_tok=16 => seq=272) via
# models/dit/boogu_dit.build_boogu_rope_tables, and prints cos/sin shapes, a few
# values, and std. The orchestrator owns parity vs the torch oracle
# (boogu_c2_oracle.py: cos↔real, sin↔imag); this probe only proves the Mojo code
# COMPILES and EXECUTES (exit 0).
#
# Run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg \
#     && pixi run mojo run -I . serenitymojo/models/dit/parity/boogu_c2_rope_probe.mojo

from std.math import sqrt
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.boogu_dit import build_boogu_rope_tables


def _std(host: List[Float32]) -> Float32:
    var n = len(host)
    if n == 0:
        return Float32(0.0)
    var mean = Float32(0.0)
    for i in range(n):
        mean += host[i]
    mean /= Float32(n)
    var var_acc = Float32(0.0)
    for i in range(n):
        var d = host[i] - mean
        var_acc += d * d
    var_acc /= Float32(n)
    return sqrt(var_acc)


def main() raises:
    var ctx = DeviceContext()

    var cap_len = 16
    var h_tok = 16
    var w_tok = 16
    var img_len = h_tok * w_tok           # 256
    var seq_len = cap_len + img_len       # 272

    var tables = build_boogu_rope_tables(cap_len, h_tok, w_tok, ctx)

    var cs = tables[0].shape()
    var ss = tables[1].shape()
    print("cos shape:", cs[0], cs[1], "(expect", seq_len, "60)")
    print("sin shape:", ss[0], ss[1], "(expect", seq_len, "60)")

    var cos_h = tables[0].to_host(ctx)
    var sin_h = tables[1].to_host(ctx)
    print("cos std:", _std(cos_h))
    print("sin std:", _std(sin_h))

    # A few representative values (token-major rows; 60 cols/row).
    # Row 0 = caption token t=0 -> pos (0,0,0) -> all angles 0 -> cos=1, sin=0.
    print("row0 col0 cos/sin:", cos_h[0], sin_h[0], "(expect 1.0 0.0)")
    print("row0 col59 cos/sin:", cos_h[59], sin_h[59], "(expect 1.0 0.0)")
    # Row 1 = caption token t=1 -> pos (1,1,1).
    print("row1 col0 cos/sin:", cos_h[60 + 0], sin_h[60 + 0])
    # First image row = row cap_len(=16) -> img token k=0 -> pos (16,0,0).
    var img0 = cap_len * 60
    print("img row0 (k=0) col0 cos/sin:", cos_h[img0 + 0], sin_h[img0 + 0])
    # axis1 block starts at col 20; img token k=0 has h=0 => angle 0 => cos=1,sin=0.
    print("img row0 (k=0) col20 cos/sin:", cos_h[img0 + 20], sin_h[img0 + 20])
    # img token k=1 -> h=0, w=1 -> axis2 (col 40) nonzero.
    var img1 = (cap_len + 1) * 60
    print("img row1 (k=1) col40 cos/sin:", cos_h[img1 + 40], sin_h[img1 + 40])

    print("boogu_c2_rope_probe OK")
