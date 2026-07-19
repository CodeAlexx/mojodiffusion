# boogu_c3_block_probe.mojo — compile+run probe for Boogu-Image C3 single-stream
# transformer block (BooguImageTransformerBlock, modulation=True).
#
# Loads single_stream_layers.0 from the REAL transformer dir, builds the joint
# 3-axis RoPE tables via build_boogu_rope_tables(16,16,16) (cap_len=16, h_tok=16,
# w_tok=16 => seq=272), feeds a synthetic hidden [1,272,3360] + temb [1,1024],
# runs BooguBlock.forward, and prints the output shape + std. The orchestrator
# owns parity vs the torch oracle (boogu_c3_oracle.py); this probe only proves
# the Mojo C3 block COMPILES and EXECUTES (exit 0).
#
# Run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg \
#     && pixi run mojo run -I . serenitymojo/models/dit/parity/boogu_c3_block_probe.mojo

from std.math import sqrt
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.dit.boogu_dit import build_boogu_rope_tables, BooguBlock


comptime TF_DIR = "/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base/transformer"
comptime CAP_LEN = 16
comptime H_TOK = 16
comptime W_TOK = 16
comptime SEQ = 272                       # cap_len + h_tok*w_tok = 16 + 256
comptime HIDDEN = 3360
comptime TEMB_DIM = 1024


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

    # ── joint RoPE tables (same case as the C2 gate) ─────────────────────────
    var tables = build_boogu_rope_tables(CAP_LEN, H_TOK, W_TOK, ctx)
    var cos = tables[0].clone(ctx)
    var sin = tables[1].clone(ctx)
    var cs = cos.shape()
    print("rope cos shape:", cs[0], cs[1], "(expect", SEQ, "60)")

    # ── load single_stream_layers.0 (modulation=True) ───────────────────────
    print("[c3-probe] loading single_stream_layers.0 from", TF_DIR)
    var st = ShardedSafeTensors.open(TF_DIR)
    var block = BooguBlock.load(st, "single_stream_layers.0", True, ctx)
    print("[c3-probe] block loaded (modulation=True)")

    # ── synthetic inputs (deterministic; orchestrator gates byte-identical) ──
    var hidden_vals = List[Float32]()
    for i in range(SEQ * HIDDEN):
        # bounded deterministic pseudo-noise in ~[-1,1].
        var x = Float32((i * 2654435761) % 2000) / Float32(1000.0) - Float32(1.0)
        hidden_vals.append(x)
    var hidden = Tensor.from_host(hidden_vals, [1, SEQ, HIDDEN], STDtype.BF16, ctx)

    var temb_vals = List[Float32]()
    for i in range(TEMB_DIM):
        var x = Float32((i * 40503) % 2000) / Float32(1000.0) - Float32(1.0)
        temb_vals.append(x)
    var temb = Tensor.from_host(temb_vals, [1, TEMB_DIM], STDtype.BF16, ctx)

    print(
        "[c3-probe] hidden std:",
        _std(hidden.to_host(ctx)),
        " temb std:",
        _std(temb.to_host(ctx)),
    )

    # ── run forward ──────────────────────────────────────────────────────────
    var out = block.forward[SEQ](hidden, temb, cos, sin, ctx)
    var os = out.shape()
    print("out shape:", os[0], os[1], os[2], "(expect 1", SEQ, HIDDEN, ")")
    print("out std:", _std(out.to_host(ctx)))
    print("[c3-probe] OK")
