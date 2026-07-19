# boogu_c4_block_probe.mojo — compile+run probe for Boogu-Image C4 double-stream
# transformer block (BooguImageDoubleStreamTransformerBlock, modulation=True).
#
# Loads double_stream_layers.0 from the REAL transformer dir, feeds a synthetic
# img [1,256,3360] + instruct [1,16,3360] + temb [1,1024], runs
# BooguDoubleStreamBlock.forward[16,256] (which builds the joint 272-row RoPE and
# the combined-img 256-row RoPE internally from cap_len=16, h_tok=w_tok=16), and
# prints both output shapes + stds. The orchestrator owns parity vs the torch
# oracle (boogu_c4_oracle.py); this probe only proves the Mojo C4 block COMPILES
# and EXECUTES (exit 0).
#
# Run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg \
#     && pixi run mojo run -I . serenitymojo/models/dit/parity/boogu_c4_block_probe.mojo

from std.math import sqrt
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.dit.boogu_dit import BooguDoubleStreamBlock


comptime TF_DIR = "/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base/transformer"
comptime L_INSTRUCT = 16
comptime L_IMG = 256
comptime H_TOK = 16
comptime W_TOK = 16                      # h_tok*w_tok = 256 = L_IMG
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

    # ── load double_stream_layers.0 (modulation=True) ───────────────────────
    print("[c4-probe] loading double_stream_layers.0 from", TF_DIR)
    var st = ShardedSafeTensors.open(TF_DIR)
    var block = BooguDoubleStreamBlock.load(st, "double_stream_layers.0", ctx)
    print("[c4-probe] block loaded (modulation=True)")

    # ── synthetic inputs (deterministic; orchestrator gates byte-identical) ──
    var img_vals = List[Float32]()
    for i in range(L_IMG * HIDDEN):
        var x = Float32((i * 2654435761) % 2000) / Float32(1000.0) - Float32(1.0)
        img_vals.append(x)
    var img = Tensor.from_host(img_vals, [1, L_IMG, HIDDEN], STDtype.BF16, ctx)

    var ins_vals = List[Float32]()
    for i in range(L_INSTRUCT * HIDDEN):
        var x = Float32((i * 2246822519) % 2000) / Float32(1000.0) - Float32(1.0)
        ins_vals.append(x)
    var instruct = Tensor.from_host(
        ins_vals, [1, L_INSTRUCT, HIDDEN], STDtype.BF16, ctx
    )

    var temb_vals = List[Float32]()
    for i in range(TEMB_DIM):
        var x = Float32((i * 40503) % 2000) / Float32(1000.0) - Float32(1.0)
        temb_vals.append(x)
    var temb = Tensor.from_host(temb_vals, [1, TEMB_DIM], STDtype.BF16, ctx)

    print(
        "[c4-probe] img std:",
        _std(img.to_host(ctx)),
        " instruct std:",
        _std(instruct.to_host(ctx)),
        " temb std:",
        _std(temb.to_host(ctx)),
    )

    # ── run forward (builds joint 272 + combined-img 256 rope internally) ────
    var outs = block.forward[L_INSTRUCT, L_IMG](
        img, instruct, temb, H_TOK, W_TOK, ctx
    )
    var ois = outs[0].shape()
    var ons = outs[1].shape()
    print("img_out shape:", ois[0], ois[1], ois[2], "(expect 1", L_IMG, HIDDEN, ")")
    print("img_out std:", _std(outs[0].to_host(ctx)))
    print(
        "instruct_out shape:",
        ons[0],
        ons[1],
        ons[2],
        "(expect 1",
        L_INSTRUCT,
        HIDDEN,
        ")",
    )
    print("instruct_out std:", _std(outs[1].to_host(ctx)))
    print("[c4-probe] OK")
