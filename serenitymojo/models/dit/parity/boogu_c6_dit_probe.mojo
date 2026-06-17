# boogu_c6_dit_probe.mojo — compile+run probe for the FULL Boogu-Image DiT
# forward (Chunk C6, T2I no-ref, batch=1).
#
# Loads the WHOLE transformer (embedders + 2 context_refiner + 2 noise_refiner +
# 8 double_stream + 32 single_stream + norm_out) RESIDENT from the REAL
# transformer dir, feeds a synthetic latent[1,16,32,32] + timestep[1] +
# instruction_feats[1,16,4096], runs BooguDiT.forward[16,16,16] (cap_len=16,
# h_tok=w_tok=16 => img_len=256, joint=272), and prints the velocity shape
# [1,16,32,32] + std. The orchestrator owns numeric parity vs the torch oracle
# (boogu_c6_oracle.py); this probe only proves the Mojo C6 wiring COMPILES and
# EXECUTES (exit 0).
#
# Run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg \
#     && pixi run mojo run -I . serenitymojo/models/dit/parity/boogu_c6_dit_probe.mojo

from std.math import sqrt
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.boogu_dit import BooguDiT


comptime TF_DIR = "/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base/transformer"
comptime CAP_LEN = 16
comptime H_TOK = 16
comptime W_TOK = 16                      # h_tok*w_tok = 256 = img_len
comptime IN_CH = 16
comptime H_LAT = 32                      # h_tok * patch_size
comptime W_LAT = 32
comptime INSTR_DIM = 4096


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

    # ── load the full DiT RESIDENT ───────────────────────────────────────────
    print("[c6-probe] loading full BooguDiT from", TF_DIR)
    var dit = BooguDiT.load(TF_DIR, ctx)
    print("[c6-probe] BooguDiT loaded (resident)")

    # ── synthetic inputs (deterministic; orchestrator gates byte-identical) ───
    var latent_vals = List[Float32]()
    for i in range(IN_CH * H_LAT * W_LAT):
        var x = Float32((i * 2654435761) % 2000) / Float32(1000.0) - Float32(1.0)
        latent_vals.append(x)
    var latent = Tensor.from_host(
        latent_vals, [1, IN_CH, H_LAT, W_LAT], STDtype.BF16, ctx
    )

    var timestep = Tensor.from_host([Float32(0.25)], [1], STDtype.F32, ctx)

    var instr_vals = List[Float32]()
    for i in range(CAP_LEN * INSTR_DIM):
        var x = Float32((i * 2246822519) % 2000) / Float32(1000.0) - Float32(1.0)
        instr_vals.append(x)
    var instr = Tensor.from_host(
        instr_vals, [1, CAP_LEN, INSTR_DIM], STDtype.BF16, ctx
    )

    print(
        "[c6-probe] latent std:",
        _std(latent.to_host(ctx)),
        " instr std:",
        _std(instr.to_host(ctx)),
    )

    # ── run the full forward ─────────────────────────────────────────────────
    var vel = dit.forward[CAP_LEN, H_TOK, W_TOK](latent, timestep, instr, ctx)
    var vs = vel.shape()
    print(
        "velocity shape:",
        vs[0], vs[1], vs[2], vs[3],
        "(expect 1", IN_CH, H_LAT, W_LAT, ")",
    )
    print("velocity std:", _std(vel.to_host(ctx)))
    print("[c6-probe] OK")
