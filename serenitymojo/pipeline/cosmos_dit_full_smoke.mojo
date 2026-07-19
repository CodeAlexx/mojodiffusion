# pipeline/cosmos_dit_full_smoke.mojo — FULL-forward parity gate for cosmos DiT.
# Loads the REAL post-trained checkpoint and runs the COMPLETE forward (LVG concat,
# padding-mask concat, crossattn_proj, patchify, x_embedder, timestep MLP, 28
# blocks, FinalLayer, unpatchify) at a LARGE latent grid (T=2,H=128,W=128 ->
# N=Tp*Hp*Wp = 2*64*64 = 8192) — a token count the old math-mode Dh=128 self-attn
# could NOT fit in 24GB (it materialized [H,N,N] ~= 16*8192*8192*4 = 4.3GB per
# scores buffer). The tiled (online-softmax) self-attn streams K/V so peak
# self-attn memory is O(N*Dh): no OOM.
#
# Deterministic LCG inputs (byte-identical to gen_cosmos_full_oracle.py). That
# Python oracle ran the same forward in torch (real flash attention, bf16 GPU)
# and wrote cosmos_full_fixture.safetensors["expected"] [out_c,T,H,W]. We compare
# the Mojo full-forward output vs that fixture; gate cos >= 0.99 (deep 28-block
# chain). The per-block numeric math is additionally covered by the block-0 gate
# (cos 0.99999605); this gate covers the FULL chain at large S via the tiled SDPA.

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.models.dit.cosmos_predict25_dit import (
    CosmosConfig, CosmosPredict25Dit,
)

comptime CKPT = "/home/alex/.cosmos-predict25/base/post-trained/cosmos_predict25_2b_dit.safetensors"
comptime FIX = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/cosmos_full_fixture.safetensors"

comptime TG = 2
comptime HG = 128
comptime WG = 128
comptime N = 8192          # (2/1)*(128/2)*(128/2) = 2*64*64
comptime TXTRAW = 16       # tiny text len (kv-len after proj)
comptime TXT = 16
comptime H = 16
comptime DH = 128


def main() raises:
    var ctx = DeviceContext()
    var cfg = CosmosConfig.v2_2b_production()
    print("loading real cosmos checkpoint (4.1GB, resident bf16)...")
    var model = CosmosPredict25Dit.load(CKPT, cfg, ctx)
    print("loaded weights:", len(model.weights))

    var x_lat = _rand4(cfg.in_channels, TG, HG, WG, STDtype.BF16, ctx)  # [16,T,H,W]
    var text = _rand2(TXTRAW, cfg.crossattn_proj_in, STDtype.BF16, ctx) # [16,100352]
    var timestep = Float32(700.0)

    print("running FULL forward (28 blocks) at N=", N, " via tiled SDPA...")
    var out = model.forward[TG, HG, WG, N, TXTRAW, TXT, H, DH](
        x_lat, timestep, text, ctx
    )
    var sh = out.shape()
    print("forward out shape:", sh[0], sh[1], sh[2], sh[3])  # [16, T, H, W]

    # magnitude (RMS) of the mojo output
    var host = out.to_host(ctx)
    var ssq = Float32(0.0)
    for i in range(len(host)):
        ssq += host[i] * host[i]
    var mag = (ssq / Float32(len(host))) ** 0.5
    print("mojo full-forward RMS magnitude:", mag, " n=", len(host))

    # ── parity vs the python oracle fixture ──
    var st = ShardedSafeTensors.open(FIX)
    var exp = Tensor.from_view(st.tensor_view("expected"), ctx)  # [out_c,T,H,W] F32
    var ph = ParityHarness(0.99)
    var ref_host = exp.to_host(ctx)
    var res = ph.compare(out, ref_host, ctx)
    print("cosmos FULL-forward parity (N=", N, "):", res)

    var shape_ok = (
        sh[0] == cfg.out_channels and sh[1] == TG
        and sh[2] == HG and sh[3] == WG
    )
    if res.passed and shape_ok:
        print("FULL-FORWARD GATE: PASS")
    else:
        print("FULL-FORWARD GATE: FAIL")


def _rand4(C: Int, F: Int, Hh: Int, W: Int, dt: STDtype, ctx: DeviceContext) raises -> Tensor:
    var n = C * F * Hh * W
    var h = List[Float32]()
    var seed = 99
    for _ in range(n):
        seed = (seed * 1103515245 + 12345) % 2147483648
        h.append((Float32(seed) / 2147483648.0 - 0.5) * 0.2)
    var shp = List[Int]()
    shp.append(C)
    shp.append(F)
    shp.append(Hh)
    shp.append(W)
    return Tensor.from_host(h^, shp^, dt, ctx)


def _rand2(R: Int, C: Int, dt: STDtype, ctx: DeviceContext) raises -> Tensor:
    var n = R * C
    var h = List[Float32]()
    var seed = 7
    for _ in range(n):
        seed = (seed * 1103515245 + 12345) % 2147483648
        h.append((Float32(seed) / 2147483648.0 - 0.5) * 0.05)
    var shp = List[Int]()
    shp.append(R)
    shp.append(C)
    return Tensor.from_host(h^, shp^, dt, ctx)
