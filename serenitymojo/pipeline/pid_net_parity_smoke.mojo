# pipeline/pid_net_parity_smoke.mojo — FULL PidNet forward parity vs PyTorch.
#
# Loads the REAL converted SD3 res2k PiD checkpoint (F32), runs the assembled
# Mojo PidNet forward (models/pid/pid_net.mojo) on the seeded reference input
# (SD3-style lq_latent [.,16,.,.] + pixel-noise init + caption embeds), and
# GATES the net velocity output vs the Python PiD reference produced by
# parity/gen_pid_net_reference.py (same checkpoint + same input).
#
# Grid (small, to make the 16k-token attention tractable while exercising the
# whole net): B=1, H=W=64 -> pH=pW=4 -> L=16 patch tokens, Ltxt=8, zH=zW=2.
#
# Gate: net_out cos>=0.999 vs golden. Also reports x0 = x - t*v cos.
# F32 throughout (oracle is F32) to isolate op correctness.
#
# Run: pixi run mojo run -I . serenitymojo/pipeline/pid_net_parity_smoke.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.ops.tensor_algebra import reshape, slice
from serenitymojo.models.pid.pid_net import pid_net_forward
from serenitymojo.sampling.pid_distill import velocity_to_x0


comptime F32 = STDtype.F32
comptime CKPT = "/home/alex/.serenity/models/pid/checkpoints/PiD_res2k_sr4x_official_sd3_distill_4step/model_ema_bf16.safetensors"
comptime REF = "/home/alex/mojodiffusion/serenitymojo/models/pid/parity/pid_net_ref.safetensors"


def _ld(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var tv = st.tensor_view(name)
    return Tensor.from_view_as_f32(tv, ctx)


def _cos(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> ParityResult:
    var h = ParityHarness()
    return h.compare(a, b.to_host(ctx), ctx)


def main() raises:
    var ctx = DeviceContext()

    # Grid (must match the reference generator).
    comptime B = 1
    comptime H = 64
    comptime W = 64
    comptime PH = 4
    comptime PW = 4
    comptime L = 16
    comptime LTXT = 8
    comptime ZH = 2
    comptime ZW = 2

    print("=== PiD FULL net parity — F32 vs PyTorch PiD refs (SD3 res2k) ===")
    print("  grid: B=", B, " H=W=", H, " pH=pW=", PH, " L=", L, " Ltxt=", LTXT, " zH=zW=", ZH)

    var ckpt = ShardedSafeTensors.open(String(CKPT))
    var refs = ShardedSafeTensors.open(String(REF))
    print("  ckpt tensors:", ckpt.num_tensors(), " refs tensors:", refs.num_tensors())

    # ── inputs from the reference dump ───────────────────────────────────────
    var x = _ld(refs, "x", ctx)                  # [B,3,H,W]
    var t = _ld(refs, "t_scaled", ctx)           # [B]
    var y = _ld(refs, "y", ctx)                  # [B,LTXT,2304]
    var lq_latent = _ld(refs, "lq_latent", ctx)  # [B,16,ZH,ZW]
    var pix_pos = _ld(refs, "pix_pos", ctx)      # [H*W,16]
    var img_cos = _ld(refs, "img_cos", ctx)      # [L,32]
    var img_sin = _ld(refs, "img_sin", ctx)
    var txt_cos = _ld(refs, "txt_cos", ctx)      # [LTXT,32]
    var txt_sin = _ld(refs, "txt_sin", ctx)
    var pix_cos = _ld(refs, "pix_cos", ctx)      # [L,36]  (pit_block recomputes internally; passed for completeness)
    var pix_sin = _ld(refs, "pix_sin", ctx)

    # ── run the full net ─────────────────────────────────────────────────────
    var out = pid_net_forward[B, H, W, PH, PW, L, LTXT, ZH, ZW](
        ckpt, x, t, y, lq_latent, Float32(0.0),
        pix_pos, img_cos, img_sin, txt_cos, txt_sin, pix_cos, pix_sin, ctx
    )

    # ── gate net velocity output ─────────────────────────────────────────────
    var golden = _ld(refs, "net_out", ctx)
    var r = _cos(out, golden, ctx)
    var tag = "PASS" if (r.cos >= 0.999) else "FAIL"
    print("net_out velocity   cos=", r.cos, " max_abs=", r.max_abs, " n=", r.n, " [", tag, "]")

    # ── gate the velocity->x0 conversion (sampler step from t=0.999) ─────────
    var x0 = velocity_to_x0(x, out, Float32(0.999), ctx)
    var golden_x0 = _ld(refs, "x0", ctx)
    var rx = _cos(x0, golden_x0, ctx)
    var tagx = "PASS" if (rx.cos >= 0.999) else "FAIL"
    print("x0 = x - t*v       cos=", rx.cos, " max_abs=", rx.max_abs, " n=", rx.n, " [", tagx, "]")

    if r.cos >= 0.999 and rx.cos >= 0.999:
        print("============================================================")
        print("FULL-NET PARITY PASS (cos>=0.999): velocity + x0")
    else:
        print("============================================================")
        print("FULL-NET PARITY BELOW THRESHOLD — run the per-block ladder.")
        raise Error("pid_net parity: cos below 0.999")
