# mageflow_dit_probe.mojo — parity gate for the FULL MageFlow DiT forward.
#
# Loads the REAL Mage-Flow-Edit-Turbo transformer (all 12 double blocks),
# loads byte-identical inputs (patchified latent + raw Qwen3-VL context) from
# the oracle dump, runs MageFlowDiT.forward, and gates the output velocity vs
# the oracle at cos >= 0.999. Taps (post img_in / txt_in / temb / block0)
# localize any divergence across the 12-block stack.
#
# Run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg \
#     && pixi run mojo run -I . -Xlinker -L/usr/lib/x86_64-linux-gnu \
#        -Xlinker -lcuda serenitymojo/models/dit/parity/mageflow_dit_probe.mojo
#   (if the linker fails on flame_cudnn_sdpa, add:
#      -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa
#    and export LD_LIBRARY_PATH to include serenitymojo/ops/cshim/lib + the
#    nvidia/cudnn wheel lib dir.)

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.models.dit.mageflow_dit import (
    MageFlowDiT,
    build_mageflow_rope_tables,
)

comptime CKPT = "/home/alex/.serenity/models/checkpoints/Mage-Flow-Edit-Turbo/transformer"
comptime DUMP = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/mageflow_dumps/mageflow_dit.safetensors"

comptime FRAME = 1
comptime H_TOK = 4
comptime W_TOK = 4
comptime N_IMG = 16   # FRAME*H_TOK*W_TOK
comptime N_TXT = 8
comptime S = 24       # N_IMG + N_TXT
comptime HEADS = 24
comptime HIDDEN = 3072
comptime CIN = 128
comptime CTX_DIM = 2560
comptime TIMESTEP = Float32(0.75)  # must match the oracle's TIMESTEP


def _shape3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    return s^


def main() raises:
    var ctx = DeviceContext()

    # ── load byte-identical inputs from the oracle dump ──────────────────────
    var fx = ShardedSafeTensors.open(DUMP)
    var img2d = cast_tensor(
        Tensor.from_view(fx.tensor_view("in_img"), ctx), STDtype.BF16, ctx
    )  # [16,128]
    var txt2d = cast_tensor(
        Tensor.from_view(fx.tensor_view("in_txt"), ctx), STDtype.BF16, ctx
    )  # [8,2560]
    var hidden = reshape(img2d, _shape3(1, N_IMG, CIN), ctx)     # [1,16,128]
    var enc = reshape(txt2d, _shape3(1, N_TXT, CTX_DIM), ctx)    # [1,8,2560]
    print("[mf-dit-probe] inputs loaded: img", hidden.shape()[1], "txt", enc.shape()[1])

    # ── load the full DiT (all 397 keys, resident) ──
    print("[mf-dit-probe] loading MageFlow DiT from", CKPT)
    var dit = MageFlowDiT.load(CKPT, ctx)
    print("[mf-dit-probe] DiT loaded")

    var h = ParityHarness(0.999)

    # ── taps: img_in / txt_in / temb ──
    var img0 = dit._img_in(hidden, ctx)          # [1,16,3072]
    var txt0 = dit._txt_in(enc, ctx)             # [1,8,3072]
    var temb = dit._temb(TIMESTEP, ctx)          # [1,3072]

    var ref_img_in = Tensor.from_view(fx.tensor_view("tap_img_in"), ctx).to_host(ctx)
    var ref_txt_in = Tensor.from_view(fx.tensor_view("tap_txt_in"), ctx).to_host(ctx)
    var ref_temb = Tensor.from_view(fx.tensor_view("tap_temb"), ctx).to_host(ctx)
    var r_img_in = h.compare(img0, ref_img_in, ctx)
    var r_txt_in = h.compare(txt0, ref_txt_in, ctx)
    var r_temb = h.compare(temb, ref_temb, ctx)
    print("[mf-dit-probe] tap img_in ", r_img_in)
    print("[mf-dit-probe] tap txt_in ", r_txt_in)
    print("[mf-dit-probe] tap temb   ", r_temb)

    # ── block-stack trajectory: run 12 blocks on clones, tap at 0/5/11 ──
    var rope = build_mageflow_rope_tables(
        FRAME, H_TOK, W_TOK, N_TXT, HEADS, Float64(10000.0), STDtype.F32, ctx
    )
    var imgb = img0.clone(ctx)
    var txtb = txt0.clone(ctx)
    var b0_ok = True
    var b5_ok = True
    var b11_ok = True
    for i in range(12):
        dit.inner._block_forward[N_IMG, N_TXT, S](
            i, imgb, txtb, temb, rope[0], rope[1], N_TXT, ctx
        )
        if i == 0 or i == 5 or i == 11:
            var tag = String("tap_block") + String(i)
            var ri = h.compare(
                imgb, Tensor.from_view(fx.tensor_view(tag + "_img"), ctx).to_host(ctx), ctx
            )
            var rt = h.compare(
                txtb, Tensor.from_view(fx.tensor_view(tag + "_txt"), ctx).to_host(ctx), ctx
            )
            print("[mf-dit-probe] tap block", i, "img", ri)
            print("[mf-dit-probe] tap block", i, "txt", rt)
            if i == 0:
                b0_ok = ri.passed and rt.passed
            elif i == 5:
                b5_ok = ri.passed and rt.passed
            else:
                b11_ok = ri.passed and rt.passed

    # ── full forward -> velocity (the primary gate) ──
    var vel = dit.forward[N_IMG, N_TXT, S](
        hidden, enc, TIMESTEP, FRAME, H_TOK, W_TOK, ctx
    )  # [1,16,128]
    var ref_vel = Tensor.from_view(fx.tensor_view("velocity"), ctx).to_host(ctx)
    var r_vel = h.compare(vel, ref_vel, ctx)
    print("[mf-dit-probe] VELOCITY   ", r_vel)

    if (
        r_img_in.passed and r_txt_in.passed and r_temb.passed
        and b0_ok and b5_ok and b11_ok and r_vel.passed
    ):
        print("[mf-dit-probe] GATE PASS (velocity + all taps cos >= 0.999)")
    else:
        print("[mf-dit-probe] GATE FAIL (velocity cos =", r_vel.cos, ")")
