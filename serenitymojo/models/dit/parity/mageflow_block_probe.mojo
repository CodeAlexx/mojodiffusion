# mageflow_block_probe.mojo — parity gate for the MageFlow DiT double block.
#
# Loads transformer_blocks.0 from the REAL Mage-Flow-Edit-Turbo transformer,
# loads byte-identical inputs (img/txt/temb) from the oracle dump, builds the
# MageFlow image-only msrope tables (text NOT roped), runs MageFlowDoubleBlock,
# and gates the img + txt outputs vs the oracle at cos >= 0.999.
#
# Run:
#   cd <worktree> && rm -f serenitymojo.mojopkg \
#     && pixi run mojo run -I . serenitymojo/models/dit/parity/mageflow_block_probe.mojo
#   (add -Xlinker -L/usr/lib/x86_64-linux-gnu -Xlinker -lcuda if the linker
#    cannot find libcuda)

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.models.dit.mageflow_dit import (
    MageFlowDoubleBlock,
    build_mageflow_rope_tables,
)

comptime CKPT = "/home/alex/.serenity/models/checkpoints/Mage-Flow-Edit-Turbo/transformer"
comptime DUMP = "/home/alex/mojodiffusion/.claude/worktrees/agent-a515ee84480d0f599/serenitymojo/models/dit/parity/mageflow_dumps/mageflow_block0.safetensors"

comptime FRAME = 1
comptime H_TOK = 4
comptime W_TOK = 4
comptime N_IMG = 16   # FRAME*H_TOK*W_TOK
comptime N_TXT = 8
comptime S = 24       # N_IMG + N_TXT
comptime HEADS = 24
comptime HIDDEN = 3072


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
        Tensor.from_view(fx.tensor_view("mf_in_img"), ctx), STDtype.BF16, ctx
    )  # [16,3072]
    var txt2d = cast_tensor(
        Tensor.from_view(fx.tensor_view("mf_in_txt"), ctx), STDtype.BF16, ctx
    )  # [8,3072]
    var temb = cast_tensor(
        Tensor.from_view(fx.tensor_view("mf_in_temb"), ctx), STDtype.BF16, ctx
    )  # [1,3072]

    var img = reshape(img2d, _shape3(1, N_IMG, HIDDEN), ctx)
    var txt = reshape(txt2d, _shape3(1, N_TXT, HIDDEN), ctx)
    print("[mf-probe] inputs loaded: img", img.shape()[1], "txt", txt.shape()[1])

    # ── MageFlow rope tables (image msrope; text identity) ───────────────────
    var rope = build_mageflow_rope_tables(
        FRAME, H_TOK, W_TOK, N_TXT, HEADS, Float64(10000.0), STDtype.F32, ctx
    )
    var rc = rope[0].shape()
    print("[mf-probe] rope cos shape:", rc[0], rc[1], "(expect", S * HEADS, "64)")

    # ── load the real block-0 weights + run ──────────────────────────────────
    print("[mf-probe] loading transformer_blocks.0 from", CKPT)
    var block = MageFlowDoubleBlock.load(CKPT, ctx)
    print("[mf-probe] block loaded; running forward")
    block.forward[N_IMG, N_TXT, S](0, img, txt, temb, rope[0], rope[1], ctx)

    # ── gate vs oracle ───────────────────────────────────────────────────────
    var ref_img = Tensor.from_view(fx.tensor_view("mf_out_img"), ctx).to_host(ctx)
    var ref_txt = Tensor.from_view(fx.tensor_view("mf_out_txt"), ctx).to_host(ctx)
    var h = ParityHarness(0.999)
    var r_img = h.compare(img, ref_img, ctx)
    var r_txt = h.compare(txt, ref_txt, ctx)
    print("[mf-probe] IMG-out", r_img)
    print("[mf-probe] TXT-out", r_txt)
    if r_img.passed and r_txt.passed:
        print("[mf-probe] GATE PASS (img+txt cos >= 0.999)")
    else:
        print("[mf-probe] GATE FAIL")
