# autograd_v2/tests/ltx2_pingpong_dhidden_parity.mojo — rung-3 ping-pong BIT gate.
#
# Unit (c) load-bearing numeric change: apply_ltx2v_slab_dev's mandatory
# copy-d_hidden-out-of-the-slab-before-rewind now lands in a caller-provided
# FIXED buffer (d_hidden_dst) instead of a fresh `.clone` alloc — the ping-pong
# that retires the last steady-state clone (W4b) and is the capture prerequisite
# (C8: replay reads stable addresses). This gate proves, same-process on REAL
# block-0 weights + synthetic NONZERO LoRA, that the FIXED-dst d_hidden is
# BYTE-IDENTICAL to the .clone-path d_hidden (only the copy destination differs,
# same bg.d_hidden bytes), AND that the returned d_hidden actually views the
# provided buffer (address == buffer base). ltx2 is MATH-MODE ⇒ a hard BIT gate.
#
#   rm -f serenitymojo.mojopkg && pixi run mojo build -O2 -I . \
#     -Xlinker -lm -Xlinker -lcuda -Xlinker -L.pixi/envs/default/lib \
#     -Xlinker -lsqlite3 -Xlinker -Lserenitymojo/ops/cshim/lib \
#     -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/autograd_v2/tests/ltx2_pingpong_dhidden_parity.mojo -o /tmp/ltx2_pingpong_dhidden_parity
#   env LD_LIBRARY_PATH=.pixi/envs/default/lib:serenitymojo/ops/cshim/lib:\
#     $HOME/.local/lib/python3.12/site-packages/nvidia/cudnn/lib \
#     /tmp/ltx2_pingpong_dhidden_parity

from std.math import sin, cos
from std.memory import ArcPointer
from std.collections import Optional
from std.gpu.host import DeviceContext, DeviceBuffer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.ltx2_dit import LTX2Config, LTX2AVBlockWeights
from serenitymojo.models.ltx2.ltx2_video_stack import (
    LTX2VideoBlockSource, video_lora_names, _attach_block_lora,
)
from serenitymojo.autograd_v2.node import TArc
from serenitymojo.autograd_v2.ltx2_video_block_graph import (
    ltx2_video_block_graph_backward_slab_dev,
)
from serenitymojo.autograd_v2.step_slab import StepSlab

comptime S_V = 256
comptime N_TXT = 1024
comptime VD = 4096
comptime H = 32
comptime DH = 128
comptime RANK = 8
comptime CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors"
comptime EPS = Float32(1.0e-6)
comptime SCALE = Float32(0.5)
comptime SLAB_BYTES = 2 * 1024 * 1024 * 1024


def _fill(n: Int, phase: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        var fi = Float32(i)
        out.append(Float32(0.03) * (sin(Float32(0.7) * fi + phase)
                   + Float32(0.4) * cos(Float32(1.3) * fi + Float32(0.2))))
    return out^


def _t(shape: List[Int], phase: Float32, ctx: DeviceContext) raises -> Tensor:
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    return Tensor.from_host(_fill(n, phase), shape.copy(), STDtype.BF16, ctx)


def _sh(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); return s^


def _sh2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^


def _attach_preset(mut w: LTX2AVBlockWeights, preset: Int, ctx: DeviceContext) raises:
    var names = video_lora_names(preset)
    var lora_a = List[ArcPointer[Tensor]]()
    var lora_b = List[ArcPointer[Tensor]]()
    for s in range(len(names)):
        var ws = w.weight_shape(names[s])   # [out, in]
        var out_f = ws[0]
        var in_f = ws[1]
        lora_a.append(ArcPointer[Tensor](_t(_sh2(RANK, in_f), Float32(s) * Float32(0.3) + Float32(1.0), ctx)))
        lora_b.append(ArcPointer[Tensor](_t(_sh2(out_f, RANK), Float32(s) * Float32(0.3) + Float32(2.0), ctx)))
    _attach_block_lora(w, 0, names, lora_a, lora_b, SCALE)


def _cmp(a: List[Float32], b: List[Float32]) -> Tuple[Int, Bool]:
    if len(a) != len(b):
        return (-1, False)
    var nm = 0
    var nz = False
    for i in range(len(a)):
        if a[i] != b[i]:
            nm += 1
        if a[i] != Float32(0.0):
            nz = True
    return (nm, nz)


def main() raises:
    print("=== LTX2 rung-3 ping-pong d_hidden BIT gate (fixed-dst == .clone) ===")
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()
    var src = LTX2VideoBlockSource.open(CKPT, cfg, False)
    var w = src.get_block(0, ctx)
    _attach_preset(w, 0, ctx)

    # synthetic-but-real-shaped inputs (bf16, the block dtype).
    var hidden = _t(_sh(1, S_V, VD), Float32(0.11), ctx)          # block_input
    var enc = _t(_sh(1, N_TXT, VD), Float32(0.23), ctx)
    var v_temb = _t(_sh(1, S_V, 9 * VD), Float32(0.31), ctx)
    var v_prompt_ts = _t(_sh(1, N_TXT, 2 * VD), Float32(0.41), ctx)
    var v_cos = _t(_sh2(S_V * H, DH // 2), Float32(0.51), ctx)
    var v_sin = _t(_sh2(S_V * H, DH // 2), Float32(0.61), ctx)
    var d_video = _t(_sh(1, S_V, VD), Float32(0.71), ctx)          # d_x seed

    var slab = StepSlab(ctx, SLAB_BYTES)

    # ARM 1: default .clone path (d_hidden_dst = None).
    var bg_none = ltx2_video_block_graph_backward_slab_dev[S_V, N_TXT](
        TArc(Tensor(d_video.buf.copy(), d_video.shape(), d_video.dtype())),
        w, TArc(Tensor(hidden.buf.copy(), hidden.shape(), hidden.dtype())),
        enc, v_temb, v_prompt_ts, v_cos, v_sin, EPS, ctx, slab,
    )
    var dh_none = bg_none.d_hidden.to_host(ctx)

    # ARM 2: FIXED-buffer ping-pong dst.
    var fixed_buf = ctx.enqueue_create_buffer[DType.uint8](d_video.nbytes())
    var fixed_addr = Int(fixed_buf.unsafe_ptr())
    var bg_dst = ltx2_video_block_graph_backward_slab_dev[S_V, N_TXT](
        TArc(Tensor(d_video.buf.copy(), d_video.shape(), d_video.dtype())),
        w, TArc(Tensor(hidden.buf.copy(), hidden.shape(), hidden.dtype())),
        enc, v_temb, v_prompt_ts, v_cos, v_sin, EPS, ctx, slab,
        d_hidden_dst=Optional[DeviceBuffer[DType.uint8]](fixed_buf.copy()),
    )
    var dh_addr = Int(bg_dst.d_hidden.buf.unsafe_ptr())
    var dh_dst = bg_dst.d_hidden.to_host(ctx)

    var allok = True

    # (1) byte-identity.
    var r = _cmp(dh_none, dh_dst)
    var nm = r[0]
    var nz = r[1]
    if nm != 0 or not nz:
        allok = False
    var verdict = "PASS" if (nm == 0 and nz) else "FAIL"
    print("  ", verdict, "d_hidden fixed-dst vs .clone  n_mismatch=", nm,
          " nonzero=", nz, " n=", len(dh_none))

    # (2) the returned d_hidden actually views the provided fixed buffer.
    if dh_addr == fixed_addr:
        print("   PASS d_hidden buf address == provided fixed buffer base")
    else:
        print("   FAIL d_hidden addr", dh_addr, "!= fixed buffer base", fixed_addr)
        allok = False

    # ── ARM 2: 2-block PING-PONG indexing. Block A→B through TWO alternating
    #    buffers (A reads pp0 writes pp1; B reads pp1 writes pp0). Proves block B
    #    reads block A's ACTUAL output from the OTHER buffer (== clone-chained
    #    reference, not stale/aliased), and that A and B read DISTINCT buffers
    #    (FAIL if both blocks read the same buffer — the ping-pong-not-alternating
    #    bug). ─────────────────────────────────────────────────────────────────
    var hidden_b = _t(_sh(1, S_V, VD), Float32(0.37), ctx)   # block B input (differs from A)

    # reference: clone-chain A→B (dst=None both).
    var bgA = ltx2_video_block_graph_backward_slab_dev[S_V, N_TXT](
        TArc(Tensor(d_video.buf.copy(), d_video.shape(), d_video.dtype())),
        w, TArc(Tensor(hidden.buf.copy(), hidden.shape(), hidden.dtype())),
        enc, v_temb, v_prompt_ts, v_cos, v_sin, EPS, ctx, slab,
    )
    var bgB_ref = ltx2_video_block_graph_backward_slab_dev[S_V, N_TXT](
        TArc(Tensor(bgA.d_hidden.buf.copy(), bgA.d_hidden.shape(), bgA.d_hidden.dtype())),
        w, TArc(Tensor(hidden_b.buf.copy(), hidden_b.shape(), hidden_b.dtype())),
        enc, v_temb, v_prompt_ts, v_cos, v_sin, EPS, ctx, slab,
    )
    var dhB_ref = bgB_ref.d_hidden.to_host(ctx)

    # ping-pong: pp0/pp1; seed pp0 with d_video.
    var nb = d_video.nbytes()
    var pp0 = ctx.enqueue_create_buffer[DType.uint8](nb)
    var pp1 = ctx.enqueue_create_buffer[DType.uint8](nb)
    var pp0_addr = Int(pp0.unsafe_ptr())
    var pp1_addr = Int(pp1.unsafe_ptr())
    var seed = pp0.create_sub_buffer[DType.uint8](0, nb)
    ctx.enqueue_copy(dst_buf=seed, src_buf=d_video.buf)

    # Block A: read pp0, write pp1.
    var a_read = pp0.create_sub_buffer[DType.uint8](0, nb)
    var a_read_addr = Int(a_read.unsafe_ptr())
    var bgA_pp = ltx2_video_block_graph_backward_slab_dev[S_V, N_TXT](
        TArc(Tensor(a_read^, d_video.shape(), d_video.dtype())),
        w, TArc(Tensor(hidden.buf.copy(), hidden.shape(), hidden.dtype())),
        enc, v_temb, v_prompt_ts, v_cos, v_sin, EPS, ctx, slab,
        d_hidden_dst=Optional[DeviceBuffer[DType.uint8]](pp1.copy()),
    )
    _ = bgA_pp^

    # Block B: read pp1 (= A's write), write pp0.
    var b_read = pp1.create_sub_buffer[DType.uint8](0, nb)
    var b_read_addr = Int(b_read.unsafe_ptr())
    var bgB_pp = ltx2_video_block_graph_backward_slab_dev[S_V, N_TXT](
        TArc(Tensor(b_read^, d_video.shape(), d_video.dtype())),
        w, TArc(Tensor(hidden_b.buf.copy(), hidden_b.shape(), hidden_b.dtype())),
        enc, v_temb, v_prompt_ts, v_cos, v_sin, EPS, ctx, slab,
        d_hidden_dst=Optional[DeviceBuffer[DType.uint8]](pp0.copy()),
    )
    _ = bgB_pp^

    var dhB_pp = Tensor(pp0.create_sub_buffer[DType.uint8](0, nb),
                        d_video.shape(), d_video.dtype()).to_host(ctx)

    # (3) functional: ping-pong B output == clone-chain B output (B read A's REAL
    #     output through pp1). A stale/aliased buffer would diverge here.
    var rB = _cmp(dhB_ref, dhB_pp)
    var nmB = rB[0]
    var nzB = rB[1]
    if nmB != 0 or not nzB:
        allok = False
    var vB = "PASS" if (nmB == 0 and nzB) else "FAIL"
    print("  ", vB, "2-block ping-pong B out == clone-chain  n_mismatch=", nmB,
          " nonzero=", nzB, " n=", len(dhB_ref))

    # (4) indexing: distinct buffers; A and B read DIFFERENT buffers; B reads A's
    #     write buffer (chain). FAIL if both blocks read the same buffer.
    if pp0_addr == pp1_addr:
        print("   FAIL ping-pong buffers alias (pp0 == pp1)")
        allok = False
    elif a_read_addr == b_read_addr:
        print("   FAIL block A and block B read the SAME buffer (ping-pong not alternating)")
        allok = False
    elif b_read_addr != pp1_addr:
        print("   FAIL block B did not read block A's write buffer (chain broken)")
        allok = False
    else:
        print("   PASS ping-pong indexing: A_read", a_read_addr, "!= B_read", b_read_addr,
              "(distinct; B reads A's write buffer)")

    if allok:
        print("GATE ltx2_pingpong_dhidden_parity: ALL PASS",
              "(fixed-dst d_hidden byte-exact vs .clone; views the fixed buffer;",
              "2-block ping-pong == clone-chain with distinct alternating buffers)")
    else:
        print("GATE ltx2_pingpong_dhidden_parity: FAIL")
        raise Error("ltx2_pingpong_dhidden_parity gate FAILED")
