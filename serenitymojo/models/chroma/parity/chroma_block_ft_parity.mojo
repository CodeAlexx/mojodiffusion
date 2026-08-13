# serenitymojo/models/chroma/parity/chroma_block_ft_parity.mojo
#
# PARITY GATE for the Chroma (= Flux) DOUBLE + SINGLE block FULL-FINETUNE
# device backwards (chroma_double_block_ft_backward_dev /
# chroma_single_block_ft_backward_dev, re-exported from
# models/flux/flux_block_ft.mojo — the flux-shared implementation, so a PASS
# here gates the flux FT block math too at the same real dims D=3072/H=24/
# Dh=128). TWO backward runs per block kind:
#   1. SURFACE_V2=False (the DEFAULT — flux's CURRENT 10-arm behavior): d_x +
#      the matmul dW vs torch, cos >= 0.999. This run IS the flux regression
#      (FULL_SURFACE_PLAN Phase B row 3 keeps flux's v1 arm untouched).
#   2. SURFACE_V2=True (chroma's v2 FULL surface): the same v1 captures
#      (unchanged bar) PLUS the NEW arms — 8+2 linear bias d_b (d_y row-sum)
#      and 4+2 q/k rms-scale d_g (full rms_norm_backward) — at the Phase B
#      campaign bar cos >= 0.9999 WHERE THE CHAIN PERMITS. Measured plan
#      delta (2026-07-08): the arms fed by a grad tensor that ALREADY gates a
#      v1 matmul dW at this block's chain class CANNOT exceed that class —
#      d_bmlp0 tracks d_wmlp0 (~0.9998x), s_d_b1 EQUALS s_d_w1 (0.99979, the
#      identical d_fused row-sum vs contraction), single d_g sits in the
#      s_d_w1/w2 class. Those chain-class arms gate at cos >= 0.9997 — the
#      chroma FT block's ORIGINAL torch-gated class (chroma_full_ft.mojo
#      header / commit 26089e4) — while every chain-clean arm (bproj/bmlp2/
#      b2 from F32 d_y, bqkv, double q/k d_g) holds the 0.9999 campaign bar.
#
# The forward is the DEVICE-RESIDENT base block (chroma_*_lora_forward_device
# with NO adapters — full-FT has no LoRA), MATH attention arm (FLASH=False,
# the bit-oracle mode). The oracle is chroma_block_oracle.py in its BASE-ONLY
# mode (CHROMA_FT_ORACLE=1 -> d_ftref_*/s_ftref_* refs): the existing LoRA-
# inclusive refs are NOT valid for FT (the adapters shift both the dX chain
# and, through downstream activations, the dW values — krea2 trap). The
# oracle ALREADY dumps the bias/norm-scale grads (d_ftref_*_d_b*/…_d_{q,k}_
# norm, s_ftref_d_b{1,2}/…) — mod-vec grads stay uncompared (frozen
# approximator rows, Phase C).
#
# Run (oracle FIRST, as SEPARATE commands — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   CHROMA_FT_ORACLE=1 /home/alex/SerenityTrainer/venv/bin/python \
#       serenitymojo/models/chroma/parity/chroma_block_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build --optimization-level 2 --target-accelerator sm_120 \
#       -I . -I /home/alex/MOJO-libs -Xlinker -lm \
#       -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#       -Xlinker -rpath -Xlinker $PWD/serenitymojo/ops/cshim/lib \
#       -Xlinker -rpath -Xlinker /home/alex/.serenity/cudnn/lib \
#       serenitymojo/models/chroma/parity/chroma_block_ft_parity.mojo \
#       -o /tmp/chroma_block_ft_parity
#   /tmp/chroma_block_ft_parity

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import alloc, ArcPointer
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.chroma.chroma_block import (
    ChromaStreamWeights, ChromaDoubleBlockWeights, ChromaModVecs,
    ChromaSingleBlockWeights, ChromaSingleModVecs,
    chroma_double_block_ft_backward_dev, chroma_single_block_ft_backward_dev,
)
from serenitymojo.models.chroma.chroma_block_device import (
    ChromaLoraAdapterDevice, StreamLoraDevice, DoubleBlockLoraDevice,
    SingleBlockLoraDevice,
    modvecs_to_device, single_modvecs_to_device,
    chroma_double_block_lora_forward_device,
    chroma_single_block_lora_forward_device,
)

comptime TArc = ArcPointer[Tensor]
comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/chroma/parity/"

# dims MUST match chroma_block_oracle.py
comptime H = 24
comptime Dh = 128
comptime D = H * Dh        # 3072
comptime N_IMG = 4
comptime N_TXT = 3
comptime S_D = N_IMG + N_TXT
comptime S_SINGLE = 6
comptime FMLP = 32
comptime EPS = Float32(1e-06)
comptime BAR_V2 = 0.9999    # new arms, chain-clean (Phase B campaign bar)
comptime BAR_CHAIN = 0.9997  # new arms whose source grad already gates a v1
                             # matmul dW at this block's chain class (header)


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing ref (run the FT oracle first): ") + path)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var nf = n // 4
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(nf):
        out.append(fp[i])
    buf.free()
    return out^


def _in(name: String) raises -> List[Float32]:
    return _read_bin_f32(REF_DIR + name + ".bin")


# row-stack list of host buffers (mirror weights.mojo::_row_stack)
def _row_stack(var parts: List[List[Float32]]) -> List[Float32]:
    var out = List[Float32]()
    for p in range(len(parts)):
        for i in range(len(parts[p])):
            out.append(parts[p][i])
    return out^


# Build a ChromaStreamWeights from the SEPARATE oracle weights (the loader's job).
def _load_double_stream(prefix: String, ctx: DeviceContext) raises -> ChromaStreamWeights:
    var wq = _in("d_in_" + prefix + "_to_q")
    var wk = _in("d_in_" + prefix + "_to_k")
    var wv = _in("d_in_" + prefix + "_to_v")
    var wparts = List[List[Float32]]()
    wparts.append(wq^); wparts.append(wk^); wparts.append(wv^)
    var wqkv = _row_stack(wparts^)
    var bq = _in("d_in_" + prefix + "_to_q_b")
    var bk = _in("d_in_" + prefix + "_to_k_b")
    var bv = _in("d_in_" + prefix + "_to_v_b")
    var bparts = List[List[Float32]]()
    bparts.append(bq^); bparts.append(bk^); bparts.append(bv^)
    var bqkv = _row_stack(bparts^)
    return ChromaStreamWeights(
        wqkv^, bqkv^,
        _in("d_in_" + prefix + "_out"), _in("d_in_" + prefix + "_out_b"),
        _in("d_in_" + prefix + "_mlp0"), _in("d_in_" + prefix + "_mlp0_b"),
        _in("d_in_" + prefix + "_mlp2"), _in("d_in_" + prefix + "_mlp2_b"),
        _in("d_in_" + prefix + "_q_norm"), _in("d_in_" + prefix + "_k_norm"),
        D, FMLP, Dh, ctx,
    )


def _load_mod(prefix: String) raises -> ChromaModVecs:
    return ChromaModVecs(
        _in("d_in_" + prefix + "_shift1"), _in("d_in_" + prefix + "_scale1"),
        _in("d_in_" + prefix + "_gate1"),
        _in("d_in_" + prefix + "_shift2"), _in("d_in_" + prefix + "_scale2"),
        _in("d_in_" + prefix + "_gate2"),
    )


def _none_stream_lora() -> StreamLoraDevice:
    return StreamLoraDevice(
        Optional[ChromaLoraAdapterDevice](None), Optional[ChromaLoraAdapterDevice](None),
        Optional[ChromaLoraAdapterDevice](None), Optional[ChromaLoraAdapterDevice](None),
        Optional[ChromaLoraAdapterDevice](None), Optional[ChromaLoraAdapterDevice](None),
    )


def _check(
    mut harness: ParityHarness, name: String,
    actual: List[Float32], expected: List[Float32], mut allok: Bool,
) raises:
    var r = harness.compare_host(actual, expected)
    print("  cos(", name, ") =", r.cos, "  max_abs =", r.max_abs,
          "  n =", r.n, "  ", "PASS" if r.passed else "FAIL")
    if not r.passed:
        allok = False


def main() raises:
    var ctx = DeviceContext()
    print("==== chroma_block_ft_parity (Chroma=Flux double+single full-FT dW backward vs torch) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " N_IMG=", N_IMG, " N_TXT=", N_TXT,
          " FMLP=", FMLP, " (base-only oracle: CHROMA_FT_ORACLE=1)")
    var harness = ParityHarness()
    var allok = True

    # ════════════════════════ DOUBLE BLOCK ════════════════════════
    print("")
    print("################ DOUBLE BLOCK ################")
    var img = _in("d_in_img")
    var txt = _in("d_in_txt")
    var iw = _load_double_stream("iw", ctx)
    var tw = _load_double_stream("tw", ctx)
    var im = _load_mod("im")
    var tm = _load_mod("tm")
    var im_dev = modvecs_to_device(im, D, ctx)
    var tm_dev = modvecs_to_device(tm, D, ctx)
    var cos_h = _in("d_in_cos")
    var sin_h = _in("d_in_sin")
    var cos = Tensor.from_host(cos_h, [S_D * H, Dh // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(sin_h, [S_D * H, Dh // 2], STDtype.F32, ctx)
    var w = ChromaDoubleBlockWeights(iw^, tw^)

    # NO adapters: the full-FT forward is the base block.
    var lora = DoubleBlockLoraDevice(_none_stream_lora(), _none_stream_lora())

    # forward (device-resident base, MATH arm; saves the tape the FT bwd consumes)
    var img_x = TArc(Tensor.from_host(img.copy(), [N_IMG, D], STDtype.BF16, ctx))
    var txt_x = TArc(Tensor.from_host(txt.copy(), [N_TXT, D], STDtype.BF16, ctx))
    var fwd = chroma_double_block_lora_forward_device[H, Dh, N_IMG, N_TXT, S_D, False](
        img_x, txt_x, w, im_dev, tm_dev, lora, cos, sin, D, FMLP, EPS, ctx,
    )

    print("---- forward outputs vs torch ----")
    var img_out_h = fwd.img_out[].to_host(ctx)
    var txt_out_h = fwd.txt_out[].to_host(ctx)
    _check(harness, "img_out", img_out_h, _in("d_ftref_img_out"), allok)
    _check(harness, "txt_out", txt_out_h, _in("d_ftref_txt_out"), allok)

    # ── RUN 1: SURFACE_V2=False (flux's CURRENT 10-arm behavior — the flux
    # regression): d_img/d_txt + the 8 matmul dW, dw list len 8. ──────────────
    var d_img_h = _in("d_in_d_img")
    var d_txt_h = _in("d_in_d_txt")
    var d_io = TArc(Tensor.from_host(d_img_h, [N_IMG, D], STDtype.F32, ctx))
    var d_to = TArc(Tensor.from_host(d_txt_h, [N_TXT, D], STDtype.F32, ctx))
    var g = chroma_double_block_ft_backward_dev[H, Dh, N_IMG, N_TXT, S_D, False](
        d_io, d_to, w, im_dev, tm_dev, fwd.saved, cos, sin, D, FMLP, EPS, ctx,
    )
    if len(g.dw) != 8:
        raise Error("FLUX REGRESSION FAIL: default (v1) double dw list is not len 8")

    print("---- [v1/flux regression] input grads vs torch ----")
    var d_img_x_h = g.d_img_x[].to_host(ctx)
    var d_txt_x_h = g.d_txt_x[].to_host(ctx)
    _check(harness, "d_img", d_img_x_h, _in("d_ftref_d_img"), allok)
    _check(harness, "d_txt", d_txt_x_h, _in("d_ftref_d_txt"), allok)

    print("---- [v1/flux regression] d_W vs torch (8 matmuls, fixed dw order) ----")
    var names = List[String]()
    names.append(String("im_d_wqkv"))
    names.append(String("im_d_wproj"))
    names.append(String("im_d_wmlp0"))
    names.append(String("im_d_wmlp2"))
    names.append(String("tm_d_wqkv"))
    names.append(String("tm_d_wproj"))
    names.append(String("tm_d_wmlp0"))
    names.append(String("tm_d_wmlp2"))
    for i in range(8):
        var dw_h = g.dw[i][].to_host(ctx)
        _check(harness, names[i], dw_h, _in("d_ftref_" + names[i]), allok)

    # ── RUN 2: SURFACE_V2=True (chroma's v2 FULL surface): v1 captures at the
    # same bar + the 12 NEW double-block arms at cos >= 0.9999. ───────────────
    var h2 = ParityHarness(BAR_V2)
    var g2 = chroma_double_block_ft_backward_dev[
        H, Dh, N_IMG, N_TXT, S_D, False, True
    ](
        d_io, d_to, w, im_dev, tm_dev, fwd.saved, cos, sin, D, FMLP, EPS, ctx,
    )
    if len(g2.dw) != 20:
        raise Error("V2 FAIL: SURFACE_V2 double dw list is not len 20")

    print("---- [v2] input grads vs torch ----")
    _check(harness, "v2 d_img", g2.d_img_x[].to_host(ctx), _in("d_ftref_d_img"), allok)
    _check(harness, "v2 d_txt", g2.d_txt_x[].to_host(ctx), _in("d_ftref_d_txt"), allok)
    print("---- [v2] d_W vs torch (8 matmuls, slots 0-7) ----")
    for i in range(8):
        var dw_h = g2.dw[i][].to_host(ctx)
        _check(harness, String("v2 ") + names[i], dw_h, _in("d_ftref_" + names[i]), allok)

    print("---- [v2 NEW] bias d_b + q/k rms d_g vs torch (slots 8-19; bar 0.9999,")
    print("     chain-class arms d_bmlp0 at 0.9997 — paired to d_wmlp0, header) ----")
    var h_chain = ParityHarness(BAR_CHAIN)
    var v2_names = List[String]()
    v2_names.append(String("im_d_bqkv"))    # 8
    v2_names.append(String("im_d_bproj"))   # 9
    v2_names.append(String("im_d_bmlp0"))   # 10  (chain class: d_wmlp0's grad)
    v2_names.append(String("im_d_bmlp2"))   # 11
    v2_names.append(String("tm_d_bqkv"))    # 12
    v2_names.append(String("tm_d_bproj"))   # 13
    v2_names.append(String("tm_d_bmlp0"))   # 14  (chain class)
    v2_names.append(String("tm_d_bmlp2"))   # 15
    v2_names.append(String("im_d_q_norm"))  # 16
    v2_names.append(String("im_d_k_norm"))  # 17
    v2_names.append(String("tm_d_q_norm"))  # 18
    v2_names.append(String("tm_d_k_norm"))  # 19
    for i in range(len(v2_names)):
        var dg_h = g2.dw[8 + i][].to_host(ctx)
        if i == 2 or i == 6:   # im/tm d_bmlp0 — chain class
            _check(h_chain, v2_names[i], dg_h, _in("d_ftref_" + v2_names[i]), allok)
        else:
            _check(h2, v2_names[i], dg_h, _in("d_ftref_" + v2_names[i]), allok)

    # ════════════════════════ SINGLE BLOCK ════════════════════════
    print("")
    print("################ SINGLE BLOCK ################")
    var sx = _in("s_in_x")
    # row-stack to_q/to_k/to_v/proj_mlp -> w1 ; biases -> b1 (the loader's job)
    var swq = _in("s_in_w_to_q"); var swk = _in("s_in_w_to_k")
    var swv = _in("s_in_w_to_v"); var swm = _in("s_in_w_proj_mlp")
    var w1parts = List[List[Float32]]()
    w1parts.append(swq^); w1parts.append(swk^); w1parts.append(swv^); w1parts.append(swm^)
    var w1 = _row_stack(w1parts^)
    var sbq = _in("s_in_w_to_q_b"); var sbk = _in("s_in_w_to_k_b")
    var sbv = _in("s_in_w_to_v_b"); var sbm = _in("s_in_w_proj_mlp_b")
    var b1parts = List[List[Float32]]()
    b1parts.append(sbq^); b1parts.append(sbk^); b1parts.append(sbv^); b1parts.append(sbm^)
    var b1 = _row_stack(b1parts^)
    var sw = ChromaSingleBlockWeights(
        w1^, b1^, _in("s_in_w_w2"), _in("s_in_w_b2"),
        _in("s_in_w_q_norm"), _in("s_in_w_k_norm"),
        D, FMLP, Dh, ctx,
    )
    var smv = ChromaSingleModVecs(
        _in("s_in_m_shift"), _in("s_in_m_scale"), _in("s_in_m_gate"),
    )
    var smv_dev = single_modvecs_to_device(smv, D, ctx)
    var scos_h = _in("s_in_cos")
    var ssin_h = _in("s_in_sin")
    var scos = Tensor.from_host(scos_h, [S_SINGLE * H, Dh // 2], STDtype.F32, ctx)
    var ssin = Tensor.from_host(ssin_h, [S_SINGLE * H, Dh // 2], STDtype.F32, ctx)

    var slora = SingleBlockLoraDevice(
        Optional[ChromaLoraAdapterDevice](None), Optional[ChromaLoraAdapterDevice](None),
        Optional[ChromaLoraAdapterDevice](None), Optional[ChromaLoraAdapterDevice](None),
        Optional[ChromaLoraAdapterDevice](None),
    )

    var sx_d = TArc(Tensor.from_host(sx.copy(), [S_SINGLE, D], STDtype.BF16, ctx))
    var sfwd = chroma_single_block_lora_forward_device[H, Dh, S_SINGLE, False](
        sx_d, sw, smv_dev, slora, scos, ssin, D, FMLP, EPS, ctx,
    )
    print("---- forward output vs torch ----")
    var s_out_h = sfwd.out[].to_host(ctx)
    _check(harness, "s_out", s_out_h, _in("s_ftref_out"), allok)

    var s_d_out_h = _in("s_in_d_out")
    var s_d_out = TArc(Tensor.from_host(s_d_out_h, [S_SINGLE, D], STDtype.F32, ctx))

    # ── RUN 1: SURFACE_V2=False (flux regression, dw len 2) ──────────────────
    var sg = chroma_single_block_ft_backward_dev[H, Dh, S_SINGLE, False](
        s_d_out, sw, smv_dev, sfwd.saved, scos, ssin, D, FMLP, EPS, ctx,
    )
    if len(sg.dw) != 2:
        raise Error("FLUX REGRESSION FAIL: default (v1) single dw list is not len 2")

    print("---- [v1/flux regression] input grad + trainable weight grads vs torch ----")
    var s_d_x_h = sg.d_x[].to_host(ctx)
    _check(harness, "s_d_x ", s_d_x_h, _in("s_ftref_d_x"), allok)
    var s_d_w1_h = sg.dw[0][].to_host(ctx)
    _check(harness, "s_d_w1", s_d_w1_h, _in("s_ftref_d_w1"), allok)
    var s_d_w2_h = sg.dw[1][].to_host(ctx)
    _check(harness, "s_d_w2", s_d_w2_h, _in("s_ftref_d_w2"), allok)

    # ── RUN 2: SURFACE_V2=True (v2 full surface, dw len 6) ───────────────────
    var sh2 = ParityHarness(BAR_V2)
    var sg2 = chroma_single_block_ft_backward_dev[H, Dh, S_SINGLE, False, True](
        s_d_out, sw, smv_dev, sfwd.saved, scos, ssin, D, FMLP, EPS, ctx,
    )
    if len(sg2.dw) != 6:
        raise Error("V2 FAIL: SURFACE_V2 single dw list is not len 6")

    print("---- [v2] input grad + matmul dW vs torch ----")
    _check(harness, "v2 s_d_x ", sg2.d_x[].to_host(ctx), _in("s_ftref_d_x"), allok)
    _check(harness, "v2 s_d_w1", sg2.dw[0][].to_host(ctx), _in("s_ftref_d_w1"), allok)
    _check(harness, "v2 s_d_w2", sg2.dw[1][].to_host(ctx), _in("s_ftref_d_w2"), allok)

    print("---- [v2 NEW] d_b1/d_b2 + q/k rms d_g vs torch (slots 2-5; b2 bar 0.9999,")
    print("     chain-class b1/q_norm/k_norm at 0.9997 — s_d_w1/w2 pairing, header) ----")
    var sh_chain = ParityHarness(BAR_CHAIN)
    _check(sh_chain, "s_d_b1    ", sg2.dw[2][].to_host(ctx), _in("s_ftref_d_b1"), allok)
    _check(sh2, "s_d_b2    ", sg2.dw[3][].to_host(ctx), _in("s_ftref_d_b2"), allok)
    _check(sh_chain, "s_d_q_norm", sg2.dw[4][].to_host(ctx), _in("s_ftref_d_q_norm"), allok)
    _check(sh_chain, "s_d_k_norm", sg2.dw[5][].to_host(ctx), _in("s_ftref_d_k_norm"), allok)

    print("")
    if allok:
        print("VERDICT: PASS — Chroma(=Flux) double+single full-FT backward matches torch")
        print("  (v1/flux-regression + v2 captures cos>=0.999; 16 NEW v2 arms: chain-clean")
        print("   at cos>=0.9999, chain-class [d_bmlp0 x2, s_d_b1, s_d_{q,k}_norm] at")
        print("   cos>=0.9997 — the block's original FT torch-gated class)")
    else:
        print("VERDICT: FAIL — at least one grad diverged (see FAIL lines above)")
