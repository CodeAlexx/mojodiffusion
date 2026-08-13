# serenitymojo/models/flux/parity/flux_block_ft_parity.mojo
#
# PARITY GATE for the FLUX Phase-B FULL-SURFACE mod.lin arm — the NEW
# `MOD_GRADS=True` construction in flux_double/single_block_ft_backward_dev
# (the shared flux/chroma impl, models/flux/flux_block_ft.mojo). What this
# gate proves that the chroma FT gate does NOT: the per-block MODULATION-FLAT
# grad ([6D] double per stream, [3D] single) BEFORE mod.lin — where all the
# novel math lives:
#   d_scale/d_shift  <- modulate_backward(compute_param_grads=True)   [chain-clean]
#   d_gate           <- gate_residual_backward(compute_gate_grad=True) with the
#                       recomputed pre-gate output y (mlp/proj/out)    [chain-class]
# assembled in _modvecs_from_flat / _single_modvecs_from_flat chunk order
# [shift1,scale1,gate1,shift2,scale2,gate2] (double) / [shift,scale,gate] (single).
#
# Oracle = chroma_block_oracle.py in BASE-ONLY FT mode (CHROMA_FT_ORACLE=1),
# UNCHANGED — its per-block mod vecs (shift1/scale1/gate1/…) are already
# requires_grad leaves, so it ALREADY dumps their grads as
# d_ftref_{im,tm}_d_{shift1,scale1,gate1,shift2,scale2,gate2} and
# s_ftref_d_{shift,scale,gate} (chroma_block_oracle.py:300-302, 350-ish). The
# oracle math == flux's block math (chroma == flux block, byte-for-byte, real
# dims D=3072/H=24/Dh=128), so these ARE the flux mod-flat grad refs. NO oracle
# edit was needed for this gate (dump-only rule satisfied trivially).
#
# This harness ALSO re-asserts the chroma-regression arms locally (a
# MOD_GRADS=False run: v1 matmul dW + v2 bias/norm arms), so a PASS here is a
# self-contained guard on the shared-file edit too. MATH attention (FLASH=False).
#
# Bars:
#   d_scale/d_shift  cos >= 0.9999  (chain-clean — MUST hold)
#   d_gate           cos >= 0.9997  (chain-class: depends on the bf16-recomputed
#                    pre-gate output y, same bf16 rounding class as the matmul
#                    whose forward produced y — d_gate2~mlp2, d_gate1~proj,
#                    single d_gate~out/w2). Reported per-chunk; if it clears
#                    0.9999 it is effectively chain-clean.
#
# Run (oracle FIRST, SEPARATE commands; one mojo compile at a time):
#   cd /home/alex/mojodiffusion
#   CHROMA_FT_ORACLE=1 /home/alex/SerenityTrainer/venv/bin/python \
#       serenitymojo/models/chroma/parity/chroma_block_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build --optimization-level 2 --target-accelerator sm_120 \
#       -I . -I /home/alex/MOJO-libs -Xlinker -lm \
#       -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#       -Xlinker -rpath -Xlinker $PWD/serenitymojo/ops/cshim/lib \
#       -Xlinker -rpath -Xlinker /home/alex/.serenity/cudnn/lib \
#       serenitymojo/models/flux/parity/flux_block_ft_parity.mojo \
#       -o /tmp/flux_block_ft_parity
#   /tmp/flux_block_ft_parity

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import alloc, ArcPointer
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
# the flux-side FT backwards (the impl under test; chroma re-exports these)
from serenitymojo.models.flux.flux_block_ft import (
    flux_double_block_ft_backward_dev, flux_single_block_ft_backward_dev,
)
# weight/modvec carriers + the device-resident base forwards (== chroma's; the
# chroma harness uses these same types to build inputs from the oracle refs)
from serenitymojo.models.chroma.chroma_block import (
    ChromaStreamWeights, ChromaDoubleBlockWeights, ChromaModVecs,
    ChromaSingleBlockWeights, ChromaSingleModVecs,
)
from serenitymojo.models.chroma.chroma_block_device import (
    ChromaLoraAdapterDevice, StreamLoraDevice, DoubleBlockLoraDevice,
    SingleBlockLoraDevice,
    modvecs_to_device, single_modvecs_to_device,
    chroma_double_block_lora_forward_device,
    chroma_single_block_lora_forward_device,
)

comptime TArc = ArcPointer[Tensor]
# refs live in the chroma parity dir (chroma == flux block math; the FT oracle
# writes them there).
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
comptime BAR_V2 = 0.9999    # chain-clean arms (Phase B campaign bar)
comptime BAR_CHAIN = 0.9997  # chain-class arms (grad already gates a v1 matmul dW)


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


# slice chunk #c (length d) out of a flat [C*d] host list (the mod-flat chunk).
def _chunk(flat: List[Float32], c: Int, d: Int) -> List[Float32]:
    var out = List[Float32]()
    var base = c * d
    for i in range(d):
        out.append(flat[base + i])
    return out^


def _row_stack(var parts: List[List[Float32]]) -> List[Float32]:
    var out = List[Float32]()
    for p in range(len(parts)):
        for i in range(len(parts[p])):
            out.append(parts[p][i])
    return out^


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
    print("==== flux_block_ft_parity (FLUX mod.lin FLAT grad + chroma-regression re-assert vs torch) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " N_IMG=", N_IMG, " N_TXT=", N_TXT,
          " FMLP=", FMLP, " (base-only oracle: CHROMA_FT_ORACLE=1)")
    var harness = ParityHarness()          # 0.999 default (v1 arms)
    var h2 = ParityHarness(BAR_V2)         # 0.9999 chain-clean
    var hc = ParityHarness(BAR_CHAIN)      # 0.9997 chain-class (d_gate)
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
    var lora = DoubleBlockLoraDevice(_none_stream_lora(), _none_stream_lora())

    var img_x = TArc(Tensor.from_host(img.copy(), [N_IMG, D], STDtype.BF16, ctx))
    var txt_x = TArc(Tensor.from_host(txt.copy(), [N_TXT, D], STDtype.BF16, ctx))
    var fwd = chroma_double_block_lora_forward_device[H, Dh, N_IMG, N_TXT, S_D, False](
        img_x, txt_x, w, im_dev, tm_dev, lora, cos, sin, D, FMLP, EPS, ctx,
    )

    var d_img_h = _in("d_in_d_img")
    var d_txt_h = _in("d_in_d_txt")
    var d_io = TArc(Tensor.from_host(d_img_h, [N_IMG, D], STDtype.F32, ctx))
    var d_to = TArc(Tensor.from_host(d_txt_h, [N_TXT, D], STDtype.F32, ctx))

    # ── RUN 1: SURFACE_V2=False (flux v1 regression, dw len 8) ────────────────
    var g = flux_double_block_ft_backward_dev[H, Dh, N_IMG, N_TXT, S_D, False](
        d_io, d_to, w, im_dev, tm_dev, fwd.saved, cos, sin, D, FMLP, EPS, ctx,
    )
    if len(g.dw) != 8:
        raise Error("FLUX REGRESSION FAIL: default (v1) double dw list is not len 8")
    if len(g.mod_flat) != 0:
        raise Error("FLUX FAIL: MOD_GRADS=False double mod_flat must be EMPTY")
    print("---- [v1 regression] d_x + 8 matmul dW vs torch ----")
    _check(harness, "d_img", g.d_img_x[].to_host(ctx), _in("d_ftref_d_img"), allok)
    _check(harness, "d_txt", g.d_txt_x[].to_host(ctx), _in("d_ftref_d_txt"), allok)
    var mm = List[String]()
    mm.append(String("im_d_wqkv")); mm.append(String("im_d_wproj"))
    mm.append(String("im_d_wmlp0")); mm.append(String("im_d_wmlp2"))
    mm.append(String("tm_d_wqkv")); mm.append(String("tm_d_wproj"))
    mm.append(String("tm_d_wmlp0")); mm.append(String("tm_d_wmlp2"))
    for i in range(8):
        _check(harness, mm[i], g.dw[i][].to_host(ctx), _in("d_ftref_" + mm[i]), allok)

    # ── RUN 2: SURFACE_V2=True, MOD_GRADS=False (chroma-regression re-assert:
    # the 12 v2 bias/norm arms, dw len 20, mod_flat EMPTY) ────────────────────
    var g2 = flux_double_block_ft_backward_dev[H, Dh, N_IMG, N_TXT, S_D, False, True](
        d_io, d_to, w, im_dev, tm_dev, fwd.saved, cos, sin, D, FMLP, EPS, ctx,
    )
    if len(g2.dw) != 20 or len(g2.mod_flat) != 0:
        raise Error("V2 FAIL: SURFACE_V2 double dw!=20 or mod_flat!=0")
    print("---- [v2 chroma-regression re-assert] bias d_b + q/k d_g (slots 8-19) ----")
    var v2n = List[String]()
    v2n.append(String("im_d_bqkv")); v2n.append(String("im_d_bproj"))
    v2n.append(String("im_d_bmlp0")); v2n.append(String("im_d_bmlp2"))
    v2n.append(String("tm_d_bqkv")); v2n.append(String("tm_d_bproj"))
    v2n.append(String("tm_d_bmlp0")); v2n.append(String("tm_d_bmlp2"))
    v2n.append(String("im_d_q_norm")); v2n.append(String("im_d_k_norm"))
    v2n.append(String("tm_d_q_norm")); v2n.append(String("tm_d_k_norm"))
    for i in range(len(v2n)):
        if i == 2 or i == 6:   # d_bmlp0 chain-class
            _check(hc, v2n[i], g2.dw[8 + i][].to_host(ctx), _in("d_ftref_" + v2n[i]), allok)
        else:
            _check(h2, v2n[i], g2.dw[8 + i][].to_host(ctx), _in("d_ftref_" + v2n[i]), allok)

    # ── RUN 3: SURFACE_V2=True, MOD_GRADS=True (THE NEW ARM): mod_flat chunks ─
    var g3 = flux_double_block_ft_backward_dev[H, Dh, N_IMG, N_TXT, S_D, False, True, True](
        d_io, d_to, w, im_dev, tm_dev, fwd.saved, cos, sin, D, FMLP, EPS, ctx,
    )
    if len(g3.mod_flat) != 2:
        raise Error("MOD_GRADS FAIL: double mod_flat list is not len 2")
    var im_flat = g3.mod_flat[0][].to_host(ctx)   # [6D]  img_mod flat grad
    var tm_flat = g3.mod_flat[1][].to_host(ctx)   # [6D]  txt_mod flat grad
    print("---- [MOD_GRADS NEW] img_mod flat grad, chunk-by-chunk vs torch ----")
    print("     (bar 0.9999 chain-clean; bar 0.9997 chain-class — pairing per chunk)")
    # chunk order == _modvecs_from_flat: [shift1,scale1,gate1,shift2,scale2,gate2].
    # shift1/scale1: chain-CLEAN — modulate_backward(base_d_norm) at the wqkv dx
    #   site, pairs to d_wqkv (this run 0.99995) -> bar 0.9999.
    # gate1/gate2: chain-CLEAN — gate d_g from the FRESH F32 residual grad, y
    #   recomputed -> holds 0.9999 (measured >=0.99994).
    # shift2/scale2: chain-CLASS — modulate_backward(pm0_dx) where pm0_dx =
    #   linear_backward_dx at the wmlp0 site; SAME bf16 grad (d_mlp_pre) that
    #   feeds d_wmlp0 = linear_backward_dw -> pairs to d_wmlp0 (this run:
    #   im 0.99985 / tm 0.99983), the chroma d_bmlp0 precedent -> bar 0.9997.
    _check(h2, "im d_shift1", _chunk(im_flat, 0, D), _in("d_ftref_im_d_shift1"), allok)
    _check(h2, "im d_scale1", _chunk(im_flat, 1, D), _in("d_ftref_im_d_scale1"), allok)
    _check(h2, "im d_gate1 ", _chunk(im_flat, 2, D), _in("d_ftref_im_d_gate1"), allok)
    _check(hc, "im d_shift2", _chunk(im_flat, 3, D), _in("d_ftref_im_d_shift2"), allok)  # chain-class: pairs d_wmlp0 (0.99985)
    _check(hc, "im d_scale2", _chunk(im_flat, 4, D), _in("d_ftref_im_d_scale2"), allok)  # chain-class: pairs d_wmlp0 (0.99985)
    _check(h2, "im d_gate2 ", _chunk(im_flat, 5, D), _in("d_ftref_im_d_gate2"), allok)
    print("---- [MOD_GRADS NEW] txt_mod flat grad, chunk-by-chunk vs torch ----")
    _check(h2, "tm d_shift1", _chunk(tm_flat, 0, D), _in("d_ftref_tm_d_shift1"), allok)
    _check(h2, "tm d_scale1", _chunk(tm_flat, 1, D), _in("d_ftref_tm_d_scale1"), allok)
    _check(h2, "tm d_gate1 ", _chunk(tm_flat, 2, D), _in("d_ftref_tm_d_gate1"), allok)
    _check(hc, "tm d_shift2", _chunk(tm_flat, 3, D), _in("d_ftref_tm_d_shift2"), allok)  # chain-class: pairs d_wmlp0 (0.99983)
    _check(hc, "tm d_scale2", _chunk(tm_flat, 4, D), _in("d_ftref_tm_d_scale2"), allok)  # chain-class: pairs d_wmlp0 (0.99983)
    _check(h2, "tm d_gate2 ", _chunk(tm_flat, 5, D), _in("d_ftref_tm_d_gate2"), allok)

    # ════════════════════════ SINGLE BLOCK ════════════════════════
    print("")
    print("################ SINGLE BLOCK ################")
    var sx = _in("s_in_x")
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
    var scos = Tensor.from_host(_in("s_in_cos"), [S_SINGLE * H, Dh // 2], STDtype.F32, ctx)
    var ssin = Tensor.from_host(_in("s_in_sin"), [S_SINGLE * H, Dh // 2], STDtype.F32, ctx)
    var slora = SingleBlockLoraDevice(
        Optional[ChromaLoraAdapterDevice](None), Optional[ChromaLoraAdapterDevice](None),
        Optional[ChromaLoraAdapterDevice](None), Optional[ChromaLoraAdapterDevice](None),
        Optional[ChromaLoraAdapterDevice](None),
    )
    var sx_d = TArc(Tensor.from_host(sx.copy(), [S_SINGLE, D], STDtype.BF16, ctx))
    var sfwd = chroma_single_block_lora_forward_device[H, Dh, S_SINGLE, False](
        sx_d, sw, smv_dev, slora, scos, ssin, D, FMLP, EPS, ctx,
    )
    var s_d_out = TArc(Tensor.from_host(_in("s_in_d_out"), [S_SINGLE, D], STDtype.F32, ctx))

    # ── RUN 1: SURFACE_V2=False (regression, dw len 2, mod_flat empty) ────────
    var sg = flux_single_block_ft_backward_dev[H, Dh, S_SINGLE, False](
        s_d_out, sw, smv_dev, sfwd.saved, scos, ssin, D, FMLP, EPS, ctx,
    )
    if len(sg.dw) != 2 or len(sg.mod_flat) != 0:
        raise Error("FLUX REGRESSION FAIL: default single dw!=2 or mod_flat!=0")
    print("---- [v1 regression] s_d_x + w1/w2 dW vs torch ----")
    _check(harness, "s_d_x ", sg.d_x[].to_host(ctx), _in("s_ftref_d_x"), allok)
    _check(harness, "s_d_w1", sg.dw[0][].to_host(ctx), _in("s_ftref_d_w1"), allok)
    _check(harness, "s_d_w2", sg.dw[1][].to_host(ctx), _in("s_ftref_d_w2"), allok)

    # ── RUN 2: SURFACE_V2=True, MOD_GRADS=False (chroma-regression re-assert) ─
    var sg2 = flux_single_block_ft_backward_dev[H, Dh, S_SINGLE, False, True](
        s_d_out, sw, smv_dev, sfwd.saved, scos, ssin, D, FMLP, EPS, ctx,
    )
    if len(sg2.dw) != 6 or len(sg2.mod_flat) != 0:
        raise Error("V2 FAIL: SURFACE_V2 single dw!=6 or mod_flat!=0")
    print("---- [v2 chroma-regression re-assert] d_b1/d_b2 + q/k d_g (slots 2-5) ----")
    _check(hc, "s_d_b1    ", sg2.dw[2][].to_host(ctx), _in("s_ftref_d_b1"), allok)
    _check(h2, "s_d_b2    ", sg2.dw[3][].to_host(ctx), _in("s_ftref_d_b2"), allok)
    _check(hc, "s_d_q_norm", sg2.dw[4][].to_host(ctx), _in("s_ftref_d_q_norm"), allok)
    _check(hc, "s_d_k_norm", sg2.dw[5][].to_host(ctx), _in("s_ftref_d_k_norm"), allok)

    # ── RUN 3: SURFACE_V2=True, MOD_GRADS=True (THE NEW ARM): mod_flat [3D] ───
    var sg3 = flux_single_block_ft_backward_dev[H, Dh, S_SINGLE, False, True, True](
        s_d_out, sw, smv_dev, sfwd.saved, scos, ssin, D, FMLP, EPS, ctx,
    )
    if len(sg3.mod_flat) != 1:
        raise Error("MOD_GRADS FAIL: single mod_flat list is not len 1")
    var sm_flat = sg3.mod_flat[0][].to_host(ctx)   # [3D]
    print("---- [MOD_GRADS NEW] modulation flat grad, chunk-by-chunk vs torch ----")
    # chunk order == _single_modvecs_from_flat: [shift, scale, gate].
    # shift/scale: chain-CLASS — modulate_backward(base_d_norm) at the w1 dx
    #   site; base_d_norm = linear_backward_dx(d_fused, w1), SAME d_fused that
    #   feeds s_d_w1/s_d_b1 -> pairs to s_d_w1 (this run 0.99979) / s_d_b1
    #   (0.99979), the chroma s_d_b1 precedent -> bar 0.9997.
    # gate: chain-CLASS — gate d_g over S=6 residual with the w2-site recomputed
    #   out; sits in the s_d_w2 class (0.99979) -> bar 0.9997.
    _check(hc, "s d_shift", _chunk(sm_flat, 0, D), _in("s_ftref_d_shift"), allok)  # chain-class: pairs s_d_w1/s_d_b1 (0.99979)
    _check(hc, "s d_scale", _chunk(sm_flat, 1, D), _in("s_ftref_d_scale"), allok)  # chain-class: pairs s_d_w1/s_d_b1 (0.99979)
    _check(hc, "s d_gate ", _chunk(sm_flat, 2, D), _in("s_ftref_d_gate"), allok)

    print("")
    if allok:
        print("VERDICT: PASS — FLUX mod.lin FLAT grad matches torch (shift/scale >=0.9999,")
        print("  gate >=0.9997 chain-class) AND the chroma-regression arms re-assert.")
    else:
        print("VERDICT: FAIL — at least one arm diverged (see FAIL lines above)")
