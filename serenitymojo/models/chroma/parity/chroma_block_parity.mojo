# serenitymojo/models/chroma/parity/chroma_block_parity.mojo
#
# PARITY GATE for the Chroma1-HD DOUBLE + SINGLE DiT block TRAINING surface
# (models/chroma/chroma_block.mojo, which reuses the proven Flux block after the
# Chroma loader's separate->fused row-stack). Loads the EXACT inputs + torch-
# autograd reference grads dumped by chroma_block_oracle.py, ROW-STACKS the
# SEPARATE Chroma weights into the fused block structs (exactly as
# models/chroma/weights.mojo does), runs chroma_double/single_block forward +
# backward (base AND LoRA), and compares forward outputs, input grads, every
# fused weight+bias grad, the modulation grads, AND every LoRA d_A/d_B at
# cos >= 0.999.
#
# This proves: (a) the Chroma block backward (= reused Flux block) matches torch
# at REAL H=24/Dh=128; (b) the separate<->fused weight/grad mapping the loader
# performs is correct (the fused d_wqkv/d_w1 match torch's stacked separate
# grads); (c) the per-slice LoRA on Chroma's separate to_q/to_k/to_v + proj_mlp
# matches torch.
#
# Run (oracle FIRST, SEPARATE command — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/chroma/parity/chroma_block_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/chroma/parity/chroma_block_parity.mojo

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.chroma.chroma_block import (
    ChromaStreamWeights, ChromaDoubleBlockWeights, ChromaModVecs,
    ChromaSingleBlockWeights, ChromaSingleModVecs,
    ChromaStreamLora, ChromaDoubleBlockLora, ChromaSingleBlockLora,
    chroma_double_block_lora_forward, chroma_double_block_lora_backward,
    chroma_single_block_lora_forward, chroma_single_block_lora_backward,
    D_SQ, D_SK, D_SV, D_PROJ, D_MLP0, D_MLP2,
    S_SQ, S_SK, S_SV, S_PMLP, S_L2,
)


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/chroma/parity/"

# dims MUST match chroma_block_oracle.py
comptime H = 24
comptime Dh = 128
comptime D = H * Dh        # 3072
comptime N_IMG = 4
comptime N_TXT = 3
comptime S_SINGLE = 6
comptime FMLP = 32
comptime RANK = 4
comptime LSCALE = Float32(2.0)   # ALPHA/RANK = 8/4
comptime EPS = Float32(1e-06)


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing ref (run the oracle first): ") + path)
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


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(0.0)
    return o^


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


def _adapter(tag: String, in_f: Int, out_f: Int) raises -> LoraAdapter:
    return LoraAdapter(
        _in(tag + "_a"), _in(tag + "_b"),
        RANK, in_f, out_f, LSCALE,
        _zeros(RANK * in_f), _zeros(RANK * in_f),
        _zeros(out_f * RANK), _zeros(out_f * RANK),
    )


def _stream_lora(nm: String, mlp_hidden: Int, ctx: DeviceContext) raises -> ChromaStreamLora:
    # slots: to_q[D,D] to_k[D,D] to_v[D,D] proj[D,D] mlp0[D,FMLP] mlp2[FMLP,D]
    return ChromaStreamLora(
        Optional[LoraAdapter](_adapter("d_in_" + nm + "_to_q", D, D)),
        Optional[LoraAdapter](_adapter("d_in_" + nm + "_to_k", D, D)),
        Optional[LoraAdapter](_adapter("d_in_" + nm + "_to_v", D, D)),
        Optional[LoraAdapter](_adapter("d_in_" + nm + "_proj", D, D)),
        Optional[LoraAdapter](_adapter("d_in_" + nm + "_mlp0", D, mlp_hidden)),
        Optional[LoraAdapter](_adapter("d_in_" + nm + "_mlp2", mlp_hidden, D)),
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
    print("==== chroma block parity (Chroma DiT block fwd+bwd+LoRA vs torch) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " N_IMG=", N_IMG, " N_TXT=", N_TXT,
          " FMLP=", FMLP, " RANK=", RANK)
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
    var cos_h = _in("d_in_cos")
    var sin_h = _in("d_in_sin")
    var cos = Tensor.from_host(cos_h, [(N_IMG + N_TXT) * H, Dh // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(sin_h, [(N_IMG + N_TXT) * H, Dh // 2], STDtype.F32, ctx)
    var w = ChromaDoubleBlockWeights(iw^, tw^)
    var ilora = _stream_lora("ilo", FMLP, ctx)
    var tlora = _stream_lora("tlo", FMLP, ctx)
    var lora = ChromaDoubleBlockLora(ilora^, tlora^)

    var fwd = chroma_double_block_lora_forward[H, Dh, N_IMG, N_TXT, N_IMG + N_TXT](
        img.copy(), txt.copy(), w, im, tm, lora, cos, sin, D, FMLP, EPS, ctx,
    )
    print("---- forward outputs vs torch ----")
    _check(harness, "img_out", fwd.img_out, _in("d_ref_img_out"), allok)
    _check(harness, "txt_out", fwd.txt_out, _in("d_ref_txt_out"), allok)

    var d_img = _in("d_in_d_img")
    var d_txt = _in("d_in_d_txt")
    var g = chroma_double_block_lora_backward[H, Dh, N_IMG, N_TXT, N_IMG + N_TXT](
        d_img, d_txt, w, im, tm, lora, fwd.saved, cos, sin, D, FMLP, EPS, ctx,
    )

    print("---- INPUT grads vs torch ----")
    _check(harness, "d_img", g.base.img.d_x, _in("d_ref_d_img"), allok)
    _check(harness, "d_txt", g.base.txt.d_x, _in("d_ref_d_txt"), allok)

    print("---- IMG fused weight+bias grads vs torch ----")
    _check(harness, "img d_wqkv ", g.base.img.d_wqkv, _in("d_ref_im_d_wqkv"), allok)
    _check(harness, "img d_bqkv ", g.base.img.d_bqkv, _in("d_ref_im_d_bqkv"), allok)
    _check(harness, "img d_wproj", g.base.img.d_wproj, _in("d_ref_im_d_wproj"), allok)
    _check(harness, "img d_bproj", g.base.img.d_bproj, _in("d_ref_im_d_bproj"), allok)
    _check(harness, "img d_wmlp0", g.base.img.d_wmlp0, _in("d_ref_im_d_wmlp0"), allok)
    _check(harness, "img d_bmlp0", g.base.img.d_bmlp0, _in("d_ref_im_d_bmlp0"), allok)
    _check(harness, "img d_wmlp2", g.base.img.d_wmlp2, _in("d_ref_im_d_wmlp2"), allok)
    _check(harness, "img d_bmlp2", g.base.img.d_bmlp2, _in("d_ref_im_d_bmlp2"), allok)
    _check(harness, "img d_qnorm", g.base.img.d_q_norm, _in("d_ref_im_d_q_norm"), allok)
    _check(harness, "img d_knorm", g.base.img.d_k_norm, _in("d_ref_im_d_k_norm"), allok)

    print("---- IMG modulation grads vs torch ----")
    _check(harness, "img d_shift1", g.base.img.d_shift1, _in("d_ref_im_d_shift1"), allok)
    _check(harness, "img d_scale1", g.base.img.d_scale1, _in("d_ref_im_d_scale1"), allok)
    _check(harness, "img d_gate1 ", g.base.img.d_gate1, _in("d_ref_im_d_gate1"), allok)
    _check(harness, "img d_shift2", g.base.img.d_shift2, _in("d_ref_im_d_shift2"), allok)
    _check(harness, "img d_scale2", g.base.img.d_scale2, _in("d_ref_im_d_scale2"), allok)
    _check(harness, "img d_gate2 ", g.base.img.d_gate2, _in("d_ref_im_d_gate2"), allok)

    print("---- TXT fused weight+bias grads vs torch ----")
    _check(harness, "txt d_wqkv ", g.base.txt.d_wqkv, _in("d_ref_tm_d_wqkv"), allok)
    _check(harness, "txt d_bqkv ", g.base.txt.d_bqkv, _in("d_ref_tm_d_bqkv"), allok)
    _check(harness, "txt d_wproj", g.base.txt.d_wproj, _in("d_ref_tm_d_wproj"), allok)
    _check(harness, "txt d_bproj", g.base.txt.d_bproj, _in("d_ref_tm_d_bproj"), allok)
    _check(harness, "txt d_wmlp0", g.base.txt.d_wmlp0, _in("d_ref_tm_d_wmlp0"), allok)
    _check(harness, "txt d_bmlp0", g.base.txt.d_bmlp0, _in("d_ref_tm_d_bmlp0"), allok)
    _check(harness, "txt d_wmlp2", g.base.txt.d_wmlp2, _in("d_ref_tm_d_wmlp2"), allok)
    _check(harness, "txt d_bmlp2", g.base.txt.d_bmlp2, _in("d_ref_tm_d_bmlp2"), allok)
    _check(harness, "txt d_qnorm", g.base.txt.d_q_norm, _in("d_ref_tm_d_q_norm"), allok)
    _check(harness, "txt d_knorm", g.base.txt.d_k_norm, _in("d_ref_tm_d_k_norm"), allok)

    print("---- TXT modulation grads vs torch ----")
    _check(harness, "txt d_shift1", g.base.txt.d_shift1, _in("d_ref_tm_d_shift1"), allok)
    _check(harness, "txt d_scale1", g.base.txt.d_scale1, _in("d_ref_tm_d_scale1"), allok)
    _check(harness, "txt d_gate1 ", g.base.txt.d_gate1, _in("d_ref_tm_d_gate1"), allok)
    _check(harness, "txt d_shift2", g.base.txt.d_shift2, _in("d_ref_tm_d_shift2"), allok)
    _check(harness, "txt d_scale2", g.base.txt.d_scale2, _in("d_ref_tm_d_scale2"), allok)
    _check(harness, "txt d_gate2 ", g.base.txt.d_gate2, _in("d_ref_tm_d_gate2"), allok)

    print("---- IMG LoRA d_A/d_B vs torch ----")
    _check(harness, "img loA to_q", g.lora.img.d_a[D_SQ], _in("d_ref_ilo_to_q_d_a"), allok)
    _check(harness, "img loB to_q", g.lora.img.d_b[D_SQ], _in("d_ref_ilo_to_q_d_b"), allok)
    _check(harness, "img loA to_k", g.lora.img.d_a[D_SK], _in("d_ref_ilo_to_k_d_a"), allok)
    _check(harness, "img loB to_k", g.lora.img.d_b[D_SK], _in("d_ref_ilo_to_k_d_b"), allok)
    _check(harness, "img loA to_v", g.lora.img.d_a[D_SV], _in("d_ref_ilo_to_v_d_a"), allok)
    _check(harness, "img loB to_v", g.lora.img.d_b[D_SV], _in("d_ref_ilo_to_v_d_b"), allok)
    _check(harness, "img loA proj", g.lora.img.d_a[D_PROJ], _in("d_ref_ilo_proj_d_a"), allok)
    _check(harness, "img loB proj", g.lora.img.d_b[D_PROJ], _in("d_ref_ilo_proj_d_b"), allok)
    _check(harness, "img loA mlp0", g.lora.img.d_a[D_MLP0], _in("d_ref_ilo_mlp0_d_a"), allok)
    _check(harness, "img loB mlp0", g.lora.img.d_b[D_MLP0], _in("d_ref_ilo_mlp0_d_b"), allok)
    _check(harness, "img loA mlp2", g.lora.img.d_a[D_MLP2], _in("d_ref_ilo_mlp2_d_a"), allok)
    _check(harness, "img loB mlp2", g.lora.img.d_b[D_MLP2], _in("d_ref_ilo_mlp2_d_b"), allok)

    print("---- TXT LoRA d_A/d_B vs torch ----")
    _check(harness, "txt loA to_q", g.lora.txt.d_a[D_SQ], _in("d_ref_tlo_to_q_d_a"), allok)
    _check(harness, "txt loB to_q", g.lora.txt.d_b[D_SQ], _in("d_ref_tlo_to_q_d_b"), allok)
    _check(harness, "txt loA to_k", g.lora.txt.d_a[D_SK], _in("d_ref_tlo_to_k_d_a"), allok)
    _check(harness, "txt loB to_k", g.lora.txt.d_b[D_SK], _in("d_ref_tlo_to_k_d_b"), allok)
    _check(harness, "txt loA to_v", g.lora.txt.d_a[D_SV], _in("d_ref_tlo_to_v_d_a"), allok)
    _check(harness, "txt loB to_v", g.lora.txt.d_b[D_SV], _in("d_ref_tlo_to_v_d_b"), allok)
    _check(harness, "txt loA proj", g.lora.txt.d_a[D_PROJ], _in("d_ref_tlo_proj_d_a"), allok)
    _check(harness, "txt loB proj", g.lora.txt.d_b[D_PROJ], _in("d_ref_tlo_proj_d_b"), allok)
    _check(harness, "txt loA mlp0", g.lora.txt.d_a[D_MLP0], _in("d_ref_tlo_mlp0_d_a"), allok)
    _check(harness, "txt loB mlp0", g.lora.txt.d_b[D_MLP0], _in("d_ref_tlo_mlp0_d_b"), allok)
    _check(harness, "txt loA mlp2", g.lora.txt.d_a[D_MLP2], _in("d_ref_tlo_mlp2_d_a"), allok)
    _check(harness, "txt loB mlp2", g.lora.txt.d_b[D_MLP2], _in("d_ref_tlo_mlp2_d_b"), allok)

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
    var scos_h = _in("s_in_cos")
    var ssin_h = _in("s_in_sin")
    var scos = Tensor.from_host(scos_h, [S_SINGLE * H, Dh // 2], STDtype.F32, ctx)
    var ssin = Tensor.from_host(ssin_h, [S_SINGLE * H, Dh // 2], STDtype.F32, ctx)
    # single LoRA slots: to_q,to_k,to_v[D,D], proj_mlp[D,FMLP], linear2[D+FMLP,D]
    var slora = ChromaSingleBlockLora(
        Optional[LoraAdapter](_adapter("s_in_to_q", D, D)),
        Optional[LoraAdapter](_adapter("s_in_to_k", D, D)),
        Optional[LoraAdapter](_adapter("s_in_to_v", D, D)),
        Optional[LoraAdapter](_adapter("s_in_proj_mlp", D, FMLP)),
        Optional[LoraAdapter](_adapter("s_in_linear2", D + FMLP, D)),
    )

    var sfwd = chroma_single_block_lora_forward[H, Dh, S_SINGLE](
        sx.copy(), sw, smv, slora, scos, ssin, D, FMLP, EPS, ctx,
    )
    print("---- forward output vs torch ----")
    _check(harness, "s_out", sfwd.out, _in("s_ref_out"), allok)

    var s_d_out = _in("s_in_d_out")
    var sg = chroma_single_block_lora_backward[H, Dh, S_SINGLE](
        s_d_out, sw, smv, slora, sfwd.saved, scos, ssin, D, FMLP, EPS, ctx,
    )
    print("---- INPUT + fused weight grads vs torch ----")
    _check(harness, "s_d_x  ", sg.base.d_x, _in("s_ref_d_x"), allok)
    _check(harness, "s_d_w1 ", sg.base.d_w1, _in("s_ref_d_w1"), allok)
    _check(harness, "s_d_b1 ", sg.base.d_b1, _in("s_ref_d_b1"), allok)
    _check(harness, "s_d_w2 ", sg.base.d_w2, _in("s_ref_d_w2"), allok)
    _check(harness, "s_d_b2 ", sg.base.d_b2, _in("s_ref_d_b2"), allok)
    _check(harness, "s_d_qn ", sg.base.d_q_norm, _in("s_ref_d_q_norm"), allok)
    _check(harness, "s_d_kn ", sg.base.d_k_norm, _in("s_ref_d_k_norm"), allok)
    print("---- modulation grads vs torch ----")
    _check(harness, "s_d_shift", sg.base.d_shift, _in("s_ref_d_shift"), allok)
    _check(harness, "s_d_scale", sg.base.d_scale, _in("s_ref_d_scale"), allok)
    _check(harness, "s_d_gate ", sg.base.d_gate, _in("s_ref_d_gate"), allok)
    print("---- LoRA d_A/d_B vs torch ----")
    _check(harness, "s loA to_q", sg.lora.d_a[S_SQ], _in("s_ref_to_q_d_a"), allok)
    _check(harness, "s loB to_q", sg.lora.d_b[S_SQ], _in("s_ref_to_q_d_b"), allok)
    _check(harness, "s loA to_k", sg.lora.d_a[S_SK], _in("s_ref_to_k_d_a"), allok)
    _check(harness, "s loB to_k", sg.lora.d_b[S_SK], _in("s_ref_to_k_d_b"), allok)
    _check(harness, "s loA to_v", sg.lora.d_a[S_SV], _in("s_ref_to_v_d_a"), allok)
    _check(harness, "s loB to_v", sg.lora.d_b[S_SV], _in("s_ref_to_v_d_b"), allok)
    _check(harness, "s loA pmlp", sg.lora.d_a[S_PMLP], _in("s_ref_proj_mlp_d_a"), allok)
    _check(harness, "s loB pmlp", sg.lora.d_b[S_PMLP], _in("s_ref_proj_mlp_d_b"), allok)
    _check(harness, "s loA lin2", sg.lora.d_a[S_L2], _in("s_ref_linear2_d_a"), allok)
    _check(harness, "s loB lin2", sg.lora.d_b[S_L2], _in("s_ref_linear2_d_b"), allok)

    print("")
    if allok:
        print("VERDICT: PASS — Chroma double+single block fwd+bwd+LoRA matches torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one output diverged (see FAIL lines above)")
