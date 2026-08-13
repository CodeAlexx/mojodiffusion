# serenitymojo/models/mageflow/parity/mageflow_block_lora_parity.mojo
#
# PARITY GATE for the Mage-Flow double-stream block LoRA training path.
# The block IS the Qwen-Image block (qwenimage_block.mojo::
# double_block_lora_forward/backward) at the REAL MageFlow dims
# (D=3072, H=24, Dh=128, FFN=12288); the ONE MageFlow delta — text tokens NOT
# roped — comes in via build_mageflow_rope_tables (models/dit/mageflow_dit.mojo:
# image msrope rows, text rows cos=1/sin=0).
#
# The oracle (mageflow_block_lora_oracle.py) runs torch autograd over the REAL
# MageFlowTransformerBlock (mage_layers.py) with REAL Mage-Flow-Turbo block-0
# weights and the REAL MageFlowEmbedRope. This gate:
#   1. cross-checks the built rope tables against the oracle's expanded
#      mage-freqs dump (proves the text-identity table itself),
#   2. gates x_out (img+txt), d_img, d_txt, and all 24 LoRA dA/dB at
#      cos >= 0.999 — proving the text-identity rope through the BACKWARD too.
#
# Run (oracle FIRST, SEPARATE command):
#   python3 \
#       serenitymojo/models/mageflow/parity/mageflow_block_lora_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/mageflow/parity/mageflow_block_lora_parity.mojo

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.klein.lora_block import LoraAdapter
from serenitymojo.models.dit.mageflow_dit import build_mageflow_rope_tables
from serenitymojo.models.qwenimage.qwenimage_block import (
    StreamWeights, DoubleBlockWeights, ModVecs,
    StreamLora, DoubleBlockLora,
    double_block_lora_forward, double_block_lora_backward,
)


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/mageflow/parity/"

# REAL MageFlow block dims (MageFlowConfig.mage_flow)
comptime H = 24
comptime Dh = 128
comptime D = H * Dh        # 3072
comptime FRAME = 1
comptime H_TOK = 4
comptime W_TOK = 4
comptime N_IMG = FRAME * H_TOK * W_TOK   # 16
comptime N_TXT = 8
comptime F = 12288
comptime RANK = 8
comptime ALPHA = Float32(8.0)
comptime SCALE_LORA = ALPHA / Float32(RANK)
comptime EPS = Float32(1e-06)
comptime THETA = Float64(10000.0)


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


def _load_stream(prefix: String, ctx: DeviceContext) raises -> StreamWeights:
    return StreamWeights(
        _in("lin_" + prefix + "_wq"), _in("lin_" + prefix + "_wk"), _in("lin_" + prefix + "_wv"),
        _in("lin_" + prefix + "_bq"), _in("lin_" + prefix + "_bk"), _in("lin_" + prefix + "_bv"),
        _in("lin_" + prefix + "_wout"), _in("lin_" + prefix + "_bout"),
        _in("lin_" + prefix + "_wup"), _in("lin_" + prefix + "_bup"),
        _in("lin_" + prefix + "_wdn"), _in("lin_" + prefix + "_bdn"),
        _in("lin_" + prefix + "_q_norm"), _in("lin_" + prefix + "_k_norm"),
        D, F, Dh, ctx,
    )


def _load_mod(prefix: String) raises -> ModVecs:
    return ModVecs(
        _in("lin_" + prefix + "_shift1"), _in("lin_" + prefix + "_scale1"),
        _in("lin_" + prefix + "_gate1"),
        _in("lin_" + prefix + "_shift2"), _in("lin_" + prefix + "_scale2"),
        _in("lin_" + prefix + "_gate2"),
    )


def _adapter(prefix: String, key: String, in_f: Int, out_f: Int) raises -> Optional[LoraAdapter]:
    var a = _in("lin_" + prefix + "_" + key + "_A")
    var b = _in("lin_" + prefix + "_" + key + "_B")
    return Optional[LoraAdapter](LoraAdapter(
        a^, b^, RANK, in_f, out_f, SCALE_LORA,
        _zeros(RANK * in_f), _zeros(RANK * in_f),
        _zeros(out_f * RANK), _zeros(out_f * RANK),
    ))


def _load_lora(prefix: String) raises -> StreamLora:
    return StreamLora(
        _adapter(prefix, "q", D, D), _adapter(prefix, "k", D, D),
        _adapter(prefix, "v", D, D), _adapter(prefix, "out", D, D),
        _adapter(prefix, "ff_up", D, F), _adapter(prefix, "ff_down", F, D),
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
    print("==== mageflow_block_lora_parity (MageFlow LoRA fwd+bwd vs torch over the REAL mage_flow block) ====")

    var img = _in("lin_img")
    var txt = _in("lin_txt")
    var iw = _load_stream("iw", ctx)
    var tw = _load_stream("tw", ctx)
    var im = _load_mod("im")
    var tm = _load_mod("tm")
    var ilo = _load_lora("ilo")
    var tlo = _load_lora("tlo")
    var w = DoubleBlockWeights(iw^, tw^)
    var lora = DoubleBlockLora(ilo^, tlo^)

    # MageFlow rope: image msrope rows, text rows identity (NOT roped)
    var rope = build_mageflow_rope_tables(
        FRAME, H_TOK, W_TOK, N_TXT, H, THETA, STDtype.F32, ctx
    )

    var harness = ParityHarness()
    var allok = True

    print("")
    print("---- rope tables: build_mageflow_rope_tables vs REAL MageFlowEmbedRope ----")
    _check(harness, "rope cos", rope[0].to_host(ctx), _in("lin_cos"), allok)
    _check(harness, "rope sin", rope[1].to_host(ctx), _in("lin_sin"), allok)

    var fwd = double_block_lora_forward[H, Dh, N_IMG, N_TXT, N_IMG + N_TXT](
        img.copy(), txt.copy(), w, im, tm, lora, rope[0], rope[1], D, F, EPS, ctx,
    )

    print("")
    print("---- forward x_out vs torch ----")
    _check(harness, "x_img_out", fwd.img_out, _in("lref_img_out"), allok)
    _check(harness, "x_txt_out", fwd.txt_out, _in("lref_txt_out"), allok)

    var d_img = _in("lin_d_img")
    var d_txt = _in("lin_d_txt")
    var g = double_block_lora_backward[H, Dh, N_IMG, N_TXT, N_IMG + N_TXT](
        d_img, d_txt, w, im, tm, lora, fwd.saved, rope[0], rope[1], D, F, EPS, ctx,
    )

    print("")
    print("---- input grads vs torch ----")
    _check(harness, "d_img", g.base.img.d_x, _in("lref_d_img"), allok)
    _check(harness, "d_txt", g.base.txt.d_x, _in("lref_d_txt"), allok)

    print("")
    print("---- IMG LoRA d_A/d_B vs torch ----")
    _check(harness, "img q dA  ", g.img.q_d_a, _in("lref_img_q_dA"), allok)
    _check(harness, "img q dB  ", g.img.q_d_b, _in("lref_img_q_dB"), allok)
    _check(harness, "img k dA  ", g.img.k_d_a, _in("lref_img_k_dA"), allok)
    _check(harness, "img k dB  ", g.img.k_d_b, _in("lref_img_k_dB"), allok)
    _check(harness, "img v dA  ", g.img.v_d_a, _in("lref_img_v_dA"), allok)
    _check(harness, "img v dB  ", g.img.v_d_b, _in("lref_img_v_dB"), allok)
    _check(harness, "img out dA", g.img.out_d_a, _in("lref_img_out_dA"), allok)
    _check(harness, "img out dB", g.img.out_d_b, _in("lref_img_out_dB"), allok)
    _check(harness, "img ffu dA", g.img.ff_up_d_a, _in("lref_img_ff_up_dA"), allok)
    _check(harness, "img ffu dB", g.img.ff_up_d_b, _in("lref_img_ff_up_dB"), allok)
    _check(harness, "img ffd dA", g.img.ff_down_d_a, _in("lref_img_ff_down_dA"), allok)
    _check(harness, "img ffd dB", g.img.ff_down_d_b, _in("lref_img_ff_down_dB"), allok)

    print("")
    print("---- TXT LoRA d_A/d_B vs torch (text-identity rope backward) ----")
    _check(harness, "txt q dA  ", g.txt.q_d_a, _in("lref_txt_q_dA"), allok)
    _check(harness, "txt q dB  ", g.txt.q_d_b, _in("lref_txt_q_dB"), allok)
    _check(harness, "txt k dA  ", g.txt.k_d_a, _in("lref_txt_k_dA"), allok)
    _check(harness, "txt k dB  ", g.txt.k_d_b, _in("lref_txt_k_dB"), allok)
    _check(harness, "txt v dA  ", g.txt.v_d_a, _in("lref_txt_v_dA"), allok)
    _check(harness, "txt v dB  ", g.txt.v_d_b, _in("lref_txt_v_dB"), allok)
    _check(harness, "txt out dA", g.txt.out_d_a, _in("lref_txt_out_dA"), allok)
    _check(harness, "txt out dB", g.txt.out_d_b, _in("lref_txt_out_dB"), allok)
    _check(harness, "txt ffu dA", g.txt.ff_up_d_a, _in("lref_txt_ff_up_dA"), allok)
    _check(harness, "txt ffu dB", g.txt.ff_up_d_b, _in("lref_txt_ff_up_dB"), allok)
    _check(harness, "txt ffd dA", g.txt.ff_down_d_a, _in("lref_txt_ff_down_dA"), allok)
    _check(harness, "txt ffd dB", g.txt.ff_down_d_b, _in("lref_txt_ff_down_dB"), allok)

    print("")
    if allok:
        print("VERDICT: PASS — MageFlow LoRA fwd+bwd matches torch over the REAL mage_flow block (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one tensor diverged")
