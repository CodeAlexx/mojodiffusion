# serenitymojo/models/scail2/parity/scail2_stack_lora_parity.mojo
#
# PARITY GATE for the SCAIL-2 i2v BLOCK-STACK LoRA composition (CHUNK 2a):
#   models/scail2/scail2_stack_lora.mojo::scail2_stack_lora_{forward,backward}.
# Consumes the N=3 chained-block dump from scail2_stack_lora_oracle.py (torch
# autograd, F32-residual SCAIL-2 blocks; independent per-block weights + LoRA).
#
# Gates (all cos>=0.999): final forward output, final d_x (grad w.r.t. the input
# sequence), and EVERY block's every LoRA d_A / d_B (3 blocks x 12 adapters).
# ALSO asserts per-block-RECOMPUTE backward == save-all backward at cos>=0.9999.
#
# Run (oracle FIRST, SEPARATE command):
#   /home/alex/torchref-image/venv/bin/python \
#       serenitymojo/models/scail2/parity/scail2_stack_lora_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/scail2/parity/scail2_stack_lora_parity.mojo

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.wan22.wan22_block import (
    WanBlockWeights, WanModVecs, WanI2VBlockWeights,
)
from serenitymojo.models.scail2.scail2_stack_lora import (
    Scail2LoraSet, scail2_stack_lora_forward, scail2_stack_lora_backward,
)


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/scail2/parity/"

comptime H = 4
comptime DH = 128
comptime DIM = H * DH           # 512
comptime S = 9
comptime TXT = 16
comptime IMG = 257
comptime FFN = 1024
comptime EPS = Float32(1e-06)
comptime RANK = 8
comptime LSCALE = Float32(1.0)   # alpha/rank = 8/8
comptime NBLK = 3

def _adapters() -> List[String]:
    return [
        String("sa_q"), String("sa_k"), String("sa_v"), String("sa_o"),
        String("ca_q"), String("ca_k"), String("ca_v"), String("ca_o"),
        String("ffn0"), String("ffn2"), String("img_k"), String("img_v"),
    ]


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


def _bin(bi: Int, name: String) raises -> List[Float32]:
    return _in(String("s2s_b") + String(bi) + "_" + name)


def _refin(bi: Int, name: String) raises -> List[Float32]:
    return _in(String("s2s_ref_b") + String(bi) + "_" + name)


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(0.0)
    return o^


def _load_block_weights(bi: Int, ctx: DeviceContext) raises -> WanI2VBlockWeights:
    var base = WanBlockWeights(
        _bin(bi, "sa_wq"), _bin(bi, "sa_wk"), _bin(bi, "sa_wv"), _bin(bi, "sa_wo"),
        _bin(bi, "sa_bq"), _bin(bi, "sa_bk"), _bin(bi, "sa_bv"), _bin(bi, "sa_bo"),
        _bin(bi, "sa_qn"), _bin(bi, "sa_kn"),
        _bin(bi, "ca_wq"), _bin(bi, "ca_wk"), _bin(bi, "ca_wv"), _bin(bi, "ca_wo"),
        _bin(bi, "ca_bq"), _bin(bi, "ca_bk"), _bin(bi, "ca_bv"), _bin(bi, "ca_bo"),
        _bin(bi, "ca_qn"), _bin(bi, "ca_kn"),
        _bin(bi, "n3_w"), _bin(bi, "n3_b"),
        _bin(bi, "ffn0_w"), _bin(bi, "ffn0_b"), _bin(bi, "ffn2_w"), _bin(bi, "ffn2_b"),
        DIM, FFN, DH, ctx,
    )
    return WanI2VBlockWeights(
        base^,
        _bin(bi, "ca_wk_img"), _bin(bi, "ca_bk_img"),
        _bin(bi, "ca_wv_img"), _bin(bi, "ca_bv_img"),
        _bin(bi, "ca_kn_img"),
        DIM, ctx,
    )


def _load_block_mod(bi: Int) raises -> WanModVecs:
    return WanModVecs(
        _bin(bi, "shift_sa"), _bin(bi, "scale_sa"), _bin(bi, "gate_sa"),
        _bin(bi, "shift_ffn"), _bin(bi, "scale_ffn"), _bin(bi, "gate_ffn"),
    )


def _make_adapter(a: List[Float32], b: List[Float32], in_f: Int, out_f: Int) -> LoraAdapter:
    return LoraAdapter(
        a.copy(), b.copy(), RANK, in_f, out_f, LSCALE,
        _zeros(RANK * in_f), _zeros(RANK * in_f),
        _zeros(out_f * RANK), _zeros(out_f * RANK),
    )


# Build the flat 12-slot-per-block adapter list from the oracle A/B dumps.
def _load_lora_set() raises -> Scail2LoraSet:
    var names = _adapters()
    var ad = List[LoraAdapter]()
    for bi in range(NBLK):
        # sa_{q,k,v,o}, ca_{q,k,v,o}: all in=out=DIM
        for si in range(8):
            var nm = names[si]
            ad.append(_make_adapter(_bin(bi, nm + "_A"), _bin(bi, nm + "_B"), DIM, DIM))
        # ffn0: dim->ffn ; ffn2: ffn->dim
        ad.append(_make_adapter(_bin(bi, "ffn0_A"), _bin(bi, "ffn0_B"), DIM, FFN))
        ad.append(_make_adapter(_bin(bi, "ffn2_A"), _bin(bi, "ffn2_B"), FFN, DIM))
        # img_k, img_v: in=out=DIM
        ad.append(_make_adapter(_bin(bi, "img_k_A"), _bin(bi, "img_k_B"), DIM, DIM))
        ad.append(_make_adapter(_bin(bi, "img_v_A"), _bin(bi, "img_v_B"), DIM, DIM))
    return Scail2LoraSet(ad^, NBLK, RANK)


def _check(
    mut harness: ParityHarness, name: String,
    actual: List[Float32], expected: List[Float32], mut allok: Bool,
) raises -> Float64:
    var r = harness.compare_host(actual, expected)
    print("  cos(", name, ") =", r.cos, "  max_abs =", r.max_abs,
          "  n =", r.n, "  ", "PASS" if r.passed else "FAIL")
    if not r.passed:
        allok = False
    return r.cos


def _cos_only(mut harness: ParityHarness, actual: List[Float32], expected: List[Float32]) raises -> Float64:
    return harness.compare_host(actual, expected).cos


def main() raises:
    var ctx = DeviceContext()
    print("==== scail2_stack_lora_parity (SCAIL-2 i2v F32-residual 3-block LoRA composition) ====")
    print("H=", H, " DH=", DH, " DIM=", DIM, " S=", S, " TXT=", TXT, " IMG=", IMG,
          " FFN=", FFN, " RANK=", RANK, " NBLK=", NBLK)

    var sequence = _in("s2s_seq")
    var context_txt = _in("s2s_context_txt")
    var context_img = _in("s2s_context_img")
    var cos = Tensor.from_host(_in("s2s_cos"), [S, DH // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(_in("s2s_sin"), [S, DH // 2], STDtype.F32, ctx)

    var base_blocks = List[WanI2VBlockWeights]()
    var modvecs = List[WanModVecs]()
    for bi in range(NBLK):
        base_blocks.append(_load_block_weights(bi, ctx))
        modvecs.append(_load_block_mod(bi))
    var lora_set = _load_lora_set()

    var harness = ParityHarness()      # cos>=0.999
    var allok = True

    # ── forward ──
    var fwd = scail2_stack_lora_forward[H, DH, S, TXT, IMG](
        sequence.copy(), modvecs, context_txt.copy(), context_img.copy(),
        cos, sin, base_blocks, lora_set, DIM, FFN, EPS, ctx,
    )
    print("")
    print("---- forward output vs torch (3-block F32 residual chain) ----")
    _ = _check(harness, "x_out", fwd.x_out, _in("s2s_ref_x_out"), allok)

    var d_out = _in("s2s_d_out")

    # ── backward (per-block RECOMPUTE) ──
    var bwd = scail2_stack_lora_backward[H, DH, S, TXT, IMG](
        d_out.copy(), modvecs, context_txt.copy(), context_img.copy(),
        cos, sin, base_blocks, lora_set, fwd, DIM, FFN, EPS, ctx, recompute=True,
    )
    print("")
    print("---- input grad vs torch ----")
    _ = _check(harness, "d_x (sequence)", bwd.d_x, _in("s2s_ref_d_x"), allok)

    print("")
    print("---- per-block LoRA d_A / d_B vs torch (3 blocks x 12 adapters; gate 0.999) ----")
    var worst_da = Float64(2.0)
    var worst_db = Float64(2.0)
    var names = _adapters()
    for bi in range(NBLK):
        print("  -- block", bi, "--")
        for si in range(len(names)):
            var nm = names[si]
            var slot = bi * 12 + si
            var ca = _check(harness, String("b") + String(bi) + " " + nm + " dA",
                            bwd.grads.d_a[slot], _refin(bi, nm + "_dA"), allok)
            var cb = _check(harness, String("b") + String(bi) + " " + nm + " dB",
                            bwd.grads.d_b[slot], _refin(bi, nm + "_dB"), allok)
            if ca < worst_da:
                worst_da = ca
            if cb < worst_db:
                worst_db = cb

    # ── per-block RECOMPUTE vs save-all backward (bit-stability; cos>=0.9999) ──
    var bwd_sa = scail2_stack_lora_backward[H, DH, S, TXT, IMG](
        d_out.copy(), modvecs, context_txt.copy(), context_img.copy(),
        cos, sin, base_blocks, lora_set, fwd, DIM, FFN, EPS, ctx, recompute=False,
    )
    print("")
    print("---- per-block-recompute backward == save-all backward (gate 0.9999) ----")
    var rc_ok = True
    var rc_worst = Float64(2.0)
    var c_dx = _cos_only(harness, bwd.d_x, bwd_sa.d_x)
    if c_dx < rc_worst:
        rc_worst = c_dx
    var n_ad = NBLK * 12
    for slot in range(n_ad):
        var cda = _cos_only(harness, bwd.grads.d_a[slot], bwd_sa.grads.d_a[slot])
        var cdb = _cos_only(harness, bwd.grads.d_b[slot], bwd_sa.grads.d_b[slot])
        if cda < rc_worst:
            rc_worst = cda
        if cdb < rc_worst:
            rc_worst = cdb
    if rc_worst < Float64(0.9999):
        rc_ok = False
    print("  d_x cos(recompute, save-all) =", c_dx)
    print("  worst cos over d_x + all 36 adapters' dA/dB =", rc_worst,
          "  ", "PASS" if rc_ok else "FAIL")

    print("")
    print("  worst LoRA dA cos =", worst_da, "   worst LoRA dB cos =", worst_db)
    print("")
    if allok and rc_ok:
        print("VERDICT: PASS -- SCAIL-2 i2v 3-block LoRA composition matches torch autograd;",
              "recompute == save-all")
    else:
        print("VERDICT: FAIL -- at least one output diverged (see FAIL lines above)")
