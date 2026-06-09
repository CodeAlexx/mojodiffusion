# serenitymojo/models/zimage/parity/zimage_block_lora_parity.mojo
#
# PARITY GATE for the Z-Image (NextDiT) MAIN-LAYER DiT block training unit WITH
# LoRA adapters on all 7 trainable projections (to_q/k/v/out + SwiGLU w1/w3/w2)
# (models/zimage/lora_block.mojo). Loads the EXACT inputs + torch-autograd
# reference grads dumped by zimage_block_lora_oracle.py, runs
# zimage_block_lora_forward + zimage_block_lora_backward, and compares the
# forward output, base d_x, every base weight grad, the RAW adaLN modulation-vector
# grads, AND every per-slot LoRA d_A / d_B at cos >= 0.999.
#
# This brings Z-Image to the same LoRA-grad parity bar as Qwen-Image + SD3.5.
#
# slot order is canonical (lora_block.mojo): q,k,v,out,w1,w3,w2.
# LoRA adapters stored bf16 (LoraAdapter) — same as the live trainer; cos>=0.999
# tolerates the bf16-matmul accumulation diff vs the F64 torch oracle.
#
# Run (oracle FIRST, as a SEPARATE command — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/zimage/parity/zimage_block_lora_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/models/zimage/parity/zimage_block_lora_parity.mojo \
#       -o /tmp/zimage_block_lora_parity
#   /tmp/zimage_block_lora_parity

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import alloc, ArcPointer
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.zimage.weights import ZImageBlockWeights
from serenitymojo.models.zimage.block import ZImageModVecs
from serenitymojo.models.zimage.lora_block import (
    ZImageBlockLora, zimage_block_lora_forward, zimage_block_lora_backward,
    SLOT_Q, SLOT_K, SLOT_V, SLOT_O, SLOT_W1, SLOT_W3, SLOT_W2,
)


comptime TArc = ArcPointer[Tensor]
comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/zimage/parity/"

# dims MUST match zimage_block_lora_oracle.py
comptime H = 30
comptime Dh = 128
comptime D = H * Dh        # 3840
comptime S = 8
comptime F = 96
comptime HALF = Dh // 2    # 64
comptime EPS = Float32(1e-05)

# LoRA hyperparams MUST match the oracle
comptime RANK = 8
comptime SCALE_LORA = Float32(2.0)   # ALPHA(16)/RANK(8)


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


def _t1(vals: List[Float32], n: Int, ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_host(vals, [n], STDtype.F32, ctx))


def _t2(vals: List[Float32], a: Int, b: Int, ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_host(vals, [a, b], STDtype.F32, ctx))


def _load_weights(ctx: DeviceContext) raises -> ZImageBlockWeights:
    return ZImageBlockWeights(
        _t1(_in("lin_w_n1"), D, ctx),
        _t2(_in("lin_w_wq"), D, D, ctx),
        _t2(_in("lin_w_wk"), D, D, ctx),
        _t2(_in("lin_w_wv"), D, D, ctx),
        _t2(_in("lin_w_wo"), D, D, ctx),
        _t1(_in("lin_w_q_norm"), Dh, ctx),
        _t1(_in("lin_w_k_norm"), Dh, ctx),
        _t1(_in("lin_w_n2"), D, ctx),
        _t1(_in("lin_w_fn1"), D, ctx),
        _t2(_in("lin_w_w1"), F, D, ctx),
        _t2(_in("lin_w_w3"), F, D, ctx),
        _t2(_in("lin_w_w2"), D, F, ctx),
        _t1(_in("lin_w_fn2"), D, ctx),
    )


def _load_mod() raises -> ZImageModVecs:
    return ZImageModVecs(
        _in("lin_m_scale_msa"), _in("lin_m_gate_msa"),
        _in("lin_m_scale_mlp"), _in("lin_m_gate_mlp"),
    )


# build one LoRA adapter from dumped A[rank,in] + B[out,rank]; zero moments.
def _adapter(name: String, in_f: Int, out_f: Int) raises -> LoraAdapter:
    var a = _in("lin_lora_" + name + "_A")
    var b = _in("lin_lora_" + name + "_B")
    return LoraAdapter(
        a^, b^, RANK, in_f, out_f, SCALE_LORA,
        _zeros(RANK * in_f), _zeros(RANK * in_f),
        _zeros(out_f * RANK), _zeros(out_f * RANK),
    )


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(0.0))
    return o^


def _load_lora() raises -> ZImageBlockLora:
    return ZImageBlockLora(
        Optional[LoraAdapter](_adapter("q", D, D)),
        Optional[LoraAdapter](_adapter("k", D, D)),
        Optional[LoraAdapter](_adapter("v", D, D)),
        Optional[LoraAdapter](_adapter("out", D, D)),
        Optional[LoraAdapter](_adapter("w1", D, F)),
        Optional[LoraAdapter](_adapter("w3", D, F)),
        Optional[LoraAdapter](_adapter("w2", F, D)),
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
    print("==== zimage_block_lora_parity (Z-Image block + 7 LoRA slots vs torch) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " S=", S, " F=", F,
          " RANK=", RANK, " scale=", SCALE_LORA)

    var x = _in("lin_x")
    var w = _load_weights(ctx)
    var mv = _load_mod()
    var lora = _load_lora()

    var cos = Tensor.from_host(_in("lin_cos"), [S * H, HALF], STDtype.F32, ctx)
    var sin = Tensor.from_host(_in("lin_sin"), [S * H, HALF], STDtype.F32, ctx)

    # ── forward ──
    var fwd = zimage_block_lora_forward[H, Dh, S](
        x.copy(), w, mv, lora, cos, sin, D, F, EPS, ctx,
    )

    var harness = ParityHarness()
    var allok = True

    print("")
    print("---- forward output vs torch ----")
    _check(harness, "out", fwd.out, _in("lref_out"), allok)

    # ── backward ──
    var d_out = _in("lin_d_out")
    var g = zimage_block_lora_backward[H, Dh, S](
        d_out, w, mv, lora, fwd.saved, cos, sin, D, F, EPS, ctx,
    )

    print("")
    print("---- base input grad vs torch ----")
    _check(harness, "d_x", g.base.d_x, _in("lref_d_x"), allok)

    # NOTE: the 7 projection weights (wq/wk/wv/wo/w1/w3/w2) are FROZEN in LoRA
    # training — zimage_block_lora_backward leaves their grads empty by design.
    # Only the (still-trainable) norm scales + adaLN mod vectors are checked here.
    print("")
    print("---- trainable norm-scale grads vs torch ----")
    _check(harness, "d_n1     ", g.base.d_n1, _in("lref_d_n1"), allok)
    _check(harness, "d_q_norm ", g.base.d_q_norm, _in("lref_d_q_norm"), allok)
    _check(harness, "d_k_norm ", g.base.d_k_norm, _in("lref_d_k_norm"), allok)
    _check(harness, "d_n2     ", g.base.d_n2, _in("lref_d_n2"), allok)
    _check(harness, "d_fn1    ", g.base.d_fn1, _in("lref_d_fn1"), allok)
    _check(harness, "d_fn2    ", g.base.d_fn2, _in("lref_d_fn2"), allok)

    print("")
    print("---- RAW adaLN modulation-vector grads vs torch ----")
    _check(harness, "d_scale_msa", g.base.d_scale_msa, _in("lref_d_scale_msa"), allok)
    _check(harness, "d_gate_msa ", g.base.d_gate_msa, _in("lref_d_gate_msa"), allok)
    _check(harness, "d_scale_mlp", g.base.d_scale_mlp, _in("lref_d_scale_mlp"), allok)
    _check(harness, "d_gate_mlp ", g.base.d_gate_mlp, _in("lref_d_gate_mlp"), allok)

    print("")
    print("---- LoRA d_A / d_B per slot vs torch ----")
    _check(harness, "q  d_A", g.lora.d_a[SLOT_Q], _in("lref_q_dA"), allok)
    _check(harness, "q  d_B", g.lora.d_b[SLOT_Q], _in("lref_q_dB"), allok)
    _check(harness, "k  d_A", g.lora.d_a[SLOT_K], _in("lref_k_dA"), allok)
    _check(harness, "k  d_B", g.lora.d_b[SLOT_K], _in("lref_k_dB"), allok)
    _check(harness, "v  d_A", g.lora.d_a[SLOT_V], _in("lref_v_dA"), allok)
    _check(harness, "v  d_B", g.lora.d_b[SLOT_V], _in("lref_v_dB"), allok)
    _check(harness, "out d_A", g.lora.d_a[SLOT_O], _in("lref_out_dA"), allok)
    _check(harness, "out d_B", g.lora.d_b[SLOT_O], _in("lref_out_dB"), allok)
    _check(harness, "w1 d_A", g.lora.d_a[SLOT_W1], _in("lref_w1_dA"), allok)
    _check(harness, "w1 d_B", g.lora.d_b[SLOT_W1], _in("lref_w1_dB"), allok)
    _check(harness, "w3 d_A", g.lora.d_a[SLOT_W3], _in("lref_w3_dA"), allok)
    _check(harness, "w3 d_B", g.lora.d_b[SLOT_W3], _in("lref_w3_dB"), allok)
    _check(harness, "w2 d_A", g.lora.d_a[SLOT_W2], _in("lref_w2_dA"), allok)
    _check(harness, "w2 d_B", g.lora.d_b[SLOT_W2], _in("lref_w2_dB"), allok)

    print("")
    if allok:
        print("VERDICT: PASS — Z-Image block + 7 LoRA slots fwd+bwd match torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one output diverged (see FAIL lines above)")
