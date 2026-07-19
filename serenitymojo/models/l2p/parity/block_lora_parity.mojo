# serenitymojo/models/l2p/parity/block_lora_parity.mojo
#
# PARITY GATE for the Z-Image L2P (pixel-space) MAIN-LAYER DiT block training unit
# *WITH LoRA*. L2P reuses the Z-Image-Turbo DiT body verbatim, so the per-block
# backward surface == the Z-Image main block; this gate exercises the REUSED
# models/zimage/lora_block.mojo::zimage_block_lora_{forward,backward} (the host-list
# variant) and compares vs the torch-autograd reference dumped by
# block_lora_oracle.py at cos >= 0.999 on:
#   * forward output
#   * input grad d_x
#   * every base weight grad (n1, wq/wk/wv/wo, q_norm/k_norm, n2, fn1, w1/w3/w2, fn2)
#   * every RAW adaLN mod-vec grad (scale_msa/gate_msa/scale_mlp/gate_mlp)
#   * every LoRA d_A / d_B on the 7 trained projections (to_q/to_k/to_v/to_out/w1/w3/w2)
#
# REAL Z-Image/L2P head count H=30, head_dim Dh=128 (hidden D=3840). Small S/F to
# keep the torch oracle + GPU memory bounded. RoPE INTERLEAVED, half-width tables.
# NON-DEGENERATE sinusoidal data (rope cos std ~0.32). LoRA B perturbed non-zero
# (LIVE adapters). It imports ONLY lora_block.mojo (NOT zimage_stack_lora.mojo).
#
# Run (oracle FIRST, SEPARATE command; JIT, no build/package):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/l2p/parity/block_lora_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/l2p/parity/block_lora_parity.mojo

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import alloc, ArcPointer
from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.zimage.weights import ZImageBlockWeights
from serenitymojo.models.zimage.block import ZImageModVecs
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.zimage.lora_block import (
    ZImageBlockLora,
    zimage_block_lora_forward, zimage_block_lora_backward,
    SLOT_Q, SLOT_K, SLOT_V, SLOT_O, SLOT_W1, SLOT_W3, SLOT_W2,
)


comptime TArc = ArcPointer[Tensor]
comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/l2p/parity/"

# dims MUST match block_lora_oracle.py
comptime H = 30
comptime Dh = 128
comptime D = H * Dh        # 3840
comptime S = 8
comptime F = 96
comptime HALF = Dh // 2    # 64 (interleaved rope table width)
comptime EPS = Float32(1e-05)
comptime RANK = 8
comptime LSCALE = Float32(16.0) / Float32(RANK)   # alpha/rank = 2.0


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
        _t1(_in("in_w_n1"), D, ctx),
        _t2(_in("in_w_wq"), D, D, ctx),
        _t2(_in("in_w_wk"), D, D, ctx),
        _t2(_in("in_w_wv"), D, D, ctx),
        _t2(_in("in_w_wo"), D, D, ctx),
        _t1(_in("in_w_q_norm"), Dh, ctx),
        _t1(_in("in_w_k_norm"), Dh, ctx),
        _t1(_in("in_w_n2"), D, ctx),
        _t1(_in("in_w_fn1"), D, ctx),
        _t2(_in("in_w_w1"), F, D, ctx),
        _t2(_in("in_w_w3"), F, D, ctx),
        _t2(_in("in_w_w2"), D, F, ctx),
        _t1(_in("in_w_fn2"), D, ctx),
    )


def _load_mod() raises -> ZImageModVecs:
    return ZImageModVecs(
        _in("in_m_scale_msa"), _in("in_m_gate_msa"),
        _in("in_m_scale_mlp"), _in("in_m_gate_mlp"),
    )


# Build one LoRA adapter from oracle-dumped A/B (moments zeroed; not used in bwd).
def _adapter(tag: String, in_f: Int, out_f: Int) raises -> LoraAdapter:
    var z_a = List[Float32]()
    for _ in range(RANK * in_f):
        z_a.append(Float32(0.0))
    var z_b = List[Float32]()
    for _ in range(out_f * RANK):
        z_b.append(Float32(0.0))
    return LoraAdapter(
        _in("in_l" + tag + "_A"), _in("in_l" + tag + "_B"),
        RANK, in_f, out_f, LSCALE,
        z_a.copy(), z_a.copy(), z_b.copy(), z_b.copy(),
    )


def _load_lora() raises -> ZImageBlockLora:
    return ZImageBlockLora(
        Optional[LoraAdapter](_adapter("to_q", D, D)),
        Optional[LoraAdapter](_adapter("to_k", D, D)),
        Optional[LoraAdapter](_adapter("to_v", D, D)),
        Optional[LoraAdapter](_adapter("to_out", D, D)),
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
    print("==== l2p block_lora_parity (Z-Image L2P main block + LoRA vs torch) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " S=", S, " F=", F, " RANK=", RANK)

    var x = _in("in_x")
    var w = _load_weights(ctx)
    var mv = _load_mod()
    var lora = _load_lora()

    var cos = Tensor.from_host(_in("in_cos"), [S * H, HALF], STDtype.F32, ctx)
    var sin = Tensor.from_host(_in("in_sin"), [S * H, HALF], STDtype.F32, ctx)

    # ── forward ──
    var fwd = zimage_block_lora_forward[H, Dh, S](
        x.copy(), w, mv, lora, cos, sin, D, F, EPS, ctx,
    )

    var harness = ParityHarness()
    var allok = True

    print("")
    print("---- forward output vs torch ----")
    _check(harness, "out", fwd.out, _in("ref_out"), allok)

    # ── backward ──
    var d_out = _in("in_d_out")
    var bg = zimage_block_lora_backward[H, Dh, S](
        d_out, w, mv, lora, fwd.saved, cos, sin, D, F, EPS, ctx,
    )
    var g = bg.base.copy()

    print("")
    print("---- input grad vs torch ----")
    _check(harness, "d_x", g.d_x, _in("ref_d_x"), allok)

    # NOTE: the LoRA backward (lora_block.mojo) does NOT materialize the frozen
    # base projection-matrix grads (d_wq/d_wk/d_wv/d_wo/d_w1/d_w3/d_w2 are empty —
    # base weights are FROZEN during LoRA training, only d_x flows through them).
    # The trainable NORM grads (norms ARE trained) + RAW mod-vec grads + LoRA d_A/d_B
    # are materialized and gated below.
    print("")
    print("---- trainable norm-weight grads vs torch ----")
    _check(harness, "d_n1     ", g.d_n1, _in("ref_d_n1"), allok)
    _check(harness, "d_q_norm ", g.d_q_norm, _in("ref_d_q_norm"), allok)
    _check(harness, "d_k_norm ", g.d_k_norm, _in("ref_d_k_norm"), allok)
    _check(harness, "d_n2     ", g.d_n2, _in("ref_d_n2"), allok)
    _check(harness, "d_fn1    ", g.d_fn1, _in("ref_d_fn1"), allok)
    _check(harness, "d_fn2    ", g.d_fn2, _in("ref_d_fn2"), allok)

    print("")
    print("---- RAW adaLN modulation-vector grads vs torch ----")
    _check(harness, "d_scale_msa", g.d_scale_msa, _in("ref_d_scale_msa"), allok)
    _check(harness, "d_gate_msa ", g.d_gate_msa, _in("ref_d_gate_msa"), allok)
    _check(harness, "d_scale_mlp", g.d_scale_mlp, _in("ref_d_scale_mlp"), allok)
    _check(harness, "d_gate_mlp ", g.d_gate_mlp, _in("ref_d_gate_mlp"), allok)

    print("")
    print("---- LoRA d_A / d_B (7 trained projections) vs torch ----")
    _check(harness, "d_to_q_A  ", bg.lora.d_a[SLOT_Q], _in("ref_dto_q_A"), allok)
    _check(harness, "d_to_q_B  ", bg.lora.d_b[SLOT_Q], _in("ref_dto_q_B"), allok)
    _check(harness, "d_to_k_A  ", bg.lora.d_a[SLOT_K], _in("ref_dto_k_A"), allok)
    _check(harness, "d_to_k_B  ", bg.lora.d_b[SLOT_K], _in("ref_dto_k_B"), allok)
    _check(harness, "d_to_v_A  ", bg.lora.d_a[SLOT_V], _in("ref_dto_v_A"), allok)
    _check(harness, "d_to_v_B  ", bg.lora.d_b[SLOT_V], _in("ref_dto_v_B"), allok)
    _check(harness, "d_to_out_A", bg.lora.d_a[SLOT_O], _in("ref_dto_out_A"), allok)
    _check(harness, "d_to_out_B", bg.lora.d_b[SLOT_O], _in("ref_dto_out_B"), allok)
    _check(harness, "d_w1_A    ", bg.lora.d_a[SLOT_W1], _in("ref_dw1_A"), allok)
    _check(harness, "d_w1_B    ", bg.lora.d_b[SLOT_W1], _in("ref_dw1_B"), allok)
    _check(harness, "d_w3_A    ", bg.lora.d_a[SLOT_W3], _in("ref_dw3_A"), allok)
    _check(harness, "d_w3_B    ", bg.lora.d_b[SLOT_W3], _in("ref_dw3_B"), allok)
    _check(harness, "d_w2_A    ", bg.lora.d_a[SLOT_W2], _in("ref_dw2_A"), allok)
    _check(harness, "d_w2_B    ", bg.lora.d_b[SLOT_W2], _in("ref_dw2_B"), allok)

    print("")
    if allok:
        print("VERDICT: PASS — L2P main block fwd+bwd (+LoRA) matches torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one output diverged (see FAIL lines above)")
