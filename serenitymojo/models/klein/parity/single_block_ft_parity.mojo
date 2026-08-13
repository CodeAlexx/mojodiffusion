# serenitymojo/models/klein/parity/single_block_ft_parity.mojo
#
# PARITY GATE for the Klein SINGLE-STREAM block FULL-FINETUNE backward
# (single_block_ft_backward_dev): the trainable weight grads d_w1/d_w2 (F32,
# device) + the input grad d_x vs torch autograd, cos >= 0.999. The forward is
# the DEVICE-RESIDENT base block with NO adapters (full-FT has no LoRA); the
# oracle is the EXISTING base-only single_block_oracle.py (its refs ref_d_w1 /
# ref_d_w2 / ref_d_x are torch W.grad / x.grad with no adapter in the graph —
# exactly the FT surface; dW = d_yᵀx is what the F32 optimizer consumes).
# v1 FROZEN params (q/k rms scales, mod vecs) are NOT compared — the FT
# backward intentionally skips them (see single_block.mojo FT header).
#
# Run (oracle FIRST, as a SEPARATE command — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/SerenityTrainer/venv/bin/python \
#       serenitymojo/models/klein/parity/single_block_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build --optimization-level 2 --target-accelerator sm_120 \
#       -I . -I /home/alex/MOJO-libs -Xlinker -lm \
#       -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#       -Xlinker -rpath -Xlinker $PWD/serenitymojo/ops/cshim/lib \
#       -Xlinker -rpath -Xlinker /home/alex/.serenity/cudnn/lib \
#       serenitymojo/models/klein/parity/single_block_ft_parity.mojo \
#       -o /tmp/single_block_ft_parity
#   /tmp/single_block_ft_parity

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import alloc, ArcPointer
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.klein.lora_block import LoraAdapterDevice
from serenitymojo.models.klein.single_block import (
    SingleBlockWeights, SingleModVecs, single_modvecs_to_device,
    SingleBlockLoraDevice,
    single_block_lora_forward_device_resident,
    single_block_ft_backward_dev,
)

comptime TArc = ArcPointer[Tensor]
comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/klein/parity/"

# dims MUST match single_block_oracle.py
comptime H = 32
comptime Dh = 16
comptime D = H * Dh        # 512
comptime S = 6
comptime F = 24
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


def _chunk_host(src: List[Float32], idx: Int, d: Int) -> List[Float32]:
    var o = List[Float32]()
    var base = idx * d
    for i in range(d):
        o.append(src[base + i])
    return o^


def _load_weights(ctx: DeviceContext) raises -> SingleBlockWeights:
    return SingleBlockWeights(
        _in("in_w_w1"), _in("in_w_w2"),
        _in("in_w_q_norm"), _in("in_w_k_norm"),
        D, F, Dh, ctx,
    )


def _load_mod() raises -> SingleModVecs:
    return SingleModVecs(
        _in("in_m_shift"), _in("in_m_scale"), _in("in_m_gate"),
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
    print("==== single_block_ft_parity (Klein single-block full-FT dW backward vs torch) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " S=", S, " F=", F)

    # ── load inputs (byte-identical to the oracle) ──
    var x = _in("in_x")
    var w = _load_weights(ctx)
    var mv = _load_mod()
    var mv_dev = single_modvecs_to_device(mv, D, ctx)
    var cos_h = _in("in_cos")
    var sin_h = _in("in_sin")
    var cos = Tensor.from_host(cos_h, [S * H, Dh // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(sin_h, [S * H, Dh // 2], STDtype.F32, ctx)

    # NO adapters: the full-FT forward is the base block.
    var lora = SingleBlockLoraDevice(
        Optional[LoraAdapterDevice](None), Optional[LoraAdapterDevice](None),
    )

    # ── forward (device-resident base; saves the tape the FT backward consumes) ──
    var x_t = TArc(Tensor.from_host(x, [S, D], STDtype.F32, ctx))
    var fwd = single_block_lora_forward_device_resident[H, Dh, S](
        x_t, w, mv_dev, lora, cos, sin, D, F, EPS, ctx,
    )

    var harness = ParityHarness()
    var allok = True

    print("")
    print("---- forward output vs torch ----")
    var out_h = fwd.out[].to_host(ctx)
    _check(harness, "out", out_h, _in("ref_out"), allok)

    # ── FT backward: d_x + d_w1 + d_w2 ──
    var d_out_h = _in("in_d_out")
    var d_out = TArc(Tensor.from_host(d_out_h, [S, D], STDtype.F32, ctx))
    var g = single_block_ft_backward_dev[H, Dh, S, True](
        d_out, w, mv_dev, fwd.saved, cos, sin, D, F, EPS, ctx,
    )

    print("")
    print("---- input grad d_x vs torch ----")
    var d_x_h = g.d_x[].to_host(ctx)
    _check(harness, "d_x", d_x_h, _in("ref_d_x"), allok)

    print("")
    print("---- trainable weight grads d_W vs torch (w1, w2) ----")
    var d_w1_h = g.dw[0][].to_host(ctx)
    _check(harness, "d_w1", d_w1_h, _in("ref_d_w1"), allok)
    var d_w2_h = g.dw[1][].to_host(ctx)
    _check(harness, "d_w2", d_w2_h, _in("ref_d_w2"), allok)

    # ── SURF (Phase B): q/k rms-scale grads ──
    print("")
    print("---- q/k rms-scale grads vs torch (Phase B) ----")
    _check(harness, "d_qnorm", g.d_qk[0][].to_host(ctx), _in("ref_d_qnorm"), allok)
    _check(harness, "d_knorm", g.d_qk[1][].to_host(ctx), _in("ref_d_knorm"), allok)

    # ── SURF (Phase B): shared single_stream_modulation.lin grad, per chunk ──
    print("")
    print("---- shared modulation-linear grads vs torch, per chunk (Phase B) ----")
    var m_flat = g.mod_flat[0][].to_host(ctx)   # [3D] = [shift|scale|gate]
    _check(harness, "d_shift", _chunk_host(m_flat, 0, D), _in("ref_d_shift"), allok)
    _check(harness, "d_scale", _chunk_host(m_flat, 1, D), _in("ref_d_scale"), allok)
    _check(harness, "d_gate", _chunk_host(m_flat, 2, D), _in("ref_d_gate"), allok)

    print("")
    if allok:
        print("VERDICT: PASS — Klein single-block full-FT backward matches torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one grad diverged (see FAIL lines above)")
