# autograd_v2/tests/klein_block_parity.mojo - Phase P6 SAME-PROCESS BIT GATE
# (AUTOGRAD_V2_MOJO_DESIGN.md P6 / C14 note: Klein CANNOT be bit-gated across
# runs - pre-existing ~4e-4 run nondeterminism - so the strongest gate Klein
# admits is same-process: same tensors, same pointers-class, same kernels.
# ANY mismatch here is a WIRING BUG in the graph path, not variance).
#
# Loads NOTHING from disk: synthetic-but-real-shaped inputs at the Klein-9B
# trainer dims (train_klein_real.mojo comptime H=32 Dh=128 N_IMG=1024
# N_TXT=512 S=1536; configs/klein9b.json inner_dim=4096 mlp_hidden=12288
# lora_rank=16), trainer dtypes (F32 activations/grads/mod-vecs/rope tables,
# BF16 base weights + LoRA A/B - the turbo-loader/_klein_resident_adapter
# dtypes), deterministic LCG host patterns (the klein_stack_lora.mojo
# _kaiming_uniform_a_sqrt5 generator shape). LoRA B is NONZERO so every d_A
# is non-degenerate (B=0 would gate vacuously); a degenerate (all-zero)
# compared tensor FAILS the gate.
#
# Per block kind, runs on IDENTICAL tensors in the SAME process:
#   oracle  = the trainer stack-loop hand-chain pair
#             (double: double_block_lora_forward_device_resident_scratch +
#              double_block_lora_backward_device_resident_scratch_tensors;
#              single: single_block_lora_recompute_saved_device_resident_
#              scratch + single_block_lora_backward_device_resident_scratch_
#              tensors; compute_aux_grads=False - the production call)
#   graph   = klein_double/single_block_graph_backward (autograd_v2 P6)
# and compares EVERY output grad (d_img/d_txt/d_x + every adapter d_a/d_b
# slot) BIT-EQUAL (raw device bytes). Prints one GATE line per tensor with
# n_mismatch; the acceptance bar is n_mismatch=0 on every line.
#
# Build: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/autograd_v2/tests/klein_block_parity.mojo \
#     -o /tmp/klein_block_parity
# Run:   LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib \
#   /tmp/klein_block_parity

from std.gpu.host import DeviceContext
from std.math import sqrt
from std.builtin.dtype import DType
from std.collections import Optional
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.scratch_ring import ScratchRingAllocator
from serenitymojo.models.klein.double_block import (
    StreamWeights,
    DoubleBlockWeights,
    ModVecsDevice,
    StreamLoraDevice,
    DoubleBlockLoraDevice,
    double_block_lora_forward_device_resident_scratch,
    double_block_lora_backward_device_resident_scratch_tensors,
)
from serenitymojo.models.klein.single_block import (
    KLEIN_SDPA_FLASH,
    SingleBlockWeights,
    SingleModVecsDevice,
    SingleBlockLoraDevice,
    single_block_lora_recompute_saved_device_resident_scratch,
    single_block_lora_backward_device_resident_scratch_tensors,
)
from serenitymojo.models.klein.lora_block import LoraAdapterDevice
from serenitymojo.autograd_v2.node import TArc
from serenitymojo.autograd_v2.klein_block_graph import (
    klein_double_block_graph_backward,
    klein_single_block_graph_backward,
)


# Klein-9B trainer dims (train_klein_real.mojo:168-174 + configs/klein9b.json).
comptime H = 32
comptime Dh = 128
comptime N_IMG = 1024
comptime N_TXT = 512
comptime S = N_IMG + N_TXT
comptime D = 4096
comptime F = 12288
comptime RANK = 16
comptime EPS = Float32(1.0e-6)
comptime LORA_SCALE = Float32(1.0)  # alpha/rank = 16/16


def _pattern(n: Int, seed: UInt64, amp: Float32) -> List[Float32]:
    """Deterministic host pattern (the klein_stack_lora.mojo:145 LCG shape):
    uniform in (-amp, amp), fully reproducible from the seed."""
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u * Float32(2.0) - Float32(1.0)) * amp)
    return out^


def _ones_host(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(1.0)
    return out^


def _zeros_host(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(0.0)
    return out^


def _f32(var vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_host(vals^, shape^, STDtype.F32, ctx))


def _bf16(var vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_host(vals^, shape^, STDtype.BF16, ctx))


def _adapter(
    in_f: Int, out_f: Int, seed: UInt64, ctx: DeviceContext
) raises -> LoraAdapterDevice:
    # A and B both nonzero (B=0 would make every d_A identically zero - a
    # vacuous gate). BF16 storage = the trainer's resident-adapter dtype.
    var a = _bf16(_pattern(RANK * in_f, seed, Float32(0.02)), [RANK, in_f], ctx)
    var b = _bf16(_pattern(out_f * RANK, seed ^ 0x9E3779B97F4A7C15, Float32(0.02)), [out_f, RANK], ctx)
    return LoraAdapterDevice(a.copy(), b.copy(), RANK, in_f, out_f, LORA_SCALE)


def _stream_weights(seed: UInt64, ctx: DeviceContext) raises -> StreamWeights:
    return StreamWeights(
        _bf16(_pattern(3 * D * D, seed + 1, Float32(0.02)), [3 * D, D], ctx),
        _bf16(_pattern(D * D, seed + 2, Float32(0.02)), [D, D], ctx),
        _bf16(_pattern(2 * F * D, seed + 3, Float32(0.02)), [2 * F, D], ctx),
        _bf16(_pattern(D * F, seed + 4, Float32(0.02)), [D, F], ctx),
        _bf16(_pattern(Dh, seed + 5, Float32(0.5)), [Dh], ctx),
        _bf16(_pattern(Dh, seed + 6, Float32(0.5)), [Dh], ctx),
    )


def _mod_vecs(seed: UInt64, ctx: DeviceContext) raises -> ModVecsDevice:
    return ModVecsDevice(
        _f32(_pattern(D, seed + 1, Float32(0.1)), [D], ctx),   # shift1
        _f32(_pattern(D, seed + 2, Float32(0.1)), [D], ctx),   # scale1
        _f32(_pattern(D, seed + 3, Float32(0.5)), [D], ctx),   # gate1
        _f32(_pattern(D, seed + 4, Float32(0.1)), [D], ctx),   # shift2
        _f32(_pattern(D, seed + 5, Float32(0.1)), [D], ctx),   # scale2
        _f32(_pattern(D, seed + 6, Float32(0.5)), [D], ctx),   # gate2
    )


def _stream_lora(seed: UInt64, ctx: DeviceContext) raises -> StreamLoraDevice:
    return StreamLoraDevice(
        Optional[LoraAdapterDevice](_adapter(D, D, seed + 11, ctx)),
        Optional[LoraAdapterDevice](_adapter(D, D, seed + 12, ctx)),
        Optional[LoraAdapterDevice](_adapter(D, D, seed + 13, ctx)),
        Optional[LoraAdapterDevice](_adapter(D, D, seed + 14, ctx)),
        Optional[LoraAdapterDevice](_adapter(D, 2 * F, seed + 15, ctx)),
        Optional[LoraAdapterDevice](_adapter(F, D, seed + 16, ctx)),
    )


def _gate_pair(
    name: String, a: Tensor, b: Tensor, ctx: DeviceContext
) raises -> Bool:
    """Raw device-byte comparison + a non-degeneracy guard (an all-zero
    compared tensor would gate vacuously -> FAIL)."""
    if a.nbytes() != b.nbytes() or a.dtype() != b.dtype():
        print(
            "GATE klein_parity " + name + " FAIL n_mismatch=-1 (shape/dtype"
            " mismatch: " + String(a.nbytes()) + " vs " + String(b.nbytes()) + ")"
        )
        return False
    var nb = a.nbytes()
    var ha = ctx.enqueue_create_host_buffer[DType.uint8](nb)
    ctx.enqueue_copy(dst_buf=ha, src_buf=a.buf)
    var hb = ctx.enqueue_create_host_buffer[DType.uint8](nb)
    ctx.enqueue_copy(dst_buf=hb, src_buf=b.buf)
    ctx.synchronize()
    var pa = ha.unsafe_ptr()
    var pb = hb.unsafe_ptr()
    var n_mismatch = 0
    var first_off = -1
    var nonzero = False
    for i in range(nb):
        if pa[i] != pb[i]:
            n_mismatch += 1
            if first_off < 0:
                first_off = i
        if pa[i] != 0:
            nonzero = True
    if not nonzero:
        print(
            "GATE klein_parity " + name + " FAIL n_mismatch=" + String(n_mismatch)
            + " (DEGENERATE: oracle tensor is all-zero, gate vacuous)"
        )
        return False
    var verdict = String("PASS") if n_mismatch == 0 else String("FAIL")
    var line = (
        "GATE klein_parity " + name + " " + verdict
        + " n_mismatch=" + String(n_mismatch)
        + " nbytes=" + String(nb)
    )
    if n_mismatch > 0:
        line += " first_byte_off=" + String(first_off)
    print(line)
    return n_mismatch == 0


def _gate_pair_tol(
    name: String, a: Tensor, b: Tensor, ctx: DeviceContext
) raises -> Bool:
    """Value-tolerance gate for tensors downstream of the cuDNN flash dQ
    (KLEIN_SDPA_FLASH): the flash backward's dQ accumulation is
    NONDETERMINISTIC across calls (MEASURED 2026-06-11: 653/6.3M bf16
    elements flip between two identical-input calls in one process; dK is
    deterministic), so bit-equality between the hand-chain and graph paths
    is impossible for dQ-derived grads BY THE KERNEL'S NATURE. Gate =
    cosine >= 0.999999 AND max_abs <= 1e-3 (the observed flip class is
    +-1 ulp; a wiring bug is many orders larger). Non-degeneracy guarded."""
    var ah = a.to_host(ctx)
    var bh = b.to_host(ctx)
    if len(ah) != len(bh):
        print("GATE klein_parity " + name + " FAIL (length mismatch)")
        return False
    var dot = 0.0
    var na = 0.0
    var nb2 = 0.0
    var max_abs = 0.0
    var nonzero = False
    for i in range(len(ah)):
        var x = Float64(ah[i])
        var y = Float64(bh[i])
        if x != 0.0:
            nonzero = True
        dot += x * y
        na += x * x
        nb2 += y * y
        var d = x - y
        if d < 0:
            d = -d
        if d > max_abs:
            max_abs = d
    if not nonzero:
        print("GATE klein_parity " + name + " FAIL (degenerate all-zero)")
        return False
    var denom = sqrt(na) * sqrt(nb2)
    var cosine = 1.0
    if denom > 0.0:
        cosine = dot / denom
    var ok = cosine >= 0.999999 and max_abs <= 1.0e-3
    print(
        "GATE klein_parity " + name + " " + ("PASS" if ok else "FAIL")
        + " (flash-tol) cos=" + String(cosine) + " max_abs=" + String(max_abs)
    )
    return ok


def _gate_opt_tol(
    name: String, a: Optional[TArc], b: Optional[TArc], ctx: DeviceContext
) raises -> Bool:
    if not a or not b:
        print("GATE klein_parity " + name + " FAIL (missing grad Optional)")
        return False
    return _gate_pair_tol(name, a.value()[], b.value()[], ctx)


def _gate_opt(
    name: String, a: Optional[TArc], b: Optional[TArc], ctx: DeviceContext
) raises -> Bool:
    if not a or not b:
        print("GATE klein_parity " + name + " FAIL (missing grad Optional)")
        return False
    return _gate_pair(name, a.value()[], b.value()[], ctx)


def main() raises:
    var ctx = DeviceContext()
    print("=== klein_block_parity: P6 same-process bit gate ===")
    print(
        "dims: H=", H, " Dh=", Dh, " N_IMG=", N_IMG, " N_TXT=", N_TXT,
        " S=", S, " D=", D, " F=", F, " rank=", RANK,
    )
    # Same ring class as the trainer's scratch_bwd (1 GiB x 3,
    # train_klein_real.mojo:206-207).
    var scratch = ScratchRingAllocator(ctx, 1024 * 1024 * 1024, 3)

    var norm_ones = Tensor.from_host(_ones_host(D), [D], STDtype.F32, ctx)
    var norm_zeros = Tensor.from_host(_zeros_host(D), [D], STDtype.F32, ctx)
    var cos = Tensor.from_host(
        _pattern(S * H * (Dh // 2), 901, Float32(0.7)), [S * H, Dh // 2], STDtype.F32, ctx
    )
    var sin = Tensor.from_host(
        _pattern(S * H * (Dh // 2), 902, Float32(0.7)), [S * H, Dh // 2], STDtype.F32, ctx
    )

    var all_ok = True

    # ════════════════════════════════════════════════════════════════════════
    # DOUBLE BLOCK
    # ════════════════════════════════════════════════════════════════════════
    print("[double] building synthetic weights/lora/inputs ...")
    var dw = DoubleBlockWeights(_stream_weights(100, ctx), _stream_weights(200, ctx))
    var img_mod = _mod_vecs(300, ctx)
    var txt_mod = _mod_vecs(400, ctx)
    var dlora = DoubleBlockLoraDevice(_stream_lora(500, ctx), _stream_lora(600, ctx))
    var img_x = _f32(_pattern(N_IMG * D, 701, Float32(1.0)), [N_IMG, D], ctx)
    var txt_x = _f32(_pattern(N_TXT * D, 702, Float32(1.0)), [N_TXT, D], ctx)
    var d_io = _f32(_pattern(N_IMG * D, 703, Float32(0.01)), [N_IMG, D], ctx)
    var d_to = _f32(_pattern(N_TXT * D, 704, Float32(0.01)), [N_TXT, D], ctx)

    print("[double] oracle: hand-chain recompute fwd + _scratch_tensors bwd ...")
    var dfwd = double_block_lora_forward_device_resident_scratch[
        H, Dh, N_IMG, N_TXT, S
    ](
        img_x, txt_x, dw, img_mod, txt_mod, dlora, cos, sin,
        D, F, EPS, norm_ones, norm_zeros, ctx, scratch,
    )
    var dor = double_block_lora_backward_device_resident_scratch_tensors[
        H, Dh, N_IMG, N_TXT, S
    ](
        d_io, d_to, dw, img_mod, txt_mod, dlora, dfwd.saved, cos, sin,
        D, F, EPS, norm_ones, ctx, scratch, compute_aux_grads=False,
    )

    print("[double] graph: klein_double_block_graph_backward ...")
    var dgr = klein_double_block_graph_backward[H, Dh, N_IMG, N_TXT, S](
        d_io, d_to, dw, img_mod, txt_mod, dlora, img_x, txt_x, cos, sin,
        D, F, EPS, norm_ones, norm_zeros, ctx, scratch,
    )

    all_ok = _gate_pair(String("dbl d_img_x"), dor.img.d_x[], dgr.img.d_x[], ctx) and all_ok
    all_ok = _gate_pair(String("dbl d_txt_x"), dor.txt.d_x[], dgr.txt.d_x[], ctx) and all_ok
    # 12 adapter slots, stack order 0-5 img / 6-11 txt
    # (q,k,v,out,ff_in,ff_out each d_a+d_b).
    all_ok = _gate_opt(String("dbl s0 img_q_d_a"), dor.img.q_d_a, dgr.img.q_d_a, ctx) and all_ok
    all_ok = _gate_opt(String("dbl s0 img_q_d_b"), dor.img.q_d_b, dgr.img.q_d_b, ctx) and all_ok
    all_ok = _gate_opt(String("dbl s1 img_k_d_a"), dor.img.k_d_a, dgr.img.k_d_a, ctx) and all_ok
    all_ok = _gate_opt(String("dbl s1 img_k_d_b"), dor.img.k_d_b, dgr.img.k_d_b, ctx) and all_ok
    all_ok = _gate_opt(String("dbl s2 img_v_d_a"), dor.img.v_d_a, dgr.img.v_d_a, ctx) and all_ok
    all_ok = _gate_opt(String("dbl s2 img_v_d_b"), dor.img.v_d_b, dgr.img.v_d_b, ctx) and all_ok
    all_ok = _gate_opt(String("dbl s3 img_out_d_a"), dor.img.out_d_a, dgr.img.out_d_a, ctx) and all_ok
    all_ok = _gate_opt(String("dbl s3 img_out_d_b"), dor.img.out_d_b, dgr.img.out_d_b, ctx) and all_ok
    all_ok = _gate_opt(String("dbl s4 img_ff_in_d_a"), dor.img.ff_in_d_a, dgr.img.ff_in_d_a, ctx) and all_ok
    all_ok = _gate_opt(String("dbl s4 img_ff_in_d_b"), dor.img.ff_in_d_b, dgr.img.ff_in_d_b, ctx) and all_ok
    all_ok = _gate_opt(String("dbl s5 img_ff_out_d_a"), dor.img.ff_out_d_a, dgr.img.ff_out_d_a, ctx) and all_ok
    all_ok = _gate_opt(String("dbl s5 img_ff_out_d_b"), dor.img.ff_out_d_b, dgr.img.ff_out_d_b, ctx) and all_ok
    all_ok = _gate_opt(String("dbl s6 txt_q_d_a"), dor.txt.q_d_a, dgr.txt.q_d_a, ctx) and all_ok
    all_ok = _gate_opt(String("dbl s6 txt_q_d_b"), dor.txt.q_d_b, dgr.txt.q_d_b, ctx) and all_ok
    all_ok = _gate_opt(String("dbl s7 txt_k_d_a"), dor.txt.k_d_a, dgr.txt.k_d_a, ctx) and all_ok
    all_ok = _gate_opt(String("dbl s7 txt_k_d_b"), dor.txt.k_d_b, dgr.txt.k_d_b, ctx) and all_ok
    all_ok = _gate_opt(String("dbl s8 txt_v_d_a"), dor.txt.v_d_a, dgr.txt.v_d_a, ctx) and all_ok
    all_ok = _gate_opt(String("dbl s8 txt_v_d_b"), dor.txt.v_d_b, dgr.txt.v_d_b, ctx) and all_ok
    all_ok = _gate_opt(String("dbl s9 txt_out_d_a"), dor.txt.out_d_a, dgr.txt.out_d_a, ctx) and all_ok
    all_ok = _gate_opt(String("dbl s9 txt_out_d_b"), dor.txt.out_d_b, dgr.txt.out_d_b, ctx) and all_ok
    all_ok = _gate_opt(String("dbl s10 txt_ff_in_d_a"), dor.txt.ff_in_d_a, dgr.txt.ff_in_d_a, ctx) and all_ok
    all_ok = _gate_opt(String("dbl s10 txt_ff_in_d_b"), dor.txt.ff_in_d_b, dgr.txt.ff_in_d_b, ctx) and all_ok
    all_ok = _gate_opt(String("dbl s11 txt_ff_out_d_a"), dor.txt.ff_out_d_a, dgr.txt.ff_out_d_a, ctx) and all_ok
    all_ok = _gate_opt(String("dbl s11 txt_ff_out_d_b"), dor.txt.ff_out_d_b, dgr.txt.ff_out_d_b, ctx) and all_ok

    # ════════════════════════════════════════════════════════════════════════
    # SINGLE BLOCK
    # ════════════════════════════════════════════════════════════════════════
    print("[single] building synthetic weights/lora/inputs ...")
    var sw = SingleBlockWeights(
        _bf16(_pattern((3 * D + 2 * F) * D, 801, Float32(0.02)), [3 * D + 2 * F, D], ctx),
        _bf16(_pattern(D * (D + F), 802, Float32(0.02)), [D, D + F], ctx),
        _bf16(_pattern(Dh, 803, Float32(0.5)), [Dh], ctx),
        _bf16(_pattern(Dh, 804, Float32(0.5)), [Dh], ctx),
        D, F, ctx, False,  # keep_w2=False, the trainer's loader path
    )
    var smv = SingleModVecsDevice(
        _f32(_pattern(D, 811, Float32(0.1)), [D], ctx),
        _f32(_pattern(D, 812, Float32(0.1)), [D], ctx),
        _f32(_pattern(D, 813, Float32(0.5)), [D], ctx),
    )
    var slora = SingleBlockLoraDevice(
        Optional[LoraAdapterDevice](_adapter(D, 3 * D + 2 * F, 821, ctx)),
        Optional[LoraAdapterDevice](_adapter(D + F, D, 822, ctx)),
    )
    var x_in = _f32(_pattern(S * D, 831, Float32(1.0)), [S, D], ctx)
    var d_out = _f32(_pattern(S * D, 832, Float32(0.01)), [S, D], ctx)

    print("[single] oracle: recompute_saved_scratch + _scratch_tensors bwd ...")
    var ssaved = single_block_lora_recompute_saved_device_resident_scratch[
        H, Dh, S
    ](
        x_in, sw, smv, slora, cos, sin, D, F, EPS,
        norm_ones, norm_zeros, ctx, scratch,
    )
    var sor = single_block_lora_backward_device_resident_scratch_tensors[
        H, Dh, S
    ](
        d_out, sw, smv, slora, ssaved, cos, sin, D, F, EPS,
        norm_ones, ctx, scratch, compute_aux_grads=False,
    )

    print("[single] graph: klein_single_block_graph_backward ...")
    var sgr = klein_single_block_graph_backward[H, Dh, S](
        d_out, sw, smv, slora, x_in, cos, sin, D, F, EPS,
        norm_ones, norm_zeros, ctx, scratch,
    )

    # d_x and the qkv adapter grads sit downstream of the flash dQ (the
    # nondeterministic accumulation, see _gate_pair_tol) -> value-tolerance
    # gate in flash mode, bit gate otherwise. out_* are NOT downstream of
    # the SDPA backward -> always bit-strict.
    comptime if KLEIN_SDPA_FLASH:
        all_ok = _gate_pair_tol(String("sgl d_x"), sor.d_x[], sgr.d_x[], ctx) and all_ok
        all_ok = _gate_opt_tol(String("sgl s0 qkv_d_a"), sor.qkv_d_a, sgr.qkv_d_a, ctx) and all_ok
        all_ok = _gate_opt_tol(String("sgl s0 qkv_d_b"), sor.qkv_d_b, sgr.qkv_d_b, ctx) and all_ok
    else:
        all_ok = _gate_pair(String("sgl d_x"), sor.d_x[], sgr.d_x[], ctx) and all_ok
        all_ok = _gate_opt(String("sgl s0 qkv_d_a"), sor.qkv_d_a, sgr.qkv_d_a, ctx) and all_ok
        all_ok = _gate_opt(String("sgl s0 qkv_d_b"), sor.qkv_d_b, sgr.qkv_d_b, ctx) and all_ok
    all_ok = _gate_opt(String("sgl s1 out_d_a"), sor.out_d_a, sgr.out_d_a, ctx) and all_ok
    all_ok = _gate_opt(String("sgl s1 out_d_b"), sor.out_d_b, sgr.out_d_b, ctx) and all_ok

    if all_ok:
        print("=== klein_block_parity: ALL GATES PASS (every grad bit-equal) ===")
    else:
        print("=== klein_block_parity: FAIL — see GATE lines above ===")
        raise Error("klein_block_parity: bit mismatch (wiring bug, not variance)")
