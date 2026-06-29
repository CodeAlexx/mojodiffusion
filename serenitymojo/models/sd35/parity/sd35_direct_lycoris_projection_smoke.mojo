# models/sd35/parity/sd35_direct_lycoris_projection_smoke.mojo
#
# SD3.5 direct DoRA/OFT smoke: byte preflight, compact slot construction,
# projection parity, AdamW movement, and save/readback. Live block lowering is
# compiled through train_sd35_real; this is the quick direct-state gate.

from std.collections import List
from std.math import sqrt
from std.gpu.host import DeviceContext

from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.training.dora_adapter import dora_forward, dora_backward, DoRAGrads
from serenitymojo.training.dora_save import read_dora_module
from serenitymojo.training.oft_onetrainer import oft_ot_forward, oft_ot_backward
from serenitymojo.training.flat_direct_lycoris_stack import (
    FlatDirectDoRAGrads, FlatDirectOFTGrads,
    flat_direct_dora_append_from_weight, flat_direct_oft_append,
    flat_direct_dora_forward_slot, flat_direct_dora_backward_slot,
    flat_direct_oft_forward_slot, flat_direct_oft_backward_slot,
)
from serenitymojo.models.sd35.sd35_stack_lora import (
    SD35_DIRECT_24_GIB, SLOTS_PER_BLOCK,
    SLOT_CTX_QKV, SLOT_CTX_PROJ, SLOT_CTX_FC1, SLOT_CTX_FC2,
    SLOT_X_QKV, SLOT_X_PROJ, SLOT_X_FC1, SLOT_X_FC2,
    empty_sd35_direct_dora_set, empty_sd35_direct_oft_set,
    sd35_direct_dense_carrier_bytes,
    sd35_direct_dora_trainable_bytes_estimate,
    sd35_direct_oft_trainable_bytes_estimate,
    sd35_direct_dora_preflight, sd35_direct_oft_preflight,
    sd35_direct_dora_trainable_bytes, sd35_direct_oft_trainable_bytes,
    sd35_direct_dora_grad_norm, sd35_direct_oft_grad_norm,
    sd35_direct_dora_adamw_step, sd35_direct_oft_adamw_step,
    sd35_direct_dora_zero_leg_l1, sd35_direct_oft_vec_l1,
    save_sd35_direct_dora, save_sd35_direct_oft,
)


comptime DORA_SAVE_PATH = "/tmp/sd35_direct_dora_smoke.safetensors"
comptime OFT_SAVE_PATH = "/tmp/sd35_direct_oft_smoke.safetensors"


def _randn(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _add_bias(var y: List[Float32], bias: List[Float32], M: Int, out_f: Int) raises -> List[Float32]:
    if len(y) != M * out_f or len(bias) != out_f:
        raise Error("test _add_bias: shape mismatch")
    for m in range(M):
        for o in range(out_f):
            y[m * out_f + o] += bias[o]
    return y^


def _cos(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error("cos: len mismatch")
    var dot = 0.0
    var na = 0.0
    var nb = 0.0
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    return dot / (sqrt(na) * sqrt(nb))


def _check(name: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        raise Error("check: len mismatch")
    var na = 0.0
    var nb = 0.0
    for i in range(len(a)):
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na < 1.0e-24 and nb < 1.0e-24:
        print("  ", name, " both zero")
        return
    var c = _cos(a, b)
    print("  ", name, " cos=", c)
    if c < 0.999999:
        raise Error(String("GATE FAIL: ") + name)


def _dims(slot: Int, D: Int, F: Int) -> Tuple[Int, Int]:
    if slot == SLOT_CTX_QKV or slot == SLOT_X_QKV:
        return (D, 3 * D)
    if slot == SLOT_CTX_FC1 or slot == SLOT_X_FC1:
        return (D, F)
    if slot == SLOT_CTX_FC2 or slot == SLOT_X_FC2:
        return (F, D)
    return (D, D)


def _prefix(bi: Int, slot: Int) -> String:
    var bp = String("transformer.joint_blocks.") + String(bi)
    if slot == SLOT_CTX_QKV:
        return bp + String(".context_block.attn.qkv")
    if slot == SLOT_CTX_PROJ:
        return bp + String(".context_block.attn.proj")
    if slot == SLOT_CTX_FC1:
        return bp + String(".context_block.mlp.fc1")
    if slot == SLOT_CTX_FC2:
        return bp + String(".context_block.mlp.fc2")
    if slot == SLOT_X_QKV:
        return bp + String(".x_block.attn.qkv")
    if slot == SLOT_X_PROJ:
        return bp + String(".x_block.attn.proj")
    if slot == SLOT_X_FC1:
        return bp + String(".x_block.mlp.fc1")
    return bp + String(".x_block.mlp.fc2")


def _check_real_bytes() raises:
    var depth = 38
    var D = 2432
    var F = 9728
    var rank = 16
    var targets = 2
    var dense = sd35_direct_dense_carrier_bytes(depth, D, F, targets)
    var dora_direct = sd35_direct_dora_trainable_bytes_estimate(depth, D, F, rank, targets, False)
    var oft_direct = sd35_direct_oft_trainable_bytes_estimate(depth, D, F, 4, targets)
    print("[sd35-direct-bytes] dense_carrier_bytes=", dense, " direct_dora_bytes=", dora_direct, " direct_oft_bytes=", oft_direct)
    if dense <= SD35_DIRECT_24_GIB:
        raise Error("SD3.5 dense carrier unexpectedly fits 24 GiB")
    if dora_direct >= SD35_DIRECT_24_GIB or oft_direct >= SD35_DIRECT_24_GIB:
        raise Error("SD3.5 direct trainable estimate exceeds 24 GiB")
    if sd35_direct_dora_preflight(depth, D, F, rank, targets, SD35_DIRECT_24_GIB, False) != dora_direct:
        raise Error("SD3.5 DoRA preflight mismatch")
    if sd35_direct_oft_preflight(depth, D, F, 4, targets, SD35_DIRECT_24_GIB) != oft_direct:
        raise Error("SD3.5 OFT preflight mismatch")


def main() raises:
    var ctx = DeviceContext()
    _check_real_bytes()

    var D = 32
    var F = 64
    var depth = 2
    var rank = 4
    var alpha = Float32(4.0)
    var block_size = 4
    var dora = empty_sd35_direct_dora_set()
    var oft = empty_sd35_direct_oft_set()
    for bi in range(depth):
        for slot in range(SLOTS_PER_BLOCK):
            var dims = _dims(slot, D, F)
            var W = _randn(dims[0] * dims[1], UInt64(1000 + bi * 31 + slot), Float32(0.25))
            flat_direct_dora_append_from_weight(
                dora, W.copy(), dims[0], dims[1], rank, alpha,
                _prefix(bi, slot), UInt64(2000 + bi * 31 + slot), False,
            )
            flat_direct_oft_append(oft, dims[0], dims[1], block_size, _prefix(bi, slot))

    if len(dora.ad) != depth * SLOTS_PER_BLOCK or len(oft.ad) != len(dora.ad):
        raise Error("SD3.5 compact direct slot count mismatch")
    print("[sd35-direct-stream-init] dora_slots=", len(dora.ad),
          " dora_bytes=", sd35_direct_dora_trainable_bytes(dora),
          " oft_bytes=", sd35_direct_oft_trainable_bytes(oft))

    var slot = SLOT_CTX_FC1
    var dims = _dims(slot, D, F)
    var M = 3
    var x = _randn(M * dims[0], UInt64(3001), Float32(0.7))
    var W = _randn(dims[0] * dims[1], UInt64(3002), Float32(0.2))
    var bias = _randn(dims[1], UInt64(3003), Float32(0.1))
    var d_y = _randn(M * dims[1], UInt64(3004), Float32(0.3))

    var yd = flat_direct_dora_forward_slot(dora, slot, x.copy(), W.copy(), M)
    var yd_ref = dora_forward(x.copy(), W.copy(), dora.ad[slot], M)
    yd = _add_bias(yd^, bias.copy(), M, dims[1])
    yd_ref = _add_bias(yd_ref^, bias.copy(), M, dims[1])
    _check(String("dora rectangular projection+bias"), yd, yd_ref)
    var dg = flat_direct_dora_backward_slot(dora, slot, d_y.copy(), x.copy(), W.copy(), M)
    var dg_ref = dora_backward(d_y.copy(), x.copy(), W.copy(), dora.ad[slot], M)
    _check(String("dora d_A"), dg.d_a, dg_ref.d_a)
    _check(String("dora d_B"), dg.d_b, dg_ref.d_b)
    _check(String("dora d_x"), dg.d_x, dg_ref.d_x)

    var dg_all = List[DoRAGrads]()
    for i in range(len(dora.ad)):
        ref ad = dora.ad[i]
        dg_all.append(DoRAGrads(_zeros(len(ad.a)), _zeros(len(ad.b)), _zeros(len(ad.m)), List[Float32]()))
    dg_all[slot] = DoRAGrads(dg.d_a.copy(), dg.d_b.copy(), dg.d_m.copy(), List[Float32]())
    var dora_grads = FlatDirectDoRAGrads(dg_all^)
    if sd35_direct_dora_grad_norm(dora_grads) <= 0.0:
        raise Error("SD3.5 DoRA grad norm stayed zero")
    var d_before = sd35_direct_dora_zero_leg_l1(dora)
    sd35_direct_dora_adamw_step(dora, dora_grads, 1, Float32(1.0e-3), Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.0))
    var d_after = sd35_direct_dora_zero_leg_l1(dora)
    print("  dora zero_leg_l1 ", d_before, " -> ", d_after)
    if not (d_before == 0.0 and d_after > 0.0):
        raise Error("SD3.5 DoRA direct set did not step")
    _ = save_sd35_direct_dora(dora, String(DORA_SAVE_PATH), ctx)
    var rb = read_dora_module(dora.prefix[slot], String(DORA_SAVE_PATH), ctx)
    if rb.in_f != dims[0] or rb.out_f != dims[1] or rb.rank != rank:
        raise Error("SD3.5 DoRA readback shape mismatch")

    var yo = flat_direct_oft_forward_slot(oft, slot, x.copy(), W.copy(), M)
    var yo_ref = oft_ot_forward(x.copy(), oft.ad[slot].vec.copy(), W.copy(), M, dims[0], dims[1], block_size, dims[0] // block_size)
    yo = _add_bias(yo^, bias.copy(), M, dims[1])
    yo_ref = _add_bias(yo_ref^, bias.copy(), M, dims[1])
    _check(String("oft rectangular projection+bias"), yo, yo_ref)
    var og = flat_direct_oft_backward_slot(oft, slot, d_y.copy(), x.copy(), W.copy(), M)
    var og_ref = oft_ot_backward(d_y.copy(), x.copy(), oft.ad[slot].vec.copy(), W.copy(), M, dims[0], dims[1], block_size, dims[0] // block_size)
    _check(String("oft d_vec"), og.d_vec, og_ref.d_vec)
    _check(String("oft d_x"), og.d_x, og_ref.d_x)

    var og_all = List[List[Float32]]()
    for i in range(len(oft.ad)):
        og_all.append(_zeros(len(oft.ad[i].vec)))
    og_all[slot] = og.d_vec.copy()
    var oft_grads = FlatDirectOFTGrads(og_all^)
    if sd35_direct_oft_grad_norm(oft_grads) <= 0.0:
        raise Error("SD3.5 OFT grad norm stayed zero")
    var o_before = sd35_direct_oft_vec_l1(oft)
    sd35_direct_oft_adamw_step(oft, oft_grads, 1, Float32(1.0e-3), Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.0))
    var o_after = sd35_direct_oft_vec_l1(oft)
    print("  oft vec_l1 ", o_before, " -> ", o_after)
    if not (o_before == 0.0 and o_after > 0.0):
        raise Error("SD3.5 OFT direct set did not step")
    _ = save_sd35_direct_oft(oft, String(OFT_SAVE_PATH), ctx)
    _ = SafeTensors.open(String(OFT_SAVE_PATH))

    print("ALL GATES PASS -- sd35_direct_lycoris_projection_smoke")
