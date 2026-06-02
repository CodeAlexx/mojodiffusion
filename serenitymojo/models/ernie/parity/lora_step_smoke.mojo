# serenitymojo/models/ernie/parity/lora_step_smoke.mojo
#
# END-TO-END LoRA TRAINING STEP smoke for ERNIE-Image (mirrors the Klein LoRA
# proof). Uses the SAME small parity dims (L=3, S=8, reduced F) and the SAME base
# inputs (stack_oracle.py .bin files) so it is cheap on the shared 3090, but the
# pipeline is the REAL one: build adapters (B=0 init) -> ernie_stack_lora_forward
# -> MSE-flow-style upstream grad -> ernie_stack_lora_backward -> global-norm clip
# -> ernie_lora_adamw_step -> confirm LoRA-B moves 0 -> nonzero (the adapter is
# LEARNING) -> save_ernie_lora -> load_ernie_lora_resume -> assert A/B BYTE-EXACT
# round-trip. This is the "complete LoRA training STEP" deliverable (E4 gate).
#
# Run (oracle FIRST for the base inputs; SEPARATE command):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/ernie/parity/stack_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/models/ernie/parity/lora_step_smoke.mojo -o /tmp/ernie_lora_step
#   /tmp/ernie_lora_step

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import alloc, ArcPointer
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.ernie.weights import ErnieBlockWeights, ErnieStackBase
from serenitymojo.models.ernie.block import ErnieModVecs
from serenitymojo.models.ernie.lora_block import ERNIE_SLOTS
from serenitymojo.models.ernie.ernie_stack_lora import (
    ErnieLoraSet, ErnieLoraGrads, build_ernie_lora_set, ernie_lora_get,
    ernie_stack_lora_forward, ernie_stack_lora_backward,
    ernie_lora_adamw_step, save_ernie_lora, load_ernie_lora_resume,
)

comptime TArc = ArcPointer[Tensor]
comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/ernie/parity/"

comptime H = 32
comptime Dh = 128
comptime D = H * Dh        # 4096
comptime N_IMG = 6
comptime N_TXT = 2
comptime S = N_IMG + N_TXT # 8
comptime F = 96
comptime IN_CH = 16
comptime TEXT_IN = 24
comptime OUT_CH = 16
comptime L = 3
comptime EPS = Float32(1e-06)
comptime RANK = 8
comptime ALPHA = Float32(16.0)
comptime SAVE_PATH = "/tmp/ernie_lora_smoke.safetensors"


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing ref (run stack_oracle.py first): ") + path)
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


def _load_block(l: Int, ctx: DeviceContext) raises -> ErnieBlockWeights:
    var pre = String("in_blk") + String(l) + String("_")
    return ErnieBlockWeights(
        _t1(_in(pre + String("sa_norm")), D, ctx),
        _t2(_in(pre + String("wq")), D, D, ctx),
        _t2(_in(pre + String("wk")), D, D, ctx),
        _t2(_in(pre + String("wv")), D, D, ctx),
        _t2(_in(pre + String("wo")), D, D, ctx),
        _t1(_in(pre + String("q_norm")), Dh, ctx),
        _t1(_in(pre + String("k_norm")), Dh, ctx),
        _t1(_in(pre + String("mlp_norm")), D, ctx),
        _t2(_in(pre + String("wgate")), F, D, ctx),
        _t2(_in(pre + String("wup")), F, D, ctx),
        _t2(_in(pre + String("wdown")), D, F, ctx),
    )


def _load_base(ctx: DeviceContext) raises -> ErnieStackBase:
    var dummy_d = _t1(_in("in_f_scale"), D, ctx)
    return ErnieStackBase(
        _t2(_in("in_patch_w"), D, IN_CH, ctx),
        _t1(_in("in_patch_b"), D, ctx),
        _t2(_in("in_text_proj"), D, TEXT_IN, ctx),
        dummy_d, dummy_d, dummy_d, dummy_d,
        dummy_d, dummy_d,
        dummy_d, dummy_d,
        _t2(_in("in_final_lin"), OUT_CH, D, ctx),
        _t1(_in("in_final_lin_b"), OUT_CH, ctx),
    )


def _load_mod() raises -> ErnieModVecs:
    return ErnieModVecs(
        _in("in_m_shift_msa"), _in("in_m_scale_msa"), _in("in_m_gate_msa"),
        _in("in_m_shift_mlp"), _in("in_m_scale_mlp"), _in("in_m_gate_mlp"),
    )


def _absum(v: List[Float32]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(v)):
        var x = v[i]
        s += x if x >= 0.0 else -x
    return s


# host global L2 norm over the flat LoRA grads (the clip basis). Matches the
# sum-of-squares contract of optim.clip_grads_by_global_norm.
def _global_norm(grads: ErnieLoraGrads) -> Float32:
    var ss = Float32(0.0)
    var n = len(grads.d_a)
    for i in range(n):
        for j in range(len(grads.d_a[i])):
            ss += grads.d_a[i][j] * grads.d_a[i][j]
        for j in range(len(grads.d_b[i])):
            ss += grads.d_b[i][j] * grads.d_b[i][j]
    return sqrt(ss)


def _clip(mut grads: ErnieLoraGrads, max_norm: Float32):
    var gn = _global_norm(grads)
    if gn <= max_norm or gn == 0.0:
        return
    var s = max_norm / gn
    var n = len(grads.d_a)
    for i in range(n):
        for j in range(len(grads.d_a[i])):
            grads.d_a[i][j] = grads.d_a[i][j] * s
        for j in range(len(grads.d_b[i])):
            grads.d_b[i][j] = grads.d_b[i][j] * s


def main() raises:
    var ctx = DeviceContext()
    print("==== ernie LoRA STEP smoke (build -> fwd -> bwd -> clip -> AdamW -> save/load) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " S=", S, " F=", F, " L=", L,
          " RANK=", RANK, " ALPHA=", ALPHA)

    var img_tokens = _in("in_img_tokens")
    var txt_tokens = _in("in_txt_tokens")
    var base = _load_base(ctx)
    var mv = _load_mod()
    var f_scale = _in("in_f_scale")
    var f_shift = _in("in_f_shift")
    var blocks = List[ErnieBlockWeights]()
    for l in range(L):
        blocks.append(_load_block(l, ctx))
    var cos = Tensor.from_host(_in("in_cos"), [S * H, Dh], STDtype.F32, ctx)
    var sin = Tensor.from_host(_in("in_sin"), [S * H, Dh], STDtype.F32, ctx)

    # ── build the LoRA set (B=0 init -> adapter identity at step 0) ──
    var lora = build_ernie_lora_set(L, D, F, RANK, ALPHA)
    var n_adapters = L * ERNIE_SLOTS
    print("")
    print("adapter count =", n_adapters, " (7 slots x", L, "layers)")

    # confirm B==0 at init (PEFT identity), and the forward == base forward.
    var b_absum_init = Float32(0.0)
    for i in range(n_adapters):
        b_absum_init += _absum(lora.ad[i].b)
    print("LoRA-B |.|_1 at init =", b_absum_init, " (expect 0.0)")

    # ── forward ──
    var fwd = ernie_stack_lora_forward[H, Dh, N_IMG, N_TXT, S](
        img_tokens.copy(), txt_tokens.copy(), base, blocks, lora, mv,
        f_scale.copy(), f_shift.copy(), cos, sin,
        D, F, IN_CH, TEXT_IN, OUT_CH, EPS, ctx,
    )

    # flow-style upstream grad: target = the base inputs latent (proxy); use the
    # oracle d_out as a finite, deterministic upstream so the step is reproducible.
    var d_out = _in("in_d_out")   # [N_IMG, OUT_CH]

    # ── backward ──
    var grads = ernie_stack_lora_backward[H, Dh, N_IMG, N_TXT, S](
        d_out, img_tokens.copy(), txt_tokens.copy(), base, blocks, lora, mv,
        f_scale.copy(), f_shift.copy(), cos, sin, fwd,
        D, F, IN_CH, TEXT_IN, OUT_CH, EPS, ctx,
    )
    print("nonfinite_lora_grads =", grads.nonfinite_lora_grads, " (expect 0)")

    # grad |.|_1 over A and B (must be > 0 so AdamW moves params)
    var da_absum = Float32(0.0)
    var db_absum = Float32(0.0)
    for i in range(n_adapters):
        da_absum += _absum(grads.d_a[i])
        db_absum += _absum(grads.d_b[i])
    print("grad |dA|_1 =", da_absum, "  |dB|_1 =", db_absum)

    # ── global-norm clip (max_norm = 1.0) ──
    var gn_before = _global_norm(grads)
    _clip(grads, Float32(1.0))
    var gn_after = _global_norm(grads)
    print("global grad norm: before =", gn_before, " after clip(1.0) =", gn_after)

    # ── AdamW step ──
    ernie_lora_adamw_step(lora, grads, 1, Float32(1.0e-3), ctx)

    # confirm LoRA-B moved 0 -> nonzero (the adapter is LEARNING). Count B slots
    # whose |.|_1 > 0 after the step.
    var b_nonzero_slots = 0
    var b_absum_after = Float32(0.0)
    for i in range(n_adapters):
        var s = _absum(lora.ad[i].b)
        b_absum_after += s
        if s > 0.0:
            b_nonzero_slots += 1
    print("")
    print("LoRA-B |.|_1 after AdamW =", b_absum_after)
    print("LoRA-B nonzero slots =", b_nonzero_slots, "/", n_adapters,
          " ratio =", Float32(b_nonzero_slots) / Float32(n_adapters))

    var trains = (b_absum_init == 0.0) and (b_absum_after > 0.0) and (b_nonzero_slots == n_adapters)

    # ── save -> load BYTE-EXACT round-trip ──
    var npairs = save_ernie_lora(lora, SAVE_PATH, ctx)
    print("")
    print("save_ernie_lora wrote", npairs, "adapter pairs to", SAVE_PATH)
    var reloaded = load_ernie_lora_resume(L, RANK, ALPHA, SAVE_PATH, ctx)

    var max_abs_diff = Float32(0.0)
    for i in range(n_adapters):
        if len(lora.ad[i].a) != len(reloaded.ad[i].a) or len(lora.ad[i].b) != len(reloaded.ad[i].b):
            raise Error("round-trip shape mismatch")
        for j in range(len(lora.ad[i].a)):
            var d = lora.ad[i].a[j] - reloaded.ad[i].a[j]
            d = d if d >= 0.0 else -d
            if d > max_abs_diff:
                max_abs_diff = d
        for j in range(len(lora.ad[i].b)):
            var d = lora.ad[i].b[j] - reloaded.ad[i].b[j]
            d = d if d >= 0.0 else -d
            if d > max_abs_diff:
                max_abs_diff = d
    print("save/load max_abs_diff (A+B over all adapters) =", max_abs_diff,
          "  ", "BYTE-EXACT" if max_abs_diff == 0.0 else "DIVERGED")

    print("")
    var byte_exact = (max_abs_diff == Float32(0.0))
    if trains and byte_exact and grads.nonfinite_lora_grads == 0:
        print("VERDICT: PASS — ERNIE LoRA step trains (B 0->nonzero), grads finite, save/load byte-exact")
    else:
        print("VERDICT: FAIL — trains=", trains, " byte_exact=", byte_exact,
              " nonfinite=", grads.nonfinite_lora_grads)
