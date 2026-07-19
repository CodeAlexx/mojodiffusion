# serenitymojo/models/anima/parity/lora_step_smoke.mojo
#
# END-TO-END LoRA TRAINING STEP smoke for ANIMA (mirrors the Klein/Ernie LoRA proof).
# Uses the SAME small parity dims (L=3) and the SAME base inputs (stack_oracle.py
# .bin files) so it is cheap on the shared 3090, but the pipeline is the REAL one:
# build adapters (B=0 init) -> anima_stack_lora_forward -> deterministic upstream
# grad -> anima_stack_lora_backward -> global-norm clip -> anima_lora_adamw_step ->
# confirm LoRA-B moves 0 -> nonzero (the adapter is LEARNING, nonzero-slot ratio=1.0)
# -> save_anima_lora -> load_anima_lora_resume -> assert A/B BYTE-EXACT round-trip.
# This is the "complete LoRA training STEP" deliverable (Anima per-model milestone).
#
# Run (oracle FIRST for the base inputs; SEPARATE command):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/anima/parity/stack_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/models/anima/parity/lora_step_smoke.mojo -o /tmp/anima_lora_step
#   /tmp/anima_lora_step

from std.gpu.host import DeviceContext
from std.collections import List
from std.math import sqrt
from std.memory import alloc, ArcPointer
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.anima.weights import AnimaBlockWeights, AnimaStackBase
from serenitymojo.models.anima.lora_block import ANIMA_SLOTS
from serenitymojo.models.anima.anima_stack_lora import (
    AnimaLoraSet, AnimaLoraGrads, build_anima_lora_set,
    anima_stack_lora_forward, anima_stack_lora_backward,
    anima_lora_adamw_step, save_anima_lora, load_anima_lora_resume,
)


comptime TArc = ArcPointer[Tensor]
comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/anima/parity/"

comptime B = 1
comptime H = 16
comptime Dh = 128
comptime D = H * Dh        # 2048
comptime S_IMG = 6
comptime S_TXT = 8
comptime JOINT = 1024
comptime F = 32
comptime IN_PATCH = 68
comptime OUT_PATCH = 64
comptime L = 3
comptime EPS = Float32(1e-06)
comptime RANK = 8
comptime ALPHA = Float32(16.0)
comptime SAVE_PATH = "/tmp/anima_lora_smoke.safetensors"


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


def _t(name: String, var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(_in(name), shape^, STDtype.F32, ctx)


def _sh(*dims: Int) -> List[Int]:
    var o = List[Int]()
    for d in dims:
        o.append(d)
    return o^


def _load_block(l: Int, ctx: DeviceContext) raises -> AnimaBlockWeights:
    var p = String("in_blk") + String(l) + String("_")
    return AnimaBlockWeights(
        _t(p + "sa_mod1", _sh(256, D), ctx), _t(p + "sa_mod2", _sh(3 * D, 256), ctx),
        _t(p + "ca_mod1", _sh(256, D), ctx), _t(p + "ca_mod2", _sh(3 * D, 256), ctx),
        _t(p + "mlp_mod1", _sh(256, D), ctx), _t(p + "mlp_mod2", _sh(3 * D, 256), ctx),
        _t(p + "sa_q", _sh(D, D), ctx), _t(p + "sa_k", _sh(D, D), ctx),
        _t(p + "sa_v", _sh(D, D), ctx), _t(p + "sa_out", _sh(D, D), ctx),
        _t(p + "sa_qn", _sh(Dh), ctx), _t(p + "sa_kn", _sh(Dh), ctx),
        _t(p + "ca_q", _sh(D, D), ctx), _t(p + "ca_k", _sh(D, JOINT), ctx),
        _t(p + "ca_v", _sh(D, JOINT), ctx), _t(p + "ca_out", _sh(D, D), ctx),
        _t(p + "ca_qn", _sh(Dh), ctx), _t(p + "ca_kn", _sh(Dh), ctx),
        _t(p + "mlp1", _sh(F, D), ctx), _t(p + "mlp2", _sh(D, F), ctx),
    )


def _load_base(ctx: DeviceContext) raises -> AnimaStackBase:
    return AnimaStackBase(
        TArc(_t("in_x_embed", _sh(D, IN_PATCH), ctx)),
        TArc(_t("in_x_embed", _sh(D, IN_PATCH), ctx)),
        TArc(_t("in_x_embed", _sh(D, IN_PATCH), ctx)),
        TArc(_t("in_fl_lin", _sh(OUT_PATCH, D), ctx)),
        TArc(_t("in_fl_mod1", _sh(256, D), ctx)),
        TArc(_t("in_fl_mod2", _sh(2 * D, 256), ctx)),
        TArc(_t("in_fl_lin", _sh(OUT_PATCH, D), ctx)),
    )


def _expand_rope(name: String) raises -> List[Float32]:
    var half = Dh // 2
    var per_pos = _in(name)
    var out = List[Float32]()
    for _b in range(B):
        for s in range(S_IMG):
            for _h in range(H):
                for i in range(half):
                    out.append(per_pos[s * half + i])
    return out^


def _absum(v: List[Float32]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(v)):
        var x = v[i]
        s += x if x >= 0.0 else -x
    return s


# host global L2 norm over the flat LoRA grads (the clip basis).
def _global_norm(grads: AnimaLoraGrads) -> Float32:
    var ss = Float32(0.0)
    var n = len(grads.d_a)
    for i in range(n):
        for j in range(len(grads.d_a[i])):
            ss += grads.d_a[i][j] * grads.d_a[i][j]
        for j in range(len(grads.d_b[i])):
            ss += grads.d_b[i][j] * grads.d_b[i][j]
    return sqrt(ss)


def _clip(mut grads: AnimaLoraGrads, max_norm: Float32):
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
    print("==== anima LoRA STEP smoke (build -> fwd -> bwd -> clip -> AdamW -> save/load) ====")
    print("B=", B, " H=", H, " Dh=", Dh, " D=", D, " S_img=", S_IMG, " S_txt=", S_TXT,
          " F=", F, " L=", L, " RANK=", RANK, " ALPHA=", ALPHA)

    var patches = _in("in_patches")
    var t_cond = _in("in_t_cond")
    var base_adaln = _in("in_base_adaln")
    var context = _in("in_context")
    var base = _load_base(ctx)
    var blocks = List[AnimaBlockWeights]()
    for l in range(L):
        blocks.append(_load_block(l, ctx))
    var half = Dh // 2
    var cos = Tensor.from_host(_expand_rope("in_cos"), [B * S_IMG * H, half], STDtype.F32, ctx)
    var sin = Tensor.from_host(_expand_rope("in_sin"), [B * S_IMG * H, half], STDtype.F32, ctx)

    # ── build the LoRA set (B=0 init -> adapter identity at step 0) ──
    var lora = build_anima_lora_set(L, D, JOINT, F, RANK, ALPHA)
    var n_adapters = L * ANIMA_SLOTS
    print("")
    print("adapter count =", n_adapters, " (10 slots x", L, "blocks)")

    var b_absum_init = Float32(0.0)
    for i in range(n_adapters):
        b_absum_init += _absum(lora.ad[i].b)
    print("LoRA-B |.|_1 at init =", b_absum_init, " (expect 0.0)")

    # ── forward ──
    var fwd = anima_stack_lora_forward[H, Dh, S_IMG, S_TXT](
        patches.copy(), t_cond.copy(), base_adaln.copy(), context.copy(),
        base, blocks, lora, cos, sin,
        B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
    )

    # deterministic upstream grad (oracle d_out) so the step is reproducible.
    var d_out = _in("in_d_out")   # [B*S_img, OUT_PATCH]

    # ── backward ──
    var grads = anima_stack_lora_backward[H, Dh, S_IMG, S_TXT](
        d_out, patches.copy(), t_cond.copy(), base_adaln.copy(), context.copy(),
        base, blocks, lora, cos, sin, fwd,
        B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
    )
    print("nonfinite_lora_grads =", grads.nonfinite_lora_grads, " (expect 0)")

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
    var clip_applied = gn_before > Float32(1.0) and gn_after <= Float32(1.0001)

    # ── AdamW step ──
    anima_lora_adamw_step(lora, grads, 1, Float32(1.0e-3), ctx)

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
    var npairs = save_anima_lora(lora, SAVE_PATH, ctx)
    print("")
    print("save_anima_lora wrote", npairs, "adapter pairs to", SAVE_PATH)
    var reloaded = load_anima_lora_resume(L, RANK, ALPHA, SAVE_PATH, ctx)

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
    if trains and byte_exact and grads.nonfinite_lora_grads == 0 and clip_applied:
        print("VERDICT: PASS — ANIMA LoRA step trains (B 0->nonzero, ratio=1.0), grads finite, clip applied, save/load byte-exact")
    else:
        print("VERDICT: FAIL — trains=", trains, " byte_exact=", byte_exact,
              " nonfinite=", grads.nonfinite_lora_grads, " clip_applied=", clip_applied)
