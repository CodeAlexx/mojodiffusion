# Klein LoRA stack backward parity through the offloaded activation tape.
#
# This is the bounded CUDA bridge test before real Klein train-ref replay:
# 1. run the existing small-stack LoRA forward,
# 2. compute resident-tape backward as the local reference,
# 3. offload the backward tape through HostOffload raw bytes,
# 4. restore the tape and rerun backward,
# 5. compare all load-bearing LoRA/input/modulation grads.
#
# It does not load Klein-9B and does not accept product CPU_OFFLOADED parity.

from std.collections import List
from std.gpu.host import DeviceContext
from std.memory import alloc

from serenitymojo.io.ffi import O_RDONLY, file_size, sys_close, sys_open, sys_pread
from serenitymojo.models.klein.activation_tape import (
    offload_klein_stack_lora_backward_tape,
    restore_klein_stack_lora_backward_tape,
)
from serenitymojo.models.klein.double_block import (
    DoubleBlockWeights,
    ModVecs,
    StreamWeights,
)
from serenitymojo.models.klein.klein_stack import KleinStackBase
from serenitymojo.models.klein.klein_stack_lora import (
    KleinLoraGrads,
    KleinLoraSet,
    build_klein_lora_set,
    klein_stack_lora_backward,
    klein_stack_lora_forward,
)
from serenitymojo.models.klein.single_block import SingleBlockWeights, SingleModVecs
from serenitymojo.parity import ParityHarness


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/klein/parity/"

comptime H = 4
comptime Dh = 8
comptime D = H * Dh
comptime N_IMG = 4
comptime N_TXT = 2
comptime S = N_TXT + N_IMG
comptime F = 24
comptime IN_CH = 10
comptime TXT_CH = 14
comptime OUT_CH = 6
comptime NUM_DOUBLE = 1
comptime NUM_SINGLE = 1
comptime EPS = Float32(1e-06)
comptime RANK = 4
comptime ALPHA = Float32(8.0)


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing ref (run klein_stack_lora_oracle.py first): ") + path)
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


def _load_stream(prefix: String, ctx: DeviceContext) raises -> StreamWeights:
    return StreamWeights(
        _in("klin_" + prefix + "_wqkv"),
        _in("klin_" + prefix + "_wproj"),
        _in("klin_" + prefix + "_wgu"),
        _in("klin_" + prefix + "_wd"),
        _in("klin_" + prefix + "_q_norm"),
        _in("klin_" + prefix + "_k_norm"),
        D,
        F,
        Dh,
        ctx,
    )


def _load_single(prefix: String, ctx: DeviceContext) raises -> SingleBlockWeights:
    return SingleBlockWeights(
        _in("klin_" + prefix + "_w1"),
        _in("klin_" + prefix + "_w2"),
        _in("klin_" + prefix + "_q_norm"),
        _in("klin_" + prefix + "_k_norm"),
        D,
        F,
        Dh,
        ctx,
    )


def _load_mod(prefix: String) raises -> ModVecs:
    return ModVecs(
        _in("klin_" + prefix + "_shift1"),
        _in("klin_" + prefix + "_scale1"),
        _in("klin_" + prefix + "_gate1"),
        _in("klin_" + prefix + "_shift2"),
        _in("klin_" + prefix + "_scale2"),
        _in("klin_" + prefix + "_gate2"),
    )


def _load_single_mod() raises -> SingleModVecs:
    return SingleModVecs(
        _in("klin_sm_shift"),
        _in("klin_sm_scale"),
        _in("klin_sm_gate"),
    )


def _build_set() raises -> KleinLoraSet:
    return build_klein_lora_set(NUM_DOUBLE, NUM_SINGLE, D, F, RANK, ALPHA)


def _check(
    mut harness: ParityHarness,
    name: String,
    actual: List[Float32],
    expected: List[Float32],
    mut allok: Bool,
) raises:
    var r = harness.compare_host(actual, expected)
    print("  cos(", name, ") =", r.cos, " max_abs=", r.max_abs, " n=", r.n)
    if r.cos < Float64(0.99999) or r.max_abs > Float64(1.0e-5):
        allok = False


def _check_grad_group(
    mut harness: ParityHarness,
    prefix: String,
    actual: KleinLoraGrads,
    expected: KleinLoraGrads,
    mut allok: Bool,
) raises:
    _check(harness, prefix + String(" d_img_tokens"), actual.d_img_tokens, expected.d_img_tokens, allok)
    _check(harness, prefix + String(" d_txt_tokens"), actual.d_txt_tokens, expected.d_txt_tokens, allok)
    _check(harness, prefix + String(" d_img_mod"), actual.d_img_mod, expected.d_img_mod, allok)
    _check(harness, prefix + String(" d_txt_mod"), actual.d_txt_mod, expected.d_txt_mod, allok)
    _check(harness, prefix + String(" d_single_mod"), actual.d_single_mod, expected.d_single_mod, allok)
    for i in range(len(expected.dbl_d_a)):
        _check(harness, prefix + String(" dbl_d_a[") + String(i) + String("]"), actual.dbl_d_a[i], expected.dbl_d_a[i], allok)
        _check(harness, prefix + String(" dbl_d_b[") + String(i) + String("]"), actual.dbl_d_b[i], expected.dbl_d_b[i], allok)
    for i in range(len(expected.sgl_d_a)):
        _check(harness, prefix + String(" sgl_d_a[") + String(i) + String("]"), actual.sgl_d_a[i], expected.sgl_d_a[i], allok)
        _check(harness, prefix + String(" sgl_d_b[") + String(i) + String("]"), actual.sgl_d_b[i], expected.sgl_d_b[i], allok)


def main() raises:
    var ctx = DeviceContext()
    print("==== Klein LoRA offloaded-tape backward parity ====")

    var base = KleinStackBase(
        _in("klin_img_in"),
        _in("klin_txt_in"),
        _in("klin_final_lin"),
        _in("klin_final_shift"),
        _in("klin_final_scale"),
        D,
        IN_CH,
        TXT_CH,
        OUT_CH,
        ctx,
    )

    var dbw = List[DoubleBlockWeights]()
    for bi in range(NUM_DOUBLE):
        var p = String("d") + String(bi)
        dbw.append(DoubleBlockWeights(_load_stream(p + "_iw", ctx), _load_stream(p + "_tw", ctx)))
    var sbw = List[SingleBlockWeights]()
    for bi in range(NUM_SINGLE):
        sbw.append(_load_single(String("s") + String(bi), ctx))

    var im = _load_mod("im")
    var tm = _load_mod("tm")
    var sm = _load_single_mod()
    var lora = _build_set()

    var img_tokens = _in("klin_img_tokens")
    var txt_tokens = _in("klin_txt_tokens")
    var cos = _in("klin_cos")
    var sin = _in("klin_sin")

    var fwd = klein_stack_lora_forward[H, Dh, N_IMG, N_TXT, S](
        img_tokens.copy(),
        txt_tokens.copy(),
        base,
        dbw,
        sbw,
        lora,
        im,
        tm,
        sm,
        cos.copy(),
        sin.copy(),
        D,
        F,
        IN_CH,
        TXT_CH,
        OUT_CH,
        EPS,
        ctx,
    )
    var d_out = _in("klin_d_out")

    var resident = klein_stack_lora_backward[H, Dh, N_IMG, N_TXT, S](
        d_out.copy(),
        img_tokens.copy(),
        txt_tokens.copy(),
        base,
        dbw,
        sbw,
        lora,
        im,
        tm,
        sm,
        cos.copy(),
        sin.copy(),
        fwd,
        D,
        F,
        IN_CH,
        TXT_CH,
        OUT_CH,
        EPS,
        ctx,
    )

    var tape = offload_klein_stack_lora_backward_tape(fwd, ctx)
    print("  offloaded host bytes =", tape.total_host_bytes())
    if not tape.all_storage_dtype(fwd.img_out[].dtype()):
        raise Error("offloaded tape storage dtype mismatch")

    var restored = restore_klein_stack_lora_backward_tape(tape, ctx)
    var offloaded = klein_stack_lora_backward[H, Dh, N_IMG, N_TXT, S](
        d_out.copy(),
        img_tokens.copy(),
        txt_tokens.copy(),
        base,
        dbw,
        sbw,
        lora,
        im,
        tm,
        sm,
        cos.copy(),
        sin.copy(),
        restored,
        D,
        F,
        IN_CH,
        TXT_CH,
        OUT_CH,
        EPS,
        ctx,
    )

    var harness = ParityHarness()
    var allok = True
    _check_grad_group(harness, String("offloaded_vs_resident"), offloaded, resident, allok)

    if allok:
        print("klein_stack_lora_offloaded_tape_parity PASS")
    else:
        print("klein_stack_lora_offloaded_tape_parity FAIL")
        raise Error("Klein offloaded tape backward parity failed")
