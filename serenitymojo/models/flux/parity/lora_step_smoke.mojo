# serenitymojo/models/flux/parity/lora_step_smoke.mojo
#
# END-TO-END LoRA TRAINING STEP smoke for Flux (flux1-dev) — mirrors the proven
# Ernie/Klein LoRA step proof. Uses the SAME reduced parity dims + base inputs
# (lora_stack_oracle.py lin_* .bin files) so it is cheap on the shared 3090, but
# the pipeline is the REAL one:
#   build_flux_lora_set (B=0 init -> PEFT identity) -> flux_stack_lora_forward
#   -> oracle d_out upstream -> flux_stack_lora_backward -> global-norm clip(1.0)
#   -> flux_lora_adamw_step -> confirm LoRA-B moves 0 -> nonzero (nonzero-slot
#   ratio = 1.0) -> save_flux_lora -> load_flux_lora_resume -> assert A/B
#   BYTE-EXACT round-trip (max_abs_diff = 0.0).
# This is the "complete LoRA training STEP" deliverable.
#
# Run (oracle FIRST; SEPARATE command):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/flux/parity/lora_stack_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/models/flux/parity/lora_step_smoke.mojo -o /tmp/flux_lora_step
#   /tmp/flux_lora_step

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import alloc
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.flux.block import (
    StreamWeights, DoubleBlockWeights, SingleBlockWeights,
)
from serenitymojo.models.flux.flux_stack import FluxStackBase, EmbedMlp, ModLin, DoubleModLin
from serenitymojo.models.flux.flux_stack_lora import (
    FluxLoraSet, FluxLoraGradSet, build_flux_lora_set, total_adapters,
    flux_stack_lora_forward, flux_stack_lora_backward,
    flux_lora_adamw_step, save_flux_lora, load_flux_lora_resume,
)


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/flux/parity/"

comptime H = 24
comptime Dh = 128
comptime D = H * Dh
comptime N_IMG = 4
comptime N_TXT = 3
comptime S = N_TXT + N_IMG
comptime FMLP = 32
comptime IN_CH = 64
comptime TXT_CH = 40
comptime OUT_CH = 64
comptime T_DIM = 16
comptime VEC_DIM = 20
comptime NUM_DOUBLE = 2
comptime NUM_SINGLE = 2
comptime EPS = Float32(1e-06)
comptime MAX_PERIOD = Float32(10000.0)
comptime RANK = 16
comptime ALPHA = Float32(1.0)        # OT default
comptime SAVE_PATH = "/tmp/flux_lora_smoke.safetensors"


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing ref (run lora_stack_oracle.py first): ") + path)
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
        _in("lin_" + prefix + "_wqkv"), _in("lin_" + prefix + "_bqkv"),
        _in("lin_" + prefix + "_wproj"), _in("lin_" + prefix + "_bproj"),
        _in("lin_" + prefix + "_wmlp0"), _in("lin_" + prefix + "_bmlp0"),
        _in("lin_" + prefix + "_wmlp2"), _in("lin_" + prefix + "_bmlp2"),
        _in("lin_" + prefix + "_q_norm"), _in("lin_" + prefix + "_k_norm"),
        D, FMLP, Dh, ctx,
    )


def _load_single(prefix: String, ctx: DeviceContext) raises -> SingleBlockWeights:
    return SingleBlockWeights(
        _in("lin_" + prefix + "_w1"), _in("lin_" + prefix + "_b1"),
        _in("lin_" + prefix + "_w2"), _in("lin_" + prefix + "_b2"),
        _in("lin_" + prefix + "_q_norm"), _in("lin_" + prefix + "_k_norm"),
        D, FMLP, Dh, ctx,
    )


def _load_embed(tag: String, in_dim: Int, ctx: DeviceContext) raises -> EmbedMlp:
    return EmbedMlp(
        _in("lin_" + tag + "_in_w"), _in("lin_" + tag + "_in_b"),
        _in("lin_" + tag + "_out_w"), _in("lin_" + tag + "_out_b"),
        in_dim, D, ctx,
    )


def _absum(v: List[Float32]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(v)):
        var x = v[i]
        s += x if x >= 0.0 else -x
    return s


def _global_norm(grads: FluxLoraGradSet) -> Float32:
    var ss = Float32(0.0)
    var n = len(grads.d_a)
    for i in range(n):
        for j in range(len(grads.d_a[i])):
            ss += grads.d_a[i][j] * grads.d_a[i][j]
        for j in range(len(grads.d_b[i])):
            ss += grads.d_b[i][j] * grads.d_b[i][j]
    return sqrt(ss)


def _clip(mut grads: FluxLoraGradSet, max_norm: Float32):
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
    print("==== flux LoRA STEP smoke (build -> fwd -> bwd -> clip -> AdamW -> save/load) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " S=", S, " FMLP=", FMLP,
          " NUM_DOUBLE=", NUM_DOUBLE, " NUM_SINGLE=", NUM_SINGLE,
          " RANK=", RANK, " ALPHA=", ALPHA)

    var time_in = _load_embed("time", T_DIM, ctx)
    var guidance_in = _load_embed("guid", T_DIM, ctx)
    var vector_in = _load_embed("vec", VEC_DIM, ctx)
    var dbl_mod = List[DoubleModLin]()
    for bi in range(NUM_DOUBLE):
        var p = String(bi)
        var im = ModLin(_in("lin_d" + p + "_imod_w"), _in("lin_d" + p + "_imod_b"), 6 * D, D, ctx)
        var tm = ModLin(_in("lin_d" + p + "_tmod_w"), _in("lin_d" + p + "_tmod_b"), 6 * D, D, ctx)
        dbl_mod.append(DoubleModLin(im^, tm^))
    var sgl_mod = List[ModLin]()
    for bi in range(NUM_SINGLE):
        var p = String(bi)
        sgl_mod.append(ModLin(_in("lin_s" + p + "_mod_w"), _in("lin_s" + p + "_mod_b"), 3 * D, D, ctx))
    var base = FluxStackBase(
        _in("lin_img_in"), _in("lin_img_in_b"), _in("lin_txt_in"), _in("lin_txt_in_b"),
        time_in^, True, guidance_in^, vector_in^,
        dbl_mod^, sgl_mod^,
        _in("lin_final_adaln_w"), _in("lin_final_adaln_b"),
        _in("lin_final_lin"), _in("lin_final_lin_b"),
        D, IN_CH, TXT_CH, OUT_CH, ctx,
    )
    var dbw = List[DoubleBlockWeights]()
    for bi in range(NUM_DOUBLE):
        var p = String("d") + String(bi)
        dbw.append(DoubleBlockWeights(_load_stream(p + "_iw", ctx), _load_stream(p + "_tw", ctx)))
    var sbw = List[SingleBlockWeights]()
    for bi in range(NUM_SINGLE):
        sbw.append(_load_single(String("s") + String(bi), ctx))

    var img_tokens = _in("lin_img_tokens")
    var txt_tokens = _in("lin_txt_tokens")
    var timestep = _in("lin_timestep")
    var guidance = Optional[List[Float32]](_in("lin_guidance"))
    var vector = _in("lin_vector")
    var cos = _in("lin_cos")
    var sin = _in("lin_sin")

    # ── build the LoRA set (B=0 init -> adapter identity at step 0) ──
    var lora = build_flux_lora_set(NUM_DOUBLE, NUM_SINGLE, D, FMLP, RANK, ALPHA)
    var n_adapters = total_adapters(lora)
    print("")
    print("adapter count =", n_adapters,
          " (", NUM_DOUBLE, "x12 double +", NUM_SINGLE, "x5 single)")

    var b_absum_init = Float32(0.0)
    for i in range(n_adapters):
        b_absum_init += _absum(lora.ad[i].b)
    print("LoRA-B |.|_1 at init =", b_absum_init, " (expect 0.0)")

    var fwd = flux_stack_lora_forward[H, Dh, N_IMG, N_TXT, S](
        img_tokens.copy(), txt_tokens.copy(), timestep.copy(), guidance, vector.copy(),
        base, dbw, sbw, lora, cos.copy(), sin.copy(),
        D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, EPS, ctx,
    )

    var d_out = _in("lin_d_out")
    var grads = flux_stack_lora_backward[H, Dh, N_IMG, N_TXT, S](
        d_out, img_tokens.copy(), txt_tokens.copy(), base, dbw, sbw, lora,
        cos.copy(), sin.copy(), fwd,
        D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, MAX_PERIOD, EPS, ctx,
    )
    print("nonfinite_lora_grads =", grads.nonfinite_lora_grads, " (expect 0)")

    var da_absum = Float32(0.0)
    var db_absum = Float32(0.0)
    for i in range(n_adapters):
        da_absum += _absum(grads.d_a[i])
        db_absum += _absum(grads.d_b[i])
    print("grad |dA|_1 =", da_absum, "  |dB|_1 =", db_absum)

    var gn_before = _global_norm(grads)
    _clip(grads, Float32(1.0))
    var gn_after = _global_norm(grads)
    print("global grad norm: before =", gn_before, " after clip(1.0) =", gn_after)

    flux_lora_adamw_step(lora, grads, 1, Float32(1.0e-3), ctx)

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

    var npairs = save_flux_lora(lora, SAVE_PATH, ctx)
    print("")
    print("save_flux_lora wrote", npairs, "adapter pairs to", SAVE_PATH)
    var reloaded = load_flux_lora_resume(NUM_DOUBLE, NUM_SINGLE, RANK, ALPHA, SAVE_PATH, ctx)

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
        print("VERDICT: PASS — Flux LoRA step trains (B 0->nonzero, ratio=1.0), grads finite, save/load byte-exact")
    else:
        print("VERDICT: FAIL — trains=", trains, " byte_exact=", byte_exact,
              " nonfinite=", grads.nonfinite_lora_grads)
