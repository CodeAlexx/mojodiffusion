# serenitymojo/models/flux/parity/lora_stack_parity.mojo
#
# COMPOSITION PARITY GATE for the Flux FULL DiT STACK *WITH LoRA* (reduced depth,
# REAL H/Dh/D) — models/flux/flux_stack_lora.mojo. Loads the EXACT base weights +
# LoRA A/B masters + torch-autograd reference grads dumped by
# lora_stack_oracle.py, runs flux_stack_lora_forward + flux_stack_lora_backward
# (NUM_DOUBLE double + NUM_SINGLE single), and compares:
#   - base-no-regression: forward `out` cos>=0.999 vs the LoRA-on torch oracle,
#   - EVERY adapter's d_A and d_B cos>=0.999 vs torch autograd,
#   - the load-bearing input-token + embed grads cos>=0.999,
#   - 0 nonfinite LoRA grads.
#
# B is NONZERO in the oracle so every grad arm is exercised; the Mojo gate seeds
# its FluxLoraSet from the SAME A/B masters (lin_*_A / lin_*_B). alpha=1.0 (OT
# default); the gate is alpha-agnostic for grad correctness — oracle and Mojo
# share LSCALE = alpha/rank.
#
# Run (oracle FIRST, SEPARATE command):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/flux/parity/lora_stack_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/models/flux/parity/lora_stack_parity.mojo -o /tmp/flux_lora_parity
#   /tmp/flux_lora_parity

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.flux.block import (
    StreamWeights, DoubleBlockWeights, SingleBlockWeights,
)
from serenitymojo.models.flux.flux_stack import FluxStackBase, EmbedMlp, ModLin, DoubleModLin
from serenitymojo.models.flux.flux_stack_lora import (
    FluxLoraSet, FluxLoraGradSet, flux_stack_lora_forward, flux_stack_lora_backward,
    DBL_SLOTS_PER_BLOCK,
)
from serenitymojo.models.flux.lora_block import DBL_STREAM_SLOTS, SGL_SLOTS


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/flux/parity/"

# dims MUST match lora_stack_oracle.py
comptime H = 24
comptime Dh = 128
comptime D = H * Dh            # 3072
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
comptime RANK = 4
comptime ALPHA = Float32(1.0)
comptime LSCALE = ALPHA / Float32(RANK)


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


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(0.0)
    return o^


def _adapter(tag: String, in_f: Int, out_f: Int) raises -> LoraAdapter:
    return LoraAdapter(
        _in("lin_" + tag + "_A"), _in("lin_" + tag + "_B"),
        RANK, in_f, out_f, LSCALE,
        _zeros(RANK * in_f), _zeros(RANK * in_f),
        _zeros(out_f * RANK), _zeros(out_f * RANK),
    )


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


# Build the FluxLoraSet from the oracle's A/B masters in the EXACT flat order
# build_flux_lora_set produces (doubles: img stream 6 slots then txt stream 6
# slots; singles: 5 slots). Slot order {to_q,to_k,to_v,proj,mlp0,mlp2} for double
# streams; {to_q,to_k,to_v,proj_mlp,linear2} for single blocks.
def _build_set_from_oracle(ctx: DeviceContext) raises -> FluxLoraSet:
    var ad = List[LoraAdapter]()
    for bi in range(NUM_DOUBLE):
        var p = String("d") + String(bi)
        for stream in ["img", "txt"]:
            var sp = p + "_" + stream
            ad.append(_adapter(sp + "_to_q", D, D))
            ad.append(_adapter(sp + "_to_k", D, D))
            ad.append(_adapter(sp + "_to_v", D, D))
            ad.append(_adapter(sp + "_proj", D, D))
            ad.append(_adapter(sp + "_mlp0", D, FMLP))
            ad.append(_adapter(sp + "_mlp2", FMLP, D))
    for bi in range(NUM_SINGLE):
        var sp = String("s") + String(bi)
        ad.append(_adapter(sp + "_to_q", D, D))
        ad.append(_adapter(sp + "_to_k", D, D))
        ad.append(_adapter(sp + "_to_v", D, D))
        ad.append(_adapter(sp + "_proj_mlp", D, FMLP))
        ad.append(_adapter(sp + "_linear2", D + FMLP, D))
    return FluxLoraSet(ad^, NUM_DOUBLE, NUM_SINGLE, RANK)


def _check(
    mut harness: ParityHarness, name: String,
    actual: List[Float32], expected: List[Float32], mut allok: Bool, mut npass: Int, mut nfail: Int,
) raises:
    var r = harness.compare_host(actual, expected)
    if not r.passed:
        print("  cos(", name, ") =", r.cos, "  max_abs =", r.max_abs, "  n =", r.n, "   FAIL")
        allok = False
        nfail += 1
    else:
        npass += 1


def main() raises:
    var ctx = DeviceContext()
    print("==== flux lora_stack_parity (Flux FULL stack + LoRA vs torch) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " N_IMG=", N_IMG, " N_TXT=", N_TXT,
          " FMLP=", FMLP, " RANK=", RANK, " ALPHA=", ALPHA,
          " num_double=", NUM_DOUBLE, " num_single=", NUM_SINGLE)

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

    var lora = _build_set_from_oracle(ctx)

    var img_tokens = _in("lin_img_tokens")
    var txt_tokens = _in("lin_txt_tokens")
    var timestep = _in("lin_timestep")
    var guidance = Optional[List[Float32]](_in("lin_guidance"))
    var vector = _in("lin_vector")
    var cos = _in("lin_cos")
    var sin = _in("lin_sin")

    var fwd = flux_stack_lora_forward[H, Dh, N_IMG, N_TXT, S](
        img_tokens.copy(), txt_tokens.copy(), timestep.copy(), guidance, vector.copy(),
        base, dbw, sbw, lora, cos.copy(), sin.copy(),
        D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, EPS, ctx,
    )

    var harness = ParityHarness()
    var allok = True
    var npass = 0
    var nfail = 0

    print("")
    print("---- forward output (LoRA-on, base-no-regression vs oracle) ----")
    _check(harness, "out", fwd.out, _in("lr_out"), allok, npass, nfail)

    var d_out = _in("lin_d_out")
    var g = flux_stack_lora_backward[H, Dh, N_IMG, N_TXT, S](
        d_out, img_tokens.copy(), txt_tokens.copy(), base, dbw, sbw, lora,
        cos.copy(), sin.copy(), fwd,
        D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, MAX_PERIOD, EPS, ctx,
    )

    print("")
    print("---- load-bearing input-token + embed grads ----")
    _check(harness, "d_img_tokens", g.d_img_tokens, _in("lr_d_img_tokens"), allok, npass, nfail)
    _check(harness, "d_txt_tokens", g.d_txt_tokens, _in("lr_d_txt_tokens"), allok, npass, nfail)
    _check(harness, "d_vec", g.d_vec, _in("lr_d_vec"), allok, npass, nfail)
    _check(harness, "d_timestep", g.d_timestep, _in("lr_d_timestep"), allok, npass, nfail)
    _check(harness, "d_guidance", g.d_guidance, _in("lr_d_guidance"), allok, npass, nfail)
    _check(harness, "d_vector", g.d_vector, _in("lr_d_vector"), allok, npass, nfail)

    print("")
    print("---- ALL adapter d_A / d_B vs torch (only FAILs printed) ----")
    # double blocks: flat = bi*DBL_SLOTS_PER_BLOCK + stream*6 + slot
    var dbl_slots: List[String] = ["to_q", "to_k", "to_v", "proj", "mlp0", "mlp2"]
    for bi in range(NUM_DOUBLE):
        var streams: List[String] = ["img", "txt"]
        for si in range(2):
            for s in range(DBL_STREAM_SLOTS):
                var idx = bi * DBL_SLOTS_PER_BLOCK + si * DBL_STREAM_SLOTS + s
                var tag = String("d") + String(bi) + "_" + streams[si] + "_" + dbl_slots[s]
                _check(harness, tag + "_dA", g.d_a[idx], _in("lr_" + tag + "_dA"), allok, npass, nfail)
                _check(harness, tag + "_dB", g.d_b[idx], _in("lr_" + tag + "_dB"), allok, npass, nfail)
    # single blocks: flat = NUM_DOUBLE*DBL_SLOTS_PER_BLOCK + bi*SGL_SLOTS + slot
    var sgl_slots: List[String] = ["to_q", "to_k", "to_v", "proj_mlp", "linear2"]
    for bi in range(NUM_SINGLE):
        for s in range(SGL_SLOTS):
            var idx = NUM_DOUBLE * DBL_SLOTS_PER_BLOCK + bi * SGL_SLOTS + s
            var tag = String("s") + String(bi) + "_" + sgl_slots[s]
            _check(harness, tag + "_dA", g.d_a[idx], _in("lr_" + tag + "_dA"), allok, npass, nfail)
            _check(harness, tag + "_dB", g.d_b[idx], _in("lr_" + tag + "_dB"), allok, npass, nfail)

    print("")
    print("checks: PASS =", npass, " FAIL =", nfail, " nonfinite_lora_grads =", g.nonfinite_lora_grads)
    if allok and g.nonfinite_lora_grads == 0:
        print("VERDICT: PASS — Flux FULL stack + LoRA fwd+bwd composes (all A/B cos>=0.999, 0 nonfinite)")
    else:
        print("VERDICT: FAIL — at least one arm diverged or grads nonfinite (see FAIL lines)")
