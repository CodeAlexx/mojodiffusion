# serenitymojo/models/flux/parity/stack_parity.mojo
#
# COMPOSITION PARITY GATE for the Flux FULL DiT STACK (reduced depth, REAL
# H/Dh/D) — models/flux/flux_stack.mojo. Loads the EXACT inputs + torch-autograd
# reference grads dumped by stack_oracle.py, runs flux_stack_forward +
# flux_stack_backward (3 double + 3 single), and compares:
#   out, d_img_tokens, d_txt_tokens, d_vec, d_timestep, d_guidance, d_vector,
#   a deep double block's d_wqkv/d_wproj/d_wmlp0, the last double's d_wqkv,
#   deep+last single d_w1/d_w2 — all at cos >= 0.999.
#
# This proves the COMPOSITION: input proj, the embed->vec MLP chain, per-block
# modulation projections, the double->single concat/slice seam, the final layer,
# the d_img/d_txt->d_y inter-block handoff across DEPTH, AND the d_vec
# accumulation across every block + the final layer back to the embeds.
#
# Run (oracle FIRST, SEPARATE command, never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/flux/parity/stack_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/flux/parity/stack_parity.mojo

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.models.flux.block import (
    StreamWeights, DoubleBlockWeights, SingleBlockWeights,
)
from serenitymojo.models.flux.flux_stack import (
    FluxStackBase, EmbedMlp, ModLin, DoubleModLin,
    flux_stack_forward, flux_stack_backward,
)


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/flux/parity/"

# dims MUST match stack_oracle.py
comptime H = 24
comptime Dh = 128
comptime D = H * Dh            # 3072 (REAL)
comptime N_IMG = 4
comptime N_TXT = 3
comptime S = N_TXT + N_IMG
comptime FMLP = 32
comptime IN_CH = 64
comptime TXT_CH = 40
comptime OUT_CH = 64
comptime T_DIM = 16
comptime VEC_DIM = 20
comptime NUM_DOUBLE = 3
comptime NUM_SINGLE = 3
comptime EPS = Float32(1e-06)
comptime MAX_PERIOD = Float32(10000.0)


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


def _load_stream(prefix: String, ctx: DeviceContext) raises -> StreamWeights:
    return StreamWeights(
        _in("in_" + prefix + "_wqkv"), _in("in_" + prefix + "_bqkv"),
        _in("in_" + prefix + "_wproj"), _in("in_" + prefix + "_bproj"),
        _in("in_" + prefix + "_wmlp0"), _in("in_" + prefix + "_bmlp0"),
        _in("in_" + prefix + "_wmlp2"), _in("in_" + prefix + "_bmlp2"),
        _in("in_" + prefix + "_q_norm"), _in("in_" + prefix + "_k_norm"),
        D, FMLP, Dh, ctx,
    )


def _load_single(prefix: String, ctx: DeviceContext) raises -> SingleBlockWeights:
    return SingleBlockWeights(
        _in("in_" + prefix + "_w1"), _in("in_" + prefix + "_b1"),
        _in("in_" + prefix + "_w2"), _in("in_" + prefix + "_b2"),
        _in("in_" + prefix + "_q_norm"), _in("in_" + prefix + "_k_norm"),
        D, FMLP, Dh, ctx,
    )


def _load_embed(tag: String, in_dim: Int, ctx: DeviceContext) raises -> EmbedMlp:
    return EmbedMlp(
        _in("in_" + tag + "_in_w"), _in("in_" + tag + "_in_b"),
        _in("in_" + tag + "_out_w"), _in("in_" + tag + "_out_b"),
        in_dim, D, ctx,
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
    print("==== flux stack_parity (Flux FULL stack vs torch) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " N_IMG=", N_IMG, " N_TXT=", N_TXT,
          " FMLP=", FMLP, " num_double=", NUM_DOUBLE, " num_single=", NUM_SINGLE)

    # ── embeds ──
    var time_in = _load_embed("time", T_DIM, ctx)
    var guidance_in = _load_embed("guid", T_DIM, ctx)
    var vector_in = _load_embed("vec", VEC_DIM, ctx)

    # ── per-block mod.lin ──
    var dbl_mod = List[DoubleModLin]()
    for bi in range(NUM_DOUBLE):
        var p = String(bi)
        var im = ModLin(_in("in_d" + p + "_imod_w"), _in("in_d" + p + "_imod_b"), 6 * D, D, ctx)
        var tm = ModLin(_in("in_d" + p + "_tmod_w"), _in("in_d" + p + "_tmod_b"), 6 * D, D, ctx)
        dbl_mod.append(DoubleModLin(im^, tm^))
    var sgl_mod = List[ModLin]()
    for bi in range(NUM_SINGLE):
        var p = String(bi)
        sgl_mod.append(ModLin(_in("in_s" + p + "_mod_w"), _in("in_s" + p + "_mod_b"), 3 * D, D, ctx))

    # ── base ──
    var base = FluxStackBase(
        _in("in_img_in"), _in("in_img_in_b"), _in("in_txt_in"), _in("in_txt_in_b"),
        time_in^, True, guidance_in^, vector_in^,
        dbl_mod^, sgl_mod^,
        _in("in_final_adaln_w"), _in("in_final_adaln_b"),
        _in("in_final_lin"), _in("in_final_lin_b"),
        D, IN_CH, TXT_CH, OUT_CH, ctx,
    )

    # ── per-block weights ──
    var dbw = List[DoubleBlockWeights]()
    for bi in range(NUM_DOUBLE):
        var p = String("d") + String(bi)
        dbw.append(DoubleBlockWeights(_load_stream(p + "_iw", ctx), _load_stream(p + "_tw", ctx)))
    var sbw = List[SingleBlockWeights]()
    for bi in range(NUM_SINGLE):
        sbw.append(_load_single(String("s") + String(bi), ctx))

    # ── inputs ──
    var img_tokens = _in("in_img_tokens")
    var txt_tokens = _in("in_txt_tokens")
    var timestep = _in("in_timestep")
    var guidance = Optional[List[Float32]](_in("in_guidance"))
    var vector = _in("in_vector")
    var cos = _in("in_cos")
    var sin = _in("in_sin")

    # ── forward ──
    var fwd = flux_stack_forward[H, Dh, N_IMG, N_TXT, S](
        img_tokens.copy(), txt_tokens.copy(), timestep.copy(), guidance, vector.copy(),
        base, dbw, sbw, cos.copy(), sin.copy(),
        D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, EPS, ctx,
    )

    var harness = ParityHarness()
    var allok = True

    print("")
    print("---- forward output vs torch ----")
    _check(harness, "out", fwd.out, _in("ref_out"), allok)

    # ── backward ──
    var d_out = _in("in_d_out")
    var g = flux_stack_backward[H, Dh, N_IMG, N_TXT, S](
        d_out, img_tokens.copy(), txt_tokens.copy(), base, dbw, sbw,
        cos.copy(), sin.copy(), fwd,
        D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, MAX_PERIOD, EPS, ctx,
    )

    print("")
    print("---- load-bearing input-token + embed grads vs torch ----")
    _check(harness, "d_img_tokens", g.d_img_tokens, _in("ref_d_img_tokens"), allok)
    _check(harness, "d_txt_tokens", g.d_txt_tokens, _in("ref_d_txt_tokens"), allok)
    _check(harness, "d_vec       ", g.d_vec, _in("ref_d_vec"), allok)
    _check(harness, "d_timestep  ", g.d_timestep, _in("ref_d_timestep"), allok)
    _check(harness, "d_guidance  ", g.d_guidance, _in("ref_d_guidance"), allok)
    _check(harness, "d_vector    ", g.d_vector, _in("ref_d_vector"), allok)

    print("")
    print("---- sample per-block weight grads vs torch ----")
    _check(harness, "d0 img d_wqkv ", g.dbl_grads[0].img.d_wqkv, _in("ref_d0_img_wqkv"), allok)
    _check(harness, "d0 img d_wproj", g.dbl_grads[0].img.d_wproj, _in("ref_d0_img_wproj"), allok)
    _check(harness, "d0 img d_wmlp0", g.dbl_grads[0].img.d_wmlp0, _in("ref_d0_img_wmlp0"), allok)
    _check(harness, "d0 txt d_wqkv ", g.dbl_grads[0].txt.d_wqkv, _in("ref_d0_txt_wqkv"), allok)
    _check(harness, "dL img d_wqkv ", g.dbl_grads[NUM_DOUBLE - 1].img.d_wqkv, _in("ref_dL_img_wqkv"), allok)
    _check(harness, "s0 d_w1       ", g.sgl_grads[0].d_w1, _in("ref_s0_w1"), allok)
    _check(harness, "s0 d_w2       ", g.sgl_grads[0].d_w2, _in("ref_s0_w2"), allok)
    _check(harness, "sL d_w1       ", g.sgl_grads[NUM_SINGLE - 1].d_w1, _in("ref_sL_w1"), allok)

    print("")
    if allok:
        print("VERDICT: PASS — Flux FULL stack fwd+bwd composes correctly (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one output diverged (see FAIL lines above)")
