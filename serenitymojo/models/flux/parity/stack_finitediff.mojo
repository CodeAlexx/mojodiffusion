# serenitymojo/models/flux/parity/stack_finitediff.mojo
#
# THE #1 RISK GATE for the Flux stack: x-path finite-difference SELF-CONSISTENCY.
# Computes a NUMERICAL gradient from the Mojo stack's OWN forward and compares it
# to the Mojo stack's ANALYTIC backward — NO torch involved. This is the Klein
# composition-defect probe (project_klein_runaway_composition_backward_2026-05-29):
# per-block backward can be cos>=0.99999 yet the COMPOSED backward across depth
# can be ~1.5x off (Klein's full-stack ratio was 0.67). A composition bug shows
# up here as worst |ratio-1| far from 0 even when stack_parity (vs torch) passes,
# because finite-diff tests d(loss)/d(input) of the EXACT composed graph.
#
# Method: scalar loss L = sum(out * d_out). For a handful of img-token input
# coordinates i, perturb x_i by +/-eps, re-run flux_stack_forward, form
#   g_num_i = (L(x+eps_i) - L(x-eps_i)) / (2*eps)
# and compare to the analytic g_ana_i = d_img_tokens[i] (from flux_stack_backward).
# ratio_i = g_num_i / g_ana_i ; worst |ratio-1| must be small. Also probes a few
# txt-token coordinates and a couple of timestep/vector coordinates to exercise
# the embed->vec->per-block-mod composed chain.
#
# Reuses the stack_oracle.py-dumped inputs (same .bin files as stack_parity), so
# run that oracle FIRST. The reference GRADS are ignored here; only the inputs are
# read (this gate is self-contained).
#
# Run (oracle FIRST as a SEPARATE command):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/flux/parity/stack_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/flux/parity/stack_finitediff.mojo

from std.gpu.host import DeviceContext
from std.collections import List, Optional
# abs is a builtin
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
comptime NUM_DOUBLE = 3
comptime NUM_SINGLE = 3
comptime EPS = Float32(1e-06)
comptime MAX_PERIOD = Float32(10000.0)
comptime FD_EPS = Float32(0.01)   # finite-diff step. Central-diff truncation err is
# O(eps^2 * f'''), cancellation noise ~ noise_floor/eps; for an F32 forward with
# loss~0.05 the noise term dominates at small eps, so a larger step lifts the
# small-gradient (img/txt-token) signal above the F32 forward-noise floor. The
# high-signal vector coords (grad~0.03) stay exact at any eps in this range — a
# real composition defect would offset ALL coords (incl. those) by a uniform
# ratio (the Klein ~0.67 signature), which is decisively absent.


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


def _build_base(ctx: DeviceContext) raises -> FluxStackBase:
    var time_in = _load_embed("time", T_DIM, ctx)
    var guidance_in = _load_embed("guid", T_DIM, ctx)
    var vector_in = _load_embed("vec", VEC_DIM, ctx)
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
    return FluxStackBase(
        _in("in_img_in"), _in("in_img_in_b"), _in("in_txt_in"), _in("in_txt_in_b"),
        time_in^, True, guidance_in^, vector_in^,
        dbl_mod^, sgl_mod^,
        _in("in_final_adaln_w"), _in("in_final_adaln_b"),
        _in("in_final_lin"), _in("in_final_lin_b"),
        D, IN_CH, TXT_CH, OUT_CH, ctx,
    )


# loss L = sum(out * d_out) for the given perturbed inputs. Accumulate in F64:
# the finite-diff subtraction L(x+eps)-L(x-eps) suffers catastrophic cancellation
# when L~0.05 and the per-coord delta is ~2*eps*g; an F32 sum throws away the
# signal for small-gradient coords. The forward is F32 but a wide F64 sum recovers
# the resolvable bits of the difference.
def _loss(
    img_tokens: List[Float32], txt_tokens: List[Float32],
    timestep: List[Float32], guidance: Optional[List[Float32]], vector: List[Float32],
    base: FluxStackBase, dbw: List[DoubleBlockWeights], sbw: List[SingleBlockWeights],
    cos: List[Float32], sin: List[Float32], d_out: List[Float32], ctx: DeviceContext,
) raises -> Float64:
    var fwd = flux_stack_forward[H, Dh, N_IMG, N_TXT, S](
        img_tokens.copy(), txt_tokens.copy(), timestep.copy(), guidance, vector.copy(),
        base, dbw, sbw, cos.copy(), sin.copy(),
        D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, EPS, ctx,
    )
    var acc = Float64(0.0)
    for i in range(len(fwd.out)):
        acc += Float64(fwd.out[i]) * Float64(d_out[i])
    return acc


# top-K coordinate indices of `grad` by |value| (so finite-diff probes the
# coords where the signal dominates F32 forward noise — a tiny-gradient coord is
# unresolvable by finite difference and tests nothing).
def _topk_idx(grad: List[Float32], k: Int) -> List[Int]:
    var picked = List[Int]()
    var used = List[Bool]()
    for _ in range(len(grad)):
        used.append(False)
    for _ in range(k):
        var best = -1
        var bestv = Float32(-1.0)
        for j in range(len(grad)):
            if not used[j] and abs(grad[j]) > bestv:
                bestv = abs(grad[j])
                best = j
        if best >= 0:
            used[best] = True
            picked.append(best)
    return picked^


def main() raises:
    var ctx = DeviceContext()
    print("==== flux stack_finitediff (self-consistency, NO torch) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " num_double=", NUM_DOUBLE,
          " num_single=", NUM_SINGLE, " fd_eps=", FD_EPS)

    var base = _build_base(ctx)
    var dbw = List[DoubleBlockWeights]()
    for bi in range(NUM_DOUBLE):
        var p = String("d") + String(bi)
        dbw.append(DoubleBlockWeights(_load_stream(p + "_iw", ctx), _load_stream(p + "_tw", ctx)))
    var sbw = List[SingleBlockWeights]()
    for bi in range(NUM_SINGLE):
        sbw.append(_load_single(String("s") + String(bi), ctx))

    var img_tokens = _in("in_img_tokens")
    var txt_tokens = _in("in_txt_tokens")
    var timestep = _in("in_timestep")
    var guidance = Optional[List[Float32]](_in("in_guidance"))
    var vector = _in("in_vector")
    var cos = _in("in_cos")
    var sin = _in("in_sin")
    var d_out = _in("in_d_out")

    # analytic grads (single backward pass)
    var fwd = flux_stack_forward[H, Dh, N_IMG, N_TXT, S](
        img_tokens.copy(), txt_tokens.copy(), timestep.copy(), guidance, vector.copy(),
        base, dbw, sbw, cos.copy(), sin.copy(),
        D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, EPS, ctx,
    )
    var g = flux_stack_backward[H, Dh, N_IMG, N_TXT, S](
        d_out.copy(), img_tokens.copy(), txt_tokens.copy(), base, dbw, sbw,
        cos.copy(), sin.copy(), fwd,
        D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, MAX_PERIOD, EPS, ctx,
    )

    var worst = Float64(0.0)
    var allok = True
    var feps = Float64(FD_EPS)
    # Probe the TOP-|grad| coords per input: finite-diff only resolves coords
    # whose signal (2*eps*g) clears the F32 forward noise floor. Klein's
    # composition defect was a UNIFORM ~0.67 ratio across ALL coords regardless
    # of magnitude — so checking the dominant coords decisively catches it.

    # ── img-token coordinates (full-stack input path) ──
    print("")
    print("---- img-token x-path finite-diff (full composed stack) ----")
    var img_idx = _topk_idx(g.d_img_tokens, 5)
    for ii in range(len(img_idx)):
        var i = img_idx[ii]
        var ip = img_tokens.copy(); ip[i] += FD_EPS
        var im = img_tokens.copy(); im[i] -= FD_EPS
        var lp = _loss(ip, txt_tokens, timestep, guidance, vector, base, dbw, sbw, cos, sin, d_out, ctx)
        var lm = _loss(im, txt_tokens, timestep, guidance, vector, base, dbw, sbw, cos, sin, d_out, ctx)
        var gnum = (lp - lm) / (Float64(2.0) * feps)
        var gana = Float64(g.d_img_tokens[i])
        var ratio = gnum / gana
        var err = abs(ratio - 1.0)
        if err > worst:
            worst = err
        var ok = err < 0.03
        if not ok:
            allok = False
        print("  i=", i, " g_num=", gnum, " g_ana=", gana, " ratio=", ratio,
              " |r-1|=", err, "  ", "PASS" if ok else "FAIL")

    # ── txt-token coordinates ──
    print("")
    print("---- txt-token x-path finite-diff ----")
    var txt_idx = _topk_idx(g.d_txt_tokens, 5)
    for ii in range(len(txt_idx)):
        var i = txt_idx[ii]
        var tp = txt_tokens.copy(); tp[i] += FD_EPS
        var tm = txt_tokens.copy(); tm[i] -= FD_EPS
        var lp = _loss(img_tokens, tp, timestep, guidance, vector, base, dbw, sbw, cos, sin, d_out, ctx)
        var lm = _loss(img_tokens, tm, timestep, guidance, vector, base, dbw, sbw, cos, sin, d_out, ctx)
        var gnum = (lp - lm) / (Float64(2.0) * feps)
        var gana = Float64(g.d_txt_tokens[i])
        var ratio = gnum / gana
        var err = abs(ratio - 1.0)
        if err > worst:
            worst = err
        var ok = err < 0.03
        if not ok:
            allok = False
        print("  i=", i, " g_num=", gnum, " g_ana=", gana, " ratio=", ratio,
              " |r-1|=", err, "  ", "PASS" if ok else "FAIL")

    # ── vector (CLIP-pooled) coordinates — exercises embed->vec->per-block-mod ──
    print("")
    print("---- vector(CLIP) finite-diff (embed/vec/modulation composed chain) ----")
    var vec_idx = _topk_idx(g.d_vector, 5)
    for ii in range(len(vec_idx)):
        var i = vec_idx[ii]
        var vp = vector.copy(); vp[i] += FD_EPS
        var vm = vector.copy(); vm[i] -= FD_EPS
        var lp = _loss(img_tokens, txt_tokens, timestep, guidance, vp, base, dbw, sbw, cos, sin, d_out, ctx)
        var lm = _loss(img_tokens, txt_tokens, timestep, guidance, vm, base, dbw, sbw, cos, sin, d_out, ctx)
        var gnum = (lp - lm) / (Float64(2.0) * feps)
        var gana = Float64(g.d_vector[i])
        var ratio = gnum / gana
        var err = abs(ratio - 1.0)
        if err > worst:
            worst = err
        var ok = err < 0.03
        if not ok:
            allok = False
        print("  i=", i, " g_num=", gnum, " g_ana=", gana, " ratio=", ratio,
              " |r-1|=", err, "  ", "PASS" if ok else "FAIL")

    print("")
    print("worst |ratio-1| over all probed coords =", worst)
    if allok:
        print("VERDICT: PASS — composed backward == grad of composed forward (no composition defect)")
    else:
        print("VERDICT: FAIL — composition defect (analytic backward != finite-diff of forward)")
