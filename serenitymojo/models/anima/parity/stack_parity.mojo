# serenitymojo/models/anima/parity/stack_parity.mojo
#
# COMPOSITION PARITY GATE for the ANIMA FULL 28-block STACK training unit
# (models/anima/anima_stack.mojo) at small depth L=3. Loads the EXACT inputs +
# torch-autograd reference grads from stack_oracle.py, runs anima_stack_forward +
# anima_stack_backward, and compares at cos >= 0.999:
#   * forward output patches (out)
#   * input-patch grad (d_patches) — the full-chain proof through the stack
#   * SUMMED shared grads d_t_silu + d_base_adaln (the composition detail)
#   * base-weight grads d_x_embed, d_fl_lin, d_fl_mod1, d_fl_mod2
#   * per-block weight grads, DEEPEST (L-1) + SHALLOWEST (0): sa_q, mlp2, sa_mod1, ca_v
#
# This proves the COMPOSED backward = grad of the COMPOSED forward (the Klein
# composition-bug lesson: per-block-correct does NOT imply composition-correct).
#
# Run (oracle FIRST, SEPARATE command — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/anima/parity/stack_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/models/anima/parity/stack_parity.mojo -o /tmp/anima_stack_parity
#   /tmp/anima_stack_parity

from std.gpu.host import DeviceContext
from std.collections import List
from std.memory import alloc, ArcPointer
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.anima.weights import AnimaBlockWeights, AnimaStackBase
from serenitymojo.models.anima.anima_stack import (
    AnimaStackForward, AnimaStackGrads,
    anima_stack_forward, anima_stack_backward,
)


comptime TArc = ArcPointer[Tensor]
comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/anima/parity/"

# dims MUST match stack_oracle.py
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
        TArc(_t("in_x_embed", _sh(D, IN_PATCH), ctx)),   # te_lin1 (unused by gate)
        TArc(_t("in_x_embed", _sh(D, IN_PATCH), ctx)),   # te_lin2 (unused by gate)
        TArc(_t("in_fl_lin", _sh(OUT_PATCH, D), ctx)),   # t_norm  (unused by gate)
        TArc(_t("in_fl_mod1", _sh(256, D), ctx)),
        TArc(_t("in_fl_mod2", _sh(2 * D, 256), ctx)),
        TArc(_t("in_fl_lin", _sh(OUT_PATCH, D), ctx)),
    )


# Expand per-position cos/sin [S_IMG, Dh/2] to [B*S_IMG*H, Dh/2] (broadcast over
# B,H) — matches block_parity._expand_rope.
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
    print("==== anima stack_parity (ANIMA FULL STACK composition vs torch) ====")
    print("B=", B, " H=", H, " Dh=", Dh, " D=", D, " S_img=", S_IMG,
          " S_txt=", S_TXT, " F=", F, " L=", L)

    var patches = _in("in_patches")
    var t_cond = _in("in_t_cond")   # RAW t_cond; stack + block silu internally
    var base_adaln = _in("in_base_adaln")
    var context = _in("in_context")
    var base = _load_base(ctx)

    var blocks = List[AnimaBlockWeights]()
    for l in range(L):
        blocks.append(_load_block(l, ctx))

    var half = Dh // 2
    var cos = Tensor.from_host(_expand_rope("in_cos"), [B * S_IMG * H, half], STDtype.F32, ctx)
    var sin = Tensor.from_host(_expand_rope("in_sin"), [B * S_IMG * H, half], STDtype.F32, ctx)

    # ── forward ──
    var fwd = anima_stack_forward[H, Dh, S_IMG, S_TXT](
        patches.copy(), t_cond.copy(), base_adaln.copy(), context.copy(),
        base, blocks, cos, sin,
        B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
    )

    var harness = ParityHarness()
    var allok = True

    print("")
    print("---- forward output vs torch ----")
    _check(harness, "out", fwd.out, _in("ref_out"), allok)

    # ── backward ──
    var d_out = _in("in_d_out")
    var g = anima_stack_backward[H, Dh, S_IMG, S_TXT](
        d_out, patches.copy(), t_cond.copy(), base_adaln.copy(), context.copy(),
        base, blocks, cos, sin, fwd,
        B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
    )

    print("")
    print("---- input-patch grad vs torch (full-chain proof) ----")
    _check(harness, "d_patches", g.d_patches, _in("ref_d_patches"), allok)

    print("")
    print("---- SUMMED shared grads vs torch (composition detail) ----")
    _check(harness, "d_t_silu    ", g.d_t_silu, _in("ref_d_t_silu"), allok)

    print("")
    print("---- base-weight grads vs torch ----")
    _check(harness, "d_x_embed", g.d_x_embed, _in("ref_d_x_embed"), allok)
    _check(harness, "d_fl_lin ", g.d_fl_lin, _in("ref_d_fl_lin"), allok)
    _check(harness, "d_fl_mod1", g.d_fl_mod1, _in("ref_d_fl_mod1"), allok)
    _check(harness, "d_fl_mod2", g.d_fl_mod2, _in("ref_d_fl_mod2"), allok)

    print("")
    print("---- per-block weight grads (deepest L-1) vs torch ----")
    _check(harness, "d_sa_q   (deep)", g.blk_grads[L - 1].d_sa_q, _in("ref_d_sa_q_deep"), allok)
    _check(harness, "d_mlp2   (deep)", g.blk_grads[L - 1].d_mlp2, _in("ref_d_mlp2_deep"), allok)
    _check(harness, "d_sa_mod1(deep)", g.blk_grads[L - 1].d_sa_mod1, _in("ref_d_sa_mod1_deep"), allok)
    _check(harness, "d_ca_v   (deep)", g.blk_grads[L - 1].d_ca_v, _in("ref_d_ca_v_deep"), allok)

    print("")
    print("---- per-block weight grads (shallowest 0) vs torch ----")
    _check(harness, "d_sa_q   (shal)", g.blk_grads[0].d_sa_q, _in("ref_d_sa_q_shallow"), allok)
    _check(harness, "d_mlp2   (shal)", g.blk_grads[0].d_mlp2, _in("ref_d_mlp2_shallow"), allok)
    _check(harness, "d_sa_mod1(shal)", g.blk_grads[0].d_sa_mod1, _in("ref_d_sa_mod1_shallow"), allok)
    _check(harness, "d_ca_v   (shal)", g.blk_grads[0].d_ca_v, _in("ref_d_ca_v_shallow"), allok)

    print("")
    if allok:
        print("VERDICT: PASS — ANIMA full-stack composition fwd+bwd matches torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one output diverged (see FAIL lines above)")
