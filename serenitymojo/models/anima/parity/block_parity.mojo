# serenitymojo/models/anima/parity/block_parity.mojo
#
# PARITY GATE for ONE Anima MiniTrainDIT transformer block training unit
# (models/anima/block.mojo). Loads the EXACT inputs + torch-autograd reference
# grads dumped by block_oracle.py, runs anima_block_forward + anima_block_backward,
# and compares the forward output, d_x, d_t_silu, all 20 base weight grads, and
# the 6 AdaLN-LoRA-256 mod-weight grads at cos >= 0.999.
#
# REAL Anima head dims H=16, Dh=128. Small S_img/S_txt (math is S-independent).
# F32 throughout (matches the block's F32 path; clean cos, no BF16 floor).
#
# Run (oracle FIRST, SEPARATE command — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/anima/parity/block_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/anima/parity/block_parity.mojo

from std.gpu.host import DeviceContext
from std.collections import List
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.anima.weights import AnimaBlockWeights
from serenitymojo.models.anima.block import (
    AnimaBlockForward, AnimaBlockGrads,
    anima_block_forward, anima_block_backward,
)


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/anima/parity/"

# dims MUST match block_oracle.py
comptime B = 1
comptime H = 16
comptime Dh = 128
comptime D = H * Dh        # 2048
comptime S_IMG = 6
comptime S_TXT = 8
comptime JOINT = 1024
comptime F = 32
comptime EPS = Float32(1e-06)


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


def _t(name: String, var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(_in(name), shape^, STDtype.F32, ctx)


def _sh(*dims: Int) -> List[Int]:
    var o = List[Int]()
    for d in dims:
        o.append(d)
    return o^


def _load_weights(ctx: DeviceContext) raises -> AnimaBlockWeights:
    return AnimaBlockWeights(
        _t("in_w_sa_mod1", _sh(256, D), ctx), _t("in_w_sa_mod2", _sh(3 * D, 256), ctx),
        _t("in_w_ca_mod1", _sh(256, D), ctx), _t("in_w_ca_mod2", _sh(3 * D, 256), ctx),
        _t("in_w_mlp_mod1", _sh(256, D), ctx), _t("in_w_mlp_mod2", _sh(3 * D, 256), ctx),
        _t("in_w_sa_q", _sh(D, D), ctx), _t("in_w_sa_k", _sh(D, D), ctx),
        _t("in_w_sa_v", _sh(D, D), ctx), _t("in_w_sa_out", _sh(D, D), ctx),
        _t("in_w_sa_qn", _sh(Dh), ctx), _t("in_w_sa_kn", _sh(Dh), ctx),
        _t("in_w_ca_q", _sh(D, D), ctx), _t("in_w_ca_k", _sh(D, JOINT), ctx),
        _t("in_w_ca_v", _sh(D, JOINT), ctx), _t("in_w_ca_out", _sh(D, D), ctx),
        _t("in_w_ca_qn", _sh(Dh), ctx), _t("in_w_ca_kn", _sh(Dh), ctx),
        _t("in_w_mlp1", _sh(F, D), ctx), _t("in_w_mlp2", _sh(D, F), ctx),
    )


# Expand the per-position cos/sin [S_IMG, Dh/2] to [B*S_IMG*H, Dh/2], replicating
# each position's row H times (rope_halfsplit flattens BSHD x to rows
# (b*S+s)*H+h, so every (b,s,*) shares position s — matches anima.rs cos/sin
# [1,1,S,Dh/2] broadcast over B and H).
def _expand_rope(name: String) raises -> List[Float32]:
    var half = Dh // 2
    var per_pos = _in(name)   # [S_IMG, half]
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
    print("==== anima block_parity (Anima MiniTrainDIT block vs torch) ====")
    print("B=", B, " H=", H, " Dh=", Dh, " D=", D, " S_img=", S_IMG, " S_txt=", S_TXT)

    var x = _in("in_x")
    var t_cond = _in("in_t_cond")
    var base_adaln = _in("in_base_adaln")
    var context = _in("in_context")
    var w = _load_weights(ctx)

    var half = Dh // 2
    var cos = Tensor.from_host(_expand_rope("in_cos"), [B * S_IMG * H, half], STDtype.F32, ctx)
    var sin = Tensor.from_host(_expand_rope("in_sin"), [B * S_IMG * H, half], STDtype.F32, ctx)

    var fwd = anima_block_forward[H, Dh, S_IMG, S_TXT](
        x.copy(), t_cond.copy(), base_adaln.copy(), context.copy(),
        w, cos, sin, B, D, JOINT, F, EPS, ctx,
    )

    var harness = ParityHarness()
    var allok = True

    print("")
    print("---- forward output vs torch ----")
    _check(harness, "out", fwd.out, _in("ref_out"), allok)

    var d_out = _in("in_d_out")
    var g = anima_block_backward[H, Dh, S_IMG, S_TXT](
        d_out, fwd.saved, w, cos, sin, B, D, JOINT, F, EPS, ctx,
    )

    print("")
    print("---- input + t_silu grads vs torch ----")
    _check(harness, "d_x     ", g.d_x, _in("ref_d_x"), allok)
    _check(harness, "d_t_silu", g.d_t_silu, _in("ref_d_t_silu"), allok)

    print("")
    print("---- self-attn weight grads vs torch ----")
    _check(harness, "d_sa_q  ", g.d_sa_q, _in("ref_d_sa_q"), allok)
    _check(harness, "d_sa_k  ", g.d_sa_k, _in("ref_d_sa_k"), allok)
    _check(harness, "d_sa_v  ", g.d_sa_v, _in("ref_d_sa_v"), allok)
    _check(harness, "d_sa_out", g.d_sa_out, _in("ref_d_sa_out"), allok)
    _check(harness, "d_sa_qn ", g.d_sa_qn, _in("ref_d_sa_qn"), allok)
    _check(harness, "d_sa_kn ", g.d_sa_kn, _in("ref_d_sa_kn"), allok)

    print("")
    print("---- cross-attn weight grads vs torch ----")
    _check(harness, "d_ca_q  ", g.d_ca_q, _in("ref_d_ca_q"), allok)
    _check(harness, "d_ca_k  ", g.d_ca_k, _in("ref_d_ca_k"), allok)
    _check(harness, "d_ca_v  ", g.d_ca_v, _in("ref_d_ca_v"), allok)
    _check(harness, "d_ca_out", g.d_ca_out, _in("ref_d_ca_out"), allok)
    _check(harness, "d_ca_qn ", g.d_ca_qn, _in("ref_d_ca_qn"), allok)
    _check(harness, "d_ca_kn ", g.d_ca_kn, _in("ref_d_ca_kn"), allok)

    print("")
    print("---- mlp weight grads vs torch ----")
    _check(harness, "d_mlp1  ", g.d_mlp1, _in("ref_d_mlp1"), allok)
    _check(harness, "d_mlp2  ", g.d_mlp2, _in("ref_d_mlp2"), allok)

    print("")
    print("---- AdaLN-LoRA-256 modulation weight grads vs torch ----")
    _check(harness, "d_sa_mod1 ", g.d_sa_mod1, _in("ref_d_sa_mod1"), allok)
    _check(harness, "d_sa_mod2 ", g.d_sa_mod2, _in("ref_d_sa_mod2"), allok)
    _check(harness, "d_ca_mod1 ", g.d_ca_mod1, _in("ref_d_ca_mod1"), allok)
    _check(harness, "d_ca_mod2 ", g.d_ca_mod2, _in("ref_d_ca_mod2"), allok)
    _check(harness, "d_mlp_mod1", g.d_mlp_mod1, _in("ref_d_mlp_mod1"), allok)
    _check(harness, "d_mlp_mod2", g.d_mlp_mod2, _in("ref_d_mlp_mod2"), allok)

    print("")
    if allok:
        print("VERDICT: PASS — Anima block fwd+bwd matches torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one output diverged (see FAIL lines above)")
