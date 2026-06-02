# serenitymojo/models/anima/parity/anima_ot_step_parity.mojo
#
# CHUNK-A RECIPE PARITY GATE for the OneTrainer-faithful Anima LoRA STEP.
#
# Reads the FIXED inputs written by anima_ot_step_oracle.py (scaled_latent grid,
# noise grid, frozen context[1,512,1024]) and runs the SAME OT recipe through the
# already-green Anima LoRA stack (resident L=2, real checkpoint weights, LoRA B=0
# so it reduces to the base transformer the oracle computes):
#   sigma  = (FIXED_TIMESTEP+1)/N                      (delta 3)
#   noisy  = noise*sigma + scaled*(1-sigma)            (delta 3)
#   target = noise - scaled                            (delta 3)
#   t_in   = FIXED_TIMESTEP/1000  -> sinusoidal embedder (delta 4)
#   pred   = AnimaStack(noisy patches, t_cond, base_adaln, context[512])
#   loss   = mean((pred - target)^2)  (unmasked MSE)   (delta 5)
# and diffs predicted_flow (cos>=0.999) + loss (rel-err<1e-3) vs the torch oracle.
#
# S_TXT=512 (delta 1) monomorphizes the comptime cross-attn SDPA at 512 — this gate
# is also the proof it COMPILES + RUNS at 512.
#
# Run (oracle FIRST, SEPARATE commands):
#   cd /home/alex/mojodiffusion
#   python3 serenitymojo/models/anima/parity/anima_ot_step_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/models/anima/parity/anima_ot_step_parity.mojo -o /tmp/anima_ot_parity
#   /tmp/anima_ot_parity

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import alloc, ArcPointer
from std.math import sqrt as fsqrt, log as flog, cos as fcos, sin as fsin, exp as fexp

from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors

from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import silu
from serenitymojo.ops.norm import rms_norm

from serenitymojo.models.anima.config import anima
from serenitymojo.models.anima.weights import (
    AnimaBlockWeights, AnimaStackBase,
    load_anima_stack_base, load_anima_block_weights_f32, verify_anima_stack_shapes,
)
from serenitymojo.models.anima.lora_block import ANIMA_SLOTS
from serenitymojo.models.anima.anima_stack_lora import (
    AnimaLoraSet, build_anima_lora_set, anima_stack_lora_forward,
)
from serenitymojo.models.dit.anima_contract import ANIMA_HIDDEN


comptime TArc = ArcPointer[Tensor]
comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/anima/parity/"

# dims MUST match the oracle
comptime B = 1
comptime H = 16
comptime Dh = 128
comptime D = H * Dh        # 2048
comptime JOINT = 1024
comptime F = 8192          # real MLP hidden
comptime C = 16
comptime PS = 2
comptime IN_PATCH = (C + 1) * PS * PS   # 68
comptime OUT_PATCH = C * PS * PS        # 64
comptime EPS = Float32(1e-06)

comptime S_TXT = 512                    # delta 1: OT context length
comptime LATENT_HW = 16
comptime S_IMG = (LATENT_HW // PS) * (LATENT_HW // PS)   # 64
comptime NUM_TRAIN_TIMESTEPS = 1000
comptime FIXED_TIMESTEP = 500
comptime L = 2                          # gate depth (real blocks 0..L-1)
comptime RANK = 16
comptime ALPHA = Float32(16.0)


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


def _ones(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(1.0)
    return o^


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(0.0)
    return o^


# ── patchify INPUT layout (matches train_anima_real._patchify_in) ─────────────
def _patchify_in(x: List[Float32], Hd: Int, Wd: Int) -> List[Float32]:
    var nH = Hd // PS
    var nW = Wd // PS
    var Cp = C + 1
    var N = nH * nW
    var pd = Cp * PS * PS
    var out = _zeros(B * N * pd)
    for b in range(B):
        for ih in range(nH):
            for iw in range(nW):
                var pn = ih * nW + iw
                for c in range(Cp):
                    for ph in range(PS):
                        for pw in range(PS):
                            var od = (b * N + pn) * pd + (c * PS * PS + ph * PS + pw)
                            if c < C:
                                var hh = ih * PS + ph
                                var ww = iw * PS + pw
                                var src = ((b * Hd + hh) * Wd + ww) * C + c
                                out[od] = x[src]
    return out^


# ── patchify OUTPUT layout (matches train_anima_real._patchify_out, C fastest) ─
def _patchify_out(x: List[Float32], Hd: Int, Wd: Int) -> List[Float32]:
    var nH = Hd // PS
    var nW = Wd // PS
    var N = nH * nW
    var pd = C * PS * PS
    var out = _zeros(B * N * pd)
    for b in range(B):
        for ih in range(nH):
            for iw in range(nW):
                var pn = ih * nW + iw
                for ph in range(PS):
                    for pw in range(PS):
                        for c in range(C):
                            var od = (b * N + pn) * pd + (ph * PS * C + pw * C + c)
                            var hh = ih * PS + ph
                            var ww = iw * PS + pw
                            var src = ((b * Hd + hh) * Wd + ww) * C + c
                            out[od] = x[src]
    return out^


# ── cos-first sinusoidal (matches train_anima_real._sinusoidal_host) ──────────
def _sinusoidal_host(val: Float32, dim: Int) -> List[Float32]:
    var half = dim // 2
    var neg_ln = -flog(Float32(10000.0))
    var out = _zeros(dim)
    for i in range(half):
        var freq = fexp(neg_ln * (Float32(i) / Float32(half)))
        var angle = val * freq
        out[i] = fcos(angle)
        out[half + i] = fsin(angle)
    return out^


# ── 3D-RoPE (T,H,W) NTK tables (matches train_anima_ot._rope_tables) ─────────
# REAL Anima 3-axis rope (anima_dit.build_anima_3d_rope / CosmosRotaryPosEmbed),
# rope_scale (t=1.0, h=4.0, w=4.0). Replaces the OT-unfaithful single-axis table.
# Output [B*S_IMG*H, Dh/2] F32, row = (b,s,h), cols [t-bins|h-bins|w-bins].
struct _Rope(Movable):
    var cos: Tensor
    var sin: Tensor

    def __init__(out self, var cos: Tensor, var sin: Tensor):
        self.cos = cos^
        self.sin = sin^


def _rope_tables(s_img: Int, ctx: DeviceContext) raises -> _Rope:
    var half = Dh // 2            # 64
    var full_d = Dh              # 128
    var t_frames = 1
    var nh = LATENT_HW // PS      # 8
    var nw = LATENT_HW // PS      # 8
    if nh * nw != s_img:
        raise Error("rope grid mismatch: nh*nw=" + String(nh * nw)
                    + " != S_IMG=" + String(s_img))

    var dim_h = full_d // 6 * 2   # 42
    var dim_w = dim_h             # 42
    var dim_t = full_d - 2 * dim_h  # 44
    var bins_t = dim_t // 2       # 22
    var bins_h = dim_h // 2       # 21
    var bins_w = dim_w // 2       # 21

    var base_theta: Float64 = 10000.0
    var h_exp = Float64(dim_h) / (Float64(dim_h) - 2.0)
    var w_exp = Float64(dim_w) / (Float64(dim_w) - 2.0)
    var h_ntk = fexp(flog(Float64(4.0)) * h_exp)
    var w_ntk = fexp(flog(Float64(4.0)) * w_exp)
    var t_ntk = Float64(1.0)
    var theta_h = Float64(base_theta * h_ntk)
    var theta_w = Float64(base_theta * w_ntk)
    var theta_t = Float64(base_theta * t_ntk)

    var freqs_t = List[Float32]()
    for i in range(bins_t):
        var ev = Float64(2 * i) / Float64(dim_t)
        freqs_t.append(Float32(fexp(-flog(theta_t) * ev)))
    var freqs_h = List[Float32]()
    for i in range(bins_h):
        var ev = Float64(2 * i) / Float64(dim_h)
        freqs_h.append(Float32(fexp(-flog(theta_h) * ev)))
    var freqs_w = List[Float32]()
    for i in range(bins_w):
        var ev = Float64(2 * i) / Float64(dim_w)
        freqs_w.append(Float32(fexp(-flog(theta_w) * ev)))

    var cosl = List[Float32]()
    var sinl = List[Float32]()
    for _b in range(B):
        for tf in range(t_frames):
            for ih in range(nh):
                for iw in range(nw):
                    for _h in range(H):
                        for fi in range(bins_t):
                            var ang = Float32(tf) * freqs_t[fi]
                            cosl.append(fcos(ang))
                            sinl.append(fsin(ang))
                        for fi in range(bins_h):
                            var ang = Float32(ih) * freqs_h[fi]
                            cosl.append(fcos(ang))
                            sinl.append(fsin(ang))
                        for fi in range(bins_w):
                            var ang = Float32(iw) * freqs_w[fi]
                            cosl.append(fcos(ang))
                            sinl.append(fsin(ang))
    var cos = Tensor.from_host(cosl, [B * s_img * H, half], STDtype.F32, ctx)
    var sin = Tensor.from_host(sinl, [B * s_img * H, half], STDtype.F32, ctx)
    return _Rope(cos^, sin^)


def main() raises:
    var ctx = DeviceContext()
    print("==== anima_ot_step_parity (Chunk A OT recipe vs torch) ====")
    print("B=", B, " H=", H, " Dh=", Dh, " D=", D, " S_IMG=", S_IMG,
          " S_TXT=", S_TXT, " L=", L, " ts=", FIXED_TIMESTEP, " N=", NUM_TRAIN_TIMESTEPS)

    var cfg = anima()
    var st = SafeTensors.open(cfg.checkpoint)
    verify_anima_stack_shapes(st, 28)
    var base = load_anima_stack_base(st, ctx)

    # ── resident real blocks 0..L-1 ──
    var blocks = List[AnimaBlockWeights]()
    for bi in range(L):
        blocks.append(load_anima_block_weights_f32(st, bi, ctx))
    print("loaded base + ", L, " real blocks (F32 resident)")

    # ── fixed inputs from the oracle (scaled latent + noise, channels-last grid) ──
    var scaled = _in("ot_scaled_bthwc")   # [B,1,Hd,Wd,C] flat
    var noise = _in("ot_noise_bthwc")
    var context = _in("ot_context")       # [B,512,1024]
    if len(context) != B * S_TXT * JOINT:
        raise Error(String("context numel ") + String(len(context)) + " != "
                    + String(B * S_TXT * JOINT))
    var n_lat = B * LATENT_HW * LATENT_HW * C
    if len(scaled) != n_lat or len(noise) != n_lat:
        raise Error("scaled/noise numel mismatch")

    # ── delta 3: sigma = (ts+1)/N ; noisy/target ──
    var sigma = Float32(FIXED_TIMESTEP + 1) / Float32(NUM_TRAIN_TIMESTEPS)
    var noisy = List[Float32]()
    var target = List[Float32]()
    noisy.reserve(n_lat)
    target.reserve(n_lat)
    for i in range(n_lat):
        noisy.append(sigma * noise[i] + (Float32(1.0) - sigma) * scaled[i])
        target.append(noise[i] - scaled[i])

    var patches = _patchify_in(noisy, LATENT_HW, LATENT_HW)
    var target_patches = _patchify_out(target, LATENT_HW, LATENT_HW)

    # ── delta 4: timestep/1000 into the sinusoidal embedder ──
    var t_in = Float32(FIXED_TIMESTEP) / Float32(1000.0)
    var emb_l = _sinusoidal_host(t_in, D)
    var emb = Tensor.from_host(emb_l, [B, D], STDtype.F32, ctx)
    var h = linear(emb, base.te_lin1[], Optional[Tensor](None), ctx)
    var hidden = silu(h, ctx)
    var base_adaln = linear(hidden, base.te_lin2[], Optional[Tensor](None), ctx).to_host(ctx)
    var t_cond = rms_norm(emb, base.t_norm[], EPS, ctx).to_host(ctx)

    # ── rope tables ──
    var ropes = _rope_tables(S_IMG, ctx)

    # ── LoRA set with B=0 (reduces to base transformer == oracle) ──
    var lora = build_anima_lora_set(L, D, JOINT, F, RANK, ALPHA)

    # ── forward (resident L=2) at S_TXT=512 ──
    var fwd = anima_stack_lora_forward[H, Dh, S_IMG, S_TXT](
        patches.copy(), t_cond.copy(), base_adaln.copy(), context.copy(),
        base, blocks, lora, ropes.cos, ropes.sin,
        B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
    )

    # ── delta 5: unmasked MSE ──
    var npred = len(fwd.out)
    var sse = Float32(0.0)
    for i in range(npred):
        var diff = fwd.out[i] - target_patches[i]
        sse += diff * diff
    var loss = sse / Float32(npred)

    # ── compare predicted_flow + loss ──
    var ref_pred = _in("ot_pred")
    var ref_loss_l = _in("ot_loss")
    var ref_loss = ref_loss_l[0]

    var harness = ParityHarness(Float64(0.999))
    var rp = harness.compare_host(fwd.out.copy(), ref_pred.copy())
    print("  cos(predicted_flow) =", rp.cos, "  max_abs =", rp.max_abs,
          "  n =", rp.n, "  ", "PASS" if rp.passed else "FAIL")

    var rel = (loss - ref_loss) / ref_loss
    if rel < Float32(0.0):
        rel = -rel
    var loss_ok = rel < Float32(1.0e-3)
    print("  loss_mojo =", loss, "  loss_oracle =", ref_loss,
          "  rel_err =", rel, "  ", "PASS" if loss_ok else "FAIL")
    print("  sigma =", sigma, "  t_in(=ts/1000) =", t_in,
          "  scaled_std(check) computed by oracle")

    if rp.passed and loss_ok:
        print("VERDICT: PASS — predicted_flow cos>=0.999 AND loss rel-err<1e-3")
    else:
        print("VERDICT: FAIL")
