# serenitymojo/models/sd35/parity/sd35_block_device_parity.mojo
#
# HOST-vs-DEVICE parity gate for the device-resident SD3.5 joint LoRA block
# (models/sd35/sd35_block_device.mojo, MJ-1069 port rung 1).
#
# HOST oracle = sd35_joint_block_forward/backward (the all-host-list block).
# BYTE-IDENTICAL inputs; bars (qwen-port precedent):
#   fwd ctx_out/x_out + bwd d_ctx/d_x : cos >= 0.999
#   all 16 LoRA d_A/d_B slots          : cos >= 0.999 AND rel-L2 <= 1e-3
# at REAL H=38, Dh=64, D=2432 with small N_CTX=6, N_IMG=8, S=14, MLP=64.
#
# Build+run: standard -O2 + cshim link line.

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.training.train_step import LoraAdapter

from serenitymojo.models.sd35.sd35_block import (
    ModVecs, StreamWeights, JointBlockWeights,
    sd35_joint_block_forward, sd35_joint_block_backward,
    sd35_context_preonly_forward, sd35_context_preonly_backward,
)
from serenitymojo.models.sd35.sd35_block_device import (
    SD35ModVecsDevice, sd35_modvecs_to_device,
    SD35JointBlockWeightsDevice, sd35_joint_block_weights_to_device,
    SD35StreamWeightsDevice,
    SD35StreamLoraDevice, SD35JointBlockLoraDevice, sd35_opt_lora_to_device,
    sd35_joint_block_forward_device, sd35_joint_block_backward_device,
    sd35_context_preonly_forward_device, sd35_context_preonly_backward_device,
    SD_CQKV, SD_CPROJ, SD_CFC1, SD_CFC2, SD_XQKV, SD_XPROJ, SD_XFC1, SD_XFC2,
)
from serenitymojo.models.sd35.sd35_block_device import _t_f32 as _dev_f32
from serenitymojo.models.sd35.sd35_block_device import sd35_stream_weights_to_device


def sd35_stream_weights_to_device_pub(w: StreamWeights, ctx: DeviceContext) raises -> SD35StreamWeightsDevice:
    return sd35_stream_weights_to_device(w, D, MLP, Dh, ctx)


comptime TArc = ArcPointer[Tensor]
comptime H = 38
comptime Dh = 64
comptime D = H * Dh          # 2432
comptime N_CTX = 6
comptime N_IMG = 8
comptime S_D = N_CTX + N_IMG # 14
comptime MLP = 64
comptime RANK = 4
comptime LSCALE = Float32(2.0)
comptime EPS = Float32(1e-06)
comptime QK_EPS = Float32(1e-06)


def _next(mut s: UInt64) -> UInt64:
    s = s + 0x9E3779B97F4A7C15
    var z = s
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _randf(mut s: UInt64, r: Float32) -> Float32:
    var u = _next(s)
    var frac = Float32(u >> 40) / Float32(16777216.0)
    return (frac * Float32(2.0) - Float32(1.0)) * r


def _rl(mut s: UInt64, n: Int, r: Float32) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(_randf(s, r))
    return o^


def _rl_near1(mut s: UInt64, n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(1.0) + _randf(s, Float32(0.1)))
    return o^


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(0.0)
    return o^


def _adapter(mut s: UInt64, in_f: Int, out_f: Int) raises -> LoraAdapter:
    return LoraAdapter(
        _rl(s, RANK * in_f, Float32(0.05)), _rl(s, out_f * RANK, Float32(0.05)),
        RANK, in_f, out_f, LSCALE,
        _zeros(RANK * in_f), _zeros(RANK * in_f),
        _zeros(out_f * RANK), _zeros(out_f * RANK),
    )


def _stream_weights(mut s: UInt64) raises -> StreamWeights:
    return StreamWeights(
        _rl(s, 3 * D * D, Float32(0.04)), _rl(s, 3 * D, Float32(0.02)),
        _rl(s, D * D, Float32(0.04)), _rl(s, D, Float32(0.02)),
        _rl(s, MLP * D, Float32(0.04)), _rl(s, MLP, Float32(0.02)),
        _rl(s, D * MLP, Float32(0.04)), _rl(s, D, Float32(0.02)),
        _rl_near1(s, Dh), _rl_near1(s, Dh),
    )


def _mod(mut s: UInt64) raises -> ModVecs:
    return ModVecs(
        _rl(s, D, Float32(0.2)), _rl(s, D, Float32(0.2)), _rl(s, D, Float32(0.5)),
        _rl(s, D, Float32(0.2)), _rl(s, D, Float32(0.2)), _rl(s, D, Float32(0.5)),
    )


def _rel_l2(a: List[Float32], b: List[Float32]) -> Float64:
    var num: Float64 = 0.0
    var den: Float64 = 0.0
    for i in range(len(a)):
        var d = Float64(a[i]) - Float64(b[i])
        num += d * d
        den += Float64(b[i]) * Float64(b[i])
    if den == 0.0:
        return 0.0 if num == 0.0 else 1.0e30
    return sqrt(num / den)


def _ck(mut harness: ParityHarness, name: String, dev: Tensor, host: List[Float32],
        mut allok: Bool, ctx: DeviceContext) raises:
    var actual = dev.to_host(ctx)
    var r = harness.compare_host(actual, host)
    print("  cos(", name, ") =", r.cos, "  max_abs =", r.max_abs, "  n =", r.n,
          "  ", "PASS" if r.passed else "FAIL")
    if not r.passed:
        allok = False


def _ck_lora(mut harness: ParityHarness, name: String, dev: Tensor, host: List[Float32],
             mut allok: Bool, ctx: DeviceContext) raises:
    var actual = dev.to_host(ctx)
    var r = harness.compare_host(actual, host)
    var rl2 = _rel_l2(actual, host)
    var ok = r.passed and (rl2 <= 1.0e-3)
    print("  cos(", name, ") =", r.cos, "  relL2 =", rl2, "  n =", r.n,
          "  ", "PASS" if ok else "FAIL")
    if not ok:
        allok = False


def main() raises:
    var ctx = DeviceContext()
    print("==== sd35 DEVICE-vs-HOST joint-block parity (fwd+bwd+16 LoRA slots) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " N_CTX=", N_CTX, " N_IMG=", N_IMG,
          " S=", S_D, " MLP=", MLP, " RANK=", RANK)
    var harness = ParityHarness()
    var allok = True
    var s: UInt64 = 0x5D35_2026_0706_0001

    var ctx_in = _rl(s, N_CTX * D, Float32(1.0))
    var x_in = _rl(s, N_IMG * D, Float32(1.0))
    var cw = _stream_weights(s)
    var xw = _stream_weights(s)
    var w = JointBlockWeights(cw^, xw^)
    var cm = _mod(s)
    var xm = _mod(s)
    var c_qkv = _adapter(s, D, 3 * D)
    var c_proj = _adapter(s, D, D)
    var c_fc1 = _adapter(s, D, MLP)
    var c_fc2 = _adapter(s, MLP, D)
    var x_qkv = _adapter(s, D, 3 * D)
    var x_proj = _adapter(s, D, D)
    var x_fc1 = _adapter(s, D, MLP)
    var x_fc2 = _adapter(s, MLP, D)
    var d_ctx_h = _rl(s, N_CTX * D, Float32(1.0))
    var d_x_h = _rl(s, N_IMG * D, Float32(1.0))
    var attn_scale = Float32(1.0) / sqrt(Float32(Dh))

    # ── HOST oracle ──
    var hfwd = sd35_joint_block_forward[1, S_D, H, Dh](
        ctx_in.copy(), x_in.copy(), w, cm, xm,
        N_CTX, N_IMG, D, MLP, EPS, QK_EPS, attn_scale, ctx,
        None,
        Optional[LoraAdapter](c_qkv.copy()), Optional[LoraAdapter](c_proj.copy()),
        Optional[LoraAdapter](c_fc1.copy()), Optional[LoraAdapter](c_fc2.copy()),
        Optional[LoraAdapter](x_qkv.copy()), Optional[LoraAdapter](x_proj.copy()),
        Optional[LoraAdapter](x_fc1.copy()), Optional[LoraAdapter](x_fc2.copy()),
        None,
    )
    var hbwd = sd35_joint_block_backward[1, S_D, H, Dh](
        d_ctx_h.copy(), d_x_h.copy(), w, cm, xm, hfwd,
        N_CTX, N_IMG, D, MLP, EPS, QK_EPS, attn_scale, ctx,
        Optional[LoraAdapter](c_qkv.copy()), Optional[LoraAdapter](c_proj.copy()),
        Optional[LoraAdapter](c_fc1.copy()), Optional[LoraAdapter](c_fc2.copy()),
        Optional[LoraAdapter](x_qkv.copy()), Optional[LoraAdapter](x_proj.copy()),
        Optional[LoraAdapter](x_fc1.copy()), Optional[LoraAdapter](x_fc2.copy()),
    )

    # ── DEVICE under test (byte-identical carriers) ──
    var ctx_dev = TArc(Tensor.from_host(ctx_in.copy(), [N_CTX, D], STDtype.F32, ctx))
    var x_dev = TArc(Tensor.from_host(x_in.copy(), [N_IMG, D], STDtype.F32, ctx))
    var wd = sd35_joint_block_weights_to_device(w, D, MLP, Dh, ctx)
    var cmd = sd35_modvecs_to_device(cm, D, ctx)
    var xmd = sd35_modvecs_to_device(xm, D, ctx)
    var lora_d = SD35JointBlockLoraDevice(
        SD35StreamLoraDevice(
            sd35_opt_lora_to_device(Optional[LoraAdapter](c_qkv.copy()), ctx),
            sd35_opt_lora_to_device(Optional[LoraAdapter](c_proj.copy()), ctx),
            sd35_opt_lora_to_device(Optional[LoraAdapter](c_fc1.copy()), ctx),
            sd35_opt_lora_to_device(Optional[LoraAdapter](c_fc2.copy()), ctx),
        ),
        SD35StreamLoraDevice(
            sd35_opt_lora_to_device(Optional[LoraAdapter](x_qkv.copy()), ctx),
            sd35_opt_lora_to_device(Optional[LoraAdapter](x_proj.copy()), ctx),
            sd35_opt_lora_to_device(Optional[LoraAdapter](x_fc1.copy()), ctx),
            sd35_opt_lora_to_device(Optional[LoraAdapter](x_fc2.copy()), ctx),
        ),
    )
    var d_ctx_dev = TArc(Tensor.from_host(d_ctx_h.copy(), [N_CTX, D], STDtype.F32, ctx))
    var d_x_dev = TArc(Tensor.from_host(d_x_h.copy(), [N_IMG, D], STDtype.F32, ctx))

    var dfwd = sd35_joint_block_forward_device[H, Dh, N_CTX, N_IMG, S_D, False](
        ctx_dev, x_dev, wd, cmd, xmd, lora_d, D, MLP, EPS, QK_EPS, attn_scale, ctx)
    var dbwd = sd35_joint_block_backward_device[H, Dh, N_CTX, N_IMG, S_D, False](
        d_ctx_dev, d_x_dev, wd, cmd, xmd, lora_d, dfwd.saved,
        D, MLP, EPS, QK_EPS, attn_scale, ctx)

    print("---- forward outputs ----")
    _ck(harness, "ctx_out", dfwd.ctx_out[], hfwd.ctx_out, allok, ctx)
    _ck(harness, "x_out", dfwd.x_out[], hfwd.x_out, allok, ctx)
    print("---- input grads ----")
    _ck(harness, "d_ctx", dbwd.d_ctx[], hbwd.d_ctx, allok, ctx)
    _ck(harness, "d_x", dbwd.d_x[], hbwd.d_x, allok, ctx)
    print("---- CTX LoRA slots ----")
    _ck_lora(harness, "c dA qkv", dbwd.d_a[SD_CQKV].value()[], hbwd.ctx_lora.qkv_d_a, allok, ctx)
    _ck_lora(harness, "c dB qkv", dbwd.d_b[SD_CQKV].value()[], hbwd.ctx_lora.qkv_d_b, allok, ctx)
    _ck_lora(harness, "c dA proj", dbwd.d_a[SD_CPROJ].value()[], hbwd.ctx_lora.proj_d_a, allok, ctx)
    _ck_lora(harness, "c dB proj", dbwd.d_b[SD_CPROJ].value()[], hbwd.ctx_lora.proj_d_b, allok, ctx)
    _ck_lora(harness, "c dA fc1", dbwd.d_a[SD_CFC1].value()[], hbwd.ctx_lora.fc1_d_a, allok, ctx)
    _ck_lora(harness, "c dB fc1", dbwd.d_b[SD_CFC1].value()[], hbwd.ctx_lora.fc1_d_b, allok, ctx)
    _ck_lora(harness, "c dA fc2", dbwd.d_a[SD_CFC2].value()[], hbwd.ctx_lora.fc2_d_a, allok, ctx)
    _ck_lora(harness, "c dB fc2", dbwd.d_b[SD_CFC2].value()[], hbwd.ctx_lora.fc2_d_b, allok, ctx)
    print("---- X LoRA slots ----")
    _ck_lora(harness, "x dA qkv", dbwd.d_a[SD_XQKV].value()[], hbwd.x_lora.qkv_d_a, allok, ctx)
    _ck_lora(harness, "x dB qkv", dbwd.d_b[SD_XQKV].value()[], hbwd.x_lora.qkv_d_b, allok, ctx)
    _ck_lora(harness, "x dA proj", dbwd.d_a[SD_XPROJ].value()[], hbwd.x_lora.proj_d_a, allok, ctx)
    _ck_lora(harness, "x dB proj", dbwd.d_b[SD_XPROJ].value()[], hbwd.x_lora.proj_d_b, allok, ctx)
    _ck_lora(harness, "x dA fc1", dbwd.d_a[SD_XFC1].value()[], hbwd.x_lora.fc1_d_a, allok, ctx)
    _ck_lora(harness, "x dB fc1", dbwd.d_b[SD_XFC1].value()[], hbwd.x_lora.fc1_d_b, allok, ctx)
    _ck_lora(harness, "x dA fc2", dbwd.d_a[SD_XFC2].value()[], hbwd.x_lora.fc2_d_a, allok, ctx)
    _ck_lora(harness, "x dB fc2", dbwd.d_b[SD_XFC2].value()[], hbwd.x_lora.fc2_d_b, allok, ctx)

    # ════════════════ context_pre_only (final Large block) ════════════════
    print("")
    print("################ PRE_ONLY BLOCK ################")
    var pw = _stream_weights(s)   # reuse a fresh x-stream weights set
    var pm = _mod(s)
    var p_ctx_qkv_w = _rl(s, 3 * D * D, Float32(0.04))
    var p_ctx_qkv_b = _rl(s, 3 * D, Float32(0.02))
    var p_ctx_qn = _rl_near1(s, Dh)
    var p_ctx_kn = _rl_near1(s, Dh)
    var p_ctx_scale = _rl(s, D, Float32(0.2))
    var p_ctx_shift = _rl(s, D, Float32(0.2))
    var p_cqkv_lora = _adapter(s, D, 3 * D)
    var p_x_qkv = _adapter(s, D, 3 * D)
    var p_x_proj = _adapter(s, D, D)
    var p_x_fc1 = _adapter(s, D, MLP)
    var p_x_fc2 = _adapter(s, MLP, D)
    var p_ctx_in = _rl(s, N_CTX * D, Float32(1.0))
    var p_x_in = _rl(s, N_IMG * D, Float32(1.0))
    var p_dx = _rl(s, N_IMG * D, Float32(1.0))

    var hp_fwd = sd35_context_preonly_forward[1, S_D, H, Dh](
        p_ctx_in.copy(), p_x_in.copy(),
        p_ctx_qkv_w, p_ctx_qkv_b, p_ctx_qn, p_ctx_kn, p_ctx_scale, p_ctx_shift,
        pw, pm, N_CTX, N_IMG, D, MLP, EPS, QK_EPS, attn_scale, ctx,
        Optional[LoraAdapter](p_cqkv_lora.copy()),
        Optional[LoraAdapter](p_x_qkv.copy()), Optional[LoraAdapter](p_x_proj.copy()),
        Optional[LoraAdapter](p_x_fc1.copy()), Optional[LoraAdapter](p_x_fc2.copy()),
    )
    var hp_bwd = sd35_context_preonly_backward[1, S_D, H, Dh](
        p_dx.copy(), p_ctx_qkv_w, p_ctx_qn, p_ctx_kn, p_ctx_scale,
        pw, pm, hp_fwd, N_CTX, N_IMG, D, MLP, EPS, QK_EPS, attn_scale, ctx,
        Optional[LoraAdapter](p_cqkv_lora.copy()),
        Optional[LoraAdapter](p_x_qkv.copy()), Optional[LoraAdapter](p_x_proj.copy()),
        Optional[LoraAdapter](p_x_fc1.copy()), Optional[LoraAdapter](p_x_fc2.copy()),
    )

    var p_ctx_dev = TArc(Tensor.from_host(p_ctx_in.copy(), [N_CTX, D], STDtype.F32, ctx))
    var p_x_dev = TArc(Tensor.from_host(p_x_in.copy(), [N_IMG, D], STDtype.F32, ctx))
    var pwd = sd35_stream_weights_to_device_pub(pw, ctx)
    var pmd = sd35_modvecs_to_device(pm, D, ctx)
    var p_xlora_d = SD35StreamLoraDevice(
        sd35_opt_lora_to_device(Optional[LoraAdapter](p_x_qkv.copy()), ctx),
        sd35_opt_lora_to_device(Optional[LoraAdapter](p_x_proj.copy()), ctx),
        sd35_opt_lora_to_device(Optional[LoraAdapter](p_x_fc1.copy()), ctx),
        sd35_opt_lora_to_device(Optional[LoraAdapter](p_x_fc2.copy()), ctx),
    )
    var p_cqkv_d = sd35_opt_lora_to_device(Optional[LoraAdapter](p_cqkv_lora.copy()), ctx)
    var p_dx_dev = TArc(Tensor.from_host(p_dx.copy(), [N_IMG, D], STDtype.F32, ctx))
    var t_cqkv_w = TArc(_dev_f32(p_ctx_qkv_w.copy(), [3 * D, D], ctx))
    var t_cqkv_b = TArc(_dev_f32(p_ctx_qkv_b.copy(), [3 * D], ctx))
    var t_cqn = TArc(_dev_f32(p_ctx_qn.copy(), [Dh], ctx))
    var t_ckn = TArc(_dev_f32(p_ctx_kn.copy(), [Dh], ctx))
    var t_cs = TArc(_dev_f32(p_ctx_scale.copy(), [D], ctx))
    var t_csh = TArc(_dev_f32(p_ctx_shift.copy(), [D], ctx))

    var dp_fwd = sd35_context_preonly_forward_device[H, Dh, N_CTX, N_IMG, S_D, False](
        p_ctx_dev, p_x_dev, t_cqkv_w, t_cqkv_b, t_cqn, t_ckn, t_cs, t_csh,
        pwd, pmd, p_cqkv_d, p_xlora_d, D, MLP, EPS, QK_EPS, attn_scale, ctx)
    var dp_bwd = sd35_context_preonly_backward_device[H, Dh, N_CTX, N_IMG, S_D, False](
        p_dx_dev, t_cqkv_w, t_cqn, t_ckn, t_cs,
        pwd, pmd, p_cqkv_d, p_xlora_d, dp_fwd.saved,
        D, MLP, EPS, QK_EPS, attn_scale, ctx)

    print("---- pre_only forward/grads ----")
    _ck(harness, "p x_out", dp_fwd.x_out[], hp_fwd.x_out, allok, ctx)
    _ck(harness, "p d_x", dp_bwd.d_x[], hp_bwd.d_x, allok, ctx)
    _ck(harness, "p d_ctx", dp_bwd.d_ctx[], hp_bwd.d_ctx, allok, ctx)
    print("---- pre_only LoRA slots ----")
    _ck_lora(harness, "p c dA qkv", dp_bwd.d_a[SD_CQKV].value()[], hp_bwd.ctx_qkv_lora_d_a, allok, ctx)
    _ck_lora(harness, "p c dB qkv", dp_bwd.d_b[SD_CQKV].value()[], hp_bwd.ctx_qkv_lora_d_b, allok, ctx)
    _ck_lora(harness, "p x dA qkv", dp_bwd.d_a[SD_XQKV].value()[], hp_bwd.x_lora.qkv_d_a, allok, ctx)
    _ck_lora(harness, "p x dB qkv", dp_bwd.d_b[SD_XQKV].value()[], hp_bwd.x_lora.qkv_d_b, allok, ctx)
    _ck_lora(harness, "p x dA proj", dp_bwd.d_a[SD_XPROJ].value()[], hp_bwd.x_lora.proj_d_a, allok, ctx)
    _ck_lora(harness, "p x dB proj", dp_bwd.d_b[SD_XPROJ].value()[], hp_bwd.x_lora.proj_d_b, allok, ctx)
    _ck_lora(harness, "p x dA fc1", dp_bwd.d_a[SD_XFC1].value()[], hp_bwd.x_lora.fc1_d_a, allok, ctx)
    _ck_lora(harness, "p x dB fc1", dp_bwd.d_b[SD_XFC1].value()[], hp_bwd.x_lora.fc1_d_b, allok, ctx)
    _ck_lora(harness, "p x dA fc2", dp_bwd.d_a[SD_XFC2].value()[], hp_bwd.x_lora.fc2_d_a, allok, ctx)
    _ck_lora(harness, "p x dB fc2", dp_bwd.d_b[SD_XFC2].value()[], hp_bwd.x_lora.fc2_d_b, allok, ctx)

    # ── FLASH arm (value-class vs the math oracle; bf16 cuDNN flash) ──
    print("")
    print("################ FLASH ARM (value-class) ################")
    var ffwd = sd35_joint_block_forward_device[H, Dh, N_CTX, N_IMG, S_D, True](
        ctx_dev, x_dev, wd, cmd, xmd, lora_d, D, MLP, EPS, QK_EPS, attn_scale, ctx)
    var fbwd = sd35_joint_block_backward_device[H, Dh, N_CTX, N_IMG, S_D, True](
        d_ctx_dev, d_x_dev, wd, cmd, xmd, lora_d, ffwd.saved,
        D, MLP, EPS, QK_EPS, attn_scale, ctx)
    var fr1 = harness.compare_host(ffwd.x_out[].to_host(ctx), hfwd.x_out)
    var fr2 = harness.compare_host(fbwd.d_x[].to_host(ctx), hbwd.d_x)
    var fr3 = harness.compare_host(
        fbwd.d_b[SD_XQKV].value()[].to_host(ctx), hbwd.x_lora.qkv_d_b)
    print("  flash cos(x_out) =", fr1.cos, " cos(d_x) =", fr2.cos,
          " cos(x dB qkv) =", fr3.cos)
    var flash_ok = fr1.cos >= 0.999 and fr2.cos >= 0.999 and fr3.cos >= 0.995
    print("  flash value-class:", "PASS" if flash_ok else "FAIL")
    if not flash_ok:
        allok = False

    print("")
    if allok:
        print("VERDICT: PASS - sd35 device joint block matches host (standard + pre_only)")
    else:
        print("VERDICT: FAIL - at least one tensor diverged")
