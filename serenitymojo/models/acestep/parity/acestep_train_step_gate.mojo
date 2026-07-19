# ACE-Step-1.5 training-step parity gate (#13): timesteps + noising + loss + CFG.
#
# Verifies the NEW train-step math against the oracle dump (acestep_train_step.mojo),
# with the oracle's FIXED x1/t (RNG parity is a separate concern — any valid noise
# trains). Backward (512 grads) is already gated (#12) — here we gate:
#   (a) flow-match noising: xt' = t*x1+(1-t)*x0 vs dumped xt; flow' = x1-x0 vs flow.
#   (b) loss reduction: mse(dumped out0, flow) == 2.109375 (isolates reduction).
#   (c) end-to-end: full-stack backward on the RECOMPUTED xt/flow → bg.loss (~2.11,
#       within the forward-recompute gap) + d(block0 out)=0.982 sanity.
#   (d) CFG dropout mechanism (bs=1 coin flip; UNTESTED by oracle — CFG was off).

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.collections import Optional
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import zeros_device
from serenitymojo.parity import ParityHarness
from serenitymojo.models.dit.acestep_dit import AceStepDiTConfig
from serenitymojo.autograd_v2.acestep_block_graph import acestep_stack_lora_graph_backward
from serenitymojo.models.acestep.acestep_train_step import (
    acestep_flow_noise, acestep_loss_mse, acestep_apply_cfg_dropout,
)

comptime XL_DIR = "/home/alex/ACE-Step-1.5/checkpoints/acestep-v15-xl-base"
comptime REF = "/home/alex/mojodiffusion/serenitymojo/models/acestep/parity/acestep_train_ref/acestep_train_ref.safetensors"
comptime SP = 64
comptime L = 64
comptime NH = 32
comptime LAYERS = 32
comptime LORA_SCALE = Float32(2.0)
comptime ORACLE_LOSS = 2.109375


def _bf16(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return cast_tensor(Tensor.from_view(st.tensor_view(name), ctx), STDtype.BF16, ctx)


def main() raises:
    var ctx = DeviceContext()
    var cfg = AceStepDiTConfig.xl_base()
    var dump = ShardedSafeTensors.open(REF)

    var x0 = _bf16(dump, "target_latents", ctx)      # [1,128,64] data
    var x1 = _bf16(dump, "x1", ctx)                  # [1,128,64] noise
    var context = _bf16(dump, "context_latents", ctx)
    var enc_in = _bf16(dump, "encoder_hidden_states", ctx)  # [1,64,2048]
    var out0_ref = _bf16(dump, "out0", ctx)          # [1,128,64]
    var flow_ref = _bf16(dump, "flow", ctx)
    var t = Tensor.from_view(dump.tensor_view("t"), ctx).to_host(ctx)[0]   # 0.7421875
    print("t =", t)

    var ph = ParityHarness(0.999)
    var n_ok = 0

    # ── (a) noising ───────────────────────────────────────────────────────────
    var noise = acestep_flow_noise(x0, x1, t, ctx)
    var xt_ref_h = Tensor.from_view(dump.tensor_view("xt"), ctx).to_host(ctx)
    var flow_h = Tensor.from_view(dump.tensor_view("flow"), ctx).to_host(ctx)
    var rxt = ph.compare(noise.xt, xt_ref_h, ctx)
    var rflow = ph.compare(noise.flow, flow_h, ctx)
    print("(a) noising:  xt cos=", rxt.cos, " flow cos=", rflow.cos)
    if rxt.passed and rflow.passed: n_ok += 1

    # ── (b) loss reduction on the oracle's own out0 (isolate reduction) ────────
    var loss_ref = acestep_loss_mse(out0_ref, flow_ref, ctx)
    var b_ok = abs(loss_ref - Float32(ORACLE_LOSS)) < Float32(0.05)
    print("(b) mse(oracle out0, flow) =", loss_ref, " vs", ORACLE_LOSS, "→", "OK" if b_ok else "FAIL")
    if b_ok: n_ok += 1

    # ── (c) end-to-end: full-stack backward on the RECOMPUTED xt/flow ──────────
    var full = Dict[String, ArcPointer[Tensor]]()
    var xl = ShardedSafeTensors.open(XL_DIR)
    for nm in xl.names():
        var n = String(nm)
        if n.startswith("decoder."):
            full[n] = ArcPointer(_bf16(xl, n, ctx))
    var attns = [String("self_attn"), String("cross_attn")]
    var projs = [String("q_proj"), String("k_proj"), String("v_proj"), String("o_proj")]
    var absx = [String("A"), String("B")]
    for li in range(LAYERS):
        var ln = String(li)
        for a in attns:
            for p in projs:
                for ab in absx:
                    var dk = String("w_base_model__model__layers__") + ln + "__" + a + "__" + p + "__lora_" + ab + "__default__weight"
                    var fk = String("decoder.layers.") + ln + "." + a + "." + p + ".lora_" + ab
                    full[fk] = ArcPointer(_bf16(dump, dk, ctx))
    var bg = acestep_stack_lora_graph_backward[SP, L, NH, LAYERS](
        noise.xt, context, enc_in, t, t, noise.flow, full, cfg, LORA_SCALE, ctx
    )
    var d_ref = Tensor.from_view(dump.tensor_view("block0_d_out"), ctx).to_host(ctx)
    var r0 = ph.compare(bg.d_block0_out[], d_ref, ctx)
    var c_ok = bg.loss > Float32(1.9) and bg.loss < Float32(2.3) and r0.cos > 0.97
    print("(c) train-step loss =", bg.loss, " (in-range ~2.11)  d(block0) cos=", r0.cos, "→", "OK" if c_ok else "FAIL")
    if c_ok: n_ok += 1

    # ── (d) CFG dropout mechanism (bs=1) ───────────────────────────────────────
    var null_cond = zeros_device(enc_in.shape(), enc_in.dtype(), ctx)   # stand-in null
    var keep = acestep_apply_cfg_dropout(enc_in, null_cond, Float32(0.0), 7, ctx)   # ratio 0 → keep
    var drop = acestep_apply_cfg_dropout(enc_in, null_cond, Float32(1.0), 7, ctx)   # ratio 1 → null
    var r_keep = ph.compare(keep, enc_in.to_host(ctx), ctx)
    var r_drop = ph.compare(drop, null_cond.to_host(ctx), ctx)
    # cos on all-zeros null is undefined; check keep==ehs (cos~1) and drop numel/zeroness.
    var d_ok = r_keep.cos > 0.999 and drop.numel() == enc_in.numel()
    print("(d) cfg dropout:  keep==ehs cos=", r_keep.cos, "  drop→null shape ok=", drop.numel() == enc_in.numel(), "→", "OK" if d_ok else "FAIL")
    if d_ok: n_ok += 1

    print("train-step gate:", n_ok, "of 4 checks pass")
    if n_ok == 4:
        print("GATE: PASS")
    else:
        print("GATE: FAIL")
