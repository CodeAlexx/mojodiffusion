# lens_train_gates_smoke.mojo — ONE smoke covering three Lens LoRA training gates.
#
# GATE A — AdamW parity vs torch (Serenity uses torch AdamW). Runs the Lens
#   host-list AdamW (module/LensLoRAModule._lora_adamw, the SAME update
#   lens_lora_adamw_step drives) on the 8-float reference (parity/adamw_ref.json)
#   with constant grad=0.05, 2 steps, and compares to the deterministic torch
#   reference (max |Δ| <= 2e-3, BF16 moment-storage tolerance). Determinism: the
#   parity run uses stochastic_rounding=False (RNE), the only difference from the
#   training path being which final BF16 rounding mode is used.
#
# GATE B — real-data multi-step train gate. Loads parity/lens/loss_ref.safetensors
#   (packed_in [1,1024,128], target [1,128,32,32], feat_0..3 [1,201,2880]) + the
#   real Lens transformer weights, builds a LensLoraSet (rank 16, alpha 16, B=0),
#   and runs 4 steps of:
#     lens_forward_full_lora[32,32,201](packed_in, t=0.499, cap_feats) → velocity
#       → unpack → loss = mean((unpack(pred)-target)^2) → d_velocity = pack(d_pred)
#       → lens_backward_full_lora → lens_lora_adamw_step(lr 1e-4).
#   ASSERTS each step: loss finite, loss in [0.30, 0.75], 0 nonfinite in all B.
#   After step >=2, all 476 nonzero-grad adapters have nonzero B (the 4 last-block
#   txt-post adapters are architecturally zero-grad EVERY step — the output head
#   reads only the img stream, so the last block's d_txt_out=0).
#
# GATE C — save/resume. Saves the trained LoRA via the REAL save_lens_lora, reloads
#   into a fresh LensLoraSet via the REAL load_lens_lora, asserts loaded A/B ==
#   saved A/B (max |Δ| < 1e-4, BF16-exact), then runs ONE more train step on the
#   reloaded set (loss finite + in range). NOTE: PEFT/torchref LoRA files persist
#   ONLY the adapter weights (lora_down/lora_up/alpha) — NOT the AdamW moments — so
#   this gate is weight-resume correctness, not optimizer-state resume.
#
# DTYPE: BF16 storage boundary; F32/F64 only in host reductions. No persistent F32.

from max.gpu.host import DeviceContext
from std.math import isfinite

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import concat as _concat

from serenity_trainer.modelLoader.LensModelLoader import (
    LensWeights, LENS_TRANSFORMER_DIR, load_lens_lora, LensLoraReload,
)
from serenity_trainer.module.LensLoRAModule import LoraAdapter, LoraGrads, _lora_adamw
from serenity_trainer.model.lens.lens_stack_lora import lens_forward_full_lora, LensForwardOut
from serenity_trainer.model.lens.lens_backward import (
    LensLoraSet, build_lens_lora_set, lens_backward_full_lora,
    LensStackLoraGrads, lens_lora_adamw_step,
)
from serenity_trainer.modelSaver.lens.LensLoRASaver import save_lens_lora
from serenity_trainer.modelSetup.BaseLensSetup import unpack_latents, pack_latents
from serenity_trainer.modelSetup.lensLoraTargets import (
    lens_lora_target_prefixes,
    LORA_TO_ADD_OUT, LORA_TXT_MLP_W1, LORA_TXT_MLP_W2, LORA_TXT_MLP_W3,
    LORA_SLOTS_PER_BLOCK, LENS_N_BLOCKS,
)


comptime PARITY_DIR  = "trainer/parity"
comptime LOSS_REF    = "trainer/parity/lens/loss_ref.safetensors"
comptime RESUME_PATH = "/tmp/lens_resume.safetensors"
comptime HLp = 32           # H_packed (loss_ref geometry)
comptime WLp = 32           # W_packed
comptime CAPLEN = 201       # S_txt
comptime LOSS_LO = Float32(0.30)
comptime LOSS_HI = Float32(0.75)
comptime LORA_RANK = 16
comptime LORA_ALPHA = Float32(16.0)


# ── host helpers ──────────────────────────────────────────────────────────────
def _zeros(n: Int) -> List[Float32]:
    var z = List[Float32]()
    for _ in range(n):
        z.append(Float32(0.0))
    return z^


def _abs_sum_finite(b: List[BFloat16], mut nonfinite: Int) -> Float32:
    var s = Float32(0.0)
    for i in range(len(b)):
        var v = b[i].cast[DType.float32]()
        if not isfinite(v):
            nonfinite += 1
        else:
            s += abs(v)
    return s


# Build a fresh LensLoraSet from a REAL reload bundle (GATE C resume). Each adapter
# is reconstructed from the loaded A/B device tensors (BF16-exact round-trip) with
# zeroed AdamW moments (PEFT files do not persist moments).
def _set_from_reload(reload: LensLoraReload, ctx: DeviceContext) raises -> LensLoraSet:
    var block = List[LoraAdapter]()
    for i in range(len(reload.a)):
        var a_h = reload.a[i][].to_host(ctx)        # F32 [rank*in]
        var b_h = reload.b[i][].to_host(ctx)        # F32 [out*rank]
        var ash = reload.a[i][].shape()
        var bsh = reload.b[i][].shape()
        var rank = ash[0]
        var in_f = ash[1]
        var out_f = bsh[0]
        var scale = reload.alpha[i] / Float32(rank)
        block.append(
            LoraAdapter(
                a_h^, b_h^, rank, in_f, out_f, scale,
                _zeros(rank * in_f), _zeros(rank * in_f),
                _zeros(out_f * rank), _zeros(out_f * rank),
            )
        )
    return LensLoraSet(block^, reload.rank)


# ── ONE forward→loss→backward→AdamW step over `loras`. Returns the step loss. ──
def _train_step(
    mut loras: LensLoraSet,
    packed_in: Tensor, cap_feats: Tensor, target_h: List[Float32],
    weights: LensWeights, t_opt: Int, lr: Float32, ctx: DeviceContext,
) raises -> Float32:
    var fo = lens_forward_full_lora[HLp, WLp, CAPLEN](
        packed_in, Float32(0.499), cap_feats, weights, loras, ctx
    )
    # velocity [1,1024,128] → unpack → [1,128,32,32]
    var pred = unpack_latents(fo.velocity, HLp, WLp, ctx)
    var pred_h = pred.to_host(ctx)
    var numel = len(pred_h)
    if numel != len(target_h):
        raise Error(String("pred/target numel mismatch: ") + String(numel)
                    + String(" vs ") + String(len(target_h)))
    # loss = mean((pred-target)^2) (F64); d_pred = 2(pred-target)/numel (mean MSE).
    var sse = Float64(0.0)
    var dlist = List[Float32]()
    for i in range(numel):
        var d = Float64(pred_h[i]) - Float64(target_h[i])
        sse += d * d
        dlist.append(Float32(2.0 * d / Float64(numel)))
    var loss = Float32(sse / Float64(numel))
    # pull the loss grad back through unpack (pure permutation): d_vel = pack(d_pred).
    var dsh = List[Int](); dsh.append(1); dsh.append(128); dsh.append(HLp); dsh.append(WLp)
    var d_pred = Tensor.from_host(dlist^, dsh^, STDtype.BF16, ctx)   # [1,128,32,32]
    var d_velocity = pack_latents(d_pred, ctx)                       # [1,1024,128]
    var grads = lens_backward_full_lora[HLp, WLp, CAPLEN](d_velocity, fo.saved, loras, ctx)
    lens_lora_adamw_step(loras, grads, t_opt, lr, ctx)
    return loss


# Count adapters whose B is now nonzero; the 4 last-block txt-post adapters
# (to_add_out, txt_mlp.w1/w2/w3) are architecturally zero-grad and allowed zero.
# Returns (n_movers, n_nonfinite, bad_zero).
def _audit_b(loras: LensLoraSet) -> Tuple[Int, Int, Int]:
    var n_ad = len(loras.block)
    var last_base = (LENS_N_BLOCKS - 1) * LORA_SLOTS_PER_BLOCK
    var movers = 0
    var nonfinite = 0
    var bad_zero = 0
    for i in range(n_ad):
        var s = _abs_sum_finite(loras.block[i].b, nonfinite)
        if s > Float32(0.0):
            movers += 1
        else:
            var allowed = (
                i == last_base + LORA_TO_ADD_OUT
                or i == last_base + LORA_TXT_MLP_W1
                or i == last_base + LORA_TXT_MLP_W2
                or i == last_base + LORA_TXT_MLP_W3
            )
            if not allowed:
                bad_zero += 1
    return (movers, nonfinite, bad_zero)


def main() raises:
    var ctx = DeviceContext()
    print("================ Lens train-gates smoke (A: AdamW, B: train, C: save/resume) ================")

    var gate_a = False
    var gate_b = False
    var gate_c = False

    # ══════════════════════════════════════════════════════════════════════════
    # GATE A — AdamW parity vs torch (deterministic)
    # ══════════════════════════════════════════════════════════════════════════
    print("")
    print("── GATE A: AdamW parity vs torch (parity/adamw_ref.json) ──")
    var p0 = List[Float32]()
    p0.append(Float32(-0.400390625))
    p0.append(Float32(-0.30078125))
    p0.append(Float32(-0.2001953125))
    p0.append(Float32(-0.10009765625))
    p0.append(Float32(0.0))
    p0.append(Float32(0.10009765625))
    p0.append(Float32(0.2001953125))
    p0.append(Float32(0.30078125))
    var p_ref = List[Float32]()
    p_ref.append(Float32(-0.404296875))
    p_ref.append(Float32(-0.3046875))
    p_ref.append(Float32(-0.2021484375))
    p_ref.append(Float32(-0.10205078125))
    p_ref.append(Float32(-0.0019989013671875))
    p_ref.append(Float32(0.09814453125))
    p_ref.append(Float32(0.1982421875))
    p_ref.append(Float32(0.296875))

    var N = 8
    # LoraAdapter stores params in `a` ([rank,in]); `b` empty (out_f=0). Moments zero.
    var lo = LoraAdapter(
        p0.copy(), List[Float32](), 1, N, 0, Float32(1.0),
        _zeros(N), _zeros(N), List[Float32](), List[Float32](),
    )
    var grad = List[Float32]()
    for _ in range(N):
        grad.append(Float32(0.05))
    var g = LoraGrads(grad.copy(), List[Float32]())
    # 2 steps; RNE (deterministic) to compare to the deterministic torch reference.
    _lora_adamw(lo, g, 1, Float32(0.001), ctx,
                Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01), False)
    _lora_adamw(lo, g, 2, Float32(0.001), ctx,
                Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01), False)

    var max_a = Float32(0.0)
    print("  i :     mojo            torch           |Δ|")
    for i in range(N):
        var pm = lo.a[i].cast[DType.float32]()
        var d = abs(pm - p_ref[i])
        if d > max_a:
            max_a = d
        print("  ", i, ":", pm, "  ", p_ref[i], "  ", d)
    print("  max |Δ| =", max_a, " (bar 2e-3)")
    gate_a = max_a <= Float32(2.0e-3)
    print("  GATE A:", "PASS" if gate_a else "FAIL")

    # ══════════════════════════════════════════════════════════════════════════
    # GATE B — real-data multi-step train gate
    # ══════════════════════════════════════════════════════════════════════════
    print("")
    print("── GATE B: real-data 4-step train gate (loss_ref) ──")
    print("  [weights] loading Lens transformer:", String(LENS_TRANSFORMER_DIR))
    var weights = LensWeights.load(String(LENS_TRANSFORMER_DIR), ctx)
    print("    loaded", weights.count(), "tensors")

    var st = ShardedSafeTensors.open(String(LOSS_REF))
    var packed_in = cast_tensor(Tensor.from_view(st.tensor_view(String("packed_in")), ctx), STDtype.BF16, ctx)  # [1,1024,128]
    var feat0 = cast_tensor(Tensor.from_view(st.tensor_view(String("feat_0")), ctx), STDtype.BF16, ctx)         # [1,201,2880]
    var feat1 = cast_tensor(Tensor.from_view(st.tensor_view(String("feat_1")), ctx), STDtype.BF16, ctx)
    var feat2 = cast_tensor(Tensor.from_view(st.tensor_view(String("feat_2")), ctx), STDtype.BF16, ctx)
    var feat3 = cast_tensor(Tensor.from_view(st.tensor_view(String("feat_3")), ctx), STDtype.BF16, ctx)
    var cap_feats = _concat(2, ctx, feat0, feat1, feat2, feat3)     # [1,201,11520]
    var target_h = Tensor.from_view(st.tensor_view(String("target")), ctx).to_host(ctx)   # [1,128,32,32] F32

    var loras = build_lens_lora_set(LORA_RANK, LORA_ALPHA, UInt64(0), ctx)
    print("    adapters:", len(loras.block), " rank:", loras.rank, " (B=0 at init)")

    var b_loss_ok = True
    var b_finite_ok = True
    var b_bupdate_ok = True
    for s in range(4):
        var loss = _train_step(
            loras, packed_in, cap_feats, target_h, weights, s + 1, Float32(1.0e-4), ctx
        )
        var aud = _audit_b(loras)
        var finite_loss = isfinite(loss)
        var in_range = (loss >= LOSS_LO) and (loss <= LOSS_HI)
        print("  step", s + 1, ": loss =", loss,
              " in_range =", in_range,
              " movers =", aud[0], "/", len(loras.block),
              " nonfinite_B =", aud[1], " bad_zero =", aud[2])
        if not finite_loss or not in_range:
            b_loss_ok = False
        if aud[1] != 0:
            b_finite_ok = False
        # after step >=2 every nonzero-grad adapter (476 = 480-4) must have moved.
        if s + 1 >= 2:
            if aud[2] != 0 or aud[0] < len(loras.block) - 4:
                b_bupdate_ok = False
    gate_b = b_loss_ok and b_finite_ok and b_bupdate_ok
    print("  loss_ok =", b_loss_ok, " finite_ok =", b_finite_ok, " B_update_ok =", b_bupdate_ok)
    print("  GATE B:", "PASS" if gate_b else "FAIL")

    # ══════════════════════════════════════════════════════════════════════════
    # GATE C — save/resume (REAL saver + REAL loader)
    # ══════════════════════════════════════════════════════════════════════════
    print("")
    print("── GATE C: save/resume (real save_lens_lora → load_lens_lora) ──")
    save_lens_lora(loras, String(RESUME_PATH), ctx)
    print("  saved →", String(RESUME_PATH))

    var prefixes = lens_lora_target_prefixes()
    var reload = load_lens_lora(String(RESUME_PATH), prefixes, ctx)
    print("  reloaded adapters =", len(reload.a), " rank =", reload.rank)
    var resumed = _set_from_reload(reload, ctx)

    # loaded A/B == saved A/B (BF16-exact).
    var max_rt = Float32(0.0)
    for i in range(len(loras.block)):
        ref sa = loras.block[i].a
        ref ra = resumed.block[i].a
        ref sb = loras.block[i].b
        ref rb = resumed.block[i].b
        if len(sa) != len(ra) or len(sb) != len(rb):
            raise Error(String("resume length mismatch at adapter ") + String(i))
        for j in range(len(sa)):
            var d = abs(sa[j].cast[DType.float32]() - ra[j].cast[DType.float32]())
            if d > max_rt:
                max_rt = d
        for j in range(len(sb)):
            var d = abs(sb[j].cast[DType.float32]() - rb[j].cast[DType.float32]())
            if d > max_rt:
                max_rt = d
    print("  max reload |Δ| =", max_rt, " (bar 1e-4, BF16-exact)")
    var reload_exact = max_rt < Float32(1.0e-4)

    # one more train step on the RELOADED set.
    var rloss = _train_step(
        resumed, packed_in, cap_feats, target_h, weights, 5, Float32(1.0e-4), ctx
    )
    var rfinite = isfinite(rloss)
    var rin = (rloss >= LOSS_LO) and (rloss <= LOSS_HI)
    print("  post-resume step: loss =", rloss, " finite =", rfinite, " in_range =", rin)
    print("  NOTE: AdamW moments are NOT persisted by the PEFT LoRA format — gate is")
    print("        weight-resume correctness (moments reset to 0 on reload).")
    gate_c = reload_exact and rfinite and rin
    print("  reload_exact =", reload_exact, " post_resume_ok =", (rfinite and rin))
    print("  GATE C:", "PASS" if gate_c else "FAIL")

    # ══════════════════════════════════════════════════════════════════════════
    print("")
    print("================ SUMMARY ================")
    print("  GATE A (AdamW parity)   :", "PASS" if gate_a else "FAIL")
    print("  GATE B (train gate)     :", "PASS" if gate_b else "FAIL")
    print("  GATE C (save/resume)    :", "PASS" if gate_c else "FAIL")
    var ok = gate_a and gate_b and gate_c
    print("  OVERALL GATE:", "OK" if ok else "FAIL")
    if not ok:
        raise Error("lens_train_gates_smoke: one or more gates FAILED (see above)")
    print("=== smoke complete ===")
