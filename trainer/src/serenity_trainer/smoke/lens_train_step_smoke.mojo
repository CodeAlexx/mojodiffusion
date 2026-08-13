# lens_train_step_smoke.mojo — one full Lens LoRA training step gate (HAND-CHAINED).
#
# Runs ONE predict → loss → backward_lora → AdamW step on a real-weights
# LensLoRASpec and asserts the LoRA-B matrices go 0 → nonzero on EVERY adapter (and
# stay finite). This proves the train seam end-to-end: the BaseLensSetup.predict
# math (scale/noise/sigma/flow target) + the LoRA-overlaid hand-chained forward
# (model/lens/lens_stack_lora.lens_forward_full_lora) + the hand-chained backward
# (model/lens/lens_backward.lens_backward_full_lora) + the host-list AdamW update
# (lens_lora_adamw_step) all compose and push gradient into every adapter.
#
# DESIGN: HAND-CHAINED, host-list LoRA adapters (LensLoraSet.block[i].a/.b), exactly
# like the Klein LoRA trainer — NOT the shared autograd Tape / ParamSlot path. The
# B factors start at 0 (PEFT identity init), so any nonzero B after one step ⇒ a
# real, finite gradient reached that adapter.
#
# GEOMETRY (oracle, meta.json): raw latent [1,32,16,16] → patchify → [1,128,8,8] →
# pack → 64 image tokens. So the spec is comptime-shaped on the PATCHIFIED grid
# HLp=WLp=8 (N_IMG=64) and CAPLEN=16 caption tokens. cap_feats is the raw 4-layer
# GPT-OSS feature concat [1,16,11520] (the per-layer RMSNorm is applied inside the
# forward).

from max.gpu.host import DeviceContext
from std.math import isfinite
from std.builtin.dtype import DType
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import sub as _sub, mul_scalar as _mul_scalar, concat as _concat
from serenitymojo.autograd import Tape

from serenity_trainer.modelLoader.LensModelLoader import LensWeights, LENS_TRANSFORMER_DIR
from serenity_trainer.modelSetup.LensLoRASetup import make_lens_lora_spec
from serenity_trainer.model.lens.lens_backward import lens_lora_adamw_step
from serenity_trainer.modelSetup.BaseLensSetup import patchify_latents, pack_latents
from serenity_trainer.modelSetup.lensLoraTargets import (
    LORA_TO_ADD_OUT, LORA_TXT_MLP_W1, LORA_TXT_MLP_W2, LORA_TXT_MLP_W3,
    LORA_SLOTS_PER_BLOCK, LENS_N_BLOCKS,
)
from serenity_trainer.util.config.TrainConfig import TrainConfig


comptime PARITY_DIR = "trainer/parity/lens"
comptime VAE_BN_PATH = "trainer/parity/lens/vae_bn.safetensors"
# raw latent grid 16x16 (32ch) → patchify → 128ch,8x8 → pack → 64 image tokens.
# HLp/WLp are the PATCHIFIED dims (H//2, W//2) the spec is comptime-shaped on.
comptime HLp = 8
comptime WLp = 8
comptime CAPLEN = 16
comptime LORA_RANK = 4


def _load_x(name: String, ctx: DeviceContext) raises -> Tensor:
    var st = ShardedSafeTensors.open(String(PARITY_DIR) + String("/") + name)
    return Tensor.from_view(st.tensor_view(String("x")), ctx)


def _abs_sum_finite(b: List[BFloat16], mut nonfinite: Int) raises -> Float32:
    # |B|_1 over a host BF16 factor; increments `nonfinite` per non-finite element
    # (NEVER raises — the gate prints the count rather than aborting on the first).
    var s = Float32(0.0)
    for i in range(len(b)):
        var v = b[i].cast[DType.float32]()
        if not isfinite(v):
            nonfinite += 1
        else:
            s += abs(v)
    return s


def main() raises:
    var ctx = DeviceContext()
    print("=== Lens train-step smoke (hand-chained; LoRA-B 0 -> nonzero, finite) ===")

    # ── frozen transformer weights (real checkpoint) ──────────────────────────
    var weights = LensWeights.load(String(LENS_TRANSFORMER_DIR), ctx)
    print("[weights]", weights.count(), "tensors")

    # ── synthetic raw latent [1,32,16,16] BF16 + oracle text conditioning ──────
    var lat_sh = List[Int]()
    lat_sh.append(1); lat_sh.append(32); lat_sh.append(2 * HLp); lat_sh.append(2 * WLp)
    var latent = randn(lat_sh^, UInt64(1234), STDtype.BF16, ctx)

    var txt0 = cast_tensor(_load_x(String("dit_fwd_in_txt_0.safetensors"), ctx), STDtype.BF16, ctx)
    var txt1 = cast_tensor(_load_x(String("dit_fwd_in_txt_1.safetensors"), ctx), STDtype.BF16, ctx)
    var txt2 = cast_tensor(_load_x(String("dit_fwd_in_txt_2.safetensors"), ctx), STDtype.BF16, ctx)
    var txt3 = cast_tensor(_load_x(String("dit_fwd_in_txt_3.safetensors"), ctx), STDtype.BF16, ctx)
    # cap_feats = raw 4-layer concat [1,CAPLEN,11520] (per-layer RMSNorm done inside fwd)
    var cap_feats = _concat(2, ctx, txt0, txt1, txt2, txt3)

    # ── VAE batch-norm latent-scale stats (LensModel.scale_latents) ────────────
    var bn = ShardedSafeTensors.open(String(VAE_BN_PATH))
    var vae_bn_mean = Tensor.from_view(bn.tensor_view(String("running_mean")), ctx)
    var vae_bn_var  = Tensor.from_view(bn.tensor_view(String("running_var")), ctx)

    # ── cold-start spec (A~kaiming-ish, B=0 on every adapter) ──────────────────
    var spec = make_lens_lora_spec[HLp, WLp, CAPLEN](
        weights^, latent^, cap_feats^, vae_bn_mean^, vae_bn_var^,
        LORA_RANK, Float32(LORA_RANK), Float32(1.0), UInt64(0), ctx,
    )

    # ── snapshot: every B must be exactly 0 before the step (PEFT identity init) ─
    var n_ad = len(spec.loras.block)
    print("[init] adapters:", n_ad)
    var pre_nonfinite = 0
    for i in range(n_ad):
        var b0 = _abs_sum_finite(spec.loras.block[i].b, pre_nonfinite)
        if b0 != Float32(0.0):
            raise Error(String("adapter ") + String(i)
                        + String(" B not zero at init: ") + String(b0))
    print("  all", n_ad, "adapters B == 0 at init (identity overlay)")

    # ── ONE step: predict → MSE loss → backward_lora → host-list AdamW ─────────
    var cfg = TrainConfig.adamw_lora_defaults()
    var tape = Tape()
    var out = spec.predict(tape, cfg, 0, ctx)

    # MSE loss + d_predicted (mean MSE: loss = mean((p-t)^2); d = 2(p-t)/numel).
    var diff = _sub(out.predicted, out.target, ctx)
    var diff_h = diff.to_host(ctx)
    var numel = len(diff_h)
    var sse = Float64(0.0)
    for i in range(numel):
        sse += Float64(diff_h[i]) * Float64(diff_h[i])
    var loss = Float32(sse / Float64(numel))
    if not isfinite(loss):
        raise Error(String("non-finite loss: ") + String(loss))

    var d_pred = _mul_scalar(diff, Float32(2.0) / Float32(numel), ctx)  # [1,32,H,W]

    # Pull the loss grad back through unpatchify∘unpack (both pure permutations) to
    # the packed-velocity grad: d_velocity = pack(patchify(d_predicted)).
    var d_pred_flow = patchify_latents(d_pred, ctx)        # [1,128,HLp,WLp]
    var d_velocity = pack_latents(d_pred_flow, ctx)        # [1,N_IMG,128]

    var grads = spec.backward_lora(d_velocity, ctx)        # 480 LoRA d_a/d_b
    # AdamW at optimizer step t=1 over the host-list adapters (in place).
    lens_lora_adamw_step(spec.loras, grads, 1, Float32(1e-3), ctx)
    print("[step] loss =", loss)

    # ── GATE: B is now nonzero + finite on every adapter the architecture can
    #    deliver gradient to on ONE step. Count movers + nonfinite.
    #
    # ARCHITECTURAL ZERO-GRAD (NOT a bug, do not "fix"): the Lens output head reads
    # ONLY the image stream (proj_out(norm_out(img_final))), so the LAST block's txt
    # residual output is unused → lens_backward sets the last block's d_txt_out=0,
    # and the post-attention txt adapters of the last block (to_add_out, txt_mlp.
    # w1/w2/w3 — 4 adapters) receive exactly zero gradient on a single backward. The
    # last block's txt_qkv STILL moves (joint attention couples img queries to txt
    # keys/values), and all txt adapters of blocks 0..N-2 move (their output feeds
    # the next block). So the honest single-step invariant is: ALL adapters move
    # EXCEPT exactly those 4. FIXME-NUMERIC: torch autograd gives the same 4 a zero
    # grad on one step (by construction) — confirm against the Serenity oracle.
    var n_nonzero = 0
    var n_nonfinite = 0
    var bad_zero = 0
    var last_base = (LENS_N_BLOCKS - 1) * LORA_SLOTS_PER_BLOCK
    for i in range(n_ad):
        var b1 = _abs_sum_finite(spec.loras.block[i].b, n_nonfinite)
        if b1 > Float32(0.0):
            n_nonzero += 1
        else:
            var allowed = (
                i == last_base + LORA_TO_ADD_OUT
                or i == last_base + LORA_TXT_MLP_W1
                or i == last_base + LORA_TXT_MLP_W2
                or i == last_base + LORA_TXT_MLP_W3
            )
            if not allowed:
                bad_zero += 1
                print("  UNEXPECTED zero-B adapter after step: index", i)
    print("  movers:", n_nonzero, "/", n_ad,
          " (expected non-movers: 4 last-block txt-post adapters)")
    print("  nonfinite values in all B after step:", n_nonfinite)
    if n_nonfinite != 0:
        raise Error(String("non-finite LoRA-B after step: ") + String(n_nonfinite))
    if bad_zero != 0:
        raise Error(String("unexpected zero-B adapters after step: ") + String(bad_zero)
                    + String(" (only the 4 last-block txt-post adapters may be zero)"))
    if n_nonzero < n_ad - 4:
        raise Error(String("too few adapters moved: ") + String(n_nonzero)
                    + String(" / ") + String(n_ad))
    print("  GATE OK:", n_nonzero, "/", n_ad,
          "adapters moved 0 -> nonzero, all finite (loss=", loss, ")")
    print("=== smoke complete ===")
