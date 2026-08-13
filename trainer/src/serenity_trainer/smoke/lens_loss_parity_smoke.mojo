# lens_loss_parity_smoke.mojo — the REQUIRED Lens predict->loss parity gate.
#
# Gates the ported Lens trainer's predict->loss against Serenity's OWN reference
# loss on BYTE-IDENTICAL inputs. The reference (parity/lens/lens_loss_oracle.py)
# replicates BaseLensSetup.predict EXACTLY on a real image+caption with the
# deterministic path (timestep=499, sigma=0.5) and dumps:
#   loss_ref.safetensors keys:
#     packed_in [1,1024,128]  scaled+noised latent fed to the transformer
#     target    [1,128,32,32] flow target (noise - scaled), patchified-scaled space
#     feat_0..3 [1,201,2880]  the 4 GPT-OSS layer features (cropped, S_txt=201)
#   loss_ref_meta.json: {"loss":0.508090, "timestep":499, "sigma":0.5,
#                        "H_packed":32, "W_packed":32, "S_txt":201}
#
# This smoke:
#   1) loads the REAL Lens transformer weights (LensWeights.load),
#   2) loads packed_in, feat_0..3, target,
#   3) builds an all-valid mask (S_txt=201) and runs lens_forward_full_infer
#      [S_IMG=1024, S_TXT=201] with LoRA B=0 (identity overlay) at timestep
#      = 499/1000 = 0.5, img_shapes (1,32,32) -> predicted flow [1,1024,128],
#   4) unpacks predicted [1,1024,128] -> [1,128,32,32] (pure permutation; MSE
#      invariant) and computes loss = mean((pred - target)^2),
#   5) GATE: |mojo_loss - 0.508090| / 0.508090 <= 0.02 (bf16-forward vs the
#      bf16-forward f32-loss reference).
#
# DTYPE: BF16 forward boundary (trained storage); the loss is reduced in F64 on
# host. The 2% bar accounts for the bf16<->f32 gap.

from max.gpu.host import DeviceContext
from std.math import isfinite
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor

from serenity_trainer.modelLoader.LensModelLoader import LensWeights, LENS_TRANSFORMER_DIR
from serenity_trainer.model.LensDiT import lens_forward_full_infer, build_lens_lora_set
from serenity_trainer.modelSetup.BaseLensSetup import unpack_latents


comptime PARITY_DIR = "trainer/parity/lens"
comptime LOSS_REF = PARITY_DIR + "/loss_ref.safetensors"
comptime S_IMG = 1024      # H_packed * W_packed = 32 * 32
comptime S_TXT = 201       # meta.json S_txt
comptime H_PACKED = 32
comptime W_PACKED = 32
comptime LORA_RANK = 8
comptime SERENITY_LOSS = Float64(0.508089542388916)
comptime REL_BAR = Float64(0.02)


def main() raises:
    var ctx = DeviceContext()
    print("=== Lens predict->loss parity smoke (rel <=", Float32(REL_BAR), ") ===")

    # ── frozen transformer weights (real checkpoint) ──────────────────────────
    print("[weights] loading Lens transformer:", String(LENS_TRANSFORMER_DIR))
    var weights = LensWeights.load(String(LENS_TRANSFORMER_DIR), ctx)
    print("  loaded", weights.count(), "tensors")

    # ── oracle byte-identical inputs (one safetensors, multiple keys) ──────────
    var st = ShardedSafeTensors.open(String(LOSS_REF))
    var hidden = cast_tensor(Tensor.from_view(st.tensor_view(String("packed_in")), ctx), STDtype.BF16, ctx)  # [1,1024,128]
    var txt0 = cast_tensor(Tensor.from_view(st.tensor_view(String("feat_0")), ctx), STDtype.BF16, ctx)       # [1,201,2880]
    var txt1 = cast_tensor(Tensor.from_view(st.tensor_view(String("feat_1")), ctx), STDtype.BF16, ctx)
    var txt2 = cast_tensor(Tensor.from_view(st.tensor_view(String("feat_2")), ctx), STDtype.BF16, ctx)
    var txt3 = cast_tensor(Tensor.from_view(st.tensor_view(String("feat_3")), ctx), STDtype.BF16, ctx)
    var target_h = Tensor.from_view(st.tensor_view(String("target")), ctx).to_host(ctx)                      # [1,128,32,32] f32

    # ── all-valid attention mask [1, S_TXT] (every text token kept) ────────────
    var mask_vals = List[Float32]()
    for _ in range(S_TXT):
        mask_vals.append(Float32(1.0))
    var mask_sh = List[Int]()
    mask_sh.append(1); mask_sh.append(S_TXT)
    var mask = Tensor.from_host(mask_vals^, mask_sh^, STDtype.F32, ctx)

    # ── deterministic timestep: 499/1000 = 0.5 (passed AS-IS, matches oracle) ──
    var timestep = Float32(499.0) / Float32(1000.0)

    # ── B=0 LoRA overlay (identity) → forward equals the no-LoRA reference ─────
    var loras = build_lens_lora_set(LORA_RANK, Float32(LORA_RANK), ctx)

    print("[forward] lens_forward_full_infer (B=0, t=", timestep, ")")
    var out = lens_forward_full_infer[S_IMG, S_TXT](
        hidden, txt0, txt1, txt2, txt3, mask, timestep, weights, loras, ctx
    )

    # ── unpack predicted [1,1024,128] -> [1,128,32,32] (pure permutation) ──────
    var pred_unpacked = unpack_latents(out, H_PACKED, W_PACKED, ctx)
    var pred_h = pred_unpacked.to_host(ctx)

    if len(pred_h) != len(target_h):
        raise Error(String("shape mismatch: pred numel=") + String(len(pred_h))
                    + String(" target numel=") + String(len(target_h)))

    # ── MSE loss = mean((pred - target)^2), reduced in F64 ────────────────────
    var sse = Float64(0.0)
    for i in range(len(pred_h)):
        var p = pred_h[i]
        if not isfinite(p):
            raise Error(String("predicted output non-finite at i=") + String(i))
        var d = Float64(p) - Float64(target_h[i])
        sse += d * d
    var mojo_loss = sse / Float64(len(pred_h))

    var abs_diff = abs(mojo_loss - SERENITY_LOSS)
    var rel_diff = abs_diff / SERENITY_LOSS

    print("")
    print("  mojo_loss =", mojo_loss)
    print("  serenity_loss   =", SERENITY_LOSS)
    print("  abs diff  =", abs_diff)
    print("  rel diff  =", rel_diff, " (bar", REL_BAR, ")")
    if rel_diff <= REL_BAR:
        print("  PASS: Lens predict->loss matches Serenity (rel <=", REL_BAR, ")")
    else:
        print("  FAIL: Lens predict->loss diverges from Serenity")
        raise Error(String("LOSS PARITY FAIL: mojo_loss=") + String(mojo_loss)
                    + String(" serenity_loss=") + String(SERENITY_LOSS)
                    + String(" rel=") + String(rel_diff)
                    + String(" > ") + String(REL_BAR))
    print("=== smoke complete ===")
