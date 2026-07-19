# lens_forward_parity_smoke.mojo — the load-bearing Lens DiT forward parity gate.
#
# Loads the REAL Lens transformer weights + the Serenity oracle inputs, runs the
# ported LoRA-overlaid forward with LoRA B=0 (≡ base, identity overlay), and
# compares the predicted flow to the oracle output by COSINE similarity (bar
# cos >= 0.999). The oracle is Serenity's OWN dependency (lens/transformer.py)
# run on fixed inputs (parity/lens/lens_oracle.py) — reference policy: Serenity
# only.
#
# Oracle fixtures (parity/lens/, key "x", f32 on disk):
#   dit_fwd_in_hidden.safetensors   [1, 64, 128]    image latent (8x8 packed tokens)
#   dit_fwd_in_txt_{0..3}.safetensors [1, 16, 2880]  per selected-layer GPT-OSS feats
#   dit_fwd_in_mask.safetensors     [1, 16]         attention mask (1.0=valid)
#   dit_fwd_in_timestep.safetensors [1]             timestep (0.5; passed AS-IS)
#   dit_fwd_out.safetensors         [1, 64, 128]    proj_out reference
#
# DTYPE: BF16 storage in/out (the trained boundary); the oracle is f32, so a small
# BF16↔f32 gap is expected — the 0.999 cosine bar accounts for it.
#
# ── CROSS-SLICE CONTRACT (SLICE A: model/LensDiT.mojo + model/LensModel.mojo;
#    modelLoader/LensModelLoader.mojo [this slice]) ──────────────────────────────
#   LensWeights.load(LENS_TRANSFORMER_DIR)  → frozen transformer store (this slice)
#   model.LensModel: build_lens_lora_set(rank, alpha, ctx) → cold-start LoRA set
#       (A~randn, B=0 → identity overlay). 480 adapters (lensLoraTargets, attn-mlp).
#   model.LensDiT: lens_forward_full_infer[S_IMG,S_TXT](
#       hidden[1,S_img,128] BF16, txt0..3[1,S_txt,2880] BF16, mask[1,S_txt],
#       timestep f32, weights, loras, ctx) -> Tensor[1,S_img,128] BF16  (flow).

from std.gpu.host import DeviceContext
from std.math import sqrt, isfinite
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor

from serenity_trainer.modelLoader.LensModelLoader import LensWeights, LENS_TRANSFORMER_DIR
from serenity_trainer.model.LensModel import build_lens_lora_set
from serenity_trainer.model.LensDiT import lens_forward_full_infer


comptime PARITY_DIR = "trainer/parity/lens"
comptime S_IMG = 64    # meta.json s_img_tokens (8x8)
comptime S_TXT = 16    # meta.json s_txt
comptime COS_BAR = Float32(0.999)
comptime LORA_RANK = 8


def _load_x(name: String, ctx: DeviceContext) raises -> Tensor:
    var st = ShardedSafeTensors.open(String(PARITY_DIR) + String("/") + name)
    return Tensor.from_view(st.tensor_view(String("x")), ctx)


def _cosine(a: List[Float32], b: List[Float32]) raises -> Float32:
    if len(a) != len(b):
        raise Error(String("cosine: length mismatch ") + String(len(a))
                    + String(" vs ") + String(len(b)))
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na <= 0.0 or nb <= 0.0:
        raise Error("cosine: zero-norm vector")
    return Float32(dot / (na ** 0.5 * nb ** 0.5))


def main() raises:
    var ctx = DeviceContext()
    print("=== Lens DiT forward parity smoke (cos >=", COS_BAR, ") ===")

    # ── frozen transformer weights (real checkpoint) ──────────────────────────
    print("[weights] loading Lens transformer:", String(LENS_TRANSFORMER_DIR))
    var weights = LensWeights.load(String(LENS_TRANSFORMER_DIR), ctx)
    print("  loaded", weights.count(), "tensors")

    # ── oracle inputs (cast to BF16 for the forward boundary) ─────────────────
    var hidden = cast_tensor(_load_x(String("dit_fwd_in_hidden.safetensors"), ctx), STDtype.BF16, ctx)
    var txt0 = cast_tensor(_load_x(String("dit_fwd_in_txt_0.safetensors"), ctx), STDtype.BF16, ctx)
    var txt1 = cast_tensor(_load_x(String("dit_fwd_in_txt_1.safetensors"), ctx), STDtype.BF16, ctx)
    var txt2 = cast_tensor(_load_x(String("dit_fwd_in_txt_2.safetensors"), ctx), STDtype.BF16, ctx)
    var txt3 = cast_tensor(_load_x(String("dit_fwd_in_txt_3.safetensors"), ctx), STDtype.BF16, ctx)
    var mask = _load_x(String("dit_fwd_in_mask.safetensors"), ctx)            # [1,16]
    var ts_h = _load_x(String("dit_fwd_in_timestep.safetensors"), ctx).to_host(ctx)
    var timestep = ts_h[0]                                                     # 0.5, passed AS-IS

    var ref_out = _load_x(String("dit_fwd_out.safetensors"), ctx).to_host(ctx) # [1,64,128] f32

    # ── B=0 LoRA overlay (identity) → forward must equal the no-LoRA reference ─
    var loras = build_lens_lora_set(LORA_RANK, Float32(LORA_RANK), ctx)

    print("[forward] lens_forward_full_infer (B=0)")
    var out = lens_forward_full_infer[S_IMG, S_TXT](
        hidden, txt0, txt1, txt2, txt3, mask, timestep, weights, loras, ctx
    )
    var out_h = out.to_host(ctx)

    # finiteness
    for i in range(len(out_h)):
        if not isfinite(out_h[i]):
            raise Error(String("forward output non-finite at i=") + String(i))

    var cos = _cosine(out_h, ref_out)
    print("  cosine(forward, oracle) =", cos)
    if cos < COS_BAR:
        raise Error(String("FORWARD PARITY FAIL: cos=") + String(cos)
                    + String(" < ") + String(COS_BAR))
    print("  GATE OK: forward matches Serenity oracle (cos >=", COS_BAR, ")")
    print("=== smoke complete ===")
