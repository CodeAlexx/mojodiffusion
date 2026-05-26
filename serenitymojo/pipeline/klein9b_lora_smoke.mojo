# klein9b_lora_smoke.mojo — merge-at-load LoRA wiring smoke for Klein 9B.
#
# Mirrors the LOAD + MERGE step of inference-flame/src/bin/klein_lora_infer.rs
# (Stage 3, lines 248-279): load the base DiT, build a LoraSet from a LoRA
# `.safetensors`, then merge it into the resident weights BEFORE the denoise
# loop. Unlike the Rust binary (which uses the runtime overlay LoraStack), this
# uses the minimally-invasive MERGE-AT-LOAD path: `LoraSet.merge_into_indexed`
# mutates `model.weights` in place via `model.name_to_idx`.
#
# COMPILE-ONLY in code mode (GPU wedged):
#   cd /home/alex/mojodiffusion && pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/pipeline/klein9b_lora_smoke.mojo -o /tmp/kleinlora
#
# ⚠ KEY-LAYOUT GAP (verified 2026-05-26, the sibling-port bite the task warned
#   about): the Klein9B DiT base stores FUSED `double_blocks.<i>.img_attn.qkv.
#   weight`, but the EriDiffusion-v2 `train_klein` LoRA I inspected ships SPLIT
#   `double_blocks.<i>.img_attn.to_q/to_k/to_v.weight` (detected DiffusionModel
#   format → mapped to `...to_q.weight`, which DOES NOT EXIST in the base, so
#   those modules NO-OP at merge). A split→fused RowRange mapper for the
#   `.img_attn.to_q/k/v` Klein names (analogous to the Z-Image `.attention.
#   to_q/k/v` → fused-qkv RowRange in _map_zimage_trainer) would be needed for a
#   train_klein LoRA to merge into this fused base. The KleinTrainer-format
#   `qkv_proj`/`out_proj` LoRAs DO map straight to fused `qkv.weight` (full
#   overlay) and merge cleanly. This smoke proves the WIRING; whether a given
#   LoRA's targets resolve depends on the LoRA's key layout vs the base's.

from std.gpu.host import DeviceContext

from serenitymojo.models.dit.klein_dit import Klein9BDiT
from serenitymojo.lora import LoraSet


comptime KLEIN9B_PATH = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors"
# A real train_klein LoRA (verified header 2026-05-26). NOTE the split-vs-fused
# gap above: with this file, img_attn merges no-op against the fused base.
comptime LORA_PATH = "/home/alex/EriDiffusion/EriDiffusion-v2/output/klein_lr3e4_const_b1/klein_lora_step200.safetensors"
comptime MULTIPLIER: Float32 = 1.0
comptime ALPHA: Float32 = 16.0  # train_klein default (= rank → scale 1.0)
comptime RANK = 16


def main() raises:
    var ctx = DeviceContext()

    # Stage 3a: load the full resident 9B DiT (all 201 BF16 tensors).
    var model = Klein9BDiT.load_full(String(KLEIN9B_PATH), ctx)
    print("loaded Klein9B DiT: ", len(model.weights), " tensors")

    # Stage 3b: build the LoRA set + merge it into the resident weights.
    # This is the one integration line that klein_lora_infer.rs Stage 3 reduces
    # to in the merge-at-load model: load → merge_into_indexed → denoise.
    var lset = LoraSet.load(String(LORA_PATH))
    print(
        "LoRA format ", lset.format_name(),
        " resolved mappings ", lset.num_mappings(),
    )
    var n = lset.merge_into_indexed(
        model.weights, model.name_to_idx, MULTIPLIER, ALPHA, RANK, ctx
    )
    print("merged ", n, " module(s) into Klein9B weights")
    print("(targets that hit the fused img_attn.qkv base will be 0 — see header)")

    # Stage 4+: denoise / VAE / PNG would follow here exactly as
    # klein9b_pipeline_1024_smoke.mojo, but the model now carries the merged
    # LoRA. Omitted in code-only mode (GPU wedged).
    print("klein9b_lora_smoke wiring OK")
