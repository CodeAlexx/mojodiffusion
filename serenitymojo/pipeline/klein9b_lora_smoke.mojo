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
# KEY-LAYOUT (verified 2026-05-26, fixed 2026-05-26): the Klein9B DiT base
#   stores FUSED `double_blocks.<i>.{img,txt}_attn.qkv.weight`, and the
#   EriDiffusion-v2 `train_klein` LoRA ships SPLIT
#   `double_blocks.<i>.{img,txt}_attn.to_q/to_k/to_v.lora_A.weight` (detected
#   DiffusionModel format). `LoraSet.load` now routes those split Q/K/V modules
#   into the fused `qkv.weight` row-ranges via `_map_klein_split_qkv` (offsets
#   0/out/2*out, len out; out = B-tensor shape[0], read from the file, NOT
#   hardcoded), mirroring the Z-Image branch in lora.rs:730-750 but keyed on
#   Klein's `.img_attn`/`.txt_attn` naming. KleinTrainer-format
#   `qkv_proj`/`out_proj` LoRAs still map straight to fused `qkv.weight` (full
#   overlay). All valid targets (proj/mlp/single + the 48 attention QKV modules)
#   now merge.

from std.gpu.host import DeviceContext

from serenitymojo.models.dit.klein_dit import Klein9BDiT
from serenitymojo.lora import LoraSet


comptime KLEIN9B_PATH = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors"
# A real train_klein LoRA (verified header 2026-05-26). NOTE the split-vs-fused
# gap above: with this file, img_attn merges no-op against the fused base.
comptime LORA_PATH = "/home/alex/EriDiffusion/EriDiffusion-v2/output/klein_lr3e4_const_b1/klein_lora_step200.safetensors"
comptime MULTIPLIER: Float32 = 1.0
# No file-level alpha/rank: scale is PER-MODULE (alpha/module_rank, alpha
# defaulting to module_rank when absent → scale = multiplier). See lora.mojo
# _module_scale / lora.rs:273-282.


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
        model.weights, model.name_to_idx, MULTIPLIER, ctx
    )
    print("merged ", n, " module(s) into Klein9B weights")
    print("(split to_q/k/v now route into the fused img_attn/txt_attn.qkv RowRange)")

    # Stage 4+: denoise / VAE / PNG would follow here exactly as
    # klein9b_pipeline_1024_smoke.mojo, but the model now carries the merged
    # LoRA. Omitted in code-only mode (GPU wedged).
    print("klein9b_lora_smoke wiring OK")
