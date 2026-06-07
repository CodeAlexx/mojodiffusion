#!/usr/bin/env python3
"""Static guard for OneTrainer text-conditioning contract metadata.

This checks the Mojo contract file against source markers in OneTrainer and
OneTrainer-anima-ref. It is not an end-to-end tokenizer/encoder parity test.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
OT = Path("/home/alex/OneTrainer")
ANIMA_REF = Path("/home/alex/OneTrainer-anima-ref")
CONTRACT = ROOT / "serenitymojo/models/text_encoder/onetrainer_conditioning_contract.mojo"
SMOKE = ROOT / "serenitymojo/models/text_encoder/onetrainer_conditioning_contract_smoke.mojo"


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"ERROR missing file: {path}")
        raise


def require(label: str, haystack: str, needle: str) -> None:
    if needle not in haystack:
        raise AssertionError(f"{label}: missing marker {needle!r}")


def require_regex(label: str, haystack: str, pattern: str) -> None:
    if re.search(pattern, haystack, re.S) is None:
        raise AssertionError(f"{label}: missing pattern {pattern!r}")


def main() -> int:
    contract = read(CONTRACT)
    smoke = read(SMOKE)

    sources = {
        "sdxl": read(OT / "modules/model/StableDiffusionXLModel.py"),
        "sd3": read(OT / "modules/model/StableDiffusion3Model.py"),
        "qwen": read(OT / "modules/model/QwenModel.py"),
        "ernie": read(OT / "modules/model/ErnieModel.py"),
        "flux1": read(OT / "modules/model/FluxModel.py"),
        "flux2": read(OT / "modules/model/Flux2Model.py"),
        "chroma": read(OT / "modules/model/ChromaModel.py"),
        "zimage": read(OT / "modules/model/ZImageModel.py"),
        "anima": read(ANIMA_REF / "modules/model/AnimaModel.py"),
        "sdxl_setup": read(OT / "modules/modelSetup/BaseStableDiffusionXLSetup.py"),
        "sd3_setup": read(OT / "modules/modelSetup/BaseStableDiffusion3Setup.py"),
        "flux1_setup": read(OT / "modules/modelSetup/BaseFluxSetup.py"),
        "flux2_setup": read(OT / "modules/modelSetup/BaseFlux2Setup.py"),
        "chroma_setup": read(OT / "modules/modelSetup/BaseChromaSetup.py"),
        "qwen_setup": read(OT / "modules/modelSetup/BaseQwenSetup.py"),
        "ernie_setup": read(OT / "modules/modelSetup/BaseErnieSetup.py"),
        "zimage_setup": read(OT / "modules/modelSetup/BaseZImageSetup.py"),
        "anima_setup": read(ANIMA_REF / "modules/modelSetup/BaseAnimaSetup.py"),
    }

    # Reference-only markers from OneTrainer.
    require("SDXL token length", sources["sdxl"], "max_length=77")
    require("SDXL combine", sources["sdxl"], "torch.concat([text_encoder_1_output, text_encoder_2_output], dim=-1)")
    require("SDXL cached pooled field", sources["sdxl_setup"], "text_encoder_2_pooled_state")
    require("SDXL train cache tokens", sources["sdxl_setup"], "tokens_1=batch['tokens_1']")

    require("SD3.5 token length", sources["sd3"], "max_length=77")
    require("SD3.5 combine pad", sources["sd3"], "torch.nn.functional.pad")
    require("SD3.5 combine append T5", sources["sd3"], "torch.cat([prompt_embedding, text_encoder_3_output], dim=-2)")
    require("SD3.5 masks", sources["sd3_setup"], "tokens_mask_3=batch.get(\"tokens_mask_3\")")

    require("Qwen template", sources["qwen"], "DEFAULT_PROMPT_TEMPLATE")
    require("Qwen crop", sources["qwen"], "DEFAULT_PROMPT_TEMPLATE_CROP_START = 34")
    require("Qwen length", sources["qwen"], "PROMPT_MAX_LENGTH = 512")
    require("Qwen prune pad16", sources["qwen"], "max_seq_length += (16 - max_seq_length % 16)")
    require("Qwen transformer mask", sources["qwen_setup"], "encoder_hidden_states_mask=text_attention_mask")

    require("Ernie length", sources["ernie"], "PROMPT_MAX_LENGTH = 512")
    require("Ernie hidden layer", sources["ernie"], "HIDDEN_STATES_LAYER = -2")
    require("Ernie lengths", sources["ernie"], "text_lengths = tokens_mask.sum(dim=1).long()")

    require("Anima qwen+t5", sources["anima"], "Two-stage encoding: Qwen3 text encoder")
    require("Anima T5 ids", sources["anima"], "target_input_ids=t5_ids")
    require("Anima dense context", sources["anima"], "conditioner output is always (B, 512, 1024)")
    require("Anima cached dense output", sources["anima"], "text_encoder_output, when provided from cache, is already the conditioner output")

    require("Flux1 CLIP pooled", sources["flux1"], "pooled_text_encoder_1_output")
    require("Flux1 T5 fallback", sources["flux1"], "self.tokenizer_2.model_max_length")
    require("Flux1 ids", sources["flux1_setup"], "txt_ids=text_ids")

    require("Flux2 Mistral layers", sources["flux2"], "MISTRAL_HIDDEN_STATES_LAYERS = [10, 20, 30]")
    require("Flux2 Qwen layers", sources["flux2"], "QWEN3_HIDDEN_STATES_LAYERS = [9, 18, 27]")
    require("Flux2 no thinking", sources["flux2"], "enable_thinking=False")
    require("Flux2 text ids", sources["flux2_setup"], "text_ids = model.prepare_text_ids(text_encoder_output)")

    require("Chroma unmask", sources["chroma"], "unmask 1 token")
    require("Chroma pad16", sources["chroma"], "max_seq_length += (16 - max_seq_length % 16)")
    require("Chroma ids", sources["chroma_setup"], "txt_ids=text_ids.to(dtype=model.train_dtype.torch_dtype())")

    require("ZImage prompt length", sources["zimage"], "PROMPT_MAX_LENGTH = 512")
    require("ZImage thinking", sources["zimage"], "enable_thinking=True")
    require("ZImage list", sources["zimage"], "embeddings_list = [sample[bool_attention_mask[i]]")

    # Mojo contract markers.
    for label, marker in [
        ("SDXL contract", "OT_TEXT_SDXL"),
        ("SD3.5 contract", "OT_OUTPUT_SD3_PAD_AND_APPEND_T5"),
        ("Qwen crop contract", "OT_PROMPT_QWEN_IMAGE_TEMPLATE_CROP34"),
        ("Qwen token input", "token1 = 546"),
        ("Ernie lengths contract", "OT_MASK_LENGTHS"),
        ("Anima dense contract", "OT_PROMPT_ANIMA_QWEN_AND_T5"),
        ("Flux1 pooled contract", "OT_OUTPUT_FLUX_CLIP_POOLED_T5"),
        ("Flux2 dev contract", "OT_PROMPT_MISTRAL_SYSTEM_CHAT"),
        ("Klein contract", "OT_PROMPT_QWEN3_CHAT_NO_THINKING"),
        ("Chroma contract", "OT_TEXT_CHROMA"),
        ("ZImage variable contract", "OT_OUTPUT_VARIABLE_LIST"),
        ("cache readiness struct", "OneTrainerTextCacheReadinessContract"),
        ("train cache readiness", "OT_TEXT_CACHE_USE_TRAIN"),
        ("sample cache readiness", "OT_TEXT_CACHE_USE_SAMPLE"),
        ("cache field validator", "validate_ot_text_conditioning_cache_fields"),
        ("fail loud cache policy", "raise_before_sample_or_train_when_required_cache_fields_are_missing"),
        ("Flux2 text ids readiness", "runtime:model.prepare_text_ids(text_encoder_output)"),
        ("latent image ids readiness", "runtime:model.prepare_latent_image_ids"),
        ("dtype policy", "preserve_checkpoint_or_train_dtype_at_tensor_boundaries"),
    ]:
        require(label, contract, marker)

    require_regex("Qwen kept hidden", contract, r"hidden1 = 3584\s+out_hidden = 3584")
    require_regex("Flux2 layer count", contract, r"OT_OUTPUT_FLUX2_LAYER_CAT.*?layers = 3")
    require_regex("ZImage variable seq", contract, r"family == OT_TEXT_ZIMAGE:.*?out_seq = 0")
    require_regex("Qwen cached sample mask", contract, r"family == OT_TEXT_QWEN:\s+return True")
    require_regex("Anima sample dense no mask", contract, r"cached_sample_requires_mask.*?return False")
    require_regex("cache validation raises missing", contract, r"missing fields for.*?required=")

    # This contract must not introduce tensor storage casts. Scalar scheduling is
    # irrelevant here, so any explicit dtype machinery in this file is suspicious.
    forbidden_contract_markers = ["STDtype.F32", "DType.float32", "to_f32", "float32_hidden"]
    for marker in forbidden_contract_markers:
        if marker in contract or marker in smoke:
            raise AssertionError(f"dtype boundary guard: forbidden marker {marker!r}")

    print("OneTrainer conditioning contract static guard PASS")
    print("  checked source markers: OneTrainer + OneTrainer-anima-ref")
    print("  evidence type: contract/static only, not full end-to-end conditioning parity")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"OneTrainer conditioning contract static guard FAIL: {exc}")
        raise SystemExit(1)
