# chroma_lora_serenity_real_artifact_roundtrip.mojo — round-trip a REAL SerenityTrainer
# Chroma LoRA artifact through the Mojo reference trainer-key resume/save surface.
#
# Loads the artifact with load_chroma_lora_resume_for_layer_filter (reference trainer
# baseline layer_filter "attn,ff.net", 304 adapters), re-saves through
# save_chroma_lora_for_layer_filter, and asserts inventory. Payload byte
# comparison against the original is done by the companion Python step
# (scripts/check_chroma_lora_keys.py + a byte diff) — this gate proves the
# Mojo surface consumes and reproduces the real artifact without loss-of-key.
#
#   pixi run mojo run -I . serenitymojo/models/chroma/parity/\
#     chroma_lora_serenity_real_artifact_roundtrip.mojo <in.safetensors> <out.safetensors>
#
# Oracle for the 2026-07-16 gate:
#   /home/alex/mojodiffusion/output/chroma_serenity_eri2wt_100_baseline/lora.safetensors
#   (SerenityTrainer 423c3b36, "#chroma LoRA 24GB" preset, rank 16, alpha 1.0)

from std.sys import argv
from max.gpu.host import DeviceContext

from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.chroma.chroma_stack_lora import (
    load_chroma_lora_resume_for_layer_filter,
    save_chroma_lora_for_layer_filter,
)

comptime NUM_DOUBLE = 19
comptime NUM_SINGLE = 38
comptime RANK = 16
comptime ALPHA = Float32(1.0)
comptime BASELINE_FILTER = "attn,ff.net"


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def main() raises:
    var args = argv()
    _require(len(args) >= 3, String("usage: <in.safetensors> <out.safetensors>"))
    var in_path = String(args[1])
    var out_path = String(args[2])

    var ctx = DeviceContext()

    var src = SafeTensors.open(in_path)
    var n_src = len(src.tensors)
    print("[roundtrip] source tensors:", n_src)
    _require(n_src == 912, String("expected 912 source tensors (304 x alpha/down/up)"))

    var set = load_chroma_lora_resume_for_layer_filter(
        NUM_DOUBLE, NUM_SINGLE, RANK, ALPHA, String(BASELINE_FILTER), in_path, ctx
    )
    print("[roundtrip] resumed adapters:", len(set.ad))
    _require(len(set.ad) > 0, String("no adapters resumed"))

    var saved = save_chroma_lora_for_layer_filter(
        set, String(BASELINE_FILTER), out_path, ctx
    )
    print("[roundtrip] saved adapters:", saved)
    _require(saved == 304, String("expected 304 saved adapters"))

    var dst = SafeTensors.open(out_path)
    _require(len(dst.tensors) == 912, String("expected 912 re-saved tensors"))
    for key in src.tensors:
        _require(key in dst.tensors, String("re-saved artifact missing key ") + key)
    print("[roundtrip] PASS: 912/912 keys present after Mojo resume->save")
