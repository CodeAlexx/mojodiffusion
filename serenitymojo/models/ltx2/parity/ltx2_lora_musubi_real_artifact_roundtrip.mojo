# ltx2_lora_musubi_real_artifact_roundtrip.mojo — round-trip a REAL musubi
# LTX-2 LoRA comfy artifact through the Mojo resume/save surface.
#
# Loads the artifact with load_lora_for_resume (384 video-mode t2v prefixes:
# 48 blocks x {attn1,attn2} x {to_q,to_k,to_v,to_out.0}; the loader's
# diffusion_model.-prefix fallback matches musubi's convert_lora_to_comfy key
# style), re-saves through save_lora_peft with the same diffusion_model keys,
# and asserts key inventory. Payload byte comparison against the original is
# done by the companion Python step (scripts/check_ltx2_lora_keys.py) — this
# gate proves the Mojo surface consumes and reproduces the real artifact
# without loss-of-key. Pattern: chroma_lora_serenity_real_artifact_roundtrip.mojo.
#
#   pixi run mojo build -O2 -I . serenitymojo/models/ltx2/parity/\
#     ltx2_lora_musubi_real_artifact_roundtrip.mojo -o /tmp/ltx2_lora_roundtrip
#   /tmp/ltx2_lora_roundtrip <in.comfy.safetensors> <out.safetensors>
#
# Oracle for the 2026-07-16 gate:
#   /home/alex/mojodiffusion/output/ltx2_musubi_ref/ltx2_musubi_ref.comfy.safetensors
#   (musubi-tuner dd96141, video band-ref recipe, rank 32, alpha 32 -> scale 1.0;
#   comfy format = 768 keys, no alphas)

from std.sys import argv
from max.gpu.host import DeviceContext
from std.collections import List

from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.training.lora_save import (
    NamedLora, load_lora_for_resume, save_lora_peft,
)
from serenitymojo.training.ltx2.lora_surface import (
    block_prefix, diffusion_lora_prefix,
)
from serenitymojo.models.ltx2.ltx2_video_stack import video_lora_names

comptime NUM_LAYERS = 48
comptime SCALE = Float32(1.0)  # rank 32 / alpha 32 (comfy carries no alpha keys)
# t2v = 8 modules/block -> 768 keys; v2v = 10 (+2 video FFN) -> 960 keys.


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def _module_paths(v2v: Bool) raises -> List[String]:
    # video_lora_names(preset) yields "attn1.to_q.weight" etc.; strip ".weight".
    var out = List[String]()
    var names = video_lora_names(1 if v2v else 0)
    for ref n in names:
        var b = n.as_bytes()
        var keep = len(b) - 7
        var s = String("")
        for i in range(keep):
            s += chr(Int(b[i]))
        out.append(s)
    return out^


def main() raises:
    var args = argv()
    _require(len(args) >= 3, String("usage: <in.comfy.safetensors> <out.safetensors> [--v2v]"))
    var in_path = String(args[1])
    var out_path = String(args[2])
    var v2v = False
    for i in range(len(args)):
        if String(args[i]) == "--v2v":
            v2v = True

    var ctx = DeviceContext()

    var mods = _module_paths(v2v)
    var n_slots = len(mods)
    var n_mod = NUM_LAYERS * n_slots
    var n_keys = n_mod * 2  # comfy = {lora_A, lora_B} per module, no alphas
    _require(n_slots == (10 if v2v else 8),
             String("expected ") + String(10 if v2v else 8) + " module paths per block")

    var src = SafeTensors.open(in_path)
    var n_src = len(src.tensors)
    print("[roundtrip] source tensors:", n_src)
    _require(
        n_src == n_keys,
        String("expected ") + String(n_keys) + " source tensors, got " + String(n_src),
    )

    # Model-local prefixes; load_lora_for_resume falls back to the
    # diffusion_model.-prefixed keys the comfy artifact carries.
    var prefixes = List[String]()
    for bi in range(NUM_LAYERS):
        for ref m in mods:
            prefixes.append(block_prefix(bi) + String(".") + m)

    var resumed = load_lora_for_resume(prefixes, SCALE, in_path, ctx)
    print("[roundtrip] resumed adapters:", len(resumed))
    _require(len(resumed) == n_mod,
             String("expected ") + String(n_mod) + " resumed adapters")

    # Re-key with the full comfy prefix for the save.
    var named = List[NamedLora]()
    var k = 0
    for bi in range(NUM_LAYERS):
        for ref m in mods:
            named.append(NamedLora(
                diffusion_lora_prefix(bi, m), resumed[k].adapter.copy(),
            ))
            k += 1

    var saved = save_lora_peft(named, out_path, ctx)
    print("[roundtrip] saved adapter pairs:", saved)
    _require(saved == n_mod, String("expected ") + String(n_mod) + " saved pairs")

    var dst = SafeTensors.open(out_path)
    _require(
        len(dst.tensors) == n_keys,
        String("expected ") + String(n_keys) + " re-saved tensors, got "
        + String(len(dst.tensors)),
    )
    for key in src.tensors:
        _require(key in dst.tensors, String("re-saved artifact missing key ") + key)
    print("[roundtrip] PASS:", n_keys, "keys present after Mojo resume->save (v2v=", v2v, ")")
