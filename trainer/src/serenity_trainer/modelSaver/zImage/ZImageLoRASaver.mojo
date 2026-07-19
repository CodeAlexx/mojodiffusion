# ZImageLoRASaver.mojo — 1:1 port of Serenity
#   modules/modelSaver/zImage/ZImageLoRASaver.py  (ZImageLoRASaver)
# composed with the save mechanics of
#   modules/modelSaver/mixin/LoRASaverMixin.py   (_get_state_dict → __save_*)
# and the dispatch wrapper
#   modules/modelSaver/ZImageLoRAModelSaver.py    (make_lora_model_saver(...)).
#
# Serenity's ZImageLoRASaver._get_state_dict (ZImageLoRASaver.py:18-26):
#   state_dict = {}
#   state_dict |= model.transformer_lora.state_dict()   # the trained adapters
#   state_dict |= model.lora_state_dict                 # any preloaded extras
#   return state_dict
# _get_convert_key_sets returns None (ZImageLoRASaver.py:15-16) → NO OMI/legacy key
# remap; the PEFT/diffusers keys are written verbatim. LoRASaverMixin._save with
# enable_omi_format=False (the default, LoRASaverMixin.py:88-92) routes
# ModelFormat.SAFETENSORS → __save_legacy_safetensors, which (since key_sets is
# None) is byte-identical to __save_safetensors: it just save_file()s the raw
# state dict (LoRASaverMixin.py:54-68).
#
# PEFT save format (this port's LoRA module, module/LoRAModule.mojo header, and
# the Serenity LoRAModule.py save contract): for a wrapped Linear named
# <prefix>, the adapter is written as two tensors:
#   <prefix>.lora_A.weight   = lora_down.weight  = adapter.a  [rank, in]
#   <prefix>.lora_B.weight   = lora_up.weight    = adapter.b  [out, rank]
# (and optionally <prefix>.alpha as a scalar — Serenity stores alpha inside the
# module's state_dict; here alpha is recoverable from rank·scale and is emitted as
# a 1-element tensor for round-trip fidelity, matching ai-toolkit/PEFT exporters).
#
# The Z-Image LoRA prefixes are exactly the 30·7 module paths enumerated by
# modelSetup/ZImageLoRASetup.mojo::zimage_lora_target_prefixes (block-major,
# slot-minor): "layers.<b>.attention.to_q", …, "layers.<b>.feed_forward.w2".
#
# Reuses ONLY serenitymojo {tensor, io, ops}. No serenitymojo model code imported.
# Dtype: BF16 storage written verbatim (D2H raw byte copy, no F32 cast) per
# save_safetensors. A `dtype` override is supported by casting each tensor before
# write (mirrors LoRASaverMixin._convert_state_dict_dtype).

from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import zeros_device, add_scalar

from serenity_trainer.model.ZImageModel import ZImageLoraSet, ZSLOTS
# Pull lora_module_prefix from the LEAF target module (no setup-spec dep) — the
# saver only needs the host key metadata, not the predict spec.
from serenity_trainer.modelSetup import zImageLoraTargets as LT


comptime TArc = ArcPointer[Tensor]


# Whether to also emit a "<prefix>.alpha" scalar per adapter (PEFT/ai-toolkit do).
comptime EMIT_ALPHA = True


# ── build the LoRA state dict (names + tensors) from the trained adapter set ───
# Mirrors ZImageLoRASaver._get_state_dict (ZImageLoRASaver.py:18-26): iterate the
# transformer_lora adapters in the SAME order the setup created them and emit the
# PEFT key pair (+ alpha) for each.  `set.ad[idx]` is block-major/slot-minor, so
# idx == block*ZSLOTS + slot and the prefix is lora_module_prefix(block, slot).
struct ZImageLoraStateDict(Movable):
    var names: List[String]
    var tensors: List[TArc]

    def __init__(out self, var names: List[String], var tensors: List[TArc]):
        self.names = names^
        self.tensors = tensors^


def build_zimage_lora_state_dict(
    set: ZImageLoraSet,
    ctx: DeviceContext,
    dtype: STDtype = STDtype.BF16,
) raises -> ZImageLoraStateDict:
    var names = List[String]()
    var tensors = List[TArc]()
    for b in range(set.n_layers):
        for s in range(ZSLOTS):
            var idx = b * ZSLOTS + s
            # Serenity's ACTUAL saved keys (measured from a real Z-Image LoRA):
            #   transformer.layers.<b>.<module>.lora_down.weight = a [rank, in]
            #   transformer.layers.<b>.<module>.lora_up.weight   = b [out, rank]
            #   transformer.layers.<b>.<module>.alpha
            # (LoRAModule's nn.Linear submodules are named lora_down/lora_up; the
            #  state_dict carries those names verbatim — NOT PEFT lora_A/lora_B.)
            var prefix = String("transformer.") + LT.lora_module_prefix(b, s)  # "transformer.layers.<b>.<module>"
            ref ad = set.ad[idx][]

            names.append(prefix + String(".lora_down.weight"))
            tensors.append(TArc(_maybe_cast(ad.a, dtype, ctx)))

            names.append(prefix + String(".lora_up.weight"))
            tensors.append(TArc(_maybe_cast(ad.b, dtype, ctx)))

            if EMIT_ALPHA:
                names.append(prefix + String(".alpha"))
                tensors.append(TArc(_scalar(ad.alpha, dtype, ctx)))
    return ZImageLoraStateDict(names^, tensors^)


# LoRASaverMixin.save → _save → __save_legacy_safetensors (key_sets None ⇒ raw).
# Writes the single-file safetensors at `destination`. Mirrors LoRASaverMixin.py
# __save_legacy_safetensors (:54-68) + save_file (the serenitymojo writer is the
# byte-exact analogue of safetensors.torch.save_file).
def save_zimage_lora(
    set: ZImageLoraSet,
    destination: String,
    ctx: DeviceContext,
    dtype: STDtype = STDtype.BF16,
) raises:
    var sd = build_zimage_lora_state_dict(set, ctx, dtype)
    save_safetensors(sd.names, sd.tensors, destination, ctx)


# ── helpers ───────────────────────────────────────────────────────────────────
def _maybe_cast(t: Tensor, dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
    # _convert_state_dict_dtype (LoRASaverMixin._convert_state_dict_dtype): cast
    # each tensor to the requested save dtype; a dtype no-op is still a device copy
    # so the caller-owned adapter tensors are never aliased into the writer.
    return cast_tensor(t, dtype, ctx)


def _scalar(val: Float32, dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
    # Serenity stores alpha as a 0-dim scalar tensor (shape () ), not [1].
    var sh = List[Int]()          # empty shape → 0-dim
    var z = zeros_device(sh^, dtype, ctx)
    return add_scalar(z, val, ctx)
