# lora_save.mojo — PEFT/ai-toolkit LoRA adapter save & load.
#
# Save format (MOJO_TRAINER_RUNTIME_API_GUIDE.md §21-22; PEFT/ai-toolkit-
# compatible safetensors):
#     <prefix>.lora_A.weight   = A = lora_down.weight  [rank, in]
#     <prefix>.lora_B.weight   = B = lora_up.weight    [out, rank]
#     <prefix>.alpha           = scalar alpha          [] (1 elem)
# This mirrors Serenity's LoRAModuleWrapper.state_dict(), which emits one
# entry per lora sub-module keyed by `module.prefix` (LoRAModule.py line 758),
# and registers `alpha` as a buffer (LoRAModule.py line 303). Serenity's
# internal names are lora_down/lora_up; the PEFT export name is lora_A/lora_B —
# A==down, B==up (runtime guide §22, ai-toolkit convention).
#
# BF16 storage is preserved byte-for-byte: save_safetensors copies each device
# buffer D2H raw (no F32 cast), and load uses Tensor.from_view (dtype-preserving).
#
# Reuses ONLY serenitymojo {io.safetensors_writer, io.sharded, tensor,
# tensor_algebra}. No Python, no MGDS.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.tensor_algebra import zeros_device, add_scalar
from serenity_trainer.module.LoRAModule import LoraAdapter

comptime TArc = ArcPointer[Tensor]


# --- key helpers (PEFT names) -------------------------------------------------
def _key_a(prefix: String) -> String:
    return prefix + ".lora_A.weight"

def _key_b(prefix: String) -> String:
    return prefix + ".lora_B.weight"

def _key_alpha(prefix: String) -> String:
    return prefix + ".alpha"


# A 1-element BF16 tensor holding `alpha` (PEFT stores alpha as a scalar buffer;
# LoRAModule.py line 303 `register_buffer("alpha", torch.tensor(alpha))`).
def _alpha_tensor(alpha: Float32, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int](); sh.append(1)
    var z = zeros_device(sh^, STDtype.BF16, ctx)
    return add_scalar(z, alpha, ctx)


# Save a set of named adapters to one PEFT safetensors file. `prefixes[i]` names
# `adapters[i]`; emits lora_A/lora_B/alpha per adapter, all in one file.
# (Mirrors LoRAModuleWrapper.state_dict() unioning each module's state_dict.)
def save_lora(
    prefixes: List[String],
    adapters: List[ArcPointer[LoraAdapter]],
    path: String,
    ctx: DeviceContext,
) raises:
    if len(prefixes) != len(adapters):
        raise Error("save_lora: prefixes/adapters length mismatch")
    if len(prefixes) == 0:
        raise Error("save_lora: nothing to save")

    var names = List[String]()
    var tensors = List[TArc]()
    for i in range(len(prefixes)):
        var px = prefixes[i]
        ref ad = adapters[i][]
        names.append(_key_a(px))
        tensors.append(TArc(ad.a.clone(ctx)))
        names.append(_key_b(px))
        tensors.append(TArc(ad.b.clone(ctx)))
        names.append(_key_alpha(px))
        tensors.append(TArc(_alpha_tensor(ad.alpha, ctx)))

    save_safetensors(names^, tensors^, path, ctx)


# Convenience: save a single adapter under one prefix.
def save_lora_one(
    prefix: String, var adapter: LoraAdapter, path: String, ctx: DeviceContext
) raises:
    var pxs = List[String](); pxs.append(prefix)
    var boxed = List[ArcPointer[LoraAdapter]]()
    boxed.append(ArcPointer(adapter^))
    save_lora(pxs^, boxed^, path, ctx)


# Load one adapter (by prefix) from a PEFT safetensors file/dir, dtype-preserving.
# `rank`/`alpha` are recovered from the loaded tensors (A.shape[0] == rank; the
# alpha buffer if present, else fall back to the passed `default_alpha`).
# Matches Serenity's rank-from-checkpoint check (LoRAModuleWrapper line 717-719,
# `state_dict[rank_key].shape[0]`).
def load_lora_one(
    path: String, prefix: String, default_alpha: Float32, ctx: DeviceContext
) raises -> LoraAdapter:
    var src = ShardedSafeTensors.open(path)

    var tv_a = src.tensor_view(_key_a(prefix))
    var a = Tensor.from_view(tv_a, ctx)
    var tv_b = src.tensor_view(_key_b(prefix))
    var b = Tensor.from_view(tv_b, ctx)

    var rank = a.shape()[0]  # A is [rank, in]

    var alpha = default_alpha
    var akey = _key_alpha(prefix)
    if akey in src.name_to_shard:
        var tv_al = src.tensor_view(akey)
        var al = Tensor.from_view(tv_al, ctx)
        var host = al.to_host(ctx)  # F32 host read of the 1-elem scalar
        if len(host) > 0:
            alpha = host[0]

    return LoraAdapter(a^, b^, rank, alpha)
