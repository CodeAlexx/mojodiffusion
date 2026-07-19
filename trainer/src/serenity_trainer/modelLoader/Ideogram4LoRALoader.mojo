# Ideogram4LoRALoader.mojo — runtime reload for ai-toolkit/PEFT Ideogram4 LoRAs.
#
# Reads block-stack adapters from safetensors:
#   diffusion_model.layers.N.<slot>.lora_A.weight  [rank, in]
#   diffusion_model.layers.N.<slot>.lora_B.weight  [out, rank]
#   optional .alpha scalar; if absent, alpha defaults to rank.
#
# The real ai-toolkit fixture:
#   /home/alex/Downloads/dever_arcane_style_ideogram4%20%28arcvfx%29.safetensors
# uses exactly the 34*6 block target set, rank 32, BF16, no alpha tensors, no
# global adapters.
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.tensor import Tensor

from serenity_trainer.model.Ideogram4LoRABlock import Ideogram4LoraSet
from serenity_trainer.modelSetup.ideogram4LoraTargets import (
    ideogram4_block_lora_save_prefixes,
    ideogram4_convert_lora_key_before_save,
)
from serenity_trainer.module.LoRAModule import LoraAdapter


comptime LArc = ArcPointer[LoraAdapter]


def load_ideogram4_block_stack_lora(
    path: String,
    ctx: DeviceContext,
) raises -> Ideogram4LoraSet:
    var sharded = ShardedSafeTensors.open(path)
    var have = Dict[String, Int]()
    for ref nm in sharded.names():
        have[nm] = 1

    var src_prefixes = ideogram4_block_lora_save_prefixes()
    var adapters = List[LArc]()
    var rank = -1
    for i in range(len(src_prefixes)):
        var pre = ideogram4_convert_lora_key_before_save(src_prefixes[i])
        var ak = pre + String(".lora_A.weight")
        var bk = pre + String(".lora_B.weight")
        if not (ak in have):
            raise Error(String("load_ideogram4_block_stack_lora: missing ") + ak)
        if not (bk in have):
            raise Error(String("load_ideogram4_block_stack_lora: missing ") + bk)

        var a = Tensor.from_view(sharded.tensor_view(ak), ctx)
        var b = Tensor.from_view(sharded.tensor_view(bk), ctx)
        if rank < 0:
            rank = a.shape()[0]
        elif a.shape()[0] != rank or b.shape()[1] != rank:
            raise Error("load_ideogram4_block_stack_lora: inconsistent rank")

        var alpha = Float32(rank)
        var alpha_key = pre + String(".alpha")
        if alpha_key in have:
            var alt = Tensor.from_view(sharded.tensor_view(alpha_key), ctx)
            var host = alt.to_host(ctx)
            if len(host) > 0:
                alpha = host[0]

        adapters.append(LArc(LoraAdapter(a^, b^, rank, alpha)))

    return Ideogram4LoraSet(adapters^, len(src_prefixes) // 6, rank, True)
