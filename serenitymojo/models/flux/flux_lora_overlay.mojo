# flux_lora_overlay.mojo — runtime additive LoRA overlay for FLUX.1-dev inference.
#
# LoRA is ADDED, NEVER fused into the saved model (project HARD RULE): this builds
# an in-memory delta W += scale·(up @ down) and adds it onto the OFFLOADED block
# weights as they stream off disk, per denoise step. The base checkpoint on disk
# is untouched.
#
# Format: Kohya / sd-scripts BFL LoRA (`lora_unet_double_blocks_{i}_img_attn_qkv`
# .lora_down/.lora_up/.alpha). These keys map 1:1 onto the BFL module weight
# names the Mojo DiT uses, and the LoRA targets the FULL fused weights (qkv
# [9216], mlp [12288], single linear1 [21504], …) so delta = up@down has exactly
# the base weight's shape — no q/k/v slicing, no diffusers remap.
#
#   delta = (up[out,r] @ down[r,in]) * (alpha/r) * multiplier      (== base shape)
#   overlaid_weight = base + delta                                  (bf16)
#
# An EMPTY overlay (no entries) is the base path: Flux1Offloaded.forward branches
# to the untouched _block_model when len(entries)==0, so the proven base spine is
# bit-identical when no LoRA is given.

from std.collections import List, Dict
from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import add, mul_scalar, permute


# ── one LoRA target: down [r,in], up [out,r], scale = alpha/r * multiplier ────
@fieldwise_init
struct FluxLoraEntry(Movable):
    var down: Tensor    # [r, in] BF16
    var up: Tensor      # [out, r] BF16
    var scale: Float32


@fieldwise_init
struct FluxLoraOverlay(Movable):
    # keyed by FULL BFL base weight name, e.g. "double_blocks.0.img_attn.qkv.weight"
    var entries: Dict[String, ArcPointer[FluxLoraEntry]]

    @staticmethod
    def empty() -> FluxLoraOverlay:
        return FluxLoraOverlay(Dict[String, ArcPointer[FluxLoraEntry]]())

    def has(self, name: String) -> Bool:
        return name in self.entries

    def count(self) -> Int:
        return len(self.entries)

    # base + scale·(up @ down). linear(up, downᵀ) = up @ down (linear does x@Wᵀ).
    def overlaid(self, base: Tensor, name: String, ctx: DeviceContext) raises -> Tensor:
        ref e = self.entries[name][]
        var dT = permute(e.down, [1, 0], ctx)                 # [in, r]
        var delta = linear(e.up, dT, Optional[Tensor](), ctx) # [out, in] = up@down
        var ds = mul_scalar(delta, e.scale, ctx)
        var bsh = base.shape()
        var dsh = ds.shape()
        if len(bsh) != 2 or len(dsh) != 2 or bsh[0] != dsh[0] or bsh[1] != dsh[1]:
            raise Error(
                String("flux lora overlay: delta shape mismatch for ") + name
                + " base [" + String(bsh[0]) + "," + String(bsh[1] if len(bsh) > 1 else 0)
                + "] delta [" + String(dsh[0]) + "," + String(dsh[1] if len(dsh) > 1 else 0) + "]"
            )
        return add(base, ds, ctx)


# ── Kohya BFL suffix -> BFL module weight suffix (module names contain '_', so a
#    fixed lookup is required; naive '_'->'.' would corrupt img_attn etc.) ──────
def _double_targets() -> Tuple[List[String], List[String]]:
    var ko = List[String]()
    var bfl = List[String]()
    ko.append("img_attn_qkv");  bfl.append("img_attn.qkv")
    ko.append("img_attn_proj"); bfl.append("img_attn.proj")
    ko.append("img_mlp_0");     bfl.append("img_mlp.0")
    ko.append("img_mlp_2");     bfl.append("img_mlp.2")
    ko.append("img_mod_lin");   bfl.append("img_mod.lin")
    ko.append("txt_attn_qkv");  bfl.append("txt_attn.qkv")
    ko.append("txt_attn_proj"); bfl.append("txt_attn.proj")
    ko.append("txt_mlp_0");     bfl.append("txt_mlp.0")
    ko.append("txt_mlp_2");     bfl.append("txt_mlp.2")
    ko.append("txt_mod_lin");   bfl.append("txt_mod.lin")
    return (ko^, bfl^)


def _single_targets() -> Tuple[List[String], List[String]]:
    var ko = List[String]()
    var bfl = List[String]()
    ko.append("linear1");        bfl.append("linear1")
    ko.append("linear2");        bfl.append("linear2")
    ko.append("modulation_lin"); bfl.append("modulation.lin")
    return (ko^, bfl^)


def _try_add(
    st: ShardedSafeTensors,
    nameset: Dict[String, Bool],
    mut entries: Dict[String, ArcPointer[FluxLoraEntry]],
    mut n_added: Int,
    stem: String,
    bfl_weight: String,
    multiplier: Float32,
    ctx: DeviceContext,
) raises:
    var down_k = stem + ".lora_down.weight"
    var up_k = stem + ".lora_up.weight"
    if down_k not in nameset or up_k not in nameset:
        return
    var down = Tensor.from_view_as_bf16(st.tensor_view(down_k), ctx)  # [r,in]
    var up = Tensor.from_view_as_bf16(st.tensor_view(up_k), ctx)      # [out,r]
    var rank = down.shape()[0]
    var scale = multiplier
    var alpha_k = stem + ".alpha"
    if alpha_k in nameset:
        var a = Tensor.from_view_as_f32(st.tensor_view(alpha_k), ctx).to_host(ctx)
        if len(a) > 0 and rank > 0:
            scale = (a[0] / Float32(rank)) * multiplier
    entries[bfl_weight] = ArcPointer(FluxLoraEntry(down^, up^, scale))
    n_added += 1


# ── Load a Kohya BFL FLUX LoRA into an overlay (multiplier = LoRA strength) ────
def load_flux_kohya_lora(
    path: String, num_double: Int, num_single: Int,
    multiplier: Float32, ctx: DeviceContext,
) raises -> FluxLoraOverlay:
    var st = ShardedSafeTensors.open(path)
    var names = st.names()
    var nameset = Dict[String, Bool]()
    for ref nm in names:
        nameset[nm] = True

    var entries = Dict[String, ArcPointer[FluxLoraEntry]]()
    var n_added = 0

    var dt = _double_targets()
    for bi in range(num_double):
        for ti in range(len(dt[0])):
            var stem = String("lora_unet_double_blocks_") + String(bi) + "_" + dt[0][ti]
            var bfl = String("double_blocks.") + String(bi) + "." + dt[1][ti] + ".weight"
            _try_add(st, nameset, entries, n_added, stem, bfl, multiplier, ctx)

    var stg = _single_targets()
    for bi in range(num_single):
        for ti in range(len(stg[0])):
            var stem = String("lora_unet_single_blocks_") + String(bi) + "_" + stg[0][ti]
            var bfl = String("single_blocks.") + String(bi) + "." + stg[1][ti] + ".weight"
            _try_add(st, nameset, entries, n_added, stem, bfl, multiplier, ctx)

    print("[flux-lora] loaded", n_added, "LoRA targets from", path,
          "(multiplier", multiplier, ")")
    if n_added == 0:
        raise Error(
            String("flux lora: no Kohya BFL targets matched in ") + path
            + " (expected keys like lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight;"
            + " diffusers-format LoRAs are not yet supported by this overlay)"
        )
    return FluxLoraOverlay(entries^)
