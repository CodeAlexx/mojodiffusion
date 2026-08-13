# flux_lora_overlay.mojo — runtime additive LoRA overlay for FLUX.1-dev inference.
#
# LoRA is ADDED, NEVER fused into the saved model (project HARD RULE): this builds
# an in-memory delta W += scale·(up @ down) and adds it onto the OFFLOADED block
# weights as they stream off disk, per denoise step. The base checkpoint on disk
# is untouched.
#
# Formats:
#   * Kohya / sd-scripts BFL LoRA (`lora_unet_double_blocks_{i}_img_attn_qkv`
#     .lora_down/.lora_up/.alpha), whose targets already match fused BFL weights.
#   * Diffusers/PEFT FLUX LoRA (`transformer.transformer_blocks.{i}...lora_A/B`),
#     including separate q/k/v and single-block MLP factors. Those deltas are
#     applied to exact output-row slices of the fused BFL weight in memory.
#
#   delta = (up[out,r] @ down[r,in]) * (alpha/r) * multiplier      (== base shape)
#   overlaid_weight = base + delta                                  (bf16)
#
# An EMPTY overlay (no entries) is the base path: Flux1Offloaded.forward branches
# to the untouched _block_model when len(entries)==0, so the proven base spine is
# bit-identical when no LoRA is given.

from std.collections import List, Dict
from std.memory import ArcPointer
from max.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import add, concat, mul_scalar, permute, slice


# ── one LoRA target: down [r,in], up [out,r], scale = alpha/r * multiplier ────
@fieldwise_init
struct FluxLoraEntry(Movable):
    var down: Tensor    # [r, in] BF16
    var up: Tensor      # [out, r] BF16
    var scale: Float32
    var output_offset: Int


@fieldwise_init
struct FluxLoraTarget(Movable):
    var factors: List[ArcPointer[FluxLoraEntry]]


@fieldwise_init
struct FluxLoraOverlay(Movable):
    # keyed by FULL BFL base weight name, e.g. "double_blocks.0.img_attn.qkv.weight"
    var entries: Dict[String, ArcPointer[FluxLoraTarget]]

    @staticmethod
    def empty() -> FluxLoraOverlay:
        return FluxLoraOverlay(Dict[String, ArcPointer[FluxLoraTarget]]())

    def has(self, name: String) -> Bool:
        return name in self.entries

    def count(self) -> Int:
        return len(self.entries)

    # base + scale·(up @ down). linear(up, downᵀ) = up @ down (linear does x@Wᵀ).
    def overlaid(self, base: Tensor, name: String, ctx: DeviceContext) raises -> Tensor:
        var bsh = base.shape()
        if len(bsh) != 2:
            raise Error(String("flux lora overlay: base is not rank-2 for ") + name)
        var result = base.clone(ctx)
        ref target = self.entries[name][]
        for i in range(len(target.factors)):
            ref e = target.factors[i][]
            var dT = permute(e.down, [1, 0], ctx)                  # [in, r]
            var delta = linear(e.up, dT, Optional[Tensor](), ctx) # [out, in]
            var ds = mul_scalar(delta, e.scale, ctx)
            var dsh = ds.shape()
            if (
                len(dsh) != 2
                or dsh[1] != bsh[1]
                or e.output_offset < 0
                or e.output_offset + dsh[0] > bsh[0]
            ):
                raise Error(
                    String("flux lora overlay: delta slice mismatch for ") + name
                    + " base [" + String(bsh[0]) + "," + String(bsh[1])
                    + "] offset " + String(e.output_offset)
                    + " delta [" + String(dsh[0]) + "," + String(dsh[1]) + "]"
                )
            var base_slice = slice(result, 0, e.output_offset, dsh[0], ctx)
            var updated_slice = add(base_slice, ds, ctx)
            if e.output_offset == 0 and dsh[0] == bsh[0]:
                result = updated_slice^
            elif e.output_offset == 0:
                var after = slice(result, 0, dsh[0], bsh[0] - dsh[0], ctx)
                result = concat(0, ctx, updated_slice, after)
            elif e.output_offset + dsh[0] == bsh[0]:
                var before = slice(result, 0, 0, e.output_offset, ctx)
                result = concat(0, ctx, before, updated_slice)
            else:
                var before = slice(result, 0, 0, e.output_offset, ctx)
                var after = slice(
                    result,
                    0,
                    e.output_offset + dsh[0],
                    bsh[0] - e.output_offset - dsh[0],
                    ctx,
                )
                result = concat(0, ctx, before, updated_slice, after)
        return result^


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
    mut entries: Dict[String, ArcPointer[FluxLoraTarget]],
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
    _append_factor(
        entries,
        bfl_weight,
        FluxLoraEntry(down^, up^, scale, 0),
    )
    n_added += 1


def _append_factor(
    mut entries: Dict[String, ArcPointer[FluxLoraTarget]],
    bfl_weight: String,
    var factor: FluxLoraEntry,
) raises:
    var factors = List[ArcPointer[FluxLoraEntry]]()
    if bfl_weight in entries:
        ref existing = entries[bfl_weight][]
        for i in range(len(existing.factors)):
            factors.append(existing.factors[i])
    factors.append(ArcPointer(factor^))
    entries[bfl_weight] = ArcPointer(FluxLoraTarget(factors^))


def _try_add_diffusers(
    st: ShardedSafeTensors,
    nameset: Dict[String, Bool],
    mut entries: Dict[String, ArcPointer[FluxLoraTarget]],
    mut n_added: Int,
    stem: String,
    bfl_weight: String,
    output_offset: Int,
    multiplier: Float32,
    ctx: DeviceContext,
) raises:
    var down_k = stem + ".lora_A.weight"
    var up_k = stem + ".lora_B.weight"
    if down_k not in nameset or up_k not in nameset:
        return
    var down = Tensor.from_view_as_bf16(st.tensor_view(down_k), ctx)
    var up = Tensor.from_view_as_bf16(st.tensor_view(up_k), ctx)
    var rank = down.shape()[0]
    var scale = multiplier
    var alpha_k = stem + ".alpha"
    if alpha_k in nameset:
        var a = Tensor.from_view_as_f32(st.tensor_view(alpha_k), ctx).to_host(ctx)
        if len(a) > 0 and rank > 0:
            scale = (a[0] / Float32(rank)) * multiplier
    _append_factor(
        entries,
        bfl_weight,
        FluxLoraEntry(down^, up^, scale, output_offset),
    )
    n_added += 1


# ── Load Kohya-BFL or Diffusers/PEFT FLUX LoRA into an additive overlay ───────
def load_flux_kohya_lora(
    path: String, num_double: Int, num_single: Int,
    multiplier: Float32, ctx: DeviceContext,
) raises -> FluxLoraOverlay:
    var st = ShardedSafeTensors.open(path)
    var names = st.names()
    var nameset = Dict[String, Bool]()
    for ref nm in names:
        nameset[nm] = True

    var entries = Dict[String, ArcPointer[FluxLoraTarget]]()
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

    # Diffusers Transformer2DModel / PEFT format. Split q/k/v and the
    # single-block MLP are exact row slices of BFL's fused weights.
    for bi in range(num_double):
        var root = String("transformer.transformer_blocks.") + String(bi)
        var bfl_root = String("double_blocks.") + String(bi)
        _try_add_diffusers(
            st, nameset, entries, n_added, root + ".attn.to_q",
            bfl_root + ".img_attn.qkv.weight", 0, multiplier, ctx,
        )
        _try_add_diffusers(
            st, nameset, entries, n_added, root + ".attn.to_k",
            bfl_root + ".img_attn.qkv.weight", 3072, multiplier, ctx,
        )
        _try_add_diffusers(
            st, nameset, entries, n_added, root + ".attn.to_v",
            bfl_root + ".img_attn.qkv.weight", 6144, multiplier, ctx,
        )
        _try_add_diffusers(
            st, nameset, entries, n_added, root + ".attn.add_q_proj",
            bfl_root + ".txt_attn.qkv.weight", 0, multiplier, ctx,
        )
        _try_add_diffusers(
            st, nameset, entries, n_added, root + ".attn.add_k_proj",
            bfl_root + ".txt_attn.qkv.weight", 3072, multiplier, ctx,
        )
        _try_add_diffusers(
            st, nameset, entries, n_added, root + ".attn.add_v_proj",
            bfl_root + ".txt_attn.qkv.weight", 6144, multiplier, ctx,
        )
        _try_add_diffusers(
            st, nameset, entries, n_added, root + ".attn.to_out.0",
            bfl_root + ".img_attn.proj.weight", 0, multiplier, ctx,
        )
        _try_add_diffusers(
            st, nameset, entries, n_added, root + ".attn.to_add_out",
            bfl_root + ".txt_attn.proj.weight", 0, multiplier, ctx,
        )
        _try_add_diffusers(
            st, nameset, entries, n_added, root + ".ff.net.0.proj",
            bfl_root + ".img_mlp.0.weight", 0, multiplier, ctx,
        )
        _try_add_diffusers(
            st, nameset, entries, n_added, root + ".ff.net.2",
            bfl_root + ".img_mlp.2.weight", 0, multiplier, ctx,
        )
        _try_add_diffusers(
            st, nameset, entries, n_added, root + ".ff_context.net.0.proj",
            bfl_root + ".txt_mlp.0.weight", 0, multiplier, ctx,
        )
        _try_add_diffusers(
            st, nameset, entries, n_added, root + ".ff_context.net.2",
            bfl_root + ".txt_mlp.2.weight", 0, multiplier, ctx,
        )
        _try_add_diffusers(
            st, nameset, entries, n_added, root + ".norm1.linear",
            bfl_root + ".img_mod.lin.weight", 0, multiplier, ctx,
        )
        _try_add_diffusers(
            st, nameset, entries, n_added, root + ".norm1_context.linear",
            bfl_root + ".txt_mod.lin.weight", 0, multiplier, ctx,
        )

    for bi in range(num_single):
        var root = String("transformer.single_transformer_blocks.") + String(bi)
        var bfl_root = String("single_blocks.") + String(bi)
        _try_add_diffusers(
            st, nameset, entries, n_added, root + ".attn.to_q",
            bfl_root + ".linear1.weight", 0, multiplier, ctx,
        )
        _try_add_diffusers(
            st, nameset, entries, n_added, root + ".attn.to_k",
            bfl_root + ".linear1.weight", 3072, multiplier, ctx,
        )
        _try_add_diffusers(
            st, nameset, entries, n_added, root + ".attn.to_v",
            bfl_root + ".linear1.weight", 6144, multiplier, ctx,
        )
        _try_add_diffusers(
            st, nameset, entries, n_added, root + ".proj_mlp",
            bfl_root + ".linear1.weight", 9216, multiplier, ctx,
        )
        _try_add_diffusers(
            st, nameset, entries, n_added, root + ".proj_out",
            bfl_root + ".linear2.weight", 0, multiplier, ctx,
        )
        _try_add_diffusers(
            st, nameset, entries, n_added, root + ".norm.linear",
            bfl_root + ".modulation.lin.weight", 0, multiplier, ctx,
        )

    print("[flux-lora] loaded", n_added, "LoRA factors from", path,
          "(multiplier", multiplier, ")")
    if n_added == 0:
        raise Error(
            String("flux lora: no Kohya-BFL or Diffusers/PEFT targets matched in ") + path
        )
    return FluxLoraOverlay(entries^)
