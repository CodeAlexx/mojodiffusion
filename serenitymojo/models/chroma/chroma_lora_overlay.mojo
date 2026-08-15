# chroma_lora_overlay.mojo — runtime additive LoRA for Chroma1-HD inference.
#
# Chroma checkpoints use Diffusers split projections while torchref Chroma
# LoRAs may use the original BFL fused names.  This loader maps both surfaces
# to the exact projection names consumed by the product forward:
#
#   base_out + multiplier * B(A(x))
#
# The saved checkpoint is never modified.  Fused BFL QKV/linear1 `B` matrices
# are row-sliced exactly as Comfy/Swarm patch them into the split Diffusers
# projections.  With an empty overlay the caller executes the untouched base
# linear path.

from std.collections import Dict, List, Optional
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import mul_scalar, slice
from serenitymojo.tensor import Tensor


@fieldwise_init
struct ChromaLoraEntry(Movable):
    var down: Tensor  # [rank, in]
    var up: Tensor  # [out, rank]
    var scale: Float32


@fieldwise_init
struct ChromaLoraOverlay(Movable):
    # Full Diffusers base weight name, for example
    # "transformer_blocks.0.attn.to_q.weight".
    var entries: Dict[String, ArcPointer[ChromaLoraEntry]]

    @staticmethod
    def empty() -> ChromaLoraOverlay:
        return ChromaLoraOverlay(Dict[String, ArcPointer[ChromaLoraEntry]]())

    def has(self, name: String) -> Bool:
        return name in self.entries

    def count(self) -> Int:
        return len(self.entries)

    def delta(
        self, x: Tensor, name: String, ctx: DeviceContext
    ) raises -> Tensor:
        if name not in self.entries:
            raise Error(String("chroma lora: missing admitted target ") + name)
        ref entry = self.entries[name][]
        var down_out = linear(x, entry.down, Optional[Tensor](), ctx)
        var up_out = linear(down_out, entry.up, Optional[Tensor](), ctx)
        return mul_scalar(up_out, entry.scale, ctx)


def _add_factor(
    st: ShardedSafeTensors,
    nameset: Dict[String, Bool],
    mut entries: Dict[String, ArcPointer[ChromaLoraEntry]],
    mut n_added: Int,
    stem: String,
    target: String,
    up_offset: Int,
    up_rows: Int,
    multiplier: Float32,
    ctx: DeviceContext,
) raises:
    var down_key = stem + ".lora_A.weight"
    var up_key = stem + ".lora_B.weight"
    if down_key not in nameset or up_key not in nameset:
        return
    if target in entries:
        raise Error(String("chroma lora: duplicate target ") + target)

    var down = Tensor.from_view_as_bf16(st.tensor_view(down_key), ctx)
    var full_up = Tensor.from_view_as_bf16(st.tensor_view(up_key), ctx)
    var ush = full_up.shape()
    if (
        len(ush) != 2
        or up_offset < 0
        or up_rows <= 0
        or up_offset + up_rows > ush[0]
    ):
        raise Error(
            String("chroma lora: invalid B row slice for ") + stem
            + String(" offset=") + String(up_offset)
            + String(" rows=") + String(up_rows)
        )
    var up = slice(full_up, 0, up_offset, up_rows, ctx)

    var rank = down.shape()[0]
    var scale = multiplier
    var alpha_key = stem + ".alpha"
    if alpha_key in nameset:
        var alpha = Tensor.from_view_as_f32(
            st.tensor_view(alpha_key), ctx
        ).to_host(ctx)
        if len(alpha) > 0 and rank > 0:
            scale = alpha[0] / Float32(rank) * multiplier

    entries[target] = ArcPointer(ChromaLoraEntry(down^, up^, scale))
    n_added += 1


def _add_full_variants(
    st: ShardedSafeTensors,
    nameset: Dict[String, Bool],
    mut entries: Dict[String, ArcPointer[ChromaLoraEntry]],
    mut n_added: Int,
    diffusers_stem: String,
    target: String,
    out_rows: Int,
    multiplier: Float32,
    ctx: DeviceContext,
) raises:
    # PEFT files are seen both with and without a leading `transformer.`.
    _add_factor(
        st,
        nameset,
        entries,
        n_added,
        String("transformer.") + diffusers_stem,
        target,
        0,
        out_rows,
        multiplier,
        ctx,
    )
    if target not in entries:
        _add_factor(
            st,
            nameset,
            entries,
            n_added,
            diffusers_stem,
            target,
            0,
            out_rows,
            multiplier,
            ctx,
        )


def load_chroma_lora(
    path: String,
    num_double: Int,
    num_single: Int,
    multiplier: Float32,
    ctx: DeviceContext,
) raises -> ChromaLoraOverlay:
    """Load torchref BFL-fused or Diffusers/PEFT Chroma LoRA factors."""
    var st = ShardedSafeTensors.open(path)
    var nameset = Dict[String, Bool]()
    for ref name in st.names():
        nameset[name] = True

    var entries = Dict[String, ArcPointer[ChromaLoraEntry]]()
    var n_added = 0

    for bi in range(num_double):
        var bfl = String("diffusion_model.double_blocks.") + String(bi)
        var dif = String("transformer_blocks.") + String(bi)

        # Fused BFL image QKV -> split Diffusers image projections.
        _add_factor(
            st, nameset, entries, n_added, bfl + ".img_attn.qkv",
            dif + ".attn.to_q.weight", 0, 3072, multiplier, ctx,
        )
        _add_factor(
            st, nameset, entries, n_added, bfl + ".img_attn.qkv",
            dif + ".attn.to_k.weight", 3072, 3072, multiplier, ctx,
        )
        _add_factor(
            st, nameset, entries, n_added, bfl + ".img_attn.qkv",
            dif + ".attn.to_v.weight", 6144, 3072, multiplier, ctx,
        )
        # Fused BFL text QKV -> split Diffusers added-context projections.
        _add_factor(
            st, nameset, entries, n_added, bfl + ".txt_attn.qkv",
            dif + ".attn.add_q_proj.weight", 0, 3072, multiplier, ctx,
        )
        _add_factor(
            st, nameset, entries, n_added, bfl + ".txt_attn.qkv",
            dif + ".attn.add_k_proj.weight", 3072, 3072, multiplier, ctx,
        )
        _add_factor(
            st, nameset, entries, n_added, bfl + ".txt_attn.qkv",
            dif + ".attn.add_v_proj.weight", 6144, 3072, multiplier, ctx,
        )
        _add_factor(
            st, nameset, entries, n_added, bfl + ".img_attn.proj",
            dif + ".attn.to_out.0.weight", 0, 3072, multiplier, ctx,
        )
        _add_factor(
            st, nameset, entries, n_added, bfl + ".txt_attn.proj",
            dif + ".attn.to_add_out.weight", 0, 3072, multiplier, ctx,
        )
        _add_factor(
            st, nameset, entries, n_added, bfl + ".img_mlp.0",
            dif + ".ff.net.0.proj.weight", 0, 12288, multiplier, ctx,
        )
        _add_factor(
            st, nameset, entries, n_added, bfl + ".img_mlp.2",
            dif + ".ff.net.2.weight", 0, 3072, multiplier, ctx,
        )
        _add_factor(
            st, nameset, entries, n_added, bfl + ".txt_mlp.0",
            dif + ".ff_context.net.0.proj.weight", 0, 12288, multiplier, ctx,
        )
        _add_factor(
            st, nameset, entries, n_added, bfl + ".txt_mlp.2",
            dif + ".ff_context.net.2.weight", 0, 3072, multiplier, ctx,
        )

        # Native Diffusers/PEFT form.
        _add_full_variants(
            st, nameset, entries, n_added, dif + ".attn.to_q",
            dif + ".attn.to_q.weight", 3072, multiplier, ctx,
        )
        _add_full_variants(
            st, nameset, entries, n_added, dif + ".attn.to_k",
            dif + ".attn.to_k.weight", 3072, multiplier, ctx,
        )
        _add_full_variants(
            st, nameset, entries, n_added, dif + ".attn.to_v",
            dif + ".attn.to_v.weight", 3072, multiplier, ctx,
        )
        _add_full_variants(
            st, nameset, entries, n_added, dif + ".attn.add_q_proj",
            dif + ".attn.add_q_proj.weight", 3072, multiplier, ctx,
        )
        _add_full_variants(
            st, nameset, entries, n_added, dif + ".attn.add_k_proj",
            dif + ".attn.add_k_proj.weight", 3072, multiplier, ctx,
        )
        _add_full_variants(
            st, nameset, entries, n_added, dif + ".attn.add_v_proj",
            dif + ".attn.add_v_proj.weight", 3072, multiplier, ctx,
        )
        _add_full_variants(
            st, nameset, entries, n_added, dif + ".attn.to_out.0",
            dif + ".attn.to_out.0.weight", 3072, multiplier, ctx,
        )
        _add_full_variants(
            st, nameset, entries, n_added, dif + ".attn.to_add_out",
            dif + ".attn.to_add_out.weight", 3072, multiplier, ctx,
        )
        _add_full_variants(
            st, nameset, entries, n_added, dif + ".ff.net.0.proj",
            dif + ".ff.net.0.proj.weight", 12288, multiplier, ctx,
        )
        _add_full_variants(
            st, nameset, entries, n_added, dif + ".ff.net.2",
            dif + ".ff.net.2.weight", 3072, multiplier, ctx,
        )
        _add_full_variants(
            st, nameset, entries, n_added, dif + ".ff_context.net.0.proj",
            dif + ".ff_context.net.0.proj.weight", 12288, multiplier, ctx,
        )
        _add_full_variants(
            st, nameset, entries, n_added, dif + ".ff_context.net.2",
            dif + ".ff_context.net.2.weight", 3072, multiplier, ctx,
        )

    for bi in range(num_single):
        var bfl = String("diffusion_model.single_blocks.") + String(bi)
        var dif = String("single_transformer_blocks.") + String(bi)

        # BFL linear1 rows are [q | k | v | MLP].
        _add_factor(
            st, nameset, entries, n_added, bfl + ".linear1",
            dif + ".attn.to_q.weight", 0, 3072, multiplier, ctx,
        )
        _add_factor(
            st, nameset, entries, n_added, bfl + ".linear1",
            dif + ".attn.to_k.weight", 3072, 3072, multiplier, ctx,
        )
        _add_factor(
            st, nameset, entries, n_added, bfl + ".linear1",
            dif + ".attn.to_v.weight", 6144, 3072, multiplier, ctx,
        )
        _add_factor(
            st, nameset, entries, n_added, bfl + ".linear1",
            dif + ".proj_mlp.weight", 9216, 12288, multiplier, ctx,
        )
        _add_factor(
            st, nameset, entries, n_added, bfl + ".linear2",
            dif + ".proj_out.weight", 0, 3072, multiplier, ctx,
        )

        _add_full_variants(
            st, nameset, entries, n_added, dif + ".attn.to_q",
            dif + ".attn.to_q.weight", 3072, multiplier, ctx,
        )
        _add_full_variants(
            st, nameset, entries, n_added, dif + ".attn.to_k",
            dif + ".attn.to_k.weight", 3072, multiplier, ctx,
        )
        _add_full_variants(
            st, nameset, entries, n_added, dif + ".attn.to_v",
            dif + ".attn.to_v.weight", 3072, multiplier, ctx,
        )
        _add_full_variants(
            st, nameset, entries, n_added, dif + ".proj_mlp",
            dif + ".proj_mlp.weight", 12288, multiplier, ctx,
        )
        _add_full_variants(
            st, nameset, entries, n_added, dif + ".proj_out",
            dif + ".proj_out.weight", 3072, multiplier, ctx,
        )

    print(
        "[chroma-lora] loaded",
        n_added,
        "projection factors from",
        path,
        "(multiplier",
        multiplier,
        ")",
    )
    if n_added == 0:
        raise Error(
            String("chroma lora: no torchref BFL or Diffusers/PEFT targets matched in ")
            + path
        )
    return ChromaLoraOverlay(entries^)
