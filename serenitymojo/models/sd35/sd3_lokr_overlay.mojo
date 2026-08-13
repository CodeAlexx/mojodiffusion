# sd3_lokr_overlay.mojo — creator-compatible runtime LyCORIS LoKr for SD3/3.5.
#
# SimpleTuner SD3 LoKr files use flattened Diffusers module names such as:
#
#   lycoris_transformer_blocks_0_attn_add_q_proj.lokr_w1
#   lycoris_transformer_blocks_0_attn_add_q_proj.lokr_w2
#
# The installed files use full W1 and full W2.  Match ComfyUI/SwarmUI's LoKr
# bypass exactly instead of materializing kron(W1, W2):
#
#   x[..., in_m, in_n] --W2--> [..., in_m, out_k]
#       --transpose--W1--> [..., out_k, out_l] --transpose/flatten-->
#       [..., out_l*out_k]
#
# For both-full LoKr, LyCORIS/Comfy force the internal alpha scale to 1.0, so
# only the user's requested multiplier is applied.  The base checkpoint remains
# untouched and each target is added as base(x) + multiplier*lokr(x).

from std.collections import Dict, Optional
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import mul_scalar, permute, reshape
from serenitymojo.tensor import Tensor


@fieldwise_init
struct Sd3LokrEntry(Movable):
    var w1: Tensor  # [out_l, in_m]
    var w2: Tensor  # [out_k, in_n]
    var multiplier: Float32


@fieldwise_init
struct Sd3LokrOverlay(Movable):
    # Canonical Diffusers module name, without the trailing ".weight".
    var entries: Dict[String, ArcPointer[Sd3LokrEntry]]

    def has(self, name: String) -> Bool:
        return name in self.entries

    def count(self) -> Int:
        return len(self.entries)

    def delta(
        self, x: Tensor, name: String, ctx: DeviceContext
    ) raises -> Tensor:
        if name not in self.entries:
            raise Error(String("sd3 lokr: missing admitted target ") + name)
        ref entry = self.entries[name][]
        var xsh = x.shape()
        if len(xsh) < 2:
            raise Error(String("sd3 lokr: target input must be rank >= 2: ") + name)
        var w1sh = entry.w1.shape()
        var w2sh = entry.w2.shape()
        var in_m = w1sh[1]
        var out_l = w1sh[0]
        var in_n = w2sh[1]
        var out_k = w2sh[0]
        var rows = x.numel() // (in_m * in_n)

        var in3 = List[Int]()
        in3.append(rows)
        in3.append(in_m)
        in3.append(in_n)
        var h = linear(
            reshape(x, in3^, ctx),
            entry.w2,
            Optional[Tensor](),
            ctx,
        )
        var axes = List[Int]()
        axes.append(0)
        axes.append(2)
        axes.append(1)
        h = permute(h, axes.copy(), ctx)
        h = linear(h, entry.w1, Optional[Tensor](), ctx)
        h = permute(h, axes^, ctx)

        var out_sh = xsh.copy()
        out_sh[len(out_sh) - 1] = out_l * out_k
        return mul_scalar(reshape(h, out_sh^, ctx), entry.multiplier, ctx)


def _add_full_lokr(
    st: ShardedSafeTensors,
    mut entries: Dict[String, ArcPointer[Sd3LokrEntry]],
    stem: String,
    target: String,
    hidden: Int,
    multiplier: Float32,
    ctx: DeviceContext,
) raises:
    var w1_key = stem + ".lokr_w1"
    var w2_key = stem + ".lokr_w2"
    var has_w1 = st.has_tensor(w1_key)
    var has_w2 = st.has_tensor(w2_key)
    if not has_w1 and not has_w2:
        return
    if not has_w1 or not has_w2:
        raise Error(String("sd3 lokr: incomplete full factor pair for ") + stem)
    if target in entries:
        raise Error(String("sd3 lokr: duplicate target ") + target)

    var w1 = Tensor.from_view_as_bf16(st.tensor_view(w1_key), ctx)
    var w2 = Tensor.from_view_as_bf16(st.tensor_view(w2_key), ctx)
    var w1sh = w1.shape()
    var w2sh = w2.shape()
    if len(w1sh) != 2 or len(w2sh) != 2:
        raise Error(String("sd3 lokr: full factors must be rank 2 for ") + stem)
    if w1sh[1] * w2sh[1] != hidden or w1sh[0] * w2sh[0] != hidden:
        raise Error(
            String("sd3 lokr: target shape is incompatible with SD3.5 Large for ")
            + stem + String("; expected ") + String(hidden) + String("x")
            + String(hidden) + String(", got ")
            + String(w1sh[0] * w2sh[0]) + String("x")
            + String(w1sh[1] * w2sh[1])
        )
    entries[target] = ArcPointer(Sd3LokrEntry(w1^, w2^, multiplier))


def load_sd3_large_lokr(
    path: String,
    depth: Int,
    hidden: Int,
    multiplier: Float32,
    ctx: DeviceContext,
) raises -> Sd3LokrOverlay:
    """Load the full-factor SimpleTuner SD3 attention LoKr surface.

    This accepts partial attention adapters, but rejects unsupported factored
    LoKr files and model-width mismatches instead of silently dropping them.
    """
    var st = ShardedSafeTensors.open(path)
    var entries = Dict[String, ArcPointer[Sd3LokrEntry]]()

    for i in range(depth):
        var flat = String("lycoris_transformer_blocks_") + String(i) + String("_attn_")
        var dif = String("transformer_blocks.") + String(i) + String(".attn.")
        _add_full_lokr(
            st, entries, flat + String("to_q"), dif + String("to_q"),
            hidden, multiplier, ctx,
        )
        _add_full_lokr(
            st, entries, flat + String("to_k"), dif + String("to_k"),
            hidden, multiplier, ctx,
        )
        _add_full_lokr(
            st, entries, flat + String("to_v"), dif + String("to_v"),
            hidden, multiplier, ctx,
        )
        _add_full_lokr(
            st, entries, flat + String("add_q_proj"), dif + String("add_q_proj"),
            hidden, multiplier, ctx,
        )
        _add_full_lokr(
            st, entries, flat + String("add_k_proj"), dif + String("add_k_proj"),
            hidden, multiplier, ctx,
        )
        _add_full_lokr(
            st, entries, flat + String("add_v_proj"), dif + String("add_v_proj"),
            hidden, multiplier, ctx,
        )
        _add_full_lokr(
            st, entries, flat + String("to_out_0"), dif + String("to_out.0"),
            hidden, multiplier, ctx,
        )
        _add_full_lokr(
            st, entries, flat + String("to_add_out"), dif + String("to_add_out"),
            hidden, multiplier, ctx,
        )

    if len(entries) == 0:
        raise Error(
            "sd3 lokr: no supported SimpleTuner full-factor MMDiT attention"
            " targets were found"
        )
    return Sd3LokrOverlay(entries^)
