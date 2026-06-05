# models/ltx2/weights.mojo — LTX-2 video DiT stack-level base weights.
#
# Loads the FROZEN stack-level (non-block) weights from a safetensors checkpoint
# and provides per-block extractors for offload training. The legacy host-F32
# extractor remains for parity surfaces; production offload uses the BF16
# resident loader below.
#
# LTX-2 key layout (confirmed against checkpoint header, config.mojo):
#   inner_dim D = 4096, num_heads H = 32, head_dim Dh = 128, mlp_hidden FF = 16384
#   in_channels = 128 (VAE latent channel count post-patchify)
#   joint_attention_dim = 3840 (T5 text embed dim for cross-attn, but in the
#     t2v block attn1 is *self*-attn only — text goes through the AdaLN cond path
#     via scale_shift_table). The trainer's text input is the prompt embed for
#     the AdaLN timestep conditioning (see ltx2_block.mojo).
#
# STACK-LEVEL BASE (resident, frozen):
#   patchify_proj.weight  [D, in_channels]   — video token embedding (patch linear)
#   patchify_proj.bias    [D]
#   proj_out.weight       [in_channels, D]   — de-patchify linear (final output)
#   proj_out.bias         [in_channels]
#   adaln_single.*                           — AdaLN conditioning network
#     .emb.timestep_embedder.linear_1.weight [256, 256]
#     .emb.timestep_embedder.linear_1.bias   [256]
#     .emb.timestep_embedder.linear_2.weight [256, 256]
#     .emb.timestep_embedder.linear_2.bias   [256]
#     .linear.weight                          [6*D, 256]   — 6 = 2*(shift,scale,gate)
#     .linear.bias                            [6*D]
#
# PER-BLOCK (streamed, one at a time from TurboPlannedLoader):
#   For block bi (transformer_blocks.{bi}.*):
#     attn1.to_q.weight       [D, D]   ─┐  legacy narrowed LoRA targets
#     attn1.to_k.weight       [D, D]    |
#     attn1.to_v.weight       [D, D]    |
#     attn1.to_out.0.weight   [D, D]   ─┘
#     attn1.to_q.bias         [D]
#     attn1.to_k.bias         [D]
#     attn1.to_v.bias         [D]
#     attn1.to_out.0.bias     [D]
#     attn1.q_norm.weight     [D]    (full inner_dim RMSNorm scale)
#     attn1.k_norm.weight     [D]
#     attn1.to_gate_logits.weight [H, D]  per-head gate
#     attn1.to_gate_logits.bias   [H]
#     ff.net.0.proj.weight    [FF, D]
#     ff.net.0.proj.bias      [FF]
#     ff.net.2.weight         [D, FF]
#     ff.net.2.bias           [D]
#     scale_shift_table       [9, D] F32  AdaLN modvec table (9 rows, not bf16)
#       rows: [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp,
#              (3 unused rows for potential cross-attn in full AV model)]
#       IMPORTANT: scale_shift_table is the LEARNABLE block-specific adaln base;
#       it is FROZEN in LoRA training (only the LoRA weights train). The
#       modvecs for each step are derived by combining this table with the
#       adaln_single conditioning output (sigma-conditioned shift/scale).
#       In the video-only simplified path used here (matching the LoRA parity
#       surface), we read rows 0-5 as the per-block modulation base and add
#       the adaln conditioning on top (see ltx2_stack_lora.mojo).
#
# NOTE ON AV JOINT BLOCK: The full LTX-2.x checkpoint has a joint audio-video
# block (LTX2AVBlockWeights in ltx2_dit.mojo) with audio_attn*, audio_ff,
# audio_to_video_attn, video_to_audio_attn. This weights.mojo targets the legacy
# VIDEO-ONLY LoRA training surface, which only touches attn1.to_{q,k,v,out.0}.
# musubi's production T2V preset targets all AV attention modules; the audio and
# cross-modal branches are SKIPPED by this loader.
#
# Mojo 0.26.x+: def not fn; move-only Tensor; host List[Float32] carriers.

from std.collections import List
from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.offload.block_loader import Block
from serenitymojo.models.ltx2.ltx2_block import LTX2BlockWeights

comptime TArc = ArcPointer[Tensor]


def _load_dev_preserve(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


def _try_key(st: SafeTensors, key: String, fallback: String, ctx: DeviceContext) raises -> Tensor:
    # Try ComfyUI-prefixed key first, then bare key.
    var comfy = String("model.diffusion_model.") + key
    try:
        return _load_dev_preserve(st, comfy, ctx)
    except:
        return _load_dev_preserve(st, key, ctx)


# ── Stack-level base (frozen, resident) ────────────────────────────────────────
# Holds patchify_proj, proj_out, and the adaln_single conditioning network.
# The adaln_single produces per-step sigma-conditioned (shift, scale, gate)
# modulation vectors of shape [6*D] via two linears + silu: the outer loop
# calls adaln_forward_host() once per step.
struct LTX2StackBase(Movable):
    # patchify_proj: video token embedding [D, in_ch]
    var patchify_w: TArc     # [D, in_ch]
    var patchify_b: TArc     # [D]
    # proj_out: de-patchify [out_ch, D]
    var proj_out_w: TArc     # [out_ch, D]
    var proj_out_b: TArc     # [out_ch]
    # adaln_single timestep embedder (maps sigma -> 256-dim embedding)
    var adaln_lin1_w: TArc   # [256, 256]
    var adaln_lin1_b: TArc   # [256]
    var adaln_lin2_w: TArc   # [256, 256]
    var adaln_lin2_b: TArc   # [256]
    # adaln_single linear (256 -> 6*D: the 6-modvec output per block)
    var adaln_out_w: TArc    # [6*D, 256]
    var adaln_out_b: TArc    # [6*D]
    var num_layers: Int
    var D: Int
    var in_ch: Int
    var out_ch: Int

    def __init__(
        out self,
        var patchify_w: TArc, var patchify_b: TArc,
        var proj_out_w: TArc, var proj_out_b: TArc,
        var adaln_lin1_w: TArc, var adaln_lin1_b: TArc,
        var adaln_lin2_w: TArc, var adaln_lin2_b: TArc,
        var adaln_out_w: TArc, var adaln_out_b: TArc,
        num_layers: Int, D: Int, in_ch: Int, out_ch: Int,
    ):
        self.patchify_w = patchify_w^
        self.patchify_b = patchify_b^
        self.proj_out_w = proj_out_w^
        self.proj_out_b = proj_out_b^
        self.adaln_lin1_w = adaln_lin1_w^
        self.adaln_lin1_b = adaln_lin1_b^
        self.adaln_lin2_w = adaln_lin2_w^
        self.adaln_lin2_b = adaln_lin2_b^
        self.adaln_out_w = adaln_out_w^
        self.adaln_out_b = adaln_out_b^
        self.num_layers = num_layers
        self.D = D
        self.in_ch = in_ch
        self.out_ch = out_ch


def load_ltx2_stack_base(
    st: SafeTensors, num_layers: Int, D: Int, in_ch: Int, out_ch: Int,
    ctx: DeviceContext,
) raises -> LTX2StackBase:
    return LTX2StackBase(
        TArc(_try_key(st, String("patchify_proj.weight"), String("patchify_proj.weight"), ctx)),
        TArc(_try_key(st, String("patchify_proj.bias"), String("patchify_proj.bias"), ctx)),
        TArc(_try_key(st, String("proj_out.weight"), String("proj_out.weight"), ctx)),
        TArc(_try_key(st, String("proj_out.bias"), String("proj_out.bias"), ctx)),
        TArc(_try_key(st, String("adaln_single.emb.timestep_embedder.linear_1.weight"),
                      String("adaln_single.emb.timestep_embedder.linear_1.weight"), ctx)),
        TArc(_try_key(st, String("adaln_single.emb.timestep_embedder.linear_1.bias"),
                      String("adaln_single.emb.timestep_embedder.linear_1.bias"), ctx)),
        TArc(_try_key(st, String("adaln_single.emb.timestep_embedder.linear_2.weight"),
                      String("adaln_single.emb.timestep_embedder.linear_2.weight"), ctx)),
        TArc(_try_key(st, String("adaln_single.emb.timestep_embedder.linear_2.bias"),
                      String("adaln_single.emb.timestep_embedder.linear_2.bias"), ctx)),
        TArc(_try_key(st, String("adaln_single.linear.weight"),
                      String("adaln_single.linear.weight"), ctx)),
        TArc(_try_key(st, String("adaln_single.linear.bias"),
                      String("adaln_single.linear.bias"), ctx)),
        num_layers, D, in_ch, out_ch,
    )


# ── Per-block weight extraction from a streamed Block ──────────────────────────
# Returns a flat host-F32 list for each tensor the block forward needs.
# These match the LTX2BlockWeights fields in ltx2_block.mojo.

def _block_f32(block: Block, key: String, ctx: DeviceContext) raises -> List[Float32]:
    if not (key in block):
        raise Error(String("LTX2 block missing tensor: ") + key)
    return cast_tensor(block[key][], STDtype.F32, ctx).to_host(ctx)


# Per-head gate weights (attn1.to_gate_logits.*) only exist when the block was
# trained with apply_gated_attention=True. When the key is ABSENT the model uses
# an UNGATED path (gate==1). The block forward always multiplies by 2*sigmoid(gl)
# so a missing key must map to gl==0  =>  2*sigmoid(0)=1.0 (identity gate). We
# therefore zero-init gate_w/gate_b on a missing key instead of raising.
def _block_f32_or_zeros(block: Block, key: String, n: Int, ctx: DeviceContext) raises -> List[Float32]:
    if not (key in block):
        var z = List[Float32]()
        for _ in range(n):
            z.append(Float32(0.0))
        return z^
    return cast_tensor(block[key][], STDtype.F32, ctx).to_host(ctx)


def _block_bf16_tensor(block: Block, key: String) raises -> TArc:
    if not (key in block):
        raise Error(String("LTX2 block missing tensor: ") + key)
    if block[key][].dtype() != STDtype.BF16:
        raise Error(
            String("LTX2 offload block tensor is not BF16: ")
            + key + String(" dtype=") + block[key][].dtype().name()
        )
    return block[key].copy()


def _zeros(n: Int) -> List[Float32]:
    var z = List[Float32]()
    for _ in range(n):
        z.append(Float32(0.0))
    return z^


def _zeros_bf16_tensor(n: Int, var shape: List[Int], ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_host(_zeros(n), shape^, STDtype.BF16, ctx))


def _block_bf16_or_zeros(
    block: Block, key: String, n: Int, var shape: List[Int], ctx: DeviceContext,
) raises -> TArc:
    if not (key in block):
        return _zeros_bf16_tensor(n, shape^, ctx)
    return _block_bf16_tensor(block, key)


struct LTX2BlockOffloadWeights(Movable):
    var weights: LTX2BlockWeights
    var sst_shift_msa: List[Float32]
    var sst_scale_msa: List[Float32]
    var sst_gate_msa: List[Float32]
    var sst_shift_mlp: List[Float32]
    var sst_scale_mlp: List[Float32]
    var sst_gate_mlp: List[Float32]

    def __init__(
        out self,
        var weights: LTX2BlockWeights,
        var sst_shift_msa: List[Float32], var sst_scale_msa: List[Float32],
        var sst_gate_msa: List[Float32],
        var sst_shift_mlp: List[Float32], var sst_scale_mlp: List[Float32],
        var sst_gate_mlp: List[Float32],
    ):
        self.weights = weights^
        self.sst_shift_msa = sst_shift_msa^
        self.sst_scale_msa = sst_scale_msa^
        self.sst_gate_msa = sst_gate_msa^
        self.sst_shift_mlp = sst_shift_mlp^
        self.sst_scale_mlp = sst_scale_mlp^
        self.sst_gate_mlp = sst_gate_mlp^


def load_ltx2_block_offload_from_block(
    block: Block, bp: String, D: Int, H: Int, ctx: DeviceContext,
) raises -> LTX2BlockOffloadWeights:
    """Load a streamed LTX2 block for the offload training path.

    Checkpoint weights/biases/norms stay resident in BF16 device storage. The
    scale_shift_table rows remain host F32 because they are combined with the
    F32 per-step AdaLN delta before the block casts the resulting modvecs to
    BF16 at its compute boundary.
    """
    var sst = _block_f32(block, bp + String("scale_shift_table"), ctx)
    var shift_msa = List[Float32]()
    var scale_msa = List[Float32]()
    var gate_msa = List[Float32]()
    var shift_mlp = List[Float32]()
    var scale_mlp = List[Float32]()
    var gate_mlp = List[Float32]()
    for c in range(D):
        shift_msa.append(sst[0 * D + c])
        scale_msa.append(sst[1 * D + c])
        gate_msa.append(sst[2 * D + c])
        shift_mlp.append(sst[3 * D + c])
        scale_mlp.append(sst[4 * D + c])
        gate_mlp.append(sst[5 * D + c])

    var weights = LTX2BlockWeights(
        _block_bf16_tensor(block, bp + String("attn1.to_q.weight")),
        _block_bf16_tensor(block, bp + String("attn1.to_q.bias")),
        _block_bf16_tensor(block, bp + String("attn1.to_k.weight")),
        _block_bf16_tensor(block, bp + String("attn1.to_k.bias")),
        _block_bf16_tensor(block, bp + String("attn1.to_v.weight")),
        _block_bf16_tensor(block, bp + String("attn1.to_v.bias")),
        _block_bf16_tensor(block, bp + String("attn1.to_out.0.weight")),
        _block_bf16_tensor(block, bp + String("attn1.to_out.0.bias")),
        _block_bf16_tensor(block, bp + String("attn1.q_norm.weight")),
        _block_bf16_tensor(block, bp + String("attn1.k_norm.weight")),
        _block_bf16_or_zeros(
            block, bp + String("attn1.to_gate_logits.weight"), H * D, [H, D], ctx,
        ),
        _block_bf16_or_zeros(
            block, bp + String("attn1.to_gate_logits.bias"), H, [H], ctx,
        ),
        _block_bf16_tensor(block, bp + String("ff.net.0.proj.weight")),
        _block_bf16_tensor(block, bp + String("ff.net.0.proj.bias")),
        _block_bf16_tensor(block, bp + String("ff.net.2.weight")),
        _block_bf16_tensor(block, bp + String("ff.net.2.bias")),
    )
    return LTX2BlockOffloadWeights(
        weights^,
        shift_msa^, scale_msa^, gate_msa^,
        shift_mlp^, scale_mlp^, gate_mlp^,
    )


struct LTX2BlockWeightsHost(Movable):
    # All fields are F32 host lists matching LTX2BlockWeights ctor args.
    var wq: List[Float32]
    var bq: List[Float32]
    var wk: List[Float32]
    var bk: List[Float32]
    var wv: List[Float32]
    var bv: List[Float32]
    var wo: List[Float32]
    var bo: List[Float32]
    var q_norm: List[Float32]
    var k_norm: List[Float32]
    var gate_w: List[Float32]
    var gate_b: List[Float32]
    var wff0: List[Float32]
    var bff0: List[Float32]
    var wff2: List[Float32]
    var bff2: List[Float32]
    # scale_shift_table rows 0-5: [shift_msa, scale_msa, gate_msa,
    #                               shift_mlp, scale_mlp, gate_mlp] each [D]
    var sst_shift_msa: List[Float32]
    var sst_scale_msa: List[Float32]
    var sst_gate_msa: List[Float32]
    var sst_shift_mlp: List[Float32]
    var sst_scale_mlp: List[Float32]
    var sst_gate_mlp: List[Float32]

    def __init__(
        out self,
        var wq: List[Float32], var bq: List[Float32],
        var wk: List[Float32], var bk: List[Float32],
        var wv: List[Float32], var bv: List[Float32],
        var wo: List[Float32], var bo: List[Float32],
        var q_norm: List[Float32], var k_norm: List[Float32],
        var gate_w: List[Float32], var gate_b: List[Float32],
        var wff0: List[Float32], var bff0: List[Float32],
        var wff2: List[Float32], var bff2: List[Float32],
        var sst_shift_msa: List[Float32], var sst_scale_msa: List[Float32],
        var sst_gate_msa: List[Float32],
        var sst_shift_mlp: List[Float32], var sst_scale_mlp: List[Float32],
        var sst_gate_mlp: List[Float32],
    ):
        self.wq = wq^; self.bq = bq^
        self.wk = wk^; self.bk = bk^
        self.wv = wv^; self.bv = bv^
        self.wo = wo^; self.bo = bo^
        self.q_norm = q_norm^; self.k_norm = k_norm^
        self.gate_w = gate_w^; self.gate_b = gate_b^
        self.wff0 = wff0^; self.bff0 = bff0^
        self.wff2 = wff2^; self.bff2 = bff2^
        self.sst_shift_msa = sst_shift_msa^
        self.sst_scale_msa = sst_scale_msa^
        self.sst_gate_msa = sst_gate_msa^
        self.sst_shift_mlp = sst_shift_mlp^
        self.sst_scale_mlp = sst_scale_mlp^
        self.sst_gate_mlp = sst_gate_mlp^


def load_ltx2_block_weights_from_block(
    block: Block, bp: String, D: Int, H: Int, ctx: DeviceContext,
) raises -> LTX2BlockWeightsHost:
    # `bp` is the dot-terminated block prefix the TurboPlannedLoader keyed this
    # Block by (e.g. "model.diffusion_model.transformer_blocks.0."). Every Block
    # dict key is a FULL tensor name, so we prepend `bp` to each relative key.
    # Extract scale_shift_table [9, D] and slice rows 0-5.
    var sst = _block_f32(block, bp + String("scale_shift_table"), ctx)
    # rows: shift_msa=0, scale_msa=1, gate_msa=2, shift_mlp=3, scale_mlp=4, gate_mlp=5
    var shift_msa = List[Float32]()
    var scale_msa = List[Float32]()
    var gate_msa = List[Float32]()
    var shift_mlp = List[Float32]()
    var scale_mlp = List[Float32]()
    var gate_mlp = List[Float32]()
    for c in range(D):
        shift_msa.append(sst[0 * D + c])
        scale_msa.append(sst[1 * D + c])
        gate_msa.append(sst[2 * D + c])
        shift_mlp.append(sst[3 * D + c])
        scale_mlp.append(sst[4 * D + c])
        gate_mlp.append(sst[5 * D + c])

    return LTX2BlockWeightsHost(
        _block_f32(block, bp + String("attn1.to_q.weight"), ctx),
        _block_f32(block, bp + String("attn1.to_q.bias"), ctx),
        _block_f32(block, bp + String("attn1.to_k.weight"), ctx),
        _block_f32(block, bp + String("attn1.to_k.bias"), ctx),
        _block_f32(block, bp + String("attn1.to_v.weight"), ctx),
        _block_f32(block, bp + String("attn1.to_v.bias"), ctx),
        _block_f32(block, bp + String("attn1.to_out.0.weight"), ctx),
        _block_f32(block, bp + String("attn1.to_out.0.bias"), ctx),
        _block_f32(block, bp + String("attn1.q_norm.weight"), ctx),
        _block_f32(block, bp + String("attn1.k_norm.weight"), ctx),
        _block_f32_or_zeros(block, bp + String("attn1.to_gate_logits.weight"), H * D, ctx),
        _block_f32_or_zeros(block, bp + String("attn1.to_gate_logits.bias"), H, ctx),
        _block_f32(block, bp + String("ff.net.0.proj.weight"), ctx),
        _block_f32(block, bp + String("ff.net.0.proj.bias"), ctx),
        _block_f32(block, bp + String("ff.net.2.weight"), ctx),
        _block_f32(block, bp + String("ff.net.2.bias"), ctx),
        shift_msa^, scale_msa^, gate_msa^,
        shift_mlp^, scale_mlp^, gate_mlp^,
    )
