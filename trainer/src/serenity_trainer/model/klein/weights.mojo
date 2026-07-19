# models/klein/weights.mojo — G1: real safetensors -> training weight structs.
#
# Loads Klein double-block weights from a real .safetensors into the
# DoubleBlockWeights the verified double_block_forward/backward consume. The
# inference path (models/dit/klein_dit.mojo) already reads these exact keys; this
# is the cross-pollination into the TRAINING weight structs (host List[Float32]).
#
# Key layout (per double block bi), from klein_dit.mojo:112-123 — 12 tensors:
#   double_blocks.{bi}.{img,txt}_attn.qkv.weight            -> StreamWeights.wqkv
#   double_blocks.{bi}.{img,txt}_attn.proj.weight           -> .wproj
#   double_blocks.{bi}.{img,txt}_attn.norm.query_norm.scale -> .q_norm
#   double_blocks.{bi}.{img,txt}_attn.norm.key_norm.scale   -> .k_norm
#   double_blocks.{bi}.{img,txt}_mlp.0.weight               -> .wgu  (fused gate+up)
#   double_blocks.{bi}.{img,txt}_mlp.2.weight               -> .wd
#
# Key layout (per single block bi), from klein_dit.mojo `_single_block` — 4 tensors:
#   single_blocks.{bi}.linear1.weight            -> SingleBlockWeights.w1 [3D+2F, D]
#   single_blocks.{bi}.linear2.weight            -> .w2 [D, D+F]
#   single_blocks.{bi}.norm.query_norm.scale     -> .q_norm [Dh]
#   single_blocks.{bi}.norm.key_norm.scale       -> .k_norm [Dh]
# These feed the verified single_block_forward/backward (models/klein/single_block.mojo).

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import silu
from serenitymojo.ops.embeddings import t_embedder
from serenitymojo.ops.tensor_algebra import reshape_owned, slice, add as _add
from serenity_trainer.model.klein.double_block import (
    StreamWeights, DoubleBlockWeights, ModVecs, ModVecsDevice,
)
from serenity_trainer.model.klein.single_block import (
    SingleBlockWeights, SingleModVecs, SingleModVecsDevice,
)
from serenity_trainer.model.klein.klein_stack import KleinStackBase


struct KleinStepModWeights(Movable):
    """Frozen timestep/modulation weights reused across every training step.

    Guidance is gated on the LOADED CHECKPOINT'S guidance_embeds config value
    (BaseFlux2Setup.py:132 model.transformer.config.guidance_embeds). The flagship
    FLUX.2-klein-base-9B checkpoint has guidance_embeds=FALSE (verified vs
    transformer/config.json) and carries NO guidance_in.* keys, so g_in/g_out are
    None and the guidance branch is a no-op. Only guidance-distilled variants
    (guidance_embeds=True, guidance_in.* present) take it. The combined
    timestep+guidance embedding (Flux2TimestepGuidanceEmbeddings.forward,
    transformer_flux2.py:1004-1014) is:
        timesteps_emb = timestep_embedder(time_proj(timestep))
        guidance_emb  = guidance_embedder(time_proj(guidance))   # only if guidance
        temb          = timesteps_emb + guidance_emb
    The downstream modulation then applies silu(temb) (Flux2Modulation.forward,
    transformer_flux2.py:1025-1027). So vec_silu = silu(t_emb + g_emb).

    g_in/g_out are the guidance embedder MLP weights (original-format keys
    guidance_in.in_layer.weight / guidance_in.out_layer.weight; Flux2Model.py:43-46
    maps diffusers time_guidance_embed.guidance_embedder -> guidance_in). They are
    Optional: a non-guidance-distilled checkpoint (guidance_embeds=False) carries no
    guidance_in.* keys and leaves them None (guidance branch becomes a no-op, exactly
    matching transformer_flux2.py:1008 `guidance is not None and guidance_embedder is
    not None`).
    """

    var t_in: Tensor
    var t_out: Tensor
    var g_in: Optional[Tensor]    # guidance_in.in_layer.weight  (None if not guidance-distilled)
    var g_out: Optional[Tensor]   # guidance_in.out_layer.weight (None if not guidance-distilled)
    var img_mod: Tensor
    var txt_mod: Tensor
    var single_mod: Tensor
    var final_mod: Tensor   # final_layer.adaLN_modulation.1.weight [2D, D]

    def __init__(
        out self,
        var t_in: Tensor,
        var t_out: Tensor,
        var g_in: Optional[Tensor],
        var g_out: Optional[Tensor],
        var img_mod: Tensor,
        var txt_mod: Tensor,
        var single_mod: Tensor,
        var final_mod: Tensor,
    ):
        self.t_in = t_in^
        self.t_out = t_out^
        self.g_in = g_in^
        self.g_out = g_out^
        self.img_mod = img_mod^
        self.txt_mod = txt_mod^
        self.single_mod = single_mod^
        self.final_mod = final_mod^

    def has_guidance(self) -> Bool:
        return Bool(self.g_in) and Bool(self.g_out)


# Read one named tensor from the safetensors as a host List[Float32]. This is
# for small host-side reference/modulation helpers only. Persistent model weights
# must use `_load_tensor` below so BF16 checkpoint storage remains BF16 on device.
def _load_host_f32(st: SafeTensors, name: String, ctx: DeviceContext) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    var t32 = cast_tensor(t, STDtype.F32, ctx)
    return t32.to_host(ctx)


# Dim 0 of a named tensor's stored shape (for deriving D/F/Dh from the weights).
def _dim0(st: SafeTensors, name: String) raises -> Int:
    var info = st.tensor_info(name)
    return Int(info.shape[0])


# A2: StreamWeights now owns resident device tensors (TArc). Load directly from
# the safetensors view so checkpoint BF16 storage stays BF16 at tensor boundaries:
#   wqkv [3D, D]  -> D  = dim1(wqkv) = dim0(wqkv)//3 ; here from wproj [D,D] dim0.
#   wgu  [2F, D]  -> F  = dim0(wgu)//2.
#   q_norm [Dh]   -> Dh = dim0(q_norm).
def _load_stream(
    st: SafeTensors, dp: String, stream: String, ctx: DeviceContext
) raises -> StreamWeights:
    var ap = dp + String(".") + stream + String("_attn")
    var mp = dp + String(".") + stream + String("_mlp")
    return StreamWeights(
        ArcPointer[Tensor](_load_tensor(st, ap + String(".qkv.weight"), ctx)),               # wqkv
        ArcPointer[Tensor](_load_tensor(st, ap + String(".proj.weight"), ctx)),              # wproj
        ArcPointer[Tensor](_load_tensor(st, mp + String(".0.weight"), ctx)),                 # wgu
        ArcPointer[Tensor](_load_tensor(st, mp + String(".2.weight"), ctx)),                 # wd
        ArcPointer[Tensor](_load_tensor(st, ap + String(".norm.query_norm.scale"), ctx)),    # q_norm
        ArcPointer[Tensor](_load_tensor(st, ap + String(".norm.key_norm.scale"), ctx)),      # k_norm
    )


# Load double block `block_idx`'s real weights into DoubleBlockWeights.
def load_double_block_weights(
    st: SafeTensors, block_idx: Int, ctx: DeviceContext
) raises -> DoubleBlockWeights:
    var dp = String("double_blocks.") + String(block_idx)
    return DoubleBlockWeights(
        _load_stream(st, dp, String("img"), ctx),
        _load_stream(st, dp, String("txt"), ctx),
    )


# Load single block `block_idx`'s real weights into SingleBlockWeights.
# Keys: single_blocks.{bi}.linear1.weight, .linear2.weight,
#       .norm.query_norm.scale, .norm.key_norm.scale.
# A2: SingleBlockWeights owns resident device tensors. Load directly from the
# safetensors view so checkpoint BF16 storage stays BF16 on device.
# Dims from stored shapes: w2 [D, D+F] -> D = dim0(w2); F = dim1(w2) - D where
# dim1(w2) = dim0(w1)? no: derive D from w2 dim0, F from (w1 dim0 - 3D)/2... use
# the explicit shapes: w1 [3D+2F, D] (dim1=D), w2 [D, D+F] (dim0=D). So
#   D  = dim0(w2)
#   F  = (dim0(w1) - 3*D) // 2
#   Dh = dim0(q_norm)
def load_single_block_weights(
    st: SafeTensors, block_idx: Int, ctx: DeviceContext, keep_w2: Bool = True
) raises -> SingleBlockWeights:
    var sp = String("single_blocks.") + String(block_idx)
    var D = _dim0(st, sp + String(".linear2.weight"))         # w2 [D, D+F] -> D
    var F = (_dim0(st, sp + String(".linear1.weight")) - 3 * D) // 2  # w1 [3D+2F, D]
    return SingleBlockWeights(
        ArcPointer[Tensor](_load_tensor(st, sp + String(".linear1.weight"), ctx)),           # w1
        ArcPointer[Tensor](_load_tensor(st, sp + String(".linear2.weight"), ctx)),           # w2
        ArcPointer[Tensor](_load_tensor(st, sp + String(".norm.query_norm.scale"), ctx)),    # q_norm
        ArcPointer[Tensor](_load_tensor(st, sp + String(".norm.key_norm.scale"), ctx)),      # k_norm
        D, F, ctx, keep_w2,
    )


# ── shared base weights for the full stack (input proj + final layer) ─────────
# Loads img_in.weight, txt_in.weight, final_layer.linear.weight, and the two
# final-layer adaLN chunks (shift=chunk0, scale=chunk1) of
# final_layer.adaLN_modulation.1.weight applied to vec_silu. `d` is the inner dim.
# vec_silu [1,D] is the (single-timestep) modulation feature (see build_klein_modvecs).
# Dim 1 of a named tensor's stored shape.
def _dim1(st: SafeTensors, name: String) raises -> Int:
    var info = st.tensor_info(name)
    return Int(info.shape[1])


# A2: KleinStackBase owns resident device tensors. The big input/output
# projections load directly from safetensors so checkpoint BF16 storage remains
# BF16. The final adaLN shift/scale are still computed on host here (small [D],
# frozen once with the seed sigma) then uploaded once.
def load_klein_stack_base(
    st: SafeTensors, vec_silu: List[Float32], d: Int, ctx: DeviceContext
) raises -> KleinStackBase:
    var in_ch = _dim1(st, String("img_in.weight"))     # img_in [D, in_ch]
    var txt_ch = _dim1(st, String("txt_in.weight"))    # txt_in [D, txt_ch]
    var out_ch = _dim0(st, String("final_layer.linear.weight"))  # final_lin [out_ch, D]
    # final adaLN: linear(vec_silu, final_mod_w) -> [1, 2D]; chunk 0=shift, 1=scale.
    var final_mod_w = _load_host_f32(st, String("final_layer.adaLN_modulation.1.weight"), ctx)
    var final_mod = _linear_row(vec_silu, final_mod_w, d, 2 * d, ctx)   # [2D]
    var final_shift = _chunk(final_mod, 0, d)
    var final_scale = _chunk(final_mod, 1, d)
    var img_in = _load_tensor(st, String("img_in.weight"), ctx)
    var txt_in = _load_tensor(st, String("txt_in.weight"), ctx)
    var final_lin = _load_tensor(st, String("final_layer.linear.weight"), ctx)
    var final_dtype = final_lin.dtype()
    return KleinStackBase(
        ArcPointer[Tensor](img_in^),
        ArcPointer[Tensor](txt_in^),
        ArcPointer[Tensor](final_lin^),
        ArcPointer[Tensor](Tensor.from_host(final_shift^, [d], final_dtype, ctx)),
        ArcPointer[Tensor](Tensor.from_host(final_scale^, [d], final_dtype, ctx)),
    )


# linear of a single [in_dim] row by a [out_dim, in_dim] weight -> [out_dim].
def _linear_row(
    x: List[Float32], w: List[Float32], in_dim: Int, out_dim: Int, ctx: DeviceContext
) raises -> List[Float32]:
    var no_bias = Optional[Tensor](None)
    return linear(
        Tensor.from_host(x, [1, in_dim], STDtype.F32, ctx),
        Tensor.from_host(w, [out_dim, in_dim], STDtype.F32, ctx),
        no_bias^, ctx,
    ).to_host(ctx)


# extract chunk `idx` of width `d` from a flat [k*d] list.
def _chunk(src: List[Float32], idx: Int, d: Int) -> List[Float32]:
    var o = List[Float32]()
    var base = idx * d
    for i in range(d):
        o.append(src[base + i])
    return o^


# Build the SHARED modulation feature vec_silu = silu(t_embedder(timestep, ...)).
# timestep: [1] (single sample). Returns vec_silu [1, D] as a host list.
def build_klein_vec_silu(
    st: SafeTensors, timestep: Tensor, timestep_dim: Int, d: Int, ctx: DeviceContext
) raises -> List[Float32]:
    var t_in = _load_tensor(st, String("time_in.in_layer.weight"), ctx)
    var t_out = _load_tensor(st, String("time_in.out_layer.weight"), ctx)
    var vec = t_embedder(
        timestep, timestep_dim, t_in, Optional[Tensor](None), t_out, Optional[Tensor](None), ctx
    )
    var vec_silu = silu(vec, ctx)
    return vec_silu.to_host(ctx)


# Build the shared double/single modulation vectors from vec_silu and the real
# modulation linears: mod = linear(vec_silu, mod_w); chunk into ModVecs/SingleModVecs.
def build_klein_double_modvecs(
    st: SafeTensors, vec_silu: List[Float32], stream: String, d: Int, ctx: DeviceContext
) raises -> ModVecs:
    var mod_w = _load_host_f32(
        st, String("double_stream_modulation_") + stream + String(".lin.weight"), ctx
    )
    var mod = _linear_row(vec_silu, mod_w, d, 6 * d, ctx)   # [6D]
    return ModVecs(
        _chunk(mod, 0, d), _chunk(mod, 1, d), _chunk(mod, 2, d),
        _chunk(mod, 3, d), _chunk(mod, 4, d), _chunk(mod, 5, d),
    )


def build_klein_single_modvecs(
    st: SafeTensors, vec_silu: List[Float32], d: Int, ctx: DeviceContext
) raises -> SingleModVecs:
    var mod_w = _load_host_f32(st, String("single_stream_modulation.lin.weight"), ctx)
    var mod = _linear_row(vec_silu, mod_w, d, 3 * d, ctx)   # [3D]
    return SingleModVecs(_chunk(mod, 0, d), _chunk(mod, 1, d), _chunk(mod, 2, d))


def load_klein_step_mod_weights(
    st: SafeTensors, d: Int, ctx: DeviceContext
) raises -> KleinStepModWeights:
    """Load frozen per-step modulation weights once for the timed loop.

    Timestep MLP weights stay in their checkpoint dtype (BF16). The downstream
    modulation weights are promoted to F32 once so the math matches the legacy
    host `_linear_row` path that used F32 lists.
    """
    # Guidance embedder: load guidance_in.in_layer / guidance_in.out_layer IF
    # present. Absent on guidance_embeds=False checkpoints (incl. the flagship
    # FLUX.2-klein-base-9B, guidance_embeds=False per transformer/config.json),
    # which leave these Optional -> None -> guidance branch is a no-op. Presence of
    # these keys is the structural equivalent of config.guidance_embeds=True.
    var g_in = _try_load_tensor(st, String("guidance_in.in_layer.weight"), ctx)
    var g_out = _try_load_tensor(st, String("guidance_in.out_layer.weight"), ctx)
    return KleinStepModWeights(
        _load_tensor(st, String("time_in.in_layer.weight"), ctx),
        _load_tensor(st, String("time_in.out_layer.weight"), ctx),
        g_in^,
        g_out^,
        _load_tensor(st, String("double_stream_modulation_img.lin.weight"), ctx),
        _load_tensor(st, String("double_stream_modulation_txt.lin.weight"), ctx),
        _load_tensor(st, String("single_stream_modulation.lin.weight"), ctx),
        _load_tensor(st, String("final_layer.adaLN_modulation.1.weight"), ctx),
    )


def build_klein_vec_silu_device(
    weights: KleinStepModWeights,
    timestep: Tensor,
    timestep_dim: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var vec = t_embedder(
        timestep,
        timestep_dim,
        weights.t_in,
        Optional[Tensor](None),
        weights.t_out,
        Optional[Tensor](None),
        ctx,
    )
    var vec_silu = silu(vec, ctx)
    return vec_silu^


# ── guidance-aware combined timestep+guidance vec_silu (guidance_embeds variants) ─
# Flux2TimestepGuidanceEmbeddings.forward (transformer_flux2.py:1004-1014):
#   timesteps_emb = timestep_embedder(time_proj(timestep))
#   if guidance is not None and guidance_embedder is not None:
#       guidance_emb     = guidance_embedder(time_proj(guidance))
#       time_guidance_emb = timesteps_emb + guidance_emb
#   else: time_guidance_emb = timesteps_emb
# The downstream Flux2Modulation applies silu(temb) (transformer_flux2.py:1025-1027),
# so vec_silu = silu(t_emb + g_emb).  Both `timestep` and `guidance` are the INTEGER
# values the t_embedder/guidance_embedder see (diffusers re-scales timestep/guidance
# ×1000 internally, transformer_flux2.py:1231-1234; predict()/sampler feed t and
# guidance_scale already in that integer domain). When `guidance` is None (or the
# checkpoint carries no guidance embedder) this reduces to the timestep-only path.
def build_klein_vec_silu_guidance_device(
    weights: KleinStepModWeights,
    timestep: Tensor,
    guidance: Optional[Tensor],
    timestep_dim: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var t_emb = t_embedder(
        timestep,
        timestep_dim,
        weights.t_in,
        Optional[Tensor](None),
        weights.t_out,
        Optional[Tensor](None),
        ctx,
    )
    # guidance branch: only when a guidance value is provided AND the checkpoint
    # carries a guidance embedder (transformer_flux2.py:1008 conjunction).
    if guidance and weights.has_guidance():
        var g_emb = t_embedder(
            guidance.value(),
            timestep_dim,
            weights.g_in.value(),
            Optional[Tensor](None),
            weights.g_out.value(),
            Optional[Tensor](None),
            ctx,
        )
        var temb = _add(t_emb, g_emb, ctx)   # timesteps_emb + guidance_emb (1011)
        var vec_silu_g = silu(temb, ctx)
        return vec_silu_g^
    var vec_silu = silu(t_emb, ctx)
    return vec_silu^


def _modvec_from_device(mod: Tensor, d: Int, ctx: DeviceContext) raises -> ModVecs:
    var host = mod.to_host(ctx)
    return ModVecs(
        _chunk(host, 0, d), _chunk(host, 1, d), _chunk(host, 2, d),
        _chunk(host, 3, d), _chunk(host, 4, d), _chunk(host, 5, d),
    )


def _single_modvec_from_device(
    mod: Tensor, d: Int, ctx: DeviceContext
) raises -> SingleModVecs:
    var host = mod.to_host(ctx)
    return SingleModVecs(_chunk(host, 0, d), _chunk(host, 1, d), _chunk(host, 2, d))


def _chunk_tensor_1d(
    x: Tensor, start_chunk: Int, d: Int, ctx: DeviceContext
) raises -> Tensor:
    var chunk2d = slice(x, 1, start_chunk * d, d, ctx)
    return reshape_owned(chunk2d^, [d])


def _modvec_device_from_tensor(
    mod: Tensor, d: Int, ctx: DeviceContext
) raises -> ModVecsDevice:
    var shift1 = _chunk_tensor_1d(mod, 0, d, ctx)
    var scale1 = _chunk_tensor_1d(mod, 1, d, ctx)
    var gate1 = _chunk_tensor_1d(mod, 2, d, ctx)
    var shift2 = _chunk_tensor_1d(mod, 3, d, ctx)
    var scale2 = _chunk_tensor_1d(mod, 4, d, ctx)
    var gate2 = _chunk_tensor_1d(mod, 5, d, ctx)
    return ModVecsDevice(
        ArcPointer[Tensor](shift1^),
        ArcPointer[Tensor](scale1^),
        ArcPointer[Tensor](gate1^),
        ArcPointer[Tensor](shift2^),
        ArcPointer[Tensor](scale2^),
        ArcPointer[Tensor](gate2^),
    )


def _single_modvec_device_from_tensor(
    mod: Tensor, d: Int, ctx: DeviceContext
) raises -> SingleModVecsDevice:
    var shift = _chunk_tensor_1d(mod, 0, d, ctx)
    var scale = _chunk_tensor_1d(mod, 1, d, ctx)
    var gate = _chunk_tensor_1d(mod, 2, d, ctx)
    return SingleModVecsDevice(
        ArcPointer[Tensor](shift^),
        ArcPointer[Tensor](scale^),
        ArcPointer[Tensor](gate^),
    )


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ TIMESTEP TRAP — READ BEFORE TOUCHING THE timestep_value ARG.                ║
# ║                                                                            ║
# ║ `timestep_value` is the INTEGER timestep t, fed DIRECTLY to t_embedder      ║
# ║ with NO *1000.  Serenity feeds the transformer `timestep / 1000`          ║
# ║ (BaseFlux2Setup.py:144) and diffusers Flux2 multiplies it back ×1000        ║
# ║ internally (transformer_flux2.py:1231) → the t_embedder sees the INTEGER    ║
# ║ timestep t. Serenity's σ(t) = (t+1)/1000 (FlowMatchingMixin.py:24), so    ║
# ║ `sigma*1000 = t + 1` would be OFF BY ONE — do NOT pass sigma*1000. If you   ║
# ║ only have a σ, convert with timestep_from_sigma(σ) (BaseFlux2Setup.mojo).   ║
# ║                                                                            ║
# ║ build_klein_step_mods_device_cached below is the WIRED path used by         ║
# ║ Flux2LoRASetup (predict + backward_lora); it also threads the guidance      ║
# ║ value, active only for guidance_embeds=True checkpoints (the guidance branch ║
# ║ is gated on weights.has_guidance(); transformer_flux2.py:1004-1014). The     ║
# ║ flagship FLUX.2-klein-base-9B is guidance_embeds=False ⇒ guidance no-op.     ║
# ║ build_klein_step_mods_cached (host ModVecs variant) is timestep-only and    ║
# ║ retained for non-guidance callers.                                          ║
# ╚════════════════════════════════════════════════════════════════════════════╝
def build_klein_step_mods_cached(
    weights: KleinStepModWeights,
    timestep_value: Float32,   # INTEGER timestep t (NOT sigma); fed directly to t_embedder.
    timestep_dim: Int,
    d: Int,
    ctx: DeviceContext,
) raises -> Tuple[ModVecs, ModVecs, SingleModVecs]:
    var tvals = List[Float32]()
    tvals.append(timestep_value)   # NO *1000 — t_embedder sees the integer timestep t.
    var tsh = List[Int]()
    tsh.append(1)
    var ts = Tensor.from_host(tvals, tsh^, STDtype.F32, ctx)
    var vec_silu = build_klein_vec_silu_device(weights, ts, timestep_dim, ctx)

    var no_bias_img = Optional[Tensor](None)
    var img_mod = linear(vec_silu, weights.img_mod, no_bias_img^, ctx)
    var no_bias_txt = Optional[Tensor](None)
    var txt_mod = linear(vec_silu, weights.txt_mod, no_bias_txt^, ctx)
    var no_bias_single = Optional[Tensor](None)
    var single_mod = linear(vec_silu, weights.single_mod, no_bias_single^, ctx)
    return (
        _modvec_from_device(img_mod, d, ctx),
        _modvec_from_device(txt_mod, d, ctx),
        _single_modvec_from_device(single_mod, d, ctx),
    )


# Build the per-step device modulation vectors (img/txt/single) + per-step final
# adaLN shift/scale from the INTEGER timestep t and the (optional) guidance value.
#
# timestep_value: the INTEGER timestep t (NOT sigma); fed directly to t_embedder
#   (diffusers re-scales timestep/1000 ×1000, transformer_flux2.py:1231 — the
#   embedder sees t). guidance_value: the INTEGER guidance value the guidance
#   embedder sees. Serenity feeds the transformer guidance=cfg_scale
#   (BaseFlux2Setup.py:133); diffusers multiplies guidance ×1000 internally
#   (transformer_flux2.py:1234) ⇒ guidance_value = guidance_scale * 1000. Pass None
#   for non-guidance-distilled checkpoints (guidance_embeds=False).
def build_klein_step_mods_device_cached(
    weights: KleinStepModWeights,
    timestep_value: Float32,   # INTEGER timestep t (NOT sigma); fed directly to t_embedder.
    guidance_value: Optional[Float32],   # INTEGER guidance value (= guidance_scale*1000) or None.
    timestep_dim: Int,
    d: Int,
    ctx: DeviceContext,
) raises -> Tuple[ModVecsDevice, ModVecsDevice, SingleModVecsDevice, ArcPointer[Tensor], ArcPointer[Tensor]]:
    var tvals = List[Float32]()
    tvals.append(timestep_value)   # NO *1000 — t_embedder sees the integer timestep t.
    var tsh = List[Int]()
    tsh.append(1)
    var ts = Tensor.from_host(tvals, tsh^, STDtype.F32, ctx)
    # build the optional guidance tensor [1] (integer guidance value).
    var g_opt = Optional[Tensor](None)
    if guidance_value:
        var gvals = List[Float32]()
        gvals.append(guidance_value.value())
        var gsh = List[Int]()
        gsh.append(1)
        g_opt = Optional[Tensor](Tensor.from_host(gvals, gsh^, STDtype.F32, ctx))
    var vec_silu = build_klein_vec_silu_guidance_device(
        weights, ts, g_opt, timestep_dim, ctx
    )

    var no_bias_img = Optional[Tensor](None)
    var img_mod = linear(vec_silu, weights.img_mod, no_bias_img^, ctx)
    var no_bias_txt = Optional[Tensor](None)
    var txt_mod = linear(vec_silu, weights.txt_mod, no_bias_txt^, ctx)
    var no_bias_single = Optional[Tensor](None)
    var single_mod = linear(vec_silu, weights.single_mod, no_bias_single^, ctx)
    # per-step final-layer adaLN mod (FIX: was static sigma=0.5). final_mod [1,2D];
    # chunk0=shift, chunk1=scale (matches the static load_klein_stack_base path).
    var no_bias_final = Optional[Tensor](None)
    var final_mod = linear(vec_silu, weights.final_mod, no_bias_final^, ctx)
    var final_shift = _chunk_tensor_1d(final_mod, 0, d, ctx)   # [d] cols 0:d
    var final_scale = _chunk_tensor_1d(final_mod, 1, d, ctx)   # [d] cols d:2d
    return (
        _modvec_device_from_tensor(img_mod, d, ctx),
        _modvec_device_from_tensor(txt_mod, d, ctx),
        _single_modvec_device_from_tensor(single_mod, d, ctx),
        ArcPointer[Tensor](final_shift^),
        ArcPointer[Tensor](final_scale^),
    )


# load a named tensor as a device Tensor (BF16 stored) without casting to host.
def _load_tensor(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


# True if a tensor with the given name exists in the safetensors.
def _has_tensor(st: SafeTensors, name: String) -> Bool:
    var ns = st.names()
    for n in ns:
        if n == name:
            return True
    return False


# Optionally load a named device Tensor; returns None if the key is absent
# (used for the guidance embedder, which only exists on guidance-distilled Klein
# checkpoints; the flagship FLUX.2-klein-base-9B is guidance_embeds=False ⇒ absent).
def _try_load_tensor(
    st: SafeTensors, name: String, ctx: DeviceContext
) raises -> Optional[Tensor]:
    if not _has_tensor(st, name):
        return Optional[Tensor](None)
    return Optional[Tensor](_load_tensor(st, name, ctx))
