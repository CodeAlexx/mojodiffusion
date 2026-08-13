# serenitymojo/models/wan22/weights.mojo — Wan2.2 checkpoint loader.
#
# Loads RESIDENT (non-block) base tensors from the wan2.2_t2v_low_noise_14b
# checkpoint for training. Block tensors are streamed one-at-a-time by the
# TurboPlannedLoader; this module loads everything that stays GPU-resident
# across all training steps.
#
# CHECKPOINT KEY LAYOUT (confirmed from header):
#   patch_embedding.weight  [5120, 16, 1, 2, 2]   F16  (reshaped to [5120,64])
#   patch_embedding.bias    [5120]                 F16
#   text_embedding.0.weight [5120, 4096]           F16
#   text_embedding.0.bias   [5120]                 F16
#   text_embedding.2.weight [5120, 5120]           F16
#   text_embedding.2.bias   [5120]                 F16
#   time_embedding.0.weight [5120, 256]            F16
#   time_embedding.0.bias   [5120]                 F16
#   time_embedding.2.weight [5120, 5120]           F16
#   time_embedding.2.bias   [5120]                 F16
#   time_projection.1.weight [30720, 5120]         F16
#   time_projection.1.bias   [30720]               F16
#   head.head.weight         [64, 5120]            F16
#   head.head.bias           [64]                  F16
#   head.modulation          [1, 2, 5120]          F16
#
# Per-block keys (streamed, NOT loaded here):
#   blocks.{i}.{self_attn,cross_attn}.{q,k,v,o}.{weight,bias}
#   blocks.{i}.{self_attn,cross_attn}.norm_{q,k}.weight
#   blocks.{i}.norm3.{weight,bias}
#   blocks.{i}.ffn.{0,2}.{weight,bias}
#   blocks.{i}.modulation    [1, 6, 5120]
#
# Mojo 0.26.x+: def not fn; move-only Tensor; ArcPointer[Tensor] for shared
# resident tensors.

from std.collections import List, Optional
from std.memory import ArcPointer
from max.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.tensor_algebra import reshape

from serenitymojo.models.wan22.wan22_stack_lora import Wan22StackBase


comptime TArc = ArcPointer[Tensor]


# ── Auto-detect the checkpoint key prefix ────────────────────────────────────
# Wan2.2 checkpoints use BARE keys ("patch_embedding.weight"). Wan2.1 ComfyUI-
# repackaged files (wan2.1_t2v_*_fp16.safetensors) prefix EVERY key with
# "model.diffusion_model.". We probe the header for the patch_embedding sentinel
# and return the detected prefix ("" for the bare 2.2 layout,
# "model.diffusion_model." for the ComfyUI 2.1 layout). The SAME prefix must be
# threaded into build_wan22_block_plan so block streaming matches the prefixed
# block tensors. Default/absence of the prefix leaves every 2.2 path byte-
# identical.
def detect_wan22_prefix(st: SafeTensors) raises -> String:
    if String("patch_embedding.weight") in st.tensors:
        return String("")
    if String("model.diffusion_model.patch_embedding.weight") in st.tensors:
        return String("model.diffusion_model.")
    raise Error(
        String("detect_wan22_prefix: checkpoint header has neither ")
        + "'patch_embedding.weight' (bare Wan2.2) nor "
        + "'model.diffusion_model.patch_embedding.weight' (ComfyUI Wan2.1)"
    )


# ── Load a tensor from safetensors preserving checkpoint storage dtype ───────
def _load_dev_preserve(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


# ── Load the RESIDENT (frozen-base) tensors: embeddings, proj, head ──────────
# patch_embedding.weight is a Conv3d kernel [dim, in_ch, pf, ph, pw]. For the
# patch-embed linear we flatten it to [dim, in_ch*pf*ph*pw] (a byte no-op —
# contiguous row-major). in_ch is 16 for T2V-A14B (-> packed 64) and 36 for
# I2V-A14B (noisy_latent 16 + y 20 -> packed 144). We read the ACTUAL stored
# shape from the header and compute the packed width at runtime, so the same
# loader serves both variants (all other keys are identical).
def load_wan22_stack_base(
    st: SafeTensors, ctx: DeviceContext, prefix: String = String(""),
) raises -> Wan22StackBase:
    # `prefix` is "" for bare Wan2.2 keys and "model.diffusion_model." for the
    # ComfyUI Wan2.1 layout (see detect_wan22_prefix). It is prepended to EVERY
    # resident key so both checkpoint layouts load through the same code.
    var pfx = prefix
    # patch_embedding: reshape [dim, in_ch, pf, ph, pw] -> [dim, in_ch*pf*ph*pw]
    var pe_info = st.tensor_info(pfx + "patch_embedding.weight")
    var pe_out = pe_info.shape[0]                 # dim (5120)
    var pe_packed = 1
    for i in range(1, len(pe_info.shape)):        # in_ch * pf * ph * pw
        pe_packed *= pe_info.shape[i]
    var pe_w_raw = _load_dev_preserve(st, pfx + "patch_embedding.weight", ctx)
    var pe_shape = List[Int]()
    pe_shape.append(pe_out)
    pe_shape.append(pe_packed)                    # 64 (T2V) or 144 (I2V)
    var pe_w = reshape(pe_w_raw^, pe_shape^, ctx)
    var pe_b = _load_dev_preserve(st, pfx + "patch_embedding.bias", ctx)

    # text_embedding MLP: Linear -> GELU -> Linear  (frozen; produces context [TXT,dim])
    var te0_w = _load_dev_preserve(st, pfx + "text_embedding.0.weight", ctx)
    var te0_b = _load_dev_preserve(st, pfx + "text_embedding.0.bias", ctx)
    var te2_w = _load_dev_preserve(st, pfx + "text_embedding.2.weight", ctx)
    var te2_b = _load_dev_preserve(st, pfx + "text_embedding.2.bias", ctx)

    # time_embedding MLP: Linear -> SiLU -> Linear  (frozen; produces time features)
    var tme0_w = _load_dev_preserve(st, pfx + "time_embedding.0.weight", ctx)
    var tme0_b = _load_dev_preserve(st, pfx + "time_embedding.0.bias", ctx)
    var tme2_w = _load_dev_preserve(st, pfx + "time_embedding.2.weight", ctx)
    var tme2_b = _load_dev_preserve(st, pfx + "time_embedding.2.bias", ctx)

    # time_projection: SiLU -> Linear(dim -> 6*dim)  (frozen; produces per-token e0)
    var tp1_w = _load_dev_preserve(st, pfx + "time_projection.1.weight", ctx)
    var tp1_b = _load_dev_preserve(st, pfx + "time_projection.1.bias", ctx)

    # head: final linear + modulation
    var hh_w = _load_dev_preserve(st, pfx + "head.head.weight", ctx)
    var hh_b = _load_dev_preserve(st, pfx + "head.head.bias", ctx)
    var hmod = _load_dev_preserve(st, pfx + "head.modulation", ctx)   # [1,2,5120]

    return Wan22StackBase(
        TArc(pe_w^), TArc(pe_b^),
        TArc(te0_w^), TArc(te0_b^), TArc(te2_w^), TArc(te2_b^),
        TArc(tme0_w^), TArc(tme0_b^), TArc(tme2_w^), TArc(tme2_b^),
        TArc(tp1_w^), TArc(tp1_b^),
        TArc(hh_w^), TArc(hh_b^), TArc(hmod^),
    )
