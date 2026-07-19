# Forward-parity vs Serenity/diffusers FLUX.2-klein-base-9B.
#
# Same fixed transformer inputs → compare the Mojo Klein DiT velocity to
# diffusers' Flux2Transformer2DModel velocity (parity/klein_fwd.safetensors).
# LoRA B=0 → forward == base, directly comparable to the reference trainer/diffusers base
# transformer. Mirrors smoke/zimage_forward_parity.mojo.
#
# ── PARITY CONTRACT (the dump format ALL slices agree on) ─────────────────────
# parity/klein_fwd.safetensors (float32) — keys produced by
# parity/gen_klein_forward_ref.py (RNG-FREE intermediates; we feed the dumped
# transformer image input directly so neither side relies on the other's RNG):
#   scaled_noisy_patched [1, 128, 16, 16]  the RNG-free transformer image input
#                            BEFORE pack_latents (Flux2Model.patchify_latents ->
#                            scale_latents -> noise*sigma + scaled*(1-sigma);
#                            BaseFlux2Setup.py:107-130, gen ref lines 220-233). We
#                            pack it HERE (reshape -> permute, Flux2Model.pack_latents
#                            255-257) to the [N_IMG,128] token view the borrowed
#                            forward consumes — IDENTICAL to Flux2LoRASetup
#                            _forward_lora:291-296. No patchify/scale on either side
#                            past this point → no pack/scale-convention drift.
#   txt    [48, 12288]       the diffusers `encoder_hidden_states[0]`
#                            (joint_attention_dim=12288, Qwen3 hidden states). N_TXT=48.
#   velocity [1, 32, 32, 32] diffusers transformer `.sample`, then unpack_latents +
#                            unpatchify_latents (gen ref lines 258-263) = reference trainer
#                            model_output_data['predicted']. FLUX.2 does NOT negate;
#                            predicted IS the flow (BaseFlux2Setup.py:142). We compare
#                            in THIS unpatchified [1,32,32,32] layout: the Mojo flow
#                            [N_IMG,128] is unpack_latents'd + _unpatchify_packed'd the
#                            SAME way (Flux2LoRASetup _forward_lora:332-340) before cos.
#
# Timestep is FIXED at 250 (integer discrete t, gen ref TIMESTEP / meta.json) — the
# value the t_embedder sees. diffusers is fed timestep/1000 and re-scales ×1000
# internally (transformer_flux2.py:1231); the Mojo side feeds the INTEGER t directly
# to build_klein_step_mods_device_cached (NO ×1000; weights.mojo TIMESTEP TRAP).
# guidance_embeds=FALSE for klein-base-9B (transformer/config.json) → guidance=None.
#
# N_IMG/N_TXT are comptime (the dump's token counts). N_IMG MUST be a perfect square
# (the (h,w) latent grid; build_klein_rope_tables_port). 256-res default: image 256² →
# VAE /8 -> [32,32,32] -> patchify /2 -> [128,16,16] -> N_IMG=256, NTXT=48.
#
# CHECKPOINT: /home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors
# (single-file ORIGINAL-format; the SAME weights diffusers loads from the HF snapshot
# transformer/ dir, in serenitymojo/Serenity original key naming — the loaders in
# model/klein/weights.mojo read original keys: double_blocks.*, img_in.weight,
# time_in.*, final_layer.*). Equivalent weights, original-format key layout.
#
# RoPE scheme matches diffusers: Flux2Model.prepare_latent_image_ids /
# prepare_text_ids (Flux2Model.py:240-294) = cartesian_prod over 4 axes (t,h,w,l).
# img token: axis1=row(idx//W), axis2=col(idx%W); txt token: axis3=token index.
# build_klein_rope_tables_port reproduces this exact scheme (theta=2000, axes 4×16).
#
# Run (MAIN LOOP, not the agent):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build -I . -I trainer/src -Xlinker -lm \
#     trainer/smoke/klein_forward_parity.mojo -o /tmp/klein_fp && /tmp/klein_fp
from std.math import sqrt
from std.collections import List, Optional
from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import (
    reshape as _reshape, reshape_owned as _reshape_owned, permute as _permute,
)
from serenitymojo.scratch_ring import ScratchRingAllocator

from serenity_trainer.model.klein.double_block import DoubleBlockWeights
from serenity_trainer.model.klein.single_block import SingleBlockWeights
from serenity_trainer.model.klein.klein_stack import KleinStackBase
from serenity_trainer.model.klein.klein_stack_lora import (
    KleinLoraDeviceSet, klein_lora_set_to_device,
)
from serenity_trainer.model.klein.weights import (
    load_double_block_weights, load_single_block_weights,
    load_klein_stack_base, load_klein_step_mod_weights, build_klein_vec_silu,
    build_klein_step_mods_device_cached,
)
from serenity_trainer.model.KleinModel import (
    klein_inference_forward, build_klein9b_lora_set, build_klein_rope_tables_port,
    KDIM, KIN_CH, KOUT_CH, KTXT_CH, KH, KDh, KNUM_DOUBLE, KNUM_SINGLE, KTIMESTEP_DIM,
)
from serenity_trainer.model.KleinVAE import _unpatchify_packed


comptime TArc = ArcPointer[Tensor]

# token counts the reference generator dumped (256-res default). N_IMG = (16x16).
# HL/WL = patchified latent grid (16x16); N_IMG = HL*WL = 256 image tokens. NTXT=48
# is the fixed Qwen3 sequence length the generator used (gen ref NTXT=48, meta.json).
comptime HL = 16
comptime WL = 16
comptime N_IMG = HL * WL          # 256
comptime N_TXT = 48
comptime S = N_IMG + N_TXT

# fixed integer timestep the t_embedder sees (gen ref TIMESTEP=250, meta.json).
comptime TIMESTEP = Float32(250.0)

comptime PARITY = "trainer/parity/klein_fwd.safetensors"
comptime CKPT = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors"

# LoRA rank/alpha (klein9b.json: lora_rank=16, lora_alpha=16). B=0 at init → the
# overlay is identity regardless of rank/alpha, so forward == base transformer.
comptime LORA_RANK = 16
comptime LORA_ALPHA = Float32(16.0)

# scratch slab for the inference ring allocator. The 9B single-block residents need
# ~3D+2F float buffers per step; size generously (S*KF*4 bytes ≈ 304*12288*4 ≈ 15MB)
# and give a few slabs for the ring. Bump if the allocator raises out-of-slab.
comptime SCRATCH_SLAB_BYTES = 64 * 1024 * 1024
comptime SCRATCH_NUM_SLABS = 4


def main() raises:
    var ctx = DeviceContext()

    # ── fixed inputs + diffusers reference velocity ──
    var st = ShardedSafeTensors.open(String(PARITY))
    # scaled_noisy_patched [1,128,16,16] f32 — the RNG-free transformer image input
    # BEFORE pack_latents (gen ref dumps exactly this to bypass torch RNG).
    var snp_f32 = Tensor.from_view(st.tensor_view(String("scaled_noisy_patched")), ctx)
    var txt_f32 = Tensor.from_view(st.tensor_view(String("txt")), ctx)        # [48,12288] f32
    var vel_ref = Tensor.from_view(st.tensor_view(String("velocity")), ctx)   # [1,32,32,32] f32
    print("[inputs] N_IMG =", N_IMG, " N_TXT =", N_TXT, " timestep(int) =", TIMESTEP)

    # cast to BF16 (training/compute storage).
    var snp_bf = cast_tensor(snp_f32, STDtype.BF16, ctx)   # [1,128,16,16] bf16
    var txt_bf = cast_tensor(txt_f32, STDtype.BF16, ctx)   # [48,12288] bf16
    var txt_tok = _reshape_owned(txt_bf^, [N_TXT, KTXT_CH])

    # pack_latents (Flux2Model.py:255-257), IDENTICAL to Flux2LoRASetup
    # _forward_lora:291-296: [1,128,HL,WL] → reshape [1,128,N_IMG] → permute [0,2,1]
    # → [1,N_IMG,128] → 2D [N_IMG,128] the borrowed forward consumes.
    var packed3 = _reshape(snp_bf, [1, KIN_CH, N_IMG], ctx)
    var packed_p = _permute(packed3, [0, 2, 1], ctx)       # [1, N_IMG, 128]
    var img_tok = _reshape_owned(packed_p^, [N_IMG, KIN_CH])

    # ── load the Klein 9B base from the single-file original-format checkpoint ──
    print("[load]", CKPT)
    var ckpt = SafeTensors.open(String(CKPT))

    # shared modulation feature from the INTEGER timestep t (NO ×1000): vec_silu is
    # also what load_klein_stack_base uses to build final_layer.adaLN shift/scale, so
    # the final layer is timestep-consistent with the per-step modvecs below.
    var ts_dev = Tensor.from_host([TIMESTEP], [1], STDtype.F32, ctx)
    var vec_silu = build_klein_vec_silu(ckpt, ts_dev, KTIMESTEP_DIM, KDIM, ctx)

    var base = load_klein_stack_base(ckpt, vec_silu, KDIM, ctx)
    var step_mod_w = load_klein_step_mod_weights(ckpt, KDIM, ctx)

    var dbw = List[DoubleBlockWeights]()
    for bi in range(KNUM_DOUBLE):
        dbw.append(load_double_block_weights(ckpt, bi, ctx))
    var sbw = List[SingleBlockWeights]()
    for bi in range(KNUM_SINGLE):
        sbw.append(load_single_block_weights(ckpt, bi, ctx))
    print("[load] base + ", len(dbw), "double + ", len(sbw), "single blocks")

    # ── B=0 LoRA overlay (144 adapters) → identity (forward == base) ──
    var lora_host = build_klein9b_lora_set(LORA_RANK, LORA_ALPHA)
    var lora = klein_lora_set_to_device(lora_host, ctx)

    # ── RoPE tables (diffusers id scheme; theta=2000, 4 axes) ──
    var rope_tup = build_klein_rope_tables_port[N_IMG, N_TXT, KH, KDh](ctx, STDtype.BF16)
    ref cos_t = rope_tup[0]
    ref sin_t = rope_tup[1]

    # ── modulation vectors from the INTEGER timestep t; guidance=None (klein-base-9B
    #    guidance_embeds=False). build_klein_step_mods_device_cached threads the same
    #    t the diffusers t_embedder sees (re-scaled ×1000 internally). ──
    var mods = build_klein_step_mods_device_cached(
        step_mod_w, TIMESTEP, Optional[Float32](None), KTIMESTEP_DIM, KDIM, ctx
    )
    var img_mod_dev = mods[0].copy()
    var txt_mod_dev = mods[1].copy()
    var single_mod_dev = mods[2].copy()

    # ── INFERENCE forward (no tape; the sampler path) ──
    print("[forward] running Klein DiT inference forward (B=0) ...")
    var scratch = ScratchRingAllocator(ctx, SCRATCH_SLAB_BYTES, SCRATCH_NUM_SLABS)
    var img_arc = TArc(img_tok^)
    var txt_arc = TArc(txt_tok^)
    var flow = klein_inference_forward[N_IMG, N_TXT, S](
        img_arc, txt_arc, base, dbw, sbw, lora,
        img_mod_dev, txt_mod_dev, single_mod_dev, cos_t, sin_t, ctx, scratch,
    )   # List[Float32] length N_IMG*KOUT_CH (packed flow [N_IMG,128])

    # ── unpack_latents + unpatchify_latents, IDENTICAL to Flux2LoRASetup
    #    _forward_lora:330-340, so we compare in the SAME [1,32,32,32] layout the gen
    #    ref dumps `velocity` in (gen ref:258-263). flow tokens → BF16 [1,N_IMG,128] →
    #    reshape [1,HL,WL,128] → permute [0,3,1,2] → [1,128,HL,WL] → unpatchify →
    #    [1,32,2HL,2WL] = [1,32,32,32]. ──
    var flow_tokens = Tensor.from_host(flow.copy(), [1, N_IMG, KOUT_CH], STDtype.BF16, ctx)
    var flow_b = _reshape(flow_tokens, [1, HL, WL, KOUT_CH], ctx)
    var flow_perm = _permute(flow_b, [0, 3, 1, 2], ctx)        # [1,128,HL,WL]
    var predicted_flow_patch = _reshape_owned(flow_perm^, [1, KOUT_CH, HL, WL])
    var velocity = _unpatchify_packed(predicted_flow_patch, ctx)   # [1,32,32,32] bf16

    # ── compare to the diffusers reference velocity (both [1,32,32,32]) ──
    var a_host = velocity.to_host(ctx)      # bf16→f32 host, N = 1*32*32*32
    var r = vel_ref.to_host(ctx)            # [1,32,32,32] f32
    var n = len(a_host)
    if len(r) != n:
        raise Error(String("len mismatch: mojo ") + String(n) + String(" vs ref ") + String(len(r)))

    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    var mx = Float32(0.0)
    var nonfinite = 0
    for i in range(n):
        var a = a_host[i]
        var b = r[i]
        if a != a or b != b:
            nonfinite += 1
            continue
        dot += Float64(a) * Float64(b)
        na += Float64(a) * Float64(a)
        nb += Float64(b) * Float64(b)
        var d = a - b
        var ad = d if d >= 0.0 else -d
        if ad > mx:
            mx = ad
    var cos = dot / (sqrt(na) * sqrt(nb))
    print("=== KLEIN FORWARD PARITY vs Serenity/diffusers FLUX.2-klein-base-9B ===")
    print("  n =", n, " cos =", cos, " max_abs_diff =", mx, " nonfinite =", nonfinite)
    print("  Mojo[0:3] =", a_host[0], a_host[1], a_host[2], "  ref[0:3] =", r[0], r[1], r[2])
    if cos >= 0.999:
        print("  PASS: cos >= 0.999")
    else:
        print("  FAIL: cos < 0.999")
