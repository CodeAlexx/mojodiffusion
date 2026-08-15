# serenitymojo/pipeline/mageflow_lora_infer.mojo
#
# Mage-Flow-Base T2I inference with a TRAINED LoRA (torchref/PEFT format):
# loads output/mageflow_eri2/mageflow_eri2_lora_step3000.safetensors (144
# adapters, diffusion_model.transformer_blocks.{i}.<mod>.lora_A/B.weight,
# rank16 alpha16 -> live scale 1.0) into the trainer's MageFlowLoraSet and
# generates 4 images at 1024^2, 20-step flow-match, CFG 5.0 (cond+uncond).
#
# NOTHING here is new math: the denoise loop is train_mageflow_real.mojo::
# _sample_one (device arm) — mageflow_stack_lora_sample_forward_device over
# the resident-12 Base blocks with LIVE adapters — and the LoRA load is
# mageflow_stack_lora.mojo::load_mageflow_lora_resume (fail-loud on any of
# the 144 A/B pairs missing). Offload staging: Qwen3-VL encoder loaded/freed
# FIRST (conds cached host-side), then Base + pinned blocks, VAE decode
# streamed per image inside mageflow_decode_latent.
#
# BUILD (sm_120; JIT can't resolve cuMemcpyHtoDAsync_v2):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg
#   pixi run mojo build --target-accelerator sm_120 -I . \
#       serenitymojo/pipeline/mageflow_lora_infer.mojo \
#       -Xlinker -lcuda -Xlinker -lm -o output/mageflow_eri2/mageflow_lora_infer_bin
# RUN:
#   output/mageflow_eri2/mageflow_lora_infer_bin [lora.safetensors] [seed]
#
# Mojo 1.0.0b1, NVIDIA GPU (16GB: ~8.2G resident blocks + activations,
# proven by the in-train 1024^2 sampler).

from max.gpu.host import DeviceContext
from std.collections import List
from std.time import perf_counter_ns
from std.sys import argv

from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_system
from serenitymojo.tensor import Tensor
from serenitymojo.ops.cast import cast_tensor

from serenitymojo.models.dit.mageflow_dit import build_mageflow_rope_tables

from serenitymojo.offload.plan import OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader

from serenitymojo.models.mageflow.config import MageFlowTrainSpec
from serenitymojo.models.mageflow.weights import (
    MageFlowBase, load_mageflow_base, build_mageflow_block_plan,
    compute_mageflow_silu_temb,
)
from serenitymojo.models.mageflow.mageflow_stack_lora import (
    MageFlowLoraSet, mageflow_total_adapters, load_mageflow_lora_resume,
    mageflow_stack_lora_sample_forward_device,
)

from serenitymojo.sampling.flow_match import build_sigma_schedule
from serenitymojo.pipeline.mageflow_pipeline import (
    mageflow_tokenize, gaussian_noise, mageflow_decode_latent,
)
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.models.text_encoder.mageflow_qwen3vl import (
    encode_mageflow_text, load_krea2_qwen3vl_4b, MAGEFLOW_T2I_DROP_IDX,
)


# ── Mage-Flow-Base architecture (== train_mageflow_real.mojo comptimes) ───────
comptime H = 24
comptime Dh = 128
comptime DIM = H * Dh            # 3072
comptime FFN = 12288
comptime IN_CH = 128
comptime OUT_CH = 128
comptime TXT_CH = 2560
comptime DEPTH = 12
comptime EPS = Float32(1.0e-6)
comptime ROPE_THETA = Float64(10000.0)
comptime FRAME = 1

# ── generation geometry / recipe (Base: 20 steps, CFG 5.0, shift 6.0) ─────────
comptime MF_BASE_DIR = "/home/alex/.serenity/models/checkpoints/Mage-Flow-Base"
comptime MF_TE_DIR = MF_BASE_DIR + "/text_encoder"
comptime MF_TOK_JSON = MF_TE_DIR + "/tokenizer.json"
comptime MF_VAE_PATH = MF_BASE_DIR + "/vae/diffusion_pytorch_model.safetensors"
comptime MF_CKPT = MF_BASE_DIR + "/transformer/diffusion_pytorch_model.safetensors"
comptime SH = 64                    # 1024/16 latent side
comptime N_IMG = SH * SH            # 4096 image tokens
comptime STEPS = 20
comptime CFG = Float32(5.0)
comptime FLOW_SHIFT = Float32(6.0)  # Base config.json static_shift

# ── LoRA (trained: rank16 alpha16 -> adapter scale 1.0, the trainer's live) ───
comptime LORA_RANK = 16
comptime LORA_ALPHA = Float32(16.0)
comptime DEFAULT_LORA =
    "/home/alex/mojodiffusion/output/mageflow_eri2/mageflow_eri2_lora_step3000.safetensors"
comptime OUT_DIR = "/home/alex/mojodiffusion/output/mageflow_eri2/infer"
comptime DEFAULT_SEED = UInt64(42)

# ── 4 prompts (trigger vrtlEri2, subject stated as a WOMAN) + uncond ──────────
# keep counts = L_full - drop 34, measured with the Base tokenizer (probe run
# 2026-07-22); VERIFIED at runtime against the real tokenizer (fail loud).
comptime P0 = (
    "vrtlEri2, a beautiful woman, standing in a sunlit olive garden wearing a"
    " flowing white summer dress, golden hour, photorealistic"
)
comptime P1 = (
    "portrait of vrtlEri2, a beautiful woman with dark curly hair, smiling"
    " softly, 85mm studio portrait, grey backdrop, soft light"
)
comptime P2 = (
    "vrtlEri2, a beautiful woman in an elegant black evening gown on a rooftop"
    " at dusk, city lights bokeh, cinematic"
)
comptime P3 = (
    "vrtlEri2, a beautiful woman laughing, casual denim jacket, seaside"
    " boardwalk, bright daylight, candid photo"
)
comptime P0_KEEP = 35
comptime P1_KEEP = 36
comptime P2_KEEP = 32
comptime P3_KEEP = 29
comptime UNCOND_KEEP = 5


def _bf16r(x: Float32) -> Float32:
    # host bf16 RNE round-trip (pipeline latent-boundary discipline).
    return x.cast[DType.bfloat16]().cast[DType.float32]()


# ── STAGE 1: encode the 4 prompts + uncond ONCE (encoder freed on return) ─────
def _encode_prompts(ctx: DeviceContext) raises -> List[List[Float32]]:
    var prompts = List[String]()
    prompts.append(String(P0))
    prompts.append(String(P1))
    prompts.append(String(P2))
    prompts.append(String(P3))
    prompts.append(String(""))
    var expected = List[Int]()
    expected.append(P0_KEEP)
    expected.append(P1_KEEP)
    expected.append(P2_KEEP)
    expected.append(P3_KEEP)
    expected.append(UNCOND_KEEP)

    print("[encode] loading Qwen3-VL (freed before the DiT loads) ...")
    var enc = load_krea2_qwen3vl_4b(String(MF_TE_DIR), ctx)
    var out = List[List[Float32]]()
    for i in range(len(prompts)):
        var ids = mageflow_tokenize(String(MF_TOK_JSON), prompts[i])
        var keep = len(ids) - MAGEFLOW_T2I_DROP_IDX
        if keep != expected[i]:
            raise Error(
                String("_encode_prompts: prompt ") + String(i) + " keep="
                + String(keep) + " != baked comptime " + String(expected[i])
                + " — recount tokens and rebake."
            )
        var txt = encode_mageflow_text(enc, ids, MAGEFLOW_T2I_DROP_IDX, ctx)
        out.append(cast_tensor(txt, STDtype.F32, ctx).to_host(ctx))
        print("[encode]   cond", i, "keep=", keep)
    print("[encode] 4 conds + uncond cached host-side (encoder freed on return)")
    return out^


# ── denoise ONE prompt at 1024^2 with the LoRA applied live, CFG, decode ──────
# == train_mageflow_real.mojo::_sample_one, DEVICE arm only.
def _generate_one[
    NT: Int
](
    tag: String,
    txt_cond: List[Float32], txt_uncond: List[Float32],
    base: MageFlowBase, mut loader: TurboPlannedLoader,
    lora: MageFlowLoraSet, spec: MageFlowTrainSpec,
    seed: UInt64, out_png: String, ctx: DeviceContext,
) raises -> Float64:
    comptime SC = N_IMG + NT
    comptime SU = N_IMG + UNCOND_KEEP
    var t0 = perf_counter_ns()

    var rope_c = build_mageflow_rope_tables(
        FRAME, SH, SH, NT, H, ROPE_THETA, STDtype.F32, ctx
    )
    var rope_u = build_mageflow_rope_tables(
        FRAME, SH, SH, UNCOND_KEEP, H, ROPE_THETA, STDtype.F32, ctx
    )
    var sigmas = build_sigma_schedule(STEPS, FLOW_SHIFT)
    var x = gaussian_noise(N_IMG * IN_CH, seed)   # F32 host

    for i in range(STEPS):
        # bf16 latent at the step boundary (pipeline trajectory discipline).
        var x_bf = List[Float32]()
        for j in range(len(x)):
            x_bf.append(_bf16r(x[j]))
        var silu_temb = compute_mageflow_silu_temb(base, sigmas[i], spec, ctx)

        var vel = mageflow_stack_lora_sample_forward_device[H, Dh, N_IMG, NT, SC](
            x_bf, txt_cond, silu_temb, base, loader, lora,
            rope_c[0], rope_c[1], DIM, FFN, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
        )
        var vel_u = mageflow_stack_lora_sample_forward_device[
            H, Dh, N_IMG, UNCOND_KEEP, SU
        ](
            x_bf, txt_uncond, silu_temb, base, loader, lora,
            rope_u[0], rope_u[1], DIM, FFN, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
        )
        for j in range(len(vel)):
            vel[j] = vel_u[j] + CFG * (vel[j] - vel_u[j])

        var dt = sigmas[i + 1] - sigmas[i]
        for j in range(len(x)):
            x[j] = _bf16r(x_bf[j] + _bf16r(vel[j] * dt))
        print("[gen]   ", tag, " step", i + 1, "/", STEPS,
              " sigma", sigmas[i], "->", sigmas[i + 1])

    # MageVAE decode (weights loaded/freed inside) -> PNG.
    var lat = Tensor.from_host(x^, [1, N_IMG, IN_CH], STDtype.F32, ctx)
    var rgb = mageflow_decode_latent[N_IMG, SH, SH](String(MF_VAE_PATH), lat, ctx)
    save_png(rgb, out_png, ctx, ValueRange.SIGNED)
    var sec = Float64(perf_counter_ns() - t0) / 1.0e9
    print("[gen]   ", tag, " saved:", out_png, " (", sec, "sec )")
    return sec


def main() raises:
    var ctx = DeviceContext()
    var spec = MageFlowTrainSpec.mageflow_base()

    var lora_path = String(DEFAULT_LORA)
    var seed = DEFAULT_SEED
    var args = argv()
    if len(args) > 1:
        lora_path = String(args[1])
    if len(args) > 2:
        seed = UInt64(Int(String(args[2])))

    print("==== Mage-Flow-Base + LoRA T2I inference (1024^2, ", STEPS,
          "-step, CFG", CFG, ", shift", FLOW_SHIFT, ") ====")
    print("lora:", lora_path, " (rank", LORA_RANK, " alpha", LORA_ALPHA,
          " -> live scale 1.0)")
    print("seed:", seed)
    _ = sys_system(String("mkdir -p '") + String(OUT_DIR) + String("'"))

    # STAGE 1: text conds (encoder loaded + freed inside).
    var conds = _encode_prompts(ctx)

    # STAGE 2: resident Base + pinned 12 blocks + LoRA.
    print("[base] loading resident non-block base + block loader ...")
    var base_st = SafeTensors.open(String(MF_CKPT))
    var base = load_mageflow_base(base_st, spec, ctx)
    var plan = build_mageflow_block_plan(DEPTH)
    var loader = TurboPlannedLoader.open(
        String(MF_CKPT), plan^, OffloadConfig.synchronous_single(), ctx, False
    )
    var pinned = loader.pin_residents(10 * 1024 * 1024 * 1024, ctx)
    if pinned != DEPTH:
        raise Error(
            String("[resident] pinned ") + String(pinned) + " of "
            + String(DEPTH) + " blocks — refusing partial residency"
        )
    print("[base]", pinned, "blocks device-resident (~8.2GB)")

    # LoRA: fail-loud PEFT load — load_lora_for_resume raises on ANY missing
    # diffusion_model.transformer_blocks.{i}.<mod>.lora_A/B.weight pair.
    var lora = load_mageflow_lora_resume(
        DEPTH, LORA_RANK, LORA_ALPHA, lora_path, ctx
    )
    var n_adapters = mageflow_total_adapters(lora)
    if n_adapters != 144:
        raise Error(
            String("[lora] adapter count ") + String(n_adapters)
            + " != 144 — refusing"
        )
    if lora.rank != LORA_RANK:
        raise Error(
            String("[lora] file rank ") + String(lora.rank) + " != "
            + String(LORA_RANK) + " — refusing"
        )
    print("[lora] 144/144 adapters loaded (rank", lora.rank, ")")

    # STAGE 3: 4 generations (fixed per-prompt seeds; VAE streamed per image).
    var times = List[Float64]()
    times.append(_generate_one[P0_KEEP](
        String("P0(olive garden)"), conds[0], conds[4], base, loader, lora,
        spec, seed + UInt64(100), String(OUT_DIR) + "/loraP0.png", ctx,
    ))
    times.append(_generate_one[P1_KEEP](
        String("P1(85mm portrait)"), conds[1], conds[4], base, loader, lora,
        spec, seed + UInt64(101), String(OUT_DIR) + "/loraP1.png", ctx,
    ))
    times.append(_generate_one[P2_KEEP](
        String("P2(evening gown)"), conds[2], conds[4], base, loader, lora,
        spec, seed + UInt64(102), String(OUT_DIR) + "/loraP2.png", ctx,
    ))
    times.append(_generate_one[P3_KEEP](
        String("P3(boardwalk)"), conds[3], conds[4], base, loader, lora,
        spec, seed + UInt64(103), String(OUT_DIR) + "/loraP3.png", ctx,
    ))

    print("")
    print("==== DONE: 4 images at", String(OUT_DIR), "====")
    for i in range(len(times)):
        print("  loraP" + String(i) + ".png  ", times[i], "sec")
