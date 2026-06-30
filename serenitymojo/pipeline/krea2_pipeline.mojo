# pipeline/krea2_pipeline.mojo — Krea-2 (krea2) text->image inference pipeline.
#
# Full T2I path (DiT/VAE half), pure Mojo + MAX, GPU-only. Mirrors the ai-toolkit
# reference Krea2Pipeline.__call__ (krea2 src/pipeline.py:185-260) + the
# patchify/pos construction in `prepare` (pipeline.py:66-90) and
# `predict_velocity` (pipeline.py:93-130):
#
#   load cond/uncond context [1,LT,12,2560] from KREA2_CTX_*_BIN (written by the
#       krea2_encode_cli child — the Qwen3-VL TE is NOT loaded in this process)
#   init noise latent [1,16,H/8,W/8] (F32) ; build pos [1,LFULL,3] (txt 0 + img grid)
#   patchify latent [1,16,h8,w8] -> img tokens [1,imglen,64]
#   for each schedule step (t: 1 -> 0):
#       v_cond   = krea2_forward(img_bf16, context_pos, t, pos_pos)   # streamed 28 blocks
#       v_uncond = krea2_forward(img_bf16, context_neg, t, pos_neg)   # (if CFG)
#       v = v_cond + cfg*(v_cond - v_uncond)                          # krea2_cfg
#       latent_tokens = latent_tokens + (t_prev - t_cur)*v            # krea2_euler_step (F32)
#   unpatch [1,imglen,64] -> [1,16,h8,w8] -> VAE decode -> [1,3,H,W] [-1,1] -> PNG
#
# TWO-PROCESS SPLIT (VRAM, measured 2026-06-25): the Qwen3-VL-4B TE load spikes to
# ~22 GB for an ~8 GB bf16 model (the per-tensor pinned host-staging pool in
# Tensor.from_view accumulates to ~8 GB and never returns to the OS, on top of the
# ~8 GB device weights + MAX overhead). Holding that alongside the streamed DiT
# blows past 24 GB (the guarded 256² e2e was SIGKILLed at 2.5 GB free during the TE
# load). FIX = decouple (the proven serenity 24 GB-ceiling / klein9b_encode pattern):
# the krea2_encode_cli child loads the TE, encodes pos+neg, dumps the two contexts,
# and EXITS (freeing all 22 GB); THIS process loads only those small contexts +
# streams the DiT (peak ~3-6 GB). Run the child FIRST.
#
# OFFLOAD: krea2_forward streams the 28 real SingleStreamBlocks disk->GPU one at a
# time (each Tensor.from_view(st.tensor_view("blocks.N...")) copies only that
# block's ~0.87 GB H2D and frees it at loop-iteration end). The ~1.97 GB of shared
# weights (embedders/txtfusion/txtmlp/last) + one block + activations stay well
# under 24 GB, so the 26 GB checkpoint never goes fully resident. No separate
# BlockOffloader is needed — the per-block from_view IS the stream. key_prefix=""
# selects the real raw.safetensors bare keys (the parity oracle uses the default
# "w." prefix).
#
# DTYPE FLOW (matches the reference exactly):
#   * latent accumulator stays F32 (pipeline.py:225) — krea2_euler_step is F32.
#   * the model FEED (img tokens) is bf16 (pipeline.py:245 latents.to(dtype)); we
#     bf16-round the per-step latent with torch_f32_to_bf16_rne so the loop
#     rounding matches torch (the loop-rounding gotcha).
#   * context (TE stack) is bf16; t is f32; pos is f32.
#   * VAE decode consumes the F32 model-space latent and applies the latents_std/
#     mean denorm + [-1,1] clamp INTERNALLY (qwenimage_decoder.decode), so we pass
#     the RAW model-space latent (do NOT pre-denorm here — that is the reference's
#     `latents * std + mean` which the decoder already does).
#
# COMPTIME SHAPES: krea2_forward is comptime-parameterized on (LFULL, LPAD, LT,
# NBLOCKS). At b==1 inference the reference uses the prompt's NATURAL token length
# (no text padding; refiner mask = None), so LT is prompt-dependent and cond/uncond
# differ. We pin LT_POS / LT_NEG (and the derived LFULL/LPAD) as comptime constants
# for the chosen prompts and FAIL LOUD if the cached context's LT differs — the
# message prints the actual LT so the constants can be updated for a new prompt (the
# encode child prints the measured LT too). Defaults: 256x256 / 4-step validation.
#
# Run (orchestrator, guarded — two sequential processes; offload crashes X if
# unguarded; the user is remote):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg
#   # 1) TE child: encode pos+neg contexts -> output/krea2_ctx_{pos,neg}.bin, then exits
#   pixi run mojo run -I . serenitymojo/pipeline/krea2_encode_cli.mojo
#   # 2) DiT/VAE main: read the contexts + stream the 26 GB checkpoint -> PNG
#   pixi run mojo run -I . serenitymojo/pipeline/krea2_pipeline.mojo
#
# Mojo 1.0.0b1, NVIDIA GPU. Inference-only.

from std.gpu.host import DeviceContext
from std.math import sqrt
from std.collections import Optional
from std.sys import argv
from std.time import perf_counter_ns
from serenitymojo.lora import LoraSet
from serenitymojo.models.krea2.krea2_stack import build_krea2_resident_fp8
from serenitymojo.models.dit.krea2_dit import Krea2ResidentFp8

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.tensor_algebra import reshape, permute
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.torch_bf16 import torch_f32_to_bf16_rne
from serenitymojo.ops.random import randn
from serenitymojo.io.cap_cache import load_tensor_bin
from serenitymojo.models.dit.krea2_dit import krea2_forward
from serenitymojo.models.vae.qwenimage_decoder import QwenImageVaeDecoder
from serenitymojo.sampling.krea2_sampler import (
    krea2_timesteps,
    krea2_packed_seq_len,
)
from serenitymojo.sampling.krea2_sampler import krea2_cfg, krea2_euler_step
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.pipeline.krea2_paths import (
    KREA2_RAW,
    KREA2_VAE_DIR,
    KREA2_RAW_KEY_PREFIX,
    KREA2_CTX_POS_BIN,
    KREA2_CTX_NEG_BIN,
)


# ── checkpoint / VAE paths + key prefix come from krea2_paths (shared with the
# encode child). The TEXT ENCODER is NOT loaded here — it runs in the separate
# krea2_encode_cli child (which dumps the two contexts to KREA2_CTX_*_BIN) so its
# ~22 GB load never coexists with the streamed DiT. See krea2_encode_cli header
# for the measured VRAM rationale. ───────────────────────────────────────────


# ── RUN CONFIG (default: 256x256 / 4-step validation). Swap to the 1024 block. ─
# 256x256: gh=gw=16 -> IMGLEN=256 ; LH=LW=32.
comptime HEIGHT = 1024
comptime WIDTH = 1024
comptime LH = HEIGHT // 8      # 128
comptime LW = WIDTH // 8       # 128
comptime IMGLEN = (LH // 2) * (LW // 2)   # 4096 (gh*gw)
comptime STEPS = 20
comptime CFG_SCALE = Float32(6.0)
comptime SEED = UInt64(88888)
comptime OUT_PNG = "/home/alex/mojodiffusion/output/krea2_prompt2_1024.png"

# Prompt-dependent NATURAL token lengths (computed offline; FAIL-LOUD if the cached
# context's LT differs). Default prompts (krea2_paths): the astronaut prompt ->
# L_full=53, LT_POS=19 ; the empty negative -> L_full=39, LT_NEG=5. The encode
# child (krea2_encode_cli) prints the measured LT to update these for a new prompt.
comptime LT_POS = 31           # vrtlEri2 garden-portrait prompt, measured by krea2_encode_cli
comptime LT_NEG = 5
comptime LFULL_POS = LT_POS + IMGLEN   # 492 + 4096 = 4588
comptime LFULL_NEG = LT_NEG + IMGLEN   # 5   + 4096 = 4101
# LPAD = ceil(LFULL/256)*256.
comptime LPAD_POS = ((LFULL_POS + 255) // 256) * 256   # 4608
comptime LPAD_NEG = ((LFULL_NEG + 255) // 256) * 256   # 4352
comptime NBLOCKS = 28          # the real SingleStreamDiT depth.


def _load_context(
    path: String, expect_lt: Int, name: String, ctx: DeviceContext
) raises -> Tensor:
    """Load a cached context [1, LT, 12, 2560] bf16 (dumped by krea2_encode_cli),
    asserting LT == expect_lt (the comptime LFULL/LPAD depend on it). NO encoder
    code or weights touch this process — only the small context tensor is read."""
    var stack = load_tensor_bin(path, ctx)          # [1, LT, 12, 2560] bf16
    var sh = stack.shape()
    if len(sh) != 4 or sh[0] != 1 or sh[2] != 12 or sh[3] != 2560:
        raise Error(
            String("[krea2] ") + name + " cached context has unexpected shape "
            + "(expected [1, LT, 12, 2560]); regenerate with krea2_encode_cli: "
            + path
        )
    var lt = sh[1]
    print("[krea2] ", name, " context LT =", lt, " (expect ", expect_lt, ")")
    if lt != expect_lt:
        raise Error(
            String("[krea2] ") + name + " LT=" + String(lt)
            + " != comptime expectation " + String(expect_lt)
            + " — update LT_" + name + " (and LFULL/LPAD) for this prompt."
        )
    return stack^


def _build_pos(lt: Int, ctx: DeviceContext) raises -> Tensor:
    """Build pos [1, LFULL, 3] f32 = cat(txt zeros [lt,3], img grid [imglen,3]).

    Img grid (pipeline.py:80-83 / prepare): for token (gh_i, gw_i),
    pos[...,0]=0 (global axis), pos[...,1]=gh_i, pos[...,2]=gw_i. gh=LH//2,
    gw=LW//2. Built host-side then uploaded (tiny)."""
    var gh = LH // 2
    var gw = LW // 2
    var host = List[Float32]()
    # text positions: all zeros, lt rows of 3.
    for _ in range(lt * 3):
        host.append(Float32(0.0))
    # image positions in (gh, gw) row-major (matches the patchify token order).
    for hi in range(gh):
        for wi in range(gw):
            host.append(Float32(0.0))       # axis 0 (global) = 0
            host.append(Float32(hi))        # axis 1 (h)
            host.append(Float32(wi))        # axis 2 (w)
    var lfull = lt + IMGLEN
    var shape = [1, lfull, 3]
    return Tensor.from_host(host^, shape^, STDtype.F32, ctx)


def _patchify(latent_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Patchify [1,16,h8,w8] -> img tokens [1, imglen, 64] (pipeline.py:85 /
    rearrange 'b c (h ph) (w pw) -> b (h w) (c ph pw)', ph=pw=2).

    Decompose: [1,16,gh,2,gw,2] -> permute to [1,gh,gw,16,2,2] -> reshape
    [1, gh*gw, 16*2*2]. Output feature order per token is (c, ph, pw) — the
    reference's (c ph pw) flatten."""
    var gh = LH // 2
    var gw = LW // 2
    var x6 = reshape(latent_nchw, [1, 16, gh, 2, gw, 2], ctx)
    # axes:        0  1   2  3   4  5
    # want output [1, gh, gw, 16, 2, 2] = (c ph pw) inner -> perm picks
    # out0=in0(1), out1=in2(gh), out2=in4(gw), out3=in1(c), out4=in3(ph), out5=in5(pw).
    var xp = permute(x6, [0, 2, 4, 1, 3, 5], ctx)   # [1, gh, gw, 16, 2, 2]
    return reshape(xp, [1, gh * gw, 64], ctx)        # [1, imglen, 64]


def _unpatch(tokens: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Inverse of _patchify: [1, imglen, 64] -> [1, 16, h8, w8] (the
    predict_velocity rearrange inverse, pipeline.py:122-129).

    [1, gh*gw, 64] -> [1, gh, gw, 16, 2, 2] -> permute back to
    [1, 16, gh, 2, gw, 2] -> reshape [1, 16, gh*2, gw*2]."""
    var gh = LH // 2
    var gw = LW // 2
    var x6 = reshape(tokens, [1, gh, gw, 16, 2, 2], ctx)
    # invert perm [0,2,4,1,3,5]: out0=in0, out1=in3(16), out2=in1(gh), out3=in4(2ph),
    # out4=in2(gw), out5=in5(2pw).
    var xp = permute(x6, [0, 3, 1, 4, 2, 5], ctx)   # [1, 16, gh, 2, gw, 2]
    return reshape(xp, [1, 16, gh * 2, gw * 2], ctx)  # [1, 16, h8, w8]


def main() raises:
    var ctx = DeviceContext()
    print("[krea2] T2I pipeline (DiT/VAE):", HEIGHT, "x", WIDTH, " steps=", STEPS,
          " cfg=", CFG_SCALE)

    # ── --lora <path> [<mult>]: OVERLAY a trained LoRA on the streamed DiT weights
    # (W += scale·BA per block, never baked). Omit → pristine base (image A). ────
    var lora = Optional[LoraSet](None)
    var lora_mult = Float32(1.0)
    var out_png = String(OUT_PNG)
    var args = argv()
    for i in range(len(args)):
        if args[i] == String("--lora") and i + 1 < len(args):
            var lp = String(args[i + 1])
            lora = Optional[LoraSet](LoraSet.load(lp))
            out_png = String(OUT_PNG) + String(".lora.png")
            print("[krea2] LoRA OVERLAY:", lp, " adapters=", lora.value().num_mappings(),
                  " mult=", lora_mult)
        if args[i] == String("--lora-mult") and i + 1 < len(args):
            lora_mult = Float32(Float64(String(args[i + 1])))

    # ── 1) load the cond/uncond contexts dumped by the krea2_encode_cli child. ──
    # The TEXT ENCODER is NOT loaded here (its ~22 GB load would not leave room for
    # the streamed DiT — measured). Run krea2_encode_cli FIRST; it writes the two
    # bf16 contexts to KREA2_CTX_*_BIN, then this process reads them encoder-free.
    var ctx_pos = _load_context(String(KREA2_CTX_POS_BIN), LT_POS, String("POS"), ctx)
    var ctx_neg = _load_context(String(KREA2_CTX_NEG_BIN), LT_NEG, String("NEG"), ctx)

    # ── 2) build the two pos grids (cond/uncond differ only in txt-zero count). ─
    var pos_p = _build_pos(LT_POS, ctx)
    var pos_n = _build_pos(LT_NEG, ctx)

    # ── 3) init gaussian noise latent [1,16,h8,w8] F32 (pipeline.py:220-225). ──
    var latent = randn([1, 16, LH, LW], SEED, STDtype.F32, ctx)   # F32 accumulator

    # ── 4) schedule (t: 1 -> 0), exp-shift mu from packed seq len. ────────────
    var seq = krea2_packed_seq_len(HEIGHT, WIDTH)
    var ts = krea2_timesteps(seq, STEPS)
    print("[krea2] schedule seq =", seq, " steps =", len(ts) - 1, " ts[0]=", ts[0],
          " ts[-1]=", ts[len(ts) - 1])

    # ── 5) open the DiT checkpoint + build the fp8-RESIDENT base ONCE. ────────
    # Quantize the 28 frozen blocks to fp8 resident (~12GB) at startup → every
    # forward DEQUANTS the resident block (NO per-step disk read). Kills the
    # repetitive-disk-read antipattern (was ~1100 block disk reads/image). Disable
    # with --no-resident (falls back to the per-step disk stream).
    var st = ShardedSafeTensors.open(String(KREA2_RAW))
    var resident = Optional[Krea2ResidentFp8](None)
    var use_resident = True
    for i in range(len(args)):
        if args[i] == String("--no-resident"):
            use_resident = False
    if use_resident:
        print("[krea2] building fp8-resident base (28 blocks, ~12GB, ONCE) ...")
        resident = Optional[Krea2ResidentFp8](
            build_krea2_resident_fp8(st, KREA2_RAW_KEY_PREFIX, NBLOCKS, ctx)
        )
        print("[krea2] fp8-resident base DONE — ZERO per-step disk reads.")

    # ── 6) Euler integration of the flow ODE with CFG. ───────────────────────
    for si in range(STEPS):
        var t_cur = ts[si]
        var t_prev = ts[si + 1]
        # model timestep [1] f32 = t_cur (krea2's convention: t straight through).
        var t_t = Tensor.from_host([t_cur], [1], STDtype.F32, ctx)

        # patchify the CURRENT F32 latent, then bf16-round the FEED (loop rounding).
        var img_tokens_f32 = _patchify(latent, ctx)               # [1,imglen,64] f32
        var img_tokens = torch_f32_to_bf16_rne(img_tokens_f32, ctx)  # bf16 feed

        # TIMING: per-step phase split (one DiT forward vs the 2nd vs the rest).
        # External-wall discipline: sync before each stamp so the delta is the phase's
        # real GPU work (NOT the synced-timer artifact — these are coarse per-FORWARD
        # boundaries, large enough that the drain mis-attribution is small relative).
        ctx.synchronize(); var _f0 = Int(perf_counter_ns())
        var v_cond = krea2_forward[LFULL_POS, LPAD_POS, LT_POS, NBLOCKS](
            st, img_tokens, ctx_pos, t_t, pos_p, ctx, KREA2_RAW_KEY_PREFIX,
            lora, lora_mult, resident,
        )                                                          # [1,imglen,64] bf16
        ctx.synchronize(); var _f1 = Int(perf_counter_ns())
        var v_uncond = krea2_forward[LFULL_NEG, LPAD_NEG, LT_NEG, NBLOCKS](
            st, img_tokens, ctx_neg, t_t, pos_n, ctx, KREA2_RAW_KEY_PREFIX,
            lora, lora_mult, resident,
        )
        ctx.synchronize(); var _f2 = Int(perf_counter_ns())
        # CFG: v = v_cond + cfg*(v_cond - v_uncond)  (krea2_cfg), then to F32.
        var v_bf16 = krea2_cfg(v_cond, v_uncond, CFG_SCALE, ctx)
        var v_f32 = cast_tensor(v_bf16, STDtype.F32, ctx)          # [1,imglen,64] f32

        # unpatch velocity tokens -> latent layout, Euler step in F32.
        var v_latent = _unpatch(v_f32, ctx)                        # [1,16,h8,w8] f32
        latent = krea2_euler_step(latent, v_latent, t_cur, t_prev, ctx)
        ctx.synchronize(); var _f3 = Int(perf_counter_ns())
        print("[krea2] step", si, " fwd_cond=", Float64(_f1 - _f0)/1e9, "s fwd_uncond=",
              Float64(_f2 - _f1)/1e9, "s cfg+euler=", Float64(_f3 - _f2)/1e9,
              "s STEP=", Float64(_f3 - _f0)/1e9, "s")

    # ── 7) VAE decode (decoder denorms latents_std/mean + clamps [-1,1] inside). ─
    print("[krea2] decoding latent -> image ...")
    ctx.synchronize(); var _d0 = Int(perf_counter_ns())
    var dec = QwenImageVaeDecoder[LH, LW].load(String(KREA2_VAE_DIR), ctx)
    # The decoder's internal denorm multiplies the latent by bf16 latents_std/mean
    # (elementwise has no auto-cast). The reference's latent is bf16 at decode (model
    # output bf16); cast the F32 accumulator to bf16 before the decoder.
    var latent_bf16 = torch_f32_to_bf16_rne(latent, ctx)
    var image = dec.decode(latent_bf16, ctx)                       # [1,3,H,W] [-1,1]
    ctx.synchronize(); var _d1 = Int(perf_counter_ns())
    print("[krea2] VAE decode (load+decode) =", Float64(_d1 - _d0)/1e9, "s")

    # ── 8) write PNG (signed [-1,1] -> uint8, == (img+1)*127.5). ─────────────
    save_png(image, out_png, ctx, ValueRange.SIGNED)
    print("[krea2] wrote", out_png)
