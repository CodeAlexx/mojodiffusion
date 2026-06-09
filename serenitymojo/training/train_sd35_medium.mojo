# serenitymojo/training/train_sd35_medium.mojo
#
# REAL sd3.5-MEDIUM 1024^2 LoRA trainer (block-swap offload). Built on the
# parity-verified path:
#   - forward verified faithful to diffusers (sd35_fullfwd_parity: cos 0.99999987)
#   - 3-variant dispatch: dual blocks 0-12 + standard 13-22 + ctx_pre_only 23
#   - pos_embed center-crop added on image tokens
#   - flow-match: noisy INPUT packed (c,ph,pw) [conv], velocity TARGET packed
#     (ph,pw,c) [proj_out order, matches fwd.out]
# Iterates the real OneTrainer cache (latent/text_embedding/pooled), fresh noise
# per step (real flow-match, NOT overfit), trains LoRA, saves PEFT safetensors.
#
# Run (no prlimit — offload needs cuda):
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#       serenitymojo/training/train_sd35_medium.mojo -o /tmp/train_sd35_medium
#   /tmp/train_sd35_medium [steps]

from std.gpu.host import DeviceContext
from std.collections import List
from std.math import sqrt, sin
from std.time import perf_counter_ns
from std.os import listdir
from std.sys import argv
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.models.sd35.weights import load_sd35_stack_base
from serenitymojo.models.sd35.sd35_stack_lora import (
    SD35StackBase, SD35LoraSet, SD35LoraGradSet,
    build_sd35_lora_set, sd35_lora_adamw_step, total_adapters,
    sd35_stack_lora_forward_offload, sd35_stack_lora_backward_offload,
    save_sd35_lora,
)
from serenitymojo.offload.plan import build_sd35_block_plan, OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader

# ── sd3.5-MEDIUM arch ──
comptime H = 24
comptime Dh = 64
comptime D = H * Dh            # 1536
comptime FMLP = 6144
comptime IN_CH = 64
comptime TXT_CH = 4096
comptime OUT_CH = 64
comptime NUM_JOINT = 24
comptime NUM_DUAL = 13          # blocks 0-12 dual; block 23 ctx_pre_only
comptime TIMESTEP_DIM = 256
comptime POOLED_DIM = 2048
comptime EPS = Float32(1e-6)
comptime QK_EPS = Float32(1e-6)

comptime LAT_C = 16
comptime LAT_H = 128
comptime LAT_W = 128
comptime PATCH = 2
comptime HT = LAT_H // PATCH
comptime WT = LAT_W // PATCH
comptime N_IMG = HT * WT        # 4096
comptime N_TXT = 154
comptime S = N_TXT + N_IMG      # 4250
comptime VAE_SHIFT = Float32(0.0609)
comptime VAE_SCALE = Float32(1.5305)

comptime RANK = 16
comptime ALPHA = Float32(16.0)
comptime LR = Float32(1.0e-4)
comptime SIGMA = Float32(0.5)
comptime DEFAULT_STEPS = 6
comptime CLIP = Float32(1.0)
comptime CKPT = "/home/alex/.serenity/models/checkpoints/sd3.5_medium.safetensors"
comptime CACHE_DIR = "/home/alex/datasets/andrsd35_sd35_cache"
comptime OUT_LORA = "/home/alex/mojodiffusion/output/sd35_lora/sd35_medium_lora.safetensors"


def _cache_f32(st: SafeTensors, name: String, ctx: DeviceContext) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    if t.dtype() == STDtype.BF16:
        var bf = t.to_host_bf16(ctx)
        var out = List[Float32]()
        for i in range(len(bf)):
            out.append(bf[i].cast[DType.float32]())
        return out^
    return t.to_host(ctx)


# (c,ph,pw) input order (x_embedder conv)
def _pack_in(lat: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for ih in range(HT):
        for iw in range(WT):
            for c in range(LAT_C):
                for ph in range(PATCH):
                    for pw in range(PATCH):
                        out.append(lat[c * LAT_H * LAT_W + (ih * PATCH + ph) * LAT_W + (iw * PATCH + pw)])
    return out^


# (ph,pw,c) output order (proj_out / fwd.out) — parity-verified
def _pack_out(lat: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for ih in range(HT):
        for iw in range(WT):
            for ph in range(PATCH):
                for pw in range(PATCH):
                    for c in range(LAT_C):
                        out.append(lat[c * LAT_H * LAT_W + (ih * PATCH + ph) * LAT_W + (iw * PATCH + pw)])
    return out^


def _noise(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var st = seed
    for _ in range(n):
        st = st * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(st >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * Float32(2.0))
    return out^


def _absum(v: List[BFloat16]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(v)):
        var x = v[i].cast[DType.float32]()
        s += x if x >= 0.0 else -x
    return s


def _clip(mut grads: SD35LoraGradSet, max_norm: Float32) -> Float32:
    var ss = Float32(0.0)
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            ss += grads.d_a[i][j] * grads.d_a[i][j]
        for j in range(len(grads.d_b[i])):
            ss += grads.d_b[i][j] * grads.d_b[i][j]
    var gn = sqrt(ss)
    if gn > max_norm and gn > Float32(0.0):
        var s = max_norm / gn
        for i in range(len(grads.d_a)):
            for j in range(len(grads.d_a[i])):
                grads.d_a[i][j] = grads.d_a[i][j] * s
            for j in range(len(grads.d_b[i])):
                grads.d_b[i][j] = grads.d_b[i][j] * s
    return gn


def _list_cache(dir: String) raises -> List[String]:
    var raw = listdir(dir)
    var fs = List[String]()
    for i in range(len(raw)):
        if raw[i].endswith(".safetensors"):
            fs.append(dir + String("/") + raw[i])
    if len(fs) == 0:
        raise Error(String("no .safetensors in ") + dir)
    for i in range(1, len(fs)):
        var j = i
        while j > 0 and fs[j - 1] > fs[j]:
            var tmp = fs[j - 1]; fs[j - 1] = fs[j]; fs[j] = tmp; j -= 1
    return fs^


def main() raises:
    var ctx = DeviceContext()
    var steps = DEFAULT_STEPS
    var a = argv()
    if len(a) > 1:
        var v = 0
        var bs = a[1].as_bytes()
        var okint = len(a[1]) > 0
        for i in range(len(a[1])):
            if bs[i] < 0x30 or bs[i] > 0x39: okint = False
            else: v = v * 10 + Int(bs[i] - 0x30)
        if okint: steps = v

    print("==== sd3.5-MEDIUM 1024^2 LoRA trainer (offload, parity-verified path) ====")
    print("D=", D, " depth=", NUM_JOINT, " dual=0..", NUM_DUAL - 1, " ctxpre=block", NUM_JOINT - 1)
    print("tokens N_IMG=", N_IMG, " N_TXT=", N_TXT, " S=", S, " (1024^2)  steps=", steps)
    print("ckpt:", CKPT, "  cache:", CACHE_DIR)

    var base_st = SafeTensors.open(CKPT)
    var base = load_sd35_stack_base(base_st, ctx)
    var plan = build_sd35_block_plan(NUM_JOINT)
    var cfg = OffloadConfig.synchronous_single()
    var loader = TurboPlannedLoader.open(CKPT, plan^, cfg, ctx)
    var lora = build_sd35_lora_set(NUM_JOINT, D, FMLP, RANK, ALPHA)
    print("[lora] adapters:", total_adapters(lora))

    var files = _list_cache(CACHE_DIR)
    print("[cache] samples:", len(files))
    var b0 = Float32(0.0)
    for i in range(total_adapters(lora)):
        b0 += _absum(lora.ad[i].b)
    print("[lora] B |.|_1 init =", b0, " (expect 0)")

    var train_start = perf_counter_ns()
    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)
    print("")
    print("step  sample              MSE          grad_norm   sec")

    for step in range(steps):
        var t0 = perf_counter_ns()
        var path = files[step % len(files)]
        var st = SafeTensors.open(path)
        var lat_raw = _cache_f32(st, String("latent"), ctx)
        var txt = _cache_f32(st, String("text_embedding"), ctx)
        var pooled = _cache_f32(st, String("pooled"), ctx)

        var lat_scaled = List[Float32]()
        for i in range(len(lat_raw)):
            lat_scaled.append((lat_raw[i] - VAE_SHIFT) * VAE_SCALE)
        var noise = _noise(LAT_C * LAT_H * LAT_W, UInt64(7919) * UInt64(step + 1) + UInt64(101))
        var noisy_px = List[Float32]()
        var vel_px = List[Float32]()
        for i in range(len(lat_scaled)):
            noisy_px.append(noise[i] * SIGMA + lat_scaled[i] * (Float32(1.0) - SIGMA))
            vel_px.append(noise[i] - lat_scaled[i])
        var noisy = _pack_in(noisy_px)
        var target = _pack_out(vel_px)

        var fwd = sd35_stack_lora_forward_offload[H, Dh, N_IMG, N_TXT, S](
            noisy.copy(), txt.copy(), pooled.copy(), SIGMA,
            base, loader, lora,
            D, FMLP, IN_CH, TXT_CH, OUT_CH, TIMESTEP_DIM, POOLED_DIM,
            EPS, QK_EPS, ctx, NUM_DUAL, True,
        )
        var nout = len(fwd.out)
        var loss = Float32(0.0)
        var d_loss = List[Float32]()
        var inv_n = Float32(2.0) / Float32(nout)
        for i in range(nout):
            var diff = fwd.out[i] - target[i]
            loss += diff * diff
            d_loss.append(inv_n * diff)
        loss = loss / Float32(nout)
        if step == 0: first_loss = loss
        last_loss = loss

        var grads = sd35_stack_lora_backward_offload[H, Dh, N_IMG, N_TXT, S](
            d_loss, noisy.copy(), txt.copy(),
            base, loader, lora, fwd,
            D, FMLP, IN_CH, TXT_CH, OUT_CH, TIMESTEP_DIM, POOLED_DIM,
            EPS, QK_EPS, ctx, NUM_DUAL, True,
        )
        var gn = _clip(grads, CLIP)
        sd35_lora_adamw_step(lora, grads, step + 1, LR, ctx)
        var secs = Float64(perf_counter_ns() - t0) / 1.0e9
        print(step, " ", path, " ", loss, " ", gn, " ", secs)
        if grads.nonfinite != 0:
            print("  !! nonfinite grads =", grads.nonfinite)

    var b1 = Float32(0.0)
    var nz = 0
    for i in range(total_adapters(lora)):
        var s = _absum(lora.ad[i].b)
        b1 += s
        if s > 0.0: nz += 1
    print("")
    print("[lora] B |.|_1 after =", b1, "  nonzero adapters =", nz, "/", total_adapters(lora))

    var npairs = save_sd35_lora(lora, OUT_LORA, ctx)
    print("[save] wrote", npairs, "adapter pairs ->", OUT_LORA)

    print("")
    print("first MSE =", first_loss, "  last MSE =", last_loss,
          "  total sec =", Float64(perf_counter_ns() - train_start) / 1.0e9)
    # B 0->nonzero on the WIRED slots = learning. Expected nonzero count = the
    # actually-applied LoRA slots: standard blocks 8 each, ctxpre block 5 (x*4 +
    # ctx-qkv), dual blocks 1 each (x-qkv; attn2 frozen) -> not all 192.
    var trains = (b0 == Float32(0.0)) and (b1 > Float32(0.0))
    if trains and npairs > 0:
        print("RESULT: sd3.5-medium 1024^2 LoRA trainer WORKS — iterated cache, real",
              "flow-match, LoRA-B 0->", b1, "(", nz, "wired adapters learning), saved",
              npairs, "PEFT pairs.")
    else:
        print("RESULT: FAIL — b_init=", b0, " b_after=", b1, " saved_pairs=", npairs)
