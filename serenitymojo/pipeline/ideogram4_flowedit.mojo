# pipeline/ideogram4_flowedit.mojo — FlowEdit: training-free editing on Ideogram-4
# (task #25). Port of the proven krea2_flowedit.mojo (ICCV'25, arXiv 2412.08629)
# to the ideogram4 fp8-resident DiT, refit for the 16GB RTX 5080:
#
#   * SINGLE-TRUNK uncond: every uncond pass runs through the ONE resident cond
#     transformer with zeroed text features on the image-only S=NIMG sequence
#     (trainer precedent Ideogram4SampleResident.mojo:143-152). The 8.7GB
#     unconditional_transformer is NEVER loaded.
#   * LAYER-STREAMED TE (ideogram_qwen3vl_streamed, parity cos 0.99998): src+tgt
#     prompts encoded in ONE disk pass BEFORE the DiT loads; mempool trimmed.
#   * VAE decode: whole-image if >=14GiB free else 3x3 tiled (backend policy).
#
# TIME CONVENTION (ideogram vs krea2): ideogram model-time tau runs 0 -> 1 with
# tau~0 = pure noise, tau~1 = clean, Euler z += v*(tau_next - tau) — the EXACT
# ideogram4_generate.mojo loop (t_val=logitnormal(si[step+1]) ... z += v*(s-t)).
# So FlowEdit's noising is Zt = tau*Z0 + (1-tau)*N (krea2's (1-t)*Z0 + t*N with
# t = 1 - tau), and the skip window drops the FIRST steps-nmax (noisiest,
# lowest-tau) iterations:
#
#   Z0_src = VAE-encoded source tokens [1,NIMG,128] (normalized latent space)
#   Z_edit = Z0_src
#   for i in 0..steps (step = steps-1-i, ONLY i in [steps-nmax, steps-nmin)):
#     tau      = logitnormal(si[step+1], mu, 1.75)
#     tau_next = logitnormal(si[step],   mu, 1.75)
#     N       ~ randn(seed + i)                     (fresh per step)
#     Zt_src  = tau*Z0_src + (1-tau)*N
#     Zt_tgt  = Z_edit + (Zt_src - Z0_src)
#     V_src   = cfg*V(Zt_src|src) + (1-cfg)*V(Zt_src|zero)     cfg = --src-cfg
#     V_tgt   = cfg*V(Zt_tgt|tgt) + (1-cfg)*V(Zt_tgt|zero)     cfg = --tgt-cfg
#     Z_edit += (tau_next - tau) * (V_tgt - V_src)  (4 forwards/step)
#   decode Z_edit -> PNG (+ pixel MAD vs the staged source; identity-gate metric)
#
# AUTO-MASK (krea2_flowedit port, 64x64 token grid, 128-dim token blocks):
# accumulate per-token ||V_tgt - V_src||_2 across active steps; threshold at
# quantile --mask-q, dilate --mask-dilate (3x3), and past --mask-warmup active
# steps HARD-COPY Z0_src back into every token OUTSIDE the mask after each Euler
# update. Final mask saved as <out>_mask.png (64x64 nearest-upscaled x16).
#
# PROMPTS: ideogram wants structured JSON captions (docs/IDEOGRAM4_PROMPTING.md).
# For FlowEdit author an IDENTICAL src/tgt JSON pair with ONE semantic element
# changed (e.g. the dress color) — that is exactly FlowEdit's shape.
#
# BUILD:
#   cd /home/alex/mojodiffusion && \
#   pixi run mojo build --optimization-level 2 -I . -I /home/alex/MOJO-libs \
#     -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#     serenitymojo/pipeline/ideogram4_flowedit.mojo -o /tmp/i4_flowedit
# RUN:
#   LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib \
#     /tmp/i4_flowedit <src_1024.safetensors> <src_prompt.json> <tgt_prompt.json>
#       <out.png> [--steps 28] [--nmax 24] [--nmin 0] [--src-cfg 1.5]
#       [--tgt-cfg 5.0] [--seed N] [--auto-mask] [--mask-q 0.7]
#       [--mask-dilate 1] [--mask-warmup 4]
#
# Mojo 1.0.0b1, NVIDIA GPU. Inference-only.

from std.gpu.host import DeviceContext
from std.math import sqrt, log
from std.memory import ArcPointer, alloc
from std.sys import argv
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, BytePtr, O_RDONLY
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import (
    mul, add, sub, mul_scalar, reshape, permute, slice, concat,
)
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current, cu_mem_get_info
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.dit.ideogram4_resident import (
    Ideogram4Weights, ideogram4_forward_r, ideogram4_build_masks, Ideogram4Masks,
)
from serenitymojo.models.dit.ideogram4_mrope import build_ideogram4_mrope
from serenitymojo.models.vae.ldm_decoder import load_ideogram4_vae_decoder
from serenitymojo.models.vae.ideogram4_tiled_decode import ideogram4_tiled_decode
from serenitymojo.models.vae.ldm_encoder import (
    load_ideogram4_vae_encoder, encode_ideogram4_latents,
)
from serenitymojo.models.text_encoder.ideogram_qwen3vl_streamed import (
    encode_ideogram_taps_streamed,
)
from serenitymojo.sampling.ideogram4_schedule import (
    ideogram4_logitnormal, ideogram4_schedule_mean, make_step_intervals,
)

comptime TArc = ArcPointer[Tensor]

comptime COND = "/home/alex/.serenity/models/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors"
comptime TE = "/home/alex/.serenity/models/ideogram-4-fp8/text_encoder/model.safetensors"
comptime TOK_JSON = "/home/alex/.serenity/models/ideogram-4-fp8/tokenizer/tokenizer.json"
comptime VAE = "/home/alex/.serenity/models/ideogram-4-fp8/vae/diffusion_pytorch_model.safetensors"
comptime LATENT_NORM = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_latentnorm.safetensors"

comptime PAD_ID = 151643
comptime IMG_OFFSET = 65536
comptime TEXT_TOKENS = 1024
comptime FEAT_DIM = 53248
comptime HEIGHT = 1024
comptime WIDTH = 1024
comptime LH = HEIGHT // 8            # 128 (VAE latent edge)
comptime LW = WIDTH // 8
comptime GH = 64                     # token grid (= LH/2)
comptime GW = 64
comptime NIMG = GH * GW              # 4096
comptime TOTAL = TEXT_TOKENS + NIMG  # 5120
comptime LAYERS = 34
comptime HEADS = 18
comptime HEAD_DIM = 256
comptime HIDDEN = 4608

comptime STEPS_DEFAULT = 28          # FlowEdit FLUX config T=28
comptime NMAX_DEFAULT = 24           # skip the first T-n_max = 4 noisiest steps
comptime NMIN_DEFAULT = 0
comptime SRC_CFG_DEFAULT = Float32(1.5)
comptime TGT_CFG_DEFAULT = Float32(5.0)
comptime SEED_DEFAULT = UInt64(88888)

# ── auto-mask geometry: 64x64 token grid, 128-dim token blocks ────────────────
comptime TOK_GH = GH
comptime TOK_GW = GW
comptime NTOK = NIMG                 # 4096
comptime TOK_DIM = 128


def _read_text_file(path: String) raises -> String:
    var fd = sys_open(path, O_RDONLY, Int32(0))
    if fd < 0:
        raise Error(String("[i4-flowedit] file not found: ") + path)
    var bytes = List[UInt8]()
    comptime CHUNK = 65536
    var buf = alloc[UInt8](CHUNK)
    var offset = 0
    while True:
        var nread = sys_pread(fd, BytePtr(unsafe_from_address=Int(buf)), CHUNK, offset)
        if nread < 0:
            buf.free()
            _ = sys_close(fd)
            raise Error(String("[i4-flowedit] read error: ") + path)
        if nread == 0:
            break
        for i in range(nread):
            bytes.append(buf[i])
        offset += nread
        if nread < CHUNK:
            break
    buf.free()
    _ = sys_close(fd)
    while len(bytes) > 0 and (bytes[len(bytes) - 1] == 10 or bytes[len(bytes) - 1] == 13):
        _ = bytes.pop()
    return String(unsafe_from_utf8=bytes)


def _render_chat_prompt(prompt: String) -> String:
    return (
        String("<|im_start|>user\n") + prompt
        + String("<|im_end|>\n<|im_start|>assistant\n")
    )


# src+tgt prompt files -> 2 x [1,TEXT_TOKENS,53248] BF16 (pad rows ZEROED, the
# ideogram4_prepare convention) via the 16GB layer-streamed TE, ONE disk pass.
def _encode_prompt_pair(
    src_path: String, tgt_path: String, ctx: DeviceContext
) raises -> List[TArc]:
    var tok = Qwen3Tokenizer(String(TOK_JSON))
    var ids_list = List[List[Int]]()
    var real_lens = List[Int]()
    var paths = [src_path, tgt_path]
    var names = [String("SRC"), String("TGT")]
    for i in range(2):
        var ids = tok.encode(_render_chat_prompt(_read_text_file(paths[i])))
        var real_len = len(ids)
        if real_len > TEXT_TOKENS:
            raise Error(
                String("[i4-flowedit] ") + names[i] + " prompt tokenized to "
                + String(real_len) + " > fixed window " + String(TEXT_TOKENS)
            )
        for _ in range(TEXT_TOKENS - real_len):
            ids.append(PAD_ID)
        print("[i4-flowedit]", names[i], "prompt tokens =", real_len, "/", TEXT_TOKENS)
        ids_list.append(ids^)
        real_lens.append(real_len)
    var feats = encode_ideogram_taps_streamed(String(TE), ids_list^, ctx)
    var out = List[TArc]()
    for i in range(2):
        var mask_host = List[Float32]()
        for j in range(TEXT_TOKENS):
            mask_host.append(Float32(1.0) if j < real_lens[i] else Float32(0.0))
        var mask_f32 = Tensor.from_host(mask_host^, [1, TEXT_TOKENS, 1], STDtype.F32, ctx)
        var mask = cast_tensor(mask_f32, STDtype.BF16, ctx)
        out.append(TArc(cast_tensor(mul(feats[i][], mask, ctx), STDtype.BF16, ctx)))
    return out^


# staged source [1,3,1024,1024] F32 [-1,1] -> Z0_src tokens [1,NIMG,128] F32
# (normalized packed latent -> grid->token permute — gate-1-verified exact
# inverse of the generate.mojo unpatch). Encoder frees on return.
def _encode_source_tokens(src_path: String, ctx: DeviceContext) raises -> Tensor:
    var imgs = ShardedSafeTensors.open(src_path)
    var img_f32 = Tensor.from_view(imgs.tensor_view("image"), ctx)
    var sh = img_f32.shape()
    if len(sh) != 4 or sh[0] != 1 or sh[1] != 3 or sh[2] != HEIGHT or sh[3] != WIDTH:
        raise Error("[i4-flowedit] staged source must be [1,3,1024,1024] F32 [-1,1]")
    var img = cast_tensor(img_f32, STDtype.BF16, ctx)
    var ln = ShardedSafeTensors.open(String(LATENT_NORM))
    var shift = Tensor.from_view(ln.tensor_view("latent_shift"), ctx)
    var scale = Tensor.from_view(ln.tensor_view("latent_scale"), ctx)
    var venc = load_ideogram4_vae_encoder[LH, LW](String(VAE), ctx)
    var z_grid = encode_ideogram4_latents[LH, LW](venc, img, shift, scale, ctx)  # [1,128,GH,GW] F32
    var z_hwc = permute(z_grid, [0, 2, 3, 1], ctx)
    var z_tok = reshape(z_hwc, [1, NIMG, 128], ctx)
    print("[i4-flowedit] Z0_src encoded [1,", NIMG, ",128] F32 (normalized)")
    return z_tok^


def _build_inputs(ctx: DeviceContext) raises -> List[TArc]:
    var pos = List[Float32]()
    var ind = List[Float32]()
    var npos = List[Float32]()
    var nind = List[Float32]()
    for l in range(TEXT_TOKENS):
        pos.append(Float32(l)); pos.append(Float32(l)); pos.append(Float32(l))
        ind.append(3.0)
    for h in range(GH):
        for w in range(GW):
            var t0 = Float32(IMG_OFFSET)
            var hh = Float32(IMG_OFFSET + h)
            var ww = Float32(IMG_OFFSET + w)
            pos.append(t0); pos.append(hh); pos.append(ww)
            npos.append(t0); npos.append(hh); npos.append(ww)
            ind.append(2.0)
            nind.append(2.0)
    var out = List[TArc]()
    out.append(TArc(Tensor.from_host(pos^, [1, TOTAL, 3], STDtype.F32, ctx)))
    out.append(TArc(Tensor.from_host(ind^, [1, TOTAL], STDtype.F32, ctx)))
    out.append(TArc(Tensor.from_host(npos^, [1, NIMG, 3], STDtype.F32, ctx)))
    out.append(TArc(Tensor.from_host(nind^, [1, NIMG], STDtype.F32, ctx)))
    return out^


def _zeros_bf16(var shape: List[Int], n: Int, ctx: DeviceContext) raises -> Tensor:
    var h = List[Float32](capacity=n)
    for _ in range(n):
        h.append(0.0)
    return Tensor.from_host(h^, shape^, STDtype.BF16, ctx)


# One single-trunk CFG velocity in TOKEN space [1,NIMG,128] F32.
def _velocity(
    w: Ideogram4Weights,
    z_tok_f32: Tensor,          # [1,NIMG,128] F32 current latent tokens
    text_feats: Tensor,         # [1,TEXT_TOKENS,53248] BF16
    img_zeros: Tensor,          # [1,NIMG,53248] BF16 (doubles as uncond llm)
    text_zpad: Tensor,          # [1,TEXT_TOKENS,128] F32
    tau: Float32,
    cfg: Float32,
    cond_masks: Ideogram4Masks, uncond_masks: Ideogram4Masks,
    cs0: Tensor, cs1: Tensor, ncs0: Tensor, ncs1: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    var llm = concat(1, ctx, text_feats, img_zeros)             # [1,TOTAL,53248]
    var t = Tensor.from_host([tau], [1], STDtype.F32, ctx)
    var pos_z = cast_tensor(concat(1, ctx, text_zpad, z_tok_f32), STDtype.BF16, ctx)
    var cout = ideogram4_forward_r[TOTAL](
        w, pos_z, llm, t, cond_masks, cs0, cs1,
        LAYERS, HEADS, HEAD_DIM, HIDDEN, ctx,
    )
    var v_cond = slice(cout, 1, TEXT_TOKENS, NIMG, ctx)         # [1,NIMG,128] F32
    var t2 = Tensor.from_host([tau], [1], STDtype.F32, ctx)
    var z_bf = cast_tensor(z_tok_f32, STDtype.BF16, ctx)
    var v_uncond = ideogram4_forward_r[NIMG](
        w, z_bf, img_zeros, t2, uncond_masks, ncs0, ncs1,
        LAYERS, HEADS, HEAD_DIM, HIDDEN, ctx,
    )
    return add(
        mul_scalar(v_cond, cfg, ctx),
        mul_scalar(v_uncond, Float32(1.0) - cfg, ctx),
        ctx,
    )


# ── AUTO-MASK helpers (host-side; token j owns dv[j*128 .. j*128+127]) ────────
def _accum_saliency(dv_host: List[Float32], mut sal: List[Float32]):
    for j in range(NTOK):
        var acc = Float32(0.0)
        var base = j * TOK_DIM
        for k in range(TOK_DIM):
            var v = dv_host[base + k]
            acc += v * v
        sal[j] += sqrt(acc)


def _mask_from_saliency(
    sal: List[Float32], mask_q: Float32, dilate: Int
) -> List[Bool]:
    var sorted_sal = sal.copy()
    for i in range(1, len(sorted_sal)):
        var key = sorted_sal[i]
        var j = i - 1
        while j >= 0 and sorted_sal[j] > key:
            sorted_sal[j + 1] = sorted_sal[j]
            j -= 1
        sorted_sal[j + 1] = key
    var qi = Int(mask_q * Float32(NTOK))
    if qi < 0:
        qi = 0
    if qi > NTOK - 1:
        qi = NTOK - 1
    var thr = sorted_sal[qi]
    var mask = List[Bool]()
    for j in range(NTOK):
        mask.append(sal[j] >= thr)
    for _ in range(dilate):
        var grown = mask.copy()
        for gy in range(TOK_GH):
            for gx in range(TOK_GW):
                if grown[gy * TOK_GW + gx]:
                    continue
                var hit = False
                for dy in range(-1, 2):
                    var ny = gy + dy
                    if ny < 0 or ny >= TOK_GH:
                        continue
                    for dx in range(-1, 2):
                        var nx = gx + dx
                        if nx < 0 or nx >= TOK_GW:
                            continue
                        if mask[ny * TOK_GW + nx]:
                            hit = True
                if hit:
                    grown[gy * TOK_GW + gx] = True
        mask = grown^
    return mask^


def _blend_outside_mask(
    mut z_host: List[Float32], z0_host: List[Float32], mask: List[Bool]
):
    for j in range(NTOK):
        if mask[j]:
            continue
        var base = j * TOK_DIM
        for k in range(TOK_DIM):
            z_host[base + k] = z0_host[base + k]


def _save_mask_png(
    mask: List[Bool], out_png: String, ctx: DeviceContext
) raises -> String:
    var mask_path = out_png
    if mask_path.endswith(".png"):
        mask_path = String(mask_path.removesuffix(".png"))
    mask_path += "_mask.png"
    var host = List[Float32]()
    comptime scale_h = HEIGHT // TOK_GH   # 16
    comptime scale_w = WIDTH // TOK_GW    # 16
    for _ in range(3):
        for y in range(HEIGHT):
            var gy = y // scale_h
            for x in range(WIDTH):
                var gx = x // scale_w
                if mask[gy * TOK_GW + gx]:
                    host.append(Float32(1.0))
                else:
                    host.append(Float32(0.0))
    var img = Tensor.from_host(host^, [1, 3, HEIGHT, WIDTH], STDtype.F32, ctx)
    save_png(img, mask_path, ctx, ValueRange.UNIT)
    return mask_path


# The FlowEdit ODE. Loads the SINGLE cond trunk inside (frees on return, before
# the VAE decode). Returns Z_edit tokens [1,NIMG,128] F32.
def _flowedit_denoise(
    z0_src: Tensor,             # [1,NIMG,128] F32
    text_src: Tensor, text_tgt: Tensor,
    steps: Int, n_max: Int, n_min: Int,
    src_cfg: Float32, tgt_cfg: Float32, seed: UInt64,
    auto_mask: Bool, mask_q: Float32, mask_dilate: Int, mask_warmup: Int,
    out_png: String,
    ctx: DeviceContext,
) raises -> Tensor:
    var inp = _build_inputs(ctx)
    var img_zeros = _zeros_bf16([1, NIMG, FEAT_DIM], NIMG * FEAT_DIM, ctx)
    var sec = [24, 20, 20]
    var cs = build_ideogram4_mrope(inp[0][], HEAD_DIM, sec, Float32(5000000.0), ctx, STDtype.BF16)
    var ncs = build_ideogram4_mrope(inp[2][], HEAD_DIM, sec, Float32(5000000.0), ctx, STDtype.BF16)

    print("[i4-flowedit] loading SINGLE resident fp8 trunk (cond only)...")
    var w = Ideogram4Weights.load(ShardedSafeTensors.open(String(COND)), ctx)
    var cond_masks = ideogram4_build_masks(inp[1][], ctx)
    var uncond_masks = ideogram4_build_masks(inp[3][], ctx)
    ctx.synchronize()
    var mem0 = cu_mem_get_info()
    var total_bytes = mem0.total_bytes
    var min_free = mem0.free_bytes

    var zpad_h = List[Float32](capacity=TEXT_TOKENS * 128)
    for _ in range(TEXT_TOKENS * 128):
        zpad_h.append(0.0)
    var text_zpad = Tensor.from_host(zpad_h^, [1, TEXT_TOKENS, 128], STDtype.F32, ctx)

    var mean = ideogram4_schedule_mean(HEIGHT, WIDTH, 0.0)
    var si = make_step_intervals(steps)
    var skip_before = steps - n_max      # loop counter i in [0, skip_before) SKIPPED
    var stop_at = steps - n_min          # i in [stop_at, steps) SKIPPED

    var z_edit = z0_src.clone(ctx)
    var z0_host = z0_src.to_host(ctx)
    var sal = List[Float32]()
    for _ in range(NTOK):
        sal.append(Float32(0.0))
    var active_count = 0
    var step_s_sum = 0.0
    var step_n = 0

    for i in range(steps):
        var step = steps - 1 - i         # generate.mojo counts DOWN
        var tau = ideogram4_logitnormal(Float64(si[step + 1]), mean, 1.75)
        var tau_next = ideogram4_logitnormal(Float64(si[step]), mean, 1.75)
        if i < skip_before or i >= stop_at:
            print("[i4-flowedit] step", i, " tau=", tau, " SKIPPED (window)")
            continue
        ctx.synchronize()
        var t0 = Int(perf_counter_ns())
        # 1-2) fresh noise; Zt_src = tau*Z0 + (1-tau)*N  (tau~0 = pure noise)
        var noise = randn([1, NIMG, 128], seed + UInt64(i), STDtype.F32, ctx)
        var zt_src = add(
            mul_scalar(z0_src, tau, ctx),
            mul_scalar(noise, Float32(1.0) - tau, ctx),
            ctx,
        )
        # 3) Zt_tgt = Z_edit + (Zt_src - Z0_src)
        var zt_tgt = add(z_edit, sub(zt_src, z0_src, ctx), ctx)
        # 4-5) single-trunk CFG velocities
        var v_src = _velocity(
            w, zt_src, text_src, img_zeros, text_zpad, tau, src_cfg,
            cond_masks, uncond_masks, cs[0], cs[1], ncs[0], ncs[1], ctx,
        )
        var v_tgt = _velocity(
            w, zt_tgt, text_tgt, img_zeros, text_zpad, tau, tgt_cfg,
            cond_masks, uncond_masks, cs[0], cs[1], ncs[0], ncs[1], ctx,
        )
        # 6) Z_edit += (tau_next - tau) * (V_tgt - V_src)
        var dv = sub(v_tgt, v_src, ctx)
        z_edit = add(z_edit, mul_scalar(dv, tau_next - tau, ctx), ctx)
        # 7) auto-mask: accumulate saliency; past warmup, hard-blend outside.
        if auto_mask:
            active_count += 1
            var dv_host = dv.to_host(ctx)
            _accum_saliency(dv_host, sal)
            if active_count > mask_warmup:
                var mask = _mask_from_saliency(sal, mask_q, mask_dilate)
                var masked_n = 0
                for j in range(NTOK):
                    if mask[j]:
                        masked_n += 1
                var z_host = z_edit.to_host(ctx)
                _blend_outside_mask(z_host, z0_host, mask)
                z_edit = Tensor.from_host(z_host^, [1, NIMG, 128], STDtype.F32, ctx)
                print("[i4-flowedit]   auto-mask blend: edit region =", masked_n,
                      "/", NTOK, "tokens")
        ctx.synchronize()
        var t1 = Int(perf_counter_ns())
        var mem = cu_mem_get_info()
        if mem.free_bytes < min_free:
            min_free = mem.free_bytes
        step_s_sum += Float64(t1 - t0) / 1e9
        step_n += 1
        print("[i4-flowedit] step", i, " tau=", tau, "->", tau_next,
              " STEP =", Float32(Float64(t1 - t0) / 1e9), "s (4 forwards)  free =",
              Float32(Float64(mem.free_bytes) / 1073741824.0), "GiB")

    if step_n > 0:
        print("[i4-flowedit] avg s/step =", Float32(step_s_sum / Float64(step_n)),
              " (", step_n, "active steps, 4 forwards each)")
    print("[i4-flowedit] PEAK VRAM (ODE) =",
          Float32(Float64(total_bytes - min_free) / 1073741824.0), "GiB used /",
          Float32(Float64(total_bytes) / 1073741824.0), "GiB")

    if auto_mask and active_count > 0:
        var final_mask = _mask_from_saliency(sal, mask_q, mask_dilate)
        var mask_path = _save_mask_png(final_mask, out_png, ctx)
        print("[i4-flowedit] wrote mask debug artifact:", mask_path)
    return z_edit^


# denorm + unpatch + decode (whole/tiled policy) + save + pixel MAD vs source.
def _decode_and_report(
    z: Tensor, src_path: String, out_png: String, ctx: DeviceContext
) raises:
    var ln = ShardedSafeTensors.open(String(LATENT_NORM))
    var scale = reshape(Tensor.from_view(ln.tensor_view("latent_scale"), ctx), [1, 1, 128], ctx)
    var shift = reshape(Tensor.from_view(ln.tensor_view("latent_shift"), ctx), [1, 1, 128], ctx)
    var zd = add(mul(z, scale, ctx), shift, ctx)
    var z6 = reshape(zd, [1, GH, GW, 2, 2, 32], ctx)
    var zp = permute(z6, [0, 5, 1, 3, 2, 4], ctx)
    var latent = reshape(zp, [1, 32, 2 * GH, 2 * GW], ctx)
    var latent_bf = cast_tensor(latent, STDtype.BF16, ctx)
    ctx.synchronize()
    cu_mempool_trim_current(0)
    var mem = cu_mem_get_info()
    var free_gib = Float64(mem.free_bytes) / 1073741824.0
    var img: Tensor
    if mem.free_bytes > 14 * 1024 * 1024 * 1024:
        print("[i4-flowedit] WHOLE-image decode (free =", Float32(free_gib), "GiB)")
        var dec = load_ideogram4_vae_decoder[2 * GH, 2 * GW](String(VAE), ctx)
        img = dec.decode(latent_bf, ctx)
    else:
        print("[i4-flowedit] TILED 3x3 decode (free =", Float32(free_gib), "GiB)")
        img = ideogram4_tiled_decode[2 * GH, 2 * GW](latent_bf, String(VAE), ctx)
    save_png(img, out_png, ctx)
    print("[i4-flowedit] wrote", out_png)

    # pixel MAD vs the staged source ([0,1] units — compare to the gate-1 VAE
    # roundtrip floor 0.0350 / PSNR 27.65dB; identity edits should land there).
    var imgs = ShardedSafeTensors.open(src_path)
    var src_h = Tensor.from_view(imgs.tensor_view("image"), ctx).to_host(ctx)
    var out_h = img.to_host(ctx)
    var n = len(src_h)
    var ad = 0.0
    var se = 0.0
    for i in range(n):
        var d = Float64(src_h[i] - out_h[i]) * 0.5
        ad += d if d >= 0.0 else -d
        se += d * d
    var mad = ad / Float64(n)
    var psnr = 10.0 * log(1.0 / (se / Float64(n))) / log(10.0)
    print("[i4-flowedit] output-vs-source: MAD([0,1]) =", Float32(mad),
          "  PSNR =", Float32(psnr), "dB  (VAE floor: MAD 0.0350 / 27.65dB)")


def main() raises:
    var args = argv()
    if len(args) < 5:
        raise Error(
            "usage: i4_flowedit <src_1024.safetensors> <src_prompt.json>"
            " <tgt_prompt.json> <out.png> [--steps 28] [--nmax 24] [--nmin 0]"
            " [--src-cfg 1.5] [--tgt-cfg 5.0] [--seed N] [--auto-mask]"
            " [--mask-q 0.7] [--mask-dilate 1] [--mask-warmup 4]"
        )
    var src_path = String(args[1])
    var src_prompt_path = String(args[2])
    var tgt_prompt_path = String(args[3])
    var out_png = String(args[4])

    var steps = STEPS_DEFAULT
    var n_max = NMAX_DEFAULT
    var n_min = NMIN_DEFAULT
    var src_cfg = SRC_CFG_DEFAULT
    var tgt_cfg = TGT_CFG_DEFAULT
    var seed = SEED_DEFAULT
    var auto_mask = False
    var mask_q = Float32(0.7)
    var mask_dilate = 1
    var mask_warmup = 4
    for i in range(len(args)):
        if args[i] == String("--steps") and i + 1 < len(args):
            steps = Int(String(args[i + 1]))
        if args[i] == String("--nmax") and i + 1 < len(args):
            n_max = Int(String(args[i + 1]))
        if args[i] == String("--nmin") and i + 1 < len(args):
            n_min = Int(String(args[i + 1]))
        if args[i] == String("--src-cfg") and i + 1 < len(args):
            src_cfg = Float32(Float64(String(args[i + 1])))
        if args[i] == String("--tgt-cfg") and i + 1 < len(args):
            tgt_cfg = Float32(Float64(String(args[i + 1])))
        if args[i] == String("--seed") and i + 1 < len(args):
            seed = UInt64(Int(String(args[i + 1])))
        if args[i] == String("--auto-mask"):
            auto_mask = True
        if args[i] == String("--mask-q") and i + 1 < len(args):
            mask_q = Float32(Float64(String(args[i + 1])))
        if args[i] == String("--mask-dilate") and i + 1 < len(args):
            mask_dilate = Int(String(args[i + 1]))
        if args[i] == String("--mask-warmup") and i + 1 < len(args):
            mask_warmup = Int(String(args[i + 1]))
    if n_max > steps:
        n_max = steps
    if n_min < 0:
        n_min = 0

    var ctx = DeviceContext()
    print("[i4-flowedit] FlowEdit ideogram4 1024x1024  steps=", steps,
          " n_max=", n_max, " n_min=", n_min, " src_cfg=", src_cfg,
          " tgt_cfg=", tgt_cfg, " seed=", seed)
    if auto_mask:
        print("[i4-flowedit] AUTO-MASK on: q=", mask_q, " dilate=", mask_dilate,
              " warmup=", mask_warmup, " active steps")

    # 1) prompts FIRST (streamed TE, one disk pass), then trim.
    var texts = _encode_prompt_pair(src_prompt_path, tgt_prompt_path, ctx)
    ctx.synchronize()
    cu_mempool_trim_current(0)

    # 2) source image -> Z0_src tokens; encoder freed by scope, trim again.
    var z0_src = _encode_source_tokens(src_path, ctx)
    ctx.synchronize()
    cu_mempool_trim_current(0)

    # 3) FlowEdit ODE on the single trunk.
    var z_edit = _flowedit_denoise(
        z0_src, texts[0][], texts[1][],
        steps, n_max, n_min, src_cfg, tgt_cfg, seed,
        auto_mask, mask_q, mask_dilate, mask_warmup, out_png, ctx,
    )
    ctx.synchronize()
    cu_mempool_trim_current(0)

    # 4) decode + metrics.
    _decode_and_report(z_edit, src_path, out_png, ctx)
