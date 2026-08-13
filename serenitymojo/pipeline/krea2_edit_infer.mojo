# pipeline/krea2_edit_infer.mojo — Krea-2 image-EDIT (reference-conditioned) inference.
#
# Task #9: RUN the trained krea2 image-EDIT adapter (photo restore/colorize). This is
# the krea2 T2I pipeline (pipeline/krea2_pipeline.mojo) EXTENDED with the additive
# img_in_ref reference conditioning that the edit trainer added at the `first`
# projection:
#     img = first(noise_tokens) + img_in_ref(ref_tokens)
# where ref_tokens = the SOURCE image, VAE-encoded (encode_mean -> (z-mean)/std) and
# patchified to [1, IMGLEN, 64] — the SAME normalized-latent contract the edit cache
# stored (krea2_prepare_cache._encode_one_latent) and the trainer consumed
# (train_krea2 cache.load_ref -> krea2_patchify). img_in_ref.weight [6144,64] is the
# trained additive projection. The hook lives in models/dit/krea2_dit.krea2_forward
# (gated on ref_tokens+img_in_ref_w; absent => byte-identical text-to-image).
#
# The denoise STARTS FROM NOISE (t=1 -> 0), conditioned on the source ONLY via the
# additive img_in_ref term — exactly how it was trained (target = the restored image;
# the source enters through img_in_ref, NOT as an img2img init). 512x512 = the
# training resolution (edit config LFULL=1152, LTMAX=128 -> IMGLEN=1024 -> LH=LW=64).
#
# TWO-PROCESS SPLIT (same as krea2_pipeline): run krea2_encode_cli FIRST to dump the
# pos/neg contexts (the ~8GB Qwen3-VL TE must not co-reside with the streamed DiT).
#
# Run (guarded, GPU):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg
#   # 1) encode the edit-instruction prompt (writes demo/ctx_{pos,neg}.bin):
#   pixi run mojo run -I . serenitymojo/pipeline/krea2_encode_cli.mojo \
#       "restore and colorize this photo" "" \
#       <demo>/ctx_pos.bin <demo>/ctx_neg.bin
#   # 2) stage the source (Python; Mojo has no image decoder):
#   <py> serenitymojo/pipeline/krea2_stage_one.py <src.png> <demo>/source_512.safetensors 512
#   # 3) EDIT inference (LoRA + img_in_ref):
#   pixi run mojo run -I . [BUILD FLAGS] serenitymojo/pipeline/krea2_edit_infer.mojo \
#       <src_512.safetensors> <ctx_pos.bin> <ctx_neg.bin> <out.png> \
#       [--lora <lora.safetensors>] [--img-in-ref <img_in_ref.safetensors>] [--no-edit]
#   --no-edit -> baseline: NO lora, NO ref conditioning (pure t2i from the same prompt).
#
# Mojo 1.0.0b1, NVIDIA GPU. Inference-only.

from max.gpu.host import DeviceContext
from std.collections import Optional
from std.sys import argv
from std.time import perf_counter_ns

from serenitymojo.lora import LoraSet
from serenitymojo.models.krea2.krea2_stack import (
    build_krea2_resident_fp8, build_krea2_resident_int8,
)
from serenitymojo.models.dit.krea2_dit import (
    Krea2ResidentFp8, Krea2ResidentInt8, Krea2HostInt8Inf,
    Krea2LoraSideCache, build_krea2_lora_side_cache,
    Krea2SharedResident, build_krea2_shared_resident,
    build_krea2_host_int8_inf, krea2_forward,
)
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current
# int8 DISK SIDECAR (task #13): cache the load-once int8 quantize of the 28
# blocks next to the checkpoint so later launches skip the ~35s quantize.
from serenitymojo.models.krea2.krea2_int8_cache import (
    krea2_int8_cache_path, krea2_int8_cache_valid,
    load_krea2_int8_cache_resident, load_krea2_int8_cache_host,
    load_krea2_int8_cache_shared, save_krea2_int8_cache,
)
from serenitymojo.models.krea2.krea2_cache_reader import krea2_patchify
from serenitymojo.models.krea2.krea2_img_in_ref_param import (
    load_krea2_img_in_ref_resume,
    krea2_img_in_ref_w_device,
)
from serenitymojo.models.krea2.krea2_prepare_cache import (
    _mean_ch,
    _std_ch,
    _normalize_latent,
    KREA2_VAE_ENC_FILE,
)
from serenitymojo.models.vae.qwenimage_encoder import QwenImageVaeEncoder
from serenitymojo.models.vae.qwenimage_decoder import QwenImageVaeDecoder

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.cap_cache import load_tensor_bin
from serenitymojo.ops.tensor_algebra import reshape, permute, mul_scalar, add
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.torch_bf16 import torch_f32_to_bf16_rne
from serenitymojo.ops.random import randn
from serenitymojo.sampling.krea2_sampler import (
    krea2_timesteps,
    krea2_packed_seq_len,
    krea2_cfg,
    krea2_euler_step,
)
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.pipeline.krea2_paths import (
    KREA2_RAW,
    KREA2_VAE_DIR,
    KREA2_RAW_KEY_PREFIX,
)


# ── RUN CONFIG (512x512 = the edit training resolution) ───────────────────────
comptime HEIGHT = 512
comptime WIDTH = 512
comptime LH = HEIGHT // 8      # 64
comptime LW = WIDTH // 8       # 64
comptime IMGLEN = (LH // 2) * (LW // 2)   # 32*32 = 1024 (gh*gw)
comptime FEATURES = 6144
comptime OUT_CH = 64
comptime STEPS_DEFAULT = 24
comptime CFG_DEFAULT = Float32(4.0)
comptime SEED = UInt64(88888)

# Prompt-dependent NATURAL token lengths (measured by krea2_encode_cli for the
# "restore and colorize this photo" prompt + empty negative). FAIL-LOUD if a loaded
# context's LT differs.
comptime LT_POS = 11
comptime LT_NEG = 5
comptime LFULL_POS = LT_POS + IMGLEN       # 1035
comptime LFULL_NEG = LT_NEG + IMGLEN       # 1029
comptime LPAD_POS = ((LFULL_POS + 255) // 256) * 256   # 1280
comptime LPAD_NEG = ((LFULL_NEG + 255) // 256) * 256   # 1280
comptime NBLOCKS = 28

comptime DEFAULT_LORA = String(
    "/home/alex/trainings/krea2_edit_qwen/krea2_edit_3000.safetensors"
)
comptime DEFAULT_IMG_IN_REF = String(
    "/home/alex/trainings/krea2_edit_qwen/krea2_edit_3000_img_in_ref.safetensors"
)


def _load_context(
    path: String, expect_lt: Int, name: String, ctx: DeviceContext
) raises -> Tensor:
    var stack = load_tensor_bin(path, ctx)          # [1, LT, 12, 2560] bf16
    var sh = stack.shape()
    if len(sh) != 4 or sh[0] != 1 or sh[2] != 12 or sh[3] != 2560:
        raise Error(
            String("[krea2-edit] ") + name + " cached context has unexpected shape "
            + "(expected [1, LT, 12, 2560]); regenerate with krea2_encode_cli: " + path
        )
    var lt = sh[1]
    print("[krea2-edit] ", name, " context LT =", lt, " (expect ", expect_lt, ")")
    if lt != expect_lt:
        raise Error(
            String("[krea2-edit] ") + name + " LT=" + String(lt)
            + " != comptime expectation " + String(expect_lt)
            + " — update LT_" + name + " for this prompt (krea2_encode_cli prints it)."
        )
    return stack^


def _build_pos(lt: Int, ctx: DeviceContext) raises -> Tensor:
    """pos [1, lt+IMGLEN, 3] f32 = cat(txt zeros, img (0,gh_i,gw_i) grid)."""
    var gh = LH // 2
    var gw = LW // 2
    var host = List[Float32]()
    for _ in range(lt * 3):
        host.append(Float32(0.0))
    for hi in range(gh):
        for wi in range(gw):
            host.append(Float32(0.0))
            host.append(Float32(hi))
            host.append(Float32(wi))
    var lfull = lt + IMGLEN
    return Tensor.from_host(host^, [1, lfull, 3], STDtype.F32, ctx)


def _patchify_latent(latent_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
    """[1,16,LH,LW] -> [1, IMGLEN, 64] (the SAME (c ph pw) patchify as the pipeline /
    krea2_patchify). ph=pw=2."""
    var gh = LH // 2
    var gw = LW // 2
    var x6 = reshape(latent_nchw, [1, 16, gh, 2, gw, 2], ctx)
    var xp = permute(x6, [0, 2, 4, 1, 3, 5], ctx)   # [1, gh, gw, 16, 2, 2]
    return reshape(xp, [1, gh * gw, 64], ctx)        # [1, IMGLEN, 64]


def _unpatch(tokens: Tensor, ctx: DeviceContext) raises -> Tensor:
    var gh = LH // 2
    var gw = LW // 2
    var x6 = reshape(tokens, [1, gh, gw, 16, 2, 2], ctx)
    var xp = permute(x6, [0, 3, 1, 4, 2, 5], ctx)   # [1, 16, gh, 2, gw, 2]
    return reshape(xp, [1, 16, gh * 2, gw * 2], ctx)  # [1, 16, LH, LW]


def _encode_source_ref(src_path: String, ctx: DeviceContext) raises -> Tensor:
    """Source PNG (staged [1,3,512,512] f32) -> normalized VAE latent [1,16,LH,LW]
    BF16 -> patchified ref tokens [1, IMGLEN, 64] BF16. Mirrors EXACTLY the edit
    cache's ref.<i> production (krea2_prepare_cache._encode_one_latent: cast BF16 ->
    encode_mean -> F32 -> (z-mean)/std -> BF16) then krea2_patchify."""
    var imgs = ShardedSafeTensors.open(src_path)
    var img_f32 = Tensor.from_view(imgs.tensor_view(String("image")), ctx)  # [1,3,512,512]
    var img = cast_tensor(img_f32, STDtype.BF16, ctx)
    ctx.synchronize(); var _t0 = Int(perf_counter_ns())
    var enc = QwenImageVaeEncoder[HEIGHT, WIDTH].load(KREA2_VAE_ENC_FILE, ctx)
    ctx.synchronize(); var _t1 = Int(perf_counter_ns())
    print("[krea2-edit] PHASE vae-enc-load =", Float64(_t1 - _t0) / 1e9, "s")
    var lat_mean = enc.encode_mean(img, ctx)                # [1,16,LH,LW] BF16
    ctx.synchronize()
    print("[krea2-edit] PHASE vae-encode =",
          Float64(Int(perf_counter_ns()) - _t1) / 1e9, "s")
    var lat_f32 = cast_tensor(lat_mean, STDtype.F32, ctx)
    var mean_ch = _mean_ch(ctx)
    var std_ch = _std_ch(ctx)
    var clean_f32 = _normalize_latent(lat_f32, mean_ch, std_ch, ctx)  # (z-mean)/std
    var clean_bf16 = cast_tensor(clean_f32, STDtype.BF16, ctx)        # [1,16,LH,LW]
    var ref_tok = krea2_patchify[LH, LW](clean_bf16, ctx)             # [1,IMGLEN,64] bf16
    print("[krea2-edit] source VAE-encoded ref tokens =", ref_tok.shape()[0],
          ref_tok.shape()[1], ref_tok.shape()[2])
    return ref_tok^


def main() raises:
    var _t_launch = Int(perf_counter_ns())   # startup (launch -> denoise) timer
    var ctx = DeviceContext()
    var args = argv()

    # positional: <src_512.safetensors> <ctx_pos.bin> <ctx_neg.bin> <out.png>
    if len(args) < 5:
        raise Error(
            "usage: krea2_edit_infer <src_512.safetensors> <ctx_pos.bin> "
            "<ctx_neg.bin> <out.png> [--lora <p>] [--img-in-ref <p>] [--no-edit] "
            "[--steps N] [--cfg F] [--img2img F] [--stream|--resident] "
            "(default base = int8 W8A8-resident)"
        )
    var src_path = String(args[1])
    var ctx_pos_path = String(args[2])
    var ctx_neg_path = String(args[3])
    var out_png = String(args[4])

    var lora_path = String(DEFAULT_LORA)
    var img_in_ref_path = String(DEFAULT_IMG_IN_REF)
    var no_edit = False
    var steps = STEPS_DEFAULT
    var cfg_scale = CFG_DEFAULT
    var img2img = Float32(0.0)   # 0.0 => pure-noise init (byte-identical to before)
    for i in range(len(args)):
        if args[i] == String("--lora") and i + 1 < len(args):
            lora_path = String(args[i + 1])
        if args[i] == String("--img-in-ref") and i + 1 < len(args):
            img_in_ref_path = String(args[i + 1])
        if args[i] == String("--no-edit"):
            no_edit = True
        if args[i] == String("--cfg") and i + 1 < len(args):
            cfg_scale = Float32(Float64(String(args[i + 1])))
        if args[i] == String("--steps") and i + 1 < len(args):
            steps = Int(String(args[i + 1]))
        if args[i] == String("--img2img") and i + 1 < len(args):
            img2img = Float32(Float64(String(args[i + 1])))

    print("[krea2-edit] EDIT inference", HEIGHT, "x", WIDTH, " steps=", steps,
          " cfg=", cfg_scale, " no_edit=", no_edit)

    var _t_ph = Int(perf_counter_ns())

    # ── 1) source -> ref tokens (encode + free the encoder BEFORE the resident base).
    var ref_opt = Optional[Tensor](None)
    if not no_edit:
        ref_opt = Optional[Tensor](_encode_source_ref(src_path, ctx))
    ctx.synchronize()
    print("[krea2-edit] PHASE source-encode =",
          Float64(Int(perf_counter_ns()) - _t_ph) / 1e9, "s")
    _t_ph = Int(perf_counter_ns())

    # ── 2) img_in_ref trained projection [6144,64] (F32 device). ──────────────
    var img_in_ref_opt = Optional[Tensor](None)
    if not no_edit:
        var iir_param = load_krea2_img_in_ref_resume(
            FEATURES, OUT_CH, img_in_ref_path, ctx
        )
        img_in_ref_opt = Optional[Tensor](krea2_img_in_ref_w_device(iir_param, ctx))
        print("[krea2-edit] img_in_ref loaded:", img_in_ref_path)

    # ── 3) LoRA overlay. ───────────────────────────────────────────────────────
    var lora = Optional[LoraSet](None)
    var lora_mult = Float32(1.0)
    if not no_edit:
        lora = Optional[LoraSet](LoraSet.load(lora_path))
        print("[krea2-edit] LoRA:", lora_path, " adapters=", lora.value().num_mappings())

    # ── 4) contexts + pos grids. ──────────────────────────────────────────────
    var ctx_pos = _load_context(ctx_pos_path, LT_POS, String("POS"), ctx)
    var ctx_neg = _load_context(ctx_neg_path, LT_NEG, String("NEG"), ctx)
    var pos_p = _build_pos(LT_POS, ctx)
    var pos_n = _build_pos(LT_NEG, ctx)

    ctx.synchronize()
    print("[krea2-edit] PHASE iir+lora+ctx-load =",
          Float64(Int(perf_counter_ns()) - _t_ph) / 1e9, "s")
    _t_ph = Int(perf_counter_ns())

    # ── 5) noise latent [1,16,LH,LW] F32 accumulator. ─────────────────────────
    var latent = randn([1, 16, LH, LW], SEED, STDtype.F32, ctx)

    # ── 6) schedule (t: 1 -> 0). ──────────────────────────────────────────────
    var seq = krea2_packed_seq_len(HEIGHT, WIDTH)
    var ts = krea2_timesteps(seq, steps)
    print("[krea2-edit] schedule seq =", seq, " steps =", len(ts) - 1,
          " ts[0]=", ts[0], " ts[-1]=", ts[len(ts) - 1])

    # ── 6b) img2img init (opt-in). d>0: start the denoise from a NOISED version of
    # the SOURCE latent (structure-preserving) instead of pure noise. The source
    # latent is the same normalized [1,16,LH,LW] space as the denoise carrier —
    # unpatchify the already-encoded ref tokens and cast to F32. Find start_si =
    # the first schedule index where ts descends to <= d, and init the latent as
    # x = (1 - t_start)*source + t_start*noise (flow-matching: x_t interpolates
    # clean at t=0, noise at t=1). Loop then runs from start_si (skip high-t steps).
    var start_si = 0
    if img2img > Float32(0.0) and not no_edit:
        # first schedule index where ts[start_si] <= d (ts descends from ~1)
        for si in range(steps):
            if ts[si] <= img2img:
                start_si = si
                break
        var t_start = ts[start_si]
        # SOURCE latent in the denoise space: clone reusable ref tokens, unpatch,
        # cast to F32.  ref_opt is [1,IMGLEN,64] bf16.
        var src_tok = ref_opt.value().clone(ctx)                # [1,IMGLEN,64] bf16
        var src_latent_bf16 = _unpatch(src_tok, ctx)            # [1,16,LH,LW] bf16
        var src_latent = cast_tensor(src_latent_bf16, STDtype.F32, ctx)  # F32
        # x = (1 - t_start)*source + t_start*noise  (latent currently == noise)
        var src_scaled = mul_scalar(src_latent, Float32(1.0) - t_start, ctx)
        var noise_scaled = mul_scalar(latent, t_start, ctx)
        latent = add(src_scaled, noise_scaled, ctx)
        print("[krea2-edit] img2img start_si=", start_si, " t=", t_start,
              " strength=", img2img)

    # ── 7) DiT base weight source. THREE modes:
    #   DEFAULT = int8 W8A8 HYBRID (task #11): load-once tensorwise int8 quantize of
    #     all 28 blocks — the FIRST K device-resident (~433MB/block), the remainder
    #     PINNED-HOST int8 H2D'd per block per forward (fully-resident is ~12.1GB
    #     and MEASURED OOM on 16GB next to the pipeline baseline). The trainer's
    #     int8 GEMM per matmul = the dtype path the edit adapter was TRAINED against
    #     (cfg.quantized_resident=="int_w8a8"); LoRA rides as a bf16 side-branch
    #     (== training _linear_lora). --i8-resident-blocks N tunes K (default 14).
    #   --stream = per-block DISK STREAM bf16 base (the old default; ~62s/step but
    #     full-bf16 GEMM numerics, no quantization).
    #   --resident = fp8-resident base (~12GB; OOMs on 16GB with LoRA — bigger cards).
    var st = ShardedSafeTensors.open(String(KREA2_RAW))
    var use_resident_fp8 = False
    var use_stream = False
    var i8_res_blocks = 20
    for i in range(len(args)):
        if args[i] == String("--resident"):
            use_resident_fp8 = True
        if args[i] == String("--stream"):
            use_stream = True
        if args[i] == String("--int8-resident"):
            use_stream = False   # explicit opt-in == the default
        if args[i] == String("--i8-resident-blocks") and i + 1 < len(args):
            i8_res_blocks = Int(String(args[i + 1]))
    var resident = Optional[Krea2ResidentFp8](None)
    var resident_i8 = Optional[Krea2ResidentInt8](None)
    var host_i8 = Optional[Krea2HostInt8Inf](None)
    # SHARED (non-block) resident weights — filled from the int8 sidecar on the
    # fast path, else built from the checkpoint at 7c (see below).
    var shared = Optional[Krea2SharedResident](None)
    if use_resident_fp8:
        print("[krea2-edit] building fp8-resident base (28 blocks) ...")
        resident = Optional[Krea2ResidentFp8](
            build_krea2_resident_fp8(st, KREA2_RAW_KEY_PREFIX, NBLOCKS, NBLOCKS, ctx)
        )
        print("[krea2-edit] fp8-resident base DONE.")
    elif use_stream:
        print("[krea2-edit] per-block DISK STREAM base (bf16 GEMMs, slow).")
    else:
        if i8_res_blocks < 0:
            i8_res_blocks = 0
        if i8_res_blocks > NBLOCKS:
            i8_res_blocks = NBLOCKS
        # Reclaim the VAE-encode/setup transients BEFORE the big quantize so the
        # resident store builds against a clean baseline.
        ctx.synchronize()
        cu_mempool_trim_current(0)
        # DISK SIDECAR (task #13): the int8 bytes + scalar scales are a pure
        # function of the FROZEN checkpoint, so quantize ONCE and reload the
        # sidecar on later launches instead of the ~35s re-quantize. The sidecar
        # stores all 28 blocks uniformly — valid at ANY --i8-resident-blocks K.
        var i8_cache_path = krea2_int8_cache_path(String(KREA2_RAW))
        var i8_from_cache = False
        if krea2_int8_cache_valid(i8_cache_path, String(KREA2_RAW), NBLOCKS):
            print("[int8cache] fresh sidecar found:", i8_cache_path)
            if i8_res_blocks > 0:
                resident_i8 = Optional[Krea2ResidentInt8](
                    load_krea2_int8_cache_resident(i8_cache_path, i8_res_blocks, ctx)
                )
            if i8_res_blocks < NBLOCKS:
                host_i8 = Optional[Krea2HostInt8Inf](
                    load_krea2_int8_cache_host(
                        i8_cache_path, NBLOCKS, i8_res_blocks, ctx
                    )
                )
            shared = Optional[Krea2SharedResident](
                load_krea2_int8_cache_shared(i8_cache_path, ctx)
            )
            i8_from_cache = True
            print("[int8cache] loaded", NBLOCKS,
                  "blocks + shared from sidecar (skipped quantize).")
        if not i8_from_cache:
            print("[krea2-edit] building int8 W8A8 base: resident", i8_res_blocks,
                  "/", NBLOCKS, "blocks + pinned-host remainder (load-once) ...")
            if i8_res_blocks > 0:
                resident_i8 = Optional[Krea2ResidentInt8](
                    build_krea2_resident_int8(
                        st, KREA2_RAW_KEY_PREFIX, NBLOCKS, i8_res_blocks, ctx
                    )
                )
            if i8_res_blocks < NBLOCKS:
                host_i8 = Optional[Krea2HostInt8Inf](
                    build_krea2_host_int8_inf(
                        st, KREA2_RAW_KEY_PREFIX, NBLOCKS, i8_res_blocks, ctx
                    )
                )
            # Build the shared store NOW (normally 7c's job) so it rides in the
            # sidecar too — a cache launch then never opens the bf16 checkpoint's
            # big tensors at all (7c skips when `shared` is already set).
            shared = Optional[Krea2SharedResident](
                build_krea2_shared_resident(st, KREA2_RAW_KEY_PREFIX, ctx)
            )
            # Persist for future fast launches. Non-fatal: a read-only checkpoint
            # dir / disk-full just means no cache this time.
            try:
                save_krea2_int8_cache(
                    resident_i8, host_i8, shared.value(), String(KREA2_RAW),
                    i8_cache_path, NBLOCKS, ctx,
                )
                print("[int8cache] wrote sidecar for future fast launches:",
                      i8_cache_path)
            except e:
                print("[int8cache] WARN could not write sidecar (continuing):",
                      String(e))
        print("[krea2-edit] int8 base DONE (resident", i8_res_blocks, "+ host",
              NBLOCKS - i8_res_blocks, ").")
        ctx.synchronize()
        print("[krea2-edit] PHASE int8-base =",
              Float64(Int(perf_counter_ns()) - _t_ph) / 1e9, "s")
        _t_ph = Int(perf_counter_ns())

    # ── 7b) int8-path LoRA side cache: device-resident A/B/scale, built ONCE (the
    # per-forward mmap+alpha-probe fallback was MEASURED at ~4s of the step).
    var lora_cache = Optional[Krea2LoraSideCache](None)
    if (Bool(resident_i8) or Bool(host_i8)) and Bool(lora):
        lora_cache = Optional[Krea2LoraSideCache](
            build_krea2_lora_side_cache(lora.value(), lora_mult, ctx)
        )
        print("[krea2-edit] LoRA side cache resident:",
              len(lora_cache.value().base_keys), "modules")

    # ── 7c) SHARED (non-block) weights, load-once resident (~1.3GB device): the
    # per-forward disk reload of first/tmlp/tproj/txtfusion/txtmlp/last was
    # MEASURED at ~4.4s/step of single-threaded host byte-copy+cast (tproj alone
    # is 906MB F32 on disk). int8 path only (--stream keeps the old behavior as
    # the byte-identical reference arm).
    if (Bool(resident_i8) or Bool(host_i8)) and not Bool(shared):
        shared = Optional[Krea2SharedResident](
            build_krea2_shared_resident(st, KREA2_RAW_KEY_PREFIX, ctx)
        )
        print("[krea2-edit] shared weights resident (load-once).")

    ctx.synchronize()
    print("[krea2-edit] PHASE loracache+shared =",
          Float64(Int(perf_counter_ns()) - _t_ph) / 1e9, "s")
    print("[krea2-edit] STARTUP (launch -> denoise) =",
          Float64(Int(perf_counter_ns()) - _t_launch) / 1e9, "s")

    # ── 8) Euler integration with CFG + (opt-in) img_in_ref reference conditioning.
    for si in range(start_si, steps):
        var t_cur = ts[si]
        var t_prev = ts[si + 1]
        var t_t = Tensor.from_host([t_cur], [1], STDtype.F32, ctx)

        var img_tokens_f32 = _patchify_latent(latent, ctx)             # [1,IMGLEN,64] f32
        var img_tokens = torch_f32_to_bf16_rne(img_tokens_f32, ctx)    # bf16 feed

        ctx.synchronize(); var _f0 = Int(perf_counter_ns())
        var v_cond = krea2_forward[LFULL_POS, LPAD_POS, LT_POS, NBLOCKS](
            st, img_tokens, ctx_pos, t_t, pos_p, ctx, KREA2_RAW_KEY_PREFIX,
            lora, lora_mult, resident, ref_opt, img_in_ref_opt, resident_i8,
            host_i8, lora_cache, shared,
        )
        var v_uncond = krea2_forward[LFULL_NEG, LPAD_NEG, LT_NEG, NBLOCKS](
            st, img_tokens, ctx_neg, t_t, pos_n, ctx, KREA2_RAW_KEY_PREFIX,
            lora, lora_mult, resident, ref_opt, img_in_ref_opt, resident_i8,
            host_i8, lora_cache, shared,
        )
        var v_bf16 = krea2_cfg(v_cond, v_uncond, cfg_scale, ctx)
        var v_f32 = cast_tensor(v_bf16, STDtype.F32, ctx)              # [1,IMGLEN,64] f32
        var v_latent = _unpatch(v_f32, ctx)                            # [1,16,LH,LW] f32
        latent = krea2_euler_step(latent, v_latent, t_cur, t_prev, ctx)
        ctx.synchronize(); var _f1 = Int(perf_counter_ns())
        print("[krea2-edit] step", si, " t=", t_cur, " STEP=", Float64(_f1 - _f0)/1e9, "s")

    # ── 9) VAE decode -> PNG. ─────────────────────────────────────────────────
    print("[krea2-edit] decoding latent -> image ...")
    var dec = QwenImageVaeDecoder[LH, LW].load(String(KREA2_VAE_DIR), ctx)
    var latent_bf16 = torch_f32_to_bf16_rne(latent, ctx)
    var image = dec.decode(latent_bf16, ctx)                           # [1,3,H,W] [-1,1]
    save_png(image, out_png, ctx, ValueRange.SIGNED)
    print("[krea2-edit] wrote", out_png)
