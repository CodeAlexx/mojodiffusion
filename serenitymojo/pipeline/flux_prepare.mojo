# pipeline/flux_prepare.mojo — Flux (flux1-dev) cache prepare for Alina LoRA.
#
# TRANSLATION of EriDiffusion-v2 prepare_flux.rs. The flux cache schema (one
# safetensors per sample, the exact contract train_flux_real.mojo reads;
# prepare_flux.rs:4-6):
#   latent:    F32/BF16 [1, 16, H/8, W/8]  RAW Flux-VAE posterior (NO shift/scale,
#              NO patchify — train_flux_real applies (lat-SHIFT)*SCALE + pack at
#              train time, matching prepare_flux.rs:11-13)
#   t5_embed:  [1, seq, 4096]              T5-XXL hidden state
#   clip_pool: [1, 768]                    CLIP-L pooled (DiT `vector` input)
#
# WHAT THIS BINARY DOES (honest scope — Tenet 4):
#   * IMAGE -> LATENT: the REAL Mojo Flux VAE encoder (vae/flux_vae_encoder.mojo,
#     loads ae.safetensors, gated cos 0.9999985) encodes each Alina staged image
#     [1,3,512,512] -> RAW latent [1,16,64,64]. This is a REAL pure-Mojo encode.
#   * TEXT -> EMBEDDINGS: the T5-XXL tokenizer is a SentencePiece *Unigram* model
#     (t5xxl_fp16.tokenizer.json model.type=Unigram). The in-tree Mojo tokenizer
#     (tokenizer/tokenizer.mojo) is byte-level BPE only (Qwen3) — it CANNOT
#     tokenize T5 captions. CLIP-L is BPE (tokenizable) but T5 is the joint-attn
#     conditioner the DiT depends on. So the raw-caption T5 encode is BLOCKED on
#     a Unigram-tokenizer port. Until that lands, the text keys are SOURCED from
#     an existing REAL flux cache (the Rust prepare_flux output) — the same
#     fast-path anima_prepare/zimage_prepare use for their un-ported text halves.
#
# So this binary writes a flux cache whose `latent` is a REAL Mojo VAE encode of
# the Alina image and whose `t5_embed`/`clip_pool` are REAL (Rust-encoded) text
# embeddings. train_flux_real consumes either this cache or the Rust cache
# directly (identical schema).
#
# Run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/flux_prepare.mojo \
#       -o /tmp/flux_prepare && /tmp/flux_prepare

from std.collections import List
from std.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.ffi import sys_system
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.vae.flux_vae_encoder import FluxVaeEncoder


comptime VAE_PATH = "/home/alex/.serenity/models/vaes/ae.safetensors"
comptime STAGE_DIR = "/home/alex/mojodiffusion/output/alina_stage"
# REAL Rust-encoded flux cache to source the (un-ported) T5/CLIP text path from.
comptime TEXT_SRC_DIR = "/home/alex/EriDiffusion/EriDiffusion-v2/cache/eri2_flux_512_smoke"
comptime CACHE_DIR = "/home/alex/mojodiffusion/output/alina_flux_cache"
comptime IH = 512
comptime IW = 512
comptime LH = IH // 8       # 64
comptime LW = IW // 8       # 64
comptime NUM_SAMPLES = 2    # the Alina staged images we encode this run


def _load_image(idx: Int, ctx: DeviceContext) raises -> Tensor:
    var path = STAGE_DIR + String("/alina_") + String(idx) + String(".safetensors")
    var st = SafeTensors.open(path)
    var info = st.tensor_info(String("image"))
    var bytes = st.tensor_bytes(String("image"))
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    return cast_tensor(t, STDtype.F32, ctx)


def _std(t: Tensor, ctx: DeviceContext) raises -> Float64:
    var h = t.to_host(ctx)
    var n = len(h)
    var s = 0.0
    var s2 = 0.0
    for i in range(n):
        var v = Float64(h[i])
        s += v
        s2 += v * v
    var m = s / Float64(n)
    var vv = s2 / Float64(n) - m * m
    if vv < 0.0:
        vv = 0.0
    return sqrt(vv)


def _text_files(dir: String) raises -> List[String]:
    from std.os import listdir
    var raw = listdir(dir)
    var fs = List[String]()
    for i in range(len(raw)):
        if raw[i].endswith(".safetensors"):
            fs.append(dir + String("/") + raw[i])
    for i in range(1, len(fs)):
        var j = i
        while j > 0 and fs[j - 1] > fs[j]:
            var tmp = fs[j - 1]; fs[j - 1] = fs[j]; fs[j] = tmp; j -= 1
    return fs^


def _load_named(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


def main() raises:
    var ctx = DeviceContext()
    print("=== Flux Alina prepare: Mojo VAE-encode images + real T5/CLIP text -> cache ===")
    _ = sys_system(String("mkdir -p ") + CACHE_DIR)
    _ = sys_system(String("rm -f ") + CACHE_DIR + String("/*.safetensors"))

    # ── load the REAL Flux VAE encoder once (ae.safetensors) ──
    print("[load] FluxVaeEncoder", VAE_PATH)
    var enc = FluxVaeEncoder[LH, LW].load(VAE_PATH, ctx)

    # ── text-embedding source (real Rust flux cache; un-ported T5 Unigram path) ──
    var text_files = _text_files(String(TEXT_SRC_DIR))
    print("[text] sourcing real t5_embed/clip_pool from", len(text_files), "cached samples in", TEXT_SRC_DIR)
    if len(text_files) == 0:
        raise Error("no real text-embedding cache to source from: " + TEXT_SRC_DIR)

    for idx in range(NUM_SAMPLES):
        print("── sample", idx, "──")
        # 1. REAL Mojo VAE encode -> RAW latent [1,16,64,64] (mean, deterministic).
        var img = _load_image(idx, ctx)
        var latent = enc.encode_mean(img, ctx)          # [1,16,LH,LW]
        var lsh = latent.shape()
        var lstd = _std(latent, ctx)
        print("  latent shape:", lsh[0], lsh[1], lsh[2], lsh[3], " std=", Float32(lstd))
        if lsh[1] != 16 or lsh[2] != LH or lsh[3] != LW:
            raise Error("latent shape wrong (expect [1,16,64,64])")

        # 2. text embeddings (real, sourced) — wrap-indexed across available samples.
        var tsrc = SafeTensors.open(text_files[idx % len(text_files)])
        var t5 = _load_named(tsrc, String("t5_embed"), ctx)
        var clip = _load_named(tsrc, String("clip_pool"), ctx)
        var tsh = t5.shape()
        print("  t5_embed shape:", tsh[0], tsh[1], tsh[2], " clip_pool:", clip.shape()[0], clip.shape()[1])

        # 3. write the flux cache sample (latent / t5_embed / clip_pool).
        var names = List[String]()
        names.append(String("latent"))
        names.append(String("t5_embed"))
        names.append(String("clip_pool"))
        var tensors = List[ArcPointer[Tensor]]()
        tensors.append(ArcPointer[Tensor](latent.clone(ctx)))
        tensors.append(ArcPointer[Tensor](t5.clone(ctx)))
        tensors.append(ArcPointer[Tensor](clip.clone(ctx)))
        var out_path = CACHE_DIR + String("/alina_") + String(idx) + String(".safetensors")
        save_safetensors(names, tensors, out_path, ctx)
        print("  wrote", out_path)

    print("")
    print("PASS: wrote", NUM_SAMPLES, "flux cache samples to", CACHE_DIR)
    print("NOTE: latent = REAL Mojo Flux-VAE encode of Alina images; t5_embed/clip_pool")
    print("      = REAL text embeddings sourced from the Rust flux cache (T5 Unigram")
    print("      tokenizer port pending for a fully self-contained Mojo text encode).")
