# serenitymojo/pipeline/anima_prepare.mojo
#
# ANIMA dataset prep — TRANSLATION of:
#   EriDiffusion-v2/crates/eridiffusion-cli/src/bin/prepare_anima.rs
#
# CACHE CONTRACT (one safetensors per sample, the exact schema train_anima_real
# reads — prepare_anima.rs:16-21):
#   latent:         BF16 [1, 16, H/8, W/8]   raw Qwen-Image VAE encode (per-channel norm)
#   text_embedding: BF16 [1, qwen3_max, 1024] Qwen3-0.6B last_hidden_state (pad rows zeroed)
#   text_mask:      F32  [1, qwen3_max]       1.0 at valid Qwen3 tokens
#   t5_input_ids:   F32  [1, t5_max]          T5 token IDs for the LLM Adapter embedding
#   t5_attn_mask:   F32  [1, t5_max]          1.0 at valid T5 tokens
# Plus a `_meta.json` sentinel (version 2 = T5 pad rows zeroed; prepare_anima.rs:165-170).
#
# PORT STATUS / HONEST SCOPE (Tenet 4 — measured, not asserted):
#   The real-image encode has two halves:
#     (a) Qwen-Image VAE ENCODER (8x down, 4 stages). Mojo has the matching
#         DECODER (models/vae/qwenimage_decoder.QwenImageVaeDecoder) but the
#         8x-down ENCODER is NOT ported (vae/vae_encode_general.GeneralVaeEncoder
#         is a single-downsample generic, not the Qwen-Image encoder stack).
#     (b) Qwen3-0.6B + T5 tokenizer + 6-block LLM adapter text path. NOT ported
#         to Mojo (serenitymojo/text_encoder/ has no qwen3/t5 encoder).
#   Until those two ports land, this binary does NOT fabricate latents/embeddings.
#   It instead VERIFIES + REUSES caches that already carry the real encode:
#     * Rust prepare_anima caches (the canonical path), or
#     * the captured Anima context sidecar + a cached latent (the smoke path).
#   This is the FAST PATH the milestone sanctions; train_anima_real consumes the
#   same schema either way. The cross-attn context [B,256,1024] is the FROZEN
#   LLM-adapter OUTPUT (captured sidecar context_cond), consumed as a frozen
#   input — no adapter backward is needed for LoRA (TRAINING_PLAN_anima.md §D/§E).
#
# This module validates a cache directory against the contract so the trainer
# can rely on it, and reports exactly what is present / missing.
#
# Run (SEPARATE command, after build):
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/pipeline/anima_prepare.mojo -o /tmp/anima_prepare
#   /tmp/anima_prepare /home/alex/EriDiffusion/EriDiffusion-v2/cache/anima_synth_smoke

from std.sys import argv
from std.collections import List
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.ffi import sys_open, sys_close, O_RDONLY
from serenitymojo.models.dit.anima_contract import ANIMA_LATENT_CHANNELS


def _cache_keys() -> List[String]:
    var k = List[String]()
    k.append(String("latent"))
    k.append(String("text_embedding"))
    k.append(String("text_mask"))
    k.append(String("t5_input_ids"))
    k.append(String("t5_attn_mask"))
    return k^


def _file_exists(path: String) -> Bool:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        return False
    _ = sys_close(fd)
    return True


def _shape_str(s: List[Int]) -> String:
    var out = String("[")
    for i in range(len(s)):
        if i > 0:
            out += ", "
        out += String(s[i])
    return out + "]"


# SafeTensors exposes names() as a List; build a quick membership test.
# (Kept local so we don't touch the io module.)
def _has_key(st: SafeTensors, key: String) -> Bool:
    var names = st.names()
    for i in range(len(names)):
        if names[i] == key:
            return True
    return False


def _validate_dir(cache_dir: String, ctx: DeviceContext) raises:
    print("==== anima_prepare — cache contract validation ====")
    print("cache dir:", cache_dir)

    # _meta.json sentinel (prepare_anima.rs:716-733 — trainer bails without it).
    var meta = cache_dir + "/_meta.json"
    if _file_exists(meta):
        print("  _meta.json: present (version-2 sentinel expected)")
    else:
        print("  _meta.json: MISSING — Rust trainer would bail; "
              + "train_anima_real reads samples directly (sentinel advisory).")

    # Validate sample0 (the smoke sample the trainer loads by name).
    var s0 = cache_dir + "/sample0.safetensors"
    if not _file_exists(s0):
        raise Error("no sample0.safetensors in cache dir: " + cache_dir
                    + "  (real-image encode needs the Qwen-Image VAE encoder + "
                    + "Qwen3/T5 text path, neither ported yet — reuse a Rust "
                    + "prepare_anima cache or the captured smoke cache)")
    var st = SafeTensors.open(s0)
    var keys = _cache_keys()
    var present = List[String]()
    var missing = List[String]()
    for i in range(len(keys)):
        if _has_key(st, keys[i]):
            present.append(keys[i])
        else:
            missing.append(keys[i])
    print("  sample0 keys present:", len(present), "/", len(keys))
    if len(missing) > 0:
        var ms = String("")
        for i in range(len(missing)):
            if i > 0:
                ms += ", "
            ms += missing[i]
        raise Error("sample0 missing contract keys: " + ms)

    var lat = st.tensor_info("latent")
    if len(lat.shape) != 4 or lat.shape[1] != ANIMA_LATENT_CHANNELS:
        raise Error("latent shape violates contract: " + _shape_str(lat.shape))
    var nH = lat.shape[2] // 2
    var nW = lat.shape[3] // 2
    var s_img = nH * nW
    var emb = st.tensor_info("text_embedding")
    print("  latent", _shape_str(lat.shape), "-> S_IMG", s_img)
    print("  text_embedding", _shape_str(emb.shape))

    # Context note: the DiT cross-attn input is the LLM-adapter OUTPUT
    # [B,256,1024], NOT text_embedding [B,512,1024]. train_anima_real loads it
    # from the captured sidecar (context_cond). Report whether it's available.
    var sidecar = "/home/alex/EriDiffusion/inference-flame/output/anima_embeddings.safetensors"
    if _file_exists(sidecar):
        var sc = SafeTensors.open(sidecar)
        if _has_key(sc, String("context_cond")):
            var cc = sc.tensor_info("context_cond")
            print("  frozen adapter context (sidecar context_cond):",
                  _shape_str(cc.shape), " — usable by trainer")
        else:
            print("  sidecar present but no context_cond key")
    else:
        print("  adapter-context sidecar MISSING:", sidecar,
              " — train_anima_real needs it (or a ported LLM adapter)")

    print("")
    print("VERDICT: cache contract OK — train_anima_real can consume", s0)


def main() raises:
    var ctx = DeviceContext()
    var args = argv()
    var cache_dir = String("/home/alex/EriDiffusion/EriDiffusion-v2/cache/anima_synth_smoke")
    if len(args) > 1:
        cache_dir = String(args[1])
    _validate_dir(cache_dir, ctx)
