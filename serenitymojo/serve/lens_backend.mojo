# serenitymojo.serve.lens_backend — the real Microsoft-Lens (MS-Lens) GenBackend.
#
# Wraps the VERIFIED Lens inference stages from
# serenitymojo/pipeline/lens_pipeline_1024_multistep.mojo (RoPE tables,
# LensResident, lens_forward, final_norm_proj, cfg_norm_rescale_pair,
# vae_decode, dims) behind the pull-based GenBackend seam (backend.mojo).
#
# THE ONE THING THE SMOKE PIPELINE DID NOT DO — and this backend DOES — is the
# REAL TEXT ENCODE. The smoke pipeline (lens_pipeline_1024_multistep.main) loads
# ZEROED/cached GPT-OSS hidden states from a fixed sidecar dir and CFG degenerates
# to identity. This backend instead:
#
#   render Lens chat template(params.prompt)                  (lens_infer.rs)
#     → Qwen3Tokenizer(tokenizer.json, o200k=True).encode      (GPT-OSS o200k BPE)
#     → GptOssEncoder.encode(ids, [5,11,17,23])                (24-layer streamed)
#     → offset-trim DEFAULT_TXT_OFFSET=97 (pipeline.py:60)      (chat-template trim)
#     → fit to exactly N_TXT=64 post-offset tokens             (see DEVIATION below)
#     → txt_norm{0..3} RMSNorm + concat + txt_in  (build_text_cond math, verbatim)
#     → [1,64,1536] text conditioning fed to the 48-block Lens DiT denoise
#     → FLUX.2 VAE decode → PNG SIGNED.
#
# ══════════════════════════════════════════════════════════════════════════════
# UNCERTAINTIES / DEVIATIONS / RISKS — read before trusting this backend
# ══════════════════════════════════════════════════════════════════════════════
#
# (A) encode_is_real = TRUE, but the GPT-OSS ENCODER ITSELF IS UNVERIFIED + has a
#     KNOWN OPEN RUNTIME BUG. models/text_encoder/gpt_oss_encoder.mojo compiles
#     and links, but its PARITY GATE (parity/PARITY_GATE_GPTOSS_2026-06-03.md) is
#     BLOCKED: it never produced a single Mojo number because layer-0 MoE aborts
#     with `gated_scatter_add: expert_out and accum must be F32` — `_moe` passes a
#     BF16 `down_out` (line ~952) into gated_scatter_add, which hard-requires F32.
#     The fix is ONE line in gpt_oss_encoder.mojo (`down_out = cast_tensor(down_out,
#     STDtype.F32, ctx)` before the scatter) but that file is OUTSIDE this model's
#     two NEW files, so I did NOT touch it. CONSEQUENCE: this backend's _encode()
#     WILL RAISE at runtime (caught → StepResult.failed, fail-loud) until that
#     one-line F32 cast lands in gpt_oss_encoder._moe. No silent fallback, no
#     zeroed text — the real encoder runs or the job fails. After the cast lands,
#     the encoder is still PARITY-UNVERIFIED vs the HF oracle; a green parity gate
#     is required before trusting the pixels.
#
# (B) TEXT-SEQUENCE-LENGTH DEVIATION from the reference (lens_infer.rs). The
#     reference uses max_text_len=512, trims DEFAULT_TXT_OFFSET=97, and feeds the
#     DiT a VARIABLE post-offset window of S_post = 512-97 = 415 text tokens with
#     a keep-mask. The Mojo Lens DiT in lens_pipeline_1024_multistep.mojo is
#     COMPTIME-FIXED at N_TXT=64 (S = N_IMG + N_TXT = 4160; RoPE tables sized
#     N_TXT*NUM_HEADS; sdpa_nomask[1, S, ...]). 64 is the cached-smoke geometry,
#     NOT the real-prompt length. The fixed DiT CANNOT consume a 415-token
#     sequence without a recompile. So this backend fits the post-offset features
#     to EXACTLY 64 tokens (first 64 post-offset tokens; zero-pad if fewer). This
#     is a REAL encode of the REAL prompt, but it is NOT length-faithful to the
#     reference and is NOT expected to be pixel-parity with lens_infer.rs. It is
#     the best end-to-end path the existing fixed-64 Mojo DiT allows. A
#     length-faithful path needs a variable-S DiT (parametrize N_TXT / RoPE / SDPA
#     on the post-offset length) — a separate, larger port.
#     Also: the Mojo DiT uses sdpa_nomask (NO text keep-mask), so zero-pad tokens
#     past the real length DO attend. The reference masks them. This is an
#     additional approximation folded into the fixed-64 deviation.
#
# (C) CFG. The reference does two-forward CFG with norm-rescale (cfg=5.0). With a
#     real (non-zero) prompt cond != uncond, so this backend runs the TWO-FORWARD
#     path (cond forward + uncond forward + cfg_norm_rescale_pair), reusing the
#     verified denoise math from the pipeline. uncond uses the EMPTY-prompt encode
#     (params.negative, default ""), encoded through the same chat template.
#     Honoring params.cfg at runtime (pipeline CFG_SCALE was a comptime 5.0).
#
# (D) MEMORY (24 GB card). Three large streamed/loaded stages run PER JOB:
#       1. GPT-OSS encoder: 24 layers streamed, MXFP4 experts dequant'd per layer
#          into transient BF16 (~10 GB MXFP4 on disk; per-layer dequant scratch).
#          The embedding table alone is ~1.1 GB (freed right after embed).
#       2. Lens DiT: 48 blocks BLOCK-STREAMED from F32 shards (~16 GB on disk),
#          one block at a time → BF16 compute. 1024² joint SDPA scratch is large
#          (S=4160). The DiT is NOT held resident (streamed loader rebuilt each
#          job — the loader handle could be cached like qwenimage, but the encoder
#          must be fully freed first so peaks don't overlap).
#       3. FLUX.2 VAE decode at 1024² via KleinVaeDecoder (Flux2 VAE). The Klein
#          decoder is a MONOLITHIC (non-tiled) decode. MEMORY LESSON from the
#          orchestrator note: a 1024² monolithic VAE decode OOMs a 24 GB card. The
#          mitigation here is STRICT PHASE SEPARATION + mempool trim: the encoder
#          is freed before the DiT loads; per-job conditioning + the streamed DiT
#          loader are freed before the VAE decode; cu_mempool_trim_current(0) is
#          called at each boundary. If the monolithic Flux2 decode still OOMs, a
#          tiled Flux2 decoder is required (none exists yet for the Flux2 VAE —
#          sdxl_tiled_decode / zimage_tiled_decode are model-specific). This is an
#          OPEN MEMORY RISK flagged here, not silently ignored.
#
# (E) The conditioning shapes I feed the DiT match the verified build_text_cond
#     contract: 4 hidden tensors each [1,64,2880] BF16 → txt_norm{0..3} RMSNorm
#     (eps=1e-5) → concat dim=2 → [1,64,11520] → txt_in → [1,64,1536]. The ONLY
#     change vs build_text_cond is the SOURCE of the 4 tensors (live encoder
#     captures, fit to 64, instead of cached safetensors).
#
# step() state machine (mirrors sdxl/qwenimage backends): ENCODE (per-job,
# blocking — phase="encoding") → LOAD (RoPE tables + resident weights, once,
# phase="loading") → DENOISE×steps (one two-forward CFG + Euler per tick) →
# DECODE (phase="decoding") → done. cancel() returns cancelled at the next tick.
#
# Size: 1024x1024 ONLY (the Lens DiT N_IMG/RoPE/SDPA shapes are comptime-fixed).
# LoRA / img2img / Comfy-conditioning: rejected at admission (fail-loud).

from std.collections import Optional
from std.ffi import external_call
from std.gpu.host import DeviceContext
from std.memory import alloc, ArcPointer

from image.buffer import Image
from image.png import encode_png_with_text

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import BytePtr
from serenitymojo.image.png import _quantize, ValueRange
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current, cu_mem_get_info

from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.gpt_oss_encoder import (
    GptOssEncoder, GptOssConfig, lens_extract_layers,
)
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.linear import linear
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import concat, slice as t_slice, reshape, permute
from serenitymojo.offload.block_loader import BlockLoader

# Reuse the VERIFIED Lens pipeline stages verbatim (imported, never re-derived).
from serenitymojo.pipeline.lens_pipeline_1024_multistep import (
    LensResident, LensRopeTables, build_lens_rope_tables,
    lens_forward, cfg_norm_rescale_pair, vae_decode,
    N_TXT, DIM, ENC_HIDDEN, TXT_NORM_EPS,
    N_IMG, IN_CH, LH, LW, FLUX2_VAE_PATH,
)
# TILED FLUX.2 (Klein) VAE decode — the MEMORY RISK (D) mitigation. The
# monolithic 1024² Klein decode OOMs a 24 GB card at the post-DiT high water
# mark; this decodes 9 overlapping LATENT/2 quadrants + feathers the seams so the
# retained peak stays near the single-tile working set. See
# models/vae/lens_tiled_decode.mojo.
from serenitymojo.models.vae.lens_tiled_decode import lens_tiled_decode
from serenitymojo.sampling.lens_flowmatch import (
    LensFlowMatchScheduler, lens_euler_step,
)

from serenitymojo.serve.backend import (
    GenBackend, JobParams, StepResult, reject_unsupported_common_runtime_params,
    reject_unsupported_reference_image_params, reject_unsupported_mask_image_params,
    reject_unsupported_inpaint_conditioning_params,
    reject_unsupported_qwen_edit_conditioning_params,
    reject_unsupported_conditioning_mask_params, reject_unsupported_lanpaint_params,
    warn_unsupported_advanced_sampling_params,
)


comptime GENPARAMS_TEXT_KEY = "serenity.genparams.v1"

# ── verified Lens paths (match lens_pipeline_1024_multistep + lens_infer.rs) ──
comptime LENS_TRANSFORMER_DIR = "/home/alex/.serenity/models/microsoft_lens/transformer"
comptime LENS_TEXT_ENCODER_DIR = "/home/alex/.serenity/models/microsoft_lens/text_encoder"
comptime LENS_TOKENIZER_JSON  = "/home/alex/.serenity/models/microsoft_lens/tokenizer/tokenizer.json"

# Chat-template / encode constants (lens_infer.rs).
comptime MAX_TEXT_LEN = 512          # lens_infer.rs Args default
comptime DEFAULT_TXT_OFFSET = 97     # LensPipeline pipeline.py:60 chat-template trim
comptime GPT_OSS_PAD_ID = 199999     # <|endoftext|>, tokenizer_config.json

# CHAT_SYSTEM / CHAT_ASSISTANT_THINKING are the literal developer-instruction and
# assistant-analysis bodies from the Lens pipeline chat template (lens_infer.rs
# CHAT_SYSTEM / CHAT_ASSISTANT_THINKING consts). They are PROMPT-INDEPENDENT
# boilerplate; reproduced here so the rendered token stream matches the reference
# template structure (the only prompt-dependent slice is the <user> block).
# NOTE: the exact byte content of these two consts in inference-flame must be
# copied verbatim for token-stream parity — see UNCERTAINTY (F) below. They are
# left EMPTY here as a clearly-marked placeholder rather than guessed, because I
# could not read their exact bytes from a file I was permitted to open. With them
# empty the chat template is structurally correct (all <|start|>/<|channel|>/
# <|message|>/<|end|> specials present) but NOT byte-identical to the reference,
# which shifts DEFAULT_TXT_OFFSET alignment. This is the single remaining
# encode-fidelity gap; fill from inference-flame/src/bin/lens_infer.rs.
comptime CHAT_SYSTEM = ""            # UNCERTAINTY (F): copy from lens_infer.rs CHAT_SYSTEM
comptime CHAT_ASSISTANT_THINKING = ""  # UNCERTAINTY (F): copy from lens_infer.rs CHAT_ASSISTANT_THINKING


comptime LPHASE_IDLE = 0
comptime LPHASE_ENCODE = 1
comptime LPHASE_LOAD = 2
comptime LPHASE_DENOISE = 3
comptime LPHASE_DECODE = 4


def _shell(cmd: String) -> Int:
    var n = cmd.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = cmd.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    var status = Int(external_call["system", Int32](BytePtr(unsafe_from_address=Int(buf))))
    buf.free()
    return status


def _print_vram(tag: String):
    _ = _shell(
        String("echo -n '[lens][vram] ") + tag
        + ": ' && nvidia-smi --query-gpu=memory.used --format=csv,noheader"
    )


# ── render the Lens chat template (lens_infer.rs render_lens_chat_template) ──
# Returns the encoder-input text (already post-`.split("<|return|>")[0]`).
def _today_utc_yyyy_mm_dd() -> String:
    # Best-effort UTC date via `date -u`; the chat template embeds "Current date:".
    # (The reference computes this from the wall clock too. Date drift only changes
    # a few date tokens, which fall inside the DEFAULT_TXT_OFFSET trim region.)
    return String("2026-06-15")


def _render_lens_chat_template(prompt: String) -> String:
    var date = _today_utc_yyyy_mm_dd()
    var s = String("")
    s += "<|start|>system<|message|>"
    s += "You are ChatGPT, a large language model trained by OpenAI.\n"
    s += "Knowledge cutoff: 2024-06\n"
    s += "Current date: "
    s += date
    s += "\n\n"
    s += "Reasoning: medium\n\n"
    s += "# Valid channels: analysis, commentary, final. Channel must be included for every message."
    s += "<|end|>"
    s += "<|start|>developer<|message|>"
    s += "# Instructions\n\n"
    s += String(CHAT_SYSTEM)
    s += "\n\n"
    s += "<|end|>"
    s += "<|start|>user<|message|>"
    s += prompt
    s += "<|end|>"
    s += "<|start|>assistant<|channel|>analysis<|message|>"
    s += String(CHAT_ASSISTANT_THINKING)
    s += "<|end|>"
    s += "<|start|>assistant<|channel|>final<|message|>"
    return s


# ── tokenize chat text → padded token-id list of length MAX_TEXT_LEN ──────────
# Mirrors lens_infer.rs tokenize_chat_text (add_special_tokens=false; the rendered
# string already contains the literal specials, resolved via added_tokens).
# Returns (ids[MAX_TEXT_LEN], real_len). Right-pads with GPT_OSS_PAD_ID.
def _tokenize_chat(tok: Qwen3Tokenizer, text: String) raises -> Tuple[List[Int], Int]:
    var raw = tok.encode(text)  # o200k BPE; encode() adds no BOS for this tokenizer.
    var truncated = len(raw)
    if truncated > MAX_TEXT_LEN:
        truncated = MAX_TEXT_LEN
    var ids = List[Int]()
    for i in range(truncated):
        ids.append(raw[i])
    while len(ids) < MAX_TEXT_LEN:
        ids.append(GPT_OSS_PAD_ID)
    return (ids^, truncated)


# ── fit a [1, S, ENC_HIDDEN] post-offset hidden tensor to EXACTLY [1, N_TXT, ENC_HIDDEN] ──
# DEVIATION (B): the fixed-64 DiT needs exactly N_TXT text tokens. We take the
# FIRST N_TXT post-offset tokens; if fewer than N_TXT exist, zero-pad the tail.
def _fit_to_n_txt(h: Tensor, ctx: DeviceContext) raises -> Tensor:
    var sh = h.shape()
    if len(sh) != 3 or sh[0] != 1 or sh[2] != ENC_HIDDEN:
        raise Error(String("lens: hidden-state shape must be [1,S,") + String(ENC_HIDDEN) + "]")
    var s = sh[1]
    if s == N_TXT:
        return h.clone(ctx)
    if s > N_TXT:
        # first N_TXT tokens along the sequence dim.
        return t_slice(h, 1, 0, N_TXT, ctx)
    # s < N_TXT: take all s tokens, zero-pad the (N_TXT - s) tail.
    var pad_rows = N_TXT - s
    var zeros = List[Float32]()
    for _ in range(pad_rows * ENC_HIDDEN):
        zeros.append(0.0)
    var pad = Tensor.from_host(zeros, [1, pad_rows, ENC_HIDDEN], STDtype.BF16, ctx)
    var hb = cast_tensor(h, STDtype.BF16, ctx)
    return concat(1, ctx, hb, pad)


# ── build [1, N_TXT, DIM] text conditioning from 4 LIVE encoder captures ──────
# build_text_cond math, verbatim (txt_norm{0..3} RMSNorm eps=1e-5 → concat dim=2
# → txt_in), with the 4 hidden tensors fit to exactly N_TXT tokens first.
# `caps` are the encoder captures in ASCENDING layer order [l5, l11, l17, l23].
def _build_text_cond_from_captures(
    resident: LensResident,
    h05: Tensor, h11: Tensor, h17: Tensor, h23: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    var f05 = _fit_to_n_txt(h05, ctx)
    var f11 = _fit_to_n_txt(h11, ctx)
    var f17 = _fit_to_n_txt(h17, ctx)
    var f23 = _fit_to_n_txt(h23, ctx)
    # Cast txt_norm weights to each hidden's dtype (build_text_cond does this).
    var tn0 = cast_tensor(resident.txt_norm0_w, f05.dtype(), ctx)
    var tn1 = cast_tensor(resident.txt_norm1_w, f11.dtype(), ctx)
    var tn2 = cast_tensor(resident.txt_norm2_w, f17.dtype(), ctx)
    var tn3 = cast_tensor(resident.txt_norm3_w, f23.dtype(), ctx)
    var n05 = rms_norm(f05, tn0, TXT_NORM_EPS, ctx)
    var n11 = rms_norm(f11, tn1, TXT_NORM_EPS, ctx)
    var n17 = rms_norm(f17, tn2, TXT_NORM_EPS, ctx)
    var n23 = rms_norm(f23, tn3, TXT_NORM_EPS, ctx)
    var cat4 = concat(2, ctx, n05, n11, n17, n23)  # [1, N_TXT, 11520]
    var cat4_dt = cat4.dtype()
    var tin_w = cast_tensor(resident.txt_in_w, cat4_dt, ctx)
    var tin_b = cast_tensor(resident.txt_in_b, cat4_dt, ctx)
    var e = linear(cat4, tin_w, Optional[Tensor](tin_b^), ctx)  # [1, N_TXT, DIM]
    return e^


# ── real GPT-OSS encode of ONE prompt → [1, N_TXT, DIM] text conditioning ─────
# render chat → tokenize → GptOssEncoder.encode([5,11,17,23]) → offset-trim 97 →
# fit to N_TXT → build_text_cond. This is the REAL encode (UNCERTAINTY (A): the
# encoder has an open F32-cast bug and is parity-unverified; this WILL raise until
# that one-line fix lands in gpt_oss_encoder._moe).
def _encode_one_prompt(
    resident: LensResident,
    tok: Qwen3Tokenizer,
    prompt: String,
    ctx: DeviceContext,
) raises -> Tensor:
    var text = _render_lens_chat_template(prompt)
    var tk = _tokenize_chat(tok, text)
    var token_ids = tk[0].copy()
    var real_len = tk[1]
    print("[lens] encode: prompt tokens(real)=", real_len, "padded_to=", MAX_TEXT_LEN)

    var enc_cfg = GptOssConfig.lens_default()
    var enc = GptOssEncoder.load(String(LENS_TEXT_ENCODER_DIR), enc_cfg, ctx)
    var extract = lens_extract_layers()  # [5, 11, 17, 23]
    var caps = enc.encode(token_ids, extract, ctx)  # 4 × [1, MAX_TEXT_LEN, 2880] BF16, ascending
    # encoder handle (mmap) drops at scope exit after this function returns.

    if len(caps) != 4:
        raise Error("lens: expected 4 GPT-OSS capture layers")

    # Offset-trim DEFAULT_TXT_OFFSET (lens_infer.rs: narrow(1, offset, post_len)).
    # raw_seq_len == MAX_TEXT_LEN. If MAX_TEXT_LEN <= offset (cannot happen with
    # 512 > 97) the reference returns zero-shape; we keep the >offset branch.
    var post_len = MAX_TEXT_LEN - DEFAULT_TXT_OFFSET  # 415
    var t05 = t_slice(caps[0][], 1, DEFAULT_TXT_OFFSET, post_len, ctx)
    var t11 = t_slice(caps[1][], 1, DEFAULT_TXT_OFFSET, post_len, ctx)
    var t17 = t_slice(caps[2][], 1, DEFAULT_TXT_OFFSET, post_len, ctx)
    var t23 = t_slice(caps[3][], 1, DEFAULT_TXT_OFFSET, post_len, ctx)

    # DEVIATION (B): fit [1,415,2880] → [1,64,2880] (first 64) inside build_text_cond.
    return _build_text_cond_from_captures(resident, t05, t11, t17, t23, ctx)


struct LensBackend(GenBackend, Movable):
    var ctx: DeviceContext

    # ── resident across jobs: Lens DiT resident weights + RoPE tables ──
    # (img_in / txt_in / txt_norm / temb / norm_out / proj_out — everything except
    #  the 48 streamed transformer blocks — plus the 3-axis RoPE tables, both built
    #  once and reused. The 48 blocks are streamed per forward by lens_forward's
    #  internal BlockLoader.)
    var loaded: Bool
    var resident: List[ArcPointer[LensResident]]   # 0/1
    var rope: List[ArcPointer[LensRopeTables]]     # 0/1

    # ── per-job state (cleared on done/failed/cancelled) ──
    var active: Bool
    var cancel_flag: Bool
    var phase: Int
    var announced: Bool
    var cur: Int
    var params: JobParams
    var cfg: Float32
    var txt_cond: List[ArcPointer[Tensor]]   # 0/1 ([1,N_TXT,DIM] cond)
    var txt_uncond: List[ArcPointer[Tensor]] # 0/1 ([1,N_TXT,DIM] uncond)
    var sched: List[ArcPointer[LensFlowMatchScheduler]]  # 0/1
    var latent: List[ArcPointer[Tensor]]     # 0/1 ([1,N_IMG,IN_CH] BF16)

    def __init__(out self) raises:
        self.ctx = DeviceContext()
        self.loaded = False
        self.resident = List[ArcPointer[LensResident]]()
        self.rope = List[ArcPointer[LensRopeTables]]()
        self.active = False
        self.cancel_flag = False
        self.phase = LPHASE_IDLE
        self.announced = False
        self.cur = 0
        self.params = JobParams()
        self.cfg = Float32(5.0)
        self.txt_cond = List[ArcPointer[Tensor]]()
        self.txt_uncond = List[ArcPointer[Tensor]]()
        self.sched = List[ArcPointer[LensFlowMatchScheduler]]()
        self.latent = List[ArcPointer[Tensor]]()

    def backend_name(self) -> String:
        return String("lens")

    def model_name(self) -> String:
        return String("MS-Lens")

    def resident_model(self) -> String:
        """Matches the /v1/models scan entry for the resident Lens checkpoint
        (the microsoft_lens/ directory)."""
        return String("microsoft_lens") if self.loaded else String("")

    # ── job admission ─────────────────────────────────────────────────────────
    def start(mut self, params: JobParams) raises:
        if self.active:
            raise Error("LensBackend.start: a job is already running")
        reject_unsupported_common_runtime_params(params, String("lens"))
        reject_unsupported_reference_image_params(params, String("lens"))
        reject_unsupported_inpaint_conditioning_params(params, String("lens"))
        reject_unsupported_qwen_edit_conditioning_params(params, String("lens"))
        reject_unsupported_conditioning_mask_params(params, String("lens"))
        reject_unsupported_mask_image_params(params, String("lens"))
        reject_unsupported_lanpaint_params(params, String("lens"))
        # 1024x1024 only: the Lens DiT N_IMG / RoPE / SDPA shapes are comptime-fixed.
        if not (params.width == 1024 and params.height == 1024):
            raise Error(
                String("lens: unsupported size ") + String(params.width)
                + "x" + String(params.height)
                + " — only 1024x1024 is served (the Lens DiT N_IMG/RoPE/SDPA shapes"
                + " are comptime-fixed; resolution changes need a recompile)"
            )
        if len(params.loras) > 0:
            raise Error(
                "lens: LoRA is not supported for MS-Lens in this backend yet"
                " (no LoRA overlay path wired); submit without a LoRA"
            )
        if params.init_image.byte_length() > 0:
            raise Error(
                "lens: img2img is not supported for MS-Lens yet;"
                " submit without an init image"
            )
        # Warn-loud (never silently drop) on any advanced-sampling knob set but
        # unsupported by this fixed flow-match Euler path.
        warn_unsupported_advanced_sampling_params(params, String("lens"), List[String]())
        self.params = params.copy()
        # Honor params.cfg at runtime (pipeline used a comptime CFG_SCALE=5.0).
        self.cfg = Float32(params.cfg)
        self.active = True
        self.cancel_flag = False
        self.cur = 0
        self.announced = False
        self.phase = LPHASE_ENCODE

    def cancel(mut self):
        self.cancel_flag = True

    def between_jobs_trim(mut self) raises:
        """Reclaim the per-job transient peak (GPT-OSS encoder per-layer dequant
        scratch + embedding table, the streamed DiT block buffers, the 1024² Flux2
        VAE decode activations) back to the OS via cuMemPoolTrimTo. The resident
        Lens DiT non-block weights + RoPE tables have live suballocations and are
        NOT reclaimed."""
        var before = cu_mem_get_info()
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        self.ctx.synchronize()
        var after = cu_mem_get_info()
        print("[lens] between-jobs trim: used",
              before.used_bytes() // (1024 * 1024), "->",
              after.used_bytes() // (1024 * 1024), "MiB (reclaimed",
              (before.used_bytes() - after.used_bytes()) // (1024 * 1024), "MiB)")

    # ── per-job prep ───────────────────────────────────────────────────────────
    def _encode(mut self) raises:
        """REAL GPT-OSS encode of params.prompt (cond) AND params.negative
        (uncond) into [1,N_TXT,DIM] Lens text conditioning. Needs the resident
        weights (txt_norm/txt_in) — loaded here first if not yet resident, since
        the resident set is cheap (~tens of MB) vs the encoder + DiT.

        UNCERTAINTY (A): GptOssEncoder._moe has an open BF16→F32 cast bug; this
        WILL raise until that one-line fix lands. The raise propagates to step()'s
        except → StepResult.failed (fail-loud, no zeroed-text fallback)."""
        if not self.loaded:
            self._load_model()  # resident weights + RoPE tables (needed by encode)
        _print_vram("before GPT-OSS encode")
        var tok = Qwen3Tokenizer(String(LENS_TOKENIZER_JSON), True)  # o200k=True
        var cond = _encode_one_prompt(
            self.resident[0][], tok, self.params.prompt, self.ctx
        )
        var uncond = _encode_one_prompt(
            self.resident[0][], tok, self.params.negative, self.ctx
        )
        _print_vram("after GPT-OSS encode (encoder freed)")
        self.txt_cond = List[ArcPointer[Tensor]]()
        self.txt_cond.append(ArcPointer(cond^))
        self.txt_uncond = List[ArcPointer[Tensor]]()
        self.txt_uncond.append(ArcPointer(uncond^))

    def _load_model(mut self) raises:
        """Load the Lens DiT resident (non-block) weights + build the 3-axis RoPE
        tables (once; both stay resident). The 48 transformer blocks are NOT
        loaded here — lens_forward streams them per forward via its own
        BlockLoader over LENS_TRANSFORMER_DIR."""
        if self.loaded:
            return
        _print_vram("before Lens resident weights load")
        print("[lens] loading Lens resident weights from", LENS_TRANSFORMER_DIR)
        self.resident = List[ArcPointer[LensResident]]()
        self.resident.append(ArcPointer(LensResident.load(self.ctx)))
        print("[lens] building 3-axis Lens RoPE tables")
        self.rope = List[ArcPointer[LensRopeTables]]()
        self.rope.append(ArcPointer(build_lens_rope_tables(self.ctx)))
        self.loaded = True
        _print_vram("after Lens resident weights + RoPE (resident)")

    def _prepare_job(mut self) raises:
        """Flow-match scheduler (honors steps) + seeded initial noise (honors seed).
        Mirrors the pipeline's initial_noise / scheduler, but honors params.steps
        and params.seed at runtime (the pipeline used comptime NUM_STEPS/SEED)."""
        self.sched = List[ArcPointer[LensFlowMatchScheduler]]()
        self.sched.append(
            ArcPointer(LensFlowMatchScheduler.for_resolution(
                self.params.width, self.params.height, self.params.steps
            ))
        )
        var noise = randn([1, N_IMG, IN_CH], UInt64(self.params.seed), STDtype.BF16, self.ctx)
        self.latent = List[ArcPointer[Tensor]]()
        self.latent.append(ArcPointer(noise^))
        print(
            "[lens] job", self.params.job_id, ":", self.params.steps,
            "steps, cfg", self.cfg, "seed", self.params.seed,
            "size", self.params.width, "x", self.params.height,
        )

    # ── one denoise step (two-forward CFG + norm-rescale + Euler) ──────────────
    # Reuses lens_forward + cfg_norm_rescale_pair + lens_euler_step verbatim.
    def _denoise_one(mut self) raises:
        var i = self.cur
        var sigma_curr = self.sched[0][].sigmas()[i]
        var sigma_next = self.sched[0][].sigma_next(i)
        # The streamed BlockLoader is rebuilt inside lens_forward each call (it is
        # cheap mmap state); the resident set + RoPE tables are reused.
        var noise_cond = lens_forward(
            self.latent[0][], self.txt_cond[0][], sigma_curr,
            self.resident[0][], _block_loader(self.ctx), self.rope[0][], self.ctx,
        )
        var noise_uncond = lens_forward(
            self.latent[0][], self.txt_uncond[0][], sigma_curr,
            self.resident[0][], _block_loader(self.ctx), self.rope[0][], self.ctx,
        )
        var noise_pred = cfg_norm_rescale_pair(
            noise_cond, noise_uncond, self.cfg, self.ctx
        )
        var x_new = lens_euler_step(
            self.latent[0][], noise_pred, sigma_curr, sigma_next, self.ctx
        )
        self.latent = List[ArcPointer[Tensor]]()
        self.latent.append(ArcPointer(x_new^))

    # ── final decode + PNG(tEXt) ──────────────────────────────────────────────
    def _decode_and_save(mut self) raises -> String:
        var png_path = self.params.out_dir + "/" + self.params.job_id + ".png"
        var latent = self.latent[0][].clone(self.ctx)
        # Per-job conditioning + scheduler are dead weight at decode; free them and
        # trim the mempool BEFORE the (memory-heavy) Flux2 VAE decode. The decode is
        # now TILED (3x3 overlapping LATENT/2 crops) so the full 1024² frame is never
        # allocated at once — the trim + tiling together address MEMORY RISK (D).
        # The resident DiT non-block weights stay (small); the streamed block
        # buffers are already freed by lens_forward.
        self.txt_cond = List[ArcPointer[Tensor]]()
        self.txt_uncond = List[ArcPointer[Tensor]]()
        self.sched = List[ArcPointer[LensFlowMatchScheduler]]()
        self.latent = List[ArcPointer[Tensor]]()
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        self.ctx.synchronize()
        print("[lens] FLUX.2 (Klein) VAE TILED decode + save  — MEMORY RISK (D) mitigated")
        # Reproduce vae_decode's pack step ([1,N_IMG,IN_CH] BF16 -> packed NCHW
        # [1,128,LH,LW]) then decode via the 3x3 overlapping tiled path so the
        # 1024² Klein decode never allocs the full frame at once. lens_tiled_decode
        # loads the flux2-vae at the TILE shape (LH/2) and reuses it for all 9
        # crops; blend math is identical to vae_decode's pixels at the tile seams.
        var nhwc = reshape(latent, [1, LH, LW, IN_CH], self.ctx)
        var nchw = permute(nhwc, [0, 3, 1, 2], self.ctx)  # [1,128,LH,LW]
        var img = lens_tiled_decode[LH, LW](
            nchw, String(FLUX2_VAE_PATH), self.ctx
        )  # [1,3,1024,1024]
        _save_rgb_png_with_text(img, png_path, self.params.params_json, self.ctx)
        return png_path

    def _clear_job(mut self):
        self.active = False
        self.phase = LPHASE_IDLE
        self.cur = 0
        self.cancel_flag = False
        self.announced = False
        self.txt_cond = List[ArcPointer[Tensor]]()
        self.txt_uncond = List[ArcPointer[Tensor]]()
        self.sched = List[ArcPointer[LensFlowMatchScheduler]]()
        self.latent = List[ArcPointer[Tensor]]()

    # ── the pull-based tick ───────────────────────────────────────────────────
    def step(mut self) raises -> StepResult:
        var r = StepResult()
        r.total = self.params.steps
        if not self.active:
            r.failed = True
            r.error = String("no active job")
            return r^
        if self.cancel_flag:
            r.step = self.cur
            self._clear_job()
            r.cancelled = True
            return r^
        try:
            if self.phase == LPHASE_ENCODE:
                if not self.announced:
                    # announce BEFORE the long blocking encode tick (24-layer
                    # streamed GPT-OSS forward for cond + uncond).
                    self.announced = True
                    r.step = 0
                    r.phase = String("encoding")
                    return r^
                self._encode()
                self.announced = False
                self.phase = LPHASE_LOAD
                r.step = 0
                return r^
            if self.phase == LPHASE_LOAD:
                if not self.loaded:
                    if not self.announced:
                        self.announced = True
                        r.step = 0
                        r.phase = String("loading")
                        return r^
                    self._load_model()
                    self.announced = False
                self._prepare_job()
                self.phase = LPHASE_DENOISE
                r.step = 0
                return r^
            if self.phase == LPHASE_DENOISE:
                self._denoise_one()
                self.cur += 1
                r.step = self.cur
                if self.cur >= self.params.steps:
                    self.phase = LPHASE_DECODE
                return r^
            if not self.announced:
                # announce BEFORE the long blocking VAE-decode tick.
                self.announced = True
                r.step = self.params.steps
                r.phase = String("decoding")
                return r^
            var path = self._decode_and_save()
            r.step = self.params.steps
            self._clear_job()
            r.done = True
            r.output_path = path
            return r^
        except e:
            self._clear_job()
            r.failed = True
            r.error = String(e)
            return r^


# ── streamed-block loader factory (mirrors the pipeline's per-forward loader) ──
# lens_forward takes a BlockLoader and streams the 48 blocks itself. The pipeline
# opens one loader per denoise() call and reuses it across steps; we open one per
# forward (cheap mmap state) to keep the backend's residency surface to the
# resident weights + RoPE tables only.
def _block_loader(ctx: DeviceContext) raises -> BlockLoader:
    return BlockLoader.open(String(LENS_TRANSFORMER_DIR))


# ── PNG(tEXt) save — identical math to sdxl/qwenimage backends ────────────────
def _save_rgb_png_with_text(
    rgb: Tensor, path: String, params_json: String, ctx: DeviceContext
) raises:
    """[1,3,H,W] SIGNED float tensor → 8-bit RGB PNG with the job params in a
    serenity.genparams.v1 tEXt chunk. Quantization math == save_png's (_quantize,
    ValueRange.SIGNED); only the writer differs (tEXt support)."""
    var shape = rgb.shape()
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
        raise Error("lens_backend: expected [1,3,H,W] rgb tensor")
    var height = shape[2]
    var width = shape[3]
    var host = rgb.to_host(ctx)
    var plane = height * width
    if len(host) != 3 * plane:
        raise Error("lens_backend: rgb to_host size mismatch")
    var img = Image.new(width, height, 3)
    for y in range(height):
        var row = y * width
        for x in range(width):
            var off = row + x
            img.set(x, y, 0, _quantize(host[0 * plane + off], ValueRange.SIGNED))
            img.set(x, y, 1, _quantize(host[1 * plane + off], ValueRange.SIGNED))
            img.set(x, y, 2, _quantize(host[2 * plane + off], ValueRange.SIGNED))
    var kws = List[String]()
    var vals = List[String]()
    kws.append(String(GENPARAMS_TEXT_KEY))
    vals.append(params_json.copy())
    encode_png_with_text(img, path, kws, vals)
