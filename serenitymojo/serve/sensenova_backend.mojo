# serenitymojo.serve.sensenova_backend — the real SenseNova-U1-8B-MoT GenBackend.
#
# Wraps the VERIFIED SenseNova-U1 T2I pipeline (serenitymojo/pipeline/
# sensenova_u1_gen_real.mojo + sensenova_u1_gen_smoke.mojo + models/dit/
# sensenova_u1.mojo) behind the pull-based GenBackend seam (backend.mojo).
#
# Unlike the gen_smoke/gen_real pipelines — where the PROMPT is a comptime
# constant — THIS backend tokenizes + encodes the REAL params.prompt at runtime.
# The "text encoder" here IS the model itself: SenseNova-U1 is a Mixture-of-
# TRANSFORMERS (the Qwen3 backbone runs the BASE path = forward_und for the text
# prefix, populating a per-layer KV cache; the _mot_gen path = forward_gen runs
# the image tokens each ODE step, never updating the cache). There is NO separate
# text encoder and NO VAE — SenseNova-U1 is a PIXEL-SPACE flow-matching model
# (fm_head predicts patch pixels directly; patchify/unpatchify operate in RGB
# pixel space). So decode is just unpatchify + denorm + PNG — no tiled VAE.
#
# Numeric conventions are reused VERBATIM from sensenova_u1_gen_real.mojo (the
# patchify/unpatchify, time schedule, t2i query, noise_scale, CFG velocity and
# Euler update are copied unchanged; only the prompt source and the pull-based
# tick wrapper differ).
#
# Residency model (single-GPU):
#   * The SenseNovaU1[L_TOKENS, TEXT_LEN] handle (sharded checkpoint mmap +
#     PlannedBlockLoader + the T2I-resident shared tensors ~1.2 GiB) is loaded
#     ONCE (first job) and STAYS RESIDENT across jobs. Per-layer transformer
#     weights stream from disk one block at a time (PlannedBlockLoader), so the
#     full 8B weight set is NOT all-resident — the residency win is the loader
#     state + shared tensors, like Qwen-Image's offloader handle.
#   * The two prefix KV caches (cond + uncond, from forward_und) are built PER
#     JOB inside ENCODE and freed at job end.
#
# step() state machine: LOAD (model, once, announced phase="loading")
#   → ENCODE (per-job: tokenize params.prompt + 2x forward_und, blocking,
#     announced phase="encoding") → DENOISE×steps (one CFG dual forward_gen +
#     Euler update per tick) → DECODE (unpatchify + PNG, announced
#     phase="decoding") → done. cancel() makes the next step() return cancelled
#   and frees all per-job tensors.
#
# Size support: 512x512 ONLY (geometry is comptime-fixed here: L_TOKENS=256,
# TEXT_LEN=320 cap; the model derives SDPA shapes from runtime tensor shapes but
# the comptime struct tags are pinned to this resolution). steps/cfg/seed ARE
# honored at runtime (the denoise loop reads params.steps/cfg/seed).
#
# Sampler/scheduler: SenseNova-U1 runs its OWN fixed flow-match Euler schedule
# (the exponential time-shift schedule built inside the denoise loop), not a
# UI-selectable sampler. The registry has no `sensenova` admission entry (and we
# must not edit it), so we do NOT gate on it; instead any SET advanced-sampling
# knob the model cannot honor is warned-loud (never silently dropped).
#
# LoRA / img2img / edit / inpaint / mask / reference: NOT supported — rejected at
# admission so they never silently no-op.
#
# WEIGHTS RISK: the checkpoint at WEIGHTS_DIR may be ABSENT (HF cache holds only
# config.json + the safetensors index, no shards; /home/alex/.serenity/models/
# sensenova_u1/ does not exist). When weights are missing, the first job FAILS
# LOUD at SenseNovaU1.load (a clear BlockLoader.open error) — it does NOT fake an
# image.
#
# Mojo 1.0.0b1, NVIDIA GPU. BF16 storage, F32 accumulation in ops.

from std.ffi import external_call
from std.gpu.host import DeviceContext
from std.memory import alloc, ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import BytePtr
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current, cu_mem_get_info
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import (
    reshape, permute, add, sub, mul_scalar, add_scalar,
)
from serenitymojo.models.dit.sensenova_u1 import (
    SenseNovaU1, SenseNovaU1Config, KvCache,
)
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.serve.backend import (
    GenBackend, JobParams, StepResult, reject_unsupported_common_runtime_params,
    reject_unsupported_reference_image_params, reject_unsupported_mask_image_params,
    reject_unsupported_inpaint_conditioning_params,
    reject_unsupported_qwen_edit_conditioning_params,
    reject_unsupported_conditioning_mask_params, reject_unsupported_lanpaint_params,
    warn_unsupported_advanced_sampling_params,
)


# ── verified model + tokenizer paths (match sensenova_u1_gen_real.mojo) ──
comptime WEIGHTS_DIR = "/home/alex/.serenity/models/sensenova_u1"
comptime VOCAB_JSON = "/home/alex/.serenity/models/sensenova_u1/vocab.json"
comptime MERGES_TXT = "/home/alex/.serenity/models/sensenova_u1/merges.txt"
comptime ADDED_TOKENS_JSON = "/home/alex/.serenity/models/sensenova_u1/added_tokens.json"

# ── geometry (512x512, matching sensenova_u1_gen_smoke's coherence geometry) ──
comptime WIDTH = 512
comptime HEIGHT = 512
comptime PATCH = 16
comptime MERGE = 2
comptime GRID_H = HEIGHT // PATCH        # 32
comptime GRID_W = WIDTH // PATCH         # 32
comptime TOKEN_H = GRID_H // MERGE       # 16
comptime TOKEN_W = GRID_W // MERGE       # 16
comptime L_TOKENS = TOKEN_H * TOKEN_W    # 256
# TEXT_LEN is a struct-tag upper bound (the model dimensions attention from the
# runtime token count); pinned to the gen_real conservative cap. The exact cond/
# uncond prefix lengths are printed, never asserted.
comptime TEXT_LEN = 320
comptime FM_OUT = (PATCH * MERGE) * (PATCH * MERGE) * 3  # 3072

comptime DEFAULT_CFG = Float32(4.0)
comptime TIMESTEP_SHIFT = Float32(3.0)
comptime T_EPS = Float32(0.05)

# The real system message conditioning the gen prefix (sensenova_u1_gen.rs:27-46;
# copied verbatim from sensenova_u1_gen_real.mojo).
comptime SYSTEM_MESSAGE_FOR_GEN = String(
    "You are an image generation and editing assistant that accurately understands and executes "
    + "user intent.\n\nYou support two modes:\n\n"
    + "1. Think Mode:\nIf the task requires reasoning, you MUST start with a <think></think> block. "
    + "Put all reasoning inside the block using plain text. DO NOT include any image tags. "
    + "Keep it reasonable and directly useful for producing the final image.\n\n"
    + "2. Non-Think Mode:\nIf no reasoning is needed, directly produce the final image.\n\n"
    + "Task Types:\n\nA. Text-to-Image Generation:\n"
    + "- Generate a high-quality image based on the user's description.\n"
    + "- Ensure visual clarity, semantic consistency, and completeness.\n"
    + "- DO NOT introduce elements that contradict or override the user's intent.\n\n"
    + "B. Image Editing:\n"
    + "- Use the provided image(s) as input or reference for modification or transformation.\n"
    + "- The result can be an edited image or a new image based on the reference(s).\n"
    + "- Preserve all unspecified attributes unless explicitly changed.\n\n"
    + "General Rules:\n"
    + "- For any visible text in the image, follow the language specified for the rendered text in "
    + "the user's description, not the language of the prompt. If no language is specified, use the "
    + "user's input language."
)


comptime SPHASE_IDLE = 0
comptime SPHASE_LOAD = 1
comptime SPHASE_ENCODE = 2
comptime SPHASE_DENOISE = 3
comptime SPHASE_DECODE = 4


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
        String("echo -n '[sensenova][vram] ") + tag
        + ": ' && nvidia-smi --query-gpu=memory.used --format=csv,noheader"
    )


# ── patchify / unpatchify (verbatim from sensenova_u1_gen_real.mojo) ─────────
def _patchify(
    img: Tensor, p: Int, channel_first: Bool, ctx: DeviceContext
) raises -> Tensor:
    var dims = img.shape()
    var b = dims[0]
    var hh = dims[2]
    var ww = dims[3]
    var gh = hh // p
    var gw = ww // p
    var x6 = _reshape6(img, b, 3, gh, p, gw, p, ctx)
    var perm = List[Int]()
    if channel_first:
        perm.append(0); perm.append(2); perm.append(4)
        perm.append(1); perm.append(3); perm.append(5)
    else:
        perm.append(0); perm.append(2); perm.append(4)
        perm.append(3); perm.append(5); perm.append(1)
    var xp = permute(x6, perm, ctx)
    return _reshape3(xp, b, gh * gw, p * p * 3, ctx)


def _unpatchify(
    x: Tensor, p: Int, h: Int, w: Int, ctx: DeviceContext
) raises -> Tensor:
    var dims = x.shape()
    var b = dims[0]
    var gh = h // p
    var gw = w // p
    var x6 = _reshape6(x, b, gh, gw, p, p, 3, ctx)
    var perm = List[Int]()
    perm.append(0); perm.append(5); perm.append(1)
    perm.append(3); perm.append(2); perm.append(4)
    var xp = permute(x6, perm, ctx)
    return _reshape4(xp, b, 3, gh * p, gw * p, ctx)


def _reshape3(x: Tensor, a: Int, b: Int, c: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(a); sh.append(b); sh.append(c)
    return reshape(x, sh^, ctx)


def _reshape4(x: Tensor, a: Int, b: Int, c: Int, d: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(a); sh.append(b); sh.append(c); sh.append(d)
    return reshape(x, sh^, ctx)


def _reshape6(
    x: Tensor, a: Int, b: Int, c: Int, d: Int, e: Int, f: Int, ctx: DeviceContext
) raises -> Tensor:
    var sh = List[Int]()
    sh.append(a); sh.append(b); sh.append(c)
    sh.append(d); sh.append(e); sh.append(f)
    return reshape(x, sh^, ctx)


def _reshape_pixel(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dims = x.shape()
    var sh = List[Int]()
    sh.append(dims[0] * dims[1])
    sh.append(dims[2])
    return reshape(x, sh^, ctx)


# Standard exponential time-shift schedule (verbatim from gen_real).
def _apply_time_schedule(
    t_uniform: List[Float32], shift: Float32
) raises -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(t_uniform)):
        var t = t_uniform[i]
        var sigma = Float32(1.0) - t
        var shifted = shift * sigma / (Float32(1.0) + (shift - Float32(1.0)) * sigma)
        out.append(Float32(1.0) - shifted)
    return out^


# Chat-template t2i query assembly (verbatim from gen_real).
def _t2i_query(system: String, user: String, append: String) -> String:
    var q = String("")
    if system.byte_length() > 0:
        q += "<|im_start|>system\n"
        q += system
        q += "<|im_end|>\n"
    q += "<|im_start|>user\n"
    q += user
    q += "<|im_end|>\n<|im_start|>assistant\n"
    q += append
    return q^


# ── per-job conditioning bundle: the two prefix KV caches ────────────────────
struct SensenovaCaps(Movable):
    var cond: KvCache
    var uncond: KvCache

    def __init__(out self, var cond: KvCache, var uncond: KvCache):
        self.cond = cond^
        self.uncond = uncond^


struct SensenovaBackend(GenBackend, Movable):
    var ctx: DeviceContext

    # ── resident across jobs (model handle, loaded once on first job) ──
    var loaded: Bool
    var model: List[ArcPointer[SenseNovaU1[L_TOKENS, TEXT_LEN]]]  # 0/1

    # ── per-job state (cleared on done/failed/cancelled) ──
    var active: Bool
    var cancel_flag: Bool
    var phase: Int
    var announced: Bool
    var cur: Int
    var params: JobParams
    var cfg: Float32
    var noise_scale: Float32
    var s_norm: Float32
    var caps: List[ArcPointer[SensenovaCaps]]   # 0/1 (cond + uncond KV caches)
    var img: List[ArcPointer[Tensor]]           # 0/1 ([1,3,H,W] BF16 image)
    var tsched: List[Float32]                   # NUM_STEPS+1 timestep grid

    def __init__(out self) raises:
        self.ctx = DeviceContext()
        self.loaded = False
        self.model = List[ArcPointer[SenseNovaU1[L_TOKENS, TEXT_LEN]]]()
        self.active = False
        self.cancel_flag = False
        self.phase = SPHASE_IDLE
        self.announced = False
        self.cur = 0
        self.params = JobParams()
        self.cfg = DEFAULT_CFG
        self.noise_scale = Float32(1.0)
        self.s_norm = Float32(0.0)
        self.caps = List[ArcPointer[SensenovaCaps]]()
        self.img = List[ArcPointer[Tensor]]()
        self.tsched = List[Float32]()

    def backend_name(self) -> String:
        return String("sensenova")

    def model_name(self) -> String:
        return String("SenseNova-U1")

    def resident_model(self) -> String:
        """Best-effort match to a /v1/models scan entry for the resident
        checkpoint (the sensenova_u1/ directory). The dispatch is wired by the
        orchestrator regardless of whether the scan lists it."""
        return String("sensenova_u1") if self.loaded else String("")

    # ── job admission ─────────────────────────────────────────────────────────
    def start(mut self, params: JobParams) raises:
        if self.active:
            raise Error("SensenovaBackend.start: a job is already running")
        reject_unsupported_common_runtime_params(params, String("sensenova"))
        reject_unsupported_reference_image_params(params, String("sensenova"))
        reject_unsupported_inpaint_conditioning_params(params, String("sensenova"))
        reject_unsupported_qwen_edit_conditioning_params(params, String("sensenova"))
        reject_unsupported_conditioning_mask_params(params, String("sensenova"))
        reject_unsupported_mask_image_params(params, String("sensenova"))
        reject_unsupported_lanpaint_params(params, String("sensenova"))
        # 512x512 only: the geometry comptime tags (L_TOKENS=256, TEXT_LEN=320)
        # are pinned to this resolution. Reject other sizes up front (no false
        # advertising) — a resolution change needs a recompiled specialization.
        if not (params.width == 512 and params.height == 512):
            raise Error(
                String("sensenova: unsupported size ") + String(params.width)
                + "x" + String(params.height)
                + " — only 512x512 is served (the SenseNova-U1 geometry tags"
                + " L_TOKENS=256/TEXT_LEN=320 are comptime-fixed; resolution"
                + " changes need a recompiled specialization)"
            )
        if len(params.loras) > 0:
            raise Error(
                "sensenova: LoRA is not supported for SenseNova-U1 yet"
                " (no LoRA overlay path wired); submit without a LoRA"
            )
        if params.init_image.byte_length() > 0:
            raise Error(
                "sensenova: img2img is not supported for SenseNova-U1 yet;"
                " submit without an init image"
            )
        # SenseNova-U1 runs its own fixed flow-match Euler schedule (the
        # exponential time-shift schedule) — sampler/scheduler are not selectable
        # for this model, and the registry has no `sensenova` admission entry.
        # Warn-loud (never silently drop) on any advanced-sampling knob the model
        # cannot honor.
        warn_unsupported_advanced_sampling_params(params, String("sensenova"), List[String]())
        self.params = params.copy()
        self.cfg = Float32(params.cfg) if params.cfg > 0.0 else DEFAULT_CFG
        self.active = True
        self.cancel_flag = False
        self.cur = 0
        self.announced = False
        self.phase = SPHASE_LOAD

    def cancel(mut self):
        self.cancel_flag = True

    def between_jobs_trim(mut self) raises:
        """Reclaim the per-job transient peak (the two prefix KV caches, the
        per-step gen-path activations, the streamed per-layer block buffers) back
        to the OS via cuMemPoolTrimTo. The resident model handle + shared tensors
        have live suballocations and are NOT reclaimed."""
        var before = cu_mem_get_info()
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        self.ctx.synchronize()
        var after = cu_mem_get_info()
        print("[sensenova] between-jobs trim: used",
              before.used_bytes() // (1024 * 1024), "->",
              after.used_bytes() // (1024 * 1024), "MiB (reclaimed",
              (before.used_bytes() - after.used_bytes()) // (1024 * 1024), "MiB)")

    # ── per-job prep ───────────────────────────────────────────────────────────
    def _load_model(mut self) raises:
        """Load the SenseNovaU1 handle (once; stays resident). FAILS LOUD here if
        the checkpoint shards are absent (BlockLoader.open error) — no fake
        image."""
        if self.loaded:
            return
        _print_vram("before SenseNova-U1 load")
        print("[sensenova] loading SenseNovaU1[", L_TOKENS, ",", TEXT_LEN,
              "] from", WEIGHTS_DIR)
        self.model = List[ArcPointer[SenseNovaU1[L_TOKENS, TEXT_LEN]]]()
        self.model.append(
            ArcPointer(SenseNovaU1[L_TOKENS, TEXT_LEN].load(WEIGHTS_DIR, self.ctx))
        )
        self.loaded = True
        _print_vram("after SenseNova-U1 load (resident)")

    def _encode(mut self) raises:
        """Runtime REAL text encode: tokenize params.prompt with the model's own
        Qwen3 tokenizer, then run forward_und over the cond + uncond prefixes to
        build the two KV caches. SenseNova-U1's backbone IS the text encoder."""
        var tok = Qwen3Tokenizer(
            String(VOCAB_JSON), String(MERGES_TXT), String(ADDED_TOKENS_JSON)
        )
        # cond query = full system message + the REAL params.prompt + non-think
        # assistant append (mirrors sensenova_u1_gen_real.mojo's run_t2i, with the
        # comptime PROMPT replaced by the runtime prompt).
        var cond_ids = tok.encode(
            _t2i_query(
                SYSTEM_MESSAGE_FOR_GEN, self.params.prompt,
                String("<think>\n\n</think>\n\n<img>")
            )
        )
        var uncond_ids = tok.encode(_t2i_query(String(""), String(""), String("<img>")))
        print("[sensenova] cond tokens=", len(cond_ids),
              " uncond tokens=", len(uncond_ids))
        if len(cond_ids) > TEXT_LEN:
            print("[sensenova] WARN: cond prefix length", len(cond_ids),
                  "exceeds TEXT_LEN tag", TEXT_LEN,
                  "(attention is runtime-dimensioned so this still runs; the tag"
                  " is documentation only)")

        var cond_cache = self.model[0][].forward_und(cond_ids, self.ctx)
        var uncond_cache = self.model[0][].forward_und(uncond_ids, self.ctx)
        print("[sensenova] prefix forwards done")
        _print_vram("after prefix forwards (KV caches built)")
        self.caps = List[ArcPointer[SensenovaCaps]]()
        self.caps.append(ArcPointer(SensenovaCaps(cond_cache^, uncond_cache^)))

    def _prepare_job(mut self) raises:
        """Resolution noise scale + seeded scaled initial image + timestep grid
        (honors steps + seed)."""
        self.noise_scale = self.model[0][].compute_noise_scale(GRID_H, GRID_W)
        self.s_norm = self.noise_scale / self.model[0][].config.noise_scale_max
        print("[sensenova] noise_scale=", self.noise_scale, " s_norm=", self.s_norm)

        var noise_sh = List[Int]()
        noise_sh.append(1); noise_sh.append(3)
        noise_sh.append(HEIGHT); noise_sh.append(WIDTH)
        var img = randn(noise_sh^, UInt64(self.params.seed), STDtype.BF16, self.ctx)
        img = mul_scalar(img, self.noise_scale, self.ctx)
        self.img = List[ArcPointer[Tensor]]()
        self.img.append(ArcPointer(img^))

        self.tsched = List[Float32]()
        var t_uniform = List[Float32]()
        for i in range(self.params.steps + 1):
            t_uniform.append(Float32(i) / Float32(self.params.steps))
        self.tsched = _apply_time_schedule(t_uniform, TIMESTEP_SHIFT)
        print(
            "[sensenova] job", self.params.job_id, ":", self.params.steps,
            "steps, cfg", self.cfg, "seed", self.params.seed,
            "size", self.params.width, "x", self.params.height,
        )

    # ── one denoise step (CFG dual forward_gen + Euler) ────────────────────────
    # Verbatim from sensenova_u1_gen_real.mojo's per-step body, with the prompt-
    # independent geometry constants substituted.
    def _denoise_one(mut self) raises:
        var step = self.cur
        var cfg = self.model[0][].config
        var t = self.tsched[step]
        var t_next = self.tsched[step + 1]

        var img = self.img[0][].clone(self.ctx)
        var z = _patchify(img, PATCH * MERGE, False, self.ctx)       # [1,L,3072]
        var pixel_values = _patchify(img, PATCH, True, self.ctx)     # [1,gh*gw,768]
        var pixel_flat = _reshape_pixel(pixel_values, self.ctx)      # [gh*gw,768]

        var image_embeds = self.model[0][].extract_feature_gen(
            pixel_flat, GRID_H, GRID_W, self.ctx
        )
        var t_vec = List[Float32]()
        for _ in range(L_TOKENS):
            t_vec.append(t)
        var t_sh = List[Int]()
        t_sh.append(L_TOKENS)
        var t_tensor = Tensor.from_host(t_vec, t_sh^, STDtype.F32, self.ctx)
        var t_emb = self.model[0][].time_or_scale_embed(t_tensor, "timestep", self.ctx)
        var t_emb3 = _reshape3(t_emb, 1, L_TOKENS, cfg.hidden_size, self.ctx)

        var s_vec = List[Float32]()
        for _ in range(L_TOKENS):
            s_vec.append(self.s_norm)
        var s_sh = List[Int]()
        s_sh.append(L_TOKENS)
        var s_tensor = Tensor.from_host(s_vec, s_sh^, STDtype.F32, self.ctx)
        var s_emb = self.model[0][].time_or_scale_embed(s_tensor, "noise", self.ctx)
        var s_emb3 = _reshape3(s_emb, 1, L_TOKENS, cfg.hidden_size, self.ctx)

        var additive = add(t_emb3, s_emb3, self.ctx)
        image_embeds = add(image_embeds, additive, self.ctx)

        var h_cond = self.model[0][].forward_gen(
            image_embeds, self.caps[0][].cond.next_t_index,
            TOKEN_H, TOKEN_W, self.caps[0][].cond, self.ctx
        )
        var h_uncond = self.model[0][].forward_gen(
            image_embeds, self.caps[0][].uncond.next_t_index,
            TOKEN_H, TOKEN_W, self.caps[0][].uncond, self.ctx
        )
        var x_cond = self.model[0][].fm_head_forward(h_cond, self.ctx)    # [1,L,3072]
        var x_uncond = self.model[0][].fm_head_forward(h_uncond, self.ctx)

        var denom = Float32(1.0) - t
        if denom < T_EPS:
            denom = T_EPS
        var inv_denom = Float32(1.0) / denom
        var v_cond = mul_scalar(sub(x_cond, z, self.ctx), inv_denom, self.ctx)
        var v_uncond = mul_scalar(sub(x_uncond, z, self.ctx), inv_denom, self.ctx)

        var v_diff = sub(v_cond, v_uncond, self.ctx)
        var v = add(v_uncond, mul_scalar(v_diff, self.cfg, self.ctx), self.ctx)

        var z_next = add(z, mul_scalar(v, t_next - t, self.ctx), self.ctx)
        var img_next = _unpatchify(z_next, PATCH * MERGE, HEIGHT, WIDTH, self.ctx)
        self.img = List[ArcPointer[Tensor]]()
        self.img.append(ArcPointer(img_next^))
        print("[sensenova] step", step + 1, "/", self.params.steps,
              " t=", t, "->", t_next)

    # ── final decode (pixel-space: denorm + PNG; no VAE) ──────────────────────
    def _decode_and_save(mut self) raises -> String:
        var png_path = self.params.out_dir + "/" + self.params.job_id + ".png"
        var img = self.img[0][].clone(self.ctx)
        # Per-job conditioning is dead weight at decode; free before save.
        self.caps = List[ArcPointer[SensenovaCaps]]()
        self.img = List[ArcPointer[Tensor]]()
        # denorm ((img*0.5+0.5)) and save (UNIT range), exactly as gen_real.
        var final_img = add_scalar(
            mul_scalar(img, Float32(0.5), self.ctx), Float32(0.5), self.ctx
        )
        var final_f32 = cast_tensor(final_img, STDtype.F32, self.ctx)
        save_png(final_f32, png_path, self.ctx, ValueRange.UNIT)
        print("[sensenova] saved ->", png_path)
        return png_path

    def _clear_job(mut self):
        self.active = False
        self.phase = SPHASE_IDLE
        self.cur = 0
        self.cancel_flag = False
        self.announced = False
        self.caps = List[ArcPointer[SensenovaCaps]]()
        self.img = List[ArcPointer[Tensor]]()
        self.tsched = List[Float32]()

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
            if self.phase == SPHASE_LOAD:
                if not self.loaded:
                    if not self.announced:
                        self.announced = True
                        r.step = 0
                        r.phase = String("loading")
                        return r^
                    self._load_model()
                    self.announced = False
                self.phase = SPHASE_ENCODE
                r.step = 0
                return r^
            if self.phase == SPHASE_ENCODE:
                if not self.announced:
                    # announce BEFORE the long blocking encode tick (tokenize +
                    # 2x forward_und over the streamed 42-layer prefix).
                    self.announced = True
                    r.step = 0
                    r.phase = String("encoding")
                    return r^
                self._encode()
                self._prepare_job()
                self.announced = False
                self.phase = SPHASE_DENOISE
                r.step = 0
                return r^
            if self.phase == SPHASE_DENOISE:
                self._denoise_one()
                self.cur += 1
                r.step = self.cur
                if self.cur >= self.params.steps:
                    self.phase = SPHASE_DECODE
                return r^
            if not self.announced:
                # announce BEFORE the (short) decode tick.
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
