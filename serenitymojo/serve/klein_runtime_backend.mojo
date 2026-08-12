# serenitymojo.serve.klein_runtime_backend — the real, IN-PROCESS Klein GenBackend.
#
# Sibling to klein_backend.mojo (which runs Klein through a SEPARATE precache
# process + a separate staged sampler process, writing caps_pos/caps_neg sidecar
# files). THIS backend does the Qwen3 tokenize+encode of params.prompt /
# params.negative INLINE — no precache files, no sidecar processes — then runs
# the existing verified Klein sampler denoise + Klein VAE decode + PNG save, all
# in one Mojo process behind the pull-based GenBackend seam (backend.mojo).
#
# What it reuses VERBATIM (no math re-derived):
#   * Qwen3 text encode  — Qwen3Tokenizer + Qwen3Encoder.encode_klein, the EXACT
#     path klein9b_precache_sample_prompts.mojo runs (_klein_template chat
#     wrapper, 512-token pad with PAD_ID=151643, encode_klein stacked layers
#     [8,17,26] -> [1,512,joint]). The only difference: the embedding lands in a
#     device Tensor instead of a cap-cache .bin file.
#   * Klein denoise + VAE + save — sampling/klein_sampler's shared denoise/decode
#     math with a backend-owned TurboPlannedLoader (Euler flow-match, live LoRA,
#     memory-resident blocks, KleinVaeDecoder).
#   * Model arch + paths — read_model_config(klein9b.json | klein4b.json) ->
#     TrainConfig (the single source of truth, same as the staged CLI).
#
# Residency model (24 GB GPU, one big model at a time):
#   * The Qwen3 encoder (~16 GB for 9B / ~8 GB for 4B) is loaded -> used -> freed
#     INSIDE the ENCODE tick (Movable-not-Copyable Qwen3Encoder drops at scope
#     exit in _encode_text_pair). The runtime then trims the CUDA pool before
#     SAMPLE. Only the tiny pos/neg conditioning Tensors ([512,joint] BF16 ~12 MB
#     each for 9B) survive into the SAMPLE tick — so the encoder and the Klein
#     DiT do not intentionally co-reside, exactly like the SDXL/Qwen-Image
#     backends free their encoders before the denoiser loads.
#   * The selected checkpoint's complete block store stays in pinned host RAM
#     across jobs. Per-job base/LoRA GPU weights free before VAE decode, and the
#     two transient block-staging GPU slabs are explicitly discarded. This keeps
#     repeat jobs warm without retaining a full DiT in VRAM during decode.
#
# step() state machine (pull-based announce ticks; the sampler loop emits
# machine-readable progress events for every denoise step over the worker IPC fd):
#   ENCODE  : announce phase="encoding" -> next tick runs the (blocking) Qwen3
#             encode of prompt+negative.
#   SAMPLE  : announce phase="sampling" -> next tick runs klein_sample denoise +
#             VAE decode + PNG save. During denoise the sampler writes
#             {"ev":"progress","step":N,...} lines on the same IPC socket, then
#             the backend re-embeds serenity.genparams.v1 and returns done.
# cancel() flips a flag; the next step() returns cancelled and frees per-job
# state. (klein_sample is a single blocking call — cancellation is honored at
# the tick boundaries, not mid-denoise, matching the staged backend.)
#
# Size support: 512 square plus the finite seven-shape 1MP product ladder (the
# Klein attention shape N_IMG/S/LH/LW is comptime; klein_sample dispatches those
# exact specializations). steps/cfg/seed honored at runtime. One live LoRA is
# supported at the user's requested multiplier through the creator sampler's
# existing lora_multiplier argument; multiple adapters remain uncomposed.
# Native ReferenceLatent edit accepts one 512x512 or
# 1024x1024 source image for both 9B and 4B. The preserved two-reference legacy
# path remains 512x512. Ordinary img2img is rejected loudly.
#
# Mojo 1.0.0b1: `def` not `fn`.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer, UnsafePointer, alloc
from std.builtin.type_aliases import MutExternalOrigin
from std.ffi import external_call
from std.time import perf_counter_ns, sleep

from image.png import encode_png_with_text

from serenitymojo.tensor import Tensor
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current, cu_mem_get_info

from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config, _clone
from serenitymojo.training.train_config import TrainConfig
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.ops.tensor_algebra import concat, reshape
from serenitymojo.sampling.klein_sampler import (
    build_klein_memory_resident_loader, klein_sample_with_loader,
    klein_sample_with_reference_latent,
    klein_sample_with_reference_latents2,
)
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.models.vae.klein_encoder import KleinVaeEncoder
from serenitymojo.training.aspect_buckets import (
    DEFAULT_ASPECT_LADDER_LEN, DEFAULT_ASPECT_LADDER_X100,
    aspect_lat_h_units, aspect_lat_w_units,
)
from serenitymojo.serve.image_io import image_to_signed_nchw
from image.transform import resize_bilinear
from serenitymojo.serve.image_io import decode_image_any
from serenitymojo.io.dtype import STDtype
from serenitymojo.serve.model_scan import LORAS_DIR
from serenitymojo.io.ffi import sys_open, sys_close, O_RDONLY
from serenitymojo.io.cap_cache import save_tensor_bin, load_tensor_bin
from serenitymojo.serve.proc_ipc import (
    build_argv, cstr, sys_execv, sys__exit, sys_waitpid, proc_kill_wait,
    SELF_EXE, SIGKILL, WNOHANG,
)
from net.syscalls import sys_fork, errno_str
from serenitymojo.serve.backend import (
    GenBackend, JobParams, StepResult, reject_unsupported_common_runtime_params,
    reject_unsupported_mask_image_params,
    reject_unsupported_inpaint_conditioning_params,
    reject_unsupported_qwen_edit_conditioning_params,
    reject_unsupported_conditioning_mask_params,
    reject_unsupported_lanpaint_params,
    warn_unsupported_advanced_sampling_params,
)
from serenitymojo.serve.product_manifest import (
    json_bool, json_escape, peak_vram_mib, write_text_file,
)


comptime GENPARAMS_TEXT_KEY = "serenity.genparams.v1"

comptime KLEIN9B_CONFIG = "serenitymojo/configs/klein9b.json"
comptime KLEIN4B_CONFIG = "serenitymojo/configs/klein4b.json"
comptime KLEIN9B_SERENITY_FP8 = (
    "/home/alex/.serenity/models/checkpoints/"
    "flux-2-klein-base-9b_fp8_e4m3fn.safetensors"
)
comptime QWEN4_DIR = "models/qwen3-4b"
comptime QWEN8_DIR = "models/qwen3-8b"
comptime PAD_ID = 151643
comptime SEQ = 512

# Klein comptime resolution specializations (mirrors klein_sample_cli.mojo).
comptime N_TXT = 512
comptime H_9B = 32
comptime H_4B = 24
comptime Dh = 128
comptime KLEIN_PRODUCT_EDGE_UNITS = 16
comptime LH_512 = 32
comptime LW_512 = 32
comptime N_IMG_512 = 1024
comptime S_512 = N_IMG_512 + N_TXT
# ReferenceLatent edit shapes (mirrors klein_sample_cli):
# 1-ref = target + 1 reference block; 2-ref = target + 2 reference blocks.
comptime N_EDIT_IMG_512 = 2 * N_IMG_512
comptime S_EDIT_512 = N_EDIT_IMG_512 + N_TXT
comptime N_EDIT2_IMG_512 = 3 * N_IMG_512
comptime S_EDIT2_512 = N_EDIT2_IMG_512 + N_TXT
comptime LH_1024 = 64
comptime LW_1024 = 64
comptime N_IMG_1024 = 4096
comptime S_1024 = N_IMG_1024 + N_TXT
comptime N_EDIT_IMG_1024 = 2 * N_IMG_1024
comptime S_EDIT_1024 = N_EDIT_IMG_1024 + N_TXT

comptime KRPHASE_IDLE = 0
comptime KRPHASE_ENCODE = 1
comptime KRPHASE_SAMPLE = 2

# Qwen3-8B layer-26 encoding peaks near 22 GiB on the validated 24 GiB host.
# It must run in a fresh process: the long-lived Mojo CUDA pool retained that
# entire high-water mark after object destruction and cuMemPoolTrimTo(0).
comptime _KLEIN_ENCODE_CHILD_TIMEOUT_S = 600.0
comptime _KLEIN_ENCODE_POLL_S = 0.05
# Direct-host inline baseline peaked at 22,175 MiB for the entire worker,
# including its parent context and cap tensors. A 22,000 MiB device-global free
# floor therefore preserves measured headroom while admitting the validated
# card with the desktop compositor resident (22,488 MiB free).
comptime _KLEIN_ENCODE_CHILD_MIN_FREE_BYTES = Int(22000) * 1024 * 1024


def _getpid() -> Int:
    return Int(external_call["getpid", Int32]())


def _unlink_file(path: String):
    _ = external_call["unlink", Int32](cstr(path))


def _lower(s: String) -> String:
    return String(s.lower())


def _klein_shape_supported(width: Int, height: Int) -> Bool:
    if width == 512 and height == 512:
        return True
    comptime for bi in range(DEFAULT_ASPECT_LADDER_LEN):
        comptime X100_BI = DEFAULT_ASPECT_LADDER_X100[bi]
        # Shared helpers return stride-8 VAE latent units. Klein's packed-token
        # grid has 16 output pixels per cell, so it is exactly half that grid.
        comptime LH_BI = aspect_lat_h_units(X100_BI, KLEIN_PRODUCT_EDGE_UNITS) // 2
        comptime LW_BI = aspect_lat_w_units(X100_BI, KLEIN_PRODUCT_EDGE_UNITS) // 2
        if width == LW_BI * 16 and height == LH_BI * 16:
            return True
    return False


def _path_exists(path: String) -> Bool:
    if path == String(""):
        return False
    var fd = sys_open(path, O_RDONLY, 0)
    if fd < 0:
        return False
    _ = sys_close(fd)
    return True


def _require_file(label: String, path: String) raises:
    if not _path_exists(path):
        raise Error(String("klein_runtime: missing ") + label + String(": ") + path)


def _model_variant(model: String) -> String:
    var m = _lower(model)
    if m.find("4b") >= 0:
        return String("4b")
    return String("9b")


def _config_for_variant(variant: String) -> String:
    if variant == String("4b"):
        return String(KLEIN4B_CONFIG)
    return String(KLEIN9B_CONFIG)


def _qwen_dir_for_variant(variant: String) -> String:
    if variant == String("4b"):
        return String(QWEN4_DIR)
    return String(QWEN8_DIR)


def _qwen_cfg_for_variant(variant: String) -> Qwen3Config:
    if variant == String("4b"):
        return Qwen3Config.klein_4b()
    return Qwen3Config.klein_9b()


def _resolve_klein_lora_path(name: String) raises -> String:
    """Resolve a LoRA name to a file path (same search order as klein_backend)."""
    if name == String(""):
        raise Error("klein_runtime: empty LoRA name")
    if _path_exists(name):
        return name.copy()
    if _path_exists(name + ".safetensors"):
        return name + ".safetensors"
    if _path_exists(String(LORAS_DIR) + String("/") + name):
        return String(LORAS_DIR) + String("/") + name
    if _path_exists(String(LORAS_DIR) + String("/") + name + String(".safetensors")):
        return String(LORAS_DIR) + String("/") + name + String(".safetensors")
    raise Error(
        String("klein_runtime: LoRA file not found: ") + name
        + String(" (tried as a path and under ") + String(LORAS_DIR) + String(")")
    )


def _klein_lora_path(params: JobParams) raises -> String:
    """The single supported LoRA path ("" = base).

    The creator sampler already accepts a runtime lora_multiplier, so admission
    must not hard-code the user weight to 1.0.
    """
    if len(params.loras) == 0:
        return String("")
    if len(params.loras) > 1:
        raise Error(
            "klein_runtime: exactly one LoRA is supported (the sampler applies a"
            " single live adapter); submit one LoRA"
        )
    return _resolve_klein_lora_path(params.loras[0].name)


# ── Qwen3 inline tokenize: chat-template wrap + 512-token pad (VERBATIM from
#    klein9b_precache_sample_prompts._klein_template / _tokenize_512). ──────────
def _klein_template(prompt: String) -> String:
    return (
        String("<|im_start|>user\n")
        + prompt
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _tokenize_512(tok: Qwen3Tokenizer, label: String, prompt: String) raises -> List[Int]:
    var ids_full = tok.encode(_klein_template(prompt))
    if len(ids_full) > SEQ:
        raise Error(String("klein_runtime: prompt too long for 512 tokens: ") + label)
    var ids = List[Int](capacity=SEQ)
    for i in range(len(ids_full)):
        ids.append(ids_full[i])
    for _ in range(SEQ - len(ids_full)):
        ids.append(PAD_ID)
    print("[klein_runtime] ", label, " tokens ", len(ids_full), " -> ", SEQ)
    return ids^


def _encode_klein26(enc: Qwen3Encoder, ids: List[Int], ctx: DeviceContext) raises -> Tensor:
    """Klein stacked conditioning from layers [8,17,26], stopping the forward at
    layer 26 — identical output to encode_klein (running layers past 26 cannot
    change an earlier state) but pairs with Qwen3Encoder.load(..., max_layer=26)
    below, which skips lm_head + layers 27..35 (~4.8GB): the difference between
    the 16GB 5080 and OOM (same recipe as klein9b_precache_sample_prompts)."""
    var states = enc.encode_layer_states(ids, ctx, 26)
    var h8 = _clone(states[8][], ctx)
    var h17 = _clone(states[17][], ctx)
    var h26 = _clone(states[26][], ctx)
    return concat(2, ctx, h8, h17, h26)


def _encode_text_pair(
    variant: String, prompt: String, negative: String, joint: Int, ctx: DeviceContext,
    mut pos_out: List[ArcPointer[Tensor]], mut neg_out: List[ArcPointer[Tensor]],
) raises:
    """Load Qwen3, encode both captions, and APPEND the reshaped [N_TXT, joint]
    text tensors into pos_out/neg_out. Keeping this in a helper makes the encoder
    lifetime end (enc drops at return) before the caller trims the CUDA pool and
    starts Klein sampling.

    Appending straight into the ArcPointer slots (instead of returning a Movable
    struct of two Tensors) is the fix for the 2026-06-27 regression that kept the
    klein worker from rebuilding: current Mojo rejects the auto-synthesized
    destructor of such a struct ("field 'pair.pos.buf._handle' destroyed out of the
    middle of a value"). Moving the locals into ArcPointers sidesteps the struct
    entirely."""
    var qwen_dir = _qwen_dir_for_variant(variant)
    var qwen_cfg = _qwen_cfg_for_variant(variant)
    _require_file(String("Qwen3 tokenizer"), qwen_dir + String("/tokenizer.json"))

    var tok = Qwen3Tokenizer(qwen_dir + String("/tokenizer.json"))
    # max_layer=26: the klein taps are layers [8,17,26]; skipping lm_head +
    # layers 27..35 is what makes the 9B encoder fit the 16GB 5080 (the full
    # load is ~16.4GB and OOMs before denoise ever starts).
    var enc = Qwen3Encoder.load(qwen_dir, qwen_cfg, ctx, 26)

    var pos_ids = _tokenize_512(tok, String("pos"), prompt)
    var neg_ids = _tokenize_512(tok, String("neg"), negative)
    # _encode_klein26 -> [1, 512, joint]; reshape to [N_TXT, joint] (the shape
    # klein_sample's pos_txt/neg_txt expect, exactly as klein_sample_cli's
    # _load_pos_txt/_load_neg_txt do).
    var pos_full = _encode_klein26(enc, pos_ids, ctx)
    var neg_full = _encode_klein26(enc, neg_ids, ctx)

    var txt_sh = List[Int]()
    txt_sh.append(N_TXT)
    txt_sh.append(joint)
    var pos2 = reshape(pos_full, txt_sh.copy(), ctx)
    var neg2 = reshape(neg_full, txt_sh.copy(), ctx)
    pos_out.append(ArcPointer(pos2^))
    neg_out.append(ArcPointer(neg2^))


def klein_encode_child_run(
    prefix: String, variant: String, prompt: String, negative: String, joint: Int,
) raises:
    """Fresh-process Qwen3 encode used by the persistent Serenity worker.

    The tensors are serialized as raw BF16 cap-cache bytes. Process exit is the
    ownership boundary that releases every encoder allocation before the parent
    starts Klein denoising.
    """
    var ctx = DeviceContext()
    var pos = List[ArcPointer[Tensor]]()
    var neg = List[ArcPointer[Tensor]]()
    _encode_text_pair(variant, prompt, negative, joint, ctx, pos, neg)
    save_tensor_bin(pos[0][], prefix + String(".pos.bin"), ctx)
    save_tensor_bin(neg[0][], prefix + String(".neg.bin"), ctx)
    print("[klein-encode-child] wrote caps", prefix)


def _encode_text_pair_subprocess(
    variant: String, prompt: String, negative: String, joint: Int,
    ctx: DeviceContext,
    mut pos_out: List[ArcPointer[Tensor]],
    mut neg_out: List[ArcPointer[Tensor]],
) raises:
    """Encode on the GPU in a fork+exec child and load exact BF16 caps.

    Unlike the old in-process fallback, this fails clearly when another process
    has consumed the encoder's required VRAM. Retrying inline would recreate the
    measured 22 GiB retained-pool failure and make the later decode OOM.
    """
    var free_bytes = cu_mem_get_info().free_bytes
    if free_bytes < _KLEIN_ENCODE_CHILD_MIN_FREE_BYTES:
        raise Error(
            String("klein_runtime: Qwen3 GPU encode needs ")
            + String(_KLEIN_ENCODE_CHILD_MIN_FREE_BYTES // (1024 * 1024))
            + String(" MiB free, found ")
            + String(free_bytes // (1024 * 1024))
            + String(" MiB")
        )

    var prefix = String("/tmp/serenity_klein_caps_") + String(_getpid())
    var pos_path = prefix + String(".pos.bin")
    var neg_path = prefix + String(".neg.bin")
    _unlink_file(pos_path)
    _unlink_file(neg_path)

    var args = List[String]()
    args.append(SELF_EXE)
    args.append(String("encode-child"))
    args.append(prefix)
    args.append(variant)
    args.append(prompt)
    args.append(negative)
    args.append(String(joint))
    var argv = build_argv(args)
    var path = cstr(SELF_EXE)

    print("[klein_runtime] fork Qwen3 encoder child (parent pid", _getpid(), ")")
    var pid = sys_fork()
    if pid == 0:
        _ = sys_execv(path, argv)
        sys__exit(127)
    if pid < 0:
        raise Error(String("klein_runtime: Qwen3 encoder fork failed: ") + errno_str())

    var st = alloc[Int32](1)
    var stp = rebind[UnsafePointer[Int32, MutExternalOrigin]](st)
    var waited = 0.0
    var reaped = Int32(0)
    while waited < _KLEIN_ENCODE_CHILD_TIMEOUT_S:
        reaped = sys_waitpid(pid, stp, WNOHANG)
        if reaped == pid or reaped < 0:
            break
        sleep(_KLEIN_ENCODE_POLL_S)
        waited += _KLEIN_ENCODE_POLL_S
    var status = Int(st[0])
    st.free()

    if reaped != pid:
        proc_kill_wait(pid, SIGKILL)
        _unlink_file(pos_path)
        _unlink_file(neg_path)
        raise Error("klein_runtime: Qwen3 encoder child timed out")
    var exited_ok = (status & 0x7F) == 0 and ((status >> 8) & 0xFF) == 0
    if not exited_ok:
        _unlink_file(pos_path)
        _unlink_file(neg_path)
        raise Error(
            String("klein_runtime: Qwen3 encoder child failed, status ")
            + String(status)
        )

    try:
        var pos = load_tensor_bin(pos_path, ctx)
        var neg = load_tensor_bin(neg_path, ctx)
        _unlink_file(pos_path)
        _unlink_file(neg_path)
        pos_out.append(ArcPointer(pos^))
        neg_out.append(ArcPointer(neg^))
        print("[klein_runtime] encoder child reaped; exact BF16 caps loaded")
    except e:
        _unlink_file(pos_path)
        _unlink_file(neg_path)
        raise Error(String("klein_runtime: encoder cap read-back failed: ") + String(e))


def _embed_genparams_in_png(path: String, params_json: String) raises:
    """Re-open the PNG klein_sample saved (plain RGB) and rewrite it WITH a
    serenity.genparams.v1 tEXt chunk. Same approach as klein_backend.mojo (the
    klein sampler's save_image has no tEXt support)."""
    var img = decode_image_any(path)
    var keys = List[String]()
    var vals = List[String]()
    keys.append(String(GENPARAMS_TEXT_KEY))
    vals.append(params_json.copy())
    encode_png_with_text(img, path, keys, vals)


struct KleinRuntimeBackend(GenBackend, Movable):
    var ctx: DeviceContext

    # ── per-job state (cleared on done/failed/cancelled) ──
    var active: Bool
    var cancel_flag: Bool
    var phase: Int
    var announced: Bool
    var params: JobParams
    var variant: String
    var config_path: String
    var cfg: List[ArcPointer[TrainConfig]]   # 0/1 (per-job, loaded at admission)
    var lora_path: String
    var lora_multiplier: Float32
    var out_png: String
    var job_t0_ns: UInt
    var encode_seconds: Float64
    var sample_decode_seconds: Float64
    var total_vram_bytes: Int
    var min_free_bytes: Int
    var progress_fd: Int32
    # encoded conditioning, produced in ENCODE, consumed in SAMPLE.
    # pos/neg are [N_TXT, joint] (already reshaped for klein_sample).
    var pos_txt: List[ArcPointer[Tensor]]    # 0/1
    var neg_txt: List[ArcPointer[Tensor]]    # 0/1
    # One exact conditioning entry, matching the existing Z-Image policy.
    # It is tiny (~25 MiB for Klein 9B) and avoids reloading Qwen3 when only
    # seed, steps, guidance, or sampler controls change.
    var cap_cache_variant: String
    var cap_cache_prompt: String
    var cap_cache_negative: String
    var cap_cache_pos: List[ArcPointer[Tensor]]
    var cap_cache_neg: List[ArcPointer[Tensor]]
    # Complete block store survives VAE decode and repeat jobs. GPU staging is
    # discarded before each decode; only pinned host bytes remain resident.
    var loader: List[ArcPointer[TurboPlannedLoader]]
    var loader_checkpoint: String
    var warmed_denoisers: List[String]

    def __init__(out self) raises:
        self.ctx = DeviceContext()
        self.active = False
        self.cancel_flag = False
        self.phase = KRPHASE_IDLE
        self.announced = False
        self.params = JobParams()
        self.variant = String("9b")
        self.config_path = String(KLEIN9B_CONFIG)
        self.cfg = List[ArcPointer[TrainConfig]]()
        self.lora_path = String("")
        self.lora_multiplier = Float32(1.0)
        self.out_png = String("")
        self.job_t0_ns = UInt(0)
        self.encode_seconds = 0.0
        self.sample_decode_seconds = 0.0
        self.total_vram_bytes = 0
        self.min_free_bytes = 0
        self.progress_fd = Int32(-1)
        self.pos_txt = List[ArcPointer[Tensor]]()
        self.neg_txt = List[ArcPointer[Tensor]]()
        self.cap_cache_variant = String("")
        self.cap_cache_prompt = String("")
        self.cap_cache_negative = String("")
        self.cap_cache_pos = List[ArcPointer[Tensor]]()
        self.cap_cache_neg = List[ArcPointer[Tensor]]()
        self.loader = List[ArcPointer[TurboPlannedLoader]]()
        self.loader_checkpoint = String("")
        self.warmed_denoisers = List[String]()

    def backend_name(self) -> String:
        return String("klein")

    def model_name(self) -> String:
        if self.params.model.byte_length() > 0:
            return self.params.model.copy()
        return String("flux2-klein")

    def resident_model(self) -> String:
        if len(self.loader) == 1:
            return self.loader_checkpoint.copy()
        return String("")

    def set_progress_fd(mut self, fd: Int32):
        self.progress_fd = fd

    # ── job admission ─────────────────────────────────────────────────────────
    def start(mut self, params: JobParams) raises:
        if self.active:
            raise Error("KleinRuntimeBackend.start: a job is already running")
        reject_unsupported_common_runtime_params(params, String("klein"))
        reject_unsupported_inpaint_conditioning_params(params, String("klein"))
        reject_unsupported_qwen_edit_conditioning_params(params, String("klein"))
        reject_unsupported_conditioning_mask_params(params, String("klein"))
        reject_unsupported_mask_image_params(params, String("klein"))
        reject_unsupported_lanpaint_params(params, String("klein"))

        var model = _lower(params.model)
        if model.find("flux2-dev") >= 0 or model.find("flux-2-dev") >= 0 or model.find("flux2_dev") >= 0:
            raise Error(
                String("klein_runtime: Flux2-dev model '") + params.model
                + "' is not a Klein model and must not route through the Klein runner"
            )
        # Finite compile-time set: 512 square plus the seven-shape 1MP ladder.
        if not _klein_shape_supported(params.width, params.height):
            raise Error(
                String("klein_runtime: unsupported size ") + String(params.width)
                + "x" + String(params.height)
                + " (the Klein attention shape is comptime; served sizes are 512x512"
                + " plus the seven-shape 1MP product ladder)"
            )
        if params.steps <= 0:
            raise Error("klein_runtime: steps must be positive")
        if params.seed < 0:
            raise Error("klein_runtime: seed must be nonnegative")
        if params.cfg <= 0.0:
            raise Error("klein_runtime: cfg must be positive")
        var sampler = _lower(params.sampler)
        if not (sampler == "euler" or sampler == "flowmatch_euler"):
            raise Error(
                String("klein_runtime: unsupported sampler '") + params.sampler
                + "' (Flux2/Klein runtime sampler is Euler-only)"
            )
        var scheduler = _lower(params.scheduler)
        if not (scheduler == "flux2" or scheduler == "simple"):
            raise Error(
                String("klein_runtime: unsupported scheduler '") + params.scheduler
                + "' (accepted: flux2 or simple)"
            )
        if params.variation_strength > 0.0:
            raise Error("klein_runtime: variation noise is not wired for this backend")
        if params.init_image.byte_length() > 0 and params.reference_latent_count == 0:
            raise Error(
                "klein_runtime: init_image/img2img is not wired here (init_image"
                " is consumed only by a ReferenceLatent edit)"
            )
        # ReferenceLatent edit: one 512² or 1024² source, or the legacy
        # two-source 512² shape, via the parity-gated edit samplers.
        if params.reference_latent_count > 0:
            if params.reference_latent_count > 2:
                raise Error(
                    "klein_runtime: at most 2 reference latents are compiled ("
                    + String(params.reference_latent_count) + " requested)"
                )
            var one_reference_1024 = (
                params.reference_latent_count == 1
                and params.width == 1024
                and params.height == 1024
            )
            var edit_512 = params.width == 512 and params.height == 512
            if not (edit_512 or one_reference_1024):
                raise Error(
                    "klein_runtime: ReferenceLatent edit admits one source at"
                    " 512x512 or 1024x1024; two-source edit remains 512x512"
                )
            if params.reference_image.byte_length() == 0:
                raise Error("klein_runtime: reference_latent_count set but reference_image is empty")
            if params.reference_latent_count == 2 and params.init_image.byte_length() == 0:
                raise Error(
                    "klein_runtime: 2-reference edit needs init_image (ref A) +"
                    " reference_image (ref B) — init_image is empty"
                )
        elif params.reference_image.byte_length() > 0:
            raise Error("klein_runtime: reference_image set but reference_latent_count is 0")
        # Warn-loud (never silently drop) on any advanced-sampling knob set but
        # unsupported by this fixed Euler path.
        warn_unsupported_advanced_sampling_params(params, String("klein"), List[String]())

        self.variant = _model_variant(params.model)
        self.config_path = _config_for_variant(self.variant)
        _require_file(String("Klein model config"), self.config_path)

        # Read the model config (arch + paths) and validate the required files.
        var loaded_cfg = read_model_config(self.config_path)
        # The product selection owns the checkpoint. For 9B, default to the
        # installed Serenity scalar-FP8 artifact instead of the config's legacy
        # BF16 training path when an explicit path was not supplied.
        var selected_checkpoint = params.checkpoint_path.copy()
        if self.variant == String("9b"):
            # The current UI's "Klein 9B Base" preset still resolves the 18.16 GB
            # BF16 file. Product inference deliberately uses the installed
            # 9.28 GB Serenity FP8 counterpart; training configs remain untouched.
            selected_checkpoint = String(KLEIN9B_SERENITY_FP8)
        if selected_checkpoint.byte_length() > 0:
            loaded_cfg.checkpoint = selected_checkpoint.copy()
        _require_file(String("checkpoint"), loaded_cfg.checkpoint)
        _require_file(String("VAE"), loaded_cfg.vae)
        # Sanity: the config head/dim must match the variant comptime specializations.
        if loaded_cfg.head_dim != Dh:
            raise Error(
                String("klein_runtime: config head_dim ") + String(loaded_cfg.head_dim)
                + " != " + String(Dh)
            )
        if self.variant == String("9b") and loaded_cfg.n_heads != H_9B:
            raise Error(
                String("klein_runtime: 9b config n_heads ") + String(loaded_cfg.n_heads)
                + " != " + String(H_9B)
            )
        if self.variant == String("4b") and loaded_cfg.n_heads != H_4B:
            raise Error(
                String("klein_runtime: 4b config n_heads ") + String(loaded_cfg.n_heads)
                + " != " + String(H_4B)
            )

        if (
            len(self.loader) == 1
            and self.loader_checkpoint != loaded_cfg.checkpoint
        ):
            print(
                "[klein_runtime] selected checkpoint changed; dropping old host store"
            )
            self.loader = List[ArcPointer[TurboPlannedLoader]]()
            self.loader_checkpoint = String("")

        self.lora_path = _klein_lora_path(params)
        self.lora_multiplier = (
            Float32(params.loras[0].weight)
            if len(params.loras) == 1
            else Float32(1.0)
        )
        if self.lora_path != String(""):
            _require_file(String("LoRA"), self.lora_path)

        var out_dir = params.out_dir.copy()
        if out_dir == String(""):
            raise Error("klein_runtime: out_dir is required")
        self.out_png = out_dir + String("/") + params.job_id + String(".png")

        self.cfg = List[ArcPointer[TrainConfig]]()
        self.cfg.append(ArcPointer(loaded_cfg^))
        self.params = params.copy()
        self.pos_txt = List[ArcPointer[Tensor]]()
        self.neg_txt = List[ArcPointer[Tensor]]()
        self.active = True
        self.cancel_flag = False
        self.announced = False
        self.phase = KRPHASE_ENCODE
        self.job_t0_ns = perf_counter_ns()
        self.encode_seconds = 0.0
        self.sample_decode_seconds = 0.0
        var mem = cu_mem_get_info()
        self.total_vram_bytes = mem.total_bytes
        self.min_free_bytes = mem.free_bytes
        self._record_vram()

    def cancel(mut self):
        self.cancel_flag = True

    def between_jobs_trim(mut self) raises:
        """Return per-job GPU allocations while retaining the host block store."""
        var before = cu_mem_get_info()
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        self.ctx.synchronize()
        var after = cu_mem_get_info()
        print("[klein_runtime] between-jobs trim: used",
              before.used_bytes() // (1024 * 1024), "->",
              after.used_bytes() // (1024 * 1024), "MiB (reclaimed",
              (before.used_bytes() - after.used_bytes()) // (1024 * 1024), "MiB)")

    def _record_vram(mut self) raises:
        var mem = cu_mem_get_info()
        if self.total_vram_bytes == 0:
            self.total_vram_bytes = mem.total_bytes
        if self.min_free_bytes == 0 or mem.free_bytes < self.min_free_bytes:
            self.min_free_bytes = mem.free_bytes

    def _ensure_loader(mut self) raises:
        if len(self.loader) == 1:
            self.loader[0][].require_all_blocks_memory_resident()
            print(
                "[klein_runtime] reusing complete host store for",
                self.loader_checkpoint,
            )
            return
        print(
            "[klein_runtime] building complete host store from",
            self.cfg[0][].checkpoint,
        )
        var built = build_klein_memory_resident_loader(self.cfg[0][], self.ctx)
        self.loader.append(ArcPointer(built^))
        self.loader_checkpoint = self.cfg[0][].checkpoint.copy()

    def _denoiser_warm_key(self) -> String:
        return (
            self.variant + String(":") + String(self.params.width)
            + String("x") + String(self.params.height)
            + (String(":single") if self.params.cfg == 1.0 else String(":cfg"))
            + (String(":lora") if self.lora_path != String("") else String(":base"))
        )

    def _denoiser_needs_warm(self) -> Bool:
        var key = self._denoiser_warm_key()
        for warmed in self.warmed_denoisers:
            if warmed == key:
                return False
        return True

    def _mark_denoiser_warm(mut self):
        var key = self._denoiser_warm_key()
        for warmed in self.warmed_denoisers:
            if warmed == key:
                return
        self.warmed_denoisers.append(key^)

    def _clear_job(mut self):
        self.active = False
        self.phase = KRPHASE_IDLE
        self.cancel_flag = False
        self.announced = False
        self.cfg = List[ArcPointer[TrainConfig]]()
        self.pos_txt = List[ArcPointer[Tensor]]()
        self.neg_txt = List[ArcPointer[Tensor]]()
        self.lora_path = String("")
        self.lora_multiplier = Float32(1.0)
        self.out_png = String("")

    # ── per-job prep ───────────────────────────────────────────────────────────
    def _encode(mut self) raises:
        """Qwen3 tokenize+encode of params.prompt AND params.negative INLINE.
        Encoder + tokenizer load -> encode_klein -> reshape to [N_TXT, joint] ->
        encoder drops when _encode_text_pair returns, then the pool is trimmed
        before the Klein DiT loads in SAMPLE.
        Mirrors klein9b_precache_sample_prompts._encode_one, but the embeddings
        land in device Tensors (kept in ArcPointers), not cap-cache .bin files."""
        var t0 = perf_counter_ns()
        self._record_vram()
        if (
            len(self.cap_cache_pos) == 1
            and len(self.cap_cache_neg) == 1
            and self.cap_cache_variant == self.variant
            and self.cap_cache_prompt == self.params.prompt
            and self.cap_cache_negative == self.params.negative
        ):
            self.pos_txt = List[ArcPointer[Tensor]]()
            self.neg_txt = List[ArcPointer[Tensor]]()
            self.pos_txt.append(ArcPointer(self.cap_cache_pos[0][].clone(self.ctx)))
            self.neg_txt.append(ArcPointer(self.cap_cache_neg[0][].clone(self.ctx)))
            self.ctx.synchronize()
            self.encode_seconds = Float64(perf_counter_ns() - t0) / 1.0e9
            self._record_vram()
            print("[klein_runtime] conditioning cache HIT")
            return
        var joint = self.cfg[0][].joint_attention_dim
        # Encode INTO the ArcPointer slots directly: _encode_text_pair loads the
        # Qwen3 encoder, encodes pos+neg, appends the [N_TXT, joint] embeddings, and
        # drops the encoder at return (before the pool trim below). Appending avoids
        # a Movable struct-of-two-Tensors return, whose auto-synthesized destructor
        # current Mojo rejects — the 2026-06-27 regression that kept this worker from
        # rebuilding.
        self.pos_txt = List[ArcPointer[Tensor]]()
        self.neg_txt = List[ArcPointer[Tensor]]()
        _encode_text_pair_subprocess(
            self.variant, self.params.prompt, self.params.negative, joint, self.ctx,
            self.pos_txt, self.neg_txt,
        )
        self.cap_cache_pos = List[ArcPointer[Tensor]]()
        self.cap_cache_neg = List[ArcPointer[Tensor]]()
        self.cap_cache_pos.append(ArcPointer(self.pos_txt[0][].clone(self.ctx)))
        self.cap_cache_neg.append(ArcPointer(self.neg_txt[0][].clone(self.ctx)))
        self.cap_cache_variant = self.variant.copy()
        self.cap_cache_prompt = self.params.prompt.copy()
        self.cap_cache_negative = self.params.negative.copy()

        var before = cu_mem_get_info()
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        self.ctx.synchronize()
        var after = cu_mem_get_info()
        self.encode_seconds = Float64(perf_counter_ns() - t0) / 1.0e9
        self._record_vram()
        print("[klein_runtime] Qwen3 child encode done; encoder process exited; used",
              before.used_bytes() // (1024 * 1024), "->",
              after.used_bytes() // (1024 * 1024), "MiB after trim")

    def _write_result_manifest(mut self, png_path: String) raises -> String:
        self._record_vram()
        var manifest_path = png_path + String(".klein_daemon_result.json")
        var sample_per_step = Float64(0.0)
        if self.params.steps > 0:
            sample_per_step = self.sample_decode_seconds / Float64(self.params.steps)
        var total_wall_seconds = Float64(perf_counter_ns() - self.job_t0_ns) / 1.0e9
        var peak_mib = Float64(0.0)
        if self.total_vram_bytes > 0 and self.min_free_bytes > 0:
            peak_mib = peak_vram_mib(self.total_vram_bytes, self.min_free_bytes)

        var content = String("{\n")
        content += String('  "schema":"serenity.klein_daemon_result.v1",\n')
        content += String('  "backend":"klein_runtime",\n')
        content += String('  "model":"klein",\n')
        content += String('  "readiness_label":"experimental",\n')
        content += String('  "accepted_sampler_parity":false,\n')
        content += String('  "accepted_speed_parity":false,\n')
        content += String('  "run_identity":{\n')
        content += String('    "job_id":"') + json_escape(self.params.job_id) + String('",\n')
        content += String('    "prompt":"') + json_escape(self.params.prompt) + String('",\n')
        content += String('    "negative":"') + json_escape(self.params.negative) + String('",\n')
        content += String('    "seed":') + String(self.params.seed) + String(",\n")
        content += String('    "resolution":{"width":') + String(self.params.width) + String(',"height":') + String(self.params.height) + String("},\n")
        content += String('    "steps":') + String(self.params.steps) + String(",\n")
        content += String('    "guidance":') + String(self.params.cfg) + String(",\n")
        content += String('    "sampler_registry_backend":"flux2",\n')
        content += String('    "requested_sampler":"') + json_escape(self.params.sampler) + String('",\n')
        content += String('    "requested_scheduler":"') + json_escape(self.params.scheduler) + String('",\n')
        content += String('    "executed_sampler":"klein_euler",\n')
        content += String('    "executed_scheduler":"simple",\n')
        content += String('    "variation_seed":') + String(self.params.variation_seed) + String(",\n")
        content += String('    "variation_strength":') + String(self.params.variation_strength) + String(",\n")
        content += String('    "variation_applied":') + json_bool(self.params.variation_strength > 0.0) + String(",\n")
        content += String('    "image_index":') + String(self.params.image_index) + String(",\n")
        content += String('    "image_count":') + String(self.params.image_count) + String(",\n")
        content += String('    "variant":"') + json_escape(self.variant) + String('",\n')
        content += String('    "config_path":"') + json_escape(self.config_path) + String('",\n')
        content += String('    "lora_count":') + String(len(self.params.loras)) + String(",\n")
        content += String('    "lora_path":"') + json_escape(self.lora_path) + String('",\n')
        content += String('    "lora_weight":') + String(self.lora_multiplier) + String(",\n")
        content += String('    "checkpoint_path":"') + json_escape(self.cfg[0][].checkpoint) + String('",\n')
        content += String('    "dtype":"')
        content += (
            String("serenity_fp8_e4m3_scalar_to_bf16")
            if _lower(self.cfg[0][].checkpoint).find("fp8") >= 0
            else String("bf16_klein_runtime")
        )
        content += String('"\n')
        content += String("  },\n")
        content += String('  "mojo":{\n')
        content += String('    "text_encode_seconds":') + String(self.encode_seconds) + String(",\n")
        content += String('    "sample_decode_seconds":') + String(self.sample_decode_seconds) + String(",\n")
        content += String('    "sample_decode_seconds_per_step":') + String(sample_per_step) + String(",\n")
        content += String('    "total_wall_seconds":') + String(total_wall_seconds) + String(",\n")
        content += String('    "peak_vram_mib":') + String(peak_mib) + String(",\n")
        content += String('    "artifact_paths":["') + json_escape(png_path) + String('","') + json_escape(manifest_path) + String('"]\n')
        content += String("  },\n")
        content += String('  "output_png":"') + json_escape(png_path) + String('",\n')
        content += String('  "note":"Rust-server Mojo Klein runtime product-path result; timing and VRAM are measured in the worker process. Sampler and speed parity remain unaccepted until paired baseline evidence exists."\n')
        content += String("}\n")
        write_text_file(manifest_path, content)
        return manifest_path

    # ── ReferenceLatent edit: PNG -> 512² staged -> Klein VAE latent ──────────
    def _encode_reference_512(mut self, path: String) raises -> Tensor:
        """Port of klein_sample_cli._encode_reference_512: decode -> bilinear
        512² -> signed NCHW f32 -> KleinVaeEncoder.encode. The encoder loads and
        drops inside this call (RAII), before the sampler's base stack loads."""
        _require_file(String("reference image"), path)
        var img = decode_image_any(path)
        var resized = resize_bilinear(img, 512, 512)
        var host = image_to_signed_nchw(resized)
        var image_t = Tensor.from_host(host, [1, 3, 512, 512], STDtype.F32, self.ctx)
        print("[klein_runtime] reference", path, "(", img.width, "x", img.height,
              ") -> 512x512 VAE encode")
        var enc = KleinVaeEncoder[512, 512].load(self.cfg[0][].vae, self.ctx)
        return enc.encode(image_t, self.ctx)

    def _encode_reference_1024(mut self, path: String) raises -> Tensor:
        _require_file(String("reference image"), path)
        var img = decode_image_any(path)
        var resized = resize_bilinear(img, 1024, 1024)
        var host = image_to_signed_nchw(resized)
        var image_t = Tensor.from_host(host, [1, 3, 1024, 1024], STDtype.F32, self.ctx)
        print("[klein_runtime] reference", path, "(", img.width, "x", img.height,
              ") -> 1024x1024 VAE encode")
        var enc = KleinVaeEncoder[1024, 1024].load(self.cfg[0][].vae, self.ctx)
        return enc.encode(image_t, self.ctx)

    def _sample_product_shape[
        LH_: Int, LW_: Int, N_IMG_: Int, S_: Int
    ](
        mut self, pos: Tensor, neg: Tensor, cfg_scale: Float32,
        steps: Int, seed: UInt64,
    ) raises:
        self._ensure_loader()
        var needs_warm = self._denoiser_needs_warm()
        if self.variant == String("9b"):
            var _img = klein_sample_with_loader[
                N_IMG_, N_TXT, S_, LH_, LW_, H_9B, Dh
            ](
                self.cfg[0][], self.lora_path, pos, neg, cfg_scale, steps,
                seed, self.out_png, self.ctx, self.loader[0][],
                progress_fd=self.progress_fd,
                allow_child_decode=True,
                lora_multiplier=self.lora_multiplier,
                warm_denoiser=needs_warm,
            )
        else:
            var _img = klein_sample_with_loader[
                N_IMG_, N_TXT, S_, LH_, LW_, H_4B, Dh
            ](
                self.cfg[0][], self.lora_path, pos, neg, cfg_scale, steps,
                seed, self.out_png, self.ctx, self.loader[0][],
                progress_fd=self.progress_fd,
                allow_child_decode=True,
                lora_multiplier=self.lora_multiplier,
                warm_denoiser=needs_warm,
            )
        self._mark_denoiser_warm()

    def _sample_product(
        mut self, pos: Tensor, neg: Tensor, cfg_scale: Float32,
        steps: Int, seed: UInt64,
    ) raises:
        comptime for bi in range(DEFAULT_ASPECT_LADDER_LEN):
            comptime X100_BI = DEFAULT_ASPECT_LADDER_X100[bi]
            comptime LH_BI = aspect_lat_h_units(X100_BI, KLEIN_PRODUCT_EDGE_UNITS) // 2
            comptime LW_BI = aspect_lat_w_units(X100_BI, KLEIN_PRODUCT_EDGE_UNITS) // 2
            comptime N_IMG_BI = LH_BI * LW_BI
            comptime S_BI = N_IMG_BI + N_TXT
            if self.params.width == LW_BI * 16 and self.params.height == LH_BI * 16:
                self._sample_product_shape[LH_BI, LW_BI, N_IMG_BI, S_BI](
                    pos, neg, cfg_scale, steps, seed,
                )
                return
        raise Error("klein_runtime: admitted product shape was not compiled")

    # ── denoise + VAE decode + save (one long blocking tick) ──────────────────
    def _sample_and_save(mut self) raises -> String:
        var t0 = perf_counter_ns()
        self._record_vram()
        var cfg_scale = Float32(self.params.cfg)
        var seed = UInt64(self.params.seed)
        var steps = self.params.steps
        var pos = self.pos_txt[0][].clone(self.ctx)
        var neg = self.neg_txt[0][].clone(self.ctx)

        # ── ReferenceLatent edit dispatch. ──
        if self.params.reference_latent_count == 1:
            var ref_path = self.params.reference_image.copy()
            if self.params.width == 1024 and self.params.height == 1024:
                var ref_lat = self._encode_reference_1024(ref_path)
                if self.variant == String("9b"):
                    var _e = klein_sample_with_reference_latent[
                        N_IMG_1024, N_EDIT_IMG_1024, N_TXT, S_EDIT_1024, LH_1024, LW_1024, H_9B, Dh
                    ](
                        self.cfg[0][], self.lora_path, pos, neg, cfg_scale, steps,
                        seed, ref_lat^, self.out_png, self.ctx,
                        lora_multiplier=self.lora_multiplier,
                    )
                else:
                    var _e = klein_sample_with_reference_latent[
                        N_IMG_1024, N_EDIT_IMG_1024, N_TXT, S_EDIT_1024, LH_1024, LW_1024, H_4B, Dh
                    ](
                        self.cfg[0][], self.lora_path, pos, neg, cfg_scale, steps,
                        seed, ref_lat^, self.out_png, self.ctx,
                        lora_multiplier=self.lora_multiplier,
                    )
            else:
                var ref_lat = self._encode_reference_512(ref_path)
                if self.variant == String("9b"):
                    var _e = klein_sample_with_reference_latent[
                        N_IMG_512, N_EDIT_IMG_512, N_TXT, S_EDIT_512, LH_512, LW_512, H_9B, Dh
                    ](
                        self.cfg[0][], self.lora_path, pos, neg, cfg_scale, steps,
                        seed, ref_lat^, self.out_png, self.ctx,
                        lora_multiplier=self.lora_multiplier,
                    )
                else:
                    var _e = klein_sample_with_reference_latent[
                        N_IMG_512, N_EDIT_IMG_512, N_TXT, S_EDIT_512, LH_512, LW_512, H_4B, Dh
                    ](
                        self.cfg[0][], self.lora_path, pos, neg, cfg_scale, steps,
                        seed, ref_lat^, self.out_png, self.ctx,
                        lora_multiplier=self.lora_multiplier,
                    )
        elif self.params.reference_latent_count == 2:
            # Lowering contract (serenityflow__klein9b_edit golden): ref A rides
            # init_image, ref B rides reference_image.
            var ref_a_path = self.params.init_image.copy()
            var ref_b_path = self.params.reference_image.copy()
            var ref_a = self._encode_reference_512(ref_a_path)
            var ref_b = self._encode_reference_512(ref_b_path)
            if self.variant == String("9b"):
                var _e = klein_sample_with_reference_latents2[
                    N_IMG_512, N_EDIT2_IMG_512, N_TXT, S_EDIT2_512, LH_512, LW_512, H_9B, Dh
                ](
                    self.cfg[0][], self.lora_path, pos, neg, cfg_scale, steps,
                    seed, ref_a^, ref_b^, self.out_png, self.ctx,
                    lora_multiplier=self.lora_multiplier,
                )
            else:
                var _e = klein_sample_with_reference_latents2[
                    N_IMG_512, N_EDIT2_IMG_512, N_TXT, S_EDIT2_512, LH_512, LW_512, H_4B, Dh
                ](
                    self.cfg[0][], self.lora_path, pos, neg, cfg_scale, steps,
                    seed, ref_a^, ref_b^, self.out_png, self.ctx,
                    lora_multiplier=self.lora_multiplier,
                )
        elif self.params.width == 512 and self.params.height == 512:
            self._ensure_loader()
            var needs_warm = self._denoiser_needs_warm()
            if self.variant == String("9b"):
                var _img = klein_sample_with_loader[
                    N_IMG_512, N_TXT, S_512, LH_512, LW_512, H_9B, Dh
                ](
                    self.cfg[0][], self.lora_path, pos, neg, cfg_scale, steps,
                    seed, self.out_png, self.ctx, self.loader[0][],
                    progress_fd=self.progress_fd,
                    allow_child_decode=True,
                    lora_multiplier=self.lora_multiplier,
                    warm_denoiser=needs_warm,
                )
            else:
                var _img = klein_sample_with_loader[
                    N_IMG_512, N_TXT, S_512, LH_512, LW_512, H_4B, Dh
                ](
                    self.cfg[0][], self.lora_path, pos, neg, cfg_scale, steps,
                    seed, self.out_png, self.ctx, self.loader[0][],
                    progress_fd=self.progress_fd,
                    allow_child_decode=True,
                    lora_multiplier=self.lora_multiplier,
                    warm_denoiser=needs_warm,
                )
            self._mark_denoiser_warm()
        else:
            self._sample_product(pos, neg, cfg_scale, steps, seed)

        if not _path_exists(self.out_png):
            raise Error(String("klein_runtime: sampler did not produce ") + self.out_png)
        _embed_genparams_in_png(self.out_png, self.params.params_json)
        self.sample_decode_seconds = Float64(perf_counter_ns() - t0) / 1.0e9
        self._record_vram()
        var out_path = self.out_png.copy()
        var manifest = self._write_result_manifest(out_path)
        print("[klein_runtime][manifest] saved:", manifest)
        return self.out_png.copy()

    # ── the pull-based tick ───────────────────────────────────────────────────
    def step(mut self) raises -> StepResult:
        var r = StepResult()
        r.total = self.params.steps
        if not self.active:
            r.failed = True
            r.error = String("klein_runtime: no active job")
            return r^
        if self.cancel_flag:
            self._clear_job()
            r.cancelled = True
            return r^
        try:
            if self.phase == KRPHASE_ENCODE:
                if not self.announced:
                    # announce BEFORE the long blocking Qwen3 encode tick.
                    self.announced = True
                    r.step = 0
                    r.phase = String("encoding")
                    return r^
                self._encode()
                self.announced = False
                self.phase = KRPHASE_SAMPLE
                r.step = 0
                return r^
            if self.phase == KRPHASE_SAMPLE:
                if not self.announced:
                    # Loader construction and kernel activation occur inside the
                    # next blocking tick. Keep the UI in loading until the sampler
                    # emits its exact post-warm-up sampling step-0 IPC event.
                    self.announced = True
                    r.step = 0
                    r.phase = String("loading")
                    return r^
                var path = self._sample_and_save()
                r.step = self.params.steps
                self._clear_job()
                r.done = True
                r.output_path = path^
                return r^
            raise Error("klein_runtime: invalid backend phase")
        except e:
            self._clear_job()
            r.failed = True
            r.error = String(e)
            return r^
