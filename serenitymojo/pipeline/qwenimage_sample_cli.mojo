# serenitymojo/pipeline/qwenimage_sample_cli.mojo
#
# UI-driven CLI adapter for Qwen-Image text→image generation.
# This is the proof-of-pattern adapter; other model adapters (chroma, sd3,
# sdxl, ernie, flux) should copy this file and replace the encode/denoise/vae
# calls with their own model equivalents.
#
# Contract (the UI bridge calls it exactly this way):
#
#   qwenimage_sample_cli  <config.json>  <lora|->  <sample_prompts.json>  <prompt_id>  <out.png>
#
#   argv[1]  config JSON path (model dirs are comptime constants in the runner;
#            this argument is ACCEPTED BUT IGNORED TODAY — document override
#            instructions in the config file).
#
#   argv[2]  LoRA safetensors path, or "-"/"base"/"" for base model.
#            Qwen-Image has no LoRA path today; the value is ACCEPTED AND
#            IGNORED.  When Qwen-Image LoRA lands, wire it into encode_captions.
#
#   argv[3]  sample_prompts JSON (serenity.sample_prompts.v1 schema).
#            Read with `read_sample_prompt_config`.
#
#   argv[4]  Prompt id/label to select from the JSON, or "" for the first entry.
#
#   argv[5]  Output PNG path.  Written via save_png(…, ValueRange.SIGNED).
#
# ──────────────────────────────────────────────────────────────────────────────
# Request fields honored vs fixed:
#
#   HONORED at runtime:
#     • prompt    — threaded through _encode_trimmed (runtime String, not comptime)
#     • negative  — threaded through _encode_trimmed (runtime String, not comptime)
#     • steps     — scheduler sigma table is built per request
#     • cfg       — passed into CFG combine per step
#     • seed      — passed into initial latent noise generation
#
#   FIXED at comptime (from qwenimage_pipeline_1024_multistep.mojo):
#     • width  = LW * 8  (1024, latent LW=128)
#     • height = LH * 8  (1024, latent LH=128)
#
#   The Qwen-Image DiT attention shape is a comptime constant (N_IMG, N_TXT_KEPT,
#   S_POS, S_NEG), so resolution changes require a recompile.  Non-1024 sample
#   requests fail loudly instead of silently running the wrong size.
#
# ──────────────────────────────────────────────────────────────────────────────
# Generate path: REAL, not a stub.
#   Calls encode_captions_from_strings → denoise → unpatchify → tiled Qwen VAE decode.
#   The only change vs the standalone runner is that the prompt/negative pair
#   comes from the sample_prompts JSON rather than the PROMPT/NEGATIVE comptime.
#
# Build:
#   cd /home/alex/mojodiffusion && pixi run mojo build -I . \
#     -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/pipeline/qwenimage_sample_cli.mojo \
#     -o /tmp/qwenimage_sample_cli
#
# Note for porter agents (chroma/sd3/sdxl/ernie/flux):
#   1. Replace the import block (text encoder, DiT, VAE, ops) with your model's.
#   2. Replace encode_captions_from_strings / denoise / vae decode with your model's.
#   3. Keep the argv contract identical — the UI bridge is the same for all adapters.
#   4. Keep _select_prompt / _load_prompt_json unchanged — they are shared infra.

from std.sys import argv
from std.gpu.host import DeviceContext
from std.memory import alloc, UnsafePointer
from std.builtin.type_aliases import MutExternalOrigin
from std.ffi import external_call
from std.time import sleep

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.cap_cache import save_tensor_bin, load_tensor_bin
from serenitymojo.io.ffi import (
    BytePtr, sys_open, sys_pwrite, sys_pread, sys_close,
    O_WRONLY, O_CREAT, O_TRUNC, O_RDONLY,
)
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.qwen25vl_encoder import (
    Qwen25VLEncoder,
    Qwen25VLConfig,
)
from serenitymojo.models.dit.qwenimage_dit import (
    QwenImageDitOffloaded,
    qwenimage_resident_pin_budget,
)
from serenitymojo.models.vae.qwenimage_tiled_decode import qwenimage_tiled_decode
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current, cu_mem_get_info
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.layout import patchify, unpatchify
from serenitymojo.ops.tensor_algebra import slice
from serenitymojo.sampling.flow_match import Scheduler, cfg_qwen_device
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.training.sample_prompt_config import (
    SamplePrompt, SamplePromptConfig, read_sample_prompt_config,
)
from serenitymojo.serve.proc_ipc import (
    build_argv, cstr, sys_execv, sys__exit, sys_waitpid, proc_kill_wait,
    SELF_EXE, SIGKILL, WNOHANG,
)
from net.syscalls import sys_fork, errno_str

# ── Model paths (comptime; override by editing this file or adding config support) ──
comptime QWENIMAGE_DIR = "/home/alex/.serenity/models/checkpoints/qwen-image-2512"
comptime TEXT_ENCODER_DIR = QWENIMAGE_DIR + "/text_encoder"
comptime TOK_JSON = QWENIMAGE_DIR + "/tokenizer/tokenizer.json"
comptime DIT_DIR = QWENIMAGE_DIR + "/transformer"
comptime VAE_DIR = QWENIMAGE_DIR + "/vae"

# ── Tokenizer / encoder constants (verbatim from the runner) ──
comptime PAD_ID = 151643
comptime DROP_IDX = 34
comptime N_TXT_KEPT = 512
comptime N_ENC = N_TXT_KEPT + DROP_IDX   # 546
comptime EXTRACT_LAYER = 27
comptime _ENCODE_CHILD_TIMEOUT_S = 300.0
comptime _ENCODE_POLL_S = 0.05
comptime _ENCODE_CHILD_MIN_FREE_BYTES = Int(17400) * 1024 * 1024
comptime _META_MAGIC = Int64(0x51494D4341505631)  # "QIMCAPV1"

# ── Latent / DiT shape constants (comptime-fixed; see header for rationale) ──
comptime LH = 128
comptime LW = 128
comptime PATCH = 2
comptime N_IMG = (LH // PATCH) * (LW // PATCH)
comptime S_POS = N_IMG + N_TXT_KEPT
comptime S_NEG = N_IMG + N_TXT_KEPT
comptime FRAME = 1
comptime FH = LH // PATCH
comptime FW = LW // PATCH

# ── Caption pair produced by the text encoder ──
@fieldwise_init
struct QwenCaps(Movable):
    var pos: Tensor
    var neg: Tensor
    var real_pos: Int
    var real_neg: Int


@fieldwise_init
struct EncodedCaption(Movable):
    var hidden: Tensor
    var real_len: Int

    def into_caps(deinit self, deinit neg: EncodedCaption) -> QwenCaps:
        return QwenCaps(self.hidden^, neg.hidden^, self.real_len, neg.real_len)


def _getpid() -> Int:
    return Int(external_call["getpid", Int32]())


def _write_meta(path: String, real_pos: Int, real_neg: Int) raises:
    var fd = sys_open(path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
    if fd < 0:
        raise Error(String("qwenimage_sample_cli: meta open failed: ") + path)
    var tmp = alloc[Int64](3)
    tmp[0] = _META_MAGIC
    tmp[1] = Int64(real_pos)
    tmp[2] = Int64(real_neg)
    var p = BytePtr(unsafe_from_address=Int(tmp))
    var w = sys_pwrite(fd, p, 24, 0)
    tmp.free()
    _ = sys_close(fd)
    if w != 24:
        raise Error("qwenimage_sample_cli: short meta write")


def _read_meta(path: String) raises -> List[Int]:
    var fd = sys_open(path, O_RDONLY, 0)
    if fd < 0:
        raise Error(String("qwenimage_sample_cli: meta open failed: ") + path)
    var tmp = alloc[Int64](3)
    var p = BytePtr(unsafe_from_address=Int(tmp))
    var r = sys_pread(fd, p, 24, 0)
    var magic = tmp[0]
    var rp = Int(tmp[1])
    var rn = Int(tmp[2])
    tmp.free()
    _ = sys_close(fd)
    if r != 24:
        raise Error("qwenimage_sample_cli: short meta read")
    if magic != _META_MAGIC:
        raise Error("qwenimage_sample_cli: bad meta magic")
    var out = List[Int]()
    out.append(rp)
    out.append(rn)
    return out^


# ── Qwen chat template ──
def _qwen_template(prompt: String) -> String:
    return (
        String("<|im_start|>system\nDescribe the image by detailing the color,"
        " shape, size, texture, quantity, text, spatial relationships of the"
        " objects and background:<|im_end|>\n<|im_start|>user\n")
        + prompt
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


# Tokenize and pad to N_ENC; return (ids, real_kept_len).
def _tokenize_for_encoder(
    tok: Qwen3Tokenizer, prompt: String
) raises -> Tuple[List[Int], Int]:
    var ids_full = tok.encode(_qwen_template(prompt))
    var real_len = len(ids_full)
    if real_len <= DROP_IDX:
        raise Error(
            String("qwenimage_sample_cli: prompt tokenized to ")
            + String(real_len)
            + " tokens, not enough past DROP_IDX="
            + String(DROP_IDX)
        )
    if real_len > N_ENC:
        raise Error(
            String("qwenimage_sample_cli: prompt tokenized to ")
            + String(real_len)
            + " tokens, exceeding N_ENC="
            + String(N_ENC)
            + " (N_TXT_KEPT="
            + String(N_TXT_KEPT)
            + "); shorten the prompt or raise N_TXT_KEPT"
        )
    var real_kept_len = real_len - DROP_IDX
    var ids = List[Int](capacity=N_ENC)
    for i in range(real_len):
        ids.append(ids_full[i])
    for _ in range(N_ENC - real_len):
        ids.append(PAD_ID)
    print("  tokens:", real_len, "-> drop", DROP_IDX, "-> kept", real_kept_len)
    return (ids^, real_kept_len)


# Encode a single runtime prompt string → [1, N_TXT_KEPT, 3584] trimmed hidden.
def _encode_trimmed(
    enc: Qwen25VLEncoder, tok: Qwen3Tokenizer, prompt: String, ctx: DeviceContext
) raises -> EncodedCaption:
    var tup = _tokenize_for_encoder(tok, prompt)
    var ids = tup[0].copy()
    var real_kept_len = tup[1]
    var pre = enc.encode(ids, EXTRACT_LAYER, ctx)
    var full = enc.final_norm(pre, ctx)
    var hidden = slice(full, 1, DROP_IDX, N_TXT_KEPT, ctx)
    return EncodedCaption(hidden^, real_kept_len)


# Encode runtime prompt + negative into a QwenCaps pair.
# This is the key difference from the standalone runner: prompt and negative
# are runtime String arguments, not the PROMPT/NEGATIVE comptime constants.
def encode_captions_from_strings(
    prompt: String, negative: String, ctx: DeviceContext
) raises -> QwenCaps:
    print("[text] Qwen2.5-VL encoder, N_TXT_KEPT=", N_TXT_KEPT, "DROP_IDX=", DROP_IDX)
    var tok = Qwen3Tokenizer(TOK_JSON)
    var enc = Qwen25VLEncoder.load(
        TEXT_ENCODER_DIR, Qwen25VLConfig.qwen_image(), ctx
    )
    var pos = _encode_trimmed(enc, tok, prompt, ctx)
    var neg = _encode_trimmed(enc, tok, negative, ctx)
    return pos^.into_caps(neg^)


def encode_child_run(prefix: String, prompt: String, negative: String) raises:
    """Child process body for CLI caption encoding.

    The parent must not keep the Qwen2.5-VL encoder in its CUDA pool before DiT
    denoise. This child loads the encoder, writes BF16 caps, and exits so the OS
    releases the encoder VRAM before the parent touches the transformer.
    """
    var ctx = DeviceContext()
    var caps = encode_captions_from_strings(prompt, negative, ctx)
    save_tensor_bin(caps.pos, prefix + String(".pos.bin"), ctx)
    save_tensor_bin(caps.neg, prefix + String(".neg.bin"), ctx)
    _write_meta(prefix + String(".meta"), caps.real_pos, caps.real_neg)
    print(
        "[qwenimage-sample-encode-child] wrote caps", prefix,
        "real_pos=", caps.real_pos, "real_neg=", caps.real_neg,
    )


def encode_captions_subprocess_cli(
    prompt: String, negative: String, ctx: DeviceContext
) raises -> QwenCaps:
    """Run Qwen2.5-VL in a child so the CLI parent starts DiT with free VRAM."""
    var free_bytes = cu_mem_get_info().free_bytes
    if free_bytes < _ENCODE_CHILD_MIN_FREE_BYTES:
        raise Error(
            String("qwenimage_sample_cli: encoder child preflight failed: free VRAM ")
            + String(free_bytes // (1024 * 1024))
            + String(" MiB < required ")
            + String(_ENCODE_CHILD_MIN_FREE_BYTES // (1024 * 1024))
            + String(" MiB")
        )

    var prefix = String("/tmp/qwenimage_sample_cli_caps_") + String(_getpid())
    var pos_path = prefix + String(".pos.bin")
    var neg_path = prefix + String(".neg.bin")
    var meta_path = prefix + String(".meta")

    var args = List[String]()
    args.append(SELF_EXE)
    args.append(String("encode-child"))
    args.append(prefix)
    args.append(prompt)
    args.append(negative)
    var argv_child = build_argv(args)
    var path = cstr(SELF_EXE)

    print("[text] fork encoder child for Qwen2.5-VL caps")
    var pid = sys_fork()
    if pid == 0:
        _ = sys_execv(path, argv_child)
        sys__exit(127)
    if pid < 0:
        raise Error(String("qwenimage_sample_cli: encoder child fork failed: ") + errno_str())

    var st = alloc[Int32](1)
    var stp = rebind[UnsafePointer[Int32, MutExternalOrigin]](st)
    var waited = 0.0
    var reaped = Int32(0)
    while waited < _ENCODE_CHILD_TIMEOUT_S:
        reaped = sys_waitpid(pid, stp, WNOHANG)
        if reaped == pid:
            break
        if reaped < 0:
            break
        sleep(_ENCODE_POLL_S)
        waited += _ENCODE_POLL_S
    var status = Int(st[0])
    st.free()

    if reaped != pid:
        proc_kill_wait(pid, SIGKILL)
        raise Error("qwenimage_sample_cli: encoder child timed out or waitpid failed")
    var exited_ok = (status & 0x7F) == 0 and ((status >> 8) & 0xFF) == 0
    if not exited_ok:
        raise Error(
            String("qwenimage_sample_cli: encoder child abnormal exit status ")
            + String(status)
        )

    var meta = _read_meta(meta_path)
    var pos = load_tensor_bin(pos_path, ctx)
    var neg = load_tensor_bin(neg_path, ctx)
    print("[text] encoder child reaped; parent loaded BF16 caps only")
    return QwenCaps(pos^, neg^, meta[0], meta[1])


def _trim_driver_pool(ctx: DeviceContext) raises:
    var before = cu_mem_get_info()
    ctx.synchronize()
    cu_mempool_trim_current(0)
    ctx.synchronize()
    var after = cu_mem_get_info()
    print(
        "[vram] after text encode trim: used",
        before.used_bytes() // (1024 * 1024), "->",
        after.used_bytes() // (1024 * 1024), "MiB",
        "free", after.free_bytes // (1024 * 1024), "MiB",
    )


# Build initial latent packed tensor.
def initial_latent_packed(ctx: DeviceContext, seed: UInt64) raises -> Tensor:
    var nchw_shape = List[Int]()
    nchw_shape.append(1)
    nchw_shape.append(16)
    nchw_shape.append(LH)
    nchw_shape.append(LW)
    var noise = randn(nchw_shape^, seed, STDtype.BF16, ctx)
    return patchify(noise, PATCH, ctx)


# CFG denoise loop. Shape remains comptime-fixed; schedule/cfg/seed are runtime.
def denoise(
    caps: QwenCaps, steps: Int, cfg: Float32, seed: UInt64, ctx: DeviceContext
) raises -> Tensor:
    print("[denoise] loading Qwen-Image MMDiT (block-streamed)")
    var model = QwenImageDitOffloaded.load(DIT_DIR, ctx)
    var free_info = cu_mem_get_info()
    var resident_budget = qwenimage_resident_pin_budget(free_info.free_bytes)
    var pinned_blocks = model.pin_resident_blocks(resident_budget, ctx)
    print(
        "[denoise] resident Qwen blocks pinned:", pinned_blocks,
        "budget_bytes=", resident_budget,
        "free_before_pin_mib=", free_info.free_bytes // (1024 * 1024),
    )
    var sched = Scheduler.qwen(steps, Float32(N_IMG))
    var sigmas = sched.sigmas()
    print("[denoise]", steps, "steps, CFG", cfg, "seed", seed)
    print("  real_pos=", caps.real_pos, "real_neg=", caps.real_neg)
    var x = initial_latent_packed(ctx, seed)
    for i in range(steps):
        print("  step", i + 1, "/", steps, "sigma", sigmas[i], "->", sigmas[i + 1])
        var preds = model.forward_cfg_mixed_text[
            N_IMG, N_TXT_KEPT, S_POS, N_TXT_KEPT, S_NEG
        ](
            x, caps.pos, caps.neg, sigmas[i],
            caps.real_pos, caps.real_neg,
            FRAME, FH, FW, ctx,
        )
        var pred = cfg_qwen_device(preds.pos, preds.neg, cfg, ctx)
        x = sched.step(x, pred, i, ctx)
    return x^


# ── Prompt selection helpers (verbatim pattern from zimage_generate.mojo) ──

def _select_prompt(sample_cfg: SamplePromptConfig, wanted: String) raises -> SamplePrompt:
    if len(sample_cfg.prompts) == 0:
        raise Error("qwenimage_sample_cli: sample prompt JSON has no prompts")
    if wanted == String(""):
        for i in range(len(sample_cfg.prompts)):
            if sample_cfg.prompts[i].enabled:
                return sample_cfg.prompts[i].copy()
        raise Error("qwenimage_sample_cli: sample prompt JSON has no enabled prompts")
    for i in range(len(sample_cfg.prompts)):
        if sample_cfg.prompts[i].label == wanted:
            if not sample_cfg.prompts[i].enabled:
                raise Error(String("qwenimage_sample_cli: prompt is disabled: ") + wanted)
            return sample_cfg.prompts[i].copy()
    raise Error(String("qwenimage_sample_cli: prompt id not found: ") + wanted)


def _load_prompt_json(path: String, wanted: String) raises -> SamplePrompt:
    var sample_cfg = read_sample_prompt_config(path)
    var p = _select_prompt(sample_cfg, wanted)
    if p.frames != 1:
        raise Error("qwenimage_sample_cli: only image prompts (frames=1) are supported")
    if p.width != LW * 8 or p.height != LH * 8:
        raise Error(
            String("qwenimage_sample_cli: this runner is compiled for 1024x1024, got ")
            + String(p.width) + String("x") + String(p.height)
        )
    if p.steps <= 0:
        raise Error("qwenimage_sample_cli: steps must be > 0")
    if p.random_seed:
        raise Error("qwenimage_sample_cli: random_seed is not supported; provide a fixed seed")
    if p.noise_scheduler.byte_length() > 0:
        raise Error("qwenimage_sample_cli: noise_scheduler override is not supported")
    if p.sample_inpainting:
        raise Error("qwenimage_sample_cli: sample_inpainting is not supported")
    if p.base_image_path.byte_length() > 0 or p.mask_image_path.byte_length() > 0:
        raise Error("qwenimage_sample_cli: base/mask image prompts are not supported")
    if p.caps_pos.byte_length() > 0 or p.caps_neg.byte_length() > 0:
        raise Error("qwenimage_sample_cli: precomputed caps are not supported by the live Qwen text encoder path")
    print(
        "  [info] sample prompt honored: steps=", p.steps, "cfg=", p.cfg,
        "seed=", p.seed, "size=", p.width, "x", p.height,
    )
    return p^


# ── Main entry ──────────────────────────────────────────────────────────────

def main() raises:
    var a = argv()
    if len(a) == 5 and String(a[1]) == String("encode-child"):
        encode_child_run(String(a[2]), String(a[3]), String(a[4]))
        return
    if len(a) != 6:
        print(
            "usage: qwenimage_sample_cli <config.json> <lora|-> <sample_prompts.json>"
            " <prompt_id> <out.png>"
        )
        print("  argv[1] config   — must be '-' or /dev/null; model dirs are comptime")
        print("  argv[2] lora     — must be '-' or base; no LoRA support yet")
        print("  argv[3] prompts  — serenity.sample_prompts.v1 JSON")
        print("  argv[4] id       — prompt label, or '' for first")
        print("  argv[5] out.png  — output image path")
        raise Error("qwenimage_sample_cli: need exactly 5 arguments")

    # argv[1]: config — no runtime model-dir override is wired yet.
    var config_path = String(a[1])
    if (
        config_path != String("")
        and config_path != String("-")
        and config_path != String("/dev/null")
    ):
        raise Error(
            "qwenimage_sample_cli: config path overrides are not supported yet; pass '-'"
        )

    # argv[2]: lora path or sentinel; Qwen-Image LoRA is not wired yet.
    var lora_raw = String(a[2])
    if lora_raw != String("-") and lora_raw != String("base") and lora_raw != String(""):
        raise Error(
            "qwenimage_sample_cli: LoRA is not supported for Qwen-Image yet"
        )

    # argv[3]: sample prompts JSON
    var prompts_json = String(a[3])

    # argv[4]: prompt id
    var prompt_id = String(a[4])

    # argv[5]: output PNG
    var out_png = String(a[5])

    # Load runtime sample request from the JSON.
    var req_prompt = _load_prompt_json(prompts_json, prompt_id)
    var prompt = req_prompt.prompt.copy()
    var negative = req_prompt.negative.copy()

    print("=== Qwen-Image sample CLI ===")
    print("  prompts:", prompts_json, " id:", prompt_id)
    print("  output:", out_png)
    print("  [prompt]", prompt)
    if negative != String(""):
        print("  [negative]", negative)

    var ctx = DeviceContext()

    # Encode runtime prompt + negative in a child process. Process exit is the
    # reliable way to return Qwen2.5-VL encoder VRAM before DiT denoise.
    var caps = encode_captions_subprocess_cli(prompt, negative, ctx)
    _trim_driver_pool(ctx)

    # Denoise (shape is comptime-fixed; steps/cfg/seed come from sample JSON).
    var tokens = denoise(caps, req_prompt.steps, req_prompt.cfg, req_prompt.seed, ctx)

    # VAE decode.
    print("[vae] unpack + tiled decode")
    var latent = unpatchify(tokens, 16, LH, LW, PATCH, ctx)
    latent = cast_tensor(latent, STDtype.BF16, ctx)
    var img = qwenimage_tiled_decode[LH, LW](latent, VAE_DIR, ctx)
    var sh = img.shape()
    print("  image shape:", sh[0], sh[1], sh[2], sh[3])

    # Save.
    save_png(img, out_png, ctx, ValueRange.SIGNED)
    print("[done] saved:", out_png)
