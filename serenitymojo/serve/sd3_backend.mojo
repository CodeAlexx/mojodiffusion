# serenitymojo.serve.sd3_backend — the real SD3.5 Large 1024x1024 GenBackend.
#
# Wraps the VERIFIED SD3.5 Large inference stages behind the pull-based GenBackend
# seam (backend.mojo). Unlike sd3_sample_cli.mojo — which loads a PRE-CACHED
# triple-encoder sidecar (context_cond/context_uncond/pooled_cond/pooled_uncond) —
# THIS backend encodes the REAL params.prompt + params.negative at runtime through
# the verified CLIP-L + CLIP-G + T5-XXL modules, assembling SD3's context+pooled
# EXACTLY as inference-flame's sd3_infer.rs (the reference that produced the
# verified sidecars the Mojo SD3 MMDiT was anchored against):
#
#   # CLIP-L (CLIPTextModelWithProjection, but the SD3 CLIP-L safetensors ships NO
#   #         text_projection → raw EOS pool):
#   clip_l_hidden  = clip_l.encode_sd3(ids).hidden_states[-2]   [1,77,768]
#   clip_l_pooled  = pooler_output (post-final-LN @ EOS)          [1,768]
#   # CLIP-G (text_projection.weight [1280,1280] present → projected pool):
#   clip_g_hidden  = clip_g.encode_sd3(ids).hidden_states[-2]   [1,77,1280]
#   clip_g_pooled  = text_projection(pooler_output)              [1,1280]
#   # T5-XXL v1.1 (256 tokens, narrowed):
#   t5_hidden      = t5.encode(ids)                              [1,256,4096]
#
#   context = cat([ pad(clip_l_hidden, 0..4096),
#                   pad(clip_g_hidden, 0..4096),
#                   t5_hidden ], dim=1)                          [1,410,4096]
#   pooled  = cat([ clip_l_pooled, clip_g_pooled ], dim=1)       [1,2048]
#   (same for the negative prompt -> context_uncond / pooled_uncond.)
#
# THREE numeric details replicated verbatim from sd3_infer.rs (measured load-bearing):
#   1. CLIP uses hidden_states[-2] — the encoder output after layer (num_layers-2),
#      NOT the post-final-LN last_hidden_state. (encode_sdxl returns the WRONG tensor
#      for SD3; this backend reimplements the SD3 CLIP forward over the verified
#      ClipEncoder building blocks: _embed -> _layer[77]×L capturing penultimate.)
#   2. CLIP pooled = final_layer_norm(last layer) @ EOS, then text_projection IF the
#      file ships it (CLIP-G yes, CLIP-L no). Without CLIP-G's projection the pooled
#      has the wrong scale and SD3 Large blows up in block 0 (uniform output).
#   3. CLIP attention uses a PURE CAUSAL mask (no key-padding) — the reference
#      build_causal_mask masks only j<=i, so padded positions (after EOS) attend to
#      all earlier real tokens. We get pure-causal from _build_pad_mask(S,heads,S-1)
#      (valid_key_end=S-1 makes the j<=valid_key_end clause always true).
#   4. T5 ids: the Mojo T5Tokenizer.encode appends ONE EOS=1; sd3_infer.rs ALSO does
#      ids.push(1) after HF tokenization → a SECOND trailing EOS=1 before pad-to-256
#      (pad id 0). We replicate the second push to match the reference token stream.
#
# The denoise (28-step shifted rectified-flow CFG Euler) and VAE decode reuse
# sd3_sample_cli's exact math (_sd3_large_forward streams 38 joint blocks via
# BlockLoader; sd3_cfg/sd3_euler_step; load_sd3_embedded_ldm_decoder).
#
# Residency model (single-GPU, 24 GB):
#   * The SD3.5 Large MMDiT is BLOCK-STREAMED from disk (38 joint blocks, ~426 MB
#     each, streamed one at a time inside the forward). The RESIDENT win is the
#     small pre/post-block gate (SD3MMDiTPreBlockGate: patch/pos/context/cond/final
#     weights) PLUS the BlockLoader handle (mmap'd checkpoint) — both loaded ONCE
#     (first job) and kept across jobs. The 38 block weights are never all-resident
#     (they cannot fit); the per-forward peak is one block + activations.
#   * The CLIP-L (~250 MB) + CLIP-G (~1.4 GB) + T5-XXL (~9.5 GB BF16) encoders are
#     loaded -> used -> freed PER JOB inside the ENCODE step (Movable-not-Copyable
#     Tensors drop at scope exit; the encode is staged so they never co-reside with
#     the resident gate at peak beyond the encoder footprint).
#   * The VAE decoder (~330 MB) is loaded PER JOB inside the DECODE step and freed.
#     SD3 keeps no large resident denoiser at decode (blocks are streamed, freed),
#     so the resident footprint at decode is just the small gate -> the monolithic
#     SD3 VAE decode (verified in sd3_sample_cli) fits. We still trim the mempool
#     before decode defensively.
#
# step() state machine: ENCODE (per-job, blocking — announced phase="encoding")
#   → LOAD (gate + BlockLoader, once, announced phase="loading") → DENOISE×steps
#   (two streamed forwards + Euler update per tick) → DECODE (announced
#   phase="decoding") → done. cancel() makes the next step() return cancelled and
#   frees all per-job tensors.
#
# Size support: 1024x1024 ONLY (the SD3 Large DiT attention shape S_JOINT and the
# pre-block gate are comptime-fixed at LH=LW=128). steps/cfg/seed ARE honored at
# runtime (the denoise loop reads them from JobParams; the flow-match sigma table
# is built per job for params.steps with the large shift).
#
# LoRA / img2img: NOT supported yet — rejected at admission so they never silently
# no-op (matches sd3_sample_cli's "accepted-and-ignored" caveat, made fail-loud).

from std.collections import Optional
from std.ffi import external_call
from std.gpu.host import DeviceContext
from std.memory import alloc, ArcPointer

from image.buffer import Image
from image.png import encode_png_with_text

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import BytePtr
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.image.png import _quantize, ValueRange
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current, cu_mem_get_info
from serenitymojo.offload.block_loader import BlockLoader, unload_block

from serenitymojo.tokenizer.clip_tokenizer import ClipTokenizer
from serenitymojo.tokenizer.t5_tokenizer import T5Tokenizer
from serenitymojo.models.text_encoder.clip_encoder import (
    ClipEncoder, ClipConfig, _build_pad_mask, slice_seq,
)
from serenitymojo.models.text_encoder.t5_encoder import T5Encoder, T5Config
from serenitymojo.models.dit.sd3_contract import (
    SD3_LARGE_DEPTH, SD3_LARGE_HEAD_DIM, SD3_LARGE_HIDDEN,
    SD3_LARGE_IMAGE_TOKENS, SD3_LARGE_LATENT_CHANNELS, SD3_LARGE_LATENT_H,
    SD3_LARGE_LATENT_W, SD3_LARGE_NUM_HEADS, SD3_LARGE_TEXT_TOKENS,
    sd3_large_schedule_shift,
)
from serenitymojo.models.dit.sd3_mmdit import (
    SD3MMDiTPreBlockGate, _sd3_joint_block,
)
from serenitymojo.models.vae.ldm_decoder import load_sd3_embedded_ldm_decoder
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm as _ops_layer_norm
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import reshape, concat
from serenitymojo.sampling.sd3_flow_match import (
    SD3FlowMatchScheduler, sd3_cfg, sd3_euler_step,
)
from serenitymojo.sampling.variation_noise import swarm_variation_noise_chw
from serenitymojo.sampling.sampler_registry import (
    sampler_admission_for_backend, scheduler_admission_for_backend,
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

# ── shape constants (1024x1024, matching sd3_sample_cli + the SD3 Large contract) ──
comptime WIDTH = 1024
comptime HEIGHT = 1024
comptime LH = SD3_LARGE_LATENT_H            # 128
comptime LW = SD3_LARGE_LATENT_W            # 128
comptime LC = SD3_LARGE_LATENT_CHANNELS     # 16

comptime CLIP_LEN = 77
comptime T5_LEN = 256

# SD3 Large joint sequence: image + text tokens (text = 77 + 77 + 256 = 410).
comptime N_CTX = SD3_LARGE_TEXT_TOKENS      # 410
comptime N_IMG = SD3_LARGE_IMAGE_TOKENS     # 4096
comptime S_JOINT = N_CTX + N_IMG            # 4506
comptime DEPTH = SD3_LARGE_DEPTH            # 38
comptime H_HEADS = SD3_LARGE_NUM_HEADS      # 38
comptime H_DIM = SD3_LARGE_HEAD_DIM         # 64
comptime HIDDEN = SD3_LARGE_HIDDEN          # 2432
comptime DUAL_BLOCKS = 0                    # No dual attention in Large
comptime CONTEXT_DIM = 4096                 # joint_attention_dim (T5 hidden width)

# ── verified model + tokenizer paths (match sd3_infer.rs constants exactly) ──
comptime MODEL_PATH = "/home/alex/.serenity/models/checkpoints/sd3.5_large.safetensors"
comptime CLIP_L_PATH = "/home/alex/.serenity/models/text_encoders/clip_l.safetensors"
comptime CLIP_G_PATH = "/home/alex/.serenity/models/text_encoders/clip_g.safetensors"
comptime T5_PATH = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.safetensors"
comptime CLIP_L_TOK = "/home/alex/.serenity/models/text_encoders/clip_l.tokenizer.json"
comptime CLIP_G_TOK = "/home/alex/.serenity/models/text_encoders/clip_g.tokenizer.json"
comptime T5_TOK = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.tokenizer.json"
comptime CLIP_G_TEXT_PROJ = "text_projection.weight"

comptime CLIP_PAD_ID = 49407   # CLIP eos == pad
comptime CLIP_EOS_ID = 49407
comptime T5_EOS_ID = 1
comptime T5_PAD_ID = 0


comptime S3PHASE_IDLE = 0
comptime S3PHASE_ENCODE = 1
comptime S3PHASE_LOAD = 2
comptime S3PHASE_DENOISE = 3
comptime S3PHASE_DECODE = 4


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
        String("echo -n '[sd3][vram] ") + tag
        + ": ' && nvidia-smi --query-gpu=memory.used --format=csv,noheader"
    )


def _to_bf16(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """F16/F32/BF16 -> BF16 (F16 goes through F32 to avoid a direct F16->BF16 path)."""
    if x.dtype() == STDtype.BF16:
        return cast_tensor(x, STDtype.BF16, ctx)
    if x.dtype() == STDtype.F16:
        var x_f32 = cast_tensor(x, STDtype.F32, ctx)
        return cast_tensor(x_f32, STDtype.BF16, ctx)
    return cast_tensor(x, STDtype.BF16, ctx)


def _fit_clip_ids(var ids: List[Int]) -> List[Int]:
    """Pad/truncate CLIP ids to 77, keeping a real EOS at the tail (HF CLIP: pad==eos).
    ClipTokenizer.encode already wrapped with BOS(49406)+EOS(49407). Matches
    sd3_infer.rs tokenize_clip (truncate to 77, pad 49407)."""
    if len(ids) > CLIP_LEN:
        var trimmed = List[Int]()
        for i in range(CLIP_LEN):
            trimmed.append(ids[i])
        trimmed[CLIP_LEN - 1] = CLIP_EOS_ID
        return trimmed^
    while len(ids) < CLIP_LEN:
        ids.append(CLIP_PAD_ID)
    return ids^


def _fit_t5_ids(var ids: List[Int]) -> List[Int]:
    """T5Tokenizer.encode already appended ONE EOS=1; sd3_infer.rs ALSO does
    ids.push(1) after HF tokenization -> append a SECOND EOS=1, then truncate to 256
    and pad with 0 (matches tokenize_t5 verbatim)."""
    ids.append(T5_EOS_ID)
    if len(ids) > T5_LEN:
        var trimmed = List[Int]()
        for i in range(T5_LEN):
            trimmed.append(ids[i])
        return trimmed^
    while len(ids) < T5_LEN:
        ids.append(T5_PAD_ID)
    return ids^


def _save_rgb_png_with_text(
    rgb: Tensor, path: String, params_json: String, ctx: DeviceContext
) raises:
    """[1,3,H,W] SIGNED float tensor → 8-bit RGB PNG with the job params in a
    serenity.genparams.v1 tEXt chunk. Quantization math == save_png's
    (_quantize, ValueRange.SIGNED); only the writer differs (tEXt support).
    Identical to sdxl_backend / qwenimage_backend._save_rgb_png_with_text."""
    var shape = rgb.shape()
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
        raise Error("sd3_backend: expected [1,3,H,W] rgb tensor")
    var height = shape[2]
    var width = shape[3]
    var host = rgb.to_host(ctx)
    var plane = height * width
    if len(host) != 3 * plane:
        raise Error("sd3_backend: rgb to_host size mismatch")
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


# ── SD3 CLIP encode: (penultimate_hidden [1,77,hidden], pooled [1,proj_dim]) ──
# Reimplements inference-flame ClipEncoder::encode_sd3 over the verified Mojo
# ClipEncoder building blocks (_embed, _layer[77], _w, slice_seq, _build_pad_mask):
#   - hidden_states[-2] = output after layer (num_layers - 2), captured pre-final-LN.
#   - pooled = final_layer_norm(last-layer hidden) @ EOS, then text_projection if
#     `text_proj` is provided (CLIP-G) else raw (CLIP-L).
#   - PURE CAUSAL mask (no key-padding): _build_pad_mask(S, heads, S-1).
# `text_proj` is an Optional CLIP-G text_projection.weight [hidden,hidden]; pass
# None for CLIP-L. Returns BF16 tensors (the encoders store BF16; layer math is
# F32-accum, BF16-store — matches the verified encode_sdxl path).
def _clip_encode_sd3(
    enc: ClipEncoder,
    var token_ids: List[Int],
    text_proj: Optional[Tensor],
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    var cfg = enc.config
    var hid = cfg.hidden_size
    var num_layers = cfg.num_layers
    var heads = cfg.num_heads

    # pad / truncate to 77, EOS-pad (already done by _fit_clip_ids, but be safe).
    token_ids = _fit_clip_ids(token_ids^)

    # first EOS position (argmax over id==eos returns the FIRST 1).
    var real_eos = CLIP_LEN - 1
    for i in range(CLIP_LEN):
        if token_ids[i] == CLIP_EOS_ID:
            real_eos = i
            break

    var hidden = enc._embed(token_ids, ctx)         # [1,77,hidden]
    var dtype = hidden.dtype()

    # PURE causal mask [1, H, 77, 77]: valid_key_end = S-1 => the j<=valid_key_end
    # clause is always true, leaving only the causal j<=i term (matches the
    # reference build_causal_mask).
    var mask_data = _build_pad_mask(CLIP_LEN, heads, CLIP_LEN - 1)
    var msh = List[Int]()
    msh.append(1)
    msh.append(heads)
    msh.append(CLIP_LEN)
    msh.append(CLIP_LEN)
    var mask = Tensor.from_host(mask_data, msh^, dtype, ctx)

    # transformer layers, capturing hidden_states[-2] (output after layer L-2).
    # `penultimate` must be definitely-initialized before the loop (the capture is
    # conditional), so seed it with a clone of the embeddings (cheap, no extra
    # embed kernel); it is overwritten at i == penultimate_idx below.
    var penultimate_idx = num_layers - 2
    var penultimate = hidden.clone(ctx)  # placeholder, overwritten below
    for i in range(num_layers):
        hidden = enc._layer[CLIP_LEN](i, hidden, mask, ctx)
        if i == penultimate_idx:
            penultimate = hidden.clone(ctx)

    # pooled = final_layer_norm(last-layer hidden) @ EOS, optionally projected.
    # _w returns a borrow of a Movable-not-Copyable Tensor, so bind with `ref`
    # (matching the encoder's own encode_sdxl) — `var` would attempt a copy.
    ref fw = enc._w(String("text_model.final_layer_norm.weight"))
    ref fb = enc._w(String("text_model.final_layer_norm.bias"))
    # The encoder exposes no public LN, so apply the verified ops.norm.layer_norm
    # directly (same affine LN the encoder uses for its own final_layer_norm).
    var last_hidden = _ops_layer_norm(hidden, fw, fb, cfg.layer_norm_eps, ctx)
    var pooled_slice = slice_seq(last_hidden, real_eos, ctx)   # [1,1,hidden]
    var psh = List[Int]()
    psh.append(1)
    psh.append(hid)
    var pooled = reshape(pooled_slice, psh^, ctx)              # [1,hidden]

    if text_proj:
        # text_embeds = pooled @ text_projection^T  (HF: text_projection is a
        # no-bias Linear [out,in]; ops.linear does y = x @ W^T).
        pooled = linear(pooled, text_proj.value(), Optional[Tensor](None), ctx)

    return (penultimate^, _to_bf16(pooled, ctx))


# ── zero-pad a [1,77,hidden] CLIP hidden state's last dim up to 4096 ──
def _zero_pad_last_dim(x: Tensor, target: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = x.shape()
    if len(sh) != 3:
        raise Error("sd3_backend: _zero_pad_last_dim expects [1,S,C]")
    var cur = sh[2]
    if cur == target:
        return _to_bf16(x, ctx)
    if cur > target:
        raise Error("sd3_backend: _zero_pad_last_dim source wider than target")
    var pad_c = target - cur
    var zeros_host = List[Float32]()
    var nz = sh[0] * sh[1] * pad_c
    for _ in range(nz):
        zeros_host.append(Float32(0.0))
    var zsh = List[Int]()
    zsh.append(sh[0])
    zsh.append(sh[1])
    zsh.append(pad_c)
    var zeros = Tensor.from_host(zeros_host, zsh^, STDtype.BF16, ctx)
    return concat(2, ctx, _to_bf16(x, ctx), zeros)


# ── SD3 conditioning bundle (per job) ──────────────────────────────────────────
struct Sd3Caps(Movable):
    var context: Tensor         # [1,410,4096] BF16 (cond)
    var context_uncond: Tensor  # [1,410,4096] BF16 (uncond)
    var pooled: Tensor          # [1,2048]     BF16 (cond)
    var pooled_uncond: Tensor   # [1,2048]     BF16 (uncond)

    def __init__(
        out self, var context: Tensor, var context_uncond: Tensor,
        var pooled: Tensor, var pooled_uncond: Tensor,
    ):
        self.context = context^
        self.context_uncond = context_uncond^
        self.pooled = pooled^
        self.pooled_uncond = pooled_uncond^


# ── assemble SD3 context [1,410,4096] + pooled [1,2048] for ONE prompt ──
def _assemble_one(
    text: String,
    clip_l: ClipEncoder,
    clip_g: ClipEncoder,
    text_proj_g: Tensor,
    t5: T5Encoder[T5_LEN],
    clip_l_tok: ClipTokenizer,
    clip_g_tok: ClipTokenizer,
    t5_tok: T5Tokenizer,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    # CLIP-L: penultimate hidden [1,77,768], raw pooled [1,768] (no text_projection).
    var l_ids = _fit_clip_ids(clip_l_tok.encode(text))
    var l_out = _clip_encode_sd3(clip_l, l_ids^, Optional[Tensor](None), ctx)
    var l_hidden = l_out[0].clone(ctx)
    var l_pool = l_out[1].clone(ctx)

    # CLIP-G: penultimate hidden [1,77,1280], projected pooled [1,1280].
    var g_ids = _fit_clip_ids(clip_g_tok.encode(text))
    var g_out = _clip_encode_sd3(
        clip_g, g_ids^, Optional[Tensor](text_proj_g.clone(ctx)), ctx
    )
    var g_hidden = g_out[0].clone(ctx)
    var g_pool = g_out[1].clone(ctx)

    # T5-XXL: [1,256,4096] (encode appends one EOS; _fit_t5_ids adds the second + pad).
    var t5_ids = _fit_t5_ids(t5_tok.encode(text))
    var t5_hidden = _to_bf16(t5.encode(t5_ids^, ctx), ctx)     # [1,256,4096]

    # context = cat([pad(l_hidden,4096), pad(g_hidden,4096), t5_hidden], dim=1).
    var l_pad = _zero_pad_last_dim(l_hidden, CONTEXT_DIM, ctx)  # [1,77,4096]
    var g_pad = _zero_pad_last_dim(g_hidden, CONTEXT_DIM, ctx)  # [1,77,4096]
    var context = concat(1, ctx, l_pad, g_pad, t5_hidden)      # [1,410,4096]
    context = _to_bf16(context, ctx)

    # pooled = cat([l_pool, g_pool], dim=1) -> [1,2048].
    var pooled = concat(1, ctx, _to_bf16(l_pool, ctx), _to_bf16(g_pool, ctx))
    pooled = _to_bf16(pooled, ctx)

    return (context^, pooled^)


# ── SD3 Large MMDiT forward (one pass) — verbatim from sd3_sample_cli ──
def _sd3_large_forward(
    latent: Tensor,          # [1, 16, LH, LW] BF16
    sigma: Float32,
    context: Tensor,         # [1, N_CTX, 4096] BF16
    pooled: Tensor,          # [1, 2048] BF16
    gate: SD3MMDiTPreBlockGate,
    loader: BlockLoader,
    ctx: DeviceContext,
) raises -> Tensor:
    var x_tokens = gate.latent_patch_embed[LH, LW](latent, ctx)
    var c = gate.conditioning(sigma, pooled, ctx)
    var ctx_tokens = gate.context_embed[N_CTX](context, ctx)

    for i in range(DEPTH):
        var is_last = (i == DEPTH - 1)
        var block_prefix = String("model.diffusion_model.joint_blocks.") + String(i)
        loader.prefetch_block(block_prefix)
        var blk = loader.load_block(block_prefix, ctx)
        _sd3_joint_block[1, S_JOINT, N_CTX, N_IMG, H_HEADS, H_DIM](
            ctx_tokens, x_tokens, c, blk, i, is_last, DUAL_BLOCKS, HIDDEN, ctx
        )
        unload_block(blk^)

    var patch_out = gate.final_layer_tokens(x_tokens, c, ctx)
    return gate.final_unpatchify[LH, LW](patch_out, ctx)


struct Sd3Backend(GenBackend, Movable):
    var ctx: DeviceContext

    # ── resident across jobs (pre/post gate + BlockLoader handle, loaded once) ──
    var loaded: Bool
    var gate: List[ArcPointer[SD3MMDiTPreBlockGate]]  # 0/1 (resident gate)
    var loader: List[ArcPointer[BlockLoader]]         # 0/1 (mmap handle)

    # ── per-job state (cleared on done/failed/cancelled) ──
    var active: Bool
    var cancel_flag: Bool
    var phase: Int
    var announced: Bool
    var cur: Int
    var params: JobParams
    var cfg: Float32
    var caps: List[ArcPointer[Sd3Caps]]                 # 0/1
    var sched: List[ArcPointer[SD3FlowMatchScheduler]]  # 0/1
    var latent: List[ArcPointer[Tensor]]                # 0/1 ([1,16,LH,LW] BF16)

    def __init__(out self) raises:
        self.ctx = DeviceContext()
        self.loaded = False
        self.gate = List[ArcPointer[SD3MMDiTPreBlockGate]]()
        self.loader = List[ArcPointer[BlockLoader]]()
        self.active = False
        self.cancel_flag = False
        self.phase = S3PHASE_IDLE
        self.announced = False
        self.cur = 0
        self.params = JobParams()
        self.cfg = Float32(4.5)
        self.caps = List[ArcPointer[Sd3Caps]]()
        self.sched = List[ArcPointer[SD3FlowMatchScheduler]]()
        self.latent = List[ArcPointer[Tensor]]()

    def backend_name(self) -> String:
        return String("sd3")

    def model_name(self) -> String:
        return String("SD3.5 Large")

    def resident_model(self) -> String:
        """Matches the /v1/models scan entry for the resident checkpoint
        (the sd3.5_large.safetensors checkpoint)."""
        return String("sd3.5_large.safetensors") if self.loaded else String("")

    # ── job admission ─────────────────────────────────────────────────────────
    def start(mut self, params: JobParams) raises:
        if self.active:
            raise Error("Sd3Backend.start: a job is already running")
        reject_unsupported_common_runtime_params(params, String("sd3"))
        reject_unsupported_reference_image_params(params, String("sd3"))
        reject_unsupported_inpaint_conditioning_params(params, String("sd3"))
        reject_unsupported_qwen_edit_conditioning_params(params, String("sd3"))
        reject_unsupported_conditioning_mask_params(params, String("sd3"))
        reject_unsupported_mask_image_params(params, String("sd3"))
        reject_unsupported_lanpaint_params(params, String("sd3"))
        var sampler_admission = sampler_admission_for_backend(String("sd3"), params.sampler)
        if not sampler_admission.supported:
            raise Error(
                String("sd3: unsupported sampler '") + params.sampler
                + String("'; ") + sampler_admission.reason
            )
        var scheduler_admission = scheduler_admission_for_backend(String("sd3"), params.scheduler)
        if not scheduler_admission.supported:
            raise Error(
                String("sd3: unsupported scheduler '") + params.scheduler
                + String("'; ") + scheduler_admission.reason
            )
        # 1024x1024 only: the SD3 Large DiT attention shape (S_JOINT) and the
        # pre-block gate are comptime-fixed at LH=LW=128.
        if not (params.width == 1024 and params.height == 1024):
            raise Error(
                String("sd3: unsupported size ") + String(params.width)
                + "x" + String(params.height)
                + " — only 1024x1024 is served (the SD3.5 Large DiT attention"
                + " shape is comptime-fixed; resolution changes need a recompile)"
            )
        if len(params.loras) > 0:
            raise Error(
                "sd3: LoRA is not supported for SD3.5 Large in this backend yet"
                " (no LoRA overlay path wired); submit without a LoRA"
            )
        if params.init_image.byte_length() > 0:
            raise Error(
                "sd3: img2img is not supported for SD3.5 Large yet;"
                " submit without an init image"
            )
        # Warn-loud (never silently drop) on any advanced-sampling knob set but
        # unsupported by this fixed flow-match Euler path.
        warn_unsupported_advanced_sampling_params(params, String("sd3"), List[String]())
        self.params = params.copy()
        self.cfg = Float32(params.cfg)
        self.active = True
        self.cancel_flag = False
        self.cur = 0
        self.announced = False
        self.phase = S3PHASE_ENCODE

    def cancel(mut self):
        self.cancel_flag = True

    def between_jobs_trim(mut self) raises:
        """Reclaim the per-job transient peak (CLIP-L+G+T5 encoders ~11 GB, the VAE
        decoder ~330 MB, per-forward streamed block + activations) back to the OS via
        cuMemPoolTrimTo. The resident pre/post gate weights have live suballocations
        and are NOT reclaimed; the BlockLoader holds an mmap (no device weights)."""
        var before = cu_mem_get_info()
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        self.ctx.synchronize()
        var after = cu_mem_get_info()
        print("[sd3] between-jobs trim: used",
              before.used_bytes() // (1024 * 1024), "->",
              after.used_bytes() // (1024 * 1024), "MiB (reclaimed",
              (before.used_bytes() - after.used_bytes()) // (1024 * 1024), "MiB)")

    # ── per-job prep ───────────────────────────────────────────────────────────
    def _encode(mut self) raises:
        """Runtime CLIP-L+G+T5 encode of params.prompt AND params.negative into the
        SD3 context [1,410,4096] / pooled [1,2048] (encoders loaded then freed)."""
        _print_vram("before CLIP-L+G + T5 load")
        var clip_l = ClipEncoder.load(String(CLIP_L_PATH), ClipConfig.clip_l(), self.ctx)
        var clip_g = ClipEncoder.load(String(CLIP_G_PATH), ClipConfig.clip_g(), self.ctx)
        # CLIP-G text_projection.weight lives OUTSIDE text_model.* so ClipEncoder.load
        # skips it; load it directly from the CLIP-G safetensors. [1280,1280].
        var g_st = ShardedSafeTensors.open(String(CLIP_G_PATH))
        var text_proj_g = Tensor.from_view(g_st.tensor_view(String(CLIP_G_TEXT_PROJ)), self.ctx)
        var t5 = T5Encoder[T5_LEN].load(String(T5_PATH), T5Config.t5_xxl(), self.ctx)
        var clip_l_tok = ClipTokenizer(String(CLIP_L_TOK))
        var clip_g_tok = ClipTokenizer(String(CLIP_G_TOK))
        var t5_tok = T5Tokenizer(String(T5_TOK))

        var pos = _assemble_one(
            self.params.prompt, clip_l, clip_g, text_proj_g, t5,
            clip_l_tok, clip_g_tok, t5_tok, self.ctx,
        )
        var neg = _assemble_one(
            self.params.negative, clip_l, clip_g, text_proj_g, t5,
            clip_l_tok, clip_g_tok, t5_tok, self.ctx,
        )
        # Tensor is Movable-not-Copyable and a tuple subscript yields a BORROW, so
        # materialize each owned conditioning tensor via the proven .clone(ctx) idiom
        # before the encoders drop at scope exit.
        var caps = Sd3Caps(
            pos[0].clone(self.ctx), neg[0].clone(self.ctx),
            pos[1].clone(self.ctx), neg[1].clone(self.ctx),
        )
        # clip_l/clip_g/t5/text_proj_g drop here (Movable-not-Copyable -> freed).
        _print_vram("after CLIP+T5 encode (encoders freed)")
        self.caps = List[ArcPointer[Sd3Caps]]()
        self.caps.append(ArcPointer(caps^))

    def _load_model(mut self) raises:
        """Load the SD3.5 Large pre/post gate + BlockLoader (once; stays resident)."""
        if self.loaded:
            return
        _print_vram("before SD3 gate + BlockLoader load")
        print("[sd3] loading SD3.5 Large pre/post gate (resident) + block loader for", DEPTH, "blocks")
        self.gate = List[ArcPointer[SD3MMDiTPreBlockGate]]()
        self.gate.append(ArcPointer(SD3MMDiTPreBlockGate.load_large_default(self.ctx)))
        self.loader = List[ArcPointer[BlockLoader]]()
        self.loader.append(ArcPointer(BlockLoader.open(String(MODEL_PATH))))
        self.loaded = True
        _print_vram("after SD3 gate + BlockLoader load (resident)")

    def _prepare_job(mut self) raises:
        """Flow-match scheduler (honors steps + large shift) + seeded BF16 latent."""
        self.sched = List[ArcPointer[SD3FlowMatchScheduler]]()
        self.sched.append(
            ArcPointer(SD3FlowMatchScheduler(self.params.steps, sd3_large_schedule_shift()))
        )
        var nsh = [1, LC, LH, LW]
        var noise = randn(nsh.copy(), UInt64(self.params.seed), STDtype.BF16, self.ctx)
        if self.params.variation_strength > 0.0:
            var vnoise = randn(
                nsh.copy(),
                UInt64(self.params.variation_seed + self.params.image_index),
                STDtype.BF16,
                self.ctx,
            )
            var base_h = noise.to_host(self.ctx)
            var var_h = vnoise.to_host(self.ctx)
            var blended = swarm_variation_noise_chw(
                base_h, var_h, LC, LH, LW, self.params.variation_strength
            )
            noise = Tensor.from_host(blended, nsh.copy(), STDtype.BF16, self.ctx)
        self.latent = List[ArcPointer[Tensor]]()
        self.latent.append(ArcPointer(noise^))
        print(
            "[sd3] job", self.params.job_id, ":", self.params.steps,
            "steps, cfg", self.cfg, "seed", self.params.seed,
            "size", self.params.width, "x", self.params.height,
        )

    # ── one denoise step (CFG dual streamed forward + flow-match Euler) ──────────
    def _denoise_one(mut self) raises:
        var i = self.cur
        var sigma = self.sched[0][].timestep(i)
        var dt = self.sched[0][].dt(i)
        var v_cond = _sd3_large_forward(
            self.latent[0][], sigma, self.caps[0][].context, self.caps[0][].pooled,
            self.gate[0][], self.loader[0][], self.ctx,
        )
        var v_uncond = _sd3_large_forward(
            self.latent[0][], sigma, self.caps[0][].context_uncond,
            self.caps[0][].pooled_uncond, self.gate[0][], self.loader[0][], self.ctx,
        )
        var velocity = sd3_cfg(v_cond, v_uncond, self.cfg, self.ctx)
        var x_new = sd3_euler_step(self.latent[0][], velocity, dt, self.ctx)
        self.latent = List[ArcPointer[Tensor]]()
        self.latent.append(ArcPointer(x_new^))

    # ── final decode + PNG(tEXt) ──────────────────────────────────────────────
    def _decode_and_save(mut self) raises -> String:
        var png_path = self.params.out_dir + "/" + self.params.job_id + ".png"
        var latent = self.latent[0][].clone(self.ctx)
        # Per-job conditioning is dead weight at decode; free before the decoder.
        self.caps = List[ArcPointer[Sd3Caps]]()
        self.sched = List[ArcPointer[SD3FlowMatchScheduler]]()
        self.latent = List[ArcPointer[Tensor]]()
        # SD3 streams its blocks (no large resident denoiser at decode); the only
        # resident weights are the small pre/post gate. Trim the mempool before the
        # VAE decode to reclaim the per-forward block/activation peak. (The gate +
        # BlockLoader stay resident for the next job.)
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        self.ctx.synchronize()
        print("[sd3] loading embedded SD3 VAE decoder + decode")
        var vae = load_sd3_embedded_ldm_decoder[LH, LW](String(MODEL_PATH), self.ctx)
        var img = vae.decode(latent, self.ctx)
        _save_rgb_png_with_text(img, png_path, self.params.params_json, self.ctx)
        return png_path

    def _clear_job(mut self):
        self.active = False
        self.phase = S3PHASE_IDLE
        self.cur = 0
        self.cancel_flag = False
        self.announced = False
        self.caps = List[ArcPointer[Sd3Caps]]()
        self.sched = List[ArcPointer[SD3FlowMatchScheduler]]()
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
            if self.phase == S3PHASE_ENCODE:
                if not self.announced:
                    # announce BEFORE the long blocking encode tick (per-job
                    # CLIP-L + CLIP-G + T5-XXL load + dual-prompt forward).
                    self.announced = True
                    r.step = 0
                    r.phase = String("encoding")
                    return r^
                self._encode()
                self.announced = False
                self.phase = S3PHASE_LOAD
                r.step = 0
                return r^
            if self.phase == S3PHASE_LOAD:
                if not self.loaded:
                    if not self.announced:
                        self.announced = True
                        r.step = 0
                        r.phase = String("loading")
                        return r^
                    self._load_model()
                    self.announced = False
                self._prepare_job()
                self.phase = S3PHASE_DENOISE
                r.step = 0
                return r^
            if self.phase == S3PHASE_DENOISE:
                self._denoise_one()
                self.cur += 1
                r.step = self.cur
                if self.cur >= self.params.steps:
                    self.phase = S3PHASE_DECODE
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
