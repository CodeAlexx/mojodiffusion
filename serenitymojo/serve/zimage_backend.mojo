# serenitymojo.serve.zimage_backend — the real Z-Image GenBackend.
#
# Wraps the VERIFIED serenitymojo/pipeline/zimage_generate.mojo stages behind
# the pull-based GenBackend seam (backend.mojo). EVERY numeric convention is
# reused from zimage_generate (its helpers are imported, NOT re-derived):
#   tokenizer → Qwen3-4B layer-34 encode → CFG dual-forward denoise
#   (rectified-flow Euler, shift=6.0 sigmas) → Z-Image VAE decode → PNG SIGNED.
#
# Residency model (24 GB GPU):
#   * DiT weights (aux + nr/cr/main blocks + final linear) and BOTH VAE decoder
#     specializations (64²- and 128²-latent) are loaded ONCE (first job) and
#     stay resident across jobs.
#   * The Qwen3-4B text encoder (~7.5 GB bf16) is loaded → used → freed PER JOB
#     inside the ENCODE step (encode_captions_fixed frees it on return), because
#     DiT-resident (~12 GB) + encoder + dual-forward activations does fit, but
#     keeping the encoder resident too leaves no headroom for denoise
#     activations at S=4352. Per-job encoder reload is served from page cache.
#
# step() state machine: LOAD(once) → ENCODE → DENOISE×steps → DECODE → done.
# One denoise step (CFG dual forward + Euler update) per step() call; cancel()
# makes the next step() return cancelled and frees all per-job tensors.
#
# Size support: the latent grid is comptime in the verified stack, so only the
# specializations wired here are valid: 512x512 (HL=WL=64) and 1024x1024
# (HL=WL=128). Other sizes fail the job with a clear error at start().
#
# LoRA: forward overlay only (pipeline's own load_zimage_lora_main_only_resume
# path) — never fused. At most ONE overlay per job (the verified path loads a
# single adapter set); missing file or >1 entries fail the job at start().

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
from serenitymojo.models.vae.zimage_decoder import ZImageDecoder
from serenitymojo.models.zimage.weights import (
    ZImageBlockWeights, load_zimage_block_weights_prefixed_mixed,
)
from serenitymojo.models.zimage.zimage_stack_lora import (
    ZImageLoraDeviceSet, load_zimage_lora_main_only_resume,
    build_zimage_zero_lora_device_set, zimage_lora_set_to_device,
)
from serenitymojo.models.zimage.real_weights import (
    ZImageRealAux, load_zimage_real_aux, build_rope, build_positions,
)
from serenitymojo.ops.tensor_algebra import mul_scalar, add
from serenitymojo.pipeline.zimage_generate import (
    encode_captions_fixed, gaussian_noise, _cast, _cap_seq_with_pad,
    _unified_positions, _cfg_pred_overlay, _build_sigmas, _path_exists,
    TRANSFORMER, VAE_DIR,
    H, Dh, D, F, PATCH, CAPLEN_MAX, ROPE_THETA, AXIS0, AXIS1, AXIS2,
    NUM_NR, NUM_CR, MAIN_DEPTH, RANK, ALPHA,
)
from serenitymojo.serve.backend import GenBackend, JobParams, StepResult

comptime TArc = ArcPointer[Tensor]

comptime PHASE_IDLE = 0
comptime PHASE_LOAD = 1
comptime PHASE_ENCODE = 2
comptime PHASE_DENOISE = 3
comptime PHASE_DECODE = 4

comptime GENPARAMS_TEXT_KEY = "serenity.genparams.v1"


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
        String("echo -n '[zimage][vram] ") + tag
        + ": ' && nvidia-smi --query-gpu=memory.used --format=csv,noheader"
    )


def _blocks_bytes(blocks: List[ZImageBlockWeights]) -> Int:
    var total = 0
    for i in range(len(blocks)):
        total += blocks[i].n1[].nbytes()
        total += blocks[i].wq[].nbytes() + blocks[i].wk[].nbytes()
        total += blocks[i].wv[].nbytes() + blocks[i].wo[].nbytes()
        total += blocks[i].q_norm[].nbytes() + blocks[i].k_norm[].nbytes()
        total += blocks[i].n2[].nbytes() + blocks[i].fn1[].nbytes()
        total += blocks[i].w1[].nbytes() + blocks[i].w3[].nbytes()
        total += blocks[i].w2[].nbytes() + blocks[i].fn2[].nbytes()
    return total


def _save_rgb_png_with_text(
    rgb: Tensor, path: String, params_json: String, ctx: DeviceContext
) raises:
    """[1,3,H,W] SIGNED float tensor → 8-bit RGB PNG with the job params in a
    serenity.genparams.v1 tEXt chunk. Quantization math == save_png's
    (_quantize, ValueRange.SIGNED); only the writer differs (tEXt support)."""
    var shape = rgb.shape()
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
        raise Error("zimage_backend: expected [1,3,H,W] rgb tensor")
    var height = shape[2]
    var width = shape[3]
    var host = rgb.to_host(ctx)
    var plane = height * width
    if len(host) != 3 * plane:
        raise Error("zimage_backend: rgb to_host size mismatch")
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


struct ZImageBackend(GenBackend, Movable):
    var ctx: DeviceContext

    # ── resident across jobs (loaded once, first job) ──
    var loaded: Bool
    var aux: List[ArcPointer[ZImageRealAux]]            # 0/1
    var nr_blocks: List[ZImageBlockWeights]
    var cr_blocks: List[ZImageBlockWeights]
    var main_blocks: List[ZImageBlockWeights]
    var final_lin: List[TArc]                           # [w, b]
    var x_pad_h: List[Float32]
    var cap_pad_h: List[Float32]
    var vae64: List[ArcPointer[ZImageDecoder[64, 64]]]  # 0/1
    var vae128: List[ArcPointer[ZImageDecoder[128, 128]]]

    # ── per-job state (cleared on done/failed/cancelled) ──
    var active: Bool
    var cancel_flag: Bool
    var phase: Int
    var cur: Int
    var params: JobParams
    var hl: Int
    var wl: Int
    var cfg: Float32
    var lora_path: String       # resolved file ("" = base / zero overlay)
    var lora_mult: Float32
    var sigmas: List[Float32]
    var cap_seq_cond: List[Float32]
    var cap_seq_uncond: List[Float32]
    var rope: List[TArc]        # 12: cond x/cap/uni cos+sin, then uncond
    var lora_dev: List[ZImageLoraDeviceSet]  # 0/1
    var latent: List[TArc]      # 0/1

    def __init__(out self) raises:
        self.ctx = DeviceContext()
        self.loaded = False
        self.aux = List[ArcPointer[ZImageRealAux]]()
        self.nr_blocks = List[ZImageBlockWeights]()
        self.cr_blocks = List[ZImageBlockWeights]()
        self.main_blocks = List[ZImageBlockWeights]()
        self.final_lin = List[TArc]()
        self.x_pad_h = List[Float32]()
        self.cap_pad_h = List[Float32]()
        self.vae64 = List[ArcPointer[ZImageDecoder[64, 64]]]()
        self.vae128 = List[ArcPointer[ZImageDecoder[128, 128]]]()
        self.active = False
        self.cancel_flag = False
        self.phase = PHASE_IDLE
        self.cur = 0
        self.params = JobParams()
        self.hl = 0
        self.wl = 0
        self.cfg = Float32(0.0)
        self.lora_path = String("")
        self.lora_mult = Float32(1.0)
        self.sigmas = List[Float32]()
        self.cap_seq_cond = List[Float32]()
        self.cap_seq_uncond = List[Float32]()
        self.rope = List[TArc]()
        self.lora_dev = List[ZImageLoraDeviceSet]()
        self.latent = List[TArc]()

    def backend_name(self) -> String:
        return String("zimage")

    def model_name(self) -> String:
        return String("Z-Image")

    # ── job admission ─────────────────────────────────────────────────────────
    def start(mut self, params: JobParams) raises:
        if self.active:
            raise Error("ZImageBackend.start: a job is already running")
        var hl = params.height // 8
        var wl = params.width // 8
        if not ((hl == 64 and wl == 64) or (hl == 128 and wl == 128)):
            raise Error(
                String("zimage: unsupported size ") + String(params.width)
                + "x" + String(params.height)
                + " — the latent grid is comptime-specialized; supported:"
                + " 512x512 and 1024x1024"
            )
        # LoRA: the verified pipeline path applies ONE forward-overlay adapter
        # set; resolve + validate up front so a bad request fails immediately.
        var lora_path = String("")
        var lora_mult = Float32(1.0)
        if len(params.loras) > 1:
            raise Error(
                "zimage: at most one LoRA overlay per job is supported"
                " (the verified pipeline LoRA path loads a single adapter set)"
            )
        if len(params.loras) == 1:
            var name = params.loras[0].name
            if _path_exists(name):
                lora_path = name.copy()
            elif _path_exists(name + ".safetensors"):
                lora_path = name + ".safetensors"
            else:
                raise Error(String("zimage: LoRA file not found: ") + name)
            lora_mult = Float32(params.loras[0].weight)
        self.params = params.copy()
        self.hl = hl
        self.wl = wl
        self.cfg = Float32(params.cfg)
        self.lora_path = lora_path^
        self.lora_mult = lora_mult
        self.active = True
        self.cancel_flag = False
        self.cur = 0
        self.phase = PHASE_LOAD

    def cancel(mut self):
        self.cancel_flag = True

    # ── resident weights (DiT + both VAE grids) ──────────────────────────────
    def _ensure_resident(mut self) raises:
        if self.loaded:
            return
        _print_vram("before resident load")
        print("[zimage] loading resident DiT weights from", TRANSFORMER)
        var st = ShardedSafeTensors.open(String(TRANSFORMER))
        var aux = load_zimage_real_aux(st, NUM_NR, MAIN_DEPTH, self.ctx)
        for i in range(NUM_NR):
            self.nr_blocks.append(
                load_zimage_block_weights_prefixed_mixed(
                    st, String("noise_refiner.") + String(i), self.ctx
                )
            )
        for i in range(NUM_CR):
            self.cr_blocks.append(
                load_zimage_block_weights_prefixed_mixed(
                    st, String("context_refiner.") + String(i), self.ctx
                )
            )
        for i in range(MAIN_DEPTH):
            self.main_blocks.append(
                load_zimage_block_weights_prefixed_mixed(
                    st, String("layers.") + String(i), self.ctx
                )
            )
        self.final_lin.append(TArc(aux.final_lin_w[].clone(self.ctx)))
        self.final_lin.append(TArc(aux.final_lin_b[].clone(self.ctx)))
        self.x_pad_h = aux.x_pad_token[].to_host(self.ctx)
        self.cap_pad_h = aux.cap_pad_token[].to_host(self.ctx)
        self.aux.append(ArcPointer(aux^))
        # VAE decoders are loaded lazily per grid on first decode (and then
        # stay resident): the 128-grid decode is VRAM-critical and must
        # allocate AFTER the DiT release, not before the first job.
        self.loaded = True
        var dit_bytes = (
            _blocks_bytes(self.nr_blocks) + _blocks_bytes(self.cr_blocks)
            + _blocks_bytes(self.main_blocks)
            + self.final_lin[0][].nbytes() + self.final_lin[1][].nbytes()
        )
        print("[zimage] resident DiT block bytes:", dit_bytes,
              "(", dit_bytes // (1024 * 1024), "MiB )")
        _print_vram("after resident load")

    # ── per-job prep: text encode + rope + noise + sigmas + LoRA overlay ─────
    def _prepare_job(mut self) raises:
        # Qwen3-4B is loaded, used, and freed inside encode_captions_fixed.
        var caps = encode_captions_fixed(
            self.params.prompt, self.params.negative, self.ctx
        )
        _print_vram("after text encode (encoder freed)")
        self.cap_seq_cond = _cap_seq_with_pad(
            self.aux[0][], caps.cond, caps.real_cond, self.cap_pad_h, self.ctx
        )
        self.cap_seq_uncond = _cap_seq_with_pad(
            self.aux[0][], caps.uncond, caps.real_uncond, self.cap_pad_h, self.ctx
        )

        var ht = self.hl // PATCH
        var wt = self.wl // PATCH
        var n_img_real = ht * wt
        var img_pad = (32 - (n_img_real % 32)) % 32
        var n_img = n_img_real + img_pad

        # rope tables: cond then uncond, each x/cap/uni cos+sin (== _denoise).
        self.rope = List[TArc]()
        var pos_cond = build_positions(n_img, ht, wt, CAPLEN_MAX, caps.real_cond)
        var x_pos_cond = pos_cond[0].copy()
        var cap_pos_cond = pos_cond[1].copy()
        var uni_pos_cond = _unified_positions(x_pos_cond, cap_pos_cond)
        var xr_cond = build_rope(x_pos_cond, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, self.ctx)
        self.rope.append(xr_cond[0].copy()); self.rope.append(xr_cond[1].copy())
        var cr_cond = build_rope(cap_pos_cond, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, self.ctx)
        self.rope.append(cr_cond[0].copy()); self.rope.append(cr_cond[1].copy())
        var ur_cond = build_rope(uni_pos_cond, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, self.ctx)
        self.rope.append(ur_cond[0].copy()); self.rope.append(ur_cond[1].copy())

        var pos_uncond = build_positions(n_img, ht, wt, CAPLEN_MAX, caps.real_uncond)
        var x_pos_uncond = pos_uncond[0].copy()
        var cap_pos_uncond = pos_uncond[1].copy()
        var uni_pos_uncond = _unified_positions(x_pos_uncond, cap_pos_uncond)
        var xr_uncond = build_rope(x_pos_uncond, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, self.ctx)
        self.rope.append(xr_uncond[0].copy()); self.rope.append(xr_uncond[1].copy())
        var cr_uncond = build_rope(cap_pos_uncond, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, self.ctx)
        self.rope.append(cr_uncond[0].copy()); self.rope.append(cr_uncond[1].copy())
        var ur_uncond = build_rope(uni_pos_uncond, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, self.ctx)
        self.rope.append(ur_uncond[0].copy()); self.rope.append(ur_uncond[1].copy())

        # LoRA forward overlay (NEVER fused) — or the zero overlay in base mode.
        self.lora_dev = List[ZImageLoraDeviceSet]()
        if self.lora_path.byte_length() > 0:
            var lora_alpha = ALPHA * self.lora_mult
            print("[zimage][lora] loading overlay", self.lora_path)
            var lora = load_zimage_lora_main_only_resume(
                NUM_NR, NUM_CR, MAIN_DEPTH, RANK, lora_alpha, D, F,
                self.lora_path, self.ctx,
            )
            self.lora_dev.append(zimage_lora_set_to_device(lora, self.ctx))
        else:
            self.lora_dev.append(
                build_zimage_zero_lora_device_set(NUM_NR, NUM_CR, MAIN_DEPTH, self.ctx)
            )

        # seeded initial latent (verbatim noise math), kept BF16 at the boundary.
        var noise = gaussian_noise(16 * self.hl * self.wl, UInt64(self.params.seed))
        var nshape = [1, 16, self.hl, self.wl]
        self.latent = List[TArc]()
        self.latent.append(
            TArc(Tensor.from_host(noise, nshape^, STDtype.BF16, self.ctx))
        )
        self.sigmas = _build_sigmas(self.params.steps)
        print(
            "[zimage] job", self.params.job_id, ":", self.params.steps,
            "steps, cfg", self.cfg, "seed", self.params.seed,
            "size", self.params.width, "x", self.params.height,
        )

    # ── one denoise step (CFG dual forward + Euler), comptime grid ───────────
    def _denoise_one[HL: Int, WL: Int](mut self) raises:
        comptime HT = HL // PATCH
        comptime WT = WL // PATCH
        comptime N_IMG_REAL = HT * WT
        comptime IMG_PAD = (32 - (N_IMG_REAL % 32)) % 32
        comptime N_IMG = N_IMG_REAL + IMG_PAD
        comptime N_TXT = CAPLEN_MAX
        comptime S = N_IMG + N_TXT
        var i = self.cur
        var t = Float32(1.0) - self.sigmas[i]  # DiT timestep convention (= 1 - sigma)
        var x = self.latent[0].copy()
        # pred = -(vc + cfg*(vc - vu)) — _cfg_pred_overlay applies the negation.
        var pred = _cfg_pred_overlay[HL, WL, N_IMG, N_TXT, S](
            x[], t, self.cfg, self.cap_seq_cond, self.cap_seq_uncond,
            self.nr_blocks, self.cr_blocks, self.main_blocks, self.lora_dev[0],
            self.aux[0][], self.final_lin[0][], self.final_lin[1][], self.x_pad_h,
            self.rope[0][], self.rope[1][], self.rope[2][], self.rope[3][],
            self.rope[4][], self.rope[5][],
            self.rope[6][], self.rope[7][], self.rope[8][], self.rope[9][],
            self.rope[10][], self.rope[11][],
            False, self.ctx,
        )
        # Euler: x += (sigma_next - sigma) * pred (latent F32 compute, BF16 carrier)
        var dt = self.sigmas[i + 1] - self.sigmas[i]
        var x_compute = _cast(x[], STDtype.F32, self.ctx)
        var x_new = _cast(
            add(x_compute, mul_scalar(pred, dt, self.ctx), self.ctx),
            STDtype.BF16, self.ctx,
        )
        self.latent = List[TArc]()
        self.latent.append(TArc(x_new^))

    # ── final decode + PNG(tEXt) ──────────────────────────────────────────────
    def _decode_and_save(mut self) raises -> String:
        var png_path = self.params.out_dir + "/" + self.params.job_id + ".png"
        var lat = self.latent[0].copy()
        # Per-job conditioning tensors are dead weight at decode time; free
        # them BEFORE the decode allocations (the 1024² decode is VRAM-tight).
        self.rope = List[TArc]()
        self.cap_seq_cond = List[Float32]()
        self.cap_seq_uncond = List[Float32]()
        self.lora_dev = List[ZImageLoraDeviceSet]()
        self.latent = List[TArc]()
        if self.hl == 64:
            if len(self.vae64) == 0:
                print("[zimage] loading VAE decoder (64-latent grid; stays resident)")
                self.vae64.append(ArcPointer(ZImageDecoder[64, 64].load(VAE_DIR, self.ctx)))
            var rgb = self.vae64[0][].decode(
                _cast(lat[], STDtype.BF16, self.ctx), self.ctx
            )
            _save_rgb_png_with_text(rgb, png_path, self.params.params_json, self.ctx)
        else:
            if len(self.vae128) == 0:
                print("[zimage] loading VAE decoder (128-latent grid; stays resident)")
                self.vae128.append(ArcPointer(ZImageDecoder[128, 128].load(VAE_DIR, self.ctx)))
            var rgb = self.vae128[0][].decode(
                _cast(lat[], STDtype.BF16, self.ctx), self.ctx
            )
            _save_rgb_png_with_text(rgb, png_path, self.params.params_json, self.ctx)
        return png_path

    def _clear_job(mut self):
        """Free per-job tensors (resident weights stay)."""
        self.active = False
        self.phase = PHASE_IDLE
        self.cur = 0
        self.cancel_flag = False
        self.sigmas = List[Float32]()
        self.cap_seq_cond = List[Float32]()
        self.cap_seq_uncond = List[Float32]()
        self.rope = List[TArc]()
        self.lora_dev = List[ZImageLoraDeviceSet]()
        self.latent = List[TArc]()

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
            if self.phase == PHASE_LOAD:
                self._ensure_resident()
                self.phase = PHASE_ENCODE
                r.step = 0
                return r^
            if self.phase == PHASE_ENCODE:
                self._prepare_job()
                self.phase = PHASE_DENOISE
                r.step = 0
                return r^
            if self.phase == PHASE_DENOISE:
                if self.hl == 64:
                    self._denoise_one[64, 64]()
                else:
                    self._denoise_one[128, 128]()
                self.cur += 1
                r.step = self.cur
                if self.cur >= self.params.steps:
                    self.phase = PHASE_DECODE
                return r^
            # PHASE_DECODE: VAE decode + PNG with serenity.genparams.v1 tEXt.
            # MEASURED (2026-06-10, 24 GB GPU): resident DiT (13.2 GiB) +
            # whole-frame 1024² decode peaks at 23.6 GiB → CUDA OOM. The
            # verified standalone pipeline always freed the DiT before its
            # 1024² decode; mirror that for the 128-grid only: release the
            # resident DiT, decode, and reload on the next job. 512² decode
            # fits with the DiT resident and keeps the full residency win.
            if self.hl == 128 and self.loaded:
                print("[zimage] 1024x1024 decode: releasing resident DiT to fit decode (next job reloads)")
                self.nr_blocks = List[ZImageBlockWeights]()
                self.cr_blocks = List[ZImageBlockWeights]()
                self.main_blocks = List[ZImageBlockWeights]()
                self.aux = List[ArcPointer[ZImageRealAux]]()
                self.final_lin = List[TArc]()
                self.x_pad_h = List[Float32]()
                self.cap_pad_h = List[Float32]()
                self.loaded = False
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
