# serenitymojo.serve.krea2_backend — GenBackend for Krea-2 (krea2) txt2img.
#
# Wraps the VERIFIED krea2 inference sampler stack (models/krea2/krea2_infer.mojo,
# hoisted from the trainer's proven inline sampler) behind the daemon/worker
# GenBackend seam, exactly like ZImageBackend. One resident worker per GPU, spawned
# lazily by serenity-server as `serenity_worker_krea2 <fd>`.
#
# V1 VRAM design (single 24 GB card): the Qwen3-VL-4B TE (~9.6 GB) and the ~12 GB
# fp8-resident DiT do NOT coexist, so each job runs the PROVEN two-process sequence:
#   1) fork+execv a fresh `serenity_worker_krea2 encode-child` → writes the POS/NEG
#      context .bins, EXITS (encoder VRAM reclaimed by the OS).
#   2) build the fp8-resident base (~12 GB) + conditioning weights + final layer.
#   3) fixed-LTMAX length-bucket denoise (real_len flash-padmask) + LoRA overlay + CFG.
#   4) VAE decode → <out_dir>/<job_id>.png.
# resident_model() is "" (job-scoped residency). Cross-job persistent residency is a
# follow-up (only if a measured TE device peak fits alongside the 12 GB base).
#
# step() runs the whole job in ONE blocking tick (the driver has no per-step
# watchdog — only a 15 s Ready handshake, sent at worker-loop start before any load).
# cancel() is honored between the major phases.

from std.gpu.host import DeviceContext
from std.collections import Optional

from serenitymojo.serve.backend import (
    GenBackend, JobParams, StepResult, reject_unsupported_common_runtime_params,
)
from serenitymojo.serve.model_scan import LORAS_DIR
from serenitymojo.io.env import env_int
from serenitymojo.io.ffi import sys_open, sys_close, O_RDONLY
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current

from serenitymojo.serve.krea2_encode_subprocess import (
    krea2_encode_contexts_subprocess,
)
from serenitymojo.models.krea2.krea2_infer import (
    Krea2ResidentCond, load_krea2_resident_cond,
    Krea2InlineCond, inline_cond_from_bin,
    load_krea2_stack_lora, empty_krea2_stack_lora,
    krea2_sample_latent, krea2_decode_latent_to_png,
)
from serenitymojo.models.krea2.krea2_stack import (
    Krea2StackLora, Krea2StreamFinal,
    Krea2ResidentFp8, build_krea2_resident_fp8,
)
from serenitymojo.models.krea2.krea2_buckets import (
    KREA2_LADDER_LEN, KREA2_LADDER_X100, krea2_lat_h, krea2_lat_w,
)
from serenitymojo.pipeline.krea2_paths import (
    KREA2_RAW, KREA2_TURBO, KREA2_VAE_DIR, KREA2_RAW_KEY_PREFIX,
)

# ── krea2 FlowEdit worker mode (Phase C1 part 2): reuse the VERIFIED pipeline
#    helpers (pipeline/krea2_flowedit.mojo, the 512² 18s/edit vertical) behind
#    the GenBackend seam. edit_src_image != "" routes step() to _step_flowedit.
from serenitymojo.pipeline.krea2_flowedit import (
    HEIGHT as FE_HEIGHT, WIDTH as FE_WIDTH, LH as FE_LH, LW as FE_LW,
    NTOK as FE_NTOK, NBLOCKS as FE_NBLOCKS, LT_SHARED as FE_LT_SHARED,
    _build_pos, _load_context_padded, _velocity,
    _accum_saliency, _mask_from_saliency, _blend_outside_mask, _save_mask_png,
)
from serenitymojo.models.dit.krea2_dit import (
    Krea2ResidentInt8, Krea2HostInt8Inf, Krea2SharedResident,
    build_krea2_shared_resident, build_krea2_host_int8_inf,
)
from serenitymojo.models.krea2.krea2_stack import build_krea2_resident_int8
from serenitymojo.models.krea2.krea2_int8_cache import (
    krea2_int8_cache_path, krea2_int8_cache_valid,
    load_krea2_int8_cache_resident, load_krea2_int8_cache_host,
    load_krea2_int8_cache_shared, save_krea2_int8_cache,
)
from serenitymojo.models.krea2.krea2_prepare_cache import (
    _mean_ch, _std_ch, _normalize_latent, KREA2_VAE_ENC_FILE,
)
from serenitymojo.models.vae.qwenimage_encoder import QwenImageVaeEncoder
from serenitymojo.models.vae.qwenimage_decoder import QwenImageVaeDecoder
from serenitymojo.sampling.krea2_sampler import krea2_timesteps, krea2_packed_seq_len
from serenitymojo.serve.image_io import decode_image_any, image_to_signed_nchw
from image.transform import resize_bilinear
from serenitymojo.ops.tensor_algebra import mul_scalar, add, sub
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.torch_bf16 import torch_f32_to_bf16_rne
from serenitymojo.ops.random import randn
from serenitymojo.io.dtype import STDtype
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.tensor import Tensor

# ── compiled config: 1024px-area, seven-arm trainer-parity aspect ladder. ─────
# The trainer/cache producer already owns and gates this exact SimpleTuner ladder.
# Reuse it here so inference, training, and staged cache geometry cannot drift.
comptime KREA2_EDGE_UNITS = 16
# Square uses the proven 768 text bucket. Non-square ladder arms carry 4032 or
# 4160 image tokens, so their text bucket is padded by the minimum extra tail
# needed to keep the joint flash-attention buffer 128-aligned. Real text length
# is unchanged and the existing padmask excludes every added tail token.
comptime KREA2_LTMAX_BASE = 768
comptime KREA2_NBLOCKS = 28


def _krea2_t2i_shape_supported(width: Int, height: Int) -> Bool:
    comptime for bi in range(KREA2_LADDER_LEN):
        comptime X100_BI = KREA2_LADDER_X100[bi]
        comptime LH_BI = krea2_lat_h(X100_BI, KREA2_EDGE_UNITS)
        comptime LW_BI = krea2_lat_w(X100_BI, KREA2_EDGE_UNITS)
        if width == LW_BI * 8 and height == LH_BI * 8:
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


def _resolve_krea2_lora_path(name: String) raises -> String:
    # Bare names resolve against the scanner's LoRA dir; absolute/relative paths
    # (developer / imported use) are accepted as-is.
    if _path_exists(name):
        return name.copy()
    if _path_exists(name + ".safetensors"):
        return name + ".safetensors"
    if _path_exists(String(LORAS_DIR) + "/" + name):
        return String(LORAS_DIR) + "/" + name
    if _path_exists(String(LORAS_DIR) + "/" + name + ".safetensors"):
        return String(LORAS_DIR) + "/" + name + ".safetensors"
    raise Error(
        String("krea2: LoRA file not found: ") + name
        + " (tried as a path and under " + LORAS_DIR + ")"
    )


struct Krea2Backend(GenBackend, Movable):
    var ctx: DeviceContext
    var active: Bool
    var cancel_flag: Bool
    var params: JobParams
    var lora_path: String        # resolved file ("" = base / zero overlay)
    var lora_mult: Float32
    var progress_fd: Int32

    def __init__(out self) raises:
        self.ctx = DeviceContext()
        self.active = False
        self.cancel_flag = False
        self.params = JobParams()
        self.lora_path = String("")
        self.lora_mult = Float32(1.0)
        self.progress_fd = Int32(-1)

    def set_progress_fd(mut self, fd: Int32):
        self.progress_fd = fd

    def backend_name(self) -> String:
        return String("krea2")

    def model_name(self) -> String:
        return String("Krea-2")

    def resident_model(self) -> String:
        # Job-scoped residency (v1 rebuilds the fp8 base per job); nothing persists
        # between jobs, so report no resident checkpoint.
        return String("")

    def start(mut self, params: JobParams) raises:
        if self.active:
            raise Error("Krea2Backend.start: a job is already running")
        reject_unsupported_common_runtime_params(params, String("krea2"))
        # T2I dispatches over the existing trainer/cache-producer 1024px-area
        # aspect ladder. FlowEdit remains its separately compiled 512x512 path.
        if params.edit_src_image == String(""):
            if not _krea2_t2i_shape_supported(params.width, params.height):
                raise Error(
                    String("krea2: unsupported size ") + String(params.width) + "x"
                    + String(params.height)
                    + " — choose a compiled Krea-2 1024px-area aspect bucket."
                )
        if params.steps < 1:
            raise Error("krea2: steps must be >= 1")
        # Resolve at most one LoRA overlay up front (fail before expensive work).
        var lora_path = String("")
        var lora_mult = Float32(1.0)
        if len(params.loras) > 0:
            lora_path = _resolve_krea2_lora_path(params.loras[0].name)
            lora_mult = Float32(params.loras[0].weight)
        self.params = params.copy()
        self.lora_path = lora_path^
        self.lora_mult = lora_mult
        self.active = True
        self.cancel_flag = False

    def cancel(mut self):
        self.cancel_flag = True

    def _cancelled(mut self, mut r: StepResult) -> Bool:
        if self.cancel_flag:
            self.active = False
            r.step = 0
            r.cancelled = True
            return True
        return False

    def _step_t2i_shape[LH_: Int, LW_: Int, LTMAX_: Int, LFULL_: Int](
        mut self, pos_bin: String, neg_bin: String,
    ) raises -> StepResult:
        """Run the existing inference path at serenity trainer-parity bucket arm."""
        var r = StepResult()
        r.total = self.params.steps

        # 2) load the two contexts (small H2D; LT derived from tensor shape).
        var cond = inline_cond_from_bin[LH_, LW_, LTMAX_](pos_bin, self.ctx)
        var uncond = inline_cond_from_bin[LH_, LW_, LTMAX_](neg_bin, self.ctx)

        # 3) build the fp8-resident base + conditioning weights + final layer.
        var turbo = self.params.model.lower().find("turbo") >= 0
        var checkpoint = String(KREA2_TURBO) if turbo else String(KREA2_RAW)
        print("[krea2] model=", "Krea-2-Turbo" if turbo else "Krea-2-Raw",
              " checkpoint=", checkpoint)
        var st = ShardedSafeTensors.open(checkpoint)
        var cond_w = load_krea2_resident_cond(st, String(KREA2_RAW_KEY_PREFIX), self.ctx)
        var fin = Krea2StreamFinal.load(st, String(KREA2_RAW_KEY_PREFIX), self.ctx)
        # 16GB fit: PARTIAL fp8 residency. The full 28-block fp8 base (~12GB) +
        # 1024-area activations OOMs on a 16GB card, so keep only the FIRST K
        # blocks resident and stream the remainder through the same block math.
        var res_k = env_int(String("KREA2_FP8_RESIDENT_BLOCKS"), 14)
        if res_k < 0:
            res_k = 0
        if res_k > KREA2_NBLOCKS:
            res_k = KREA2_NBLOCKS
        print("[krea2] building fp8-resident base (", res_k, "of", KREA2_NBLOCKS,
              "blocks resident, rest streamed per step) ...")
        var resident = Optional[Krea2ResidentFp8](
            build_krea2_resident_fp8(
                st, String(KREA2_RAW_KEY_PREFIX), KREA2_NBLOCKS, res_k, self.ctx,
            )
        )
        if self._cancelled(r):
            return r^

        # 4) LoRA overlay (ADDED, never fused) or base-only.
        var lora: Krea2StackLora
        if self.lora_path != String(""):
            lora = load_krea2_stack_lora(self.lora_path, self.lora_mult, self.ctx)
        else:
            lora = empty_krea2_stack_lora()

        # 5) fixed-LTMAX length-bucket denoise + creator CFG semantics -> latent.
        # Krea's public guidance value is already cond + cfg*(cond-uncond), and
        # cfg=0 disables the unconditional branch. Do not translate it into the
        # unrelated textbook/Comfy convention at this model boundary.
        var krea_guidance = Float32(self.params.cfg)
        print("[krea2] creator CFG=", self.params.cfg,
              " fixed_mu_1_15=", turbo)
        var latent = krea2_sample_latent[LH_, LW_, LTMAX_, LFULL_](
            st, String(KREA2_RAW_KEY_PREFIX), cond_w, fin, lora, cond, uncond,
            self.params.steps, krea_guidance, UInt64(self.params.seed),
            resident, self.ctx,
            use_fixed_mu_1_15=turbo,
            progress_fd=self.progress_fd,
        )
        if self._cancelled(r):
            return r^

        # 6) VAE decode -> <out_dir>/<job_id>.png.
        var png_path = self.params.out_dir + "/" + self.params.job_id + ".png"
        krea2_decode_latent_to_png[LH_, LW_](
            latent, String(KREA2_VAE_DIR), png_path, self.ctx,
        )

        self.active = False
        r.step = self.params.steps
        r.done = True
        r.output_path = png_path
        return r^

    def step(mut self) raises -> StepResult:
        var r = StepResult()
        r.total = self.params.steps
        if not self.active:
            r.failed = True
            r.error = String("no active job")
            return r^
        if self._cancelled(r):
            return r^
        try:
            if self.params.edit_src_image != String(""):
                return self._step_flowedit()
            var jid = self.params.job_id
            var pos_bin = String("/tmp/serenity_krea2_ctx_") + jid + String(".pos.bin")
            var neg_bin = String("/tmp/serenity_krea2_ctx_") + jid + String(".neg.bin")

            # 1) ENCODE (fork+execv child TE → context bins, VRAM reclaimed on exit).
            krea2_encode_contexts_subprocess(
                self.params.prompt, self.params.negative, pos_bin, neg_bin, self.ctx,
            )
            if self._cancelled(r):
                return r^

            # Runtime request -> existing COMPTIME trainer/cache bucket arm.
            comptime for bi in range(KREA2_LADDER_LEN):
                comptime X100_BI = KREA2_LADDER_X100[bi]
                comptime LH_BI = krea2_lat_h(X100_BI, KREA2_EDGE_UNITS)
                comptime LW_BI = krea2_lat_w(X100_BI, KREA2_EDGE_UNITS)
                comptime IMGLEN_BI = (LH_BI // 2) * (LW_BI // 2)
                comptime LUNALIGNED_BI = KREA2_LTMAX_BASE + IMGLEN_BI
                comptime LTMAX_BI = (
                    KREA2_LTMAX_BASE + (128 - (LUNALIGNED_BI % 128)) % 128
                )
                comptime LFULL_BI = (
                    LTMAX_BI + IMGLEN_BI
                )
                if (
                    self.params.width == LW_BI * 8
                    and self.params.height == LH_BI * 8
                ):
                    return self._step_t2i_shape[LH_BI, LW_BI, LTMAX_BI, LFULL_BI](
                        pos_bin, neg_bin,
                    )
            raise Error("krea2: admitted aspect bucket dispatch was not compiled")
        except e:
            self.active = False
            r.failed = True
            r.error = String(e)
            return r^

    def _step_flowedit(mut self) raises -> StepResult:
        """FlowEdit worker mode (Phase C1): training-free instruction edit at the
        pipeline's verified 512² geometry. Ports pipeline/krea2_flowedit.main:
        encode 4 contexts in-process -> stage+VAE-encode the source PNG ->
        int8 W8A8 base (sidecar cache) -> FlowEdit ODE (4 forwards/step,
        optional auto-mask) -> decode + save. Fail-loud on unsupported geometry
        and over-length prompts (LT_SHARED), never silent."""
        var r = StepResult()
        r.total = self.params.steps
        var jid = self.params.job_id
        if self.params.width != FE_WIDTH or self.params.height != FE_HEIGHT:
            raise Error(
                String("krea2 FlowEdit: only ") + String(FE_WIDTH) + "x"
                + String(FE_HEIGHT) + " is compiled in this worker (requested "
                + String(self.params.width) + "x" + String(self.params.height)
                + "); resubmit at 512x512 or rebuild the edit geometry"
            )
        if len(self.params.loras) > 0:
            raise Error("krea2 FlowEdit: LoRA is not admitted in the edit mode (training-free path)")

        # ── 1) FOUR contexts via the in-process TE (two encode passes). ──
        var sp_bin = String("/tmp/serenity_krea2_edit_") + jid + String(".srcpos.bin")
        var sn_bin = String("/tmp/serenity_krea2_edit_") + jid + String(".srcneg.bin")
        var tp_bin = String("/tmp/serenity_krea2_edit_") + jid + String(".tgtpos.bin")
        var tn_bin = String("/tmp/serenity_krea2_edit_") + jid + String(".tgtneg.bin")
        krea2_encode_contexts_subprocess(
            self.params.edit_src_prompt, self.params.edit_src_negative,
            sp_bin, sn_bin, self.ctx,
        )
        if self._cancelled(r):
            return r^
        krea2_encode_contexts_subprocess(
            self.params.prompt, self.params.negative, tp_bin, tn_bin, self.ctx,
        )
        if self._cancelled(r):
            return r^
        # _load_context_padded pads to LT_SHARED and FAIL-LOUDS on over-length
        # prompts ("shorten the prompt") — the C-phase node contract.
        var src_pos_pair = _load_context_padded(sp_bin, String("SRC_POS"), self.ctx)
        var src_neg_pair = _load_context_padded(sn_bin, String("SRC_NEG"), self.ctx)
        var tgt_pos_pair = _load_context_padded(tp_bin, String("TGT_POS"), self.ctx)
        var tgt_neg_pair = _load_context_padded(tn_bin, String("TGT_NEG"), self.ctx)
        var ctx_src_pos = src_pos_pair[0].clone(self.ctx)
        var lt_src_pos = src_pos_pair[1]
        var ctx_src_neg = src_neg_pair[0].clone(self.ctx)
        var lt_src_neg = src_neg_pair[1]
        var ctx_tgt_pos = tgt_pos_pair[0].clone(self.ctx)
        var lt_tgt_pos = tgt_pos_pair[1]
        var ctx_tgt_neg = tgt_neg_pair[0].clone(self.ctx)
        var lt_tgt_neg = tgt_neg_pair[1]
        var pos_grid = _build_pos(self.ctx)

        # ── 2) source PNG -> staged 512² signed NCHW -> normalized Z0_src. ──
        var img = decode_image_any(self.params.edit_src_image)
        var resized = resize_bilinear(img, FE_WIDTH, FE_HEIGHT)
        var host = image_to_signed_nchw(resized)
        var image_t = Tensor.from_host(host, [1, 3, FE_HEIGHT, FE_WIDTH], STDtype.F32, self.ctx)
        var img_bf16 = cast_tensor(image_t, STDtype.BF16, self.ctx)
        var enc = QwenImageVaeEncoder[FE_HEIGHT, FE_WIDTH].load(KREA2_VAE_ENC_FILE, self.ctx)
        var lat_mean = enc.encode_mean(img_bf16, self.ctx)
        var lat_f32 = cast_tensor(lat_mean, STDtype.F32, self.ctx)
        var mean_ch = _mean_ch(self.ctx)
        var std_ch = _std_ch(self.ctx)
        var z0_src = _normalize_latent(lat_f32, mean_ch, std_ch, self.ctx)
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        if self._cancelled(r):
            return r^

        # ── 3) int8 W8A8 base, sidecar-cache-first (the fast startup path). ──
        var st = ShardedSafeTensors.open(String(KREA2_RAW))
        var i8_res_blocks = env_int(String("KREA2_EDIT_I8_RESIDENT_BLOCKS"), 20)
        if i8_res_blocks < 0:
            i8_res_blocks = 0
        if i8_res_blocks > FE_NBLOCKS:
            i8_res_blocks = FE_NBLOCKS
        var resident_i8 = Optional[Krea2ResidentInt8](None)
        var host_i8 = Optional[Krea2HostInt8Inf](None)
        var shared = Optional[Krea2SharedResident](None)
        var i8_cache = krea2_int8_cache_path(String(KREA2_RAW))
        if krea2_int8_cache_valid(i8_cache, String(KREA2_RAW), FE_NBLOCKS):
            print("[krea2-edit] int8 sidecar found:", i8_cache)
            if i8_res_blocks > 0:
                resident_i8 = Optional[Krea2ResidentInt8](
                    load_krea2_int8_cache_resident(i8_cache, i8_res_blocks, self.ctx)
                )
            if i8_res_blocks < FE_NBLOCKS:
                host_i8 = Optional[Krea2HostInt8Inf](
                    load_krea2_int8_cache_host(i8_cache, FE_NBLOCKS, i8_res_blocks, self.ctx)
                )
            shared = Optional[Krea2SharedResident](
                load_krea2_int8_cache_shared(i8_cache, self.ctx)
            )
        else:
            print("[krea2-edit] building int8 W8A8 base: resident", i8_res_blocks,
                  "/", FE_NBLOCKS, "blocks ...")
            if i8_res_blocks > 0:
                resident_i8 = Optional[Krea2ResidentInt8](
                    build_krea2_resident_int8(
                        st, KREA2_RAW_KEY_PREFIX, FE_NBLOCKS, i8_res_blocks, self.ctx
                    )
                )
            if i8_res_blocks < FE_NBLOCKS:
                host_i8 = Optional[Krea2HostInt8Inf](
                    build_krea2_host_int8_inf(
                        st, KREA2_RAW_KEY_PREFIX, FE_NBLOCKS, i8_res_blocks, self.ctx
                    )
                )
            shared = Optional[Krea2SharedResident](
                build_krea2_shared_resident(st, KREA2_RAW_KEY_PREFIX, self.ctx)
            )
            try:
                save_krea2_int8_cache(
                    resident_i8, host_i8, shared.value(), String(KREA2_RAW),
                    i8_cache, FE_NBLOCKS, self.ctx,
                )
            except e:
                print("[krea2-edit] WARN could not write int8 sidecar:", String(e))
        if self._cancelled(r):
            return r^

        # ── 4) FlowEdit ODE (Euler on the velocity DIFFERENCE). ──
        var steps = self.params.steps
        var n_max = self.params.edit_nmax
        if n_max < 0:
            n_max = 24
        if n_max > steps:
            n_max = steps
        var n_min = self.params.edit_nmin
        if n_min < 0:
            n_min = 0
        var src_cfg = Float32(self.params.edit_src_cfg)
        if self.params.edit_src_cfg < 0.0:
            src_cfg = Float32(1.5)
        var tgt_cfg = Float32(self.params.cfg)
        var seed = UInt64(self.params.seed)
        var auto_mask = self.params.edit_auto_mask
        var mask_q = Float32(self.params.edit_mask_q)
        var mask_dilate = self.params.edit_mask_dilate
        var mask_warmup = self.params.edit_mask_warmup
        var seq = krea2_packed_seq_len(FE_HEIGHT, FE_WIDTH)
        var ts = krea2_timesteps(seq, steps)
        var skip_before = steps - n_max
        var stop_at = steps - n_min
        print("[krea2-edit] FlowEdit", FE_HEIGHT, "x", FE_WIDTH, " steps=", steps,
              " window=[", skip_before, ",", stop_at, ") src_cfg=", src_cfg,
              " tgt_cfg=", tgt_cfg, " seed=", seed, " auto_mask=", auto_mask)

        var z_edit = z0_src.clone(self.ctx)
        var z0_host = z0_src.to_host(self.ctx)
        var sal = List[Float32]()
        for _ in range(FE_NTOK):
            sal.append(Float32(0.0))
        var active_count = 0
        for si in range(steps):
            if si < skip_before or si >= stop_at:
                continue
            if self._cancelled(r):
                return r^
            var t_cur = ts[si]
            var t_prev = ts[si + 1]
            var t_t = Tensor.from_host([t_cur], [1], STDtype.F32, self.ctx)
            var noise = randn([1, 16, FE_LH, FE_LW], seed + UInt64(si), STDtype.F32, self.ctx)
            var zt_src = add(
                mul_scalar(z0_src, Float32(1.0) - t_cur, self.ctx),
                mul_scalar(noise, t_cur, self.ctx),
                self.ctx,
            )
            var zt_tgt = add(z_edit, sub(zt_src, z0_src, self.ctx), self.ctx)
            var v_src = _velocity(
                st, zt_src, ctx_src_pos, lt_src_pos, ctx_src_neg, lt_src_neg,
                pos_grid, t_t, src_cfg, resident_i8, host_i8, shared, self.ctx,
            )
            var v_tgt = _velocity(
                st, zt_tgt, ctx_tgt_pos, lt_tgt_pos, ctx_tgt_neg, lt_tgt_neg,
                pos_grid, t_t, tgt_cfg, resident_i8, host_i8, shared, self.ctx,
            )
            var dv = sub(v_tgt, v_src, self.ctx)
            z_edit = add(z_edit, mul_scalar(dv, t_prev - t_cur, self.ctx), self.ctx)
            if auto_mask:
                active_count += 1
                var dv_host = dv.to_host(self.ctx)
                _accum_saliency(dv_host, sal)
                if active_count > mask_warmup:
                    var mask = _mask_from_saliency(sal, mask_q, mask_dilate)
                    var z_host = z_edit.to_host(self.ctx)
                    _blend_outside_mask(z_host, z0_host, mask)
                    z_edit = Tensor.from_host(
                        z_host^, [1, 16, FE_LH, FE_LW], STDtype.F32, self.ctx
                    )
            r.step = si + 1
            print("[krea2-edit] step", si, "/", steps, " t=", t_cur)

        # ── 5) decode + save (+ mask debug artifact). ──
        var png_path = self.params.out_dir + "/" + jid + ".png"
        if auto_mask and active_count > 0:
            var final_mask = _mask_from_saliency(sal, mask_q, mask_dilate)
            var mask_path = _save_mask_png(final_mask, png_path, self.ctx)
            print("[krea2-edit] mask artifact:", mask_path)
        var dec = QwenImageVaeDecoder[FE_LH, FE_LW].load(String(KREA2_VAE_DIR), self.ctx)
        var latent_bf16 = torch_f32_to_bf16_rne(z_edit, self.ctx)
        var image = dec.decode(latent_bf16, self.ctx)
        save_png(image, png_path, self.ctx, ValueRange.SIGNED)
        print("[krea2-edit] wrote", png_path)

        self.active = False
        r.step = self.params.steps
        r.done = True
        r.output_path = png_path
        return r^

    def between_jobs_trim(mut self) raises:
        # Job-scoped residency: everything allocated in step() drops at its end;
        # trim the pool back to the OS between jobs.
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        self.ctx.synchronize()
