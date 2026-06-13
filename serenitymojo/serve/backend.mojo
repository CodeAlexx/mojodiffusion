# serenitymojo.serve.backend — the generation-backend seam.
#
# THE CONTRACT the real Z-Image binding must implement (and the stub does):
#
#   trait GenBackend:
#     backend_name() / model_name()  — reported by GET /v1/health.
#     start(params)                  — accept ONE job; the daemon guarantees a
#                                      single in-flight job (one worker, serial
#                                      execution — matches single-GPU reality).
#     step() -> StepResult           — advance ONE unit of work (one denoise
#                                      step, or one bounded chunk of it) and
#                                      RETURN; the daemon calls it once per
#                                      event-loop tick. Pull-based progress:
#                                      every returned StepResult becomes a WS
#                                      /v1/progress event + a job-state update.
#                                      A step() call must be bounded (~100 ms
#                                      class) so HTTP stays responsive — Mojo
#                                      gives us no threads here, the worker
#                                      runs INSIDE the event loop.
#     cancel()                       — request cooperative cancellation; the
#                                      next step() must return cancelled=True
#                                      promptly (without finishing the job).
#
# Terminal StepResults (done / failed / cancelled) end the job; on done,
# `output_path` must point at the produced image and the backend must have
# written the sidecar/metadata itself (stage 1: <out_dir>/<job_id>.json with
# the full param JSON, because the PNG encoder has no tEXt support yet).

comptime BACKEND_PROTOCOL_VERSION = 1


struct LoraSpec(Copyable, Movable):
    """One LoRA overlay: name (file stem or registry key) + strength."""

    var name: String
    var weight: Float64

    def __init__(out self, name: String, weight: Float64):
        self.name = name
        self.weight = weight


struct JobParams(Copyable, Movable):
    """Everything a backend needs to run one generation job."""

    var job_id: String
    var model: String
    var prompt: String
    var negative: String
    var width: Int
    var height: Int
    var steps: Int
    var seed: Int
    var cfg: Float64
    var cfg_override: Float64
    var cfg_override_start_percent: Float64
    var cfg_override_end_percent: Float64
    var sampler: String
    var scheduler: String
    var sigma_shift: Float64
    var variation_seed: Int
    var variation_strength: Float64
    var images: Int       # requested output count from the UI request
    var image_index: Int  # 0-based output index for this backend job
    var image_count: Int  # total outputs for the original UI request
    var workflow_save_prefix: String  # Comfy SaveImage.filename_prefix metadata
    var init_image: String   # P7 img2img: init image path ("" = txt2img)
    # Comfy SetLatentNoiseMask/Inpaint: mask image path associated with a
    # latent. Backends must either execute mask semantics or reject it loudly.
    var mask_image: String
    var lanpaint_mask_channel: String
    # Comfy InpaintModelConditioning carries concat_latent_image/concat_mask on
    # conditioning, which is richer than SetLatentNoiseMask. Backends must
    # implement that conditioning path or reject it loudly.
    var inpaint_conditioning_image: String
    var inpaint_conditioning_mask: String
    var inpaint_conditioning_noise_mask: Bool
    var outpaint_left: Int
    var outpaint_top: Int
    var outpaint_right: Int
    var outpaint_bottom: Int
    var outpaint_feathering: Int
    var threshold_mask_value: Float64
    var threshold_mask_operator: String
    var lanpaint_mask_blend_overlap: Int
    var lanpaint_num_steps: Int
    var lanpaint_lambda: Float64
    var lanpaint_step_size: Float64
    var lanpaint_beta: Float64
    var lanpaint_friction: Float64
    var lanpaint_prompt_mode: String
    var lanpaint_inpainting_mode: String
    var lanpaint_add_noise: String
    var lanpaint_noise_seed: Int
    var lanpaint_start_at_step: Int
    var lanpaint_end_at_step: Int
    var lanpaint_return_with_leftover_noise: String
    var lanpaint_early_stop: Int
    var lanpaint_inner_threshold: Float64
    var lanpaint_inner_patience: Int
    # Comfy ReferenceLatent/Klein edit source image. This is not ordinary
    # img2img denoise init; Flux2/Klein consumes it through conditioning tokens.
    var reference_image: String
    var reference_latent_method: String
    var reference_latent_count: Int
    var creativity: Float64  # P7: 0..1 — fraction of the sigma schedule the
                             # denoise starts from (1.0 = pure noise/txt2img)
    var loras: List[LoraSpec]
    var params_json: String  # canonical full-request JSON (persisted verbatim)
    var out_dir: String      # where the backend writes <job_id>.png (+ sidecar)

    def __init__(out self):
        self.job_id = String("")
        self.model = String("")
        self.prompt = String("")
        self.negative = String("")
        self.width = 512
        self.height = 512
        self.steps = 20
        self.seed = 0
        self.cfg = 4.5
        self.cfg_override = -1.0
        self.cfg_override_start_percent = 0.0
        self.cfg_override_end_percent = 1.0
        self.sampler = String("")
        self.scheduler = String("")
        self.sigma_shift = 3.0
        self.variation_seed = 0
        self.variation_strength = 0.0
        self.images = 1
        self.image_index = 0
        self.image_count = 1
        self.workflow_save_prefix = String("")
        self.init_image = String("")
        self.mask_image = String("")
        self.lanpaint_mask_channel = String("")
        self.inpaint_conditioning_image = String("")
        self.inpaint_conditioning_mask = String("")
        self.inpaint_conditioning_noise_mask = False
        self.outpaint_left = -1
        self.outpaint_top = -1
        self.outpaint_right = -1
        self.outpaint_bottom = -1
        self.outpaint_feathering = -1
        self.threshold_mask_value = -1.0
        self.threshold_mask_operator = String("")
        self.lanpaint_mask_blend_overlap = -1
        self.lanpaint_num_steps = -1
        self.lanpaint_lambda = -1.0
        self.lanpaint_step_size = -1.0
        self.lanpaint_beta = -1.0
        self.lanpaint_friction = -1.0
        self.lanpaint_prompt_mode = String("")
        self.lanpaint_inpainting_mode = String("")
        self.lanpaint_add_noise = String("")
        self.lanpaint_noise_seed = -1
        self.lanpaint_start_at_step = -1
        self.lanpaint_end_at_step = -1
        self.lanpaint_return_with_leftover_noise = String("")
        self.lanpaint_early_stop = -1
        self.lanpaint_inner_threshold = -1.0
        self.lanpaint_inner_patience = -1
        self.reference_image = String("")
        self.reference_latent_method = String("")
        self.reference_latent_count = 0
        self.creativity = 0.5
        self.loras = List[LoraSpec]()
        self.params_json = String("")
        self.out_dir = String("")


def reject_unsupported_common_runtime_params(
    params: JobParams, backend_name: String
) raises:
    """Validate runtime fields shared by all image backends."""
    if params.images < 1 or params.image_count < 1:
        raise Error(
            backend_name + String(": image count must be positive")
        )
    if params.images != params.image_count:
        raise Error(
            backend_name + String(": images and image_count must match")
        )
    if params.image_index < 0 or params.image_index >= params.image_count:
        raise Error(
            backend_name + String(": image_index out of range for image_count")
        )


def reject_unsupported_reference_image_params(
    params: JobParams, backend_name: String
) raises:
    """Reject Comfy ReferenceLatent/Klein edit fields on backends that do not
    implement reference-image conditioning."""
    if params.reference_image.byte_length() > 0 or params.reference_latent_count > 0:
        raise Error(
            backend_name
            + String(": Comfy ReferenceLatent/reference image conditioning is not supported by this backend yet")
        )


def reject_unsupported_mask_image_params(
    params: JobParams, backend_name: String
) raises:
    """Reject Comfy SetLatentNoiseMask/inpaint fields on backends that do not
    implement mask-aware denoise."""
    if params.mask_image.byte_length() > 0:
        raise Error(
            backend_name
            + String(": Comfy SetLatentNoiseMask/inpaint mask conditioning is not supported by this backend yet")
        )


def reject_unsupported_inpaint_conditioning_params(
    params: JobParams, backend_name: String
) raises:
    """Reject Comfy InpaintModelConditioning fields on backends that do not
    implement concat_latent_image/concat_mask conditioning."""
    if (
        params.inpaint_conditioning_image.byte_length() > 0
        or params.inpaint_conditioning_mask.byte_length() > 0
    ):
        raise Error(
            backend_name
            + String(": Comfy InpaintModelConditioning concat conditioning is not supported by this backend yet")
        )


def has_lanpaint_runtime_params(params: JobParams) -> Bool:
    return (
        params.lanpaint_mask_blend_overlap >= 0
        or has_lanpaint_sampler_runtime_params(params)
    )


def has_lanpaint_sampler_runtime_params(params: JobParams) -> Bool:
    return (
        params.outpaint_left >= 0
        or params.outpaint_top >= 0
        or params.outpaint_right >= 0
        or params.outpaint_bottom >= 0
        or params.outpaint_feathering >= 0
        or params.threshold_mask_value >= 0.0
        or params.threshold_mask_operator.byte_length() > 0
        or params.lanpaint_num_steps >= 0
        or params.lanpaint_lambda >= 0.0
        or params.lanpaint_step_size >= 0.0
        or params.lanpaint_beta >= 0.0
        or params.lanpaint_friction >= 0.0
        or params.lanpaint_prompt_mode.byte_length() > 0
        or params.lanpaint_inpainting_mode.byte_length() > 0
        or params.lanpaint_add_noise.byte_length() > 0
        or params.lanpaint_noise_seed >= 0
        or params.lanpaint_start_at_step >= 0
        or params.lanpaint_end_at_step >= 0
        or params.lanpaint_return_with_leftover_noise.byte_length() > 0
        or params.lanpaint_early_stop >= 0
        or params.lanpaint_inner_threshold >= 0.0
        or params.lanpaint_inner_patience >= 0
    )


def reject_unsupported_lanpaint_sampler_params(
    params: JobParams, backend_name: String
) raises:
    """Reject LanPaint sampler/preprocess fields on backends that do not
    implement the mask-aware inner loop. LanPaint_MaskBlend can be handled as a
    separate final image blend by backends that opt into it."""
    if has_lanpaint_sampler_runtime_params(params):
        raise Error(
            backend_name
            + String(": LanPaint inpaint sampler semantics are not supported by this backend yet")
        )


def reject_unsupported_lanpaint_params(
    params: JobParams, backend_name: String
) raises:
    """Reject LanPaint sampler/blend fields on backends that do not implement
    the LanPaint mask-aware inner loop and final blend semantics."""
    if has_lanpaint_runtime_params(params):
        raise Error(
            backend_name
            + String(": LanPaint inpaint sampler/blend semantics are not supported by this backend yet")
        )


struct StepResult(Copyable, Movable):
    """Outcome of one backend tick. Exactly one of {progress, done, failed,
    cancelled}: terminal when done/failed/cancelled is set."""

    var step: Int          # steps completed so far (1-based after first tick)
    var total: Int         # total steps for this job
    var done: Bool         # job finished OK; output_path is valid
    var failed: Bool       # job aborted with an error; `error` is set
    var cancelled: Bool    # job ended due to cancel()
    var error: String
    var output_path: String
    var preview: String    # optional inline preview (e.g. base64 PNG); "" = none
    var phase: String      # optional sub-state for long non-denoise ticks
                           # ("loading"|"encoding"|"decoding"); "" = plain step.
                           # Rides the WS event as a `phase` key so clients can
                           # show what a slow tick is doing (F6).

    def __init__(out self):
        self.step = 0
        self.total = 0
        self.done = False
        self.failed = False
        self.cancelled = False
        self.error = String("")
        self.output_path = String("")
        self.preview = String("")
        self.phase = String("")

    def is_terminal(self) -> Bool:
        return self.done or self.failed or self.cancelled


trait GenBackend(Movable):
    """The seam between the daemon and a model runtime (stub now, Z-Image later)."""

    def backend_name(self) -> String:
        ...

    def model_name(self) -> String:
        ...

    def resident_model(self) -> String:
        """Name of the checkpoint currently resident on the device ("" = none).
        Reported by GET /v1/health and matched against the /v1/models scan
        for the per-model `loaded` flag."""
        ...

    def start(mut self, params: JobParams) raises:
        ...

    def step(mut self) raises -> StepResult:
        ...

    def cancel(mut self):
        ...

    def between_jobs_trim(mut self) raises:
        """Reclaim a finished job's transient device memory at the job boundary
        (F3 pool retention). The daemon calls this after every terminal job.
        Single-model backends with no pool-trim need (stub / pool-managed-by-the-Mojo-runtime)
        implement it as a no-op; the multi-model DispatchBackend trims the CUDA
        mempool here so idle VRAM tracks the resident footprint."""
        ...
