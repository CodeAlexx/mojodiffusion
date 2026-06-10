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
    var init_image: String   # P7 img2img: init image path ("" = txt2img)
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
        self.init_image = String("")
        self.creativity = 0.5
        self.loras = List[LoraSpec]()
        self.params_json = String("")
        self.out_dir = String("")


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
