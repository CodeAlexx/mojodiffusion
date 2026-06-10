# serenitymojo.serve.stub_backend — stage-1 GenBackend: no model, no GPU.
#
# Simulates `steps` denoise steps (~100 ms each, slept INSIDE step() so the
# single-threaded daemon stays responsive between ticks), then renders a real
# PNG via the MOJO-libs image lib: a seed-driven gradient with a stripe
# overlay, saved to <out_dir>/<job_id>.png. The MOJO-libs PNG encoder has no
# tEXt-chunk support landed in MOJO-libs png (2026-06-10, PIL-gated):
# the job's full param JSON is embedded as a serenity.genparams.v1 tEXt chunk.

from std.time import sleep

from image.buffer import Image
from image.png import encode_png_with_text

from serenitymojo.serve.backend import GenBackend, JobParams, StepResult

comptime STEP_SLEEP_S = 0.1  # simulated per-step latency


def _render_stub_png(p: JobParams, path: String, params_json: String) raises:
    """A deterministic seed-driven gradient + stripes — a real, decodable PNG."""
    var img = Image.new(p.width, p.height, 3)
    var dw = p.width - 1 if p.width > 1 else 1
    var dh = p.height - 1 if p.height > 1 else 1
    var s = p.seed if p.seed >= 0 else -p.seed
    # P7 stub echo: an init-image job tints the gradient by creativity so the
    # img2img params are VISIBLE end-to-end (blue plane = 255*creativity).
    var img2img = p.init_image.byte_length() > 0
    var tint_b = Int(p.creativity * 255.0)
    for y in range(p.height):
        for x in range(p.width):
            var r = (x * 255) // dw
            var g = (y * 255) // dh
            var stripe = ((x + y) // 16 + s) % 3
            var b = 40 + stripe * 80 + (s % 16)
            if img2img:
                b = tint_b
            if b > 255:
                b = 255
            img.set(x, y, 0, UInt8(r & 0xFF))
            img.set(x, y, 1, UInt8(g & 0xFF))
            img.set(x, y, 2, UInt8(b & 0xFF))
    # trivial "draw" pass: a 2px white frame so the output is visibly composed
    for x in range(p.width):
        for t in range(2):
            img.set(x, t, 0, 255)
            img.set(x, t, 1, 255)
            img.set(x, t, 2, 255)
            img.set(x, p.height - 1 - t, 0, 255)
            img.set(x, p.height - 1 - t, 1, 255)
            img.set(x, p.height - 1 - t, 2, 255)
    var kws = List[String]()
    var vals = List[String]()
    kws.append(String("serenity.genparams.v1"))
    vals.append(params_json.copy())
    encode_png_with_text(img, path, kws, vals)


struct StubBackend(GenBackend, Movable):
    var active: Bool
    var cancel_flag: Bool
    var cur: Int
    var params: JobParams

    def __init__(out self):
        self.active = False
        self.cancel_flag = False
        self.cur = 0
        self.params = JobParams()

    def backend_name(self) -> String:
        return String("stub")

    def model_name(self) -> String:
        return String("-")

    def resident_model(self) -> String:
        return String("")  # the stub never holds device weights

    def start(mut self, params: JobParams) raises:
        if self.active:
            raise Error("StubBackend.start: a job is already running")
        self.params = params.copy()
        self.active = True
        self.cancel_flag = False
        self.cur = 0

    def cancel(mut self):
        self.cancel_flag = True

    def step(mut self) raises -> StepResult:
        var r = StepResult()
        r.total = self.params.steps
        if not self.active:
            r.failed = True
            r.error = String("no active job")
            return r^
        if self.cancel_flag:
            self.active = False
            r.step = self.cur
            r.cancelled = True
            return r^
        sleep(STEP_SLEEP_S)
        self.cur += 1
        r.step = self.cur
        # preview plumbing probe: exactly ONE mid-job step carries a dummy
        # preview payload, so the WS event JSON's optional `preview` field can
        # be gated end-to-end without a real latent->RGB path yet.
        if self.cur * 2 == self.params.steps:
            r.preview = String("stub-preview-step-") + String(self.cur)
        if self.cur < self.params.steps:
            return r^
        # final step: produce the output image; gen params ride a PNG tEXt
        # chunk (serenity.genparams.v1) — MOJO-libs png now supports tEXt
        # (gated vs PIL 2026-06-10), so the sidecar JSON is gone.
        var png_path = self.params.out_dir + "/" + self.params.job_id + ".png"
        try:
            _render_stub_png(self.params, png_path, self.params.params_json)
        except e:
            self.active = False
            r.failed = True
            r.error = String("output write failed: ") + String(e)
            return r^
        self.active = False
        r.done = True
        r.output_path = png_path
        return r^
