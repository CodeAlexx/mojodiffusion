# serenitymojo.serve.klein_backend -- Flux2/Klein daemon contract adapter.
#
# This backend intentionally runs Klein through the existing process-separated
# staged path:
#   1. write a per-job serenity.sample_prompts.v1 request,
#   2. run Qwen3 cap-cache precache in its own process,
#   3. run serenitymojo/sampling/klein_sample_cli in its own process,
#   4. rewrap the produced PNG with daemon genparams metadata.
#
# That keeps Qwen3 text encoder memory and the Klein image model out of the
# same process, matching the proven training/sample cadence route.

from std.memory import alloc

from image.png import encode_png_with_text
from serenitymojo.io.cap_cache import validate_klein_cap_cache_header
from serenitymojo.io.ffi import (
    BytePtr, sys_open, sys_close, sys_pwrite,
    O_RDONLY, O_WRONLY, O_CREAT, O_TRUNC,
)
from serenitymojo.serve.backend import (
    GenBackend, JobParams, StepResult, reject_unsupported_common_runtime_params,
    reject_unsupported_mask_image_params,
    reject_unsupported_inpaint_conditioning_params,
    reject_unsupported_qwen_edit_conditioning_params,
    reject_unsupported_conditioning_mask_params,
    reject_unsupported_lanpaint_params,
)
from serenitymojo.serve.external_command import ExternalCommand
from serenitymojo.serve.image_io import decode_image_any
from serenitymojo.sampling.klein_reference_latent_bridge import (
    plan_klein_reference_latent_bridge,
)
from serenitymojo.serve.model_scan import LORAS_DIR


comptime DEFAULT_OUT_DIR = "/home/alex/mojodiffusion/output/serenity_daemon"
comptime KLEIN9B_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/klein9b.json"
comptime KLEIN4B_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/klein4b.json"
comptime KLEIN_PRECACHE_BIN = "/home/alex/mojodiffusion/output/bin/klein_precache_sample_prompts"
comptime KLEIN_SAMPLER_BIN = "/home/alex/mojodiffusion/output/bin/klein_sample_cli"
comptime KLEIN_PRECACHE_SRC = "serenitymojo/pipeline/klein9b_precache_sample_prompts.mojo"
comptime KLEIN_SAMPLER_SRC = "serenitymojo/sampling/klein_sample_cli.mojo"
comptime KLEIN_REFERENCE_EDIT_SHIFT = "2.02"
comptime KLEIN_REFERENCE_T_OFFSET = "10.0"

comptime KPHASE_IDLE = 0
comptime KPHASE_PREPARE = 1
comptime KPHASE_ENCODE = 2
comptime KPHASE_SAMPLE = 3


def _lower(s: String) -> String:
    return String(s.lower())


def _byte_string(c: UInt8) raises -> String:
    var b = List[UInt8]()
    b.append(c)
    return String(from_utf8=b)


def _json_escape(s: String) raises -> String:
    var out = String("")
    var bs = s.as_bytes()
    for i in range(s.byte_length()):
        var ch = bs[i]
        if ch == 0x22:
            out += String("\\\"")
        elif ch == 0x5C:
            out += String("\\\\")
        elif ch == 0x0A:
            out += String("\\n")
        elif ch == 0x0D:
            out += String("\\r")
        elif ch == 0x09:
            out += String("\\t")
        else:
            out += _byte_string(ch)
    return out^


def _shell_quote(s: String) raises -> String:
    var out = String("'")
    var bs = s.as_bytes()
    for i in range(s.byte_length()):
        var ch = bs[i]
        if ch == 0x27:
            out += String("'\\''")
        else:
            out += _byte_string(ch)
    out += String("'")
    return out^


def _path_exists(path: String) -> Bool:
    if path == String(""):
        return False
    var fd = sys_open(path, O_RDONLY, 0)
    if fd < 0:
        return False
    _ = sys_close(fd)
    return True


def _write_text_file(path: String, content: String) raises:
    var fd = sys_open(path, O_CREAT | O_WRONLY | O_TRUNC, Int32(0o644))
    if fd < 0:
        raise Error(String("klein_backend: cannot create ") + path)
    var n = content.byte_length()
    var buf = alloc[UInt8](n if n > 0 else 1)
    var src = content.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    var wrote = sys_pwrite(fd, BytePtr(unsafe_from_address=Int(buf)), n, 0)
    buf.free()
    _ = sys_close(fd)
    if wrote != n:
        raise Error(String("klein_backend: short write to ") + path)


def _run_quick_command(label: String, cmd: String) raises:
    var sidecar = ExternalCommand()
    sidecar.start(label, cmd)
    var spins = 0
    while spins < 1000000:
        if sidecar.poll():
            sidecar.require_success()
            return
        spins += 1
    sidecar.kill()
    raise Error(String("klein: quick command timed out label=") + label)


def _mkdir_p(path: String) raises:
    if path == String(""):
        return
    _run_quick_command(String("mkdir job dir"), String("mkdir -p ") + _shell_quote(path))


def _build_command_hint() -> String:
    return (
        String("build required binaries with: pixi run build-klein-precache && pixi run build-klein-sampler")
    )


def _require_file(label: String, path: String) raises:
    if not _path_exists(path):
        raise Error(
            String("klein: missing ") + label + String(": ") + path
            + String("; ") + _build_command_hint()
        )


def _resolve_klein_lora_path(name: String) raises -> String:
    if name == String(""):
        raise Error("klein: empty LoRA name")
    if _path_exists(name):
        return name.copy()
    if _path_exists(name + ".safetensors"):
        return name + ".safetensors"
    if _path_exists(String(LORAS_DIR) + String("/") + name):
        return String(LORAS_DIR) + String("/") + name
    if _path_exists(String(LORAS_DIR) + String("/") + name + String(".safetensors")):
        return String(LORAS_DIR) + String("/") + name + String(".safetensors")
    raise Error(
        String("klein: LoRA file not found: ") + name
        + String(" (tried as a path and under ") + String(LORAS_DIR) + String(")")
    )


def _klein_lora_path(params: JobParams) raises -> String:
    if len(params.loras) == 0:
        return String("")
    if len(params.loras) > 1:
        raise Error(
            "klein: staged daemon backend supports exactly one LoRA until the sampler path exposes stack semantics"
        )
    if params.loras[0].weight != 1.0:
        raise Error(
            "klein: LoRA weight other than 1.0 is not wired through the staged sampler CLI yet"
        )
    return _resolve_klein_lora_path(params.loras[0].name)


def _model_variant(model: String) -> String:
    var m = _lower(model)
    if m.find("4b") >= 0:
        return String("4b")
    return String("9b")


def _config_for_variant(variant: String) -> String:
    if variant == String("4b"):
        return String(KLEIN4B_CONFIG)
    return String(KLEIN9B_CONFIG)


def _joint_dim_for_variant(variant: String) -> Int:
    if variant == String("4b"):
        return 7680
    return 12288


def _validate_klein_request(params: JobParams) raises:
    reject_unsupported_common_runtime_params(params, String("klein"))
    reject_unsupported_inpaint_conditioning_params(params, String("klein"))
    reject_unsupported_qwen_edit_conditioning_params(params, String("klein"))
    reject_unsupported_conditioning_mask_params(params, String("klein"))
    reject_unsupported_mask_image_params(params, String("klein"))
    reject_unsupported_lanpaint_params(params, String("klein"))
    var model = _lower(params.model)
    if model.find("flux2-dev") >= 0 or model.find("flux-2-dev") >= 0 or model.find("flux2_dev") >= 0:
        raise Error(
            String("klein: Flux2-dev model '") + params.model
            + "' is not a Klein model and must not route through the Klein runner"
        )
    if not (
        (params.width == 512 and params.height == 512)
        or (params.width == 1024 and params.height == 1024)
    ):
        raise Error(
            String("klein: unsupported size ") + String(params.width)
            + "x" + String(params.height)
            + " (existing staged sampler supports square 512 or 1024)"
        )
    if params.steps <= 0:
        raise Error("klein: steps must be positive")
    if params.seed < 0:
        raise Error("klein: seed must be nonnegative for staged sample prompts")
    var sampler = _lower(params.sampler)
    if not (sampler == "euler" or sampler == "flowmatch_euler"):
        raise Error(
            String("klein: unsupported sampler '") + params.sampler
            + "' (existing Flux2/Klein staged sampler is Euler-only)"
        )
    var scheduler = _lower(params.scheduler)
    if not (scheduler == "flux2" or scheduler == "simple"):
        raise Error(
            String("klein: unsupported scheduler '") + params.scheduler
            + "' (accepted imported metadata: flux2 or simple)"
        )
    if params.variation_strength > 0.0:
        raise Error("klein: variation noise is not wired for the daemon backend")
    if params.cfg <= 0.0:
        raise Error("klein: cfg must be positive")
    var has_reference = params.reference_image.byte_length() > 0 or params.reference_latent_count > 0
    if has_reference:
        _ = plan_klein_reference_latent_bridge(params)
    elif params.init_image.byte_length() > 0:
        raise Error(
            "klein: init_image/img2img is not wired for the staged daemon backend yet; use the ReferenceLatent bridge work item, not ordinary img2img"
        )
    _ = _klein_lora_path(params)


def _sample_prompts_json(
    params: JobParams, caps_pos: String, caps_neg: String
) raises -> String:
    var out = String("{\n")
    out += String('  "schema":"serenity.sample_prompts.v1",\n')
    out += String('  "defaults":{\n')
    out += String('    "precache_required":true,\n')
    out += String('    "enforce_min_image_size":false,\n')
    out += String('    "width":') + String(params.width) + String(",\n")
    out += String('    "height":') + String(params.height) + String(",\n")
    out += String('    "frames":1,\n')
    out += String('    "steps":') + String(params.steps) + String(",\n")
    out += String('    "cfg":') + String(params.cfg) + String(",\n")
    out += String('    "seed":') + String(params.seed) + String(",\n")
    out += String('    "noise_scheduler":"') + _json_escape(params.scheduler) + String('",\n')
    out += String('    "negative":"') + _json_escape(params.negative) + String('"\n')
    out += String("  },\n")
    out += String('  "prompts":[\n')
    out += String("    {\n")
    out += String('      "id":"job",\n')
    out += String('      "prompt":"') + _json_escape(params.prompt) + String('",\n')
    out += String('      "negative":"') + _json_escape(params.negative) + String('",\n')
    out += String('      "width":') + String(params.width) + String(",\n")
    out += String('      "height":') + String(params.height) + String(",\n")
    out += String('      "steps":') + String(params.steps) + String(",\n")
    out += String('      "cfg":') + String(params.cfg) + String(",\n")
    out += String('      "seed":') + String(params.seed) + String(",\n")
    out += String('      "caps":{"positive":"') + _json_escape(caps_pos) + String('","negative":"') + _json_escape(caps_neg) + String('"}\n')
    out += String("    }\n")
    out += String("  ]\n")
    out += String("}\n")
    return out^


def _embed_genparams_in_png(path: String, params_json: String) raises:
    var img = decode_image_any(path)
    var keys = List[String]()
    var vals = List[String]()
    keys.append(String("serenity.genparams.v1"))
    vals.append(params_json.copy())
    encode_png_with_text(img, path, keys, vals)


struct KleinBackend(GenBackend, Movable):
    var active: Bool
    var cancel_flag: Bool
    var params: JobParams
    var phase: Int
    var announced: Bool
    var variant: String
    var config_path: String
    var job_dir: String
    var sample_file: String
    var caps_pos: String
    var caps_neg: String
    var out_png: String
    var edit_parity_dir: String
    var lora_path: String
    var precache_bin: String
    var sampler_bin: String
    var sidecar: ExternalCommand

    def __init__(out self):
        self.active = False
        self.cancel_flag = False
        self.params = JobParams()
        self.phase = KPHASE_IDLE
        self.announced = False
        self.variant = String("9b")
        self.config_path = String(KLEIN9B_CONFIG)
        self.job_dir = String("")
        self.sample_file = String("")
        self.caps_pos = String("")
        self.caps_neg = String("")
        self.out_png = String("")
        self.edit_parity_dir = String("")
        self.lora_path = String("")
        self.precache_bin = String(KLEIN_PRECACHE_BIN)
        self.sampler_bin = String(KLEIN_SAMPLER_BIN)
        self.sidecar = ExternalCommand()

    def backend_name(self) -> String:
        return String("klein")

    def model_name(self) -> String:
        if self.params.model.byte_length() > 0:
            return self.params.model.copy()
        return String("flux2-klein")

    def resident_model(self) -> String:
        return String("")

    def start(mut self, params: JobParams) raises:
        if self.active:
            raise Error("KleinBackend.start: a job is already running")
        _validate_klein_request(params)
        self.params = params.copy()
        self.variant = _model_variant(params.model)
        self.config_path = _config_for_variant(self.variant)
        self.lora_path = _klein_lora_path(params)
        _require_file(String("Klein model config"), self.config_path)
        _require_file(String("Klein cap-cache precache binary"), self.precache_bin)
        _require_file(String("Klein staged sampler binary"), self.sampler_bin)
        var out_root = params.out_dir.copy()
        if out_root == String(""):
            out_root = String(DEFAULT_OUT_DIR)
        self.job_dir = out_root + String("/klein_jobs/") + params.job_id
        self.sample_file = self.job_dir + String("/sample_prompts.json")
        self.caps_pos = self.job_dir + String("/caps_pos.bin")
        self.caps_neg = self.job_dir + String("/caps_neg.bin")
        self.out_png = out_root + String("/") + params.job_id + String(".png")
        self.edit_parity_dir = self.job_dir + String("/edit_parity")
        self.phase = KPHASE_PREPARE
        self.announced = False
        self.cancel_flag = False
        self.active = True

    def cancel(mut self):
        self.cancel_flag = True
        self.sidecar.kill()

    def between_jobs_trim(mut self) raises:
        pass

    def _clear_job(mut self):
        self.sidecar.kill()
        self.active = False
        self.cancel_flag = False
        self.phase = KPHASE_IDLE
        self.announced = False
        self.job_dir = String("")
        self.sample_file = String("")
        self.caps_pos = String("")
        self.caps_neg = String("")
        self.out_png = String("")
        self.edit_parity_dir = String("")
        self.lora_path = String("")

    def _prepare_job_files(mut self) raises:
        _mkdir_p(self.job_dir)
        if self.params.reference_image.byte_length() > 0 or self.params.reference_latent_count > 0:
            _mkdir_p(self.edit_parity_dir)
        var sample_json = _sample_prompts_json(self.params, self.caps_pos, self.caps_neg)
        _write_text_file(self.sample_file, sample_json)
        _write_text_file(self.job_dir + String("/genparams.json"), self.params.params_json)

    def _precache_command(self) raises -> String:
        return (
            String("MODULAR_DEVICE_CONTEXT_SYNC_MODE=true ")
            + _shell_quote(self.precache_bin)
            + String(" ")
            + _shell_quote(self.sample_file)
            + String(" ")
            + _shell_quote(self.variant)
        )

    def _sample_command(self) raises -> String:
        var lora_arg = String("base")
        if self.lora_path.byte_length() > 0:
            lora_arg = self.lora_path.copy()
        var cmd = (
            String("MODULAR_DEVICE_CONTEXT_SYNC_MODE=true ")
            + _shell_quote(self.sampler_bin)
            + String(" ")
            + _shell_quote(self.config_path)
            + String(" ")
            + _shell_quote(lora_arg)
            + String(" ")
            + _shell_quote(self.sample_file)
            + String(" ")
            + _shell_quote(String("job"))
            + String(" ")
            + _shell_quote(self.out_png)
        )
        if self.params.reference_image.byte_length() > 0 or self.params.reference_latent_count > 0:
            cmd += (
                String(" ")
                + _shell_quote(String("-"))
                + String(" ")
                + _shell_quote(self.params.reference_image)
                + String(" ")
                + _shell_quote(String(self.params.creativity))
                + String(" ")
                + _shell_quote(String(KLEIN_REFERENCE_EDIT_SHIFT))
                + String(" ")
                + _shell_quote(String(KLEIN_REFERENCE_T_OFFSET))
                + String(" ")
                + _shell_quote(self.edit_parity_dir)
            )
        return cmd^

    def _write_result_manifest(self) raises -> String:
        var manifest = self.out_png + String(".klein_daemon_result.json")
        var out = String("{\n")
        out += String('  "schema":"serenity.klein_daemon_result.v1",\n')
        out += String('  "backend":"klein",\n')
        out += String('  "variant":"') + _json_escape(self.variant) + String('",\n')
        out += String('  "model":"') + _json_escape(self.params.model) + String('",\n')
        out += String('  "config_path":"') + _json_escape(self.config_path) + String('",\n')
        out += String('  "sample_prompts":"') + _json_escape(self.sample_file) + String('",\n')
        out += String('  "caps_positive":"') + _json_escape(self.caps_pos) + String('",\n')
        out += String('  "caps_negative":"') + _json_escape(self.caps_neg) + String('",\n')
        out += String('  "precache_binary":"') + _json_escape(self.precache_bin) + String('",\n')
        out += String('  "sampler_binary":"') + _json_escape(self.sampler_bin) + String('",\n')
        out += String('  "output_png":"') + _json_escape(self.out_png) + String('",\n')
        out += String('  "lora_count":') + String(len(self.params.loras)) + String(",\n")
        if len(self.params.loras) > 0:
            out += String('  "lora_name":"') + _json_escape(self.params.loras[0].name) + String('",\n')
            out += String('  "lora_path":"') + _json_escape(self.lora_path) + String('",\n')
            out += String('  "lora_weight":') + String(self.params.loras[0].weight) + String(",\n")
        if self.params.reference_image.byte_length() > 0 or self.params.reference_latent_count > 0:
            out += String('  "mode":"reference_latent_edit",\n')
            out += String('  "reference_image":"') + _json_escape(self.params.reference_image) + String('",\n')
            out += String('  "reference_latent_count":') + String(self.params.reference_latent_count) + String(",\n")
            out += String('  "edit_denoise":') + String(self.params.creativity) + String(",\n")
            out += String('  "edit_shift":') + String(KLEIN_REFERENCE_EDIT_SHIFT) + String(",\n")
            out += String('  "reference_t_offset":') + String(KLEIN_REFERENCE_T_OFFSET) + String(",\n")
            out += String('  "edit_parity_sidecar_dir":"') + _json_escape(self.edit_parity_dir) + String('",\n')
            out += String('  "edit_parity_manifest":"') + _json_escape(self.edit_parity_dir + String("/manifest.json")) + String('",\n')
            out += String('  "reference_vae_latent":"') + _json_escape(self.edit_parity_dir + String("/reference_vae_latent.bin")) + String('",\n')
        out += String('  "metadata_key":"serenity.genparams.v1",\n')
        out += String('  "note":"Klein daemon path ran process-separated Qwen3 cap-cache precache followed by the existing staged Klein sampler/edit runner; no placeholder image was written."\n')
        out += String("}\n")
        _write_text_file(manifest, out)
        return manifest

    def step(mut self) raises -> StepResult:
        var r = StepResult()
        r.total = self.params.steps
        if not self.active:
            r.failed = True
            r.error = String("klein: no active job")
            return r^
        if self.cancel_flag:
            self._clear_job()
            r.cancelled = True
            return r^
        try:
            if self.phase == KPHASE_PREPARE:
                self._prepare_job_files()
                self.phase = KPHASE_ENCODE
                r.step = 0
                r.phase = String("preparing")
                return r^
            if self.phase == KPHASE_ENCODE:
                if not self.announced:
                    self.sidecar.start(String("precache caps"), self._precache_command())
                    self.announced = True
                    r.step = 0
                    r.phase = String("encoding")
                    return r^
                if not self.sidecar.poll():
                    r.step = 0
                    r.phase = String("encoding")
                    return r^
                self.sidecar.require_success()
                var joint_dim = _joint_dim_for_variant(self.variant)
                validate_klein_cap_cache_header(self.caps_pos, joint_dim)
                validate_klein_cap_cache_header(self.caps_neg, joint_dim)
                self.announced = False
                self.phase = KPHASE_SAMPLE
                r.step = 0
                r.phase = String("encoding")
                return r^
            if self.phase == KPHASE_SAMPLE:
                if not self.announced:
                    self.sidecar.start(String("sample"), self._sample_command())
                    self.announced = True
                    r.step = 0
                    r.phase = String("sampling")
                    return r^
                if not self.sidecar.poll():
                    r.step = 0
                    r.phase = String("sampling")
                    return r^
                self.sidecar.require_success()
                if not _path_exists(self.out_png):
                    raise Error(String("klein: sampler did not produce output png: ") + self.out_png)
                _embed_genparams_in_png(self.out_png, self.params.params_json)
                var manifest = self._write_result_manifest()
                print("[klein][manifest] saved:", manifest)
                var path = self.out_png.copy()
                self._clear_job()
                r.step = self.params.steps
                r.done = True
                r.output_path = path^
                return r^
            raise Error("klein: invalid backend phase")
        except e:
            self._clear_job()
            r.failed = True
            r.error = String(e)
            return r^
