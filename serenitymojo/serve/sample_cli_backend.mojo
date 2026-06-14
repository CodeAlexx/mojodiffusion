# serenitymojo.serve.sample_cli_backend -- daemon wrapper for existing sample CLIs.
#
# This backend does not implement new inference math. It adapts existing Mojo
# sample binaries that already know how to run a model behind the daemon's
# GenBackend contract:
#
#   <cli> <config.json> <lora|-> <sample_prompts.json> <prompt_id> <out.png>
#
# SDXL and Anima currently require pre-encoded text conditioning sidecars. The
# daemon carries those paths as JobParams.sample_caps_pos/sample_caps_neg and
# writes them into serenity.sample_prompts.v1 caps.{positive,negative}.

from std.memory import alloc

from image.png import encode_png_with_text
from serenitymojo.io.ffi import (
    BytePtr, sys_open, sys_close, sys_pwrite,
    O_RDONLY, O_WRONLY, O_CREAT, O_TRUNC,
)
from serenitymojo.serve.backend import (
    GenBackend, JobParams, StepResult, reject_unsupported_common_runtime_params,
    reject_unsupported_reference_image_params,
    reject_unsupported_mask_image_params,
    reject_unsupported_inpaint_conditioning_params,
    reject_unsupported_qwen_edit_conditioning_params,
    reject_unsupported_conditioning_mask_params,
    reject_unsupported_lanpaint_params,
)
from serenitymojo.serve.external_command import ExternalCommand
from serenitymojo.serve.image_io import decode_image_any
from serenitymojo.serve.model_scan import LORAS_DIR


comptime DEFAULT_OUT_DIR = "/home/alex/mojodiffusion/output/serenity_daemon"
comptime SAMPLE_CLI_SDXL_BIN = "/home/alex/mojodiffusion/output/bin/sdxl_sample_cli"
comptime SAMPLE_CLI_ANIMA_BIN = "/home/alex/mojodiffusion/output/bin/anima_serenity_cli"
comptime SAMPLE_CLI_SDXL_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/sdxl.json"
comptime SAMPLE_CLI_ANIMA_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/anima.json"

comptime SCPHASE_IDLE = 0
comptime SCPHASE_PREPARE = 1
comptime SCPHASE_SAMPLE = 2


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
        raise Error(String("sample_cli_backend: cannot create ") + path)
    var n = content.byte_length()
    var buf = alloc[UInt8](n if n > 0 else 1)
    var src = content.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    var wrote = sys_pwrite(fd, BytePtr(unsafe_from_address=Int(buf)), n, 0)
    buf.free()
    _ = sys_close(fd)
    if wrote != n:
        raise Error(String("sample_cli_backend: short write to ") + path)


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
    raise Error(String("sample_cli_backend: quick command timed out label=") + label)


def _mkdir_p(path: String) raises:
    if path == String(""):
        return
    _run_quick_command(String("mkdir sample cli job dir"), String("mkdir -p ") + _shell_quote(path))


def _build_command_hint(family: String) -> String:
    if family == String("sdxl"):
        return String("build required binary with: pixi run build-sdxl-cli")
    if family == String("anima"):
        return String("build required binary with: pixi run build-anima-cli")
    return String("build the requested sample CLI binary")


def _require_file(family: String, label: String, path: String) raises:
    if not _path_exists(path):
        raise Error(
            String("sample-cli/") + family + String(": missing ") + label
            + String(": ") + path + String("; ") + _build_command_hint(family)
        )


def _resolve_lora_path(family: String, name: String) raises -> String:
    if name == String(""):
        raise Error(String("sample-cli/") + family + String(": empty LoRA name"))
    if _path_exists(name):
        return name.copy()
    if _path_exists(name + ".safetensors"):
        return name + ".safetensors"
    if _path_exists(String(LORAS_DIR) + String("/") + name):
        return String(LORAS_DIR) + String("/") + name
    if _path_exists(String(LORAS_DIR) + String("/") + name + String(".safetensors")):
        return String(LORAS_DIR) + String("/") + name + String(".safetensors")
    raise Error(
        String("sample-cli/") + family + String(": LoRA file not found: ") + name
        + String(" (tried as a path and under ") + String(LORAS_DIR) + String(")")
    )


def _lora_arg_for_family(params: JobParams, family: String) raises -> String:
    if family == String("sdxl"):
        if len(params.loras) > 0:
            raise Error("sample-cli/sdxl: LoRA is rejected because sdxl_sample_cli accepts but ignores the LoRA argument today")
        return String("-")
    if len(params.loras) == 0:
        return String("-")
    if len(params.loras) > 1:
        raise Error("sample-cli/anima: exactly one LoRA is supported until stack semantics are exposed")
    if params.loras[0].weight != 1.0:
        raise Error("sample-cli/anima: LoRA weight other than 1.0 is not wired through anima_serenity_cli")
    return _resolve_lora_path(family, params.loras[0].name)


def _config_for_family(family: String) raises -> String:
    if family == String("sdxl"):
        return String(SAMPLE_CLI_SDXL_CONFIG)
    if family == String("anima"):
        return String(SAMPLE_CLI_ANIMA_CONFIG)
    raise Error(String("sample-cli: unsupported family '") + family + String("'"))


def _bin_for_family(family: String) raises -> String:
    if family == String("sdxl"):
        return String(SAMPLE_CLI_SDXL_BIN)
    if family == String("anima"):
        return String(SAMPLE_CLI_ANIMA_BIN)
    raise Error(String("sample-cli: unsupported family '") + family + String("'"))


def _validate_fixed_cli_request(params: JobParams, family: String) raises:
    reject_unsupported_common_runtime_params(params, String("sample-cli/") + family)
    reject_unsupported_reference_image_params(params, String("sample-cli/") + family)
    reject_unsupported_mask_image_params(params, String("sample-cli/") + family)
    reject_unsupported_inpaint_conditioning_params(params, String("sample-cli/") + family)
    reject_unsupported_qwen_edit_conditioning_params(params, String("sample-cli/") + family)
    reject_unsupported_conditioning_mask_params(params, String("sample-cli/") + family)
    reject_unsupported_lanpaint_params(params, String("sample-cli/") + family)
    if params.init_image.byte_length() > 0:
        raise Error(
            String("sample-cli/") + family
            + String(": img2img/init image is not supported by the existing sample CLI path")
        )
    if params.width != 1024 or params.height != 1024:
        raise Error(
            String("sample-cli/") + family + String(": unsupported size ")
            + String(params.width) + String("x") + String(params.height)
            + String(" (existing CLI is compiled for 1024x1024)")
        )
    if params.steps != 30:
        raise Error(
            String("sample-cli/") + family
            + String(": steps must be 30 because the existing CLI has a fixed denoise loop")
        )
    if params.variation_strength > 0.0:
        raise Error(String("sample-cli/") + family + String(": variation noise is not wired"))
    var sampler = _lower(params.sampler)
    if sampler != String("euler"):
        raise Error(
            String("sample-cli/") + family + String(": unsupported sampler '")
            + params.sampler + String("' (existing CLI path is Euler-only)")
        )
    var scheduler = _lower(params.scheduler)
    if family == String("sdxl"):
        if scheduler != String("normal"):
            raise Error("sample-cli/sdxl: scheduler must be normal for the existing fixed SDXL Euler path")
        if params.cfg != 7.5:
            raise Error("sample-cli/sdxl: cfg must be 7.5 because sdxl_sample_cli is fixed at compile time")
        if params.seed != 42:
            raise Error("sample-cli/sdxl: seed must be 42 because sdxl_sample_cli is fixed at compile time")
        if params.sample_caps_pos.byte_length() == 0:
            raise Error("sample-cli/sdxl: sample_caps_pos/caps_pos sidecar is required; refusing default developer-test embedding fallback")
    elif family == String("anima"):
        if scheduler != String("normal"):
            raise Error("sample-cli/anima: scheduler must be normal for the existing fixed Anima Euler path")
        if params.cfg != 4.5:
            raise Error("sample-cli/anima: cfg must be 4.5 because anima_serenity_cli is fixed at compile time")
        if params.sample_caps_pos.byte_length() == 0 or params.sample_caps_neg.byte_length() == 0:
            raise Error("sample-cli/anima: sample_caps_pos and sample_caps_neg sidecars are required")
    else:
        raise Error(String("sample-cli: unsupported family '") + family + String("'"))
    _ = _lora_arg_for_family(params, family)


def _sample_caps_neg_for_family(params: JobParams, family: String) -> String:
    if params.sample_caps_neg.byte_length() > 0:
        return params.sample_caps_neg.copy()
    if family == String("sdxl"):
        # SDXL uses a single sidecar containing cond and uncond tensors. Fill
        # caps.negative so sample_prompt_config's precache_required validation
        # can remain enabled while the CLI still reads caps_pos.
        return params.sample_caps_pos.copy()
    return String("")


def _sample_prompts_json(params: JobParams, family: String) raises -> String:
    var caps_neg = _sample_caps_neg_for_family(params, family)
    var out = String("{\n")
    out += String('  "schema":"serenity.sample_prompts.v1",\n')
    out += String('  "defaults":{\n')
    out += String('    "precache_required":true,\n')
    out += String('    "enforce_min_image_size":false,\n')
    out += String('    "width":1024,\n')
    out += String('    "height":1024,\n')
    out += String('    "frames":1,\n')
    out += String('    "steps":30,\n')
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
    out += String('      "width":1024,\n')
    out += String('      "height":1024,\n')
    out += String('      "frames":1,\n')
    out += String('      "steps":30,\n')
    out += String('      "cfg":') + String(params.cfg) + String(",\n")
    out += String('      "seed":') + String(params.seed) + String(",\n")
    out += String('      "caps":{"positive":"') + _json_escape(params.sample_caps_pos) + String('","negative":"') + _json_escape(caps_neg) + String('"}\n')
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


struct SampleCliBackend(GenBackend, Movable):
    var family: String
    var active: Bool
    var cancel_flag: Bool
    var params: JobParams
    var phase: Int
    var announced: Bool
    var config_path: String
    var cli_bin: String
    var job_dir: String
    var sample_file: String
    var out_png: String
    var lora_arg: String
    var sidecar: ExternalCommand

    def __init__(out self, family: String):
        self.family = family.copy()
        self.active = False
        self.cancel_flag = False
        self.params = JobParams()
        self.phase = SCPHASE_IDLE
        self.announced = False
        self.config_path = String("")
        self.cli_bin = String("")
        self.job_dir = String("")
        self.sample_file = String("")
        self.out_png = String("")
        self.lora_arg = String("-")
        self.sidecar = ExternalCommand()

    def backend_name(self) -> String:
        return String("sample-cli/") + self.family

    def model_name(self) -> String:
        if self.params.model.byte_length() > 0:
            return self.params.model.copy()
        return self.family.copy()

    def resident_model(self) -> String:
        return String("")

    def start(mut self, params: JobParams) raises:
        if self.active:
            raise Error("SampleCliBackend.start: a job is already running")
        _validate_fixed_cli_request(params, self.family)
        self.params = params.copy()
        self.config_path = _config_for_family(self.family)
        self.cli_bin = _bin_for_family(self.family)
        self.lora_arg = _lora_arg_for_family(params, self.family)
        _require_file(self.family, String("model config"), self.config_path)
        _require_file(self.family, String("sample CLI binary"), self.cli_bin)
        if params.sample_caps_pos.byte_length() > 0:
            _require_file(self.family, String("positive conditioning sidecar"), params.sample_caps_pos)
        if params.sample_caps_neg.byte_length() > 0:
            _require_file(self.family, String("negative conditioning sidecar"), params.sample_caps_neg)
        var out_root = params.out_dir.copy()
        if out_root == String(""):
            out_root = String(DEFAULT_OUT_DIR)
        self.job_dir = out_root + String("/") + self.family + String("_jobs/") + params.job_id
        self.sample_file = self.job_dir + String("/sample_prompts.json")
        self.out_png = out_root + String("/") + params.job_id + String(".png")
        self.phase = SCPHASE_PREPARE
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
        self.phase = SCPHASE_IDLE
        self.announced = False
        self.job_dir = String("")
        self.sample_file = String("")
        self.out_png = String("")
        self.lora_arg = String("-")

    def _prepare_job_files(mut self) raises:
        _mkdir_p(self.job_dir)
        var sample_json = _sample_prompts_json(self.params, self.family)
        _write_text_file(self.sample_file, sample_json)
        _write_text_file(self.job_dir + String("/genparams.json"), self.params.params_json)

    def _sample_command(self) raises -> String:
        return (
            String("MODULAR_DEVICE_CONTEXT_SYNC_MODE=true ")
            + _shell_quote(self.cli_bin)
            + String(" ")
            + _shell_quote(self.config_path)
            + String(" ")
            + _shell_quote(self.lora_arg)
            + String(" ")
            + _shell_quote(self.sample_file)
            + String(" ")
            + _shell_quote(String("job"))
            + String(" ")
            + _shell_quote(self.out_png)
        )

    def _write_result_manifest(self) raises -> String:
        var manifest = self.out_png + String(".") + self.family + String("_daemon_result.json")
        var caps_neg = _sample_caps_neg_for_family(self.params, self.family)
        var out = String("{\n")
        out += String('  "schema":"serenity.sample_cli_daemon_result.v1",\n')
        out += String('  "backend":"sample-cli/",\n')
        out += String('  "family":"') + _json_escape(self.family) + String('",\n')
        out += String('  "model":"') + _json_escape(self.params.model) + String('",\n')
        out += String('  "config_path":"') + _json_escape(self.config_path) + String('",\n')
        out += String('  "sample_prompts":"') + _json_escape(self.sample_file) + String('",\n')
        out += String('  "caps_positive":"') + _json_escape(self.params.sample_caps_pos) + String('",\n')
        out += String('  "caps_negative":"') + _json_escape(caps_neg) + String('",\n')
        out += String('  "sample_cli_binary":"') + _json_escape(self.cli_bin) + String('",\n')
        out += String('  "output_png":"') + _json_escape(self.out_png) + String('",\n')
        out += String('  "lora_count":') + String(len(self.params.loras)) + String(",\n")
        out += String('  "metadata_key":"serenity.genparams.v1",\n')
        out += String('  "note":"Existing Mojo sample CLI executed behind the daemon; no placeholder image was written."\n')
        out += String("}\n")
        _write_text_file(manifest, out)
        return manifest

    def step(mut self) raises -> StepResult:
        var r = StepResult()
        r.total = self.params.steps
        if not self.active:
            r.failed = True
            r.error = String("sample-cli/") + self.family + String(": no active job")
            return r^
        if self.cancel_flag:
            self._clear_job()
            r.cancelled = True
            return r^
        try:
            if self.phase == SCPHASE_PREPARE:
                self._prepare_job_files()
                self.phase = SCPHASE_SAMPLE
                r.step = 0
                r.phase = String("preparing")
                return r^
            if self.phase == SCPHASE_SAMPLE:
                if not self.announced:
                    self.sidecar.start(String("sample-cli ") + self.family, self._sample_command())
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
                    raise Error(
                        String("sample-cli/") + self.family
                        + String(": CLI did not produce output png: ") + self.out_png
                    )
                _embed_genparams_in_png(self.out_png, self.params.params_json)
                var manifest = self._write_result_manifest()
                print("[sample-cli][manifest] saved:", manifest)
                var path = self.out_png.copy()
                self._clear_job()
                r.step = self.params.steps
                r.done = True
                r.output_path = path^
                return r^
            raise Error(String("sample-cli/") + self.family + String(": invalid backend phase"))
        except e:
            self._clear_job()
            r.failed = True
            r.error = String(e)
            return r^
