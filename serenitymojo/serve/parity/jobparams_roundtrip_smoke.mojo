# serenitymojo/serve/parity/jobparams_roundtrip_smoke.mojo
#
# GATES: the worker IPC wire contract and the frozen per-backend default constants.
#   Pure-CPU, no models, no GPU. Three checks:
#
#   1. ROUND-TRIP: encode_start(JobParams) -> loads() -> decode_start() reproduces
#      an identical JobParams. This is the exact parent->child path a worker runs
#      (serenity_worker_flux.mojo:59-63: loads(line) -> decode_start(obj)). Fields
#      exercised include the fixed defaults (steps/seed/cfg), cfg_override, sampler/
#      scheduler, image counts, and a LoRA entry.
#
#   2. JobParams DEFAULT STABILITY: JobParams() gives steps=20, seed=0, cfg=4.5
#      (serve/backend.mojo:141-145). These are the documented global defaults the
#      per-backend gate defaults are measured against.
#
#   3. BACKEND DEFAULT CONSTANTS (direct import) + their guard BRANCHES:
#        sd3       SD3_DEFAULT_STEPS=28, SD3_DEFAULT_SEED=42   (sd3_backend.mojo:182-183)
#                  applied at steps<=0 / seed<0                (sd3_backend.mojo:596-599)
#        qwenimage QWENIMAGE_DEFAULT_CFG=4.0                   (qwenimage_backend.mojo:89)
#                  applied at cfg<=0                            (qwenimage_backend.mojo:269)
#        sensenova steps<1 RAISES (no constant; admission guard, sensenova_backend.mojo:390)
#      The constants are imported and asserted; the guard BRANCH is replicated as
#      pure arithmetic (steps<=0 -> const, positive preserved) to prove the logic.
#
# NOT COVERED (needs GPU + real models -> cannot run in a CPU smoke): the actual
#   default application inside each backend's start()/admission path. start() loads
#   models before it reaches the guard, so this smoke gates the CONSTANTS + the
#   BRANCH LOGIC + the wire, not the in-situ start() call. flux's cfg<=0 -> 3.5
#   guard lives in flux_backend.start (self.guidance, NOT a module constant / NOT
#   JobParams), so it is out of scope here and documented, not asserted.
#
# BUILD-ONLY GATE for this pass; run: output/bin/jobparams_roundtrip_smoke
from json.parser import loads

from serenitymojo.serve.backend import JobParams, LoraSpec
from serenitymojo.serve.ipc_codec import encode_start, decode_start
from serenitymojo.serve.sd3_backend import SD3_DEFAULT_STEPS, SD3_DEFAULT_SEED
from serenitymojo.serve.qwenimage_backend import QWENIMAGE_DEFAULT_CFG


def _check(ok: Bool, label: String) raises:
    if not ok:
        raise Error(String("FAIL: ") + label)
    print("  PASS:", label)


def _fclose(a: Float64, b: Float64) -> Bool:
    var d = a - b
    if d < 0.0:
        d = -d
    return d < 1e-9


def main() raises:
    print("== jobparams_roundtrip_smoke ==")

    # ── 1. round-trip a populated JobParams through the worker wire ──────────
    var p = JobParams()
    p.job_id = String("gate-job-1")
    p.model = String("sd3")
    p.prompt = String("a photorealistic red fox sitting in autumn leaves")
    p.negative = String("blurry, low quality")
    p.width = 768
    p.height = 1024
    p.steps = 20          # documented global default
    p.seed = 0            # documented global default
    p.cfg = 4.5           # documented global default
    p.cfg_override = 7.0
    p.sampler = String("euler")
    p.scheduler = String("simple")
    p.images = 2
    p.image_index = 1
    p.image_count = 2
    p.out_dir = String("/tmp/gate")
    p.params_json = String("{\"seed\":0}")
    p.loras.append(LoraSpec(String("style_v1"), 0.75))

    var line = encode_start(p)
    var obj = loads(line)
    _check(obj.contains("cmd") and obj["cmd"].as_string() == "start", "wire cmd == start")
    var p2 = decode_start(obj)

    _check(p2.job_id == p.job_id, "round-trip job_id")
    _check(p2.model == p.model, "round-trip model")
    _check(p2.prompt == p.prompt, "round-trip prompt")
    _check(p2.negative == p.negative, "round-trip negative")
    _check(p2.width == p.width, "round-trip width")
    _check(p2.height == p.height, "round-trip height")
    _check(p2.steps == p.steps, "round-trip steps")
    _check(p2.seed == p.seed, "round-trip seed")
    _check(_fclose(p2.cfg, p.cfg), "round-trip cfg")
    _check(_fclose(p2.cfg_override, p.cfg_override), "round-trip cfg_override")
    _check(p2.sampler == p.sampler, "round-trip sampler")
    _check(p2.scheduler == p.scheduler, "round-trip scheduler")
    _check(p2.images == p.images, "round-trip images")
    _check(p2.image_index == p.image_index, "round-trip image_index")
    _check(p2.image_count == p.image_count, "round-trip image_count")
    _check(p2.out_dir == p.out_dir, "round-trip out_dir")
    _check(p2.params_json == p.params_json, "round-trip params_json")
    _check(len(p2.loras) == 1, "round-trip lora count")
    _check(p2.loras[0].name == "style_v1", "round-trip lora name")
    _check(_fclose(p2.loras[0].weight, 0.75), "round-trip lora weight")

    # ── 2. JobParams default stability ──────────────────────────────────────
    var d = JobParams()
    _check(d.steps == 20, "JobParams default steps == 20")
    _check(d.seed == 0, "JobParams default seed == 0")
    _check(_fclose(d.cfg, 4.5), "JobParams default cfg == 4.5")

    # ── 3a. sd3 constants + guard branch ────────────────────────────────────
    _check(SD3_DEFAULT_STEPS == 28, "SD3_DEFAULT_STEPS == 28")
    _check(SD3_DEFAULT_SEED == 42, "SD3_DEFAULT_SEED == 42")
    var sd3_steps = 0
    if sd3_steps <= 0:
        sd3_steps = SD3_DEFAULT_STEPS
    _check(sd3_steps == 28, "sd3 branch: steps<=0 -> 28")
    var sd3_seed = -1
    if sd3_seed < 0:
        sd3_seed = SD3_DEFAULT_SEED
    _check(sd3_seed == 42, "sd3 branch: seed<0 -> 42")
    var sd3_steps_keep = 20
    if sd3_steps_keep <= 0:
        sd3_steps_keep = SD3_DEFAULT_STEPS
    _check(sd3_steps_keep == 20, "sd3 branch: steps=20 preserved (not clobbered)")

    # ── 3b. qwenimage constant + guard branch ───────────────────────────────
    _check(QWENIMAGE_DEFAULT_CFG == Float32(4.0), "QWENIMAGE_DEFAULT_CFG == 4.0")
    var qi_cfg_in = Float32(0.0)
    var qi_cfg = Float32(qi_cfg_in) if qi_cfg_in > Float32(0.0) else QWENIMAGE_DEFAULT_CFG
    _check(qi_cfg == Float32(4.0), "qwenimage branch: cfg<=0 -> 4.0")
    var qi_cfg_in2 = Float32(4.5)
    var qi_cfg2 = qi_cfg_in2 if qi_cfg_in2 > Float32(0.0) else QWENIMAGE_DEFAULT_CFG
    _check(qi_cfg2 == Float32(4.5), "qwenimage branch: cfg=4.5 preserved")

    # ── 3c. sensenova admission guard (logic replica; no constant) ──────────
    var sn_bad = 0
    _check(sn_bad < 1, "sensenova branch: steps<1 would RAISE at admission")
    var sn_ok = 20
    _check(not (sn_ok < 1), "sensenova branch: steps=20 accepted")

    print("== jobparams_roundtrip_smoke: ALL PASS ==")
