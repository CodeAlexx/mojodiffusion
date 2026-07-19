# zimage_sweep_request_smoke — CPU gate for T1.F validation adapter sweeps.
#
# Gates the sampler-request side of the SimpleTuner --validation_adapter_config
# analogue WITHOUT touching the GPU or model weights:
#   1. writes a sweep sample-request JSON (the exact shape the
#      train_zimage_real _write_zimage_sample_request patch emits),
#   2. parses it back through the REAL product reader
#      (_load_zimage_sample_request) and asserts a field round-trip,
#   3. builds the sweep render plan (build_zimage_sweep_plan) and asserts
#      item count / adapter labels / strengths / adapter-tagged filenames,
#   4. exercises the sweep loop with a STUB render (file write per item),
#   5. asserts backward compatibility (no sweep fields → empty plan) and the
#      validation failures (length mismatch, strength<=0, missing lora file).
#
# Run: no GPU needed.
#   pixi run mojo build -I . -Xlinker -lm \
#     serenitymojo/pipeline/zimage_sweep_request_smoke.mojo \
#     -o /tmp/zimage_sweep_request_smoke && /tmp/zimage_sweep_request_smoke

from serenitymojo.io.ffi import sys_system
from serenitymojo.pipeline.zimage_generate import (
    ZImageSampleRequest,
    ZImageSweepItem,
    _load_zimage_sample_request,
    build_zimage_sweep_plan,
    _adapter_label,
    _sweep_output_path,
    _write_text_file,
    _path_exists,
)

comptime DIR = "/tmp/zimage_sweep_request_smoke"


def _check(cond: Bool, label: String) raises:
    if not cond:
        raise Error(String("FAIL: ") + label)
    print("  ok:", label)


# Mirrors the train_zimage_real.mojo _write_zimage_sample_request patch: same
# required fields, plus the OPTIONAL sweep_loras/sweep_strengths arrays.
def _request_json(
    completed_step: Int,
    lora_path: String,
    state_path: String,
    sample_file: String,
    output_png: String,
    result_manifest: String,
    sweep_loras: List[String],
    sweep_strengths: List[String],  # pre-formatted numbers (writer emits text)
) raises -> String:
    var content = String("{\n")
    content += String('  "schema":"serenity.zimage.sample_request.v1",\n')
    content += String('  "model":"zimage",\n')
    content += String('  "sampler_mode":"split_process_after_train_memory_release",\n')
    content += String('  "completed_step":') + String(completed_step) + String(",\n")
    content += String('  "lora_path":"') + lora_path + String('",\n')
    content += String('  "state_path":"') + state_path + String('",\n')
    content += String('  "sample_file":"') + sample_file + String('",\n')
    content += String('  "output_png":"') + output_png + String('",\n')
    content += String('  "result_manifest":"') + result_manifest + String('",\n')
    if len(sweep_loras) > 0:
        content += String('  "sweep_loras":[')
        for i in range(len(sweep_loras)):
            if i > 0:
                content += String(",")
            content += String('"') + sweep_loras[i] + String('"')
        content += String("],\n")
        content += String('  "sweep_strengths":[')
        for i in range(len(sweep_strengths)):
            if i > 0:
                content += String(",")
            content += sweep_strengths[i]
        content += String("],\n")
    content += String('  "accepted_parity":false,\n')
    content += String('  "note":"smoke fixture"\n')
    content += String("}\n")
    return content^


def main() raises:
    print("=== zimage sweep request smoke (T1.F, CPU only) ===")
    _ = sys_system(String("rm -rf ") + String(DIR))
    _ = sys_system(String("mkdir -p ") + String(DIR))

    # ── fixture files (the reader _require_file's every referenced path) ──
    var lora = String(DIR) + String("/zimage_lora_step120.safetensors")
    var state = String(DIR) + String("/zimage_lora_step120.state.json")
    var samples = String(DIR) + String("/sample_prompts.json")
    var sweep_a = String(DIR) + String("/Style-Alpha v2.safetensors")
    var sweep_b = String(DIR) + String("/detail_boost.safetensors")
    _write_text_file(lora, String("stub"))
    _write_text_file(state, String("stub"))
    _write_text_file(samples, String("stub"))
    _write_text_file(sweep_a, String("stub"))
    _write_text_file(sweep_b, String("stub"))
    var out_png = String(DIR) + String("/step120_sample.png")
    var manifest = String(DIR) + String("/step120_sample_result.json")

    # ── 1+2: sweep request round-trip through the product reader ──
    var sweep_paths = [sweep_a.copy(), sweep_b.copy()]
    var sweep_strengths_txt: List[String] = [String("0.8"), String("1.25")]
    var req_path = String(DIR) + String("/step120_request.json")
    _write_text_file(req_path, _request_json(
        120, lora, state, samples, out_png, manifest,
        sweep_paths, sweep_strengths_txt,
    ))
    var req = _load_zimage_sample_request(req_path)
    _check(req.completed_step == 120, String("completed_step round-trip"))
    _check(req.lora_path == lora, String("lora_path round-trip"))
    _check(req.output_png == out_png, String("output_png round-trip"))
    _check(len(req.sweep_loras) == 2, String("sweep_loras length round-trip"))
    _check(req.sweep_loras[0] == sweep_a, String("sweep_loras[0] round-trip"))
    _check(req.sweep_loras[1] == sweep_b, String("sweep_loras[1] round-trip"))
    _check(len(req.sweep_strengths) == 2, String("sweep_strengths length round-trip"))
    _check(abs(req.sweep_strengths[0] - Float32(0.8)) < Float32(1e-6), String("sweep_strengths[0] == 0.8"))
    _check(abs(req.sweep_strengths[1] - Float32(1.25)) < Float32(1e-6), String("sweep_strengths[1] == 1.25"))

    # ── 3: sweep plan — base item + one per adapter, tagged filenames ──
    var plan = build_zimage_sweep_plan(req)
    _check(len(plan) == 3, String("plan = base + 2 adapters"))
    _check(plan[0].label == String("base"), String("plan[0] is the no-adapter baseline"))
    _check(plan[0].lora_path == String(""), String("baseline renders with empty lora_path"))
    _check(
        plan[0].output_png == String(DIR) + String("/step120_sample_base.png"),
        String("baseline output naming"),
    )
    _check(plan[1].label == String("style_alpha_v2"), String("adapter stem slugified"))
    _check(plan[1].lora_path == sweep_a, String("plan[1] lora path"))
    _check(abs(plan[1].strength - Float32(0.8)) < Float32(1e-6), String("plan[1] strength 0.8"))
    _check(
        plan[1].output_png == String(DIR) + String("/step120_sample_style_alpha_v2.png"),
        String("adapter-tagged output naming"),
    )
    _check(plan[2].label == String("detail_boost"), String("plan[2] label"))
    _check(abs(plan[2].strength - Float32(1.25)) < Float32(1e-6), String("plan[2] strength 1.25"))
    _check(
        plan[2].result_manifest
        == String(DIR) + String("/step120_sample_detail_boost.png.zimage_result.json"),
        String("per-adapter manifest naming"),
    )

    # ── 4: sweep loop with a stub render (file write per item) ──
    for i in range(len(plan)):
        var item = plan[i].copy()
        _write_text_file(item.output_png, String("stub render ") + item.label)
        _write_text_file(item.result_manifest, String("stub manifest ") + item.label)
    for i in range(len(plan)):
        _check(_path_exists(plan[i].output_png), String("stub render exists: ") + plan[i].label)
        _check(_path_exists(plan[i].result_manifest), String("stub manifest exists: ") + plan[i].label)

    # ── label dedup: same adapter listed twice → _2 suffix ──
    var dup_req = ZImageSampleRequest(
        lora.copy(), state.copy(), samples.copy(), out_png.copy(), manifest.copy(),
        120, [sweep_a.copy(), sweep_a.copy()], List[Float32](),
    )
    var dup_plan = build_zimage_sweep_plan(dup_req)
    _check(len(dup_plan) == 3, String("dup plan = base + 2"))
    _check(dup_plan[1].label == String("style_alpha_v2"), String("dup first label"))
    _check(dup_plan[2].label == String("style_alpha_v2_2"), String("dup second label gets _2"))
    _check(
        abs(dup_plan[1].strength - Float32(1.0)) < Float32(1e-6),
        String("missing strengths default to 1.0"),
    )

    # ── 5a: backward compatibility — no sweep fields → empty plan ──
    var plain_path = String(DIR) + String("/step121_request.json")
    _write_text_file(plain_path, _request_json(
        121, lora, state, samples, out_png, manifest,
        List[String](), List[String](),
    ))
    var plain = _load_zimage_sample_request(plain_path)
    _check(len(plain.sweep_loras) == 0, String("plain request has no sweep loras"))
    var plain_plan = build_zimage_sweep_plan(plain)
    _check(len(plain_plan) == 0, String("plain request → empty sweep plan (current behavior)"))

    # ── 5b: validation failures ──
    var bad1 = String(DIR) + String("/bad_len_request.json")
    _write_text_file(bad1, _request_json(
        122, lora, state, samples, out_png, manifest,
        [sweep_a.copy(), sweep_b.copy()], [String("0.8")],
    ))
    var raised = False
    try:
        _ = _load_zimage_sample_request(bad1)
    except e:
        raised = True
        print("  (expected)", e)
    _check(raised, String("length mismatch raises"))

    var bad2 = String(DIR) + String("/bad_strength_request.json")
    _write_text_file(bad2, _request_json(
        123, lora, state, samples, out_png, manifest,
        [sweep_a.copy()], [String("0.0")],
    ))
    raised = False
    try:
        _ = _load_zimage_sample_request(bad2)
    except e:
        raised = True
        print("  (expected)", e)
    _check(raised, String("strength <= 0 raises"))

    var bad3 = String(DIR) + String("/bad_missing_request.json")
    _write_text_file(bad3, _request_json(
        124, lora, state, samples, out_png, manifest,
        [String(DIR) + String("/does_not_exist.safetensors")], [String("1.0")],
    ))
    raised = False
    try:
        _ = _load_zimage_sample_request(bad3)
    except e:
        raised = True
        print("  (expected)", e)
    _check(raised, String("missing sweep lora file raises"))

    # ── helper micro-checks ──
    _check(_adapter_label(String("/a/b/My LoRA.v3.safetensors")) == String("my_lora_v3"),
           String("_adapter_label keeps last-dot stem, slugifies"))
    _check(_sweep_output_path(String("/x/out"), String("z")) == String("/x/out_z.png"),
           String("_sweep_output_path appends .png when missing"))

    print("PASS: zimage sweep request smoke (T1.F)")
