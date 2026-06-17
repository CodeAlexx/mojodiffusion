#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");

const ROOT = path.resolve(__dirname, "..");
const BASE_URL = process.env.SERENITY_BASE_URL || "http://127.0.0.1:8787";
const CASE_FILTER = new Set(
  (process.env.SERENITY_MATRIX_CASES || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
);
const REPORT_PATH = process.env.SERENITY_MATRIX_REPORT || "";
const NODE_PATH_DEFAULT = path.join(process.env.TMPDIR || "/tmp", "mojodiffusion-playwright-tools", "node_modules");
const CHROME_DEFAULT = firstExisting(["/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser"]);
const PRODUCTION_PROMPT_V2 =
  "richly rendered Dutch master style surreal oil painting, atmospheric and moody chiaroscuro, a woman bent under a heavy load carrying an entire miniature landscape on her back, distant hills and trees rising from the burden, coherent anatomy, expressive face, detailed hands, textured brushwork, dramatic controlled light, no text, no watermark";
const PRODUCTION_NEGATIVE =
  "low quality, abstract texture field, tiled cells, noisy pattern, malformed body, extra limbs, distorted face, broken hands, unreadable text, watermark";

const CASES = [
  {
    id: "zimage",
    backend: "zimage",
    model: "z_image_base_bf16",
    width: 1024,
    height: 1024,
    steps: 16,
    cfg: 5,
    scheduler: "simple",
    seed: 2026061661,
    negative: PRODUCTION_NEGATIVE,
    requireWorkerResultManifest: true,
    prompt: PRODUCTION_PROMPT_V2,
  },
  {
    id: "ideogram4",
    backend: "ideogram4",
    model: "ideogram-4-fp8",
    width: 1024,
    height: 1024,
    steps: 20,
    cfg: 7,
    denoise: 0.5,
    creativity: 0.5,
    scheduler: "ideogram_logitnormal",
    seed: 2026061664,
    negative: "",
    requireWorkerResultManifest: true,
    prompt: PRODUCTION_PROMPT_V2,
    promptJson: {
      high_level_description:
        "A richly rendered Dutch master style surreal oil painting. Atmospheric and moody. A woman bent under a heavy load, carrying an entire miniature landscape on her back.",
      style_description: {
        aesthetics: "surreal Dutch master oil painting",
        lighting: "moody chiaroscuro",
        medium: "oil painting",
      },
      compositional_deconstruction: {
        background: "dark atmospheric studio space with distant hills fading into shadow",
        elements: [
          {
            description: "woman bent under a heavy load, carrying a miniature landscape of hills, trees, and paths on her back",
            bbox: [120, 220, 930, 820],
          },
        ],
      },
    },
  },
  {
    id: "sdxl",
    backend: "sdxl",
    model: "sd_xl_base_1.0",
    width: 1024,
    height: 1024,
    steps: 20,
    cfg: 7,
    scheduler: "normal",
    seed: 2026061662,
    requireWorkerResultManifest: true,
    prompt: PRODUCTION_PROMPT_V2,
  },
  {
    id: "anima",
    backend: "anima",
    model: "anima",
    width: 1024,
    height: 1024,
    steps: 20,
    cfg: 4.5,
    scheduler: "normal",
    seed: 2026061665,
    negative: PRODUCTION_NEGATIVE,
    requireWorkerResultManifest: true,
    prompt: PRODUCTION_PROMPT_V2,
  },
  {
    id: "sd3",
    backend: "sd3",
    model: "sd3.5_large",
    width: 1024,
    height: 1024,
    steps: 28,
    cfg: 4.5,
    scheduler: "simple",
    seed: 2026061666,
    negative: PRODUCTION_NEGATIVE,
    requireWorkerResultManifest: true,
    prompt: PRODUCTION_PROMPT_V2,
  },
  {
    id: "flux",
    backend: "flux",
    model: "flux1-dev",
    width: 1024,
    height: 1024,
    steps: 20,
    cfg: 4,
    scheduler: "simple",
    seed: 2026061667,
    negative: "",
    requireWorkerResultManifest: true,
    prompt: PRODUCTION_PROMPT_V2,
  },
  {
    id: "klein",
    backend: "flux2",
    model: "flux-2-klein-base-9b_fp8_e4m3fn",
    width: 512,
    height: 512,
    steps: 4,
    cfg: 4,
    scheduler: "simple",
    seed: 2026061663,
    negative: "",
    requireWorkerResultManifest: false,
    prompt: "richly rendered cinematic surreal portrait, a woman in a dark coat bent under a heavy load carrying a miniature landscape on her back, coherent face, detailed eyes and hands, moody studio light, painterly texture, centered subject, no text, no watermark",
  },
  {
    id: "sensenova",
    backend: "sensenova",
    model: "sensenova-u1",
    width: 1024,
    height: 1024,
    steps: 30,
    cfg: 4,
    scheduler: "simple",
    seed: 2026061730,
    negative: "",
    requireWorkerResultManifest: true,
    prompt: "high quality cinematic photo, a woman bent under a heavy load carrying a miniature landscape on her back, coherent face and hands, moody directional light, realistic fabric, detailed terrain, centered subject, no text, no watermark",
  },
];

function firstExisting(candidates) {
  for (const candidate of candidates) {
    if (candidate && fs.existsSync(candidate)) return candidate;
  }
  return "";
}

function selectedCases(admittedBackends) {
  const admitted = new Set(admittedBackends || []);
  if (!CASE_FILTER.size) return CASES.filter((c) => admitted.has(c.backend));
  return CASES.filter((c) => CASE_FILTER.has(c.id) || CASE_FILTER.has(c.model));
}

function runCase(c) {
  return new Promise((resolve, reject) => {
    const env = {
      ...process.env,
      SERENITY_BASE_URL: BASE_URL,
      SERENITY_VALID_IMAGE_MODEL: c.model,
      SERENITY_VALID_IMAGE_WIDTH: String(c.width),
      SERENITY_VALID_IMAGE_HEIGHT: String(c.height),
      SERENITY_VALID_IMAGE_STEPS: String(c.steps),
      SERENITY_VALID_IMAGE_CFG: String(c.cfg),
      SERENITY_VALID_IMAGE_DENOISE: String(c.denoise ?? 1.0),
      SERENITY_VALID_IMAGE_CREATIVITY: String(c.creativity ?? c.denoise ?? 1.0),
      SERENITY_VALID_IMAGE_SCHEDULER: c.scheduler,
      SERENITY_VALID_IMAGE_PROMPT: c.prompt,
      SERENITY_VALID_IMAGE_SEED: String(c.seed),
      PLAYWRIGHT_CHROMIUM_EXECUTABLE: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE || CHROME_DEFAULT,
    };
    if (c.promptJson !== undefined) env.SERENITY_VALID_IMAGE_PROMPT_JSON = JSON.stringify(c.promptJson);
    if (c.negative !== undefined) env.SERENITY_VALID_IMAGE_NEGATIVE = c.negative;
    if (!env.NODE_PATH && fs.existsSync(NODE_PATH_DEFAULT)) env.NODE_PATH = NODE_PATH_DEFAULT;

    const script = path.join(ROOT, "scripts", "check_serenity_browser_generate_valid_image.js");
    const child = spawn(process.execPath, [script], { cwd: ROOT, env });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      const text = chunk.toString();
      stdout += text;
      process.stdout.write(`[${c.id}] ${text}`);
    });
    child.stderr.on("data", (chunk) => {
      const text = chunk.toString();
      stderr += text;
      process.stderr.write(`[${c.id}] ${text}`);
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) {
        const err = new Error(`${c.id}: browser generate gate exited ${code}`);
        err.stdout = stdout;
        err.stderr = stderr;
        reject(err);
        return;
      }
      try {
        resolve(parseGateJson(stdout));
      } catch (err) {
        err.message = `${c.id}: ${err.message}`;
        err.stdout = stdout;
        reject(err);
      }
    });
  });
}

function parseGateJson(stdout) {
  const marker = '{\n  "schema": "serenity.browser_valid_image_gate.v1"';
  const start = stdout.lastIndexOf(marker);
  if (start < 0) throw new Error("missing browser gate JSON in stdout");
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < stdout.length; i += 1) {
    const ch = stdout[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === "\"") inString = false;
      continue;
    }
    if (ch === "\"") inString = true;
    else if (ch === "{") depth += 1;
    else if (ch === "}") {
      depth -= 1;
      if (depth === 0) return JSON.parse(stdout.slice(start, i + 1));
    }
  }
  throw new Error("unterminated browser gate JSON in stdout");
}

async function fetchJson(url) {
  const res = await fetch(url);
  const text = await res.text();
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}: ${text.slice(0, 500)}`);
  return JSON.parse(text);
}

function assertResult(caseDef, gate, result) {
  const failures = [];
  if (result.job_id !== gate.job_id) failures.push(`result job_id mismatch ${result.job_id} != ${gate.job_id}`);
  if (result.output_path !== gate.output_path) failures.push("result output_path mismatch");
  const outputLocation = result.output_location || {};
  if (gate.canonical_output_dir && outputLocation.root && outputLocation.root !== gate.canonical_output_dir) {
    failures.push(`output root mismatch ${outputLocation.root} != ${gate.canonical_output_dir}`);
  }
  if (outputLocation.root_kind !== "ui_workflow_gallery") {
    failures.push(`output_location root_kind ${outputLocation.root_kind || "missing"}`);
  }
  if (outputLocation.inside_root !== true) {
    failures.push("output_location is not inside product gallery root");
  }
  if (!outputLocation.relative_path) {
    failures.push("output_location relative_path missing");
  }
  const visual = result.visual_health || {};
  if (visual.status !== "pass") failures.push(`visual_health status ${visual.status || "missing"}`);
  if (visual.width !== caseDef.width || visual.height !== caseDef.height) {
    failures.push(`visual dimensions ${visual.width}x${visual.height}, expected ${caseDef.width}x${caseDef.height}`);
  }
  const refs = result.result_manifests || {};
  const serverRef = refs.server_result_manifest || {};
  const workerRef = refs.worker_result_manifest || {};
  if (!serverRef.present) failures.push("server result manifest missing");
  if (caseDef.requireWorkerResultManifest && !workerRef.present) failures.push("worker result manifest missing");
  const serverOutputLocation = (((result.server_result || {}).output || {}).location) || {};
  if (serverOutputLocation.root_kind !== "ui_workflow_gallery") {
    failures.push(`server_result output.location root_kind ${serverOutputLocation.root_kind || "missing"}`);
  }
  if (serverOutputLocation.inside_root !== true) {
    failures.push("server_result output.location is not inside product gallery root");
  }
  if (failures.length) throw new Error(`${caseDef.id}: ${failures.join("; ")}`);
  return {
    id: caseDef.id,
    backend: caseDef.backend,
    model: caseDef.model,
    job_id: gate.job_id,
    output_path: gate.output_path,
    output_location: outputLocation,
    width: gate.width,
    height: gate.height,
    steps: gate.steps,
    seed: gate.seed,
    prompt: caseDef.prompt,
    negative: caseDef.negative || "",
    visual_health: visual,
    worker_result_manifest_present: !!workerRef.present,
    worker_schema: result.worker_result ? result.worker_result.schema || null : null,
  };
}

async function admittedTextToImageBackends() {
  const cap = await fetchJson(`${BASE_URL}/v1/capabilities`);
  return (cap.backends || [])
    .filter((b) => b.production_status === "admitted")
    .filter((b) => (((b.features || {}).text_to_image || {}).supported === true))
    .map((b) => b.backend)
    .sort();
}

async function main() {
  const admittedBackends = await admittedTextToImageBackends();
  const cases = selectedCases(admittedBackends);
  if (!cases.length) throw new Error(`no selected cases match SERENITY_MATRIX_CASES=${process.env.SERENITY_MATRIX_CASES || ""}`);
  const explicitFilter = CASE_FILTER.size > 0;
  if (explicitFilter) {
    const blocked = cases.filter((c) => admittedBackends.indexOf(c.backend) < 0);
    if (blocked.length) {
      throw new Error(`selected cases are not admitted text-to-image backends: ${blocked.map((c) => `${c.id}(${c.backend})`).join(", ")}`);
    }
  }
  const selectedBackends = new Set(cases.map((c) => c.backend));
  const missing = admittedBackends.filter((backend) => !selectedBackends.has(backend));
  if (missing.length && !explicitFilter) {
    throw new Error(`matrix missing admitted text-to-image backend cases: ${missing.join(", ")}`);
  }
  const results = [];
  for (const c of cases) {
    console.log(`\n=== ${c.id} (${c.model}) ===`);
    const gate = await runCase(c);
    const result = await fetchJson(`${BASE_URL}/v1/job/${gate.job_id}/result`);
    results.push(assertResult(c, gate, result));
  }
  const report = {
    schema: "serenity.supported_generate_matrix.v1",
    base_url: BASE_URL,
    generated_at: new Date().toISOString(),
    passed: true,
    admitted_text_to_image_backends: admittedBackends,
    selected_backends: Array.from(selectedBackends).sort(),
    uncovered_admitted_backends: missing,
    case_count: results.length,
    cases: results,
    prompt_suite: {
      name: "production_prompt_v2",
      purpose: "harder prompt-adherence surface than object/not-blank smoke prompts",
      automated_gate_scope: "PNG integrity, dimensions, nonblank/nonflat/nonplaceholder heuristics only",
      manual_review_required: true,
    },
    limits: [
      "This proves current supported text-to-image browser generation and artifact validity for the selected cases; it is not prompt-adherence, aesthetic-quality, or sampler parity.",
      "A production readiness claim still requires visual inspection that the output matches the recorded prompt.",
      "Klein worker-side result manifests remain optional until the installed worker is rebuilt with the source-side sidecar writer.",
    ],
  };
  const rendered = JSON.stringify(report, null, 2);
  console.log(rendered);
  if (REPORT_PATH) {
    fs.mkdirSync(path.dirname(path.resolve(REPORT_PATH)), { recursive: true });
    fs.writeFileSync(REPORT_PATH, `${rendered}\n`);
  }
}

main().catch((err) => {
  console.error("supported generate matrix: FAIL");
  console.error(err.stack || err.message || String(err));
  process.exit(1);
});
