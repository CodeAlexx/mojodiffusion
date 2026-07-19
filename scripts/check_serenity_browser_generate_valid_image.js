#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const zlib = require("zlib");

let chromium;
try {
  ({ chromium } = require("playwright"));
} catch (err) {
  console.error("browser generate valid image: FAIL");
  console.error("Playwright is not installed. Set NODE_PATH to a node_modules containing playwright.");
  process.exit(2);
}

const ROOT = path.resolve(__dirname, "..");
const BASE_URL = process.env.SERENITY_BASE_URL || "http://127.0.0.1:8787";
const CANONICAL_OUTPUT_DIR = resolveRepoPath(
  process.env.SERENITY_VALID_IMAGE_OUTPUT_DIR || path.join("output", "run_serenity_ui")
);
const CHROME_BIN = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ||
  firstExisting(["/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser"]);
const MODEL = process.env.SERENITY_VALID_IMAGE_MODEL || "sd_xl_base_1.0";
const WIDTH = Number(process.env.SERENITY_VALID_IMAGE_WIDTH || 1024);
const HEIGHT = Number(process.env.SERENITY_VALID_IMAGE_HEIGHT || 1024);
const STEPS = Number(process.env.SERENITY_VALID_IMAGE_STEPS || 20);
const CFG = Number(process.env.SERENITY_VALID_IMAGE_CFG || 7.0);
const GUIDANCE = Object.prototype.hasOwnProperty.call(process.env, "SERENITY_VALID_IMAGE_GUIDANCE")
  ? Number(process.env.SERENITY_VALID_IMAGE_GUIDANCE)
  : null;
const DENOISE = Number(process.env.SERENITY_VALID_IMAGE_DENOISE || 1.0);
const CREATIVITY = Number(process.env.SERENITY_VALID_IMAGE_CREATIVITY || DENOISE);
const MIN_PRODUCTION_DIM = Number(process.env.SERENITY_VALID_IMAGE_MIN_DIM || 512);
const SEED = Number(process.env.SERENITY_VALID_IMAGE_SEED || Date.now() % 2147483647);
const PROMPT = process.env.SERENITY_VALID_IMAGE_PROMPT ||
  "a richly rendered Dutch master style oil painting portrait, moody atmospheric chiaroscuro, coherent anatomy, detailed face, detailed hands, textured brushwork";
const PROMPT_JSON = Object.prototype.hasOwnProperty.call(process.env, "SERENITY_VALID_IMAGE_PROMPT_JSON")
  ? process.env.SERENITY_VALID_IMAGE_PROMPT_JSON
  : "";
const NEGATIVE = Object.prototype.hasOwnProperty.call(process.env, "SERENITY_VALID_IMAGE_NEGATIVE")
  ? process.env.SERENITY_VALID_IMAGE_NEGATIVE
  : "low quality, abstract pattern, tiled cells, broken anatomy, distorted face, text, watermark";
const SAMPLER = process.env.SERENITY_VALID_IMAGE_SAMPLER || "euler";
const SCHEDULER = process.env.SERENITY_VALID_IMAGE_SCHEDULER || "euler";

function resolveRepoPath(value) {
  return path.resolve(path.isAbsolute(value) ? value : path.join(ROOT, value));
}

function isPathInside(child, parent) {
  const rel = path.relative(parent, child);
  return rel === "" || (!!rel && !rel.startsWith("..") && !path.isAbsolute(rel));
}

function assertCanonicalOutputPath(outputPath) {
  const resolved = path.resolve(outputPath);
  if (!isPathInside(resolved, CANONICAL_OUTPUT_DIR)) {
    throw new Error(
      `generated image landed outside canonical UI output dir: ${resolved}; expected under ${CANONICAL_OUTPUT_DIR}`
    );
  }
}

function firstExisting(candidates) {
  for (const candidate of candidates) {
    if (candidate && fs.existsSync(candidate)) return candidate;
  }
  return "";
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitUntil(fn, timeoutMs, label) {
  const end = Date.now() + timeoutMs;
  let lastError = null;
  while (Date.now() < end) {
    try {
      const value = await fn();
      if (value) return value;
    } catch (err) {
      if (err && String(err.message || err).startsWith("preflight rejected:")) throw err;
      lastError = err;
    }
    await sleep(500);
  }
  const renderedLabel = typeof label === "function" ? label() : label;
  throw new Error(`${renderedLabel} timed out${lastError ? `: ${lastError.message}` : ""}`);
}

function evidenceForFailure(lastPreflight, lastGenerateResponse) {
  return `preflight=${JSON.stringify(lastPreflight || null)} generate=${JSON.stringify(lastGenerateResponse || null)}`;
}

function paeth(a, b, c) {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  if (pb <= pc) return b;
  return c;
}

function decodePngRgb(filePath) {
  const data = fs.readFileSync(filePath);
  const sig = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  if (data.length < 8 || !data.subarray(0, 8).equals(sig)) {
    throw new Error("png: bad signature");
  }
  let pos = 8;
  let width = 0;
  let height = 0;
  let bitDepth = 0;
  let colorType = 0;
  let interlace = 0;
  const idats = [];
  while (pos + 12 <= data.length) {
    const len = data.readUInt32BE(pos);
    const type = data.toString("ascii", pos + 4, pos + 8);
    const chunk = data.subarray(pos + 8, pos + 8 + len);
    if (pos + 12 + len > data.length) throw new Error(`png: ${type} chunk overruns file`);
    if (type === "IHDR") {
      width = chunk.readUInt32BE(0);
      height = chunk.readUInt32BE(4);
      bitDepth = chunk[8];
      colorType = chunk[9];
      interlace = chunk[12];
    } else if (type === "IDAT") {
      idats.push(chunk);
    } else if (type === "IEND") {
      break;
    }
    pos += 12 + len;
  }
  if (width <= 0 || height <= 0) throw new Error("png: missing IHDR");
  if (bitDepth !== 8) throw new Error(`png: unsupported bit depth ${bitDepth}`);
  if (interlace !== 0) throw new Error("png: interlaced PNG unsupported by gate");
  const channels = colorType === 2 ? 3 : colorType === 6 ? 4 : 0;
  if (!channels) throw new Error(`png: unsupported color type ${colorType}`);
  const raw = zlib.inflateSync(Buffer.concat(idats));
  const rowBytes = width * channels;
  const rgb = Buffer.alloc(width * height * 3);
  let src = 0;
  let dst = 0;
  let prev = Buffer.alloc(rowBytes);
  for (let y = 0; y < height; y += 1) {
    const filter = raw[src++];
    const row = Buffer.from(raw.subarray(src, src + rowBytes));
    src += rowBytes;
    for (let x = 0; x < rowBytes; x += 1) {
      const left = x >= channels ? row[x - channels] : 0;
      const up = prev[x] || 0;
      const upLeft = x >= channels ? prev[x - channels] || 0 : 0;
      if (filter === 1) row[x] = (row[x] + left) & 255;
      else if (filter === 2) row[x] = (row[x] + up) & 255;
      else if (filter === 3) row[x] = (row[x] + Math.floor((left + up) / 2)) & 255;
      else if (filter === 4) row[x] = (row[x] + paeth(left, up, upLeft)) & 255;
      else if (filter !== 0) throw new Error(`png: unsupported filter ${filter}`);
    }
    for (let x = 0; x < width; x += 1) {
      const base = x * channels;
      rgb[dst++] = row[base];
      rgb[dst++] = row[base + 1];
      rgb[dst++] = row[base + 2];
    }
    prev = row;
  }
  return { width, height, rgb };
}

function statsForImage(img) {
  const n = img.width * img.height;
  const sums = [0, 0, 0];
  const sums2 = [0, 0, 0];
  let minLum = 255;
  let maxLum = 0;
  let edgeSum = 0;
  let edgeCount = 0;
  const bins = new Set();
  for (let y = 0; y < img.height; y += 1) {
    for (let x = 0; x < img.width; x += 1) {
      const i = (y * img.width + x) * 3;
      const r = img.rgb[i];
      const g = img.rgb[i + 1];
      const b = img.rgb[i + 2];
      sums[0] += r; sums[1] += g; sums[2] += b;
      sums2[0] += r * r; sums2[1] += g * g; sums2[2] += b * b;
      const lum = (r + g + b) / 3;
      minLum = Math.min(minLum, lum);
      maxLum = Math.max(maxLum, lum);
      bins.add(`${r >> 4},${g >> 4},${b >> 4}`);
      if (x > 0) {
        const j = i - 3;
        edgeSum += Math.abs(r - img.rgb[j]) + Math.abs(g - img.rgb[j + 1]) + Math.abs(b - img.rgb[j + 2]);
        edgeCount += 3;
      }
      if (y > 0) {
        const j = i - img.width * 3;
        edgeSum += Math.abs(r - img.rgb[j]) + Math.abs(g - img.rgb[j + 1]) + Math.abs(b - img.rgb[j + 2]);
        edgeCount += 3;
      }
    }
  }
  const mean = sums.map((v) => v / n);
  const stddev = sums.map((v, c) => Math.sqrt(Math.max(0, sums2[c] / n - mean[c] * mean[c])));
  return {
    width: img.width,
    height: img.height,
    mean,
    stddev,
    avgStddev: stddev.reduce((a, b) => a + b, 0) / 3,
    luminanceRange: maxLum - minLum,
    edgeEnergy: edgeSum / Math.max(1, edgeCount),
    colorBins: bins.size,
  };
}

function assertValidImage(stats, expectedWidth, expectedHeight) {
  const failures = [];
  if (stats.width !== expectedWidth || stats.height !== expectedHeight) {
    failures.push(`wrong dimensions ${stats.width}x${stats.height}, expected ${expectedWidth}x${expectedHeight}`);
  }
  if (expectedWidth < MIN_PRODUCTION_DIM || expectedHeight < MIN_PRODUCTION_DIM) {
    failures.push(`below product evidence minimum ${expectedWidth}x${expectedHeight}; 256px-scale artifacts are smoke tests only`);
  }
  if (stats.avgStddev < 18 && (stats.luminanceRange < 55 || stats.edgeEnergy < 1.0 || stats.colorBins < 48)) {
    failures.push(`low RGB stddev ${stats.avgStddev.toFixed(2)} with a second flat/blank signal`);
  }
  if (stats.luminanceRange < 55) failures.push(`low luminance range ${stats.luminanceRange.toFixed(2)} (washed/blank risk)`);
  if (stats.edgeEnergy < 1.0) failures.push(`low edge energy ${stats.edgeEnergy.toFixed(2)} (missing detail risk)`);
  if (stats.colorBins < 48) failures.push(`too few color bins ${stats.colorBins} (posterized/placeholder risk)`);
  if (failures.length) throw new Error(`invalid generated image: ${failures.join("; ")}`);
}

async function run() {
  if (!CHROME_BIN || !fs.existsSync(CHROME_BIN)) throw new Error("Chrome/Chromium executable not found");
  const browser = await chromium.launch({ executablePath: CHROME_BIN, headless: true, args: ["--no-sandbox"] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  let jobId = "";
  let lastPreflight = null;
  let lastGenerateResponse = null;
  page.on("response", async (response) => {
    const url = response.url();
    const method = response.request().method();
    if (url === `${BASE_URL}/v1/preflight` && method === "POST") {
      try {
        lastPreflight = await response.json();
      } catch (_) {
        lastPreflight = { status: response.status(), parse_error: true };
      }
      const admitted = lastPreflight && lastPreflight.admitted;
      const backend = lastPreflight && (lastPreflight.backend || (lastPreflight.capability_profile || {}).backend);
      console.log(`preflight response ${response.status()} admitted=${admitted} backend=${backend || ""}`);
    }
    if ((url === `${BASE_URL}/prompt` || url === `${BASE_URL}/v1/generate`) && method === "POST") {
      try {
        lastGenerateResponse = await response.json();
      } catch (_) {
        lastGenerateResponse = { status: response.status(), parse_error: true };
      }
      jobId = lastGenerateResponse.job_id || lastGenerateResponse.id || lastGenerateResponse.prompt_id || "";
      console.log(`generate response ${response.status()} job=${jobId}`);
    }
  });
  await page.goto(BASE_URL, { waitUntil: "domcontentloaded" });
  await page.waitForSelector("#gen-btn", { state: "attached", timeout: 15000 });
  await waitUntil(
    () => page.evaluate(() => !!(
      window.GenerateTab
      && window.SerenityAPI
      && GenerateTab.state
      && Array.isArray(GenerateTab.state.allModels)
      && GenerateTab.state.allModels.length > 0
    )),
    15000,
    "production Generate tab model inventory ready"
  );
  const requestedModelInInventory = await page.evaluate(
    (model) => GenerateTab.state.allModels.some((candidate) => candidate && candidate.name === model),
    MODEL
  );
  if (!requestedModelInInventory) {
    await browser.close();
    throw new Error(`requested model is not present in the live Generate picker: ${MODEL}`);
  }
  await page.evaluate((cfg) => {
    Object.assign(GenerateTab.state, {
      generating: false,
      model: cfg.model,
      // Ideogram-4 is trained on structured JSON captions.  The matrix passes
      // one explicitly; exercise that exact production input instead of
      // silently discarding it and falling back to the higher-false-positive
      // plain-text wrapper.
      prompt: cfg.promptJson || cfg.prompt,
      negPrompt: cfg.negative,
      width: cfg.width,
      height: cfg.height,
      steps: cfg.steps,
      cfg: cfg.cfg,
      guidance: cfg.guidance != null ? cfg.guidance : GenerateTab.state.guidance,
      seed: cfg.seed,
      batchCount: 1,
      scheduler: cfg.scheduler,
      stylePreset: "none",
      loras: [],
      vae: "default",
      refinerModel: "",
    });
    const btn = document.querySelector("#gen-btn");
    if (btn) { btn.disabled = false; btn.classList.remove("is-busy"); }
  }, { model: MODEL, prompt: PROMPT, promptJson: PROMPT_JSON, negative: NEGATIVE, width: WIDTH, height: HEIGHT, steps: STEPS, cfg: CFG, guidance: GUIDANCE, denoise: DENOISE, creativity: CREATIVITY, seed: SEED, sampler: SAMPLER, scheduler: SCHEDULER });
  await page.locator("#gen-btn").click();
  await waitUntil(async () => {
    if (lastPreflight && lastPreflight.admitted === false) {
      throw new Error(`preflight rejected: ${evidenceForFailure(lastPreflight, lastGenerateResponse)}`);
    }
    if (jobId) return jobId;
    return "";
  }, 120000, () => `generate response (${evidenceForFailure(lastPreflight, lastGenerateResponse)})`);
  // Reproduce the reported race: the user selects another model while this
  // job is still running. Gallery metadata must come from the immutable job,
  // never the mutable GenerateTab state at completion time.
  const alternateModel = await page.evaluate((submittedModel) => {
    const alternate = GenerateTab.state.allModels.find((model) => model.name !== submittedModel);
    if (!alternate) return "";
    GenerateTab.state.model = alternate.name;
    return alternate.name;
  }, MODEL);
  if (!alternateModel) throw new Error("model-swap gallery race gate requires a second model");
  let terminal = null;
  let last = "";
  await waitUntil(async () => {
    const res = await fetch(`${BASE_URL}/v1/job/${jobId}`);
    const body = await res.json();
    const cur = `${body.state} ${body.progress} ${body.step}/${body.total}`;
    if (cur !== last) { console.log(`job ${jobId} ${cur}`); last = cur; }
    if (["done", "failed", "cancelled", "interrupted"].includes(body.state)) {
      terminal = body;
      return true;
    }
    return false;
  }, 900000, "job terminal");
  if (!terminal || terminal.state !== "done") {
    await browser.close();
    throw new Error(`job ${jobId} did not finish cleanly: ${JSON.stringify(terminal)}`);
  }
  if (terminal.model !== MODEL) {
    await browser.close();
    throw new Error(`job ${jobId} recorded model=${terminal.model}, expected ${MODEL}`);
  }
  const galleryMetadata = await waitUntil(
    () => page.evaluate((id) => {
      const expectedFilename = `${id}.png`;
      const item = GenerateTab.state.gallery.find((candidate) => {
        try {
          return new URL(candidate.src, location.origin).searchParams.get("filename") === expectedFilename;
        } catch (_) {
          return false;
        }
      });
      if (!item) return null;
      return {
        model: item.model,
        width: item.width,
        height: item.height,
        steps: item.steps,
        cfg: item.cfg,
        guidance: item.guidance,
      };
    }, jobId),
    15000,
    `gallery metadata for ${jobId}`
  );
  await browser.close();
  if (galleryMetadata.model !== MODEL) {
    throw new Error(
      `gallery model race: submitted=${MODEL}, switched=${alternateModel}, recorded=${galleryMetadata.model}`
    );
  }
  if (galleryMetadata.width !== WIDTH || galleryMetadata.height !== HEIGHT) {
    throw new Error(
      `gallery size mismatch: ${galleryMetadata.width}x${galleryMetadata.height}, expected ${WIDTH}x${HEIGHT}`
    );
  }
  const expectedExecutedCfg = GUIDANCE != null ? GUIDANCE : CFG;
  if (galleryMetadata.steps !== STEPS || Number(galleryMetadata.cfg) !== expectedExecutedCfg) {
    throw new Error(
      `gallery sampler metadata mismatch: steps=${galleryMetadata.steps} cfg=${galleryMetadata.cfg}, expected ${STEPS}/${expectedExecutedCfg}`
    );
  }
  if (GUIDANCE != null && Number(galleryMetadata.guidance) !== GUIDANCE) {
    throw new Error(`gallery guidance mismatch: ${galleryMetadata.guidance}, expected ${GUIDANCE}`);
  }
  const outputPath = terminal.output_path;
  assertCanonicalOutputPath(outputPath);
  const img = decodePngRgb(outputPath);
  const stats = statsForImage(img);
  assertValidImage(stats, WIDTH, HEIGHT);
  console.log(JSON.stringify({
    schema: "serenity.browser_valid_image_gate.v1",
    job_id: jobId,
    output_path: outputPath,
    canonical_output_dir: CANONICAL_OUTPUT_DIR,
    model: MODEL,
    width: stats.width,
    height: stats.height,
    steps: STEPS,
    guidance: GUIDANCE,
    seed: SEED,
    gallery_model_after_swap: galleryMetadata.model,
    alternate_model_selected_while_running: alternateModel,
    avg_stddev: Number(stats.avgStddev.toFixed(3)),
    luminance_range: Number(stats.luminanceRange.toFixed(3)),
    edge_energy: Number(stats.edgeEnergy.toFixed(3)),
    color_bins: stats.colorBins,
  }, null, 2));
}

run().catch((err) => {
  console.error("browser generate valid image: FAIL");
  console.error(err.stack || err.message || String(err));
  process.exit(1);
});
