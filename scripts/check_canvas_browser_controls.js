#!/usr/bin/env node
"use strict";

const fs = require("fs");
const http = require("http");
const net = require("net");
const os = require("os");
const path = require("path");
const { spawn } = require("child_process");

let chromium;
try {
  ({ chromium } = require("playwright"));
} catch (err) {
  console.error("canvas browser controls: FAIL");
  console.error("Playwright is not installed for this Node environment.");
  console.error("Install outside the repo, for example:");
  console.error("  tmp=${TMPDIR:-/tmp}/mojodiffusion-playwright-tools");
  console.error("  mkdir -p \"$tmp\" && cd \"$tmp\" && npm init -y");
  console.error("  npm install --no-audit --no-fund playwright@latest");
  console.error("  NODE_PATH=\"$tmp/node_modules\" node scripts/check_canvas_browser_controls.js");
  process.exit(2);
}

const ROOT = path.resolve(__dirname, "..");
const SERVER_BIN = process.env.SERENITY_SERVER_BIN ||
  path.join(ROOT, "serenity-server/target/debug/serenity-server");
const WORKER_BIN = process.env.SERENITY_WORKER_BIN ||
  path.join(ROOT, "output/bin/serenity_worker_stub");
const CHROME_BIN = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ||
  firstExisting(["/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser"]);
const TINY_PNG = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=",
  "base64",
);

function firstExisting(candidates) {
  for (const candidate of candidates) {
    if (candidate && fs.existsSync(candidate)) return candidate;
  }
  return "";
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function getFreePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.on("error", reject);
    srv.listen(0, "127.0.0.1", () => {
      const port = srv.address().port;
      srv.close(() => resolve(port));
    });
  });
}

function httpJson(url) {
  return new Promise((resolve, reject) => {
    const req = http.get(url, { timeout: 500 }, (res) => {
      let body = "";
      res.setEncoding("utf8");
      res.on("data", (chunk) => { body += chunk; });
      res.on("end", () => {
        try {
          resolve({ status: res.statusCode, body: body ? JSON.parse(body) : null });
        } catch (err) {
          reject(err);
        }
      });
    });
    req.on("timeout", () => req.destroy(new Error("timeout")));
    req.on("error", reject);
  });
}

async function waitForHealth(baseUrl, proc) {
  let lastError = null;
  for (let i = 0; i < 80; i += 1) {
    if (proc.exitCode !== null) throw new Error(`server exited early: ${proc.exitCode}`);
    try {
      const res = await httpJson(`${baseUrl}/v1/health`);
      if (res.status === 200 && res.body && res.body.status === "ok") return res.body;
    } catch (err) {
      lastError = err;
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`server did not become healthy: ${lastError && lastError.message}`);
}

function parsePostJson(request) {
  const raw = request.postData() || "{}";
  try {
    return JSON.parse(raw);
  } catch (err) {
    throw new Error(`request body was not JSON: ${raw.slice(0, 200)}`);
  }
}

function forbiddenForwardedKeys(body) {
  const forbidden = [
    ["denoise", (v) => v !== 1.0],
    ["init_image", (v) => v !== ""],
    ["mask_image", (v) => v !== ""],
    ["lanpaint_mask_channel", (v) => v !== ""],
    ["refiner", (v) => v !== null],
    ["outpaint", (v) => v !== null],
    ["outpaint_enabled", (v) => v !== false],
    ["refiner_model", (v) => v !== ""],
    ["refiner_steps", (v) => v !== 0],
    ["refiner_cfg", (v) => v !== -1],
    ["refiner_method", (v) => v !== ""],
    ["refiner_control", (v) => v !== -1],
    ["refiner_tiling", (v) => v !== false],
    ["upscaler", (v) => v !== null],
    ["upscaler_model", (v) => v !== ""],
    ["upscale_by", (v) => v !== 1.0],
  ];
  return forbidden
    .filter(([key, bad]) => Object.prototype.hasOwnProperty.call(body, key) && bad(body[key]))
    .map(([key]) => key);
}

function assertWorkflowSubmitBody(body, label) {
  assert(body && typeof body === "object", `${label} body was not an object`);
  assert(body.workflow && typeof body.workflow === "object", `${label} body missing workflow graph`);
  assert(body.workflow_client === "serenity.canvas.generate_ws", `${label} body missing workflow client marker`);
  const flatKeys = [
    "model",
    "prompt",
    "negative",
    "width",
    "height",
    "steps",
    "cfg",
    "sampler",
    "scheduler",
    "init_image",
    "mask_image",
  ].filter((key) => Object.prototype.hasOwnProperty.call(body, key));
  assert(flatKeys.length === 0, `${label} body leaked flat fields beside workflow: ${flatKeys.join(", ")}`);
  const classes = Object.values(body.workflow)
    .map((node) => node && node.class_type)
    .filter(Boolean);
  for (const cls of ["CheckpointLoaderSimple", "CLIPTextEncode", "KSampler", "VAEDecode", "SaveImage"]) {
    assert(classes.includes(cls), `${label} workflow missing ${cls}`);
  }
}

function assertWorkflowParamsSubmitBody(body, label) {
  assert(body && typeof body === "object", `${label} body was not an object`);
  assert(body.workflow && typeof body.workflow === "object", `${label} body missing workflow envelope`);
  assert(body.workflow_client === "serenity.canvas.generate_ws", `${label} body missing workflow client marker`);
  const params = body.workflow.params;
  assert(params && typeof params === "object" && !Array.isArray(params), `${label} workflow missing params adapter`);
  const flatKeys = [
    "model",
    "prompt",
    "negative",
    "width",
    "height",
    "steps",
    "cfg",
    "sampler",
    "scheduler",
  ].filter((key) => Object.prototype.hasOwnProperty.call(body, key));
  assert(flatKeys.length === 0, `${label} body leaked flat fields beside workflow: ${flatKeys.join(", ")}`);
  return params;
}

function assertGridWorkflowSubmitBody(body, label) {
  assert(body && typeof body === "object", `${label} body was not an object`);
  assert(body.workflow && typeof body.workflow === "object", `${label} body missing workflow envelope`);
  assert(body.workflow_client === "serenity.canvas.grid_xyz", `${label} body missing grid workflow client marker`);
  const params = body.workflow.params;
  assert(params && typeof params === "object" && !Array.isArray(params), `${label} workflow missing params adapter`);
  const flatKeys = [
    "model",
    "prompt",
    "negative",
    "width",
    "height",
    "steps",
    "cfg",
    "sampler",
    "scheduler",
    "seed",
    "lora",
    "loras",
  ].filter((key) => Object.prototype.hasOwnProperty.call(body, key));
  assert(flatKeys.length === 0, `${label} body leaked base generate fields beside workflow: ${flatKeys.join(", ")}`);
  return params;
}

function workflowNodes(body) {
  return Object.values((body && body.workflow) || {});
}

function firstWorkflowNode(body, classType) {
  return workflowNodes(body).find((node) => node && node.class_type === classType) || null;
}

function clipTextValues(body) {
  return workflowNodes(body)
    .filter((node) => node && node.class_type === "CLIPTextEncode")
    .map((node) => node.inputs && node.inputs.text)
    .filter((text) => typeof text === "string");
}

function assertPromptSyntaxPatchedWorkflow(
  body,
  label,
  expectedSeed = 1234,
  promptPrefix = "browser ",
  promptSuffix = " smoke",
  minClipTextNodes = 2,
) {
  const sampler = firstWorkflowNode(body, "KSampler");
  assert(sampler && sampler.inputs, `${label} missing KSampler inputs`);
  assert(sampler.inputs.seed === expectedSeed, `${label} KSampler seed was not resolved to ${expectedSeed}`);
  const texts = clipTextValues(body);
  assert(texts.length >= minClipTextNodes, `${label} missing CLIPTextEncode prompts`);
  assert(texts[0].startsWith(promptPrefix), `${label} positive prompt missing prefix: ${texts[0]}`);
  assert(texts[0].endsWith(promptSuffix), `${label} positive prompt missing suffix: ${texts[0]}`);
  assert(!texts[0].includes("<random:"), `${label} positive prompt still contains random syntax: ${texts[0]}`);
}

async function waitUntil(predicate, timeoutMs, label) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (await predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error(`timed out waiting for ${label}`);
}

async function main() {
  assert(fs.existsSync(SERVER_BIN), `server binary not found: ${SERVER_BIN}`);
  assert(fs.existsSync(WORKER_BIN), `worker binary not found: ${WORKER_BIN}`);
  assert(CHROME_BIN && fs.existsSync(CHROME_BIN), "Chrome/Chromium executable not found");

  const port = await getFreePort();
  const baseUrl = `http://127.0.0.1:${port}`;
  const outDir = fs.mkdtempSync(path.join(os.tmpdir(), "serenity-browser-smoke-"));
  const serverLog = [];
  const server = spawn(
    SERVER_BIN,
    ["--worker", WORKER_BIN, "--kind", "stub", "--port", String(port), "--out-dir", outDir],
    { cwd: path.join(ROOT, "serenity-server"), stdio: ["ignore", "pipe", "pipe"] },
  );
  const keepLog = (chunk) => {
    serverLog.push(String(chunk));
    while (serverLog.join("").length > 6000) serverLog.shift();
  };
  server.stdout.on("data", keepLog);
  server.stderr.on("data", keepLog);

  let browser;
  try {
    const health = await waitForHealth(baseUrl, server);
    browser = await chromium.launch({
      headless: true,
      executablePath: CHROME_BIN,
      args: ["--no-sandbox", "--disable-dev-shm-usage"],
    });
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const pageErrors = [];
    const consoleErrors = [];
    const requestFailures = [];
    page.on("pageerror", (err) => pageErrors.push(err.message));
    page.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push(msg.text());
    });
    page.on("requestfailed", (req) => {
      const failure = req.failure();
      requestFailures.push(`${req.url()} ${failure ? failure.errorText : "failed"}`);
    });

    const preflightBodies = [];
    const generateBodies = [];
    const gridBodies = [];
    let rejectNextGenerate = false;
    await page.route("**/v1/preflight", async (route) => {
      preflightBodies.push(parsePostJson(route.request()));
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          schema: "serenity.generate.preflight.v1",
          admitted: true,
          backend: "zimage",
          same_gate_as_generate: true,
          block_profile: { profile: "browser_smoke" },
          artifact_profile: { ready: true },
        }),
      });
    });
    await page.route("**/v1/generate", async (route) => {
      generateBodies.push(parsePostJson(route.request()));
      if (rejectNextGenerate) {
        rejectNextGenerate = false;
        await route.fulfill({
          status: 400,
          contentType: "application/json",
          body: JSON.stringify({
            schema: "serenity.generate.error.v1",
            admitted: false,
            error: "simulated capability rejection",
            same_gate_as_preflight: true,
            enqueue_blocked: true,
            capability_profile: {
              schema: "serenity.capability_profile.v1",
              backend: "zimage",
              production_status: "admitted",
            },
          }),
        });
        return;
      }
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ job_id: "job-browser-smoke" }),
      });
    });
    await page.route("**/v1/grid", async (route) => {
      gridBodies.push(parsePostJson(route.request()));
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          grid_id: "grid-browser-smoke",
          cells: [{ job_id: "job-grid-smoke", state: "done" }],
          paths: [path.join(outDir, "grid-browser-smoke.png")],
        }),
      });
    });
    await page.route("**/out/gallery-smoke.png", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "image/png",
        body: TINY_PNG,
      });
    });

    await page.goto(baseUrl, { waitUntil: "commit", timeout: 10000 });
    await page.waitForSelector("#btn-generate", { state: "attached", timeout: 10000 });
    await page.waitForSelector("#gx-launch", { state: "attached", timeout: 10000 });
    await page.waitForFunction(() => {
      return window.Serenity && Serenity.ctx && Serenity.gridXyz &&
        Serenity.modules && Serenity.modules.generateWS && Serenity.workflows &&
        Serenity.ideogram4Nodes;
    });
    const refinerPanelState = await page.evaluate(() => {
      const panel = Array.from(document.querySelectorAll("#param-rail details"))
        .find((el) => ((el.querySelector("summary") || {}).textContent || "").trim() === "Refine / Upscale");
      if (!panel) return null;
      const checkbox = panel.querySelector('input[type="checkbox"]');
      const inputs = Array.from(panel.querySelectorAll("input, select"));
      return {
        panelDisabled: panel.classList.contains("ru-disabled"),
        checkboxDisabled: checkbox ? checkbox.disabled : null,
        disabledCount: inputs.filter((el) => el.disabled).length,
      };
    });
    assert(refinerPanelState, "Refine / Upscale panel not found");
    assert(refinerPanelState.panelDisabled === false, "Refine / Upscale panel is route-disabled");
    assert(refinerPanelState.checkboxDisabled === false, "Refine / Upscale checkbox is route-disabled");
    await page.evaluate(() => {
      window.__canvasBrowserSmoke = { unsupported: [], rejected: [] };
      Serenity.bus.on("generate:unsupported_canvas_conditioning", (payload) => {
        window.__canvasBrowserSmoke.unsupported.push(payload);
      });
      Serenity.bus.on("generate:rejected", (payload) => {
        window.__canvasBrowserSmoke.rejected.push(payload);
      });
    });

    const capabilityState = await page.evaluate(async () => {
      await Serenity.api.capabilities(true);
      const apiBackendKlein = Serenity.api.backendForModelName("flux-2-klein-base-9b_fp8_e4m3fn");
      const apiBackendSdxl = Serenity.api.backendForModelName("sd_xl_base_1.0");

      Serenity.set("params.width", 1024);
      Serenity.set("params.height", 1024);
      Serenity.set("params.aspect", "1:1");
      Serenity.set("params.scheduler", "simple");
      Serenity.set("params.model", "flux-2-klein-base-9b_fp8_e4m3fn");
      await new Promise((resolve) => setTimeout(resolve, 0));
      const klein = {
        backend: apiBackendKlein,
        width: Serenity.get("params.width"),
        height: Serenity.get("params.height"),
        scheduler: Serenity.get("params.scheduler"),
      };

      Serenity.set("params.scheduler", "simple");
      Serenity.set("params.model", "sd_xl_base_1.0");
      await new Promise((resolve) => setTimeout(resolve, 0));
      const sdxl = {
        backend: apiBackendSdxl,
        width: Serenity.get("params.width"),
        height: Serenity.get("params.height"),
        scheduler: Serenity.get("params.scheduler"),
      };
      return { klein, sdxl };
    });
    assert(
      capabilityState.klein.backend === "flux2",
      `Klein model was not classified as flux2: ${JSON.stringify(capabilityState)}`,
    );
    assert(
      capabilityState.klein.width === 512 && capabilityState.klein.height === 512,
      `Klein capability limits did not clamp to 512x512: ${JSON.stringify(capabilityState.klein)}`,
    );
    assert(
      capabilityState.klein.scheduler === "simple",
      `Klein scheduler should remain simple: ${JSON.stringify(capabilityState.klein)}`,
    );
    assert(
      capabilityState.sdxl.backend === "sdxl",
      `SDXL model was not classified as sdxl: ${JSON.stringify(capabilityState)}`,
    );
    assert(
      capabilityState.sdxl.width === 1024 && capabilityState.sdxl.height === 1024,
      `SDXL capability limits did not restore 1024x1024: ${JSON.stringify(capabilityState.sdxl)}`,
    );
    assert(
      capabilityState.sdxl.scheduler === "normal",
      `SDXL capability scheduler did not normalize to normal: ${JSON.stringify(capabilityState.sdxl)}`,
    );

    await page.evaluate(() => {
      Serenity.set("params.model", "ideogram4");
      Serenity.set("params.negative", "stale negative should clear");
    });
    await page.waitForFunction(() => Serenity.get("params.negative") === "");
    const negativeState = await page.locator(".pb-neg").evaluate((el) => ({
      disabled: el.disabled,
      value: el.value,
      placeholder: el.getAttribute("placeholder") || "",
    }));
    assert(negativeState.disabled === true, "negative prompt did not disable for Ideogram");
    assert(negativeState.value === "", "negative prompt textarea did not clear for Ideogram");

    await page.evaluate(() => {
      Serenity.set("params.model", "Z-Image (base)");
      Serenity.set("params.prompt", "legacy <random:green|gold> smoke");
      Serenity.set("params.negative", "");
      Serenity.set("params.seed", 4321);
      Serenity.set("params.refiner", null);
      Serenity.set("params.upscaler", null);
      Serenity.set("params.outpaint", null);
      Serenity.set("params.outpaint_enabled", false);
      Serenity.set("layers", []);
      window.__canvasBrowserSmoke.rejected = [];
      window.__canvasBrowserSmoke.unsupported = [];
    });
    rejectNextGenerate = true;
    const legacyResult = await page.evaluate(async () => {
      try {
        await Serenity.api.submitPrompt(null, Serenity.ctx && Serenity.ctx.clientId);
        return { ok: true };
      } catch (err) {
        return {
          ok: false,
          name: err && err.name,
          message: err && err.message,
          response: (err && (err.generateError || err.response)) || null,
        };
      }
    });
    assert(legacyResult.name === "GenerateRejected", `legacy fallback did not surface GenerateRejected: ${JSON.stringify(legacyResult)}`);
    assert(preflightBodies.length === 1, "legacy fallback did not call /v1/preflight once");
    assert(generateBodies.length === 1, "legacy fallback did not call /v1/generate once");
    assert(
      JSON.stringify(preflightBodies[0]) === JSON.stringify(generateBodies[0]),
      "legacy fallback preflight and generate bodies differ",
    );
    const legacyParams = assertWorkflowParamsSubmitBody(generateBodies[0], "legacy fallback generate");
    assert(legacyParams.model === "z-image", `legacy fallback model was not normalized: ${legacyParams.model}`);
    assert(legacyParams.prompt.startsWith("legacy "), `legacy fallback prompt prefix mismatch: ${legacyParams.prompt}`);
    assert(legacyParams.prompt.endsWith(" smoke"), `legacy fallback prompt suffix mismatch: ${legacyParams.prompt}`);
    assert(!legacyParams.prompt.includes("<random:"), `legacy fallback prompt syntax was not resolved: ${legacyParams.prompt}`);
    assert(
      typeof legacyParams.prompt_raw === "string" && legacyParams.prompt_raw.includes("<random:"),
      "legacy fallback did not preserve prompt_raw",
    );
    assert(legacyParams.seed === 4321, "legacy fallback seed was not pinned");

    preflightBodies.length = 0;
    generateBodies.length = 0;
    await page.evaluate(() => {
      Serenity.set("progress.running", false);
      const btn = document.querySelector("#btn-generate");
      if (btn) {
        btn.disabled = false;
        btn.classList.remove("is-busy");
      }
      window.__canvasBrowserSmoke.rejected = [];
      window.__canvasBrowserSmoke.unsupported = [];
    });

    await page.evaluate(() => {
      Serenity.set("params.model", "Z-Image (base)");
      Serenity.set("params.prompt", "browser stale smoke");
      Serenity.set("params.seed", 1234);
      Serenity.set("params.refiner", { enabled: true, model: "stale-refiner" });
    });
    rejectNextGenerate = true;
    await page.locator("#btn-generate").click();
    await waitUntil(
      () => preflightBodies.length >= 1 && generateBodies.length >= 1,
      3000,
      "stale refiner workflow submit",
    );
    const stalePreflight = preflightBodies[preflightBodies.length - 1];
    const staleGenerate = generateBodies[generateBodies.length - 1];
    assert(
      JSON.stringify(stalePreflight) === JSON.stringify(staleGenerate),
      "stale refiner preflight and generate bodies differ",
    );
    assertWorkflowSubmitBody(staleGenerate, "stale refiner generate");
    assertPromptSyntaxPatchedWorkflow(staleGenerate, "stale refiner generate", 1234, "browser ", " smoke");
    const staleCheckpoint = firstWorkflowNode(staleGenerate, "CheckpointLoaderSimple");
    assert(
      staleCheckpoint && staleCheckpoint.inputs && staleCheckpoint.inputs.ckpt_name === "z-image",
      `display model label leaked into generate workflow: ${JSON.stringify(staleCheckpoint && staleCheckpoint.inputs)}`,
    );
    const staleRefinerIntent = firstWorkflowNode(staleGenerate, "SerenityRefinerUpscaleIntent");
    assert(
      staleRefinerIntent && staleRefinerIntent.inputs && staleRefinerIntent.inputs.refiner_model === "stale-refiner",
      `stale refiner state did not become workflow intent: ${JSON.stringify(staleRefinerIntent && staleRefinerIntent.inputs)}`,
    );
    assert(forbiddenForwardedKeys(staleGenerate).length === 0, "stale refiner leaked as flat generate params");
    const staleEvents = await page.evaluate(() => window.__canvasBrowserSmoke.unsupported.length);
    assert(staleEvents === 0, "stale UI state was blocked by browser instead of workflow preflight");

    preflightBodies.length = 0;
    generateBodies.length = 0;
    await page.evaluate(() => {
      Serenity.set("progress.running", false);
      Serenity.set("params.refiner", null);
      const btn = document.querySelector("#btn-generate");
      if (btn) {
        btn.disabled = false;
        btn.classList.remove("is-busy");
      }
      window.__canvasBrowserSmoke.rejected = [];
      window.__canvasBrowserSmoke.unsupported = [];
    });

    await page.evaluate(() => {
      Serenity.set("params.model", "z-image");
      Serenity.set("params.prompt", "browser raster smoke");
      Serenity.set("params.seed", 1234);
      Serenity.set("params.negative", "");
      Serenity.set("params.width", 1024);
      Serenity.set("params.height", 1024);
      Serenity.set("params.steps", 1);
      Serenity.set("params.cfg", 4.0);
      Serenity.set("params.sampler", "euler");
      Serenity.set("params.scheduler", "simple");
      Serenity.set("layers", [{
        id: "init-smoke",
        name: "init smoke",
        type: "raster",
        visible: true,
        dataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=",
      }]);
    });
    rejectNextGenerate = true;
    await page.locator("#btn-generate").click();
    await waitUntil(
      () => preflightBodies.length >= 1 && generateBodies.length >= 1,
      3000,
      "raster workflow submit",
    );
    const rasterGenerate = generateBodies[generateBodies.length - 1];
    assertWorkflowSubmitBody(rasterGenerate, "raster generate");
    const rasterClasses = workflowNodes(rasterGenerate).map((node) => node && node.class_type);
    assert(rasterClasses.includes("LoadImage"), "raster workflow missing LoadImage node");
    assert(rasterClasses.includes("VAEEncode"), "raster workflow missing VAEEncode node");
    const loadImage = firstWorkflowNode(rasterGenerate, "LoadImage");
    assert(
      loadImage && loadImage.inputs && typeof loadImage.inputs.image === "string" &&
        loadImage.inputs.image.includes("/uploads/"),
      `raster LoadImage did not use uploaded path: ${JSON.stringify(loadImage && loadImage.inputs)}`,
    );

    preflightBodies.length = 0;
    generateBodies.length = 0;
    await page.evaluate((imageUrl) => {
      Serenity.set("progress.running", false);
      Serenity.set("params.model", "z-image");
      Serenity.set("params.prompt", "gallery source should be reused");
      Serenity.set("params.negative", "");
      Serenity.set("params.width", 640);
      Serenity.set("params.height", 768);
      Serenity.set("params.steps", 1);
      Serenity.set("params.cfg", 4.0);
      Serenity.set("params.sampler", "euler");
      Serenity.set("params.scheduler", "simple");
      Serenity.set("params.seed", 1234);
      Serenity.set("params.denoise", 1.0);
      Serenity.set("canvas.bbox", { x: 0, y: 0, width: 640, height: 768 });
      Serenity.set("layers", []);
      window.__canvasBrowserSmoke.rejected = [];
      window.__canvasBrowserSmoke.unsupported = [];
      const btn = document.querySelector("#btn-generate");
      if (btn) {
        btn.disabled = false;
        btn.classList.remove("is-busy");
      }
      Serenity.bus.emit("result:ready", {
        id: "job-gallery-smoke",
        filename: "gallery-smoke.png",
        url: imageUrl,
        params: {
          model: "z-image",
          prompt: "gallery <random:white|black> source",
          negative: "",
          width: 512,
          height: 512,
          steps: 1,
          cfg: 4.0,
          sampler: "euler",
          scheduler: "simple",
          seed: 9876,
          denoise: 1.0,
        },
      });
    }, `${baseUrl}/out/gallery-smoke.png`);
    const galleryCard = page.locator('#gallery .gal-card[data-id="job-gallery-smoke"]');
    await galleryCard.waitFor({ state: "visible", timeout: 5000 });
    await waitUntil(
      () => page.locator('#gallery .gal-card[data-id="job-gallery-smoke"] .gp-check').count().then((n) => n > 0),
      3000,
      "gallery pro decoration",
    );
    await galleryCard.click({ button: "right" });
    await page.waitForSelector(".gp-ctxmenu", { state: "visible", timeout: 3000 });
    const menuLabels = await page.locator(".gp-ctxmenu .gp-ctx-item").evaluateAll((buttons) =>
      buttons.map((button) => button.textContent || ""),
    );
    assert(menuLabels.includes("Send to img2img"), `gallery menu missing img2img: ${menuLabels.join(", ")}`);
    assert(menuLabels.includes("Upscale 2×"), `gallery menu missing upscale: ${menuLabels.join(", ")}`);
    rejectNextGenerate = true;
    await page.locator(".gp-ctxmenu .gp-ctx-item", { hasText: "Upscale 2×" }).click();
    await waitUntil(
      () => preflightBodies.length >= 1 && generateBodies.length >= 1,
      5000,
      "gallery upscale workflow submit",
    );
    const galleryUpscaleGenerate = generateBodies[generateBodies.length - 1];
    assertWorkflowSubmitBody(galleryUpscaleGenerate, "gallery upscale generate");
    const galleryClasses = workflowNodes(galleryUpscaleGenerate).map((node) => node && node.class_type);
    assert(galleryClasses.includes("LoadImage"), "gallery upscale workflow missing LoadImage node");
    assert(galleryClasses.includes("VAEEncode"), "gallery upscale workflow missing VAEEncode node");
    const galleryUpscaleState = await page.evaluate(() => ({
      width: Serenity.get("params.width"),
      height: Serenity.get("params.height"),
      bbox: Serenity.get("canvas.bbox"),
    }));
    assert(
      galleryUpscaleState.width === 1024 && galleryUpscaleState.height === 1024 &&
        galleryUpscaleState.bbox && galleryUpscaleState.bbox.width === 1024 &&
        galleryUpscaleState.bbox.height === 1024,
      `gallery upscale dimensions were not doubled after source params: ${JSON.stringify(galleryUpscaleState)}`,
    );
    const gallerySampler = firstWorkflowNode(galleryUpscaleGenerate, "KSampler");
    assert(
      gallerySampler && gallerySampler.inputs && gallerySampler.inputs.denoise === 0.4,
      `gallery upscale did not set light denoise: ${JSON.stringify(gallerySampler && gallerySampler.inputs)}`,
    );

    preflightBodies.length = 0;
    generateBodies.length = 0;
    await page.evaluate(() => {
      Serenity.set("progress.running", false);
      Serenity.set("params.init_image", "");
      Serenity.set("params.mask_image", "");
      Serenity.set("params.denoise", 1.0);
      Serenity.set("layers", []);
      const btn = document.querySelector("#btn-generate");
      if (btn) {
        btn.disabled = false;
        btn.classList.remove("is-busy");
      }
      window.__canvasBrowserSmoke.rejected = [];
      window.__canvasBrowserSmoke.unsupported = [];
    });

    await page.evaluate(() => {
      Serenity.set("params.refiner", null);
      Serenity.set("params.upscaler", null);
      Serenity.set("params.outpaint", null);
      Serenity.set("params.outpaint_enabled", false);
      Serenity.set("params.init_image", "");
      Serenity.set("params.mask_image", "");
      Serenity.set("params.denoise", 1.0);
      Serenity.set("params.prompt", "browser <random:red|blue> smoke");
      Serenity.set("params.seed", 1234);
      Serenity.set("layers", []);
    });
    rejectNextGenerate = true;
    await page.locator("#btn-generate").click();
    await waitUntil(
      async () => {
        const state = await page.evaluate(() => ({
          preflightCount: window.__preflightCount || 0,
          rejected:
            window.__canvasBrowserSmoke && window.__canvasBrowserSmoke.rejected.length,
        }));
        return preflightBodies.length >= 1 && generateBodies.length >= 1 && state.rejected >= 1;
      },
      3000,
      "structured generate rejection",
    );
    const rejectedState = await page.evaluate(() => ({
      status:
        document.querySelector("#gen-queue .gen-status") &&
        document.querySelector("#gen-queue .gen-status").textContent,
      event: window.__canvasBrowserSmoke.rejected[0],
    }));
    assert(
      /generate blocked: simulated capability rejection/.test(rejectedState.status || ""),
      `structured generate rejection did not show blocked status: ${rejectedState.status}`,
    );
    assert(
      rejectedState.event &&
        rejectedState.event.response &&
        rejectedState.event.response.schema === "serenity.generate.error.v1",
      "structured generate rejection event did not preserve response schema",
    );
    assert(
      rejectedState.event.response.capability_profile &&
        rejectedState.event.response.capability_profile.backend === "zimage",
      "structured generate rejection event did not preserve capability_profile",
    );
    assert(
      JSON.stringify(preflightBodies[0]) === JSON.stringify(generateBodies[0]),
      "rejected preflight and generate bodies differ",
    );
    assertWorkflowSubmitBody(preflightBodies[0], "rejected preflight");
    assertWorkflowSubmitBody(generateBodies[0], "rejected generate");
    assertPromptSyntaxPatchedWorkflow(preflightBodies[0], "rejected preflight");
    assertPromptSyntaxPatchedWorkflow(generateBodies[0], "rejected generate");

    await page.locator("#btn-generate").click();
    try {
      await waitUntil(() => preflightBodies.length >= 2 && generateBodies.length >= 2, 3000, "sanitized generate requests");
    } catch (err) {
      const diag = await page.evaluate(() => ({
        status: document.querySelector("#gen-queue .gen-status") &&
          document.querySelector("#gen-queue .gen-status").textContent,
        buttonDisabled: !!(document.querySelector("#btn-generate") &&
          document.querySelector("#btn-generate").disabled),
        buttonBusy: !!(document.querySelector("#btn-generate") &&
          document.querySelector("#btn-generate").classList.contains("is-busy")),
        params: Serenity && Serenity.state ? Serenity.state.params : null,
        layers: Serenity && Serenity.state ? Serenity.state.layers : null,
        unsupported: window.__canvasBrowserSmoke && window.__canvasBrowserSmoke.unsupported,
      }));
      throw new Error(`${err.message}; diagnostic=${JSON.stringify(diag)}`);
    }
    assert(preflightBodies.length === 2, "sanitized generate did not call /v1/preflight twice");
    assert(generateBodies.length === 2, "sanitized generate did not call /v1/generate twice");
    assert(
      JSON.stringify(preflightBodies[1]) === JSON.stringify(generateBodies[1]),
      "preflight and generate bodies differ",
    );
    assertWorkflowSubmitBody(preflightBodies[1], "sanitized preflight");
    assertWorkflowSubmitBody(generateBodies[1], "sanitized generate");
    assertPromptSyntaxPatchedWorkflow(preflightBodies[1], "sanitized preflight");
    assertPromptSyntaxPatchedWorkflow(generateBodies[1], "sanitized generate");
    const badGenerate = forbiddenForwardedKeys(generateBodies[1]);
    assert(badGenerate.length === 0, `generate forwarded disabled fields: ${badGenerate.join(", ")}`);

    await page.evaluate(() => {
      Serenity.set("params.init_image", "/tmp/stale-init.png");
      Serenity.set("params.mask_image", "/tmp/stale-mask.png");
      Serenity.set("params.refiner", { enabled: true, model: "stale-refiner" });
      Serenity.set("params.upscaler", { enabled: true, model: "stale-upscaler", factor: 2 });
      Serenity.set("params.outpaint", { left: 64, right: 0, top: 0, bottom: 0 });
      Serenity.set("params.outpaint_enabled", true);
    });
    await page.locator("#gx-launch").click();
    await page.locator("#gx-overlay .gx-axis select").first().selectOption("seed");
    await page.locator("#gx-overlay .gx-axis textarea").first().fill("1");
    await page.locator("#gx-run").click();
    await page.waitForFunction(() => document.querySelector("#gx-run") && !document.querySelector("#gx-run").disabled);
    assert(gridBodies.length === 1, "grid run did not POST /v1/grid once");
    const gridBody = gridBodies[0];
    const gridParams = assertGridWorkflowSubmitBody(gridBody, "grid request");
    const gridForbidden = [
      "denoise",
      "init_image",
      "mask_image",
      "lanpaint_mask_channel",
      "refiner",
      "outpaint",
      "outpaint_enabled",
      "upscaler",
      "upscale_by",
    ].filter((key) => Object.prototype.hasOwnProperty.call(gridBody, key));
    assert(gridForbidden.length === 0, `grid forwarded disabled fields: ${gridForbidden.join(", ")}`);
    assert(gridBody.x_axis === "seed" && gridBody.x_values === "1", "grid axis body mismatch");
    assert(gridParams.model === "z-image", `grid workflow params did not preserve model: ${JSON.stringify(gridParams.model)}`);
    assert(
      typeof gridParams.prompt === "string" && gridParams.prompt.startsWith("browser "),
      "grid workflow params did not preserve prompt",
    );
    assert(!gridParams.prompt.includes("<random:"), "grid workflow prompt syntax was not resolved");
    assert(
      typeof gridParams.prompt_raw === "string" && gridParams.prompt_raw.includes("<random:"),
      "grid workflow params did not preserve raw prompt syntax",
    );
    assert(gridParams.seed === 1234, "grid workflow params seed was not pinned to the resolver seed");

    const workflowPreflightBefore = preflightBodies.length;
    const workflowGenerateBefore = generateBodies.length;
    await page.evaluate(() => {
      Serenity.nav.show("workflows");
      Serenity.set("params.model", "z-image");
      Serenity.set("params.negative", "");
      Serenity.set("params.width", 1024);
      Serenity.set("params.height", 1024);
      Serenity.set("params.steps", 1);
      Serenity.set("params.seed", 5678);
      Serenity.set("params.cfg", 4.0);
      Serenity.set("params.sampler", "euler");
      Serenity.set("params.scheduler", "simple");
    });
    await page.waitForSelector("#view-workflows.show #wf-bar .wf-prompt", { state: "visible", timeout: 5000 });
    await page.locator("#wf-bar .wf-prompt").fill("workflow tab <random:cyan|magenta> smoke");
    await page.locator("#wf-bar .btn-primary").click();
    await waitUntil(
      () => preflightBodies.length >= workflowPreflightBefore + 1 &&
        generateBodies.length >= workflowGenerateBefore + 1,
      3000,
      "workflow tab submit",
    );
    const workflowPreflight = preflightBodies[preflightBodies.length - 1];
    const workflowGenerate = generateBodies[generateBodies.length - 1];
    assert(
      JSON.stringify(workflowPreflight) === JSON.stringify(workflowGenerate),
      "workflow tab preflight and generate bodies differ",
    );
    assertWorkflowSubmitBody(workflowPreflight, "workflow tab preflight");
    assertWorkflowSubmitBody(workflowGenerate, "workflow tab generate");
    assert(
      workflowNodes(workflowGenerate).some((node) => node && node.class_type === "ConditioningZeroOut"),
      "workflow tab graph did not synthesize zero negative conditioning",
    );
    assertPromptSyntaxPatchedWorkflow(
      workflowGenerate,
      "workflow tab generate",
      5678,
      "workflow tab ",
      " smoke",
      1,
    );

    const ideogramPreflightBefore = preflightBodies.length;
    const ideogramGenerateBefore = generateBodies.length;
    await page.evaluate(() => {
      const queue = document.querySelector("#wf-bar .btn-primary");
      if (queue) {
        queue.disabled = false;
        queue.textContent = "▶ Generate";
      }
      Serenity.set("params.seed", 2468);
      Serenity.set("params.prompt", "");
      Serenity.set("params.negative", "");
      const node = Serenity.workflows.addNodeByType(Serenity.ideogram4Nodes.BBOX_TYPE, 80, 80);
      const promptJson = {
        high_level_description: "ideogram workflow bbox smoke",
        style_description: {
          aesthetics: "clean product mockup",
          lighting: "soft studio light",
          medium: "painting",
        },
        compositional_deconstruction: {
          background: "muted studio wall",
          elements: [{
            type: "obj",
            bbox: [120, 180, 760, 820],
            desc: "package face with label area",
            color_palette: ["#6C8CFF"],
          }],
        },
      };
      const data = node.getAttr("data") || {};
      data.width = 1024;
      data.height = 1024;
      data.high_level_description = promptJson.high_level_description;
      data.background = promptJson.compositional_deconstruction.background;
      data.aesthetics = promptJson.style_description.aesthetics;
      data.lighting = promptJson.style_description.lighting;
      data.medium = promptJson.style_description.medium;
      data.elements = [{
        id: 1,
        type: "obj",
        desc: "package face with label area",
        color: "#6c8cff",
        x: 0.18,
        y: 0.12,
        w: 0.64,
        h: 0.64,
      }];
      data.caption = JSON.stringify(promptJson);
      node.setAttr("data", data);
    });
    await page.locator("#wf-bar .btn-primary").click();
    await waitUntil(
      () => preflightBodies.length >= ideogramPreflightBefore + 1 &&
        generateBodies.length >= ideogramGenerateBefore + 1,
      3000,
      "ideogram workflow bbox submit",
    );
    const ideogramPreflight = preflightBodies[preflightBodies.length - 1];
    const ideogramGenerate = generateBodies[generateBodies.length - 1];
    assert(
      JSON.stringify(ideogramPreflight) === JSON.stringify(ideogramGenerate),
      "ideogram workflow preflight and generate bodies differ",
    );
    const ideogramParams = assertWorkflowParamsSubmitBody(ideogramGenerate, "ideogram workflow generate");
    assert(ideogramParams.model === "ideogram4", "ideogram workflow did not select ideogram4");
    assert(ideogramParams.seed === 2468, "ideogram workflow seed was not resolved to 2468");
    assert(ideogramParams.negative === "", "ideogram workflow should submit empty negative prompt");
    assert(
      ideogramParams.prompt_json &&
        ideogramParams.prompt_json.compositional_deconstruction &&
        Array.isArray(ideogramParams.prompt_json.compositional_deconstruction.elements),
      "ideogram workflow missing prompt_json elements",
    );
    const ideogramBox = ideogramParams.prompt_json.compositional_deconstruction.elements[0].bbox;
    assert(
      JSON.stringify(ideogramBox) === JSON.stringify([120, 180, 760, 820]),
      `ideogram workflow bbox changed: ${JSON.stringify(ideogramBox)}`,
    );

    assert(pageErrors.length === 0, `browser page errors: ${pageErrors.join(" | ")}`);
    assert(requestFailures.length === 0, `browser request failures: ${requestFailures.join(" | ")}`);
    const bootFailures = consoleErrors.filter((line) => /\[boot\] init failed/.test(line));
    assert(bootFailures.length === 0, `boot failures: ${bootFailures.join(" | ")}`);

    console.log("canvas browser controls: PASS");
    console.log(JSON.stringify({
      base_url: baseUrl,
      health,
      generate_body_keys: Object.keys(generateBodies[0]).sort(),
      workflow_tab_body_keys: Object.keys(workflowGenerate).sort(),
      ideogram_workflow_body_keys: Object.keys(ideogramGenerate).sort(),
      grid_body_keys: Object.keys(gridBodies[0]).sort(),
    }, null, 2));
  } finally {
    if (browser) await browser.close().catch(() => {});
    server.kill("SIGTERM");
    await new Promise((resolve) => {
      const timer = setTimeout(() => {
        server.kill("SIGKILL");
        resolve();
      }, 3000);
      server.on("exit", () => {
        clearTimeout(timer);
        resolve();
      });
    });
    if (process.exitCode) {
      console.error("--- server log tail ---");
      console.error(serverLog.join("").slice(-6000));
    }
  }
}

main().catch((err) => {
  console.error("canvas browser controls: FAIL");
  console.error(err && err.stack ? err.stack : String(err));
  process.exitCode = 1;
});
