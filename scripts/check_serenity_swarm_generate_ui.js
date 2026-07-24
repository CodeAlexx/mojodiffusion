#!/usr/bin/env node
"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");

function loadChromium() {
  try {
    return require("playwright").chromium;
  } catch (_) { /* tooling may live in the npx cache */ }
  const roots = [];
  const npxRoot = path.join(os.homedir(), ".npm", "_npx");
  try {
    for (const entry of fs.readdirSync(npxRoot)) {
      roots.push(path.join(npxRoot, entry, "node_modules", "playwright"));
    }
  } catch (_) { /* no npx cache */ }
  for (const candidate of roots) {
    try {
      return require(candidate).chromium;
    } catch (_) { /* continue */ }
  }
  return null;
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function selectModel(page, fragment) {
  await page.locator("#gen-model-search").click();
  await page.locator("#gen-model-search").fill(fragment);
  const item = page.locator(".gen-model-dropdown-item").filter({ hasText: fragment }).first();
  await item.waitFor({ state: "visible" });
  await item.click();
}

async function run() {
  const chromium = loadChromium();
  if (!chromium) throw new Error("Playwright is not installed");
  const baseUrl = process.env.SERENITY_BASE_URL || "http://127.0.0.1:7811";
  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE || "/usr/bin/google-chrome",
    args: ["--no-sandbox", "--disable-dev-shm-usage"],
  });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  const errors = [];
  const preflightBodies = [];
  const generateBodies = [];
  let promptPosts = 0;
  page.on("pageerror", (error) => errors.push(String(error)));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  await page.route("**/v1/preflight", async (route) => {
    preflightBodies.push(JSON.parse(route.request().postData() || "{}"));
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ admitted: true, error: "" }),
    });
  });
  await page.route("**/v1/generate", async (route) => {
    generateBodies.push(JSON.parse(route.request().postData() || "{}"));
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ job_id: "job-swarm-ui-gate" }),
    });
  });
  page.on("request", (request) => {
    if (request.method() === "POST" && new URL(request.url()).pathname === "/prompt") {
      promptPosts += 1;
    }
  });

  try {
    await page.goto(baseUrl, { waitUntil: "networkidle" });
    await page.locator('.nav-btn[data-tab="generate"]').click();
    await page.waitForSelector(".gen-swarm-layout");
    await page.waitForFunction(() =>
      window.GenerateTab && GenerateTab.state.allModels && GenerateTab.state.allModels.length > 0
    );

    const surface = await page.evaluate(() => ({
      gridColumns: getComputedStyle(document.querySelector(".gen-swarm-layout")).gridTemplateColumns,
      promptDock: !!document.querySelector(".gen-swarm-prompt-dock"),
      batchPanel: !!document.querySelector(".gen-swarm-batch-panel"),
      libraryTabs: Array.from(document.querySelectorAll(".gen-library-tab")).map((node) => node.textContent.trim()),
      groupTitles: Array.from(document.querySelectorAll(".gen-swarm-group-title")).map((node) => node.textContent.trim()),
      visibleVideo: Array.from(document.querySelectorAll("#panel-generate [id*=video]"))
        .filter((node) => node.offsetParent !== null && getComputedStyle(node).display !== "none")
        .map((node) => node.id),
      visibleInpaintText: document.querySelector("#panel-generate").innerText.toLowerCase().includes("inpaint"),
      videoModels: GenerateTab.state.allModels
        .map((model) => model.name)
        .filter((name) => /(?:ltx|wan|bernini|scail)/i.test(name)),
      encoderOrEditModels: GenerateTab.state.allModels
        .map((model) => model.name)
        .filter((name) => /(?:qwen[_-]?2\\.5[_-]?vl|qwen.*image.*edit)/i.test(name)),
    }));
    assert(surface.gridColumns.split(" ").length === 3, `expected 3 workspace columns: ${surface.gridColumns}`);
    assert(surface.promptDock && surface.batchPanel, "prompt dock or current-batch panel missing");
    assert(
      ["History", "Presets", "Models", "LoRAs"].every((name) => surface.libraryTabs.includes(name)),
      `library tabs incomplete: ${surface.libraryTabs.join(", ")}`
    );
    assert(
      ["Core Parameters?", "Resolution?", "Sampling?", "LoRAs?"].every((name) => surface.groupTitles.includes(name)),
      `parameter groups incomplete: ${surface.groupTitles.join(", ")}`
    );
    assert(surface.visibleVideo.length === 0, `video controls visible: ${surface.visibleVideo.join(", ")}`);
    assert(!surface.visibleInpaintText, "inpainting controls leaked into Generate");
    assert(surface.videoModels.length === 0, `video models leaked into image picker: ${surface.videoModels.join(", ")}`);
    assert(
      surface.encoderOrEditModels.length === 0,
      `encoder/edit artifacts leaked into text-to-image picker: ${surface.encoderOrEditModels.join(", ")}`
    );

    await selectModel(page, "zimage_base");
    const zimage = await page.evaluate(() => ({
      initVisible: document.querySelector("#gen-init-section").offsetParent !== null,
      variationVisible: document.querySelector("#gen-variation-section").offsetParent !== null,
      samplers: Array.from(document.querySelectorAll("#gen-sampler option")).map((node) => node.value),
      schedulers: Array.from(document.querySelectorAll("#gen-scheduler option")).map((node) => node.value),
      sizes: Array.from(document.querySelectorAll("#gen-aspect-dropdown option")).map((node) => node.value),
    }));
    assert(zimage.initVisible, "Z-Image init-image controls are not visible");
    assert(zimage.variationVisible, "Z-Image variation controls are not visible");
    assert(
      ["euler", "flowmatch_euler", "dpmpp_2m", "uni_pc", "uni_pc_bh2"]
        .every((value) => zimage.samplers.includes(value)),
      `Z-Image sampler inventory drifted: ${zimage.samplers.join(", ")}`
    );
    assert(
      ["simple", "sgm_uniform"].every((value) => zimage.schedulers.includes(value)),
      `Z-Image scheduler inventory drifted: ${zimage.schedulers.join(", ")}`
    );
    assert(zimage.sizes.includes("1024×1024"), `Z-Image compiled shapes missing: ${zimage.sizes.join(", ")}`);

    const workflowSampling = await page.evaluate(() => {
      const graph = WorkflowBuilder.build({
        model: GenerateTab.state.model,
        prompt: "workflow sampler sync",
        negPrompt: "",
        width: 1024,
        height: 1024,
        steps: 12,
        cfg: 4,
        seed: 42,
        sampler: "uni_pc_bh2",
        scheduler: "sgm_uniform",
        noiseScheduler: "sgm_uniform",
        loras: [],
      });
      const sampler = Object.values(graph).find((node) =>
        node.class_type === "KSampler" || node.class_type === "KSamplerAdvanced"
      );
      return sampler && {
        sampler: sampler.inputs.sampler_name,
        scheduler: sampler.inputs.scheduler,
      };
    });
    assert(workflowSampling, "Generate parameters did not build a workflow sampler node");
    assert(
      workflowSampling.sampler === "uni_pc_bh2" && workflowSampling.scheduler === "sgm_uniform",
      `workflow sampler handoff drifted: ${JSON.stringify(workflowSampling)}`
    );

    await page.locator("#gen-param-filter").fill("sampler");
    assert(await page.locator("#gen-sampling-header").isVisible(), "parameter filter hid Sampling");
    assert(!(await page.locator("#gen-core-header").isVisible()), "parameter filter did not hide unrelated groups");
    await page.locator("#gen-param-filter").fill("");

    await page.evaluate(() => {
      GenerateTab.state.prompt = "Swarm style request contract probe";
      GenerateTab.state.variationSeed = 9876;
      GenerateTab.state.variationStrength = 0.25;
      GenerateTab.state.initImagePath = "/tmp/serenity-init-contract.png";
      GenerateTab.state.creativity = 0.6;
      document.querySelector("#gen-prompt").value = GenerateTab.state.prompt;
      document.querySelector("#gen-variation-seed").value = "9876";
      document.querySelector("#gen-variation-strength").value = "0.25";
      document.querySelector("#gen-creativity").value = "0.6";
    });
    await page.locator("#gen-btn").click();
    await page.waitForFunction(() => window.__neverSet === undefined);
    await page.waitForTimeout(100);
    assert(preflightBodies.length === 1, `expected one preflight request, got ${preflightBodies.length}`);
    assert(generateBodies.length === 1, `expected one flat generate request, got ${generateBodies.length}`);
    const body = generateBodies[0];
    assert(
      JSON.stringify(preflightBodies[0]) === JSON.stringify(body),
      "preflight and generate request bodies differ"
    );
    assert(body.model.includes("zimage"), `wrong request model: ${body.model}`);
    assert(body.prompt === "Swarm style request contract probe", `wrong request prompt: ${body.prompt}`);
    assert(body.sampler && body.scheduler, "sampler/scheduler missing from request");
    assert(body.variation_seed === 9876 && body.variation_strength === 0.25, "variation fields missing");
    assert(body.init_image === "/tmp/serenity-init-contract.png" && body.creativity === 0.6, "init-image fields missing");
    assert(body.mask_image === undefined, "mask/inpaint field leaked into request");
    assert(promptPosts === 0, "Generate still routed through the legacy /prompt workflow adapter");
    assert(errors.length === 0, `browser errors: ${errors.join(" | ")}`);
    console.log("serenity swarm generate ui: PASS");
    console.log(JSON.stringify({
      model: body.model,
      sampler: body.sampler,
      scheduler: body.scheduler,
      variation: body.variation_strength,
      initImage: true,
      libraryTabs: surface.libraryTabs,
    }, null, 2));
  } finally {
    await browser.close();
  }
}

run().catch((error) => {
  console.error("serenity swarm generate ui: FAIL");
  console.error(error && error.stack ? error.stack : String(error));
  process.exit(1);
});
