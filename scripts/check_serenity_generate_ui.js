#!/usr/bin/env node
"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawnSync } = require("child_process");

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
  const videoBodies = [];
  let videoStatusPolls = 0;
  let promptPosts = 0;
  page.on("pageerror", (error) => errors.push(String(error)));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  await page.addInitScript(() => {
    localStorage.removeItem("sf-gen-left-width");
    localStorage.removeItem("sf-gen-right-width");
    localStorage.removeItem("sf-gen-prompt-height");
    localStorage.removeItem("sf-gen-library-height");
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
      body: JSON.stringify({ job_id: "job-serenity-ui-gate" }),
    });
  });
  await page.route("**/v1/video", async (route) => {
    if (route.request().method() !== "POST") {
      await route.continue();
      return;
    }
    videoBodies.push(JSON.parse(route.request().postData() || "{}"));
    await route.fulfill({
      status: 202,
      contentType: "application/json",
      body: JSON.stringify({
        video_id: "video-playwright",
        prompt_id: "video-playwright",
        state: "queued",
        status_url: "/out/video-playwright/status.json",
        result_url: "/out/video-playwright/result.json",
      }),
    });
  });
  await page.route("**/out/video-playwright/status.json", async (route) => {
    videoStatusPolls += 1;
    const status = videoStatusPolls === 1
      ? { state: "running", step: 12, total: 48, message: "Encoding LTX2 prompt · Gemma layer 12 / 48" }
      : videoStatusPolls === 2
        ? { state: "running", step: 4, total: 8, message: "sampling" }
        : { state: "done", step: 8, total: 8, message: "Video complete" };
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(status),
    });
  });
  await page.route("**/out/video-playwright/result.json", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        state: "done",
        artifact_path: "/tmp/video-playwright/ltx2_t2v_hq.mp4",
        mp4_url: "/out/video_projects/project-1784913016021-437/media/1784913017255-ltx2_t2v_hq.mp4",
      }),
    });
  });
  const restoreVideoFixture = path.join(os.tmpdir(), "serenity-generate-ui-video-fixture.mp4");
  if (!fs.existsSync(restoreVideoFixture) || fs.statSync(restoreVideoFixture).size < 1024) {
    const generated = spawnSync("ffmpeg", [
      "-hide_banner", "-loglevel", "error", "-y",
      "-f", "lavfi", "-i", "color=c=black:s=320x180:d=1:r=24",
      "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
      "-movflags", "+faststart", restoreVideoFixture,
    ], { encoding: "utf8" });
    assert(generated.status === 0 && fs.existsSync(restoreVideoFixture),
      `failed to create browser video fixture: ${generated.stderr || generated.error || generated.status}`);
  }
  await page.route("**/out/video_projects/project-1784913016021-437/media/1784913017255-ltx2_t2v_hq.mp4", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "video/mp4",
      body: fs.readFileSync(restoreVideoFixture),
    });
  });
  await page.route("**/out/video-restore-probe/ltx2_t2v_hq.mp4", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "video/mp4",
      body: fs.readFileSync(restoreVideoFixture),
    });
  });
  await page.route("**/view?*", async (route) => {
    const url = new URL(route.request().url());
    if (url.searchParams.get("subfolder") !== "video-restore-probe") {
      await route.continue();
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: "video/mp4",
      body: fs.readFileSync(restoreVideoFixture),
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
    await page.waitForSelector(".gen-workspace-layout");
    await page.waitForFunction(() =>
      window.GenerateTab && GenerateTab.state.allModels && GenerateTab.state.allModels.length > 0
    );

    const surface = await page.evaluate(() => ({
      gridColumns: getComputedStyle(document.querySelector(".gen-workspace-layout")).gridTemplateColumns,
      promptDock: !!document.querySelector(".gen-workspace-prompt-dock"),
      batchPanel: !!document.querySelector(".gen-workspace-batch-panel"),
      libraryTabs: Array.from(document.querySelectorAll(".gen-library-tab")).map((node) => node.textContent.trim()),
      groupTitles: Array.from(document.querySelectorAll(".gen-workspace-group-title")).map((node) => node.textContent.trim()),
      visibleVideo: Array.from(document.querySelectorAll("#panel-generate [id*=video]"))
        .filter((node) => node.offsetParent !== null && getComputedStyle(node).display !== "none")
        .map((node) => node.id),
      canvasOwnedText: ["init image", "inpaint", "controlnet", "regional prompting", "segment refining"]
        .filter((term) => document.querySelector("#panel-generate").innerText.toLowerCase().includes(term)),
      advancedRows: document.querySelectorAll(".gen-workspace-advanced-only .gen-param-row").length,
      advancedCount: document.querySelector("#gen-advanced-count").textContent.trim(),
      railText: document.querySelector(".gen-workspace-parameters").textContent,
      imageVideoControlsDisabled: Array.from(
        document.querySelectorAll("#gen-video-section input, #gen-video-section select, #gen-video-conditioning-section input")
      ).every((node) => node.disabled),
      videoModels: GenerateTab.state.allModels
        .map((model) => model.name)
        .filter((name) => /(?:ltx|wan|bernini|scail)/i.test(name)),
      encoderOrEditModels: GenerateTab.state.allModels
        .map((model) => model.name)
        .filter((name) => /(?:qwen[_-]?2\\.5[_-]?vl|qwen.*image.*edit)/i.test(name)),
      activity: (() => {
        const topbar = document.querySelector(".gen-workspace-topbar").getBoundingClientRect();
        const status = document.querySelector("#gen-activity-status").getBoundingClientRect();
        return {
          text: document.querySelector("#gen-activity-text").textContent.trim(),
          state: document.querySelector("#gen-activity-status").dataset.state,
          centerDelta: Math.abs(
            (status.left + status.width / 2) - (topbar.left + topbar.width / 2)
          ),
        };
      })(),
    }));
    assert(surface.gridColumns.split(" ").length === 5, `expected 3 panels plus 2 resize borders: ${surface.gridColumns}`);
    assert(surface.promptDock && surface.batchPanel, "prompt dock or current-batch panel missing");
    assert(
      ["History", "Presets", "Models", "LoRAs"].every((name) => surface.libraryTabs.includes(name)),
      `library tabs incomplete: ${surface.libraryTabs.join(", ")}`
    );
    assert(
      [
        "Core Parameters?", "Resolution?", "Sampling?", "Video?", "Video Conditioning?",
        "LoRAs?", "Refine / Upscale?", "Runtime & Output?", "Advanced Video?",
        "Video Extend?", "Advanced Model Addons?", "Dynamic Thresholding?",
        "Advanced Sampling?", "Alternate Guidance?", "Output?"
      ]
        .every((name) => surface.groupTitles.includes(name)),
      `parameter groups incomplete: ${surface.groupTitles.join(", ")}`
    );
    assert(
      surface.visibleVideo.includes("gen-video-section") &&
      surface.visibleVideo.includes("gen-video-conditioning-section"),
      `video sections are not discoverable: ${surface.visibleVideo.join(", ")}`
    );
    assert(surface.imageVideoControlsDisabled, "video controls were enabled for an image model");
    assert(surface.canvasOwnedText.length === 0, `Canvas-owned controls leaked into Generate: ${surface.canvasOwnedText.join(", ")}`);
    assert(
      surface.activity.text === "Idle" &&
        surface.activity.state === "idle" &&
        surface.activity.centerDelta < 2,
      `generation activity is not centered and idle at startup: ${JSON.stringify(surface.activity)}`
    );
    assert(surface.advancedRows >= 80, `full advanced parameter rail is incomplete: ${surface.advancedRows}`);
    assert(/\(\d+\)/.test(surface.advancedCount), `advanced option count missing: ${surface.advancedCount}`);
    const nestedInventory = [
      "Refiner VAE", "Do Not Save Intermediates", "Wildcard Seed Behavior",
      "Video Motion Bucket", "Video Augmentation Level", "GPT-OSS Model",
      "Negative Model Include LoRAs", "CFG Scale Minimum", "Interpolate Phi",
      "VAE Temporal Tile Overlap", "Shifted Latent Average Init", "FreeU Skip Two"
    ];
    const missingNested = nestedInventory.filter((name) => !surface.railText.includes(name));
    assert(missingNested.length === 0, `nested parameter inventory is incomplete: ${missingNested.join(", ")}`);
    const expectedLtxVideoModels = [
      "ltx-2.3-22b-dev",
      "ltx-2.3-22b-dev-fp8",
      "ltx-2.3-22b-dev-fp8-dequant-bf16",
      "ltx-2.3-22b-distilled",
      "ltx-2.3-22b-distilled-fp8",
      "ltx-2.3-22b-distilled-fp8-dequant-bf16",
    ];
    assert(
      surface.videoModels.length === expectedLtxVideoModels.length &&
        expectedLtxVideoModels.every((model) => surface.videoModels.includes(model)),
      `admitted video picker inventory drifted: ${surface.videoModels.join(", ")}`
    );
    assert(
      surface.encoderOrEditModels.length === 0,
      `encoder/edit artifacts leaked into text-to-image picker: ${surface.encoderOrEditModels.join(", ")}`
    );

    await page.locator("#gen-model-search").click();
    await page.locator("#gen-model-dropdown").waitFor({ state: "visible" });
    const modelMenuLayering = await page.evaluate(() => {
      const dropdown = document.querySelector("#gen-model-dropdown");
      const nextHeader = document.querySelector("#gen-core-header");
      const dropdownRect = dropdown.getBoundingClientRect();
      const nextRect = nextHeader.getBoundingClientRect();
      const x = dropdownRect.left + Math.min(40, dropdownRect.width / 2);
      const y = Math.max(dropdownRect.top + 18, nextRect.top + 8);
      const hit = document.elementFromPoint(x, y);
      return {
        dropdown: { top: dropdownRect.top, bottom: dropdownRect.bottom },
        nextHeader: { top: nextRect.top, bottom: nextRect.bottom },
        hit: hit && `${hit.tagName}.${hit.className}`,
        menuOwnsOverlap: !!(hit && hit.closest("#gen-model-dropdown")),
      };
    });
    assert(
      modelMenuLayering.menuOwnsOverlap,
      `model menu is painted beneath Core Parameters: ${JSON.stringify(modelMenuLayering)}`
    );
    await page.locator("#gen-model-search").press("Escape");

    const imageModelProfiles = await page.evaluate(() =>
      GenerateTab.state.allModels
        .filter((model) => model.generationRoute === "image")
        .map((model) => ({
          name: model.name,
          defaults: model.generationDefaults || {},
        }))
    );
    const modelDefaultAudit = [];
    for (const model of imageModelProfiles) {
      assert(
        Number.isFinite(Number(model.defaults.steps)) &&
        Number.isFinite(Number(model.defaults.cfg)),
        `model card is missing exact generation defaults: ${JSON.stringify(model)}`
      );
      await selectModel(page, model.name);
      const observed = await page.evaluate(() => ({
        model: GenerateTab.state.model,
        steps: GenerateTab.state.steps,
        cfg: GenerateTab.state.cfg,
        inputSteps: Number(document.querySelector("#gen-steps").value),
        rangeSteps: Number(document.querySelector("#gen-steps-range").value),
        inputMax: Number(document.querySelector("#gen-steps").max),
        rangeMax: Number(document.querySelector("#gen-steps-range").max),
      }));
      assert(
        observed.steps === Number(model.defaults.steps) &&
        observed.inputSteps === Number(model.defaults.steps) &&
        observed.rangeSteps === Number(model.defaults.steps) &&
        observed.cfg === Number(model.defaults.cfg) &&
        observed.inputMax >= 500 &&
        observed.rangeMax >= 150,
        `canvas did not honor ${model.name}'s model-card defaults: expected ${JSON.stringify(model.defaults)}, got ${JSON.stringify(observed)}`
      );
      modelDefaultAudit.push(observed);
    }

    await selectModel(page, "zimage_base");
    const zimage = await page.evaluate(() => ({
      initPresent: !!document.querySelector("#gen-init-section"),
      variationVisible: document.querySelector("#gen-variation-section").offsetParent !== null,
      samplers: Array.from(document.querySelectorAll("#gen-sampler option")).map((node) => node.value),
      schedulers: Array.from(document.querySelectorAll("#gen-scheduler option")).map((node) => node.value),
      sizes: Array.from(document.querySelectorAll("#gen-aspect-dropdown option")).map((node) => node.value),
      sigmaShiftEnabled: !document.querySelector("#gen-sigma-shift").disabled,
    }));
    assert(!zimage.initPresent, "Canvas-owned init-image controls leaked into Generate");
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
    assert(zimage.sigmaShiftEnabled, "Z-Image Sigma Shift capability is not enabled");

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
      GenerateTab.state.prompt = "Serenity request contract probe";
      GenerateTab.state.variationSeed = 9876;
      GenerateTab.state.variationStrength = 0.25;
      GenerateTab.state.initImagePath = "/tmp/serenity-init-contract.png";
      document.querySelector("#gen-prompt").value = GenerateTab.state.prompt;
      document.querySelector("#gen-variation-seed").value = "9876";
      document.querySelector("#gen-variation-strength").value = "0.25";
      document.querySelector("#gen-sigma-shift").value = "4.25";
      document.querySelector("#gen-sigma-shift").dispatchEvent(new Event("input", { bubbles: true }));
      document.querySelector("#gen-no-seed-increment").checked = true;
      document.querySelector("#gen-no-seed-increment").dispatchEvent(new Event("change", { bubbles: true }));
      document.querySelector("#gen-continue-after-errors").checked = true;
      document.querySelector("#gen-continue-after-errors").dispatchEvent(new Event("change", { bubbles: true }));
      document.querySelector("#gen-personal-note").value = "Playwright reusable note";
      document.querySelector("#gen-personal-note").dispatchEvent(new Event("input", { bubbles: true }));
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
    assert(body.prompt === "Serenity request contract probe", `wrong request prompt: ${body.prompt}`);
    assert(body.sampler && body.scheduler, "sampler/scheduler missing from request");
    assert(body.variation_seed === 9876 && body.variation_strength === 0.25, "variation fields missing");
    assert(body.sigma_shift === 4.25, `Sigma Shift did not reach the request: ${body.sigma_shift}`);
    assert(body.init_image === undefined && body.creativity === undefined, "Canvas-owned init-image fields leaked into request");
    assert(body.mask_image === undefined, "mask/inpaint field leaked into request");
    const reusableImageParams = await page.evaluate(() => GenerateTab.getParams());
    assert(
      reusableImageParams.noSeedIncrement === true &&
      reusableImageParams.continueAfterErrors === true &&
      reusableImageParams.personalNote === "Playwright reusable note" &&
      reusableImageParams.sigmaShift === 4.25,
      `advanced reusable parameters drifted: ${JSON.stringify(reusableImageParams)}`
    );
    assert(promptPosts === 0, "Generate still routed through the legacy /prompt workflow adapter");
    await page.evaluate(() => {
      GenerateTab.state.generating = false;
      GenerateTab.state.pendingBatch = 0;
      const button = document.querySelector("#gen-btn");
      button.disabled = false;
      button.classList.remove("generating");
    });

    const beforeResize = await page.locator(".gen-workspace-stage").boundingBox();
    const leftHandle = page.locator("#gen-left-resizer");
    const leftBox = await leftHandle.boundingBox();
    await page.mouse.move(leftBox.x + leftBox.width / 2, leftBox.y + 120);
    await page.mouse.down();
    await page.mouse.move(leftBox.x + leftBox.width / 2 + 90, leftBox.y + 120);
    await page.mouse.up();
    const afterResize = await page.locator(".gen-workspace-stage").boundingBox();
    assert(afterResize.x > beforeResize.x + 70, "left resize border did not resize the Parameters panel");
    const persistedLeft = await page.evaluate(() => Number(localStorage.getItem("sf-gen-left-width")));
    assert(persistedLeft >= 450, `left border size was not persisted: ${persistedLeft}`);

    const previewBeforeVerticalResize = await page.locator(".gen-workspace-stage").boundingBox();
    const promptHandle = page.locator("#gen-prompt-resizer");
    const promptHandleBox = await promptHandle.boundingBox();
    await page.mouse.move(promptHandleBox.x + promptHandleBox.width / 2, promptHandleBox.y + promptHandleBox.height / 2);
    await page.mouse.down();
    await page.mouse.move(promptHandleBox.x + promptHandleBox.width / 2, promptHandleBox.y + 70);
    await page.mouse.up();
    const libraryHandle = page.locator("#gen-library-resizer");
    const libraryHandleBox = await libraryHandle.boundingBox();
    await page.mouse.move(libraryHandleBox.x + libraryHandleBox.width / 2, libraryHandleBox.y + libraryHandleBox.height / 2);
    await page.mouse.down();
    await page.mouse.move(libraryHandleBox.x + libraryHandleBox.width / 2, libraryHandleBox.y + 110);
    await page.mouse.up();
    const previewAfterVerticalResize = await page.locator(".gen-workspace-stage").boundingBox();
    const persistedVertical = await page.evaluate(() => ({
      prompt: Number(localStorage.getItem("sf-gen-prompt-height")),
      library: Number(localStorage.getItem("sf-gen-library-height")),
    }));
    assert(
      previewAfterVerticalResize.height > previewBeforeVerticalResize.height + 150,
      `middle dividers did not enlarge the preview: before=${previewBeforeVerticalResize.height}, after=${previewAfterVerticalResize.height}`
    );
    assert(
      persistedVertical.prompt <= 100 && persistedVertical.library <= 125,
      `middle divider sizes were not persisted: ${JSON.stringify(persistedVertical)}`
    );

    await page.evaluate(() => {
      GenerateTab.displayResult(
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
        false,
        { params: { model: GenerateTab.state.model, prompt: "responsive image", width: 1024, height: 1024 } }
      );
    });
    await page.locator("#gen-preview-img").waitFor({ state: "visible" });
    const imageBounds = await page.locator("#gen-preview-img").boundingBox();
    const centerBounds = await page.locator(".gen-center-content").boundingBox();
    assert(
      imageBounds.width <= centerBounds.width && imageBounds.height <= centerBounds.height,
      `center image overflowed after resize: image=${JSON.stringify(imageBounds)} center=${JSON.stringify(centerBounds)}`
    );

    await selectModel(page, "ltx-2.3-22b-dev-fp8");
    const videoUi = await page.evaluate(() => ({
      sectionVisible: document.querySelector("#gen-video-section").offsetParent !== null,
      conditioningVisible: document.querySelector("#gen-video-conditioning-section").offsetParent !== null,
      width: Number(document.querySelector("#gen-custom-width").value),
      height: Number(document.querySelector("#gen-custom-height").value),
      frames: Number(document.querySelector("#gen-frames").value),
      fps: Number(document.querySelector("#gen-fps").value),
      sampler: document.querySelector("#gen-sampler").value,
      scheduler: document.querySelector("#gen-scheduler").value,
      output: document.querySelector("#gen-output-format").value,
      profileNote: document.querySelector("#gen-video-profile-note").textContent.trim(),
      aspectOptions: Array.from(document.querySelector("#gen-aspect-dropdown").options)
        .map((option) => option.value),
      frameInput: {
        min: Number(document.querySelector("#gen-frames").min),
        max: Number(document.querySelector("#gen-frames").max),
        step: Number(document.querySelector("#gen-frames").step),
      },
      dimensionsLocked: ["#gen-custom-width", "#gen-custom-height"]
        .every((selector) => document.querySelector(selector).disabled),
      timelineControlsEnabled: ["#gen-frames", "#gen-seconds", "#gen-fps-range", "#gen-fps"]
        .every((selector) => !document.querySelector(selector).disabled),
      postUpscalers: Array.from(document.querySelector("#gen-post-upscaler").options)
        .map((option) => ({
          value: option.value,
          disabled: option.disabled,
          text: option.textContent.trim(),
        })),
    }));
    assert(videoUi.sectionVisible && videoUi.conditioningVisible, "LTX2 video controls did not open");
    assert(
      videoUi.width === 512 && videoUi.height === 768 && videoUi.frames === 121 && videoUi.fps === 25,
      `LTX2 UI did not use the published compiled profile: ${JSON.stringify(videoUi)}`
    );
    assert(
      videoUi.sampler === "euler" && videoUi.scheduler === "ltx2_distilled" && videoUi.output === "MP4",
      `LTX2 sampling/output profile drifted: ${JSON.stringify(videoUi)}`
    );
    assert(
      videoUi.dimensionsLocked && videoUi.timelineControlsEnabled &&
      videoUi.aspectOptions.includes("960×544") &&
      videoUi.aspectOptions.includes("1920×1088") &&
      videoUi.frameInput.min === 121 &&
      videoUi.frameInput.max === 121 &&
      videoUi.frameInput.step === 8 &&
      videoUi.profileNote.includes("exact AOT Mojo runner"),
      `LTX2 native profile controls are incomplete: ${JSON.stringify(videoUi)}`
    );
    const realesrganOption = videoUi.postUpscalers.find(
      (option) => option.value === "realesrgan-x4plus"
    );
    const seedvr2Option = videoUi.postUpscalers.find(
      (option) => option.value === "seedvr2-3b"
    );
    const realesrganFastOption = videoUi.postUpscalers.find(
      (option) => option.value === "realesrgan-fast-x4v3"
    );
    assert(
      realesrganOption && !realesrganOption.disabled &&
      realesrganFastOption && realesrganFastOption.disabled &&
      seedvr2Option && seedvr2Option.disabled,
      `LTX2 post-upscaler readiness is not truthful: ${JSON.stringify(videoUi.postUpscalers)}`
    );
    if (await page.locator("#gen-image-header").evaluate((node) => node.classList.contains("closed"))) {
      await page.locator("#gen-image-header").click();
    }
    if (await page.locator("#gen-video-header").evaluate((node) => node.classList.contains("closed"))) {
      await page.locator("#gen-video-header").click();
    }
    await page.locator("#gen-aspect-dropdown").selectOption("960×544");
    await page.locator("#gen-seconds").fill("8");
    const selectedNativeProfile = await page.evaluate(() => ({
      width: GenerateTab.state.width,
      height: GenerateTab.state.height,
      frames: GenerateTab.state.frames,
      fps: GenerateTab.state.fps,
      seconds: GenerateTab.state.seconds,
      frameMin: Number(document.querySelector("#gen-frames").min),
      frameMax: Number(document.querySelector("#gen-frames").max),
      profileNote: document.querySelector("#gen-video-profile-note").textContent.trim(),
    }));
    assert(
      selectedNativeProfile.width === 960 &&
      selectedNativeProfile.height === 544 &&
      selectedNativeProfile.frames === 193 &&
      selectedNativeProfile.fps === 24 &&
      selectedNativeProfile.seconds === 8 &&
      selectedNativeProfile.frameMin === 121 &&
      selectedNativeProfile.frameMax === 481 &&
      selectedNativeProfile.profileNote.includes("exact AOT Mojo runner"),
      `LTX2 native profile selection did not stay coherent: ${JSON.stringify(selectedNativeProfile)}`
    );
    await page.locator("#gen-fps").fill("60");
    const unsupportedFps = await page.evaluate(() => ({
      fps: GenerateTab.state.fps,
      frames: GenerateTab.state.frames,
      profileNote: document.querySelector("#gen-video-profile-note").textContent.trim(),
      invalid: document.querySelector("#gen-video-profile-note").classList.contains("invalid"),
    }));
    assert(
      unsupportedFps.fps === 60 && unsupportedFps.frames === 481 &&
      unsupportedFps.invalid && unsupportedFps.profileNote.includes("No compiled native runner"),
      `LTX2 editable FPS did not fail closed for an unsupported profile: ${JSON.stringify(unsupportedFps)}`
    );
    await page.locator("#gen-fps").fill("24");
    const restoredFps = await page.evaluate(() => ({
      fps: GenerateTab.state.fps,
      frames: GenerateTab.state.frames,
      invalid: document.querySelector("#gen-video-profile-note").classList.contains("invalid"),
    }));
    assert(
      restoredFps.fps === 24 && restoredFps.frames === 193 && !restoredFps.invalid,
      `LTX2 supported FPS did not restore the native profile: ${JSON.stringify(restoredFps)}`
    );
    if (await page.locator("#gen-refine-header").evaluate((node) => node.classList.contains("closed"))) {
      await page.locator("#gen-refine-header").click();
    }
    await page.locator("#gen-post-upscaler").selectOption("realesrgan-x4plus");
    await page.locator("#gen-post-upscale-factor").selectOption("2");
    const selectedPostUpscale = await page.evaluate(() => ({
      id: GenerateTab.state.postUpscaler,
      factor: GenerateTab.state.postUpscaleFactor,
      note: document.querySelector("#gen-post-upscale-note").textContent.trim(),
      factorText: document.querySelector("#gen-post-upscale-factor")
        .selectedOptions[0].textContent.trim(),
    }));
    assert(
      selectedPostUpscale.id === "realesrgan-x4plus" &&
      selectedPostUpscale.factor === 2 &&
      selectedPostUpscale.note.includes("1920×1088") &&
      selectedPostUpscale.factorText.includes("1920×1088"),
      `LTX2 post-upscale output dimensions are not coherent: ${JSON.stringify(selectedPostUpscale)}`
    );
    const eriLoraName = "ltx2_eri2_step3000";
    if (await page.locator("#gen-lora-header").evaluate((node) => node.classList.contains("closed"))) {
      await page.locator("#gen-lora-header").click();
    }
    await page.locator(`#gen-lora-picker option[value="${eriLoraName}"]`).waitFor({ state: "attached" });
    await page.locator("#gen-lora-picker").selectOption(eriLoraName);
    const ltxLoraUi = await page.evaluate((name) => ({
      state: GenerateTab.state.loras.map((lora) => ({
        name: lora.name,
        strength: lora.strength,
        enabled: lora.enabled,
      })),
      rowText: document.querySelector("#gen-lora-list").textContent.trim(),
      pickerPrompt: document.querySelector("#gen-lora-picker option").textContent.trim(),
    }), eriLoraName);
    assert(
      ltxLoraUi.state.length === 1 &&
        ltxLoraUi.state[0].name === eriLoraName &&
        ltxLoraUi.state[0].strength === 1 &&
        ltxLoraUi.state[0].enabled === true,
      `LTX2 Eri LoRA selection was not retained: ${JSON.stringify(ltxLoraUi)}`
    );
    assert(ltxLoraUi.rowText.includes(eriLoraName), `LTX2 Eri LoRA row is missing: ${JSON.stringify(ltxLoraUi)}`);
    assert(
      ltxLoraUi.pickerPrompt === "1 LoRA loaded · add another…",
      `LTX2 LoRA picker falsely reports an empty selection: ${JSON.stringify(ltxLoraUi)}`
    );
    await page.locator("#gen-prompt").fill("Playwright LTX2 request");
    await page.locator("#gen-neg-prompt").fill("watermark");
    await page.locator("#gen-video-conditioning-header").click();
    await page.locator("#gen-btn").click();
    await page.waitForFunction(() => {
      const node = document.querySelector("#gen-activity-status");
      return node && node.dataset.state === "encoding" &&
        node.textContent.includes("Gemma layer 12 / 48");
    });
    const encodingActivity = await page.locator("#gen-activity-status").textContent();
    await page.waitForFunction(() => {
      const node = document.querySelector("#gen-activity-status");
      return node && node.dataset.state === "sampling" &&
        node.textContent.includes("Step 4 / 8");
    });
    const samplingActivity = await page.locator("#gen-activity-status").textContent();
    await page.waitForFunction(() => document.querySelector("#gen-preview-video").style.display === "block");
    await page.waitForTimeout(500);
    assert(videoBodies.length === 1, `expected one video request, got ${videoBodies.length}`);
    assert(
      videoBodies[0].caps_positive === "" && videoBodies[0].caps_negative === "",
      `LTX2 automatic conditioning should submit blank manual overrides: ${JSON.stringify(videoBodies[0])}`
    );
    assert(
      videoBodies[0].width === 960 && videoBodies[0].height === 544 &&
      videoBodies[0].frames === 193 && videoBodies[0].fps === 24,
      `video request did not preserve the published profile: ${JSON.stringify(videoBodies[0])}`
    );
    assert(
      Array.isArray(videoBodies[0].lora) &&
        videoBodies[0].lora.length === 1 &&
        videoBodies[0].lora[0].name === eriLoraName &&
        videoBodies[0].lora[0].weight === 1,
      `LTX2 Eri LoRA did not reach the video request: ${JSON.stringify(videoBodies[0])}`
    );
    assert(
      videoBodies[0].post_upscale &&
      videoBodies[0].post_upscale.id === "realesrgan-x4plus" &&
      videoBodies[0].post_upscale.factor === 2,
      `LTX2 post-upscale did not reach the video request: ${JSON.stringify(videoBodies[0])}`
    );
    const videoPlayback = await page.evaluate(() => {
      const video = document.querySelector("#gen-preview-video");
      return {
        display: getComputedStyle(video).display,
        controls: video.controls,
        muted: video.muted,
        paused: video.paused,
        readyState: video.readyState,
        currentSrc: video.currentSrc,
        batchVideos: document.querySelectorAll("#gen-batch-strip video").length,
      };
    });
    assert(videoPlayback.display !== "none", "video preview is still CSS-hidden");
    assert(videoPlayback.controls && videoPlayback.muted, "video preview is not configured for reliable playback");
    assert(videoPlayback.readyState >= 2 && !videoPlayback.paused, `video did not play: ${JSON.stringify(videoPlayback)}`);
    assert(videoPlayback.batchVideos === 1, "video result is missing from Current Batch");
    const historyVideosBeforeLateEvent = await page.locator("#gen-gallery-grid .gen-thumb-video").count();
    assert(
      historyVideosBeforeLateEvent === 0,
      `Current Batch movie was also duplicated in History: ${historyVideosBeforeLateEvent}`
    );
    await page.evaluate(() => {
      SerenityWS._emit("executed", {
        prompt_id: "video-playwright",
        output: {
          videos: [{
            filename: "ltx2_t2v_hq.mp4",
            subfolder: "video-playwright",
            type: "output",
          }],
        },
      });
    });
    await page.waitForTimeout(500);
    const historyVideosAfterLateEvent = await page.locator("#gen-gallery-grid .gen-thumb-video").count();
    assert(
      historyVideosAfterLateEvent === 0,
      `late WebSocket completion duplicated the LTX2 movie: ${historyVideosAfterLateEvent}`
    );

    await page.locator("#gen-prompt").fill("mutated after render");
    await page.locator("#gen-post-upscaler").selectOption("none");
    await page.locator("#gen-reuse-params").click();
    const reused = await page.evaluate(() => ({
      prompt: document.querySelector("#gen-prompt").value,
      model: GenerateTab.state.model,
      caps: document.querySelector("#gen-caps-positive").value,
      frames: Number(document.querySelector("#gen-frames").value),
      fps: Number(document.querySelector("#gen-fps").value),
      postUpscaler: GenerateTab.state.postUpscaler,
      postUpscaleFactor: GenerateTab.state.postUpscaleFactor,
    }));
    assert(
      reused.prompt === "Playwright LTX2 request" &&
      reused.model.includes("ltx-2.3-22b-dev-fp8") &&
      reused.caps === "" &&
      reused.frames === 193 && reused.fps === 24 &&
      reused.postUpscaler === "realesrgan-x4plus" &&
      reused.postUpscaleFactor === 2,
      `Reuse parameters did not restore the full request: ${JSON.stringify(reused)}`
    );
    fs.mkdirSync(path.join(process.cwd(), "output", "checks"), { recursive: true });
    await page.screenshot({
      path: path.join(process.cwd(), "output", "checks", "serenity_generate_video.png"),
      fullPage: true,
    });
    await selectModel(page, "krea2-raw");
    const imageStepBoundsAfterVideo = await page.evaluate(() => {
      const input = document.querySelector("#gen-steps");
      const range = document.querySelector("#gen-steps-range");
      input.value = "80";
      input.dispatchEvent(new Event("input", { bubbles: true }));
      return {
        inputMax: Number(input.max),
        rangeMax: Number(range.max),
        inputValue: Number(input.value),
        rangeValue: Number(range.value),
        stateSteps: GenerateTab.state.steps,
      };
    });
    assert(
      imageStepBoundsAfterVideo.inputMax >= 500 &&
      imageStepBoundsAfterVideo.rangeMax >= 150 &&
      imageStepBoundsAfterVideo.inputValue === 80 &&
      imageStepBoundsAfterVideo.rangeValue === 80 &&
      imageStepBoundsAfterVideo.stateSteps === 80,
      `LTX2's 20-step ceiling leaked into image generation: ${JSON.stringify(imageStepBoundsAfterVideo)}`
    );
    await page.evaluate(() => {
      document.querySelectorAll(".gen-workspace-group-title.closed, .gen-workspace-group-body.closed")
        .forEach((node) => node.classList.remove("closed"));
      const panel = document.querySelector("#panel-generate");
      const layout = document.querySelector(".gen-workspace-layout");
      const rail = document.querySelector(".gen-workspace-parameters");
      const scroll = document.querySelector(".gen-workspace-param-scroll");
      panel.style.overflow = "visible";
      panel.style.height = "auto";
      layout.style.display = "block";
      layout.style.height = "auto";
      layout.style.overflow = "visible";
      scroll.style.height = "auto";
      scroll.style.overflow = "visible";
      scroll.style.flex = "none";
      rail.style.width = "470px";
      rail.style.height = "auto";
      rail.style.overflow = "visible";
    });
    await page.locator(".gen-workspace-parameters").screenshot({
      path: path.join(process.cwd(), "output", "checks", "serenity_generate_full_parameters_2026-07-24.png"),
    });
    await page.evaluate(() => {
      localStorage.setItem("sf-gallery", JSON.stringify([
        {
          src: "/view?filename=ltx2_t2v_hq.mp4&subfolder=video-restore-probe&type=output",
          isVideo: true,
          prompt: "",
          timestamp: 2,
        },
        {
          src: "/out/video-restore-probe/ltx2_t2v_hq.mp4",
          isVideo: true,
          prompt: "metadata-bearing copy",
          params: { prompt: "metadata-bearing copy", model: "ltx-2.3-22b-dev-fp8" },
          timestamp: 1,
        },
      ]));
    });
    await page.reload({ waitUntil: "networkidle" });
    await page.waitForFunction(() => window.GenerateTab && GenerateTab.state.gallery.length > 0);
    const restoredVideoDedup = await page.evaluate(() => ({
      count: GenerateTab.state.gallery.filter((item) => item.isVideo).length,
      prompt: GenerateTab.state.gallery[0] && GenerateTab.state.gallery[0].prompt,
      persistedCount: JSON.parse(localStorage.getItem("sf-gallery") || "[]").length,
    }));
    assert(
      restoredVideoDedup.count === 1 &&
      restoredVideoDedup.persistedCount === 1 &&
      restoredVideoDedup.prompt === "metadata-bearing copy",
      `persisted LTX2 duplicate cleanup failed: ${JSON.stringify(restoredVideoDedup)}`
    );
    const shortPage = await browser.newPage({ viewport: { width: 1920, height: 576 } });
    const shortErrors = [];
    shortPage.on("pageerror", (error) => shortErrors.push(String(error)));
    await shortPage.addInitScript(() => {
      localStorage.removeItem("sf-gen-prompt-height");
      localStorage.removeItem("sf-gen-library-height");
    });
    await shortPage.goto(`${baseUrl}/?tab=generate`, { waitUntil: "networkidle" });
    await shortPage.waitForSelector(".gen-workspace-layout");
    const shortBefore = await shortPage.evaluate(() => {
      const stage = document.querySelector(".gen-workspace-stage").getBoundingClientRect();
      const promptHandle = document.querySelector("#gen-prompt-resizer").getBoundingClientRect();
      const libraryHandle = document.querySelector("#gen-library-resizer").getBoundingClientRect();
      return {
        viewportHeight: window.innerHeight,
        stage: { top: stage.top, bottom: stage.bottom, height: stage.height },
        promptHandle: { top: promptHandle.top, bottom: promptHandle.bottom },
        libraryHandle: { top: libraryHandle.top, bottom: libraryHandle.bottom },
      };
    });
    assert(shortBefore.stage.height >= 230, `short landscape preview is too small: ${JSON.stringify(shortBefore)}`);
    assert(
      shortBefore.promptHandle.bottom <= shortBefore.viewportHeight &&
      shortBefore.libraryHandle.bottom <= shortBefore.viewportHeight,
      `landscape resize handles escaped the viewport: ${JSON.stringify(shortBefore)}`
    );
    const shortLibraryBefore = await shortPage.locator(".gen-workspace-library").boundingBox();
    const shortLibraryHandle = await shortPage.locator("#gen-library-resizer").boundingBox();
    await shortPage.mouse.move(
      shortLibraryHandle.x + shortLibraryHandle.width / 2,
      shortLibraryHandle.y + shortLibraryHandle.height / 2
    );
    await shortPage.mouse.down();
    await shortPage.mouse.move(
      shortLibraryHandle.x + shortLibraryHandle.width / 2,
      shortLibraryHandle.y - 70
    );
    await shortPage.mouse.up();
    const shortLibraryRaised = await shortPage.locator(".gen-workspace-library").boundingBox();
    const shortStageAfterLibraryRaise = await shortPage.locator(".gen-workspace-stage").boundingBox();
    assert(
      shortLibraryRaised.height > shortLibraryBefore.height + 55 &&
      shortStageAfterLibraryRaise.height < shortBefore.stage.height - 55,
      `raising History did not trade preview height for panel height: ` +
      `library=${shortLibraryBefore.height}->${shortLibraryRaised.height}, ` +
      `stage=${shortBefore.stage.height}->${shortStageAfterLibraryRaise.height}`
    );
    await shortPage.locator("#gen-prompt-resizer").dblclick();
    const shortExpanded = await shortPage.locator(".gen-workspace-stage").boundingBox();
    assert(
      shortExpanded.height > shortBefore.stage.height + 70,
      `double-click did not collapse prompts and History: before=${shortBefore.stage.height}, after=${shortExpanded.height}`
    );
    await shortPage.evaluate(() => {
      GenerateTab.displayResult(
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
        false,
        { params: { model: "layout-probe", prompt: "landscape preview", width: 1024, height: 1024 } }
      );
    });
    const shortMedia = await shortPage.locator("#gen-preview-img").boundingBox();
    assert(
      shortMedia.height >= shortExpanded.height - 120,
      `preview media did not scale into the enlarged stage: media=${JSON.stringify(shortMedia)} stage=${JSON.stringify(shortExpanded)}`
    );
    await shortPage.screenshot({
      path: path.join(process.cwd(), "output", "checks", "serenity_generate_short_landscape.png"),
      fullPage: true,
    });
    await shortPage.close();
    assert(shortErrors.length === 0, `short-viewport browser errors: ${shortErrors.join(" | ")}`);
    assert(errors.length === 0, `browser errors: ${errors.join(" | ")}`);
    console.log("serenity generate ui: PASS");
    console.log(JSON.stringify({
      model: body.model,
      sampler: body.sampler,
      scheduler: body.scheduler,
      variation: body.variation_strength,
      resizePersisted: persistedLeft,
      verticalResizePersisted: persistedVertical,
      advancedRows: surface.advancedRows,
      videoPlayback,
      historyVideosAfterLateEvent,
      restoredVideoDedup,
      shortLandscape: { before: shortBefore, expanded: shortExpanded, media: shortMedia },
      encodingActivity,
      samplingActivity,
      reused,
      imageStepBoundsAfterVideo,
      modelDefaultAudit,
      libraryTabs: surface.libraryTabs,
    }, null, 2));
  } finally {
    await browser.close();
  }
}

run().catch((error) => {
  console.error("serenity generate ui: FAIL");
  console.error(error && error.stack ? error.stack : String(error));
  process.exit(1);
});
