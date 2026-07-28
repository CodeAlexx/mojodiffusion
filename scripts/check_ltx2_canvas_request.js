#!/usr/bin/env node
"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");

function loadChromium() {
  try { return require("playwright").chromium; } catch (_) {}
  const npxRoot = path.join(os.homedir(), ".npm", "_npx");
  if (fs.existsSync(npxRoot)) {
    for (const entry of fs.readdirSync(npxRoot)) {
      try {
        return require(path.join(npxRoot, entry, "node_modules", "playwright")).chromium;
      } catch (_) {}
    }
  }
  return require(path.join(os.tmpdir(), "mojodiffusion-playwright-tools", "node_modules", "playwright")).chromium;
}

function assert(value, message) {
  if (!value) throw new Error(message);
}

(async () => {
  const baseUrl = process.env.SERENITY_URL || "http://127.0.0.1:7811";
  const chromium = loadChromium();
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  const errors = [];
  const realRun = process.env.REAL_LTX2 === "1";
  const sourceVideoPath = String(process.env.LTX2_V2V_SOURCE || "").trim();
  const sourceImagePath = String(process.env.LTX2_I2V_SOURCE || "").trim();
  const lastImagePath = String(process.env.LTX2_LAST_FRAME || "").trim();
  const cameraMotion = String(process.env.LTX2_CAMERA_MOTION || "none").trim() || "none";
  const canvasMode = String(process.env.LTX2_CANVAS_MODE ||
    (sourceImagePath ? "i2v_ltx23" : "create")).trim();
  const temporalEditMode = canvasMode === "retake_ltx23" ||
    canvasMode === "extend_ltx23";
  const screenshotPath = String(process.env.LTX2_UI_SCREENSHOT || "").trim();
  const featureId = String(process.env.LTX2_FEATURE || "standard").trim() || "standard";
  const featureWeightText = String(process.env.LTX2_FEATURE_WEIGHT || "").trim();
  const promptText = String(process.env.LTX2_PROMPT || "").trim();
  const v2vStrengthText = String(process.env.LTX2_V2V_STRENGTH || "").trim();
  const nativeResolution = String(process.env.LTX2_NATIVE_RESOLUTION ||
    (temporalEditMode ? "540p-source-landscape" : "540p-portrait")).trim();
  const durationText = String(process.env.LTX2_DURATION ||
    (canvasMode === "extend_ltx23" ? "6" : "5")).trim();
  const quantization = String(process.env.LTX2_QUANT || "bf16").trim();
  const skipTemplate = process.env.LTX2_SKIP_TEMPLATE === "1";
  let request = null;
  let expectedProfile = null;

  page.on("pageerror", (error) => errors.push(String(error)));
  page.on("request", (browserRequest) => {
    if (browserRequest.method() === "POST" && /\/v1\/video$/.test(browserRequest.url())) {
      request = browserRequest.postDataJSON();
    }
  });
  if (!realRun) {
    await page.route("**/v1/video", async (route) => {
      if (route.request().method() !== "POST") return route.continue();
      await route.fulfill({
        status: 202,
        contentType: "application/json",
        body: JSON.stringify({
          schema: "serenity.video_job.v1",
          video_id: "playwright-ltx2",
          prompt_id: "playwright-ltx2",
          state: "queued",
          status_url: "/out/playwright-ltx2/status.json",
          result_url: "/out/playwright-ltx2/result.json",
        }),
      });
    });
    await page.route("**/out/playwright-ltx2/status.json", (route) => route.fulfill({
      status: 202,
      contentType: "application/json",
      body: JSON.stringify({
        schema: "serenity.ltx2.status.v1",
        state: "done",
        phase: "done",
        step: 11,
        total: 11,
        message: "Video ready",
      }),
    }));
    await page.route("**/out/playwright-ltx2/result.json", (route) => route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        schema: "serenity.ltx2.result.v1",
        state: "done",
        artifact_path: "/tmp/playwright-ltx2/ltx2_t2v_hq.mp4",
        width: 512,
        height: 768,
        frame_count: 121,
        fps: 25,
        steps: 8,
        seed: 42,
      }),
    }));
    await page.route("**/out/playwright-ltx2/ltx2_t2v_hq.mp4", (route) => route.fulfill({
      status: 200,
      contentType: "video/mp4",
      body: Buffer.alloc(0),
    }));
  }

  try {
    assert(!(sourceVideoPath && sourceImagePath), "choose only one LTX2 source");
    assert(!lastImagePath || sourceImagePath,
      "LTX2_LAST_FRAME requires LTX2_I2V_SOURCE");
    await page.addInitScript(() => {
      localStorage.removeItem("serenity-canvas-panel-offsets-v1");
    });
    await page.goto(baseUrl, { waitUntil: "domcontentloaded" });
    await page.click('.nav-btn[data-tab="canvas"]');
    await page.waitForSelector("#cv-model");
    assert(await page.locator("#cv-parameters-handle").count() === 1,
      "missing movable parameters-panel handle");
    assert(await page.locator("#cv-entities-handle").count() === 1,
      "missing movable entities-panel handle");
    const initialCenterWidth = await page.locator(".cv-center").evaluate((node) => node.getBoundingClientRect().width);
    const parametersHandleBox = await page.locator("#cv-parameters-handle").boundingBox();
    assert(parametersHandleBox, "parameters-panel handle is not visible");
    await page.mouse.move(
      parametersHandleBox.x + parametersHandleBox.width / 2,
      parametersHandleBox.y + parametersHandleBox.height / 2,
    );
    await page.mouse.down();
    await page.mouse.move(
      parametersHandleBox.x + parametersHandleBox.width / 2 - 110,
      parametersHandleBox.y + parametersHandleBox.height / 2,
      { steps: 6 },
    );
    await page.mouse.up();
    await page.waitForTimeout(250);
    const partialParametersOffset = await page.locator(".cv-layout").evaluate(
      (node) => parseFloat(getComputedStyle(node).getPropertyValue("--cv-parameters-offset")) || 0,
    );
    assert(partialParametersOffset >= 90,
      `parameters panel did not follow drag: offset=${partialParametersOffset}`);
    const expandedCenterWidth = await page.locator(".cv-center").evaluate((node) => node.getBoundingClientRect().width);
    assert(expandedCenterWidth >= initialCenterWidth + 80,
      `Canvas did not expand after panel drag: ${initialCenterWidth} -> ${expandedCenterWidth}`);
    await page.click("#cv-parameters-handle");
    await page.waitForTimeout(220);
    assert(await page.locator(".cv-layout").evaluate((node) => node.classList.contains("cv-parameters-hidden")),
      "parameters panel did not move fully off-screen");
    await page.click("#cv-parameters-handle");
    await page.waitForTimeout(220);
    assert(!(await page.locator(".cv-layout").evaluate((node) => node.classList.contains("cv-parameters-hidden"))),
      "parameters panel did not restore");
    const entitiesHandleBox = await page.locator("#cv-entities-handle").boundingBox();
    assert(entitiesHandleBox, "entities-panel handle is not visible");
    await page.mouse.move(
      entitiesHandleBox.x + entitiesHandleBox.width / 2,
      entitiesHandleBox.y + entitiesHandleBox.height / 2,
    );
    await page.mouse.down();
    await page.mouse.move(
      entitiesHandleBox.x + entitiesHandleBox.width / 2 + 100,
      entitiesHandleBox.y + entitiesHandleBox.height / 2,
      { steps: 6 },
    );
    await page.mouse.up();
    await page.waitForTimeout(250);
    const partialEntitiesOffset = await page.locator(".cv-layout").evaluate(
      (node) => parseFloat(getComputedStyle(node).getPropertyValue("--cv-entities-offset")) || 0,
    );
    assert(partialEntitiesOffset >= 80,
      `entities panel did not follow drag: offset=${partialEntitiesOffset}`);
    await page.click("#cv-entities-handle");
    await page.waitForTimeout(220);
    await page.click("#cv-entities-handle");
    await page.waitForTimeout(220);
    await page.waitForFunction(() => Array.from(
      document.querySelectorAll("#cv-model option"),
    ).some((option) => option.value === "ltx-2.3-22b-dev-fp8"));
    const chromaModel = await page.locator("#cv-model option").evaluateAll((options) => {
      const match = options.find((option) => /chroma/i.test(option.value || option.textContent || ""));
      return match ? match.value : "";
    });
    assert(chromaModel, "missing installed Chroma model for Canvas sampler capability check");
    await page.selectOption("#cv-model", chromaModel);
    await page.waitForFunction(() => (
      document.querySelector("#cv-scheduler")?.value === "beta"
      && Array.from(document.querySelectorAll("#cv-sampler option"))
        .some((option) => option.value === "dpmpp_2m")
    ));
    const chromaSamplers = await page.locator("#cv-sampler option").evaluateAll(
      (options) => options.map((option) => option.value),
    );
    const chromaSchedulers = await page.locator("#cv-scheduler option").evaluateAll(
      (options) => options.map((option) => option.value),
    );
    assert(!chromaSamplers.includes("euler_ancestral"),
      `Canvas exposed unsupported Chroma sampler: ${chromaSamplers.join(",")}`);
    assert([
      "beta", "simple", "normal", "sgm_uniform", "ddim_uniform", "karras",
      "exponential", "linear_quadratic", "kl_optimal",
    ].every((scheduler) => chromaSchedulers.includes(scheduler)),
    `Canvas omitted Chroma schedulers: ${chromaSchedulers.join(",")}`);
    const i2vMode = await page.locator('#cv-edit-mode option[value="i2v_ltx23"]').textContent();
    assert(i2vMode === "I2V - LTX 2.3", `I2V mode label=${i2vMode}`);
    assert(await page.locator('#cv-edit-mode option[value="retake_ltx23"]').textContent() ===
      "Retake - LTX 2.3", "missing Retake - LTX 2.3 mode");
    assert(await page.locator('#cv-edit-mode option[value="extend_ltx23"]').textContent() ===
      "Extend - LTX 2.3", "missing Extend - LTX 2.3 mode");
    if (sourceImagePath) {
      await page.selectOption("#cv-edit-mode", "i2v_ltx23");
      await page.waitForFunction(() => (
        document.querySelector("#cv-edit-mode")?.value === "i2v_ltx23"
        && /ltx[-_. ]?2[.-]3/i.test(document.querySelector("#cv-model")?.value || "")
      ));
      assert(await page.locator("#cv-import-file").getAttribute("accept") === "image/*",
        "I2V - LTX 2.3 must accept image sources only");
      assert(!(await page.locator("#cv-model-row").isVisible()),
        "I2V - LTX 2.3 must own and hide its model selection");
      assert(await page.locator("#cv-generate-btn").textContent() === "Generate I2V",
        "I2V - LTX 2.3 must expose an explicit Generate I2V action");
      assert(await page.locator("#cv-ltx2-last-frame-row").isVisible(),
        "I2V - LTX 2.3 must expose the optional last-frame keyframe");
    } else if (canvasMode === "retake_ltx23" || canvasMode === "extend_ltx23") {
      await page.selectOption("#cv-edit-mode", canvasMode);
      await page.waitForFunction((mode) => (
        document.querySelector("#cv-edit-mode")?.value === mode
        && /ltx[-_. ]?2[.-]3/i.test(document.querySelector("#cv-model")?.value || "")
      ), canvasMode);
      assert(await page.locator("#cv-import-file").getAttribute("accept") === "video/*",
        `${canvasMode} must accept video sources only`);
      assert(!(await page.locator("#cv-model-row").isVisible()),
        `${canvasMode} must own and hide its model selection`);
      assert(await page.locator("#cv-generate-btn").textContent() ===
        (canvasMode === "retake_ltx23" ? "Retake selected window" : "Extend video"),
        `${canvasMode} has the wrong generation action`);
    } else {
      await page.selectOption("#cv-model", "ltx-2.3-22b-dev-fp8");
    }
    await page.waitForSelector("#cv-load-ltx2-template", { state: "visible" });
    await page.waitForFunction(() => (
      Boolean(document.querySelector("#cv-ltx2-profile")?.value)
      && Number(document.querySelector("#cv-bbox-w")?.value) > 0
      && Number(document.querySelector("#cv-bbox-h")?.value) > 0
      && Number(document.querySelector("#cv-frames")?.value) > 0
      && Number(document.querySelector("#cv-fps")?.value) > 0
    ));
    assert(await page.locator("#cv-bbox-w").isDisabled(), "LTX2 width input must be profile-locked");
    assert(await page.locator("#cv-bbox-h").isDisabled(), "LTX2 height input must be profile-locked");
    assert(await page.locator("#cv-frames").isDisabled(), "LTX2 frames input must be profile-locked");
    assert(await page.locator("#cv-fps").isDisabled(), "LTX2 FPS input must be profile-locked");
    const profileValues = await page.locator("#cv-ltx2-profile option").evaluateAll(
      (options) => options.map((option) => option.value),
    );
    assert(profileValues.length >= 13, `expected all compiled mode-compatible LTX2 profiles, got ${profileValues.length}`);
    assert(profileValues.includes(temporalEditMode
      ? "960x544_121f_24fps"
      : "960x512_121f_24fps"), `missing 540p landscape 5s ${temporalEditMode ? "source-native " : ""}profile`);
    assert(profileValues.includes(temporalEditMode
      ? "544x960_121f_24fps"
      : "512x960_121f_24fps"), `missing 540p portrait 5s ${temporalEditMode ? "source-native " : ""}profile`);
    assert(profileValues.includes("1088x1920_121f_24fps"), "missing 1080p portrait profile");
    assert(Number(await page.locator("#cv-frames").getAttribute("max")) >= 481,
      "frames control must represent the native 20-second / 481-frame profile");
    const resolutionValues = await page.locator("#cv-ltx2-resolution option").evaluateAll(
      (options) => options.map((option) => option.value),
    );
    assert(resolutionValues.length === 7,
      `expected all 7 native LTX2 resolutions, got ${resolutionValues.join(",")}`);
    for (const expected of [
      "creator-portrait",
      ...(temporalEditMode
        ? ["540p-source-landscape", "540p-source-portrait"]
        : ["540p-landscape", "540p-portrait"]),
      "720p-landscape", "720p-portrait", "1080p-landscape", "1080p-portrait",
    ]) {
      assert(resolutionValues.includes(expected), `missing native resolution ${expected}`);
    }
    const requiredDurations = {
      "creator-portrait": ["4.84"],
      [temporalEditMode ? "540p-source-landscape" : "540p-landscape"]: ["5"],
      [temporalEditMode ? "540p-source-portrait" : "540p-portrait"]: ["5"],
      "720p-landscape": ["5", "6", "8", "10"],
      "720p-portrait": ["5", "6", "8", "10"],
      "1080p-landscape": ["5"],
      "1080p-portrait": ["5"],
    };
    for (const [resolution, durations] of Object.entries(requiredDurations)) {
      await page.selectOption("#cv-ltx2-resolution", resolution);
      const actual = await page.locator("#cv-ltx2-duration-options option").evaluateAll(
        (options) => options.map((option) => option.value),
      );
      for (const duration of durations)
        assert(actual.includes(duration), `${resolution} missing duration=${duration}; actual=${actual.join(",")}`);
    }
    const features = await page.locator("#cv-ltx2-feature option").evaluateAll(
      (options) => options.map((option) => ({
        value: option.value,
        disabled: option.disabled,
        text: option.textContent,
      })),
    );
    const cinemagraph = features.find((feature) => feature.value === "cinemagraph");
    const foley = features.find((feature) => feature.value === "foley-v2a");
    const icPending = features.find((feature) => feature.value === "clean-plate");
    assert(cinemagraph && !cinemagraph.disabled, "Cinemagraph feature must be admitted");
    assert(foley && !foley.disabled, "Foley feature must be admitted");
    assert(icPending && icPending.disabled, "IC-LoRA feature must remain disabled until its runner is admitted");
    assert(await page.locator("#cv-ltx2-feature-weight").isDisabled(),
      "feature weight must be disabled for the standard workflow");
    const upscalers = await page.locator("#cv-ltx2-post-upscaler option").evaluateAll(
      (options) => options.map((option) => ({
        value: option.value,
        disabled: option.disabled,
        text: option.textContent,
      })),
    );
    const realesrgan = upscalers.find((upscaler) => upscaler.value === "realesrgan-x4plus");
    const realesrganFast = upscalers.find((upscaler) => upscaler.value === "realesrgan-fast-x4v3");
    const seedvr2 = upscalers.find((upscaler) => upscaler.value === "seedvr2-3b");
    assert(realesrgan && !realesrgan.disabled, "installed Real-ESRGAN must be selectable");
    assert(realesrganFast && realesrganFast.disabled, "missing fast Real-ESRGAN must be disabled");
    assert(seedvr2 && seedvr2.disabled, "source-only SeedVR2 must be disabled");
    await page.selectOption("#cv-ltx2-post-upscaler", "realesrgan-x4plus");
    assert(!(await page.locator("#cv-ltx2-post-upscale-factor").isDisabled()),
      "post-upscale factor must be enabled for an admitted upscaler");
    await page.selectOption("#cv-ltx2-post-upscaler", "none");
    await page.selectOption("#cv-ltx2-resolution", nativeResolution);
    await page.waitForFunction((duration) => Array.from(
      document.querySelectorAll("#cv-ltx2-duration-options option"),
    ).some((option) => option.value === duration), durationText);
    await page.locator("#cv-ltx2-duration").fill(durationText);
    await page.locator("#cv-ltx2-duration").dispatchEvent("change");
    await page.selectOption("#cv-ltx2-quant", quantization);
    await page.selectOption("#cv-ltx2-camera-motion", cameraMotion);
    expectedProfile = await page.evaluate(() => ({
      width: Number(document.querySelector("#cv-bbox-w")?.value),
      height: Number(document.querySelector("#cv-bbox-h")?.value),
      frames: Number(document.querySelector("#cv-frames")?.value),
      fps: Number(document.querySelector("#cv-fps")?.value),
      duration: Number(document.querySelector("#cv-ltx2-duration")?.value),
      quant: document.querySelector("#cv-ltx2-quant")?.value,
    }));
    assert(expectedProfile.duration === Number(durationText),
      `duration=${expectedProfile.duration}`);
    if (nativeResolution === "540p-portrait" && durationText === "10") {
      assert(expectedProfile.width === 512 && expectedProfile.height === 960,
        `10s portrait geometry=${expectedProfile.width}x${expectedProfile.height}`);
      assert(expectedProfile.frames === 241 && expectedProfile.fps === 24,
        `10s portrait profile=${expectedProfile.frames}f@${expectedProfile.fps}`);
    }
    if (skipTemplate) {
      await page.locator("#cv-prompt").fill(promptText || (featureId === "cinemagraph"
        ? "CINEMAGRAPH_MOTION only the candle flame moves"
        : "A calm portrait comes alive with subtle natural motion and natural Foley sound"));
      await page.locator("#cv-caps-positive").fill("");
      await page.locator("#cv-caps-positive").dispatchEvent("input");
    } else {
      await page.click("#cv-load-ltx2-template");
      await page.waitForFunction(() => (
        document.querySelector("#cv-steps")?.value === "8"
        && document.querySelector("#cv-ltx2-mode")?.value === "distilled"
        && document.querySelector("#cv-ltx2-quant")?.value === "bf16"
        && document.querySelector("#cv-model")?.value === "ltx-2.3-22b-distilled"
        && document.querySelector("#cv-caps-positive")?.value === ""
        && document.querySelectorAll("#cv-lora-list .cv-lora-row").length === 0
      ));
      assert(await page.locator("#cv-lora-clear").isDisabled(),
        "creator profile must not inject a LoRA");
      if (await page.locator("#cv-model-row").isVisible()) {
        await page.selectOption("#cv-model", "flux1-dev");
        await page.waitForFunction(() => (
          document.querySelectorAll("#cv-lora-list .cv-lora-row").length === 0
          && document.querySelector("#cv-lora-clear")?.disabled
          && document.querySelector("#cv-lora-compat")?.textContent.includes("0 loaded")
        ));
        await page.selectOption("#cv-model", "ltx-2.3-22b-dev-fp8");
        await page.waitForSelector("#cv-load-ltx2-template", { state: "visible" });
        await page.click("#cv-load-ltx2-template");
        await page.waitForFunction(() => (
          document.querySelector("#cv-model")?.value === "ltx-2.3-22b-distilled"
          && document.querySelectorAll("#cv-lora-list .cv-lora-row").length === 0
        ));
      }
    }
    if (!(await page.locator("#cv-prompt").inputValue()).trim()) {
      await page.locator("#cv-prompt").fill(promptText ||
        "A calm portrait comes alive with subtle natural motion and natural Foley sound");
    }
    if (sourceVideoPath) {
      assert(fs.existsSync(sourceVideoPath), `missing V2V source: ${sourceVideoPath}`);
      await page.setInputFiles("#cv-import-file", sourceVideoPath);
      await page.waitForFunction(() => {
        const video = document.querySelector("#cv-source-video");
        const fallback = document.querySelector("#cv-source-video-fallback");
        return (video && video.videoWidth > 0) ||
          (fallback && fallback.style.display === "block");
      });
      await page.waitForFunction(() =>
        /Source ready/.test(document.querySelector("#cv-source-video-note")?.textContent || ""));
      if (v2vStrengthText) {
        await page.locator("#cv-ltx2-source-strength").fill(v2vStrengthText);
        await page.locator("#cv-ltx2-source-strength").dispatchEvent("input");
      } else {
        assert(await page.locator("#cv-ltx2-source-strength").inputValue() === "0",
          "V2V import must default to Replace subject with zero source preservation");
        if (canvasMode === "create")
          assert(await page.locator("#cv-ltx2-v2v-presets button.active").textContent() === "Replace subject",
            "Replace subject preset must be visibly active");
      }
      expectedProfile = await page.evaluate(() => ({
        width: Number(document.querySelector("#cv-bbox-w")?.value),
        height: Number(document.querySelector("#cv-bbox-h")?.value),
        frames: Number(document.querySelector("#cv-frames")?.value),
        fps: Number(document.querySelector("#cv-fps")?.value),
        duration: Number(document.querySelector("#cv-ltx2-duration")?.value),
        quant: document.querySelector("#cv-ltx2-quant")?.value,
      }));
    }
    if (sourceImagePath) {
      assert(fs.existsSync(sourceImagePath), `missing I2V source: ${sourceImagePath}`);
      await page.setInputFiles("#cv-import-file", sourceImagePath);
      await page.waitForSelector("#cv-source-preview", { state: "visible" });
      await page.waitForTimeout(100);
      assert(await page.locator("#cv-ltx2-source-strength").inputValue() === "1",
        "I2V source preservation must default to 1.0");
      if (lastImagePath) {
        assert(fs.existsSync(lastImagePath), `missing last-frame source: ${lastImagePath}`);
        await page.setInputFiles("#cv-ltx2-last-frame-file", lastImagePath);
        await page.waitForFunction(() =>
          /Final keyframe ready/.test(
            document.querySelector("#cv-ltx2-last-frame-note")?.textContent || ""
          ));
      }
    }
    if (featureId !== "standard") {
      await page.selectOption("#cv-ltx2-feature", featureId);
      if (featureWeightText) {
        await page.locator("#cv-ltx2-feature-weight").fill(featureWeightText);
        await page.locator("#cv-ltx2-feature-weight").dispatchEvent("input");
      }
    }
    if (screenshotPath)
      await page.screenshot({ path: screenshotPath, fullPage: true });
    await page.click("#cv-generate-btn");
    if (realRun) {
      const startedAt = Date.now();
      let lastStatus = "";
      while (Date.now() - startedAt < 15 * 60 * 1000) {
        const staged = await page.locator("#staging-panel").evaluateAll(
          (nodes) => nodes.some((node) => node.style.display === "flex"),
        );
        if (staged) break;
        const error = await page.locator("#cv-error-banner.visible").textContent().catch(() => "");
        if (error) throw new Error(error);
        const status = await page.locator("#cv-progress-label").textContent().catch(() => "");
        if (status && status !== lastStatus) {
          console.log(`[ltx2] ${status}`);
          lastStatus = status;
        }
        await page.waitForTimeout(1000);
      }
    } else {
      await page.waitForFunction(() => document.querySelector("#staging-panel")?.style.display === "flex");
    }
    assert(
      await page.locator("#staging-panel").evaluateAll(
        (nodes) => nodes.some((node) => node.style.display === "flex"),
      ),
      "completed video did not reach Canvas staging",
    );

    assert(request, "Canvas did not submit /v1/video");
    assert(request.runner === "ltx2_mojo_request", `runner=${request.runner}`);
    assert(request.guidance_mode === "distilled", `guidance_mode=${request.guidance_mode}`);
    assert(request.steps === 8, `steps=${request.steps}`);
    assert(request.sampler === "euler", `sampler=${request.sampler}`);
    assert(request.scheduler === "ltx2_distilled", `scheduler=${request.scheduler}`);
    assert(request.width === expectedProfile.width && request.height === expectedProfile.height,
      `geometry=${request.width}x${request.height}`);
    assert(request.frames === expectedProfile.frames && request.fps === expectedProfile.fps,
      `video=${request.frames}f@${request.fps}`);
    assert(request.quant === expectedProfile.quant, `quant=${request.quant}`);
    assert(request.camera_motion === cameraMotion,
      `camera_motion=${request.camera_motion}`);
    assert(request.checkpoint === (expectedProfile.quant === "bf16"
      ? "ltx-2.3-22b-distilled"
      : "ltx-2.3-22b-dev-fp8"), `checkpoint=${request.checkpoint}`);
    if (sourceVideoPath) {
      assert(typeof request.video_path === "string" && request.video_path.endsWith(".mp4"),
        `video_path=${request.video_path}`);
      const expectedVideoStrength = featureId === "foley-v2a"
        ? 1
        : (v2vStrengthText
          ? Number(v2vStrengthText)
          : 0);
      assert(request.video_strength === expectedVideoStrength,
        `video_strength=${request.video_strength}`);
      assert(!request.image_path, `unexpected image_path=${request.image_path}`);
      const expectedEditMode = canvasMode === "retake_ltx23" ? "retake" :
        (canvasMode === "extend_ltx23" ? "extend_end" : "standard");
      assert(request.video_edit_mode === expectedEditMode,
        `video_edit_mode=${request.video_edit_mode}`);
      if (canvasMode === "retake_ltx23") {
        assert(request.video_edit_start === 0, `retake start=${request.video_edit_start}`);
        assert(request.video_edit_end === 2, `retake end=${request.video_edit_end}`);
      }
      if (canvasMode === "extend_ltx23") {
        assert(request.frames > 121, `extend target frames=${request.frames}`);
      }
      if (canvasMode === "retake_ltx23") {
        assert(request.audio_policy === "preserve",
          `Retake must preserve source audio, got ${request.audio_policy}`);
      }
      if (canvasMode === "extend_ltx23") {
        assert(request.audio_policy === "generate",
          `Extend must regenerate the extension audio, got ${request.audio_policy}`);
      }
    }
    if (sourceImagePath) {
      assert(typeof request.image_path === "string" && request.image_path.length > 0,
        `image_path=${request.image_path}`);
      assert(fs.existsSync(request.image_path), `missing staged I2V upload=${request.image_path}`);
      assert(fs.readFileSync(request.image_path).equals(fs.readFileSync(sourceImagePath)),
        `I2V upload differs from the exact selected source: ${request.image_path}`);
      assert(request.image_strength === 1, `image_strength=${request.image_strength}`);
      assert(!request.video_path, `unexpected video_path=${request.video_path}`);
      if (lastImagePath) {
        assert(typeof request.last_image_path === "string" &&
          request.last_image_path.length > 0,
          `last_image_path=${request.last_image_path}`);
        assert(fs.existsSync(request.last_image_path),
          `missing staged last-frame upload=${request.last_image_path}`);
        assert(fs.readFileSync(request.last_image_path).equals(
          fs.readFileSync(lastImagePath)),
          `last-frame upload differs from exact selected source: ${request.last_image_path}`);
        assert(request.last_image_strength === 1,
          `last_image_strength=${request.last_image_strength}`);
      }
    }
    assert(Array.isArray(request.lora), "lora request field is not an array");
    assert(request.feature_id === featureId, `feature_id=${request.feature_id}`);
    if (featureId !== "standard") {
      const expectedWeight = featureWeightText
        ? Number(featureWeightText)
        : (featureId === "cinemagraph" ? 0.9 : 1.0);
      assert(request.feature_weight === expectedWeight,
        `feature_weight=${request.feature_weight}`);
    }
    if (featureId === "foley-v2a") {
      assert(request.include_audio === true, "Foley must generate audio");
      assert(request.audio_policy === "generate", `audio_policy=${request.audio_policy}`);
    }
    assert(request.lora.length === 0, `unexpected lora count=${request.lora.length}`);
    assert(request.caps_positive === "", `automatic conditioning override=${request.caps_positive}`);
    assert(errors.length === 0, `page errors: ${errors.join(" | ")}`);
    console.log(JSON.stringify({
      status: "PASS",
      real_run: realRun,
      v2v: Boolean(sourceVideoPath),
      i2v: Boolean(sourceImagePath),
      request,
      staged_video_src: await page.locator("#staging-panel video").getAttribute("src"),
    }, null, 2));
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error.stack || error);
  process.exit(1);
});
