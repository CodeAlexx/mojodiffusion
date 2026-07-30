#!/usr/bin/env node
"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");

function loadChromium() {
  try {
    return require("playwright").chromium;
  } catch (_) { /* tooling may live in the npx cache */ }
  const npxRoot = path.join(os.homedir(), ".npm", "_npx");
  try {
    for (const entry of fs.readdirSync(npxRoot)) {
      try {
        return require(path.join(
          npxRoot, entry, "node_modules", "playwright"
        )).chromium;
      } catch (_) { /* continue */ }
    }
  } catch (_) { /* no npx cache */ }
  return null;
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function run() {
  const chromium = loadChromium();
  if (!chromium) throw new Error("Playwright is not installed");
  const baseUrl = process.env.SERENITY_BASE_URL || "http://127.0.0.1:7811";
  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ||
      "/usr/bin/google-chrome",
    args: ["--no-sandbox", "--disable-dev-shm-usage"],
  });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  const videoBodies = [];
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(String(error)));
  page.on("console", (message) => {
    if (message.type() === "error") pageErrors.push(message.text());
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
        video_id: "wan22-ui-gate",
        state: "queued",
        status_url: "/out/wan22-ui-gate/status.json",
        result_url: "/out/wan22-ui-gate/result.json",
      }),
    });
  });
  await page.route("**/out/wan22-ui-gate/status.json", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        state: "running", step: 1, total: 50, message: "Wan UI gate",
      }),
    });
  });

  try {
    await page.goto(baseUrl, { waitUntil: "networkidle" });
    await page.locator('.nav-btn[data-tab="generate"]').click();
    await page.waitForFunction(() =>
      window.GenerateTab && GenerateTab.state.allModels &&
      GenerateTab.state.allModels.length > 0
    );
    const wanModel = await page.evaluate(() => {
      const model = GenerateTab.state.allModels.find((candidate) =>
        ModelUtils.archForModel(candidate.name) === "wan"
      );
      return model && model.name;
    });
    assert(wanModel, "Wan2.2 TI2V-5B is missing from the model registry");
    await page.evaluate((name) => GenerateTab.selectModel(name), wanModel);

    const t2v = await page.evaluate(() => ({
      model: GenerateTab.state.model,
      arch: GenerateTab.state.arch,
      width: GenerateTab.state.width,
      height: GenerateTab.state.height,
      frames: GenerateTab.state.frames,
      fps: GenerateTab.state.fps,
      steps: GenerateTab.state.steps,
      quant: GenerateTab.state.videoQuant,
      sampler: GenerateTab.state.sampler,
      scheduler: GenerateTab.state.scheduler,
      note: document.querySelector("#gen-video-profile-note").textContent.trim(),
      options: Array.from(
        document.querySelector("#gen-aspect-dropdown").options
      ).map((option) => option.value),
      nativeProfiles: (() => {
        const runners = GenerateTab.state.videoStatus &&
          GenerateTab.state.videoStatus.candidate_runners;
        const runner = Array.isArray(runners)
          ? runners.find((entry) => entry && entry.model === "wan22_t2v")
          : null;
        return runner && Array.isArray(runner.native_profiles)
          ? runner.native_profiles.map((profile) =>
            `${profile.width}×${profile.height}`)
          : [];
      })(),
    }));
    assert(
      t2v.arch === "wan" && t2v.width === 1280 && t2v.height === 704 &&
      t2v.frames === 121 && t2v.fps === 24 && t2v.steps === 50 &&
      t2v.quant === "bf16" && t2v.sampler === "uni_pc" &&
      t2v.scheduler === "normal" && t2v.note.includes("BF16"),
      `Wan T2V UI contract drifted: ${JSON.stringify(t2v)}`
    );
    assert(
      ["1280×704", "704×1280", "1248×704", "704×1248"]
        .every((size) => t2v.nativeProfiles.includes(size)),
      `Wan native profiles are incomplete: ${t2v.nativeProfiles.join(", ")}`
    );

    await page.evaluate((name) => {
      GenerateTab.state.initImagePath = "/tmp/wan-cyborg-544x960.png";
      GenerateTab.state.initImageWidth = 544;
      GenerateTab.state.initImageHeight = 960;
      GenerateTab.selectModel(name);
      const prompt = document.querySelector("#gen-prompt");
      prompt.value = "A cyborg woman blinks naturally while staying identical.";
      prompt.dispatchEvent(new Event("input", { bubbles: true }));
    }, wanModel);
    const i2v = await page.evaluate(() => ({
      width: GenerateTab.state.width,
      height: GenerateTab.state.height,
      steps: GenerateTab.state.steps,
      quant: GenerateTab.state.videoQuant,
      note: document.querySelector("#gen-video-profile-note").textContent.trim(),
    }));
    assert(
      i2v.width === 704 && i2v.height === 1248 && i2v.steps === 50 &&
      i2v.quant === "bf16" && i2v.note.includes("creator source-aspect"),
      `Wan creator-sized I2V UI contract drifted: ${JSON.stringify(i2v)}`
    );

    await page.locator("#gen-btn").click();
    await page.waitForFunction(() => document.querySelector(
      "#gen-activity-status"
    ).dataset.state !== "idle");
    assert(videoBodies.length === 1, "Wan I2V request was not submitted");
    const request = videoBodies[0];
    assert(
      request.model === "wan22" && request.width === 704 &&
      request.height === 1248 && request.frames === 121 &&
      request.fps === 24 && request.steps === 50 &&
      request.guidance === 5 && request.quant === "bf16" &&
      request.sampler === "uni_pc" && request.scheduler === "normal" &&
      request.image_path === "/tmp/wan-cyborg-544x960.png",
      `Wan I2V request drifted: ${JSON.stringify(request)}`
    );
    assert(pageErrors.length === 0, `browser errors: ${pageErrors.join(" | ")}`);
    console.log("serenity Wan2.2 UI: PASS");
    console.log(JSON.stringify({ model: wanModel, t2v, i2v, request }, null, 2));
  } finally {
    await browser.close();
  }
}

run().catch((error) => {
  console.error("serenity Wan2.2 UI: FAIL");
  console.error(error && error.stack || error);
  process.exit(1);
});
