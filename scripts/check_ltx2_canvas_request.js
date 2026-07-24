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
  let request = null;

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
    await page.goto(baseUrl, { waitUntil: "domcontentloaded" });
    await page.click('.nav-btn[data-tab="canvas"]');
    await page.waitForSelector("#cv-model");
    await page.waitForFunction(() => Array.from(
      document.querySelectorAll("#cv-model option"),
    ).some((option) => option.value === "ltx-2.3-22b-dev-fp8"));
    await page.selectOption("#cv-model", "ltx-2.3-22b-dev-fp8");
    await page.waitForSelector("#cv-load-ltx2-template", { state: "visible" });
    await page.click("#cv-load-ltx2-template");
    await page.waitForFunction(() => (
      document.querySelector("#cv-steps")?.value === "8"
      && document.querySelector("#cv-ltx2-mode")?.value === "distilled"
      && document.querySelector("#cv-caps-positive")?.value.length > 0
    ));
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
    assert(request.width === 512 && request.height === 768, `geometry=${request.width}x${request.height}`);
    assert(request.frames === 121 && request.fps === 25, `video=${request.frames}f@${request.fps}`);
    assert(Array.isArray(request.lora) && request.lora.length === 1, `lora count=${request.lora?.length}`);
    assert(request.lora[0].name === "ltx2_eri2_step3000", `lora=${request.lora[0].name}`);
    assert(errors.length === 0, `page errors: ${errors.join(" | ")}`);
    console.log(JSON.stringify({
      status: "PASS",
      real_run: realRun,
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
