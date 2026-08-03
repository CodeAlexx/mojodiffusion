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
    executablePath:
      process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE || "/usr/bin/google-chrome",
    args: ["--no-sandbox", "--disable-dev-shm-usage"],
  });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  const errors = [];
  page.on("pageerror", (error) => errors.push(String(error)));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  await page.addInitScript(() => {
    localStorage.clear();
    localStorage.setItem("sf-active-tab", "workflows");
  });
  await page.route("**/v1/jobs", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: "[]",
    });
  });
  await page.route("**/v1/job/workflow-handoff", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id: "workflow-handoff",
        created: "Thu, 30 Jul 2026 21:15:00 GMT",
        model: "ideogram-4-fp8",
        state: "done",
        metadata: {
          model: "ideogram-4-fp8",
          prompt: "Workflow handoff result",
          width: 1024,
          height: 1024,
          steps: 20,
          cfg: 7,
          sampler: "euler",
          scheduler: "simple",
          seed: 12345,
          params: {
            model: "ideogram-4-fp8",
            prompt: "Workflow handoff result",
            width: 1024,
            height: 1024,
            steps: 20,
            cfg: 7,
            sampler: "euler",
            scheduler: "simple",
            seed: 12345,
          },
        },
      }),
    });
  });
  await page.route("**/view?*", async (route) => {
    const url = new URL(route.request().url());
    if (url.searchParams.get("filename") !== "workflow-handoff.png") {
      await route.continue();
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: "image/png",
      body: Buffer.from(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
        "base64"
      ),
    });
  });

  try {
    await page.goto(baseUrl, { waitUntil: "networkidle" });
    await page.waitForFunction(() =>
      window.GenerateTab &&
      window.SerenityWS &&
      typeof sfToolbar !== "undefined"
    );
    const isolation = await page.evaluate(() => {
      GenerateTab.init();
      GenerateTab.state.generating = false;
      GenerateTab.state.pendingBatch = 0;
      localStorage.removeItem("sf-view-image");
      sfToolbar._activePromptId = "workflow-handoff";
      sfToolbar._awaitingPrompt = false;
      SerenityWS._emit("execution_start", {
        prompt_id: "workflow-handoff",
      });
      return {
        generating: GenerateTab.state.generating,
        activity: document.querySelector("#gen-activity-text").textContent.trim(),
        activityState: document.querySelector("#gen-activity-status").dataset.state,
        batch: document.querySelector("#gen-batch-status").textContent.trim(),
        globalActivity: document.querySelector(".queue-label").textContent.trim(),
      };
    });
    assert(
      isolation.generating === false &&
      isolation.activity.includes("Workflow") &&
      isolation.activityState === "loading" &&
      isolation.batch === "Idle" &&
      isolation.globalActivity.startsWith("Running"),
      `Workflow activity was not shared without batch leakage: ${JSON.stringify(isolation)}`
    );

    const running = await page.evaluate(() => {
      SerenityWS._emit("progress", {
        prompt_id: "workflow-handoff",
        phase: "Encoding prompt",
        value: 12,
        max: 48,
      });
      switchTab("generate");
      return {
        tab: localStorage.getItem("sf-active-tab"),
        generating: GenerateTab.state.generating,
        activity: document.querySelector("#gen-activity-text").textContent.trim(),
        activityState: document.querySelector("#gen-activity-status").dataset.state,
        batch: document.querySelector("#gen-batch-status").textContent.trim(),
        globalActivity: document.querySelector(".queue-label").textContent.trim(),
      };
    });
    assert(
      running.tab === "generate" &&
      running.generating === false &&
      running.activity.includes("Workflow") &&
      running.activity.includes("Encoding prompt") &&
      running.activityState === "encoding" &&
      running.batch === "Idle" &&
      running.globalActivity.startsWith("Running"),
      `Running Workflow status did not survive the Generate switch: ${JSON.stringify(running)}`
    );

    await page.evaluate(() => {
      SerenityWS._emit("executed", {
        prompt_id: "workflow-handoff",
        node: "save",
        output: {
          images: [{
            filename: "workflow-handoff.png",
            subfolder: "",
            type: "output",
          }],
        },
      });
      SerenityWS._emit("execution_success", {
        prompt_id: "workflow-handoff",
      });
    });

    await page.waitForFunction(() =>
      GenerateTab.state.currentImage &&
      GenerateTab.state.currentImage.includes("workflow-handoff.png") &&
      GenerateTab.state.gallery.some((item) =>
        item.prompt === "Workflow handoff result" &&
        item.model === "ideogram-4-fp8"
      )
    );
    const handoff = await page.evaluate(() => ({
      currentImage: GenerateTab.state.currentImage,
      generating: GenerateTab.state.generating,
      pendingView: localStorage.getItem("sf-view-image"),
      activity: document.querySelector("#gen-activity-text").textContent.trim(),
      activityState: document.querySelector("#gen-activity-status").dataset.state,
      batch: document.querySelector("#gen-batch-status").textContent.trim(),
      globalActivity: document.querySelector(".queue-label").textContent.trim(),
      galleryMatches: GenerateTab.state.gallery.filter((item) =>
        item.prompt === "Workflow handoff result" &&
        item.model === "ideogram-4-fp8"
      ).length,
      previewVisible:
        getComputedStyle(document.querySelector("#gen-preview-img")).display !== "none",
    }));
    assert(
      handoff.generating === false &&
      handoff.pendingView === null &&
      handoff.activity === "Idle" &&
      handoff.activityState === "idle" &&
      handoff.batch === "Idle" &&
      handoff.globalActivity === "Idle" &&
      handoff.galleryMatches === 1 &&
      handoff.previewVisible,
      `Workflow result did not open in Generate: ${JSON.stringify(handoff)}`
    );
    assert(errors.length === 0, `browser errors: ${errors.join(" | ")}`);
    console.log("serenity workflow result handoff: PASS");
    console.log(JSON.stringify({ isolation, running, handoff }, null, 2));
  } finally {
    await browser.close();
  }
}

run().catch((error) => {
  console.error("serenity workflow result handoff: FAIL");
  console.error(error && error.stack ? error.stack : String(error));
  process.exit(1);
});
