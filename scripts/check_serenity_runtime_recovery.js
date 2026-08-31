#!/usr/bin/env node
"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");

function loadChromium() {
  try { return require("playwright").chromium; } catch (_) { /* continue */ }
  const npxRoot = path.join(os.homedir(), ".npm", "_npx");
  try {
    for (const entry of fs.readdirSync(npxRoot)) {
      try { return require(path.join(npxRoot, entry, "node_modules", "playwright")).chromium; }
      catch (_) { /* continue */ }
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
  const baseUrl = process.env.SERENITY_BASE_URL || "http://127.0.0.1:7801";
  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE || "/usr/bin/google-chrome",
    args: ["--no-sandbox", "--disable-dev-shm-usage"],
  });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  let jobs = [];
  page.on("pageerror", (error) => errors.push(String(error)));
  await page.route("**/v1/jobs", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(jobs),
    });
  });
  try {
    console.log("[gate] load");
    await page.goto(baseUrl, { waitUntil: "domcontentloaded", timeout: 20000 });
    await page.locator('.nav-btn[data-tab="generate"]').click();
    await page.waitForFunction(() => window.SerenityWS && SerenityWS.isConnected());
    console.log("[gate] websocket");
    const reconnectVisible = await page.locator("#gen-ws-indicator").evaluate((node) =>
      node.classList.contains("visible")
    );
    assert(!reconnectVisible, "healthy socket is still labeled Reconnecting");

    await page.evaluate(() => {
      GenerateTab.state.prompt = "";
      document.querySelector("#gen-prompt").value = "";
      document.querySelector("#gen-btn").click();
    });
    await page.locator("#gen-error-banner.visible").waitFor({ state: "visible" });
    await page.waitForTimeout(5200);
    console.log("[gate] error expiry");
    const clearedError = await page.locator("#gen-error-banner").evaluate((node) => ({
      visible: node.classList.contains("visible"),
      text: node.textContent,
    }));
    assert(!clearedError.visible && clearedError.text === "",
      `expired Generate error was retained: ${JSON.stringify(clearedError)}`);

    await page.locator('.nav-btn[data-tab="queue"]').click();
    await page.evaluate(() => {
      QueueTab.init();
      QueueTab.registerPending({
        promptId: "job-race",
        prompt: "race completion",
        model: "zimage_base",
        queuedAt: Date.now(),
      });
      SerenityWS._emit("execution_success", { prompt_id: "job-race" });
    });
    await page.waitForFunction(() =>
      QueueTab.state.pending.every((entry) => entry.promptId !== "job-race") &&
      QueueTab.state.history.some((entry) => entry.promptId === "job-race" && entry.status === "success")
    );
    console.log("[gate] terminal race");

    jobs = [{
      id: "job-reconcile",
      model: "krea2",
      state: "done",
      error: "",
      params: { prompt: "server reconciliation" },
      output_location: { relative_path: "job-reconcile.png" },
    }];
    await page.evaluate(() => {
      QueueTab.registerPending({
        promptId: "job-reconcile",
        prompt: "server reconciliation",
        model: "krea2",
        queuedAt: Date.now(),
      });
      QueueTab.reconcile();
    });
    await page.waitForFunction(() =>
      QueueTab.state.pending.every((entry) => entry.promptId !== "job-reconcile") &&
      QueueTab.state.history.some((entry) => entry.promptId === "job-reconcile" && entry.status === "success")
    );
    console.log("[gate] server reconciliation");
    assert(errors.length === 0, `browser errors: ${errors.join(" | ")}`);
    console.log("Serenity runtime recovery gate passed");
  } finally {
    await browser.close();
  }
}

run().catch((error) => {
  console.error(error.stack || error);
  process.exit(1);
});
