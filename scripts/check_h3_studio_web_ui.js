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
        return require(path.join(npxRoot, entry, "node_modules", "playwright")).chromium;
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
    executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE || "/usr/bin/google-chrome",
    args: ["--no-sandbox", "--disable-dev-shm-usage"],
  });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  const pageErrors = [];
  let videoPosts = 0;

  page.on("pageerror", (error) => pageErrors.push(String(error)));
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (request.method() === "POST" && url.pathname === "/v1/video") videoPosts += 1;
  });
  await page.addInitScript(() => {
    localStorage.setItem("sf-active-tab", "h3-studio");
    if (!sessionStorage.getItem("h3-studio-gate-initialized")) {
      localStorage.removeItem("serenity-h3-current-project-v1");
      sessionStorage.setItem("h3-studio-gate-initialized", "1");
    }
  });

  try {
    await page.goto(baseUrl, { waitUntil: "networkidle" });
    await page.locator("#panel-h3-studio .h3s-app").waitFor({ state: "visible" });

    const initial = await page.evaluate(() => {
      const panel = document.querySelector("#panel-h3-studio");
      const workspace = document.querySelector(".h3s-workspace");
      return {
        activeRail: document.querySelector('.nav-btn[data-tab="h3-studio"]').classList.contains("active"),
        panelVisible: panel.offsetParent !== null,
        format: document.querySelector('[data-project-field="project_kind"]').value,
        targetSeconds: Number(document.querySelector('[data-project-field="target_duration_seconds"]').value),
        directorActions: document.querySelectorAll("#h3s-director-action option").length,
        shotCards: document.querySelectorAll(".h3s-shot-card").length,
        continuitySpine: document.querySelector(".h3s-timeline").textContent.includes("Continuity spine"),
        directorButton: document.querySelector('[data-h3-action="prepare-director"]').textContent.trim(),
        runtimeUnavailable: document.querySelector(".h3s-header-stats").innerText.includes("Runtime unavailable"),
        noGpuStatus: document.querySelector("#h3s-status-message").textContent.includes("never starts GPU work"),
        gridColumns: getComputedStyle(workspace).gridTemplateColumns,
        panelBounds: panel.getBoundingClientRect().toJSON(),
        bodyWidth: document.body.scrollWidth,
      };
    });

    assert(initial.activeRail && initial.panelVisible, "H3 Studio did not become the active inference tab");
    assert(initial.format === "1" && initial.targetSeconds === 120, "movie-first defaults drifted");
    assert(initial.directorActions === 11, `expected 11 Director workflows, found ${initial.directorActions}`);
    assert(initial.shotCards === 1 && initial.continuitySpine,
      `project shot deck or continuity spine missing: cards=${initial.shotCards}, spine=${initial.continuitySpine}`);
    assert(initial.directorButton === "Prepare Qwen Director pass", "Qwen Director preparation control missing");
    assert(initial.runtimeUnavailable && initial.noGpuStatus, "fail-closed runtime status is not visible");
    assert(initial.gridColumns.split(" ").length === 3, `expected three-column filmmaker workspace: ${initial.gridColumns}`);
    assert(initial.bodyWidth <= 1920, `desktop page overflows viewport: ${initial.bodyWidth}px`);
    assert(videoPosts === 0, "opening H3 Studio launched a video request");

    const directorInput = page.locator('.h3s-stage [data-project-field="director_brief"]');
    await directorInput.fill("A two-minute love story: two people meet under odd circumstances, collide in funny recurring scenes, and finally click.");
    await page.locator('[data-h3-action="prepare-director"]').click();
    await page.locator(".h3s-request").waitFor({ state: "visible" });
    const prepared = await page.locator(".h3s-request pre").textContent();
    assert(prepared.includes('"schema": "serenity.h3.caption.v2"'), "Director request schema missing");
    assert(prepared.includes('"operation": "dream_project"'), "Dream Project operation missing");
    assert(prepared.includes('"minimum_shots": 8') && prepared.includes('"maximum_shots": 24'), "two-minute shot envelope missing");
    assert(videoPosts === 0, "preparing a Director request launched GPU work");

    await page.locator("#h3s-director-action").selectOption("character_sheet");
    await page.locator("#h3s-character-panels").selectOption("4");
    const fourPanelWarning = await page.locator(".h3s-stage .h3s-warning").textContent();
    assert(fourPanelWarning.includes("metadata only") && fourPanelWarning.includes("five-second"), "four-panel safety warning missing");
    assert(videoPosts === 0, "Character Sheet planning launched GPU work");

    await page.locator("#h3s-project-title").fill("Odd Terms");
    await page.locator('[data-h3-action="add-shot"]').click();
    assert(await page.locator(".h3s-shot-card").count() === 2, "Add shot did not update the project");
    await page.reload({ waitUntil: "networkidle" });
    await page.locator("#panel-h3-studio .h3s-app").waitFor({ state: "visible" });
    assert(await page.locator("#h3s-project-title").inputValue() === "Odd Terms", "project title did not persist");
    assert(await page.locator(".h3s-shot-card").count() === 2, "shot deck did not persist");
    assert(videoPosts === 0, "reload launched GPU work");

    const artifactDir = path.join(process.cwd(), "output", "checks", "h3_studio_web");
    fs.mkdirSync(artifactDir, { recursive: true });
    const screenshot = path.join(artifactDir, "h3-studio-desktop.png");
    await page.screenshot({ path: screenshot, fullPage: true });

    assert(pageErrors.length === 0, `browser page errors: ${pageErrors.join(" | ")}`);
    console.log(JSON.stringify({
      status: "PASS",
      baseUrl,
      videoPosts,
      directorActions: initial.directorActions,
      persistedShots: 2,
      screenshot,
    }, null, 2));
  } finally {
    await browser.close();
  }
}

run().catch((error) => {
  console.error(error.stack || error);
  process.exit(1);
});
