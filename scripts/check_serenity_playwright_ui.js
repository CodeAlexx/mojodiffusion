#!/usr/bin/env node
"use strict";

const fs = require("fs");
const http = require("http");
const net = require("net");
const os = require("os");
const path = require("path");
const { spawn } = require("child_process");

function loadChromium() {
  try {
    return require("playwright").chromium;
  } catch (_) {
    // Playwright is intentionally tooling-only and may live in npm's npx cache
    // or the documented /tmp tool directory instead of this public repo.
  }
  const candidates = [
    path.join(os.tmpdir(), "mojodiffusion-playwright-tools", "node_modules", "playwright"),
  ];
  const npxRoot = path.join(os.homedir(), ".npm", "_npx");
  try {
    for (const entry of fs.readdirSync(npxRoot)) {
      candidates.push(path.join(npxRoot, entry, "node_modules", "playwright"));
    }
  } catch (_) { /* no npx cache */ }
  for (const candidate of candidates) {
    try {
      return require(candidate).chromium;
    } catch (_) { /* try the next tooling location */ }
  }
  return null;
}

const chromium = loadChromium();
if (!chromium) {
  console.error("serenity playwright ui: FAIL");
  console.error("Playwright is not installed for this Node environment.");
  process.exit(2);
}

const ROOT = path.resolve(__dirname, "..");
const SERVER_BIN = process.env.SERENITY_SERVER_BIN ||
  path.join(ROOT, "serenity-server/target/debug/serenity-server");
const WORKER_BIN = process.env.SERENITY_WORKER_BIN ||
  path.join(ROOT, "output/bin/serenity_worker_stub");
const CHROME_BIN = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ||
  firstExisting(["/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser"]);
const H3_CONTINUE_MP4_BASE64 = "AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAMybW9vdgAAAGxtdmhkAAAAAAAAAAAAAAAAAAAD6AAAA+gAAQAAAQAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAlx0cmFrAAAAXHRraGQAAAADAAAAAAAAAAAAAAABAAAAAAAAA+gAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAACAAAAAgAAAAAAAkZWR0cwAAABxlbHN0AAAAAAAAAAEAAAPoAAAAAAABAAAAAAHUbWRpYQAAACBtZGhkAAAAAAAAAAAAAAAAAABAAAAAQABVxAAAAAAALWhkbHIAAAAAAAAAAHZpZGUAAAAAAAAAAAAAAABWaWRlb0hhbmRsZXIAAAABf21pbmYAAAAUdm1oZAAAAAEAAAAAAAAAAAAAACRkaW5mAAAAHGRyZWYAAAAAAAAAAQAAAAx1cmwgAAAAAQAAAT9zdGJsAAAAv3N0c2QAAAAAAAAAAQAAAK9hdmMxAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAACAAIABIAAAASAAAAAAAAAABFUxhdmM2MC4zMS4xMDIgbGlieDI2NAAAAAAAAAAAAAAAGP//AAAANWF2Y0MBZAAK/+EAGGdkAAqs2UlsBEAAAAMAQAAAAwEDxIllgAEABmjr48siwP34+AAAAAAQcGFzcAAAAAEAAAABAAAAFGJ0cnQAAAAAAAAW6AAAFugAAAAYc3R0cwAAAAAAAAABAAAAAgAAIAAAAAAUc3RzcwAAAAAAAAABAAAAAQAAABxzdHNjAAAAAAAAAAEAAAABAAAAAgAAAAEAAAAcc3RzegAAAAAAAAAAAAAAAgAAAtAAAAANAAAAFHN0Y28AAAAAAAAAAQAAA2IAAABidWR0YQAAAFptZXRhAAAAAAAAACFoZGxyAAAAAAAAAABtZGlyYXBwbAAAAAAAAAAAAAAAAC1pbHN0AAAAJal0b28AAAAdZGF0YQAAAAEAAAAATGF2ZjYwLjE2LjEwMAAAAAhmcmVlAAAC5W1kYXQAAAKtBgX//6ncRem95tlIt5Ys2CDZI+7veDI2NCAtIGNvcmUgMTY0IHIzMTA4IDMxZTE5ZjkgLSBILjI2NC9NUEVHLTQgQVZDIGNvZGVjIC0gQ29weWxlZnQgMjAwMy0yMDIzIC0gaHR0cDovL3d3dy52aWRlb2xhbi5vcmcveDI2NC5odG1sIC0gb3B0aW9uczogY2FiYWM9MSByZWY9MyBkZWJsb2NrPTE6MDowIGFuYWx5c2U9MHgzOjB4MTEzIG1lPWhleCBzdWJtZT03IHBzeT0xIHBzeV9yZD0xLjAwOjAuMDAgbWl4ZWRfcmVmPTEgbWVfcmFuZ2U9MTYgY2hyb21hX21lPTEgdHJlbGxpcz0xIDh4OGRjdD0xIGNxbT0wIGRlYWR6b25lPTIxLDExIGZhc3RfcHNraXA9MSBjaHJvbWFfcXBfb2Zmc2V0PS0yIHRocmVhZHM9MSBsb29rYWhlYWRfdGhyZWFkcz0xIHNsaWNlZF90aHJlYWRzPTAgbnI9MCBkZWNpbWF0ZT0xIGludGVybGFjZWQ9MCBibHVyYXlfY29tcGF0PTAgY29uc3RyYWluZWRfaW50cmE9MCBiZnJhbWVzPTMgYl9weXJhbWlkPTIgYl9hZGFwdD0xIGJfYmlhcz0wIGRpcmVjdD0xIHdlaWdodGI9MSBvcGVuX2dvcD0wIHdlaWdodHA9MiBrZXlpbnQ9MjUwIGtleWludF9taW49MiBzY2VuZWN1dD00MCBpbnRyYV9yZWZyZXNoPTAgcmNfbG9va2FoZWFkPTQwIHJjPWNyZiBtYnRyZWU9MSBjcmY9MjMuMCBxY29tcD0wLjYwIHFwbWluPTAgcXBtYXg9NjkgcXBzdGVwPTQgaXBfcmF0aW89MS40MCBhcT0xOjEuMDAAgAAAABtliIQAFP/+7Np+BTcMVvn10yG94AC3K4+Aln0AAAAJQZohbEEv/rXA";

function firstExisting(candidates) {
  return candidates.find((candidate) => candidate && fs.existsSync(candidate)) || "";
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function assertExactValues(actual, expected, label) {
  assert(
    actual.length === expected.length && actual.every((value, index) => value === expected[index]),
    `${label} exposed [${actual.join(", ")}], expected [${expected.join(", ")}]`,
  );
}

function getFreePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const port = server.address().port;
      server.close(() => resolve(port));
    });
  });
}

function getJson(url) {
  return new Promise((resolve, reject) => {
    const request = http.get(url, { timeout: 1000 }, (response) => {
      let body = "";
      response.setEncoding("utf8");
      response.on("data", (chunk) => { body += chunk; });
      response.on("end", () => {
        try {
          resolve({ status: response.statusCode, body: JSON.parse(body) });
        } catch (error) {
          reject(error);
        }
      });
    });
    request.once("timeout", () => request.destroy(new Error("health timeout")));
    request.once("error", reject);
  });
}

async function waitForHealth(baseUrl, server) {
  let lastError = null;
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (server && server.exitCode !== null) {
      throw new Error(`server exited before health check: ${server.exitCode}`);
    }
    try {
      const result = await getJson(`${baseUrl}/v1/health`);
      if (result.status === 200 && result.body && result.body.status === "ok") return result.body;
    } catch (error) {
      lastError = error;
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`server did not become healthy: ${lastError ? lastError.message : "unknown"}`);
}

function parseRequestJson(request) {
  try {
    return JSON.parse(request.postData() || "{}");
  } catch (error) {
    throw new Error(`request body is not JSON: ${error.message}`);
  }
}

async function loadTemplate(page, index) {
  await page.locator("#btn-templates").click();
  const item = page.locator("#templates-list .wf-template-item").nth(index);
  await item.click();
  await page.waitForTimeout(50);
}

async function run() {
  let server = null;
  let outDir = null;
  let baseUrl = process.env.SERENITY_BASE_URL || process.argv[2] || "";
  const serverLog = [];

  if (!baseUrl) {
    assert(fs.existsSync(SERVER_BIN), `server binary not found: ${SERVER_BIN}`);
    assert(fs.existsSync(WORKER_BIN), `worker binary not found: ${WORKER_BIN}`);
    const port = await getFreePort();
    outDir = fs.mkdtempSync(path.join(os.tmpdir(), "serenity-playwright-ui-"));
    baseUrl = `http://127.0.0.1:${port}`;
    server = spawn(SERVER_BIN, [
      "--worker", WORKER_BIN,
      "--kind", "stub",
      "--port", String(port),
      "--out-dir", outDir,
    ], {
      cwd: ROOT,
      env: process.env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    server.stdout.on("data", (chunk) => serverLog.push(chunk.toString()));
    server.stderr.on("data", (chunk) => serverLog.push(chunk.toString()));
  }

  let health;
  try {
    health = await waitForHealth(baseUrl, server);
  } catch (error) {
    if (serverLog.length) console.error(serverLog.join("").slice(-6000));
    throw error;
  }
  const browser = await chromium.launch({
    headless: true,
    executablePath: CHROME_BIN || undefined,
    args: ["--no-sandbox", "--disable-dev-shm-usage"],
  });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const pageErrors = [];
  const consoleErrors = [];
  const requestFailures = [];
  const promptRequests = [];
  const generateRequests = [];
  const videoRequests = [];
  let modelRegistryAttempts = 0;
  let releaseVideoIdentity;
  const videoIdentityGate = new Promise((resolve) => { releaseVideoIdentity = resolve; });

  page.on("pageerror", (error) => pageErrors.push(String(error)));
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("requestfailed", (request) => {
    if (request.url().startsWith(baseUrl)) {
      requestFailures.push(`${request.method()} ${request.url()}: ${request.failure()?.errorText || "failed"}`);
    }
  });
  await page.route("**/v1/models?refresh=1", async (route) => {
    modelRegistryAttempts += 1;
    if (modelRegistryAttempts === 1) {
      await route.fulfill({
        status: 503,
        contentType: "application/json",
        body: JSON.stringify({ error: "simulated server restart" }),
      });
      return;
    }
    await route.continue();
  });
  await page.route("**/prompt", async (route) => {
    promptRequests.push(parseRequestJson(route.request()));
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ prompt_id: `playwright-image-${promptRequests.length}` }),
    });
  });
  await page.route("**/v1/generate", async (route) => {
    generateRequests.push(parseRequestJson(route.request()));
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ job_id: `playwright-generate-${generateRequests.length}` }),
    });
  });
  await page.route("**/v1/video", async (route) => {
    if (route.request().method() !== "POST") {
      await route.continue();
      return;
    }
    const request = parseRequestJson(route.request());
    videoRequests.push(request);
    if (request.prompt === "Immutable Wan video identity") await videoIdentityGate;
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        schema: "serenity.video_result.v1",
        video_id: `playwright-video-${videoRequests.length}`,
        state: "done",
        accepted_video_artifact: true,
        mp4_url: "/out/playwright-video/ltx2_refhq.mp4",
        width: request.width,
        height: request.height,
        steps: request.steps,
        guidance: request.guidance,
      }),
    });
  });
  await page.route("**/playwright-h3-continue.mp4", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "video/mp4",
      body: Buffer.from(H3_CONTINUE_MP4_BASE64, "base64"),
    });
  });
  const restoredGalleryJob = {
    id: "job-9999",
    model: "sd_xl_base_1.0",
    state: "done",
    output_path: "job-9999.png",
    metadata: {
      prompt: "Immutable SDXL gallery identity",
      model: "sd_xl_base_1.0",
      seed: 2468,
      steps: 20,
      cfg: 7,
      scheduler: "euler",
      width: 1024,
      height: 1024,
    },
    params: {
      model: "sd_xl_base_1.0",
      prompt: "Immutable SDXL gallery identity",
      seed: 2468,
      steps: 20,
      cfg: 7,
      scheduler: "euler",
      width: 1024,
      height: 1024,
    },
  };
  await page.route("**/v1/jobs", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([restoredGalleryJob]),
    });
  });
  await page.route("**/v1/history/artifacts", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        schema: "serenity.history_artifacts.v1",
        items: [{
          id: "root-000:job-9999.png",
          url: "/out/job-9999.png",
          media_type: "image",
          timestamp: 1,
          params: restoredGalleryJob.params,
        }],
      }),
    });
  });
  await page.route("**/view?filename=job-9999.png*", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "image/png",
      body: Buffer.from(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
        "base64",
      ),
    });
  });
  await page.route("**/out/job-9999.png", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "image/png",
      body: Buffer.from(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
        "base64",
      ),
    });
  });
  await page.route("**/out/playwright-video/ltx2_refhq.mp4", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "video/mp4",
      body: Buffer.alloc(0),
    });
  });
  try {
    await page.goto(baseUrl, { waitUntil: "domcontentloaded", timeout: 15000 });
    await page.waitForSelector("#gen-btn", { state: "attached", timeout: 15000 });
    await page.waitForFunction(() => !!(
      window.GenerateTab
      && window.SerenityAPI
      && window.WorkflowBuilder
      && GenerateTab.state
      && Array.isArray(GenerateTab.state.allModels)
      && GenerateTab.state.allModels.length > 0
    ));

    const initialModel = await page.evaluate(() => GenerateTab.state.model);
    assert(modelRegistryAttempts >= 2,
      `model registry did not retry after a transient server restart (${modelRegistryAttempts} request)`);
    assert(/krea.*turbo/i.test(initialModel), `Generate defaulted to ${initialModel}, expected Krea-2 Turbo`);
    if (process.env.SERENITY_REGISTRY_RETRY_ONLY === "1") {
      const loadedModelCount = await page.evaluate(() => GenerateTab.state.allModels.length);
      assert(loadedModelCount > 0, "model registry retry completed with an empty picker");
      assert(pageErrors.length === 0, `page errors: ${pageErrors.join(" | ")}`);
      assert(requestFailures.length === 0,
        `same-origin request failures: ${requestFailures.join(" | ")}`);
      console.log("serenity model registry restart retry: PASS");
      console.log(JSON.stringify({
        schema: "serenity.playwright.model_registry_retry.v1",
        model_registry_attempts: modelRegistryAttempts,
        loaded_model_count: loadedModelCount,
        selected_model: initialModel,
      }));
      return;
    }

    const mainVideoAudioState = await page.evaluate(() => {
      const video = document.querySelector("#gen-preview-video");
      return video && {
        muted: video.muted,
        defaultMuted: video.defaultMuted,
        mutedAttribute: video.hasAttribute("muted"),
        controls: video.controls,
      };
    });
    assert(mainVideoAudioState, "Generate video preview is absent");
    assert(
      !mainVideoAudioState.muted
        && !mainVideoAudioState.defaultMuted
        && !mainVideoAudioState.mutedAttribute,
      `Generate video preview forced mute: ${JSON.stringify(mainVideoAudioState)}`,
    );
    assert(mainVideoAudioState.controls, "Generate video preview has no playback controls");

    const generateMetadataPlacement = await page.evaluate(() => ({
      inBatchPanel: !!document.querySelector(
        ".gen-workspace-batch-panel > #gen-metadata-panel",
      ),
      overPreview: !!document.querySelector(
        ".gen-workspace-stage #gen-metadata-panel",
      ),
    }));
    assert(generateMetadataPlacement.inBatchPanel && !generateMetadataPlacement.overPreview,
      `Generate result metadata still covers the preview: ${JSON.stringify(generateMetadataPlacement)}`);

    await page.waitForFunction(() => {
      const item = GenerateTab.state.gallery && GenerateTab.state.gallery[0];
      return item && item.model === "sd_xl_base_1.0" && item.width === 1024 && item.height === 1024;
    });
    const restoredGalleryIdentity = await page.evaluate(() => {
      const item = GenerateTab.state.gallery[0];
      return { item };
    });
    for (const [label, item] of Object.entries(restoredGalleryIdentity)) {
      assert(item.model === "sd_xl_base_1.0", `${label} gallery model remained ${item.model}`);
      assert(item.prompt === "Immutable SDXL gallery identity", `${label} gallery prompt remained ${item.prompt}`);
      assert(item.seed === 2468, `${label} gallery seed remained ${item.seed}`);
      assert(item.steps === 20 && item.cfg === 7, `${label} gallery settings remained steps=${item.steps} cfg=${item.cfg}`);
      assert(item.width === 1024 && item.height === 1024, `${label} gallery size remained ${item.width}x${item.height}`);
      assert(item.arch === "sdxl", `${label} gallery architecture remained ${item.arch}`);
    }
    const restoredGalleryBadges = await page.evaluate(() => {
      const thumb = document.querySelector("#gen-gallery-grid .gen-thumb-wrap");
      if (!thumb) return { error: "restored gallery thumbnail was not rendered" };
      thumb.click();
      const model = document.querySelector("#gen-model-badge");
      const arch = document.querySelector("#gen-arch-badge");
      return {
        model: model ? model.textContent : "",
        modelTitle: model ? model.title : "",
        arch: arch ? arch.textContent : "",
        archKey: arch ? arch.dataset.arch : "",
      };
    });
    assert(!restoredGalleryBadges.error, restoredGalleryBadges.error || "gallery badge test failed");
    assert(
      restoredGalleryBadges.modelTitle === "sd_xl_base_1.0" && restoredGalleryBadges.archKey === "sdxl",
      `clicked SDXL gallery item showed model=${restoredGalleryBadges.modelTitle} arch=${restoredGalleryBadges.archKey}`,
    );
    assert(restoredGalleryBadges.arch === "SDXL", `clicked SDXL gallery badge read ${restoredGalleryBadges.arch}`);

    // A video POST remains open for the full render. Prove that changing the
    // selected model while it is in flight cannot rewrite the completed
    // gallery item's model/prompt/seed identity.
    await page.evaluate(() => GenerateTab.applyParams({
      model: "Wan2.2-TI2V-5B-Mojo",
      prompt: "Immutable Wan video identity",
      negPrompt: "",
      width: 832,
      height: 480,
      steps: 50,
      cfg: 5,
      scheduler: "uni_pc",
      seed: 4242,
      frames: 121,
      fps: 24,
    }));
    const identityVideoBefore = videoRequests.length;
    await page.locator("#gen-btn").click();
    for (let attempt = 0; attempt < 100 && videoRequests.length === identityVideoBefore; attempt += 1) {
      await page.waitForTimeout(20);
    }
    assert(videoRequests.length === identityVideoBefore + 1, "identity video request was not submitted");
    await page.evaluate(() => GenerateTab.applyParams({
      model: "krea2-turbo",
      prompt: "A later Krea selection",
      seed: 9999,
    }));
    releaseVideoIdentity();
    await page.waitForFunction(() => (GenerateTab.state.gallery || []).some(
      (item) => item.prompt === "Immutable Wan video identity" && item.isVideo,
    ));
    const immutableVideoIdentity = await page.evaluate(() => (GenerateTab.state.gallery || []).find(
      (item) => item.prompt === "Immutable Wan video identity" && item.isVideo,
    ));
    assert(
      immutableVideoIdentity.model === "Wan2.2-TI2V-5B-Mojo"
        && immutableVideoIdentity.seed === 4242
        && immutableVideoIdentity.width === 832
        && immutableVideoIdentity.height === 480
        && immutableVideoIdentity.steps === 50
        && immutableVideoIdentity.cfg === 5
        && immutableVideoIdentity.guidance === 5
        && immutableVideoIdentity.arch === "wan",
      `video gallery identity drifted: ${JSON.stringify(immutableVideoIdentity)}`,
    );

    const galleryScrollLayout = await page.evaluate(() => {
      const grid = document.querySelector("#gen-gallery-grid");
      const original = grid && grid.querySelector(".gen-thumb-wrap");
      const groupItems = grid && grid.querySelector(".gen-gallery-date-items");
      const library = grid && grid.closest(".gen-workspace-library");
      if (!grid || !original || !groupItems || !library) return { error: "history layout elements are absent" };
      for (let index = 1; index < 50; index += 1) {
        const clone = original.cloneNode(true);
        clone.dataset.layoutFixture = "true";
        groupItems.appendChild(clone);
      }
      const items = Array.from(groupItems.querySelectorAll(".gen-thumb-wrap"));
      const rects = items.map((item) => item.getBoundingClientRect());
      const gridRect = grid.getBoundingClientRect();
      const libraryRect = library.getBoundingClientRect();
      let overlaps = 0;
      for (let left = 0; left < rects.length; left += 1) {
        for (let right = left + 1; right < rects.length; right += 1) {
          const a = rects[left];
          const b = rects[right];
          if (Math.min(a.right, b.right) > Math.max(a.left, b.left)
              && Math.min(a.bottom, b.bottom) > Math.max(a.top, b.top)) overlaps += 1;
        }
      }
      const result = {
        overflowY: getComputedStyle(grid).overflowY,
        flexDirection: getComputedStyle(grid).flexDirection,
        clientHeight: grid.clientHeight,
        scrollHeight: grid.scrollHeight,
        itemCount: items.length,
        overlaps,
        gridBottom: gridRect.bottom,
        libraryBottom: libraryRect.bottom,
      };
      grid.scrollTop = grid.scrollHeight;
      result.scrollTop = grid.scrollTop;
      result.maxScroll = grid.scrollHeight - grid.clientHeight;
      grid.querySelectorAll('[data-layout-fixture="true"]').forEach((node) => node.remove());
      grid.scrollTop = 0;
      return result;
    });
    assert(!galleryScrollLayout.error, galleryScrollLayout.error || "gallery layout test failed");
    assert(galleryScrollLayout.overflowY === "scroll", `history overflow was ${galleryScrollLayout.overflowY}`);
    assert(galleryScrollLayout.flexDirection === "column", `history direction was ${galleryScrollLayout.flexDirection}`);
    assert(galleryScrollLayout.itemCount === 50 && galleryScrollLayout.overlaps === 0,
      `history thumbnails overlapped: ${JSON.stringify(galleryScrollLayout)}`);
    assert(
      galleryScrollLayout.scrollHeight > galleryScrollLayout.clientHeight
        && galleryScrollLayout.scrollTop === galleryScrollLayout.maxScroll,
      `gallery did not scroll: ${galleryScrollLayout.scrollTop}/${galleryScrollLayout.maxScroll}`,
    );
    assert(
      galleryScrollLayout.gridBottom <= galleryScrollLayout.libraryBottom + 1,
      `history escaped library bounds: grid bottom ${galleryScrollLayout.gridBottom}, library bottom ${galleryScrollLayout.libraryBottom}`,
    );

    // Generate -> Workflow: the graph must be rebuilt from the same bounded
    // product request, including prompt, shape, sampler controls, and seed.
    const generateToWorkflow = await page.evaluate(async () => {
      GenerateTab.applyParams({
        model: "zimage_base",
        prompt: "cross-screen Generate to Workflow smoke",
        negPrompt: "",
        width: 512,
        height: 512,
        steps: 16,
        cfg: 5,
        guidance: 3.5,
        scheduler: "euler",
        seed: 4242,
        frames: 121,
        fps: 24,
        loras: [],
      });
      document.querySelector('.nav-btn[data-tab="workflows"]').click();
      await new Promise((resolve) => setTimeout(resolve, 50));
      const workflow = serializeWorkflow(sfCanvas).prompt;
      return {
        nodeCount: Object.keys(workflow).length,
        params: WorkflowSync.extract(workflow),
      };
    });
    assert(generateToWorkflow.nodeCount > 0, "Generate -> Workflow produced an empty graph");
    assert(generateToWorkflow.params.model === "zimage_base", "Generate model did not reach Workflow");
    assert(
      generateToWorkflow.params.prompt === "cross-screen Generate to Workflow smoke",
      "Generate prompt did not reach Workflow",
    );
    assert(
      generateToWorkflow.params.width === 512 && generateToWorkflow.params.height === 512,
      `Generate shape became ${generateToWorkflow.params.width}x${generateToWorkflow.params.height} in Workflow`,
    );
    assert(
      generateToWorkflow.params.steps === 16 && generateToWorkflow.params.cfg === 5 &&
        generateToWorkflow.params.seed === 4242,
      `Generate settings drifted in Workflow: ${JSON.stringify(generateToWorkflow.params)}`,
    );

    // Workflow template -> Generate: both screens must now describe the same
    // request, and Ctrl+Enter belongs only to the visible Workflow screen.
    await page.locator("#btn-templates").click();
    await page.waitForFunction(
      () => Array.from(document.querySelectorAll("#templates-list .wf-template-item"))
        .some((item) => /SDXL/.test(item.textContent || "")),
    );
    const sdxlTemplate = page.locator("#templates-list .wf-template-item").filter({ hasText: "SDXL" }).first();
    await sdxlTemplate.click();
    await page.waitForTimeout(50);
    const templateSync = await page.evaluate(() => ({
      workflowName: document.querySelector("#workflow-name").value,
      graph: WorkflowSync.extract(serializeWorkflow(sfCanvas).prompt),
      generate: GenerateTab.getParams(),
    }));
    assert(templateSync.workflowName === "SDXL · Text to Image", `workflow name stayed ${templateSync.workflowName}`);
    for (const key of ["model", "prompt", "width", "height", "steps", "cfg", "scheduler"]) {
      assert(
        templateSync.graph[key] === templateSync.generate[key],
        `template ${key} split across screens: ${templateSync.graph[key]} vs ${templateSync.generate[key]}`,
      );
    }
    const shortcutBefore = promptRequests.length;
    await page.keyboard.press("Control+Enter");
    for (let attempt = 0; attempt < 50 && promptRequests.length === shortcutBefore; attempt += 1) {
      await page.waitForTimeout(20);
    }
    assert(
      promptRequests.length === shortcutBefore + 1,
      `Workflow Ctrl+Enter submitted ${promptRequests.length - shortcutBefore} requests instead of one`,
    );
    await page.locator('.nav-btn[data-tab="generate"]').click();
    const generateAfterTemplate = await page.evaluate(() => ({
      params: GenerateTab.getParams(),
      modelInput: document.querySelector("#gen-model-search").value,
      promptInput: document.querySelector("#gen-prompt").value,
    }));
    assert(generateAfterTemplate.params.model === "sdxl_unet_bf16", "SDXL template did not select SDXL in Generate");
    assert(generateAfterTemplate.modelInput === generateAfterTemplate.params.model, "Generate model control is stale");
    assert(generateAfterTemplate.promptInput === generateAfterTemplate.params.prompt, "Generate prompt control is stale");

    const modelResults = [];
    const modelNames = await page.evaluate(() => GenerateTab.state.allModels.map((model) => model.name));
    for (const modelName of modelNames) {
      const result = await page.evaluate(async (name) => {
        const search = document.querySelector("#gen-model-search");
        search.value = "";
        search.dispatchEvent(new Event("input", { bubbles: true }));
        search.click();
        const item = Array.from(document.querySelectorAll(".gen-model-dropdown-item"))
          .find((candidate) => candidate.dataset.model === name);
        if (!item) return { name, error: "model is absent from rendered picker" };
        item.click();
        let workflow;
        try {
          workflow = WorkflowBuilder.build({
            model: GenerateTab.state.model,
            prompt: "Playwright model routing test",
            negPrompt: "",
            width: GenerateTab.state.width,
            height: GenerateTab.state.height,
            steps: GenerateTab.state.steps,
            cfg: GenerateTab.state.cfg,
            guidance: GenerateTab.state.guidance,
            sampler: GenerateTab.state.sampler,
            scheduler: GenerateTab.state.scheduler,
            seed: 1234,
            frames: GenerateTab.state.frames,
            fps: GenerateTab.state.fps,
            ltx2Mode: GenerateTab.state.videoGuidanceMode,
            quantization: GenerateTab.state.videoQuant,
            ltx2WorkflowProfile: GenerateTab.state.videoWorkflowProfile,
            ltx2PromptEnhancer: GenerateTab.state.videoPromptEnhancer,
            ltx2AudioPolicy: GenerateTab.state.audioPolicy,
            ltx2PostUpscaler: GenerateTab.state.postUpscaler,
            ltx2PostUpscaleFactor: GenerateTab.state.postUpscaleFactor,
            ltx2CameraMotion: GenerateTab.state.cameraMotion,
            loras: [],
          });
        } catch (error) {
          return { name, selected: GenerateTab.state.model, error: String(error) };
        }
        const classes = Object.values(workflow).map((node) => node.class_type);
        const video = SerenityAPI.videoRequestFromWorkflow(workflow);
        if (video) {
          const isLtx = video.model === "ltx2";
          const isWan = video.model === "wan22";
          const isBernini = video.model === "bernini";
          const isH3 = video.model === "minimax_h3";
          const ltxRunner = isLtx
            ? (GenerateTab.state.videoStatus.candidate_runners || [])
              .find((entry) => entry && entry.model === "ltx2_t2v_av")
            : null;
          const ltxProfiles = ltxRunner && ltxRunner.modes
            && ltxRunner.modes.ltx2_mojo_request
            && Array.isArray(ltxRunner.modes.ltx2_mojo_request.supported_profiles)
            ? ltxRunner.modes.ltx2_mojo_request.supported_profiles : [];
          const exactAvailableLtxProfile = !isLtx || ltxProfiles.some((profile) =>
            profile.available === true
              && Array.isArray(profile.modes) && profile.modes.includes("standard")
              && Number(profile.width) === Number(video.width)
              && Number(profile.height) === Number(video.height)
              && Number(profile.frames) === Number(video.frames)
              && Number(profile.fps) === Number(video.fps));
          const h3Runner = isH3
            ? (GenerateTab.state.videoStatus.candidate_runners || [])
              .find((entry) => entry && entry.model === "minimax_h3_t2va")
            : null;
          const h3Geometry = h3Runner && h3Runner.geometry_constraints || {};
          const h3Quant = h3Runner && Array.isArray(h3Runner.quant_modes)
            ? h3Runner.quant_modes.find((mode) => mode.id === video.quant) : null;
          const h3Step = Number(h3Geometry.dimension_step) || 32;
          const h3Width = Number(video.width);
          const h3Height = Number(video.height);
          const exactAvailableH3Profile = !isH3 || !!(h3Runner
            && h3Runner.available === true && h3Quant && h3Quant.available === true
            && h3Width >= Number(h3Geometry.width_min)
            && h3Width <= Number(h3Geometry.width_max)
            && h3Height >= Number(h3Geometry.height_min)
            && h3Height <= Number(h3Geometry.height_max)
            && h3Width % h3Step === 0 && h3Height % h3Step === 0
            && h3Width * h3Height <= Number(h3Geometry.max_pixels)
            && Number(video.fps) >= Number(h3Geometry.fps_min)
            && Number(video.fps) <= Number(h3Geometry.fps_max)
            && Number(video.frames) >= 1);
          const admittedLtxDev = video.guidance_mode === "dev"
            && video.steps === 15 && video.sampler === "res2s" && video.scheduler === "ltx2";
          const admittedLtxDistilled = video.guidance_mode === "distilled"
            && video.steps === 8
            && ((video.sampler === "euler" && video.scheduler === "ltx2_distilled")
              || (video.sampler === "euler_ancestral_cfg_pp"
                && video.scheduler === "sulphur_creator_8_3"));
          return {
            name,
            selected: GenerateTab.state.model,
            nodes: Object.keys(workflow).length,
            sink: classes.includes("SaveVideo") ? "SaveVideo"
              : (classes.includes("SaveAnimatedWEBP") ? "SaveAnimatedWEBP" : ""),
            route: "/v1/video",
            admitted: (isLtx && video.runner === "ltx2_mojo_request"
                && ["fp8", "bf16", "int4"].includes(video.quant)
                && exactAvailableLtxProfile
                && (admittedLtxDev || admittedLtxDistilled))
              || (isWan && video.steps === 50 && video.guidance === 5
                && ["bf16", "fp8"].includes(video.quant))
              || (isH3 && video.runner === "minimax_h3_mojo_request"
                && ["int8-fast", "int8", "bf16"].includes(video.quant)
                && exactAvailableH3Profile)
              || (isBernini && video.width === 848 && video.height === 480
                && video.frames === 81 && video.fps === 16 && video.steps === 40
                && video.guidance === 4 && video.quant === "fp8"),
            backend: video.model,
            runner: video.runner,
            steps: video.steps,
            sampler: video.sampler,
            scheduler: video.scheduler,
            guidanceMode: video.guidance_mode,
            checkpoint: video.checkpoint,
            width: video.width,
            height: video.height,
            frames: video.frames,
            fps: video.fps,
            guidance: video.guidance,
            quant: video.quant,
            exactAvailableLtxProfile,
            exactAvailableH3Profile,
          };
        }
        const response = await fetch("/v1/preflight", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ workflow }),
        });
        const preflight = await response.json();
        return {
          name,
          selected: GenerateTab.state.model,
          nodes: Object.keys(workflow).length,
          sink: classes.includes("SaveImage") ? "SaveImage" : "",
          route: "/prompt",
          admitted: response.ok && preflight.admitted === true,
          backend: preflight.backend || (preflight.capability_profile || {}).backend || "",
          rejection: preflight.reason || preflight.error || "",
        };
      }, modelName);
      modelResults.push(result);
    }
    for (const result of modelResults) {
      assert(!result.error, `${result.name}: ${result.error}`);
      assert(result.selected === result.name, `${result.name}: picker selected ${result.selected}`);
      assert(result.nodes > 0, `${result.name}: generated an empty workflow`);
      assert(result.sink, `${result.name}: workflow has no media sink`);
      if ((result.backend === "ltx2" && !result.exactAvailableLtxProfile)
          || (result.backend === "minimax_h3" && !result.exactAvailableH3Profile)) {
        assert(!result.admitted,
          `${result.name}: LTX route bypassed the unavailable runtime ${JSON.stringify(result)}`);
      } else {
        assert(result.admitted, `${result.name}: route was not admitted ${JSON.stringify(result)}`);
      }
    }
    const h3Model = modelResults.find((result) => result.backend === "minimax_h3");
    if (h3Model) {
      const expectedH3Attention = await page.evaluate(() => {
        const runners = GenerateTab.state.videoStatus
          && GenerateTab.state.videoStatus.candidate_runners || [];
        const runner = runners.find((entry) => entry && entry.model === "minimax_h3_t2va");
        const ck = runner && Array.isArray(runner.attention_backends)
          ? runner.attention_backends.find((entry) => entry && entry.id === "ck-int8") : null;
        return ck && ck.available === true ? "ck-int8" : "cudnn";
      });
      await page.locator('.nav-btn[data-tab="generate"]').click();
      await page.evaluate((name) => {
        const search = document.querySelector("#gen-model-search");
        search.value = "";
        search.dispatchEvent(new Event("input", { bubbles: true }));
        search.click();
        const item = Array.from(document.querySelectorAll(".gen-model-dropdown-item"))
          .find((candidate) => candidate.dataset.model === name);
        if (!item) throw new Error(`H3 model ${name} is absent from Generate`);
        item.click();
      }, h3Model.name);
      await page.waitForFunction((name) => GenerateTab.state.model === name, h3Model.name);
      const readGenerateH3Controls = () => page.evaluate(() => ({
        aspect: document.querySelector("#gen-aspect-dropdown").value,
        aspects: Array.from(document.querySelector("#gen-aspect-dropdown").options)
          .map((option) => option.value),
        width: Number(document.querySelector("#gen-custom-width").value),
        widthMin: Number(document.querySelector("#gen-custom-width").min),
        widthMax: Number(document.querySelector("#gen-custom-width").max),
        widthStep: Number(document.querySelector("#gen-custom-width").step),
        widthDisabled: document.querySelector("#gen-custom-width").disabled,
        height: Number(document.querySelector("#gen-custom-height").value),
        heightMin: Number(document.querySelector("#gen-custom-height").min),
        heightMax: Number(document.querySelector("#gen-custom-height").max),
        heightStep: Number(document.querySelector("#gen-custom-height").step),
        heightDisabled: document.querySelector("#gen-custom-height").disabled,
        duration: Number(document.querySelector("#gen-seconds").value),
        durationMin: Number(document.querySelector("#gen-seconds").min),
        durationMax: Number(document.querySelector("#gen-seconds").max),
        steps: Number(document.querySelector("#gen-steps").value),
        stepsDisabled: document.querySelector("#gen-steps").disabled,
        stepCache: document.querySelector("#gen-h3-step-cache").value,
        frames: Number(document.querySelector("#gen-frames").value),
        framesMin: Number(document.querySelector("#gen-frames").min),
        framesMax: Number(document.querySelector("#gen-frames").max),
        fps: Number(document.querySelector("#gen-fps").value),
        fpsMin: Number(document.querySelector("#gen-fps").min),
        fpsMax: Number(document.querySelector("#gen-fps").max),
        quant: document.querySelector("#gen-video-quant").value,
        quants: Array.from(document.querySelector("#gen-video-quant").options)
          .map((option) => ({ id: option.value, disabled: option.disabled })),
        attention: document.querySelector("#gen-h3-attention").value,
        attentionDisabled: document.querySelector("#gen-h3-attention").disabled,
        attentionOptions: Array.from(document.querySelector("#gen-h3-attention").options)
          .map((option) => ({ id: option.value, disabled: option.disabled })),
      }));
      const setGenerateH3Geometry = (aspect, width, height, seconds, fps) => page.evaluate(
        ({ aspect, width, height, seconds, fps }) => {
          const fpsControl = document.querySelector("#gen-fps");
          fpsControl.value = String(fps);
          fpsControl.dispatchEvent(new Event("input", { bubbles: true }));
          const aspectControl = document.querySelector("#gen-aspect-dropdown");
          aspectControl.value = aspect;
          aspectControl.dispatchEvent(new Event("change", { bubbles: true }));
          if (aspect === "Free") {
            const widthControl = document.querySelector("#gen-custom-width");
            const heightControl = document.querySelector("#gen-custom-height");
            widthControl.value = String(width);
            widthControl.dispatchEvent(new Event("blur"));
            heightControl.value = String(height);
            heightControl.dispatchEvent(new Event("blur"));
          }
          const durationControl = document.querySelector("#gen-seconds");
          durationControl.value = String(seconds);
          durationControl.dispatchEvent(new Event("input", { bubbles: true }));
        },
        { aspect, width, height, seconds, fps },
      );

      const generateH3Default = await readGenerateH3Controls();
      assert(
        JSON.stringify(generateH3Default.aspects) === JSON.stringify([
          "1536×672", "1344×768", "1024×768",
          "768×768", "768×1024", "768×1344", "Free",
        ])
          && generateH3Default.widthMin === 32 && generateH3Default.widthMax === 2048
          && generateH3Default.heightMin === 32 && generateH3Default.heightMax === 2048
          && generateH3Default.widthStep === 32 && generateH3Default.heightStep === 32
          && !generateH3Default.widthDisabled && !generateH3Default.heightDisabled
          && generateH3Default.durationMin === 1 && generateH3Default.durationMax === 15
          && generateH3Default.steps === 20 && !generateH3Default.stepsDisabled
          && generateH3Default.stepCache === "exact"
          && generateH3Default.framesMin === 1 && generateH3Default.framesMax === 1800
          && generateH3Default.fpsMin === 1 && generateH3Default.fpsMax === 120
          && JSON.stringify(generateH3Default.quants.map((mode) => mode.id))
            === JSON.stringify(["int8-fast", "int8", "bf16"])
          && generateH3Default.quants.every((mode) => !mode.disabled)
          && generateH3Default.attention === expectedH3Attention
          && (expectedH3Attention === "ck-int8"
            || generateH3Default.attentionOptions.some((option) =>
              option.id === "ck-int8" && option.disabled)),
        `H3 Generate controls are locked or incomplete: ${JSON.stringify(generateH3Default)}`,
      );

      await page.locator("#gen-steps").fill("31");
      const generateH3AuthoredSteps = await readGenerateH3Controls();
      assert(generateH3AuthoredSteps.steps === 31 && !generateH3AuthoredSteps.stepsDisabled,
        `H3 Generate steps were not user-adjustable: ${JSON.stringify(generateH3AuthoredSteps)}`);

      const testedGenerateH3Presets = [
        ["1536×672", 1536, 672], ["1344×768", 1344, 768],
        ["1024×768", 1024, 768], ["768×768", 768, 768],
        ["768×1024", 768, 1024], ["768×1344", 768, 1344],
      ];
      for (const [aspect, width, height] of testedGenerateH3Presets) {
        await setGenerateH3Geometry(aspect, width, height, 2, 24);
        const controls = await readGenerateH3Controls();
        assert(controls.aspect === aspect
          && controls.width === width && controls.height === height
          && controls.duration === 2 && controls.frames === 48 && controls.fps === 24,
        `Generate H3 preset ${aspect} drifted: ${JSON.stringify(controls)}`);
      }
      await setGenerateH3Geometry("Free", 32, 2048, 1, 24);
      const generateH3Boundary = await readGenerateH3Controls();
      assert(generateH3Boundary.aspect === "Free"
        && generateH3Boundary.width === 32 && generateH3Boundary.height === 2048
        && generateH3Boundary.duration === 1 && generateH3Boundary.frames === 24,
      `Generate H3 32-to-2048 boundary was clamped: ${JSON.stringify(generateH3Boundary)}`);
      await setGenerateH3Geometry("Free", 992, 576, 4, 24);
      for (const quant of ["int8-fast", "int8", "bf16"]) {
        await page.evaluate((value) => {
          const control = document.querySelector("#gen-video-quant");
          control.value = value;
          control.dispatchEvent(new Event("change", { bubbles: true }));
        }, quant);
        const controls = await readGenerateH3Controls();
        assert(controls.quant === quant
          && controls.attention === expectedH3Attention
          && !controls.attentionDisabled
          && controls.width === 992 && controls.height === 576
          && controls.duration === 4 && controls.frames === 96,
        `Generate H3 ${quant} changed authored geometry: ${JSON.stringify(controls)}`);
      }
      await setGenerateH3Geometry("Free", 320, 192, 180, 24);
      const generateH3LongContext = await readGenerateH3Controls();
      assert(generateH3LongContext.aspect === "Free"
        && generateH3LongContext.width === 320 && generateH3LongContext.height === 192
        && generateH3LongContext.durationMax === 180
        && generateH3LongContext.duration === 180
        && generateH3LongContext.frames === 4320,
      `Generate H3 long-context controls did not expand: ${JSON.stringify(generateH3LongContext)}`);
      await page.locator("#gen-prompt").fill("Playwright H3 monolithic single-pass 180-second request");
      const generateH3VideoBefore = videoRequests.length;
      await page.locator("#gen-btn").click();
      for (let attempt = 0; attempt < 100 && videoRequests.length === generateH3VideoBefore; attempt += 1) {
        await page.waitForTimeout(20);
      }
      assert(videoRequests.length === generateH3VideoBefore + 1,
        "H3 Generate did not POST /v1/video");
      const generateH3Request = videoRequests[videoRequests.length - 1];
      assert(generateH3Request.model === "minimax_h3"
        && generateH3Request.runner === "minimax_h3_mojo_request"
        && generateH3Request.task === "t2va"
        && !Object.prototype.hasOwnProperty.call(generateH3Request, "source_image")
        && generateH3Request.quant === "bf16"
        && generateH3Request.attention_backend === expectedH3Attention
        && generateH3Request.width === 320 && generateH3Request.height === 192
        && generateH3Request.duration_seconds === 180
        && generateH3Request.frames === 4320 && generateH3Request.fps === 24
        && generateH3Request.steps === 31
        && generateH3Request.step_cache === "exact",
      `H3 Generate submitted the wrong runtime geometry: ${JSON.stringify(generateH3Request)}`);

      const h3GenerateSource = "/tmp/serenity-h3-generate-i2va-source.png";
      await page.evaluate((source) => {
        GenerateTab.state.initImagePath = source;
        GenerateTab.state.initImageName = "serenity-h3-generate-i2va-source.png";
      }, h3GenerateSource);
      await page.locator("#gen-prompt").fill("Playwright H3 Generate I2VA request");
      const generateH3I2vaBefore = videoRequests.length;
      await page.locator("#gen-btn").click();
      for (let attempt = 0;
        attempt < 100 && videoRequests.length === generateH3I2vaBefore;
        attempt += 1) {
        await page.waitForTimeout(20);
      }
      assert(videoRequests.length === generateH3I2vaBefore + 1,
        "H3 Generate I2VA did not POST /v1/video");
      const generateH3I2vaRequest = videoRequests[videoRequests.length - 1];
      assert(generateH3I2vaRequest.model === "minimax_h3"
        && generateH3I2vaRequest.runner === "minimax_h3_mojo_request"
        && generateH3I2vaRequest.task === "i2va"
        && generateH3I2vaRequest.source_image === h3GenerateSource
        && generateH3I2vaRequest.prompt === "Playwright H3 Generate I2VA request",
      `H3 Generate dropped its selected I2VA source: ${JSON.stringify(generateH3I2vaRequest)}`);
      await page.waitForFunction(() => {
        const panel = document.querySelector("#gen-metadata-panel");
        return panel && panel.classList.contains("visible");
      });
      const visibleMetadataPlacement = await page.evaluate(() => {
        const panel = document.querySelector("#gen-metadata-panel");
        const stage = document.querySelector(".gen-workspace-stage");
        const right = document.querySelector(".gen-workspace-batch-panel");
        const panelRect = panel.getBoundingClientRect();
        const stageRect = stage.getBoundingClientRect();
        return {
          visible: panel.offsetParent !== null,
          parentIsBatch: panel.parentElement === right,
          panelLeft: panelRect.left,
          stageRight: stageRect.right,
          overlapsStage: panelRect.left < stageRect.right
            && panelRect.right > stageRect.left
            && panelRect.top < stageRect.bottom
            && panelRect.bottom > stageRect.top,
        };
      });
      assert(visibleMetadataPlacement.visible
        && visibleMetadataPlacement.parentIsBatch
        && !visibleMetadataPlacement.overlapsStage
        && visibleMetadataPlacement.panelLeft >= visibleMetadataPlacement.stageRight,
      `visible result metadata still covers the preview: ${JSON.stringify(visibleMetadataPlacement)}`);
      await page.evaluate(() => {
        GenerateTab.state.initImagePath = "";
        GenerateTab.state.initImageName = "";
      });
      if (process.env.SERENITY_H3_GENERATE_ONLY === "1") {
        console.log("serenity H3 Generate source and metadata placement: PASS");
        console.log(JSON.stringify({
          schema: "serenity.playwright.h3_generate_source.v1",
          metadata_placement: generateMetadataPlacement,
          visible_metadata_placement: visibleMetadataPlacement,
          t2va_request: {
            task: generateH3Request.task,
            has_source_image: Object.prototype.hasOwnProperty.call(
              generateH3Request, "source_image",
            ),
          },
          i2va_request: {
            task: generateH3I2vaRequest.task,
            source_image: generateH3I2vaRequest.source_image,
          },
        }));
        return;
      }

      await page.locator('.nav-btn[data-tab="canvas"]').click();
      await page.waitForFunction(() => document.querySelectorAll("#cv-model option").length > 1);
      await page.locator("#cv-model").selectOption(h3Model.name);
      await page.waitForFunction(() => {
        const duration = document.querySelector("#cv-h3-duration");
        const resolution = document.querySelector("#cv-h3-resolution");
        return duration && duration.value !== "" && resolution &&
          resolution.options.length === 7 && resolution.value !== "";
      });
      const readH3Controls = () => page.evaluate(() => ({
        mode: document.querySelector("#cv-h3-mode").value,
        quant: document.querySelector("#cv-h3-quant").value,
        attention: document.querySelector("#cv-h3-attention").value,
        attentionDisabled: document.querySelector("#cv-h3-attention").disabled,
        attentionOptions: Array.from(document.querySelector("#cv-h3-attention").options)
          .map((option) => option.value),
        attentionOptionStates: Array.from(document.querySelector("#cv-h3-attention").options)
          .map((option) => ({ id: option.value, disabled: option.disabled })),
        resolution: document.querySelector("#cv-h3-resolution").value,
        resolutions: Array.from(document.querySelector("#cv-h3-resolution").options)
          .map((option) => option.value),
        width: Number(document.querySelector("#cv-h3-width").value),
        height: Number(document.querySelector("#cv-h3-height").value),
        bboxWidth: Number(document.querySelector("#cv-bbox-w").value),
        bboxHeight: Number(document.querySelector("#cv-bbox-h").value),
        duration: Number(document.querySelector("#cv-h3-duration").value),
        durationMin: Number(document.querySelector("#cv-h3-duration").min),
        durationMax: Number(document.querySelector("#cv-h3-duration").max),
        stepCache: document.querySelector("#cv-h3-step-cache").value,
        steps: Number(document.querySelector("#cv-steps").value),
        stepsDisabled: document.querySelector("#cv-steps").disabled,
        frames: Number(document.querySelector("#cv-frames").value),
        framesMin: Number(document.querySelector("#cv-frames").min),
        framesMax: Number(document.querySelector("#cv-frames").max),
        fps: Number(document.querySelector("#cv-h3-fps").value),
        fpsMin: Number(document.querySelector("#cv-h3-fps").min),
        fpsMax: Number(document.querySelector("#cv-h3-fps").max),
        geometryDisabled: [
          "#cv-h3-resolution", "#cv-h3-width", "#cv-h3-height",
          "#cv-h3-duration", "#cv-h3-fps",
        ].some((selector) => document.querySelector(selector).disabled),
        bboxGeometryLocked: ["#cv-bbox-w", "#cv-bbox-h"]
          .every((selector) => document.querySelector(selector).disabled),
        modeControlHidden: document.querySelector("#cv-h3-mode").type === "hidden",
        routeText: document.querySelector("#cv-h3-mode-note").textContent,
        mediaVisible: getComputedStyle(
          document.querySelector("#cv-h3-media-row")).display !== "none",
        genericFramesVisible: getComputedStyle(
          document.querySelector("#cv-video-frames-row")).display !== "none",
        genericFpsVisible: getComputedStyle(
          document.querySelector("#cv-video-fps-row")).display !== "none",
        continuationVisible: getComputedStyle(
          document.querySelector("#cv-h3-continue-row")).display !== "none",
        continuationSource: document.querySelector("#cv-h3-continue-source").textContent,
        contextFrames: Number(document.querySelector("#cv-h3-context-frames").value),
      }));
      const setH3Geometry = (resolution, seconds, fps, width, height) => page.evaluate(
        ({ resolution, seconds, fps, width, height }) => {
          const fpsControl = document.querySelector("#cv-h3-fps");
          fpsControl.value = String(fps);
          fpsControl.dispatchEvent(new Event("change", { bubbles: true }));
          const resolutionControl = document.querySelector("#cv-h3-resolution");
          resolutionControl.value = resolution;
          resolutionControl.dispatchEvent(new Event("change", { bubbles: true }));
          if (resolution === "custom") {
            const widthControl = document.querySelector("#cv-h3-width");
            const heightControl = document.querySelector("#cv-h3-height");
            widthControl.value = String(width);
            heightControl.value = String(height);
            widthControl.dispatchEvent(new Event("change", { bubbles: true }));
            heightControl.dispatchEvent(new Event("change", { bubbles: true }));
          }
          const duration = document.querySelector("#cv-h3-duration");
          duration.value = String(seconds);
          duration.dispatchEvent(new Event("change", { bubbles: true }));
        },
        { resolution, seconds, fps, width, height },
      );

      const h3Default = await readH3Controls();
      assert(
        JSON.stringify(h3Default.resolutions) === JSON.stringify([
          "1536x672", "1344x768", "1024x768",
          "768x768", "768x1024", "768x1344", "custom",
        ])
          && h3Default.resolution === "1344x768"
          && h3Default.width === 1344 && h3Default.height === 768
          && h3Default.bboxWidth === 1344 && h3Default.bboxHeight === 768
          && h3Default.durationMin === 1 && h3Default.durationMax === 15
          && h3Default.attention === expectedH3Attention && !h3Default.attentionDisabled
          && (expectedH3Attention === "ck-int8"
            || h3Default.attentionOptionStates.some((option) =>
              option.id === "ck-int8" && option.disabled))
          && JSON.stringify(h3Default.attentionOptions)
            === JSON.stringify(["ck-int8", "sage-int8", "cudnn"])
          && h3Default.stepCache === "exact"
          && h3Default.steps === 20 && !h3Default.stepsDisabled
          && h3Default.framesMin === 1 && h3Default.framesMax === 1800
          && h3Default.fpsMin === 1 && h3Default.fpsMax === 120
          && h3Default.mode === "t2va" && h3Default.modeControlHidden
          && h3Default.mediaVisible
          && !h3Default.genericFramesVisible && !h3Default.genericFpsVisible
          && !h3Default.geometryDisabled && h3Default.bboxGeometryLocked,
        `H3 native resolution controls are wrong: ${JSON.stringify(h3Default)}`,
      );

      await page.locator("#cv-steps").fill("37");
      const h3AuthoredSteps = await readH3Controls();
      assert(h3AuthoredSteps.steps === 37 && !h3AuthoredSteps.stepsDisabled,
        `H3 steps were not user-adjustable: ${JSON.stringify(h3AuthoredSteps)}`);

      await page.locator(".cv-h3-advanced summary").click();
      await page.locator("#cv-h3-attention").selectOption("cudnn");
      await page.locator("#cv-h3-quant").selectOption("int8-fast");
      const h3CudnnSwitch = await readH3Controls();
      assert(h3CudnnSwitch.attention === "cudnn"
        && !h3CudnnSwitch.attentionDisabled,
      `H3 cuDNN/Sage switch was overridden by INT8 Fast: ${JSON.stringify(h3CudnnSwitch)}`);
      await page.locator("#cv-h3-attention").selectOption("sage-int8");

      const testedH3Presets = [
        ["1536x672", 1536, 672], ["1344x768", 1344, 768],
        ["1024x768", 1024, 768], ["768x768", 768, 768],
        ["768x1024", 768, 1024], ["768x1344", 768, 1344],
      ];
      for (const [preset, width, height] of testedH3Presets) {
        await setH3Geometry(preset, 2, 24);
        const controls = await readH3Controls();
        assert(controls.resolution === preset
          && controls.width === width && controls.height === height
          && controls.bboxWidth === width && controls.bboxHeight === height
          && controls.duration === 2 && controls.frames === 48,
        `H3 preset ${preset} did not resolve exactly: ${JSON.stringify(controls)}`);
      }
      await setH3Geometry("custom", 3, 24, 992, 576);
      const h3Custom = await readH3Controls();
      assert(h3Custom.resolution === "custom"
        && h3Custom.width === 992 && h3Custom.height === 576
        && h3Custom.bboxWidth === 992 && h3Custom.bboxHeight === 576
        && h3Custom.duration === 3 && h3Custom.frames === 72,
      `H3 custom runtime geometry did not resolve: ${JSON.stringify(h3Custom)}`);

      await page.locator("#cv-h3-attention").selectOption(expectedH3Attention);
      await page.locator("#cv-h3-quant").selectOption("bf16");
      await setH3Geometry("1344x768", 2, 24);
      const h3TwoSeconds = await readH3Controls();
      assert(
        h3TwoSeconds.quant === "bf16"
          && h3TwoSeconds.attention === expectedH3Attention
          && !h3TwoSeconds.attentionDisabled
          && h3TwoSeconds.width === 1344 && h3TwoSeconds.height === 768
          && h3TwoSeconds.duration === 2 && h3TwoSeconds.frames === 48
          && h3TwoSeconds.fps === 24,
        `H3 2-second authored geometry did not remain exact: ${JSON.stringify(h3TwoSeconds)}`,
      );

      await page.locator("#cv-h3-quant").selectOption("int8");
      const h3Int8Attention = await readH3Controls();
      assert(h3Int8Attention.attention === expectedH3Attention
          && !h3Int8Attention.attentionDisabled,
        `H3 Sage switch did not return for INT8 Quality: ${JSON.stringify(h3Int8Attention)}`);
      await page.locator("#cv-h3-attention").selectOption("sage-int8");

      await setH3Geometry("1344x768", 4, 24);
      const h3FourSeconds = await readH3Controls();
      assert(
        h3FourSeconds.width === 1344 && h3FourSeconds.height === 768
          && h3FourSeconds.duration === 4 && h3FourSeconds.frames === 96,
        `H3 4-second control changed resolution or duration: ${JSON.stringify(h3FourSeconds)}`,
      );

      await setH3Geometry("1536x672", 15, 120);
      const h3Maximum = await readH3Controls();
      assert(
        h3Maximum.resolution === "1536x672"
          && h3Maximum.width === 1536 && h3Maximum.height === 672
          && h3Maximum.duration === 15 && h3Maximum.frames === 1800
          && h3Maximum.fps === 120,
        `H3 maximum runtime controls did not resolve: ${JSON.stringify(h3Maximum)}`,
      );

      await page.locator(".cv-h3-advanced summary").click();
      const h3UnifiedSurface = await page.evaluate(() => ({
        modeControl: document.querySelector("#cv-h3-mode").type,
        start: getComputedStyle(document.querySelector("#cv-h3-source-row")).display !== "none",
        references: getComputedStyle(document.querySelector("#cv-h3-references-row")).display !== "none",
        lastFrame: getComputedStyle(document.querySelector("#cv-h3-last-frame-row")).display !== "none",
        advancedOpen: document.querySelector(".cv-h3-advanced").open,
      }));
      assert(h3UnifiedSurface.modeControl === "hidden"
        && h3UnifiedSurface.start && h3UnifiedSurface.references
        && h3UnifiedSurface.lastFrame && !h3UnifiedSurface.advancedOpen,
      `H3 Canvas is not one progressive-disclosure surface: ${JSON.stringify(h3UnifiedSurface)}`);

      await setH3Geometry("custom", 180, 24, 320, 192);
      const h3LongT2va = await readH3Controls();
      assert(h3LongT2va.durationMax === 180 && h3LongT2va.duration === 180
        && h3LongT2va.frames === 4320,
      `H3 Canvas T2VA long context did not expand: ${JSON.stringify(h3LongT2va)}`);
      const h3SourceChooserPromise = page.waitForEvent("filechooser");
      await page.locator("#cv-h3-source-button").click();
      const h3SourceChooser = await h3SourceChooserPromise;
      await h3SourceChooser.setFiles({
        name: "alien-baby.png",
        mimeType: "image/png",
        buffer: Buffer.from(
          "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
          "base64",
        ),
      });
      await page.waitForFunction(() => {
        const note = document.querySelector("#cv-h3-source-note");
        const preview = document.querySelector("#cv-source-preview");
        return note && note.textContent.includes("alien-baby.png")
          && preview && preview.style.display === "block"
          && document.querySelector("#cv-h3-mode").value === "i2va";
      });
      const h3StartOnly = await readH3Controls();
      assert(h3StartOnly.mode === "i2va" && h3StartOnly.routeText.includes("Start image"),
        `H3 did not infer start-image generation: ${JSON.stringify(h3StartOnly)}`);

      await page.locator("#cv-h3-last-frame-file").setInputFiles({
        name: "cockpit.png",
        mimeType: "image/png",
        buffer: Buffer.from(
          "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
          "base64",
        ),
      });
      await page.waitForFunction(() => document.querySelector("#cv-h3-mode").value === "fl2va");
      const h3FirstLast = await readH3Controls();
      assert(h3FirstLast.routeText.includes("Start + end images"),
        `H3 did not infer first-plus-last generation: ${JSON.stringify(h3FirstLast)}`);
      await page.locator("#cv-h3-source-clear").click();
      await page.waitForFunction(() => document.querySelector("#cv-h3-mode").value === "l2va");
      await page.locator("#cv-h3-last-frame-clear").click();
      await page.waitForFunction(() => document.querySelector("#cv-h3-mode").value === "t2va");

      await page.locator("#cv-h3-references-row summary").click();
      const h3ReferenceSurface = await page.evaluate(() => ({
        visible: document.querySelector("#cv-h3-references-row").style.display !== "none",
        open: document.querySelector("#cv-h3-references-row details").open,
        multiple: document.querySelector("#cv-h3-references-file").multiple,
        accept: document.querySelector("#cv-h3-references-file").accept,
        buttonTop: document.querySelector('label[for="cv-h3-references-file"]')
          .getBoundingClientRect().top,
        panelBottom: document.querySelector("#panel-canvas .cv-right")
          .getBoundingClientRect().bottom,
      }));
      assert(h3ReferenceSurface.visible && h3ReferenceSurface.open && h3ReferenceSurface.multiple
        && h3ReferenceSurface.accept.includes("image/*")
        && h3ReferenceSurface.accept.includes("video/*")
        && h3ReferenceSurface.accept.includes("audio/*")
        && h3ReferenceSurface.buttonTop < h3ReferenceSurface.panelBottom,
      `H3 omni-reference picker is incomplete: ${JSON.stringify(h3ReferenceSurface)}`);
      await page.locator("#cv-h3-references-file").setInputFiles([
        { name: "identity.png", mimeType: "image/png", buffer: Buffer.from("h3-image") },
        { name: "motion.mp4", mimeType: "video/mp4", buffer: Buffer.from("h3-video") },
        { name: "voice.wav", mimeType: "audio/wav", buffer: Buffer.from("h3-audio") },
      ]);
      await page.waitForFunction(() => {
        const labels = Array.from(document.querySelectorAll("#cv-h3-references-list .cv-lora-name"));
        return labels.length === 3 && labels.every((label) => label.textContent.includes("ready"))
          && document.querySelector("#cv-h3-mode").value === "ref2va";
      });
      const h3LongRef2va = await readH3Controls();
      assert(h3LongRef2va.durationMax === 15 && h3LongRef2va.duration === 15
        && h3LongRef2va.frames === 360,
      `H3 references did not infer Ref2VA and retain its trained duration cap: ${JSON.stringify(h3LongRef2va)}`);
      await page.locator("#cv-h3-quant").selectOption("bf16");
      await page.evaluate((expectedAttention) => {
        const attention = document.querySelector("#cv-h3-attention");
        attention.value = expectedAttention;
        attention.dispatchEvent(new Event("change", { bubbles: true }));
      }, expectedH3Attention);
      const h3AudioRoles = page.locator("#cv-h3-references-list select");
      await h3AudioRoles.nth(0).selectOption("reuse");
      await h3AudioRoles.nth(1).selectOption("voice_timbre");
      await setH3Geometry("1344x768", 4, 24);
      await page.locator("#cv-prompt").fill(
        "Use <Image 1> for identity, <Video 1> for motion, reuse <Audio 1>, and use <Audio 2> for voice timbre.",
      );
      const h3RefVideoBefore = videoRequests.length;
      await page.locator("#cv-generate-btn").click();
      for (let attempt = 0; attempt < 100 && videoRequests.length === h3RefVideoBefore; attempt += 1) {
        await page.waitForTimeout(20);
      }
      assert(videoRequests.length === h3RefVideoBefore + 1,
        "H3 Ref2VA Canvas did not POST /v1/video");
      const h3RefRequest = videoRequests[videoRequests.length - 1];
      assert(h3RefRequest.task === "ref2va"
        && h3RefRequest.quant === "bf16"
        && h3RefRequest.attention_backend === expectedH3Attention
        && Array.isArray(h3RefRequest.references)
        && h3RefRequest.references.length === 3
        && h3RefRequest.references[0].kind === "image"
        && h3RefRequest.references[0].path.endsWith(".png")
        && h3RefRequest.references[1].kind === "video"
        && h3RefRequest.references[1].path.endsWith(".mp4")
        && h3RefRequest.references[1].audio_use === "reuse"
        && h3RefRequest.references[2].kind === "audio"
        && h3RefRequest.references[2].path.endsWith(".wav")
        && h3RefRequest.references[2].audio_use === "voice_timbre",
      `H3 ordered omni-references drifted: ${JSON.stringify(h3RefRequest)}`);
      const h3ContinueContract = await page.evaluate(() => new Promise((resolve, reject) => {
        document.addEventListener("sf-staging-continue-h3", (event) => {
          resolve({
            detail: event.detail && event.detail.result,
            mode: document.querySelector("#cv-h3-mode").value,
          });
        }, { once: true });
        CanvasStaging.activate([{
          src: "/playwright-h3-continue.mp4",
          isVideo: true,
          filename: "completed-h3.mp4",
          metadata: {
            model: "minimax_h3",
            width: 768,
            height: 768,
            fps: 24,
            duration_seconds: 15,
          },
        }], CanvasTab.getToolContext());
        const button = document.querySelector("#stg-continue-h3");
        if (!button) {
          reject(new Error("video staging has no Continue H3 action"));
          return;
        }
        button.click();
      }));
      assert(h3ContinueContract.detail
        && h3ContinueContract.detail.isVideo === true
        && h3ContinueContract.detail.filename === "completed-h3.mp4",
      `H3 continuation action lost its completed clip: ${JSON.stringify(h3ContinueContract)}`);
      try {
        await page.waitForFunction(() => {
          const note = document.querySelector("#cv-h3-source-note");
          return note && note.textContent.includes("I2VA fallback frame ready");
        }, null, { timeout: 5000 });
      } catch (error) {
        const state = await page.evaluate(() => ({
          note: document.querySelector("#cv-h3-source-note")?.textContent || "",
          error: document.querySelector("#cv-error-banner")?.textContent || "",
          preview: document.querySelector("#cv-source-preview")?.src || "",
        }));
        throw new Error(`H3 continuation extraction stalled: ${JSON.stringify(state)}; ${error.message}`);
      }
      const h3ContinuationReady = await page.evaluate(() => ({
        mode: document.querySelector("#cv-h3-mode").value,
        quant: document.querySelector("#cv-h3-quant").value,
        width: Number(document.querySelector("#cv-h3-width").value),
        height: Number(document.querySelector("#cv-h3-height").value),
        duration: Number(document.querySelector("#cv-h3-duration").value),
        fps: Number(document.querySelector("#cv-h3-fps").value),
        source: document.querySelector("#cv-source-preview").src,
      }));
      assert(h3ContinuationReady.mode === "i2va"
        && h3ContinuationReady.quant === "bf16"
        && h3ContinuationReady.width === 768 && h3ContinuationReady.height === 768
        && h3ContinuationReady.duration === 15 && h3ContinuationReady.fps === 24
        && h3ContinuationReady.source.startsWith("data:image/png;base64,"),
      `H3 final-frame continuation was not ready: ${JSON.stringify(h3ContinuationReady)}`);

      const h3NativeContinueContract = await page.evaluate(() => new Promise((resolve, reject) => {
        document.addEventListener("sf-staging-continue-h3", (event) => {
          resolve({ detail: event.detail && event.detail.result });
        }, { once: true });
        CanvasStaging.activate([{
          src: "/out/video-0420/video.mp4",
          isVideo: true,
          filename: "native-h3.mp4",
          metadata: {
            promptId: "video-0420",
            width: 768,
            height: 768,
            fps: 24,
            duration_seconds: 10,
            serverResult: {
              state: "done",
              model: "minimax_h3",
              width: 768,
              height: 768,
              fps: 24,
              frames: 240,
              motion_context_available: true,
              motion_context_windows: [5, 22, 39],
            },
          },
        }], CanvasTab.getToolContext());
        const button = document.querySelector("#stg-continue-h3");
        if (!button) {
          reject(new Error("native video staging has no Continue H3 action"));
          return;
        }
        button.click();
      }));
      assert(h3NativeContinueContract.detail
        && h3NativeContinueContract.detail.metadata.promptId === "video-0420",
      `H3 native continuation event lost its source: ${JSON.stringify(h3NativeContinueContract)}`);
      await page.waitForFunction(() => document.querySelector("#cv-h3-mode").value === "continue");
      await page.locator("#cv-h3-context-frames").selectOption("39");
      const h3NativeContinuationReady = await readH3Controls();
      assert(h3NativeContinuationReady.mode === "continue"
        && h3NativeContinuationReady.continuationVisible
        && h3NativeContinuationReady.continuationSource.includes("video-0420")
        && h3NativeContinuationReady.contextFrames === 39
        && h3NativeContinuationReady.durationMax >= 15
        && h3NativeContinuationReady.width === 768
        && h3NativeContinuationReady.height === 768
        && h3NativeContinuationReady.duration === 10
        && h3NativeContinuationReady.fps === 24
        && h3NativeContinuationReady.stepCache === "exact",
      `H3 native latent continuation was not ready: ${JSON.stringify(h3NativeContinuationReady)}`);
      await page.locator("#cv-prompt").fill("Continue the same action and sound without a cut.");
      const h3NativeVideoBefore = videoRequests.length;
      await page.locator("#cv-generate-btn").click();
      for (let attempt = 0; attempt < 100 && videoRequests.length === h3NativeVideoBefore; attempt += 1) {
        await page.waitForTimeout(20);
      }
      assert(videoRequests.length === h3NativeVideoBefore + 1,
        "H3 native continuation did not POST /v1/video");
      const h3NativeRequest = videoRequests[videoRequests.length - 1];
      assert(h3NativeRequest.task === "continue"
        && h3NativeRequest.continue_from === "video-0420"
        && h3NativeRequest.motion_context_frames === 39
        && Array.isArray(h3NativeRequest.references)
        && h3NativeRequest.references.length === 0
        && h3NativeRequest.width === 768 && h3NativeRequest.height === 768
        && h3NativeRequest.duration_seconds === 10
        && h3NativeRequest.fps === 24
        && h3NativeRequest.step_cache === "exact"
        && h3NativeRequest.include_audio === true,
      `H3 native continuation request drifted: ${JSON.stringify(h3NativeRequest)}`);
      await page.locator('.nav-btn[data-tab="h3-studio"]').click();
      await page.waitForFunction(() => !!document.querySelector(
        '#panel-h3-studio select[data-shot-field="attention_backend"]',
      ));
      await page.waitForFunction((expectedAttention) => {
        const control = document.querySelector(
          '#panel-h3-studio select[data-shot-field="attention_backend"]',
        );
        return control && control.value === expectedAttention;
      }, expectedH3Attention);
      const h3StudioAttention = await page.evaluate(() => {
        const control = document.querySelector(
          '#panel-h3-studio select[data-shot-field="attention_backend"]',
        );
        return {
          attention: control.value,
          disabled: control.disabled,
          options: Array.from(control.options).map((option) => ({
            id: option.value,
            disabled: option.disabled,
            label: option.textContent,
          })),
        };
      });
      assert(h3StudioAttention.attention === expectedH3Attention
        && !h3StudioAttention.disabled
        && (expectedH3Attention === "ck-int8"
          || h3StudioAttention.options.some((option) =>
            option.id === "ck-int8" && option.disabled)),
      `H3 Studio did not honor GPU attention admission: ${JSON.stringify(h3StudioAttention)}`);
      if (process.env.SERENITY_H3_ONLY === "1") {
        assert(pageErrors.length === 0, `page errors: ${pageErrors.join(" | ")}`);
        assert(requestFailures.length === 0,
          `same-origin request failures: ${requestFailures.join(" | ")}`);
        console.log("serenity playwright H3 runtime controls: PASS");
        console.log(JSON.stringify({
          schema: "serenity.playwright.h3_runtime_controls.v1",
          base_url: baseUrl,
          generate_default_controls: generateH3Default,
          generate_boundary_controls: generateH3Boundary,
          generate_long_context_controls: generateH3LongContext,
          generate_submitted_request: generateH3Request,
          default_controls: h3Default,
          two_second_controls: h3TwoSeconds,
          four_second_controls: h3FourSeconds,
          maximum_controls: h3Maximum,
          long_context_t2va_controls: h3LongT2va,
          long_context_ref2va_controls: h3LongRef2va,
          custom_controls: h3Custom,
          ref2va_submitted_request: h3RefRequest,
          continuation_action: h3ContinueContract,
          continuation_ready: h3ContinuationReady,
          native_continuation_ready: h3NativeContinuationReady,
          native_continuation_request: h3NativeRequest,
          studio_attention: h3StudioAttention,
        }, null, 2));
        return;
      }
      await page.locator('.nav-btn[data-tab="generate"]').click();
    }

    const ltxModels = modelResults.filter((result) => result.backend === "ltx2");
    const ltxNames = ltxModels.map((result) => result.name);
    for (const required of [
      "ltx-2.3-22b-dev-fp8",
      "ltx-2.3-22b-dev-fp8-dequant-bf16",
      "sulphur_dev_fp8_serenity",
    ]) assert(ltxNames.includes(required), `Generate omitted admitted LTX checkpoint ${required}`);
    const sulphur = ltxModels.find((result) => result.name === "sulphur_dev_fp8_serenity");
    assert(
      sulphur.runner === "ltx2_mojo_request"
        && sulphur.sampler === "euler_ancestral_cfg_pp"
        && sulphur.scheduler === "sulphur_creator_8_3"
        && sulphur.steps === 8,
      `Sulphur creator route drifted: ${JSON.stringify(sulphur)}`,
    );
    const ltxControls = await page.evaluate(() => {
      const search = document.querySelector("#gen-model-search");
      search.value = "";
      search.dispatchEvent(new Event("input", { bubbles: true }));
      search.click();
      const item = Array.from(document.querySelectorAll(".gen-model-dropdown-item"))
        .find((candidate) => candidate.dataset.model === "ltx-2.3-22b-dev-fp8");
      if (!item) return { error: "LTX dev FP8 model is absent" };
      item.click();
      return {
        durationMax: document.querySelector("#gen-seconds").max,
        cameraOptions: Array.from(document.querySelector("#gen-camera-motion").options)
          .map((option) => ({
            id: option.value,
            label: option.textContent,
            disabled: option.disabled,
          })),
      };
    });
    assert(!ltxControls.error, ltxControls.error || "LTX controls failed");
    assert(ltxControls.durationMax === "120",
      `LTX duration cap regressed to ${ltxControls.durationMax}`);
    const dollyLeft = ltxControls.cameraOptions.find((option) => option.id === "dolly_left");
    assert(dollyLeft && !dollyLeft.disabled && /camera LoRA/.test(dollyLeft.label),
      `LTX dolly-left is not a real available camera adapter: ${JSON.stringify(ltxControls.cameraOptions)}`);
    const focusShift = ltxControls.cameraOptions.find((option) => option.id === "focus_shift");
    assert(focusShift && /prompt guidance/.test(focusShift.label),
      `LTX focus-shift is not labeled prompt-only: ${JSON.stringify(ltxControls.cameraOptions)}`);

    // Wan may not be installed in every UI-test model inventory, so exercise
    // the same builder/translator used by both screens with its product name.
    const wanContract = await page.evaluate(() => {
      const workflow = WorkflowBuilder.build({
        model: "Wan2.2-TI2V-5B-Diffusers",
        prompt: "Playwright Wan product routing test",
        negPrompt: "",
        width: 1280,
        height: 704,
        steps: 50,
        cfg: 5,
        guidance: 5,
        scheduler: "euler",
        seed: 1234,
        frames: 121,
        fps: 24,
        loras: [],
      });
      return { workflow, request: SerenityAPI.videoRequestFromWorkflow(workflow) };
    });
    assert(wanContract.request, "Wan workflow did not route to /v1/video");
    assert(wanContract.request.model === "wan22", `Wan route model changed: ${wanContract.request.model}`);
    assert(
      wanContract.request.width === 1280 && wanContract.request.height === 704
        && wanContract.request.frames === 121 && wanContract.request.fps === 24,
      `Wan product geometry drifted: ${JSON.stringify(wanContract.request)}`,
    );
    assert(
      wanContract.request.steps === 50 && wanContract.request.guidance === 5
        && wanContract.request.quant === "bf16",
      `Wan quality profile drifted: ${JSON.stringify(wanContract.request)}`,
    );
    const wanI2vContract = await page.evaluate(() => {
      const workflow = WorkflowBuilder.build({
        model: "Wan2.2-TI2V-5B-Diffusers",
        prompt: "crush it, a blonde cyborg crushes a metal can",
        negPrompt: "",
        width: 704,
        height: 1248,
        steps: 50,
        cfg: 5,
        seed: 123,
        frames: 121,
        fps: 24,
        initImageName: "/tmp/wan-first-frame.png",
        ltx2CameraMotion: "dolly_in",
        loras: [{
          name: "wan22_5b_i2v_crush_it_lora",
          strength: 1,
        }],
      });
      return { workflow, request: SerenityAPI.videoRequestFromWorkflow(workflow) };
    });
    assert(wanI2vContract.request, "Wan I2V workflow did not route to /v1/video");
    assert(
      wanI2vContract.request.model === "wan22"
        && wanI2vContract.request.image_path === "/tmp/wan-first-frame.png",
      `Wan I2V lost its source image: ${JSON.stringify(wanI2vContract.request)}`,
    );
    assert(
      wanI2vContract.request.width === 704 && wanI2vContract.request.height === 1248
        && wanI2vContract.request.frames === 121 && wanI2vContract.request.fps === 24
        && wanI2vContract.request.steps === 50,
      `Wan I2V creator profile drifted: ${JSON.stringify(wanI2vContract.request)}`,
    );
    assert(
      wanI2vContract.request.camera_motion === "dolly_in"
        && wanI2vContract.request.lora.length === 1
        && wanI2vContract.request.lora[0].name === "wan22_5b_i2v_crush_it_lora"
        && wanI2vContract.request.lora[0].weight === 1,
      `Wan I2V lost camera or LoRA controls: ${JSON.stringify(wanI2vContract.request)}`,
    );

    // Bernini remains hidden until its machine-local product gate passes, but
    // the shared Gen/Workflow builder and /v1/video translation contract can
    // still be exercised deterministically with the gated product identity.
    const berniniContract = await page.evaluate(() => {
      const workflow = WorkflowBuilder.build({
        model: "Bernini-R-Diffusers",
        prompt: "Playwright Bernini-R product routing test",
        negPrompt: "",
        width: 1024,
        height: 1024,
        steps: 5,
        cfg: 1,
        scheduler: "euler",
        seed: 42,
        frames: 9,
        fps: 24,
        loras: [],
      });
      return {
        arch: ModelUtils.detectArchFromFilename("Bernini-R-Diffusers"),
        workflow,
        request: SerenityAPI.videoRequestFromWorkflow(workflow),
      };
    });
    assert(berniniContract.arch === "bernini", `Bernini arch changed: ${berniniContract.arch}`);
    assert(berniniContract.request, "Bernini workflow did not route to /v1/video");
    assert(berniniContract.request.model === "bernini", `Bernini route model changed: ${berniniContract.request.model}`);
    assert(
      berniniContract.request.width === 848 && berniniContract.request.height === 480
        && berniniContract.request.frames === 81 && berniniContract.request.fps === 16,
      `Bernini product geometry drifted: ${JSON.stringify(berniniContract.request)}`,
    );
    assert(
      berniniContract.request.steps === 40 && berniniContract.request.guidance === 4
        && berniniContract.request.quant === "fp8",
      `Bernini quality profile drifted: ${JSON.stringify(berniniContract.request)}`,
    );

    const measuredImageContracts = await page.evaluate(() => {
      const specs = [
        { key: "flux", label: "Flux", arch: "flux" },
        { key: "anima", label: "Anima", arch: "anima" },
        { key: "klein", label: "Klein Base", arch: "klein", base: true },
        { key: "chroma", label: "Chroma", arch: "chroma" },
        { key: "ideogram", label: "Ideogram 4", arch: "ideogram4" },
      ];
      const results = {};
      for (const spec of specs) {
        const model = GenerateTab.state.allModels.find((candidate) => {
          const arch = ModelUtils.detectArchFromFilename(candidate.name);
          return arch === spec.arch && (!spec.base || /base.*9b|9b.*base/i.test(candidate.name));
        });
        if (!model) {
          results[spec.key] = { error: `${spec.label} model is absent from the picker` };
          continue;
        }
        const search = document.querySelector("#gen-model-search");
        search.value = "";
        search.dispatchEvent(new Event("input", { bubbles: true }));
        search.click();
        const item = Array.from(document.querySelectorAll(".gen-model-dropdown-item"))
          .find((candidate) => candidate.dataset.model === model.name);
        if (!item) {
          results[spec.key] = { error: `${spec.label} picker item was not rendered` };
          continue;
        }
        item.click();
        const select = document.querySelector("#gen-aspect-dropdown");
        const profile = GenerateTab.state.capabilities.backends.find((candidate) =>
          candidate.backend === ModelUtils.backendForArch(spec.arch));
        const defaults = Object.assign({}, (profile && profile.defaults) || {}, model.generationDefaults || {});
        results[spec.key] = {
          model: GenerateTab.state.model,
          options: Array.from(select.options).map((option) => option.value),
          expectedOptions: (((profile && profile.limits) || {}).sizes || [])
            .map((size) => `${size.width}×${size.height}`),
          width: GenerateTab.state.width,
          height: GenerateTab.state.height,
          steps: GenerateTab.state.steps,
          cfg: GenerateTab.state.cfg,
          expectedSteps: Number(defaults.steps),
          expectedCfg: Number(defaults.cfg),
        };
      }
      return results;
    });
    for (const [key, actual] of Object.entries(measuredImageContracts)) {
      assert(actual && !actual.error, (actual && actual.error) || `${key} contract test failed`);
      assertExactValues(actual.options, actual.expectedOptions, key);
      assert(
        actual.width === 1024 && actual.height === 1024,
        `${key} defaulted to ${actual.width}x${actual.height}`,
      );
      assert(
        actual.steps === actual.expectedSteps && actual.cfg === actual.expectedCfg,
        `${key} defaults were steps=${actual.steps} cfg=${actual.cfg}; server advertised ${actual.expectedSteps}/${actual.expectedCfg}`,
      );
    }

    const kreaAspect = await page.evaluate(async () => {
      const krea = GenerateTab.state.allModels.find((model) =>
        /krea.*raw/i.test(`${model.name} ${model.arch || ""}`));
      if (!krea) return { error: "Krea-2 Raw model is absent from the picker" };
      const search = document.querySelector("#gen-model-search");
      search.value = "";
      search.dispatchEvent(new Event("input", { bubbles: true }));
      search.click();
      const item = Array.from(document.querySelectorAll(".gen-model-dropdown-item"))
        .find((candidate) => candidate.dataset.model === krea.name);
      if (!item) return { error: "Krea-2 picker item was not rendered" };
      item.click();
      const defaultWidth = GenerateTab.state.width;
      const defaultHeight = GenerateTab.state.height;
      const select = document.querySelector("#gen-aspect-dropdown");
      const options = Array.from(select.options).map((option) => option.textContent.trim());
      const landscape = Array.from(select.options).find((option) => option.value === "1344×768");
      if (!landscape) return { error: "Krea-2 16:9 shape is absent", options };
      select.value = landscape.value;
      select.dispatchEvent(new Event("change", { bubbles: true }));
      const workflow = WorkflowBuilder.build({
        model: GenerateTab.state.model,
        prompt: "Playwright Krea aspect routing test",
        negPrompt: "",
        width: GenerateTab.state.width,
        height: GenerateTab.state.height,
        steps: GenerateTab.state.steps,
        cfg: GenerateTab.state.cfg,
        guidance: GenerateTab.state.guidance,
        sampler: GenerateTab.state.sampler,
        scheduler: GenerateTab.state.scheduler,
        seed: 1234,
        loras: [],
      });
      const response = await fetch("/v1/preflight", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workflow }),
      });
      const preflight = await response.json();
      return {
        options,
        defaultWidth,
        defaultHeight,
        defaultSteps: GenerateTab.state.steps,
        defaultCfg: GenerateTab.state.cfg,
        width: GenerateTab.state.width,
        height: GenerateTab.state.height,
        admitted: response.ok && preflight.admitted === true,
        backend: preflight.backend || (preflight.capability_profile || {}).backend || "",
        rejection: preflight.reason || preflight.error || "",
      };
    });
    assert(!kreaAspect.error, kreaAspect.error || "Krea-2 aspect test failed");
    assert(kreaAspect.options.length === 7, `Krea-2 exposed ${kreaAspect.options.length} shapes`);
    assert(
      kreaAspect.defaultWidth === 1024 && kreaAspect.defaultHeight === 1024,
      `Krea-2 defaulted to ${kreaAspect.defaultWidth}x${kreaAspect.defaultHeight}`,
    );
    assert(
      kreaAspect.defaultSteps === 52 && kreaAspect.defaultCfg === 3.5,
      `Krea-2 defaults were steps=${kreaAspect.defaultSteps} cfg=${kreaAspect.defaultCfg}`,
    );

    const kreaTurboDefaults = await page.evaluate(() => {
      const turbo = GenerateTab.state.allModels.find((model) =>
        /krea.*turbo/i.test(`${model.name} ${model.arch || ""}`));
      if (!turbo) return { error: "Krea-2 Turbo model is absent from the picker" };
      const search = document.querySelector("#gen-model-search");
      search.value = "";
      search.dispatchEvent(new Event("input", { bubbles: true }));
      search.click();
      const item = Array.from(document.querySelectorAll(".gen-model-dropdown-item"))
        .find((candidate) => candidate.dataset.model === turbo.name);
      if (!item) return { error: "Krea-2 Turbo picker item was not rendered" };
      item.click();
      return {
        model: GenerateTab.state.model,
        steps: GenerateTab.state.steps,
        cfg: GenerateTab.state.cfg,
        cfgInput: Number(document.querySelector("#gen-cfg").value),
        cfgRange: Number(document.querySelector("#gen-cfg-range").value),
        cfgMin: Number(document.querySelector("#gen-cfg-range").min),
      };
    });
    assert(!kreaTurboDefaults.error, kreaTurboDefaults.error || "Krea-2 Turbo picker test failed");
    assert(/turbo/i.test(kreaTurboDefaults.model), `Krea-2 Turbo selected ${kreaTurboDefaults.model}`);
    assert(
      kreaTurboDefaults.steps === 8 && kreaTurboDefaults.cfg === 0,
      `Krea-2 Turbo defaults were steps=${kreaTurboDefaults.steps} cfg=${kreaTurboDefaults.cfg}`,
    );
    assert(
      kreaTurboDefaults.cfgInput === 0 && kreaTurboDefaults.cfgRange === 0 && kreaTurboDefaults.cfgMin === 0,
      `Krea-2 Turbo CFG controls were input=${kreaTurboDefaults.cfgInput} range=${kreaTurboDefaults.cfgRange} min=${kreaTurboDefaults.cfgMin}`,
    );
    assert(
      kreaAspect.width === 1344 && kreaAspect.height === 768,
      `Krea-2 16:9 selected ${kreaAspect.width}x${kreaAspect.height}`,
    );
    assert(
      kreaAspect.admitted && kreaAspect.backend === "krea2",
      `Krea-2 16:9 preflight rejected: ${kreaAspect.rejection || kreaAspect.backend}`,
    );

    const qwenAspect = await page.evaluate(async () => {
      const qwen = GenerateTab.state.allModels.find((model) =>
        `${model.name} ${model.arch || ""}`.toLowerCase().includes("qwen"));
      if (!qwen) return { error: "Qwen-Image model is absent from the picker" };
      const search = document.querySelector("#gen-model-search");
      search.value = "";
      search.dispatchEvent(new Event("input", { bubbles: true }));
      search.click();
      const item = Array.from(document.querySelectorAll(".gen-model-dropdown-item"))
        .find((candidate) => candidate.dataset.model === qwen.name);
      if (!item) return { error: "Qwen-Image picker item was not rendered" };
      item.click();
      const defaultWidth = GenerateTab.state.width;
      const defaultHeight = GenerateTab.state.height;
      const select = document.querySelector("#gen-aspect-dropdown");
      const options = Array.from(select.options).map((option) => option.textContent.trim());
      const landscape = Array.from(select.options).find((option) => option.value === "1152×896");
      if (!landscape) return { error: "Qwen-Image 4:3 shape is absent", options };
      select.value = landscape.value;
      select.dispatchEvent(new Event("change", { bubbles: true }));
      const workflow = WorkflowBuilder.build({
        model: GenerateTab.state.model,
        prompt: "Playwright Qwen aspect routing test",
        negPrompt: "",
        width: GenerateTab.state.width,
        height: GenerateTab.state.height,
        steps: GenerateTab.state.steps,
        cfg: GenerateTab.state.cfg,
        guidance: GenerateTab.state.guidance,
        sampler: GenerateTab.state.sampler,
        scheduler: GenerateTab.state.scheduler,
        seed: 1234,
        loras: [],
      });
      const response = await fetch("/v1/preflight", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workflow }),
      });
      const preflight = await response.json();
      return {
        options,
        defaultWidth,
        defaultHeight,
        width: GenerateTab.state.width,
        height: GenerateTab.state.height,
        admitted: response.ok && preflight.admitted === true,
        backend: preflight.backend || (preflight.capability_profile || {}).backend || "",
        rejection: preflight.reason || preflight.error || "",
      };
    });
    assert(!qwenAspect.error, qwenAspect.error || "Qwen-Image aspect test failed");
    assert(qwenAspect.options.length === 7, `Qwen-Image exposed ${qwenAspect.options.length} shapes`);
    assert(
      qwenAspect.defaultWidth === 1024 && qwenAspect.defaultHeight === 1024,
      `Qwen-Image defaulted to ${qwenAspect.defaultWidth}x${qwenAspect.defaultHeight}`,
    );
    assert(
      qwenAspect.width === 1152 && qwenAspect.height === 896,
      `Qwen-Image 4:3 selected ${qwenAspect.width}x${qwenAspect.height}`,
    );
    assert(
      qwenAspect.admitted && qwenAspect.backend === "qwenimage",
      `Qwen-Image 4:3 preflight rejected: ${qwenAspect.rejection || qwenAspect.backend}`,
    );

    const sdxlAspect = await page.evaluate(async () => {
      const model = GenerateTab.state.allModels.find((candidate) =>
        ModelUtils.detectArchFromFilename(candidate.name) === "sdxl");
      if (!model) return { error: "SDXL model is absent from the picker" };
      const search = document.querySelector("#gen-model-search");
      search.value = "";
      search.dispatchEvent(new Event("input", { bubbles: true }));
      search.click();
      const item = Array.from(document.querySelectorAll(".gen-model-dropdown-item"))
        .find((candidate) => candidate.dataset.model === model.name);
      if (!item) return { error: "SDXL picker item was not rendered" };
      item.click();
      const defaultWidth = GenerateTab.state.width;
      const defaultHeight = GenerateTab.state.height;
      const select = document.querySelector("#gen-aspect-dropdown");
      const options = Array.from(select.options).map((option) => option.textContent.trim());
      const landscape = Array.from(select.options).find((option) => option.value === "1344×768");
      if (!landscape) return { error: "SDXL 16:9 shape is absent", options };
      select.value = landscape.value;
      select.dispatchEvent(new Event("change", { bubbles: true }));
      const workflow = WorkflowBuilder.build({
        model: GenerateTab.state.model,
        prompt: "Playwright SDXL aspect routing test",
        negPrompt: "",
        width: GenerateTab.state.width,
        height: GenerateTab.state.height,
        steps: GenerateTab.state.steps,
        cfg: GenerateTab.state.cfg,
        guidance: GenerateTab.state.guidance,
        sampler: GenerateTab.state.sampler,
        scheduler: GenerateTab.state.scheduler,
        seed: 1234,
        loras: [],
      });
      const response = await fetch("/v1/preflight", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workflow }),
      });
      const preflight = await response.json();
      return {
        options,
        defaultWidth,
        defaultHeight,
        width: GenerateTab.state.width,
        height: GenerateTab.state.height,
        admitted: response.ok && preflight.admitted === true,
        backend: preflight.backend || (preflight.capability_profile || {}).backend || "",
        rejection: preflight.reason || preflight.error || "",
      };
    });
    assert(!sdxlAspect.error, sdxlAspect.error || "SDXL aspect test failed");
    assert(sdxlAspect.options.length === 7, `SDXL exposed ${sdxlAspect.options.length} shapes`);
    assert(
      sdxlAspect.defaultWidth === 1024 && sdxlAspect.defaultHeight === 1024,
      `SDXL defaulted to ${sdxlAspect.defaultWidth}x${sdxlAspect.defaultHeight}`,
    );
    assert(
      sdxlAspect.width === 1344 && sdxlAspect.height === 768,
      `SDXL 16:9 selected ${sdxlAspect.width}x${sdxlAspect.height}`,
    );
    assert(
      sdxlAspect.admitted && sdxlAspect.backend === "sdxl",
      `SDXL 16:9 preflight rejected: ${sdxlAspect.rejection || sdxlAspect.backend}`,
    );

    const generateBefore = generateRequests.length;
    await page.evaluate(() => {
      GenerateTab.applyParams({
        model: "zimage_base",
        prompt: "Playwright Generate button routing test",
        negPrompt: "",
        width: 1024,
        height: 1024,
        steps: 16,
        cfg: 5,
        sampler: "euler",
        scheduler: "simple",
        seed: 1234,
        batchCount: 1,
        loras: [],
      });
      GenerateTab.state.generating = false;
      const button = document.querySelector("#gen-btn");
      button.disabled = false;
      button.classList.remove("is-busy");
    });
    await page.locator("#gen-btn").click();
    for (let attempt = 0; attempt < 50 && generateRequests.length === generateBefore; attempt += 1) {
      await page.waitForTimeout(20);
    }
    assert(generateRequests.length === generateBefore + 1, "Generate button did not POST /v1/generate");
    const generateBody = generateRequests[generateRequests.length - 1];
    assert(
      generateBody.model === "zimage_base"
        && generateBody.prompt === "Playwright Generate button routing test"
        && generateBody.width === 1024 && generateBody.height === 1024
        && generateBody.steps === 16 && generateBody.cfg === 5
        && generateBody.sampler === "euler" && generateBody.scheduler === "simple",
      `Generate request drifted: ${JSON.stringify(generateBody)}`,
    );

    await page.locator('.nav-btn[data-tab="workflows"]').click();
    await page.locator("#btn-templates").click();
    await page.waitForFunction(
      () => document.querySelectorAll("#templates-list .wf-template-item").length > 0,
    );
    const templateNames = await page.locator("#templates-list .wf-template-item").allTextContents();
    assert(new Set(templateNames).size === templateNames.length,
      `workflow template names are not unique: ${JSON.stringify(templateNames)}`);
    // Close the inventory opened for names; loadTemplate opens it for each click.
    await page.locator("#btn-templates").click();

    const templateResults = [];
    for (let index = 0; index < templateNames.length; index += 1) {
      await loadTemplate(page, index);
      const result = await page.evaluate(async (name) => {
        const workflow = serializeWorkflow(sfCanvas).prompt;
        const classes = Object.values(workflow).map((node) => node.class_type);
        const video = SerenityAPI.videoRequestFromWorkflow(workflow);
        const sync = {
          workflow: WorkflowSync.extract(workflow),
          generate: GenerateTab.getParams(),
        };
        if (video) {
          const isLtx = video.model === "ltx2";
          const isWan = video.model === "wan22";
          const isBernini = video.model === "bernini";
          const ltxRunner = isLtx
            ? (GenerateTab.state.videoStatus.candidate_runners || [])
              .find((entry) => entry && entry.model === "ltx2_t2v_av")
            : null;
          const ltxProfiles = ltxRunner && ltxRunner.modes
            && ltxRunner.modes.ltx2_mojo_request
            && Array.isArray(ltxRunner.modes.ltx2_mojo_request.supported_profiles)
            ? ltxRunner.modes.ltx2_mojo_request.supported_profiles : [];
          const exactAvailableLtxProfile = !isLtx || ltxProfiles.some((profile) =>
            profile.available === true
              && Array.isArray(profile.modes) && profile.modes.includes("standard")
              && Number(profile.width) === Number(video.width)
              && Number(profile.height) === Number(video.height)
              && Number(profile.frames) === Number(video.frames)
              && Number(profile.fps) === Number(video.fps));
          const admittedLtxDev = video.guidance_mode === "dev"
            && video.steps === 15 && video.sampler === "res2s" && video.scheduler === "ltx2";
          const admittedLtxDistilled = video.guidance_mode === "distilled"
            && video.steps === 8
            && ((video.sampler === "euler" && video.scheduler === "ltx2_distilled")
              || (video.sampler === "euler_ancestral_cfg_pp"
                && video.scheduler === "sulphur_creator_8_3"));
          return {
            name,
            nodes: Object.keys(workflow).length,
            sink: classes.includes("SaveVideo") ? "SaveVideo"
              : (classes.includes("SaveAnimatedWEBP") ? "SaveAnimatedWEBP" : ""),
            route: "/v1/video",
            admitted: (isLtx && video.runner === "ltx2_mojo_request"
                && ["fp8", "bf16", "int4"].includes(video.quant)
                && exactAvailableLtxProfile
                && (admittedLtxDev || admittedLtxDistilled))
              || (isWan && video.steps === 50 && video.guidance === 5
                && ["bf16", "fp8"].includes(video.quant))
              || (isBernini && video.width === 848 && video.height === 480
                && video.frames === 81 && video.fps === 16 && video.steps === 40
                && video.guidance === 4 && video.quant === "fp8"),
            backend: video.model,
            exactAvailableLtxProfile,
            video,
            sync,
          };
        }
        const response = await fetch("/v1/preflight", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ workflow }),
        });
        const preflight = await response.json();
        return {
          name,
          nodes: Object.keys(workflow).length,
          sink: classes.includes("SaveImage") ? "SaveImage" : "",
          route: "/prompt",
          admitted: response.ok && preflight.admitted === true,
          backend: preflight.backend || (preflight.capability_profile || {}).backend || "",
          rejection: preflight.reason || preflight.error || "",
          steps: preflight.request && preflight.request.steps,
          cfg: preflight.request && preflight.request.cfg,
          sync,
        };
      }, templateNames[index]);
      templateResults.push(result);
    }
    for (const result of templateResults) {
      assert(result.nodes > 0, `${result.name}: template serialized an empty workflow`);
      assert(result.sink, `${result.name}: template has no media sink`);
      if (result.backend === "ltx2" && !result.exactAvailableLtxProfile) {
        assert(!result.admitted,
          `${result.name}: LTX template bypassed the unavailable runtime ${JSON.stringify(result.video || result)}`);
      } else {
        assert(result.admitted,
          `${result.name}: template route was not admitted ${JSON.stringify(result.video || result)}`);
      }
      for (const key of ["model", "width", "height", "steps", "cfg", "scheduler"]) {
        assert(
          result.sync.workflow[key] === result.sync.generate[key],
          `${result.name}: ${key} split across Workflow/Generate: ` +
            `${result.sync.workflow[key]} vs ${result.sync.generate[key]}`,
        );
      }
    }
    const kreaTemplate = templateResults.find((result) => result.backend === "krea2");
    assert(kreaTemplate, "Krea-2 production template was absent");
    assert(
      kreaTemplate.steps === 8 && kreaTemplate.cfg === 0,
      `Krea-2 template used steps=${kreaTemplate.steps} cfg=${kreaTemplate.cfg}`,
    );
    const ideogramTemplate = templateResults.find((result) => /Ideogram 4/i.test(result.name));
    assert(ideogramTemplate, "fallback Ideogram 4 production template was absent");
    assert(
      ideogramTemplate.backend === "ideogram4" && ideogramTemplate.steps === 20 && ideogramTemplate.cfg === 7,
      `Ideogram 4 template used backend=${ideogramTemplate.backend} steps=${ideogramTemplate.steps} cfg=${ideogramTemplate.cfg}`,
    );
    const senseNovaTemplate = templateResults.find((result) => /SenseNova/i.test(result.name));
    if (senseNovaTemplate) assert(
      senseNovaTemplate.backend === "sensenova" && senseNovaTemplate.steps === 30 && senseNovaTemplate.cfg === 4,
      `SenseNova template used backend=${senseNovaTemplate.backend} steps=${senseNovaTemplate.steps} cfg=${senseNovaTemplate.cfg}`,
    );
    const wanTemplate = templateResults.find((result) => /Wan 2\.2/i.test(result.name));
    if (wanTemplate) assert(
      wanTemplate.backend === "wan22"
        && wanTemplate.video.width === 1280 && wanTemplate.video.height === 704
        && wanTemplate.video.frames === 121 && wanTemplate.video.fps === 24
        && wanTemplate.video.steps === 50 && wanTemplate.video.guidance === 5
        && wanTemplate.sync.generate.cfg === 5
        && wanTemplate.sync.generate.guidance === 5
        && wanTemplate.video.quant === "bf16",
      `Wan template profile drifted: ${JSON.stringify(wanTemplate)}`,
    );
    const berniniTemplate = templateResults.find((result) => /Bernini-R/i.test(result.name));
    if (berniniTemplate) {
      assert(
        berniniTemplate.backend === "bernini"
          && berniniTemplate.video.width === 848 && berniniTemplate.video.height === 480
          && berniniTemplate.video.frames === 81 && berniniTemplate.video.fps === 16
          && berniniTemplate.video.steps === 40 && berniniTemplate.video.guidance === 4
          && berniniTemplate.video.quant === "fp8",
        `Bernini template profile drifted: ${JSON.stringify(berniniTemplate.video)}`,
      );
    }

    await loadTemplate(page, 0);
    const workflowPromptBefore = promptRequests.length;
    await page.locator("#btn-queue").click();
    for (let attempt = 0; attempt < 50 && promptRequests.length === workflowPromptBefore; attempt += 1) {
      await page.waitForTimeout(20);
    }
    assert(promptRequests.length === workflowPromptBefore + 1, "image template Queue did not POST /prompt");

    const ltxTemplateIndex = templateResults.findIndex((result) => result.backend === "ltx2");
    assert(ltxTemplateIndex >= 0, "LTX template was absent");
    const expectedLtxRequest = templateResults[ltxTemplateIndex].video;
    await loadTemplate(page, ltxTemplateIndex);
    const videoBefore = videoRequests.length;
    await page.locator("#btn-queue").click();
    for (let attempt = 0; attempt < 50 && videoRequests.length === videoBefore; attempt += 1) {
      await page.waitForTimeout(20);
    }
    assert(videoRequests.length === videoBefore + 1, "LTX template Queue did not POST /v1/video");
    const ltxRequest = videoRequests[videoRequests.length - 1];
    for (const key of ["model", "runner", "checkpoint", "quant", "steps", "width", "height", "frames", "fps", "sampler", "scheduler", "guidance_mode"]) {
      assert(ltxRequest[key] === expectedLtxRequest[key],
        `LTX queue ${key} changed: ${ltxRequest[key]} vs ${expectedLtxRequest[key]}`);
    }
    assert(ltxRequest.runner === "ltx2_mojo_request", `LTX queue runner changed: ${ltxRequest.runner}`);
    assert(Number.isInteger(ltxRequest.seed) && ltxRequest.seed >= 0, `LTX queue seed changed: ${ltxRequest.seed}`);

    assert(pageErrors.length === 0, `page errors: ${pageErrors.join(" | ")}`);
    assert(requestFailures.length === 0, `same-origin request failures: ${requestFailures.join(" | ")}`);
    const relevantConsoleErrors = consoleErrors.filter((line) => !/favicon|lucide/i.test(line));
    assert(relevantConsoleErrors.length === 0, `console errors: ${relevantConsoleErrors.join(" | ")}`);

    console.log("serenity playwright ui: PASS");
    console.log(JSON.stringify({
      schema: "serenity.playwright.current_ui.v1",
      base_url: baseUrl,
      health,
      model_count: modelResults.length,
      models: modelResults,
      template_count: templateResults.length,
      templates: templateResults,
      generate_route: "/v1/generate",
      workflow_image_route: "/prompt",
      workflow_video_route: "/v1/video",
      restored_gallery_identity: restoredGalleryIdentity,
      restored_gallery_badges: restoredGalleryBadges,
      immutable_video_identity: immutableVideoIdentity,
      gallery_scroll_layout: galleryScrollLayout,
      page_errors: pageErrors,
      console_errors: relevantConsoleErrors,
      request_failures: requestFailures,
    }, null, 2));
  } finally {
    await browser.close().catch(() => {});
    if (server) {
      server.kill("SIGTERM");
      await new Promise((resolve) => {
        const timer = setTimeout(() => {
          server.kill("SIGKILL");
          resolve();
        }, 3000);
        server.once("exit", () => {
          clearTimeout(timer);
          resolve();
        });
      });
    }
    if (outDir) fs.rmSync(outDir, { recursive: true, force: true });
  }
}

run().catch((error) => {
  console.error("serenity playwright ui: FAIL");
  console.error(error && error.stack ? error.stack : String(error));
  process.exitCode = 1;
});
