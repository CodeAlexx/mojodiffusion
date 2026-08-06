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
    assert(/krea.*turbo/i.test(initialModel), `Generate defaulted to ${initialModel}, expected Krea-2 Turbo`);

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
          const h3Profiles = h3Runner && Array.isArray(h3Runner.supported_profiles)
            ? h3Runner.supported_profiles : [];
          const exactAvailableH3Profile = !isH3 || h3Profiles.some((profile) =>
            profile.available === true
              && profile.available_modes && profile.available_modes[video.quant] === true
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
      if (result.backend === "ltx2" && !result.exactAvailableLtxProfile) {
        assert(!result.admitted,
          `${result.name}: LTX route bypassed the unavailable runtime ${JSON.stringify(result)}`);
      } else {
        assert(result.admitted, `${result.name}: route was not admitted ${JSON.stringify(result)}`);
      }
    }
    const h3Model = modelResults.find((result) => result.backend === "minimax_h3");
    if (h3Model) {
      await page.locator('.nav-btn[data-tab="canvas"]').click();
      await page.waitForFunction(() => document.querySelectorAll("#cv-model option").length > 1);
      await page.locator("#cv-model").selectOption(h3Model.name);
      await page.waitForFunction(() => {
        const duration = document.querySelector("#cv-h3-duration");
        return duration && duration.value !== "";
      });
      const readH3Controls = () => page.evaluate(() => ({
        quant: document.querySelector("#cv-h3-quant").value,
        resolution: document.querySelector("#cv-h3-resolution").value,
        duration: Number(document.querySelector("#cv-h3-duration").value),
        frames: Number(document.querySelector("#cv-frames").value),
        fps: Number(document.querySelector("#cv-fps").value),
        resolutions: Array.from(document.querySelector("#cv-h3-resolution").options)
          .map((option) => ({
            value: option.value,
            label: option.textContent,
            disabled: option.disabled,
          })),
      }));
      const h3Default = await readH3Controls();
      assert(
        h3Default.resolution === "512x320_24fps"
          && h3Default.duration === 15.08 && h3Default.frames === 362,
        `H3 did not default to the admitted long profile: ${JSON.stringify(h3Default)}`,
      );
      await page.locator("#cv-h3-quant").selectOption("bf16");
      await page.locator("#cv-h3-resolution").selectOption("832x480_24fps");
      const h3LongBf16 = await readH3Controls();
      assert(
        h3LongBf16.duration === 15.08 && h3LongBf16.frames === 362
          && h3LongBf16.resolution === "832x480_24fps",
        `H3 BF16 high-resolution selection lost 15 seconds: ${JSON.stringify(h3LongBf16)}`,
      );
      const h3HighShortOption = h3LongBf16.resolutions.find(
        (option) => option.value === "960x544_24fps",
      );
      assert(
        h3HighShortOption && h3HighShortOption.disabled
          && h3HighShortOption.label.includes("2.33s"),
        `H3 did not mark the 960 short-only profile: ${JSON.stringify(h3HighShortOption)}`,
      );
      await page.evaluate(() => {
        const resolution = document.querySelector("#cv-h3-resolution");
        resolution.value = "960x544_24fps";
        resolution.dispatchEvent(new Event("change", { bubbles: true }));
      });
      const h3RejectedShortFallback = await readH3Controls();
      assert(
        h3RejectedShortFallback.duration === 15.08
          && h3RejectedShortFallback.frames === 362
          && h3RejectedShortFallback.resolution === "832x480_24fps",
        `H3 silently fell back to 2.33 seconds: ${JSON.stringify(h3RejectedShortFallback)}`,
      );
      await page.evaluate(() => {
        const duration = document.querySelector("#cv-h3-duration");
        duration.value = "2.33";
        duration.dispatchEvent(new Event("change", { bubbles: true }));
      });
      const h3ExplicitShort = await readH3Controls();
      assert(
        h3ExplicitShort.duration === 2.33 && h3ExplicitShort.frames === 56
          && h3ExplicitShort.resolution === "960x544_24fps",
        `H3 could not explicitly select the 2.33-second profile: ${JSON.stringify(h3ExplicitShort)}`,
      );
      await page.evaluate(() => {
        const duration = document.querySelector("#cv-h3-duration");
        duration.value = "15.08";
        duration.dispatchEvent(new Event("change", { bubbles: true }));
      });
      const h3RestoredLong = await readH3Controls();
      assert(
        h3RestoredLong.duration === 15.08 && h3RestoredLong.frames === 362
          && h3RestoredLong.resolution === "832x480_24fps",
        `H3 could not restore the 15-second BF16 profile: ${JSON.stringify(h3RestoredLong)}`,
      );
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
