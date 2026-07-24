#!/usr/bin/env node
"use strict";

const crypto = require("crypto");
const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawnSync } = require("child_process");

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
  return require(path.join(
    os.tmpdir(),
    "mojodiffusion-playwright-tools",
    "node_modules",
    "playwright",
  )).chromium;
}

function assert(value, message) {
  if (!value) throw new Error(message);
}

function canvasHash(dataUrl) {
  return crypto.createHash("sha256").update(dataUrl).digest("hex");
}

function runChecked(command, args) {
  const result = spawnSync(command, args, { encoding: "utf8" });
  if (result.status !== 0) {
    throw new Error(`${command} failed: ${result.stderr || result.stdout}`);
  }
  return result.stdout;
}

(async () => {
  const baseUrl = process.env.SERENITY_URL || "http://127.0.0.1:7811";
  const sourceVideo = path.resolve(
    process.env.GENESIS_SOURCE_VIDEO
      || "output/serenity_ui_out/video-0409/ltx2_t2v_hq.mp4",
  );
  const proofDir = path.resolve(
    process.env.GENESIS_PROOF_DIR || "output/checks/genesis_browser",
  );
  assert(fs.existsSync(sourceVideo), `source video missing: ${sourceVideo}`);
  fs.mkdirSync(proofDir, { recursive: true });

  const chromium = loadChromium();
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  const requestFailures = [];
  const pageErrors = [];
  const consoleErrors = [];
  page.on("requestfailed", (request) => {
    requestFailures.push(
      `${request.method()} ${request.url()} ${request.failure()?.errorText || ""}`,
    );
  });
  page.on("response", (response) => {
    if (response.status() >= 400) {
      requestFailures.push(
        `${response.status()} ${response.request().method()} ${response.url()}`,
      );
    }
  });
  page.on("pageerror", (error) => pageErrors.push(String(error)));
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });

  let result;
  try {
    const created = await page.request.post(`${baseUrl}/video_edit/projects`, {
      data: {
        name: "Genesis Playwright proof",
        fps: 25,
        width: 512,
        height: 768,
        tracks: [],
      },
    });
    assert(created.ok(), `project create failed: ${created.status()}`);
    const project = await created.json();
    const projectId = project.id;
    assert(projectId, "project create returned no id");

    const imported = await page.request.post(
      `${baseUrl}/video_edit/projects/${projectId}/import_clip`,
      {
        multipart: {
          file: {
            name: path.basename(sourceVideo),
            mimeType: "video/mp4",
            buffer: fs.readFileSync(sourceVideo),
          },
        },
      },
    );
    assert(imported.ok(), `clip import failed: ${imported.status()}`);
    const media = await imported.json();
    assert(media.duration_frames === 121, `imported frames=${media.duration_frames}`);
    assert(Math.abs(media.source_fps - 25) < 0.001, `imported fps=${media.source_fps}`);

    const proofFrames = 12;
    const projectBody = {
      name: "Genesis Playwright proof",
      fps: media.source_fps,
      width: 512,
      height: 768,
      tracks: [{
        id: "track-video",
        name: "Video 1",
        type: "video",
        clips: [{
          id: "clip-proof",
          startFrame: 0,
          endFrame: proofFrames,
          label: "LTX2 Genesis proof",
          color: "#4a7dff",
          source_path: media.source_path,
          source_fps: media.source_fps,
          effects: [],
        }],
      }],
    };
    const updated = await page.request.put(
      `${baseUrl}/video_edit/projects/${projectId}`,
      { data: projectBody },
    );
    assert(updated.ok(), `project update failed: ${updated.status()}`);

    await page.addInitScript((id) => {
      localStorage.setItem("ve-project-id", id);
    }, projectId);
    await page.goto(baseUrl, { waitUntil: "domcontentloaded" });
    const isProjectPreview = (response) => {
      if (!response.url().endsWith("/video_edit/preview") || response.status() !== 200) {
        return false;
      }
      try {
        const body = response.request().postDataJSON();
        return body.project_id === projectId && body.project?.tracks?.some(
          (track) => track.clips?.some((clip) => !!clip.source_path),
        );
      } catch (_) {
        return false;
      }
    };
    const initialPreview = page.waitForResponse(
      isProjectPreview,
      { timeout: 30000 },
    );
    await page.click('.nav-btn[data-tab="video-edit"]');
    await initialPreview;
    await page.waitForFunction(() => (
      document.querySelector("#ve-engine-status")?.textContent.includes("Ready")
    ));

    const readCanvasHash = async () => canvasHash(
      await page.locator("#ve-preview-canvas").evaluate(
        (canvas) => canvas.toDataURL("image/png"),
      ),
    );
    const waitForPreview = async () => {
      const response = await page.waitForResponse(isProjectPreview, { timeout: 30000 });
      await page.waitForFunction(() => (
        document.querySelector("#ve-engine-status")?.textContent.includes("Ready")
      ));
      return response;
    };
    const baseHash = await readCanvasHash();

    await page.locator("#ve-timeline-canvas").click({
      button: "right",
      position: { x: 140, y: 50 },
    });
    await page.getByText("Properties...", { exact: true }).click();
    await page.locator(".ve-add-effect-btn").click();
    const addPreview = waitForPreview();
    await page.getByText("Saturation", { exact: true }).click();
    await addPreview;

    const effectSlider = page.locator(
      '.ve-effect-card input[type="range"]',
    ).first();
    const changedPreview = waitForPreview();
    await effectSlider.evaluate((element) => {
      element.value = "0";
      element.dispatchEvent(new Event("input", { bubbles: true }));
      element.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await changedPreview;
    const effectHash = await readCanvasHash();
    assert(effectHash !== baseHash, "saturation edit did not change the Genesis preview");

    const enabledCheckbox = page.locator(
      '.ve-effect-card input[type="checkbox"]',
    ).first();
    const disabledPreview = waitForPreview();
    await enabledCheckbox.uncheck();
    await disabledPreview;
    const disabledHash = await readCanvasHash();
    assert(disabledHash === baseHash, "disabled effect still changed the Genesis preview");

    const reenabledPreview = waitForPreview();
    await enabledCheckbox.check();
    await reenabledPreview;
    const reenabledHash = await readCanvasHash();
    assert(reenabledHash === effectHash, "re-enabled effect did not restore edited pixels");
    await page.waitForTimeout(2500);

    const screenshotPath = path.join(proofDir, "editor.png");
    await page.screenshot({ path: screenshotPath, fullPage: true });
    await page.locator(".ve-props-close").click();
    await page.click("#ve-btn-export");
    const resolution = await page.locator("#ve-exp-res option").first().textContent();
    const fps = await page.locator("#ve-exp-fps").inputValue();
    assert(resolution === "512x768", `export resolution=${resolution}`);
    assert(fps === "25", `export fps=${fps}`);
    assert(
      await page.locator("#ve-exp-fps").getAttribute("readonly") !== null,
      "export fps is not bound to the project timeline",
    );
    await page.uncheck("#ve-exp-audio");
    await page.click("#ve-exp-start");
    await page.waitForFunction(() => {
      const text = document.querySelector("#ve-exp-progress-text")?.textContent || "";
      return text.startsWith("Export complete:") || text.startsWith("Error:");
    }, { timeout: 120000 });
    const exportText = await page.locator("#ve-exp-progress-text").textContent();
    assert(exportText.startsWith("Export complete: "), exportText);
    const outputPath = exportText.slice("Export complete: ".length);
    assert(fs.existsSync(outputPath), `export missing: ${outputPath}`);

    const probe = JSON.parse(runChecked("ffprobe", [
      "-v", "error",
      "-show_entries", "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames",
      "-show_entries", "format=duration,size",
      "-of", "json",
      outputPath,
    ]));
    const stream = probe.streams?.[0] || {};
    assert(stream.codec_name === "h264", `codec=${stream.codec_name}`);
    assert(stream.width === 512 && stream.height === 768, `size=${stream.width}x${stream.height}`);
    assert(stream.r_frame_rate === "25/1", `r_frame_rate=${stream.r_frame_rate}`);
    assert(stream.avg_frame_rate === "25/1", `avg_frame_rate=${stream.avg_frame_rate}`);
    assert(Number(stream.nb_frames) === proofFrames, `frames=${stream.nb_frames}`);

    const contactSheet = path.join(proofDir, "export-contact.png");
    runChecked("ffmpeg", [
      "-v", "error", "-i", outputPath,
      "-vf", "select=eq(n\\,0)+eq(n\\,6)+eq(n\\,11),scale=256:-1,tile=3x1",
      "-frames:v", "1", "-y", contactSheet,
    ]);

    const saved = await page.request.get(
      `${baseUrl}/video_edit/projects/${projectId}`,
    );
    const savedProject = await saved.json();
    const savedEffect = savedProject.tracks?.[0]?.clips?.[0]?.effects?.[0];
    assert(savedEffect?.type === "saturation", "browser effect was not persisted");
    assert(savedEffect?.enabled === true, "persisted browser effect is disabled");
    assert(savedEffect?.params?.value === 0, "persisted saturation value is not zero");
    assert(requestFailures.length === 0, `request failures: ${requestFailures.join(" | ")}`);
    assert(pageErrors.length === 0, `page errors: ${pageErrors.join(" | ")}`);
    assert(consoleErrors.length === 0, `console errors: ${consoleErrors.join(" | ")}`);

    result = {
      schema: "serenity.genesis_browser_gate.v1",
      status: "PASS",
      project_id: projectId,
      source_video: sourceVideo,
      source_frames: media.duration_frames,
      source_fps: media.source_fps,
      effect: "saturation=0",
      effect_pixels_changed: effectHash !== baseHash,
      disabled_effect_restored_base: disabledHash === baseHash,
      resolution,
      fps: Number(fps),
      output_path: outputPath,
      output_frames: Number(stream.nb_frames),
      output_duration: Number(probe.format?.duration),
      screenshot: screenshotPath,
      contact_sheet: contactSheet,
      request_failures: requestFailures,
      page_errors: pageErrors,
      console_errors: consoleErrors,
    };
    fs.writeFileSync(
      path.join(proofDir, "result.json"),
      `${JSON.stringify(result, null, 2)}\n`,
    );
  } finally {
    await browser.close();
  }
  console.log(JSON.stringify(result, null, 2));
})().catch((error) => {
  console.error(error.stack || error);
  process.exit(1);
});
