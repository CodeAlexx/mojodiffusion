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
  const sourceAudio = process.env.GENESIS_SOURCE_AUDIO
    ? path.resolve(process.env.GENESIS_SOURCE_AUDIO)
    : path.join(proofDir, "music-fixture.wav");
  if (!process.env.GENESIS_SOURCE_AUDIO && !fs.existsSync(sourceAudio)) {
    runChecked("ffmpeg", [
      "-v", "error",
      "-f", "lavfi",
      "-i", "sine=frequency=220:sample_rate=48000:duration=5.8",
      "-filter:a", "volume=0.25",
      "-ac", "2",
      "-c:a", "pcm_s16le",
      "-y", sourceAudio,
    ]);
  }
  assert(fs.existsSync(sourceAudio), `source audio missing: ${sourceAudio}`);

  const chromium = loadChromium();
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  const requestFailures = [];
  const cancelledMediaRequests = [];
  const pageErrors = [];
  const consoleErrors = [];
  page.on("requestfailed", (request) => {
    const failure = `${request.method()} ${request.url()} ${request.failure()?.errorText || ""}`;
    if (
      request.method() === "GET"
      && request.url().includes("/video_edit/media/")
      && request.failure()?.errorText === "net::ERR_ABORTED"
    ) {
      cancelledMediaRequests.push(failure);
    } else {
      requestFailures.push(failure);
    }
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
    const inspectProjectId = process.env.GENESIS_INSPECT_PROJECT_ID || "";
    if (inspectProjectId) {
      await page.addInitScript((id) => {
        localStorage.setItem("ve-project-id", id);
      }, inspectProjectId);
      await page.goto(baseUrl, { waitUntil: "domcontentloaded" });
      const previewReady = page.waitForResponse((response) => {
        if (!response.url().endsWith("/video_edit/preview") || response.status() !== 200) {
          return false;
        }
        try {
          return response.request().postDataJSON().project_id === inspectProjectId;
        } catch (_) {
          return false;
        }
      }, { timeout: 30000 });
      await page.click('.nav-btn[data-tab="video-edit"]');
      await previewReady;
      await page.waitForFunction(() => {
        const diagnostics = window.VideoEditTab?.getDiagnostics?.();
        return diagnostics?.thumbnailClipIds?.length > 0
          && diagnostics?.tracks?.some((track) => (
            track.type === "video"
            && track.clips.some((clip) => clip.source_path && clip.startFrame === 0)
          ));
      }, { timeout: 30000 });
      const diagnostics = await page.evaluate(
        () => window.VideoEditTab.getDiagnostics(),
      );
      const visiblePixels = await page.locator("#ve-preview-canvas").evaluate((canvas) => {
        const pixels = canvas.getContext("2d").getImageData(
          0, 0, canvas.width, canvas.height,
        ).data;
        let count = 0;
        for (let index = 0; index < pixels.length; index += 4) {
          if (pixels[index] + pixels[index + 1] + pixels[index + 2] > 90) count++;
        }
        return count;
      });
      assert(visiblePixels > 1000, `saved project preview pixels=${visiblePixels}`);
      assert(diagnostics.mediaCardCount >= 1,
        `saved project media cards=${diagnostics.mediaCardCount}`);
      assert(diagnostics.dockTabCount === 4,
        `saved project dock tabs=${diagnostics.dockTabCount}`);
      assert(await page.locator("#ve-edit-toolbar .ve-edit-btn").count() >= 12,
        "saved project timeline toolbar is incomplete");
      const previewHashBeforePlayback = canvasHash(
        await page.locator("#ve-preview-canvas").evaluate(
          (canvas) => canvas.toDataURL("image/png"),
        ),
      );
      await page.click("#ve-btn-play");
      await page.waitForFunction(() => {
        const state = window.VideoEditTab?.getDiagnostics?.();
        return state?.currentFrame >= 24 && state?.previewVideoTime >= 0.5;
      }, { timeout: 5000 });
      const playbackDiagnostics = await page.evaluate(
        () => window.VideoEditTab.getDiagnostics(),
      );
      const previewHashDuringPlayback = canvasHash(
        await page.locator("#ve-preview-canvas").evaluate(
          (canvas) => canvas.toDataURL("image/png"),
        ),
      );
      assert(
        previewHashDuringPlayback !== previewHashBeforePlayback,
        "saved project large preview did not change during playback",
      );
      assert(
        playbackDiagnostics.previewVideoPaused === false,
        "saved project preview video was paused while timeline was playing",
      );
      await page.click("#ve-btn-play");
      await page.locator(".ve-media-card").first().click();
      const restoredProgramPreview = page.waitForResponse((response) => {
        if (!response.url().endsWith("/video_edit/preview") || response.status() !== 200) {
          return false;
        }
        try {
          return response.request().postDataJSON().project_id === inspectProjectId;
        } catch (_) {
          return false;
        }
      }, { timeout: 30000 });
      await page.locator('.ve-monitor-tab[data-monitor="program"]').click();
      await restoredProgramPreview;
      await page.locator('.ve-dock-tab[data-dock="properties"]').click();
      assert(await page.locator(".ve-props-panel").isVisible(),
        "saved project clip parameters are not visible in Properties");
      const nativePropertyControls = await page.locator(
        ".ve-props-panel .ve-genesis-control",
      ).count();
      assert(nativePropertyControls >= 100,
        `native Genesis property controls=${nativePropertyControls}`);
      await page.locator('.ve-dock-tab[data-dock="filters"]').click();
      assert(await page.locator(".ve-genesis-filter-catalog").isVisible(),
        "saved project native filter catalog is missing");
      const nativeFilterCount = await page.locator(
        ".ve-filter-options button",
      ).count();
      assert(nativeFilterCount >= 20,
        `native Genesis filter catalog entries=${nativeFilterCount}`);
      await page.locator('.ve-dock-tab[data-dock="scopes"]').click();
      assert(await page.locator(".ve-scope-grid canvas").count() === 4,
        "saved project does not expose all four scopes");
      await page.locator('.ve-dock-tab[data-dock="audio"]').click();
      const mixerTrackCount = await page.locator(".ve-audio-mixer-strip").count();
      assert(mixerTrackCount >= 1,
        `saved project mixer tracks=${mixerTrackCount}`);
      assert(await page.locator("#ve-audio-waveform").isVisible(),
        "saved project audio waveform scope is missing");
      await page.locator('.ve-dock-tab[data-dock="properties"]').click();
      assert(
        !diagnostics.tracks.some((track) => track.clips.some(
          (clip) => ["Intro", "Scene 1", "Overlay", "Music.mp3"].includes(clip.label)
            && !clip.source_path,
        )),
        "legacy demo clips remain in the saved project",
      );
      const screenshotPath = path.join(proofDir, "saved-project.png");
      await page.screenshot({ path: screenshotPath, fullPage: true });
      assert(requestFailures.length === 0, `request failures: ${requestFailures.join(" | ")}`);
      assert(pageErrors.length === 0, `page errors: ${pageErrors.join(" | ")}`);
      assert(consoleErrors.length === 0, `console errors: ${consoleErrors.join(" | ")}`);
      result = {
        schema: "serenity.genesis_saved_project_gate.v1",
        status: "PASS",
        project_id: inspectProjectId,
        visible_preview_pixels: visiblePixels,
        playback_frame_reached: playbackDiagnostics.currentFrame,
        preview_video_time: playbackDiagnostics.previewVideoTime,
        preview_pixels_changed_during_playback:
          previewHashDuringPlayback !== previewHashBeforePlayback,
        genesis_layout: {
          media_cards: diagnostics.mediaCardCount,
          program_source_monitors: true,
          timeline_toolbar_buttons:
            await page.locator("#ve-edit-toolbar .ve-edit-btn").count(),
          persistent_dock_tabs: diagnostics.dockTabCount,
          properties_parameter_controls: nativePropertyControls,
          native_filter_catalog_entries: nativeFilterCount,
          live_scopes: 4,
          audio_mixer_tracks: mixerTrackCount,
        },
        thumbnail_clip_count: diagnostics.thumbnailClipIds.length,
        tracks: diagnostics.tracks,
        screenshot: screenshotPath,
        request_failures: requestFailures,
        cancelled_media_requests: cancelledMediaRequests,
        page_errors: pageErrors,
        console_errors: consoleErrors,
      };
      fs.writeFileSync(
        path.join(proofDir, "saved-project-result.json"),
        `${JSON.stringify(result, null, 2)}\n`,
      );
      console.log(JSON.stringify(result, null, 2));
      return;
    }

    const created = await page.request.post(`${baseUrl}/video_edit/projects`, {
      data: {
        name: "Genesis Playwright proof",
        fps: 30,
        width: 1280,
        height: 720,
        tracks: [
          { id: "track-video", name: "Video 1", type: "video", clips: [] },
          { id: "track-audio", name: "Audio", type: "audio", clips: [] },
        ],
      },
    });
    assert(created.ok(), `project create failed: ${created.status()}`);
    const project = await created.json();
    const projectId = project.id;
    assert(projectId, "project create returned no id");

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
    await page.waitForFunction(() => (
      window.VideoEditTab?.getActiveProjectId?.() !== null
    ));
    assert(await page.locator(".ve-project-properties").isVisible(),
      "empty project did not show project settings in Properties");
    assert(
      await page.locator(".ve-project-properties input").count() >= 4,
      "project Properties is missing settings controls",
    );
    await page.click("#ve-btn-import-video");
    const videoImport = page.waitForResponse((response) => (
      response.url().includes(`/video_edit/projects/${projectId}/import_clip`)
      && response.request().method() === "POST"
    ), { timeout: 30000 });
    await page.locator('input[data-media-kind="video"]').setInputFiles(sourceVideo);
    const imported = await videoImport;
    assert(imported.ok(), `video import failed: ${imported.status()}`);
    const media = await imported.json();
    assert(media.media_type === "video", `video media_type=${media.media_type}`);
    assert(media.duration_frames === 121, `imported frames=${media.duration_frames}`);
    assert(Math.abs(media.source_fps - 25) < 0.001, `imported fps=${media.source_fps}`);
    await initialPreview;
    await page.waitForFunction(() => (
      document.querySelector("#ve-engine-status")?.textContent.includes("Ready")
    ));
    await page.waitForFunction(() => {
      const diagnostics = window.VideoEditTab?.getDiagnostics?.();
      return diagnostics?.thumbnailClipIds?.length === 1
        && diagnostics?.width === 512
        && diagnostics?.height === 768
        && diagnostics?.fps === 25
        && diagnostics?.mediaCardCount === 1
        && diagnostics?.dockTabCount === 4;
    }, { timeout: 30000 });

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
    const layoutDiagnostics = await page.evaluate(
      () => window.VideoEditTab.getDiagnostics(),
    );
    assert(layoutDiagnostics.mediaCardCount === 1,
      `media cards=${layoutDiagnostics.mediaCardCount}`);
    assert(layoutDiagnostics.dockTabCount === 4,
      `dock tabs=${layoutDiagnostics.dockTabCount}`);
    assert(await page.locator("#ve-edit-toolbar .ve-edit-btn").count() >= 12,
      "timeline editing toolbar is incomplete");

    await page.locator(".ve-media-card").first().click();
    await page.waitForFunction(() => (
      window.VideoEditTab?.getDiagnostics?.().monitorMode === "source"
      && !!window.VideoEditTab?.getDiagnostics?.().sourceMonitorClipId
    ));
    const returnToProgram = waitForPreview();
    await page.locator('.ve-monitor-tab[data-monitor="program"]').click();
    await returnToProgram;
    const baseHash = await readCanvasHash();

    assert(await page.locator(".ve-props-panel").isVisible(), "video import did not expose edits");
    const effectSlider = page.locator(
      '.ve-genesis-control[data-property="sat"] input[type="range"]',
    );
    assert(await effectSlider.count() === 1,
      "native Genesis Saturation property is missing");
    const changedPreview = waitForPreview();
    await effectSlider.evaluate((element) => {
      element.value = "0";
      element.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await changedPreview;
    const effectHash = await readCanvasHash();
    assert(effectHash !== baseHash, "saturation edit did not change the Genesis preview");

    await page.locator('.ve-dock-tab[data-dock="filters"]').click();
    assert(await page.locator(".ve-genesis-filter-catalog").isVisible(),
      "native Genesis filter catalog did not open");
    const enabledCheckbox = page.locator(".ve-filter-applied .ve-filter-row input").first();
    const disabledPreview = waitForPreview();
    await enabledCheckbox.click();
    await disabledPreview;
    const disabledHash = await readCanvasHash();
    assert(disabledHash === baseHash, "removed native filter still changed the Genesis preview");

    await page.locator('.ve-dock-tab[data-dock="properties"]').click();
    const reenabledPreview = waitForPreview();
    await effectSlider.evaluate((element) => {
      element.value = "0";
      element.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await reenabledPreview;
    const reenabledHash = await readCanvasHash();
    assert(reenabledHash === effectHash, "re-enabled effect did not restore edited pixels");

    await page.locator('.ve-dock-tab[data-dock="scopes"]').click();
    const scopePixels = await page.locator("#ve-scope-histogram").evaluate((canvas) => {
      const pixels = canvas.getContext("2d").getImageData(
        0, 0, canvas.width, canvas.height,
      ).data;
      let lit = 0;
      for (let index = 0; index < pixels.length; index += 4) {
        if (pixels[index] + pixels[index + 1] + pixels[index + 2] > 60) lit++;
      }
      return lit;
    });
    assert(scopePixels > 100, `scope pixels=${scopePixels}`);
    for (const selector of [
      "#ve-scope-waveform",
      "#ve-scope-vectorscope",
      "#ve-scope-parade",
    ]) {
      const lit = await page.locator(selector).evaluate((canvas) => {
        const pixels = canvas.getContext("2d").getImageData(
          0, 0, canvas.width, canvas.height,
        ).data;
        let count = 0;
        for (let index = 0; index < pixels.length; index += 4) {
          if (pixels[index] + pixels[index + 1] + pixels[index + 2] > 60) count++;
        }
        return count;
      });
      assert(lit > 100, `${selector} pixels=${lit}`);
    }
    await page.locator('.ve-dock-tab[data-dock="filters"]').click();
    assert(await page.locator(".ve-genesis-filter-catalog").isVisible(),
      "filter catalog did not open");
    await page.locator('.ve-dock-tab[data-dock="properties"]').click();
    assert(await page.locator(".ve-props-panel").isVisible(),
      "properties dock did not restore clip parameters");
    await page.click("#ve-edit-marker");
    assert((await page.evaluate(
      () => window.VideoEditTab.getDiagnostics().markerFrames,
    )).includes(0), "timeline marker was not added");
    await page.click("#ve-edit-snap");
    assert(await page.evaluate(
      () => window.VideoEditTab.getDiagnostics().snapEnabled,
    ) === false, "snap did not turn off");
    await page.click("#ve-edit-snap");
    assert(await page.evaluate(
      () => window.VideoEditTab.getDiagnostics().snapEnabled,
    ) === true, "snap did not turn back on");
    await page.click("#ve-edit-copy");
    await page.click("#ve-edit-paste");
    await page.waitForFunction(() => (
      window.VideoEditTab.getDiagnostics().tracks
        .find((track) => track.type === "video")?.clips.length === 2
    ));
    await page.click("#ve-btn-undo-top");
    await page.waitForFunction(() => (
      window.VideoEditTab.getDiagnostics().tracks
        .find((track) => track.type === "video")?.clips.length === 1
    ));
    await page.locator(".ve-media-card").first().click();
    const restoreProgramAfterUndo = waitForPreview();
    await page.locator('.ve-monitor-tab[data-monitor="program"]').click();
    await restoreProgramAfterUndo;

    await page.click("#ve-btn-import-music");
    const audioImport = page.waitForResponse((response) => (
      response.url().includes(`/video_edit/projects/${projectId}/import_clip`)
      && response.request().method() === "POST"
    ), { timeout: 30000 });
    const waveformReady = page.waitForResponse((response) => (
      response.url().endsWith("/video_edit/waveform")
      && response.request().method() === "POST"
      && response.status() === 200
    ), { timeout: 30000 });
    await page.locator('input[data-media-kind="audio"]').setInputFiles(sourceAudio);
    const importedAudioResponse = await audioImport;
    assert(importedAudioResponse.ok(), `audio import failed: ${importedAudioResponse.status()}`);
    const audioMedia = await importedAudioResponse.json();
    assert(audioMedia.media_type === "audio", `audio media_type=${audioMedia.media_type}`);
    const expectedAudioFrames = Math.round(audioMedia.duration_seconds * media.source_fps);
    assert(audioMedia.duration_frames === expectedAudioFrames,
      `audio frames=${audioMedia.duration_frames}, expected=${expectedAudioFrames}`);
    assert(audioMedia.duration_seconds >= 5 && audioMedia.duration_seconds <= 6,
      `audio duration=${audioMedia.duration_seconds}`);
    const waveformResponse = await waveformReady;
    const waveform = await waveformResponse.json();
    assert(waveform.peaks?.length >= 170,
      `music waveform samples=${waveform.peaks?.length}`);
    assert(Math.max(...waveform.peaks) > 0.01, "music waveform is silent");
    await page.waitForFunction(() => (
      window.VideoEditTab?.getDiagnostics?.().waveformClipIds.length === 1
    ), { timeout: 30000 });
    await page.locator('.ve-dock-tab[data-dock="audio"]').click();
    assert(await page.locator(".ve-audio-mixer-strip").count() >= 2,
      "audio mixer did not expose video and music tracks");
    assert(await page.locator("#ve-audio-waveform").isVisible(),
      "audio waveform scope is missing");
    const muteButton = page.locator(".ve-audio-mixer-strip").last()
      .locator(".ve-audio-mixer-buttons button").first();
    await muteButton.click();
    assert(await muteButton.evaluate((button) => button.classList.contains("ve-active")),
      "audio track did not mute");
    await muteButton.click();
    assert(!await muteButton.evaluate((button) => button.classList.contains("ve-active")),
      "audio track did not unmute");

    const previewHashBeforePlayback = await readCanvasHash();
    await page.click("#ve-btn-play");
    await page.waitForFunction(() => (
      window.VideoEditTab?.getDiagnostics?.().currentFrame >= 15
      && window.VideoEditTab?.getDiagnostics?.().previewVideoTime >= 0.4
    ), { timeout: 5000 });
    const playbackDiagnostics = await page.evaluate(
      () => window.VideoEditTab.getDiagnostics(),
    );
    const previewHashDuringPlayback = await readCanvasHash();
    await page.click("#ve-btn-play");
    assert(playbackDiagnostics.currentFrame >= 15,
      `playback frame=${playbackDiagnostics.currentFrame}`);
    assert(playbackDiagnostics.previewVideoPaused === false,
      "preview video was paused while timeline was playing");
    assert(previewHashDuringPlayback !== previewHashBeforePlayback,
      "large preview did not change during playback");
    await page.waitForTimeout(2500);

    await page.locator(".ve-media-card").first().click();
    const exportSelectionPreview = waitForPreview();
    await page.locator('.ve-monitor-tab[data-monitor="program"]').click();
    await exportSelectionPreview;
    const screenshotPath = path.join(proofDir, "editor.png");
    await page.screenshot({ path: screenshotPath, fullPage: true });
    await page.click("#ve-btn-export");
    const resolution = await page.locator("#ve-exp-res option").first().textContent();
    const fps = await page.locator("#ve-exp-fps").inputValue();
    assert(resolution === "512x768", `export resolution=${resolution}`);
    assert(fps === "25", `export fps=${fps}`);
    assert(
      await page.locator("#ve-exp-fps").getAttribute("readonly") !== null,
      "export fps is not bound to the project timeline",
    );
    assert(await page.locator("#ve-exp-audio").isChecked(), "audio export is not enabled");
    await page.check('input[name="ve-exp-range"][value="selection"]');
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
      "-show_entries", "stream=codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,sample_rate,channels",
      "-show_entries", "format=duration,size",
      "-of", "json",
      outputPath,
    ]));
    const videoStream = probe.streams?.find((stream) => stream.codec_type === "video") || {};
    const audioStream = probe.streams?.find((stream) => stream.codec_type === "audio") || {};
    assert(videoStream.codec_name === "h264", `codec=${videoStream.codec_name}`);
    assert(videoStream.width === 512 && videoStream.height === 768,
      `size=${videoStream.width}x${videoStream.height}`);
    assert(videoStream.r_frame_rate === "25/1", `r_frame_rate=${videoStream.r_frame_rate}`);
    assert(videoStream.avg_frame_rate === "25/1", `avg_frame_rate=${videoStream.avg_frame_rate}`);
    assert(Number(videoStream.nb_frames) === media.duration_frames,
      `frames=${videoStream.nb_frames}`);
    assert(audioStream.codec_name === "aac", `audio codec=${audioStream.codec_name}`);
    assert(Number(audioStream.sample_rate) === 48000,
      `audio sample_rate=${audioStream.sample_rate}`);
    assert(Number(audioStream.channels) === 2, `audio channels=${audioStream.channels}`);
    const volumeProbe = spawnSync("ffmpeg", [
      "-hide_banner", "-i", outputPath,
      "-map", "0:a:0",
      "-af", "volumedetect",
      "-f", "null", "-",
    ], { encoding: "utf8" });
    assert(volumeProbe.status === 0, `audio volume probe failed: ${volumeProbe.stderr}`);
    const meanVolumeMatch = volumeProbe.stderr.match(/mean_volume:\s*(-?[0-9.]+) dB/);
    const outputAudioMeanDb = Number(meanVolumeMatch?.[1]);
    assert(Number.isFinite(outputAudioMeanDb) && outputAudioMeanDb > -80,
      `exported music is silent: mean_volume=${meanVolumeMatch?.[1] || "missing"}`);

    const contactSheet = path.join(proofDir, "export-contact.png");
    runChecked("ffmpeg", [
      "-v", "error", "-i", outputPath,
      "-vf", "select=eq(n\\,0)+eq(n\\,60)+eq(n\\,120),scale=256:-1,tile=3x1",
      "-frames:v", "1", "-y", contactSheet,
    ]);

    const saved = await page.request.get(
      `${baseUrl}/video_edit/projects/${projectId}`,
    );
    const savedProject = await saved.json();
    const savedGenesis = savedProject.tracks?.[0]?.clips?.[0]?.genesis;
    const savedMusic = savedProject.tracks
      ?.find((track) => track.type === "audio")
      ?.clips?.[0];
    assert(savedGenesis?.sat === 0, "native Genesis saturation was not persisted");
    assert(savedMusic?.source_path, "music clip was not persisted");
    assert(savedMusic.endFrame - savedMusic.startFrame === audioMedia.duration_frames,
      "music duration was not persisted");
    assert(requestFailures.length === 0, `request failures: ${requestFailures.join(" | ")}`);
    assert(pageErrors.length === 0, `page errors: ${pageErrors.join(" | ")}`);
    assert(consoleErrors.length === 0, `console errors: ${consoleErrors.join(" | ")}`);

    result = {
      schema: "serenity.genesis_browser_gate.v2",
      status: "PASS",
      project_id: projectId,
      source_video: sourceVideo,
      source_frames: media.duration_frames,
      source_fps: media.source_fps,
      source_audio: sourceAudio,
      music_frames: audioMedia.duration_frames,
      music_duration: audioMedia.duration_seconds,
      thumbnail_strip_ready: true,
      waveform_ready: true,
      playback_frame_reached: playbackDiagnostics.currentFrame,
      preview_video_time: playbackDiagnostics.previewVideoTime,
      preview_pixels_changed_during_playback:
        previewHashDuringPlayback !== previewHashBeforePlayback,
      genesis_layout: {
        media_bin: layoutDiagnostics.mediaCardCount === 1,
        program_source_monitors: true,
        timeline_toolbar_buttons:
          await page.locator("#ve-edit-toolbar .ve-edit-btn").count(),
        persistent_dock_tabs: layoutDiagnostics.dockTabCount,
        live_scope_pixels: scopePixels,
        audio_mixer_tracks: await page.locator(".ve-audio-mixer-strip").count(),
        marker_action: true,
        snap_toggle: true,
        copy_paste_undo: true,
        audio_mute_toggle: true,
        project_properties_without_selection: true,
      },
      effect: "saturation=0",
      effect_pixels_changed: effectHash !== baseHash,
      disabled_effect_restored_base: disabledHash === baseHash,
      resolution,
      fps: Number(fps),
      output_path: outputPath,
      output_frames: Number(videoStream.nb_frames),
      output_duration: Number(probe.format?.duration),
      output_audio_codec: audioStream.codec_name,
      output_audio_sample_rate: Number(audioStream.sample_rate),
      output_audio_channels: Number(audioStream.channels),
      output_audio_mean_db: outputAudioMeanDb,
      screenshot: screenshotPath,
      contact_sheet: contactSheet,
      request_failures: requestFailures,
      cancelled_media_requests: cancelledMediaRequests,
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
