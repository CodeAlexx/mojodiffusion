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

function checkpoint(message) {
  process.stderr.write(`[canvas-parity] ${message}\n`);
}

(async () => {
  const chromium = loadChromium();
  checkpoint("launch browser");
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 }, acceptDownloads: true });
  const errors = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(String(error)));

  await page.goto(process.env.SERENITY_URL || "http://127.0.0.1:7811", { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => window.SerenityAPI && typeof SerenityAPI.uploadImage === "function");
  const uploadedPath = await page.evaluate(() => SerenityAPI.uploadImage(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=",
    "canvas_path_contract",
  ));
  try {
    assert(path.isAbsolute(uploadedPath) && uploadedPath.includes(`${path.sep}uploads${path.sep}`),
      `Canvas upload returned a non-worker-readable path: ${uploadedPath}`);
    assert(fs.existsSync(uploadedPath), `Canvas upload path does not exist: ${uploadedPath}`);
  } finally {
    if (path.isAbsolute(uploadedPath) && fs.existsSync(uploadedPath)) fs.unlinkSync(uploadedPath);
  }
  checkpoint("worker-readable upload path passed");
  checkpoint("open Canvas");
  await page.click('.nav-btn[data-tab="canvas"]');
  await page.waitForSelector("#cv-model option:not([disabled])", { state: "attached" });
  await page.waitForSelector(".cv-gallery-item");

  const layout = await page.evaluate(() => {
    const panel = document.querySelector("#panel-canvas").getBoundingClientRect();
    const center = document.querySelector(".cv-center").getBoundingClientRect();
    const sam = document.querySelector('.cv-tool-btn[data-tool="sam"]');
    return {
      panel: { width: panel.width, height: panel.height },
      center: { width: center.width, height: center.height },
      samDisabled: sam.disabled,
      samReason: sam.title,
    };
  });
  assert(layout.center.width >= 800 && layout.center.height >= 700, "Canvas workspace is not usable at 1600x900");
  assert(!layout.samDisabled, `SAM3 should be available from the installed mask service: ${layout.samReason}`);
  checkpoint("layout and capability admission passed");

  await page.selectOption("#cv-edit-mode", "flowedit");
  await page.dispatchEvent("#cv-edit-mode", "change");
  await page.setInputFiles("#cv-import-file", path.join(process.cwd(), "output/serenity_ui_out/job-0021.png"));
  await page.waitForFunction(() => {
    const preview = document.querySelector("#cv-source-preview");
    return preview && preview.complete && preview.naturalWidth > 0 && getComputedStyle(preview).display !== "none";
  });
  const editLayout = await page.evaluate(() => {
    const source = document.querySelector("#cv-source-pane").getBoundingClientRect();
    const result = document.querySelector(".cv-result-pane").getBoundingClientRect();
    const bbox = CanvasTab.getCompositorContext().boundingBox;
    return {
      source: { x: source.x, width: source.width, height: source.height },
      result: { x: result.x, width: result.width, height: result.height },
      sourceVisible: getComputedStyle(document.querySelector("#cv-source-pane")).display !== "none",
      accept: document.querySelector("#cv-import-file").accept,
      button: document.querySelector("#cv-import-btn").textContent,
      engine: document.querySelector("#cv-edit-engine").value,
      bboxWidth: bbox.width(),
      bboxHeight: bbox.height(),
      sourcePlaceholder: document.querySelector("#cv-edit-source-prompt").placeholder,
    };
  });
  assert(editLayout.sourceVisible, "FlowEdit did not open the source pane");
  assert(editLayout.source.x < editLayout.result.x, "Source and result panes are not side-by-side");
  assert(Math.abs(editLayout.source.width - editLayout.result.width) < 3, "Source and result panes are not equal width");
  assert(editLayout.accept === "image/*" && editLayout.button.includes("source image"), "Edit file picker is not image-scoped");
  assert(editLayout.engine === "krea2_raw_1024" && editLayout.bboxWidth === 1024 && editLayout.bboxHeight === 1024,
    `FlowEdit defaulted to ${editLayout.engine} at ${editLayout.bboxWidth}x${editLayout.bboxHeight}`);
  assert(/auto-described/i.test(editLayout.sourcePlaceholder),
    "FlowEdit still presents source description as a required manual field");

  const flowedit = await page.evaluate(() => WorkflowBuilder.buildFlowEdit({
    engine: "krea2_raw_512",
    initImageName: "/tmp/source.png",
    sourcePrompt: "a person standing indoors",
    prompt: "the same person standing outdoors",
    negPrompt: "",
    steps: 28,
    nmax: 24,
    nmin: 0,
    srcCfg: 1.5,
    cfg: 5.5,
    seed: 42,
    autoMask: true,
    maskQ: 0.7,
    maskDilate: 1,
    maskWarmup: 4,
  }));
  assert(flowedit["8"].class_type === "Krea2FlowEdit", "Canvas did not build the Krea2 FlowEdit node");
  assert(flowedit["8"].inputs.width === 512 && flowedit["8"].inputs.height === 512,
    "Canvas did not preserve Krea2 FlowEdit compiled geometry");
  assert(flowedit["8"].inputs.nmax === 24 && flowedit["8"].inputs.auto_mask === true,
    "Canvas substituted FlowEdit controls");
  const turboFlowedit = await page.evaluate(() => WorkflowBuilder.buildFlowEdit({
    engine: "krea2_turbo_1024",
    initImageName: "/tmp/source.png",
    sourcePrompt: "a person standing indoors",
    prompt: "the same person standing outdoors",
    negPrompt: "",
    steps: 28,
    nmax: 24,
    nmin: 0,
    srcCfg: 1.5,
    cfg: 5.5,
    seed: 42,
    autoMask: true,
    maskQ: 0.7,
    maskDilate: 1,
    maskWarmup: 4,
  }));
  assert(turboFlowedit["1"].inputs.unet_name === "krea2_turbo.safetensors",
    "Krea2 Turbo FlowEdit did not select the Turbo checkpoint");
  assert(turboFlowedit["8"].inputs.width === 1024 && turboFlowedit["8"].inputs.height === 1024,
    "Krea2 Turbo FlowEdit did not preserve the 1024 compiled geometry");
  const inpaint = await page.evaluate(() => WorkflowBuilder.buildInpaint({
    model: "zimage_base",
    prompt: "replace the background",
    negPrompt: "",
    initImageName: "/tmp/source.png",
    maskImageName: "/tmp/mask.png",
    width: 1024,
    height: 1024,
    steps: 4,
    cfg: 1,
    denoise: 0.65,
    seed: 42,
  }));
  assert(inpaint["10"].class_type === "ImageToMask" && inpaint["10"].inputs.channel === "red",
    "Z-Image inpaint did not extract the uploaded grayscale mask channel");
  assert(inpaint["11"].class_type === "SetLatentNoiseMask" && inpaint["11"].inputs.mask[0] === "10",
    "Z-Image inpaint did not connect the mask to the latent");

  await page.selectOption("#cv-edit-mode", "inpaint");
  await page.dispatchEvent("#cv-edit-mode", "change");
  const maskedModeLabel = await page.locator('#cv-edit-mode option[value="inpaint"]').textContent();
  assert(maskedModeLabel === "Masked Edit - LanPaint", `masked-edit mode label changed: ${maskedModeLabel}`);
  const lanpaintLayout = await page.evaluate(() => {
    const bbox = CanvasTab.getCompositorContext().boundingBox;
    return {
      engine: document.querySelector("#cv-lanpaint-engine").value,
      engines: Array.from(document.querySelector("#cv-lanpaint-engine").options).map((option) => ({
        value: option.value,
        label: option.textContent,
        disabled: option.disabled,
      })),
      model: CanvasTab.getCompositorContext().genState.model,
      width: bbox.width(),
      height: bbox.height(),
      steps: Number(document.querySelector("#cv-steps").value),
      cfg: Number(document.querySelector("#cv-cfg").value),
      denoise: Number(document.querySelector("#cv-denoise").value),
      controlsVisible: getComputedStyle(document.querySelector("#cv-lanpaint-section")).display !== "none",
      modelHidden: getComputedStyle(document.querySelector("#cv-model-row")).display === "none",
    };
  });
  assert(lanpaintLayout.engine === "krea2_turbo_1024" && /krea/i.test(lanpaintLayout.model),
    `LanPaint selected ${lanpaintLayout.engine} with model ${lanpaintLayout.model}`);
  const enabledMaskedEditEngines = lanpaintLayout.engines
    .filter((engine) => !engine.disabled)
    .map((engine) => engine.value);
  assert(JSON.stringify(enabledMaskedEditEngines) ===
    JSON.stringify(["krea2_turbo_1024", "krea2_raw_1024", "zimage_base_1024"]),
  `masked-edit engines do not match admitted model routes: ${JSON.stringify(lanpaintLayout.engines)}`);
  const deferredMaskedEditEngines = lanpaintLayout.engines
    .filter((engine) => engine.disabled)
    .map((engine) => engine.value);
  const expectedDeferredMaskedEditEngines = [
    "zimage_turbo_1024", "ideogram4_1024", "anima_1024",
    "flux2_klein_1024", "flux2_dev_1024", "qwen_image_1024",
    "qwen_image_edit_1024", "flux1_dev_1024", "sdxl_1024",
    "sd35_1024", "hunyuan_1024", "wan22_t2i_1024", "hidream_1024",
    "sd15_512",
  ];
  assert(expectedDeferredMaskedEditEngines.every((engine) => deferredMaskedEditEngines.includes(engine)),
    `LanPaint upstream model inventory is incomplete: ${JSON.stringify(lanpaintLayout.engines)}`);
  const deferredLanPaintGraphs = await page.evaluate(() => {
    const models = [
      "ideogram4_fp8_scaled.safetensors",
      "anima.safetensors",
      "flux-2-klein-base-9b.safetensors",
      "flux2-dev.safetensors",
      "qwen-image-2512.safetensors",
      "qwen-image-edit-2509.safetensors",
      "flux1-dev.safetensors",
      "sdxl-base-1.0.safetensors",
      "sd3.5-large.safetensors",
      "sd15-base.safetensors",
    ];
    return models.map((model) => {
      const workflow = WorkflowBuilder.buildLanPaintCandidate({
        model, prompt: "replace the masked object", negPrompt: "",
        initImageName: "/tmp/source.png", maskImageName: "/tmp/mask.png",
        width: model.includes("sd15") ? 512 : 1024,
        height: model.includes("sd15") ? 512 : 1024,
        steps: 8, cfg: 1, sampler: "euler", scheduler: "simple", seed: 42,
        lanpaintNumSteps: 5, lanpaintLambda: 16, lanpaintStepSize: 0.2,
        lanpaintBeta: 1, lanpaintFriction: 15, lanpaintPromptMode: "Image First",
        lanpaintEarlyStop: 0, lanpaintInnerThreshold: 0.001,
        lanpaintInnerPatience: 0, lanpaintBlendOverlap: 9,
      });
      const types = Object.values(workflow).map((node) => node.class_type);
      return { model, types };
    });
  });
  deferredLanPaintGraphs.forEach(({ model, types }) => {
    ["VAEEncode", "ImageToMask", "SetLatentNoiseMask", "LanPaint_KSamplerAdvanced", "LanPaint_MaskBlend"]
      .forEach((nodeType) => assert(types.includes(nodeType),
        `${model} deferred LanPaint graph is missing ${nodeType}: ${JSON.stringify(types)}`));
  });
  assert(lanpaintLayout.width === 1024 && lanpaintLayout.height === 1024,
    `LanPaint Canvas initialized at ${lanpaintLayout.width}x${lanpaintLayout.height}`);
  assert(lanpaintLayout.steps === 8 && lanpaintLayout.cfg === 1 && lanpaintLayout.denoise === 1,
    "LanPaint Canvas did not expose the tested full-denoise Turbo defaults");
  assert(lanpaintLayout.controlsVisible && lanpaintLayout.modelHidden,
    "LanPaint controls or authoritative engine selector are not visible");

  await page.selectOption("#cv-lanpaint-engine", "zimage_base_1024");
  await page.dispatchEvent("#cv-lanpaint-engine", "change");
  const zimageMaskedLayout = await page.evaluate(() => {
    const ctx = CanvasTab.getCompositorContext();
    return {
      engine: document.querySelector("#cv-lanpaint-engine").value,
      model: ctx.genState.model,
      arch: ctx.genState.arch,
      steps: Number(document.querySelector("#cv-steps").value),
      cfg: Number(document.querySelector("#cv-cfg").value),
      denoise: Number(document.querySelector("#cv-denoise").value),
      denoiseDisabled: document.querySelector("#cv-denoise").disabled,
      lanpaintControlsVisible: getComputedStyle(document.querySelector("#cv-lanpaint-controls")).display !== "none",
      helper: document.querySelector("#cv-masked-edit-helper").textContent,
      button: document.querySelector("#cv-generate-btn").textContent,
    };
  });
  assert(zimageMaskedLayout.engine === "zimage_base_1024" && zimageMaskedLayout.model === "zimage_base" &&
    zimageMaskedLayout.arch === "zimage", `Z-Image masked engine did not select its canonical model: ${JSON.stringify(zimageMaskedLayout)}`);
  assert(zimageMaskedLayout.steps === 4 && zimageMaskedLayout.cfg === 1 && zimageMaskedLayout.denoise === 0.65,
    `Z-Image masked profile is not visible in the controls: ${JSON.stringify(zimageMaskedLayout)}`);
  assert(!zimageMaskedLayout.denoiseDisabled && !zimageMaskedLayout.lanpaintControlsVisible &&
    /Z-Image/.test(zimageMaskedLayout.helper) && zimageMaskedLayout.button === "Edit masked area",
  `Z-Image did not reuse the shared masked-edit workspace correctly: ${JSON.stringify(zimageMaskedLayout)}`);

  const zimageLoraInpaint = await page.evaluate(() => WorkflowBuilder.buildInpaint({
    model: "zimage_base", prompt: "replace the background", negPrompt: "",
    initImageName: "/tmp/source.png", maskImageName: "/tmp/mask.png",
    width: 1024, height: 1024, steps: 4, cfg: 1, denoise: 0.65, seed: 42,
    loras: [{ name: "zimage_test_lora.safetensors", strength: 0.8 }],
  }));
  const zimageLoraNode = Object.values(zimageLoraInpaint).find((node) => node.class_type === "LoraLoader");
  assert(zimageLoraNode && zimageLoraInpaint["4"].inputs.model[0] !== "1",
    "Z-Image masked edit did not preserve the shared LoRA-loader path");

  const zimageInpaintPreflight = await page.evaluate(async (workflow) => {
    const response = await fetch("/v1/preflight", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ workflow }),
    });
    return { status: response.status, body: await response.json() };
  }, inpaint);
  assert(zimageInpaintPreflight.status === 200 && zimageInpaintPreflight.body.admitted === true,
    `Z-Image masked-edit graph failed real server preflight: ${JSON.stringify(zimageInpaintPreflight.body)}`);

  const maskedEditScreenshot = process.env.SERENITY_CANVAS_MASKED_EDIT_SCREENSHOT || "/tmp/serenity-canvas-masked-edit-parity.png";
  await page.screenshot({ path: maskedEditScreenshot, fullPage: true });

  await page.selectOption("#cv-lanpaint-engine", "krea2_turbo_1024");
  await page.dispatchEvent("#cv-lanpaint-engine", "change");

  const lanpaint = await page.evaluate(() => WorkflowBuilder.buildInpaint({
    model: "krea2-turbo",
    prompt: "replace the hand with a red leather glove",
    negPrompt: "",
    initImageName: "/tmp/source.png",
    maskImageName: "/tmp/mask.png",
    width: 1024,
    height: 1024,
    steps: 8,
    cfg: 1,
    seed: 42,
    lanpaintNumSteps: 5,
    lanpaintLambda: 16,
    lanpaintStepSize: 0.2,
    lanpaintBeta: 1,
    lanpaintFriction: 15,
    lanpaintPromptMode: "Image First",
    lanpaintBlendOverlap: 9,
    lanpaintEarlyStop: 1,
    lanpaintInnerThreshold: 0,
    lanpaintInnerPatience: 1,
    loras: [],
  }));
  assert(lanpaint["13"].class_type === "LanPaint_KSamplerAdvanced",
    "Canvas did not build the advanced LanPaint sampler");
  assert(lanpaint["13"].inputs.LanPaint_NumSteps === 5 &&
    lanpaint["13"].inputs.LanPaint_Lambda === 16 &&
    lanpaint["13"].inputs.LanPaint_StepSize === 0.2 &&
    lanpaint["13"].inputs.LanPaint_Friction === 15,
    "Canvas substituted visible LanPaint controls");
  assert(lanpaint["15"].class_type === "LanPaint_MaskBlend" &&
    lanpaint["15"].inputs.blend_overlap === 9,
    "Canvas did not build the final source-preserving LanPaint mask blend");
  const lanpaintPreflight = await page.evaluate(async (workflow) => {
    const response = await fetch("/v1/preflight", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ workflow }),
    });
    return { status: response.status, body: await response.json() };
  }, lanpaint);
  assert(lanpaintPreflight.status === 200 && lanpaintPreflight.body.admitted === true,
    `Krea2 Canvas LanPaint failed real server preflight: ${JSON.stringify(lanpaintPreflight.body)}`);
  checkpoint("Krea2 LanPaint Canvas layout, controls, graph, and preflight passed");

  await page.selectOption("#cv-edit-mode", "flowedit");
  await page.dispatchEvent("#cv-edit-mode", "change");

  await page.click('.cv-tool-btn[data-tool="sam"]');
  await page.waitForSelector("#sam-toolbar", { state: "visible" });
  await page.fill("#sam-text-input", "blue teapot");
  await page.click("#sam-detect-btn");
  await page.waitForSelector(".sam-result-item", { state: "visible", timeout: 60_000 });
  const samResult = await page.evaluate(() => ({
    count: document.querySelectorAll(".sam-result-item").length,
    confidence: document.querySelector(".sam-result-conf").textContent,
    status: document.querySelector("#sam-status").textContent,
    statusVisible: getComputedStyle(document.querySelector("#sam-status")).display !== "none",
  }));
  assert(samResult.count > 0, "Canvas SAM3 request returned no masks");
  assert(!samResult.statusVisible && samResult.status === "", "Canvas SAM3 stayed in its loading state after success");
  await page.click("#sam-apply");
  await page.waitForFunction(() => Array.from(document.querySelectorAll(".cv-layer-name"))
    .some((node) => node.textContent.startsWith("Mask:")));
  checkpoint("live Canvas-to-SAM3 masking passed");

  const editScreenshot = process.env.SERENITY_CANVAS_EDIT_SCREENSHOT || "/tmp/serenity-canvas-edit-parity.png";
  await page.screenshot({ path: editScreenshot, fullPage: true });

  await page.selectOption("#cv-edit-mode", "dynaedit");
  await page.dispatchEvent("#cv-edit-mode", "change");
  const dyna = await page.evaluate(() => ({
    note: document.querySelector("#cv-edit-runtime-note").textContent,
    accept: document.querySelector("#cv-import-file").accept,
    button: document.querySelector("#cv-generate-btn").textContent,
  }));
  assert(dyna.note.includes("no web request runner"), "DynaEdit does not explain its exact product boundary");
  assert(dyna.accept === "video/*" && dyna.button.includes("unavailable"), "DynaEdit UI silently implies a runnable backend");
  checkpoint("two-pane editing and FlowEdit request passed");

  const gallerySource = await page.locator(".cv-gallery-media img").first().getAttribute("src");
  assert(gallerySource, "Canvas gallery has no selectable image result");
  await page.locator(".cv-gallery-media").filter({ has: page.locator("img") }).first().click();
  const gallerySelection = await page.evaluate(() => {
    const bbox = CanvasTab.getCompositorContext().boundingBox;
    const preview = document.querySelector("#cv-source-preview");
    return {
      width: bbox.width(),
      height: bbox.height(),
      source: preview && preview.getAttribute("src"),
    };
  });
  assert(gallerySelection.width === 1024 && gallerySelection.height === 1024,
    `Canvas gallery selection used ${gallerySelection.width}x${gallerySelection.height}, expected 1024x1024`);
  assert(gallerySelection.source === gallerySource,
    "Canvas gallery selection did not become the exact style source");
  await page.evaluate(() => CanvasStaging.deactivate());
  checkpoint("Canvas gallery selection preserves the 1024 style source contract");

  let stylePromptBody = null;
  const captionRequests = [];
  await page.route("**/v1/caption", async (route) => {
    captionRequests.push(route.request().postDataJSON());
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ caption: JSON.stringify({
        source: "a woman seated beside a pool, preserving her pose, framing, and background",
        style: "polished liquid chrome sculpture, mirror reflections, monochrome metallic finish, crisp highlights",
      }) }),
    });
  });
  await page.route("**/prompt", async (route) => {
    stylePromptBody = route.request().postDataJSON();
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ prompt_id: "style-parity" }),
    });
  });
  await page.setInputFiles("#ref-file-input", path.join(process.cwd(), "output/serenity_ui_out/job-0025.png"));
  await page.waitForFunction(() => {
    const source = document.querySelector("#cv-source-preview");
    const style = document.querySelector("#cv-style-preview");
    return document.querySelector("#cv-edit-mode").value === "style" &&
      source && source.complete && source.naturalWidth > 0 &&
      style && style.complete && style.naturalWidth > 0 &&
      getComputedStyle(style).display !== "none";
  });
  await page.selectOption("#cv-edit-engine", "krea2_raw_1024");
  await page.dispatchEvent("#cv-edit-engine", "change");
  // Clear gallery metadata so this case exercises the one-pass combined
  // source/style analyzer contract represented by the mocked caption below.
  await page.fill("#cv-edit-source-prompt", "");
  const styleLayout = await page.evaluate(() => {
    const sourcePane = document.querySelector("#cv-source-pane").getBoundingClientRect();
    const resultPane = document.querySelector(".cv-result-pane").getBoundingClientRect();
    const primary = document.querySelector("#cv-primary-source-slot").getBoundingClientRect();
    const style = document.querySelector("#cv-style-source-slot").getBoundingClientRect();
    const bbox = CanvasTab.getCompositorContext().boundingBox;
    return {
      sourceWidth: sourcePane.width,
      resultWidth: resultPane.width,
      primaryHeight: primary.height,
      styleHeight: style.height,
      bboxWidth: bbox.width(),
      bboxHeight: bbox.height(),
      title: document.querySelector("#cv-result-pane-title").textContent,
      engine: document.querySelector("#cv-edit-engine").value,
      engineDisabled: document.querySelector("#cv-edit-engine").disabled,
      referenceMethod: document.querySelector(".ref-method").value,
      entireImageVisible: getComputedStyle(document.querySelector("#cv-style-entire-image")).display !== "none",
      entireImageChecked: document.querySelector("#cv-style-entire-image").checked,
      sourceNaturalWidth: document.querySelector("#cv-source-preview").naturalWidth,
      sourceNaturalHeight: document.querySelector("#cv-source-preview").naturalHeight,
      toolRailX: document.querySelector(".cv-tools").getBoundingClientRect().x,
      resultX: resultPane.x,
    };
  });
  assert(styleLayout.resultWidth > styleLayout.sourceWidth * 1.7,
    "Style result pane is not the large right-hand pane");
  assert(Math.abs(styleLayout.primaryHeight - styleLayout.styleHeight) < 3,
    "Style source and reference panes are not stacked evenly");
  assert(styleLayout.bboxWidth === 1024 && styleLayout.bboxHeight === 1024,
    "Style result did not initialize at 1024x1024");
  assert(styleLayout.title.includes("1024 × 1024"), "Style result size is not visible in the pane header");
  assert(styleLayout.engine === "krea2_raw_1024" && !styleLayout.engineDisabled,
    "Style mode did not expose the Krea2 1024 FlowEdit profile");
  assert(styleLayout.referenceMethod === "style", "Reference Method=Style did not select style mode");
  assert(styleLayout.entireImageVisible, "Style mode did not expose the full-frame style checkbox");
  assert(styleLayout.entireImageChecked,
    "Style mode did not default to the coherent full-frame path");
  assert(styleLayout.sourceNaturalWidth > 0 && styleLayout.sourceNaturalHeight > 0,
    "Style mode lost the exact selected gallery source");
  assert(styleLayout.toolRailX >= styleLayout.resultX,
    "Canvas tool rail obscures the source/style reference panes");

  const styleScreenshot = process.env.SERENITY_CANVAS_STYLE_SCREENSHOT || "/tmp/serenity-canvas-style-parity.png";
  await page.screenshot({ path: styleScreenshot, fullPage: true });
  await page.check("#cv-style-entire-image");
  await page.fill("#cv-prompt", "keep the source person unchanged");
  const stylePromptResponse = page.waitForResponse((response) => response.url().endsWith("/prompt"));
  await page.click("#cv-generate-btn");
  await stylePromptResponse;
  assert(captionRequests.length === 1, "Style mode should analyze source and style in one Mojo vision pass");
  assert(stylePromptBody && stylePromptBody.prompt, "Style mode did not submit a graph workflow");
  const styleWorkflow = stylePromptBody.prompt;
  assert(styleWorkflow["8"].class_type === "Krea2FlowEdit", "Style mode did not use Krea2 FlowEdit");
  assert(styleWorkflow["1"].inputs.unet_name === "krea2_raw.safetensors",
    "Krea2 1024 Style mode did not select the FlowEdit-compatible Raw checkpoint");
  assert(styleWorkflow["8"].inputs.width === 1024 && styleWorkflow["8"].inputs.height === 1024,
    "Style workflow did not preserve the 1024x1024 compiled profile");
  assert(styleWorkflow["8"].inputs.steps === 28 && styleWorkflow["8"].inputs.nmax === 24 &&
    styleWorkflow["8"].inputs.src_cfg === 1.5 && styleWorkflow["8"].inputs.tgt_cfg === 5.5 &&
    styleWorkflow["8"].inputs.mask_warmup === 4,
    "Krea2 Raw Style mode did not use the visible FlowEdit profile");
  assert(styleWorkflow["8"].inputs.auto_mask === false,
    "Entire image style checkbox did not disable the localized change mask");
  assert(path.isAbsolute(styleWorkflow["3"].inputs.image) && styleWorkflow["3"].inputs.image.includes(`${path.sep}uploads${path.sep}`),
    "Style FlowEdit did not preserve the server-returned absolute upload path");
  const targetCaption = styleWorkflow["6"].inputs.text;
  assert(targetCaption.startsWith("a woman seated beside a pool"),
    "Style target conditioning does not begin with the complete source scene");
  assert(targetCaption.includes("polished liquid chrome sculpture"),
    "Selected style image analysis did not reach FlowEdit target conditioning");
  assert(targetCaption.includes("same subject identities") && targetCaption.includes("spatial composition"),
    "Style target conditioning does not explicitly preserve source content and layout");
  assert(targetCaption.indexOf("a woman seated beside a pool") < targetCaption.indexOf("polished liquid chrome sculpture"),
    "Style conditioning is not source-anchored before the rendering style");
  assert(!Object.values(styleWorkflow).some((node) => /IPAdapter/i.test(node.class_type)),
    "Style reference was incorrectly duplicated as an IP-Adapter branch");
  const stylePreflight = await page.evaluate(async (workflow) => {
    const response = await fetch("/v1/preflight", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ workflow }),
    });
    return { status: response.status, body: await response.json() };
  }, styleWorkflow);
  assert(stylePreflight.status === 200 && stylePreflight.body.admitted === true,
    `Style FlowEdit failed real server preflight: ${JSON.stringify(stylePreflight.body)}`);
  checkpoint("style reference layout and FlowEdit conditioning passed");

  await page.unroute("**/v1/caption");
  await page.unroute("**/prompt");
  await page.reload({ waitUntil: "domcontentloaded" });
  await page.click('.nav-btn[data-tab="canvas"]');
  await page.waitForSelector(".cv-layer-row");
  await page.waitForSelector("#cv-model option:not([disabled])", { state: "attached" });

  await page.selectOption("#cv-edit-mode", "create");
  await page.dispatchEvent("#cv-edit-mode", "change");
  await page.locator(".cv-layer-row").filter({ has: page.locator(".cv-layer-badge.draw") }).first().click();

  const transparency = await page.evaluate(() => {
    const ctx = CanvasTab.getToolContext();
    const layer = ctx.getActiveLayer();
    layer.konvaLayer.add(new Konva.Rect({ x: 20, y: 20, width: 80, height: 80, fill: "#fff", listening: false }));
    layer.konvaLayer.batchDraw();
    ctx.toggleActiveLayerTransparency();
    const tool = CanvasTools.get("brush");
    const oldPointer = ctx.getRelativePointerPosition;
    ctx.getRelativePointerPosition = () => ({ x: 35, y: 35 });
    tool.onMouseDown(ctx, { target: { name: () => "" }, evt: {} });
    tool.onMouseUp(ctx);
    const child = layer.konvaLayer.getChildren()[layer.konvaLayer.getChildren().length - 1];
    ctx.getRelativePointerPosition = oldPointer;
    return {
      locked: layer.data.lockTransparency,
      composite: child.globalCompositeOperation(),
      rowMarked: document.querySelector(".cv-layer-row.active").classList.contains("transparency-locked"),
    };
  });
  assert(transparency.locked && transparency.composite === "source-atop", "Lock transparency did not clip paint");
  assert(transparency.rowMarked, "Lock transparency is not visible in the layer list");
  checkpoint("lock transparency passed");

  await page.evaluate(() => localStorage.removeItem("serenity-canvas-gallery-boards-v1"));
  await page.reload({ waitUntil: "domcontentloaded" });
  checkpoint("reload for board round trip");
  await page.click('.nav-btn[data-tab="canvas"]');
  await page.waitForSelector(".cv-gallery-item");
  page.once("dialog", async (dialog) => dialog.accept("Parity Board"));
  await page.click("#cv-gallery-board-new");
  await page.selectOption("#cv-gallery-board-filter", "all");
  await page.waitForSelector(".cv-gallery-board-assign");
  await page.selectOption(".cv-gallery-board-assign", "Parity Board");
  await page.selectOption("#cv-gallery-board-filter", "Parity Board");
  await page.waitForFunction(() => document.querySelectorAll(".cv-gallery-item").length === 1);
  const boardState = await page.evaluate(() => JSON.parse(localStorage.getItem("serenity-canvas-gallery-boards-v1")));
  assert(boardState.boards.includes("Parity Board"), "Gallery board was not persisted");
  assert(Object.values(boardState.assignments).includes("Parity Board"), "Gallery assignment was not persisted");
  checkpoint("gallery boards passed");

  await page.selectOption("#cv-model", "ltx-2.3-22b-dev-fp8");
  await page.dispatchEvent("#cv-model", "change");
  await page.click("#cv-load-ltx2-template");
  await page.waitForSelector("#cv-lora-list .cv-lora-row");
  const ltx = await page.evaluate(() => {
    const bbox = CanvasTab.getCompositorContext().boundingBox;
    const workflow = WorkflowBuilder.build({
      model: document.querySelector("#cv-model").value,
      prompt: document.querySelector("#cv-prompt").value,
      negPrompt: document.querySelector("#cv-negative").value,
      width: bbox.width(),
      height: bbox.height(),
      steps: Number(document.querySelector("#cv-steps").value),
      seed: Number(document.querySelector("#cv-seed").value),
      frames: Number(document.querySelector("#cv-frames").value),
      fps: Number(document.querySelector("#cv-fps").value),
      sampler: document.querySelector("#cv-sampler").value,
      scheduler: document.querySelector("#cv-scheduler").value,
      capsPositive: document.querySelector("#cv-caps-positive").value,
      capsNegative: document.querySelector("#cv-caps-negative").value,
      noiseFixture: document.querySelector("#cv-noise-fixture").value,
      includeAudio: document.querySelector("#cv-include-audio").checked,
      loras: [{
        name: document.querySelector(".cv-lora-select").value,
        strength: Number(document.querySelector(".cv-lora-strength").value),
      }],
    });
    return SerenityAPI.videoRequestFromWorkflow(workflow);
  });
  assert(ltx.runner === "ltx2_mojo_request", "Canvas did not select the pure-Mojo LTX2 runner");
  assert(ltx.width === 512 && ltx.height === 768 && ltx.frames === 121, "Canvas substituted LTX2 geometry");
  assert(ltx.steps === 20 && ltx.fps === 25 && ltx.seed === 42, "Canvas substituted the authored LTX2 schedule");
  assert(ltx.sampler === "res2s" && ltx.scheduler === "ltx2", "Canvas substituted the LTX2 sampler profile");
  assert(ltx.lora.length === 1 && ltx.lora[0].name === "ltx2_eri2_step3000" && ltx.lora[0].weight === 1,
    "Canvas did not preserve the verified LTX2 LoRA");
  assert(ltx.prompt.includes("vrtlEri2"), "Canvas lost the trained identity trigger");
  checkpoint("LTX2 and LoRA request passed");

  assert(errors.length === 0, `browser errors: ${errors.join(" | ")}`);
  const screenshot = process.env.SERENITY_CANVAS_SCREENSHOT || "/tmp/serenity-canvas-invoke-parity.png";
  await page.screenshot({ path: screenshot, fullPage: true });
  checkpoint("screenshot captured");
  console.log(JSON.stringify({ layout, editLayout, flowedit, inpaint, zimageMaskedLayout, zimageInpaintPreflight, maskedEditScreenshot, samResult, editScreenshot, dyna, styleLayout, stylePreflight, styleScreenshot, transparency, boardState, ltx, screenshot }, null, 2));
  await browser.close();
})().catch((error) => {
  console.error(error.stack || error);
  process.exit(1);
});
