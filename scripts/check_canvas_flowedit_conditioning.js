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
      try { return require(path.join(npxRoot, entry, "node_modules", "playwright")).chromium; } catch (_) {}
    }
  }
  return require(path.join(os.tmpdir(), "mojodiffusion-playwright-tools", "node_modules", "playwright")).chromium;
}

function assert(value, message) {
  if (!value) throw new Error(message);
}

(async () => {
  const chromium = loadChromium();
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
    await page.goto(process.env.SERENITY_URL || "http://127.0.0.1:7811", { waitUntil: "domcontentloaded" });
    await page.click('.nav-btn[data-tab="canvas"]');
    await page.waitForSelector("#cv-edit-mode");
    await page.waitForSelector('#cv-model option[value="krea2-turbo"]', { state: "attached" });

    let captionRequest = null;
    let promptBody = null;
    let captionMode = "flowedit";
    await page.route("**/v1/caption", async (route) => {
      captionRequest = route.request().postDataJSON();
      const caption = captionMode === "style" && captionRequest.prompt.includes('"medium"') ? {
        medium: "anime digital illustration",
        rendering: "polished painting with smooth gradients",
        palette: "vibrant pastels and cherry-blossom pink accents",
        line_work: "fine clean line work",
        lighting: "soft glowing light",
        texture: "silky detailed fabric and hair",
        finish: "glossy dreamy finish with floating petals",
      } : captionMode === "style" ? {
        source: "A dark-haired woman in a beige trench coat walks through a crowded city street at golden hour, surrounded by pedestrians, preserving the full-body pose, camera angle, square crop, and urban background.",
        style: "vibrant polished anime illustration with fine line work, smooth gradients, soft glowing light, cherry-blossom pink accents, and glossy digital-painting finish",
      } : {
        source: "A blonde woman in a fur hood stands beside a rain-covered window at dusk, preserving her face, pose, clothing, framing, and warm indoor light.",
        target: "The same blonde woman in the same fur hood, face, pose, clothing, and framing stands beside a clear sunlit window on a bright sunny day with warm daylight outside.",
      };
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ caption: JSON.stringify(caption) }),
      });
    });
    await page.route("**/prompt", async (route) => {
      promptBody = route.request().postDataJSON();
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ prompt_id: "flowedit-conditioning-check" }),
      });
    });

    await page.selectOption("#cv-edit-mode", "flowedit");
    await page.dispatchEvent("#cv-edit-mode", "change");
    await page.setInputFiles(
      "#cv-import-file",
      path.join(process.cwd(), "output/serenity_ui_out/uploads/flowedit_source-0085.png")
    );
    await page.waitForFunction(() => {
      const preview = document.querySelector("#cv-source-preview");
      return preview && preview.complete && preview.naturalWidth > 0;
    });
    await page.selectOption("#cv-edit-engine", "krea2_turbo_1024");
    await page.dispatchEvent("#cv-edit-engine", "change");
    const turboProfile = await page.evaluate(() => ({
      steps: Number(document.querySelector("#cv-steps").value),
      cfg: Number(document.querySelector("#cv-cfg").value),
      nmax: Number(document.querySelector("#cv-edit-nmax").value),
      srcCfg: Number(document.querySelector("#cv-edit-src-cfg").value),
      warmup: Number(document.querySelector("#cv-edit-mask-warmup").value),
    }));
    assert(turboProfile.steps === 8 && turboProfile.cfg === 0 &&
      turboProfile.nmax === 8 && turboProfile.srcCfg === 0 && turboProfile.warmup === 1,
      `Turbo did not expose its native fast profile: ${JSON.stringify(turboProfile)}`);
    await page.setInputFiles(
      "#ref-file-input",
      path.join(process.cwd(), "output/serenity_ui_out/job-0094.png")
    );
    await page.waitForFunction(() => document.querySelector("#cv-edit-mode").value === "style");
    let selectorState = await page.evaluate(() => ({
      engine: document.querySelector("#cv-edit-engine").value,
      model: document.querySelector("#cv-model").value,
      modelRowVisible: getComputedStyle(document.querySelector("#cv-model-row")).display !== "none",
      badge: document.querySelector(".model-badge").textContent,
    }));
    assert(selectorState.engine === "krea2_turbo_1024",
      `Style mode reverted the visible engine to ${selectorState.engine}`);
    assert(/turbo/i.test(selectorState.model) && /turbo/i.test(selectorState.badge),
      `Style mode did not synchronize its underlying model and badge to Turbo: ${JSON.stringify(selectorState)}`);
    assert(!selectorState.modelRowVisible,
      "Style mode still exposes the contradictory create-model selector");

    await page.selectOption("#cv-edit-mode", "flowedit");
    await page.dispatchEvent("#cv-edit-mode", "change");
    selectorState = await page.evaluate(() => ({
      engine: document.querySelector("#cv-edit-engine").value,
      model: document.querySelector("#cv-model").value,
      modelRowVisible: getComputedStyle(document.querySelector("#cv-model-row")).display !== "none",
    }));
    assert(selectorState.engine === "krea2_turbo_1024" && /turbo/i.test(selectorState.model),
      "FlowEdit mode did not preserve the Turbo selection from Style mode");
    assert(!selectorState.modelRowVisible,
      "FlowEdit still exposes the contradictory create-model selector");

    // Reload before the submission check so staged Style UI state cannot mask
    // the independent paired-conditioning request contract.
    await page.goto(process.env.SERENITY_URL || "http://127.0.0.1:7811", { waitUntil: "domcontentloaded" });
    await page.click('.nav-btn[data-tab="canvas"]');
    await page.waitForSelector('#cv-model option[value="krea2-turbo"]', { state: "attached" });
    await page.selectOption("#cv-edit-mode", "flowedit");
    await page.dispatchEvent("#cv-edit-mode", "change");
    await page.setInputFiles(
      "#cv-import-file",
      path.join(process.cwd(), "output/serenity_ui_out/uploads/flowedit_source-0085.png")
    );
    await page.waitForFunction(() => {
      const preview = document.querySelector("#cv-source-preview");
      return preview && preview.complete && preview.naturalWidth > 0;
    });
    await page.selectOption("#cv-edit-engine", "krea2_turbo_1024");
    await page.dispatchEvent("#cv-edit-engine", "change");
    await page.fill("#cv-edit-source-prompt", "");
    await page.fill("#cv-prompt", "change background to a sunny day");

    const submitted = page.waitForResponse((response) => response.url().endsWith("/prompt"));
    await page.click("#cv-generate-btn");
    await submitted;

    assert(captionRequest, "FlowEdit did not ask the vision captioner for a source/target pair");
    assert(captionRequest.prompt.includes("change background to a sunny day"),
      "The requested edit did not reach target-caption synthesis");
    assert(captionRequest.prompt.includes('"source"') && captionRequest.prompt.includes('"target"'),
      "FlowEdit did not request paired source and target descriptions");
    assert(promptBody && promptBody.prompt, "FlowEdit did not submit a graph workflow");

    const workflow = promptBody.prompt;
    assert(workflow["1"].inputs.unet_name === "krea2_turbo.safetensors",
      "The visible Turbo choice did not select the Turbo checkpoint");
    assert(workflow["8"].inputs.width === 1024 && workflow["8"].inputs.height === 1024,
      "The Turbo workflow did not preserve the visible 1024x1024 profile");
    assert(workflow["8"].inputs.steps === 8 && workflow["8"].inputs.nmax === 8 &&
      workflow["8"].inputs.src_cfg === 0 && workflow["8"].inputs.tgt_cfg === 0,
      "The Turbo workflow did not preserve the visible native fast profile");
    assert(workflow["4"].inputs.text.startsWith("A blonde woman"),
      "The generated source description did not reach source conditioning");
    assert(workflow["6"].inputs.text.startsWith("The same blonde woman"),
      "The complete generated target description did not reach target conditioning");
    assert(workflow["6"].inputs.text !== "change background to a sunny day",
      "The bare edit instruction was incorrectly sent as target conditioning");

    captionMode = "style";
    captionRequest = null;
    promptBody = null;
    await page.goto(process.env.SERENITY_URL || "http://127.0.0.1:7811", { waitUntil: "domcontentloaded" });
    await page.click('.nav-btn[data-tab="canvas"]');
    await page.waitForSelector("#cv-edit-mode");
    await page.setInputFiles(
      "#cv-import-file",
      path.join(process.cwd(), "output/serenity_ui_out/uploads/style_source-0151.png")
    );
    await page.waitForFunction(() => {
      const preview = document.querySelector("#cv-source-preview");
      return preview && preview.complete && preview.naturalWidth === 1024;
    });
    await page.setInputFiles(
      "#ref-file-input",
      path.join(process.cwd(), "output/serenity_ui_out/uploads/style_reference-0152.png")
    );
    await page.waitForFunction(() => document.querySelector("#cv-edit-mode").value === "style");
    await page.selectOption("#cv-edit-engine", "krea2_turbo_1024");
    await page.dispatchEvent("#cv-edit-engine", "change");
    await page.fill("#cv-edit-source-prompt", "A dark-haired woman in a beige trench coat walks through a crowded city street at golden hour, surrounded by pedestrians, preserving the full-body pose, camera angle, square crop, and urban background.");
    await page.fill("#cv-prompt", "");

    const styleSubmitted = page.waitForResponse((response) => response.url().endsWith("/prompt"));
    await page.click("#cv-generate-btn");
    await styleSubmitted;

    assert(captionRequest && captionRequest.prompt.includes('"medium"') &&
      captionRequest.prompt.includes("Never describe the subject"),
      "Style mode did not request a strict rendering-only style description");
    assert(promptBody && promptBody.prompt, "Style mode did not submit a graph workflow");
    const styleWorkflow = promptBody.prompt;
    const styleSource = styleWorkflow["4"].inputs.text;
    const styleTarget = styleWorkflow["6"].inputs.text;
    assert(styleTarget.startsWith(styleSource.replace(/[.\\s]+$/, "")),
      "Style target does not begin with the complete source description");
    assert(styleTarget.includes("dark-haired woman in a beige trench coat") &&
      styleTarget.includes("crowded city street") && styleTarget.includes("full-body pose"),
      "Style target lost source subject, clothing, pose, or scene anchors");
    assert(styleTarget.includes("anime digital illustration") &&
      styleTarget.includes("cherry-blossom pink accents"),
      "Style target lost the analyzed rendering style");
    assert(!styleTarget.includes("shot type") && !styleTarget.includes("full-body portrait"),
      "Style target copied subject or composition language from the style reference");
    assert(styleTarget.split(/\s+/).length <= 160,
      "Style target can overflow Krea FlowEdit's compiled 256-token text bucket");
    assert(styleWorkflow["8"].inputs.auto_mask === false,
      "Entire-image style mode did not disable the localized mask");

    process.stdout.write("canvas FlowEdit + source-anchored entire-image Style + Krea Turbo 1024: PASS\n");
  } finally {
    await browser.close();
  }
})().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exit(1);
});
