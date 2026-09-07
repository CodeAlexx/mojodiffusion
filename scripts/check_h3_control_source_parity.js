#!/usr/bin/env node
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const repo = path.resolve(__dirname, "..");
const serenityFlowRoot = process.env.SERENITYFLOW_ROOT || "/home/alex/serenityflow";
const sourceCanvas = path.join(serenityFlowRoot, "serenityflow", "canvas");
const targetCanvas = path.join(repo, "serenity-server", "canvas");

function read(file) { return fs.readFileSync(file, "utf8"); }
function load(paths, extras) {
  const context = vm.createContext(Object.assign({ console }, extras || {}));
  for (const file of paths) vm.runInContext(read(file), context, { filename: file });
  return context;
}

// The visible graph and CSS remain source-faithful to SerenityFlow. Only the
// browser/server execution boundary is Mojo-specific.
assert.strictEqual(
  read(path.join(targetCanvas, "css", "h3-control.css")).trimEnd(),
  read(path.join(sourceCanvas, "css", "h3-control.css")).trimEnd(),
  "H3 ControlNet CSS drifted from the SerenityFlow product reference",
);

const source = load([
  path.join(sourceCanvas, "js", "model-utils.js"),
  path.join(sourceCanvas, "js", "workflow-builder.js"),
  path.join(sourceCanvas, "js", "h3-control-workflow.js"),
]);
const target = load([path.join(targetCanvas, "js", "h3-control-workflow.js")]);

// 7 Fast / 20 Quality are the two schedules SerenityFlow admits.
const SOURCE_SCHEDULE_POINTS = 20;

const params = {
  model: "MiniMax-H3/FL2VA/transformer_int8_rowscale/model.safetensors.index.json",
  controlNet: "MiniMax-H3-Fun-Controlnet-Union.safetensors",
  clipName: "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
  videoVae: "minimax_h3_video_vae_fp16.safetensors",
  audioVae: "minimax_h3_audio_vae_fp32.safetensors",
  controls: [
    { controlMedia: "/tmp/motion.mp4", preprocessor: "canny", resizeMode: "crop",
      cannyLow: 90, cannyHigh: 180, strength: 1.1, startPercent: 0, endPercent: 0.8 },
    { controlMedia: "/tmp/depth.mp4", preprocessor: "prepared", resizeMode: "pad",
      cannyLow: 100, cannyHigh: 200, strength: 0.65, startPercent: 0.2, endPercent: 1 },
  ],
  loras: [],
  sourceMedia: "/tmp/source.mp4",
  maskMedia: "/tmp/mask.png",
  invertMask: true,
  prompt: "integrated_multimodal_description: [Shot 1] Test.",
  width: 1344, height: 768, durationSeconds: 5, steps: SOURCE_SCHEDULE_POINTS - 1, seed: 43,
  outputFormat: "mp4", window: 3, prefetch: 2, videoShift: 12, audioShift: 3,
  filenamePrefix: "source_parity",
};

// SerenityFlow's `steps` counts SIGMA POINTS; the dedicated native screen
// exposes model CALLS instead, emits `calls + 1` sigma points into the graph,
// and api.js lowers that back with -1 -- so the number the operator types is
// the number of model evaluations they get.  That is the one intentional
// divergence.  Feeding each builder the input that denotes the SAME schedule
// keeps this an exact, byte-for-byte comparison of the whole visual graph.
const expected = JSON.parse(JSON.stringify(source.H3ControlWorkflow.build(
  Object.assign({}, params, { steps: SOURCE_SCHEDULE_POINTS }))));
const actual = JSON.parse(JSON.stringify(target.H3ControlWorkflow.build(params)));
assert.deepStrictEqual(actual, expected, "native UI graph differs from SerenityFlow reference graph");
const roundTrip = JSON.parse(JSON.stringify(target.H3ControlWorkflow.parse(actual)));
assert.strictEqual(roundTrip.controls.length, 2);
assert.strictEqual(roundTrip.sourceMedia, "/tmp/source.mp4");
assert.strictEqual(roundTrip.maskMedia, "/tmp/mask.png");

// Load the shared native API without opening a browser connection. The H3
// graph must lower to /v1/video data, not fall through to /prompt or /sf.
const api = load([path.join(targetCanvas, "js", "api.js")], {
  fetch: function () { throw new Error("static lowerer test must not fetch"); },
  FormData: function () {},
  SerenityWS: { getClientId: function () { return "gate"; } },
});
const lowered = JSON.parse(JSON.stringify(api.SerenityAPI.videoRequestFromWorkflow(actual)));
assert.strictEqual(lowered.model, "minimax_h3");
assert.strictEqual(lowered.runner, "minimax_h3_mojo_request");
assert.strictEqual(lowered.task, "controlnet");
assert.strictEqual(lowered.controlnet, params.controlNet);
assert.strictEqual(lowered.controls.length, 2);
assert.strictEqual(lowered.controls[0].path, params.controls[0].controlMedia);
assert.strictEqual(lowered.controls[0].source_path, params.sourceMedia);
assert.strictEqual(lowered.controls[0].mask_path, params.maskMedia);
assert.strictEqual(lowered.steps, params.steps);
assert.strictEqual(lowered.frames, 120);
assert.strictEqual(lowered.fps, 24);
assert.strictEqual(lowered.step_cache, "exact");

const relevant = [
  "js/h3-control.js", "js/h3-control-bridge.js", "js/h3-control-workflow.js",
  "js/api.js", "index.html",
].map(function (name) { return read(path.join(targetCanvas, name)); }).join("\n");
assert(!relevant.includes("/sf/"), "native H3 ControlNet UI still contains a SerenityFlow route");
assert(!relevant.includes("fetch('/sf"), "native H3 ControlNet UI still fetches SerenityFlow");
assert(read(path.join(targetCanvas, "js", "h3-control-bridge.js"))
  .includes("SerenityAPI.postPrompt(workflow, metadata)"));

const mojoRunner = read(path.join(repo, "serenitymojo", "pipeline", "minimax_h3_t2va.mojo"));
const mojoControl = read(path.join(repo, "serenitymojo", "models", "dit", "minimax_h3_controlnet.mojo"));
assert(mojoRunner.includes("--controlnet=") && mojoRunner.includes("minimax_h3_control_inject_active"));
assert(mojoControl.includes("control_blocks.") && mojoControl.includes("index * 10"));

console.log(JSON.stringify({
  status: "PASS",
  uiReferenceGraph: "SerenityFlow",
  executionStack: "Mojo",
  task: lowered.task,
  controls: lowered.controls.length,
  steps: lowered.steps,
  outputFrames: lowered.frames,
  serenityFlowRoutes: 0,
}, null, 2));
