"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const context = { console };
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(__dirname, "h3-project.js"), "utf8"), context, { filename: "h3-project.js" });
const C = context.H3ProjectContracts;

assert.deepEqual(Array.from(C.planEndlessFrames(31, 15)), [360, 264, 120]);
assert.deepEqual(Array.from(C.planEndlessFrames(16, 15)), [264, 120]);
assert.deepEqual(Array.from(C.planEndlessFrames(5, 5)), [120]);
assert.deepEqual(Array.from(C.planEndlessFrames(5.1, 5)), [122]);
assert.deepEqual(Array.from(C.planEndlessDurations(5.1, 5)), [122 / 24]);
assert.match(C.validateEndlessSettings(4, 10, "continue"), /between 5 seconds/);
assert.match(C.validateEndlessSettings(30, 16, "continue"), /between 5 and 15/);
assert.match(C.validateEndlessSettings(30, 10, ""), /explicit continuation direction/);

let random = 0x51e7a11;
function unit() { random = (Math.imul(random, 1664525) + 1013904223) >>> 0; return random / 0x100000000; }
for (let i = 0; i < 5000; i++) {
    const target = 5 + unit() * 3595;
    const preferred = 5 + unit() * 10;
    const plan = Array.from(C.planEndlessFrames(target, preferred));
    assert.equal(plan.reduce((sum, value) => sum + value, 0), Math.round(target * 24));
    assert(plan.every((value) => Number.isInteger(value) && value >= 120 && value <= 360));
    assert(plan.every((value) => Math.round((value / 24) * 24) === value));
}

const project = C.createProject();
const shot = project.shots[0];
shot.id = 7; project.next_shot_id = 8; shot.title = "Car chase";
shot.duration_seconds = 8; shot.seed = 900;
shot.brief = "Two red pursuit cars race through a wet tunnel.";
shot.shot_description = "The two red cars hold formation as the camera tracks low behind them.";
shot.references = [Object.assign(C.createReference("image", "/uploads/red-car.png"), { note: "the same red pursuit car, decals, body geometry, and wet paint" })];
const direction = "Continue the same two-car chase, preserving both red cars, tunnel direction, camera motion, and engine audio.";
const run = C.createEndlessRun(shot, 31, 15, direction);

assert.equal(run.schema, C.ENDLESS_SCHEMA);
assert.equal(run.base_shot_id, 7);
assert.equal(run.target_frames, 744);
assert.deepEqual(Array.from(run.segment_output_frames), [360, 264, 120]);
assert.match(run.fingerprint, /^fnv1a32-[0-9a-f]{8}$/);

const firstRequest = C.renderRequest(C.endlessSegmentShot(run, 0));
assert.equal(firstRequest.task, "ref2va");
assert.equal(firstRequest.output_frames, 360);
assert.equal(firstRequest.duration_seconds, 15);
assert.equal(firstRequest.seed, 900);
assert.equal(firstRequest.continue_from, undefined);

run.completed_job_ids.push("video-0042");
const secondRequest = C.renderRequest(C.endlessSegmentShot(run, 1));
assert.equal(secondRequest.task, "continue");
assert.equal(secondRequest.output_frames, 264);
assert.equal(secondRequest.duration_seconds, 11);
assert.equal(secondRequest.seed, 901);
assert.equal(secondRequest.continue_from, "video-0042");
assert.equal(secondRequest.motion_context_frames, 22);
assert.equal(secondRequest.trim_start_frames, 22);
assert.equal(JSON.stringify(secondRequest.references), JSON.stringify(firstRequest.references));
assert.match(secondRequest.prompt, /^subject_definitions:/);
assert.match(secondRequest.prompt, /Continue the same two-car chase/);

const sameSnapshot = C.endlessBaseSnapshot(shot, run.target_seconds, run.segment_seconds, run.continuation_direction);
assert(C.endlessSnapshotsEqual(sameSnapshot, run.base_snapshot));
shot.id = 99; shot.title = "Display rename"; shot.duration_seconds = 15;
const bookkeepingSnapshot = C.endlessBaseSnapshot(shot, run.target_seconds, run.segment_seconds, run.continuation_direction);
assert(C.endlessSnapshotsEqual(bookkeepingSnapshot, run.base_snapshot));
shot.steps += 1;
const changedSnapshot = C.endlessBaseSnapshot(shot, run.target_seconds, run.segment_seconds, run.continuation_direction);
assert(!C.endlessSnapshotsEqual(changedSnapshot, run.base_snapshot));
const fakeSameFingerprint = run.fingerprint;
assert.equal(fakeSameFingerprint, run.fingerprint);
assert(!C.endlessSnapshotsEqual(changedSnapshot, run.base_snapshot), "full canonical equality, not FNV32, must gate resume");
shot.steps -= 1;

assert.equal(JSON.stringify(C.videoJobUrls("video-0042")), JSON.stringify({ video_id: "video-0042", status_url: "/out/video-0042/status.json", result_url: "/out/video-0042/result.json" }));
assert.equal(C.videoJobUrls("https://evil.invalid/video-1"), null);
const imported = C.createProject();
imported.endless = JSON.parse(JSON.stringify(run));
imported.endless.status = "running";
imported.endless.active_job = { video_id: "../../evil", segment_index: 1, status_url: "https://evil.invalid/x" };
const rejected = C.normalizeProject(JSON.parse(JSON.stringify(imported)));
assert.equal(rejected.endless.status, "failed");
assert.equal(rejected.endless.active_job, null);

const fractionImport = C.createProject();
fractionImport.endless.target_seconds = 5.1;
fractionImport.endless.segment_seconds = 5.1;
delete fractionImport.endless.target_frames;
delete fractionImport.endless.segment_frames;
const aligned = C.normalizeProject(JSON.parse(JSON.stringify(fractionImport))).endless;
assert.equal(aligned.target_frames, 122);
assert.equal(aligned.target_seconds, 122 / 24);

const chainProject = C.createProject();
const chainBase = chainProject.shots[0];
chainBase.brief = shot.brief; chainBase.shot_description = shot.shot_description;
chainBase.references = shot.references.map(C.copy); chainBase.seed = 900;
const chain = C.createEndlessRun(chainBase, 31, 15, direction);
C.recordEndlessSegmentTake(chainProject, chain, 0, "video-0100", "/out/video-0100/final.mp4");
chain.completed_job_ids.push("video-0100");
chain.completed_output_paths.push("/out/video-0100/final.mp4");
C.recordEndlessSegmentTake(chainProject, chain, 1, "video-0101", "/out/video-0101/final.mp4");
chain.completed_job_ids.push("video-0101");
chain.completed_output_paths.push("/out/video-0101/final.mp4");
assert.equal(chainProject.shots.length, 2);
assert.equal(chainProject.shots[0].duration_seconds * 24, 360);
assert.equal(chainProject.shots[1].duration_seconds * 24, 264);
assert.equal(chainProject.shots[1].continue_from, "video-0100");
assert.equal(JSON.stringify(chainProject.shots[1].references), JSON.stringify(chainBase.references));
assert.equal(chainProject.shots[1].seed, 901);
assert.equal(chainProject.shots[1].status, "Ready");
assert.equal(chainProject.shots[1].selected_take, 0);
assert.equal(JSON.stringify(chain.segment_shot_ids), JSON.stringify(Array.from(chainProject.shots, (item) => item.id)));
C.recordEndlessSegmentTake(chainProject, chain, 1, "video-0101", "/out/video-0101/final.mp4");
assert.equal(chainProject.shots.length, 2, "replay must update, not duplicate, a completed segment shot");
assert.equal(JSON.stringify(Array.from(C.deliveryManifest(chainProject).shots, (item) => item.job_id)), JSON.stringify(["video-0100", "video-0101"]));
chainProject.endless = chain;
const chainRoundTrip = C.normalizeProject(JSON.parse(JSON.stringify(chainProject)));
assert.equal(chainRoundTrip.endless.status, "running");
chainRoundTrip.shots.splice(1, 1);
assert.equal(C.normalizeProject(JSON.parse(JSON.stringify(chainRoundTrip))).endless.status, "failed");

const badDirectionShot = C.createShot(1, "Unsafe prompt");
badDirectionShot.brief = "A valid opening.";
badDirectionShot.references = [C.createReference("image", "/uploads/ref.png")];
assert.throws(() => C.createEndlessRun(badDirectionShot, 20, 10, "subject_definitions: duplicate marker"), /canonical section/i);

console.log("h3 endless frame planning, resume safety, URLs, references, and edit chain: ok");
