#!/usr/bin/env node
"use strict";

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const root = path.resolve(__dirname, '..');
const attentionPath = path.join(root, 'serenity-server/canvas/js/h3-attention.js');
const projectPath = path.join(root, 'serenity-server/canvas/js/h3-project.js');
const studioPath = path.join(root, 'serenity-server/canvas/js/h3-studio.js');
const indexPath = path.join(root, 'serenity-server/canvas/index.html');
const cssPath = path.join(root, 'serenity-server/canvas/css/h3-studio.css');

function must(condition, message) {
    if (!condition) throw new Error(message);
}

const context = {};
vm.createContext(context);
vm.runInContext(fs.readFileSync(attentionPath, 'utf8'), context, { filename: attentionPath });
vm.runInContext(fs.readFileSync(projectPath, 'utf8'), context, { filename: projectPath });
const A = context.H3AttentionContracts;
const C = context.H3ProjectContracts;
must(A, 'H3AttentionContracts did not load');
must(C, 'H3ProjectContracts did not load');

const portableOnly = [
    { id: 'ck-int8', label: 'CK INT8 · GPU-tuned', available: false,
        quant_modes: ['int8-fast', 'int8', 'bf16'] },
    { id: 'cudnn', label: 'cU-DNN', available: true,
        quant_modes: ['int8-fast', 'int8', 'bf16'] },
    { id: 'sage-int8', label: 'Sage INT8', available: true,
        quant_modes: ['int8-fast', 'int8'] },
];
must(A.resolveBackend('bf16', 'ck-int8', portableOnly) === 'cudnn',
    'unavailable CK must fall back to cU-DNN');
must(A.resolveBackend('bf16', 'sage-int8', portableOnly) === 'cudnn',
    'BF16 Sage must fall back to cU-DNN');
portableOnly[0].available = true;
must(A.resolveBackend('bf16', 'ck-int8', portableOnly) === 'ck-int8',
    'an exact-SM admitted CK backend must remain selectable');

const project = C.createProject();
must(project.schema === 'serenity.h3.movie.v1', 'movie schema drift');
must(project.project_kind === 1 && project.target_duration_seconds === 120, 'movie-first defaults drift');
must(C.ACTIONS.length === 11, 'Director must expose eleven workflows');
must(C.ACTIONS.some((item) => item.id === 'character_sheet'), 'Character Sheet workflow missing');

const shot = project.shots[0];
shot.brief = 'A woman enters a quiet station, hears a train, and looks toward camera.';
shot.shot_description = 'A locked wide shot becomes a slow shoulder-level push. <d>[English] S1: I knew you would come.</d>';
must(C.detectMode(shot) === 't2va', 'T2VA detection failed');
let prompt = C.compilePrompt(shot);
must(prompt.indexOf('integrated_multimodal_description: [Shot 1]') === 0, 'base prompt start drift');
must(!C.promptComplianceIssue(prompt, 't2va', 8), 'T2VA prompt compliance failed');

shot.first_frame = '/srv/uploads/open.png';
must(C.detectMode(shot) === 'i2va', 'I2VA detection failed');
must(C.compilePrompt(shot).indexOf('at 0.00 seconds') >= 0, 'I2VA alignment missing');
shot.last_frame = '/srv/uploads/end.png';
must(C.detectMode(shot) === 'fl2va', 'FL2VA detection failed');
must(C.compilePrompt(shot).indexOf('8.00-second mark') >= 0, 'FL2VA endpoint missing');
shot.first_frame = '';
must(C.detectMode(shot) === 'l2va', 'L2VA detection failed');
shot.last_frame = '';

shot.references.push(Object.assign(C.createReference('image', '/srv/uploads/person.png'), {
    note: 'the same adult face, red wool coat, brass buttons, and black leather gloves'
}));
shot.references.push(Object.assign(C.createReference('video', '/srv/uploads/source.mp4'), {
    role: 'source_video', duration_seconds: 8, note: 'the source blocking, cuts, camera, light, and unchanged people'
}));
must(C.detectMode(shot) === 'ref2va', 'Ref2VA detection failed');
prompt = C.compilePrompt(shot);
must(prompt.indexOf('subject_definitions:') === 0, 'Ref2VA prompt start drift');
must(!C.promptComplianceIssue(prompt, 'ref2va', 8), 'Ref2VA six-section compliance failed');
let render = C.renderRequest(shot);
must(render.model === 'minimax_h3' && render.runner === 'minimax_h3_mojo_request', 'production H3 route drift');
must(render.task === 'ref2va' && render.include_audio === true, 'Ref2VA render request drift');
must(render.fps === 24 && render.duration_seconds === 8, 'H3 native timing drift');
shot.quant = 'bf16';
shot.attention_backend = 'ck-int8';
render = C.renderRequest(shot);
must(render.attention_backend === 'ck-int8', 'BF16 must preserve CK attention');
shot.attention_backend = 'sage-int8';
render = C.renderRequest(shot);
must(render.attention_backend === 'cudnn', 'BF16 must reject Sage attention');
shot.quant = 'int8-fast';
shot.attention_backend = 'ck-int8';

project.director_brief = 'A two-minute odd-couple love story. Their meetings recur as visual comedy; in the final scene they simply click.';
let dream = C.captionRequest(project, shot, 'dream_project');
must(dream.schema === 'serenity.h3.caption.v2', 'caption schema drift');
must(dream.minimum_shots === 8 && dream.maximum_shots === 24, '120-second shot envelope failed');
must(dream.generation_count_rule.indexOf('sum of takes') >= 0, 'generation count rule missing');
must(dream.system_prompt.indexOf('5–15 second') >= 0, 'bounded H3 shot rule missing');
must(dream.system_prompt.indexOf('Do not propose LoRAs') >= 0, 'no-LoRA prompt rule missing');

shot.references = [
    Object.assign(C.createReference('image', '/srv/uploads/face.png'), { note: 'face geometry and dark curly hair; ignore background and pose' }),
    Object.assign(C.createReference('image', '/srv/uploads/wardrobe.png'), { note: 'blue linen jacket with horn buttons; ignore the model face' })
];
let sheetRequest = C.captionRequest(project, shot, 'character_sheet', { panel_count: 6, style_mode: 'standard_orbit' });
let sheet = sheetRequest.character_sheet;
must(sheet.panel_count === 6 && sheet.internal_frames === 124, 'canonical six-panel geometry drift');
must(JSON.stringify(sheet.frame_indices) === JSON.stringify([2, 21, 42, 63, 84, 113]), 'six-panel extraction drift');
must(sheet.render_admitted === true && sheet.plan_status.indexOf('canonical') === 0, 'six-panel admission drift');
must(sheet.source_revision === 'ccc9d411b6b7056b43edf2690503e063560a5acd', 'Character Sheet provenance drift');
must((sheet.a_prompt.match(/<Picture /g) || []).length === 2, 'A prompt must contain one row per supplied picture');
must(!C.promptComplianceIssue(sheet.h3_prompt, 'ref2va', 5), 'Character Sheet Ref2VA compliance failed');

sheetRequest = C.captionRequest(project, shot, 'character_sheet', { panel_count: 4, style_mode: 'anime_to_real' });
sheet = sheetRequest.character_sheet;
must(sheet.internal_frames === 73 && sheet.render_admitted === false, 'four-panel must remain planning-only');
must(JSON.stringify(sheet.frame_indices) === JSON.stringify([2, 24, 45, 68]), 'four-panel extraction drift');
must(sheet.warning.indexOf('below Serenity’s 5-second') >= 0 && sheet.warning.indexOf('not render-admitted') >= 0, 'four-panel warning missing');

shot.references.push(C.createReference('audio', '/srv/uploads/voice.wav'));
must(C.validateDirector(project, shot, 'character_sheet').indexOf('image references only') >= 0, 'Character Sheet non-image rejection failed');
shot.references = [];
must(C.validateDirector(project, shot, 'character_sheet').indexOf('between 1 and 9') >= 0, 'Character Sheet empty rejection failed');
for (let i = 0; i < 10; i++) shot.references.push(C.createReference('image', '/srv/uploads/ref-' + i + '.png'));
must(C.validateDirector(project, shot, 'character_sheet').indexOf('between 1 and 9') >= 0, 'Character Sheet ten-image rejection failed');

const roundTrip = C.normalizeProject(JSON.parse(JSON.stringify(project)));
must(roundTrip.schema === project.schema && roundTrip.shots.length === project.shots.length, 'project round-trip failed');
const edit = C.deliveryManifest(project);
must(edit.schema === 'serenity.h3.edit.v1' && edit.source_fps === 24, 'edit manifest drift');
const endlessFrames = Array.from(C.planEndlessFrames(31, 15));
must(JSON.stringify(endlessFrames) === JSON.stringify([360, 264, 120]), 'Endless exact 24-fps frame plan drift');
must(endlessFrames.reduce((sum, frames) => sum + frames, 0) === 31 * 24, 'Endless plan does not cover the target exactly');

const index = fs.readFileSync(indexPath, 'utf8');
const studio = fs.readFileSync(studioPath, 'utf8');
const css = fs.readFileSync(cssPath, 'utf8');
must(index.includes('data-tab="h3-studio"') && index.includes('id="panel-h3-studio"'), 'H3 Studio navigation not mounted');
must(index.includes('js/h3-attention.js') && index.includes('js/h3-project.js') && index.includes('js/h3-studio.js') && index.includes('css/h3-studio.css'), 'H3 Studio assets not loaded');
must(studio.includes("SerenityAPI.postVideo(request)"), 'H3 render is not wired to production /v1/video API');
must(studio.includes('resolvedH3Attention(shot)'), 'H3 Studio does not consume shared attention admission');
must(!studio.includes('CK INT8 · fastest'), 'H3 Studio still makes a universal fastest claim');
must(studio.includes("window.confirm('Queue one "), 'GPU render must require explicit confirmation');
must(studio.includes('opening Studio never starts GPU work'), 'no-auto-generation status missing');
must(studio.includes('Start endless story') && studio.includes('Stop after current'), 'Endless explicit controls missing');
must(studio.includes('C.videoJobUrls') && studio.includes('C.endlessSnapshotsEqual'), 'Endless safe resume/URL reconstruction missing');
must(studio.includes('browser never rebuilds model state or starts training'), 'Endless inference-only contract missing');
must(studio.includes('Continuity spine') && studio.includes('Prepare Qwen Director pass'), 'filmmaker workspace surfaces missing');
must(css.includes('.h3s-workspace') && css.includes('.h3s-spine-shot') && css.includes('@media (prefers-reduced-motion: reduce)'), 'film layout/accessibility CSS missing');
const shell = fs.readFileSync(path.join(root, 'serenity-server/canvas/js/shell.js'), 'utf8');
must(shell.includes("new URLSearchParams(window.location.search).get('tab')"), 'direct H3 Studio deep link missing');

console.log('PASS: production H3 web Studio project, prompt, render, Director, Character Sheet, persistence, and UI mount contracts');
