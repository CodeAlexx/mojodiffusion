"use strict";
/**
 * H3StudioTab — project-based MiniMax H3 movie/commercial workspace.
 * Rendering is explicit and uses the existing production /v1/video route.
 */
var H3StudioTab = (function () {
    'use strict';

    var C = H3ProjectContracts;
    var STORAGE_KEY = 'serenity-h3-current-project-v1';
    var state = {
        initialized: false,
        project: null,
        selectedShotId: 1,
        stageTab: 'director',
        inspectorTab: 'shot',
        bibleTab: 'director_brief',
        directorAction: 'dream_project',
        characterPanels: 6,
        characterStyle: 'standard_orbit',
        requestJson: '',
        directorRunning: false,
        readiness: null,
        status: 'Ready · opening Studio never starts GPU work',
        statusTone: '',
        videoPollToken: 0,
        endlessPollToken: 0,
        endlessSubmitting: false,
        modal: ''
    };

    function escapeHtml(value) {
        return String(value == null ? '' : value)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;').replace(/'/g, '&#039;');
    }
    function attr(value) { return escapeHtml(value).replace(/\n/g, '&#10;'); }
    function checked(value) { return value ? ' checked' : ''; }
    function selected(value, expected) { return String(value) === String(expected) ? ' selected' : ''; }
    function disabled(value) { return value ? ' disabled' : ''; }

    function loadProject() {
        try {
            var raw = localStorage.getItem(STORAGE_KEY);
            if (raw) return C.normalizeProject(JSON.parse(raw));
        } catch (error) {
            console.warn('[H3Studio] project restore failed:', error);
        }
        return C.createProject();
    }

    function saveProject(message) {
        state.project.updated_at = new Date().toISOString();
        localStorage.setItem(STORAGE_KEY, JSON.stringify(state.project));
        if (message) setStatus(message, '');
    }

    function showToast(message, tone) {
        var node = document.getElementById('h3s-toast');
        if (!node) {
            node = document.createElement('div'); node.id = 'h3s-toast';
            node.style.cssText = 'position:fixed;top:14px;left:50%;transform:translateX(-50%);z-index:9999;max-width:720px;padding:10px 16px;border-radius:8px;font:13px/1.4 system-ui,sans-serif;box-shadow:0 6px 24px rgba(0,0,0,.45);pointer-events:none;transition:opacity .3s';
            document.body.appendChild(node);
        }
        node.textContent = message;
        node.style.background = tone === 'error' ? '#7a1f1f' : (tone === 'live' ? '#1f4d7a' : '#1f5a2e');
        node.style.color = '#fff'; node.style.opacity = '1';
        clearTimeout(showToast._timer);
        showToast._timer = setTimeout(function () { node.style.opacity = '0'; }, tone === 'error' ? 9000 : 5000);
    }

    function setStatus(message, tone) {
        state.status = message;
        state.statusTone = tone || '';
        var node = document.getElementById('h3s-status-message');
        if (node) {
            node.textContent = state.status;
            node.className = tone === 'error' ? 'is-live h3s-status-error' : (tone === 'live' ? 'is-live' : '');
        }
    }

    function selectedShotIndex() {
        var index = state.project.shots.findIndex(function (shot) { return shot.id === state.selectedShotId; });
        return index >= 0 ? index : 0;
    }
    function selectedShot() { return state.project.shots[selectedShotIndex()]; }
    function totalSeconds() {
        return state.project.shots.reduce(function (total, shot) { return total + Number(shot.duration_seconds || 0); }, 0);
    }
    function timecode(seconds) {
        var frames = Math.round(Number(seconds || 0) * C.NATIVE_FPS);
        var ff = frames % C.NATIVE_FPS;
        var total = Math.floor(frames / C.NATIVE_FPS);
        var ss = total % 60;
        var mm = Math.floor(total / 60) % 60;
        var hh = Math.floor(total / 3600);
        return [hh, mm, ss, ff].map(function (n) { return String(n).padStart(2, '0'); }).join(':');
    }

    function h3Runner() {
        var runners = state.readiness && Array.isArray(state.readiness.candidate_runners)
            ? state.readiness.candidate_runners : [];
        return runners.find(function (entry) { return entry && entry.model === 'minimax_h3_t2va'; }) || null;
    }
    function h3AttentionBackends() {
        var runner = h3Runner();
        return runner && runner.attention_backends;
    }
    function resolvedH3Attention(shot) {
        return H3AttentionContracts.resolveBackend(
            shot.quant,
            shot.attention_backend,
            h3AttentionBackends()
        );
    }
    function h3Ready() {
        var runner = h3Runner();
        return !!runner && (runner.available === true || [
            'ready', 'quality_profile_ready', 'experimental_request_runner_ready',
            'runtime_geometry_ready', 'conditioned_runtime_geometry_ready'
        ].indexOf(String(runner.status || '')) >= 0);
    }

    function headerHtml() {
        var ready = h3Ready();
        return '<header class="h3s-header">' +
            '<div class="h3s-brand"><div class="h3s-mark">H3</div><div class="h3s-title-stack">' +
            '<div class="h3s-eyebrow">MiniMax filmmaker workspace</div>' +
            '<input id="h3s-project-title" class="h3s-project-title" value="' + attr(state.project.title) + '" aria-label="Project title">' +
            '</div></div>' +
            '<div class="h3s-header-stats">' +
            '<span class="h3s-chip"><strong>' + state.project.shots.length + '</strong> shots</span>' +
            '<span class="h3s-chip"><strong>' + C.secondsText(totalSeconds()) + 's</strong> cut</span>' +
            '<span class="h3s-chip ' + (ready ? 'is-ready' : 'is-blocked') + '">' + (ready ? 'Runtime ready' : 'Runtime unavailable') + '</span>' +
            '</div>' +
            '<div class="h3s-header-actions">' +
            '<button class="h3s-btn is-quiet" data-h3-action="new-project">New</button>' +
            '<button class="h3s-btn is-quiet" data-h3-action="import-project">Import</button>' +
            '<button class="h3s-btn" data-h3-action="export-project">Export project</button>' +
            '<button class="h3s-btn" data-h3-action="export-edit">Export edit</button>' +
            '</div></header>';
    }

    function projectControlsHtml() {
        return '<div class="h3s-section"><div class="h3s-section-head"><span class="h3s-kicker">Project deck</span><span class="h3s-count">24 FPS SOURCE</span></div>' +
            '<label class="h3s-field"><span>Format</span><select class="h3s-select" data-project-field="project_kind">' +
            '<option value="0"' + selected(state.project.project_kind, 0) + '>Single clip</option>' +
            '<option value="1"' + selected(state.project.project_kind, 1) + '>Movie</option>' +
            '<option value="2"' + selected(state.project.project_kind, 2) + '>Brand commercial</option></select></label>' +
            '<div class="h3s-grid-2"><label class="h3s-field"><span>Target runtime</span><input class="h3s-input" type="number" min="5" step="5" data-project-field="target_duration_seconds" value="' + attr(state.project.target_duration_seconds) + '"></label>' +
            '<label class="h3s-field"><span>Takes / shot</span><input class="h3s-input" type="number" min="1" max="8" data-project-field="takes_per_shot" value="' + attr(state.project.takes_per_shot) + '"></label></div>' +
            '<div class="h3s-grid-2"><label class="h3s-field"><span>Delivery FPS</span><select class="h3s-select" data-project-field="delivery_fps"><option' + selected(state.project.delivery_fps, 24) + '>24</option><option' + selected(state.project.delivery_fps, 25) + '>25</option><option' + selected(state.project.delivery_fps, 30) + '>30</option></select></label>' +
            '<label class="h3s-field"><span>Director model</span><select class="h3s-select" data-project-field="caption_tier"><option value="0"' + selected(state.project.caption_tier, 0) + '>Qwen3-VL 8B</option><option value="1"' + selected(state.project.caption_tier, 1) + '>Qwen3-VL 32B</option></select></label></div>' +
            '<label class="h3s-check"><input type="checkbox" data-project-field="adult_mode"' + checked(state.project.adult_mode) + '><span>Lawful consensual adult 18+ captioning. Never minors, coercion, or exploitative non-consent.</span></label></div>';
    }

    function biblesHtml() {
        var tabs = [
            ['director_brief', 'Director'], ['continuity_bible', 'Continuity'], ['brand_bible', 'Brand']
        ];
        var placeholder = state.bibleTab === 'director_brief'
            ? 'Movie outline, script, commercial brief, dialogue, or general story intent…'
            : (state.bibleTab === 'continuity_bible'
                ? 'Identity, wardrobe, props, location, lighting, eyeline, screen direction, motion, dialogue and sound state…'
                : 'Product, logo, visual language, claims, colors, typography, audience and mandatory brand beats…');
        return '<div class="h3s-section"><div class="h3s-section-head"><span class="h3s-kicker">Project bibles</span><span class="h3s-count">AUTOSAVED</span></div>' +
            '<div class="h3s-bible-tabs">' + tabs.map(function (tab) {
                return '<button class="h3s-tab ' + (state.bibleTab === tab[0] ? 'is-active' : '') + '" data-bible-tab="' + tab[0] + '">' + tab[1] + '</button>';
            }).join('') + '</div>' +
            '<label class="h3s-field" style="margin-top:8px"><textarea class="h3s-textarea" data-project-field="' + state.bibleTab + '" placeholder="' + attr(placeholder) + '">' + escapeHtml(state.project[state.bibleTab] || '') + '</textarea></label></div>';
    }

    function shotListHtml() {
        return '<div class="h3s-section"><div class="h3s-section-head"><span class="h3s-kicker">Shot deck</span><span class="h3s-count">' + state.project.shots.length + '</span></div>' +
            '<div class="h3s-shot-list">' + state.project.shots.map(function (shot, index) {
                var mode = C.detectMode(shot);
                return '<button class="h3s-shot-card ' + (shot.id === state.selectedShotId ? 'is-active' : '') + '" data-select-shot="' + shot.id + '">' +
                    '<span class="h3s-shot-number">' + String(index + 1).padStart(2, '0') + '</span><span><span class="h3s-shot-name">' + escapeHtml(shot.title) + '</span>' +
                    '<span class="h3s-shot-meta">' + escapeHtml(mode.toUpperCase()) + ' · ' + C.secondsText(shot.duration_seconds) + 's · ' + escapeHtml(shot.status) + '</span></span>' +
                    (shot.locked ? '<span class="h3s-lock">◆</span>' : '<span></span>') + '</button>';
            }).join('') + '</div>' +
            '<div class="h3s-button-row" style="margin-top:8px"><button class="h3s-btn is-primary" data-h3-action="add-shot">Add shot</button><button class="h3s-btn" data-h3-action="duplicate-shot">Duplicate</button><button class="h3s-btn is-danger" data-h3-action="delete-shot">Delete</button></div>' +
            '<div class="h3s-button-row" style="margin-top:6px"><button class="h3s-btn" data-h3-action="shot-left">← Earlier</button><button class="h3s-btn" data-h3-action="shot-right">Later →</button></div></div>';
    }

    function leftHtml() {
        return '<aside class="h3s-left">' + projectControlsHtml() + biblesHtml() + shotListHtml() + '</aside>';
    }

    function monitorHtml() {
        var shot = selectedShot();
        var mode = C.detectMode(shot);
        var take = shot.selected_take >= 0 ? shot.take_output_paths[shot.selected_take] : '';
        var src = take || shot.output_path || '';
        var content = src
            ? '<video controls preload="metadata" src="' + attr(src) + '"></video>'
            : '<div class="h3s-monitor-empty"><div><div class="h3s-monitor-code">SHOT ' + String(selectedShotIndex() + 1).padStart(2, '0') + ' · ' + mode.toUpperCase() + '</div><div class="h3s-monitor-title">' + escapeHtml(shot.title) + '</div><div class="h3s-monitor-copy">' + escapeHtml(shot.brief || shot.shot_description || 'Write the beat, stage the references, then prepare or render this shot. No GPU work begins until Queue H3 take is confirmed.') + '</div></div></div>';
        return '<div class="h3s-monitor-wrap"><div class="h3s-monitor">' + content + '<div class="h3s-monitor-bars"></div>' +
            '<span class="h3s-safe-corner tl"></span><span class="h3s-safe-corner tr"></span><span class="h3s-safe-corner bl"></span><span class="h3s-safe-corner br"></span>' +
            '<div class="h3s-monitor-hud"><span>' + shot.width + '×' + shot.height + ' · ' + C.secondsText(shot.duration_seconds) + 's</span><span>H3 NATIVE 24 FPS · SYNC AUDIO</span></div></div></div>';
    }

    function briefStageHtml() {
        var shot = selectedShot();
        return '<div class="h3s-stage-copy">Write intent, dialogue, action, emotional turn, and the result the shot must leave behind.</div>' +
            '<textarea class="h3s-textarea is-script" data-shot-field="brief" placeholder="At the first frame… dialogue uses stable speaker IDs and exact &lt;d&gt; tags…">' + escapeHtml(shot.brief) + '</textarea>';
    }

    function planStageHtml() {
        var shot = selectedShot();
        return '<div class="h3s-stage-copy">Timed visible and audible action. Name composition, blocking, lighting, camera type, amplitude, speed, cuts, dialogue and physical sound.</div>' +
            '<textarea class="h3s-textarea is-script" data-shot-field="shot_description" placeholder="[Shot 1] Close medium…">' + escapeHtml(shot.shot_description) + '</textarea>';
    }

    function promptStageHtml() {
        var shot = selectedShot();
        var prompt = C.compilePrompt(shot);
        var issue = C.promptComplianceIssue(prompt, C.detectMode(shot), shot.duration_seconds);
        return '<div class="h3s-panel-head"><div><div class="h3s-kicker">Canonical H3 prompt</div><div class="h3s-stage-copy" style="margin:4px 0 0">Advanced override. Clear the override to compile from shot fields again.</div></div>' +
            '<button class="h3s-btn" data-h3-action="compile-prompt">Compile fields</button></div>' +
            (issue ? '<div class="h3s-warning h3s-error">' + escapeHtml(issue) + '</div>' : '<div class="h3s-warning" style="border-color:rgba(126,175,120,.35);background:rgba(126,175,120,.08);color:#a7c8a2">Prompt structure passes the local H3 contract.</div>') +
            '<textarea class="h3s-textarea is-prompt" data-shot-field="prompt_override" placeholder="Canonical prompt…">' + escapeHtml(prompt) + '</textarea>';
    }

    function endlessState() {
        if (!state.project.endless) state.project.endless = C.createEmptyEndless(state.project.target_duration_seconds);
        return state.project.endless;
    }

    function endlessStageHtml() {
        var run = endlessState();
        var active = ['submitting', 'running', 'stopping'].indexOf(run.status) >= 0;
        var completed = run.completed_job_ids.length;
        var planned = run.segment_durations.length || 0;
        var renderedFrames = (run.segment_output_frames || []).slice(0, completed).reduce(function (sum, frames) { return sum + Number(frames || 0); }, 0);
        var activeText = run.active_job ? (' · active ' + escapeHtml(run.active_job.video_id || 'job')) : '';
        var outputs = run.completed_job_ids.map(function (jobId, index) {
            var path = run.completed_output_paths[index] || '';
            var playablePath = /^\/(?!\/)/.test(path) ? path : '';
            return '<li><strong>' + escapeHtml(jobId) + '</strong> · segment ' + (index + 1) +
                (playablePath ? ' · <a href="' + attr(playablePath) + '" target="_blank" rel="noopener">open MP4</a>' : '') + '</li>';
        }).join('');
        return '<div class="h3s-endless-card"><div class="h3s-panel-head"><div><div class="h3s-kicker">Endless story · inference only</div>' +
            '<div class="h3s-stage-copy" style="margin:4px 0 0">Serial H3 renders reuse the prior job’s native 22-frame motion/audio context. The browser never rebuilds model state or starts training.</div></div>' +
            '<span class="h3s-chip ' + (run.status === 'failed' ? 'is-blocked' : (run.status === 'completed' ? 'is-ready' : '')) + '">' + escapeHtml(String(run.status || 'idle').toUpperCase()) + '</span></div>' +
            '<div class="h3s-grid-2"><label class="h3s-field"><span>Target runtime · 5–3600s</span><input class="h3s-input" type="number" min="5" max="3600" step="0.041666667" data-endless-field="target_seconds" value="' + attr(run.target_seconds) + '"' + disabled(active) + '></label>' +
            '<label class="h3s-field"><span>Preferred segment · 5–15s</span><input class="h3s-input" type="number" min="5" max="15" step="0.041666667" data-endless-field="segment_seconds" value="' + attr(run.segment_seconds) + '"' + disabled(active) + '></label></div>' +
            '<label class="h3s-field"><span>Direction repeated for every continuation</span><textarea class="h3s-textarea" data-endless-field="continuation_direction"' + disabled(active) + '>' + escapeHtml(run.continuation_direction) + '</textarea></label>' +
            '<div class="h3s-button-row"><button class="h3s-btn is-primary" data-h3-action="start-endless"' + disabled(active) + '>Start endless story</button>' +
            '<button class="h3s-btn" data-h3-action="stop-endless"' + disabled(!active || run.status === 'stopping') + '>Stop after current</button>' +
            '<button class="h3s-btn is-quiet" data-h3-action="reset-endless"' + disabled(active) + '>Reset run</button></div>' +
            '<div class="h3s-endless-progress"><div><strong>' + completed + (planned ? ' / ' + planned : '') + '</strong> segments · ' + renderedFrames + ' frames / ' + Number(run.target_frames || 0) + ' · ' + C.secondsText(renderedFrames / C.NATIVE_FPS) + 's rendered' + activeText + '</div>' +
            '<div class="h3s-progress-track"><span style="width:' + (planned ? Math.min(100, completed * 100 / planned) : 0) + '%"></span></div></div>' +
            (run.error ? '<div class="h3s-warning h3s-error">' + escapeHtml(run.error) + '</div>' : '') +
            (outputs ? '<ol class="h3s-endless-outputs">' + outputs + '</ol>' : '') + '</div>';
    }

    function directorStageHtml() {
        var action = C.actionById(state.directorAction);
        var minShots = C.directorMinimumShots(state.project);
        var maxShots = C.directorMaximumShots(state.project);
        var character = state.directorAction === 'character_sheet';
        return '<div class="h3s-director-head"><label class="h3s-field"><span>Director operation</span><select id="h3s-director-action" class="h3s-select">' + C.ACTIONS.map(function (item) {
            return '<option value="' + item.id + '"' + selected(item.id, state.directorAction) + '>' + item.group + ' · ' + item.label + '</option>';
        }).join('') + '</select></label>' +
            '<div class="h3s-action-note"><strong>' + escapeHtml(action.help) + '</strong><em>Needs:</em> ' + escapeHtml(action.needs) + '</div></div>' +
            (character ? '<div class="h3s-grid-2"><label class="h3s-field"><span>Sheet plan</span><select id="h3s-character-panels" class="h3s-select"><option value="6"' + selected(state.characterPanels, 6) + '>6 panels · 124 frames · canonical</option><option value="4"' + selected(state.characterPanels, 4) + '>4 panels · 73 intended · experimental</option></select></label>' +
                '<label class="h3s-field"><span>Style</span><select id="h3s-character-style" class="h3s-select"><option value="standard_orbit"' + selected(state.characterStyle, 'standard_orbit') + '>Keep Picture 1 style</option><option value="anime_to_real"' + selected(state.characterStyle, 'anime_to_real') + '>Anime to photoreal</option></select></label></div>' +
                (state.characterPanels === 4 ? '<div class="h3s-warning">Four-panel is metadata only: the upstream geometry conflicts and 73 frames is below Serenity’s five-second render minimum.</div>' : '') : '') +
            '<label class="h3s-field"><span>Director input</span><textarea class="h3s-textarea is-script" data-project-field="director_brief" placeholder="Story outline, script, edit request, shot diagnosis, or character extraction instructions…">' + escapeHtml(state.project.director_brief) + '</textarea></label>' +
            '<div class="h3s-button-row"><button class="h3s-btn is-primary" data-h3-action="run-director"' + disabled(state.directorRunning) + '>' + (state.directorRunning ? 'Director pass running…' : 'Run Qwen Director pass') + '</button><button class="h3s-btn" data-h3-action="apply-director"' + disabled(!directorResultJson()) + '>Apply result to project</button><button class="h3s-btn" data-h3-action="prepare-director">Prepare Qwen Director pass</button><button class="h3s-btn" data-h3-action="copy-request"' + disabled(!state.requestJson) + '>Copy request</button><button class="h3s-btn" data-h3-action="download-request"' + disabled(!state.requestJson) + '>Download request</button></div>' +
            '<div class="h3s-help" style="margin-top:7px">Plan envelope: ' + minShots + '–' + maxShots + ' shots · ' + state.project.takes_per_shot + ' take(s) each. Run launches the pure-Mojo Qwen3-VL H3 captioner on the GPU (about 1–3 minutes, no H3 render). Prepare only is CPU-only.</div>' +
            directorResultHtml() +
            (state.requestJson ? '<details class="h3s-request"><summary>Prepared serenity.h3.caption.v2 request</summary><pre>' + escapeHtml(state.requestJson) + '</pre></details>' : '');
    }

    function stageHtml() {
        var tabs = [['brief', 'Brief'], ['plan', 'Shot plan'], ['prompt', 'H3 prompt'], ['director', 'Director'], ['endless', 'Endless']];
        var body = state.stageTab === 'brief' ? briefStageHtml() : state.stageTab === 'plan' ? planStageHtml() : state.stageTab === 'prompt' ? promptStageHtml() : state.stageTab === 'endless' ? endlessStageHtml() : directorStageHtml();
        return '<div class="h3s-stage"><div class="h3s-stage-tabs">' + tabs.map(function (tab) {
            return '<button class="h3s-tab ' + (state.stageTab === tab[0] ? 'is-active' : '') + '" data-stage-tab="' + tab[0] + '">' + tab[1] + '</button>';
        }).join('') + '</div><div class="h3s-stage-body">' + body + '</div>' +
            '<div class="h3s-button-row" style="margin-top:9px"><button class="h3s-btn is-primary" data-h3-action="render-shot">Queue H3 take</button><button class="h3s-btn" data-h3-action="continue-take"' + disabled(!selectedTakeDone()) + '>Continue selected take</button></div></div>';
    }

    function centerHtml() { return '<main class="h3s-center">' + monitorHtml() + stageHtml() + '</main>'; }

    function shotInspectorHtml() {
        var shot = selectedShot();
        var lock = shot.locked;
        var attentionBackends = h3AttentionBackends();
        var effectiveAttention = resolvedH3Attention(shot);
        var attentionOptions = H3AttentionContracts.definitions(attentionBackends)
            .filter(function (backend) {
                return backend && ['ck-int8', 'cudnn', 'sage-int8'].indexOf(backend.id) >= 0;
            }).map(function (backend) {
                var available = H3AttentionContracts.isAvailable(
                    shot.quant, backend.id, attentionBackends);
                return '<option value="' + attr(backend.id) + '"' +
                    selected(effectiveAttention, backend.id) + disabled(!available) + '>' +
                    escapeHtml(backend.label || backend.id) +
                    (available ? '' : ' · unavailable') + '</option>';
            }).join('');
        return '<label class="h3s-check"><input type="checkbox" data-shot-field="locked"' + checked(lock) + '><span>Lock approved shot. Protect prompt, references, sound, order and takes.</span></label>' +
            '<label class="h3s-field" style="margin-top:10px"><span>Shot name</span><input class="h3s-input" data-shot-field="title" value="' + attr(shot.title) + '"' + disabled(lock) + '></label>' +
            '<div class="h3s-grid-2"><label class="h3s-field"><span>Duration · 5–15s</span><input class="h3s-input" type="number" min="5" max="15" step="0.25" data-shot-field="duration_seconds" value="' + attr(shot.duration_seconds) + '"' + disabled(lock) + '></label>' +
            '<label class="h3s-field"><span>Resolution</span><select id="h3s-resolution" class="h3s-select"' + disabled(lock) + '>' + C.RESOLUTIONS.map(function (row) {
                return '<option value="' + row.width + 'x' + row.height + '"' + selected(shot.width + 'x' + shot.height, row.width + 'x' + row.height) + '>' + row.label + '</option>';
            }).join('') + '</select></label></div>' +
            '<div class="h3s-grid-2"><label class="h3s-field"><span>Steps</span><input class="h3s-input" type="number" min="2" max="50" data-shot-field="steps" value="' + shot.steps + '"' + disabled(lock) + '></label><label class="h3s-field"><span>Seed</span><input class="h3s-input" type="number" min="0" max="4294967295" data-shot-field="seed" value="' + shot.seed + '"' + disabled(lock) + '></label></div>' +
            '<label class="h3s-field"><span>Opening frame</span><div class="h3s-button-row"><input class="h3s-input" data-shot-field="first_frame" value="' + attr(shot.first_frame) + '" placeholder="Server-uploaded path"' + disabled(lock) + '><button class="h3s-btn" data-h3-action="upload-first"' + disabled(lock) + '>Choose</button></div></label>' +
            '<label class="h3s-field"><span>Ending frame</span><div class="h3s-button-row"><input class="h3s-input" data-shot-field="last_frame" value="' + attr(shot.last_frame) + '" placeholder="Server-uploaded path"' + disabled(lock) + '><button class="h3s-btn" data-h3-action="upload-last"' + disabled(lock) + '>Choose</button></div></label>' +
            '<label class="h3s-field"><span>Continue from · video-XXXX</span><input class="h3s-input" data-shot-field="continue_from" value="' + attr(shot.continue_from) + '"' + disabled(lock) + '></label>' +
            '<label class="h3s-field"><span>Motion seam</span><select class="h3s-select" data-shot-field="motion_context_frames"' + disabled(lock) + '><option value="5"' + selected(shot.motion_context_frames, 5) + '>Short · 5 frames</option><option value="22"' + selected(shot.motion_context_frames, 22) + '>Balanced · 22 frames</option><option value="39"' + selected(shot.motion_context_frames, 39) + '>Long · 39 frames</option></select></label>' +
            '<div class="h3s-grid-2"><label class="h3s-field"><span>Precision</span><select class="h3s-select" data-shot-field="quant"' + disabled(lock) + '><option value="int8-fast"' + selected(shot.quant, 'int8-fast') + '>INT8 Fast</option><option value="int8"' + selected(shot.quant, 'int8') + '>INT8 Quality</option><option value="bf16"' + selected(shot.quant, 'bf16') + '>BF16</option></select></label>' +
            '<label class="h3s-field"><span>Attention</span><select class="h3s-select" data-shot-field="attention_backend"' + disabled(lock) + '>' + attentionOptions + '</select></label></div>' +
            '<label class="h3s-field"><span>Denoise acceleration</span><select class="h3s-select" data-shot-field="step_cache"' + disabled(lock) + '><option value="exact"' + selected(shot.step_cache, 'exact') + '>Exact · quality default</option><option value="high"' + selected(shot.step_cache, 'high') + '>High · approximate</option></select></label>';
    }

    function soundInspectorHtml() {
        var shot = selectedShot(), lock = shot.locked;
        return '<div class="h3s-warning">H3 always generates synchronized sound. Dialogue stays inside the timed shot plan using stable speaker IDs.</div>' +
            '<label class="h3s-field" style="margin-top:10px"><span>Ambience · Foley · physical sound</span><textarea class="h3s-textarea" data-shot-field="soundscape"' + disabled(lock) + '>' + escapeHtml(shot.soundscape) + '</textarea></label>' +
            '<label class="h3s-field"><span>Audience-only score · instrumentation · tempo</span><textarea class="h3s-textarea" data-shot-field="music"' + disabled(lock) + '>' + escapeHtml(shot.music) + '</textarea></label>';
    }

    function referenceInspectorHtml() {
        var shot = selectedShot(), lock = shot.locked;
        return '<div class="h3s-panel-head"><div><div class="h3s-kicker">Ordered reference pack</div><div class="h3s-help" style="margin-top:4px">9 images · 3 videos · 3 audio · 12 total</div></div><button class="h3s-btn is-primary" data-h3-action="upload-references"' + disabled(lock) + '>Add files</button></div>' +
            '<div class="h3s-ref-list">' + (shot.references.length ? shot.references.map(function (item, index) {
                return '<div class="h3s-ref"><div class="h3s-ref-top"><span class="h3s-ref-order">' + String(index + 1).padStart(2, '0') + '</span><span class="h3s-ref-kind">' + escapeHtml(item.kind) + '</span><span class="h3s-ref-path" title="' + attr(item.path) + '">' + escapeHtml(item.path) + '</span><span class="h3s-ref-actions"><button class="h3s-btn h3s-icon-btn" data-ref-move="-1" data-ref-index="' + index + '">↑</button><button class="h3s-btn h3s-icon-btn" data-ref-move="1" data-ref-index="' + index + '">↓</button><button class="h3s-btn h3s-icon-btn is-danger" data-ref-remove="' + index + '">×</button></span></div>' +
                    (item.kind === 'audio' ? '<select class="h3s-select" data-ref-field="audio_use" data-ref-index="' + index + '"><option value="reference"' + selected(item.audio_use, 'reference') + '>Reference signal</option><option value="reuse"' + selected(item.audio_use, 'reuse') + '>Reuse signal</option><option value="voice_timbre"' + selected(item.audio_use, 'voice_timbre') + '>Voice timbre</option></select>' : '<select class="h3s-select" data-ref-field="role" data-ref-index="' + index + '"><option value="subject"' + selected(item.role, 'subject') + '>Subject / identity</option><option value="source_video"' + selected(item.role, 'source_video') + '>Source video / edit</option><option value="keyframe"' + selected(item.role, 'keyframe') + '>Keyframe / composition</option><option value="motion_camera"' + selected(item.role, 'motion_camera') + '>Motion / camera</option><option value="environment_style"' + selected(item.role, 'environment_style') + '>Environment / style</option></select>') +
                    '<input class="h3s-input" data-ref-field="note" data-ref-index="' + index + '" value="' + attr(item.note) + '" placeholder="What to keep/use and what to ignore/remove">' +
                    ((item.kind === 'video' || item.kind === 'audio') ? '<input class="h3s-input" type="number" min="2" max="15" step=".1" data-ref-field="duration_seconds" data-ref-index="' + index + '" value="' + attr(item.duration_seconds) + '" aria-label="Reference duration">' : '') + '</div>';
            }).join('') : '<div class="h3s-help">Add image, video, or audio files. Uploads are stored by the server; browser-local paths are never invented.</div>') + '</div>';
    }

    function assetInspectorHtml() {
        return '<div class="h3s-panel-head"><div><div class="h3s-kicker">Project assets</div><div class="h3s-help" style="margin-top:4px">Reusable identity, location, product, logo, sound and motion sources</div></div><button class="h3s-btn is-primary" data-h3-action="upload-asset">Add asset</button></div>' +
            '<div class="h3s-asset-list">' + (state.project.assets.length ? state.project.assets.map(function (asset, index) {
                return '<div class="h3s-asset"><div class="h3s-asset-top"><span class="h3s-ref-kind">' + escapeHtml(asset.kind) + '</span><strong class="h3s-ref-path">' + escapeHtml(asset.name) + '</strong><button class="h3s-btn h3s-icon-btn is-danger" data-asset-remove="' + index + '">×</button></div><div class="h3s-asset-path" title="' + attr(asset.path) + '">' + escapeHtml(asset.path) + '</div></div>';
            }).join('') : '<div class="h3s-help">No reusable project assets yet.</div>') + '</div>';
    }

    function rightHtml() {
        var tabs = [['shot', 'Shot'], ['sound', 'Sound'], ['references', 'References'], ['assets', 'Assets']];
        var body = state.inspectorTab === 'shot' ? shotInspectorHtml() : state.inspectorTab === 'sound' ? soundInspectorHtml() : state.inspectorTab === 'references' ? referenceInspectorHtml() : assetInspectorHtml();
        return '<aside class="h3s-right"><div class="h3s-inspector-tabs">' + tabs.map(function (tab) { return '<button class="h3s-tab ' + (state.inspectorTab === tab[0] ? 'is-active' : '') + '" data-inspector-tab="' + tab[0] + '">' + tab[1] + '</button>'; }).join('') + '</div><div class="h3s-inspector-body">' + body + '</div></aside>';
    }

    function timelineHtml() {
        var cumulative = 0;
        var rows = [];
        state.project.shots.forEach(function (shot, index) {
            if (index) rows.push('<span class="h3s-spine-join"></span>');
            var start = cumulative; cumulative += Number(shot.duration_seconds || 0);
            rows.push('<button class="h3s-spine-shot ' + (shot.id === state.selectedShotId ? 'is-active ' : '') + (shot.locked ? 'is-locked' : '') + '" data-select-shot="' + shot.id + '"><div class="h3s-spine-title">' + escapeHtml(shot.title) + '</div><div class="h3s-spine-meta">' + timecode(start) + ' → ' + timecode(cumulative) + '</div><div class="h3s-spine-meta">' + C.detectMode(shot).toUpperCase() + ' · ' + shot.take_job_ids.length + ' TAKE(S)</div></button>');
        });
        return '<section class="h3s-timeline"><div class="h3s-timeline-head"><span class="h3s-kicker">Continuity spine</span><span class="h3s-timecode">' + timecode(totalSeconds()) + ' · DELIVERY ' + state.project.delivery_fps + ' FPS</span></div><div class="h3s-spine">' + rows.join('') + '</div></section>';
    }

    function statusHtml() {
        return '<footer class="h3s-statusbar"><span id="h3s-status-message" class="' + (state.statusTone === 'live' ? 'is-live' : '') + '">' + escapeHtml(state.status) + '</span><span>PROJECT ' + escapeHtml(C.PROJECT_SCHEMA) + ' · H3 SOURCE 24 FPS · NO TRAINING</span></footer>';
    }

    function hiddenInputsHtml() {
        return '<input id="h3s-project-import" type="file" accept=".json,.serenitymovie.json" hidden>' +
            '<input id="h3s-first-file" type="file" accept="image/*" hidden><input id="h3s-last-file" type="file" accept="image/*" hidden>' +
            '<input id="h3s-reference-files" type="file" accept="image/*,video/*,audio/*" multiple hidden><input id="h3s-asset-file" type="file" accept="image/*,video/*,audio/*" hidden>';
    }

    var renderDepth = 0;
    var renderPending = false;
    function render() {
        // Field blur/change handlers can re-enter render() while innerHTML is
        // being replaced (NotFoundError on detached nodes). Coalesce instead.
        if (renderDepth > 0) { renderPending = true; return; }
        renderDepth += 1;
        try { renderNow(); }
        finally {
            renderDepth -= 1;
            if (renderPending) { renderPending = false; setTimeout(render, 0); }
        }
    }

    function renderNow() {
        var panel = document.getElementById('panel-h3-studio');
        if (!panel) return;
        if (!state.project) state.project = loadProject();
        if (!state.project.shots.some(function (shot) { return shot.id === state.selectedShotId; })) state.selectedShotId = state.project.shots[0].id;
        panel.innerHTML = '<div class="h3s-app">' + headerHtml() + '<div class="h3s-workspace">' + leftHtml() + centerHtml() + rightHtml() + '</div>' + timelineHtml() + statusHtml() + hiddenInputsHtml() + '</div>';
        bindRenderedEvents();
        if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons();
    }

    function setProjectField(field, target) {
        var value = target.type === 'checkbox' ? target.checked : target.value;
        if (['project_kind', 'caption_tier', 'delivery_fps', 'target_duration_seconds', 'takes_per_shot'].indexOf(field) >= 0) value = Number(value);
        state.project[field] = value;
        if (field === 'target_duration_seconds' && endlessState().status === 'idle') {
            endlessState().target_seconds = value;
            endlessState().target_frames = Math.round(value * C.NATIVE_FPS);
        }
        saveProject();
    }

    function setShotField(field, target) {
        var shot = selectedShot();
        if (baseShotMutationBlocked(shot)) {
            showToast('The endless base shot is immutable until this run stops or completes.', 'error'); render(); return;
        }
        var value = target.type === 'checkbox' ? target.checked : target.value;
        if (['duration_seconds', 'steps', 'seed', 'motion_context_frames'].indexOf(field) >= 0) value = Number(value);
        shot[field] = value;
        if (field !== 'prompt_override' && field !== 'locked' && ['title', 'seed', 'steps', 'quant', 'attention_backend', 'step_cache', 'motion_context_frames'].indexOf(field) < 0) shot.prompt_override = '';
        if (field === 'quant' || field === 'attention_backend')
            shot.attention_backend = resolvedH3Attention(shot);
        saveProject();
    }

    function bindRenderedEvents() {
        var panel = document.getElementById('panel-h3-studio');
        panel.querySelectorAll('[data-select-shot]').forEach(function (node) {
            node.addEventListener('click', function () { state.selectedShotId = Number(node.dataset.selectShot); state.requestJson = ''; render(); });
        });
        panel.querySelectorAll('[data-stage-tab]').forEach(function (node) { node.addEventListener('click', function () { state.stageTab = node.dataset.stageTab; render(); }); });
        panel.querySelectorAll('[data-inspector-tab]').forEach(function (node) { node.addEventListener('click', function () { state.inspectorTab = node.dataset.inspectorTab; render(); }); });
        panel.querySelectorAll('[data-bible-tab]').forEach(function (node) { node.addEventListener('click', function () { state.bibleTab = node.dataset.bibleTab; render(); }); });
        panel.querySelectorAll('[data-project-field]').forEach(function (node) {
            var eventName = node.tagName === 'TEXTAREA' || node.type === 'text' ? 'input' : 'change';
            node.addEventListener(eventName, function () { setProjectField(node.dataset.projectField, node); });
        });
        panel.querySelectorAll('[data-shot-field]').forEach(function (node) {
            var eventName = node.tagName === 'TEXTAREA' || node.type === 'text' ? 'input' : 'change';
            node.addEventListener(eventName, function () { setShotField(node.dataset.shotField, node); if (eventName === 'change') render(); });
        });
        panel.querySelectorAll('[data-endless-field]').forEach(function (node) {
            var eventName = node.tagName === 'TEXTAREA' ? 'input' : 'change';
            node.addEventListener(eventName, function () {
                var field = node.dataset.endlessField;
                var run = endlessState();
                run[field] = field === 'continuation_direction' ? node.value : Number(node.value);
                if (field === 'target_seconds') { run.target_frames = Math.round(run.target_seconds * C.NATIVE_FPS); run.target_seconds = run.target_frames / C.NATIVE_FPS; }
                if (field === 'segment_seconds') { run.segment_frames = Math.round(run.segment_seconds * C.NATIVE_FPS); run.segment_seconds = run.segment_frames / C.NATIVE_FPS; }
                run.error = '';
                saveProject();
            });
        });
        panel.querySelectorAll('[data-ref-field]').forEach(function (node) {
            node.addEventListener(node.tagName === 'INPUT' ? 'input' : 'change', function () {
                if (baseShotMutationBlocked(selectedShot())) { showToast('The endless base references are immutable during the active run.', 'error'); render(); return; }
                var index = Number(node.dataset.refIndex); var value = node.dataset.refField === 'duration_seconds' ? Number(node.value) : node.value;
                selectedShot().references[index][node.dataset.refField] = value; selectedShot().prompt_override = ''; saveProject();
            });
        });
        panel.querySelectorAll('[data-ref-move]').forEach(function (node) { node.addEventListener('click', function () { moveReference(Number(node.dataset.refIndex), Number(node.dataset.refMove)); }); });
        panel.querySelectorAll('[data-ref-remove]').forEach(function (node) { node.addEventListener('click', function () { if (baseShotMutationBlocked(selectedShot())) { showToast('The endless base references are immutable during the active run.', 'error'); return; } selectedShot().references.splice(Number(node.dataset.refRemove), 1); selectedShot().prompt_override = ''; saveProject(); render(); }); });
        panel.querySelectorAll('[data-asset-remove]').forEach(function (node) { node.addEventListener('click', function () { state.project.assets.splice(Number(node.dataset.assetRemove), 1); saveProject(); render(); }); });
        panel.querySelectorAll('[data-h3-action]').forEach(function (node) { node.addEventListener('click', function () { handleAction(node.dataset.h3Action); }); });
        var projectTitle = document.getElementById('h3s-project-title');
        if (projectTitle) projectTitle.addEventListener('input', function () { state.project.title = projectTitle.value; saveProject(); });
        var resolution = document.getElementById('h3s-resolution');
        if (resolution) resolution.addEventListener('change', function () { if (baseShotMutationBlocked(selectedShot())) { showToast('The endless base dimensions are immutable during the active run.', 'error'); render(); return; } var parts = resolution.value.split('x'); selectedShot().width = Number(parts[0]); selectedShot().height = Number(parts[1]); saveProject(); render(); });
        var action = document.getElementById('h3s-director-action');
        if (action) action.addEventListener('change', function () { state.directorAction = action.value; state.requestJson = ''; render(); });
        var panels = document.getElementById('h3s-character-panels');
        if (panels) panels.addEventListener('change', function () { state.characterPanels = Number(panels.value); state.requestJson = ''; render(); });
        var style = document.getElementById('h3s-character-style');
        if (style) style.addEventListener('change', function () { state.characterStyle = style.value; state.requestJson = ''; render(); });
        bindFileInputs();
    }

    function moveReference(index, delta) {
        if (baseShotMutationBlocked(selectedShot())) { showToast('The endless base references are immutable during the active run.', 'error'); return; }
        var refs = selectedShot().references, target = index + delta;
        if (target < 0 || target >= refs.length) return;
        var item = refs.splice(index, 1)[0]; refs.splice(target, 0, item); selectedShot().prompt_override = ''; saveProject(); render();
    }

    function bindFileInputs() {
        var projectInput = document.getElementById('h3s-project-import');
        projectInput.addEventListener('change', function () { importProject(projectInput.files[0]); });
        document.getElementById('h3s-first-file').addEventListener('change', function (event) { uploadSingle(event.target.files[0], 'first_frame'); });
        document.getElementById('h3s-last-file').addEventListener('change', function (event) { uploadSingle(event.target.files[0], 'last_frame'); });
        document.getElementById('h3s-reference-files').addEventListener('change', function (event) { uploadReferences(Array.from(event.target.files || [])); });
        document.getElementById('h3s-asset-file').addEventListener('change', function (event) { uploadAsset(event.target.files[0]); });
    }

    function mediaKind(file) {
        var type = String(file && file.type || '');
        if (type.indexOf('video/') === 0) return 'video';
        if (type.indexOf('audio/') === 0) return 'audio';
        return 'image';
    }

    function uploadSingle(file, field) {
        if (!file) return;
        if (baseShotMutationBlocked(selectedShot())) { showToast('The endless base inputs are immutable during the active run.', 'error'); return; }
        setStatus('Uploading ' + file.name + '…', 'live');
        SerenityAPI.uploadMediaDetails(file).then(function (data) {
            selectedShot()[field] = data.path || data.name || '';
            selectedShot().prompt_override = ''; saveProject(); setStatus('Uploaded ' + file.name, ''); render();
        }).catch(function (error) { setStatus('Upload failed: ' + error.message, 'error'); });
    }

    function uploadReferences(files) {
        if (!files.length) return;
        if (baseShotMutationBlocked(selectedShot())) { showToast('The endless base references are immutable during the active run.', 'error'); return; }
        setStatus('Uploading ' + files.length + ' reference file(s)…', 'live');
        var chain = Promise.resolve();
        files.forEach(function (file) {
            chain = chain.then(function () { return SerenityAPI.uploadMediaDetails(file).then(function (data) {
                var kind = mediaKind(file); var ref = C.createReference(kind, data.path || data.name || '');
                ref.role = kind === 'video' && selectedShot().references.every(function (item) { return item.role !== 'source_video'; }) ? 'source_video' : 'subject';
                ref.note = file.name; selectedShot().references.push(ref);
            }); });
        });
        chain.then(function () { selectedShot().prompt_override = ''; saveProject(); setStatus('References uploaded and ordered', ''); render(); })
            .catch(function (error) { setStatus('Reference upload failed: ' + error.message, 'error'); });
    }

    function uploadAsset(file) {
        if (!file) return;
        setStatus('Uploading project asset…', 'live');
        SerenityAPI.uploadMediaDetails(file).then(function (data) {
            state.project.assets.push({ id: state.project.next_asset_id++, kind: mediaKind(file), name: file.name, path: data.path || data.name || '', notes: '' });
            saveProject(); setStatus('Project asset added', ''); render();
        }).catch(function (error) { setStatus('Asset upload failed: ' + error.message, 'error'); });
    }

    function handleAction(action) {
        if (action === 'new-project') newProject();
        else if (action === 'import-project') document.getElementById('h3s-project-import').click();
        else if (action === 'export-project') downloadJson(safeName(state.project.title) + '.serenitymovie.json', state.project);
        else if (action === 'export-edit') downloadJson(safeName(state.project.title) + '.serenityedit.json', C.deliveryManifest(state.project));
        else if (action === 'add-shot') addShot();
        else if (action === 'duplicate-shot') duplicateShot();
        else if (action === 'delete-shot') deleteShot();
        else if (action === 'shot-left') moveShot(-1);
        else if (action === 'shot-right') moveShot(1);
        else if (action === 'compile-prompt') { if (baseShotMutationBlocked(selectedShot())) { showToast('The endless base prompt is immutable during the active run.', 'error'); return; } selectedShot().prompt_override = ''; saveProject(); render(); }
        else if (action === 'prepare-director') prepareDirector();
        else if (action === 'run-director') runDirector();
        else if (action === 'apply-director') applyDirector();
        else if (action === 'copy-request') copyRequest();
        else if (action === 'download-request') downloadPreparedRequest();
        else if (action === 'upload-first') document.getElementById('h3s-first-file').click();
        else if (action === 'upload-last') document.getElementById('h3s-last-file').click();
        else if (action === 'upload-references') document.getElementById('h3s-reference-files').click();
        else if (action === 'upload-asset') document.getElementById('h3s-asset-file').click();
        else if (action === 'render-shot') renderShot();
        else if (action === 'continue-take') continueTake();
        else if (action === 'start-endless') startEndless();
        else if (action === 'stop-endless') stopEndless();
        else if (action === 'reset-endless') resetEndless();
    }

    function newProject() {
        if (['submitting', 'running', 'stopping'].indexOf(endlessState().status) >= 0) {
            showToast('Stop the active endless story before replacing this project', 'error'); return;
        }
        if (!window.confirm('Create a new H3 project? Export the current project first if you need a file copy.')) return;
        state.project = C.createProject(); state.selectedShotId = 1; state.requestJson = ''; saveProject('New movie project created'); render();
    }
    function addShot() {
        var id = state.project.next_shot_id++; var shot = C.createShot(id, 'Shot ' + (state.project.shots.length + 1));
        state.project.shots.push(shot); state.selectedShotId = id; saveProject('Shot added'); render();
    }
    function duplicateShot() {
        var original = selectedShot(); var clone = C.copy(original); clone.id = state.project.next_shot_id++; clone.title += ' copy'; clone.take_job_ids = []; clone.take_states = []; clone.take_output_paths = []; clone.selected_take = -1; clone.status = 'Unrendered'; clone.output_path = ''; clone.locked = false;
        state.project.shots.splice(selectedShotIndex() + 1, 0, clone); state.selectedShotId = clone.id; saveProject('Shot duplicated'); render();
    }
    function deleteShot() {
        if (state.project.shots.length === 1) { setStatus('A project must keep at least one shot', 'error'); return; }
        if (endlessChainShotLocked(selectedShot())) { showToast('An active endless chain shot cannot be deleted.', 'error'); return; }
        if (!window.confirm('Delete selected shot from this project?')) return;
        var index = selectedShotIndex(); state.project.shots.splice(index, 1); state.selectedShotId = state.project.shots[Math.min(index, state.project.shots.length - 1)].id; saveProject('Shot deleted'); render();
    }
    function moveShot(delta) {
        if (endlessChainShotLocked(selectedShot())) { showToast('An active endless chain shot cannot be reordered.', 'error'); return; }
        var index = selectedShotIndex(), target = index + delta; if (target < 0 || target >= state.project.shots.length) return;
        var shot = state.project.shots.splice(index, 1)[0]; state.project.shots.splice(target, 0, shot); saveProject('Shot order updated'); render();
    }

    function directorResultJson() {
        var last = state.project.last_director_result;
        return last && last.json && typeof last.json === 'object' ? last.json : null;
    }

    function directorResultHtml() {
        var last = state.project.last_director_result;
        if (!last) return '';
        var json = directorResultJson();
        var summary = json ? String(json.summary || '') : '';
        var warnings = json && Array.isArray(json.warnings) ? json.warnings : [];
        var shotCount = json && Array.isArray(json.shots) ? json.shots.length : 0;
        return '<div class="h3s-director-result" style="margin-top:9px"><div class="h3s-help"><strong>Director result</strong> · ' + escapeHtml(String(last.operation || '')) + ' · ' + escapeHtml(String(last.ran_at || '')) + (last.elapsed_ms ? ' · ' + Math.round(last.elapsed_ms / 1000) + ' s' : '') + (shotCount ? ' · ' + shotCount + ' shots planned' : '') + (json ? '' : ' · <em>not valid JSON — shown raw</em>') + '</div>' +
            (summary ? '<div class="h3s-stage-copy">' + escapeHtml(summary) + '</div>' : '') +
            (warnings.length ? '<div class="h3s-warning">' + escapeHtml(warnings.join(' · ')) + '</div>' : '') +
            '<details class="h3s-request" open><summary>' + (json ? 'Result JSON' : 'Raw model output') + '</summary><pre>' + escapeHtml(json ? JSON.stringify(json, null, 2) : String(last.text || '')) + '</pre></details></div>';
    }

    function directorUserPrompt(request) {
        var body = C.copy(request);
        delete body.system_prompt;
        var brief = String(state.project.director_brief || '').trim();
        return (brief ? 'DIRECTOR INPUT:\n' + brief + '\n\n' : '') + 'CONTEXT (serenity.h3.caption.v2 request without system_prompt):\n' + JSON.stringify(body, null, 2) + '\n\nRespond with the JSON object only.';
    }

    function runDirector() {
        if (state.directorRunning) return;
        try {
            var request = C.captionRequest(state.project, selectedShot(), state.directorAction, { panel_count: state.characterPanels, style_mode: state.characterStyle });
            state.requestJson = JSON.stringify(request, null, 2);
            var image = '';
            (request.inputs || []).some(function (item) { if (item && item.kind === 'image' && String(item.path || '').trim()) { image = String(item.path).trim(); return true; } return false; });
            var payload = { system_prompt: request.system_prompt, prompt: directorUserPrompt(request), max_new: 1500 };
            if (image) payload.image_path = image;
            state.directorRunning = true; saveProject(); render();
            setStatus('Qwen Director pass running (' + state.directorAction + ')… about 1–3 minutes', 'live'); showToast('Director pass running on the GPU (Qwen3-VL)… about 1–3 minutes', 'live');
            var startedAt = Date.now();
            SerenityAPI.postH3Director(payload).then(function (data) {
                state.directorRunning = false;
                state.project.last_director_result = { schema: 'serenity.h3.director.run.v1', operation: state.directorAction, ran_at: new Date().toISOString(), elapsed_ms: Number(data.elapsed_ms) || (Date.now() - startedAt), text: String(data.text || ''), json: data.json && typeof data.json === 'object' ? data.json : null, image_path: image };
                saveProject('Director pass finished'); render();
                var ok = !!state.project.last_director_result.json;
                setStatus(ok ? 'Director pass done · review the result, then Apply result to project' : 'Director pass returned text that is not valid JSON; shown raw', ok ? '' : 'error');
                showToast(ok ? 'Director pass done in ' + Math.round((Number(data.elapsed_ms) || 0) / 1000) + ' s — review, then Apply' : 'Director pass finished but returned non-JSON text (shown raw)', ok ? '' : 'error');
            }).catch(function (error) {
                state.directorRunning = false; render();
                setStatus('Director pass failed: ' + error.message, 'error'); showToast('Director pass failed: ' + error.message, 'error');
            });
        } catch (error) { state.directorRunning = false; setStatus(error.message, 'error'); showToast(error.message, 'error'); }
    }

    function applyDirector() {
        if (endlessRunActive()) { showToast('Stop the endless run before applying a Director result to the project.', 'error'); return; }
        var last = state.project.last_director_result;
        var json = directorResultJson();
        if (!json) { showToast('No JSON director result to apply', 'error'); return; }
        try {
            var applied = C.applyDirectorResult(state.project, state.selectedShotId, last.operation || state.directorAction, json);
            state.project = applied.project; state.selectedShotId = applied.focusShotId; state.stageTab = 'brief';
            saveProject(applied.message); render(); setStatus(applied.message, ''); showToast(applied.message, '');
        } catch (error) { setStatus(error.message, 'error'); showToast(error.message, 'error'); }
    }

    function prepareDirector() {
        try {
            var request = C.captionRequest(state.project, selectedShot(), state.directorAction, { panel_count: state.characterPanels, style_mode: state.characterStyle });
            state.requestJson = JSON.stringify(request, null, 2); saveProject(); setStatus('Qwen Director request prepared · no model launched', ''); render();
        } catch (error) { setStatus(error.message, 'error'); }
    }
    function copyRequest() {
        if (!state.requestJson) return;
        navigator.clipboard.writeText(state.requestJson).then(function () { setStatus('Director request copied', ''); }).catch(function () { setStatus('Clipboard unavailable; download the request instead', 'error'); });
    }
    function downloadPreparedRequest() {
        if (!state.requestJson) return;
        downloadJson(safeName(state.project.title) + '-' + state.directorAction + '.h3caption.json', JSON.parse(state.requestJson));
    }

    function safeName(name) { return String(name || 'h3-project').trim().replace(/[^a-z0-9._-]+/gi, '-').replace(/^-+|-+$/g, '') || 'h3-project'; }
    function downloadJson(name, value) {
        var blob = new Blob([JSON.stringify(value, null, 2) + '\n'], { type: 'application/json' });
        var url = URL.createObjectURL(blob); var a = document.createElement('a'); a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove(); setTimeout(function () { URL.revokeObjectURL(url); }, 0);
    }
    function importProject(file) {
        if (!file) return;
        if (['submitting', 'running', 'stopping'].indexOf(endlessState().status) >= 0) {
            showToast('Stop the active endless story before importing another project', 'error'); return;
        }
        file.text().then(function (text) { var project = C.normalizeProject(JSON.parse(text)); state.project = project; state.selectedShotId = project.shots[0].id; state.requestJson = ''; saveProject('Project imported'); render(); })
            .catch(function (error) { setStatus('Project import failed: ' + error.message, 'error'); });
    }

    function selectedTakeDone() {
        var shot = selectedShot();
        if (!shot || shot.selected_take < 0) return false;
        return shot.take_states[shot.selected_take] === 'done';
    }

    function continueTake() {
        var shot = selectedShot();
        if (shot.selected_take < 0) { showToast('Select a finished take first (queue one, wait for Ready)', 'error'); return; }
        if (!selectedTakeDone()) { showToast('The selected take is still ' + (shot.take_states[shot.selected_take] || 'pending') + ' — continuation needs its finished motion context', 'error'); return; }
        var jobId = shot.take_job_ids[shot.selected_take] || '';
        if (!jobId) { setStatus('Selected take has no H3 job id', 'error'); showToast('Selected take has no H3 job id', 'error'); return; }
        addShot(); var next = selectedShot(); next.continue_from = jobId; next.motion_context_frames = 22; next.brief = 'Continue naturally from the approved take while preserving identity, motion, camera, lighting, sound and story state.'; next.title = 'Continue ' + shot.title; saveProject('Continuation shot created'); state.inspectorTab = 'shot'; render();
    }

    function endlessBaseShot(run) {
        return state.project.shots.find(function (shot) { return shot.id === Number(run.base_shot_id); }) || null;
    }

    function endlessRunActive(run) {
        return ['submitting', 'running', 'stopping'].indexOf((run || endlessState()).status) >= 0;
    }

    function baseShotMutationBlocked(shot) {
        var run = endlessState();
        return endlessRunActive(run) && shot && Number(shot.id) === Number(run.base_shot_id);
    }

    function endlessChainShotLocked(shot) {
        var run = endlessState();
        return endlessRunActive(run) && shot && ([Number(run.base_shot_id)].concat(run.segment_shot_ids || [])).indexOf(Number(shot.id)) >= 0;
    }

    function currentEndlessSnapshot(run) {
        var base = endlessBaseShot(run);
        if (!base) throw new Error('The base shot for this endless run no longer exists.');
        return C.endlessBaseSnapshot(base, run.target_seconds, run.segment_seconds, run.continuation_direction);
    }

    function assertEndlessResumeSafe(run) {
        if (!run.base_snapshot || !C.endlessSnapshotsEqual(currentEndlessSnapshot(run), run.base_snapshot))
            throw new Error('Endless resume rejected: the base prompt, inputs, dimensions, inference settings, target, segment length, or continuation direction changed. Reset the run to start from the new settings.');
    }

    function startEndless() {
        var draft = endlessState();
        try {
            var base = selectedShot();
            var resolvedAttention = resolvedH3Attention(base);
            if (base.attention_backend !== resolvedAttention) base.attention_backend = resolvedAttention;
            var run = C.createEndlessRun(base, draft.target_seconds, draft.segment_seconds, draft.continuation_direction);
            if (!window.confirm('Queue ' + run.segment_durations.length + ' serial H3 render(s) for ' + C.secondsText(run.target_seconds) + ' seconds total? Each completed segment automatically starts the next GPU job.')) return;
            state.project.endless = run;
            state.stageTab = 'endless';
            saveProject(); render();
            setStatus('Endless story authorized · preparing segment 1 of ' + run.segment_durations.length, 'live');
            submitNextEndlessSegment();
        } catch (error) {
            setStatus(error.message, 'error'); showToast(error.message, 'error');
        }
    }

    function stopEndless() {
        var run = endlessState();
        if (['submitting', 'running', 'stopping'].indexOf(run.status) < 0) return;
        run.status = 'stopping';
        run.updated_at = new Date().toISOString();
        if (!run.active_job && !state.endlessSubmitting) {
            run.status = 'stopped'; run.stopped_at = run.updated_at;
        }
        saveProject(); render();
        setStatus(run.status === 'stopped' ? 'Endless story stopped' : 'Stop requested · the active server job will finish, then no next segment will be submitted', '');
    }

    function resetEndless() {
        var run = endlessState();
        if (['submitting', 'running', 'stopping'].indexOf(run.status) >= 0) {
            showToast('Use Stop after current before resetting an active run', 'error'); return;
        }
        state.endlessPollToken += 1;
        state.project.endless = C.createEmptyEndless(state.project.target_duration_seconds);
        saveProject('Endless run reset'); render();
    }

    function registerEndlessQueue(jobId, run, segmentIndex) {
        if (typeof QueueTab === 'undefined') return;
        QueueTab.init();
        QueueTab.registerPending({
            promptId: jobId,
            prompt: run.continuation_direction,
            model: 'MiniMax H3',
            queuedAt: Date.now(),
            batchLabel: state.project.title + ' · Endless ' + (segmentIndex + 1) + '/' + run.segment_durations.length
        });
    }

    function submitNextEndlessSegment() {
        var run = endlessState();
        if (state.endlessSubmitting || run.active_job) return;
        if (run.status === 'stopping') {
            run.status = 'stopped'; run.stopped_at = new Date().toISOString(); saveProject(); render(); return;
        }
        if (run.status !== 'running') return;
        var index = run.completed_job_ids.length;
        if (index >= run.segment_durations.length) {
            run.status = 'completed'; run.updated_at = new Date().toISOString(); saveProject(); render(); return;
        }
        try {
            assertEndlessResumeSafe(run);
            var segmentShot = C.endlessSegmentShot(run, index);
            var request = C.renderRequest(segmentShot);
            var expectedSnapshot = C.copy(run.base_snapshot);
            state.endlessSubmitting = true;
            run.status = 'submitting'; run.error = ''; run.updated_at = new Date().toISOString();
            saveProject(); render();
            setStatus('Submitting endless segment ' + (index + 1) + ' of ' + run.segment_durations.length + '…', 'live');
            SerenityAPI.postVideo(request).then(function (job) {
                var current = endlessState();
                state.endlessSubmitting = false;
                if (!C.endlessSnapshotsEqual(current.base_snapshot, expectedSnapshot) || ['submitting', 'stopping'].indexOf(current.status) < 0) return;
                assertEndlessResumeSafe(current);
                if (!job || !(job.video_id || job.prompt_id)) throw new Error('server did not return a video job id');
                var videoId = String(job.video_id || job.prompt_id);
                var urls = C.videoJobUrls(videoId);
                if (!urls) throw new Error('server returned an invalid video job id');
                current.active_job = Object.assign({}, urls, {
                    segment_index: index,
                    not_found_count: 0,
                    submitted_at: new Date().toISOString()
                });
                if (current.status !== 'stopping') current.status = 'running';
                current.updated_at = new Date().toISOString();
                registerEndlessQueue(videoId, current, index);
                saveProject(); render();
                setStatus(videoId + ' · endless segment ' + (index + 1) + '/' + current.segment_durations.length + ' queued', 'live');
                pollEndlessActive();
            }).catch(function (error) {
                state.endlessSubmitting = false;
                var current = endlessState(); current.status = 'failed'; current.error = 'Segment submission failed: ' + error.message; current.updated_at = new Date().toISOString();
                saveProject(); render(); setStatus(current.error, 'error'); showToast(current.error, 'error');
            });
        } catch (error) {
            state.endlessSubmitting = false;
            run.status = 'failed'; run.error = error.message; run.updated_at = new Date().toISOString();
            saveProject(); render(); setStatus(error.message, 'error'); showToast(error.message, 'error');
        }
    }

    function recordEndlessSegmentTake(run, segmentIndex, videoId, src) {
        C.recordEndlessSegmentTake(state.project, run, segmentIndex, videoId, src);
    }

    function pollEndlessActive() {
        var run = endlessState();
        if (!run.active_job) return;
        var job = C.copy(run.active_job);
        var safeUrls = C.videoJobUrls(job.video_id);
        if (!safeUrls) {
            run.status = 'failed'; run.error = 'Refusing to poll an invalid endless video job id.'; saveProject(); render(); return;
        }
        job.status_url = safeUrls.status_url; job.result_url = safeUrls.result_url;
        var token = ++state.endlessPollToken;
        function poll() {
            var current = endlessState();
            if (token !== state.endlessPollToken || !current.active_job || current.active_job.video_id !== job.video_id) return;
            fetch(job.status_url, { cache: 'no-store' }).then(function (response) {
                if (!response.ok) throw new Error('status HTTP ' + response.status);
                return response.json();
            }).then(function (status) {
                var activeRun = endlessState();
                if (activeRun.active_job && activeRun.active_job.not_found_count) {
                    activeRun.active_job.not_found_count = 0; saveProject();
                }
                var phase = String(status.message || status.phase || 'H3 running');
                setStatus(job.video_id + ' · endless ' + (job.segment_index + 1) + '/' + activeRun.segment_durations.length + ' · ' + phase, 'live');
                if (status.state === 'failed' || status.state === 'error') throw new Error(phase);
                if (status.state !== 'done') { setTimeout(poll, 750); return null; }
                return fetch(job.result_url, { cache: 'no-store' }).then(function (response) {
                    if (!response.ok) throw new Error('result HTTP ' + response.status);
                    return response.json();
                });
            }).then(function (manifest) {
                if (!manifest) return;
                var activeRun = endlessState();
                var artifact = String(manifest.mp4_url || manifest.artifact_path || '');
                var src = manifest.mp4_url || (artifact ? '/out/' + encodeURIComponent(job.video_id) + '/' + encodeURIComponent(artifact.split('/').pop()) : '');
                if (!src) throw new Error('completed video manifest has no playable MP4');
                if (activeRun.completed_job_ids.length !== Number(job.segment_index))
                    throw new Error('Endless completion order no longer matches the saved plan; refusing duplicate submission.');
                recordEndlessSegmentTake(activeRun, Number(job.segment_index), job.video_id, src);
                activeRun.completed_job_ids.push(job.video_id);
                activeRun.completed_output_paths.push(src);
                activeRun.active_job = null; activeRun.error = ''; activeRun.updated_at = new Date().toISOString();
                if (activeRun.status === 'stopping') {
                    activeRun.status = 'stopped'; activeRun.stopped_at = activeRun.updated_at;
                } else if (activeRun.completed_job_ids.length >= activeRun.segment_durations.length) {
                    activeRun.status = 'completed';
                } else {
                    activeRun.status = 'running';
                }
                saveProject(); render();
                if (activeRun.status === 'completed') {
                    setStatus('Endless story complete · ' + activeRun.completed_job_ids.length + ' segments ready', '');
                    showToast('Endless story complete', '');
                } else if (activeRun.status === 'stopped') {
                    setStatus('Endless story stopped after ' + activeRun.completed_job_ids.length + ' completed segment(s)', '');
                } else {
                    setStatus(job.video_id + ' ready · preparing the next continuation', 'live');
                    setTimeout(submitNextEndlessSegment, 0);
                }
            }).catch(function (error) {
                if (token !== state.endlessPollToken) return;
                if (/status HTTP 404/.test(error.message)) {
                    var missing = endlessState();
                    if (!missing.active_job) return;
                    missing.active_job.not_found_count = Math.max(0, Number(missing.active_job.not_found_count) || 0) + 1;
                    missing.updated_at = new Date().toISOString(); saveProject();
                    if (missing.active_job.not_found_count < 80) { setTimeout(poll, 750); return; }
                    error = new Error('status remained unavailable for 60 seconds; preserving ' + job.video_id + ' for Queue inspection');
                }
                var failed = endlessState(); failed.status = 'failed'; failed.error = 'Endless segment ' + (Number(job.segment_index) + 1) + ' failed: ' + error.message; failed.updated_at = new Date().toISOString();
                saveProject(); render(); setStatus(failed.error, 'error'); showToast(failed.error, 'error');
            });
        }
        setTimeout(poll, 300);
    }

    function resumeEndless() {
        var run = endlessState();
        if (['submitting', 'running', 'stopping'].indexOf(run.status) < 0) return;
        try {
            assertEndlessResumeSafe(run);
            if (run.active_job) {
                setStatus('Resuming status tracking for ' + run.active_job.video_id + '…', 'live');
                pollEndlessActive();
            } else if (run.status === 'stopping') {
                run.status = 'stopped'; run.stopped_at = new Date().toISOString(); saveProject(); render(); setStatus('Endless story stopped', '');
            } else if (run.status === 'submitting') {
                run.status = 'failed';
                run.error = 'The browser reloaded before the server returned a job id. Automatic retry and cancellation are blocked to avoid submitting a duplicate or pretending the server job was cancelled; verify the Queue, then reset this run.';
                saveProject(); render(); setStatus(run.error, 'error');
            } else {
                submitNextEndlessSegment();
            }
        } catch (error) {
            run.status = 'failed'; run.error = error.message; run.updated_at = new Date().toISOString(); saveProject(); render(); setStatus(error.message, 'error');
        }
    }

    function renderShot() {
        var shot = selectedShot();
        if (endlessChainShotLocked(shot)) {
            showToast('The active endless chain owns this shot. Select another shot to queue an independent manual render.', 'error'); return;
        }
        try {
            var resolvedAttention = resolvedH3Attention(shot);
            if (shot.attention_backend !== resolvedAttention && !baseShotMutationBlocked(shot)) {
                shot.attention_backend = resolvedAttention;
                saveProject('Attention fell back to cU-DNN for this GPU');
                render();
            }
            var requestShot = shot;
            if (shot.attention_backend !== resolvedAttention) {
                requestShot = C.copy(shot); requestShot.attention_backend = resolvedAttention;
            }
            var request = C.renderRequest(requestShot);
            if (!window.confirm('Queue one ' + C.secondsText(shot.duration_seconds) + '-second H3 take at ' + shot.width + '×' + shot.height + '? This starts GPU work.')) return;
            setStatus('Submitting H3 take…', 'live');
            SerenityAPI.postVideo(request).then(function (job) {
                if (!job || !(job.video_id || job.prompt_id)) throw new Error('server did not return a video job id');
                var id = String(job.video_id || job.prompt_id); shot.take_job_ids.push(id); shot.take_states.push('queued'); shot.take_output_paths.push(''); shot.selected_take = shot.take_job_ids.length - 1; shot.status = 'Queued'; saveProject();
                if (typeof QueueTab !== 'undefined') { QueueTab.init(); QueueTab.registerPending({ promptId: id, prompt: shot.brief || shot.shot_description, model: 'MiniMax H3', queuedAt: Date.now(), batchLabel: state.project.title + ' · ' + shot.title }); }
                setStatus('Queued ' + id + ' · waiting for H3 runtime', 'live'); render(); pollVideo(job, shot.id);
            }).catch(function (error) { setStatus('H3 submission failed: ' + error.message, 'error'); showToast('H3 submission failed: ' + error.message, 'error'); });
        } catch (error) { setStatus(error.message, 'error'); showToast(error.message, 'error'); }
    }

    function pollVideo(job, shotId) {
        var videoId = String(job.video_id || job.prompt_id || '');
        var statusUrl = String(job.status_url || ('/out/' + encodeURIComponent(videoId) + '/status.json'));
        var resultUrl = String(job.result_url || ('/out/' + encodeURIComponent(videoId) + '/result.json'));
        var token = ++state.videoPollToken;
        function findShot() { return state.project.shots.find(function (shot) { return shot.id === shotId; }); }
        function poll() {
            if (token !== state.videoPollToken) return;
            fetch(statusUrl, { cache: 'no-store' }).then(function (response) {
                if (!response.ok) throw new Error('status HTTP ' + response.status);
                return response.json();
            }).then(function (status) {
                var shot = findShot(); if (!shot) return null;
                var take = shot.take_job_ids.indexOf(videoId); var phase = String(status.message || status.phase || 'H3 running');
                shot.status = phase; if (take >= 0) shot.take_states[take] = status.state || 'running'; saveProject(); setStatus(videoId + ' · ' + phase, 'live');
                if (status.state === 'failed' || status.state === 'error') throw new Error(phase);
                if (status.state !== 'done') { setTimeout(poll, 750); return null; }
                return fetch(resultUrl, { cache: 'no-store' }).then(function (response) { if (!response.ok) throw new Error('result HTTP ' + response.status); return response.json(); });
            }).then(function (manifest) {
                if (!manifest) return;
                var shot = findShot(); if (!shot) return;
                var artifact = String(manifest.mp4_url || manifest.artifact_path || ''); var src = manifest.mp4_url || (artifact ? '/out/' + encodeURIComponent(videoId) + '/' + encodeURIComponent(artifact.split('/').pop()) : '');
                if (!src) throw new Error('completed video manifest has no playable MP4');
                var take = shot.take_job_ids.indexOf(videoId); if (take >= 0) { shot.take_states[take] = 'done'; shot.take_output_paths[take] = src; shot.selected_take = take; }
                shot.status = 'Ready'; shot.output_path = src; saveProject(); setStatus(videoId + ' ready', ''); showToast(videoId + ' ready — playing in the monitor', ''); render();
            }).catch(function (error) {
                if (token !== state.videoPollToken) return;
                if (/status HTTP 404/.test(error.message)) { setTimeout(poll, 750); return; }
                var shot = findShot(); if (shot) shot.status = 'Failed'; saveProject(); setStatus('H3 generation failed: ' + error.message, 'error'); showToast('H3 generation failed: ' + error.message, 'error'); render();
            });
        }
        setTimeout(poll, 300);
    }

    function loadReadiness() {
        fetch('/v1/video', { cache: 'no-store' }).then(function (response) { if (!response.ok) throw new Error('HTTP ' + response.status); return response.json(); })
            .then(function (data) { state.readiness = data; render(); })
            .catch(function () { state.readiness = null; render(); });
    }

    function init() {
        if (state.initialized) return;
        state.initialized = true; state.project = loadProject(); state.selectedShotId = state.project.shots[0].id;
        render(); loadReadiness(); resumeEndless();
    }

    return { init: init, render: render, state: state, contracts: C };
})();
