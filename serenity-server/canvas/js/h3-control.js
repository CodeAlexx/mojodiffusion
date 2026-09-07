// @ts-nocheck -- mirrored to js/h3-control.js for the browser build.
"use strict";
/** Dedicated MiniMax-H3 Fun ControlNet screen. */
var H3ControlTab = (function () {
    'use strict';
    var initialized = false;
    var wsBound = false;
    var state = {
        uploads: {
            control: { path: '', url: '', file: null, objectUrl: '', meta: null },
            source: { path: '', url: '', file: null, objectUrl: '', meta: null },
            mask: { path: '', url: '', file: null, objectUrl: '', meta: null }
        },
        extraControls: [],
        nextControlId: 2,
        activePreview: 'control',
        compareMode: 'single',
        loras: [],
        availableLoras: [],
        uploadsInFlight: 0,
        clip: '',
        videoVae: '',
        audioVae: '',
        promptId: '',
        ownExecuting: false,
        running: false,
        resultUrl: '',
        resultFile: null,
        resultMeta: null,
        resultParams: null,
        lastParams: null,
        queueNumber: 0,
        startedAt: 0,
        samplingStartedAt: 0,
        elapsedTimer: null,
        sampleStep: 0,
        sampleMax: 0,
        activePhase: 'ready',
        recent: [],
        modelsLoaded: false,
        syncingFromWorkflow: false,
        pendingWorkflowParams: null,
        connectionInterrupted: false
    };
    var els = {};
    var h3ResolutionPresets = {
        '1536x672': { width: 1536, height: 672 },
        '1344x768': { width: 1344, height: 768 },
        '1024x768': { width: 1024, height: 768 },
        '768x768': { width: 768, height: 768 },
        '768x1024': { width: 768, height: 1024 },
        '768x1344': { width: 768, height: 1344 }
    };
    function h3Resolution(value) {
        var key = String(value || '');
        var preset = h3ResolutionPresets[key] || h3ResolutionPresets['1344x768'];
        return { key: preset.width + 'x' + preset.height, width: preset.width, height: preset.height };
    }
    function uploadMarkup(role, title, note, accept, refNumber) {
        return '<div class="h3ct-upload" id="h3ct-' + role + '-drop" data-role="' + role + '">' +
            '<input id="h3ct-' + role + '-input" type="file" accept="' + accept + '" hidden>' +
            '<div class="h3ct-upload-icon"><i data-lucide="upload-cloud"></i></div>' +
            '<div class="h3ct-upload-copy"><strong>' + title + (refNumber ? ' <em>#' + refNumber + '</em>' : '') + '</strong><span>' + note + '</span></div>' +
            '<button class="h3ct-small-button" type="button">Choose</button>' +
            '<div class="h3ct-upload-status" id="h3ct-' + role + '-status">No file</div>' +
            '</div>';
    }
    function render() {
        var panel = document.getElementById('panel-h3-ct');
        if (!panel)
            return;
        panel.innerHTML =
            '<div class="h3ct-layout">' +
                '<header class="h3ct-header">' +
                '<div><div class="h3ct-eyebrow">Native Mojo H3 inference</div>' +
                '<h1>MiniMax H3 Control</h1>' +
                '<p>Union ControlNet for prepared motion maps or native Canny video.</p></div>' +
                '<div class="h3ct-fixed-contract">' +
                '<span>24 FPS</span><span>CK-INT8</span><span>Guidance 1.0</span><span>Euler / Normal</span><span>Audio on</span>' +
                '</div>' +
                '</header>' +
                '<div id="h3ct-error" class="h3ct-error" role="alert"></div>' +
                '<div class="h3ct-body">' +
                '<div class="h3ct-controls">' +
                '<section class="h3ct-card">' +
                '<div class="h3ct-card-heading"><span class="h3ct-step">01</span><div><h2>Model & control</h2><p>Official dense Alibaba checkpoint for the current full-embed H3 base.</p></div></div>' +
                '<label class="h3ct-field"><span>H3 base checkpoint</span><select id="h3ct-model"><option>Loading…</option></select></label>' +
                '<label class="h3ct-field"><span>H3 Union ControlNet</span><select id="h3ct-controlnet"><option>Loading…</option></select></label>' +
                '<div id="h3ct-model-note" class="h3ct-note">Reading installed native Mojo H3 models…</div>' +
                '<div class="h3ct-track-heading"><strong>Control track <span id="h3ct-control-count">1</span></strong><span>Selection order is control order</span></div>' +
                uploadMarkup('control', 'Control', 'Drop video or prepared map · required', 'video/*,image/gif,image/png,image/jpeg,image/webp', 1) +
                '<div class="h3ct-grid-2">' +
                '<label class="h3ct-field"><span>Input handling</span><select id="h3ct-preprocessor"><option value="prepared">Prepared map</option><option value="canny">Native Canny</option></select></label>' +
                '<label class="h3ct-field"><span>Canvas fit</span><select id="h3ct-resize"><option value="crop">Center crop</option><option value="pad">Fit + pad</option><option value="stretch">Stretch</option></select></label>' +
                '</div>' +
                '<div id="h3ct-preprocess-note" class="h3ct-note">Prepared accepts Canny, Depth, HED, MLSD, or Pose maps. Unavailable raw preprocessors are not faked.</div>' +
                '<div id="h3ct-canny-controls" class="h3ct-grid-2 h3ct-hidden">' +
                '<label class="h3ct-field"><span>Canny low</span><input id="h3ct-canny-low" type="number" min="0" max="254" step="1" value="100"></label>' +
                '<label class="h3ct-field"><span>Canny high</span><input id="h3ct-canny-high" type="number" min="1" max="255" step="1" value="200"></label>' +
                '</div>' +
                '<div class="h3ct-grid-3">' +
                '<label class="h3ct-field"><span>Strength</span><input id="h3ct-strength" type="number" min="-4" max="4" step="0.05" value="1"></label>' +
                '<label class="h3ct-field"><span>Start</span><input id="h3ct-start" type="number" min="0" max="1" step="0.05" value="0"></label>' +
                '<label class="h3ct-field"><span>End</span><input id="h3ct-end" type="number" min="0" max="1" step="0.05" value="1"></label>' +
                '</div>' +
                '<div id="h3ct-extra-controls" class="h3ct-extra-controls"></div>' +
                '<button id="h3ct-add-control" class="h3ct-add-track" type="button"><i data-lucide="plus"></i>Add control track</button>' +
                '<div class="h3ct-subpanel h3ct-lora-panel">' +
                '<div class="h3ct-subpanel-head"><div><strong>H3 LoRAs</strong><span>Ordered overlays with independent strengths</span></div><span id="h3ct-lora-count" class="h3ct-count-badge">0</span></div>' +
                '<select id="h3ct-lora-picker" class="h3ct-wide-select"><option value="" disabled selected>Loading LoRAs…</option></select>' +
                '<div id="h3ct-lora-list" class="h3ct-lora-list"></div>' +
                '</div>' +
                '</section>' +
                '<section class="h3ct-card">' +
                '<div class="h3ct-card-heading"><span class="h3ct-step">02</span><div><h2>Optional inpaint</h2><p>White mask repaints; black mask preserves the source.</p></div>' +
                '<label class="h3ct-switch"><input id="h3ct-inpaint-toggle" type="checkbox"><span></span></label></div>' +
                '<div id="h3ct-inpaint-fields" class="h3ct-hidden h3ct-stack">' +
                uploadMarkup('source', 'Source video or image', 'Aligned with the same 24 FPS crop', 'video/*,image/*') +
                uploadMarkup('mask', 'Mask image or video', 'White = repaint · black = preserve', 'video/*,image/*') +
                '<label class="h3ct-check"><input id="h3ct-invert-mask" type="checkbox"><span>Invert mask</span></label>' +
                '</div>' +
                '</section>' +
                '<section class="h3ct-card">' +
                '<div class="h3ct-card-heading"><span class="h3ct-step">03</span><div><h2>Prompt</h2><p>Describe the final scene, subject, camera, motion, and audio.</p></div></div>' +
                '<label class="h3ct-field"><textarea id="h3ct-prompt" rows="7" placeholder="A cinematic shot…"></textarea></label>' +
                '</section>' +
                '<section class="h3ct-card">' +
                '<div class="h3ct-card-heading"><span class="h3ct-step">04</span><div><h2>Output</h2><p>Whole seconds; frames and H3 padding are automatic.</p></div></div>' +
                '<div class="h3ct-grid-2">' +
                '<label class="h3ct-field"><span>Resolution</span><select id="h3ct-resolution">' +
                '<option value="1536x672">21:9 · 1536×672</option>' +
                '<option value="1344x768" selected>16:9 · 1344×768</option>' +
                '<option value="1024x768">4:3 · 1024×768</option>' +
                '<option value="768x768">1:1 · 768×768</option>' +
                '<option value="768x1024">3:4 · 768×1024</option>' +
                '<option value="768x1344">9:16 · 768×1344</option>' +
                '</select></label>' +
                '<label class="h3ct-field"><span>Seconds</span><input id="h3ct-duration" type="number" min="5" max="15" step="1" value="5"></label>' +
                '</div>' +
                '<div class="h3ct-contract-row">' +
                '<div><span>Output</span><strong id="h3ct-output-frames">120 frames</strong></div>' +
                '<div><span>H3 internal</span><strong id="h3ct-internal-frames">124 frames</strong></div>' +
                '<div><span>Codec / color</span><strong>H.264 · 8-bit 4:2:0</strong></div>' +
                '</div>' +
                '<div class="h3ct-grid-3">' +
                '<label class="h3ct-field"><span>Steps</span><input id="h3ct-steps" type="number" min="1" max="100" step="1" value="40"></label>' +
                '<label class="h3ct-field"><span>Seed</span><div class="h3ct-inline"><input id="h3ct-seed" type="number" step="1" value="-1"><button id="h3ct-random-seed" class="h3ct-icon-button" type="button" title="Random seed"><i data-lucide="dices"></i></button></div></label>' +
                '<label class="h3ct-field"><span>Format</span><select id="h3ct-format"><option value="mp4">MP4</option><option value="mov">MOV</option><option value="mkv">MKV</option></select></label>' +
                '</div>' +
                '<div class="h3ct-note">Six supported H3 presets only. Long control video trims; short video visibly holds its last frame.</div>' +
                '<details class="h3ct-advanced" id="h3ct-advanced">' +
                '<summary>Advanced workflow controls</summary>' +
                '<div class="h3ct-advanced-body">' +
                '<label class="h3ct-field"><span>Text encoder</span><select id="h3ct-clip"><option>Loading…</option></select></label>' +
                '<div class="h3ct-grid-2"><label class="h3ct-field"><span>Video VAE</span><select id="h3ct-video-vae"><option>Loading…</option></select></label><label class="h3ct-field"><span>Audio VAE</span><select id="h3ct-audio-vae"><option>Loading…</option></select></label></div>' +
                '<div class="h3ct-grid-2"><label class="h3ct-field"><span>Streaming window</span><input id="h3ct-window" type="number" min="2" max="8" step="1" value="2"></label><label class="h3ct-field"><span>Prefetch blocks</span><input id="h3ct-prefetch" type="number" min="1" max="4" step="1" value="1"></label></div>' +
                '<div class="h3ct-grid-2"><label class="h3ct-field"><span>Video flow shift</span><input id="h3ct-video-shift" type="number" min="0" max="30" step="0.1" value="12"></label><label class="h3ct-field"><span>Audio flow shift</span><input id="h3ct-audio-shift" type="number" min="0" max="30" step="0.1" value="3"></label></div>' +
                '<label class="h3ct-field"><span>Filename prefix</span><input id="h3ct-filename" type="text" value="minimax_h3_controlnet_union"></label>' +
                '<div class="h3ct-workflow-actions"><button id="h3ct-open-workflow" type="button">Open live workflow</button><button id="h3ct-copy-workflow" type="button">Copy API JSON</button></div>' +
                '</div>' +
                '</details>' +
                '</section>' +
                '</div>' +
                '<aside class="h3ct-stage">' +
                '<div class="h3ct-stage-head"><div><span>Control deck</span><strong id="h3ct-control-name">No control video</strong></div><span id="h3ct-live-dot" class="h3ct-live-dot empty"></span></div>' +
                '<div class="h3ct-deck-toolbar"><div id="h3ct-preview-tabs" class="h3ct-preview-tabs"><button class="active" data-preview="control">Control #1</button></div><div class="h3ct-view-modes"><button id="h3ct-view-single" class="active" type="button">Single</button><button id="h3ct-view-split" type="button" disabled>Split</button></div></div>' +
                '<div id="h3ct-control-preview" class="h3ct-preview"><div class="h3ct-preview-empty"><i data-lucide="scan-line"></i><strong>Drop a control video</strong><span>The exact uploaded media appears here.</span></div></div>' +
                '<div id="h3ct-transport" class="h3ct-transport">' +
                '<button id="h3ct-frame-back" type="button" title="Back one frame">−1f</button><button id="h3ct-deck-play" type="button" title="Play or pause"><i data-lucide="play"></i></button><button id="h3ct-frame-forward" type="button" title="Forward one frame">+1f</button>' +
                '<input id="h3ct-scrubber" type="range" min="0" max="1000" value="0" step="1" aria-label="Preview time"><span id="h3ct-timecode">00:00 / 00:00</span>' +
                '</div>' +
                '<div id="h3ct-media-meta" class="h3ct-media-meta">Waiting for media metadata</div>' +
                '<div class="h3ct-route">' +
                '<span>Control map</span><i data-lucide="arrow-right"></i><span>VAE align</span><i data-lucide="arrow-right"></i><span>Layers 0 · 10 · 20 · 30 · 40</span>' +
                '</div>' +
                '<video id="h3ct-result" class="h3ct-hidden" playsinline></video>' +
                '<div id="h3ct-result-card" class="h3ct-result-card h3ct-hidden">' +
                '<div class="h3ct-result-actions"><button id="h3ct-download" type="button"><i data-lucide="download"></i>Download</button><button id="h3ct-reuse" type="button"><i data-lucide="rotate-ccw"></i>Reuse settings</button><button id="h3ct-timeline" type="button"><i data-lucide="film"></i>Video Edit</button></div>' +
                '</div>' +
                '<section class="h3ct-recent"><div class="h3ct-stage-head"><div><span>Recent video outputs</span><strong>Mojo inference outputs</strong></div><button id="h3ct-refresh-recent" class="h3ct-text-button" type="button">Refresh</button></div><div id="h3ct-recent-list" class="h3ct-recent-list"><span>Loading recent outputs…</span></div></section>' +
                '<div class="h3ct-run-card">' +
                '<div id="h3ct-phase-track" class="h3ct-phase-track"><span data-phase="prepare">Prepare</span><span data-phase="control">Control</span><span data-phase="sample">Sample</span><span data-phase="decode">Decode</span><span data-phase="save">Save</span></div>' +
                '<div class="h3ct-run-status"><span id="h3ct-status">Ready</span><span id="h3ct-progress-value">0%</span></div>' +
                '<div class="h3ct-progress"><span id="h3ct-progress-bar"></span></div>' +
                '<div class="h3ct-run-metrics"><span id="h3ct-step-label">No sampling step</span><span id="h3ct-elapsed">Elapsed 00:00</span><span id="h3ct-eta">ETA —</span></div>' +
                '<div class="h3ct-actions"><button id="h3ct-generate" class="h3ct-generate" type="button"><i data-lucide="play"></i><span>Generate H3 CT</span></button><button id="h3ct-cancel" class="h3ct-cancel" type="button" disabled>Cancel</button></div>' +
                '<div class="h3ct-shortcut">Ctrl + Enter generates from this tab</div>' +
                '</div>' +
                '</aside>' +
                '</div>' +
                '</div>';
        if (typeof lucide !== 'undefined')
            lucide.createIcons();
    }
    function cacheElements() {
        [
            'model', 'controlnet', 'model-note', 'preprocessor', 'preprocess-note',
            'resize', 'canny-controls', 'canny-low', 'canny-high', 'strength',
            'start', 'end', 'control-count', 'extra-controls', 'add-control',
            'lora-picker', 'lora-list', 'lora-count',
            'inpaint-toggle', 'inpaint-fields', 'invert-mask',
            'prompt', 'resolution', 'duration', 'output-frames',
            'internal-frames', 'steps', 'seed', 'random-seed', 'format', 'error',
            'clip', 'video-vae', 'audio-vae', 'window', 'prefetch', 'video-shift',
            'audio-shift', 'filename', 'open-workflow', 'copy-workflow',
            'control-preview', 'control-name', 'live-dot', 'preview-tabs',
            'view-single', 'view-split', 'transport', 'frame-back', 'deck-play',
            'frame-forward', 'scrubber', 'timecode', 'media-meta',
            'result-card', 'result', 'download', 'reuse', 'timeline',
            'refresh-recent', 'recent-list', 'phase-track', 'status',
            'progress-value', 'progress-bar', 'step-label', 'elapsed', 'eta',
            'generate', 'cancel'
        ].forEach(function (name) {
            els[name.replace(/-([a-z])/g, function (_m, c) { return c.toUpperCase(); })] =
                document.getElementById('h3ct-' + name);
        });
    }
    function choices(info, nodeName, inputName) {
        var node = info && info[nodeName];
        var spec = node && node.input && node.input.required && node.input.required[inputName];
        return spec && Array.isArray(spec[0]) ? spec[0].filter(function (item) {
            return item && item !== '(no models found)';
        }) : [];
    }
    function fillSelect(select, values, preferred, emptyLabel) {
        select.innerHTML = '';
        if (!values.length) {
            var empty = document.createElement('option');
            empty.value = '';
            empty.textContent = emptyLabel;
            select.appendChild(empty);
            select.disabled = true;
            select.title = '';
            return '';
        }
        select.disabled = false;
        var labelCounts = {};
        values.forEach(function (value) {
            var label = ModelUtils.displayModelName(value);
            labelCounts[label] = (labelCounts[label] || 0) + 1;
        });
        values.forEach(function (value) {
            var option = document.createElement('option');
            var label = ModelUtils.displayModelName(value);
            if (labelCounts[label] > 1) {
                var pieces = String(value).split('/');
                label += ' \u00b7 ' + (pieces.length > 1
                    ? pieces.slice(0, -1).join(' / ') : 'direct file');
            }
            option.value = value;
            option.textContent = label;
            option.title = value;
            select.appendChild(option);
        });
        var selected = values.indexOf(preferred) >= 0 ? preferred : values[0];
        select.value = selected;
        select.title = selected;
        return selected;
    }
    function updateSelectTitle(select) {
        if (select)
            select.title = select.value || '';
    }
    function loadModels() {
        fetch('/object_info', { cache: 'no-store' }).then(function (response) {
            if (!response.ok)
                throw new Error('HTTP ' + response.status);
            return response.json();
        }).then(function (info) {
            SerenityH3WorkflowRegistry.install(info);
            var bases = choices(info, 'MiniMaxH3Loader', 'unet_name').filter(function (name) {
                return /minimax.*h3/i.test(name);
            });
            bases.sort(function (a, b) {
                var ap = /fl2va.*transformer_int8_rowscale/i.test(a) ? 0 : 1;
                var bp = /fl2va.*transformer_int8_rowscale/i.test(b) ? 0 : 1;
                return ap - bp || a.localeCompare(b);
            });
            var allControls = choices(info, 'MiniMaxH3FunControlNetLoader', 'control_net_name');
            var controls = allControls.filter(function (name) {
                return /(?:minimax.*h3.*(?:control|union|fun)|h3.*(?:control|union|fun))/i.test(name) &&
                    !/(?:int8.*convrot|convrot.*int8)/i.test(name);
            });
            var vaes = choices(info, 'VAELoader', 'vae_name');
            var clips = choices(info, 'MiniMaxH3TextEncode', 'clip_name').filter(function (name) {
                return /minimax.*h3|qwen3vl/i.test(name);
            });
            var videoVaes = vaes.filter(function (name) {
                return /minimax.*h3.*video.*vae/i.test(name);
            });
            var audioVaes = vaes.filter(function (name) {
                return /minimax.*h3.*audio.*vae/i.test(name);
            });
            var storedModel = localStorage.getItem('sf-h3ct-model') || '';
            var storedControl = localStorage.getItem('sf-h3ct-controlnet') || '';
            fillSelect(els.model, bases, storedModel, 'No MiniMax H3 base installed');
            fillSelect(els.controlnet, controls, storedControl, 'No supported H3 ControlNet installed');
            state.clip = fillSelect(els.clip, clips, localStorage.getItem('sf-h3ct-clip') || '', 'No H3 text encoder installed');
            state.videoVae = fillSelect(els.videoVae, videoVaes, localStorage.getItem('sf-h3ct-video-vae') || '', 'No H3 video VAE installed');
            state.audioVae = fillSelect(els.audioVae, audioVaes, localStorage.getItem('sf-h3ct-audio-vae') || '', 'No H3 audio VAE installed');
            state.modelsLoaded = true;
            if (!controls.length) {
                els.modelNote.innerHTML = 'Official dense ControlNet is not installed. <a href="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union" target="_blank" rel="noreferrer">Open the Alibaba model page</a>.';
                els.modelNote.classList.add('h3ct-warning');
            }
            else if (!state.videoVae || !state.audioVae) {
                els.modelNote.textContent = 'MiniMax H3 video or audio VAE is missing.';
                els.modelNote.classList.add('h3ct-warning');
            }
            else {
                els.modelNote.textContent = 'Official dense ControlNet installed; Serenity row-scale INT8 and CK-INT8 run the H3 base.';
                els.modelNote.classList.remove('h3ct-warning');
            }
            loadLoras();
            updateReadiness();
            if (state.pendingWorkflowParams) {
                var pending = state.pendingWorkflowParams;
                state.pendingWorkflowParams = null;
                applyWorkflowParams(pending);
            }
        }).catch(function (error) {
            state.modelsLoaded = false;
            showError('Could not read native workflow models: ' + error.message);
            els.modelNote.textContent = 'Model registry unavailable';
            updateReadiness();
        });
    }
    function escapeHtml(value) {
        var div = document.createElement('div');
        div.textContent = String(value == null ? '' : value);
        return div.innerHTML;
    }
    function formatBytes(value) {
        var bytes = Number(value || 0);
        if (bytes < 1024)
            return bytes + ' B';
        if (bytes < 1048576)
            return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / 1048576).toFixed(1) + ' MB';
    }
    function loadLoras() {
        // The native dense ControlNet runner intentionally rejects base-model
        // LoRA overlays until their combined memory/quality gate exists.
        state.availableLoras = [];
        state.loras = [];
        els.loraPicker.innerHTML =
            '<option value="" disabled selected>Disabled for native H3 ControlNet</option>';
        els.loraPicker.disabled = true;
        renderLoras();
    }
    function renderLoras() {
        els.loraCount.textContent = String(state.loras.length);
        els.loraList.innerHTML = '';
        state.loras.forEach(function (lora, index) {
            if (!Number.isFinite(Number(lora.strength)))
                lora.strength = 1;
            var row = document.createElement('div');
            row.className = 'h3ct-lora-row' + (lora.enabled === false ? ' disabled' : '');
            row.innerHTML =
                '<div class="h3ct-lora-main"><button class="h3ct-lora-toggle' + (lora.enabled === false ? '' : ' on') + '" type="button" aria-label="Enable LoRA"></button>' +
                    '<span title="' + escapeHtml(lora.name) + '">' + escapeHtml(lora.name) + '</span>' +
                    '<div class="h3ct-order-buttons"><button type="button" data-order="up" aria-label="Move LoRA up">↑</button><button type="button" data-order="down" aria-label="Move LoRA down">↓</button></div>' +
                    '<button class="h3ct-lora-remove" type="button" aria-label="Remove LoRA">×</button></div>' +
                    '<label class="h3ct-lora-strength"><span>Strength</span><input type="range" min="-1" max="2" step="0.05" value="' + Number(lora.strength).toFixed(2) + '"><input type="number" min="-10" max="10" step="0.05" value="' + Number(lora.strength).toFixed(2) + '"></label>';
            var range = row.querySelector('input[type="range"]');
            var number = row.querySelector('input[type="number"]');
            range.addEventListener('input', function () {
                lora.strength = Number(this.value);
                number.value = Number(lora.strength).toFixed(2);
                saveDraft();
            });
            number.addEventListener('change', function () {
                var value = Math.max(-10, Math.min(10, Number(this.value) || 0));
                lora.strength = value;
                this.value = value.toFixed(2);
                range.value = String(Math.max(-1, Math.min(2, value)));
                saveDraft();
            });
            row.querySelector('.h3ct-lora-toggle').addEventListener('click', function () {
                lora.enabled = lora.enabled === false;
                renderLoras();
                saveDraft();
            });
            row.querySelector('.h3ct-lora-remove').addEventListener('click', function () {
                state.loras.splice(index, 1);
                renderLoras();
                saveDraft();
            });
            row.querySelectorAll('[data-order]').forEach(function (button) {
                button.addEventListener('click', function () {
                    var target = this.dataset.order === 'up' ? index - 1 : index + 1;
                    if (target < 0 || target >= state.loras.length)
                        return;
                    var moved = state.loras.splice(index, 1)[0];
                    state.loras.splice(target, 0, moved);
                    renderLoras();
                    saveDraft();
                });
            });
            els.loraList.appendChild(row);
        });
    }
    function makeControlTrack(id) {
        return {
            id: id,
            upload: { path: '', url: '', file: null, objectUrl: '', meta: null },
            preprocessor: 'prepared', resizeMode: 'crop', cannyLow: 100,
            cannyHigh: 200, strength: 1, startPercent: 0, endPercent: 1,
            enabled: true
        };
    }
    function trackForRole(role) {
        var match = /^control-(\d+)$/.exec(role);
        if (!match)
            return null;
        var id = Number(match[1]);
        return state.extraControls.find(function (track) { return track.id === id; }) || null;
    }
    function slotForRole(role) {
        var track = trackForRole(role);
        return track ? track.upload : state.uploads[role];
    }
    function renderExtraControls() {
        els.extraControls.innerHTML = '';
        state.extraControls.forEach(function (track) {
            var role = 'control-' + track.id;
            var card = document.createElement('div');
            card.className = 'h3ct-control-track' + (track.enabled === false ? ' disabled' : '');
            card.dataset.trackId = String(track.id);
            card.innerHTML =
                '<div class="h3ct-control-track-title"><div><button class="h3ct-track-toggle' + (track.enabled === false ? '' : ' on') + '" type="button" aria-label="Enable control track"></button><strong>Control #' + track.id + '</strong><span>Experimental stacked control</span></div><button class="h3ct-remove-track" type="button">Remove</button></div>' +
                    uploadMarkup(role, 'Control', 'Video or prepared map', 'video/*,image/gif,image/png,image/jpeg,image/webp', track.id) +
                    '<div class="h3ct-grid-2"><label class="h3ct-field"><span>Input handling</span><select data-field="preprocessor"><option value="prepared">Prepared map</option><option value="canny">Native Canny</option></select></label><label class="h3ct-field"><span>Canvas fit</span><select data-field="resizeMode"><option value="crop">Center crop</option><option value="pad">Fit + pad</option><option value="stretch">Stretch</option></select></label></div>' +
                    '<div class="h3ct-grid-2 h3ct-extra-canny' + (track.preprocessor === 'canny' ? '' : ' h3ct-hidden') + '"><label class="h3ct-field"><span>Canny low</span><input data-field="cannyLow" type="number" min="0" max="254" step="1" value="' + track.cannyLow + '"></label><label class="h3ct-field"><span>Canny high</span><input data-field="cannyHigh" type="number" min="1" max="255" step="1" value="' + track.cannyHigh + '"></label></div>' +
                    '<div class="h3ct-grid-3"><label class="h3ct-field"><span>Strength</span><input data-field="strength" type="number" min="-4" max="4" step="0.05" value="' + track.strength + '"></label><label class="h3ct-field"><span>Start</span><input data-field="startPercent" type="number" min="0" max="1" step="0.05" value="' + track.startPercent + '"></label><label class="h3ct-field"><span>End</span><input data-field="endPercent" type="number" min="0" max="1" step="0.05" value="' + track.endPercent + '"></label></div>';
            els.extraControls.appendChild(card);
            card.querySelector('[data-field="preprocessor"]').value = track.preprocessor;
            card.querySelector('[data-field="resizeMode"]').value = track.resizeMode;
            bindUpload(role);
            card.querySelectorAll('[data-field]').forEach(function (input) {
                input.addEventListener('change', function () {
                    var field = this.dataset.field;
                    track[field] = ['preprocessor', 'resizeMode'].indexOf(field) >= 0 ? this.value : Number(this.value);
                    if (field === 'preprocessor')
                        card.querySelector('.h3ct-extra-canny').classList.toggle('h3ct-hidden', track.preprocessor !== 'canny');
                    renderPreviewDeck();
                    updateReadiness();
                    saveDraft();
                });
            });
            card.querySelector('.h3ct-track-toggle').addEventListener('click', function () {
                track.enabled = track.enabled === false;
                renderExtraControls();
                renderPreviewDeck();
                updateReadiness();
                saveDraft();
            });
            card.querySelector('.h3ct-remove-track').addEventListener('click', function () {
                if (track.upload.objectUrl)
                    URL.revokeObjectURL(track.upload.objectUrl);
                state.extraControls = state.extraControls.filter(function (item) { return item.id !== track.id; });
                if (state.activePreview === role)
                    state.activePreview = 'control';
                renderExtraControls();
                renderPreviewDeck();
                updateReadiness();
                saveDraft();
            });
            if (track.upload.path) {
                var status = document.getElementById('h3ct-' + role + '-status');
                status.textContent = mediaSummary(track.upload);
                document.getElementById('h3ct-' + role + '-drop').classList.add('ready');
            }
        });
        els.controlCount.textContent = String(1 + state.extraControls.length);
        els.addControl.disabled = state.extraControls.length >= 3;
        if (typeof lucide !== 'undefined')
            lucide.createIcons();
    }
    function updateContract() {
        var duration = Math.max(5, Math.min(15, Math.round(Number(els.duration.value || 5))));
        var resolution = h3Resolution(els.resolution.value);
        els.duration.value = String(duration);
        els.resolution.value = resolution.key;
        updateContractLabels();
    }
    function updateContractLabels() {
        var duration = Math.max(5, Math.min(15, Math.round(Number(els.duration.value || 5))));
        var output = duration * 24;
        var internal = output + (17 - ((output - 5) % 17)) % 17;
        els.outputFrames.textContent = output + ' frames';
        els.internalFrames.textContent = internal + ' frames';
        updateReadiness();
    }
    function showError(message) {
        els.error.textContent = message;
        els.error.classList.add('visible');
    }
    function clearError() {
        els.error.textContent = '';
        els.error.classList.remove('visible');
    }
    function formatClock(seconds) {
        var total = Math.max(0, Math.round(Number(seconds || 0)));
        var hours = Math.floor(total / 3600);
        var minutes = Math.floor((total % 3600) / 60);
        var secs = total % 60;
        return (hours ? String(hours).padStart(2, '0') + ':' : '') + String(minutes).padStart(2, '0') + ':' + String(secs).padStart(2, '0');
    }
    function updateElapsed() {
        var elapsed = state.startedAt ? (Date.now() - state.startedAt) / 1000 : 0;
        els.elapsed.textContent = 'Elapsed ' + formatClock(elapsed);
        if (state.sampleStep > 0 && state.sampleMax > state.sampleStep && state.samplingStartedAt) {
            var sampled = (Date.now() - state.samplingStartedAt) / 1000;
            var eta = sampled / state.sampleStep * (state.sampleMax - state.sampleStep);
            els.eta.textContent = 'ETA ' + formatClock(eta);
        }
        else {
            els.eta.textContent = 'ETA —';
        }
    }
    function startElapsedTimer() {
        if (state.elapsedTimer)
            clearInterval(state.elapsedTimer);
        updateElapsed();
        state.elapsedTimer = setInterval(updateElapsed, 1000);
    }
    function stopElapsedTimer() {
        if (state.elapsedTimer)
            clearInterval(state.elapsedTimer);
        state.elapsedTimer = null;
        updateElapsed();
    }
    // Map a runner phase onto one of the five visible stages. The progress
    // stream carries phases from the whole pipeline (conditioning, control
    // encode, denoise, decode, save), not only sampling; forcing 'sample' here
    // made the stage bar lie for the several minutes the control VAE encode
    // takes. Unknown phases keep the previous 'sample' behaviour.
    function stageForPhase(phase) {
        var p = String(phase || '').toLowerCase();
        if (/decode/.test(p)) return 'decode';
        if (/\bsave\b|mux/.test(p)) return 'save';
        if (/denois|sampl/.test(p)) return 'sample';
        if (/control/.test(p)) return 'control';
        if (/condition|modulation|prepare|token/.test(p)) return 'prepare';
        return 'sample';
    }

    function setPhase(phase) {
        state.activePhase = phase;
        var order = ['prepare', 'control', 'sample', 'decode', 'save'];
        var complete = phase === 'complete';
        var active = complete ? order.length : order.indexOf(phase);
        els.phaseTrack.querySelectorAll('[data-phase]').forEach(function (item) {
            var index = order.indexOf(item.dataset.phase);
            item.classList.toggle('active', !complete && index === active);
            item.classList.toggle('done', complete || (active >= 0 && index < active));
        });
    }
    function setStatus(label, percent, step, maxStep) {
        els.status.textContent = label;
        var value = Math.max(0, Math.min(100, Number(percent || 0)));
        els.progressValue.textContent = Math.round(value) + '%';
        els.progressBar.style.width = value + '%';
        if (maxStep > 0) {
            state.sampleStep = Number(step || 0);
            state.sampleMax = Number(maxStep || 0);
            els.stepLabel.textContent = 'Step ' + state.sampleStep + ' / ' + state.sampleMax;
        }
        else if (!state.running) {
            els.stepLabel.textContent = 'No sampling step';
        }
        updateElapsed();
    }
    function updateReadiness() {
        if (!els.generate)
            return;
        var supportedResolution = Object.prototype.hasOwnProperty.call(h3ResolutionPresets, String(els.resolution.value || ''));
        var enabledExtras = state.extraControls.filter(function (track) { return track.enabled !== false; });
        var checks = [
            { label: 'H3 base and official ControlNet', ok: !!els.model.value && !!els.controlnet.value },
            { label: 'Text, video, and audio assets', ok: !!state.clip && !!state.videoVae && !!state.audioVae },
            { label: 'Prompt entered', ok: !!els.prompt.value.trim() },
            { label: 'Control #1 uploaded', ok: !!state.uploads.control.path },
            { label: 'Extra controls uploaded', ok: enabledExtras.every(function (track) { return !!track.upload.path; }), hidden: !enabledExtras.length },
            { label: 'Supported H3 resolution', ok: supportedResolution },
            { label: 'Uploads complete', ok: state.uploadsInFlight === 0 }
        ];
        if (els.inpaintToggle.checked) {
            checks.push({ label: 'Source and mask uploaded', ok: !!state.uploads.source.path && !!state.uploads.mask.path });
        }
        // The per-item checklist is not rendered; these checks only gate the
        // Generate button and the live dot.
        var ready = !state.running && checks.every(function (check) { return check.hidden || check.ok; });
        els.generate.disabled = !ready;
        els.liveDot.className = 'h3ct-live-dot ' + (state.running ? 'running' : (ready ? 'ready' : 'empty'));
    }
    function setRunning(running) {
        state.running = running;
        els.generate.classList.toggle('running', running);
        els.generate.querySelector('span').textContent = running ? 'Queued / running' : 'Generate H3 CT';
        els.cancel.disabled = !(running && state.ownExecuting);
        if (!running)
            stopElapsedTimer();
        updateReadiness();
    }
    function isVideoSlot(slot) {
        var name = slot && slot.file ? slot.file.name : (slot && slot.path || '');
        var type = slot && slot.file ? slot.file.type : '';
        return type.indexOf('video/') === 0 || /\.(mp4|mov|mkv|webm|avi|m4v)$/i.test(name);
    }
    function mediaSummary(slot) {
        if (!slot)
            return 'No file';
        if (!slot.file)
            return slot.path ? slot.path + ' · restored upload' : 'No file';
        var meta = slot.meta || {};
        var pieces = [slot.file.name];
        if (meta.width && meta.height)
            pieces.push(meta.width + '×' + meta.height);
        if (meta.duration)
            pieces.push(meta.duration.toFixed(2) + 's');
        if (isVideoSlot(slot))
            pieces.push('aligned to 24 FPS');
        pieces.push(formatBytes(slot.file.size));
        return pieces.join(' · ');
    }
    function probeMedia(role, slot) {
        slot.meta = { name: slot.file.name, size: slot.file.size, width: 0, height: 0, duration: 0 };
        var done = function () {
            var status = document.getElementById('h3ct-' + role + '-status');
            if (status && slot.path)
                status.textContent = mediaSummary(slot);
            renderPreviewDeck();
        };
        if (isVideoSlot(slot)) {
            var video = document.createElement('video');
            video.preload = 'metadata';
            video.onloadedmetadata = function () {
                slot.meta.width = video.videoWidth;
                slot.meta.height = video.videoHeight;
                slot.meta.duration = Number.isFinite(video.duration) ? video.duration : 0;
                done();
            };
            video.onerror = done;
            video.src = slot.objectUrl;
        }
        else {
            var image = new Image();
            image.onload = function () {
                slot.meta.width = image.naturalWidth;
                slot.meta.height = image.naturalHeight;
                done();
            };
            image.onerror = done;
            image.src = slot.objectUrl;
        }
    }
    function previewEntries() {
        var entries = [];
        function add(key, label, slot, fit) {
            if (!slot || (!slot.objectUrl && !slot.url))
                return;
            entries.push({ key: key, label: label, url: slot.objectUrl || slot.url, slot: slot, fit: fit || 'contain', video: isVideoSlot(slot) });
        }
        add('control', 'Control #1', state.uploads.control, els.resize ? els.resize.value : 'crop');
        state.extraControls.forEach(function (track) {
            if (track.enabled !== false)
                add('control-' + track.id, 'Control #' + track.id, track.upload, track.resizeMode);
        });
        add('source', 'Source', state.uploads.source, els.resize ? els.resize.value : 'crop');
        add('mask', els.invertMask && els.invertMask.checked ? 'Mask · inverted' : 'Mask', state.uploads.mask, els.resize ? els.resize.value : 'crop');
        if (state.resultUrl)
            entries.push({ key: 'result', label: 'Result', url: state.resultUrl, slot: { meta: state.resultMeta || {}, path: state.resultFile && state.resultFile.filename || '' }, fit: 'contain', video: true });
        return entries;
    }
    function addPreviewPane(container, entry) {
        var pane = document.createElement('div');
        pane.className = 'h3ct-media-pane ' + (entry.key === 'mask' ? 'mask' : '');
        pane.innerHTML = '<span>' + escapeHtml(entry.label) + '</span>';
        var media = document.createElement(entry.video ? 'video' : 'img');
        media.src = entry.url;
        media.dataset.previewKey = entry.key;
        media.style.objectFit = entry.fit === 'crop' ? 'cover' : (entry.fit === 'stretch' ? 'fill' : 'contain');
        if (entry.video) {
            media.preload = 'metadata';
            media.playsInline = true;
            media.muted = entry.key !== 'result';
        }
        else {
            media.alt = entry.label + ' preview';
        }
        pane.appendChild(media);
        container.appendChild(pane);
    }
    function renderPreviewDeck() {
        if (!els.controlPreview)
            return;
        var entries = previewEntries();
        if (!entries.some(function (entry) { return entry.key === state.activePreview; })) {
            state.activePreview = entries.length ? entries[0].key : 'control';
        }
        els.previewTabs.innerHTML = entries.length ? entries.map(function (entry) {
            return '<button type="button" class="' + (entry.key === state.activePreview && state.compareMode === 'single' ? 'active' : '') + '" data-preview="' + entry.key + '">' + escapeHtml(entry.label) + '</button>';
        }).join('') : '<button class="active" data-preview="control">Control #1</button>';
        els.previewTabs.querySelectorAll('[data-preview]').forEach(function (button) {
            button.addEventListener('click', function () {
                state.activePreview = this.dataset.preview;
                state.compareMode = 'single';
                renderPreviewDeck();
            });
        });
        var control = entries.find(function (entry) { return entry.key === 'control'; });
        var result = entries.find(function (entry) { return entry.key === 'result'; });
        els.viewSplit.disabled = !(control && result);
        els.viewSingle.classList.toggle('active', state.compareMode === 'single');
        els.viewSplit.classList.toggle('active', state.compareMode === 'split');
        els.controlPreview.innerHTML = '';
        if (!entries.length) {
            els.controlPreview.innerHTML = '<div class="h3ct-preview-empty"><i data-lucide="scan-line"></i><strong>Drop a control video</strong><span>The exact uploaded media appears here.</span></div>';
            els.controlName.textContent = 'No control video';
            els.mediaMeta.textContent = 'Waiting for media metadata';
            wireTransport();
            if (typeof lucide !== 'undefined')
                lucide.createIcons();
            return;
        }
        if (state.compareMode === 'split' && control && result) {
            els.controlPreview.classList.add('split');
            addPreviewPane(els.controlPreview, control);
            addPreviewPane(els.controlPreview, result);
            els.controlName.textContent = 'Control #1 ↔ generated result';
            els.mediaMeta.textContent = 'Synchronized comparison · 24 FPS frame stepping';
        }
        else {
            els.controlPreview.classList.remove('split');
            var active = entries.find(function (entry) { return entry.key === state.activePreview; }) || entries[0];
            addPreviewPane(els.controlPreview, active);
            els.controlName.textContent = active.key === 'result' && state.resultFile
                ? state.resultFile.filename
                : (active.slot && active.slot.file ? active.slot.file.name : active.label);
            if (active.key === 'result') {
                var resultPieces = [state.resultFile && state.resultFile.filename || 'Generated result'];
                if (state.resultMeta && state.resultMeta.width && state.resultMeta.height)
                    resultPieces.push(state.resultMeta.width + '×' + state.resultMeta.height);
                if (state.resultMeta && state.resultMeta.duration)
                    resultPieces.push(Number(state.resultMeta.duration).toFixed(2) + 's');
                resultPieces.push('24 FPS source');
                els.mediaMeta.textContent = resultPieces.join(' · ');
            }
            else {
                els.mediaMeta.textContent = active.slot && active.slot.file ? mediaSummary(active.slot) : active.label;
            }
        }
        wireTransport();
        if (typeof lucide !== 'undefined')
            lucide.createIcons();
    }
    function wireTransport() {
        var videos = Array.from(els.controlPreview.querySelectorAll('video'));
        var master = videos.find(function (video) { return video.dataset.previewKey === 'result'; }) || videos[0];
        [els.frameBack, els.deckPlay, els.frameForward, els.scrubber].forEach(function (control) { control.disabled = !master; });
        if (!master) {
            els.timecode.textContent = '00:00 / 00:00';
            els.scrubber.value = '0';
            return;
        }
        function duration() {
            var values = videos.map(function (video) { return Number.isFinite(video.duration) ? video.duration : 0; }).filter(Boolean);
            return values.length ? Math.min.apply(null, values) : 0;
        }
        function updateTime() {
            var end = duration();
            els.scrubber.value = end ? String(Math.round(master.currentTime / end * 1000)) : '0';
            els.timecode.textContent = formatClock(master.currentTime) + ' / ' + formatClock(end);
            videos.forEach(function (video) {
                if (video !== master && Math.abs(video.currentTime - master.currentTime) > 0.08)
                    video.currentTime = Math.min(master.currentTime, video.duration || master.currentTime);
            });
        }
        master.ontimeupdate = updateTime;
        master.onloadedmetadata = updateTime;
        master.onended = function () { els.deckPlay.innerHTML = '<i data-lucide="play"></i>'; if (typeof lucide !== 'undefined')
            lucide.createIcons(); };
        els.scrubber.oninput = function () {
            var target = duration() * Number(this.value) / 1000;
            videos.forEach(function (video) { video.currentTime = Math.min(target, video.duration || target); });
            updateTime();
        };
        els.deckPlay.onclick = function () {
            if (master.paused) {
                videos.forEach(function (video) { video.play().catch(function () { }); });
                this.innerHTML = '<i data-lucide="pause"></i>';
            }
            else {
                videos.forEach(function (video) { video.pause(); });
                this.innerHTML = '<i data-lucide="play"></i>';
            }
            if (typeof lucide !== 'undefined')
                lucide.createIcons();
        };
        function step(delta) {
            videos.forEach(function (video) { video.pause(); video.currentTime = Math.max(0, Math.min(video.duration || Infinity, video.currentTime + delta / 24)); });
            els.deckPlay.innerHTML = '<i data-lucide="play"></i>';
            updateTime();
            if (typeof lucide !== 'undefined')
                lucide.createIcons();
        }
        els.frameBack.onclick = function () { step(-1); };
        els.frameForward.onclick = function () { step(1); };
    }
    function uploadFile(role, file) {
        if (!file)
            return;
        clearError();
        var slot = slotForRole(role);
        if (!slot)
            return;
        if (slot.objectUrl)
            URL.revokeObjectURL(slot.objectUrl);
        slot.objectUrl = URL.createObjectURL(file);
        slot.file = file;
        slot.path = '';
        slot.url = '';
        slot.meta = null;
        state.activePreview = role;
        probeMedia(role, slot);
        renderPreviewDeck();
        var status = document.getElementById('h3ct-' + role + '-status');
        status.textContent = 'Uploading ' + file.name + '…';
        var drop = document.getElementById('h3ct-' + role + '-drop');
        drop.classList.add('uploading');
        state.uploadsInFlight += 1;
        updateReadiness();
        SerenityH3API.uploadMediaDetails(file).then(function (data) {
            slot.path = data.path;
            slot.url = data.url;
            status.textContent = mediaSummary(slot);
            drop.classList.add('ready');
            saveDraft();
        }).catch(function (error) {
            slot.path = '';
            status.textContent = 'Upload failed';
            showError('Could not upload ' + file.name + ': ' + error.message);
        }).finally(function () {
            state.uploadsInFlight = Math.max(0, state.uploadsInFlight - 1);
            drop.classList.remove('uploading');
            renderPreviewDeck();
            updateReadiness();
        });
    }
    function bindUpload(role) {
        var input = document.getElementById('h3ct-' + role + '-input');
        var drop = document.getElementById('h3ct-' + role + '-drop');
        if (!input || !drop || drop.dataset.bound === 'true')
            return;
        drop.dataset.bound = 'true';
        drop.tabIndex = 0;
        drop.setAttribute('role', 'button');
        drop.setAttribute('aria-label', 'Choose or drop media');
        var button = drop.querySelector('button');
        button.addEventListener('click', function (event) {
            event.stopPropagation();
            input.click();
        });
        drop.addEventListener('click', function () { input.click(); });
        drop.addEventListener('keydown', function (event) {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                input.click();
            }
        });
        input.addEventListener('change', function () {
            uploadFile(role, input.files && input.files[0]);
        });
        ['dragenter', 'dragover'].forEach(function (type) {
            drop.addEventListener(type, function (event) {
                event.preventDefault();
                drop.classList.add('dragging');
            });
        });
        ['dragleave', 'drop'].forEach(function (type) {
            drop.addEventListener(type, function (event) {
                event.preventDefault();
                drop.classList.remove('dragging');
            });
        });
        drop.addEventListener('drop', function (event) {
            uploadFile(role, event.dataTransfer && event.dataTransfer.files[0]);
        });
    }
    function controlDraft(track) {
        return {
            id: track.id,
            controlMedia: track.upload.path,
            preprocessor: track.preprocessor,
            resizeMode: track.resizeMode,
            cannyLow: track.cannyLow,
            cannyHigh: track.cannyHigh,
            strength: track.strength,
            startPercent: track.startPercent,
            endPercent: track.endPercent,
            enabled: track.enabled !== false
        };
    }
    function saveDraft() {
        if (!els.prompt)
            return;
        var resolution = h3Resolution(els.resolution.value);
        try {
            localStorage.setItem('sf-h3ct-draft', JSON.stringify({
                version: 1,
                prompt: els.prompt.value,
                preprocessor: els.preprocessor.value,
                resizeMode: els.resize.value,
                cannyLow: Number(els.cannyLow.value),
                cannyHigh: Number(els.cannyHigh.value),
                strength: Number(els.strength.value),
                startPercent: Number(els.start.value),
                endPercent: Number(els.end.value),
                inpaint: els.inpaintToggle.checked,
                invertMask: els.invertMask.checked,
                resolution: resolution.key,
                width: resolution.width,
                height: resolution.height,
                durationSeconds: Math.round(Number(els.duration.value || 5)),
                steps: Math.round(Number(els.steps.value || 40)),
                seed: Number(els.seed.value),
                outputFormat: els.format.value,
                window: Math.round(Number(els.window.value || 2)),
                prefetch: Math.round(Number(els.prefetch.value || 1)),
                videoShift: Number(els.videoShift.value),
                audioShift: Number(els.audioShift.value),
                filenamePrefix: els.filename.value,
                controlMedia: state.uploads.control.path,
                sourceMedia: state.uploads.source.path,
                maskMedia: state.uploads.mask.path,
                loras: state.loras.map(function (lora) {
                    return { name: lora.name, strength: Number(lora.strength), enabled: lora.enabled !== false };
                }),
                controls: state.extraControls.map(controlDraft)
            }));
        }
        catch (_error) { }
        if (!state.syncingFromWorkflow)
            syncWorkflowFromControls();
    }
    function setDraftValue(element, value) {
        if (element && value != null)
            element.value = String(value);
    }
    function restoreDraft() {
        var draft;
        try {
            draft = JSON.parse(localStorage.getItem('sf-h3ct-draft') || 'null');
        }
        catch (_error) {
            return;
        }
        if (!draft || draft.version !== 1)
            return;
        setDraftValue(els.prompt, draft.prompt);
        setDraftValue(els.preprocessor, draft.preprocessor);
        setDraftValue(els.resize, draft.resizeMode);
        setDraftValue(els.cannyLow, draft.cannyLow);
        setDraftValue(els.cannyHigh, draft.cannyHigh);
        setDraftValue(els.strength, draft.strength);
        setDraftValue(els.start, draft.startPercent);
        setDraftValue(els.end, draft.endPercent);
        setDraftValue(els.resolution, h3Resolution(draft.resolution || String(draft.width) + 'x' + String(draft.height)).key);
        setDraftValue(els.duration, draft.durationSeconds);
        setDraftValue(els.steps, draft.steps);
        setDraftValue(els.seed, draft.seed);
        setDraftValue(els.format, draft.outputFormat);
        setDraftValue(els.window, draft.window);
        setDraftValue(els.prefetch, draft.prefetch);
        setDraftValue(els.videoShift, draft.videoShift);
        setDraftValue(els.audioShift, draft.audioShift);
        setDraftValue(els.filename, draft.filenamePrefix);
        els.inpaintToggle.checked = draft.inpaint === true;
        els.invertMask.checked = draft.invertMask === true;
        els.inpaintFields.classList.toggle('h3ct-hidden', !els.inpaintToggle.checked);
        els.cannyControls.classList.toggle('h3ct-hidden', els.preprocessor.value !== 'canny');
        if (Array.isArray(draft.loras)) {
            state.loras = draft.loras.filter(function (lora) { return lora && lora.name; }).map(function (lora) {
                return { name: String(lora.name), strength: Number(lora.strength), enabled: lora.enabled !== false };
            });
        }
        if (Array.isArray(draft.controls)) {
            state.extraControls = draft.controls.slice(0, 3).map(function (saved, index) {
                var track = makeControlTrack(Number(saved.id) || index + 2);
                track.upload.path = String(saved.controlMedia || '');
                track.upload.url = inputUrl(track.upload.path);
                ['preprocessor', 'resizeMode'].forEach(function (field) {
                    if (saved[field] != null)
                        track[field] = String(saved[field]);
                });
                ['cannyLow', 'cannyHigh', 'strength', 'startPercent', 'endPercent'].forEach(function (field) {
                    if (saved[field] != null && Number.isFinite(Number(saved[field])))
                        track[field] = Number(saved[field]);
                });
                track.enabled = saved.enabled !== false;
                return track;
            });
            state.nextControlId = state.extraControls.reduce(function (maximum, track) {
                return Math.max(maximum, track.id + 1);
            }, 2);
        }
        restoreUploadSlot('control', state.uploads.control, draft.controlMedia || '');
        restoreUploadSlot('source', state.uploads.source, draft.sourceMedia || '');
        restoreUploadSlot('mask', state.uploads.mask, draft.maskMedia || '');
        renderLoras();
        renderExtraControls();
    }
    function copyText(value) {
        if (navigator.clipboard && navigator.clipboard.writeText) {
            return navigator.clipboard.writeText(value);
        }
        var field = document.createElement('textarea');
        field.value = value;
        field.style.position = 'fixed';
        field.style.opacity = '0';
        document.body.appendChild(field);
        field.select();
        document.execCommand('copy');
        field.remove();
        return Promise.resolve();
    }
    function bindEvents() {
        bindUpload('control');
        bindUpload('source');
        bindUpload('mask');
        els.model.addEventListener('change', function () {
            updateSelectTitle(els.model);
            localStorage.setItem('sf-h3ct-model', els.model.value);
            updateReadiness();
            saveDraft();
        });
        els.controlnet.addEventListener('change', function () {
            updateSelectTitle(els.controlnet);
            localStorage.setItem('sf-h3ct-controlnet', els.controlnet.value);
            updateReadiness();
            saveDraft();
        });
        els.preprocessor.addEventListener('change', function () {
            var canny = els.preprocessor.value === 'canny';
            els.cannyControls.classList.toggle('h3ct-hidden', !canny);
            els.preprocessNote.textContent = canny
                ? 'OpenCV Canny runs on every resampled 24 FPS frame.'
                : 'Prepared accepts Canny, Depth, HED, MLSD, or Pose maps. Unavailable raw preprocessors are not faked.';
            saveDraft();
        });
        els.inpaintToggle.addEventListener('change', function () {
            els.inpaintFields.classList.toggle('h3ct-hidden', !els.inpaintToggle.checked);
            updateReadiness();
            renderPreviewDeck();
            saveDraft();
        });
        [els.resolution, els.duration].forEach(function (input) {
            input.addEventListener('input', updateContractLabels);
            input.addEventListener('change', function () { updateContract(); saveDraft(); });
        });
        els.randomSeed.addEventListener('click', function () {
            els.seed.value = String(Math.floor(Math.random() * 2147483647));
            saveDraft();
        });
        els.addControl.addEventListener('click', function () {
            if (state.extraControls.length >= 3)
                return;
            state.extraControls.push(makeControlTrack(state.nextControlId++));
            renderExtraControls();
            renderPreviewDeck();
            updateReadiness();
            saveDraft();
        });
        els.loraPicker.addEventListener('change', function () {
            var name = this.value;
            if (name && !state.loras.some(function (lora) { return lora.name === name; })) {
                state.loras.push({ name: name, strength: 1, enabled: true });
                renderLoras();
                saveDraft();
            }
            this.value = '';
        });
        [els.resize, els.cannyLow, els.cannyHigh, els.strength, els.start, els.end,
            els.invertMask, els.prompt, els.steps, els.seed, els.format, els.window,
            els.prefetch, els.videoShift, els.audioShift, els.filename].forEach(function (input) {
            input.addEventListener('change', function () {
                renderPreviewDeck();
                updateReadiness();
                saveDraft();
            });
        });
        [els.cannyLow, els.cannyHigh, els.strength, els.start, els.end,
            els.prompt, els.duration, els.steps, els.seed,
            els.window, els.prefetch, els.videoShift, els.audioShift, els.filename
        ].forEach(function (input) {
            input.addEventListener('input', syncWorkflowFromControls);
        });
        els.prompt.addEventListener('input', updateReadiness);
        [
            { element: els.clip, key: 'sf-h3ct-clip', stateKey: 'clip' },
            { element: els.videoVae, key: 'sf-h3ct-video-vae', stateKey: 'videoVae' },
            { element: els.audioVae, key: 'sf-h3ct-audio-vae', stateKey: 'audioVae' }
        ].forEach(function (binding) {
            binding.element.addEventListener('change', function () {
                updateSelectTitle(this);
                state[binding.stateKey] = this.value;
                localStorage.setItem(binding.key, this.value);
                updateReadiness();
                saveDraft();
            });
        });
        els.viewSingle.addEventListener('click', function () {
            state.compareMode = 'single';
            renderPreviewDeck();
        });
        els.viewSplit.addEventListener('click', function () {
            if (els.viewSplit.disabled)
                return;
            state.compareMode = 'split';
            renderPreviewDeck();
        });
        els.openWorkflow.addEventListener('click', function () {
            clearError();
            var workflow;
            try {
                workflow = H3ControlWorkflow.build(buildParams());
            }
            catch (error) {
                showError(error.message);
                return;
            }
            if (typeof mirrorPromptToWorkflow === 'function') {
                mirrorPromptToWorkflow(workflow, {
                    name: 'MiniMax H3 ControlNet Union',
                    description: 'Live settings from the H3 CT screen'
                });
            }
            if (typeof switchTab === 'function')
                switchTab('workflows');
        });
        els.copyWorkflow.addEventListener('click', function () {
            clearError();
            try {
                var workflow = H3ControlWorkflow.build(buildParams());
                copyText(JSON.stringify(workflow, null, 2)).then(function () {
                    els.copyWorkflow.textContent = 'Copied API JSON';
                    setTimeout(function () { els.copyWorkflow.textContent = 'Copy API JSON'; }, 1500);
                }).catch(function (error) { showError('Copy failed: ' + error.message); });
            }
            catch (error) {
                showError(error.message);
            }
        });
        els.download.addEventListener('click', downloadResult);
        els.reuse.addEventListener('click', reuseResultSettings);
        els.timeline.addEventListener('click', sendResultToTimeline);
        els.refreshRecent.addEventListener('click', loadRecent);
        els.generate.addEventListener('click', generate);
        els.cancel.addEventListener('click', cancel);
    }
    function buildParams(resolveRandomSeed) {
        var resolution = h3Resolution(els.resolution.value);
        var seed = Number(els.seed.value);
        if (resolveRandomSeed && seed === -1) {
            seed = Math.floor(Math.random() * 4294967296);
            els.seed.value = String(seed);
        }
        state.clip = els.clip.value;
        state.videoVae = els.videoVae.value;
        state.audioVae = els.audioVae.value;
        var controls = [{
                enabled: true,
                controlMedia: state.uploads.control.path,
                preprocessor: els.preprocessor.value,
                resizeMode: els.resize.value,
                cannyLow: Number(els.cannyLow.value),
                cannyHigh: Number(els.cannyHigh.value),
                strength: Number(els.strength.value),
                startPercent: Number(els.start.value),
                endPercent: Number(els.end.value)
            }].concat(state.extraControls.map(function (track) {
            return {
                enabled: track.enabled !== false,
                controlMedia: track.upload.path,
                preprocessor: track.preprocessor,
                resizeMode: track.resizeMode,
                cannyLow: Number(track.cannyLow),
                cannyHigh: Number(track.cannyHigh),
                strength: Number(track.strength),
                startPercent: Number(track.startPercent),
                endPercent: Number(track.endPercent)
            };
        }));
        return {
            model: els.model.value,
            controlNet: els.controlnet.value,
            clipName: state.clip,
            videoVae: state.videoVae,
            audioVae: state.audioVae,
            controlMedia: state.uploads.control.path,
            controls: controls,
            loras: [],
            sourceMedia: els.inpaintToggle.checked ? state.uploads.source.path : '',
            maskMedia: els.inpaintToggle.checked ? state.uploads.mask.path : '',
            invertMask: els.invertMask.checked,
            preprocessor: els.preprocessor.value,
            resizeMode: els.resize.value,
            cannyLow: Number(els.cannyLow.value),
            cannyHigh: Number(els.cannyHigh.value),
            strength: Number(els.strength.value),
            startPercent: Number(els.start.value),
            endPercent: Number(els.end.value),
            prompt: els.prompt.value,
            width: resolution.width,
            height: resolution.height,
            durationSeconds: Number(els.duration.value),
            steps: Number(els.steps.value),
            seed: seed,
            outputFormat: els.format.value,
            window: Number(els.window.value),
            prefetch: Number(els.prefetch.value),
            videoShift: Number(els.videoShift.value),
            audioShift: Number(els.audioShift.value),
            filenamePrefix: els.filename.value
        };
    }
    function syncWorkflowFromControls() {
        if (!initialized || state.syncingFromWorkflow ||
            typeof H3ControlWorkflow === 'undefined' ||
            typeof mirrorPromptToWorkflow !== 'function')
            return false;
        try {
            var params = buildParams();
            var workflow = H3ControlWorkflow.build(params);
            return mirrorPromptToWorkflow(workflow, {
                name: 'MiniMax H3 ControlNet Union',
                description: 'Live settings from the H3 CT screen'
            });
        }
        catch (_error) {
            return false;
        }
    }
    function selectContains(select, value) {
        return !!select && Array.from(select.options).some(function (option) {
            return option.value === value;
        });
    }
    function applyWorkflowParams(params) {
        if (!initialized)
            init();
        if (!state.modelsLoaded || els.model.disabled || els.controlnet.disabled ||
            !els.model.options.length || !els.controlnet.options.length) {
            state.pendingWorkflowParams = params;
            return true;
        }
        var missing = [
            [els.model, params.model, 'base checkpoint'],
            [els.controlnet, params.controlNet, 'ControlNet checkpoint'],
            [els.clip, params.clipName, 'text encoder'],
            [els.videoVae, params.videoVae, 'video VAE'],
            [els.audioVae, params.audioVae, 'audio VAE']
        ].filter(function (entry) {
            return entry[1] && !selectContains(entry[0], entry[1]);
        });
        if (missing.length) {
            showError('Workflow ' + missing[0][2] + ' is not installed: ' + missing[0][1]);
            return false;
        }
        state.syncingFromWorkflow = true;
        try {
            applyParams(params);
        }
        finally {
            state.syncingFromWorkflow = false;
        }
        return true;
    }
    function syncFromWorkflow(prompt) {
        var params;
        try {
            params = H3ControlWorkflow.parse(prompt);
        }
        catch (error) {
            if (initialized)
                showError(error.message);
            return false;
        }
        if (!params)
            return false;
        return applyWorkflowParams(params);
    }
    function generate() {
        if (!initialized || state.running || els.generate.disabled)
            return;
        clearError();
        var params;
        var workflow;
        try {
            params = buildParams(true);
            workflow = H3ControlWorkflow.build(params);
        }
        catch (error) {
            showError(error.message);
            return;
        }
        var submissionToken = SerenityH3API.createPromptId();
        state.promptId = 'submitting:' + submissionToken;
        state.ownExecuting = false;
        state.lastParams = JSON.parse(JSON.stringify(params));
        state.queueNumber = 0;
        state.startedAt = Date.now();
        state.samplingStartedAt = 0;
        state.sampleStep = 0;
        state.sampleMax = 0;
        setPhase('prepare');
        setRunning(true);
        startElapsedTimer();
        setStatus('Submitting native workflow', 0);
        saveDraft();
        if (typeof mirrorPromptToWorkflow === 'function') {
            mirrorPromptToWorkflow(workflow, {
                name: 'MiniMax H3 ControlNet Union',
                description: 'Running H3 CT workflow'
            });
        }
        SerenityH3API.postPrompt(workflow, {
            prompt: params.prompt,
            model: params.model,
            width: params.width,
            height: params.height,
            seed: params.seed,
            steps: params.steps,
            guidance: 1.0,
            scheduler: 'normal',
            arch: 'minimax_h3',
            batchLabel: 'H3 CT'
        }).then(function (data) {
            if (state.promptId !== 'submitting:' + submissionToken)
                return;
            var nativePromptId = String(data.prompt_id || data.video_id || '');
            if (!nativePromptId)
                throw new Error('native H3 server did not return a job id');
            state.promptId = nativePromptId;
            state.queueNumber = Number(data.number || 0);
            setStatus(state.queueNumber ? 'Queued #' + state.queueNumber + ' · H3 CT' : 'Queued · H3 CT', 0);
        }).catch(function (error) {
            if (state.promptId !== 'submitting:' + submissionToken)
                return;
            state.promptId = '';
            setRunning(false);
            setPhase('ready');
            setStatus('Submission failed', 0);
            showError(error.message);
        });
    }
    function cancel() {
        if (!state.running || !state.ownExecuting)
            return;
        els.cancel.disabled = true;
        setStatus('Cancel requested', Number(els.progressValue.textContent.replace('%', '')) || 0);
        SerenityH3API.interrupt().catch(function (error) {
            showError('Cancel failed: ' + error.message);
        });
    }
    function isOwn(data) {
        return !!state.promptId && String(data && data.prompt_id || '') === state.promptId;
    }
    function handleOwnExecutionError(data) {
        if (!isOwn(data))
            return;
        state.ownExecuting = false;
        setRunning(false);
        setPhase('ready');
        setStatus('Generation failed', 0);
        showError(data.exception_message || 'H3 CT generation failed');
        if (typeof showExecutionIssue === 'function')
            showExecutionIssue(data);
        state.promptId = '';
    }
    function normalizeOutputFile(file) {
        return {
            filename: String(file && (file.filename || file.name) || ''),
            subfolder: String(file && file.subfolder || ''),
            type: String(file && file.type || 'output') === 'video' ? 'output' : String(file && file.type || 'output')
        };
    }
    function selectResultFile(file, params) {
        var normalized = normalizeOutputFile(file);
        if (!normalized.filename)
            return;
        state.resultFile = normalized;
        state.resultUrl = SerenityH3API.viewUrl(normalized.filename, normalized.subfolder, normalized.type);
        state.resultParams = params ? JSON.parse(JSON.stringify(params)) : null;
        state.resultMeta = {
            width: state.resultParams && state.resultParams.width || 0,
            height: state.resultParams && state.resultParams.height || 0,
            duration: state.resultParams && state.resultParams.durationSeconds || 0
        };
        els.result.src = state.resultUrl;
        els.result.onloadedmetadata = function () {
            if (!state.resultMeta.width)
                state.resultMeta.width = els.result.videoWidth || 0;
            if (!state.resultMeta.height)
                state.resultMeta.height = els.result.videoHeight || 0;
            if (!state.resultMeta.duration && Number.isFinite(els.result.duration))
                state.resultMeta.duration = els.result.duration;
            renderPreviewDeck();
        };
        els.resultCard.classList.remove('h3ct-hidden');
        state.activePreview = 'result';
        renderPreviewDeck();
    }
    function displayResult(output) {
        var out = output && (output.ui || output);
        var items = out && (out.videos || out.images);
        if (!items || !items.length)
            return;
        selectResultFile(items[0], state.lastParams);
        loadRecent();
    }
    function downloadResult() {
        if (!state.resultUrl)
            return;
        var link = document.createElement('a');
        link.href = state.resultUrl;
        link.download = state.resultFile && state.resultFile.filename || ('minimax_h3_controlnet_union_' + Date.now() + '.mp4');
        link.click();
    }
    function inputUrl(path) {
        var normalized = String(path || '').replace(/^\/+/, '');
        var split = normalized.lastIndexOf('/');
        var subfolder = split >= 0 ? normalized.slice(0, split) : '';
        var filename = split >= 0 ? normalized.slice(split + 1) : normalized;
        return filename ? SerenityH3API.viewUrl(filename, subfolder, 'input') : '';
    }
    function restoreUploadSlot(role, slot, path) {
        path = String(path || '');
        if (slot.path !== path) {
            if (slot.objectUrl)
                URL.revokeObjectURL(slot.objectUrl);
            slot.file = null;
            slot.objectUrl = '';
            slot.meta = null;
        }
        slot.path = path;
        slot.url = inputUrl(path);
        var status = document.getElementById('h3ct-' + role + '-status');
        var drop = document.getElementById('h3ct-' + role + '-drop');
        if (status)
            status.textContent = mediaSummary(slot);
        if (drop)
            drop.classList.toggle('ready', !!path);
    }
    function applyParams(params) {
        if (!params)
            return;
        if (params.model && Array.from(els.model.options).some(function (option) { return option.value === params.model; }))
            els.model.value = params.model;
        if (params.controlNet && Array.from(els.controlnet.options).some(function (option) { return option.value === params.controlNet; }))
            els.controlnet.value = params.controlNet;
        updateSelectTitle(els.model);
        updateSelectTitle(els.controlnet);
        localStorage.setItem('sf-h3ct-model', els.model.value);
        localStorage.setItem('sf-h3ct-controlnet', els.controlnet.value);
        setDraftValue(els.prompt, params.prompt);
        setDraftValue(els.resolution, h3Resolution(String(params.width) + 'x' + String(params.height)).key);
        setDraftValue(els.duration, params.durationSeconds);
        setDraftValue(els.steps, params.steps);
        setDraftValue(els.seed, params.seed);
        setDraftValue(els.format, params.outputFormat);
        setDraftValue(els.window, params.window);
        setDraftValue(els.prefetch, params.prefetch);
        setDraftValue(els.videoShift, params.videoShift);
        setDraftValue(els.audioShift, params.audioShift);
        setDraftValue(els.filename, params.filenamePrefix);
        setDraftValue(els.preprocessor, params.preprocessor);
        setDraftValue(els.resize, params.resizeMode);
        setDraftValue(els.cannyLow, params.cannyLow);
        setDraftValue(els.cannyHigh, params.cannyHigh);
        setDraftValue(els.strength, params.strength);
        setDraftValue(els.start, params.startPercent);
        setDraftValue(els.end, params.endPercent);
        els.inpaintToggle.checked = !!(params.sourceMedia && params.maskMedia);
        els.invertMask.checked = params.invertMask === true;
        els.inpaintFields.classList.toggle('h3ct-hidden', !els.inpaintToggle.checked);
        els.cannyControls.classList.toggle('h3ct-hidden', els.preprocessor.value !== 'canny');
        if (params.clipName && Array.from(els.clip.options).some(function (option) { return option.value === params.clipName; }))
            els.clip.value = params.clipName;
        if (params.videoVae && Array.from(els.videoVae.options).some(function (option) { return option.value === params.videoVae; }))
            els.videoVae.value = params.videoVae;
        if (params.audioVae && Array.from(els.audioVae.options).some(function (option) { return option.value === params.audioVae; }))
            els.audioVae.value = params.audioVae;
        updateSelectTitle(els.clip);
        updateSelectTitle(els.videoVae);
        updateSelectTitle(els.audioVae);
        state.clip = els.clip.value;
        state.videoVae = els.videoVae.value;
        state.audioVae = els.audioVae.value;
        localStorage.setItem('sf-h3ct-clip', state.clip);
        localStorage.setItem('sf-h3ct-video-vae', state.videoVae);
        localStorage.setItem('sf-h3ct-audio-vae', state.audioVae);
        var savedControls = params.controls || [];
        restoreUploadSlot('control', state.uploads.control, params.controlMedia || savedControls[0] && savedControls[0].controlMedia || '');
        restoreUploadSlot('source', state.uploads.source, params.sourceMedia || '');
        restoreUploadSlot('mask', state.uploads.mask, params.maskMedia || '');
        state.loras = (params.loras || []).map(function (lora) {
            return { name: lora.name, strength: Number(lora.strength), enabled: lora.enabled !== false };
        });
        var oldTracks = state.extraControls.slice();
        state.extraControls = savedControls.slice(1, 4).map(function (control, index) {
            var track = oldTracks[index] || makeControlTrack(index + 2);
            track.preprocessor = control.preprocessor || 'prepared';
            track.resizeMode = control.resizeMode || 'crop';
            track.cannyLow = Number(control.cannyLow == null ? 100 : control.cannyLow);
            track.cannyHigh = Number(control.cannyHigh == null ? 200 : control.cannyHigh);
            track.strength = Number(control.strength == null ? 1 : control.strength);
            track.startPercent = Number(control.startPercent == null ? 0 : control.startPercent);
            track.endPercent = Number(control.endPercent == null ? 1 : control.endPercent);
            track.enabled = control.enabled !== false;
            restoreUploadSlot('control-' + track.id, track.upload, control.controlMedia || '');
            return track;
        });
        state.nextControlId = state.extraControls.reduce(function (maximum, track) { return Math.max(maximum, track.id + 1); }, 2);
        renderLoras();
        renderExtraControls();
        updateContract();
        renderPreviewDeck();
        updateReadiness();
        saveDraft();
    }
    function reuseResultSettings() {
        if (!state.resultParams) {
            showError('This result has no settings stored in the current session');
            return;
        }
        applyParams(state.resultParams);
        clearError();
    }
    function waitForVideoProject(timeoutMs) {
        var deadline = Date.now() + timeoutMs;
        return new Promise(function (resolve, reject) {
            function check() {
                var projectId = VideoEditTab.getActiveProjectId && VideoEditTab.getActiveProjectId();
                if (projectId) {
                    resolve(projectId);
                    return;
                }
                if (Date.now() >= deadline) {
                    reject(new Error('Video Edit project did not become ready'));
                    return;
                }
                setTimeout(check, 100);
            }
            check();
        });
    }
    function sendResultToTimeline() {
        if (!state.resultUrl || els.timeline.disabled)
            return;
        if (typeof VideoEditTab === 'undefined' || !VideoEditTab.init || !VideoEditTab.addClipFromExternal) {
            showError('Video Edit is unavailable');
            return;
        }
        clearError();
        var original = els.timeline.innerHTML;
        els.timeline.disabled = true;
        els.timeline.textContent = 'Importing…';
        var label = String(state.resultParams && state.resultParams.prompt || 'H3 Control result').trim().slice(0, 40) || 'H3 Control result';
        var authoredSeconds = Number(state.resultParams && state.resultParams.durationSeconds || state.resultMeta && state.resultMeta.duration || 0);
        var filename = state.resultFile && state.resultFile.filename || 'h3-control-result.mp4';
        if (typeof switchTab === 'function')
            switchTab('video-edit');
        if (!VideoEditTab._initialized)
            VideoEditTab.init();
        requestAnimationFrame(function () { if (VideoEditTab.resize)
            VideoEditTab.resize(); });
        waitForVideoProject(5000).then(function (projectId) {
            return fetch(state.resultUrl, { cache: 'no-store' }).then(function (response) {
                if (!response.ok)
                    throw new Error('Could not read generated video: HTTP ' + response.status);
                return response.blob();
            }).then(function (blob) {
                var form = new FormData();
                form.append('file', blob, filename);
                return fetch('/video_edit/projects/' + encodeURIComponent(projectId) + '/import_clip', {
                    method: 'POST', body: form
                });
            }).then(function (response) {
                return response.json().then(function (data) {
                    if (!response.ok || data.error)
                        throw new Error(data.error || ('HTTP ' + response.status));
                    return data;
                });
            });
        }).then(function (data) {
            var importedSeconds = Number(data.duration_frames || 0) / Math.max(1, Number(data.fps || 30));
            var durationFrames = Math.max(1, Math.round((authoredSeconds || importedSeconds || 5) * 30));
            VideoEditTab.addClipFromExternal(data.source_path, label, durationFrames);
            if (VideoEditTab.resize)
                VideoEditTab.resize();
            if (typeof sfToolbar !== 'undefined' && sfToolbar._toast)
                sfToolbar._toast('H3 result added to Video Edit', 'success');
        }).catch(function (error) {
            showError('Video Edit import failed: ' + error.message);
            if (typeof sfToolbar !== 'undefined' && sfToolbar._toast)
                sfToolbar._toast('Video Edit import failed: ' + error.message, 'error');
        }).finally(function () {
            els.timeline.disabled = false;
            els.timeline.innerHTML = original;
            if (typeof lucide !== 'undefined')
                lucide.createIcons();
        });
    }
    function loadRecent() {
        if (!els.recentList)
            return;
        els.refreshRecent.disabled = true;
        fetch('/output_files', { cache: 'no-store' }).then(function (response) {
            if (!response.ok)
                throw new Error('HTTP ' + response.status);
            return response.json();
        }).then(function (files) {
            state.recent = (Array.isArray(files) ? files : []).filter(function (file) {
                return file.type === 'video';
            }).slice(0, 8);
            els.recentList.innerHTML = '';
            if (!state.recent.length) {
                els.recentList.innerHTML = '<span>No Mojo video outputs found yet.</span>';
                return;
            }
            state.recent.forEach(function (file) {
                var button = document.createElement('button');
                button.type = 'button';
                button.className = 'h3ct-recent-item';
                var subfolder = String(file.subfolder || '');
                var url = SerenityH3API.viewUrl(file.name, subfolder, 'output');
                button.innerHTML = '<video src="' + escapeHtml(url) + '#t=0.1" muted playsinline preload="metadata"></video>' +
                    '<span><strong>' + escapeHtml(file.name) + '</strong><small>' + formatBytes(file.size_bytes) + '</small></span>';
                button.addEventListener('click', function () {
                    selectResultFile({ filename: file.name, subfolder: subfolder, type: 'output' }, null);
                });
                els.recentList.appendChild(button);
            });
        }).catch(function (error) {
            els.recentList.innerHTML = '<span>Could not load output history: ' + escapeHtml(error.message) + '</span>';
        }).finally(function () {
            els.refreshRecent.disabled = false;
        });
    }
    function bindWS() {
        if (wsBound)
            return;
        wsBound = true;
        SerenityH3WS.on('disconnected', function () {
            if (!state.running || !state.promptId)
                return;
            state.connectionInterrupted = true;
            state.ownExecuting = false;
            setRunning(false);
            setPhase('ready');
            setStatus('Connection lost', 0);
        });
        SerenityH3WS.on('execution_start', function (data) {
            if (!isOwn(data))
                return;
            state.ownExecuting = true;
            if (!state.startedAt)
                state.startedAt = Date.now();
            els.cancel.disabled = false;
            setPhase('prepare');
            setStatus('Starting H3 CT workflow', 0);
        });
        SerenityH3WS.on('executing', function (data) {
            if (!isOwn(data) || data.node == null)
                return;
            state.ownExecuting = true;
            els.cancel.disabled = false;
            var nodeType = String(data.node_type || data.node || '');
            var phaseKey = 'prepare';
            var label = nodeType;
            if (/textencode/i.test(nodeType))
                label = 'Encoding prompt';
            else if (/controlnetloader/i.test(nodeType)) {
                phaseKey = 'control';
                label = 'Loading H3 ControlNet';
            }
            else if (/applymedia/i.test(nodeType)) {
                phaseKey = 'control';
                label = 'Aligning and encoding control media';
            }
            else if (/sampler/i.test(nodeType)) {
                phaseKey = 'sample';
                label = 'Sampling H3 ControlNet';
                if (!state.samplingStartedAt)
                    state.samplingStartedAt = Date.now();
            }
            else if (/decode/i.test(nodeType)) {
                phaseKey = 'decode';
                label = 'Decoding synchronized video and audio';
            }
            else if (/createvideo|savevideo/i.test(nodeType)) {
                phaseKey = 'save';
                label = /savevideo/i.test(nodeType) ? 'Saving video' : 'Encoding output video';
            }
            else if (/loader/i.test(nodeType))
                label = 'Loading H3 base';
            setPhase(phaseKey);
            setStatus(label, Number(els.progressValue.textContent.replace('%', '')) || 0, state.sampleStep, state.sampleMax);
        });
        SerenityH3WS.on('progress', function (data) {
            if (!isOwn(data))
                return;
            state.ownExecuting = true;
            els.cancel.disabled = false;
            var percent = data.max > 0 ? data.value / data.max * 100 : 0;
            var phase = data.phase || data.message || 'Sampling H3 ControlNet';
            if (!state.samplingStartedAt)
                state.samplingStartedAt = Date.now();
            setPhase(stageForPhase(data.phase || data.message));
            setStatus(phase, percent, Number(data.value || 0), Number(data.max || 0));
        });
        SerenityH3WS.on('executed', function (data) {
            if (!isOwn(data) || !data.output)
                return;
            displayResult(data.output);
        });
        SerenityH3WS.on('execution_success', function (data) {
            if (!isOwn(data))
                return;
            state.ownExecuting = false;
            setRunning(false);
            setPhase('complete');
            setStatus('Complete', 100, state.sampleMax || state.sampleStep, state.sampleMax || state.sampleStep);
            loadRecent();
            state.promptId = '';
        });
        SerenityH3WS.on('execution_error', function (data) {
            handleOwnExecutionError(data);
        });
        SerenityH3WS.on('status', function (data) {
            var lastError = data && data.status && data.status.last_error;
            if (lastError) {
                handleOwnExecutionError(lastError);
                state.connectionInterrupted = false;
                return;
            }
            if (!state.connectionInterrupted || !state.promptId || !data || !data.status)
                return;
            var activeIds = (data.status.queue_running || []).concat(data.status.queue_pending || [])
                .map(function (item) { return String(item && (item.prompt_id || item) || ''); });
            state.connectionInterrupted = false;
            if (activeIds.indexOf(state.promptId) >= 0) {
                setRunning(true);
                setStatus('Reconnected · queued / running', Number(els.progressValue.textContent.replace('%', '')) || 0);
                return;
            }
            var endedPromptId = state.promptId;
            state.promptId = '';
            setRunning(false);
            setPhase('ready');
            setStatus('Ended while disconnected · check History', 0);
            loadRecent();
            if (typeof showExecutionIssue === 'function')
                showExecutionIssue({
                    prompt_id: endedPromptId,
                    node_type: 'Server',
                    exception_message: 'Workflow ended while disconnected. Check History for its result or error.'
                });
        });
        SerenityH3WS.on('execution_interrupted', function (data) {
            if (!isOwn(data))
                return;
            state.ownExecuting = false;
            setRunning(false);
            setPhase('ready');
            setStatus('Cancelled', 0);
            state.promptId = '';
        });
    }
    function init() {
        if (initialized) {
            loadModels();
            return;
        }
        initialized = true;
        render();
        cacheElements();
        bindEvents();
        restoreDraft();
        bindWS();
        updateContract();
        renderPreviewDeck();
        setPhase('ready');
        loadRecent();
        loadModels();
    }
    return {
        init: init,
        generate: generate,
        cancel: cancel,
        syncFromWorkflow: syncFromWorkflow,
        syncToWorkflow: syncWorkflowFromControls,
        state: state
    };
})();
//# sourceMappingURL=h3-control.js.map
