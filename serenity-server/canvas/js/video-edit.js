"use strict";
/**
 * VideoEditTab — AI-native non-linear video editor with Konva.js timeline.
 * Phase V1: Tab shell + timeline rendering.
 * Phase V2: Clip selection, drag, trim, snap, undo, context menu, backend CRUD.
 * Phase V3: Preview player, thumbnail strips, waveform display, synchronized playback.
 * Phase V4: AI retake, bridge shots, multi-take nesting.
 * Phase V5: Export, SRT subtitles, XML import/export.
 */
var VideoEditTab = (function () {
    // --- Constants ---
    var FPS = 30;
    var TRACK_HEIGHT = 48;
    var RULER_HEIGHT = 28;
    var TRACK_HEADER_WIDTH = 120;
    var MIN_PPF = 0.5;
    var MAX_PPF = 20;
    var DEFAULT_PPF = 4;
    var SNAP_THRESHOLD = 5;       // frames
    var TRIM_HANDLE_WIDTH = 6;    // px
    var MAX_UNDO = 50;
    var AUTOSAVE_DELAY = 2000;    // ms

    // --- V6: Effect Registry ---
    var EFFECT_REGISTRY = {
        brightness: { name: 'Brightness', category: 'color', defaults: { value: 0 }, range: { value: [-1, 1, 0.05] } },
        contrast:   { name: 'Contrast',   category: 'color', defaults: { value: 1 }, range: { value: [0.2, 3, 0.05] } },
        saturation: { name: 'Saturation', category: 'color', defaults: { value: 1 }, range: { value: [0, 3, 0.05] } },
        hue:        { name: 'Hue Shift',  category: 'color', defaults: { degrees: 0 }, range: { degrees: [-180, 180, 1] } },
        gamma:      { name: 'Gamma',      category: 'color', defaults: { value: 1 }, range: { value: [0.1, 4, 0.05] } },
        blur:       { name: 'Blur',       category: 'stylize', defaults: { radius: 5 }, range: { radius: [0, 30, 1] } },
        sharpen:    { name: 'Sharpen',    category: 'stylize', defaults: { amount: 1 }, range: { amount: [0, 5, 0.1] } },
        denoise:    { name: 'Denoise',    category: 'stylize', defaults: { strength: 4 }, range: { strength: [1, 15, 1] } },
        glow:       { name: 'Glow',       category: 'stylize', defaults: { radius: 10 }, range: { radius: [1, 40, 1] } },
        vignette:   { name: 'Vignette',   category: 'stylize', defaults: { angle: 0.4 }, range: { angle: [0, 1, 0.05] } },
        speed:      { name: 'Speed',      category: 'utility', defaults: { rate: 1 }, range: { rate: [0.25, 4, 0.25] } },
        opacity:    { name: 'Opacity',    category: 'utility', defaults: { value: 1 }, range: { value: [0, 1, 0.05] } },
        flip_h:     { name: 'Flip H',     category: 'utility', defaults: {}, range: {} },
        flip_v:     { name: 'Flip V',     category: 'utility', defaults: {}, range: {} },
    };

    var TRANSITION_TYPES = [
        { type: 'none', name: 'None' },
        { type: 'fade', name: 'Fade' },
        { type: 'dissolve', name: 'Dissolve' },
        { type: 'wipeleft', name: 'Wipe Left' },
        { type: 'wiperight', name: 'Wipe Right' },
        { type: 'wipeup', name: 'Wipe Up' },
        { type: 'wipedown', name: 'Wipe Down' },
        { type: 'slideleft', name: 'Slide Left' },
        { type: 'slideright', name: 'Slide Right' },
        { type: 'circleopen', name: 'Circle Open' },
        { type: 'circleclose', name: 'Circle Close' },
        { type: 'fadeblack', name: 'Fade Black' },
        { type: 'fadewhite', name: 'Fade White' },
    ];

    // --- State ---
    var stage = null;
    var rulerLayer = null;
    var timelineLayer = null;
    var overlayLayer = null;

    var pixelsPerFrame = DEFAULT_PPF;
    var scrollOffsetX = 0;
    var currentFrame = 0;
    var totalFrames = 900;
    var isPlaying = false;
    var isScrubbing = false;

    var playStartTime = null;
    var playStartFrame = 0;
    var animFrameId = null;

    var _initialized = false;

    // --- V2 State ---
    var selectedClipIds = new Set();
    var undoStack = [];
    var redoStack = [];

    // Drag state
    var isDragging = false;
    var dragClipId = null;
    var dragStartX = 0;
    var dragStartY = 0;
    var dragOriginals = {};    // clipId -> { startFrame, endFrame, trackId }
    var dragTrackOffset = 0;   // vertical track offset during drag

    // Trim state
    var isTrimming = false;
    var trimClipId = null;
    var trimTrackId = null;
    var trimEdge = null;       // 'left' or 'right'
    var trimOrigStart = 0;
    var trimOrigEnd = 0;
    var trimStartMouseX = 0;

    // Snap state
    var snapPoints = [];
    var activeSnapFrame = null;

    // Context menu
    var contextMenuEl = null;

    // Backend
    var projectId = null;
    var autosaveTimer = null;

    // Clip ID counter
    var nextClipNum = 100;

    // --- V3 State: Preview + Thumbnails + Waveforms ---
    var previewCanvas = null;
    var previewCtx = null;
    var previewVideo = null;
    var previewTcEl = null;
    var previewActiveClipId = null;
    var previewDebounceTimer = null;
    var PREVIEW_DEBOUNCE_MS = 66; // ~15fps max during scrub

    var thumbnailCache = new Map();   // clipId -> { img, thumbW, thumbH, frameCount }
    var thumbnailLoading = new Set(); // clipIds currently loading

    var waveformCache = new Map();    // clipId -> { peaks[], sample_rate, duration_seconds }
    var waveformLoading = new Set();  // clipIds currently loading

    var audioElements = new Map();    // clipId -> HTMLAudioElement

    // --- V4 State: AI Features ---
    var retakeMode = false;
    var retakeClipId = null;
    var retakeRegionStart = 0;
    var retakeRegionEnd = 0;
    var retakePanelEl = null;
    var bridgePanelEl = null;
    var takeDropdownEl = null;
    var generatingClips = new Set();  // clipIds currently generating

    // --- V5 State: Export + SRT + XML ---
    var exportDialogEl = null;
    var exportId = null;
    var isExporting = false;
    var fileMenuEl = null;
    var subtitleEditorEl = null;
    var subtitleEditingClipId = null;

    // Hidden file inputs for SRT/XML import
    var srtFileInput = null;
    var xmlFileInput = null;

    // --- V6 State: Effects & Properties Panel ---
    var propsPanelEl = null;
    var propsPanelClipId = null;
    var addEffectDropdownEl = null;

    // --- V7 State: RIFE Interpolation ---
    var activeRifeJobId = null;
    var rifeProcessingClipId = null;
    var rifeProgressPercent = 0;

    // --- V8 State: Face Restoration ---
    var activeFaceJobId = null;
    var faceProcessingClipId = null;
    var faceProgressPercent = 0;
    var faceDialogEl = null;

    // --- V9 State: ESRGAN Upscale ---
    var activeEsrganJobId = null;
    var esrganProcessingClipId = null;
    var esrganProgressPercent = 0;
    var esrganDialogEl = null;

    // --- V10 State: Deflicker ---
    var activeDeflickerJobId = null;
    var deflickerProcessingClipId = null;
    var deflickerProgressPercent = 0;
    var deflickerDialogEl = null;

    // --- V11 State: Audio Enhancement ---
    var activeAudioJobId = null;
    var audioProcessingClipId = null;
    var audioProgressPercent = 0;
    var audioDialogEl = null;
    var audioPresetsCache = null;

    // --- Default project data ---
    var project = {
        fps: 30,
        name: 'Untitled Project',
        tracks: [
            {
                id: 'track-1', name: 'Video 1', type: 'video',
                clips: [
                    { id: 'clip-1', startFrame: 0, endFrame: 150, label: 'Intro', color: '#4a7dff' },
                    { id: 'clip-2', startFrame: 180, endFrame: 450, label: 'Scene 1', color: '#4a7dff' }
                ]
            },
            {
                id: 'track-2', name: 'Video 2', type: 'video',
                clips: [
                    { id: 'clip-3', startFrame: 90, endFrame: 300, label: 'Overlay', color: '#7b4aff' }
                ]
            },
            {
                id: 'track-3', name: 'Audio', type: 'audio',
                clips: [
                    { id: 'clip-4', startFrame: 0, endFrame: 900, label: 'Music.mp3', color: '#2a9d5c' }
                ]
            },
            {
                id: 'track-4', name: 'Subtitles', type: 'text',
                clips: [
                    { id: 'clip-5', startFrame: 30, endFrame: 120, label: 'Hello world', color: '#d4a72c' },
                    { id: 'clip-6', startFrame: 200, endFrame: 350, label: 'Second line', color: '#d4a72c' }
                ]
            }
        ]
    };

    // ===== Utilities =====

    function frameToTimecode(frame) {
        var totalSeconds = Math.floor(frame / FPS);
        var mm = String(Math.floor(totalSeconds / 60)).padStart(2, '0');
        var ss = String(totalSeconds % 60).padStart(2, '0');
        var ff = String(frame % FPS).padStart(2, '0');
        return mm + ':' + ss + ':' + ff;
    }

    function clamp(val, min, max) {
        return Math.max(min, Math.min(max, val));
    }

    function getMaxScroll() {
        if (!stage) return 0;
        return Math.max(0, totalFrames * pixelsPerFrame - (stage.width() - TRACK_HEADER_WIDTH));
    }

    function generateClipId() {
        return 'clip-' + (nextClipNum++);
    }

    function getApiBase() {
        return window.location.protocol + '//' + window.location.host;
    }

    function findClipById(clipId) {
        for (var i = 0; i < project.tracks.length; i++) {
            var track = project.tracks[i];
            for (var j = 0; j < track.clips.length; j++) {
                if (track.clips[j].id === clipId) {
                    return { clip: track.clips[j], track: track, clipIndex: j, trackIndex: i };
                }
            }
        }
        return null;
    }

    function pixelToFrame(px) {
        return Math.round((px - TRACK_HEADER_WIDTH + scrollOffsetX) / pixelsPerFrame);
    }

    function trackIndexAtY(y) {
        var idx = Math.floor((y - RULER_HEIGHT) / TRACK_HEIGHT);
        return clamp(idx, 0, project.tracks.length - 1);
    }

    function recalcTotalFrames() {
        var maxFrame = 300; // minimum 10s
        project.tracks.forEach(function (track) {
            track.clips.forEach(function (clip) {
                if (clip.endFrame > maxFrame) maxFrame = clip.endFrame;
            });
        });
        totalFrames = maxFrame + FPS * 5; // 5s padding
        var el = document.getElementById('ve-duration');
        if (el) el.textContent = frameToTimecode(totalFrames);
    }

    // ===== Undo/Redo =====

    function pushUndo() {
        undoStack.push(JSON.parse(JSON.stringify(project.tracks)));
        if (undoStack.length > MAX_UNDO) undoStack.shift();
        redoStack.length = 0;
    }

    function undo() {
        if (!undoStack.length) return;
        redoStack.push(JSON.parse(JSON.stringify(project.tracks)));
        project.tracks = undoStack.pop();
        selectedClipIds.clear();
        recalcTotalFrames();
        renderTimeline();
        refreshPropertiesPanel();
        scheduleAutosave();
    }

    function redo() {
        if (!redoStack.length) return;
        undoStack.push(JSON.parse(JSON.stringify(project.tracks)));
        project.tracks = redoStack.pop();
        selectedClipIds.clear();
        recalcTotalFrames();
        renderTimeline();
        refreshPropertiesPanel();
        scheduleAutosave();
    }

    // ===== Snap =====

    function collectSnapPoints() {
        var points = new Set();
        points.add(currentFrame); // playhead

        project.tracks.forEach(function (track) {
            track.clips.forEach(function (clip) {
                if (!selectedClipIds.has(clip.id)) {
                    points.add(clip.startFrame);
                    points.add(clip.endFrame);
                }
            });
        });

        // Second boundaries
        for (var f = 0; f <= totalFrames; f += FPS) {
            points.add(f);
        }

        snapPoints = Array.from(points);
    }

    function findSnap(frame) {
        var best = null;
        var bestDist = SNAP_THRESHOLD + 1;
        for (var i = 0; i < snapPoints.length; i++) {
            var dist = Math.abs(snapPoints[i] - frame);
            if (dist < bestDist) {
                bestDist = dist;
                best = snapPoints[i];
            }
        }
        return best;
    }

    // ===== Backend =====

    function scheduleAutosave() {
        if (autosaveTimer) clearTimeout(autosaveTimer);
        autosaveTimer = setTimeout(function () {
            saveProject();
        }, AUTOSAVE_DELAY);
    }

    function saveProject() {
        if (!projectId) return;
        var body = {
            name: project.name,
            fps: project.fps,
            width: project.width || 1280,
            height: project.height || 720,
            tracks: project.tracks
        };
        fetch(getApiBase() + '/video_edit/projects/' + projectId, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        }).catch(function (err) {
            console.warn('VideoEdit autosave failed:', err);
        });
    }

    function loadOrCreateProject() {
        // Try loading last project ID from localStorage
        var savedId = localStorage.getItem('ve-project-id');
        if (savedId) {
            fetch(getApiBase() + '/video_edit/projects/' + savedId)
                .then(function (r) {
                    if (!r.ok) throw new Error('not found');
                    return r.json();
                })
                .then(function (data) {
                    projectId = data.id;
                    project.name = data.name || 'Untitled Project';
                    project.fps = data.fps || 30;
                    if (data.tracks && data.tracks.length > 0) {
                        project.tracks = data.tracks;
                    }
                    recalcTotalFrames();
                    renderTimeline();
                })
                .catch(function () {
                    // Project not found, create new
                    createNewProject();
                });
        } else {
            createNewProject();
        }
    }

    function createNewProject() {
        fetch(getApiBase() + '/video_edit/projects', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: project.name,
                fps: project.fps,
                tracks: project.tracks
            })
        })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            projectId = data.id;
            localStorage.setItem('ve-project-id', projectId);
        })
        .catch(function (err) {
            console.warn('VideoEdit: could not create project on server:', err);
        });
    }

    // ===== DOM Building =====

    function buildToolbar() {
        var toolbar = document.getElementById('ve-toolbar');
        if (!toolbar) return;
        toolbar.innerHTML =
            '<div class="ve-toolbar-inner">' +
                '<div style="position:relative">' +
                    '<button id="ve-btn-file" class="ve-tb-btn" title="File" style="font-size:11px;width:auto;padding:0 8px;">File</button>' +
                    '<div id="ve-file-dropdown" class="ve-file-dropdown" style="display:none"></div>' +
                '</div>' +
                '<button id="ve-btn-start" class="ve-tb-btn" title="Go to start (Home)">&#9198;</button>' +
                '<button id="ve-btn-play" class="ve-tb-btn" title="Play/Pause (Space)">&#9654;</button>' +
                '<button id="ve-btn-stop" class="ve-tb-btn" title="Stop">&#9632;</button>' +
                '<button id="ve-btn-end" class="ve-tb-btn" title="Go to end (End)">&#9197;</button>' +
                '<span id="ve-timecode" class="ve-timecode">00:00:00</span>' +
                '<span class="ve-spacer"></span>' +
                '<button id="ve-btn-retake" class="ve-tb-btn" title="Retake Selection" style="font-size:11px;width:auto;padding:0 8px;">Retake</button>' +
                '<span class="ve-spacer"></span>' +
                '<span id="ve-zoom-label" class="ve-zoom-label">100%</span>' +
                '<input id="ve-zoom-slider" type="range" min="5" max="2000" value="400" class="ve-zoom-slider">' +
                '<span id="ve-duration" class="ve-timecode">' + frameToTimecode(totalFrames) + '</span>' +
                '<button id="ve-btn-export" class="ve-tb-btn" title="Export" style="font-size:11px;width:auto;padding:0 8px;">Export</button>' +
            '</div>';

        var btnPlay = document.getElementById('ve-btn-play');
        var btnStop = document.getElementById('ve-btn-stop');
        var btnStart = document.getElementById('ve-btn-start');
        var btnEnd = document.getElementById('ve-btn-end');
        var zoomSlider = document.getElementById('ve-zoom-slider');

        var btnRetake = document.getElementById('ve-btn-retake');
        if (btnRetake) btnRetake.addEventListener('click', function () {
            if (selectedClipIds.size === 1) {
                var clipId = selectedClipIds.values().next().value;
                enterRetakeMode(clipId);
            }
        });
        var btnExport = document.getElementById('ve-btn-export');
        if (btnExport) btnExport.addEventListener('click', showExportDialog);
        var btnFile = document.getElementById('ve-btn-file');
        if (btnFile) btnFile.addEventListener('click', toggleFileMenu);
        if (btnPlay) btnPlay.addEventListener('click', togglePlayback);
        if (btnStop) btnStop.addEventListener('click', function () {
            stopPlayback();
            currentFrame = 0;
            scrollOffsetX = 0;
            updateTimecodeDisplay();
            renderTimeline();
            updatePreview();
        });
        if (btnStart) btnStart.addEventListener('click', function () {
            stopPlayback();
            currentFrame = 0;
            scrollOffsetX = 0;
            updateTimecodeDisplay();
            renderTimeline();
            updatePreview();
        });
        if (btnEnd) btnEnd.addEventListener('click', function () {
            stopPlayback();
            currentFrame = totalFrames;
            var viewWidth = stage ? stage.width() - TRACK_HEADER_WIDTH : 500;
            scrollOffsetX = Math.max(0, currentFrame * pixelsPerFrame - viewWidth + 60);
            updateTimecodeDisplay();
            renderTimeline();
            updatePreview();
        });
        if (zoomSlider) {
            zoomSlider.addEventListener('input', function () {
                var val = parseFloat(this.value);
                pixelsPerFrame = val / 100;
                pixelsPerFrame = clamp(pixelsPerFrame, MIN_PPF, MAX_PPF);
                scrollOffsetX = clamp(scrollOffsetX, 0, getMaxScroll());
                updateZoomDisplay();
                renderTimeline();
            });
        }
    }

    function buildContextMenu() {
        contextMenuEl = document.createElement('div');
        contextMenuEl.id = 've-context-menu';
        contextMenuEl.className = 've-context-menu';
        contextMenuEl.style.display = 'none';
        document.getElementById('panel-video-edit').appendChild(contextMenuEl);
    }

    function showContextMenu(x, y, items) {
        if (!contextMenuEl) return;
        var html = '';
        items.forEach(function (item) {
            if (item.separator) {
                html += '<div class="ve-ctx-sep"></div>';
            } else {
                html += '<div class="ve-ctx-item" data-action="' + item.action + '">' + item.label + '</div>';
            }
        });
        contextMenuEl.innerHTML = html;
        contextMenuEl.style.display = 'block';
        contextMenuEl.style.left = x + 'px';
        contextMenuEl.style.top = y + 'px';

        // Bind click handlers
        contextMenuEl.querySelectorAll('.ve-ctx-item').forEach(function (el) {
            el.addEventListener('click', function (e) {
                var action = e.target.dataset.action;
                hideContextMenu();
                handleContextAction(action);
            });
        });
    }

    function hideContextMenu() {
        if (contextMenuEl) contextMenuEl.style.display = 'none';
    }

    var contextTargetClipId = null;
    var contextTargetTrackId = null;
    var contextClickFrame = 0;

    function handleContextAction(action) {
        if (action === 'delete') {
            deleteSelectedClips();
        } else if (action === 'split') {
            splitClipAtPlayhead(contextTargetClipId, contextTargetTrackId);
        } else if (action === 'duplicate') {
            duplicateClip(contextTargetClipId, contextTargetTrackId);
        } else if (action === 'properties') {
            openPropertiesPanel(contextTargetClipId);
        } else if (action === 'add-track') {
            addTrack();
        } else if (action === 'add-clip') {
            addPlaceholderClip(contextTargetTrackId, contextClickFrame);
        } else if (action === 'retake') {
            enterRetakeMode(contextTargetClipId);
        } else if (action === 'switch-take') {
            showTakeSwitcher(contextTargetClipId);
        } else if (action === 'fill-gap') {
            showBridgePanel(contextTargetTrackId, contextClickFrame);
        } else if (action === 'rife-2x' || action === 'rife-4x') {
            var multi = action === 'rife-2x' ? 2 : 4;
            startRifeInterpolation(contextTargetClipId, multi);
        } else if (action === 'face-restore') {
            showFaceRestoreDialog(contextTargetClipId);
        } else if (action === 'esrgan-upscale') {
            showEsrganDialog(contextTargetClipId);
        } else if (action === 'deflicker') {
            showDeflickerDialog(contextTargetClipId);
        } else if (action === 'audio-enhance') {
            showAudioEnhanceDialog(contextTargetClipId);
        }
    }

    // ===== Clip Operations =====

    function deleteSelectedClips() {
        if (selectedClipIds.size === 0) return;
        // Close properties panel if its clip is being deleted
        if (propsPanelClipId && selectedClipIds.has(propsPanelClipId)) {
            closePropertiesPanel();
        }
        pushUndo();
        project.tracks.forEach(function (track) {
            track.clips = track.clips.filter(function (clip) {
                return !selectedClipIds.has(clip.id);
            });
        });
        selectedClipIds.clear();
        recalcTotalFrames();
        renderTimeline();
        scheduleAutosave();
    }

    function splitClipAtPlayhead(clipId, trackId) {
        if (!clipId) return;
        var info = findClipById(clipId);
        if (!info) return;
        var clip = info.clip;
        if (currentFrame <= clip.startFrame || currentFrame >= clip.endFrame) return;

        pushUndo();
        var newClip = {
            id: generateClipId(),
            startFrame: currentFrame,
            endFrame: clip.endFrame,
            label: clip.label + ' (R)',
            color: clip.color,
            effects: clip.effects ? JSON.parse(JSON.stringify(clip.effects)) : [],
            transition_in: null,
        };
        clip.endFrame = currentFrame;
        clip.label = clip.label.replace(/ \([LR]\)$/, '') + ' (L)';
        info.track.clips.splice(info.clipIndex + 1, 0, newClip);
        renderTimeline();
        scheduleAutosave();
    }

    function duplicateClip(clipId, trackId) {
        var info = findClipById(clipId);
        if (!info) return;
        pushUndo();
        var clip = info.clip;
        var duration = clip.endFrame - clip.startFrame;
        var newClip = {
            id: generateClipId(),
            startFrame: clip.endFrame,
            endFrame: clip.endFrame + duration,
            label: clip.label + ' copy',
            color: clip.color,
            effects: clip.effects ? JSON.parse(JSON.stringify(clip.effects)) : [],
            transition_in: clip.transition_in ? JSON.parse(JSON.stringify(clip.transition_in)) : null,
        };
        info.track.clips.push(newClip);
        recalcTotalFrames();
        renderTimeline();
        scheduleAutosave();
    }

    function addTrack() {
        pushUndo();
        var num = project.tracks.filter(function (t) { return t.type === 'video'; }).length + 1;
        project.tracks.push({
            id: 'track-' + Date.now(),
            name: 'Video ' + num,
            type: 'video',
            clips: []
        });
        recalcTotalFrames();
        resize();
        scheduleAutosave();
    }

    function addPlaceholderClip(trackId, frame) {
        var track = null;
        for (var i = 0; i < project.tracks.length; i++) {
            if (project.tracks[i].id === trackId) { track = project.tracks[i]; break; }
        }
        if (!track) return;
        pushUndo();
        var duration = FPS * 5; // 5 seconds
        track.clips.push({
            id: generateClipId(),
            startFrame: Math.max(0, frame),
            endFrame: Math.max(0, frame) + duration,
            label: 'New Clip',
            color: track.type === 'audio' ? '#2a9d5c' : track.type === 'text' ? '#d4a72c' : '#4a7dff',
            effects: [],
            transition_in: null,
        });
        recalcTotalFrames();
        renderTimeline();
        scheduleAutosave();
    }

    function selectAllClips() {
        selectedClipIds.clear();
        project.tracks.forEach(function (track) {
            track.clips.forEach(function (clip) {
                selectedClipIds.add(clip.id);
            });
        });
        renderTimeline();
    }

    // ===== Preview Player =====

    function initPreview() {
        var container = document.getElementById('ve-preview');
        if (!container) return;
        container.innerHTML =
            '<video id="ve-preview-video" muted preload="auto"></video>' +
            '<canvas id="ve-preview-canvas"></canvas>' +
            '<div id="ve-preview-subtitle"></div>' +
            '<div id="ve-preview-overlay"><span id="ve-preview-tc">00:00:00</span></div>' +
            '<span id="ve-preview-placeholder">No clip at playhead</span>';

        previewVideo = document.getElementById('ve-preview-video');
        previewCanvas = document.getElementById('ve-preview-canvas');
        previewCtx = previewCanvas ? previewCanvas.getContext('2d') : null;
        previewTcEl = document.getElementById('ve-preview-tc');

        // Size canvas to 16:9 within container
        resizePreview();
    }

    function resizePreview() {
        if (!previewCanvas) return;
        var container = document.getElementById('ve-preview');
        if (!container) return;
        var rect = container.getBoundingClientRect();
        // Fit 16:9 inside container
        var targetW = rect.width;
        var targetH = rect.width * 9 / 16;
        if (targetH > rect.height) {
            targetH = rect.height;
            targetW = rect.height * 16 / 9;
        }
        previewCanvas.width = Math.round(targetW);
        previewCanvas.height = Math.round(targetH);
    }

    function findActiveClipAtFrame(frame) {
        // Topmost video clip at frame (track 0 = top of timeline = highest priority)
        for (var i = 0; i < project.tracks.length; i++) {
            var track = project.tracks[i];
            if (track.type !== 'video') continue;
            for (var j = 0; j < track.clips.length; j++) {
                var clip = track.clips[j];
                if (frame >= clip.startFrame && frame < clip.endFrame) {
                    return clip;
                }
            }
        }
        return null;
    }

    function updatePreview() {
        if (!previewCtx || !previewCanvas) return;
        var placeholder = document.getElementById('ve-preview-placeholder');

        var clip = findActiveClipAtFrame(currentFrame);

        if (!clip) {
            // Black frame — no clip at playhead
            previewCtx.fillStyle = '#111';
            previewCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
            previewActiveClipId = null;
            if (placeholder) placeholder.style.display = '';
            if (previewTcEl) previewTcEl.textContent = frameToTimecode(currentFrame);
            renderSubtitleOverlay();
            return;
        }

        if (placeholder) placeholder.style.display = 'none';

        // Placeholder clip (no source file) — show color card with label
        if (!clip.source_path) {
            previewActiveClipId = clip.id;
            previewCtx.fillStyle = clip.color || '#333';
            previewCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
            previewCtx.fillStyle = '#fff';
            previewCtx.font = '20px sans-serif';
            previewCtx.textAlign = 'center';
            previewCtx.textBaseline = 'middle';
            previewCtx.fillText(clip.label || 'Untitled', previewCanvas.width / 2, previewCanvas.height / 2);
            previewCtx.textAlign = 'start';
            if (previewTcEl) previewTcEl.textContent = frameToTimecode(currentFrame);
            renderSubtitleOverlay();
            return;
        }

        var sourceOffset = clip.source_start || 0;
        var frameInSource = currentFrame - clip.startFrame + sourceOffset;
        var timeInSource = frameInSource / FPS;

        // Load video source if clip changed
        if (previewActiveClipId !== clip.id && previewVideo) {
            var relPath = clip.source_path;
            if (relPath.indexOf('/') === 0 || relPath.indexOf('\\') >= 0) {
                previewVideo.src = getApiBase() + '/video_edit/media/' + encodeURIComponent(relPath.split('/video_projects/').pop() || relPath);
            } else {
                previewVideo.src = getApiBase() + '/video_edit/media/' + relPath;
            }
            previewActiveClipId = clip.id;
            previewVideo.load();
        }

        if (previewVideo && previewVideo.readyState >= 1) {
            previewVideo.currentTime = timeInSource;
            previewVideo.onseeked = function () {
                if (previewCtx && previewCanvas) {
                    previewCtx.fillStyle = '#111';
                    previewCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
                    // Center the video frame
                    var vw = previewVideo.videoWidth || previewCanvas.width;
                    var vh = previewVideo.videoHeight || previewCanvas.height;
                    var scale = Math.min(previewCanvas.width / vw, previewCanvas.height / vh);
                    var dw = vw * scale;
                    var dh = vh * scale;
                    var dx = (previewCanvas.width - dw) / 2;
                    var dy = (previewCanvas.height - dh) / 2;
                    previewCtx.drawImage(previewVideo, dx, dy, dw, dh);
                }
            };
        }

        // Update timecode overlay
        if (previewTcEl) {
            previewTcEl.textContent = frameToTimecode(currentFrame);
        }

        // Subtitle overlay
        renderSubtitleOverlay();
    }

    function updatePreviewDebounced() {
        if (previewDebounceTimer) return; // already pending
        previewDebounceTimer = setTimeout(function () {
            previewDebounceTimer = null;
            updatePreview();
        }, PREVIEW_DEBOUNCE_MS);
    }

    function _projectsMediaPrefix() {
        return '/video_edit/media/';
    }

    // ===== Thumbnail Loading =====

    function loadThumbnails(clip) {
        if (!clip.source_path || thumbnailCache.has(clip.id) || thumbnailLoading.has(clip.id)) return;
        thumbnailLoading.add(clip.id);

        fetch(getApiBase() + '/video_edit/thumbnails', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                source_path: clip.source_path,
                height: TRACK_HEIGHT - 12,
            }),
        })
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function (data) {
            thumbnailLoading.delete(clip.id);
            if (!data || !data.sprite_url) return;

            var img = new window.Image();
            img.crossOrigin = 'anonymous';
            img.src = getApiBase() + data.sprite_url;
            img.onload = function () {
                thumbnailCache.set(clip.id, {
                    img: img,
                    thumbW: data.thumb_width,
                    thumbH: data.thumb_height,
                    frameCount: data.frame_count,
                });
                renderTracks();
            };
        })
        .catch(function () {
            thumbnailLoading.delete(clip.id);
        });
    }

    // ===== Waveform Loading =====

    function loadWaveform(clip) {
        if (!clip.source_path || waveformCache.has(clip.id) || waveformLoading.has(clip.id)) return;
        waveformLoading.add(clip.id);

        fetch(getApiBase() + '/video_edit/waveform', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                source_path: clip.source_path,
                samples_per_second: 30,
            }),
        })
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function (data) {
            waveformLoading.delete(clip.id);
            if (!data || !data.peaks) return;
            waveformCache.set(clip.id, data);
            renderTracks();
        })
        .catch(function () {
            waveformLoading.delete(clip.id);
        });
    }

    // ===== Audio Sync =====

    function syncAudio() {
        project.tracks.forEach(function (track) {
            track.clips.forEach(function (clip) {
                if (!clip.source_path) return;
                if (track.type !== 'audio' && track.type !== 'video') return;

                var shouldPlay = isPlaying &&
                    currentFrame >= clip.startFrame &&
                    currentFrame < clip.endFrame;

                var audio = audioElements.get(clip.id);

                if (shouldPlay) {
                    if (!audio) {
                        audio = new Audio();
                        var relPath = clip.source_path;
                        if (relPath.indexOf('/') === 0 || relPath.indexOf('\\') >= 0) {
                            audio.src = getApiBase() + '/video_edit/media/' + encodeURIComponent(relPath.split('/video_projects/').pop() || relPath);
                        } else {
                            audio.src = getApiBase() + '/video_edit/media/' + relPath;
                        }
                        audioElements.set(clip.id, audio);
                    }
                    var sourceTime = (currentFrame - clip.startFrame + (clip.source_start || 0)) / FPS;
                    // Correct drift > 0.1s
                    if (Math.abs(audio.currentTime - sourceTime) > 0.1) {
                        audio.currentTime = sourceTime;
                    }
                    if (audio.paused) {
                        audio.play().catch(function () {}); // ignore autoplay blocks
                    }
                } else {
                    if (audio && !audio.paused) {
                        audio.pause();
                    }
                }
            });
        });
    }

    function stopAllAudio() {
        audioElements.forEach(function (audio) {
            if (!audio.paused) audio.pause();
        });
    }

    // ===== V4: Gap Detection =====

    function findGapAtFrame(trackId, frame) {
        var track = null;
        for (var i = 0; i < project.tracks.length; i++) {
            if (project.tracks[i].id === trackId) { track = project.tracks[i]; break; }
        }
        if (!track) return null;

        // Is there a clip at this frame?
        for (var j = 0; j < track.clips.length; j++) {
            var c = track.clips[j];
            if (frame >= c.startFrame && frame < c.endFrame) return null;
        }

        // Find clip before and after
        var before = null, after = null;
        track.clips.forEach(function (c) {
            if (c.endFrame <= frame && (!before || c.endFrame > before.endFrame)) before = c;
            if (c.startFrame >= frame && (!after || c.startFrame < after.startFrame)) after = c;
        });

        return {
            before: before,
            after: after,
            gapStart: before ? before.endFrame : 0,
            gapEnd: after ? after.startFrame : totalFrames,
        };
    }

    // ===== V4: Retake Mode =====

    function buildRetakePanel() {
        retakePanelEl = document.createElement('div');
        retakePanelEl.className = 've-retake-panel';
        retakePanelEl.innerHTML =
            '<h4>Retake Selection</h4>' +
            '<textarea class="ve-retake-prompt" placeholder="Describe the correction..."></textarea>' +
            '<div class="ve-retake-row">' +
                '<label>Strength</label>' +
                '<input type="range" min="30" max="100" value="70" class="ve-retake-strength">' +
                '<span class="ve-retake-val">0.70</span>' +
            '</div>' +
            '<div class="ve-retake-row">' +
                '<label>Region</label>' +
                '<span class="ve-retake-region-info"></span>' +
            '</div>' +
            '<div class="ve-retake-actions">' +
                '<button class="ve-retake-btn ve-retake-btn-secondary ve-retake-cancel">Cancel</button>' +
                '<button class="ve-retake-btn ve-retake-btn-primary ve-retake-generate">Generate Retake</button>' +
            '</div>';
        document.getElementById('panel-video-edit').appendChild(retakePanelEl);

        var strengthSlider = retakePanelEl.querySelector('.ve-retake-strength');
        var strengthVal = retakePanelEl.querySelector('.ve-retake-val');
        strengthSlider.addEventListener('input', function () {
            strengthVal.textContent = (parseInt(this.value) / 100).toFixed(2);
        });

        retakePanelEl.querySelector('.ve-retake-cancel').addEventListener('click', exitRetakeMode);
        retakePanelEl.querySelector('.ve-retake-generate').addEventListener('click', submitRetake);
    }

    function enterRetakeMode(clipId) {
        var info = findClipById(clipId);
        if (!info) return;
        var clip = info.clip;

        retakeMode = true;
        retakeClipId = clipId;

        // Default region: middle 50%
        var duration = clip.endFrame - clip.startFrame;
        retakeRegionStart = clip.startFrame + Math.floor(duration * 0.25);
        retakeRegionEnd = clip.endFrame - Math.floor(duration * 0.25);

        // Show panel
        if (!retakePanelEl) buildRetakePanel();
        retakePanelEl.style.display = 'block';
        retakePanelEl.style.left = '50%';
        retakePanelEl.style.top = '50%';
        retakePanelEl.style.transform = 'translate(-50%, -50%)';

        updateRetakeRegionInfo();
        renderTimeline();
    }

    function exitRetakeMode() {
        retakeMode = false;
        retakeClipId = null;
        if (retakePanelEl) retakePanelEl.style.display = 'none';
        renderTimeline();
    }

    function updateRetakeRegionInfo() {
        if (!retakePanelEl) return;
        var el = retakePanelEl.querySelector('.ve-retake-region-info');
        if (el) {
            el.textContent = frameToTimecode(retakeRegionStart) + ' - ' + frameToTimecode(retakeRegionEnd) +
                ' (' + ((retakeRegionEnd - retakeRegionStart) / FPS).toFixed(1) + 's)';
        }
    }

    function submitRetake() {
        if (!retakeClipId || !projectId) return;

        var prompt = retakePanelEl.querySelector('.ve-retake-prompt').value.trim();
        if (!prompt) {
            retakePanelEl.querySelector('.ve-retake-prompt').focus();
            return;
        }

        var strength = parseInt(retakePanelEl.querySelector('.ve-retake-strength').value) / 100;

        // Capture before exitRetakeMode nulls these
        var capturedClipId = retakeClipId;
        var capturedRegionStart = retakeRegionStart;
        var capturedRegionEnd = retakeRegionEnd;

        generatingClips.add(capturedClipId);
        exitRetakeMode();
        renderTimeline();

        fetch(getApiBase() + '/video_edit/retake', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project_id: projectId,
                clip_id: capturedClipId,
                region_start_frame: capturedRegionStart,
                region_end_frame: capturedRegionEnd,
                prompt: prompt,
                strength: strength,
            }),
        })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            if (data.status === 'error') {
                console.warn('Retake failed:', data.error);
                generatingClips.delete(capturedClipId);
                renderTimeline();
                return;
            }
            monitorGeneration(data.prompt_id, capturedClipId, data.output_path, 'retake');
        })
        .catch(function (err) {
            console.warn('Retake request failed:', err);
            generatingClips.delete(capturedClipId);
            renderTimeline();
        });
    }

    // ===== V4: Bridge Shot =====

    function buildBridgePanel() {
        bridgePanelEl = document.createElement('div');
        bridgePanelEl.className = 've-bridge-panel';
        bridgePanelEl.innerHTML =
            '<h4>Fill Gap with AI</h4>' +
            '<div class="ve-bridge-info"></div>' +
            '<textarea class="ve-retake-prompt" placeholder="Describe the transition (optional)..."></textarea>' +
            '<div class="ve-retake-actions">' +
                '<button class="ve-retake-btn ve-retake-btn-secondary ve-bridge-cancel">Cancel</button>' +
                '<button class="ve-retake-btn ve-retake-btn-primary ve-bridge-generate">Generate Bridge</button>' +
            '</div>';
        document.getElementById('panel-video-edit').appendChild(bridgePanelEl);
        bridgePanelEl.querySelector('.ve-bridge-cancel').addEventListener('click', function () {
            bridgePanelEl.style.display = 'none';
        });
    }

    var bridgeGapData = null;

    function showBridgePanel(trackId, frame) {
        var gap = findGapAtFrame(trackId, frame);
        if (!gap) return;

        bridgeGapData = { trackId: trackId, gap: gap };

        if (!bridgePanelEl) buildBridgePanel();

        var durationSec = ((gap.gapEnd - gap.gapStart) / FPS).toFixed(1);
        var infoEl = bridgePanelEl.querySelector('.ve-bridge-info');
        infoEl.textContent = 'Gap: ' + frameToTimecode(gap.gapStart) + ' - ' + frameToTimecode(gap.gapEnd) +
            ' (' + durationSec + 's)';

        bridgePanelEl.querySelector('.ve-retake-prompt').value = '';
        bridgePanelEl.style.display = 'block';
        bridgePanelEl.style.left = '50%';
        bridgePanelEl.style.top = '50%';
        bridgePanelEl.style.transform = 'translate(-50%, -50%)';

        // Re-bind generate to current gap
        var genBtn = bridgePanelEl.querySelector('.ve-bridge-generate');
        var newBtn = genBtn.cloneNode(true);
        genBtn.parentNode.replaceChild(newBtn, genBtn);
        newBtn.addEventListener('click', function () {
            submitBridgeShot();
        });
    }

    function submitBridgeShot() {
        if (!bridgeGapData || !projectId) return;

        var prompt = bridgePanelEl.querySelector('.ve-retake-prompt').value.trim() || 'smooth cinematic transition';
        var gap = bridgeGapData.gap;
        var trackId = bridgeGapData.trackId;

        bridgePanelEl.style.display = 'none';

        fetch(getApiBase() + '/video_edit/bridge_shot', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project_id: projectId,
                track_id: trackId,
                gap_start_frame: gap.gapStart,
                gap_end_frame: gap.gapEnd,
                before_clip_id: gap.before ? gap.before.id : null,
                after_clip_id: gap.after ? gap.after.id : null,
                prompt: prompt,
            }),
        })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            if (data.status === 'error') {
                console.warn('Bridge shot failed:', data.error);
                return;
            }
            // Add placeholder clip to local state
            var track = null;
            for (var i = 0; i < project.tracks.length; i++) {
                if (project.tracks[i].id === trackId) { track = project.tracks[i]; break; }
            }
            if (track) {
                pushUndo();
                track.clips.push({
                    id: data.clip_id,
                    startFrame: gap.gapStart,
                    endFrame: gap.gapEnd,
                    label: 'Bridge Shot',
                    color: '#ff8c42',
                    source_path: data.output_path,
                });
                generatingClips.add(data.clip_id);
                recalcTotalFrames();
                renderTimeline();
                monitorGeneration(data.prompt_id, data.clip_id, data.output_path, 'bridge');
            }
        })
        .catch(function (err) {
            console.warn('Bridge shot request failed:', err);
        });
    }

    // ===== V4: Multi-Take =====

    function buildTakeDropdown() {
        takeDropdownEl = document.createElement('div');
        takeDropdownEl.className = 've-take-dropdown';
        document.getElementById('panel-video-edit').appendChild(takeDropdownEl);
    }

    function showTakeSwitcher(clipId) {
        var info = findClipById(clipId);
        if (!info || !info.clip.takes || info.clip.takes.length < 2) return;

        if (!takeDropdownEl) buildTakeDropdown();

        var html = '';
        info.clip.takes.forEach(function (take) {
            var isActive = take.active;
            html += '<div class="ve-take-item' + (isActive ? ' ve-take-active' : '') +
                '" data-take-id="' + take.id + '" data-clip-id="' + clipId + '">' +
                '<span class="ve-take-radio"></span>' +
                '<span>' + (take.label || take.id) + '</span>' +
                '</div>';
        });
        takeDropdownEl.innerHTML = html;
        takeDropdownEl.style.display = 'block';
        takeDropdownEl.style.left = '50%';
        takeDropdownEl.style.top = '50%';
        takeDropdownEl.style.transform = 'translate(-50%, -50%)';

        takeDropdownEl.querySelectorAll('.ve-take-item').forEach(function (el) {
            el.addEventListener('click', function () {
                var takeId = el.dataset.takeId;
                var cid = el.dataset.clipId;
                switchTake(cid, takeId);
                takeDropdownEl.style.display = 'none';
            });
        });
    }

    function switchTake(clipId, takeId) {
        var info = findClipById(clipId);
        if (!info || !info.clip.takes) return;

        pushUndo();
        info.clip.takes.forEach(function (take) {
            take.active = (take.id === takeId);
            if (take.active) {
                info.clip.source_path = take.source_path;
            }
        });

        // Clear cached thumbnails for this clip so they reload
        thumbnailCache.delete(clipId);
        thumbnailLoading.delete(clipId);
        waveformCache.delete(clipId);
        waveformLoading.delete(clipId);

        renderTimeline();
        scheduleAutosave();
    }

    function addTakeToClip(clipId, sourcePath, label) {
        var info = findClipById(clipId);
        if (!info) return;

        pushUndo();
        var clip = info.clip;

        // Initialize takes array if needed
        if (!clip.takes || clip.takes.length === 0) {
            clip.takes = [{
                id: 'take-original',
                source_path: clip.source_path || '',
                label: 'Original',
                active: false,
            }];
        }

        // Add new take
        var newTake = {
            id: 'take-' + Date.now(),
            source_path: sourcePath,
            label: label || ('Take ' + (clip.takes.length)),
            active: true,
        };

        // Deactivate all others
        clip.takes.forEach(function (t) { t.active = false; });
        clip.takes.push(newTake);
        clip.source_path = sourcePath;

        // Clear caches
        thumbnailCache.delete(clipId);
        thumbnailLoading.delete(clipId);

        renderTimeline();
        scheduleAutosave();
    }

    // ===== V4: Generation Monitoring =====

    function monitorGeneration(promptId, clipId, outputPath, type) {
        if (!promptId) {
            generatingClips.delete(clipId);
            renderTimeline();
            return;
        }

        // Listen for completion via SerenityWS if available
        if (typeof SerenityWS !== 'undefined') {
            var handler = function (data) {
                if (!data || data.prompt_id !== promptId) return;
                generatingClips.delete(clipId);

                if (type === 'retake') {
                    addTakeToClip(clipId, outputPath, 'Retake');
                } else if (type === 'bridge') {
                    // Clip already added; just clear generating flag
                    var info = findClipById(clipId);
                    if (info && info.clip) {
                        delete info.clip.generating;
                    }
                }

                renderTimeline();
                SerenityWS.off('execution_success', handler);
                SerenityWS.off('execution_error', errorHandler);
            };
            var errorHandler = function (data) {
                if (!data || data.prompt_id !== promptId) return;
                generatingClips.delete(clipId);
                console.warn('Generation failed for', clipId);
                renderTimeline();
                SerenityWS.off('execution_success', handler);
                SerenityWS.off('execution_error', errorHandler);
            };
            SerenityWS.on('execution_success', handler);
            SerenityWS.on('execution_error', errorHandler);
        } else {
            // Fallback: poll every 5s
            var pollInterval = setInterval(function () {
                fetch(getApiBase() + '/history/' + promptId)
                    .then(function (r) { return r.ok ? r.json() : null; })
                    .then(function (data) {
                        if (data && data[promptId]) {
                            clearInterval(pollInterval);
                            generatingClips.delete(clipId);
                            if (type === 'retake') {
                                addTakeToClip(clipId, outputPath, 'Retake');
                            }
                            renderTimeline();
                        }
                    })
                    .catch(function () {});
            }, 5000);

            // Timeout after 5 minutes
            setTimeout(function () {
                clearInterval(pollInterval);
                generatingClips.delete(clipId);
                renderTimeline();
            }, 300000);
        }
    }

    // ===== V5: File Menu =====

    function toggleFileMenu() {
        var dd = document.getElementById('ve-file-dropdown');
        if (!dd) return;
        if (dd.style.display !== 'none') {
            dd.style.display = 'none';
            return;
        }
        dd.innerHTML =
            '<div class="ve-ctx-item" data-action="import-video">Import Video/Audio...</div>' +
            '<div class="ve-ctx-sep"></div>' +
            '<div class="ve-ctx-item" data-action="import-srt">Import SRT...</div>' +
            '<div class="ve-ctx-item" data-action="export-srt">Export SRT</div>' +
            '<div class="ve-ctx-sep"></div>' +
            '<div class="ve-ctx-item" data-action="import-xml">Import XML...</div>' +
            '<div class="ve-ctx-item" data-action="export-fcpxml">Export FCP XML</div>' +
            '<div class="ve-ctx-item" data-action="export-premiere">Export Premiere XML</div>';
        dd.style.display = 'block';

        dd.querySelectorAll('.ve-ctx-item').forEach(function (el) {
            el.addEventListener('click', function (e) {
                dd.style.display = 'none';
                handleFileAction(e.target.dataset.action);
            });
        });

        // Close on outside click (one-shot)
        setTimeout(function () {
            var closer = function (e) {
                if (!dd.contains(e.target)) {
                    dd.style.display = 'none';
                    document.removeEventListener('click', closer);
                }
            };
            document.addEventListener('click', closer);
        }, 0);
    }

    var videoFileInput = null;

    function handleFileAction(action) {
        if (action === 'import-video') {
            if (!videoFileInput) {
                videoFileInput = document.createElement('input');
                videoFileInput.type = 'file';
                videoFileInput.accept = 'video/*,audio/*';
                videoFileInput.multiple = true;
                videoFileInput.style.display = 'none';
                document.body.appendChild(videoFileInput);
                videoFileInput.addEventListener('change', function () {
                    for (var i = 0; i < videoFileInput.files.length; i++) {
                        importMediaFile(videoFileInput.files[i]);
                    }
                    videoFileInput.value = '';
                });
            }
            videoFileInput.click();
        } else if (action === 'import-srt') {
            if (!srtFileInput) {
                srtFileInput = document.createElement('input');
                srtFileInput.type = 'file';
                srtFileInput.accept = '.srt';
                srtFileInput.style.display = 'none';
                document.body.appendChild(srtFileInput);
                srtFileInput.addEventListener('change', function () {
                    if (srtFileInput.files[0]) importSRTFile(srtFileInput.files[0]);
                    srtFileInput.value = '';
                });
            }
            srtFileInput.click();
        } else if (action === 'export-srt') {
            exportSRT();
        } else if (action === 'import-xml') {
            if (!xmlFileInput) {
                xmlFileInput = document.createElement('input');
                xmlFileInput.type = 'file';
                xmlFileInput.accept = '.xml,.fcpxml';
                xmlFileInput.style.display = 'none';
                document.body.appendChild(xmlFileInput);
                xmlFileInput.addEventListener('change', function () {
                    if (xmlFileInput.files[0]) importXMLFile(xmlFileInput.files[0]);
                    xmlFileInput.value = '';
                });
            }
            xmlFileInput.click();
        } else if (action === 'export-fcpxml') {
            exportXML('fcpxml');
        } else if (action === 'export-premiere') {
            exportXML('premiere');
        }
    }

    // ===== V5: Media Import =====

    function importMediaFile(file) {
        if (!projectId) return;
        var form = new FormData();
        form.append('file', file);
        fetch(getApiBase() + '/video_edit/projects/' + projectId + '/import_clip', {
            method: 'POST',
            body: form,
        })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            if (data.error) {
                alert('Import failed: ' + data.error);
                return;
            }
            // Determine track type from file
            var isAudio = file.type && file.type.startsWith('audio');
            var trackType = isAudio ? 'audio' : 'video';
            var trackColor = isAudio ? '#2a9d5c' : '#4a7dff';

            // Find or create matching track
            var track = null;
            for (var i = 0; i < project.tracks.length; i++) {
                if (project.tracks[i].type === trackType) { track = project.tracks[i]; break; }
            }
            if (!track) {
                track = { id: 'track-' + Date.now(), name: isAudio ? 'Audio' : 'Video 1', type: trackType, clips: [] };
                project.tracks.push(track);
            }

            // Place at end of existing content on that track
            var endFrame = 0;
            track.clips.forEach(function (c) {
                if (c.endFrame > endFrame) endFrame = c.endFrame;
            });

            pushUndo();
            var dur = data.duration_frames || (FPS * 5);
            track.clips.push({
                id: data.clip_id || generateClipId(),
                startFrame: endFrame,
                endFrame: endFrame + dur,
                label: file.name.replace(/\.[^.]+$/, '').slice(0, 30),
                color: trackColor,
                source_path: data.source_path,
            });
            recalcTotalFrames();
            renderTimeline();
            scheduleAutosave();

            // Scroll to show new clip
            var clipEndPx = (endFrame + dur) * pixelsPerFrame;
            var viewWidth = stage ? stage.width() - TRACK_HEADER_WIDTH : 500;
            if (clipEndPx > scrollOffsetX + viewWidth) {
                scrollOffsetX = Math.max(0, clipEndPx - viewWidth + 100);
                renderTimeline();
            }
        })
        .catch(function (err) {
            alert('Import failed: ' + (err.message || err));
        });
    }

    // ===== V5: SRT Import/Export =====

    function importSRTFile(file) {
        if (!projectId) return;
        var form = new FormData();
        form.append('file', file);
        fetch(getApiBase() + '/video_edit/projects/' + projectId + '/import_srt', {
            method: 'POST',
            body: form,
        })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            if (data.error) {
                console.warn('SRT import failed:', data.error);
                return;
            }
            // Reload project to get the new track
            loadOrCreateProject();
        })
        .catch(function (err) {
            console.warn('SRT import error:', err);
        });
    }

    function exportSRT() {
        if (!projectId) { alert('No project. Open Video Edit tab first.'); return; }
        // Save then download
        saveProject();
        fetch(getApiBase() + '/video_edit/projects/' + projectId + '/export_srt')
            .then(function (r) {
                if (!r.ok) throw new Error('Export failed');
                return r.blob();
            })
            .then(function (blob) {
                var a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = 'subtitles_' + projectId + '.srt';
                a.click();
                URL.revokeObjectURL(a.href);
            })
            .catch(function (err) { alert('SRT export failed: ' + err.message); });
    }

    function parseSRTLocal(content) {
        var blocks = content.trim().split(/\n\n+/);
        var clips = [];
        blocks.forEach(function (block) {
            var lines = block.split('\n');
            if (lines.length < 3) return;
            var m = lines[1].match(
                /(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})/
            );
            if (!m) return;
            var startSec = parseInt(m[1]) * 3600 + parseInt(m[2]) * 60 + parseInt(m[3]) + parseInt(m[4]) / 1000;
            var endSec = parseInt(m[5]) * 3600 + parseInt(m[6]) * 60 + parseInt(m[7]) + parseInt(m[8]) / 1000;
            clips.push({
                id: generateClipId(),
                startFrame: Math.round(startSec * FPS),
                endFrame: Math.round(endSec * FPS),
                label: lines.slice(2).join('\n'),
                color: '#d4a72c',
            });
        });
        return clips;
    }

    function frameToSRTTime(frame) {
        var totalMs = Math.round(frame / FPS * 1000);
        var h = Math.floor(totalMs / 3600000);
        var m = Math.floor((totalMs % 3600000) / 60000);
        var s = Math.floor((totalMs % 60000) / 1000);
        var ms = totalMs % 1000;
        return String(h).padStart(2, '0') + ':' + String(m).padStart(2, '0') + ':' +
               String(s).padStart(2, '0') + ',' + String(ms).padStart(3, '0');
    }

    // ===== V5: XML Import/Export =====

    function importXMLFile(file) {
        if (!projectId) return;
        var form = new FormData();
        form.append('file', file);
        fetch(getApiBase() + '/video_edit/projects/' + projectId + '/import_xml', {
            method: 'POST',
            body: form,
        })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            if (data.error) {
                console.warn('XML import failed:', data.error);
                return;
            }
            if (data.warnings && data.warnings.length > 0) {
                console.warn('XML import warnings:', data.warnings);
            }
            loadOrCreateProject();
        })
        .catch(function (err) {
            console.warn('XML import error:', err);
        });
    }

    function exportXML(format) {
        if (!projectId) { alert('No project. Open Video Edit tab first.'); return; }
        saveProject();
        var ext = format === 'fcpxml' ? '.fcpxml' : '.xml';
        fetch(getApiBase() + '/video_edit/projects/' + projectId + '/export_xml?format=' + format)
            .then(function (r) {
                if (!r.ok) throw new Error('Export failed');
                return r.blob();
            })
            .then(function (blob) {
                var a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = 'timeline_' + projectId + ext;
                a.click();
                URL.revokeObjectURL(a.href);
            })
            .catch(function (err) { alert('XML export failed: ' + err.message); });
    }

    // ===== V5: Export Dialog =====

    function showExportDialog() {
        if (exportDialogEl) {
            exportDialogEl.remove();
        }

        var ext = '.mp4';
        exportDialogEl = document.createElement('div');
        exportDialogEl.className = 've-export-overlay';
        exportDialogEl.innerHTML =
            '<div class="ve-export-dialog">' +
                '<h3>Export Video</h3>' +
                '<div class="ve-export-row"><label>Format</label>' +
                    '<select id="ve-exp-format"><option value="h264">H.264 (.mp4)</option><option value="prores">ProRes (.mov)</option><option value="vp9">WebM (VP9)</option></select></div>' +
                '<div class="ve-export-row"><label>Resolution</label>' +
                    '<select id="ve-exp-res"><option value="project">' + (project.width || 1280) + 'x' + (project.height || 720) + '</option><option value="1920x1080">1920x1080</option><option value="1280x720">1280x720</option><option value="854x480">854x480</option></select></div>' +
                '<div class="ve-export-row"><label>FPS</label>' +
                    '<input type="number" id="ve-exp-fps" value="' + FPS + '" min="1" max="60" style="width:60px"></div>' +
                '<div class="ve-export-row"><label>Quality</label>' +
                    '<select id="ve-exp-quality"><option value="high">High</option><option value="medium">Medium</option><option value="low">Low</option><option value="lossless">Lossless</option></select></div>' +
                '<div class="ve-export-row"><label>Color Grade</label>' +
                    '<select id="ve-exp-lut"><option value="">None</option></select></div>' +
                '<div class="ve-export-row"><label>Audio</label>' +
                    '<input type="checkbox" id="ve-exp-audio" checked> <span style="font-size:12px;color:#aaa">Include audio</span></div>' +
                '<div class="ve-export-radio-group">' +
                    '<label><input type="radio" name="ve-exp-range" value="full" checked> Full timeline</label>' +
                    '<label><input type="radio" name="ve-exp-range" value="selection"> Selection only</label>' +
                '</div>' +
                '<div id="ve-exp-progress" class="ve-export-progress" style="display:none">' +
                    '<div class="ve-export-progress-bar"><div id="ve-exp-fill" class="ve-export-progress-fill"></div></div>' +
                    '<div id="ve-exp-progress-text" class="ve-export-progress-text">Preparing...</div>' +
                '</div>' +
                '<div class="ve-export-actions">' +
                    '<button id="ve-exp-cancel" class="ve-export-btn ve-export-btn-secondary">Cancel</button>' +
                    '<button id="ve-exp-start" class="ve-export-btn ve-export-btn-primary">Export</button>' +
                '</div>' +
            '</div>';

        document.body.appendChild(exportDialogEl);

        // Populate LUT dropdown in export dialog
        (function () {
            var sel = document.getElementById('ve-exp-lut');
            if (!sel) return;
            fetch(getApiBase() + '/video_edit/luts')
                .then(function (r) { return r.json(); })
                .then(function (luts) {
                    luts.forEach(function (lut) {
                        var opt = document.createElement('option');
                        opt.value = lut.path;
                        opt.textContent = lut.name;
                        sel.appendChild(opt);
                    });
                })
                .catch(function () {});
        })();

        document.getElementById('ve-exp-cancel').addEventListener('click', function () {
            if (isExporting && exportId) {
                fetch(getApiBase() + '/video_edit/export/' + exportId + '/cancel', { method: 'POST' }).catch(function () {});
            }
            closeExportDialog();
        });

        document.getElementById('ve-exp-start').addEventListener('click', startExport);

        // Listen for export events
        if (typeof SerenityWS !== 'undefined') {
            SerenityWS.on('export_progress', onExportProgress);
            SerenityWS.on('export_complete', onExportComplete);
            SerenityWS.on('export_error', onExportError);
        }
    }

    function closeExportDialog() {
        isExporting = false;
        exportId = null;
        if (exportDialogEl) {
            exportDialogEl.remove();
            exportDialogEl = null;
        }
        if (typeof SerenityWS !== 'undefined') {
            SerenityWS.off('export_progress', onExportProgress);
            SerenityWS.off('export_complete', onExportComplete);
            SerenityWS.off('export_error', onExportError);
        }
    }

    function startExport() {
        if (!projectId || isExporting) return;

        var fmt = document.getElementById('ve-exp-format').value;
        var resVal = document.getElementById('ve-exp-res').value;
        var w, h;
        if (resVal === 'project') {
            w = project.width || 1280;
            h = project.height || 720;
        } else {
            var parts = resVal.split('x');
            w = parseInt(parts[0]);
            h = parseInt(parts[1]);
        }
        var fps = parseInt(document.getElementById('ve-exp-fps').value) || 30;
        var quality = document.getElementById('ve-exp-quality').value;
        var includeAudio = document.getElementById('ve-exp-audio').checked;
        var lutSel = document.getElementById('ve-exp-lut');
        var lutPath = lutSel ? lutSel.value : '';
        var rangeRadio = document.querySelector('input[name="ve-exp-range"]:checked');
        var rangeFull = !rangeRadio || rangeRadio.value === 'full';

        var extMap = { h264: '.mp4', prores: '.mov', vp9: '.webm' };
        var filename = 'export_' + (projectId || 'video') + '_' + Date.now() + (extMap[fmt] || '.mp4');

        var body = {
            project_id: projectId,
            format: fmt,
            width: w,
            height: h,
            fps: fps,
            quality: quality,
            include_audio: includeAudio,
            output_filename: filename,
        };
        if (lutPath) body.lut_path = lutPath;

        if (!rangeFull && selectedClipIds.size > 0) {
            // Use selection range
            var minF = Infinity, maxF = 0;
            selectedClipIds.forEach(function (id) {
                var info = findClipById(id);
                if (info) {
                    if (info.clip.startFrame < minF) minF = info.clip.startFrame;
                    if (info.clip.endFrame > maxF) maxF = info.clip.endFrame;
                }
            });
            if (minF < maxF) {
                body.range_start_frame = minF;
                body.range_end_frame = maxF;
            }
        }

        isExporting = true;
        document.getElementById('ve-exp-start').disabled = true;
        document.getElementById('ve-exp-progress').style.display = '';

        fetch(getApiBase() + '/video_edit/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            if (data.error) {
                onExportError({ data: { error: data.error } });
                return;
            }
            exportId = data.export_id;
        })
        .catch(function (err) {
            onExportError({ data: { error: String(err) } });
        });
    }

    function onExportProgress(msg) {
        var d = msg.data || msg;
        if (exportId && d.export_id !== exportId) return;
        var fill = document.getElementById('ve-exp-fill');
        var text = document.getElementById('ve-exp-progress-text');
        if (fill) fill.style.width = (d.percent || 0) + '%';
        if (text) text.textContent = 'Encoding frame ' + (d.frame || 0) + ' of ' + (d.total_frames || '?') + ' (' + (d.percent || 0) + '%)';
    }

    function onExportComplete(msg) {
        var d = msg.data || msg;
        if (exportId && d.export_id !== exportId) return;
        var text = document.getElementById('ve-exp-progress-text');
        if (text) text.textContent = 'Export complete: ' + (d.output_path || '');
        var fill = document.getElementById('ve-exp-fill');
        if (fill) fill.style.width = '100%';
        isExporting = false;
        var startBtn = document.getElementById('ve-exp-start');
        if (startBtn) {
            startBtn.disabled = false;
            startBtn.textContent = 'Done';
            startBtn.onclick = closeExportDialog;
        }
    }

    function onExportError(msg) {
        var d = msg.data || msg;
        var text = document.getElementById('ve-exp-progress-text');
        if (text) text.textContent = 'Error: ' + (d.error || 'Unknown error');
        isExporting = false;
        var startBtn = document.getElementById('ve-exp-start');
        if (startBtn) startBtn.disabled = false;
    }

    // ===== V5: Subtitle Overlay in Preview =====

    function renderSubtitleOverlay() {
        var subEl = document.getElementById('ve-preview-subtitle');
        if (!subEl) return;

        // Find text clip at current frame
        var text = null;
        for (var i = 0; i < project.tracks.length; i++) {
            var track = project.tracks[i];
            if (track.type !== 'text') continue;
            for (var j = 0; j < track.clips.length; j++) {
                var clip = track.clips[j];
                if (currentFrame >= clip.startFrame && currentFrame < clip.endFrame) {
                    text = clip.label;
                    break;
                }
            }
            if (text) break;
        }

        if (text) {
            subEl.textContent = text;
            subEl.style.display = '';
        } else {
            subEl.style.display = 'none';
        }
    }

    // ===== V5: Inline Subtitle Editor =====

    function openSubtitleEditor(clipId) {
        var info = findClipById(clipId);
        if (!info || !info.track || info.track.type !== 'text') return;

        closeSubtitleEditor();

        subtitleEditingClipId = clipId;
        var clip = info.clip;

        // Position editor relative to the panel
        var panel = document.getElementById('panel-video-edit');
        var timelineContainer = document.getElementById('ve-timeline-container');
        if (!panel || !timelineContainer) return;

        var tcRect = timelineContainer.getBoundingClientRect();
        var panelRect = panel.getBoundingClientRect();
        var offsetTop = tcRect.top - panelRect.top;

        var clipX = TRACK_HEADER_WIDTH + (clip.startFrame * pixelsPerFrame) - scrollOffsetX;
        var clipY = offsetTop + RULER_HEIGHT + (info.trackIndex * TRACK_HEIGHT);

        subtitleEditorEl = document.createElement('div');
        subtitleEditorEl.className = 've-sub-editor';
        subtitleEditorEl.style.left = Math.max(clipX, TRACK_HEADER_WIDTH) + 'px';
        subtitleEditorEl.style.top = (clipY + TRACK_HEIGHT + 4) + 'px';

        var ta = document.createElement('textarea');
        ta.value = clip.label || '';
        subtitleEditorEl.appendChild(ta);

        // Append to panel (not timeline container) so it's above Konva canvas
        panel.appendChild(subtitleEditorEl);
        ta.focus();
        ta.select();

        ta.addEventListener('blur', function () {
            var newText = ta.value.trim();
            if (newText !== clip.label) {
                pushUndo();
                clip.label = newText || '(empty)';
                renderTimeline();
                scheduleAutosave();
            }
            closeSubtitleEditor();
        });

        ta.addEventListener('keydown', function (e) {
            if (e.key === 'Escape') {
                ta.blur();
            } else if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                ta.blur();
            }
        });
    }

    function closeSubtitleEditor() {
        subtitleEditingClipId = null;
        if (subtitleEditorEl) {
            subtitleEditorEl.remove();
            subtitleEditorEl = null;
        }
    }

    // ===== Konva Setup =====

    function initKonva() {
        var container = document.getElementById('ve-timeline-canvas');
        if (!container) return;
        var rect = container.getBoundingClientRect();
        var stageHeight = RULER_HEIGHT + (project.tracks.length * TRACK_HEIGHT) + 20;

        stage = new Konva.Stage({
            container: 've-timeline-canvas',
            width: rect.width,
            height: Math.max(stageHeight, rect.height)
        });

        rulerLayer = new Konva.Layer();
        timelineLayer = new Konva.Layer();
        overlayLayer = new Konva.Layer();

        stage.add(rulerLayer);
        stage.add(timelineLayer);
        stage.add(overlayLayer);
    }

    // ===== Rendering =====

    function renderRuler() {
        rulerLayer.destroyChildren();

        rulerLayer.add(new Konva.Rect({
            x: 0, y: 0,
            width: stage.width(), height: RULER_HEIGHT,
            fill: '#1a1a2e'
        }));

        rulerLayer.add(new Konva.Rect({
            x: 0, y: 0,
            width: TRACK_HEADER_WIDTH, height: RULER_HEIGHT,
            fill: '#1a1a2e'
        }));

        var visibleStartFrame = Math.floor(scrollOffsetX / pixelsPerFrame);
        var visibleEndFrame = Math.ceil((scrollOffsetX + stage.width() - TRACK_HEADER_WIDTH) / pixelsPerFrame);

        for (var f = Math.max(0, visibleStartFrame); f <= Math.min(totalFrames, visibleEndFrame); f++) {
            var x = TRACK_HEADER_WIDTH + (f * pixelsPerFrame) - scrollOffsetX;
            if (x < TRACK_HEADER_WIDTH || x > stage.width()) continue;

            if (f % FPS === 0) {
                rulerLayer.add(new Konva.Line({
                    points: [x, RULER_HEIGHT - 14, x, RULER_HEIGHT],
                    stroke: '#888', strokeWidth: 1
                }));
                rulerLayer.add(new Konva.Text({
                    x: x + 3, y: RULER_HEIGHT - 24,
                    text: frameToTimecode(f),
                    fontSize: 10, fill: '#aaa', fontFamily: 'monospace'
                }));
            } else if (f % 5 === 0) {
                rulerLayer.add(new Konva.Line({
                    points: [x, RULER_HEIGHT - 6, x, RULER_HEIGHT],
                    stroke: '#555', strokeWidth: 1
                }));
            }
        }

        rulerLayer.batchDraw();
    }

    // ===== V6: Effects & Properties Panel =====

    function openPropertiesPanel(clipId) {
        var info = findClipById(clipId);
        if (!info) return;

        // Don't open for text/subtitle or audio-only clips
        if (info.track.type === 'text' || info.track.type === 'audio') return;

        propsPanelClipId = clipId;
        var clip = info.clip;

        // Initialize effects array if missing
        if (!clip.effects) clip.effects = [];
        if (!clip.transition_in) clip.transition_in = null;

        // Remove existing panel
        closePropertiesPanel();

        var panel = document.createElement('div');
        panel.className = 've-props-panel';
        propsPanelEl = panel;

        renderPropertiesPanelContent(panel, clip);

        var container = document.getElementById('panel-video-edit');
        container.appendChild(panel);
    }

    function refreshPropertiesPanel() {
        if (!propsPanelEl || !propsPanelClipId) return;
        var info = findClipById(propsPanelClipId);
        if (!info) {
            // Clip was removed by undo
            closePropertiesPanel();
            return;
        }
        renderPropertiesPanelContent(propsPanelEl, info.clip);
    }

    function closePropertiesPanel() {
        if (propsPanelEl && propsPanelEl.parentNode) {
            propsPanelEl.parentNode.removeChild(propsPanelEl);
        }
        propsPanelEl = null;
        propsPanelClipId = null;
        if (addEffectDropdownEl && addEffectDropdownEl.parentNode) {
            addEffectDropdownEl.parentNode.removeChild(addEffectDropdownEl);
        }
        addEffectDropdownEl = null;
    }

    function renderPropertiesPanelContent(panel, clip) {
        panel.innerHTML = '';

        // Header
        var header = document.createElement('h3');
        header.textContent = 'Clip Properties';
        panel.appendChild(header);

        var closeBtn = document.createElement('span');
        closeBtn.className = 've-props-close';
        closeBtn.textContent = '\u00D7';
        closeBtn.onclick = function () { closePropertiesPanel(); };
        panel.appendChild(closeBtn);

        // Label input
        var labelRow = document.createElement('div');
        labelRow.className = 've-props-row';
        var labelLbl = document.createElement('label');
        labelLbl.textContent = 'Label';
        var labelInput = document.createElement('input');
        labelInput.type = 'text';
        labelInput.value = clip.label || '';
        labelInput.onchange = function () {
            pushUndo();
            clip.label = labelInput.value;
            renderTimeline();
            scheduleAutosave();
        };
        labelRow.appendChild(labelLbl);
        labelRow.appendChild(labelInput);
        panel.appendChild(labelRow);

        // Effects section
        var effectsLabel = document.createElement('div');
        effectsLabel.className = 've-section-label';
        effectsLabel.textContent = 'Effects';
        panel.appendChild(effectsLabel);

        // Effect cards
        (clip.effects || []).forEach(function (eff, idx) {
            var reg = EFFECT_REGISTRY[eff.type];
            if (!reg) return;

            var card = document.createElement('div');
            card.className = 've-effect-card';

            var cardHeader = document.createElement('div');
            cardHeader.className = 've-effect-card-header';

            var lbl = document.createElement('label');
            var cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.checked = eff.enabled !== false;
            cb.onchange = function () {
                pushUndo();
                eff.enabled = cb.checked;
                scheduleAutosave();
            };
            lbl.appendChild(cb);
            lbl.appendChild(document.createTextNode(' ' + reg.name));

            var delBtn = document.createElement('span');
            delBtn.className = 've-effect-delete';
            delBtn.textContent = '\u2715';
            delBtn.onclick = function () {
                pushUndo();
                clip.effects.splice(idx, 1);
                renderPropertiesPanelContent(panel, clip);
                renderTimeline();
                scheduleAutosave();
            };

            cardHeader.appendChild(lbl);
            cardHeader.appendChild(delBtn);
            card.appendChild(cardHeader);

            // Param sliders
            var paramKeys = Object.keys(reg.range || {});
            paramKeys.forEach(function (key) {
                var range = reg.range[key]; // [min, max, step]
                var row = document.createElement('div');
                row.className = 've-effect-slider-row';

                var paramLabel = document.createElement('span');
                paramLabel.className = 've-param-label';
                paramLabel.textContent = key;

                var slider = document.createElement('input');
                slider.type = 'range';
                slider.min = range[0];
                slider.max = range[1];
                slider.step = range[2];
                slider.value = eff.params[key] !== undefined ? eff.params[key] : reg.defaults[key];

                var valDisplay = document.createElement('span');
                valDisplay.className = 've-val';
                valDisplay.textContent = parseFloat(slider.value).toFixed(range[2] < 1 ? 2 : 0);

                slider.oninput = function () {
                    var v = parseFloat(slider.value);
                    eff.params[key] = v;
                    valDisplay.textContent = v.toFixed(range[2] < 1 ? 2 : 0);
                };
                slider.onchange = function () {
                    pushUndo();
                    scheduleAutosave();
                };

                row.appendChild(paramLabel);
                row.appendChild(slider);
                row.appendChild(valDisplay);
                card.appendChild(row);
            });

            panel.appendChild(card);
        });

        // Add Effect button
        var addBtn = document.createElement('div');
        addBtn.className = 've-add-effect-btn';
        addBtn.textContent = '+ Add Effect';
        addBtn.onclick = function (e) {
            e.stopPropagation();
            showAddEffectDropdown(panel, clip, addBtn);
        };
        panel.appendChild(addBtn);

        // LUT / Color Grade section
        var lutSection = document.createElement('div');
        lutSection.className = 've-lut-section';
        var lutLabel = document.createElement('div');
        lutLabel.className = 've-section-label';
        lutLabel.textContent = 'Color Grade (LUT)';
        lutSection.appendChild(lutLabel);

        var lutSelect = document.createElement('select');
        lutSelect.className = 've-lut-select';
        var noneOpt = document.createElement('option');
        noneOpt.value = '';
        noneOpt.textContent = 'None';
        lutSelect.appendChild(noneOpt);
        lutSection.appendChild(lutSelect);

        // Populate LUT dropdown
        (function (sel, c) {
            fetch(getApiBase() + '/video_edit/luts')
                .then(function (r) { return r.json(); })
                .then(function (luts) {
                    luts.forEach(function (lut) {
                        var opt = document.createElement('option');
                        opt.value = lut.path;
                        opt.textContent = lut.name;
                        if (c.lut_path === lut.path) opt.selected = true;
                        sel.appendChild(opt);
                    });
                })
                .catch(function () {});
        })(lutSelect, clip);

        lutSelect.onchange = function () {
            pushUndo();
            if (lutSelect.value) {
                clip.lut_path = lutSelect.value;
                clip.lut_name = lutSelect.options[lutSelect.selectedIndex].textContent;
            } else {
                clip.lut_path = null;
                clip.lut_name = null;
            }
            renderTimeline();
            scheduleAutosave();
        };

        // Upload new .cube file
        var uploadBtn = document.createElement('button');
        uploadBtn.className = 've-lut-upload-btn';
        uploadBtn.textContent = 'Upload New .cube\u2026';
        uploadBtn.onclick = function () {
            var fi = document.createElement('input');
            fi.type = 'file';
            fi.accept = '.cube';
            fi.onchange = function () {
                if (!fi.files || !fi.files[0]) return;
                var fd = new FormData();
                fd.append('file', fi.files[0]);
                fetch(getApiBase() + '/video_edit/luts/upload', { method: 'POST', body: fd })
                    .then(function (r) { return r.json(); })
                    .then(function (data) {
                        if (data.error) { alert(data.error); return; }
                        // Select the new LUT
                        clip.lut_path = data.path;
                        clip.lut_name = data.name;
                        renderPropertiesPanelContent(panel, clip);
                        renderTimeline();
                        scheduleAutosave();
                    })
                    .catch(function (e) { alert('Upload failed: ' + e); });
            };
            fi.click();
        };
        lutSection.appendChild(uploadBtn);

        // Preview row (original vs graded)
        if (clip.lut_path && clip.source_path) {
            var previewRow = document.createElement('div');
            previewRow.className = 've-lut-preview-row';

            var origCol = document.createElement('div');
            var origImg = document.createElement('img');
            origImg.alt = 'Original';
            var origLabel = document.createElement('div');
            origLabel.className = 've-lut-preview-label';
            origLabel.textContent = 'Original';
            origCol.appendChild(origImg);
            origCol.appendChild(origLabel);

            var gradedCol = document.createElement('div');
            var gradedImg = document.createElement('img');
            gradedImg.alt = 'Graded';
            var gradedLabel = document.createElement('div');
            gradedLabel.className = 've-lut-preview-label';
            gradedLabel.textContent = 'Graded';
            gradedCol.appendChild(gradedImg);
            gradedCol.appendChild(gradedLabel);

            previewRow.appendChild(origCol);
            previewRow.appendChild(gradedCol);
            lutSection.appendChild(previewRow);

            // Fetch original frame
            var seekSec = (clip.source_start || 0) / FPS;
            fetch(getApiBase() + '/video_edit/preview_effect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ source_path: clip.source_path, effects: [], seek_sec: seekSec }),
            })
            .then(function (r) { return r.blob(); })
            .then(function (blob) { origImg.src = URL.createObjectURL(blob); })
            .catch(function () {});

            // Fetch graded frame
            fetch(getApiBase() + '/video_edit/luts/preview', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ source_path: clip.source_path, lut_path: clip.lut_path, seek_sec: seekSec }),
            })
            .then(function (r) { return r.blob(); })
            .then(function (blob) { gradedImg.src = URL.createObjectURL(blob); })
            .catch(function () {});
        }

        panel.appendChild(lutSection);

        // Transition section
        var transLabel = document.createElement('div');
        transLabel.className = 've-section-label';
        transLabel.textContent = 'Transition In';
        panel.appendChild(transLabel);

        var transRow = document.createElement('div');
        transRow.className = 've-props-row';
        var transLbl = document.createElement('label');
        transLbl.textContent = 'Type';
        var transSelect = document.createElement('select');
        var currentTrans = (clip.transition_in && clip.transition_in.type) || 'none';
        TRANSITION_TYPES.forEach(function (tt) {
            var opt = document.createElement('option');
            opt.value = tt.type;
            opt.textContent = tt.name;
            if (tt.type === currentTrans) opt.selected = true;
            transSelect.appendChild(opt);
        });
        transSelect.onchange = function () {
            pushUndo();
            if (transSelect.value === 'none') {
                clip.transition_in = null;
            } else {
                clip.transition_in = {
                    type: transSelect.value,
                    duration: (clip.transition_in && clip.transition_in.duration) || 15,
                };
            }
            renderPropertiesPanelContent(panel, clip);
            renderTimeline();
            scheduleAutosave();
        };
        transRow.appendChild(transLbl);
        transRow.appendChild(transSelect);
        panel.appendChild(transRow);

        // Transition duration slider (only if transition active)
        if (clip.transition_in && clip.transition_in.type !== 'none') {
            var durRow = document.createElement('div');
            durRow.className = 've-effect-slider-row';
            var durLabel = document.createElement('span');
            durLabel.className = 've-param-label';
            durLabel.textContent = 'Duration';
            var durSlider = document.createElement('input');
            durSlider.type = 'range';
            durSlider.min = 5;
            durSlider.max = 60;
            durSlider.step = 1;
            durSlider.value = clip.transition_in.duration || 15;
            var durVal = document.createElement('span');
            durVal.className = 've-val';
            durVal.textContent = durSlider.value + 'f';
            durSlider.oninput = function () {
                clip.transition_in.duration = parseInt(durSlider.value);
                durVal.textContent = durSlider.value + 'f';
            };
            durSlider.onchange = function () {
                pushUndo();
                scheduleAutosave();
            };
            durRow.appendChild(durLabel);
            durRow.appendChild(durSlider);
            durRow.appendChild(durVal);
            panel.appendChild(durRow);

            var transNote = document.createElement('div');
            transNote.style.cssText = 'font-size:10px;color:#666;margin-top:4px;';
            transNote.textContent = 'Transitions applied on export (V2)';
            panel.appendChild(transNote);
        }
    }

    function showAddEffectDropdown(panel, clip, anchorEl) {
        // Remove existing dropdown
        if (addEffectDropdownEl && addEffectDropdownEl.parentNode) {
            addEffectDropdownEl.parentNode.removeChild(addEffectDropdownEl);
            addEffectDropdownEl = null;
            return; // toggle off
        }

        var dropdown = document.createElement('div');
        dropdown.className = 've-add-effect-dropdown';
        addEffectDropdownEl = dropdown;

        var categories = {};
        var keys = Object.keys(EFFECT_REGISTRY);
        keys.forEach(function (key) {
            var reg = EFFECT_REGISTRY[key];
            if (!categories[reg.category]) categories[reg.category] = [];
            categories[reg.category].push({ key: key, reg: reg });
        });

        Object.keys(categories).forEach(function (cat) {
            var catLabel = document.createElement('div');
            catLabel.className = 've-add-effect-category';
            catLabel.textContent = cat;
            dropdown.appendChild(catLabel);

            categories[cat].forEach(function (item) {
                var itemEl = document.createElement('div');
                itemEl.className = 've-add-effect-item';
                itemEl.textContent = item.reg.name;
                itemEl.onclick = function (e) {
                    e.stopPropagation();
                    pushUndo();
                    var newEff = {
                        id: 'eff_' + Math.random().toString(16).substr(2, 8),
                        type: item.key,
                        enabled: true,
                        params: JSON.parse(JSON.stringify(item.reg.defaults)),
                    };
                    clip.effects.push(newEff);
                    // Remove dropdown
                    if (addEffectDropdownEl && addEffectDropdownEl.parentNode) {
                        addEffectDropdownEl.parentNode.removeChild(addEffectDropdownEl);
                    }
                    addEffectDropdownEl = null;
                    renderPropertiesPanelContent(panel, clip);
                    renderTimeline();
                    scheduleAutosave();
                };
                dropdown.appendChild(itemEl);
            });
        });

        // Position below the add button
        var rect = anchorEl.getBoundingClientRect();
        var panelRect = panel.getBoundingClientRect();
        dropdown.style.left = (rect.left - panelRect.left) + 'px';
        dropdown.style.top = (rect.bottom - panelRect.top + 2) + 'px';
        panel.appendChild(dropdown);

        // Close on outside click
        var closeHandler = function (e) {
            if (!dropdown.contains(e.target) && e.target !== anchorEl) {
                if (dropdown.parentNode) dropdown.parentNode.removeChild(dropdown);
                addEffectDropdownEl = null;
                document.removeEventListener('mousedown', closeHandler);
            }
        };
        setTimeout(function () {
            document.addEventListener('mousedown', closeHandler);
        }, 0);
    }

    // ===== V7: RIFE Frame Interpolation =====

    function startRifeInterpolation(clipId, multiplier) {
        fetch(getApiBase() + '/video_edit/rife/status')
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (!data.available) {
                    alert('RIFE model not found.\n\nPlace flownet.pkl from Practical-RIFE v4.25 into:\n' + data.model_path);
                    return;
                }
                if (data.processing) {
                    alert('RIFE is already processing another clip. Please wait.');
                    return;
                }
                if (!confirm('Interpolate this clip ' + multiplier + '\u00D7 using RIFE?\nThis may take a few minutes depending on clip length.')) {
                    return;
                }
                fetch(getApiBase() + '/video_edit/rife/interpolate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        project_id: projectId,
                        clip_id: clipId,
                        multiplier: multiplier,
                        replace_source: true,
                    }),
                })
                .then(function (r) { return r.json(); })
                .then(function (res) {
                    if (res.error) {
                        alert('RIFE error: ' + res.error);
                        return;
                    }
                    activeRifeJobId = res.job_id;
                    rifeProcessingClipId = clipId;
                    rifeProgressPercent = 0;
                    renderTimeline();

                    // Subscribe to WS events
                    if (typeof SerenityWS !== 'undefined') {
                        SerenityWS.on('rife_progress', onRifeProgress);
                        SerenityWS.on('rife_complete', onRifeComplete);
                        SerenityWS.on('rife_error', onRifeError);
                    }
                })
                .catch(function (err) { alert('RIFE request failed: ' + err); });
            })
            .catch(function (err) { alert('Could not check RIFE status: ' + err); });
    }

    function onRifeProgress(msg) {
        var d = msg.data || msg;
        if (activeRifeJobId && d.job_id !== activeRifeJobId) return;
        rifeProgressPercent = d.percent || 0;
        renderTimeline();
    }

    function onRifeComplete(msg) {
        var d = msg.data || msg;
        if (activeRifeJobId && d.job_id !== activeRifeJobId) return;

        // Update clip source in memory if replace_source was true
        if (d.replace_source !== false && d.clip_id && d.output_path) {
            var info = findClipById(d.clip_id);
            if (info) {
                info.clip.source_path = d.output_path;
                info.clip.rife_multiplier = d.multiplier || 2;
            }
        }

        cleanupRifeState();
        renderTimeline();
        scheduleAutosave();
    }

    function onRifeError(msg) {
        var d = msg.data || msg;
        if (activeRifeJobId && d.job_id !== activeRifeJobId) return;
        cleanupRifeState();
        renderTimeline();
        alert('RIFE interpolation failed: ' + (d.error || 'Unknown error'));
    }

    function cleanupRifeState() {
        activeRifeJobId = null;
        rifeProcessingClipId = null;
        rifeProgressPercent = 0;
        if (typeof SerenityWS !== 'undefined') {
            SerenityWS.off('rife_progress', onRifeProgress);
            SerenityWS.off('rife_complete', onRifeComplete);
            SerenityWS.off('rife_error', onRifeError);
        }
    }

    // ===== V8: Face Restoration (CodeFormer) =====

    function showFaceRestoreDialog(clipId) {
        fetch(getApiBase() + '/video_edit/facetools/status')
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (!data.available) {
                    var missing = [];
                    if (data.models) {
                        for (var k in data.models) {
                            if (!data.models[k]) missing.push(k);
                        }
                    }
                    alert('Face restoration models not found.\n\nMissing: ' + missing.join(', ') +
                        '\n\nPlace model files in:\n' + data.model_dir);
                    return;
                }
                if (data.processing) {
                    alert('Face restoration is already processing another clip. Please wait.');
                    return;
                }
                _buildFaceDialog(clipId);
            })
            .catch(function (err) { alert('Could not check face restore status: ' + err); });
    }

    function _buildFaceDialog(clipId) {
        // Remove existing dialog
        if (faceDialogEl) {
            faceDialogEl.remove();
            faceDialogEl = null;
        }

        var clipInfo = findClipById(clipId);
        var sourcePath = clipInfo ? clipInfo.clip.source_path : '';

        var overlay = document.createElement('div');
        overlay.className = 've-face-overlay';
        overlay.innerHTML =
            '<div class="ve-face-dialog">' +
                '<div class="ve-face-header">' +
                    '<span>Restore Faces (CodeFormer)</span>' +
                    '<span class="ve-face-close">\u00D7</span>' +
                '</div>' +
                '<div class="ve-face-body">' +
                    '<div class="ve-face-slider-row">' +
                        '<label>Fidelity</label>' +
                        '<input type="range" min="0" max="1" step="0.05" value="0.7" class="ve-face-fidelity">' +
                        '<span class="ve-face-val">0.70</span>' +
                    '</div>' +
                    '<div class="ve-face-hint">' +
                        '\u2190 Higher quality &nbsp;&nbsp;&nbsp;&nbsp; Higher fidelity \u2192' +
                    '</div>' +
                    '<div class="ve-face-preview-area" style="display:none;">' +
                        '<img class="ve-face-preview-img" alt="Preview">' +
                    '</div>' +
                    '<div class="ve-face-license">' +
                        '\u2139 Non-commercial license (S-Lab 1.0)' +
                    '</div>' +
                '</div>' +
                '<div class="ve-face-actions">' +
                    '<button class="ve-face-btn ve-face-btn-preview">Preview</button>' +
                    '<div style="flex:1"></div>' +
                    '<button class="ve-face-btn ve-face-btn-secondary ve-face-btn-cancel">Cancel</button>' +
                    '<button class="ve-face-btn ve-face-btn-primary ve-face-btn-start">Start</button>' +
                '</div>' +
            '</div>';

        document.getElementById('panel-video-edit').appendChild(overlay);
        faceDialogEl = overlay;

        var slider = overlay.querySelector('.ve-face-fidelity');
        var valLabel = overlay.querySelector('.ve-face-val');
        var previewArea = overlay.querySelector('.ve-face-preview-area');
        var previewImg = overlay.querySelector('.ve-face-preview-img');

        slider.addEventListener('input', function () {
            valLabel.textContent = parseFloat(slider.value).toFixed(2);
        });

        overlay.querySelector('.ve-face-close').addEventListener('click', function () {
            overlay.remove();
            faceDialogEl = null;
        });

        overlay.querySelector('.ve-face-btn-cancel').addEventListener('click', function () {
            overlay.remove();
            faceDialogEl = null;
        });

        overlay.querySelector('.ve-face-btn-preview').addEventListener('click', function () {
            if (!sourcePath) { alert('No source path for this clip'); return; }
            var btn = overlay.querySelector('.ve-face-btn-preview');
            btn.textContent = 'Loading...';
            btn.disabled = true;
            fetch(getApiBase() + '/video_edit/facetools/preview', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    project_id: projectId,
                    clip_id: clipId,
                    seek_sec: 2.0,
                    fidelity: parseFloat(slider.value),
                }),
            })
            .then(function (r) {
                if (!r.ok) throw new Error('Preview failed');
                return r.blob();
            })
            .then(function (blob) {
                var url = URL.createObjectURL(blob);
                previewImg.src = url;
                previewArea.style.display = 'block';
                btn.textContent = 'Preview';
                btn.disabled = false;
            })
            .catch(function (err) {
                alert('Preview failed: ' + err);
                btn.textContent = 'Preview';
                btn.disabled = false;
            });
        });

        overlay.querySelector('.ve-face-btn-start').addEventListener('click', function () {
            var fidelity = parseFloat(slider.value);
            overlay.remove();
            faceDialogEl = null;
            startFaceRestore(clipId, fidelity);
        });
    }

    function startFaceRestore(clipId, fidelity) {
        fetch(getApiBase() + '/video_edit/facetools/restore', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project_id: projectId,
                clip_id: clipId,
                fidelity: fidelity,
            }),
        })
        .then(function (r) { return r.json(); })
        .then(function (res) {
            if (res.error) {
                alert('Face restore error: ' + res.error);
                return;
            }
            activeFaceJobId = res.job_id;
            faceProcessingClipId = clipId;
            faceProgressPercent = 0;
            renderTimeline();

            if (typeof SerenityWS !== 'undefined') {
                SerenityWS.on('face_restore_progress', onFaceProgress);
                SerenityWS.on('face_restore_complete', onFaceComplete);
                SerenityWS.on('face_restore_error', onFaceError);
            }
        })
        .catch(function (err) { alert('Face restore request failed: ' + err); });
    }

    function onFaceProgress(msg) {
        var d = msg.data || msg;
        if (activeFaceJobId && d.job_id !== activeFaceJobId) return;
        faceProgressPercent = d.percent || 0;
        renderTimeline();
    }

    function onFaceComplete(msg) {
        var d = msg.data || msg;
        if (activeFaceJobId && d.job_id !== activeFaceJobId) return;

        if (d.clip_id && d.output_path) {
            var info = findClipById(d.clip_id);
            if (info) {
                info.clip.source_path = d.output_path;
                info.clip.face_restored = true;
            }
        }

        cleanupFaceState();
        renderTimeline();
        scheduleAutosave();
    }

    function onFaceError(msg) {
        var d = msg.data || msg;
        if (activeFaceJobId && d.job_id !== activeFaceJobId) return;
        cleanupFaceState();
        renderTimeline();
        alert('Face restoration failed: ' + (d.error || 'Unknown error'));
    }

    function cleanupFaceState() {
        activeFaceJobId = null;
        faceProcessingClipId = null;
        faceProgressPercent = 0;
        if (typeof SerenityWS !== 'undefined') {
            SerenityWS.off('face_restore_progress', onFaceProgress);
            SerenityWS.off('face_restore_complete', onFaceComplete);
            SerenityWS.off('face_restore_error', onFaceError);
        }
    }

    // ===== V9: Real-ESRGAN Upscale =====

    function showEsrganDialog(clipId) {
        fetch(getApiBase() + '/video_edit/esrgan/status')
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (!data.available) {
                    alert('Real-ESRGAN model not found.\nPlace RealESRGAN_x4plus.pth in:\n' + (data.model_path || 'models/esrgan/'));
                    return;
                }
                if (data.processing) {
                    alert('ESRGAN is already processing another clip.');
                    return;
                }
                _buildEsrganDialog(clipId);
            })
            .catch(function (err) { alert('Could not check ESRGAN status: ' + err); });
    }

    function _buildEsrganDialog(clipId) {
        if (esrganDialogEl) esrganDialogEl.remove();

        var info = findClipById(clipId);
        var srcW = info && info.clip.source_width ? info.clip.source_width : '?';
        var srcH = info && info.clip.source_height ? info.clip.source_height : '?';

        var overlay = document.createElement('div');
        overlay.className = 've-esrgan-overlay';
        overlay.innerHTML =
            '<div class="ve-esrgan-dialog">' +
                '<div class="ve-esrgan-header">' +
                    '<span>Upscale with Real-ESRGAN</span>' +
                    '<span class="ve-esrgan-close">\u00D7</span>' +
                '</div>' +
                '<div class="ve-esrgan-body">' +
                    '<div class="ve-esrgan-radio-row">' +
                        '<label>Scale:</label>' +
                        '<label><input type="radio" name="esrgan-scale" value="2" checked /> 2\u00D7</label>' +
                        '<label><input type="radio" name="esrgan-scale" value="4" /> 4\u00D7</label>' +
                    '</div>' +
                    '<div class="ve-esrgan-output-info">Output: ' + srcW + '\u00D7' + srcH + ' \u2192 <span class="ve-esrgan-out-dims">' + (srcW !== '?' ? srcW * 2 : '?') + '\u00D7' + (srcH !== '?' ? srcH * 2 : '?') + '</span></div>' +
                    '<div class="ve-esrgan-quality-row">' +
                        '<label>Quality:</label>' +
                        '<select class="ve-esrgan-quality">' +
                            '<option value="0">Fast (whole frame)</option>' +
                            '<option value="512" selected>Balanced (512 tile)</option>' +
                            '<option value="256">Safe (256 tile)</option>' +
                        '</select>' +
                    '</div>' +
                    '<div class="ve-esrgan-warning">\u26A0 Output file will be larger. 4\u00D7 upscale is slow (~2-5 sec/frame).</div>' +
                    '<div class="ve-esrgan-preview-area"><img class="ve-esrgan-preview-img" style="display:none" /></div>' +
                    '<div class="ve-esrgan-actions">' +
                        '<button class="ve-esrgan-btn ve-esrgan-btn-preview">Preview</button>' +
                        '<button class="ve-esrgan-btn ve-esrgan-btn-secondary ve-esrgan-btn-cancel">Cancel</button>' +
                        '<button class="ve-esrgan-btn ve-esrgan-btn-primary ve-esrgan-btn-start">Start</button>' +
                    '</div>' +
                '</div>' +
            '</div>';

        document.body.appendChild(overlay);
        esrganDialogEl = overlay;

        // Update output dimensions when scale changes
        var radios = overlay.querySelectorAll('input[name="esrgan-scale"]');
        var outDims = overlay.querySelector('.ve-esrgan-out-dims');
        radios.forEach(function (r) {
            r.addEventListener('change', function () {
                var s = parseInt(r.value);
                if (srcW !== '?' && srcH !== '?') {
                    outDims.textContent = (srcW * s) + '\u00D7' + (srcH * s);
                }
            });
        });

        overlay.querySelector('.ve-esrgan-close').addEventListener('click', function () {
            overlay.remove();
            esrganDialogEl = null;
        });

        overlay.querySelector('.ve-esrgan-btn-cancel').addEventListener('click', function () {
            overlay.remove();
            esrganDialogEl = null;
        });

        overlay.querySelector('.ve-esrgan-btn-preview').addEventListener('click', function () {
            var scale = parseInt(overlay.querySelector('input[name="esrgan-scale"]:checked').value);
            var tile = parseInt(overlay.querySelector('.ve-esrgan-quality').value);
            var btn = overlay.querySelector('.ve-esrgan-btn-preview');
            btn.textContent = 'Loading...';
            btn.disabled = true;
            fetch(getApiBase() + '/video_edit/esrgan/preview', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    project_id: projectId,
                    clip_id: clipId,
                    seek_sec: 0,
                    scale: scale,
                    tile_size: tile,
                }),
            })
            .then(function (r) {
                if (!r.ok) throw new Error('Preview failed');
                return r.blob();
            })
            .then(function (blob) {
                var img = overlay.querySelector('.ve-esrgan-preview-img');
                if (img.src && img.src.startsWith('blob:')) URL.revokeObjectURL(img.src);
                img.src = URL.createObjectURL(blob);
                img.style.display = 'block';
                btn.textContent = 'Preview';
                btn.disabled = false;
            })
            .catch(function (err) {
                btn.textContent = 'Preview';
                btn.disabled = false;
                alert('Preview failed: ' + err);
            });
        });

        overlay.querySelector('.ve-esrgan-btn-start').addEventListener('click', function () {
            var scale = parseInt(overlay.querySelector('input[name="esrgan-scale"]:checked').value);
            var tile = parseInt(overlay.querySelector('.ve-esrgan-quality').value);
            overlay.remove();
            esrganDialogEl = null;
            startEsrganUpscale(clipId, scale, tile);
        });
    }

    function startEsrganUpscale(clipId, scale, tileSize) {
        fetch(getApiBase() + '/video_edit/esrgan/upscale', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project_id: projectId,
                clip_id: clipId,
                scale: scale,
                tile_size: tileSize,
            }),
        })
        .then(function (r) { return r.json(); })
        .then(function (res) {
            if (res.error) {
                alert('ESRGAN error: ' + res.error);
                return;
            }
            activeEsrganJobId = res.job_id;
            esrganProcessingClipId = clipId;
            esrganProgressPercent = 0;
            renderTimeline();

            if (typeof SerenityWS !== 'undefined') {
                SerenityWS.on('esrgan_progress', onEsrganProgress);
                SerenityWS.on('esrgan_complete', onEsrganComplete);
                SerenityWS.on('esrgan_error', onEsrganError);
            }
        })
        .catch(function (err) { alert('ESRGAN request failed: ' + err); });
    }

    function onEsrganProgress(msg) {
        var d = msg.data || msg;
        if (activeEsrganJobId && d.job_id !== activeEsrganJobId) return;
        esrganProgressPercent = d.percent || 0;
        renderTimeline();
    }

    function onEsrganComplete(msg) {
        var d = msg.data || msg;
        if (activeEsrganJobId && d.job_id !== activeEsrganJobId) return;

        if (d.clip_id && d.output_path) {
            var info = findClipById(d.clip_id);
            if (info) {
                info.clip.source_path = d.output_path;
                info.clip.esrgan_scale = d.scale;
                info.clip.esrgan_output = d.output_path;
            }
        }

        cleanupEsrganState();
        renderTimeline();
        scheduleAutosave();
    }

    function onEsrganError(msg) {
        var d = msg.data || msg;
        if (activeEsrganJobId && d.job_id !== activeEsrganJobId) return;
        cleanupEsrganState();
        renderTimeline();
        alert('Upscale failed: ' + (d.error || 'Unknown error'));
    }

    function cleanupEsrganState() {
        activeEsrganJobId = null;
        esrganProcessingClipId = null;
        esrganProgressPercent = 0;
        if (typeof SerenityWS !== 'undefined') {
            SerenityWS.off('esrgan_progress', onEsrganProgress);
            SerenityWS.off('esrgan_complete', onEsrganComplete);
            SerenityWS.off('esrgan_error', onEsrganError);
        }
    }

    // ===== V10: Deflicker =====

    function showDeflickerDialog(clipId) {
        if (deflickerDialogEl) deflickerDialogEl.remove();

        var overlay = document.createElement('div');
        overlay.className = 've-deflicker-overlay';
        overlay.innerHTML =
            '<div class="ve-deflicker-dialog">' +
                '<div class="ve-deflicker-header">' +
                    '<span>Deflicker</span>' +
                    '<span class="ve-deflicker-close">\u00D7</span>' +
                '</div>' +
                '<div class="ve-deflicker-body">' +
                    '<div class="ve-deflicker-mode-row">' +
                        '<label><input type="radio" name="df-mode" value="light" checked /> Light \u2014 ffmpeg filter, instant</label>' +
                        '<label><input type="radio" name="df-mode" value="medium" /> Medium \u2014 histogram matching, fast</label>' +
                        '<label><input type="radio" name="df-mode" value="heavy" /> Heavy \u2014 optical flow blend, slow</label>' +
                    '</div>' +
                    '<div class="ve-deflicker-settings ve-deflicker-light-settings">' +
                        '<div class="ve-deflicker-slider-row"><label>Window:</label>' +
                            '<input type="range" class="df-window" min="3" max="15" step="2" value="5" />' +
                            '<span class="ve-deflicker-val df-window-val">5</span>' +
                        '</div>' +
                    '</div>' +
                    '<div class="ve-deflicker-settings ve-deflicker-medium-settings" style="display:none">' +
                        '<div class="ve-deflicker-slider-row"><label>Strength:</label>' +
                            '<input type="range" class="df-strength" min="0" max="1" step="0.05" value="0.7" />' +
                            '<span class="ve-deflicker-val df-strength-val">0.70</span>' +
                        '</div>' +
                        '<div class="ve-deflicker-slider-row"><label>EMA Decay:</label>' +
                            '<input type="range" class="df-ema" min="0.5" max="0.99" step="0.01" value="0.85" />' +
                            '<span class="ve-deflicker-val df-ema-val">0.85</span>' +
                        '</div>' +
                    '</div>' +
                    '<div class="ve-deflicker-settings ve-deflicker-heavy-settings" style="display:none">' +
                        '<div class="ve-deflicker-slider-row"><label>Blend Alpha:</label>' +
                            '<input type="range" class="df-alpha" min="0.05" max="0.3" step="0.01" value="0.15" />' +
                            '<span class="ve-deflicker-val df-alpha-val">0.15</span>' +
                        '</div>' +
                    '</div>' +
                    '<div class="ve-deflicker-actions">' +
                        '<button class="ve-deflicker-btn ve-deflicker-btn-secondary ve-deflicker-btn-cancel">Cancel</button>' +
                        '<button class="ve-deflicker-btn ve-deflicker-btn-primary ve-deflicker-btn-start">Start</button>' +
                    '</div>' +
                '</div>' +
            '</div>';

        document.body.appendChild(overlay);
        deflickerDialogEl = overlay;

        // Wire mode radio to show/hide settings
        var radios = overlay.querySelectorAll('input[name="df-mode"]');
        radios.forEach(function (r) {
            r.addEventListener('change', function () {
                overlay.querySelector('.ve-deflicker-light-settings').style.display = r.value === 'light' ? '' : 'none';
                overlay.querySelector('.ve-deflicker-medium-settings').style.display = r.value === 'medium' ? '' : 'none';
                overlay.querySelector('.ve-deflicker-heavy-settings').style.display = r.value === 'heavy' ? '' : 'none';
            });
        });

        // Wire sliders to value display
        var windowSlider = overlay.querySelector('.df-window');
        windowSlider.addEventListener('input', function () {
            overlay.querySelector('.df-window-val').textContent = windowSlider.value;
        });
        var strengthSlider = overlay.querySelector('.df-strength');
        strengthSlider.addEventListener('input', function () {
            overlay.querySelector('.df-strength-val').textContent = parseFloat(strengthSlider.value).toFixed(2);
        });
        var emaSlider = overlay.querySelector('.df-ema');
        emaSlider.addEventListener('input', function () {
            overlay.querySelector('.df-ema-val').textContent = parseFloat(emaSlider.value).toFixed(2);
        });
        var alphaSlider = overlay.querySelector('.df-alpha');
        alphaSlider.addEventListener('input', function () {
            overlay.querySelector('.df-alpha-val').textContent = parseFloat(alphaSlider.value).toFixed(2);
        });

        overlay.querySelector('.ve-deflicker-close').addEventListener('click', function () {
            overlay.remove(); deflickerDialogEl = null;
        });
        overlay.querySelector('.ve-deflicker-btn-cancel').addEventListener('click', function () {
            overlay.remove(); deflickerDialogEl = null;
        });

        overlay.querySelector('.ve-deflicker-btn-start').addEventListener('click', function () {
            var mode = overlay.querySelector('input[name="df-mode"]:checked').value;
            var params = { mode: mode };
            if (mode === 'light') params.window = parseInt(windowSlider.value);
            if (mode === 'medium') {
                params.strength = parseFloat(strengthSlider.value);
                params.ema_decay = parseFloat(emaSlider.value);
            }
            if (mode === 'heavy') params.blend_alpha = parseFloat(alphaSlider.value);
            overlay.remove(); deflickerDialogEl = null;
            startDeflicker(clipId, params);
        });
    }

    function startDeflicker(clipId, params) {
        var body = {
            project_id: projectId,
            clip_id: clipId,
        };
        for (var k in params) body[k] = params[k];

        fetch(getApiBase() + '/video_edit/deflicker', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        })
        .then(function (r) { return r.json(); })
        .then(function (res) {
            if (res.error) {
                alert('Deflicker error: ' + res.error);
                return;
            }
            activeDeflickerJobId = res.job_id;
            deflickerProcessingClipId = clipId;
            deflickerProgressPercent = 0;
            renderTimeline();

            if (typeof SerenityWS !== 'undefined') {
                SerenityWS.on('deflicker_progress', onDeflickerProgress);
                SerenityWS.on('deflicker_complete', onDeflickerComplete);
                SerenityWS.on('deflicker_error', onDeflickerError);
            }
        })
        .catch(function (err) { alert('Deflicker request failed: ' + err); });
    }

    function onDeflickerProgress(msg) {
        var d = msg.data || msg;
        if (activeDeflickerJobId && d.job_id !== activeDeflickerJobId) return;
        deflickerProgressPercent = d.percent || 0;
        renderTimeline();
    }

    function onDeflickerComplete(msg) {
        var d = msg.data || msg;
        if (activeDeflickerJobId && d.job_id !== activeDeflickerJobId) return;

        if (d.clip_id && d.output_path) {
            var info = findClipById(d.clip_id);
            if (info) {
                info.clip.source_path = d.output_path;
                info.clip.deflickered = d.mode;
            }
        }

        cleanupDeflickerState();
        renderTimeline();
        scheduleAutosave();
    }

    function onDeflickerError(msg) {
        var d = msg.data || msg;
        if (activeDeflickerJobId && d.job_id !== activeDeflickerJobId) return;
        cleanupDeflickerState();
        renderTimeline();
        alert('Deflicker failed: ' + (d.error || 'Unknown error'));
    }

    function cleanupDeflickerState() {
        activeDeflickerJobId = null;
        deflickerProcessingClipId = null;
        deflickerProgressPercent = 0;
        if (typeof SerenityWS !== 'undefined') {
            SerenityWS.off('deflicker_progress', onDeflickerProgress);
            SerenityWS.off('deflicker_complete', onDeflickerComplete);
            SerenityWS.off('deflicker_error', onDeflickerError);
        }
    }

    // ===== V11: Audio Enhancement =====

    function showAudioEnhanceDialog(clipId) {
        if (audioDialogEl) audioDialogEl.remove();

        function buildDialog(presets) {
            var overlay = document.createElement('div');
            overlay.className = 've-audio-overlay';

            var ffKeys = Object.keys(presets.ffmpeg);
            var optionsHtml = ffKeys.map(function (k) {
                return '<option value="' + k + '"' + (k === 'clean_speech' ? ' selected' : '') + '>' + presets.ffmpeg[k].name + '</option>';
            }).join('');

            var dfAvail = presets.deepfilter && presets.deepfilter.available;

            overlay.innerHTML =
                '<div class="ve-audio-dialog">' +
                    '<div class="ve-audio-header">' +
                        '<span>Enhance Audio</span>' +
                        '<span class="ve-audio-close">\u00D7</span>' +
                    '</div>' +
                    '<div class="ve-audio-body">' +
                        '<div class="ve-audio-mode-row">' +
                            '<label><input type="radio" name="audio-mode" value="ffmpeg" checked /> ffmpeg Presets</label>' +
                            '<label><input type="radio" name="audio-mode" value="deepfilter" ' + (dfAvail ? '' : 'disabled') + ' /> AI (DeepFilter)' + (dfAvail ? '' : ' <span style="color:#888">(not installed)</span>') + '</label>' +
                        '</div>' +
                        '<div class="ve-audio-settings ve-audio-ffmpeg-settings">' +
                            '<div class="ve-audio-preset-row">' +
                                '<label>Preset:</label>' +
                                '<select class="ve-audio-preset">' + optionsHtml + '</select>' +
                            '</div>' +
                            '<div class="ve-audio-desc">' + (presets.ffmpeg.clean_speech ? presets.ffmpeg.clean_speech.description : '') + '</div>' +
                        '</div>' +
                        '<div class="ve-audio-settings ve-audio-df-settings" style="display:none">' +
                            '<div class="ve-audio-desc">AI-powered speech enhancement. Removes background noise, reverb, and non-speech sounds while preserving voice quality.' + (dfAvail ? '' : '<br><br><code>pip install deepfilternet</code>') + '</div>' +
                        '</div>' +
                        '<div class="ve-audio-actions">' +
                            '<button class="ve-audio-btn ve-audio-btn-preview">Preview 5s</button>' +
                            '<button class="ve-audio-btn ve-audio-btn-secondary ve-audio-btn-cancel">Cancel</button>' +
                            '<button class="ve-audio-btn ve-audio-btn-primary ve-audio-btn-apply">Apply</button>' +
                        '</div>' +
                    '</div>' +
                '</div>';

            document.body.appendChild(overlay);
            audioDialogEl = overlay;

            // Mode toggle
            var radios = overlay.querySelectorAll('input[name="audio-mode"]');
            radios.forEach(function (r) {
                r.addEventListener('change', function () {
                    overlay.querySelector('.ve-audio-ffmpeg-settings').style.display = r.value === 'ffmpeg' ? '' : 'none';
                    overlay.querySelector('.ve-audio-df-settings').style.display = r.value === 'deepfilter' ? '' : 'none';
                });
            });

            // Preset change updates description
            var presetSelect = overlay.querySelector('.ve-audio-preset');
            var descEl = overlay.querySelector('.ve-audio-ffmpeg-settings .ve-audio-desc');
            presetSelect.addEventListener('change', function () {
                var p = presets.ffmpeg[presetSelect.value];
                descEl.textContent = p ? p.description : '';
            });

            overlay.querySelector('.ve-audio-close').addEventListener('click', function () {
                overlay.remove(); audioDialogEl = null;
            });
            overlay.querySelector('.ve-audio-btn-cancel').addEventListener('click', function () {
                overlay.remove(); audioDialogEl = null;
            });

            // Preview 5s
            overlay.querySelector('.ve-audio-btn-preview').addEventListener('click', function () {
                var btn = overlay.querySelector('.ve-audio-btn-preview');
                btn.textContent = 'Loading...';
                btn.disabled = true;
                fetch(getApiBase() + '/video_edit/audio/preview', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        project_id: projectId,
                        clip_id: clipId,
                        preset: presetSelect.value,
                        duration: 5,
                    }),
                })
                .then(function (r) {
                    if (!r.ok) throw new Error('Preview failed');
                    return r.blob();
                })
                .then(function (blob) {
                    var url = URL.createObjectURL(blob);
                    var audio = new Audio(url);
                    audio.addEventListener('ended', function () { URL.revokeObjectURL(url); });
                    audio.play();
                    btn.textContent = 'Preview 5s';
                    btn.disabled = false;
                })
                .catch(function (err) {
                    btn.textContent = 'Preview 5s';
                    btn.disabled = false;
                    alert('Preview failed: ' + err);
                });
            });

            // Apply
            overlay.querySelector('.ve-audio-btn-apply').addEventListener('click', function () {
                var mode = overlay.querySelector('input[name="audio-mode"]:checked').value;
                var preset = presetSelect.value;
                overlay.remove(); audioDialogEl = null;
                startAudioEnhance(clipId, mode, preset);
            });
        }

        // Fetch presets (cache after first load)
        if (audioPresetsCache) {
            buildDialog(audioPresetsCache);
        } else {
            fetch(getApiBase() + '/video_edit/audio/presets')
                .then(function (r) { return r.json(); })
                .then(function (data) {
                    audioPresetsCache = data;
                    buildDialog(data);
                })
                .catch(function (err) { alert('Could not load audio presets: ' + err); });
        }
    }

    function startAudioEnhance(clipId, mode, preset) {
        fetch(getApiBase() + '/video_edit/audio/enhance', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project_id: projectId,
                clip_id: clipId,
                mode: mode,
                preset: preset,
            }),
        })
        .then(function (r) { return r.json(); })
        .then(function (res) {
            if (res.error) {
                alert('Audio enhance error: ' + res.error);
                return;
            }
            activeAudioJobId = res.job_id;
            audioProcessingClipId = clipId;
            audioProgressPercent = 0;
            renderTimeline();

            if (typeof SerenityWS !== 'undefined') {
                SerenityWS.on('audio_enhance_progress', onAudioProgress);
                SerenityWS.on('audio_enhance_complete', onAudioComplete);
                SerenityWS.on('audio_enhance_error', onAudioError);
            }
        })
        .catch(function (err) { alert('Audio enhance request failed: ' + err); });
    }

    function onAudioProgress(msg) {
        var d = msg.data || msg;
        if (activeAudioJobId && d.job_id !== activeAudioJobId) return;
        audioProgressPercent = d.percent || 0;
        renderTimeline();
    }

    function onAudioComplete(msg) {
        var d = msg.data || msg;
        if (activeAudioJobId && d.job_id !== activeAudioJobId) return;

        if (d.clip_id && d.output_path) {
            var info = findClipById(d.clip_id);
            if (info) {
                info.clip.source_path = d.output_path;
                info.clip.audio_enhanced = d.preset;
            }
        }

        cleanupAudioState();
        renderTimeline();
        scheduleAutosave();
    }

    function onAudioError(msg) {
        var d = msg.data || msg;
        if (activeAudioJobId && d.job_id !== activeAudioJobId) return;
        cleanupAudioState();
        renderTimeline();
        alert('Audio enhancement failed: ' + (d.error || 'Unknown error'));
    }

    function cleanupAudioState() {
        activeAudioJobId = null;
        audioProcessingClipId = null;
        audioProgressPercent = 0;
        if (typeof SerenityWS !== 'undefined') {
            SerenityWS.off('audio_enhance_progress', onAudioProgress);
            SerenityWS.off('audio_enhance_complete', onAudioComplete);
            SerenityWS.off('audio_enhance_error', onAudioError);
        }
    }

    function renderTracks() {
        timelineLayer.destroyChildren();

        var timelineHeight = project.tracks.length * TRACK_HEIGHT;

        // Timeline body background (clickable for deselect)
        var bgRect = new Konva.Rect({
            x: TRACK_HEADER_WIDTH, y: RULER_HEIGHT,
            width: stage.width() - TRACK_HEADER_WIDTH, height: timelineHeight,
            fill: '#12121e'
        });
        bgRect.setAttr('isBg', true);
        timelineLayer.add(bgRect);

        for (var i = 0; i < project.tracks.length; i++) {
            (function (trackIdx) {
                var track = project.tracks[trackIdx];
                var y = RULER_HEIGHT + (trackIdx * TRACK_HEIGHT);

                // Track header
                timelineLayer.add(new Konva.Rect({
                    x: 0, y: y,
                    width: TRACK_HEADER_WIDTH, height: TRACK_HEIGHT,
                    fill: '#1a1a2e'
                }));
                timelineLayer.add(new Konva.Text({
                    x: 10, y: y + 16,
                    text: track.name,
                    fontSize: 12, fill: '#ccc', fontFamily: 'sans-serif'
                }));

                // Separator
                timelineLayer.add(new Konva.Line({
                    points: [0, y + TRACK_HEIGHT, stage.width(), y + TRACK_HEIGHT],
                    stroke: '#2a2a3a', strokeWidth: 1
                }));

                // Clips
                track.clips.forEach(function (clip) {
                    var clipX = TRACK_HEADER_WIDTH + (clip.startFrame * pixelsPerFrame) - scrollOffsetX;
                    var clipW = (clip.endFrame - clip.startFrame) * pixelsPerFrame;

                    if (clipX + clipW < TRACK_HEADER_WIDTH || clipX > stage.width()) return;

                    var clipH = TRACK_HEIGHT - 8;
                    var clipY = y + 4;
                    var visibleW = clipW - Math.max(0, TRACK_HEADER_WIDTH - clipX);
                    var rectOffset = Math.max(0, TRACK_HEADER_WIDTH - clipX);
                    var isSelected = selectedClipIds.has(clip.id);

                    var group = new Konva.Group({
                        x: Math.max(clipX, TRACK_HEADER_WIDTH),
                        y: clipY,
                        clipFunc: function (ctx) {
                            ctx.rect(0, 0, visibleW, clipH);
                        }
                    });
                    group.setAttr('clipId', clip.id);
                    group.setAttr('trackId', track.id);
                    group.setAttr('trackIdx', trackIdx);

                    // Thumbnails / waveform (behind clip color overlay)
                    var hasThumbs = thumbnailCache.has(clip.id);
                    var hasWaveform = waveformCache.has(clip.id);

                    // Clip body (semi-transparent if thumbnails/waveform present)
                    group.add(new Konva.Rect({
                        x: -rectOffset, y: 0,
                        width: clipW, height: clipH,
                        fill: clip.color,
                        opacity: (hasThumbs || hasWaveform) ? (isSelected ? 0.6 : 0.5) : (isSelected ? 1.0 : 0.85),
                        cornerRadius: 4,
                        stroke: isSelected ? '#fff' : null,
                        strokeWidth: isSelected ? 2 : 0,
                        name: 'clipBody'
                    }));

                    // Render thumbnails on video clips
                    if (hasThumbs && track.type === 'video') {
                        var tc = thumbnailCache.get(clip.id);
                        var thumbX = 0;
                        var sec = 0;
                        var sourceStartSec = (clip.source_start || 0) / FPS;
                        while (thumbX < clipW && sec < tc.frameCount) {
                            var spriteIdx = Math.min(Math.floor(sourceStartSec + sec), tc.frameCount - 1);
                            group.add(new Konva.Image({
                                x: thumbX - rectOffset, y: 0,
                                width: tc.thumbW, height: tc.thumbH,
                                image: tc.img,
                                crop: { x: spriteIdx * tc.thumbW, y: 0, width: tc.thumbW, height: tc.thumbH },
                                opacity: 0.45,
                                listening: false,
                            }));
                            thumbX += tc.thumbW;
                            sec++;
                        }
                    }

                    // Render waveform on audio clips (or video clips with audio)
                    if (hasWaveform && (track.type === 'audio' || track.type === 'video')) {
                        var wd = waveformCache.get(clip.id);
                        var midY = clipH / 2;
                        var sourceStartSample = Math.floor((clip.source_start || 0) / FPS * wd.sample_rate);
                        var clipDurFrames = clip.endFrame - clip.startFrame;
                        var samplesInClip = Math.floor(clipDurFrames / FPS * wd.sample_rate);
                        var wfPoints = [];
                        // Draw every 2px for performance
                        var step = Math.max(2, Math.floor(clipW / 300));
                        for (var px = 0; px < clipW; px += step) {
                            var sIdx = sourceStartSample + Math.floor(px / clipW * samplesInClip);
                            if (sIdx >= wd.peaks.length) break;
                            var amp = wd.peaks[sIdx] || 0;
                            var barH = amp * (clipH * 0.8) / 2;
                            // Top line
                            group.add(new Konva.Line({
                                points: [px - rectOffset, midY - barH, px - rectOffset, midY + barH],
                                stroke: 'rgba(255,255,255,0.45)',
                                strokeWidth: Math.max(1, step - 1),
                                listening: false,
                            }));
                        }
                    }

                    // Loading indicator
                    if (clip.source_path && !hasThumbs && !hasWaveform) {
                        if (track.type === 'video' && thumbnailLoading.has(clip.id)) {
                            group.add(new Konva.Text({
                                x: 6 - rectOffset, y: clipH / 2 - 5,
                                text: 'Loading...', fontSize: 9,
                                fill: 'rgba(255,255,255,0.4)',
                                fontFamily: 'sans-serif', listening: false,
                            }));
                        }
                        if (track.type === 'audio' && waveformLoading.has(clip.id)) {
                            group.add(new Konva.Text({
                                x: 6 - rectOffset, y: clipH / 2 - 5,
                                text: 'Loading audio...', fontSize: 9,
                                fill: 'rgba(255,255,255,0.4)',
                                fontFamily: 'sans-serif', listening: false,
                            }));
                        }
                    }

                    // Lazy-load thumbnails/waveforms for visible clips
                    if (clip.source_path) {
                        if (track.type === 'video') loadThumbnails(clip);
                        if (track.type === 'audio') loadWaveform(clip);
                    }

                    // Label (on top of everything)
                    group.add(new Konva.Text({
                        x: 6 - rectOffset, y: clipH / 2 - 6,
                        text: clip.label,
                        fontSize: 11, fill: '#fff', fontFamily: 'sans-serif',
                        listening: false
                    }));

                    // Multi-take badge
                    if (clip.takes && clip.takes.length > 1) {
                        var badgeText = clip.takes.length + ' takes';
                        var badgeW = badgeText.length * 5.5 + 10;
                        group.add(new Konva.Rect({
                            x: clipW - rectOffset - badgeW - 4, y: clipH - 15,
                            width: badgeW, height: 13,
                            fill: 'rgba(0,0,0,0.5)', cornerRadius: 3,
                            listening: false,
                        }));
                        group.add(new Konva.Text({
                            x: clipW - rectOffset - badgeW, y: clipH - 14,
                            text: badgeText,
                            fontSize: 9, fill: '#ff8c42', fontFamily: 'sans-serif',
                            listening: false,
                        }));
                    }

                    // Generating indicator
                    if (generatingClips.has(clip.id)) {
                        group.add(new Konva.Rect({
                            x: -rectOffset, y: 0,
                            width: clipW, height: clipH,
                            fill: '#ff8c42', opacity: 0.2,
                            cornerRadius: 4, listening: false,
                        }));
                        group.add(new Konva.Text({
                            x: clipW / 2 - rectOffset - 30, y: clipH / 2 - 5,
                            text: 'Generating...',
                            fontSize: 10, fill: '#ff8c42', fontFamily: 'sans-serif',
                            listening: false,
                        }));
                    }

                    // V6: Effects badge
                    var badgeOffsetX = 4;
                    if (clip.effects && clip.effects.length > 0) {
                        var fxBadgeW = 22;
                        group.add(new Konva.Rect({
                            x: badgeOffsetX - rectOffset, y: clipH - 14,
                            width: fxBadgeW, height: 12,
                            fill: 'rgba(0,0,0,0.55)', cornerRadius: 3,
                            listening: false,
                        }));
                        group.add(new Konva.Text({
                            x: badgeOffsetX + 2 - rectOffset, y: clipH - 13,
                            text: 'FX',
                            fontSize: 9, fill: '#7c6ff0', fontFamily: 'sans-serif',
                            fontStyle: 'bold', listening: false,
                        }));
                        badgeOffsetX += fxBadgeW + 3;
                    }

                    // V7: LUT badge
                    if (clip.lut_path) {
                        var lutBadgeW = 26;
                        group.add(new Konva.Rect({
                            x: badgeOffsetX - rectOffset, y: clipH - 14,
                            width: lutBadgeW, height: 12,
                            fill: 'rgba(0,0,0,0.55)', cornerRadius: 3,
                            listening: false,
                        }));
                        group.add(new Konva.Text({
                            x: badgeOffsetX + 2 - rectOffset, y: clipH - 13,
                            text: 'LUT',
                            fontSize: 9, fill: '#e8a040', fontFamily: 'sans-serif',
                            fontStyle: 'bold', listening: false,
                        }));
                    }

                    // V7: RIFE multiplier badge (after interpolation completes)
                    if (clip.rife_multiplier && !rifeProcessingClipId) {
                        var rifeBadgeText = clip.rife_multiplier + '\u00D7';
                        var rifeBadgeW = rifeBadgeText.length * 6 + 8;
                        group.add(new Konva.Rect({
                            x: badgeOffsetX - rectOffset, y: clipH - 14,
                            width: rifeBadgeW, height: 12,
                            fill: 'rgba(0,0,0,0.55)', cornerRadius: 3,
                            listening: false,
                        }));
                        group.add(new Konva.Text({
                            x: badgeOffsetX + 2 - rectOffset, y: clipH - 13,
                            text: rifeBadgeText,
                            fontSize: 9, fill: '#50c878', fontFamily: 'sans-serif',
                            fontStyle: 'bold', listening: false,
                        }));
                    }

                    // V7: RIFE processing overlay
                    if (rifeProcessingClipId === clip.id) {
                        group.add(new Konva.Rect({
                            x: -rectOffset, y: 0,
                            width: clipW, height: clipH,
                            fill: '#7c6ff0', opacity: 0.18,
                            cornerRadius: 4, listening: false,
                        }));
                        // Progress bar
                        var barY = clipH / 2 - 3;
                        var barW = Math.max(20, clipW - 20);
                        group.add(new Konva.Rect({
                            x: 10 - rectOffset, y: barY,
                            width: barW, height: 6,
                            fill: 'rgba(0,0,0,0.5)', cornerRadius: 3,
                            listening: false,
                        }));
                        var fillW = Math.max(0, barW * rifeProgressPercent / 100);
                        group.add(new Konva.Rect({
                            x: 10 - rectOffset, y: barY,
                            width: fillW, height: 6,
                            fill: '#7c6ff0', cornerRadius: 3,
                            listening: false,
                        }));
                        group.add(new Konva.Text({
                            x: clipW / 2 - rectOffset - 30, y: barY + 9,
                            text: 'RIFE ' + rifeProgressPercent + '%',
                            fontSize: 9, fill: '#ccc', fontFamily: 'sans-serif',
                            listening: false,
                        }));
                    }

                    // V8: Face restoration processing overlay
                    if (faceProcessingClipId === clip.id) {
                        group.add(new Konva.Rect({
                            x: -rectOffset, y: 0,
                            width: clipW, height: clipH,
                            fill: '#f07c6f', opacity: 0.18,
                            cornerRadius: 4, listening: false,
                        }));
                        var fBarY = clipH / 2 - 3;
                        var fBarW = Math.max(20, clipW - 20);
                        group.add(new Konva.Rect({
                            x: 10 - rectOffset, y: fBarY,
                            width: fBarW, height: 6,
                            fill: 'rgba(0,0,0,0.5)', cornerRadius: 3,
                            listening: false,
                        }));
                        var fFillW = Math.max(0, fBarW * faceProgressPercent / 100);
                        group.add(new Konva.Rect({
                            x: 10 - rectOffset, y: fBarY,
                            width: fFillW, height: 6,
                            fill: '#f07c6f', cornerRadius: 3,
                            listening: false,
                        }));
                        group.add(new Konva.Text({
                            x: clipW / 2 - rectOffset - 40, y: fBarY + 9,
                            text: 'Faces ' + faceProgressPercent + '%',
                            fontSize: 9, fill: '#ccc', fontFamily: 'sans-serif',
                            listening: false,
                        }));
                    }

                    // V8: Face restored badge
                    if (clip.face_restored && faceProcessingClipId !== clip.id) {
                        var faceBadgeW = 30;
                        group.add(new Konva.Rect({
                            x: badgeOffsetX - rectOffset, y: clipH - 14,
                            width: faceBadgeW, height: 12,
                            fill: 'rgba(0,0,0,0.55)', cornerRadius: 3,
                            listening: false,
                        }));
                        group.add(new Konva.Text({
                            x: badgeOffsetX + 2 - rectOffset, y: clipH - 13,
                            text: 'FACE',
                            fontSize: 9, fill: '#f0a06f', fontFamily: 'sans-serif',
                            fontStyle: 'bold', listening: false,
                        }));
                        badgeOffsetX += faceBadgeW + 3;
                    }

                    // V9: ESRGAN processing overlay
                    if (esrganProcessingClipId === clip.id) {
                        group.add(new Konva.Rect({
                            x: -rectOffset, y: 0,
                            width: clipW, height: clipH,
                            fill: '#6fc8f0', opacity: 0.18,
                            cornerRadius: 4, listening: false,
                        }));
                        var eBarY = clipH / 2 - 3;
                        var eBarW = Math.max(20, clipW - 20);
                        group.add(new Konva.Rect({
                            x: 10 - rectOffset, y: eBarY,
                            width: eBarW, height: 6,
                            fill: 'rgba(0,0,0,0.5)', cornerRadius: 3,
                            listening: false,
                        }));
                        var eFillW = Math.max(0, eBarW * esrganProgressPercent / 100);
                        group.add(new Konva.Rect({
                            x: 10 - rectOffset, y: eBarY,
                            width: eFillW, height: 6,
                            fill: '#6fc8f0', cornerRadius: 3,
                            listening: false,
                        }));
                        group.add(new Konva.Text({
                            x: clipW / 2 - rectOffset - 40, y: eBarY + 9,
                            text: 'Upscale ' + esrganProgressPercent + '%',
                            fontSize: 9, fill: '#ccc', fontFamily: 'sans-serif',
                            listening: false,
                        }));
                    }

                    // V9: ESRGAN upscale badge
                    if (clip.esrgan_scale && esrganProcessingClipId !== clip.id) {
                        var esrganBadgeText = clip.esrgan_scale + '\u00D7';
                        var esrganBadgeW = esrganBadgeText.length * 6 + 8;
                        group.add(new Konva.Rect({
                            x: badgeOffsetX - rectOffset, y: clipH - 14,
                            width: esrganBadgeW, height: 12,
                            fill: 'rgba(0,0,0,0.55)', cornerRadius: 3,
                            listening: false,
                        }));
                        group.add(new Konva.Text({
                            x: badgeOffsetX + 2 - rectOffset, y: clipH - 13,
                            text: esrganBadgeText,
                            fontSize: 9, fill: '#6fc8f0', fontFamily: 'sans-serif',
                            fontStyle: 'bold', listening: false,
                        }));
                        badgeOffsetX += esrganBadgeW + 3;
                    }

                    // V10: Deflicker processing overlay
                    if (deflickerProcessingClipId === clip.id) {
                        group.add(new Konva.Rect({
                            x: -rectOffset, y: 0,
                            width: clipW, height: clipH,
                            fill: '#a0d468', opacity: 0.18,
                            cornerRadius: 4, listening: false,
                        }));
                        var dBarY = clipH / 2 - 3;
                        var dBarW = Math.max(20, clipW - 20);
                        group.add(new Konva.Rect({
                            x: 10 - rectOffset, y: dBarY,
                            width: dBarW, height: 6,
                            fill: 'rgba(0,0,0,0.5)', cornerRadius: 3,
                            listening: false,
                        }));
                        var dFillW = Math.max(0, dBarW * deflickerProgressPercent / 100);
                        group.add(new Konva.Rect({
                            x: 10 - rectOffset, y: dBarY,
                            width: dFillW, height: 6,
                            fill: '#a0d468', cornerRadius: 3,
                            listening: false,
                        }));
                        group.add(new Konva.Text({
                            x: clipW / 2 - rectOffset - 40, y: dBarY + 9,
                            text: 'Deflicker ' + deflickerProgressPercent + '%',
                            fontSize: 9, fill: '#ccc', fontFamily: 'sans-serif',
                            listening: false,
                        }));
                    }

                    // V10: Deflicker badge
                    if (clip.deflickered && deflickerProcessingClipId !== clip.id) {
                        var dfBadgeW = 24;
                        group.add(new Konva.Rect({
                            x: badgeOffsetX - rectOffset, y: clipH - 14,
                            width: dfBadgeW, height: 12,
                            fill: 'rgba(0,0,0,0.55)', cornerRadius: 3,
                            listening: false,
                        }));
                        group.add(new Konva.Text({
                            x: badgeOffsetX + 2 - rectOffset, y: clipH - 13,
                            text: 'DF',
                            fontSize: 9, fill: '#a0d468', fontFamily: 'sans-serif',
                            fontStyle: 'bold', listening: false,
                        }));
                        badgeOffsetX += dfBadgeW + 3;
                    }

                    // V11: Audio enhancement processing overlay
                    if (audioProcessingClipId === clip.id) {
                        group.add(new Konva.Rect({
                            x: -rectOffset, y: 0,
                            width: clipW, height: clipH,
                            fill: '#d4a0e8', opacity: 0.18,
                            cornerRadius: 4, listening: false,
                        }));
                        var aBarY = clipH / 2 - 3;
                        var aBarW = Math.max(20, clipW - 20);
                        group.add(new Konva.Rect({
                            x: 10 - rectOffset, y: aBarY,
                            width: aBarW, height: 6,
                            fill: 'rgba(0,0,0,0.5)', cornerRadius: 3,
                            listening: false,
                        }));
                        var aFillW = Math.max(0, aBarW * audioProgressPercent / 100);
                        group.add(new Konva.Rect({
                            x: 10 - rectOffset, y: aBarY,
                            width: aFillW, height: 6,
                            fill: '#d4a0e8', cornerRadius: 3,
                            listening: false,
                        }));
                        group.add(new Konva.Text({
                            x: clipW / 2 - rectOffset - 40, y: aBarY + 9,
                            text: 'Audio ' + audioProgressPercent + '%',
                            fontSize: 9, fill: '#ccc', fontFamily: 'sans-serif',
                            listening: false,
                        }));
                    }

                    // V11: Audio enhanced badge
                    if (clip.audio_enhanced && audioProcessingClipId !== clip.id) {
                        var audioBadgeW = 28;
                        group.add(new Konva.Rect({
                            x: badgeOffsetX - rectOffset, y: clipH - 14,
                            width: audioBadgeW, height: 12,
                            fill: 'rgba(0,0,0,0.55)', cornerRadius: 3,
                            listening: false,
                        }));
                        group.add(new Konva.Text({
                            x: badgeOffsetX + 2 - rectOffset, y: clipH - 13,
                            text: '\u266A',
                            fontSize: 9, fill: '#d4a0e8', fontFamily: 'sans-serif',
                            fontStyle: 'bold', listening: false,
                        }));
                        badgeOffsetX += audioBadgeW + 3;
                    }

                    // V6: Transition-in indicator
                    if (clip.transition_in && clip.transition_in.type !== 'none') {
                        group.add(new Konva.Text({
                            x: -rectOffset + 2, y: 1,
                            text: '\u25B6',
                            fontSize: 8, fill: 'rgba(124,111,240,0.7)',
                            listening: false,
                        }));
                    }

                    // Left trim handle
                    var leftHandle = new Konva.Rect({
                        x: -rectOffset, y: 0,
                        width: TRIM_HANDLE_WIDTH, height: clipH,
                        fill: 'transparent',
                        name: 'trimLeft'
                    });
                    group.add(leftHandle);

                    // Right trim handle
                    var rightHandle = new Konva.Rect({
                        x: -rectOffset + clipW - TRIM_HANDLE_WIDTH, y: 0,
                        width: TRIM_HANDLE_WIDTH, height: clipH,
                        fill: 'transparent',
                        name: 'trimRight'
                    });
                    group.add(rightHandle);

                    timelineLayer.add(group);
                });
            })(i);
        }

        timelineLayer.batchDraw();
    }

    function renderPlayhead() {
        overlayLayer.destroyChildren();

        // Snap guide line
        if (activeSnapFrame !== null) {
            var snapX = TRACK_HEADER_WIDTH + (activeSnapFrame * pixelsPerFrame) - scrollOffsetX;
            if (snapX >= TRACK_HEADER_WIDTH && snapX <= stage.width()) {
                var fullH = RULER_HEIGHT + (project.tracks.length * TRACK_HEIGHT);
                overlayLayer.add(new Konva.Line({
                    points: [snapX, 0, snapX, fullH],
                    stroke: '#00e5ff', strokeWidth: 1,
                    dash: [4, 4], opacity: 0.7
                }));
            }
        }

        // Retake region overlay
        if (retakeMode && retakeClipId) {
            var rInfo = findClipById(retakeClipId);
            if (rInfo) {
                var rTrackY = RULER_HEIGHT + (rInfo.trackIndex * TRACK_HEIGHT);
                var rStartX = TRACK_HEADER_WIDTH + (retakeRegionStart * pixelsPerFrame) - scrollOffsetX;
                var rEndX = TRACK_HEADER_WIDTH + (retakeRegionEnd * pixelsPerFrame) - scrollOffsetX;
                if (rEndX > TRACK_HEADER_WIDTH && rStartX < stage.width()) {
                    // Orange highlight on region
                    overlayLayer.add(new Konva.Rect({
                        x: Math.max(rStartX, TRACK_HEADER_WIDTH),
                        y: rTrackY + 4,
                        width: Math.min(rEndX, stage.width()) - Math.max(rStartX, TRACK_HEADER_WIDTH),
                        height: TRACK_HEIGHT - 8,
                        fill: '#ff8c42', opacity: 0.25,
                    }));
                    // Left marker
                    if (rStartX >= TRACK_HEADER_WIDTH) {
                        overlayLayer.add(new Konva.Rect({
                            x: rStartX - 2, y: rTrackY + 2,
                            width: 4, height: TRACK_HEIGHT - 4,
                            fill: '#ff8c42', cornerRadius: 2,
                        }));
                    }
                    // Right marker
                    if (rEndX <= stage.width()) {
                        overlayLayer.add(new Konva.Rect({
                            x: rEndX - 2, y: rTrackY + 2,
                            width: 4, height: TRACK_HEIGHT - 4,
                            fill: '#ff8c42', cornerRadius: 2,
                        }));
                    }
                }
            }
        }

        // Playhead
        var x = TRACK_HEADER_WIDTH + (currentFrame * pixelsPerFrame) - scrollOffsetX;
        if (x >= TRACK_HEADER_WIDTH) {
            var fullHeight = RULER_HEIGHT + (project.tracks.length * TRACK_HEIGHT);

            overlayLayer.add(new Konva.RegularPolygon({
                x: x, y: 6,
                sides: 3, radius: 6,
                fill: '#ff4444', rotation: 180
            }));

            overlayLayer.add(new Konva.Line({
                points: [x, 10, x, fullHeight],
                stroke: '#ff4444', strokeWidth: 2
            }));
        }

        overlayLayer.batchDraw();
    }

    function renderTimeline() {
        if (!stage) return;
        renderRuler();
        renderTracks();
        renderPlayhead();
    }

    // ===== Display Updates =====

    function updateTimecodeDisplay() {
        var el = document.getElementById('ve-timecode');
        if (el) el.textContent = frameToTimecode(currentFrame);
    }

    function updateZoomDisplay() {
        var label = document.getElementById('ve-zoom-label');
        var slider = document.getElementById('ve-zoom-slider');
        var pct = Math.round(pixelsPerFrame / DEFAULT_PPF * 100);
        if (label) label.textContent = pct + '%';
        if (slider) slider.value = String(Math.round(pixelsPerFrame * 100));
    }

    function updatePlayButton() {
        var btn = document.getElementById('ve-btn-play');
        if (!btn) return;
        btn.innerHTML = isPlaying ? '&#9646;&#9646;' : '&#9654;';
        btn.classList.toggle('ve-active', isPlaying);
    }

    // ===== Playback =====

    function startPlayback() {
        if (currentFrame >= totalFrames) currentFrame = 0;
        isPlaying = true;
        playStartTime = performance.now();
        playStartFrame = currentFrame;
        updatePlayButton();
        tick();
    }

    function tick() {
        if (!isPlaying) return;
        var elapsed = (performance.now() - playStartTime) / 1000;
        currentFrame = playStartFrame + Math.floor(elapsed * FPS);

        if (currentFrame >= totalFrames) {
            currentFrame = totalFrames;
            stopPlayback();
            updateTimecodeDisplay();
            renderPlayhead();
            updatePreview();
            return;
        }

        var playheadX = TRACK_HEADER_WIDTH + (currentFrame * pixelsPerFrame) - scrollOffsetX;
        if (playheadX > stage.width() - 60) {
            scrollOffsetX += (stage.width() - TRACK_HEADER_WIDTH) * 0.5;
            scrollOffsetX = clamp(scrollOffsetX, 0, getMaxScroll());
            renderTimeline();
        } else {
            renderPlayhead();
        }

        updateTimecodeDisplay();
        syncAudio();
        updatePreview();
        animFrameId = requestAnimationFrame(tick);
    }

    function stopPlayback() {
        isPlaying = false;
        if (animFrameId) cancelAnimationFrame(animFrameId);
        animFrameId = null;
        stopAllAudio();
        updatePlayButton();
    }

    function togglePlayback() {
        isPlaying ? stopPlayback() : startPlayback();
    }

    // ===== Event Handling =====

    function bindEvents() {
        if (!stage) return;

        // Wheel: zoom / scroll
        stage.on('wheel', function (e) {
            e.evt.preventDefault();
            var pointer = stage.getPointerPosition();
            if (!pointer) return;

            // Ctrl+wheel = zoom, plain wheel = horizontal scroll
            if (e.evt.ctrlKey || e.evt.metaKey) {
                // Zoom toward cursor
                if (pointer.x < TRACK_HEADER_WIDTH) return;

                var frameAtCursor = (pointer.x - TRACK_HEADER_WIDTH + scrollOffsetX) / pixelsPerFrame;
                var factor = e.evt.deltaY > 0 ? 0.9 : 1.1;
                pixelsPerFrame = clamp(pixelsPerFrame * factor, MIN_PPF, MAX_PPF);

                scrollOffsetX = (frameAtCursor * pixelsPerFrame) - (pointer.x - TRACK_HEADER_WIDTH);
                scrollOffsetX = clamp(scrollOffsetX, 0, getMaxScroll());

                updateZoomDisplay();
                renderTimeline();
            } else {
                // Horizontal scroll
                var delta = e.evt.shiftKey ? e.evt.deltaY : e.evt.deltaX || e.evt.deltaY;
                scrollOffsetX += delta * 2;
                scrollOffsetX = clamp(scrollOffsetX, 0, getMaxScroll());
                renderTimeline();
            }
        });

        // --- Mouse events for selection, drag, trim, scrub ---
        stage.on('mousedown', function (e) {
            hideContextMenu();
            var pos = stage.getPointerPosition();
            if (!pos) return;

            // Ruler scrubbing
            if (pos.y <= RULER_HEIGHT && pos.x >= TRACK_HEADER_WIDTH) {
                isScrubbing = true;
                scrubToPosition(pos.x);
                return;
            }

            // Find what was clicked — try Konva target first, fall back to position
            var target = e.target;
            var group = null;
            var _n = target;
            while (_n) {
                if (_n.getAttr && _n.getAttr('clipId')) { group = _n; break; }
                _n = _n.parent;
            }

            // Position-based fallback if Konva target didn't resolve
            var clipId = null, trackId = null;
            if (group) {
                clipId = group.getAttr('clipId');
                trackId = group.getAttr('trackId');
            } else if (pos.y > RULER_HEIGHT && pos.x >= TRACK_HEADER_WIDTH) {
                var mdTrackIdx = trackIndexAtY(pos.y);
                var mdFrame = pixelToFrame(pos.x);
                if (mdTrackIdx >= 0 && mdTrackIdx < project.tracks.length) {
                    var mdTrack = project.tracks[mdTrackIdx];
                    for (var mci = 0; mci < mdTrack.clips.length; mci++) {
                        if (mdFrame >= mdTrack.clips[mci].startFrame && mdFrame < mdTrack.clips[mci].endFrame) {
                            clipId = mdTrack.clips[mci].id;
                            trackId = mdTrack.id;
                            break;
                        }
                    }
                }
            }

            if (clipId) {
                var targetName = target.name ? target.name() : '';

                // Trim handle check (only works with Konva target)
                if (targetName === 'trimLeft' || targetName === 'trimRight') {
                    startTrim(clipId, trackId, targetName === 'trimLeft' ? 'left' : 'right', pos.x);
                    return;
                }

                // Selection
                if (e.evt.ctrlKey || e.evt.metaKey) {
                    if (selectedClipIds.has(clipId)) {
                        selectedClipIds.delete(clipId);
                    } else {
                        selectedClipIds.add(clipId);
                    }
                } else {
                    if (!selectedClipIds.has(clipId)) {
                        selectedClipIds.clear();
                        selectedClipIds.add(clipId);
                    }
                }

                renderTimeline();

                // Start drag
                startDrag(clipId, pos.x, pos.y);
                return;
            }

            // Click on empty timeline area — deselect
            if (pos.y > RULER_HEIGHT && pos.x >= TRACK_HEADER_WIDTH) {
                if (!e.evt.ctrlKey && !e.evt.metaKey) {
                    selectedClipIds.clear();
                    renderTimeline();
                }
            }
        });

        stage.on('mousemove', function (e) {
            var pos = stage.getPointerPosition();
            if (!pos) return;

            if (isScrubbing) {
                scrubToPosition(pos.x);
                return;
            }

            if (isDragging) {
                handleDragMove(pos.x, pos.y);
                return;
            }

            if (isTrimming) {
                handleTrimMove(pos.x);
                return;
            }

            // Cursor: check if hovering trim handles
            var target = e.target;
            var targetName = (target && target.name) ? target.name() : '';
            if (targetName === 'trimLeft' || targetName === 'trimRight') {
                stage.container().style.cursor = 'col-resize';
            } else {
                stage.container().style.cursor = 'default';
            }
        });

        stage.on('mouseup', function () {
            if (isScrubbing) {
                isScrubbing = false;
                return;
            }
            if (isDragging) {
                endDrag();
                return;
            }
            if (isTrimming) {
                endTrim();
                return;
            }
        });

        stage.on('mouseleave', function () {
            if (isScrubbing) isScrubbing = false;
            if (isDragging) endDrag();
            if (isTrimming) endTrim();
            stage.container().style.cursor = 'default';
        });

        // Double-click: add placeholder clip
        stage.on('dblclick', function (e) {
            // Cancel any drag that started from the first click
            if (isDragging) { isDragging = false; dragOriginals = {}; dragUndoSnapshot = null; }
            var pos = stage.getPointerPosition();
            if (!pos || pos.y <= RULER_HEIGHT || pos.x < TRACK_HEADER_WIDTH) return;

            // Use selected clip from first click if available (most reliable)
            if (selectedClipIds.size === 1) {
                var selId = selectedClipIds.values().next().value;
                var selInfo = findClipById(selId);
                if (selInfo && selInfo.track && selInfo.track.type === 'text') {
                    openSubtitleEditor(selId);
                    return;
                }
                // Non-text clip was double-clicked — do nothing extra
                return;
            }

            // No clip selected — add placeholder at position
            var dblTrackIdx = trackIndexAtY(pos.y);
            var dblFrame = pixelToFrame(pos.x);
            addPlaceholderClip(project.tracks[dblTrackIdx].id, Math.max(0, dblFrame));
        });

        // Right-click context menu
        stage.on('contextmenu', function (e) {
            e.evt.preventDefault();
            var pos = stage.getPointerPosition();
            if (!pos) return;

            // Find clip at click position (position-based — reliable regardless of Konva listening flags)
            var clickedClipId = null;
            var clickedTrackId = null;
            if (pos.y > RULER_HEIGHT && pos.x >= TRACK_HEADER_WIDTH) {
                var ctxTrackIdx = trackIndexAtY(pos.y);
                var ctxFrame = pixelToFrame(pos.x);
                if (ctxTrackIdx >= 0 && ctxTrackIdx < project.tracks.length) {
                    var ctxTrack = project.tracks[ctxTrackIdx];
                    for (var ci = 0; ci < ctxTrack.clips.length; ci++) {
                        var cc = ctxTrack.clips[ci];
                        if (ctxFrame >= cc.startFrame && ctxFrame < cc.endFrame) {
                            clickedClipId = cc.id;
                            clickedTrackId = ctxTrack.id;
                            break;
                        }
                    }
                }
            }

            // Calculate menu position relative to panel
            var containerRect = stage.container().getBoundingClientRect();
            var panelRect = document.getElementById('panel-video-edit').getBoundingClientRect();
            var menuX = containerRect.left - panelRect.left + pos.x;
            var menuY = containerRect.top - panelRect.top + pos.y;

            if (clickedClipId) {
                var clipId = clickedClipId;
                var trackId = clickedTrackId;

                // Select if not already
                if (!selectedClipIds.has(clipId)) {
                    selectedClipIds.clear();
                    selectedClipIds.add(clipId);
                    renderTimeline();
                }

                contextTargetClipId = clipId;
                contextTargetTrackId = trackId;

                var clipInfo = findClipById(clipId);
                var clipHasTakes = clipInfo && clipInfo.clip.takes && clipInfo.clip.takes.length > 1;

                var menuItems = [
                    { label: 'Split at Playhead', action: 'split' },
                    { label: 'Duplicate', action: 'duplicate' },
                    { label: 'Retake Selection...', action: 'retake' },
                ];
                if (clipHasTakes) {
                    menuItems.push({ label: 'Switch Take (' + clipInfo.clip.takes.length + ')', action: 'switch-take' });
                }
                menuItems.push({ separator: true });
                menuItems.push({ label: 'Delete', action: 'delete' });
                // RIFE interpolation (video clips only)
                if (clipInfo && clipInfo.track && clipInfo.track.type === 'video' && clipInfo.clip.source_path) {
                    menuItems.push({ separator: true });
                    menuItems.push({ label: 'Interpolate 2\u00D7 (RIFE)', action: 'rife-2x' });
                    menuItems.push({ label: 'Interpolate 4\u00D7 (RIFE)', action: 'rife-4x' });
                    menuItems.push({ label: 'Restore Faces...', action: 'face-restore' });
                    menuItems.push({ label: 'Upscale (Real-ESRGAN)...', action: 'esrgan-upscale' });
                    menuItems.push({ label: 'Deflicker...', action: 'deflicker' });
                    menuItems.push({ label: 'Enhance Audio...', action: 'audio-enhance' });
                }
                menuItems.push({ separator: true });
                menuItems.push({ label: 'Properties...', action: 'properties' });

                showContextMenu(menuX, menuY, menuItems);
            } else if (pos.y > RULER_HEIGHT) {
                var trackIdx = trackIndexAtY(pos.y);
                contextTargetTrackId = project.tracks[trackIdx] ? project.tracks[trackIdx].id : null;
                contextClickFrame = pixelToFrame(pos.x);

                var gapItems = [
                    { label: 'Add Clip Here', action: 'add-clip' },
                ];

                // Check for gap with adjacent clips
                var gap = findGapAtFrame(contextTargetTrackId, contextClickFrame);
                if (gap && (gap.before || gap.after)) {
                    gapItems.push({ label: 'Fill Gap with AI...', action: 'fill-gap' });
                }

                gapItems.push({ label: 'Add Track', action: 'add-track' });
                showContextMenu(menuX, menuY, gapItems);
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', handleKeydown);

        // Close context menu on outside click
        document.addEventListener('click', function (e) {
            if (contextMenuEl && !contextMenuEl.contains(e.target)) {
                hideContextMenu();
            }
        });
    }

    // ===== Drag =====

    var dragUndoSnapshot = null;

    function startDrag(clipId, mouseX, mouseY) {
        isDragging = true;
        dragClipId = clipId;
        dragStartX = mouseX;
        dragStartY = mouseY;
        dragOriginals = {};
        dragTrackOffset = 0;

        // Capture undo snapshot BEFORE any mutations
        dragUndoSnapshot = JSON.parse(JSON.stringify(project.tracks));

        collectSnapPoints();

        selectedClipIds.forEach(function (id) {
            var info = findClipById(id);
            if (info) {
                dragOriginals[id] = {
                    startFrame: info.clip.startFrame,
                    endFrame: info.clip.endFrame,
                    trackId: info.track.id,
                    trackIndex: info.trackIndex
                };
            }
        });
    }

    function handleDragMove(mouseX, mouseY) {
        if (!isDragging) return;

        var frameDelta = Math.round((mouseX - dragStartX) / pixelsPerFrame);

        // Snap: check both start and end edges, pick closer
        var primary = dragOriginals[dragClipId];
        if (primary) {
            activeSnapFrame = null;
            var newStart = primary.startFrame + frameDelta;
            var newEnd = primary.endFrame + frameDelta;
            var snappedStart = findSnap(newStart);
            var snappedEnd = findSnap(newEnd);

            var startDist = snappedStart !== null ? Math.abs(newStart - snappedStart) : Infinity;
            var endDist = snappedEnd !== null ? Math.abs(newEnd - snappedEnd) : Infinity;

            if (startDist <= endDist && snappedStart !== null) {
                frameDelta = snappedStart - primary.startFrame;
                activeSnapFrame = snappedStart;
            } else if (snappedEnd !== null) {
                frameDelta = snappedEnd - primary.endFrame;
                activeSnapFrame = snappedEnd;
            }
        }

        // Clamp: no clip goes below frame 0
        selectedClipIds.forEach(function (id) {
            var orig = dragOriginals[id];
            if (orig && orig.startFrame + frameDelta < 0) {
                frameDelta = -orig.startFrame;
            }
        });

        // Vertical track movement
        var trackDeltaY = mouseY - dragStartY;
        var newTrackOffset = 0;
        if (Math.abs(trackDeltaY) > TRACK_HEIGHT / 2) {
            newTrackOffset = Math.round(trackDeltaY / TRACK_HEIGHT);
        }

        // Apply with overlap prevention
        selectedClipIds.forEach(function (id) {
            var orig = dragOriginals[id];
            if (!orig) return;
            var info = findClipById(id);
            if (!info) return;

            var duration = orig.endFrame - orig.startFrame;
            var newStart = Math.max(0, orig.startFrame + frameDelta);
            var newEnd = newStart + duration;

            // Track move
            if (newTrackOffset !== dragTrackOffset) {
                var newTrackIdx = clamp(orig.trackIndex + newTrackOffset, 0, project.tracks.length - 1);
                var destTrack = project.tracks[newTrackIdx];
                if (destTrack && destTrack.id !== info.track.id) {
                    var curIdx = info.track.clips.indexOf(info.clip);
                    if (curIdx >= 0) info.track.clips.splice(curIdx, 1);
                    destTrack.clips.push(info.clip);
                }
            }

            // Overlap prevention: check against other clips on same track
            var currentTrack = findClipById(id);
            if (currentTrack) {
                var track = currentTrack.track;
                for (var oi = 0; oi < track.clips.length; oi++) {
                    var other = track.clips[oi];
                    if (other.id === id || selectedClipIds.has(other.id)) continue;
                    // Check overlap
                    if (newStart < other.endFrame && newEnd > other.startFrame) {
                        // Collision — push to nearest edge
                        var pushLeft = other.startFrame - duration;
                        var pushRight = other.endFrame;
                        if (Math.abs(pushLeft - newStart) < Math.abs(pushRight - newStart)) {
                            newStart = Math.max(0, pushLeft);
                        } else {
                            newStart = pushRight;
                        }
                        newEnd = newStart + duration;
                    }
                }
            }

            info.clip.startFrame = newStart;
            info.clip.endFrame = newEnd;
        });

        dragTrackOffset = newTrackOffset;
        renderTracks();
        renderPlayhead();
    }

    function endDrag() {
        if (!isDragging) return;
        isDragging = false;
        activeSnapFrame = null;
        stage.container().style.cursor = 'default';

        // Check if anything actually moved
        var moved = false;
        selectedClipIds.forEach(function (id) {
            var orig = dragOriginals[id];
            var info = findClipById(id);
            if (orig && info) {
                if (info.clip.startFrame !== orig.startFrame || info.track.id !== orig.trackId) {
                    moved = true;
                }
            }
        });

        if (moved && dragUndoSnapshot) {
            // Push the pre-drag snapshot, not the current (already mutated) state
            undoStack.push(dragUndoSnapshot);
            if (undoStack.length > MAX_UNDO) undoStack.shift();
            redoStack.length = 0;
            recalcTotalFrames();
            scheduleAutosave();
        }

        dragUndoSnapshot = null;
        dragOriginals = {};
        renderTimeline();
    }

    // ===== Trim =====

    var trimUndoSnapshot = null;

    function startTrim(clipId, trackId, edge, mouseX) {
        var info = findClipById(clipId);
        if (!info) return;

        trimUndoSnapshot = JSON.parse(JSON.stringify(project.tracks));
        isTrimming = true;
        trimClipId = clipId;
        trimTrackId = trackId;
        trimEdge = edge;
        trimOrigStart = info.clip.startFrame;
        trimOrigEnd = info.clip.endFrame;
        trimStartMouseX = mouseX;

        // Select the clip being trimmed
        if (!selectedClipIds.has(clipId)) {
            selectedClipIds.clear();
            selectedClipIds.add(clipId);
        }

        collectSnapPoints();
        stage.container().style.cursor = 'col-resize';
    }

    function handleTrimMove(mouseX) {
        if (!isTrimming) return;
        var info = findClipById(trimClipId);
        if (!info) return;

        var frameDelta = Math.round((mouseX - trimStartMouseX) / pixelsPerFrame);
        activeSnapFrame = null;

        if (trimEdge === 'left') {
            var newStart = trimOrigStart + frameDelta;

            // Snap
            var snapped = findSnap(newStart);
            if (snapped !== null) {
                newStart = snapped;
                activeSnapFrame = snapped;
            }

            // Constrain: min 1 frame width, can't go negative
            newStart = clamp(newStart, 0, trimOrigEnd - 1);
            info.clip.startFrame = newStart;
        } else {
            var newEnd = trimOrigEnd + frameDelta;

            // Snap
            var snapped = findSnap(newEnd);
            if (snapped !== null) {
                newEnd = snapped;
                activeSnapFrame = snapped;
            }

            // Constrain: min 1 frame width
            newEnd = Math.max(trimOrigStart + 1, newEnd);
            info.clip.endFrame = newEnd;
        }

        renderTracks();
        renderPlayhead();
    }

    function endTrim() {
        if (!isTrimming) return;
        isTrimming = false;
        activeSnapFrame = null;
        stage.container().style.cursor = 'default';

        // Check if trim actually changed anything
        var info = findClipById(trimClipId);
        if (info) {
            if (info.clip.startFrame !== trimOrigStart || info.clip.endFrame !== trimOrigEnd) {
                if (trimUndoSnapshot) {
                    undoStack.push(trimUndoSnapshot);
                    if (undoStack.length > MAX_UNDO) undoStack.shift();
                    redoStack.length = 0;
                }
                recalcTotalFrames();
                scheduleAutosave();
            }
        }

        trimUndoSnapshot = null;
        renderTimeline();
    }

    // ===== Scrub =====

    function scrubToPosition(x) {
        if (x < TRACK_HEADER_WIDTH) x = TRACK_HEADER_WIDTH;
        currentFrame = Math.round((x - TRACK_HEADER_WIDTH + scrollOffsetX) / pixelsPerFrame);
        currentFrame = clamp(currentFrame, 0, totalFrames);
        updateTimecodeDisplay();
        renderPlayhead();
        updatePreview();
    }

    // ===== Keyboard =====

    function handleKeydown(e) {
        var panel = document.getElementById('panel-video-edit');
        if (!panel || panel.offsetParent === null) return;

        var tag = e.target.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || e.target.isContentEditable) return;

        // Ctrl+Z / Ctrl+Shift+Z / Ctrl+Y — Undo/Redo
        if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
            e.preventDefault();
            undo();
            return;
        }
        if ((e.ctrlKey || e.metaKey) && (e.key === 'Z' || e.key === 'y')) {
            e.preventDefault();
            redo();
            return;
        }

        // Ctrl+A — Select all
        if ((e.ctrlKey || e.metaKey) && e.key === 'a') {
            e.preventDefault();
            selectAllClips();
            return;
        }

        // Delete / Backspace
        if (e.key === 'Delete' || e.key === 'Backspace') {
            e.preventDefault();
            deleteSelectedClips();
            return;
        }

        // Escape — deselect / close context menu
        if (e.key === 'Escape') {
            hideContextMenu();
            if (propsPanelEl) { closePropertiesPanel(); return; }
            if (retakeMode) exitRetakeMode();
            if (bridgePanelEl) bridgePanelEl.style.display = 'none';
            if (takeDropdownEl) takeDropdownEl.style.display = 'none';
            closeSubtitleEditor();
            selectedClipIds.clear();
            renderTimeline();
            return;
        }

        // Space — play/pause
        if (e.key === ' ') {
            e.preventDefault();
            togglePlayback();
            return;
        }

        // Home / End
        if (e.key === 'Home') {
            e.preventDefault();
            stopPlayback();
            currentFrame = 0;
            scrollOffsetX = 0;
            updateTimecodeDisplay();
            renderTimeline();
            updatePreview();
            return;
        }
        if (e.key === 'End') {
            e.preventDefault();
            stopPlayback();
            currentFrame = totalFrames;
            var viewWidth = stage ? stage.width() - TRACK_HEADER_WIDTH : 500;
            scrollOffsetX = Math.max(0, currentFrame * pixelsPerFrame - viewWidth + 60);
            updateTimecodeDisplay();
            renderTimeline();
            updatePreview();
            return;
        }
    }

    // ===== Resize =====

    function resize() {
        if (!stage) return;
        var container = document.getElementById('ve-timeline-canvas');
        if (!container) return;
        var rect = container.getBoundingClientRect();
        if (rect.width === 0) return;
        stage.width(rect.width);
        var stageHeight = RULER_HEIGHT + (project.tracks.length * TRACK_HEIGHT) + 20;
        stage.height(Math.max(stageHeight, rect.height));
        resizePreview();
        renderTimeline();
    }

    // ===== Init =====

    function init() {
        if (_initialized) return;
        _initialized = true;

        var panel = document.getElementById('panel-video-edit');
        if (!panel) return;

        panel.innerHTML =
            '<div id="ve-preview"></div>' +
            '<div id="ve-toolbar"></div>' +
            '<div id="ve-timeline-container">' +
                '<div id="ve-timeline-canvas"></div>' +
            '</div>';

        initPreview();
        buildToolbar();
        buildContextMenu();
        initKonva();
        renderTimeline();
        updateTimecodeDisplay();
        updateZoomDisplay();
        updatePreview();
        bindEvents();

        // Load or create backend project
        loadOrCreateProject();
    }

    // ===== Public API =====

    function loadProject(pid) {
        if (!pid) return;
        projectId = pid;
        localStorage.setItem('ve-project-id', pid);
        fetch(getApiBase() + '/video_edit/projects/' + pid)
            .then(function (r) { return r.ok ? r.json() : null; })
            .then(function (data) {
                if (!data) return;
                project.name = data.name || 'Untitled Project';
                project.fps = data.fps || 30;
                if (data.tracks && data.tracks.length > 0) {
                    project.tracks = data.tracks;
                }
                recalcTotalFrames();
                renderTimeline();
            })
            .catch(function () {});
    }

    function addClipFromExternal(sourcePath, label, durationFrames) {
        if (!projectId) return;

        // Find first video track or create one
        var track = null;
        for (var i = 0; i < project.tracks.length; i++) {
            if (project.tracks[i].type === 'video') { track = project.tracks[i]; break; }
        }
        if (!track) {
            track = { id: 'track-' + Date.now(), name: 'Video 1', type: 'video', clips: [] };
            project.tracks.unshift(track);
        }

        // Place at end of content
        var endFrame = 0;
        track.clips.forEach(function (c) {
            if (c.endFrame > endFrame) endFrame = c.endFrame;
        });

        pushUndo();
        var dur = durationFrames || (FPS * 5);
        var newClip = {
            id: generateClipId(),
            startFrame: endFrame,
            endFrame: endFrame + dur,
            label: label || 'Generated',
            color: '#4a7dff',
            source_path: sourcePath,
        };
        track.clips.push(newClip);
        recalcTotalFrames();
        renderTimeline();
        scheduleAutosave();

        // Scroll to show new clip
        var clipEndPx = newClip.endFrame * pixelsPerFrame;
        var viewWidth = stage ? stage.width() - TRACK_HEADER_WIDTH : 500;
        if (clipEndPx > scrollOffsetX + viewWidth) {
            scrollOffsetX = Math.max(0, clipEndPx - viewWidth + 100);
            renderTimeline();
        }
    }

    return {
        init: function () {
            init();
            this._initialized = true;
        },
        resize: resize,
        _initialized: false,
        getActiveProjectId: function () { return projectId; },
        loadProject: loadProject,
        addClipFromExternal: addClipFromExternal,
    };
})();
