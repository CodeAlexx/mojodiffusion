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
    var TRACK_HEIGHT = 62;
    var RULER_HEIGHT = 28;
    var TRACK_HEADER_WIDTH = 156;
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
    var snapEnabled = true;
    var clipClipboard = [];
    var monitorMode = 'program';
    var sourceMonitorClipId = null;
    var dockTab = 'properties';

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
    var genesisPreviewInFlight = false;
    var genesisPreviewPending = false;
    var genesisPreviewSequence = 0;

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
    var exportPollTimer = null;
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
        width: 1280,
        height: 720,
        name: 'Untitled Project',
        markers: [],
        tracks: [
            {
                id: 'track-1', name: 'Video 1', type: 'video',
                clips: []
            },
            {
                id: 'track-2', name: 'Audio', type: 'audio',
                clips: []
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
        updateEditButton();
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
        updateEditButton();
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
        if (!snapEnabled) return null;
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
            markers: project.markers || [],
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

    function applyLoadedProject(data) {
        project.name = data.name || 'Untitled Project';
        project.fps = clamp(Math.round(Number(data.fps) || 30), 1, 120);
        project.width = clamp(Math.round(Number(data.width) || 1280), 16, 8192);
        project.height = clamp(Math.round(Number(data.height) || 720), 16, 8192);
        project.markers = Array.isArray(data.markers) ? data.markers : [];
        FPS = project.fps;
        if (Array.isArray(data.tracks)) {
            project.tracks = data.tracks;
        }
        thumbnailCache.clear();
        thumbnailLoading.clear();
        waveformCache.clear();
        waveformLoading.clear();
        stopAllAudio();
        audioElements.clear();
        previewActiveClipId = null;
        sourceMonitorClipId = null;
        monitorMode = 'program';
        selectedClipIds.clear();
        currentFrame = 0;
        scrollOffsetX = 0;
        recalcTotalFrames();
        currentFrame = clamp(currentFrame, 0, totalFrames);
        var initialClip = findActiveClipAtFrame(currentFrame);
        if (!initialClip) {
            for (var trackIndex = 0; trackIndex < project.tracks.length && !initialClip; trackIndex++) {
                if (project.tracks[trackIndex].clips.length) {
                    initialClip = project.tracks[trackIndex].clips[0];
                }
            }
        }
        if (initialClip) {
            selectedClipIds.add(initialClip.id);
            sourceMonitorClipId = initialClip.id;
        }
        dockTab = 'properties';
        updateTimecodeDisplay();
        renderTimeline();
        updatePreview();
        updateEditButton();
        renderMediaBin();
        renderDock();
        updateMonitorTabs();
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
                    applyLoadedProject(data);
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
                width: project.width,
                height: project.height,
                markers: project.markers,
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
                '<span class="ve-editor-title">Genesis Editor</span>' +
                '<div style="position:relative">' +
                    '<button id="ve-btn-file" class="ve-tb-btn ve-tb-btn-label" title="Project and interchange actions">File</button>' +
                    '<div id="ve-file-dropdown" class="ve-file-dropdown" style="display:none"></div>' +
                '</div>' +
                '<button id="ve-btn-import-video" class="ve-tb-btn ve-tb-btn-label" title="Import video into the media bin and timeline">+ Video</button>' +
                '<button id="ve-btn-import-music" class="ve-tb-btn ve-tb-btn-label" title="Import music or audio">+ Music</button>' +
                '<button id="ve-btn-save" class="ve-tb-btn ve-tb-btn-label" title="Save project now">Save</button>' +
                '<span class="ve-toolbar-divider"></span>' +
                '<button id="ve-btn-undo-top" class="ve-tb-btn" title="Undo (Ctrl+Z)">&#8630;</button>' +
                '<button id="ve-btn-redo-top" class="ve-tb-btn" title="Redo (Ctrl+Shift+Z)">&#8631;</button>' +
                '<span class="ve-spacer"></span>' +
                '<span id="ve-engine-status" class="ve-engine-status">Genesis: checking...</span>' +
                '<button id="ve-btn-retake" class="ve-tb-btn ve-tb-btn-label" title="Generate a replacement for the selected region">Retake</button>' +
                '<button id="ve-btn-export" class="ve-tb-btn ve-tb-btn-label ve-primary-action" title="Render and export">Render</button>' +
            '</div>';

        var transport = document.getElementById('ve-monitor-transport');
        if (transport) {
            transport.innerHTML =
                '<button id="ve-btn-start" class="ve-transport-btn" title="Go to start (Home)">&#9198;</button>' +
                '<button id="ve-btn-rewind" class="ve-transport-btn" title="Back one second">&#9664;&#9664;</button>' +
                '<button id="ve-btn-play" class="ve-transport-btn ve-play-btn" title="Play/Pause (Space)">&#9654;</button>' +
                '<button id="ve-btn-stop" class="ve-transport-btn" title="Stop">&#9632;</button>' +
                '<button id="ve-btn-forward" class="ve-transport-btn" title="Forward one second">&#9654;&#9654;</button>' +
                '<button id="ve-btn-end" class="ve-transport-btn" title="Go to end (End)">&#9197;</button>' +
                '<span id="ve-timecode" class="ve-monitor-timecode">00:00:00</span>' +
                '<span class="ve-timecode-divider">/</span>' +
                '<span id="ve-duration" class="ve-monitor-timecode">' + frameToTimecode(totalFrames) + '</span>';
        }

        var editToolbar = document.getElementById('ve-edit-toolbar');
        if (editToolbar) {
            editToolbar.innerHTML =
                '<span class="ve-edit-toolbar-title">TIMELINE</span>' +
                '<button id="ve-edit-split" class="ve-edit-btn" title="Split selected clip at playhead">Split</button>' +
                '<button id="ve-edit-razor" class="ve-edit-btn" title="Split every clip crossing the playhead">Razor all</button>' +
                '<button id="ve-edit-lift" class="ve-edit-btn" title="Delete selection and leave the gap">Lift</button>' +
                '<button id="ve-edit-ripple" class="ve-edit-btn" title="Delete selection and close the gap">Ripple</button>' +
                '<button id="ve-edit-copy" class="ve-edit-btn" title="Copy selection">Copy</button>' +
                '<button id="ve-edit-paste" class="ve-edit-btn" title="Paste at playhead">Paste</button>' +
                '<button id="ve-edit-duplicate" class="ve-edit-btn" title="Duplicate selection">Duplicate</button>' +
                '<span class="ve-toolbar-divider"></span>' +
                '<button id="ve-edit-marker" class="ve-edit-btn" title="Toggle marker at playhead">Marker</button>' +
                '<button id="ve-edit-snap" class="ve-edit-btn ve-active" title="Toggle snapping">Snap</button>' +
                '<button id="ve-edit-add-video" class="ve-edit-btn" title="Add video track">+V</button>' +
                '<button id="ve-edit-add-audio" class="ve-edit-btn" title="Add audio track">+A</button>' +
                '<span class="ve-spacer"></span>' +
                '<button id="ve-edit-zoom-fit" class="ve-edit-btn" title="Fit the complete project in the timeline">Fit</button>' +
                '<span id="ve-zoom-label" class="ve-zoom-label">100%</span>' +
                '<input id="ve-zoom-slider" type="range" min="5" max="2000" value="400" class="ve-zoom-slider">';
        }

        var btnPlay = document.getElementById('ve-btn-play');
        var btnStop = document.getElementById('ve-btn-stop');
        var btnStart = document.getElementById('ve-btn-start');
        var btnEnd = document.getElementById('ve-btn-end');
        var zoomSlider = document.getElementById('ve-zoom-slider');
        var btnImportVideo = document.getElementById('ve-btn-import-video');
        var btnImportMusic = document.getElementById('ve-btn-import-music');

        if (btnImportVideo) btnImportVideo.addEventListener('click', function () {
            openMediaPicker('video');
        });
        if (btnImportMusic) btnImportMusic.addEventListener('click', function () {
            openMediaPicker('audio');
        });
        document.getElementById('ve-btn-save').addEventListener('click', function () {
            saveProject();
            setGenesisStatus('Project saved', false);
        });
        document.getElementById('ve-btn-undo-top').addEventListener('click', undo);
        document.getElementById('ve-btn-redo-top').addEventListener('click', redo);

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
        document.getElementById('ve-btn-rewind').addEventListener('click', function () {
            seekTimelineFrame(currentFrame - FPS);
        });
        document.getElementById('ve-btn-forward').addEventListener('click', function () {
            seekTimelineFrame(currentFrame + FPS);
        });
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

        document.getElementById('ve-edit-split').addEventListener('click', splitSelectedAtPlayhead);
        document.getElementById('ve-edit-razor').addEventListener('click', razorAllAtPlayhead);
        document.getElementById('ve-edit-lift').addEventListener('click', deleteSelectedClips);
        document.getElementById('ve-edit-ripple').addEventListener('click', rippleDeleteSelectedClips);
        document.getElementById('ve-edit-copy').addEventListener('click', copySelectedClips);
        document.getElementById('ve-edit-paste').addEventListener('click', pasteClipsAtPlayhead);
        document.getElementById('ve-edit-duplicate').addEventListener('click', duplicateSelectedClips);
        document.getElementById('ve-edit-marker').addEventListener('click', toggleMarkerAtPlayhead);
        document.getElementById('ve-edit-snap').addEventListener('click', function () {
            snapEnabled = !snapEnabled;
            this.classList.toggle('ve-active', snapEnabled);
            renderTimeline();
        });
        document.getElementById('ve-edit-add-video').addEventListener('click', function () {
            addTrack('video');
        });
        document.getElementById('ve-edit-add-audio').addEventListener('click', function () {
            addTrack('audio');
        });
        document.getElementById('ve-edit-zoom-fit').addEventListener('click', zoomTimelineToFit);

        document.querySelectorAll('.ve-monitor-tab').forEach(function (tab) {
            tab.addEventListener('click', function () {
                setMonitorMode(tab.dataset.monitor);
            });
        });
        document.querySelectorAll('.ve-dock-tab').forEach(function (tab) {
            tab.addEventListener('click', function () {
                dockTab = tab.dataset.dock;
                renderDock();
            });
        });

        fetch(getApiBase() + '/video_edit/status')
            .then(function (r) { return r.ok ? r.json() : Promise.reject(new Error('status ' + r.status)); })
            .then(function (data) {
                var status = document.getElementById('ve-engine-status');
                if (!status) return;
                status.textContent = data.worker_ready
                    ? 'Genesis · Rust/C · Ready'
                    : 'Genesis worker missing';
                status.classList.toggle('ve-engine-error', !data.worker_ready);
            })
            .catch(function () {
                var status = document.getElementById('ve-engine-status');
                if (status) {
                    status.textContent = 'Genesis unavailable';
                    status.classList.add('ve-engine-error');
                }
            });
    }

    function seekTimelineFrame(frame) {
        stopPlayback();
        currentFrame = clamp(Math.round(frame), 0, totalFrames);
        updateTimecodeDisplay();
        renderPlayhead();
        updatePreview();
    }

    function splitSelectedAtPlayhead() {
        Array.from(selectedClipIds).forEach(function (clipId) {
            var info = findClipById(clipId);
            if (info) splitClipAtPlayhead(clipId, info.track.id);
        });
    }

    function razorAllAtPlayhead() {
        var targets = [];
        project.tracks.forEach(function (track) {
            track.clips.forEach(function (clip) {
                if (currentFrame > clip.startFrame && currentFrame < clip.endFrame) {
                    targets.push({ clipId: clip.id, trackId: track.id });
                }
            });
        });
        targets.forEach(function (target) {
            splitClipAtPlayhead(target.clipId, target.trackId);
        });
    }

    function rippleDeleteSelectedClips() {
        if (!selectedClipIds.size) return;
        pushUndo();
        project.tracks.forEach(function (track) {
            var removed = track.clips
                .filter(function (clip) { return selectedClipIds.has(clip.id); })
                .map(function (clip) {
                    return {
                        start: clip.startFrame,
                        end: clip.endFrame,
                        duration: clip.endFrame - clip.startFrame,
                    };
                })
                .sort(function (a, b) { return a.start - b.start; });
            if (!removed.length) return;
            track.clips = track.clips.filter(function (clip) {
                return !selectedClipIds.has(clip.id);
            });
            track.clips.forEach(function (clip) {
                var shift = removed.reduce(function (sum, gap) {
                    return sum + (gap.end <= clip.startFrame ? gap.duration : 0);
                }, 0);
                clip.startFrame -= shift;
                clip.endFrame -= shift;
            });
        });
        selectedClipIds.clear();
        closePropertiesPanel();
        recalcTotalFrames();
        renderTimeline();
        renderMediaBin();
        renderDock();
        updateEditButton();
        scheduleAutosave();
        updatePreview();
    }

    function copySelectedClips() {
        if (!selectedClipIds.size) return;
        var selected = [];
        project.tracks.forEach(function (track) {
            track.clips.forEach(function (clip) {
                if (selectedClipIds.has(clip.id)) {
                    selected.push({
                        trackType: track.type,
                        clip: JSON.parse(JSON.stringify(clip)),
                    });
                }
            });
        });
        var firstFrame = selected.reduce(function (min, item) {
            return Math.min(min, item.clip.startFrame);
        }, Infinity);
        clipClipboard = selected.map(function (item) {
            item.offset = item.clip.startFrame - firstFrame;
            return item;
        });
        setGenesisStatus(clipClipboard.length + ' clip(s) copied', false);
    }

    function pasteClipsAtPlayhead() {
        if (!clipClipboard.length) return;
        pushUndo();
        selectedClipIds.clear();
        clipClipboard.forEach(function (item) {
            var track = project.tracks.find(function (candidate) {
                return candidate.type === item.trackType;
            });
            if (!track) {
                track = addTrack(item.trackType, true);
            }
            var pasted = JSON.parse(JSON.stringify(item.clip));
            var duration = pasted.endFrame - pasted.startFrame;
            pasted.id = generateClipId();
            pasted.startFrame = currentFrame + item.offset;
            pasted.endFrame = pasted.startFrame + duration;
            pasted.label = (pasted.label || 'Clip') + ' copy';
            track.clips.push(pasted);
            selectedClipIds.add(pasted.id);
        });
        recalcTotalFrames();
        renderTimeline();
        renderMediaBin();
        renderDock();
        updateEditButton();
        scheduleAutosave();
        updatePreview();
    }

    function duplicateSelectedClips() {
        Array.from(selectedClipIds).forEach(function (clipId) {
            var info = findClipById(clipId);
            if (info) duplicateClip(clipId, info.track.id);
        });
    }

    function toggleMarkerAtPlayhead() {
        if (!project.markers) project.markers = [];
        var index = project.markers.indexOf(currentFrame);
        if (index >= 0) project.markers.splice(index, 1);
        else project.markers.push(currentFrame);
        project.markers.sort(function (a, b) { return a - b; });
        renderTimeline();
        scheduleAutosave();
    }

    function zoomTimelineToFit() {
        if (!stage) return;
        pixelsPerFrame = clamp(
            (stage.width() - TRACK_HEADER_WIDTH - 20) / Math.max(1, totalFrames),
            MIN_PPF,
            MAX_PPF
        );
        scrollOffsetX = 0;
        updateZoomDisplay();
        renderTimeline();
    }

    function selectedVideoClip() {
        if (selectedClipIds.size === 1) {
            var info = findClipById(selectedClipIds.values().next().value);
            if (info && info.track.type === 'video') return info.clip;
        }
        return null;
    }

    function setMonitorMode(mode) {
        if (mode === 'source') {
            var selected = selectedVideoClip();
            if (selected) sourceMonitorClipId = selected.id;
            if (!sourceMonitorClipId) {
                var active = findActiveClipAtFrame(currentFrame);
                if (active) sourceMonitorClipId = active.id;
            }
            if (!sourceMonitorClipId) return;
        }
        monitorMode = mode === 'source' ? 'source' : 'program';
        previewActiveClipId = null;
        updateMonitorTabs();
        updatePreview();
    }

    function updateMonitorTabs() {
        document.querySelectorAll('.ve-monitor-tab').forEach(function (tab) {
            tab.classList.toggle('ve-active', tab.dataset.monitor === monitorMode);
        });
    }

    function renderMediaBin() {
        var grid = document.getElementById('ve-media-grid');
        var count = document.getElementById('ve-media-count');
        if (!grid) return;
        grid.innerHTML = '';
        var media = [];
        var seen = new Set();
        project.tracks.forEach(function (track) {
            track.clips.forEach(function (clip) {
                if (!clip.source_path || seen.has(clip.source_path)) return;
                seen.add(clip.source_path);
                media.push({ clip: clip, type: track.type });
            });
        });
        if (count) count.textContent = String(media.length);
        if (!media.length) {
            grid.innerHTML =
                '<div class="ve-media-empty">No media yet<br><button id="ve-media-empty-add">+ Add media</button></div>';
            var emptyAdd = document.getElementById('ve-media-empty-add');
            if (emptyAdd) emptyAdd.addEventListener('click', function () {
                openMediaPicker('video');
            });
            return;
        }
        media.forEach(function (item) {
            var card = document.createElement('div');
            card.className = 've-media-card';
            card.dataset.clipId = item.clip.id;
            if (sourceMonitorClipId === item.clip.id) card.classList.add('ve-selected');

            var thumb = document.createElement('div');
            thumb.className = 've-media-thumb';
            var cached = thumbnailCache.get(item.clip.id);
            if (cached && cached.img) {
                thumb.style.backgroundImage = 'url("' + cached.img.src + '")';
                thumb.style.backgroundSize = 'auto 100%';
                thumb.style.backgroundPosition = 'left center';
            } else if (item.type === 'audio') {
                thumb.classList.add('ve-audio-thumb');
                thumb.textContent = '\u266B';
            } else {
                thumb.textContent = 'VIDEO';
                if (!thumbnailLoading.has(item.clip.id)) loadThumbnails(item.clip);
            }

            var label = document.createElement('div');
            label.className = 've-media-label';
            label.textContent = item.clip.label || 'Untitled';
            var meta = document.createElement('div');
            meta.className = 've-media-meta';
            meta.textContent = item.type.toUpperCase() + ' \u00B7 ' +
                frameToTimecode(item.clip.endFrame - item.clip.startFrame);
            var add = document.createElement('button');
            add.className = 've-media-add-timeline';
            add.textContent = '+ Timeline';
            add.title = 'Add this media at the playhead';
            add.addEventListener('click', function (event) {
                event.stopPropagation();
                addMediaToTimeline(item.clip, item.type);
            });
            card.appendChild(thumb);
            card.appendChild(label);
            card.appendChild(meta);
            card.appendChild(add);
            card.addEventListener('click', function () {
                sourceMonitorClipId = item.clip.id;
                selectedClipIds.clear();
                selectedClipIds.add(item.clip.id);
                currentFrame = item.clip.startFrame;
                renderTimeline();
                updateEditButton();
                renderDock();
                setMonitorMode('source');
                renderMediaBin();
            });
            grid.appendChild(card);
        });
    }

    function addMediaToTimeline(sourceClip, trackType) {
        var track = project.tracks.find(function (candidate) {
            return candidate.type === trackType;
        });
        if (!track) track = addTrack(trackType, true);
        pushUndo();
        var clone = JSON.parse(JSON.stringify(sourceClip));
        var duration = clone.endFrame - clone.startFrame;
        clone.id = generateClipId();
        clone.startFrame = currentFrame;
        clone.endFrame = currentFrame + duration;
        clone.label = (clone.label || 'Clip') + ' copy';
        track.clips.push(clone);
        selectedClipIds.clear();
        selectedClipIds.add(clone.id);
        recalcTotalFrames();
        renderTimeline();
        updateEditButton();
        scheduleAutosave();
        renderMediaBin();
        renderDock();
    }

    function renderDock() {
        var content = document.getElementById('ve-dock-content');
        if (!content) return;
        document.querySelectorAll('.ve-dock-tab').forEach(function (tab) {
            tab.classList.toggle('ve-active', tab.dataset.dock === dockTab);
        });
        propsPanelEl = null;
        propsPanelClipId = null;
        content.innerHTML = '';
        var info = selectedClipIds.size === 1
            ? findClipById(selectedClipIds.values().next().value)
            : null;

        if (dockTab === 'filters') {
            if (!info || info.track.type !== 'video') {
                content.innerHTML = '<div class="ve-dock-empty">Select a video clip to manage its filter stack.</div>';
                return;
            }
            renderFilterCatalog(content, info.clip);
            return;
        }

        if (dockTab === 'audio') {
            var audioTracks = project.tracks.filter(function (track) {
                return track.type === 'audio' || track.type === 'video';
            });
            var heading = document.createElement('div');
            heading.className = 've-dock-heading';
            heading.textContent = 'TRACK MIXER';
            content.appendChild(heading);
            audioTracks.forEach(function (track) {
                var strip = document.createElement('div');
                strip.className = 've-audio-strip';
                var name = document.createElement('strong');
                name.textContent = track.name;
                var mute = document.createElement('button');
                mute.textContent = track.muted ? 'Muted' : 'Mute';
                mute.className = track.muted ? 've-active' : '';
                mute.addEventListener('click', function () {
                    track.muted = !track.muted;
                    renderDock();
                    renderTimeline();
                    scheduleAutosave();
                });
                var clips = document.createElement('span');
                clips.textContent = track.clips.length + ' clip(s)';
                strip.appendChild(name);
                strip.appendChild(clips);
                strip.appendChild(mute);
                content.appendChild(strip);
            });
            return;
        }

        if (dockTab === 'scopes') {
            content.innerHTML =
                '<div class="ve-dock-heading">LIVE HISTOGRAM</div>' +
                '<canvas id="ve-scope-histogram" width="288" height="150"></canvas>' +
                '<div class="ve-scope-note">Updates from the current program frame.</div>';
            updateScopes();
            return;
        }

        if (!info) {
            renderProjectProperties(content);
            return;
        }
        var clip = info.clip;
        if (info.track.type === 'video') {
            if (!clip.effects) clip.effects = [];
            var panel = document.createElement('div');
            panel.className = 've-props-panel';
            propsPanelEl = panel;
            propsPanelClipId = clip.id;
            content.appendChild(panel);
            renderPropertiesPanelContent(panel, clip);

            var actions = document.createElement('div');
            actions.className = 've-dock-actions';
            [
                ['Retake', function () { enterRetakeMode(clip.id); }],
                ['Restore Faces', function () { showFaceRestoreDialog(clip.id); }],
                ['Upscale', function () { showEsrganDialog(clip.id); }],
                ['Deflicker', function () { showDeflickerDialog(clip.id); }],
                ['Enhance Audio', function () { showAudioEnhanceDialog(clip.id); }],
            ].forEach(function (action) {
                var button = document.createElement('button');
                button.textContent = action[0];
                button.addEventListener('click', action[1]);
                actions.appendChild(button);
            });
            panel.appendChild(actions);
            return;
        }

        var heading = document.createElement('div');
        heading.className = 've-dock-heading';
        heading.textContent = 'PROPERTIES';
        content.appendChild(heading);
        var details = document.createElement('div');
        details.className = 've-property-summary';
        details.innerHTML =
            '<strong></strong>' +
            '<span>' + info.track.type.toUpperCase() + '</span>' +
            '<span>' + frameToTimecode(clip.endFrame - clip.startFrame) + '</span>' +
            '<span>' + (clip.source_fps || FPS) + ' FPS</span>';
        details.querySelector('strong').textContent = clip.label || 'Untitled';
        content.appendChild(details);
    }

    function renderProjectProperties(content) {
        var panel = document.createElement('div');
        panel.className = 've-props-panel ve-project-properties';
        var heading = document.createElement('h3');
        heading.textContent = 'Project Properties';
        panel.appendChild(heading);

        function addRow(label, input) {
            var row = document.createElement('div');
            row.className = 've-props-row';
            var caption = document.createElement('label');
            caption.textContent = label;
            row.appendChild(caption);
            row.appendChild(input);
            panel.appendChild(row);
        }

        var name = document.createElement('input');
        name.type = 'text';
        name.value = project.name || 'Untitled Project';
        name.addEventListener('change', function () {
            project.name = name.value.trim() || 'Untitled Project';
            scheduleAutosave();
        });
        addRow('Name', name);

        var width = document.createElement('input');
        width.type = 'number';
        width.min = '16';
        width.max = '8192';
        width.value = project.width;
        width.addEventListener('change', function () {
            project.width = clamp(Math.round(Number(width.value) || project.width), 16, 8192);
            width.value = project.width;
            resizePreview();
            updatePreview();
            scheduleAutosave();
        });
        addRow('Width', width);

        var height = document.createElement('input');
        height.type = 'number';
        height.min = '16';
        height.max = '8192';
        height.value = project.height;
        height.addEventListener('change', function () {
            project.height = clamp(Math.round(Number(height.value) || project.height), 16, 8192);
            height.value = project.height;
            resizePreview();
            updatePreview();
            scheduleAutosave();
        });
        addRow('Height', height);

        var fps = document.createElement('input');
        fps.type = 'text';
        fps.value = FPS + ' FPS';
        fps.readOnly = true;
        addRow('Timeline', fps);

        var summary = document.createElement('div');
        summary.className = 've-project-summary';
        var clipCount = project.tracks.reduce(function (total, track) {
            return total + track.clips.length;
        }, 0);
        summary.innerHTML =
            '<span><strong>' + project.tracks.length + '</strong> tracks</span>' +
            '<span><strong>' + clipCount + '</strong> clips</span>' +
            '<span><strong>' + frameToTimecode(totalFrames) + '</strong> duration</span>';
        panel.appendChild(summary);

        var note = document.createElement('div');
        note.className = 've-project-id';
        note.textContent = projectId ? 'Project ' + projectId : 'Unsaved project';
        panel.appendChild(note);

        var actions = document.createElement('div');
        actions.className = 've-dock-actions';
        [
            ['+ Video', function () { openMediaPicker('video'); }],
            ['+ Music', function () { openMediaPicker('audio'); }],
            ['Render', showExportDialog],
        ].forEach(function (action) {
            var button = document.createElement('button');
            button.textContent = action[0];
            button.addEventListener('click', action[1]);
            actions.appendChild(button);
        });
        panel.appendChild(actions);
        content.appendChild(panel);
    }

    function addFilterFromCatalog(clip, type) {
        if (!clip.effects) clip.effects = [];
        var reg = EFFECT_REGISTRY[type];
        if (!reg) return;
        var existing = clip.effects.find(function (effect) {
            return effect.type === type;
        });
        if (!existing) {
            pushUndo();
            clip.effects.push({
                type: type,
                enabled: true,
                params: JSON.parse(JSON.stringify(reg.defaults || {})),
            });
            renderTimeline();
            scheduleAutosave();
            updatePreviewDebounced();
        }
        dockTab = 'properties';
        renderDock();
    }

    function renderFilterCatalog(content, clip) {
        if (!clip.effects) clip.effects = [];
        var wrapper = document.createElement('div');
        wrapper.className = 've-filter-catalog';
        wrapper.innerHTML =
            '<div class="ve-dock-heading">FILTER STACK</div>' +
            '<div class="ve-filter-count"></div>' +
            '<input class="ve-filter-search" type="search" placeholder="Search filters">' +
            '<div class="ve-filter-applied"></div>' +
            '<div class="ve-dock-heading">ADD FILTER</div>' +
            '<div class="ve-filter-options"></div>';
        content.appendChild(wrapper);

        var count = wrapper.querySelector('.ve-filter-count');
        count.textContent = clip.effects.length + ' active filter(s)';
        var applied = wrapper.querySelector('.ve-filter-applied');
        if (!clip.effects.length) {
            applied.innerHTML = '<div class="ve-filter-empty">No filters applied.</div>';
        } else {
            clip.effects.forEach(function (effect, index) {
                var row = document.createElement('div');
                row.className = 've-filter-row';
                var enabled = document.createElement('input');
                enabled.type = 'checkbox';
                enabled.checked = effect.enabled !== false;
                enabled.addEventListener('change', function () {
                    pushUndo();
                    effect.enabled = enabled.checked;
                    renderTimeline();
                    scheduleAutosave();
                    updatePreviewDebounced();
                });
                var name = document.createElement('button');
                name.textContent = EFFECT_REGISTRY[effect.type]
                    ? EFFECT_REGISTRY[effect.type].name
                    : effect.type;
                name.addEventListener('click', function () {
                    dockTab = 'properties';
                    renderDock();
                });
                var remove = document.createElement('button');
                remove.textContent = '\u00D7';
                remove.title = 'Remove filter';
                remove.addEventListener('click', function () {
                    pushUndo();
                    clip.effects.splice(index, 1);
                    renderFilterCatalog(content, clip);
                    wrapper.remove();
                    renderTimeline();
                    scheduleAutosave();
                    updatePreviewDebounced();
                });
                row.appendChild(enabled);
                row.appendChild(name);
                row.appendChild(remove);
                applied.appendChild(row);
            });
        }

        var options = wrapper.querySelector('.ve-filter-options');
        function populateOptions(query) {
            options.innerHTML = '';
            Object.keys(EFFECT_REGISTRY).forEach(function (type) {
                var reg = EFFECT_REGISTRY[type];
                if (query && reg.name.toLowerCase().indexOf(query) < 0) return;
                var button = document.createElement('button');
                button.textContent = '+ ' + reg.name;
                button.disabled = clip.effects.some(function (effect) {
                    return effect.type === type;
                });
                button.addEventListener('click', function () {
                    addFilterFromCatalog(clip, type);
                });
                options.appendChild(button);
            });
        }
        populateOptions('');
        wrapper.querySelector('.ve-filter-search').addEventListener('input', function () {
            populateOptions(this.value.trim().toLowerCase());
        });
    }

    function updateScopes() {
        if (dockTab !== 'scopes' || !previewCanvas) return;
        var scope = document.getElementById('ve-scope-histogram');
        if (!scope) return;
        var ctx = scope.getContext('2d');
        var source;
        try {
            source = previewCtx.getImageData(0, 0, previewCanvas.width, previewCanvas.height);
        } catch (_) {
            return;
        }
        var bins = new Uint32Array(64);
        var pixels = source.data;
        for (var i = 0; i < pixels.length; i += 16) {
            var luma = Math.round(0.2126 * pixels[i] + 0.7152 * pixels[i + 1] + 0.0722 * pixels[i + 2]);
            bins[Math.min(63, Math.floor(luma / 4))]++;
        }
        var max = Math.max.apply(null, Array.from(bins)) || 1;
        ctx.clearRect(0, 0, scope.width, scope.height);
        ctx.fillStyle = '#101017';
        ctx.fillRect(0, 0, scope.width, scope.height);
        var gradient = ctx.createLinearGradient(0, scope.height, 0, 0);
        gradient.addColorStop(0, '#5551dd');
        gradient.addColorStop(1, '#62d8ff');
        ctx.fillStyle = gradient;
        bins.forEach(function (value, index) {
            var h = value / max * (scope.height - 16);
            ctx.fillRect(index * (scope.width / 64), scope.height - h, scope.width / 64 - 1, h);
        });
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
        renderMediaBin();
        renderDock();
        updateEditButton();
        scheduleAutosave();
        updatePreview();
    }

    function splitClipAtPlayhead(clipId, trackId) {
        if (!clipId) return;
        var info = findClipById(clipId);
        if (!info) return;
        var clip = info.clip;
        if (currentFrame <= clip.startFrame || currentFrame >= clip.endFrame) return;

        pushUndo();
        var newClip = JSON.parse(JSON.stringify(clip));
        newClip.id = generateClipId();
        newClip.startFrame = currentFrame;
        newClip.endFrame = clip.endFrame;
        newClip.label = clip.label + ' (R)';
        newClip.transition_in = null;
        if (clip.source_path) {
            var timelineOffset = currentFrame - clip.startFrame;
            newClip.source_start = (clip.source_start || 0) +
                Math.round(timelineOffset * (clip.source_fps || FPS) / FPS);
        }
        clip.endFrame = currentFrame;
        clip.label = clip.label.replace(/ \([LR]\)$/, '') + ' (L)';
        info.track.clips.splice(info.clipIndex + 1, 0, newClip);
        renderTimeline();
        renderMediaBin();
        renderDock();
        scheduleAutosave();
    }

    function duplicateClip(clipId, trackId) {
        var info = findClipById(clipId);
        if (!info) return;
        pushUndo();
        var clip = info.clip;
        var duration = clip.endFrame - clip.startFrame;
        var newClip = JSON.parse(JSON.stringify(clip));
        newClip.id = generateClipId();
        newClip.startFrame = clip.endFrame;
        newClip.endFrame = clip.endFrame + duration;
        newClip.label = clip.label + ' copy';
        info.track.clips.push(newClip);
        recalcTotalFrames();
        renderTimeline();
        renderMediaBin();
        scheduleAutosave();
    }

    function addTrack(type, skipUndo) {
        type = type === 'audio' ? 'audio' : 'video';
        if (!skipUndo) pushUndo();
        var num = project.tracks.filter(function (t) { return t.type === type; }).length + 1;
        var track = {
            id: 'track-' + Date.now(),
            name: type === 'audio' ? 'Audio ' + num : 'Video ' + num,
            type: type,
            clips: []
        };
        project.tracks.push(track);
        recalcTotalFrames();
        resize();
        renderDock();
        if (!skipUndo) scheduleAutosave();
        return track;
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
        updateEditButton();
    }

    function updateEditButton() {
        var button = document.getElementById('ve-btn-edit-clip');
        var editable = false;
        if (selectedClipIds.size === 1) {
            var info = findClipById(selectedClipIds.values().next().value);
            editable = !!(info && info.track.type === 'video' && info.clip.source_path);
        }
        if (button) button.disabled = !editable;
        ['ve-edit-split', 've-edit-lift', 've-edit-ripple', 've-edit-copy', 've-edit-duplicate']
            .forEach(function (id) {
                var control = document.getElementById(id);
                if (control) control.disabled = selectedClipIds.size === 0;
            });
        renderDock();
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
            '<div id="ve-preview-placeholder" class="ve-preview-empty">' +
                '<strong>No video at the playhead</strong>' +
                '<span>Import a video, then select it to edit effects.</span>' +
                '<div><button id="ve-empty-import-video">Import Video</button>' +
                '<button id="ve-empty-import-music">Add Music</button></div>' +
            '</div>';

        previewVideo = document.getElementById('ve-preview-video');
        previewCanvas = document.getElementById('ve-preview-canvas');
        previewCtx = previewCanvas ? previewCanvas.getContext('2d') : null;
        previewTcEl = document.getElementById('ve-preview-tc');
        document.getElementById('ve-empty-import-video').addEventListener('click', function () {
            openMediaPicker('video');
        });
        document.getElementById('ve-empty-import-music').addEventListener('click', function () {
            openMediaPicker('audio');
        });

        // Size canvas to the current project aspect within the preview container.
        resizePreview();
    }

    function resizePreview() {
        if (!previewCanvas) return;
        var container = document.getElementById('ve-preview');
        if (!container) return;
        var rect = container.getBoundingClientRect();
        var aspect = (project.width > 0 && project.height > 0)
            ? project.width / project.height
            : 16 / 9;
        var targetW = rect.width;
        var targetH = rect.width / aspect;
        if (targetH > rect.height) {
            targetH = rect.height;
            targetW = rect.height * aspect;
        }
        previewCanvas.width = Math.round(targetW);
        previewCanvas.height = Math.round(targetH);
    }

    function findActiveClipAtFrame(frame) {
        // Topmost video clip at frame (track 0 = top of timeline = highest priority)
        for (var i = 0; i < project.tracks.length; i++) {
            var track = project.tracks[i];
            if (track.type !== 'video' || track.hidden) continue;
            for (var j = 0; j < track.clips.length; j++) {
                var clip = track.clips[j];
                if (frame >= clip.startFrame && frame < clip.endFrame) {
                    return clip;
                }
            }
        }
        return null;
    }

    function previewMediaUrl(sourcePath) {
        if (sourcePath.indexOf('/') === 0 || sourcePath.indexOf('\\') >= 0) {
            return getApiBase() + '/video_edit/media/' +
                encodeURIComponent(sourcePath.split('/video_projects/').pop() || sourcePath);
        }
        return getApiBase() + '/video_edit/media/' + sourcePath;
    }

    function drawPreviewVideoFrame() {
        if (!previewVideo || !previewCtx || !previewCanvas || previewVideo.readyState < 2) {
            return;
        }
        previewCtx.fillStyle = '#111';
        previewCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
        var vw = previewVideo.videoWidth || previewCanvas.width;
        var vh = previewVideo.videoHeight || previewCanvas.height;
        var scale = Math.min(previewCanvas.width / vw, previewCanvas.height / vh);
        var dw = vw * scale;
        var dh = vh * scale;
        var dx = (previewCanvas.width - dw) / 2;
        var dy = (previewCanvas.height - dh) / 2;
        previewCtx.drawImage(previewVideo, dx, dy, dw, dh);
        renderSubtitleOverlay();
        updateScopes();
    }

    function loadPreviewVideo(clip, timeInSource) {
        if (!previewVideo) return;
        var clipChanged = previewActiveClipId !== clip.id;
        if (!clipChanged) return;

        previewVideo.pause();
        previewVideo.src = previewMediaUrl(clip.source_path);
        previewActiveClipId = clip.id;
        previewVideo.load();
        previewVideo.onloadedmetadata = function () {
            if (previewActiveClipId !== clip.id) return;
            var duration = Number.isFinite(previewVideo.duration)
                ? previewVideo.duration
                : timeInSource;
            previewVideo.currentTime = clamp(timeInSource, 0, Math.max(0, duration - 0.001));
            if (isPlaying) {
                previewVideo.play().catch(function () {});
            }
        };
        previewVideo.onloadeddata = function () {
            if (previewActiveClipId === clip.id) drawPreviewVideoFrame();
        };
    }

    function updatePreview() {
        if (!previewCtx || !previewCanvas) return;
        var placeholder = document.getElementById('ve-preview-placeholder');

        var clip = monitorMode === 'source' && sourceMonitorClipId
            ? (findClipById(sourceMonitorClipId) || {}).clip
            : findActiveClipAtFrame(currentFrame);

        if (!clip) {
            // Invalidate a compositor reply that was requested for an older playhead.
            genesisPreviewSequence++;
            genesisPreviewPending = false;
            if (previewVideo) previewVideo.pause();
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
            genesisPreviewSequence++;
            genesisPreviewPending = false;
            if (previewVideo) previewVideo.pause();
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
        var sourceFps = clip.source_fps || FPS;
        var sourceTimelineFrame = monitorMode === 'source'
            ? clamp(currentFrame, clip.startFrame, clip.endFrame - 1)
            : currentFrame;
        var timeInSource = sourceOffset / sourceFps +
            (sourceTimelineFrame - clip.startFrame) / FPS;

        loadPreviewVideo(clip, timeInSource);

        if (isPlaying) {
            // Playback must follow the browser's native media clock. Seeking on every
            // animation frame and queueing a compositor render made the large preview
            // lag far behind the timeline. Genesis still renders the exact effected
            // frame whenever playback is paused or the playhead is scrubbed.
            genesisPreviewSequence++;
            genesisPreviewPending = false;
            if (previewVideo && previewVideo.readyState >= 1) {
                var drift = Math.abs(previewVideo.currentTime - timeInSource);
                if (drift > 0.25) {
                    previewVideo.currentTime = timeInSource;
                }
                if (previewVideo.paused) {
                    previewVideo.play().catch(function () {});
                }
                drawPreviewVideoFrame();
            }
            if (previewTcEl) previewTcEl.textContent = frameToTimecode(currentFrame);
            renderSubtitleOverlay();
            setGenesisStatus('Playback · live', false);
            return;
        }

        if (previewVideo) {
            previewVideo.pause();
        }
        if (previewVideo && previewVideo.readyState >= 1) {
            previewVideo.onseeked = function () {
                drawPreviewVideoFrame();
            };
            if (Math.abs(previewVideo.currentTime - timeInSource) > (0.5 / sourceFps)) {
                previewVideo.currentTime = timeInSource;
            } else {
                drawPreviewVideoFrame();
            }
        }

        // Update timecode overlay
        if (previewTcEl) {
            previewTcEl.textContent = frameToTimecode(currentFrame);
        }

        // Subtitle overlay
        renderSubtitleOverlay();

        if (monitorMode === 'source') {
            genesisPreviewSequence++;
            genesisPreviewPending = false;
            setGenesisStatus('Source monitor', false);
            return;
        }

        // The browser video element above is an immediate fallback. Genesis replaces it
        // with the actual composited/effected timeline frame when the worker replies.
        requestGenesisPreview();
    }

    function setGenesisStatus(text, isError) {
        var status = document.getElementById('ve-engine-status');
        if (!status) return;
        status.textContent = text;
        status.classList.toggle('ve-engine-error', !!isError);
    }

    function requestGenesisPreview() {
        genesisPreviewSequence++;
        if (genesisPreviewInFlight) {
            genesisPreviewPending = true;
            return;
        }

        var sequence = genesisPreviewSequence;
        var frame = currentFrame;
        genesisPreviewInFlight = true;
        setGenesisStatus('Genesis · rendering frame ' + frame, false);

        fetch(getApiBase() + '/video_edit/preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project_id: projectId,
                project: project,
                frame: frame,
            }),
        })
        .then(function (r) {
            if (!r.ok) {
                return r.text().then(function (body) {
                    throw new Error(body || ('preview status ' + r.status));
                });
            }
            return r.blob();
        })
        .then(function (blob) {
            return createImageBitmap(blob);
        })
        .then(function (bitmap) {
            if (sequence === genesisPreviewSequence && previewCtx && previewCanvas) {
                previewCtx.fillStyle = '#111';
                previewCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
                var scale = Math.min(
                    previewCanvas.width / bitmap.width,
                    previewCanvas.height / bitmap.height
                );
                var dw = bitmap.width * scale;
                var dh = bitmap.height * scale;
                previewCtx.drawImage(
                    bitmap,
                    (previewCanvas.width - dw) / 2,
                    (previewCanvas.height - dh) / 2,
                    dw,
                    dh
                );
                renderSubtitleOverlay();
                updateScopes();
            }
            bitmap.close();
            setGenesisStatus('Genesis · Rust/C · Ready', false);
        })
        .catch(function (err) {
            console.warn('Genesis preview failed:', err);
            setGenesisStatus('Genesis preview failed', true);
        })
        .finally(function () {
            genesisPreviewInFlight = false;
            if (genesisPreviewPending) {
                genesisPreviewPending = false;
                requestGenesisPreview();
            }
        });
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
                fps: clip.source_fps || FPS,
            }),
        })
        .then(function (r) {
            if (r.ok) return r.json();
            return r.text().then(function (body) {
                throw new Error(body || ('thumbnail status ' + r.status));
            });
        })
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
                renderMediaBin();
            };
        })
        .catch(function (err) {
            thumbnailLoading.delete(clip.id);
            console.warn('Thumbnail load failed for ' + (clip.label || clip.id) + ':', err);
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
                fps: clip.source_fps || FPS,
            }),
        })
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function (data) {
            waveformLoading.delete(clip.id);
            if (!data || !data.peaks) return;
            waveformCache.set(clip.id, data);
            renderTracks();
            if (dockTab === 'audio') renderDock();
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

                var shouldPlay = isPlaying && !track.muted &&
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
                    var sourceTime = (currentFrame - clip.startFrame) / FPS +
                        (clip.source_start || 0) / (clip.source_fps || FPS);
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
            '<div class="ve-ctx-item" data-action="import-video">Import Video...</div>' +
            '<div class="ve-ctx-item" data-action="import-audio">Add Music/Audio...</div>' +
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
    var audioFileInput = null;

    function openMediaPicker(kind) {
        var isAudio = kind === 'audio';
        var input = isAudio ? audioFileInput : videoFileInput;
        if (!input) {
            input = document.createElement('input');
            input.type = 'file';
            input.accept = isAudio ? 'audio/*' : 'video/*';
            input.multiple = true;
            input.style.display = 'none';
            input.dataset.mediaKind = kind;
            document.body.appendChild(input);
            input.addEventListener('change', function () {
                var files = Array.from(input.files || []);
                files.forEach(importMediaFile);
                input.value = '';
            });
            if (isAudio) audioFileInput = input;
            else videoFileInput = input;
        }
        input.click();
    }

    function handleFileAction(action) {
        if (action === 'import-video') {
            openMediaPicker('video');
        } else if (action === 'import-audio') {
            openMediaPicker('audio');
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
        form.append('project_fps', String(FPS));
        form.append('file', file);
        fetch(getApiBase() + '/video_edit/projects/' + projectId + '/import_clip', {
            method: 'POST',
            body: form,
        })
        .then(function (r) {
            return r.json().then(function (body) {
                if (!r.ok) throw new Error(body.error || ('import status ' + r.status));
                return body;
            });
        })
        .then(function (data) {
            if (data.error) {
                alert('Import failed: ' + data.error);
                return;
            }
            // Determine track type from file
            var isAudio = data.media_type === 'audio' ||
                (file.type && file.type.startsWith('audio'));
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
            var sourceFps = clamp(Math.round(Number(data.source_fps) || FPS), 1, 120);
            var sourceFrames = Math.max(1, Math.round(Number(data.duration_frames) || sourceFps));
            var isFirstMedia = endFrame === 0 && !project.tracks.some(function (candidateTrack) {
                return candidateTrack.clips.some(function (candidateClip) {
                    return !!candidateClip.source_path;
                });
            });
            if (isFirstMedia && !isAudio) {
                FPS = sourceFps;
                project.fps = sourceFps;
                if (Number(data.width) > 0 && Number(data.height) > 0) {
                    project.width = Math.round(Number(data.width));
                    project.height = Math.round(Number(data.height));
                    resizePreview();
                }
            }
            var dur = Math.max(1, Math.round(sourceFrames * FPS / sourceFps));
            var newClip = {
                id: data.clip_id || generateClipId(),
                startFrame: endFrame,
                endFrame: endFrame + dur,
                label: file.name.replace(/\.[^.]+$/, '').slice(0, 30),
                color: trackColor,
                source_path: data.source_path,
                source_fps: sourceFps,
                source_frames: sourceFrames,
                duration_seconds: Number(data.duration_seconds) || dur / FPS,
                has_audio: !!data.has_audio,
                media_type: isAudio ? 'audio' : 'video',
            };
            track.clips.push(newClip);
            selectedClipIds.clear();
            selectedClipIds.add(newClip.id);
            currentFrame = newClip.startFrame;
            recalcTotalFrames();
            renderTimeline();
            renderMediaBin();
            scheduleAutosave();
            updatePreview();
            updateEditButton();
            if (!isAudio) openPropertiesPanel(newClip.id);

            // Keep the selected clip's beginning and the playhead on screen.
            scrollOffsetX = Math.max(0, endFrame * pixelsPerFrame - 40);
            renderTimeline();
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
                    '<input type="number" id="ve-exp-fps" value="' + FPS + '" min="1" max="120" readonly title="Genesis exports at the project timeline FPS" style="width:60px"></div>' +
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
        if (exportPollTimer) {
            clearTimeout(exportPollTimer);
            exportPollTimer = null;
        }
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
            pollExportStatus();
        })
        .catch(function (err) {
            onExportError({ data: { error: String(err) } });
        });
    }

    function pollExportStatus() {
        if (!exportId || !isExporting) return;
        var expectedId = exportId;
        fetch(getApiBase() + '/video_edit/export/' + encodeURIComponent(expectedId))
            .then(function (r) {
                if (!r.ok) throw new Error('export status ' + r.status);
                return r.json();
            })
            .then(function (data) {
                if (exportId !== expectedId) return;
                if (data.state === 'complete') {
                    onExportComplete({ data: data });
                    return;
                }
                if (data.state === 'failed') {
                    onExportError({ data: data });
                    return;
                }
                var text = document.getElementById('ve-exp-progress-text');
                if (text) text.textContent = 'Genesis is encoding...';
                exportPollTimer = setTimeout(pollExportStatus, 750);
            })
            .catch(function (err) {
                if (exportId !== expectedId || !isExporting) return;
                var text = document.getElementById('ve-exp-progress-text');
                if (text) text.textContent = 'Waiting for Genesis: ' + String(err);
                exportPollTimer = setTimeout(pollExportStatus, 1250);
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
        if (exportPollTimer) {
            clearTimeout(exportPollTimer);
            exportPollTimer = null;
        }
        var text = document.getElementById('ve-exp-progress-text');
        if (text) text.textContent = 'Export complete: ' + (d.output_path || d.output_url || '');
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
        if (exportPollTimer) {
            clearTimeout(exportPollTimer);
            exportPollTimer = null;
        }
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

        (project.markers || []).forEach(function (markerFrame) {
            var markerX = TRACK_HEADER_WIDTH + (markerFrame * pixelsPerFrame) - scrollOffsetX;
            if (markerX < TRACK_HEADER_WIDTH || markerX > stage.width()) return;
            rulerLayer.add(new Konva.RegularPolygon({
                x: markerX,
                y: 6,
                sides: 3,
                radius: 6,
                fill: '#ffd166',
                rotation: 180,
            }));
        });

        rulerLayer.batchDraw();
    }

    // ===== V6: Effects & Properties Panel =====

    function openPropertiesPanel(clipId) {
        var info = findClipById(clipId);
        if (!info) return;

        // Don't open for text/subtitle or audio-only clips
        if (info.track.type === 'text' || info.track.type === 'audio') return;

        selectedClipIds.clear();
        selectedClipIds.add(clipId);
        dockTab = 'properties';
        renderTimeline();
        renderMediaBin();
        renderDock();
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
        if (dockTab === 'properties') {
            dockTab = 'filters';
            renderDock();
        }
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
                renderTimeline();
                scheduleAutosave();
                updatePreviewDebounced();
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
                updatePreviewDebounced();
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
                    updatePreviewDebounced();
                };
                slider.onchange = function () {
                    pushUndo();
                    scheduleAutosave();
                    updatePreviewDebounced();
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
            updatePreviewDebounced();
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
            var seekSec = (clip.source_start || 0) / (clip.source_fps || FPS);
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
                    updatePreviewDebounced();
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
                    x: 10, y: y + 8,
                    text: track.name,
                    fontSize: 12, fill: '#ccc', fontFamily: 'sans-serif'
                }));
                [
                    {
                        x: 10,
                        text: track.hidden ? '\u25CB' : '\u25C9',
                        title: 'visibility',
                        toggle: function () { track.hidden = !track.hidden; },
                    },
                    {
                        x: 42,
                        text: track.muted ? 'M\u00D7' : 'M',
                        title: 'mute',
                        toggle: function () { track.muted = !track.muted; },
                    },
                    {
                        x: 76,
                        text: track.locked ? '\u25A0' : '\u25A1',
                        title: 'lock',
                        toggle: function () { track.locked = !track.locked; },
                    },
                ].forEach(function (control) {
                    var button = new Konva.Text({
                        x: control.x,
                        y: y + 34,
                        width: 26,
                        height: 18,
                        align: 'center',
                        text: control.text,
                        fontSize: 11,
                        fill: control.title === 'mute' && track.muted ? '#ff8a8a' : '#9b9bab',
                        fontFamily: 'sans-serif',
                        name: 'trackControl',
                    });
                    button.on('mousedown', function (event) {
                        event.cancelBubble = true;
                    });
                    button.on('click', function (event) {
                        event.cancelBubble = true;
                        pushUndo();
                        control.toggle();
                        renderTimeline();
                        renderDock();
                        scheduleAutosave();
                        updatePreview();
                    });
                    timelineLayer.add(button);
                });

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
                        opacity: track.hidden ? 0.34 : 1,
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
                        var sourceStartSec = (clip.source_start || 0) / (clip.source_fps || FPS);
                        var thumbSlotW = Math.max(24, FPS * pixelsPerFrame);
                        while (thumbX < clipW && sec < tc.frameCount) {
                            var spriteIdx = Math.min(Math.floor(sourceStartSec + sec), tc.frameCount - 1);
                            group.add(new Konva.Image({
                                x: thumbX - rectOffset, y: 0,
                                width: Math.min(thumbSlotW, clipW - thumbX), height: clipH,
                                image: tc.img,
                                crop: { x: spriteIdx * tc.thumbW, y: 0, width: tc.thumbW, height: tc.thumbH },
                                opacity: 0.72,
                                listening: false,
                            }));
                            thumbX += thumbSlotW;
                            sec++;
                        }
                    }

                    // Render waveform on audio clips (or video clips with audio)
                    if (hasWaveform && (track.type === 'audio' || track.type === 'video')) {
                        var wd = waveformCache.get(clip.id);
                        var midY = clipH / 2;
                        if (!wd.displayMax) {
                            wd.displayMax = Math.max(
                                0.01,
                                wd.peaks.reduce(function (maxValue, value) {
                                    return Math.max(maxValue, Math.abs(value || 0));
                                }, 0)
                            );
                        }
                        var peaksPerSecond = wd.duration_seconds > 0
                            ? wd.peaks.length / wd.duration_seconds
                            : 30;
                        var sourceStartSample = Math.floor(
                            (clip.source_start || 0) / (clip.source_fps || FPS) * peaksPerSecond
                        );
                        var clipDurFrames = clip.endFrame - clip.startFrame;
                        var samplesInClip = Math.floor(clipDurFrames / FPS * peaksPerSecond);
                        var wfPoints = [];
                        // Draw every 2px for performance
                        var step = Math.max(2, Math.floor(clipW / 300));
                        for (var px = 0; px < clipW; px += step) {
                            var sIdx = sourceStartSample + Math.floor(px / clipW * samplesInClip);
                            if (sIdx >= wd.peaks.length) break;
                            var amp = Math.min(1, Math.abs(wd.peaks[sIdx] || 0) / wd.displayMax);
                            var barH = amp * (clipH * 0.8) / 2;
                            // Top line
                            group.add(new Konva.Line({
                                points: [px - rectOffset, midY - barH, px - rectOffset, midY + barH],
                                stroke: track.type === 'audio'
                                    ? 'rgba(216,255,242,0.9)'
                                    : 'rgba(255,255,255,0.55)',
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
        if (previewVideo) previewVideo.pause();
        stopAllAudio();
        updatePlayButton();
    }

    function togglePlayback() {
        if (isPlaying) {
            stopPlayback();
            updatePreview();
        } else {
            startPlayback();
        }
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
                var targetInfo = findClipById(clipId);
                var targetLocked = !!(targetInfo && targetInfo.track.locked);

                // Trim handle check (only works with Konva target)
                if (targetName === 'trimLeft' || targetName === 'trimRight') {
                    if (targetLocked) return;
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
                updateEditButton();

                // Start drag
                if (!targetLocked) startDrag(clipId, pos.x, pos.y);
                return;
            }

            // Click on empty timeline area — deselect
            if (pos.y > RULER_HEIGHT && pos.x >= TRACK_HEADER_WIDTH) {
                if (!e.evt.ctrlKey && !e.evt.metaKey) {
                    selectedClipIds.clear();
                    renderTimeline();
                    updateEditButton();
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
                    updateEditButton();
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
            updateEditButton();
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
            updateEditButton();
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
            '<div id="ve-toolbar"></div>' +
            '<div id="ve-workspace">' +
                '<aside id="ve-media-panel">' +
                    '<div class="ve-panel-heading"><span>MEDIA</span><span id="ve-media-count">0</span></div>' +
                    '<button id="ve-media-add" class="ve-wide-action">+ Add media</button>' +
                    '<div class="ve-bin-row"><span>Bin</span><strong>Project Media</strong></div>' +
                    '<div id="ve-media-grid"></div>' +
                '</aside>' +
                '<section id="ve-monitor-shell">' +
                    '<div id="ve-monitor-tabs">' +
                        '<button class="ve-monitor-tab ve-active" data-monitor="program">Program</button>' +
                        '<button class="ve-monitor-tab" data-monitor="source">Source</button>' +
                    '</div>' +
                    '<div id="ve-preview"></div>' +
                    '<div id="ve-monitor-transport"></div>' +
                '</section>' +
                '<aside id="ve-dock">' +
                    '<div id="ve-dock-tabs">' +
                        '<button class="ve-dock-tab ve-active" data-dock="properties">Properties</button>' +
                        '<button class="ve-dock-tab" data-dock="filters">Filters</button>' +
                        '<button class="ve-dock-tab" data-dock="scopes">Scopes</button>' +
                        '<button class="ve-dock-tab" data-dock="audio">Audio</button>' +
                    '</div>' +
                    '<div id="ve-dock-content"></div>' +
                '</aside>' +
            '</div>' +
            '<div id="ve-edit-toolbar"></div>' +
            '<div id="ve-timeline-container">' +
                '<div id="ve-timeline-canvas"></div>' +
            '</div>';

        initPreview();
        buildToolbar();
        document.getElementById('ve-media-add').addEventListener('click', function () {
            openMediaPicker('video');
        });
        buildContextMenu();
        initKonva();
        renderMediaBin();
        renderDock();
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
                applyLoadedProject(data);
            })
            .catch(function () {});
    }

    function addClipFromExternal(sourcePath, label, durationFrames, sourceFps) {
        if (!projectId) return;

        var hasMedia = project.tracks.some(function (candidateTrack) {
            return candidateTrack.clips.some(function (candidateClip) {
                return !!candidateClip.source_path;
            });
        });
        var normalizedSourceFps = clamp(Math.round(Number(sourceFps) || FPS), 1, 120);
        if (!hasMedia) {
            FPS = normalizedSourceFps;
            project.fps = normalizedSourceFps;
        }

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
        var sourceFrames = Math.max(
            1,
            Math.round(Number(durationFrames) || normalizedSourceFps)
        );
        var dur = Math.max(1, Math.round(sourceFrames * FPS / normalizedSourceFps));
        var newClip = {
            id: generateClipId(),
            startFrame: endFrame,
            endFrame: endFrame + dur,
            label: label || 'Generated',
            color: '#4a7dff',
            source_path: sourcePath,
            source_fps: normalizedSourceFps,
        };
        track.clips.push(newClip);
        recalcTotalFrames();
        renderTimeline();
        scheduleAutosave();
        updatePreview();

        selectedClipIds.clear();
        selectedClipIds.add(newClip.id);
        currentFrame = newClip.startFrame;
        scrollOffsetX = Math.max(0, newClip.startFrame * pixelsPerFrame - 40);
        renderTimeline();
        updatePreview();
        updateEditButton();
        openPropertiesPanel(newClip.id);
    }

    return {
        init: function () {
            init();
            this._initialized = true;
        },
        resize: resize,
        _initialized: false,
        getActiveProjectId: function () { return projectId; },
        getDiagnostics: function () {
            return {
                projectId: projectId,
                fps: FPS,
                width: project.width,
                height: project.height,
                currentFrame: currentFrame,
                isPlaying: isPlaying,
                monitorMode: monitorMode,
                sourceMonitorClipId: sourceMonitorClipId,
                dockTab: dockTab,
                markerFrames: (project.markers || []).slice(),
                snapEnabled: snapEnabled,
                previewVideoTime: previewVideo ? previewVideo.currentTime : 0,
                previewVideoPaused: previewVideo ? previewVideo.paused : true,
                thumbnailClipIds: Array.from(thumbnailCache.keys()),
                waveformClipIds: Array.from(waveformCache.keys()),
                mediaCardCount: document.querySelectorAll('.ve-media-card').length,
                dockTabCount: document.querySelectorAll('.ve-dock-tab').length,
                tracks: project.tracks.map(function (track) {
                    return {
                        id: track.id,
                        type: track.type,
                        clips: track.clips.map(function (clip) {
                            return {
                                id: clip.id,
                                label: clip.label,
                                startFrame: clip.startFrame,
                                endFrame: clip.endFrame,
                                source_path: clip.source_path || '',
                                source_fps: clip.source_fps || FPS,
                            };
                        }),
                    };
                }),
            };
        },
        loadProject: loadProject,
        addClipFromExternal: addClipFromExternal,
    };
})();
