"use strict";
/**
 * Canvas Tab — SerenityFlow Phase 4
 * Konva.js infinite canvas with layers, bounding box, brush/eraser, inpaint/outpaint.
 * Completely separate Konva instance from the Workflows graph editor.
 */
var CanvasTab = (function () {
    'use strict';
    // Layer data types and CanvasLayer are defined in layer-types.ts (global).
    // Legacy compat: map old 'raster' type to new 'draw' type.
    function migrateLayerType(t) {
        if (t === 'raster')
            return 'draw';
        if (t === 'ipadapter')
            return 'control'; // IP-Adapter folded into control
        var valid = ['draw', 'mask', 'guidance', 'control', 'adjustment', 'text'];
        return valid.indexOf(t) >= 0 ? t : 'draw';
    }
    var initialized = false;
    var konvaReady = false;
    // Konva objects
    var stage = null;
    var backgroundLayer = null;
    var canvasLayers = [];
    var uiLayer = null;
    var boundingBox = null;
    var sizeLabel = null;
    var resizeHandles = [];
    var brushCursor = null;
    var bgRect = null;
    var layerTransformer = null;
    // State
    var activeTool = 'select';
    var activeLayerId = null;
    var brushSize = 20;
    var brushColor = '#ffffff';
    // Drawing state moved to canvas-tools.ts — these kept for backward compat
    var isDrawing = false;
    var currentLine = null;
    var isPanning = false;
    var isSpaceHeld = false;
    var panStart = { x: 0, y: 0 };
    var stageStart = { x: 0, y: 0 };
    var canvasGenerating = false;
    var pendingCanvasPromptId = '';
    var pendingCanvasMetadata = null;
    var layerIdCounter = 0;
    var activeHandle = null;
    var handleStartBox = null;
    var handleStartMouse = null;
    var brushHardness = 1;
    var drawScheduled = false;
    var historyDebounce = null;
    var bboxAspectLocked = false;
    var bboxLockedRatio = 1;
    // Generation state
    var genState = {
        model: null,
        prompt: '',
        denoise: 0.75,
        steps: 20,
        cfg: 7.0,
        guidance: 3.5,
        seed: -1,
        negative: '',
        sampler: 'euler',
        scheduler: 'simple',
        batch: 1,
        arch: 'sd15',
        frames: 97,
        fps: 24,
        capsPositive: '',
        capsNegative: '',
        noiseFixture: '',
        includeAudio: false,
        editMode: 'create',
        editEngine: 'krea2_raw_1024',
        editSourcePrompt: '',
        editSourceNegative: '',
        editNmax: 24,
        editNmin: 0,
        editSourceCfg: 1.5,
        editAutoMask: true,
        editMaskQ: 0.7,
        editMaskDilate: 1,
        editMaskWarmup: 4,
        styleEntireImage: true,
        lanpaintEngine: 'krea2_turbo_1024',
        lanpaintNumSteps: 5,
        lanpaintLambda: 16,
        lanpaintStepSize: 0.2,
        lanpaintBeta: 1,
        lanpaintFriction: 15,
        lanpaintPromptMode: 'Image First',
        lanpaintBlendOverlap: 9,
        lanpaintEarlyStop: 1,
        lanpaintInnerThreshold: 0,
        lanpaintInnerPatience: 1
    };
    var editSourceFile = null;
    var editSourceDataUrl = '';
    var editSourceIsVideo = false;
    var styleReferenceDataUrl = '';
    var styleReferenceId = '';
    var styleModeInitialized = false;
    var lanpaintModeInitialized = false;
    var STYLE_RESULT_SIZE = 1024;
    // Krea FlowEdit's compiled LT_SHARED bucket is 256 tokens. These word
    // budgets leave room for its chat template and the preservation sentence.
    var STYLE_SOURCE_WORD_LIMIT = 80;
    var STYLE_DESCRIPTION_WORD_LIMIT = 35;
    var STYLE_USER_DIRECTION_WORD_LIMIT = 20;
    // One shared masked-edit workspace serves every product-admitted backend.
    // The registry owns only UI routing and visible presets; server capabilities
    // remain authoritative about which entries are actually offered.
    var MASKED_EDIT_ENGINES = [
        {
            id: 'krea2_turbo_1024', label: 'Krea2 Turbo · 1024', backend: 'krea2', arch: 'krea2',
            width: 1024, height: 1024, lanpaint: true, maxLoras: 1, implemented: true, graphReady: true,
            modelIncludes: ['turbo'], preferredModels: ['krea2-turbo'],
            profile: { steps: 8, cfg: 1.0, denoise: 1.0, sampler: 'euler', scheduler: 'simple' }
        },
        {
            id: 'krea2_raw_1024', label: 'Krea2 Raw · 1024', backend: 'krea2', arch: 'krea2',
            width: 1024, height: 1024, lanpaint: true, maxLoras: 1, implemented: true, graphReady: true,
            modelExcludes: ['turbo'], preferredModels: ['krea2-raw'],
            profile: { steps: 8, cfg: 1.0, denoise: 1.0, sampler: 'euler', scheduler: 'simple' }
        },
        {
            id: 'zimage_base_1024', label: 'Z-Image Base · 1024', backend: 'zimage', arch: 'zimage',
            width: 1024, height: 1024, lanpaint: false, maxLoras: null, implemented: true, graphReady: true,
            modelIncludes: ['base'], modelExcludes: ['turbo'], preferredModels: ['zimage_base'],
            profile: { steps: 4, cfg: 1.0, denoise: 0.65, sampler: 'euler', scheduler: 'simple' }
        },
        {
            id: 'zimage_turbo_1024', label: 'Z-Image Turbo · 1024', backend: 'zimage', arch: 'zimage',
            width: 1024, height: 1024, lanpaint: true, maxLoras: null, implemented: false, graphReady: true,
            modelIncludes: ['turbo'], blocked: 'Base-only Mojo worker',
            profile: { steps: 9, cfg: 1.0, denoise: 1.0, sampler: 'euler', scheduler: 'simple' }
        },
        {
            id: 'ideogram4_1024', label: 'Ideogram4 · 1024', backend: 'ideogram4', arch: 'ideogram4',
            width: 1024, height: 1024, lanpaint: true, maxLoras: null, implemented: false, graphReady: true,
            blocked: 'Mojo LanPaint route not wired',
            profile: { steps: 48, cfg: 4.5, denoise: 1.0, sampler: 'euler', scheduler: 'simple' }
        },
        {
            id: 'anima_1024', label: 'Anima · 1024', backend: 'anima', arch: 'anima',
            width: 1024, height: 1024, lanpaint: true, maxLoras: null, implemented: false, graphReady: true,
            blocked: 'Mojo LanPaint route not wired',
            profile: { steps: 30, cfg: 5.0, denoise: 1.0, sampler: 'euler', scheduler: 'simple' }
        },
        {
            id: 'flux2_klein_1024', label: 'Flux.2 Klein · 1024', backend: 'flux2', arch: 'klein',
            width: 1024, height: 1024, lanpaint: true, maxLoras: null, implemented: false, graphReady: true,
            blocked: 'Mojo LanPaint route not wired',
            profile: { steps: 35, cfg: 3.5, denoise: 1.0, sampler: 'euler', scheduler: 'simple' }
        },
        {
            id: 'flux2_dev_1024', label: 'Flux.2 Dev · 1024', backend: 'flux', arch: 'flux',
            width: 1024, height: 1024, lanpaint: true, maxLoras: null, implemented: false, graphReady: true,
            modelIncludes: ['flux2'], blocked: 'Model absent; Mojo LanPaint route not wired',
            profile: { steps: 28, cfg: 1.0, denoise: 1.0, sampler: 'euler', scheduler: 'simple' }
        },
        {
            id: 'qwen_image_1024', label: 'Qwen Image · 1024', backend: 'qwenimage', arch: 'qwen',
            width: 1024, height: 1024, lanpaint: true, maxLoras: null, implemented: false, graphReady: true,
            modelExcludes: ['edit'], preferredModels: ['qwen-image-2512'], blocked: 'Mojo LanPaint route not wired',
            profile: { steps: 20, cfg: 1.0, denoise: 1.0, sampler: 'euler', scheduler: 'simple' }
        },
        {
            id: 'qwen_image_edit_1024', label: 'Qwen Image Edit · 1024', backend: 'qwenimage', arch: 'qwen',
            width: 1024, height: 1024, lanpaint: true, maxLoras: null, implemented: false, graphReady: true,
            modelIncludes: ['edit'], blocked: 'Mojo masked-edit route not wired',
            profile: { steps: 20, cfg: 1.0, denoise: 1.0, sampler: 'euler', scheduler: 'simple' }
        },
        {
            id: 'flux1_dev_1024', label: 'Flux.1 Dev · 1024', backend: 'flux', arch: 'flux',
            width: 1024, height: 1024, lanpaint: true, maxLoras: null, implemented: false, graphReady: true,
            modelExcludes: ['flux2'], preferredModels: ['flux1-dev'], blocked: 'Mojo LanPaint route not wired',
            profile: { steps: 28, cfg: 1.0, denoise: 1.0, sampler: 'euler', scheduler: 'simple' }
        },
        {
            id: 'sdxl_1024', label: 'SDXL · 1024', backend: 'sdxl', arch: 'sdxl',
            width: 1024, height: 1024, lanpaint: true, maxLoras: null, implemented: false, graphReady: true,
            blocked: 'Mojo LanPaint route not wired',
            profile: { steps: 30, cfg: 7.0, denoise: 1.0, sampler: 'euler', scheduler: 'simple' }
        },
        {
            id: 'sd35_1024', label: 'SD 3.5 · 1024', backend: 'sd3', arch: 'sd3',
            width: 1024, height: 1024, lanpaint: true, maxLoras: null, implemented: false, graphReady: true,
            blocked: 'Mojo LanPaint route not wired',
            profile: { steps: 28, cfg: 5.0, denoise: 1.0, sampler: 'euler', scheduler: 'simple' }
        },
        {
            id: 'hunyuan_1024', label: 'Hunyuan T2I · 1024', backend: 'hunyuan', arch: 'hunyuan',
            width: 1024, height: 1024, lanpaint: true, maxLoras: null, implemented: false, graphReady: false,
            blocked: 'Model absent; Mojo LanPaint route not wired'
        },
        {
            id: 'wan22_t2i_1024', label: 'Wan 2.2 T2I · 1024', backend: 'wan', arch: 'wan',
            width: 1024, height: 1024, lanpaint: true, maxLoras: null, implemented: false, graphReady: false,
            blocked: 'Video-based route excluded here'
        },
        {
            id: 'hidream_1024', label: 'HiDream · 1024', backend: 'hidream', arch: 'sd15',
            width: 1024, height: 1024, lanpaint: true, maxLoras: null, implemented: false, graphReady: false,
            modelIncludes: ['hidream'], blocked: 'Model absent; base graph and Mojo route not wired'
        },
        {
            id: 'sd15_512', label: 'SD 1.5 · 512', backend: 'sd15', arch: 'sd15',
            width: 512, height: 512, lanpaint: true, maxLoras: null, implemented: false, graphReady: true,
            modelIncludes: ['sd1'], blocked: 'Model absent; Mojo LanPaint route not wired',
            profile: { steps: 30, cfg: 7.0, denoise: 1.0, sampler: 'euler', scheduler: 'simple' }
        }
    ];
    function maskedEditEngineDefinition(engine) {
        return MASKED_EDIT_ENGINES.find(function (candidate) { return candidate.id === engine; }) || null;
    }
    function maskedEditBackendProfile(definition) {
        if (!definition || !canvasCapabilities || !Array.isArray(canvasCapabilities.backends))
            return null;
        return canvasCapabilities.backends.find(function (profile) {
            return profile && profile.backend === definition.backend;
        }) || null;
    }
    function maskedEditBackendAdmitted(definition) {
        var profile = maskedEditBackendProfile(definition);
        var inpaint = profile && profile.features && profile.features.inpaint;
        return !!(profile && profile.production_status === 'admitted' && inpaint && inpaint.supported === true);
    }
    function maskedEditModelScore(definition, modelName) {
        var name = String(modelName || '').toLowerCase();
        if (ModelUtils.detectArchFromFilename(modelName) !== definition.arch)
            return -1;
        if ((definition.modelIncludes || []).some(function (needle) { return name.indexOf(needle) < 0; }))
            return -1;
        if ((definition.modelExcludes || []).some(function (needle) { return name.indexOf(needle) >= 0; }))
            return -1;
        var preferred = (definition.preferredModels || []).indexOf(name);
        return preferred >= 0 ? 100 - preferred : 50;
    }
    function findCanvasModelForMaskedEditEngine(definition) {
        if (!definition || !els.model)
            return null;
        return Array.from(els.model.options || []).map(function (option) {
            return { option: option, score: maskedEditModelScore(definition, option.value) };
        }).filter(function (candidate) { return candidate.score >= 0; })
            .sort(function (a, b) { return b.score - a.score; })[0] || null;
    }
    function populateMaskedEditEngineOptions() {
        if (!els.lanpaintEngine)
            return [];
        var available = MASKED_EDIT_ENGINES.filter(function (definition) {
            return definition.implemented === true && maskedEditBackendAdmitted(definition) && !!findCanvasModelForMaskedEditEngine(definition);
        });
        els.lanpaintEngine.innerHTML = '';
        var availableGroup = document.createElement('optgroup');
        availableGroup.label = 'Available on this machine';
        available.forEach(function (definition) {
            var option = document.createElement('option');
            option.value = definition.id;
            option.textContent = definition.label;
            availableGroup.appendChild(option);
        });
        if (available.length > 0)
            els.lanpaintEngine.appendChild(availableGroup);
        var unavailableGroup = document.createElement('optgroup');
        unavailableGroup.label = 'LanPaint supported · needs Mojo route/model';
        MASKED_EDIT_ENGINES.filter(function (definition) {
            return available.indexOf(definition) < 0;
        }).forEach(function (definition) {
            var model = findCanvasModelForMaskedEditEngine(definition);
            var profile = maskedEditBackendProfile(definition);
            var inpaint = profile && profile.features && profile.features.inpaint;
            var reason = !model ? 'model not installed' :
                (definition.implemented !== true ? definition.blocked :
                    ((inpaint && inpaint.reason) || 'backend does not admit masked editing'));
            var option = document.createElement('option');
            option.value = definition.id;
            option.textContent = definition.label + ' — ' + reason;
            option.title = reason;
            option.disabled = true;
            unavailableGroup.appendChild(option);
        });
        if (unavailableGroup.children.length > 0)
            els.lanpaintEngine.appendChild(unavailableGroup);
        if (available.length === 0) {
            var unavailable = document.createElement('option');
            unavailable.disabled = true;
            unavailable.selected = true;
            unavailable.textContent = 'No admitted masked-edit engines';
            els.lanpaintEngine.appendChild(unavailable);
            els.lanpaintEngine.disabled = true;
            return available;
        }
        els.lanpaintEngine.disabled = false;
        if (!available.some(function (definition) { return definition.id === genState.lanpaintEngine; }))
            genState.lanpaintEngine = available[0].id;
        els.lanpaintEngine.value = genState.lanpaintEngine;
        return available;
    }
    function updateMaskedEditControlSurface(definition) {
        var usesLanPaint = !!(definition && definition.lanpaint);
        if (els.lanpaintControls)
            els.lanpaintControls.style.display = usesLanPaint ? '' : 'none';
        if (els.maskedEditHelper) {
            els.maskedEditHelper.textContent = usesLanPaint ?
                'Every LanPaint value is submitted to the Mojo sampler. Unsupported values fail before model loading.' :
                'Z-Image uses its Mojo-native mask-aware denoise path. Steps, CFG, denoise, prompt, seed, mask, and LoRAs remain visible request controls.';
        }
        if (els.denoise)
            els.denoise.disabled = usesLanPaint;
        if (els.denoiseHelp) {
            els.denoiseHelp.textContent = usesLanPaint ?
                'LanPaint uses the complete schedule at full denoise.' :
                'Low = preserve more source detail · High = stronger masked change';
        }
    }
    function applyMaskedEditEngineProfile(engine) {
        var definition = maskedEditEngineDefinition(engine);
        if (!definition)
            return false;
        var profile = definition.profile;
        genState.steps = profile.steps;
        genState.cfg = profile.cfg;
        genState.denoise = profile.denoise;
        genState.sampler = profile.sampler;
        genState.scheduler = profile.scheduler;
        els.steps.value = String(profile.steps);
        els.stepsRange.value = String(profile.steps);
        els.cfg.value = String(profile.cfg);
        els.cfgRange.value = String(profile.cfg);
        els.denoise.value = String(profile.denoise);
        els.denoiseVal.textContent = Number(profile.denoise).toFixed(2);
        els.sampler.value = profile.sampler;
        els.scheduler.value = profile.scheduler;
        updateMaskedEditControlSurface(definition);
        return true;
    }
    function normalizeFlowEditEngine(engine) {
        if (engine === 'krea2_1024')
            return 'krea2_raw_1024';
        if (engine === 'krea2')
            return 'krea2_raw_512';
        return engine || 'krea2_raw_1024';
    }
    function isKreaFlowEditEngine(engine) {
        return String(normalizeFlowEditEngine(engine)).indexOf('krea2_') === 0;
    }
    function isKreaTurboFlowEditEngine(engine) {
        return isKreaFlowEditEngine(engine) && String(normalizeFlowEditEngine(engine)).indexOf('_turbo_') >= 0;
    }
    function is1024FlowEditEngine(engine) {
        return engine === 'ideogram4' || String(normalizeFlowEditEngine(engine)).endsWith('_1024');
    }
    function flowEditEngineSize(engine) {
        return is1024FlowEditEngine(engine) ? 1024 : 512;
    }
    function flowEditEngineLabel(engine) {
        if (engine === 'ideogram4')
            return 'Ideogram4';
        return isKreaTurboFlowEditEngine(engine) ? 'Krea2 Turbo' : 'Krea2 Raw';
    }
    function upgradeKreaFlowEditEngineTo1024(engine) {
        var normalized = normalizeFlowEditEngine(engine);
        if (!isKreaFlowEditEngine(normalized) || normalized.endsWith('_1024'))
            return normalized;
        return isKreaTurboFlowEditEngine(normalized) ? 'krea2_turbo_1024' : 'krea2_raw_1024';
    }
    function syncCanvasModelFromFlowEditEngine() {
        if (!els.model)
            return false;
        var engine = normalizeFlowEditEngine(genState.editEngine);
        var options = Array.from(els.model.options || []);
        var match = options.find(function (option) {
            var arch = ModelUtils.detectArchFromFilename(option.value);
            if (engine === 'ideogram4')
                return arch === 'ideogram4';
            if (!isKreaFlowEditEngine(engine) || arch !== 'krea2')
                return false;
            return (String(option.value).toLowerCase().indexOf('turbo') >= 0) === isKreaTurboFlowEditEngine(engine);
        });
        if (!match)
            return false;
        els.model.value = match.value;
        genState.model = match.value;
        genState.arch = ModelUtils.detectArchFromFilename(match.value);
        updateTopbarModel(match.value);
        return true;
    }
    function syncFlowEditEngineFromCanvasModel(modelName) {
        var arch = ModelUtils.detectArchFromFilename(modelName);
        if (arch === 'ideogram4')
            genState.editEngine = 'ideogram4';
        else if (arch === 'krea2') {
            var turbo = String(modelName).toLowerCase().indexOf('turbo') >= 0;
            var size = genState.editMode === 'style' ? 1024 : flowEditEngineSize(genState.editEngine);
            genState.editEngine = 'krea2_' + (turbo ? 'turbo' : 'raw') + '_' + size;
        }
        else {
            return false;
        }
        if (els.editEngine)
            els.editEngine.value = genState.editEngine;
        applyFlowEditEngineProfile(genState.editEngine);
        return true;
    }
    var availableCanvasLoras = [];
    var canvasLoras = [];
    var canvasCapabilities = null;
    var samAvailable = false;
    var GALLERY_BOARD_KEY = 'serenity-canvas-gallery-boards-v1';
    var galleryBoards = ['Uncategorized'];
    var galleryBoardAssignments = {};
    var galleryBoardFilter = 'all';
    // DOM refs
    var els = {};
    // Handle size constant
    var HANDLE_SIZE = 12;
    // ── Lucide SVG icons ──
    var ICONS = {
        mousePointer: '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3l7.07 16.97 2.51-7.39 7.39-2.51L3 3z"/><path d="M13 13l6 6"/></svg>',
        brush: '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m9.06 11.9 8.07-8.06a2.85 2.85 0 1 1 4.03 4.03l-8.06 8.08"/><path d="M7.07 14.94c-1.66 0-3 1.35-3 3.02 0 1.33-2.5 1.52-2 2.02 1.08 1.1 2.49 2.02 4 2.02 2.2 0 4-1.8 4-4.04a3.01 3.01 0 0 0-3-3.02z"/></svg>',
        eraser: '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m7 21-4.3-4.3c-1-1-1-2.5 0-3.4l9.6-9.6c1-1 2.5-1 3.4 0l5.6 5.6c1 1 1 2.5 0 3.4L13 21"/><path d="M22 21H7"/><path d="m5 11 9 9"/></svg>',
        move: '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="5 9 2 12 5 15"/><polyline points="9 5 12 2 15 5"/><polyline points="15 19 12 22 9 19"/><polyline points="19 9 22 12 19 15"/><line x1="2" y1="12" x2="22" y2="12"/><line x1="12" y1="2" x2="12" y2="22"/></svg>',
        maximize: '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>',
        eye: '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>',
        eyeOff: '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.88 9.88a3 3 0 1 0 4.24 4.24"/><path d="M10.73 5.08A10.43 10.43 0 0 1 12 5c7 0 10 7 10 7a13.16 13.16 0 0 1-1.67 2.68"/><path d="M6.61 6.61A13.526 13.526 0 0 0 2 12s3 7 10 7a9.74 9.74 0 0 0 5.39-1.61"/><line x1="2" y1="2" x2="22" y2="22"/></svg>',
        mask: '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M12 3v18"/><path d="M3 12h18"/></svg>',
        undo: '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7v6h6"/><path d="M21 17a9 9 0 0 0-9-9 9 9 0 0 0-6 2.3L3 13"/></svg>',
        redo: '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 7v6h-6"/><path d="M3 17a9 9 0 0 1 9-9 9 9 0 0 1 6 2.3L21 13"/></svg>',
        trash: '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/></svg>',
        lock: '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>',
        unlock: '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 9.9-1"/></svg>'
    };
    // ── Helpers ──
    function snapTo64(val) { return Math.max(64, Math.round(val / 64) * 64); }
    function snapTo32(val) { return Math.max(64, Math.round(val / 32) * 32); }
    function clampDim(val) { return Math.max(256, Math.min(4096, val)); }
    function isVideoArch() { return genState.arch === 'ltxv' || genState.arch === 'wan' || genState.arch === 'bernini'; }
    function snapDimForArch(val) {
        return isVideoArch() ? snapTo32(val) : snapTo64(val);
    }
    function clampDimForArch(val) {
        if (isVideoArch())
            return Math.max(64, Math.min(1280, snapTo32(val)));
        return clampDim(snapTo64(val));
    }
    function getRelativePointerPosition() {
        var transform = stage.getAbsoluteTransform().copy().invert();
        var pos = stage.getPointerPosition();
        if (!pos)
            return { x: 0, y: 0 };
        return transform.point(pos);
    }
    function getActiveKonvaLayer() {
        for (var i = 0; i < canvasLayers.length; i++) {
            if (canvasLayers[i].data.id === activeLayerId)
                return canvasLayers[i].konvaLayer;
        }
        return canvasLayers.length > 0 ? canvasLayers[0].konvaLayer : null;
    }
    function getLayerById(id) {
        for (var i = 0; i < canvasLayers.length; i++) {
            if (canvasLayers[i].data.id === id)
                return canvasLayers[i];
        }
        return null;
    }
    function getActiveLayer() {
        return getLayerById(activeLayerId);
    }
    // ── Snapshot-based Undo/Redo (uses CanvasSnapshot from layer-types.ts) ──
    var History = (function () {
        var stack = [];
        var cursor = -1;
        var MAX = 50;
        var snapshotting = false;
        var strokeGroupTimer = null;
        var STROKE_GROUP_MS = 500; // group rapid strokes within this window
        function snapshot() {
            if (!stage || !boundingBox)
                return null;
            return {
                layers: canvasLayers.map(function (l) {
                    return {
                        data: JSON.parse(JSON.stringify(l.data)),
                        imageData: l.konvaLayer.toDataURL()
                    };
                }),
                bbox: {
                    x: boundingBox.x(), y: boundingBox.y(),
                    width: boundingBox.width(), height: boundingBox.height()
                },
                activeLayerId: activeLayerId
            };
        }
        function push() {
            if (snapshotting) {
                console.log('[History] push blocked by snapshotting');
                return;
            }
            var snap = snapshot();
            if (!snap)
                return;
            stack.splice(cursor + 1);
            stack.push(snap);
            if (stack.length > MAX)
                stack.shift();
            cursor = stack.length - 1;
            console.log('[History] push: cursor=' + cursor + ' stack.length=' + stack.length);
            updateButtons();
        }
        /** Push with stroke grouping — rapid strokes become one undo entry */
        function pushGrouped() {
            if (strokeGroupTimer)
                clearTimeout(strokeGroupTimer);
            strokeGroupTimer = setTimeout(function () {
                push();
                strokeGroupTimer = null;
            }, STROKE_GROUP_MS);
        }
        function undo() {
            // Flush any pending grouped stroke — save it as a new entry
            if (strokeGroupTimer) {
                clearTimeout(strokeGroupTimer);
                strokeGroupTimer = null;
                // Save current state (with the stroke) before undoing
                var snap = snapshot();
                if (snap) {
                    stack.splice(cursor + 1); // remove any old redo entries
                    stack.push(snap);
                    cursor = stack.length - 1;
                }
            }
            if (cursor <= 0)
                return;
            cursor--;
            restore(stack[cursor]);
            console.log('[History] undo: cursor=' + cursor + ' stack.length=' + stack.length);
            updateButtons();
        }
        function redo() {
            if (cursor >= stack.length - 1) {
                console.log('[History] redo blocked: cursor=' + cursor + ' stack.length=' + stack.length);
                return;
            }
            cursor++;
            restore(stack[cursor]);
            console.log('[History] redo: cursor=' + cursor + ' stack.length=' + stack.length);
            updateButtons();
        }
        function restore(entry) {
            if (!entry || !stage)
                return;
            snapshotting = true;
            canvasLayers.forEach(function (l) { l.konvaLayer.destroy(); });
            canvasLayers = [];
            var loaded = 0;
            var total = entry.layers.length;
            entry.layers.forEach(function (saved) {
                var konvaLayer = new Konva.Layer();
                stage.add(konvaLayer);
                if (uiLayer && uiLayer.parent)
                    uiLayer.moveToTop();
                var data = LayerValidation.sanitise(JSON.parse(JSON.stringify(saved.data)));
                var cl = { data: data, konvaLayer: konvaLayer };
                konvaLayer.opacity(data.opacity);
                if (!data.visible)
                    konvaLayer.hide();
                // Apply blend mode for draw layers
                if (data.type === 'draw') {
                    var composite = BlendModeUtil.toCompositeOp(data.blendMode);
                    if (composite !== 'source-over') {
                        konvaLayer.canvas()._canvas.style.mixBlendMode = composite;
                    }
                }
                canvasLayers.push(cl);
                if (saved.imageData && saved.imageData !== 'data:,') {
                    var img = new Image();
                    img.onload = function () {
                        var kImg = new Konva.Image({ image: img, x: 0, y: 0 });
                        konvaLayer.add(kImg);
                        konvaLayer.batchDraw();
                        loaded++;
                        if (loaded >= total)
                            finishRestore(entry);
                    };
                    img.onerror = function () {
                        loaded++;
                        if (loaded >= total)
                            finishRestore(entry);
                    };
                    img.src = saved.imageData;
                }
                else {
                    // Text layers: re-create Konva.Text node
                    if (data.type === 'text') {
                        var td = data;
                        var kText = new Konva.Text({
                            x: td.position.x, y: td.position.y,
                            text: td.text, fontSize: td.fontSize,
                            fontFamily: td.fontFamily, fill: td.color,
                            fontStyle: td.fontWeight, align: td.alignment,
                            lineHeight: td.lineHeight, draggable: true,
                        });
                        konvaLayer.add(kText);
                        attachInlineTextEdit(kText, data, konvaLayer);
                    }
                    loaded++;
                    if (loaded >= total)
                        finishRestore(entry);
                }
            });
            if (total === 0)
                finishRestore(entry);
        }
        function finishRestore(entry) {
            if (entry.bbox) {
                boundingBox.x(entry.bbox.x);
                boundingBox.y(entry.bbox.y);
                boundingBox.width(entry.bbox.width);
                boundingBox.height(entry.bbox.height);
                updateHandles();
                updateSizeLabel();
            }
            activeLayerId = entry.activeLayerId;
            var maxId = 0;
            canvasLayers.forEach(function (l) { if (l.data.id > maxId)
                maxId = l.data.id; });
            LayerDefaults.setIdCounter(maxId);
            renderLayerList();
            stage.batchDraw();
            snapshotting = false;
        }
        function updateButtons() {
            var undoBtn = document.getElementById('cv-undo');
            var redoBtn = document.getElementById('cv-redo');
            if (undoBtn) {
                undoBtn.disabled = cursor <= 0;
                undoBtn.title = 'Undo (Ctrl+Z)' + (cursor > 0 ? ' (' + cursor + ')' : '');
            }
            if (redoBtn) {
                redoBtn.disabled = cursor >= stack.length - 1;
            }
        }
        return { push: push, pushGrouped: pushGrouped, undo: undo, redo: redo, updateButtons: updateButtons };
    })();
    function debouncedHistoryPush() {
        History.pushGrouped();
    }
    // ── Floating Preview Panel ──
    var lastPreviewSrc = null;
    var lastPreviewIsVideo = false;
    function showCanvasPreview(src, isVideo) {
        lastPreviewSrc = src;
        lastPreviewIsVideo = isVideo;
        var body = document.getElementById('canvas-preview-body');
        if (!body)
            return;
        body.innerHTML = isVideo
            ? '<video src="' + src + '" autoplay loop muted playsinline controls></video>'
            : '<img src="' + src + '">';
        var panel = document.getElementById('canvas-preview-panel');
        if (panel)
            panel.style.display = 'block';
    }
    function hideCanvasPreview() {
        var panel = document.getElementById('canvas-preview-panel');
        if (panel)
            panel.style.display = 'none';
        lastPreviewSrc = null;
    }
    function acceptCanvasPreview() {
        if (!lastPreviewSrc)
            return;
        if (lastPreviewIsVideo) {
            placeVideoOverlayOnCanvas(lastPreviewSrc);
        }
        else {
            placeResultOnCanvas(lastPreviewSrc);
        }
        hideCanvasPreview();
    }
    function setupPreviewPanel() {
        var panel = document.getElementById('canvas-preview-panel');
        var header = document.getElementById('canvas-preview-header');
        var closeBtn = document.getElementById('canvas-preview-close');
        var acceptBtn = document.getElementById('canvas-preview-accept');
        var discardBtn = document.getElementById('canvas-preview-discard');
        var downloadBtn = document.getElementById('canvas-preview-download');
        if (!panel || !header)
            return;
        // Drag behavior
        var dragging = false, startX = 0, startY = 0, startLeft = 0, startTop = 0;
        header.addEventListener('mousedown', function (e) {
            dragging = true;
            startX = e.clientX;
            startY = e.clientY;
            startLeft = panel.offsetLeft;
            startTop = panel.offsetTop;
        });
        document.addEventListener('mousemove', function (e) {
            if (!dragging)
                return;
            var newLeft = Math.max(0, Math.min(startLeft + e.clientX - startX, window.innerWidth - panel.offsetWidth));
            var newTop = Math.max(0, Math.min(startTop + e.clientY - startY, window.innerHeight - 40));
            panel.style.left = newLeft + 'px';
            panel.style.top = newTop + 'px';
            panel.style.right = 'auto';
        });
        document.addEventListener('mouseup', function () { dragging = false; });
        if (closeBtn)
            closeBtn.addEventListener('click', hideCanvasPreview);
        if (discardBtn)
            discardBtn.addEventListener('click', hideCanvasPreview);
        if (acceptBtn)
            acceptBtn.addEventListener('click', acceptCanvasPreview);
        if (downloadBtn) {
            downloadBtn.addEventListener('click', function () {
                if (!lastPreviewSrc)
                    return;
                var a = document.createElement('a');
                a.href = lastPreviewSrc;
                a.download = 'serenityflow_canvas_' + Date.now() + (lastPreviewIsVideo ? '.mp4' : '.png');
                a.click();
            });
        }
    }
    // ── Image upload for Control layers ──
    function handleLayerImageUpload(layerId, file, wellEl) {
        var reader = new FileReader();
        reader.onload = function (ev) {
            var result = ev.target.result;
            var cl = getLayerById(layerId);
            if (cl && cl.data.type === 'control') {
                var cd = cl.data;
                cd.refImageSrc = result;
                cd.refImageName = file.name;
                if (wellEl) {
                    wellEl.innerHTML = '<img src="' + result + '">';
                }
                // Also place on canvas as semi-transparent overlay
                var img = new Image();
                img.onload = function () {
                    cl.konvaLayer.destroyChildren();
                    var kImg = new Konva.Image({
                        image: img,
                        x: boundingBox.x(), y: boundingBox.y(),
                        width: boundingBox.width(), height: boundingBox.height(),
                        opacity: 0.5,
                        draggable: activeTool === 'select'
                    });
                    cl.konvaLayer.add(kImg);
                    cl.konvaLayer.batchDraw();
                    History.push();
                };
                img.src = result;
            }
        };
        reader.readAsDataURL(file);
    }
    // ── Checkerboard ──
    function createCheckerboardImage() {
        var size = 16;
        var c = document.createElement('canvas');
        c.width = size * 2;
        c.height = size * 2;
        var ctx = c.getContext('2d');
        ctx.fillStyle = '#181824';
        ctx.fillRect(0, 0, size * 2, size * 2);
        ctx.fillStyle = '#1e1e2e';
        ctx.fillRect(0, 0, size, size);
        ctx.fillRect(size, size, size, size);
        var img = new Image();
        img.src = c.toDataURL();
        return img;
    }
    // ── Build DOM ──
    function buildUI() {
        var panel = document.getElementById('panel-canvas');
        if (!panel)
            return;
        panel.innerHTML = '';
        var layout = document.createElement('div');
        layout.className = 'cv-layout';
        var left = document.createElement('div');
        left.className = 'cv-left';
        left.innerHTML = buildLeftHTML();
        layout.appendChild(left);
        var center = document.createElement('div');
        center.className = 'cv-center';
        center.innerHTML =
            '<div class="cv-edit-workspace" id="cv-edit-workspace">' +
            '<section class="cv-edit-pane cv-source-pane" id="cv-source-pane" aria-label="Edit source">' +
            '<div class="cv-source-slot cv-primary-source-slot" id="cv-primary-source-slot">' +
            '<div class="cv-edit-pane-header"><span>Source</span><button id="cv-source-browse" type="button">Choose file</button></div>' +
            '<div class="cv-source-dropzone" id="cv-source-dropzone" role="button" tabindex="0">' +
            '<span class="cv-source-empty" id="cv-source-empty">Drop an image here<br><small>or choose it from the file manager</small></span>' +
            '<img id="cv-source-preview" alt="Edit source preview">' +
            '<video id="cv-source-video" controls muted></video>' +
            '</div></div>' +
            '<div class="cv-source-slot cv-style-source-slot" id="cv-style-source-slot">' +
            '<div class="cv-edit-pane-header"><span>Style reference</span><div class="cv-style-header-actions">' +
            '<label class="cv-style-entire"><input id="cv-style-entire-image" type="checkbox" checked>Entire image</label>' +
            '<button id="cv-style-browse" type="button">Choose style</button></div></div>' +
            '<div class="cv-source-dropzone cv-style-dropzone" id="cv-style-dropzone" role="button" tabindex="0">' +
            '<span class="cv-source-empty" id="cv-style-empty">Drop a style image here<br><small>or choose it from the file manager</small></span>' +
            '<img id="cv-style-preview" alt="Style reference preview">' +
            '</div><input type="file" id="cv-style-file" accept="image/*" style="display:none">' +
            '</div></section>' +
            '<section class="cv-edit-pane cv-result-pane" aria-label="Edit result">' +
            '<div class="cv-edit-pane-header"><span id="cv-result-pane-title">Canvas</span></div>' +
            '<div id="canvas-stage-container"></div>' +
            '</section></div>';
        // Add bbox toolbar to center
        var bboxToolbar = document.createElement('div');
        bboxToolbar.id = 'cv-bbox-toolbar';
        bboxToolbar.className = 'cv-bbox-toolbar';
        bboxToolbar.innerHTML =
            '<input type="number" id="cv-bbox-w" class="cv-bbox-input" min="64" max="4096" step="64" value="1024">' +
                '<span class="cv-bbox-x">&times;</span>' +
                '<input type="number" id="cv-bbox-h" class="cv-bbox-input" min="64" max="4096" step="64" value="1024">' +
                '<span class="cv-bbox-sep">|</span>' +
                '<select id="cv-bbox-snap" class="cv-bbox-snap">' +
                '<option value="32">Snap: 32</option>' +
                '<option value="64" selected>Snap: 64</option>' +
                '<option value="128">Snap: 128</option>' +
                '</select>' +
                '<button id="cv-bbox-reset" class="cv-bbox-btn" title="Reset to 1024x1024">Reset</button>' +
                '<button id="cv-bbox-lock" class="cv-bbox-btn" title="Lock aspect ratio">&#128274;</button>' +
                '<button id="cv-bbox-fit" class="cv-bbox-btn" title="Fit to active layer">Fit</button>';
        center.querySelector('.cv-result-pane').appendChild(bboxToolbar);
        layout.appendChild(center);
        var right = document.createElement('div');
        right.className = 'cv-right';
        right.innerHTML = buildRightHTML();
        layout.appendChild(right);
        panel.appendChild(layout);
        cacheElements();
    }
    function buildLeftHTML() {
        return '' +
            '<div class="cv-tools">' +
            '<button class="cv-tool-btn active" data-tool="select" title="Select / Bbox (V)">' + ICONS.mousePointer + '</button>' +
            '<button class="cv-tool-btn" data-tool="move" title="Move (M)">' + ICONS.move + '</button>' +
            '<button class="cv-tool-btn" data-tool="brush" title="Brush (B)">' + ICONS.brush + '</button>' +
            '<button class="cv-tool-btn" data-tool="eraser" title="Eraser (E)">' + ICONS.eraser + '</button>' +
            '<button class="cv-tool-btn" data-tool="mask" title="Mask Paint">' + ICONS.mask + '</button>' +
            '<button class="cv-tool-btn" data-tool="rect" title="Rectangle (R)"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/></svg></button>' +
            '<button class="cv-tool-btn" data-tool="gradient" title="Gradient (G)"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v20M2 12h20"/><circle cx="12" cy="12" r="9" opacity="0.3"/></svg></button>' +
            '<button class="cv-tool-btn" data-tool="fill" title="Fill (F)"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 22l1-1h3l9-9"/><path d="M3 21v-3l9-9 3 3-9 9z"/><path d="M14 6l3-3 3 3-3 3z"/><path d="M19 13c.3 1 1.5 3 0 4.5S16 19 16 19"/></svg></button>' +
            '<button class="cv-tool-btn" data-tool="text" title="Text (T)"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="4 7 4 4 20 4 20 7"/><line x1="9.5" y1="20" x2="14.5" y2="20"/><line x1="12" y1="4" x2="12" y2="20"/></svg></button>' +
            '<button class="cv-tool-btn" data-tool="lasso" title="Lasso (L)"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M7 22a5 5 0 0 1-2-4"/><path d="M7 16.93c.96.43 1.96.74 2.99.91"/><path d="M3.34 14A6.8 6.8 0 0 1 2 10c0-4.42 4.48-8 10-8s10 3.58 10 8-4.48 8-10 8a12 12 0 0 1-3-.38"/><path d="M5 18a2 2 0 1 0 0-4 2 2 0 0 0 0 4z"/></svg></button>' +
            '<button class="cv-tool-btn" data-tool="pencil" title="Pencil (P)"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/></svg></button>' +
            '<button class="cv-tool-btn" data-tool="speechbubble" title="Speech Bubble (U)"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg></button>' +
            '<button class="cv-tool-btn" data-tool="clonestamp" title="Clone Stamp (C)"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="4"/><path d="M16 8V5a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3"/><path d="M8 16v3a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1v-3"/></svg></button>' +
            '<button class="cv-tool-btn" data-tool="sam" title="Select Object / SAM (S)"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 11V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h6"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/><path d="M19 17v4m2-2h-4"/></svg></button>' +
            '<button class="cv-tool-btn" data-tool="colorpicker" title="Color Picker (I)"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m2 22 1-1h3l9-9"/><path d="M3 21v-3l9-9 3 3-9 9z"/></svg></button>' +
            '<button class="cv-tool-btn" data-tool="pan" title="Pan (H)"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 11V6a2 2 0 0 0-4 0v5"/><path d="M14 10V4a2 2 0 0 0-4 0v6"/><path d="M10 10.5V6a2 2 0 0 0-4 0v8"/><path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"/></svg></button>' +
            '<button class="cv-tool-btn" data-tool="resetView" title="Reset View">' + ICONS.maximize + '</button>' +
            '<hr class="cv-tool-separator">' +
            '<button class="cv-tool-btn cv-undo-btn" id="cv-undo" title="Undo (Ctrl+Z)">' + ICONS.undo + '</button>' +
            '<button class="cv-tool-btn cv-redo-btn" id="cv-redo" title="Redo (Ctrl+Shift+Z)">' + ICONS.redo + '</button>' +
            '</div>' +
            '<div id="cv-mask-actions" class="cv-mask-actions" style="display:none">' +
            '<button class="cv-mask-action-btn" id="cv-mask-fill">Fill All</button>' +
            '<button class="cv-mask-action-btn" id="cv-mask-clear">Clear Mask</button>' +
            '<span class="cv-mask-sep">|</span>' +
            '<span class="cv-mask-label">Opacity</span>' +
            '<input type="range" id="cv-mask-opacity" class="cv-mask-opacity-range" min="0.1" max="1" step="0.05" value="0.6">' +
            '</div>' +
            '<div class="cv-layers">' +
            '<div class="cv-layers-header">' +
            '<span class="cv-layers-title">Layers</span>' +
            '<div class="cv-session-actions">' +
            '<button class="cv-session-btn" id="cv-session-new" title="New Canvas project">New</button>' +
            '<button class="cv-session-btn" id="cv-session-save" title="Save Canvas project">Save</button>' +
            '<button class="cv-session-btn" id="cv-session-load" title="Load Canvas project">Load</button>' +
            '<button class="cv-session-btn" id="cv-hotkeys" title="Canvas keyboard shortcuts">?</button>' +
            '<input type="file" id="cv-session-file" accept=".serenity-canvas,.json,application/json" style="display:none">' +
            '</div>' +
            '<div class="cv-layers-add-wrap">' +
            '<button class="cv-layers-add" id="cv-layers-add-btn" title="Add layer">+</button>' +
            '<div id="cv-layer-type-menu" class="cv-layer-type-menu" style="display:none">' +
            '<div class="cv-layer-type-item" data-type="draw">Draw Layer</div>' +
            '<div class="cv-layer-type-item" data-type="mask">Inpaint Mask</div>' +
            '<div class="cv-layer-type-item" data-type="guidance">Guidance (Regional)</div>' +
            '<div class="cv-layer-type-item" data-type="control">Control Layer</div>' +
            '<div class="cv-layer-type-item" data-type="adjustment">Adjustment</div>' +
            '<div class="cv-layer-type-item" data-type="text">Text Layer</div>' +
            '</div>' +
            '</div>' +
            '</div>' +
            '<div class="cv-layer-list" id="cv-layer-list"></div>' +
            '<div class="cv-canvas-gallery">' +
            '<div class="cv-gallery-header"><span>Gallery</span><div class="cv-gallery-board-controls">' +
            '<select id="cv-gallery-board-filter" class="cv-gallery-board-filter" title="Filter gallery board"><option value="all">All boards</option></select>' +
            '<button id="cv-gallery-board-new" class="cv-session-btn" type="button" title="Create a gallery board">+ Board</button>' +
            '<button id="cv-gallery-refresh" class="cv-session-btn" type="button">Refresh</button></div></div>' +
            '<div id="cv-gallery-grid" class="cv-gallery-grid"><span class="cv-gallery-empty">Loading...</span></div>' +
            '</div>' +
            '</div>';
    }
    function buildRightHTML() {
        return '' +
            '<div id="cv-brush-section" class="cv-brush-settings" style="display:none">' +
            '<div class="cv-section-title">Brush</div>' +
            '<div class="cv-setting-row">' +
            '<span class="cv-setting-label">Size</span>' +
            '<input type="range" id="cv-brush-size" class="cv-range" min="1" max="200" value="20">' +
            '<span id="cv-brush-size-val" class="cv-setting-value">20</span>' +
            '</div>' +
            '<div class="cv-setting-row">' +
            '<span class="cv-setting-label">Hardness</span>' +
            '<input type="range" id="cv-brush-hardness" class="cv-range" min="0" max="1" step="0.05" value="1">' +
            '<span id="cv-brush-hardness-val" class="cv-setting-value">1.0</span>' +
            '</div>' +
            '<div class="cv-setting-row">' +
            '<span class="cv-setting-label">Color</span>' +
            '<input type="color" id="cv-brush-color" class="cv-color-swatch" value="#ffffff">' +
            '</div>' +
            '<hr class="cv-separator">' +
            '</div>' +
            '<div id="cv-shape-section" class="cv-brush-settings cv-tool-options" style="display:none">' +
            '<div class="cv-section-title">Shapes</div>' +
            '<div class="cv-setting-row"><span class="cv-setting-label">Shape</span>' +
            '<select id="cv-shape-mode" class="cv-select">' +
            '<option value="rectangle">Rectangle</option><option value="ellipse">Oval</option>' +
            '<option value="polygon">Polygon</option><option value="freehand">Freehand</option>' +
            '</select>' +
            '<button id="cv-shape-subtract" class="cv-option-toggle" type="button" title="Erase the shape from the active layer">Subtract</button></div>' +
            '<div class="cv-helper-text">Polygon: click points, then double-click or press Enter.</div>' +
            '</div>' +
            '<div id="cv-gradient-section" class="cv-brush-settings cv-tool-options" style="display:none">' +
            '<div class="cv-section-title">Gradient</div>' +
            '<div class="cv-setting-row"><span class="cv-setting-label">Mode</span>' +
            '<select id="cv-gradient-mode" class="cv-select"><option value="linear">Linear</option><option value="radial">Radial</option></select>' +
            '<input type="color" id="cv-gradient-start" class="cv-color-swatch" value="#ffffff" title="Start color">' +
            '<input type="color" id="cv-gradient-end" class="cv-color-swatch" value="#000000" title="End color">' +
            '<button id="cv-gradient-clip" class="cv-option-toggle active" type="button" title="Use the end color instead of fading to transparent">Clip</button></div>' +
            '</div>' +
            '<div id="cv-lasso-section" class="cv-brush-settings cv-tool-options" style="display:none">' +
            '<div class="cv-section-title">Lasso</div>' +
            '<div class="cv-setting-row"><span class="cv-setting-label">Mode</span>' +
            '<select id="cv-lasso-mode" class="cv-select"><option value="freehand">Freehand</option><option value="polygon">Polygon</option></select>' +
            '<button id="cv-lasso-subtract" class="cv-option-toggle" type="button">Subtract</button>' +
            '<button id="cv-lasso-auto-mask" class="cv-option-toggle active" type="button">Auto Mask</button></div>' +
            '<div class="cv-helper-text">Polygon: click points, then double-click or press Enter.</div>' +
            '</div>' +
            '<div id="cv-fill-section" class="cv-brush-settings" style="display:none">' +
            '<div class="cv-section-title">Fill</div>' +
            '<div class="cv-setting-row">' +
            '<span class="cv-setting-label">Threshold</span>' +
            '<input type="range" id="cv-fill-threshold" class="cv-range" min="0" max="255" value="32">' +
            '<span id="cv-fill-threshold-val" class="cv-setting-value">32</span>' +
            '</div>' +
            '<hr class="cv-separator">' +
            '</div>' +
            '<div id="cv-control-panel" class="cv-type-panel" style="display:none">' +
            '<div class="cv-section-title">Control Layer</div>' +
            '<div class="cv-setting-row">' +
            '<span class="cv-setting-label">Preprocessor</span>' +
            '<select id="cv-control-type" class="cv-select">' +
            '<option value="canny">Canny Edge</option>' +
            '<option value="depth">Depth</option>' +
            '<option value="lineart">Lineart</option>' +
            '<option value="pose">Pose</option>' +
            '<option value="soft_edge">Soft Edge</option>' +
            '<option value="tile">Tile</option>' +
            '<option value="normal">Normal Map</option>' +
            '<option value="color">Color Map</option>' +
            '<option value="scribble">Scribble</option>' +
            '</select>' +
            '</div>' +
            '<button class="cv-preprocess-btn" id="cv-preprocess-btn">Process</button>' +
            '<div class="cv-setting-row">' +
            '<span class="cv-setting-label">Weight</span>' +
            '<input type="range" id="cv-control-weight" class="cv-range" min="0" max="2" step="0.05" value="1">' +
            '<span id="cv-control-weight-val" class="cv-setting-value">1.00</span>' +
            '</div>' +
            '<div class="cv-setting-row">' +
            '<span class="cv-setting-label">Start</span>' +
            '<input type="range" id="cv-control-start" class="cv-range" min="0" max="1" step="0.05" value="0">' +
            '<span id="cv-control-start-val" class="cv-setting-value">0.00</span>' +
            '</div>' +
            '<div class="cv-setting-row">' +
            '<span class="cv-setting-label">End</span>' +
            '<input type="range" id="cv-control-end" class="cv-range" min="0" max="1" step="0.05" value="1">' +
            '<span id="cv-control-end-val" class="cv-setting-value">1.00</span>' +
            '</div>' +
            '<div class="cv-image-well" id="cv-control-well">' +
            '<span class="cv-image-well-placeholder">Drop image or click to upload</span>' +
            '</div>' +
            '<input type="file" id="cv-control-file" accept="image/*" style="display:none">' +
            '</div>' +
            '<div id="cv-ipadapter-panel" class="cv-type-panel" style="display:none">' +
            '<div class="cv-section-title">IP-Adapter</div>' +
            '<div class="cv-setting-row">' +
            '<span class="cv-setting-label">Weight</span>' +
            '<input type="range" id="cv-ipa-weight" class="cv-range" min="0" max="2" step="0.05" value="1">' +
            '<span id="cv-ipa-weight-val" class="cv-setting-value">1.00</span>' +
            '</div>' +
            '<div class="cv-setting-row">' +
            '<span class="cv-setting-label">Method</span>' +
            '<select id="cv-ipa-method" class="cv-select">' +
            '<option value="style">Style</option>' +
            '<option value="composition">Composition</option>' +
            '<option value="style_composition">Style + Composition</option>' +
            '</select>' +
            '</div>' +
            '<div class="cv-image-well" id="cv-ipa-well">' +
            '<span class="cv-image-well-placeholder">Drop image or click to upload</span>' +
            '</div>' +
            '<input type="file" id="cv-ipa-file" accept="image/*" style="display:none">' +
            '<div class="cv-helper-text">IP-Adapter nodes will be added when available on the backend.</div>' +
            '</div>' +
            '<div id="cv-regional-panel" class="cv-type-panel" style="display:none">' +
            '<div class="cv-section-title">Guidance (Regional Prompt)</div>' +
            '<label class="cv-setting-label">Region Prompt</label>' +
            '<textarea id="cv-regional-prompt" class="cv-textarea" rows="3" placeholder="Prompt for this region..."></textarea>' +
            '<label class="cv-setting-label" style="margin-top:8px">Negative</label>' +
            '<textarea id="cv-regional-neg" class="cv-textarea" rows="2" placeholder="Negative for this region..."></textarea>' +
            '<div class="cv-helper-text">Draw the region on the canvas. Regional conditioning will be applied when supported.</div>' +
            '</div>' +
            '<div id="cv-adjustment-panel" class="cv-type-panel" style="display:none">' +
            '<div class="cv-section-title">Adjustment</div>' +
            '<div class="cv-setting-row"><span class="cv-setting-label">Brightness</span>' +
            '<input type="range" id="cv-adj-brightness" class="cv-range" min="-1" max="1" step="0.05" value="0">' +
            '<span id="cv-adj-brightness-val" class="cv-setting-value">0</span></div>' +
            '<div class="cv-setting-row"><span class="cv-setting-label">Contrast</span>' +
            '<input type="range" id="cv-adj-contrast" class="cv-range" min="-1" max="1" step="0.05" value="0">' +
            '<span id="cv-adj-contrast-val" class="cv-setting-value">0</span></div>' +
            '<div class="cv-setting-row"><span class="cv-setting-label">Saturation</span>' +
            '<input type="range" id="cv-adj-saturation" class="cv-range" min="-1" max="1" step="0.05" value="0">' +
            '<span id="cv-adj-saturation-val" class="cv-setting-value">0</span></div>' +
            '<div class="cv-setting-row"><span class="cv-setting-label">Temperature</span>' +
            '<input type="range" id="cv-adj-temperature" class="cv-range" min="-1" max="1" step="0.05" value="0">' +
            '<span id="cv-adj-temperature-val" class="cv-setting-value">0</span></div>' +
            '<div class="cv-setting-row"><span class="cv-setting-label">Tint</span>' +
            '<input type="range" id="cv-adj-tint" class="cv-range" min="-1" max="1" step="0.05" value="0">' +
            '<span id="cv-adj-tint-val" class="cv-setting-value">0</span></div>' +
            '<div class="cv-setting-row"><span class="cv-setting-label">Sharpness</span>' +
            '<input type="range" id="cv-adj-sharpness" class="cv-range" min="0" max="1" step="0.05" value="0">' +
            '<span id="cv-adj-sharpness-val" class="cv-setting-value">0</span></div>' +
            '<div class="cv-helper-text">Non-destructive adjustments applied at render time.</div>' +
            '</div>' +
            '<div id="cv-text-panel" class="cv-type-panel" style="display:none">' +
            '<div class="cv-section-title">Text</div>' +
            '<div class="cv-setting-row"><span class="cv-setting-label">Content</span>' +
            '<input type="text" id="cv-text-content" class="cv-text-input" value="Text" placeholder="Enter text...">' +
            '</div>' +
            '<div class="cv-setting-row"><span class="cv-setting-label">Font</span>' +
            '<select id="cv-text-font" class="cv-select">' +
            '<option value="Inter, system-ui, sans-serif">Inter</option>' +
            '<option value="Georgia, serif">Georgia</option>' +
            '<option value="Courier New, monospace">Courier New</option>' +
            '<option value="Arial, sans-serif">Arial</option>' +
            '<option value="Verdana, sans-serif">Verdana</option>' +
            '</select>' +
            '</div>' +
            '<div class="cv-setting-row"><span class="cv-setting-label">Size</span>' +
            '<input type="range" id="cv-text-size" class="cv-range" min="8" max="200" value="32">' +
            '<span id="cv-text-size-val" class="cv-setting-value">32</span></div>' +
            '<div class="cv-setting-row"><span class="cv-setting-label">Color</span>' +
            '<input type="color" id="cv-text-color" class="cv-color-swatch" value="#ffffff"></div>' +
            '<div class="cv-setting-row"><span class="cv-setting-label">Weight</span>' +
            '<select id="cv-text-weight" class="cv-select">' +
            '<option value="normal">Normal</option>' +
            '<option value="bold">Bold</option>' +
            '<option value="italic">Italic</option>' +
            '<option value="bold italic">Bold Italic</option>' +
            '</select>' +
            '</div>' +
            '</div>' +
            '<div class="cv-gen-settings">' +
            '<div class="cv-section-title">Generation</div>' +
            '<div class="cv-setting-row"><span class="cv-setting-label">Mode</span>' +
            '<select id="cv-edit-mode" class="cv-select">' +
            '<option value="create">Create / canvas</option>' +
            '<option value="flowedit">Image edit — FlowEdit</option>' +
            '<option value="style">Style transfer — FlowEdit</option>' +
            '<option value="inpaint">Masked Edit - LanPaint</option>' +
            '<option value="dynaedit">Video edit — DynaEdit</option>' +
            '</select></div>' +
            '<div id="cv-flowedit-section" class="cv-edit-settings" style="display:none">' +
            '<div class="cv-setting-row"><span class="cv-setting-label">Engine</span>' +
            '<select id="cv-edit-engine" class="cv-select">' +
            '<option value="krea2_raw_1024">Krea2 Raw · 1024</option>' +
            '<option value="krea2_turbo_1024">Krea2 Turbo · 1024</option>' +
            '<option value="krea2_raw_512">Krea2 Raw · 512</option>' +
            '<option value="krea2_turbo_512">Krea2 Turbo · 512</option>' +
            '<option value="ideogram4">Ideogram4 · 1024</option></select></div>' +
            '<label class="cv-setting-label" for="cv-edit-source-prompt">Source description (optional)</label>' +
            '<textarea id="cv-edit-source-prompt" class="cv-textarea cv-textarea-compact" rows="2" placeholder="Auto-described when empty; edit to override..."></textarea>' +
            '<details class="cv-edit-advanced"><summary>FlowEdit controls</summary>' +
            '<div class="cv-edit-grid"><label>N max<input id="cv-edit-nmax" type="number" min="0" max="150" value="24"></label>' +
            '<label>N min<input id="cv-edit-nmin" type="number" min="0" max="150" value="0"></label>' +
            '<label>Source CFG<input id="cv-edit-src-cfg" type="number" min="0" max="20" step="0.1" value="1.5"></label>' +
            '<label>Mask Q<input id="cv-edit-mask-q" type="number" min="0" max="1" step="0.05" value="0.7"></label>' +
            '<label>Dilate<input id="cv-edit-mask-dilate" type="number" min="0" max="32" value="1"></label>' +
            '<label>Warmup<input id="cv-edit-mask-warmup" type="number" min="0" max="150" value="4"></label></div>' +
            '<label class="cv-check-row"><input id="cv-edit-auto-mask" type="checkbox" checked> Automatic change mask</label></details>' +
            '</div>' +
            '<div id="cv-lanpaint-section" class="cv-edit-settings" style="display:none">' +
            '<div class="cv-setting-row"><span class="cv-setting-label">Engine</span>' +
            '<select id="cv-lanpaint-engine" class="cv-select">' +
            '<option disabled selected>Loading supported engines...</option></select></div>' +
            '<details id="cv-lanpaint-controls" class="cv-edit-advanced" open><summary>LanPaint controls</summary>' +
            '<div class="cv-edit-grid">' +
            '<label>Inner steps<input id="cv-lanpaint-inner-steps" type="number" min="0" max="64" value="5"></label>' +
            '<label>Lambda<input id="cv-lanpaint-lambda" type="number" min="0.01" max="100" step="0.1" value="16"></label>' +
            '<label>Step size<input id="cv-lanpaint-step-size" type="number" min="0.001" max="5" step="0.01" value="0.2"></label>' +
            '<label>Beta<input id="cv-lanpaint-beta" type="number" min="0.01" max="100" step="0.1" value="1"></label>' +
            '<label>Friction<input id="cv-lanpaint-friction" type="number" min="0.01" max="100" step="0.1" value="15"></label>' +
            '<label>Prompt mode<select id="cv-lanpaint-prompt-mode" class="cv-select"><option value="Image First">Image First</option><option value="Prompt First">Prompt First</option></select></label>' +
            '<label>Blend overlap<input id="cv-lanpaint-blend-overlap" type="number" min="1" max="51" step="2" value="9"></label>' +
            '<label>Early stop<input id="cv-lanpaint-early-stop" type="number" min="0" max="64" value="1"></label>' +
            '<label>Inner threshold<input id="cv-lanpaint-inner-threshold" type="number" min="0" max="100" step="0.01" value="0"></label>' +
            '<label>Inner patience<input id="cv-lanpaint-inner-patience" type="number" min="0" max="64" value="1"></label>' +
            '</div></details>' +
            '<div id="cv-masked-edit-helper" class="cv-helper-text">Loading masked-edit capabilities...</div>' +
            '</div>' +
            '<div id="cv-edit-runtime-note" class="cv-capability-note" style="display:none"></div>' +
            '<label class="cv-setting-label" style="margin-bottom:2px">Prompt</label>' +
            '<textarea id="cv-prompt" class="cv-textarea" rows="3" placeholder="Describe the content..."></textarea>' +
            '<label class="cv-setting-label" style="margin-bottom:2px">Negative</label>' +
            '<textarea id="cv-negative" class="cv-textarea cv-textarea-compact" rows="2" placeholder="Content to avoid..."></textarea>' +
            '<div id="cv-denoise-row" class="cv-setting-row" style="margin-top:4px">' +
            '<span class="cv-setting-label">Denoise</span>' +
            '<input type="range" id="cv-denoise" class="cv-range" min="0" max="1" step="0.01" value="0.75">' +
            '<span id="cv-denoise-val" class="cv-setting-value">0.75</span>' +
            '</div>' +
            '<div id="cv-denoise-help" class="cv-helper-text">Low = subtle changes &middot; High = full reimagining</div>' +
            '<div class="cv-setting-row">' +
            '<span class="cv-setting-label">Steps</span>' +
            '<input type="number" id="cv-steps" class="cv-number-input" min="1" max="150" value="20">' +
            '<input type="range" id="cv-steps-range" class="cv-range" min="1" max="150" value="20">' +
            '</div>' +
            '<div id="cv-cfg-row" class="cv-setting-row">' +
            '<span class="cv-setting-label">CFG</span>' +
            '<input type="number" id="cv-cfg" class="cv-number-input" min="0" max="20" step="0.5" value="7.0">' +
            '<input type="range" id="cv-cfg-range" class="cv-range" min="0" max="20" step="0.5" value="7.0">' +
            '</div>' +
            '<div id="cv-guidance-row" class="cv-setting-row" style="display:none">' +
            '<span class="cv-setting-label">Guidance</span>' +
            '<input type="number" id="cv-guidance" class="cv-number-input" min="1" max="10" step="0.5" value="3.5">' +
            '<input type="range" id="cv-guidance-range" class="cv-range" min="1" max="10" step="0.5" value="3.5">' +
            '</div>' +
            '<div class="cv-setting-row"><span class="cv-setting-label">Sampler</span>' +
            '<select id="cv-sampler" class="cv-select"><option value="euler">Euler</option><option value="euler_ancestral">Euler Ancestral</option><option value="dpmpp_2m">DPM++ 2M</option></select></div>' +
            '<div class="cv-setting-row"><span class="cv-setting-label">Scheduler</span>' +
            '<select id="cv-scheduler" class="cv-select"><option value="simple">Simple</option><option value="normal">Normal</option><option value="karras">Karras</option></select></div>' +
            '<div class="cv-setting-row"><span class="cv-setting-label">Seed</span>' +
            '<input type="number" id="cv-seed" class="cv-number-input cv-seed-input" min="-1" max="4294967295" value="-1" title="-1 chooses a random seed">' +
            '<span id="cv-batch-label" class="cv-setting-label">Batch</span><input type="number" id="cv-batch" class="cv-number-input" min="1" max="8" value="1"></div>' +
            '<div id="cv-video-section" style="display:none">' +
            '<div class="cv-section-title" style="margin-top:8px">Video</div>' +
            '<div class="cv-setting-row">' +
            '<span class="cv-setting-label">Frames</span>' +
            '<input type="number" id="cv-frames" class="cv-number-input" min="9" max="257" step="8" value="97">' +
            '<input type="range" id="cv-frames-range" class="cv-range" min="9" max="257" step="8" value="97">' +
            '</div>' +
            '<div class="cv-setting-row">' +
            '<span class="cv-setting-label">FPS</span>' +
            '<input type="number" id="cv-fps" class="cv-number-input" min="8" max="60" value="24">' +
            '<input type="range" id="cv-fps-range" class="cv-range" min="8" max="60" value="24">' +
            '</div>' +
            '<div id="cv-duration-hint" class="cv-duration-hint"></div>' +
            '<div id="cv-ltx2-section" class="cv-ltx2-section" style="display:none">' +
            '<div class="cv-section-title" style="margin-top:8px">LTX2 Mojo request</div>' +
            '<label class="cv-setting-label cv-path-label" for="cv-caps-positive">Conditioning</label>' +
            '<input id="cv-caps-positive" class="cv-path-input" type="text" placeholder="Server path to prompt-matched conditioning JSON or safetensors">' +
            '<label class="cv-setting-label cv-path-label" for="cv-caps-negative">Negative conditioning</label>' +
            '<input id="cv-caps-negative" class="cv-path-input" type="text" placeholder="Optional when the sidecar contains negative conditioning">' +
            '<label class="cv-setting-label cv-path-label" for="cv-noise-fixture">Noise fixture</label>' +
            '<input id="cv-noise-fixture" class="cv-path-input" type="text" placeholder="Optional server path; blank uses the authored seed">' +
            '<label class="cv-check-row"><input id="cv-include-audio" type="checkbox"> Decode and mux audio</label>' +
            '<button id="cv-load-ltx2-template" class="cv-import-btn" type="button">Load verified LTX2 + LoRA template</button>' +
            '<div class="cv-helper-text">The Mojo runner validates these exact values and rejects unsupported compiled profiles before model loading.</div>' +
            '</div>' +
            '</div>' +
            '<div id="cv-model-row"><label class="cv-setting-label" style="margin-top:4px">Model</label>' +
            '<select id="cv-model" class="cv-select"><option disabled selected>Loading models...</option></select></div>' +
            '<div id="cv-capability-note" class="cv-capability-note" style="display:none"></div>' +
            '<div id="cv-lora-section" class="cv-lora-section">' +
            '<div class="cv-lora-header"><span class="cv-section-title">LoRA loaders</span><span id="cv-lora-compat" class="cv-lora-compat"></span></div>' +
            '<div class="cv-lora-add-row"><select id="cv-lora-picker" class="cv-select"><option disabled selected>Loading LoRAs...</option></select>' +
            '<button id="cv-lora-add" class="cv-option-toggle" type="button">Add</button></div>' +
            '<div id="cv-lora-list" class="cv-lora-list"></div>' +
            '</div>' +
            '<hr class="cv-separator">' +
            '<button id="cv-import-btn" class="cv-import-btn">Import Image</button>' +
            '<input type="file" id="cv-import-file" accept="image/*,video/*" style="display:none">' +
            '<button id="cv-generate-btn" class="cv-generate-btn">Generate</button>' +
            '<div id="cv-progress" class="cv-progress"><div id="cv-progress-bar" class="cv-progress-bar"></div></div>' +
            '<div id="cv-progress-label" class="cv-progress-label"></div>' +
            '<div id="cv-error-banner" class="cv-error-banner"></div>' +
            '</div>';
    }
    function cacheElements() {
        els.layerList = document.getElementById('cv-layer-list');
        els.brushSection = document.getElementById('cv-brush-section');
        els.brushSizeInput = document.getElementById('cv-brush-size');
        els.brushSizeVal = document.getElementById('cv-brush-size-val');
        els.brushColorInput = document.getElementById('cv-brush-color');
        els.prompt = document.getElementById('cv-prompt');
        els.editMode = document.getElementById('cv-edit-mode');
        els.editEngine = document.getElementById('cv-edit-engine');
        els.editSourcePrompt = document.getElementById('cv-edit-source-prompt');
        els.editNmax = document.getElementById('cv-edit-nmax');
        els.editNmin = document.getElementById('cv-edit-nmin');
        els.editSourceCfg = document.getElementById('cv-edit-src-cfg');
        els.editAutoMask = document.getElementById('cv-edit-auto-mask');
        els.editMaskQ = document.getElementById('cv-edit-mask-q');
        els.editMaskDilate = document.getElementById('cv-edit-mask-dilate');
        els.editMaskWarmup = document.getElementById('cv-edit-mask-warmup');
        els.flowEditSection = document.getElementById('cv-flowedit-section');
        els.lanpaintSection = document.getElementById('cv-lanpaint-section');
        els.lanpaintEngine = document.getElementById('cv-lanpaint-engine');
        els.lanpaintControls = document.getElementById('cv-lanpaint-controls');
        els.maskedEditHelper = document.getElementById('cv-masked-edit-helper');
        els.lanpaintNumSteps = document.getElementById('cv-lanpaint-inner-steps');
        els.lanpaintLambda = document.getElementById('cv-lanpaint-lambda');
        els.lanpaintStepSize = document.getElementById('cv-lanpaint-step-size');
        els.lanpaintBeta = document.getElementById('cv-lanpaint-beta');
        els.lanpaintFriction = document.getElementById('cv-lanpaint-friction');
        els.lanpaintPromptMode = document.getElementById('cv-lanpaint-prompt-mode');
        els.lanpaintBlendOverlap = document.getElementById('cv-lanpaint-blend-overlap');
        els.lanpaintEarlyStop = document.getElementById('cv-lanpaint-early-stop');
        els.lanpaintInnerThreshold = document.getElementById('cv-lanpaint-inner-threshold');
        els.lanpaintInnerPatience = document.getElementById('cv-lanpaint-inner-patience');
        els.editRuntimeNote = document.getElementById('cv-edit-runtime-note');
        els.editWorkspace = document.getElementById('cv-edit-workspace');
        els.sourcePane = document.getElementById('cv-source-pane');
        els.sourceBrowse = document.getElementById('cv-source-browse');
        els.sourceDropzone = document.getElementById('cv-source-dropzone');
        els.sourcePreview = document.getElementById('cv-source-preview');
        els.sourceVideo = document.getElementById('cv-source-video');
        els.sourceEmpty = document.getElementById('cv-source-empty');
        els.styleSourceSlot = document.getElementById('cv-style-source-slot');
        els.styleBrowse = document.getElementById('cv-style-browse');
        els.styleDropzone = document.getElementById('cv-style-dropzone');
        els.stylePreview = document.getElementById('cv-style-preview');
        els.styleEmpty = document.getElementById('cv-style-empty');
        els.styleFile = document.getElementById('cv-style-file');
        els.styleEntireImage = document.getElementById('cv-style-entire-image');
        els.resultPaneTitle = document.getElementById('cv-result-pane-title');
        els.negative = document.getElementById('cv-negative');
        els.denoise = document.getElementById('cv-denoise');
        els.denoiseVal = document.getElementById('cv-denoise-val');
        els.denoiseRow = document.getElementById('cv-denoise-row');
        els.denoiseHelp = document.getElementById('cv-denoise-help');
        els.steps = document.getElementById('cv-steps');
        els.stepsRange = document.getElementById('cv-steps-range');
        els.cfgRow = document.getElementById('cv-cfg-row');
        els.cfg = document.getElementById('cv-cfg');
        els.cfgRange = document.getElementById('cv-cfg-range');
        els.guidanceRow = document.getElementById('cv-guidance-row');
        els.guidance = document.getElementById('cv-guidance');
        els.guidanceRange = document.getElementById('cv-guidance-range');
        els.sampler = document.getElementById('cv-sampler');
        els.scheduler = document.getElementById('cv-scheduler');
        els.seed = document.getElementById('cv-seed');
        els.batch = document.getElementById('cv-batch');
        els.batchLabel = document.getElementById('cv-batch-label');
        els.videoSection = document.getElementById('cv-video-section');
        els.framesInput = document.getElementById('cv-frames');
        els.framesRange = document.getElementById('cv-frames-range');
        els.fpsInput = document.getElementById('cv-fps');
        els.fpsRange = document.getElementById('cv-fps-range');
        els.durationHint = document.getElementById('cv-duration-hint');
        els.ltx2Section = document.getElementById('cv-ltx2-section');
        els.capsPositive = document.getElementById('cv-caps-positive');
        els.capsNegative = document.getElementById('cv-caps-negative');
        els.noiseFixture = document.getElementById('cv-noise-fixture');
        els.includeAudio = document.getElementById('cv-include-audio');
        els.loadLtx2Template = document.getElementById('cv-load-ltx2-template');
        els.modelRow = document.getElementById('cv-model-row');
        els.model = document.getElementById('cv-model');
        els.capabilityNote = document.getElementById('cv-capability-note');
        els.loraSection = document.getElementById('cv-lora-section');
        els.loraPicker = document.getElementById('cv-lora-picker');
        els.loraAdd = document.getElementById('cv-lora-add');
        els.loraList = document.getElementById('cv-lora-list');
        els.loraCompat = document.getElementById('cv-lora-compat');
        els.importBtn = document.getElementById('cv-import-btn');
        els.importFile = document.getElementById('cv-import-file');
        els.generateBtn = document.getElementById('cv-generate-btn');
        els.progress = document.getElementById('cv-progress');
        els.progressBar = document.getElementById('cv-progress-bar');
        els.progressLabel = document.getElementById('cv-progress-label');
        els.errorBanner = document.getElementById('cv-error-banner');
    }
    // ── Konva Stage ──
    function initKonva() {
        var container = document.getElementById('canvas-stage-container');
        if (!container)
            return;
        var w = container.offsetWidth;
        var h = container.offsetHeight;
        // Bail if layout hasn't happened yet — resize() will fix it
        if (w < 100 || h < 100) {
            w = 800;
            h = 600;
        }
        stage = new Konva.Stage({
            container: 'canvas-stage-container',
            width: w,
            height: h,
            draggable: false
        });
        // Background layer
        backgroundLayer = new Konva.Layer({ listening: false });
        stage.add(backgroundLayer);
        var checkerImg = createCheckerboardImage();
        checkerImg.onload = function () {
            bgRect = new Konva.Rect({
                x: -10000, y: -10000,
                width: 20000, height: 20000,
                fillPatternImage: checkerImg,
                fillPatternRepeat: 'repeat',
                listening: false
            });
            backgroundLayer.add(bgRect);
            backgroundLayer.batchDraw();
        };
        // UI layer (will be moved to top after raster layers)
        uiLayer = new Konva.Layer();
        // Initial draw layer
        addLayer('Draw Layer', 'draw');
        // UI layer on top
        stage.add(uiLayer);
        // Bounding box
        initBoundingBox(w, h);
        // Brush cursor
        brushCursor = new Konva.Circle({
            x: 0, y: 0,
            radius: brushSize / 2,
            stroke: '#6c6af5',
            strokeWidth: 1.5,
            visible: false,
            listening: false
        });
        uiLayer.add(brushCursor);
        setupStageEvents();
        uiLayer.batchDraw();
        konvaReady = true;
    }
    // ── Bounding Box ──
    function initBoundingBox(stageW, stageH) {
        var bw = 1024;
        var bh = 1024;
        var cx = Math.round(stageW / 2 - bw / 2);
        var cy = Math.round(stageH / 2 - bh / 2);
        boundingBox = new Konva.Rect({
            x: cx, y: cy,
            width: bw, height: bh,
            stroke: '#6c6af5',
            strokeWidth: 2,
            dash: [10, 5],
            fill: 'rgba(108,106,245,0.05)',
            draggable: true,
            name: 'bounding-box',
            shadowColor: '#6c6af5',
            shadowBlur: 6,
            shadowOpacity: 0.3,
            shadowEnabled: true
        });
        uiLayer.add(boundingBox);
        sizeLabel = new Konva.Text({
            x: cx, y: cy + bh + 8,
            text: bw + ' \u00d7 ' + bh,
            fontSize: 12,
            fontFamily: 'Inter, system-ui, sans-serif',
            fill: '#6c6af5',
            opacity: 0.9,
            listening: false,
            padding: 2
        });
        uiLayer.add(sizeLabel);
        boundingBox.on('dragmove', function () {
            var snap = isVideoArch() ? 32 : 64;
            boundingBox.x(Math.round(boundingBox.x() / snap) * snap);
            boundingBox.y(Math.round(boundingBox.y() / snap) * snap);
            updateHandles();
            updateSizeLabel();
            updateVideoOverlayPosition();
        });
        boundingBox.on('dragend', function () {
            History.push();
        });
        createResizeHandles();
    }
    function createResizeHandles() {
        resizeHandles.forEach(function (h) { h.destroy(); });
        resizeHandles = [];
        var positions = [
            { name: 'tl', cursor: 'nwse-resize' },
            { name: 'tc', cursor: 'ns-resize' },
            { name: 'tr', cursor: 'nesw-resize' },
            { name: 'ml', cursor: 'ew-resize' },
            { name: 'mr', cursor: 'ew-resize' },
            { name: 'bl', cursor: 'nesw-resize' },
            { name: 'bc', cursor: 'ns-resize' },
            { name: 'br', cursor: 'nwse-resize' }
        ];
        var hs = HANDLE_SIZE;
        var half = hs / 2;
        positions.forEach(function (p) {
            var handle = new Konva.Rect({
                width: hs, height: hs,
                fill: '#6c6af5',
                stroke: '#ffffff',
                strokeWidth: 1.5,
                cornerRadius: 2,
                draggable: true,
                name: 'handle-' + p.name
            });
            handle.on('mouseenter', function () {
                stage.container().style.cursor = p.cursor;
            });
            handle.on('mouseleave', function () {
                if (!activeHandle)
                    updateCursor();
            });
            handle.on('dragstart', function () {
                activeHandle = p.name;
                handleStartBox = {
                    x: boundingBox.x(), y: boundingBox.y(),
                    w: boundingBox.width(), h: boundingBox.height()
                };
                handleStartMouse = getRelativePointerPosition();
            });
            handle.on('dragmove', function () {
                handle.position(handle.position());
                var pos = getRelativePointerPosition();
                var dx = pos.x - handleStartMouse.x;
                var dy = pos.y - handleStartMouse.y;
                var newX = handleStartBox.x;
                var newY = handleStartBox.y;
                var newW = handleStartBox.w;
                var newH = handleStartBox.h;
                var nm = activeHandle;
                if (nm.indexOf('l') >= 0) {
                    newX = handleStartBox.x + dx;
                    newW = handleStartBox.w - dx;
                }
                else if (nm.indexOf('r') >= 0) {
                    newW = handleStartBox.w + dx;
                }
                if (nm.indexOf('t') >= 0) {
                    newY = handleStartBox.y + dy;
                    newH = handleStartBox.h - dy;
                }
                else if (nm.indexOf('b') >= 0) {
                    newH = handleStartBox.h + dy;
                }
                // Aspect ratio lock enforcement
                if (bboxAspectLocked && bboxLockedRatio > 0) {
                    var isCorner = nm === 'tl' || nm === 'tr' || nm === 'bl' || nm === 'br';
                    var isHorizontal = nm === 'ml' || nm === 'mr';
                    var isVertical = nm === 'tc' || nm === 'bc';
                    if (isCorner) {
                        // Use the larger change to determine the other
                        if (Math.abs(dx) > Math.abs(dy)) {
                            newH = Math.round(newW / bboxLockedRatio);
                        } else {
                            newW = Math.round(newH * bboxLockedRatio);
                        }
                    } else if (isHorizontal) {
                        newH = Math.round(newW / bboxLockedRatio);
                    } else if (isVertical) {
                        newW = Math.round(newH * bboxLockedRatio);
                    }
                }
                newW = clampDimForArch(newW);
                newH = clampDimForArch(newH);
                if (nm.indexOf('l') >= 0)
                    newX = handleStartBox.x + handleStartBox.w - newW;
                if (nm.indexOf('t') >= 0)
                    newY = handleStartBox.y + handleStartBox.h - newH;
                boundingBox.x(newX);
                boundingBox.y(newY);
                boundingBox.width(newW);
                boundingBox.height(newH);
                updateHandles();
                updateSizeLabel();
                updateVideoOverlayPosition();
            });
            handle.on('dragend', function () {
                activeHandle = null;
                handleStartBox = null;
                handleStartMouse = null;
                updateCursor();
                History.push();
            });
            uiLayer.add(handle);
            resizeHandles.push(handle);
        });
        updateHandles();
    }
    function updateHandles() {
        if (!boundingBox)
            return;
        var bx = boundingBox.x();
        var by = boundingBox.y();
        var bw = boundingBox.width();
        var bh = boundingBox.height();
        var half = HANDLE_SIZE / 2;
        var coords = [
            { x: bx - half, y: by - half }, // tl
            { x: bx + bw / 2 - half, y: by - half }, // tc
            { x: bx + bw - half, y: by - half }, // tr
            { x: bx - half, y: by + bh / 2 - half }, // ml
            { x: bx + bw - half, y: by + bh / 2 - half }, // mr
            { x: bx - half, y: by + bh - half }, // bl
            { x: bx + bw / 2 - half, y: by + bh - half }, // bc
            { x: bx + bw - half, y: by + bh - half } // br
        ];
        resizeHandles.forEach(function (h, i) {
            h.x(coords[i].x);
            h.y(coords[i].y);
        });
    }
    function updateSizeLabel() {
        if (!sizeLabel || !boundingBox)
            return;
        var bw = boundingBox.width();
        var bh = boundingBox.height();
        var label = bw + ' \u00d7 ' + bh;
        if (isVideoArch()) {
            label += ' \u00b7 ' + genState.frames + 'f';
        }
        sizeLabel.text(label);
        // Center label under bbox
        sizeLabel.x(boundingBox.x() + bw / 2 - sizeLabel.width() / 2);
        sizeLabel.y(boundingBox.y() + bh + 8);
        uiLayer.batchDraw();
        updateBboxInputs();
    }
    function updateBboxInputs() {
        var bboxW = document.getElementById('cv-bbox-w');
        var bboxH = document.getElementById('cv-bbox-h');
        if (bboxW && boundingBox)
            bboxW.value = String(Math.round(boundingBox.width()));
        if (bboxH && boundingBox)
            bboxH.value = String(Math.round(boundingBox.height()));
    }
    // ── Stage Events ──
    function setupStageEvents() {
        var container = stage.container();
        // Pan: middle mouse OR space+drag OR pan tool
        container.addEventListener('mousedown', function (e) {
            if (e.button === 1 || (e.button === 0 && (isSpaceHeld || activeTool === 'pan'))) {
                isPanning = true;
                panStart.x = e.clientX;
                panStart.y = e.clientY;
                stageStart.x = stage.x();
                stageStart.y = stage.y();
                container.style.cursor = 'grabbing';
                e.preventDefault();
            }
        });
        container.addEventListener('mousemove', function (e) {
            if (isPanning) {
                stage.position({
                    x: stageStart.x + (e.clientX - panStart.x),
                    y: stageStart.y + (e.clientY - panStart.y)
                });
                stage.batchDraw();
                updateVideoOverlayPosition();
            }
        });
        container.addEventListener('mouseup', function () {
            if (isPanning) {
                isPanning = false;
                updateCursor();
            }
        });
        container.addEventListener('auxclick', function (e) {
            if (e.button === 1)
                e.preventDefault();
        });
        // Zoom / Alt+scroll opacity
        stage.on('wheel', function (e) {
            e.evt.preventDefault();
            // Alt+scroll: adjust brush opacity
            if (e.evt.altKey) {
                var delta = e.evt.deltaY < 0 ? 0.05 : -0.05;
                var newOp = Math.max(0, Math.min(1, CanvasTools.getBrushOpacity() + delta));
                CanvasTools.setBrushOpacity(newOp);
                return;
            }
            var scaleBy = 1.08;
            var oldScale = stage.scaleX();
            var pointer = stage.getPointerPosition();
            if (!pointer)
                return;
            var newScale = e.evt.deltaY < 0 ? oldScale * scaleBy : oldScale / scaleBy;
            var clampedScale = Math.min(Math.max(newScale, 0.1), 10);
            var mousePointTo = {
                x: (pointer.x - stage.x()) / oldScale,
                y: (pointer.y - stage.y()) / oldScale
            };
            stage.scale({ x: clampedScale, y: clampedScale });
            stage.position({
                x: pointer.x - mousePointTo.x * clampedScale,
                y: pointer.y - mousePointTo.y * clampedScale
            });
            stage.batchDraw();
            updateVideoOverlayPosition();
            if (typeof CanvasStatusBar !== 'undefined')
                CanvasStatusBar.updateZoom(Math.round(clampedScale * 100));
        });
        // Tool event delegation
        stage.on('mousedown', function (e) {
            if (isPanning || isSpaceHeld)
                return;
            if (typeof CanvasStaging !== 'undefined' && CanvasStaging.isActive() && CanvasStaging.isPartialMaskMode()) {
                CanvasStaging.handlePartialMouseDown(getToolContext(), getRelativePointerPosition());
                return;
            }
            var tool = CanvasTools.get(activeTool);
            if (tool && tool.onMouseDown) {
                tool.onMouseDown(getToolContext(), e);
            }
        });
        stage.on('mousemove', function () {
            var tool = CanvasTools.get(activeTool);
            var pos = getRelativePointerPosition();
            // Update status bar cursor position
            if (typeof CanvasStatusBar !== 'undefined') {
                CanvasStatusBar.updateCursor(pos.x, pos.y);
            }
            if (typeof CanvasStaging !== 'undefined' && CanvasStaging.isActive() && CanvasStaging.isPartialMaskMode()) {
                CanvasStaging.handlePartialMouseMove(pos);
                return;
            }
            // Update brush cursor for tools that show it
            if (tool && tool.showsBrushCursor && brushCursor) {
                brushCursor.x(pos.x);
                brushCursor.y(pos.y);
                brushCursor.radius(brushSize / 2 / stage.scaleX());
                brushCursor.strokeWidth(1.5 / stage.scaleX());
                if (!brushCursor.visible())
                    brushCursor.visible(true);
                uiLayer.batchDraw();
            }
            if (tool && tool.onMouseMove) {
                tool.onMouseMove(getToolContext(), pos);
            }
        });
        stage.on('mouseup', function () {
            if (typeof CanvasStaging !== 'undefined' && CanvasStaging.isActive() && CanvasStaging.isPartialMaskMode()) {
                CanvasStaging.handlePartialMouseUp();
                return;
            }
            var tool = CanvasTools.get(activeTool);
            if (tool && tool.onMouseUp) {
                tool.onMouseUp(getToolContext());
            }
        });
        stage.on('mouseleave', function () {
            // End any active drawing
            var tool = CanvasTools.get(activeTool);
            if (tool && tool.onMouseUp) {
                tool.onMouseUp(getToolContext());
            }
            if (brushCursor) {
                brushCursor.visible(false);
                uiLayer.batchDraw();
            }
        });
        // Drag and drop
        container.addEventListener('dragover', function (e) { e.preventDefault(); });
        container.addEventListener('drop', function (e) {
            e.preventDefault();
            var file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
            if (!file || !file.type.startsWith('image/'))
                return;
            if (genState.editMode === 'create')
                loadImageFile(file);
            else
                loadEditSourceFile(file);
        });
        // Right-click context menu
        container.addEventListener('contextmenu', function (e) {
            e.preventDefault();
            if (typeof CanvasContextMenu !== 'undefined' && stage) {
                CanvasContextMenu.show(e.clientX, e.clientY, getToolContext());
            }
        });
    }
    // ── Image loading ──
    function loadImageFile(file) {
        var reader = new FileReader();
        reader.onload = function (ev) {
            editSourceFile = file;
            editSourceDataUrl = String(ev.target.result || '');
            editSourceIsVideo = false;
            genState.editSourcePrompt = '';
            if (els.editSourcePrompt)
                els.editSourcePrompt.value = '';
            if (els.sourceEmpty)
                els.sourceEmpty.style.display = 'none';
            if (els.sourceVideo) {
                els.sourceVideo.removeAttribute('src');
                els.sourceVideo.style.display = 'none';
            }
            if (els.sourcePreview) {
                els.sourcePreview.src = editSourceDataUrl;
                els.sourcePreview.style.display = 'block';
            }
            addImageDataUrlToCanvas(editSourceDataUrl);
        };
        reader.readAsDataURL(file);
    }
    function addImageDataUrlToCanvas(dataUrl) {
        var img = new Image();
        img.onload = function () {
            var bw = boundingBox.width();
            var bh = boundingBox.height();
            var iw = img.naturalWidth || img.width;
            var ih = img.naturalHeight || img.height;
            var scale = Math.min(bw / iw, bh / ih);
            var dw = iw * scale;
            var dh = ih * scale;
            var dx = boundingBox.x() + (bw - dw) / 2;
            var dy = boundingBox.y() + (bh - dh) / 2;
            var kImg = new Konva.Image({
                image: img, x: dx, y: dy, width: dw, height: dh,
                draggable: activeTool === 'select'
            });
            var layer = getActiveKonvaLayer();
            if (layer) {
                layer.add(kImg);
                layer.batchDraw();
                History.push();
                if (genState.editMode !== 'create')
                    requestAnimationFrame(resetView);
            }
        };
        img.src = dataUrl;
    }
    function loadEditSourceFile(file) {
        if (!file)
            return;
        var isVideo = file.type.startsWith('video/');
        if (isVideo && genState.editMode !== 'dynaedit') {
            showError('Image edit and inpaint require an image source');
            return;
        }
        if (!isVideo && !file.type.startsWith('image/')) {
            showError('Choose an image or video file');
            return;
        }
        var reader = new FileReader();
        reader.onload = function (ev) {
            editSourceFile = file;
            editSourceDataUrl = String(ev.target.result || '');
            editSourceIsVideo = isVideo;
            genState.editSourcePrompt = '';
            els.editSourcePrompt.value = '';
            els.sourceEmpty.style.display = 'none';
            if (isVideo) {
                els.sourcePreview.removeAttribute('src');
                els.sourcePreview.style.display = 'none';
                els.sourceVideo.src = editSourceDataUrl;
                els.sourceVideo.style.display = 'block';
            }
            else {
                prefer1024FlowEditForNewImage();
                els.sourceVideo.removeAttribute('src');
                els.sourceVideo.style.display = 'none';
                els.sourcePreview.src = editSourceDataUrl;
                els.sourcePreview.style.display = 'block';
                addImageDataUrlToCanvas(editSourceDataUrl);
            }
        };
        reader.readAsDataURL(file);
    }
    function setStyleReference(ref) {
        if (!ref || !ref.src)
            return;
        styleReferenceDataUrl = String(ref.src);
        styleReferenceId = String(ref.id || '');
        els.styleEmpty.style.display = 'none';
        els.stylePreview.src = styleReferenceDataUrl;
        els.stylePreview.style.display = 'block';
    }
    function loadStyleReferenceFile(file) {
        if (!file)
            return;
        if (!file.type.startsWith('image/')) {
            showError('Choose an image for the style reference');
            return;
        }
        var reader = new FileReader();
        reader.onload = function (event) {
            var dataUrl = String(event.target.result || '');
            if (typeof CanvasRefImages !== 'undefined')
                CanvasRefImages.add(dataUrl);
            else
                enterStyleMode({ id: '', src: dataUrl, method: 'style' });
        };
        reader.readAsDataURL(file);
    }
    function captureCurrentCanvasAsStyleSource() {
        if (!stage || !boundingBox)
            return Promise.resolve(false);
        return checkBboxContent().then(function (hasContent) {
            if (!hasContent)
                return false;
            return exportBoundingBoxRegion().then(function (base64) {
                editSourceFile = null;
                editSourceDataUrl = 'data:image/png;base64,' + base64;
                editSourceIsVideo = false;
                genState.editSourcePrompt = '';
                els.editSourcePrompt.value = '';
                els.sourceEmpty.style.display = 'none';
                els.sourceVideo.removeAttribute('src');
                els.sourceVideo.style.display = 'none';
                els.sourcePreview.src = editSourceDataUrl;
                els.sourcePreview.style.display = 'block';
                return true;
            });
        });
    }
    function enterStyleMode(ref) {
        if (ref)
            setStyleReference(ref);
        var entering = genState.editMode !== 'style';
        // Preserve the exact imported/gallery image when one is available. A
        // bbox composite is only the fallback for an authored Canvas that has
        // no selected source image of its own.
        var capture = entering && !editSourceDataUrl ? captureCurrentCanvasAsStyleSource() : Promise.resolve(!!editSourceDataUrl);
        return capture.finally(function () {
            if (editSourceDataUrl) {
                els.sourceEmpty.style.display = 'none';
                els.sourceVideo.removeAttribute('src');
                els.sourceVideo.style.display = 'none';
                els.sourcePreview.src = editSourceDataUrl;
                els.sourcePreview.style.display = 'block';
            }
            genState.editMode = 'style';
            var previousEngine = genState.editEngine;
            genState.editEngine = normalizeFlowEditEngine(genState.editEngine);
            if (isKreaFlowEditEngine(genState.editEngine))
                genState.editEngine = upgradeKreaFlowEditEngineTo1024(genState.editEngine);
            else if (genState.editEngine !== 'ideogram4')
                genState.editEngine = 'ideogram4';
            var engineChanged = previousEngine !== genState.editEngine;
            if (els.editMode)
                els.editMode.value = 'style';
            if (els.editEngine)
                els.editEngine.value = genState.editEngine;
            if (engineChanged)
                applyFlowEditEngineProfile(genState.editEngine);
            syncCanvasModelFromFlowEditEngine();
            updateEditWorkspace();
        });
    }
    function applyFlowEditEngineProfile(engine) {
        // These are visible, editable presets, not hidden request overrides.
        // Turbo keeps the creator's distilled 8-step, CFG-off sampling contract.
        // Raw and Ideogram keep their non-distilled FlowEdit schedules.
        var profile = engine === 'ideogram4' ?
            { steps: 28, cfg: 5.0, nmax: 24, nmin: 0, srcCfg: 1.5, warmup: 4 } :
            (isKreaTurboFlowEditEngine(engine) ?
                { steps: 8, cfg: 0.0, nmax: 8, nmin: 0, srcCfg: 0.0, warmup: 1 } :
                { steps: 28, cfg: 5.5, nmax: 24, nmin: 0, srcCfg: 1.5, warmup: 4 });
        genState.steps = profile.steps;
        genState.cfg = profile.cfg;
        genState.editNmax = profile.nmax;
        genState.editNmin = profile.nmin;
        genState.editSourceCfg = profile.srcCfg;
        genState.editMaskWarmup = profile.warmup;
        els.steps.value = String(profile.steps);
        els.stepsRange.value = String(profile.steps);
        els.cfg.value = String(profile.cfg);
        els.cfgRange.value = String(profile.cfg);
        els.editNmax.value = String(profile.nmax);
        els.editNmin.value = String(profile.nmin);
        els.editSourceCfg.value = String(profile.srcCfg);
        els.editMaskWarmup.value = String(profile.warmup);
    }
    function prefer1024FlowEditForNewImage() {
        // New gallery/import/drop edit sources use the production 1024 profile.
        // The 512 profile remains selectable afterward for an intentional
        // bounded request, but stale saved/default state must not shrink a new
        // source behind the user's back.
        var normalized = normalizeFlowEditEngine(genState.editEngine);
        if (isKreaFlowEditEngine(normalized)) {
            genState.editEngine = upgradeKreaFlowEditEngineTo1024(normalized);
            if (els.editEngine)
                els.editEngine.value = genState.editEngine;
            if (normalized !== genState.editEngine)
                applyFlowEditEngineProfile(genState.editEngine);
        }
        if (boundingBox) {
            boundingBox.width(STYLE_RESULT_SIZE);
            boundingBox.height(STYLE_RESULT_SIZE);
            updateHandles();
            updateSizeLabel();
            updateBboxInputs();
            stage.batchDraw();
        }
    }
    function syncCanvasModelFromLanPaintEngine() {
        var definition = maskedEditEngineDefinition(genState.lanpaintEngine);
        var match = findCanvasModelForMaskedEditEngine(definition);
        if (!definition || !match)
            return false;
        els.model.value = match.option.value;
        genState.model = match.option.value;
        updateTopbarModel(match.option.value);
        updateCanvasUIForArch(definition.arch);
        return true;
    }
    function selectFirstCanvasModelForArch(arch) {
        if (!els.model)
            return false;
        var options = Array.from(els.model.options || []);
        var match = options.find(function (option) {
            return ModelUtils.detectArchFromFilename(option.value) === arch;
        });
        if (!match)
            return false;
        els.model.value = match.value;
        genState.model = match.value;
        updateTopbarModel(match.value);
        updateCanvasUIForArch(arch);
        return true;
    }
    function updateEditWorkspace() {
        var mode = genState.editMode;
        var editing = mode !== 'create';
        var flowEditing = mode === 'flowedit' || mode === 'style';
        var engineDrivenEditing = flowEditing || mode === 'inpaint';
        var maskedEditEngine = maskedEditEngineDefinition(genState.lanpaintEngine);
        els.editWorkspace.classList.toggle('edit-active', editing);
        els.editWorkspace.classList.toggle('style-active', mode === 'style');
        els.sourcePane.style.display = editing ? 'flex' : 'none';
        els.flowEditSection.style.display = (mode === 'flowedit' || mode === 'style') ? 'block' : 'none';
        els.lanpaintSection.style.display = mode === 'inpaint' ? 'block' : 'none';
        if (els.modelRow)
            els.modelRow.style.display = engineDrivenEditing ? 'none' : 'block';
        els.denoise.disabled = mode === 'inpaint' && (!maskedEditEngine || maskedEditEngine.lanpaint);
        els.sampler.disabled = mode === 'inpaint';
        els.scheduler.disabled = mode === 'inpaint';
        els.editEngine.disabled = false;
        Array.from(els.editEngine.querySelectorAll('option[value$="_512"]')).forEach(function (option) {
            option.disabled = mode === 'style';
        });
        els.editRuntimeNote.style.display = 'none';
        els.resultPaneTitle.textContent = mode === 'inpaint' ? 'Result + mask' : (mode === 'style' ? 'Result · 1024 × 1024' : (editing ? 'Result' : 'Canvas'));
        els.importBtn.textContent = editing ? (mode === 'dynaedit' ? 'Load source video' : 'Load source image') : 'Import Image';
        els.importFile.accept = mode === 'dynaedit' ? 'video/*' : 'image/*';
        if (mode !== 'style')
            styleModeInitialized = false;
        if (mode === 'flowedit') {
            if (boundingBox) {
                var editSize = flowEditEngineSize(genState.editEngine);
                boundingBox.width(editSize);
                boundingBox.height(editSize);
                updateHandles();
                updateSizeLabel();
                updateBboxInputs();
                stage.batchDraw();
            }
            els.generateBtn.textContent = 'Run FlowEdit';
            els.editRuntimeNote.textContent = 'Pure-Mojo FlowEdit. Krea2 Raw and Turbo have compiled 512×512 and 1024×1024 profiles; Ideogram4 is 1024×1024.';
            els.editRuntimeNote.style.display = 'block';
        }
        else if (mode === 'style') {
            genState.editEngine = normalizeFlowEditEngine(genState.editEngine);
            if (isKreaFlowEditEngine(genState.editEngine))
                genState.editEngine = upgradeKreaFlowEditEngineTo1024(genState.editEngine);
            else if (genState.editEngine !== 'ideogram4')
                genState.editEngine = 'ideogram4';
            els.editEngine.value = genState.editEngine;
            syncCanvasModelFromFlowEditEngine();
            if (!styleModeInitialized && boundingBox) {
                boundingBox.width(STYLE_RESULT_SIZE);
                boundingBox.height(STYLE_RESULT_SIZE);
                updateHandles();
                updateSizeLabel();
                updateBboxInputs();
                stage.batchDraw();
                styleModeInitialized = true;
            }
            els.generateBtn.textContent = 'Apply reference style';
            els.editRuntimeNote.textContent = 'The pure-Mojo vision captioner analyzes the lower-left reference, then ' +
                flowEditEngineLabel(genState.editEngine) +
                ' FlowEdit applies that style to the upper-left source at 1024×1024. Entire image disables the localized change mask.';
            els.editRuntimeNote.style.display = 'block';
        }
        else if (mode === 'inpaint') {
            if (!lanpaintModeInitialized) {
                applyMaskedEditEngineProfile(genState.lanpaintEngine);
                lanpaintModeInitialized = true;
            }
            if (maskedEditEngine)
                els.lanpaintEngine.value = genState.lanpaintEngine;
            updateMaskedEditControlSurface(maskedEditEngine);
            if (boundingBox) {
                boundingBox.width(maskedEditEngine ? maskedEditEngine.width : 1024);
                boundingBox.height(maskedEditEngine ? maskedEditEngine.height : 1024);
                updateHandles();
                updateSizeLabel();
                updateBboxInputs();
                stage.batchDraw();
            }
            els.generateBtn.textContent = 'Edit masked area';
            if (!syncCanvasModelFromLanPaintEngine()) {
                els.editRuntimeNote.textContent = 'No registered model matches the selected masked-edit engine. The edit cannot start.';
                els.editRuntimeNote.style.display = 'block';
            }
            else {
                els.editRuntimeNote.textContent = maskedEditEngine && maskedEditEngine.lanpaint ?
                    'Mojo-native Krea2 LanPaint at 1024×1024. The painted mask selects the editable area; the final feathered blend preserves everything outside it.' :
                    'Mojo-native Z-Image masked edit at 1024×1024. Denoise controls the strength inside the painted mask while the unmasked source is preserved.';
                els.editRuntimeNote.style.display = 'block';
            }
        }
        else if (mode === 'dynaedit') {
            els.generateBtn.textContent = 'DynaEdit runner unavailable';
            els.editRuntimeNote.textContent = 'Pure-Mojo DynaEdit exists in lingbot_flowedit.mojo, but no web request runner is connected yet.';
            els.editRuntimeNote.style.display = 'block';
        }
        else {
            els.generateBtn.textContent = isVideoArch() ? 'Generate Video' : 'Generate';
        }
        if (canvasCapabilities)
            updateCanvasCapabilityUI();
        requestAnimationFrame(function () {
            resizeStage();
            positionCanvasToolRail();
            if (editing && stage)
                resetView();
        });
    }
    function positionCanvasToolRail() {
        var rail = document.querySelector('#panel-canvas .cv-tools');
        var resultPane = document.querySelector('#panel-canvas .cv-result-pane');
        if (!rail || !resultPane)
            return;
        var bounds = resultPane.getBoundingClientRect();
        if (bounds.width > 0)
            rail.style.left = Math.round(bounds.left + 10) + 'px';
    }
    // ── Layer Management ──
    function addLayer(name, type) {
        var layerType = migrateLayerType(type || 'draw');
        var data = LayerDefaults.createByType(layerType, name || undefined);
        var konvaLayer = new Konva.Layer();
        stage.add(konvaLayer);
        if (uiLayer && uiLayer.parent)
            uiLayer.moveToTop();
        // Text layers get a Konva.Text node
        if (data.type === 'text') {
            var td = data;
            var kText = new Konva.Text({
                x: boundingBox ? boundingBox.x() + 20 : 20,
                y: boundingBox ? boundingBox.y() + 20 : 20,
                text: td.text, fontSize: td.fontSize,
                fontFamily: td.fontFamily, fill: td.color,
                fontStyle: td.fontWeight, align: td.alignment,
                lineHeight: td.lineHeight, draggable: true,
            });
            konvaLayer.add(kText);
            attachInlineTextEdit(kText, data, konvaLayer);
        }
        var cl = { data: data, konvaLayer: konvaLayer };
        canvasLayers.push(cl);
        activeLayerId = data.id;
        renderLayerList();
        History.push();
        return cl;
    }
    function duplicateLayer(layerId) {
        var source = getLayerById(layerId);
        if (!source)
            return null;
        var newData = JSON.parse(JSON.stringify(source.data));
        newData.id = LayerDefaults.nextId();
        newData.name = source.data.name + ' (copy)';
        var konvaLayer = new Konva.Layer();
        stage.add(konvaLayer);
        if (uiLayer && uiLayer.parent)
            uiLayer.moveToTop();
        konvaLayer.opacity(newData.opacity);
        if (!newData.visible)
            konvaLayer.hide();
        var cl = { data: newData, konvaLayer: konvaLayer };
        // Copy pixel content
        var srcDataUrl = source.konvaLayer.toDataURL();
        if (srcDataUrl && srcDataUrl !== 'data:,') {
            var img = new Image();
            img.onload = function () {
                var kImg = new Konva.Image({ image: img, x: 0, y: 0 });
                konvaLayer.add(kImg);
                konvaLayer.batchDraw();
            };
            img.src = srcDataUrl;
        }
        // Insert above source
        var idx = canvasLayers.indexOf(source);
        canvasLayers.splice(idx + 1, 0, cl);
        activeLayerId = newData.id;
        // Reorder Konva layers to match
        syncKonvaOrder();
        renderLayerList();
        History.push();
        return cl;
    }
    function mergeDown(layerId) {
        var idx = -1;
        for (var i = 0; i < canvasLayers.length; i++) {
            if (canvasLayers[i].data.id === layerId) {
                idx = i;
                break;
            }
        }
        if (idx <= 0)
            return; // nothing below to merge into
        var upper = canvasLayers[idx];
        var lower = canvasLayers[idx - 1];
        // Only merge draw-type layers
        if (upper.data.type !== 'draw' || lower.data.type !== 'draw')
            return;
        // Rasterize upper onto lower's canvas
        var upperUrl = upper.konvaLayer.toDataURL();
        var img = new Image();
        img.onload = function () {
            var kImg = new Konva.Image({ image: img, x: 0, y: 0, opacity: upper.data.opacity });
            lower.konvaLayer.add(kImg);
            lower.konvaLayer.batchDraw();
            // Remove upper layer
            upper.konvaLayer.destroy();
            canvasLayers.splice(idx, 1);
            if (activeLayerId === upper.data.id)
                activeLayerId = lower.data.id;
            renderLayerList();
            History.push();
        };
        img.src = upperUrl;
    }
    function flattenVisible() {
        if (canvasLayers.length <= 1)
            return;
        // Create temp canvas to composite all visible layers
        var w = stage.width();
        var h = stage.height();
        var tmpCanvas = document.createElement('canvas');
        tmpCanvas.width = w;
        tmpCanvas.height = h;
        var ctx = tmpCanvas.getContext('2d');
        var drawPromises = [];
        canvasLayers.forEach(function (l) {
            if (!l.data.visible)
                return;
            if (l.data.type === 'adjustment')
                return; // skip non-raster
            drawPromises.push(new Promise(function (resolve) {
                var url = l.konvaLayer.toDataURL();
                if (!url || url === 'data:,') {
                    resolve();
                    return;
                }
                var img = new Image();
                img.onload = function () {
                    ctx.globalAlpha = l.data.opacity;
                    if (l.data.type === 'draw') {
                        ctx.globalCompositeOperation = BlendModeUtil.toCompositeOp(l.data.blendMode);
                    }
                    else {
                        ctx.globalCompositeOperation = 'source-over';
                    }
                    ctx.drawImage(img, 0, 0);
                    ctx.globalAlpha = 1;
                    ctx.globalCompositeOperation = 'source-over';
                    resolve();
                };
                img.onerror = function () { resolve(); };
                img.src = url;
            }));
        });
        Promise.all(drawPromises).then(function () {
            // Remove all layers
            canvasLayers.forEach(function (l) { l.konvaLayer.destroy(); });
            canvasLayers = [];
            // Create single flattened draw layer
            var data = LayerDefaults.draw('Flattened');
            var konvaLayer = new Konva.Layer();
            stage.add(konvaLayer);
            if (uiLayer && uiLayer.parent)
                uiLayer.moveToTop();
            var flatImg = new Image();
            flatImg.onload = function () {
                var kImg = new Konva.Image({ image: flatImg, x: 0, y: 0 });
                konvaLayer.add(kImg);
                konvaLayer.batchDraw();
            };
            flatImg.src = tmpCanvas.toDataURL();
            var cl = { data: data, konvaLayer: konvaLayer };
            canvasLayers.push(cl);
            activeLayerId = data.id;
            renderLayerList();
            History.push();
        });
    }
    function syncKonvaOrder() {
        canvasLayers.forEach(function (l) { l.konvaLayer.moveToBottom(); });
        if (backgroundLayer)
            backgroundLayer.moveToBottom();
        if (uiLayer)
            uiLayer.moveToTop();
        stage.batchDraw();
    }
    function renderLayerList() {
        if (!els.layerList)
            return;
        els.layerList.innerHTML = '';
        for (var i = canvasLayers.length - 1; i >= 0; i--) {
            var cl = canvasLayers[i];
            var d = cl.data;
            var row = document.createElement('div');
            row.className = 'cv-layer-row' + (d.id === activeLayerId ? ' active' : '') + (d.lockTransparency ? ' transparency-locked' : '');
            row.dataset.layerId = String(d.id);
            row.draggable = true;
            (function (dragCl, dragRow) {
                dragRow.addEventListener('dragstart', function (e) {
                    e.dataTransfer.setData('text/plain', String(dragCl.data.id));
                    dragRow.classList.add('cv-layer-dragging');
                });
                dragRow.addEventListener('dragend', function () {
                    dragRow.classList.remove('cv-layer-dragging');
                });
                dragRow.addEventListener('dragover', function (e) {
                    e.preventDefault();
                    dragRow.classList.add('cv-layer-dragover');
                });
                dragRow.addEventListener('dragleave', function () {
                    dragRow.classList.remove('cv-layer-dragover');
                });
                dragRow.addEventListener('drop', function (e) {
                    e.preventDefault();
                    dragRow.classList.remove('cv-layer-dragover');
                    var draggedId = parseInt(e.dataTransfer.getData('text/plain'));
                    var targetId = dragCl.data.id;
                    if (draggedId === targetId)
                        return;
                    reorderLayer(draggedId, targetId);
                });
            })(cl, row);
            var eyeBtn = document.createElement('button');
            eyeBtn.className = 'cv-layer-eye' + (d.visible ? '' : ' hidden-layer');
            eyeBtn.innerHTML = d.visible ? ICONS.eye : ICONS.eyeOff;
            eyeBtn.dataset.layerId = String(d.id);
            var badge = document.createElement('span');
            var badgeLabel = LAYER_TYPE_LABELS[d.type] || d.type;
            badge.className = 'cv-layer-badge ' + d.type;
            badge.textContent = badgeLabel.toUpperCase();
            if (d.lockTransparency)
                badge.title = 'Transparency locked';
            var nameSpan = document.createElement('span');
            nameSpan.className = 'cv-layer-name';
            nameSpan.textContent = d.name;
            var deleteBtn = document.createElement('button');
            deleteBtn.className = 'cv-layer-delete';
            deleteBtn.innerHTML = ICONS.trash;
            deleteBtn.dataset.layerId = String(d.id);
            deleteBtn.title = 'Delete layer';
            var lockBtn = document.createElement('button');
            lockBtn.className = 'cv-layer-lock' + (d.locked ? ' locked' : '');
            lockBtn.innerHTML = d.locked ? ICONS.lock : ICONS.unlock;
            lockBtn.dataset.layerId = String(d.id);
            lockBtn.title = d.locked ? 'Unlock layer' : 'Lock layer';
            var opacitySlider = document.createElement('input');
            opacitySlider.type = 'range';
            opacitySlider.className = 'cv-layer-opacity';
            opacitySlider.min = '0';
            opacitySlider.max = '1';
            opacitySlider.step = '0.05';
            opacitySlider.value = String(d.opacity);
            opacitySlider.title = 'Opacity';
            opacitySlider.dataset.layerId = String(d.id);
            // Inline rename on double-click
            (function (thisD, thisSpan) {
                thisSpan.addEventListener('dblclick', function (e) {
                    e.stopPropagation();
                    var input = document.createElement('input');
                    input.type = 'text';
                    input.className = 'cv-layer-rename-input';
                    input.value = thisD.name;
                    thisSpan.replaceWith(input);
                    input.focus();
                    input.select();
                    function finishRename() {
                        var newName = input.value.trim() || thisD.name;
                        var changed = newName !== thisD.name;
                        thisD.name = newName;
                        renderLayerList();
                        if (changed)
                            History.push();
                    }
                    input.addEventListener('blur', finishRename);
                    input.addEventListener('keydown', function (ev) {
                        if (ev.key === 'Enter') {
                            ev.preventDefault();
                            finishRename();
                        }
                        if (ev.key === 'Escape') {
                            renderLayerList();
                        }
                    });
                });
            })(d, nameSpan);
            row.appendChild(eyeBtn);
            row.appendChild(lockBtn);
            row.appendChild(badge);
            row.appendChild(nameSpan);
            row.appendChild(opacitySlider);
            row.appendChild(deleteBtn);
            (function (rowLayerId) {
                row.addEventListener('contextmenu', function (e) {
                    e.preventDefault();
                    e.stopPropagation();
                    activeLayerId = rowLayerId;
                    renderLayerList();
                    if (typeof CanvasContextMenu !== 'undefined')
                        CanvasContextMenu.show(e.clientX, e.clientY, getToolContext());
                });
            })(d.id);
            els.layerList.appendChild(row);
        }
        // Layer operations footer
        var opsRow = document.createElement('div');
        opsRow.className = 'cv-layer-ops';
        opsRow.innerHTML =
            '<button class="cv-layer-op-btn" id="cv-layer-dup" title="Duplicate layer">Dup</button>' +
                '<button class="cv-layer-op-btn" id="cv-layer-merge" title="Merge down">Merge</button>' +
                '<button class="cv-layer-op-btn" id="cv-layer-flatten" title="Flatten visible">Flatten</button>';
        els.layerList.appendChild(opsRow);
        if (isVideoArch()) {
            var frameIndicator = document.createElement('div');
            frameIndicator.className = 'cv-frame-indicator';
            frameIndicator.textContent = genState.frames + ' frames @ ' + genState.fps + 'fps';
            els.layerList.appendChild(frameIndicator);
        }
        // Event delegation for layer list
        els.layerList.onclick = function (e) {
            var target = e.target;
            // Layer operation buttons
            if (target.id === 'cv-layer-dup' || target.closest('#cv-layer-dup')) {
                if (activeLayerId !== null)
                    duplicateLayer(activeLayerId);
                return;
            }
            if (target.id === 'cv-layer-merge' || target.closest('#cv-layer-merge')) {
                if (activeLayerId !== null)
                    mergeDown(activeLayerId);
                return;
            }
            if (target.id === 'cv-layer-flatten' || target.closest('#cv-layer-flatten')) {
                flattenVisible();
                return;
            }
            var delEl = target.closest('.cv-layer-delete');
            if (delEl) {
                deleteLayer(parseInt(delEl.dataset.layerId));
                e.stopPropagation();
                return;
            }
            var eyeEl = target.closest('.cv-layer-eye');
            if (eyeEl) {
                toggleLayerVisibility(parseInt(eyeEl.dataset.layerId));
                e.stopPropagation();
                return;
            }
            var lockEl = target.closest('.cv-layer-lock');
            if (lockEl) {
                var lockId = parseInt(lockEl.dataset.layerId);
                var lyr = getLayerById(lockId);
                if (lyr) {
                    lyr.data.locked = !lyr.data.locked;
                    renderLayerList();
                    History.push();
                }
                e.stopPropagation();
                return;
            }
            var rowEl = target.closest('.cv-layer-row');
            if (rowEl) {
                activeLayerId = parseInt(rowEl.dataset.layerId);
                renderLayerList();
            }
        };
        els.layerList.addEventListener('input', function (e) {
            var inputTarget = e.target;
            if (inputTarget.classList.contains('cv-layer-opacity')) {
                var lid = parseInt(inputTarget.dataset.layerId);
                var lyr = getLayerById(lid);
                if (lyr) {
                    lyr.data.opacity = parseFloat(inputTarget.value);
                    lyr.konvaLayer.opacity(lyr.data.opacity);
                    lyr.konvaLayer.batchDraw();
                }
            }
        });
        els.layerList.addEventListener('change', function (e) {
            var inputTarget = e.target;
            if (inputTarget.classList.contains('cv-layer-opacity'))
                History.push();
        });
        updateMaskActions();
        updateTypePanels();
        // Update status bar with active layer info
        if (typeof CanvasStatusBar !== 'undefined') {
            var al = getActiveLayer();
            if (al)
                CanvasStatusBar.updateActiveLayer(al.data.name, al.data.type);
        }
        if (activeTool === 'move')
            refreshLayerTransformer();
    }
    function updateMaskActions() {
        var maskActions = document.getElementById('cv-mask-actions');
        if (!maskActions)
            return;
        var al = getActiveLayer();
        maskActions.style.display = (al && al.data.type === 'mask') ? 'flex' : 'none';
    }
    function updateTypePanels() {
        var controlPanel = document.getElementById('cv-control-panel');
        var ipaPanel = document.getElementById('cv-ipadapter-panel');
        var regionalPanel = document.getElementById('cv-regional-panel');
        var adjustPanel = document.getElementById('cv-adjustment-panel');
        var textPanel = document.getElementById('cv-text-panel');
        var al = getActiveLayer();
        var type = al ? al.data.type : 'draw';
        if (controlPanel)
            controlPanel.style.display = type === 'control' ? 'block' : 'none';
        if (ipaPanel)
            ipaPanel.style.display = 'none'; // folded into control
        if (regionalPanel)
            regionalPanel.style.display = type === 'guidance' ? 'block' : 'none';
        if (adjustPanel)
            adjustPanel.style.display = type === 'adjustment' ? 'block' : 'none';
        if (textPanel)
            textPanel.style.display = type === 'text' ? 'block' : 'none';
    }
    function reorderLayer(fromId, toId) {
        var fromIdx = -1, toIdx = -1;
        for (var i = 0; i < canvasLayers.length; i++) {
            if (canvasLayers[i].data.id === fromId)
                fromIdx = i;
            if (canvasLayers[i].data.id === toId)
                toIdx = i;
        }
        if (fromIdx === -1 || toIdx === -1)
            return;
        var moved = canvasLayers.splice(fromIdx, 1)[0];
        canvasLayers.splice(toIdx, 0, moved);
        syncKonvaOrder();
        renderLayerList();
        History.push();
    }
    function moveLayerBy(layerId, delta) {
        var index = canvasLayers.findIndex(function (layer) { return layer.data.id === layerId; });
        if (index < 0)
            return;
        var target = Math.max(0, Math.min(canvasLayers.length - 1, index + delta));
        if (target === index)
            return;
        var moved = canvasLayers.splice(index, 1)[0];
        canvasLayers.splice(target, 0, moved);
        syncKonvaOrder();
        renderLayerList();
        History.push();
    }
    function beginRenameActiveLayer() {
        if (activeLayerId === null)
            return;
        var name = document.querySelector('.cv-layer-row[data-layer-id="' + activeLayerId + '"] .cv-layer-name');
        if (name)
            name.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
    }
    function toggleActiveLayerLock() {
        var layer = getActiveLayer();
        if (!layer)
            return;
        layer.data.locked = !layer.data.locked;
        renderLayerList();
        History.push();
    }
    function toggleActiveLayerTransparency() {
        var layer = getActiveLayer();
        if (!layer || layer.data.type !== 'draw')
            return;
        layer.data.lockTransparency = !layer.data.lockTransparency;
        renderLayerList();
        History.push();
    }
    function toggleLayerVisibility(layerId) {
        for (var i = 0; i < canvasLayers.length; i++) {
            if (canvasLayers[i].data.id === layerId) {
                canvasLayers[i].data.visible = !canvasLayers[i].data.visible;
                canvasLayers[i].data.visible ? canvasLayers[i].konvaLayer.show() : canvasLayers[i].konvaLayer.hide();
                canvasLayers[i].konvaLayer.batchDraw();
                renderLayerList();
                History.push();
                break;
            }
        }
    }
    function deleteLayer(layerId) {
        if (canvasLayers.length <= 1)
            return;
        var idx = -1;
        for (var i = 0; i < canvasLayers.length; i++) {
            if (canvasLayers[i].data.id === layerId) {
                idx = i;
                break;
            }
        }
        if (idx === -1)
            return;
        canvasLayers[idx].konvaLayer.remove();
        canvasLayers.splice(idx, 1);
        History.push();
        if (activeLayerId === layerId) {
            activeLayerId = canvasLayers.length > 0 ? canvasLayers[canvasLayers.length - 1].data.id : null;
        }
        renderLayerList();
    }
    function removeLayerById(layerId, _skipUndo) {
        var idx = -1;
        for (var i = 0; i < canvasLayers.length; i++) {
            if (canvasLayers[i].data.id === layerId) {
                idx = i;
                break;
            }
        }
        if (idx === -1)
            return;
        canvasLayers[idx].konvaLayer.remove();
        canvasLayers.splice(idx, 1);
        if (activeLayerId === layerId && canvasLayers.length > 0) {
            activeLayerId = canvasLayers[canvasLayers.length - 1].data.id;
        }
        renderLayerList();
    }
    // ── Tool Context (bridge between tools and canvas-tab state) ──
    function getToolContext() {
        return {
            stage: stage,
            uiLayer: uiLayer,
            backgroundLayer: backgroundLayer,
            getCanvasLayers: function () { return canvasLayers.slice(); },
            brushCursor: brushCursor,
            boundingBox: boundingBox,
            getActiveLayer: function () { return getActiveLayer(); },
            getActiveKonvaLayer: function () { return getActiveKonvaLayer(); },
            getRelativePointerPosition: getRelativePointerPosition,
            pushHistory: function () { History.push(); },
            pushHistoryGrouped: function () { History.pushGrouped(); },
            setActiveTool: function (name) { setTool(name); },
            getBrushSize: function () { return brushSize; },
            setBrushSize: function (s) { brushSize = s; if (els.brushSizeInput)
                els.brushSizeInput.value = String(s); if (els.brushSizeVal)
                els.brushSizeVal.textContent = String(s); },
            getBrushColor: function () { return brushColor; },
            setBrushColor: function (c) { brushColor = c; if (els.brushColorInput)
                els.brushColorInput.value = c; },
            getBrushHardness: function () { return brushHardness; },
            setBrushHardness: function (h) { brushHardness = h; },
            getBrushOpacity: function () { return CanvasTools.getBrushOpacity(); },
            setBrushOpacity: function (o) { CanvasTools.setBrushOpacity(o); },
            addLayer: addLayer,
            deleteActiveLayer: function () { if (activeLayerId !== null)
                deleteLayer(activeLayerId); },
            flattenVisible: flattenVisible,
            duplicateActiveLayer: function () { if (activeLayerId !== null)
                duplicateLayer(activeLayerId); },
            renameActiveLayer: beginRenameActiveLayer,
            moveActiveLayerForward: function () { if (activeLayerId !== null)
                moveLayerBy(activeLayerId, 1); },
            moveActiveLayerBackward: function () { if (activeLayerId !== null)
                moveLayerBy(activeLayerId, -1); },
            toggleActiveLayerLock: toggleActiveLayerLock,
            toggleActiveLayerTransparency: toggleActiveLayerTransparency,
            transformActiveLayer: function () { setTool('move'); },
            copyActiveLayer: copyActiveLayerToClipboard,
            saveActiveLayer: saveActiveLayerImage,
            cropToActiveLayer: cropBboxToActiveLayer,
            convertActiveLayerTo: convertActiveLayerTo,
            runActiveLayerWorkflow: runActiveLayerWorkflow,
            createRasterLayer: function (name) { return addLayer(name, 'draw'); },
            placeVideoResult: function (src, metadata) {
                var layer = getActiveLayer();
                if (layer)
                    layer.data.generationMetadata = JSON.parse(JSON.stringify(metadata || {}));
                placeVideoOverlayOnCanvas(src);
                History.push();
            },
        };
    }
    // ── Tool Switching ──
    function clearLayerTransformer() {
        if (layerTransformer) {
            layerTransformer.destroy();
            layerTransformer = null;
            if (uiLayer)
                uiLayer.batchDraw();
        }
    }
    function refreshLayerTransformer() {
        clearLayerTransformer();
        if (!stage || !uiLayer || activeTool !== 'move')
            return;
        var active = getActiveLayer();
        if (!active || active.data.locked)
            return;
        var nodes = active.konvaLayer.getChildren().filter(function (node) {
            return node.isVisible() && node.getClientRect({ skipTransform: false }).width > 0;
        });
        if (!nodes.length)
            return;
        nodes.forEach(function (node) {
            node.listening(true);
            node.draggable(true);
        });
        layerTransformer = new Konva.Transformer({
            nodes: nodes,
            rotateEnabled: true,
            flipEnabled: false,
            keepRatio: false,
            borderStroke: '#f59e0b',
            anchorStroke: '#f59e0b',
            anchorFill: '#181824',
            anchorSize: 10,
            boundBoxFunc: function (oldBox, newBox) {
                return Math.abs(newBox.width) < 4 || Math.abs(newBox.height) < 4 ? oldBox : newBox;
            }
        });
        layerTransformer.on('transformend dragend', function () { History.push(); });
        uiLayer.add(layerTransformer);
        layerTransformer.moveToTop();
        uiLayer.batchDraw();
    }
    function setTool(tool) {
        if (tool === 'resetView') {
            resetView();
            return;
        }
        if (tool === 'sam' && !samAvailable) {
            showError('SAM object selection is unavailable: /canvas/sam3/status is not ready');
            return;
        }
        // Deactivate previous tool
        var prevTool = CanvasTools.get(activeTool);
        if (prevTool && prevTool.onDeactivate && stage) {
            prevTool.onDeactivate(getToolContext());
        }
        activeTool = tool;
        clearLayerTransformer();
        // Activate new tool
        var newTool = CanvasTools.get(tool);
        if (newTool && newTool.onActivate && stage) {
            newTool.onActivate(getToolContext());
        }
        document.querySelectorAll('.cv-tool-btn').forEach(function (btn) {
            btn.classList.toggle('active', btn.dataset.tool === tool);
        });
        // Show brush settings for drawing tools
        var showBrush = tool === 'brush' || tool === 'eraser' || tool === 'mask' || tool === 'clonestamp';
        if (els.brushSection) {
            els.brushSection.style.display = showBrush ? 'flex' : 'none';
        }
        // Show fill threshold for fill tool
        var fillSection = document.getElementById('cv-fill-section');
        if (fillSection)
            fillSection.style.display = tool === 'fill' ? 'flex' : 'none';
        var shapeSection = document.getElementById('cv-shape-section');
        if (shapeSection)
            shapeSection.style.display = tool === 'rect' ? 'flex' : 'none';
        var gradientSection = document.getElementById('cv-gradient-section');
        if (gradientSection)
            gradientSection.style.display = tool === 'gradient' ? 'flex' : 'none';
        var lassoSection = document.getElementById('cv-lasso-section');
        if (lassoSection)
            lassoSection.style.display = tool === 'lasso' ? 'flex' : 'none';
        if (brushCursor) {
            brushCursor.visible(false);
            uiLayer.batchDraw();
        }
        // Toggle image draggable in move/select mode
        var draggable = tool === 'move' || tool === 'select';
        canvasLayers.forEach(function (l) {
            l.konvaLayer.getChildren().forEach(function (node) {
                var canMove = draggable && l.data.id === activeLayerId && !l.data.locked;
                node.draggable(canMove);
                node.listening(canMove || node instanceof Konva.Text);
            });
        });
        // Bbox only interactive in select mode — other tools need to click through it
        if (boundingBox) {
            var bboxInteractive = tool === 'select';
            boundingBox.draggable(bboxInteractive);
            boundingBox.listening(bboxInteractive);
            resizeHandles.forEach(function (h) {
                h.draggable(bboxInteractive);
                h.listening(bboxInteractive);
            });
        }
        if (tool === 'move')
            refreshLayerTransformer();
        updateCursor();
    }
    function updateCursor() {
        if (!stage)
            return;
        var c = stage.container();
        var tool = CanvasTools.get(activeTool);
        if (tool) {
            c.style.cursor = tool.cursor;
        }
        else {
            c.style.cursor = 'default';
        }
    }
    function resetView() {
        if (!stage || !boundingBox)
            return;
        var container = document.getElementById('canvas-stage-container');
        if (!container)
            return;
        var cw = container.offsetWidth;
        var ch = container.offsetHeight;
        var bw = boundingBox.width();
        var bh = boundingBox.height();
        var padding = 100;
        var scale = Math.min((cw - padding * 2) / bw, (ch - padding * 2) / bh, 1.5);
        var bx = boundingBox.x();
        var by = boundingBox.y();
        stage.scale({ x: scale, y: scale });
        stage.position({
            x: cw / 2 - (bx + bw / 2) * scale,
            y: ch / 2 - (by + bh / 2) * scale
        });
        stage.batchDraw();
        updateVideoOverlayPosition();
    }
    // ── Keyboard ──
    function handleKeyDown(e) {
        if (localStorage.getItem('sf-active-tab') !== 'canvas')
            return;
        var tag = e.target.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT')
            return;
        var ctrl = e.ctrlKey || e.metaKey;
        // Ctrl combos
        if (ctrl) {
            switch (e.code) {
                case 'KeyZ':
                    e.preventDefault();
                    if (e.shiftKey) {
                        History.redo();
                    }
                    else {
                        History.undo();
                    }
                    return;
                case 'KeyY':
                    e.preventDefault();
                    History.redo();
                    return;
                case 'KeyD':
                    e.preventDefault();
                    CanvasTools.clearLassoSelection(getToolContext());
                    return;
                case 'KeyA':
                    e.preventDefault();
                    // Select all — no-op for now (future: select layer content)
                    return;
                case 'KeyC':
                    e.preventDefault();
                    copyActiveLayerToClipboard();
                    return;
                case 'KeyV':
                    e.preventDefault();
                    pasteFromClipboard();
                    return;
                case 'KeyE':
                    if (e.shiftKey) {
                        e.preventDefault();
                        flattenVisible();
                    }
                    return;
            }
            return;
        }
        // Number keys: brush opacity
        if (e.code >= 'Digit1' && e.code <= 'Digit9' && !e.shiftKey) {
            var digit = parseInt(e.code.charAt(5));
            CanvasTools.setBrushOpacity(digit / 10);
            return;
        }
        if (e.code === 'Digit0') {
            CanvasTools.setBrushOpacity(1);
            return;
        }
        // Bracket keys: brush size / hardness
        if (e.code === 'BracketLeft') {
            if (e.shiftKey) {
                brushHardness = Math.max(0, brushHardness - 0.1);
                var hInput = document.getElementById('cv-brush-hardness');
                var hVal = document.getElementById('cv-brush-hardness-val');
                if (hInput)
                    hInput.value = String(brushHardness);
                if (hVal)
                    hVal.textContent = brushHardness.toFixed(1);
            }
            else {
                brushSize = Math.max(1, brushSize - 5);
                if (els.brushSizeInput)
                    els.brushSizeInput.value = String(brushSize);
                if (els.brushSizeVal)
                    els.brushSizeVal.textContent = String(brushSize);
            }
            return;
        }
        if (e.code === 'BracketRight') {
            if (e.shiftKey) {
                brushHardness = Math.min(1, brushHardness + 0.1);
                var hInput2 = document.getElementById('cv-brush-hardness');
                var hVal2 = document.getElementById('cv-brush-hardness-val');
                if (hInput2)
                    hInput2.value = String(brushHardness);
                if (hVal2)
                    hVal2.textContent = brushHardness.toFixed(1);
            }
            else {
                brushSize = Math.min(200, brushSize + 5);
                if (els.brushSizeInput)
                    els.brushSizeInput.value = String(brushSize);
                if (els.brushSizeVal)
                    els.brushSizeVal.textContent = String(brushSize);
            }
            return;
        }
        // Delete active layer
        if (e.code === 'Delete' || e.code === 'Backspace') {
            if (activeTool === 'move') {
                // Let move tool handle it
                var tool = CanvasTools.get('move');
                if (tool && tool.onKeyDown)
                    tool.onKeyDown(getToolContext(), e);
                return;
            }
        }
        // Tab: toggle layer panel
        if (e.code === 'Tab') {
            e.preventDefault();
            var leftPanel = document.querySelector('.cv-left');
            if (leftPanel)
                leftPanel.style.display = leftPanel.style.display === 'none' ? '' : 'none';
            return;
        }
        // Tool shortcuts (single key, no modifiers)
        switch (e.code) {
            case 'KeyV':
                setTool('select');
                break;
            case 'KeyB':
                setTool('brush');
                break;
            case 'KeyE':
                setTool('eraser');
                break;
            case 'KeyM':
                setTool('move');
                break;
            case 'KeyR':
                setTool('rect');
                break;
            case 'KeyT':
                setTool('text');
                break;
            case 'KeyG':
                setTool('gradient');
                break;
            case 'KeyF':
                setTool('fill');
                break;
            case 'KeyL':
                setTool('lasso');
                break;
            case 'KeyC':
                setTool('clonestamp');
                break;
            case 'KeyP':
                setTool('pencil');
                break;
            case 'KeyU':
                setTool('speechbubble');
                break;
            case 'KeyI':
                setTool('colorpicker');
                break;
            case 'KeyS':
                setTool('sam');
                break;
            case 'KeyH':
                setTool('pan');
                break;
            case 'Space':
                if (!isSpaceHeld) {
                    isSpaceHeld = true;
                    if (activeTool !== 'pan' && stage)
                        stage.container().style.cursor = 'grab';
                }
                e.preventDefault();
                break;
            case 'Slash':
                if (e.shiftKey) {
                    e.preventDefault();
                    showCanvasHotkeys();
                }
                break;
        }
        // Forward to active tool's key handler
        var activToolObj = CanvasTools.get(activeTool);
        if (activToolObj && activToolObj.onKeyDown) {
            activToolObj.onKeyDown(getToolContext(), e);
        }
    }
    function handleKeyUp(e) {
        if (e.code === 'Space') {
            isSpaceHeld = false;
            if (!isPanning)
                updateCursor();
        }
    }
    function copyActiveLayerToClipboard() {
        var al = getActiveLayer();
        if (!al)
            return;
        try {
            var url = al.konvaLayer.toDataURL();
            CanvasTools.setClipboard(url);
        }
        catch (_) { }
    }
    function saveActiveLayerImage() {
        var active = getActiveLayer();
        if (!active)
            return;
        try {
            var anchor = document.createElement('a');
            anchor.href = active.konvaLayer.toDataURL({ pixelRatio: 1 });
            anchor.download = active.data.name.replace(/[^a-z0-9_-]+/gi, '_') + '.png';
            anchor.click();
        }
        catch (error) {
            showError('Could not save layer: ' + error.message);
        }
    }
    function cropBboxToActiveLayer() {
        var active = getActiveLayer();
        if (!active)
            return;
        var rect = active.konvaLayer.getClientRect({ skipTransform: false });
        if (rect.width < 1 || rect.height < 1)
            return;
        boundingBox.position({ x: rect.x, y: rect.y });
        boundingBox.width(clampDimForArch(rect.width));
        boundingBox.height(clampDimForArch(rect.height));
        updateHandles();
        updateSizeLabel();
        updateBboxInputs();
        stage.batchDraw();
        History.push();
    }
    function convertActiveLayerTo(type) {
        var active = getActiveLayer();
        if (!active || ['draw', 'mask', 'guidance'].indexOf(type) < 0 || active.data.type === type)
            return;
        var defaults = LayerDefaults.createByType(type, active.data.name);
        var prior = active.data;
        defaults.id = prior.id;
        defaults.name = prior.name;
        defaults.visible = prior.visible;
        defaults.locked = prior.locked;
        defaults.opacity = prior.opacity;
        defaults.position = prior.position;
        if (type === 'draw' && prior.lockTransparency)
            defaults.lockTransparency = true;
        if (prior.generationMetadata)
            defaults.generationMetadata = prior.generationMetadata;
        active.data = defaults;
        renderLayerList();
        History.push();
    }
    function runActiveLayerWorkflow() {
        var active = getActiveLayer();
        if (!active || canvasGenerating)
            return;
        if (!genState.model || !genState.prompt.trim()) {
            showError('Choose a model and prompt before running the layer workflow');
            return;
        }
        var video = isVideoArch();
        if (video) {
            showError('The selected video runner is text-to-video only; active-layer video input is not supported');
            return;
        }
        if (!canvasFeature('image_to_image').supported) {
            showError(canvasFeatureReason('image_to_image'));
            return;
        }
        var loras = enabledCanvasLoras();
        if (loras.length > 0 && !canvasFeature('lora').supported) {
            showError(canvasFeatureReason('lora'));
            return;
        }
        if (loras.length > 1 && !canvasFeature('multi_lora').supported) {
            showError(canvasFeatureReason('multi_lora'));
            return;
        }
        setCanvasGenerating(true);
        var width = video ? ModelUtils.clampVideoDimension(boundingBox.width()) : ModelUtils.clampDimension(boundingBox.width());
        var height = video ? ModelUtils.clampVideoDimension(boundingBox.height()) : ModelUtils.clampDimension(boundingBox.height());
        var base64 = active.konvaLayer.toDataURL({
            x: boundingBox.x(), y: boundingBox.y(), width: width, height: height, pixelRatio: 1
        }).split(',')[1];
        uploadInitImage(base64).then(function (imageName) {
            var seed = genState.seed === -1 ? Math.floor(Math.random() * 4294967296) : genState.seed;
            pendingCanvasMetadata = {
                prompt: genState.prompt, model: genState.model, arch: genState.arch,
                width: width, height: height, steps: genState.steps,
                cfg: genState.cfg, guidance: genState.guidance, seed: seed,
                negative: genState.negative, sampler: genState.sampler, scheduler: genState.scheduler, batch: genState.batch,
                loras: enabledCanvasLoras(), capsPositive: genState.capsPositive,
                capsNegative: genState.capsNegative, noiseFixture: genState.noiseFixture,
                includeAudio: genState.includeAudio,
                sourceLayerId: active.data.id, sourceLayerName: active.data.name,
                submittedAt: new Date().toISOString()
            };
            var params = {
                model: genState.model, prompt: genState.prompt, initImageName: imageName,
                width: width, height: height, steps: genState.steps, cfg: genState.cfg,
                guidance: genState.guidance, denoise: genState.denoise, seed: seed,
                negPrompt: genState.negative, sampler: genState.sampler, scheduler: genState.scheduler, batch: genState.batch,
                frames: genState.frames, fps: genState.fps, loras: enabledCanvasLoras(),
                capsPositive: genState.capsPositive, capsNegative: genState.capsNegative,
                noiseFixture: genState.noiseFixture, includeAudio: genState.includeAudio
            };
            queueWorkflow(video ? WorkflowBuilder.build(params) : WorkflowBuilder.buildImg2Img(params));
        }).catch(function (error) {
            showError('Layer workflow failed: ' + error.message);
            setCanvasGenerating(false);
        });
    }
    function pasteFromClipboard() {
        var data = CanvasTools.getClipboard();
        if (!data)
            return;
        var img = new Image();
        img.onload = function () {
            var kImg = new Konva.Image({
                image: img,
                x: boundingBox.x(), y: boundingBox.y(),
                width: img.width, height: img.height,
                draggable: activeTool === 'select' || activeTool === 'move'
            });
            var layer = getActiveKonvaLayer();
            if (layer) {
                layer.add(kImg);
                layer.batchDraw();
                History.push();
            }
        };
        img.src = data;
    }
    // ── Right Panel ──
    function bindRightPanelEvents() {
        els.brushSizeInput.addEventListener('input', function () {
            brushSize = parseInt(this.value);
            els.brushSizeVal.textContent = String(brushSize);
        });
        var hardnessInput = document.getElementById('cv-brush-hardness');
        var hardnessVal = document.getElementById('cv-brush-hardness-val');
        if (hardnessInput) {
            hardnessInput.addEventListener('input', function () {
                brushHardness = parseFloat(this.value);
                if (hardnessVal)
                    hardnessVal.textContent = brushHardness.toFixed(1);
            });
        }
        // Fill threshold
        var fillThreshold = document.getElementById('cv-fill-threshold');
        var fillThresholdVal = document.getElementById('cv-fill-threshold-val');
        if (fillThreshold) {
            fillThreshold.addEventListener('input', function () {
                var val = parseInt(this.value);
                CanvasTools.setFillThreshold(val);
                if (fillThresholdVal)
                    fillThresholdVal.textContent = String(val);
            });
        }
        var shapeMode = document.getElementById('cv-shape-mode');
        var shapeSubtract = document.getElementById('cv-shape-subtract');
        if (shapeMode)
            shapeMode.addEventListener('change', function () { CanvasTools.setShapeMode(this.value); });
        if (shapeSubtract)
            shapeSubtract.addEventListener('click', function () {
                CanvasTools.setShapeSubtract(!CanvasTools.getShapeSubtract());
                this.classList.toggle('active', CanvasTools.getShapeSubtract());
            });
        var gradientMode = document.getElementById('cv-gradient-mode');
        var gradientStart = document.getElementById('cv-gradient-start');
        var gradientEnd = document.getElementById('cv-gradient-end');
        var gradientClip = document.getElementById('cv-gradient-clip');
        function syncGradientColors() {
            CanvasTools.setGradientColors(gradientStart && gradientStart.value, gradientEnd && gradientEnd.value);
        }
        if (gradientMode)
            gradientMode.addEventListener('change', function () { CanvasTools.setGradientMode(this.value); });
        if (gradientStart)
            gradientStart.addEventListener('input', syncGradientColors);
        if (gradientEnd)
            gradientEnd.addEventListener('input', syncGradientColors);
        if (gradientClip)
            gradientClip.addEventListener('click', function () {
                CanvasTools.setGradientClip(!CanvasTools.getGradientClip());
                this.classList.toggle('active', CanvasTools.getGradientClip());
            });
        var lassoMode = document.getElementById('cv-lasso-mode');
        var lassoSubtract = document.getElementById('cv-lasso-subtract');
        var lassoAutoMask = document.getElementById('cv-lasso-auto-mask');
        if (lassoMode)
            lassoMode.addEventListener('change', function () { CanvasTools.setLassoMode(this.value); });
        if (lassoSubtract)
            lassoSubtract.addEventListener('click', function () {
                CanvasTools.setLassoSubtract(!CanvasTools.getLassoSubtract());
                this.classList.toggle('active', CanvasTools.getLassoSubtract());
            });
        if (lassoAutoMask)
            lassoAutoMask.addEventListener('click', function () {
                CanvasTools.setLassoAutoMask(!CanvasTools.getLassoAutoMask());
                this.classList.toggle('active', CanvasTools.getLassoAutoMask());
            });
        els.brushColorInput.addEventListener('input', function () {
            brushColor = this.value;
        });
        els.prompt.addEventListener('input', function () {
            genState.prompt = this.value;
            this.style.height = 'auto';
            this.style.height = this.scrollHeight + 'px';
        });
        els.negative.addEventListener('input', function () { genState.negative = this.value; });
        els.editMode.addEventListener('change', function () {
            var nextMode = this.value;
            if (nextMode === 'style') {
                enterStyleMode();
                return;
            }
            genState.editMode = nextMode;
            if (nextMode === 'flowedit') {
                genState.editEngine = normalizeFlowEditEngine(genState.editEngine);
                if (isKreaFlowEditEngine(genState.editEngine))
                    genState.editEngine = upgradeKreaFlowEditEngineTo1024(genState.editEngine);
                els.editEngine.value = genState.editEngine;
                applyFlowEditEngineProfile(genState.editEngine);
                syncCanvasModelFromFlowEditEngine();
            }
            updateEditWorkspace();
        });
        els.editEngine.addEventListener('change', function () {
            genState.editEngine = this.value;
            applyFlowEditEngineProfile(this.value);
            syncCanvasModelFromFlowEditEngine();
            var size = flowEditEngineSize(this.value);
            if (boundingBox) {
                boundingBox.width(size);
                boundingBox.height(size);
                updateHandles();
                updateSizeLabel();
                updateBboxInputs();
                stage.batchDraw();
                resetView();
            }
            updateEditWorkspace();
        });
        els.lanpaintEngine.addEventListener('change', function () {
            genState.lanpaintEngine = this.value;
            if (!syncCanvasModelFromLanPaintEngine()) {
                showError('No registered model matches ' + this.options[this.selectedIndex].textContent);
                return;
            }
            applyMaskedEditEngineProfile(this.value);
            updateEditWorkspace();
            resetView();
        });
        els.lanpaintNumSteps.addEventListener('input', function () { genState.lanpaintNumSteps = Math.max(0, parseInt(this.value) || 0); });
        els.lanpaintLambda.addEventListener('input', function () { genState.lanpaintLambda = Number(this.value); });
        els.lanpaintStepSize.addEventListener('input', function () { genState.lanpaintStepSize = Number(this.value); });
        els.lanpaintBeta.addEventListener('input', function () { genState.lanpaintBeta = Number(this.value); });
        els.lanpaintFriction.addEventListener('input', function () { genState.lanpaintFriction = Number(this.value); });
        els.lanpaintPromptMode.addEventListener('change', function () { genState.lanpaintPromptMode = this.value; });
        els.lanpaintBlendOverlap.addEventListener('input', function () { genState.lanpaintBlendOverlap = Math.floor(Number(this.value)); });
        els.lanpaintEarlyStop.addEventListener('input', function () { genState.lanpaintEarlyStop = Math.max(0, parseInt(this.value) || 0); });
        els.lanpaintInnerThreshold.addEventListener('input', function () { genState.lanpaintInnerThreshold = Number(this.value); });
        els.lanpaintInnerPatience.addEventListener('input', function () { genState.lanpaintInnerPatience = Math.max(0, parseInt(this.value) || 0); });
        els.editSourcePrompt.addEventListener('input', function () { genState.editSourcePrompt = this.value; });
        els.editNmax.addEventListener('input', function () { genState.editNmax = Math.max(0, parseInt(this.value) || 0); });
        els.editNmin.addEventListener('input', function () { genState.editNmin = Math.max(0, parseInt(this.value) || 0); });
        els.editSourceCfg.addEventListener('input', function () { genState.editSourceCfg = Number(this.value); });
        els.editAutoMask.addEventListener('change', function () { genState.editAutoMask = this.checked; });
        els.editMaskQ.addEventListener('input', function () { genState.editMaskQ = Number(this.value); });
        els.editMaskDilate.addEventListener('input', function () { genState.editMaskDilate = Math.max(0, parseInt(this.value) || 0); });
        els.editMaskWarmup.addEventListener('input', function () { genState.editMaskWarmup = Math.max(0, parseInt(this.value) || 0); });
        els.styleEntireImage.addEventListener('change', function () { genState.styleEntireImage = this.checked; });
        els.sampler.addEventListener('change', function () { genState.sampler = this.value; });
        els.scheduler.addEventListener('change', function () { genState.scheduler = this.value; });
        els.capsPositive.addEventListener('input', function () { genState.capsPositive = this.value.trim(); });
        els.capsNegative.addEventListener('input', function () { genState.capsNegative = this.value.trim(); });
        els.noiseFixture.addEventListener('input', function () { genState.noiseFixture = this.value.trim(); });
        els.includeAudio.addEventListener('change', function () { genState.includeAudio = this.checked; });
        els.loraAdd.addEventListener('click', addCanvasLora);
        els.loadLtx2Template.addEventListener('click', loadLtx2TemplateProfile);
        els.seed.addEventListener('input', function () {
            var value = Number(this.value);
            genState.seed = Number.isFinite(value) ? Math.max(-1, Math.min(4294967295, Math.floor(value))) : -1;
        });
        els.batch.addEventListener('input', function () {
            genState.batch = Math.max(1, Math.min(8, parseInt(this.value) || 1));
        });
        els.denoise.addEventListener('input', function () {
            genState.denoise = parseFloat(this.value);
            els.denoiseVal.textContent = genState.denoise.toFixed(2);
        });
        els.steps.addEventListener('input', function () {
            genState.steps = parseInt(this.value) || 20;
            els.stepsRange.value = this.value;
        });
        els.stepsRange.addEventListener('input', function () {
            genState.steps = parseInt(this.value);
            els.steps.value = this.value;
        });
        els.cfg.addEventListener('input', function () {
            var value = parseFloat(this.value);
            genState.cfg = Number.isFinite(value) ? value : 7.0;
            els.cfgRange.value = this.value;
        });
        els.cfgRange.addEventListener('input', function () {
            genState.cfg = parseFloat(this.value);
            els.cfg.value = this.value;
        });
        els.model.addEventListener('change', function () {
            genState.model = this.value;
            updateTopbarModel(this.value);
            updateCanvasUIForArch(ModelUtils.detectArchFromFilename(this.value));
            if (genState.editMode === 'flowedit' || genState.editMode === 'style')
                syncFlowEditEngineFromCanvasModel(this.value);
        });
        els.guidance.addEventListener('input', function () {
            genState.guidance = parseFloat(this.value) || 3.5;
            els.guidanceRange.value = this.value;
        });
        els.guidanceRange.addEventListener('input', function () {
            genState.guidance = parseFloat(this.value);
            els.guidance.value = this.value;
        });
        // Frames sync
        els.framesInput.addEventListener('input', function () {
            genState.frames = parseInt(this.value) || 97;
            els.framesRange.value = this.value;
            updateCanvasDurationHint();
            updateSizeLabel();
        });
        els.framesRange.addEventListener('input', function () {
            genState.frames = parseInt(this.value);
            els.framesInput.value = this.value;
            updateCanvasDurationHint();
            updateSizeLabel();
        });
        // FPS sync
        els.fpsInput.addEventListener('input', function () {
            genState.fps = parseInt(this.value) || 24;
            els.fpsRange.value = this.value;
            updateCanvasDurationHint();
        });
        els.fpsRange.addEventListener('input', function () {
            genState.fps = parseInt(this.value);
            els.fpsInput.value = this.value;
            updateCanvasDurationHint();
        });
        els.importBtn.addEventListener('click', function () { els.importFile.click(); });
        els.sourceBrowse.addEventListener('click', function (event) {
            event.stopPropagation();
            els.importFile.click();
        });
        els.sourceDropzone.addEventListener('click', function () { els.importFile.click(); });
        els.sourceDropzone.addEventListener('dragover', function (event) {
            event.preventDefault();
            els.sourceDropzone.classList.add('drag-over');
        });
        els.sourceDropzone.addEventListener('dragleave', function () { els.sourceDropzone.classList.remove('drag-over'); });
        els.sourceDropzone.addEventListener('drop', function (event) {
            event.preventDefault();
            event.stopPropagation();
            els.sourceDropzone.classList.remove('drag-over');
            loadEditSourceFile(event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0]);
        });
        els.styleBrowse.addEventListener('click', function (event) {
            event.stopPropagation();
            els.styleFile.click();
        });
        els.styleDropzone.addEventListener('click', function () { els.styleFile.click(); });
        els.styleDropzone.addEventListener('dragover', function (event) {
            event.preventDefault();
            els.styleDropzone.classList.add('drag-over');
        });
        els.styleDropzone.addEventListener('dragleave', function () { els.styleDropzone.classList.remove('drag-over'); });
        els.styleDropzone.addEventListener('drop', function (event) {
            event.preventDefault();
            event.stopPropagation();
            els.styleDropzone.classList.remove('drag-over');
            loadStyleReferenceFile(event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0]);
        });
        els.styleFile.addEventListener('change', function () {
            if (this.files && this.files[0])
                loadStyleReferenceFile(this.files[0]);
            this.value = '';
        });
        els.importFile.addEventListener('change', function () {
            if (this.files && this.files[0]) {
                if (genState.editMode === 'create')
                    loadImageFile(this.files[0]);
                else
                    loadEditSourceFile(this.files[0]);
                this.value = '';
            }
        });
        els.generateBtn.addEventListener('click', function () {
            if (canvasGenerating)
                cancelCanvasGeneration();
            else
                startGeneration();
        });
        document.querySelectorAll('.cv-tool-btn').forEach(function (btn) {
            btn.addEventListener('click', function () { setTool(btn.dataset.tool); });
        });
        // Undo/redo buttons
        var undoBtn = document.getElementById('cv-undo');
        var redoBtn = document.getElementById('cv-redo');
        if (undoBtn)
            undoBtn.addEventListener('click', function () { History.undo(); });
        if (redoBtn)
            redoBtn.addEventListener('click', function () { History.redo(); });
        var sessionSave = document.getElementById('cv-session-save');
        var sessionLoad = document.getElementById('cv-session-load');
        var sessionNew = document.getElementById('cv-session-new');
        var sessionFile = document.getElementById('cv-session-file');
        if (sessionSave)
            sessionSave.addEventListener('click', saveCanvasSession);
        if (sessionLoad && sessionFile)
            sessionLoad.addEventListener('click', function () { sessionFile.click(); });
        if (sessionNew)
            sessionNew.addEventListener('click', newCanvasProject);
        if (sessionFile) {
            sessionFile.addEventListener('change', function () {
                if (sessionFile.files && sessionFile.files[0])
                    loadCanvasSession(sessionFile.files[0]);
                sessionFile.value = '';
            });
        }
        var galleryRefresh = document.getElementById('cv-gallery-refresh');
        if (galleryRefresh)
            galleryRefresh.addEventListener('click', loadCanvasGallery);
        var galleryBoardFilterEl = document.getElementById('cv-gallery-board-filter');
        if (galleryBoardFilterEl)
            galleryBoardFilterEl.addEventListener('change', function () {
                galleryBoardFilter = this.value;
                loadCanvasGallery();
            });
        var galleryBoardNew = document.getElementById('cv-gallery-board-new');
        if (galleryBoardNew)
            galleryBoardNew.addEventListener('click', createCanvasGalleryBoard);
        var hotkeys = document.getElementById('cv-hotkeys');
        if (hotkeys)
            hotkeys.addEventListener('click', showCanvasHotkeys);
        // Control layer type panel sliders — persist values to active layer's data
        var controlType = document.getElementById('cv-control-type');
        if (controlType) {
            controlType.addEventListener('change', function () {
                var cl = getActiveLayer();
                if (cl && cl.data.type === 'control')
                    cl.data.controlModel = this.value;
            });
        }
        var controlWeight = document.getElementById('cv-control-weight');
        var controlWeightVal = document.getElementById('cv-control-weight-val');
        if (controlWeight) {
            controlWeight.addEventListener('input', function () {
                var val = parseFloat(this.value);
                if (controlWeightVal)
                    controlWeightVal.textContent = val.toFixed(2);
                var cl = getActiveLayer();
                if (cl && cl.data.type === 'control')
                    cl.data.weight = val;
            });
        }
        var controlStart = document.getElementById('cv-control-start');
        var controlStartVal = document.getElementById('cv-control-start-val');
        if (controlStart) {
            controlStart.addEventListener('input', function () {
                var val = parseFloat(this.value);
                if (controlStartVal)
                    controlStartVal.textContent = val.toFixed(2);
                var cl = getActiveLayer();
                if (cl && cl.data.type === 'control')
                    cl.data.beginStep = val;
            });
        }
        var controlEnd = document.getElementById('cv-control-end');
        var controlEndVal = document.getElementById('cv-control-end-val');
        if (controlEnd) {
            controlEnd.addEventListener('input', function () {
                var val = parseFloat(this.value);
                if (controlEndVal)
                    controlEndVal.textContent = val.toFixed(2);
                var cl = getActiveLayer();
                if (cl && cl.data.type === 'control')
                    cl.data.endStep = val;
            });
        }
        // Control layer image upload
        var controlWell = document.getElementById('cv-control-well');
        var controlFile = document.getElementById('cv-control-file');
        if (controlWell && controlFile) {
            controlWell.addEventListener('click', function () { controlFile.click(); });
            controlFile.addEventListener('change', function () {
                if (this.files && this.files[0]) {
                    handleLayerImageUpload(activeLayerId, this.files[0], controlWell);
                    this.value = '';
                }
            });
            controlWell.addEventListener('dragover', function (e) { e.preventDefault(); });
            controlWell.addEventListener('drop', function (e) {
                e.preventDefault();
                var file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
                if (file && file.type.startsWith('image/')) {
                    handleLayerImageUpload(activeLayerId, file, controlWell);
                }
            });
        }
        // Preprocess button
        var preprocessBtn = document.getElementById('cv-preprocess-btn');
        if (preprocessBtn) {
            preprocessBtn.addEventListener('click', function () {
                var methodSelect = document.getElementById('cv-control-type');
                var method = methodSelect ? methodSelect.value : 'canny';
                if (typeof CanvasPreprocess !== 'undefined' && stage) {
                    CanvasPreprocess.processActiveControlLayer(method, getToolContext());
                }
            });
        }
        // IP-Adapter image upload
        var ipaWell = document.getElementById('cv-ipa-well');
        var ipaFile = document.getElementById('cv-ipa-file');
        if (ipaWell && ipaFile) {
            ipaWell.addEventListener('click', function () { ipaFile.click(); });
            ipaFile.addEventListener('change', function () {
                if (this.files && this.files[0]) {
                    handleLayerImageUpload(activeLayerId, this.files[0], ipaWell);
                    this.value = '';
                }
            });
            ipaWell.addEventListener('dragover', function (e) { e.preventDefault(); });
            ipaWell.addEventListener('drop', function (e) {
                e.preventDefault();
                var file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
                if (file && file.type.startsWith('image/')) {
                    handleLayerImageUpload(activeLayerId, file, ipaWell);
                }
            });
        }
        // Layer add menu
        var addBtn = document.getElementById('cv-layers-add-btn');
        var typeMenu = document.getElementById('cv-layer-type-menu');
        if (addBtn && typeMenu) {
            // Move dropdown to document.body so it's not clipped by overflow:hidden parents
            document.body.appendChild(typeMenu);
            addBtn.addEventListener('click', function (e) {
                e.stopPropagation();
                if (typeMenu.style.display !== 'none') {
                    typeMenu.style.display = 'none';
                    return;
                }
                // Position toward the available viewport space. The Invoke-style
                // entity panel keeps this button near the top; older layouts kept
                // it near the bottom.
                var btnRect = addBtn.getBoundingClientRect();
                typeMenu.style.position = 'fixed';
                typeMenu.style.left = 'auto'; typeMenu.style.right = Math.max(8, window.innerWidth - btnRect.right) + 'px';
                if (btnRect.top < window.innerHeight / 2) {
                    typeMenu.style.top = (btnRect.bottom + 4) + 'px';
                    typeMenu.style.bottom = 'auto';
                }
                else {
                    typeMenu.style.top = 'auto';
                    typeMenu.style.bottom = (window.innerHeight - btnRect.top + 4) + 'px';
                }
                typeMenu.style.display = 'block';
            });
            typeMenu.addEventListener('click', function (e) {
                var item = e.target.closest('.cv-layer-type-item');
                if (!item)
                    return;
                var type = item.dataset.type;
                var label = LAYER_TYPE_LABELS[type] || type;
                addLayer(label + ' ' + (canvasLayers.length + 1), type);
                typeMenu.style.display = 'none';
            });
            document.addEventListener('click', function () {
                typeMenu.style.display = 'none';
            });
        }
        // Bbox toolbar
        var bboxW = document.getElementById('cv-bbox-w');
        var bboxH = document.getElementById('cv-bbox-h');
        var bboxSnap = document.getElementById('cv-bbox-snap');
        var bboxReset = document.getElementById('cv-bbox-reset');
        function bboxKeydown(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                e.stopPropagation();
                e.target.blur();
            }
        }
        if (bboxW) {
            bboxW.addEventListener('change', function () {
                var v = clampDimForArch(parseInt(this.value) || 1024);
                this.value = String(v);
                boundingBox.width(v);
                updateHandles();
                updateSizeLabel();
                stage.batchDraw();
                History.push();
            });
            bboxW.addEventListener('keydown', bboxKeydown);
        }
        if (bboxH) {
            bboxH.addEventListener('change', function () {
                var v = clampDimForArch(parseInt(this.value) || 1024);
                this.value = String(v);
                boundingBox.height(v);
                updateHandles();
                updateSizeLabel();
                stage.batchDraw();
                History.push();
            });
            bboxH.addEventListener('keydown', bboxKeydown);
        }
        if (bboxReset) {
            bboxReset.addEventListener('click', function () {
                var container = document.getElementById('canvas-stage-container');
                if (!container)
                    return;
                var w = 1024, h = 1024;
                boundingBox.width(w);
                boundingBox.height(h);
                boundingBox.x(container.offsetWidth / 2 / stage.scaleX() - w / 2 - stage.x() / stage.scaleX());
                boundingBox.y(container.offsetHeight / 2 / stage.scaleY() - h / 2 - stage.y() / stage.scaleY());
                updateHandles();
                updateSizeLabel();
                updateBboxInputs();
                stage.batchDraw();
            });
        }
        var bboxLock = document.getElementById('cv-bbox-lock');
        if (bboxLock) {
            bboxLock.addEventListener('click', function () {
                bboxAspectLocked = !bboxAspectLocked;
                this.classList.toggle('active', bboxAspectLocked);
                if (bboxAspectLocked) {
                    bboxLockedRatio = boundingBox.width() / boundingBox.height();
                }
            });
        }
        var bboxFit = document.getElementById('cv-bbox-fit');
        if (bboxFit) {
            bboxFit.addEventListener('click', function () {
                var layer = getActiveKonvaLayer();
                if (!layer)
                    return;
                var rect = layer.getClientRect({ skipTransform: true });
                if (rect.width < 1 || rect.height < 1)
                    return;
                boundingBox.x(rect.x);
                boundingBox.y(rect.y);
                boundingBox.width(clampDimForArch(rect.width));
                boundingBox.height(clampDimForArch(rect.height));
                updateHandles();
                updateSizeLabel();
                stage.batchDraw();
            });
        }
        var maskFill = document.getElementById('cv-mask-fill');
        var maskClear = document.getElementById('cv-mask-clear');
        if (maskFill) {
            maskFill.addEventListener('click', function () {
                var layer = getLayerById(activeLayerId);
                if (!layer || layer.data.type !== 'mask')
                    return;
                var rect = new Konva.Rect({
                    x: boundingBox.x(), y: boundingBox.y(),
                    width: boundingBox.width(), height: boundingBox.height(),
                    fill: 'rgba(239, 68, 68, 0.5)',
                    name: 'full-frame-mask',
                    listening: false
                });
                layer.konvaLayer.add(rect);
                layer.konvaLayer.batchDraw();
                History.push();
            });
        }
        if (maskClear) {
            maskClear.addEventListener('click', function () {
                var layer = getLayerById(activeLayerId);
                if (!layer || layer.data.type !== 'mask')
                    return;
                layer.konvaLayer.destroyChildren();
                layer.konvaLayer.batchDraw();
                History.push();
            });
        }
    }
    // ── Adjustment Panel Bindings ──
    function bindAdjustmentPanel() {
        var fields = ['brightness', 'contrast', 'saturation', 'temperature', 'tint', 'sharpness'];
        fields.forEach(function (field) {
            var slider = document.getElementById('cv-adj-' + field);
            var valSpan = document.getElementById('cv-adj-' + field + '-val');
            if (!slider)
                return;
            slider.addEventListener('input', function () {
                var val = parseFloat(this.value);
                if (valSpan)
                    valSpan.textContent = val.toFixed(2);
                var cl = getActiveLayer();
                if (cl && cl.data.type === 'adjustment') {
                    cl.data[field] = val;
                    // Apply adjustment via Konva filters on underlying layers
                    applyAdjustmentPreview(cl);
                }
            });
        });
    }
    function applyAdjustmentPreview(cl) {
        // Adjustment layers apply CSS filters on their Konva layer's canvas element
        if (cl.data.type !== 'adjustment')
            return;
        var ad = cl.data;
        var canvas = cl.konvaLayer.canvas()._canvas;
        var filters = [];
        if (ad.brightness !== 0)
            filters.push('brightness(' + (1 + ad.brightness) + ')');
        if (ad.contrast !== 0)
            filters.push('contrast(' + (1 + ad.contrast) + ')');
        if (ad.saturation !== 0)
            filters.push('saturate(' + (1 + ad.saturation) + ')');
        if (ad.temperature !== 0) {
            // Approximate temperature with hue-rotate
            filters.push('hue-rotate(' + (ad.temperature * 30) + 'deg)');
        }
        canvas.style.filter = filters.length > 0 ? filters.join(' ') : '';
    }
    // ── Inline Text Editing ──
    function attachInlineTextEdit(kText, layerData, konvaLayer) {
        kText.on('dblclick dbltap', function () {
            // Hide Konva text while editing
            kText.hide();
            konvaLayer.batchDraw();

            var stageContainer = document.getElementById('canvas-stage-container');
            if (!stageContainer) return;

            // Get screen-space bounding box of the text node
            var box = kText.getClientRect();
            var containerRect = stageContainer.getBoundingClientRect();
            var scale = stage.scaleX();

            var textarea = document.createElement('textarea');
            textarea.value = kText.text();
            textarea.style.position = 'absolute';
            textarea.style.left = (box.x - containerRect.left) + 'px';
            textarea.style.top = (box.y - containerRect.top) + 'px';
            textarea.style.width = Math.max(box.width, 120) + 'px';
            textarea.style.minHeight = Math.max(box.height, 30) + 'px';
            textarea.style.fontSize = (kText.fontSize() * scale) + 'px';
            textarea.style.fontFamily = kText.fontFamily();
            textarea.style.fontStyle = kText.fontStyle();
            textarea.style.color = kText.fill();
            textarea.style.background = 'rgba(0,0,0,0.85)';
            textarea.style.border = '2px solid #6c6af5';
            textarea.style.borderRadius = '4px';
            textarea.style.padding = '4px';
            textarea.style.margin = '0';
            textarea.style.outline = 'none';
            textarea.style.resize = 'both';
            textarea.style.overflow = 'hidden';
            textarea.style.lineHeight = String(kText.lineHeight());
            textarea.style.zIndex = '1000';
            textarea.style.boxSizing = 'border-box';
            stageContainer.appendChild(textarea);
            textarea.focus();
            textarea.select();

            function finishEdit() {
                var newText = textarea.value;
                layerData.text = newText;
                kText.text(newText);
                kText.show();
                konvaLayer.batchDraw();
                textarea.remove();
                // Sync the right panel input
                var panelInput = document.getElementById('cv-text-content');
                if (panelInput) panelInput.value = newText;
                History.push();
            }

            textarea.addEventListener('blur', finishEdit);
            textarea.addEventListener('keydown', function (ev) {
                // Enter without shift commits; shift+enter inserts newline
                if (ev.key === 'Enter' && !ev.shiftKey) {
                    ev.preventDefault();
                    textarea.blur();
                }
                if (ev.key === 'Escape') {
                    textarea.value = kText.text(); // revert
                    textarea.blur();
                }
                // Stop propagation so canvas shortcuts don't fire
                ev.stopPropagation();
            });
        });
    }

    // ── Text Panel Bindings ──
    function bindTextPanel() {
        var textContent = document.getElementById('cv-text-content');
        var textFont = document.getElementById('cv-text-font');
        var textSize = document.getElementById('cv-text-size');
        var textSizeVal = document.getElementById('cv-text-size-val');
        var textColor = document.getElementById('cv-text-color');
        var textWeight = document.getElementById('cv-text-weight');
        function updateTextLayer(updater) {
            var cl = getActiveLayer();
            if (!cl || cl.data.type !== 'text')
                return;
            var td = cl.data;
            var kText = cl.konvaLayer.findOne('Text');
            if (!kText)
                return;
            updater(td, kText);
            cl.konvaLayer.batchDraw();
        }
        if (textContent)
            textContent.addEventListener('input', function () {
                var v = textContent.value;
                updateTextLayer(function (td, kText) { td.text = v; kText.text(v); });
            });
        if (textFont)
            textFont.addEventListener('change', function () {
                var v = textFont.value;
                updateTextLayer(function (td, kText) { td.fontFamily = v; kText.fontFamily(v); });
            });
        if (textSize)
            textSize.addEventListener('input', function () {
                var val = parseInt(textSize.value);
                if (textSizeVal)
                    textSizeVal.textContent = String(val);
                updateTextLayer(function (td, kText) { td.fontSize = val; kText.fontSize(val); });
            });
        if (textColor)
            textColor.addEventListener('input', function () {
                var v = textColor.value;
                updateTextLayer(function (td, kText) { td.color = v; kText.fill(v); });
            });
        if (textWeight)
            textWeight.addEventListener('change', function () {
                var v = textWeight.value;
                updateTextLayer(function (td, kText) { td.fontWeight = v; kText.fontStyle(v); });
            });
    }
    // ── Session Save/Load ──
    function newCanvasProject() {
        if (!window.confirm('Start a new Canvas project? Save the current project first if you want to keep it.'))
            return;
        if (typeof CanvasStaging !== 'undefined')
            CanvasStaging.deactivate();
        removeVideoOverlay();
        canvasLayers.forEach(function (layer) { layer.konvaLayer.destroy(); });
        canvasLayers = [];
        activeLayerId = null;
        if (typeof CanvasRefImages !== 'undefined')
            CanvasRefImages.restore([]);
        genState.styleEntireImage = true;
        if (els.styleEntireImage)
            els.styleEntireImage.checked = true;
        addLayer('Draw Layer', 'draw');
        resetView();
        History.push();
    }
    function showCanvasHotkeys() {
        var existing = document.getElementById('cv-hotkeys-dialog');
        if (existing) {
            existing.remove();
            return;
        }
        var overlay = document.createElement('div');
        overlay.id = 'cv-hotkeys-dialog';
        overlay.className = 'cv-dialog-backdrop';
        overlay.innerHTML = '<div class="cv-dialog" role="dialog" aria-modal="true" aria-label="Canvas shortcuts">' +
            '<div class="cv-dialog-header"><span>Canvas shortcuts</span><button type="button" class="cv-dialog-close">&times;</button></div>' +
            '<div class="cv-hotkey-grid">' +
            '<kbd>V</kbd><span>Select bounding box</span><kbd>M</kbd><span>Move / transform layer</span>' +
            '<kbd>B / E</kbd><span>Brush / eraser</span><kbd>R / G</kbd><span>Shapes / gradient</span>' +
            '<kbd>L / S</kbd><span>Lasso / SAM object select</span><kbd>H / Space</kbd><span>Pan canvas</span>' +
            '<kbd>Ctrl Z</kbd><span>Undo</span><kbd>Ctrl Shift Z</kbd><span>Redo</span>' +
            '<kbd>[ / ]</kbd><span>Brush size</span><kbd>Shift [ / ]</kbd><span>Brush hardness</span>' +
            '<kbd>1–0</kbd><span>Brush opacity</span><kbd>Arrows</kbd><span>Nudge active layer</span>' +
            '<kbd>?</kbd><span>Toggle this help</span><kbd>Esc</kbd><span>Cancel active operation</span>' +
            '</div></div>';
        overlay.addEventListener('click', function (event) {
            if (event.target === overlay || event.target.closest('.cv-dialog-close'))
                overlay.remove();
        });
        document.body.appendChild(overlay);
    }
    function saveCanvasSession() {
        if (!stage || !boundingBox)
            return;
        LayerSerializer.buildSessionState(canvasLayers, { x: boundingBox.x(), y: boundingBox.y(), width: boundingBox.width(), height: boundingBox.height() }, activeLayerId, { prompt: genState.prompt, denoise: genState.denoise, steps: genState.steps,
            cfg: genState.cfg, guidance: genState.guidance, seed: genState.seed,
            negative: genState.negative, sampler: genState.sampler, scheduler: genState.scheduler, batch: genState.batch,
            model: genState.model, arch: genState.arch, frames: genState.frames, fps: genState.fps,
            capsPositive: genState.capsPositive, capsNegative: genState.capsNegative,
            noiseFixture: genState.noiseFixture, includeAudio: genState.includeAudio,
            editMode: genState.editMode, editEngine: genState.editEngine,
            styleEntireImage: genState.styleEntireImage,
            lanpaintEngine: genState.lanpaintEngine,
            lanpaintNumSteps: genState.lanpaintNumSteps,
            lanpaintLambda: genState.lanpaintLambda,
            lanpaintStepSize: genState.lanpaintStepSize,
            lanpaintBeta: genState.lanpaintBeta,
            lanpaintFriction: genState.lanpaintFriction,
            lanpaintPromptMode: genState.lanpaintPromptMode,
            lanpaintBlendOverlap: genState.lanpaintBlendOverlap,
            lanpaintEarlyStop: genState.lanpaintEarlyStop,
            lanpaintInnerThreshold: genState.lanpaintInnerThreshold,
            lanpaintInnerPatience: genState.lanpaintInnerPatience,
            loras: canvasLoras.map(function (lora) {
                return { name: lora.name, strength: Number(lora.strength), enabled: lora.enabled !== false };
            }) }).then(function (state) {
            state.references = typeof CanvasRefImages !== 'undefined' ? CanvasRefImages.serialize() : [];
            state.toolSettings = {
                brushSize: brushSize,
                brushColor: brushColor,
                brushHardness: brushHardness,
                brushOpacity: CanvasTools.getBrushOpacity(),
                shapeMode: CanvasTools.getShapeMode(),
                shapeSubtract: CanvasTools.getShapeSubtract(),
                gradientMode: CanvasTools.getGradientMode(),
                gradientColors: CanvasTools.getGradientColors(),
                gradientClip: CanvasTools.getGradientClip(),
                lassoMode: CanvasTools.getLassoMode(),
                lassoSubtract: CanvasTools.getLassoSubtract(),
                lassoAutoMask: CanvasTools.getLassoAutoMask()
            };
            LayerSerializer.downloadAsFile(state);
        });
    }
    function loadCanvasSession(file) {
        LayerSerializer.loadFromFile(file).then(function (state) {
            if (!state || !stage)
                return;
            // Clear existing
            canvasLayers.forEach(function (l) { l.konvaLayer.destroy(); });
            canvasLayers = [];
            // Rebuild
            state.layers.forEach(function (saved) {
                var konvaLayer = new Konva.Layer();
                stage.add(konvaLayer);
                if (uiLayer && uiLayer.parent)
                    uiLayer.moveToTop();
                var data = LayerValidation.sanitise(JSON.parse(JSON.stringify(saved.data)));
                var cl = { data: data, konvaLayer: konvaLayer };
                konvaLayer.opacity(data.opacity);
                if (!data.visible)
                    konvaLayer.hide();
                canvasLayers.push(cl);
                if (saved.imageData && saved.imageData !== 'data:,') {
                    var img = new Image();
                    img.onload = function () {
                        konvaLayer.add(new Konva.Image({ image: img, x: 0, y: 0 }));
                        konvaLayer.batchDraw();
                    };
                    img.src = saved.imageData;
                }
                if (data.type === 'text') {
                    var td = data;
                    var kText = new Konva.Text({
                        x: td.position.x, y: td.position.y,
                        text: td.text, fontSize: td.fontSize,
                        fontFamily: td.fontFamily, fill: td.color,
                        fontStyle: td.fontWeight, align: td.alignment,
                        lineHeight: td.lineHeight, draggable: true,
                    });
                    konvaLayer.add(kText);
                    attachInlineTextEdit(kText, data, konvaLayer);
                }
            });
            // Restore bbox
            if (state.bbox && boundingBox) {
                boundingBox.x(state.bbox.x);
                boundingBox.y(state.bbox.y);
                boundingBox.width(state.bbox.width);
                boundingBox.height(state.bbox.height);
                updateHandles();
                updateSizeLabel();
            }
            // Restore gen settings
            if (state.genSettings) {
                genState.prompt = state.genSettings.prompt || '';
                genState.denoise = state.genSettings.denoise;
                genState.steps = state.genSettings.steps;
                genState.cfg = state.genSettings.cfg;
                genState.guidance = state.genSettings.guidance;
                genState.seed = state.genSettings.seed;
                genState.negative = state.genSettings.negative || '';
                genState.sampler = state.genSettings.sampler || genState.sampler;
                genState.scheduler = state.genSettings.scheduler || genState.scheduler;
                genState.batch = state.genSettings.batch || 1;
                genState.model = state.genSettings.model || genState.model;
                genState.arch = state.genSettings.arch || genState.arch;
                genState.frames = state.genSettings.frames || genState.frames;
                genState.fps = state.genSettings.fps || genState.fps;
                genState.capsPositive = state.genSettings.capsPositive || '';
                genState.capsNegative = state.genSettings.capsNegative || '';
                genState.noiseFixture = state.genSettings.noiseFixture || '';
                genState.includeAudio = state.genSettings.includeAudio === true;
                genState.editMode = state.genSettings.editMode || genState.editMode;
                genState.editEngine = state.genSettings.editEngine || genState.editEngine;
                genState.styleEntireImage = state.genSettings.styleEntireImage === true;
                genState.lanpaintEngine = state.genSettings.lanpaintEngine || genState.lanpaintEngine;
                genState.lanpaintNumSteps = Number.isFinite(Number(state.genSettings.lanpaintNumSteps)) ? Number(state.genSettings.lanpaintNumSteps) : genState.lanpaintNumSteps;
                genState.lanpaintLambda = Number.isFinite(Number(state.genSettings.lanpaintLambda)) ? Number(state.genSettings.lanpaintLambda) : genState.lanpaintLambda;
                genState.lanpaintStepSize = Number.isFinite(Number(state.genSettings.lanpaintStepSize)) ? Number(state.genSettings.lanpaintStepSize) : genState.lanpaintStepSize;
                genState.lanpaintBeta = Number.isFinite(Number(state.genSettings.lanpaintBeta)) ? Number(state.genSettings.lanpaintBeta) : genState.lanpaintBeta;
                genState.lanpaintFriction = Number.isFinite(Number(state.genSettings.lanpaintFriction)) ? Number(state.genSettings.lanpaintFriction) : genState.lanpaintFriction;
                genState.lanpaintPromptMode = state.genSettings.lanpaintPromptMode || genState.lanpaintPromptMode;
                genState.lanpaintBlendOverlap = Number.isFinite(Number(state.genSettings.lanpaintBlendOverlap)) ? Number(state.genSettings.lanpaintBlendOverlap) : genState.lanpaintBlendOverlap;
                genState.lanpaintEarlyStop = Number.isFinite(Number(state.genSettings.lanpaintEarlyStop)) ? Number(state.genSettings.lanpaintEarlyStop) : genState.lanpaintEarlyStop;
                genState.lanpaintInnerThreshold = Number.isFinite(Number(state.genSettings.lanpaintInnerThreshold)) ? Number(state.genSettings.lanpaintInnerThreshold) : genState.lanpaintInnerThreshold;
                genState.lanpaintInnerPatience = Number.isFinite(Number(state.genSettings.lanpaintInnerPatience)) ? Number(state.genSettings.lanpaintInnerPatience) : genState.lanpaintInnerPatience;
                lanpaintModeInitialized = state.genSettings.editMode === 'inpaint';
                canvasLoras = Array.isArray(state.genSettings.loras) ? state.genSettings.loras.map(function (lora) {
                    return {
                        name: String(lora.name || ''),
                        strength: Number.isFinite(Number(lora.strength)) ? Number(lora.strength) : 1,
                        enabled: lora.enabled !== false
                    };
                }).filter(function (lora) { return lora.name.length > 0; }) : [];
                if (els.prompt)
                    els.prompt.value = genState.prompt;
                if (els.negative)
                    els.negative.value = genState.negative;
                if (els.denoise)
                    els.denoise.value = String(genState.denoise);
                if (els.steps)
                    els.steps.value = String(genState.steps);
                if (els.model && genState.model)
                    els.model.value = genState.model;
                if (els.sampler)
                    els.sampler.value = genState.sampler;
                if (els.scheduler)
                    els.scheduler.value = genState.scheduler;
                if (els.seed)
                    els.seed.value = String(genState.seed);
                if (els.batch)
                    els.batch.value = String(genState.batch);
                if (els.framesInput)
                    els.framesInput.value = String(genState.frames);
                if (els.framesRange)
                    els.framesRange.value = String(genState.frames);
                if (els.fpsInput)
                    els.fpsInput.value = String(genState.fps);
                if (els.fpsRange)
                    els.fpsRange.value = String(genState.fps);
                if (els.capsPositive)
                    els.capsPositive.value = genState.capsPositive;
                if (els.capsNegative)
                    els.capsNegative.value = genState.capsNegative;
                if (els.noiseFixture)
                    els.noiseFixture.value = genState.noiseFixture;
                if (els.includeAudio)
                    els.includeAudio.checked = genState.includeAudio;
                if (els.styleEntireImage)
                    els.styleEntireImage.checked = genState.styleEntireImage;
                if (els.editMode)
                    els.editMode.value = genState.editMode;
                if (els.lanpaintEngine)
                    els.lanpaintEngine.value = genState.lanpaintEngine;
                if (els.lanpaintNumSteps)
                    els.lanpaintNumSteps.value = String(genState.lanpaintNumSteps);
                if (els.lanpaintLambda)
                    els.lanpaintLambda.value = String(genState.lanpaintLambda);
                if (els.lanpaintStepSize)
                    els.lanpaintStepSize.value = String(genState.lanpaintStepSize);
                if (els.lanpaintBeta)
                    els.lanpaintBeta.value = String(genState.lanpaintBeta);
                if (els.lanpaintFriction)
                    els.lanpaintFriction.value = String(genState.lanpaintFriction);
                if (els.lanpaintPromptMode)
                    els.lanpaintPromptMode.value = genState.lanpaintPromptMode;
                if (els.lanpaintBlendOverlap)
                    els.lanpaintBlendOverlap.value = String(genState.lanpaintBlendOverlap);
                if (els.lanpaintEarlyStop)
                    els.lanpaintEarlyStop.value = String(genState.lanpaintEarlyStop);
                if (els.lanpaintInnerThreshold)
                    els.lanpaintInnerThreshold.value = String(genState.lanpaintInnerThreshold);
                if (els.lanpaintInnerPatience)
                    els.lanpaintInnerPatience.value = String(genState.lanpaintInnerPatience);
                updateCanvasUIForArch(genState.arch);
                updateEditWorkspace();
                renderCanvasLoras();
            }
            if (typeof CanvasRefImages !== 'undefined')
                CanvasRefImages.restore(state.references || []);
            if (state.toolSettings) {
                var tools = state.toolSettings;
                brushSize = Number(tools.brushSize == null ? brushSize : tools.brushSize);
                brushColor = tools.brushColor || brushColor;
                brushHardness = Number(tools.brushHardness == null ? brushHardness : tools.brushHardness);
                CanvasTools.setBrushOpacity(Number(tools.brushOpacity == null ? CanvasTools.getBrushOpacity() : tools.brushOpacity));
                CanvasTools.setShapeMode(tools.shapeMode || 'rectangle');
                CanvasTools.setShapeSubtract(!!tools.shapeSubtract);
                CanvasTools.setGradientMode(tools.gradientMode || 'linear');
                if (tools.gradientColors)
                    CanvasTools.setGradientColors(tools.gradientColors.start, tools.gradientColors.end);
                CanvasTools.setGradientClip(tools.gradientClip !== false);
                CanvasTools.setLassoMode(tools.lassoMode || 'freehand');
                CanvasTools.setLassoSubtract(!!tools.lassoSubtract);
                CanvasTools.setLassoAutoMask(tools.lassoAutoMask !== false);
                if (els.brushSizeInput)
                    els.brushSizeInput.value = String(brushSize);
                if (els.brushSizeVal)
                    els.brushSizeVal.textContent = String(brushSize);
                if (els.brushColorInput)
                    els.brushColorInput.value = brushColor;
                var shapeModeInput = document.getElementById('cv-shape-mode');
                var gradientModeInput = document.getElementById('cv-gradient-mode');
                var lassoModeInput = document.getElementById('cv-lasso-mode');
                if (shapeModeInput)
                    shapeModeInput.value = CanvasTools.getShapeMode();
                if (gradientModeInput)
                    gradientModeInput.value = CanvasTools.getGradientMode();
                if (lassoModeInput)
                    lassoModeInput.value = CanvasTools.getLassoMode();
                document.getElementById('cv-shape-subtract').classList.toggle('active', CanvasTools.getShapeSubtract());
                document.getElementById('cv-gradient-clip').classList.toggle('active', CanvasTools.getGradientClip());
                document.getElementById('cv-lasso-subtract').classList.toggle('active', CanvasTools.getLassoSubtract());
                document.getElementById('cv-lasso-auto-mask').classList.toggle('active', CanvasTools.getLassoAutoMask());
            }
            activeLayerId = state.activeLayerId;
            var maxId = 0;
            canvasLayers.forEach(function (l) { if (l.data.id > maxId)
                maxId = l.data.id; });
            LayerDefaults.setIdCounter(maxId);
            renderLayerList();
            stage.batchDraw();
            History.push();
        });
    }
    // ── Models and LoRA registry ──
    function loraArchForCanvas(arch) {
        if (arch === 'ltxv')
            return 'ltx2';
        if (arch === 'qwen')
            return 'qwen-image';
        return arch;
    }
    function canvasBackendProfile() {
        if (!canvasCapabilities || !Array.isArray(canvasCapabilities.backends) || genState.arch === 'ltxv')
            return null;
        var backend = ModelUtils.backendForArch(genState.arch);
        return canvasCapabilities.backends.find(function (profile) {
            return profile && profile.backend === backend;
        }) || null;
    }
    function canvasFeature(name) {
        var profile = canvasBackendProfile();
        var feature = profile && profile.features && profile.features[name];
        return feature || { supported: false, policy: 'fail_loud', reason: 'The selected backend does not advertise ' + name };
    }
    function canvasFeatureReason(name) {
        var feature = canvasFeature(name);
        return feature.reason || (name + ' is not supported by the selected backend');
    }
    function updateCanvasCapabilityUI() {
        if (!els.capabilityNote)
            return;
        if (genState.editMode === 'style') {
            els.capabilityNote.textContent = 'Style mode routes through the selected 1024×1024 ' +
                flowEditEngineLabel(genState.editEngine) +
                ' FlowEdit worker after pure-Mojo visual analysis.';
            els.capabilityNote.style.display = 'block';
        }
        else if (genState.editMode === 'inpaint') {
            var maskedDefinition = maskedEditEngineDefinition(genState.lanpaintEngine);
            var maskedFeature = canvasFeature('inpaint');
            els.capabilityNote.textContent = maskedDefinition ?
                maskedDefinition.label + ': ' + (maskedFeature.note || maskedFeature.mask_contract || 'masked editing is admitted by this backend') :
                'No admitted masked-edit engine is available.';
            els.capabilityNote.style.display = 'block';
        }
        else if (genState.arch === 'ltxv') {
            els.capabilityNote.textContent = 'Mojo text-to-video request profile; Canvas pixels are not consumed.';
            els.capabilityNote.style.display = 'block';
        }
        else {
            var profile = canvasBackendProfile();
            if (!profile) {
                els.capabilityNote.textContent = 'No capability profile is available for this model.';
                els.capabilityNote.style.display = 'block';
            }
            else {
                var unsupported = [];
                ['image_to_image', 'inpaint', 'controlnet', 'image_conditioning'].forEach(function (name) {
                    if (!canvasFeature(name).supported)
                        unsupported.push(name.replaceAll('_', ' '));
                });
                els.capabilityNote.textContent = unsupported.length ?
                    'Generation guard: ' + unsupported.join(', ') + ' unavailable.' : '';
                els.capabilityNote.style.display = unsupported.length ? 'block' : 'none';
            }
        }
        var loraSupported = genState.arch === 'ltxv' || canvasFeature('lora').supported;
        els.loraSection.style.display = loraSupported ? 'flex' : 'none';
        var controlItem = document.querySelector('.cv-layer-type-item[data-type="control"]');
        if (controlItem) {
            var controlSupported = genState.arch !== 'ltxv' && canvasFeature('controlnet').supported;
            controlItem.classList.toggle('capability-unavailable', !controlSupported);
            controlItem.title = controlSupported ? '' : (genState.arch === 'ltxv' ? 'LTX2 request runner is text-to-video only' : canvasFeatureReason('controlnet'));
        }
        var maskItem = document.querySelector('.cv-layer-type-item[data-type="mask"]');
        if (maskItem) {
            var inpaintSupported = genState.arch !== 'ltxv' && canvasFeature('inpaint').supported;
            maskItem.classList.toggle('capability-unavailable', !inpaintSupported);
            maskItem.title = inpaintSupported ? '' : (genState.arch === 'ltxv' ? 'LTX2 request runner is text-to-video only' : canvasFeatureReason('inpaint'));
        }
        if (typeof CanvasRefImages !== 'undefined' && CanvasRefImages.setCompatibility) {
            var refsSupported = genState.editMode === 'style' ||
                (genState.arch !== 'ltxv' && canvasFeature('image_conditioning').supported);
            CanvasRefImages.setCompatibility(refsSupported, genState.editMode === 'style' ? '' :
                (genState.arch === 'ltxv' ? 'LTX2 request runner is text-to-video only' : canvasFeatureReason('image_conditioning')));
        }
    }
    function validateCanvasGenerationFeatures(hasContent, hasMask) {
        if (genState.arch === 'ltxv')
            return hasContent ? 'The selected video runner is text-to-video only; clear the Canvas bounding box before generating' : '';
        var loras = enabledCanvasLoras();
        if (loras.length > 0 && !canvasFeature('lora').supported)
            return canvasFeatureReason('lora');
        if (loras.length > 1 && !canvasFeature('multi_lora').supported)
            return canvasFeatureReason('multi_lora');
        if (collectControlLayers().length > 0 && !canvasFeature('controlnet').supported)
            return canvasFeatureReason('controlnet');
        if (collectIPALayers().length > 0 && !canvasFeature('image_conditioning').supported)
            return canvasFeatureReason('image_conditioning');
        if (hasMask && !canvasFeature('inpaint').supported)
            return canvasFeatureReason('inpaint');
        if (hasContent && genState.editMode !== 'inpaint' && !canvasFeature('image_to_image').supported)
            return canvasFeatureReason('image_to_image');
        return '';
    }
    function compatibleCanvasLoras() {
        var wanted = loraArchForCanvas(genState.arch);
        return availableCanvasLoras.filter(function (lora) {
            var arch = String(lora.arch || '').toLowerCase();
            return arch === wanted;
        });
    }
    function enabledCanvasLoras() {
        var compatible = compatibleCanvasLoras().map(function (lora) { return lora.name; });
        return canvasLoras.filter(function (lora) {
            return lora.enabled !== false && compatible.indexOf(lora.name) >= 0;
        }).map(function (lora) {
            return { name: lora.name, strength: Number(lora.strength) };
        });
    }
    function appendLoraOptions(select, choices, selected) {
        select.innerHTML = '';
        choices.forEach(function (lora) {
            var option = document.createElement('option');
            option.value = lora.name;
            option.textContent = lora.name;
            option.selected = lora.name === selected;
            select.appendChild(option);
        });
    }
    function renderCanvasLoras() {
        if (!els.loraPicker || !els.loraList)
            return;
        var choices = compatibleCanvasLoras();
        appendLoraOptions(els.loraPicker, choices, els.loraPicker.value);
        els.loraPicker.disabled = choices.length === 0;
        els.loraAdd.disabled = choices.length === 0;
        if (els.loraCompat)
            els.loraCompat.textContent = choices.length + ' compatible';
        els.loraList.innerHTML = '';
        var compatibleNames = choices.map(function (lora) { return lora.name; });
        var rows = canvasLoras.filter(function (lora) {
            return compatibleNames.indexOf(lora.name) >= 0;
        });
        if (rows.length === 0) {
            var empty = document.createElement('div');
            empty.className = 'cv-lora-empty';
            empty.textContent = choices.length ? 'No LoRA loaders added' : 'No compatible LoRAs found';
            els.loraList.appendChild(empty);
            return;
        }
        rows.forEach(function (lora) {
            var row = document.createElement('div');
            row.className = 'cv-lora-row';
            var enabled = document.createElement('input');
            enabled.type = 'checkbox';
            enabled.checked = lora.enabled !== false;
            enabled.title = 'Enable LoRA';
            enabled.addEventListener('change', function () { lora.enabled = enabled.checked; });
            var select = document.createElement('select');
            select.className = 'cv-select cv-lora-select';
            appendLoraOptions(select, choices, lora.name);
            select.addEventListener('change', function () { lora.name = select.value; });
            var strength = document.createElement('input');
            strength.type = 'number';
            strength.className = 'cv-number-input cv-lora-strength';
            strength.min = '-10';
            strength.max = '10';
            strength.step = '0.05';
            strength.value = String(Number(lora.strength));
            strength.title = 'Model strength';
            strength.addEventListener('input', function () {
                var value = Number(strength.value);
                lora.strength = Number.isFinite(value) ? Math.max(-10, Math.min(10, value)) : 1;
            });
            var remove = document.createElement('button');
            remove.type = 'button';
            remove.className = 'cv-lora-remove';
            remove.title = 'Remove LoRA loader';
            remove.textContent = '×';
            remove.addEventListener('click', function () {
                canvasLoras = canvasLoras.filter(function (candidate) { return candidate !== lora; });
                renderCanvasLoras();
            });
            row.appendChild(enabled);
            row.appendChild(select);
            row.appendChild(strength);
            row.appendChild(remove);
            els.loraList.appendChild(row);
        });
    }
    function loadCanvasLoraRegistry() {
        return fetch('/models', { cache: 'no-store' })
            .then(function (resp) {
            if (!resp.ok)
                throw new Error('LoRA registry HTTP ' + resp.status);
            return resp.json();
        })
            .then(function (registry) {
            availableCanvasLoras = Array.isArray(registry.loras) ? registry.loras.filter(function (lora) {
                return lora && typeof lora.name === 'string' && lora.name.length > 0;
            }) : [];
            renderCanvasLoras();
        })
            .catch(function (error) {
            availableCanvasLoras = [];
            renderCanvasLoras();
            if (els.loraCompat)
                els.loraCompat.textContent = error.message;
        });
    }
    function addCanvasLora() {
        var name = els.loraPicker && els.loraPicker.value;
        if (!name)
            return;
        canvasLoras.push({ name: name, strength: 1, enabled: true });
        renderCanvasLoras();
    }
    function serializedNodeValues(node, objectInfo) {
        var definition = objectInfo && objectInfo[node.type];
        var required = definition && definition.input && definition.input.required;
        var names = required ? Object.keys(required).filter(function (name) {
            var spec = required[name];
            return !(Array.isArray(spec) && spec.length === 1 && typeof spec[0] === 'string' &&
                ['MODEL', 'CLIP', 'VAE', 'IMAGE', 'LATENT', 'CONDITIONING'].indexOf(spec[0]) >= 0);
        }) : [];
        var values = {};
        names.forEach(function (name, index) {
            values[name] = (node.widgets_values || [])[index];
        });
        return values;
    }
    function loadLtx2TemplateProfile() {
        els.loadLtx2Template.disabled = true;
        Promise.all([
            fetch('/templates', { cache: 'no-store' }).then(function (resp) {
                if (!resp.ok)
                    throw new Error('template catalog HTTP ' + resp.status);
                return resp.json();
            }),
            ModelUtils.loadObjectInfo()
        ]).then(function (results) {
            var catalog = Array.isArray(results[0]) ? results[0] : [];
            var entry = catalog.find(function (candidate) {
                return candidate && /ltx2.*lora/i.test(candidate.name || candidate.file || '');
            });
            if (!entry || !entry.url)
                throw new Error('no active LTX2 LoRA template is available');
            return fetch(entry.url, { cache: 'no-store' }).then(function (resp) {
                if (!resp.ok)
                    throw new Error('template HTTP ' + resp.status);
                return resp.json();
            }).then(function (template) {
                return { template: template, objectInfo: results[1] };
            });
        }).then(function (loaded) {
            var nodes = Array.isArray(loaded.template.nodes) ? loaded.template.nodes : [];
            var loader = nodes.find(function (node) { return node.type === 'LTXVLoader'; });
            var sampler = nodes.find(function (node) { return node.type === 'LTXVSampler'; });
            if (!loader || !sampler)
                throw new Error('template is missing LTXVLoader or LTXVSampler');
            var loaderValues = serializedNodeValues(loader, loaded.objectInfo);
            var samplerValues = serializedNodeValues(sampler, loaded.objectInfo);
            var authoredSampler = String(samplerValues.sampler || '');
            var authoredScheduler = String(samplerValues.scheduler || '');
            genState.model = String(loaderValues.checkpoint_path || '');
            genState.prompt = String(samplerValues.prompt || '');
            genState.negative = String(samplerValues.negative_prompt || '');
            genState.steps = Number(samplerValues.steps);
            genState.cfg = Number(samplerValues.cfg);
            genState.seed = Number(samplerValues.seed);
            genState.frames = Number(samplerValues.num_frames);
            genState.fps = Number(samplerValues.frame_rate);
            genState.sampler = authoredSampler;
            genState.scheduler = authoredScheduler;
            genState.capsPositive = String(samplerValues.caps_positive || '');
            genState.capsNegative = String(samplerValues.caps_negative || '');
            genState.noiseFixture = String(samplerValues.noise_fixture || '');
            genState.includeAudio = samplerValues.include_audio === true;
            canvasLoras = nodes.filter(function (node) {
                return node.type === 'LoraLoader' || node.type === 'LoraLoaderModelOnly';
            }).map(function (node) {
                var values = serializedNodeValues(node, loaded.objectInfo);
                return { name: String(values.lora_name || ''), strength: Number(values.strength_model), enabled: true };
            }).filter(function (lora) { return lora.name.length > 0; });
            if (!Number.isFinite(genState.steps) || !Number.isFinite(genState.frames) || !Number.isFinite(genState.fps))
                throw new Error('template generation values are incomplete');
            els.model.value = genState.model;
            updateTopbarModel(genState.model);
            updateCanvasUIForArch('ltxv');
            genState.sampler = authoredSampler;
            genState.scheduler = authoredScheduler;
            els.prompt.value = genState.prompt;
            els.negative.value = genState.negative;
            els.steps.value = String(genState.steps);
            els.stepsRange.value = String(genState.steps);
            els.cfg.value = String(genState.cfg);
            els.cfgRange.value = String(genState.cfg);
            els.seed.value = String(genState.seed);
            els.framesInput.value = String(genState.frames);
            els.framesRange.value = String(genState.frames);
            els.fpsInput.value = String(genState.fps);
            els.fpsRange.value = String(genState.fps);
            els.sampler.value = genState.sampler;
            els.scheduler.value = genState.scheduler;
            els.capsPositive.value = genState.capsPositive;
            els.capsNegative.value = genState.capsNegative;
            els.noiseFixture.value = genState.noiseFixture;
            els.includeAudio.checked = genState.includeAudio;
            if (boundingBox) {
                boundingBox.width(ModelUtils.clampVideoDimension(Number(samplerValues.width)));
                boundingBox.height(ModelUtils.clampVideoDimension(Number(samplerValues.height)));
                updateHandles();
                updateSizeLabel();
                updateBboxInputs();
                stage.batchDraw();
            }
            updateCanvasDurationHint();
            renderCanvasLoras();
        }).catch(function (error) {
            showError('LTX2 template failed: ' + error.message);
        }).finally(function () {
            els.loadLtx2Template.disabled = false;
        });
    }
    function loadModels() {
        Promise.all([ModelUtils.fetchAllModels(), ModelUtils.loadCapabilities()])
            .then(function (loaded) {
            var models = loaded[0];
            canvasCapabilities = loaded[1];
            if (!models.length)
                throw new Error('empty');
            els.model.innerHTML = '';
            models.forEach(function (m) {
                var opt = document.createElement('option');
                opt.value = m.name;
                opt.textContent = m.name.split('/').pop() || m.name;
                els.model.appendChild(opt);
            });
            // Same preferred-image-family default as the Generate tab —
            // models[0] can be a VIDEO model (LTX/Wan) whose graph the image
            // queue rejects.
            var preferred = ['krea2', 'klein', 'flux2', 'zimage', 'ideogram', 'sdxl', 'sd3', 'qwen'];
            var pick = models[0];
            outer: for (var pi = 0; pi < preferred.length; pi++) {
                for (var mi = 0; mi < models.length; mi++) {
                    var nm = (models[mi].name + ' ' + (models[mi].arch || '')).toLowerCase();
                    if (nm.indexOf(preferred[pi]) >= 0) { pick = models[mi]; break outer; }
                }
            }
            els.model.value = pick.name;
            genState.model = pick.name;
            updateTopbarModel(pick.name);
            updateCanvasUIForArch(ModelUtils.detectArchFromFilename(pick.name));
            populateMaskedEditEngineOptions();
            if (genState.editMode === 'flowedit' || genState.editMode === 'style')
                syncCanvasModelFromFlowEditEngine();
            else if (genState.editMode === 'inpaint')
                syncCanvasModelFromLanPaintEngine();
        })
            .catch(function () {
            els.model.innerHTML = '<option disabled selected>No models found</option>';
        });
        loadCanvasLoraRegistry();
    }
    function updateTopbarModel(modelName) {
        var badge = document.querySelector('.model-badge');
        if (!badge)
            return;
        var short = modelName ? modelName.split('/').pop().replace(/\.\w+$/, '') : 'No model loaded';
        badge.textContent = short;
        if (typeof CanvasStatusBar !== 'undefined')
            CanvasStatusBar.updateModel(modelName);
    }
    // ── Generation ──
    function updateCanvasUIForArch(arch) {
        genState.arch = arch;
        var isFlux = arch === 'flux';
        var isVideo = arch === 'ltxv' || arch === 'wan' || arch === 'bernini';
        var isLtx2 = arch === 'ltxv';
        els.cfgRow.style.display = (isFlux || isVideo) ? 'none' : 'flex';
        els.guidanceRow.style.display = isFlux ? 'flex' : 'none';
        els.videoSection.style.display = isVideo ? 'block' : 'none';
        els.denoiseRow.style.display = isVideo ? 'none' : 'flex';
        els.denoiseHelp.style.display = isVideo ? 'none' : 'block';
        els.ltx2Section.style.display = isLtx2 ? 'flex' : 'none';
        els.batch.style.display = isVideo ? 'none' : '';
        els.batchLabel.style.display = isVideo ? 'none' : '';
        if (isLtx2) {
            setCanvasSelectOptions(els.sampler, [
                { value: 'res2s', label: 'Res2S' },
                { value: 'euler', label: 'Euler (not compiled)' }
            ], 'res2s');
            setCanvasSelectOptions(els.scheduler, [
                { value: 'ltx2', label: 'LTX2' },
                { value: 'ltx2_distilled', label: 'LTX2 Distilled (not compiled)' }
            ], 'ltx2');
        }
        else {
            setCanvasSelectOptions(els.sampler, [
                { value: 'euler', label: 'Euler' },
                { value: 'euler_ancestral', label: 'Euler Ancestral' },
                { value: 'dpmpp_2m', label: 'DPM++ 2M' }
            ], 'euler');
            setCanvasSelectOptions(els.scheduler, [
                { value: 'simple', label: 'Simple' },
                { value: 'normal', label: 'Normal' },
                { value: 'karras', label: 'Karras' }
            ], 'simple');
        }
        // Update generate button label
        els.generateBtn.textContent = genState.editMode === 'style' ? 'Apply reference style' :
            (genState.editMode === 'flowedit' ? 'Run FlowEdit' :
                (genState.editMode === 'inpaint' ? 'Edit masked area' : (isVideo ? 'Generate Video' : 'Generate')));
        if (arch === 'bernini') {
            genState.width = 848;
            genState.height = 480;
            genState.frames = 81;
            genState.fps = 16;
            genState.steps = 40;
            genState.cfg = 4.0;
            genState.scheduler = 'uni_pc';
        }
        // Update size label to show/hide frame count
        updateSizeLabel();
        if (isVideo)
            updateCanvasDurationHint();
        renderCanvasLoras();
        updateCanvasCapabilityUI();
    }
    function setCanvasSelectOptions(select, options, fallback) {
        var prior = select.value;
        select.innerHTML = '';
        options.forEach(function (item) {
            var option = document.createElement('option');
            option.value = item.value;
            option.textContent = item.label;
            select.appendChild(option);
        });
        select.value = options.some(function (item) { return item.value === prior; }) ? prior : fallback;
        if (select === els.sampler)
            genState.sampler = select.value;
        if (select === els.scheduler)
            genState.scheduler = select.value;
    }
    function updateCanvasDurationHint() {
        if (!els.durationHint)
            return;
        var secs = (genState.frames / genState.fps).toFixed(1);
        els.durationHint.textContent = '\u2248 ' + secs + 's at ' + genState.fps + 'fps';
    }
    function captureBoundingBoxDataURL(pixelRatio, hideMaskLayers) {
        var hiddenMasks = [];
        if (hideMaskLayers) {
            canvasLayers.forEach(function (layer) {
                if (layer.data.type === 'mask' && layer.konvaLayer.visible()) {
                    hiddenMasks.push(layer);
                    layer.konvaLayer.hide();
                }
            });
        }
        var oldPosition = { x: stage.x(), y: stage.y() };
        var oldScale = { x: stage.scaleX(), y: stage.scaleY() };
        var uiWasVisible = uiLayer.visible();
        var backgroundWasVisible = backgroundLayer.visible();
        try {
            uiLayer.hide();
            backgroundLayer.hide();
            // Konva's crop rectangle is expressed in export-canvas pixels.
            // Neutralize the interactive pan/zoom transform so a fitted 1024
            // scene exports as 1024 pixels instead of a small viewport-scaled
            // thumbnail centered in a black frame.
            stage.position({ x: 0, y: 0 });
            stage.scale({ x: 1, y: 1 });
            stage.draw();
            return stage.toDataURL({
                x: boundingBox.x(), y: boundingBox.y(),
                width: boundingBox.width(), height: boundingBox.height(),
                pixelRatio: pixelRatio
            });
        }
        finally {
            stage.position(oldPosition);
            stage.scale(oldScale);
            if (uiWasVisible)
                uiLayer.show();
            if (backgroundWasVisible)
                backgroundLayer.show();
            hiddenMasks.forEach(function (layer) { layer.konvaLayer.show(); });
            stage.draw();
        }
    }
    function getMaskLayer() {
        for (var i = 0; i < canvasLayers.length; i++) {
            if (canvasLayers[i].data.type === 'mask' && canvasLayers[i].data.visible)
                return canvasLayers[i];
        }
        return null;
    }
    function exportMaskAsBW() {
        var maskLayer = getMaskLayer();
        if (!maskLayer)
            return Promise.resolve(null);
        var children = maskLayer.konvaLayer.getChildren();
        if (children.length === 0)
            return Promise.resolve(null);
        return new Promise(function (resolve) {
            // Export this layer directly. Exporting the whole stage makes its
            // transparent background opaque on some browsers, which turns a
            // small painted region into a full-frame mask.
            var bw = Math.round(boundingBox.width());
            var bh = Math.round(boundingBox.height());
            var oldPosition = { x: stage.x(), y: stage.y() };
            var oldScale = { x: stage.scaleX(), y: stage.scaleY() };
            var maskDataURL = '';
            try {
                stage.position({ x: 0, y: 0 });
                stage.scale({ x: 1, y: 1 });
                stage.draw();
                maskDataURL = maskLayer.konvaLayer.toDataURL({
                    x: boundingBox.x(), y: boundingBox.y(),
                    width: bw, height: bh, pixelRatio: 1
                });
            }
            finally {
                stage.position(oldPosition);
                stage.scale(oldScale);
                stage.draw();
            }
            // Convert to B&W: any non-transparent pixel becomes white
            var offscreen = document.createElement('canvas');
            offscreen.width = bw;
            offscreen.height = bh;
            var ctx = offscreen.getContext('2d');
            var img = new Image();
            img.onload = function () {
                // Preserve the transparent background while reading the mask
                // alpha. Filling black first would make every pixel opaque and
                // accidentally turn every painted mask into a full-frame mask.
                ctx.clearRect(0, 0, bw, bh);
                ctx.drawImage(img, 0, 0, bw, bh);
                // Convert painted alpha to white/edit and transparency to
                // black/preserve, then make the uploaded grayscale mask opaque.
                var imageData = ctx.getImageData(0, 0, bw, bh);
                var data = imageData.data;
                var paintedPixels = 0;
                for (var i = 0; i < data.length; i += 4) {
                    var painted = data[i + 3] > 127;
                    if (painted) {
                        paintedPixels += 1;
                        data[i] = 255;
                        data[i + 1] = 255;
                        data[i + 2] = 255;
                    }
                    else {
                        data[i] = 0;
                        data[i + 1] = 0;
                        data[i + 2] = 0;
                    }
                    data[i + 3] = 255;
                }
                ctx.putImageData(imageData, 0, 0);
                var coverage = paintedPixels / (bw * bh);
                var fullFrameRequested = children.some(function (child) {
                    return child.name && child.name() === 'full-frame-mask';
                });
                resolve({
                    base64: offscreen.toDataURL('image/png').split(',')[1],
                    coverage: coverage,
                    fullFrameRequested: fullFrameRequested
                });
            };
            img.onerror = function () { resolve(null); };
            img.src = maskDataURL;
        });
    }
    function startGeneration() {
        if (canvasGenerating)
            return;
        if (genState.editMode === 'dynaedit') {
            showError('DynaEdit is Mojo-native, but its web request runner is not connected yet');
            return;
        }
        if (genState.editMode === 'flowedit') {
            startFlowEditGeneration();
            return;
        }
        if (genState.editMode === 'style') {
            startStyleFlowEditGeneration();
            return;
        }
        if (!genState.model) {
            showError('No model selected');
            return;
        }
        if (!genState.prompt.trim()) {
            showError('Enter a prompt');
            return;
        }
        if (genState.arch === 'ltxv' && !genState.capsPositive) {
            showError('LTX2 requires a prompt-matched conditioning artifact path');
            return;
        }
        setCanvasGenerating(true);
        var isVideo = isVideoArch();
        var maskedEditEngine = genState.editMode === 'inpaint' ? maskedEditEngineDefinition(genState.lanpaintEngine) : null;
        checkBboxContent().then(function (hasContent) {
            var maskLayerInfo = getMaskLayer();
            var hasMask = maskLayerInfo && maskLayerInfo.konvaLayer.getChildren().length > 0;
            if (genState.editMode === 'inpaint' && !hasMask) {
                showError('Paint or generate a mask before starting the inpaint');
                setCanvasGenerating(false);
                return;
            }
            if (genState.editMode === 'inpaint' && (!maskedEditEngine || genState.arch !== maskedEditEngine.arch)) {
                showError('The selected masked-edit engine does not match its registered model');
                setCanvasGenerating(false);
                return;
            }
            var featureError = validateCanvasGenerationFeatures(hasContent, hasMask);
            if (featureError) {
                showError(featureError);
                setCanvasGenerating(false);
                return;
            }
            var seed = genState.seed === -1 ? Math.floor(Math.random() * 4294967296) : genState.seed;
            var bw = isVideo ? ModelUtils.clampVideoDimension(boundingBox.width()) : ModelUtils.clampDimension(boundingBox.width());
            var bh = isVideo ? ModelUtils.clampVideoDimension(boundingBox.height()) : ModelUtils.clampDimension(boundingBox.height());
            var activeLoras = enabledCanvasLoras();
            if (maskedEditEngine && maskedEditEngine.maxLoras !== null && activeLoras.length > maskedEditEngine.maxLoras) {
                showError(maskedEditEngine.label + ' admits at most ' + maskedEditEngine.maxLoras + ' LoRA loader per request');
                setCanvasGenerating(false);
                return;
            }
            pendingCanvasMetadata = {
                prompt: genState.prompt,
                model: genState.model,
                arch: genState.arch,
                width: bw,
                height: bh,
                frames: isVideo ? genState.frames : null,
                fps: isVideo ? genState.fps : null,
                steps: genState.steps,
                cfg: genState.cfg,
                guidance: genState.guidance,
                seed: seed,
                negative: genState.negative,
                sampler: genState.sampler,
                scheduler: genState.scheduler,
                batch: genState.batch,
                loras: activeLoras,
                capsPositive: genState.capsPositive,
                capsNegative: genState.capsNegative,
                noiseFixture: genState.noiseFixture,
                includeAudio: genState.includeAudio,
                editMode: genState.editMode,
                maskedEdit: maskedEditEngine ? {
                    engine: maskedEditEngine.id,
                    label: maskedEditEngine.label,
                    backend: maskedEditEngine.backend,
                    model: genState.model
                } : null,
                lanpaint: maskedEditEngine && maskedEditEngine.lanpaint ? {
                    engine: genState.lanpaintEngine,
                    innerSteps: genState.lanpaintNumSteps,
                    lambda: genState.lanpaintLambda,
                    stepSize: genState.lanpaintStepSize,
                    beta: genState.lanpaintBeta,
                    friction: genState.lanpaintFriction,
                    promptMode: genState.lanpaintPromptMode,
                    blendOverlap: genState.lanpaintBlendOverlap,
                    earlyStop: genState.lanpaintEarlyStop,
                    innerThreshold: genState.lanpaintInnerThreshold,
                    innerPatience: genState.lanpaintInnerPatience
                } : null,
                submittedAt: new Date().toISOString()
            };
            if (!hasContent) {
                queueWorkflow(WorkflowBuilder.build({
                    model: genState.model || '', prompt: genState.prompt,
                    width: bw, height: bh,
                    steps: genState.steps, cfg: genState.cfg,
                    guidance: genState.guidance, seed: seed,
                    negPrompt: genState.negative, sampler: genState.sampler, scheduler: genState.scheduler, batch: genState.batch,
                    frames: genState.frames, fps: genState.fps, loras: activeLoras,
                    capsPositive: genState.capsPositive, capsNegative: genState.capsNegative,
                    noiseFixture: genState.noiseFixture, includeAudio: genState.includeAudio
                }));
            }
            else if (isVideo) {
                exportBoundingBoxRegion().then(function (base64) {
                    return uploadInitImage(base64);
                }).then(function (imageName) {
                    queueWorkflow(WorkflowBuilder.build({
                        model: genState.model || '', prompt: genState.prompt,
                        initImageName: imageName,
                        width: bw, height: bh,
                        steps: genState.steps, cfg: genState.cfg,
                        guidance: genState.guidance, seed: seed,
                        negPrompt: genState.negative, sampler: genState.sampler, scheduler: genState.scheduler, batch: genState.batch,
                        frames: genState.frames, fps: genState.fps, loras: activeLoras,
                        capsPositive: genState.capsPositive, capsNegative: genState.capsNegative,
                        noiseFixture: genState.noiseFixture, includeAudio: genState.includeAudio
                    }));
                }).catch(function (err) {
                    showError('Video init upload failed: ' + err.message);
                    setCanvasGenerating(false);
                });
            }
            else if (hasMask && !isVideo) {
                // Inpaint: export both init image and mask
                exportBoundingBoxRegion().then(function (initBase64) {
                    return uploadInitImage(initBase64).then(function (initName) {
                        return exportMaskAsBW().then(function (maskExport) {
                            if (!maskExport || !maskExport.base64)
                                throw new Error('The painted mask could not be exported');
                            if (maskExport.coverage <= 0)
                                throw new Error('The painted mask is empty');
                            if (maskExport.coverage >= 0.995 && !maskExport.fullFrameRequested)
                                throw new Error('The mask unexpectedly covers the entire image; clear it and paint the edit area again');
                            return uploadInitImage(maskExport.base64).then(function (maskName) {
                                queueWorkflow(WorkflowBuilder.buildInpaint({
                                    model: genState.model || '', prompt: genState.prompt,
                                    negPrompt: genState.negative, initImageName: initName, maskImageName: maskName,
                                    width: bw, height: bh,
                                    steps: genState.steps, cfg: genState.cfg,
                                    guidance: genState.guidance, denoise: maskedEditEngine && maskedEditEngine.lanpaint ? 1.0 : genState.denoise, seed: seed,
                                    sampler: genState.editMode === 'inpaint' ? 'euler' : genState.sampler,
                                    scheduler: genState.editMode === 'inpaint' ? 'simple' : genState.scheduler, batch: genState.batch,
                                    lanpaintNumSteps: genState.lanpaintNumSteps,
                                    lanpaintLambda: genState.lanpaintLambda,
                                    lanpaintStepSize: genState.lanpaintStepSize,
                                    lanpaintBeta: genState.lanpaintBeta,
                                    lanpaintFriction: genState.lanpaintFriction,
                                    lanpaintPromptMode: genState.lanpaintPromptMode,
                                    lanpaintBlendOverlap: genState.lanpaintBlendOverlap,
                                    lanpaintEarlyStop: genState.lanpaintEarlyStop,
                                    lanpaintInnerThreshold: genState.lanpaintInnerThreshold,
                                    lanpaintInnerPatience: genState.lanpaintInnerPatience,
                                    loras: activeLoras
                                }));
                            });
                        });
                    });
                }).catch(function (err) {
                    showError('Inpaint failed: ' + err.message);
                    setCanvasGenerating(false);
                });
            }
            else {
                exportBoundingBoxRegion().then(function (base64) {
                    return uploadInitImage(base64);
                }).then(function (imageName) {
                    queueWorkflow(WorkflowBuilder.buildImg2Img({
                        model: genState.model || '', prompt: genState.prompt,
                        initImageName: imageName, width: bw, height: bh,
                        steps: genState.steps, cfg: genState.cfg,
                        guidance: genState.guidance,
                        denoise: genState.denoise, seed: seed,
                        negPrompt: genState.negative, sampler: genState.sampler, scheduler: genState.scheduler, batch: genState.batch,
                        loras: activeLoras
                    }));
                }).catch(function (err) {
                    showError('Upload failed: ' + err.message);
                    setCanvasGenerating(false);
                });
            }
        });
    }
    function startFlowEditGeneration() {
        if (!editSourceDataUrl || editSourceIsVideo) {
            showError('Load a source image for FlowEdit');
            return;
        }
        if (!genState.prompt.trim()) {
            showError('Enter the target description in Prompt');
            return;
        }
        var size = flowEditEngineSize(genState.editEngine);
        var seed = genState.seed === -1 ? Math.floor(Math.random() * 4294967296) : genState.seed;
        setCanvasGenerating(true);
        if (typeof CanvasStatusBar !== 'undefined')
            CanvasStatusBar.updateGenStatus('Preparing source image');
        imageDataUrlAsPngBase64(editSourceDataUrl).then(function (base64) {
            return SerenityAPI.uploadImageDetails(base64, 'flowedit_source');
        }).then(function (upload) {
            var sourceDescription = genState.editSourcePrompt.trim();
            if (typeof CanvasStatusBar !== 'undefined')
                CanvasStatusBar.updateGenStatus(sourceDescription ? 'Writing target description' : 'Describing source and target');
            return prepareFlowEditCaptionPair(upload.path, sourceDescription, genState.prompt).then(function (captions) {
                captions.upload = upload;
                return captions;
            });
        }).then(function (prepared) {
            genState.editSourcePrompt = prepared.sourceDescription;
            els.editSourcePrompt.value = prepared.sourceDescription;
            var ideogram = genState.editEngine === 'ideogram4';
            var sourceConditioning = ideogram ? ideogramFlowEditCaption(prepared.sourceDescription) : prepared.sourceDescription;
            var targetConditioning = ideogram ? ideogramFlowEditCaption(prepared.targetDescription) : prepared.targetDescription;
            pendingCanvasMetadata = {
                prompt: genState.prompt,
                sourcePrompt: prepared.sourceDescription,
                flowEditTargetPrompt: prepared.targetDescription,
                model: genState.editEngine,
                arch: genState.editEngine,
                editMode: 'flowedit',
                width: size,
                height: size,
                steps: genState.steps,
                cfg: genState.cfg,
                seed: seed,
                submittedAt: new Date().toISOString()
            };
            if (typeof CanvasStatusBar !== 'undefined')
                CanvasStatusBar.updateGenStatus('Queueing FlowEdit');
            var workflow = WorkflowBuilder.buildFlowEdit({
                engine: genState.editEngine,
                initImageName: prepared.upload.path,
                sourcePrompt: sourceConditioning,
                sourceNegative: genState.editSourceNegative,
                prompt: targetConditioning,
                negPrompt: genState.negative,
                steps: genState.steps,
                nmax: genState.editNmax,
                nmin: genState.editNmin,
                srcCfg: genState.editSourceCfg,
                cfg: genState.cfg,
                seed: seed,
                autoMask: genState.editAutoMask,
                maskQ: genState.editMaskQ,
                maskDilate: genState.editMaskDilate,
                maskWarmup: genState.editMaskWarmup
            });
            queueWorkflow(workflow);
        }).catch(function (error) {
            showError('FlowEdit preparation failed: ' + error.message);
            setCanvasGenerating(false);
        });
    }
    function imageDataUrlAsPngBase64(dataUrl) {
        return new Promise(function (resolve, reject) {
            var image = new Image();
            image.onload = function () {
                var canvas = document.createElement('canvas');
                canvas.width = image.naturalWidth || image.width;
                canvas.height = image.naturalHeight || image.height;
                var context = canvas.getContext('2d');
                context.drawImage(image, 0, 0);
                resolve(canvas.toDataURL('image/png').split(',')[1]);
            };
            image.onerror = function () { reject(new Error('Could not decode the selected image')); };
            image.src = dataUrl;
        });
    }
    function loadDataUrlImage(dataUrl) {
        return new Promise(function (resolve, reject) {
            var image = new Image();
            image.onload = function () { resolve(image); };
            image.onerror = function () { reject(new Error('Could not decode a style analysis image')); };
            image.src = dataUrl;
        });
    }
    function drawContainedImage(context, image, x, y, width, height) {
        var iw = image.naturalWidth || image.width;
        var ih = image.naturalHeight || image.height;
        var scale = Math.min(width / iw, height / ih);
        var dw = iw * scale;
        var dh = ih * scale;
        context.drawImage(image, x + (width - dw) / 2, y + (height - dh) / 2, dw, dh);
    }
    function composeStyleAnalysisPngBase64(sourceDataUrl, styleDataUrl) {
        return Promise.all([loadDataUrlImage(sourceDataUrl), loadDataUrlImage(styleDataUrl)]).then(function (images) {
            var canvas = document.createElement('canvas');
            var size = STYLE_RESULT_SIZE;
            var dividerHeight = 40;
            var panelHeight = (size - dividerHeight) / 2;
            canvas.width = size;
            canvas.height = size;
            var context = canvas.getContext('2d');
            context.fillStyle = '#111117';
            context.fillRect(0, 0, size, size);
            drawContainedImage(context, images[0], 0, 0, size, panelHeight);
            drawContainedImage(context, images[1], 0, panelHeight + dividerHeight, size, panelHeight);
            context.fillStyle = '#08080c';
            context.fillRect(0, panelHeight, size, dividerHeight);
            context.fillStyle = '#ffffff';
            context.font = 'bold 24px sans-serif';
            context.textBaseline = 'middle';
            context.fillText('SOURCE ABOVE  ·  STYLE BELOW', 24, size / 2);
            return canvas.toDataURL('image/png').split(',')[1];
        });
    }
    function captionUploadedImage(imagePath, instruction, maxNew) {
        return fetch('/v1/caption', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_path: imagePath, prompt: instruction, max_new: maxNew || 160 })
        }).then(function (response) {
            return response.text().then(function (body) {
                var data = {};
                try {
                    data = JSON.parse(body);
                }
                catch (_) {
                    data = { error: body || ('HTTP ' + response.status) };
                }
                if (!response.ok)
                    throw new Error(data.error || data.detail || ('HTTP ' + response.status));
                if (!String(data.caption || '').trim())
                    throw new Error('The visual analyzer returned an empty description');
                return String(data.caption).trim();
            });
        });
    }
    function parseFlowEditCaptionPair(caption) {
        var text = String(caption || '').trim();
        var start = text.indexOf('{');
        var end = text.lastIndexOf('}');
        if (start >= 0 && end > start) {
            try {
                var parsed = JSON.parse(text.slice(start, end + 1));
                var source = String(parsed.source || parsed.source_description || '').trim();
                var target = String(parsed.target || parsed.target_description || '').trim();
                if (source && target)
                    return { sourceDescription: source, targetDescription: target };
            }
            catch (_) { }
        }
        throw new Error('The visual analyzer did not return the required source/target JSON');
    }
    function prepareFlowEditCaptionPair(imagePath, sourceDescription, editInstruction) {
        var source = String(sourceDescription || '').trim();
        var instruction = String(editInstruction || '').trim();
        if (source) {
            return captionUploadedImage(
                imagePath,
                'Write one complete standalone target-image description for FlowEdit. Use the visible image and this source description: ' +
                    JSON.stringify(source) + '. Apply this requested change: ' + JSON.stringify(instruction) +
                    '. Preserve every unchanged subject, identity, pose, composition, framing, clothing, lighting, and background detail. Do not mention an edit, request, source image, or instruction. Return only the target description, at most 110 words.',
                220
            ).then(function (targetDescription) {
                return { sourceDescription: source, targetDescription: targetDescription };
            });
        }
        return captionUploadedImage(
            imagePath,
            'Analyze the visible source image and apply this requested change conceptually: ' + JSON.stringify(instruction) +
                '. Return only valid compact JSON with exactly two string fields: {"source":"complete source image description","target":"complete target image description"}. ' +
                'The descriptions must be parallel in detail and ordering. The target must preserve every unchanged subject, identity, pose, composition, framing, clothing, lighting, and background detail while incorporating the requested change. ' +
                'Do not mention an edit, request, source image, or instruction. Keep each description at most 110 words.',
            320
        ).then(parseFlowEditCaptionPair);
    }
    function ideogramFlowEditCaption(description) {
        return JSON.stringify({
            aspect_ratio: '1:1',
            high_level_description: description,
            compositional_deconstruction: {
                background: 'Preserve the source composition and spatial arrangement',
                elements: []
            }
        });
    }
    function parseStylePairCaption(caption) {
        var text = String(caption || '').trim();
        var start = text.indexOf('{');
        var end = text.lastIndexOf('}');
        if (start >= 0 && end > start) {
            try {
                var parsed = JSON.parse(text.slice(start, end + 1));
                var source = String(parsed.source || '').trim();
                var style = String(parsed.style || '').trim();
                if (source && style)
                    return { sourceText: source, styleText: style };
            }
            catch (_) { }
        }
        throw new Error('The visual analyzer did not return the required source/style JSON');
    }
    function parseStyleOnlyCaption(caption) {
        var text = String(caption || '').trim();
        var start = text.indexOf('{');
        var end = text.lastIndexOf('}');
        if (start >= 0 && end > start) {
            try {
                var parsed = JSON.parse(text.slice(start, end + 1));
                var fields = ['medium', 'rendering', 'palette', 'line_work', 'lighting', 'texture', 'finish'];
                var values = fields.map(function (field) { return String(parsed[field] || '').trim(); }).filter(Boolean);
                if (values.length >= 4)
                    return values.join('; ');
            }
            catch (_) { }
        }
        throw new Error('The visual analyzer did not return the required style-only JSON');
    }
    function limitDescriptionWords(description, limit) {
        var words = String(description || '').trim().split(/\s+/).filter(Boolean);
        return words.slice(0, limit).join(' ').replace(/[.\s]+$/, '');
    }
    function buildStyleTargetCaption(sourceDescription, styleDescription, userInstruction) {
        var source = limitDescriptionWords(sourceDescription, STYLE_SOURCE_WORD_LIMIT);
        var style = limitDescriptionWords(styleDescription, STYLE_DESCRIPTION_WORD_LIMIT);
        var instruction = limitDescriptionWords(userInstruction, STYLE_USER_DIRECTION_WORD_LIMIT);
        var target = source + '. Render this exact scene in the following visual style: ' + style + '.';
        if (instruction)
            target += ' Additional target detail: ' + instruction + '.';
        target += ' Keep the same subject identities, number of people, poses, clothing, objects, positions, camera angle, crop, spatial composition, and setting.';
        return target;
    }
    function startStyleFlowEditGeneration() {
        if (!editSourceDataUrl) {
            showError('Select an image on the Canvas or load a source image');
            return;
        }
        if (!styleReferenceDataUrl) {
            showError('Choose the lower-left style reference image');
            return;
        }
        var seed = genState.seed === -1 ? Math.floor(Math.random() * 4294967296) : genState.seed;
        setCanvasGenerating(true);
        if (typeof CanvasStatusBar !== 'undefined')
            CanvasStatusBar.updateGenStatus('Preparing source and style');
        Promise.all([
            imageDataUrlAsPngBase64(editSourceDataUrl),
            imageDataUrlAsPngBase64(styleReferenceDataUrl),
            composeStyleAnalysisPngBase64(editSourceDataUrl, styleReferenceDataUrl)
        ]).then(function (images) {
            return Promise.all([
                SerenityAPI.uploadImageDetails(images[0], 'style_source'),
                SerenityAPI.uploadImageDetails(images[1], 'style_reference'),
                SerenityAPI.uploadImageDetails(images[2], 'style_analysis')
            ]);
        }).then(function (uploads) {
            var sourceUpload = uploads[0];
            var styleUpload = uploads[1];
            var analysisUpload = uploads[2];
            var sourceDescription = genState.editSourcePrompt.trim();
            if (typeof CanvasStatusBar !== 'undefined')
                CanvasStatusBar.updateGenStatus('Analyzing source and style');
            var analysis = sourceDescription ? captionUploadedImage(
                    styleUpload.path,
                    'Analyze only how this reference is rendered. Return only valid compact JSON with exactly these string fields: ' +
                        '{"medium":"art medium such as anime digital illustration","rendering":"rendering technique","palette":"color palette and contrast","line_work":"line quality","lighting":"lighting treatment","texture":"surface texture","finish":"final finish and effects"}. ' +
                        'Each value must be eight words or fewer. Never describe the subject, person, clothing, pose, shot type, framing, objects, setting, or composition.'
                ).then(parseStyleOnlyCaption).then(function (styleText) { return { sourceText: sourceDescription, styleText: styleText }; }) :
                captionUploadedImage(
                    analysisUpload.path,
                    'The image is a two-panel analysis sheet: SOURCE is above the divider and STYLE is below it. Return only valid compact JSON with exactly two string fields: {"source":"complete literal source description","style":"visual style only"}. ' +
                        'For source, precisely describe the main subject identity and appearance, exact body pose and direction of movement, full-body versus crop, every visible garment and shoe, carried objects, camera angle, framing, lighting, setting, and the number and relative placement of surrounding people and major background objects. ' +
                        'Keep source at most 80 words. For style, use at most 35 words and describe only the lower image medium, rendering technique, palette, contrast, lighting treatment, texture, line quality, and finish; never copy its subject, clothing, pose, objects, setting, or composition.',
                    320
                ).then(parseStylePairCaption);
            return analysis.then(function (captions) {
                captions.sourceText = limitDescriptionWords(captions.sourceText, STYLE_SOURCE_WORD_LIMIT);
                captions.styleText = limitDescriptionWords(captions.styleText, STYLE_DESCRIPTION_WORD_LIMIT);
                captions.sourceUpload = sourceUpload;
                captions.styleUpload = styleUpload;
                return captions;
            });
        }).then(function (analysis) {
            genState.editSourcePrompt = analysis.sourceText;
            els.editSourcePrompt.value = analysis.sourceText;
            var flowEngine = isKreaFlowEditEngine(genState.editEngine) ?
                upgradeKreaFlowEditEngineTo1024(genState.editEngine) : 'ideogram4';
            var ideogramStyle = flowEngine === 'ideogram4';
            var instruction = genState.prompt.trim();
            var targetText = buildStyleTargetCaption(analysis.sourceText, analysis.styleText, instruction);
            pendingCanvasMetadata = {
                prompt: targetText,
                sourcePrompt: analysis.sourceText,
                styleCaption: analysis.styleText,
                styleReferenceId: styleReferenceId,
                styleReferencePath: analysis.styleUpload.path,
                styleEntireImage: genState.styleEntireImage,
                model: ideogramStyle ? 'ideogram4' : (isKreaTurboFlowEditEngine(flowEngine) ? 'krea2_turbo' : 'krea2_raw'),
                arch: ideogramStyle ? 'ideogram4' : 'krea2',
                editMode: 'style',
                width: STYLE_RESULT_SIZE,
                height: STYLE_RESULT_SIZE,
                steps: genState.steps,
                cfg: genState.cfg,
                seed: seed,
                submittedAt: new Date().toISOString()
            };
            if (typeof CanvasStatusBar !== 'undefined')
                CanvasStatusBar.updateGenStatus('Queueing FlowEdit');
            queueWorkflow(WorkflowBuilder.buildFlowEdit({
                engine: flowEngine,
                initImageName: analysis.sourceUpload.path,
                sourcePrompt: ideogramStyle ? ideogramFlowEditCaption(analysis.sourceText) : analysis.sourceText,
                sourceNegative: genState.editSourceNegative,
                prompt: ideogramStyle ? ideogramFlowEditCaption(targetText) : targetText,
                negPrompt: genState.negative,
                steps: genState.steps,
                nmax: genState.editNmax,
                nmin: genState.editNmin,
                srcCfg: genState.editSourceCfg,
                cfg: genState.cfg,
                seed: seed,
                autoMask: genState.styleEntireImage ? false : genState.editAutoMask,
                maskQ: genState.editMaskQ,
                maskDilate: genState.editMaskDilate,
                maskWarmup: genState.editMaskWarmup
            }));
        }).catch(function (error) {
            showError('Style FlowEdit failed: ' + error.message);
            setCanvasGenerating(false);
        });
    }
    // Check if there's actual pixel content in the bbox region by sampling pixels
    function checkBboxContent() {
        return new Promise(function (resolve) {
            // A mask is not source content. Exclude it while checking so an
            // empty Canvas with only mask strokes cannot start an inpaint.
            var dataURL = captureBoundingBoxDataURL(0.1, true);
            // Check if the exported region has any non-transparent pixels
            var testImg = new Image();
            testImg.onload = function () {
                var c = document.createElement('canvas');
                c.width = testImg.width;
                c.height = testImg.height;
                var ctx = c.getContext('2d');
                ctx.drawImage(testImg, 0, 0);
                var data = ctx.getImageData(0, 0, c.width, c.height).data;
                var hasPixels = false;
                for (var i = 3; i < data.length; i += 16) {
                    if (data[i] > 0) {
                        hasPixels = true;
                        break;
                    }
                }
                resolve(hasPixels);
            };
            testImg.onerror = function () { resolve(false); };
            testImg.src = dataURL;
        });
    }
    function exportBoundingBoxRegion() {
        return new Promise(function (resolve) {
            // Source pixels and the painted mask are uploaded separately.
            var dataURL = captureBoundingBoxDataURL(1, true);
            resolve(dataURL.split(',')[1]);
        });
    }
    function uploadInitImage(base64Data) {
        return SerenityAPI.uploadImage(base64Data, 'canvas_init');
    }
    function collectControlLayers() {
        var controls = [];
        canvasLayers.forEach(function (l) {
            if (l.data.type === 'control' && l.data.visible) {
                var cd = l.data;
                if (!cd.refImageSrc)
                    return;
                controls.push({
                    imageName: cd.refImageName || null,
                    refImageSrc: cd.refImageSrc,
                    controlNetModel: cd.controlModel || undefined,
                    weight: cd.weight,
                    startStep: cd.beginStep,
                    endStep: cd.endStep,
                });
            }
        });
        return controls;
    }
    function collectIPALayers() {
        if (typeof CanvasRefImages === 'undefined')
            return [];
        // Style mode consumes the selected reference through the FlowEdit
        // target-conditioning analysis above. It must not also inject that
        // image as an unrelated IP-Adapter branch into the FlowEdit graph.
        if (genState.editMode === 'style')
            return [];
        return CanvasRefImages.getAll().filter(function (ref) { return !!ref.src; }).map(function (ref) {
            return {
                refImageSrc: ref.src,
                imageName: ref.imageName || '',
                weight: ref.weight,
                ipaModel: ref.model,
                method: ref.method,
                startStep: ref.stepRange && ref.stepRange[0],
                endStep: ref.stepRange && ref.stepRange[1]
            };
        });
    }
    function uploadLayerImages(layers) {
        var promises = layers.map(function (l) {
            if (l.imageName)
                return Promise.resolve(l);
            if (!l.refImageSrc)
                return Promise.resolve(l);
            var base64 = l.refImageSrc.split(',')[1];
            if (!base64)
                return Promise.resolve(l);
            return uploadInitImage(base64).then(function (name) {
                l.imageName = name;
                return l;
            });
        });
        return Promise.all(promises);
    }
    function queueWorkflow(workflow) {
        // Apply ControlNet and IP-Adapter nodes if layers exist
        var controlLayers = collectControlLayers();
        var ipaLayers = collectIPALayers();
        var prepare = Promise.resolve();
        if (controlLayers.length > 0 || ipaLayers.length > 0) {
            prepare = Promise.all([
                uploadLayerImages(controlLayers),
                uploadLayerImages(ipaLayers)
            ]).then(function (results) {
                var controls = results[0].filter(function (l) { return l.imageName; });
                var ipas = results[1].filter(function (l) { return l.imageName; });
                if (controls.length > 0) {
                    workflow = WorkflowBuilder.applyControlNetNodes(workflow, controls);
                }
                if (ipas.length > 0) {
                    workflow = WorkflowBuilder.applyIPAdapterNodes(workflow, ipas);
                }
            });
        }
        prepare.then(function () {
            return SerenityAPI.postPrompt(workflow, {
                prompt: genState.prompt,
                model: pendingCanvasMetadata && pendingCanvasMetadata.model || genState.model,
                width: pendingCanvasMetadata && pendingCanvasMetadata.width,
                height: pendingCanvasMetadata && pendingCanvasMetadata.height,
                seed: pendingCanvasMetadata && pendingCanvasMetadata.seed,
                steps: genState.steps,
                cfg: genState.cfg
            });
        })
            .then(function (result) {
            pendingCanvasPromptId = result && result.prompt_id ? String(result.prompt_id) : '';
            if (pendingCanvasMetadata)
                pendingCanvasMetadata.promptId = pendingCanvasPromptId;
            if (typeof CanvasStatusBar !== 'undefined')
                CanvasStatusBar.updateGenStatus('queued');
        })
            .catch(function (err) {
            showError('Failed to queue: ' + err.message);
            pendingCanvasPromptId = '';
            setCanvasGenerating(false);
        });
    }
    function setCanvasGenerating(v) {
        canvasGenerating = v;
        if (typeof CanvasStatusBar !== 'undefined') {
            CanvasStatusBar.updateGenStatus(v ? 'generating' : 'idle');
        }
        var isVideo = isVideoArch();
        els.generateBtn.disabled = false;
        if (v) {
            els.generateBtn.textContent = 'Cancel Generation';
        }
        else {
            if (genState.editMode === 'flowedit')
                els.generateBtn.textContent = 'Run FlowEdit';
            else if (genState.editMode === 'style')
                els.generateBtn.textContent = 'Apply reference style';
            else if (genState.editMode === 'inpaint')
                els.generateBtn.textContent = 'Inpaint masked area';
            else if (genState.editMode === 'dynaedit')
                els.generateBtn.textContent = 'DynaEdit runner unavailable';
            else
                els.generateBtn.textContent = isVideo ? 'Generate Video' : 'Generate';
        }
        els.generateBtn.classList.toggle('generating', v);
        if (v) {
            els.progress.classList.add('active');
            els.progressBar.style.width = '100%';
        }
        else {
            els.progress.classList.remove('active');
            els.progressBar.style.width = '0%';
            els.progressLabel.classList.remove('visible');
        }
    }
    function cancelCanvasGeneration() {
        if (!canvasGenerating)
            return;
        els.generateBtn.disabled = true;
        els.generateBtn.textContent = 'Cancelling...';
        if (typeof CanvasStatusBar !== 'undefined')
            CanvasStatusBar.updateGenStatus('Cancelling...');
        SerenityAPI.interrupt().catch(function (error) {
            showError('Cancel failed: ' + error.message);
            els.generateBtn.disabled = false;
            els.generateBtn.textContent = 'Cancel Generation';
        });
    }
    function placeResultOnCanvas(src) {
        editSourceFile = null;
        editSourceDataUrl = String(src || '');
        editSourceIsVideo = false;
        var img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = function () {
            var kImg = new Konva.Image({
                image: img,
                x: boundingBox.x(),
                y: boundingBox.y(),
                width: boundingBox.width(),
                height: boundingBox.height(),
                draggable: activeTool === 'select'
            });
            var layer = getActiveKonvaLayer();
            if (layer) {
                layer.add(kImg);
                layer.batchDraw();
                History.push();
            }
        };
        img.src = src;
    }
    function selectGalleryImageAsCanvasSource(src, metadata) {
        if (!src || !boundingBox)
            return;
        // Gallery images are first-class Canvas/style sources. Do not inherit a
        // stale 512px FlowEdit box: the style workspace and selected result are
        // always presented at the compiled 1024px square profile.
        prefer1024FlowEditForNewImage();
        editSourceFile = null;
        editSourceDataUrl = String(src);
        editSourceIsVideo = false;
        genState.editSourcePrompt = String(metadata && metadata.prompt || '').trim();
        if (els.editSourcePrompt)
            els.editSourcePrompt.value = genState.editSourcePrompt;
        if (els.sourceEmpty)
            els.sourceEmpty.style.display = 'none';
        if (els.sourceVideo) {
            els.sourceVideo.removeAttribute('src');
            els.sourceVideo.style.display = 'none';
        }
        if (els.sourcePreview) {
            els.sourcePreview.src = editSourceDataUrl;
            els.sourcePreview.style.display = 'block';
        }
        stage.batchDraw();
    }
    function loadCanvasGallery() {
        var grid = document.getElementById('cv-gallery-grid');
        if (!grid)
            return;
        renderCanvasGalleryBoardControls();
        fetch('/v1/jobs').then(function (response) {
            if (!response.ok)
                throw new Error('HTTP ' + response.status);
            return response.json();
        }).then(function (jobs) {
            grid.innerHTML = '';
            var completed = (Array.isArray(jobs) ? jobs : []).filter(function (job) {
                return job.state === 'done' && job.output_path;
            }).filter(function (job) {
                return galleryBoardFilter === 'all' || canvasGalleryBoardFor(job.id) === galleryBoardFilter;
            }).slice(-24).reverse();
            if (!completed.length) {
                grid.innerHTML = '<span class="cv-gallery-empty">No completed results</span>';
                return;
            }
            completed.forEach(function (job) {
                var relative = job.output_location && job.output_location.relative_path
                    ? job.output_location.relative_path
                    : String(job.output_path).split('/').pop();
                var parts = String(relative).split('/');
                var filename = parts.pop();
                var subfolder = parts.join('/');
                var isVideo = /\.(mp4|webm|mov|gif)$/i.test(filename);
                var src = SerenityAPI.viewUrl(filename, subfolder, 'output');
                var card = document.createElement('div');
                card.className = 'cv-gallery-item';
                var button = document.createElement('button');
                button.type = 'button';
                button.className = 'cv-gallery-media';
                button.title = (job.metadata && job.metadata.prompt) || job.model || job.id;
                if (isVideo) {
                    var video = document.createElement('video');
                    video.src = src;
                    video.muted = true;
                    video.preload = 'metadata';
                    button.appendChild(video);
                }
                else {
                    var image = document.createElement('img');
                    image.src = src;
                    image.loading = 'lazy';
                    image.alt = job.id || 'Canvas result';
                    button.appendChild(image);
                }
                var label = document.createElement('span');
                label.textContent = job.model || job.id;
                button.appendChild(label);
                button.addEventListener('click', function () {
                    var metadata = Object.assign({}, job.metadata && job.metadata.params || job.params || {}, {
                        promptId: job.id,
                        model: job.model,
                        serverJob: job
                    });
                    if (!isVideo)
                        selectGalleryImageAsCanvasSource(src, metadata);
                    CanvasStaging.activate([{ src: src, isVideo: isVideo, filename: filename, metadata: metadata }], getToolContext());
                    if (typeof CanvasStatusBar !== 'undefined')
                        CanvasStatusBar.updateGenStatus('staging');
                });
                var boardSelect = document.createElement('select');
                boardSelect.className = 'cv-gallery-board-assign';
                boardSelect.title = 'Assign result to a board';
                galleryBoards.forEach(function (board) {
                    var option = document.createElement('option');
                    option.value = board;
                    option.textContent = board;
                    boardSelect.appendChild(option);
                });
                boardSelect.value = canvasGalleryBoardFor(job.id);
                boardSelect.addEventListener('change', function () {
                    galleryBoardAssignments[String(job.id)] = boardSelect.value;
                    saveCanvasGalleryBoards();
                    if (galleryBoardFilter !== 'all')
                        loadCanvasGallery();
                });
                card.appendChild(button);
                card.appendChild(boardSelect);
                grid.appendChild(card);
            });
        }).catch(function (error) {
            grid.innerHTML = '';
            var message = document.createElement('span');
            message.className = 'cv-gallery-empty';
            message.textContent = 'Gallery unavailable: ' + error.message;
            grid.appendChild(message);
        });
    }
    function loadCanvasGalleryBoards() {
        try {
            var saved = JSON.parse(localStorage.getItem(GALLERY_BOARD_KEY) || '{}');
            if (Array.isArray(saved.boards))
                galleryBoards = saved.boards.map(String).filter(Boolean);
            if (!galleryBoards.includes('Uncategorized'))
                galleryBoards.unshift('Uncategorized');
            galleryBoardAssignments = saved.assignments && typeof saved.assignments === 'object' ? saved.assignments : {};
        }
        catch (_) {
            galleryBoards = ['Uncategorized'];
            galleryBoardAssignments = {};
        }
    }
    function saveCanvasGalleryBoards() {
        localStorage.setItem(GALLERY_BOARD_KEY, JSON.stringify({
            boards: galleryBoards,
            assignments: galleryBoardAssignments
        }));
    }
    function canvasGalleryBoardFor(jobId) {
        var board = galleryBoardAssignments[String(jobId)] || 'Uncategorized';
        return galleryBoards.includes(board) ? board : 'Uncategorized';
    }
    function renderCanvasGalleryBoardControls() {
        var select = document.getElementById('cv-gallery-board-filter');
        if (!select)
            return;
        select.innerHTML = '';
        var all = document.createElement('option');
        all.value = 'all';
        all.textContent = 'All boards';
        select.appendChild(all);
        galleryBoards.forEach(function (board) {
            var option = document.createElement('option');
            option.value = board;
            option.textContent = board;
            select.appendChild(option);
        });
        if (galleryBoardFilter !== 'all' && !galleryBoards.includes(galleryBoardFilter))
            galleryBoardFilter = 'all';
        select.value = galleryBoardFilter;
    }
    function createCanvasGalleryBoard() {
        var name = prompt('Board name:', 'Favorites');
        name = name ? name.trim() : '';
        if (!name || galleryBoards.includes(name))
            return;
        galleryBoards.push(name);
        galleryBoardFilter = name;
        saveCanvasGalleryBoards();
        renderCanvasGalleryBoardControls();
        loadCanvasGallery();
    }
    function isPendingCanvasEvent(data) {
        if (!canvasGenerating)
            return false;
        if (!data || !data.prompt_id || !pendingCanvasPromptId)
            return true;
        return String(data.prompt_id) === pendingCanvasPromptId;
    }
    function canvasResultMetadata(promptId) {
        var local = JSON.parse(JSON.stringify(pendingCanvasMetadata || {}));
        local.promptId = promptId || pendingCanvasPromptId || '';
        var url = /^video-/.test(local.promptId)
            ? '/out/' + encodeURIComponent(local.promptId) + '/result.json'
            : '/v1/job/' + encodeURIComponent(local.promptId) + '/result';
        if (!local.promptId)
            return Promise.resolve(local);
        return fetch(url).then(function (response) {
            if (!response.ok)
                throw new Error('HTTP ' + response.status);
            return response.json();
        }).then(function (server) {
            local.serverResult = server;
            var manifest = server.server_result || server.worker_result || server;
            ['width', 'height', 'frame_count', 'fps', 'steps', 'seed', 'executed_sampler', 'executed_scheduler', 'peak_vram_mib'].forEach(function (key) {
                if (manifest[key] != null)
                    local[key === 'frame_count' ? 'frames' : key] = manifest[key];
            });
            return local;
        }).catch(function () {
            return local;
        });
    }
    function showError(msg) {
        if (!els.errorBanner)
            return;
        els.errorBanner.textContent = msg;
        els.errorBanner.classList.add('visible');
        setTimeout(function () { els.errorBanner.classList.remove('visible'); }, 5000);
    }
    // ── Video overlay on canvas ──
    function placeVideoOverlayOnCanvas(src) {
        removeVideoOverlay();
        var container = document.getElementById('canvas-stage-container');
        if (!container || !boundingBox || !stage)
            return;
        var overlay = document.createElement('div');
        overlay.id = 'cv-video-overlay';
        overlay.innerHTML = '<video src="' + src + '" autoplay loop muted playsinline controls' +
            ' style="width:100%;height:100%;object-fit:cover;border-radius:4px;"></video>';
        container.appendChild(overlay);
        updateVideoOverlayPosition();
    }
    function updateVideoOverlayPosition() {
        var overlay = document.getElementById('cv-video-overlay');
        if (!overlay || !boundingBox || !stage)
            return;
        var scale = stage.scaleX();
        var stagePos = stage.position();
        var bx = boundingBox.x() * scale + stagePos.x;
        var by = boundingBox.y() * scale + stagePos.y;
        var bw = boundingBox.width() * scale;
        var bh = boundingBox.height() * scale;
        overlay.style.cssText = 'position:absolute;left:' + bx + 'px;top:' + by + 'px;' +
            'width:' + bw + 'px;height:' + bh + 'px;pointer-events:auto;overflow:hidden;z-index:50;border-radius:4px;';
    }
    function removeVideoOverlay() {
        var overlay = document.getElementById('cv-video-overlay');
        if (overlay)
            overlay.remove();
    }
    // ── WebSocket ──
    function connectWS() {
        SerenityWS.on('execution_start', function (data) {
            if (!isPendingCanvasEvent(data))
                return;
            if (typeof CanvasStatusBar !== 'undefined')
                CanvasStatusBar.updateGenStatus('starting');
        });
        SerenityWS.on('preview', function (data) {
            if (!canvasGenerating || !data || !data.blob)
                return;
            var url = URL.createObjectURL(data.blob);
            var body = document.getElementById('canvas-preview-body');
            var panel = document.getElementById('canvas-preview-panel');
            if (body && panel) {
                // Revoke old preview URL
                if (body._previewUrl)
                    URL.revokeObjectURL(body._previewUrl);
                body._previewUrl = url;
                body.innerHTML = '<img src="' + url + '" style="opacity:0.85">';
                panel.style.display = 'block';
            }
        });
        SerenityWS.on('executed', function (data) {
            if (!isPendingCanvasEvent(data))
                return;
            // Clean up live preview URL
            var body = document.getElementById('canvas-preview-body');
            if (body && body._previewUrl) {
                URL.revokeObjectURL(body._previewUrl);
                body._previewUrl = undefined;
            }
            if (!data || !data.output)
                return;
            var out = data.output.ui || data.output;
            var items = out.images;
            var isVideoFile = false;
            if (!items && out.videos) {
                items = out.videos;
                isVideoFile = true;
            }
            if (!items || !items.length)
                return;
            var promptId = data.prompt_id || pendingCanvasPromptId;
            canvasResultMetadata(promptId).then(function (metadata) {
                var results = items.map(function (file) {
                    var src = '/view?filename=' + encodeURIComponent(file.filename) +
                        '&subfolder=' + encodeURIComponent(file.subfolder || '') +
                        '&type=' + encodeURIComponent(file.type || 'output');
                    var itemIsVideo = isVideoFile || /\.(webp|mp4|gif|webm|mov)$/i.test(file.filename);
                    return {
                        src: src,
                        isVideo: itemIsVideo,
                        filename: file.filename,
                        metadata: JSON.parse(JSON.stringify(metadata))
                    };
                });
                hideCanvasPreview();
                if (typeof CanvasStaging !== 'undefined')
                    CanvasStaging.activate(results, getToolContext());
                else
                    showCanvasPreview(results[0].src, results[0].isVideo);
                setCanvasGenerating(false);
                if (typeof CanvasStatusBar !== 'undefined')
                    CanvasStatusBar.updateGenStatus('staging');
                pendingCanvasPromptId = '';
                loadCanvasGallery();
            });
        });
        SerenityWS.on('progress', function (data) {
            if (!isPendingCanvasEvent(data))
                return;
            var hasSteps = Number(data.max) > 0;
            var pct = hasSteps ? (data.value / data.max * 100).toFixed(0) : '4';
            els.progressBar.style.width = pct + '%';
            var phase = data.phase ? String(data.phase) : 'Generating';
            els.progressLabel.textContent = phase + (hasSteps ? ' · Step ' + data.value + ' / ' + data.max : '');
            els.progressLabel.classList.add('visible');
            if (typeof CanvasStatusBar !== 'undefined')
                CanvasStatusBar.updateGenStatus(phase);
        });
        SerenityWS.on('execution_error', function (data) {
            if (!isPendingCanvasEvent(data))
                return;
            showError((data && data.exception_message) || 'Generation failed');
            setCanvasGenerating(false);
            if (typeof CanvasStatusBar !== 'undefined')
                CanvasStatusBar.updateGenStatus('error');
            pendingCanvasPromptId = '';
        });
        SerenityWS.on('execution_interrupted', function (data) {
            if (!isPendingCanvasEvent(data))
                return;
            setCanvasGenerating(false);
            if (typeof CanvasStatusBar !== 'undefined')
                CanvasStatusBar.updateGenStatus('interrupted');
            pendingCanvasPromptId = '';
        });
        // Stagehand telemetry → VRAM display
        SerenityWS.on('stagehand_telemetry', function (data) {
            if (typeof CanvasStatusBar !== 'undefined' && data && typeof data.vram_used_mb === 'number') {
                CanvasStatusBar.updateVram(data.vram_used_mb);
            }
        });
    }
    // ── State Persistence ──
    function saveState() {
        if (!stage || !boundingBox)
            return;
        try {
            localStorage.setItem('sf-canvas-state', JSON.stringify({
                stageX: stage.x(), stageY: stage.y(), stageScale: stage.scaleX(),
                bboxX: boundingBox.x(), bboxY: boundingBox.y(),
                bboxW: boundingBox.width(), bboxH: boundingBox.height(),
                activeTool: activeTool, brushSize: brushSize
            }));
        }
        catch (e) { /* ignore */ }
    }
    function restoreState() {
        try {
            var saved = JSON.parse(localStorage.getItem('sf-canvas-state'));
            if (!saved)
                return;
            if (typeof saved.stageX === 'number')
                stage.position({ x: saved.stageX, y: saved.stageY });
            if (typeof saved.stageScale === 'number')
                stage.scale({ x: saved.stageScale, y: saved.stageScale });
            if (typeof saved.bboxX === 'number') {
                boundingBox.position({ x: saved.bboxX, y: saved.bboxY });
                boundingBox.width(clampDim(saved.bboxW || 1024));
                boundingBox.height(clampDim(saved.bboxH || 1024));
                updateHandles();
                updateSizeLabel();
            }
            if (saved.activeTool)
                setTool(saved.activeTool);
            if (typeof saved.brushSize === 'number') {
                brushSize = saved.brushSize;
                if (els.brushSizeInput)
                    els.brushSizeInput.value = String(brushSize);
                if (els.brushSizeVal)
                    els.brushSizeVal.textContent = String(brushSize);
            }
            stage.batchDraw();
        }
        catch (e) { /* ignore */ }
    }
    // ── Resize ──
    function resizeStage() {
        positionCanvasToolRail();
        if (!stage)
            return;
        var container = document.getElementById('canvas-stage-container');
        if (!container)
            return;
        var w = container.offsetWidth;
        var h = container.offsetHeight;
        if (w < 10 || h < 10)
            return;
        stage.width(w);
        stage.height(h);
        stage.batchDraw();
    }
    // ── Init ──
    function init() {
        if (initialized)
            return;
        initialized = true;
        buildUI();
        bindRightPanelEvents();
        updateEditWorkspace();
        bindAdjustmentPanel();
        bindTextPanel();
        loadModels();
        connectWS();
        setupPreviewPanel();
        document.addEventListener('sf-staging-regenerate', startGeneration);
        document.addEventListener('sf-style-reference-selected', function (event) {
            var ref = event.detail && event.detail.ref;
            enterStyleMode(ref);
        });
        loadCanvasGalleryBoards();
        loadCanvasGallery();
        // Register SAM tool if canvas-sam.ts loaded
        if (typeof CanvasSAM !== 'undefined') {
            CanvasTools.registerTool('sam', CanvasSAM.getTool());
        }
        checkCanvasSamAvailability();
        document.addEventListener('keydown', handleKeyDown);
        document.addEventListener('keyup', handleKeyUp);
        // Status bar
        var centerPanel = document.querySelector('.cv-center');
        if (centerPanel && typeof CanvasStatusBar !== 'undefined') {
            CanvasStatusBar.create(centerPanel);
        }
        // Ref images panel
        if (typeof CanvasRefImages !== 'undefined') {
            CanvasRefImages.showPanel();
        }
        // Defer Konva init to next frame so layout is computed
        requestAnimationFrame(function () {
            initKonva();
            restoreState();
            // Expose tool context for SAM toolbar (needs stage to be ready)
            window._samToolContext = getToolContext();
            // Check for image sent from Simple mode
            var pendingImage = localStorage.getItem('sf-send-to-canvas');
            if (pendingImage) {
                localStorage.removeItem('sf-send-to-canvas');
                try {
                    var data = JSON.parse(pendingImage);
                    if (data.src && !data.isVideo) {
                        var img = new Image();
                        img.crossOrigin = 'anonymous';
                        img.onload = function () {
                            var kImg = new Konva.Image({
                                image: img,
                                x: boundingBox.x(),
                                y: boundingBox.y(),
                                width: boundingBox.width(),
                                height: boundingBox.height(),
                                draggable: activeTool === 'select'
                            });
                            var layer = getActiveKonvaLayer();
                            if (layer) {
                                layer.add(kImg);
                                layer.batchDraw();
                            }
                        };
                        img.src = data.src;
                    }
                    if (data.prompt)
                        setPrompt(data.prompt);
                    if (data.model)
                        setModel(data.model);
                }
                catch (e) { }
            }
            resetView();
            updateCursor();
            History.push(); // Initial snapshot
            setInterval(saveState, 5000);
        });
    }
    function checkCanvasSamAvailability() {
        var button = document.querySelector('.cv-tool-btn[data-tool="sam"]');
        var reason = 'SAM object selection unavailable';
        fetch('/canvas/sam3/status', { cache: 'no-store' })
            .then(function (resp) {
            if (!resp.ok)
                throw new Error('HTTP ' + resp.status);
            return resp.json();
        })
        .then(function (status) {
            samAvailable = status && status.available === true;
            if (samAvailable && Number(status.api_port) > 0) {
                window.SERENITY_SAM3_API_BASE = window.location.protocol + '//' + window.location.hostname + ':' + Number(status.api_port);
            }
            if (status && status.reason)
                reason = String(status.reason);
        })
            .catch(function () { samAvailable = false; })
            .finally(function () {
            if (!button)
                return;
            button.disabled = !samAvailable;
            button.classList.toggle('capability-unavailable', !samAvailable);
            button.title = samAvailable ? 'Select Object / SAM (S)' : reason;
        });
    }
    // ── Public API for Simple Mode ──
    function loadImageFromURL(src) {
        return new Promise(function (resolve) {
            if (!konvaReady) {
                resolve();
                return;
            }
            var img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = function () {
                var kImg = new Konva.Image({
                    image: img,
                    x: boundingBox.x(),
                    y: boundingBox.y(),
                    width: boundingBox.width(),
                    height: boundingBox.height(),
                    draggable: activeTool === 'select'
                });
                var layer = getActiveKonvaLayer();
                if (layer) {
                    layer.add(kImg);
                    layer.batchDraw();
                }
                History.push();
                resolve();
            };
            img.onerror = function () { resolve(); };
            img.src = src;
        });
    }
    function setPrompt(text) {
        if (els.prompt) {
            els.prompt.value = text || '';
            genState.prompt = text || '';
        }
    }
    function setModel(modelName) {
        if (els.model && modelName) {
            els.model.value = modelName;
            genState.model = modelName;
            updateTopbarModel(modelName);
            updateCanvasUIForArch(ModelUtils.detectArchFromFilename(modelName));
        }
    }
    /** Expose compositor context for external callers (Compositor, CanvasZoom, etc.) */
    function getCompositorContext() {
        if (!stage || !boundingBox || !backgroundLayer || !uiLayer)
            return null;
        return {
            stage: stage,
            boundingBox: boundingBox,
            backgroundLayer: backgroundLayer,
            uiLayer: uiLayer,
            canvasLayers: canvasLayers,
            genState: genState,
            uploadImage: function (base64) { return uploadInitImage(base64); },
        };
    }
    return {
        init: init,
        resize: resizeStage,
        saveState: saveState,
        loadImageFromURL: loadImageFromURL,
        setPrompt: setPrompt,
        setModel: setModel,
        getCompositorContext: getCompositorContext,
        exportMaskAsBW: exportMaskAsBW,
        getToolContext: function () { return stage ? getToolContext() : null; },
        getCanvasLayers: function () { return canvasLayers; },
        getStage: function () { return stage; },
    };
})();
//# sourceMappingURL=canvas-tab.js.map
