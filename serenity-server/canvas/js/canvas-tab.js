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
    var pendingVideoPollToken = 0;
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
        ltx2AudioPolicy: 'none',
        ltx2FeatureId: 'standard',
        ltx2FeatureWeight: 1.0,
        ltx2PostUpscaler: 'none',
        ltx2PostUpscaleFactor: 2,
        ltx2Mode: 'distilled',
        ltx2ProfileKey: '',
        ltx2Quant: 'bf16',
        ltx2SourceStrength: 0.7,
        ltx2CameraMotion: 'none',
        ltx2RetakeStart: 0,
        ltx2RetakeDuration: 2,
        ltx2ExtendDirection: 'end',
        ltx2ExtendSeconds: 3,
        h3Mode: 't2va',
        h3ProfileKey: '',
        h3Quant: 'int8-fast',
        h3AttentionBackend: 'cudnn',
        editMode: 'create',
        editEngine: 'krea2_turbo_1024',
        editModelEngine: 'klein9b',
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
        lanpaintLambda: 8,
        lanpaintStepSize: 0.15,
        lanpaintBeta: 1,
        lanpaintFriction: 15,
        lanpaintPromptMode: 'Image First',
        lanpaintBlendOverlap: 9,
        lanpaintContextExpand: 0,
        lanpaintEarlyStop: 1,
        lanpaintInnerThreshold: 0,
        lanpaintInnerPatience: 1
    };
    var editSourceFile = null;
    var editSourceDataUrl = '';
    var editSourceIsVideo = false;
    var editSourceUploadedPath = '';
    var editSourceUploadPromise = null;
    var ltx2LastFrameFile = null;
    var ltx2LastFrameUploadedPath = '';
    var ltx2LastFrameUploadPromise = null;
    var h3LastFrameFile = null;
    var h3LastFrameUploadedPath = '';
    var h3LastFrameUploadPromise = null;
    var editSourceVideoProbe = null;
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
    // Native instruction-edit models use one deliberately small Canvas surface:
    // source image + instruction + engine. Each engine resolves to an installed
    // checkpoint and a production-admitted model-specific graph.
    var EDIT_MODEL_ENGINES = [
        {
            id: 'klein9b', label: 'Klein 9B', backend: 'flux2', arch: 'klein',
            modelIncludes: ['klein', '9b'], modelExcludes: ['fp8'],
            preferredModels: ['flux-2-klein-base-9b'],
            implemented: true,
            profile: { steps: 35, cfg: 3.5, denoise: 1.0, sampler: 'euler', scheduler: 'flux2' }
        },
        {
            id: 'klein4b', label: 'Klein 4B', backend: 'flux2', arch: 'klein',
            modelIncludes: ['klein', '4b'], modelExcludes: ['fp8'],
            preferredModels: ['flux-2-klein-base-4b'],
            implemented: true,
            profile: { steps: 35, cfg: 3.5, denoise: 1.0, sampler: 'euler', scheduler: 'flux2' }
        },
        {
            id: 'qwen_edit', label: 'Qwen Edit', backend: 'qwenimage', arch: 'qwen',
            modelIncludes: ['qwen', 'edit'], modelExcludes: ['fp8'],
            preferredModels: ['qwen_image_edit_2511_bf16'],
            implemented: false,
            blocked: 'image-aware Qwen Edit conditioning is not production-wired'
        }
    ];
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
    function editModelEngineDefinition(engine) {
        return EDIT_MODEL_ENGINES.find(function (candidate) { return candidate.id === engine; }) || null;
    }
    function editModelBackendProfile(definition) {
        if (!definition || !canvasCapabilities || !Array.isArray(canvasCapabilities.backends))
            return null;
        return canvasCapabilities.backends.find(function (profile) {
            return profile && profile.backend === definition.backend;
        }) || null;
    }
    function editModelBackendAdmitted(definition) {
        var profile = editModelBackendProfile(definition);
        var edit = profile && profile.features && profile.features.instruction_edit;
        return !!(profile && profile.production_status === 'admitted' && edit && edit.supported === true);
    }
    function editModelScore(definition, modelName) {
        var name = String(modelName || '').toLowerCase();
        if (ModelUtils.archForModel(modelName) !== definition.arch)
            return -1;
        if ((definition.modelIncludes || []).some(function (needle) { return name.indexOf(needle) < 0; }))
            return -1;
        if ((definition.modelExcludes || []).some(function (needle) { return name.indexOf(needle) >= 0; }))
            return -1;
        var preferred = (definition.preferredModels || []).indexOf(name);
        return preferred >= 0 ? 100 - preferred : 50;
    }
    function findCanvasModelForEditModelEngine(definition) {
        if (!definition || !els.model)
            return null;
        return Array.from(els.model.options || []).map(function (option) {
            return { option: option, score: editModelScore(definition, option.value) };
        }).filter(function (candidate) { return candidate.score >= 0; })
            .sort(function (a, b) { return b.score - a.score; })[0] || null;
    }
    function editModelUnavailableReason(definition) {
        var model = findCanvasModelForEditModelEngine(definition);
        var profile = editModelBackendProfile(definition);
        var edit = profile && profile.features && profile.features.instruction_edit;
        if (!model)
            return 'model not installed';
        if (definition.implemented !== true)
            return definition.blocked || 'runtime not wired';
        return (edit && edit.reason) || 'backend does not admit native editing';
    }
    function populateEditModelEngineOptions() {
        if (!els.editModelEngine)
            return [];
        var available = EDIT_MODEL_ENGINES.filter(function (definition) {
            return definition.implemented === true &&
                editModelBackendAdmitted(definition) &&
                !!findCanvasModelForEditModelEngine(definition);
        });
        els.editModelEngine.innerHTML = '';
        available.forEach(function (definition) {
            var option = document.createElement('option');
            option.value = definition.id;
            option.textContent = definition.label;
            els.editModelEngine.appendChild(option);
        });
        EDIT_MODEL_ENGINES.filter(function (definition) {
            return available.indexOf(definition) < 0;
        }).forEach(function (definition) {
            var reason = editModelUnavailableReason(definition);
            var option = document.createElement('option');
            option.value = definition.id;
            option.textContent = definition.label + ' — ' + reason;
            option.title = reason;
            option.disabled = true;
            els.editModelEngine.appendChild(option);
        });
        if (available.length === 0) {
            els.editModelEngine.disabled = true;
            return available;
        }
        els.editModelEngine.disabled = false;
        if (!available.some(function (definition) { return definition.id === genState.editModelEngine; }))
            genState.editModelEngine = available[0].id;
        els.editModelEngine.value = genState.editModelEngine;
        return available;
    }
    function applyEditModelEngineProfile(engine) {
        var definition = editModelEngineDefinition(engine);
        if (!definition || !definition.profile)
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
        return true;
    }
    function syncCanvasModelFromEditModelEngine() {
        var definition = editModelEngineDefinition(genState.editModelEngine);
        var match = findCanvasModelForEditModelEngine(definition);
        if (!definition || !match)
            return false;
        els.model.value = match.option.value;
        genState.model = match.option.value;
        genState.arch = definition.arch;
        updateTopbarModel(match.option.value);
        updateCanvasUIForArch(definition.arch);
        return true;
    }
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
        if (ModelUtils.archForModel(modelName) !== definition.arch)
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
                'Describe the requested change in Prompt. The vision captioner writes a complete positive target description before the Mojo LanPaint sampler runs; every visible LanPaint value is submitted unchanged.' :
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
            var arch = ModelUtils.archForModel(option.value);
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
        genState.arch = ModelUtils.archForModel(match.value);
        updateTopbarModel(match.value);
        return true;
    }
    function syncFlowEditEngineFromCanvasModel(modelName) {
        var arch = ModelUtils.archForModel(modelName);
        if (arch === 'ideogram4')
            genState.editEngine = 'ideogram4';
        else if (arch === 'krea2') {
            var turbo = String(modelName).toLowerCase().indexOf('turbo') >= 0;
            var size = genState.editMode === 'style' ? 1024 : flowEditEngineSize(genState.editEngine);
            genState.editEngine = 'krea2_' + (turbo ? 'turbo' : 'raw') + '_' + size;
        }
        else if (arch === 'wan') {
            genState.width = 1280;
            genState.height = 704;
            genState.frames = 121;
            genState.fps = 24;
            genState.steps = 50;
            genState.cfg = 5.0;
            genState.sampler = 'uni_pc';
            genState.scheduler = 'normal';
            setCanvasLtx2GeometryLocked(false);
            [els.framesInput, els.framesRange, els.fpsInput, els.fpsRange].forEach(function (control) {
                if (!control)
                    return;
                control.disabled = true;
                control.title = 'Wan2.2 TI2V-5B is admitted at 121 frames and 24 FPS';
            });
            els.framesInput.value = '121';
            els.framesRange.value = '121';
            els.fpsInput.value = '24';
            els.fpsRange.value = '24';
            els.steps.value = '50';
            els.stepsRange.value = '50';
            setCanvasSelectOptions(
                els.sampler,
                [{ value: 'uni_pc', label: 'UniPC (Wan creator)' }],
                'uni_pc',
                false
            );
            setCanvasSelectOptions(
                els.scheduler,
                [{ value: 'normal', label: 'Wan flow schedule' }],
                'normal',
                false
            );
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
    var canvasVideoStatus = null;
    var samAvailable = false;
    var GALLERY_BOARD_KEY = 'serenity-canvas-gallery-boards-v1';
    var CANVAS_PANEL_STATE_KEY = 'serenity-canvas-panel-offsets-v1';
    var galleryBoards = ['Uncategorized'];
    var galleryBoardAssignments = {};
    var galleryBoardFilter = 'all';
    var canvasPanelOffsets = { parameters: 0, entities: 0 };
    var canvasPanelHidden = { parameters: false, entities: false };
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
    function isVideoArch() { return genState.arch === 'ltxv' || genState.arch === 'minimax_h3' || genState.arch === 'wan' || genState.arch === 'bernini'; }
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
            '<img id="cv-source-video-fallback" alt="Source video poster">' +
            '<div id="cv-source-video-note" class="cv-source-video-note"></div>' +
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
        var parametersHandle = document.createElement('button');
        parametersHandle.id = 'cv-parameters-handle';
        parametersHandle.className = 'cv-panel-handle cv-parameters-handle';
        parametersHandle.type = 'button';
        parametersHandle.setAttribute('aria-label', 'Move or hide parameters panel');
        parametersHandle.textContent = '\u2039';
        layout.appendChild(parametersHandle);
        var entitiesHandle = document.createElement('button');
        entitiesHandle.id = 'cv-entities-handle';
        entitiesHandle.className = 'cv-panel-handle cv-entities-handle';
        entitiesHandle.type = 'button';
        entitiesHandle.setAttribute('aria-label', 'Move or hide layers and gallery panel');
        entitiesHandle.textContent = '\u203a';
        layout.appendChild(entitiesHandle);
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
            '<option value="i2v_ltx23">I2V - LTX 2.3</option>' +
            '<option value="retake_ltx23">Retake - LTX 2.3</option>' +
            '<option value="extend_ltx23">Extend - LTX 2.3</option>' +
            '<option value="edit_models">Edit Models</option>' +
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
            '<div id="cv-edit-model-section" class="cv-edit-settings" style="display:none">' +
            '<div class="cv-setting-row"><span class="cv-setting-label">Engine</span>' +
            '<select id="cv-edit-model-engine" class="cv-select">' +
            '<option disabled selected>Loading edit models...</option></select></div>' +
            '<div class="cv-helper-text">Load one source image, describe the change, and generate a 1024 × 1024 edit.</div>' +
            '</div>' +
            '<div id="cv-lanpaint-section" class="cv-edit-settings" style="display:none">' +
            '<div class="cv-setting-row"><span class="cv-setting-label">Engine</span>' +
            '<select id="cv-lanpaint-engine" class="cv-select">' +
            '<option disabled selected>Loading supported engines...</option></select></div>' +
            '<details id="cv-lanpaint-controls" class="cv-edit-advanced" open><summary>LanPaint controls</summary>' +
            '<div class="cv-edit-grid">' +
            '<label>Inner steps<input id="cv-lanpaint-inner-steps" type="number" min="0" max="64" value="5"></label>' +
            '<label>Lambda<input id="cv-lanpaint-lambda" type="number" min="0.01" max="100" step="0.1" value="8"></label>' +
            '<label>Step size<input id="cv-lanpaint-step-size" type="number" min="0.001" max="5" step="0.01" value="0.15"></label>' +
            '<label>Beta<input id="cv-lanpaint-beta" type="number" min="0.01" max="100" step="0.1" value="1"></label>' +
            '<label>Friction<input id="cv-lanpaint-friction" type="number" min="0.01" max="100" step="0.1" value="15"></label>' +
            '<label>Prompt mode<select id="cv-lanpaint-prompt-mode" class="cv-select"><option value="Image First">Image First</option><option value="Prompt First">Prompt First</option></select></label>' +
            '<label>Blend overlap<input id="cv-lanpaint-blend-overlap" type="number" min="1" max="51" step="2" value="9"></label>' +
            '<label>Context expand<input id="cv-lanpaint-context-expand" type="number" min="0" max="256" step="8" value="0"></label>' +
            '<label>Early stop<input id="cv-lanpaint-early-stop" type="number" min="0" max="64" value="1"></label>' +
            '<label>Inner threshold<input id="cv-lanpaint-inner-threshold" type="number" min="0" max="100" step="0.01" value="0"></label>' +
            '<label>Inner patience<input id="cv-lanpaint-inner-patience" type="number" min="0" max="64" value="1"></label>' +
            '</div></details>' +
            '<div id="cv-masked-edit-helper" class="cv-helper-text">Loading masked-edit capabilities...</div>' +
            '</div>' +
            '<div id="cv-edit-runtime-note" class="cv-capability-note" style="display:none"></div>' +
            '<label class="cv-setting-label" style="margin-bottom:2px">Prompt / requested change</label>' +
            '<textarea id="cv-prompt" class="cv-textarea" rows="3" placeholder="Describe the content or requested change..."></textarea>' +
            '<div id="cv-advanced-generation-settings">' +
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
            '<select id="cv-sampler" class="cv-select"><option value="">Loading model capabilities...</option></select></div>' +
            '<div class="cv-setting-row"><span class="cv-setting-label">Scheduler</span>' +
            '<select id="cv-scheduler" class="cv-select"><option value="">Loading model capabilities...</option></select></div>' +
            '<div class="cv-setting-row"><span class="cv-setting-label">Seed</span>' +
            '<input type="number" id="cv-seed" class="cv-number-input cv-seed-input" min="-1" max="4294967295" value="-1" title="-1 chooses a random seed">' +
            '<span id="cv-batch-label" class="cv-setting-label">Batch</span><input type="number" id="cv-batch" class="cv-number-input" min="1" max="8" value="1"></div>' +
            '<div id="cv-video-section" style="display:none">' +
            '<div class="cv-section-title" style="margin-top:8px">Video</div>' +
            '<div class="cv-setting-row">' +
            '<span class="cv-setting-label">Frames</span>' +
            '<input type="number" id="cv-frames" class="cv-number-input" min="9" max="481" step="8" value="97">' +
            '<input type="range" id="cv-frames-range" class="cv-range" min="9" max="481" step="8" value="97">' +
            '</div>' +
            '<div class="cv-setting-row">' +
            '<span class="cv-setting-label">FPS</span>' +
            '<input type="number" id="cv-fps" class="cv-number-input" min="8" max="60" value="24">' +
            '<input type="range" id="cv-fps-range" class="cv-range" min="8" max="60" value="24">' +
            '</div>' +
            '<div id="cv-duration-hint" class="cv-duration-hint"></div>' +
            '<div id="cv-ltx2-section" class="cv-ltx2-section" style="display:none">' +
            '<div class="cv-section-title" style="margin-top:8px">LTX2 Mojo request</div>' +
            '<label class="cv-setting-label" for="cv-ltx2-resolution">Native resolution</label>' +
            '<select id="cv-ltx2-resolution" class="cv-select"><option value="">Loading supported resolutions...</option></select>' +
            '<label class="cv-setting-label" for="cv-ltx2-duration">Seconds</label>' +
            '<input type="number" id="cv-ltx2-duration" class="cv-select" min="4.84" max="20" step="0.01" list="cv-ltx2-duration-options" placeholder="Enter seconds">' +
            '<datalist id="cv-ltx2-duration-options"></datalist>' +
            '<label class="cv-setting-label" for="cv-ltx2-quant">Quality</label>' +
            '<select id="cv-ltx2-quant" class="cv-select">' +
            '<option value="bf16">Highest quality · BF16</option>' +
            '<option value="fp8">Balanced · FP8</option>' +
            '<option value="int4">Fastest · INT4</option>' +
            '</select>' +
            '<select id="cv-ltx2-profile" hidden aria-hidden="true"><option value=""></option></select>' +
            '<div id="cv-ltx2-profile-note" class="cv-helper-text">Enter supported seconds; Canvas resolves the native frame count and FPS automatically.</div>' +
            '<div class="cv-setting-row"><span class="cv-setting-label">Guidance</span>' +
            '<select id="cv-ltx2-mode" class="cv-select">' +
            '<option value="distilled">Fast distilled (single pass)</option>' +
            '<option value="dev">Dev CFG-star (quality, 3 pass)</option>' +
            '</select></div>' +
            '<label class="cv-setting-label" for="cv-ltx2-camera-motion">Camera motion</label>' +
            '<select id="cv-ltx2-camera-motion" class="cv-select">' +
            '<option value="none">None</option>' +
            '<option value="static">Static / locked off</option>' +
            '<option value="focus_shift">Focus shift / rack focus</option>' +
            '<option value="dolly_in">Dolly in</option>' +
            '<option value="dolly_out">Dolly out</option>' +
            '<option value="dolly_left">Dolly left</option>' +
            '<option value="dolly_right">Dolly right</option>' +
            '<option value="jib_up">Jib up</option>' +
            '<option value="jib_down">Jib down</option>' +
            '</select>' +
            '<div id="cv-ltx2-last-frame-row" style="display:none">' +
            '<div class="cv-section-title">Last-frame keyframe</div>' +
            '<input id="cv-ltx2-last-frame-file" type="file" accept="image/*" hidden>' +
            '<label for="cv-ltx2-last-frame-file" class="cv-import-btn">Choose optional last frame</label>' +
            '<button id="cv-ltx2-last-frame-clear" class="cv-lora-clear" type="button" disabled>Clear</button>' +
            '<div id="cv-ltx2-last-frame-note" class="cv-helper-text">Optional. LTX 2.3 interpolates from the loaded source image to this clean final keyframe.</div>' +
            '</div>' +
            '<div id="cv-ltx2-source-strength-row" class="cv-setting-row">' +
            '<span class="cv-setting-label">Source preservation</span>' +
            '<input type="range" id="cv-ltx2-source-strength" class="cv-range" min="0" max="1" step="0.01" value="0.70">' +
            '<span id="cv-ltx2-source-strength-val" class="cv-setting-value">0.70</span></div>' +
            '<div id="cv-ltx2-v2v-presets" class="cv-ltx2-v2v-presets" style="display:none">' +
            '<button type="button" data-strength="0">Replace subject</button>' +
            '<button type="button" data-strength="0.3">Transform</button>' +
            '<button type="button" data-strength="0.7">Preserve</button></div>' +
            '<div id="cv-ltx2-source-strength-help" class="cv-helper-text">Higher values preserve more of the source. Lower values allow a larger transformation.</div>' +
            '<div id="cv-ltx2-video-edit-section" class="cv-edit-settings" style="display:none">' +
            '<div id="cv-ltx2-retake-controls" style="display:none">' +
            '<div class="cv-section-title">Retake window</div>' +
            '<div class="cv-edit-grid">' +
            '<label>Start (seconds)<input id="cv-ltx2-retake-start" type="number" min="0" step="0.01" value="0"></label>' +
            '<label>Duration (seconds)<input id="cv-ltx2-retake-duration" type="number" min="2" step="0.01" value="2"></label>' +
            '</div>' +
            '<div class="cv-helper-text">Only this time window is regenerated. Everything before and after it is frozen from the source clip.</div>' +
            '</div>' +
            '<div id="cv-ltx2-extend-controls" style="display:none">' +
            '<div class="cv-section-title">Extend clip</div>' +
            '<div class="cv-edit-grid">' +
            '<label>Direction<select id="cv-ltx2-extend-direction" class="cv-select"><option value="end">After source</option><option value="start">Before source</option></select></label>' +
            '<label>Add seconds<input id="cv-ltx2-extend-seconds" type="number" min="2" max="20" step="0.01" value="3"></label>' +
            '</div>' +
            '<div class="cv-helper-text">Canvas selects the smallest compiled native duration that can hold the source plus the requested extension and feathers 0.5 seconds across the seam.</div>' +
            '</div>' +
            '<div id="cv-ltx2-video-edit-note" class="cv-helper-text">Load a source video to inspect its native geometry and duration.</div>' +
            '</div>' +
            '<label class="cv-setting-label cv-path-label" for="cv-caps-positive">Conditioning</label>' +
            '<input id="cv-caps-positive" class="cv-path-input" type="text" placeholder="Server path to prompt-matched conditioning JSON or safetensors">' +
            '<label class="cv-setting-label cv-path-label" for="cv-caps-negative">Negative conditioning</label>' +
            '<input id="cv-caps-negative" class="cv-path-input" type="text" placeholder="Optional when the sidecar contains negative conditioning">' +
            '<label class="cv-setting-label cv-path-label" for="cv-noise-fixture">Noise fixture</label>' +
            '<input id="cv-noise-fixture" class="cv-path-input" type="text" placeholder="Optional server path; blank uses the authored seed">' +
            '<div class="cv-setting-row"><span class="cv-setting-label">Audio</span>' +
            '<select id="cv-ltx2-audio-policy" class="cv-select">' +
            '<option value="none">No audio</option>' +
            '<option value="generate">Generate audio</option>' +
            '<option value="preserve">Preserve V2V source audio</option>' +
            '</select></div>' +
            '<label class="cv-setting-label" for="cv-ltx2-feature">Feature workflow</label>' +
            '<select id="cv-ltx2-feature" class="cv-select"><option value="standard">Standard T2V / I2V / V2V</option></select>' +
            '<div class="cv-setting-row"><span class="cv-setting-label">Feature weight</span>' +
            '<input type="number" id="cv-ltx2-feature-weight" class="cv-number-input" min="-10" max="10" step="0.05" value="1.0"></div>' +
            '<div id="cv-ltx2-feature-note" class="cv-helper-text">Standard LTX2 generation without a dedicated feature adapter.</div>' +
            '<div class="cv-setting-row"><span class="cv-setting-label">Post-upscale</span>' +
            '<select id="cv-ltx2-post-upscaler" class="cv-select"><option value="none">None</option></select></div>' +
            '<div class="cv-setting-row"><span class="cv-setting-label">Upscale factor</span>' +
            '<select id="cv-ltx2-post-upscale-factor" class="cv-select"><option value="2">2\u00d7</option><option value="4">4\u00d7</option></select></div>' +
            '<div id="cv-ltx2-post-upscale-note" class="cv-helper-text">Runs after native LTX2 decode. Availability and speed are reported by the server.</div>' +
            '<button id="cv-load-ltx2-template" class="cv-import-btn" type="button">Load LTX Desktop creator profile</button>' +
            '<div class="cv-helper-text">The Mojo runner validates these exact values and rejects unsupported compiled profiles before model loading.</div>' +
            '</div>' +
            '<div id="cv-h3-section" class="cv-ltx2-section" style="display:none">' +
            '<div class="cv-section-title" style="margin-top:8px">MiniMax-H3 Mojo request</div>' +
            '<label class="cv-setting-label" for="cv-h3-mode">Feature</label>' +
            '<select id="cv-h3-mode" class="cv-select">' +
            '<option value="t2va">Text to video + audio</option>' +
            '<option value="i2va">First frame to video + audio</option>' +
            '<option value="l2va">Last frame to video + audio</option>' +
            '<option value="fl2va">First + last frame to video + audio</option>' +
            '<option value="ref2va">Reference image to video + audio</option>' +
            '</select>' +
            '<label class="cv-setting-label" for="cv-h3-resolution">Resolution</label>' +
            '<select id="cv-h3-resolution" class="cv-select"><option value="">Loading supported H3 resolutions...</option></select>' +
            '<label class="cv-setting-label" for="cv-h3-duration">Seconds</label>' +
            '<input type="number" id="cv-h3-duration" class="cv-select" min="0.01" step="0.01" list="cv-h3-duration-options" placeholder="Enter supported seconds">' +
            '<datalist id="cv-h3-duration-options"></datalist>' +
            '<div id="cv-h3-profile-note" class="cv-helper-text">One H3 runner applies resolution, seconds, frames, FPS, and precision at request time.</div>' +
            '<label class="cv-setting-label" for="cv-h3-quant">Quality / speed</label>' +
            '<select id="cv-h3-quant" class="cv-select">' +
            '<option value="int8-fast">INT8 Fast DiT · W8A8</option>' +
            '<option value="int8">INT8 Quality DiT · groupwise</option>' +
            '<option value="bf16">BF16 DiT · INT8 encoder</option>' +
            '</select>' +
            '<label class="cv-setting-label" for="cv-h3-attention">Attention</label>' +
            '<select id="cv-h3-attention" class="cv-select">' +
            '<option value="cudnn">cU-DNN · quality default</option>' +
            '<option value="sage-int8">Sage INT8 · experimental</option>' +
            '</select>' +
            '<div id="cv-h3-last-frame-row" style="display:none">' +
            '<div class="cv-section-title">Last-frame keyframe</div>' +
            '<input id="cv-h3-last-frame-file" type="file" accept="image/*" hidden>' +
            '<label for="cv-h3-last-frame-file" class="cv-import-btn">Choose last frame</label>' +
            '<button id="cv-h3-last-frame-clear" class="cv-lora-clear" type="button" disabled>Clear</button>' +
            '<div id="cv-h3-last-frame-note" class="cv-helper-text">Required for L2V and first+last-frame video.</div>' +
            '</div>' +
            '<div id="cv-h3-mode-note" class="cv-helper-text">T2V uses the compiled profile. I2V uses the Source as frame zero. Ref2VA uses it only for identity and style—the image is not inserted into the output.</div>' +
            '<div class="cv-helper-text">Conditioned modes are fixed at 768 × 768, 124 frames, 24 FPS and generate synchronized audio. GPU model execution only. Ref2VA has separate DiT caches and shares only the identical row-scaled encoder cache.</div>' +
            '</div>' +
            '</div>' +
            '</div>' +
            '<div id="cv-model-row"><label class="cv-setting-label" style="margin-top:4px">Model</label>' +
            '<select id="cv-model" class="cv-select"><option disabled selected>Loading models...</option></select></div>' +
            '<div id="cv-capability-note" class="cv-capability-note" style="display:none"></div>' +
            '<div id="cv-lora-section" class="cv-lora-section">' +
            '<div class="cv-lora-header"><span class="cv-section-title">LoRA loaders</span><div class="cv-lora-header-actions">' +
            '<span id="cv-lora-compat" class="cv-lora-compat"></span>' +
            '<button id="cv-lora-clear" class="cv-lora-clear" type="button" disabled>Clear all</button></div></div>' +
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
        els.layout = document.querySelector('#panel-canvas .cv-layout');
        els.parametersPanel = document.querySelector('#panel-canvas .cv-right');
        els.entitiesPanel = document.querySelector('#panel-canvas .cv-left');
        els.parametersHandle = document.getElementById('cv-parameters-handle');
        els.entitiesHandle = document.getElementById('cv-entities-handle');
        els.layerList = document.getElementById('cv-layer-list');
        els.brushSection = document.getElementById('cv-brush-section');
        els.brushSizeInput = document.getElementById('cv-brush-size');
        els.brushSizeVal = document.getElementById('cv-brush-size-val');
        els.brushColorInput = document.getElementById('cv-brush-color');
        els.prompt = document.getElementById('cv-prompt');
        els.editMode = document.getElementById('cv-edit-mode');
        els.editEngine = document.getElementById('cv-edit-engine');
        els.editModelSection = document.getElementById('cv-edit-model-section');
        els.editModelEngine = document.getElementById('cv-edit-model-engine');
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
        els.lanpaintContextExpand = document.getElementById('cv-lanpaint-context-expand');
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
        els.sourceVideoFallback = document.getElementById('cv-source-video-fallback');
        els.sourceVideoNote = document.getElementById('cv-source-video-note');
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
        els.h3Section = document.getElementById('cv-h3-section');
        els.h3Mode = document.getElementById('cv-h3-mode');
        els.h3Resolution = document.getElementById('cv-h3-resolution');
        els.h3Duration = document.getElementById('cv-h3-duration');
        els.h3DurationList = document.getElementById('cv-h3-duration-options');
        els.h3ProfileNote = document.getElementById('cv-h3-profile-note');
        els.h3Quant = document.getElementById('cv-h3-quant');
        els.h3Attention = document.getElementById('cv-h3-attention');
        els.h3LastFrameRow = document.getElementById('cv-h3-last-frame-row');
        els.h3LastFrameFile = document.getElementById('cv-h3-last-frame-file');
        els.h3LastFrameClear = document.getElementById('cv-h3-last-frame-clear');
        els.h3LastFrameNote = document.getElementById('cv-h3-last-frame-note');
        els.h3ModeNote = document.getElementById('cv-h3-mode-note');
        els.ltx2Resolution = document.getElementById('cv-ltx2-resolution');
        els.ltx2Duration = document.getElementById('cv-ltx2-duration');
        els.ltx2DurationList = document.getElementById('cv-ltx2-duration-options');
        els.ltx2Quant = document.getElementById('cv-ltx2-quant');
        els.ltx2Profile = document.getElementById('cv-ltx2-profile');
        els.ltx2ProfileNote = document.getElementById('cv-ltx2-profile-note');
        els.ltx2Mode = document.getElementById('cv-ltx2-mode');
        els.ltx2CameraMotion = document.getElementById('cv-ltx2-camera-motion');
        els.ltx2LastFrameRow = document.getElementById('cv-ltx2-last-frame-row');
        els.ltx2LastFrameFile = document.getElementById('cv-ltx2-last-frame-file');
        els.ltx2LastFrameClear = document.getElementById('cv-ltx2-last-frame-clear');
        els.ltx2LastFrameNote = document.getElementById('cv-ltx2-last-frame-note');
        els.ltx2SourceStrength = document.getElementById('cv-ltx2-source-strength');
        els.ltx2SourceStrengthVal = document.getElementById('cv-ltx2-source-strength-val');
        els.ltx2SourceStrengthHelp = document.getElementById('cv-ltx2-source-strength-help');
        els.ltx2V2vPresets = document.getElementById('cv-ltx2-v2v-presets');
        els.ltx2VideoEditSection = document.getElementById('cv-ltx2-video-edit-section');
        els.ltx2RetakeControls = document.getElementById('cv-ltx2-retake-controls');
        els.ltx2RetakeStart = document.getElementById('cv-ltx2-retake-start');
        els.ltx2RetakeDuration = document.getElementById('cv-ltx2-retake-duration');
        els.ltx2ExtendControls = document.getElementById('cv-ltx2-extend-controls');
        els.ltx2ExtendDirection = document.getElementById('cv-ltx2-extend-direction');
        els.ltx2ExtendSeconds = document.getElementById('cv-ltx2-extend-seconds');
        els.ltx2VideoEditNote = document.getElementById('cv-ltx2-video-edit-note');
        els.advancedGenerationSettings = document.getElementById('cv-advanced-generation-settings');
        els.capsPositive = document.getElementById('cv-caps-positive');
        els.capsNegative = document.getElementById('cv-caps-negative');
        els.noiseFixture = document.getElementById('cv-noise-fixture');
        els.ltx2AudioPolicy = document.getElementById('cv-ltx2-audio-policy');
        els.ltx2Feature = document.getElementById('cv-ltx2-feature');
        els.ltx2FeatureWeight = document.getElementById('cv-ltx2-feature-weight');
        els.ltx2FeatureNote = document.getElementById('cv-ltx2-feature-note');
        els.ltx2PostUpscaler = document.getElementById('cv-ltx2-post-upscaler');
        els.ltx2PostUpscaleFactor = document.getElementById('cv-ltx2-post-upscale-factor');
        els.ltx2PostUpscaleNote = document.getElementById('cv-ltx2-post-upscale-note');
        els.loadLtx2Template = document.getElementById('cv-load-ltx2-template');
        els.modelRow = document.getElementById('cv-model-row');
        els.model = document.getElementById('cv-model');
        els.capabilityNote = document.getElementById('cv-capability-note');
        els.loraSection = document.getElementById('cv-lora-section');
        els.loraPicker = document.getElementById('cv-lora-picker');
        els.loraAdd = document.getElementById('cv-lora-add');
        els.loraClear = document.getElementById('cv-lora-clear');
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
    function canvasPanelWidth(side) {
        var panel = side === 'parameters' ? els.parametersPanel : els.entitiesPanel;
        return panel ? Math.max(0, panel.offsetWidth) : 0;
    }
    function saveCanvasPanelOffsets() {
        try {
            localStorage.setItem(CANVAS_PANEL_STATE_KEY, JSON.stringify({
                parameters: canvasPanelOffsets.parameters,
                entities: canvasPanelOffsets.entities,
                parametersHidden: canvasPanelHidden.parameters,
                entitiesHidden: canvasPanelHidden.entities
            }));
        }
        catch (error) { /* local storage is optional */ }
    }
    function updateCanvasPanelHandle(side) {
        var hidden = canvasPanelHidden[side];
        var handle = side === 'parameters' ? els.parametersHandle : els.entitiesHandle;
        if (!handle)
            return;
        if (side === 'parameters')
            handle.textContent = hidden ? '\u203a' : '\u2039';
        else
            handle.textContent = hidden ? '\u2039' : '\u203a';
        var name = side === 'parameters' ? 'parameters' : 'layers and gallery';
        handle.title = 'Drag to move the ' + name + ' panel. Click to ' + (hidden ? 'show' : 'hide') + ' it.';
        handle.setAttribute('aria-expanded', hidden ? 'false' : 'true');
    }
    function applyCanvasPanelOffsets(save) {
        if (!els.layout)
            return;
        ['parameters', 'entities'].forEach(function (side) {
            var width = canvasPanelWidth(side);
            if (canvasPanelHidden[side])
                canvasPanelOffsets[side] = width;
            else
                canvasPanelOffsets[side] = Math.max(0, Math.min(width, Number(canvasPanelOffsets[side]) || 0));
            if (width > 0 && canvasPanelOffsets[side] >= width - 1)
                canvasPanelHidden[side] = true;
            els.layout.style.setProperty('--cv-' + side + '-offset', canvasPanelOffsets[side] + 'px');
            els.layout.classList.toggle('cv-' + side + '-hidden', canvasPanelHidden[side]);
            updateCanvasPanelHandle(side);
        });
        if (save)
            saveCanvasPanelOffsets();
        requestAnimationFrame(function () {
            resizeStage();
            positionCanvasToolRail();
        });
    }
    function restoreCanvasPanelOffsets() {
        try {
            var saved = JSON.parse(localStorage.getItem(CANVAS_PANEL_STATE_KEY) || '{}');
            if (Number.isFinite(Number(saved.parameters)))
                canvasPanelOffsets.parameters = Math.max(0, Number(saved.parameters));
            if (Number.isFinite(Number(saved.entities)))
                canvasPanelOffsets.entities = Math.max(0, Number(saved.entities));
            canvasPanelHidden.parameters = saved.parametersHidden === true;
            canvasPanelHidden.entities = saved.entitiesHidden === true;
        }
        catch (error) {
            canvasPanelOffsets = { parameters: 0, entities: 0 };
            canvasPanelHidden = { parameters: false, entities: false };
        }
        applyCanvasPanelOffsets(false);
    }
    function toggleCanvasPanel(side) {
        var width = canvasPanelWidth(side);
        var show = canvasPanelHidden[side] || canvasPanelOffsets[side] >= width - 1;
        canvasPanelHidden[side] = !show;
        canvasPanelOffsets[side] = show ? 0 : width;
        applyCanvasPanelOffsets(true);
    }
    function moveCanvasPanel(side, offset) {
        var width = canvasPanelWidth(side);
        canvasPanelOffsets[side] = Math.max(0, Math.min(width, offset));
        canvasPanelHidden[side] = width > 0 && canvasPanelOffsets[side] >= width - 1;
        applyCanvasPanelOffsets(false);
    }
    function bindCanvasPanelHandle(side, handle) {
        if (!handle)
            return;
        var drag = null;
        handle.addEventListener('pointerdown', function (event) {
            if (event.button !== 0)
                return;
            event.preventDefault();
            drag = {
                pointerId: event.pointerId,
                startX: event.clientX,
                startOffset: canvasPanelOffsets[side],
                moved: false
            };
            handle.setPointerCapture(event.pointerId);
            if (els.layout)
                els.layout.classList.add('cv-panel-dragging');
        });
        handle.addEventListener('pointermove', function (event) {
            if (!drag || drag.pointerId !== event.pointerId)
                return;
            var delta = event.clientX - drag.startX;
            if (Math.abs(delta) > 3)
                drag.moved = true;
            moveCanvasPanel(side, drag.startOffset + (side === 'parameters' ? -delta : delta));
        });
        function finishDrag(event) {
            if (!drag || drag.pointerId !== event.pointerId)
                return;
            var moved = drag.moved;
            drag = null;
            if (handle.hasPointerCapture(event.pointerId))
                handle.releasePointerCapture(event.pointerId);
            if (els.layout)
                els.layout.classList.remove('cv-panel-dragging');
            if (moved)
                applyCanvasPanelOffsets(true);
            else
                toggleCanvasPanel(side);
        }
        handle.addEventListener('pointerup', finishDrag);
        handle.addEventListener('pointercancel', finishDrag);
        handle.addEventListener('keydown', function (event) {
            var amount = event.shiftKey ? 64 : 24;
            if (event.code === 'Enter' || event.code === 'Space') {
                event.preventDefault();
                toggleCanvasPanel(side);
            }
            else if (event.code === 'Home') {
                event.preventDefault();
                canvasPanelHidden[side] = false;
                canvasPanelOffsets[side] = 0;
                applyCanvasPanelOffsets(true);
            }
            else if (event.code === 'End') {
                event.preventDefault();
                canvasPanelHidden[side] = true;
                canvasPanelOffsets[side] = canvasPanelWidth(side);
                applyCanvasPanelOffsets(true);
            }
            else if (event.code === 'ArrowLeft' || event.code === 'ArrowRight') {
                event.preventDefault();
                var direction = event.code === 'ArrowLeft' ? -1 : 1;
                var change = side === 'parameters' ? -direction * amount : direction * amount;
                moveCanvasPanel(side, canvasPanelOffsets[side] + change);
                applyCanvasPanelOffsets(true);
            }
        });
    }
    function bindCanvasPanelMotion() {
        bindCanvasPanelHandle('parameters', els.parametersHandle);
        bindCanvasPanelHandle('entities', els.entitiesHandle);
        window.addEventListener('resize', function () {
            applyCanvasPanelOffsets(false);
        });
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
            if (!file)
                return;
            if (file.type.startsWith('video/') && genState.arch === 'ltxv')
                loadEditSourceFile(file);
            else if (file.type.startsWith('image/') && genState.editMode === 'create')
                loadImageFile(file);
            else if (file.type.startsWith('image/') || file.type.startsWith('video/'))
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
    function setLtx2SourceStrength(value) {
        genState.ltx2SourceStrength = Math.max(0, Math.min(1, Number(value)));
        if (els.ltx2SourceStrength)
            els.ltx2SourceStrength.value = String(genState.ltx2SourceStrength);
        if (els.ltx2SourceStrengthVal)
            els.ltx2SourceStrengthVal.textContent = genState.ltx2SourceStrength.toFixed(2);
        updateLtx2SourceStrengthHelp();
    }
    function clearLtx2LastFrame() {
        ltx2LastFrameFile = null;
        ltx2LastFrameUploadedPath = '';
        ltx2LastFrameUploadPromise = null;
        if (els.ltx2LastFrameFile)
            els.ltx2LastFrameFile.value = '';
        if (els.ltx2LastFrameClear)
            els.ltx2LastFrameClear.disabled = true;
        if (els.ltx2LastFrameNote)
            els.ltx2LastFrameNote.textContent =
                'Optional. LTX 2.3 interpolates from the loaded source image to this clean final keyframe.';
    }
    function clearH3LastFrame() {
        h3LastFrameFile = null;
        h3LastFrameUploadedPath = '';
        h3LastFrameUploadPromise = null;
        if (els.h3LastFrameFile)
            els.h3LastFrameFile.value = '';
        if (els.h3LastFrameClear)
            els.h3LastFrameClear.disabled = true;
        if (els.h3LastFrameNote)
            els.h3LastFrameNote.textContent =
                'Required for L2V and first+last-frame video.';
    }
    function updateLtx2SourceStrengthHelp() {
        if (!els.ltx2SourceStrengthHelp)
            return;
        var temporalEditMode = genState.editMode === 'retake_ltx23' ||
            genState.editMode === 'extend_ltx23';
        if (els.ltx2V2vPresets) {
            els.ltx2V2vPresets.style.display =
                editSourceIsVideo && !temporalEditMode ? 'flex' : 'none';
            els.ltx2V2vPresets.querySelectorAll('button[data-strength]').forEach(function (button) {
                button.classList.toggle(
                    'active',
                    Math.abs(Number(button.dataset.strength) - genState.ltx2SourceStrength) < 0.001
                );
            });
        }
        if (temporalEditMode) {
            els.ltx2SourceStrengthHelp.textContent =
                'Retake and Extend use LTX Desktop’s binary temporal mask: selected tokens regenerate and every other source token stays frozen.';
        }
        else if (editSourceIsVideo) {
            if (genState.ltx2SourceStrength <= 0.05) {
                els.ltx2SourceStrengthHelp.textContent =
                    'Prompt takeover: replaces the source subject while using the clip as video guidance.';
            }
            else if (genState.ltx2SourceStrength <= 0.35) {
                els.ltx2SourceStrengthHelp.textContent =
                    'Strong transformation, but the original subject may still remain in a full-sequence edit.';
            }
            else {
                els.ltx2SourceStrengthHelp.textContent =
                    'Preserves the source subject and layout. Choose Replace subject for a total character change.';
            }
        }
        else {
            els.ltx2SourceStrengthHelp.textContent =
                'Higher values preserve more of the source. Lower values allow a larger transformation.';
        }
    }
    function clearCanvasVideoPreview() {
        editSourceUploadedPath = '';
        editSourceUploadPromise = null;
        editSourceVideoProbe = null;
        if (els.sourceVideoFallback) {
            els.sourceVideoFallback.removeAttribute('src');
            els.sourceVideoFallback.style.display = 'none';
        }
        if (els.sourceVideoNote) {
            els.sourceVideoNote.textContent = '';
            els.sourceVideoNote.style.display = 'none';
            els.sourceVideoNote.classList.remove('error');
        }
    }
    function clearCanvasEditSource() {
        clearCanvasVideoPreview();
        clearLtx2LastFrame();
        editSourceFile = null;
        editSourceDataUrl = '';
        editSourceIsVideo = false;
        if (els.sourceVideo) {
            els.sourceVideo.pause();
            els.sourceVideo.removeAttribute('src');
            els.sourceVideo.load();
            els.sourceVideo.style.display = 'none';
        }
        if (els.sourcePreview) {
            els.sourcePreview.removeAttribute('src');
            els.sourcePreview.style.display = 'none';
        }
        if (els.sourceEmpty)
            els.sourceEmpty.style.display = '';
        updateLtx2SourceStrengthHelp();
        updateLtx2VideoEditControls();
    }
    function setCanvasVideoNote(message, error) {
        if (!els.sourceVideoNote)
            return;
        els.sourceVideoNote.textContent = message || '';
        els.sourceVideoNote.style.display = message ? 'block' : 'none';
        els.sourceVideoNote.classList.toggle('error', error === true);
    }
    function showCanvasVideoFallback(file, uploadedPath) {
        if (!file || file !== editSourceFile || !uploadedPath)
            return;
        fetch('/video_edit/thumbnails', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                source_path: uploadedPath,
                height: 120,
                fps: 30
            })
        }).then(function (response) {
            if (!response.ok)
                throw new Error('thumbnail HTTP ' + response.status);
            return response.json();
        }).then(function (thumbnail) {
            if (file !== editSourceFile || !thumbnail.sprite_url)
                return;
            var sprite = new Image();
            sprite.onload = function () {
                if (file !== editSourceFile || !els.sourceVideo ||
                    els.sourceVideo.videoWidth > 0)
                    return;
                var tileWidth = Math.max(1, Number(thumbnail.thumb_width) || sprite.naturalWidth);
                var tileHeight = Math.max(1, Number(thumbnail.thumb_height) || sprite.naturalHeight);
                var tile = document.createElement('canvas');
                tile.width = tileWidth;
                tile.height = tileHeight;
                var tileContext = tile.getContext('2d');
                tileContext.drawImage(sprite, 0, 0, tileWidth, tileHeight, 0, 0, tileWidth, tileHeight);
                var pixels = tileContext.getImageData(0, 0, tileWidth, tileHeight).data;
                var minX = tileWidth;
                var maxX = -1;
                for (var y = 0; y < tileHeight; y++) {
                    for (var x = 0; x < tileWidth; x++) {
                        var index = (y * tileWidth + x) * 4;
                        if (pixels[index] > 8 || pixels[index + 1] > 8 || pixels[index + 2] > 8) {
                            minX = Math.min(minX, x);
                            maxX = Math.max(maxX, x);
                        }
                    }
                }
                var cropX = maxX >= minX ? Math.max(0, minX - 2) : 0;
                var cropWidth = maxX >= minX ? Math.min(tileWidth - cropX, maxX - cropX + 3) : tileWidth;
                var poster = document.createElement('canvas');
                poster.width = cropWidth;
                poster.height = tileHeight;
                poster.getContext('2d').drawImage(
                    tile, cropX, 0, cropWidth, tileHeight,
                    0, 0, cropWidth, tileHeight
                );
                els.sourceVideoFallback.src = poster.toDataURL('image/png');
                els.sourceVideoFallback.style.display = 'block';
                els.sourceVideo.style.display = 'none';
                setCanvasVideoNote(
                    'HEVC source loaded. Chrome cannot play this codec here, so Canvas is showing a poster; V2V uses the original clip.',
                    false
                );
            };
            sprite.src = thumbnail.sprite_url;
        }).catch(function (error) {
            if (file !== editSourceFile)
                return;
            setCanvasVideoNote('Source video loaded, but its browser preview failed: ' + error.message, true);
        });
    }
    function uploadCanvasVideoSource(file) {
        clearCanvasVideoPreview();
        setCanvasVideoNote('Uploading source video and preparing its Canvas preview…', false);
        editSourceUploadPromise = SerenityAPI.uploadMediaDetails(file).then(function (details) {
            if (file !== editSourceFile)
                throw new Error('Source video changed during upload');
            editSourceUploadedPath = String(details.path || details.name || '');
            if (!editSourceUploadedPath)
                throw new Error('Media upload did not return a worker-readable path');
            setCanvasVideoNote('Source video loaded · probing native geometry…', false);
            showCanvasVideoFallback(file, editSourceUploadedPath);
            return fetch('/v1/video/probe?path=' + encodeURIComponent(editSourceUploadedPath))
                .then(function (response) {
                    if (!response.ok)
                        return response.text().then(function (body) {
                            throw new Error(body || ('video probe HTTP ' + response.status));
                        });
                    return response.json();
                }).then(function (probe) {
                    if (file !== editSourceFile)
                        throw new Error('Source video changed during probe');
                    if (!probe || probe.muxing !== 'probe_ok')
                        throw new Error('The selected file is not a complete video clip');
                    editSourceVideoProbe = probe;
                    if (genState.editMode === 'retake_ltx23') {
                        genState.ltx2AudioPolicy = probe.has_audio ? 'preserve' : 'none';
                        genState.includeAudio = false;
                        if (els.ltx2AudioPolicy)
                            els.ltx2AudioPolicy.value = genState.ltx2AudioPolicy;
                    }
                    else if (genState.editMode === 'extend_ltx23') {
                        genState.ltx2AudioPolicy = 'generate';
                        genState.includeAudio = true;
                        if (els.ltx2AudioPolicy)
                            els.ltx2AudioPolicy.value = 'generate';
                    }
                    setCanvasVideoNote(
                        'Source ready · ' + probe.width + '×' + probe.height +
                        ' · ' + probe.frame_count + ' frames @ ' +
                        Number(probe.fps).toFixed(3).replace(/\.?0+$/, '') + ' FPS',
                        false
                    );
                    updateLtx2VideoEditControls(true);
                    return editSourceUploadedPath;
                });
        }).catch(function (error) {
            if (file === editSourceFile)
                setCanvasVideoNote('Source video upload failed: ' + error.message, true);
            throw error;
        });
        editSourceUploadPromise.catch(function () { /* surfaced in the source pane and on Generate */ });
        return editSourceUploadPromise;
    }
    function uploadCanvasImageSource(file) {
        clearCanvasVideoPreview();
        editSourceUploadPromise = SerenityAPI.uploadMediaDetails(file).then(function (details) {
            if (file !== editSourceFile)
                throw new Error('Source image changed during upload');
            editSourceUploadedPath = String(details.path || details.name || '');
            if (!editSourceUploadedPath)
                throw new Error('Image upload did not return a worker-readable path');
            return editSourceUploadedPath;
        });
        editSourceUploadPromise.catch(function () { /* surfaced when Generate is pressed */ });
        return editSourceUploadPromise;
    }
    function loadImageFile(file) {
        if (file && file.type.startsWith('video/')) {
            loadEditSourceFile(file);
            return;
        }
        if (!file || !file.type.startsWith('image/')) {
            showError('Choose an image file');
            return;
        }
        var reader = new FileReader();
        reader.onload = function (ev) {
            editSourceFile = file;
            uploadCanvasImageSource(file);
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
            if (genState.arch === 'ltxv')
                setLtx2SourceStrength(1.0);
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
        var i2vLtx23Mode = genState.editMode === 'i2v_ltx23';
        var ltx2TemporalEditMode = genState.editMode === 'retake_ltx23' ||
            genState.editMode === 'extend_ltx23';
        var ltx2VideoSource = isVideo && genState.arch === 'ltxv' &&
            (genState.editMode === 'create' || ltx2TemporalEditMode);
        if (isVideo && i2vLtx23Mode) {
            showError('I2V - LTX 2.3 requires an image source');
            return;
        }
        if (!isVideo && ltx2TemporalEditMode) {
            showError((genState.editMode === 'retake_ltx23' ? 'Retake' : 'Extend') +
                ' - LTX 2.3 requires a source video');
            return;
        }
        if (isVideo && genState.editMode !== 'dynaedit' && !ltx2VideoSource) {
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
                if (genState.arch === 'ltxv')
                    setLtx2SourceStrength(0);
                els.sourcePreview.removeAttribute('src');
                els.sourcePreview.style.display = 'none';
                els.sourceVideo.src = editSourceDataUrl;
                els.sourceVideo.style.display = 'block';
                uploadCanvasVideoSource(file);
            }
            else {
                uploadCanvasImageSource(file);
                if (genState.arch === 'ltxv')
                    setLtx2SourceStrength(1.0);
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
                clearCanvasVideoPreview();
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
        // LTX video profiles own their compiled output geometry. Importing an
        // I2V guide image must not replace it with FlowEdit's square profile.
        if (genState.arch === 'ltxv' || genState.arch === 'minimax_h3')
            return;
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
            return ModelUtils.archForModel(option.value) === arch;
        });
        if (!match)
            return false;
        els.model.value = match.value;
        genState.model = match.value;
        updateTopbarModel(match.value);
        updateCanvasUIForArch(arch);
        return true;
    }
    function selectCanvasLtx23Model() {
        if (!els.model)
            return false;
        var options = Array.from(els.model.options || []);
        var temporalEdit = genState.editMode === 'retake_ltx23' ||
            genState.editMode === 'extend_ltx23';
        if (temporalEdit) {
            // Desktop Retake/Extend use the complete BF16 checkpoint because
            // source video and source audio both pass through bundled VAE
            // encoders. The partial FP8 diffusion files cannot implement that
            // creator contract.
            genState.ltx2Quant = 'bf16';
            if (els.ltx2Quant)
                els.ltx2Quant.value = 'bf16';
        }
        var current = options.find(function (option) {
            if (option.value !== genState.model ||
                ModelUtils.archForModel(option.value) !== 'ltxv')
                return false;
            if (temporalEdit)
                return option.value === 'ltx-2.3-22b-distilled';
            // Keep an arbitrary user-selected LTX2 BF16 finetune. Only replace
            // the known partial FP8 official artifact when BF16 is selected.
            return !(genState.ltx2Quant === 'bf16' &&
                option.value === 'ltx-2.3-22b-dev-fp8');
        });
        var exactCheckpoint = temporalEdit
            ? 'ltx-2.3-22b-distilled'
            : (genState.ltx2Quant === 'bf16'
                ? 'ltx-2.3-22b-distilled'
                : 'ltx-2.3-22b-dev-fp8');
        var match = options.find(function (option) {
            return option.value === exactCheckpoint;
        });
        if (!temporalEdit && current)
            match = current;
        match = match || options.find(function (option) {
            return ModelUtils.archForModel(option.value) === 'ltxv' &&
                /ltx[-_. ]?2[.-]3/i.test(option.value);
        }) || options.find(function (option) {
            return ModelUtils.archForModel(option.value) === 'ltxv';
        });
        if (!match)
            return false;
        els.model.value = match.value;
        genState.model = match.value;
        var checkpoint = String(match.value).replace(/\.safetensors$/i, '');
        var officialDev = checkpoint === 'ltx-2.3-22b-dev-fp8' ||
            checkpoint === 'ltx-2.3-22b-dev-fp8-dequant-bf16';
        genState.ltx2Mode = officialDev || /distill/i.test(checkpoint)
            ? 'distilled' : 'dev';
        if (els.ltx2Mode)
            els.ltx2Mode.value = genState.ltx2Mode;
        updateTopbarModel(match.value);
        updateCanvasUIForArch('ltxv');
        return true;
    }
    function sourceMatchingLtx2Profiles() {
        if (!editSourceVideoProbe)
            return [];
        var width = Number(editSourceVideoProbe.width);
        var height = Number(editSourceVideoProbe.height);
        var fps = Number(editSourceVideoProbe.fps);
        return activeCanvasLtx2Profiles().filter(function (profile) {
            return Number(profile.width) === width &&
                Number(profile.height) === height &&
                Math.abs(Number(profile.fps) - fps) <= 0.01;
        }).sort(function (a, b) {
            return Number(a.frames) - Number(b.frames);
        });
    }
    function updateLtx2VideoEditControls(syncProfile) {
        if (!els.ltx2VideoEditSection)
            return;
        var retake = genState.editMode === 'retake_ltx23';
        var extend = genState.editMode === 'extend_ltx23';
        var active = retake || extend;
        els.ltx2VideoEditSection.style.display = active ? 'block' : 'none';
        els.ltx2RetakeControls.style.display = retake ? 'block' : 'none';
        els.ltx2ExtendControls.style.display = extend ? 'block' : 'none';
        if (!active)
            return;
        if (!editSourceVideoProbe) {
            els.ltx2VideoEditNote.textContent =
                'Load a source video to inspect its native geometry, FPS, frame count, and available compiled edit targets.';
            return;
        }
        var sourceFrames = Number(editSourceVideoProbe.frame_count);
        var sourceFps = Number(editSourceVideoProbe.fps);
        var sourceDuration = sourceFrames > 0 && sourceFps > 0
            ? (sourceFrames - 1) / sourceFps : Number(editSourceVideoProbe.duration);
        var profiles = sourceMatchingLtx2Profiles();
        if (retake) {
            var exact = profiles.find(function (profile) {
                return Number(profile.frames) === sourceFrames;
            });
            var maxDuration = Math.max(0, sourceDuration - genState.ltx2RetakeStart);
            els.ltx2RetakeStart.max = String(Math.max(0, sourceDuration - 2));
            els.ltx2RetakeDuration.max = String(maxDuration);
            if (genState.ltx2RetakeStart + genState.ltx2RetakeDuration > sourceDuration)
                genState.ltx2RetakeDuration = Math.max(2, sourceDuration - genState.ltx2RetakeStart);
            els.ltx2RetakeStart.value = String(Number(genState.ltx2RetakeStart.toFixed(2)));
            els.ltx2RetakeDuration.value = String(Number(genState.ltx2RetakeDuration.toFixed(2)));
            if (!exact) {
                els.ltx2VideoEditNote.textContent =
                    'No compiled LTX 2.3 Retake runner exactly matches this ' +
                    editSourceVideoProbe.width + '×' + editSourceVideoProbe.height +
                    ', ' + sourceFrames + '-frame clip at ' +
                    Number(sourceFps.toFixed(3)) + ' FPS.';
                return;
            }
            if (syncProfile)
                applyCanvasLtx2Profile(exact);
            els.ltx2VideoEditNote.textContent =
                'Source matches ' + exact.width + '×' + exact.height + ' · ' +
                exact.frames + ' frames @ ' + exact.fps +
                ' FPS. The selected window will regenerate; the rest stays frozen.';
            return;
        }
        var requested = Math.max(2, Number(genState.ltx2ExtendSeconds) || 2);
        var candidates = profiles.filter(function (profile) {
            return Number(profile.frames) > sourceFrames;
        });
        var target = candidates.find(function (profile) {
            return (Number(profile.frames) - sourceFrames) / sourceFps + 0.001 >= requested;
        });
        if (!target) {
            els.ltx2VideoEditNote.textContent = candidates.length
                ? 'The requested extension exceeds the largest compiled target for this native resolution.'
                : 'No longer compiled LTX 2.3 profile matches this source resolution and FPS.';
            return;
        }
        var actualSeconds = (Number(target.frames) - sourceFrames) / sourceFps;
        genState.ltx2ExtendSeconds = actualSeconds;
        els.ltx2ExtendSeconds.value = String(Number(actualSeconds.toFixed(2)));
        if (syncProfile)
            applyCanvasLtx2Profile(target);
        els.ltx2VideoEditNote.textContent =
            'Output target: ' + target.width + '×' + target.height + ' · ' +
            target.frames + ' frames @ ' + target.fps + ' FPS. Adds ' +
            actualSeconds.toFixed(2).replace(/\.?0+$/, '') +
            ' seconds ' + (genState.ltx2ExtendDirection === 'start' ? 'before' : 'after') +
            ' the source with a 0.5-second seam.';
    }
    function updateEditWorkspace() {
        var mode = genState.editMode;
        var i2vLtx23Mode = mode === 'i2v_ltx23';
        var retakeLtx23Mode = mode === 'retake_ltx23';
        var extendLtx23Mode = mode === 'extend_ltx23';
        var temporalLtx23Mode = retakeLtx23Mode || extendLtx23Mode;
        var editing = mode !== 'create';
        var flowEditing = mode === 'flowedit' || mode === 'style';
        var nativeModelEditing = mode === 'edit_models';
        var engineDrivenEditing = flowEditing || nativeModelEditing ||
            mode === 'inpaint' || i2vLtx23Mode || temporalLtx23Mode;
        var ltx2SourceMode = i2vLtx23Mode || temporalLtx23Mode ||
            (mode === 'create' && genState.arch === 'ltxv');
        var h3SourceMode = mode === 'create' && genState.arch === 'minimax_h3' &&
            (genState.h3Mode === 'i2va' || genState.h3Mode === 'fl2va' ||
                genState.h3Mode === 'ref2va');
        var maskedEditEngine = maskedEditEngineDefinition(genState.lanpaintEngine);
        var sourceStrengthRow = document.getElementById('cv-ltx2-source-strength-row');
        if (els.ltx2LastFrameRow)
            els.ltx2LastFrameRow.style.display = i2vLtx23Mode ? 'block' : 'none';
        if (sourceStrengthRow)
            sourceStrengthRow.style.display = temporalLtx23Mode ? 'none' : 'flex';
        if (els.ltx2V2vPresets && temporalLtx23Mode)
            els.ltx2V2vPresets.style.display = 'none';
        if (els.ltx2SourceStrengthHelp)
            els.ltx2SourceStrengthHelp.style.display =
                temporalLtx23Mode ? 'none' : 'block';
        if (!temporalLtx23Mode)
            updateLtx2SourceStrengthHelp();
        els.editWorkspace.classList.toggle('edit-active', editing || ltx2SourceMode || h3SourceMode);
        els.editWorkspace.classList.toggle('style-active', mode === 'style');
        els.sourcePane.style.display = (editing || ltx2SourceMode || h3SourceMode) ? 'flex' : 'none';
        els.flowEditSection.style.display = (mode === 'flowedit' || mode === 'style') ? 'block' : 'none';
        els.editModelSection.style.display = nativeModelEditing ? 'block' : 'none';
        els.lanpaintSection.style.display = mode === 'inpaint' ? 'block' : 'none';
        if (els.advancedGenerationSettings)
            els.advancedGenerationSettings.style.display = nativeModelEditing ? 'none' : 'block';
        if (els.modelRow)
            els.modelRow.style.display = engineDrivenEditing ? 'none' : 'block';
        if (els.loraSection)
            els.loraSection.style.display = nativeModelEditing ? 'none' : '';
        els.denoise.disabled = mode === 'inpaint' && (!maskedEditEngine || maskedEditEngine.lanpaint);
        els.sampler.disabled = mode === 'inpaint';
        els.scheduler.disabled = mode === 'inpaint';
        els.editEngine.disabled = false;
        Array.from(els.editEngine.querySelectorAll('option[value$="_512"]')).forEach(function (option) {
            option.disabled = mode === 'style';
        });
        els.editRuntimeNote.style.display = 'none';
        var resultPaneTitle = 'Canvas';
        if (i2vLtx23Mode)
            resultPaneTitle = 'I2V result';
        else if (retakeLtx23Mode)
            resultPaneTitle = 'Retake result';
        else if (extendLtx23Mode)
            resultPaneTitle = 'Extended result';
        else if (mode === 'inpaint')
            resultPaneTitle = 'Result + mask';
        else if (mode === 'style' || nativeModelEditing)
            resultPaneTitle = 'Result · 1024 × 1024';
        else if (editing || ltx2SourceMode)
            resultPaneTitle = 'Result';
        els.resultPaneTitle.textContent = resultPaneTitle;
        els.importBtn.textContent = i2vLtx23Mode ? 'Load source image' :
            (temporalLtx23Mode ? 'Load source video' :
            (ltx2SourceMode ? 'Load source image / video' :
            (h3SourceMode ? (genState.h3Mode === 'ref2va'
                ? 'Load reference image' : 'Load source image') :
                (editing ? (mode === 'dynaedit' ? 'Load source video' : 'Load source image') : 'Import Image'))));
        els.importFile.accept = i2vLtx23Mode ? 'image/*' :
            (temporalLtx23Mode ? 'video/*' :
            (ltx2SourceMode ? 'image/*,video/*' :
                (mode === 'dynaedit' ? 'video/*' : 'image/*')));
        if (els.sourceEmpty && i2vLtx23Mode)
            els.sourceEmpty.innerHTML = 'Drop the first-frame image here<br><small>LTX 2.3 will animate it</small>';
        else if (els.sourceEmpty && temporalLtx23Mode)
            els.sourceEmpty.innerHTML = 'Drop the source video here<br><small>LTX 2.3 will preserve the unedited source region</small>';
        else if (els.sourceEmpty && ltx2SourceMode)
            els.sourceEmpty.innerHTML = 'Drop an image or video here<br><small>or choose it from the file manager</small>';
        else if (els.sourceEmpty && h3SourceMode && genState.h3Mode === 'ref2va')
            els.sourceEmpty.innerHTML = 'Drop the identity / style reference here<br><small>it will not become frame zero</small>';
        else if (els.sourceEmpty && h3SourceMode)
            els.sourceEmpty.innerHTML = 'Drop the first-frame image here<br><small>MiniMax-H3 will animate it</small>';
        else if (els.sourceEmpty)
            els.sourceEmpty.innerHTML = 'Drop an image here<br><small>or choose it from the file manager</small>';
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
        else if (nativeModelEditing) {
            populateEditModelEngineOptions();
            applyEditModelEngineProfile(genState.editModelEngine);
            if (boundingBox) {
                boundingBox.width(1024);
                boundingBox.height(1024);
                updateHandles();
                updateSizeLabel();
                updateBboxInputs();
                stage.batchDraw();
            }
            els.generateBtn.textContent = 'Generate Edit';
            if (!syncCanvasModelFromEditModelEngine()) {
                els.editRuntimeNote.textContent = 'No production-admitted installed model matches the selected edit engine.';
            }
            else {
                var editDefinition = editModelEngineDefinition(genState.editModelEngine);
                els.editRuntimeNote.textContent = editDefinition.label +
                    ' uses its native image-reference edit path at 1024 × 1024.';
            }
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
        else if (i2vLtx23Mode) {
            els.generateBtn.textContent = 'Generate I2V';
            els.editRuntimeNote.textContent =
                'LTX 2.3 encodes this image as frame-zero guidance at both native stages, then generates the video with the selected compiled profile.';
            els.editRuntimeNote.style.display = 'block';
        }
        else if (retakeLtx23Mode) {
            els.generateBtn.textContent = 'Retake selected window';
            els.editRuntimeNote.textContent =
                'LTX 2.3 encodes the complete source clip, regenerates only the authored time window with a binary temporal mask, and freezes every video token outside it.';
            els.editRuntimeNote.style.display = 'block';
        }
        else if (extendLtx23Mode) {
            els.generateBtn.textContent = 'Extend video';
            els.editRuntimeNote.textContent =
                'LTX 2.3 pads the source at the selected edge, generates only the new duration, and feathers 0.5 seconds into the kept source at the seam.';
            els.editRuntimeNote.style.display = 'block';
        }
        else {
            els.generateBtn.textContent = isVideoArch() ? 'Generate Video' : 'Generate';
        }
        if (canvasCapabilities)
            updateCanvasCapabilityUI();
        updateLtx2VideoEditControls(false);
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
            var resizeInteractive = bboxInteractive &&
                !(genState.arch === 'ltxv' && activeCanvasLtx2Profiles().length > 0);
            boundingBox.draggable(bboxInteractive);
            boundingBox.listening(bboxInteractive);
            resizeHandles.forEach(function (h) {
                h.draggable(resizeInteractive);
                h.listening(resizeInteractive);
                h.opacity(resizeInteractive ? 1 : 0.45);
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
            toggleCanvasPanel('entities');
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
                ltx2AudioPolicy: genState.ltx2AudioPolicy,
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
                noiseFixture: genState.noiseFixture, includeAudio: genState.includeAudio,
                ltx2AudioPolicy: genState.ltx2AudioPolicy
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
            else if (nextMode === 'edit_models') {
                populateEditModelEngineOptions();
                applyEditModelEngineProfile(genState.editModelEngine);
                syncCanvasModelFromEditModelEngine();
            }
            else if (nextMode === 'i2v_ltx23' ||
                nextMode === 'retake_ltx23' ||
                nextMode === 'extend_ltx23') {
                setLtx2SourceStrength(nextMode === 'i2v_ltx23' ? 1.0 : 0.0);
                if (!selectCanvasLtx23Model())
                    showError('No installed LTX 2.3 model is available for ' +
                        (nextMode === 'i2v_ltx23' ? 'I2V' :
                            (nextMode === 'retake_ltx23' ? 'Retake' : 'Extend')));
                updateLtx2VideoEditControls(true);
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
        els.editModelEngine.addEventListener('change', function () {
            genState.editModelEngine = this.value;
            if (!applyEditModelEngineProfile(this.value) || !syncCanvasModelFromEditModelEngine()) {
                showError('The selected edit model is not available on this machine');
                return;
            }
            updateEditWorkspace();
            resetView();
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
        els.lanpaintContextExpand.addEventListener('input', function () { genState.lanpaintContextExpand = Math.max(0, Math.floor(Number(this.value))); });
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
        els.ltx2AudioPolicy.addEventListener('change', function () {
            genState.ltx2AudioPolicy = this.value;
            genState.includeAudio = this.value === 'generate';
        });
        els.ltx2Feature.addEventListener('change', function () {
            applyCanvasLtx2Feature(this.value);
        });
        els.ltx2FeatureWeight.addEventListener('input', function () {
            var value = Number(this.value);
            if (Number.isFinite(value))
                genState.ltx2FeatureWeight = value;
        });
        els.ltx2PostUpscaler.addEventListener('change', function () {
            applyCanvasLtx2PostUpscaler(this.value);
        });
        els.ltx2PostUpscaleFactor.addEventListener('change', function () {
            genState.ltx2PostUpscaleFactor = Number(this.value);
        });
        els.ltx2Resolution.addEventListener('change', function () {
            applyCanvasLtx2ResolutionDuration(
                this.value, Number(els.ltx2Duration.value), true
            );
        });
        els.ltx2Duration.addEventListener('change', function () {
            if (applyCanvasLtx2ResolutionDuration(
                els.ltx2Resolution.value, Number(this.value), false
            ))
                return;
            var supported = canvasLtx2ProfilesForResolution(
                els.ltx2Resolution.value
            ).map(function (profile) {
                return Number(canvasLtx2ProfileDuration(profile).toFixed(2));
            });
            showError(
                'This native resolution supports ' +
                supported.join(', ') + ' seconds.'
            );
            applyCanvasLtx2Profile(preferredCanvasLtx2Profile());
        });
        els.ltx2Quant.addEventListener('change', function () {
            genState.ltx2Quant = this.value;
            selectCanvasLtx23Model();
        });
        els.ltx2Profile.addEventListener('change', function () {
            var key = this.value;
            var profile = activeCanvasLtx2Profiles().find(function (candidate) {
                return canvasLtx2ProfileKey(candidate) === key;
            });
            applyCanvasLtx2Profile(profile);
        });
        els.ltx2Mode.addEventListener('change', function () {
            genState.ltx2Mode = this.value;
            applyCanvasLtx2GuidanceMode();
        });
        els.ltx2CameraMotion.addEventListener('change', function () {
            genState.ltx2CameraMotion = this.value || 'none';
        });
        els.ltx2LastFrameFile.addEventListener('change', function () {
            var file = this.files && this.files[0];
            if (!file || !file.type.startsWith('image/')) {
                clearLtx2LastFrame();
                if (file)
                    showError('Choose an image file for the final keyframe');
                return;
            }
            ltx2LastFrameFile = file;
            ltx2LastFrameUploadedPath = '';
            if (els.ltx2LastFrameClear)
                els.ltx2LastFrameClear.disabled = false;
            if (els.ltx2LastFrameNote)
                els.ltx2LastFrameNote.textContent =
                    'Uploading final keyframe · ' + file.name;
            ltx2LastFrameUploadPromise = SerenityAPI.uploadMedia(file).then(function (path) {
                if (file !== ltx2LastFrameFile)
                    throw new Error('Last-frame image changed during upload');
                ltx2LastFrameUploadedPath = String(path || '');
                if (!ltx2LastFrameUploadedPath)
                    throw new Error('Media upload did not return a worker-readable path');
                if (els.ltx2LastFrameNote)
                    els.ltx2LastFrameNote.textContent =
                        'Final keyframe ready · ' + file.name;
                return ltx2LastFrameUploadedPath;
            }).catch(function (error) {
                if (file === ltx2LastFrameFile && els.ltx2LastFrameNote)
                    els.ltx2LastFrameNote.textContent =
                        'Final keyframe upload failed · ' + error.message;
                throw error;
            });
            ltx2LastFrameUploadPromise.catch(function () {
                /* surfaced beside the picker and again on Generate */
            });
        });
        els.ltx2LastFrameClear.addEventListener('click', clearLtx2LastFrame);
        els.h3Mode.addEventListener('change', function () {
            genState.h3Mode = this.value;
            refreshCanvasH3Controls();
            updateEditWorkspace();
            els.generateBtn.textContent = genState.h3Mode === 't2va'
                ? 'Generate H3 Video + Audio'
                : 'Generate H3 ' + genState.h3Mode.toUpperCase() + ' + Audio';
        });
        els.h3Resolution.addEventListener('change', function () {
            var requestedResolution = this.value;
            var requestedDuration = Number(els.h3Duration.value);
            if (applyCanvasH3ResolutionDuration(
                requestedResolution, requestedDuration
            ))
                return;
            showError(
                'The H3 runner did not report ' + requestedResolution +
                ' at ' + requestedDuration + ' seconds.'
            );
            applyCanvasH3Profile(preferredCanvasH3Profile());
        });
        els.h3Duration.addEventListener('change', function () {
            if (genState.h3Mode !== 't2va')
                return;
            var requestedDuration = Number(this.value);
            if (applyCanvasH3ResolutionDuration(
                els.h3Resolution.value, requestedDuration
            ))
                return;
            showError(
                'The H3 runner did not report ' + els.h3Resolution.value +
                ' at ' + requestedDuration + ' seconds.'
            );
            applyCanvasH3Profile(preferredCanvasH3Profile());
        });
        els.h3Quant.addEventListener('change', function () {
            genState.h3Quant = this.value;
            if (this.value === 'int8-fast')
                genState.h3AttentionBackend = 'cudnn';
            refreshCanvasH3Controls();
        });
        els.h3Attention.addEventListener('change', function () {
            genState.h3AttentionBackend = this.value === 'sage-int8'
                ? 'sage-int8' : 'cudnn';
        });
        els.h3LastFrameFile.addEventListener('change', function () {
            var file = this.files && this.files[0];
            if (!file || !file.type.startsWith('image/')) {
                clearH3LastFrame();
                if (file)
                    showError('Choose an image file for the MiniMax-H3 final keyframe');
                return;
            }
            h3LastFrameFile = file;
            h3LastFrameUploadedPath = '';
            els.h3LastFrameClear.disabled = false;
            els.h3LastFrameNote.textContent = 'Uploading H3 final keyframe · ' + file.name;
            h3LastFrameUploadPromise = imageFileAsDataUrl(file)
                .then(h3SquareKeyframePngBase64)
                .then(uploadInitImage)
                .then(function (path) {
                if (file !== h3LastFrameFile)
                    throw new Error('H3 last-frame image changed during upload');
                h3LastFrameUploadedPath = String(path || '');
                if (!h3LastFrameUploadedPath)
                    throw new Error('Media upload did not return a worker-readable path');
                els.h3LastFrameNote.textContent =
                    'H3 final keyframe ready · 768 × 768 center crop · ' + file.name;
                return h3LastFrameUploadedPath;
            }).catch(function (error) {
                if (file === h3LastFrameFile)
                    els.h3LastFrameNote.textContent =
                        'H3 final keyframe upload failed · ' + error.message;
                throw error;
            });
            h3LastFrameUploadPromise.catch(function () {
                /* surfaced beside the picker and again on Generate */
            });
        });
        els.h3LastFrameClear.addEventListener('click', clearH3LastFrame);
        els.ltx2SourceStrength.addEventListener('input', function () {
            genState.ltx2SourceStrength = Math.max(0, Math.min(1, Number(this.value)));
            els.ltx2SourceStrengthVal.textContent = genState.ltx2SourceStrength.toFixed(2);
            updateLtx2SourceStrengthHelp();
        });
        els.ltx2RetakeStart.addEventListener('change', function () {
            genState.ltx2RetakeStart = Math.max(0, Number(this.value) || 0);
            updateLtx2VideoEditControls(false);
        });
        els.ltx2RetakeDuration.addEventListener('change', function () {
            genState.ltx2RetakeDuration = Math.max(2, Number(this.value) || 2);
            updateLtx2VideoEditControls(false);
        });
        els.ltx2ExtendDirection.addEventListener('change', function () {
            genState.ltx2ExtendDirection = this.value === 'start' ? 'start' : 'end';
            updateLtx2VideoEditControls(false);
        });
        els.ltx2ExtendSeconds.addEventListener('change', function () {
            genState.ltx2ExtendSeconds = Math.max(2, Math.min(20, Number(this.value) || 2));
            updateLtx2VideoEditControls(true);
        });
        els.ltx2V2vPresets.addEventListener('click', function (event) {
            var button = event.target.closest('button[data-strength]');
            if (!button)
                return;
            setLtx2SourceStrength(Number(button.dataset.strength));
        });
        els.loraAdd.addEventListener('click', addCanvasLora);
        els.loraClear.addEventListener('click', clearCanvasLoras);
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
            if (ModelUtils.archForModel(this.value) === 'ltxv') {
                var checkpoint = String(this.value).replace(/\.safetensors$/i, '');
                if (/bf16/i.test(checkpoint))
                    genState.ltx2Quant = 'bf16';
                else if (/fp8/i.test(checkpoint))
                    genState.ltx2Quant = 'fp8';
                var officialDev = checkpoint === 'ltx-2.3-22b-dev-fp8' ||
                    checkpoint === 'ltx-2.3-22b-dev-fp8-dequant-bf16';
                genState.ltx2Mode = officialDev || /distill/i.test(checkpoint)
                    ? 'distilled' : 'dev';
            }
            updateTopbarModel(this.value);
            updateCanvasUIForArch(ModelUtils.archForModel(this.value));
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
                if (genState.arch === 'ltxv' && activeCanvasLtx2Profiles().length > 0) {
                    applyCanvasLtx2Profile(preferredCanvasLtx2Profile());
                    History.push();
                    return;
                }
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
        clearCanvasEditSource();
        canvasLoras = [];
        renderCanvasLoras();
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
            ltx2AudioPolicy: genState.ltx2AudioPolicy,
            ltx2FeatureId: genState.ltx2FeatureId,
            ltx2FeatureWeight: genState.ltx2FeatureWeight,
            ltx2PostUpscaler: genState.ltx2PostUpscaler,
            ltx2PostUpscaleFactor: genState.ltx2PostUpscaleFactor,
            ltx2Mode: genState.ltx2Mode, ltx2ProfileKey: genState.ltx2ProfileKey,
            ltx2Quant: genState.ltx2Quant,
            ltx2SourceStrength: genState.ltx2SourceStrength,
            ltx2CameraMotion: genState.ltx2CameraMotion,
            ltx2RetakeStart: genState.ltx2RetakeStart,
            ltx2RetakeDuration: genState.ltx2RetakeDuration,
            ltx2ExtendDirection: genState.ltx2ExtendDirection,
            ltx2ExtendSeconds: genState.ltx2ExtendSeconds,
            h3Mode: genState.h3Mode,
            h3ProfileKey: genState.h3ProfileKey,
            h3Quant: genState.h3Quant,
            h3AttentionBackend: genState.h3AttentionBackend,
            editMode: genState.editMode, editEngine: genState.editEngine,
            editModelEngine: genState.editModelEngine,
            styleEntireImage: genState.styleEntireImage,
            lanpaintEngine: genState.lanpaintEngine,
            lanpaintNumSteps: genState.lanpaintNumSteps,
            lanpaintLambda: genState.lanpaintLambda,
            lanpaintStepSize: genState.lanpaintStepSize,
            lanpaintBeta: genState.lanpaintBeta,
            lanpaintFriction: genState.lanpaintFriction,
            lanpaintPromptMode: genState.lanpaintPromptMode,
            lanpaintBlendOverlap: genState.lanpaintBlendOverlap,
            lanpaintContextExpand: genState.lanpaintContextExpand,
            lanpaintEarlyStop: genState.lanpaintEarlyStop,
            lanpaintInnerThreshold: genState.lanpaintInnerThreshold,
            lanpaintInnerPatience: genState.lanpaintInnerPatience,
            loras: canvasLoras.map(function (lora) {
                return {
                    name: lora.name,
                    strength: Number(lora.strength),
                    enabled: lora.enabled !== false,
                    role: lora.role === 'distillation'
                        ? 'distillation' : 'overlay'
                };
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
                genState.ltx2AudioPolicy = state.genSettings.ltx2AudioPolicy ||
                    (genState.includeAudio ? 'generate' : 'none');
                genState.ltx2FeatureId = state.genSettings.ltx2FeatureId || 'standard';
                genState.ltx2FeatureWeight =
                    Number.isFinite(Number(state.genSettings.ltx2FeatureWeight))
                        ? Number(state.genSettings.ltx2FeatureWeight) : 1.0;
                genState.ltx2PostUpscaler =
                    state.genSettings.ltx2PostUpscaler || 'none';
                genState.ltx2PostUpscaleFactor =
                    Number(state.genSettings.ltx2PostUpscaleFactor) === 4 ? 4 : 2;
                genState.ltx2Mode = state.genSettings.ltx2Mode || 'distilled';
                genState.ltx2ProfileKey = state.genSettings.ltx2ProfileKey || '';
                genState.ltx2Quant = state.genSettings.ltx2Quant || 'bf16';
                genState.ltx2SourceStrength = Number.isFinite(Number(state.genSettings.ltx2SourceStrength)) ?
                    Math.max(0, Math.min(1, Number(state.genSettings.ltx2SourceStrength))) :
                    genState.ltx2SourceStrength;
                genState.ltx2CameraMotion =
                    state.genSettings.ltx2CameraMotion || 'none';
                genState.ltx2RetakeStart = Number.isFinite(Number(state.genSettings.ltx2RetakeStart))
                    ? Math.max(0, Number(state.genSettings.ltx2RetakeStart)) : 0;
                genState.ltx2RetakeDuration = Number.isFinite(Number(state.genSettings.ltx2RetakeDuration))
                    ? Math.max(2, Number(state.genSettings.ltx2RetakeDuration)) : 2;
                genState.ltx2ExtendDirection =
                    state.genSettings.ltx2ExtendDirection === 'start' ? 'start' : 'end';
                genState.ltx2ExtendSeconds = Number.isFinite(Number(state.genSettings.ltx2ExtendSeconds))
                    ? Math.max(2, Math.min(20, Number(state.genSettings.ltx2ExtendSeconds))) : 3;
                genState.h3Mode = ['t2va', 'i2va', 'l2va', 'fl2va', 'ref2va']
                    .indexOf(state.genSettings.h3Mode) >= 0
                    ? state.genSettings.h3Mode : 't2va';
                genState.h3ProfileKey = state.genSettings.h3ProfileKey || '';
                genState.h3Quant = ['int8-fast', 'int8', 'bf16']
                    .indexOf(state.genSettings.h3Quant) >= 0
                    ? state.genSettings.h3Quant : 'int8-fast';
                genState.h3AttentionBackend =
                    state.genSettings.h3AttentionBackend === 'sage-int8'
                        ? 'sage-int8' : 'cudnn';
                genState.editMode = state.genSettings.editMode || genState.editMode;
                genState.editEngine = state.genSettings.editEngine || genState.editEngine;
                genState.editModelEngine = state.genSettings.editModelEngine || genState.editModelEngine;
                genState.styleEntireImage = state.genSettings.styleEntireImage === true;
                genState.lanpaintEngine = state.genSettings.lanpaintEngine || genState.lanpaintEngine;
                genState.lanpaintNumSteps = Number.isFinite(Number(state.genSettings.lanpaintNumSteps)) ? Number(state.genSettings.lanpaintNumSteps) : genState.lanpaintNumSteps;
                genState.lanpaintLambda = Number.isFinite(Number(state.genSettings.lanpaintLambda)) ? Number(state.genSettings.lanpaintLambda) : genState.lanpaintLambda;
                genState.lanpaintStepSize = Number.isFinite(Number(state.genSettings.lanpaintStepSize)) ? Number(state.genSettings.lanpaintStepSize) : genState.lanpaintStepSize;
                genState.lanpaintBeta = Number.isFinite(Number(state.genSettings.lanpaintBeta)) ? Number(state.genSettings.lanpaintBeta) : genState.lanpaintBeta;
                genState.lanpaintFriction = Number.isFinite(Number(state.genSettings.lanpaintFriction)) ? Number(state.genSettings.lanpaintFriction) : genState.lanpaintFriction;
                genState.lanpaintPromptMode = state.genSettings.lanpaintPromptMode || genState.lanpaintPromptMode;
                genState.lanpaintBlendOverlap = Number.isFinite(Number(state.genSettings.lanpaintBlendOverlap)) ? Number(state.genSettings.lanpaintBlendOverlap) : genState.lanpaintBlendOverlap;
                genState.lanpaintContextExpand = Number.isFinite(Number(state.genSettings.lanpaintContextExpand)) ? Number(state.genSettings.lanpaintContextExpand) : genState.lanpaintContextExpand;
                genState.lanpaintEarlyStop = Number.isFinite(Number(state.genSettings.lanpaintEarlyStop)) ? Number(state.genSettings.lanpaintEarlyStop) : genState.lanpaintEarlyStop;
                genState.lanpaintInnerThreshold = Number.isFinite(Number(state.genSettings.lanpaintInnerThreshold)) ? Number(state.genSettings.lanpaintInnerThreshold) : genState.lanpaintInnerThreshold;
                genState.lanpaintInnerPatience = Number.isFinite(Number(state.genSettings.lanpaintInnerPatience)) ? Number(state.genSettings.lanpaintInnerPatience) : genState.lanpaintInnerPatience;
                lanpaintModeInitialized = state.genSettings.editMode === 'inpaint';
                canvasLoras = Array.isArray(state.genSettings.loras) ? state.genSettings.loras.map(function (lora) {
                    return {
                        name: String(lora.name || ''),
                        strength: Number.isFinite(Number(lora.strength)) ? Number(lora.strength) : 1,
                        enabled: lora.enabled !== false,
                        role: lora.role === 'distillation'
                            ? 'distillation' : 'overlay'
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
                if (els.ltx2AudioPolicy)
                    els.ltx2AudioPolicy.value = genState.ltx2AudioPolicy;
                if (els.ltx2Mode)
                    els.ltx2Mode.value = genState.ltx2Mode;
                if (els.ltx2CameraMotion)
                    els.ltx2CameraMotion.value = genState.ltx2CameraMotion;
                setLtx2SourceStrength(genState.ltx2SourceStrength);
                if (els.ltx2RetakeStart)
                    els.ltx2RetakeStart.value = String(genState.ltx2RetakeStart);
                if (els.ltx2RetakeDuration)
                    els.ltx2RetakeDuration.value = String(genState.ltx2RetakeDuration);
                if (els.ltx2ExtendDirection)
                    els.ltx2ExtendDirection.value = genState.ltx2ExtendDirection;
                if (els.ltx2ExtendSeconds)
                    els.ltx2ExtendSeconds.value = String(genState.ltx2ExtendSeconds);
                if (els.styleEntireImage)
                    els.styleEntireImage.checked = genState.styleEntireImage;
                if (els.editMode)
                    els.editMode.value = genState.editMode;
                if (els.editModelEngine)
                    els.editModelEngine.value = genState.editModelEngine;
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
                if (els.lanpaintContextExpand)
                    els.lanpaintContextExpand.value = String(genState.lanpaintContextExpand);
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
        if (arch === 'wan')
            return 'wan2.2';
        if (arch === 'qwen')
            return 'qwen-image';
        return arch;
    }
    function activeCanvasH3Runner() {
        var candidates = canvasVideoStatus && canvasVideoStatus.candidate_runners;
        if (!Array.isArray(candidates))
            return null;
        return candidates.find(function (entry) {
            return entry && entry.model === 'minimax_h3_t2va';
        }) || null;
    }
    function activeCanvasH3Mode() {
        var runner = activeCanvasH3Runner();
        var modes = runner && Array.isArray(runner.conditioned_modes)
            ? runner.conditioned_modes : [];
        return modes.find(function (mode) {
            return mode && mode.id === genState.h3Mode;
        }) || null;
    }
    function activeCanvasH3Profiles() {
        var runner = activeCanvasH3Runner();
        var profiles = runner && Array.isArray(runner.supported_profiles)
            ? runner.supported_profiles : [];
        return profiles.filter(function (profile) {
            return profile && profile.available === true &&
                (!profile.available_modes ||
                    profile.available_modes[genState.h3Quant] === true);
        });
    }
    function canvasH3ProfileKey(profile) {
        if (!profile)
            return '';
        return String(profile.id || [
            Number(profile.width) + 'x' + Number(profile.height),
            Number(profile.frames) + 'f',
            Number(profile.fps) + 'fps'
        ].join('_'));
    }
    function canvasH3ResolutionKey(profile) {
        if (!profile)
            return '';
        return [
            Number(profile.width) + 'x' + Number(profile.height),
            Number(profile.fps) + 'fps'
        ].join('_');
    }
    function canvasH3ProfileDuration(profile) {
        var duration = Number(profile && profile.duration);
        if (!Number.isFinite(duration) || duration <= 0)
            duration = Number(profile && profile.frames) /
                Number(profile && profile.fps);
        return duration;
    }
    function canvasH3ProfilesForResolution(resolutionKey) {
        return activeCanvasH3Profiles().filter(function (profile) {
            return canvasH3ResolutionKey(profile) === resolutionKey;
        });
    }
    function exactCanvasH3Profile() {
        if (!boundingBox)
            return null;
        var width = Math.round(boundingBox.width());
        var height = Math.round(boundingBox.height());
        return activeCanvasH3Profiles().find(function (profile) {
            return Number(profile.width) === width &&
                Number(profile.height) === height &&
                Number(profile.frames) === Number(genState.frames) &&
                Number(profile.fps) === Number(genState.fps);
        }) || null;
    }
    function preferredCanvasH3Profile() {
        var profiles = activeCanvasH3Profiles();
        if (!profiles.length)
            return null;
        var selected = profiles.find(function (profile) {
            return canvasH3ProfileKey(profile) === genState.h3ProfileKey;
        });
        if (selected)
            return selected;
        // Enter H3 on its longest admitted profile. A stale canvas bounding box
        // must not silently select one of the short high-resolution probes.
        return profiles.slice().sort(function (left, right) {
            var durationDelta = canvasH3ProfileDuration(right) -
                canvasH3ProfileDuration(left);
            if (Math.abs(durationDelta) > 0.011)
                return durationDelta;
            return Number(right.width) * Number(right.height) -
                Number(left.width) * Number(left.height);
        })[0];
    }
    function refreshCanvasH3DurationControl(selectedDuration) {
        if (!els.h3Duration)
            return;
        var profiles = activeCanvasH3Profiles();
        if (els.h3DurationList)
            els.h3DurationList.innerHTML = '';
        var durations = [];
        profiles.forEach(function (profile) {
            var duration = Number(canvasH3ProfileDuration(profile).toFixed(2));
            if (durations.indexOf(duration) >= 0)
                return;
            durations.push(duration);
        });
        durations.sort(function (left, right) { return left - right; });
        durations.forEach(function (duration) {
            var option = document.createElement('option');
            option.value = String(duration);
            option.label = duration + ' seconds';
            if (els.h3DurationList)
                els.h3DurationList.appendChild(option);
        });
        els.h3Duration.disabled = !profiles.length;
        if (!profiles.length) {
            els.h3Duration.value = '';
            els.h3Duration.title = '';
            return;
        }
        els.h3Duration.min = String(Math.min.apply(Math, durations));
        els.h3Duration.max = String(Math.max.apply(Math, durations));
        els.h3Duration.title = 'Supported H3 seconds at every resolution: ' +
            durations.join(', ') + '.';
        els.h3Duration.value = String(Number(Number(selectedDuration).toFixed(2)));
    }
    function applyCanvasH3Geometry(profile) {
        if (!profile)
            return false;
        var width = Number(profile.width);
        var height = Number(profile.height);
        genState.width = width;
        genState.height = height;
        genState.frames = Number(profile.frames);
        genState.fps = Number(profile.fps);
        genState.steps = Number(profile.steps) || 20;
        if (boundingBox) {
            var centerX = boundingBox.x() + boundingBox.width() / 2;
            var centerY = boundingBox.y() + boundingBox.height() / 2;
            boundingBox.width(width);
            boundingBox.height(height);
            boundingBox.x(centerX - width / 2);
            boundingBox.y(centerY - height / 2);
            updateHandles();
            updateSizeLabel();
            updateBboxInputs();
            if (stage)
                stage.batchDraw();
        }
        els.steps.value = String(genState.steps);
        els.stepsRange.value = String(genState.steps);
        els.framesInput.value = String(genState.frames);
        els.framesRange.value = String(genState.frames);
        els.fpsInput.value = String(genState.fps);
        els.fpsRange.value = String(genState.fps);
        updateCanvasDurationHint();
        return true;
    }
    function applyCanvasH3Profile(profile) {
        if (!applyCanvasH3Geometry(profile))
            return false;
        genState.h3ProfileKey = canvasH3ProfileKey(profile);
        var resolutionKey = canvasH3ResolutionKey(profile);
        var duration = canvasH3ProfileDuration(profile);
        if (els.h3Resolution)
            els.h3Resolution.value = resolutionKey;
        refreshCanvasH3DurationControl(duration);
        if (els.h3ProfileNote) {
            els.h3ProfileNote.textContent =
                profile.width + '\u00d7' + profile.height + ' \u00b7 ' +
                profile.frames + ' frames \u00b7 ' + profile.fps +
                ' FPS \u00b7 ' + Number(duration.toFixed(2)) +
                's \u00b7 one runtime-selectable Mojo runner';
        }
        return true;
    }
    function applyCanvasH3ResolutionDuration(resolutionKey, requestedDuration) {
        var profiles = canvasH3ProfilesForResolution(resolutionKey);
        if (!profiles.length)
            return false;
        var duration = Number(requestedDuration);
        var profile = profiles.find(function (candidate) {
            return Math.abs(canvasH3ProfileDuration(candidate) - duration) < 0.011;
        });
        return profile ? applyCanvasH3Profile(profile) : false;
    }
    function refreshCanvasH3ProfileControls(conditioned, mode) {
        if (!els.h3Resolution || !els.h3Duration)
            return;
        els.h3Resolution.innerHTML = '';
        els.h3Duration.value = '';
        if (els.h3DurationList)
            els.h3DurationList.innerHTML = '';
        var lockReason = 'MiniMax-H3 Resolution and Seconds are independent controls on one runtime request runner.';
        if (conditioned) {
            var definition = mode || (activeCanvasH3Runner() || {}).conditioned_profile;
            if (!definition)
                return;
            var option = document.createElement('option');
            option.value = 'conditioned';
            option.textContent = Number(definition.width) + '\u00d7' +
                Number(definition.height) + ' \u00b7 conditioned';
            els.h3Resolution.appendChild(option);
            els.h3Resolution.value = 'conditioned';
            els.h3Resolution.disabled = true;
            var duration = Number(definition.frames) / Number(definition.fps);
            var durationOption = document.createElement('option');
            durationOption.value = String(Number(duration.toFixed(2)));
            durationOption.label = Number(duration.toFixed(2)) + ' seconds';
            if (els.h3DurationList)
                els.h3DurationList.appendChild(durationOption);
            els.h3Duration.value = String(Number(duration.toFixed(2)));
            els.h3Duration.min = els.h3Duration.value;
            els.h3Duration.max = els.h3Duration.value;
            els.h3Duration.disabled = true;
            els.h3Duration.title = 'This conditioned runner has exact fixed geometry.';
            applyCanvasH3Geometry(definition);
            if (els.h3ProfileNote) {
                els.h3ProfileNote.textContent =
                    definition.width + '\u00d7' + definition.height + ' \u00b7 ' +
                    definition.frames + ' frames \u00b7 ' + definition.fps +
                    ' FPS \u00b7 ' + Number(duration.toFixed(2)) +
                    's \u00b7 exact conditioned runner';
            }
            setCanvasLtx2GeometryLocked(true, lockReason);
            return;
        }
        var profiles = activeCanvasH3Profiles();
        if (!profiles.length) {
            var unavailable = document.createElement('option');
            unavailable.value = '';
            unavailable.textContent = 'No admitted H3 resolutions available';
            els.h3Resolution.appendChild(unavailable);
            els.h3Resolution.disabled = true;
            els.h3Duration.disabled = true;
            els.h3Duration.placeholder = 'No admitted H3 seconds available';
            if (els.h3ProfileNote)
                els.h3ProfileNote.textContent = 'No executable H3 request runner was reported by the server.';
            setCanvasLtx2GeometryLocked(false);
            return;
        }
        var seen = {};
        profiles.forEach(function (profile) {
            var key = canvasH3ResolutionKey(profile);
            if (seen[key])
                return;
            seen[key] = true;
            var option = document.createElement('option');
            option.value = key;
            option.textContent = profile.width + '\u00d7' + profile.height +
                ' \u00b7 ' + profile.fps + ' FPS';
            els.h3Resolution.appendChild(option);
        });
        els.h3Resolution.disabled = false;
        applyCanvasH3Profile(preferredCanvasH3Profile());
        setCanvasLtx2GeometryLocked(true, lockReason);
    }
    function refreshCanvasH3Controls() {
        if (!els.h3Mode)
            return;
        var runner = activeCanvasH3Runner();
        var modes = runner && Array.isArray(runner.conditioned_modes)
            ? runner.conditioned_modes : [];
        Array.from(els.h3Mode.options).forEach(function (option) {
            if (option.value === 't2va') {
                option.disabled = !(runner && Array.isArray(runner.supported_profiles) &&
                    runner.supported_profiles.some(function (profile) {
                        return profile && profile.available === true;
                    }));
                return;
            }
            var definition = modes.find(function (mode) {
                return mode && mode.id === option.value;
            });
            option.disabled = !definition || definition.available !== true;
            if (definition && definition.available !== true && definition.reason)
                option.title = definition.reason;
        });
        if (!Array.from(els.h3Mode.options).some(function (option) {
            return option.value === genState.h3Mode && !option.disabled;
        })) {
            var firstReady = Array.from(els.h3Mode.options).find(function (option) {
                return !option.disabled;
            });
            genState.h3Mode = firstReady ? firstReady.value : 't2va';
        }
        els.h3Mode.value = genState.h3Mode;
        var mode = activeCanvasH3Mode();
        Array.from(els.h3Quant.options).forEach(function (option) {
            var ready = genState.h3Mode === 't2va'
                ? !!(runner && Array.isArray(runner.quant_modes) &&
                    runner.quant_modes.some(function (candidate) {
                        return candidate && candidate.id === option.value && candidate.available === true;
                    }))
                : !!(mode && mode.available_modes && mode.available_modes[option.value] === true);
            option.disabled = !ready;
        });
        if (!Array.from(els.h3Quant.options).some(function (option) {
            return option.value === genState.h3Quant && !option.disabled;
        })) {
            var firstQuant = Array.from(els.h3Quant.options).find(function (option) {
                return !option.disabled;
            });
            genState.h3Quant = firstQuant ? firstQuant.value : 'int8-fast';
        }
        els.h3Quant.value = genState.h3Quant;
        if (genState.h3Quant === 'int8-fast')
            genState.h3AttentionBackend = 'cudnn';
        els.h3Attention.value = genState.h3AttentionBackend;
        els.h3Attention.disabled = genState.h3Quant === 'int8-fast';
        var conditioned = genState.h3Mode !== 't2va';
        var needsLast = genState.h3Mode === 'l2va' || genState.h3Mode === 'fl2va';
        els.h3LastFrameRow.style.display = needsLast ? 'block' : 'none';
        if (conditioned) {
            if (runner && typeof runner.conditioned_prompt === 'string') {
                genState.prompt = runner.conditioned_prompt;
                els.prompt.value = runner.conditioned_prompt;
            }
        } else {
            if (runner && typeof runner.test_prompt === 'string') {
                genState.prompt = runner.test_prompt;
                els.prompt.value = runner.test_prompt;
            }
        }
        refreshCanvasH3ProfileControls(conditioned, mode);
        els.steps.value = String(genState.steps);
        els.stepsRange.value = String(genState.steps);
        els.framesInput.value = String(genState.frames);
        els.framesRange.value = String(genState.frames);
        els.fpsInput.value = String(genState.fps);
        els.fpsRange.value = String(genState.fps);
        [els.framesInput, els.framesRange, els.fpsInput, els.fpsRange].forEach(function (control) {
            control.disabled = true;
            control.title = conditioned
                ? 'This conditioned MiniMax-H3 runner has exact fixed geometry.'
                : 'Choose H3 Resolution and Seconds; Canvas resolves the exact frame count and FPS.';
        });
        [els.steps, els.stepsRange].forEach(function (control) {
            control.disabled = conditioned;
            control.title = conditioned
                ? 'Conditioned MiniMax-H3 uses the installed 20-step modulation cache.'
                : '';
        });
        if (boundingBox) {
            boundingBox.width(genState.width);
            boundingBox.height(genState.height);
            updateHandles();
            updateSizeLabel();
            updateBboxInputs();
            stage.batchDraw();
        }
        if (els.h3ModeNote) {
            els.h3ModeNote.textContent = genState.h3Mode === 't2va'
                ? 'Choose Resolution and Seconds from the admitted T2V profiles. One H3 runner applies resolution, frames, FPS, and precision at request time.'
                : (genState.h3Mode === 'ref2va'
                    ? 'Choose one Source image as an identity/style reference. It is center-cropped to 768 × 768 for conditioning and is not inserted as frame zero.'
                : (genState.h3Mode === 'l2va'
                    ? 'Choose a required final keyframe below. The generated clip evolves toward it.'
                    : (genState.h3Mode === 'fl2va'
                        ? 'Choose a Source image for frame zero, then choose the required final keyframe. The Source is center-cropped to square.'
                        : 'The selected Source image is center-cropped to square and becomes frame zero. It is not a reference-only image.')));
        }
        genState.includeAudio = true;
        updateCanvasDurationHint();
    }
    function activeCanvasWanRunner() {
        var candidates = canvasVideoStatus && canvasVideoStatus.candidate_runners;
        if (!Array.isArray(candidates))
            return null;
        return candidates.find(function (entry) {
            return entry && entry.model === 'wan22_t2v';
        }) || null;
    }
    function canvasWanCreatorI2vSize(sourceWidth, sourceHeight) {
        var area = 1280 * 704;
        var ratio = sourceWidth / sourceHeight;
        var outputWidth = Math.sqrt(area * ratio);
        var outputHeight = area / outputWidth;
        var widthFirst = Math.floor(outputWidth / 32) * 32;
        var heightFromWidth = Math.floor(area / widthFirst / 32) * 32;
        var ratioWidthFirst = widthFirst / heightFromWidth;
        var heightFirst = Math.floor(outputHeight / 32) * 32;
        var widthFromHeight = Math.floor(area / heightFirst / 32) * 32;
        var ratioHeightFirst = widthFromHeight / heightFirst;
        var distortionWidth = Math.max(
            ratio / ratioWidthFirst, ratioWidthFirst / ratio);
        var distortionHeight = Math.max(
            ratio / ratioHeightFirst, ratioHeightFirst / ratio);
        return distortionWidth < distortionHeight
            ? { width: widthFirst, height: heightFromWidth }
            : { width: widthFromHeight, height: heightFirst };
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
        if (genState.arch === 'minimax_h3') {
            if (name === 'image_to_image' || name === 'image_conditioning') {
                return {
                    supported: genState.h3Mode === 'i2va' || genState.h3Mode === 'fl2va' ||
                        genState.h3Mode === 'ref2va',
                    policy: genState.h3Mode === 'ref2va' ? 'reference_only' : 'keyframe',
                    note: genState.h3Mode === 'ref2va'
                        ? 'The Source guides identity/style without becoming frame zero'
                        : 'The Canvas image becomes the H3 first-frame keyframe'
                };
            }
            return {
                supported: false,
                policy: 'fail_loud',
                reason: 'MiniMax-H3 Canvas admits only its explicit T2V/I2V/L2V/FL2V/Ref2VA controls'
            };
        }
        if (genState.arch === 'wan') {
            var runner = activeCanvasWanRunner();
            var loraMode = runner && runner.modes && runner.modes.lora;
            if (name === 'lora') {
                return {
                    supported: !!(loraMode && loraMode.available),
                    max_count: 1,
                    policy: 'fail_loud',
                    reason: 'Wan LoRAs must target the installed TI2V-5B base'
                };
            }
            if (name === 'multi_lora') {
                return {
                    supported: false,
                    policy: 'fail_loud',
                    reason: 'Wan2.2 TI2V-5B currently accepts one resident LoRA per render'
                };
            }
            if (name === 'image_to_image' || name === 'image_conditioning') {
                return {
                    supported: !!(runner && runner.status === 'quality_profile_ready'),
                    policy: 'first_frame',
                    note: 'The Canvas image becomes Wan TI2V frame zero'
                };
            }
        }
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
        else if (genState.editMode === 'edit_models') {
            var editDefinition = editModelEngineDefinition(genState.editModelEngine);
            var instructionEdit = canvasFeature('instruction_edit');
            els.capabilityNote.textContent = editDefinition ?
                editDefinition.label + ': ' +
                    (instructionEdit.note || instructionEdit.reason || 'native instruction editing') :
                'No edit model is selected.';
            els.capabilityNote.style.display = 'block';
        }
        else if (genState.editMode === 'i2v_ltx23') {
            els.capabilityNote.textContent =
                'I2V - LTX 2.3 uses one Canvas image as frame-zero guidance, encodes it at both native stages, and generates through the selected compiled Mojo profile.';
            els.capabilityNote.style.display = 'block';
        }
        else if (genState.editMode === 'retake_ltx23') {
            els.capabilityNote.textContent =
                'Retake - LTX 2.3 uses a temporal denoise mask: the selected window is generated and all source-video tokens outside it are frozen.';
            els.capabilityNote.style.display = 'block';
        }
        else if (genState.editMode === 'extend_ltx23') {
            els.capabilityNote.textContent =
                'Extend - LTX 2.3 preserves the source, generates a new leading or trailing region, and overlaps the temporal mask by 0.5 seconds at the seam.';
            els.capabilityNote.style.display = 'block';
        }
        else if (genState.arch === 'ltxv') {
            els.capabilityNote.textContent = 'Mojo T2V/I2V/V2V request profile. Leave the source and Canvas empty for text-to-video, load an image to condition frame 0, or load a video to preserve its motion through full-video conditioning.';
            els.capabilityNote.style.display = 'block';
        }
        else if (genState.arch === 'wan') {
            els.capabilityNote.textContent =
                'Wan2.2 TI2V-5B: an empty Canvas generates T2V; Canvas content becomes the creator first-frame I2V source. One shape-validated 5B LoRA may be loaded.';
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
        var loraSupported = genState.editMode !== 'edit_models' &&
            (genState.arch === 'ltxv' || canvasFeature('lora').supported);
        els.loraSection.style.display = loraSupported ? 'flex' : 'none';
        var controlItem = document.querySelector('.cv-layer-type-item[data-type="control"]');
        if (controlItem) {
            var controlSupported = genState.arch !== 'ltxv' && canvasFeature('controlnet').supported;
            controlItem.classList.toggle('capability-unavailable', !controlSupported);
            controlItem.title = controlSupported ? '' : (genState.arch === 'ltxv' ? 'LTX2 I2V admits one frame-zero guide image; ControlNet is not admitted' : canvasFeatureReason('controlnet'));
        }
        var maskItem = document.querySelector('.cv-layer-type-item[data-type="mask"]');
        if (maskItem) {
            var inpaintSupported = genState.arch !== 'ltxv' && canvasFeature('inpaint').supported;
            maskItem.classList.toggle('capability-unavailable', !inpaintSupported);
            maskItem.title = inpaintSupported ? '' : (genState.arch === 'ltxv' ? 'LTX2 I2V does not admit a painted mask' : canvasFeatureReason('inpaint'));
        }
        if (typeof CanvasRefImages !== 'undefined' && CanvasRefImages.setCompatibility) {
            var refsSupported = genState.editMode === 'style' ||
                (genState.arch !== 'ltxv' && canvasFeature('image_conditioning').supported);
            CanvasRefImages.setCompatibility(refsSupported, genState.editMode === 'style' ? '' :
                (genState.arch === 'ltxv' ? 'Use the selected Canvas image as the LTX2 frame-zero guide' : canvasFeatureReason('image_conditioning')));
        }
    }
    function validateCanvasGenerationFeatures(hasContent, hasMask) {
        if (genState.arch === 'ltxv') {
            if (genState.editMode === 'i2v_ltx23' && editSourceIsVideo)
                return 'I2V - LTX 2.3 requires an image source, not a video';
            if ((genState.editMode === 'retake_ltx23' ||
                genState.editMode === 'extend_ltx23') && !editSourceIsVideo)
                return 'Retake and Extend - LTX 2.3 require a source video';
            if (hasMask && !(editSourceIsVideo && editSourceFile))
                return 'An LTX2 painted mask requires a loaded V2V source video';
            return '';
        }
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
            return {
                name: lora.name,
                strength: Number(lora.strength),
                role: lora.role === 'distillation'
                    ? 'distillation' : 'overlay'
            };
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
        var loadedCount = canvasLoras.length;
        if (els.loraCompat)
            els.loraCompat.textContent = choices.length + ' compatible · ' + loadedCount + ' loaded';
        if (els.loraClear)
            els.loraClear.disabled = loadedCount === 0;
        els.loraList.innerHTML = '';
        var compatibleNames = choices.map(function (lora) { return lora.name; });
        var rows = canvasLoras.filter(function (lora) {
            return compatibleNames.indexOf(lora.name) >= 0;
        });
        if (rows.length === 0) {
            var empty = document.createElement('div');
            empty.className = 'cv-lora-empty';
            if (loadedCount > 0)
                empty.textContent = loadedCount + ' loader' + (loadedCount === 1 ? '' : 's') + ' stored for other models · Clear all removes them';
            else
                empty.textContent = choices.length ? 'No LoRA loaders added' : 'No compatible LoRAs found';
            els.loraList.appendChild(empty);
            return;
        }
        rows.forEach(function (lora) {
            if (lora.role !== 'distillation')
                lora.role = 'overlay';
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
            var role = null;
            if (genState.arch === 'ltxv') {
                role = document.createElement('select');
                role.className = 'cv-select cv-lora-role';
                role.title = "Use as an authored overlay or as this checkpoint's sampling distillation adapter";
                ['overlay', 'distillation'].forEach(function (value) {
                    var option = document.createElement('option');
                    option.value = value;
                    option.textContent = value === 'distillation'
                        ? 'Distillation' : 'Overlay';
                    option.selected = lora.role === value;
                    role.appendChild(option);
                });
                role.addEventListener('change', function () {
                    if (role.value === 'distillation') {
                        canvasLoras.forEach(function (candidate) {
                            if (candidate !== lora)
                                candidate.role = 'overlay';
                        });
                    }
                    lora.role = role.value === 'distillation'
                        ? 'distillation' : 'overlay';
                    renderCanvasLoras();
                });
            }
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
            if (role)
                row.appendChild(role);
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
        canvasLoras.push({
            name: name,
            strength: 1,
            enabled: true,
            role: genState.arch === 'ltxv' && /distill/i.test(name)
                ? 'distillation' : 'overlay'
        });
        renderCanvasLoras();
    }
    function clearCanvasLoras() {
        canvasLoras = [];
        renderCanvasLoras();
    }
    function loadLtx2TemplateProfile() {
        els.loadLtx2Template.disabled = true;
        Promise.resolve().then(function () {
            var exactCheckpoint = 'ltx-2.3-22b-distilled';
            var exactOption = Array.from(els.model.options || []).find(function (option) {
                return option.value === exactCheckpoint;
            });
            if (!exactOption)
                throw new Error('the complete LTX Desktop creator checkpoint is not registered');

            genState.model = exactCheckpoint;
            genState.ltx2Quant = 'bf16';
            genState.ltx2Mode = 'distilled';
            genState.capsPositive = '';
            genState.capsNegative = '';
            genState.noiseFixture = '';
            genState.includeAudio = true;
            genState.ltx2AudioPolicy = genState.editMode === 'retake_ltx23'
                ? (editSourceVideoProbe && editSourceVideoProbe.has_audio ? 'preserve' : 'none')
                : 'generate';
            genState.ltx2FeatureId = 'standard';
            genState.ltx2FeatureWeight = 1.0;
            genState.ltx2PostUpscaler = 'none';
            genState.ltx2PostUpscaleFactor = 2;
            canvasLoras = [];

            els.model.value = exactCheckpoint;
            els.ltx2Quant.value = 'bf16';
            els.ltx2Mode.value = 'distilled';
            els.capsPositive.value = '';
            els.capsNegative.value = '';
            els.noiseFixture.value = '';
            els.ltx2AudioPolicy.value = genState.ltx2AudioPolicy;
            updateTopbarModel(exactCheckpoint);
            updateCanvasUIForArch('ltxv');
            applyCanvasLtx2GuidanceMode();
            refreshCanvasLtx2FeatureControls();
            refreshCanvasLtx2PostUpscaleControls();
            updateCanvasDurationHint();
            renderCanvasLoras();
        }).catch(function (error) {
            showError('LTX Desktop creator profile failed: ' + error.message);
        }).finally(function () {
            els.loadLtx2Template.disabled = false;
        });
    }
    function activeCanvasLtx2RequestMode() {
        var candidates = canvasVideoStatus && canvasVideoStatus.candidate_runners;
        if (!Array.isArray(candidates))
            return null;
        var runner = candidates.find(function (entry) {
            return entry && entry.model === 'ltx2_t2v_av';
        });
        return runner && runner.modes && runner.modes.ltx2_mojo_request || null;
    }
    function canvasLtx2CheckpointWorkflow() {
        var mode = activeCanvasLtx2RequestMode();
        var profiles = mode && Array.isArray(mode.checkpoint_workflows)
            ? mode.checkpoint_workflows : [];
        var checkpoint = String(genState.model || '')
            .replace(/\.safetensors$/i, '').toLowerCase();
        return profiles.find(function (profile) {
            return profile && Array.isArray(profile.checkpoints) &&
                profile.checkpoints.some(function (name) {
                    return String(name || '').replace(/\.safetensors$/i, '')
                        .toLowerCase() === checkpoint;
                });
        }) || null;
    }
    function activeCanvasLtx2Profiles() {
        var mode = activeCanvasLtx2RequestMode();
        var profiles = mode && Array.isArray(mode.supported_profiles)
            ? mode.supported_profiles : [];
        var requestMode = genState.editMode === 'retake_ltx23' ? 'retake' :
            (genState.editMode === 'extend_ltx23'
                ? 'extend_' + (genState.ltx2ExtendDirection === 'start' ? 'start' : 'end')
                : 'standard');
        return profiles.filter(function (profile) {
            return profile && profile.available === true &&
                (!Array.isArray(profile.modes) || profile.modes.indexOf(requestMode) !== -1);
        });
    }
    function refreshCanvasLtx2QuantControls() {
        if (!els.ltx2Quant)
            return;
        var mode = activeCanvasLtx2RequestMode();
        var modes = mode && Array.isArray(mode.quant_modes) ? mode.quant_modes : [];
        var creatorTemporalEdit = genState.editMode === 'retake_ltx23' ||
            genState.editMode === 'extend_ltx23';
        if (modes.length) {
            Array.from(els.ltx2Quant.options).forEach(function (option) {
                var definition = modes.find(function (candidate) {
                    return candidate && candidate.id === option.value;
                });
                option.disabled = !definition || definition.available !== true ||
                    (creatorTemporalEdit && option.value !== 'bf16');
            });
        }
        if (creatorTemporalEdit)
            genState.ltx2Quant = 'bf16';
        var selected = Array.from(els.ltx2Quant.options).find(function (option) {
            return option.value === genState.ltx2Quant && !option.disabled;
        }) || Array.from(els.ltx2Quant.options).find(function (option) {
            return !option.disabled;
        });
        genState.ltx2Quant = selected ? selected.value : '';
        els.ltx2Quant.value = genState.ltx2Quant;
        els.ltx2Quant.disabled = !selected;
    }
    function activeCanvasLtx2Features() {
        var mode = activeCanvasLtx2RequestMode();
        return mode && Array.isArray(mode.feature_adapters)
            ? mode.feature_adapters : [];
    }
    function canvasLtx2Feature(id) {
        return activeCanvasLtx2Features().find(function (feature) {
            return feature && String(feature.id || '') === String(id || '');
        }) || null;
    }
    function admittedCanvasLtx2Feature(feature) {
        return !!feature && feature.installed === true &&
            (feature.status === 'overlay_admitted' ||
                feature.status === 'v2a_admitted');
    }
    function applyCanvasLtx2Feature(id) {
        var feature = id === 'standard' ? null : canvasLtx2Feature(id);
        if (feature && !admittedCanvasLtx2Feature(feature))
            feature = null;
        var changed = genState.ltx2FeatureId !==
            (feature ? String(feature.id) : 'standard');
        genState.ltx2FeatureId = feature ? String(feature.id) : 'standard';
        if (feature && (changed || !Number.isFinite(Number(genState.ltx2FeatureWeight)))) {
            genState.ltx2FeatureWeight = Number(feature.weight_default);
            if (!Number.isFinite(genState.ltx2FeatureWeight))
                genState.ltx2FeatureWeight = 1.0;
        }
        if (els.ltx2Feature)
            els.ltx2Feature.value = genState.ltx2FeatureId;
        if (els.ltx2FeatureWeight) {
            els.ltx2FeatureWeight.disabled = !feature;
            els.ltx2FeatureWeight.value = String(genState.ltx2FeatureWeight);
            els.ltx2FeatureWeight.min = String(feature && feature.weight_min != null
                ? feature.weight_min : -10);
            els.ltx2FeatureWeight.max = String(feature && feature.weight_max != null
                ? feature.weight_max : 10);
        }
        if (!feature) {
            if (els.ltx2FeatureNote)
                els.ltx2FeatureNote.textContent =
                    'Standard LTX2 generation without a dedicated feature adapter.';
            return;
        }
        if (feature.id === 'cinemagraph') {
            if (els.ltx2FeatureNote)
                els.ltx2FeatureNote.textContent =
                    'Requires a loaded source image and the exact prompt trigger CINEMAGRAPH_MOTION. Recommended weight range ' +
                    feature.weight_min + '\u2013' + feature.weight_max + '.';
        }
        else if (feature.id === 'foley-v2a') {
            genState.ltx2SourceStrength = 1.0;
            genState.ltx2AudioPolicy = 'generate';
            genState.includeAudio = true;
            if (els.ltx2SourceStrength) els.ltx2SourceStrength.value = '1';
            if (els.ltx2SourceStrengthVal) els.ltx2SourceStrengthVal.textContent = '1.00';
            if (els.ltx2AudioPolicy) els.ltx2AudioPolicy.value = 'generate';
            if (els.ltx2FeatureNote)
                els.ltx2FeatureNote.textContent =
                    'Requires a source video that exactly matches the selected native profile. The video is frozen; only Foley audio is generated.';
        }
    }
    function refreshCanvasLtx2FeatureControls() {
        if (!els.ltx2Feature)
            return;
        els.ltx2Feature.innerHTML = '';
        var standard = document.createElement('option');
        standard.value = 'standard';
        standard.textContent = 'Standard T2V / I2V / V2V';
        els.ltx2Feature.appendChild(standard);
        activeCanvasLtx2Features().forEach(function (feature) {
            if (!feature || feature.status === 'companion_only')
                return;
            var option = document.createElement('option');
            option.value = String(feature.id || '');
            option.disabled = !admittedCanvasLtx2Feature(feature);
            option.textContent = String(feature.label || feature.id) +
                (feature.installed !== true ? ' \u00b7 weights missing' :
                    (option.disabled ? ' \u00b7 runtime pending' : ''));
            option.title = option.disabled
                ? 'Installed artifact requires a dedicated reference-token feature runner'
                : '';
            els.ltx2Feature.appendChild(option);
        });
        var selected = canvasLtx2Feature(genState.ltx2FeatureId);
        applyCanvasLtx2Feature(
            selected && admittedCanvasLtx2Feature(selected)
                ? genState.ltx2FeatureId : 'standard'
        );
    }
    function activeCanvasLtx2PostUpscalers() {
        var mode = activeCanvasLtx2RequestMode();
        return mode && Array.isArray(mode.post_upscalers)
            ? mode.post_upscalers : [];
    }
    function canvasLtx2PostUpscaler(id) {
        return activeCanvasLtx2PostUpscalers().find(function (upscaler) {
            return upscaler && String(upscaler.id || '') === String(id || '');
        }) || null;
    }
    function applyCanvasLtx2PostUpscaler(id) {
        var upscaler = id === 'none' ? null : canvasLtx2PostUpscaler(id);
        if (upscaler && upscaler.available !== true)
            upscaler = null;
        genState.ltx2PostUpscaler = upscaler ? String(upscaler.id) : 'none';
        if (els.ltx2PostUpscaler)
            els.ltx2PostUpscaler.value = genState.ltx2PostUpscaler;
        var scales = upscaler && Array.isArray(upscaler.scales)
            ? upscaler.scales.map(Number) : [2, 4];
        if (upscaler && scales.indexOf(genState.ltx2PostUpscaleFactor) < 0)
            genState.ltx2PostUpscaleFactor = scales[0];
        if (els.ltx2PostUpscaleFactor) {
            Array.from(els.ltx2PostUpscaleFactor.options).forEach(function (option) {
                option.disabled = upscaler && scales.indexOf(Number(option.value)) < 0;
            });
            els.ltx2PostUpscaleFactor.value = String(genState.ltx2PostUpscaleFactor);
            els.ltx2PostUpscaleFactor.disabled = !upscaler;
        }
        if (els.ltx2PostUpscaleNote) {
            els.ltx2PostUpscaleNote.textContent = !upscaler
                ? 'Runs after native LTX2 decode. Availability and speed are reported by the server.'
                : String(upscaler.reason || upscaler.label || upscaler.id);
        }
    }
    function refreshCanvasLtx2PostUpscaleControls() {
        if (!els.ltx2PostUpscaler)
            return;
        els.ltx2PostUpscaler.innerHTML = '';
        var none = document.createElement('option');
        none.value = 'none';
        none.textContent = 'None';
        els.ltx2PostUpscaler.appendChild(none);
        activeCanvasLtx2PostUpscalers().forEach(function (upscaler) {
            if (!upscaler)
                return;
            var option = document.createElement('option');
            option.value = String(upscaler.id || '');
            option.disabled = upscaler.available !== true;
            option.textContent = String(upscaler.label || upscaler.id) +
                (upscaler.status === 'experimental_slow' ? ' \u00b7 slow' :
                    (option.disabled ? ' \u00b7 unavailable' : ''));
            els.ltx2PostUpscaler.appendChild(option);
        });
        var selected = canvasLtx2PostUpscaler(genState.ltx2PostUpscaler);
        applyCanvasLtx2PostUpscaler(
            selected && selected.available === true
                ? genState.ltx2PostUpscaler : 'none'
        );
    }
    function canvasLtx2ProfileKey(profile) {
        if (!profile)
            return '';
        return [
            Number(profile.width) + 'x' + Number(profile.height),
            Number(profile.frames) + 'f',
            Number(profile.fps) + 'fps'
        ].join('_');
    }
    function canvasLtx2ResolutionKey(profile) {
        if (!profile)
            return '';
        return String(profile.id || [
            Number(profile.width) + 'x' + Number(profile.height),
            Number(profile.fps) + 'fps'
        ].join('_'));
    }
    function canvasLtx2ProfileDuration(profile) {
        var duration = Number(profile && profile.duration);
        if (!Number.isFinite(duration) || duration <= 0)
            duration = Number(profile && profile.frames) / Number(profile && profile.fps);
        return duration;
    }
    function canvasLtx2ProfilesForResolution(resolutionKey) {
        return activeCanvasLtx2Profiles().filter(function (profile) {
            return canvasLtx2ResolutionKey(profile) === resolutionKey;
        });
    }
    function refreshCanvasLtx2DurationControl(resolutionKey, selectedDuration) {
        if (!els.ltx2Duration)
            return;
        var profiles = canvasLtx2ProfilesForResolution(resolutionKey);
        if (els.ltx2DurationList)
            els.ltx2DurationList.innerHTML = '';
        var durations = [];
        profiles.forEach(function (profile) {
            var duration = canvasLtx2ProfileDuration(profile);
            durations.push(Number(duration.toFixed(2)));
            var option = document.createElement('option');
            option.value = String(Number(duration.toFixed(2)));
            option.label = Number(duration.toFixed(2)) + ' seconds';
            if (els.ltx2DurationList)
                els.ltx2DurationList.appendChild(option);
        });
        els.ltx2Duration.disabled = !profiles.length;
        if (profiles.length) {
            els.ltx2Duration.min = String(Math.min.apply(Math, durations));
            els.ltx2Duration.max = String(Math.max.apply(Math, durations));
            els.ltx2Duration.title = 'Supported seconds: ' + durations.join(', ');
            els.ltx2Duration.value = String(Number(Number(selectedDuration).toFixed(2)));
        }
        else {
            els.ltx2Duration.value = '';
            els.ltx2Duration.title = '';
        }
    }
    function applyCanvasLtx2ResolutionDuration(
        resolutionKey, requestedDuration, allowNearest
    ) {
        var profiles = canvasLtx2ProfilesForResolution(resolutionKey);
        if (!profiles.length)
            return false;
        var duration = Number(requestedDuration);
        var profile = profiles.find(function (candidate) {
            return Math.abs(canvasLtx2ProfileDuration(candidate) - duration) < 0.001;
        });
        if (!profile && allowNearest) {
            profile = profiles.slice().sort(function (left, right) {
                return Math.abs(canvasLtx2ProfileDuration(left) - duration) -
                    Math.abs(canvasLtx2ProfileDuration(right) - duration);
            })[0];
        }
        if (!profile)
            return false;
        return applyCanvasLtx2Profile(profile);
    }
    function exactCanvasLtx2Profile() {
        if (!boundingBox)
            return null;
        var width = Math.round(boundingBox.width());
        var height = Math.round(boundingBox.height());
        return activeCanvasLtx2Profiles().find(function (profile) {
            return Number(profile.width) === width &&
                Number(profile.height) === height &&
                Number(profile.frames) === Number(genState.frames) &&
                Number(profile.fps) === Number(genState.fps);
        }) || null;
    }
    function preferredCanvasLtx2Profile() {
        var profiles = activeCanvasLtx2Profiles();
        if (!profiles.length)
            return null;
        var exact = exactCanvasLtx2Profile();
        if (exact)
            return exact;
        var selected = profiles.find(function (profile) {
            return canvasLtx2ProfileKey(profile) === genState.ltx2ProfileKey;
        });
        if (selected)
            return selected;
        var mode = activeCanvasLtx2RequestMode();
        var compiled = mode && mode.compiled_profile;
        var compiledProfile = compiled && profiles.find(function (profile) {
            return canvasLtx2ProfileKey(profile) === canvasLtx2ProfileKey(compiled);
        });
        if (compiledProfile)
            return compiledProfile;
        var currentWidth = boundingBox ? Math.max(1, boundingBox.width()) : 512;
        var currentHeight = boundingBox ? Math.max(1, boundingBox.height()) : 768;
        var currentRatio = currentWidth / currentHeight;
        var currentArea = currentWidth * currentHeight;
        return profiles.slice().sort(function (left, right) {
            function score(profile) {
                var ratio = Number(profile.width) / Number(profile.height);
                var area = Number(profile.width) * Number(profile.height);
                return Math.abs(Math.log(ratio / currentRatio)) * 3 +
                    Math.abs(Math.log(area / currentArea)) * 0.25 +
                    Math.abs(Number(profile.frames) - Number(genState.frames)) / 1000;
            }
            return score(left) - score(right);
        })[0];
    }
    function setCanvasLtx2GeometryLocked(locked, overrideReason) {
        var bboxW = document.getElementById('cv-bbox-w');
        var bboxH = document.getElementById('cv-bbox-h');
        var reason = overrideReason ||
            'LTX2 Native resolution and Seconds resolve width, height, frames, and FPS on one compiled Mojo runner.';
        [bboxW, bboxH].forEach(function (control) {
            if (!control)
                return;
            control.disabled = locked;
            control.title = locked ? reason : '';
        });
        resizeHandles.forEach(function (handle) {
            var interactive = activeTool === 'select' && !locked;
            handle.draggable(interactive);
            handle.listening(interactive);
            handle.opacity(locked ? 0.45 : 1);
        });
    }
    function applyCanvasLtx2Profile(profile) {
        if (!profile)
            return false;
        var width = Number(profile.width);
        var height = Number(profile.height);
        genState.frames = Number(profile.frames);
        genState.fps = Number(profile.fps);
        genState.ltx2ProfileKey = canvasLtx2ProfileKey(profile);
        if (boundingBox) {
            var centerX = boundingBox.x() + boundingBox.width() / 2;
            var centerY = boundingBox.y() + boundingBox.height() / 2;
            boundingBox.width(width);
            boundingBox.height(height);
            boundingBox.x(centerX - width / 2);
            boundingBox.y(centerY - height / 2);
            updateHandles();
            updateSizeLabel();
            updateBboxInputs();
            if (stage)
                stage.batchDraw();
        }
        if (els.framesInput)
            els.framesInput.value = String(genState.frames);
        if (els.framesRange)
            els.framesRange.value = String(genState.frames);
        if (els.fpsInput)
            els.fpsInput.value = String(genState.fps);
        if (els.fpsRange)
            els.fpsRange.value = String(genState.fps);
        if (els.ltx2Profile)
            els.ltx2Profile.value = genState.ltx2ProfileKey;
        var duration = canvasLtx2ProfileDuration(profile);
        if (els.ltx2Resolution)
            els.ltx2Resolution.value = canvasLtx2ResolutionKey(profile);
        refreshCanvasLtx2DurationControl(
            canvasLtx2ResolutionKey(profile),
            duration
        );
        if (els.ltx2ProfileNote) {
            els.ltx2ProfileNote.textContent =
                width + '\u00d7' + height + ' \u00b7 ' + genState.frames +
                ' frames \u00b7 ' + genState.fps + ' FPS \u00b7 ' +
                Number(duration.toFixed(2)) + 's \u00b7 exact Mojo runner';
        }
        updateCanvasDurationHint();
        return true;
    }
    function refreshCanvasLtx2ProfileControls() {
        if (!els.ltx2Profile)
            return;
        var profiles = activeCanvasLtx2Profiles();
        els.ltx2Profile.innerHTML = '';
        els.ltx2Resolution.innerHTML = '';
        els.ltx2Duration.value = '';
        if (els.ltx2DurationList)
            els.ltx2DurationList.innerHTML = '';
        if (!profiles.length) {
            var unavailable = document.createElement('option');
            unavailable.value = '';
            unavailable.textContent = 'No compiled LTX2 profiles available';
            els.ltx2Profile.appendChild(unavailable);
            var unavailableResolution = document.createElement('option');
            unavailableResolution.value = '';
            unavailableResolution.textContent = 'No compiled LTX2 resolutions available';
            els.ltx2Resolution.appendChild(unavailableResolution);
            els.ltx2Duration.placeholder = 'No compiled LTX2 seconds available';
            els.ltx2Profile.disabled = true;
            els.ltx2Resolution.disabled = true;
            els.ltx2Duration.disabled = true;
            if (els.ltx2ProfileNote)
                els.ltx2ProfileNote.textContent = 'No executable native LTX2 profile runner was reported by the server.';
            setCanvasLtx2GeometryLocked(false);
            return;
        }
        profiles.forEach(function (profile) {
            var option = document.createElement('option');
            option.value = canvasLtx2ProfileKey(profile);
            option.textContent = String(profile.label || 'LTX2') + ' \u00b7 ' +
                profile.width + '\u00d7' + profile.height + ' \u00b7 ' +
                profile.frames + 'f @ ' + profile.fps + ' FPS \u00b7 ' +
                Number(profile.duration) + 's';
            els.ltx2Profile.appendChild(option);
        });
        var seenResolutions = {};
        profiles.forEach(function (profile) {
            var key = canvasLtx2ResolutionKey(profile);
            if (seenResolutions[key])
                return;
            seenResolutions[key] = true;
            var option = document.createElement('option');
            option.value = key;
            option.textContent = String(profile.label || 'LTX2') + ' \u00b7 ' +
                profile.width + '\u00d7' + profile.height + ' \u00b7 ' +
                profile.fps + ' FPS native';
            els.ltx2Resolution.appendChild(option);
        });
        els.ltx2Profile.disabled = false;
        els.ltx2Resolution.disabled = false;
        refreshCanvasLtx2QuantControls();
        applyCanvasLtx2Profile(preferredCanvasLtx2Profile());
        setCanvasLtx2GeometryLocked(true);
        [els.framesInput, els.framesRange, els.fpsInput, els.fpsRange].forEach(function (control) {
            if (!control)
                return;
            control.disabled = true;
            control.title = 'Choose Native resolution and Duration; Canvas resolves the exact frame count and FPS.';
        });
    }
    function applyCanvasLtx2GuidanceMode() {
        var mode = activeCanvasLtx2RequestMode();
        var modes = mode && mode.guidance_modes || {};
        var workflow = canvasLtx2CheckpointWorkflow();
        if (workflow && genState.editMode !== 'retake_ltx23' &&
            genState.editMode !== 'extend_ltx23') {
            genState.ltx2Mode = String(workflow.guidance_mode || 'distilled');
            genState.steps = Number(workflow.steps) || 8;
            genState.sampler = String(workflow.sampler || 'euler');
            genState.scheduler = String(workflow.scheduler || 'ltx2_distilled');
            if (els.ltx2Mode) {
                els.ltx2Mode.value = genState.ltx2Mode;
                els.ltx2Mode.disabled = true;
                els.ltx2Mode.title = 'The selected checkpoint creator workflow owns guidance and sampling.';
            }
            els.steps.value = String(genState.steps);
            els.stepsRange.value = String(genState.steps);
            els.sampler.value = genState.sampler;
            els.scheduler.value = genState.scheduler;
            return;
        }
        if (els.ltx2Mode) {
            els.ltx2Mode.disabled = false;
            els.ltx2Mode.title = '';
        }
        var config = modes[genState.ltx2Mode] || {};
        if (genState.ltx2Mode === 'distilled') {
            genState.steps = Number(config.steps) || 8;
            genState.sampler = String(config.sampler || 'euler');
            genState.scheduler = String(config.scheduler || 'ltx2_distilled');
        }
        else {
            genState.steps = Number(config.default_steps) || 15;
            genState.sampler = String(config.sampler || 'res2s');
            genState.scheduler = String(config.scheduler || 'ltx2');
        }
        els.steps.value = String(genState.steps);
        els.stepsRange.value = String(genState.steps);
        els.sampler.value = genState.sampler;
        els.scheduler.value = genState.scheduler;
    }
    function loadModels() {
        var videoReadiness = fetch('/v1/video', { cache: 'no-store' })
            .then(function (response) {
                if (!response.ok)
                    throw new Error('video readiness HTTP ' + response.status);
                return response.json();
            })
            .catch(function () { return null; });
        Promise.all([ModelUtils.fetchAllModels(), ModelUtils.loadCapabilities(), videoReadiness])
            .then(function (loaded) {
            var models = loaded[0];
            canvasCapabilities = loaded[1];
            canvasVideoStatus = loaded[2];
            models = models.filter(function (model) {
                return ModelUtils.archForModel(model.name) !== 'scail2';
            });
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
            updateCanvasUIForArch(ModelUtils.archForModel(pick.name));
            populateEditModelEngineOptions();
            populateMaskedEditEngineOptions();
            if (genState.editMode === 'flowedit' || genState.editMode === 'style')
                syncCanvasModelFromFlowEditEngine();
            else if (genState.editMode === 'edit_models')
                syncCanvasModelFromEditModelEngine();
            else if (genState.editMode === 'inpaint')
                syncCanvasModelFromLanPaintEngine();
            else if (genState.editMode === 'i2v_ltx23')
                selectCanvasLtx23Model();
            else if (genState.editMode === 'retake_ltx23' ||
                genState.editMode === 'extend_ltx23')
                selectCanvasLtx23Model();
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
        var previousArch = genState.arch;
        genState.arch = arch;
        var isFlux = arch === 'flux';
        var isVideo = arch === 'ltxv' || arch === 'minimax_h3' || arch === 'wan' || arch === 'bernini';
        var isLtx2 = arch === 'ltxv';
        var isH3 = arch === 'minimax_h3';
        els.cfgRow.style.display = (isFlux || isVideo) ? 'none' : 'flex';
        els.guidanceRow.style.display = isFlux ? 'flex' : 'none';
        els.videoSection.style.display = isVideo ? 'block' : 'none';
        els.denoiseRow.style.display = isVideo ? 'none' : 'flex';
        els.denoiseHelp.style.display = isVideo ? 'none' : 'block';
        els.ltx2Section.style.display = isLtx2 ? 'flex' : 'none';
        els.h3Section.style.display = isH3 ? 'flex' : 'none';
        els.batch.style.display = isVideo ? 'none' : '';
        els.batchLabel.style.display = isVideo ? 'none' : '';
        if (isLtx2) {
            var checkpointWorkflow = canvasLtx2CheckpointWorkflow();
            var samplerOptions = [
                { value: 'res2s', label: 'Res2S' },
                { value: 'euler', label: 'Euler (fast distilled)' }
            ];
            var schedulerOptions = [
                { value: 'ltx2', label: 'LTX2' },
                { value: 'ltx2_distilled', label: 'LTX2 Distilled (8 steps)' }
            ];
            if (checkpointWorkflow) {
                if (!samplerOptions.some(function (row) {
                    return row.value === checkpointWorkflow.sampler;
                })) {
                    samplerOptions.push({
                        value: String(checkpointWorkflow.sampler),
                        label: 'Creator · ' + String(checkpointWorkflow.sampler)
                    });
                }
                if (!schedulerOptions.some(function (row) {
                    return row.value === checkpointWorkflow.scheduler;
                })) {
                    schedulerOptions.push({
                        value: String(checkpointWorkflow.scheduler),
                        label: 'Creator · ' + String(checkpointWorkflow.scheduler)
                    });
                }
            }
            setCanvasSelectOptions(
                els.sampler, samplerOptions,
                checkpointWorkflow ? String(checkpointWorkflow.sampler) :
                    (genState.ltx2Mode === 'distilled' ? 'euler' : 'res2s')
            );
            setCanvasSelectOptions(
                els.scheduler, schedulerOptions,
                checkpointWorkflow ? String(checkpointWorkflow.scheduler) :
                    (genState.ltx2Mode === 'distilled'
                        ? 'ltx2_distilled' : 'ltx2')
            );
            refreshCanvasLtx2ProfileControls();
            refreshCanvasLtx2FeatureControls();
            refreshCanvasLtx2PostUpscaleControls();
            applyCanvasLtx2GuidanceMode();
        }
        else if (isH3) {
            setCanvasSelectOptions(els.sampler, [
                { value: 'flowmatch_euler', label: 'H3 flow scheduler' }
            ], 'flowmatch_euler', false);
            setCanvasSelectOptions(els.scheduler, [
                { value: 'h3_flow', label: 'H3 native schedule' }
            ], 'h3_flow', false);
            refreshCanvasH3Controls();
        }
        else {
            setCanvasLtx2GeometryLocked(false);
            [els.steps, els.stepsRange].forEach(function (control) {
                control.disabled = false;
                control.title = '';
            });
            [els.framesInput, els.framesRange, els.fpsInput, els.fpsRange].forEach(function (control) {
                if (!control)
                    return;
                control.disabled = false;
                control.title = '';
            });
            var profile = canvasBackendProfile();
            var samplerNames = profile && profile.samplers &&
                Array.isArray(profile.samplers.supported_samplers)
                ? profile.samplers.supported_samplers : ['euler'];
            var schedulerNames = profile && profile.samplers &&
                Array.isArray(profile.samplers.supported_schedulers)
                ? profile.samplers.supported_schedulers : ['simple'];
            var defaults = profile && profile.defaults || {};
            setCanvasSelectOptions(els.sampler, samplerNames.map(function (name) {
                return { value: name, label: canvasSamplerLabel(name) };
            }), defaults.sampler || samplerNames[0] || 'euler', previousArch === arch);
            setCanvasSelectOptions(els.scheduler, schedulerNames.map(function (name) {
                return { value: name, label: canvasSchedulerLabel(name) };
            }), defaults.scheduler || schedulerNames[0] || 'simple', previousArch === arch);
        }
        // Update generate button label
        els.generateBtn.textContent = genState.editMode === 'style' ? 'Apply reference style' :
            (genState.editMode === 'flowedit' ? 'Run FlowEdit' :
                (genState.editMode === 'edit_models' ? 'Generate Edit' :
                    (genState.editMode === 'inpaint' ? 'Edit masked area' :
                        (genState.editMode === 'i2v_ltx23' ? 'Generate I2V' :
                            (genState.editMode === 'retake_ltx23' ? 'Retake selected window' :
                                (genState.editMode === 'extend_ltx23' ? 'Extend video' :
                            (isH3 ? 'Generate H3 Video + Audio' :
                                (isVideo ? 'Generate Video' : 'Generate'))))))));
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
        updateEditWorkspace();
    }
    function canvasSamplerLabel(name) {
        var labels = {
            euler: 'Euler',
            flowmatch_euler: 'FlowMatch Euler',
            ddim: 'DDIM',
            dpmpp_2m: 'DPM++ 2M',
            uni_pc: 'UniPC',
            uni_pc_bh2: 'UniPC BH2'
        };
        return labels[name] || String(name || '').replace(/_/g, ' ');
    }
    function canvasSchedulerLabel(name) {
        var labels = {
            normal: 'Normal',
            karras: 'Karras',
            exponential: 'Exponential',
            simple: 'Simple',
            ddim_uniform: 'DDIM Uniform',
            sgm_uniform: 'SGM Uniform',
            beta: 'Beta',
            linear_quadratic: 'Linear Quadratic',
            kl_optimal: 'KL Optimal',
            ideogram_logitnormal: 'Ideogram Logit-Normal',
            flux2: 'FLUX.2'
        };
        return labels[name] || String(name || '').replace(/_/g, ' ');
    }
    function setCanvasSelectOptions(select, options, fallback, preservePrior) {
        var prior = select.value;
        select.innerHTML = '';
        options.forEach(function (item) {
            var option = document.createElement('option');
            option.value = item.value;
            option.textContent = item.label;
            select.appendChild(option);
        });
        select.value = preservePrior !== false &&
            options.some(function (item) { return item.value === prior; }) ? prior : fallback;
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
        if (genState.editMode === 'edit_models') {
            startEditModelGeneration();
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
        if (genState.editMode === 'i2v_ltx23' && genState.arch !== 'ltxv') {
            showError('I2V - LTX 2.3 requires an installed LTX 2.3 model');
            return;
        }
        var temporalLtx2Mode = genState.editMode === 'retake_ltx23' ||
            genState.editMode === 'extend_ltx23';
        if (temporalLtx2Mode && genState.arch !== 'ltxv') {
            showError('Retake and Extend require an installed LTX 2.3 model');
            return;
        }
        if (genState.arch === 'ltxv') {
            if (!exactCanvasLtx2Profile()) {
                showError('Choose a supported Native video profile. LTX2 resolution, frame count, and FPS must match one available compiled Mojo runner.');
                return;
            }
            var checkpointWorkflow = canvasLtx2CheckpointWorkflow();
            var validLtx2Sampler = checkpointWorkflow
                ? genState.ltx2Mode === checkpointWorkflow.guidance_mode &&
                    genState.sampler === checkpointWorkflow.sampler &&
                    genState.scheduler === checkpointWorkflow.scheduler &&
                    genState.steps === Number(checkpointWorkflow.steps)
                : (genState.ltx2Mode === 'distilled'
                    ? genState.sampler === 'euler' &&
                        genState.scheduler === 'ltx2_distilled' &&
                        genState.steps === 8
                    : genState.sampler === 'res2s' &&
                        genState.scheduler === 'ltx2' &&
                        genState.steps >= 1 && genState.steps <= 20);
            if (!validLtx2Sampler) {
                showError(checkpointWorkflow
                    ? 'The selected LTX2 checkpoint requires its registered creator sampler, scheduler, guidance mode, and step count'
                    : (genState.ltx2Mode === 'distilled'
                        ? 'Fast distilled LTX2 requires Euler, the LTX2 Distilled scheduler, and 8 steps'
                        : 'Dev LTX2 requires Res2S, the LTX2 scheduler, and 1–20 steps'));
                return;
            }
        }
        if (genState.arch === 'minimax_h3') {
            if (genState.h3Mode === 't2va') {
                if (!exactCanvasH3Profile()) {
                    showError('Choose an admitted H3 Resolution and Seconds profile before generating.');
                    return;
                }
            }
            else {
                var h3ModeProfile = activeCanvasH3Mode();
                if (!h3ModeProfile || !boundingBox ||
                    Math.round(boundingBox.width()) !== Number(h3ModeProfile.width) ||
                    Math.round(boundingBox.height()) !== Number(h3ModeProfile.height) ||
                    Number(genState.frames) !== Number(h3ModeProfile.frames) ||
                    Number(genState.fps) !== Number(h3ModeProfile.fps)) {
                    showError('The selected H3 conditioned feature requires its exact admitted geometry.');
                    return;
                }
            }
        }
        setCanvasGenerating(true);
        var isVideo = isVideoArch();
        var hasLtx2VideoSource = genState.arch === 'ltxv' &&
            editSourceIsVideo && !!editSourceFile;
        var activeLtx2Feature = genState.arch === 'ltxv' &&
            genState.ltx2FeatureId !== 'standard'
            ? canvasLtx2Feature(genState.ltx2FeatureId) : null;
        if (genState.arch === 'ltxv' &&
            genState.ltx2FeatureId !== 'standard' &&
            !admittedCanvasLtx2Feature(activeLtx2Feature)) {
            showError('The selected LTX2 feature workflow is not runtime-admitted');
            setCanvasGenerating(false);
            return;
        }
        if (activeLtx2Feature && activeLtx2Feature.id === 'cinemagraph' &&
            genState.prompt.indexOf('CINEMAGRAPH_MOTION') < 0) {
            showError('Cinemagraph requires the exact prompt trigger CINEMAGRAPH_MOTION');
            setCanvasGenerating(false);
            return;
        }
        if (activeLtx2Feature && activeLtx2Feature.id === 'foley-v2a' &&
            (!hasLtx2VideoSource || genState.ltx2SourceStrength !== 1 ||
                genState.ltx2AudioPolicy !== 'generate')) {
            showError('Foley requires a loaded V2V source, Source strength 1.00, and Audio Generate');
            setCanvasGenerating(false);
            return;
        }
        if (genState.arch === 'ltxv' &&
            genState.ltx2AudioPolicy === 'preserve' &&
            !hasLtx2VideoSource) {
            showError('Preserve source audio requires a loaded V2V source video with audio');
            setCanvasGenerating(false);
            return;
        }
        var maskedEditEngine = genState.editMode === 'inpaint' ? maskedEditEngineDefinition(genState.lanpaintEngine) : null;
        checkBboxContent().then(function (hasContent) {
            var maskLayerInfo = getMaskLayer();
            var hasMask = maskLayerInfo && maskLayerInfo.konvaLayer.getChildren().length > 0;
            // A source chosen through "Load source image" is authoritative for
            // LTX I2V. It may have been added while a mask layer was active, in
            // which case the mask-free Canvas composite intentionally excludes
            // it. Keep the original selected image separate from that composite.
            var hasExactLtx2ImageSource = genState.arch === 'ltxv' &&
                !!editSourceDataUrl && !editSourceIsVideo;
            var hasExactH3ImageSource = genState.arch === 'minimax_h3' &&
                !!editSourceDataUrl && !editSourceIsVideo;
            var h3NeedsFirst = genState.arch === 'minimax_h3' &&
                (genState.h3Mode === 'i2va' || genState.h3Mode === 'fl2va' ||
                    genState.h3Mode === 'ref2va');
            var h3NeedsLast = genState.arch === 'minimax_h3' &&
                (genState.h3Mode === 'l2va' || genState.h3Mode === 'fl2va');
            if (h3NeedsFirst && (genState.h3Mode === 'ref2va'
                ? !hasExactH3ImageSource
                : (!hasContent && !hasExactH3ImageSource))) {
                showError('MiniMax-H3 ' + genState.h3Mode.toUpperCase() +
                    (genState.h3Mode === 'ref2va'
                        ? ' requires a loaded Source image reference'
                        : ' requires a loaded image or composed Canvas first frame'));
                setCanvasGenerating(false);
                return;
            }
            if (h3NeedsLast && !h3LastFrameFile) {
                showError('MiniMax-H3 ' + genState.h3Mode.toUpperCase() +
                    ' requires a selected final keyframe');
                setCanvasGenerating(false);
                return;
            }
            if (genState.arch === 'minimax_h3' && hasMask) {
                showError('MiniMax-H3 keyframe generation does not accept a painted mask');
                setCanvasGenerating(false);
                return;
            }
            if (activeLtx2Feature && activeLtx2Feature.id === 'cinemagraph' &&
                ((!hasContent && !hasExactLtx2ImageSource) || hasLtx2VideoSource)) {
                showError('Cinemagraph requires one source image, not T2V or V2V');
                setCanvasGenerating(false);
                return;
            }
            if (activeLtx2Feature && activeLtx2Feature.id === 'foley-v2a' && hasMask) {
                showError('Foley freezes the complete source video and cannot use a painted mask');
                setCanvasGenerating(false);
                return;
            }
            if (genState.editMode === 'i2v_ltx23' && !hasExactLtx2ImageSource) {
                showError('Load one source image into I2V - LTX 2.3 before generating');
                setCanvasGenerating(false);
                return;
            }
            if (temporalLtx2Mode && (!hasLtx2VideoSource ||
                !editSourceVideoProbe || !editSourceUploadedPath)) {
                showError('Load a source video and wait for its native profile probe before ' +
                    (genState.editMode === 'retake_ltx23' ? 'Retake' : 'Extend'));
                setCanvasGenerating(false);
                return;
            }
            if (temporalLtx2Mode && hasMask) {
                showError('Retake and Extend use a time window and cannot also use a painted spatial mask');
                setCanvasGenerating(false);
                return;
            }
            if (genState.editMode === 'retake_ltx23') {
                var sourceDuration = (Number(editSourceVideoProbe.frame_count) - 1) /
                    Number(editSourceVideoProbe.fps);
                if (genState.ltx2RetakeDuration < 2 ||
                    genState.ltx2RetakeStart < 0 ||
                    genState.ltx2RetakeStart + genState.ltx2RetakeDuration > sourceDuration + 0.001) {
                    showError('Retake window must be at least 2 seconds and stay within the source clip (' +
                        sourceDuration.toFixed(2) + ' seconds)');
                    setCanvasGenerating(false);
                    return;
                }
            }
            if (genState.editMode === 'inpaint' && !hasContent) {
                showError('Load or select an image before starting the masked edit');
                setCanvasGenerating(false);
                return;
            }
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
            // Create/canvas is text-to-image. Existing result layers are output
            // placement context, not an implicit img2img request. Explicit
            // source-consuming modes (FlowEdit/style/inpaint) own editing.
            var consumesCanvasSource = genState.editMode === 'inpaint' ||
                (genState.arch === 'ltxv' &&
                    (hasContent || hasExactLtx2ImageSource || hasLtx2VideoSource)) ||
                (genState.arch === 'minimax_h3' && h3NeedsFirst &&
                    (hasContent || hasExactH3ImageSource)) ||
                (genState.arch === 'wan' && hasContent);
            var featureError = validateCanvasGenerationFeatures(
                consumesCanvasSource && (hasContent || hasExactLtx2ImageSource || hasExactH3ImageSource),
                consumesCanvasSource && hasMask
            );
            if (featureError) {
                showError(featureError);
                setCanvasGenerating(false);
                return;
            }
            var seed = genState.seed === -1 ? Math.floor(Math.random() * 4294967296) : genState.seed;
            var bw = isVideo ? ModelUtils.clampVideoDimension(boundingBox.width()) : ModelUtils.clampDimension(boundingBox.width());
            var bh = isVideo ? ModelUtils.clampVideoDimension(boundingBox.height()) : ModelUtils.clampDimension(boundingBox.height());
            if (genState.arch === 'wan') {
                var wanI2v = hasContent;
                if (wanI2v) {
                    var wanCreatorSize = canvasWanCreatorI2vSize(
                        boundingBox.width(), boundingBox.height());
                    bw = wanCreatorSize.width;
                    bh = wanCreatorSize.height;
                }
                else {
                    bw = 1280;
                    bh = 704;
                }
                genState.width = bw;
                genState.height = bh;
                genState.frames = 121;
                genState.fps = 24;
                genState.steps = 50;
            }
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
                ltx2AudioPolicy: genState.ltx2AudioPolicy,
                ltx2FeatureId: genState.ltx2FeatureId,
                ltx2FeatureWeight: genState.ltx2FeatureWeight,
                ltx2PostUpscaler: genState.ltx2PostUpscaler,
                ltx2PostUpscaleFactor: genState.ltx2PostUpscaleFactor,
                ltx2Quant: genState.ltx2Quant,
                ltx2CameraMotion: genState.ltx2CameraMotion,
                ltx2LastFrame: ltx2LastFrameFile
                    ? ltx2LastFrameFile.name : '',
                h3Mode: genState.h3Mode,
                h3Quant: genState.h3Quant,
                h3AttentionBackend: genState.h3AttentionBackend,
                h3LastFrame: h3LastFrameFile ? h3LastFrameFile.name : '',
                ltx2RetakeStart: genState.ltx2RetakeStart,
                ltx2RetakeDuration: genState.ltx2RetakeDuration,
                ltx2ExtendDirection: genState.ltx2ExtendDirection,
                ltx2ExtendSeconds: genState.ltx2ExtendSeconds,
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
                    contextExpand: genState.lanpaintContextExpand,
                    earlyStop: genState.lanpaintEarlyStop,
                    innerThreshold: genState.lanpaintInnerThreshold,
                    innerPatience: genState.lanpaintInnerPatience
                } : null,
                submittedAt: new Date().toISOString()
            };
            var ltx2V2V = genState.arch === 'ltxv' && isVideo && hasLtx2VideoSource;
            var ltx2I2V = genState.arch === 'ltxv' && isVideo &&
                (hasExactLtx2ImageSource || hasContent) && !ltx2V2V;
            var ltx2CheckpointWorkflow = genState.arch === 'ltxv'
                ? canvasLtx2CheckpointWorkflow() : null;
            var ltx2WorkflowProfile = ltx2CheckpointWorkflow
                ? String(ltx2CheckpointWorkflow.id || '') : '';
            if (genState.arch === 'minimax_h3') {
                var h3SourceUpload = h3NeedsFirst
                    ? (hasExactH3ImageSource
                        ? h3SquareKeyframePngBase64(editSourceDataUrl).then(uploadInitImage)
                        : (genState.h3Mode === 'ref2va'
                            ? Promise.reject(new Error('Ref2VA requires one selected Source image'))
                            : exportBoundingBoxRegion().then(uploadInitImage)))
                    : Promise.resolve('');
                var h3LastUpload = h3NeedsLast
                    ? (h3LastFrameUploadedPath
                        ? Promise.resolve(h3LastFrameUploadedPath)
                        : (h3LastFrameUploadPromise || imageFileAsDataUrl(h3LastFrameFile)
                            .then(h3SquareKeyframePngBase64)
                            .then(uploadInitImage)))
                    : Promise.resolve('');
                Promise.all([h3SourceUpload, h3LastUpload]).then(function (images) {
                    queueWorkflow(WorkflowBuilder.build({
                        model: genState.model || '',
                        prompt: genState.prompt,
                        width: genState.width,
                        height: genState.height,
                        steps: genState.steps,
                        seed: seed,
                        frames: genState.frames,
                        fps: genState.fps,
                        includeAudio: true,
                        quantization: genState.h3Quant,
                        h3AttentionBackend: genState.h3AttentionBackend,
                        h3Task: genState.h3Mode,
                        initImageName: images[0],
                        lastImageName: images[1]
                    }));
                }).catch(function (error) {
                    showError('MiniMax-H3 keyframe upload failed: ' + error.message);
                    setCanvasGenerating(false);
                });
            }
            else if (ltx2V2V) {
                var sourceUpload = editSourceUploadedPath
                    ? Promise.resolve(editSourceUploadedPath)
                    : (editSourceUploadPromise || SerenityAPI.uploadMedia(editSourceFile));
                Promise.all([
                    sourceUpload,
                    hasMask ? exportMaskAsBW().then(function (mask) {
                        if (!mask || !mask.base64)
                            throw new Error('The painted V2V mask could not be exported');
                        return uploadInitImage(mask.base64);
                    }) : Promise.resolve('')
                ]).then(function (uploaded) {
                    var videoName = uploaded[0];
                    var videoMaskName = uploaded[1];
                    queueWorkflow(WorkflowBuilder.build({
                        model: genState.model || '', prompt: genState.prompt,
                        initVideoName: videoName,
                        videoMaskName: videoMaskName,
                        videoStrength: genState.editMode === 'retake_ltx23' ||
                            genState.editMode === 'extend_ltx23'
                            ? 0 : genState.ltx2SourceStrength,
                        width: bw, height: bh,
                        steps: genState.steps, cfg: genState.cfg,
                        guidance: genState.guidance, seed: seed,
                        negPrompt: genState.negative, sampler: genState.sampler, scheduler: genState.scheduler, batch: genState.batch,
                        frames: genState.frames, fps: genState.fps, loras: activeLoras,
                        capsPositive: genState.capsPositive, capsNegative: genState.capsNegative,
                        noiseFixture: genState.noiseFixture, includeAudio: genState.includeAudio,
                        ltx2AudioPolicy: genState.ltx2AudioPolicy,
                        ltx2FeatureId: genState.ltx2FeatureId,
                        ltx2FeatureWeight: genState.ltx2FeatureWeight,
                        ltx2PostUpscaler: genState.ltx2PostUpscaler,
                        ltx2PostUpscaleFactor: genState.ltx2PostUpscaleFactor,
                        ltx2Mode: genState.ltx2Mode,
                        ltx2WorkflowProfile: ltx2WorkflowProfile,
                        ltx2PromptEnhancer: 'none',
                        ltx2CameraMotion: genState.ltx2CameraMotion,
                        quantization: genState.ltx2Quant,
                        ltx2VideoEditMode: genState.editMode === 'retake_ltx23'
                            ? 'retake'
                            : (genState.editMode === 'extend_ltx23'
                                ? 'extend_' + genState.ltx2ExtendDirection
                                : 'standard'),
                        ltx2VideoEditStart: genState.editMode === 'retake_ltx23'
                            ? genState.ltx2RetakeStart : 0,
                        ltx2VideoEditEnd: genState.editMode === 'retake_ltx23'
                            ? genState.ltx2RetakeStart + genState.ltx2RetakeDuration : 0
                    }));
                }).catch(function (err) {
                    showError('V2V source upload failed: ' + err.message);
                    setCanvasGenerating(false);
                });
            }
            else if ((genState.editMode === 'create' && !ltx2I2V) ||
                (!hasContent && !hasExactLtx2ImageSource)) {
                queueWorkflow(WorkflowBuilder.build({
                    model: genState.model || '', prompt: genState.prompt,
                    width: bw, height: bh,
                    steps: genState.steps, cfg: genState.cfg,
                    guidance: genState.guidance, seed: seed,
                    negPrompt: genState.negative, sampler: genState.sampler, scheduler: genState.scheduler, batch: genState.batch,
                    frames: genState.frames, fps: genState.fps, loras: activeLoras,
                    capsPositive: genState.capsPositive, capsNegative: genState.capsNegative,
                    noiseFixture: genState.noiseFixture, includeAudio: genState.includeAudio,
                    ltx2AudioPolicy: genState.ltx2AudioPolicy,
                    ltx2FeatureId: genState.ltx2FeatureId,
                    ltx2FeatureWeight: genState.ltx2FeatureWeight,
                    ltx2PostUpscaler: genState.ltx2PostUpscaler,
                    ltx2PostUpscaleFactor: genState.ltx2PostUpscaleFactor,
                    ltx2Mode: genState.ltx2Mode,
                    ltx2WorkflowProfile: ltx2WorkflowProfile,
                    ltx2PromptEnhancer: 'none',
                    ltx2CameraMotion: genState.ltx2CameraMotion,
                    quantization: genState.ltx2Quant
                }));
            }
            else if (isVideo) {
                var exactImageUpload = hasExactLtx2ImageSource &&
                    (editSourceUploadedPath || editSourceUploadPromise)
                    ? (editSourceUploadedPath
                        ? Promise.resolve(editSourceUploadedPath)
                        : editSourceUploadPromise)
                    : null;
                var imageUpload = exactImageUpload || (
                    hasExactLtx2ImageSource
                        ? imageDataUrlAsPngBase64(editSourceDataUrl).then(function (base64) {
                            return uploadInitImage(base64);
                        })
                        : exportBoundingBoxRegion().then(function (base64) {
                            return uploadInitImage(base64);
                        })
                );
                var lastImageUpload = ltx2LastFrameFile
                    ? (ltx2LastFrameUploadedPath
                        ? Promise.resolve(ltx2LastFrameUploadedPath)
                        : (ltx2LastFrameUploadPromise ||
                            SerenityAPI.uploadMedia(ltx2LastFrameFile)))
                    : Promise.resolve('');
                Promise.all([imageUpload, lastImageUpload]).then(function (images) {
                    var imageName = images[0];
                    var lastImageName = images[1];
                    queueWorkflow(WorkflowBuilder.build({
                        model: genState.model || '', prompt: genState.prompt,
                        initImageName: imageName,
                        imageStrength: genState.ltx2SourceStrength,
                        lastImageName: lastImageName,
                        lastImageStrength: lastImageName ? 1.0 : undefined,
                        width: bw, height: bh,
                        steps: genState.steps, cfg: genState.cfg,
                        guidance: genState.guidance, seed: seed,
                        negPrompt: genState.negative, sampler: genState.sampler, scheduler: genState.scheduler, batch: genState.batch,
                        frames: genState.frames, fps: genState.fps, loras: activeLoras,
                        capsPositive: genState.capsPositive, capsNegative: genState.capsNegative,
                        noiseFixture: genState.noiseFixture, includeAudio: genState.includeAudio,
                        ltx2AudioPolicy: genState.ltx2AudioPolicy,
                        ltx2FeatureId: genState.ltx2FeatureId,
                        ltx2FeatureWeight: genState.ltx2FeatureWeight,
                        ltx2PostUpscaler: genState.ltx2PostUpscaler,
                        ltx2PostUpscaleFactor: genState.ltx2PostUpscaleFactor,
                        ltx2Mode: genState.ltx2Mode,
                        ltx2WorkflowProfile: ltx2WorkflowProfile,
                        ltx2PromptEnhancer: 'none',
                        ltx2CameraMotion: genState.ltx2CameraMotion,
                        quantization: genState.ltx2Quant
                    }));
                }).catch(function (err) {
                    showError('Video keyframe upload failed: ' + err.message);
                    setCanvasGenerating(false);
                });
            }
            else if (hasMask && !isVideo) {
                // Inpaint: export both init image and mask
                exportBoundingBoxRegion().then(function (initBase64) {
                    return uploadInitImage(initBase64);
                }).then(function (initName) {
                    if (!maskedEditEngine || !maskedEditEngine.lanpaint) {
                        return {
                            initName: initName,
                            sourceDescription: '',
                            targetDescription: genState.prompt
                        };
                    }
                    if (typeof CanvasStatusBar !== 'undefined')
                        CanvasStatusBar.updateGenStatus('Describing masked target');
                    return prepareMaskedEditTargetCaption(initName, genState.prompt).then(function (captions) {
                        captions.initName = initName;
                        return captions;
                    });
                }).then(function (prepared) {
                    if (pendingCanvasMetadata) {
                        pendingCanvasMetadata.maskedEditInstruction = genState.prompt;
                        pendingCanvasMetadata.maskedEditSourcePrompt = prepared.sourceDescription;
                        pendingCanvasMetadata.maskedEditTargetPrompt = prepared.fullTargetDescription || prepared.targetDescription;
                        pendingCanvasMetadata.maskedEditRegionalPrompt = prepared.targetDescription;
                    }
                    if (typeof CanvasStatusBar !== 'undefined')
                        CanvasStatusBar.updateGenStatus('Preparing mask');
                    return exportMaskAsBW().then(function (maskExport) {
                            if (!maskExport || !maskExport.base64)
                                throw new Error('The painted mask could not be exported');
                            if (maskExport.coverage <= 0)
                                throw new Error('The painted mask is empty');
                            if (maskExport.coverage >= 0.995 && !maskExport.fullFrameRequested)
                                throw new Error('The mask unexpectedly covers the entire image; clear it and paint the edit area again');
                            return uploadInitImage(maskExport.base64).then(function (maskName) {
                                queueWorkflow(WorkflowBuilder.buildInpaint({
                                    model: genState.model || '', prompt: prepared.targetDescription,
                                    negPrompt: genState.negative, initImageName: prepared.initName, maskImageName: maskName,
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
                                    lanpaintContextExpand: genState.lanpaintContextExpand,
                                    lanpaintEarlyStop: genState.lanpaintEarlyStop,
                                    lanpaintInnerThreshold: genState.lanpaintInnerThreshold,
                                    lanpaintInnerPatience: genState.lanpaintInnerPatience,
                                    loras: activeLoras
                                }));
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
        }).catch(function (error) {
            if (!canvasGenerating)
                return;
            showError('Generation failed: ' + (error && error.message ? error.message : String(error)));
            setCanvasGenerating(false);
        });
    }
    function startEditModelGeneration() {
        if (!editSourceDataUrl || editSourceIsVideo) {
            showError('Load a source image for Edit Models');
            return;
        }
        if (!genState.prompt.trim()) {
            showError('Describe the requested change in Prompt');
            return;
        }
        var definition = editModelEngineDefinition(genState.editModelEngine);
        if (!definition || definition.implemented !== true || !editModelBackendAdmitted(definition)) {
            showError(definition ? editModelUnavailableReason(definition) : 'Choose an edit model');
            return;
        }
        if (!syncCanvasModelFromEditModelEngine()) {
            showError('The selected edit model is not installed');
            return;
        }
        var seed = genState.seed === -1 ? Math.floor(Math.random() * 4294967296) : genState.seed;
        setCanvasGenerating(true);
        if (typeof CanvasStatusBar !== 'undefined')
            CanvasStatusBar.updateGenStatus('Preparing source image');
        imageDataUrlAsPngBase64(editSourceDataUrl).then(function (base64) {
            return SerenityAPI.uploadImageDetails(base64, 'edit_model_source');
        }).then(function (upload) {
            pendingCanvasMetadata = {
                prompt: genState.prompt,
                model: genState.model,
                arch: genState.arch,
                editMode: 'edit_models',
                editEngine: definition.id,
                width: 1024,
                height: 1024,
                steps: genState.steps,
                cfg: genState.cfg,
                seed: seed,
                sourceImage: upload.path,
                submittedAt: new Date().toISOString()
            };
            if (typeof CanvasStatusBar !== 'undefined')
                CanvasStatusBar.updateGenStatus('Queueing ' + definition.label);
            queueWorkflow(WorkflowBuilder.buildEditModel({
                model: genState.model,
                prompt: genState.prompt,
                negPrompt: '',
                initImageName: upload.path,
                width: 1024,
                height: 1024,
                steps: genState.steps,
                cfg: genState.cfg,
                denoise: genState.denoise,
                seed: seed,
                batch: 1
            }));
        }).catch(function (error) {
            showError('Edit preparation failed: ' + error.message);
            setCanvasGenerating(false);
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
    function imageFileAsDataUrl(file) {
        return new Promise(function (resolve, reject) {
            var reader = new FileReader();
            reader.onload = function (event) {
                resolve(String(event.target.result || ''));
            };
            reader.onerror = function () {
                reject(new Error('Could not read the selected image'));
            };
            reader.readAsDataURL(file);
        });
    }
    function h3SquareKeyframePngBase64(dataUrl) {
        return new Promise(function (resolve, reject) {
            var image = new Image();
            image.onload = function () {
                var size = 768;
                var iw = image.naturalWidth || image.width;
                var ih = image.naturalHeight || image.height;
                var scale = Math.max(size / iw, size / ih);
                var dw = iw * scale;
                var dh = ih * scale;
                var canvas = document.createElement('canvas');
                canvas.width = size;
                canvas.height = size;
                var context = canvas.getContext('2d');
                context.drawImage(image, (size - dw) / 2, (size - dh) / 2, dw, dh);
                resolve(canvas.toDataURL('image/png').split(',')[1]);
            };
            image.onerror = function () {
                reject(new Error('Could not decode the selected H3 keyframe'));
            };
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
    function prepareMaskedEditTargetCaption(imagePath, editInstruction) {
        var instruction = String(editInstruction || '').trim();
        return captionUploadedImage(
            imagePath,
            'Analyze the visible source image and apply this requested change conceptually: ' + JSON.stringify(instruction) +
                '. The text may be a short edit command or a complete desired result; infer the intended final visible state. ' +
                'Return only valid compact JSON with three string fields: {"source":"complete source image description","target":"complete target image description","masked_target":"concise positive description of the desired content inside the painted region"}. ' +
                'Keep source and target parallel in detail and ordering. The complete target must preserve every unchanged subject, identity, pose, composition, framing, clothing, lighting, and background detail. ' +
                'For masked_target, state the changed subject and desired visible result first, then only the medium, lighting, and nearby context needed to render that region. Do not dilute it with unrelated background details. ' +
                'Express the requested change as a positive, present-tense visible state. Do not mention an edit, request, source image, instruction, removal, absence, negation, or use phrases such as "no" or "without". ' +
                'Return only the JSON. Keep source and target at most 110 words and masked_target at most 35 words.',
            320
        ).then(function (caption) {
            var text = String(caption || '').trim();
            var start = text.indexOf('{');
            var end = text.lastIndexOf('}');
            if (start >= 0 && end > start) {
                try {
                    var parsed = JSON.parse(text.slice(start, end + 1));
                    var source = String(parsed.source || '').trim();
                    var fullTarget = String(parsed.target || '').trim();
                    var maskedTarget = String(parsed.masked_target || '').trim();
                    if (source && fullTarget) {
                        return {
                            sourceDescription: source,
                            targetDescription: maskedTarget || fullTarget,
                            fullTargetDescription: fullTarget
                        };
                    }
                }
                catch (_) { }
            }
            throw new Error('The visual analyzer did not return the required masked-edit JSON');
        });
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
        if (genState.editMode === 'style' || genState.editMode === 'edit_models')
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
        var exactNativeEdit = genState.editMode === 'edit_models';
        var controlLayers = exactNativeEdit ? [] : collectControlLayers();
        var ipaLayers = exactNativeEdit ? [] : collectIPALayers();
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
            if (typeof WorkflowSync !== 'undefined' && WorkflowSync.stageWorkflow) {
                var modeNames = {
                    create: 'Create',
                    i2v_ltx23: 'I2V - LTX 2.3',
                    retake_ltx23: 'Retake - LTX 2.3',
                    extend_ltx23: 'Extend - LTX 2.3',
                    edit_models: 'Edit Models',
                    flowedit: 'FlowEdit',
                    style: 'Style Transfer',
                    inpaint: 'Masked Edit - LanPaint',
                    dynaedit: 'DynaEdit'
                };
                WorkflowSync.stageWorkflow(workflow, {
                    name: 'Canvas - ' + (modeNames[genState.editMode] || 'Workflow')
                });
            }
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
            if (result && result.video_pending && result.video_result)
                pollVideoJob(result.video_result);
        })
            .catch(function (err) {
            showError('Failed to queue: ' + err.message);
            pendingCanvasPromptId = '';
            setCanvasGenerating(false);
        });
    }
    function stagePolledVideo(videoId, manifest) {
        if (!canvasGenerating || pendingCanvasPromptId !== videoId)
            return;
        var artifact = String(manifest.artifact_path || '');
        var filename = artifact.split('/').pop();
        if (!filename)
            throw new Error('LTX2 result did not contain an artifact filename');
        canvasResultMetadata(videoId).then(function (metadata) {
            var result = {
                src: '/out/' + encodeURIComponent(videoId) + '/' + encodeURIComponent(filename),
                isVideo: true,
                filename: filename,
                metadata: JSON.parse(JSON.stringify(metadata))
            };
            hideCanvasPreview();
            if (typeof CanvasStaging !== 'undefined')
                CanvasStaging.activate([result], getToolContext());
            else
                showCanvasPreview(result.src, true);
            setCanvasGenerating(false);
            if (typeof CanvasStatusBar !== 'undefined')
                CanvasStatusBar.updateGenStatus('staging');
            pendingCanvasPromptId = '';
            pendingVideoPollToken++;
            loadCanvasGallery();
        });
    }
    function pollVideoJob(job) {
        var videoId = String(job.prompt_id || job.video_id || pendingCanvasPromptId || '');
        var statusUrl = String(job.status_url || ('/out/' + encodeURIComponent(videoId) + '/status.json'));
        var resultUrl = String(job.result_url || ('/out/' + encodeURIComponent(videoId) + '/result.json'));
        var token = ++pendingVideoPollToken;
        function poll() {
            if (token !== pendingVideoPollToken || !canvasGenerating || pendingCanvasPromptId !== videoId)
                return;
            fetch(statusUrl, { cache: 'no-store' }).then(function (response) {
                if (!response.ok)
                    throw new Error('status HTTP ' + response.status);
                return response.json();
            }).then(function (status) {
                if (token !== pendingVideoPollToken)
                    return;
                var step = Number(status.step) || 0;
                var total = Number(status.total) || 0;
                var phase = String(status.message || status.phase || 'LTX2 running');
                els.progressBar.style.width = (total > 0 ? Math.max(0, Math.min(100, step / total * 100)) : 4) + '%';
                els.progressLabel.textContent = phase + (total > 0 ? ' · Step ' + step + ' / ' + total : '');
                els.progressLabel.classList.add('visible');
                if (typeof CanvasStatusBar !== 'undefined')
                    CanvasStatusBar.updateGenStatus(phase);
                if (status.state === 'failed' || status.state === 'error')
                    throw new Error(phase);
                if (status.state !== 'done') {
                    setTimeout(poll, 500);
                    return;
                }
                return fetch(resultUrl, { cache: 'no-store' }).then(function (response) {
                    if (!response.ok)
                        throw new Error('result HTTP ' + response.status);
                    return response.json();
                }).then(function (manifest) {
                    stagePolledVideo(videoId, manifest);
                });
            }).catch(function (error) {
                if (token !== pendingVideoPollToken)
                    return;
                if (/HTTP 404/.test(error.message)) {
                    setTimeout(poll, 500);
                    return;
                }
                showError('Video generation failed: ' + error.message);
                pendingVideoPollToken++;
                pendingCanvasPromptId = '';
                setCanvasGenerating(false);
            });
        }
        setTimeout(poll, 250);
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
            else if (genState.editMode === 'edit_models')
                els.generateBtn.textContent = 'Generate Edit';
            else if (genState.editMode === 'inpaint')
                els.generateBtn.textContent = 'Inpaint masked area';
            else if (genState.editMode === 'i2v_ltx23')
                els.generateBtn.textContent = 'Generate I2V';
            else if (genState.editMode === 'retake_ltx23')
                els.generateBtn.textContent = 'Retake selected window';
            else if (genState.editMode === 'extend_ltx23')
                els.generateBtn.textContent = 'Extend video';
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
        clearCanvasVideoPreview();
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
        clearCanvasVideoPreview();
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
                pendingVideoPollToken++;
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
            pendingVideoPollToken++;
        });
        SerenityWS.on('execution_interrupted', function (data) {
            if (!isPendingCanvasEvent(data))
                return;
            setCanvasGenerating(false);
            if (typeof CanvasStatusBar !== 'undefined')
                CanvasStatusBar.updateGenStatus('interrupted');
            pendingCanvasPromptId = '';
            pendingVideoPollToken++;
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
        bindCanvasPanelMotion();
        restoreCanvasPanelOffsets();
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
            updateCanvasUIForArch(ModelUtils.archForModel(modelName));
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
