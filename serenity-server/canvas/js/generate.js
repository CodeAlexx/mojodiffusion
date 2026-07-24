"use strict";
/**
 * Generate Tab — SerenityFlow Phase 2
 * Prompt-to-image generation wired to ComfyUI-compatible backend.
 * SerenityFlow visual overhaul.
 */
var GenerateTab = (function () {
    'use strict';
    var state = {
        model: null,
        prompt: '',
        negPrompt: '',
        width: 1024,
        height: 1024,
        steps: 20,
        cfg: 7.0,
        guidance: 3.5,
        sampler: 'euler',
        scheduler: 'simple',
        seed: -1,
        variationSeed: 0,
        variationStrength: 0,
        initImagePath: '',
        initImageName: '',
        creativity: 0.5,
        generating: false,
        currentImage: null,
        currentIsVideo: false,
        currentVideoFrames: null,
        currentVideoFps: null,
        gallery: [],
        arch: 'sd15',
        frames: 121,
        fps: 24,
        seconds: 10,
        lastSeed: null,
        batchCount: 1,
        loras: [], // [{name, strength, enabled}]
        aspectLocked: false,
        lockedRatio: 1,
        // New state for overhaul
        leftPanelVisible: true,
        rightPanelVisible: true,
        galleryTab: 'gallery', // 'layers' | 'gallery'
        gallerySubTab: 'images', // 'images' | 'assets'
        boardsVisible: true,
        selectedAspect: '1:1',
        pendingBatch: 0,
        // Prompt history
        promptHistory: [],
        promptHistoryIndex: -1,
        // Style preset
        stylePreset: 'none',
        // Model picker search
        modelPickerOpen: false,
        modelSearchQuery: '',
        allModels: [],
        capabilities: null,
        // Phase 2: gallery enhancements
        selectedImages: [],
        lastSelectedIndex: -1,
        gallerySearch: '',
        thumbSize: 90,
        sortNewestFirst: true,
        starredFirst: false,
        autoSwitchNew: true,
        galleryPage: 0,
        galleryPageSize: 50,
        contextMenuIndex: -1,
        gallerySettingsOpen: false,
        metadataExpanded: false,
        // Phase 3: Advanced section
        vae: 'default',
        clipSkip: 0,
        cfgRescale: 0,
        seamlessX: false,
        seamlessY: false,
        // Phase 3: Refiner section (SDXL)
        refinerModel: '',
        refinerScheduler: 'euler',
        refinerSteps: 20,
        refinerCfg: 7.0,
        refinerStart: 0.8,
        aestheticScore: 6.0,
        negAestheticScore: 2.5,
        // Phase 3: Compositing section (inpaint)
        coherenceMode: 'gaussian',
        edgeSize: 32,
        minDenoise: 0.0,
        maskBlur: 4,
        infillMethod: 'patchmatch',
        referenceImagePath: '',
        referenceMaskPath: '',
        drivingVideoPath: '',
        drivingMaskVideoPath: '',
        scail2Mode: 'animation',
        additionalReferenceImagePaths: [],
        additionalReferenceMaskPaths: [],
        mediaUploadsInFlight: 0
    };
    var initialized = false;
    // DOM refs (set in init)
    var els = {};
    // Convert seconds + fps to frame count (round to nearest valid frame count)
    function secondsToFrames(seconds, fps) {
        // frames = seconds * fps + 1 (first frame is the start frame)
        return Math.max(9, Math.round(seconds * fps) + 1);
    }
    // ── Aspect ratio definitions ──
    var videoAspects = [
        { label: 'Free', w: 0, h: 0, vw: 16, vh: 16 },
        { label: '1:1', w: 768, h: 768, vw: 16, vh: 16 },
        { label: '4:3', w: 768, h: 576, vw: 18, vh: 14 },
        { label: '3:2', w: 768, h: 512, vw: 18, vh: 12 },
        { label: '16:9', w: 768, h: 432, vw: 20, vh: 11 },
        { label: '21:9', w: 768, h: 320, vw: 21, vh: 9 },
        { label: '3:4', w: 576, h: 768, vw: 14, vh: 18 },
        { label: '2:3', w: 512, h: 768, vw: 12, vh: 18 },
        { label: '9:16', w: 432, h: 768, vw: 11, vh: 20 },
        { label: '9:21', w: 320, h: 768, vw: 9, vh: 21 }
    ];
    // ── Style Presets ──
    var stylePresets = [
        { id: 'none', label: 'None', suffix: '' },
        { id: 'photo', label: 'Photo', suffix: ', photorealistic, high quality photo, detailed, sharp focus, natural lighting' },
        { id: 'anime', label: 'Anime', suffix: ', anime style, vibrant colors, detailed illustration, anime art' },
        { id: 'oil', label: 'Oil Paint', suffix: ', oil painting style, textured brushstrokes, rich colors, classical art' },
        { id: '3d', label: '3D Render', suffix: ', 3D render, octane render, detailed, volumetric lighting, CGI' },
        { id: 'cinematic', label: 'Cinematic', suffix: ', cinematic, dramatic lighting, film grain, anamorphic, movie still' },
        { id: 'watercolor', label: 'Watercolor', suffix: ', watercolor painting, soft colors, artistic, delicate brushwork' },
        { id: 'pixel', label: 'Pixel Art', suffix: ', pixel art, 8-bit, retro gaming style, low resolution aesthetic' },
        { id: 'sketch', label: 'Sketch', suffix: ', pencil sketch, hand drawn, detailed linework, graphite' },
        { id: 'fantasy', label: 'Fantasy', suffix: ', fantasy art, magical, ethereal, detailed illustration, concept art' },
        { id: 'neon', label: 'Neon', suffix: ', neon lights, cyberpunk, glowing, dark background, vibrant' },
        { id: 'minimal', label: 'Minimal', suffix: ', minimalist, clean, simple, modern design, white space' }
    ];
    // ── Scheduler/Sampler definitions ──
    // Only expose the sampler that every production worker currently admits.
    // Wider sampler inventory remains available through the capability API and
    // returns to this picker only after model-specific artifact gates pass.
    var schedulerOptions = [
        { value: 'euler', label: 'Euler' },
        { value: 'uni_pc', label: 'UniPC (Wan)' }
    ];
    function getActiveAspects() {
        var arch = ModelUtils.detectArchFromFilename(state.model);
        if (arch === 'ltxv')
            return [{ label: '1920×1088', w: 1920, h: 1088, vw: 30, vh: 17 }];
        if (arch === 'wan')
            return [{ label: '832×480', w: 832, h: 480, vw: 26, vh: 15 }];
        if (arch === 'bernini')
            return [{ label: '848×480', w: 848, h: 480, vw: 53, vh: 30 }];
        if (arch === 'scail2')
            return [{ label: '896×512', w: 896, h: 512, vw: 28, vh: 16 }];
        return ModelUtils.aspectsForArch(state.capabilities, arch);
    }
    function buildAspectOptions() {
        var aspects = getActiveAspects();
        var html = '';
        aspects.forEach(function (a) {
            html += '<option value="' + a.label + '">' + a.label + '</option>';
        });
        return html;
    }
    function buildStyleOptions() {
        var html = '';
        stylePresets.forEach(function (s) {
            html += '<option value="' + s.id + '">' + s.label + '</option>';
        });
        return html;
    }
    function buildSchedulerOptions() {
        var html = '';
        schedulerOptions.forEach(function (s) {
            html += '<option value="' + s.value + '">' + s.label + '</option>';
        });
        return html;
    }
    // ── Build DOM ──
    function buildUI() {
        var panel = document.getElementById('panel-generate');
        if (!panel)
            return;
        panel.innerHTML = '';
        var layout = document.createElement('div');
        layout.className = 'gen-layout gen-swarm-layout';
        // Left panel
        var left = document.createElement('div');
        left.className = 'gen-left gen-swarm-parameters';
        left.id = 'gen-left-panel';
        left.innerHTML = buildSwarmLeftHTML();
        layout.appendChild(left);
        // Center panel
        var center = document.createElement('div');
        center.className = 'gen-center gen-swarm-stage';
        center.innerHTML = buildSwarmTopToolbarHTML() + buildCenterHTML();
        layout.appendChild(center);
        // Floating side toolbar (inside center so it positions relative to center)
        var floatBar = document.createElement('div');
        floatBar.className = 'gen-floating-toolbar';
        floatBar.id = 'gen-floating-toolbar';
        floatBar.innerHTML = buildFloatingToolbarHTML();
        center.appendChild(floatBar);
        // Right panel
        var right = document.createElement('div');
        right.className = 'gen-right gen-swarm-batch-panel';
        right.id = 'gen-right-panel';
        right.innerHTML = buildSwarmBatchHTML();
        layout.appendChild(right);
        var promptDock = document.createElement('div');
        promptDock.className = 'gen-swarm-prompt-dock';
        promptDock.innerHTML = buildSwarmPromptHTML();
        layout.appendChild(promptDock);
        var library = document.createElement('div');
        library.className = 'gen-swarm-library';
        library.innerHTML = buildSwarmLibraryHTML();
        layout.appendChild(library);
        panel.appendChild(layout);
        cacheElements();
    }

    function swarmGroup(id, title, body, open, help) {
        return '<section class="gen-swarm-group gen-param-group" data-param-search="' +
            (title + ' ' + (help || '')).toLowerCase() + '">' +
            '<button type="button" id="' + id + '" class="gen-accordion-header gen-swarm-group-title' +
            (open ? '' : ' closed') + '">' +
            '<span><i data-lucide="chevron-down"></i>' + title + '</span>' +
            (help ? '<span class="gen-swarm-help" title="' + help.replace(/"/g, '&quot;') + '">?</span>' : '') +
            '</button>' +
            '<div id="' + id.replace('-header', '-body') + '" class="gen-accordion-body gen-swarm-group-body' +
            (open ? '' : ' closed') + '">' + body + '</div></section>';
    }

    function buildSwarmLeftHTML() {
        var modelBody =
            '<div class="gen-param-row" data-param-search="model checkpoint search">' +
            '<label class="gen-label" for="gen-model-search">Model</label>' +
            '<div class="gen-model-row"><div class="gen-model-picker-wrap" id="gen-model-picker-wrap">' +
            '<input type="text" id="gen-model-search" class="gen-select gen-model-search-input" placeholder="Loading image models..." autocomplete="off">' +
            '<div id="gen-model-dropdown" class="gen-model-dropdown" style="display:none"><div id="gen-model-dropdown-list" class="gen-model-dropdown-list"></div></div>' +
            '</div><button id="gen-model-refresh" class="gen-model-action-btn" title="Refresh local models"><i data-lucide="refresh-cw"></i></button></div>' +
            '<input type="hidden" id="gen-model" value=""><div id="gen-model-warn" class="gen-model-warning"></div></div>';
        var coreBody =
            '<div class="gen-setting-row gen-param-row" id="gen-batch-section" data-param-search="images batch count">' +
            '<label class="gen-setting-label" for="gen-batch">Images</label><input id="gen-batch" type="number" class="gen-number-input" min="1" max="8" value="1">' +
            '<span class="gen-batch-hint">queued as separate jobs</span></div>' +
            '<div class="gen-setting-row gen-param-row" data-param-search="seed random reuse">' +
            '<label class="gen-setting-label" for="gen-seed">Seed</label><input id="gen-seed" type="number" class="gen-number-input" value="-1">' +
            '<button id="gen-seed-shuffle" class="gen-seed-btn" title="Random seed">↻</button>' +
            '<button id="gen-seed-prev" class="gen-seed-btn" title="Reuse previous seed">↶</button>' +
            '<button id="gen-seed-random-toggle" class="gen-toggle on" title="Random seed"></button></div>' +
            '<div class="gen-setting-row gen-param-row" data-param-search="steps sampling iterations">' +
            '<label class="gen-setting-label" for="gen-steps">Steps</label><input type="range" id="gen-steps-range" class="gen-range" min="1" max="100" value="20">' +
            '<input type="number" id="gen-steps" class="gen-number-input" min="1" max="500" value="20"></div>' +
            '<div class="gen-setting-row gen-param-row" id="gen-cfg-row" data-param-search="cfg classifier free guidance">' +
            '<label class="gen-setting-label" for="gen-cfg">CFG Scale</label><input type="range" id="gen-cfg-range" class="gen-range" min="0" max="20" step="0.1" value="7">' +
            '<input type="number" id="gen-cfg" class="gen-number-input" min="0" max="200" step="0.1" value="7"></div>' +
            '<div class="gen-setting-row gen-param-row" id="gen-guidance-row" style="display:none" data-param-search="distilled guidance">' +
            '<label class="gen-setting-label" for="gen-guidance">Guidance</label><input type="range" id="gen-guidance-range" class="gen-range" min="0" max="20" step="0.1" value="3.5">' +
            '<input type="number" id="gen-guidance" class="gen-number-input" min="0" max="200" step="0.1" value="3.5"></div>';
        var variationBody =
            '<div class="gen-setting-row gen-param-row" data-param-search="variation seed">' +
            '<label class="gen-setting-label" for="gen-variation-seed">Variation Seed</label><input id="gen-variation-seed" type="number" class="gen-number-input" value="0">' +
            '<button id="gen-variation-shuffle" class="gen-seed-btn" title="Random variation seed">↻</button></div>' +
            '<div class="gen-setting-row gen-param-row" data-param-search="variation strength">' +
            '<label class="gen-setting-label" for="gen-variation-strength">Strength</label><input id="gen-variation-strength-range" type="range" class="gen-range" min="0" max="1" step="0.01" value="0">' +
            '<input id="gen-variation-strength" type="number" class="gen-number-input" min="0" max="1" step="0.01" value="0"></div>';
        var resolutionBody =
            '<div class="gen-param-row" data-param-search="aspect resolution size width height">' +
            '<div class="gen-aspect-row"><label class="gen-label" for="gen-aspect-dropdown">Aspect</label>' +
            '<select id="gen-aspect-dropdown" class="gen-select">' + buildAspectOptions() + '</select>' +
            '<button id="gen-swap-btn" class="gen-aspect-action" title="Swap admitted dimensions"><i data-lucide="arrow-left-right"></i></button>' +
            '<button id="gen-optimal-btn" class="gen-aspect-action" title="Restore model default"><i data-lucide="sparkles"></i></button>' +
            '<button id="gen-aspect-lock" class="gen-aspect-action" style="display:none" aria-hidden="true"></button></div>' +
            '<div class="gen-dim-row"><span class="gen-dim-label">Width</span><input type="range" id="gen-width-slider" class="gen-range" min="256" max="2048" step="64" value="1024" disabled>' +
            '<input type="number" id="gen-custom-width" class="gen-number-input" value="1024" disabled></div>' +
            '<div class="gen-dim-row"><span class="gen-dim-label">Height</span><input type="range" id="gen-height-slider" class="gen-range" min="256" max="2048" step="64" value="1024" disabled>' +
            '<input type="number" id="gen-custom-height" class="gen-number-input" value="1024" disabled></div>' +
            '<div id="gen-aspect-preview" class="gen-aspect-preview"><span>1024×1024</span></div></div>';
        var samplingBody =
            '<div class="gen-param-row" data-param-search="sampler algorithm"><label class="gen-label" for="gen-sampler">Sampler</label><select id="gen-sampler" class="gen-select"></select></div>' +
            '<div class="gen-param-row" data-param-search="scheduler noise schedule"><label class="gen-label" for="gen-scheduler">Scheduler</label><select id="gen-scheduler" class="gen-select"></select></div>';
        var initBody =
            '<div class="gen-param-row" data-param-search="init image image to image source denoise creativity">' +
            '<div id="gen-init-drop" class="gen-init-drop"><input id="gen-init-image-input" type="file" accept="image/*">' +
            '<div id="gen-init-empty"><i data-lucide="image-plus"></i><span>Choose or drop a source image</span></div>' +
            '<img id="gen-init-preview" alt="Init image preview" style="display:none"></div>' +
            '<div id="gen-init-name" class="gen-init-name">No source image</div>' +
            '<button id="gen-init-clear" type="button" class="gen-small-btn" disabled>Clear image</button>' +
            '<div class="gen-setting-row"><label class="gen-setting-label" for="gen-creativity">Creativity</label>' +
            '<input id="gen-creativity-range" type="range" class="gen-range" min="0" max="1" step="0.01" value="0.5">' +
            '<input id="gen-creativity" type="number" class="gen-number-input" min="0" max="1" step="0.01" value="0.5"></div></div>';
        var loraBody =
            '<div id="gen-lora-capability" class="gen-capability-note"></div><div id="gen-lora-list" class="gen-lora-list"></div>' +
            '<select id="gen-lora-picker" class="gen-lora-dropdown"><option disabled selected>No compatible LoRAs loaded</option></select>';
        return '<div class="gen-swarm-parameter-head"><div><strong>Parameters</strong><span id="gen-arch-badge" class="gen-arch-badge">IMAGE</span></div>' +
            '<button id="gen-quick-reset" type="button" class="gen-small-btn" title="Reset selected model defaults">Reset</button></div>' +
            '<div class="gen-swarm-filter"><i data-lucide="search"></i><input id="gen-param-filter" type="search" placeholder="Filter parameters..."></div>' +
            '<div class="gen-swarm-param-scroll">' +
            swarmGroup('gen-settings-header', 'Model', modelBody, true, 'Select an installed image model. Video models are intentionally excluded here.') +
            swarmGroup('gen-core-header', 'Core Parameters', coreBody, true, 'The controls used by every admitted image backend.') +
            '<section id="gen-variation-section">' + swarmGroup('gen-variation-header', 'Variation Seed', variationBody, false, 'Blend deterministic secondary noise into supported model families.') + '</section>' +
            swarmGroup('gen-image-header', 'Resolution', resolutionBody, false, 'Only compiled, production-admitted shapes are listed.') +
            swarmGroup('gen-sampling-header', 'Sampling', samplingBody, false, 'Sampler and scheduler values come from the selected backend capability report.') +
            '<section id="gen-init-section">' + swarmGroup('gen-init-header', 'Init Image', initBody, false, 'Image-to-image is shown only when the selected backend admits it.') + '</section>' +
            '<section id="gen-lora-section">' + swarmGroup('gen-lora-header', 'LoRAs', loraBody, false, 'Only backend-admitted LoRA counts are allowed.') + '</section>' +
            '</div>' +
            '<div id="gen-left-progress" class="gen-progress gen-left-progress"><div id="gen-left-progress-bar" class="gen-progress-bar"></div></div>' +
            '<div id="gen-left-progress-label" class="gen-left-progress-label"></div>' +
            '<div hidden aria-hidden="true"><div id="gen-video-section"></div><div id="gen-scail2-section"></div>' +
            '<input id="gen-seconds" value="1"><input id="gen-fps" value="1"><input id="gen-fps-range" value="1">' +
            '<div id="gen-duration-hint"></div></div>';
    }

    function buildSwarmTopToolbarHTML() {
        return '<div class="gen-swarm-topbar">' +
            '<div class="gen-swarm-model-info"><span id="gen-model-badge" class="gen-model-badge"></span><span id="gen-runtime-label">Serenity image runtime</span></div>' +
            '<div class="gen-swarm-quick-tools"><button id="gen-toolbar-copy" class="gen-toolbar-btn" title="Copy current image URL"><i data-lucide="copy"></i></button>' +
            '<button id="gen-toolbar-delete" class="gen-toolbar-btn" title="Clear preview"><i data-lucide="trash-2"></i></button>' +
            '<button id="gen-toolbar-toggle-gallery" class="gen-toolbar-btn active" title="Toggle batch panel"><i data-lucide="panel-right"></i></button></div></div>';
    }

    function buildSwarmPromptHTML() {
        return '<div class="gen-swarm-prompts"><div class="gen-swarm-prompt-main">' +
            '<div class="gen-prompt-label-row"><label class="gen-label" for="gen-prompt">Prompt</label><span id="gen-token-count" class="gen-token-count">~0 tokens</span></div>' +
            '<textarea id="gen-prompt" class="gen-textarea gen-swarm-prompt-textarea" rows="3" placeholder="Describe the image you want..."></textarea>' +
            '<div class="gen-swarm-prompt-options"><label class="gen-label" for="gen-style-preset">Style</label><select id="gen-style-preset" class="gen-select">' + buildStyleOptions() + '</select>' +
            '<div id="gen-style-preview" class="gen-style-preview" style="display:none"></div></div></div>' +
            '<div id="gen-neg-section" class="gen-swarm-negative"><label class="gen-label" for="gen-neg-prompt">Negative Prompt</label>' +
            '<textarea id="gen-neg-prompt" class="gen-textarea" rows="3" placeholder="What to avoid..."></textarea></div></div>' +
            '<div class="gen-swarm-generate-column"><button id="gen-btn" class="gen-btn"><i data-lucide="wand-2"></i><span>Generate</span></button>' +
            '<div class="gen-toolbar-batch"><label for="gen-toolbar-batch-input">Images</label><input type="number" id="gen-toolbar-batch-input" class="gen-toolbar-batch-input" min="1" max="8" value="1">' +
            '<button id="gen-toolbar-batch-up" title="Increase batch">+</button><button id="gen-toolbar-batch-down" title="Decrease batch">−</button></div></div>';
    }

    function buildSwarmBatchHTML() {
        return '<div class="gen-swarm-batch-head"><strong>Current Batch</strong><span id="gen-batch-status">Idle</span></div>' +
            '<div id="gen-batch-strip" class="gen-swarm-batch-strip"><div class="gen-swarm-batch-empty">New results appear here</div></div>';
    }

    function buildSwarmLibraryHTML() {
        return '<div class="gen-swarm-library-tabs">' +
            '<button class="gen-library-tab active" data-library="history">History</button>' +
            '<button class="gen-library-tab" data-library="presets">Presets</button>' +
            '<button class="gen-library-tab" data-library="models">Models</button>' +
            '<button class="gen-library-tab" data-library="loras">LoRAs</button>' +
            '<div class="gen-library-spacer"></div>' +
            '<button id="gen-gallery-upload-btn" class="gen-gallery-subtab-btn" title="Import image into history"><i data-lucide="upload"></i></button>' +
            '<input type="file" id="gen-gallery-upload-input" accept="image/*" multiple style="display:none">' +
            '<button id="gen-gallery-search-btn" class="gen-gallery-subtab-btn" title="Search history"><i data-lucide="search"></i></button>' +
            '<input id="gen-gallery-search-input" class="gen-gallery-search" type="text" placeholder="Search history...">' +
            '<button id="gen-gallery-settings-btn" class="gen-gallery-subtab-btn" title="History display settings"><i data-lucide="settings"></i></button>' +
            '<button id="gen-gallery-clear" class="gen-gallery-clear">Clear</button></div>' +
            '<div id="gen-gallery-popover" class="gen-gallery-popover"><div class="gen-popover-row"><span class="gen-popover-label">Thumbnail size</span>' +
            '<input type="range" id="gen-thumb-size-slider" class="gen-range gen-popover-slider" min="45" max="200" step="5" value="90"><span id="gen-thumb-size-val">90</span></div>' +
            '<div class="gen-popover-row"><span class="gen-popover-label">Newest first</span><button id="gen-sort-direction-toggle" class="gen-toggle on"></button><span id="gen-sort-direction-label">Newest</span></div>' +
            '<div class="gen-popover-row"><span class="gen-popover-label">Starred first</span><button id="gen-starred-first-toggle" class="gen-toggle"></button></div>' +
            '<div class="gen-popover-row"><span class="gen-popover-label">Auto-switch new</span><button id="gen-auto-switch-toggle" class="gen-toggle on"></button></div></div>' +
            '<div class="gen-library-panel active" data-library-panel="history"><div id="gen-images-content"><div class="gen-gallery-grid-wrap">' +
            '<div id="gen-gallery-grid" class="gen-gallery-grid"></div><div id="gen-selection-badge" class="gen-selection-badge"></div></div></div>' +
            '<div id="gen-bulk-bar" class="gen-bulk-bar"><button id="gen-bulk-star" class="gen-bulk-btn">Star All</button><button id="gen-bulk-unstar" class="gen-bulk-btn">Unstar All</button>' +
            '<button id="gen-bulk-download" class="gen-bulk-btn">Download</button><button id="gen-bulk-delete" class="gen-bulk-btn destructive">Delete</button></div>' +
            '<div id="gen-gallery-pagination" class="gen-gallery-pagination"><button id="gen-page-prev" disabled>&lt; Prev</button><span id="gen-page-info">Page 1</span><button id="gen-page-next">Next &gt;</button></div></div>' +
            '<div class="gen-library-panel" data-library-panel="presets"><div class="gen-preset-toolbar"><input id="gen-preset-name" class="gen-select" placeholder="Preset name">' +
            '<button id="gen-preset-save" class="gen-small-btn">Save current</button></div><div id="gen-preset-list" class="gen-library-card-grid"></div></div>' +
            '<div class="gen-library-panel" data-library-panel="models"><div id="gen-library-models" class="gen-library-card-grid"></div></div>' +
            '<div class="gen-library-panel" data-library-panel="loras"><div id="gen-library-loras" class="gen-library-card-grid"></div></div>';
    }
    function buildLeftHTML() {
        return '' +
            // Generate button + progress bar at top
            '<div class="gen-section gen-top-actions">' +
            '<button id="gen-btn" class="gen-btn"><i data-lucide="wand-2"></i> Generate</button>' +
            '<div id="gen-left-progress" class="gen-progress gen-left-progress"><div id="gen-left-progress-bar" class="gen-progress-bar"></div></div>' +
            '<div id="gen-left-progress-label" class="gen-left-progress-label"></div>' +
            '</div>' +
            // Prompt
            '<div class="gen-section">' +
            '<div class="gen-prompt-label-row">' +
            '<label class="gen-label">Positive Prompt</label>' +
            '<div class="gen-prompt-label-actions">' +
            '<button class="gen-prompt-label-btn" title="Dynamic prompts">' +
            '&lt;/&gt;' +
            '<span class="gen-tooltip">Dynamic prompts coming soon</span>' +
            '</button>' +
            '<button class="gen-prompt-label-btn" title="Templates">' +
            '{}' +
            '<span class="gen-tooltip">Templates coming soon</span>' +
            '</button>' +
            '</div>' +
            '</div>' +
            '<textarea id="gen-prompt" class="gen-textarea" rows="4" placeholder="Describe your image..."></textarea>' +
            '<div id="gen-token-count" class="gen-token-count">~0 tokens</div>' +
            // Style preset dropdown
            '<div style="margin-top:6px">' +
            '<label class="gen-label">Style</label>' +
            '<select id="gen-style-preset" class="gen-select">' +
            buildStyleOptions() +
            '</select>' +
            '<div id="gen-style-preview" class="gen-style-preview" style="display:none"></div>' +
            '</div>' +
            '</div>' +
            // Negative prompt
            '<div class="gen-section" id="gen-neg-section">' +
            '<label class="gen-label">Negative Prompt</label>' +
            '<textarea id="gen-neg-prompt" class="gen-textarea" rows="2" placeholder="What to avoid..."></textarea>' +
            '</div>' +
            // Image Settings accordion
            '<div class="gen-section">' +
            '<div id="gen-image-header" class="gen-accordion-header">' +
            '<span>Image</span>' +
            '<span class="gen-accordion-arrow"><i data-lucide="chevron-down"></i></span>' +
            '</div>' +
            '<div id="gen-image-body" class="gen-accordion-body" style="margin-top:8px">' +
            // Aspect row: dropdown + swap + lock + optimal
            '<div class="gen-image-content">' +
            '<div class="gen-image-controls">' +
            '<div class="gen-aspect-row">' +
            '<span class="gen-label" style="margin-bottom:0;min-width:44px">Aspect</span>' +
            '<select id="gen-aspect-dropdown" class="gen-select" style="flex:1">' +
            buildAspectOptions() +
            '</select>' +
            '<button id="gen-swap-btn" class="gen-aspect-action" title="Swap width and height">' +
            '<i data-lucide="arrow-left-right"></i>' +
            '</button>' +
            '<button id="gen-aspect-lock" class="gen-aspect-action' + (state.aspectLocked ? ' active' : '') + '" title="Lock aspect ratio">' +
            '<i data-lucide="lock"></i>' +
            '</button>' +
            '<button id="gen-optimal-btn" class="gen-aspect-action" title="Set optimal size for model">' +
            '<i data-lucide="sparkles"></i>' +
            '</button>' +
            '</div>' +
            '<div class="gen-dim-row">' +
            '<span class="gen-dim-label">Width</span>' +
            '<input type="range" id="gen-width-slider" class="gen-range" min="256" max="2048" step="64" value="1024">' +
            '<input type="number" id="gen-custom-width" class="gen-number-input" min="256" max="4096" step="64" value="1024">' +
            '</div>' +
            '<div class="gen-dim-row">' +
            '<span class="gen-dim-label">Height</span>' +
            '<input type="range" id="gen-height-slider" class="gen-range" min="256" max="2048" step="64" value="1024">' +
            '<input type="number" id="gen-custom-height" class="gen-number-input" min="256" max="4096" step="64" value="1024">' +
            '</div>' +
            // Seed inside Image section
            '<div style="margin-top:8px">' +
            '<label class="gen-label">Seed</label>' +
            '<div class="gen-seed-row">' +
            '<input id="gen-seed" type="number" class="gen-number-input" value="-1">' +
            '<button id="gen-seed-shuffle" class="gen-seed-btn" title="Shuffle seed">\u21BB</button>' +
            '<button id="gen-seed-prev" class="gen-seed-btn" title="Use previous seed">\u21BA</button>' +
            '<span class="gen-seed-random-label">Random</span>' +
            '<button id="gen-seed-random-toggle" class="gen-toggle' + (state.seed === -1 ? ' on' : '') + '"></button>' +
            '</div>' +
            '</div>' +
            // Advanced Options disclosure
            '<div id="gen-image-adv-disclosure" class="gen-adv-disclosure">' +
            '<i data-lucide="chevron-right"></i>' +
            '<span>Advanced Options</span>' +
            '</div>' +
            '<div id="gen-image-adv-body" class="gen-adv-body"></div>' +
            '</div>' +
            // Visual aspect preview box
            '<div class="gen-image-preview-col">' +
            '<div id="gen-aspect-preview" class="gen-aspect-preview" style="width:100px;height:100px">' +
            '<span>1:1</span>' +
            '</div>' +
            '</div>' +
            '</div>' +
            '</div>' +
            '</div>' +
            // Generation Settings accordion
            '<div class="gen-section">' +
            '<div id="gen-settings-header" class="gen-accordion-header">' +
            '<div class="gen-section-header-row">' +
            '<span>Generation</span>' +
            '<span id="gen-model-badge" class="gen-model-badge" style="display:none"></span>' +
            '<span id="gen-arch-badge" class="gen-arch-badge">SD1.5</span>' +
            '</div>' +
            '<span class="gen-accordion-arrow"><i data-lucide="chevron-down"></i></span>' +
            '</div>' +
            '<div id="gen-settings-body" class="gen-accordion-body" style="margin-top:8px">' +
            // Model
            '<label class="gen-label">Model</label>' +
            '<div class="gen-model-row">' +
            '<div class="gen-model-picker-wrap" id="gen-model-picker-wrap">' +
            '<input type="text" id="gen-model-search" class="gen-select gen-model-search-input" placeholder="Loading models..." autocomplete="off">' +
            '<div id="gen-model-dropdown" class="gen-model-dropdown" style="display:none">' +
            '<div id="gen-model-dropdown-list" class="gen-model-dropdown-list"></div>' +
            '</div>' +
            '</div>' +
            '<button id="gen-model-refresh" class="gen-model-action-btn" title="Refresh models"><i data-lucide="refresh-cw"></i></button>' +
            '<button class="gen-model-action-btn" title="Model settings (coming soon)"><i data-lucide="settings"></i></button>' +
            '</div>' +
            '<input type="hidden" id="gen-model" value="">' +
            '<div id="gen-model-warn" class="gen-model-warning"></div>' +
            // Concepts / LoRA
            '<label class="gen-label" style="margin-top:8px">Concepts</label>' +
            '<div id="gen-lora-list" class="gen-lora-list"></div>' +
            '<select id="gen-lora-picker" class="gen-lora-dropdown">' +
            '<option disabled selected>No LoRAs loaded</option>' +
            '</select>' +
            // Advanced Options for Generation (default OPEN)
            '<div id="gen-gen-adv-disclosure" class="gen-adv-disclosure open" style="margin-top:8px">' +
            '<i data-lucide="chevron-right"></i>' +
            '<span>Advanced Options</span>' +
            '</div>' +
            '<div id="gen-gen-adv-body" class="gen-adv-body open">' +
            // Steps
            '<div class="gen-setting-row" style="margin-top:6px">' +
            '<span class="gen-label">Steps</span>' +
            '<input id="gen-steps-range" type="range" class="gen-range" min="1" max="150" value="20">' +
            '<input id="gen-steps" type="number" class="gen-number-input" min="1" max="500" value="20">' +
            '</div>' +
            // CFG
            '<div id="gen-cfg-row" class="gen-setting-row">' +
            '<span class="gen-label">CFG</span>' +
            '<input id="gen-cfg-range" type="range" class="gen-range" min="1" max="20" step="0.5" value="7.0">' +
            '<input id="gen-cfg" type="number" class="gen-number-input" min="1" max="200" step="0.5" value="7.0">' +
            '</div>' +
            // Guidance (FLUX only)
            '<div id="gen-guidance-row" class="gen-setting-row" style="display:none">' +
            '<span class="gen-label">Guidance</span>' +
            '<input id="gen-guidance-range" type="range" class="gen-range" min="1" max="10" step="0.5" value="3.5">' +
            '<input id="gen-guidance" type="number" class="gen-number-input" min="1" max="10" step="0.5" value="3.5">' +
            '</div>' +
            // Scheduler
            '<div class="gen-setting-row">' +
            '<span class="gen-label">Sampler</span>' +
            '<select id="gen-scheduler" class="gen-select" style="flex:1">' +
            buildSchedulerOptions() +
            '</select>' +
            '</div>' +
            // Batch
            '<div class="gen-setting-row" id="gen-batch-section">' +
            '<span class="gen-label">Batch</span>' +
            '<input id="gen-batch" type="number" class="gen-number-input" min="1" max="8" value="1">' +
            '<span class="gen-batch-hint">images per run</span>' +
            '</div>' +
            '</div>' +
            '</div>' +
            '</div>' +
            // Video controls (hidden by default, shown for video models)
            '<div class="gen-section" id="gen-video-section" style="display:none">' +
            '<label class="gen-label">Video</label>' +
            '<div class="gen-setting-row">' +
            '<span class="gen-label" style="min-width:52px;margin-bottom:0">Seconds</span>' +
            '<input type="number" id="gen-seconds" class="gen-number-input" min="1" max="30" step="1" value="10" style="width:64px">' +
            '</div>' +
            '<div class="gen-setting-row">' +
            '<span class="gen-label" style="min-width:52px;margin-bottom:0">FPS</span>' +
            '<input type="range" id="gen-fps-range" class="gen-range" min="8" max="60" step="1" value="24">' +
            '<input type="number" id="gen-fps" class="gen-number-input" min="8" max="60" step="1" value="24">' +
            '</div>' +
            '<div id="gen-duration-hint" class="gen-duration-hint"></div>' +
            '</div>' +
            '<div class="gen-section" id="gen-scail2-section" style="display:none">' +
            '<label class="gen-label">SCAIL-2 Inputs</label>' +
            '<div class="gen-duration-hint">Reference image + mask and driving video + mask are required. Audio is preserved from the driving video.</div>' +
            '<label class="gen-label" style="margin-top:8px">Mode<select id="gen-scail2-mode" class="gen-select"><option value="animation">Animation</option><option value="replacement">Character replacement</option></select></label>' +
            '<label class="gen-label">Reference image<input id="gen-scail2-reference-image" type="file" accept="image/*" class="gen-select"></label>' +
            '<label class="gen-label">Reference mask<input id="gen-scail2-reference-mask" type="file" accept="image/*" class="gen-select"></label>' +
            '<label class="gen-label">Driving video<input id="gen-scail2-driving-video" type="file" accept="video/*" class="gen-select"></label>' +
            '<label class="gen-label">Driving mask video<input id="gen-scail2-driving-mask-video" type="file" accept="video/*" class="gen-select"></label>' +
            '<label class="gen-label">Additional references (optional, up to 3)<input id="gen-scail2-additional-images" type="file" accept="image/*" multiple class="gen-select"></label>' +
            '<label class="gen-label">Additional reference masks<input id="gen-scail2-additional-masks" type="file" accept="image/*" multiple class="gen-select"></label>' +
            '<div id="gen-scail2-upload-status" class="gen-duration-hint"></div>' +
            '</div>' +
            // Compositing section
            '<div class="gen-section">' +
            '<div id="gen-compositing-header" class="gen-accordion-header closed">' +
            '<span>Compositing</span>' +
            '<span class="gen-accordion-arrow"><i data-lucide="chevron-down"></i></span>' +
            '</div>' +
            '<div id="gen-compositing-body" class="gen-accordion-body closed" style="margin-top:8px">' +
            '<div class="gen-setting-row">' +
            '<span class="gen-label" style="min-width:80px;margin-bottom:0">Coherence</span>' +
            '<select id="gen-coherence-mode" class="gen-select" style="flex:1">' +
            '<option value="gaussian">Gaussian Blur</option>' +
            '<option value="box">Box Blur</option>' +
            '<option value="staged">Staged</option>' +
            '</select>' +
            '</div>' +
            '<div class="gen-setting-row">' +
            '<span class="gen-label" style="min-width:80px;margin-bottom:0">Edge Size</span>' +
            '<input type="range" id="gen-edge-size-range" class="gen-range" min="0" max="128" step="4" value="32">' +
            '<input type="number" id="gen-edge-size" class="gen-number-input" min="0" max="128" step="4" value="32">' +
            '</div>' +
            '<div class="gen-setting-row">' +
            '<span class="gen-label" style="min-width:80px;margin-bottom:0">Min Denoise</span>' +
            '<input type="range" id="gen-min-denoise-range" class="gen-range" min="0" max="1" step="0.05" value="0">' +
            '<input type="number" id="gen-min-denoise" class="gen-number-input" min="0" max="1" step="0.05" value="0">' +
            '</div>' +
            '<div class="gen-setting-row">' +
            '<span class="gen-label" style="min-width:80px;margin-bottom:0">Mask Blur</span>' +
            '<input type="range" id="gen-mask-blur-range" class="gen-range" min="0" max="128" step="1" value="4">' +
            '<input type="number" id="gen-mask-blur" class="gen-number-input" min="0" max="128" step="1" value="4">' +
            '</div>' +
            '<div class="gen-setting-row">' +
            '<span class="gen-label" style="min-width:80px;margin-bottom:0">Infill</span>' +
            '<select id="gen-infill-method" class="gen-select" style="flex:1">' +
            '<option value="patchmatch">PatchMatch</option>' +
            '<option value="lama">LaMa</option>' +
            '<option value="cv2">CV2 Inpaint</option>' +
            '<option value="color">Solid Color</option>' +
            '<option value="tile">Tile</option>' +
            '</select>' +
            '</div>' +
            '</div>' +
            '</div>' +
            // Refiner section (SDXL only)
            '<div class="gen-section" id="gen-refiner-section">' +
            '<div id="gen-refiner-header" class="gen-accordion-header closed">' +
            '<span>Refiner</span>' +
            '<span class="gen-accordion-arrow"><i data-lucide="chevron-down"></i></span>' +
            '</div>' +
            '<div id="gen-refiner-body" class="gen-accordion-body closed" style="margin-top:8px">' +
            '<div class="gen-setting-row">' +
            '<span class="gen-label" style="min-width:80px;margin-bottom:0">Model</span>' +
            '<select id="gen-refiner-model" class="gen-select" style="flex:1">' +
            '<option value="">None</option>' +
            '</select>' +
            '</div>' +
            '<div class="gen-setting-row">' +
            '<span class="gen-label" style="min-width:80px;margin-bottom:0">Scheduler</span>' +
            '<select id="gen-refiner-scheduler" class="gen-select" style="flex:1">' +
            buildSchedulerOptions() +
            '</select>' +
            '</div>' +
            '<div class="gen-setting-row">' +
            '<span class="gen-label" style="min-width:80px;margin-bottom:0">Steps</span>' +
            '<input type="range" id="gen-refiner-steps-range" class="gen-range" min="1" max="100" value="20">' +
            '<input type="number" id="gen-refiner-steps" class="gen-number-input" min="1" max="500" value="20">' +
            '</div>' +
            '<div class="gen-setting-row">' +
            '<span class="gen-label" style="min-width:80px;margin-bottom:0">CFG</span>' +
            '<input type="range" id="gen-refiner-cfg-range" class="gen-range" min="1" max="20" step="0.5" value="7.0">' +
            '<input type="number" id="gen-refiner-cfg" class="gen-number-input" min="1" max="200" step="0.5" value="7.0">' +
            '</div>' +
            '<div class="gen-setting-row">' +
            '<span class="gen-label" style="min-width:80px;margin-bottom:0">Start</span>' +
            '<input type="range" id="gen-refiner-start-range" class="gen-range" min="0" max="1" step="0.05" value="0.8">' +
            '<input type="number" id="gen-refiner-start" class="gen-number-input" min="0" max="1" step="0.05" value="0.80">' +
            '</div>' +
            '<div class="gen-setting-row">' +
            '<span class="gen-label" style="min-width:80px;margin-bottom:0">Aesthetic +</span>' +
            '<input type="range" id="gen-aesthetic-pos-range" class="gen-range" min="1" max="10" step="0.5" value="6.0">' +
            '<input type="number" id="gen-aesthetic-pos" class="gen-number-input" min="1" max="10" step="0.5" value="6.0">' +
            '</div>' +
            '<div class="gen-setting-row">' +
            '<span class="gen-label" style="min-width:80px;margin-bottom:0">Aesthetic -</span>' +
            '<input type="range" id="gen-aesthetic-neg-range" class="gen-range" min="1" max="10" step="0.5" value="2.5">' +
            '<input type="number" id="gen-aesthetic-neg" class="gen-number-input" min="1" max="10" step="0.5" value="2.5">' +
            '</div>' +
            '</div>' +
            '</div>' +
            // Advanced section
            '<div class="gen-section">' +
            '<div id="gen-advanced-header" class="gen-accordion-header closed">' +
            '<span>Advanced</span>' +
            '<span class="gen-accordion-arrow"><i data-lucide="chevron-down"></i></span>' +
            '</div>' +
            '<div id="gen-advanced-body" class="gen-accordion-body closed" style="margin-top:8px">' +
            '<div class="gen-setting-row">' +
            '<span class="gen-label" style="min-width:80px;margin-bottom:0">VAE</span>' +
            '<select id="gen-vae-picker" class="gen-select" style="flex:1">' +
            '<option value="default">Default (from checkpoint)</option>' +
            '</select>' +
            '</div>' +
            '<div class="gen-setting-row" id="gen-clip-skip-row">' +
            '<span class="gen-label" style="min-width:80px;margin-bottom:0">CLIP Skip</span>' +
            '<input type="range" id="gen-clip-skip-range" class="gen-range" min="0" max="12" step="1" value="0">' +
            '<input type="number" id="gen-clip-skip" class="gen-number-input" min="0" max="12" step="1" value="0">' +
            '</div>' +
            '<div class="gen-setting-row">' +
            '<span class="gen-label" style="min-width:80px;margin-bottom:0">CFG Rescale</span>' +
            '<input type="range" id="gen-cfg-rescale-range" class="gen-range" min="0" max="0.99" step="0.01" value="0">' +
            '<input type="number" id="gen-cfg-rescale" class="gen-number-input" min="0" max="0.99" step="0.01" value="0">' +
            '</div>' +
            '<div class="gen-setting-row">' +
            '<span class="gen-label" style="min-width:80px;margin-bottom:0">Seamless</span>' +
            '<div style="display:flex;gap:8px;align-items:center">' +
            '<button id="gen-seamless-x" class="gen-seamless-btn" title="Tile horizontally">X</button>' +
            '<button id="gen-seamless-y" class="gen-seamless-btn" title="Tile vertically">Y</button>' +
            '</div>' +
            '</div>' +
            '</div>' +
            '</div>';
    }
    function buildTopToolbarHTML() {
        return '' +
            '<div class="gen-top-toolbar" id="gen-top-toolbar">' +
            // Generate button
            '<button id="gen-toolbar-generate" class="gen-toolbar-btn gen-toolbar-generate" title="Generate">' +
            '<i data-lucide="wand-2"></i>' +
            '<span>Generate</span>' +
            '</button>' +
            // Batch count with spinners
            '<div class="gen-toolbar-batch">' +
            '<input type="number" id="gen-toolbar-batch-input" class="gen-toolbar-batch-input" min="1" max="8" value="1">' +
            '<div class="gen-toolbar-batch-spin">' +
            '<button id="gen-toolbar-batch-up" title="Increase batch">\u25B2</button>' +
            '<button id="gen-toolbar-batch-down" title="Decrease batch">\u25BC</button>' +
            '</div>' +
            '</div>' +
            '<span class="gen-toolbar-sep"></span>' +
            // View mode icons (stubs)
            '<button class="gen-toolbar-btn active" title="Image viewer"><i data-lucide="eye"></i></button>' +
            '<button class="gen-toolbar-btn" title="Brush mode"><i data-lucide="paintbrush"></i></button>' +
            '<button class="gen-toolbar-btn" title="List view"><i data-lucide="list"></i></button>' +
            '<button class="gen-toolbar-btn" title="Close view"><i data-lucide="x"></i></button>' +
            '<span class="gen-toolbar-sep"></span>' +
            // Center: info
            '<button class="gen-toolbar-btn" title="Image info"><i data-lucide="info"></i></button>' +
            '<span class="gen-toolbar-spacer"></span>' +
            // Toggle gallery panel
            '<button id="gen-toolbar-toggle-gallery" class="gen-toolbar-btn active" title="Toggle gallery"><i data-lucide="image"></i></button>' +
            '<span class="gen-toolbar-sep"></span>' +
            // Right group: undo, redo, flip-h, flip-v, star, copy, delete
            '<button class="gen-toolbar-btn" title="Undo" disabled><i data-lucide="undo-2"></i></button>' +
            '<button class="gen-toolbar-btn" title="Redo" disabled><i data-lucide="redo-2"></i></button>' +
            '<button class="gen-toolbar-btn" title="Flip horizontal" disabled><i data-lucide="flip-horizontal"></i></button>' +
            '<button class="gen-toolbar-btn" title="Flip vertical" disabled><i data-lucide="flip-vertical"></i></button>' +
            '<button class="gen-toolbar-btn" title="Favorite" disabled><i data-lucide="star"></i></button>' +
            '<button id="gen-toolbar-copy" class="gen-toolbar-btn" title="Copy image URL"><i data-lucide="copy"></i></button>' +
            '<button id="gen-toolbar-delete" class="gen-toolbar-btn" title="Delete"><i data-lucide="trash-2"></i></button>' +
            '</div>';
    }
    function buildFloatingToolbarHTML() {
        return '' +
            '<button id="gen-float-toggle-left" class="gen-float-btn" title="Toggle left panel"><i data-lucide="panel-left"></i></button>' +
            '<button id="gen-float-generate" class="gen-float-btn gen-float-generate" title="Generate"><i data-lucide="sparkles"></i></button>' +
            '<button id="gen-float-cancel" class="gen-float-btn gen-float-cancel" title="Cancel"><i data-lucide="x"></i></button>' +
            '<button id="gen-float-delete" class="gen-float-btn gen-float-delete" title="Delete current"><i data-lucide="trash-2"></i></button>';
    }
    function buildCenterHTML() {
        return '' +
            '<div class="gen-center-content">' +
            '<div id="gen-empty" class="gen-empty">' +
            '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/></svg>' +
            '<span class="gen-empty-text">Your image will appear here</span>' +
            '</div>' +
            '<img id="gen-preview-img" class="gen-preview-img" style="display:none" alt="Generated image">' +
            '<video id="gen-preview-video" class="gen-preview-video" style="display:none" autoplay loop playsinline controls></video>' +
            '<div id="gen-action-bar" class="gen-action-bar" style="display:none">' +
            '<button class="gen-action-btn" id="gen-download" title="Download"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg></button>' +
            '<button class="gen-action-btn" id="gen-to-canvas" title="Coming in Canvas tab" disabled><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M9 21V9"/></svg></button>' +
            '<button class="gen-action-btn" id="gen-to-timeline" title="Send to Timeline"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"/><line x1="7" y1="2" x2="7" y2="22"/><line x1="17" y1="2" x2="17" y2="22"/><line x1="2" y1="12" x2="22" y2="12"/><line x1="2" y1="7" x2="7" y2="7"/><line x1="2" y1="17" x2="7" y2="17"/><line x1="17" y1="7" x2="22" y2="7"/><line x1="17" y1="17" x2="22" y2="17"/></svg></button>' +
            '<button class="gen-action-btn" id="gen-clear-preview" title="Clear"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/></svg></button>' +
            '</div>' +
            // Metadata panel (below action bar)
            '<div id="gen-metadata-panel" class="gen-metadata-panel">' +
            '<div id="gen-metadata-summary" class="gen-metadata-summary"></div>' +
            '<div id="gen-metadata-full" class="gen-metadata-full"></div>' +
            '</div>' +
            '</div>' +
            '<div id="gen-progress-label" class="gen-progress-label"></div>' +
            '<div id="gen-progress" class="gen-progress"><div id="gen-progress-bar" class="gen-progress-bar"></div></div>' +
            '<div id="gen-error-banner" class="gen-error-banner"></div>' +
            '<div id="gen-ws-indicator" class="gen-ws-indicator"><span class="gen-ws-dot"></span><span>Reconnecting...</span></div>';
    }
    function buildRightHTML() {
        return '' +
            // Tab switcher: Layers | Gallery | close
            '<div class="gen-gallery-tabs">' +
            '<button id="gen-tab-layers" class="gen-gallery-tab">Layers</button>' +
            '<button id="gen-tab-gallery" class="gen-gallery-tab active">Gallery</button>' +
            '<button id="gen-gallery-close" class="gen-gallery-close-btn" title="Close panel"><i data-lucide="x"></i></button>' +
            '</div>' +
            // Layers tab content
            '<div id="gen-layers-content" class="gen-gallery-tab-content">' +
            '<div class="gen-layers-placeholder">Coming soon</div>' +
            '</div>' +
            // Gallery tab content
            '<div id="gen-gallery-content" class="gen-gallery-tab-content active">' +
            // Boards section
            '<div class="gen-boards-section">' +
            '<div id="gen-boards-header" class="gen-boards-header">' +
            '<span>Hide Boards</span>' +
            '<span class="gen-accordion-arrow"><i data-lucide="chevron-up"></i></span>' +
            '</div>' +
            '<div id="gen-boards-body" class="gen-boards-body">' +
            '<div class="gen-board-item">' +
            '<span><span class="gen-board-icon">\uD83D\uDCC1</span> Uncategorized</span>' +
            '<span class="gen-board-badge">AUTO</span>' +
            '</div>' +
            '</div>' +
            '</div>' +
            // Sub-tabs: Images | Assets | actions
            '<div class="gen-gallery-subtabs" style="position:relative">' +
            '<button id="gen-subtab-images" class="gen-gallery-subtab active">Images</button>' +
            '<button id="gen-subtab-assets" class="gen-gallery-subtab">Assets</button>' +
            '<div class="gen-gallery-subtab-actions">' +
            '<button id="gen-gallery-upload-btn" class="gen-gallery-subtab-btn" title="Upload image"><i data-lucide="upload"></i></button>' +
            '<input type="file" id="gen-gallery-upload-input" accept="image/*" multiple style="display:none">' +
            '<button id="gen-gallery-settings-btn" class="gen-gallery-subtab-btn" title="Gallery settings"><i data-lucide="settings"></i></button>' +
            '<button id="gen-gallery-search-btn" class="gen-gallery-subtab-btn" title="Search"><i data-lucide="search"></i></button>' +
            '<input id="gen-gallery-search-input" class="gen-gallery-search" type="text" placeholder="Search...">' +
            '</div>' +
            '<div id="gen-gallery-popover" class="gen-gallery-popover">' +
            '<div class="gen-popover-row">' +
            '<span class="gen-popover-label">Image Size</span>' +
            '<input type="range" id="gen-thumb-size-slider" class="gen-range gen-popover-slider" min="45" max="200" step="5" value="90">' +
            '<span id="gen-thumb-size-val" style="color:var(--shell-text);font-size:11px;min-width:28px;text-align:right">90</span>' +
            '</div>' +
            '<div class="gen-popover-row">' +
            '<span class="gen-popover-label">Sort Direction</span>' +
            '<button id="gen-sort-direction-toggle" class="gen-toggle on"></button>' +
            '<span id="gen-sort-direction-label" style="color:var(--shell-text-muted);font-size:10px;min-width:60px">Newest</span>' +
            '</div>' +
            '<div class="gen-popover-row">' +
            '<span class="gen-popover-label">Starred First</span>' +
            '<button id="gen-starred-first-toggle" class="gen-toggle"></button>' +
            '</div>' +
            '<div class="gen-popover-row">' +
            '<span class="gen-popover-label">Auto-Switch New</span>' +
            '<button id="gen-auto-switch-toggle" class="gen-toggle on"></button>' +
            '</div>' +
            '</div>' +
            '</div>' +
            '<div id="gen-images-content">' +
            '<div class="gen-gallery-header">' +
            '<span class="gen-gallery-title">Gallery</span>' +
            '<button id="gen-gallery-clear" class="gen-gallery-clear">Clear</button>' +
            '</div>' +
            '<div class="gen-gallery-grid-wrap">' +
            '<div id="gen-gallery-grid" class="gen-gallery-grid"></div>' +
            '<div id="gen-selection-badge" class="gen-selection-badge"></div>' +
            '</div>' +
            '</div>' +
            '<div id="gen-assets-content" style="display:none">' +
            '<div class="gen-assets-placeholder">No assets uploaded</div>' +
            '</div>' +
            '<div id="gen-bulk-bar" class="gen-bulk-bar">' +
            '<button id="gen-bulk-star" class="gen-bulk-btn">Star All</button>' +
            '<button id="gen-bulk-unstar" class="gen-bulk-btn">Unstar All</button>' +
            '<button id="gen-bulk-download" class="gen-bulk-btn">Download</button>' +
            '<button id="gen-bulk-delete" class="gen-bulk-btn destructive">Delete</button>' +
            '</div>' +
            '<div id="gen-gallery-pagination" class="gen-gallery-pagination">' +
            '<button id="gen-page-prev" disabled>&lt; Prev</button>' +
            '<span id="gen-page-info">Page 1</span>' +
            '<button id="gen-page-next">&gt; Next</button>' +
            '</div>' +
            '</div>';
    }
    function cacheElements() {
        els.model = document.getElementById('gen-model');
        els.modelSearch = document.getElementById('gen-model-search');
        els.modelDropdown = document.getElementById('gen-model-dropdown');
        els.modelDropdownList = document.getElementById('gen-model-dropdown-list');
        els.modelPickerWrap = document.getElementById('gen-model-picker-wrap');
        els.modelWarn = document.getElementById('gen-model-warn');
        els.prompt = document.getElementById('gen-prompt');
        els.negPrompt = document.getElementById('gen-neg-prompt');
        els.negSection = document.getElementById('gen-neg-section');
        els.customWidth = document.getElementById('gen-custom-width');
        els.customHeight = document.getElementById('gen-custom-height');
        els.videoSection = document.getElementById('gen-video-section');
        els.secondsInput = document.getElementById('gen-seconds');
        els.fpsInput = document.getElementById('gen-fps');
        els.fpsRange = document.getElementById('gen-fps-range');
        els.durationHint = document.getElementById('gen-duration-hint');
        els.scail2Section = document.getElementById('gen-scail2-section');
        els.scail2Mode = document.getElementById('gen-scail2-mode');
        els.scail2UploadStatus = document.getElementById('gen-scail2-upload-status');
        els.imageHeader = document.getElementById('gen-image-header');
        els.imageBody = document.getElementById('gen-image-body');
        els.settingsHeader = document.getElementById('gen-settings-header');
        els.settingsBody = document.getElementById('gen-settings-body');
        els.steps = document.getElementById('gen-steps');
        els.stepsRange = document.getElementById('gen-steps-range');
        els.cfgRow = document.getElementById('gen-cfg-row');
        els.cfg = document.getElementById('gen-cfg');
        els.cfgRange = document.getElementById('gen-cfg-range');
        els.guidanceRow = document.getElementById('gen-guidance-row');
        els.guidance = document.getElementById('gen-guidance');
        els.guidanceRange = document.getElementById('gen-guidance-range');
        els.sampler = document.getElementById('gen-sampler');
        els.scheduler = document.getElementById('gen-scheduler');
        els.seed = document.getElementById('gen-seed');
        els.seedShuffle = document.getElementById('gen-seed-shuffle');
        els.btn = document.getElementById('gen-btn');
        els.empty = document.getElementById('gen-empty');
        els.previewImg = document.getElementById('gen-preview-img');
        els.previewVideo = document.getElementById('gen-preview-video');
        els.actionBar = document.getElementById('gen-action-bar');
        els.download = document.getElementById('gen-download');
        els.clearPreview = document.getElementById('gen-clear-preview');
        els.progress = document.getElementById('gen-progress');
        els.progressBar = document.getElementById('gen-progress-bar');
        els.progressLabel = document.getElementById('gen-progress-label');
        els.leftProgress = document.getElementById('gen-left-progress');
        els.leftProgressBar = document.getElementById('gen-left-progress-bar');
        els.leftProgressLabel = document.getElementById('gen-left-progress-label');
        els.errorBanner = document.getElementById('gen-error-banner');
        els.wsIndicator = document.getElementById('gen-ws-indicator');
        els.galleryGrid = document.getElementById('gen-gallery-grid');
        els.galleryClear = document.getElementById('gen-gallery-clear');
        // New elements
        els.leftPanel = document.getElementById('gen-left-panel');
        els.rightPanel = document.getElementById('gen-right-panel');
        els.aspectDropdown = document.getElementById('gen-aspect-dropdown');
        els.aspectPreview = document.getElementById('gen-aspect-preview');
        els.modelBadge = document.getElementById('gen-model-badge');
        els.toolbarBatchInput = document.getElementById('gen-toolbar-batch-input');
        els.batchStrip = document.getElementById('gen-batch-strip');
        els.batchStatus = document.getElementById('gen-batch-status');
    }
    // ── Aspect Preview Box ──
    function updateAspectPreview() {
        if (!els.aspectPreview)
            return;
        var maxDim = 100;
        var w = state.width;
        var h = state.height;
        var scale;
        if (w >= h) {
            scale = maxDim / w;
        }
        else {
            scale = maxDim / h;
        }
        var pw = Math.max(16, Math.round(w * scale));
        var ph = Math.max(16, Math.round(h * scale));
        els.aspectPreview.style.width = pw + 'px';
        els.aspectPreview.style.height = ph + 'px';
        // Determine label
        var label = state.width + '\u00d7' + state.height;
        var aspects = getActiveAspects();
        for (var i = 0; i < aspects.length; i++) {
            if (aspects[i].w === state.width && aspects[i].h === state.height) {
                label = aspects[i].label;
                break;
            }
        }
        els.aspectPreview.innerHTML = '<span>' + label + '</span>';
    }
    // ── Aspect Dropdown Sync ──
    function syncAspectDropdown() {
        if (!els.aspectDropdown)
            return;
        var aspects = getActiveAspects();
        var found = false;
        for (var i = 0; i < aspects.length; i++) {
            if (aspects[i].label === 'Free')
                continue; // skip Free when matching
            if (aspects[i].w === state.width && aspects[i].h === state.height) {
                els.aspectDropdown.value = aspects[i].label;
                found = true;
                break;
            }
        }
        if (!found) {
            // If current dropdown is "Free", keep it; otherwise set to custom
            if (els.aspectDropdown.value !== 'Free') {
                els.aspectDropdown.value = 'custom';
            }
        }
    }
    // ── Event Binding ──
    function bindScail2Upload(inputId, stateKey, label) {
        var input = document.getElementById(inputId);
        if (!input)
            return;
        input.addEventListener('change', function () {
            var file = input.files && input.files[0];
            if (!file)
                return;
            var form = new FormData();
            form.append('file', file, file.name);
            state.mediaUploadsInFlight += 1;
            state[stateKey] = '';
            if (els.scail2UploadStatus)
                els.scail2UploadStatus.textContent = 'Uploading ' + label + '…';
            fetch('/upload/media', { method: 'POST', body: form })
                .then(function (response) {
                return response.text().then(function (body) {
                    var data = {};
                    try { data = JSON.parse(body); }
                    catch (error) { data = { detail: body }; }
                    if (!response.ok || !data.path)
                        throw new Error(data.detail || ('HTTP ' + response.status));
                    return data;
                });
            })
                .then(function (data) {
                state[stateKey] = data.path;
                if (els.scail2UploadStatus)
                    els.scail2UploadStatus.textContent = label + ' ready: ' + data.name;
            })
                .catch(function (error) {
                state[stateKey] = '';
                showError('Could not upload ' + label + ': ' + error.message);
            })
                .finally(function () {
                state.mediaUploadsInFlight = Math.max(0, state.mediaUploadsInFlight - 1);
            });
        });
    }
    function bindScail2MultiUpload(inputId, stateKey, label) {
        var input = document.getElementById(inputId);
        if (!input)
            return;
        input.addEventListener('change', function () {
            var files = Array.prototype.slice.call(input.files || []);
            if (!files.length)
                return;
            if (files.length > 3) {
                input.value = '';
                state[stateKey] = [];
                showError('SCAIL-2 accepts at most 3 ' + label);
                return;
            }
            state.mediaUploadsInFlight += files.length;
            state[stateKey] = [];
            Promise.all(files.map(function (file) {
                var form = new FormData();
                form.append('file', file, file.name);
                return fetch('/upload/media', { method: 'POST', body: form })
                    .then(function (response) {
                    return response.text().then(function (body) {
                        var data = {};
                        try { data = JSON.parse(body); }
                        catch (error) { data = { detail: body }; }
                        if (!response.ok || !data.path)
                            throw new Error(data.detail || ('HTTP ' + response.status));
                        return data.path;
                    });
                });
            }))
                .then(function (paths) {
                state[stateKey] = paths;
                if (els.scail2UploadStatus)
                    els.scail2UploadStatus.textContent = paths.length + ' ' + label + ' ready';
            })
                .catch(function (error) {
                state[stateKey] = [];
                showError('Could not upload ' + label + ': ' + error.message);
            })
                .finally(function () {
                state.mediaUploadsInFlight = Math.max(0, state.mediaUploadsInFlight - files.length);
            });
        });
    }
    function bindEvents() {
        bindScail2Upload('gen-scail2-reference-image', 'referenceImagePath', 'reference image');
        bindScail2Upload('gen-scail2-reference-mask', 'referenceMaskPath', 'reference mask');
        bindScail2Upload('gen-scail2-driving-video', 'drivingVideoPath', 'driving video');
        bindScail2Upload('gen-scail2-driving-mask-video', 'drivingMaskVideoPath', 'driving mask video');
        bindScail2MultiUpload('gen-scail2-additional-images', 'additionalReferenceImagePaths', 'additional reference images');
        bindScail2MultiUpload('gen-scail2-additional-masks', 'additionalReferenceMaskPaths', 'additional reference masks');
        if (els.scail2Mode) {
            els.scail2Mode.addEventListener('change', function () {
                state.scail2Mode = this.value === 'replacement' ? 'replacement' : 'animation';
            });
        }
        // Auto-grow prompt textarea
        els.prompt.addEventListener('input', function () {
            state.prompt = this.value;
            this.style.height = 'auto';
            this.style.height = this.scrollHeight + 'px';
            updateTokenCount();
            updateStylePreview();
        });
        els.negPrompt.addEventListener('input', function () {
            state.negPrompt = this.value;
        });
        // Image accordion
        els.imageHeader.addEventListener('click', function () {
            this.classList.toggle('closed');
            els.imageBody.classList.toggle('closed');
        });
        // Generation Settings accordion
        els.settingsHeader.addEventListener('click', function () {
            this.classList.toggle('closed');
            els.settingsBody.classList.toggle('closed');
        });
        // Section accordions
        bindAccordion('gen-compositing-header', 'gen-compositing-body');
        bindAccordion('gen-refiner-header', 'gen-refiner-body');
        bindAccordion('gen-advanced-header', 'gen-advanced-body');
        // ── Phase 3: Compositing controls ──
        bindSliderPair('gen-edge-size-range', 'gen-edge-size', function (v) { state.edgeSize = parseInt(v); });
        bindSliderPair('gen-min-denoise-range', 'gen-min-denoise', function (v) { state.minDenoise = parseFloat(v); });
        bindSliderPair('gen-mask-blur-range', 'gen-mask-blur', function (v) { state.maskBlur = parseInt(v); });
        var coherenceEl = document.getElementById('gen-coherence-mode');
        if (coherenceEl)
            coherenceEl.addEventListener('change', function () { state.coherenceMode = this.value; });
        var infillEl = document.getElementById('gen-infill-method');
        if (infillEl)
            infillEl.addEventListener('change', function () { state.infillMethod = this.value; });
        // ── Phase 3: Refiner controls ──
        bindSliderPair('gen-refiner-steps-range', 'gen-refiner-steps', function (v) { state.refinerSteps = parseInt(v); });
        bindSliderPair('gen-refiner-cfg-range', 'gen-refiner-cfg', function (v) { state.refinerCfg = parseFloat(v); });
        bindSliderPair('gen-refiner-start-range', 'gen-refiner-start', function (v) { state.refinerStart = parseFloat(v); });
        bindSliderPair('gen-aesthetic-pos-range', 'gen-aesthetic-pos', function (v) { state.aestheticScore = parseFloat(v); });
        bindSliderPair('gen-aesthetic-neg-range', 'gen-aesthetic-neg', function (v) { state.negAestheticScore = parseFloat(v); });
        var refSchedEl = document.getElementById('gen-refiner-scheduler');
        if (refSchedEl)
            refSchedEl.addEventListener('change', function () { state.refinerScheduler = this.value; });
        var refModelEl = document.getElementById('gen-refiner-model');
        if (refModelEl)
            refModelEl.addEventListener('change', function () { state.refinerModel = this.value; });
        // ── Phase 3: Advanced controls ──
        var vaeEl = document.getElementById('gen-vae-picker');
        if (vaeEl)
            vaeEl.addEventListener('change', function () { state.vae = this.value; });
        bindSliderPair('gen-clip-skip-range', 'gen-clip-skip', function (v) { state.clipSkip = parseInt(v); });
        bindSliderPair('gen-cfg-rescale-range', 'gen-cfg-rescale', function (v) { state.cfgRescale = parseFloat(v); });
        var seamXBtn = document.getElementById('gen-seamless-x');
        var seamYBtn = document.getElementById('gen-seamless-y');
        if (seamXBtn)
            seamXBtn.addEventListener('click', function () {
                state.seamlessX = !state.seamlessX;
                this.classList.toggle('on', state.seamlessX);
            });
        if (seamYBtn)
            seamYBtn.addEventListener('click', function () {
                state.seamlessY = !state.seamlessY;
                this.classList.toggle('on', state.seamlessY);
            });
        // Image Advanced Options disclosure
        var imgAdvDisc = document.getElementById('gen-image-adv-disclosure');
        var imgAdvBody = document.getElementById('gen-image-adv-body');
        if (imgAdvDisc && imgAdvBody) {
            imgAdvDisc.addEventListener('click', function () {
                this.classList.toggle('open');
                imgAdvBody.classList.toggle('open');
            });
        }
        // Generation Advanced Options disclosure
        var genAdvDisc = document.getElementById('gen-gen-adv-disclosure');
        var genAdvBody = document.getElementById('gen-gen-adv-body');
        if (genAdvDisc && genAdvBody) {
            genAdvDisc.addEventListener('click', function () {
                this.classList.toggle('open');
                genAdvBody.classList.toggle('open');
            });
        }
        // Steps sync
        els.steps.addEventListener('input', function () {
            state.steps = parseInt(this.value) || 20;
            els.stepsRange.value = this.value;
        });
        els.stepsRange.addEventListener('input', function () {
            state.steps = parseInt(this.value);
            els.steps.value = this.value;
        });
        // CFG sync
        els.cfg.addEventListener('input', function () {
            var parsed = parseFloat(this.value);
            state.cfg = Number.isFinite(parsed) ? parsed : 7.0;
            els.cfgRange.value = this.value;
        });
        els.cfgRange.addEventListener('input', function () {
            state.cfg = parseFloat(this.value);
            els.cfg.value = this.value;
        });
        // Scheduler
        if (els.sampler) {
            els.sampler.addEventListener('change', function () {
                state.sampler = this.value;
            });
        }
        els.scheduler.addEventListener('change', function () {
            state.scheduler = this.value;
        });
        bindSliderPair('gen-variation-strength-range', 'gen-variation-strength', function (v) {
            state.variationStrength = Math.max(0, Math.min(1, parseFloat(v) || 0));
        });
        var variationSeed = document.getElementById('gen-variation-seed');
        if (variationSeed) {
            variationSeed.addEventListener('input', function () {
                state.variationSeed = Math.max(0, parseInt(this.value) || 0);
            });
        }
        var variationShuffle = document.getElementById('gen-variation-shuffle');
        if (variationShuffle) {
            variationShuffle.addEventListener('click', function () {
                state.variationSeed = Math.floor(Math.random() * 4294967296);
                if (variationSeed)
                    variationSeed.value = String(state.variationSeed);
            });
        }
        bindSliderPair('gen-creativity-range', 'gen-creativity', function (v) {
            state.creativity = Math.max(0, Math.min(1, parseFloat(v) || 0));
        });
        bindInitImage();
        // Model searchable picker
        bindModelPicker();
        // Model refresh button
        var modelRefresh = document.getElementById('gen-model-refresh');
        if (modelRefresh) {
            modelRefresh.addEventListener('click', function (e) {
                e.stopPropagation();
                loadModels();
            });
        }
        // Guidance sync (FLUX)
        els.guidance.addEventListener('input', function () {
            state.guidance = parseFloat(this.value) || 3.5;
            els.guidanceRange.value = this.value;
        });
        els.guidanceRange.addEventListener('input', function () {
            state.guidance = parseFloat(this.value);
            els.guidance.value = this.value;
        });
        // Aspect dropdown
        if (els.aspectDropdown) {
            els.aspectDropdown.addEventListener('change', function () {
                var val = this.value;
                if (val === 'custom' || val === 'Free')
                    return; // Free = keep current W/H
                var aspects = getActiveAspects();
                for (var i = 0; i < aspects.length; i++) {
                    if (aspects[i].label === val) {
                        state.width = aspects[i].w;
                        state.height = aspects[i].h;
                        syncDimensionInputs();
                        updateAspectPreview();
                        break;
                    }
                }
            });
        }
        // Swap button
        var swapBtn = document.getElementById('gen-swap-btn');
        if (swapBtn) {
            swapBtn.addEventListener('click', function () {
                var tmp = state.width;
                state.width = state.height;
                state.height = tmp;
                syncDimensionInputs();
                syncAspectDropdown();
                updateAspectPreview();
            });
        }
        // Aspect lock
        var lockBtn = document.getElementById('gen-aspect-lock');
        if (lockBtn) {
            lockBtn.addEventListener('click', function () {
                state.aspectLocked = !state.aspectLocked;
                this.classList.toggle('active', state.aspectLocked);
                if (state.aspectLocked) {
                    state.lockedRatio = state.width / state.height;
                }
            });
        }
        // Optimal size button
        var optBtn = document.getElementById('gen-optimal-btn');
        if (optBtn) {
            optBtn.addEventListener('click', function () {
                var defaults = getDefaultsForArch(state.arch);
                state.width = defaults.w;
                state.height = defaults.h;
                syncDimensionInputs();
                syncAspectDropdown();
                updateAspectPreview();
            });
        }
        // Custom resolution inputs
        els.customWidth.addEventListener('blur', function () {
            var isVideo = ModelUtils.isVideoModel(state.model);
            var v = isVideo
                ? ModelUtils.clampVideoDimension(parseInt(this.value) || 512)
                : ModelUtils.clampDimension(parseInt(this.value) || 1024);
            this.value = String(v);
            state.width = v;
            var wsl = document.getElementById('gen-width-slider');
            if (wsl)
                wsl.value = String(v);
            if (state.aspectLocked && state.lockedRatio) {
                var newH = isVideo
                    ? ModelUtils.clampVideoDimension(Math.round(v / state.lockedRatio))
                    : ModelUtils.clampDimension(Math.round(v / state.lockedRatio));
                state.height = newH;
                if (els.customHeight)
                    els.customHeight.value = String(newH);
                var hsl = document.getElementById('gen-height-slider');
                if (hsl)
                    hsl.value = String(newH);
            }
            syncAspectDropdown();
            updateAspectPreview();
        });
        els.customHeight.addEventListener('blur', function () {
            var isVideo = ModelUtils.isVideoModel(state.model);
            var v = isVideo
                ? ModelUtils.clampVideoDimension(parseInt(this.value) || 512)
                : ModelUtils.clampDimension(parseInt(this.value) || 1024);
            this.value = String(v);
            state.height = v;
            var hsl = document.getElementById('gen-height-slider');
            if (hsl)
                hsl.value = String(v);
            if (state.aspectLocked && state.lockedRatio) {
                var newW = isVideo
                    ? ModelUtils.clampVideoDimension(Math.round(v * state.lockedRatio))
                    : ModelUtils.clampDimension(Math.round(v * state.lockedRatio));
                state.width = newW;
                if (els.customWidth)
                    els.customWidth.value = String(newW);
                var wsl = document.getElementById('gen-width-slider');
                if (wsl)
                    wsl.value = String(newW);
            }
            syncAspectDropdown();
            updateAspectPreview();
        });
        // Width slider sync
        var widthSlider = document.getElementById('gen-width-slider');
        if (widthSlider) {
            widthSlider.addEventListener('input', function () {
                var v = parseInt(this.value);
                state.width = v;
                if (els.customWidth)
                    els.customWidth.value = String(v);
                if (state.aspectLocked && state.lockedRatio) {
                    var isVideo = ModelUtils.isVideoModel(state.model);
                    var newH = isVideo
                        ? ModelUtils.clampVideoDimension(Math.round(v / state.lockedRatio))
                        : ModelUtils.clampDimension(Math.round(v / state.lockedRatio));
                    state.height = newH;
                    if (els.customHeight)
                        els.customHeight.value = String(newH);
                    var hs = document.getElementById('gen-height-slider');
                    if (hs)
                        hs.value = String(newH);
                }
                syncAspectDropdown();
                updateAspectPreview();
            });
        }
        var heightSlider = document.getElementById('gen-height-slider');
        if (heightSlider) {
            heightSlider.addEventListener('input', function () {
                var v = parseInt(this.value);
                state.height = v;
                if (els.customHeight)
                    els.customHeight.value = String(v);
                if (state.aspectLocked && state.lockedRatio) {
                    var isVideo = ModelUtils.isVideoModel(state.model);
                    var newW = isVideo
                        ? ModelUtils.clampVideoDimension(Math.round(v * state.lockedRatio))
                        : ModelUtils.clampDimension(Math.round(v * state.lockedRatio));
                    state.width = newW;
                    if (els.customWidth)
                        els.customWidth.value = String(newW);
                    var ws = document.getElementById('gen-width-slider');
                    if (ws)
                        ws.value = String(newW);
                }
                syncAspectDropdown();
                updateAspectPreview();
            });
        }
        // Seed
        els.seed.addEventListener('input', function () {
            state.seed = parseInt(this.value);
        });
        els.seedShuffle.addEventListener('click', function () {
            var s = Math.floor(Math.random() * 4294967296);
            state.seed = s;
            if (els.seed)
                els.seed.value = String(s);
        });
        // Seed previous
        var seedPrev = document.getElementById('gen-seed-prev');
        if (seedPrev) {
            seedPrev.addEventListener('click', function () {
                if (state.lastSeed !== null) {
                    state.seed = state.lastSeed;
                    if (els.seed)
                        els.seed.value = String(state.lastSeed);
                }
            });
        }
        // Seed random toggle
        var randomToggle = document.getElementById('gen-seed-random-toggle');
        if (randomToggle) {
            randomToggle.addEventListener('click', function () {
                var isRandom = state.seed !== -1;
                if (isRandom) {
                    state.seed = -1;
                    if (els.seed)
                        els.seed.value = String(-1);
                }
                else {
                    state.seed = Math.floor(Math.random() * 4294967296);
                    if (els.seed)
                        els.seed.value = String(state.seed);
                }
                this.classList.toggle('on', state.seed === -1);
            });
        }
        // Batch count (left panel)
        var batchInput = document.getElementById('gen-batch');
        if (batchInput) {
            batchInput.addEventListener('input', function () {
                state.batchCount = Math.max(1, Math.min(8, parseInt(this.value) || 1));
                if (els.toolbarBatchInput)
                    els.toolbarBatchInput.value = String(state.batchCount);
            });
        }
        // Batch count (toolbar)
        if (els.toolbarBatchInput) {
            els.toolbarBatchInput.addEventListener('input', function () {
                state.batchCount = Math.max(1, Math.min(8, parseInt(this.value) || 1));
                if (batchInput)
                    batchInput.value = String(state.batchCount);
            });
        }
        var batchUp = document.getElementById('gen-toolbar-batch-up');
        var batchDown = document.getElementById('gen-toolbar-batch-down');
        if (batchUp) {
            batchUp.addEventListener('click', function () {
                state.batchCount = Math.min(8, state.batchCount + 1);
                if (els.toolbarBatchInput)
                    els.toolbarBatchInput.value = String(state.batchCount);
                if (batchInput)
                    batchInput.value = String(state.batchCount);
            });
        }
        if (batchDown) {
            batchDown.addEventListener('click', function () {
                state.batchCount = Math.max(1, state.batchCount - 1);
                if (els.toolbarBatchInput)
                    els.toolbarBatchInput.value = String(state.batchCount);
                if (batchInput)
                    batchInput.value = String(state.batchCount);
            });
        }
        // LoRA picker (concepts dropdown)
        var loraPicker = document.getElementById('gen-lora-picker');
        if (loraPicker) {
            loraPicker.addEventListener('change', function () {
                if (this.value && this.selectedIndex > 0) {
                    addLora(this.value);
                    this.selectedIndex = 0;
                }
            });
        }
        // Seconds → compute frames
        els.secondsInput.addEventListener('input', function () {
            state.seconds = Math.max(1, Math.min(30, parseInt(this.value) || 10));
            state.frames = secondsToFrames(state.seconds, state.fps);
            updateDurationHint();
        });
        // FPS sync — recompute frames from seconds
        els.fpsInput.addEventListener('input', function () {
            state.fps = parseInt(this.value) || 24;
            els.fpsRange.value = this.value;
            state.frames = secondsToFrames(state.seconds, state.fps);
            updateDurationHint();
        });
        els.fpsRange.addEventListener('input', function () {
            state.fps = parseInt(this.value);
            els.fpsInput.value = this.value;
            state.frames = secondsToFrames(state.seconds, state.fps);
            updateDurationHint();
        });
        // Generate (left panel button)
        els.btn.addEventListener('click', function () {
            generate();
        });
        // Action bar
        els.download.addEventListener('click', function () {
            if (!state.currentImage)
                return;
            var a = document.createElement('a');
            a.href = state.currentImage;
            var ext = state.currentIsVideo ? '.mp4' : '.png';
            a.download = 'serenityflow_' + Date.now() + ext;
            a.click();
        });
        els.clearPreview.addEventListener('click', function () {
            clearPreview();
        });
        // Send to Timeline
        var toTimelineBtn = document.getElementById('gen-to-timeline');
        if (toTimelineBtn) {
            toTimelineBtn.addEventListener('click', function () {
                if (!state.currentImage) return;
                if (typeof VideoEditTab === 'undefined') {
                    alert('Open the Video Edit tab first.');
                    return;
                }
                var pid = VideoEditTab.getActiveProjectId();
                if (!pid) {
                    alert('Open the Video Edit tab first to create a project.');
                    return;
                }
                var label = 'Generated';
                try {
                    var promptEl = document.getElementById('gen-prompt');
                    if (promptEl && promptEl.value) {
                        label = promptEl.value.slice(0, 30).trim() || 'Generated';
                    }
                } catch (e) {}
                var timelineFps = Math.max(
                    1,
                    Math.min(120, Math.round(Number(state.currentVideoFps || state.fps) || 30))
                );
                var durationFrames = state.currentIsVideo
                    ? Math.max(1, Math.round(Number(state.currentVideoFrames) || Number(state.frames) || timelineFps))
                    : timelineFps * 3;

                // Resolve /view URL to a real filesystem path before passing to timeline
                var viewUrl = state.currentImage;
                var params = {};
                try {
                    var urlObj = new URL(viewUrl, window.location.origin);
                    urlObj.searchParams.forEach(function (v, k) { params[k] = v; });
                } catch (e) {
                    // Not a URL — pass as-is (may be a direct path)
                    VideoEditTab.addClipFromExternal(viewUrl, label, durationFrames, timelineFps);
                    if (typeof switchTab === 'function') switchTab('video-edit');
                    return;
                }

                if (params.filename) {
                    fetch('/video_edit/resolve_view_path', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            filename: params.filename,
                            subfolder: params.subfolder || '',
                            type: params.type || 'output',
                        }),
                    })
                    .then(function (r) { return r.json(); })
                    .then(function (data) {
                        if (data.path) {
                            VideoEditTab.addClipFromExternal(data.path, label, durationFrames, timelineFps);
                        } else {
                            // Fallback: use the URL as-is
                            VideoEditTab.addClipFromExternal(viewUrl, label, durationFrames, timelineFps);
                        }
                        if (typeof switchTab === 'function') switchTab('video-edit');
                    })
                    .catch(function () {
                        VideoEditTab.addClipFromExternal(viewUrl, label, durationFrames, timelineFps);
                        if (typeof switchTab === 'function') switchTab('video-edit');
                    });
                } else {
                    VideoEditTab.addClipFromExternal(viewUrl, label, durationFrames, timelineFps);
                    if (typeof switchTab === 'function') switchTab('video-edit');
                }
            });
        }
        // Gallery clear
        els.galleryClear.addEventListener('click', function () {
            if (state.gallery.length === 0)
                return;
            if (!confirm('Clear all ' + state.gallery.length + ' gallery items?'))
                return;
            state.gallery = [];
            state.selectedImages = [];
            state.galleryPage = 0;
            renderGallery();
            updateSelectionUI();
            updateMetadataPanel();
            localStorage.removeItem('sf-gallery');
        });
        // ── Top Toolbar ──
        var toolbarGenerate = document.getElementById('gen-toolbar-generate');
        if (toolbarGenerate) {
            toolbarGenerate.addEventListener('click', function () { generate(); });
        }
        var toolbarDelete = document.getElementById('gen-toolbar-delete');
        if (toolbarDelete) {
            toolbarDelete.addEventListener('click', function () { clearPreview(); });
        }
        var toolbarToggleGallery = document.getElementById('gen-toolbar-toggle-gallery');
        if (toolbarToggleGallery) {
            toolbarToggleGallery.addEventListener('click', function () {
                state.rightPanelVisible = !state.rightPanelVisible;
                els.rightPanel.classList.toggle('gen-panel-hidden', !state.rightPanelVisible);
                this.classList.toggle('active', state.rightPanelVisible);
            });
        }
        var toolbarCopy = document.getElementById('gen-toolbar-copy');
        if (toolbarCopy) {
            toolbarCopy.addEventListener('click', function () {
                if (state.currentImage) {
                    var fullUrl = window.location.origin + state.currentImage;
                    navigator.clipboard.writeText(fullUrl).catch(function () { });
                }
            });
        }
        // ── Floating Toolbar ──
        var floatToggleLeft = document.getElementById('gen-float-toggle-left');
        if (floatToggleLeft) {
            floatToggleLeft.addEventListener('click', function () {
                state.leftPanelVisible = !state.leftPanelVisible;
                els.leftPanel.classList.toggle('gen-panel-hidden', !state.leftPanelVisible);
                this.classList.toggle('active', state.leftPanelVisible);
            });
        }
        var floatGenerate = document.getElementById('gen-float-generate');
        if (floatGenerate) {
            floatGenerate.addEventListener('click', function () { generate(); });
        }
        var floatCancel = document.getElementById('gen-float-cancel');
        if (floatCancel) {
            floatCancel.addEventListener('click', function () {
                if (state.generating) {
                    state.pendingBatch = 0;
                    SerenityAPI.interrupt();
                    setGenerating(false);
                }
            });
        }
        var floatDelete = document.getElementById('gen-float-delete');
        if (floatDelete) {
            floatDelete.addEventListener('click', function () { clearPreview(); });
        }
        // ── Gallery Tabs ──
        var tabLayers = document.getElementById('gen-tab-layers');
        var tabGallery = document.getElementById('gen-tab-gallery');
        var layersContent = document.getElementById('gen-layers-content');
        var galleryContent = document.getElementById('gen-gallery-content');
        if (tabLayers && tabGallery) {
            tabLayers.addEventListener('click', function () {
                state.galleryTab = 'layers';
                tabLayers.classList.add('active');
                tabGallery.classList.remove('active');
                layersContent.classList.add('active');
                galleryContent.classList.remove('active');
            });
            tabGallery.addEventListener('click', function () {
                state.galleryTab = 'gallery';
                tabGallery.classList.add('active');
                tabLayers.classList.remove('active');
                galleryContent.classList.add('active');
                layersContent.classList.remove('active');
            });
        }
        // Gallery close
        var galleryClose = document.getElementById('gen-gallery-close');
        if (galleryClose) {
            galleryClose.addEventListener('click', function () {
                state.rightPanelVisible = false;
                els.rightPanel.classList.add('gen-panel-hidden');
                var galleryToggle = document.getElementById('gen-toolbar-toggle-gallery');
                if (galleryToggle)
                    galleryToggle.classList.remove('active');
            });
        }
        // Boards toggle
        var boardsHeader = document.getElementById('gen-boards-header');
        var boardsBody = document.getElementById('gen-boards-body');
        if (boardsHeader && boardsBody) {
            boardsHeader.addEventListener('click', function () {
                state.boardsVisible = !state.boardsVisible;
                boardsBody.classList.toggle('closed', !state.boardsVisible);
                boardsHeader.classList.toggle('closed', !state.boardsVisible);
                boardsHeader.querySelector('span:first-child').textContent = state.boardsVisible ? 'Hide Boards' : 'Show Boards';
            });
        }
        // Sub-tabs
        var subtabImages = document.getElementById('gen-subtab-images');
        var subtabAssets = document.getElementById('gen-subtab-assets');
        var imagesContent = document.getElementById('gen-images-content');
        var assetsContent = document.getElementById('gen-assets-content');
        if (subtabImages && subtabAssets) {
            subtabImages.addEventListener('click', function () {
                state.gallerySubTab = 'images';
                subtabImages.classList.add('active');
                subtabAssets.classList.remove('active');
                imagesContent.style.display = '';
                assetsContent.style.display = 'none';
            });
            subtabAssets.addEventListener('click', function () {
                state.gallerySubTab = 'assets';
                subtabAssets.classList.add('active');
                subtabImages.classList.remove('active');
                assetsContent.style.display = '';
                imagesContent.style.display = 'none';
            });
        }
        // Gallery search toggle + filtering
        var searchBtn = document.getElementById('gen-gallery-search-btn');
        var searchInput = document.getElementById('gen-gallery-search-input');
        if (searchBtn && searchInput) {
            searchBtn.addEventListener('click', function () {
                searchInput.classList.toggle('open');
                if (searchInput.classList.contains('open')) {
                    searchInput.focus();
                }
                else {
                    searchInput.value = '';
                    state.gallerySearch = '';
                    state.galleryPage = 0;
                    renderGallery();
                }
            });
            searchInput.addEventListener('input', function () {
                state.gallerySearch = this.value;
                state.galleryPage = 0;
                renderGallery();
            });
        }
        // ── Style Preset ──
        var styleSelect = document.getElementById('gen-style-preset');
        if (styleSelect) {
            styleSelect.addEventListener('change', function () {
                state.stylePreset = this.value;
                updateStylePreview();
            });
        }
        // ── Prompt History (Alt+Up/Down) ──
        if (els.prompt) {
            els.prompt.addEventListener('keydown', function (e) {
                // Prompt attention weight (Ctrl+Up/Down)
                if (e.ctrlKey && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) {
                    e.preventDefault();
                    handlePromptWeight(this, e.key === 'ArrowUp');
                    return;
                }
                // Prompt history (Alt+Up/Down)
                if (e.altKey && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) {
                    e.preventDefault();
                    if (e.key === 'ArrowUp') {
                        navigatePromptHistory(-1);
                    }
                    else {
                        navigatePromptHistory(1);
                    }
                }
            });
        }
        bindSwarmControls();
    }
    function bindInitImage() {
        var input = document.getElementById('gen-init-image-input');
        var drop = document.getElementById('gen-init-drop');
        var clear = document.getElementById('gen-init-clear');
        if (!input || !drop)
            return;
        function upload(file) {
            if (!file || !/^image\//.test(file.type || '')) {
                showError('Choose a readable image file');
                return;
            }
            var form = new FormData();
            form.append('image', file, file.name || 'init.png');
            var preview = document.getElementById('gen-init-preview');
            var empty = document.getElementById('gen-init-empty');
            var name = document.getElementById('gen-init-name');
            if (preview) {
                preview.src = URL.createObjectURL(file);
                preview.style.display = 'block';
            }
            if (empty)
                empty.style.display = 'none';
            if (name)
                name.textContent = 'Uploading ' + (file.name || 'image') + '…';
            fetch('/upload/image', { method: 'POST', body: form })
                .then(function (resp) {
                return resp.text().then(function (body) {
                    var data = {};
                    try { data = JSON.parse(body); }
                    catch (e) { data = { error: body }; }
                    if (!resp.ok)
                        throw new Error(data.error || data.detail || ('HTTP ' + resp.status));
                    return data;
                });
            })
                .then(function (data) {
                state.initImagePath = data.path || data.name || '';
                state.initImageName = data.name || file.name || state.initImagePath;
                if (!state.initImagePath)
                    throw new Error('upload returned no image path');
                if (name)
                    name.textContent = state.initImageName;
                if (clear)
                    clear.disabled = false;
            })
                .catch(function (err) {
                clearInitImage();
                showError('Init image upload failed: ' + err.message);
            });
        }
        input.addEventListener('change', function () {
            upload(this.files && this.files[0]);
        });
        ['dragenter', 'dragover'].forEach(function (eventName) {
            drop.addEventListener(eventName, function (event) {
                event.preventDefault();
                drop.classList.add('dragover');
            });
        });
        ['dragleave', 'drop'].forEach(function (eventName) {
            drop.addEventListener(eventName, function (event) {
                event.preventDefault();
                drop.classList.remove('dragover');
            });
        });
        drop.addEventListener('drop', function (event) {
            upload(event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0]);
        });
        if (clear)
            clear.addEventListener('click', clearInitImage);
    }
    function clearInitImage() {
        state.initImagePath = '';
        state.initImageName = '';
        var input = document.getElementById('gen-init-image-input');
        var preview = document.getElementById('gen-init-preview');
        var empty = document.getElementById('gen-init-empty');
        var name = document.getElementById('gen-init-name');
        var clear = document.getElementById('gen-init-clear');
        if (input)
            input.value = '';
        if (preview) {
            if (preview.src && preview.src.indexOf('blob:') === 0)
                URL.revokeObjectURL(preview.src);
            preview.removeAttribute('src');
            preview.style.display = 'none';
        }
        if (empty)
            empty.style.display = '';
        if (name)
            name.textContent = 'No source image';
        if (clear)
            clear.disabled = true;
    }
    function bindSwarmControls() {
        ['gen-core', 'gen-variation', 'gen-sampling', 'gen-init', 'gen-lora'].forEach(function (prefix) {
            bindAccordion(prefix + '-header', prefix + '-body');
        });
        var filter = document.getElementById('gen-param-filter');
        if (filter) {
            filter.addEventListener('input', function () {
                var query = this.value.trim().toLowerCase();
                document.querySelectorAll('#gen-left-panel .gen-param-group').forEach(function (group) {
                    var haystack = group.getAttribute('data-param-search') || '';
                    var rows = Array.prototype.slice.call(group.querySelectorAll('.gen-param-row'));
                    var groupMatch = !query || haystack.indexOf(query) >= 0;
                    var rowMatch = false;
                    rows.forEach(function (row) {
                        var visible = !query || groupMatch ||
                            (row.getAttribute('data-param-search') || '').indexOf(query) >= 0 ||
                            row.textContent.toLowerCase().indexOf(query) >= 0;
                        row.style.display = visible ? '' : 'none';
                        rowMatch = rowMatch || visible;
                    });
                    group.style.display = (!query || groupMatch || rowMatch) ? '' : 'none';
                    if (query && (groupMatch || rowMatch)) {
                        var body = group.querySelector('.gen-swarm-group-body');
                        var header = group.querySelector('.gen-swarm-group-title');
                        if (body) body.classList.remove('closed');
                        if (header) header.classList.remove('closed');
                    }
                });
            });
        }
        var reset = document.getElementById('gen-quick-reset');
        if (reset) {
            reset.addEventListener('click', function () {
                state.variationSeed = 0;
                state.variationStrength = 0;
                clearInitImage();
                updateUIForArch(state.arch);
                var variationSeed = document.getElementById('gen-variation-seed');
                var variationStrength = document.getElementById('gen-variation-strength');
                var variationRange = document.getElementById('gen-variation-strength-range');
                if (variationSeed) variationSeed.value = '0';
                if (variationStrength) variationStrength.value = '0';
                if (variationRange) variationRange.value = '0';
            });
        }
        document.querySelectorAll('.gen-library-tab').forEach(function (button) {
            button.addEventListener('click', function () {
                var target = this.dataset.library;
                document.querySelectorAll('.gen-library-tab').forEach(function (tab) {
                    tab.classList.toggle('active', tab.dataset.library === target);
                });
                document.querySelectorAll('.gen-library-panel').forEach(function (panel) {
                    panel.classList.toggle('active', panel.dataset.libraryPanel === target);
                });
                if (target === 'presets')
                    loadGenerationPresets();
                if (target === 'models')
                    renderModelLibrary();
                if (target === 'loras')
                    renderLoraLibrary();
            });
        });
        var presetSave = document.getElementById('gen-preset-save');
        if (presetSave) {
            presetSave.addEventListener('click', function () {
                var input = document.getElementById('gen-preset-name');
                var name = input ? input.value.trim() : '';
                if (!name) {
                    showError('Enter a preset name');
                    return;
                }
                fetch('/v1/presets', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name: name, params: getParams() })
                }).then(function (resp) {
                    if (!resp.ok)
                        return resp.text().then(function (text) { throw new Error(text || ('HTTP ' + resp.status)); });
                    return resp.json();
                }).then(function () {
                    if (input) input.value = '';
                    loadGenerationPresets();
                }).catch(function (err) {
                    showError('Preset save failed: ' + err.message);
                });
            });
        }
    }
    function loadGenerationPresets() {
        fetch('/v1/presets', { cache: 'no-store' })
            .then(function (resp) {
            if (!resp.ok)
                throw new Error('HTTP ' + resp.status);
            return resp.json();
        })
            .then(function (doc) {
            var list = document.getElementById('gen-preset-list');
            if (!list)
                return;
            list.innerHTML = '';
            var presets = Array.isArray(doc.presets) ? doc.presets : [];
            if (!presets.length) {
                list.innerHTML = '<div class="gen-library-empty">No saved presets</div>';
                return;
            }
            presets.forEach(function (preset) {
                var card = document.createElement('div');
                card.className = 'gen-library-card';
                var title = document.createElement('strong');
                title.textContent = preset.name || 'Unnamed';
                var load = document.createElement('button');
                load.className = 'gen-small-btn';
                load.textContent = 'Load';
                load.addEventListener('click', function () { applyParams(preset.params || {}); });
                var del = document.createElement('button');
                del.className = 'gen-small-btn destructive';
                del.textContent = 'Delete';
                del.addEventListener('click', function () {
                    fetch('/v1/presets/' + encodeURIComponent(preset.name), { method: 'DELETE' })
                        .then(function (resp) {
                        if (!resp.ok)
                            throw new Error('HTTP ' + resp.status);
                        loadGenerationPresets();
                    }).catch(function (err) { showError('Preset delete failed: ' + err.message); });
                });
                card.appendChild(title);
                card.appendChild(load);
                card.appendChild(del);
                list.appendChild(card);
            });
        })
            .catch(function (err) {
            showError('Preset load failed: ' + err.message);
        });
    }
    function renderModelLibrary() {
        var list = document.getElementById('gen-library-models');
        if (!list)
            return;
        list.innerHTML = '';
        (state.allModels || []).forEach(function (model) {
            var card = document.createElement('button');
            card.className = 'gen-library-card gen-library-model-card';
            card.type = 'button';
            card.textContent = model.name;
            card.title = model.name;
            card.classList.toggle('active', model.name === state.model);
            card.addEventListener('click', function () { selectModel(model.name); renderModelLibrary(); });
            list.appendChild(card);
        });
        if (!list.children.length)
            list.innerHTML = '<div class="gen-library-empty">No admitted image models found</div>';
    }
    function renderLoraLibrary() {
        var list = document.getElementById('gen-library-loras');
        var picker = document.getElementById('gen-lora-picker');
        if (!list)
            return;
        list.innerHTML = '';
        if (picker) {
            Array.prototype.slice.call(picker.options).forEach(function (option, index) {
                if (index === 0 || !option.value)
                    return;
                var card = document.createElement('button');
                card.className = 'gen-library-card gen-library-model-card';
                card.type = 'button';
                card.textContent = option.textContent;
                card.addEventListener('click', function () { addLora(option.value); });
                list.appendChild(card);
            });
        }
        if (!list.children.length)
            list.innerHTML = '<div class="gen-library-empty">No local LoRAs found</div>';
    }
    function bindAccordion(headerId, bodyId) {
        var header = document.getElementById(headerId);
        var body = document.getElementById(bodyId);
        if (header && body) {
            header.addEventListener('click', function () {
                this.classList.toggle('closed');
                body.classList.toggle('closed');
            });
        }
    }
    // Helper: bind a range slider ↔ number input pair with state callback
    function bindSliderPair(rangeId, inputId, onUpdate) {
        var range = document.getElementById(rangeId);
        var input = document.getElementById(inputId);
        if (range && input) {
            range.addEventListener('input', function () {
                input.value = this.value;
                if (onUpdate)
                    onUpdate(this.value);
            });
            input.addEventListener('input', function () {
                var v = parseFloat(this.value);
                if (!isNaN(v)) {
                    var min = parseFloat(range.min);
                    var max = parseFloat(range.max);
                    range.value = String(Math.min(max, Math.max(min, v)));
                    if (onUpdate)
                        onUpdate(this.value);
                }
            });
        }
    }
    function syncDimensionInputs() {
        if (els.customWidth)
            els.customWidth.value = String(state.width);
        if (els.customHeight)
            els.customHeight.value = String(state.height);
        var ws = document.getElementById('gen-width-slider');
        var hs = document.getElementById('gen-height-slider');
        if (ws)
            ws.value = String(state.width);
        if (hs)
            hs.value = String(state.height);
    }
    function getDefaultsForArch(arch) {
        var defaults = {
            sd15: { w: 512, h: 512 },
            sdxl: { w: 1024, h: 1024 },
            sd3: { w: 1024, h: 1024 },
            flux: { w: 1024, h: 1024 },
            klein: { w: 1024, h: 1024 },
            krea2: { w: 1024, h: 1024 },
            qwen: { w: 1024, h: 1024 },
            ltxv: { w: 1920, h: 1088 },
            wan: { w: 832, h: 480 },
            bernini: { w: 848, h: 480 },
            scail2: { w: 896, h: 512 }
        };
        return defaults[arch] || { w: 1024, h: 1024 };
    }
    function activeCapabilityProfile(arch) {
        var resolvedArch = arch || state.arch;
        var backend = ModelUtils.backendForArch(resolvedArch);
        if (!backend || !state.capabilities || !Array.isArray(state.capabilities.backends))
            return null;
        return state.capabilities.backends.find(function (entry) {
            return entry && entry.backend === backend;
        }) || null;
    }
    function featureSupported(profile, name) {
        return !!(profile && profile.features && profile.features[name] &&
            profile.features[name].supported === true);
    }
    function variationSupportedForArch(arch) {
        // These are the worker implementations that actually blend secondary
        // noise. Other families reject or ignore the fields and must not expose
        // the control.
        return ['zimage', 'sdxl', 'sd3', 'flux', 'chroma'].indexOf(arch) >= 0;
    }
    function displaySamplerName(name) {
        var names = {
            euler: 'Euler',
            flowmatch_euler: 'FlowMatch Euler',
            dpmpp_2m: 'DPM++ 2M',
            uni_pc: 'UniPC',
            uni_pc_bh2: 'UniPC BH2'
        };
        return names[name] || String(name || '').replace(/_/g, ' ');
    }
    function displaySchedulerName(name) {
        var names = {
            simple: 'Simple',
            normal: 'Normal',
            sgm_uniform: 'SGM Uniform',
            ideogram_logitnormal: 'Ideogram Logit-Normal',
            flux2: 'FLUX.2'
        };
        return names[name] || String(name || '').replace(/_/g, ' ');
    }
    // ── Model Loading ──
    function loadModels() {
        Promise.all([ModelUtils.fetchAllModels(), ModelUtils.loadCapabilities()])
            .then(function (loaded) {
            var models = loaded[0].filter(function (model) {
                return !ModelUtils.isVideoModel(model.name);
            });
            state.capabilities = loaded[1];
            if (!models.length)
                throw new Error('empty');
            state.allModels = models;
            // Default to a KNOWN-GOOD image family, not models[0] blindly —
            // an alphabetical/scan-ordered first entry can be a VIDEO model
            // (LTX/Wan), whose graph the image queue rejects with 501
            // "workflow route 'video'" on every Generate click.
            // Krea Raw remains selectable as the training/base checkpoint, but
            // the creator's distilled Turbo checkpoint is the inference default.
            var preferred = ['krea2-turbo', 'krea2', 'klein', 'flux2', 'zimage', 'ideogram', 'sdxl', 'sd3', 'qwen'];
            var pick = state.model ? models.find(function (model) { return model.name === state.model; }) : null;
            if (!pick) {
                pick = models[0];
                outer: for (var pi = 0; pi < preferred.length; pi++) {
                    for (var mi = 0; mi < models.length; mi++) {
                        var nm = (models[mi].name + ' ' + (models[mi].arch || '')).toLowerCase();
                        if (nm.indexOf(preferred[pi]) >= 0) { pick = models[mi]; break outer; }
                    }
                }
            }
            els.model.value = pick.name;
            state.model = pick.name;
            var globalBadge = document.querySelector('#topbar .model-badge');
            if (globalBadge) {
                globalBadge.textContent = pick.name;
                globalBadge.title = pick.name;
            }
            if (els.modelSearch) {
                els.modelSearch.value = pick.name;
                els.modelSearch.placeholder = 'Search models...';
            }
            updateUIForArch(ModelUtils.detectArchFromFilename(pick.name));
            els.modelWarn.classList.remove('visible');
        })
            .catch(function () {
            state.allModels = [];
            if (els.modelSearch) {
                els.modelSearch.value = '';
                els.modelSearch.placeholder = 'No models found';
            }
            els.modelWarn.textContent = 'Could not load models or server capabilities';
            els.modelWarn.classList.add('visible');
        });
    }
    // ── Load VAEs and refiner models from /object_info ──
    function loadAdvancedOptions() {
        fetch('/object_info', { cache: 'no-store' })
            .then(function (r) { return r.ok ? r.json() : {}; })
            .then(function (info) {
            // Populate VAE picker
            var vaePicker = document.getElementById('gen-vae-picker');
            if (vaePicker && info.VAELoader && info.VAELoader.input && info.VAELoader.input.required && info.VAELoader.input.required.vae_name) {
                var vaeList = info.VAELoader.input.required.vae_name[0];
                if (Array.isArray(vaeList)) {
                    vaeList.forEach(function (name) {
                        var opt = document.createElement('option');
                        opt.value = name;
                        opt.textContent = name;
                        vaePicker.appendChild(opt);
                    });
                }
            }
            // Populate refiner model picker (SDXL checkpoints)
            var refinerPicker = document.getElementById('gen-refiner-model');
            if (refinerPicker && info.CheckpointLoaderSimple && info.CheckpointLoaderSimple.input && info.CheckpointLoaderSimple.input.required && info.CheckpointLoaderSimple.input.required.ckpt_name) {
                var ckptList = info.CheckpointLoaderSimple.input.required.ckpt_name[0];
                if (Array.isArray(ckptList)) {
                    ckptList.forEach(function (name) {
                        var opt = document.createElement('option');
                        opt.value = name;
                        opt.textContent = name;
                        refinerPicker.appendChild(opt);
                    });
                }
            }
        })
            .catch(function () { });
    }
    function setPreviewModelBadges(model, arch) {
        // These badges describe the active request controls. Result-specific
        // identity is rendered in the metadata panel and must not silently
        // rewrite the selected model when an older History item is viewed.
        var resolvedModel = state.model || model || '';
        var resolvedArch = state.arch ||
            (resolvedModel && ModelUtils.detectArchFromFilename(resolvedModel)) ||
            arch || 'other';
        var archBadge = document.getElementById('gen-arch-badge');
        if (archBadge) {
            var archNames = { flux: 'FLUX', sdxl: 'SDXL', anima: 'ANIMA', sd3: 'SD3', sd15: 'SD1.5', ltxv: 'LTX-V', wan: 'WAN', bernini: 'BERNINI-R', scail2: 'SCAIL-2', klein: 'KLEIN', krea2: 'KREA2', chroma: 'CHROMA', qwen: 'QWEN', zimage: 'Z-IMAGE', ideogram4: 'IDEOGRAM 4', sensenova: 'SENSENOVA' };
            archBadge.textContent = archNames[resolvedArch] || resolvedArch.toUpperCase();
            archBadge.dataset.arch = resolvedArch;
        }
        if (els.modelBadge) {
            var shortName = resolvedModel.length > 20 ? resolvedModel.substring(0, 18) + '...' : resolvedModel;
            els.modelBadge.textContent = shortName;
            els.modelBadge.style.display = resolvedModel ? '' : 'none';
            els.modelBadge.title = resolvedModel;
        }
    }
    // ── Arch-aware UI ──
    function updateUIForArch(arch) {
        updateSwarmUIForArch(arch);
        return;
        /* Legacy layout logic retained below only for source-map compatibility.
           The Swarm-style Generate surface uses the capability-driven path. */
        state.arch = arch;
        var isFlux = arch === 'flux' || arch === 'klein';
        var noNegative = isFlux || arch === 'ideogram4' || arch === 'sensenova';
        var isVideo = arch === 'ltxv' || arch === 'wan' || arch === 'bernini' || arch === 'scail2';
        // Each admitted product exposes only its measured sampler. Wan uses
        // creator-parity Flow-UniPC; image families and LTX keep their current
        // Euler-facing workflow control.
        if (els.scheduler) {
            var admittedScheduler = arch === 'wan' || arch === 'bernini' || arch === 'scail2'
                ? { value: 'uni_pc', label: arch === 'bernini' ? 'UniPC (Bernini-R)' : (arch === 'scail2' ? 'UniPC (SCAIL-2)' : 'UniPC (Wan)') }
                : { value: 'euler', label: 'Euler' };
            els.scheduler.innerHTML = '<option value="' + admittedScheduler.value + '">' +
                admittedScheduler.label + '</option>';
            state.scheduler = admittedScheduler.value;
            els.scheduler.value = state.scheduler;
        }
        // Krea creator semantics use cfg=0 to disable CFG (Turbo default).
        // Set the DOM bounds before assigning the model default so a range
        // input cannot silently clamp 0 back to 1.
        var cfgMin = arch === 'krea2' ? 0 : 1;
        if (els.cfg) els.cfg.min = String(cfgMin);
        if (els.cfgRange) els.cfgRange.min = String(cfgMin);
        var productDefaults = {
            zimage: { steps: 16, cfg: 5.0 }, qwen: { steps: 20, cfg: 4.0 },
            ideogram4: { steps: 20, cfg: 7.0 }, sdxl: { steps: 20, cfg: 7.0 },
            anima: { steps: 20, cfg: 4.5 }, sd3: { steps: 28, cfg: 4.5 },
            flux: { steps: 20, cfg: 4.0 },
            // BFL: undistilled Klein Base is 50-step / CFG 4; the distilled
            // production checkpoint is 4-step / guidance 1.
            klein: /base/i.test(state.model) ? { steps: 50, cfg: 4.0 } : { steps: 4, cfg: 1.0 },
            sensenova: { steps: 30, cfg: 4.0 },
            krea2: /turbo/i.test(state.model) ? { steps: 8, cfg: 0.0 } : { steps: 52, cfg: 3.5 },
            chroma: { steps: 30, cfg: 4.0 }
        };
        var defaults = productDefaults[arch];
        if (defaults) {
            state.steps = defaults.steps;
            state.cfg = defaults.cfg;
            if (els.steps) els.steps.value = String(state.steps);
            if (els.stepsRange) els.stepsRange.value = String(state.steps);
            if (els.cfg) els.cfg.value = String(state.cfg);
            if (els.cfgRange) els.cfgRange.value = String(state.cfg);
        }
        // Neg prompt: hide for flux only (video needs it for CFG)
        els.negSection.style.display = noNegative ? 'none' : 'block';
        if (noNegative)
            state.negPrompt = '';
        // CFG: hide for flux only (Wan's admitted profile uses CFG 5.0)
        els.cfgRow.style.display = isFlux ? 'none' : 'flex';
        // Guidance: flux only
        els.guidanceRow.style.display = isFlux ? 'flex' : 'none';
        // Video controls
        els.videoSection.style.display = isVideo ? 'block' : 'none';
        if (els.scail2Section)
            els.scail2Section.style.display = arch === 'scail2' ? 'block' : 'none';
        if (els.secondsInput) els.secondsInput.disabled = arch === 'scail2';
        if (els.fpsInput) els.fpsInput.disabled = arch === 'scail2';
        if (els.fpsRange) els.fpsRange.disabled = arch === 'scail2';
        // Batch: hide for video models
        var batchSection = document.getElementById('gen-batch-section');
        if (batchSection)
            batchSection.style.display = isVideo ? 'none' : 'flex';
        // Button label
        els.btn.innerHTML = isVideo
            ? '<i data-lucide="wand-2"></i> Generate Video'
            : '<i data-lucide="wand-2"></i> Generate';
        if (typeof lucide !== 'undefined')
            lucide.createIcons({ nameAttr: 'data-lucide' });
        setPreviewModelBadges(state.model, arch);
        // Refiner: SDXL only
        var refinerSection = document.getElementById('gen-refiner-section');
        if (refinerSection)
            refinerSection.style.display = (arch === 'sdxl') ? '' : 'none';
        // CLIP Skip: SD1.5 only
        var clipSkipRow = document.getElementById('gen-clip-skip-row');
        if (clipSkipRow)
            clipSkipRow.style.display = (arch === 'sd15') ? 'flex' : 'none';
        // Update custom res input constraints
        var ws = document.getElementById('gen-width-slider');
        var hs = document.getElementById('gen-height-slider');
        // Resolution specializations are compiled and parity-gated. Keep raw
        // dimensions read-only and drive them from the admitted shape picker.
        if (els.customWidth) els.customWidth.disabled = true;
        if (els.customHeight) els.customHeight.disabled = true;
        if (els.customWidth) els.customWidth.title = 'Use the Aspect picker for compiled model sizes';
        if (els.customHeight) els.customHeight.title = 'Use the Aspect picker for compiled model sizes';
        if (ws) ws.disabled = true;
        if (hs) hs.disabled = true;
        var swap = document.getElementById('gen-swap-btn');
        var lock = document.getElementById('gen-aspect-lock');
        var admittedAspects = getActiveAspects();
        var hasSwappedShape = admittedAspects.some(function (a) {
            return admittedAspects.some(function (b) { return a.w === b.h && a.h === b.w; });
        });
        if (swap) swap.style.display = admittedAspects.length > 1 && hasSwappedShape ? '' : 'none';
        if (lock) lock.style.display = 'none';
        // Update aspect dropdown options for video vs image
        if (els.aspectDropdown) {
            els.aspectDropdown.innerHTML = buildAspectOptions();
            els.aspectDropdown.value = '1:1';
        }
        // Select default aspect ratio for the new mode
        var aspects = admittedAspects;
        if (aspects.length > 0) {
            var preferredSize = getDefaultsForArch(arch);
            var defaultIdx = aspects.findIndex(function (a) {
                return a.w === preferredSize.w && a.h === preferredSize.h;
            });
            if (defaultIdx < 0)
                defaultIdx = 0;
            var defaultAspect = aspects[defaultIdx];
            state.width = defaultAspect.w;
            state.height = defaultAspect.h;
            syncDimensionInputs();
            syncAspectDropdown();
            updateAspectPreview();
        }
        // Set video-appropriate defaults
        if (isVideo) {
            state.cfg = arch === 'wan' ? 5.0 : (arch === 'bernini' ? 4.0 : (arch === 'scail2' ? 5.0 : 3.0));
            state.frames = arch === 'bernini' ? 81 : (arch === 'scail2' ? 65 : 121);
            state.fps = arch === 'bernini' || arch === 'scail2' ? 16 : 24;
            if (arch === 'scail2') {
                state.seconds = 4;
                if (els.secondsInput) els.secondsInput.value = '4';
            }
            if (arch === 'ltxv')
                state.steps = 15;
            if (arch === 'wan')
                state.steps = 50;
            if (arch === 'bernini')
                state.steps = 40;
            if (arch === 'scail2')
                state.steps = 40;
            if (els.steps) els.steps.value = String(state.steps);
            if (els.stepsRange) els.stepsRange.value = String(state.steps);
            if (els.fpsInput) els.fpsInput.value = String(state.fps);
            if (els.fpsRange) els.fpsRange.value = String(state.fps);
            if (els.cfg)
                els.cfg.value = String(state.cfg);
            if (els.cfgRange)
                els.cfgRange.value = String(state.cfg);
            updateDurationHint();
        }
    }
    function updateSwarmUIForArch(arch) {
        state.arch = arch;
        var profile = activeCapabilityProfile(arch);
        if (!profile) {
            if (els.modelWarn) {
                els.modelWarn.textContent = 'This model family is not admitted by the image runtime';
                els.modelWarn.classList.add('visible');
            }
            return;
        }
        if (els.modelWarn)
            els.modelWarn.classList.remove('visible');
        var defaults = profile.defaults || {};
        // Krea publishes one backend profile but has distinct Raw and Turbo
        // checkpoint defaults. Keep the checkpoint identity visible and exact.
        if (arch === 'krea2' && /turbo/i.test(state.model || '')) {
            defaults = Object.assign({}, defaults, { steps: 8, cfg: 0.0 });
        }
        state.steps = Number(defaults.steps) || 20;
        state.cfg = Number(defaults.cfg);
        if (!Number.isFinite(state.cfg))
            state.cfg = 4.5;
        state.sampler = defaults.sampler || 'euler';
        state.scheduler = defaults.scheduler || 'simple';
        if (els.steps) els.steps.value = String(state.steps);
        if (els.stepsRange) els.stepsRange.value = String(state.steps);
        if (els.cfg) els.cfg.value = String(state.cfg);
        if (els.cfgRange) els.cfgRange.value = String(state.cfg);
        if (els.cfg) els.cfg.min = '0';
        if (els.cfgRange) els.cfgRange.min = '0';
        var samplers = profile.samplers || {};
        var supportedSamplers = Array.isArray(samplers.supported_samplers)
            ? samplers.supported_samplers : [state.sampler];
        var supportedSchedulers = Array.isArray(samplers.supported_schedulers)
            ? samplers.supported_schedulers : [state.scheduler];
        if (els.sampler) {
            els.sampler.innerHTML = '';
            supportedSamplers.forEach(function (name) {
                var option = document.createElement('option');
                option.value = name;
                option.textContent = displaySamplerName(name);
                els.sampler.appendChild(option);
            });
            if (supportedSamplers.indexOf(state.sampler) < 0)
                state.sampler = supportedSamplers[0] || 'euler';
            els.sampler.value = state.sampler;
        }
        if (els.scheduler) {
            els.scheduler.innerHTML = '';
            supportedSchedulers.forEach(function (name) {
                var option = document.createElement('option');
                option.value = name;
                option.textContent = displaySchedulerName(name);
                els.scheduler.appendChild(option);
            });
            if (supportedSchedulers.indexOf(state.scheduler) < 0)
                state.scheduler = supportedSchedulers[0] || 'simple';
            els.scheduler.value = state.scheduler;
        }
        var negativeSupported = featureSupported(profile, 'negative_prompt');
        if (els.negSection)
            els.negSection.style.display = negativeSupported ? '' : 'none';
        if (!negativeSupported) {
            state.negPrompt = '';
            if (els.negPrompt) els.negPrompt.value = '';
        }
        if (els.cfgRow)
            els.cfgRow.style.display = featureSupported(profile, 'cfg') ? 'flex' : 'none';
        if (els.guidanceRow)
            els.guidanceRow.style.display = 'none';
        var variationSection = document.getElementById('gen-variation-section');
        if (variationSection)
            variationSection.style.display = variationSupportedForArch(arch) ? '' : 'none';
        if (!variationSupportedForArch(arch))
            state.variationStrength = 0;
        var initSection = document.getElementById('gen-init-section');
        var imageToImage = featureSupported(profile, 'image_to_image');
        if (initSection)
            initSection.style.display = imageToImage ? '' : 'none';
        if (!imageToImage)
            clearInitImage();
        var loraFeature = profile.features && profile.features.lora;
        var loraSupported = !!(loraFeature && loraFeature.supported);
        var loraSection = document.getElementById('gen-lora-section');
        if (loraSection)
            loraSection.style.display = loraSupported ? '' : 'none';
        var loraNote = document.getElementById('gen-lora-capability');
        var maxLoras = loraFeature && loraFeature.max_count;
        if (loraNote)
            loraNote.textContent = maxLoras == null
                ? 'Multiple LoRAs supported'
                : ('Maximum ' + maxLoras + ' LoRA' + (maxLoras === 1 ? '' : 's'));
        if (!loraSupported) {
            state.loras = [];
            renderLoraList();
        } else if (Number.isInteger(maxLoras) && state.loras.length > maxLoras) {
            state.loras = state.loras.slice(0, maxLoras);
            renderLoraList();
        }
        if (els.btn)
            els.btn.innerHTML = '<i data-lucide="wand-2"></i><span>Generate</span>';
        setPreviewModelBadges(state.model, arch);
        var runtimeLabel = document.getElementById('gen-runtime-label');
        if (runtimeLabel)
            runtimeLabel.textContent = profile.backend + ' · Mojo image worker · admitted';
        var aspects = getActiveAspects();
        if (els.aspectDropdown) {
            els.aspectDropdown.innerHTML = buildAspectOptions();
        }
        if (aspects.length) {
            var defaultWidth = Number(defaults.width);
            var defaultHeight = Number(defaults.height);
            var selected = aspects.find(function (size) {
                return size.w === defaultWidth && size.h === defaultHeight;
            }) || aspects[0];
            state.width = selected.w;
            state.height = selected.h;
            syncDimensionInputs();
            syncAspectDropdown();
            updateAspectPreview();
        }
        var swap = document.getElementById('gen-swap-btn');
        if (swap) {
            swap.style.display = aspects.some(function (size) {
                return aspects.some(function (other) {
                    return size.w === other.h && size.h === other.w;
                });
            }) ? '' : 'none';
        }
        renderModelLibrary();
        if (typeof lucide !== 'undefined')
            lucide.createIcons({ nameAttr: 'data-lucide' });
    }
    function updateDurationHint() {
        if (!els.durationHint)
            return;
        els.durationHint.textContent = state.frames + ' frames \u00b7 ' + state.seconds + 's at ' + state.fps + 'fps';
    }
    // ── Token Counter ──
    function updateTokenCount() {
        var count = state.prompt.trim() ? state.prompt.trim().split(/\s+/).length : 0;
        var el = document.getElementById('gen-token-count');
        if (el)
            el.textContent = '~' + count + ' tokens';
    }
    // ── LoRA Management ──
    function loadLoras() {
        fetch('/models/loras')
            .then(function (r) { return r.ok ? r.json() : []; })
            .then(function (list) {
            if (!list || !list.length)
                return;
            var picker = document.getElementById('gen-lora-picker');
            if (!picker)
                return;
            picker.innerHTML = '<option disabled selected>No LoRAs loaded</option>';
            list.forEach(function (name) {
                var opt = document.createElement('option');
                opt.value = name;
                opt.textContent = name;
                picker.appendChild(opt);
            });
        })
            .catch(function () { });
    }
    function addLora(name) {
        if (state.loras.some(function (l) { return l.name === name; }))
            return;
        var profile = activeCapabilityProfile();
        var feature = profile && profile.features && profile.features.lora;
        if (!feature || feature.supported !== true) {
            showError('The selected model does not admit LoRA overlays');
            return;
        }
        var maxCount = feature.max_count;
        if (Number.isInteger(maxCount) && state.loras.length >= maxCount) {
            showError('The selected model admits at most ' + maxCount + ' LoRA' + (maxCount === 1 ? '' : 's'));
            return;
        }
        state.loras.push({ name: name, strength: 1.0, enabled: true });
        renderLoraList();
    }
    function removeLora(idx) {
        state.loras.splice(idx, 1);
        renderLoraList();
    }
    function renderLoraList() {
        var list = document.getElementById('gen-lora-list');
        if (!list)
            return;
        list.innerHTML = '';
        state.loras.forEach(function (lora, idx) {
            // Ensure enabled property exists (migration from old format)
            if (lora.enabled === undefined)
                lora.enabled = true;
            var disabledClass = lora.enabled ? '' : ' gen-lora-disabled';
            var row = document.createElement('div');
            row.className = 'gen-lora-row' + disabledClass;
            row.innerHTML =
                '<button class="gen-toggle gen-lora-toggle' + (lora.enabled ? ' on' : '') + '" data-idx="' + idx + '"></button>' +
                    '<span class="gen-lora-name">' + lora.name + '</span>' +
                    '<input type="range" class="gen-range gen-lora-strength" min="-1" max="2" step="0.05" value="' + lora.strength + '"' + (lora.enabled ? '' : ' disabled') + '>' +
                    '<input type="number" class="gen-number-input gen-lora-val-input" min="-10" max="10" step="0.05" value="' + lora.strength.toFixed(2) + '"' + (lora.enabled ? '' : ' disabled') + '>' +
                    '<button class="gen-lora-remove" data-idx="' + idx + '">&times;</button>';
            list.appendChild(row);
        });
        // Bind events
        list.querySelectorAll('.gen-lora-strength').forEach(function (slider, idx) {
            slider.addEventListener('input', function () {
                state.loras[idx].strength = parseFloat(this.value);
                var valInput = this.parentElement.querySelector('.gen-lora-val-input');
                if (valInput)
                    valInput.value = state.loras[idx].strength.toFixed(2);
            });
        });
        list.querySelectorAll('.gen-lora-val-input').forEach(function (input, idx) {
            input.addEventListener('change', function () {
                var v = Math.max(-10, Math.min(10, parseFloat(this.value) || 0));
                this.value = v.toFixed(2);
                state.loras[idx].strength = v;
                var slider = this.parentElement.querySelector('.gen-lora-strength');
                if (slider)
                    slider.value = String(Math.max(-1, Math.min(2, v))); // clamp slider visual
            });
        });
        list.querySelectorAll('.gen-lora-toggle').forEach(function (btn) {
            btn.addEventListener('click', function () {
                var i = parseInt(this.dataset.idx);
                state.loras[i].enabled = !state.loras[i].enabled;
                renderLoraList();
            });
        });
        list.querySelectorAll('.gen-lora-remove').forEach(function (btn) {
            btn.addEventListener('click', function () {
                removeLora(parseInt(this.dataset.idx));
            });
        });
    }
    // ── Workflow Builder ──
    function buildWorkflow() {
        // Append style suffix to prompt
        var finalPrompt = state.prompt;
        if (state.stylePreset && state.stylePreset !== 'none') {
            for (var i = 0; i < stylePresets.length; i++) {
                if (stylePresets[i].id === state.stylePreset) {
                    finalPrompt += stylePresets[i].suffix;
                    break;
                }
            }
        }
        // Filter disabled LoRAs
        var enabledLoras = state.loras.filter(function (l) { return l.enabled !== false; });
        return WorkflowBuilder.build({
            model: state.model || '',
            prompt: finalPrompt,
            negPrompt: state.negPrompt,
            width: state.width,
            height: state.height,
            steps: state.steps,
            cfg: state.cfg,
            guidance: state.guidance,
            sampler: state.sampler,
            scheduler: state.scheduler,
            noiseScheduler: state.scheduler,
            seed: state.seed,
            frames: state.frames,
            fps: state.fps,
            referenceImagePath: state.referenceImagePath,
            referenceMaskPath: state.referenceMaskPath,
            drivingVideoPath: state.drivingVideoPath,
            drivingMaskVideoPath: state.drivingMaskVideoPath,
            scail2Mode: state.scail2Mode,
            additionalReferenceImagePaths: state.additionalReferenceImagePaths.slice(),
            additionalReferenceMaskPaths: state.additionalReferenceMaskPaths.slice(),
            loras: enabledLoras,
            // Phase 3: Advanced
            vae: state.vae,
            clipSkip: state.clipSkip,
            cfgRescale: state.cfgRescale,
            seamlessX: state.seamlessX,
            seamlessY: state.seamlessY,
            // Phase 3: Refiner
            refinerModel: state.refinerModel,
            refinerScheduler: state.refinerScheduler,
            refinerSteps: state.refinerSteps,
            refinerCfg: state.refinerCfg,
            refinerStart: state.refinerStart,
            aestheticScore: state.aestheticScore,
            negAestheticScore: state.negAestheticScore,
            // Phase 3: Compositing
            coherenceMode: state.coherenceMode,
            edgeSize: state.edgeSize,
            minDenoise: state.minDenoise,
            maskBlur: state.maskBlur,
            infillMethod: state.infillMethod
        });
    }
    function buildGenerateRequest(seed) {
        var finalPrompt = state.prompt;
        if (state.stylePreset && state.stylePreset !== 'none') {
            for (var i = 0; i < stylePresets.length; i++) {
                if (stylePresets[i].id === state.stylePreset) {
                    finalPrompt += stylePresets[i].suffix;
                    break;
                }
            }
        }
        var request = {
            model: state.model || '',
            prompt: finalPrompt,
            width: state.width,
            height: state.height,
            steps: state.steps,
            seed: seed,
            sampler: state.sampler || 'euler',
            scheduler: state.scheduler || 'simple',
            cfg: state.cfg,
            images: 1
        };
        var profile = activeCapabilityProfile();
        if (featureSupported(profile, 'negative_prompt') && state.negPrompt)
            request.negative = state.negPrompt;
        if (variationSupportedForArch(state.arch) && state.variationStrength > 0) {
            request.variation_seed = state.variationSeed;
            request.variation_strength = state.variationStrength;
        }
        if (featureSupported(profile, 'image_to_image') && state.initImagePath) {
            request.init_image = state.initImagePath;
            request.creativity = state.creativity;
        }
        var enabledLoras = state.loras.filter(function (lora) { return lora.enabled !== false; });
        if (enabledLoras.length) {
            request.lora = enabledLoras.map(function (lora) {
                return { name: lora.name, weight: Number(lora.strength) };
            });
        }
        return request;
    }
    // ── Generate ──
    function generate() {
        if (state.generating)
            return;
        if (!state.model) {
            showError('No model selected');
            return;
        }
        if (!state.prompt.trim()) {
            showError('Enter a prompt');
            return;
        }
        // Save to prompt history
        pushPromptHistory(state.prompt);
        var batchN = state.batchCount || 1;
        state.pendingBatch = batchN;
        setGenerating(true);
        var resolvedSeed = state.seed === -1 ? Math.floor(Math.random() * 4294967296) : state.seed;
        state.lastSeed = resolvedSeed;
        // Build and queue all batch workflows upfront
        var originalSeed = state.seed;
        var submissions = [];
        for (var i = 0; i < batchN; i++) {
            state.seed = (i === 0) ? resolvedSeed : Math.floor(Math.random() * 4294967296);
            // Bind completion metadata to this immutable submission. Video
            // requests block until the render completes, so reading `state`
            // in their completion handler would mislabel the gallery item if
            // the user selected another model while the render was running.
            submissions.push({
                request: buildGenerateRequest(state.seed),
                prompt: state.prompt,
                model: state.model,
                seed: state.seed,
                steps: state.steps,
                cfg: state.cfg,
                guidance: state.guidance,
                scheduler: state.scheduler,
                width: state.width,
                height: state.height,
                arch: ModelUtils.detectArchFromFilename(state.model)
            });
        }
        state.seed = originalSeed;
        // Queue all batch items (server handles sequencing)
        var queueFailed = false;
        submissions.forEach(function (submission, i) {
            if (queueFailed)
                return;
            if (batchN > 1) {
                els.btn.innerHTML = '<i data-lucide="wand-2"></i> Generating ' + (i + 1) + ' / ' + batchN + '...';
                if (typeof lucide !== 'undefined')
                    lucide.createIcons({ nameAttr: 'data-lucide' });
            }
            SerenityAPI.postGenerate(submission.request, {
                prompt: submission.prompt,
                model: submission.model,
                batchLabel: batchN > 1 ? ('(' + (i + 1) + '/' + batchN + ')') : ''
            })
                .then(function (result) {
                if (!result || !result.prompt_id)
                    throw new Error('server did not return a job id');
            })
                .catch(function (err) {
                queueFailed = true;
                showError('Failed to queue: ' + err.message);
                setGenerating(false);
            });
        });
    }
    // ── State Helpers ──
    function setGenerating(v) {
        state.generating = v;
        var isVideo = ModelUtils.isVideoModel(state.model);
        els.btn.disabled = v;
        if (v) {
            els.btn.innerHTML = '<i data-lucide="wand-2"></i> Generating...';
        }
        else {
            els.btn.innerHTML = isVideo
                ? '<i data-lucide="wand-2"></i> Generate Video'
                : '<i data-lucide="wand-2"></i> Generate';
        }
        if (typeof lucide !== 'undefined')
            lucide.createIcons({ nameAttr: 'data-lucide' });
        els.btn.classList.toggle('generating', v);
        if (els.batchStatus)
            els.batchStatus.textContent = v
                ? ('Running · ' + Math.max(1, state.pendingBatch) + ' queued')
                : 'Idle';
        if (v) {
            els.progress.classList.add('active');
            els.progressBar.style.width = '100%';
            if (els.leftProgress) {
                els.leftProgress.classList.add('active');
                els.leftProgressBar.style.width = '100%';
            }
        }
        else {
            els.progress.classList.remove('active');
            els.progressBar.style.width = '0%';
            els.progressLabel.classList.remove('visible');
            if (els.leftProgress) {
                els.leftProgress.classList.remove('active');
                els.leftProgressBar.style.width = '0%';
            }
            if (els.leftProgressLabel) {
                els.leftProgressLabel.textContent = '';
            }
        }
    }
    function displayImage(src) {
        state.currentImage = src;
        state.currentIsVideo = false;
        els.previewImg.src = src;
        els.previewImg.style.display = 'block';
        els.previewVideo.style.display = 'none';
        els.previewVideo.pause();
        els.actionBar.style.display = 'flex';
        els.empty.style.display = 'none';
    }
    function displayVideo(src, metadata) {
        state.currentImage = src;
        state.currentIsVideo = true;
        state.currentVideoFrames = Number(
            metadata && (metadata.frame_count != null ? metadata.frame_count : metadata.frames)
        ) || Number(state.frames) || null;
        state.currentVideoFps = Number(metadata && metadata.fps) || Number(state.fps) || null;
        els.previewVideo.src = src;
        els.previewVideo.style.display = 'block';
        els.previewImg.style.display = 'none';
        els.actionBar.style.display = 'flex';
        els.empty.style.display = 'none';
    }
    function clearPreview() {
        state.currentImage = null;
        state.currentIsVideo = false;
        state.currentVideoFrames = null;
        state.currentVideoFps = null;
        els.previewImg.style.display = 'none';
        els.previewImg.removeAttribute('src');
        els.previewVideo.style.display = 'none';
        els.previewVideo.pause();
        els.previewVideo.removeAttribute('src');
        els.actionBar.style.display = 'none';
        els.empty.style.display = 'flex';
        // Deselect gallery thumbs
        state.selectedImages = [];
        state.lastSelectedIndex = -1;
        updateSelectionUI();
        updateMetadataPanel();
        setPreviewModelBadges(state.model, state.arch);
    }
    function addToGallery(src, isVideo, metadata) {
        function value(key, fallback) {
            return metadata && Object.prototype.hasOwnProperty.call(metadata, key)
                ? metadata[key]
                : fallback;
        }
        var item = {
            src: src,
            isVideo: !!isVideo,
            prompt: value('prompt', state.prompt || ''),
            model: value('model', state.model || ''),
            seed: value('seed', state.lastSeed),
            steps: value('steps', state.steps),
            cfg: value('cfg', state.cfg),
            guidance: value('guidance', state.guidance),
            sampler: value('sampler', state.sampler),
            scheduler: value('scheduler', state.scheduler),
            width: value('width', state.width),
            height: value('height', state.height),
            frame_count: value('frame_count', state.currentVideoFrames),
            fps: value('fps', state.currentVideoFps),
            arch: value('arch', state.arch),
            timestamp: Date.now(),
            starred: false
        };
        state.gallery.unshift(item);
        if (state.gallery.length > 200)
            state.gallery.pop();
        state.galleryPage = 0;
        renderGallery();
        // Auto-switch to new image
        if (state.autoSwitchNew) {
            state.selectedImages = [0];
            state.lastSelectedIndex = 0;
            updateSelectionUI();
            updateMetadataPanel();
            setPreviewModelBadges(item.model, item.arch);
        }
        saveGallery();
    }
    function createThumb(item, galleryIndex) {
        var src = item.src;
        var isVideo = item.isVideo;
        var starred = item.starred;
        var isSelected = state.selectedImages.indexOf(galleryIndex) >= 0;
        var wrap = document.createElement('div');
        wrap.className = 'gen-thumb-wrap' + (isSelected ? ' gen-selected' : '') + (isVideo ? ' gen-thumb-video' : '');
        wrap.dataset.galleryIndex = String(galleryIndex);
        var starClass = starred ? ' starred' : '';
        var starChar = starred ? '\u2605' : '\u2606';
        if (isVideo) {
            wrap.innerHTML =
                '<button class="gen-thumb-star' + starClass + '" data-idx="' + galleryIndex + '">' + starChar + '</button>' +
                    '<video src="' + src + '#t=0.1" muted preload="metadata" style="width:100%;height:100%;object-fit:cover;display:block;"></video>' +
                    '<div class="gen-thumb-play">\u25b6</div>';
        }
        else {
            wrap.innerHTML =
                '<button class="gen-thumb-star' + starClass + '" data-idx="' + galleryIndex + '">' + starChar + '</button>' +
                    '<img src="' + src + '" alt="thumbnail" loading="lazy">' +
                    '<div class="gen-thumb-overlay">' +
                    '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>' +
                    '</div>';
        }
        // Star click
        wrap.querySelector('.gen-thumb-star').addEventListener('click', function (e) {
            e.stopPropagation();
            var idx = parseInt(this.dataset.idx);
            state.gallery[idx].starred = !state.gallery[idx].starred;
            saveGallery();
            renderGallery();
        });
        // Click with multi-select support
        wrap.addEventListener('click', function (e) {
            handleThumbClick(galleryIndex, e);
        });
        // Right-click context menu
        wrap.addEventListener('contextmenu', function (e) {
            e.preventDefault();
            showContextMenu(e, galleryIndex);
        });
        return wrap;
    }
    function saveGallery() {
        try {
            localStorage.setItem('sf-gallery', JSON.stringify(state.gallery));
        }
        catch (e) { /* quota exceeded */ }
    }
    function galleryMetadataFromJob(job) {
        var meta = (job && job.metadata) || {};
        var params = meta.params || (job && job.params) || {};
        var model = (job && job.model) || meta.model || params.model || '';
        var arch = params.arch || ModelUtils.detectArchFromFilename(model) || '';
        var executedCfg = meta.cfg != null ? meta.cfg : params.cfg;
        return {
            prompt: meta.prompt != null ? meta.prompt : (params.prompt || ''),
            model: model,
            seed: meta.seed != null ? meta.seed : params.seed,
            steps: meta.steps != null ? meta.steps : params.steps,
            cfg: executedCfg,
            // FLUX lowers its distilled guidance into JobParams.cfg. Preserve
            // that executed value under the UI's Guidance label too.
            guidance: arch === 'flux' ? executedCfg : params.guidance,
            sampler: meta.sampler || params.sampler || '',
            scheduler: meta.scheduler || params.scheduler || '',
            width: meta.width != null ? meta.width : params.width,
            height: meta.height != null ? meta.height : params.height,
            frame_count: meta.frame_count != null ? meta.frame_count : params.frames,
            fps: meta.fps != null ? meta.fps : params.fps,
            arch: arch
        };
    }
    function restoredGalleryJobId(item) {
        try {
            var filename = new URL(item.src, location.origin).searchParams.get('filename') || '';
            var match = filename.match(/^(job-\d+)\.png$/);
            return match ? match[1] : '';
        }
        catch (e) {
            return '';
        }
    }
    function repairRestoredGalleryMetadata() {
        fetch('/v1/jobs')
            .then(function (resp) {
            if (!resp.ok)
                throw new Error('HTTP ' + resp.status);
            return resp.json();
        })
            .then(function (jobs) {
            var byId = {};
            (Array.isArray(jobs) ? jobs : []).forEach(function (job) {
                if (job && job.id)
                    byId[job.id] = job;
            });
            var changed = false;
            state.gallery.forEach(function (item) {
                var id = restoredGalleryJobId(item);
                var job = id ? byId[id] : null;
                if (!job)
                    return;
                var corrected = galleryMetadataFromJob(job);
                Object.keys(corrected).forEach(function (key) {
                    if (corrected[key] != null && corrected[key] !== '' && item[key] !== corrected[key]) {
                        item[key] = corrected[key];
                        changed = true;
                    }
                });
            });
            if (changed) {
                saveGallery();
                renderGallery();
                updateMetadataPanel();
            }
        })
            .catch(function (err) {
            console.error('Saved gallery metadata repair failed: ' + err.message);
        });
    }
    function restoreGallery() {
        try {
            var saved = JSON.parse(localStorage.getItem('sf-gallery'));
            if (saved && Array.isArray(saved)) {
                state.gallery = saved.slice(0, 200).map(function (item) {
                    if (typeof item === 'string')
                        return { src: item, isVideo: false, prompt: '', starred: false, timestamp: 0 };
                    // Ensure backward compat: add missing fields
                    if (item.starred === undefined)
                        item.starred = false;
                    if (item.timestamp === undefined)
                        item.timestamp = 0;
                    return item;
                });
                renderGallery();
                repairRestoredGalleryMetadata();
            }
        }
        catch (e) { /* ignore */ }
        // Restore thumb size
        try {
            var ts = parseInt(localStorage.getItem('sf-thumb-size'));
            if (ts >= 45 && ts <= 200) {
                state.thumbSize = ts;
                applyThumbSize();
            }
        }
        catch (e) { /* ignore */ }
    }
    // ── Prompt History ──
    function loadPromptHistory() {
        try {
            var saved = JSON.parse(localStorage.getItem('sf-prompt-history'));
            if (saved && Array.isArray(saved)) {
                state.promptHistory = saved.slice(0, 20);
            }
        }
        catch (e) { /* ignore */ }
    }
    function savePromptHistory() {
        try {
            localStorage.setItem('sf-prompt-history', JSON.stringify(state.promptHistory));
        }
        catch (e) { /* quota exceeded */ }
    }
    function pushPromptHistory(prompt) {
        if (!prompt || !prompt.trim())
            return;
        var trimmed = prompt.trim();
        // Don't duplicate the last entry
        if (state.promptHistory.length > 0 && state.promptHistory[state.promptHistory.length - 1] === trimmed)
            return;
        state.promptHistory.push(trimmed);
        if (state.promptHistory.length > 20)
            state.promptHistory.shift();
        state.promptHistoryIndex = -1;
        savePromptHistory();
    }
    function navigatePromptHistory(direction) {
        if (state.promptHistory.length === 0)
            return;
        if (direction < 0) {
            // Up = older
            if (state.promptHistoryIndex === -1) {
                // Save current prompt as temp
                state._tempPrompt = state.prompt;
                state.promptHistoryIndex = state.promptHistory.length - 1;
            }
            else if (state.promptHistoryIndex > 0) {
                state.promptHistoryIndex--;
            }
        }
        else {
            // Down = newer
            if (state.promptHistoryIndex === -1)
                return;
            if (state.promptHistoryIndex < state.promptHistory.length - 1) {
                state.promptHistoryIndex++;
            }
            else {
                // Back to current
                state.promptHistoryIndex = -1;
                state.prompt = state._tempPrompt || '';
                if (els.prompt) {
                    els.prompt.value = state.prompt;
                    els.prompt.style.height = 'auto';
                    els.prompt.style.height = els.prompt.scrollHeight + 'px';
                }
                updateTokenCount();
                return;
            }
        }
        if (state.promptHistoryIndex >= 0 && state.promptHistoryIndex < state.promptHistory.length) {
            state.prompt = state.promptHistory[state.promptHistoryIndex];
            if (els.prompt) {
                els.prompt.value = state.prompt;
                els.prompt.style.height = 'auto';
                els.prompt.style.height = els.prompt.scrollHeight + 'px';
            }
            updateTokenCount();
        }
    }
    // ── Style Preview ──
    function updateStylePreview() {
        var el = document.getElementById('gen-style-preview');
        if (!el)
            return;
        if (state.stylePreset === 'none' || !state.stylePreset) {
            el.style.display = 'none';
            return;
        }
        var suffix = '';
        for (var i = 0; i < stylePresets.length; i++) {
            if (stylePresets[i].id === state.stylePreset) {
                suffix = stylePresets[i].suffix;
                break;
            }
        }
        if (suffix) {
            var preview = state.prompt ? state.prompt + suffix : '(your prompt)' + suffix;
            el.textContent = 'Full prompt: ' + preview;
            el.style.display = 'block';
        }
        else {
            el.style.display = 'none';
        }
    }
    // ── Searchable Model Picker ──
    function bindModelPicker() {
        if (!els.modelSearch)
            return;
        els.modelSearch.addEventListener('focus', function () {
            this.select();
            openModelDropdown();
        });
        els.modelSearch.addEventListener('input', function () {
            state.modelSearchQuery = this.value.toLowerCase();
            renderModelDropdown();
            if (els.modelDropdown)
                els.modelDropdown.style.display = '';
        });
        els.modelSearch.addEventListener('keydown', function (e) {
            if (e.key === 'Escape') {
                closeModelDropdown();
                this.blur();
            }
        });
        // Close on outside click
        document.addEventListener('click', function (e) {
            if (els.modelPickerWrap && !els.modelPickerWrap.contains(e.target)) {
                closeModelDropdown();
            }
        });
    }
    function openModelDropdown() {
        if (!els.modelDropdown)
            return;
        state.modelSearchQuery = '';
        renderModelDropdown();
        els.modelDropdown.style.display = '';
    }
    function closeModelDropdown() {
        if (els.modelDropdown)
            els.modelDropdown.style.display = 'none';
        // Restore display name
        if (els.modelSearch && state.model) {
            els.modelSearch.value = state.model;
        }
    }
    function selectModel(name) {
        state.model = name;
        els.model.value = name;
        if (els.modelSearch)
            els.modelSearch.value = name;
        closeModelDropdown();
        updateUIForArch(ModelUtils.detectArchFromFilename(name));
        var globalBadge = document.querySelector('#topbar .model-badge');
        if (globalBadge) {
            globalBadge.textContent = name;
            globalBadge.title = name;
        }
    }
    function renderModelDropdown() {
        if (!els.modelDropdownList)
            return;
        var models = state.allModels || [];
        var query = state.modelSearchQuery || '';
        var filtered = models.filter(function (m) {
            return !query || m.name.toLowerCase().indexOf(query) >= 0;
        });
        // Group by architecture
        var groups = {};
        var groupOrder = ['krea2', 'klein', 'flux2', 'flux', 'ideogram4', 'sdxl', 'sd3', 'zimage', 'qwen', 'hunyuan', 'sd15', 'ltxv', 'wan', 'bernini', 'scail2', 'other'];
        var groupLabels = { krea2: 'KREA2', klein: 'KLEIN', flux2: 'FLUX 2', flux: 'FLUX', ideogram4: 'IDEOGRAM 4', sdxl: 'SDXL', sd3: 'SD3', zimage: 'Z-IMAGE', qwen: 'QWEN', hunyuan: 'HUNYUAN', sd15: 'SD 1.5', ltxv: 'Video (LTX)', wan: 'Video (WAN)', bernini: 'Video (BERNINI-R)', scail2: 'Video (SCAIL-2)', other: 'Other' };
        filtered.forEach(function (m) {
            var arch = ModelUtils.detectArchFromFilename(m.name) || 'other';
            if (!groups[arch])
                groups[arch] = [];
            groups[arch].push(m);
        });
        var html = '';
        if (filtered.length === 0) {
            html = '<div class="gen-model-dropdown-empty">No models match</div>';
        }
        else {
            // Any arch not in groupOrder still renders (never silently drop a model)
            Object.keys(groups).forEach(function (arch) {
                if (groupOrder.indexOf(arch) < 0)
                    groupOrder.push(arch);
            });
            groupOrder.forEach(function (arch) {
                if (!groups[arch] || groups[arch].length === 0)
                    return;
                html += '<div class="gen-model-dropdown-group">' + (groupLabels[arch] || arch.toUpperCase()) + '</div>';
                groups[arch].forEach(function (m) {
                    var archVal = ModelUtils.detectArchFromFilename(m.name) || 'other';
                    html += '<div class="gen-model-dropdown-item" data-model="' + m.name + '">' +
                        '<span class="gen-model-dropdown-name">' + m.name + '</span>' +
                        '<span class="gen-arch-badge" data-arch="' + archVal + '">' + (groupLabels[archVal] || archVal.toUpperCase()) + '</span>' +
                        '</div>';
                });
            });
        }
        els.modelDropdownList.innerHTML = html;
        // Bind click
        els.modelDropdownList.querySelectorAll('.gen-model-dropdown-item').forEach(function (item) {
            item.addEventListener('click', function () {
                selectModel(this.dataset.model);
            });
        });
    }
    // ── Prompt Attention Weight (Ctrl+Up/Down) ──
    function handlePromptWeight(textarea, increase) {
        var start = textarea.selectionStart;
        var end = textarea.selectionEnd;
        if (start === end)
            return; // No selection
        var text = textarea.value;
        var selected = text.substring(start, end);
        var newText;
        var newStart, newEnd;
        if (increase) {
            // Check if already weighted with explicit value: (word:N.N)
            var weightMatch = selected.match(/^\((.+):(\d+\.?\d*)\)$/);
            if (weightMatch) {
                var newWeight = (parseFloat(weightMatch[2]) + 0.1).toFixed(1);
                newText = '(' + weightMatch[1] + ':' + newWeight + ')';
            }
            else if (/^\(+[^)]+\)+$/.test(selected)) {
                // Already in parens, add another layer
                newText = '(' + selected + ')';
            }
            else {
                // Wrap in parens
                newText = '(' + selected + ')';
            }
        }
        else {
            // Decrease
            var weightMatch2 = selected.match(/^\((.+):(\d+\.?\d*)\)$/);
            if (weightMatch2) {
                var decreased = (parseFloat(weightMatch2[2]) - 0.1).toFixed(1);
                if (parseFloat(decreased) <= 1.0) {
                    newText = '(' + weightMatch2[1] + ')';
                }
                else {
                    newText = '(' + weightMatch2[1] + ':' + decreased + ')';
                }
            }
            else if (/^\(\((.+)\)\)$/.test(selected)) {
                // Remove one layer of parens
                newText = selected.substring(1, selected.length - 1);
            }
            else if (/^\((.+)\)$/.test(selected)) {
                // Remove parens entirely
                newText = selected.substring(1, selected.length - 1);
            }
            else {
                return; // Nothing to decrease
            }
        }
        textarea.value = text.substring(0, start) + newText + text.substring(end);
        state.prompt = textarea.value;
        // Restore selection
        newStart = start;
        newEnd = start + newText.length;
        textarea.setSelectionRange(newStart, newEnd);
        updateTokenCount();
    }
    // ── Error Display ──
    function showError(msg) {
        els.errorBanner.textContent = msg;
        els.errorBanner.classList.add('visible');
        setTimeout(function () {
            els.errorBanner.classList.remove('visible');
        }, 5000);
    }
    // ── WebSocket (via shared SerenityWS) ──
    function connectWS() {
        SerenityWS.on('connected', function () {
            els.wsIndicator.classList.remove('visible');
        });
        SerenityWS.on('disconnected', function () {
            els.wsIndicator.classList.add('visible');
        });
        SerenityWS.on('execution_start', function () {
            setGenerating(true);
        });
        SerenityWS.on('progress', function (data) {
            if (!data)
                return;
            var pct = data.max > 0 ? (data.value / data.max * 100).toFixed(0) : '0';
            els.progressBar.style.width = pct + '%';
            els.progressLabel.textContent = 'Step ' + data.value + ' / ' + data.max;
            els.progressLabel.classList.add('visible');
            if (els.leftProgressBar) {
                els.leftProgressBar.style.width = pct + '%';
            }
            if (els.leftProgressLabel) {
                els.leftProgressLabel.textContent = 'Step ' + data.value + ' / ' + data.max;
            }
        });
        SerenityWS.on('preview', function (data) {
            if (!data || !data.blob || !state.generating)
                return;
            var url = URL.createObjectURL(data.blob);
            if (els.previewImg) {
                // Revoke old preview URL to prevent memory leaks
                if (els.previewImg._previewUrl)
                    URL.revokeObjectURL(els.previewImg._previewUrl);
                els.previewImg._previewUrl = url;
                els.previewImg.src = url;
                els.previewImg.style.display = 'block';
                if (els.empty)
                    els.empty.style.display = 'none';
                els.previewImg.classList.add('gen-preview-live');
            }
        });
        SerenityWS.on('executed', function (data) {
            if (!data || !data.output)
                return;
            // Clean up live preview state
            if (els.previewImg) {
                els.previewImg.classList.remove('gen-preview-live');
                if (els.previewImg._previewUrl) {
                    URL.revokeObjectURL(els.previewImg._previewUrl);
                    els.previewImg._previewUrl = undefined;
                }
            }
            // Image outputs
            var out = data.output.ui || data.output;
            var items = out.images;
            var isVideo = false;
            // Video outputs (SaveAnimatedWEBP, SaveVideo)
            if (!items && out.videos) {
                items = out.videos;
                isVideo = true;
            }
            if (!items || !items.length)
                return;
            var file = items[0];
            var src = '/view?filename=' + encodeURIComponent(file.filename) +
                '&subfolder=' + encodeURIComponent(file.subfolder || '') +
                '&type=' + encodeURIComponent(file.type || 'output');
            // Also detect video from filename extension
            if (!isVideo)
                isVideo = /\.(webp|mp4|gif)$/i.test(file.filename);
            if (isVideo) {
                displayVideo(src);
            }
            else {
                displayImage(src);
            }
            // The selected model can change while a queued job is running. Bind
            // gallery metadata to the completed job record, never mutable UI state.
            // /v1/job/:id exposes the PNG-derived params captured at Done-time.
            var promptId = data.prompt_id || '';
            fetch('/v1/job/' + encodeURIComponent(promptId))
                .then(function (resp) {
                if (!resp.ok)
                    throw new Error('HTTP ' + resp.status);
                return resp.json();
            })
                .then(function (job) {
                addToGallery(src, isVideo, galleryMetadataFromJob(job));
            })
                .catch(function (err) {
                console.error('Completed job metadata unavailable for ' + promptId + ': ' + err.message);
                addToGallery(src, isVideo, {
                    prompt: '',
                    model: '',
                    seed: null,
                    steps: null,
                    cfg: null,
                    scheduler: '',
                    width: null,
                    height: null,
                    arch: ''
                });
            })
                .finally(function () {
                state.pendingBatch = Math.max(0, state.pendingBatch - 1);
                if (state.pendingBatch <= 0) {
                    setGenerating(false);
                }
            });
        });
        SerenityWS.on('execution_error', function (data) {
            state.pendingBatch = 0;
            var errMsg = (data && data.exception_message) || 'Generation failed';
            showError(errMsg);
            setGenerating(false);
        });
    }
    // ── Phase 2: Gallery Rendering ──
    function getFilteredGallery() {
        var items = state.gallery;
        // Search filter
        if (state.gallerySearch) {
            var q = state.gallerySearch.toLowerCase();
            items = items.filter(function (item) {
                return (item.prompt && item.prompt.toLowerCase().indexOf(q) >= 0);
            });
        }
        // Sort
        if (!state.sortNewestFirst) {
            items = items.slice().reverse();
        }
        // Starred first
        if (state.starredFirst) {
            var starred = items.filter(function (i) { return i.starred; });
            var unstarred = items.filter(function (i) { return !i.starred; });
            items = starred.concat(unstarred);
        }
        return items;
    }
    function getGalleryIndexMap() {
        // Maps filtered display position to actual gallery index
        var map = [];
        var filtered = getFilteredGallery();
        filtered.forEach(function (item) {
            map.push(state.gallery.indexOf(item));
        });
        return map;
    }
    function renderGallery() {
        if (!els.galleryGrid)
            return;
        var filtered = getFilteredGallery();
        var indexMap = getGalleryIndexMap();
        var pageSize = state.galleryPageSize;
        var totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
        if (state.galleryPage >= totalPages)
            state.galleryPage = totalPages - 1;
        if (state.galleryPage < 0)
            state.galleryPage = 0;
        var startIdx = state.galleryPage * pageSize;
        var pageItems = filtered.slice(startIdx, startIdx + pageSize);
        var pageIndices = indexMap.slice(startIdx, startIdx + pageSize);
        els.galleryGrid.innerHTML = '';
        pageItems.forEach(function (item, i) {
            var realIdx = pageIndices[i];
            var thumb = createThumb(item, realIdx);
            els.galleryGrid.appendChild(thumb);
        });
        // Update grid template based on thumb size
        applyThumbSize();
        // Pagination
        var pageInfo = document.getElementById('gen-page-info');
        var prevBtn = document.getElementById('gen-page-prev');
        var nextBtn = document.getElementById('gen-page-next');
        if (pageInfo)
            pageInfo.textContent = 'Page ' + (state.galleryPage + 1) + ' of ' + totalPages;
        if (prevBtn)
            prevBtn.disabled = state.galleryPage <= 0;
        if (nextBtn)
            nextBtn.disabled = state.galleryPage >= totalPages - 1;
        renderBatchStrip();
        if (typeof lucide !== 'undefined')
            lucide.createIcons({ nameAttr: 'data-lucide' });
    }
    function renderBatchStrip() {
        if (!els.batchStrip)
            return;
        els.batchStrip.innerHTML = '';
        var recent = state.gallery.slice(0, 12);
        if (!recent.length) {
            els.batchStrip.innerHTML = '<div class="gen-swarm-batch-empty">New results appear here</div>';
            return;
        }
        recent.forEach(function (item) {
            if (item.isVideo)
                return;
            var button = document.createElement('button');
            button.type = 'button';
            button.className = 'gen-swarm-batch-thumb';
            var image = document.createElement('img');
            image.src = item.src;
            image.alt = item.prompt || 'Generated image';
            button.appendChild(image);
            var caption = document.createElement('span');
            caption.textContent = (item.width && item.height)
                ? (item.width + '×' + item.height)
                : 'Image';
            button.appendChild(caption);
            button.addEventListener('click', function () {
                var index = state.gallery.indexOf(item);
                if (index >= 0)
                    handleThumbClick(index, {});
            });
            els.batchStrip.appendChild(button);
        });
    }
    function applyThumbSize() {
        if (!els.galleryGrid)
            return;
        els.galleryGrid.style.gridTemplateColumns = 'repeat(auto-fill, minmax(' + state.thumbSize + 'px, 1fr))';
    }
    // ── Phase 2: Multi-select ──
    function handleThumbClick(galleryIndex, e) {
        var item = state.gallery[galleryIndex];
        if (!item)
            return;
        if (e.ctrlKey || e.metaKey) {
            // Toggle this item in selection
            var pos = state.selectedImages.indexOf(galleryIndex);
            if (pos >= 0) {
                state.selectedImages.splice(pos, 1);
            }
            else {
                state.selectedImages.push(galleryIndex);
            }
            state.lastSelectedIndex = galleryIndex;
        }
        else if (e.shiftKey && state.lastSelectedIndex >= 0) {
            // Range select
            var start = Math.min(state.lastSelectedIndex, galleryIndex);
            var end = Math.max(state.lastSelectedIndex, galleryIndex);
            state.selectedImages = [];
            for (var i = start; i <= end; i++) {
                if (state.selectedImages.indexOf(i) < 0) {
                    state.selectedImages.push(i);
                }
            }
        }
        else {
            // Normal click: select only this
            state.selectedImages = [galleryIndex];
            state.lastSelectedIndex = galleryIndex;
        }
        // Display the clicked item in preview
        if (item.isVideo) {
            displayVideo(item.src, item);
        }
        else {
            displayImage(item.src);
        }
        setPreviewModelBadges(item.model, item.arch);
        updateSelectionUI();
        updateMetadataPanel();
    }
    function updateSelectionUI() {
        // Update thumb selection classes
        els.galleryGrid.querySelectorAll('.gen-thumb-wrap').forEach(function (wrap) {
            var idx = parseInt(wrap.dataset.galleryIndex);
            wrap.classList.toggle('gen-selected', state.selectedImages.indexOf(idx) >= 0);
            wrap.classList.remove('active'); // remove old style
        });
        // Selection badge
        var badge = document.getElementById('gen-selection-badge');
        if (badge) {
            if (state.selectedImages.length > 1) {
                badge.textContent = state.selectedImages.length + ' selected';
                badge.classList.add('visible');
            }
            else {
                badge.classList.remove('visible');
            }
        }
        // Bulk action bar
        var bulkBar = document.getElementById('gen-bulk-bar');
        if (bulkBar) {
            bulkBar.classList.toggle('visible', state.selectedImages.length > 1);
        }
    }
    // ── Phase 2: Metadata Panel ──
    function updateMetadataPanel() {
        var panel = document.getElementById('gen-metadata-panel');
        var summary = document.getElementById('gen-metadata-summary');
        var full = document.getElementById('gen-metadata-full');
        if (!panel || !summary || !full)
            return;
        // Only show for single selection
        if (state.selectedImages.length !== 1) {
            panel.classList.remove('visible');
            return;
        }
        var item = state.gallery[state.selectedImages[0]];
        if (!item || (!item.prompt && !item.model && !item.seed)) {
            panel.classList.remove('visible');
            return;
        }
        var truncPrompt = item.prompt ? (item.prompt.length > 100 ? item.prompt.substring(0, 97) + '...' : item.prompt) : '';
        var pairs = [];
        if (truncPrompt)
            pairs.push('<span class="gen-metadata-key">Prompt:</span> <span class="gen-metadata-val">' + escapeHtml(truncPrompt) + '</span>');
        if (item.model)
            pairs.push('<span class="gen-metadata-key">Model:</span> <span class="gen-metadata-val">' + escapeHtml(item.model) + '</span>');
        if (item.seed != null)
            pairs.push('<span class="gen-metadata-key">Seed:</span> <span class="gen-metadata-val">' + item.seed + '</span>');
        if (item.width && item.height)
            pairs.push('<span class="gen-metadata-key">Size:</span> <span class="gen-metadata-val">' + item.width + '\u00d7' + item.height + '</span>');
        if (item.steps)
            pairs.push('<span class="gen-metadata-key">Steps:</span> <span class="gen-metadata-val">' + item.steps + '</span>');
        if (item.cfg != null)
            pairs.push('<span class="gen-metadata-key">' + (item.arch === 'flux' ? 'Guidance' : 'CFG') + ':</span> <span class="gen-metadata-val">' + item.cfg + '</span>');
        summary.innerHTML = pairs.join(' ');
        panel.classList.add('visible');
        // Full metadata
        var fullPairs = [];
        if (item.prompt)
            fullPairs.push('<span class="gen-metadata-key">Full Prompt:</span> <span class="gen-metadata-val">' + escapeHtml(item.prompt) + '</span>');
        if (item.sampler)
            fullPairs.push('<span class="gen-metadata-key">Sampler:</span> <span class="gen-metadata-val">' + escapeHtml(item.sampler) + '</span>');
        if (item.scheduler)
            fullPairs.push('<span class="gen-metadata-key">Scheduler:</span> <span class="gen-metadata-val">' + escapeHtml(item.scheduler) + '</span>');
        if (item.guidance && item.arch !== 'flux')
            fullPairs.push('<span class="gen-metadata-key">Guidance:</span> <span class="gen-metadata-val">' + item.guidance + '</span>');
        if (item.arch)
            fullPairs.push('<span class="gen-metadata-key">Arch:</span> <span class="gen-metadata-val">' + item.arch + '</span>');
        if (item.timestamp)
            fullPairs.push('<span class="gen-metadata-key">Time:</span> <span class="gen-metadata-val">' + new Date(item.timestamp).toLocaleString() + '</span>');
        full.innerHTML = fullPairs.join('<br>');
        // Toggle expand on click
        summary.onclick = function () {
            state.metadataExpanded = !state.metadataExpanded;
            full.classList.toggle('open', state.metadataExpanded);
        };
    }
    function escapeHtml(str) {
        if (!str)
            return '';
        var div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
    // ── Phase 2: Context Menu ──
    function showContextMenu(e, galleryIndex) {
        state.contextMenuIndex = galleryIndex;
        var menu = document.getElementById('gen-context-menu');
        if (!menu) {
            menu = document.createElement('div');
            menu.id = 'gen-context-menu';
            menu.className = 'gen-context-menu';
            document.body.appendChild(menu);
        }
        var item = state.gallery[galleryIndex] || {};
        var starLabel = item.starred ? 'Unstar' : 'Star';
        menu.innerHTML =
            '<button class="gen-context-item" data-action="download">Download</button>' +
                '<button class="gen-context-item" data-action="star">' + starLabel + '</button>' +
                '<div class="gen-context-sep"></div>' +
                '<button class="gen-context-item" data-action="use-prompt">Use Prompt</button>' +
                '<button class="gen-context-item" data-action="use-seed">Use Seed</button>' +
                '<button class="gen-context-item" data-action="use-all">Use All Parameters</button>' +
                '<div class="gen-context-sep"></div>' +
                '<button class="gen-context-item destructive" data-action="delete">Delete</button>';
        // Position
        menu.style.left = e.clientX + 'px';
        menu.style.top = e.clientY + 'px';
        menu.classList.add('visible');
        // Ensure menu stays in viewport
        requestAnimationFrame(function () {
            var rect = menu.getBoundingClientRect();
            if (rect.right > window.innerWidth)
                menu.style.left = (window.innerWidth - rect.width - 4) + 'px';
            if (rect.bottom > window.innerHeight)
                menu.style.top = (window.innerHeight - rect.height - 4) + 'px';
        });
        // Bind actions
        menu.querySelectorAll('.gen-context-item').forEach(function (btn) {
            btn.addEventListener('click', function () {
                handleContextAction(this.dataset.action, galleryIndex);
                hideContextMenu();
            });
        });
    }
    function hideContextMenu() {
        var menu = document.getElementById('gen-context-menu');
        if (menu)
            menu.classList.remove('visible');
    }
    function handleContextAction(action, idx) {
        var item = state.gallery[idx];
        if (!item)
            return;
        switch (action) {
            case 'download':
                var a = document.createElement('a');
                a.href = item.src;
                var ext = item.isVideo ? '.mp4' : '.png';
                a.download = 'serenityflow_' + Date.now() + ext;
                a.click();
                break;
            case 'star':
                item.starred = !item.starred;
                saveGallery();
                renderGallery();
                break;
            case 'use-prompt':
                if (item.prompt && els.prompt) {
                    els.prompt.value = item.prompt;
                    state.prompt = item.prompt;
                    els.prompt.style.height = 'auto';
                    els.prompt.style.height = els.prompt.scrollHeight + 'px';
                    updateTokenCount();
                }
                break;
            case 'use-seed':
                if (item.seed != null && els.seed) {
                    els.seed.value = String(item.seed);
                    state.seed = item.seed;
                    var toggle = document.getElementById('gen-seed-random-toggle');
                    if (toggle)
                        toggle.classList.toggle('on', state.seed === -1);
                }
                break;
            case 'use-all':
                if (item.prompt && els.prompt) {
                    els.prompt.value = item.prompt;
                    state.prompt = item.prompt;
                    els.prompt.style.height = 'auto';
                    els.prompt.style.height = els.prompt.scrollHeight + 'px';
                    updateTokenCount();
                }
                if (item.seed != null && els.seed) {
                    els.seed.value = String(item.seed);
                    state.seed = item.seed;
                }
                if (item.steps && els.steps) {
                    els.steps.value = String(item.steps);
                    if (els.stepsRange)
                        els.stepsRange.value = String(item.steps);
                    state.steps = item.steps;
                }
                if (item.cfg != null && els.cfg) {
                    els.cfg.value = String(item.cfg);
                    if (els.cfgRange)
                        els.cfgRange.value = String(item.cfg);
                    state.cfg = item.cfg;
                }
                if (item.guidance && els.guidance) {
                    els.guidance.value = String(item.guidance);
                    if (els.guidanceRange)
                        els.guidanceRange.value = String(item.guidance);
                    state.guidance = item.guidance;
                }
                if (item.scheduler && els.scheduler) {
                    els.scheduler.value = item.scheduler;
                    state.scheduler = item.scheduler;
                }
                if (item.width && item.height) {
                    state.width = item.width;
                    state.height = item.height;
                    syncDimensionInputs();
                    syncAspectDropdown();
                    updateAspectPreview();
                }
                break;
            case 'delete':
                state.gallery.splice(idx, 1);
                state.selectedImages = state.selectedImages.filter(function (si) { return si !== idx; }).map(function (si) { return si > idx ? si - 1 : si; });
                saveGallery();
                renderGallery();
                updateSelectionUI();
                updateMetadataPanel();
                break;
        }
    }
    // ── Phase 2: Bulk Operations ──
    function bindBulkActions() {
        var bulkStar = document.getElementById('gen-bulk-star');
        var bulkUnstar = document.getElementById('gen-bulk-unstar');
        var bulkDownload = document.getElementById('gen-bulk-download');
        var bulkDelete = document.getElementById('gen-bulk-delete');
        if (bulkStar)
            bulkStar.addEventListener('click', function () {
                state.selectedImages.forEach(function (idx) {
                    if (state.gallery[idx])
                        state.gallery[idx].starred = true;
                });
                saveGallery();
                renderGallery();
            });
        if (bulkUnstar)
            bulkUnstar.addEventListener('click', function () {
                state.selectedImages.forEach(function (idx) {
                    if (state.gallery[idx])
                        state.gallery[idx].starred = false;
                });
                saveGallery();
                renderGallery();
            });
        if (bulkDownload)
            bulkDownload.addEventListener('click', function () {
                state.selectedImages.forEach(function (idx) {
                    var item = state.gallery[idx];
                    if (!item)
                        return;
                    var a = document.createElement('a');
                    a.href = item.src;
                    a.download = 'serenityflow_' + (item.timestamp || Date.now()) + (item.isVideo ? '.mp4' : '.png');
                    a.click();
                });
            });
        if (bulkDelete)
            bulkDelete.addEventListener('click', function () {
                if (!confirm('Delete ' + state.selectedImages.length + ' images?'))
                    return;
                // Sort descending to splice safely
                var sorted = state.selectedImages.slice().sort(function (a, b) { return b - a; });
                sorted.forEach(function (idx) { state.gallery.splice(idx, 1); });
                state.selectedImages = [];
                saveGallery();
                renderGallery();
                updateSelectionUI();
                updateMetadataPanel();
                clearPreview();
            });
    }
    // ── Phase 7: Gallery Upload & Drag-Drop ──
    function bindGalleryUpload() {
        var uploadBtn = document.getElementById('gen-gallery-upload-btn');
        var uploadInput = document.getElementById('gen-gallery-upload-input');
        if (uploadBtn && uploadInput) {
            uploadBtn.addEventListener('click', function () { uploadInput.click(); });
            uploadInput.addEventListener('change', function () {
                if (this.files && this.files.length > 0) {
                    handleUploadFiles(this.files);
                    this.value = '';
                }
            });
        }
        // Drag-and-drop on gallery grid
        var galleryGrid = els.galleryGrid;
        if (galleryGrid) {
            galleryGrid.addEventListener('dragover', function (e) {
                e.preventDefault();
                e.stopPropagation();
                galleryGrid.classList.add('gen-drag-over');
            });
            galleryGrid.addEventListener('dragleave', function (e) {
                e.preventDefault();
                galleryGrid.classList.remove('gen-drag-over');
            });
            galleryGrid.addEventListener('drop', function (e) {
                e.preventDefault();
                e.stopPropagation();
                galleryGrid.classList.remove('gen-drag-over');
                if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                    handleUploadFiles(e.dataTransfer.files);
                }
            });
        }
    }
    function handleUploadFiles(files) {
        Array.prototype.forEach.call(files, function (file) {
            if (!file.type.startsWith('image/'))
                return;
            // Upload to server
            var formData = new FormData();
            formData.append('image', file);
            fetch('/upload/image', { method: 'POST', body: formData })
                .then(function (r) { return r.ok ? r.json() : null; })
                .then(function (data) {
                if (data && data.name) {
                    var src = SerenityAPI.viewUrl(data.name, data.subfolder || '', data.type || 'input');
                    addToGallery(src, false, { prompt: '(uploaded)' });
                    renderGallery();
                }
            })
                .catch(function (err) {
                // Fallback: add as local blob URL
                var reader = new FileReader();
                reader.onload = function (ev) {
                    addToGallery(ev.target.result, false, { prompt: '(uploaded)' });
                    renderGallery();
                };
                reader.readAsDataURL(file);
            });
        });
    }
    // ── Phase 2: Gallery Settings Popover ──
    function bindGallerySettings() {
        var settingsBtn = document.getElementById('gen-gallery-settings-btn');
        var popover = document.getElementById('gen-gallery-popover');
        if (!settingsBtn || !popover)
            return;
        settingsBtn.addEventListener('click', function (e) {
            e.stopPropagation();
            state.gallerySettingsOpen = !state.gallerySettingsOpen;
            popover.classList.toggle('visible', state.gallerySettingsOpen);
        });
        // Close on outside click
        document.addEventListener('click', function (e) {
            if (popover && !popover.contains(e.target) && e.target !== settingsBtn) {
                state.gallerySettingsOpen = false;
                popover.classList.remove('visible');
            }
        });
        // Thumb size slider
        var thumbSlider = document.getElementById('gen-thumb-size-slider');
        var thumbVal = document.getElementById('gen-thumb-size-val');
        if (thumbSlider) {
            thumbSlider.value = String(state.thumbSize);
            if (thumbVal)
                thumbVal.textContent = String(state.thumbSize);
            thumbSlider.addEventListener('input', function () {
                state.thumbSize = parseInt(this.value);
                if (thumbVal)
                    thumbVal.textContent = String(state.thumbSize);
                applyThumbSize();
                try {
                    localStorage.setItem('sf-thumb-size', String(state.thumbSize));
                }
                catch (e) { }
            });
        }
        // Sort direction toggle
        var sortToggle = document.getElementById('gen-sort-direction-toggle');
        var sortLabel = document.getElementById('gen-sort-direction-label');
        if (sortToggle) {
            sortToggle.addEventListener('click', function () {
                state.sortNewestFirst = !state.sortNewestFirst;
                this.classList.toggle('on', state.sortNewestFirst);
                if (sortLabel)
                    sortLabel.textContent = state.sortNewestFirst ? 'Newest' : 'Oldest';
                state.galleryPage = 0;
                renderGallery();
            });
        }
        // Starred first toggle
        var starredToggle = document.getElementById('gen-starred-first-toggle');
        if (starredToggle) {
            starredToggle.addEventListener('click', function () {
                state.starredFirst = !state.starredFirst;
                this.classList.toggle('on', state.starredFirst);
                state.galleryPage = 0;
                renderGallery();
            });
        }
        // Auto-switch toggle
        var autoToggle = document.getElementById('gen-auto-switch-toggle');
        if (autoToggle) {
            autoToggle.addEventListener('click', function () {
                state.autoSwitchNew = !state.autoSwitchNew;
                this.classList.toggle('on', state.autoSwitchNew);
            });
        }
    }
    // ── Phase 2: Pagination ──
    function bindPagination() {
        var prevBtn = document.getElementById('gen-page-prev');
        var nextBtn = document.getElementById('gen-page-next');
        if (prevBtn) {
            prevBtn.addEventListener('click', function () {
                if (state.galleryPage > 0) {
                    state.galleryPage--;
                    renderGallery();
                }
            });
        }
        if (nextBtn) {
            nextBtn.addEventListener('click', function () {
                var totalPages = Math.ceil(getFilteredGallery().length / state.galleryPageSize);
                if (state.galleryPage < totalPages - 1) {
                    state.galleryPage++;
                    renderGallery();
                }
            });
        }
    }
    // ── Phase 2: Global Event Listeners ──
    function bindGlobalPhase2() {
        // Close context menu on click outside or ESC
        document.addEventListener('click', function () {
            hideContextMenu();
        });
        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape') {
                hideContextMenu();
            }
        });
    }

    function getParams() {
        return {
            model: state.model || '',
            prompt: state.prompt || '',
            negPrompt: state.negPrompt || '',
            width: state.width,
            height: state.height,
            steps: state.steps,
            cfg: state.cfg,
            guidance: state.guidance,
            sampler: state.sampler,
            scheduler: state.scheduler,
            noiseScheduler: state.scheduler,
            seed: state.seed,
            variationSeed: state.variationSeed,
            variationStrength: state.variationStrength,
            creativity: state.creativity,
            batchCount: state.batchCount,
            stylePreset: state.stylePreset,
            loras: state.loras.map(function (lora) {
                return { name: lora.name, strength: lora.strength, enabled: lora.enabled !== false };
            })
        };
    }

    function applyParams(params) {
        if (!params)
            return;
        if (!initialized)
            init();
        if (typeof params.model === 'string' && params.model) {
            state.model = params.model;
            if (els.model)
                els.model.value = params.model;
            if (els.modelSearch)
                els.modelSearch.value = params.model;
            updateUIForArch(ModelUtils.detectArchFromFilename(params.model));
        }
        if (typeof params.prompt === 'string')
            state.prompt = params.prompt;
        if (typeof params.negPrompt === 'string')
            state.negPrompt = params.negPrompt;
        ['width', 'height', 'steps', 'cfg', 'guidance', 'seed', 'variationSeed',
            'variationStrength', 'creativity', 'batchCount'].forEach(function (key) {
            if (params[key] != null && Number.isFinite(Number(params[key])))
                state[key] = Number(params[key]);
        });
        var profile = activeCapabilityProfile();
        var supportedSamplers = profile && profile.samplers &&
            Array.isArray(profile.samplers.supported_samplers)
            ? profile.samplers.supported_samplers : [];
        var supportedSchedulers = profile && profile.samplers &&
            Array.isArray(profile.samplers.supported_schedulers)
            ? profile.samplers.supported_schedulers : [];
        if (typeof params.sampler === 'string' && supportedSamplers.indexOf(params.sampler) >= 0)
            state.sampler = params.sampler;
        if (typeof params.scheduler === 'string' && params.scheduler) {
            if (supportedSchedulers.indexOf(params.scheduler) >= 0)
                state.scheduler = params.scheduler;
            else if (supportedSamplers.indexOf(params.scheduler) >= 0)
                state.sampler = params.scheduler;
        }
        if (typeof params.noiseScheduler === 'string' &&
            supportedSchedulers.indexOf(params.noiseScheduler) >= 0)
            state.scheduler = params.noiseScheduler;
        if (typeof params.stylePreset === 'string')
            state.stylePreset = params.stylePreset;
        if (Array.isArray(params.loras)) {
            state.loras = params.loras.map(function (lora) {
                return {
                    name: lora.name || '',
                    strength: Number(lora.strength == null ? 1 : lora.strength),
                    enabled: lora.enabled !== false
                };
            }).filter(function (lora) { return !!lora.name; });
            renderLoraList();
        }
        if (els.prompt) {
            els.prompt.value = state.prompt;
            els.prompt.style.height = 'auto';
            els.prompt.style.height = els.prompt.scrollHeight + 'px';
        }
        if (els.negPrompt)
            els.negPrompt.value = state.negPrompt;
        if (els.steps)
            els.steps.value = String(state.steps);
        if (els.stepsRange)
            els.stepsRange.value = String(state.steps);
        if (els.cfg)
            els.cfg.value = String(state.cfg);
        if (els.cfgRange)
            els.cfgRange.value = String(state.cfg);
        if (els.guidance)
            els.guidance.value = String(state.guidance);
        if (els.guidanceRange)
            els.guidanceRange.value = String(state.guidance);
        if (els.sampler)
            els.sampler.value = state.sampler;
        if (els.scheduler)
            els.scheduler.value = state.scheduler;
        if (els.seed)
            els.seed.value = String(state.seed);
        var variationSeed = document.getElementById('gen-variation-seed');
        var variationStrength = document.getElementById('gen-variation-strength');
        var variationRange = document.getElementById('gen-variation-strength-range');
        var creativity = document.getElementById('gen-creativity');
        var creativityRange = document.getElementById('gen-creativity-range');
        var batch = document.getElementById('gen-batch');
        var style = document.getElementById('gen-style-preset');
        if (variationSeed) variationSeed.value = String(state.variationSeed);
        if (variationStrength) variationStrength.value = String(state.variationStrength);
        if (variationRange) variationRange.value = String(state.variationStrength);
        if (creativity) creativity.value = String(state.creativity);
        if (creativityRange) creativityRange.value = String(state.creativity);
        if (batch) batch.value = String(state.batchCount);
        if (els.toolbarBatchInput) els.toolbarBatchInput.value = String(state.batchCount);
        if (style) style.value = state.stylePreset;
        syncDimensionInputs();
        syncAspectDropdown();
        updateAspectPreview();
        updateTokenCount();
        updateStylePreview();
        closeModelDropdown();
    }

    // ── Public API ──
    function init() {
        if (initialized)
            return;
        initialized = true;
        buildUI();
        bindEvents();
        loadModels();
        loadLoras();
        loadAdvancedOptions();
        restoreGallery();
        loadPromptHistory();
        connectWS();
        updateAspectPreview();
        bindBulkActions();
        bindGalleryUpload();
        bindGallerySettings();
        bindPagination();
        bindGlobalPhase2();
        // Render lucide icons
        if (typeof lucide !== 'undefined') {
            lucide.createIcons({ nameAttr: 'data-lucide' });
        }
    }
    return {
        state: state,
        init: init,
        generate: generate,
        getParams: getParams,
        applyParams: applyParams,
        displayResult: function (src, isVideo) {
            if (!initialized)
                init();
            if (isVideo)
                displayVideo(src);
            else
                displayImage(src);
        }
    };
})();
//# sourceMappingURL=generate.js.map
