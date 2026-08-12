"use strict";
/**
 * Simple Mode — SerenityFlow
 * Streamlined single-screen UI for average users.
 * Uses WorkflowBuilder for generation (same backend as Generate tab).
 */
var SimpleMode = (function () {
    'use strict';
    var initialized = false;
    var state = {
        model: null,
        prompt: '',
        negPrompt: '',
        width: 1024,
        height: 1024,
        steps: 20,
        cfg: 7.0,
        guidance: 3.5,
        scheduler: 'euler',
        seed: -1,
        generating: false,
        currentImage: null,
        currentIsVideo: false,
        recent: [],
        arch: 'sd15',
        frames: 97,
        fps: 24,
        activeStyle: 'none',
        quality: 'balanced',
        duration: 'medium',
        capabilities: null
    };
    var els = {
        model: null,
        prompt: null,
        negPrompt: null,
        advToggle: null,
        advBody: null,
        aspectGrid: null,
        qualityRow: null,
        durationSection: null,
        durationRow: null,
        genBtn: null,
        recentGrid: null,
        empty: null,
        previewImg: null,
        previewVideo: null,
        actionBar: null,
        download: null,
        variations: null,
        toAdvanced: null,
        clearPreview: null,
        progress: null,
        progressBar: null,
        errorBanner: null,
        enhanceBtn: null,
        enhanceResult: null,
        styleScroll: null,
        archBadge: null,
        showIntroLink: null
    };
    // ── Presets ──
    var imageStylePresets = [
        { id: 'none', label: 'None', suffix: '', bg: 'var(--shell-bg-panel)' },
        { id: 'photo', label: 'Photo', suffix: ', photorealistic, 8k, sharp focus, natural lighting', bg: 'linear-gradient(135deg, #1a1a2e, #2d4a6e)' },
        { id: 'anime', label: 'Anime', suffix: ', anime style, vibrant colors, cel shaded, detailed', bg: 'linear-gradient(135deg, #ff6b9d, #c44569)' },
        { id: 'oil', label: 'Oil Paint', suffix: ', oil painting, impressionist, textured brushstrokes, museum quality', bg: 'linear-gradient(135deg, #8b4513, #d2691e)' },
        { id: '3d', label: '3D Render', suffix: ', 3D render, octane render, subsurface scattering, studio lighting', bg: 'linear-gradient(135deg, #0f3460, #533483)' },
        { id: 'cinematic', label: 'Cinematic', suffix: ', cinematic, anamorphic lens, film grain, dramatic lighting, color graded', bg: 'linear-gradient(135deg, #0d0d0d, #1a0a00)' },
        { id: 'watercolor', label: 'Watercolor', suffix: ', watercolor painting, soft edges, paper texture, flowing colors', bg: 'linear-gradient(135deg, #89c4e1, #c3e8f7)' },
        { id: 'pixel', label: 'Pixel Art', suffix: ', pixel art, 16-bit, retro game style, limited palette', bg: 'linear-gradient(135deg, #2d1b69, #11998e)' },
        { id: 'sketch', label: 'Sketch', suffix: ', pencil sketch, crosshatching, graphite, detailed linework', bg: 'linear-gradient(135deg, #636363, #a2a2a2)' },
        { id: 'fantasy', label: 'Fantasy', suffix: ', fantasy art, magical, ethereal lighting, detailed environment, epic', bg: 'linear-gradient(135deg, #6a3093, #a044ff)' },
        { id: 'neon', label: 'Neon Noir', suffix: ', neon noir, cyberpunk, rain-slicked streets, neon lights, dark atmosphere', bg: 'linear-gradient(135deg, #0f0c29, #302b63)' },
        { id: 'minimal', label: 'Minimal', suffix: ', minimalist, clean lines, negative space, simple composition, elegant', bg: 'linear-gradient(135deg, #e8e8e8, #c0c0c0)' }
    ];
    var videoStylePresets = [
        { id: 'none', label: 'None', suffix: '', bg: 'var(--shell-bg-panel)' },
        { id: 'cinematic', label: 'Cinematic', suffix: ', cinematic movement, smooth motion, film quality', bg: 'linear-gradient(135deg, #0d0d0d, #1a0a00)' },
        { id: 'timelapse', label: 'Timelapse', suffix: ', time-lapse, smooth transition, flowing movement', bg: 'linear-gradient(135deg, #1a3a5c, #4a8db7)' },
        { id: 'dynamic', label: 'Dynamic', suffix: ', dynamic motion, energy, fast movement, action', bg: 'linear-gradient(135deg, #c0392b, #e74c3c)' },
        { id: 'slow', label: 'Slow-Mo', suffix: ', slow motion, graceful, fluid movement, detail', bg: 'linear-gradient(135deg, #2c3e50, #3498db)' }
    ];
    var qualityPresets = {
        draft: { steps: 8 },
        balanced: { steps: 20 },
        quality: { steps: 40 }
    };
    var durationPresets = {
        short: { frames: 49, fps: 24 },
        medium: { frames: 97, fps: 24 },
        long: { frames: 129, fps: 24 }
    };
    var videoAspects = [
        { label: '1:1', w: 768, h: 768, vw: 16, vh: 16 },
        { label: '4:3', w: 768, h: 576, vw: 18, vh: 14 },
        { label: '16:9', w: 768, h: 432, vw: 20, vh: 11 },
        { label: '3:2', w: 768, h: 512, vw: 18, vh: 12 },
        { label: '9:16', w: 432, h: 768, vw: 11, vh: 20 }
    ];
    function isVideoModel() {
        return ModelUtils.isVideoModel(state.model);
    }
    function getActiveAspects() {
        if (state.arch === 'ltxv')
            return [{ label: '1920×1088', w: 1920, h: 1088, vw: 30, vh: 17 }];
        if (state.arch === 'wan')
            return [{ label: '832×480', w: 832, h: 480, vw: 26, vh: 15 }];
        if (state.arch === 'bernini')
            return [{ label: '848×480', w: 848, h: 480, vw: 53, vh: 30 }];
        if (isVideoModel())
            return videoAspects;
        return ModelUtils.aspectsForArch(state.capabilities, state.arch);
    }
    function getActiveStylePresets() {
        return isVideoModel() ? videoStylePresets : imageStylePresets;
    }
    // ── Quality Config (arch-aware) ──
    function getQualityConfig(quality) {
        var base = qualityPresets[quality] || qualityPresets.balanced;
        var cfg, scheduler;
        var isVideo = state.arch === 'ltxv' || state.arch === 'wan' || state.arch === 'bernini';
        if (state.arch === 'flux') {
            cfg = 1.0;
            scheduler = 'euler';
        }
        else if (state.arch === 'wan') {
            return { steps: 50, cfg: 5.0, scheduler: 'uni_pc' };
        }
        else if (state.arch === 'bernini') {
            return { steps: 40, cfg: 4.0, scheduler: 'uni_pc' };
        }
        else if (isVideo) {
            cfg = 3.0;
            scheduler = 'euler';
        }
        else if (quality === 'draft') {
            cfg = 5.0;
            scheduler = 'euler';
        }
        else if (quality === 'quality') {
            cfg = 7.5;
            scheduler = 'dpmpp_2m';
        }
        else {
            cfg = 7.0;
            scheduler = 'euler';
        }
        return { steps: state.arch === 'ltxv' ? 15 : base.steps, cfg: cfg, scheduler: scheduler };
    }
    // ── Build DOM ──
    function buildUI() {
        var container = document.getElementById('simple-mode-container');
        if (!container)
            return;
        container.innerHTML = '';
        var layout = document.createElement('div');
        layout.className = 'simple-layout';
        var left = document.createElement('div');
        left.className = 'simple-left';
        left.innerHTML = buildLeftHTML();
        layout.appendChild(left);
        var center = document.createElement('div');
        center.className = 'simple-center';
        center.innerHTML = buildCenterHTML();
        layout.appendChild(center);
        container.appendChild(layout);
        // Onboarding overlay (appended to container, not layout)
        var onboarding = document.createElement('div');
        onboarding.id = 'simple-onboarding';
        onboarding.className = 'simple-onboarding-backdrop';
        onboarding.style.display = 'none';
        onboarding.innerHTML =
            '<div class="simple-onboarding-card">' +
                '<div class="onboarding-title">Welcome to SerenityFlow</div>' +
                '<div class="onboarding-subtitle">Create images and videos with AI in just a few clicks.</div>' +
                '<div class="onboarding-steps">' +
                '<div class="onboarding-step"><span class="onboarding-num">1</span> Pick a model from the sidebar</div>' +
                '<div class="onboarding-step"><span class="onboarding-num">2</span> Describe what you want to create</div>' +
                '<div class="onboarding-step"><span class="onboarding-num">3</span> Choose a style and hit Create</div>' +
                '</div>' +
                '<div class="onboarding-note">You can switch to Advanced mode anytime for full control.</div>' +
                '<button id="onboarding-start-btn" class="onboarding-start-btn">Get Started</button>' +
                '</div>';
        container.appendChild(onboarding);
        cacheElements();
    }
    function buildLeftHTML() {
        return '' +
            // Model
            '<div class="simple-section">' +
            '<label class="simple-label">Model</label>' +
            '<select id="simple-model" class="simple-select"><option disabled selected>Loading models...</option></select>' +
            '<div id="simple-arch-badge" class="simple-arch-badge" data-arch="sd15">SD1.5 \u00b7 Image</div>' +
            '</div>' +
            // Prompt
            '<div class="simple-section">' +
            '<label class="simple-label">Prompt</label>' +
            '<textarea id="simple-prompt" class="simple-prompt" rows="5" placeholder="Describe what you want to create..."></textarea>' +
            '<button id="simple-enhance-btn" class="simple-enhance-btn">+ Enhance prompt</button>' +
            '<div id="simple-enhance-result" class="simple-enhance-result" style="display:none"></div>' +
            '<div id="simple-adv-toggle" class="simple-disclosure">' +
            '<span class="simple-disclosure-arrow">&#9654;</span> Advanced prompt' +
            '</div>' +
            '<div id="simple-adv-body" class="simple-disclosure-body">' +
            '<textarea id="simple-neg-prompt" class="simple-neg-prompt" rows="2" placeholder="What to avoid..."></textarea>' +
            '</div>' +
            '</div>' +
            // Style presets (cards)
            '<div class="simple-section">' +
            '<label class="simple-label">Style</label>' +
            '<div id="simple-style-scroll" class="simple-style-scroll"></div>' +
            '<div id="simple-prompt-preview" class="simple-prompt-preview" style="display:none">' +
            '<span class="simple-prompt-preview-label">Full prompt:</span>' +
            '<span id="simple-prompt-preview-text" class="simple-prompt-preview-text"></span>' +
            '</div>' +
            '</div>' +
            // Quick Settings
            '<div class="simple-section simple-quick">' +
            '<span class="simple-quick-label">Aspect Ratio</span>' +
            '<div id="simple-aspects" class="simple-aspect-grid"></div>' +
            '<span class="simple-quick-label">Quality</span>' +
            '<div id="simple-quality" class="simple-quality-row">' +
            '<button class="simple-quality-btn" data-quality="draft">Draft</button>' +
            '<button class="simple-quality-btn active" data-quality="balanced">Balanced</button>' +
            '<button class="simple-quality-btn" data-quality="quality">Quality</button>' +
            '</div>' +
            '<div id="simple-duration-section" class="simple-duration-section">' +
            '<span class="simple-quick-label">Duration</span>' +
            '<div id="simple-duration" class="simple-duration-row">' +
            '<button class="simple-duration-btn" data-duration="short">Short (2s)</button>' +
            '<button class="simple-duration-btn active" data-duration="medium">Medium (4s)</button>' +
            '<button class="simple-duration-btn" data-duration="long">Long (5s)</button>' +
            '</div>' +
            '</div>' +
            '</div>' +
            // Advanced Options (CFG, Sampler, Negative Prompt)
            '<div class="simple-section">' +
            '<div id="simple-adv-options-toggle" class="simple-disclosure">' +
            '<span class="simple-disclosure-arrow">&#9654;</span> Advanced Options' +
            '</div>' +
            '<div id="simple-adv-options-body" class="simple-disclosure-body">' +
            '<div class="simple-adv-row">' +
            '<span class="simple-quick-label">CFG</span>' +
            '<input type="range" id="simple-cfg-slider" class="simple-slider" min="1" max="20" step="0.5" value="3.0">' +
            '<span id="simple-cfg-value" class="simple-slider-value">3.0</span>' +
            '</div>' +
            '<div class="simple-adv-row">' +
            '<span class="simple-quick-label">Steps</span>' +
            '<input type="range" id="simple-steps-slider" class="simple-slider" min="1" max="50" step="1" value="20">' +
            '<span id="simple-steps-value" class="simple-slider-value">20</span>' +
            '</div>' +
            '<div class="simple-adv-row">' +
            '<span class="simple-quick-label">Sampler</span>' +
            '<select id="simple-sampler" class="simple-select-sm">' +
            '<option value="euler" selected>Euler</option>' +
            '<option value="euler_ancestral">Euler A</option>' +
            '<option value="dpmpp_2m">DPM++ 2M</option>' +
            '<option value="dpmpp_sde">DPM++ SDE</option>' +
            '<option value="ddim">DDIM</option>' +
            '<option value="uni_pc">UniPC</option>' +
            '</select>' +
            '</div>' +
            '<div class="simple-adv-row">' +
            '<span class="simple-quick-label">Negative Prompt</span>' +
            '<textarea id="simple-neg-prompt-adv" class="simple-neg-prompt" rows="2" placeholder="What to avoid (blur, low quality, etc.)"></textarea>' +
            '</div>' +
            '<div class="simple-adv-row">' +
            '<span class="simple-quick-label">Seed</span>' +
            '<input type="number" id="simple-seed" class="simple-input-sm" value="-1" min="-1">' +
            '</div>' +
            '<div class="simple-adv-row">' +
            '<span class="simple-quick-label">Upscale</span>' +
            '<select id="simple-upscale" class="simple-select-sm">' +
            '<option value="none" selected>None</option>' +
            '<option value="2x">2x</option>' +
            '<option value="4x">4x</option>' +
            '</select>' +
            '</div>' +
            '</div>' +
            '</div>' +
            // Generate button
            '<div class="simple-section">' +
            '<button id="simple-gen-btn" class="simple-gen-btn">\u2726 Create Image</button>' +
            '</div>' +
            // Recent
            '<div class="simple-recent-section">' +
            '<div class="simple-recent-label">Recent</div>' +
            '<div id="simple-recent-grid" class="simple-recent-grid"></div>' +
            '</div>' +
            // Show intro link
            '<div class="simple-show-intro">' +
            '<a id="simple-show-intro-link">Show intro</a>' +
            '</div>';
    }
    function buildCenterHTML() {
        return '' +
            '<div id="simple-empty" class="simple-empty">' +
            '<div class="simple-empty-icon">\u2726</div>' +
            '<div class="simple-empty-title">Ready to create</div>' +
            '<div class="simple-empty-subtitle">Pick a model, describe your idea, and hit Create</div>' +
            '<div class="simple-empty-examples">' +
            '<span class="simple-example-prompt">A fox in a rainy forest at night</span>' +
            '<span class="simple-example-prompt">Portrait of an astronaut, oil painting style</span>' +
            '<span class="simple-example-prompt">Neon city street at midnight, cinematic</span>' +
            '</div>' +
            '</div>' +
            '<img id="simple-preview-img" class="simple-preview-img" style="display:none" alt="Generated">' +
            '<video id="simple-preview-video" class="simple-preview-video" style="display:none" autoplay loop muted playsinline controls></video>' +
            '<div id="simple-action-bar" class="simple-action-bar" style="display:none">' +
            '<button class="simple-action-btn" id="simple-download" title="Download">' +
            '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>' +
            '</button>' +
            '<button class="simple-action-btn" id="simple-variations" title="Variations (new seed)">' +
            '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 18h1.4c1.3 0 2.5-.6 3.3-1.7l6.1-8.6c.7-1.1 2-1.7 3.3-1.7H22"/><path d="m18 2 4 4-4 4"/><path d="M2 6h1.9c1.5 0 2.9.9 3.6 2.2"/><path d="M22 18h-5.9c-1.3 0-2.6-.7-3.3-1.8l-.5-.8"/><path d="m18 14 4 4-4 4"/></svg>' +
            '</button>' +
            '<button class="simple-action-btn" id="simple-to-advanced" title="Send to Advanced">' +
            '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M9 21V9"/></svg>' +
            '</button>' +
            '<button class="simple-action-btn" id="simple-clear-preview" title="Clear">' +
            '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/></svg>' +
            '</button>' +
            '</div>' +
            '<div id="simple-progress" class="simple-progress"><div id="simple-progress-bar" class="simple-progress-bar"></div></div>' +
            '<div id="simple-error-banner" class="simple-error-banner"></div>';
    }
    function cacheElements() {
        els.model = document.getElementById('simple-model');
        els.prompt = document.getElementById('simple-prompt');
        els.negPrompt = document.getElementById('simple-neg-prompt');
        els.advToggle = document.getElementById('simple-adv-toggle');
        els.advBody = document.getElementById('simple-adv-body');
        els.aspectGrid = document.getElementById('simple-aspects');
        els.qualityRow = document.getElementById('simple-quality');
        els.durationSection = document.getElementById('simple-duration-section');
        els.durationRow = document.getElementById('simple-duration');
        els.genBtn = document.getElementById('simple-gen-btn');
        els.recentGrid = document.getElementById('simple-recent-grid');
        els.empty = document.getElementById('simple-empty');
        els.previewImg = document.getElementById('simple-preview-img');
        els.previewVideo = document.getElementById('simple-preview-video');
        els.actionBar = document.getElementById('simple-action-bar');
        els.download = document.getElementById('simple-download');
        els.variations = document.getElementById('simple-variations');
        els.toAdvanced = document.getElementById('simple-to-advanced');
        els.clearPreview = document.getElementById('simple-clear-preview');
        els.progress = document.getElementById('simple-progress');
        els.progressBar = document.getElementById('simple-progress-bar');
        els.errorBanner = document.getElementById('simple-error-banner');
        els.enhanceBtn = document.getElementById('simple-enhance-btn');
        els.enhanceResult = document.getElementById('simple-enhance-result');
        els.styleScroll = document.getElementById('simple-style-scroll');
        els.archBadge = document.getElementById('simple-arch-badge');
        els.showIntroLink = document.getElementById('simple-show-intro-link');
    }
    // ── Style Presets (visual cards) ──
    function renderStylePresets(isVideo) {
        var presets = isVideo ? videoStylePresets : imageStylePresets;
        var scroll = document.getElementById('simple-style-scroll');
        if (!scroll)
            return;
        scroll.innerHTML = '';
        // Restore saved preset or default to none
        var savedPreset = localStorage.getItem('sf-simple-preset') || 'none';
        if (!presets.find(function (p) { return p.id === savedPreset; }))
            savedPreset = 'none';
        state.activeStyle = savedPreset;
        presets.forEach(function (p) {
            var card = document.createElement('div');
            card.className = 'simple-style-card' + (p.id === savedPreset ? ' active' : '');
            card.dataset.style = p.id;
            card.innerHTML =
                '<div class="simple-style-swatch" style="background:' + p.bg + '"></div>' +
                    '<span class="simple-style-card-label">' + p.label + '</span>';
            scroll.appendChild(card);
        });
        scroll.onclick = function (e) {
            var card = e.target.closest('.simple-style-card');
            if (!card)
                return;
            state.activeStyle = card.dataset.style || 'none';
            localStorage.setItem('sf-simple-preset', state.activeStyle);
            scroll.querySelectorAll('.simple-style-card').forEach(function (c) {
                c.classList.toggle('active', c.dataset.style === state.activeStyle);
            });
            updatePromptPreview();
        };
    }
    // ── Prompt Preview ──
    function updatePromptPreview() {
        var previewEl = document.getElementById('simple-prompt-preview');
        var textEl = document.getElementById('simple-prompt-preview-text');
        if (!previewEl || !textEl)
            return;
        if (state.activeStyle === 'none' || !state.prompt.trim()) {
            previewEl.style.display = 'none';
            return;
        }
        previewEl.style.display = 'block';
        textEl.textContent = getEffectivePrompt();
    }
    // ── Aspect Ratio Buttons ──
    function buildAspectButtons() {
        var aspects = getActiveAspects();
        if (!els.aspectGrid)
            return;
        els.aspectGrid.innerHTML = '';
        aspects.forEach(function (a) {
            var btn = document.createElement('button');
            var isActive = a.w === state.width && a.h === state.height;
            btn.className = 'simple-aspect-btn' + (isActive ? ' active' : '');
            var maxDim = 20;
            var sw = Math.round(a.vw / maxDim * 14);
            var sh = Math.round(a.vh / maxDim * 14);
            btn.innerHTML =
                '<svg width="' + (sw + 4) + '" height="' + (sh + 4) + '" viewBox="0 0 ' + (sw + 4) + ' ' + (sh + 4) + '" fill="none" stroke-width="1.5">' +
                    '<rect x="1" y="1" width="' + sw + '" height="' + sh + '" rx="2"/>' +
                    '</svg>' +
                    '<span class="simple-aspect-btn-label">' + a.label + '</span>';
            btn.addEventListener('click', function () {
                state.width = a.w;
                state.height = a.h;
                if (els.aspectGrid) {
                    els.aspectGrid.querySelectorAll('.simple-aspect-btn').forEach(function (b) {
                        b.classList.remove('active');
                    });
                }
                btn.classList.add('active');
            });
            els.aspectGrid.appendChild(btn);
        });
    }
    // ── Smart Defaults ──
    function applySmartDefaults(arch) {
        var isVideo = arch === 'ltxv' || arch === 'wan' || arch === 'bernini';
        // Set quality
        setQualityPreset('balanced');
        // Set aspect (16:9 for video, 1:1 for image)
        var aspects = getActiveAspects();
        var defaultIdx = arch === 'ltxv' ? 0 : (isVideo ? 2 : 0);
        if (aspects[defaultIdx]) {
            state.width = aspects[defaultIdx].w;
            state.height = aspects[defaultIdx].h;
        }
        // Duration
        if (arch === 'bernini') {
            state.frames = 81;
            state.fps = 16;
        }
        else if (isVideo) {
            state.frames = 121;
            state.fps = 24;
        }
        // Render style presets
        renderStylePresets(isVideo);
    }
    function setQualityPreset(q) {
        state.quality = q;
        var config = getQualityConfig(q);
        state.steps = config.steps;
        state.cfg = config.cfg;
        state.scheduler = config.scheduler;
        var row = document.getElementById('simple-quality');
        if (row) {
            row.querySelectorAll('.simple-quality-btn').forEach(function (b) {
                b.classList.toggle('active', b.dataset.quality === q);
            });
        }
    }
    function setDurationPreset(d) {
        state.duration = d;
        var dp = durationPresets[d];
        state.frames = state.arch === 'bernini' ? 81 : (state.arch === 'ltxv' || state.arch === 'wan' ? 121 : Math.min(dp.frames, 129));
        state.fps = state.arch === 'bernini' ? 16 : (state.arch === 'ltxv' || state.arch === 'wan' ? 24 : dp.fps);
        var row = document.getElementById('simple-duration');
        if (row) {
            row.querySelectorAll('.simple-duration-btn').forEach(function (b) {
                b.classList.toggle('active', b.dataset.duration === d);
            });
        }
    }
    // ── Prompt Enhancer (local, no API) ──
    function enhancePrompt() {
        var original = state.prompt.trim();
        if (!original)
            return;
        var enhanceBtn = document.getElementById('simple-enhance-btn');
        if (enhanceBtn) {
            enhanceBtn.textContent = 'Enhancing...';
            enhanceBtn.disabled = true;
        }
        // Try backend endpoint first, fall back to local
        fetch('/enhance_prompt', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: original, arch: state.arch })
        })
            .then(function (resp) {
            if (!resp.ok)
                throw new Error('HTTP ' + resp.status);
            return resp.json();
        })
            .then(function (data) {
            if (data.enhanced) {
                showEnhancedResult(original, data.enhanced, data.source || 'backend');
            }
        })
            .catch(function () {
            // Fallback to local enhancement
            var enhanced = localEnhance(original, state.arch);
            showEnhancedResult(original, enhanced, 'local');
        })
            .finally(function () {
            if (enhanceBtn) {
                enhanceBtn.textContent = '+ Enhance prompt';
                enhanceBtn.disabled = false;
            }
        });
    }
    function localEnhance(prompt, arch) {
        var details = [];
        var lower = prompt.toLowerCase();
        // Add lighting if not mentioned
        if (lower.indexOf('light') === -1 && lower.indexOf('lit') === -1) {
            var lightOptions = ['soft natural lighting', 'golden hour lighting', 'dramatic side lighting', 'studio lighting with rim light', 'ambient diffused lighting'];
            details.push(lightOptions[Math.floor(Math.random() * lightOptions.length)]);
        }
        // Add composition if not mentioned
        if (lower.indexOf('composit') === -1 && lower.indexOf('angle') === -1 && lower.indexOf('shot') === -1) {
            var compOptions = ['carefully composed', 'rule of thirds composition', 'centered symmetrical composition', 'dynamic diagonal composition'];
            details.push(compOptions[Math.floor(Math.random() * compOptions.length)]);
        }
        // Add detail level
        if (lower.indexOf('detail') === -1 && lower.indexOf('quality') === -1) {
            details.push('highly detailed');
        }
        // Add mood/atmosphere if not mentioned
        if (lower.indexOf('mood') === -1 && lower.indexOf('atmosphere') === -1 && lower.indexOf('vibe') === -1) {
            var moodOptions = ['atmospheric', 'evocative atmosphere', 'rich atmosphere'];
            details.push(moodOptions[Math.floor(Math.random() * moodOptions.length)]);
        }
        // Add technical quality
        if (lower.indexOf('8k') === -1 && lower.indexOf('4k') === -1 && lower.indexOf('hd') === -1) {
            details.push('high resolution');
        }
        // Arch-specific additions
        if (arch === 'flux' || arch === 'sdxl') {
            details.push('masterful execution');
        }
        return prompt + ', ' + details.join(', ');
    }
    function showEnhancedResult(original, enhanced, source) {
        var container = document.getElementById('simple-enhance-result');
        if (!container)
            return;
        container.innerHTML =
            '<div class="enhance-original">' + escapeHtml(original) + '</div>' +
                '<div class="enhance-arrow">\u2193 Enhanced</div>' +
                '<div class="enhance-source">' + (source === 'llm' ? 'AI enhanced' : 'Auto-enhanced') + '</div>' +
                '<div class="enhance-new">' + escapeHtml(enhanced) + '</div>' +
                '<div class="enhance-actions">' +
                '<button id="enhance-accept" class="enhance-accept-btn">Use this \u25b6</button>' +
                '<button id="enhance-dismiss" class="enhance-dismiss-btn">Keep original \u2715</button>' +
                '</div>';
        container.style.display = 'block';
        document.getElementById('enhance-accept').onclick = function () {
            state.prompt = enhanced;
            if (els.prompt) {
                els.prompt.value = enhanced;
                els.prompt.style.height = 'auto';
                els.prompt.style.height = Math.max(100, els.prompt.scrollHeight) + 'px';
            }
            container.style.display = 'none';
            updatePromptPreview();
        };
        document.getElementById('enhance-dismiss').onclick = function () {
            container.style.display = 'none';
        };
    }
    function escapeHtml(str) {
        var div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
    // ── Onboarding ──
    function showOnboarding() {
        var overlay = document.getElementById('simple-onboarding');
        if (overlay) {
            overlay.style.display = 'flex';
            overlay.style.opacity = '0';
            requestAnimationFrame(function () {
                overlay.style.opacity = '1';
            });
        }
    }
    function dismissOnboarding() {
        var overlay = document.getElementById('simple-onboarding');
        if (overlay) {
            overlay.style.opacity = '0';
            setTimeout(function () { overlay.style.display = 'none'; }, 300);
        }
        localStorage.setItem('sf-has-visited', '1');
    }
    // ── Event Binding ──
    function bindEvents() {
        // Prompt auto-grow + preview update
        if (els.prompt) {
            els.prompt.addEventListener('input', function () {
                state.prompt = this.value;
                this.style.height = 'auto';
                this.style.height = Math.max(100, this.scrollHeight) + 'px';
                updatePromptPreview();
            });
        }
        if (els.negPrompt) {
            els.negPrompt.addEventListener('input', function () {
                state.negPrompt = this.value;
            });
        }
        // Advanced prompt disclosure
        if (els.advToggle) {
            els.advToggle.addEventListener('click', function () {
                this.classList.toggle('open');
                if (els.advBody)
                    els.advBody.classList.toggle('open');
            });
        }
        // Advanced Options disclosure + controls
        var advOptionsToggle = document.getElementById('simple-adv-options-toggle');
        var advOptionsBody = document.getElementById('simple-adv-options-body');
        if (advOptionsToggle && advOptionsBody) {
            advOptionsToggle.addEventListener('click', function () {
                this.classList.toggle('open');
                advOptionsBody.classList.toggle('open');
            });
        }
        var cfgSlider = document.getElementById('simple-cfg-slider');
        var cfgValue = document.getElementById('simple-cfg-value');
        if (cfgSlider) {
            cfgSlider.addEventListener('input', function () {
                state.cfg = parseFloat(this.value);
                if (cfgValue)
                    cfgValue.textContent = this.value;
            });
        }
        var stepsSlider = document.getElementById('simple-steps-slider');
        var stepsValue = document.getElementById('simple-steps-value');
        if (stepsSlider) {
            stepsSlider.addEventListener('input', function () {
                state.steps = parseInt(this.value);
                if (stepsValue)
                    stepsValue.textContent = this.value;
            });
        }
        var samplerSelect = document.getElementById('simple-sampler');
        if (samplerSelect) {
            samplerSelect.addEventListener('change', function () {
                state.scheduler = this.value;
            });
        }
        var negPromptAdv = document.getElementById('simple-neg-prompt-adv');
        if (negPromptAdv) {
            negPromptAdv.addEventListener('input', function () {
                state.negPrompt = this.value;
            });
        }
        var seedInput = document.getElementById('simple-seed');
        if (seedInput) {
            seedInput.addEventListener('change', function () {
                state.seed = parseInt(this.value) || -1;
            });
        }
        // Quality presets (with arch-aware config)
        if (els.qualityRow) {
            els.qualityRow.addEventListener('click', function (e) {
                var btn = e.target.closest('.simple-quality-btn');
                if (!btn)
                    return;
                setQualityPreset(btn.dataset.quality || 'balanced');
            });
        }
        // Duration presets
        if (els.durationRow) {
            els.durationRow.addEventListener('click', function (e) {
                var btn = e.target.closest('.simple-duration-btn');
                if (!btn)
                    return;
                setDurationPreset(btn.dataset.duration || 'medium');
            });
        }
        // Model change
        if (els.model) {
            els.model.addEventListener('change', function () {
                state.model = this.value;
                var arch = ModelUtils.archForModel(this.value);
                updateUIForArch(arch);
                applySmartDefaults(arch);
                // Update topbar model badge
                var badge = document.querySelector('.model-badge');
                if (badge)
                    badge.textContent = this.value;
                ModelUtils.warmModel(this.value);
            });
        }
        // Generate
        if (els.genBtn) {
            els.genBtn.addEventListener('click', function () {
                generate();
            });
        }
        // Enhance prompt
        if (els.enhanceBtn) {
            els.enhanceBtn.addEventListener('click', function () {
                enhancePrompt();
            });
        }
        // Show intro link
        if (els.showIntroLink) {
            els.showIntroLink.addEventListener('click', function (e) {
                e.preventDefault();
                showOnboarding();
            });
        }
        // Onboarding dismiss
        var onboardingStartBtn = document.getElementById('onboarding-start-btn');
        if (onboardingStartBtn) {
            onboardingStartBtn.addEventListener('click', dismissOnboarding);
        }
        var onboardingBackdrop = document.getElementById('simple-onboarding');
        if (onboardingBackdrop) {
            onboardingBackdrop.addEventListener('click', function (e) {
                if (e.target === onboardingBackdrop)
                    dismissOnboarding();
            });
        }
        // Action bar
        if (els.download) {
            els.download.addEventListener('click', function () {
                if (!state.currentImage)
                    return;
                var a = document.createElement('a');
                a.href = state.currentImage;
                var ext = state.currentIsVideo ? '.mp4' : '.png';
                a.download = 'serenityflow_' + Date.now() + ext;
                a.click();
            });
        }
        if (els.variations) {
            els.variations.addEventListener('click', function () {
                if (state.generating || !state.prompt.trim())
                    return;
                state.seed = -1; // force new random seed
                generate();
            });
        }
        if (els.toAdvanced) {
            els.toAdvanced.addEventListener('click', function () {
                if (!state.currentImage)
                    return;
                // Store data for Canvas tab
                localStorage.setItem('sf-send-to-canvas', JSON.stringify({
                    src: state.currentImage,
                    isVideo: state.currentIsVideo,
                    prompt: state.prompt,
                    model: state.model
                }));
                if (typeof setMode === 'function') {
                    setMode('advanced');
                    switchTab('canvas');
                }
            });
        }
        if (els.clearPreview) {
            els.clearPreview.addEventListener('click', function () {
                clearPreview();
            });
        }
        // Example prompts
        document.querySelectorAll('.simple-example-prompt').forEach(function (el) {
            el.addEventListener('click', function () {
                state.prompt = this.textContent || '';
                if (els.prompt) {
                    els.prompt.value = this.textContent || '';
                    els.prompt.style.height = 'auto';
                    els.prompt.style.height = Math.max(100, els.prompt.scrollHeight) + 'px';
                }
                updatePromptPreview();
            });
        });
    }
    // ── Model Loading ──
    function loadModels() {
        Promise.all([ModelUtils.fetchAllModels(), ModelUtils.loadCapabilities()])
            .then(function (loaded) {
            // SCAIL-2 needs four uploaded media inputs; its complete surface is
            // the Generate tab, so do not expose an unusable text-only card in
            // Simple mode.
            var models = loaded[0].filter(function (model) {
                return ModelUtils.archForModel(model.name) !== 'scail2';
            });
            state.capabilities = loaded[1];
            if (!models.length)
                throw new Error('empty');
            if (!els.model)
                return;
            els.model.innerHTML = '';
            models.forEach(function (m) {
                var opt = document.createElement('option');
                opt.value = m.name;
                opt.textContent = m.name;
                els.model.appendChild(opt);
            });
            state.model = models[0].name;
            var arch = ModelUtils.archForModel(models[0].name);
            updateUIForArch(arch);
            applySmartDefaults(arch);
            // Update topbar badge
            var badge = document.querySelector('.model-badge');
            if (badge)
                badge.textContent = models[0].name;
            ModelUtils.warmModel(models[0].name);
        })
            .catch(function () {
            if (els.model)
                els.model.innerHTML = '<option disabled selected>No models found</option>';
        });
    }
    // ── Arch-aware UI ──
    function updateUIForArch(arch) {
        state.arch = arch;
        var isFlux = arch === 'flux';
        var isVideo = arch === 'ltxv' || arch === 'wan' || arch === 'bernini';
        // Auto-set CFG/Guidance based on arch
        if (isFlux) {
            state.cfg = 1.0;
            state.guidance = 3.5;
        }
        else if (isVideo) {
            state.cfg = arch === 'wan' ? 5.0 : (arch === 'bernini' ? 4.0 : 3.0);
        }
        else {
            state.cfg = 7.0;
        }
        // Video duration section
        if (els.durationSection)
            els.durationSection.style.display = 'none';
        // Button label
        if (els.genBtn)
            els.genBtn.textContent = isVideo ? '\u2726 Create Video' : '\u2726 Create Image';
        // Rebuild aspect buttons
        buildAspectButtons();
        // Select first aspect
        var aspects = getActiveAspects();
        if (aspects.length > 0) {
            state.width = aspects[0].w;
            state.height = aspects[0].h;
        }
        // Apply current duration preset for video
        if (isVideo) {
            var dp = durationPresets[state.duration];
            state.frames = arch === 'bernini' ? 81 : 121;
            state.fps = arch === 'bernini' ? 16 : 24;
        }
        // Update arch badge
        var archNames = {
            flux: 'FLUX \u00b7 Image', sdxl: 'SDXL \u00b7 Image', sd3: 'SD3 \u00b7 Image',
            sd15: 'SD1.5 \u00b7 Image', ltxv: 'LTX-V \u00b7 Video', wan: 'Wan \u00b7 Video',
            bernini: 'Bernini-R \u00b7 Video',
            klein: 'Klein \u00b7 Image'
        };
        var badge = document.getElementById('simple-arch-badge');
        if (badge) {
            badge.textContent = archNames[arch] || arch;
            badge.dataset.arch = arch;
        }
    }
    // ── Workflow Builder ──
    function getEffectivePrompt() {
        var prompt = state.prompt;
        if (state.activeStyle && state.activeStyle !== 'none') {
            var presets = getActiveStylePresets();
            var style = presets.find(function (s) { return s.id === state.activeStyle; });
            if (style && style.suffix) {
                prompt += style.suffix;
            }
        }
        return prompt;
    }
    function buildWorkflow() {
        // Apply quality config before building
        var config = getQualityConfig(state.quality);
        var upscaleEl = document.getElementById('simple-upscale');
        var upscaleVal = upscaleEl ? upscaleEl.value : 'none';
        return WorkflowBuilder.build({
            model: state.model || '',
            prompt: getEffectivePrompt(),
            negPrompt: state.negPrompt,
            width: state.width,
            height: state.height,
            steps: config.steps,
            cfg: config.cfg,
            guidance: state.guidance,
            scheduler: config.scheduler,
            seed: state.seed,
            frames: state.frames,
            fps: state.fps,
            upscale: upscaleVal
        });
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
        setGenerating(true);
        var workflow = buildWorkflow();
        SerenityAPI.postPrompt(workflow, {
            prompt: getEffectivePrompt(),
            model: state.model
        })
            .catch(function (err) {
            showError('Failed to queue: ' + err.message);
            setGenerating(false);
        });
    }
    // ── State Helpers ──
    function setGenerating(v) {
        state.generating = v;
        if (els.genBtn) {
            els.genBtn.disabled = v;
            if (v) {
                els.genBtn.textContent = 'Creating...';
            }
            else {
                els.genBtn.textContent = isVideoModel() ? '\u2726 Create Video' : '\u2726 Create Image';
            }
            els.genBtn.classList.toggle('generating', v);
        }
        if (v) {
            if (els.progress)
                els.progress.classList.add('active');
            if (els.progressBar)
                els.progressBar.style.width = '100%';
        }
        else {
            if (els.progress)
                els.progress.classList.remove('active');
            if (els.progressBar)
                els.progressBar.style.width = '0%';
        }
    }
    function displayImage(src) {
        state.currentImage = src;
        state.currentIsVideo = false;
        if (els.previewImg) {
            els.previewImg.src = src;
            els.previewImg.style.display = 'block';
        }
        if (els.previewVideo) {
            els.previewVideo.style.display = 'none';
            els.previewVideo.pause();
        }
        if (els.actionBar)
            els.actionBar.style.display = 'flex';
        if (els.empty)
            els.empty.style.display = 'none';
    }
    function displayVideo(src) {
        state.currentImage = src;
        state.currentIsVideo = true;
        if (els.previewVideo) {
            els.previewVideo.src = src;
            els.previewVideo.style.display = 'block';
        }
        if (els.previewImg)
            els.previewImg.style.display = 'none';
        if (els.actionBar)
            els.actionBar.style.display = 'flex';
        if (els.empty)
            els.empty.style.display = 'none';
    }
    function clearPreview() {
        state.currentImage = null;
        state.currentIsVideo = false;
        if (els.previewImg) {
            els.previewImg.style.display = 'none';
            els.previewImg.removeAttribute('src');
        }
        if (els.previewVideo) {
            els.previewVideo.style.display = 'none';
            els.previewVideo.pause();
            els.previewVideo.removeAttribute('src');
        }
        if (els.actionBar)
            els.actionBar.style.display = 'none';
        if (els.empty)
            els.empty.style.display = 'flex';
        if (els.recentGrid) {
            els.recentGrid.querySelectorAll('.simple-recent-thumb').forEach(function (t) {
                t.classList.remove('active');
            });
        }
    }
    function addToRecent(src, isVideo) {
        state.recent.unshift({ src: src, isVideo: !!isVideo });
        if (state.recent.length > 8)
            state.recent.pop();
        renderRecent();
        saveRecent();
    }
    function renderRecent() {
        if (!els.recentGrid)
            return;
        els.recentGrid.innerHTML = '';
        state.recent.forEach(function (item, i) {
            var thumb = document.createElement('div');
            thumb.className = 'simple-recent-thumb' + (i === 0 ? ' active' : '');
            if (item.isVideo) {
                thumb.innerHTML =
                    '<video src="' + item.src + '" muted preload="metadata"></video>' +
                        '<div class="simple-recent-play">\u25b6</div>';
            }
            else {
                thumb.innerHTML = '<img src="' + item.src + '" alt="recent">';
            }
            thumb.addEventListener('click', function () {
                if (item.isVideo) {
                    displayVideo(item.src);
                }
                else {
                    displayImage(item.src);
                }
                if (els.recentGrid) {
                    els.recentGrid.querySelectorAll('.simple-recent-thumb').forEach(function (t) {
                        t.classList.remove('active');
                    });
                }
                thumb.classList.add('active');
            });
            els.recentGrid.appendChild(thumb);
        });
    }
    function saveRecent() {
        try {
            localStorage.setItem('sf-simple-recent', JSON.stringify(state.recent));
        }
        catch (e) { /* quota */ }
    }
    function restoreRecent() {
        try {
            var saved = JSON.parse(localStorage.getItem('sf-simple-recent'));
            if (saved && Array.isArray(saved)) {
                state.recent = saved.slice(0, 8);
                renderRecent();
            }
        }
        catch (e) { /* ignore */ }
    }
    // ── Error Display ──
    function showError(msg) {
        if (!els.errorBanner)
            return;
        els.errorBanner.textContent = msg;
        els.errorBanner.classList.add('visible');
        setTimeout(function () {
            if (els.errorBanner)
                els.errorBanner.classList.remove('visible');
        }, 5000);
    }
    // ── WebSocket ──
    function connectWS() {
        SerenityWS.on('progress', function (data) {
            if (!data || !state.generating)
                return;
            var pct = (data.value / data.max * 100).toFixed(0);
            if (els.progressBar)
                els.progressBar.style.width = pct + '%';
        });
        SerenityWS.on('preview', function (data) {
            if (!data || !data.blob || !state.generating)
                return;
            var url = URL.createObjectURL(data.blob);
            if (els.previewImg) {
                if (els.previewImg._previewUrl)
                    URL.revokeObjectURL(els.previewImg._previewUrl);
                els.previewImg._previewUrl = url;
                els.previewImg.src = url;
                els.previewImg.style.display = 'block';
                if (els.empty)
                    els.empty.style.display = 'none';
                els.previewImg.classList.add('simple-preview-live');
            }
        });
        SerenityWS.on('executed', function (data) {
            if (!data || !data.output || !state.generating)
                return;
            // Clean up live preview state
            if (els.previewImg) {
                els.previewImg.classList.remove('simple-preview-live');
                if (els.previewImg._previewUrl) {
                    URL.revokeObjectURL(els.previewImg._previewUrl);
                    els.previewImg._previewUrl = null;
                }
            }
            var out = data.output.ui || data.output;
            var items = out.images;
            var isVideo = false;
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
            if (!isVideo)
                isVideo = /\.(webp|mp4|gif)$/i.test(file.filename);
            if (isVideo) {
                displayVideo(src);
            }
            else {
                displayImage(src);
            }
            addToRecent(src, isVideo);
            setGenerating(false);
            // Pop animation on preview
            var previewEl = isVideo ? els.previewVideo : els.previewImg;
            if (previewEl) {
                previewEl.classList.add('simple-pop');
                setTimeout(function () { previewEl.classList.remove('simple-pop'); }, 400);
            }
            // Brief "Done" on button
            if (els.genBtn) {
                els.genBtn.textContent = '\u2713 Done';
                setTimeout(function () {
                    if (!state.generating && els.genBtn) {
                        els.genBtn.textContent = isVideoModel() ? '\u2726 Create Video' : '\u2726 Create Image';
                    }
                }, 1500);
            }
        });
        SerenityWS.on('execution_error', function (data) {
            var errMsg = (data && data.exception_message) || 'Generation failed';
            showError(errMsg);
            setGenerating(false);
        });
    }
    // ── Public API ──
    function init() {
        if (initialized)
            return;
        initialized = true;
        buildUI();
        buildAspectButtons();
        bindEvents();
        loadModels();
        restoreRecent();
        connectWS();
        renderStylePresets(false);
        // Show onboarding for first-time visitors
        if (!localStorage.getItem('sf-has-visited')) {
            showOnboarding();
        }
    }
    return {
        state: state,
        init: init,
        generate: generate
    };
})();
